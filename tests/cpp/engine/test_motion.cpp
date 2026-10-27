// The planner's motion on piecewise trajectories and meshless carriers (docs/design/phase3/blender.md §3.7, §4.7.7,
// T5 and T5b; Phase 3 BL-5b): TrajectoryMotion's pose per piece equal to the renderer's body_state at b - 1, b and
// b + 1 (camera_pose, the carriers' mid-window poses), window bounds per piece (at least the sampled speeds across a
// C0 join, only the pieces a window touches), holds across a join, the step join's refusal, and a meshless light
// carrier left out of the planner's body lists while its light rides it, through a host engine run.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/traj/spline_build.h"
#include "engine/engine.h"
#include "engine/motion.h"
#include "render/pose.h"
#include "render/scene_data.h"
#include "scenes.h"

using namespace balla;
using namespace balla::engine;
using namespace balla::render::test;

namespace {

constexpr int64_t kW = int64_t{1} << 23, kT0 = 120 * kW, kT1 = kT0 + 6 * kW;

// The kinematics of piece p's own spline (nodes from its rows alone) at t_ns.
traj::Kinematics<double> piece_eval(const render::Trajectory& t, size_t p, int64_t t_ns) {
  const render::TrajectoryPiece& q = t.pieces[p];
  std::vector<Quat<double>> qs;
  std::vector<Vec3<double>> ps;
  for (int32_t c = q.ctrl_begin; c < q.ctrl_end; ++c) {
    const auto i = static_cast<size_t>(c);
    qs.push_back({t.rot[4 * i], t.rot[4 * i + 1], t.rot[4 * i + 2], t.rot[4 * i + 3]});
    ps.push_back({t.pos[3 * i], t.pos[3 * i + 1], t.pos[3 * i + 2]});
  }
  const auto nodes = traj::build_spline_nodes(qs, ps, {0, 0, 0});
  const auto h = static_cast<int32_t>(q.h_ns);
  const traj::SplineView<double> v{nodes.data(), static_cast<int32_t>(nodes.size()) - t.order + 1, h, 1e9 / h};
  traj::SegmentTime st{};
  REQUIRE(traj::locate(t_ns, q.t0_ns, h, v.num_segments, &st));
  return t.order == 4 ? traj::evaluate<4>(v, st) : traj::evaluate<6>(v, st);
}

// Moves piece p's controls rigidly so that its pose at its begin equals piece p - 1's there (to rounding): a C0 join.
// Left-multiplying every control rotation by one rotation turns the whole piece (its increments are unchanged), and
// adding one offset to every control position shifts it (the basis sums to 1).
void join_c0(render::Trajectory& t, size_t p) {
  const int64_t b = t.pieces[p].begin_ns;
  const traj::Kinematics<double> left = piece_eval(t, p - 1, b), right = piece_eval(t, p, b);
  const Quat<double> turn = left.q * conjugate(right.q);
  for (int32_t c = t.pieces[p].ctrl_begin; c < t.pieces[p].ctrl_end; ++c) {
    const auto i = static_cast<size_t>(c);
    const Quat<double> q = turn * Quat<double>{t.rot[4 * i], t.rot[4 * i + 1], t.rot[4 * i + 2], t.rot[4 * i + 3]};
    t.rot[4 * i] = q.w;
    t.rot[4 * i + 1] = q.x;
    t.rot[4 * i + 2] = q.y;
    t.rot[4 * i + 3] = q.z;
  }
  const traj::Kinematics<double> turned = piece_eval(t, p, b);
  for (int32_t c = t.pieces[p].ctrl_begin; c < t.pieces[p].ctrl_end; ++c) {
    const auto i = static_cast<size_t>(c);
    t.pos[3 * i] += left.p.x - turned.p.x;
    t.pos[3 * i + 1] += left.p.y - turned.p.y;
    t.pos[3 * i + 2] += left.p.z - turned.p.z;
  }
}

// Three pieces: a slow one, a fast spinning one joined C0, and a slow one joined C1 in name (the planner reads no
// join class but -1).
render::Trajectory pieces3() {
  const int64_t b1 = kT0 + 2 * kW + 12'345, b2 = kT0 + 4 * kW + 1, h1 = 1'999'993;
  render::Trajectory t = piecewise_trajectory(
      kT1 + kW, {{kT0 - kW, kT0 - kW - 3 * kKnotNs, kKnotNs, {0.1, 0.0, 1.5}, {0.1, 0.0, 0.0}, {0.0, 0.2, 0.0}, 0},
                 {b1, b1 - 2 * h1 - 7, h1, {0.0, 0.0, 1.5}, {-1.5, 0.8, 0.0}, {4.0, -9.0, 2.0}, 0},
                 {b2, b2 - 3, 10'000'000, {0.0, 0.0, 1.5}, {0.05, 0.0, 0.0}, {0.0, 0.0, 0.3}, 1}});
  join_c0(t, 1);
  join_c0(t, 2);
  return t;
}

// A camera on a piecewise rig (T_rig_cam the identity), a wall, a box on a piecewise trajectory, and a point light
// carried by a third body on a piecewise trajectory: meshless or a small box.
render::SceneBundle scene(bool meshless_carrier) {
  render::SceneBundle s;
  s.light_groups = {"default"};
  render::Trajectory rig = pieces3();
  for (size_t i = 0; i < rig.pos.size(); i += 3) rig.pos[i + 2] -= 1.5;  // the rig near the origin
  s.trajectories.push_back(rig);
  s.trajectories.push_back(pieces3());
  render::Trajectory lamp_path = pieces3();
  for (size_t i = 0; i < lamp_path.pos.size(); i += 3) lamp_path.pos[i + 1] -= 0.5;
  s.trajectories.push_back(lamp_path);
  s.cameras.push_back(pinhole(40, 30, 40.0, 0));
  render::Body wall = make_body(1, plane_mesh(2.0, 2), {0.3f, 0.3f, 0.3f});
  wall.T_world_body = translation(0.0, 0.0, 2.0);
  s.bodies.push_back(wall);
  render::Body box = make_body(2, box_mesh(0.1f, 0.1f, 0.1f), {0.9f, 0.9f, 0.9f});
  box.trajectory = 1;
  s.bodies.push_back(box);
  render::Body carrier = make_body(9, meshless_carrier ? render::Mesh{} : box_mesh(0.02f, 0.02f, 0.02f),
                                   {0.8f, 0.8f, 0.8f});
  carrier.trajectory = 2;
  s.bodies.push_back(carrier);
  render::Light lamp = point_light({0.0, 0.1, 0.2}, 2.0f, true);
  lamp.body = 2;
  s.lights.push_back(lamp);
  s.lights.push_back(ambient_light(0.3f));
  s.t_start_ns = kT0 - kW;
  s.t_end_ns = kT1 + kW;
  return s;
}

// Bitwise equality of two 3 x 3 matrices (the engine's Mat3d or render's Mat3<double>) or two vectors.
template <class A, class B>
bool same(const A& a, const B& b) {
  static_assert(sizeof(A) == sizeof(B));
  return std::memcmp(&a, &b, sizeof(A)) == 0;
}

Vec3d rotate(const Mat3d& r, const Vec3d& x) {
  return {r.m[0][0] * x.x + r.m[0][1] * x.y + r.m[0][2] * x.z, r.m[1][0] * x.x + r.m[1][1] * x.y + r.m[1][2] * x.z,
          r.m[2][0] * x.x + r.m[2][1] * x.y + r.m[2][2] * x.z};
}

std::vector<int64_t> breakpoint_times(const render::Trajectory& t) {
  std::vector<int64_t> out;
  for (size_t p = 1; p < t.pieces.size(); ++p) {
    for (int64_t d = -1; d <= 1; ++d) out.push_back(t.pieces[p].begin_ns + d);
  }
  return out;
}

}  // namespace

TEST_CASE("TrajectoryMotion's pose is the renderer's body_state per piece: camera_pose and carriers at b - 1, b, b + 1") {
  const render::SceneBundle s = scene(false);
  const render::SceneData data(s, render::SceneSettings{});
  const SceneMotion motion(s, {}, true);
  const render::Trajectory& rig = s.trajectories[0];
  std::vector<int64_t> times = breakpoint_times(rig);
  for (int64_t t = kT0; t < kT1; t += 3'456'789) times.push_back(t);
  for (const int64_t t : times) {
    INFO("t = " << t);
    const size_t p = static_cast<size_t>(
        std::upper_bound(rig.pieces.begin(), rig.pieces.end(), t,
                         [](int64_t x, const render::TrajectoryPiece& q) { return x < q.begin_ns; }) -
        rig.pieces.begin() - 1);
    const traj::Kinematics<double> k = piece_eval(rig, p, t);
    render::BodyState st{};
    REQUIRE(render::body_state(data.trajectories()[0], t, &st));
    CHECK(same(st.pose.p, k.p));
    CHECK(same(st.pose.r, to_matrix(k.q)));
    const auto [r_wc, c] = motion.camera_pose(0, t);  // T_rig_cam = I: the rig's pose
    CHECK(same(c, k.p));
    CHECK(same(r_wc, quat_to_matrix(k.q)));
    // The carriers' mid-window poses (SceneMotion::carrier): a body window centred on t, and its light.
    const traj::Kinematics<double> kb = piece_eval(s.trajectories[1], p, t), kl = piece_eval(s.trajectories[2], p, t);
    const std::vector<BodyWindow> bw = motion.body_windows(t - 1000, t + 1000);
    REQUIRE(bw.size() == 3);
    CHECK(same(bw[1].centre, kb.p));
    CHECK(same(bw[1].rotation, quat_to_matrix(kb.q)));
    CHECK(same(bw[2].centre, kl.p));
    const std::vector<LightWindow> lw = motion.light_windows(t - 1000, t + 1000);
    REQUIRE(lw.size() == 1);
    render::BodyState sl{};
    REQUIRE(render::body_state(data.trajectories()[2], t, &sl));
    CHECK(same(sl.pose.p, kl.p));
    const Vec3d expect = kl.p + rotate(quat_to_matrix(kl.q), Vec3d{0.0, 0.1, 0.2});
    CHECK(std::abs(lw[0].position.x - expect.x) <= 1e-15);
    CHECK(std::abs(lw[0].position.y - expect.y) <= 1e-15);
    CHECK(std::abs(lw[0].position.z - expect.z) <= 1e-15);
  }
  CHECK_THROWS_AS(motion.camera_pose(0, rig.t0_ns - 1), std::invalid_argument);
}

TEST_CASE("window bounds per piece: at least the sampled speeds across a C0 join, only the pieces a window touches") {
  const render::Trajectory t = pieces3();
  const TrajectoryMotion m(t, {});
  const render::SceneData data([&] {
    render::SceneBundle b = scene(false);
    b.trajectories[1] = t;
    return b;
  }(), render::SceneSettings{});
  const render::TrajectoryView& tv = data.trajectories()[1];
  const auto sampled = [&](int64_t a, int64_t b) {
    double v = 0.0, w = 0.0;
    for (int64_t x = a; x <= b; x += std::max<int64_t>(1, (b - a) / 4000)) {
      render::BodyState st{};
      REQUIRE(render::body_state(tv, x, &st));
      v = std::max(v, norm(st.v));
      w = std::max(w, norm(st.omega));
    }
    return std::pair{v, w};
  };
  // The C0 joins are continuous to rounding: the pose at b and just before it agree.
  for (size_t p = 1; p < 3; ++p) {
    const int64_t b = t.pieces[p].begin_ns;
    CHECK(norm(piece_eval(t, p, b).p - piece_eval(t, p - 1, b).p) <= 1e-12);
  }
  const int64_t b1 = t.pieces[1].begin_ns, b2 = t.pieces[2].begin_ns;
  for (const auto& [a, b] : {std::pair{b1 - kW / 2, b1 + kW / 2}, std::pair{b2 - kW / 2, b2 + kW / 2},
                             std::pair{b1 - 1, b1 + kW}, std::pair{kT0, kT0 + kW}, std::pair{b1, b1 + kW},
                             std::pair{b2 - kW, b2 - 1}, std::pair{b2, b2 + kW}, std::pair{kT0, kT1}}) {
    INFO("[" << a << ", " << b << "]");
    const WindowMotion w = m.window(a, b);
    const auto [v, omega] = sampled(a, b);
    CHECK(w.v >= v);
    CHECK(w.omega >= omega);
    CHECK_FALSE(w.held);
  }
  // Piece 1 spins fast (|w| ~ 10 rad/s): a window touching it takes its bounds, one that does not stays slow.
  CHECK(m.window(b1 - 1, b1 + kW).omega > 5.0);
  CHECK(m.window(kT0, b1 - 1).omega < 1.0);
  CHECK(m.window(b2, b2 + kW).omega < 1.0);
  CHECK(m.window(b2 - 1, b2 + kW).omega > 5.0);  // b2 - 1 is piece 1's
  // A window lying inside one piece takes exactly that piece's segment bounds.
  const TrajectoryMotion alone([&] {
    render::Trajectory one;
    one.order = t.order;
    const render::TrajectoryPiece& q = t.pieces[2];
    one.t0_ns = q.t0_ns;
    one.h_ns = q.h_ns;
    one.rot.assign(t.rot.begin() + 4 * q.ctrl_begin, t.rot.end());
    one.pos.assign(t.pos.begin() + 3 * q.ctrl_begin, t.pos.end());
    one.span_begin_ns = q.begin_ns;
    one.span_end_ns = t.span_end_ns;
    return one;
  }(), {});
  const WindowMotion x = m.window(b2 + 5, b2 + kW), y = alone.window(b2 + 5, b2 + kW);
  CHECK(std::memcmp(&x.v, &y.v, sizeof(double)) == 0);
  CHECK(std::memcmp(&x.omega, &y.omega, sizeof(double)) == 0);
  for (const int64_t q : {b2, b2 + 1, b2 + kW, t.span_end_ns}) {
    const auto [r1, p1] = m.pose(q);
    const auto [r2, p2] = alone.pose(q);
    CHECK(same(r1, r2));
    CHECK(same(p1, p2));
  }
  // A window given backwards takes the segment of its start, as a single spline does.
  CHECK(m.window(b1 + 10, b1 - 10).omega == m.window(b1 + 10, b1 + 10).omega);
}

TEST_CASE("a hold across a join holds the body; a step join is refused by the planner (BL-12)") {
  // Pieces 1 and 2 of a body hold one control point over the join's neighbourhood.
  render::SceneBundle s = scene(false);
  render::Trajectory& t = s.trajectories[1];
  const int k = t.order;
  const render::TrajectoryPiece q1 = t.pieces[1], q2 = t.pieces[2];
  const int32_t lo = q1.ctrl_end - k - 1;
  const std::vector<double> rq(t.rot.begin() + 4 * lo, t.rot.begin() + 4 * lo + 4);
  const std::vector<double> pq(t.pos.begin() + 3 * lo, t.pos.begin() + 3 * lo + 3);
  for (int32_t c = lo; c < q1.ctrl_end; ++c) {
    std::copy(rq.begin(), rq.end(), t.rot.begin() + 4 * c);
    std::copy(pq.begin(), pq.end(), t.pos.begin() + 3 * c);
  }
  for (int32_t c = q2.ctrl_begin; c < q2.ctrl_begin + k; ++c) {
    std::copy(rq.begin(), rq.end(), t.rot.begin() + 4 * c);
    std::copy(pq.begin(), pq.end(), t.pos.begin() + 3 * c);
  }
  const int64_t hb = q1.t0_ns + int64_t{q1.ctrl_end - q1.ctrl_begin - k - 1} * q1.h_ns, he = q2.t0_ns + q2.h_ns;
  REQUIRE(hb < q2.begin_ns);
  REQUIRE(q2.begin_ns < he);
  std::vector<Holds> holds(3);
  holds[1] = {{hb, he}};
  const SceneMotion motion(s, holds, true);
  const std::vector<uint32_t> held = motion.held_bodies(hb, he - 1);
  CHECK(std::find(held.begin(), held.end(), 2u) != held.end());
  const TrajectoryMotion tm(t, holds[1]);
  const WindowMotion w = tm.window(hb, he - 1);
  CHECK(w.held);
  CHECK(w.v == 0.0);  // every segment under the hold is exactly still, in both pieces
  CHECK(w.omega == 0.0);
  const auto [r0, p0] = tm.pose(hb);
  for (const int64_t x : {q2.begin_ns - 1, q2.begin_ns, he - 1}) {
    const auto [r, p] = tm.pose(x);
    CHECK(same(r, r0));
    CHECK(same(p, p0));
  }
  // A step join: the pose jumps, which no window bound covers.
  render::SceneBundle stepped = scene(false);
  stepped.trajectories[2].pieces[2].join = -1;
  render::validate(stepped);  // C1 admits it; the planner does not
  try {
    const SceneMotion refused(stepped, {}, true);
    FAIL("a step join was accepted");
  } catch (const std::invalid_argument& e) {
    CHECK(std::string(e.what()).find("trajectory 2 has a step join") != std::string::npos);
  }
}

TEST_CASE("a join labelled C0 or C1 keeps the pose (kJoinTol*): a jump is refused, zero bounds across a join not held") {
  // Two still pieces (v = w = 0: every segment bound exactly 0), the second's pose offset by (dx, angle about z).
  const int64_t b = kT0 + 2 * kW + 12'345, h = 1'999'993;
  const auto still = [&](double dx, double angle, int8_t join) {
    render::Trajectory t = piecewise_trajectory(
        kT1 + kW, {{kT0 - kW, kT0 - kW - 3 * kKnotNs, kKnotNs, {0.1, 0.0, 1.0}, {0, 0, 0}, {0, 0, 0}, 0},
                   {b, b - 2 * h - 7, h, {0.1 + dx, 0.0, 1.0}, {0, 0, 0}, {0, 0, 0}, join}});
    const Quat<double> turn = so3::exp(Vec3<double>{0.0, 0.0, angle});
    for (int32_t c = t.pieces[1].ctrl_begin; c < t.pieces[1].ctrl_end; ++c) {
      const auto i = static_cast<size_t>(c);
      const Quat<double> q = turn * Quat<double>{t.rot[4 * i], t.rot[4 * i + 1], t.rot[4 * i + 2], t.rot[4 * i + 3]};
      t.rot[4 * i] = q.w;
      t.rot[4 * i + 1] = q.x;
      t.rot[4 * i + 2] = q.y;
      t.rot[4 * i + 3] = q.z;
    }
    return t;
  };
  const auto with_body = [&](const render::Trajectory& t) {
    render::SceneBundle s = scene(false);
    s.trajectories[1] = t;
    render::validate(s);  // C1 checks no continuity: the planner does
    return s;
  };
  const auto refusal = [&](const render::Trajectory& t) {
    const render::SceneBundle s = with_body(t);  // SceneMotion keeps a reference to its bundle
    try {
      const SceneMotion motion(s, {}, true);
    } catch (const std::invalid_argument& e) {
      return std::string(e.what());
    }
    return std::string();
  };
  // Within the tolerance (and bitwise equal): planned, but a window across b is held only inside one piece, since the
  // two pieces' constants differ (the reviewed case: held across b would bake the left pose past b).
  for (const auto& [dx, angle] : {std::pair{0.0, 0.0}, std::pair{5e-9, 0.0}, std::pair{0.0, 5e-9}}) {
    INFO("dx = " << dx << ", angle = " << angle);
    const render::Trajectory t = still(dx, angle, 0);
    const TrajectoryMotion m(t, {});
    CHECK_FALSE(m.jump().has_value());
    for (const auto& [a, e] : {std::pair{b - kW / 2, b + kW / 2}, std::pair{b - 1, b}, std::pair{kT0, kT1}}) {
      const WindowMotion w = m.window(a, e);
      CHECK(w.v == 0.0);
      CHECK(w.omega == 0.0);
      CHECK_FALSE(w.held);
    }
    CHECK(m.window(kT0, b - 1).held);
    CHECK(m.window(b, b + kW).held);
    const render::SceneBundle s = with_body(t);
    const SceneMotion motion(s, {}, true);
    const auto held = [&](int64_t a, int64_t e) {
      const std::vector<uint32_t> ids = motion.held_bodies(a, e);
      return std::find(ids.begin(), ids.end(), 2u) != ids.end();
    };
    CHECK_FALSE(held(b - kW / 2, b + kW / 2));
    CHECK(held(b, b + kW));
    CHECK(motion.body_windows(b - kW / 2, b + kW / 2)[1].held == false);
  }
  // A jump in position or rotation beyond 1e-8 under a C0 or C1 label: a step in fact, refused as one.
  for (const auto& [dx, angle, join] : {std::tuple{0.5, 0.0, int8_t{0}}, std::tuple{2e-8, 0.0, int8_t{0}},
                                        std::tuple{0.0, 2e-8, int8_t{1}}, std::tuple{0.0, 0.3, int8_t{1}}}) {
    INFO("dx = " << dx << ", angle = " << angle << ", join " << int{join});
    const render::Trajectory t = still(dx, angle, join);
    const TrajectoryMotion m(t, {});
    REQUIRE(m.jump().has_value());
    CHECK(m.jump()->piece == 1);
    CHECK(m.jump()->join == join);
    CHECK(m.jump()->at_ns == b);
    CHECK(m.jump()->m == doctest::Approx(dx).epsilon(1e-6));
    CHECK(m.jump()->rad == doctest::Approx(angle).epsilon(1e-6));
    CHECK_FALSE(m.window(b - kW / 2, b + kW / 2).held);
    const std::string what = refusal(t);
    CHECK(what.find("trajectory 1 piece 1 is joined C" + std::to_string(join)) != std::string::npos);
    CHECK(what.find("pose jumps by") != std::string::npos);
  }
  // The C0 joins of pieces3 (join_c0: continuous to rounding) pass.
  CHECK_FALSE(TrajectoryMotion(pieces3(), {}).jump().has_value());
  CHECK(refusal(pieces3()).empty());
}

TEST_CASE("a meshless light carrier: no mesh distance, out of the body lists, its light riding it (T5b)") {
  const render::SceneBundle meshless = scene(true), meshed = scene(false);
  render::validate(meshless);
  const SceneMotion a(meshless, {}, true), b(meshed, {}, true);
  CHECK(a.mesh(2) == nullptr);
  CHECK(b.mesh(2) != nullptr);
  const int64_t t = kT0 + 3 * kW;
  const std::vector<BodyWindow> wa = a.body_windows(t, t + kW), wb = b.body_windows(t, t + kW);
  REQUIRE(wa.size() == 2);
  REQUIRE(wb.size() == 3);
  CHECK(wa[0].index == 0);
  CHECK(wa[1].index == 1);
  CHECK(wb[2].index == 2);
  const std::vector<LightWindow> la = a.light_windows(t, t + kW), lb = b.light_windows(t, t + kW);
  REQUIRE(la.size() == 1);
  CHECK(la[0].moving);
  CHECK(la[0].body == 2);
  CHECK(same(la[0].position, lb[0].position));  // the same carrier pose either way
  CHECK(la[0].v == lb[0].v);
  // A held meshless carrier joins no held set (it has no geometry); the meshed one does.
  render::SceneBundle still_a = meshless, still_b = meshed;
  for (render::SceneBundle* s : {&still_a, &still_b}) {
    s->trajectories[2] = static_trajectory(kT0 - kW, kT1 + kW, quat_identity<double>(), {0.0, -0.5, 1.0});
  }
  const SceneMotion ha(still_a, {}, true), hb(still_b, {}, true);
  const std::vector<uint32_t> held_a = ha.held_bodies(t, t + kW), held_b = hb.held_bodies(t, t + kW);
  CHECK(std::find(held_a.begin(), held_a.end(), 9u) == held_a.end());
  CHECK(std::find(held_b.begin(), held_b.end(), 9u) != held_b.end());

  // The host engine end to end: planned windows, events, and the carrier's id in no ground truth.
  EngineConfig c;
  c.render = options(render::BackendKind::kEmbree, 4);
  c.render.scene.n_subsamples = 16;
  c.light_groups = {sensor::make_constant_waveform()};
  SensorSpec spec;
  spec.params = sensor::preset("prophesee_gen3");
  spec.params.illuminance_scale = 200.0;
  spec.precision = sensor::Precision::kFp32;
  spec.num_threads = 4;
  const std::unique_ptr<Engine> e = make_engine(meshless, c, {spec});
  CameraEngine& cam = e->camera(0);
  cam.start(kT0, 3 * kW, false);
  size_t events = 0;
  for (int w = 0; w < 3; ++w) {
    const StepResult r = cam.step(kT0 + w * kW);
    CHECK(r.planned);
    CHECK(r.plan.moving_lights == 1);
    CHECK(r.plan.moving_bodies == 1);  // the box; the carrier is in no list
    for (const OutEvent& ev : r.ready) CHECK(ev.instance != 9u);
    events += r.ready.size();
  }
  const std::vector<OutEvent> rest = cam.flush();
  for (const OutEvent& ev : rest) CHECK(ev.instance != 9u);
  CHECK(events + rest.size() > 0);
}
