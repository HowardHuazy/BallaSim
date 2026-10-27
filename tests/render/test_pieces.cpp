// Piecewise trajectories in the renderer (C1 "pieces", docs/design/phase3/blender.md §3.7, §4.7.7, T5; Phase 3
// BL-5b): render::validate's per-piece rules, body_state's right-continuous piece lookup against each piece's own
// spline (b - 1 the left piece, b and b + 1 the right one), SceneData's per-piece nodes, and whole frames of a piecewise
// mover equal to those of each piece's own single spline, bit for bit. The device copy's parity is in test_device.cpp.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/traj/spline_build.h"
#include "render/bundle.h"
#include "render/pose.h"
#include "render/renderer.h"
#include "render/scene_data.h"
#include "scenes.h"

namespace balla::render::test {
namespace {

constexpr int64_t kT0 = 1'000'000'000, kT1 = 1'200'000'000;

// Three pieces with an odd breakpoint and spacing, knot origins before their begins, and other motions per piece.
std::vector<PieceSpec> spec() {
  const int64_t b1 = kT0 + 70'012'345, b2 = kT0 + 130'000'000, h1 = 2'000'007;
  return {{kT0, kT0 - 3 * kKnotNs, kKnotNs, {0.1, 0.2, 1.5}, {0.4, -0.2, 0.3}, {0.3, 1.1, -0.4}, 0},
          {b1, b1 - 2 * h1 - 7, h1, {0.15, 0.1, 1.6}, {-0.5, 0.25, 0.1}, {4.0, -9.0, 2.0}, 0},
          {b2, b2 - 3, 10'000'000, {0.05, 0.12, 1.4}, {0.2, 0.3, -0.1}, {-1.0, 0.5, 0.7}, 1}};
}

// A static rig, a floor plane at z = 3 m and a box on the piecewise trajectory carrying a point light.
SceneBundle scene() {
  SceneBundle s;
  s.light_groups = {"default"};
  s.trajectories.push_back(static_trajectory(kT0, kT1, Quat<double>{1, 0, 0, 0}, {0, 0, 0}));
  s.trajectories.push_back(piecewise_trajectory(kT1, spec()));
  s.cameras.push_back(pinhole(64, 48, 50.0, 0));
  Body floor = make_body(1, plane_mesh(3.0, 4), {0.6f, 0.6f, 0.6f});
  floor.T_world_body = translation(0.0, 0.0, 3.0);
  s.bodies.push_back(floor);
  Body box = make_body(5, box_mesh(0.2f, 0.15f, 0.1f), {0.9f, 0.5f, 0.3f});
  box.trajectory = 1;
  s.bodies.push_back(box);
  Light lamp = point_light({0.0, 0.3, -0.2}, 2.0f);
  lamp.body = 1;
  s.lights.push_back(lamp);
  s.lights.push_back(ambient_light(0.1f));
  return s;
}

// Piece p of trajectory t as the single spline of its rows, serving [first, last].
Trajectory piece_alone(const Trajectory& t, size_t p, int64_t first, int64_t last) {
  const TrajectoryPiece& q = t.pieces[p];
  Trajectory out;
  out.order = t.order;
  out.t0_ns = q.t0_ns;
  out.h_ns = q.h_ns;
  out.rot.assign(t.rot.begin() + 4 * q.ctrl_begin, t.rot.begin() + 4 * q.ctrl_end);
  out.pos.assign(t.pos.begin() + 3 * q.ctrl_begin, t.pos.begin() + 3 * q.ctrl_end);
  out.span_begin_ns = first;
  out.span_end_ns = last;
  return out;
}

// The kinematics of piece p's own spline (nodes from its rows alone) at t_ns.
traj::Kinematics<double> piece_eval(const Trajectory& t, size_t p, int64_t t_ns) {
  const TrajectoryPiece& q = t.pieces[p];
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

bool same(const Mat3<double>& a, const Mat3<double>& b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }
bool same(const Vec3<double>& a, const Vec3<double>& b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

template <class Edit>
std::string refusal(Edit edit) {
  SceneBundle s = scene();
  edit(s.trajectories[1]);
  try {
    validate(s);
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "";
}

}  // namespace

TEST_CASE("C1 pieces in render::validate: a piecewise trajectory passes; every rule refuses, naming the piece") {
  CHECK_NOTHROW(validate(scene()));
  using T = Trajectory;
  const auto has = [](const std::string& msg, const char* what) {
    INFO(msg);
    CHECK(msg.find(what) != std::string::npos);
  };
  has(refusal([](T& t) { t.pieces.resize(1); }), "at least 2 pieces");
  has(refusal([](T& t) { t.pieces[0].begin_ns += 1; }), "piece 0 must begin at the span's begin");
  has(refusal([](T& t) { t.pieces[2].begin_ns = t.span_end_ns; }), "the last piece before its end");
  has(refusal([](T& t) { t.t0_ns -= 1; }), "t0_ns and h_ns must equal piece 0's");
  has(refusal([](T& t) { t.h_ns += 1; }), "t0_ns and h_ns must equal piece 0's");
  has(refusal([](T& t) { t.pieces[2].begin_ns = t.pieces[1].begin_ns; }), "piece 2: begin_ns must exceed");
  has(refusal([](T& t) { t.pieces[1].h_ns = 0; }), "piece 1: h_ns must be in [1, 2^31)");
  has(refusal([](T& t) { t.pieces[1].h_ns = int64_t{1} << 31; }), "piece 1: h_ns must be in [1, 2^31)");
  has(refusal([](T& t) { t.pieces[1].ctrl_begin += 1; }), "piece 1: control rows must follow");
  has(refusal([](T& t) { t.pieces[2].ctrl_end = t.pieces[2].ctrl_begin + 5; }), "piece 2: control rows");
  has(refusal([](T& t) { t.pieces[2].ctrl_end += 1; }), "piece 2: control rows");  // beyond rot and pos
  has(refusal([](T& t) {
        t.rot.insert(t.rot.end(), {1, 0, 0, 0});
        t.pos.insert(t.pos.end(), {0, 0, 0});
      }),
      "must cover rot and pos");
  has(refusal([](T& t) { t.pieces[0].join = 1; }), "piece 0: join must be");
  has(refusal([](T& t) { t.pieces[1].join = 2; }), "piece 1: join must be");
  has(refusal([](T& t) { t.pieces[1].t0_ns = t.pieces[1].begin_ns + 1; }), "piece 1: its spline's support");
  has(refusal([](T& t) { t.pieces[1].h_ns /= 2; }), "piece 1: its spline's support");
  // The pi/2 rule over a piece's own controls; a join may turn by more (each piece is its own spline).
  has(refusal([](T& t) {
        const auto i = static_cast<size_t>(t.pieces[1].ctrl_begin + 3);
        t.rot[4 * i] = 0.0;
        t.rot[4 * i + 1] = 1.0;
        t.rot[4 * i + 2] = 0.0;
        t.rot[4 * i + 3] = 0.0;
      }),
      "piece 1: consecutive control rotations differ by more than pi/2");
  SceneBundle turned = scene();
  Trajectory& t = turned.trajectories[1];
  for (int32_t c = t.pieces[2].ctrl_begin; c < t.pieces[2].ctrl_end; ++c) {  // piece 2 turned by pi about x
    const auto i = static_cast<size_t>(c);
    const Quat<double> q{t.rot[4 * i], t.rot[4 * i + 1], t.rot[4 * i + 2], t.rot[4 * i + 3]};
    const Quat<double> r = Quat<double>{0, 1, 0, 0} * q;
    t.rot[4 * i] = r.w;
    t.rot[4 * i + 1] = r.x;
    t.rot[4 * i + 2] = r.y;
    t.rot[4 * i + 3] = r.z;
  }
  CHECK_NOTHROW(validate(turned));
}

TEST_CASE("body_state takes the piece serving a time: b - 1 the left piece's spline, b and b + 1 the right one's") {
  const SceneBundle s = scene();
  const Trajectory& tr = s.trajectories[1];
  const SceneData data(s, SceneSettings{});
  const TrajectoryView& tv = data.trajectories()[1];
  REQUIRE(tv.num_pieces == 3);
  REQUIRE(data.piece_views()[1].size() == 3);
  CHECK(data.piece_views()[0].empty());
  CHECK(data.trajectories()[0].num_pieces == 0);
  CHECK(data.spline_nodes()[1].size() == tr.pos.size() / 3);  // the pieces' nodes one after the other
  for (size_t p = 0; p < 3; ++p) {
    CHECK(tv.pieces[p].begin_ns == tr.pieces[p].begin_ns);
    CHECK(tv.pieces[p].spline.nodes == data.spline_nodes()[1].data() + tr.pieces[p].ctrl_begin);
  }
  std::mt19937_64 rng(11);
  for (size_t p = 0; p < 3; ++p) {
    const int64_t a = tr.pieces[p].begin_ns, b = p < 2 ? tr.pieces[p + 1].begin_ns : kT1 + 1;
    std::vector<int64_t> times{a, a + 1, b - 1};
    std::uniform_int_distribution<int64_t> in(a, b - 1);
    for (int i = 0; i < 200; ++i) times.push_back(in(rng));
    for (const int64_t t : times) {
      INFO("piece " << p << ", t = " << t);
      BodyState st{};
      REQUIRE(body_state(tv, t, &st));
      const traj::Kinematics<double> k = piece_eval(tr, p, t);
      CHECK(same(st.pose.r, to_matrix(k.q)));
      CHECK(same(st.pose.p, k.p));
      CHECK(same(st.v, k.v));
      CHECK(same(st.omega, st.pose.r * k.omega));
    }
    if (p > 0) {  // at b the left piece's support still reaches, with another pose: the right piece is taken
      BodyState st{};
      REQUIRE(body_state(tv, a, &st));
      CHECK_FALSE(same(st.pose.p, piece_eval(tr, p - 1, a).p));
    }
  }
  BodyState st{};
  CHECK_FALSE(body_state(tv, kT0 - 1, &st));
  CHECK_FALSE(body_state(tv, kT1 + 1, &st));
  CHECK(body_state(tv, kT1, &st));  // the span is closed
}

TEST_CASE("a piecewise mover renders as each piece's own single spline, bit for bit (frames, gt and light rows)") {
  const SceneBundle s = scene();
  const Trajectory& tr = s.trajectories[1];
  Renderer whole(s, options(BackendKind::kEmbree, 2));
  bool hit = false;
  for (size_t p = 0; p < 3; ++p) {
    const int64_t a = tr.pieces[p].begin_ns, b = p < 2 ? tr.pieces[p + 1].begin_ns - 1 : kT1;
    SceneBundle alone = s;
    alone.trajectories[1] = piece_alone(tr, p, a, b);
    alone.trajectories[0] = static_trajectory(a, b, Quat<double>{1, 0, 0, 0}, {0, 0, 0});
    Renderer single(alone, options(BackendKind::kEmbree, 2));
    for (const int64_t t : {a, a + 1, (a + b) / 2, b}) {
      INFO("piece " << p << ", t = " << t);
      const DenseFrame x = whole.dense(0, t, true, true), y = single.dense(0, t, true, true);
      REQUIRE(x.irradiance.size() == y.irradiance.size());
      CHECK(std::memcmp(x.irradiance.data(), y.irradiance.data(), x.irradiance.size() * sizeof(float)) == 0);
      REQUIRE(x.gt.size() == y.gt.size());
      CHECK(std::memcmp(x.gt.data(), y.gt.data(), x.gt.size() * sizeof(GtSample)) == 0);
      for (const GtSample& g : x.gt) hit = hit || g.instance == 5;
    }
  }
  CHECK(hit);  // the mover is in view
}

}  // namespace balla::render::test
