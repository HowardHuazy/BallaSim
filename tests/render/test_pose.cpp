// Pose table rows (render/pose.h, render/pose_table.h): camera-relative transforms against direct
// fp64 composition, relative twists against finite differences of the motion (M2 D3), and span
// checks.
#include <doctest/doctest.h>

#include <cmath>

#include "core/time/time.h"
#include "core/traj/spline_build.h"
#include "render/pose_table.h"
#include "render/scene_data.h"
#include "scenes.h"

using namespace balla;
using namespace balla::render;
using namespace balla::render::test;

namespace {

constexpr int64_t kT0 = 1'000'000'000, kT1 = 1'200'000'000;

SceneBundle moving_scene() {
  SceneBundle s;
  s.light_groups = {"default"};
  // Rig: translating and turning; camera offset from the rig and rotated about x.
  s.trajectories.push_back(moving_trajectory(kT0, kT1, so3::exp(Vec3<double>{0.1, -0.2, 0.05}),
                                             {0.3, -0.1, 0.2}, {0.4, -0.2, 0.3}, {0.3, 1.1, -0.4}));
  // Mover: translating and spinning fast.
  s.trajectories.push_back(moving_trajectory(kT0, kT1, so3::exp(Vec3<double>{-0.3, 0.2, 0.7}),
                                             {0.1, 0.2, 1.5}, {-0.5, 0.25, 0.1}, {4.0, -9.0, 2.0}));
  Camera cam = pinhole(64, 48, 50.0, 0);
  cam.T_rig_cam = {1, 0, 0, 0.05, 0, std::cos(0.3), -std::sin(0.3), -0.02, 0, std::sin(0.3), std::cos(0.3), 0.01, 0, 0, 0, 1};
  s.cameras.push_back(cam);
  s.bodies.push_back(make_body(3, box_mesh(0.1f, 0.05f, 0.2f)));
  s.bodies.back().trajectory = 1;
  Body floor = make_body(1, plane_mesh(2.0));
  floor.T_world_body = translation(0.0, 0.0, 3.0);
  s.bodies.push_back(floor);
  Light lamp = point_light({0.02, 0.03, -0.1}, 1.0f);
  lamp.body = 0;  // carried by the mover (bodies[0])
  s.lights.push_back(lamp);
  Light sun;
  sun.kind = BundleLightKind::kDirectional;
  sun.direction = {0.0, 0.6, 0.8};
  sun.intensity = 1.0f;
  s.lights.push_back(sun);
  return s;
}

Mat3<double> quat_matrix(Quat<double> q) { return to_matrix(q); }

// Direct fp64 composition of T_body_cam and the camera-frame position of a body point.
struct Direct {
  Mat3<double> r_wc, r_wb;
  Vec3<double> p_wc, p_wb;
};

Direct direct(const SceneBundle& s, int64_t t) {
  auto eval = [&](const Trajectory& tr, Quat<double>* q, Vec3<double>* p) {
    std::vector<Quat<double>> qs;
    std::vector<Vec3<double>> ps;
    for (size_t c = 0; c < tr.pos.size() / 3; ++c) {
      qs.push_back({tr.rot[4 * c], tr.rot[4 * c + 1], tr.rot[4 * c + 2], tr.rot[4 * c + 3]});
      ps.push_back({tr.pos[3 * c], tr.pos[3 * c + 1], tr.pos[3 * c + 2]});
    }
    const auto nodes = traj::build_spline_nodes(qs, ps, {0, 0, 0});
    const traj::SplineView<double> v{nodes.data(), static_cast<int32_t>(nodes.size()) - 5,
                                     static_cast<int32_t>(tr.h_ns), 1e9 / static_cast<double>(tr.h_ns)};
    traj::SegmentTime st{};
    REQUIRE(traj::locate(t, tr.t0_ns, v.h_ns, v.num_segments, &st));
    const auto k = traj::evaluate<6>(v, st);
    *q = k.q;
    *p = k.p;
  };
  Quat<double> q_rig, q_b;
  Vec3<double> p_rig, p_b;
  eval(s.trajectories[0], &q_rig, &p_rig);
  eval(s.trajectories[1], &q_b, &p_b);
  const auto& T = s.cameras[0].T_rig_cam;
  const Mat3<double> r_rc{{{T[0], T[1], T[2]}, {T[4], T[5], T[6]}, {T[8], T[9], T[10]}}};
  const Mat3<double> r_rig = quat_matrix(q_rig);
  Direct d;
  d.r_wc = r_rig * r_rc;
  d.p_wc = p_rig + r_rig * Vec3<double>{T[3], T[7], T[11]};
  d.r_wb = quat_matrix(q_b);
  d.p_wb = p_b;
  return d;
}

Vec3<double> mat_t_vec(const Mat3<double>& a, Vec3<double> v) { return transpose(a) * v; }

}  // namespace

TEST_CASE("pose rows equal the direct fp64 composition, rounded to fp32") {
  const SceneBundle s = moving_scene();
  const SceneData scene(s, SceneSettings{});
  REQUIRE(scene.geometries().size() == 2);  // world (floor) + mover
  ThreadPool pool(2);
  PoseTable table(2, 2, 16);
  table.reset(kT0);
  const int64_t t = kT0 + 73'456'789;
  uint32_t slot;
  REQUIRE(table.slot(0, t, &slot));
  table.evaluate(scene.pose_view(), pool);
  const PoseTableView v = table.view();
  const Direct d = direct(s, t);

  // World row: T_world_cam.
  const PoseRow& w = pose_row(v, slot, 0);
  for (int i = 0; i < 3; ++i) {
    const float r[3] = {w.m[i].x, w.m[i].y, w.m[i].z};
    for (int j = 0; j < 3; ++j) CHECK(r[j] == doctest::Approx(d.r_wc.m[i][j]).epsilon(1e-6).scale(1.0));
  }
  CHECK(w.m[2].w == doctest::Approx(d.p_wc.z).epsilon(1e-6));
  CHECK(std::isinf(w.sphere.w));

  // Mover row: R_bc = R_wb^T R_wc, p_bc = R_wb^T (p_wc - p_wb); sphere centre in the camera frame.
  const PoseRow& m = pose_row(v, slot, 1);
  const Mat3<double> r_bc = transpose(d.r_wb) * d.r_wc;
  const Vec3<double> p_bc = mat_t_vec(d.r_wb, d.p_wc - d.p_wb);
  for (int i = 0; i < 3; ++i) {
    const float r[3] = {m.m[i].x, m.m[i].y, m.m[i].z};
    for (int j = 0; j < 3; ++j) CHECK(r[j] == doctest::Approx(r_bc.m[i][j]).epsilon(1e-6).scale(1.0));
  }
  CHECK(m.m[0].w == doctest::Approx(p_bc.x).epsilon(1e-6).scale(1.0));
  CHECK(m.m[1].w == doctest::Approx(p_bc.y).epsilon(1e-6).scale(1.0));
  CHECK(m.m[2].w == doctest::Approx(p_bc.z).epsilon(1e-6).scale(1.0));
  const Vec3<double> c_cam = mat_t_vec(d.r_wc, d.p_wb - d.p_wc);  // box is centred at its origin
  CHECK(m.sphere.z == doctest::Approx(c_cam.z).epsilon(1e-6));
  CHECK(m.sphere.w == doctest::Approx(std::sqrt(0.01 + 0.0025 + 0.04)).epsilon(1e-5));

  // Light carried by the mover: camera-frame position of its body-frame offset.
  const LightRow& l = light_row(v, slot, 0);
  const Vec3<double> lp = mat_t_vec(d.r_wc, (d.r_wb * Vec3<double>{0.02, 0.03, -0.1} + d.p_wb) - d.p_wc);
  CHECK(l.position[0] == doctest::Approx(lp.x).epsilon(1e-6).scale(1.0));
  CHECK(l.position[2] == doctest::Approx(lp.z).epsilon(1e-6));
  const LightRow& sun = light_row(v, slot, 1);
  const Vec3<double> sd = mat_t_vec(d.r_wc, Vec3<double>{0.0, 0.6, 0.8});
  CHECK(sun.direction[1] == doctest::Approx(sd.y).epsilon(1e-6).scale(1.0));
}

TEST_CASE("relative twists predict the camera-frame motion of body points (M2 D3)") {
  const SceneBundle s = moving_scene();
  const SceneData scene(s, SceneSettings{});
  ThreadPool pool(1);
  const int64_t t = kT0 + 51'234'567, dt = 2'000;  // central differences over +-2 us
  for (int32_t g = 0; g < 2; ++g) {
    // Camera-frame position of a fixed body (or world) point at time t + delta, in fp64.
    auto x_cam = [&](int64_t tt, Vec3<double> x_body) {
      const Direct d = direct(s, tt);
      const Vec3<double> xw = g == 0 ? x_body : d.r_wb * x_body + d.p_wb;
      return mat_t_vec(d.r_wc, xw - d.p_wc);
    };
    PoseTable table(2, 2, 16);
    table.reset(kT0);
    uint32_t slot;
    REQUIRE(table.slot(0, t, &slot));
    table.evaluate(scene.pose_view(), pool);
    const PoseRow& row = pose_row(table.view(), slot, g);
    for (const Vec3<double> xb : {Vec3<double>{0.05, -0.02, 0.1}, Vec3<double>{-0.1, 0.04, -0.15}}) {
      const Vec3<double> xdot_fd = (0.5e9 / dt) * (x_cam(t + dt, xb) - x_cam(t - dt, xb));
      const Vec3<double> x = x_cam(t, xb);
      const Vec3<double> w{row.omega.x, row.omega.y, row.omega.z}, v{row.v.x, row.v.y, row.v.z};
      const Vec3<double> xdot = cross(w, x) + v;
      INFO("geometry " << g);
      CHECK(norm(xdot - xdot_fd) <= 1e-5 * (1.0 + norm(xdot_fd)));
    }
  }
}

TEST_CASE("pose table: slot reuse, re-base and out-of-span times") {
  const SceneBundle s = moving_scene();
  const SceneData scene(s, SceneSettings{});
  ThreadPool pool(2);
  PoseTable table(2, 2, 3);
  table.reset(kT0);
  uint32_t a, b, c;
  REQUIRE(table.slot(0, kT0 + 10, &a));
  REQUIRE(table.slot(0, kT0 + 20, &b));
  CHECK(b != a);
  REQUIRE(table.slot(0, kT0 + 10, &c));
  CHECK(c == a);
  CHECK(table.num_slots() == 2);
  table.evaluate(scene.pose_view(), pool);
  const uint64_t gen = table.generation();
  REQUIRE(table.slot(0, kT1 + 1, &c));  // beyond every span
  CHECK_FALSE(table.slot(0, kT0 + 30, &c));  // capacity 3 reached
  CHECK_THROWS_AS(table.evaluate(scene.pose_view(), pool), std::invalid_argument);
  CHECK(table.num_slots() == 2);  // the failed slot is forgotten, earlier rows stay valid
  REQUIRE(table.slot(0, kT0 + 20, &c));
  CHECK(c == b);
  REQUIRE(table.slot(0, kT0 + 30, &c));  // room again
  table.reset(kT1);
  CHECK(table.generation() == gen + 1);
  CHECK(table.num_slots() == 0);
  // Many keys through the open-addressing index: every key maps back to its own slot.
  PoseTable big(2, 2, 5000);
  big.reset(0);
  for (int round = 0; round < 2; ++round) {
    for (uint32_t i = 0; i < 5000; ++i) {
      REQUIRE(big.slot(static_cast<uint16_t>(i % 3), int64_t{i} * 7919 - 1000000, &c));
      CHECK(c == i);
    }
  }
  big.reset(0);
  REQUIRE(big.slot(1, 42, &c));
  CHECK(c == 0u);
}

TEST_CASE("window clock slots lie in [t_w, t_w + W]; slot 0 is t_w") {
  // begin_window checks [t_w, t_w + W] only, so every slot a window renders must lie there. The fixed-rate
  // grid's inner slots are the multiples of the period inside (t_w, t_w + W) (engine fixed_rate_samples).
  const int64_t t_w = time::window_start(time::window_index(kT0) + 1);  // no multiple of 1 ms
  auto fixed = [&](int64_t period) {
    uint32_t inner = 0;
    for (int64_t k = time::floor_div(t_w, period) + 1; k * period < t_w + time::kWindowNs; ++k) ++inner;
    return WindowClock{t_w, 0, inner + 1, period};
  };
  for (const WindowClock& c : {WindowClock{t_w, 9, 0, 0}, WindowClock{t_w, 10, 0, 0}, fixed(1'000'000), fixed(1 << 20)}) {
    INFO("shift " << c.shift << ", period " << c.period_ns << " ns");
    CHECK(slot_time(c, 0) == t_w);
    CHECK(slot_time(c, num_slots(c) - 1) == t_w + time::kWindowNs);
    for (uint32_t k = 1; k < num_slots(c); ++k) {
      REQUIRE(slot_time(c, k) > slot_time(c, k - 1));
      REQUIRE(slot_time(c, k) <= t_w + time::kWindowNs);
      if (c.period_ns > 0 && k < c.m) CHECK(slot_time(c, k) % c.period_ns == 0);
    }
  }
  CHECK(fixed(1'000'000).m == 10);
}
