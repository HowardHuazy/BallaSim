// P2-F F9a, mover AO by sphere-set proxies (accuracy.mover_ao proxy; render/ao_proxy.h, render/shade/ao.h; plan p2f
// §4.7): the form factor against the plan's quadrature (f9_sphere_ff) and contact checks (r2_ao_contact), its C2
// branches, the window and its omission bound, the fit (a sphere is one sphere, elongated and flat bodies more, the
// result deterministic), the scene's proxy tables and held sets, the host's lane schedule against a serial fp64
// product, and the renderer: off unchanged, the floor beside a sphere darkened by its form factor, receivers beyond the
// window and the sphere itself bit for bit, signatures untouched. Host = device is test_device.cpp's (GPU).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <random>
#include <vector>

#include "core/math/quat.h"
#include "core/math/so3.h"
#include "core/traj/spline_build.h"
#include "render/ao_proxy.h"
#include "render/renderer.h"
#include "render/scene_data.h"
#include "render/shade/ao.h"
#include "render/shade/shade.h"
#include "scenes.h"

using namespace balla;
using namespace balla::render;
using namespace balla::render::test;

namespace {

constexpr int64_t kT0 = 1'000'000'000, kT1 = 1'250'000'000;
constexpr double kPi = std::numbers::pi;

// UV sphere of radius r about the origin (smooth normals), rings every 180 / n_lat degrees.
Mesh uv_sphere(float r, int n_lon, int n_lat) {
  Mesh m;
  for (int i = 1; i < n_lat; ++i) {
    for (int j = 0; j < n_lon; ++j) {
      const double th = kPi * i / n_lat, ph = 2 * kPi * j / n_lon;
      const double d[3] = {std::sin(th) * std::cos(ph), std::cos(th), std::sin(th) * std::sin(ph)};
      for (int a = 0; a < 3; ++a) m.positions.push_back(static_cast<float>(static_cast<double>(r) * d[a]));
      for (int a = 0; a < 3; ++a) m.normals.push_back(static_cast<float>(d[a]));
    }
  }
  const auto top = static_cast<uint32_t>(m.positions.size() / 3), bot = top + 1;
  m.positions.insert(m.positions.end(), {0.0f, r, 0.0f, 0.0f, -r, 0.0f});
  m.normals.insert(m.normals.end(), {0.0f, 1.0f, 0.0f, 0.0f, -1.0f, 0.0f});
  const auto at = [&](int i, int j) { return static_cast<uint32_t>(i * n_lon + (j % n_lon)); };
  for (int i = 0; i + 1 < n_lat - 1; ++i) {
    for (int j = 0; j < n_lon; ++j) {
      m.indices.insert(m.indices.end(), {at(i, j), at(i + 1, j + 1), at(i, j + 1), at(i, j), at(i + 1, j),
                                         at(i + 1, j + 1)});
    }
  }
  for (int j = 0; j < n_lon; ++j) {
    m.indices.insert(m.indices.end(), {top, at(0, j), at(0, j + 1), bot, at(n_lat - 2, j + 1), at(n_lat - 2, j)});
  }
  return m;
}

Mesh shifted(Mesh m, float dx, float dy, float dz) {
  for (size_t v = 0; v < m.positions.size() / 3; ++v) {
    m.positions[3 * v] += dx;
    m.positions[3 * v + 1] += dy;
    m.positions[3 * v + 2] += dz;
  }
  return m;
}

double ff(double h, double phi) { return sphere_form_factor(h - 1.0, std::cos(phi)); }

// A trajectory's position at t as the renderer evaluates it (test_patches.cpp's pose_at).
Vec3<double> position_at(const Trajectory& tr, int64_t t) {
  std::vector<Quat<double>> qs;
  std::vector<Vec3<double>> ps;
  for (size_t c = 0; c < tr.pos.size() / 3; ++c) {
    qs.push_back({tr.rot[4 * c], tr.rot[4 * c + 1], tr.rot[4 * c + 2], tr.rot[4 * c + 3]});
    ps.push_back({tr.pos[3 * c], tr.pos[3 * c + 1], tr.pos[3 * c + 2]});
  }
  const auto nodes = traj::build_spline_nodes(qs, ps, {0, 0, 0});
  const traj::SplineView<double> v{nodes.data(), static_cast<int32_t>(nodes.size()) - tr.order + 1,
                                   static_cast<int32_t>(tr.h_ns), 1e9 / static_cast<double>(tr.h_ns)};
  traj::SegmentTime st{};
  REQUIRE(traj::locate(t, tr.t0_ns, v.h_ns, v.num_segments, &st));
  return traj::evaluate<6>(v, st).p;
}

}  // namespace

TEST_CASE("mover AO (P2-F F9a): the sphere form factor against quadrature and the closed form (f9_sphere_ff)") {
  // (H, phi, the plan's closed form in libm fp64, its 1600 x 3200 polar quadrature over the cap): f9_sphere_ff.json.
  struct Row {
    double h, phi, closed, quad;
  };
  const Row rows[] = {
      {1.05, 0.0, 0.90702947845804982, 0.90702957234973036}, {1.05, 0.6, 0.76264359961476536, 0.76264366099560732},
      {1.05, 1.2, 0.48582177928903963, 0.48582179717147267}, {1.05, 1.5, 0.34148246537948923, 0.34148246135734772},
      {1.05, 1.7, 0.25204684041092351, 0.25204682812682161}, {1.05, 2.0, 0.13694973436276398, 0.13694971796735286},
      {1.5, 0.0, 0.44444444444444442, 0.44444445985250192},  {1.5, 0.6, 0.36681582884874592, 0.36681584156556446},
      {1.5, 1.2, 0.17579972551995984, 0.17579972666866223},  {1.5, 1.5, 0.090621469604363875, 0.090621465231082166},
      {1.5, 1.7, 0.048110081059059701, 0.048110075716660407}, {1.5, 2.0, 0.0097130632527829552, 0.0097130591736127071},
      {3.0, 0.0, 0.1111111111111111, 0.1111111119465359},    {3.0, 0.6, 0.091703957212186479, 0.091703957901692321},
      {3.0, 1.2, 0.040261972719630404, 0.040261973022353063}, {3.0, 1.5, 0.012567640058766982, 0.012567639477223915},
      {3.0, 1.7, 0.0026289475456319766, 0.0026289469925192752}, {3.0, 2.0, 0, 0},
      {10.0, 0.0, 0.01, 0.010000000006532237},               {10.0, 0.6, 0.0082533561490967834, 0.0082533561544880715},
      {10.0, 1.2, 0.0036235775447667363, 0.0036235775471337426},
      {10.0, 1.5, 0.00071825048152310278, 0.00071825047319355978},
      {10.0, 1.7, 0, 0},                                     {10.0, 2.0, 0, 0}};
  double worst_closed = 0, worst_quad = 0, worst_fp32 = 0;
  for (const Row& r : rows) {
    INFO("H = " << r.h << ", phi = " << r.phi);
    const double f = ff(r.h, r.phi);
    worst_closed = std::max(worst_closed, std::abs(f - r.closed));
    worst_quad = std::max(worst_quad, std::abs(f - r.quad));
    const float f32 = sphere_form_factor(static_cast<float>(r.h - 1.0), static_cast<float>(std::cos(r.phi)));
    worst_fp32 = std::max(worst_fp32, std::abs(static_cast<double>(f32) - f));
  }
  MESSAGE("form factor: |F - closed| " << worst_closed << ", |F - quadrature| " << worst_quad << ", |F32 - F64| "
                                       << worst_fp32);
  CHECK(worst_closed <= 1e-14);  // the plan's libm form: pmath's atan2 route agrees to 2.2e-16 (r2_ao_contact)
  CHECK(worst_quad <= 1e-7);     // the plan's 9.4e-8
  CHECK(worst_fp32 <= 1e-6);
  // Over a dense (H, mu) grid, fp32 stays within 1e-6 of fp64 and F within [0, 1 / H_e^2].
  double worst = 0;
  for (double e : {5e-4, 1e-3, 3e-3, 0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 9.0, 40.0}) {
    for (int i = 0; i <= 2000; ++i) {
      const double mu = -1.0 + i / 1000.0;
      const double f = sphere_form_factor(e, mu);
      REQUIRE(std::isfinite(f));
      CHECK(f >= 0.0);
      CHECK(f <= 1.0 / ((1 + e) * (1 + e)) * (1 + 1e-12));
      const float f32 = sphere_form_factor(static_cast<float>(e), static_cast<float>(mu));
      const double e64 = static_cast<double>(static_cast<float>(e)), mu64 = static_cast<double>(static_cast<float>(mu));
      worst = std::max(worst, std::abs(static_cast<double>(f32) - sphere_form_factor(e64, mu64)));
    }
  }
  MESSAGE("fp32 against fp64 on the grid: " << worst);
  CHECK(worst <= 1e-6);
  // Every fp32 ulp of mu within 5000 of both horizon joins (mu = +-1/H_e), against fp64 at the same inputs: no
  // rounded ratio goes through asin or acos where their slope is unbounded (the F9a review measured 1.4e-4 at H = 10
  // for that form; the atan2 form stays within 3e-7), and below the lower join F is exactly 0.
  double worst_join = 0;
  for (const double big_h : {1.0005, 1.001, 1.01, 1.05, 1.5, 3.0, 10.0, 41.0}) {
    const float e = static_cast<float>(big_h - 1.0);
    for (const float sign : {1.0f, -1.0f}) {
      float mu = sign / (1.0f + e);
      for (int k = 0; k < 5000; ++k) mu = std::nextafter(mu, -2.0f);
      for (int k = 0; k < 10000; ++k, mu = std::nextafter(mu, 2.0f)) {
        const double f64 = sphere_form_factor(static_cast<double>(e), static_cast<double>(mu));
        const float f32 = sphere_form_factor(e, mu);
        worst_join = std::max(worst_join, std::abs(static_cast<double>(f32) - f64));
        if ((1.0f + e) * mu <= -1.0f) REQUIRE(f32 == 0.0f);
      }
    }
  }
  MESSAGE("fp32 against fp64 within 5000 ulps of the horizon joins: " << worst_join);
  CHECK(worst_join <= 1e-6);
}

TEST_CASE("mover AO: the contact floor (r2_ao_contact) and C2 at the branches") {
  // (H, H_e, F(H_e, phi = 0)) of r2_ao_contact.json: receivers inside a proxy (H < 1) see contact, finite.
  struct Row {
    double h, he, f0;
  };
  const Row rows[] = {{0.0, 1.0005, 0.99900074950031237},       {0.5, 1.0005, 0.99900074950031237},
                      {0.9, 1.0005, 0.99900074950031237},       {0.999, 1.0005, 0.99900074950031237},
                      {1.0, 1.0005, 0.99900074950031237},       {1.0004, 1.0005512, 0.99889851079491554},
                      {1.0005, 1.00059375, 0.99881355678052819}, {1.001, 1.001, 0.99800299600499431},
                      {1.002, 1.002, 0.99601196807980852},      {1.01, 1.01, 0.98029604940692083},
                      {1.05, 1.05, 0.90702947845804982}};
  std::vector<double> prev;
  for (const Row& r : rows) {
    INFO("H = " << r.h);
    const double e = ao_contact(r.h, 1.0);
    CHECK(1.0 + e == doctest::Approx(r.he).epsilon(1e-15));
    CHECK(sphere_form_factor(e, 1.0) == doctest::Approx(r.f0).epsilon(1e-14));
    std::vector<double> f;
    for (int i = 0; i <= 180; ++i) f.push_back(sphere_form_factor(e, std::cos(kPi * i / 180)));
    for (const double v : f) REQUIRE(std::isfinite(v));
    if (!prev.empty()) {
      for (size_t i = 0; i < f.size(); ++i) CHECK(f[i] <= prev[i] + 1e-12);  // non-increasing in H
    }
    prev = f;
  }
  // H_e is monotone and C1 at the floor's joins (H - 1 = 0 and delta); q's C2 is bsdf.h's (test_bsdf.cpp).
  for (const double x0 : {0.0, kAoContactDelta}) {
    const double h = 1e-7;
    const double dl = (ao_contact(1.0 + x0, 1.0) - ao_contact(1.0 + x0 - h, 1.0)) / h;
    const double dr = (ao_contact(1.0 + x0 + h, 1.0) - ao_contact(1.0 + x0, 1.0)) / h;
    CHECK(std::abs(dl - dr) < 1e-5);
  }
  // Across both horizon joins (mu = +-1/H_e) F is continuous with its slope, and the one-sided second differences
  // close as h shrinks (the plan's F'' gap 0.089 / 0.047 / 0.027 at H = 1.5: proportional to h^(1/2)).
  for (const double big_h : {1.5, 3.0}) {
    for (const double sign : {1.0, -1.0}) {
      const double phi0 = kPi / 2 - sign * std::asin(1 / big_h);
      double last_gap = 1e9;
      for (const double h : {1e-2, 3e-3, 1e-3}) {
        double f[5];
        for (int k = -2; k <= 2; ++k) f[k + 2] = ff(big_h, phi0 + k * h);
        const double d1l = (f[2] - f[1]) / h, d1r = (f[3] - f[2]) / h;
        const double d2l = (f[2] - 2 * f[1] + f[0]) / (h * h), d2r = (f[4] - 2 * f[3] + f[2]) / (h * h);
        INFO("H = " << big_h << ", join " << sign << ", h = " << h << ": slopes " << d1l << " / " << d1r
                    << ", second differences " << d2l << " / " << d2r);
        CHECK(std::abs(d1l - d1r) < 3 * h);
        const double gap = std::abs(d2l - d2r);
        CHECK(gap < last_gap);
        last_gap = gap;
      }
      CHECK(last_gap < 0.035);
    }
  }
}

TEST_CASE("mover AO: the window, its C2 joins and the omission bound (r2_ao_contact)") {
  CHECK(ao_window(0.5, 1.0) == 1.0);
  CHECK(ao_window(1.0, 1.0) == 1.0);
  CHECK(ao_window(1.5, 1.0) == doctest::Approx(0.5).epsilon(1e-15));
  CHECK(ao_window(2.0, 1.0) == 0.0);
  CHECK(ao_window(3.0, 1.0) == 0.0);
  // The omitted occlusion sum_i (1 - w_i) F_i of random sphere sets (k <= 8) from receivers at least d0 from one
  // centre stays below eps_AO per body (the plan measured 0.29 eps_AO over 20,000 sets).
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::normal_distribution<double> g(0.0, 1.0);
  double worst = 0;
  for (int trial = 0; trial < 20000; ++trial) {
    const int k = 1 + static_cast<int>(u01(rng) * 8) % 8;
    std::vector<double> r(static_cast<size_t>(k));
    std::vector<std::array<double, 3>> c(static_cast<size_t>(k));
    double rb2 = 0;
    for (int i = 0; i < k; ++i) {
      r[static_cast<size_t>(i)] = (0.2 + 0.8 * u01(rng)) * 0.05;
      for (double& x : c[static_cast<size_t>(i)]) x = 0.05 * g(rng);
      rb2 += r[static_cast<size_t>(i)] * r[static_cast<size_t>(i)];
    }
    const double d0 = std::sqrt(rb2) * kAoWindowScale;
    double dir[3] = {g(rng), g(rng), g(rng)}, n[3] = {g(rng), g(rng), g(rng)};
    const double ld = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    const double ln = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    const double dist = d0 * (1.0 + 1.2 * u01(rng));
    const auto& c0 = c[static_cast<size_t>(u01(rng) * k) % static_cast<size_t>(k)];
    const double x[3] = {c0[0] + dist * dir[0] / ld, c0[1] + dist * dir[1] / ld, c0[2] + dist * dir[2] / ld};
    double omit = 0;
    for (int i = 0; i < k; ++i) {
      const auto& ci = c[static_cast<size_t>(i)];
      const double v[3] = {ci[0] - x[0], ci[1] - x[1], ci[2] - x[2]};
      const double d = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      const double mu = (n[0] * v[0] + n[1] * v[1] + n[2] * v[2]) / (ln * d);
      omit += (1.0 - ao_window(d, d0)) * sphere_form_factor(ao_contact(d, r[static_cast<size_t>(i)]), mu);
    }
    worst = std::max(worst, omit);
  }
  MESSAGE("window omission: worst " << worst / kAoEps << " eps_AO");
  CHECK(worst <= kAoEps);
  CHECK(kAoWindowScale == doctest::Approx(1.0 / std::sqrt(kAoEps)).epsilon(1e-15));
}

TEST_CASE("mover AO: proxy fits (a sphere, an elongated box, a flat card, no area) and their determinism") {
  {  // A UV sphere is one sphere at its centre with its radius (up to the facets' sag).
    const Mesh m = shifted(uv_sphere(0.05f, 32, 16), 0.1f, -0.2f, 0.3f);
    const AoProxy p = fit_ao_proxy(m.positions, m.indices);
    REQUIRE(p.spheres.size() == 1);
    const auto& s = p.spheres[0];
    MESSAGE("UV sphere 32 x 16: centre off by " << std::hypot(s.centre[0] - 0.1, s.centre[1] + 0.2, s.centre[2] - 0.3)
                                                << " m, radius " << s.radius / 0.05 << " R");
    CHECK(std::hypot(s.centre[0] - 0.1, s.centre[1] + 0.2, s.centre[2] - 0.3) < 5e-3 * 0.05);
    CHECK(s.radius / 0.05 == doctest::Approx(1.0).epsilon(0.01));
    CHECK(p.radius_b == s.radius);
    CHECK(p.d0 == p.radius_b * kAoWindowScale);
    const AoProxy q = fit_ao_proxy(m.positions, m.indices);  // deterministic: the same bits
    CHECK(std::memcmp(q.spheres.data(), p.spheres.data(), sizeof(AoProxySphere)) == 0);
  }
  {  // A 4:1:1 box: spheres strung along its long axis, symmetric about its centre. Its two minor axes are degenerate,
     // so the PCA basis between them (and with it e_min, 0.1 to 0.14) is the solver's: k = ceil(4 or 2.8 - 0.1).
    const Mesh m = box_mesh(0.2f, 0.05f, 0.05f);
    const AoProxy p = fit_ao_proxy(m.positions, m.indices);
    MESSAGE("4:1:1 box: k = " << p.spheres.size());
    REQUIRE(p.spheres.size() >= 3);
    REQUIRE(p.spheres.size() <= 4);
    std::vector<double> xs;
    double rb2 = 0;
    for (const auto& s : p.spheres) {
      CHECK(std::abs(s.centre[1]) < 2e-3);
      CHECK(std::abs(s.centre[2]) < 2e-3);
      CHECK(s.radius > 0.04);
      CHECK(s.radius < 0.1);
      xs.push_back(s.centre[0]);
      rb2 += s.radius * s.radius;
    }
    std::sort(xs.begin(), xs.end());
    CHECK(xs.front() == doctest::Approx(-xs.back()).epsilon(0.02));
    CHECK(xs.front() < -0.1);
    CHECK(p.radius_b == doctest::Approx(std::sqrt(rb2)).epsilon(1e-15));
  }
  {  // A flat card (e_min = 0) and a 4:1:0.5 box (distinct axes, ratio 8): the most spheres.
    const Mesh m = plane_mesh(0.1, 4);
    CHECK(fit_ao_proxy(m.positions, m.indices).spheres.size() == static_cast<size_t>(kAoMaxSpheres));
    const Mesh b = box_mesh(0.2f, 0.05f, 0.025f);
    CHECK(fit_ao_proxy(b.positions, b.indices).spheres.size() == static_cast<size_t>(kAoMaxSpheres));
  }
  {  // No area: no proxy.
    const std::vector<float> pos{0, 0, 0, 1, 0, 0, 2, 0, 0};
    const std::vector<uint32_t> idx{0, 1, 2};
    CHECK(fit_ao_proxy(pos, idx).spheres.empty());
    CHECK(fit_ao_proxy(pos, {}).spheres.empty());
  }
}

namespace {

// A floor (static, facing the camera at z = 2 m) and a UV sphere mover of radius r whose centre passes `gap` in
// front of it, ambient light only; plus, optionally, a 4:1 box mover.
SceneBundle ao_scene(float r, double gap, bool box) {
  SceneBundle s;
  s.light_groups = {"default"};
  const Quat<double> id = quat_identity<double>();
  s.trajectories.push_back(static_trajectory(kT0, kT1, id, {0, 0, 0}));
  s.trajectories.push_back(moving_trajectory(kT0, kT1, id, {-0.01, 0.003, 2.0 - gap}, {0.02, 0.0, 0.0}, {0, 0, 0}));
  s.trajectories.push_back(moving_trajectory(kT0, kT1, so3::exp(Vec3<double>{0, 0, 0.3}), {0.5, 0.3, 1.8},
                                             {0, 0.05, 0}, {0, 0, 0.2}));
  s.cameras.push_back(pinhole(64, 48, 64.0, 0));
  Body floor = make_body(1, plane_mesh(3.0, 6), {0.8f, 0.8f, 0.8f});
  floor.T_world_body = translation(0, 0, 2.0);
  s.bodies.push_back(floor);
  Body ball = make_body(7, uv_sphere(r, 32, 16), {0.5f, 0.5f, 0.5f});
  ball.trajectory = 1;
  s.bodies.push_back(ball);
  if (box) {
    Body b = make_body(9, box_mesh(0.2f, 0.05f, 0.05f), {0.4f, 0.4f, 0.4f});
    b.trajectory = 2;
    s.bodies.push_back(b);
  }
  s.lights.push_back(ambient_light(0.5f));
  return s;
}

}  // namespace

TEST_CASE("mover AO: the scene's proxy tables, held sets and the views") {
  SceneSettings off, proxy;
  proxy.mover_ao = MoverAo::kProxy;
  const SceneBundle s = ao_scene(0.05f, 0.15, true);
  const SceneData d_off(s, off), d(s, proxy);
  CHECK(d_off.ao_bodies().empty());
  CHECK(d_off.ao_spheres().empty());
  CHECK(d_off.host_view().mover_ao == 0);
  CHECK(d_off.host_view().ao_spheres == nullptr);
  REQUIRE(d.ao_bodies().size() == 2);
  CHECK(d.ao_bodies()[0].instance == 7);
  CHECK(d.ao_bodies()[0].geometry == 1);
  CHECK(d.ao_bodies()[1].instance == 9);
  CHECK(d.ao_bodies()[1].geometry == 2);
  const size_t k_box = fit_ao_proxy(s.bodies[2].mesh.positions, s.bodies[2].mesh.indices).spheres.size();
  REQUIRE(d.ao_spheres().size() == 1 + k_box);  // the sphere's 1, the box's
  for (size_t i = 0; i < d.ao_spheres().size(); ++i) CHECK(d.ao_spheres()[i].body == (i == 0 ? 0 : 1));
  const AoProxy ball = fit_ao_proxy(s.bodies[1].mesh.positions, s.bodies[1].mesh.indices);
  CHECK(d.ao_bodies()[0].d0 == static_cast<float>(ball.d0));
  CHECK(d.ao_spheres()[0].radius == static_cast<float>(ball.spheres[0].radius));
  const SceneView v = d.host_view();
  CHECK(v.mover_ao == 1);
  CHECK(v.num_ao_spheres == static_cast<int32_t>(1 + k_box));
  CHECK(v.ao_bodies == d.ao_bodies().data());
  // Holding the box: its spheres move to the world frame at its pose, posed by the world's rows (geometry 0); the
  // sphere keeps its own trace geometry (1 of the set).
  const RigidPose pose{to_matrix(so3::exp(Vec3<double>{0.1, 0.2, 0.3})), {0.4, -0.1, 1.7}};
  const int32_t held[] = {2};
  const auto t = d.trace_set(held, std::span<const RigidPose>(&pose, 1));
  REQUIRE(t->ao_bodies.size() == 2);
  CHECK(t->ao_bodies[0].geometry == 1);
  CHECK(t->ao_bodies[1].geometry == 0);
  CHECK(t->ao_spheres[0].centre[0] == d.ao_spheres()[0].centre[0]);
  for (size_t i = 1; i < 1 + k_box; ++i) {
    const auto& c = d.ao_spheres()[i].centre;
    const Vec3<double> w = pose.r * Vec3<double>{c[0], c[1], c[2]} + pose.p;
    CHECK(t->ao_spheres[i].centre[0] == doctest::Approx(w.x).epsilon(1e-6));
    CHECK(t->ao_spheres[i].centre[1] == doctest::Approx(w.y).epsilon(1e-6));
    CHECK(t->ao_spheres[i].centre[2] == doctest::Approx(w.z).epsilon(1e-6));
    CHECK(t->ao_spheres[i].radius == d.ao_spheres()[i].radius);
  }
}

TEST_CASE("mover AO: the host's lane schedule against a serial fp64 product; own body, range and passes") {
  // 8 bodies of 2..8 spheres (40: two passes), posed by identity rows; random receivers near them.
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  std::vector<AoBodyView> bodies;
  std::vector<AoSphere> spheres;
  const int ks[8] = {8, 7, 6, 5, 4, 3, 2, 5};
  for (int b = 0; b < 8; ++b) {
    const int k = ks[b];
    bodies.push_back({100u + static_cast<uint32_t>(b), b % 3, 0.0f});
    double rb2 = 0;
    for (int i = 0; i < k; ++i) {
      const float r = 0.02f + 0.01f * (u(rng) + 1.0f);
      spheres.push_back({{0.3f * u(rng), 0.3f * u(rng), 1.0f + 0.3f * u(rng)}, r, b});
      rb2 += static_cast<double>(r) * static_cast<double>(r);
    }
    bodies.back().d0 = static_cast<float>(std::sqrt(rb2) * kAoWindowScale);
  }
  REQUIRE(spheres.size() == 40);
  SceneView s{};
  s.mover_ao = 1;
  s.num_ao_spheres = static_cast<int32_t>(spheres.size());
  s.ao_bodies = bodies.data();
  s.ao_spheres = spheres.data();
  PoseRow identity{};
  identity.m[0] = {1, 0, 0, 0};
  identity.m[1] = {0, 1, 0, 0};
  identity.m[2] = {0, 0, 1, 0};
  const std::vector<PoseRow> rows(3, identity);
  const PoseTableView poses{rows.data(), nullptr, 3, 0};
  const auto serial = [&](uint32_t own, Vec3f x, Vec3f n) {
    double vis = 1;
    for (const AoSphere& sp : spheres) {
      const AoBodyView& b = bodies[static_cast<size_t>(sp.body)];
      if (b.instance == own) continue;
      const auto dbl = [](float f) { return static_cast<double>(f); };
      const double v[3] = {dbl(sp.centre[0]) - dbl(x.x), dbl(sp.centre[1]) - dbl(x.y), dbl(sp.centre[2]) - dbl(x.z)};
      const double d = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      const double mu = (dbl(n.x) * v[0] + dbl(n.y) * v[1] + dbl(n.z) * v[2]) / d;
      vis *= 1.0 - ao_window(d, static_cast<double>(b.d0)) *
                       sphere_form_factor(ao_contact(d, static_cast<double>(sp.radius)), mu);
    }
    return vis;
  };
  double worst = 0, min_vis = 1;
  for (int i = 0; i < 2000; ++i) {
    const Vec3f x{0.4f * u(rng), 0.4f * u(rng), 1.0f + 0.4f * u(rng)};
    const Vec3f n = normalize(Vec3f{u(rng), u(rng), u(rng)});
    const uint32_t own = i % 4 == 0 ? 103u : 0u;
    const float vis = ao_visibility_host(s, poses, 0, own, x, n);
    const double ref = serial(own, x, n);
    worst = std::max(worst, std::abs(static_cast<double>(vis) - ref));
    min_vis = std::min(min_vis, ref);
    CHECK(vis == ao_visibility_host(s, poses, 0, own, x, n));
  }
  MESSAGE("lanes against fp64: worst |1 - O| difference " << worst << " (smallest 1 - O " << min_vis << ")");
  CHECK(worst <= 2e-6);  // 40 fp32 factors, each within the form factor's 3e-7
  CHECK(min_vis < 0.5);
  // Beyond every body's 2 d0: exactly 1.
  CHECK(ao_visibility_host(s, poses, 0, 0, Vec3f{50.0f, 0.0f, 1.0f}, Vec3f{0.0f, 0.0f, 1.0f}) == 1.0f);
  // Every sphere its own body's: exactly 1.
  std::vector<AoBodyView> one = bodies;
  for (AoBodyView& b : one) b.instance = 5;
  s.ao_bodies = one.data();
  CHECK(ao_visibility_host(s, poses, 0, 5, Vec3f{0.0f, 0.0f, 1.0f}, Vec3f{0.0f, 0.0f, -1.0f}) == 1.0f);
  // The butterfly product is the warp's: every lane holds the same value.
  float v[kLanes];
  for (uint32_t l = 0; l < kLanes; ++l) v[l] = 1.0f - 0.01f * static_cast<float>(l % 7);
  const float p = butterfly_product(v);
  for (uint32_t l = 0; l < kLanes; ++l) CHECK(v[l] == p);
}

TEST_CASE("mover AO: the renderer under proxy against off (Embree; OptiX when present)") {
  // Two spheres passing in front of the floor: R = 8 cm at 20 cm (its own pixels, deep occlusion beside it) and R = 2 cm
  // at 5 cm (2 d0 = 0.89 m: the frame's corners lie beyond the window). Each floor pixel's ratio proxy / off is 1 - F
  // at its cluster's mean point, compared with 1 - F at the pixel centre's floor point (fp64, the fitted sphere).
  constexpr uint32_t kW = 64, kH = 48;
  const int64_t t = kT0 + 100'000'000;
  std::vector<uint16_t> cam(kW * kH, 0);
  std::vector<uint32_t> pix(kW * kH);
  std::vector<int64_t> times(kW * kH, t);
  for (uint32_t p = 0; p < kW * kH; ++p) pix[p] = p;
  struct Config {
    float r;
    double gap;
  };
  for (const Config& c : {Config{0.08f, 0.2}, Config{0.02f, 0.05}}) {
    const SceneBundle s = ao_scene(c.r, c.gap, false);
    const AoProxy proxy = fit_ao_proxy(s.bodies[1].mesh.positions, s.bodies[1].mesh.indices);
    REQUIRE(proxy.spheres.size() == 1);
    const Vec3<double> at = position_at(s.trajectories[1], t);  // the sphere translates without turning
    const double cx = at.x + proxy.spheres[0].centre[0], cy = at.y + proxy.spheres[0].centre[1];
    const double cz = at.z + proxy.spheres[0].centre[2];
    for (BackendKind backend : backends()) {
      INFO(backend_name(backend) << ", R = " << c.r);
      RendererOptions o = options(backend, 4);
      const RenderFlags flags{true, false};
      Renderer r_off(s, o);
      o.scene.mover_ao = MoverAo::kProxy;
      Renderer r_ao(s, o);
      const RenderResult a = r_off.render(cam, pix, times, flags), b = r_ao.render(cam, pix, times, flags);
      CHECK(a.signature == b.signature);  // AO enters no signature
      const uint32_t n = o.scene.n_subsamples;
      size_t floor_px = 0, near = 0, far = 0, ball = 0;
      double worst = 0;
      for (uint32_t p = 0; p < kW * kH; ++p) {
        bool all_floor = true, all_ball = true;
        for (uint32_t k = 0; k < n; ++k) {
          all_floor = all_floor && a.sub_instance[p * n + k] == 1;
          all_ball = all_ball && a.sub_instance[p * n + k] == 7;
        }
        if (all_ball) {
          ++ball;
          CHECK(a.irradiance[p] == b.irradiance[p]);  // its own proxy does not occlude it
        }
        if (!all_floor) continue;
        ++floor_px;
        const double px = (static_cast<double>(p % kW) - 0.5 * (kW - 1)) / 64.0 * 2.0;
        const double py = (static_cast<double>(p / kW) - 0.5 * (kH - 1)) / 64.0 * 2.0;
        const double v[3] = {cx - px, cy - py, cz - 2.0};
        const double d = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (d > 2 * proxy.d0 + 0.07) {  // beyond the window by more than a pixel's reach: bit for bit
          ++far;
          CHECK(a.irradiance[p] == b.irradiance[p]);
          continue;
        }
        const double f = sphere_form_factor(ao_contact(d, proxy.spheres[0].radius), -v[2] / d);
        const double expected = 1.0 - ao_window(d, proxy.d0) * f;
        const double ratio = static_cast<double>(b.irradiance[p]) / static_cast<double>(a.irradiance[p]);
        worst = std::max(worst, std::abs(ratio - expected));
        near += expected < 0.97;
      }
      MESSAGE("R = " << c.r << ": floor pixels " << floor_px << ", " << near << " occluded by > 3 %, " << far
                     << " beyond the window, " << ball << " on the sphere; |ratio - (1 - F(pixel centre))| <= "
                     << worst);
      CHECK(near > 8);
      CHECK(worst < 5e-4);
      if (c.r > 0.05f) {
        CHECK(ball > 0);
      } else {
        CHECK(far > 100);
      }
      // Holding the sphere: its spheres posed in the world frame give the same values (fp32 posing aside).
      const uint32_t ids[] = {7};
      REQUIRE(r_ao.hold(ids, t));
      const RenderResult h = r_ao.render(cam, pix, times, flags);
      double held = 0;
      for (uint32_t p = 0; p < kW * kH; ++p) {
        held = std::max(held,
                        std::abs(static_cast<double>(h.irradiance[p]) / static_cast<double>(b.irradiance[p]) - 1));
      }
      MESSAGE("held against moving: " << held);
      CHECK(held < 1e-5);
    }
  }
}

// P2-F F9b: the traced form (accuracy.mover_ao traced; render/shade/ao_traced.h).
TEST_CASE("mover AO traced (P2-F F9b): directions, frame, open share and the scene's reach table") {
  // Directions: unit, above the plane, the first 64 / 256 Sobol points one per cell of the 8 x 8 / 16 x 16 grid (a
  // stratified, nested set), cosine-distributed (E[cos] = 2/3); the frame orthonormal and right-handed for any unit n.
  for (const uint32_t pattern : {0u, 7u, 0xDEADBEEFu}) {
    const uint32_t seed = ao_direction_seed(pattern);
    for (const uint32_t g : {8u, 16u}) {
      std::vector<int> cells(g * g, 0);
      for (uint32_t k = 0; k < g * g; ++k) {
        uint32_t ux, uy;
        rng::owen_sobol_2d(k, seed, &ux, &uy);
        ++cells[(uy >> 24) * g / 256 * g + (ux >> 24) * g / 256];
      }
      CHECK(std::all_of(cells.begin(), cells.end(), [](int c) { return c == 1; }));
    }
    double mz = 0, mx = 0, my = 0;
    for (uint32_t k = 0; k < 4096; ++k) {
      const Vec3f w = ao_local_direction(seed, k);
      CHECK(std::abs(dot(w, w) - 1.0f) < 2e-6f);
      CHECK(w.z >= 0.0f);
      mx += static_cast<double>(w.x);
      my += static_cast<double>(w.y);
      mz += static_cast<double>(w.z);
    }
    CHECK(std::abs(mz / 4096 - 2.0 / 3.0) < 2e-3);
    CHECK(std::abs(mx / 4096) < 5e-3);
    CHECK(std::abs(my / 4096) < 5e-3);
  }
  std::mt19937 rng(3);
  std::normal_distribution<float> g;
  std::vector<Vec3f> normals{{0, 0, 1}, {0, 0, -1}, {1, 0, 0}, {0, -1, 0}, {0.6f, 0.0f, -0.8f}};
  for (int i = 0; i < 1000; ++i) normals.push_back(normalize(Vec3f{g(rng), g(rng), g(rng)}));
  for (const Vec3f& n : normals) {
    Vec3f t, b;
    ao_frame(n, &t, &b);
    CHECK(std::abs(dot(t, t) - 1.0f) < 1e-5f);
    CHECK(std::abs(dot(b, b) - 1.0f) < 1e-5f);
    CHECK(std::abs(dot(t, b)) < 1e-5f);
    CHECK(std::abs(dot(t, n)) < 1e-5f);
    CHECK(std::abs(dot(b, n)) < 1e-5f);
    const Vec3f c = cross(t, b);
    CHECK(std::abs(c.x - n.x) + std::abs(c.y - n.y) + std::abs(c.z - n.z) < 3e-5f);
  }
  CHECK(ao_traced_visibility(64, 0) == 1.0f);
  CHECK(ao_traced_visibility(64, 16) == 0.75f);
  CHECK(ao_traced_visibility(3, 3) == 0.0f);
  // The reach table: per trace geometry 2 d0 of the mover's proxy, 0 for the world; no spheres; off and proxy none.
  const SceneBundle s = ao_scene(0.05f, 0.15, true);
  SceneSettings traced, proxy;
  traced.mover_ao = MoverAo::kTraced;
  proxy.mover_ao = MoverAo::kProxy;
  const SceneData d(s, traced), d_proxy(s, proxy), d_off(s, SceneSettings{});
  REQUIRE(d.ao_reach().size() == 3);
  CHECK(d.ao_reach()[0] == 0.0f);
  for (const size_t i : {size_t{1}, size_t{2}}) {
    const double d0 = fit_ao_proxy(s.bodies[i].mesh.positions, s.bodies[i].mesh.indices).d0;
    CHECK(d.ao_reach()[i] == static_cast<float>(2 * d0));
  }
  CHECK(d.traces_ao());
  CHECK(d.host_view().mover_ao == kMoverAoTraced);
  CHECK(d.host_view().ao_reach == d.ao_reach().data());
  CHECK(d.host_view().num_ao_spheres == 0);
  CHECK(!d_proxy.traces_ao());
  CHECK(d_proxy.host_view().ao_reach == nullptr);
  CHECK(d_off.host_view().ao_reach == nullptr);
  // The AO bit is bit 31 of the occlusion word: 31 shadow lights at most.
  SceneBundle lit = s;
  for (int i = 0; i < kMaxShadowLights; ++i) lit.lights.push_back(point_light({0.1 * i, -1.0, 0.5}, 0.1f));
  CHECK_THROWS_AS(SceneData(lit, traced), std::invalid_argument);
  lit.lights.pop_back();
  CHECK_NOTHROW(SceneData(lit, traced));
}

TEST_CASE("mover AO traced: the renderer against off and the sphere's form factor (Embree; OptiX when present)") {
  // ao_scene's two spheres (R = 8 cm at 20 cm, R = 2 cm at 5 cm), N = 256: each floor pixel's ratio traced / off is its
  // members' open share (a multiple of 1/256), against 1 - F at the pixel centre's floor point (the fitted sphere;
  // the mesh is a 32 x 16 UV sphere). The error is the 256-direction estimator's (F9b's study: |dO| p99 0.028 for a
  // sphere) and unbiased; pixels beyond 2 d0 and the sphere's own pixels keep off's values and signatures bit for bit;
  // some signatures change (the AO bit is in them). Holding a mover is refused.
  constexpr uint32_t kW = 64, kH = 48;
  const int64_t t = kT0 + 100'000'000;
  std::vector<uint16_t> cam(kW * kH, 0);
  std::vector<uint32_t> pix(kW * kH);
  std::vector<int64_t> times(kW * kH, t);
  for (uint32_t p = 0; p < kW * kH; ++p) pix[p] = p;
  struct Config {
    float r;
    double gap;
  };
  std::vector<std::vector<float>> by_backend[2];
  for (const Config& c : {Config{0.08f, 0.2}, Config{0.02f, 0.05}}) {
    const SceneBundle s = ao_scene(c.r, c.gap, false);
    const AoProxy proxy = fit_ao_proxy(s.bodies[1].mesh.positions, s.bodies[1].mesh.indices);
    REQUIRE(proxy.spheres.size() == 1);
    const Vec3<double> at = position_at(s.trajectories[1], t);
    const double cx = at.x + proxy.spheres[0].centre[0], cy = at.y + proxy.spheres[0].centre[1];
    const double cz = at.z + proxy.spheres[0].centre[2];
    for (BackendKind backend : backends()) {
      INFO(backend_name(backend) << ", R = " << c.r);
      RendererOptions o = options(backend, 4);
      o.scene.n_subsamples = 256;
      const RenderFlags flags{true, false};
      Renderer r_off(s, o);
      o.scene.mover_ao = MoverAo::kTraced;
      Renderer r_tr(s, o);
      const RenderResult a = r_off.render(cam, pix, times, flags), b = r_tr.render(cam, pix, times, flags);
      by_backend[backend == BackendKind::kOptix ? 1 : 0].push_back(b.irradiance);
      const uint32_t n = o.scene.n_subsamples;
      size_t floor_px = 0, near = 0, far = 0, ball = 0, changed = 0;
      double sum = 0, worst = 0;
      std::vector<double> err;
      for (uint32_t p = 0; p < kW * kH; ++p) {
        bool all_floor = true, all_ball = true;
        for (uint32_t k = 0; k < n; ++k) {
          all_floor = all_floor && a.sub_instance[p * n + k] == 1;
          all_ball = all_ball && a.sub_instance[p * n + k] == 7;
        }
        changed += a.signature[p] != b.signature[p] ? 1 : 0;
        if (all_ball) {
          ++ball;
          CHECK(a.irradiance[p] == b.irradiance[p]);  // a body occludes none of its own ambient
          CHECK(a.signature[p] == b.signature[p]);
        }
        if (!all_floor) continue;
        ++floor_px;
        const double px = (static_cast<double>(p % kW) - 0.5 * (kW - 1)) / 64.0 * 2.0;
        const double py = (static_cast<double>(p / kW) - 0.5 * (kH - 1)) / 64.0 * 2.0;
        const double v[3] = {cx - px, cy - py, cz - 2.0};
        const double d = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (d > 2 * proxy.d0 + proxy.spheres[0].radius + 0.07) {  // no ray reaches the sphere: bit for bit
          ++far;
          CHECK(a.irradiance[p] == b.irradiance[p]);
          CHECK(a.signature[p] == b.signature[p]);
          continue;
        }
        const double expected = sphere_form_factor(ao_contact(d, proxy.spheres[0].radius), -v[2] / d);
        const double o_px = 1.0 - static_cast<double>(b.irradiance[p]) / static_cast<double>(a.irradiance[p]);
        err.push_back(std::abs(o_px - expected));
        sum += o_px - expected;
        worst = std::max(worst, err.back());
        near += expected > 0.03;
      }
      std::sort(err.begin(), err.end());
      const double p99 = err.empty() ? 0 : err[err.size() * 99 / 100];
      MESSAGE("R = " << c.r << ": floor pixels " << floor_px << ", " << near << " occluded by > 3 %, " << far
                     << " beyond the reach, " << ball << " on the sphere, " << changed
                     << " signatures changed; |O - F| p99 " << p99 << ", max " << worst << ", mean signed "
                     << sum / std::max<size_t>(1, err.size()));
      CHECK(near > 4);
      CHECK(changed > 0);
      CHECK(p99 < 0.05);
      CHECK(std::abs(sum / std::max<size_t>(1, err.size())) < 0.006);
      if (c.r > 0.05f) {
        CHECK(ball > 0);
      } else {
        CHECK(far > 100);
      }
      const uint32_t ids[] = {7};
      CHECK_THROWS_AS(r_tr.hold(ids, t), std::invalid_argument);
      CHECK_NOTHROW(r_tr.hold({}, t));
    }
  }
  if (by_backend[1].size() == by_backend[0].size() && !by_backend[1].empty()) {
    // OptiX and Embree trace the same rays and differ only in triangle intersection arithmetic: the open shares agree
    // on all but a few pixels (rays grazing a silhouette).
    for (size_t c = 0; c < by_backend[0].size(); ++c) {
      size_t differ = 0;
      for (size_t p = 0; p < by_backend[0][c].size(); ++p) {
        differ += std::abs(by_backend[0][c][p] - by_backend[1][c][p]) > 1e-5f * by_backend[0][c][p] ? 1 : 0;
      }
      MESSAGE("config " << c << ": OptiX against Embree, " << differ << " of " << by_backend[0][c].size()
                        << " pixels differ");
      CHECK(differ <= by_backend[0][c].size() / 50);
    }
  }
}
