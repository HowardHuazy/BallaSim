// P2-D D1 (F021; ballasim_out/p2_plans/p2d.md §3.7, gate G-D8 (a) and (b)): the planner's shadow-edge speed bound
// with the receivers' obliquity factors (plan.h obliquity, PixelPlan under PlanParams::obliquity) against the image
// speed of the exact shadow point, a port of study M4 / M4b part A (docs/design/phase2-checks/p2_d_m4_f021.py).
//
// Configurations (10^5 per family and motion case, seeded): a point light L, static or moving; an occluder point P on
// the segment from L to the receiver point X at lambda = |X - L| / |P - L| in [1, 30], moving; a receiver plane through
// X with unit normal n, static or moving and turning about S; a static camera C. Families: random, EVIMO2-like (a table
// under a light 2-3.5 m up), a desk lamp, walls (vertical receivers under an overhead light). X and n are snapped to
// what the plan pass hands the pixel stage (a float depth, a float normal), so the bound is evaluated exactly where the
// truth is. The truth: the shadow point X(t), the line L(t) P(t) met with the plane(t), differentiated by a Richardson
// central difference in long double; its image angular speed |P_view dX/dt| / |X - C|.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "engine/plan.h"
#include "plan_cases.h"
#include "scenes.h"

using namespace balla;
using namespace balla::engine;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
using LD = long double;
struct VL {
  LD x, y, z;
};
VL operator+(VL a, VL b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
VL operator-(VL a, VL b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
VL operator*(LD s, VL a) { return {s * a.x, s * a.y, s * a.z}; }
LD dotl(VL a, VL b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
VL crossl(VL a, VL b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
VL ld(Vec3d v) { return {v.x, v.y, v.z}; }

Vec3d unit(Vec3d v) { return (1.0 / norm(v)) * v; }

// v rotated by the angle |w| t about w (Rodrigues).
VL rotate(VL v, Vec3d w, LD t) {
  const LD th = std::sqrt(static_cast<LD>(dot(w, w)));
  if (th == 0) return v;
  const VL k = (1 / th) * ld(w);
  const LD a = th * t, c = std::cos(a), s = std::sin(a);
  return c * v + s * crossl(k, v) + (dotl(k, v) * (1 - c)) * k;
}

struct Config {
  Vec3d c, x, n, l, p;   // camera, receiver point, its normal (facing the light and the camera), light, occluder point
  Vec3d vp, vl, vr, wr, s;  // occluder, light and receiver velocities, the receiver's spin about s
  float depth;           // the plan pass's camera z of X (the camera frame is the world's)
  Vec3d ray;
  float normal[3];
  bool ok;
};

// One configuration of a family (p2_d_m4_f021.py config() and part_a()), snapped to the plan pass's float depth and
// normal.
Config make(const char* family, std::mt19937_64& rng, bool moving_light, bool moving_recv) {
  std::normal_distribution<double> gauss;
  const auto u = [&rng](double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); };
  const auto g3 = [&] { return Vec3d{gauss(rng), gauss(rng), gauss(rng)}; };
  Config k{};
  const std::string f = family;
  if (f == "random") {
    k.c = 0.3 * g3();
    k.x = 0.5 * g3() + Vec3d{0, 0, 1.5};
    k.n = unit(g3());
    k.l = k.x + u(0.3, 4.0) * unit(g3());
  } else if (f == "evimo2") {
    k.x = {u(-0.4, 0.4), u(-0.4, 0.4), 0.0};
    k.n = {0, 0, 1};
    k.l = {u(-0.5, 0.5), u(-0.5, 0.5), u(2.0, 3.5)};
    const double el = u(30, 60) * std::numbers::pi / 180, az = u(0, 2 * std::numbers::pi), r = u(0.5, 1.0);
    k.c = {r * std::cos(el) * std::cos(az), r * std::cos(el) * std::sin(az), r * std::sin(el)};
  } else if (f == "lamp") {
    k.x = {u(-0.5, 0.5), u(-0.5, 0.5), 0.0};
    k.n = {0, 0, 1};
    const double az = u(0, 2 * std::numbers::pi), dist = u(0.1, 0.8);
    k.l = k.x + Vec3d{dist * std::cos(az), dist * std::sin(az), u(0.2, 0.5)};
    const double el = u(20, 60) * std::numbers::pi / 180, az2 = u(0, 2 * std::numbers::pi), r = u(0.5, 1.2);
    k.c = {r * std::cos(el) * std::cos(az2), r * std::cos(el) * std::sin(az2), r * std::sin(el)};
  } else {  // walls
    k.x = {u(-0.5, 0.5), 0.0, u(0.0, 1.0)};
    k.n = {0, -1, 0};
    k.l = {u(-0.5, 0.5), u(-1.5, -0.2), u(2.0, 3.0)};
    k.c = {u(-0.5, 0.5), u(-2.0, -0.6), u(0.2, 1.2)};
  }
  const double lambda = std::exp(u(0.0, std::log(30.0)));
  k.vp = u(0.0, 2.0) * g3();
  k.vl = moving_light ? u(0.0, 1.0) * g3() : Vec3d{};
  k.vr = moving_recv ? u(0.0, 0.5) * g3() : Vec3d{};
  k.wr = moving_recv ? u(0.0, 2.0) * g3() : Vec3d{};
  k.s = k.x + 0.2 * g3();
  if (dot(k.l - k.x, k.n) < 0) k.n = -k.n;
  // The plan pass: the pixel's unit ray, the float depth (camera z), the float normal; PixelPlan's x and unit normal.
  k.ray = unit(k.x - k.c);
  k.depth = static_cast<float>(norm(k.x - k.c) * k.ray.z);
  const double s = static_cast<double>(k.depth) / k.ray.z;
  k.x = {k.c.x + k.ray.x * s, k.c.y + k.ray.y * s, k.c.z + k.ray.z * s};
  for (int i = 0; i < 3; ++i) k.normal[i] = static_cast<float>(i == 0 ? k.n.x : (i == 1 ? k.n.y : k.n.z));
  const Vec3d nf{k.normal[0], k.normal[1], k.normal[2]};
  k.n = (1.0 / norm(nf)) * nf;
  k.p = k.l + (1.0 / lambda) * (k.x - k.l);
  k.ok = std::fabs(k.ray.z) > 0.05 && dot(k.c - k.x, k.n) > 0 && dot(k.l - k.x, k.n) > 1e-6;
  return k;
}

// The shadow point at time t: the line L(t) P(t) met with the receiver plane(t).
VL shadow_at(const Config& k, LD t) {
  const VL l = ld(k.l) + t * ld(k.vl), p = ld(k.p) + t * ld(k.vp);
  const VL n = rotate(ld(k.n), k.wr, t);
  const VL q = ld(k.s) + rotate(ld(k.x) - ld(k.s), k.wr, t) + t * ld(k.vr);
  const VL d = p - l;
  return l + (dotl(q - l, n) / dotl(d, n)) * d;
}

// The image angular speed of the shadow point (rad/s), from a Richardson central difference.
double true_speed(const Config& k, double cos_psi) {
  const LD h = 1e-7L * std::max<LD>(cos_psi, 1e-3);
  const auto diff = [&](LD dt) { return (1 / (2 * dt)) * (shadow_at(k, dt) - shadow_at(k, -dt)); };
  const VL d1 = diff(h), d2 = diff(h / 2);
  const VL xd = (1.0L / 3) * (4.0L * d2 - d1);
  const VL view = ld(unit(k.x - k.c));
  const VL perp = xd - dotl(xd, view) * view;
  return static_cast<double>(std::sqrt(dotl(perp, perp)) / static_cast<LD>(norm(k.x - k.c)));
}

// The pixel stage's rate for the configuration's one pixel (a static camera, so k_static = 0) and its plan bits.
struct Planned {
  double k;
  uint8_t bits;
};

Planned plan_pixel(const Config& k, bool moving_light, bool moving_recv) {
  PixelPlan pp{};
  PlanParams& p = pp.p;
  p.jmax = 1.0;
  p.big = 1;
  p.lights = 1;
  p.width = p.height = 1;
  p.r_wc = Mat3d{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  p.c_c = k.c;
  p.obliquity = 1;
  p.bits = 1;
  const ShadowLight sl{0, 0, 1, k.l, 0};
  ShadowBody sb{};
  sb.mode = ShadowBody::kBeyond;
  sb.near = norm(k.p - k.l);
  sb.speed = norm(k.vp);
  MovingLight ml{MovingLight::kPoint, 0, 0, k.l, norm(k.vl), (norm(k.vl) + norm(k.vp)) / norm(k.p - k.l), 0.0};
  // The movers in the engine's order (Planner::terms: pair sl.first + m is movers[m]'s): the occluder's body (id 9;
  // the pair's), then the receiver's (id 5), so the receiver's pair is no self-shadow (D1f).
  const MoverBody movers[2] = {{9, k.p, norm(k.vp), 0.0, 0.0}, {5, k.s, norm(k.vr), norm(k.wr), 0.0}};
  const uint32_t instance = moving_recv ? 5 : 1;
  p.n_shadow = moving_light ? 0 : 1;
  p.n_moving = moving_light ? 1 : 0;
  p.n_movers = moving_recv ? 2 : 0;
  double out_k = 0;
  int32_t reach = 0;
  uint8_t flags = 0, bits = 0;
  double k_base = 0;
  pp.depth = &k.depth;
  pp.depth_stride = 1;
  pp.rays = &k.ray.x;
  pp.n = 1;
  pp.shadow_lights = &sl;
  pp.shadow_bodies = &sb;
  pp.moving = &ml;
  pp.spheres = nullptr;
  pp.k = &out_k;
  pp.reach = &reach;
  pp.flags = &flags;
  pp.instance = &instance;
  pp.movers = movers;
  pp.normal = k.normal;
  pp.pv_plan = &bits;
  pp.k_base = &k_base;
  pp(0);
  return {out_k, bits};
}

// P2-D D1f: a self-shadow. A rigid body B (velocity v, spin w about its centre s) carries the occluder point P, the
// receiver point X and its plane (normal n); the light is static: a point at l, or (directional) the direction u.
struct SelfConfig {
  Vec3d c, x, n, l, p, u, v, w, s;
  float depth;
  Vec3d ray;
  float normal[3];
  bool ok;
};

SelfConfig make_self(std::mt19937_64& rng, bool directional) {
  std::normal_distribution<double> gauss;
  const auto u = [&rng](double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); };
  const auto g3 = [&] { return Vec3d{gauss(rng), gauss(rng), gauss(rng)}; };
  SelfConfig k{};
  k.c = 0.3 * g3();
  k.x = 0.4 * g3() + Vec3d{0, 0, 1.5};
  k.n = unit(g3());
  k.u = unit(g3());                                   // directional: the light's propagation direction
  k.l = k.x + u(0.5, 3.0) * unit(g3());               // point
  const Vec3d to_light = directional ? -1.0 * k.u : unit(k.l - k.x);
  if (dot(to_light, k.n) < 0) k.n = -k.n;
  k.v = u(0.0, 1.0) * g3();
  k.w = u(0.0, 4.0) * g3();
  k.s = k.x + 0.1 * g3();
  k.ray = unit(k.x - k.c);
  k.depth = static_cast<float>(norm(k.x - k.c) * k.ray.z);
  const double sx = static_cast<double>(k.depth) / k.ray.z;
  k.x = {k.c.x + k.ray.x * sx, k.c.y + k.ray.y * sx, k.c.z + k.ray.z * sx};
  for (int i = 0; i < 3; ++i) k.normal[i] = static_cast<float>(i == 0 ? k.n.x : (i == 1 ? k.n.y : k.n.z));
  k.n = unit(Vec3d{k.normal[0], k.normal[1], k.normal[2]});
  const double gap = u(0.002, 0.2);                   // |X - P|, on the body (at most its diameter)
  k.p = directional ? k.x - gap * k.u : k.x + gap * unit(k.l - k.x);
  if (!directional && norm(k.l - k.x) <= gap * 1.05) k.p = k.x + 0.5 * (k.l - k.x);
  const Vec3d d = directional ? k.u : unit(k.x - k.l);
  k.ok = std::fabs(k.ray.z) > 0.05 && dot(k.c - k.x, k.n) > 0 && std::fabs(dot(k.n, d)) > 1e-6;
  return k;
}

// The self-shadow's point at time t: B's rigid motion moves P and the plane; the light stays.
VL self_shadow_at(const SelfConfig& k, bool directional, LD t) {
  const auto move = [&](Vec3d q) { return ld(k.s) + rotate(ld(q) - ld(k.s), k.w, t) + t * ld(k.v); };
  const VL p = move(k.p), q = move(k.x), n = rotate(ld(k.n), k.w, t);
  const VL dir = directional ? ld(k.u) : p - ld(k.l);
  return p + (dotl(q - p, n) / dotl(dir, n)) * dir;
}

double self_true_speed(const SelfConfig& k, bool directional, double cos_psi) {
  const LD h = 1e-7L * std::max<LD>(cos_psi, 1e-3);
  const auto diff = [&](LD dt) {
    return (1 / (2 * dt)) * (self_shadow_at(k, directional, dt) - self_shadow_at(k, directional, -dt));
  };
  const VL d1 = diff(h), d2 = diff(h / 2);
  const VL xd = (1.0L / 3) * (4.0L * d2 - d1);
  const VL view = ld(unit(k.x - k.c));
  const VL perp = xd - dotl(xd, view) * view;
  return static_cast<double>(std::sqrt(dotl(perp, perp)) / static_cast<LD>(norm(k.x - k.c)));
}

}  // namespace

TEST_CASE("obliquity: closed forms under a fronto-parallel view, and the cap") {
  // The camera looks along +z at a receiver with normal -z; the light comes in at incidence psi in the x-z plane: the
  // shadow stretches by 1 / cos psi along x and the view keeps it (g = 1 / cos psi), and the receiver's normal motion
  // moves the shadow along d, seen as tan psi (h).
  const Vec3d n{0, 0, -1}, r{0, 0, 1};
  for (const double deg : {0.0, 30.0, 60.0, 80.0}) {
    const double psi = deg * std::numbers::pi / 180;
    const Obliquity o = obliquity(n, r, Vec3d{std::sin(psi), 0, std::cos(psi)}, kObliquityCap);
    INFO("psi " << deg);
    CHECK(o.g == doctest::Approx(1 / std::cos(psi)).epsilon(1e-12));
    CHECK(o.h == doctest::Approx(std::tan(psi)).epsilon(1e-12).scale(1));
    CHECK(!o.beyond);
  }
  // A view along the stretch (grazing too) shrinks it: g < 1 / cos psi; d normal to the view's plane, g = 1.
  const double psi = 70 * std::numbers::pi / 180;
  const Vec3d d{std::sin(psi), 0, std::cos(psi)};
  CHECK(obliquity(n, unit(Vec3d{0.9, 0, 0.1}), d, kObliquityCap).g < 0.5 / std::cos(psi));
  CHECK(obliquity(n, r, Vec3d{0, std::sin(psi), std::cos(psi)}, kObliquityCap).g ==
        doctest::Approx(1 / std::cos(psi)).epsilon(1e-12));
  // Beyond the cap (cos psi < 1/8): the factors take the cap; a grazing light, a non-finite normal.
  const double steep = std::acos(0.1);
  const Vec3d grazing{std::sin(steep), 0, std::cos(steep)};
  const Obliquity capped = obliquity(n, r, grazing, kObliquityCap);
  CHECK((capped.beyond && capped.g == kObliquityCap && capped.h == kObliquityCap));
  const Obliquity uncapped = obliquity(n, r, grazing, kInf);
  CHECK((!uncapped.beyond && uncapped.g == doctest::Approx(10.0).epsilon(1e-12)));
  CHECK(obliquity(n, r, Vec3d{1, 0, 0}, kObliquityCap).beyond);
  CHECK(obliquity(Vec3d{std::numeric_limits<double>::quiet_NaN(), 0, 0}, r, d, kObliquityCap).beyond);
  // The factors do not depend on the normal's sign.
  const Obliquity a = obliquity(n, r, d, kObliquityCap), b = obliquity(-n, r, d, kObliquityCap);
  CHECK((a.g == doctest::Approx(b.g).epsilon(1e-15) && a.h == doctest::Approx(b.h).epsilon(1e-15)));
}

TEST_CASE("G-D8 (a), (b): the shadow-edge bound with the obliquity factors holds against the exact shadow point") {
  // (a) the pixel stage's rate (capped) on cos psi >= 1/8, where the cap does not bind, and the uncapped functor with
  // the full receiver term on every configuration: max ratio <= 1 + 1e-9; (b) the capped rate on every configuration:
  // every violation lies beyond the cap (the plan bit R6), and the share beyond it is reported (M4b: random 10 %,
  // walls 3.6 %, EVIMO2-like and lamp 0).
  const int64_t m = 100'000;
  uint64_t seed = 20261013;
  for (const char* family : {"random", "evimo2", "lamp", "walls"}) {
    for (const bool moving_light : {false, true}) {
      for (const bool moving_recv : {false, true}) {
        std::mt19937_64 rng(seed++);
        int64_t n_ok = 0, n_inside = 0, beyond = 0, violations = 0, unflagged = 0, uncapped_over = 0, inside_over = 0;
        double inside_max = 0, uncapped_max = 0;
        for (int64_t i = 0; i < m; ++i) {
          const Config k = make(family, rng, moving_light, moving_recv);
          if (!k.ok) continue;
          ++n_ok;
          const Vec3d d = unit(k.x - k.l);
          const double cos_psi = std::fabs(dot(k.n, d));
          const double truth = true_speed(k, cos_psi);
          const Planned pl = plan_pixel(k, moving_light, moving_recv);
          const bool flagged = (pl.bits & kPvBeyond) != 0;
          beyond += flagged ? 1 : 0;
          const bool over = truth > pl.k * (1 + 1e-9) + 1e-12;
          violations += over ? 1 : 0;
          unflagged += over && !flagged ? 1 : 0;
          if (cos_psi >= 1.0 / kObliquityCap) {
            ++n_inside;
            inside_max = std::max(inside_max, truth / pl.k);
            inside_over += over ? 1 : 0;
          }
          // The uncapped functor with the M4 study's bound (lambda |dP| + |lambda - 1| |dL|; |n . v_R| + |w_R| |X - S|
          // for the receiver).
          const Obliquity o = obliquity(k.n, unit(k.x - k.c), d, kInf);
          const double lambda = norm(k.x - k.l) / norm(k.p - k.l);
          const double recv = std::fabs(dot(k.n, k.vr)) + norm(k.wr) * norm(k.x - k.s);
          const double bound =
              (o.g * (lambda * norm(k.vp) + std::fabs(lambda - 1) * norm(k.vl)) + o.h * recv) / norm(k.x - k.c);
          if (cos_psi >= 1e-3) {
            uncapped_max = std::max(uncapped_max, truth / bound);
            uncapped_over += truth > bound * (1 + 1e-9) + 1e-12 ? 1 : 0;
          }
        }
        INFO(std::string(family) << ", moving light " << moving_light << ", moving receiver " << moving_recv);
        MESSAGE(std::string(family) << " light " << moving_light << " receiver " << moving_recv << ": " << n_ok
                                    << " configurations, "
                                    << 100.0 * static_cast<double>(beyond) / static_cast<double>(n_ok)
                                    << " % beyond the cap, " << violations << " capped violations, max ratio "
                                    << inside_max << " on cos psi >= 1/8 (" << n_inside << "), uncapped max ratio "
                                    << uncapped_max);
        CHECK(n_ok > m / 4);
        CHECK(inside_over == 0);
        CHECK(inside_max <= 1 + 1e-9);
        CHECK(uncapped_over == 0);
        CHECK(unflagged == 0);
      }
    }
  }
}

TEST_CASE("the pixel stage's rates without the factors are as built, with or without the plan bits") {
  // Under PlanParams::obliquity 0 the rates ignore the normals (P&V's plan bits only read them), and k_base (the rate
  // with g = h = 1) equals the as-built rate under the factors.
  std::mt19937_64 rng(7);
  for (int i = 0; i < 2000; ++i) {
    const Config k = make("random", rng, i % 2 == 1, i % 3 == 0);
    if (!k.ok) continue;
    const bool ml = i % 2 == 1, mr = i % 3 == 0;
    PixelPlan pp{};
    PlanParams& p = pp.p;
    p.jmax = 1.0;
    p.big = 1;
    p.lights = 1;
    p.width = p.height = 1;
    p.r_wc = Mat3d{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    p.c_c = k.c;
    const ShadowLight sl{0, 0, 1, k.l, 0};
    ShadowBody sb{};
    sb.mode = ShadowBody::kBeyond;
    sb.near = norm(k.p - k.l);
    sb.speed = norm(k.vp);
    MovingLight mlight{MovingLight::kPoint, 0, 0, k.l, norm(k.vl), (norm(k.vl) + norm(k.vp)) / norm(k.p - k.l), 0.0};
    const MoverBody mover{5, k.s, norm(k.vr), norm(k.wr), 0.0};
    const uint32_t instance = mr ? 5 : 1;
    p.n_shadow = ml ? 0 : 1;
    p.n_moving = ml ? 1 : 0;
    p.n_movers = mr ? 1 : 0;
    double out[3] = {}, base = 0;
    int32_t reach = 0;
    uint8_t flags = 0, bits = 0;
    pp.depth = &k.depth;
    pp.depth_stride = 1;
    pp.rays = &k.ray.x;
    pp.n = 1;
    pp.shadow_lights = &sl;
    pp.shadow_bodies = &sb;
    pp.moving = &mlight;
    pp.reach = &reach;
    pp.flags = &flags;
    pp.instance = &instance;
    pp.movers = &mover;
    pp.k = &out[0];
    pp(0);  // as built: no normals
    pp.normal = k.normal;
    pp.pv_plan = &bits;
    pp.k = &out[1];
    pp(0);  // the plan bits, no factors
    p.obliquity = 1;
    pp.k_base = &base;
    pp.k = &out[2];
    pp(0);  // the factors
    CHECK(out[1] == out[0]);
    CHECK(base == out[0]);
    CHECK((bits & kPvCone) != 0);
  }
}

TEST_CASE("the pixel stage's plan bits on random receivers: every bit and the factors exercised (P2-D D1)") {
  // plan_cases.h's input (test_device.cpp compares it host = device): the factors move some rates (k against k_base),
  // and PvPlanPass counts cone pixels with g_max > 1, R6, rough and R4 pixels; R4 needs a width.
  const balla::render::Camera cam = balla::render::test::pinhole(96, 72, 80.0, 0);
  const CameraGeometry geo(cam, 0.95);
  for (const bool band : {false, true}) {
    INFO("band and seeds " << band);
    namespace t = balla::engine::test;
    const t::PixelStageRun h = t::pixel_stage_run<Space::kHost>(t::pixel_stage_input(geo, band, 31));
    size_t moved = 0, thin_ok = 0;
    for (size_t i = 0; i < geo.n_pixels; ++i) {
      moved += h.k[i] != h.k_base[i] ? 1 : 0;
      thin_ok += (h.pv_plan[i] & kPvThin) == 0 || std::isfinite(h.width[i]) ? 1 : 0;
    }
    CHECK(moved > geo.n_pixels / 10);
    CHECK(thin_ok == geo.n_pixels);
    for (const unsigned long long c : h.counts) CHECK(c > 0);
    CHECK(h.tiles.k_base.size() == h.tiles.k.size());
  }
}

TEST_CASE("the plan bits alone leave the pixel stage's plan as built, with and without I8's band (P2-D D1)") {
  // plan_cases.h's input, whose normals are random (not the planes its depths imply): with the plan bits and R4 but
  // without the factors, every rate, reach, flag and tile total equals the pixel stage's without normals; the band's
  // region (d) keeps its finite-difference foreshortening (the normal's is R4's and, under the factors, the band's).
  const balla::render::Camera cam = balla::render::test::pinhole(96, 72, 80.0, 0);
  const CameraGeometry geo(cam, 0.95);
  namespace t = balla::engine::test;
  for (const bool band : {false, true}) {
    INFO("band and seeds " << band);
    t::PixelStageInput in = t::pixel_stage_input(geo, band, 31);
    in.st.params.obliquity = 0;
    const t::PixelStageRun bits = t::pixel_stage_run<Space::kHost>(in);
    in.st.params.bits = in.st.params.pv = 0;
    const t::PixelStageRun none = t::pixel_stage_run<Space::kHost>(in);
    CHECK(bits.k == none.k);
    CHECK(bits.reach == none.reach);
    CHECK(bits.flags == none.flags);
    CHECK(bits.tiles.k == none.tiles.k);
    CHECK(bits.tiles.reach == none.tiles.reach);
    CHECK(bits.tiles.depend == none.tiles.depend);
    CHECK(bits.tiles.kcone == none.tiles.kcone);
    CHECK(bits.tiles.ksource == none.tiles.ksource);
    CHECK(bits.tiles.sreach == none.tiles.sreach);
    CHECK(bits.tiles.cone == none.tiles.cone);
    CHECK(bits.tiles.seed == none.tiles.seed);
    CHECK(bits.tiles.shadow_pixels == none.tiles.shadow_pixels);
    CHECK(bits.tiles.seed_pixels == none.tiles.seed_pixels);
    CHECK(none.pv_plan.empty());
    CHECK(bits.pv_plan.size() == geo.n_pixels);
    CHECK(bits.counts[0] > 0);
  }
}

TEST_CASE("D1f: under the factors the pixel stage's as-built outputs equal the outputs with the factors off") {
  // plan_cases.h's input with I8's band, whose random normals make region (d) decide otherwise under the factors (cos_n
  // in place of the finite-difference foreshortening): k_base and the band's kcone0, ksource0 and sreach0 under the
  // factors are the rates, cone and source rates and reaches with the factors off, per pixel and per tile, bit for bit,
  // so that Planner::finish's levels without the factors are the as-built ones.
  const balla::render::Camera cam = balla::render::test::pinhole(96, 72, 80.0, 0);
  const CameraGeometry geo(cam, 0.95);
  namespace t = balla::engine::test;
  t::PixelStageInput in = t::pixel_stage_input(geo, true, 31);
  in.st.params.obliquity = 1;
  const t::PixelStageRun on = t::pixel_stage_run<Space::kHost>(in);
  in.st.params.obliquity = 0;
  const t::PixelStageRun off = t::pixel_stage_run<Space::kHost>(in);
  CHECK(on.k_base == off.k);
  CHECK(on.tiles.k_base == off.tiles.k);
  CHECK(on.tiles.kcone0 == off.tiles.kcone);
  CHECK(on.tiles.ksource0 == off.tiles.ksource);
  CHECK(on.tiles.sreach0 == off.tiles.sreach);
  CHECK(on.tiles.kcone != off.tiles.kcone);  // the factors move the band's cone rates
  CHECK(on.tiles.ksource != off.tiles.ksource);
  CHECK(off.tiles.kcone0.empty());
}

TEST_CASE("PvPlanPass: D_bound takes the cone gain only with the rates as built (P2-D D1)") {
  // One cone pixel with g_max = 5 (q = 9) on a tile at level 3 and v_hat 100 px/s: D_bound = 100 W / 8 (about 0.105
  // px) times 5 as built, times 1 under the factors (v_hat holds them); a 0.6 px shadow is thin (R4) only as built.
  const uint32_t tile = 0;
  const int8_t level = 3;
  const float v_hat = 100.0f, width = 0.6f;
  for (const uint32_t obliquity : {0u, 1u}) {
    uint8_t b = static_cast<uint8_t>(9 | kPvCone);
    unsigned long long counts[4] = {};
    PvPlanPass{&b, &width, &tile, &level, &v_hat, 6, 1.0 / 16, 1, 0, 0, 1, 1, counts, obliquity}(0);
    CHECK(((b & kPvThin) != 0) == (obliquity == 0));
    CHECK(counts[0] == 1);
    CHECK(counts[3] == (obliquity == 0 ? 1u : 0u));
  }
}

TEST_CASE("P2-F F9b: the pixel stage's AO receivers against the reach test, and nothing else moved") {
  // plan_cases.h's input with two moving bodies' AO reach (test_device.cpp compares it host = device), with and without
  // I8's band and the seeds, against the same input without them: a pixel is an AO receiver (kPlanAo) exactly where its
  // centre ray's receiver point (as the stage computes it) lies within some body's reach, and its rate and k_base are
  // then the largest of its own and those bodies' receiver rates; every other flag, reach, seed, plan bit and count is
  // unchanged, and a tile depends on movers where it did or where it has an AO receiver.
  const balla::render::Camera cam = balla::render::test::pinhole(96, 72, 80.0, 0);
  const CameraGeometry geo(cam, 0.95);
  namespace t = balla::engine::test;
  const size_t n = geo.n_pixels;
  for (const bool band : {false, true}) {
    INFO("band and seeds " << band);
    t::PixelStageInput in = t::pixel_stage_input(geo, band, 31, true);
    const t::PixelStageRun ao = t::pixel_stage_run<Space::kHost>(in);
    const std::vector<AoMover> movers = in.st.ao;
    in.st.ao.clear();
    in.st.params.n_ao = 0;
    const t::PixelStageRun none = t::pixel_stage_run<Space::kHost>(in);
    const PlanParams& p = in.st.params;
    size_t marked = 0, raised = 0;
    std::vector<uint8_t> tile_ao(ao.tiles.depend.size(), 0);
    for (size_t i = 0; i < n; ++i) {
      const Vec3d ray{geo.rays[3 * i], geo.rays[3 * i + 1], geo.rays[3 * i + 2]};
      const double rng = static_cast<double>(in.depth[i]) / ray.z;
      bool within = false;
      double rate = 0.0;
      if (std::isfinite(rng)) {
        const Vec3d x{p.c_c.x + np::dot_fma_a(ray, p.r_wc.row(0)) * rng,
                      p.c_c.y + np::dot_fma_a(ray, p.r_wc.row(1)) * rng,
                      p.c_c.z + np::dot_fma_a(ray, p.r_wc.row(2)) * rng};
        for (const AoMover& m : movers) {
          const Vec3d rel = x - m.centre;
          if (!(np::dot_seq(rel, rel) <= m.reach * m.reach)) continue;
          within = true;
          rate = np::maximum(rate, receiver_rate(p, rng - p.v * kWindowS, m.speed));
        }
      }
      CHECK(((ao.flags[i] & kPlanAo) != 0) == within);
      CHECK((ao.flags[i] & ~kPlanAo) == none.flags[i]);
      CHECK(ao.k[i] == (within ? np::maximum(none.k[i], rate) : none.k[i]));
      CHECK(ao.k_base[i] == (within ? np::maximum(none.k_base[i], rate) : none.k_base[i]));
      CHECK(ao.reach[i] == none.reach[i]);
      marked += within ? 1 : 0;
      raised += ao.k[i] > none.k[i] ? 1 : 0;
      if (within) tile_ao[geo.pixel_tile[i]] = 1;
    }
    CHECK(marked > n / 20);
    CHECK(marked < n / 2);
    CHECK(raised > 0);
    CHECK(ao.tiles.ao_pixels == marked);
    CHECK(none.tiles.ao_pixels == 0);
    for (size_t tl = 0; tl < tile_ao.size(); ++tl) {
      CHECK(ao.tiles.depend[tl] == ((none.tiles.depend[tl] != 0 || tile_ao[tl] != 0) ? 1 : 0));
    }
    CHECK(ao.tiles.shadow_pixels == none.tiles.shadow_pixels);
    CHECK(ao.tiles.reach == none.tiles.reach);
    CHECK(ao.tiles.kcone == none.tiles.kcone);
    CHECK(ao.tiles.ksource == none.tiles.ksource);
    CHECK(ao.tiles.sreach == none.tiles.sreach);
    CHECK(ao.tiles.seed == none.tiles.seed);
    CHECK(ao.tiles.seed_pixels == none.tiles.seed_pixels);
    CHECK(ao.pv_plan == none.pv_plan);
    CHECK(std::memcmp(ao.width.data(), none.width.data(), n * sizeof(float)) == 0);
    CHECK(ao.counts == none.counts);
  }
}

TEST_CASE("D1f: a self-shadow's bound in the body's frame holds against the exact shadow point") {
  // A point light through the pixel stage (the receiver on the pair's own body: movers[0] is the pair's), a directional
  // light through PixelPlan::self_edge / self_rel; the body translates and spins, so the world-frame form (g lambda
  // s_P + h s_r) counts its motion twice: the body frame's s_r + g (lambda - 1) |dL/dt in B's frame| (point; g |X - P|
  // w for a direction) is the lesser in most configurations, and the rate never falls below the truth where the cap
  // does not bind (cos psi >= 1/8). The world-frame form is no bound where h alone is beyond the cap: there the body's
  // is taken alone.
  const int64_t m = 50'000;
  for (const bool directional : {false, true}) {
    std::mt19937_64 rng(directional ? 41 : 40);
    int64_t n_ok = 0, n_inside = 0, over = 0, tighter = 0;
    double worst = 0, gain_sum = 0;
    for (int64_t i = 0; i < m; ++i) {
      const SelfConfig k = make_self(rng, directional);
      if (!k.ok) continue;
      ++n_ok;
      const Vec3d d = directional ? k.u : unit(k.x - k.l);
      const double cos_psi = std::fabs(dot(k.n, d));
      const double truth = self_true_speed(k, directional, cos_psi);
      const MoverBody body{5, k.s, norm(k.v), norm(k.w), 0.0};
      const double s_p = norm(k.v) + norm(k.w) * norm(k.p - k.s);  // the occluder's point speed bound
      const Obliquity ob = obliquity(k.n, k.ray, d, kObliquityCap);
      const double range = norm(k.x - k.c), s_r = rigid_speed(body, k.x);
      const double lambda = directional ? 1.0 : norm(k.x - k.l) / norm(k.p - k.l);
      double rate = 0;
      if (directional) {
        PixelPlan::Oblique o{true, unit(k.n), k.ray, s_r, &body};
        const ShadowLight sl{1, 0, 1, -1.0 * k.u, 0};
        rate = PixelPlan::self_edge(&o, ob, lambda * s_p, PixelPlan::self_rel(sl, body, 1.0, norm(k.x - k.p))) / range;
      } else {
        PixelPlan pp{};
        PlanParams& p = pp.p;
        p.jmax = 1.0;
        p.big = 1;
        p.lights = 1;
        p.width = p.height = 1;
        p.r_wc = Mat3d{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
        p.c_c = k.c;
        p.obliquity = 1;
        p.bits = 1;
        p.n_shadow = 1;
        p.n_movers = 1;
        const ShadowLight sl{0, 0, 1, k.l, 0};
        ShadowBody sb{};
        sb.mode = ShadowBody::kBeyond;
        sb.near = norm(k.p - k.l);
        sb.speed = s_p;
        const uint32_t instance = 5;
        double out_k = 0, k_base = 0;
        int32_t reach = 0;
        uint8_t flags = 0, bits = 0;
        pp.depth = &k.depth;
        pp.depth_stride = 1;
        pp.rays = &k.ray.x;
        pp.n = 1;
        pp.shadow_lights = &sl;
        pp.shadow_bodies = &sb;
        pp.k = &out_k;
        pp.reach = &reach;
        pp.flags = &flags;
        pp.instance = &instance;
        pp.movers = &body;
        pp.normal = k.normal;
        pp.pv_plan = &bits;
        pp.k_base = &k_base;
        pp(0);
        rate = out_k;
      }
      const double world = (ob.g * lambda * s_p + ob.h * s_r) / range;
      if (!(ob.beyond && !ob.g_beyond)) CHECK(rate <= world * (1 + 1e-12));
      tighter += rate < world * (1 - 1e-9) ? 1 : 0;
      gain_sum += rate / world;
      if (cos_psi >= 1.0 / kObliquityCap) {
        ++n_inside;
        worst = std::max(worst, truth / rate);
        over += truth > rate * (1 + 1e-9) + 1e-12 ? 1 : 0;
      }
    }
    INFO("directional " << directional);
    MESSAGE(std::string(directional ? "directional" : "point")
            << ": " << n_ok << " configurations, the body frame tighter in "
            << 100.0 * static_cast<double>(tighter) / static_cast<double>(n_ok) << " %, mean rate / world form "
            << gain_sum / static_cast<double>(n_ok) << ", max truth / rate " << worst << " on cos psi >= 1/8 ("
            << n_inside << ")");
    CHECK(n_ok > m / 4);
    CHECK(over == 0);
    CHECK(worst <= 1 + 1e-9);
    CHECK(tighter > n_ok / 2);
  }
}

TEST_CASE("D1f: R6 on a static receiver takes g alone; h only where the receiver moves") {
  // A configuration with h beyond the cap and g within it (about 0.16 % of random (n, r, d); h - g stays below 1, so
  // only near the cap: h 8-9, g 7-8): the cap's flag is set,
  // g's is not; the pixel stage marks R6 only on a moving receiver that is not the pair's own body.
  std::mt19937_64 rng(5);
  std::normal_distribution<double> gauss;
  const auto g3 = [&] { return unit(Vec3d{gauss(rng), gauss(rng), gauss(rng)}); };
  Vec3d n{}, r{}, d{};
  Obliquity o{};
  for (int i = 0; i < 10'000'000; ++i) {
    n = g3();
    r = g3();
    d = g3();
    const Obliquity raw = obliquity(n, r, d, kInf);
    if (r.z > 0.2 && raw.h > 8.2 && raw.g < 7.95) break;  // clear of the cap after the plan pass's float normal
  }
  o = obliquity(n, r, d, kObliquityCap);
  REQUIRE((o.beyond && !o.g_beyond));
  CHECK(o.h == kObliquityCap);
  CHECK(o.g < kObliquityCap);
  CHECK(obliquity(n, r, Vec3d{n.y, -n.x, 0.0}, kObliquityCap).g_beyond);  // d in the plane: cos psi 0
  // Through the pixel stage: camera at the origin, the receiver X on the ray r at range 1.5, its normal n facing the
  // camera, a point light on the line through X along -d (the occluder between).
  if (dot(n, r) > 0) n = -n;
  const Vec3d x = 1.5 * r, l = x - 1.0 * d;
  for (const int recv : {0, 1, 2}) {  // static, moving (another body's pair), moving (its own body's pair)
    PixelPlan pp{};
    PlanParams& p = pp.p;
    p.jmax = 1.0;
    p.big = 1;
    p.lights = 1;
    p.width = p.height = 1;
    p.r_wc = Mat3d{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    p.bits = 1;
    p.n_shadow = 1;
    p.n_movers = 2;
    const ShadowLight sl{0, 0, 1, l, 0};
    ShadowBody sb{};
    sb.mode = ShadowBody::kBeyond;
    sb.near = 0.5;
    sb.speed = 1.0;
    const MoverBody movers[2] = {{9, x + 0.5 * d, 1.0, 0.0, 0.0}, {5, x, 0.5, 0.5, 0.0}};
    const uint32_t instance = recv == 0 ? 1 : (recv == 1 ? 5 : 9);
    const float depth = static_cast<float>(x.z);
    const Vec3d ray = r;
    const float normal[3] = {static_cast<float>(n.x), static_cast<float>(n.y), static_cast<float>(n.z)};
    double out_k = 0;
    int32_t reach = 0;
    uint8_t flags = 0, bits = 0;
    pp.depth = &depth;
    pp.depth_stride = 1;
    pp.rays = &ray.x;
    pp.n = 1;
    pp.shadow_lights = &sl;
    pp.shadow_bodies = &sb;
    pp.k = &out_k;
    pp.reach = &reach;
    pp.flags = &flags;
    pp.instance = &instance;
    pp.movers = movers;
    pp.normal = normal;
    pp.pv_plan = &bits;
    pp(0);
    INFO("receiver " << recv);
    CHECK((bits & kPvCone) != 0);
    CHECK(((bits & kPvBeyond) != 0) == (recv == 1));
  }
}
