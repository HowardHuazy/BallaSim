#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "core/camera/injective_domain.h"
#include "parity.h"
#include "render/shade/bsdf_tables.h"

using namespace balla;
using namespace balla::gpu_test;

namespace {

template <class Case, class... Context>
auto evaluate_on_host(std::span<const Case> cases, const Context&... ctx) {
  std::vector<decltype(evaluate(ctx..., cases[0]))> out;
  out.reserve(cases.size());
  for (const Case& c : cases) out.push_back(evaluate(ctx..., c));
  return out;
}

// Bitwise comparison; NaN payloads and signed zeros must match too.
template <class Result>
void check_bitwise(const std::vector<Result>& host, const std::vector<Result>& device) {
  REQUIRE(host.size() == device.size());
  size_t mismatches = 0, first = 0;
  for (size_t i = 0; i < host.size(); ++i) {
    if (std::memcmp(&host[i], &device[i], sizeof(Result)) != 0 && mismatches++ == 0) first = i;
  }
  INFO("first mismatch at case " << first);
  CHECK(mismatches == 0);
}

template <class T>
camera::RadtanCamera<T> samsung() {
  // EVIMO2 Samsung (M2 §D.3) with its injective domain.
  camera::RadtanCamera<double> c{520.257996, 520.297974, 321.259003, 242.490005, 0.107593,
                                 -0.156436,  0.001883,   -0.000958,  0.0,        0.0,
                                 0.0};
  c.r_lim = camera::radtan_injective_domain(c, 640, 480, 0.95).r_lim;
  return {T(c.fx), T(c.fy), T(c.cx), T(c.cy), T(c.k1), T(c.k2), T(c.p1), T(c.p2), T(c.k3), T(c.r_lim),
          T(0.05)};
}

template <class T>
camera::KannalaBrandtCamera<T> fisheye() {
  camera::KannalaBrandtCamera<double> c{300.0, 300.5, 640.0, 480.0, -0.013, 0.0021, -0.0009, 0.0001,
                                        0.0};
  c.theta_lim = camera::kannala_brandt_injective_domain(c, 1280, 960, 0.95).theta_lim;
  return {T(c.fx), T(c.fy), T(c.cx), T(c.cy), T(c.k1), T(c.k2), T(c.k3), T(c.k4), T(c.theta_lim)};
}

// Points over and beyond the field of view (incl. behind the camera and on the axis) and pixels
// over and beyond the image, so both the valid and the rejecting branches run.
template <class T>
std::vector<CameraCase<T>> camera_cases(int width, int height, double max_angle) {
  std::mt19937_64 rng(31);
  std::uniform_real_distribution<double> theta(0.0, max_angle), phi(0.0, 2 * std::numbers::pi),
      range(0.1, 8.0), u(-60.0, width + 60.0), v(-60.0, height + 60.0);
  std::vector<CameraCase<T>> cases;
  for (int i = 0; i < 20000; ++i) {
    const double th = theta(rng), ph = phi(rng), r = range(rng);
    cases.push_back({{T(r * std::sin(th) * std::cos(ph)), T(r * std::sin(th) * std::sin(ph)),
                      T(r * std::cos(th))},
                     {T(u(rng)), T(v(rng))}});
  }
  cases.push_back({{T(0), T(0), T(2)}, {T(width / 2), T(height / 2)}});
  cases.push_back({{T(0), T(0), T(-1)}, {T(-1e4), T(1e4)}});
  return cases;
}

}  // namespace

TEST_CASE("Philox, purpose-tagged seeds and Owen-Sobol are bitwise equal on host and device") {
  std::vector<RngCase> cases = {
      // Random123 known-answer counters and keys.
      {{0u, 0u, 0u, 0u}, {0u, 0u}, 0, 0, 0, 0},
      {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xffffffffu}, 255, 0xffffffffu,
       ~uint64_t{0}, 0xffffffffu},
      {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u}, 2, 307199,
       0x0123456789abcdefull, 63},
  };
  std::mt19937_64 rng(5);
  for (int i = 0; i < 50000; ++i) {
    const uint64_t a = rng(), b = rng(), c = rng();
    cases.push_back({{uint32_t(a), uint32_t(a >> 32), uint32_t(b), uint32_t(b >> 32)},
                     {uint32_t(c), uint32_t(c >> 32)},
                     uint32_t(rng() % 4),
                     uint32_t(rng() % 307200),
                     rng(),
                     uint32_t(i < 25000 ? i : rng())});
  }
  const std::vector<RngResult> device = evaluate_on_device(std::span<const RngCase>(cases));
  check_bitwise(evaluate_on_host(std::span<const RngCase>(cases)), device);
  // The device reproduces the known answers and the golden pattern seeds too.
  CHECK(device[0].philox[0] == 0x6627e8d5u);
  CHECK(device[1].philox[3] == 0x6d5451fdu);
  CHECK(device[2].philox[0] == 0xd16cfe09u);
  CHECK(device[0].pattern_seed == 0xc2972a5au);
  CHECK(device[2].pattern_seed == 0x8d4933a3u);
}

TEST_CASE("sub-sample offsets through the fp32 warp table are bitwise equal on host and device") {
  const rng::WarpTable<float> table(rng::PixelResponse(1.0, 0.15, 3.0), 256);
  std::vector<OffsetCase> cases;
  std::mt19937_64 rng(6);
  for (uint32_t pixel = 0; pixel < 2000; ++pixel) {
    const uint32_t seed = rng::subsample_pattern_seed(rng(), 0, pixel);
    for (uint32_t k = 0; k < 64; ++k) cases.push_back({seed, k});
  }
  const std::span<const OffsetCase> view(cases);
  check_bitwise(evaluate_on_host(view, table.view()), evaluate_on_device(table, view));
}

TEST_CASE_TEMPLATE("pmath sin, cos, sincos, atan2, asin and acos are bitwise equal on host and device", T, float,
                   double) {
  std::mt19937_64 rng(7);
  const double max_arg = std::is_same_v<T, float> ? 6000.0 : 1e6;
  std::uniform_real_distribution<double> wide(-max_arg, max_arg), unit(-4.0, 4.0);
  std::vector<TrigCase<T>> cases;
  for (int i = 0; i < 100000; ++i) {
    const bool small = i % 2 == 0;
    cases.push_back({T(small ? unit(rng) : wide(rng)), T(unit(rng))});
  }
  for (const T x : {T(0), -T(0), T(1e-30), std::numbers::pi_v<T> / 4, std::numbers::pi_v<T>}) {
    for (const T y : {T(0), -T(0), T(1), T(-1), T(1e-30), T(4), T(-4), T(3.9999)}) cases.push_back({x, y});
  }
  const std::span<const TrigCase<T>> view(cases);
  check_bitwise(evaluate_on_host(view), evaluate_on_device(view));
}

TEST_CASE_TEMPLATE("pmath exponentials, logarithms and special functions are bitwise equal on host and "
                   "device",
                   T, float, double) {
  constexpr bool kFloat = std::is_same_v<T, float>;
  std::mt19937_64 rng(9);
  // x over the full exp range (both underflow and overflow branches) and the Taylor/Rybicki
  // switches; y log-uniform over the normal range plus subnormals.
  std::uniform_real_distribution<double> wide(kFloat ? -100.0 : -720.0, kFloat ? 100.0 : 720.0), unit(-2.0, 2.0),
      decade(kFloat ? -40.0 : -310.0, kFloat ? 38.0 : 307.0), small(0.0, 3.0);
  std::vector<ExpLogCase<T>> cases;
  for (int i = 0; i < 100000; ++i) {
    const double x = i % 2 == 0 ? unit(rng) : wide(rng);
    const double y = i % 3 == 0 ? small(rng) : std::pow(10.0, decade(rng));
    cases.push_back({T(x), T(y) > T(0) ? T(y) : std::numeric_limits<T>::denorm_min()});
  }
  for (const T x : {T(0), -T(0), T(0.35), T(-0.35), T(0.2), T(1), T(1e8), T(2e8), T(-3e8)}) {
    for (const T y : {T(1), T(0.5), T(1e-30), std::numeric_limits<T>::denorm_min(), std::numeric_limits<T>::max()}) {
      cases.push_back({x, y});
    }
  }
  const std::span<const ExpLogCase<T>> view(cases);
  check_bitwise(evaluate_on_host(view), evaluate_on_device(view));
}

TEST_CASE_TEMPLATE("radtan projection, Jacobian and unprojection are bitwise equal on host and device",
                   T, float, double) {
  const camera::RadtanCamera<T> cam = samsung<T>();
  const std::vector<CameraCase<T>> cases = camera_cases<T>(640, 480, 1.2);
  const std::span<const CameraCase<T>> view(cases);
  const std::vector<CameraResult<T>> device = evaluate_on_device(cam, view);
  check_bitwise(evaluate_on_host(view, cam), device);
  size_t projected = 0, unprojected = 0;
  for (const CameraResult<T>& r : device) {
    projected += r.projected == T(1);
    unprojected += r.unprojected == T(1);
  }
  CHECK(projected > 0);  // both branches ran
  CHECK(projected < device.size());
  CHECK(unprojected > 0);
  CHECK(unprojected < device.size());
}

TEST_CASE_TEMPLATE("Kannala-Brandt projection, Jacobian and unprojection are bitwise equal on host "
                   "and device",
                   T, float, double) {
  const camera::KannalaBrandtCamera<T> cam = fisheye<T>();
  const std::vector<CameraCase<T>> cases = camera_cases<T>(1280, 960, std::numbers::pi);
  const std::span<const CameraCase<T>> view(cases);
  check_bitwise(evaluate_on_host(view, cam), evaluate_on_device(cam, view));
}

TEST_CASE("ray LUT lookups are bitwise equal on host and device") {
  const camera::RadtanCamera<double> cam = samsung<double>();
  const camera::RayLut<float> lut(cam, 640, 480, 0.95, 2);
  std::mt19937_64 rng(8);
  std::uniform_real_distribution<float> u(-0.95f, 639.95f), v(-0.95f, 479.95f);
  std::vector<Vec2<float>> pixels;
  for (int i = 0; i < 100000; ++i) pixels.push_back({u(rng), v(rng)});
  // Far outside the grid: cells beyond int32 extrapolate from the same border cell.
  pixels.insert(pixels.end(),
                {{3e9f, 100.f}, {-3e9f, 100.f}, {100.f, 3e9f}, {100.f, -3e9f}, {1e5f, -1e5f}});
  const std::span<const Vec2<float>> view(pixels);
  check_bitwise(evaluate_on_host(view, lut.view()), evaluate_on_device(lut, view));
}

TEST_CASE_TEMPLATE("the radiance v1 BSDF (tables, floors, lobe, LTC) is bitwise equal on host and device", T, float,
                   double) {
  // Random normals and directions over both hemispheres (mu_v < 0 exercises the floor, mu_l <= 0 the horizon), every
  // roughness incl. below the reflection policy, and all material parameters; plus the s = m = 0 Lambert limit,
  // white metals (F0 = F90 = 1, so E_s = A + B), whose near-mirror rows reach the cap of E_s, and exact normal views
  // (the LTC frame's fallback basis).
  std::mt19937_64 rng(37);
  std::uniform_real_distribution<double> u(-1.0, 1.0), u01(0.0, 1.0);
  const auto unit = [&] {
    Vec3<double> a{u(rng), u(rng), u(rng)};
    while (dot(a, a) < 1e-4) a = {u(rng), u(rng), u(rng)};
    return (1.0 / std::sqrt(dot(a, a))) * a;
  };
  const auto cast_vec = [](Vec3<double> a) { return Vec3<T>{T(a.x), T(a.y), T(a.z)}; };
  std::vector<BsdfCase<T>> cases;
  for (int i = 0; i < 100000; ++i) {
    const Vec3<double> n = unit();
    Vec3<double> v = unit(), l = unit();
    if (i % 4 != 0 && dot(n, v) < 0.0) v = -v;  // mostly front-facing views
    if (i % 11 == 5) v = n;                      // exact normal view: the LTC frame's fallback basis
    if (i % 3 != 0 && dot(n, l) < 0.0) l = -l;
    const double r = u01(rng), rho = u01(rng), metallic = i % 5 == 0 ? 0.0 : u01(rng);
    const bool white_metal = i % 7 == 3;
    cases.push_back({cast_vec(n), cast_vec(v), cast_vec(l), T(white_metal ? 1.0 : rho), T(white_metal ? 1.0 : metallic),
                     T(r * r * r * r), T(i % 5 == 0 ? 0.0 : 0.1 * u01(rng)), T(0.01 * u01(rng)),
                     T(i % 2 == 0 ? 1.0 : u01(rng))});
  }
  const std::span<const BsdfCase<T>> view(cases);
  const render::bsdf::BsdfTables tables = render::bsdf::host_tables();
  check_bitwise(evaluate_on_host(view, tables), evaluate_on_device(tables, view));
}
