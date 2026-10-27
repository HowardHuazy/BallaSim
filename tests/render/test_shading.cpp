// Shared tracing and shading maths: lights, shadow-ray set-up, re-intersection, ray offsets,
// sphere culling and the texture footprint.
#include <doctest/doctest.h>

#include <cmath>
#include <numbers>
#include <random>

#include "render/ray.h"
#include "render/shade/lights.h"
#include "render/shade/shade.h"
#include "render/surface.h"

using namespace balla;
using namespace balla::render;

namespace {

PoseRow identity_row() {
  PoseRow r{};
  r.m[0].x = r.m[1].y = r.m[2].z = 1.0f;
  r.sphere.w = 1.0f;
  return r;
}

LightRow light_at(Vec3f p, Vec3f d = {0.0f, 0.0f, 1.0f}) { return {{p.x, p.y, p.z}, {d.x, d.y, d.z}, {0.0f, 0.0f, 0.0f}}; }

}  // namespace

TEST_CASE("point, spot and directional incidence") {
  const LightView point{LightKind::kPoint, 0, 8.0f, 0.0f, 0.0f, -1, 0};
  Incidence in = incidence(point, light_at({0.0f, 0.0f, 2.0f}), {0.0f, 0.0f, 0.0f});
  CHECK(in.e == doctest::Approx(2.0f));  // I / d^2
  CHECK(in.omega.z == doctest::Approx(1.0f));
  // Lambert is linear in the (unnormalized) normal and clips at the horizon.
  const Vec3f n{0.0f, 0.6f, 0.8f};
  CHECK(lambert_irradiance(point, light_at({0.0f, 0.0f, 2.0f}), {0.0f, 0.0f, 0.0f}, n) == doctest::Approx(1.6f));
  CHECK(lambert_irradiance(point, light_at({0.0f, 0.0f, 2.0f}), {0.0f, 0.0f, 0.0f}, 0.5f * n) ==
        doctest::Approx(0.8f));
  CHECK(lambert_irradiance(point, light_at({0.0f, 0.0f, 2.0f}), {0.0f, 0.0f, 0.0f}, -1.0f * n) == 0.0f);

  // Spot pointing down -z from z = 2: full inside the inner cone, smootherstep between, 0 outside.
  const float inner = 0.2f, outer = 0.4f;
  const LightView spot{LightKind::kSpot, 0, 4.0f, std::cos(outer), std::cos(inner), -1, 0};
  const LightRow srow = light_at({0.0f, 0.0f, 2.0f}, {0.0f, 0.0f, -1.0f});
  CHECK(incidence(spot, srow, {0.0f, 0.0f, 0.0f}).e == doctest::Approx(1.0f));
  for (float a : {0.25f, 0.3f, 0.35f}) {
    const Vec3f x{2.0f * std::tan(a), 0.0f, 0.0f};
    const float d2 = 4.0f + x.x * x.x;
    const float expected = 4.0f / d2 * pmath::smootherstep((std::cos(a) - std::cos(outer)) / (std::cos(inner) - std::cos(outer)));
    CHECK(incidence(spot, srow, x).e == doctest::Approx(expected).epsilon(1e-5));
  }
  CHECK(incidence(spot, srow, {2.0f * std::tan(0.45f), 0.0f, 0.0f}).e == 0.0f);

  const LightView sun{LightKind::kDirectional, 0, 3.0f, 0.0f, 0.0f, 0, 0};
  in = incidence(sun, light_at({0.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}), {5.0f, 5.0f, 5.0f});
  CHECK(in.e == 3.0f);
  CHECK(in.omega.y == 1.0f);
}

TEST_CASE("shadow-ray set-up: trace, self-occlusion and no-contribution cases") {
  // Camera at the origin looking at the plane z = 1 (triangle normal +z, facing away).
  const Triangle tri{{-1.0f, -1.0f, 1.0f}, {1.0f, -1.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, 0, 1, 2};
  const PoseRow row = identity_row();
  const Vec3f d{0.1f, 0.05f, 1.0f};
  const Ray ray = primary_to_geometry(row, d);
  const SurfaceHit sh = reintersect(tri, ray);
  REQUIRE(sh.t == doctest::Approx(1.0f));
  const LightView point{LightKind::kPoint, 0, 1.0f, 0.0f, 0.0f, 0, 0};

  // Light on the camera side: traced, origin offset towards the camera, light at t = 1.
  ShadowRay sr = shadow_ray(point, light_at({0.5f, 0.5f, 0.2f}), row, d, ray, sh);
  REQUIRE(sr.state == ShadowState::kTrace);
  CHECK(sr.origin_cam.z < 1.0f);
  CHECK(sr.origin_cam.z > 0.9999f);
  CHECK(sr.tmax == 1.0f);
  const Vec3f end = sr.origin_cam + sr.dir_cam;
  CHECK(end.x == doctest::Approx(0.5f));
  CHECK(end.z == doctest::Approx(0.2f));

  // Light behind the plane: the surface itself blocks it.
  sr = shadow_ray(point, light_at({0.0f, 0.0f, 3.0f}), row, d, ray, sh);
  CHECK(sr.state == ShadowState::kSelfOccluded);

  // Point outside a spot cone: no contribution, no ray.
  const LightView spot{LightKind::kSpot, 0, 1.0f, std::cos(0.1f), std::cos(0.05f), 0, 0};
  sr = shadow_ray(spot, light_at({5.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}), row, d, ray, sh);
  CHECK(sr.state == ShadowState::kNoContribution);

  // Directional light towards the camera side: infinite segment.
  const LightView sun{LightKind::kDirectional, 0, 1.0f, 0.0f, 0.0f, 0, 0};
  sr = shadow_ray(sun, light_at({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}), row, d, ray, sh);
  REQUIRE(sr.state == ShadowState::kTrace);
  CHECK(std::isinf(sr.tmax));
  CHECK(sr.dir_cam.z == -1.0f);
}

TEST_CASE("re-intersection gives the exact plane distance and clamps barycentrics") {
  const Triangle tri{{0.0f, 0.0f, 2.0f}, {1.0f, 0.0f, 2.0f}, {0.0f, 1.0f, 2.0f}, 0, 1, 2};
  SurfaceHit h = reintersect(tri, {{0.0f, 0.0f, 0.0f}, {0.25f, 0.5f, 2.0f}});  // through (0.25, 0.5, 2)
  CHECK(h.t == doctest::Approx(1.0f));
  CHECK(h.b1 == doctest::Approx(0.25f));
  CHECK(h.b2 == doctest::Approx(0.5f));
  h = reintersect(tri, {{0.0f, 0.0f, 0.0f}, {0.8f, 0.8f, 2.0f}});  // just outside the hypotenuse
  CHECK(h.t == doctest::Approx(1.0f));
  CHECK(h.b1 + h.b2 == doctest::Approx(1.0f));
  CHECK(h.b1 == doctest::Approx(0.5f));
}

TEST_CASE("plane gradients of linear functions are exact") {
  std::mt19937 rng(6);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  for (int i = 0; i < 1000; ++i) {
    const Vec3f e1{u(rng), u(rng), u(rng)}, e2{u(rng), u(rng), u(rng)};
    const Vec3f n = cross(e1, e2);
    if (dot(n, n) < 1e-2f) continue;
    const float d1 = u(rng), d2 = u(rng);
    const Vec3f g = plane_gradient(e1, e2, n, d1, d2);
    CHECK(dot(g, e1) == doctest::Approx(d1).epsilon(1e-4).scale(1.0));
    CHECK(dot(g, e2) == doctest::Approx(d2).epsilon(1e-4).scale(1.0));
    CHECK(std::abs(dot(g, n)) <= 1e-4f * std::sqrt(dot(n, n)) * std::sqrt(dot(g, g)) + 1e-6f);
  }
}

TEST_CASE("offset origins leave the surface on the normal side") {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f), mag(-6.0f, 3.0f);
  for (int i = 0; i < 10000; ++i) {
    const float s = std::exp2(mag(rng));
    const Vec3f p{s * u(rng), s * u(rng), s * u(rng)};
    const Vec3f n = normalize(Vec3f{u(rng), u(rng), u(rng)});
    const Vec3f o = offset_origin(p, n);
    CHECK(dot(o - p, n) > 0.0f);
    CHECK(norm(o - p) <= 1e-4f * (1.0f + norm(p)));
  }
}

TEST_CASE("sphere culling is conservative") {
  std::mt19937_64 rng(8);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  int kept = 0;
  for (int i = 0; i < 200000; ++i) {
    PoseRow row = identity_row();
    const double cx = 2 * u(rng), cy = 2 * u(rng), cz = 3 + u(rng), r = 0.05 + 0.5 * std::abs(u(rng));
    row.sphere = {static_cast<float>(cx), static_cast<float>(cy), static_cast<float>(cz), static_cast<float>(r)};
    const Vec3f d{static_cast<float>(u(rng)), static_cast<float>(u(rng)), 1.0f};
    const double tmax = 2.0 + 3.0 * std::abs(u(rng));
    // Exact test in fp64 on the fp32 inputs.
    const double dx = d.x, dy = d.y, dz = d.z, ccx = row.sphere.x, ccy = row.sphere.y, ccz = row.sphere.z;
    const double a = dx * dx + dy * dy + dz * dz, b = -(dx * ccx + dy * ccy + dz * ccz);
    const double c = ccx * ccx + ccy * ccy + ccz * ccz - static_cast<double>(row.sphere.w) * static_cast<double>(row.sphere.w);
    const double disc = b * b - a * c;
    const bool exact = disc >= 0 && (-b - std::sqrt(disc)) / a <= tmax && (-b + std::sqrt(disc)) / a >= 0;
    const bool got = sphere_overlaps(row, {0.0f, 0.0f, 0.0f}, d, static_cast<float>(tmax));
    if (exact) {
      ++kept;
      CHECK(got);
    }
  }
  CHECK(kept > 1000);
}

TEST_CASE("texture footprint of a fronto-parallel plane under a pinhole") {
  CameraView cam{};
  cam.model = CameraModel::kRadtan;
  const float f = 400.0f;
  cam.radtan = {f, f, 320.0f, 240.0f, 0, 0, 0, 0, 0, 10.0f, 0.0f};
  // Plane z = 2 with u = X / 0.5 (so du/dX = 2) and v = Y / 0.25.
  const Vec3f x{0.1f, -0.05f, 2.0f}, n{0.0f, 0.0f, -1.0f};
  Sym2f cov;
  REQUIRE(footprint_uv_covariance(cam, x, n, {2.0f, 0.0f, 0.0f}, {0.0f, 4.0f, 0.0f}, 0.1f, &cov));
  // d(uv)/d(pixel) = (Z / f) * (2, 4) on the diagonal.
  CHECK(cov.xx == doctest::Approx(0.1 * std::pow(2.0 * 2.0 / static_cast<double>(f), 2)).epsilon(1e-5));
  CHECK(cov.yy == doctest::Approx(0.1 * std::pow(2.0 * 4.0 / static_cast<double>(f), 2)).epsilon(1e-5));
  CHECK(std::abs(cov.xy) <= 1e-12f);
  // Edge-on planes have no footprint.
  CHECK_FALSE(footprint_uv_covariance(cam, {0.0f, 0.0f, 2.0f}, {1.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f},
                                      {0.0f, 4.0f, 0.0f}, 0.1f, &cov));
}

TEST_CASE("class word: the shadow-edge bit marks a shadow boundary, not a shadowed sample") {
  const auto edge = [](uint32_t occ_or, uint32_t occ_and, int32_t lights) {
    return (sample_class(0u, false, class_masks(occ_or, occ_and, lights), false) & kClassShadowEdge) != 0;
  };
  CHECK_FALSE(edge(0u, 0u, 1));  // lit everywhere
  CHECK_FALSE(edge(1u, 1u, 1));  // in the umbra everywhere
  CHECK(edge(1u, 0u, 1));        // a boundary of light 0
  CHECK_FALSE(edge(1u, 1u, 2));  // light 0 blocked everywhere, light 1 lit everywhere
  CHECK(edge(3u, 1u, 2));        // a boundary of light 1
  CHECK_FALSE(edge(0u, ~0u, 0)); // no hit sub-sample (occ_and keeps its identity) and no shadow lights
  CHECK_FALSE(edge(~0u, ~0u, 32));
  CHECK(edge(1u << 31, 0u, 32));  // light 31 folds onto bit 15
  // accuracy.mover_ao traced's AO bit (31) is no shadow light's bit: with 16 to 31 shadow lights it must not fold onto
  // light 15's (P2-F F9b review).
  CHECK_FALSE(edge(1u << 31, 0u, 16));
  CHECK_FALSE(edge(1u << 31, 0u, 31));
  CHECK(edge((1u << 31) | (1u << 15), 0u, 16));  // light 15's own boundary still is one
}
