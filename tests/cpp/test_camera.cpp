#include <doctest/doctest.h>

#include <cmath>
#include <numbers>
#include <random>
#include <string>

#include "core/camera/injective_domain.h"
#include "core/camera/kannala_brandt.h"
#include "core/camera/radtan.h"
#include "core/camera/ray_lut.h"
#include "core/camera/relative_illumination.h"
#include "core/camera/solid_angle.h"

using namespace balla;
using namespace balla::camera;

namespace {

// Pixel-area boundary (the image boundary of M2's tables) and the reach of the default
// sub-sample warp, x_max = 0.95 px (DESIGN §6.1).
constexpr double kPixelArea = 0.5;
constexpr double kSupport = 0.95;

// EVIMO2 Samsung DVS (scene10 calibration from meta['meta'], M2 §D.3).
RadtanCamera<double> samsung() {
  return {520.257996, 520.297974, 321.259003, 242.490005, 0.107593, -0.156436, 0.001883,
          -0.000958,  0.0,        0.0,        0.0};
}

// EVIMO2 left Prophesee, scene13 calibration.
RadtanCamera<double> prophesee_left() {
  return {556.573975, 556.086975, 321.846008, 225.617996, -0.10828, 0.206233, -0.00065,
          -0.000555,  0.0,        0.0,        0.0};
}

template <class Cam>
Vec2<double> fd_pixel(const Cam& c, Vec3<double> X) {
  Vec2<double> p{NAN, NAN};
  project(c, X, &p);
  return p;
}

}  // namespace

TEST_CASE("radtan injective domain of the EVIMO2 Samsung (M2 Theorem D1)") {
  const RadtanDomain d = radtan_injective_domain(samsung(), 640, 480, kPixelArea);
  CHECK(std::abs(d.r_mono - 1.1608) < 2e-4);  // M2 C5: 1.1608 (49.26 deg)
  CHECK(std::abs(d.r_img - 0.7688) < 2e-4);
  CHECK(std::abs(d.r_lim - 0.9225) < 2e-4);
  CHECK(d.r_lim == doctest::Approx(std::min(0.99 * d.r_mono, 1.2 * d.r_img)).epsilon(1e-15));
  // The sub-sample support reaches 0.45 px further out, so r_img grows slightly.
  const RadtanDomain ds = radtan_injective_domain(samsung(), 640, 480, kSupport);
  CHECK(ds.r_mono == d.r_mono);
  CHECK(ds.r_img > d.r_img);
  CHECK(ds.r_img < d.r_img + 2e-3);
  CHECK(ds.r_lim == doctest::Approx(1.2 * ds.r_img).epsilon(1e-15));
  RadtanCamera<double> c = samsung();
  c.r_lim = d.r_lim;
  // J_d is positive definite on the whole domain.
  for (int i = 0; i < 360; ++i) {
    for (int k = 1; k <= 50; ++k) {
      const double r = d.r_lim * k / 50, phi = 2 * std::numbers::pi * i / 360;
      const Sym2<double> j = distort_jacobian(c, Vec2<double>{r * std::cos(phi), r * std::sin(phi)});
      CHECK(j.xx * j.yy - j.xy * j.xy > 0.0);
      CHECK(j.xx > 0.0);
    }
  }
  // The fold-back ghost of M2 §D.3: a point 58 degrees off-axis (outside the field of view) lands
  // inside the image under the unrestricted polynomial, and is rejected by the domain.
  const double phi = std::atan2(-c.cy, -c.cx);
  const Vec2<double> xg{1.60 * std::cos(phi), 1.60 * std::sin(phi)};
  const Vec2<double> ghost = distort(c, xg);
  CHECK(c.fx * ghost.x + c.cx == doctest::Approx(154.5).epsilon(1e-3));
  CHECK(c.fy * ghost.y + c.cy == doctest::Approx(120.1).epsilon(1e-3));
  Vec2<double> pix{};
  CHECK_FALSE(project(c, Vec3<double>{xg.x, xg.y, 1.0}, &pix));
}

TEST_CASE("radtan domain of the EVIMO2 left Prophesee is capped by 1.2 r_img") {
  const RadtanDomain d = radtan_injective_domain(prophesee_left(), 640, 480, kPixelArea);
  CHECK(std::isinf(d.r_mono));
  CHECK(std::abs(d.r_img - 0.7360) < 2e-4);
  CHECK(d.r_lim == doctest::Approx(1.2 * d.r_img).epsilon(1e-15));
}

TEST_CASE("radtan round trips over every pixel and Jacobians against finite differences") {
  RadtanCamera<double> c = samsung();
  c.r_lim = radtan_injective_domain(c, 640, 480, kSupport).r_lim;
  RadtanCamera<float> cf{static_cast<float>(c.fx), static_cast<float>(c.fy), static_cast<float>(c.cx),
                         static_cast<float>(c.cy), static_cast<float>(c.k1), static_cast<float>(c.k2),
                         static_cast<float>(c.p1), static_cast<float>(c.p2), 0.0f,
                         static_cast<float>(c.r_lim), 0.0f};
  double worst64 = 0, worst32 = 0;
  for (int v = 0; v < 480; v += 3) {
    for (int u = 0; u < 640; u += 3) {
      Vec3<double> ray{};
      REQUIRE(unproject(c, Vec2<double>{double(u), double(v)}, &ray));
      Vec2<double> back{};
      REQUIRE(project(c, ray, &back));
      worst64 = std::max(worst64, std::hypot(back.x - u, back.y - v));
      Vec3<float> rayf{};
      REQUIRE(unproject(cf, Vec2<float>{float(u), float(v)}, &rayf));
      Vec2<float> backf{};
      REQUIRE(project(cf, rayf, &backf));
      worst32 = std::max(worst32, std::hypot(double(backf.x) - u, double(backf.y) - v));
    }
  }
  CHECK(worst64 < 1e-10);
  CHECK(worst32 < 2e-4);

  std::mt19937_64 rng(3);
  std::uniform_real_distribution<double> uni(-0.6, 0.6), depth(0.3, 3.0);
  for (int i = 0; i < 2000; ++i) {
    const double z = depth(rng);
    const Vec3<double> X{uni(rng) * z, uni(rng) * z, z};
    Mat23<double> j{};
    REQUIRE(project_jacobian(c, X, &j));
    const double e = 1e-6;
    for (int k = 0; k < 3; ++k) {
      Vec3<double> dp = X, dm = X;
      (&dp.x)[k] += e;
      (&dm.x)[k] -= e;
      const Vec2<double> fd = (0.5 / e) * (fd_pixel(c, dp) - fd_pixel(c, dm));
      CHECK(std::abs(j.m[0][k] - fd.x) < 1e-6 * c.fx / z);
      CHECK(std::abs(j.m[1][k] - fd.y) < 1e-6 * c.fy / z);
    }
  }
}

TEST_CASE("Kannala-Brandt: fold angle, round trips beyond 90 degrees, Jacobian") {
  // Fold example of M2 §D.3: theta_d' = 0 at 109.02 degrees.
  const KannalaBrandtCamera<double> fold{300, 300, 160, 120, -0.08, -0.002, 0, 0, 0};
  const KannalaBrandtDomain dfold = kannala_brandt_injective_domain(fold, 320, 240, kSupport);
  CHECK(dfold.theta_mono * 180 / std::numbers::pi == doctest::Approx(109.02).epsilon(1e-4));
  // An image whose corners lie beyond the fold has no injective domain.
  CHECK_THROWS_AS(kannala_brandt_injective_domain(KannalaBrandtCamera<double>{300, 300, 320, 240, -0.08,
                                                                               -0.002, 0, 0, 0},
                                                  640, 480, kSupport),
                  std::domain_error);

  KannalaBrandtCamera<double> c{300.0, 300.5, 640.0, 480.0, -0.013, 0.0021, -0.0009, 0.0001, 0.0};
  const KannalaBrandtDomain d = kannala_brandt_injective_domain(c, 1280, 960, kSupport);
  CHECK(d.theta_mono == doctest::Approx(std::numbers::pi));
  CHECK(d.theta_img > std::numbers::pi / 2);  // a fisheye that sees behind its image plane
  c.theta_lim = d.theta_lim;
  std::mt19937_64 rng(4);
  std::uniform_real_distribution<double> th(0.0, d.theta_lim), ph(0, 2 * std::numbers::pi), rr(0.3, 3);
  for (int i = 0; i < 5000; ++i) {
    const double t = th(rng), p = ph(rng), r = rr(rng);
    const Vec3<double> X{r * std::sin(t) * std::cos(p), r * std::sin(t) * std::sin(p), r * std::cos(t)};
    Vec2<double> pix{};
    REQUIRE(project(c, X, &pix));
    Vec3<double> ray{};
    REQUIRE(unproject(c, pix, &ray));
    CHECK(norm(ray - (1.0 / r) * X) < 1e-12);
    if (t < 1.4 && t > 0.01) {
      Mat23<double> j{};
      REQUIRE(project_jacobian(c, X, &j));
      const double e = 1e-7;
      for (int k = 0; k < 3; ++k) {
        Vec3<double> dp = X, dm = X;
        (&dp.x)[k] += e;
        (&dm.x)[k] -= e;
        const Vec2<double> fd = (0.5 / e) * (fd_pixel(c, dp) - fd_pixel(c, dm));
        CHECK(std::abs(j.m[0][k] - fd.x) < 2e-6 * c.fx / r);
        CHECK(std::abs(j.m[1][k] - fd.y) < 2e-6 * c.fy / r);
      }
    }
  }
  // On the optical axis the Jacobian is the pinhole one.
  Mat23<double> j{};
  REQUIRE(project_jacobian(c, Vec3<double>{0, 0, 2}, &j));
  CHECK(j.m[0][0] == doctest::Approx(c.fx / 2));
  CHECK(j.m[1][2] == 0.0);
}

TEST_CASE("ray LUT interpolates unprojected rays to well below a millipixel") {
  RadtanCamera<double> c = samsung();
  c.r_lim = radtan_injective_domain(c, 640, 480, kSupport).r_lim;
  const double max_offset = kSupport;
  const RayLut<float> lut(c, 640, 480, max_offset, 2);
  const RayLutView<float> view = lut.view();
  std::mt19937_64 rng(5);
  std::uniform_real_distribution<double> u(-max_offset, 639 + max_offset), v(-max_offset, 479 + max_offset);
  double worst = 0;
  for (int i = 0; i < 100000; ++i) {
    const Vec2<double> pix{u(rng), v(rng)};
    const Vec3<float> ray = lut_ray(view, cast<float>(pix));
    REQUIRE(std::isfinite(ray.x));
    CHECK(ray.z == doctest::Approx(1.0f));
    Vec2<double> back{};
    REQUIRE(project(c, cast<double>(ray), &back));
    worst = std::max(worst, std::hypot(back.x - pix.x, back.y - pix.y));
  }
  CHECK(worst < 1.3e-4);  // bilinear error 6e-5 px plus fp32 rounding; 9.4e-5 px measured
}

namespace {

// Worst fp64 reprojection error (px) of an fp32 ray LUT over the sub-sample support.
template <class Cam>
double lut_round_trip_px(const Cam& c, int w, int h, int spp) {
  const RayLut<float> lut(c, w, h, kSupport, spp);
  const RayLutView<float> view = lut.view();
  std::mt19937_64 rng(5);
  std::uniform_real_distribution<double> u(-kSupport, w - 1 + kSupport), v(-kSupport, h - 1 + kSupport);
  double worst = 0;
  for (int i = 0; i < 100000; ++i) {
    const Vec2<float> pf = cast<float>(Vec2<double>{u(rng), v(rng)});
    Vec2<double> back{};
    REQUIRE(project(c, cast<double>(lut_ray(view, pf)), &back));
    worst = std::max(worst, std::hypot(back.x - double(pf.x), back.y - double(pf.y)));
  }
  return worst;
}

}  // namespace

TEST_CASE("ray LUT accuracy for Kannala-Brandt cameras and near the fold") {
  // The fisheye above, which sees beyond 90 degrees: 2.5e-4 px measured.
  KannalaBrandtCamera<double> fisheye{300.0, 300.5, 640.0, 480.0, -0.013, 0.0021, -0.0009, 0.0001, 0.0};
  fisheye.theta_lim = kannala_brandt_injective_domain(fisheye, 1280, 960, kSupport).theta_lim;
  CHECK(lut_round_trip_px(fisheye, 1280, 960, 2) < 3.5e-4);
  // Strong distortion per pixel on a 70 x 70 image (the fold sweep below): 2.0e-3 px at f = 45.
  KannalaBrandtCamera<double> strong{45, 45, 34.5, 34.5, -0.08, -0.002, 0, 0, 0};
  strong.theta_lim = kannala_brandt_injective_domain(strong, 70, 70, kSupport).theta_lim;
  CHECK(lut_round_trip_px(strong, 70, 70, 2) < 3e-3);
  // At 96 % of the fold the error reaches 5.8e-2 px; more nodes per px reduce it (2.2e-2 px at 4).
  KannalaBrandtCamera<double> near_fold{38.6, 38.6, 34.5, 34.5, -0.08, -0.002, 0, 0, 0};
  const KannalaBrandtDomain d = kannala_brandt_injective_domain(near_fold, 70, 70, kSupport);
  CHECK(d.theta_img / d.theta_mono > 0.95);
  near_fold.theta_lim = d.theta_lim;
  const double err2 = lut_round_trip_px(near_fold, 70, 70, 2);
  const double err4 = lut_round_trip_px(near_fold, 70, 70, 4);
  CHECK(err2 < 8e-2);
  CHECK(err4 < 3e-2);
  CHECK(err4 < 0.5 * err2);
}

TEST_CASE("ray LUT extrapolates from the border cell at any distance") {
  // The cell is clamped before the int32 conversion, which is undefined beyond 2^31 (the host
  // took cell 0 there and the device the last cell).
  RadtanCamera<double> c = samsung();
  c.r_lim = radtan_injective_domain(c, 64, 48, kSupport).r_lim;
  const RayLut<double> lut(c, 64, 48, kSupport, 2);
  const RayLutView<double> view = lut.view();
  const double y = view.v0 + 10;  // node row 20, ty = 0
  for (const double x : {1e5, 3e9, -3e9}) {
    const double fx = (x - view.u0) * view.samples_per_px;
    const int32_t i = x > 0 ? view.nx - 2 : 0;
    const double tx = fx - i;
    const Vec3<double>* r = view.rays + 20 * view.nx + i;
    const Vec3<double> expected = (1 - tx) * r[0] + tx * r[1];
    const Vec3<double> ray = lut_ray(view, Vec2<double>{x, y});
    CHECK(ray.x == doctest::Approx(expected.x).epsilon(1e-12));
    CHECK(ray.y == doctest::Approx(expected.y).epsilon(1e-12));
    CHECK(ray.z == doctest::Approx(expected.z).epsilon(1e-12));
  }
}

namespace {

// Every sub-sample position of a camera accepted near its fold must get a finite ray: the
// domain covers the support, and the LUT either covers it with finite nodes or refuses (its
// outermost nodes lie up to 1/samples_per_px beyond the support). Returns whether the LUT was built.
template <class Cam>
bool check_support_is_covered(const Cam& c, int width, int height) {
  const double u1 = width - 1 + kSupport, v1 = height - 1 + kSupport;
  for (int i = 0; i <= 40; ++i) {
    const double t = i / 40.0;
    for (const Vec2<double> pix : {Vec2<double>{-kSupport + t * (u1 + kSupport), -kSupport},
                                   Vec2<double>{-kSupport + t * (u1 + kSupport), v1},
                                   Vec2<double>{-kSupport, -kSupport + t * (v1 + kSupport)},
                                   Vec2<double>{u1, -kSupport + t * (v1 + kSupport)}}) {
      Vec3<double> ray{};
      REQUIRE(unproject(c, pix, &ray));
    }
  }
  try {
    const RayLut<double> lut(c, width, height, kSupport, 2);
    for (const Vec3<double>& ray : lut.rays()) REQUIRE(std::isfinite(ray.x + ray.y + ray.z));
    return true;
  } catch (const std::domain_error&) {
    return false;
  }
}

struct SweepCounts {
  int rejected = 0, lut_refused = 0, covered = 0;
};

}  // namespace

TEST_CASE("cameras near the fold: rejected, or every sub-sample position has a ray") {
  // Radial-only radtan with r* = 1.171 and r_d(r*) = 1.0013. The reviewer's 424 x 424 cases:
  // before the support was checked, f = 299.5 was accepted with r_lim < r_img and f = 300 gave
  // NaN rays for corner sub-samples.
  CHECK_THROWS_AS(radtan_injective_domain({299.5, 299.5, 211.5, 211.5, 0.1, -0.15, 0, 0, 0, 0, 0},
                                          424, 424, kSupport),
                  std::domain_error);
  CHECK_THROWS_AS(radtan_injective_domain({300, 300, 211.5, 211.5, 0.1, -0.15, 0, 0, 0, 0, 0},
                                          424, 424, kSupport),
                  std::domain_error);
  // The same lens on a 106 x 106 image folds at f = 75.5; the LUT's outer nodes leave the domain
  // up to f = 75.59. The sweep crosses both thresholds and the 1.2 r_img regime above.
  SweepCounts radtan;
  for (const double f : {75.0, 75.25, 75.45, 75.5, 75.55, 75.6, 75.7, 76.0, 80.0, 90.0, 120.0}) {
    RadtanCamera<double> c{f, f, 52.5, 52.5, 0.1, -0.15, 0, 0, 0, 0, 0};
    RadtanDomain d{};
    try {
      d = radtan_injective_domain(c, 106, 106, kSupport);
    } catch (const std::domain_error&) {
      ++radtan.rejected;
      continue;
    }
    CHECK(d.r_img <= d.r_lim);
    CHECK(d.r_lim < d.r_mono);
    c.r_lim = d.r_lim;
    ++(check_support_is_covered(c, 106, 106) ? radtan.covered : radtan.lut_refused);
  }
  CHECK(radtan.rejected > 0);
  CHECK(radtan.lut_refused > 0);
  CHECK(radtan.covered > 0);

  // Kannala-Brandt with theta_mono = 109.02 degrees: the reviewer's 276 x 276 case, then a
  // 70 x 70 image that folds at f = 38.52 (LUT nodes up to f = 38.57).
  CHECK_THROWS_AS(kannala_brandt_injective_domain({150, 150, 137.5, 137.5, -0.08, -0.002, 0, 0, 0},
                                                  276, 276, kSupport),
                  std::domain_error);
  SweepCounts kb;
  for (const double f : {38.0, 38.4, 38.5, 38.55, 38.6, 38.8, 40.0, 45.0, 60.0}) {
    KannalaBrandtCamera<double> c{f, f, 34.5, 34.5, -0.08, -0.002, 0, 0, 0};
    KannalaBrandtDomain d{};
    try {
      d = kannala_brandt_injective_domain(c, 70, 70, kSupport);
    } catch (const std::domain_error&) {
      ++kb.rejected;
      continue;
    }
    CHECK(d.theta_img <= d.theta_lim);
    CHECK(d.theta_lim < d.theta_mono);
    c.theta_lim = d.theta_lim;
    ++(check_support_is_covered(c, 70, 70) ? kb.covered : kb.lut_refused);
  }
  CHECK(kb.rejected > 0);
  CHECK(kb.lut_refused > 0);
  CHECK(kb.covered > 0);
}

TEST_CASE("injective domain and ray LUT reject invalid arguments") {
  CHECK_THROWS_AS(radtan_injective_domain(samsung(), 0, 480, kSupport), std::invalid_argument);
  CHECK_THROWS_AS(radtan_injective_domain(samsung(), 640, 480, -1.0), std::invalid_argument);
  CHECK_THROWS_AS(radtan_injective_domain(samsung(), 640, 480, NAN), std::invalid_argument);
  // A LUT reaching further than the domain was built for is refused rather than filled with NaN.
  RadtanCamera<double> c{299.5, 299.5, 211.5, 211.5, 0.1, -0.15, 0, 0, 0, 0, 0};
  c.r_lim = radtan_injective_domain(c, 424, 424, 0.0).r_lim;
  CHECK_THROWS_AS(RayLut<double>(c, 424, 424, 3.0, 2), std::domain_error);
  // Node grids beyond 2^24 per axis are refused (the lookup clamps cells in T), as is an
  // infinite offset.
  const RadtanCamera<double> pinhole{100, 100, 1.5, 1.5, 0, 0, 0, 0, 0, 10.0, 0};
  CHECK_THROWS_AS(RayLut<double>(pinhole, 4, 4, 0.5, 1 << 30), std::invalid_argument);
  CHECK_THROWS_AS(RayLut<double>(pinhole, 4, 4, INFINITY, 2), std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
// V-LAW (P2-F F2, PHASE2 F079): the natural-vignetting law cos(theta) dOmega/dA (core/camera/solid_angle.h) against
// the closed forms and a finite-difference solid angle of the unprojection, and the three EVIMO2 calibrations of
// scene13_dyn_test_00 (python/ballasim/scene/authoring.py CAMERA_PRESETS) against the plan's script
// (ballasim_out/p2_plans/p2f_work/scripts/r2_vignetting_3cam.py -> r2/r2_vignetting_3cam.json).
// ---------------------------------------------------------------------------------------------

namespace {

Vec3<double> unit(Vec3<double> r) { return (1.0 / norm(r)) * r; }

// The gain from a finite-difference solid angle: dOmega/dA = |(dw/du x dw/dv).w| of the unit rays w(u, v), central
// differences of the unprojection at h px, normalized on the axis (the principal point) the same way. Independent of
// project_jacobian.
template <class Cam>
double fd_gain(const Cam& c, Vec2<double> px, double h = 1e-3) {
  const auto density = [&](Vec2<double> q) {
    Vec3<double> r[4];
    const Vec2<double> d[4] = {{h, 0}, {-h, 0}, {0, h}, {0, -h}};
    for (int i = 0; i < 4; ++i) {
      REQUIRE(unproject(c, q + d[i], &r[i]));
      r[i] = unit(r[i]);
    }
    Vec3<double> w{};
    REQUIRE(unproject(c, q, &w));
    w = unit(w);
    return std::abs(dot(cross((0.5 / h) * (r[0] - r[1]), (0.5 / h) * (r[2] - r[3])), w));
  };
  Vec3<double> w{};
  REQUIRE(unproject(c, px, &w));
  return unit(w).z * density(px) / density(Vec2<double>{c.cx, c.cy});
}

template <class Cam>
double law(const Cam& c, Vec2<double> px) {
  Vec3<double> w{};
  REQUIRE(unproject(c, px, &w));
  double g = NAN;
  REQUIRE(vignetting_gain(c, unit(w), &g));
  return g;
}

// cos^4(theta) / det J_d at the undistorted point (x_n, y_n) of a pixel: the radtan closed form.
double radtan_closed(const RadtanCamera<double>& c, Vec2<double> px) {
  Vec3<double> r{};
  REQUIRE(unproject(c, px, &r));
  const Sym2<double> j = distort_jacobian(c, Vec2<double>{r.x, r.y});
  const double s = 1.0 + r.x * r.x + r.y * r.y;
  return 1.0 / (s * s) / (j.xx * j.yy - j.xy * j.xy);
}

// The scene13_dyn_test_00 calibrations (authoring.py CAMERA_PRESETS evimo2_left, evimo2_right, evimo2_samsung).
struct Evimo2Law {
  const char* name;
  RadtanCamera<double> cam;
  // r2_vignetting_3cam.json: law_corners at pixels (0, 0), (639, 0), (0, 479), (639, 479) (the third is the
  // max-radius corner on all three), law_range and det_jd_range over every pixel centre; whether the law is darker
  // than cos^4 at that corner (det J_d > 1 there).
  double corners[4], law_lo, law_hi, det_lo, det_hi;
  bool darker;
};

const Evimo2Law kEvimo2Law[3] = {
    {"left",
     {556.573975, 556.086975, 321.846008, 225.617996, -0.10828, 0.206233, -0.00065, -0.000555, 0.0, 0.0, 0.0},
     {0.40515652477410796, 0.41315741856086197, 0.3739363617458815, 0.3811223327406977},
     0.3739363617458815, 1.0000057156125968, 0.9595932285710392, 1.1274513366457282, true},
    {"right",
     {554.585022, 554.210022, 327.556, 202.320999, -0.094841, 0.187073, 0.001747, -0.001038, 0.0, 0.0, 0.0},
     {0.4207004823264279, 0.449224448684788, 0.3398393410343088, 0.36131039004435367},
     0.3398393410343088, 1.000040744284099, 0.9615087747567418, 1.174023434625777, true},
    {"samsung",
     {520.189026, 520.166016, 324.177002, 239.169998, 0.102258, -0.139354, 0.000379, 0.000844, 0.0, 0.0, 0.0},
     {0.41950979632581714, 0.42371405959217195, 0.41831464252701395, 0.42251581420271106},
     0.41831464252701395, 1.000003677873521, 0.9451235362585879, 1.0543398652950906, false},
};

}  // namespace

TEST_CASE("V-LAW: the pinhole's law is cos^4 (any focal lengths and principal point)") {
  const RadtanCamera<double> c{400.0, 430.0, 300.0, 200.0, 0, 0, 0, 0, 0, INFINITY, 0.0};
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<double> th(0.0, 1.45), ph(0, 2 * std::numbers::pi);
  double worst = 0;
  for (int i = 0; i < 20000; ++i) {
    const double t = th(rng), p = ph(rng);
    const Vec3<double> w{std::sin(t) * std::cos(p), std::sin(t) * std::sin(p), std::cos(t)};
    double g = NAN;
    REQUIRE(vignetting_gain(c, w, &g));
    const double c4 = std::pow(std::cos(t), 4);
    worst = std::max(worst, std::abs(g / c4 - 1));
  }
  CHECK(worst < 1e-7);
  double g = NAN;
  REQUIRE(vignetting_gain(c, Vec3<double>{0, 0, 1}, &g));
  CHECK(g == 1.0);
  // The density itself: cos^3(theta) / (fx fy) sr per px^2.
  double rho = NAN;
  const Vec3<double> w = unit(Vec3<double>{0.3, -0.4, 1.0});
  REQUIRE(solid_angle_density(c, w, &rho));
  CHECK(rho == doctest::Approx(std::pow(w.z, 3) / (c.fx * c.fy)).epsilon(1e-14));
  // 90 degrees or more off axis, and outside the domain: no gain.
  CHECK_FALSE(vignetting_gain(c, Vec3<double>{1, 0, 0}, &g));
  CHECK_FALSE(vignetting_gain(c, Vec3<double>{0.6, 0, -0.8}, &g));
  RadtanCamera<double> bounded = c;
  bounded.r_lim = 0.5;
  CHECK_FALSE(vignetting_gain(bounded, unit(Vec3<double>{0.6, 0, 1}), &g));
  CHECK(vignetting_gain(bounded, unit(Vec3<double>{0.4, 0, 1}), &g));
}

TEST_CASE("V-LAW: radtan is cos^4 / det J_d and the finite-difference solid angle") {
  for (const Evimo2Law& e : kEvimo2Law) {
    INFO(e.name);
    RadtanCamera<double> c = e.cam;
    c.r_lim = radtan_injective_domain(c, 640, 480, kSupport).r_lim;
    double worst_closed = 0, worst_fd = 0;
    for (int v = 0; v < 480; v += 7) {
      for (int u = 0; u < 640; u += 7) {
        const Vec2<double> px{double(u), double(v)};
        const double g = law(c, px);
        worst_closed = std::max(worst_closed, std::abs(g / radtan_closed(c, px) - 1));
        worst_fd = std::max(worst_fd, std::abs(g / fd_gain(c, px) - 1));
      }
    }
    CHECK(worst_closed < 1e-12);
    CHECK(worst_fd < 1e-6);
  }
}

TEST_CASE("V-LAW: Kannala-Brandt is cos(theta) sin(theta) / (theta_d theta_d') and the finite-difference law") {
  KannalaBrandtCamera<double> c{300.0, 300.5, 640.0, 480.0, -0.013, 0.0021, -0.0009, 0.0001, 0.0};
  c.theta_lim = kannala_brandt_injective_domain(c, 1280, 960, kSupport).theta_lim;
  double worst_closed = 0, worst_fd = 0, theta_max = 0;
  for (int v = 0; v < 960; v += 13) {
    for (int u = 0; u < 1280; u += 13) {
      const Vec2<double> px{double(u), double(v)};
      Vec3<double> w{};
      REQUIRE(unproject(c, px, &w));
      const double t = std::atan2(std::hypot(w.x, w.y), w.z);
      if (!(t < 1.5)) continue;  // the law's cos(theta) vanishes at 90 degrees (the pupil edge-on)
      theta_max = std::max(theta_max, t);
      const double g = law(c, px);
      const double closed = t > 0 ? std::cos(t) * std::sin(t) / (theta_distorted(c, t) *
                                                                theta_distorted_derivative(c, t))
                                  : 1.0;
      worst_closed = std::max(worst_closed, std::abs(g / closed - 1));
      worst_fd = std::max(worst_fd, std::abs(g / fd_gain(c, px) - 1));
    }
  }
  CHECK(theta_max > 1.4);
  CHECK(worst_closed < 1e-12);
  CHECK(worst_fd < 1e-6);
  // Near the axis (the Jacobian's pinhole branch) the law tends to 1.
  CHECK(law(c, Vec2<double>{c.cx, c.cy}) == 1.0);
  CHECK(law(c, Vec2<double>{c.cx + 1e-9, c.cy}) == doctest::Approx(1.0).epsilon(1e-12));
  double g = NAN;
  CHECK_FALSE(vignetting_gain(c, unit(Vec3<double>{1.0, 0.0, -0.1}), &g));  // beyond 90 degrees: no gain
}

TEST_CASE("V-LAW: the three EVIMO2 calibrations reproduce r2_vignetting_3cam.json") {
  for (const Evimo2Law& e : kEvimo2Law) {
    INFO(e.name);
    RadtanCamera<double> c = e.cam;
    c.r_lim = radtan_injective_domain(c, 640, 480, kSupport).r_lim;
    const Vec2<double> corners[4] = {{0, 0}, {639, 0}, {0, 479}, {639, 479}};
    for (int i = 0; i < 4; ++i) CHECK(std::abs(law(c, corners[i]) - e.corners[i]) < 1e-6);
    double lo = INFINITY, hi = 0, det_lo = INFINITY, det_hi = 0;
    for (int v = 0; v < 480; ++v) {
      for (int u = 0; u < 640; ++u) {
        Vec3<double> r{};
        REQUIRE(unproject(c, Vec2<double>{double(u), double(v)}, &r));
        double g = NAN;
        REQUIRE(vignetting_gain(c, unit(r), &g));
        lo = std::min(lo, g);
        hi = std::max(hi, g);
        const Sym2<double> j = distort_jacobian(c, Vec2<double>{r.x, r.y});
        const double det = j.xx * j.yy - j.xy * j.xy;
        det_lo = std::min(det_lo, det);
        det_hi = std::max(det_hi, det);
      }
    }
    CHECK(std::abs(lo - e.law_lo) < 1e-6);
    CHECK(std::abs(hi - e.law_hi) < 1e-6);
    CHECK(std::abs(det_lo - e.det_lo) < 1e-6);
    CHECK(std::abs(det_hi - e.det_hi) < 1e-6);
    // At the max-radius corner the law is darker than cos^4 on both Prophesees, lighter on the Samsung.
    Vec3<double> w{};
    REQUIRE(unproject(c, corners[2], &w));
    CHECK((law(c, corners[2]) < std::pow(unit(w).z, 4)) == e.darker);
  }
}

TEST_CASE("relative illumination (P2-F F2b): G = 1 + c2 r^2 + c4 r^4 over the farthest corner, positive on the image") {
  // r_corner: the farthest corner pixel centre from the principal point (EVIMO2 left: pixel (0, 479)).
  CHECK(corner_radius_sq(640, 480, 321.846008, 225.617996) ==
        doctest::Approx(321.846008 * 321.846008 + 253.382004 * 253.382004).epsilon(1e-15));
  CHECK(corner_radius_sq(64, 48, 31.5, 23.5) == 31.5 * 31.5 + 23.5 * 23.5);
  CHECK(corner_radius_sq(1, 1, 0.0, 0.0) == 0.0);
  CHECK(relative_illumination(-0.15, 0.0, 0.0) == 1.0);
  CHECK(relative_illumination(-0.15, 0.0, 1.0) == 0.85);
  CHECK(relative_illumination(-0.2, 0.05, 0.25) == doctest::Approx(1.0 - 0.05 + 0.003125).epsilon(1e-15));
  // The minimum over u = r^2 in [0, 1]: an end, or the vertex of a convex G inside (brute force on a fine grid).
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<double> c(-2.0, 2.0);
  for (int i = 0; i < 2000; ++i) {
    const double c2 = c(rng), c4 = c(rng);
    double lo = INFINITY;
    for (int k = 0; k <= 100000; ++k) lo = std::min(lo, relative_illumination(c2, c4, k * 1e-5));
    INFO(c2 << " " << c4);
    CHECK(relative_illumination_min(c2, c4) <= lo + 1e-15);
    CHECK(relative_illumination_min(c2, c4) >= lo - 1e-9);
  }
  CHECK(relative_illumination_min(-1.6, 0.8) == doctest::Approx(0.2).epsilon(1e-15));  // vertex at u = 1
  CHECK(relative_illumination_min(-2.4, 1.6) == doctest::Approx(0.1).epsilon(1e-15));  // vertex at u = 0.75
  // The accepted values: [] and finite [c2, c4] with G > 0 on the image.
  CHECK(relative_illumination_error({}) == nullptr);
  CHECK(relative_illumination_error({-0.15, 0.0}) == nullptr);
  CHECK(relative_illumination_error({0.3, -0.1}) == nullptr);  // brighter towards the corner: allowed, G > 0
  CHECK(std::string(relative_illumination_error({-0.15})) == "expected [] or [c2, c4]");
  CHECK(std::string(relative_illumination_error({-0.15, 0.0, 0.0})) == "expected [] or [c2, c4]");
  CHECK(std::string(relative_illumination_error({NAN, 0.0})) == "the coefficients must be finite");
  CHECK(relative_illumination_error({-1.0, 0.0}) != nullptr);  // G(1) = 0
  CHECK(relative_illumination_error({-2.4, 1.6}) == nullptr);  // min 0.1 inside
  CHECK(relative_illumination_error({-2.6, 1.6}) != nullptr);  // G(1) = 0 > its vertex value -0.056
  CHECK(relative_illumination_error({-2.0, 1.0}) != nullptr);  // vertex at u = 1, G = 0
}
