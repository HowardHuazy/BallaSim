// U5 and I-M3 (route S, synthetic) of docs/design/phase3/frames.md §8.1, §8.3: route S's resampler (§4.7.2).
//
// The scene of E3b/F3 (frames/calibration_timing/e3/e3b_resample.py, frames/design_doc/exp/f3_resample_s8.py): a
// 10:1 checkerboard rotated 17 degrees with 6 px squares on the normalized plane, seen by the EVIMO2-left radtan
// camera and by the same intrinsics without distortion. Each source pixel is the box integral of its cell (24 x 24
// strata, as Blender's BOX film); the truth is F3's: per pixel 48 x 48 box strata x 3 x 3 Gauss-Hermite nodes of
// box(a) ⊛ G(sigma) through our exact unprojection. Errors are ln(out) - ln(truth) over 32 x 32 crops, p99 as
// numpy's linear quantile. U5 (the default, local sigma_r): radtan corner s = 4 RMS <= 0.006, p99 <= 0.025; pinhole
// <= 0.002; cells at sigma = 0 within 1e-6 of the exact box integral; weights sum to 1 within 1e-7; NaN outside the
// lens domain; an ROI operator equals the full operator's rows. Also a Kannala-Brandt corner, FRM-KB-WIDE, the
// overscan source, tiles and refusals. I-M3 (synthetic): 99 small discs (E3c's points) through an s = 4 route S
// render keep their centroids within 0.015 px RMS. The gate ("E3b/F3 reproduced", skipped by default; run with
// -tc='*gate*' --no-skip) reproduces F3's 16 table entries with the nominal sigma_r (Blur::kNominal), compares both
// rules with BallaSim's exact truncated response, and reports the full-frame operator's size and build time.
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/rng/warp_table.h"
#include "frames/resample.h"
#include "render/thread_pool.h"

namespace {

using namespace balla;
using frames::Blur;
using frames::LensModel;
using frames::ResampleOperator;
using frames::ResampleSpec;
using frames::Roi;
using frames::Rule;
using frames::SourceCamera;

constexpr double kFx = 556.573975, kFy = 556.086975, kCx = 321.846008, kCy = 225.617996;
constexpr double kSupport = 0.5 + 3 * 0.15;  // render::support_px at a = 1, sigma = 0.15, n = 3

render::Camera evimo2_left(bool distorted, int32_t w = 640, int32_t h = 480, double scale = 1.0) {
  render::Camera c;
  c.name = "left";
  c.model = render::CameraModelKind::kRadtan;
  c.width = w, c.height = h;
  c.fx = kFx * scale, c.fy = kFy * scale, c.cx = kCx * scale, c.cy = kCy * scale;
  c.dist = distorted ? std::vector<double>{-0.10828, 0.206233, -0.00065, -0.000555, 0.0} : std::vector<double>{};
  return c;
}

using Radiance = std::function<double(double, double)>;  // on the normalized plane

// F3's checkerboard: 10 where floor(a) + floor(b) is even; squares of sq on the normalized plane (6 px at fx).
double checker_sq(double xn, double yn, double sq) {
  const double ang = 17.0 * std::numbers::pi / 180.0;
  const double a = (std::cos(ang) * xn + std::sin(ang) * yn) / sq;
  const double b = (-std::sin(ang) * xn + std::cos(ang) * yn) / sq;
  const auto n = static_cast<int64_t>(std::floor(a)) + static_cast<int64_t>(std::floor(b));
  return (n & 1) == 0 ? 10.0 : 1.0;
}
double checker(double xn, double yn) { return checker_sq(xn, yn, 6.0 / kFx); }

double stratum(int i, int n) { return (i + 0.5) / n - 0.5; }

// Source pixels of r: each the mean of L over n x n strata of its cell.
void render_into(std::vector<float>& img, const SourceCamera& sc, const Radiance& L, int n, Roi r) {
  render::ThreadPool pool(16);
  pool.parallel_for(static_cast<size_t>(r.y1 - r.y0), 1, [&](size_t b, size_t e) {
    for (size_t jj = b; jj < e; ++jj) {
      const int32_t j = r.y0 + static_cast<int32_t>(jj);
      for (int32_t i = r.x0; i < r.x1; ++i) {
        double s = 0;
        for (int b2 = 0; b2 < n; ++b2) {
          for (int a2 = 0; a2 < n; ++a2) {
            s += L((i + stratum(a2, n) - sc.cx) / sc.fx, (j + stratum(b2, n) - sc.cy) / sc.fy);
          }
        }
        img[static_cast<size_t>(j) * sc.width + i] = static_cast<float>(s / (n * n));
      }
    }
  });
}
std::vector<float> render_source(const SourceCamera& sc, const Radiance& L, int n) {
  std::vector<float> img(static_cast<size_t>(sc.width) * sc.height, 0.0f);
  render_into(img, sc, L, n, {0, 0, sc.width, sc.height});
  return img;
}

void must_unproject(const LensModel& m, double u, double v, Vec3<double>* ray);

bool unproject(const LensModel& m, double u, double v, Vec3<double>* ray) {
  return m.kind == render::CameraModelKind::kRadtan ? camera::unproject(m.radtan, Vec2<double>{u, v}, ray)
                                                     : camera::unproject(m.kb, Vec2<double>{u, v}, ray);
}
void must_unproject(const LensModel& m, double u, double v, Vec3<double>* ray) {  // thread-safe failure
  if (!unproject(m, u, v, ray)) throw std::runtime_error("truth: a point without a preimage");
}

// F3's truth: 48 x 48 box strata (scaled by a) x 3 x 3 Gauss-Hermite nodes of N(0, sigma^2) per pixel.
std::vector<double> truth_gh3(const LensModel& lens, Roi crop, double a, double sigma, const Radiance& L,
                              int strata = 48) {
  const double gx[3] = {-std::sqrt(3.0), 0.0, std::sqrt(3.0)}, gw[3] = {1.0 / 6, 2.0 / 3, 1.0 / 6};
  std::vector<double> o, w;
  for (int i = 0; i < strata; ++i) {
    for (int j = 0; j < 3; ++j) o.push_back(stratum(i, strata) * a + gx[j] * sigma), w.push_back(gw[j] / strata);
  }
  const int32_t W = crop.x1 - crop.x0, H = crop.y1 - crop.y0;
  std::vector<double> out(static_cast<size_t>(W) * H);
  render::ThreadPool pool(16);
  pool.parallel_for(out.size(), 4, [&](size_t b, size_t e) {
    for (size_t p = b; p < e; ++p) {
      const double u = crop.x0 + static_cast<double>(p % W), v = crop.y0 + static_cast<double>(p / W);
      double s = 0;
      for (size_t iy = 0; iy < o.size(); ++iy) {
        double sx = 0;
        for (size_t ix = 0; ix < o.size(); ++ix) {
          Vec3<double> r{};
          must_unproject(lens, u + o[ix], v + o[iy], &r);
          sx += w[ix] * L(r.x / r.z, r.y / r.z);
        }
        s += w[iy] * sx;
      }
      out[p] = s;
    }
  });
  return out;
}

// BallaSim's own response (rng::PixelResponse: box(a) ⊛ G(sigma) truncated at a/2 + 3 sigma) by its inverse CDF at
// n midpoint strata per axis.
std::vector<double> truth_exact(const LensModel& lens, Roi crop, double a, double sigma, const Radiance& L, int n) {
  const rng::PixelResponse h(a, sigma, 3.0);
  std::vector<double> o(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double u = (i + 0.5) / n;
    double lo = -h.x_max(), hi = h.x_max();
    for (int it = 0; it < 80; ++it) (h.cdf(0.5 * (lo + hi)) < u ? lo : hi) = 0.5 * (lo + hi);
    o[static_cast<size_t>(i)] = 0.5 * (lo + hi);
  }
  const int32_t W = crop.x1 - crop.x0, H = crop.y1 - crop.y0;
  std::vector<double> out(static_cast<size_t>(W) * H);
  render::ThreadPool pool(16);
  pool.parallel_for(out.size(), 4, [&](size_t b, size_t e) {
    for (size_t p = b; p < e; ++p) {
      const double u = crop.x0 + static_cast<double>(p % W), v = crop.y0 + static_cast<double>(p / W);
      double s = 0;
      for (double dy : o) {
        for (double dx : o) {
          Vec3<double> r{};
          must_unproject(lens, u + dx, v + dy, &r);
          s += L(r.x / r.z, r.y / r.z);
        }
      }
      out[p] = s / (static_cast<double>(n) * n);
    }
  });
  return out;
}

struct Err {
  double rms, p99;
};
Err log_error(const std::vector<float>& out, const std::vector<double>& truth) {
  std::vector<double> a;
  double s = 0;
  for (size_t i = 0; i < truth.size(); ++i) {
    const double e = std::log(static_cast<double>(out[i])) - std::log(truth[i]);
    s += e * e;
    a.push_back(std::abs(e));
  }
  std::sort(a.begin(), a.end());
  const double pos = 0.99 * static_cast<double>(a.size() - 1);
  const auto lo = static_cast<size_t>(pos);
  const double p99 = lo + 1 < a.size() ? a[lo] + (pos - static_cast<double>(lo)) * (a[lo + 1] - a[lo]) : a[lo];
  return {std::sqrt(s / static_cast<double>(truth.size())), p99};
}

// F3's source raster: s x the pinhole with our fx and fy (not square), cells on our pixel edges, covering the crop.
SourceCamera f3_source(const LensModel& lens, Roi crop, int32_t s) {
  SourceCamera sc = frames::overscan_pinhole(lens, s, 3.0, &crop);
  const double oy = lens.radtan.cy + 0.5 - (sc.cy + 0.5) / s;
  sc.fy = s * lens.radtan.fy;
  sc.cy = s * (lens.radtan.cy + 0.5 - oy) - 0.5;
  return sc;
}

// E3b's source raster: as F3's, but with cell centres (not edges) on multiples of 1/s of pinhole pixels.
SourceCamera e3b_source(const LensModel& lens, Roi crop, int32_t s) {
  SourceCamera sc = f3_source(lens, crop, s);
  const double ax = s * lens.radtan.cx - sc.cx, ay = s * lens.radtan.cy - sc.cy;  // F3: s c - a0, a0 not integer
  sc.cx = s * lens.radtan.cx - std::floor(ax), sc.cy = s * lens.radtan.cy - std::floor(ay);
  return sc;
}

// One F3 configuration: route S at s through the quadrature operator against the truth (kNominal: F3's sigma_r).
Err f3_case(const LensModel& lens, Roi crop, double a, int32_t s, const std::vector<double>& truth, Blur blur,
            std::optional<ResampleOperator>* keep = nullptr, bool e3b = false) {
  const SourceCamera sc = e3b ? e3b_source(lens, crop, s) : f3_source(lens, crop, s);
  const std::vector<float> src = render_source(sc, checker, 24);
  ResampleOperator op = ResampleOperator::build(lens, sc, {a, 0.15, 3.0, Rule::kQuadrature, 16, blur}, crop);
  std::vector<float> out(truth.size());
  op.apply(src.data(), 1, out.data());
  if (keep != nullptr) keep->emplace(std::move(op));
  return log_error(out, truth);
}

constexpr Roi kCorner{4, 4, 36, 36}, kCentre{300, 220, 332, 252};

void check_weight_sums(const ResampleOperator& op) {
  const Roi r = op.roi();
  double worst = 0;
  for (int32_t y = 0; y < r.y1 - r.y0; ++y) {
    for (int32_t x = 0; x < r.x1 - r.x0; ++x) {
      double s = 0;
      for (const auto& [i, w] : op.weights(x, y)) {
        CHECK(w > 0);
        s += w;
      }
      worst = std::max(worst, std::abs(s - 1.0));
    }
  }
  CHECK(worst <= 1e-7);
}

// source_rect() is the plan's render border: it must hold every tap the operator reads, and be tight (a tap on each
// of its edges).
void check_source_rect(const ResampleOperator& op) {
  const Roi r = op.roi(), sr = op.source_rect();
  const int64_t sw = op.source_width();
  int64_t xlo = std::numeric_limits<int64_t>::max(), ylo = xlo, xhi = std::numeric_limits<int64_t>::min(), yhi = xhi;
  bool inside = true;
  for (int32_t y = 0; y < r.y1 - r.y0; ++y) {
    for (int32_t x = 0; x < r.x1 - r.x0; ++x) {
      for (const auto& [i, w] : op.weights(x, y)) {
        const int64_t xs = i % sw, ys = i / sw;
        inside = inside && xs >= sr.x0 && xs < sr.x1 && ys >= sr.y0 && ys < sr.y1;
        xlo = std::min(xlo, xs), xhi = std::max(xhi, xs), ylo = std::min(ylo, ys), yhi = std::max(yhi, ys);
      }
    }
  }
  CHECK(inside);
  CHECK(xlo == sr.x0);
  CHECK(xhi == sr.x1 - 1);
  CHECK(ylo == sr.y0);
  CHECK(yhi == sr.y1 - 1);
}

}  // namespace

TEST_CASE("U5: radtan corner at s = 4 (CSR) within 0.006 RMS and 0.025 p99; the nominal rule reproduces F3") {
  const LensModel lens = frames::lens_model(evimo2_left(true), kSupport);
  const std::vector<double> truth = truth_gh3(lens, kCorner, 1.0, 0.15, checker);
  const Err f3 = f3_case(lens, kCorner, 1.0, 4, truth, Blur::kNominal);
  MESSAGE("radtan corner s4, nominal sigma_r: rms " << f3.rms << " p99 " << f3.p99 << " (F3 0.004467 / 0.020831)");
  CHECK(f3.rms == doctest::Approx(0.004467469773757339).epsilon(0.01));
  CHECK(f3.p99 == doctest::Approx(0.020831096170107886).epsilon(0.01));
  std::optional<ResampleOperator> keep;
  const Err e = f3_case(lens, kCorner, 1.0, 4, truth, Blur::kLocal, &keep);
  const ResampleOperator& op = *keep;
  MESSAGE("radtan corner s4, local sigma_r: rms " << e.rms << " p99 " << e.p99);
  CHECK(e.rms <= 0.006);
  CHECK(e.p99 <= 0.025);
  CHECK_FALSE(op.separable());
  CHECK(op.nan_pixels() == 0);
  CHECK(op.excess_variance_px2() == 0.0);
  CHECK(op.warnings().empty());
  CHECK(static_cast<double>(op.nnz()) / (32 * 32) <= 49);  // E3e: <= 49 taps per pixel at s = 4
  check_weight_sums(op);
}

TEST_CASE("U5: pinhole (separable) within 0.002; separable equals CSR; weights sum to 1") {
  const LensModel lens = frames::lens_model(evimo2_left(false), kSupport);
  const std::vector<double> truth = truth_gh3(lens, kCentre, 1.0, 0.15, checker);
  std::optional<ResampleOperator> keep;
  const Err e = f3_case(lens, kCentre, 1.0, 4, truth, Blur::kLocal, &keep);
  const ResampleOperator& op = *keep;
  MESSAGE("pinhole centre s4: rms " << e.rms << " p99 " << e.p99 << " (F3 0.000837 / 0.003102)");
  CHECK(e.rms <= 0.002);
  CHECK(e.rms == doctest::Approx(0.0008366524400522779).epsilon(0.01));
  CHECK(op.separable());
  check_weight_sums(op);

  // A distortion of 1e-13 takes the CSR path with the same map: the same weights within float rounding.
  render::Camera tiny = evimo2_left(false);
  tiny.dist = {1e-13, 0, 0, 0, 0};
  const LensModel lt = frames::lens_model(tiny, kSupport);
  const ResampleOperator csr = ResampleOperator::build(lt, f3_source(lens, kCentre, 4), {1, 0.15, 3, Rule::kQuadrature},
                                                       kCentre);
  CHECK_FALSE(csr.separable());
  CHECK(csr.nnz() == op.nnz());
  double worst = 0;
  for (int32_t y = 0; y < 32; y += 3) {
    for (int32_t x = 0; x < 32; x += 3) {
      const auto a = op.weights(x, y), b = csr.weights(x, y);
      REQUIRE(a.size() == b.size());
      for (size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].first == b[i].first);
        worst = std::max(worst, std::abs(a[i].second - b[i].second));
      }
    }
  }
  CHECK(worst <= 1e-7);
}

TEST_CASE("U5: cells at sigma = 0 equal the exact box integral within 1e-6; FRM-APERTURE-QUANT") {
  // L = 2 + sin(alpha x + phi) cos(beta y) on the normalized plane, with closed-form rectangle integrals.
  const double al = 61.0, ph = 0.3, be = 47.0;
  const auto prim_x = [&](double x) { return -std::cos(al * x + ph) / al; };
  const auto prim_y = [&](double y) { return std::sin(be * y) / be; };
  const auto mean_rect = [&](double x0, double x1, double y0, double y1) {
    return 2.0 + (prim_x(x1) - prim_x(x0)) * (prim_y(y1) - prim_y(y0)) / ((x1 - x0) * (y1 - y0));
  };
  render::Camera cam = evimo2_left(false);
  cam.fy = cam.fx;  // the identity map needs square pixels
  for (const double a : {1.0, 0.5}) {
    const LensModel lens = frames::lens_model(cam, 0.5 * a);
    const Roi roi{100, 80, 140, 110};
    const SourceCamera sc = frames::overscan_pinhole(lens, 4, 2.0, &roi);
    CHECK(sc.fx == 4 * cam.fx);
    CHECK(sc.fy == 4 * cam.fx);
    std::vector<float> src(static_cast<size_t>(sc.width) * sc.height);
    for (int32_t j = 0; j < sc.height; ++j) {
      for (int32_t i = 0; i < sc.width; ++i) {
        src[static_cast<size_t>(j) * sc.width + i] = static_cast<float>(
            mean_rect((i - 0.5 - sc.cx) / sc.fx, (i + 0.5 - sc.cx) / sc.fx, (j - 0.5 - sc.cy) / sc.fy,
                      (j + 0.5 - sc.cy) / sc.fy));
      }
    }
    const ResampleOperator op = ResampleOperator::build(lens, sc, {a, 0.0, 3.0, Rule::kCells}, roi);
    CHECK(op.separable());
    CHECK(op.warnings().empty());
    CHECK(op.nnz() == static_cast<size_t>(30 * 40) * static_cast<size_t>(a * 4 * a * 4));
    std::vector<float> out(30 * 40);
    op.apply(src.data(), 1, out.data());
    double worst = 0;
    for (int32_t y = 0; y < 30; ++y) {
      for (int32_t x = 0; x < 40; ++x) {
        const double u = roi.x0 + x, v = roi.y0 + y;
        const double exact = mean_rect((u - a / 2 - cam.cx) / cam.fx, (u + a / 2 - cam.cx) / cam.fx,
                                       (v - a / 2 - cam.cy) / cam.fx, (v + a / 2 - cam.cy) / cam.fx);
        worst = std::max(worst, std::abs(static_cast<double>(out[static_cast<size_t>(y) * 40 + x]) / exact - 1.0));
      }
    }
    MESSAGE("cells a = " << a << ", sigma = 0: max relative error " << worst);
    CHECK(worst <= 1e-6);
    check_weight_sums(op);
  }
  // a s = 3 with s = 4: the box's edges fall inside cells; quantized (warned) when sigma < 1/s, not when sigma >= 1/s.
  const LensModel lens = frames::lens_model(cam, 0.75);
  const Roi roi{10, 10, 20, 20};
  const SourceCamera sc = frames::overscan_pinhole(lens, 4, 2.0, &roi);
  const ResampleOperator q = ResampleOperator::build(lens, sc, {0.75, 0.1, 3.0, Rule::kCells}, roi);
  REQUIRE(q.warnings().size() == 1);
  CHECK(q.warnings()[0].code == "FRM-APERTURE-QUANT");
  const auto w = q.weights(0, 0);
  CHECK(w.size() > 0);
  check_weight_sums(q);
  CHECK(ResampleOperator::build(lens, sc, {0.75, 0.3, 3.0, Rule::kCells}, roi).warnings().empty());
  // The cells rule needs the identity map.
  const LensModel dist = frames::lens_model(evimo2_left(true), kSupport);
  CHECK_THROWS_AS(ResampleOperator::build(dist, frames::overscan_pinhole(dist, 4, 2.0, &roi),
                                          {1, 0.15, 3, Rule::kCells}, roi),
                  std::invalid_argument);
  const LensModel rect = frames::lens_model(evimo2_left(false), kSupport);  // fx != fy
  CHECK_THROWS_AS(ResampleOperator::build(rect, frames::overscan_pinhole(rect, 4, 2.0, &roi),
                                          {1, 0.15, 3, Rule::kCells}, roi),
                  std::invalid_argument);
}

TEST_CASE("U5: cells with a point aperture (a = 0): G centred on the pixel; FRM-APERTURE-QUANT when sigma < 1/s") {
  render::Camera cam = evimo2_left(false);
  cam.fy = cam.fx;
  const Roi roi{100, 80, 112, 90};
  for (const int32_t s : {2, 3, 4, 8}) {
    // 1e-160 and 1e-200: (sigma s)^2 underflows, and the squared offsets over it would overflow.
    for (const double sigma : {1e-200, 1e-160, 0.001, 0.1, 0.3, 0.6}) {
      CAPTURE(s);
      CAPTURE(sigma);
      const LensModel lens = frames::lens_model(cam, 3 * sigma);
      const SourceCamera sc = frames::overscan_pinhole(lens, s, 2.0, &roi);
      const ResampleOperator op = ResampleOperator::build(lens, sc, {0.0, sigma, 3.0, Rule::kCells}, roi);
      REQUIRE(op.separable());
      CHECK(op.warnings().size() == (sigma < 1.0 / s ? 1u : 0u));
      if (!op.warnings().empty()) CHECK(op.warnings()[0].code == "FRM-APERTURE-QUANT");
      check_weight_sums(op);
      // Source cell centres in our px (identity map): the taps' mean is the pixel centre, their variance sigma^2
      // (within 3.3 %: the 3-sigma truncation) once sigma s >= 1. The point split into the two cells at +-1/(2s) and
      // blurred at whole-cell offsets (FR-3's first rule) adds 1/(4 s^2): +15 % at s = 4, sigma = 0.3.
      double worst_mean = 0, worst_var = 0;
      for (int32_t y = 0; y < roi.y1 - roi.y0; ++y) {
        for (int32_t x = 0; x < roi.x1 - roi.x0; ++x) {
          double mx = 0, my = 0, vx = 0, vy = 0;
          for (const auto& [i, w] : op.weights(x, y)) {
            const double dx = cam.cx + (static_cast<double>(i % sc.width) - sc.cx) / s - (roi.x0 + x);
            const double dy = cam.cy + (static_cast<double>(i / sc.width) - sc.cy) / s - (roi.y0 + y);
            mx += w * dx, my += w * dy, vx += w * dx * dx, vy += w * dy * dy;
          }
          worst_mean = std::max({worst_mean, std::abs(mx), std::abs(my)});
          worst_var = std::max({worst_var, std::abs(vx / (sigma * sigma) - 1), std::abs(vy / (sigma * sigma) - 1)});
        }
      }
      CHECK(worst_mean <= 1e-6);
      if (sigma * s >= 1.0) CHECK(worst_var <= 0.05);
      // A tiny sigma s keeps finite weights: the nearest cell (odd s) or the two cells at +-1/(2s) (even s).
      if (sigma * s < 0.01) CHECK(op.weights(0, 0).size() == (s % 2 == 0 ? 4u : 1u));
      std::vector<float> src(static_cast<size_t>(sc.width) * sc.height, 1.0f), out(12 * 10);
      op.apply(src.data(), 1, out.data());
      for (const float o : out) CHECK(o == doctest::Approx(1.0).epsilon(1e-6));
    }
  }
  // The checkerboard at a = 0, sigma = 0.3 against BallaSim's truncated response (three Gauss-Hermite nodes per axis
  // are no truth for a point aperture). The cells carry their 1/(12 s^2) and reach 1/(2s) past 3 sigma: +8 % variance
  // at s = 4 (0.014 RMS), +1 % at s = 8 (0.0022); the earlier rule's +1/(4 s^2) would add 18 % and 4 % more. The
  // quadrature (reported) has only its 3 x 3 nodes at a = 0.
  const LensModel lens = frames::lens_model(cam, 0.9);
  const std::vector<double> truth = truth_exact(lens, kCentre, 0.0, 0.3, checker, 160);
  for (const int32_t s : {4, 8}) {
    const SourceCamera sc = frames::overscan_pinhole(lens, s, 2.0, &kCentre);
    const std::vector<float> src = render_source(sc, checker, 24);
    std::vector<float> out(32 * 32);
    ResampleOperator::build(lens, sc, {0.0, 0.3, 3.0, Rule::kCells}, kCentre).apply(src.data(), 1, out.data());
    const Err e = log_error(out, truth);
    ResampleOperator::build(lens, sc, {0.0, 0.3, 3.0, Rule::kQuadrature}, kCentre).apply(src.data(), 1, out.data());
    const Err q = log_error(out, truth);
    MESSAGE("a = 0, sigma = 0.3, s" << s << ": cells rms " << e.rms << " p99 " << e.p99 << "; quadrature rms "
                                    << q.rms << " p99 " << q.p99);
    CHECK(e.rms <= (s == 4 ? 0.02 : 0.004));
  }
}

TEST_CASE("U5: cells continuous in a: cut pieces at their centroids, excess below 1/(12 s^2), collapsed boxes") {
  render::Camera cam = evimo2_left(false);
  cam.fy = cam.fx;
  const Roi roi{100, 80, 108, 86};
  const auto tap_moments = [&](const ResampleOperator& op, const SourceCamera& sc, int32_t s, double* worst_mean,
                               double* var) {  // per axis about the pixel centre, worst over the ROI
    *worst_mean = 0, *var = 0;
    for (int32_t y = 0; y < roi.y1 - roi.y0; ++y) {
      for (int32_t x = 0; x < roi.x1 - roi.x0; ++x) {
        double mx = 0, my = 0, vx = 0, vy = 0;
        for (const auto& [i, w] : op.weights(x, y)) {
          const double dx = cam.cx + (static_cast<double>(i % sc.width) - sc.cx) / s - (roi.x0 + x);
          const double dy = cam.cy + (static_cast<double>(i / sc.width) - sc.cy) / s - (roi.y0 + y);
          mx += w * dx, my += w * dy, vx += w * dx * dx, vy += w * dy * dy;
        }
        *worst_mean = std::max({*worst_mean, std::abs(mx), std::abs(my)});
        *var = std::max({*var, vx, vy});
      }
    }
  };
  const auto max_diff = [&](const ResampleOperator& p, const ResampleOperator& q) {
    double worst = 0;
    for (int32_t y = 0; y < roi.y1 - roi.y0; ++y) {
      for (int32_t x = 0; x < roi.x1 - roi.x0; ++x) {
        const auto a = p.weights(x, y), b = q.weights(x, y);
        REQUIRE(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i) {
          REQUIRE(a[i].first == b[i].first);
          worst = std::max(worst, std::abs(static_cast<double>(a[i].second) - b[i].second));
        }
      }
    }
    return worst;
  };
  for (const int32_t s : {2, 3, 4, 8}) {
    for (const double sigma : {0.3, 0.6}) {
      const double cell2 = 1.0 / (12.0 * s * s);
      const LensModel lens = frames::lens_model(cam, 0.5 * 1.2 + 3 * sigma);
      const SourceCamera sc = frames::overscan_pinhole(lens, s, 2.0, &roi);
      const ResampleOperator point = ResampleOperator::build(lens, sc, {0.0, sigma, 3.0, Rule::kCells}, roi);
      for (const double a : {0.0, 1e-300, 1e-20, 1e-16, 1e-6, 0.01, 0.1, 0.2, 0.25, 0.3, 0.375, 0.5, 0.6, 0.75, 1.0,
                             1.1, 1.2}) {
        CAPTURE(s);
        CAPTURE(sigma);
        CAPTURE(a);
        const ResampleOperator op = ResampleOperator::build(lens, sc, {a, sigma, 3.0, Rule::kCells}, roi);
        const double as = a * s;
        const bool whole = as >= 1 - 1e-9 && std::abs(as - std::round(as)) <= 1e-9 &&
                           static_cast<int64_t>(std::round(as)) % 2 == s % 2;
        CHECK(op.warnings().size() == (!whole && sigma < 1.0 / s ? 1u : 0u));
        check_weight_sums(op);
        // The response's variance per axis (the taps' about the pixel centre plus the cell's 1/(12 s^2)) against
        // a^2/12 + sigma^2. Each cut piece adds at most its fraction of 1/(12 s^2) (its cell's extent beyond its own);
        // the 3-sigma truncation takes up to 3 % (once sigma s >= 1; below it G's sampling adds or takes a few %, and
        // FRM-APERTURE-QUANT warns when sigma < 1/s). Placing each piece at its cell's centre (FR-3's first rule) gave
        // 1/(3 s^2) - a^2/12 for even s and a s < 2: +20 % at s = 4, sigma = 0.3 for a below 0.1.
        double mean = 0, var = 0;
        tap_moments(op, sc, s, &mean, &var);
        const double target = a * a / 12 + sigma * sigma, excess = var + cell2 - target;
        CHECK(mean <= 1e-6);
        if (sigma * s >= 1.0) {
          CHECK(excess <= cell2 + 1e-9);
          CHECK(excess >= -0.035 * target);
        }
        if (sigma == 0.3 && (s == 4 || s == 8) && (a == 0.0 || a == 1e-6 || a == 0.01 || a == 0.1 || a == 0.25)) {
          MESSAGE("s = " << s << ", sigma = 0.3, a = " << a << ": variance " << 100 * excess / target << " %");
        }
        // Continuous at a = 0: a box that rounding collapses is the point, a tiny one within 1e-5.
        if (a <= 1e-6) CHECK(max_diff(op, point) <= (a <= 1e-16 ? 1e-7 : 1e-5));
        std::vector<float> src(static_cast<size_t>(sc.width) * sc.height, 1.0f), out(8 * 6);
        op.apply(src.data(), 1, out.data());
        for (const float o : out) CHECK(o == doctest::Approx(1.0).epsilon(1e-6));
      }
    }
  }
  // sigma = 0 with a box below rounding: the point on a cell edge (even s) splits between the two cells.
  for (const int32_t s : {3, 4}) {
    const LensModel lens = frames::lens_model(cam, 0.5);
    const SourceCamera sc = frames::overscan_pinhole(lens, s, 2.0, &roi);
    const ResampleOperator op = ResampleOperator::build(lens, sc, {1e-300, 0.0, 3.0, Rule::kCells}, roi);
    CHECK(op.weights(0, 0).size() == (s % 2 == 0 ? 4u : 1u));
    check_weight_sums(op);
    // A box or a blur beyond any source raster is refused (bad_cover), never cast out of range.
    const std::pair<double, double> huge[] = {{1e300, 0.3}, {1e308, 0.3}, {1.0, 1e300}};
    for (const auto& [a, sigma] : huge) {
      CHECK_THROWS_AS(ResampleOperator::build(lens, sc, {a, sigma, 3.0, Rule::kCells}, roi), std::invalid_argument);
    }
  }
  // The checkerboard at a s = 2.4 (s = 4, sigma = 0.3: cut pieces, no warning) against BallaSim's truncated response:
  // 0.0014 RMS, where the pieces at their cells' centres (FR-3's first rule) gave 0.017.
  const LensModel lens = frames::lens_model(cam, 0.3 + 0.9);
  const std::vector<double> truth = truth_exact(lens, kCentre, 0.6, 0.3, checker, 160);
  const SourceCamera sc = frames::overscan_pinhole(lens, 4, 2.0, &kCentre);
  const std::vector<float> src = render_source(sc, checker, 24);
  std::vector<float> out(32 * 32);
  ResampleOperator::build(lens, sc, {0.6, 0.3, 3.0, Rule::kCells}, kCentre).apply(src.data(), 1, out.data());
  const Err e = log_error(out, truth);
  MESSAGE("a = 0.6, sigma = 0.3, s4: cells rms " << e.rms << " p99 " << e.p99);
  CHECK(e.rms <= 0.004);
}

TEST_CASE("U5: cells and quadrature at sigma = 0.15 on the pinhole checkerboard (reported)") {
  render::Camera cam = evimo2_left(false);
  cam.fy = cam.fx;
  const LensModel lens = frames::lens_model(cam, kSupport);
  const std::vector<double> truth = truth_gh3(lens, kCentre, 1.0, 0.15, checker);
  for (const int32_t s : {4, 8}) {
    const SourceCamera sc = frames::overscan_pinhole(lens, s, 2.0, &kCentre);
    const std::vector<float> src = render_source(sc, checker, 24);
    std::vector<float> out(32 * 32);
    for (const Rule rule : {Rule::kCells, Rule::kQuadrature}) {
      const ResampleOperator op = ResampleOperator::build(lens, sc, {1, 0.15, 3, rule}, kCentre);
      op.apply(src.data(), 1, out.data());
      const Err e = log_error(out, truth);
      const std::string name = rule == Rule::kCells ? "cells" : "quadrature";  // a const char* prints as "1"
      MESSAGE(name << " s" << s << ": rms " << e.rms << " p99 " << e.p99);
      CHECK(e.rms <= (rule == Rule::kCells ? 0.011 : 0.002));
    }
  }
}

TEST_CASE("U5: NaN outside the lens domain; an ROI operator equals the full operator's rows; threads") {
  // A 64 x 48 EVIMO2-like camera (intrinsics scaled by 0.1) keeps the full operator small.
  const render::Camera cam = evimo2_left(true, 64, 48, 0.1);
  const LensModel lens = frames::lens_model(cam, kSupport);
  const SourceCamera sc = frames::overscan_pinhole(lens, 4);
  const ResampleSpec spec{1, 0.15, 3, Rule::kQuadrature};
  const ResampleOperator full = ResampleOperator::build(lens, sc, spec, {0, 0, 64, 48}, 3);
  CHECK(full.nan_pixels() == 0);
  const Roi roi{10, 7, 40, 30};
  const ResampleOperator part = ResampleOperator::build(lens, sc, spec, roi, 1);
  for (int32_t y = 0; y < 23; ++y) {
    for (int32_t x = 0; x < 30; ++x) CHECK(part.weights(x, y) == full.weights(x + 10, y + 7));
  }
  const Roi sr = part.source_rect();
  CHECK((sr.x0 >= 0 && sr.y0 >= 0 && sr.x1 <= sc.width && sr.y1 <= sc.height && sr.x0 < sr.x1 && sr.y0 < sr.y1));
  check_source_rect(part);
  check_source_rect(full);
  // Separable operators (quadrature and cells on an undistorted square-pixel camera): the same rectangle contract.
  render::Camera und = evimo2_left(false, 64, 48, 0.1);
  und.fy = und.fx;
  const LensModel ul = frames::lens_model(und, kSupport);
  const SourceCamera us = frames::overscan_pinhole(ul, 4);
  for (const Rule rule : {Rule::kQuadrature, Rule::kCells}) {
    const ResampleOperator sep = ResampleOperator::build(ul, us, {1, 0.15, 3, rule}, roi);
    REQUIRE(sep.separable());
    check_source_rect(sep);
  }

  // Shrinking r_lim leaves the outer pixels without a preimage: NaN there, finite inside.
  LensModel small = lens;
  small.radtan.r_lim = 0.25;
  const ResampleOperator cut = ResampleOperator::build(small, sc, spec, {0, 0, 64, 48});
  CHECK(cut.nan_pixels() > 0);
  CHECK(cut.nan_pixels() < 64 * 48);
  std::vector<float> src(static_cast<size_t>(sc.width) * sc.height * 2, 1.0f), out(64 * 48 * 2);
  cut.apply(src.data(), 2, out.data());
  for (int32_t y = 0; y < 48; ++y) {
    for (int32_t x = 0; x < 64; ++x) {
      const bool empty = cut.weights(x, y).empty();
      const double r = std::hypot((x - cam.cx) / cam.fx, (y - cam.cy) / cam.fy);
      if (r > 0.3) CHECK(empty);
      if (r < 0.2) CHECK_FALSE(empty);
      for (int c = 0; c < 2; ++c) {
        const float o = out[(static_cast<size_t>(y) * 64 + x) * 2 + c];
        if (empty) {
          CHECK(std::isnan(o));
        } else {
          CHECK(o == doctest::Approx(1.0).epsilon(1e-6));
        }
      }
    }
  }

  // Only the quadrature points decide NaN (§4.7.2): with a = 0 and sigma = 0.05 (< (1/s)/2, so sigma_r = 0) every
  // point is the pixel centre, and Blur::kLocal's Jacobian probes at +-0.5 px reach past r_lim for the pixels on the
  // domain's edge. Those take the nominal rule: kLocal equals kNominal pixel for pixel (no extra NaN, no throw).
  const ResampleSpec centre_local{0, 0.05, 3, Rule::kQuadrature, 16, Blur::kLocal};
  const ResampleSpec centre_nominal{0, 0.05, 3, Rule::kQuadrature, 16, Blur::kNominal};
  const ResampleOperator loc = ResampleOperator::build(small, sc, centre_local, {0, 0, 64, 48});
  const ResampleOperator nom = ResampleOperator::build(small, sc, centre_nominal, {0, 0, 64, 48});
  int64_t probe_fails = 0;
  for (int32_t y = 0; y < 48; ++y) {
    for (int32_t x = 0; x < 64; ++x) {
      Vec3<double> ray{};
      bool all = unproject(small, x, y, &ray);
      const bool centre = all;
      for (const double d : {0.5, -0.5}) {
        all = all && unproject(small, x + d, y, &ray) && unproject(small, x, y + d, &ray);
      }
      if (centre && !all) ++probe_fails;
      const auto a = loc.weights(x, y), b = nom.weights(x, y);
      CHECK(a.empty() == !centre);
      REQUIRE(a.size() == b.size());
      for (size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].first == b[i].first);
        CHECK(std::abs(a[i].second - b[i].second) <= 1e-7);
      }
    }
  }
  MESSAGE("pixels with a preimage whose +-0.5 px probes leave the domain: " << probe_fails);
  CHECK(probe_fails > 0);
  CHECK(loc.nan_pixels() == nom.nan_pixels());
}

TEST_CASE("U5: apply_tiles equals apply (bitwise for one tile); multi-channel; coverage and refusals") {
  for (const bool distorted : {true, false}) {
    const LensModel lens = frames::lens_model(evimo2_left(distorted), kSupport);
    const SourceCamera sc = frames::overscan_pinhole(lens, 8, 2.0, &kCorner);
    const std::vector<float> mono = render_source(sc, checker, 4);
    std::vector<float> src(mono.size() * 3);
    for (size_t i = 0; i < mono.size(); ++i) {
      src[3 * i] = mono[i], src[3 * i + 1] = 2.0f * mono[i], src[3 * i + 2] = 11.0f - mono[i];
    }
    const ResampleOperator op = ResampleOperator::build(lens, sc, {1, 0.15, 3, Rule::kQuadrature}, kCorner);
    CHECK(op.separable() == !distorted);
    std::vector<float> ref(32 * 32 * 3), one(ref.size(), 0.0f), tiled(ref.size(), 0.0f), m1(32 * 32);
    op.apply(src.data(), 3, ref.data());
    op.apply(mono.data(), 1, m1.data());
    for (size_t p = 0; p < m1.size(); ++p) CHECK(ref[3 * p] == m1[p]);
    op.apply_tiles({sc.height, sc.width, 3, src.data()}, 0, 3, one.data());
    CHECK(one == ref);
    const int32_t cuts[] = {0, 37, sc.height / 2, sc.height};
    for (int t = 2; t >= 0; --t) {  // any tile order
      const int32_t y0 = cuts[t], h = cuts[t + 1] - cuts[t];
      op.apply_tiles({h, sc.width, 3, src.data() + static_cast<size_t>(y0) * sc.width * 3}, y0, 3, tiled.data());
    }
    double worst = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(tiled[i]) / static_cast<double>(ref[i]) - 1.0));
    }
    CHECK(worst <= 1e-6);
    CHECK_THROWS_AS(op.apply_tiles({4, sc.width - 1, 3, src.data()}, 0, 3, tiled.data()), std::invalid_argument);
    CHECK_THROWS_AS(op.apply_tiles({4, sc.width, 3, src.data()}, sc.height - 3, 3, tiled.data()),
                    std::invalid_argument);
  }
  // A source raster that misses the ROI's preimage is refused at build time.
  const LensModel lens = frames::lens_model(evimo2_left(true), kSupport);
  SourceCamera sc = frames::overscan_pinhole(lens, 4, 2.0, &kCentre);
  sc.width -= 12;
  CHECK_THROWS_AS(ResampleOperator::build(lens, sc, {1, 0.15, 3, Rule::kQuadrature}, kCentre), std::invalid_argument);
  sc.width += 12;
  CHECK_THROWS_AS(ResampleOperator::build(lens, sc, {0, 0, 3, Rule::kQuadrature}, kCentre), std::invalid_argument);
  // An empty or out-of-image ROI is FRM-VALUE (frames.md §6), for the operator and for the overscan source.
  for (const Roi bad : {Roi{0, 0, 641, 10}, Roi{5, 5, 5, 10}, Roi{-1, 0, 10, 10}}) {
    try {
      (void)ResampleOperator::build(lens, sc, {1, 0.15, 3, Rule::kQuadrature}, bad);
      FAIL("a bad ROI must be refused");
    } catch (const frames::FramesError& e) {
      CHECK(e.code == "FRM-VALUE");
    }
    try {
      (void)frames::overscan_pinhole(lens, 4, 2.0, &bad);
      FAIL("a bad ROI must be refused");
    } catch (const frames::FramesError& e) {
      CHECK(e.code == "FRM-VALUE");
    }
  }
  // strata in [1, 64].
  for (const int32_t strata : {0, 65, 4096}) {
    CHECK_THROWS_AS(ResampleOperator::build(lens, sc, {1, 0.15, 3, Rule::kQuadrature, strata}, kCentre),
                    std::invalid_argument);
  }
  CHECK_NOTHROW((void)ResampleOperator::build(lens, sc, {1, 0.15, 3, Rule::kQuadrature, 64}, {316, 236, 318, 237}));
  sc.kind = SourceCamera::Kind::kEquidistant;
  try {
    (void)ResampleOperator::build(lens, sc, {1, 0.15, 3, Rule::kQuadrature}, kCentre);
    FAIL("the equidistant source must be refused");
  } catch (const frames::FramesError& e) {
    CHECK(e.code == "FRM-KB-WIDE");
  }
}

TEST_CASE("U5: FRM-RESAMPLE-BLUR reports the excess variance when sigma^2 < (1/s)^2/4") {
  const LensModel lens = frames::lens_model(evimo2_left(true), 0.5 + 0.3);
  const SourceCamera sc = frames::overscan_pinhole(lens, 4, 2.0, &kCentre);
  const Roi roi{300, 220, 304, 224};
  const ResampleOperator op =
      ResampleOperator::build(lens, sc, {1, 0.1, 3, Rule::kQuadrature, 16, Blur::kNominal}, roi);
  CHECK(op.excess_variance_px2() == doctest::Approx(0.25 / 16 - 0.01));
  REQUIRE(op.warnings().size() == 1);
  CHECK(op.warnings()[0].code == "FRM-RESAMPLE-BLUR");
  check_weight_sums(op);
  // Locally the magnification is about 1 at the centre: the same excess within 1 %.
  const ResampleOperator loc = ResampleOperator::build(lens, sc, {1, 0.1, 3, Rule::kQuadrature}, roi);
  CHECK(loc.excess_variance_px2() == doctest::Approx(0.25 / 16 - 0.01).epsilon(0.01));
  REQUIRE(loc.warnings().size() == 1);
  CHECK(loc.warnings()[0].code == "FRM-RESAMPLE-BLUR");
  CHECK(ResampleOperator::build(lens, sc, {1, 0.15, 3, Rule::kQuadrature}, roi).warnings().empty());
}

TEST_CASE("overscan: the design's square source, margins, E3a's cover; Kannala-Brandt cover and FRM-KB-WIDE") {
  // Identity map: symmetric integer margins m = ceil(support + margin - 0.5), H_b = s (H + 2m), cells on our edges.
  render::Camera sq = evimo2_left(false);
  sq.fy = sq.fx;
  const SourceCamera id = frames::overscan_pinhole(sq, kSupport, 4);
  CHECK(id.margin_rows == 3);
  CHECK(id.height == 4 * (480 + 6));
  CHECK(id.width == 4 * (640 + 6));
  CHECK(id.cx == doctest::Approx(4 * (sq.cx + 0.5 + 3) - 0.5));
  CHECK(id.fx == 4 * sq.fx);
  CHECK(id.fy == 4 * sq.fx);
  // EVIMO2 left: E3a's pinhole cover of the support is 649.3 x 488.0 px; with the 2 px margins on each side.
  const SourceCamera ev = frames::overscan_pinhole(evimo2_left(true), kSupport, 4);
  CHECK(ev.width / 4.0 >= 649.3 + 4.0);
  CHECK(ev.width / 4.0 <= 649.3 + 4.0 + 2.0);
  CHECK(ev.height / 4.0 >= 488.0 + 4.0);
  CHECK(ev.height / 4.0 <= 488.0 * kFx / kFy + 4.0 + 2.0);  // E3a used fy for rows, the square source fx
  // Its corners' operators read inside the full-image source.
  const LensModel lens = frames::lens_model(evimo2_left(true), kSupport);
  for (const Roi r : {Roi{0, 0, 4, 4}, Roi{636, 0, 640, 4}, Roi{0, 476, 4, 480}, Roi{636, 476, 640, 480}}) {
    CHECK_NOTHROW((void)ResampleOperator::build(lens, ev, {1, 0.15, 3, Rule::kQuadrature}, r));
  }

  // A 320 x 240 Kannala-Brandt camera (about 38 degrees at the corners) is covered; its checkerboard error at s = 4.
  render::Camera kb;
  kb.model = render::CameraModelKind::kKannalaBrandt;
  kb.width = 320, kb.height = 240;
  kb.fx = 300, kb.fy = 300.5, kb.cx = 160, kb.cy = 120;
  kb.dist = {-0.013, 0.0021, -0.0009, 0.0001};
  const LensModel kl = frames::lens_model(kb, kSupport);
  const SourceCamera ks = frames::overscan_pinhole(kl, 4, 2.0, &kCorner);
  const Radiance kb_checker = [](double x, double y) { return checker_sq(x, y, 6.0 / 300.0); };
  const std::vector<double> truth = truth_gh3(kl, kCorner, 1.0, 0.15, kb_checker);
  const std::vector<float> src = render_source(ks, kb_checker, 24);
  std::vector<float> out(32 * 32);
  for (const Blur blur : {Blur::kLocal, Blur::kNominal}) {
    const ResampleOperator op = ResampleOperator::build(kl, ks, {1, 0.15, 3, Rule::kQuadrature, 16, blur}, kCorner);
    op.apply(src.data(), 1, out.data());
    const Err e = log_error(out, truth);
    const std::string name = blur == Blur::kLocal ? "local" : "nominal";
    MESSAGE("kb corner s4, " << name << " sigma_r: rms " << e.rms << " p99 " << e.p99);
    if (blur == Blur::kLocal) {  // the nominal rule over-sharpens where the source is 1.2-1.6 x denser (0.017)
      CHECK(e.rms <= 0.006);
      CHECK(e.p99 <= 0.025);
    }
  }
  // The 1280 x 960 fisheye of test_camera.cpp sees beyond 90 degrees: no pinhole cover.
  kb.width = 1280, kb.height = 960, kb.cx = 640, kb.cy = 480;
  try {
    (void)frames::overscan_pinhole(kb, kSupport, 4);
    FAIL("a fisheye beyond 90 degrees must be refused");
  } catch (const frames::FramesError& err) {
    CHECK(err.code == "FRM-KB-WIDE");
  }
}

namespace {

// E3c's 99 points: an 11 x 9 grid of target pixels.
std::vector<std::array<double, 2>> e3c_points() {
  std::vector<std::array<double, 2>> uv;
  for (int j = 0; j < 9; ++j) {
    for (int i = 0; i < 11; ++i) uv.push_back({12.3 + (627.7 - 12.3) * i / 10, 10.6 + (469.2 - 10.6) * j / 8});
  }
  return uv;
}

// Centroid errors of discs of radius r_n (normalized) at the points, through an s = 4 route S render.
std::vector<std::array<double, 2>> centroid_errors(std::vector<std::array<double, 2>>* gt_errors) {
  const LensModel lens = frames::lens_model(evimo2_left(true), kSupport);
  const SourceCamera sc = frames::overscan_pinhole(lens, 4);
  const double rn = 0.002;
  std::vector<std::array<double, 2>> xn;
  for (const auto& p : e3c_points()) {
    Vec3<double> r{};
    must_unproject(lens, p[0], p[1], &r);
    xn.push_back({r.x, r.y});
  }
  // The points are >= 46 px apart: each window and each disc's cells see one disc.
  const auto disc = [&](const std::array<double, 2>& c) {
    return Radiance([c, rn](double x, double y) {
      return (x - c[0]) * (x - c[0]) + (y - c[1]) * (y - c[1]) <= rn * rn ? 1.0 : 0.0;
    });
  };
  std::vector<float> src(static_cast<size_t>(sc.width) * sc.height, 0.0f);
  for (const auto& c : xn) {  // only the cells near each disc
    const double qx = sc.fx * c[0] + sc.cx, qy = sc.fy * c[1] + sc.cy, rq = sc.fx * rn + 2;
    render_into(src, sc, disc(c), 16,
                {static_cast<int32_t>(qx - rq), static_cast<int32_t>(qy - rq), static_cast<int32_t>(qx + rq) + 1,
                 static_cast<int32_t>(qy + rq) + 1});
  }
  std::vector<std::array<double, 2>> err;
  const auto pts = e3c_points();
  for (size_t k = 0; k < pts.size(); ++k) {
    const auto& p = pts[k];
    const int32_t iu = static_cast<int32_t>(std::lround(p[0])), iv = static_cast<int32_t>(std::lround(p[1]));
    const Roi win{std::max(iu - 6, 0), std::max(iv - 6, 0), std::min(iu + 7, 640), std::min(iv + 7, 480)};
    const ResampleOperator op = ResampleOperator::build(lens, sc, {1, 0.15, 3, Rule::kQuadrature}, win);
    const int32_t W = win.x1 - win.x0, H = win.y1 - win.y0;
    std::vector<float> out(static_cast<size_t>(W) * H);
    op.apply(src.data(), 1, out.data());
    const auto centroid = [&](const auto& img) {
      double s = 0, sx = 0, sy = 0;
      for (int32_t y = 0; y < H; ++y) {
        for (int32_t x = 0; x < W; ++x) {
          const double v = static_cast<double>(img[static_cast<size_t>(y) * W + x]);
          s += v, sx += v * (win.x0 + x), sy += v * (win.y0 + y);
        }
      }
      return std::array<double, 2>{sx / s, sy / s};
    };
    const auto c = centroid(out);
    err.push_back({c[0] - p[0], c[1] - p[1]});
    if (gt_errors != nullptr) {
      const auto g = centroid(truth_gh3(lens, win, 1.0, 0.15, disc(xn[k]), 24));
      gt_errors->push_back({c[0] - g[0], c[1] - g[1]});
    }
  }
  return err;
}

double rms2(const std::vector<std::array<double, 2>>& e) {
  double s = 0;
  for (const auto& d : e) s += d[0] * d[0] + d[1] * d[1];
  return std::sqrt(s / static_cast<double>(e.size()));
}

}  // namespace

TEST_CASE("I-M3 (route S, synthetic): 99 discs keep their centroids within 0.015 px RMS at s = 4") {
  const auto err = centroid_errors(nullptr);
  REQUIRE(err.size() == 99);
  MESSAGE("route S s4 centroid RMS vs target " << rms2(err) << " px (E3d's Cycles OSL radtan: 0.0122)");
  CHECK(rms2(err) <= 0.015);
}

TEST_CASE("gate: E3b/F3 reproduced (16 configurations), full-frame operator size and build time" *
          doctest::skip()) {
  struct Ref {
    bool radtan;
    Roi crop;
    double a;
    int32_t s;
    double rms, p99;
  };
  const Ref refs[] = {
      {true, kCorner, 1.0, 4, 0.004467469773757339, 0.020831096170107886},
      {true, kCorner, 1.0, 8, 0.0016224346428308507, 0.007000330793641386},
      {true, kCorner, 0.5, 4, 0.014696658525073675, 0.0679825826233336},
      {true, kCorner, 0.5, 8, 0.006554204798185718, 0.027648084172369282},
      {true, kCentre, 1.0, 4, 0.0008232410000806246, 0.003085503813356197},
      {true, kCentre, 1.0, 8, 0.001339645195407695, 0.005123347973084234},
      {true, kCentre, 0.5, 4, 0.0021868881099883506, 0.008687801009721906},
      {true, kCentre, 0.5, 8, 0.005235331108920574, 0.021051626969939924},
      {false, kCorner, 1.0, 4, 0.0008094851180796891, 0.0029868698458019706},
      {false, kCorner, 1.0, 8, 0.0012793028688998132, 0.0052023093695006215},
      {false, kCorner, 0.5, 4, 0.002159446577810207, 0.008368172941209907},
      {false, kCorner, 0.5, 8, 0.004866049299083531, 0.020103981239071513},
      {false, kCentre, 1.0, 4, 0.0008366524400522779, 0.003101666841894046},
      {false, kCentre, 1.0, 8, 0.0013466203120044108, 0.005017591320373067},
      {false, kCentre, 0.5, 4, 0.002104423166785137, 0.008198261361217929},
      {false, kCentre, 0.5, 8, 0.005239004943508872, 0.020641770058329223},
  };
  for (const Ref& r : refs) {
    const LensModel lens = frames::lens_model(evimo2_left(r.radtan), 0.5 * r.a + 0.45);
    const std::vector<double> truth = truth_gh3(lens, r.crop, r.a, 0.15, checker);
    const Err e = f3_case(lens, r.crop, r.a, r.s, truth, Blur::kNominal);
    const Err l = f3_case(lens, r.crop, r.a, r.s, truth, Blur::kLocal);
    std::printf("F3 %-7s %-6s a%.1f x%d: rms %.6f (F3 %.6f)  p99 %.6f (F3 %.6f) | local rms %.6f p99 %.6f\n",
                r.radtan ? "radtan" : "pinhole", r.crop.x0 == 4 ? "corner" : "centre", r.a, r.s, e.rms, r.rms, e.p99,
                r.p99, l.rms, l.p99);
    CHECK(e.rms == doctest::Approx(r.rms).epsilon(0.01));
    CHECK(e.p99 == doctest::Approx(r.p99).epsilon(0.02));
  }
  // E3b (its own cell alignment), radtan, a = 1, "dense_bilinear": s = 2 and 3 leave excess variance (sigma_r = 0).
  const Ref e3b[] = {
      {true, kCorner, 1.0, 2, 0.0768815401662566, 0.2884033457983986},
      {true, kCorner, 1.0, 3, 0.017228597882817914, 0.08250897618368534},
      {true, kCorner, 1.0, 4, 0.004933285166457318, 0.023137225228521113},
      {true, kCentre, 1.0, 2, 0.08107788186292948, 0.2859538286965592},
      {true, kCentre, 1.0, 3, 0.009897016812923341, 0.0351479627209718},
      {true, kCentre, 1.0, 4, 0.005108023039811626, 0.019101054909123137},
  };
  for (const Ref& r : e3b) {
    const LensModel lens = frames::lens_model(evimo2_left(true), kSupport);
    const std::vector<double> truth = truth_gh3(lens, r.crop, 1.0, 0.15, checker);
    const Err e = f3_case(lens, r.crop, 1.0, r.s, truth, Blur::kNominal, nullptr, true);
    std::printf("E3b radtan %-6s x%d: rms %.6f (E3b %.6f)  p99 %.6f (E3b %.6f)\n", r.crop.x0 == 4 ? "corner" : "centre",
                r.s, e.rms, r.rms, e.p99, r.p99);
    CHECK(e.rms == doctest::Approx(r.rms).epsilon(0.01));
    CHECK(e.p99 == doctest::Approx(r.p99).epsilon(0.02));
  }
  // Against BallaSim's own response (truncated box ⊛ G by its inverse CDF), radtan, a = 1.
  for (const Roi crop : {kCorner, kCentre}) {
    const LensModel lens = frames::lens_model(evimo2_left(true), kSupport);
    const std::vector<double> exact = truth_exact(lens, crop, 1.0, 0.15, checker, 160);
    for (const int32_t s : {4, 8}) {
      const Err e = f3_case(lens, crop, 1.0, s, exact, Blur::kNominal);
      const Err l = f3_case(lens, crop, 1.0, s, exact, Blur::kLocal);
      std::printf("vs exact response, radtan %s x%d: nominal rms %.6f p99 %.6f | local rms %.6f p99 %.6f\n",
                  crop.x0 == 4 ? "corner" : "centre", s, e.rms, e.p99, l.rms, l.p99);
    }
  }
  // E3e: the full VGA operator at s = 4 and s = 8.
  const LensModel lens = frames::lens_model(evimo2_left(true), kSupport);
  for (const int32_t s : {4, 8}) {
    const SourceCamera sc = frames::overscan_pinhole(lens, s);
    const auto t0 = std::chrono::steady_clock::now();
    const ResampleOperator op = ResampleOperator::build(lens, sc, {1, 0.15, 3, Rule::kQuadrature}, {0, 0, 640, 480});
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<float> src(static_cast<size_t>(sc.width) * sc.height * 3, 1.0f), out(640 * 480 * 3);
    const auto t1 = std::chrono::steady_clock::now();
    op.apply(src.data(), 3, out.data());
    const double sa = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("VGA s%d: source %dx%d, build %.2f s, nnz/px %.1f, %.1f MB, apply RGB %.3f s (one thread)\n", s,
                sc.width, sc.height, sec, static_cast<double>(op.nnz()) / (640 * 480), op.bytes() / 1e6, sa);
  }
  std::vector<std::array<double, 2>> gt;
  const auto err = centroid_errors(&gt);
  std::printf("I-M3 synthetic s4: centroid RMS vs target %.5f px, vs the truth image's centroid %.5f px\n", rms2(err),
              rms2(gt));
}
