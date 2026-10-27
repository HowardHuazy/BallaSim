// Route S's resampler: the operator's build (quadrature, cells), its application, and the overscan source
// (docs/design/phase3/frames.md §4.7.2). See resample.h.
#include "frames/resample.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <thread>

#include "core/camera/injective_domain.h"
#include "render/thread_pool.h"

namespace balla::frames {

namespace {

constexpr int32_t kMaxRasterSide = 65536;  // Blender's RenderSettings.resolution_x/y range (Python API)
constexpr double kAlignTol = 1e-9;         // px: integer margins, identity-map checks

[[noreturn]] void bad(const std::string& what) { throw std::invalid_argument("resample: " + what); }

int pool_size(int threads) {
  const unsigned hw = std::thread::hardware_concurrency();
  return threads > 0 ? threads : static_cast<int>(hw > 0 ? hw : 1);
}

// frames.md §6: an empty or out-of-image ROI is FRM-VALUE, as in reference.cpp and aps_frames.cpp.
void check_roi(const Roi& r, int32_t w, int32_t h) {
  if (r.x0 < 0 || r.y0 < 0 || r.x1 > w || r.y1 > h || r.x0 >= r.x1 || r.y0 >= r.y1) {
    throw FramesError("FRM-VALUE", "frames resample: the ROI must be non-empty and inside the " + std::to_string(w) +
                                       "x" + std::to_string(h) + " image");
  }
}

bool zero_distortion(const LensModel& m) {
  const auto& c = m.radtan;
  return m.kind == render::CameraModelKind::kRadtan && c.k1 == 0.0 && c.k2 == 0.0 && c.p1 == 0.0 && c.p2 == 0.0 &&
         c.k3 == 0.0;
}

bool unproject(const LensModel& m, double u, double v, Vec3<double>* ray) {
  return m.kind == render::CameraModelKind::kRadtan ? camera::unproject(m.radtan, Vec2<double>{u, v}, ray)
                                                     : camera::unproject(m.kb, Vec2<double>{u, v}, ray);
}

[[noreturn]] void kb_wide(const std::string& where) {
  throw FramesError("FRM-KB-WIDE", "route S: " + where + " sees a ray at or beyond 90 degrees, which a pinhole "
                                   "source cannot cover (FR-3b); use route osl");
}

// Gauss-Hermite nodes and weights of N(0, 1), 3 points (exact for polynomials of degree 5).
constexpr double kGhNode[3] = {-std::numbers::sqrt3, 0.0, std::numbers::sqrt3};
constexpr double kGhWeight[3] = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};

// The blur left to the quadrature, sigma^2 I - (1/4) A A^T with A = d(ours)/d(source) (the source cell adds 1/12 and
// the bilinear tent 1/6 per source axis), as its symmetric square root L, clamped at 0; excess: the clamped part's
// largest variance. A diagonal A gives L = diag(sigma_r,x, sigma_r,y) with sigma_r^2 = sigma^2 - a^2/4 exactly.
struct ResidualBlur {
  double l00, l01, l10, l11, excess;
};
ResidualBlur residual_blur(double sigma, double a00, double a01, double a10, double a11) {
  const double sg = sigma * sigma;
  const double ra = sg - 0.25 * (a00 * a00 + a01 * a01), rc = sg - 0.25 * (a10 * a10 + a11 * a11);
  const double rb = -0.25 * (a00 * a10 + a01 * a11);
  if (rb == 0.0) {
    return {ra > 0.0 ? std::sqrt(ra) : 0.0, 0.0, 0.0, rc > 0.0 ? std::sqrt(rc) : 0.0, std::max({0.0, -ra, -rc})};
  }
  const double m = 0.5 * (ra + rc), d = std::hypot(0.5 * (ra - rc), rb);
  const double l1 = m + d, l2 = m - d;
  const double nv = std::hypot(rb, l1 - ra);  // > 0: the eigenvector (rb, l1 - ra) of l1, rb != 0
  const double vx = rb / nv, vy = (l1 - ra) / nv;
  const double s1 = std::sqrt(std::max(0.0, l1)), s2 = std::sqrt(std::max(0.0, l2));
  const double off = (s1 - s2) * vx * vy;
  return {s1 * vx * vx + s2 * vy * vy, off, off, s1 * vy * vy + s2 * vx * vx, std::max(0.0, -l2)};
}

// 1-D quadrature of box(a) ⊛ N(0, sigma_r^2): `strata` box strata (one when a = 0) x 3 Gauss-Hermite nodes (one
// when sigma_r = 0), offsets from the pixel centre and weights, strata-major.
struct Points1 {
  std::vector<double> o, w;
};
Points1 quadrature_points(double a, double sigma_r, int32_t strata) {
  const int32_t n = a > 0.0 ? strata : 1;
  Points1 p;
  for (int32_t i = 0; i < n; ++i) {
    const double beta = ((i + 0.5) / n - 0.5) * a;
    if (sigma_r > 0.0) {
      for (int j = 0; j < 3; ++j) {
        p.o.push_back(beta + sigma_r * kGhNode[j]);
        p.w.push_back(kGhWeight[j] / n);
      }
    } else {
      p.o.push_back(beta);
      p.w.push_back(1.0 / n);
    }
  }
  return p;
}

// Normalizes w to sum 1 and stores it as float with the rounding residual on the largest weight. A pixel without a
// positive finite weight sum is refused, so a numeric collapse can never index past its taps.
void store_weights(std::vector<double>& w, std::vector<float>* out) {
  double sum = 0.0;
  for (double x : w) sum += x;
  if (w.empty() || !(std::isfinite(sum) && sum > 0.0)) bad("a pixel's weights have no positive finite sum");
  size_t imax = 0;
  for (size_t i = 0; i < w.size(); ++i) {
    w[i] /= sum;
    if (w[i] > w[imax]) imax = i;
  }
  const size_t base = out->size();
  double fsum = 0.0;
  for (double x : w) {
    out->push_back(static_cast<float>(x));
    fsum += static_cast<double>(out->back());
  }
  float& big = (*out)[base + imax];
  big = static_cast<float>(static_cast<double>(big) + (1.0 - fsum));
}

// Appends the positive entries of acc (indices first + j) and their weights (store_weights); a non-finite entry is
// refused, not dropped.
void emit(const std::vector<double>& acc, int64_t first, std::vector<int32_t>& idx, std::vector<float>& wout) {
  std::vector<double> w;
  for (size_t j = 0; j < acc.size(); ++j) {
    if (!std::isfinite(acc[j])) bad("a non-finite weight");
    if (acc[j] > 0.0) {
      idx.push_back(static_cast<int32_t>(first + static_cast<int64_t>(j)));
      w.push_back(acc[j]);
    }
  }
  store_weights(w, &wout);
}

[[noreturn]] void bad_cover(const std::string& what) {
  bad(what + ", outside the source raster; the source must cover the ROI's preimage (overscan_pinhole)");
}

// Dense accumulator of one pixel's bilinear taps over their bounding box.
struct Accumulator {
  int64_t x0 = 0, y0 = 0, bw = 0, bh = 0;
  std::vector<double> grid;
  void reset(int64_t x_lo, int64_t y_lo, int64_t x_hi, int64_t y_hi) {
    x0 = x_lo, y0 = y_lo, bw = x_hi - x_lo + 1, bh = y_hi - y_lo + 1;
    grid.assign(static_cast<size_t>(bw * bh), 0.0);
  }
  void add(int64_t x, int64_t y, double w) { grid[static_cast<size_t>((y - y0) * bw + (x - x0))] += w; }
};

// Bilinear taps of q (source px): (floor q, frac q); the second tap only when its weight is non-zero.
struct Tap1 {
  int64_t i;
  double f;
};
Tap1 tap1(double q) {
  const double fl = std::floor(q);
  return {static_cast<int64_t>(fl), q - fl};
}

void check_tap(int64_t x, int64_t y, int32_t w, int32_t h, int32_t u, int32_t v) {
  if (x < 0 || y < 0 || x >= w || y >= h) {
    bad_cover("our pixel (" + std::to_string(u) + ", " + std::to_string(v) + ") reads source pixel (" +
              std::to_string(x) + ", " + std::to_string(y) + ") of " + std::to_string(w) + "x" + std::to_string(h));
  }
}
void check_axis(int64_t lo, int64_t hi, int32_t n, const char* name, int32_t k) {
  if (lo < 0 || hi >= n) {
    bad_cover(std::string("our ") + name + " " + std::to_string(k) + " reads source " + name + "s " +
              std::to_string(lo) + ".." + std::to_string(hi) + " of " + std::to_string(n));
  }
}

// One axis of a separable operator: output k in [k0, k1) of our image, source cells n_src.
struct Axis {
  int32_t k0, k1, n_src;
};

// Quadrature, zero distortion: q = f_s ((p - c) / f) + c_s per axis (the radtan unprojection of p is ((p - c)/f, 1)
// exactly, so this equals the 2-D path's projection bitwise).
void axis_quadrature(const Axis& ax, const Points1& pts, double c, double f, double c_s, double f_s,
                     std::vector<int32_t>& ptr, std::vector<int32_t>& idx, std::vector<float>& wout,
                     const char* name) {
  ptr.assign(1, 0);
  std::vector<double> acc;
  for (int32_t k = ax.k0; k < ax.k1; ++k) {
    int64_t lo = std::numeric_limits<int64_t>::max(), hi = std::numeric_limits<int64_t>::min();
    std::vector<Tap1> taps(pts.o.size());
    for (size_t i = 0; i < pts.o.size(); ++i) {
      taps[i] = tap1(f_s * ((static_cast<double>(k) + pts.o[i] - c) / f) + c_s);
      lo = std::min(lo, taps[i].i);
      hi = std::max(hi, taps[i].f > 0.0 ? taps[i].i + 1 : taps[i].i);
    }
    check_axis(lo, hi, ax.n_src, name, k);
    acc.assign(static_cast<size_t>(hi - lo + 1), 0.0);
    for (size_t i = 0; i < taps.size(); ++i) {
      acc[static_cast<size_t>(taps[i].i - lo)] += pts.w[i] * (1.0 - taps[i].f);
      if (taps[i].f > 0.0) acc[static_cast<size_t>(taps[i].i + 1 - lo)] += pts.w[i] * taps[i].f;
    }
    emit(acc, lo, idx, wout);
    ptr.push_back(static_cast<int32_t>(idx.size()));
  }
}

// Cells, identity map: cell i of the axis covers [origin - 0.5 + i/s, origin - 0.5 + (i + 1)/s) of our pixels.
// In cell units (cell m covers [m, m + 1)) the box [k - a/2, k + a/2] is cut at the cell edges (edges within
// kAlignTol of one snap to it). Each piece keeps its fraction b of the box and spreads from its centroid y by G
// sampled at the cell centres within n_sigma sigma of y (at least the nearest cell, or the two at a tie), normalized
// per piece. A whole cell (y = m + 1/2) takes the table g, G at whole-cell offsets: the design's indicator rule. A
// point (a = 0, or a box that rounding collapses) is one piece at the pixel centre. Returns whether every box is whole
// cells (edges on cell edges).
bool axis_cells(const Axis& ax, int32_t s, int64_t origin, double a, double sigma, double n_sigma,
                std::vector<int32_t>& ptr, std::vector<int32_t>& idx, std::vector<float>& wout, const char* name) {
  const double ss = sigma * s;
  const double reach_d = sigma > 0.0 ? std::floor(n_sigma * ss + kAlignTol) : 0.0;
  if (!(reach_d < ax.n_src)) {  // also keeps the casts below in range
    bad_cover(std::string("the blur reaches ") + std::to_string(reach_d) + " source " + name + "s of " +
              std::to_string(ax.n_src));
  }
  const auto reach = static_cast<int32_t>(reach_d);
  std::vector<double> g(static_cast<size_t>(2 * reach + 1));
  double zg = 0.0;
  for (int32_t d = -reach; d <= reach; ++d) {
    const double z = d / ss;
    g[static_cast<size_t>(d + reach)] = reach == 0 ? 1.0 : std::exp(-0.5 * z * z);
    zg += g[static_cast<size_t>(d + reach)];
  }
  const double half = std::max(n_sigma * ss, 0.5) + kAlignTol, ss2 = ss * ss;  // a cut piece's window about y
  struct Piece {
    int64_t m;     // its cell
    double b, y;   // fraction of the box, centroid
    bool whole() const { return y == static_cast<double>(m) + 0.5; }
  };
  std::vector<Piece> pieces;
  std::vector<double> acc, gp;
  bool aligned = true;
  ptr.assign(1, 0);
  for (int32_t k = ax.k0; k < ax.k1; ++k) {
    const double centre = (static_cast<double>(k - origin) + 0.5) * s;
    double lo = centre - 0.5 * a * s, hi = centre + 0.5 * a * s;
    if (std::abs(lo - std::round(lo)) <= kAlignTol) lo = std::round(lo);
    if (std::abs(hi - std::round(hi)) <= kAlignTol) hi = std::round(hi);
    if (!(lo >= 0.0 && hi <= ax.n_src)) {  // in double: a huge a must not reach the casts
      bad_cover(std::string("our ") + name + " " + std::to_string(k) + "'s box spans source " + name + "s " +
                std::to_string(lo) + " to " + std::to_string(hi) + " of " + std::to_string(ax.n_src));
    }
    pieces.clear();
    if (hi > lo) {
      aligned = aligned && lo == std::round(lo) && hi == std::round(hi);
      for (auto m = static_cast<int64_t>(std::floor(lo)); static_cast<double>(m) < hi; ++m) {
        const double p0 = std::max(static_cast<double>(m), lo), p1 = std::min(static_cast<double>(m + 1), hi);
        if (p1 > p0) pieces.push_back({m, (p1 - p0) / (hi - lo), 0.5 * (p0 + p1)});
      }
    } else {
      aligned = false;
      pieces.push_back({static_cast<int64_t>(std::floor(centre)), 1.0, centre});
    }
    // The window of each piece; pieces and windows ascend with y.
    const auto window = [&](const Piece& p) {
      return p.whole() ? std::pair<int64_t, int64_t>{p.m - reach, p.m + reach}
                       : std::pair<int64_t, int64_t>{static_cast<int64_t>(std::ceil(p.y - 0.5 - half)),
                                                     static_cast<int64_t>(std::floor(p.y - 0.5 + half))};
    };
    int64_t first = std::numeric_limits<int64_t>::max(), last = std::numeric_limits<int64_t>::min();
    for (const Piece& p : pieces) {
      const auto [f, l] = window(p);
      first = std::min(first, f), last = std::max(last, l);
    }
    check_axis(first, last, ax.n_src, name, k);
    acc.assign(static_cast<size_t>(last - first + 1), 0.0);
    for (const Piece& p : pieces) {
      if (p.whole()) {
        for (int32_t d = -reach; d <= reach; ++d) {
          acc[static_cast<size_t>(p.m + d - first)] += p.b * g[static_cast<size_t>(d + reach)];
        }
        continue;
      }
      // Relative to the nearest cell's weight (1, exactly, at a tie), so neither a tiny nor a huge sigma s can make
      // them all underflow or overflow; scaled to g's sum so that every piece carries b.
      const auto [f, l] = window(p);
      gp.assign(static_cast<size_t>(l - f + 1), 0.0);
      double d2min = std::numeric_limits<double>::infinity();
      for (int64_t m = f; m <= l; ++m) {
        const double d = static_cast<double>(m) + 0.5 - p.y;
        gp[static_cast<size_t>(m - f)] = d * d;
        d2min = std::min(d2min, d * d);
      }
      double zp = 0.0;
      for (double& v : gp) {
        v = v == d2min ? 1.0 : std::exp(-0.5 * ((v - d2min) / ss2));  // ss2 = 0 (sigma = 0): exp(-inf) = 0
        zp += v;
      }
      const double scale = p.b * zg / zp;
      for (int64_t m = f; m <= l; ++m) acc[static_cast<size_t>(m - first)] += scale * gp[static_cast<size_t>(m - f)];
    }
    emit(acc, first, idx, wout);
    ptr.push_back(static_cast<int32_t>(idx.size()));
  }
  return aligned;
}

// Each quadrature pixel projects (3 strata)^2 points, held per build thread: 64 strata (the design uses 16) is 36864
// points per pixel, 16 times the default's work and 0.6 MB per thread; far more would exhaust memory or never finish.
constexpr int32_t kMaxStrata = 64;

void check_spec(const ResampleSpec& r) {
  const bool ok = std::isfinite(r.aperture_px) && r.aperture_px >= 0.0 && std::isfinite(r.sigma_px) &&
                  r.sigma_px >= 0.0 && r.aperture_px + r.sigma_px > 0.0 && std::isfinite(r.truncation_sigmas) &&
                  r.truncation_sigmas > 0.0 && r.strata >= 1 && r.strata <= kMaxStrata &&
                  (r.rule == Rule::kCells || r.rule == Rule::kQuadrature) &&
                  (r.blur == Blur::kLocal || r.blur == Blur::kNominal);
  if (!ok) {
    bad("needs aperture_px >= 0, sigma_px >= 0 (not both 0), truncation_sigmas > 0 and strata in [1, " +
        std::to_string(kMaxStrata) + "]");
  }
}

void check_source(const SourceCamera& s) {
  if (s.kind == SourceCamera::Kind::kEquidistant) {
    throw FramesError("FRM-KB-WIDE", "route S: the equidistant source for wide Kannala-Brandt cameras is FR-3b; use "
                                     "route osl");
  }
  const bool ok = std::isfinite(s.fx) && s.fx > 0.0 && std::isfinite(s.fy) && s.fy > 0.0 && std::isfinite(s.cx) &&
                  std::isfinite(s.cy) && s.width >= 1 && s.height >= 1 && s.s >= 1 &&
                  static_cast<uint64_t>(s.width) * static_cast<uint64_t>(s.height) <= (uint64_t{1} << 32);
  if (!ok) bad("the source camera needs fx, fy > 0, finite centres, s >= 1 and 1 <= width height <= 2^32 pixels");
}

}  // namespace

double resample_support_px(const ResampleSpec& r) { return 0.5 * r.aperture_px + r.truncation_sigmas * r.sigma_px; }

LensModel lens_model(const render::Camera& c, double support_px) {
  if (c.width <= 0 || c.height <= 0 || !(std::isfinite(c.fx) && c.fx > 0.0) || !(std::isfinite(c.fy) && c.fy > 0.0)) {
    bad("our camera needs a positive size and focal lengths");
  }
  const auto k = [&](size_t i) { return i < c.dist.size() ? c.dist[i] : 0.0; };
  LensModel m{};
  m.kind = c.model;
  m.width = c.width;
  m.height = c.height;
  m.support_px = support_px;
  if (c.model == render::CameraModelKind::kRadtan) {
    m.radtan = {c.fx, c.fy, c.cx, c.cy, k(0), k(1), k(2), k(3), k(4), 0.0, 0.0};
    m.radtan.r_lim = camera::radtan_injective_domain(m.radtan, c.width, c.height, support_px).r_lim;
  } else {
    m.kb = {c.fx, c.fy, c.cx, c.cy, k(0), k(1), k(2), k(3), 0.0};
    m.kb.theta_lim = camera::kannala_brandt_injective_domain(m.kb, c.width, c.height, support_px).theta_lim;
  }
  return m;
}

ResampleOperator ResampleOperator::build(const render::Camera& ours, const SourceCamera& src, const ResampleSpec& spec,
                                         Roi roi, int threads) {
  check_spec(spec);
  return build(lens_model(ours, resample_support_px(spec)), src, spec, roi, threads);
}

ResampleOperator ResampleOperator::build(const LensModel& lens, const SourceCamera& src, const ResampleSpec& spec,
                                         Roi roi, int threads) {
  check_spec(spec);
  check_source(src);
  check_roi(roi, lens.width, lens.height);
  ResampleOperator op;
  op.roi_ = roi;
  op.src_w_ = src.width;
  op.src_h_ = src.height;
  const int32_t W = roi.x1 - roi.x0, H = roi.y1 - roi.y0;
  const double cell = 1.0 / src.s;
  const auto& rc = lens.radtan;
  const double fx = lens.kind == render::CameraModelKind::kRadtan ? rc.fx : lens.kb.fx;

  if (spec.rule == Rule::kCells) {
    // Identity map: zero distortion, fx = fy, source cells that subdivide our pixels.
    const double ox = rc.cx - (src.cx + 0.5) / src.s + 0.5, oy = rc.cy - (src.cy + 0.5) / src.s + 0.5;
    const double fs = src.s * fx;
    if (!zero_distortion(lens) || rc.fx != rc.fy || std::abs(src.fx - fs) > 1e-12 * fs ||
        std::abs(src.fy - fs) > 1e-12 * fs || std::abs(ox - std::round(ox)) > kAlignTol ||
        std::abs(oy - std::round(oy)) > kAlignTol) {
      bad("the cells rule needs the identity map: zero distortion, fx = fy, and a source with fx = fy = s fx whose "
          "cells subdivide our pixels (overscan_pinhole)");
    }
    op.separable_ = true;
    const bool ax = axis_cells({roi.x0, roi.x1, src.width}, src.s, static_cast<int64_t>(std::round(ox)),
                               spec.aperture_px, spec.sigma_px, spec.truncation_sigmas, op.tx_.ptr, op.tx_.idx,
                               op.tx_.w, "column");
    const bool ay = axis_cells({roi.y0, roi.y1, src.height}, src.s, static_cast<int64_t>(std::round(oy)),
                               spec.aperture_px, spec.sigma_px, spec.truncation_sigmas, op.ty_.ptr, op.ty_.idx,
                               op.ty_.w, "row");
    if (!(ax && ay) && spec.sigma_px < cell) {
      char as[32];
      std::snprintf(as, sizeof as, "%.6g", spec.aperture_px * src.s);
      op.warnings_.push_back({"FRM-APERTURE-QUANT", std::string("route S cells rule: the box (a s = ") + as +
                                                        " source cells) is not whole cells and sigma < 1/s, so its "
                                                        "cut pieces are quantized to cells"});
    }
    op.nan_.assign(static_cast<size_t>(W) * static_cast<size_t>(H), 0);
  } else {
    const bool local = spec.blur == Blur::kLocal;
    if (zero_distortion(lens)) {
      // Affine per axis: A = diag(f / f_s) everywhere (kNominal: 1/s).
      op.separable_ = true;
      const ResidualBlur rb = residual_blur(spec.sigma_px, local ? rc.fx / src.fx : cell, 0.0, 0.0,
                                            local ? rc.fy / src.fy : cell);
      op.excess_ = rb.excess;
      const Points1 px = quadrature_points(spec.aperture_px, rb.l00, spec.strata);
      const Points1 py = quadrature_points(spec.aperture_px, rb.l11, spec.strata);
      axis_quadrature({roi.x0, roi.x1, src.width}, px, rc.cx, rc.fx, src.cx, src.fx, op.tx_.ptr, op.tx_.idx,
                      op.tx_.w, "column");
      axis_quadrature({roi.y0, roi.y1, src.height}, py, rc.cy, rc.fy, src.cy, src.fy, op.ty_.ptr, op.ty_.idx,
                      op.ty_.w, "row");
      // A pixel has a preimage when its farthest point does (the corners of its point grid).
      const auto [xmin, xmax] = std::minmax_element(px.o.begin(), px.o.end());
      const auto [ymin, ymax] = std::minmax_element(py.o.begin(), py.o.end());
      op.nan_.assign(static_cast<size_t>(W) * static_cast<size_t>(H), 0);
      for (int32_t y = 0; y < H; ++y) {
        for (int32_t x = 0; x < W; ++x) {
          for (double dx : {*xmin, *xmax}) {
            for (double dy : {*ymin, *ymax}) {
              Vec3<double> ray{};
              if (!unproject(lens, roi.x0 + x + dx, roi.y0 + y + dy, &ray)) {
                op.nan_[static_cast<size_t>(y) * W + x] = 1;
              }
            }
          }
        }
      }
    } else {
      // CSR: per ROI row, each pixel's taps in source order. Points (ix, iy), i = stratum * ng + node per axis:
      // offset (b_x + L00 g_x + L01 g_y, b_y + L10 g_x + L11 g_y), weight (gw_x / n)(gw_y / n).
      const ResidualBlur nominal = residual_blur(spec.sigma_px, cell, 0.0, 0.0, cell);
      const int32_t n = spec.aperture_px > 0.0 ? spec.strata : 1;
      const int32_t ng = spec.sigma_px > 0.0 && (local || nominal.l00 > 0.0) ? 3 : 1;
      std::vector<double> bo(static_cast<size_t>(n)), go(static_cast<size_t>(ng)), gw(static_cast<size_t>(ng));
      for (int32_t i = 0; i < n; ++i) bo[static_cast<size_t>(i)] = ((i + 0.5) / n - 0.5) * spec.aperture_px;
      for (int32_t j = 0; j < ng; ++j) {
        go[static_cast<size_t>(j)] = ng == 1 ? 0.0 : kGhNode[j];
        gw[static_cast<size_t>(j)] = (ng == 1 ? 1.0 : kGhWeight[j]) / n;
      }
      const size_t n1 = static_cast<size_t>(n) * static_cast<size_t>(ng);
      struct Row {
        std::vector<int64_t> count;
        std::vector<uint32_t> col;
        std::vector<float> w;
        std::vector<uint8_t> nan;
        double excess = 0.0;
      };
      std::vector<Row> rows(static_cast<size_t>(H));
      render::ThreadPool pool(pool_size(threads));
      pool.parallel_for(static_cast<size_t>(H), 1, [&](size_t b, size_t e) {
        std::vector<double> qx(n1 * n1), qy(n1 * n1);
        Accumulator acc;
        std::vector<double> w;
        // A quadrature point: false without a preimage; FRM-KB-WIDE for a ray at or beyond 90 degrees.
        const auto project = [&](double pu, double pv, double* ox, double* oy) {
          Vec3<double> ray{};
          if (!unproject(lens, pu, pv, &ray)) return false;
          if (!(ray.z > 0.0)) kb_wide("our point (" + std::to_string(pu) + ", " + std::to_string(pv) + ")");
          *ox = src.fx * (ray.x / ray.z) + src.cx;
          *oy = src.fy * (ray.y / ray.z) + src.cy;
          return true;
        };
        // A Jacobian probe: never fails the pixel or throws (only the quadrature points decide NaN, §4.7.2).
        const auto probe = [&](double pu, double pv, double* ox, double* oy) {
          Vec3<double> ray{};
          if (!unproject(lens, pu, pv, &ray) || !(ray.z > 0.0)) return false;
          *ox = src.fx * (ray.x / ray.z) + src.cx;
          *oy = src.fy * (ray.y / ray.z) + src.cy;
          return std::isfinite(*ox) && std::isfinite(*oy);
        };
        const double h = std::min(0.5, lens.support_px);  // probes stay inside the certified support rectangle
        for (size_t yr = b; yr < e; ++yr) {
          Row& row = rows[yr];
          const int32_t v = roi.y0 + static_cast<int32_t>(yr);
          for (int32_t x = 0; x < W; ++x) {
            const int32_t u = roi.x0 + x;
            bool ok = true;
            ResidualBlur rb = nominal;
            if (local) {  // J = d(source)/d(ours) by central differences at +-h; A = J^-1; else the nominal rule
              double a[8];
              if (h > 0.0 && probe(u + h, v, &a[0], &a[1]) && probe(u - h, v, &a[2], &a[3]) &&
                  probe(u, v + h, &a[4], &a[5]) && probe(u, v - h, &a[6], &a[7])) {
                const double j00 = (a[0] - a[2]) / (2 * h), j10 = (a[1] - a[3]) / (2 * h);
                const double j01 = (a[4] - a[6]) / (2 * h), j11 = (a[5] - a[7]) / (2 * h);
                const double det = j00 * j11 - j01 * j10;
                if (det > 0.0 && std::isfinite(det)) {
                  rb = residual_blur(spec.sigma_px, j11 / det, -j01 / det, -j10 / det, j00 / det);
                }
              }
            }
            for (size_t iy = 0; iy < n1 && ok; ++iy) {
              const double by = bo[iy / static_cast<size_t>(ng)], gy = go[iy % static_cast<size_t>(ng)];
              for (size_t ix = 0; ix < n1; ++ix) {
                const double bx = bo[ix / static_cast<size_t>(ng)], gx = go[ix % static_cast<size_t>(ng)];
                const double ox = bx + rb.l00 * gx + rb.l01 * gy, oy = by + rb.l10 * gx + rb.l11 * gy;
                if (!project(u + ox, v + oy, &qx[iy * n1 + ix], &qy[iy * n1 + ix])) {
                  ok = false;
                  break;
                }
              }
            }
            row.nan.push_back(ok ? 0 : 1);
            if (!ok) {
              row.count.push_back(0);
              continue;
            }
            row.excess = std::max(row.excess, rb.excess);
            int64_t xlo = std::numeric_limits<int64_t>::max(), ylo = xlo, xhi = std::numeric_limits<int64_t>::min(),
                    yhi = xhi;
            for (size_t i = 0; i < n1 * n1; ++i) {
              const Tap1 tx = tap1(qx[i]), ty = tap1(qy[i]);
              xlo = std::min(xlo, tx.i), ylo = std::min(ylo, ty.i);
              xhi = std::max(xhi, tx.f > 0.0 ? tx.i + 1 : tx.i), yhi = std::max(yhi, ty.f > 0.0 ? ty.i + 1 : ty.i);
            }
            check_tap(xlo, ylo, src.width, src.height, u, v);
            check_tap(xhi, yhi, src.width, src.height, u, v);
            acc.reset(xlo, ylo, xhi, yhi);
            for (size_t iy = 0; iy < n1; ++iy) {
              for (size_t ix = 0; ix < n1; ++ix) {
                const size_t i = iy * n1 + ix;
                const double pw = gw[ix % static_cast<size_t>(ng)] * gw[iy % static_cast<size_t>(ng)];
                const Tap1 tx = tap1(qx[i]), ty = tap1(qy[i]);
                acc.add(tx.i, ty.i, pw * (1.0 - tx.f) * (1.0 - ty.f));
                if (tx.f > 0.0) acc.add(tx.i + 1, ty.i, pw * tx.f * (1.0 - ty.f));
                if (ty.f > 0.0) acc.add(tx.i, ty.i + 1, pw * (1.0 - tx.f) * ty.f);
                if (tx.f > 0.0 && ty.f > 0.0) acc.add(tx.i + 1, ty.i + 1, pw * tx.f * ty.f);
              }
            }
            w.clear();
            const size_t before = row.col.size();
            for (int64_t j = 0; j < acc.bh; ++j) {
              for (int64_t i = 0; i < acc.bw; ++i) {
                const double g = acc.grid[static_cast<size_t>(j * acc.bw + i)];
                if (g > 0.0) {
                  row.col.push_back(static_cast<uint32_t>((acc.y0 + j) * src.width + (acc.x0 + i)));
                  w.push_back(g);
                }
              }
            }
            store_weights(w, &row.w);
            row.count.push_back(static_cast<int64_t>(row.col.size() - before));
          }
        }
      });
      op.excess_ = local ? 0.0 : nominal.excess;  // local: the largest over the pixels with a preimage
      op.ptr_.assign(1, 0);
      for (Row& row : rows) {
        for (int64_t c : row.count) op.ptr_.push_back(op.ptr_.back() + c);
        op.col_.insert(op.col_.end(), row.col.begin(), row.col.end());
        op.w_.insert(op.w_.end(), row.w.begin(), row.w.end());
        op.nan_.insert(op.nan_.end(), row.nan.begin(), row.nan.end());
        op.excess_ = std::max(op.excess_, row.excess);
        row = Row{};
      }
    }
    if (op.excess_ > 0.0) {
      op.warnings_.push_back({"FRM-RESAMPLE-BLUR", "route S: the source cell and the bilinear taps add up to " +
                                                       std::to_string(op.excess_) +
                                                       " px^2 of variance beyond the pixel response (sigma^2 < "
                                                       "(1/s)^2/4 at the local magnification)"});
    }
  }

  // Source rows per ROI row (tiles) and the source rectangle read.
  op.row_lo_.assign(static_cast<size_t>(H), std::numeric_limits<int32_t>::max());
  op.row_hi_.assign(static_cast<size_t>(H), std::numeric_limits<int32_t>::min());
  int32_t xmin = std::numeric_limits<int32_t>::max(), xmax = std::numeric_limits<int32_t>::min();
  if (op.separable_) {
    for (int32_t y = 0; y < H; ++y) {
      const auto b = static_cast<size_t>(op.ty_.ptr[static_cast<size_t>(y)]);
      const auto e = static_cast<size_t>(op.ty_.ptr[static_cast<size_t>(y) + 1]);
      op.row_lo_[static_cast<size_t>(y)] = op.ty_.idx[b];
      op.row_hi_[static_cast<size_t>(y)] = op.ty_.idx[e - 1];
    }
    xmin = op.tx_.idx.front(), xmax = op.tx_.idx.back();
  } else {
    for (int32_t y = 0; y < H; ++y) {
      for (int32_t x = 0; x < W; ++x) {
        const size_t p = static_cast<size_t>(y) * W + x;
        for (int64_t t = op.ptr_[p]; t < op.ptr_[p + 1]; ++t) {
          const auto c = op.col_[static_cast<size_t>(t)];
          const auto r = static_cast<int32_t>(c / static_cast<uint32_t>(src.width));
          const auto cx = static_cast<int32_t>(c % static_cast<uint32_t>(src.width));
          op.row_lo_[static_cast<size_t>(y)] = std::min(op.row_lo_[static_cast<size_t>(y)], r);
          op.row_hi_[static_cast<size_t>(y)] = std::max(op.row_hi_[static_cast<size_t>(y)], r);
          xmin = std::min(xmin, cx), xmax = std::max(xmax, cx);
        }
      }
    }
  }
  const auto [ylo, yhi] = std::pair{*std::min_element(op.row_lo_.begin(), op.row_lo_.end()),
                                    *std::max_element(op.row_hi_.begin(), op.row_hi_.end())};
  op.src_rect_ = ylo <= yhi ? Roi{xmin, ylo, xmax + 1, yhi + 1} : Roi{0, 0, 0, 0};
  return op;
}

void ResampleOperator::apply(const float* src, int32_t channels, float* out) const {
  if (channels < 1) bad("channels must be >= 1");
  apply_rows(src, 0, src_h_, channels, out, false);
}

void ResampleOperator::apply_tiles(const RawImage& tile, int32_t y0, int32_t channels, float* out) const {
  if (channels < 1 || tile.c != channels || tile.w != src_w_ || tile.h < 1 || y0 < 0 || y0 > src_h_ - tile.h) {
    bad("a tile must hold whole source rows inside the " + std::to_string(src_w_) + "x" + std::to_string(src_h_) +
        " raster with `channels` channels");
  }
  apply_rows(tile.data, y0, tile.h, channels, out, true);
}

// Rows [row0, row0 + nrows) of the source are at `rows`; add: out += their contribution, else out = it (all rows).
void ResampleOperator::apply_rows(const float* rows, int32_t row0, int32_t nrows, int32_t channels, float* out,
                                  bool add) const {
  const int32_t W = roi_.x1 - roi_.x0, H = roi_.y1 - roi_.y0;
  const auto C = static_cast<size_t>(channels);
  const int32_t row1 = row0 + nrows;  // exclusive
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const auto put = [&](size_t o, double v) {
    out[o] = add ? static_cast<float>(static_cast<double>(out[o]) + v) : static_cast<float>(v);
  };
  if (separable_) {
    constexpr int32_t kBlock = 16;  // output rows per horizontal pass
    std::vector<double> tmp, acc(C);
    for (int32_t yb = 0; yb < H; yb += kBlock) {
      const int32_t ye = std::min(H, yb + kBlock);
      const int32_t lo = std::max(row0, row_lo_[static_cast<size_t>(yb)]);
      const int32_t hi = std::min(row1 - 1, row_hi_[static_cast<size_t>(ye - 1)]);
      if (lo <= hi) {
        tmp.assign(static_cast<size_t>(hi - lo + 1) * static_cast<size_t>(W) * C, 0.0);
        for (int32_t j = lo; j <= hi; ++j) {
          const float* srow = rows + static_cast<size_t>(j - row0) * static_cast<size_t>(src_w_) * C;
          double* trow = tmp.data() + static_cast<size_t>(j - lo) * static_cast<size_t>(W) * C;
          for (int32_t x = 0; x < W; ++x) {
            for (int32_t t = tx_.ptr[static_cast<size_t>(x)]; t < tx_.ptr[static_cast<size_t>(x) + 1]; ++t) {
              const double w = static_cast<double>(tx_.w[static_cast<size_t>(t)]);
              const float* s = srow + static_cast<size_t>(tx_.idx[static_cast<size_t>(t)]) * C;
              for (size_t c = 0; c < C; ++c) trow[static_cast<size_t>(x) * C + c] += w * static_cast<double>(s[c]);
            }
          }
        }
      }
      for (int32_t y = yb; y < ye; ++y) {
        for (int32_t x = 0; x < W; ++x) {
          const size_t o = (static_cast<size_t>(y) * W + x) * C;
          if (nan_[static_cast<size_t>(y) * W + x] != 0) {
            for (size_t c = 0; c < C; ++c) out[o + c] = nan;
            continue;
          }
          std::fill(acc.begin(), acc.end(), 0.0);
          for (int32_t t = ty_.ptr[static_cast<size_t>(y)]; t < ty_.ptr[static_cast<size_t>(y) + 1]; ++t) {
            const int32_t j = ty_.idx[static_cast<size_t>(t)];
            if (j < lo || j > hi) continue;
            const double w = static_cast<double>(ty_.w[static_cast<size_t>(t)]);
            const double* trow = tmp.data() + (static_cast<size_t>(j - lo) * W + x) * C;
            for (size_t c = 0; c < C; ++c) acc[c] += w * trow[c];
          }
          for (size_t c = 0; c < C; ++c) put(o + c, acc[c]);
        }
      }
    }
    return;
  }
  std::vector<double> acc(C);
  const auto sw = static_cast<uint32_t>(src_w_);
  for (int32_t y = 0; y < H; ++y) {
    const bool touches = row_lo_[static_cast<size_t>(y)] < row1 && row_hi_[static_cast<size_t>(y)] >= row0;
    for (int32_t x = 0; x < W; ++x) {
      const size_t p = static_cast<size_t>(y) * W + x, o = p * C;
      if (nan_[p] != 0) {
        for (size_t c = 0; c < C; ++c) out[o + c] = nan;
        continue;
      }
      if (!touches && add) continue;
      std::fill(acc.begin(), acc.end(), 0.0);
      for (int64_t t = ptr_[p]; t < ptr_[p + 1]; ++t) {
        const uint32_t col = col_[static_cast<size_t>(t)];
        const auto r = static_cast<int32_t>(col / sw);
        if (r < row0 || r >= row1) continue;
        const double w = static_cast<double>(w_[static_cast<size_t>(t)]);
        const float* s = rows + (static_cast<size_t>(r - row0) * sw + col % sw) * C;
        for (size_t c = 0; c < C; ++c) acc[c] += w * static_cast<double>(s[c]);
      }
      for (size_t c = 0; c < C; ++c) put(o + c, acc[c]);
    }
  }
}

size_t ResampleOperator::nnz() const {
  if (!separable_) return col_.size();
  const int32_t W = roi_.x1 - roi_.x0, H = roi_.y1 - roi_.y0;
  size_t n = 0;
  for (int32_t y = 0; y < H; ++y) {
    const auto ny = static_cast<size_t>(ty_.ptr[static_cast<size_t>(y) + 1] - ty_.ptr[static_cast<size_t>(y)]);
    for (int32_t x = 0; x < W; ++x) {
      if (nan_[static_cast<size_t>(y) * W + x] != 0) continue;
      n += ny * static_cast<size_t>(tx_.ptr[static_cast<size_t>(x) + 1] - tx_.ptr[static_cast<size_t>(x)]);
    }
  }
  return n;
}

int64_t ResampleOperator::nan_pixels() const { return std::count(nan_.begin(), nan_.end(), uint8_t{1}); }

size_t ResampleOperator::bytes() const {
  const auto tab = [](const Taps1& t) { return 4 * (t.ptr.size() + t.idx.size() + t.w.size()); };
  return tab(tx_) + tab(ty_) + 8 * ptr_.size() + 4 * (col_.size() + w_.size()) + 4 * (row_lo_.size() + row_hi_.size()) +
         nan_.size();
}

std::vector<std::pair<int64_t, double>> ResampleOperator::weights(int32_t x, int32_t y) const {
  const int32_t W = roi_.x1 - roi_.x0, H = roi_.y1 - roi_.y0;
  if (x < 0 || y < 0 || x >= W || y >= H) bad("weights: pixel outside the ROI");
  const size_t p = static_cast<size_t>(y) * W + x;
  std::vector<std::pair<int64_t, double>> out;
  if (nan_[p] != 0) return out;
  if (!separable_) {
    for (int64_t t = ptr_[p]; t < ptr_[p + 1]; ++t) {
      out.emplace_back(col_[static_cast<size_t>(t)], static_cast<double>(w_[static_cast<size_t>(t)]));
    }
    return out;
  }
  for (int32_t ty = ty_.ptr[static_cast<size_t>(y)]; ty < ty_.ptr[static_cast<size_t>(y) + 1]; ++ty) {
    for (int32_t tx = tx_.ptr[static_cast<size_t>(x)]; tx < tx_.ptr[static_cast<size_t>(x) + 1]; ++tx) {
      out.emplace_back(static_cast<int64_t>(ty_.idx[static_cast<size_t>(ty)]) * src_w_ +
                           tx_.idx[static_cast<size_t>(tx)],
                       static_cast<double>(ty_.w[static_cast<size_t>(ty)]) *
                           static_cast<double>(tx_.w[static_cast<size_t>(tx)]));
    }
  }
  return out;
}

SourceCamera overscan_pinhole(const render::Camera& ours, double support_px, int32_t s, double margin_px,
                              const Roi* roi) {
  return overscan_pinhole(lens_model(ours, support_px), s, margin_px, roi);
}

SourceCamera overscan_pinhole(const LensModel& lens, int32_t s, double margin_px, const Roi* roi) {
  if (s < 1 || !(std::isfinite(margin_px) && margin_px >= 0.0)) bad("overscan needs s >= 1 and margin_px >= 0");
  const Roi r = roi != nullptr ? *roi : Roi{0, 0, lens.width, lens.height};
  check_roi(r, lens.width, lens.height);
  const bool kb = lens.kind == render::CameraModelKind::kKannalaBrandt;
  const double fx = kb ? lens.kb.fx : lens.radtan.fx, cx = kb ? lens.kb.cx : lens.radtan.cx;
  const double cy = kb ? lens.kb.cy : lens.radtan.cy;
  // The preimage of the support rectangle is bounded by the preimage of its boundary (our unprojection is injective
  // on the domain), sampled every 1/8 px; the margin covers the bilinear taps and the sampling.
  double umin = std::numeric_limits<double>::infinity(), umax = -umin, vmin = umin, vmax = -umin;
  const double sup = lens.support_px;
  const double u0 = r.x0 - sup, u1 = r.x1 - 1 + sup, v0 = r.y0 - sup, v1 = r.y1 - 1 + sup;
  const auto visit = [&](double u, double v) {
    Vec3<double> ray{};
    if (!unproject(lens, u, v, &ray)) bad("our camera's domain does not cover its support rectangle");
    if (!(ray.z > 0.0)) kb_wide("our image");
    const double pu = fx * (ray.x / ray.z) + cx, pv = fx * (ray.y / ray.z) + cy;
    umin = std::min(umin, pu), umax = std::max(umax, pu), vmin = std::min(vmin, pv), vmax = std::max(vmax, pv);
  };
  const auto nu = static_cast<int64_t>(std::ceil((u1 - u0) * 8.0));
  const auto nv = static_cast<int64_t>(std::ceil((v1 - v0) * 8.0));
  for (int64_t i = 0; i <= nu; ++i) {
    const double u = nu == 0 ? u0 : u0 + (u1 - u0) * static_cast<double>(i) / static_cast<double>(nu);
    visit(u, v0), visit(u, v1);
  }
  for (int64_t i = 0; i <= nv; ++i) {
    const double v = nv == 0 ? v0 : v0 + (v1 - v0) * static_cast<double>(i) / static_cast<double>(nv);
    visit(u0, v), visit(u1, v);
  }
  const auto margin = [](double need) { return std::max(0.0, std::ceil(need - kAlignTol)); };  // integer-valued
  const double mld = margin(r.x0 - 0.5 - umin + margin_px), mrd = margin(umax + margin_px - (r.x1 - 0.5));
  const double mtd = margin(r.y0 - 0.5 - vmin + margin_px), mbd = margin(vmax + margin_px - (r.y1 - 0.5));
  const double wd = s * (r.x1 - r.x0 + mld + mrd), hd = s * (r.y1 - r.y0 + mtd + mbd);
  if (!(wd <= kMaxRasterSide && hd <= kMaxRasterSide)) {  // also a ray near 90 degrees (huge or infinite cover)
    throw FramesError(kb ? "FRM-KB-WIDE" : "FRM-ROUTE",
                      "route S: the pinhole cover needs a " + std::to_string(wd) + " x " + std::to_string(hd) +
                          " px raster, beyond Blender's 65536 px per side; use route osl");
  }
  const auto ml = static_cast<int64_t>(mld), mt = static_cast<int64_t>(mtd);
  const auto width = static_cast<int64_t>(wd), height = static_cast<int64_t>(hd);
  SourceCamera out{};
  out.kind = SourceCamera::Kind::kPinhole;
  out.fx = out.fy = s * fx;
  out.cx = s * (cx + 0.5 - static_cast<double>(r.x0 - ml)) - 0.5;
  out.cy = s * (cy + 0.5 - static_cast<double>(r.y0 - mt)) - 0.5;
  out.width = static_cast<int32_t>(width);
  out.height = static_cast<int32_t>(height);
  out.s = s;
  out.margin_rows = static_cast<int32_t>(mt);
  return out;
}

}  // namespace balla::frames
