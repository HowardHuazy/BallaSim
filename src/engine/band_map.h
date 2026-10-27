// P2-B B6c: the band N's per-pixel map (accuracy.band_growth pixel; ENGINE_CPP §0.1 "P2-B B6c notes"), shared by the
// host and the device. Each seed i of the plan pass (plan.h PixelPlan: plan seeds, carried seeds) reaches the pixels q
// with ||q - i||_inf <= r_i, r_i = floor(rho_i + s + M_i): rho_i 1 for a plan seed (a discontinuity between centre rays
// lies within 1 px), s for a carried seed (its sub-sample lies within the support s), s the support again (the pixels
// whose sub-samples reach the edge), and M_i the edge's motion over the window in px, max(sigma_cam M_static, sigma_mov
// M_mov) (PixelPlan::seed_m; a carried seed also takes its tile's touches() floor, scaled by sigma_mov), at most
// max(w, h). The band is the exact union of these L-inf balls, by the radius cascade (the band memo, pixel_design §4):
// from the largest radius down, acc = D_d(acc | seeds of radius r_j), d = r_j - r_{j+1} (r_K = 0), each D_d a separable
// sliding-window OR along the rows, then the columns. D_a D_b = D_{a+b} for squares, also clipped to a rectangle (per
// axis an interval of points within both reaches stays inside it), so the cascade is exact on its domain: there the
// union of the balls of the domain's seeds. The camera's domain is the crop grown by the largest radius R (clipped to
// the image), where every seed that reaches the crop lies, so the band is exact on the crop, the only pixels that read
// it (the active pixels); its cost is K levels of two line passes over the domain, K the distinct radii
// (band_radius_levels).
#pragma once

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "core/hd.h"
#include "engine/par.h"
#include "engine/plan.h"

namespace balla::engine {

// Each pixel's radius (px; -1: not a seed) and the histogram of the radii (levels[r] += 1, r <= cap). rho: 1 for a plan
// seed (plan.h kSeedPlan), s for a carried seed (kSeedCarried, which also takes its tile's touches() floor), the larger
// for both.
struct BandRadius {
  const double* seed_m;      // PixelPlan's M (px)
  const uint8_t* seed_kind;  // kSeedPlan | kSeedCarried, 0: not a seed
  const double* tile_floor;  // per tile: sigma_mov times the touches() floor (px; NaN or inf: unbounded)
  const uint32_t* pixel_tile;
  double support;            // s
  int32_t cap;               // max(width, height)
  int32_t* radius;
  uint32_t* levels;          // cap + 1
  BALLA_HD void operator()(size_t i) const {
    const uint8_t kind = seed_kind[i];
    int32_t r = -1;
    if (kind != 0) {
      const bool carried = (kind & kSeedCarried) != 0;
      const double m = carried ? np::maximum(seed_m[i], tile_floor[pixel_tile[i]]) : seed_m[i];
      const double rho_plan = (kind & kSeedPlan) != 0 ? 1.0 : 0.0, rho_carried = carried ? support : 0.0;
      const double x = floor(((rho_plan > rho_carried ? rho_plan : rho_carried) + support) + m);
      r = x < static_cast<double>(cap) ? static_cast<int32_t>(x) : cap;  // NaN and inf: the cap
      atomic_increment(&levels[r]);
    }
    radius[i] = r;
  }
};

// The cascade's domain, a rectangle of the image: x0 <= x < x1, y0 <= y < y1.
struct BandDomain {
  int32_t x0, y0, x1, y1;
};

// The domain of a crop (the image without one) at the largest radius r: the crop grown by r, clipped to the image.
inline BandDomain band_domain(int32_t x0, int32_t y0, int32_t x1, int32_t y1, int32_t r, int32_t width,
                              int32_t height) {
  const int32_t g = r > 0 ? r : 0;
  return {x0 - g > 0 ? x0 - g : 0, y0 - g > 0 ? y0 - g : 0, x1 + g < width ? x1 + g : width,
          y1 + g < height ? y1 + g : height};
}

// One sliding-window OR of half-width d along every row (rows) or column of the domain, line `line` of it: out[q] = 1
// where a set pixel of the domain lies within d of q on the line; set: in[q] != 0, or radius[q] == level (radius not
// null). Two scans per line (the nearest set pixel on each side), so O(line) for any d. Pixels by id (width a row).
struct DilateLines {
  const uint8_t* in;
  const int32_t* radius;
  int32_t level;
  uint8_t* out;
  int32_t width;
  BandDomain dom;
  bool rows;
  int32_t d;
  BALLA_HD bool set(size_t q) const { return in[q] != 0 || (radius != nullptr && radius[q] == level); }
  BALLA_HD void operator()(size_t line) const {
    const int32_t n = rows ? dom.x1 - dom.x0 : dom.y1 - dom.y0;
    const size_t step = rows ? 1 : static_cast<size_t>(width);
    const auto l = static_cast<int32_t>(line);
    const size_t base = static_cast<size_t>(rows ? dom.y0 + l : dom.y0) * static_cast<size_t>(width) +
                        static_cast<size_t>(rows ? dom.x0 : dom.x0 + l);
    int32_t last = -d - 1;  // none: out of reach
    for (int32_t k = 0; k < n; ++k) {
      const size_t q = base + static_cast<size_t>(k) * step;
      if (set(q)) last = k;
      out[q] = k - last <= d ? 1 : 0;
    }
    int32_t next = n + d;  // none: out of reach
    for (int32_t k = n - 1; k >= 0; --k) {
      const size_t q = base + static_cast<size_t>(k) * step;
      if (set(q)) next = k;
      if (next - k <= d) out[q] = 1;
    }
  }
};

// The distinct radii of a histogram (BandRadius::levels), descending.
inline std::vector<int32_t> band_levels(const std::vector<uint32_t>& histogram) {
  std::vector<int32_t> out;
  for (size_t r = histogram.size(); r-- > 0;) {
    if (histogram[r] != 0) out.push_back(static_cast<int32_t>(r));
  }
  return out;
}

// On the domain dom: the union of the L-inf balls of radius[q] around every q of dom with radius[q] >= 0, whose
// distinct radii are among `levels` (descending), clipped to dom, into band (width x height bytes by pixel id, 1: in
// the band; 0 outside dom), through tmp (as large; written and read inside dom only); queued on the stream.
template <Space S>
void band_cascade(const int32_t* radius, const std::vector<int32_t>& levels, int32_t width, int32_t height,
                  BandDomain dom, uint8_t* band, uint8_t* tmp, Stream stream) {
  Par<S>::zero(band, static_cast<size_t>(width) * static_cast<size_t>(height), stream);
  if (dom.x1 <= dom.x0 || dom.y1 <= dom.y0) return;
  const auto nx = static_cast<size_t>(dom.x1 - dom.x0), ny = static_cast<size_t>(dom.y1 - dom.y0);
  for (size_t j = 0; j < levels.size(); ++j) {
    const int32_t d = levels[j] - (j + 1 < levels.size() ? levels[j + 1] : 0);
    Par<S>::for_each(ny, DilateLines{band, radius, levels[j], tmp, width, dom, true, d}, stream);
    Par<S>::for_each(nx, DilateLines{tmp, nullptr, 0, band, width, dom, false, d}, stream);
  }
}

// The whole image as the domain.
template <Space S>
void band_cascade(const int32_t* radius, const std::vector<int32_t>& levels, int32_t width, int32_t height,
                  uint8_t* band, uint8_t* tmp, Stream stream) {
  band_cascade<S>(radius, levels, width, height, BandDomain{0, 0, width, height}, band, tmp, stream);
}

}  // namespace balla::engine
