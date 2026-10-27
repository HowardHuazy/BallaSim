// Relative illumination of a lens: a radial factor on the natural-vignetting gain map (P2-F F2b, PHASE2 F079; SENSOR.md
// "Photometry"; optics.relative_illumination).
//
//   G(r~) = 1 + c2 r~^2 + c4 r~^4,   r~ = |x - c| / r_corner
//
// x a pixel centre (integer coordinates, CONTRACTS.md), c the principal point (cx, cy) and r_corner the largest
// distance from c to the four corner pixel centres, so r~ <= 1 on the image and G(0) = 1 (on the axis the gain stays
// the law's, the relative photometry's scale its on-axis value). The radius is image height in pixels (square pixels:
// fx and fy differ by calibration noise only). G multiplies optics.vignetting_law's g; it models what the law leaves
// out, the pupil that shrinks or moves off axis (mechanical vignetting, pupil aberration), as an even polynomial in
// image height, the form of lensfun's "pa" and Adobe's lens-profile vignetting models (to r^6 there).
//
// Conventions: Blender Cycles and Eevee have no lens vignetting (compositor only); Mitsuba 3's perspective and thinlens
// sensors weigh every ray 1; pbrt-v4's RealisticCamera traces the lens system, its exit pupil bounded per film radius
// (BoundExitPupil over radial segments of the film), so its relative illumination is a radial function of image height
// too, from a lens prescription instead of fitted coefficients.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace balla::camera {

// r_corner^2: the largest squared distance from (cx, cy) to the corner pixel centres of a width x height image.
inline double corner_radius_sq(int32_t width, int32_t height, double cx, double cy) {
  double r2 = 0.0;
  for (const double x : {0.0, static_cast<double>(width - 1)}) {
    for (const double y : {0.0, static_cast<double>(height - 1)}) {
      r2 = std::fmax(r2, (x - cx) * (x - cx) + (y - cy) * (y - cy));
    }
  }
  return r2;
}

// G at u = r~^2.
inline double relative_illumination(double c2, double c4, double u) { return 1.0 + u * (c2 + u * c4); }

// The smallest G over the image (u in [0, 1]): the ends, and the vertex u = -c2 / (2 c4) of a convex G inside.
inline double relative_illumination_min(double c2, double c4) {
  double g = std::fmin(1.0, relative_illumination(c2, c4, 1.0));
  if (c4 > 0.0 && c2 < 0.0 && -c2 < 2.0 * c4) g = std::fmin(g, relative_illumination(c2, c4, -c2 / (2.0 * c4)));
  return g;
}

// Why the coefficients are refused, or nullptr: [] (G = 1) or [c2, c4], finite, with G > 0 over the image.
inline const char* relative_illumination_error(const std::vector<double>& c) {
  if (c.empty()) return nullptr;
  if (c.size() != 2) return "expected [] or [c2, c4]";
  if (!std::isfinite(c[0]) || !std::isfinite(c[1])) return "the coefficients must be finite";
  if (!(relative_illumination_min(c[0], c[1]) > 0.0)) return "G = 1 + c2 r^2 + c4 r^4 must stay > 0 for r in [0, 1]";
  return nullptr;
}

}  // namespace balla::camera
