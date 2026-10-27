// Host-side computation of the injective domains of the camera models (DESIGN §8.1, M2 §D.3).
//
// The domain must contain the preimage of every position that ray generation unprojects: pixel
// centres +- support_px, the reach of the sub-sample warp (x_max = 0.95 px by default, DESIGN
// §6.1). r_img (theta_img) is therefore the largest undistorted radius (angle) on the boundary of
// that support rectangle, [-s, W - 1 + s] x [-s, H - 1 + s] for s = support_px; at s = 0.5 it is
// the image boundary of M2's tables.
//
// Radtan: r_mono is the smallest |x_n| at which lambda_min(J_d) reaches 0 (J_d is symmetric, so
// J_d > 0 on the disc |x_n| < r_mono makes the map injective there, M2 Theorem D1), and
// r_lim = min(0.99 r_mono, 1.2 r_img). Kannala-Brandt: theta_mono is the first zero of
// theta_d'(theta) (capped at pi), and theta_lim = min(0.99 theta_mono, 1.2 theta_img).
// M2 admits any r_lim in [r_img, r_mono), so a camera whose support reaches beyond 0.99 r_mono
// (0.99 theta_mono) is rejected with std::domain_error rather than given r_lim < r_img.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "core/camera/kannala_brandt.h"
#include "core/camera/radtan.h"

namespace balla::camera {

struct RadtanDomain {
  double r_mono;  // +inf if J_d stays positive definite up to the search radius
  double r_img;
  double r_lim;
};

struct KannalaBrandtDomain {
  double theta_mono;
  double theta_img;
  double theta_lim;
};

namespace detail {

inline constexpr double kMonotoneMargin = 0.99;

inline double lambda_min(const Sym2<double>& j) {
  const double mean = 0.5 * (j.xx + j.yy);
  const double half_diff = 0.5 * (j.xx - j.yy);
  return mean - std::sqrt(half_diff * half_diff + j.xy * j.xy);
}

inline void check_support(int width, int height, double support_px) {
  if (width <= 0 || height <= 0 || !(support_px >= 0.0) || !std::isfinite(support_px)) {
    throw std::invalid_argument("injective domain: invalid image size or sub-sample support");
  }
}

// Calls fn(u, v) at points spaced <= `step` px along the boundary of the support rectangle
// [-s, width - 1 + s] x [-s, height - 1 + s], corners included.
template <class Fn>
void for_each_support_boundary_point(int width, int height, double support_px, double step,
                                     Fn&& fn) {
  const double u0 = -support_px, u1 = width - 1 + support_px;
  const double v0 = -support_px, v1 = height - 1 + support_px;
  const int nu = std::max(1, static_cast<int>(std::ceil((u1 - u0) / step)));
  const int nv = std::max(1, static_cast<int>(std::ceil((v1 - v0) / step)));
  for (int i = 0; i <= nu; ++i) {
    const double u = u0 + (u1 - u0) * i / nu;
    fn(u, v0);
    fn(u, v1);
  }
  for (int i = 0; i <= nv; ++i) {
    const double v = v0 + (v1 - v0) * i / nv;
    fn(u0, v);
    fn(u1, v);
  }
}

}  // namespace detail

// The r_lim and z_min fields of `cam` are ignored.
inline RadtanDomain radtan_injective_domain(RadtanCamera<double> cam, int width, int height,
                                            double support_px) {
  detail::check_support(width, height, support_px);
  constexpr int kAngles = 1440;
  constexpr double kSearchRadius = 10.0;  // 84 degrees off-axis; radtan is meaningless beyond
  constexpr double kRadialStep = 2e-3;
  double r_mono = std::numeric_limits<double>::infinity();
  for (int a = 0; a < kAngles; ++a) {
    const double phi = 2.0 * std::numbers::pi * a / kAngles;
    const Vec2<double> dir{std::cos(phi), std::sin(phi)};
    auto positive = [&](double r) { return detail::lambda_min(distort_jacobian(cam, r * dir)) > 0.0; };
    double lo = 0.0;
    for (int k = 1; k * kRadialStep <= kSearchRadius; ++k) {
      const double r = k * kRadialStep;
      if (!positive(r)) {
        double hi = r;
        for (int it = 0; it < 60; ++it) {
          const double mid = 0.5 * (lo + hi);
          (positive(mid) ? lo : hi) = mid;
        }
        r_mono = std::min(r_mono, lo);
        break;
      }
      lo = r;
    }
  }

  // Unprojection restricted to the largest admissible r_lim fails exactly where the support
  // leaves it.
  cam.z_min = 0.0;
  cam.r_lim = std::isfinite(r_mono) ? detail::kMonotoneMargin * r_mono : kSearchRadius;
  double r_img = 0.0;
  detail::for_each_support_boundary_point(width, height, support_px, 0.25, [&](double u, double v) {
    Vec3<double> ray;
    if (!unproject(cam, Vec2<double>{u, v}, &ray)) {
      throw std::domain_error("radtan camera is not injective over the image: its sub-sample "
                              "support reaches beyond 0.99 r_mono");
    }
    r_img = std::max(r_img, std::hypot(ray.x, ray.y));
  });
  return {r_mono, r_img, std::min(detail::kMonotoneMargin * r_mono, 1.2 * r_img)};
}

// The theta_lim field of `cam` is ignored.
inline KannalaBrandtDomain kannala_brandt_injective_domain(KannalaBrandtCamera<double> cam,
                                                           int width, int height,
                                                           double support_px) {
  detail::check_support(width, height, support_px);
  constexpr double kStep = 1e-4;
  double theta_mono = std::numbers::pi;
  double lo = 0.0;
  for (int k = 1; k * kStep <= std::numbers::pi; ++k) {
    const double t = k * kStep;
    if (!(theta_distorted_derivative(cam, t) > 0.0)) {
      double hi = t;
      for (int it = 0; it < 60; ++it) {
        const double mid = 0.5 * (lo + hi);
        (theta_distorted_derivative(cam, mid) > 0.0 ? lo : hi) = mid;
      }
      theta_mono = lo;
      break;
    }
    lo = t;
  }

  cam.theta_lim = detail::kMonotoneMargin * theta_mono;
  double theta_img = 0.0;
  detail::for_each_support_boundary_point(width, height, support_px, 0.25, [&](double u, double v) {
    Vec3<double> ray;
    if (!unproject(cam, Vec2<double>{u, v}, &ray)) {
      throw std::domain_error("Kannala-Brandt camera is not injective over the image: its "
                              "sub-sample support reaches beyond 0.99 theta_mono");
    }
    theta_img = std::max(theta_img, std::atan2(std::hypot(ray.x, ray.y), ray.z));
  });
  return {theta_mono, theta_img, std::min(detail::kMonotoneMargin * theta_mono, 1.2 * theta_img)};
}

}  // namespace balla::camera
