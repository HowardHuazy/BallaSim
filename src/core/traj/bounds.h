// Rigorous per-segment motion bounds of the cumulative spline (M2 Lemma B1, Theorem B3).
//
// The derivative of the cumulative basis is a non-negative partition of unity, so on segment s
//   |omega|     <= max_j |d_{s+j}| / h                                     (B6)
//   |v|         <= max_j |dp_{s+j}| / h                                    (B7)
//   |a|         <= max_{j>=2} |dp_{s+j} - dp_{s+j-1}| / h^2                (B8)
//   |omega_dot| <= min(D (D + c_K), max_{j>=2} |d_{s+j} - d_{s+j-1}| + kappa_K D^2) / h^2   (B9)
// with D = max_j |d_{s+j}|. The bounds hold for the floating-point evaluation in spline.h, not
// only in exact arithmetic: (B6) and (B7) are attained for constant increments, so every result
// carries a relative slack of 16 ulp, and (B8)-(B9) bound sums whose weights lambda''_j cancel,
// so they also include the absolute rounding error of those sums (~ulp * c_K * max increment).
// Relative slack covers rounding only while nothing underflows. norm() squares its argument, so
// below |x| ~ sqrt(denorm_min) (1.5e-162 in fp64, 5e-23 in fp32) the squares lose their relative
// accuracy or vanish: norm(dp) of |dp| = 3.5e-163 is 0, while v = h^-1 sum lambda'_j dp_j is not.
// With gradual underflow each square errs by at most denorm_min / 2, which costs the computed norm
// at most sqrt(1.5 denorm_min) absolutely, so the norm of every non-zero increment carries
// NormFloor on top (detail::norm_upper). The evaluation's own underflow (products of subnormals, a
// few denorm_min) is far below it. Exactly zero increments keep a zero norm: a segment whose K
// controls are equal has bounds of exactly 0 (M2 §D.5), and its evaluation is exactly still.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"
#include "core/traj/spline.h"

namespace balla::traj {

template <class T>
struct MotionBounds {
  T omega;      // rad/s, body angular speed
  T omega_dot;  // rad/s^2
  T v;          // m/s
  T a;          // m/s^2
};

namespace detail {
// c_K = max_u sum_j |lambda''_j(u)|; kappa_K = (1 + max_u sum_j lambda'_j(u)^2) / 2 + 1, both
// attained at u = 0 and u = 1/2 respectively (M2 §B.4; sum lambda'^2 = 38/64 and 64454/147456).
template <int K, class T>
struct BoundConstants;
template <class T>
struct BoundConstants<4, T> {
  static constexpr T kC = T(2);
  static constexpr T kKappa = T(1.796875);
};
template <class T>
struct BoundConstants<6, T> {
  static constexpr T kC = T(4) / T(3);
  static constexpr T kKappa = T(1.5) + T(32227) / T(147456);
};

// A power of two >= sqrt(1.5 denorm_min): 2^-536 >= 1.22 * 2^-537 (fp64), 2^-74 >= 1.22 * 2^-74.5
// (fp32).
template <class T>
struct NormFloor;
template <>
struct NormFloor<double> {
  static constexpr double kValue = 0x1p-536;
};
template <>
struct NormFloor<float> {
  static constexpr float kValue = 0x1p-74f;
};

// norm(v) plus the underflow allowance for v != 0: >= |v| (1 - 3 ulp) for every finite v.
template <class T>
BALLA_HD T norm_upper(Vec3<T> v) {
  if (v.x == T(0) && v.y == T(0) && v.z == T(0)) return T(0);
  return norm(v) + NormFloor<T>::kValue;
}
}  // namespace detail

template <int K, class T>
BALLA_HD MotionBounds<T> segment_bounds(const SplineView<T>& spline, int32_t segment) {
  const SplineNode<T>* n = spline.nodes + segment;
  T d_max = T(0), dd_max = T(0), dp_max = T(0), ddp_max = T(0);
BALLA_UNROLL
  for (int j = 1; j < K; ++j) {
    d_max = max(d_max, detail::norm_upper(n[j].d));
    dp_max = max(dp_max, detail::norm_upper(n[j].dp));
    if (j >= 2) {
      dd_max = max(dd_max, detail::norm_upper(n[j].d - n[j - 1].d));
      ddp_max = max(ddp_max, detail::norm_upper(n[j].dp - n[j - 1].dp));
    }
  }
  using C = detail::BoundConstants<K, T>;
  const T eps16 = T(16) * FloatTraits<T>::kEpsilon;
  const T slack = T(1) + eps16;
  const T ih = spline.inv_h_s;
  const T ih2 = ih * ih;
  const T wd = min(d_max * (d_max + C::kC), dd_max + C::kKappa * (d_max * d_max)) +
               eps16 * (C::kC * d_max + d_max * d_max);
  const T acc = ddp_max + eps16 * (C::kC * dp_max);
  return {slack * (d_max * ih), slack * (wd * ih2), slack * (dp_max * ih), slack * (acc * ih2)};
}

}  // namespace balla::traj
