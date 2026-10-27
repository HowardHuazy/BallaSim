// SO(3) exponential and logarithm on unit quaternions (M2 §0).
//
// exp(phi) = (cos(theta/2), sin(theta/2) phi/theta) with theta = |phi|.
// log(q) uses the representative with w >= 0 and theta = 2 atan2(|u|, w), so it is invariant
// under q -> -q, returns |phi| <= pi, and stays accurate near both theta = 0 and theta = pi
// (an acos(w) formulation loses half the digits near 0, an asin(|u|) one near pi).
#pragma once

#include "core/hd.h"
#include "core/math/quat.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"
#include "core/pmath/trig.h"

namespace balla::so3 {

namespace detail {

// Below these squared angles the Taylor series is used. Truncation is beyond the type's precision
// (the first dropped terms are theta^6/46080 and (|u|/w)^6/7).
template <class T>
struct SmallAngle;
template <>
struct SmallAngle<double> {
  static constexpr double kTheta2 = 1e-8;
  static constexpr double kRatio2 = 1e-8;
};
template <>
struct SmallAngle<float> {
  static constexpr float kTheta2 = 1e-4f;
  static constexpr float kRatio2 = 1e-4f;
};

}  // namespace detail

// Domain: |phi| <= 2 kMaxArg of pmath::sincos (2e6 rad in fp64, 12000 in fp32); longer vectors give NaN. The spline
// evaluates it on increments of at most pi.
template <class T>
BALLA_HD Quat<T> exp(Vec3<T> phi) {
  const T theta2 = dot(phi, phi);
  T c, s_over_theta;
  if (theta2 < detail::SmallAngle<T>::kTheta2) {
    // cos(theta/2) and sin(theta/2)/theta to O(theta^4).
    c = T(1) - theta2 * (T(1) / T(8)) + (theta2 * theta2) * (T(1) / T(384));
    s_over_theta = T(0.5) - theta2 * (T(1) / T(48)) + (theta2 * theta2) * (T(1) / T(3840));
  } else {
    const T theta = sqrt(theta2);
    T s;
    pmath::sincos(T(0.5) * theta, &s, &c);
    s_over_theta = s / theta;
  }
  return {c, s_over_theta * phi.x, s_over_theta * phi.y, s_over_theta * phi.z};
}

template <class T>
BALLA_HD Vec3<T> log(Quat<T> q) {
  if (q.w < T(0)) q = {-q.w, -q.x, -q.y, -q.z};
  const Vec3<T> u{q.x, q.y, q.z};
  const T n2 = dot(u, u);
  T f;  // phi = f u, f = theta / |u|
  if (n2 < detail::SmallAngle<T>::kRatio2 * (q.w * q.w)) {
    // 2 atan(n/w)/n = (2/w) (1 - (n/w)^2/3 + (n/w)^4/5) for n << w.
    const T r2 = n2 / (q.w * q.w);
    f = (T(2) / q.w) * (T(1) - r2 * (T(1) / T(3)) + (r2 * r2) * (T(1) / T(5)));
  } else {
    const T n = sqrt(n2);
    f = T(2) * pmath::atan2(n, q.w) / n;
  }
  return f * u;
}

}  // namespace balla::so3
