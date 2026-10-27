// balla::pmath trigonometric kernels: sin, cos, sincos, atan2, asin, acos (DESIGN §4.4, §12.1).
//
// libm and the CUDA math library implement these differently, so calling them would break the
// bitwise CPU/GPU parity of the shared maths. These kernels use only +, -, *, / and exact
// operations in a fixed order, which both compilers round identically under the FP policy.
//
// Accuracy against libm (tests/cpp/test_pmath.cpp): within 2 ulp in fp64 and 3 ulp in fp32 on
// the supported domain; asin and acos within 3 and 4 ulp (measured 2 and 3.1). Taylor
// coefficients suffice because the reduced arguments are small: |r| <= pi/4 for sin/cos and
// |r| <= tan(pi/16) for atan.
#pragma once

#include "core/hd.h"
#include "core/math/scalar.h"

namespace balla::pmath {

namespace detail {

// Cody-Waite reduction constants: pi/2 = P1 + P2 + P3, where P1 and P2 have few enough
// significant bits that k * P1 and k * P2 are exact for every quadrant k of |x| <= kMaxArg:
// bits(k) + bits(P) <= 53 in fp64 (20 + 32) and <= 24 in fp32 (12 + 12).
template <class T>
struct TrigConstants;

template <>
struct TrigConstants<double> {
  static constexpr double kTwoOverPi = 0x1.45f306dc9c883p-1;
  static constexpr double kP1 = 0x1.921fb544p+0;  // 31 significant bits
  static constexpr double kP2 = 0x1.0b4611a6p-34;  // 32 significant bits
  static constexpr double kP3 = 0x1.3198a2e037073p-69;
  static constexpr double kMaxArg = 1.0e6;  // |k| < 2^20
  static constexpr double kPi = 0x1.921fb54442d18p+1;
  static constexpr double kPiOver2 = 0x1.921fb54442d18p+0;
  static constexpr double kPiOver4 = 0x1.921fb54442d18p-1;
  static constexpr double kPiOver8 = 0x1.921fb54442d18p-2;
  static constexpr double kTanPiOver8 = 0x1.a827999fcef32p-2;     // sqrt(2) - 1
  static constexpr double kTanPiOver16 = 0x1.975f5e0553158p-3;    // 0.19891236737965800
  static constexpr double kTan3PiOver16 = 0x1.561b82ab7f990p-1;   // 0.66817863791929892
};

template <>
struct TrigConstants<float> {
  static constexpr float kTwoOverPi = 0x1.45f306p-1f;
  static constexpr float kP1 = 0x1.922p+0f;  // 12 significant bits
  static constexpr float kP2 = -0x1.2aep-18f;  // 12 significant bits
  static constexpr float kP3 = -0x1.de973ep-31f;
  static constexpr float kMaxArg = 6000.0f;  // |k| < 2^12
  static constexpr float kPi = 0x1.921fb6p+1f;
  static constexpr float kPiOver2 = 0x1.921fb6p+0f;
  static constexpr float kPiOver4 = 0x1.921fb6p-1f;
  static constexpr float kPiOver8 = 0x1.921fb6p-2f;
  static constexpr float kTanPiOver8 = 0x1.a8279ap-2f;
  static constexpr float kTanPiOver16 = 0x1.975f5ep-3f;
  static constexpr float kTan3PiOver16 = 0x1.561b82p-1f;
};

// sin(r) and cos(r) for |r| <= pi/4 (truncation error below 1e-19 / 2e-9 relative).
BALLA_HD inline void sincos_kernel(double r, double* s, double* c) {
  const double z = r * r;
  double ps = 2.8114572543455207632e-15;  // 1/17!
  ps = ps * z - 7.6471637318198164759e-13;
  ps = ps * z + 1.6059043836821614599e-10;
  ps = ps * z - 2.5052108385441718775e-08;
  ps = ps * z + 2.7557319223985890653e-06;
  ps = ps * z - 1.9841269841269841270e-04;
  ps = ps * z + 8.3333333333333333333e-03;
  ps = ps * z - 1.6666666666666666667e-01;
  *s = r + (r * z) * ps;
  double pc = -1.5619206968586225e-16;  // -1/18!
  pc = pc * z + 4.7794773323873852974e-14;
  pc = pc * z - 1.1470745597729724714e-11;
  pc = pc * z + 2.0876756987868098979e-09;
  pc = pc * z - 2.7557319223985890653e-07;
  pc = pc * z + 2.4801587301587301587e-05;
  pc = pc * z - 1.3888888888888888889e-03;
  pc = pc * z + 4.1666666666666666667e-02;
  *c = (1.0 - 0.5 * z) + (z * z) * pc;
}

BALLA_HD inline void sincos_kernel(float r, float* s, float* c) {
  const float z = r * r;
  float ps = 2.7557319e-06f;  // 1/9!
  ps = ps * z - 1.9841270e-04f;
  ps = ps * z + 8.3333333e-03f;
  ps = ps * z - 1.6666667e-01f;
  *s = r + (r * z) * ps;
  float pc = -2.7557319e-07f;  // -1/10!
  pc = pc * z + 2.4801587e-05f;
  pc = pc * z - 1.3888889e-03f;
  pc = pc * z + 4.1666668e-02f;
  *c = (1.0f - 0.5f * z) + (z * z) * pc;
}

// atan(r) for |r| <= tan(pi/16) (truncation error below 2e-17 / 1e-8 relative).
BALLA_HD inline double atan_kernel(double r) {
  const double z = r * r;
  double p = 1.0 / 21.0;
  p = p * z - 1.0 / 19.0;
  p = p * z + 1.0 / 17.0;
  p = p * z - 1.0 / 15.0;
  p = p * z + 1.0 / 13.0;
  p = p * z - 1.0 / 11.0;
  p = p * z + 1.0 / 9.0;
  p = p * z - 1.0 / 7.0;
  p = p * z + 1.0 / 5.0;
  p = p * z - 1.0 / 3.0;
  return r + (r * z) * p;
}

BALLA_HD inline float atan_kernel(float r) {
  const float z = r * r;
  float p = 1.0f / 9.0f;
  p = p * z - 1.0f / 7.0f;
  p = p * z + 1.0f / 5.0f;
  p = p * z - 1.0f / 3.0f;
  return r + (r * z) * p;
}

// atan(t) for 0 <= t <= 1, via atan(t) = atan(c) + atan((t - c) / (1 + t c)) with
// c in {0, tan(pi/8), 1}, which leaves |r| <= tan(pi/16).
template <class T>
BALLA_HD T atan_unit(T t) {
  using C = TrigConstants<T>;
  if (t <= C::kTanPiOver16) return atan_kernel(t);
  if (t <= C::kTan3PiOver16) {
    return C::kPiOver8 + atan_kernel((t - C::kTanPiOver8) / (T(1) + t * C::kTanPiOver8));
  }
  return C::kPiOver4 + atan_kernel((t - T(1)) / (T(1) + t));
}

}  // namespace detail

// Both outputs are NaN for |x| > kMaxArg (1e6 for double, 6000 for float), infinities and NaN.
template <class T>
BALLA_HD void sincos(T x, T* s, T* c) {
  using C = detail::TrigConstants<T>;
  if (!(abs(x) <= C::kMaxArg)) {
    *s = *c = quiet_nan<T>();
    return;
  }
  if (x == T(0)) {  // keeps the sign of -0, which the kernel's r + r z p form would drop
    *s = x;
    *c = T(1);
    return;
  }
  const T k = rint(x * C::kTwoOverPi);
  const T r = k == T(0) ? x : ((x - k * C::kP1) - k * C::kP2) - k * C::kP3;
  T sr, cr;
  detail::sincos_kernel(r, &sr, &cr);
  switch (static_cast<int>(k) & 3) {
    case 0: *s = sr;  *c = cr;  break;
    case 1: *s = cr;  *c = -sr; break;
    case 2: *s = -sr; *c = -cr; break;
    default: *s = -cr; *c = sr; break;
  }
}

template <class T>
BALLA_HD T sin(T x) {
  T s, c;
  sincos(x, &s, &c);
  return s;
}

template <class T>
BALLA_HD T cos(T x) {
  T s, c;
  sincos(x, &s, &c);
  return c;
}

// atan2(y, x) in [-pi, pi] for finite arguments, with the IEEE sign conventions for zeros.
template <class T>
BALLA_HD T atan2(T y, T x) {
  using C = detail::TrigConstants<T>;
  const T ax = abs(x), ay = abs(y);
  const bool x_negative = copysign(T(1), x) < T(0);
  T a;
  if (ax == T(0) && ay == T(0)) {
    a = x_negative ? C::kPi : T(0);
  } else if (ay <= ax) {
    a = detail::atan_unit(ay / ax);
    if (x_negative) a = C::kPi - a;
  } else {
    a = C::kPiOver2 - detail::atan_unit(ax / ay);
    if (x_negative) a = C::kPi - a;
  }
  return copysign(a, y);
}

// asin(x) and acos(x) on [-1, 1] through atan2 (p2f.md §4.7): asin x = atan2(x, c), acos x = atan2(c, x) with the
// cosine side c = sqrt((1 - x)(1 + x)). For |x| >= 1/2 the cancelling factor 1 - |x| is exact (Sterbenz) and the other
// has relative error <= u/2, so c keeps its relative accuracy as |x| -> 1, where 1 - x * x would lose it. NaN for
// |x| > 1 and NaN.
template <class T>
BALLA_HD T asin(T x) {
  if (!(abs(x) <= T(1))) return quiet_nan<T>();
  return atan2(x, sqrt((T(1) - x) * (T(1) + x)));
}

template <class T>
BALLA_HD T acos(T x) {
  if (!(abs(x) <= T(1))) return quiet_nan<T>();
  return atan2(sqrt((T(1) - x) * (T(1) + x)), x);
}

}  // namespace balla::pmath
