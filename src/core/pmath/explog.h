// balla::pmath exponentials and logarithms: exp, expm1, phi1, log, log1p and log2 in fp32 and
// fp64, exp2 in fp32, and the exact power-of-two helpers ldexp_exact and frexp_exact (DESIGN §4.6,
// §12.1).
//
// libm and the CUDA math library round these differently, so calling them would break the bitwise
// CPU/GPU parity of the shared maths. Like trig.h, every function here uses only +, -, *, /, sqrt
// and exact operations (rint, exponent-field arithmetic) in a fixed order, which both compilers
// evaluate identically under the floating-point policy (no contraction, no fast math).
//
// Underflow policy (DESIGN §8.3 "Precision"): exp(x) returns exactly 0 for x < kMinArg (-87.33 in
// fp32, -708.39 in fp64), just above ln of the smallest normal (-87.336, -708.396), so no subnormal is
// ever produced and x86 never takes a subnormal assist; the normal results of that thin band are
// flushed too.
//
// Accuracy (tests/cpp/test_special.cpp): exp, log and log1p within 2 ulp (fp64) and 3 ulp (fp32) of
// correctly rounded results; expm1 within 4 ulp in both (exp(x) - 1 just above |x| = 0.35, where
// it leaves its kernel); exp2 within 2 ulp on [-126, 127] and exact at integers;
// log2 within 2 ulp of the result plus 2 ulp of 1/2 (the rounding of ln(m) log2(e) for the reduced
// mantissa m), and exact at powers of two.
#pragma once

#include <stdint.h>
#include <string.h>

#include "core/hd.h"
#include "core/math/scalar.h"

namespace balla::pmath {

template <class T>
struct ExpConstants;

template <>
struct ExpConstants<double> {
  static constexpr double kLog2e = 0x1.71547652b82fep+0;
  static constexpr double kLn2Hi = 0x1.62e42fee00000p-1;  // 32 significant bits: k * kLn2Hi is exact
  static constexpr double kLn2Lo = 0x1.a39ef35793c76p-33;
  static constexpr double kMinArg = -708.39;  // exp(kMinArg) > DBL_MIN, see the underflow policy
  static constexpr double kMaxArg = 709.78;
  static constexpr double kSqrtHalf = 0x1.6a09e667f3bcdp-1;
};

template <>
struct ExpConstants<float> {
  static constexpr float kLog2e = 0x1.715476p+0f;
  static constexpr float kLn2Hi = 0x1.62e400p-1f;  // 16 significant bits: k * kLn2Hi is exact
  static constexpr float kLn2Lo = 0x1.7f7d1cp-20f;
  static constexpr float kMinArg = -87.33f;  // exp(kMinArg) > FLT_MIN
  static constexpr float kMaxArg = 88.72f;
  static constexpr float kSqrtHalf = 0x1.6a09e6p-1f;
};

// Exact power-of-two scaling and splitting by exponent-field arithmetic: identical to ldexp and
// frexp for the arguments used here, but inline (the libm calls cost more than the polynomials).
// x * 2^e for a normal result and |e| <= 2 * max exponent.
BALLA_HD inline double ldexp_exact(double x, int e) {
  const int e1 = e > 1023 ? 1023 : (e < -1022 ? -1022 : e);
  const int e2 = e - e1;  // at most one more factor, only near the extremes of the range
  uint64_t b1 = static_cast<uint64_t>(e1 + 1023) << 52, b2 = static_cast<uint64_t>(e2 + 1023) << 52;
  double p1, p2;
  memcpy(&p1, &b1, 8);
  memcpy(&p2, &b2, 8);
  return e2 == 0 ? x * p1 : (x * p1) * p2;
}
BALLA_HD inline float ldexp_exact(float x, int e) {
  const int e1 = e > 127 ? 127 : (e < -126 ? -126 : e);
  const int e2 = e - e1;
  uint32_t b1 = static_cast<uint32_t>(e1 + 127) << 23, b2 = static_cast<uint32_t>(e2 + 127) << 23;
  float p1, p2;
  memcpy(&p1, &b1, 4);
  memcpy(&p2, &b2, 4);
  return e2 == 0 ? x * p1 : (x * p1) * p2;
}
// Mantissa in [0.5, 1) and exponent of a finite x > 0 (subnormals included).
BALLA_HD inline double frexp_exact(double x, int* e) {
  int bias = 0;
  if (x < 0x1p-1022) {
    x = x * 0x1p54;
    bias = 54;
  }
  uint64_t b;
  memcpy(&b, &x, 8);
  *e = static_cast<int>((b >> 52) & 0x7ff) - 1022 - bias;
  b = (b & 0x800fffffffffffffull) | (uint64_t{1022} << 52);
  double m;
  memcpy(&m, &b, 8);
  return m;
}
BALLA_HD inline float frexp_exact(float x, int* e) {
  int bias = 0;
  if (x < 0x1p-126f) {
    x = x * 0x1p25f;
    bias = 25;
  }
  uint32_t b;
  memcpy(&b, &x, 4);
  *e = static_cast<int>((b >> 23) & 0xff) - 126 - bias;
  b = (b & 0x807fffffu) | (126u << 23);
  float m;
  memcpy(&m, &b, 4);
  return m;
}

namespace detail {

// e^r for |r| <= ln2 / 2 by its Taylor polynomial: degree 13 leaves 4e-18 relative in fp64 and
// degree 7 leaves 5e-9 in fp32.
BALLA_HD inline double exp_kernel(double r) {
  double p = 1.0 / 6227020800.0;  // 1/13!
  p = p * r + 1.0 / 479001600.0;
  p = p * r + 1.0 / 39916800.0;
  p = p * r + 1.0 / 3628800.0;
  p = p * r + 1.0 / 362880.0;
  p = p * r + 1.0 / 40320.0;
  p = p * r + 1.0 / 5040.0;
  p = p * r + 1.0 / 720.0;
  p = p * r + 1.0 / 120.0;
  p = p * r + 1.0 / 24.0;
  p = p * r + 1.0 / 6.0;
  p = p * r + 0.5;
  p = p * r + 1.0;
  return p * r + 1.0;
}

BALLA_HD inline float exp_kernel(float r) {
  float p = 1.0f / 5040.0f;
  p = p * r + 1.0f / 720.0f;
  p = p * r + 1.0f / 120.0f;
  p = p * r + 1.0f / 24.0f;
  p = p * r + 1.0f / 6.0f;
  p = p * r + 0.5f;
  p = p * r + 1.0f;
  return p * r + 1.0f;
}

// R = 2 s^2/3 + 2 s^4/5 + ... of log(1 + f) = f - s (f - R), s = f / (2 + f), |s| <= 0.1716.
// Nine terms leave 8e-18 in fp64, four leave 7e-10 in fp32.
BALLA_HD inline double log_series(double z) {
  double p = 2.0 / 19.0;
  p = p * z + 2.0 / 17.0;
  p = p * z + 2.0 / 15.0;
  p = p * z + 2.0 / 13.0;
  p = p * z + 2.0 / 11.0;
  p = p * z + 2.0 / 9.0;
  p = p * z + 2.0 / 7.0;
  p = p * z + 2.0 / 5.0;
  p = p * z + 2.0 / 3.0;
  return p * z;
}

BALLA_HD inline float log_series(float z) {
  float p = 2.0f / 9.0f;
  p = p * z + 2.0f / 7.0f;
  p = p * z + 2.0f / 5.0f;
  p = p * z + 2.0f / 3.0f;
  return p * z;
}

// Argument reduction of the logarithms for finite x > 0: x = m 2^e with m in [sqrt(1/2), sqrt(2)),
// both exact; returns ln(m).
template <class T>
BALLA_HD T log_reduced(T x, int* e) {
  T m = frexp_exact(x, e);  // m in [0.5, 1), exact
  if (m < ExpConstants<T>::kSqrtHalf) {
    m = m + m;
    *e -= 1;
  }
  const T f = m - T(1);  // exact (Sterbenz)
  const T s = f / (T(2) + f);
  return f - s * (f - log_series(s * s));
}

// e^x - 1 for |x| < 0.35 by its Taylor polynomial (degree 14 in fp64, 8 in fp32).
BALLA_HD inline double expm1_kernel(double x) {
  double p = 1.0 / 87178291200.0;  // 1/14!
  p = p * x + 1.0 / 6227020800.0;
  p = p * x + 1.0 / 479001600.0;
  p = p * x + 1.0 / 39916800.0;
  p = p * x + 1.0 / 3628800.0;
  p = p * x + 1.0 / 362880.0;
  p = p * x + 1.0 / 40320.0;
  p = p * x + 1.0 / 5040.0;
  p = p * x + 1.0 / 720.0;
  p = p * x + 1.0 / 120.0;
  p = p * x + 1.0 / 24.0;
  p = p * x + 1.0 / 6.0;
  p = p * x + 0.5;
  return (p * x) * x + x;
}

BALLA_HD inline float expm1_kernel(float x) {
  float p = 1.0f / 40320.0f;
  p = p * x + 1.0f / 5040.0f;
  p = p * x + 1.0f / 720.0f;
  p = p * x + 1.0f / 120.0f;
  p = p * x + 1.0f / 24.0f;
  p = p * x + 1.0f / 6.0f;
  p = p * x + 0.5f;
  return (p * x) * x + x;
}

}  // namespace detail

// e^x with the underflow policy above; NaN propagates, +inf above kMaxArg.
template <class T>
BALLA_HD T exp(T x) {
  using C = ExpConstants<T>;
  if (!(x >= C::kMinArg)) return x != x ? x : T(0);
  if (x > C::kMaxArg) return infinity<T>();
  const T k = rint(x * C::kLog2e);
  const T r = (x - k * C::kLn2Hi) - k * C::kLn2Lo;
  return ldexp_exact(detail::exp_kernel(r), static_cast<int>(k));
}

// Natural logarithm: NaN for x < 0 or NaN, -inf for 0, +inf for +inf.
template <class T>
BALLA_HD T log(T x) {
  using C = ExpConstants<T>;
  if (!(x > T(0))) return x == T(0) ? -infinity<T>() : quiet_nan<T>();
  if (x == infinity<T>()) return x;
  int e = 0;
  const T logm = detail::log_reduced(x, &e);
  const T ef = static_cast<T>(e);
  return ef * C::kLn2Hi + (logm + ef * C::kLn2Lo);
}

// Base-2 logarithm with the special values of log; exact at powers of two.
template <class T>
BALLA_HD T log2(T x) {
  if (!(x > T(0))) return x == T(0) ? -infinity<T>() : quiet_nan<T>();
  if (x == infinity<T>()) return x;
  int e = 0;
  const T logm = detail::log_reduced(x, &e);
  return static_cast<T>(e) + logm * ExpConstants<T>::kLog2e;
}

// 2^x in fp32, the cheapest exponential here: no Cody-Waite reduction (x - rint(x) is exact) and a
// single exact scaling, so per-texel weights (the renderer's EWA filter) use it rather than exp.
// Returns exactly 0 below -126, so results never become subnormal (the underflow policy above), and
// saturates at 2^127 above 127; NaN maps to 0 (callers never pass it).
BALLA_HD inline float exp2(float x) {
  if (!(x >= -126.0f)) return 0.0f;
  if (x > 127.0f) x = 127.0f;
  const float n = rint(x);  // exact; x - n is exact by Sterbenz for |x| <= 127
  const float f = x - n;    // [-0.5, 0.5]
  // 2^f = e^(f ln 2): Taylor to degree 7 in f ln 2 (|f ln 2| <= 0.347, truncation < 6e-9).
  const float y = f * 0.693147180559945309f;
  float p = 1.98412698412698413e-4f;  // 1/7!
  p = p * y + 1.38888888888888889e-3f;
  p = p * y + 8.33333333333333333e-3f;
  p = p * y + 4.16666666666666667e-2f;
  p = p * y + 1.66666666666666667e-1f;
  p = p * y + 0.5f;
  p = p * y + 1.0f;
  p = p * y + 1.0f;
  // 2^n for n in [-126, 127] as a normal float: exact scaling.
  return p * bits_to_float(static_cast<uint32_t>(static_cast<int32_t>(n) + 127) << 23);
}

template <class T>
BALLA_HD T expm1(T x) {
  if (abs(x) < T(0.35)) return detail::expm1_kernel(x);
  return exp(x) - T(1);
}

// phi_1(x) = (e^x - 1) / x, with phi_1(0) = 1; bounded and positive for every x <= 0.
template <class T>
BALLA_HD T phi1(T x) {
  if (x == T(0)) return T(1);
  return expm1(x) / x;
}

// log(1 + x) for x > -1 (Goldberg's correction keeps it accurate for small x).
template <class T>
BALLA_HD T log1p(T x) {
  const T u = T(1) + x;
  if (u == T(1)) return x;
  return log(u) * (x / (u - T(1)));
}

}  // namespace balla::pmath
