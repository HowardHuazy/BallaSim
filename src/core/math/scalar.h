// Exact or correctly rounded scalar operations with identical results on host and device.
//
// Every function here is either exact (abs, floor, rint, copysign, min, max, clamp, the bit casts)
// or correctly rounded by IEEE 754 (sqrt; nvcc keeps -prec-sqrt=true under the floating-point
// policy), so it never breaks bitwise CPU/GPU parity. Transcendentals live in core/pmath.
#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "core/hd.h"

namespace balla {

template <class T>
struct FloatTraits;

template <>
struct FloatTraits<float> {
  static constexpr float kEpsilon = 0x1p-23f;  // ulp(1)
};

template <>
struct FloatTraits<double> {
  static constexpr double kEpsilon = 0x1p-52;
};

BALLA_HD inline float abs(float x) { return fabsf(x); }
BALLA_HD inline double abs(double x) { return fabs(x); }
BALLA_HD inline float sqrt(float x) { return sqrtf(x); }
BALLA_HD inline double sqrt(double x) { return ::sqrt(x); }
BALLA_HD inline float floor(float x) { return floorf(x); }
BALLA_HD inline double floor(double x) { return ::floor(x); }
BALLA_HD inline float rint(float x) { return rintf(x); }  // round half to even
BALLA_HD inline double rint(double x) { return ::rint(x); }
BALLA_HD inline float copysign(float x, float s) { return copysignf(x, s); }
BALLA_HD inline double copysign(double x, double s) { return ::copysign(x, s); }

template <class T>
BALLA_HD constexpr T min(T a, T b) {
  return b < a ? b : a;
}
template <class T>
BALLA_HD constexpr T max(T a, T b) {
  return a < b ? b : a;
}
template <class T>
BALLA_HD constexpr T clamp(T x, T lo, T hi) {
  return x < lo ? lo : (hi < x ? hi : x);
}

template <class T>
BALLA_HD T quiet_nan() {
  return T(NAN);
}

template <class T>
BALLA_HD T infinity() {
  return T(INFINITY);
}

// The IEEE 754 bit pattern of a float and back.
BALLA_HD inline uint32_t float_to_bits(float f) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(f);
#else
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return u;
#endif
}

BALLA_HD inline float bits_to_float(uint32_t u) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(u);
#else
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
#endif
}

}  // namespace balla
