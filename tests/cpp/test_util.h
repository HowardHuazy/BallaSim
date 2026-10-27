// Helpers shared by the C++ unit tests.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>

#include "core/math/quat.h"
#include "core/math/vec.h"

namespace balla::test {

template <class T>
Vec3<T> random_unit(std::mt19937_64& rng) {
  std::normal_distribution<double> n;
  const Vec3<double> v{n(rng), n(rng), n(rng)};
  return cast<T>((1.0 / norm(v)) * v);
}

template <class T>
double distance(Vec3<T> a, Vec3<T> b) {
  return norm(cast<double>(a) - cast<double>(b));
}

// Rotation angle between two unit quaternions (sign-invariant).
template <class T>
double angle_between(Quat<T> a, Quat<T> b) {
  const Quat<double> d = cast<double>(conjugate(a)) * cast<double>(b);
  const double vn = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
  return 2.0 * std::atan2(vn, std::abs(d.w));
}

// Distance in units in the last place between two finite values of the same sign.
inline int64_t ulp_distance(double a, double b) {
  int64_t ia, ib;
  std::memcpy(&ia, &a, 8);
  std::memcpy(&ib, &b, 8);
  if (ia < 0) ia = INT64_MIN - ia;
  if (ib < 0) ib = INT64_MIN - ib;
  return ia > ib ? ia - ib : ib - ia;
}

inline int64_t ulp_distance(float a, float b) {
  int32_t ia, ib;
  std::memcpy(&ia, &a, 4);
  std::memcpy(&ib, &b, 4);
  if (ia < 0) ia = INT32_MIN - ia;
  if (ib < 0) ib = INT32_MIN - ib;
  return ia > ib ? int64_t{ia} - ib : int64_t{ib} - ia;
}

// erfcx(x) = e^{x^2} erfc(x) in long double (exact to ~1e-18 for 0 <= x <= 100), asymptotic beyond.
inline long double erfcx_ref(long double x) {
  if (x <= 100) return std::exp(x * x) * std::erfc(x);
  const long double inv2x2 = 1.0L / (2 * x * x);
  long double term = 1, sum = 1;
  for (int n = 1; n < 12; ++n) {
    term *= -(2 * n - 1) * inv2x2;
    sum += term;
  }
  return sum / (x * std::sqrt(3.14159265358979323846264338327950288L));
}

// Dawson's integral F(x) = int_0^x exp(-s (2x - s)) ds by composite 8-point Gauss-Legendre in long
// double; the integrand decreases monotonically from 1, so [0, min(x, 50/x)] suffices.
inline long double dawson_ref(long double x) {
  if (x == 0) return 0;
  const long double ax = std::fabs(x);
  const long double s_max = std::min(ax, 50.0L / ax);
  static const long double gx[4] = {0.18343464249564980493947614236018398L, 0.52553240991632898581773904918924635L,
                                    0.79666647741362673959155393647583044L, 0.96028985649753623168356086856947299L};
  static const long double gw[4] = {0.36268378337836198296515044927719561L, 0.31370664587788728733796220198660131L,
                                    0.22238103445337447054435599442624088L, 0.10122853629037625915253135430996219L};
  const int panels = 2000;
  const long double w = s_max / panels;
  long double sum = 0;
  for (int p = 0; p < panels; ++p) {
    const long double c = (p + 0.5L) * w;
    for (int i = 0; i < 4; ++i) {
      for (int sgn = -1; sgn <= 1; sgn += 2) {
        const long double s = c + sgn * 0.5L * w * gx[i];
        sum += gw[i] * std::exp(-s * (2 * ax - s));
      }
    }
  }
  return std::copysign(sum * 0.5L * w, x);
}

}  // namespace balla::test
