// Cumulative uniform B-spline basis lambda_j(u) and its first two derivatives (M2 §A.1, §B.1).
//
// lambda(u) = M~ [1, u, ..., u^(K-1)]^T with the exact integer matrices of M2 eq. (A4) over the
// common denominator (K-1)!. Integer coefficients are exact in fp32 and fp64; Horner's scheme
// fixes the evaluation order, and the final division is correctly rounded on host and device.
// (De Boor's recursion, whose terms are all non-negative, was measured as an alternative: it
// lowers the fp32 acceleration error by ~40 % but raises the velocity error by ~20 %.)
#pragma once

#include "core/hd.h"

namespace balla::traj {

template <class T, int K>
struct CumulativeBasis {
  T lambda[K - 1];    // lambda_j(u) for j = 1..K-1 at index j-1 (lambda_0 = 1 is implicit)
  T dlambda[K - 1];   // d lambda_j / du
  T ddlambda[K - 1];  // d^2 lambda_j / du^2
};

namespace detail {

// Rows j = 1..K-1 of (K-1)! * M~, coefficient of u^n.
template <int K>
BALLA_HD constexpr int cumulative_numerator(int j, int n);

template <>
BALLA_HD constexpr int cumulative_numerator<4>(int j, int n) {
  constexpr int c[3][4] = {{5, 3, -3, 1}, {1, 3, 3, -2}, {0, 0, 0, 1}};
  return c[j - 1][n];
}

template <>
BALLA_HD constexpr int cumulative_numerator<6>(int j, int n) {
  constexpr int c[5][6] = {{119, 5, -10, 10, -5, 1},
                           {93, 55, -30, -10, 15, -4},
                           {27, 55, 30, -10, -15, 6},
                           {1, 5, 10, 10, 5, -4},
                           {0, 0, 0, 0, 0, 1}};
  return c[j - 1][n];
}

BALLA_HD constexpr int factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }

}  // namespace detail

template <int K, class T>
BALLA_HD CumulativeBasis<T, K> cumulative_basis(T u) {
  static_assert(K == 4 || K == 6, "BallaSim supports cumulative splines of order 4 and 6");
  constexpr int kDen = detail::factorial(K - 1);
  CumulativeBasis<T, K> b;
BALLA_UNROLL
  for (int j = 1; j < K; ++j) {
    T p = T(detail::cumulative_numerator<K>(j, K - 1));
    T d = T((K - 1) * detail::cumulative_numerator<K>(j, K - 1));
    T dd = T((K - 1) * (K - 2) * detail::cumulative_numerator<K>(j, K - 1));
BALLA_UNROLL
    for (int n = K - 2; n >= 0; --n) {
      p = p * u + T(detail::cumulative_numerator<K>(j, n));
      if (n >= 1) d = d * u + T(n * detail::cumulative_numerator<K>(j, n));
      if (n >= 2) dd = dd * u + T(n * (n - 1) * detail::cumulative_numerator<K>(j, n));
    }
    b.lambda[j - 1] = p / T(kDen);
    b.dlambda[j - 1] = d / T(kDen);
    b.ddlambda[j - 1] = dd / T(kDen);
  }
  return b;
}

}  // namespace balla::traj
