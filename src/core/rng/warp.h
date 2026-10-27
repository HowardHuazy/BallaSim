// Separable warp of unit-square points to the pixel response h = box(w) ⊛ Gaussian(sigma),
// truncated at |u| <= x_max = w/2 + n_sigma * sigma (DESIGN §6.1, §8.1).
//
// The 1-D inverse CDF is tabulated by WarpTable (warp_table.h) at nodes uniform in x, with the CDF
// and its derivative dx/dF at each node, and evaluated by cubic Hermite interpolation in F. Uniform
// x spacing keeps the tails resolved: the interpolation error is below 5e-8 px at 256 cells for
// sigma = 0.15 px (tests/cpp/test_warp.cpp). Only the left half x in [-x_max, 0] is stored; the
// right half follows from symmetry, which also makes the warp exactly antisymmetric.
// The cell containing a CDF value p is found from a guide table, uniform in the CDF value, by an
// upward scan: max{i < n : cdf[i] <= p}, the cell a binary search over the non-decreasing CDF
// returns (tests/cpp/test_warp.cpp compares the two bit for bit).
// The table is data, so host and device evaluate the same arithmetic on the same values.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/vec.h"

namespace balla::rng {

// Cells of the search guide: guide[g] = max{i < n : cdf[i] <= g / (2 G)} for g < G. A power of
// two, so the guide index of a CDF value p in [0, 1/2] is the exact floor of p 2G.
inline constexpr int32_t kWarpGuideCells = 1024;

template <class T>
struct WarpTableView {
  const T* cdf;           // n + 1 values of the truncated CDF at x_i = (i - n) dx; cdf[n] = 0.5
  const T* slope;         // n + 1 values of dx/dF = 1 / pdf at the same nodes
  const uint16_t* guide;  // kWarpGuideCells start cells of the cell search (n <= 65535)
  int32_t n;              // number of cells; x_max = n dx
  T dx;
};

// Inverse CDF: xi in [0, 1) -> offset in [-x_max, x_max] (pixels).
template <class T>
BALLA_HD T warp_inverse_cdf(const WarpTableView<T>& t, T xi) {
  const bool upper = xi >= T(0.5);
  const T p = upper ? T(1) - xi : xi;  // exact (Sterbenz) for xi in [0.5, 1)
  // guide[g] <= the cell, because cdf[guide[g]] <= g / (2G) <= p; scan up to it.
  int32_t g = static_cast<int32_t>(p * static_cast<T>(2 * kWarpGuideCells));
  if (g >= kWarpGuideCells) g = kWarpGuideCells - 1;
  int32_t lo = t.guide[g];
  // About n / (2 kWarpGuideCells) steps per sample on average: 0.1 at the default 256 cells, 32 at 65535.
  while (lo + 1 < t.n && t.cdf[lo + 1] <= p) ++lo;
  const T f0 = t.cdf[lo];
  const T df = t.cdf[lo + 1] - f0;
  const T s = (p - f0) / df;
  const T x0 = T(lo - t.n) * t.dx;
  const T x1 = T(lo + 1 - t.n) * t.dx;
  const T m0 = t.slope[lo] * df;
  const T m1 = t.slope[lo + 1] * df;
  const T one_minus = T(1) - s;
  const T h00 = (T(1) + T(2) * s) * (one_minus * one_minus);
  const T h10 = s * (one_minus * one_minus);
  const T h01 = (s * s) * (T(3) - T(2) * s);
  const T h11 = (s * s) * (s - T(1));
  const T x = ((h00 * x0 + h10 * m0) + h01 * x1) + h11 * m1;
  return upper ? -x : x;
}

template <class T>
BALLA_HD Vec2<T> warp(const WarpTableView<T>& t, Vec2<T> xi) {
  return {warp_inverse_cdf(t, xi.x), warp_inverse_cdf(t, xi.y)};
}

}  // namespace balla::rng
