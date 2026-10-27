// Uniform cubic B-spline tables over [0, 1]^2 for the BSDF's view- and roughness-dependent terms (p2f.md §4.3, F111).
//
// A table holds the coefficients of the C2 cubic spline that interpolates 64 x 64 node values at (i / 63, j / 63) of
// its axes (x, r): x the view coordinate (bsdf.h's t), r = sqrt(alpha).
// The coefficients come from Unser's (1999) tridiagonal (1/6, 4/6, 1/6) prefilter per axis, solved in fp64 by the
// generator (tools/bsdf_tables/gen_tables.cpp), with one ghost coefficient on each side of each axis that carries the
// boundary condition: even or odd at x = 0 (normal view), natural (S'' = 0) at x = 1, r = 0 and r = 1. Evaluation is
// 4 x 4 taps with fixed-order fma, so it is bitwise equal on host and device for one T. Coefficients are fp32
// (kTableSize floats per table); T = double evaluates the same spline in fp64 (the conformance gate, DESIGN §7.11).
//
// Unlike Cycles' bilinear lookup_table_read_2D (util/lookup_table.h; its ggx_E is 32 x 32 over (roughness, mu)),
// the spline is C2 across cells, so a view sweeping through the table does not put kinks into the radiance (F111).
#pragma once

#include <math.h>

#include "core/hd.h"
#include "core/math/scalar.h"

namespace balla::render {

inline constexpr int kTableNodes = 64;                         // nodes per axis, at i / (kTableNodes - 1)
inline constexpr int kTableStride = kTableNodes + 2;           // coefficients per axis: the nodes and two ghosts
inline constexpr int kTableSize = kTableStride * kTableStride;  // floats per table, stored [r][x]

// The 4 taps of one axis: coefficient offset of the first tap (ghost-shifted, so tap k reads offset base + k) and
// the cubic B-spline weights at the cell fraction f.
template <class T>
struct BsplineTap {
  int base;
  T w[4];
};

// Taps at cell i (0 <= i <= kTableNodes - 2) and fraction f in [0, 1]; f = 1 in cell i and f = 0 in cell i + 1 give
// the same knot. w1(f) = w2(1 - f), so the weights are mirror-symmetric in their rounding too.
template <class T>
BALLA_HD BsplineTap<T> bspline_tap_at(int i, T f) {
  const T g = T(1) - f;
  const T f2 = f * f, g2 = g * g;
  BsplineTap<T> t;
  t.base = i;
  t.w[0] = g2 * g / T(6);
  t.w[1] = (T(4) - T(6) * f2 + T(3) * (f2 * f)) / T(6);
  t.w[2] = (T(4) - T(6) * g2 + T(3) * (g2 * g)) / T(6);
  t.w[3] = f2 * f / T(6);
  return t;
}

// Taps of coordinate x, clamped to [0, 1] (NaN maps to 0).
template <class T>
BALLA_HD BsplineTap<T> bspline_tap(T x) {
  const T xc = x > T(0) ? min(x, T(1)) : T(0);
  const T t = xc * T(kTableNodes - 1);
  const T i = min(floor(t), T(kTableNodes - 2));
  return bspline_tap_at(static_cast<int>(i), t - i);
}

namespace detail {

BALLA_HD inline float table_fma(float a, float b, float c) { return ::fmaf(a, b, c); }
BALLA_HD inline double table_fma(double a, double b, double c) { return ::fma(a, b, c); }

BALLA_HD inline float table_load(const float* p) {
#if defined(__CUDA_ARCH__)
  return __ldg(p);
#else
  return *p;
#endif
}

}  // namespace detail

// The spline of coefficient table c (kTableSize floats, [r][x]) at the taps of x and r.
template <class T>
BALLA_HD T bspline_eval(const float* c, const BsplineTap<T>& tx, const BsplineTap<T>& tr) {
  T row[4];
  BALLA_UNROLL
  for (int b = 0; b < 4; ++b) {
    const float* p = c + (tr.base + b) * kTableStride + tx.base;
    T s = tx.w[0] * T(detail::table_load(p));
    s = detail::table_fma(tx.w[1], T(detail::table_load(p + 1)), s);
    s = detail::table_fma(tx.w[2], T(detail::table_load(p + 2)), s);
    row[b] = detail::table_fma(tx.w[3], T(detail::table_load(p + 3)), s);
  }
  T v = tr.w[0] * row[0];
  v = detail::table_fma(tr.w[1], row[1], v);
  v = detail::table_fma(tr.w[2], row[2], v);
  return detail::table_fma(tr.w[3], row[3], v);
}

}  // namespace balla::render
