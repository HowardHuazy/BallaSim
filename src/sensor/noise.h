// Counter-based draws and the kernel side of the noise lattice (M3 §D.4-D.7, DESIGN §8.6).
//
// Noise states live at absolute times k 2^16 ns. The state at a lattice node is advanced from the
// previous node with the exact per-cell transition, and a tree node at level l is the Gaussian
// bridge of its parent's endpoints. Every draw is keyed by Philox(seed, purpose, camera, pixel,
// node time in int64 ns) through core/rng/purpose.h, so a node's value depends only on the seed and
// its ancestors, whatever other nodes are evaluated and in whatever order (M3 Theorem D.1): noise
// and events are invariant to the render schedule.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/pmath/explog.h"
#include "core/pmath/trig.h"
#include "core/rng/purpose.h"
#include "sensor/noise_tables.h"

namespace balla::sensor {

// Philox word -> uniform in the open interval (0, 1). fp64 uses all 32 bits (normals reach 6.66
// sigma); fp32 uses 23 bits so that the value is exact in a float (normals reach 5.77 sigma).
BALLA_HD inline double open_uniform(uint32_t u, double) { return (static_cast<double>(u) + 0.5) * 0x1p-32; }
BALLA_HD inline float open_uniform(uint32_t u, float) {
  return (static_cast<float>(u >> 9) + 0.5f) * 0x1p-23f;
}

// Three standard normals from one Philox call (Box-Muller with the deterministic log and pmath
// sincos, so the bits agree on host and device).
template <class T>
BALLA_HD void normals3(uint64_t seed, rng::Purpose purpose, uint32_t camera, uint32_t pixel,
                       uint64_t index, T* out) {
  const rng::U32x4 r = rng::draw(seed, purpose, camera, pixel, index);
  const T u0 = open_uniform(r.v[0], T(0)), u1 = open_uniform(r.v[1], T(0));
  const T u2 = open_uniform(r.v[2], T(0)), u3 = open_uniform(r.v[3], T(0));
  constexpr double kTwoPi = 0x1.921fb54442d18p+2;
  const T r0 = sqrt(T(-2) * pmath::log(u0)), r1 = sqrt(T(-2) * pmath::log(u2));
  T s0, c0, s1, c1;
  pmath::sincos(static_cast<T>(kTwoPi) * u1, &s0, &c0);
  pmath::sincos(static_cast<T>(kTwoPi) * u3, &s1, &c1);
  out[0] = r0 * c0;
  out[1] = r0 * s0;
  out[2] = r1 * c1;
}

// y = L x for a packed lower-triangular 3x3 (00, 10, 11, 20, 21, 22).
template <class T>
BALLA_HD void lower3_mul(const T* L, const T* x, T* y) {
  y[0] = L[0] * x[0];
  y[1] = L[1] * x[0] + L[2] * x[1];
  y[2] = L[3] * x[0] + L[4] * x[1] + L[5] * x[2];
}

// y = L^T x for a packed lower-triangular 3x3.
template <class T>
BALLA_HD void lower3_tmul(const T* L, const T* x, T* y) {
  y[0] = L[0] * x[0] + L[1] * x[1] + L[3] * x[2];
  y[1] = L[2] * x[1] + L[4] * x[2];
  y[2] = L[5] * x[2];
}

// ---------------------------------------------------------------------------------------------
// Cells with a diffusion ratio rho != 1 (accuracy.noise_diffusion transient; PHASE2 D-P2-2). Q = rho Q1 + Q2 from the
// split tables (noise_tables.h); the kernel factors it and forms the bridges itself. Symmetric 3x3 matrices are packed
// as their lower triangle (00, 10, 11, 20, 21, 22), like the factors. The factorizations' accuracy depends on the
// matrices' correlations, not on their scale (van der Sluis: Q33 ~ h^3 on short nodes is a diagonal scaling), and the
// correlations stay moderate (tests/sensor/test_noise.cpp checks the laws against the long-double construction).
// ---------------------------------------------------------------------------------------------

// Lower Cholesky factor of a packed symmetric PSD 3x3; a pivot at or below a tiny fraction of the trace is clamped to 0
// with the rest of its column, as the host construction does.
template <class T>
BALLA_HD void chol3(const T* q, T* c) {
  const T tiny = (abs(q[0]) + abs(q[2]) + abs(q[5])) * T(1e-20);
  T d = q[0];
  c[0] = d > tiny ? sqrt(d) : T(0);
  c[1] = c[0] > T(0) ? q[1] / c[0] : T(0);
  c[3] = c[0] > T(0) ? q[3] / c[0] : T(0);
  d = q[2] - c[1] * c[1];
  c[2] = d > tiny ? sqrt(d) : T(0);
  c[4] = c[2] > T(0) ? (q[4] - c[3] * c[1]) / c[2] : T(0);
  d = q[5] - c[3] * c[3] - c[4] * c[4];
  c[5] = d > tiny ? sqrt(d) : T(0);
}

// The lattice step's covariance factor of a cell with ratio rho: chol(rho Q1 + Q2), q12 = (Q1, Q2).
template <class T>
BALLA_HD void split_chol(const T* q12, T rho, T* c) {
  T q[6];
  for (int i = 0; i < 6; ++i) q[i] = rho * q12[i] + q12[6 + i];
  chol3(q, c);
}

// Bridge midpoint of a node in a cell with ratio rho (M3 D.5 with Q = rho Q1 + Q2), from the half-node's split block
// `blk` = (Phi_l, Q1_l, Q2_l), in information form, which subtracts no covariances: the midpoint's precision is
// J = Q^-1 + Phi^T Q^-1 Phi and its mean J^-1 (Q^-1 Phi x_L + Phi^T Q^-1 x_R). With C = chol Q, N = C^-1 and M = N Phi
// (lower triangular), J = N^T N + M^T M and b = N^T (M x_L) + M^T (N x_R); with U = chol J, x_m = U^-T (U^-1 b + xi),
// whose covariance is (U U^T)^-1 = J^-1. A different factor of the same law than the tables' chol(Sigma_m).
template <class T>
BALLA_HD void bridge_midpoint_split(const T* blk, T rho, const T* xl, const T* xr, const T* xi, T* xm) {
  const T* phi = blk;
  T c[6];
  split_chol(blk + 6, rho, c);
  T n[6];  // N = C^-1
  n[0] = T(1) / c[0];
  n[2] = T(1) / c[2];
  n[5] = T(1) / c[5];
  n[1] = -c[1] * n[0] * n[2];
  n[4] = -c[4] * n[2] * n[5];
  n[3] = -(c[3] * n[0] + c[4] * n[1]) * n[5];
  T m[6];  // M = N Phi
  m[0] = n[0] * phi[0];
  m[1] = n[1] * phi[0] + n[2] * phi[1];
  m[2] = n[2] * phi[2];
  m[3] = n[3] * phi[0] + n[4] * phi[1] + n[5] * phi[3];
  m[4] = n[4] * phi[2] + n[5] * phi[4];
  m[5] = n[5] * phi[5];
  T j[6];  // J = N^T N + M^T M
  j[0] = (n[0] * n[0] + n[1] * n[1] + n[3] * n[3]) + (m[0] * m[0] + m[1] * m[1] + m[3] * m[3]);
  j[1] = (n[2] * n[1] + n[4] * n[3]) + (m[2] * m[1] + m[4] * m[3]);
  j[2] = (n[2] * n[2] + n[4] * n[4]) + (m[2] * m[2] + m[4] * m[4]);
  j[3] = n[5] * n[3] + m[5] * m[3];
  j[4] = n[5] * n[4] + m[5] * m[4];
  j[5] = n[5] * n[5] + m[5] * m[5];
  T u[6];
  chol3(j, u);
  T a[3], b1[3], b2[3];
  lower3_mul(m, xl, a);
  lower3_tmul(n, a, b1);
  lower3_mul(n, xr, a);
  lower3_tmul(m, a, b2);
  T y[3];
  y[0] = (b1[0] + b2[0]) / u[0];
  y[1] = ((b1[1] + b2[1]) - u[1] * y[0]) / u[2];
  y[2] = ((b1[2] + b2[2]) - u[3] * y[0] - u[4] * y[1]) / u[5];
  for (int i = 0; i < 3; ++i) y[i] = y[i] + xi[i];
  xm[2] = y[2] / u[5];
  xm[1] = (y[1] - u[4] * xm[2]) / u[2];
  xm[0] = (y[0] - u[1] * xm[1] - u[3] * xm[2]) / u[0];
}

// Lattice step x_next = Phi x + chol(Q) xi (M3 D.4).
template <class T>
BALLA_HD void lattice_step(const T* phi, const T* chol, const T* x, const T* xi, T* x_next) {
  T a[3], b[3];
  lower3_mul(phi, x, a);
  lower3_mul(chol, xi, b);
  for (int i = 0; i < 3; ++i) x_next[i] = a[i] + b[i];
}

// Bridge midpoint x_m = A x_L + B x_R + L xi (M3 D.6); `bridge` holds A (9), B (9), L (6).
template <class T>
BALLA_HD void bridge_midpoint(const T* bridge, const T* xl, const T* xr, const T* xi, T* xm) {
  T c[3];
  lower3_mul(bridge + 18, xi, c);
  for (int i = 0; i < 3; ++i) {
    xm[i] = (bridge[3 * i] * xl[0] + bridge[3 * i + 1] * xl[1] + bridge[3 * i + 2] * xl[2]) +
            (bridge[9 + 3 * i] * xr[0] + bridge[9 + 3 * i + 1] * xr[1] + bridge[9 + 3 * i + 2] * xr[2]) +
            c[i];
  }
}

// Cubic Hermite of n3 on a node of length h (s) through the endpoint values and the exact slopes
// n3' = a3 (n2 - n3) (M3 D.7), in node-local seconds.
template <class T>
BALLA_HD void noise_hermite(const T* xl, const T* xr, T a3, T h, T* c) {
  const T dl = a3 * (xl[1] - xl[2]), dr = a3 * (xr[1] - xr[2]);
  const T delta = (xr[2] - xl[2]) / h;
  c[0] = xl[2];
  c[1] = dl;
  c[2] = (T(3) * delta - T(2) * dl - dr) / h;
  c[3] = (dl + dr - T(2) * delta) / (h * h);
}

// Exclusion margin of a level-l node (M3 D.8): z sigma_br + g |W (x_L, x_R)|.
template <class T>
BALLA_HD T noise_margin(const T* m, T z, const T* xl, const T* xr) {
  const T e[6] = {xl[0], xl[1], xl[2], xr[0], xr[1], xr[2]};
  const T* W = m + 2;
  T norm2 = T(0);
  int idx = 0;
  for (int i = 0; i < 6; ++i) {
    T s = T(0);
    for (int j = 0; j <= i; ++j) s = s + W[idx++] * e[j];
    norm2 = norm2 + s * s;
  }
  return z * m[0] + m[1] * sqrt(norm2);
}

}  // namespace balla::sensor
