// Tabulated transition, bridge and margin matrices of the in-path noise (M3 §D.2-D.5, DESIGN §8.6).
//
// The linearized noise n = (n1, n2, n3) obeys dn = A n dt + G dW with
//   A = [[-a1, 0, 0], [a2, -a2, 0], [0, a3, -a3]],  G G^T = diag(2 k q a1, 2 sigma_o^2 a2, 0)
// (M3 A.10 with the per-cell frozen rate a1 of M3 D.1). Only a1 varies, so every matrix the kernel
// needs is tabulated against ln a1 on [1e-2, 1e9] s^-1 at 64 nodes per decade and interpolated
// linearly (M3 V7b: interpolation error <= 1.6e-4 relative). Per node and tree level l
// (cells of h_l = 2^(16-l) ns):
//   - level 0: the lattice step Phi(h_0) and chol Q(h_0) (M3 D.2-D.4);
//   - levels 1..L: the midpoint bridge x_m = A_m x_L + B_m x_R + L_m xi (M3 D.5-D.6);
//   - levels 0..L-1: the exclusion margin m_l(e) = z sigma_br,l + g_l |W_l e| (M3 D.8), with
//     W_l = chol(C_ee)^-1 so that |W_l e| is the Mahalanobis norm of the endpoint pair;
//   - the stationary covariance factor chol P_inf for the initial state (M3 A.20);
//   - with NoiseModel::split (accuracy.noise_diffusion transient, PHASE2 D-P2-2), after those: Q_l's two parts at every
//     level, Q1 driven by the input node (D1 = 2 k q a1) and Q2 by the output node, and Phi_l for l >= 1. A cell whose
//     diffusion is rho D1 has Q_l = rho Q1_l + Q2_l (Q is linear in D1 for fixed poles), so the 1-D tables serve every
//     rho: the kernel factors the lattice step's Q and forms a refinement's bridge itself (noise.h); and per margin
//     level the input node's largest share f_l of the bridge residual (the tight margin of a cell with rho > 1).
// Construction (host, long double): Phi and Q by scaling and squaring. Phi is entrywise
// non-negative (A is Metzler) and so is Q, so squaring and the doubling Q(2h) = Phi Q Phi^T + Q add
// non-negative terms without cancellation, for every a1 h up to 1e9 * 65.5 us and also for
// confluent rates; this replaces both the modal closed form, which needs care when a1 meets a2 or
// a3, and Van Loan's block exponential, which fails by a1 h ~ 200 (M3 V7b). The node rates a1 use
// the host's long-double exp (pmath has no long-double overload), the one host-libm transcendental
// behind the sensor's events: CPU and GPU kernels read the same host-built tables, so their bitwise
// parity (DESIGN §12 L1) holds, but the tables' last bits follow the host's long double and libm.
// The kernel interpolates Cholesky factors rather than covariances: a convex combination of
// factors is again a valid factor, so no in-kernel Cholesky (which fp32 could break on the
// h^3-ill-conditioned Q of short cells) is needed.
#pragma once

#include <cstdint>
#include <vector>

#include "core/hd.h"

namespace balla::sensor {

inline constexpr int kNoiseCellLog2 = 16;
inline constexpr int64_t kNoiseCellNs = int64_t{1} << kNoiseCellLog2;  // 65.536 us (M3 §D.4)

struct NoiseModel {
  double kq;       // k q: stationary input-node variance (log-e^2), M3 A.11
  double a2, a3;   // 1/tau_o, 1/tau_sf (s^-1)
  double sigma_o;  // unfiltered output-node noise (log-e)
  int levels;      // leaf level L
  double margin_z; // z_m
  bool split = false;  // also tabulate Phi_l and the parts Q1_l, Q2_l of Q_l (accuracy.noise_diffusion transient)
};

// Packed per-node layout; lower-triangular 3x3 matrices store (00, 10, 11, 20, 21, 22), full 3x3
// row-major, lower-triangular 6x6 row-major packed (21 entries).
struct NoiseLayout {
  int levels;
  bool split = false;
  static constexpr int kStepPhi = 0;    // 6
  static constexpr int kStepChol = 6;   // 6
  static constexpr int kPinfChol = 12;  // 6
  BALLA_HD int stride() const { return 18 + 47 * levels + (split ? 12 + 19 * levels : 0); }
  BALLA_HD int bridge(int l) const { return 18 + 24 * (l - 1); }          // l = 1..L: A 9, B 9, L 6
  BALLA_HD int margin(int l) const { return 18 + 24 * levels + 23 * l; }  // l = 0..L-1: sigma_br, g, W 21
  // Split tables only (symmetric 3x3 packed as the lower triangle): Q1_0, Q2_0 of the lattice step (12), then per
  // level l = 1..L: Phi_l (6, lower), Q1_l (6), Q2_l (6); then per level l = 0..L-1 the input node's largest share f_l
  // of the bridge residual of n3 (1; pixel.h noise_margin_rho).
  BALLA_HD int split_step() const { return 18 + 47 * levels; }
  BALLA_HD int split_level(int l) const { return 18 + 47 * levels + 12 + 18 * (l - 1); }
  BALLA_HD int split_margin(int l) const { return 18 + 47 * levels + 12 + 18 * levels + l; }
};

inline constexpr double kNoiseLnA1Min = -4.605170185988091;   // ln 1e-2
inline constexpr double kNoiseLnA1Max = 20.72326583694641;    // ln 1e9
inline constexpr int kNoiseNodesPerDecade = 64;

class NoiseTables {
 public:
  explicit NoiseTables(const NoiseModel& model);
  const NoiseModel& model() const { return model_; }
  NoiseLayout layout() const { return {model_.levels, model_.split}; }
  int n_nodes() const { return n_nodes_; }
  double ln_a1_min() const { return kNoiseLnA1Min; }
  double inv_dln() const { return inv_dln_; }
  const std::vector<double>& data64() const { return data64_; }
  const std::vector<float>& data32() const { return data32_; }

 private:
  NoiseModel model_;
  int n_nodes_ = 0;
  double inv_dln_ = 0;
  std::vector<double> data64_;
  std::vector<float> data32_;
};

// Exact (untabulated) matrices for one rate, used to build the tables and by the tests.
struct NoiseMatrices {
  double phi[3][3];
  double q[3][3];
};
NoiseMatrices noise_transition(const NoiseModel& m, double a1, double h_s);
void noise_stationary_cov(const NoiseModel& m, double a1, double p[3][3]);

// Kernel view of the tables (raw pointers into host or device memory).
template <class T>
struct NoiseTableView {
  const T* data;
  int n_nodes;
  int stride;
  int levels;
  T ln_a1_min;
  T inv_dln;
};

// Interpolation position of one ln a1 on the grid (clamped to the table range).
template <class T>
struct NoisePos {
  int i;
  T f;
};

// The clamp is written so that NaN lands on the first node: the position indexes memory, and a NaN
// converted to int is undefined (in practice INT_MIN, far outside the table).
template <class T>
BALLA_HD NoisePos<T> noise_pos(const NoiseTableView<T>& v, T ln_a1) {
  T x = (ln_a1 - v.ln_a1_min) * v.inv_dln;
  const T x_max = static_cast<T>(v.n_nodes - 1);
  x = x > T(0) ? (x < x_max ? x : x_max) : T(0);
  int i = static_cast<int>(x);
  if (i > v.n_nodes - 2) i = v.n_nodes - 2;
  return {i, x - static_cast<T>(i)};
}

// Interpolated block of `count` consecutive entries at `offset`.
template <class T>
BALLA_HD void noise_lookup(const NoiseTableView<T>& v, NoisePos<T> p, int offset, int count, T* out) {
  const T* a = v.data + static_cast<int64_t>(p.i) * v.stride + offset;
  const T* b = a + v.stride;
  for (int j = 0; j < count; ++j) out[j] = a[j] + p.f * (b[j] - a[j]);
}

}  // namespace balla::sensor
