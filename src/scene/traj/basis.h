// Uniform B-spline bases, knot grids and the roughness penalty of the fitter (ballasim.traj.basis; M2 §A.1, §A.4,
// §C.3, §C.5), bitwise as the Python fitter builds them: the basis rows through np.matmul (cblas_dgemm), the
// matrices with SciPy's sparse structure.
#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "scene/sparse.h"

namespace balla::scene::traj {

// Uniform knots t_c = t0 + c h (int64 ns); segment s covers [t0 + s h, t0 + (s+1) h).
struct KnotGrid {
  int64_t t0_ns = 0;
  int64_t h_ns = 1;
  int order = 6;
  int64_t num_ctrl = 6;

  KnotGrid() = default;
  KnotGrid(int64_t t0, int64_t h, int k, int64_t n);   // validates as the Python dataclass
  // The grid with t0 = t_begin whose closed support [t0, t0 + N_seg h] covers t_end.
  static KnotGrid covering(int64_t t_begin_ns, int64_t t_end_ns, int64_t h_ns, int order);
  int64_t num_segments() const { return num_ctrl - order + 1; }
  // Segment index and u in [0, 1] per time (exact integer arithmetic; closed on the right). Throws outside.
  void locate(std::span<const int64_t> t_ns, std::vector<int64_t>& s, std::vector<double>& u) const;
  // B with (B c)_n = sum_j b_j(u_n) c_{s_n + j} (sp.csr_matrix((data, cols, ptr))).
  sparse::Matrix design_matrix(std::span<const int64_t> t_ns) const;
  // Control-point centres tau_c = t0 + (c - K/2 + 1) h.
  std::vector<int64_t> greville_ns() const;
  // Controls whose basis is non-zero on [t_begin, t_end], clipped to the grid.
  std::pair<int64_t, int64_t> controls_touching(int64_t t_begin_ns, int64_t t_end_ns) const;
  // The knot-aligned [t_begin, t_end) on which the spline is constant when controls c0..c1 are equal.
  std::pair<int64_t, int64_t> still_interval(int64_t c0, int64_t c1) const;
};

// basis(u, K) (n, K) and cumulative_basis(u, K) (n, K - 1), row-major: _powers(u) @ M.T as np.matmul computes it.
std::vector<double> basis(std::span<const double> u, int order);
std::vector<double> cumulative_basis(std::span<const double> u, int order);

// segment_gram(K, m) for (6, 3) and (4, 2): the Gauss-Legendre values the Python computes (constants, checked by the
// tests against Python's), k x k row-major with k = K - m.
std::span<const double> segment_gram(int order, int m);

sparse::Matrix difference_matrix(int64_t n, int order);
sparse::Matrix difference_gram(int64_t num_ctrl, int order, int m);
sparse::Matrix roughness_matrix(int64_t num_ctrl, int order, int m);

}  // namespace balla::scene::traj
