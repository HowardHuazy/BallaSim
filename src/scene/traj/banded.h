// Banded SPD algebra of the fitter (ballasim.traj.banded; M2 §C.4, §C.6): LAPACK upper band storage, the band
// Cholesky (dpbtrf) and its solve (dpbtrs) as scipy.linalg.cholesky_banded / cho_solve_banded call them, and
// Takahashi's selected inversion with the trace tr(A^-1 M), in NumPy's einsum order.
#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "scene/sparse.h"

namespace balla::scene::traj {

// LAPACK upper band storage, the values of to_upper_band's (u + 1, n) array, kept column-major as dpbtrf takes it:
// ab[(u + i - j) + j (u + 1)] = a[i, j] for i <= j <= i + u, zero elsewhere.
struct Band {
  int64_t n = 0, u = 0;
  std::vector<double> ab;

  Band() = default;
  // LAPACK's sizes: 0 <= n, u < 2^31 (u may exceed n - 1, as LAPACK and the Python allow).
  Band(int64_t n_, int64_t u_) : n(checked(n_, "n")), u(checked(u_, "u")), ab(static_cast<size_t>((u + 1) * n), 0.0) {}
  double& at(int64_t i, int64_t j) { return ab[static_cast<size_t>((u + i - j) + j * (u + 1))]; }
  double at(int64_t i, int64_t j) const { return ab[static_cast<size_t>((u + i - j) + j * (u + 1))]; }

 private:
  static int64_t checked(int64_t v, const char* what) {
    if (!(0 <= v && v < (int64_t{1} << 31))) throw std::invalid_argument(std::string("band ") + what + " outside [0, 2^31)");
    return v;
  }
};

// to_upper_band of a sparse matrix (duplicates summed first, as sum_duplicates).
Band to_upper_band(const sparse::Matrix& a, int64_t u);

class BandedSPD {
 public:
  // u < 0: the bandwidth of a's stored entries. Throws for a non-finite band (check_finite) or a matrix that is not
  // positive definite (LinAlgError).
  BandedSPD(const sparse::Matrix& a, int64_t u = -1);
  // From the band itself (what to_upper_band of the matrix gives).
  explicit BandedSPD(const Band& band);
  // Solves A x = b for k right-hand sides (b (n, k) C-ordered), dpbtrs.
  std::vector<double> solve(std::span<const double> b, int64_t k = 1) const;
  int64_t n() const { return n_; }
  int64_t u() const { return u_; }
  // The factor in NumPy's (u + 1, n) row-major layout (made on first use).
  const std::vector<double>& factor() const;

 private:
  void factorize(const std::vector<double>& ab);
  int64_t n_, u_;
  std::vector<double> fortran_;           // LAPACK's column-major band (ldab = u + 1)
  mutable std::vector<double> factor_;    // NumPy's layout, for the selected inversion
};

// selected_inverse_band of L factors (each (u + 1, n) row-major): S[l][d][i] = (A_l^-1)[i, i + d], returned as
// (L, u + 1, n) row-major.
std::vector<double> selected_inverse_band(const std::vector<const std::vector<double>*>& factors, int64_t u, int64_t n);

// trace_of_product: tr(A_l^-1 M) for each l, from the band of A_l^-1 and the upper band of a symmetric M.
std::vector<double> trace_of_product(const std::vector<double>& s_band, int64_t lcount, const Band& m);

}  // namespace balla::scene::traj
