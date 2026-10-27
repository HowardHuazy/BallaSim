// SciPy's sparse matrices as the fitter uses them (docs/dev/CPP_PORT.md item 5, E2): CSR and CSC with SciPy's
// structure, so every product sums its terms in SciPy's order.
//
// The operations follow scipy.sparse 1.18 and its sparsetools: csr_matmat (SMMP, output columns in the order of the
// linked list, exact zeros dropped), csr_tocsc, coo_tocsr then sum_duplicates when not canonical (std::sort of
// (index, value) pairs per row, sequential sums), csr_binop_csr (canonical merge or the general linked list),
// csr_matvec / csr_matvecs / csc_matvec / csc_matvecs, multiply by a dense row or column (COO data times the
// vector, explicit zeros kept), sp.diags, sp.kron with the 3 x 3 identity (COO, then CSR).
#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace balla::scene::sparse {

enum class Format { kCsr, kCsc };

// A compressed sparse matrix: CSR (major = rows) or CSC (major = columns), shape (rows, cols).
struct Matrix {
  Format format = Format::kCsr;
  int64_t rows = 0, cols = 0;
  std::vector<int64_t> indptr;   // major + 1
  std::vector<int64_t> indices;  // minor index per entry
  std::vector<double> data;

  int64_t major() const { return format == Format::kCsr ? rows : cols; }
  int64_t minor() const { return format == Format::kCsr ? cols : rows; }
  int64_t nnz() const { return static_cast<int64_t>(data.size()); }
};

// sp.csr_matrix((data, indices, indptr), shape): the arrays as given.
Matrix csr(int64_t rows, int64_t cols, std::vector<int64_t> indptr, std::vector<int64_t> indices,
           std::vector<double> data);

// sp.csr_matrix((data, (row, col)), shape) and coo.tocsr(): coo_tocsr, then sum_duplicates unless canonical.
Matrix coo_to_csr(int64_t rows, int64_t cols, std::span<const int64_t> row, std::span<const int64_t> col,
                  std::span<const double> data);

// .T: the same arrays in the other format.
Matrix transpose(const Matrix& a);
// .tocsr() / .tocsc() (a copy when already in that format: callers only read it).
Matrix to_csr(const Matrix& a);
Matrix to_csc(const Matrix& a);

// a @ b for two sparse matrices (_matmul_sparse: b converted to a's format, csr_matmat or csc_matmat).
Matrix matmul(const Matrix& a, const Matrix& b);
// a @ x for a dense vector (zeros, then csr_matvec / csc_matvec).
std::vector<double> matvec(const Matrix& a, std::span<const double> x);
// a @ X for a dense C-ordered (cols, k) matrix (zeros, then csr_matvecs / csc_matvecs).
std::vector<double> matvecs(const Matrix& a, std::span<const double> x, int64_t k);

// a + b (CSR, _binopt '_plus_': canonical merge when both are canonical, else the general linked list; zeros pruned).
Matrix add(const Matrix& a, const Matrix& b);
// s * a (data times s).
Matrix scaled(const Matrix& a, double s);
// a.multiply(w[None, :]).tocsr() and a.multiply(w[:, None]).tocsr(): COO data times w at each column / row.
Matrix multiply_columns(const Matrix& a, std::span<const double> w);
Matrix multiply_rows(const Matrix& a, std::span<const double> w);
// a.diagonal() (csr_diagonal: 0 + the sum of the diagonal entries of each row).
std::vector<double> diagonal(const Matrix& a);
// a + sp.diags(d) for a CSR a (the diagonal as CSR without its zeros).
Matrix add_diagonal(const Matrix& a, std::span<const double> d);
// sp.kron(a, sp.identity(3), format="csr") for a CSR a.
Matrix kron_identity3(const Matrix& a);

// csr_has_canonical_format: sorted, no duplicates.
bool has_canonical_format(const Matrix& a);
// sum_duplicates() in place (sort_indices, csr_sum_duplicates, prune) unless canonical.
void sum_duplicates(Matrix& a);
// The largest |row - col| over the stored entries (fit.banded.bandwidth: a.tocoo()).
int64_t bandwidth(const Matrix& a);

}  // namespace balla::scene::sparse
