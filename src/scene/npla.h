// NumPy's matmul and linalg as the EVIMO2 adapter calls them (docs/dev/CPP_PORT.md item 5, E1/E2): the BLAS or LAPACK
// call NumPy 2.5 makes for each operand layout (umath matmul.c.src, umath_linalg.cpp), with its arguments and
// workspace queries, so every result is bitwise NumPy's.
#pragma once

#include <cstdint>
#include <vector>

namespace balla::scene::npo {

// A strided 2-D float64 view: element (i, j) at p[i s0 + j s1] (strides in elements, as NumPy's byte strides / 8).
struct View2 {
  const double* p;
  int64_t rows, cols, s0, s1;
  double at(int64_t i, int64_t j) const { return p[i * s0 + j * s1]; }
  View2 t() const { return {p, cols, rows, s1, s0}; }   // the transposed view (a.T)
};

// A C-contiguous (rows, cols) matrix.
inline View2 c_view(const double* p, int64_t rows, int64_t cols) { return {p, rows, cols, cols, 1}; }

// np.matmul(a, b) of one pair of matrices into the C-contiguous (a.rows, b.cols) `out`: DOUBLE_matmul's choice of
// cblas_dgemm (both operands BLAS-able in C or F order), cblas_dgemv (a vector operand), DOUBLE_dot (1 x 1 result) or
// its plain loop, with NumPy's transpose flags and leading dimensions.
void matmul(View2 a, View2 b, double* out);
std::vector<double> matmul(View2 a, View2 b);

// np.linalg.inv of an n x n row-major matrix (dgesv on the identity, as umath_linalg's inv).
std::vector<double> inv(const double* a, int n);

// np.linalg.det of an n x n row-major matrix: sign * exp(sum log |u_ii|) of dgetrf's LU (0 when singular).
double det(const double* a, int n);

// np.linalg.svd of an m x n row-major matrix (dgesdd): full_matrices, u (m x m) and vt (n x n) row-major, s
// descending; or only s (compute_uv=False, jobz 'N').
void svd(const double* a, int m, int n, double* u, double* s, double* vt);
void svd_values(const double* a, int m, int n, double* s);

// np.linspace(start, stop, num) (endpoint, float64).
std::vector<double> linspace(double start, double stop, int64_t num);

}  // namespace balla::scene::npo
