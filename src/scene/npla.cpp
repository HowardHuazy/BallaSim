// Contains C++ translations of NumPy (umath matmul.c.src's BLAS dispatch, umath_linalg's det and its LAPACK layouts,
// np.linspace), under their BSD licences: see THIRD_PARTY_NOTICES.md, "Ported code".
#include "scene/npla.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scene/blas.h"
#include "scene/errors.h"

namespace balla::scene::npo {

namespace {

constexpr int64_t kBlasMax = std::numeric_limits<int>::max();

// is_blasable2d (strides in elements): unit stride along the second axis, a leading stride >= the second extent.
bool blasable(int64_t s1, int64_t s2, int64_t d2) { return s2 == 1 && s1 >= d2 && s1 <= kBlasMax; }

// DOUBLE_dot: cblas_ddot on BLAS-able strides (0.0 + the sum), else a sequential sum from 0.
double vec_dot(const double* x, int64_t sx, const double* y, int64_t sy, int64_t n) {
  if (sx > 0 && sy > 0 && sx <= kBlasMax && sy <= kBlasMax && n <= kBlasMax) {
    double sum = 0.0;
    sum += cblas_ddot(static_cast<int>(n), x, static_cast<int>(sx), y, static_cast<int>(sy));
    return sum;
  }
  double sum = 0.0;
  for (int64_t i = 0; i < n; ++i) sum += x[i * sx] * y[i * sy];
  return sum;
}

void noblas(View2 a, View2 b, double* out) {
  for (int64_t i = 0; i < a.rows; ++i) {
    for (int64_t j = 0; j < b.cols; ++j) {
      double acc = 0.0;
      for (int64_t k = 0; k < a.cols; ++k) acc += a.at(i, k) * b.at(k, j);
      out[i * b.cols + j] = acc;
    }
  }
}

// DOUBLE_gemv(ip1, is1_m, is1_n, ip2, is2_n, ...): y = A x with A (m x n).
void gemv(const double* a, int64_t s_m, int64_t s_n, int64_t m, int64_t n, const double* x, int64_t sx, double* y,
          int64_t sy) {
  int order, mm, nn, lda;
  if (blasable(s_m, s_n, n)) {
    order = blas::kColMajor;
    mm = static_cast<int>(n);
    nn = static_cast<int>(m);
    lda = static_cast<int>(s_m);
  } else {
    order = blas::kRowMajor;
    mm = static_cast<int>(m);
    nn = static_cast<int>(n);
    lda = static_cast<int>(s_n);
  }
  cblas_dgemv(order, blas::kTrans, nn, mm, 1.0, a, lda, x, static_cast<int>(sx), 0.0, y, static_cast<int>(sy));
}

}  // namespace

void matmul(View2 a, View2 b, double* out) {
  if (a.cols != b.rows) throw std::invalid_argument("matmul: dimension mismatch");
  const int64_t dm = a.rows, dn = a.cols, dp = b.cols;
  const bool any_zero = dm == 0 || dn == 0 || dp == 0;
  const bool too_big = dm > kBlasMax || dn > kBlasMax || dp > kBlasMax;
  if (any_zero || too_big) {
    if (dm * dp > 0) noblas(a, b, out);
    return;
  }
  const bool i1_c = blasable(a.s0, a.s1, dn), i1_f = blasable(a.s1, a.s0, dm);
  const bool i2_c = blasable(b.s0, b.s1, dp), i2_f = blasable(b.s1, b.s0, dn);
  const bool i1 = i1_c || i1_f, i2 = i2_c || i2_f;
  const int64_t os_m = dp, os_p = 1;   // the C-contiguous output
  const bool special = dm == 1 || dn == 1 || dp == 1;
  if (special) {
    const bool scalar_out = dm == 1 && dp == 1;
    const bool scalar_vec = dn == 1 && (dp == 1 || dm == 1);
    const bool vector_matrix = dm == 1 && i2 && blasable(a.s1, 1, 1);
    const bool matrix_vector = dp == 1 && i1 && blasable(b.s0, 1, 1);
    if (scalar_out) {
      out[0] = vec_dot(a.p, a.s1, b.p, b.s0, dn);
    } else if (scalar_vec) {
      noblas(a, b, out);
    } else if (vector_matrix) {
      gemv(b.p, b.s1, b.s0, dp, dn, a.p, a.s1, out, os_p);
    } else if (matrix_vector) {
      gemv(a.p, a.s0, a.s1, dm, dn, b.p, b.s0, out, os_m);
    } else {
      noblas(a, b, out);
    }
    return;
  }
  if (!(i1 && i2)) {
    noblas(a, b, out);
    return;
  }
  int trans1, trans2, lda, ldb;
  if (blasable(a.s0, a.s1, dn)) {
    trans1 = blas::kNoTrans;
    lda = static_cast<int>(a.s0);
  } else {
    trans1 = blas::kTrans;
    lda = static_cast<int>(a.s1);
  }
  if (blasable(b.s0, b.s1, dp)) {
    trans2 = blas::kNoTrans;
    ldb = static_cast<int>(b.s0);
  } else {
    trans2 = blas::kTrans;
    ldb = static_cast<int>(b.s1);
  }
  if (a.p == b.p && dm == dp && a.s0 == b.s1 && a.s1 == b.s0 && trans1 != trans2) {
    throw std::logic_error("matmul: a @ a.T (NumPy's syrk path) is not ported");
  }
  cblas_dgemm(blas::kRowMajor, trans1, trans2, static_cast<int>(dm), static_cast<int>(dp), static_cast<int>(dn), 1.0,
              a.p, lda, b.p, ldb, 0.0, out, static_cast<int>(os_m));
}

std::vector<double> matmul(View2 a, View2 b) {
  std::vector<double> out(static_cast<size_t>(a.rows * b.cols));
  matmul(a, b, out.data());
  return out;
}

std::vector<double> inv(const double* a, int n) {
  std::vector<double> af(static_cast<size_t>(n * n)), bf(static_cast<size_t>(n * n), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) af[static_cast<size_t>(j * n + i)] = a[i * n + j];
    bf[static_cast<size_t>(i * n + i)] = 1.0;
  }
  std::vector<int> ipiv(static_cast<size_t>(n));
  const int ld = std::max(n, 1);
  int info = 0;
  dgesv_(&n, &n, af.data(), &ld, ipiv.data(), bf.data(), &ld, &info);
  if (info > 0) throw LinAlgError("Singular matrix");
  if (info < 0) blas::lapack_error("dgesv", info);
  std::vector<double> out(static_cast<size_t>(n * n));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) out[static_cast<size_t>(i * n + j)] = bf[static_cast<size_t>(j * n + i)];
  }
  return out;
}

double det(const double* a, int n) {
  std::vector<double> af(static_cast<size_t>(n * n));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) af[static_cast<size_t>(j * n + i)] = a[i * n + j];
  }
  std::vector<int> ipiv(static_cast<size_t>(n));
  const int ld = std::max(n, 1);
  int info = 0;
  dgetrf_(&n, &n, af.data(), &ld, ipiv.data(), &info);
  if (info < 0) blas::lapack_error("dgetrf", info);
  if (info > 0) return 0.0 * std::exp(-std::numeric_limits<double>::infinity());
  int changes = 0;
  for (int i = 0; i < n; ++i) changes += ipiv[static_cast<size_t>(i)] != i + 1;
  double sign = changes % 2 ? -1.0 : 1.0, logdet = 0.0;
  for (int i = 0; i < n; ++i) {
    double d = af[static_cast<size_t>(i * n + i)];
    if (d < 0.0) {
      sign = -sign;
      d = -d;
    }
    logdet += std::log(d);
  }
  return sign * std::exp(logdet);
}

namespace {

void gesdd(char jobz, const double* a, int m, int n, double* u, double* s, double* vt) {
  std::vector<double> af(static_cast<size_t>(m * n));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) af[static_cast<size_t>(j * m + i)] = a[i * n + j];
  }
  const int lda = std::max(m, 1), mn = std::min(m, n);
  const bool all = jobz == 'A';
  const int ldu = all ? std::max(m, 1) : 1, ldvt = all ? std::max(n, 1) : 1;
  std::vector<double> uf(all ? static_cast<size_t>(m * m) : 1), vtf(all ? static_cast<size_t>(n * n) : 1);
  std::vector<int> iwork(static_cast<size_t>(8 * mn));
  int info = 0, lwork = -1;
  double query = 0.0;
  dgesdd_(&jobz, &m, &n, af.data(), &lda, s, uf.data(), &ldu, vtf.data(), &ldvt, &query, &lwork, iwork.data(), &info,
          1);
  if (info != 0) blas::lapack_error("dgesdd (workspace query)", info);
  lwork = static_cast<int>(query);
  if (lwork == 0) lwork = 1;
  std::vector<double> work(static_cast<size_t>(lwork));
  dgesdd_(&jobz, &m, &n, af.data(), &lda, s, uf.data(), &ldu, vtf.data(), &ldvt, work.data(), &lwork, iwork.data(),
          &info, 1);
  if (info > 0) throw LinAlgError("SVD did not converge");
  if (info < 0) blas::lapack_error("dgesdd", info);
  if (all) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) u[i * m + j] = uf[static_cast<size_t>(j * m + i)];
    }
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) vt[i * n + j] = vtf[static_cast<size_t>(j * n + i)];
    }
  }
}

}  // namespace

void svd(const double* a, int m, int n, double* u, double* s, double* vt) { gesdd('A', a, m, n, u, s, vt); }

void svd_values(const double* a, int m, int n, double* s) { gesdd('N', a, m, n, nullptr, s, nullptr); }

std::vector<double> linspace(double start, double stop, int64_t num) {
  if (num < 0) throw std::invalid_argument("Number of samples, " + std::to_string(num) + ", must be non-negative.");
  std::vector<double> y(static_cast<size_t>(num));
  const int64_t div = num - 1;
  const double delta = stop - start;
  if (div > 0) {
    const double step = delta / static_cast<double>(div);
    for (int64_t i = 0; i < num; ++i) {
      y[static_cast<size_t>(i)] = step == 0.0 ? (static_cast<double>(i) / static_cast<double>(div)) * delta
                                              : static_cast<double>(i) * step;
    }
  } else {
    for (int64_t i = 0; i < num; ++i) y[static_cast<size_t>(i)] = static_cast<double>(i) * delta;
  }
  for (double& v : y) v += start;
  if (num > 1) y.back() = stop;
  return y;
}

}  // namespace balla::scene::npo
