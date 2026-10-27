// Contains C++ translations of NumPy (pairwise_sum_DOUBLE; the median, percentile, std and arange order), under their BSD licences:
// see THIRD_PARTY_NOTICES.md, "Ported code".
#include "scene/npo.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scene/blas.h"
#include "scene/errors.h"

namespace balla::scene::npo {

double pairwise_sum(const double* a, int64_t n, int64_t stride) {
  if (n < 8) {
    double res = 0.0;
    for (int64_t i = 0; i < n; ++i) res += a[i * stride];
    return res;
  }
  if (n <= 128) {
    double r[8];
    for (int k = 0; k < 8; ++k) r[k] = a[k * stride];
    int64_t i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (int k = 0; k < 8; ++k) r[k] += a[(i + k) * stride];
    }
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i * stride];
    return res;
  }
  int64_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum(a, n2, stride) + pairwise_sum(a + n2 * stride, n - n2, stride);
}

namespace {

// np.median's partition and mean of the middle one or two values, on float64 values.
double median_of(std::vector<double>& a) {
  const size_t n = a.size();
  if (n == 0) return std::numeric_limits<double>::quiet_NaN();
  const size_t h = n / 2;
  std::nth_element(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(h), a.end());
  const double hi = a[h];
  if (n % 2 == 1) return (0.0 + (0.0 + hi)) / 1.0;
  const double lo = *std::max_element(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(h));
  return (0.0 + ((0.0 + lo) + hi)) / 2.0;
}

}  // namespace

double median(std::vector<double> a) {
  for (const double x : a) {
    if (std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();
  }
  return median_of(a);
}

double median(std::span<const int64_t> a) {
  std::vector<int64_t> s(a.begin(), a.end());
  const size_t n = s.size();
  if (n == 0) return std::numeric_limits<double>::quiet_NaN();
  const size_t h = n / 2;
  std::nth_element(s.begin(), s.begin() + static_cast<std::ptrdiff_t>(h), s.end());
  const auto hi = static_cast<double>(s[h]);
  if (n % 2 == 1) return (0.0 + (0.0 + hi)) / 1.0;
  const auto lo = static_cast<double>(*std::max_element(s.begin(), s.begin() + static_cast<std::ptrdiff_t>(h)));
  return (0.0 + ((0.0 + lo) + hi)) / 2.0;
}

double percentile(std::vector<double> a, double q) {
  const auto n = static_cast<int64_t>(a.size());
  if (n == 0) throw std::invalid_argument("percentile of an empty array");
  for (const double x : a) {
    if (std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();
  }
  const double quantile = q / 100.0;
  const double virt = static_cast<double>(n - 1) * quantile;
  std::sort(a.begin(), a.end());
  double prev = std::floor(virt);
  double next = prev + 1.0;
  int64_t ip, in;
  if (virt >= static_cast<double>(n - 1)) {
    ip = in = n - 1;
  } else if (virt < 0) {
    ip = in = 0;
  } else {
    ip = static_cast<int64_t>(prev);
    in = static_cast<int64_t>(next);
  }
  const double gamma = virt - prev;
  const double lo = a[static_cast<size_t>(ip)], hi = a[static_cast<size_t>(in)];
  const double diff = hi - lo;
  if (gamma >= 0.5) return hi - diff * (1.0 - gamma);
  return lo + diff * gamma;
}

double std0(std::span<const double> a) {
  const auto n = static_cast<double>(a.size());
  const double m = sum(a) / n;
  std::vector<double> x(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - m;
    x[i] = d * d;
  }
  return std::sqrt(sum(x) / n);
}

double py_pow(double x, double y) {
  static double (*volatile pow_fn)(double, double) = &::pow;
  return pow_fn(x, y);
}

std::vector<double> arange(double start, double stop, double step) {
  // NumPy's errors: a zero step divides by zero, a NaN length cannot be computed, a length past intp is too big.
  if (step == 0.0) throw ZeroDivisionError("division by zero");
  const double span = stop - start, ratio = span / step;
  // A ratio that underflows to zero from a nonzero span counts as one element (or none, if negative), as in NumPy.
  const double len = ratio == 0.0 && span != 0.0 ? (std::signbit(ratio) ? 0.0 : 1.0) : std::ceil(ratio);
  if (std::isnan(len)) throw std::invalid_argument("arange: cannot compute length");
  if (!(len > 0)) return {};
  if (!(len < 0x1p63)) throw std::invalid_argument("Maximum allowed size exceeded");
  const auto n = static_cast<size_t>(len);
  std::vector<double> out(n);
  out[0] = start;
  if (n > 1) {
    out[1] = start + step;
    const double delta = out[1] - start;
    for (size_t i = 2; i < n; ++i) out[i] = start + static_cast<double>(i) * delta;
  }
  return out;
}

double round_decimals(double x, int decimals) {
  const double f = py_pow(10.0, decimals);
  return std::nearbyint(x * f) / f;
}

int64_t argmin(std::span<const double> a) {
  if (a.empty()) throw std::invalid_argument("attempt to get argmin of an empty sequence");
  int64_t best = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::isnan(a[i])) return static_cast<int64_t>(i);
    if (a[i] < a[static_cast<size_t>(best)]) best = static_cast<int64_t>(i);
  }
  return best;
}

int64_t argmax(std::span<const double> a) {
  if (a.empty()) throw std::invalid_argument("attempt to get argmax of an empty sequence");
  int64_t best = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::isnan(a[i])) return static_cast<int64_t>(i);
    if (a[i] > a[static_cast<size_t>(best)]) best = static_cast<int64_t>(i);
  }
  return best;
}

int64_t nanargmin(std::span<const double> a) {
  if (a.empty()) throw std::invalid_argument("attempt to get argmin of an empty sequence");
  std::vector<double> b(a.begin(), a.end());
  bool all_nan = !b.empty();
  for (double& x : b) {
    if (std::isnan(x)) {
      x = std::numeric_limits<double>::infinity();
    } else {
      all_nan = false;
    }
  }
  if (all_nan) throw std::invalid_argument("All-NaN slice encountered");
  return argmin(b);
}

double max(std::span<const double> a) {
  if (a.empty()) throw std::invalid_argument("zero-size array to reduction operation maximum which has no identity");
  double m = a[0];
  for (const double x : a) {
    if (std::isnan(x)) return x;
    if (x > m) m = x;
  }
  return m;
}

double min(std::span<const double> a) {
  if (a.empty()) throw std::invalid_argument("zero-size array to reduction operation minimum which has no identity");
  double m = a[0];
  for (const double x : a) {
    if (std::isnan(x)) return x;
    if (x < m) m = x;
  }
  return m;
}

std::vector<double> lstsq(std::span<const double> a, int m, int n, std::span<const double> b, double rcond) {
  if (m < n || static_cast<int>(b.size()) != m) throw std::invalid_argument("lstsq: need m >= n and b of length m");
  std::vector<double> af(static_cast<size_t>(m) * static_cast<size_t>(n));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      af[static_cast<size_t>(j) * static_cast<size_t>(m) + static_cast<size_t>(i)] =
          a[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)];
    }
  }
  std::vector<double> bf(b.begin(), b.end()), s(static_cast<size_t>(n));
  const int nrhs = 1, lda = std::max(m, 1), ldb = std::max(std::max(m, n), 1);
  int rank = 0, info = 0, lwork = -1, iwork_query = 0;
  double work_query = 0.0;
  dgelsd_(&m, &n, &nrhs, af.data(), &lda, bf.data(), &ldb, s.data(), &rcond, &rank, &work_query, &lwork, &iwork_query,
          &info);
  if (info != 0) blas::lapack_error("dgelsd (workspace query)", info);
  lwork = static_cast<int>(work_query);
  std::vector<double> work(static_cast<size_t>(lwork));
  std::vector<int> iwork(static_cast<size_t>(std::max(iwork_query, 1)));
  dgelsd_(&m, &n, &nrhs, af.data(), &lda, bf.data(), &ldb, s.data(), &rcond, &rank, work.data(), &lwork, iwork.data(),
          &info);
  if (info > 0) throw LinAlgError("SVD did not converge in Linear Least Squares");
  if (info != 0) blas::lapack_error("dgelsd", info);
  bf.resize(static_cast<size_t>(n));
  return bf;
}

std::vector<double> polyfit2(const double x[3], const double y[3]) {
  double lhs[9];
  for (int i = 0; i < 3; ++i) {
    const double xi = x[i] + 0.0;
    lhs[3 * i + 0] = xi * xi;
    lhs[3 * i + 1] = xi;
    lhs[3 * i + 2] = 1.0;
  }
  double scale[3];
  for (int j = 0; j < 3; ++j) {
    double acc = 0.0;
    for (int i = 0; i < 3; ++i) acc += lhs[3 * i + j] * lhs[3 * i + j];
    scale[j] = std::sqrt(acc);
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) lhs[3 * i + j] /= scale[j];
  }
  const double yy[3] = {y[0] + 0.0, y[1] + 0.0, y[2] + 0.0};
  std::vector<double> c = lstsq(lhs, 3, 3, yy, 3.0 * std::numeric_limits<double>::epsilon());
  for (int j = 0; j < 3; ++j) c[static_cast<size_t>(j)] /= scale[j];
  return c;
}

void solve3(const double a[9], double b[9]) {
  double af[9], bf[9];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      af[3 * j + i] = a[3 * i + j];
      bf[3 * j + i] = b[3 * i + j];
    }
  }
  const int n = 3, nrhs = 3, lda = 3, ldb = 3;
  int ipiv[3], info = 0;
  dgesv_(&n, &nrhs, af, &lda, ipiv, bf, &ldb, &info);
  if (info > 0) throw LinAlgError("Singular matrix");
  if (info < 0) blas::lapack_error("dgesv", info);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) b[3 * i + j] = bf[3 * j + i];
  }
}

void eigh4(const double a[16], double w[4], double v[16]) {
  double af[16];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) af[4 * j + i] = a[4 * i + j];
  }
  const char jobz = 'V', uplo = 'L';
  const int n = 4, lda = 4;
  int info = 0, lwork = -1, liwork = -1, iwork_query = 0;
  double work_query = 0.0;
  dsyevd_(&jobz, &uplo, &n, af, &lda, w, &work_query, &lwork, &iwork_query, &liwork, &info, 1, 1);
  if (info != 0) blas::lapack_error("dsyevd (workspace query)", info);
  lwork = static_cast<int>(work_query);
  liwork = iwork_query;
  std::vector<double> work(static_cast<size_t>(lwork));
  std::vector<int> iwork(static_cast<size_t>(std::max(liwork, 1)));
  dsyevd_(&jobz, &uplo, &n, af, &lda, w, work.data(), &lwork, iwork.data(), &liwork, &info, 1, 1);
  if (info > 0) throw LinAlgError("Eigenvalues did not converge");
  if (info != 0) blas::lapack_error("dsyevd", info);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) v[4 * i + j] = af[4 * j + i];
  }
}

}  // namespace balla::scene::npo
