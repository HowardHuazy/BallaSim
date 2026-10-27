#include "scene/traj/basis.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "core/time/time.h"
#include "scene/blas.h"
#include "scene/errors.h"
#include "scene/npo.h"

namespace balla::scene::traj {

namespace {

int64_t comb(int64_t n, int64_t k) {
  if (k < 0 || k > n) return 0;
  int64_t r = 1;
  for (int64_t i = 1; i <= k; ++i) r = r * (n - k + i) / i;
  return r;
}

int64_t factorial(int64_t n) { return n <= 1 ? 1 : n * factorial(n - 1); }

int64_t ipow(int64_t b, int64_t e) {
  int64_t r = 1;
  for (int64_t i = 0; i < e; ++i) r *= b;
  return r;
}

// (K-1)! * M^(K) of M2 (A3), exact integers: b(u) = M [1, u, ..., u^(K-1)]^T.
std::vector<int64_t> blend_numerators(int k) {
  std::vector<int64_t> out(static_cast<size_t>(k * k));
  for (int j = 0; j < k; ++j) {
    for (int n = 0; n < k; ++n) {
      int64_t acc = 0;
      for (int l = j; l < k; ++l) {
        acc += ((l - j) % 2 ? -1 : 1) * comb(k, l - j) * ipow(k - 1 - l, k - 1 - n);
      }
      out[static_cast<size_t>(j * k + n)] = comb(k - 1, n) * acc;
    }
  }
  return out;
}

// float(Fraction) of the blend (cumulative: summed over l >= j) matrix, row-major K x K.
std::vector<double> blend_matrix(int k, bool cumulative) {
  const std::vector<int64_t> num = blend_numerators(k);
  const auto den = static_cast<double>(factorial(k - 1));
  std::vector<double> out(static_cast<size_t>(k * k));
  for (int j = 0; j < k; ++j) {
    for (int n = 0; n < k; ++n) {
      int64_t v = 0;
      for (int l = cumulative ? j : j; l < (cumulative ? k : j + 1); ++l) v += num[static_cast<size_t>(l * k + n)];
      out[static_cast<size_t>(j * k + n)] = static_cast<double>(v) / den;
    }
  }
  return out;
}

// _powers(u, K, 0): u ** n (n = 1: u itself, n = 2: np.square, else np.power -> pow).
std::vector<double> powers(std::span<const double> u, int k) {
  std::vector<double> out(u.size() * static_cast<size_t>(k));
  for (size_t i = 0; i < u.size(); ++i) {
    for (int n = 0; n < k; ++n) {
      const double x = u[i];
      const double p = n == 1 ? x : n == 2 ? x * x : npo::py_pow(x, static_cast<double>(n));
      out[i * static_cast<size_t>(k) + static_cast<size_t>(n)] = 1.0 * p;
    }
  }
  return out;
}

// np.matmul(P, B.T) for P (n, k) row-major and B (p, k) row-major (a transposed view): dgemm, or for one row
// NumPy's vector @ matrix gemv.
std::vector<double> matmul_bt(const std::vector<double>& p, size_t n, int k, const double* b, int rows_b) {
  std::vector<double> out(n * static_cast<size_t>(rows_b));
  if (n == 0) return out;
  if (n == 1) {
    cblas_dgemv(blas::kColMajor, blas::kTrans, k, rows_b, 1.0, b, k, p.data(), 1, 0.0, out.data(), 1);
  } else {
    cblas_dgemm(blas::kRowMajor, blas::kNoTrans, blas::kTrans, static_cast<int>(n), rows_b, k, 1.0, p.data(), k, b, k,
                0.0, out.data(), rows_b);
  }
  return out;
}

constexpr double kGram63[9] = {0x1.9999999999999p-5, 0x1.bbbbbbbbbbbbcp-4, 0x1.1111111111111p-7,
                               0x1.bbbbbbbbbbbbcp-4, 0x1.ccccccccccccdp-2, 0x1.bbbbbbbbbbbbcp-4,
                               0x1.1111111111111p-7, 0x1.bbbbbbbbbbbbcp-4, 0x1.9999999999999p-5};
constexpr double kGram42[4] = {0x1.5555555555554p-2, 0x1.5555555555558p-3, 0x1.5555555555558p-3,
                               0x1.5555555555554p-2};

}  // namespace

KnotGrid::KnotGrid(int64_t t0, int64_t h, int k, int64_t n) : t0_ns(t0), h_ns(h), order(k), num_ctrl(n) {
  if (order != 4 && order != 6) throw std::invalid_argument("spline order must be 4 or 6");
  if (!(1 <= h_ns && h_ns < (int64_t{1} << 31))) throw std::invalid_argument("knot spacing outside [1, 2^31)");
  if (num_ctrl < order) throw std::invalid_argument("need at least `order` control points");
}

KnotGrid KnotGrid::covering(int64_t t_begin_ns, int64_t t_end_ns, int64_t h_ns, int order) {
  // The Python's `//` raises ZeroDivisionError on h = 0; any other spacing outside [1, 2^31) fails the constructor's
  // check, made here before the division (floor_div needs h > 0).
  if (h_ns == 0) throw ZeroDivisionError("integer division or modulo by zero");
  if (!(1 <= h_ns && h_ns < (int64_t{1} << 31))) throw std::invalid_argument("knot spacing outside [1, 2^31)");
  const int64_t n_seg = std::max<int64_t>(1, -time::floor_div(t_begin_ns - t_end_ns, h_ns));
  return KnotGrid(t_begin_ns, h_ns, order, n_seg + order - 1);
}

void KnotGrid::locate(std::span<const int64_t> t_ns, std::vector<int64_t>& s, std::vector<double>& u) const {
  s.resize(t_ns.size());
  u.resize(t_ns.size());
  const int64_t nseg = num_segments();
  for (size_t i = 0; i < t_ns.size(); ++i) {
    const int64_t d = t_ns[i] - t0_ns;
    int64_t si = time::floor_div(d, h_ns);
    int64_t r = d - si * h_ns;
    if (si == nseg && r == 0) {
      si -= 1;
      r = h_ns;
    }
    if (si < 0 || si >= nseg) throw std::invalid_argument("sample time outside the knot grid's support");
    s[i] = si;
    u[i] = static_cast<double>(r) / static_cast<double>(h_ns);
  }
}

sparse::Matrix KnotGrid::design_matrix(std::span<const int64_t> t_ns) const {
  std::vector<int64_t> s;
  std::vector<double> u;
  locate(t_ns, s, u);
  std::vector<double> data = basis(u, order);
  const size_t n = s.size(), k = static_cast<size_t>(order);
  std::vector<int64_t> cols(n * k), ptr(n + 1);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < k; ++j) cols[i * k + j] = s[i] + static_cast<int64_t>(j);
  }
  for (size_t i = 0; i <= n; ++i) ptr[i] = static_cast<int64_t>(i * k);
  return sparse::csr(static_cast<int64_t>(n), num_ctrl, std::move(ptr), std::move(cols), std::move(data));
}

std::vector<int64_t> KnotGrid::greville_ns() const {
  std::vector<int64_t> out(static_cast<size_t>(num_ctrl));
  for (int64_t c = 0; c < num_ctrl; ++c) out[static_cast<size_t>(c)] = t0_ns + (c - order / 2 + 1) * h_ns;
  return out;
}

std::pair<int64_t, int64_t> KnotGrid::controls_touching(int64_t t_begin_ns, int64_t t_end_ns) const {
  const int64_t c0 = time::floor_div(t_begin_ns - t0_ns, h_ns);
  const int64_t c1 = -time::floor_div(t0_ns - t_end_ns, h_ns) + order - 2;
  return {std::max<int64_t>(0, c0), std::min<int64_t>(num_ctrl - 1, c1)};
}

std::pair<int64_t, int64_t> KnotGrid::still_interval(int64_t c0, int64_t c1) const {
  const int64_t s0 = c0, s1 = c1 - order + 1;
  if (s1 < s0) throw std::invalid_argument("fewer than K tied control points: the spline is not still anywhere");
  return {t0_ns + s0 * h_ns, t0_ns + (s1 + 1) * h_ns};
}

std::vector<double> basis(std::span<const double> u, int order) {
  const std::vector<double> m = blend_matrix(order, false);
  return matmul_bt(powers(u, order), u.size(), order, m.data(), order);
}

std::vector<double> cumulative_basis(std::span<const double> u, int order) {
  const std::vector<double> m = blend_matrix(order, true);
  return matmul_bt(powers(u, order), u.size(), order, m.data() + order, order - 1);
}

std::span<const double> segment_gram(int order, int m) {
  if (order == 6 && m == 3) return kGram63;
  if (order == 4 && m == 2) return kGram42;
  throw std::invalid_argument("segment_gram: only (K, m) = (6, 3) and (4, 2)");
}

sparse::Matrix difference_matrix(int64_t n, int order) {
  std::vector<int64_t> ptr, cols;
  std::vector<double> data;
  if (order == 0) {
    for (int64_t i = 0; i <= n; ++i) ptr.push_back(i);
    for (int64_t i = 0; i < n; ++i) {
      cols.push_back(i);
      data.push_back(1.0);
    }
    return sparse::csr(n, n, std::move(ptr), std::move(cols), std::move(data));
  }
  std::vector<double> coeff(static_cast<size_t>(order + 1));
  for (int k = 0; k <= order; ++k) coeff[static_cast<size_t>(k)] = static_cast<double>((k % 2 ? -1 : 1) * comb(order, k));
  std::reverse(coeff.begin(), coeff.end());
  const int64_t rows = n - order;
  for (int64_t r = 0; r < rows; ++r) {
    for (int k = 0; k <= order; ++k) {
      data.push_back(coeff[static_cast<size_t>(k)]);
      cols.push_back(r + k);
    }
  }
  for (int64_t r = 0; r <= rows; ++r) ptr.push_back(r * (order + 1));
  return sparse::csr(rows, n, std::move(ptr), std::move(cols), std::move(data));
}

sparse::Matrix difference_gram(int64_t num_ctrl, int order, int m) {
  const std::span<const double> g = segment_gram(order, m);
  const int k = order - m;
  const int64_t n_seg = num_ctrl - order + 1, rows = num_ctrl - m;
  std::vector<int64_t> r, c;
  std::vector<double> v;
  for (int64_t s = 0; s < n_seg; ++s) {
    for (int a = 0; a < k; ++a) {
      for (int b = 0; b < k; ++b) {
        r.push_back(s + a);
        c.push_back(s + b);
        v.push_back(g[static_cast<size_t>(a * k + b)]);
      }
    }
  }
  return sparse::coo_to_csr(rows, rows, r, c, v);
}

sparse::Matrix roughness_matrix(int64_t num_ctrl, int order, int m) {
  const sparse::Matrix d = difference_matrix(num_ctrl, m);
  return sparse::to_csr(sparse::matmul(sparse::matmul(sparse::transpose(d), difference_gram(num_ctrl, order, m)), d));
}

}  // namespace balla::scene::traj
