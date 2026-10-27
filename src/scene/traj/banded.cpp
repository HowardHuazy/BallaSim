// Contains C++ translations of LAPACK (dpbtf2), OpenBLAS (its dsyr/daxpy split) and NumPy (einsum's kernels), under their BSD licences:
// see THIRD_PARTY_NOTICES.md, "Ported code".
#include "scene/traj/banded.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "scene/blas.h"
#include "scene/errors.h"

namespace balla::scene::traj {

Band to_upper_band(const sparse::Matrix& a_in, int64_t u) {
  if (a_in.rows != a_in.cols) throw std::invalid_argument("to_upper_band: matrix must be square");
  sparse::Matrix a = sparse::to_csr(a_in);
  // Without duplicate entries sum_duplicates only reorders: fill the band directly (the values are the same).
  std::vector<int64_t> seen(static_cast<size_t>(a.cols), -1);
  bool duplicates = false;
  for (int64_t i = 0; i < a.rows && !duplicates; ++i) {
    for (int64_t jj = a.indptr[static_cast<size_t>(i)]; jj < a.indptr[static_cast<size_t>(i + 1)]; ++jj) {
      int64_t& s = seen[static_cast<size_t>(a.indices[static_cast<size_t>(jj)])];
      if (s == i) {
        duplicates = true;
        break;
      }
      s = i;
    }
  }
  if (duplicates) sparse::sum_duplicates(a);
  Band band(a.rows, u);
  for (int64_t i = 0; i < a.rows; ++i) {
    for (int64_t jj = a.indptr[static_cast<size_t>(i)]; jj < a.indptr[static_cast<size_t>(i + 1)]; ++jj) {
      const int64_t j = a.indices[static_cast<size_t>(jj)];
      if (j >= i && j - i <= u) band.at(i, j) = a.data[static_cast<size_t>(jj)];
    }
  }
  return band;
}

namespace {

// LAPACK's dpbtf2 (what dpbtrf runs for kd <= 64: ilaenv gives NB = 1) on the column-major band (ldab = kd + 1), with
// OpenBLAS's dscal (x * alpha) and dsyr (a column-wise daxpy of alpha x_q: its Haswell kernel fuses the first
// multiple of 16 elements, the C tail does not). Returns LAPACK's info.
int pbtf2_inline(int64_t n, int64_t kd, double* ab) {
  const int64_t ldab = kd + 1;
  auto at = [ab, kd, ldab](int64_t i, int64_t j) -> double& { return ab[(kd + i - j) + j * ldab]; };
  double x[65];
  for (int64_t j = 0; j < n; ++j) {
    double ajj = at(j, j);
    if (ajj <= 0.0) return static_cast<int>(j + 1);
    ajj = std::sqrt(ajj);
    at(j, j) = ajj;
    const int64_t kn = std::min(kd, n - 1 - j);
    if (kn <= 0) continue;
    const double r = 1.0 / ajj;
    for (int64_t p = 0; p < kn; ++p) {
      double& v = at(j, j + 1 + p);
      v = r * v;
      x[p] = v;
    }
    for (int64_t q = 0; q < kn; ++q) {
      if (x[q] == 0.0) continue;
      const double da = -1.0 * x[q];
      const int64_t fused = (q + 1) & -16;
      for (int64_t p = 0; p <= q; ++p) {
        double& y = at(j + 1 + p, j + 1 + q);
        y = p < fused ? std::fma(da, x[p], y) : y + da * x[p];
      }
    }
  }
  return 0;
}

// Whether pbtf2_inline is bitwise dpbtrf on this machine for this kd (checked once per kd on random SPD bands, from
// below kd + 1 rows to a few hundred, with zero band entries).
bool pbtf2_matches(int64_t kd) {
  static std::mutex m;
  static std::atomic<int> verdict[65] = {};   // 0 unknown, 1 yes, 2 no
  if (kd < 1 || kd > 64) return false;
  const int known = verdict[kd].load(std::memory_order_acquire);
  if (known != 0) return known == 1;
  const std::lock_guard<std::mutex> lock(m);
  if (verdict[kd].load(std::memory_order_relaxed) == 0) {
    std::mt19937_64 rng(static_cast<uint64_t>(kd));
    std::uniform_real_distribution<double> u(-0.5, 0.5);
    bool ok = true;
    for (int trial = 0; trial < 8 && ok; ++trial) {
      const int64_t n = trial < 2 ? 1 + trial * (kd / 2) : kd + 1 + trial * 37, ldab = kd + 1;
      std::vector<double> ab(static_cast<size_t>(ldab * n), 0.0);
      for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = std::max<int64_t>(0, j - kd); i < j; ++i) {
          ab[static_cast<size_t>((kd + i - j) + j * ldab)] = (i + j) % 5 == 0 ? 0.0 : u(rng) * std::exp(6 * u(rng));
        }
        ab[static_cast<size_t>(kd + j * ldab)] = 20.0 + 4.0 * static_cast<double>(kd) * std::exp(6 * u(rng));
      }
      std::vector<double> ref = ab;
      const char uplo = 'U';
      const int ni = static_cast<int>(n), kdi = static_cast<int>(kd), ld = static_cast<int>(ldab);
      int info = 0;
      dpbtrf_(&uplo, &ni, &kdi, ref.data(), &ld, &info, 1);
      const int mine = pbtf2_inline(n, kd, ab.data());
      ok = mine == info && std::memcmp(ab.data(), ref.data(), ab.size() * sizeof(double)) == 0;
    }
    verdict[kd].store(ok ? 1 : 2, std::memory_order_release);
  }
  return verdict[kd].load(std::memory_order_relaxed) == 1;
}

}  // namespace

BandedSPD::BandedSPD(const sparse::Matrix& a, int64_t u) : n_(a.rows), u_(u < 0 ? sparse::bandwidth(a) : u) {
  factorize(to_upper_band(a, u_).ab);   // to_upper_band checks a is square and u in range
}

BandedSPD::BandedSPD(const Band& band) : n_(band.n), u_(band.u) { factorize(band.ab); }

void BandedSPD::factorize(const std::vector<double>& ab) {
  for (const double x : ab) {
    if (!std::isfinite(x)) throw std::invalid_argument("array must not contain infs or NaNs");
  }
  const int64_t ldab = u_ + 1;
  std::vector<double> f = ab;
  int info = 0;
  if (pbtf2_matches(u_)) {
    info = pbtf2_inline(n_, u_, f.data());
  } else {
    const char uplo = 'U';
    const int n = static_cast<int>(n_), kd = static_cast<int>(u_), ld = static_cast<int>(ldab);
    dpbtrf_(&uplo, &n, &kd, f.data(), &ld, &info, 1);
  }
  if (info > 0) throw LinAlgError(std::to_string(info) + "-th leading minor not positive definite");
  if (info < 0) blas::lapack_error("dpbtrf", info);
  fortran_ = std::move(f);
}

const std::vector<double>& BandedSPD::factor() const {
  if (factor_.empty() && !fortran_.empty()) {
    const int64_t ldab = u_ + 1;
    factor_.resize(fortran_.size());
    for (int64_t r = 0; r < ldab; ++r) {
      for (int64_t c = 0; c < n_; ++c) factor_[static_cast<size_t>(r * n_ + c)] = fortran_[static_cast<size_t>(r + c * ldab)];
    }
  }
  return factor_;
}

std::vector<double> BandedSPD::solve(std::span<const double> b, int64_t k) const {
  if (static_cast<int64_t>(b.size()) != n_ * k) throw std::invalid_argument("shapes of cb and b are not compatible.");
  std::vector<double> x(b.size());
  for (int64_t i = 0; i < n_; ++i) {
    for (int64_t j = 0; j < k; ++j) x[static_cast<size_t>(i + j * n_)] = b[static_cast<size_t>(i * k + j)];
  }
  const char uplo = 'U';
  const int n = static_cast<int>(n_), kd = static_cast<int>(u_), nrhs = static_cast<int>(k),
            ld = static_cast<int>(u_ + 1), ldb = static_cast<int>(n_);
  int info = 0;
  dpbtrs_(&uplo, &n, &kd, &nrhs, fortran_.data(), &ld, x.data(), &ldb, &info, 1);
  if (info != 0) blas::lapack_error("dpbtrs", info);
  if (k == 1) return x;
  std::vector<double> out(b.size());
  for (int64_t i = 0; i < n_; ++i) {
    for (int64_t j = 0; j < k; ++j) out[static_cast<size_t>(i * k + j)] = x[static_cast<size_t>(i + j * n_)];
  }
  return out;
}

namespace {

// np.einsum's contig_contig_outstride0_two kernel on SSE2 (2 lanes, unrolled by 4, no FMA): the sum of a_k b_k.
double einsum_lanes(const double* a, const double* b, int64_t count) {
  double l0 = 0.0, l1 = 0.0;
  while (count >= 8) {
    for (int blk = 3; blk >= 0; --blk) {
      l0 = a[2 * blk] * b[2 * blk] + l0;
      l1 = a[2 * blk + 1] * b[2 * blk + 1] + l1;
    }
    a += 8;
    b += 8;
    count -= 8;
  }
  while (count > 0) {
    l0 = a[0] * b[0] + l0;
    l1 = (count > 1 ? a[1] * b[1] : 0.0 * 0.0) + l1;
    a += 2;
    b += 2;
    count -= 2;
  }
  return l0 + l1;
}

}  // namespace

std::vector<double> selected_inverse_band(const std::vector<const std::vector<double>*>& factors, int64_t u, int64_t n) {
  const auto lc = static_cast<int64_t>(factors.size());
  if (u < 0 || n < 0) throw std::invalid_argument("selected_inverse_band: negative shape");
  const int64_t ub = u + 1;
  for (const std::vector<double>* f : factors) {
    if (static_cast<int64_t>(f->size()) != ub * n) throw std::invalid_argument("selected_inverse_band: factor shapes differ");
  }
  if (lc == 0 || n == 0) return {};   // the Python's empty (L, u + 1, n) result
  // inv_d[i][l]; urt[i][k-1][l] = U_l[i, i + k] (0 past the matrix): the factors' rows, l fastest.
  std::vector<double> inv_d(static_cast<size_t>(n * lc)), urt(static_cast<size_t>(n * u * lc));
  for (int64_t l = 0; l < lc; ++l) {
    const std::vector<double>& f = *factors[static_cast<size_t>(l)];
    for (int64_t i = 0; i < n; ++i) {
      inv_d[static_cast<size_t>(i * lc + l)] = 1.0 / f[static_cast<size_t>(u * n + i)];
      for (int64_t k = 1; k <= u; ++k) {
        urt[static_cast<size_t>((i * u + k - 1) * lc + l)] =
            i + k < n ? f[static_cast<size_t>((u - k) * n + i + k)] : 0.0;
      }
    }
  }
  // s[row][d][l] = Sigma_l[row, row + d]; rows past n are zero.
  std::vector<double> s(static_cast<size_t>((n + u) * ub * lc), 0.0);
  // Offsets of Sigma[i + k, i + j] = s[i + min(k, j)][|k - j|] from row i, per (k, j) (0-based k, j).
  std::vector<int64_t> rel(static_cast<size_t>(u * u));
  for (int64_t k = 0; k < u; ++k) {
    for (int64_t j = 0; j < u; ++j) {
      rel[static_cast<size_t>(k * u + j)] = ((std::min(k, j) + 1) * ub + std::abs(k - j)) * lc;
    }
  }
  std::vector<double> acc(static_cast<size_t>(lc)), off(static_cast<size_t>(u * lc)), ur_l(static_cast<size_t>(u)),
      off_l(static_cast<size_t>(u));
  for (int64_t i = n - 1; i >= 0; --i) {
    const double* ur = urt.data() + i * u * lc;
    const double* id = inv_d.data() + i * lc;
    double* srow = s.data() + i * ub * lc;
    // einsum("lk,kjl->jl") * -inv_d: per (j, l) a sum over k from 0 (each l its own accumulator).
    for (int64_t j = 0; j < u; ++j) {
      std::fill(acc.begin(), acc.end(), 0.0);
      for (int64_t k = 0; k < u; ++k) {
        const double* sk = srow + rel[static_cast<size_t>(k * u + j)];
        const double* urk = ur + k * lc;
        for (int64_t l = 0; l < lc; ++l) acc[static_cast<size_t>(l)] += urk[l] * sk[l];
      }
      for (int64_t l = 0; l < lc; ++l) off[static_cast<size_t>(j * lc + l)] = (0.0 + acc[static_cast<size_t>(l)]) * -id[l];
    }
    for (int64_t j = 0; j < u; ++j) {
      for (int64_t l = 0; l < lc; ++l) srow[(1 + j) * lc + l] = off[static_cast<size_t>(j * lc + l)];
    }
    // einsum("lk,kl->l"): several factors sum over k from 0; one factor runs NumPy's SSE2 two-lane kernel.
    if (lc > 1) {
      std::fill(acc.begin(), acc.end(), 0.0);
      for (int64_t k = 0; k < u; ++k) {
        const double* urk = ur + k * lc;
        const double* ok = off.data() + k * lc;
        for (int64_t l = 0; l < lc; ++l) acc[static_cast<size_t>(l)] += urk[l] * ok[l];
      }
      for (int64_t l = 0; l < lc; ++l) srow[l] = (id[l] - (0.0 + acc[static_cast<size_t>(l)])) * id[l];
    } else {
      for (int64_t k = 0; k < u; ++k) {
        ur_l[static_cast<size_t>(k)] = ur[k];
        off_l[static_cast<size_t>(k)] = off[static_cast<size_t>(k)];
      }
      const double e = 0.0 + einsum_lanes(ur_l.data(), off_l.data(), u);
      srow[0] = (id[0] - e) * id[0];
    }
  }
  std::vector<double> out(static_cast<size_t>(lc * ub * n));
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t d = 0; d < ub; ++d) {
      for (int64_t l = 0; l < lc; ++l) {
        out[static_cast<size_t>((l * ub + d) * n + i)] = s[static_cast<size_t>((i * ub + d) * lc + l)];
      }
    }
  }
  return out;
}

namespace {

// np.einsum("li,i->l", a, b) for one l: with several l the inner loop runs over l and each output accumulates
// a_i b_i + out in i order; with one l it runs over i (the strided outstride0 kernel, no SIMD), the iterator handing
// it NumPy's buffersize of 8192 elements at a time: one accumulator per chunk, added to the output chunk by chunk.
double einsum_li_i(const double* a, const double* b, int64_t b_stride, int64_t count, bool single) {
  if (!single) {
    double out = 0.0;
    for (int64_t i = 0; i < count; ++i) out = a[i] * b[i * b_stride] + out;
    return out;
  }
  constexpr int64_t kBuffer = 8192;
  double out = 0.0;
  for (int64_t start = 0; start < count; start += kBuffer) {
    double acc = 0.0;
    const int64_t end = std::min(count, start + kBuffer);
    for (int64_t i = start; i < end; ++i) acc += a[i] * b[i * b_stride];
    out = out + acc;
  }
  return out;
}

}  // namespace

std::vector<double> trace_of_product(const std::vector<double>& s_band, int64_t lcount, const Band& m) {
  const int64_t u = m.u, n = m.n, ub = u + 1;
  if (lcount < 0 || static_cast<int64_t>(s_band.size()) != lcount * ub * n) {
    throw std::invalid_argument("trace_of_product: band shapes differ");
  }
  // The Python's s_band[:, d, :n - d] counts a negative stop from the end: for n < d < 2n - 1 its length (2n - d > 1)
  // differs from mb[u - d, d:]'s (0) and einsum cannot broadcast them.
  if (u > n && n >= 3) throw std::invalid_argument("operands could not be broadcast together with remapped shapes");
  std::vector<double> total(static_cast<size_t>(lcount), 0.0);
  const bool single = lcount == 1;
  for (int64_t l = 0; l < lcount; ++l) {
    const double* s0 = s_band.data() + (l * ub + 0) * n;
    total[static_cast<size_t>(l)] = einsum_li_i(s0, m.ab.data() + u, ub, n, single);
  }
  for (int64_t d = 1; d <= u; ++d) {
    for (int64_t l = 0; l < lcount; ++l) {
      const double* sd = s_band.data() + (l * ub + d) * n;
      const double e = einsum_li_i(sd, m.ab.data() + (u - d) + d * ub, ub, n - d, single);
      total[static_cast<size_t>(l)] += 2.0 * e;
    }
  }
  return total;
}

}  // namespace balla::scene::traj
