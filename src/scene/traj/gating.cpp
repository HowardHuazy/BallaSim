#include "scene/traj/gating.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "scene/npo.h"

namespace balla::scene::traj {

namespace {

using rot::Q;
using rot::V;

constexpr double kTukeyC = 4.685;
constexpr double kInf = std::numeric_limits<double>::infinity();

// _masked_median over one row of m values: sort with the invalid as +inf, the mean of the middle valid ones.
double masked_median(const double* x, const uint8_t* valid, int64_t m, std::vector<double>& scratch) {
  scratch.resize(static_cast<size_t>(m));
  int64_t c = 0;
  for (int64_t k = 0; k < m; ++k) {
    scratch[static_cast<size_t>(k)] = valid[k] ? x[k] : kInf;
    c += valid[k] ? 1 : 0;
  }
  std::sort(scratch.begin(), scratch.end(), [](double a, double b) {
    return a < b || (!std::isnan(a) && std::isnan(b));
  });
  const double lo = scratch[static_cast<size_t>(std::max<int64_t>(c - 1, 0) / 2)];
  const double hi = scratch[static_cast<size_t>(std::min<int64_t>(c / 2, m - 1))];
  return c > 0 ? 0.5 * (lo + hi) : 0.0;
}

// _tukey_quadratics: per row n, robust y ~ c0 + c1 x + c2 x^2 over the m window slots; returns c0 (n, 3).
std::vector<V> tukey_quadratics(const std::vector<double>& x, const std::vector<V>& y, const std::vector<uint8_t>& valid,
                                int64_t n, int64_t m, int iterations = 10) {
  std::vector<double> a(static_cast<size_t>(n * m * 3)), w(static_cast<size_t>(n * m)), r(static_cast<size_t>(n * m));
  for (int64_t i = 0; i < n * m; ++i) {
    const double xi = x[static_cast<size_t>(i)];
    a[static_cast<size_t>(3 * i)] = 1.0;
    a[static_cast<size_t>(3 * i + 1)] = xi;
    a[static_cast<size_t>(3 * i + 2)] = xi * xi;
    w[static_cast<size_t>(i)] = valid[static_cast<size_t>(i)] ? 1.0 : 0.0;
  }
  std::vector<uint8_t> frozen(static_cast<size_t>(n), 0);
  std::vector<double> coef(static_cast<size_t>(n * 9), 0.0), scratch;
  auto ycomp = [&y](int64_t i, int k) {
    const V& v = y[static_cast<size_t>(i)];
    return k == 0 ? v.x : k == 1 ? v.y : v.z;
  };
  // One weighted solve of row i with weights wt (m values): coef[i] = solve(A^T W A + 1e-18 I, A^T W y).
  auto fit_row = [&](int64_t i, const double* wt, double* out) {
    double ata[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, aty[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t k = 0; k < m; ++k) {
      const int64_t e = i * m + k;
      const double* ar = &a[static_cast<size_t>(3 * e)];
      for (int ii = 0; ii < 3; ++ii) {
        const double aw = ar[ii] * wt[k];
        for (int j = 0; j < 3; ++j) ata[3 * ii + j] = ar[j] * aw + ata[3 * ii + j];
        for (int j = 0; j < 3; ++j) aty[3 * ii + j] = ycomp(e, j) * aw + aty[3 * ii + j];
      }
    }
    for (int ii = 0; ii < 3; ++ii) {
      for (int j = 0; j < 3; ++j) ata[3 * ii + j] = ata[3 * ii + j] + 1e-18 * (ii == j ? 1.0 : 0.0);
    }
    npo::solve3(ata, aty);
    for (int k = 0; k < 9; ++k) out[k] = aty[k];
  };
  std::vector<double> w_new(static_cast<size_t>(m));
  for (int it = 0; it < iterations; ++it) {
    for (int64_t i = 0; i < n; ++i) fit_row(i, &w[static_cast<size_t>(i * m)], &coef[static_cast<size_t>(9 * i)]);
    for (int64_t i = 0; i < n; ++i) {
      const double* c = &coef[static_cast<size_t>(9 * i)];
      for (int64_t k = 0; k < m; ++k) {
        const int64_t e = i * m + k;
        const double* ar = &a[static_cast<size_t>(3 * e)];
        double d[3];
        for (int j = 0; j < 3; ++j) {
          double acc = 0.0;
          for (int ii = 0; ii < 3; ++ii) acc = c[3 * ii + j] * ar[ii] + acc;
          d[j] = ycomp(e, j) - acc;
        }
        r[static_cast<size_t>(e)] = std::sqrt(0.0 + (((0.0 + d[0] * d[0]) + d[1] * d[1]) + d[2] * d[2]));
      }
    }
    for (int64_t i = 0; i < n; ++i) {
      const auto row = static_cast<size_t>(i * m);
      const double s = 1.4826 * masked_median(&r[row], &valid[row], m, scratch) + 1e-12;
      const double den = kTukeyC * s;
      int64_t pos = 0;
      for (int64_t k = 0; k < m; ++k) {
        const int64_t e = i * m + k;
        const double uu = r[static_cast<size_t>(e)] / den;
        const double t = 1.0 - uu * uu;
        w_new[static_cast<size_t>(k)] = (valid[static_cast<size_t>(e)] && uu < 1.0) ? t * t : 0.0;
        pos += w_new[static_cast<size_t>(k)] > 0 ? 1 : 0;
      }
      if (pos < 3) frozen[static_cast<size_t>(i)] = 1;
      for (int64_t k = 0; k < m; ++k) {
        const int64_t e = i * m + k;
        const double vf = valid[static_cast<size_t>(e)] ? 1.0 : 0.0;
        w[static_cast<size_t>(e)] = frozen[static_cast<size_t>(i)] ? vf : w_new[static_cast<size_t>(k)];
      }
    }
  }
  std::vector<double> vw(static_cast<size_t>(m));
  for (int64_t i = 0; i < n; ++i) {
    if (!frozen[static_cast<size_t>(i)]) continue;
    for (int64_t k = 0; k < m; ++k) vw[static_cast<size_t>(k)] = valid[static_cast<size_t>(i * m + k)] ? 1.0 : 0.0;
    fit_row(i, vw.data(), &coef[static_cast<size_t>(9 * i)]);
  }
  std::vector<V> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const double* c = &coef[static_cast<size_t>(9 * i)];
    out[static_cast<size_t>(i)] = {c[0], c[1], c[2]};
  }
  return out;
}

struct Loo {
  std::vector<double> err_rot, err_pos;
  std::vector<uint8_t> enough;
};

// _loo_errors: leave-one-out prediction errors of `targets` from their +-half nearest neighbours in `pool`.
Loo loo_errors(std::span<const int64_t> t, std::span<const Q> q, std::span<const V> p, const std::vector<int64_t>& targets,
               const std::vector<int64_t>& pool, int64_t half) {
  const auto n = static_cast<int64_t>(targets.size());
  const int64_t m = 2 * half;
  const auto np = static_cast<int64_t>(pool.size());
  std::vector<int64_t> idx(static_cast<size_t>(n * m));
  std::vector<uint8_t> valid(static_cast<size_t>(n * m));
  Loo out{std::vector<double>(static_cast<size_t>(n)), std::vector<double>(static_cast<size_t>(n)),
          std::vector<uint8_t>(static_cast<size_t>(n))};
  for (int64_t i = 0; i < n; ++i) {
    const int64_t tg = targets[static_cast<size_t>(i)];
    const auto it = std::lower_bound(pool.begin(), pool.end(), tg);
    const int64_t pos = it - pool.begin();
    const int64_t own = (it != pool.end() && *it == tg) ? 1 : 0;
    int64_t count = 0;
    for (int64_t k = 0; k < m; ++k) {
      const int64_t slot = k < half ? pos - (half - k) : pos + own + (k - half);
      const bool v = slot >= 0 && slot < np;
      valid[static_cast<size_t>(i * m + k)] = v ? 1 : 0;
      idx[static_cast<size_t>(i * m + k)] = pool[static_cast<size_t>(std::clamp<int64_t>(slot, 0, np - 1))];
      count += v ? 1 : 0;
    }
    out.enough[static_cast<size_t>(i)] = count >= 4 ? 1 : 0;
  }
  std::vector<Q> ref(static_cast<size_t>(n));
  std::vector<double> dist(static_cast<size_t>(m * m)), med(static_cast<size_t>(m)), scratch;
  std::vector<uint8_t> vrow(static_cast<size_t>(m));
  std::vector<V> phi(static_cast<size_t>(n * m)), dp(static_cast<size_t>(n * m));
  std::vector<double> x(static_cast<size_t>(n * m));
  for (int64_t i = 0; i < n; ++i) {
    const int64_t* ix = &idx[static_cast<size_t>(i * m)];
    const uint8_t* va = &valid[static_cast<size_t>(i * m)];
    for (int64_t a = 0; a < m; ++a) {
      for (int64_t b = 0; b < m; ++b) {
        dist[static_cast<size_t>(a * m + b)] = rot::rotation_angle(q[static_cast<size_t>(ix[a])], q[static_cast<size_t>(ix[b])]);
      }
    }
    for (int64_t a = 0; a < m; ++a) {
      med[static_cast<size_t>(a)] = va[a] ? masked_median(&dist[static_cast<size_t>(a * m)], va, m, scratch) : kInf;
    }
    const int64_t best = npo::argmin(med);
    ref[static_cast<size_t>(i)] = q[static_cast<size_t>(ix[best])];
    const Q rc = rot::quat_conjugate(ref[static_cast<size_t>(i)]);
    const int64_t tg = targets[static_cast<size_t>(i)];
    for (int64_t k = 0; k < m; ++k) {
      const size_t e = static_cast<size_t>(i * m + k);
      phi[e] = rot::so3_log(rot::quat_multiply(rc, q[static_cast<size_t>(ix[k])]));
      x[e] = static_cast<double>(t[static_cast<size_t>(ix[k])] - t[static_cast<size_t>(tg)]) * 1e-9;
      dp[e] = p[static_cast<size_t>(ix[k])] - p[static_cast<size_t>(tg)];
    }
  }
  const std::vector<V> c_rot = tukey_quadratics(x, phi, valid, n, m);
  const std::vector<V> c_pos = tukey_quadratics(x, dp, valid, n, m);
  for (int64_t i = 0; i < n; ++i) {
    const size_t s = static_cast<size_t>(i);
    const Q pred = rot::quat_multiply(ref[s], rot::so3_exp(c_rot[s]));
    out.err_rot[s] = out.enough[s] ? rot::rotation_angle(pred, q[static_cast<size_t>(targets[s])]) : 0.0;
    out.err_pos[s] = out.enough[s] ? rot::norm_axis(c_pos[s]) : 0.0;
  }
  return out;
}

double py_max(double a, double b) { return b > a ? b : a; }

}  // namespace

std::vector<uint8_t> minority_branches(std::span<const int64_t> t, std::span<const Q> q, std::span<const uint8_t> keep,
                                       double jump_rad, double rate_factor) {
  std::vector<int64_t> kept;
  for (size_t i = 0; i < keep.size(); ++i) {
    if (keep[i]) kept.push_back(static_cast<int64_t>(i));
  }
  std::vector<uint8_t> out(t.size(), 0);
  if (kept.size() < 3) return out;
  const size_t nk = kept.size();
  std::vector<double> steps(nk - 1), dt_s(nk - 1), rate(nk - 1);
  for (size_t i = 0; i + 1 < nk; ++i) {
    steps[i] = rot::rotation_angle(q[static_cast<size_t>(kept[i])], q[static_cast<size_t>(kept[i + 1])]);
    dt_s[i] = static_cast<double>(t[static_cast<size_t>(kept[i + 1])] - t[static_cast<size_t>(kept[i])]) * 1e-9;
    rate[i] = steps[i] / dt_s[i];
  }
  const double rate95 = npo::percentile(rate, 95.0);
  const double alias = 0.9 * std::numbers::pi;
  const double fr = rate_factor * rate95;
  std::vector<size_t> starts{0};
  for (size_t i = 0; i + 1 < nk; ++i) {
    const bool jump = (steps[i] > jump_rad && steps[i] > fr * dt_s[i]) || steps[i] > alias;
    if (jump) starts.push_back(i + 1);
  }
  if (starts.size() == 1) return out;
  std::vector<Q> latest;
  std::vector<int64_t> counts;
  std::vector<int64_t> label(nk);
  for (size_t r = 0; r < starts.size(); ++r) {
    const size_t a = starts[r], b = r + 1 < starts.size() ? starts[r + 1] : nk;
    int64_t j = -1;
    double dj = 0;
    if (!latest.empty()) {
      std::vector<double> d(latest.size());
      for (size_t k = 0; k < latest.size(); ++k) d[k] = rot::rotation_angle(latest[k], q[static_cast<size_t>(kept[a])]);
      j = npo::argmin(d);
      dj = d[static_cast<size_t>(j)];
    }
    if (j < 0 || dj > jump_rad) {
      j = static_cast<int64_t>(latest.size());
      latest.push_back(q[static_cast<size_t>(kept[b - 1])]);
      counts.push_back(0);
    }
    latest[static_cast<size_t>(j)] = q[static_cast<size_t>(kept[b - 1])];
    counts[static_cast<size_t>(j)] += static_cast<int64_t>(b - a);
    for (size_t k = a; k < b; ++k) label[k] = j;
  }
  const int64_t best = std::max_element(counts.begin(), counts.end()) - counts.begin();
  for (size_t k = 0; k < nk; ++k) {
    if (label[k] != best) out[static_cast<size_t>(kept[k])] = 1;
  }
  return out;
}

GateResult gate_poses(std::span<const int64_t> t, std::span<const Q> q, std::span<const V> p, const GateOptions& o) {
  const size_t n = t.size();
  if (q.size() != n || p.size() != n) throw std::invalid_argument("gate_poses: t, q and p differ in length");
  GateResult g;
  if (n < 5) {
    g.keep.assign(n, 1);
    g.err_rot.assign(n, 0.0);
    g.err_pos.assign(n, 0.0);
    g.threshold_rot = g.threshold_pos = kInf;
    return g;
  }
  std::vector<double> rates(n - 1);
  for (size_t i = 0; i + 1 < n; ++i) {
    rates[i] = rot::rotation_angle(q[i], q[i + 1]) / (static_cast<double>(t[i + 1] - t[i]) * 1e-9);
  }
  const double w95 = npo::percentile(rates, 95.0);
  const double mx = 1e-6 > w95 ? 1e-6 : w95;
  const double fl = std::floor(std::numbers::pi / (((4.0 * mx) * o.period_ns) * 1e-9));
  if (std::isnan(fl)) throw std::invalid_argument("cannot convert float NaN to integer (gating window)");
  const auto half = static_cast<int64_t>(std::clamp(fl, 3.0, 5.0));
  std::vector<int64_t> everyone(n);
  for (size_t i = 0; i < n; ++i) everyone[i] = static_cast<int64_t>(i);
  Loo l1 = loo_errors(t, q, p, everyone, everyone, half);
  std::vector<double> er, ep;
  for (size_t i = 0; i < n; ++i) {
    if (l1.enough[i]) {
      er.push_back(l1.err_rot[i]);
      ep.push_back(l1.err_pos[i]);
    }
  }
  const double sr = er.empty() ? 0.0 : 1.4826 * npo::median(er);
  const double spos = er.empty() ? 0.0 : 1.4826 * npo::median(ep);
  const double thr_r = py_max(o.sigma_k * sr, o.rot_floor_rad);
  const double thr_p = py_max(o.sigma_k * spos, o.pos_floor_m);
  std::vector<double> err_rot = l1.err_rot, err_pos = l1.err_pos;
  std::vector<uint8_t> bad(n);
  for (size_t i = 0; i < n; ++i) bad[i] = (err_rot[i] > thr_r || err_pos[i] > thr_p) ? 1 : 0;
  auto rejudge = [&](const std::vector<uint8_t>& exclude) {
    std::vector<int64_t> flagged, pool;
    for (size_t i = 0; i < n; ++i) {
      if (bad[i]) flagged.push_back(static_cast<int64_t>(i));
      if (!exclude[i]) pool.push_back(static_cast<int64_t>(i));
    }
    const Loo l2 = loo_errors(t, q, p, flagged, pool, half);
    for (size_t k = 0; k < flagged.size(); ++k) {
      const auto i = static_cast<size_t>(flagged[k]);
      err_rot[i] = l2.err_rot[k];
      err_pos[i] = l2.err_pos[k];
      const bool b2 = (l2.enough[k] && (l2.err_rot[k] > thr_r || l2.err_pos[k] > thr_p)) || (!l2.enough[k] && bad[i]);
      bad[i] = b2 ? 1 : 0;
    }
  };
  const auto num_bad = static_cast<size_t>(std::count(bad.begin(), bad.end(), 1));
  if (num_bad > 0 && n - num_bad >= 5) rejudge(bad);
  std::vector<uint8_t> good(n);
  for (size_t i = 0; i < n; ++i) good[i] = bad[i] ? 0 : 1;
  const std::vector<uint8_t> branch = minority_branches(t, q, good, o.branch_jump_rad, o.branch_rate_factor);
  const bool any_branch = std::count(branch.begin(), branch.end(), 1) > 0;
  const bool any_bad = std::count(bad.begin(), bad.end(), 1) > 0;
  if (any_branch && any_bad) {
    std::vector<uint8_t> excl(n);
    for (size_t i = 0; i < n; ++i) excl[i] = (bad[i] || branch[i]) ? 1 : 0;
    rejudge(excl);
  }
  g.keep.resize(n);
  for (size_t i = 0; i < n; ++i) g.keep[i] = (bad[i] || branch[i]) ? 0 : 1;
  g.err_rot = std::move(err_rot);
  g.err_pos = std::move(err_pos);
  g.threshold_rot = thr_r;
  g.threshold_pos = thr_p;
  g.half_window = half;
  g.sigma_rot = sr;
  g.sigma_pos = spos;
  g.num_branch_rejected = std::count(branch.begin(), branch.end(), 1);
  return g;
}

}  // namespace balla::scene::traj
