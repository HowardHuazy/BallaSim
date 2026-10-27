// Breakpoint detection's pure pieces (detect.h; blender.md §4.7.5).
#include "scene/motion/detect.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

#include "core/math/so3.h"

namespace balla::scene::motion {

namespace {

constexpr size_t kSide = 6;          // one-sided cubics on 6 samples per side (§4.7.5)
constexpr double kNoiseFrac = 1e-2;  // the standard error's floor, eps/100 (the prototype's 1e-4 px at eps = 0.01)
constexpr double kSpikeRatio = 8.0;  // a spacing spike: shorter than the neighbourhood's median / 8

using Y = std::array<double, 6>;

Y sub(const Y& a, const Y& b) {
  Y r;
  for (size_t k = 0; k < 6; ++k) r[k] = a[k] - b[k];
  return r;
}
double dot(const Y& a, const Y& b) {
  double s = 0.0;
  for (size_t k = 0; k < 6; ++k) s += a[k] * b[k];
  return s;
}
double norm(const Y& a) { return std::sqrt(dot(a, a)); }

// The signal in px relative to a reference pose (detect.h).
struct Signal {
  double w_pos, w_rot;
  rot::Q q_ref;
  rot::V p_ref;
  Y operator()(const rot::Q& q, const rot::V& p) const {
    const rot::V d = p - p_ref, phi = so3::log(conjugate(q_ref) * q);
    return {w_pos * d.x, w_pos * d.y, w_pos * d.z, w_rot * phi.x, w_rot * phi.y, w_rot * phi.z};
  }
};

// y(x) = sum_k c[k] x^k per component (x in s), a least-squares fit.
struct Poly {
  int deg = 0;
  std::array<Y, 4> c{};
  double s2 = 0.0;              // the residual variance, summed over the components
  double v00 = 0.0, v11 = 0.0;  // the diagonal of (V^T V)^-1 at c0 and c1
};

Y value(const Poly& f, double x) {
  Y r{};
  for (int k = f.deg; k >= 0; --k) {
    for (size_t j = 0; j < 6; ++j) r[j] = r[j] * x + f.c[static_cast<size_t>(k)][j];
  }
  return r;
}
Y slope(const Poly& f, double x) {
  Y r{};
  for (int k = f.deg; k >= 1; --k) {
    for (size_t j = 0; j < 6; ++j) r[j] = r[j] * x + k * f.c[static_cast<size_t>(k)][j];
  }
  return r;
}

// Least squares of degree `deg` through (x, y) by Householder QR of the Vandermonde matrix in x / max|x|.
std::optional<Poly> polyfit(const std::vector<double>& x, const std::vector<Y>& y, int deg) {
  const size_t m = x.size(), k = static_cast<size_t>(deg) + 1;
  if (deg < 0 || deg > 3 || m < k) return std::nullopt;
  double s = 0.0;
  for (const double v : x) s = std::max(s, std::fabs(v));
  if (!(s > 0.0)) return std::nullopt;
  std::vector<std::array<double, 4>> a(m);
  std::vector<Y> b = y;
  for (size_t r = 0; r < m; ++r) {
    double xp = 1.0;
    for (size_t j = 0; j < k; ++j) {
      a[r][j] = xp;
      xp *= x[r] / s;
    }
  }
  for (size_t j = 0; j < k; ++j) {
    double nrm = 0.0;
    for (size_t r = j; r < m; ++r) nrm += a[r][j] * a[r][j];
    nrm = std::sqrt(nrm);
    if (!(nrm > 0.0)) return std::nullopt;
    const double alpha = a[j][j] > 0.0 ? -nrm : nrm;
    std::vector<double> v(m - j);
    for (size_t r = j; r < m; ++r) v[r - j] = a[r][j];
    v[0] -= alpha;
    double vv = 0.0;
    for (const double e : v) vv += e * e;
    if (vv > 0.0) {
      for (size_t l = j; l < k; ++l) {
        double f = 0.0;
        for (size_t r = j; r < m; ++r) f += v[r - j] * a[r][l];
        f *= 2.0 / vv;
        for (size_t r = j; r < m; ++r) a[r][l] -= f * v[r - j];
      }
      for (size_t comp = 0; comp < 6; ++comp) {
        double f = 0.0;
        for (size_t r = j; r < m; ++r) f += v[r - j] * b[r][comp];
        f *= 2.0 / vv;
        for (size_t r = j; r < m; ++r) b[r][comp] -= f * v[r - j];
      }
    }
  }
  for (size_t j = 0; j < k; ++j) {
    if (!(std::fabs(a[j][j]) > 1e-12 * std::fabs(a[0][0]))) return std::nullopt;
  }
  Poly f;
  f.deg = deg;
  for (size_t comp = 0; comp < 6; ++comp) {  // R c = (Q^T y)[0:k]
    for (size_t j = k; j-- > 0;) {
      double v = b[j][comp];
      for (size_t l = j + 1; l < k; ++l) v -= a[j][l] * f.c[l][comp];
      f.c[j][comp] = v / a[j][j];
    }
    double rss = 0.0;
    for (size_t r = k; r < m; ++r) rss += b[r][comp] * b[r][comp];
    f.s2 += rss / static_cast<double>(std::max<size_t>(m - k, 1));
  }
  // (V^T V)^-1 = R^-1 R^-T: its diagonal from R^-1's rows
  std::array<std::array<double, 4>, 4> ri{};
  for (size_t j = k; j-- > 0;) {
    ri[j][j] = 1.0 / a[j][j];
    for (size_t l = j + 1; l < k; ++l) {
      double v = 0.0;
      for (size_t t = j + 1; t <= l; ++t) v += a[j][t] * ri[t][l];
      ri[j][l] = -v / a[j][j];
    }
  }
  auto vdiag = [&](size_t j) {
    double v = 0.0;
    for (size_t l = j; l < k; ++l) v += ri[j][l] * ri[j][l];
    return v;
  };
  double sp = 1.0;
  for (size_t j = 0; j < k; ++j) {
    for (size_t comp = 0; comp < 6; ++comp) f.c[j][comp] /= sp;
    sp *= s;
  }
  f.v00 = vdiag(0);
  f.v11 = k > 1 ? vdiag(1) / (s * s) : 0.0;
  return f;
}

// A one-sided fit of fit[lo, hi) about t_split: degree min(3, n - 2), at least 3 samples.
struct Side {
  std::optional<Poly> poly;
  double f1 = 0.0;
};
Side one_sided(const Track& fit, size_t lo, size_t hi, int64_t t_split, const Signal& sig) {
  Side out;
  if (hi < lo + 3) return out;
  std::vector<double> x;
  std::vector<Y> y;
  for (size_t r = lo; r < hi; ++r) {
    x.push_back(static_cast<double>(fit.t_ns[r] - t_split) * 1e-9);
    y.push_back(sig(fit.q[r], fit.p[r]));
    out.f1 = std::max(out.f1, fit.floor1[r]);
  }
  out.poly = polyfit(x, y, std::min<int>(3, static_cast<int>(hi - lo) - 2));
  return out;
}

// The rate of the pose distance (detect.h pose_px) along a slope of the signal: w_pos |dp/dt| + w_rot |dphi/dt|.
double pose_speed(const Y& v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + std::sqrt(v[3] * v[3] + v[4] * v[4] + v[5] * v[5]);
}

bool stronger(const Breakpoint& a, const Breakpoint& b) {
  const bool sa = a.cls == Continuity::kStep, sb = b.cls == Continuity::kStep;
  if (sa != sb) return sa;
  return sa ? a.jump_px > b.jump_px : a.dv_pxps > b.dv_pxps;
}

}  // namespace

std::vector<int64_t> candidates(const SampleSet& set, std::span<const DeclaredKey> keys, double eps_px) {
  const Track& f = set.fit;
  const size_t n = f.size();
  std::vector<int64_t> out;
  for (const auto& [a, b] : set.unresolved) out.push_back(a + (b - a) / 2);
  if (n >= 2) {  // spacing spikes
    std::vector<int64_t> d(n - 1);
    for (size_t i = 0; i + 1 < n; ++i) d[i] = f.t_ns[i + 1] - f.t_ns[i];
    for (size_t i = 4; i + 4 < d.size(); ++i) {
      std::array<int64_t, 6> nb{d[i - 4], d[i - 3], d[i - 2], d[i + 2], d[i + 3], d[i + 4]};
      std::sort(nb.begin(), nb.end());
      const double med = 0.5 * (static_cast<double>(nb[2]) + static_cast<double>(nb[3]));
      if (static_cast<double>(d[i]) * kSpikeRatio < med && d[i] <= d[i - 1] && d[i] <= d[i + 1]) {
        out.push_back(f.t_ns[i] + d[i] / 2);
      }
    }
  }
  if (n >= 5) {  // the node scan
    std::vector<double> dv(n, 0.0), kap(n, 0.0);
    for (size_t i = 2; i + 2 < n; ++i) {
      const Signal sig{f.w_pos[i], f.w_rot[i], f.q[i], f.p[i]};
      double x[5];
      Y y[5];
      for (size_t k = 0; k < 5; ++k) {
        x[k] = static_cast<double>(f.t_ns[i + k - 2] - f.t_ns[i]) * 1e-9;
        y[k] = sig(f.q[i + k - 2], f.p[i + k - 2]);
      }
      // the slope at the node of the quadratic through the node and its two left (right) neighbours (y = 0 there)
      const double l0 = -x[1] / ((x[0] - x[1]) * x[0]), l1 = -x[0] / ((x[1] - x[0]) * x[1]);
      const double r1 = -x[4] / (x[3] * (x[3] - x[4])), r2 = -x[3] / (x[4] * (x[4] - x[3]));
      Y dd;
      for (size_t j = 0; j < 6; ++j) dd[j] = (r1 * y[3][j] + r2 * y[4][j]) - (l0 * y[0][j] + l1 * y[1][j]);
      dv[i] = norm(dd);
      kap[i] = dv[i] * std::min(x[3], -x[1]);
    }
    for (size_t i = 2; i + 2 < n; ++i) {
      const double local = *std::max_element(dv.begin() + static_cast<ptrdiff_t>(i - 2),
                                             dv.begin() + static_cast<ptrdiff_t>(std::min(i + 3, n)));
      if (kap[i] > eps_px / 4.0 && dv[i] >= local) out.push_back(f.t_ns[i]);
    }
  }
  if (n >= 2) {
    for (const DeclaredKey& k : keys) {
      if (k.hint != Continuity::kC1 && k.t_ns >= f.t_ns.front() && k.t_ns <= f.t_ns.back()) out.push_back(k.t_ns);
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

int64_t StepBracket::limit_ns(int64_t a, int64_t c) const {
  int64_t lim = 2 * h_min_ns;
  if (time) {
    int32_t Fa = 0, Fc = 0;
    float s = 0.0f;
    time->request(a, &Fa, &s);
    time->request(c, &Fc, &s);
    const double q = std::max(time->resolution_ns(Fa), time->resolution_ns(Fc));
    lim = std::max(lim, static_cast<int64_t>(std::ceil(2.0 * q)) + 2);
  }
  if (plateaus) lim += plateaus->dropped_ns(a, c);
  return lim;
}

Classified classify(const Track& fit, int64_t tc, double eps_px, const StepBracket& bracket, double k_sig) {
  Classified out;
  const size_t n = fit.size();
  if (n < 2) return out;
  const auto i = static_cast<size_t>(std::lower_bound(fit.t_ns.begin(), fit.t_ns.end(), tc) - fit.t_ns.begin());
  const size_t iref = std::min(i == 0 ? size_t{0} : i - 1, n - 1);
  const Signal sig{fit.w_pos[iref], fit.w_rot[iref], fit.q[iref], fit.p[iref]};
  out.w_pos = sig.w_pos;
  out.w_rot = sig.w_rot;
  std::vector<size_t> splits;
  for (const size_t j : {i, i + 1, i - 1}) {  // i - 1 wraps for i = 0 and is skipped
    if (j >= 1 && j < n) splits.push_back(j);
  }
  auto pair_at = [&](size_t j) {
    const Side l = one_sided(fit, j >= kSide ? j - kSide : 0, j, fit.t_ns[j], sig);
    const Side r = one_sided(fit, j, std::min(n, j + kSide), fit.t_ns[j], sig);
    return std::make_pair(l, r);
  };
  auto valid = [&](const Side& s) { return s.poly && std::sqrt(s.poly->s2) <= eps_px / 2.0 + s.f1; };
  for (const size_t j : splits) {  // a value step within a bracket the sampler could not split (StepBracket)
    if (fit.t_ns[j] - fit.t_ns[j - 1] > bracket.limit_ns(fit.t_ns[j - 1], fit.t_ns[j])) continue;
    const auto [l, r] = pair_at(j);
    if (!(valid(l) && valid(r))) continue;
    const double f1 = std::max(l.f1, r.f1), noise = std::max(kNoiseFrac * eps_px, f1);
    const double sg = std::sqrt(std::max({l.poly->s2, r.poly->s2, noise * noise}));
    const double d0 = norm(sub(r.poly->c[0], l.poly->c[0])), se0 = sg * std::sqrt(l.poly->v00 + r.poly->v00);
    if (d0 > std::max(eps_px / 4.0 + 2.0 * f1, k_sig * se0)) {
      out.kind = Classified::Kind::kStep;
      out.t_ns = fit.t_ns[j];
      out.a = j - 1;
      out.c = j;
      out.jump_px = d0;
      out.dv_pxps = norm(sub(r.poly->c[1], l.poly->c[1]));
      out.v_pxps = std::max(pose_speed(l.poly->c[1]), pose_speed(r.poly->c[1]));
      out.f1 = f1;
      return out;
    }
  }
  for (const size_t j : splits) {  // a kink: the one-sided slopes differ
    const auto [l, r] = pair_at(j);
    if (!(valid(l) && valid(r))) continue;
    const double f1 = std::max(l.f1, r.f1), noise = std::max(kNoiseFrac * eps_px, f1);
    const double sg = std::sqrt(std::max({l.poly->s2, r.poly->s2, noise * noise}));
    const double dv = norm(sub(r.poly->c[1], l.poly->c[1])), se1 = sg * std::sqrt(l.poly->v11 + r.poly->v11);
    out.t_ns = fit.t_ns[j];
    out.f1 = f1;
    out.jump_px = norm(sub(r.poly->c[0], l.poly->c[0]));
    out.dv_pxps = dv;
    if (dv <= k_sig * se1) {
      out.kind = Classified::Kind::kSmooth;
      return out;
    }
    double x = 0.0;  // the one-sided cubics cross at the kink
    for (int it = 0; it < 30; ++it) {
      const Y g = sub(value(*r.poly, x), value(*l.poly, x)), dg = sub(slope(*r.poly, x), slope(*l.poly, x));
      const double st = dot(g, dg) / std::max(dot(dg, dg), 1e-300);
      x -= st;
      if (std::fabs(st) < 1e-13) break;
    }
    const double lo = static_cast<double>(fit.t_ns[j - 1] - fit.t_ns[j]) * 1e-9;
    const double hi = static_cast<double>(fit.t_ns[std::min(j + 1, n - 1)] - fit.t_ns[j]) * 1e-9;
    x = std::clamp(x, lo, hi);
    out.kind = Classified::Kind::kKink;
    out.t_ns = fit.t_ns[j] + std::llround(x * 1e9);
    return out;
  }
  out.t_ns = tc;
  return out;
}

double pose_px(double w_pos, double w_rot, const rot::Q& qa, const rot::V& pa, const rot::Q& qb, const rot::V& pb) {
  return w_pos * norm(pa - pb) + w_rot * norm(so3::log(conjugate(qa) * qb));
}

std::vector<int64_t> burst_requests(int64_t t_ns, int64_t w_ns, int n) {
  if (n < 1 || w_ns < 1) throw std::invalid_argument("burst_requests: n and the window must be >= 1");
  const double w = static_cast<double>(w_ns), t = static_cast<double>(t_ns);
  std::vector<int64_t> out;
  for (int k = 0; k < n; ++k) {
    const double f = n == 1 ? 0.0 : static_cast<double>(k) / static_cast<double>(n - 1);
    out.push_back(static_cast<int64_t>(std::nearbyint(t - (w + (w / 16.0 - w) * f))));
    out.push_back(static_cast<int64_t>(std::nearbyint(t + (w / 16.0 + (w - w / 16.0) * f))));
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

Crossing burst_crossing(const Track& s, int64_t tc, int64_t w_ns, double w_pos, double w_rot) {
  Crossing out;
  const auto split =
      static_cast<size_t>(std::lower_bound(s.t_ns.begin(), s.t_ns.end(), tc) - s.t_ns.begin());
  if (split < 3 || s.size() - split < 3) return out;
  const Signal sig{w_pos, w_rot, s.q[split - 1], s.p[split - 1]};
  auto fit_side = [&](size_t lo, size_t hi) {
    std::vector<double> x;
    std::vector<Y> y;
    for (size_t r = lo; r < hi; ++r) {
      x.push_back(static_cast<double>(s.t_ns[r] - tc) * 1e-9);
      y.push_back(sig(s.q[r], s.p[r]));
    }
    return polyfit(x, y, 2);
  };
  const std::optional<Poly> l = fit_side(0, split), r = fit_side(split, s.size());
  if (!l || !r) return out;
  double x = 0.0;
  for (int it = 0; it < 40; ++it) {
    const Y g = sub(value(*r, x), value(*l, x)), dg = sub(slope(*r, x), slope(*l, x));
    const double st = dot(g, dg) / std::max(dot(dg, dg), 1e-300);
    x -= st;
    if (std::fabs(st) < 1e-12) break;
  }
  const double w = static_cast<double>(w_ns) * 1e-9;
  x = std::clamp(x, -w, w);
  out.ok = true;
  out.t_ns = tc + std::llround(x * 1e9);
  out.dv_pxps = norm(sub(slope(*r, x), slope(*l, x)));
  out.jump_px = norm(sub(value(*r, x), value(*l, x)));
  return out;
}

std::vector<Breakpoint> merge_breakpoints(std::vector<Breakpoint> bps, int64_t merge_ns) {
  std::stable_sort(bps.begin(), bps.end(), [](const Breakpoint& a, const Breakpoint& b) { return a.t_ns < b.t_ns; });
  std::vector<Breakpoint> out;
  for (const Breakpoint& b : bps) {
    if (!out.empty() && b.t_ns - out.back().t_ns < merge_ns) {
      if (stronger(b, out.back())) out.back() = b;
      continue;
    }
    out.push_back(b);
  }
  return out;
}

int snap_breakpoints(std::vector<Breakpoint>& bps, std::span<const DeclaredKey> keys, const TimeMap& time) {
  auto half_width = [&](const DeclaredKey& k) -> int64_t {
    return (k.plateau ? static_cast<int64_t>(std::llround(k.plateau_frames * time.period_ns())) : 0) + 1000;
  };
  if (!std::is_sorted(keys.begin(), keys.end(),
                      [](const DeclaredKey& a, const DeclaredKey& b) { return a.t_ns < b.t_ns; })) {
    throw std::invalid_argument("snap_breakpoints: the declared keys must be ascending in time");
  }
  int64_t reach = 1000;  // the widest snap window: only the keys within it of a breakpoint are visited
  for (const DeclaredKey& k : keys) reach = std::max(reach, half_width(k));
  int snapped = 0;
  for (Breakpoint& b : bps) {
    const DeclaredKey* best = nullptr;
    int64_t best_d = std::numeric_limits<int64_t>::max();
    auto it = std::lower_bound(keys.begin(), keys.end(), b.t_ns - reach,
                               [](const DeclaredKey& k, int64_t v) { return k.t_ns < v; });
    for (; it != keys.end() && it->t_ns <= b.t_ns + reach; ++it) {
      const int64_t d = b.t_ns > it->t_ns ? b.t_ns - it->t_ns : it->t_ns - b.t_ns;
      if (d <= half_width(*it) && d < best_d) {
        best = &*it;
        best_d = d;
      }
    }
    if (!best) continue;
    int32_t F = 0;
    float s = 0.0f;
    time.request(best->t_ns, &F, &s);
    b.t_ns = best->t_ns;
    b.declared = b.snapped = true;
    b.dt_bound_ns = time.resolution_ns(F);
    ++snapped;
  }
  return snapped;
}

}  // namespace balla::scene::motion
