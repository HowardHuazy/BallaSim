#include "render/ao_proxy.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace balla::render {
namespace {

using V3 = std::array<double, 3>;

V3 sub(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const V3& a, const V3& b) { return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]; }
double dist2(const V3& a, const V3& b) { return dot(sub(a, b), sub(a, b)); }

// Eigenvectors (columns of v) of a symmetric 3 x 3 matrix by cyclic Jacobi rotations (Golub and Van Loan, §8.5).
void jacobi3(double a[3][3], double v[3][3]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) v[i][j] = i == j ? 1.0 : 0.0;
  }
  for (int sweep = 0; sweep < 64; ++sweep) {
    const double off = (a[0][1] * a[0][1] + a[0][2] * a[0][2]) + a[1][2] * a[1][2];
    const double diag = (a[0][0] * a[0][0] + a[1][1] * a[1][1]) + a[2][2] * a[2][2];
    if (!(off > 1e-30 * diag)) return;
    for (int p = 0; p < 2; ++p) {
      for (int q = p + 1; q < 3; ++q) {
        if (a[p][q] == 0.0) continue;
        const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (int k = 0; k < 3; ++k) {  // a <- a J (columns p, q)
          const double akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {  // a <- J^T a (rows p, q)
          const double apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
          const double vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
}

// Index of each sample's nearest centre (the lowest index on ties).
void assign_nearest(const std::vector<V3>& x, const std::vector<V3>& c, std::vector<int32_t>& out) {
  for (size_t i = 0; i < x.size(); ++i) {
    int32_t best = 0;
    double d_best = dist2(x[i], c[0]);
    for (size_t j = 1; j < c.size(); ++j) {
      const double d = dist2(x[i], c[j]);
      if (d < d_best) {
        d_best = d;
        best = static_cast<int32_t>(j);
      }
    }
    out[i] = best;
  }
}

// Each centre the mean of its samples, summed in sample order; a centre without samples keeps its place.
void update_means(const std::vector<V3>& x, const std::vector<int32_t>& a, std::vector<V3>& c) {
  std::vector<V3> sum(c.size(), V3{0.0, 0.0, 0.0});
  std::vector<size_t> n(c.size(), 0);
  for (size_t i = 0; i < x.size(); ++i) {
    V3& s = sum[static_cast<size_t>(a[i])];
    for (int e = 0; e < 3; ++e) s[e] += x[i][e];
    ++n[static_cast<size_t>(a[i])];
  }
  for (size_t j = 0; j < c.size(); ++j) {
    if (n[j] == 0) continue;
    const double inv = 1.0 / static_cast<double>(n[j]);
    c[j] = {sum[j][0] * inv, sum[j][1] * inv, sum[j][2] * inv};
  }
}

}  // namespace

AoProxy fit_ao_proxy(std::span<const float> positions, std::span<const uint32_t> indices) {
  AoProxy out;
  const size_t nt = indices.size() / 3;
  const auto vertex = [&](uint32_t i) {
    return V3{positions[3 * size_t{i}], positions[3 * size_t{i} + 1], positions[3 * size_t{i} + 2]};
  };
  std::vector<double> cdf(nt);
  double total = 0.0;
  for (size_t t = 0; t < nt; ++t) {
    const V3 a = vertex(indices[3 * t]);
    const V3 e1 = sub(vertex(indices[3 * t + 1]), a), e2 = sub(vertex(indices[3 * t + 2]), a);
    const V3 n{e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    total += 0.5 * std::sqrt(dot(n, n));
    cdf[t] = total;
  }
  if (!(total > 0.0) || !std::isfinite(total)) return out;

  // Area-weighted samples: stratified over the cumulative area, R2 barycentrics.
  constexpr double kPlastic = 1.32471795724474602596;
  constexpr double kA1 = 1.0 / kPlastic, kA2 = 1.0 / (kPlastic * kPlastic);
  std::vector<V3> x(kAoFitSamples);
  V3 m{0.0, 0.0, 0.0};
  for (uint32_t i = 0; i < kAoFitSamples; ++i) {
    const double target = (static_cast<double>(i) + 0.5) / static_cast<double>(kAoFitSamples) * total;
    const size_t t = std::min<size_t>(static_cast<size_t>(std::upper_bound(cdf.begin(), cdf.end(), target) -
                                                          cdf.begin()), nt - 1);
    double r1 = 0.5 + kA1 * static_cast<double>(i), r2 = 0.5 + kA2 * static_cast<double>(i);
    r1 -= std::floor(r1);
    r2 -= std::floor(r2);
    if (r1 + r2 > 1.0) {
      r1 = 1.0 - r1;
      r2 = 1.0 - r2;
    }
    const V3 a = vertex(indices[3 * t]);
    const V3 e1 = sub(vertex(indices[3 * t + 1]), a), e2 = sub(vertex(indices[3 * t + 2]), a);
    for (int e = 0; e < 3; ++e) x[i][e] = a[e] + (r1 * e1[e] + r2 * e2[e]);
    for (int e = 0; e < 3; ++e) m[e] += x[i][e];
  }
  const double inv_n = 1.0 / static_cast<double>(kAoFitSamples);
  m = {m[0] * inv_n, m[1] * inv_n, m[2] * inv_n};

  // Principal axes and the extents along them.
  double cov[3][3] = {};
  for (const V3& p : x) {
    const V3 d = sub(p, m);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) cov[r][c] += d[r] * d[c];
    }
  }
  double axes[3][3];
  jacobi3(cov, axes);
  double e_min = std::numeric_limits<double>::infinity(), e_max = 0.0;
  for (int j = 0; j < 3; ++j) {
    const V3 u{axes[0][j], axes[1][j], axes[2][j]};
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (const V3& p : x) {
      const double s = dot(sub(p, m), u);
      lo = std::min(lo, s);
      hi = std::max(hi, s);
    }
    e_min = std::min(e_min, hi - lo);
    e_max = std::max(e_max, hi - lo);
  }
  int32_t k = kAoMaxSpheres;
  if (e_min > 0.0) {
    k = static_cast<int32_t>(std::clamp(std::ceil(e_max / e_min - kAoFitSlack), 1.0, double{kAoMaxSpheres}));
  }

  // Farthest-point seeds, then Lloyd.
  std::vector<V3> c;
  if (k == 1) {
    c.push_back(m);
  } else {
    std::vector<double> dmin(x.size());
    for (size_t i = 0; i < x.size(); ++i) dmin[i] = dist2(x[i], m);
    for (int32_t j = 0; j < k; ++j) {
      const size_t far = static_cast<size_t>(std::max_element(dmin.begin(), dmin.end()) - dmin.begin());
      if (j > 0 && !(dmin[far] > 0.0)) break;  // fewer distinct samples than k
      c.push_back(x[far]);
      for (size_t i = 0; i < x.size(); ++i) {
        const double d = dist2(x[i], c.back());
        dmin[i] = j == 0 ? d : std::min(dmin[i], d);
      }
    }
  }
  std::vector<int32_t> assign(x.size()), next(x.size());
  assign_nearest(x, c, assign);
  for (int32_t it = 0; it < kAoFitIterations; ++it) {
    update_means(x, assign, c);
    assign_nearest(x, c, next);
    if (next == assign) break;
    assign.swap(next);
  }
  update_means(x, assign, c);

  // Radii: each sphere's RMS distance to its samples.
  std::vector<double> ss(c.size(), 0.0);
  std::vector<size_t> n(c.size(), 0);
  for (size_t i = 0; i < x.size(); ++i) {
    ss[static_cast<size_t>(assign[i])] += dist2(x[i], c[static_cast<size_t>(assign[i])]);
    ++n[static_cast<size_t>(assign[i])];
  }
  double rb2 = 0.0;
  for (size_t j = 0; j < c.size(); ++j) {
    if (n[j] == 0) continue;
    const double r = std::sqrt(ss[j] / static_cast<double>(n[j]));
    if (!(r > 0.0)) continue;
    out.spheres.push_back({c[j], r});
    rb2 += r * r;
  }
  out.radius_b = std::sqrt(rb2);
  out.d0 = out.radius_b * kAoWindowScale;
  return out;
}

}  // namespace balla::render
