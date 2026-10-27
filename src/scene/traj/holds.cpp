#include "scene/traj/holds.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "scene/npo.h"

namespace balla::scene::traj {

namespace {

using rot::Q;
using rot::V;

using npo::maximum;
using npo::max;

// HoldView's per-camera gains f (1 + r^2) (zip(focal_px * (1.0 + corner_r2), cameras_body_m) truncates to the shorter)
// and the depth bound of input sample i (NumPy would raise IndexError past its end).
std::vector<double> view_gains(const HoldView& v) {
  if (v.focal_px.size() != v.corner_r2.size()) {
    throw std::invalid_argument("HoldView: focal_px and corner_r2 differ in length");
  }
  std::vector<double> g(v.focal_px.size());
  for (size_t c = 0; c < g.size(); ++c) g[c] = v.focal_px[c] * (1.0 + v.corner_r2[c]);
  if (v.has_cameras && v.cameras_body.size() < g.size()) g.resize(v.cameras_body.size());
  return g;
}

double depth_at(const HoldView& v, int64_t i) {
  if (i < 0 || static_cast<size_t>(i) >= v.depth_m.size()) {
    throw std::invalid_argument("HoldView: depth_m has no entry for a sample");
  }
  return v.depth_m[static_cast<size_t>(i)];
}
// Python's max(a, b) / min(a, b) on floats.
double py_max(double a, double b) { return b > a ? b : a; }
double py_min(double a, double b) { return b < a ? b : a; }

constexpr double kCreepPercentile = 20.0;

std::vector<std::pair<int64_t, int64_t>> runs_of(const std::vector<uint8_t>& mask) {
  std::vector<std::pair<int64_t, int64_t>> out;
  const auto n = static_cast<int64_t>(mask.size());
  int64_t i = 0;
  while (i < n) {
    if (mask[static_cast<size_t>(i)]) {
      int64_t j = i;
      while (j + 1 < n && mask[static_cast<size_t>(j + 1)]) ++j;
      out.emplace_back(i, j);
      i = j + 1;
    } else {
      ++i;
    }
  }
  return out;
}

std::vector<std::pair<int64_t, int64_t>> quiet_runs(std::span<const int64_t> t, std::span<const double> speed, int64_t i0,
                                                    int64_t i1, double moving_speed, int64_t min_duration_ns,
                                                    double edge_floor, double edge_factor) {
  const std::span<const double> s = speed.subspan(static_cast<size_t>(i0), static_cast<size_t>(i1 - i0 + 1));
  const auto len = static_cast<int64_t>(s.size());
  std::vector<uint8_t> fast(s.size()), moving(s.size(), 0);
  for (size_t k = 0; k < s.size(); ++k) fast[k] = s[k] > moving_speed ? 1 : 0;
  for (const auto& [a, b] : runs_of(fast)) {
    if (t[static_cast<size_t>(i0 + b)] - t[static_cast<size_t>(i0 + a)] >= min_duration_ns) {
      for (int64_t k = a; k <= b; ++k) moving[static_cast<size_t>(k)] = 1;
    }
  }
  std::vector<uint8_t> still(s.size());
  for (size_t k = 0; k < s.size(); ++k) still[k] = moving[k] ? 0 : 1;
  std::vector<std::pair<int64_t, int64_t>> runs;
  const int64_t last = len - 1;
  for (const auto& [a, b] : runs_of(still)) {
    const std::vector<double> seg(s.begin() + a, s.begin() + b + 1);
    const double edge = py_min(0.5 * moving_speed, py_max(edge_factor * npo::median(seg), edge_floor));
    int64_t first = -1, lastin = -1;
    for (int64_t k = 0; k <= b - a; ++k) {
      if (seg[static_cast<size_t>(k)] <= edge) {
        if (first < 0) first = k;
        lastin = k;
      }
    }
    if (first < 0) continue;
    runs.emplace_back(i0 + (a == 0 ? a : a + first), i0 + (b == last ? b : a + lastin));
  }
  std::vector<std::pair<int64_t, int64_t>> out;
  for (const auto& r : runs) {
    if (t[static_cast<size_t>(r.second)] - t[static_cast<size_t>(r.first)] >= min_duration_ns) out.push_back(r);
  }
  return out;
}

}  // namespace

double sphere_displacement(Q qa, V pa, Q qb, V pb, V center, double radius) {
  const V ca = rot::quat_rotate(qa, center) + pa;
  const V cb = rot::quat_rotate(qb, center) + pb;
  const double theta = rot::rotation_angle(qa, qb);
  return rot::norm_axis(ca - cb) + (2.0 * radius) * std::sin(0.5 * theta);
}

double sphere_speed(V v, V omega, Q q, V center, double radius) {
  const V vc = v + rot::quat_rotate(q, rot::cross(omega, center));
  return rot::norm_axis(vc) + radius * rot::norm_axis(omega);
}

std::vector<double> HoldView::speed(std::span<const Q> q, std::span<const V> v, std::span<const V> omega, V center,
                                    double radius, std::span<const int64_t> idx) const {
  const size_t n = idx.size();
  std::vector<double> out(n, 0.0);
  const std::vector<double> gains = view_gains(*this);
  if (!has_cameras) {
    const double g = max(gains);
    for (size_t i = 0; i < n; ++i) {
      out[i] = (g * sphere_speed(v[i], omega[i], q[i], center, radius)) / depth_at(*this, idx[i]);
    }
    return out;
  }
  for (size_t c = 0; c < gains.size(); ++c) {
    for (size_t i = 0; i < n; ++i) {
      const double z = depth_at(*this, idx[i]);
      const double w = rot::norm_axis(omega[i]);
      const double vv = rot::norm_axis(v[i] + rot::quat_rotate(q[i], rot::cross(omega[i], cameras_body[c])));
      out[i] = maximum(out[i], gains[c] * (w + vv / z));
    }
  }
  return out;
}

double HoldView::excursion(std::span<const Q> q, std::span<const V> p, V center, double radius,
                           std::span<const int64_t> idx) const {
  const auto [qm, pm] = rot::robust_mean_pose(q, p);
  const size_t n = q.size();
  const std::vector<double> gains = view_gains(*this);
  std::vector<double> out(n, 0.0);
  if (!has_cameras) {
    const double g = max(gains);
    for (size_t i = 0; i < n; ++i) {
      out[i] = (g * sphere_displacement(qm, pm, q[i], p[i], center, radius)) / depth_at(*this, idx[i]);
    }
    return max(out);
  }
  std::vector<double> theta(n);
  for (size_t i = 0; i < n; ++i) theta[i] = rot::rotation_angle(qm, q[i]);
  for (size_t c = 0; c < gains.size(); ++c) {
    for (size_t i = 0; i < n; ++i) {
      const double z = depth_at(*this, idx[i]);
      out[i] = maximum(out[i], gains[c] * (theta[i] + sphere_displacement(qm, pm, q[i], p[i], cameras_body[c], 0.0) / z));
    }
  }
  return max(out);
}

HoldResult detect_static(std::span<const int64_t> t, std::span<const Q> q, std::span<const V> p,
                         std::span<const double> speed, V center, double radius, const StaticOptions& o,
                         const std::function<double(int64_t, int64_t)>& excursion) {
  if (q.size() != t.size() || p.size() != t.size() || speed.size() != t.size()) {
    throw std::invalid_argument("detect_static: t, q, p and speed differ in length");
  }
  HoldResult res;
  auto default_check = [&](int64_t i0, int64_t i1) {
    const auto len = static_cast<size_t>(i1 - i0 + 1);
    const std::span<const Q> qs = q.subspan(static_cast<size_t>(i0), len);
    const std::span<const V> ps = p.subspan(static_cast<size_t>(i0), len);
    const auto [qm, pm] = rot::robust_mean_pose(qs, ps);
    std::vector<double> disp(len);
    for (size_t k = 0; k < len; ++k) disp[k] = sphere_displacement(qm, pm, qs[k], ps[k], center, radius);
    return max(disp);
  };
  auto check = [&](int64_t i0, int64_t i1) { return excursion ? excursion(i0, i1) : default_check(i0, i1); };
  auto reject = [&](int64_t i0, int64_t i1, double d, const char* reason) {
    res.rejected.push_back({t[static_cast<size_t>(i0)], t[static_cast<size_t>(i1)], d, reason});
  };
  using Piece = std::tuple<int64_t, int64_t, double>;
  struct Split {
    std::vector<Piece> pieces, failed, shorts;
  };
  auto split_at = [&](int64_t i0, int64_t i1, double pct) {
    const std::vector<double> seg(speed.begin() + i0, speed.begin() + i1 + 1);
    const double creep = py_max(o.edge_noise_factor * npo::percentile(seg, pct), o.edge_speed_floor);
    std::vector<std::optional<Piece>> pieces;
    Split out;
    for (const auto& [a, b] : quiet_runs(t, speed, i0, i1, creep, o.min_duration_ns, o.edge_speed_floor,
                                         o.edge_noise_factor)) {
      const double e = check(a, b);
      if (e > o.max_disp) {
        out.failed.emplace_back(a, b, e);
        pieces.emplace_back(std::nullopt);
        continue;
      }
      if (!pieces.empty() && pieces.back()) {
        const double u = check(std::get<0>(*pieces.back()), b);
        if (u <= o.max_disp) {
          pieces.back() = Piece{std::get<0>(*pieces.back()), b, u};
          continue;
        }
      }
      pieces.emplace_back(Piece{a, b, e});
    }
    for (const auto& x : pieces) {
      if (!x) continue;
      const int64_t dur = t[static_cast<size_t>(std::get<1>(*x))] - t[static_cast<size_t>(std::get<0>(*x))];
      (dur >= o.min_piece_ns ? out.pieces : out.shorts).push_back(*x);
    }
    return out;
  };
  std::vector<Piece> accepted;
  for (const auto& [i0, i1] : quiet_runs(t, speed, 0, static_cast<int64_t>(t.size()) - 1, o.moving_speed,
                                         o.min_duration_ns, o.edge_speed_floor, o.edge_noise_factor)) {
    const double d = check(i0, i1);
    std::vector<Piece> pieces{Piece{i0, i1, d}};
    if (d > o.max_disp) {
      if (!o.split) {
        reject(i0, i1, d, "displacement");
        continue;
      }
      Split sp = split_at(i0, i1, 50.0);
      if (sp.pieces.empty()) sp = split_at(i0, i1, kCreepPercentile);
      std::vector<Piece> drift = sp.failed;
      if (drift.empty() && sp.pieces.empty() && sp.shorts.empty()) drift.emplace_back(i0, i1, d);
      for (const auto& x : drift) reject(std::get<0>(x), std::get<1>(x), std::get<2>(x), "drift");
      for (const auto& x : sp.shorts) reject(std::get<0>(x), std::get<1>(x), std::get<2>(x), "short");
      pieces = sp.pieces;
    }
    for (const auto& [a, b, dd] : pieces) {
      if (!accepted.empty() &&
          t[static_cast<size_t>(a)] - t[static_cast<size_t>(std::get<1>(accepted.back()))] < o.min_duration_ns) {
        const double d_union = check(std::get<0>(accepted.back()), b);
        if (d_union <= o.max_disp) {
          accepted.back() = Piece{std::get<0>(accepted.back()), b, d_union};
          continue;
        }
      }
      accepted.emplace_back(a, b, dd);
    }
  }
  for (const auto& [a, b, d] : accepted) {
    res.accepted.push_back({a, b, t[static_cast<size_t>(a)], t[static_cast<size_t>(b)], d});
  }
  return res;
}

HoldResult detect_static_image(std::span<const int64_t> t, std::span<const Q> q, std::span<const V> p,
                               std::span<const V> v, std::span<const V> omega, const HoldView& view,
                               std::span<const int64_t> idx, V center, double radius, int64_t min_duration_ns,
                               double max_disp_px, int64_t min_piece_ns) {
  const size_t n = t.size();
  if (q.size() != n || p.size() != n || v.size() != n || omega.size() != n || idx.size() != n) {
    throw std::invalid_argument("detect_static_image: t, q, p, v, omega and idx differ in length");
  }
  const double eps_on = max_disp_px / (static_cast<double>(min_duration_ns) * 1e-9);
  auto excursion = [&](int64_t i0, int64_t i1) {
    const auto len = static_cast<size_t>(i1 - i0 + 1);
    return view.excursion(q.subspan(static_cast<size_t>(i0), len), p.subspan(static_cast<size_t>(i0), len), center,
                          radius, idx.subspan(static_cast<size_t>(i0), len));
  };
  const std::vector<double> speed = view.speed(q, v, omega, center, radius, idx);
  StaticOptions o;
  o.max_disp = max_disp_px;
  o.min_duration_ns = min_duration_ns;
  o.moving_speed = eps_on;
  o.edge_speed_floor = eps_on / 20;
  o.split = true;
  o.min_piece_ns = min_piece_ns;
  return detect_static(t, q, p, speed, center, radius, o, excursion);
}

TieMap::TieMap(int64_t n, std::vector<std::pair<int64_t, int64_t>> g) : num_ctrl(n), groups(std::move(g)) {
  // Groups are control ranges (hold_groups clips them to [0, n - 1]); the Python's f[c0:c1 + 1] = c0 also clips at n
  // and leaves c0 > c1 empty, but a negative bound would count from the end there: refused here.
  if (n < 0) throw std::invalid_argument("TieMap: negative number of control points");
  for (const auto& [c0, c1] : groups) {
    if (c0 < 0 || c1 < 0) throw std::invalid_argument("TieMap: negative control index in a group");
  }
  std::vector<int64_t> f(static_cast<size_t>(n));
  for (int64_t c = 0; c < n; ++c) f[static_cast<size_t>(c)] = c;
  for (const auto& [c0, c1] : groups) {
    for (int64_t c = c0; c <= c1 && c < n; ++c) f[static_cast<size_t>(c)] = c0;
  }
  std::vector<int64_t> u = f;
  std::sort(u.begin(), u.end());
  u.erase(std::unique(u.begin(), u.end()), u.end());
  free_index.resize(static_cast<size_t>(n));
  for (int64_t c = 0; c < n; ++c) {
    free_index[static_cast<size_t>(c)] = std::lower_bound(u.begin(), u.end(), f[static_cast<size_t>(c)]) - u.begin();
  }
  num_free = n ? *std::max_element(free_index.begin(), free_index.end()) + 1 : 0;
}

sparse::Matrix TieMap::matrix(int block) const {
  std::vector<int64_t> rows, cols;
  std::vector<double> ones;
  for (int64_t c = 0; c < num_ctrl; ++c) {
    for (int b = 0; b < block; ++b) {
      rows.push_back(c * block + b);
      cols.push_back(free_index[static_cast<size_t>(c)] * block + b);
      ones.push_back(1.0);
    }
  }
  return sparse::coo_to_csr(num_ctrl * block, num_free * block, rows, cols, ones);
}

std::pair<std::vector<std::pair<int64_t, int64_t>>, std::vector<std::pair<int64_t, int64_t>>> hold_groups(
    const KnotGrid& grid, std::vector<std::pair<int64_t, int64_t>> intervals) {
  std::sort(intervals.begin(), intervals.end());
  std::vector<std::pair<int64_t, int64_t>> groups;
  for (const auto& [ta, tb] : intervals) {
    const auto [c0, c1] = grid.controls_touching(ta, tb);
    if (!groups.empty() && c0 <= groups.back().second) {
      groups.back().second = std::max(groups.back().second, c1);
    } else {
      groups.emplace_back(c0, c1);
    }
  }
  std::vector<std::pair<int64_t, int64_t>> out, still;
  for (const auto& g : groups) {
    if (g.second - g.first + 1 >= grid.order) out.push_back(g);
  }
  for (const auto& g : out) still.push_back(grid.still_interval(g.first, g.second));
  return {out, still};
}

}  // namespace balla::scene::traj
