#include "scene/traj/pin.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

#include "scene/npo.h"
#include "scene/traj/basis.h"
#include "scene/traj/holds.h"

namespace balla::scene::traj {

namespace {

using rot::Q;
using rot::V;

constexpr double kNearFloorM = 0.05;   // as the image-space hold rule's depth floor (build.cpp)

double depth_at(const PinSamples& s, size_t i) { return std::max(s.depth_m[i], kNearFloorM); }

// The image displacement (px) of the body's surface between two poses, seen by a still camera at depth z.
double image_displacement(const PinSamples& s, const PinOptions& o, Q qa, V pa, Q qb, V pb, double z) {
  return (o.focal_px * sphere_displacement(qa, pa, qb, pb, s.center, s.radius_m)) / z;
}

// A kept sample's pivot in the world and its unit rotation, so that the all-pairs window diameter below does not
// re-derive them per pair. pair_distance is sphere_displacement's bound (holds.cpp): the pivot's travel plus
// 2 R sin(theta / 2), with sin(theta / 2) the vector part of the relative unit quaternion.
struct Pivot {
  V c;
  Q q;
};

double pair_distance(const Pivot& a, const Pivot& b, double radius) {
  const Q rel = rot::quat_multiply(rot::quat_conjugate(a.q), b.q);
  const double sin_half = std::min(1.0, std::sqrt((rel.x * rel.x + rel.y * rel.y) + rel.z * rel.z));
  return rot::norm_axis(a.c - b.c) + (2.0 * radius) * sin_half;
}

void check(const PinSamples& s, const PinOptions& o) {
  const size_t n = s.t_ns.size();
  if (s.q.size() != n || s.p.size() != n || s.depth_m.size() != n) {
    throw std::invalid_argument("pin: t, q, p and depth differ in length");
  }
  if (n == 0) throw std::invalid_argument("pin: no samples");
  if (!(o.limit_px_s > 0) || o.lag_ns <= 0 || o.min_ns < 0 || !(o.focal_px > 0)) {
    throw std::invalid_argument("pin: the limit, lag and focal length must be positive");
  }
}

}  // namespace

PinResult pin_intervals(const PinSamples& s, const PinOptions& o) {
  check(s, o);
  const size_t n = s.t_ns.size();
  PinResult r;
  r.speed_p99 = r.speed_max = std::numeric_limits<double>::quiet_NaN();
  std::vector<Pivot> pivot(n);
  for (size_t k = 0; k < n; ++k) pivot[k] = {rot::quat_rotate(s.q[k], s.center) + s.p[k], rot::normalize_quat(s.q[k])};
  std::vector<size_t> partner;
  std::vector<double> speed;
  size_t j = 0;
  for (size_t i = 0; i < n; ++i) {
    j = std::max(j, i + 1);
    while (j < n && s.t_ns[j] - s.t_ns[i] < o.lag_ns) ++j;
    if (j == n) break;
    partner.push_back(j);
    // The window's rate: its diameter (the furthest apart any two of its samples are) over its length. A steady drift
    // reads its speed (the diameter is first to last); a body that is back where it was a lag later (vibration near
    // 1 / lag and its multiples) reads the stroke it makes, in every phase, not zero.
    double diameter = 0.0;
    for (size_t k = i; k <= j; ++k) {
      for (size_t l = k + 1; l <= j; ++l) diameter = std::max(diameter, pair_distance(pivot[k], pivot[l], s.radius_m));
    }
    speed.push_back((o.focal_px * diameter / depth_at(s, i)) / (static_cast<double>(s.t_ns[j] - s.t_ns[i]) * 1e-9));
  }
  const size_t m = speed.size();
  r.pairs = static_cast<int64_t>(m);
  if (m == 0) return r;   // a run shorter than the lag: nothing to judge, nothing frozen
  r.speed_max = *std::max_element(speed.begin(), speed.end());
  r.speed_p99 = npo::percentile(speed, 99.0);
  // The union of the quiet windows [t_i, t_partner(i)]; the last window runs to the last sample. A gap no longer than
  // bridge_ns (the reach of a hold's outer controls) is bridged: hold_groups would merge the two holds anyway, and a lone
  // bad sample (it spoils every window that contains it, one lag of starts) leaves a gap of two samples.
  std::vector<std::pair<int64_t, int64_t>> iv;
  for (size_t i = 0; i < m; ++i) {
    if (!(speed[i] < o.limit_px_s)) continue;
    const int64_t a = s.t_ns[i], b = i + 1 == m ? s.t_ns[n - 1] : s.t_ns[partner[i]];
    if (!iv.empty() && a - iv.back().second <= o.bridge_ns) {
      iv.back().second = std::max(iv.back().second, b);
    } else {
      iv.emplace_back(a, b);
    }
  }
  r.whole = iv.size() == 1 && iv[0].first == s.t_ns[0] && iv[0].second == s.t_ns[n - 1];
  for (const auto& x : iv) {
    if (r.whole || x.second - x.first >= o.min_ns) r.intervals_ns.push_back(x);
  }
  return r;
}

double pin_apply(FitResult& fit, const PinSamples& s, const PinOptions& o,
                 const std::vector<std::pair<int64_t, int64_t>>& intervals_ns, bool whole) {
  check(s, o);
  const size_t n = s.t_ns.size();
  const KnotGrid grid(fit.t0_ns, fit.h_ns, fit.order, static_cast<int64_t>(fit.rot.size()));
  // The fit's own holds (its tied groups, presence holds included) join the intervals: a group is never split.
  std::vector<std::pair<int64_t, int64_t>> all = whole ? std::vector<std::pair<int64_t, int64_t>>{fit.span_ns}
                                                       : intervals_ns;
  all.insert(all.end(), fit.holds.begin(), fit.holds.end());
  const auto [groups, still] = hold_groups(grid, all);
  double excursion = 0.0;
  for (size_t g = 0; g < groups.size(); ++g) {
    const auto [t_lo, t_hi] = still[g];
    // Only the groups under one of the intervals are set; the fit's other holds keep their fitted controls.
    const auto touched = [&](const std::pair<int64_t, int64_t>& x) { return x.first < t_hi && x.second > t_lo; };
    if (!whole && std::none_of(intervals_ns.begin(), intervals_ns.end(), touched)) continue;
    std::vector<Q> qs;
    std::vector<V> ps;
    std::vector<size_t> at;
    for (size_t k = 0; k < n; ++k) {
      if (whole || (s.t_ns[k] >= t_lo && s.t_ns[k] < t_hi)) {
        qs.push_back(s.q[k]);
        ps.push_back(s.p[k]);
        at.push_back(k);
      }
    }
    if (qs.empty()) continue;
    auto [qc, pc] = rot::robust_mean_pose(qs, ps);
    // The cumulative rotation basis reads the shorter way between neighbours: take the sign nearest the control before.
    const auto c0 = static_cast<size_t>(groups[g].first), c1 = static_cast<size_t>(groups[g].second);
    const Q& nb = c0 > 0 ? fit.rot[c0 - 1] : fit.rot[std::min(c1 + 1, fit.rot.size() - 1)];
    if (qc.w * nb.w + qc.x * nb.x + qc.y * nb.y + qc.z * nb.z < 0) qc = {-qc.w, -qc.x, -qc.y, -qc.z};
    for (size_t c = c0; c <= c1; ++c) {
      fit.rot[c] = qc;
      fit.pos[c] = pc;
    }
    for (const size_t k : at) {
      excursion = std::max(excursion, image_displacement(s, o, qc, pc, s.q[k], s.p[k], depth_at(s, k)));
    }
  }
  fit.holds.assign(still.begin(), still.end());
  std::sort(fit.holds.begin(), fit.holds.end());
  fit.diag.holds_ns = fit.holds;
  if (whole) fit.diag.is_static = true;
  return excursion;
}

}  // namespace balla::scene::traj
