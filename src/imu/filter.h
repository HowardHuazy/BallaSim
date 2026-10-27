// The IMU's sensor filter, exact on the sample grid (docs/design/phase3/outputs.md §4.3.1-4.3.2). Header-only, host,
// fp64.
//
// Per channel, a first-order low-pass with tau_c = 1/(2 pi f_c) after a pure delay d = tau_table - tau_c. Over one
// sample interval [t_{k-1}, t_k]:
//
//   I_k = int (1/tau_c) e^{-(t_k - s)/tau_c} u(s) ds ~ sum_i w_i u(s_i)          (the panel rule below)
//   ubar_k = u(s_0) + sum_i what_i (u(s_i) - u(s_0)),   what_i = w_i / sum_j w_j     (the deviation form)
//   y_k = ubar_k + phi_k (y_{k-1} - ubar_k),            phi_k = e^{-dt_k/tau_c}
//
// Panel rule: split the interval where the integrand loses smoothness (the rig spline's knots shifted by d: a and
// omega_dot, so f_I, are C^2 at K = 6 knots and C^0 at K = 4 knots), cut each piece into ceil(len / (tau_c/2)) equal
// panels and use 4-point Gauss-Legendre on each. The normalized weights make the DC gain exactly 1, and the deviation
// form makes a constant input exact: every u(s_i) - u(s_0) and y - ubar is then 0, so y_k = u bitwise. The
// exponentials are pmath::exp, so results do not depend on libm.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "core/math/vec.h"
#include "core/pmath/explog.h"
#include "core/traj/pieces.h"
#include "imu/params.h"
#include "render/pose.h"

namespace balla::imu {

// Node times relative to the interval end (s <= 0) and raw Gauss-Legendre weights (half-panel length times the
// GL-4 weight), in the caller's time unit.
struct Panels {
  std::vector<double> s, w;
};

namespace detail {
// 4-point Gauss-Legendre on [-1, 1]: nodes +-sqrt(3/7 -+ 2/7 sqrt(6/5)), weights (18 +- sqrt(30))/36.
inline constexpr double kGlX[4] = {-0.86113631159405257522, -0.33998104358485626480, 0.33998104358485626480,
                                   0.86113631159405257522};
inline constexpr double kGlW[4] = {0.34785484513745385737, 0.65214515486254614263, 0.65214515486254614263,
                                   0.34785484513745385737};
}  // namespace detail

// The panel rule on [a, b] (a < b): pieces between a, the cuts (ascending, strictly inside (a, b)) and b, each in
// ceil(len / (tau_c / 2)) equal panels of GL-4 nodes. Times in one unit (tau_c's); out.s is relative to b.
inline void panel_nodes(double a, double b, double tau_c, std::span<const double> cuts, Panels& out) {
  out.s.clear();
  out.w.clear();
  const double half_tau = 0.5 * tau_c;
  double lo = a - b;  // relative to the interval end
  for (size_t c = 0; c <= cuts.size(); ++c) {
    const double hi = c < cuts.size() ? cuts[c] - b : 0.0;
    const double len = hi - lo;
    // The 1e-9 keeps a piece of exactly n half-tau lengths at n panels despite the rounding of len.
    const auto m = static_cast<int64_t>(std::max(1.0, std::ceil(len / half_tau - 1e-9)));
    for (int64_t j = 0; j < m; ++j) {
      const double e0 = lo + len * static_cast<double>(j) / static_cast<double>(m);
      const double e1 = j + 1 == m ? hi : lo + len * static_cast<double>(j + 1) / static_cast<double>(m);
      const double mid = 0.5 * (e0 + e1), half = 0.5 * (e1 - e0);
      for (int g = 0; g < 4; ++g) {
        out.s.push_back(mid + half * detail::kGlX[g]);
        out.w.push_back(half * detail::kGlW[g]);
      }
    }
    lo = hi;
  }
}

// The cuts of [a, b] (§4.3.2): the knots t0 + j h of the rig spline shifted by d that lie strictly inside (a, b),
// ascending. a, b and d are in ns relative to t_ref_ns; out is too. A piecewise rig (A11) cuts at each piece's begin
// (its join) and at the knots of each piece strictly inside the interval it serves, where the integrand's smoothness
// can break: the knots of a piece outside its interval are not on the path.
inline void knot_cuts(const render::TrajectoryView& rig, int64_t t_ref_ns, double a, double b, double d,
                      std::vector<double>& out) {
  out.clear();
  if (rig.num_pieces > 1) {
    const double inf = std::numeric_limits<double>::infinity();
    const auto push = [&](double c) {  // ascending and distinct, as rounding could merge a knot with the next begin
      if (a < c && c < b && (out.empty() || c > out.back())) out.push_back(c);
    };
    // From the piece serving an int64 time at or before t_ref + a - d (saturated; piece_index clamps to piece 0).
    const double rel = std::floor(a - d);
    int64_t t_first = rel <= -0x1p62 ? INT64_MIN : (rel >= 0x1p62 ? INT64_MAX : static_cast<int64_t>(rel));
    if (t_first != INT64_MIN && t_first != INT64_MAX && __builtin_add_overflow(t_first, t_ref_ns, &t_first)) {
      t_first = t_ref_ns > 0 ? INT64_MAX : INT64_MIN;
    }
    for (int32_t p = traj::piece_index(rig.pieces, rig.num_pieces, t_first); p < rig.num_pieces; ++p) {
      const render::PieceView& pc = rig.pieces[p];
      const double lo = p == 0 ? -inf : static_cast<double>(pc.begin_ns - t_ref_ns);  // the interval it serves
      const double hi = p + 1 < rig.num_pieces ? static_cast<double>(rig.pieces[p + 1].begin_ns - t_ref_ns) : inf;
      if (lo >= b - d) break;
      if (p > 0) push(lo + d);
      const double base = static_cast<double>(pc.t0_ns - t_ref_ns), h = pc.spline.h_ns;
      const auto j_lo = static_cast<int64_t>(std::floor((std::max(a - d, lo) - base) / h));
      const auto j_hi = static_cast<int64_t>(std::ceil((std::min(b - d, hi) - base) / h));
      for (int64_t j = std::max<int64_t>(j_lo, 0); j <= std::min<int64_t>(j_hi, pc.spline.num_segments); ++j) {
        const double x = base + static_cast<double>(j) * h;
        if (lo < x && x < hi) push(x + d);
      }
    }
    return;
  }
  const double base = static_cast<double>(rig.t0_ns - t_ref_ns), h = rig.spline.h_ns;
  const auto j_lo = static_cast<int64_t>(std::floor((a - d - base) / h));
  const auto j_hi = static_cast<int64_t>(std::ceil((b - d - base) / h));
  for (int64_t j = std::max<int64_t>(j_lo, 0); j <= std::min<int64_t>(j_hi, rig.spline.num_segments); ++j) {
    const double c = (base + static_cast<double>(j) * h) + d;
    if (a < c && c < b) out.push_back(c);
  }
}

// One step of the recursion: y_k from y_{k-1}, the panels of the interval (length dt > 0) and the input u at their
// nodes (the same unit as tau_c and dt).
inline Vec3d filter_step(const Vec3d& y_prev, const Panels& p, std::span<const Vec3d> u, double tau_c, double dt) {
  double sum = 0;
  Vec3d dev{0, 0, 0};
  const Vec3d u0 = u[0];
  for (size_t i = 0; i < p.s.size(); ++i) {
    const double w = p.w[i] * pmath::exp(p.s[i] / tau_c);
    sum += w;
    dev += w * (u[i] - u0);
  }
  const Vec3d mean = u0 + Vec3d{dev.x / sum, dev.y / sum, dev.z / sum};
  return mean + pmath::exp(-dt / tau_c) * (y_prev - mean);
}

}  // namespace balla::imu
