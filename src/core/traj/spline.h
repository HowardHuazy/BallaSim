// Split SO(3) x R^3 uniform cumulative B-spline of order K = 4 or 6 (DESIGN §5.3, M2 §A-B).
//
//   R(t) = R_s prod_{j=1..K-1} Exp(lambda_j(u) d_{s+j}),   d_c = Log(R_{c-1}^T R_c)      (M2 A6)
//   p(t) = p_s + sum_{j=1..K-1} lambda_j(u) dp_{s+j},     dp_c = p_c - p_{c-1}          (M2 A7)
//
// Positions use the difference form, relative to a per-scene fp64 origin baked into the nodes
// (fp32 standard form loses 0.08-0.16 m/s^2 of acceleration to cancellation, M2 D2). Segment s
// covers [t0 + s h, t0 + (s+1) h) and uses nodes s..s+K-1. Time stays integer until u is formed.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/quat.h"
#include "core/math/so3.h"
#include "core/math/vec.h"
#include "core/time/time.h"
#include "core/traj/basis.h"

namespace balla::traj {

// Device layout of one control point, derived in fp64 on the host (spline_build.h); 52 B in fp32.
template <class T>
struct SplineNode {
  Quat<T> q;   // R_c
  Vec3<T> d;   // Log(R_{c-1}^T R_c); zero for c = 0
  Vec3<T> p;   // p_c - origin
  Vec3<T> dp;  // p_c - p_{c-1}, differenced in fp64 before any cast; zero for c = 0
};

template <class T>
struct SplineView {
  const SplineNode<T>* nodes;
  int32_t num_segments;  // number of nodes - K + 1
  int32_t h_ns;          // knot spacing
  T inv_h_s;             // 1e9 / h_ns, the 1/h of the time derivatives
};

// Pose and derivatives at one time. omega and omega_dot are body-frame ([omega]x = R^T dR/dt);
// v and a are world-frame. Units: m, rad, s.
template <class T>
struct Kinematics {
  Quat<T> q;
  Vec3<T> p;
  Vec3<T> omega;
  Vec3<T> omega_dot;
  Vec3<T> v;
  Vec3<T> a;
};

struct SegmentTime {
  int32_t segment;
  int32_t r_ns;  // t - (t0 + segment h), in [0, h]
};

namespace detail {
BALLA_HD inline bool finish_locate(int64_t s, int64_t r, int32_t h_ns, int32_t num_segments,
                                   SegmentTime* out) {
  if (s == num_segments && r == 0) {  // the span is closed on the right: t_end maps to u = 1
    s = num_segments - 1;
    r = h_ns;
  }
  if (s < 0 || s >= num_segments) return false;
  *out = {static_cast<int32_t>(s), static_cast<int32_t>(r)};
  return true;
}
}  // namespace detail

// Host form (M2 A9): absolute int64 time. Returns false outside [t0, t0 + num_segments h]. The distance is taken in
// uint64 after the before-the-span test, so it is exact for any two int64 times (t - t0 in int64 overflowed for
// distances of 2^63 ns or more, and a wrapped one could land in the span).
BALLA_HD inline bool locate(int64_t t_ns, int64_t t0_ns, int32_t h_ns, int32_t num_segments,
                            SegmentTime* out) {
  if (t_ns < t0_ns) return false;
  const uint64_t d = static_cast<uint64_t>(t_ns) - static_cast<uint64_t>(t0_ns), q = d / static_cast<uint64_t>(h_ns);
  if (q > static_cast<uint64_t>(num_segments)) return false;
  const auto s = static_cast<int64_t>(q);
  return detail::finish_locate(s, static_cast<int64_t>(d - q * static_cast<uint64_t>(h_ns)), h_ns, num_segments, out);
}

// Device form (M2 A10): int32 offset tau from the knot-aligned base t0 + base_segment h.
// The remainder comes from floor_mod, not tau - sl * h, whose product overflows int32 for tau
// within h of INT32_MIN.
BALLA_HD inline bool locate_offset(int32_t base_segment, int32_t tau_ns, int32_t h_ns,
                                   int32_t num_segments, SegmentTime* out) {
  return detail::finish_locate(int64_t{base_segment} + time::floor_div(tau_ns, h_ns),
                               time::floor_mod(tau_ns, h_ns), h_ns, num_segments, out);
}

// Pose and derivatives at normalized time u in [0, 1] of a segment.
template <int K, class T>
BALLA_HD Kinematics<T> evaluate_u(const SplineView<T>& spline, int32_t segment, T u) {
  const SplineNode<T>* n = spline.nodes + segment;
  const CumulativeBasis<T, K> b = cumulative_basis<K>(u);
  // Both parts accumulate the small increments first and apply the base pose once:
  // R_s (A_1 ... A_{K-1}) and p_s + (sum lambda_j dp_j). Compounding onto R_s and p_s step by step
  // rounds at their full magnitude K-1 times; measured, that raises the worst fp32 rotation error
  // by 1.5x and the position error by 2.5x.
  Quat<T> inc = quat_identity<T>();
  Vec3<T> w{T(0), T(0), T(0)}, wd = w, shift = w, v = w, a = w;
BALLA_UNROLL
  for (int j = 1; j < K; ++j) {
    const Vec3<T> d = n[j].d;
    const Quat<T> step = so3::exp(b.lambda[j - 1] * d);
    inc = inc * step;
    // Sommer et al. recursions (M2 B1, B2) in units of u.
    const Vec3<T> w_next = rotate_inverse(step, w) + b.dlambda[j - 1] * d;
    wd = (b.dlambda[j - 1] * cross(w_next, d) + rotate_inverse(step, wd)) + b.ddlambda[j - 1] * d;
    w = w_next;
    shift += b.lambda[j - 1] * n[j].dp;
    v += b.dlambda[j - 1] * n[j].dp;
    a += b.ddlambda[j - 1] * n[j].dp;
  }
  const T ih = spline.inv_h_s;
  const T ih2 = ih * ih;
  return {normalized(n[0].q * inc), n[0].p + shift, ih * w, ih2 * wd, ih * v, ih2 * a};
}

template <int K, class T>
BALLA_HD Kinematics<T> evaluate(const SplineView<T>& spline, SegmentTime st) {
  return evaluate_u<K>(spline, st.segment, T(st.r_ns) / T(spline.h_ns));
}

}  // namespace balla::traj
