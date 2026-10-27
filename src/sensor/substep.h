// One deterministic sub-step of the photoreceptor cascade (M3 §B.3-B.4, DESIGN §8.3).
//
// Stage 1 advances z exactly (stage1.h). On the sub-step, x1 = -ln z is replaced by the cubic
// Hermite H through the exact endpoint values and slopes dx1/dt = k (I - 1/z) (M3 B.6-B.7). The
// sub-step is accepted only if
//   (a) the midpoint residual |H(h/2) - x1(h/2)| and the boundary-layer bound |xi0| E(P*) are both
//       <= eps_h / 1.5 (M3 B.10 with the review margin of DESIGN §8.3), and
//   (b) the stage-2/3 representation chosen below is well conditioned at this length.
// Stages 2-3 are then solved exactly for the cubic input (M3 B.11-B.14), giving
//   s3(s) = cubic(s) + C_o F(a2 s) + C_s F(a3 s),   s in [0, h],
// which the comparator consumes as an ExpCubic. With a_fast = max(a2, a3) and a_slow = min(a2, a3)
// (either pole may be the faster one), F is the plain exponential when a_fast h > 1 and the
// residual exponential R4 when a_fast h <= 1 (expcubic.h). Condition (b) rejects curved inputs in
// the window 1/a_fast < h < 1/(2 a_slow), where both splits would carry cancelling coefficients up
// to (1 / (a_slow h))^3; a linear x1 is always well conditioned in the plain split.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/pmath/special.h"
#include "sensor/expcubic.h"
#include "sensor/stage1.h"

namespace balla::sensor {

// Cascade constants in kernel units: k in (fA s)^-1, rates in s^-1.
template <class T>
struct CascadeConstants {
  T k;
  T a2, a3;       // 1/tau_o, 1/tau_sf in either order (validated: they differ by at least 10 %)
  T tau_o, tau_sf;
  T eps_accept;   // eps_h / 1.5
  T eps_drop;     // negligible: settledness tolerance and comparator-only term dropping
};

// The log states are stored relative to a per-pixel offset ln(i_scale): x1 = -ln(z i_scale), and s2,
// s3 on the same scale. Absolute log levels reach ~16 in fA units, and in fp32 their rounding,
// accumulated over the sub-steps of a stage-3 time constant, would exceed the 1e-5 log-e budget
// of the fp32 kernel (M3 O7); relative to the pixel's own level they stay O(1).
template <class T>
struct CascadeState {
  T z;   // stage-1 state, fA^-1 (absolute)
  T s2;  // deterministic output-node state, log-e relative to ln(i_scale)
  T s3;  // deterministic source-follower state (comparator input without noise), same offset
};

// A sub-step [t_a, t_b) of the deterministic pass with its stage-3 output in local s (seconds).
template <class T>
struct SubStep {
  int64_t t_a, t_b;
  ExpCubic<T> s3;
  T x1_a, x1_b;   // exact x1 at both ends (for the total-variation tracker of eps_PL)
  T int_ku;       // exact int k u over the step (M3 A.6)
  bool fallback;  // accepted at the 1 ns halving floor without meeting the tolerance
};

// Particular polynomial solution b of tau y' + y = sum c_m s^m (M3 B.11).
template <class T>
BALLA_HD void particular(const T* c, T tau, T* b) {
  b[3] = c[3];
  b[2] = c[2] - T(3) * tau * b[3];
  b[1] = c[1] - T(2) * tau * b[2];
  b[0] = c[0] - tau * b[1];
}

// M3 B.9: sup error of the cubic Hermite of a unit exponential with integrated rate P.
template <class T>
BALLA_HD T boundary_layer_factor(T P) {
  const T p2 = P * P;
  return min(p2 * p2 / T(384), T(0.15) * P + T(1));
}

// A stage-1 result already computed: z after h from z0 for the linear input I_a -> I_b (the pixel
// kernel's look-ahead computes the first piece of a cell before its sub-steps). try_substep takes it
// when its own arguments are the same bits; z0 < 0 never matches.
template <class T>
struct ZMemo {
  T z0 = T(-1), I_a = T(0), I_b = T(0), h = T(0), z = T(0);
  BALLA_HD bool matches(T z0_, T I_a_, T I_b_, T h_) const {
    return z0 == z0_ && I_a == I_a_ && I_b == I_b_ && h == h_;
  }
};

// Attempts the sub-step [0, h] with linear input I_a -> I_b from state st (log offset ln i_scale).
// On success fills sub (except t_a, t_b) and next, and returns true. `at_floor` forces acceptance
// (1 ns halving floor).
template <class T>
BALLA_HD bool try_substep(const CascadeConstants<T>& cc, const CascadeState<T>& st, T i_scale, T I_a, T I_b,
                          T h, bool at_floor, SubStep<T>& sub, CascadeState<T>& next,
                          const ZMemo<T>& memo = ZMemo<T>{}) {
  const T k = cc.k;
  const T z_a = st.z;
  const T z_b = memo.matches(z_a, I_a, I_b, h) ? memo.z : z_step(z_a, I_a, I_b, h, k);
  const T I_m = T(0.5) * (I_a + I_b);
  const T z_m = z_step(z_a, I_a, I_m, T(0.5) * h, k);
  const T x1a = -pmath::log(z_a * i_scale), x1b = -pmath::log(z_b * i_scale), x1m = -pmath::log(z_m * i_scale);
  const T da = k * (I_a - T(1) / z_a), db = k * (I_b - T(1) / z_b);
  // Cubic Hermite in monomial form (M3 B.7).
  T c[4];
  const T delta = (x1b - x1a) / h;
  c[0] = x1a;
  c[1] = da;
  c[2] = (T(3) * delta - T(2) * da - db) / h;
  c[3] = (da + db - T(2) * delta) / (h * h);
  const T hh = T(0.5) * h;
  const T mid_residual = (c[0] + hh * (c[1] + hh * (c[2] + hh * c[3]))) - x1m;
  const T P = integrated_pole(I_a, I_b, h, k);
  const T int_ku = pmath::log(z_b / z_a) + P;  // exact int k u (M3 A.6)
  // Needs h k I_a^2 normal in T down to h = 1 ns: a floor on the pixel dark current, which bounds I_a
  // below, of sqrt(min normal / (k 1 ns)) (1.1e-15 fA in fp32, 1.5e-150 fA in fp64 at k = 10 (fA s)^-1).
  const T xi0 = -pmath::log(z_a * I_a) + (I_b - I_a) / (h * k * I_a * I_a);
  const T P_star = max(P, int_ku);
  const bool accurate = abs(mid_residual) <= cc.eps_accept &&
                        abs(xi0) * boundary_layer_factor(P_star) <= cc.eps_accept;
  // Sub-tolerance curvature keeps the plain split well conditioned at any length: its blow-up
  // (tau / h)^3 then multiplies a term below eps_drop. The coefficients are kept exact, so nothing
  // is lost from the propagated state.
  const bool linear = abs(c[2]) * h * h + abs(c[3]) * h * h * h <= cc.eps_drop;
  const bool residual_form = max(cc.a2, cc.a3) * h <= T(1);
  const bool conditioned = residual_form || linear || T(2) * min(cc.a2, cc.a3) * h >= T(1);
  if (!at_floor && !(accurate && conditioned)) return false;

  // Stages 2-3 (M3 B.11-B.14). A2, A3 are the free-mode amplitudes; in the residual split they may
  // be huge but only ever multiply O((a h)^4) residual terms.
  T p2[4], p3[4];
  particular(c, cc.tau_o, p2);
  particular(p2, cc.tau_sf, p3);
  const T A2 = st.s2 - p2[0];
  const T A3 = st.s3 - p3[0];
  const T C_o = A2 * cc.a3 / (cc.a3 - cc.a2);
  const T C_s = A3 - C_o;
  // R_j(a2 h) and R_j(a3 h) of the split below (j = 4 residual, 0 plain), each evaluated once for
  // next.s2, next.s3 = g(h) and the drop test: the same arguments as g's own evaluation, so the same
  // bits (the special functions are out of line on the device, so the compiler no longer merges them).
  T r2, r3 = T(0);
  ExpCubic<T>& g = sub.s3;
  g.a[0] = cc.a2;
  g.a[1] = cc.a3;
  g.C[0] = C_o;
  g.C[1] = C_s;
  if (residual_form) {
    // Exact cubic Taylor polynomial of s2 and s3 at s = 0 from the cascade ODEs.
    T d1[4] = {c[0], c[1], T(2) * c[2], T(6) * c[3]};  // x1^(j)(0)
    T d2[4], d3[4];
    d2[0] = st.s2;
    d3[0] = st.s3;
    for (int j = 0; j < 3; ++j) {
      d2[j + 1] = cc.a2 * (d1[j] - d2[j]);
      d3[j + 1] = cc.a3 * (d2[j] - d3[j]);
    }
    g.c[0] = d3[0];
    g.c[1] = d3[1];
    g.c[2] = d3[2] / T(2);
    g.c[3] = d3[3] / T(6);
    g.order = 4;
    r2 = pmath::resid_exp(4, cc.a2 * h);
    r3 = pmath::resid_exp(4, cc.a3 * h);
    next.s2 = d2[0] + h * (d2[1] + h * (d2[2] / T(2) + h * (d2[3] / T(6)))) + A2 * r2;
  } else {
    for (int m = 0; m < 4; ++m) g.c[m] = p3[m];
    g.order = 0;
    r2 = pmath::exp(-cc.a2 * h);
    if (C_s != T(0)) r3 = pmath::exp(-cc.a3 * h);
    next.s2 = p2[0] + h * (p2[1] + h * (p2[2] + h * p2[3])) + A2 * r2;
  }
  next.s3 = g(h, r2, r3);
  next.z = z_b;
  // The comparator's copy may drop free modes that stay below eps_drop on the whole step (a pure
  // cubic is cheaper to solve); the state above was propagated with them, so nothing accumulates.
  if (abs(g.C[0]) * (residual_form ? r2 : T(1)) <= cc.eps_drop &&
      abs(g.C[1]) * (residual_form ? r3 : T(1)) <= cc.eps_drop) {
    g.C[0] = T(0);
    g.C[1] = T(0);
  }
  sub.x1_a = x1a;
  sub.x1_b = x1b;
  sub.int_ku = int_ku;
  sub.fallback = at_floor && !(accurate && conditioned);
  return true;
}

// A linear piece of photocurrent [t_a, t_b] (ns), I_a -> I_b (fA).
template <class T>
struct LinearPiece {
  int64_t t_a, t_b;
  T I_a, I_b;
  BALLA_HD T at(int64_t t) const {
    if (t == t_b) return I_b;
    return I_a + (I_b - I_a) * (static_cast<T>(t - t_a) / static_cast<T>(t_b - t_a));
  }
};

// Settled under constant input I: z = 1/I and s2 = s3 = x1 within eps_drop, so the cascade is
// stationary and a sub-step needs no transcendental evaluation.
template <class T>
BALLA_HD bool cascade_settled(const CascadeConstants<T>& cc, const CascadeState<T>& st, T x1, T I) {
  return abs(st.z * I - T(1)) <= cc.eps_drop && abs(st.s2 - x1) <= cc.eps_drop && abs(st.s3 - x1) <= cc.eps_drop;
}

struct SubStepInfo {
  bool settled;  // constant sub-step of a settled cascade
  int attempts;  // try_substep calls, including rejected halvings
};

// The next accepted sub-step from t_a inside piece pc (state st, x1 = -ln(st.z i_scale)): the whole rest
// of the piece if the cascade is settled under constant input, else the longest dyadic halving of
// the candidate [t_a, min(t_b, t_a + 2 h_hint)] that try_substep accepts (1 ns floor). h_hint
// carries the last accepted length inside a piece, so a boundary layer is not re-tried from the
// full remaining length after every step. Returns the end of the sub-step; for a settled step
// `next` is st itself.
template <class T>
BALLA_HD int64_t next_substep(const CascadeConstants<T>& cc, const CascadeState<T>& st, T i_scale, T x1,
                              const LinearPiece<T>& pc, int64_t t_a, T I_a, int64_t& h_hint,
                              SubStep<T>& sub, CascadeState<T>& next, SubStepInfo& info,
                              const ZMemo<T>& memo = ZMemo<T>{}) {
  info.attempts = 0;
  info.settled = pc.I_a == pc.I_b && cascade_settled(cc, st, x1, I_a);
  sub.t_a = t_a;
  if (info.settled) {
    sub.t_b = pc.t_b;
    const T h = static_cast<T>(pc.t_b - t_a) * T(1e-9);
    sub.s3 = ExpCubic<T>{{st.s3, T(0), T(0), T(0)}, {T(0), T(0)}, {cc.a2, cc.a3}, 0};
    sub.x1_a = sub.x1_b = x1;
    sub.int_ku = cc.k * h / st.z;
    sub.fallback = false;
    next = st;
    return pc.t_b;
  }
  int64_t t_b = h_hint > 0 ? min(pc.t_b, t_a + 2 * h_hint) : pc.t_b;
  while (true) {
    const int64_t h_ns = t_b - t_a;
    ++info.attempts;
    if (try_substep(cc, st, i_scale, I_a, pc.at(t_b), static_cast<T>(h_ns) * T(1e-9), h_ns <= 1, sub, next, memo)) break;
    t_b = t_a + h_ns / 2;
  }
  if (t_b < pc.t_b || h_hint > 0) h_hint = t_b - t_a;
  sub.t_b = t_b;
  return t_b;
}

}  // namespace balla::sensor
