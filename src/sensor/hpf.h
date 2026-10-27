// The Prophesee high-pass filter on the comparator's reference (DESIGN §8.4 A.14, §8.5; SENSOR.md "High-pass filter").
//
// A.14: dr/dt = -l + b (x3 - r), b = 1 / tau_hp. With r_off the unfiltered reference, r(t_ref) - l (t - t_ref), the
// filter is a memory m = r - r_off with m(t_ref) = 0 and dm/dt = b (Delta_off - m), and the comparator compares
// Delta = Delta_off - m (Delta_off the unfiltered Delta of M3 E.1). m is driven by Delta~, Delta_off with the cell's
// level-0 noise Hermite hc0 in place of the walk's node Hermites, so that it depends only on the lattice and the input:
// M3 Theorem D.1 (a node's value does not depend on which nodes were refined) and the bridge margins hold. The cost,
// the sub-cell noise that m does not filter, is DESIGN §6.4's noise-memory row.
//
// The forms of one interval. From its anchor t_x (u = t - t_x, seconds), Delta~ is an ExpCubic
// d(u) = p(u) + sum_i C_i E_i(u) with the cascade's rates a_i (E_i = R_j(a_i u), j = 0 or 4: expcubic.h). Then
// G = Delta~ - M solves G' + b G = d', G(0) = d(0) - m_x, and exactly
//   G(u) = Q(u) + sum_i K_i E_i(u) + w4 u^4 E4(b u),   K_i = C_i a_i / (a_i - b),   E4(x) = 24 R_4(x) / x^4,
// Q a cubic (the exact cubic Taylor polynomial of G when j = 4); the b-mode is written so that it stays exact as
// b -> 0 (b = 0: G = d - m_x, M constant). The search function g = Q + sum K_i E_i is an ExpCubic of the cascade's
// class (no third exponential: expcubic.h is unchanged); the b-mode, |w4 u^4 E4| <= |w4| w^4 on [0, w], is kept out of
// the root search as a level error of at most eps_hp, added to the exclusion margins; an interval whose bound exceeds
// eps_hp is halved (each halving divides it by 16). The memory itself advances exactly:
//   M(u) = d(u) - G(u) = (m_x - sum_i D_i E_i(0)) + b (q0 u + q1 u^2 / 2 + q2 u^3 / 3) + sum_i D_i E_i(u)
//          - w4 u^4 E4(b u),   D_i = C_i - K_i = -C_i b / (a_i - b).
// No transcendental function forms them; M takes one per mode, E4 a polynomial where b u <= 0.1. With noise the
// kernel's intervals end at lattice nodes (2^16 ns), so b u <= 0.066 at validate()'s floor of 1 ms; without it an
// interval is the sub-step (up to a 2^23 ns piece) unless the bound halves it, and E4 may take its direct form. The
// floor also keeps b / (a_i - b) <= 1/9.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/pmath/special.h"
#include "sensor/expcubic.h"

namespace balla::sensor {

// The persistent filter state of one pixel (a side array, present only while the filter is on; fp64 whatever the
// kernel precision, as PixelState): the memory m at the model time and m_n, the part of it the noise drove (labels).
// Both are 0 at every reference setting.
struct HpfState {
  double m, mn;
};

namespace detail {

// E4(x) = 24 R_4(x) / x^4 = sum_k 24 (-x)^k / (k + 4)!, 1 at x = 0: nine terms hold it to 2e-17 relative for
// x <= 0.07 (b u <= 0.066: every interval with noise) and 4e-18 at 0.1, the direct form beyond (noise-off intervals
// longer than 0.1 tau_hp).
template <class T>
BALLA_HD T hpf_e4(T x) {
  if (x > T(0.1)) {
    const T x2 = x * x;
    return T(24) * pmath::resid_exp(4, x) / (x2 * x2);
  }
  return T(1) +
         x * (T(-1.0 / 5) +
              x * (T(1.0 / 30) +
                   x * (T(-1.0 / 210) +
                        x * (T(1.0 / 1680) +
                             x * (T(-1.0 / 15120) +
                                  x * (T(1.0 / 151200) + x * (T(-1.0 / 1663200) + x * T(1.0 / 19958400))))))));
}

// The filter over one interval from its anchor (u = 0): the search function, the memory's parts and the b-mode.
template <class T>
struct HpfForm {
  ExpCubic<T> g;  // G without its b-mode: Q + sum K_i E_i (the rates and order of d)
  T mp[4];        // M's polynomial: mp0 + mp1 u + mp2 u^2 + mp3 u^3
  T D[2];         // M's amplitudes of the modes E_i (0 where d has none)
  T d0;           // sum_i D_i E_i(0): mp0 = m_x - d0
  T w4;           // the b-mode: + w4 u^4 E4(b u) in G, - in M
  T b;
};

// The forms at the anchor of an interval: d = Delta~ (or the noise's part of it) as an ExpCubic in u at the anchor,
// m_x the memory there, b the filter's rate.
template <class T>
BALLA_HD HpfForm<T> hpf_form(const ExpCubic<T>& d, T m_x, T b) {
  HpfForm<T> f;
  f.g = d;
  f.b = b;
  T d0 = T(0), s3 = T(0);  // sum_i D_i E_i(0); the modes' cubic source (order 4)
  for (int i = 0; i < 2; ++i) {
    f.D[i] = T(0);
    if (d.C[i] == T(0)) continue;
    const T beta = b / (d.a[i] - b);
    f.D[i] = -(d.C[i] * beta);
    f.g.C[i] = d.C[i] - f.D[i];  // K_i = C_i a_i / (a_i - b), exactly C_i at b = 0
    if (d.order == 0) {
      d0 = d0 + f.D[i];
    } else {
      const T a2 = d.a[i] * d.a[i];
      s3 = s3 + f.D[i] * (a2 * a2) / T(6);
    }
  }
  // Q' + b Q + (b-mode) = d's polynomial source s = p' + s3 u^3, Q(0) = G(0) - sum K_i E_i(0); Q = p - (M's
  // polynomial) term by term, so that b = 0 gives Q = p exactly.
  f.mp[0] = m_x - d0;
  const T q0 = (d.c[0] + d0) - m_x;
  f.mp[1] = b * q0;
  const T q1 = d.c[1] - f.mp[1];
  f.mp[2] = b * q1 / T(2);
  const T q2 = d.c[2] - f.mp[2];
  f.mp[3] = b * q2 / T(3);
  const T q3 = d.c[3] - f.mp[3];
  f.g.c[0] = q0;
  f.g.c[1] = q1;
  f.g.c[2] = q2;
  f.g.c[3] = q3;
  f.w4 = (s3 - b * q3) / T(4);
  f.d0 = d0;
  return f;
}

// M(u) - m_x, the memory's change over u (seconds from the anchor): the kernel adds it to m in fp64 whatever its
// precision, so that rounding does not accumulate in m over the many intervals of tau_hp (an fp32 m random-walked by
// ~ulp(m) per interval: tens of us on 0.7 s periods at 1 s; SENSOR.md "High-pass filter").
template <class T>
BALLA_HD T hpf_increment(const HpfForm<T>& f, T u) {
  T dm = u * (f.mp[1] + u * (f.mp[2] + u * f.mp[3])) - f.d0;
  for (int i = 0; i < 2; ++i) {
    if (f.D[i] != T(0)) dm = dm + f.D[i] * pmath::resid_exp(f.g.order, f.g.a[i] * u);
  }
  if (f.w4 != T(0)) {
    const T u2 = u * u;
    dm = dm - f.w4 * (u2 * u2) * hpf_e4(f.b * u);
  }
  return dm;
}

// The memory M at u (seconds from the anchor), in T.
template <class T>
BALLA_HD T hpf_memory(const HpfForm<T>& f, T u) {
  return (f.mp[0] + f.d0) + hpf_increment(f, u);
}

// The b-mode's bound on [0, w] (E4 <= 1 for x >= 0).
template <class T>
BALLA_HD T hpf_bound(const HpfForm<T>& f, T w) {
  const T w2 = w * w;
  return abs(f.w4) * (w2 * w2);
}

// The leak's and the reference offset's part of the memory, in closed form at tau = t - t_ref (s): m_l =
// -dr R_1(b tau) + (l / b) R_2(b tau); `leak_only` gets its second term (the leak's own memory). 0 at b = 0.
template <class T>
BALLA_HD T hpf_leak_memory(T leak, T dr, T b, T tau, T& leak_only) {
  leak_only = T(0);
  if (!(b > T(0))) return T(0);
  const T x = b * tau;
  leak_only = leak == T(0) ? T(0) : (leak / b) * pmath::resid_exp(2, x);
  return (dr == T(0) ? T(0) : -(dr * pmath::resid_exp(1, x))) + leak_only;
}

}  // namespace detail

}  // namespace balla::sensor
