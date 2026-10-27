// The comparator input on an elementary interval and its exact threshold crossings (M3 §C,
// DESIGN §8.5).
//
// On every elementary interval the comparator input Delta(t) = x3(t) - r(t) is a cubic plus two
// decaying exponentials with the cascade's rates a2 = 1/tau_o and a3 = 1/tau_sf (M3 C.1): the
// deterministic stage-3 output is a cubic plus two modes (M3 B.14), the noise is the cubic Hermite
// of its tree node (M3 D.7), and the reference is linear. ExpCubic stores it as
//   g(u) = c0 + c1 u + c2 u^2 + c3 u^3 + C0 R_j(a0 u) + C1 R_j(a1 u),   u in [0, w] seconds,
// where R_j is the residual exponential of order j (special.h): j = 0 is the plain exponential
// e^{-a u}; j = 4 is e^{-a u} minus its cubic Taylor polynomial. Both describe the same function
// class (the polynomial parts are re-split), but they condition differently: on sub-steps much
// shorter than the time constants the plain split carries huge, cancelling coefficients
// ~ (tau/h)^3, while the residual split carries the exact cubic Taylor polynomial plus terms of
// size O((a h)^4) (docs/dev/SENSOR.md, "Representation of the comparator input"). Each term
// C R_j(a u) is monotone in u, which keeps the exclusion hull of M3 C.3 valid in both forms.
//
// Root isolation follows the Polya-Rolle recursion of M3 §C.4: the fourth derivative is a sum of
// two plain exponentials with at most one zero in closed form (M3 C.4), and each lower derivative
// is monotone between consecutive zeros of the next one. A cheap hull of the next derivative skips
// a level whenever it already proves monotonicity. Only sign changes are tracked below the top
// level: a derivative that touches zero without changing sign does not break monotonicity of its
// integral. At the top level the crossing is resolved on the integer-ns grid (first_firing).
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/pmath/special.h"

namespace balla::sensor {

template <class T>
struct ExpCubic {
  T c[4];
  T C[2];
  T a[2];
  int order;  // residual order j of both exponential terms: 0 (plain) or 4 (residual)

  BALLA_HD T poly(T u) const { return c[0] + u * (c[1] + u * (c[2] + u * c[3])); }

  BALLA_HD T exp_term(int i, T u) const {
    return C[i] == T(0) ? T(0) : C[i] * pmath::resid_exp(order, a[i] * u);
  }

  BALLA_HD T operator()(T u) const { return poly(u) + exp_term(0, u) + exp_term(1, u); }

  // operator() with R_order(a_i u) already evaluated as r0, r1: the same operations, so the same bits.
  BALLA_HD T operator()(T u, T r0, T r1) const {
    return poly(u) + (C[0] == T(0) ? T(0) : C[0] * r0) + (C[1] == T(0) ? T(0) : C[1] * r1);
  }

  BALLA_HD bool has_exp() const { return C[0] != T(0) || C[1] != T(0); }
};

// n-th derivative (0 <= n <= 4): d^n/du^n C R_j(a u) = C (-a)^n R_{max(j-n, 0)}(a u).
template <class T>
BALLA_HD ExpCubic<T> derivative(const ExpCubic<T>& g, int n) {
  ExpCubic<T> d = g;
  for (int k = 0; k < n; ++k) {
    d.c[0] = d.c[1];
    d.c[1] = T(2) * d.c[2];
    d.c[2] = T(3) * d.c[3];
    d.c[3] = T(0);
    d.C[0] = -d.a[0] * d.C[0];
    d.C[1] = -d.a[1] * d.C[1];
  }
  d.order = g.order > n ? g.order - n : 0;
  return d;
}

// g(d + u) as an ExpCubic in u, for 0 <= d. For a residual term of order j,
//   R_j(a (d + u)) = e^{-a d} R_j(a u) + sum_{m<j} ((-a)^m / m!) R_{j-m}(a d) u^m,
// so the shift moves well-conditioned products (large C times small R) into the cubic.
template <class T>
BALLA_HD ExpCubic<T> shifted(const ExpCubic<T>& g, T d) {
  ExpCubic<T> s = g;
  s.c[0] = g.c[0] + d * (g.c[1] + d * (g.c[2] + d * g.c[3]));
  s.c[1] = g.c[1] + d * (T(2) * g.c[2] + d * (T(3) * g.c[3]));
  s.c[2] = g.c[2] + d * (T(3) * g.c[3]);
  s.c[3] = g.c[3];
  for (int i = 0; i < 2; ++i) {
    if (g.C[i] == T(0)) continue;
    const T ad = g.a[i] * d;
    s.C[i] = g.C[i] * pmath::exp(-ad);
    T coef = g.C[i];  // C (-a)^m / m!
    for (int m = 0; m < g.order; ++m) {
      s.c[m] = s.c[m] + coef * pmath::resid_exp(g.order - m, ad);
      coef = coef * (-g.a[i]) / static_cast<T>(m + 1);
    }
  }
  return s;
}

template <class T>
struct Range {
  T lo, hi;
};

// Range of the cubic c on [u0, u1] from its Bernstein coefficients (M3 C.3).
template <class T>
BALLA_HD Range<T> cubic_range(const T* c, T u0, T u1) {
  const T w = u1 - u0;
  const T q0 = c[0] + u0 * (c[1] + u0 * (c[2] + u0 * c[3]));
  const T q1 = (c[1] + u0 * (T(2) * c[2] + u0 * (T(3) * c[3]))) * w;
  const T q2 = (c[2] + u0 * (T(3) * c[3])) * (w * w);
  const T q3 = c[3] * (w * w * w);
  const T b0 = q0;
  const T b1 = q0 + q1 / T(3);
  const T b2 = q0 + (T(2) * q1 + q2) / T(3);
  const T b3 = q0 + q1 + q2 + q3;
  return {min(min(b0, b1), min(b2, b3)), max(max(b0, b1), max(b2, b3))};
}

// Conservative range of g on [u0, u1]: Bernstein hull of the cubic plus the endpoint values of
// each monotone exponential term (M3 C.3). Never violated in M3 V2 T3.
template <class T>
BALLA_HD Range<T> range(const ExpCubic<T>& g, T u0, T u1) {
  Range<T> r = cubic_range(g.c, u0, u1);
  for (int i = 0; i < 2; ++i) {
    if (g.C[i] == T(0)) continue;
    const T e0 = g.exp_term(i, u0), e1 = g.exp_term(i, u1);
    r.lo = r.lo + min(e0, e1);
    r.hi = r.hi + max(e0, e1);
  }
  return r;
}

// Sorted points in an interval (at most 5 are ever needed: M3 C.2 bounds the zeros of every level).
template <class T>
struct RootList {
  T v[6];
  int n = 0;
  BALLA_HD void push(T x) { v[n++] = x; }
};

// Tolerance for interior roots of derivatives: they only split [lo, hi] into monotone pieces.
template <class T>
BALLA_HD T root_tolerance(T hi) {
  constexpr T kAbs = sizeof(T) == 8 ? T(1e-13) : T(1e-11);  // seconds
  return max(kAbs, T(16) * FloatTraits<T>::kEpsilon * abs(hi));
}

// ITP root of f on [a, b] with fa * fb < 0 (Oliveira & Takahashi 2020; kappa1 = 0.2 / (b - a),
// kappa2 = 2, n0 = 1): at most one evaluation more than bisection, superlinear on simple roots.
template <class T, class F>
BALLA_HD T itp_root(const F& f, T a, T b, T fa, T fb, T tol) {
  const T width0 = b - a;
  const T k1 = T(0.2) / width0;
  int n_half = 0;
  for (T w = width0; w > T(2) * tol && n_half < 200; w = w * T(0.5)) ++n_half;
  const int n_max = n_half + 1;
  T scale = tol;  // tol * 2^(n_max - j), maintained without pow()
  for (int j = 0; j < n_max; ++j) scale = scale * T(2);
  for (int j = 0; j < 400 && b - a > T(2) * tol; ++j) {
    const T xh = T(0.5) * (a + b);
    const T r = scale - T(0.5) * (b - a);
    const T xf = (fb * a - fa * b) / (fb - fa);
    const T sg = xh - xf >= T(0) ? T(1) : T(-1);
    const T delta = k1 * (b - a) * (b - a);
    const T xt = delta <= abs(xh - xf) ? xf + sg * delta : xh;
    const T xi = abs(xt - xh) <= r ? xt : xh - sg * r;
    if (!(xi > a && xi < b)) break;  // no representable progress left
    const T yi = f(xi);
    if (yi == T(0)) return xi;
    if ((yi > T(0)) == (fa > T(0))) {
      a = xi;
      fa = yi;
    } else {
      b = xi;
      fb = yi;
    }
    scale = scale * T(0.5);
  }
  return T(0.5) * (a + b);
}

namespace detail {

// Zero of C0 a0^4 e^{-a0 u} + C1 a1^4 e^{-a1 u} (the fourth derivative of g) in (lo, hi), formed in
// log space (M3 C.4).
template <class T>
BALLA_HD void fourth_derivative_root(const ExpCubic<T>& g, T lo, T hi, RootList<T>& out) {
  if (g.C[0] == T(0) || g.C[1] == T(0)) return;
  if ((g.C[0] > T(0)) == (g.C[1] > T(0))) return;
  if (g.a[0] == g.a[1]) return;
  const T ln_ratio = pmath::log(abs(g.C[1]) / abs(g.C[0])) + T(4) * pmath::log(g.a[1] / g.a[0]);
  const T u = ln_ratio / (g.a[1] - g.a[0]);
  if (u > lo && u < hi) out.push(u);
}

// Sign-change zeros of g^(N) in (lo, hi), sorted (the Polya-Rolle recursion, unrolled by template).
template <int N, class T>
BALLA_HD void level_roots(const ExpCubic<T>& g, T lo, T hi, T tol, RootList<T>& out) {
  if constexpr (N >= 4) {
    fourth_derivative_root(g, lo, hi, out);
  } else {
    const ExpCubic<T> f = derivative(g, N);
    const ExpCubic<T> fd = derivative(g, N + 1);
    RootList<T> pts;
    pts.push(lo);
    const Range<T> rd = range(fd, lo, hi);
    if (rd.lo < T(0) && rd.hi > T(0)) {
      RootList<T> crit;
      level_roots<N + 1>(g, lo, hi, tol, crit);
      for (int i = 0; i < crit.n; ++i) pts.push(crit.v[i]);
    }
    pts.push(hi);
    T fa = f(pts.v[0]);
    for (int i = 0; i + 1 < pts.n; ++i) {
      const T a = pts.v[i], b = pts.v[i + 1];
      const T fb = f(b);
      if ((fa < T(0) && fb > T(0)) || (fa > T(0) && fb < T(0))) {
        out.push(itp_root(f, a, b, fa, fb, tol));
      }
      fa = fb;
    }
  }
}

}  // namespace detail

// Interior critical points of g on (lo, hi), i.e. the boundaries of its monotone pieces.
template <class T>
BALLA_HD RootList<T> critical_points(const ExpCubic<T>& g, T lo, T hi) {
  RootList<T> crit;
  detail::level_roots<1>(g, lo, hi, root_tolerance(hi), crit);
  return crit;
}

// First integer k in [k_lo, k_hi] (ns offsets from the interval start, u = k * 1e-9 s) with
// G(u_k) >= 0, or -1. G is monotone between consecutive breakpoints {k_lo, crit..., k_hi}; per piece
// the predicate "G >= 0" is monotone on the integers, so a safeguarded interpolation search finds
// the transition. This defines event times as the first integer nanosecond at which the closed
// firing condition holds (docs/dev/SENSOR.md, "Event times").
template <class T>
BALLA_HD int64_t first_firing(const ExpCubic<T>& G, const RootList<T>& crit, int64_t k_lo,
                              int64_t k_hi) {
  if (k_lo > k_hi) return -1;
  const T kNs = T(1e-9);
  auto Gk = [&](int64_t k) { return G(static_cast<T>(k) * kNs); };
  // The integers of monotone piece i are [piece_lo, floor(crit_i)]; the next piece starts one
  // integer later, so no integer is attributed to a piece it does not lie in.
  int64_t piece_lo = k_lo;
  for (int i = 0; i <= crit.n && piece_lo <= k_hi; ++i) {
    const int64_t piece_hi =
        i < crit.n ? min(static_cast<int64_t>(floor(crit.v[i] / kNs)), k_hi) : k_hi;
    if (piece_hi < piece_lo) continue;
    int64_t a = piece_lo, b = piece_hi;
    piece_lo = piece_hi + 1;
    T ga = Gk(a);
    if (ga >= T(0)) return a;
    if (b == a) continue;
    T gb = Gk(b);
    if (gb >= T(0)) {
      // G(a) < 0 <= G(b): search the transition, alternating interpolation and bisection.
      bool interpolate = true;
      while (b - a > 1) {
        int64_t m;
        if (interpolate) {
          const T frac = (-ga) / (gb - ga);
          m = a + static_cast<int64_t>(frac * static_cast<T>(b - a));
          m = min(max(m, a + 1), b - 1);
        } else {
          m = a + (b - a) / 2;
        }
        const int64_t width = b - a;
        const T gm = Gk(m);
        if (gm >= T(0)) {
          b = m;
          gb = gm;
        } else {
          a = m;
          ga = gm;
        }
        interpolate = (b - a) * 2 <= width;  // bisect next if interpolation stalled
      }
      return b;
    }
  }
  return -1;
}

}  // namespace balla::sensor
