// Stage 1 of the photoreceptor cascade in closed form (M3 §B.2, DESIGN §8.3).
//
// The large-signal photoreceptor dz/dt = k (1 - I(t) z), x1 = -ln z (M3 A.4) is linear in z, so on
// a piece of piecewise-linear photocurrent I(tau) = I0 + s tau it has the exact solution (M3 B.1)
//   z(h) = z0 e^{-P} + k int_0^h e^{-(P(h) - P(sigma))} dsigma,   P(tau) = k (I0 tau + s tau^2 / 2),
// evaluated through one of four branches:
//   - P(h) <= 1/2: 8-point Gauss-Legendre on the integral (M3 B.5), which removes the only
//     cancellation of the closed forms (relative error ~ P^16 / 16! < 1e-17); in fp32, a small slope
//     takes a first-order expansion in the curvature instead (below fp32 rounding; ENGINE_CPP I6');
//   - constant input (relative slope below one ulp): zero-order hold (M3 B.2);
//   - brightening: Dawson's integral (M3 B.3);
//   - dimming: the scaled complementary error function erfcx (M3 B.4).
// Every term is bounded (0 <= F <= 0.541, 0 < erfcx <= 1, e^{-P} <= 1), so no branch overflows.
//
// Units (M3 §B.9): current in fA, time in s, k in (fA s)^-1 (k = 10 for 1e16 A^-1 s^-1), so
// z is in fA^-1 and every argument stays O(1)-scaled.
#pragma once

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/pmath/special.h"

namespace balla::sensor {

namespace detail {

// One symmetric pair of Gauss-Legendre nodes u = (1 -+ xi)/2 on [0, 1] with weight w (sum w = 1
// over the four pairs, i.e. the usual weights halved).
template <class T>
BALLA_HD T gl8_pair(T kh, T I0, T dI, double xi, double w) {
  const T u_lo = static_cast<T>(0.5 - 0.5 * xi), u_hi = static_cast<T>(0.5 + 0.5 * xi);
  // Exponent P(h) - P(sigma) = k h (1 - u) (I0 + dI (1 + u) / 2) with sigma = u h: no cancellation.
  const T g_lo = kh * (T(1) - u_lo) * (I0 + T(0.5) * dI * (T(1) + u_lo));
  const T g_hi = kh * (T(1) - u_hi) * (I0 + T(0.5) * dI * (T(1) + u_hi));
  return static_cast<T>(w) * (pmath::exp(-g_lo) + pmath::exp(-g_hi));
}

}  // namespace detail

// Integrated input pole P = int_0^h k I = k h (I0 + I1) / 2 of a linear piece.
template <class T>
BALLA_HD T integrated_pole(T I0, T I1, T h, T k) {
  return k * h * (T(0.5) * (I0 + I1));
}

// Exact z(h) for the linear photocurrent I0 -> I1 over [0, h], I0, I1 > 0, h >= 0 (M3 B.2-B.5).
template <class T>
BALLA_HD T z_step(T z0, T I0, T I1, T h, T k) {
  const T kh = k * h;
  const T Im = T(0.5) * (I0 + I1);
  const T P = kh * Im;
  const T eP = pmath::exp(-P);
  const T dI = I1 - I0;
  if (P <= T(0.5)) {
    if constexpr (sizeof(T) == sizeof(float)) {
      // Small-slope branch (fp32 only; ENGINE_CPP I6', a numerical tolerance item). With r = h - sigma the
      // integral is int_0^h e^{-a r + b r^2} dr, a = k I1, b = k s / 2; first order in the curvature term b r^2:
      // (1 - e^{-a h}) / a + b h^3 g(a h), g(x) = x^-3 int_0^x v^2 e^{-v} dv = 1/3 - x/4 + x^2/10 - x^3/36 + x^4/168
      // - ... Where it applies (P <= 1/2, P |dI| <= 2.4e-4 Im) the dropped terms are below 1e-8 relative, under the
      // fp32 rounding of GL8 itself: one exp and one expm1 instead of eight exponentials. The fp64 kernel (the
      // oracle) keeps GL8.
      if (P > T(0) && P * abs(dI) <= T(2.4e-4) * Im) {
        const T x = kh * I1;
        const T b = T(0.5) * k * dI / h;
        const T one_m_E = -pmath::expm1(-x);
        const T g = T(1) / T(3) + x * (T(-0.25) + x * (T(0.1) + x * (T(-1) / T(36) + x * (T(1) / T(168)))));
        const T J = one_m_E / x * h + b * h * h * h * g;
        return z0 * eP + k * J;
      }
    }
    T J = detail::gl8_pair(kh, I0, dI, 0x1.77ac94f3c7344p-3, 0x1.736360b19933dp-3);
    J = J + detail::gl8_pair(kh, I0, dI, 0x1.0d129583284b4p-1, 0x1.413c50a25560ep-3);
    J = J + detail::gl8_pair(kh, I0, dI, 0x1.97e4ab249f41ep-1, 0x1.c76fb531d2b94p-4);
    J = J + detail::gl8_pair(kh, I0, dI, 0x1.ebab1cb0acc66p-1, 0x1.9ea1d04ca03aep-5);
    return z0 * eP + kh * J;
  }
  // Holding the mean current errs by up to |dI| / (2 Im) once the piece is long (z -> 1/I(h)), so
  // the zero-order hold is used only below one ulp of relative slope. The Dawson and erfcx branches
  // stay accurate for arbitrarily small slopes (their arguments grow like sqrt(P / slope)).
  if (abs(dI) <= FloatTraits<T>::kEpsilon * Im) return T(1) / Im + (z0 - T(1) / Im) * eP;
  const T s = abs(dI) / h;
  const T r = sqrt(k / (T(2) * s));            // beta / s, with beta = sqrt(k s / 2)
  const T k_over_beta = sqrt(T(2) * k / s);    // k / beta
  if (dI > T(0)) {
    // P(tau) = w(tau)^2 - w0^2 with w = beta I / s (M3 B.3).
    return z0 * eP + k_over_beta * (pmath::dawson(r * I1) - pmath::dawson(r * I0) * eP);
  }
  // P(tau) = v0^2 - v(tau)^2 with v = beta I / |s| (M3 B.4).
  constexpr double kHalfSqrtPi = 0x1.c5bf891b4ef6ap-1;
  return z0 * eP + k_over_beta * static_cast<T>(kHalfSqrtPi) *
                       (pmath::erfcx(r * I1) - pmath::erfcx(r * I0) * eP);
}

}  // namespace balla::sensor
