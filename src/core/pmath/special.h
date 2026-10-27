// balla::pmath special functions: the residual exponentials R_j, erfcx and Dawson's integral, in
// fp32 and fp64 (DESIGN §4.6, §12.1).
//
// They are built from explog.h under the same rules (only +, -, *, /, sqrt and exact operations in
// a fixed order), so host and device round them identically. The pixel model integrates its
// cascade with them (DESIGN §8.3, M3 B.2-B.14).
//
// Accuracy (tests/cpp/test_special.cpp): R_j within 1e-15 relative in fp64 (4e-15 for R_3 and R_4
// just above y = 1, where the direct form cancels) and 2e-6 in fp32, of long-double references;
// erfcx and Dawson within 1e-14 relative in fp64 and 2e-6 in fp32.
//
// On the device each is one out-of-line copy per kernel (BALLA_DEVICE_NOINLINE): the sensor calls
// them from many sites, on branches a lit cell rarely takes, and inlined there they made up 57 % of
// its general kernel's code. A caller that needs the same value twice keeps it (substep.h).
#pragma once

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/pmath/explog.h"

namespace balla::pmath {

// Residual exponentials R_j(y) = e^{-y} - sum_{i<j} (-y)^i / i!, j = 0..4, for y >= 0 (DESIGN
// §8.3: the comparator works with these on short sub-steps, docs/dev/SENSOR.md "Representation of
// the comparator input"). They satisfy dR_j/dy = -R_{j-1} (with R_{-1} := R_0), R_j(y) = O(y^j),
// sign(R_j) = (-1)^j and |R_j| monotone increasing in y. For y <= 1 a Taylor series avoids the
// cancellation of the direct form.
template <class T>
BALLA_HD BALLA_DEVICE_NOINLINE T resid_exp(int j, T y) {
  if (j <= 0) return exp(-y);
  T yj = T(1), jfact = T(1);  // (-y)^j and j!
  for (int i = 1; i <= j; ++i) {
    yj = yj * (-y);
    jfact = jfact * static_cast<T>(i);
  }
  if (y > T(1)) {
    // Direct form: the cancellation costs at most one decimal digit, at y just above 1 (3.7e-15
    // relative for R_4 in fp64).
    T t = T(1), s = T(1);
    for (int i = 1; i < j; ++i) {
      t = t * (-y) / static_cast<T>(i);
      s = s + t;
    }
    return exp(-y) - s;
  }
  // R_j(y) = ((-y)^j / j!) (1 + (-y)/(j+1) (1 + (-y)/(j+2) (1 + ...))); the truncated term is
  // below 1/(N+2)! relative at y = 1: 2e-17 for N = 17 (fp64) and 2e-9 for N = 10 (fp32).
  constexpr int kTerms = sizeof(T) == 8 ? 17 : 10;
  T q = T(1);
  for (int i = kTerms; i >= 1; --i) q = T(1) - y * q / static_cast<T>(j + i);
  return yj * (q / jfact);
}

namespace detail {

// Weideman (1994) rational approximation of the Faddeeva function on the imaginary axis:
// erfcx(x) = w(ix) = 2 p(Z) / (L + x)^2 + 1 / (sqrt(pi) (L + x)), Z = (L - x) / (L + x), x >= 0.
// The coefficients of p are the published FFT construction (docs/dev/SENSOR.md), evaluated
// offline; N = 40 gives 7e-16 relative in fp64 and N = 24 gives 8e-11, below fp32 rounding. They
// are written as code, not as an array, so device code needs no constant table.
inline constexpr double kWeidemanL40 = 0x1.545ef5c0f51a0p+2;
inline constexpr double kWeidemanL24 = 0x1.07a673243a6b3p+2;
inline constexpr double kInvSqrtPi = 0x1.20dd750429b6dp-1;

template <class T>
BALLA_HD T weideman_p40(T Z) {
  T p = static_cast<T>(-0x1.f447ea2000000p-50);
  p = p * Z + static_cast<T>(0x1.5c3d78c99999ap-50);
  p = p * Z + static_cast<T>(0x1.9f0597ae33333p-47);
  p = p * Z + static_cast<T>(-0x1.79ece3c45999ap-48);
  p = p * Z + static_cast<T>(-0x1.3e74182bb9000p-44);
  p = p * Z + static_cast<T>(0x1.eff14ed7b7333p-47);
  p = p * Z + static_cast<T>(0x1.fe7df2a515ae6p-42);
  p = p * Z + static_cast<T>(0x1.0ef56cfc644f3p-43);
  p = p * Z + static_cast<T>(-0x1.993999999999ap-39);
  p = p * Z + static_cast<T>(-0x1.7fe6666666666p-39);
  p = p * Z + static_cast<T>(0x1.37a199999999ap-36);
  p = p * Z + static_cast<T>(0x1.3177333333333p-35);
  p = p * Z + static_cast<T>(-0x1.8e3fccccccccdp-34);
  p = p * Z + static_cast<T>(-0x1.87c8200000000p-32);
  p = p * Z + static_cast<T>(0x1.cfaf99999999ap-33);
  p = p * Z + static_cast<T>(0x1.9ec2b5ccccccdp-29);
  p = p * Z + static_cast<T>(0x1.bea44b0000000p-29);
  p = p * Z + static_cast<T>(-0x1.3aa8ef4666666p-26);
  p = p * Z + static_cast<T>(-0x1.10ce7cef33333p-24);
  p = p * Z + static_cast<T>(0x1.e7dc943400000p-27);
  p = p * Z + static_cast<T>(0x1.3d67caa3b999ap-21);
  p = p * Z + static_cast<T>(0x1.8e3de31305333p-20);
  p = p * Z + static_cast<T>(-0x1.1e27ead3ba19ap-20);
  p = p * Z + static_cast<T>(-0x1.2e1d6585ae4dep-16);
  p = p * Z + static_cast<T>(-0x1.d50873c0a9b66p-15);
  p = p * Z + static_cast<T>(-0x1.4a752dec26b46p-15);
  p = p * Z + static_cast<T>(0x1.cd2bcc00e1ff0p-12);
  p = p * Z + static_cast<T>(0x1.629a596ed49c6p-9);
  p = p * Z + static_cast<T>(0x1.49424ba5ffea3p-7);
  p = p * Z + static_cast<T>(0x1.de75e8cc4682ep-6);
  p = p * Z + static_cast<T>(0x1.26308597ddc86p-4);
  p = p * Z + static_cast<T>(0x1.3d86fe9e6a9ebp-3);
  p = p * Z + static_cast<T>(0x1.33178328c1600p-2);
  p = p * Z + static_cast<T>(0x1.0da572e19e018p-1);
  p = p * Z + static_cast<T>(0x1.b1c67c927e9f3p-1);
  p = p * Z + static_cast<T>(0x1.41a238f009d09p+0);
  p = p * Z + static_cast<T>(0x1.b9b2b4b25d642p+0);
  p = p * Z + static_cast<T>(0x1.19cb343b58fc6p+1);
  p = p * Z + static_cast<T>(0x1.4edadccb56adbp+1);
  p = p * Z + static_cast<T>(0x1.7326e55b41214p+1);
  return p;
}

template <class T>
BALLA_HD T weideman_p24(T Z) {
  T p = static_cast<T>(-0x1.4ce0600000000p-33);
  p = p * Z + static_cast<T>(0x1.510e836600000p-28);
  p = p * Z + static_cast<T>(0x1.6ddffcaa55555p-30);
  p = p * Z + static_cast<T>(-0x1.0268d4267eaabp-25);
  p = p * Z + static_cast<T>(-0x1.488495943caabp-26);
  p = p * Z + static_cast<T>(0x1.926721ecdb12bp-23);
  p = p * Z + static_cast<T>(0x1.13c3e82cc4f95p-22);
  p = p * Z + static_cast<T>(-0x1.236d2231f0788p-20);
  p = p * Z + static_cast<T>(-0x1.97df93222aaabp-19);
  p = p * Z + static_cast<T>(0x1.15cb6931c22e9p-18);
  p = p * Z + static_cast<T>(0x1.ff383b780d8c0p-16);
  p = p * Z + static_cast<T>(0x1.9836a2388ff37p-16);
  p = p * Z + static_cast<T>(-0x1.b3204b21936cfp-13);
  p = p * Z + static_cast<T>(-0x1.99d134d317025p-11);
  p = p * Z + static_cast<T>(-0x1.02cf99ae1a9cbp-11);
  p = p * Z + static_cast<T>(0x1.974e8112c3b17p-8);
  p = p * Z + static_cast<T>(0x1.144306b8259ebp-5);
  p = p * Z + static_cast<T>(0x1.bbf440cf74c90p-4);
  p = p * Z + static_cast<T>(0x1.0fde4987e60c4p-2);
  p = p * Z + static_cast<T>(0x1.127d87330cc1ep-1);
  p = p * Z + static_cast<T>(0x1.d9f67e174ac80p-1);
  p = p * Z + static_cast<T>(0x1.6512e6f0fb105p+0);
  p = p * Z + static_cast<T>(0x1.db35978e18188p+0);
  p = p * Z + static_cast<T>(0x1.1953710ecfe97p+1);
  return p;
}

// One term of Rybicki's sum: c_i (e1 / d1 + 1 / (d2 e1)), then advance to the next odd n.
template <class T>
BALLA_HD void rybicki_term(T c, T e2, T& sum, T& e1, T& d1, T& d2) {
  sum = sum + c * (e1 / d1 + T(1) / (d2 * e1));
  d1 = d1 + T(2);
  d2 = d2 - T(2);
  e1 = e1 * e2;
}

// Rybicki's exponentially convergent sum for Dawson's integral with step h = 1/4:
// F(x) ~ (1/sqrt(pi)) sum_{n odd} e^{-(x - n h)^2} / n; truncation error ~ e^{-(pi/2h)^2} = 7e-18.
// The constants are c_i = exp(-((2i + 1) h)^2); 13 terms reach 1e-17.
inline constexpr double kRybickiH = 0.25;

template <class T>
BALLA_HD T rybicki_sum(T e1, T e2, T d1, T d2) {
  T sum = T(0);
  rybicki_term(static_cast<T>(0x1.e0fabfbc702a4p-1), e2, sum, e1, d1, d2);
  rybicki_term(static_cast<T>(0x1.23ba930c1568bp-1), e2, sum, e1, d1, d2);
  rybicki_term(static_cast<T>(0x1.ad48bc25771c7p-3), e2, sum, e1, d1, d2);
  rybicki_term(static_cast<T>(0x1.7f251ab1af77bp-5), e2, sum, e1, d1, d2);
  rybicki_term(static_cast<T>(0x1.9ed300c108a17p-8), e2, sum, e1, d1, d2);
  rybicki_term(static_cast<T>(0x1.1068222437d65p-11), e2, sum, e1, d1, d2);
  rybicki_term(static_cast<T>(0x1.b1fea4fbb871ap-16), e2, sum, e1, d1, d2);
  rybicki_term(static_cast<T>(0x1.a3604afdb0929p-21), e2, sum, e1, d1, d2);
  rybicki_term(static_cast<T>(0x1.eb97d4afc3bd3p-27), e2, sum, e1, d1, d2);
  rybicki_term(static_cast<T>(0x1.5d82c26ce1c09p-33), e2, sum, e1, d1, d2);
  rybicki_term(static_cast<T>(0x1.2d7026e60ab5ep-40), e2, sum, e1, d1, d2);
  rybicki_term(static_cast<T>(0x1.3b5e5c86b9440p-48), e2, sum, e1, d1, d2);
  rybicki_term(static_cast<T>(0x1.903daec8f0fb0p-57), e2, sum, e1, d1, d2);
  return sum;
}

}  // namespace detail

// erfcx(x) = e^{x^2} erfc(x) for x >= 0 (negative arguments are clamped to 0; the sensor never
// needs them). 0 < erfcx(x) <= 1; erfcx(+inf) = 0, and NaN propagates.
template <class T>
BALLA_HD BALLA_DEVICE_NOINLINE T erfcx(T x) {
  if (!(x > T(0) && x < infinity<T>())) return x > T(0) ? T(0) : x == x ? T(1) : x;
  constexpr bool kDouble = sizeof(T) == 8;
  const T L = static_cast<T>(kDouble ? detail::kWeidemanL40 : detail::kWeidemanL24);
  const T lx = L + x;
  const T Z = (L - x) / lx;
  const T p = kDouble ? detail::weideman_p40(Z) : detail::weideman_p24(Z);
  return T(2) * p / (lx * lx) + static_cast<T>(detail::kInvSqrtPi) / lx;
}

// Dawson's integral F(x) = e^{-x^2} int_0^x e^{t^2} dt (odd; 0 <= F <= 0.5410 for x >= 0).
template <class T>
BALLA_HD BALLA_DEVICE_NOINLINE T dawson(T x) {
  const T ax = abs(x);
  constexpr bool kDouble = sizeof(T) == 8;
  // Taylor series x (1 - (2x^2/3) (1 - (2x^2/5) (1 - ...))) = sum_n (-2)^n x^(2n+1) / (2n+1)!!
  // below the switch; 12 terms leave < 1e-17 at 0.2 (fp64) and 7e-9 at 1 (fp32), where the
  // alternating series is still free of cancellation. Rybicki's sum loses ~4 bits in fp32 there.
  const T taylor_limit = kDouble ? T(0.2) : T(1);
  if (ax < taylor_limit) {
    const T two_x2 = T(2) * x * x;
    T q = T(1);
    for (int n = 12; n >= 1; --n) q = T(1) - two_x2 * q / static_cast<T>(2 * n + 1);
    return x * q;
  }
  if (!(ax <= T(1e8))) return ax == ax ? copysign(T(0.5) / ax, x) : x;  // F = 1/(2x) (1 + 1/(2x^2) + ...); NaN
  const T h = static_cast<T>(detail::kRybickiH);
  const int n0 = 2 * static_cast<int>(T(0.5) * ax / h + T(0.5));
  const T xp = ax - static_cast<T>(n0) * h;
  const T e1 = exp(T(2) * xp * h);
  const T sum = detail::rybicki_sum(e1, e1 * e1, static_cast<T>(n0 + 1), static_cast<T>(n0 - 1));
  return copysign(static_cast<T>(detail::kInvSqrtPi) * exp(-xp * xp) * sum, x);
}

}  // namespace balla::pmath
