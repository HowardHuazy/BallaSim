// Deterministic exponentials, logarithms and special functions (core/pmath/explog.h, special.h,
// smootherstep.h) against libm and long-double references.
#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <random>

#include "core/pmath/explog.h"
#include "core/pmath/smootherstep.h"
#include "core/pmath/special.h"
#include "test_util.h"

namespace pmath = balla::pmath;
using balla::test::ulp_distance;

TEST_CASE("exp and log are within 2 ulp (fp64) and 3 ulp (fp32) of libm") {
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> ux(-700.0, 700.0), ul(-300.0, 300.0);
  int64_t worst_exp64 = 0, worst_log64 = 0, worst_exp32 = 0, worst_log32 = 0;
  for (int i = 0; i < 200000; ++i) {
    const double x = ux(rng);
    worst_exp64 = std::max(worst_exp64, ulp_distance(pmath::exp(x), std::exp(x)));
    const double y = std::pow(10.0, ul(rng));
    worst_log64 = std::max(worst_log64, ulp_distance(pmath::log(y), std::log(y)));
    const float xf = static_cast<float>(x / 8.1);
    const float exp_ref = static_cast<float>(std::exp(static_cast<double>(xf)));
    worst_exp32 = std::max(worst_exp32, ulp_distance(pmath::exp(xf), exp_ref));
    const float yf = static_cast<float>(std::pow(10.0, ul(rng) / 8.0));
    const float log_ref = static_cast<float>(std::log(static_cast<double>(yf)));
    worst_log32 = std::max(worst_log32, ulp_distance(pmath::log(yf), log_ref));
  }
  CHECK(worst_exp64 <= 2);
  CHECK(worst_log64 <= 2);
  CHECK(worst_exp32 <= 3);
  CHECK(worst_log32 <= 3);
}

TEST_CASE("exp flushes to exactly zero instead of producing subnormals") {
  CHECK(pmath::exp(-708.39) > 0.0);
  CHECK(pmath::exp(-708.39) >= 2.2250738585072014e-308);
  CHECK(pmath::exp(-708.4) == 0.0);
  CHECK(pmath::exp(-1e300) == 0.0);
  CHECK(pmath::exp(-87.33f) >= 1.17549435e-38f);
  CHECK(pmath::exp(-87.34f) == 0.0f);
  CHECK(std::isinf(pmath::exp(710.0)));
  CHECK(std::isnan(pmath::exp(std::nan(""))));
  CHECK(pmath::log(0.0) == -INFINITY);
  CHECK(std::isnan(pmath::log(-1.0)));
  CHECK(pmath::exp(0.0) == 1.0);
  CHECK(pmath::log(1.0) == 0.0);
}

TEST_CASE("expm1, log1p and phi1 keep relative accuracy near zero") {
  // 0.3735: expm1's worst case (4 ulp), exp(x) - 1 just above the kernel's |x| < 0.35.
  for (double x : {1e-300, -1e-20, 1e-12, -3e-6, 0.01, -0.2, 0.34, 0.36, 0.37347535370818236, -0.9, 3.0, -40.0}) {
    CHECK(ulp_distance(pmath::expm1(x), std::expm1(x)) <= 4);
    if (x > -1) CHECK(ulp_distance(pmath::log1p(x), std::log1p(x)) <= 4);
    // phi1 = expm1(x) / x: 6.1e-16 relative at worst (near 0.3735), plus the double reference's rounding.
    if (x != 0) CHECK(std::fabs(pmath::phi1(x) - std::expm1(x) / x) <= 1e-15 * std::fabs(pmath::phi1(x)));
  }
  CHECK(pmath::phi1(0.0) == 1.0);
  for (float x : {1e-30f, -1e-6f, -0.000309173658f, 0.01f, -0.3f, 0.356653512f, 2.0f}) {
    const double xd = static_cast<double>(x);
    CHECK(ulp_distance(pmath::expm1(x), static_cast<float>(std::expm1(xd))) <= 4);
    CHECK(ulp_distance(pmath::log1p(x), static_cast<float>(std::log1p(xd))) <= 3);
    CHECK(ulp_distance(pmath::phi1(x), static_cast<float>(std::expm1(xd) / xd)) <= 3);
  }
}

TEST_CASE("residual exponentials R_j match long-double references and their identities") {
  for (int j = 0; j <= 4; ++j) {
    // 1.007 to 1.016: the direct form's cancellation just above 1 (R_3 and R_4 reach 3.7e-15 there).
    for (double y : {0.0, 1e-8, 1e-3, 0.1, 0.5, 0.999, 1.0, 1.001, 1.007, 1.009, 1.016, 2.0, 7.5, 40.0}) {
      long double ref = std::exp(-static_cast<long double>(y)), term = 1;
      for (int i = 0; i < j; ++i) {
        if (i > 0) term *= -static_cast<long double>(y) / i;
        ref -= term;
      }
      // Below y = 1 the direct long-double form cancels; use its series instead.
      if (y > 0 && y < 1 && j > 0) {
        long double s = 0, t = 1;
        for (int i = 1; i <= j; ++i) t *= -static_cast<long double>(y) / i;
        for (int i = j; i < j + 40; ++i) {
          s += t;
          t *= -static_cast<long double>(y) / (i + 1);
        }
        ref = s;
      }
      const double got = pmath::resid_exp(j, y);
      const double tol = j >= 3 && y > 1 && y < 1.1 ? 4e-15 : 1e-15;
      CHECK(std::fabs(got - static_cast<double>(ref)) <= tol * std::fabs(static_cast<double>(ref)) + 1e-300);
      const float got32 = pmath::resid_exp(j, static_cast<float>(y));
      const double ref64 = static_cast<double>(ref);
      CHECK(std::fabs(static_cast<double>(got32) - ref64) <= 2e-6 * std::fabs(ref64) + 1e-38);
    }
  }
  // dR_j/dy = -R_{j-1}: central differences.
  for (int j = 1; j <= 4; ++j) {
    for (double y : {0.3, 0.9, 1.5, 3.0}) {
      const double h = 1e-5;
      const double d = (pmath::resid_exp(j, y + h) - pmath::resid_exp(j, y - h)) / (2 * h);
      CHECK(std::fabs(d + pmath::resid_exp(j - 1, y)) <= 1e-8);
    }
  }
}

TEST_CASE("erfcx and Dawson match long-double references") {
  double worst_e64 = 0, worst_d64 = 0, worst_e32 = 0, worst_d32 = 0;
  std::mt19937_64 rng(2);
  std::uniform_real_distribution<double> ue(-8.0, 9.0);
  for (int i = 0; i < 3000; ++i) {
    const double x = i < 600 ? i * 0.01 : std::pow(10.0, ue(rng));
    const double e_ref = static_cast<double>(balla::test::erfcx_ref(x));
    worst_e64 = std::max(worst_e64, std::fabs(pmath::erfcx(x) / e_ref - 1));
    const float xf = static_cast<float>(x);
    const double e_ref32 = static_cast<double>(balla::test::erfcx_ref(xf));
    worst_e32 = std::max(worst_e32, std::fabs(static_cast<double>(pmath::erfcx(xf)) / e_ref32 - 1));
    if (x > 0) {
      const double d_ref = static_cast<double>(balla::test::dawson_ref(x));
      worst_d64 = std::max(worst_d64, std::fabs(pmath::dawson(x) / d_ref - 1));
      worst_d32 = std::max(worst_d32, std::fabs(static_cast<double>(pmath::dawson(xf)) /
                                                static_cast<double>(balla::test::dawson_ref(xf)) - 1));
    }
  }
  MESSAGE("erfcx rel err fp64 " << worst_e64 << " fp32 " << worst_e32 << "; dawson fp64 " << worst_d64 << " fp32 "
                                << worst_d32);
  CHECK(worst_e64 < 1e-14);
  CHECK(worst_d64 < 1e-14);
  CHECK(worst_e32 < 2e-6);
  CHECK(worst_d32 < 2e-6);
  CHECK(pmath::erfcx(0.0) == 1.0);
  CHECK(pmath::dawson(0.0) == 0.0);
  // The edges: erfcx(+inf) = 0 (it was NaN), NaN propagates through both (erfcx gave 1, dawson converted it to int).
  const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
  CHECK(pmath::erfcx(inf) == 0.0);
  CHECK(pmath::erfcx(static_cast<float>(inf)) == 0.0f);
  CHECK(std::isnan(pmath::erfcx(nan)));
  CHECK(std::isnan(pmath::erfcx(static_cast<float>(nan))));
  CHECK(pmath::erfcx(-2.0) == 1.0);  // the documented clamp of negative arguments
  CHECK(std::isnan(pmath::dawson(nan)));
  CHECK(std::isnan(pmath::dawson(static_cast<float>(nan))));
  CHECK(pmath::dawson(inf) == 0.0);
  CHECK(pmath::dawson(-inf) == 0.0);
  CHECK(pmath::dawson(-1.5) == -pmath::dawson(1.5));
}

TEST_CASE("log2 is exact at powers of two and within 2 ulp plus the ln(m) rounding elsewhere") {
  std::mt19937_64 rng(3);
  std::uniform_real_distribution<double> e64(-1000.0, 1000.0), e32(-120.0, 120.0);
  for (int i = 0; i < 200000; ++i) {
    const double x = std::exp2(e64(rng));
    const double ref = std::log2(x);
    const double got = pmath::log2(x);
    // ln(m) log2(e) with |ln m| <= ln(2)/2 carries 2 ulp of 0.5; the sum adds 2 ulp of the result.
    CHECK(std::fabs(got - ref) <= 2 * 0x1p-53 + 2 * 0x1p-52 * std::fabs(ref));
    const float xf = std::exp2(static_cast<float>(e32(rng)));
    const double ref32 = std::log2(static_cast<double>(xf));
    CHECK(std::fabs(static_cast<double>(pmath::log2(xf)) - ref32) <= 2 * 0x1p-24 + 2 * 0x1p-23 * std::fabs(ref32));
  }
  for (int n = -1074; n <= 1023; ++n) CHECK(pmath::log2(std::ldexp(1.0, n)) == static_cast<double>(n));
  for (int n = -149; n <= 127; ++n) CHECK(pmath::log2(std::ldexp(1.0f, n)) == static_cast<float>(n));
  CHECK(pmath::log2(0.0f) == -INFINITY);
  CHECK(std::isnan(pmath::log2(-1.0f)));
  CHECK(pmath::log2(INFINITY) == INFINITY);
}

TEST_CASE("exp2 is within 2 ulp of std::exp2, exact at integers and never subnormal") {
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> x(-126.0f, 127.0f);
  for (int i = 0; i < 200000; ++i) {
    const float a = x(rng);
    const double ref = std::exp2(static_cast<double>(a));
    const double ulp = std::ldexp(1.0, std::ilogb(ref) - 23);
    CHECK(std::fabs(static_cast<double>(pmath::exp2(a)) - ref) <= 2.0 * ulp);
  }
  for (int n = -126; n <= 127; ++n) CHECK(pmath::exp2(static_cast<float>(n)) == std::ldexp(1.0f, n));
  CHECK(pmath::exp2(-126.5f) == 0.0f);
  CHECK(pmath::exp2(-1000.0f) == 0.0f);
  CHECK(pmath::exp2(1000.0f) == std::ldexp(1.0f, 127));
}

TEST_CASE("the renderer's Gaussian weights e^(-q/2) = 2^(-q/2 log2 e) on q in [0, 4]") {
  for (float q = 0.0f; q <= 4.0f; q += 0.001f) {
    const double ref = std::exp(-0.5 * static_cast<double>(q));
    CHECK(std::fabs(static_cast<double>(pmath::exp2(-0.5f * q * 1.44269504088896341f)) - ref) <= 4e-7 * ref);
    CHECK(std::fabs(static_cast<double>(pmath::exp(-0.5f * q)) - ref) <= 4e-7 * ref);
  }
}

TEST_CASE_TEMPLATE("smootherstep is C2 with value and two derivatives vanishing at the ends", T, float, double) {
  CHECK(pmath::smootherstep(T(-1)) == T(0));
  CHECK(pmath::smootherstep(T(0)) == T(0));
  CHECK(pmath::smootherstep(T(0.5)) == T(0.5));
  CHECK(pmath::smootherstep(T(1)) == T(1));
  CHECK(pmath::smootherstep(T(2)) == T(1));
  // Exactly antisymmetric about 1/2 wherever 1 - x is exact.
  for (T x : {T(0.125), T(0.3125), T(0.4375), T(0.25) + std::numeric_limits<T>::epsilon()}) {
    CHECK(pmath::smootherstep(T(1) - x) == T(1) - pmath::smootherstep(x));
  }
  const double h = 1e-3;
  // At x0 = 1 the fp32 values round to exactly 1 (1 - S(1e-3) = 1 - 1e-8), so the differences there would be 0
  // whatever the derivatives; the exact antisymmetry above carries x0 = 0's checks to that end in fp32.
  for (double x0 : {0.0, 1.0}) {
    if (sizeof(T) == 4 && x0 == 1.0) continue;
    auto f = [](double x) { return static_cast<double>(pmath::smootherstep(static_cast<T>(x))); };
    const double d1 = (f(x0 + h) - f(x0 - h)) / (2 * h);
    const double d2 = (f(x0 + h) - 2 * f(x0) + f(x0 - h)) / (h * h);
    CHECK(std::fabs(d1) < 1e-4);
    CHECK(std::fabs(d2) < 0.05);  // 60 x h at the ends (third derivative 60), on fp32 values too
  }
}
