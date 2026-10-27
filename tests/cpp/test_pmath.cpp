#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <random>

#include "core/pmath/trig.h"

namespace pm = balla::pmath;

namespace {

// Distance in units in the last place of the reference.
template <class T>
double ulp_error(T value, double reference) {
  const T ref = static_cast<T>(reference);
  const double ulp = static_cast<double>(std::nextafter(std::abs(ref), T(INFINITY)) - std::abs(ref));
  return std::abs(static_cast<double>(value) - reference) / ulp;
}

template <class T>
void check_sincos(T lo, T hi, double max_ulp, double abs_tol) {
  std::mt19937_64 rng(17);
  std::uniform_real_distribution<double> dist(static_cast<double>(lo), static_cast<double>(hi));
  double worst = 0.0;
  for (int i = 0; i < 200000; ++i) {
    const T x = static_cast<T>(dist(rng));
    T s, c;
    pm::sincos(x, &s, &c);
    const double sr = std::sin(static_cast<double>(x)), cr = std::cos(static_cast<double>(x));
    // Near zeros of sin/cos the error is bounded absolutely by the reduction error.
    if (std::abs(sr) > 1e-3) worst = std::max(worst, ulp_error(s, sr));
    if (std::abs(cr) > 1e-3) worst = std::max(worst, ulp_error(c, cr));
    REQUIRE(std::abs(static_cast<double>(s) - sr) <= abs_tol);
    REQUIRE(std::abs(static_cast<double>(c) - cr) <= abs_tol);
  }
  CHECK(worst <= max_ulp);
}

}  // namespace

TEST_CASE("pmath sincos matches libm within 2 ulp (fp64) and 3 ulp (fp32)") {
  // libm's fp64 sin/cos are themselves within 1 ulp, so the double bound includes its error.
  // The whole supported domains (kMaxArg: 1e6 and 6000), with the absolute bounds of their exact reductions
  // (measured 2.1e-16 and 1.1e-7).
  check_sincos<double>(-4.0, 4.0, 2.0, 4e-16);
  check_sincos<double>(-1e6, 1e6, 2.0, 4e-16);
  check_sincos<float>(-4.0f, 4.0f, 3.0, 2e-7);
  check_sincos<float>(-6000.0f, 6000.0f, 3.0, 2e-7);
}

TEST_CASE("pmath sincos special values") {
  double s, c;
  pm::sincos(0.0, &s, &c);
  CHECK(s == 0.0);
  CHECK(c == 1.0);
  pm::sincos(-0.0, &s, &c);
  CHECK(std::signbit(s));
  float sf, cf;
  pm::sincos(1e-30f, &sf, &cf);
  CHECK(sf == 1e-30f);
  CHECK(cf == 1.0f);
  for (double bad : {double(INFINITY), -double(INFINITY), double(NAN), 2e6}) {
    pm::sincos(bad, &s, &c);
    CHECK(std::isnan(s));
    CHECK(std::isnan(c));
  }
  CHECK(std::isnan(pm::sin(7000.0f)));
  CHECK(std::abs(static_cast<double>(pm::sin(5999.0f)) - std::sin(5999.0)) < 1e-3);
}

TEST_CASE("pmath atan2 matches libm and keeps IEEE sign conventions") {
  std::mt19937_64 rng(5);
  std::normal_distribution<double> dist;
  double worst_d = 0.0, worst_f = 0.0;
  for (int i = 0; i < 200000; ++i) {
    const double scale = std::exp(4.0 * dist(rng));
    const double y = dist(rng) * scale, x = dist(rng);
    const double ref = std::atan2(y, x);
    worst_d = std::max(worst_d, ulp_error(pm::atan2(y, x), ref));
    const float yf = static_cast<float>(y), xf = static_cast<float>(x);
    worst_f = std::max(worst_f, ulp_error(pm::atan2(yf, xf), std::atan2(static_cast<double>(yf),
                                                                          static_cast<double>(xf))));
  }
  CHECK(worst_d <= 2.0);
  CHECK(worst_f <= 3.0);

  CHECK(pm::atan2(0.0, 1.0) == 0.0);
  CHECK(std::signbit(pm::atan2(-0.0, 1.0)));
  CHECK(pm::atan2(0.0, -1.0) == doctest::Approx(M_PI).epsilon(1e-16));
  CHECK(pm::atan2(-0.0, -1.0) == doctest::Approx(-M_PI).epsilon(1e-16));
  CHECK(pm::atan2(0.0, -0.0) == doctest::Approx(M_PI).epsilon(1e-16));
  CHECK(pm::atan2(1.0, 0.0) == doctest::Approx(M_PI / 2).epsilon(1e-16));
  CHECK(pm::atan2(-1.0, 0.0) == doctest::Approx(-M_PI / 2).epsilon(1e-16));
  CHECK(pm::atan2(1.0, 1.0) == doctest::Approx(M_PI / 4).epsilon(1e-16));
  CHECK(pm::atan2(1e-300, 1.0) == 1e-300);
}

TEST_CASE("pmath asin and acos match libm within 3 ulp (fp64) and 4 ulp (fp32) on [-1, 1]") {
  // Uniform over [-1, 1] and clustered at the ends (1 - 10^-u), where acos and the cosine side of asin are steepest.
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<double> unit(-1.0, 1.0), decades(0.0, 15.0);
  double worst[2][2] = {};  // [fp64, fp32][asin, acos]
  for (int i = 0; i < 400000; ++i) {
    double x = unit(rng);
    if (i % 2 == 1) x = std::copysign(1.0 - std::pow(10.0, -decades(rng)), x);
    worst[0][0] = std::max(worst[0][0], ulp_error(pm::asin(x), std::asin(x)));
    worst[0][1] = std::max(worst[0][1], ulp_error(pm::acos(x), std::acos(x)));
    const float xf = static_cast<float>(x);
    const double xd = static_cast<double>(xf);
    worst[1][0] = std::max(worst[1][0], ulp_error(pm::asin(xf), std::asin(xd)));
    worst[1][1] = std::max(worst[1][1], ulp_error(pm::acos(xf), std::acos(xd)));
  }
  INFO("fp64 asin " << worst[0][0] << " acos " << worst[0][1] << ", fp32 asin " << worst[1][0] << " acos "
                    << worst[1][1] << " ulp");
  // atan2's own bounds (2 and 3 ulp) plus the rounding of the cosine side.
  CHECK(worst[0][0] <= 3.0);
  CHECK(worst[0][1] <= 3.0);
  CHECK(worst[1][0] <= 4.0);
  CHECK(worst[1][1] <= 4.0);

  CHECK(pm::asin(0.0) == 0.0);
  CHECK(std::signbit(pm::asin(-0.0)));
  CHECK(pm::asin(1.0) == doctest::Approx(M_PI / 2).epsilon(1e-16));
  CHECK(pm::asin(-1.0) == doctest::Approx(-M_PI / 2).epsilon(1e-16));
  CHECK(pm::acos(1.0) == 0.0);
  CHECK(pm::acos(-1.0) == doctest::Approx(M_PI).epsilon(1e-16));
  CHECK(pm::acos(0.0) == doctest::Approx(M_PI / 2).epsilon(1e-16));
  CHECK(pm::asin(1e-300) == 1e-300);
  CHECK(pm::acos(1.0f) == 0.0f);
  for (double bad : {1.0 + 1e-15, -1.0 - 1e-15, 2.0, double(INFINITY), double(NAN)}) {
    CHECK(std::isnan(pm::asin(bad)));
    CHECK(std::isnan(pm::acos(bad)));
  }
  CHECK(std::isnan(pm::asin(1.0000001f)));
}
