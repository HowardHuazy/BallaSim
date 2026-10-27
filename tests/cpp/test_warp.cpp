#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <tuple>
#include <vector>

#include "core/rng/subsample.h"
#include "core/rng/warp.h"
#include "core/rng/warp_table.h"

using namespace balla;
using namespace balla::rng;

namespace {

// Exact inverse of the truncated CDF by bisection.
double exact_inverse(const PixelResponse& h, double xi) {
  double lo = -h.x_max(), hi = h.x_max();
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi);
    (h.cdf(mid) < xi ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}

// warp_inverse_cdf as it was before the guided cell search: a binary search for max{i < n :
// cdf[i] <= p}, then the same Hermite step. The guided search must return the same bits (the
// sub-sample positions, and every golden value downstream of them, depend on it).
template <class T>
T binary_search_inverse_cdf(const WarpTableView<T>& t, T xi) {
  const bool upper = xi >= T(0.5);
  const T p = upper ? T(1) - xi : xi;
  int32_t lo = 0, hi = t.n;
  while (hi - lo > 1) {
    const int32_t mid = (lo + hi) / 2;
    if (t.cdf[mid] <= p) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const T f0 = t.cdf[lo];
  const T df = t.cdf[lo + 1] - f0;
  const T s = (p - f0) / df;
  const T x0 = T(lo - t.n) * t.dx;
  const T x1 = T(lo + 1 - t.n) * t.dx;
  const T m0 = t.slope[lo] * df;
  const T m1 = t.slope[lo + 1] * df;
  const T one_minus = T(1) - s;
  const T h00 = (T(1) + T(2) * s) * (one_minus * one_minus);
  const T h10 = s * (one_minus * one_minus);
  const T h01 = (s * s) * (T(3) - T(2) * s);
  const T h11 = (s * s) * (s - T(1));
  const T x = ((h00 * x0 + h10 * m0) + h01 * x1) + h11 * m1;
  return upper ? -x : x;
}

uint32_t bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  return u;
}

}  // namespace

TEST_CASE("pixel response: truncation and normalization") {
  const PixelResponse h(1.0, 0.15, 3.0);
  CHECK(h.x_max() == doctest::Approx(0.95));
  CHECK(h.cdf(-h.x_max()) == doctest::Approx(0.0).epsilon(1e-15));
  CHECK(h.cdf(0.0) == doctest::Approx(0.5).epsilon(1e-15));
  CHECK(h.cdf(h.x_max()) == doctest::Approx(1.0).epsilon(1e-15));
  // Interior density of a 1 px box blurred by 0.15 px: 1 at the centre, 1/2 at the box edge.
  CHECK(h.pdf(0.0) == doctest::Approx(1.0).epsilon(1e-3));
  CHECK(h.pdf(0.5) == doctest::Approx(0.5).epsilon(1e-3));
  CHECK_THROWS(PixelResponse(0.0, 0.0, 3.0));
}

TEST_CASE("warp table: exact endpoints, antisymmetry, monotonicity, accuracy") {
  const PixelResponse h(1.0, 0.15, 3.0);
  const WarpTable<double> t64(h, 256);
  const WarpTable<float> t32(h, 256);
  const WarpTableView<double> v64 = t64.view();
  const WarpTableView<float> v32 = t32.view();
  CHECK(warp_inverse_cdf(v64, 0.0) == doctest::Approx(-0.95).epsilon(1e-15));
  CHECK(warp_inverse_cdf(v64, 0.5) == 0.0);
  double prev = -1.0, worst64 = 0.0, worst32 = 0.0;
  for (int i = 0; i < 100000; ++i) {
    const double xi = (i + 0.5) / 100000.0;
    const double x = warp_inverse_cdf(v64, xi);
    CHECK(x > prev);
    prev = x;
    const double dyadic = std::ldexp(static_cast<double>(i), -17);  // 1 - dyadic is exact
    CHECK(warp_inverse_cdf(v64, 1.0 - dyadic) == -warp_inverse_cdf(v64, dyadic));
    const double exact = exact_inverse(h, xi);
    worst64 = std::max(worst64, std::abs(x - exact));
    const float xf = static_cast<float>(xi);
    worst32 = std::max(worst32, std::abs(static_cast<double>(warp_inverse_cdf(v32, xf)) -
                                         exact_inverse(h, static_cast<double>(xf))));
  }
  CHECK(worst64 < 5e-8);
  CHECK(worst32 < 2e-6);
}

TEST_CASE("warp without blur is the linear map of the box") {
  const PixelResponse h(1.0, 0.0, 3.0);
  const WarpTable<double> t(h, 16);
  for (double xi : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9}) {
    CHECK(warp_inverse_cdf(t.view(), xi) == doctest::Approx(xi - 0.5).epsilon(1e-15));
  }
}

TEST_CASE("sub-sample offsets stay inside the truncated support") {
  const PixelResponse h(1.0, 0.15, 3.0);
  const WarpTable<float> t(h, 256);
  for (uint32_t pixel = 0; pixel < 100; ++pixel) {
    const uint32_t seed = subsample_pattern_seed(42, 0, pixel);
    for (uint32_t k = 0; k < 64; ++k) {
      const Vec2<float> o = subsample_offset(t.view(), seed, k);
      CHECK(std::abs(o.x) <= 0.95f);
      CHECK(std::abs(o.y) <= 0.95f);
    }
  }
}

TEST_CASE("the guided cell search returns the binary search's bits") {
  for (auto [w, sigma, cells] : {std::tuple{1.0, 0.15, 256}, std::tuple{0.5, 0.3, 97}, std::tuple{1.0, 0.0, 64},
                                 std::tuple{1.0, 0.15, 65535}, std::tuple{1.0, 0.15, 2}}) {
    const WarpTable<float> table(PixelResponse(w, sigma, 3.0), cells);
    const WarpTableView<float> v = table.view();
    // Every fp32 value next to a cell boundary (from both sides of 1/2), plus a dense random sweep.
    std::vector<float> xis;
    for (const float c : table.cdf()) {
      for (const float p : {c, std::nextafter(c, 0.0f), std::nextafter(c, 1.0f)}) {
        xis.push_back(p);
        xis.push_back(1.0f - p);
      }
    }
    std::mt19937 rng(14);
    for (int i = 0; i < 2000000; ++i) xis.push_back(to_unit_interval<float>(rng()));
    for (const float xi : xis) {
      if (!(xi >= 0.0f && xi < 1.0f)) continue;
      REQUIRE(bits(warp_inverse_cdf(v, xi)) == bits(binary_search_inverse_cdf(v, xi)));
    }
    // The guide is a lower bound of the cell of its own CDF value (warp.h).
    for (int32_t g = 0; g < kWarpGuideCells; ++g) {
      const float p = static_cast<float>(g) / static_cast<float>(2 * kWarpGuideCells);
      CHECK(table.cdf()[table.guide()[static_cast<size_t>(g)]] <= p);
    }
  }
  const WarpTable<double> t64(PixelResponse(1.0, 0.15, 3.0), 256);
  std::mt19937_64 rng(15);
  for (int i = 0; i < 1000000; ++i) {
    const double xi = to_unit_interval<double>(static_cast<uint32_t>(rng()));
    REQUIRE(warp_inverse_cdf(t64.view(), xi) == binary_search_inverse_cdf(t64.view(), xi));
  }
  CHECK_THROWS(WarpTable<float>(PixelResponse(1.0, 0.15, 3.0), 65536));
  CHECK_THROWS(WarpTable<float>(PixelResponse(1.0, 0.15, 3.0), 1));
}
