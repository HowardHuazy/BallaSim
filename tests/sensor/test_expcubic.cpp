// Comparator function and exact crossing detection (src/sensor/expcubic.h; M3 V2 T1 and T3).
#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <vector>

#include "sensor/expcubic.h"

using namespace balla::sensor;

namespace {

ExpCubic<double> random_fn(std::mt19937_64& rng, int order, double scale_u) {
  std::normal_distribution<double> n;
  std::uniform_real_distribution<double> u(0.0, 1.0);
  ExpCubic<double> g{};
  // Coefficients scaled to the interval length so every term matters on [0, scale_u].
  g.c[0] = n(rng);
  g.c[1] = n(rng) / scale_u;
  g.c[2] = n(rng) / (scale_u * scale_u);
  g.c[3] = n(rng) / (scale_u * scale_u * scale_u);
  g.a[0] = (0.5 + 30 * u(rng)) / scale_u;
  g.a[1] = g.a[0] / (1.2 + 5 * u(rng));
  g.C[0] = n(rng) * (order == 4 ? 30.0 : 1.0);
  g.C[1] = n(rng) * (order == 4 ? 30.0 : 1.0);
  g.order = order;
  return g;
}

}  // namespace

TEST_CASE("exclusion hull is conservative in both splits (M3 V2 T3)") {
  std::mt19937_64 rng(3);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  int violations = 0;
  for (int trial = 0; trial < 5000; ++trial) {
    const double w = 65.536e-6;
    const ExpCubic<double> g = random_fn(rng, trial % 2 == 0 ? 0 : 4, w);
    const double a = 0.9 * w * u(rng), b = a + (w - a) * (1e-4 + u(rng));
    const Range<double> r = range(g, a, b);
    for (int i = 0; i <= 2000; ++i) {
      const double v = g(a + (b - a) * i / 2000);
      if (v < r.lo - 1e-12 * (1 + std::fabs(v)) || v > r.hi + 1e-12 * (1 + std::fabs(v))) {
        ++violations;
        break;
      }
    }
  }
  CHECK(violations == 0);
}

TEST_CASE("shift and derivatives are exact re-expansions") {
  std::mt19937_64 rng(4);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int trial = 0; trial < 500; ++trial) {
    const double w = 1e-5;
    const ExpCubic<double> g = random_fn(rng, trial % 2 == 0 ? 0 : 4, w);
    const double d = w * u(rng);
    const ExpCubic<double> s = shifted(g, d);
    for (double x : {0.0, 0.1 * w, 0.5 * (w - d), w - d}) {
      CHECK(std::fabs(s(x) - g(d + x)) <= 1e-12 * (1 + std::fabs(g(d + x))));
    }
    for (int n = 1; n <= 4; ++n) {
      const ExpCubic<double> dn = derivative(g, n), dm = derivative(g, n - 1);
      const double x = w * (0.2 + 0.6 * u(rng)), h = 1e-4 * w;
      const double fd = (dm(x + h) - dm(x - h)) / (2 * h);
      CHECK(std::fabs(dn(x) - fd) <= 1e-5 * (std::fabs(fd) + std::fabs(dm(x)) / w + 1.0 / std::pow(w, n)));
    }
  }
}

TEST_CASE("first_firing finds the first integer crossing exactly (0 mismatches vs a dense scan)") {
  // Interval of 65536 ns: every integer is scanned by brute force. Functions with planted roots,
  // including clustered pairs, in both splits, for ON (G = g) and OFF (G = -g) polarities.
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const int64_t K = 65535;
  const double w = K * 1e-9;
  int mismatches = 0, events = 0, max_crit = 0;
  for (int trial = 0; trial < 2000; ++trial) {
    ExpCubic<double> g = random_fn(rng, trial % 2 == 0 ? 0 : 4, w);
    // Plant a root: shift c0 so that g(r) = 0 at a random point.
    const double r = w * u(rng);
    g.c[0] -= g(r);
    if (trial % 3 == 0) {
      // Cluster: a near-tangent pair by moving the level to a local extremum minus a tiny amount.
      const RootList<double> cr = critical_points(g, 0.0, w);
      if (cr.n > 0) g.c[0] -= g(cr.v[0]) + 1e-9 * (u(rng) - 0.3);
    }
    const RootList<double> crit = critical_points(g, 0.0, w);
    max_crit = std::max(max_crit, crit.n);
    const int64_t k_lo = static_cast<int64_t>(u(rng) * 1000);
    for (int pol = 0; pol < 2; ++pol) {
      ExpCubic<double> G = g;
      if (pol == 1) {
        for (double& c : G.c) c = -c;
        G.C[0] = -G.C[0];
        G.C[1] = -G.C[1];
      }
      const int64_t got = first_firing(G, crit, k_lo, K);
      int64_t want = -1;
      for (int64_t k = k_lo; k <= K; ++k) {
        if (G(k * 1e-9) >= 0) {
          want = k;
          break;
        }
      }
      if (want >= 0) ++events;
      if (got != want) {
        // A disagreement is acceptable only where G is within rounding of zero at the integers.
        const double scale = 1e-12 * (1 + std::fabs(G.c[0]) + std::fabs(G.C[0]) + std::fabs(G.C[1]));
        const bool marginal = (want >= 0 && std::fabs(G(want * 1e-9)) <= scale) ||
                              (got >= 0 && std::fabs(G(got * 1e-9)) <= scale);
        if (!marginal) ++mismatches;
      }
    }
  }
  MESSAGE(events << " crossings, max critical points " << max_crit);
  CHECK(events > 1500);
  CHECK(mismatches == 0);
  CHECK(max_crit <= 4);  // Polya bound: <= 5 zeros, hence <= 4 interior extrema
}

TEST_CASE("Polya-Szego bound: never more than 5 sign changes on an interval (M3 V2 T1)") {
  std::mt19937_64 rng(5);
  int violations = 0;
  for (int trial = 0; trial < 3000; ++trial) {
    const ExpCubic<double> g = random_fn(rng, trial % 2 == 0 ? 0 : 4, 1.0);
    int changes = 0;
    double prev = g(0.0);
    for (int i = 1; i <= 20000; ++i) {
      const double v = g(i / 20000.0);
      if ((prev < 0) != (v < 0)) ++changes;
      prev = v;
    }
    violations += changes > 5;
  }
  CHECK(violations == 0);
}

TEST_CASE("fp32 crossing search agrees with fp64 within a few nanoseconds") {
  std::mt19937_64 rng(9);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const int64_t K = 65535;
  const double w = K * 1e-9;
  int compared = 0, misses = 0;
  int64_t worst = 0;
  for (int trial = 0; trial < 1000; ++trial) {
    ExpCubic<double> g = random_fn(rng, trial % 2 == 0 ? 0 : 4, w);
    g.c[0] -= g(w * u(rng));
    ExpCubic<float> f{};
    for (int i = 0; i < 4; ++i) f.c[i] = static_cast<float>(g.c[i]);
    for (int i = 0; i < 2; ++i) {
      f.C[i] = static_cast<float>(g.C[i]);
      f.a[i] = static_cast<float>(g.a[i]);
    }
    f.order = g.order;
    const int64_t k64 = first_firing(g, critical_points(g, 0.0, w), 0, K);
    const int64_t k32 = first_firing(f, critical_points(f, 0.0f, static_cast<float>(w)), 0, K);
    // Compare only robust crossings: G must change by > 1e-6 between neighbouring integers. An
    // fp32 search that misses such a crossing counts against the test.
    if (k64 > 0 && std::fabs(g(k64 * 1e-9) - g((k64 - 1) * 1e-9)) > 1e-6) {
      ++compared;
      if (k32 < 0)
        ++misses;
      else
        worst = std::max(worst, std::abs(k64 - k32));
    }
  }
  MESSAGE("compared " << compared << " crossings, " << misses << " fp32 misses, max |dt| = " << worst
                      << " ns");
  CHECK(compared > 300);
  CHECK(misses == 0);
  CHECK(worst <= 4);
}
