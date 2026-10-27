// U1 (docs/design/phase3/frames.md §8.1): exact interval means of the light-group waveforms, waveform_mean and
// WaveformPrefix (§4.5).
//   Fourier: against exact rational phases (f0 = p / 2^s, int64 ns, reduced in __int128) evaluated in long double,
//            within 1e-12 + sum_k |a_k| 2 pi (k+1) 4 ulp((k+1) f0 |t_mid| 1e-9); independently against the
//            antiderivative on longer windows.
//   pulse:   against the trapezoid sum of waveform_value over every ns (exact for the snapped piecewise-linear
//            pulse), relative error <= 1e-13; WaveformPrefix bitwise equal to waveform_mean.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "sensor/waveform.h"

namespace {

using balla::sensor::Waveform;
using balla::sensor::waveform_mean;
using balla::sensor::waveform_value;
using balla::sensor::WaveformPrefix;
using balla::sensor::WaveformType;

__extension__ typedef __int128 i128;

constexpr long double kPiL = 3.141592653589793238462643383279502884L;
constexpr double kPi = 3.141592653589793;

i128 pos_mod(i128 x, i128 m) {
  const i128 r = x % m;
  return r < 0 ? r + m : r;
}

// f = p / 2^s exactly.
struct Dyadic {
  i128 p;
  int s;
};
Dyadic dyadic(double f) {
  int e = 0;
  const double m = std::frexp(f, &e);
  i128 p = static_cast<i128>(std::ldexp(m, 53));
  int s = 53 - e;
  while (s > 0 && (p & 1) == 0) {
    p >>= 1;
    --s;
  }
  if (s < 0) {
    p <<= -s;
    s = 0;
  }
  return {p, s};
}

// The exact mean of harmonic k's cosine over [a, b], from rational phases: cos(phase at the midpoint) times
// sin(pi h) / (pi h), h the cycles over the window.
long double harmonic_mean_exact(const Waveform& w, int k, int64_t a, int64_t b) {
  const Dyadic f = dyadic(w.f0_hz);
  const i128 j = k + 1;
  const i128 den = (i128{1} << f.s) * 1'000'000'000;  // cycles(t) = j p t / den
  const long double mid = static_cast<long double>(pos_mod(j * f.p * (i128{a} + b), 2 * den)) /
                          static_cast<long double>(2 * den);
  const long double phase = 2 * kPiL * mid + static_cast<long double>(w.phi[k]);
  const i128 hn = j * f.p * (i128{b} - a);
  const long double x = kPiL * static_cast<long double>(hn) / static_cast<long double>(den);
  long double sinc = 1.0L;
  if (x > 1e-6L) {
    const long double h2 = static_cast<long double>(pos_mod(hn, 2 * den)) / static_cast<long double>(den);
    sinc = std::sin(kPiL * h2) / x;
  } else {
    sinc = 1.0L - x * x / 6.0L;
  }
  return std::cos(phase) * sinc;
}

long double fourier_mean_exact(const Waveform& w, int64_t a, int64_t b) {
  long double m = 1.0L;
  for (int k = 0; k < w.n_harmonics; ++k) m += static_cast<long double>(w.a[k]) * harmonic_mean_exact(w, k, a, b);
  return m;
}

// The same mean from the antiderivative, (sin phi(b) - sin phi(a)) / (2 pi h): independent of the midpoint identity,
// well conditioned for h >~ 1e-3.
long double fourier_mean_antiderivative(const Waveform& w, int64_t a, int64_t b) {
  const Dyadic f = dyadic(w.f0_hz);
  const i128 den = (i128{1} << f.s) * 1'000'000'000;
  long double m = 1.0L;
  for (int k = 0; k < w.n_harmonics; ++k) {
    const i128 j = k + 1;
    auto phase = [&](int64_t t) {
      return 2 * kPiL * static_cast<long double>(pos_mod(j * f.p * t, den)) / static_cast<long double>(den) +
             static_cast<long double>(w.phi[k]);
    };
    const long double h = static_cast<long double>(j * f.p * (i128{b} - a)) / static_cast<long double>(den);
    m += static_cast<long double>(w.a[k]) * (std::sin(phase(b)) - std::sin(phase(a))) / (2 * kPiL * h);
  }
  return m;
}

double ulp(double x) {
  x = std::abs(x);
  return std::nextafter(x, std::numeric_limits<double>::infinity()) - x;
}

// U1's bound: 1e-12 + sum_k |a_k| 2 pi (k+1) 4 ulp((k+1) f0 |t_mid| 1e-9).
double fourier_bound(const Waveform& w, int64_t a, int64_t b) {
  const double t_mid = std::abs(static_cast<double>(a) + 0.5 * static_cast<double>(b - a));
  double bound = 1e-12;
  for (int k = 0; k < w.n_harmonics; ++k) {
    const double j = k + 1;
    bound += std::abs(w.a[k]) * 2 * kPi * j * 4 * ulp(j * w.f0_hz * t_mid * 1e-9);
  }
  return bound;
}

Waveform fourier8(double f0, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> amp(-0.12, 0.12), ph(0.0, 2 * kPi);
  Waveform w;
  w.type = WaveformType::kFourier;
  w.n_harmonics = 8;
  w.f0_hz = f0;
  for (int k = 0; k < 8; ++k) {
    w.a[k] = amp(rng);
    w.phi[k] = ph(rng);
  }
  return w;
}

// A pulse train with non-integer period, phase, rise and top, so corners snap.
Waveform pulse(double period, double phase, double rise, double top, double floor_level) {
  Waveform w;
  w.type = WaveformType::kPulse;
  w.period_ns = period;
  w.phase_ns = phase;
  w.rise_ns = rise;
  w.top_ns = top;
  w.floor_level = floor_level;
  w.peak_level = floor_level + (1.0 - floor_level) * period / (top + rise);
  return w;
}

// Mean of the snapped PL pulse over [a, b] by the trapezoid rule on every ns (exact: its corners are integers),
// Neumaier-summed in long double.
long double pulse_mean_brute(const Waveform& w, int64_t a, int64_t b) {
  long double sum = 0.0L, comp = 0.0L;
  double prev = waveform_value<double>(w, a);
  for (int64_t t = a; t < b; ++t) {
    const double next = waveform_value<double>(w, t + 1);
    const long double term = 0.5L * (static_cast<long double>(prev) + static_cast<long double>(next));
    const long double s = sum + term;
    comp += std::abs(sum) >= std::abs(term) ? (sum - s) + term : (term - s) + sum;
    sum = s;
    prev = next;
  }
  return (sum + comp) / static_cast<long double>(b - a);
}

}  // namespace

TEST_CASE("U1 waveform_mean: Fourier against exact rational phases") {
  std::mt19937_64 rng(20261012);
  const struct {
    double f0;
    int n;
  } cases[] = {{120.0, 6000}, {59.94, 2500}, {1000.0, 1500}};
  int checked = 0, odd = 0, negative = 0, straddling = 0;
  double worst = 0.0;
  for (const auto& c : cases) {
    const Waveform w = fourier8(c.f0, rng);
    const double period = 1e9 / c.f0;
    const double t_max = 1e7 * period;  // f t <= 1e7 cycles
    std::uniform_real_distribution<double> mid(-t_max, t_max), u01(0.0, 1.0);
    for (int i = 0; i < c.n; ++i) {
      const double kind = u01(rng);
      int64_t d = 0;
      if (kind < 0.1) {
        d = 1;
      } else if (kind < 0.2) {
        d = 2;
      } else {
        d = std::max<int64_t>(1, std::llround(std::exp(u01(rng) * std::log(1e7 * period))));  // up to 1e7 periods
      }
      int64_t a = 0;
      if (kind > 0.9) {
        a = -std::llround(u01(rng) * static_cast<double>(d));  // zero-straddling
      } else {
        a = std::llround(mid(rng)) - d / 2;
      }
      a = std::clamp<int64_t>(a, -std::llround(t_max), std::llround(t_max) - d);
      const int64_t b = a + d;
      const double got = waveform_mean<double>(w, a, b);
      const long double want = fourier_mean_exact(w, a, b);
      const double err = static_cast<double>(std::abs(static_cast<long double>(got) - want));
      const double bound = fourier_bound(w, a, b);
      worst = std::max(worst, err / bound);
      if (err > bound) {
        FAIL_CHECK("f0 " << c.f0 << " [" << a << ", " << b << "): err " << err << " > " << bound);
      }
      ++checked;
      odd += (a + b) % 2 != 0;
      negative += b <= 0;
      straddling += a < 0 && b > 0;
    }
  }
  MESSAGE("U1 fourier: " << checked << " windows, worst err/bound " << worst);
  CHECK(checked == 10000);
  CHECK(odd > 1000);
  CHECK(negative > 1000);
  CHECK(straddling > 500);
}

TEST_CASE("U1 waveform_mean: Fourier against the antiderivative") {
  std::mt19937_64 rng(7);
  const Waveform w = fourier8(120.0, rng);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  for (int i = 0; i < 1000; ++i) {
    const int64_t d = std::llround(std::exp(std::log(1e4) + u01(rng) * std::log(1e5)));  // 10 us .. 1 s
    const int64_t a = std::llround((u01(rng) - 0.5) * 2e12);
    const double got = waveform_mean<double>(w, a, a + d);
    const long double want = fourier_mean_antiderivative(w, a, a + d);
    CHECK(static_cast<double>(std::abs(static_cast<long double>(got) - want)) <= fourier_bound(w, a, a + d) + 1e-14);
  }
}

TEST_CASE("U1 waveform_mean: pulse against the per-ns trapezoid sum") {
  const Waveform w = pulse(1000.37, -123.456, 37.3, 412.9, 0.2);
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  double worst = 0.0;
  auto check = [&](int64_t a, int64_t b) {
    const double got = waveform_mean<double>(w, a, b);
    const long double want = pulse_mean_brute(w, a, b);
    const double rel = static_cast<double>(std::abs((static_cast<long double>(got) - want) / want));
    worst = std::max(worst, rel);
    if (rel > 1e-13) FAIL_CHECK("[" << a << ", " << b << "): relative error " << rel);
  };
  // Short random windows, negative and positive times: 1 ns to 5 periods.
  for (int i = 0; i < 3000; ++i) {
    const int64_t a = std::llround((u01(rng) - 0.5) * 2e9);
    check(a, a + 1 + static_cast<int64_t>(u01(rng) * 5000.0));
  }
  // Windows inside one ramp (rise and fall), and touching its ends.
  for (int i = 0; i < 400; ++i) {
    const int64_t n = static_cast<int64_t>((u01(rng) - 0.5) * 2e6);
    const int ramp = i % 2 == 0 ? 0 : 2;
    const int64_t c0 = balla::sensor::detail::pulse_corner(w, n, ramp);
    const int64_t c1 = balla::sensor::detail::pulse_corner(w, n, ramp + 1);
    const int64_t x = c0 + static_cast<int64_t>(u01(rng) * static_cast<double>(c1 - c0 - 1));
    const int64_t y = x + 1 + static_cast<int64_t>(u01(rng) * static_cast<double>(c1 - x - 1));
    REQUIRE(c0 <= x);
    REQUIRE(y <= c1);
    check(x, y);
  }
  // Across more than 1e4 periods, at negative and positive times.
  check(-12'345'678'901, -12'345'678'901 + 10'000'371);
  check(98'765'432'109, 98'765'432'109 + 10'500'777);
  MESSAGE("U1 pulse: worst relative error " << worst);

  // A 1 kHz pulse with a 1 ns rise.
  const Waveform slow = pulse(1e6 + 0.3, 0.7, 1.0, 2.5e5, 0.0);
  for (int64_t a : {-3'000'017, 5, 123'456'789}) {
    const double got = waveform_mean<double>(slow, a, a + 2'500'001);
    const long double want = pulse_mean_brute(slow, a, a + 2'500'001);
    CHECK(static_cast<double>(std::abs((static_cast<long double>(got) - want) / want)) <= 1e-13);
  }
}

TEST_CASE("U1 waveform_mean: constant, instantaneous windows") {
  const Waveform constant;
  CHECK(waveform_mean<double>(constant, -5, 1'000'000) == 1.0);
  CHECK(waveform_mean<float>(constant, 0, 1) == 1.0f);
  std::mt19937_64 rng(3);
  const Waveform f = fourier8(120.0, rng);
  const Waveform p = pulse(1000.37, -123.456, 37.3, 412.9, 0.2);
  for (int64_t t : {int64_t{-1'000'000'007}, int64_t{0}, int64_t{1}, int64_t{987'654'321}}) {
    CHECK(waveform_mean<double>(f, t, t) == waveform_value<double>(f, t));
    CHECK(waveform_mean<double>(p, t, t) == waveform_value<double>(p, t));
  }
}

TEST_CASE("U1 WaveformPrefix equals waveform_mean bitwise") {
  std::mt19937_64 rng(5);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  const Waveform p = pulse(1000.37, -123.456, 37.3, 412.9, 0.2);
  const int64_t span_a = -9'876'543, span_b = span_a + 20'000'000;  // 2e4 periods
  const WaveformPrefix prefix(p, span_a, span_b);
  int64_t compared = 0;
  auto check = [&](int64_t t0, int64_t t1) {
    const double a = prefix.mean(t0, t1), b = waveform_mean<double>(p, t0, t1);
    if (a != b) FAIL_CHECK("[" << t0 << ", " << t1 << "): prefix " << a << " != " << b);
    ++compared;
  };
  for (int i = 0; i < 5000; ++i) {
    const int64_t t0 = span_a + static_cast<int64_t>(u01(rng) * 2e7);
    const double scale = i % 3 == 0 ? 50.0 : (i % 3 == 1 ? 5000.0 : 2e7);
    const int64_t t1 = std::min(span_b, t0 + static_cast<int64_t>(u01(rng) * scale));
    check(t0, t1);
  }
  check(span_a, span_b);
  check(span_a, span_a + 1);
  check(span_b - 1, span_b);
  check(span_a, span_a);
  CHECK(compared == 5004);
  CHECK_THROWS_AS(prefix.mean(span_a - 1, span_a + 10), std::out_of_range);
  CHECK_THROWS_AS(prefix.mean(span_b - 10, span_b + 1), std::out_of_range);

  // Fourier and constant waveforms forward to waveform_mean.
  const Waveform f = fourier8(120.0, rng);
  const WaveformPrefix fp(f, -1'000'000, 50'000'000);
  for (int i = 0; i < 100; ++i) {
    const int64_t t0 = -1'000'000 + static_cast<int64_t>(u01(rng) * 4e7);
    const int64_t t1 = t0 + static_cast<int64_t>(u01(rng) * 1e7);
    CHECK(fp.mean(t0, t1) == waveform_mean<double>(f, t0, t1));
  }
  CHECK(WaveformPrefix(Waveform{}, 0, 10).mean(2, 7) == 1.0);
}

TEST_CASE("U1 WaveformPrefix equals waveform_mean bitwise where snapped periods overlap") {
  // The factory's triangle (duty 0.5, rise 0.5/f, so 2 rise + top = period) at 59.94 Hz: snapping ends some periods'
  // falls 1 ns past the next period's corner 0 (c3(n) = c0(n + 1) + 1), and starts others 1 ns after corner 4.
  const double f = 59.94, period_s = 1.0 / f, rise_s = 0.5 / f;
  Waveform w;
  w.type = WaveformType::kPulse;
  w.period_ns = period_s * 1e9;
  w.phase_ns = 0.0039331098403222291 * 1e9;
  w.rise_ns = rise_s * 1e9;
  w.top_ns = (0.5 * period_s - rise_s) * 1e9;
  w.floor_level = 0.0;
  w.peak_level = 2.0;
  const int64_t span_a = 0, span_b = 130'000'000'000;  // ~7800 periods
  const WaveformPrefix prefix(w, span_a, span_b);
  std::vector<int64_t> seams;  // c0(n + 1) where period n's fall overlaps it or a 1 ns gap precedes it
  int overlaps = 0;
  for (int64_t n = 0; n < 7800; ++n) {
    const int64_t c3 = balla::sensor::detail::pulse_corner(w, n, 3);
    const int64_t c4 = balla::sensor::detail::pulse_corner(w, n, 4);
    const int64_t next = balla::sensor::detail::pulse_corner(w, n + 1, 0);
    if (c3 > next) ++overlaps;
    if (c3 != next || c4 != next) seams.push_back(next);
  }
  REQUIRE(overlaps >= 1);
  int64_t compared = 0;
  auto check = [&](int64_t t0, int64_t t1) {
    t0 = std::clamp(t0, span_a, span_b);
    t1 = std::clamp(t1, t0, span_b);
    const double a = prefix.mean(t0, t1), b = waveform_mean<double>(w, t0, t1);
    if (a != b) FAIL_CHECK("[" << t0 << ", " << t1 << "): prefix " << a << " != " << b);
    ++compared;
  };
  const int64_t lengths[] = {1, 2, 3, 10, 1'000, 1'000'000, 8'341'675, 16'683'350, 40'000'000};
  for (const int64_t s : seams) {
    for (int64_t d = -3; d <= 3; ++d) {
      for (const int64_t len : lengths) {
        check(s + d, s + d + len);  // windows starting at the seam
        check(s + d - len, s + d);  // windows ending at it
      }
    }
  }
  std::mt19937_64 rng(23);
  std::uniform_int_distribution<int64_t> t(span_a, span_b);
  for (int i = 0; i < 2000; ++i) {
    const int64_t t0 = t(rng);
    check(t0, t0 + t(rng) / (i % 2 == 0 ? 1'000 : 1));
  }
  check(span_a, span_b);
  MESSAGE("U1 prefix at seams: " << seams.size() << " seams (" << overlaps << " overlaps), " << compared
                                 << " windows");
}
