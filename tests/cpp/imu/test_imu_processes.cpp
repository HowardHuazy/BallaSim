// U5, U6 and U9 (docs/design/phase3/outputs.md §8.1): the IMU's random processes of processes.h, each alone at a fixed
// coarse step over hours, against the exact analytic values of the same discrete processes at FIXED cluster sizes
// (R-DOC C6: no argmax location; with these tolerances 200 of 200 seeds pass, every tolerance >= 4 sd). Pinned seeds,
// so the outcome is reproducible; every axis must pass.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <string>
#include <vector>

#include "core/pmath/explog.h"
#include "core/rng/purpose.h"
#include "imu/params.h"
#include "imu/processes.h"

using namespace balla;
using namespace balla::imu;

namespace {

// The overlapping Allan variance of a rate series x_k (sample step dt) at the cluster size m, as R-DOC C6's adev():
// with S_j = sum_{i<j} x_i, AVAR = mean_j (S_{j+2m} - 2 S_{j+m} + S_j)^2 / (2 m^2). Streams with a ring of the last
// 2m + 1 sums (S_0 = 0 at slot 0).
class Allan {
 public:
  explicit Allan(int64_t m) : m_(static_cast<size_t>(m)), ring_(2 * m_ + 1, 0.0) {}
  void add(double x) {
    s_ += x;
    head_ = head_ + 1 == ring_.size() ? 0 : head_ + 1;  // the slot of S_n
    ring_[head_] = s_;
    if (++n_ >= 2 * m_) {
      const size_t back_m = head_ >= m_ ? head_ - m_ : head_ + m_ + 1;    // S_{n-m}
      const size_t back_2m = head_ + 1 == ring_.size() ? 0 : head_ + 1;  // S_{n-2m}
      const double d = (s_ - 2 * ring_[back_m]) + ring_[back_2m];
      sum_ += d * d;
      ++count_;
    }
  }
  double adev() const {
    const auto m = static_cast<double>(m_);
    return std::sqrt(sum_ / static_cast<double>(count_) / (2.0 * m * m));
  }

 private:
  size_t m_, head_ = 0, n_ = 0, count_ = 0;
  std::vector<double> ring_;
  double s_ = 0, sum_ = 0;
};

// The exact Allan variance of m-sample averages of a stationary AR(1) sequence with variance s^2 and lag-1
// correlation phi (R-DOC C6 avar_ar1): var(mean) - cov(adjacent means).
double avar_ar1(double s, double phi, int64_t m) {
  const auto md = static_cast<double>(m);
  double var = md, cov = 0, p = 1;
  for (int64_t d = 1; d < 2 * m; ++d) {
    p *= phi;
    if (d < m) var += 2 * static_cast<double>(m - d) * p;
    cov += static_cast<double>(m - std::abs(d - m)) * p;
  }
  return s * s * (var - cov) / (md * md);
}

double axis(const Vec3d& v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

SensorErrors none() { return SensorErrors{}; }

}  // namespace

TEST_CASE("U5: white noise, random walk and bias instability alone: Allan deviation at fixed tau") {
  // White: sigma / sqrt(dt) z at dt = 1 ms for 2 h; AD(1 s) = sigma (+-3 %).
  {
    SensorErrors e = none();
    e.noise_density = 1.0e-4;
    SensorNoise n({101, 0}, 0, e, NoiseMode::kWhite);
    std::vector<Allan> ad(3, Allan(1000));
    for (int64_t k = 0; k < 7'200'000; ++k) {
      const Vec3d x = n.next(1e-3).noise;
      for (int i = 0; i < 3; ++i) ad[i].add(axis(x, i));
    }
    for (int i = 0; i < 3; ++i) {
      const double r = ad[i].adev() / e.noise_density;
      MESSAGE("white axis " << i << ": AD(1 s) / sigma = " << r);
      CHECK(std::abs(r - 1) <= 0.03);
    }
  }
  // Random walk: b += sigma_b sqrt(dt) z at dt = 10 ms for 20 h; AD(3 s) = sigma_b sqrt(tau/3 (1 + 1/(2 m^2))) (+-2 %).
  {
    SensorErrors e = none();
    e.random_walk = 2.4e-5;
    SensorNoise n({102, 3}, 1, e, NoiseMode::kWhite);
    const int64_t m = 300;
    std::vector<Allan> ad(3, Allan(m));
    for (int64_t k = 0; k < 7'200'000; ++k) {
      const Vec3d x = n.next(1e-2).drift;
      for (int i = 0; i < 3; ++i) ad[i].add(axis(x, i));
    }
    const double expect = e.random_walk * std::sqrt(3.0 / 3 * (1 + 1 / (2.0 * m * m)));
    for (int i = 0; i < 3; ++i) {
      const double r = ad[i].adev() / expect;
      MESSAGE("random walk axis " << i << ": AD(3 s) / analytic = " << r);
      CHECK(std::abs(r - 1) <= 0.02);
    }
  }
  // Bias instability: Gauss-Markov with q = 0.664 B / 0.617 and T = tau_B / 1.89 = 10 / 1.89 s at dt = 50 ms for
  // 200 h; AD at tau = T, 1.89 T, 4 T (m = 106, 200, 423) against the discrete AR(1) Allan variance (+-2 %).
  {
    SensorErrors e = none();
    e.bias_instability = 1e-4;
    e.bias_instability_tau_s = 10;
    SensorNoise n({103, 1}, 0, e, NoiseMode::kWhite);
    const int64_t ms[3] = {106, 200, 423};
    std::vector<Allan> ad;
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 3; ++i) ad.emplace_back(ms[j]);
    }
    for (int64_t k = 0; k < 14'400'000; ++k) {
      const Vec3d x = n.next(5e-2).drift;
      for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) ad[static_cast<size_t>(3 * j + i)].add(axis(x, i));
      }
    }
    const double q = 0.664 * e.bias_instability / 0.617, T = 10 / 1.89;
    for (int j = 0; j < 3; ++j) {
      const double expect = std::sqrt(avar_ar1(q, std::exp(-5e-2 / T), ms[j]));
      for (int i = 0; i < 3; ++i) {
        const double r = ad[static_cast<size_t>(3 * j + i)].adev() / expect;
        MESSAGE("Gauss-Markov m = " << ms[j] << " axis " << i << ": AD / analytic = " << r);
        CHECK(std::abs(r - 1) <= 0.02);
      }
    }
  }
}

TEST_CASE("U6: shaped noise: per-sample sd sigma / sqrt(2 tau_c), two-sided PSD sigma^2 below f_c / 4") {
  SensorErrors e = none();
  e.noise_density = 1.0e-4;
  e.filter = Filter::kFirstOrder;
  e.cutoff_hz = 92;
  e.group_delay_s = 3.9e-3;
  SensorNoise n({104, 2}, 0, e, NoiseMode::kShaped);
  const double dt = 1e-3, sd = e.noise_density / std::sqrt(2 * tau_c_s(e));
  const int seg = 1000, segments = 1200;  // 1 s segments: periodogram bins at 1 Hz
  const int bins = static_cast<int>(e.cutoff_hz / 4);  // 1 .. 23 Hz
  std::vector<std::complex<double>> twiddle(static_cast<size_t>(bins * seg));
  for (int f = 1; f <= bins; ++f) {
    for (int j = 0; j < seg; ++j) {
      twiddle[static_cast<size_t>((f - 1) * seg + j)] = std::polar(1.0, -2 * std::numbers::pi * f * j / seg);
    }
  }
  std::vector<double> sum2(3, 0.0), psd(static_cast<size_t>(bins), 0.0);
  std::vector<double> x(static_cast<size_t>(3 * seg));
  for (int s = 0; s < segments; ++s) {
    for (int j = 0; j < seg; ++j) {
      const Vec3d v = n.next(dt).noise;
      for (int i = 0; i < 3; ++i) {
        x[static_cast<size_t>(i * seg + j)] = axis(v, i);
        sum2[static_cast<size_t>(i)] += axis(v, i) * axis(v, i);
      }
    }
    for (int i = 0; i < 3; ++i) {
      for (int f = 1; f <= bins; ++f) {
        std::complex<double> acc = 0;
        const std::complex<double>* w = &twiddle[static_cast<size_t>((f - 1) * seg)];
        for (int j = 0; j < seg; ++j) acc += x[static_cast<size_t>(i * seg + j)] * w[j];
        psd[static_cast<size_t>(f - 1)] += dt / seg * std::norm(acc);
      }
    }
  }
  for (int i = 0; i < 3; ++i) {
    const double r = std::sqrt(sum2[static_cast<size_t>(i)] / (seg * segments)) / sd;
    MESSAGE("shaped axis " << i << ": sd / (sigma / sqrt(2 tau_c)) = " << r);
    CHECK(std::abs(r - 1) <= 0.02);
  }
  const double sigma2 = e.noise_density * e.noise_density;
  double mean = 0, worst = 0;
  for (int f = 1; f <= bins; ++f) {
    const double r = psd[static_cast<size_t>(f - 1)] / (3.0 * segments) / sigma2;
    mean += r / bins;
    worst = std::max(worst, std::abs(r - 1));
    CHECK(std::abs(r - 1) <= 0.10);
  }
  MESSAGE("shaped PSD over 1-" << bins << " Hz / sigma^2: mean " << mean << ", worst bin off by " << worst);
  CHECK(std::abs(mean - 1) <= 0.10);
}

TEST_CASE("U9: the sample clock: mean period, ADEV_y of the wander and Y_N / N") {
  // clock_adev = 0: tau_k = (k + phi) T, the mean period 1 / rate_hz to 1e-12 (any rate, either start phase).
  for (const double rate : {1000.0, 1026.45, 333.0}) {
    for (const StartPhase start : {StartPhase::kZero, StartPhase::kRandom}) {
      SampleClock c({105, 0}, rate, start, 0.0);
      const double tau0 = c.tau_ns();
      CHECK(tau0 >= 0);
      CHECK(tau0 < 1e9 / rate);
      if (start == StartPhase::kZero) CHECK(tau0 == 0);
      const int64_t n = 1'000'000;
      for (int64_t k = 0; k < n; ++k) c.next();
      const double period = (c.tau_ns() - tau0) / static_cast<double>(n);
      CHECK(std::abs(period / (1e9 / rate) - 1) <= 1e-12);
      if (rate == 1000.0 && start == StartPhase::kZero) CHECK(c.tau_ns() == 1e12);  // integral instants
    }
  }
  // clock_adev = 4.5e-5 over 4 h at 1 kHz: ADEV_y at 0.1, 1 and 10 s against the analytic value of the 3-GM sum
  // (+-2, 3, 8 %); the realized mean period equals T (1 - Y_N / N) from the stored draws; |Y_N / N| <= 6.7e-6.
  const double adev = 4.5e-5, T = 1e-3;
  SampleClock c({106, 1}, 1000, StartPhase::kRandom, adev);
  const double tau0 = c.tau_ns();
  const int64_t ms[3] = {100, 1000, 10000};
  std::vector<Allan> ad = {Allan(ms[0]), Allan(ms[1]), Allan(ms[2])};
  const int64_t n = 14'400'000;
  double Y = 0;
  for (int64_t k = 0; k < n; ++k) {
    c.next();
    const double y = c.y();
    Y += y;
    for (Allan& a : ad) a.add(y);
  }
  const double tol[3] = {0.02, 0.03, 0.08};
  for (int j = 0; j < 3; ++j) {
    double var = 0;
    for (const double Ti : kClockTau) var += avar_ar1(kClockScale * adev, std::exp(-T / Ti), ms[j]);
    const double r = ad[static_cast<size_t>(j)].adev() / std::sqrt(var);
    MESSAGE("clock ADEV_y(" << ms[j] * T << " s) / analytic = " << r << " (analytic / clock_adev = "
                             << std::sqrt(var) / adev << ")");
    CHECK(std::abs(r - 1) <= tol[j]);
  }
  CHECK(Y == c.Y());
  const double period = (c.tau_ns() - tau0) / static_cast<double>(n);
  CHECK(std::abs(period / (1e6 * (1 - Y / static_cast<double>(n))) - 1) <= 1e-12);
  MESSAGE("Y_N / N = " << Y / static_cast<double>(n));
  CHECK(std::abs(Y / static_cast<double>(n)) <= 6.7e-6);
}

TEST_CASE("processes: draws keyed by (seed, camera, pixel, k) only; off processes draw nothing") {
  SensorErrors e = none();
  e.noise_density = 1e-4;
  e.random_walk = 1e-5;
  e.bias_instability = 1e-4;
  e.bias = {0.1, -0.2, 0.3};
  e.bias_sd = {0.01, 0.02, 0.03};
  // Two sigmas, the same z: the outputs scale exactly (T6's paired runs).
  SensorErrors e2 = e;
  e2.noise_density = 2e-4;
  SensorNoise a({7, 2}, 0, e, NoiseMode::kWhite), b({7, 2}, 0, e2, NoiseMode::kWhite);
  CHECK(a.b0().x == b.b0().x);
  CHECK(a.b0().x == 0.1 + 0.01 * normals({7, 2}, {rng::Purpose::kImuBiasInit, 0}, 0).x);
  for (int k = 0; k < 100; ++k) {
    const double dt = 1e-3 * (1 + 1e-3 * k);
    const SensorNoise::Sample sa = a.next(dt), sb = b.next(dt);
    CHECK(sb.noise.y == 2 * sa.noise.y);
    CHECK(sa.drift.z == sb.drift.z);
  }
  // Sensors, cameras and seeds draw from distinct streams.
  SensorNoise g({7, 2}, 0, e, NoiseMode::kWhite), acc({7, 2}, 1, e, NoiseMode::kWhite);
  SensorNoise cam({7, 3}, 0, e, NoiseMode::kWhite), seed({8, 2}, 0, e, NoiseMode::kWhite);
  const double x0 = g.next(1e-3).noise.x;
  CHECK(x0 != acc.next(1e-3).noise.x);
  CHECK(x0 != cam.next(1e-3).noise.x);
  CHECK(x0 != seed.next(1e-3).noise.x);
  // All off: zero, and b0 = bias exactly.
  SensorErrors off = none();
  off.bias = {0.5, 0, -0.25};
  SensorNoise z({7, 2}, 1, off, NoiseMode::kWhite);
  CHECK_FALSE(z.on());
  CHECK(z.b0().x == 0.5);
  CHECK(z.b0().z == -0.25);
  for (int k = 0; k < 10; ++k) {
    const SensorNoise::Sample s = z.next(1e-3);
    CHECK((s.drift.x == 0 && s.drift.y == 0 && s.drift.z == 0 && s.noise.x == 0 && s.noise.y == 0 && s.noise.z == 0));
  }
  // The stationary starts: the shaped noise and the Gauss-Markov bias draw their first values from kImuBiasInit.
  SensorErrors sh = none();
  sh.noise_density = 1e-4;
  sh.filter = Filter::kFirstOrder;
  sh.cutoff_hz = 92;
  sh.group_delay_s = 3.9e-3;
  SensorNoise shaped({9, 0}, 1, sh, NoiseMode::kShaped);
  const double sd = 1e-4 / std::sqrt(2 * tau_c_s(sh));
  CHECK(shaped.next(1e-3).noise.x == sd * normals({9, 0}, {rng::Purpose::kImuBiasInit, 5}, 0).x);
}

TEST_CASE("processes: the Philox key of every draw (the §4.4 table), replayed bitwise") {
  // Each process alone, replayed from normals() at the key the table gives: a swapped pixel, tag or index (k - 1 for
  // k) changes every IMU output while the statistics (U5, U6, U9) and the self-consistency (U11) still pass.
  using rng::Purpose;
  const Stream st{0x5eed1234abcdULL, 2};
  const auto z = [&](Purpose tag, uint32_t pixel, uint64_t k) { return normals(st, {tag, pixel}, k); };
  const auto eq = [](const Vec3d& a, const Vec3d& b) { return a.x == b.x && a.y == b.y && a.z == b.z; };

  // The clock: phi = (v0 + 0.5) / 2^32 of (kImuClock, pixel 0, index 0); the wander's stationary start g_{i,0} =
  // s z(kImuClock, 2, 0)_i and its steps g_{i,k} = phi_i g_{i,k-1} + s sqrt(1 - phi_i^2) z(kImuClock, 1, k)_i.
  const double rate = 1000, T_ns = 1e9 / rate, T_s = 1.0 / rate;
  const uint32_t v0 = rng::draw(st.seed, Purpose::kImuClock, st.camera, 0, 0).v[0];
  const double phi = (static_cast<double>(v0) + 0.5) * 0x1p-32;
  const SampleClock still(st, rate, StartPhase::kRandom, 0);
  CHECK(still.phase() == phi);
  CHECK(still.tau_ns() == phi * T_ns);
  CHECK(SampleClock(st, rate, StartPhase::kZero, 0).phase() == 0);
  const double adev = 4.5e-5, s = kClockScale * adev;
  SampleClock clock(st, rate, StartPhase::kRandom, adev);
  CHECK(clock.phase() == phi);
  CHECK(clock.tau_ns() == phi * T_ns);
  double g[3], a[3], c[3];
  const Vec3d g0 = z(Purpose::kImuClock, 2, 0);
  for (int i = 0; i < 3; ++i) {
    const double r = T_s / kClockTau[static_cast<size_t>(i)];
    a[i] = pmath::exp(-r);
    c[i] = s * std::sqrt(-pmath::expm1(-2 * r));
    g[i] = s * axis(g0, i);
  }
  double Y = 0;
  for (uint64_t k = 1; k <= 3; ++k) {
    clock.next();
    const Vec3d zk = z(Purpose::kImuClock, 1, k);
    for (int i = 0; i < 3; ++i) g[i] = a[i] * g[i] + c[i] * axis(zk, i);
    const double y = (g[0] + g[1]) + g[2];
    Y += y;
    CHECK(clock.y() == y);
    CHECK(clock.Y() == Y);
    CHECK(clock.tau_ns() == (static_cast<double>(k) + phi) * T_ns - T_ns * Y);
  }

  // The sensor processes, per sensor s (pixel s, 2 + s, 4 + s), at uneven steps (one repeated: the cached
  // coefficients).
  const double dt[4] = {1e-3, 1.25e-3, 1.25e-3, 0.8e-3};
  for (int sensor = 0; sensor < 2; ++sensor) {
    CAPTURE(sensor);
    const auto px = static_cast<uint32_t>(sensor);
    // The turn-on bias: b0 = bias + bias_sd (.) z(kImuBiasInit, s, 0).
    SensorErrors tb = none();
    tb.bias = {0.1, -0.2, 0.3};
    tb.bias_sd = {0.01, 0.02, 0.03};
    const Vec3d zb = z(Purpose::kImuBiasInit, px, 0);
    CHECK(eq(SensorNoise(st, sensor, tb, NoiseMode::kWhite).b0(),
             {0.1 + 0.01 * zb.x, -0.2 + 0.02 * zb.y, 0.3 + 0.03 * zb.z}));
    // White: n_k = sigma / sqrt(dt_k) z(kImuNoise, s, k).
    SensorErrors w = none();
    w.noise_density = 1e-4;
    SensorNoise white(st, sensor, w, NoiseMode::kWhite);
    for (uint64_t k = 0; k < 4; ++k) {
      CHECK(eq(white.next(dt[k]).noise, (1e-4 / std::sqrt(dt[k])) * z(Purpose::kImuNoise, px, k)));
    }
    // Random walk: b_0 = 0, b_k = b_{k-1} + sigma_b sqrt(dt_k) z(kImuBiasWalk, s, k) from k = 1.
    SensorErrors rw = none();
    rw.random_walk = 2e-5;
    SensorNoise walk(st, sensor, rw, NoiseMode::kWhite);
    Vec3d b{0, 0, 0};
    for (uint64_t k = 0; k < 4; ++k) {
      if (k > 0) b += (2e-5 * std::sqrt(dt[k])) * z(Purpose::kImuBiasWalk, px, k);
      CHECK(eq(walk.next(dt[k]).drift, b));
    }
    // Bias instability: b_gm,0 = q z(kImuBiasInit, 2 + s, 0), b_gm,k = phi_k b_gm,k-1 + q sqrt(1 - phi_k^2)
    // z(kImuBiasWalk, 2 + s, k), q = 0.664 B / 0.617, T = tau_B / 1.89.
    SensorErrors bi = none();
    bi.bias_instability = 1e-4;
    bi.bias_instability_tau_s = 20;
    SensorNoise gm(st, sensor, bi, NoiseMode::kWhite);
    const double q = kAllanFloorOverB * 1e-4 / kGmPeakAdOverQ, T_gm = 20 / kGmPeakTauOverT;
    CHECK(std::abs(q / 1e-4 - 1.076) < 1e-3);
    Vec3d x = q * z(Purpose::kImuBiasInit, 2 + px, 0);
    CHECK(eq(gm.next(dt[0]).drift, x));
    for (uint64_t k = 1; k < 4; ++k) {
      const double r = dt[k] / T_gm;
      x = pmath::exp(-r) * x + (q * std::sqrt(-pmath::expm1(-2 * r))) * z(Purpose::kImuBiasWalk, 2 + px, k);
      CHECK(eq(gm.next(dt[k]).drift, x));
    }
    // Shaped: n_0 = sd z(kImuBiasInit, 4 + s, 0), n_k = phi_k n_{k-1} + sd sqrt(1 - phi_k^2) z(kImuNoise, s, k),
    // sd = sigma / sqrt(2 tau_c), time constant tau_c.
    SensorErrors sh = none();
    sh.noise_density = 1e-4;
    sh.filter = Filter::kFirstOrder;
    sh.cutoff_hz = 92;
    sh.group_delay_s = 3.9e-3;
    SensorNoise shaped(st, sensor, sh, NoiseMode::kShaped);
    const double tau = tau_c_s(sh), sd = 1e-4 / std::sqrt(2 * tau);
    Vec3d n = sd * z(Purpose::kImuBiasInit, 4 + px, 0);
    CHECK(eq(shaped.next(dt[0]).noise, n));
    for (uint64_t k = 1; k < 4; ++k) {
      const double r = dt[k] / tau;
      n = pmath::exp(-r) * n + (sd * std::sqrt(-pmath::expm1(-2 * r))) * z(Purpose::kImuNoise, px, k);
      CHECK(eq(shaped.next(dt[k]).noise, n));
    }
  }
}
