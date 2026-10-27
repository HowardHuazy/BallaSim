// The device model through ImuSimulator (docs/design/phase3/outputs.md §4.2-4.6, §8.1): the error matrix and turn-on
// bias, the additive processes in a paired run, U7 (filter delay), U8 (quantization, saturation and the pre-clip),
// U10 (camera clock and stamps, the drop below t_offset) and U11 (determinism).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include "core/math/so3.h"
#include "core/rng/purpose.h"
#include "core/time/time.h"
#include "imu/filter.h"
#include "imu/imu_test_util.h"
#include "imu/processes.h"
#include "imu/simulator.h"
#include "sensor/params.h"
#include "sensor/readout.h"
#include "sensor/sensor_model.h"

using namespace balla;
using namespace balla::imu;
using balla::imu::test::ideal;
using balla::imu::test::input;
using balla::imu::test::make_rig;
using balla::imu::test::random_rig;
using balla::imu::test::Rig;

namespace {

constexpr double kPi = std::numbers::pi;
const double kLsbGyro = 250.0 / 32768 * kPi / 180;  // the driver's quantum (§2.4)

// A rig whose controls at t0 + i h are R_z(theta(t)) and pos(t). The spline smooths the controls, which changes
// amplitudes but keeps a sinusoid a sinusoid of the same frequency and a polynomial a polynomial of the same degree.
template <class Theta, class Pos>
Rig sampled_rig(Theta theta, Pos pos, double seconds, int32_t h_ns = 5'000'000) {
  const int n = static_cast<int>(seconds * 1e9 / h_ns) + 6;
  std::vector<Quat<double>> q;
  std::vector<Vec3d> p;
  for (int i = 0; i < n; ++i) {
    const double t = i * (h_ns * 1e-9);
    q.push_back(so3::exp(Vec3d{0, 0, theta(t)}));
    p.push_back(pos(t));
  }
  return make_rig(q, p, 6, h_ns);
}

Vec3d zero(double) { return {0, 0, 0}; }

// The phase psi of the least-squares fit a cos(w t) + b sin(w t) + c = A cos(w t - psi) + c.
double phase_of(const std::vector<double>& t, const std::vector<double>& x, double w) {
  double s[3][3] = {}, r[3] = {};
  for (size_t i = 0; i < t.size(); ++i) {
    const double f[3] = {std::cos(w * t[i]), std::sin(w * t[i]), 1.0};
    for (int a = 0; a < 3; ++a) {
      r[a] += f[a] * x[i];
      for (int b = 0; b < 3; ++b) s[a][b] += f[a] * f[b];
    }
  }
  const auto det = [](const double m[3][3]) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  };
  double coef[2];
  for (int c = 0; c < 2; ++c) {  // Cramer's rule for a and b
    double m[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) m[a][b] = b == c ? r[a] : s[a][b];
    }
    coef[c] = det(m) / det(s);
  }
  return std::atan2(coef[1], coef[0]);
}

bool same(const ImuBlock& a, const ImuBlock& b) {
  return a.t_true_ns == b.t_true_ns && a.t_ns == b.t_ns && a.gyro == b.gyro && a.accel == b.accel &&
         a.counts_gyro == b.counts_gyro && a.counts_accel == b.counts_accel && a.flags == b.flags &&
         a.omega_true == b.omega_true && a.f_true == b.f_true && a.bias_gyro == b.bias_gyro &&
         a.bias_accel == b.bias_accel && a.q_W_I == b.q_W_I && a.p_W_I == b.p_W_I;
}

void append(ImuBlock& to, const ImuBlock& b) {
  const auto cat = [](auto& x, const auto& y) { x.insert(x.end(), y.begin(), y.end()); };
  cat(to.t_true_ns, b.t_true_ns);
  cat(to.t_ns, b.t_ns);
  cat(to.gyro, b.gyro);
  cat(to.accel, b.accel);
  cat(to.counts_gyro, b.counts_gyro);
  cat(to.counts_accel, b.counts_accel);
  cat(to.flags, b.flags);
  cat(to.omega_true, b.omega_true);
  cat(to.f_true, b.f_true);
  cat(to.bias_gyro, b.bias_gyro);
  cat(to.bias_accel, b.bias_accel);
  cat(to.q_W_I, b.q_W_I);
  cat(to.p_W_I, b.p_W_I);
}

// Samples [from, n) of b: every array cut at the same sample.
ImuBlock tail(const ImuBlock& b, size_t from) {
  ImuBlock t;
  const auto cut = [from](auto& x, const auto& y, size_t w) {
    if (!y.empty()) x.assign(y.begin() + static_cast<std::ptrdiff_t>(w * from), y.end());
  };
  cut(t.t_true_ns, b.t_true_ns, 1);
  cut(t.t_ns, b.t_ns, 1);
  cut(t.gyro, b.gyro, 3);
  cut(t.accel, b.accel, 3);
  cut(t.counts_gyro, b.counts_gyro, 3);
  cut(t.counts_accel, b.counts_accel, 3);
  cut(t.flags, b.flags, 1);
  cut(t.omega_true, b.omega_true, 3);
  cut(t.f_true, b.f_true, 3);
  cut(t.bias_gyro, b.bias_gyro, 3);
  cut(t.bias_accel, b.bias_accel, 3);
  cut(t.q_W_I, b.q_W_I, 4);
  cut(t.p_W_I, b.p_W_I, 3);
  return t;
}

// A device with every error on: the clock wander and a random start phase, shaped noise through both filters, bias
// walks and instability, turn-on biases, the error matrices and both quanta.
ImuParams full_device() {
  ImuParams p = ideal(1026.45, true);
  p.start_phase = StartPhase::kRandom;
  p.clock_adev = 4.5e-5;
  p.noise_mode = NoiseMode::kShaped;
  p.stamp_offset_ns = 1000;
  p.seed = 42;
  p.gyro.noise_density = 1e-4;
  p.accel.noise_density = 1.2e-3;
  p.gyro.random_walk = 2.4e-5;
  p.accel.random_walk = 5e-3;
  p.gyro.bias_instability = 1e-4;
  p.gyro.bias = {-0.0302, 0.0151, 0.0176};
  p.gyro.bias_sd = {1.4e-3, 2.7e-3, 2e-3};
  p.accel.bias = {0.046, 0.207, -0.167};
  p.accel.bias_sd = {0.04, 0.085, 0.05};
  p.gyro.scale = {-0.008, -0.0008, -0.0051};
  p.gyro.nonorth = {0.0007, 0.0034, -0.0007};
  p.accel.scale = {-0.0058, -0.0075, 0.0017};
  p.accel.nonorth = {0.0021, 0.0062, 0.0002};
  p.accel.misalign_deg = {0.3, -0.2, 0.1};
  p.gyro.lsb = kLsbGyro;
  p.accel.lsb = 5.987548828125e-4;
  p.T_cam_imu = test::transform({{{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}}}, {-1.3e-3, -0.5e-3, -40.9e-3});
  return p;
}

}  // namespace

TEST_CASE("error model (§4.4): u = M x + b0, M = Exp(misalign) (I + diag(scale) + Sym(nonorth)); the bias truth") {
  const Rig rig = random_rig(11, 120, 0.02, 2.5e-4, 6, 5'000'000);
  ImuParams p = ideal(1000, false);
  p.seed = 9;
  p.gyro.scale = {-0.008, 0.002, -0.005};
  p.gyro.nonorth = {0.0007, 0.0034, -0.0007};
  p.gyro.misalign_deg = {0.5, -1.0, 2.0};
  p.gyro.bias = {-0.03, 0.015, 0.017};
  p.gyro.bias_sd = {1e-3, 2e-3, 3e-3};
  p.accel.scale = {0.01, -0.02, 0.03};
  p.accel.nonorth = {0.05, -0.04, 0.03};
  p.accel.misalign_deg = {-3.0, 4.0, 5.0};
  p.accel.bias = {0.2, -0.1, 0.05};
  ImuInput in = input(rig, 60'000'000, 500'000'000);
  in.camera_index = 3;
  ImuSimulator sim(p, in);
  const ImuBlock b = sim.advance_to(INT64_MAX);
  REQUIRE(b.size() == 440);
  for (int s = 0; s < 2; ++s) {
    const SensorErrors& e = s == 0 ? p.gyro : p.accel;
    // M with libm's Rodrigues formula, independently of so3::exp.
    const double v[3] = {e.misalign_deg[0] * kPi / 180, e.misalign_deg[1] * kPi / 180, e.misalign_deg[2] * kPi / 180};
    const double th = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    const double k[3] = {v[0] / th, v[1] / th, v[2] / th};
    double r[3][3], m[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        const double kx[3][3] = {{0, -k[2], k[1]}, {k[2], 0, -k[0]}, {-k[1], k[0], 0}};
        double kk = 0;
        for (int l = 0; l < 3; ++l) kk += kx[i][l] * kx[l][j];
        r[i][j] = (i == j ? 1.0 : 0.0) + std::sin(th) * kx[i][j] + (1 - std::cos(th)) * kk;
      }
    }
    const double pm[3][3] = {{1 + e.scale[0], e.nonorth[0], e.nonorth[1]},
                             {e.nonorth[0], 1 + e.scale[1], e.nonorth[2]},
                             {e.nonorth[1], e.nonorth[2], 1 + e.scale[2]}};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) m[i][j] = r[i][0] * pm[0][j] + r[i][1] * pm[1][j] + r[i][2] * pm[2][j];
    }
    const Vec3d z = normals({9, 3}, {rng::Purpose::kImuBiasInit, static_cast<uint32_t>(s)}, 0);
    const double b0[3] = {e.bias[0] + e.bias_sd[0] * z.x, e.bias[1] + e.bias_sd[1] * z.y, e.bias[2] + e.bias_sd[2] * z.z};
    const std::vector<double>& out = s == 0 ? b.gyro : b.accel;
    const std::vector<double>& x = s == 0 ? b.omega_true : b.f_true;
    const std::vector<double>& bias = s == 0 ? b.bias_gyro : b.bias_accel;
    double worst = 0;
    for (size_t n = 0; n < b.size(); ++n) {
      for (int i = 0; i < 3; ++i) {
        const double want = m[i][0] * x[3 * n] + m[i][1] * x[3 * n + 1] + m[i][2] * x[3 * n + 2] + b0[i];
        worst = std::max(worst, std::abs(out[3 * n + static_cast<size_t>(i)] - want) /
                                    (std::abs(want) + std::abs(x[3 * n + static_cast<size_t>(i)]) + 1e-300));
        CHECK(bias[3 * n + static_cast<size_t>(i)] == b0[i]);  // no walks: b0 at every sample, bitwise
      }
    }
    MESSAGE(std::string(s == 0 ? "gyro" : "accel") << ": max relative |z - (M x + b0)| = " << worst);
    CHECK(worst <= 1e-14);
  }
  CHECK(b.counts_gyro.empty());
  CHECK(b.counts_accel.empty());
}

TEST_CASE("paired runs (T6's form): A - B is b_rw + b_gm + n, replayed in seconds through the dropped samples") {
  // A has every additive process on, both lsb null; B is A with noise_density, random_walk and bias_instability 0.
  // Draws do not depend on a sigma (§4.4), so B publishes A's filter output y, and A = (y + drift) + n bitwise, with
  // drift and n replayed by SensorNoise on dt_0 = T and dt_k = (tau_k - tau_{k-1}) / 1e9 s at every sample, the ones
  // dropped below t_offset included; A's bias truth is b0 + drift. Instants on the plain grid (tau_k = k T, an
  // independent dt) and with the clock's wander and start phase (tau_k replayed by SampleClock).
  const Rig rig = random_rig(5, 200, 0.02, 2.5e-4, 6, 5'000'000);
  const int64_t t_b = 60'000'000;
  for (const bool jitter : {false, true}) {
    for (const NoiseMode mode : {NoiseMode::kShaped, NoiseMode::kWhite}) {
      CAPTURE(jitter);
      CAPTURE(static_cast<int>(mode));
      ImuParams a = full_device();
      a.gyro.lsb = a.accel.lsb = std::nullopt;
      a.noise_mode = mode;
      a.accel.bias_instability = 2e-3;
      if (!jitter) {
        a.clock_adev = 0;
        a.start_phase = StartPhase::kZero;
      }
      ImuParams b = a;
      for (SensorErrors* e : {&b.gyro, &b.accel}) e->noise_density = e->random_walk = e->bias_instability = 0;
      ImuInput in = input(rig, t_b, 400'000'000);
      in.camera_index = 2;
      // No quantum, offset or drift: stamp = t_true + 1000 ns; the samples with t_true below t_b + 5 ms are dropped.
      in.t_offset_us = (t_b + 5'000'000 + 1000) / 1000;
      ImuSimulator sa(a, in), sb(b, in);
      const ImuBlock A = sa.advance_to(INT64_MAX), B = sb.advance_to(INT64_MAX);
      const int64_t dropped = sa.stats().dropped_before_t_offset;
      CHECK(dropped >= 5);
      CHECK(sb.stats().dropped_before_t_offset == dropped);
      REQUIRE(A.size() > 300);
      REQUIRE(B.size() == A.size());
      CHECK(A.t_true_ns == B.t_true_ns);
      CHECK(A.t_true_ns.front() >= t_b + 5'000'000);
      const Stream st{a.seed, 2};
      SampleClock clock(st, a.rate_hz, a.start_phase, a.clock_adev);
      SensorNoise noise[2] = {SensorNoise(st, 0, a.gyro, mode), SensorNoise(st, 1, a.accel, mode)};
      Vec3d b0[2];
      for (int s = 0; s < 2; ++s) {  // b0 = bias + bias_sd (.) z(kImuBiasInit, s, 0), from the table
        const SensorErrors& e = s == 0 ? a.gyro : a.accel;
        const Vec3d z = normals(st, {rng::Purpose::kImuBiasInit, static_cast<uint32_t>(s)}, 0);
        b0[s] = {e.bias[0] + e.bias_sd[0] * z.x, e.bias[1] + e.bias_sd[1] * z.y, e.bias[2] + e.bias_sd[2] * z.z};
      }
      int64_t wrong = 0;
      double prev = 0, max_drift[2] = {0, 0}, max_noise[2] = {0, 0};
      size_t j = 0;
      for (uint64_t k = 0; j < A.size(); ++k) {
        if (k > 0) clock.next();
        const double tau = jitter ? clock.tau_ns() : static_cast<double>(k) * (1e9 / a.rate_hz);
        const double dt = k == 0 ? 1.0 / a.rate_hz : (tau - prev) / 1e9;
        prev = tau;
        const SensorNoise::Sample e[2] = {noise[0].next(dt), noise[1].next(dt)};
        if (k < static_cast<uint64_t>(dropped)) continue;
        wrong += A.t_true_ns[j] != t_b + std::llround(tau);
        for (int s = 0; s < 2; ++s) {
          const std::vector<double>& za = s == 0 ? A.gyro : A.accel;
          const std::vector<double>& zb = s == 0 ? B.gyro : B.accel;
          const std::vector<double>& ba = s == 0 ? A.bias_gyro : A.bias_accel;
          const std::vector<double>& bb = s == 0 ? B.bias_gyro : B.bias_accel;
          const Vec3d y{zb[3 * j], zb[3 * j + 1], zb[3 * j + 2]};
          const Vec3d want = (y + e[s].drift) + e[s].noise, bias = b0[s] + e[s].drift;
          for (int i = 0; i < 3; ++i) {
            const size_t at = 3 * j + static_cast<size_t>(i);
            wrong += za[at] != detail::at(want, i);
            wrong += ba[at] != detail::at(bias, i);
            wrong += bb[at] != detail::at(b0[s], i);
            max_drift[s] = std::max(max_drift[s], std::abs(detail::at(e[s].drift, i)));
            max_noise[s] = std::max(max_noise[s], std::abs(detail::at(e[s].noise, i)));
          }
        }
        ++j;
      }
      CHECK(wrong == 0);
      for (int s = 0; s < 2; ++s) {  // not vacuous: every process moved the output
        CHECK(max_drift[s] > 0);
        CHECK(max_noise[s] > 0);
      }
      CHECK(A.gyro != B.gyro);
      CHECK(A.accel != B.accel);
    }
  }
}

TEST_CASE("U7: the filter delay at 1 Hz equals d + atan(2 pi f tau_c) / (2 pi f) = tau_table - w^2 tau_c^3 / 3") {
  // gyro: omega about z (rotation 0.5 sin(2 pi t) rad); accel: translation 0.1 sin(2 pi t) m along x, the IMU at the
  // rig origin. The delay is the phase lag of the output behind the truth at the same instants, from least-squares
  // fits over 20 whole periods.
  const double w = 2 * kPi;
  const ImuParams p = ideal(1000, true);
  double delay[2];
  for (int s = 0; s < 2; ++s) {
    const Rig rig = s == 0 ? sampled_rig([&](double t) { return 0.5 * std::sin(w * t); }, zero, 25.0)
                           : sampled_rig([](double) { return 0.0; },
                                         [&](double t) { return Vec3d{0.1 * std::sin(w * t), 0, 0}; }, 25.0);
    ImuSimulator sim(p, input(rig, 1'000'000'000, 23'000'000'000));
    const ImuBlock b = sim.advance_to(INT64_MAX);
    const size_t axis = s == 0 ? 2 : 0;
    const std::vector<double>& out = s == 0 ? b.gyro : b.accel;
    const std::vector<double>& truth = s == 0 ? b.omega_true : b.f_true;
    std::vector<double> t, x, y;
    for (size_t k = 0; k < b.size(); ++k) {
      if (b.t_true_ns[k] < 2'000'000'000 || b.t_true_ns[k] >= 22'000'000'000) continue;
      t.push_back(static_cast<double>(b.t_true_ns[k]) * 1e-9);
      x.push_back(truth[3 * k + axis]);
      y.push_back(out[3 * k + axis]);
    }
    REQUIRE(t.size() == 20'000);
    double lag = phase_of(t, y, w) - phase_of(t, x, w);
    lag = std::remainder(lag, 2 * kPi);
    delay[s] = lag / w;
    const SensorErrors& e = s == 0 ? p.gyro : p.accel;
    const double tau = tau_c_s(e);
    const double expect = e.group_delay_s - w * w * tau * tau * tau / 3;
    MESSAGE(std::string(s == 0 ? "gyro" : "accel") << ": delay " << delay[s] * 1e3 << " ms, expected " << expect * 1e3
                                         << " ms (off by " << (delay[s] - expect) * 1e9 << " ns)");
    CHECK(std::abs(delay[s] - expect) <= 1e-6);
  }
  MESSAGE("gyro - accel delay: " << (delay[0] - delay[1]) * 1e3 << " ms");
  CHECK(std::abs((delay[0] - delay[1]) - 1.020e-3) <= 2e-6);
}

TEST_CASE("U8(a): quantization and saturation on a +-300 deg/s ramp") {
  // omega_z = 300 deg/s^2 (t - 1.5 s): a ramp through both rails over [0.4 s, 2.6 s), filter none.
  const double alpha = 300 * kPi / 180;
  const Rig rig = sampled_rig([&](double t) { return 0.5 * alpha * (t - 1.5) * (t - 1.5); }, zero, 3.0);
  ImuParams p = ideal(1000, false);
  p.gyro.lsb = kLsbGyro;
  ImuSimulator sim(p, input(rig, 400'000'000, 2'600'000'000));
  const ImuBlock b = sim.advance_to(INT64_MAX);
  REQUIRE(b.size() == 2200);
  REQUIRE(b.counts_gyro.size() == 3 * b.size());
  CHECK(b.counts_accel.empty());
  int64_t low = 0, high = 0, mismatched = 0;
  for (size_t k = 0; k < b.size(); ++k) {
    for (size_t i = 0; i < 3; ++i) {
      const double r = b.omega_true[3 * k + i] / kLsbGyro;
      const bool above = r >= 32767.5, below = r <= -32768.5;
      const int64_t c = above ? 32767 : (below ? -32768 : std::llround(r));
      mismatched += b.counts_gyro[3 * k + i] != c;
      mismatched += b.gyro[3 * k + i] != static_cast<double>(c) * kLsbGyro;  // published == count lsb, bitwise
      mismatched += ((b.flags[k] >> i) & 1) != (above || below ? 1 : 0);
      low += below;
      high += above;
    }
    CHECK((b.flags[k] & kFlagRailAccel) == 0);
  }
  CHECK(mismatched == 0);
  MESSAGE("samples at the rails: " << low << " low, " << high << " high");
  CHECK(low > 200);
  CHECK(high > 200);
  CHECK(*std::min_element(b.counts_gyro.begin(), b.counts_gyro.end()) == -32768);
  CHECK(*std::max_element(b.counts_gyro.begin(), b.counts_gyro.end()) == 32767);
  CHECK(sim.stats().pre_clip_engaged == std::array<int64_t, 6>{});  // clip_stage output
}

TEST_CASE("U8(b): clip_stage both near the rail against a 256-panel reference split at the clip crossings") {
  // omega_z = 5 sin(4 pi t) rad/s crosses the rails +-250 deg/s four times per period. adc_bits 32 with lsb16 / 2^16
  // keeps the 16-bit rail 2^15 lsb16 while quantizing 2^16 times finer, so the comparison sees the quadrature error,
  // which must stay within 0.5 LSB of the 16-bit grid.
  const double lsb16 = kLsbGyro, rail = 32768 * lsb16;
  const double amp = 5.0 / (4 * kPi);
  const Rig rig = sampled_rig([&](double t) { return -amp * std::cos(4 * kPi * t); }, zero, 1.2);
  ImuParams p = ideal(1000, true);
  p.clip_stage = ClipStage::kBoth;
  p.adc_bits = 32;
  p.gyro.lsb = lsb16 / 65536;
  REQUIRE(validate(p).warnings.size() == 1);  // the pre-clip is inert for the accelerometer (no lsb)
  const int64_t t_b = 200'000'000;
  ImuSimulator sim(p, input(rig, t_b, 900'000'000));
  const ImuBlock b = sim.advance_to(INT64_MAX);
  REQUIRE(b.size() == 700);
  const render::TrajectoryView v = rig.view();
  const Mount m = mount_of({{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {0, 0, 0}}, p);
  const double tau = tau_c_s(p.gyro) * 1e9, d = (p.gyro.group_delay_s - tau_c_s(p.gyro)) * 1e9;
  const auto x = [&](double rel) {
    const double n = std::floor(rel);
    return truth_at(v, m, {0, 0, -9.80665}, where_right(v, t_b + static_cast<int64_t>(n), rel - n)).omega_I;
  };
  const auto clip = [&](Vec3d u) {
    u.z = std::clamp(u.z, -rail, rail);
    return u;
  };
  const auto excess = [&](double s) { return std::abs(x(s - d).z) - rail; };
  double prev = -20 * tau;  // t_0 = 0 (start_phase zero); the rig starts long before
  Vec3d y = clip(x(prev - d));
  double worst = 0;
  int crossings = 0;
  std::vector<double> cuts;
  Panels pn;
  for (size_t k = 0; k < b.size(); ++k) {
    const double t = static_cast<double>(k) * 1e6;
    knot_cuts(v, t_b, prev, t, d, cuts);
    double s0 = prev, g0 = excess(prev);
    for (int j = 1; j <= 64; ++j) {  // the clip crossings: scan, then bisect each sign change
      const double s1 = prev + (t - prev) * j / 64, g1 = excess(s1);
      if ((g0 < 0) != (g1 < 0)) {
        double lo = s0, hi = s1;
        for (int it = 0; it < 100 && lo < hi; ++it) {
          const double mid = 0.5 * (lo + hi);
          if (mid <= lo || mid >= hi) break;
          ((excess(mid) < 0) == (g0 < 0) ? lo : hi) = mid;
        }
        cuts.push_back(0.5 * (lo + hi));
        ++crossings;
      }
      s0 = s1;
      g0 = g1;
    }
    std::sort(cuts.begin(), cuts.end());
    Vec3d integral{0, 0, 0};
    double lo = prev;
    for (size_t c = 0; c <= cuts.size(); ++c) {
      const double hi = c < cuts.size() ? cuts[c] : t;
      if (!(hi > lo)) continue;
      const double panels = std::max(1.0, std::ceil(256 * (hi - lo) / (t - prev)));
      panel_nodes(lo, hi, 2 * (hi - lo) / panels, {}, pn);
      for (size_t i = 0; i < pn.s.size(); ++i) {
        const double s = hi + pn.s[i];
        integral += (pn.w[i] * std::exp((s - t) / tau) / tau) * clip(x(s - d));
      }
      lo = hi;
    }
    y = std::exp(-(t - prev) / tau) * y + integral;
    prev = t;
    worst = std::max({worst, std::abs(b.gyro[3 * k + 2] - y.z), std::abs(b.gyro[3 * k]), std::abs(b.gyro[3 * k + 1])});
  }
  MESSAGE("pre-clip: max |y - y_ref| = " << worst / lsb16 << " LSB (16-bit grid) over " << crossings << " crossings");
  CHECK(crossings >= 4);
  CHECK(worst <= 0.5 * lsb16);
  const std::array<int64_t, 6> engaged = sim.stats().pre_clip_engaged;
  MESSAGE("samples with a clipped filter input (gyro z): " << engaged[2]);
  CHECK(engaged[2] > 100);
  CHECK((engaged[0] == 0 && engaged[1] == 0 && engaged[3] == 0 && engaged[4] == 0 && engaged[5] == 0));
  // clip_stage output: the filter sees the unclipped input; nothing is counted, and the output differs at the rails.
  p.clip_stage = ClipStage::kOutput;
  ImuSimulator out(p, input(rig, t_b, 900'000'000));
  const ImuBlock bo = out.advance_to(INT64_MAX);
  CHECK(out.stats().pre_clip_engaged == std::array<int64_t, 6>{});
  CHECK(bo.gyro != b.gyro);
}

TEST_CASE("U10: camera_clock equals SensorModel::readout(); stamps equal sensor::reported_time; the t_offset drop") {
  std::vector<int64_t> phases;
  for (const uint64_t seed : {uint64_t{1}, uint64_t{7}, uint64_t{0x0123456789abcdef}}) {
    for (const uint32_t camera : {0u, 1u}) {
      sensor::SensorParams sp = sensor::preset("prophesee_gen3");
      sp.readout_quantum_s = 1e-6;
      sp.clock_offset_ns = -123'456'789;
      sp.clock_drift_ppm = 15.5;
      const sensor::SensorModel model(sp, 4, 4, seed, {sensor::make_constant_waveform()}, camera, 1);
      const sensor::Readout& r = model.readout();
      const CameraClock c = camera_clock(1e-6, -123'456'789, 15.5, seed, camera);
      CHECK(c.quantum_ns == r.quantum_ns);
      CHECK(c.phase_ns == r.phase_ns);
      CHECK(c.offset_ns == r.clock_offset_ns);
      CHECK(c.drift_ppm == r.clock_drift_ppm);
      phases.push_back(c.phase_ns);
    }
  }
  std::sort(phases.begin(), phases.end());
  CHECK(std::unique(phases.begin(), phases.end()) - phases.begin() >= 4);  // the phase is drawn, not constant
  CHECK(camera_clock(0.0, 5, -2.0, 7, 0) == CameraClock{0, 0, 5, -2.0});  // no quantum: no phase
  CHECK(sensor_clock_of(camera_clock(1e-6, 5, -2.0, 7, 0)) == CameraClock{0, 0, 5, -2.0});

  // Stamps through the counter with the delay imu.stamp_offset_ns, for the jittered instants of a full device.
  const Rig rig = random_rig(5, 200, 0.02, 2.5e-4, 6, 5'000'000);
  ImuParams p = full_device();
  p.stamp_offset_ns = -2500;
  ImuInput in = input(rig, 60'000'000, 400'000'000);
  in.clock = camera_clock(1e-6, 5'000'000'017, 15.5, 7, 1);
  in.camera_index = 1;
  in.t_offset_us = -1'000'000'000'000;
  ImuSimulator all(p, in);
  const ImuBlock b = all.advance_to(INT64_MAX);
  REQUIRE(b.size() > 300);
  const sensor::Readout r{in.clock.quantum_ns, -2500, in.clock.phase_ns, in.clock.offset_ns, in.clock.drift_ppm};
  int64_t wrong = 0;
  for (size_t k = 0; k < b.size(); ++k) {
    wrong += b.t_ns[k] != sensor::reported_time(r, b.t_true_ns[k]);
    wrong += b.t_ns[k] != reported_time(in.clock, -2500, b.t_true_ns[k]);
    if (k > 0) wrong += b.t_true_ns[k] <= b.t_true_ns[k - 1];
  }
  CHECK(wrong == 0);
  CHECK(all.stats().dropped_before_t_offset == 0);
  // The event file's t_offset just above the fourth stamp: the first four samples are dropped and counted; the rest
  // are those of the undropped run, bitwise (the processes and filters step through dropped samples).
  in.t_offset_us = time::floor_div(b.t_ns[3], int64_t{1000}) + 1;
  ImuSimulator cut(p, in);
  const ImuBlock bc = cut.advance_to(INT64_MAX);
  size_t dropped = 0;
  while (b.t_ns[dropped] < 1000 * in.t_offset_us) ++dropped;
  CHECK(dropped == 4);
  CHECK(cut.stats().dropped_before_t_offset == 4);
  CHECK(same(bc, tail(b, dropped)));
  CHECK(bc.t_ns.front() >= 1000 * in.t_offset_us);
  // A t_offset beyond every stamp drops all; one beyond int64 ns too.
  in.t_offset_us = INT64_MAX / 1000 + 1;
  ImuSimulator none(p, in);
  CHECK(none.advance_to(INT64_MAX).size() == 0);
  CHECK(none.stats().dropped_before_t_offset == static_cast<int64_t>(b.size()));
}

TEST_CASE("U11: determinism: chunking, reruns, the sensor seed and imu.seed") {
  const Rig rig = random_rig(5, 200, 0.02, 2.5e-4, 6, 5'000'000);
  const ImuParams p = full_device();
  ImuInput in = input(rig, 60'000'000, 400'000'000);
  in.clock = camera_clock(1e-6, 0, 0.0, 3, 1);
  in.camera_index = 1;
  in.t_offset_us = 0;
  ImuSimulator one(p, in), many(p, in), again(p, in);
  const ImuBlock all = one.advance_to(INT64_MAX);
  REQUIRE(all.size() > 340);
  REQUIRE(all.counts_gyro.size() == 3 * all.size());
  REQUIRE(all.counts_accel.size() == 3 * all.size());
  ImuBlock parts;
  for (int64_t j = 1; j <= 997; ++j) append(parts, many.advance_to(60'000'000 + j * 340'000'000 / 997));
  append(parts, many.advance_to(INT64_MAX));
  CHECK(same(all, parts));
  CHECK(same(all, again.advance_to(INT64_MAX)));
  CHECK(one.stats().warmup_samples == many.stats().warmup_samples);
  // The instants carry the start phase and the wander: off the plain grid t_b + k T.
  int64_t off_grid = 0;
  for (size_t k = 0; k < all.size(); ++k) {
    off_grid += all.t_true_ns[k] != 60'000'000 + std::llround(static_cast<double>(k) * 1e9 / 1026.45);
  }
  CHECK(off_grid > 300);
  // Another sensor seed changes the counter's phase, hence the stamps, and nothing else.
  ImuInput other = in;
  other.clock = camera_clock(1e-6, 0, 0.0, 4, 1);
  REQUIRE(other.clock.phase_ns != in.clock.phase_ns);
  ImuSimulator s2(p, other);
  ImuBlock b2 = s2.advance_to(INT64_MAX);
  CHECK(b2.t_ns != all.t_ns);
  b2.t_ns = all.t_ns;
  CHECK(same(all, b2));
  // Another imu.seed: other noise, biases and instants.
  ImuParams q = p;
  q.seed = 43;
  ImuSimulator s3(q, in);
  const ImuBlock b3 = s3.advance_to(INT64_MAX);
  CHECK(b3.t_true_ns != all.t_true_ns);
  CHECK(b3.bias_gyro != all.bias_gyro);
  // The camera index keys the draws too.
  ImuInput cam = in;
  cam.camera_index = 2;
  ImuSimulator s4(p, cam);
  CHECK(s4.advance_to(INT64_MAX).bias_accel != all.bias_accel);
}

TEST_CASE("clock and stamp refusals") {
  const Rig rig = random_rig(5, 40, 0.02, 2.5e-4, 6, 5'000'000);
  const auto refusal = [&](const ImuParams& p, const ImuInput& in) -> std::string {
    try {
      ImuSimulator sim(p, in);
    } catch (const Refused& e) {
      return e.what();
    }
    return "";
  };
  const ImuParams p = ideal(1000, true);
  ImuInput in = input(rig, 60'000'000, 170'000'000);
  CHECK(refusal(p, in).empty());
  in.clock = {1000, 1000, 0, 0};
  CHECK(refusal(p, in) == "clock.phase_ns must lie in [0, quantum_ns) (0 without a quantum)");
  in.clock = {0, 3, 0, 0};
  CHECK(refusal(p, in) == "clock.phase_ns must lie in [0, quantum_ns) (0 without a quantum)");
  in.clock = {-1, 0, 0, 0};
  CHECK(refusal(p, in) == "clock.quantum_ns must be >= 0");
  in.clock = {1000, 0, 0, 1e4};
  CHECK(refusal(p, in) == "clock.drift_ppm must be finite in (-1e4, 1e4)");
  in.clock = {1000, 0, 0, std::nan("")};
  CHECK(refusal(p, in) == "clock.drift_ppm must be finite in (-1e4, 1e4)");
  in.clock = {1000, 0, INT64_MAX - 100'000'000, 0};
  CHECK(refusal(p, in) == "clock: the stamps of the run span overflow int64");
  in.clock = {1000, 0, INT64_MAX - 200'000'000, 0};
  CHECK(refusal(p, in).empty());
  CHECK_THROWS_WITH_AS(camera_clock(1.0, 0, 0, 1, 0), "sensor.readout_quantum_s = 1: expected [0, 1) s", Refused);
  CHECK_THROWS_WITH_AS(camera_clock(1e-6, 0, -1e4, 1, 0), "sensor.clock_drift_ppm = -10000: expected (-1e4, 1e4)",
                       Refused);
  // purpose.h keeps the camera's low 8 bits: camera 256 would reuse camera 0's draws (SensorModel refuses it too).
  CHECK_THROWS_WITH_AS(camera_clock(1e-6, 0, 0, 7, 256), "camera_index must be < 256 (purpose.h layout)", Refused);
  CHECK(camera_clock(1e-6, 0, 0, 7, 255).quantum_ns == 1000);
  in.clock = {};
  in.camera_index = 256;
  CHECK(refusal(p, in) == "camera_index must be < 256 (purpose.h layout)");
  in.camera_index = 255;
  CHECK(refusal(p, in).empty());
  int64_t out = 0;
  CHECK_FALSE(reported_time_checked({0, 0, 1, 0}, 0, INT64_MAX, &out));
  CHECK_FALSE(reported_time_checked({1000, 0, 0, 0}, 0, INT64_MIN + 1, &out));  // the floor goes below INT64_MIN
  CHECK(reported_time_checked({1000, 7, 0, 0}, 0, INT64_MIN + 1000, &out));
  CHECK(out == sensor::reported_time({1000, 0, 7, 0, 0.0}, INT64_MIN + 1000));
  CHECK_THROWS_AS(reported_time({0, 0, 1, 0}, 0, INT64_MAX), Refused);
  CHECK(reported_time({1000, 250, -5, 200.0}, 300, 12'345'678) ==
        sensor::reported_time({1000, 300, 250, -5, 200.0}, 12'345'678));
}
