// The IMU's random processes (docs/design/phase3/outputs.md §4.2, §4.4): the sample clock and the additive error
// processes of one sensor. Each is the exact discretization of its continuous process for any step dt, so the unit
// tests drive them alone at coarse steps over hours (U5, U6, U9). Every draw is one sensor::normals3 call keyed by
// Philox (imu.seed, tag, bundle camera, pixel, sample index): a value depends on those alone, never on a sigma, on how
// advance_to is chunked or on the event simulation, and two runs that differ only in a sigma share every z (T6).
// Header-only, host, fp64; the exponentials are pmath's, so results do not depend on libm.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>

#include "core/math/vec.h"
#include "core/pmath/explog.h"
#include "core/rng/purpose.h"
#include "imu/params.h"
#include "sensor/noise.h"

namespace balla::imu {

// The key of one camera's IMU draws: imu.seed and the bundle camera index.
struct Stream {
  uint64_t seed = 1;
  uint32_t camera = 0;
};

// One Philox address short of its index: a purpose tag and a pixel.
struct DrawKey {
  rng::Purpose tag;
  uint32_t pixel;
};

// Three standard normals from one Philox call (Box-Muller with pmath's log and sincos, sensor/noise.h).
inline Vec3d normals(const Stream& st, DrawKey key, uint64_t index) {
  double z[3];
  sensor::normals3<double>(st.seed, key.tag, st.camera, key.pixel, index, z);
  return {z[0], z[1], z[2]};
}

namespace detail {
inline double& at(Vec3d& v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
inline double at(const Vec3d& v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
}  // namespace detail

// Three exactly sampled Gauss-Markov (Ornstein-Uhlenbeck) processes with the stationary standard deviation s and one
// time constant per component, stationary from the start:
//   x_0 = s z_0,   x_k = phi_k x_{k-1} + s sqrt(1 - phi_k^2) z_k,   phi_k = e^{-dt_k / T}
// z_0 from `init` (index 0), z_k from `step` (index k). With s = 0 the process is off: x = 0 and nothing is drawn.
class GaussMarkov3 {
 public:
  GaussMarkov3() = default;
  GaussMarkov3(const Stream& st, double s, const Vec3d& T_s, DrawKey init, DrawKey step)
      : st_(st), s_(s), T_(T_s), init_(init), step_(step) {}

  bool on() const { return s_ != 0; }
  uint64_t index() const { return k_; }  // the next k

  // x_k for the next k (0 first); dt_s = t_k - t_{k-1} > 0 in seconds, unused at k = 0.
  const Vec3d& next(double dt_s) {
    const uint64_t k = k_++;
    if (s_ == 0) return x_;
    if (k == 0) {
      x_ = s_ * normals(st_, init_, 0);
      return x_;
    }
    if (dt_s != dt_) {  // the coefficients of the last step, recomputed when dt changes (bitwise compare)
      dt_ = dt_s;
      for (int i = 0; i < 3; ++i) {
        if (i > 0 && detail::at(T_, i) == detail::at(T_, i - 1)) {
          phi_[i] = phi_[i - 1];
          c_[i] = c_[i - 1];
          continue;
        }
        const double r = dt_s / detail::at(T_, i);
        phi_[i] = pmath::exp(-r);
        c_[i] = s_ * std::sqrt(-pmath::expm1(-2 * r));  // s sqrt(1 - phi^2), accurate for dt << T
      }
    }
    const Vec3d z = normals(st_, step_, k);
    for (int i = 0; i < 3; ++i) detail::at(x_, i) = phi_[i] * detail::at(x_, i) + c_[i] * detail::at(z, i);
    return x_;
  }

 private:
  Stream st_{};
  double s_ = 0;
  Vec3d T_{1, 1, 1};
  DrawKey init_{}, step_{};
  uint64_t k_ = 0;
  Vec3d x_{0, 0, 0};
  double dt_ = -1;
  std::array<double, 3> phi_{}, c_{};
};

// A random walk from 0: b_0 = 0, b_k = b_{k-1} + sigma_b sqrt(dt_k) z_k (z_k from `step`, index k). sigma_b = 0: off.
class RandomWalk3 {
 public:
  RandomWalk3() = default;
  RandomWalk3(const Stream& st, double sigma_b, DrawKey step) : st_(st), sigma_b_(sigma_b), step_(step) {}

  bool on() const { return sigma_b_ != 0; }

  const Vec3d& next(double dt_s) {
    const uint64_t k = k_++;
    if (sigma_b_ == 0 || k == 0) return b_;
    b_ += (sigma_b_ * std::sqrt(dt_s)) * normals(st_, step_, k);
    return b_;
  }

 private:
  Stream st_{};
  double sigma_b_ = 0;
  DrawKey step_{};
  uint64_t k_ = 0;
  Vec3d b_{0, 0, 0};
};

// Kalibr's discrete white noise: n_k = sigma / sqrt(dt_k) z_k (z_k from `key`, index k). sigma = 0: off.
class WhiteNoise3 {
 public:
  WhiteNoise3() = default;
  WhiteNoise3(const Stream& st, double sigma, DrawKey key) : st_(st), sigma_(sigma), key_(key) {}

  bool on() const { return sigma_ != 0; }

  Vec3d next(double dt_s) {
    const uint64_t k = k_++;
    if (sigma_ == 0) return {0, 0, 0};
    return (sigma_ / std::sqrt(dt_s)) * normals(st_, key_, k);
  }

 private:
  Stream st_{};
  double sigma_ = 0;
  DrawKey key_{};
  uint64_t k_ = 0;
};

// The bias instability's Gauss-Markov process (§2.3, §4.4): its analytic Allan deviation peaks at 0.617 q for
// tau = 1.89 T, so the floor 0.664 B (IEEE-952) at tau_B gives q = 0.664 B / 0.617 and T = tau_B / 1.89.
inline constexpr double kGmPeakTauOverT = 1.89;
inline constexpr double kGmPeakAdOverQ = 0.617;
inline constexpr double kAllanFloorOverB = 0.664;

// The additive error processes of sensor s (0 gyroscope, 1 accelerometer) of §4.4, and its turn-on bias:
//   b0 = bias + bias_sd (.) z_init                                      (kImuBiasInit, pixel s, index 0)
//   b_rw: random walk, sigma_b = random_walk                            (kImuBiasWalk, pixel s)
//   b_gm: Gauss-Markov, q = 0.664 B / 0.617, T = tau_B / 1.89           (kImuBiasInit and kImuBiasWalk, pixel 2 + s)
//   n: white, sigma / sqrt(dt)                                          (kImuNoise, pixel s)
//      or shaped (OU, white noise through the DLPF): sd sigma / sqrt(2 tau_c), time constant tau_c
//                                                                       (kImuBiasInit pixel 4 + s; kImuNoise pixel s)
// b0 enters the filter input (§4.3.2); b_rw + b_gm and n are added to the filter output at each sample.
class SensorNoise {
 public:
  struct Sample {
    Vec3d drift{0, 0, 0};  // b_rw + b_gm at the sample
    Vec3d noise{0, 0, 0};  // n_k
  };

  SensorNoise() = default;
  SensorNoise(const Stream& st, int s, const SensorErrors& e, NoiseMode mode) {
    using rng::Purpose;
    const auto pixel = static_cast<uint32_t>(s);
    const Vec3d z = normals(st, {Purpose::kImuBiasInit, pixel}, 0);
    b0_ = {e.bias[0] + e.bias_sd[0] * z.x, e.bias[1] + e.bias_sd[1] * z.y, e.bias[2] + e.bias_sd[2] * z.z};
    walk_ = RandomWalk3(st, e.random_walk, {Purpose::kImuBiasWalk, pixel});
    const double T_gm = e.bias_instability_tau_s / kGmPeakTauOverT;
    gm_ = GaussMarkov3(st, kAllanFloorOverB * e.bias_instability / kGmPeakAdOverQ, {T_gm, T_gm, T_gm},
                       {Purpose::kImuBiasInit, 2 + pixel}, {Purpose::kImuBiasWalk, 2 + pixel});
    if (mode == NoiseMode::kShaped) {  // validate() requires the sensor's first-order filter
      const double tau = tau_c_s(e);
      shaped_ = GaussMarkov3(st, e.noise_density / std::sqrt(2 * tau), {tau, tau, tau},
                             {Purpose::kImuBiasInit, 4 + pixel}, {Purpose::kImuNoise, pixel});
    } else {
      white_ = WhiteNoise3(st, e.noise_density, {Purpose::kImuNoise, pixel});
    }
  }

  const Vec3d& b0() const { return b0_; }
  // Whether any process adds to the output (b0 aside).
  bool on() const { return walk_.on() || gm_.on() || white_.on() || shaped_.on(); }

  // The sample k (the next one; 0 first), dt_s = t_k - t_{k-1} in seconds (dt_0 = T, the nominal period).
  Sample next(double dt_s) {
    Sample out;
    out.drift = walk_.next(dt_s) + gm_.next(dt_s);
    out.noise = shaped_.on() ? shaped_.next(dt_s) : white_.next(dt_s);
    return out;
  }

 private:
  Vec3d b0_{0, 0, 0};
  RandomWalk3 walk_;
  GaussMarkov3 gm_, shaped_;
  WhiteNoise3 white_;
};

// The sample clock (§4.2): instants in ns relative to the run's start t_b, in fp64,
//   tau_k = (k + phi) T - T Y_k,   Y_k = sum_{j=1..k} y_j,   y_j = g_{1,j} + g_{2,j} + g_{3,j}
// T = 1e9 / rate_hz ns; phi = (v0 + 0.5) / 2^32 of Philox(imu.seed, kImuClock, camera, pixel 0, index 0) with
// start_phase random, else 0; g_i Gauss-Markov processes with T_i = 0.053, 0.53 and 5.3 s and s = 1.304 clock_adev,
// stepped at the nominal T (kImuClock pixel 1, index j: one normals3 call gives i = 1..3; stationary start pixel 2,
// index 0). Their sum has a flat ADEV_y of 0.925, 1.000 and 0.912 clock_adev at 0.1, 1 and 10 s (R-DOC B4, C6) and is
// a model constant, not a knob. With clock_adev 0 and phi 0, tau_k = k T exactly, so integral periods give integral
// instants.
inline constexpr std::array<double, 3> kClockTau = {0.053, 0.53, 5.3};
inline constexpr double kClockScale = 1.304;

class SampleClock {
 public:
  SampleClock() = default;
  SampleClock(const Stream& st, double rate_hz, StartPhase start, double clock_adev)
      : period_ns_(1e9 / rate_hz), period_s_(1.0 / rate_hz) {
    if (start == StartPhase::kRandom) {
      phase_ = (static_cast<double>(rng::draw(st.seed, rng::Purpose::kImuClock, st.camera, 0, 0).v[0]) + 0.5) * 0x1p-32;
    }
    g_ = GaussMarkov3(st, kClockScale * clock_adev, {kClockTau[0], kClockTau[1], kClockTau[2]},
                      {rng::Purpose::kImuClock, 2}, {rng::Purpose::kImuClock, 1});
    g_.next(period_s_);  // g_{i,0}, the stationary start
    tau_ns_ = phase_ * period_ns_;
  }

  uint64_t index() const { return k_; }
  double tau_ns() const { return tau_ns_; }   // tau_k of the current k
  double y() const { return y_; }             // y_k (0 at k = 0)
  double Y() const { return Y_; }             // Y_k
  double phase() const { return phase_; }     // phi
  double period_ns() const { return period_ns_; }
  double period_s() const { return period_s_; }

  // k -> k + 1.
  void next() {
    ++k_;
    if (g_.on()) {
      const Vec3d& g = g_.next(period_s_);
      y_ = (g.x + g.y) + g.z;
      Y_ += y_;
    }
    tau_ns_ = (static_cast<double>(k_) + phase_) * period_ns_ - period_ns_ * Y_;
  }

 private:
  double period_ns_ = 1e6, period_s_ = 1e-3, phase_ = 0;
  GaussMarkov3 g_;
  uint64_t k_ = 0;
  double y_ = 0, Y_ = 0, tau_ns_ = 0;
};

}  // namespace balla::imu
