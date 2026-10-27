#include "sensor/maps.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/pmath/explog.h"
#include "core/pmath/trig.h"
#include "core/rng/purpose.h"
#include "sensor/noise.h"
#include "sensor/noise_tables.h"

namespace balla::sensor {

namespace {

constexpr double kBoltzmannEv = 8.617333262e-5;  // eV / K
constexpr double kElementaryCharge = 1.602176634e-19;

// Draw blocks of rng::Purpose::kSensorMap (append-only: never renumber).
constexpr uint64_t kMapNormals = 0;   // xi_on, xi_off, xi_leak, xi_dark
constexpr uint64_t kMapUniforms = 1;  // u_phase, u_hot, u_type, u_rate
constexpr uint64_t kMapHotRate = 2;   // xi_rate of the log-normal hot law (D11; drawn for hot pixels only)
constexpr uint64_t kMapHpf = 3;       // xi_hpf of the per-pixel tau_hp (drawn only with tau_hp > 0 and a spread)

void normals4(uint64_t seed, uint32_t camera, uint32_t pixel, uint64_t block, double* out) {
  const rng::U32x4 r = rng::draw(seed, rng::Purpose::kSensorMap, camera, pixel, block);
  const double u0 = open_uniform(r.v[0], 0.0), u1 = open_uniform(r.v[1], 0.0);
  const double u2 = open_uniform(r.v[2], 0.0), u3 = open_uniform(r.v[3], 0.0);
  constexpr double kTwoPi = 0x1.921fb54442d18p+2;
  const double r0 = std::sqrt(-2.0 * pmath::log(u0)), r1 = std::sqrt(-2.0 * pmath::log(u2));
  double s0, c0, s1, c1;
  pmath::sincos(kTwoPi * u1, &s0, &c0);
  pmath::sincos(kTwoPi * u3, &s1, &c1);
  out[0] = r0 * c0;
  out[1] = r0 * s0;
  out[2] = r1 * c1;
  out[3] = r1 * s1;
}

}  // namespace

double arrhenius(double ea_eV, double temperature_K, double reference_K) {
  return pmath::exp(-ea_eV / kBoltzmannEv * (1.0 / temperature_K - 1.0 / reference_K));
}

double hpf_tau(const SensorParams& p, uint64_t seed, uint32_t camera, uint32_t pixel) {
  if (!p.hpf_on()) return 0.0;
  if (p.tau_hp_sigma_ln == 0) return p.tau_hp;
  double xi[4];
  normals4(seed, camera, pixel, kMapHpf, xi);
  return std::max(p.tau_hp * pmath::exp(p.tau_hp_sigma_ln * xi[0]), p.hpf_floor_s());
}

double median_dark_fA(const SensorParams& p) {
  return p.dark_current_fA * arrhenius(p.dark_ea_eV, p.temperature_K, p.reference_temperature_K);
}

double rice_threshold(const SensorParams& p, double rate_hz) {
  NoiseModel m{};
  m.kq = p.k * kElementaryCharge;
  m.a2 = 1.0 / p.tau_o;
  m.a3 = 1.0 / p.tau_sf;
  m.sigma_o = p.sigma_o;
  double P[3][3];
  noise_stationary_cov(m, p.k * p.hot_reference_current_fA * 1e-15, P);
  const double var_v = P[2][2];
  const double var_d = m.a3 * m.a3 * (P[1][1] - 2.0 * P[2][1] + P[2][2]);  // Var(n3'), n3' = a3 (n2 - n3)
  const double two_nu0 = std::sqrt(var_d / var_v) / 0x1.921fb54442d18p+1;  // 2 (1/2pi) sigma_d / sigma_v
  if (!(rate_hz < two_nu0)) return p.theta_floor;
  const double theta = std::sqrt(2.0 * var_v * pmath::log(two_nu0 / rate_hz));
  return theta > p.theta_floor ? theta : p.theta_floor;
}

namespace {

// Entry i of the maps is pixel id_of(i) of the full frame.
template <class IdOf>
SensorMaps draw_maps(const SensorParams& p, uint64_t seed, uint32_t camera, uint32_t n_pixels, IdOf id_of) {
  SensorMaps maps;
  maps.pixels.resize(n_pixels);
  maps.hot_target_hz.assign(n_pixels, 0.0);
  const bool hpf_map = p.hpf_on() && p.tau_hp_sigma_ln > 0;
  if (hpf_map) maps.hpf_b.resize(n_pixels);
  const double leak_temp = arrhenius(p.leak_ea_eV, p.temperature_K, p.reference_temperature_K);
  const double dark_temp = arrhenius(p.dark_ea_eV, p.temperature_K, p.reference_temperature_K);
  const double rho = p.theta_onoff_corr, rho_c = std::sqrt(1.0 - rho * rho);
  const double ln10 = pmath::log(10.0);
  const double ln_rmin = pmath::log(p.hot_rate_min_hz), ln_rmax = pmath::log(p.hot_rate_max_hz);
  for (uint32_t i = 0; i < n_pixels; ++i) {
    const uint32_t px = id_of(i);
    double xi[4];
    normals4(seed, camera, px, kMapNormals, xi);
    const rng::U32x4 u = rng::draw(seed, rng::Purpose::kSensorMap, camera, px, kMapUniforms);
    const double u_phase = open_uniform(u.v[0], 0.0), u_hot = open_uniform(u.v[1], 0.0);
    const double u_type = open_uniform(u.v[2], 0.0), u_rate = open_uniform(u.v[3], 0.0);
    PixelMap& m = maps.pixels[i];
    // Frozen thresholds, clamped at theta_floor (M3 A.19), with their own spreads and the correlation rho
    // (D11): xi_off' = rho xi_on + sqrt(1 - rho^2) xi_off; rho = 0 keeps the independent draws exactly.
    const double xi_off = rho == 0.0 ? xi[1] : rho * xi[0] + rho_c * xi[1];
    m.theta_on = std::max(p.theta_floor, p.theta_on + p.sigma_theta_on * xi[0]);
    m.theta_off = std::max(p.theta_floor, p.theta_off + p.sigma_theta_off * xi_off);
    double leak_rate = p.leak_rate_hz * pmath::exp(p.leak_sigma_log10 * xi[2] * ln10) * leak_temp;
    m.dark_fA = p.dark_current_fA * pmath::exp(p.dark_sigma_ln * xi[3]) * dark_temp;
    m.hot = HotType::kNone;
    if (u_hot < p.hot_fraction) {
      // Hot pixels are tails of the leak and threshold distributions, not a separate generator: target
      // rates log-uniform on [min, max] (M3 §A.5) or log-normal clamped to it (D11, from their own draw
      // block), split into leak-type (share hot_leak_fraction) and threshold-type.
      double rate;
      if (p.hot_rate_law == HotRateLaw::kLogUniform) {
        rate = pmath::exp(ln_rmin + u_rate * (ln_rmax - ln_rmin));
      } else {
        double z[4];
        normals4(seed, camera, px, kMapHotRate, z);
        rate = std::clamp(p.hot_rate_median_hz * pmath::exp(p.hot_rate_sigma_ln * z[0]), p.hot_rate_min_hz,
                          p.hot_rate_max_hz);
      }
      maps.hot_target_hz[i] = rate;
      if (u_type < p.hot_leak_fraction || !p.noise_on()) {
        m.hot = HotType::kLeak;
        // A physical reset adds T_r of blindness to every leak cycle, so the cycle rate
        // 1 / (T_r + 1 / R_L) meets the target R at R_L = R / (1 - R T_r) (validate() keeps
        // R T_r < 1). The ideal reset carries the overshoot across its dead time and fires at R_L.
        leak_rate = p.reset_mode == ResetMode::kPhysical ? rate / (1.0 - rate * p.refractory_s) : rate;
      } else {
        m.hot = HotType::kThreshold;
        m.theta_on = m.theta_off = rice_threshold(p, rate);
      }
    }
    m.leak = m.theta_on * leak_rate;  // M3 A.18: l = theta_on R_L
    m.phi0 = p.initial_phase == InitialPhase::kUniform     ? u_phase * m.theta_on
             : p.initial_phase == InitialPhase::kSymmetric ? -m.theta_off + u_phase * (m.theta_on + m.theta_off)
                                                           : 0.0;
    if (p.hpf_on()) {
      // The high-pass filter (DESIGN §8.4 A.14) caps leak-driven Delta at l tau: from 0 it reaches theta_on after
      // T_c = tau ln(l tau / (l tau - theta_on)), and never if l tau <= theta_on (SENSOR.md "High-pass filter").
      const double tau = hpf_tau(p, seed, camera, px);
      if (hpf_map) maps.hpf_b[i] = static_cast<float>(1.0 / tau);
      if (m.hot == HotType::kLeak) {
        // The leak-type cycle meets the target R: T_c = 1 / R - T_r under the physical reset (blind for T_r, Delta
        // from about 0 at the release), 1 / R under the ideal one (Delta carried through its dead time).
        const double rate = maps.hot_target_hz[i];
        const double t_c = p.reset_mode == ResetMode::kPhysical ? 1.0 / rate - p.refractory_s : 1.0 / rate;
        m.leak = m.theta_on / (-tau * pmath::expm1(-t_c / tau));
      }
      if (p.initial_phase == InitialPhase::kUniform) {
        // The renewal law of the filtered leak: the phase uniform in time over the cycle, Delta = l tau (1 - e^{-s/tau});
        // a pixel whose leak cannot fire sits at its asymptote l tau.
        const double lt = m.leak * tau;
        m.phi0 = lt > m.theta_on ? -lt * pmath::expm1(u_phase * pmath::log1p(-m.theta_on / lt)) : lt;
      }
    }
  }
  return maps;
}

}  // namespace

SensorMaps draw_sensor_maps(const SensorParams& p, uint64_t seed, uint32_t camera, uint32_t n_pixels) {
  return draw_maps(p, seed, camera, n_pixels, [](uint32_t i) { return i; });
}

SensorMaps draw_sensor_maps(const SensorParams& p, uint64_t seed, uint32_t camera, const std::vector<uint32_t>& ids) {
  return draw_maps(p, seed, camera, static_cast<uint32_t>(ids.size()), [&](uint32_t i) { return ids[i]; });
}

double lowest_dark_fA(const SensorParams& p, uint64_t seed, uint32_t camera, uint32_t first, uint32_t last) {
  const double dark_temp = arrhenius(p.dark_ea_eV, p.temperature_K, p.reference_temperature_K);
  double lowest = std::numeric_limits<double>::infinity();
  if (first >= last) return lowest;
  if (p.dark_sigma_ln == 0) return p.dark_current_fA * pmath::exp(0.0) * dark_temp;  // every pixel's
  for (uint32_t px = first; px < last; ++px) {
    double xi[4];
    normals4(seed, camera, px, kMapNormals, xi);
    lowest = std::min(lowest, p.dark_current_fA * pmath::exp(p.dark_sigma_ln * xi[3]) * dark_temp);  // draw_maps's
  }
  return lowest;
}

}  // namespace balla::sensor
