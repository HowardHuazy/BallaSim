// Sensor parameters and presets (DESIGN §8.9, M3 Table A.1).
//
// SensorParams holds the physical prior of one sensor family in SI units plus the numerical
// tolerances; preset() builds one from the family priors of param_fields.cpp, the parameter rows
// (docs/dev/P0_PARAMS.md §1.3; their descriptors are listed in docs/dev/PARAMS_REFERENCE.md). Kernel
// units (fA, s, log-e) are derived in sensor_model.cpp. validate() enforces every configuration rule
// the maths relies on (the rows' hard ranges, then the cross-key rules) and throws
// std::invalid_argument naming the offending field.
#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "sensor/waveform.h"

namespace balla::sensor {

enum class ResetMode : int32_t {
  kPhysical = 0,  // blind on (t_c, t_c + T_r], reference set to x3 at release (M3 A.16)
  kIdeal = 1,     // ESIM-compatible, explicitly non-physical: r += p theta (M3 A.17)
};

enum class InitialPhase : int32_t {
  kUniform = 0,    // phi0 ~ U[0, theta_on): renewal law of a leak-driven reference (M3 A.20)
  kSymmetric = 1,  // phi0 ~ U(-theta_off, theta_on) (DESIGN §8.4 option)
  kZero = 2,       // phi0 = 0: the reference at the initial state exactly (ESIM-style starts; P0d open item, I6')
};

// The comparator's state at init (accuracy.initial_state; D12, ENGINE_CPP I6').
enum class InitialState : int32_t {
  kPhase = 0,       // the reference phase of initial_phase, no history (the Phase-1 start)
  kStationary = 1,  // the stationary state of the noisy, leaky comparator with reset, from the probe (SENSOR.md)
};

// The input node's diffusion per noise cell (accuracy.noise_diffusion; PHASE2 D-P2-2, F070).
enum class NoiseDiffusion : int32_t {
  kFrozen = 0,     // D1 = 2 k q a1, the stationary value (M3 D.1): Var(n1) stays k q through transients
  kTransient = 1,  // D1 = rho 2 k q a1 with n1's node law exact: k q (a_I + a1) as a1 h -> 0 (pixel.h, SENSOR.md)
};

enum class Photoreceptor : int32_t {
  kCascade = 0,        // the three-stage photoreceptor cascade of M3 (DESIGN §8.2), with its in-path noise
  kInstantaneous = 1,  // ln I(t) itself: the textbook DVS (D3, PARAMETERS §5.9); no cascade, no noise
};

enum class HotRateLaw : int32_t {
  kLogUniform = 0,  // target rates log-uniform on [hot_rate_min_hz, hot_rate_max_hz] (M3 §A.5)
  kLogNormal = 1,   // median hot_rate_median_hz, ln-spread hot_rate_sigma_ln, clamped to [min, max] (D11)
};

// The driver's event-rate limiter on the sorted output stream (sensor.limiter; DESIGN D13-5, F043; sensor/limiter.h).
enum class LimiterMode : int32_t {
  kOff = 0,    // every event is output
  kEpoch = 1,  // a token bucket whose rate falls linearly over a free-running epoch (docs/dev/EVIMO2_LIMITER.md §5)
};

// No prior is built in (D2): the family-default fields and the required illuminance_scale start as NaN and family as
// "", which validate() refuses; preset() and a resolved configuration (docs/dev/P0_PARAMS.md §2) fill every field.
// Bool, enum and integer fields keep ordinary initializers, and the literal rows (numerics, labels) their defaults.
inline constexpr double kUnset = std::numeric_limits<double>::quiet_NaN();

struct SensorParams {
  std::string family;

  // Photoreceptor (M3 A.3-A.8): the cascade, or the instantaneous log response (D3), for which the cascade fields
  // and the noise switch are inert.
  Photoreceptor photoreceptor = Photoreceptor::kCascade;
  double k = kUnset;         // photoreceptor constant, A^-1 s^-1
  double tau_o = kUnset;     // output-node pole, s
  double tau_sf = kUnset;    // source-follower pole, s
  double sigma_o = kUnset;   // unfiltered output-node noise, log-e
  bool noise = true;         // in-path noise on the 2^16 ns lattice (M3 §D)

  // Input (M3 A.1-A.2).
  double illuminance_scale = kUnset;   // photocurrent per unit of renderer irradiance, fA (required)
  double veiling_glare = 0.0;          // veiling-glare fraction g (D13 item 3; SensorModel::glare_frame_mean)
  double dark_current_fA = kUnset;     // median dark current at reference_temperature_K
  double dark_sigma_ln = kUnset;       // lognormal spread across pixels
  double dark_ea_eV = kUnset;          // Arrhenius activation energy
  double temperature_K = kUnset;
  double reference_temperature_K = kUnset;  // T0 at which dark_current_fA and leak_rate_hz are quoted

  // Comparator (M3 A.14-A.19).
  double theta_on = kUnset, theta_off = kUnset;  // mean thresholds, log-e
  double sigma_theta_on = kUnset;                // frozen per-pixel mismatch of theta_on
  double sigma_theta_off = kUnset;               // and of theta_off (D11)
  double theta_onoff_corr = kUnset;              // correlation of the two mismatches, in [-1, 1] (D11)
  double theta_floor = kUnset;                   // clamp of the drawn thresholds
  ResetMode reset_mode = ResetMode::kPhysical;
  double refractory_s = kUnset;                  // physical refractory T_r (ideal mode: dead time)
  double eps_ci = kUnset;                        // charge-injection offset at release, log-e
  InitialPhase initial_phase = InitialPhase::kUniform;

  // Leak (M3 A.18): ON drift theta_on * R_L of the reference, log10 R_L ~ N(log10 R, sigma^2).
  double leak_rate_hz = kUnset;
  double leak_sigma_log10 = kUnset;
  double leak_ea_eV = kUnset;

  // Prophesee high-pass filter (bias_hpf; DESIGN §8.4 A.14, SENSOR.md "High-pass filter"): the reference relaxes
  // toward x3, dr/dt = -l + (x3 - r) / tau_hp. 0 is off (no published bias-to-tau map: every prior); on, validate()
  // keeps tau_hp >= hpf_floor_s(). tau_hp_sigma_ln: the per-pixel log-normal spread of tau_hp (inert at 0).
  double tau_hp = kUnset;
  double tau_hp_sigma_ln = 0.0;

  // Hot pixels as distribution tails (DESIGN §8.4, M3 §A.5).
  double hot_fraction = kUnset;
  HotRateLaw hot_rate_law = HotRateLaw::kLogUniform;
  double hot_rate_min_hz = kUnset, hot_rate_max_hz = kUnset;
  double hot_rate_median_hz = kUnset;        // log-normal law only
  double hot_rate_sigma_ln = kUnset;         // log-normal law only
  double hot_leak_fraction = kUnset;         // share of leak-type hot pixels (the rest threshold-type, with noise)
  double hot_reference_current_fA = kUnset;  // photocurrent at which threshold-type rates are set

  // Readout (M3 F.1-F.2) and sensor clock.
  double readout_quantum_s = kUnset;  // 0 = no quantization
  double readout_delay_s = kUnset;    // d0
  int64_t clock_offset_ns = 0;        // t_sensor = t_true + offset + drift (t_true)
  double clock_drift_ppm = kUnset;

  // Rate limiter (DESIGN D13-5, F043; docs/dev/EVIMO2_LIMITER.md §5; sensor/limiter.h), after the pixel loop on the
  // time-sorted output stream and the sensor clock: a token bucket of depth limiter_depth events whose rate falls
  // linearly from limiter_rate0_hz to limiter_rate1_hz over each epoch of limiter_period_s, the epochs reset at
  // limiter_phase_s + k P (null: drawn per camera from tag 0x0023). Off in every family prior; the other rows are
  // inert while off (and enter neither validate()'s cross-key rules nor the fingerprint).
  LimiterMode limiter = LimiterMode::kOff;
  double limiter_rate0_hz = kUnset;          // cap at the epoch's start, events/s
  double limiter_rate1_hz = kUnset;          // cap at its end (<= rate0)
  double limiter_period_s = kUnset;          // epoch P
  int64_t limiter_depth = 32;                // bucket depth B, events
  std::optional<double> limiter_phase_s;     // a reset time mod P on the sensor clock; null: drawn

  // Numerics (DESIGN §8.3, M3 §D.4).
  double eps_L = 0.002;           // total log-domain tolerance; eps_h = eps_L/8, eps_PL = eps_L/(8(1+V))
  int32_t noise_leaf_level = 5;   // tree leaves of 2^(16-L) ns
  double noise_margin_z = 6.0;    // exclusion margin z_m (M3 D.8)
  // The input node's diffusion in transient cells (PHASE2 D-P2-2): frozen keeps the low-level API and phase1 / v0 as
  // they were; the derived accuracy profiles choose transient (resolve.cpp).
  NoiseDiffusion noise_diffusion = NoiseDiffusion::kFrozen;

  // Initial state (D12, ENGINE_CPP I6'): the run keys accuracy.initial_state, .stationary_burnin_ns and
  // .stationary_probe_pixels applied to every camera. stationary: a probe of stationary_probe_pixels pixels runs its
  // own history over stationary_burnin_ns (0: 1.5 / R_L, clamped to [60, 600] s) before init; every other pixel takes
  // a snapshot of its nearest probe's comparator (SensorModel::stationary_init, SENSOR.md "Initial state").
  InitialState initial_state = InitialState::kPhase;
  int64_t stationary_burnin_ns = 0;
  int32_t stationary_probe_pixels = 32;

  // Labels (D14a): SIGNAL or LEAK when that increment's share of Delta reaches the cut (DESIGN §9.2).
  double label_share_cut = 0.5;

  void validate() const;
  // In-path noise is simulated: the noise switch is on and the photoreceptor is the cascade (whose noise it is).
  bool noise_on() const { return noise && photoreceptor == Photoreceptor::kCascade; }
  // The sensor has hot pixels; without them the hot-pixel rows (hot_pixel_key) are inert.
  bool hot_pixels() const { return hot_fraction != 0; }
  // The high-pass filter is on (tau_hp > 0); off, tau_hp_sigma_ln is inert and the kernels are those without it.
  bool hpf_on() const { return tau_hp > 0; }
  // The smallest tau_hp (s) validate() accepts when on, max(1 ms, 10 max(tau_o, tau_sf)): the filter's rate b = 1/tau_hp
  // stays at most a tenth of either cascade pole (b / (a - b) <= 1/9, never a = b) and b 2^16 ns <= 0.066 (the
  // comparator's truncation bound, SENSOR.md "High-pass filter"). Per-pixel draws are clamped to it.
  double hpf_floor_s() const;
  // The rate limiter is on (sensor.limiter epoch); off, its other rows are inert.
  bool limiter_on() const { return limiter != LimiterMode::kOff; }
  // The smallest depth validate() accepts while the limiter is on: ceil(rate0 q_bin) + 1 events with q_bin =
  // max(readout_quantum_s, 1 us), the limiter's bin. A shallower bucket clips the fractional refill of every saturated
  // bin, so the output falls below the cap (a bucket simulation, Poisson 10 Mev/s offered to the 6.00 -> 5.64 Mev/s
  // cap in 1 us bins: B 6 gives 5.44 Mev/s, B >= 16 the mean cap).
  int64_t limiter_min_depth() const;
};

// Prophesee Gen3 (EVIMO2 left/right), Samsung Gen3 (asynchronous Phase 1 approximation) and ideal, from the
// family priors; illuminance_scale 1.0 (the low-level API's neutral scale; runs set their own) and clock_offset_ns 0.
SensorParams preset(const std::string& name);
std::vector<std::string> preset_names();

// Waveform from the C1 fields; throws std::invalid_argument on invalid parameters.
Waveform make_constant_waveform();
Waveform make_fourier_waveform(double f0_hz, const std::vector<double>& a, const std::vector<double>& phi);
Waveform make_pulse_waveform(double f_hz, double duty, double rise_s, double phase_s, double floor_level);
// The factories' rules on a Waveform of any origin (a known type, n_harmonics <= kMaxHarmonics, sum |a_k| <= 1, a pulse
// that fits its period), with the pulse's period and phase within 2^61 ns (int64 corners); std::invalid_argument.
void validate_waveform(const Waveform& w);

// 64-bit fingerprint of the parameters (every row, in row order) and waveforms, stored with saved states. The
// high-pass filter's rows enter it only while the filter is on, and the rate limiter's only while it is on, so that a
// sensor without them keeps the fingerprint it had before them (its saved states of version 5 load as HPF-off).
uint64_t fingerprint(const SensorParams& p, const std::vector<Waveform>& groups);

}  // namespace balla::sensor
