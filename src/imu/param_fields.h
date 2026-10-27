// The IMU's parameter rows (docs/design/phase3/outputs.md §3.1; docs/dev/IMU.md): the 37 `imu.*` keys as rows without a
// target (config::row), appended to the schema by schema.cpp. Section `imu` is camera scope (param.h scope_of): one IMU
// per camera, keys cameras.<camera>.imu.*. Every row is stage `output`, so no IMU key invalidates a render or a sensor
// result, and none enters a stage digest while imu.enabled is false (resolve.cpp digests, D-P3-11).
//
// The rows are target-less because member<> cannot address the nested gyro and accel members or give them different
// defaults, and because target-less rows keep the schema audit's ELSEWHERE rule (test_stage_audit.py) and
// test_params.cpp's per-target checks as they are. The keys, types and hard ranges are params.h's kKeys (the single
// source, which from_values/to_values read too); the default of each row is ImuParams{}'s, through default_values().
// Here: the unit, the status and source, the help, the log scale, the soft range and the relevance of each key.
#pragma once

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "core/config/param.h"
#include "imu/params.h"

namespace balla::imu {

namespace detail {

struct RowMeta {
  std::string_view unit;  // numbers: the unit ("1": dimensionless or a count); empty for bool and enum rows
  config::Status status = config::Status::kChoice;  // of the default
  std::string_view source, help;                    // help: at most 120 characters (test_params.cpp)
  bool log = false;                                 // Scale::kLog: zero means off, as sensor.leak_rate_hz
  config::Range soft{};
};

inline constexpr std::string_view kDesign = "docs/design/phase3/outputs.md §3.1";
inline constexpr std::string_view kMpuDlpf = "RM-MPU-9250A-00 r1.6 §4.5 (gyro) / §4.8 (accel), DLPF_CFG 2";

// The 11 switches, the sampling and the clock (kKeys[0..10]).
inline constexpr std::array<RowMeta, 11> kSwitchMeta{{
    {.unit = "", .source = kDesign, .help = "Simulate this camera's IMU after the run (contract C5, imu/<camera>.h5)"},
    {.unit = "Hz", .source = "outputs.md §4.2; EVIMO2's MPU-9250 samples at 1010-1026 Hz (R-IMU)",
     .help = "Output rate of the IMU on the true clock", .log = true, .soft = config::closed(50, 8000)},
    {.unit = "", .source = "outputs.md §4.2",
     .help = "First sample at t_s + U T (a Philox draw) or exactly at t_s"},
    {.unit = "1", .source = "outputs.md §4.2; EVIMO2's IMU clock: 4.5e-5 (R-IMU)",
     .help = "Flat Allan deviation of the sample clock's frequency (flicker FM); 0: an ideal clock", .log = true},
    {.unit = "m", .source = "outputs.md §3.1, §4.1; Kalibr's T_cam_imu",
     .help = "Row-major 4x4 T_cam_imu (Kalibr: X_cam = T X_imu); the IMU frame is the gyro frame"},
    {.unit = "m/s^2", .source = "outputs.md §4.1",
     .help = "World gravity (x, y, z); null: the bundle's gravity_world_mps2, else (0, 0, -9.80665)"},
    {.unit = "ns", .source = "outputs.md §4.6",
     .help = "Latency from the sample instant to the camera counter's latch (kappa = L_m - tau_g - this)"},
    {.unit = "", .source = "outputs.md §4.4",
     .help = "white: sigma/sqrt(dt) per sample (Kalibr); shaped: white noise through the DLPF (needs both filters)"},
    {.unit = "", .source = "outputs.md §4.3.2, §4.5; Q6",
     .help = "output: clamp at the ADC only; both: also clip the filter input at the rail (needs an lsb)"},
    {.unit = "1", .source = "outputs.md §4.5; MPU-9250: 16 bits",
     .help = "ADC width: counts clamp to -2^(bits-1) .. 2^(bits-1) - 1 (with an lsb)"},
    {.unit = "1", .source = "outputs.md §4.4",
     .help = "Seed of every IMU draw of this camera (noise, biases, clock wander)"},
}};

// The 13 keys of one sensor, gyroscope first. kGyroMeta and kAccelMeta differ in the units, the DLPF prior and the help
// that names the axis kind.
inline constexpr std::array<RowMeta, 13> kGyroMeta{{
    {.unit = "rad/s/sqrt(Hz)", .source = "outputs.md §4.4; Kalibr's gyroscope_noise_density",
     .help = "Kalibr white-noise density sigma: the Allan deviation at 1 s; 0: none", .log = true},
    {.unit = "rad/s^2/sqrt(Hz)", .source = "outputs.md §4.4; Kalibr's gyroscope_random_walk",
     .help = "Kalibr bias random-walk sigma_b: the Allan deviation at 3 s; 0: none", .log = true},
    {.unit = "rad/s", .source = "outputs.md §4.4 (Gauss-Markov bias)",
     .help = "Bias instability B: the Allan floor is 0.664 B; 0: none", .log = true},
    {.unit = "s", .source = "outputs.md §4.4",
     .help = "Correlation time tau at the Allan floor of the bias instability (T = tau/1.89)", .log = true},
    {.unit = "rad/s", .source = "outputs.md §4.4", .help = "Turn-on bias b0 per axis"},
    {.unit = "rad/s", .source = "outputs.md §4.4", .help = "Per-run random turn-on bias spread per axis (Philox)"},
    {.unit = "1", .source = "outputs.md §4.4",
     .help = "Scale error per axis in IMU axes, as a fraction (0.01: 1 %)"},
    {.unit = "1", .source = "outputs.md §4.4",
     .help = "Non-orthogonality: the symmetric off-diagonals (xy, xz, yz), in IMU axes"},
    {.unit = "deg", .source = "outputs.md §4.4", .help = "Rotation vector of this sensor's axes against the IMU frame"},
    {.unit = "", .source = "outputs.md §4.3", .help = "Gyroscope DLPF: none (ideal point samples) or first_order"},
    {.unit = "Hz", .status = config::Status::kPrior, .source = kMpuDlpf,
     .help = "-3 dB point f_c of the first-order DLPF (MPU-9250 DLPF_CFG 2)", .log = true},
    {.unit = "s", .status = config::Status::kPrior, .source = kMpuDlpf,
     .help = "Total delay tau_table of the DLPF at DC; the pure delay is tau_table - 1/(2 pi f_c)"},
    {.unit = "rad/s", .source = "outputs.md §4.5",
     .help = "Quantum of one count; null: no quantization and no saturation", .log = true},
}};

inline constexpr std::array<RowMeta, 13> kAccelMeta{{
    {.unit = "m/s^2/sqrt(Hz)", .source = "outputs.md §4.4; Kalibr's accelerometer_noise_density",
     .help = "Kalibr white-noise density sigma: the Allan deviation at 1 s; 0: none", .log = true},
    {.unit = "m/s^3/sqrt(Hz)", .source = "outputs.md §4.4; Kalibr's accelerometer_random_walk",
     .help = "Kalibr bias random-walk sigma_b: the Allan deviation at 3 s; 0: none", .log = true},
    {.unit = "m/s^2", .source = "outputs.md §4.4 (Gauss-Markov bias)",
     .help = "Bias instability B: the Allan floor is 0.664 B; 0: none", .log = true},
    {.unit = "s", .source = "outputs.md §4.4",
     .help = "Correlation time tau at the Allan floor of the bias instability (T = tau/1.89)", .log = true},
    {.unit = "m/s^2", .source = "outputs.md §4.4", .help = "Turn-on bias b0 per axis"},
    {.unit = "m/s^2", .source = "outputs.md §4.4", .help = "Per-run random turn-on bias spread per axis (Philox)"},
    {.unit = "1", .source = "outputs.md §4.4",
     .help = "Scale error per axis in IMU axes, as a fraction (0.01: 1 %)"},
    {.unit = "1", .source = "outputs.md §4.4",
     .help = "Non-orthogonality: the symmetric off-diagonals (xy, xz, yz), in IMU axes"},
    {.unit = "deg", .source = "outputs.md §4.4",
     .help = "Rotation vector of the accelerometer's axes against the IMU (gyro) frame"},
    {.unit = "", .source = "outputs.md §4.3", .help = "Accelerometer DLPF: none (ideal point samples) or first_order"},
    {.unit = "Hz", .status = config::Status::kPrior, .source = kMpuDlpf,
     .help = "-3 dB point f_c of the first-order DLPF (MPU-9250 DLPF_CFG 2)", .log = true},
    {.unit = "s", .status = config::Status::kPrior, .source = kMpuDlpf,
     .help = "Total delay tau_table of the DLPF at DC; the pure delay is tau_table - 1/(2 pi f_c)"},
    {.unit = "m/s^2", .source = "outputs.md §4.5",
     .help = "Quantum of one count; null: no quantization and no saturation", .log = true},
}};

inline const RowMeta& meta_of(size_t key) {
  return key < 11 ? kSwitchMeta[key] : key < 24 ? kGyroMeta[key - 11] : kAccelMeta[key - 24];
}

// The 37 rows in kKeys order: stage output; class T-sensor (imu.seed: T-run); the default of ImuParams{}; relevant only
// while imu.enabled holds (imu.enabled itself always), and the DLPF's two keys only while that sensor's filter is
// first_order.
inline std::vector<config::Param> make_rows() {
  using config::Opts;
  const Values defaults = default_values();
  std::vector<config::Param> rows;
  rows.reserve(kKeys.size());
  for (size_t i = 0; i < kKeys.size(); ++i) {
    const KeySpec& spec = kKeys[i];
    const RowMeta& m = meta_of(i);
    const config::Param& shape = key_rows()[i];  // the full key and what check_value reads
    Opts o;
    o.type = spec.type;
    o.nullable = spec.nullable;
    o.length = spec.length;
    o.unit = m.unit;
    o.value = defaults.at(shape.key);
    o.choices = shape.choices;
    o.hard = spec.hard;
    o.soft = m.soft;
    o.scale = m.log ? config::Scale::kLog : config::Scale::kLinear;
    o.status = m.status;
    o.cls = shape.key == "imu.seed" ? config::Class::kTRun : config::Class::kTSensor;
    o.stage = config::Stage::kOutput;
    if (i != 0) o.when.push_back({"imu.enabled", true});
    if (i >= 11) {
      const size_t k = (i - 11) % 13;
      if (k == 10 || k == 11) o.when.push_back({i < 24 ? "imu.gyro.filter" : "imu.accel.filter", std::string("first_order")});
    }
    o.source = m.source;
    o.help = m.help;
    rows.push_back(config::row(shape.key, o));
  }
  return rows;
}

}  // namespace detail

// The schema rows of the IMU, in kKeys order.
inline const std::vector<config::Param>& param_rows() {
  static const std::vector<config::Param> rows = detail::make_rows();
  return rows;
}

}  // namespace balla::imu
