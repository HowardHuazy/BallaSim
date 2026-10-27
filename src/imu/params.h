// The IMU's parameters (docs/design/phase3/outputs.md §3.1, §3.3, §6; docs/dev/IMU.md). Header-only, host, fp64.
//
// ImuParams is the C++ form of one camera's `imu.*` keys. kKeys lists those keys with their types and hard ranges,
// the single source for from_values/to_values here and for the target-less schema rows of increment A3
// (param_fields.h adds their help, status and stage). validate() holds the hard ranges and the cross-key rules of
// §6, which per-element ranges cannot express.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "core/config/param.h"
#include "core/math/vec.h"

namespace balla::imu {

// An IMU input or option refused, named in the message. ballasim._imu translates it to
// ballasim.export.errors.ExportError (a ValueError).
struct Refused : std::runtime_error {
  using std::runtime_error::runtime_error;
};

using Vec3d = Vec3<double>;
using Mat3d = Mat3<double>;

inline constexpr double kTwoPi = 0x1.921fb54442d18p+2;  // fp64(2 pi)

enum class Filter : uint8_t { kNone, kFirstOrder };
enum class NoiseMode : uint8_t { kWhite, kShaped };
enum class ClipStage : uint8_t { kOutput, kBoth };
enum class StartPhase : uint8_t { kRandom, kZero };

// One sensor (the gyroscope or the accelerometer). Units: rad/s or m/s^2 and their densities (§3.1).
struct SensorErrors {
  double noise_density = 0, random_walk = 0, bias_instability = 0, bias_instability_tau_s = 10;
  std::array<double, 3> bias{}, bias_sd{}, scale{}, nonorth{}, misalign_deg{};
  Filter filter = Filter::kNone;
  double cutoff_hz = 0, group_delay_s = 0;  // per-sensor defaults set in ImuParams below
  std::optional<double> lsb;
  friend bool operator==(const SensorErrors&, const SensorErrors&) = default;
};

struct ImuParams {
  bool enabled = false;
  double rate_hz = 1000;
  StartPhase start_phase = StartPhase::kRandom;
  double clock_adev = 0;
  std::array<double, 16> T_cam_imu = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};  // row-major
  std::optional<std::array<double, 3>> gravity_mps2;  // null: the bundle meta, else (0, 0, -9.80665) (§4.1)
  int64_t stamp_offset_ns = 0;
  NoiseMode noise_mode = NoiseMode::kWhite;
  ClipStage clip_stage = ClipStage::kOutput;
  int32_t adc_bits = 16;
  uint64_t seed = 1;
  // MPU-9250 DLPF_CFG 2 (RM-MPU-9250A-00 r1.6 §4.5, §4.8); inert while the sensor's filter is none.
  SensorErrors gyro{.cutoff_hz = 92, .group_delay_s = 3.9e-3, .lsb = std::nullopt},
      accel{.cutoff_hz = 99, .group_delay_s = 2.88e-3, .lsb = std::nullopt};
  friend bool operator==(const ImuParams&, const ImuParams&) = default;
};

// One camera's imu.* keys ("imu.rate_hz", without "cameras.<c>.").
using Values = std::map<std::string, config::Value, std::less<>>;

// A key: its type and hard range (lists: per element); enums list their choices in the enum's order.
struct KeySpec {
  std::string_view key;
  config::Type type;
  int32_t length = 0;  // lists: the fixed length
  bool nullable = false;
  config::Range hard{};
  std::array<std::string_view, 2> choices{};
};

// The 37 keys (§3.1): 11 switches, then the 13 per-sensor keys of the gyroscope and of the accelerometer.
inline constexpr std::array<KeySpec, 37> kKeys = [] {
  using config::Type;
  constexpr double inf = config::kInf;
  std::array<KeySpec, 37> k{{
      {"imu.enabled", Type::kBool},
      {"imu.rate_hz", Type::kFloat, 0, false, {0, 1e5, true, false}},
      {"imu.start_phase", Type::kEnum, 0, false, {}, {"random", "zero"}},
      {"imu.clock_adev", Type::kFloat, 0, false, {0, 1e-2, false, true}},
      {"imu.T_cam_imu", Type::kFloatList, 16},
      {"imu.gravity_mps2", Type::kFloatList, 3, true},
      {"imu.stamp_offset_ns", Type::kInt, 0, false, {-1e7, 1e7, true, true}},
      {"imu.noise_mode", Type::kEnum, 0, false, {}, {"white", "shaped"}},
      {"imu.clip_stage", Type::kEnum, 0, false, {}, {"output", "both"}},
      {"imu.adc_bits", Type::kInt, 0, false, {8, 32, false, false}},
      {"imu.seed", Type::kInt, 0, false, {0, inf, false, false}},
  }};
  const std::array<KeySpec, 13> per_sensor{{
      {"noise_density", Type::kFloat, 0, false, {0, inf, false, false}},
      {"random_walk", Type::kFloat, 0, false, {0, inf, false, false}},
      {"bias_instability", Type::kFloat, 0, false, {0, inf, false, false}},
      {"bias_instability_tau_s", Type::kFloat, 0, false, {0, 1e5, true, false}},
      {"bias", Type::kFloatList, 3},
      {"bias_sd", Type::kFloatList, 3, false, {0, inf, false, false}},
      {"scale", Type::kFloatList, 3, false, {-0.5, 0.5, true, true}},
      {"nonorth", Type::kFloatList, 3, false, {-0.1, 0.1, true, true}},
      {"misalign_deg", Type::kFloatList, 3, false, {-10, 10, true, true}},
      {"filter", Type::kEnum, 0, false, {}, {"none", "first_order"}},
      {"cutoff_hz", Type::kFloat, 0, false, {0, inf, true, false}},
      {"group_delay_s", Type::kFloat, 0, false, {0, inf, false, false}},
      {"lsb", Type::kFloat, 0, true, {0, inf, true, false}},
  }};
  // The per-sensor rows hold the bare names; key_rows() prefixes "imu.gyro." and "imu.accel.".
  for (size_t i = 0; i < 13; ++i) k[11 + i] = per_sensor[i], k[24 + i] = per_sensor[i];
  return k;
}();

// kKeys[i] as a config::Param carrying what config::check_value reads (the full key, type, length, nullability,
// choices, hard range); A3's rows take the same values.
inline const std::vector<config::Param>& key_rows() {
  static const std::vector<config::Param> rows = [] {
    std::vector<config::Param> out(kKeys.size());
    for (size_t i = 0; i < kKeys.size(); ++i) {
      const KeySpec& s = kKeys[i];
      config::Param& p = out[i];
      p.key = (i < 11 ? "" : i < 24 ? "imu.gyro." : "imu.accel.") + std::string(s.key);
      p.type = s.type;
      p.length = s.length;
      p.nullable = s.nullable;
      p.hard = s.hard;
      if (s.type == config::Type::kEnum) p.choices = {std::string(s.choices[0]), std::string(s.choices[1])};
    }
    return out;
  }();
  return rows;
}

// The full key of kKeys[i] ("imu.gyro.cutoff_hz").
inline const std::string& key_name(size_t i) { return key_rows()[i].key; }

namespace detail {

template <class E>
std::string enum_name(size_t key, E e) {
  return std::string(kKeys[key].choices[static_cast<size_t>(e)]);
}

template <class E>
E enum_value(size_t key, const std::string& s) {
  return static_cast<E>(s == kKeys[key].choices[0] ? 0 : 1);  // check_value has checked the choice
}

inline double as_double(const config::Value& v) {
  if (const auto* i = std::get_if<int64_t>(&v)) return static_cast<double>(*i);
  return std::get<double>(v);
}

template <size_t N>
std::array<double, N> as_array(const config::Value& v) {
  std::array<double, N> out{};
  if (const auto* l = std::get_if<std::vector<int64_t>>(&v)) {
    for (size_t i = 0; i < N; ++i) out[i] = static_cast<double>((*l)[i]);
  } else {
    const auto& d = std::get<std::vector<double>>(v);
    for (size_t i = 0; i < N; ++i) out[i] = d[i];
  }
  return out;
}

// The keys in kKeys order as (ImuParams -> Value) and (Value -> ImuParams) visitors over one field each: the two
// directions are written once, side by side, so a key cannot map to different fields in from_values and to_values.
template <class F>
void each_field(F&& f, ImuParams& p) {
  f(0, p.enabled);
  f(1, p.rate_hz);
  f(2, p.start_phase);
  f(3, p.clock_adev);
  f(4, p.T_cam_imu);
  f(5, p.gravity_mps2);
  f(6, p.stamp_offset_ns);
  f(7, p.noise_mode);
  f(8, p.clip_stage);
  f(9, p.adc_bits);
  f(10, p.seed);
  for (size_t s = 0; s < 2; ++s) {
    SensorErrors& e = s == 0 ? p.gyro : p.accel;
    const size_t b = 11 + 13 * s;
    f(b + 0, e.noise_density);
    f(b + 1, e.random_walk);
    f(b + 2, e.bias_instability);
    f(b + 3, e.bias_instability_tau_s);
    f(b + 4, e.bias);
    f(b + 5, e.bias_sd);
    f(b + 6, e.scale);
    f(b + 7, e.nonorth);
    f(b + 8, e.misalign_deg);
    f(b + 9, e.filter);
    f(b + 10, e.cutoff_hz);
    f(b + 11, e.group_delay_s);
    f(b + 12, e.lsb);
  }
}

template <class T>
config::Value to_value(size_t key, const T& x) {
  if constexpr (std::is_same_v<T, bool> || std::is_floating_point_v<T>) {
    return x;
  } else if constexpr (std::is_enum_v<T>) {
    return enum_name(key, x);
  } else if constexpr (std::is_integral_v<T>) {
    return static_cast<int64_t>(x);
  } else if constexpr (config::detail::IsOptional<T>::value) {
    return x ? to_value(key, *x) : config::Value{};
  } else {
    return std::vector<double>(x.begin(), x.end());
  }
}

template <class T>
void from_value(size_t key, const config::Value& v, T& x) {
  if constexpr (std::is_same_v<T, bool>) {
    x = std::get<bool>(v);
  } else if constexpr (std::is_enum_v<T>) {
    x = enum_value<T>(key, std::get<std::string>(v));
  } else if constexpr (std::is_integral_v<T>) {
    x = static_cast<T>(std::get<int64_t>(v));  // check_value has checked the range ([8, 32], [0, inf), (-1e7, 1e7))
  } else if constexpr (std::is_floating_point_v<T>) {
    x = as_double(v);
  } else if constexpr (config::detail::IsOptional<T>::value) {
    if (std::holds_alternative<std::monostate>(v)) {
      x.reset();
    } else {
      typename T::value_type inner{};
      from_value(key, v, inner);
      x = inner;
    }
  } else {
    x = as_array<std::tuple_size_v<T>>(v);
  }
}

}  // namespace detail

// The values of `p`, one per key.
inline Values to_values(const ImuParams& p) {
  Values out;
  ImuParams copy = p;
  detail::each_field([&](size_t key, const auto& x) { out.emplace(key_name(key), detail::to_value(key, x)); }, copy);
  return out;
}

// ImuParams from one camera's resolved imu.* values. Every key is required and read once; a missing or unknown
// key, a wrong type, a value outside its hard range and a non-finite number are refused, naming the key.
inline ImuParams from_values(const Values& values) {
  const std::vector<config::Param>& rows = key_rows();
  for (const auto& [key, v] : values) {
    bool known = false;
    for (const config::Param& r : rows) known = known || r.key == key;
    if (!known) throw Refused("imu: unknown key " + key);
  }
  ImuParams p;
  detail::each_field([&](size_t key, auto& x) {
    const config::Param& row = rows[key];
    const auto it = values.find(row.key);
    if (it == values.end()) throw Refused("imu: missing key " + row.key);
    try {
      config::check_value(row, it->second, row.key);
    } catch (const std::invalid_argument& e) {
      throw Refused(e.what());
    }
    detail::from_value(key, it->second, x);
  }, p);
  return p;
}

// ImuParams{} as values: the defaults of every key.
inline Values default_values() { return to_values(ImuParams{}); }

struct Check {
  std::vector<std::string> errors, warnings;
};

// tau_c = 1 / (2 pi f_c) of a first-order sensor, in seconds; its pure delay d = group_delay_s - tau_c (§4.3.1).
inline double tau_c_s(const SensorErrors& e) { return 1.0 / (kTwoPi * e.cutoff_hz); }

namespace detail {

inline Mat3d upper_left(const std::array<double, 16>& t) {
  return {{{t[0], t[1], t[2]}, {t[4], t[5], t[6]}, {t[8], t[9], t[10]}}};
}

inline double det(const Mat3d& a) {
  const auto& m = a.m;
  return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])) +
         m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

// (A^-1)^T = cofactor(A) / det(A).
inline Mat3d inverse_transpose(const Mat3d& a) {
  const auto& m = a.m;
  const double s = 1.0 / det(a);
  Mat3d c;
  c.m[0][0] = s * (m[1][1] * m[2][2] - m[1][2] * m[2][1]);
  c.m[0][1] = s * (m[1][2] * m[2][0] - m[1][0] * m[2][2]);
  c.m[0][2] = s * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  c.m[1][0] = s * (m[0][2] * m[2][1] - m[0][1] * m[2][2]);
  c.m[1][1] = s * (m[0][0] * m[2][2] - m[0][2] * m[2][0]);
  c.m[1][2] = s * (m[0][1] * m[2][0] - m[0][0] * m[2][1]);
  c.m[2][0] = s * (m[0][1] * m[1][2] - m[0][2] * m[1][1]);
  c.m[2][1] = s * (m[0][2] * m[1][0] - m[0][0] * m[1][2]);
  c.m[2][2] = s * (m[0][0] * m[1][1] - m[0][1] * m[1][0]);
  return c;
}

// The §6 rigidity rule for a row-major 4 x 4 transform: every entry finite, the last row exactly (0, 0, 0, 1),
// |(R^T R - I)_ij| <= 1e-6 and |det R - 1| < 1e-6. Stricter than transforms.is_rigid, whose numpy rtol admits
// 1.1e-5 on the diagonal. Empty when rigid, else the reason.
inline std::string rigidity_error(const std::array<double, 16>& t) {
  for (double x : t) {
    if (!std::isfinite(x)) return "has a non-finite entry";
  }
  if (!(t[12] == 0 && t[13] == 0 && t[14] == 0 && t[15] == 1)) return "last row is not exactly (0, 0, 0, 1)";
  const Mat3d r = upper_left(t);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double g = (r.m[0][i] * r.m[0][j] + r.m[1][i] * r.m[1][j]) + r.m[2][i] * r.m[2][j];
      if (std::abs(g - (i == j ? 1.0 : 0.0)) > 1e-6) return "rotation is not orthonormal: |(R^T R - I)| > 1e-6";
    }
  }
  if (!(std::abs(det(r) - 1.0) < 1e-6)) return "rotation has |det R - 1| >= 1e-6";
  return {};
}

// The orthogonal polar factor of a near-orthogonal matrix by Newton's iteration X <- (X + X^-T) / 2, which
// converges quadratically (1e-6 -> 1e-12 -> ulp) and keeps exactly orthogonal signed permutations, the identity
// among them, exactly. Four steps, so the result is deterministic.
inline Mat3d polar(Mat3d x) {
  for (int step = 0; step < 4; ++step) {
    const Mat3d y = inverse_transpose(x);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) x.m[i][j] = 0.5 * (x.m[i][j] + y.m[i][j]);
    }
  }
  return x;
}

}  // namespace detail

// R_cam_imu of T_cam_imu, re-orthonormalized (polar); call after validate() has accepted the transform.
inline Mat3d rotation_of(const ImuParams& p) { return detail::polar(detail::upper_left(p.T_cam_imu)); }
// The IMU origin in the camera frame, p_cam_imu.
inline Vec3d translation_of(const ImuParams& p) { return {p.T_cam_imu[3], p.T_cam_imu[7], p.T_cam_imu[11]}; }
// The largest entry change that rotation_of's re-orthonormalization makes to T_cam_imu's rotation: 0 for an exactly
// orthogonal one, up to about 1e-6 for one validate() accepts (imu/<camera>.json records it, §6).
inline double rotation_change(const ImuParams& p) {
  const Mat3d raw = detail::upper_left(p.T_cam_imu), r = rotation_of(p);
  double d = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) d = std::max(d, std::abs(r.m[i][j] - raw.m[i][j]));
  }
  return d;
}

// §6: the hard ranges of kKeys (restated, so that a C++ or binding caller that skips the resolver is refused too),
// then the cross-key rules. The filter checks apply to first-order sensors only.
inline Check validate(const ImuParams& p) {
  Check c;
  const Values values = to_values(p);
  for (const config::Param& row : key_rows()) {
    try {
      config::check_value(row, values.find(row.key)->second, row.key);
    } catch (const std::invalid_argument& e) {
      c.errors.emplace_back(e.what());
    }
  }
  if (const std::string why = detail::rigidity_error(p.T_cam_imu); !why.empty()) {
    c.errors.push_back("imu.T_cam_imu " + why);
  }
  const std::pair<const char*, const SensorErrors*> sensors[] = {{"gyro", &p.gyro}, {"accel", &p.accel}};
  for (const auto& [name, e] : sensors) {
    const std::string k = std::string("imu.") + name + ".";
    if (p.noise_mode == NoiseMode::kShaped && e->filter != Filter::kFirstOrder) {
      c.errors.push_back("imu.noise_mode shaped: shaped noise is white noise through the DLPF: set " + k +
                         "filter=first_order");
    }
    if (e->filter != Filter::kFirstOrder || !(e->cutoff_hz > 0)) continue;
    if (e->group_delay_s < tau_c_s(*e)) {
      c.errors.push_back(k + "group_delay_s = " + config::format_double(e->group_delay_s) + " is below 1/(2 pi " +
                         k + "cutoff_hz) = " + config::format_double(tau_c_s(*e)) + " s (the pure delay would be negative)");
    }
    if (e->cutoff_hz >= p.rate_hz) {
      c.errors.push_back(k + "cutoff_hz = " + config::format_double(e->cutoff_hz) + " is at or above imu.rate_hz = " +
                         config::format_double(p.rate_hz));
    } else if (e->cutoff_hz >= p.rate_hz / 2) {
      c.warnings.push_back(k + "cutoff_hz = " + config::format_double(e->cutoff_hz) +
                           " is at or above the Nyquist rate imu.rate_hz / 2; aliasing above it is not modelled");
    }
  }
  if (p.clip_stage == ClipStage::kBoth) {
    if (!p.gyro.lsb && !p.accel.lsb) {
      c.errors.emplace_back("imu.clip_stage both needs imu.gyro.lsb or imu.accel.lsb (the rails are lsb counts)");
    } else {
      for (const auto& [name, e] : sensors) {
        if (!e->lsb) c.warnings.push_back(std::string("imu.clip_stage both: the pre-clip is inert for imu.") + name +
                                          " (imu." + name + ".lsb is null)");
      }
    }
  }
  if (p.gravity_mps2) {
    const auto& g = *p.gravity_mps2;
    const double n = std::sqrt((g[0] * g[0] + g[1] * g[1]) + g[2] * g[2]);
    if (!(n >= 9.7 && n <= 9.9)) {
      c.warnings.push_back("imu.gravity_mps2 has |g| = " + config::format_double(n) + " m/s^2, outside [9.7, 9.9]");
    }
  }
  return c;
}

// The per-sample variance of each sensor's additive white noise, {gyro, accel} in (rad/s)^2 and (m/s^2)^2: the ROS
// Imu covariances (outputs.md §4.8). White: sigma^2 / dt with dt = 1 / rate_hz (Kalibr's discrete form; the simulator
// draws sigma / sqrt(dt_k) at each realized step, which equals dt unless imu.clock_adev wanders the clock). Shaped:
// sigma^2 / (2 tau_c), the stationary variance of white noise through the first-order DLPF. The bias processes, the
// turn-on bias, the scale errors and the quantization are not in it.
inline std::array<double, 2> white_variance(const ImuParams& p) {
  std::array<double, 2> v{};
  const SensorErrors* sensors[2] = {&p.gyro, &p.accel};
  for (size_t i = 0; i < 2; ++i) {
    const double s2 = sensors[i]->noise_density * sensors[i]->noise_density;
    v[i] = p.noise_mode == NoiseMode::kShaped ? s2 / (2 * tau_c_s(*sensors[i])) : s2 * p.rate_hz;
  }
  return v;
}

}  // namespace balla::imu
