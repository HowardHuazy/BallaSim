// The SensorParams rows and the sensor family priors (docs/dev/P0_PARAMS.md §1.3): one descriptor per
// field, in member order. validate(), fingerprint(), preset() and the attributes of
// ballasim._sensor.SensorParams iterate these rows, and ballasim._engine's sensor overrides set fields
// through them by name (Param::field).
#pragma once

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "core/config/param.h"
#include "sensor/params.h"

namespace balla::config {

template <>
struct TargetOf<sensor::SensorParams> {
  static constexpr Target value = Target::kSensorParams;
};
template <>
struct EnumNames<sensor::ResetMode> {
  static constexpr std::array<std::string_view, 2> names{"physical", "ideal"};
};
template <>
struct EnumNames<sensor::NoiseDiffusion> {
  static constexpr std::array<std::string_view, 2> names{"frozen", "transient"};
};
template <>
struct EnumNames<sensor::Photoreceptor> {
  static constexpr std::array<std::string_view, 2> names{"cascade", "instantaneous"};
};
template <>
struct EnumNames<sensor::InitialPhase> {
  static constexpr std::array<std::string_view, 3> names{"uniform", "symmetric", "zero"};
};
template <>
struct EnumNames<sensor::InitialState> {
  static constexpr std::array<std::string_view, 2> names{"phase", "stationary"};
};
template <>
struct EnumNames<sensor::HotRateLaw> {
  static constexpr std::array<std::string_view, 2> names{"log_uniform", "log_normal"};
};
template <>
struct EnumNames<sensor::LimiterMode> {
  static constexpr std::array<std::string_view, 2> names{"off", "epoch"};
};

}  // namespace balla::config

namespace balla::sensor {

// The SensorParams rows, in member order.
const std::vector<config::Param>& param_rows();
// The row of a SensorParams field (Param::field), or nullptr.
const config::Param* param_row(std::string_view field);
// The error message for a name that is no SensorParams field; a removed field names its replacement
// ("sigma_theta": split into sigma_theta_on and sigma_theta_off, D11).
std::string unknown_field(std::string_view field);
// What replaces a removed SensorParams field ("" for any other name).
std::string_view removed_field(std::string_view field);
// The rows that describe hot pixels (sensor.hot_rate_*, .hot_leak_fraction, .hot_reference_current_fA): inert
// without hot pixels (SensorParams::hot_pixels(), hot_fraction 0), a condition a row's `when` cannot state.
bool hot_pixel_key(std::string_view key);
// The SensorParams of camera `name`'s resolved values (config::apply, docs/dev/P0_PARAMS.md §1.4): its camera keys
// from `cam`, the run keys applied to every camera (accuracy.eps_L, ...) from `run`; checked against the rows and
// validated (std::invalid_argument naming the camera).
SensorParams params_of(const config::ValueMap& run, const config::ValueMap& cam, const std::string& name);

// One value of a family prior: a family-default row's key, its value, and where it comes from. evimo2_derived: the
// value comes from EVIMO2 data or its setup (the illuminant assumed for it); no_literature: the literature gives no
// usable value, so the prior turns the feature off (D4c, PARAMETERS §5.9). The resolver warns once per run about each
// kind (resolve.cpp).
struct Prior {
  std::string key;
  config::Value value;
  config::Status status;
  std::string source;
  bool evimo2_derived = false;
  bool no_literature = false;
};

// A sensor family: a base family (whose priors it inherits, unvalidated) and its own differences.
struct Family {
  std::string name, base, help;
  std::vector<Prior> priors;
};

const std::vector<Family>& families();
const Family* find_family(std::string_view name);
// The prior of `key` for `family`: its own entry, else its base's (inherited set to true then); nullptr if
// neither has one.
const Prior* find_prior(std::string_view family, std::string_view key, bool* inherited = nullptr);

}  // namespace balla::sensor
