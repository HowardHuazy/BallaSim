// One camera's resolved imu.* values between Python and C++ (docs/dev/IMU.md; outputs.md §3.3): a dict of every
// imu.* key ({"imu.rate_hz": 1000.0, ...}, as _imu.default_values() and C5's `params` attribute hold them) to
// imu::ImuParams, typed by the keys' rows; imu::Refused on anything else. Header-only; shared by ballasim._imu and
// ballasim._export (the Kalibr values, A9), each its own binary.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <stdexcept>
#include <string>

#include "bindings/config_value.h"
#include "imu/params.h"

namespace balla::bindings {

inline imu::ImuParams imu_params_of(nanobind::handle values) {
  namespace nb = nanobind;
  if (!nb::isinstance<nb::dict>(values)) throw imu::Refused("imu: expected a dict of the imu.* values");
  imu::Values v;
  for (auto [k, x] : nb::borrow<nb::dict>(values)) {
    if (!nb::isinstance<nb::str>(k)) throw imu::Refused("imu: keys must be strings");
    const auto key = nb::cast<std::string>(k);
    try {
      const config::Node n = node_from_py(x, key);
      if (n.is_map) throw imu::Refused(key + ": expected a value, not a mapping");
      v.emplace(key, n.leaf);
    } catch (const std::invalid_argument& e) {
      throw imu::Refused(e.what());
    }
  }
  return imu::from_values(v);
}

}  // namespace balla::bindings
