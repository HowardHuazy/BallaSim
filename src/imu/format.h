// The IMU's file formats (docs/design/phase3/outputs.md §3.5, §3.8): contract C5's `<run>/imu/<camera>.h5` and the
// per-camera record `<run>/imu/<camera>.json`. Readers refuse another name or another major version. Header-only.
#pragma once

#include <string>
#include <string_view>

#include "imu/params.h"

namespace balla::imu {

inline constexpr std::string_view kFileFormat = "ballasim.imu/1";     // C5, the HDF5 root attribute `format`
inline constexpr std::string_view kRunFormat = "ballasim.imu_run/1";  // imu/<camera>.json `format`

// Accepts `name/major` or `name/major.minor` with expected's name and major (expected is `name/major`); Refused
// otherwise.
inline void check_format(std::string_view value, std::string_view expected) {
  const size_t slash = expected.rfind('/');
  const std::string_view name = expected.substr(0, slash + 1);  // with the slash
  const std::string_view major = expected.substr(slash + 1);
  bool ok = slash != std::string_view::npos && value.substr(0, name.size()) == name;
  if (ok) {
    const std::string_view version = value.substr(name.size());
    const std::string_view head = version.substr(0, version.find('.'));
    ok = head == major && (head.size() == version.size() || version.size() > head.size() + 1);
  }
  if (!ok) {
    throw Refused("format '" + std::string(value) + "': this reader takes " + std::string(expected) +
                  " (another name or major version is refused)");
  }
}

}  // namespace balla::imu
