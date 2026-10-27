// C1 waveform dicts -> sensor waveforms, shared by the extension modules that build sensors
// (ballasim._sensor, ballasim._engine).
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "sensor/params.h"

namespace balla::sensor {

namespace nb = nanobind;

namespace waveform_dict {

inline double get_double(const nb::dict& d, const char* key) {
  if (!d.contains(key)) throw std::invalid_argument(std::string("light group: missing '") + key + "'");
  return nb::cast<double>(d[key]);
}

inline std::vector<double> get_list(const nb::dict& d, const char* key) {
  if (!d.contains(key)) throw std::invalid_argument(std::string("light group: missing '") + key + "'");
  return nb::cast<std::vector<double>>(d[key]);
}

}  // namespace waveform_dict

// A C1 waveform dict ({"type": ...}) or a light group carrying one under "waveform".
inline Waveform parse_waveform(nb::handle h) {
  using waveform_dict::get_double;
  using waveform_dict::get_list;
  nb::dict d = nb::cast<nb::dict>(h);
  if (d.contains("waveform")) d = nb::cast<nb::dict>(d["waveform"]);
  const std::string type = d.contains("type") ? nb::cast<std::string>(d["type"]) : std::string("");
  if (type == "constant") return make_constant_waveform();
  if (type == "fourier") return make_fourier_waveform(get_double(d, "f0_hz"), get_list(d, "a"), get_list(d, "phi"));
  if (type == "pulse") {
    return make_pulse_waveform(get_double(d, "f_hz"), get_double(d, "duty"), get_double(d, "rise_s"),
                               get_double(d, "phase_s"), get_double(d, "floor"));
  }
  throw std::invalid_argument("light group: waveform type must be 'constant', 'fourier' or 'pulse'");
}

}  // namespace balla::sensor
