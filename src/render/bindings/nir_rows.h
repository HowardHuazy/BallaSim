// The per-camera sensor.nir_fraction rows of a run's values, for the bindings that build a Renderer outside an engine
// (ballasim._render Renderer.from_values, ballasim._frames ExposureReference; P2-F F11a). One rule, engine::make_engine's:
// a bundle camera the values do not list takes the first listed camera's row (in bundle order), so that it alone does
// not make the scene two-channel; a listed camera without the key has the empty row (f = 0).
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

namespace balla::render::bindings {

// One row per name, `names` being the bundle's camera names in order; `values` holds {"cameras": {name: {key: value}}}.
inline std::vector<std::vector<double>> nir_rows(const std::vector<std::string>& names, nanobind::handle values) {
  namespace nb = nanobind;
  std::vector<std::vector<double>> rows(names.size());
  nb::dict v = nb::cast<nb::dict>(values);
  if (!v.contains("cameras")) return rows;
  nb::dict cameras = nb::cast<nb::dict>(v["cameras"]);
  const std::vector<double>* first = nullptr;
  std::vector<bool> listed(names.size(), false);
  for (size_t i = 0; i < names.size(); ++i) {
    if (!cameras.contains(names[i].c_str())) continue;
    listed[i] = true;
    nb::dict c = nb::cast<nb::dict>(cameras[names[i].c_str()]);
    if (c.contains("sensor.nir_fraction")) rows[i] = nb::cast<std::vector<double>>(c["sensor.nir_fraction"]);
    if (first == nullptr) first = &rows[i];
  }
  for (size_t i = 0; first != nullptr && i < names.size(); ++i) {
    if (!listed[i]) rows[i] = *first;
  }
  return rows;
}

}  // namespace balla::render::bindings
