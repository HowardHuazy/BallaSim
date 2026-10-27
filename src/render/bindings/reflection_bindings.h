// _render.reflection_policy: DESIGN §7.7's reflection policy (render/reflection_policy.h; p2f.md §4.3, P2-F F4) for
// ballasim.engine.engine.run, which refuses or records it before the engine is built.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <utility>
#include <vector>

#include "bindings/config_value.h"
#include "render/bindings/bundle_convert.h"
#include "render/param_fields.h"
#include "render/reflection_policy.h"

namespace balla::render::bindings {

inline void bind_reflection(nb::module_& m) {
  m.def(
      "reflection_policy",
      [](nb::handle bundle, nb::handle values, const std::vector<std::pair<double, double>>& groups,
         const std::vector<std::vector<double>>& nir_fraction) {
        // The run's sensor RGB weights, through the SceneSettings rows as the renderer applies them.
        const RendererOptions o = apply_renderer(
            balla::bindings::values_from_py(values, {&scene_settings_rows(), &renderer_options_rows()}));
        const SceneBundle b = to_bundle(bundle);
        std::vector<ReflectionReport> reports;
        {
          nb::gil_scoped_release release;
          std::vector<GroupRange> ranges;
          for (const auto& [lo, hi] : groups) ranges.push_back({lo, hi});
          reports = reflection_policy(b, o.scene.sensor_rgb_weights, ranges, nir_fraction);
        }
        nb::list out;
        for (const ReflectionReport& r : reports) {
          nb::dict d;
          d["body"] = r.body;
          d["material"] = r.material;
          d["roughness"] = static_cast<double>(r.roughness);
          d["metallic"] = static_cast<double>(r.metallic);
          d["specular"] = static_cast<double>(r.specular);
          d["refused"] = r.refused;
          d["f_max"] = kReflectionFMax;
          d["l_max"] = r.l_max;
          d["l_diff_min"] = r.l_diff_min;
          d["bound"] = r.bound;
          out.append(d);
        }
        return out;
      },
      "bundle"_a, "values"_a, "groups"_a = std::vector<std::pair<double, double>>{},
      "nir_fraction"_a = std::vector<std::vector<double>>{},
      "DESIGN §7.7's reflection policy (render/reflection_policy.h): one dict per used material with a specular lobe\n"
      "(body and material indices, roughness, metallic, specular, refused, f_max, l_max, l_diff_min and bound =\n"
      "ln(1 + f_max l_max / l_diff_min), inf when unbounded), in band radiance under the resolved values' sensor RGB\n"
      "weights (ballasim.config.Resolved.values()). groups: (min, max) of every light group's waveform m_g(t) over\n"
      "all time (ballasim._engine.waveform_range), one per bundle.light_groups; the default () takes them constant.\n"
      "nir_fraction: the simulated cameras' sensor.nir_fraction rows (P2-F F11a; the default () is f = 0): the albedo\n"
      "bounds then take (1 - f) rho_vis + f rho_NIR over every f present.");
}

}  // namespace balla::render::bindings
