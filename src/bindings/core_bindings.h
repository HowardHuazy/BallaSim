// Registration functions of the ballasim._core submodule groups and their shared defaults.
#pragma once

#include <nanobind/nanobind.h>

#include "render/scene_data.h"

namespace balla::bindings {

// Default pixel response (DESIGN §6.1): the renderer's SceneSettings defaults, box(1 px) convolved with
// Gaussian(0.15 px), truncated at 3 sigma. Its reach x_max = w/2 + n_sigma sigma = 0.95 px (render::support_px,
// the single PSF source) is the default sub-sample support that the cameras' injective domains must cover.
inline constexpr render::SceneSettings kDefaultSettings{};
inline constexpr double kDefaultSupportPx = render::support_px(kDefaultSettings);

void bind_traj(nanobind::module_& m);
void bind_camera(nanobind::module_& m);
void bind_rng(nanobind::module_& m);
void bind_config(nanobind::module_& m);  // _core.config: the parameter schema (core_config.cpp)
void bind_wire(nanobind::module_& m);    // _core.wire: the browser UI's event wire (core_wire.cpp)

}  // namespace balla::bindings
