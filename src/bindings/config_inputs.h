// The resolver's inputs from Python (docs/dev/P0_PARAMS.md §2.3): the dict of ballasim.config (RunSpec.inputs) as
// config::Inputs, and the bundle facts (config.bundle_info) as config::BundleInfo. Shared by ballasim._core.config and
// ballasim._frames (resolve_frames), so that both read the same documents the same way.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>

#include "bindings/config_value.h"
#include "config/resolve.h"
#include "config/schema.h"

namespace balla::bindings {

namespace nb = nanobind;

// The inputs dict of ballasim.config: name, config, cli ({full key: value}), cli_profiles, profiles ({id: document}),
// facts ({key: value}), source_kind (evimo2, bundle, yaml, blender or "").
inline config::Inputs inputs_from_py(const nb::dict& d) {
  config::Inputs in;
  if (d.contains("name")) in.name = nb::cast<std::string>(d["name"]);
  if (d.contains("source_kind")) {
    in.source_kind = nb::cast<std::string>(d["source_kind"]);
    static const char* const kKinds[] = {"", "evimo2", "bundle", "yaml", "blender"};
    if (std::find(std::begin(kKinds), std::end(kKinds), in.source_kind) == std::end(kKinds)) {
      throw std::invalid_argument("resolver inputs: source_kind '" + in.source_kind +
                                  "' is not one of evimo2, bundle, yaml, blender or empty");
    }
  }
  if (!d.contains("config")) throw std::invalid_argument("resolver inputs: 'config' is required");
  in.config = node_from_py(d["config"], "");
  if (d.contains("cli")) in.cli = node_from_py(d["cli"], "cli");
  if (d.contains("cli_profiles")) in.cli_profiles = nb::cast<std::vector<std::string>>(d["cli_profiles"]);
  if (d.contains("profiles")) {
    for (auto [id, doc] : nb::cast<nb::dict>(d["profiles"])) {
      const auto key = nb::cast<std::string>(id);
      in.profiles.emplace(key, node_from_py(doc, "profile " + key));
    }
  }
  if (d.contains("facts")) {
    for (auto [k, v] : nb::cast<nb::dict>(d["facts"])) {
      const auto key = nb::cast<std::string>(k);
      const config::Param* p = config::find(key);
      if (p == nullptr) throw std::invalid_argument("resolver facts: unknown key " + key);
      in.facts.emplace(key, from_py(v, *p));
    }
  }
  return in;
}

inline config::BundleInfo bundle_from_py(const nb::dict& d) {
  config::BundleInfo b;
  for (nb::handle c : nb::cast<nb::list>(d["cameras"])) {
    const nb::dict cd = nb::cast<nb::dict>(c);
    b.cameras.push_back({nb::cast<std::string>(cd["name"]), cd.contains("sensor") ? nb::cast<std::string>(cd["sensor"]) : "",
                         nb::cast<int64_t>(cd["width"]), nb::cast<int64_t>(cd["height"]),
                         nb::cast<int64_t>(cd["clock_offset_ns"]),
                         cd.contains("pixel_pitch_m") ? nb::cast<double>(cd["pixel_pitch_m"]) : 0.0,
                         cd.contains("f_number") ? nb::cast<double>(cd["f_number"]) : 2.0});
  }
  b.t_start_ns = nb::cast<int64_t>(d["t_start_ns"]);
  b.t_end_ns = nb::cast<int64_t>(d["t_end_ns"]);
  const auto flag = [&](const char* k) { return d.contains(k) && nb::cast<bool>(d[k]); };
  // Required, so that a forgotten digest cannot drop the scene or the engine from the stage digests silently.
  const auto digest = [&](const char* k) {
    if (!d.contains(k) || !nb::isinstance<nb::str>(d[k])) {
      throw std::invalid_argument(std::string("bundle_info: '") + k + "' is required (a str; \"\" for no digest)");
    }
    return nb::cast<std::string>(d[k]);
  };
  b.device_available = flag("device_available");
  b.supplied = flag("supplied");
  b.bundle_digest = digest("bundle_digest");
  b.engine_digest = digest("engine_digest");
  if (d.contains("low_light") && !d["low_light"].is_none()) b.low_light = nb::cast<bool>(d["low_light"]);
  if (d.contains("shadow_lights")) b.shadow_lights = nb::cast<int64_t>(d["shadow_lights"]);
  if (d.contains("zero_background") && !d["zero_background"].is_none()) {
    b.zero_background = nb::cast<bool>(d["zero_background"]);
  }
  if (d.contains("baked")) {
    const nb::dict baked = nb::cast<nb::dict>(d["baked"]);
    const auto has = [&](const char* key) { return baked.contains(key) && nb::cast<bool>(baked[key]); };
    b.baked = {has("has_baked"), has("has_proxies"), has("all_lights_proxy"), has("lit_emissive")};
  }
  return b;
}

}  // namespace balla::bindings
