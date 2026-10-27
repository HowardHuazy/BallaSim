// The parameter schema (docs/dev/P0_PARAMS.md §1): every row, from the five targets (sensor/param_fields,
// render/param_fields, engine/param_fields) and the rows without a target (schema.cpp), with the sensor family
// priors and the accuracy profiles. Read by _core.config, the docs generator (tools/config/gen_params_doc.py) and the
// resolver (resolve.h).
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "core/config/param.h"

namespace balla::config {

// An accuracy profile (D8, P0_PARAMS §1.3): a table chosen by accuracy.profile. phase1 lists every accuracy key at its
// Phase-1 value (frozen); fidelity lists the keys it derives from the run's sensors (formula only), the others keep
// their literals; reference (PHASE2 P2-0a) lists every key phase1 lists, literal or derived.
struct AccuracyEntry {
  std::string key;
  Value value;          // phase1: the value
  std::string formula;  // fidelity: the derivation
};
struct AccuracyProfile {
  std::string name, help;
  std::vector<AccuracyEntry> entries;
};
const std::vector<AccuracyProfile>& accuracy_profiles();
const AccuracyProfile* find_accuracy_profile(std::string_view name);

// Every row: SensorParams, SensorSpec, SceneSettings, RendererOptions and EngineConfig rows in member order,
// then the rows without a target.
const std::vector<Param>& schema();
// The row of a key, or nullptr.
const Param* find(std::string_view key);
// The rows of one target, in member order.
std::vector<const Param*> rows_of(Target target);
// The default of a key: a literal row's value; a family row's prior for `family` (required then); throws
// std::invalid_argument for unknown keys and for required and derived defaults.
Value default_value(std::string_view key, std::string_view family = {});

}  // namespace balla::config
