// Cache adequacy and clocks (docs/design/phase3/ui.md §3.10, §4.6–4.7). Host-only; no renderer or device state.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "config/resolve.h"
#include "sensor/readout.h"

namespace balla::config {
enum class Tier { kNone, kO, kE, kS, kSTilde, kR };
inline const char* name(Tier tier) {
  constexpr const char* names[] = {"none", "O", "E", "S", "S~", "R"};
  return names[static_cast<int>(tier)];
}
struct CacheFacts {
  std::string photometry_mode;
  double scale = 1, reference_current_fA = 1;
  std::string precision = "fp32";
};
struct TierResult {
  Tier tier = Tier::kNone;
  std::vector<Stage> moved;
  std::vector<std::string> keys, reasons;
};
Resolved from_record(const Node& resolved, const Node& stage_digests);
TierResult classify(const Resolved& base, const Resolved& edit, std::string_view camera, const CacheFacts& facts);
// Actual photocurrent scale for replay: reference photometry rescales the writer's solved scale linearly.
double replay_scale(const Resolved& edit, std::string_view camera, const CacheFacts& facts);
std::string probe_digest(const Resolved&, std::string_view camera);
int64_t sensor_clock_ns(const sensor::Readout&, int64_t t_true_ns);
int64_t sensor_time_lower_bound(const sensor::Readout&, int64_t s_ns);
int64_t crossing_lower_bound(const sensor::Readout&, int64_t s_ns);
}  // namespace balla::config
