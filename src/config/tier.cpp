#include "config/tier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

#include "config/schema.h"
#include "config/sha256.h"
#include "sensor/maps.h"

namespace balla::config {
namespace {
constexpr const char* kStages[] = {"scene", "render", "schedule", "sensor", "output"};
const std::set<std::string_view> kE{"sensor.readout_delay_s", "sensor.readout_quantum_s", "sensor.clock_offset_ns",
                                      "sensor.clock_drift_ppm", "output.label_cut"};
// Every schedule row not in the monotone groups below must remain equal. New rows are refused until classified.
const std::set<std::string_view> kEqual{
    "photometry.mode", "photometry.reference_instance", "optics.vignetting", "optics.vignetting_law",
    "optics.relative_illumination", "scene.evimo2.mains_a", "scene.evimo2.strobe_width_s",
    "scene.evimo2.strobe_floor", "accuracy.refine_signature", "accuracy.max_refine_levels", "accuracy.mid_levels",
    "accuracy.tau_jump_ns", "accuracy.delta_ref", "accuracy.half_ratio", "accuracy.eps_jump", "accuracy.shadow_band",
    "accuracy.shadow_obliquity", "accuracy.localization", "accuracy.level_policy", "accuracy.min_dt_ns",
    "accuracy.d_px", "accuracy.max_samples_per_window", "run.fixed_rate_hz", "run.crop_px", "run.t_start_ns",
    "run.t_end_ns", "accuracy.warmup", "accuracy.warmup_ns"};
const std::set<std::string_view> kDarkScale{
    "sensor.dark_current_fA", "sensor.dark_ea_eV", "sensor.temperature_K", "sensor.reference_temperature_K",
    "photometry.illuminance_scale", "photometry.reference_current_fA", "sensor.qe", "sensor.kappa",
    "optics.f_number", "optics.transmittance"};

const Node* child(const Node& node, std::string_view key) {
  for (const auto& [k, n] : node.map) if (k == key) return &n;
  return nullptr;
}
const Node& required(const Node& node, std::string_view key) {
  const Node* n = child(node, key);
  if (!node.is_map || n == nullptr) throw std::invalid_argument("run record: missing mapping key " + std::string(key));
  return *n;
}
std::string text(const Node& n) {
  const auto* value = std::get_if<std::string>(&n.leaf);
  if (n.is_map || value == nullptr) throw std::invalid_argument("run record: expected a string");
  return *value;
}
Entries entries(const Node& node) {
  if (!node.is_map) throw std::invalid_argument("run record: resolved section must be a mapping");
  Entries out;
  for (const auto& [key, n] : node.map) {
    const Param* row = find(key);
    if (row == nullptr) throw std::invalid_argument("run record: unknown key " + key);
    const Node& v = required(n, "value");
    if (v.is_map) throw std::invalid_argument("run record: a value cannot be a mapping");
    Value value = v.leaf;
    if (row->type == Type::kFloat && std::holds_alternative<int64_t>(value)) {
      value = static_cast<double>(std::get<int64_t>(value));
    }
    if (row->type == Type::kFloatList && std::holds_alternative<std::vector<int64_t>>(value)) {
      const auto integers = std::get<std::vector<int64_t>>(value);
      value = std::vector<double>(integers.begin(), integers.end());
    }
    if (const auto* list = std::get_if<std::vector<double>>(&value); list != nullptr && list->empty()) {
      if (row->type == Type::kIntList) value = std::vector<int64_t>{};
      if (row->type == Type::kStringList) value = std::vector<std::string>{};
    }
    check_value(*row, value, "run record " + key);
    const Node* note = child(n, "note");
    out.emplace_back(key, Entry{std::move(value), text(required(n, "layer")), text(required(n, "status")),
                                note != nullptr ? text(*note) : ""});
  }
  return out;
}
size_t camera_index(const Resolved& r, std::string_view camera) {
  for (size_t i = 0; i < r.cameras.size(); ++i) if (r.cameras[i].first == camera) return i;
  throw std::invalid_argument("cache classification: unknown camera " + std::string(camera));
}
const Value* value(const Resolved& r, std::string_view key, std::string_view camera) {
  const Entry* e = r.find(key, camera);
  if (e == nullptr) e = r.find(key);
  return e == nullptr ? nullptr : &e->value;
}
double number(const Resolved& r, std::string_view key, std::string_view camera) {
  const Value* v = value(r, key, camera);
  if (v != nullptr) {
    if (const auto* d = std::get_if<double>(v)) return *d;
    if (const auto* i = std::get_if<int64_t>(v)) return static_cast<double>(*i);
  }
  throw std::invalid_argument("cache classification: missing number " + std::string(key));
}
std::string string(const Resolved& r, std::string_view key, std::string_view camera) {
  const Value* v = value(r, key, camera);
  if (v != nullptr) if (const auto* s = std::get_if<std::string>(v)) return *s;
  throw std::invalid_argument("cache classification: missing string " + std::string(key));
}
bool same_value(const Resolved& a, const Resolved& b, std::string_view key, std::string_view camera) {
  const Value *va = value(a, key, camera), *vb = value(b, key, camera);
  return va == nullptr || vb == nullptr ? va == vb : *va == *vb;
}
double dark(const Resolved& r, std::string_view camera) {
  sensor::SensorParams p;
  p.dark_current_fA = number(r, "sensor.dark_current_fA", camera);
  p.dark_ea_eV = number(r, "sensor.dark_ea_eV", camera);
  p.temperature_K = number(r, "sensor.temperature_K", camera);
  p.reference_temperature_K = number(r, "sensor.reference_temperature_K", camera);
  return sensor::median_dark_fA(p);
}
// Wide arithmetic makes inverse-clock comparisons defined even when the clock has no int64 representation.
using Wide = __int128_t;
void check_clock(const sensor::Readout& r) {
  if (!std::isfinite(r.clock_drift_ppm) || std::fabs(r.clock_drift_ppm) >= 10000 || r.quantum_ns < 0 || r.delay_ns < 0 ||
      (r.quantum_ns > 0 && (r.phase_ns < 0 || r.phase_ns >= r.quantum_ns))) {
    throw std::invalid_argument("sensor clock: invalid drift, delay, quantum or phase");
  }
}
Wide clock(const sensor::Readout& r, int64_t t) {
  return static_cast<Wide>(t) + r.clock_offset_ns +
         std::llround(r.clock_drift_ppm * 1e-6 * static_cast<double>(t));
}
int64_t narrow(Wide t) {
  if (t < std::numeric_limits<int64_t>::min() || t > std::numeric_limits<int64_t>::max()) {
    throw std::overflow_error("sensor clock: time outside int64");
  }
  return static_cast<int64_t>(t);
}
}  // namespace

Resolved from_record(const Node& record, const Node& stage_digests) {
  Resolved out;
  out.run = entries(required(record, "run"));
  const Node& cameras = required(record, "cameras");
  if (!cameras.is_map) throw std::invalid_argument("run record: cameras must be a mapping");
  for (const auto& [camera, n] : cameras.map) {
    out.cameras.emplace_back(camera, entries(n));
    std::array<std::string, 5> digests;
    const Node& d = required(stage_digests, camera);
    for (size_t i = 0; i < digests.size(); ++i) digests[i] = text(required(d, kStages[i]));
    out.digests.push_back(std::move(digests));
  }
  return out;
}

double replay_scale(const Resolved& edit, std::string_view camera, const CacheFacts& facts) {
  double scale = number(edit, "photometry.illuminance_scale", camera);
  if (string(edit, "photometry.mode", camera) == "reference") {
    if (facts.photometry_mode != "reference" || !(facts.reference_current_fA > 0) || !(facts.scale > 0)) {
      throw std::invalid_argument("reference cache: missing positive solved scale or reference current");
    }
    scale = facts.scale * number(edit, "photometry.reference_current_fA", camera) / facts.reference_current_fA;
  }
  if (!(scale > 0) || !std::isfinite(scale)) {
    throw std::invalid_argument("cache: photocurrent scale must be finite > 0");
  }
  return scale;
}

TierResult classify(const Resolved& base, const Resolved& edit, std::string_view camera, const CacheFacts& facts) {
  if (facts.precision != "fp32" && facts.precision != "fp64") {
    throw std::invalid_argument("cache precision: fp32 or fp64");
  }
  const size_t ib = camera_index(base, camera), ie = camera_index(edit, camera);
  if (base.digests.size() != base.cameras.size() || edit.digests.size() != edit.cameras.size()) {
    throw std::invalid_argument("cache classification needs all stage digests");
  }
  TierResult out;
  for (size_t i = 0; i < 5; ++i) {
    if (base.digests[ib][i] != edit.digests[ie][i]) out.moved.push_back(static_cast<Stage>(i));
  }
  const auto moved = [&](Stage s) { return std::find(out.moved.begin(), out.moved.end(), s) != out.moved.end(); };
  for (const Param& row : schema()) {
    if (row.stage != Stage::kNone && !same_value(base, edit, row.key, camera) &&
        (value(base, row.key, camera) != nullptr || value(edit, row.key, camera) != nullptr)) {
      out.keys.push_back(row.key);
    }
  }
  const Value* limiter = value(edit, "sensor.limiter", camera);
  if (limiter != nullptr && *limiter != Value{std::string("off")}) {
    out.tier = Tier::kR;
    out.reasons.push_back("sensor.limiter: camera-wide rate limiting cannot replay an ROI");
    return out;
  }
  if (out.moved.empty()) return out;
  if (!moved(Stage::kScene) && !moved(Stage::kRender) && !moved(Stage::kSchedule) && !moved(Stage::kSensor)) {
    out.tier = Tier::kO;
    return out;
  }
  if (!moved(Stage::kScene) && !moved(Stage::kRender) && !moved(Stage::kSchedule)) {
    const bool e = !out.keys.empty() && std::all_of(out.keys.begin(), out.keys.end(), [&](const auto& key) {
      return kE.contains(key) && !(facts.precision == "fp64" && key == "output.label_cut");
    });
    out.tier = e ? Tier::kE : Tier::kS;
    return out;
  }
  out.tier = Tier::kR;
  if (moved(Stage::kScene) || moved(Stage::kRender)) {
    out.reasons.push_back("scene or render changed; rebuild the signal cache");
    return out;
  }
  for (const Param& row : schema()) {
    if (row.stage != Stage::kSchedule) continue;
    if (kDarkScale.contains(row.key) || row.key == "accuracy.refine_tol_log") continue;
    if (!kEqual.contains(row.key) || !same_value(base, edit, row.key, camera)) {
      out.reasons.push_back(row.key + ": schedule row must remain equal (or has no adequacy rule)");
    }
  }
  const Value *a = value(base, "accuracy.refine_tol_log", camera), *b = value(edit, "accuracy.refine_tol_log", camera);
  const bool null_a = a == nullptr || std::holds_alternative<std::monostate>(*a);
  const bool null_b = b == nullptr || std::holds_alternative<std::monostate>(*b);
  if (a == nullptr || b == nullptr || null_a != null_b || (!null_a && number(base, "accuracy.refine_tol_log", camera) >
                                    number(edit, "accuracy.refine_tol_log", camera))) {
    out.reasons.push_back("accuracy.refine_tol_log: the cached tolerance must be no larger than the edit's");
  }
  if (!same_value(base, edit, "photometry.mode", camera)) return out;
  const double scale_base = facts.photometry_mode == "reference" ? facts.scale :
                            number(base, "photometry.illuminance_scale", camera);
  if (!(scale_base > 0) || !std::isfinite(scale_base) ||
      dark(base, camera) / scale_base > dark(edit, camera) / replay_scale(edit, camera, facts)) {
    out.reasons.push_back("dark/scale: the cache must have no larger dark term than the edit");
  }
  if (out.reasons.empty()) out.tier = Tier::kSTilde;
  return out;
}

std::string probe_digest(const Resolved& r, std::string_view camera) {
  const size_t i = camera_index(r, camera);
  if (r.digests.size() != r.cameras.size()) throw std::invalid_argument("probe digest needs stage digests");
  std::string text = r.digests[i][2] + "\nsensor\n";
  const Value* limiter = value(r, "sensor.limiter", camera);
  const bool limiter_off = limiter == nullptr || *limiter == Value{std::string("off")};
  for (const bool camera_rows : {false, true}) {
    for (const Param& row : schema()) {
      if ((row.scope == Scope::kCamera) != camera_rows || row.stage != Stage::kSensor || kE.contains(row.key)) continue;
      if (limiter_off && row.key.starts_with("sensor.limiter")) continue;
      const Value* v = value(r, row.key, camera);
      if (v != nullptr) text += row.key + "=" + canonical_text(*v) + "\n";
    }
    text += "--\n";
  }
  return sha256_hex(text);
}
int64_t sensor_clock_ns(const sensor::Readout& r, int64_t t) {
  check_clock(r);
  return narrow(clock(r, t));
}
int64_t sensor_time_lower_bound(const sensor::Readout& r, int64_t s) {
  check_clock(r);
  // At extreme int64 times, negative drift can make the engine's double/llround clock locally non-monotone
  // (double(t) changes in 1024 ns steps). Locate the ideal crossing then scan its bounded rounding neighbourhood.
  // |drift| < 1e4 ppm bounds conversion/product error below 32 ns, well inside this 512 ns margin.
  const long double rate = static_cast<long double>(r.clock_drift_ppm * 1e-6);
  const long double ideal = (static_cast<long double>(s) - r.clock_offset_ns) / (1 + rate);
  constexpr int64_t kMargin = 512;
  const long double low = std::numeric_limits<int64_t>::min(), high = std::numeric_limits<int64_t>::max();
  if (ideal < low - kMargin) return std::numeric_limits<int64_t>::min();
  if (ideal > high + kMargin) throw std::overflow_error("sensor clock: no representable lower bound");
  const int64_t anchor = static_cast<int64_t>(std::clamp(ideal, low, high));
  const Wide first = std::max<Wide>(std::numeric_limits<int64_t>::min(), static_cast<Wide>(anchor) - kMargin);
  const Wide last = std::min<Wide>(std::numeric_limits<int64_t>::max(), static_cast<Wide>(anchor) + kMargin);
  for (Wide t = first; t <= last; ++t) if (clock(r, static_cast<int64_t>(t)) >= s) return static_cast<int64_t>(t);
  throw std::overflow_error("sensor clock: no representable lower bound");
}
int64_t crossing_lower_bound(const sensor::Readout& r, int64_t s) {
  Wide t = sensor_time_lower_bound(r, s);
  if (r.quantum_ns > 0) {
    const Wide delta = t - r.phase_ns, q = r.quantum_ns;
    Wide k = delta / q;
    if (delta % q > 0) ++k;
    t = r.phase_ns + k * q;
    while (clock(r, narrow(t)) < s) t += q;  // a rare local rounding dip near the int64 endpoints
  }
  // The inverse clock can already accept the first representable tick. With a positive delay, that still makes
  // INT64_MIN the first crossing, rather than an unrepresentable time below it.
  return narrow(std::max<Wide>(std::numeric_limits<int64_t>::min(), t - r.delay_ns));
}
}  // namespace balla::config
