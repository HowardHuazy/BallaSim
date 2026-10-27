// The resolver (docs/dev/P0_PARAMS.md §2): layers, resolution, the checks of §2.4, the accuracy profiles and their
// derivations (§1.3), validate_run(), the window span, the stage digests (§1.2) and the record (§2.6); every error and
// warning as a Diagnostic (ui.md §3.10, UI-4a), explain, relevant and derived_from.
#include "config/resolve.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <initializer_list>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "config/schema.h"
#include "config/sha256.h"
#include "core/camera/relative_illumination.h"
#include "core/time/time.h"
#include "imu/params.h"
#include "sensor/noise_tables.h"
#include "sensor/param_fields.h"
#include "sensor/params.h"

namespace balla::config {

const Entry* Resolved::find(std::string_view key, std::string_view camera) const {
  const Entries* list = camera.empty() ? &run : nullptr;
  for (const auto& [cam, entries] : cameras) {
    if (!camera.empty() && cam == camera) list = &entries;
  }
  if (list == nullptr) return nullptr;
  for (const auto& [k, e] : *list) {
    if (k == key) return &e;
  }
  return nullptr;
}

namespace {

constexpr double kElementaryCharge = 1.602176634e-19;  // C (exact, SI)
constexpr double kPi = 3.14159265358979323846;
constexpr double kAiryGaussian = 0.42;                 // Airy core sigma / (lambda N) (lens_blur)
constexpr double kEpsRef = 0.002;                      // kappa sigma_v,ref (PARAMETERS §4.1)
constexpr double kSigmaRef = 0.02;                     // sigma_v,ref (log-e)
constexpr double kNoiseTopRate = 1e9;                  // a1 at the top node of the noise tables (kNoiseLnA1Max)
constexpr int64_t kMinDtFloorNs = 1024;                // the Phase-1 floor of min_dt_ns (lead decision 4)
constexpr int64_t kMaxFidelityN = 256;
// balanced's d_px against fidelity's (ENGINE_CPP I7: the samples study's ES, d_px 1 at fidelity's 0.25), capped at
// DESIGN's D_cap of 1 px (the I7 review: at N = 16, from design thresholds >= 0.41 log-e, 4x would be 2 px, the d_px
// that I7 did not adopt).
constexpr double kEconomyFactor = 4.0;
constexpr double kEconomyDpxCap = 1.0;
constexpr double kFastPoleS = 25e-6;                   // noise_leaf_level warning below this min(tau_o, tau_sf)
constexpr int64_t kMinStaticWarmupNs = 50'000'000;     // DESIGN §8.4's minimum of a static warm-up
// PHASE2 D-P2-4 (P2-B B4a): fidelity's band N, G1's interim cap (the user chose EFC; until then no c_loc), and the
// smallest N_base beside it (P2-B B3: T1, an interior sample's value at N >= 64 equal to its value at 64, holds bitwise
// at powers of two). F085's floor of the band threshold (log-e), and kappa_q's constants of F068 (c_cover).
constexpr int64_t kBandN = 256;
constexpr int64_t kBandBaseMin = 64;
// P2-B B6c: the smallest pixel-response reach (px) at which pooled escalation's premise holds (the 64-point hulls of
// neighbouring pixels leave no strip between them; ENGINE_CPP §0.1 "P2-B B6 notes", phase2-checks/p2_b6_pool_hull.py),
// 0.5 aperture + min(n, kPixelBandTruncationMax) sigma_blur: the threshold holds at truncations n <= 3 sigma (the
// hulls lose less than n sigma when n falls), and a wider truncation widens support_px but not the 64 points' extremes
// (at aperture 0.3 px, blur 0.15 px and n = 4, support_px 0.75, pooling misses 4 % of the tail-only edges).
constexpr double kPixelBandSupportMin = 0.75, kPixelBandTruncationMax = 3.0;
constexpr double kThetaBandFloor = 0.10;
constexpr double kKappaQ = 0.35, kKappaQRate = 0.4;
const char* const kStages[] = {"scene", "render", "schedule", "sensor", "output"};

// A diagnostic of `kind` with its message and what it concerns (`key` joins `keys` first; ui.md §3.10).
Diagnostic diag(DiagKind kind, Severity severity, std::string message, std::string key, std::string camera,
                std::string layer, std::vector<std::string> keys) {
  if (!key.empty() && std::find(keys.begin(), keys.end(), key) == keys.end()) keys.insert(keys.begin(), key);
  return {kind, severity, std::move(key), std::move(camera), std::move(layer), std::move(keys), std::move(message), {}};
}

// Errors are fail-fast: one DiagnosticError (a std::invalid_argument with the message) per resolve.
[[noreturn]] void fail(Diagnostic d) { throw DiagnosticError(std::move(d)); }
[[noreturn]] void fail(DiagKind kind, std::string message, std::string key = {}, std::string camera = {},
                       std::string layer = {}, std::vector<std::string> keys = {}) {
  fail(diag(kind, Severity::kError, std::move(message), std::move(key), std::move(camera), std::move(layer),
            std::move(keys)));
}

// The run's warnings: Resolved::warnings and their diagnostics, kept parallel.
struct Warnings {
  std::vector<std::string>& text;
  std::vector<Diagnostic>& diags;
  void operator()(DiagKind kind, std::string message, std::string key = {}, std::string camera = {},
                  std::string layer = {}, std::vector<std::string> keys = {}) {
    text.push_back(message);
    diags.push_back(diag(kind, Severity::kWarning, std::move(message), std::move(key), std::move(camera),
                         std::move(layer), std::move(keys)));
  }
};

// Three significant digits (whole numbers from 100 on), for derived quantities quoted in warnings and notes.
std::string fmt3(double v) {
  if (std::abs(v) >= 100 && std::abs(v) < 1e15) return std::to_string(std::llround(v));
  char buf[32];
  const auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::general, 3);
  return std::string(buf, r.ptr);
}

std::string section_of(std::string_view key) { return std::string(key.substr(0, key.find('.'))); }

bool is_null(const Node& n) { return !n.is_map && std::holds_alternative<std::monostate>(n.leaf); }

// A section of the schema: a prefix of some key (an empty one is written as null in YAML).
bool is_section(std::string_view key) {
  for (const Param& p : schema()) {
    if (p.key.size() > key.size() && p.key.compare(0, key.size(), key) == 0 && p.key[key.size()] == '.') return true;
  }
  return false;
}

// The key-map hint of an unknown key that format 1 had (P0_PARAMS §3.1): its sections render and schedule, its
// top-level span and crop, and its sensor fields outside the SensorParams rows (preset, overrides, ...).
std::string format1_hint(std::string_view key) {
  const std::string s = section_of(key);
  bool old = s == "render" || s == "schedule" || s == "t_start_ns" || s == "t_end_ns" || s == "crop_px";
  for (const std::string_view f : {"sensor.preset", "sensor.overrides", "sensor.illuminance_scale", "sensor.device",
                                   "sensor.seed", "sensor.num_threads", "sensor.precision", "sensor.warmup_ns"}) {
    old = old || key == f || (key.size() > f.size() && key.compare(0, f.size(), f) == 0 && key[f.size()] == '.');
  }
  return old ? "; a format-1 key: the key map is in docs/dev/P0_PARAMS.md §3.1" : "";
}

// ---------------------------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------------------------

// A value a layer sets: typed and checked, with the status and source a profile entry gives it.
struct Set {
  Value value;
  std::string status, where;  // where: the key as written (messages)
  std::string source;         // a profile entry's source (explain)
};

struct Layer {
  std::string label;   // messages: "config x.yaml", "profile evimo2/x@v0", "cli"
  std::string record;  // run.json: config, profile:<id>, cli
  bool user = false;   // the run configuration or the CLI (status user; inert warnings)
  std::map<std::string, Set, std::less<>> run;                                       // run, scene, dataset keys
  std::map<std::string, std::map<std::string, Set, std::less<>>, std::less<>> cams;  // "default" or a camera
  std::string source;  // explain: the configuration's file, "--set", the profile id
};

const Node* child(const Node& n, std::string_view k) {
  for (const auto& [key, v] : n.map) {
    if (key == k) return &v;
  }
  return nullptr;
}

std::string text(const Node& n, const std::string& where, DiagKind kind = DiagKind::kFormat) {
  if (const auto* s = n.is_map ? nullptr : std::get_if<std::string>(&n.leaf)) return *s;
  fail(kind, where + ": expected a string");
}

int edit_distance(std::string_view a, std::string_view b) {
  std::vector<int> d(b.size() + 1);
  for (size_t j = 0; j <= b.size(); ++j) d[j] = static_cast<int>(j);
  for (size_t i = 1; i <= a.size(); ++i) {
    int diag = d[0];
    d[0] = static_cast<int>(i);
    for (size_t j = 1; j <= b.size(); ++j) {
      const int up = d[j];
      d[j] = std::min({d[j] + 1, d[j - 1] + 1, diag + (a[i - 1] == b[j - 1] ? 0 : 1)});
      diag = up;
    }
  }
  return d[b.size()];
}

// The nearest key of the same placement within edit distance 2, or nullptr.
const Param* nearest(std::string_view key, bool camera) {
  const Param* best = nullptr;
  int dist = 3;
  for (const Param& p : schema()) {
    if ((p.scope == Scope::kCamera) != camera) continue;
    const int d = edit_distance(key, p.key);
    if (d < dist) best = &p, dist = d;
  }
  return best;
}

// " (did you mean K?)" for the nearest key.
std::string suggest(const Param* best) { return best ? " (did you mean " + best->key + "?)" : ""; }

// The kind of a value check_value refused: a number (or a list of numbers) of the row's type outside the hard range or
// not finite is kHardRange; null where not nullable, a wrong type, an enum outside its choices or a list of the wrong
// length is kType (P0_PARAMS §2.4 "type or enum").
DiagKind value_kind(const Param& p, const Value& v) {
  const bool num = std::holds_alternative<double>(v) || std::holds_alternative<int64_t>(v);
  switch (p.type) {
    case Type::kFloat: return num ? DiagKind::kHardRange : DiagKind::kType;
    case Type::kInt: return std::holds_alternative<int64_t>(v) ? DiagKind::kHardRange : DiagKind::kType;
    case Type::kFloatList:
    case Type::kIntList: {
      size_t n = 0;
      if (const auto* li = std::get_if<std::vector<int64_t>>(&v)) n = li->size();
      else if (const auto* l = std::get_if<std::vector<double>>(&v); l && p.type == Type::kFloatList) n = l->size();
      else return DiagKind::kType;
      return p.length < 0 || n == static_cast<size_t>(p.length) ? DiagKind::kHardRange : DiagKind::kType;
    }
    default: return DiagKind::kType;
  }
}

// The value as the row types it (an integer is a float; an empty list is any list), checked against the row.
Value typed(const Param& p, Value v, const std::string& what) {
  if (const auto* i = std::get_if<int64_t>(&v); i && p.type == Type::kFloat) v = static_cast<double>(*i);
  if (const auto* l = std::get_if<std::vector<int64_t>>(&v); l && p.type == Type::kFloatList) {
    v = std::vector<double>(l->begin(), l->end());
  }
  if (const auto* l = std::get_if<std::vector<double>>(&v); l && l->empty()) {
    if (p.type == Type::kIntList) v = std::vector<int64_t>{};
    if (p.type == Type::kStringList) v = std::vector<std::string>{};
  }
  check_value(p, v, what);
  return v;
}

// Adds `key` (below the camera block `cam`, or a run key when empty) to the layer: unknown key, scope, type and hard
// range are checked here (§2.4).
void put(Layer& L, const std::string& cam, const std::string& key, const Node& n, const std::string& where,
         const std::string& status, const std::string& source = {}) {
  const Param* p = find(key);
  if (p == nullptr && key.rfind("sensor.", 0) == 0) {
    if (const std::string_view r = sensor::removed_field(key.substr(7)); !r.empty()) {
      fail(DiagKind::kUnknownKey, L.label + ": unknown key " + where + ": " + std::string(r), key, cam, L.record);
    }
  }
  if (p == nullptr) {
    const Param* near = nearest(key, !cam.empty());
    Diagnostic d = diag(DiagKind::kUnknownKey, Severity::kError,
                        L.label + ": unknown key " + where + suggest(near) + format1_hint(key), key, cam, L.record, {});
    if (near != nullptr) d.nearest = near->key;
    fail(std::move(d));
  }
  if (!cam.empty() && p->scope != Scope::kCamera) {
    fail(DiagKind::kWrongScope,
         L.label + ": " + where + ": a " + name(p->scope) + " key; set it under " + section_of(key) + ":", key, cam,
         L.record);
  }
  if (cam.empty() && p->scope == Scope::kCamera) {
    fail(DiagKind::kWrongScope,
         L.label + ": " + where + ": a camera key; set it under cameras.default." + key + " or cameras.<camera>." + key,
         key, cam, L.record);
  }
  if (n.is_map) fail(DiagKind::kType, L.label + ": " + where + ": expected a value, not a mapping", key, cam, L.record);
  Value v;
  try {
    v = typed(*p, n.leaf, L.label + ": " + where);
  } catch (const std::invalid_argument& e) {
    fail(value_kind(*p, n.leaf), e.what(), key, cam, L.record);
  }
  Set s{std::move(v), status, where, source};
  (cam.empty() ? L.run[key] : L.cams[cam][key]) = std::move(s);
}

// The leaves below a mapping, as dotted keys; a null section is an empty one (YAML's "accuracy:" with every entry
// commented out).
void walk(Layer& L, const std::string& cam, const Node& n, const std::string& key, const std::string& where) {
  if (is_null(n) && find(key) == nullptr && is_section(key)) return;
  if (!n.is_map || find(key) != nullptr) return put(L, cam, key, n, where, "user");
  for (const auto& [k, v] : n.map) walk(L, cam, v, key + "." + k, where + "." + k);
}

void check_id(const std::string& id, const std::string& where) {
  const auto word = [](std::string_view s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) {
      return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
    });
  };
  const size_t slash = id.find('/'), at = id.rfind("@v");
  const bool ok = slash != std::string::npos && at != std::string::npos && slash < at && word(id.substr(0, slash)) &&
                  word(id.substr(slash + 1, at - slash - 1)) && at + 2 < id.size() &&
                  std::all_of(id.begin() + static_cast<std::ptrdiff_t>(at + 2), id.end(),
                              [](char c) { return c >= '0' && c <= '9'; });
  if (!ok) fail(DiagKind::kProfile, where + ": profile id '" + id + "' is not <group>/<name>@v<N>");
}

std::vector<std::string> id_list(const Node& n, const std::string& where) {
  if (is_null(n)) return {};
  if (const auto* d = n.is_map ? nullptr : std::get_if<std::vector<double>>(&n.leaf); d && d->empty()) return {};
  const auto* l = n.is_map ? nullptr : std::get_if<std::vector<std::string>>(&n.leaf);
  if (l == nullptr) fail(DiagKind::kProfile, where + ": expected a list of profile ids");
  for (const std::string& id : *l) check_id(id, where);
  return *l;
}

// The parsed layers of a run: root profiles, camera profiles by id, the configuration and the CLI.
struct Parsed {
  std::vector<std::string> root_ids;                                       // in order
  std::map<std::string, std::vector<std::string>, std::less<>> block_ids;  // camera block -> its profiles
  std::map<std::string, Layer, std::less<>> profiles;                      // by id
  Layer config, cli;
};

Layer parse_run(const Inputs& in, Parsed& P) {
  Layer L{"config " + in.name, "config", true, {}, {}, in.name};
  const Node& doc = in.config;
  if (!doc.is_map) fail(DiagKind::kFormat, L.label + ": a run configuration is a mapping", "", "", L.record);
  const Node* format = child(doc, "format");
  if (format == nullptr) {
    fail(DiagKind::kFormat,
         in.name +
             ": a format-1 run configuration (no format: key); convert it with python tools/config/convert_v1.py " +
             in.name + " (format 2 starts with 'format: ballasim.run/2'; the key map is in docs/dev/P0_PARAMS.md §3.1)",
         "", "", L.record);
  }
  if (text(*format, L.label + ": format") != "ballasim.run/2") {
    fail(DiagKind::kFormat, L.label + ": unknown format '" + text(*format, "format") + "' (expected ballasim.run/2)",
         "", "", L.record);
  }
  static const std::set<std::string, std::less<>> kSections{"scene", "run", "accuracy", "execution", "output", "dataset"};
  for (const auto& [k, v] : doc.map) {
    if (k == "format") continue;
    if (k == "profiles") {
      P.root_ids = id_list(v, L.label + ": profiles");
    } else if (k == "cameras") {
      if (is_null(v)) continue;
      if (!v.is_map) {
        fail(DiagKind::kFormat, L.label + ": cameras: expected a mapping of camera blocks", "", "", L.record);
      }
      for (const auto& [cam, block] : v.map) {
        const std::string where = "cameras." + cam;
        if (is_null(block)) {  // an empty block
          L.cams[cam];
          continue;
        }
        if (!block.is_map) {
          fail(DiagKind::kFormat,
               L.label + ": " + where + ": expected a mapping (profiles, sensor, photometry, optics, imu, frames)", "",
               cam, L.record);
        }
        L.cams[cam];  // an empty block still names a camera
        for (const auto& [bk, bv] : block.map) {
          if (bk == "profiles") P.block_ids[cam] = id_list(bv, L.label + ": " + where + ".profiles");
          else walk(L, cam, bv, bk, where + "." + bk);
        }
      }
    } else if (kSections.count(k) != 0) {
      walk(L, "", v, k, k);
    } else if (k == "sensor" || k == "photometry" || k == "optics" || k == "imu" || k == "frames") {
      fail(DiagKind::kWrongScope, L.label + ": " + k + ": camera sections go under cameras.default or cameras.<camera>",
           k, "", L.record);
    } else {
      fail(DiagKind::kUnknownKey,
           L.label + ": unknown key " + k + " (sections: format, profiles, scene, run, cameras, accuracy, execution, "
                                            "output, dataset)" + format1_hint(k),
           k, "", L.record);
    }
  }
  return L;
}

// A full dotted key of the root (cameras.<block>.<key> or a run key) into the layer.
void put_full(Layer& L, const std::string& full, const Node& v, const std::string& status,
              const std::string& source = {}) {
  if (full.rfind("cameras.", 0) == 0) {
    const size_t dot = full.find('.', 8);
    if (dot == std::string::npos) {
      fail(DiagKind::kUnknownKey, L.label + ": " + full + ": expected cameras.<camera>.<key>", full, "", L.record);
    }
    const std::string cam = full.substr(8, dot - 8), key = full.substr(dot + 1);
    if (key == "profiles") {
      fail(DiagKind::kProfile, L.label + ": " + full + ": camera profiles are listed in the run configuration", "", cam,
           L.record);
    }
    L.cams[cam];
    return put(L, cam, key, v, full, status, source);
  }
  put(L, "", full, v, full, status, source);
}

Layer parse_cli(const Node& cli) {
  Layer L{"cli", "cli", true, {}, {}, "--set"};
  if (is_null(cli)) return L;
  if (!cli.is_map) {
    fail(DiagKind::kFormat, L.label + ": the --set items are a mapping of full dotted keys to values", "", "",
         L.record);
  }
  for (const auto& [k, v] : cli.map) {
    if (k == "format" || k == "profiles") {
      fail(DiagKind::kUnknownKey, "cli: --set " + k + ": not a key (profiles: --profile ID)", k, "", L.record);
    }
    put_full(L, k, v, "user");
  }
  return L;
}

Layer parse_profile(const std::string& id, const Inputs& in, bool camera_mount) {
  Layer L{"profile " + id, "profile:" + id, false, {}, {}, "profile " + id};
  const DiagKind kP = DiagKind::kProfile;
  const auto it = in.profiles.find(id);
  if (it == in.profiles.end()) fail(kP, L.label + ": not loaded", "", "", L.record);
  const Node& doc = it->second;
  if (!doc.is_map) fail(kP, L.label + ": a profile is a mapping", "", "", L.record);
  const auto need = [&](const char* k) -> const Node& {
    const Node* n = child(doc, k);
    if (n == nullptr) fail(kP, L.label + ": " + k + " is required", "", "", L.record);
    return *n;
  };
  if (text(need("format"), L.label + ": format", kP) != "ballasim.profile/1") {
    fail(kP, L.label + ": unknown format (expected ballasim.profile/1)", "", "", L.record);
  }
  if (text(need("id"), L.label + ": id", kP) != id) {
    fail(kP, L.label + ": the file's id is " + text(need("id"), "id", kP), "", "", L.record);
  }
  const std::string mount = text(need("mount"), L.label + ": mount", kP);
  if (mount != "camera" && mount != "root") fail(kP, L.label + ": mount must be camera or root", "", "", L.record);
  if ((mount == "camera") != camera_mount) {
    fail(kP,
         L.label + ": a " + mount + "-mounted profile listed " +
             (camera_mount ? "in a camera block's profiles" : "in the root profiles"),
         "", "", L.record);
  }
  text(need("description"), L.label + ": description", kP);
  for (const auto& [k, v] : doc.map) {
    if (k == "profiles") fail(kP, L.label + ": profiles are flat; a profile cannot name profiles", "", "", L.record);
    if (k != "format" && k != "id" && k != "mount" && k != "description" && k != "valid_for" && k != "values" &&
        k != "notes") {
      fail(kP, L.label + ": unknown key " + k + " (format, id, mount, description, valid_for, values, notes)", "", "",
           L.record);
    }
  }
  const Node& values = need("values");
  if (!values.is_map) fail(kP, L.label + ": values: expected a mapping of keys to entries", "", "", L.record);
  for (const auto& [key, entry] : values.map) {
    const std::string where = L.label + ": " + key;
    if (!entry.is_map) {
      fail(kP, where + ": an entry is {value, status, source[, data, known_bias, unit]}", "", "", L.record);
    }
    for (const auto& [ek, ev] : entry.map) {
      if (ek != "value" && ek != "status" && ek != "source" && ek != "data" && ek != "known_bias" && ek != "unit") {
        fail(kP, where + ": unknown entry field " + ek, "", "", L.record);
      }
      if (ek != "value") text(ev, where + "." + ek, kP);
    }
    const Node* value = child(entry, "value");
    const Node* status = child(entry, "status");
    const Node* source = child(entry, "source");
    if (value == nullptr || status == nullptr || source == nullptr) {
      fail(kP, where + ": value, status and source are required", "", "", L.record);
    }
    const std::string st = text(*status, where + ".status", kP);
    static const std::set<std::string, std::less<>> kStatus{"prior",   "measured", "fitted", "placeholder",
                                                             "derived", "data",     "choice"};
    if (kStatus.count(st) == 0) {
      fail(kP, where + ".status = '" + st + "': one of prior, measured, fitted, placeholder, derived, data, choice", "",
           "", L.record);
    }
    const std::string src = text(*source, where + ".source", kP);
    if (camera_mount) {
      put(L, "default", key, *value, key, st, src);
    } else {
      put_full(L, key, *value, st, src);
    }
    if (const Node* unit = child(entry, "unit")) {
      const Param* p = find(key.rfind("cameras.", 0) == 0 ? key.substr(key.find('.', 8) + 1) : key);
      if (text(*unit, where + ".unit", kP) != p->unit) {
        fail(kP, where + ".unit = " + text(*unit, "unit", kP) + ": the key's unit is " + p->unit, p->key, "",
             L.record);
      }
    }
  }
  return L;
}

Parsed parse(const Inputs& in, bool with_profiles) {
  Parsed P;
  P.config = parse_run(in, P);
  P.cli = parse_cli(in.cli);
  for (const std::string& id : in.cli_profiles) {
    check_id(id, "cli: --profile");
    P.root_ids.push_back(id);
  }
  if (!with_profiles) return P;
  for (const std::string& id : P.root_ids) {
    if (P.profiles.count(id) == 0) P.profiles.emplace(id, parse_profile(id, in, false));
  }
  for (const auto& [block, ids] : P.block_ids) {
    for (const std::string& id : ids) {
      if (std::find(P.root_ids.begin(), P.root_ids.end(), id) != P.root_ids.end()) {
        fail(DiagKind::kProfile, "profile " + id + ": a root-mounted profile listed in a camera block's profiles", "",
             block, "profile:" + id);
      }
      if (P.profiles.count(id) == 0) P.profiles.emplace(id, parse_profile(id, in, true));
    }
  }
  return P;
}

// The layers that hold camera `cam`'s values (empty: run values), lowest first: root profiles, the camera's profiles
// (its block's list, else the default block's: lists replace), the configuration, the CLI.
std::vector<const Layer*> chain(const Parsed& P, const std::string& cam) {
  std::vector<const Layer*> out;
  for (const std::string& id : P.root_ids) out.push_back(&P.profiles.at(id));
  if (!cam.empty()) {
    auto it = P.block_ids.find(cam);
    if (it == P.block_ids.end()) it = P.block_ids.find("default");
    if (it != P.block_ids.end()) {
      for (const std::string& id : it->second) out.push_back(&P.profiles.at(id));
    }
  }
  out.push_back(&P.config);
  out.push_back(&P.cli);
  return out;
}

struct Found {
  const Set* set = nullptr;
  const Layer* layer = nullptr;
};

// The value of the last layer that sets the key; within a layer the camera's block beats the default block (D10).
Found lookup(const std::vector<const Layer*>& layers, const std::string& cam, std::string_view key) {
  Found f;
  for (const Layer* L : layers) {
    const Set* s = nullptr;
    if (cam.empty()) {
      if (const auto it = L->run.find(key); it != L->run.end()) s = &it->second;
    } else {
      for (const std::string& block : {cam, std::string("default")}) {
        const auto b = L->cams.find(block);
        if (b == L->cams.end()) continue;
        if (const auto it = b->second.find(key); it != b->second.end()) {
          s = &it->second;
          break;
        }
      }
    }
    if (s != nullptr) f = {s, L};
  }
  return f;
}

// ---------------------------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------------------------

// The resolved values of one scope (run, or one camera), with the layer that set each explicit one.
struct Table {
  std::map<std::string, Entry, std::less<>> e;
  std::map<std::string, Found, std::less<>> expl;  // keys set by a profile, the configuration or the CLI

  const Value* get(std::string_view k) const {
    const auto it = e.find(k);
    return it == e.end() ? nullptr : &it->second.value;
  }
  double num(std::string_view k) const {
    const Value& v = *get(k);
    return std::holds_alternative<int64_t>(v) ? static_cast<double>(std::get<int64_t>(v)) : std::get<double>(v);
  }
  void set(const std::string& k, const Found& f) {
    e[k] = {f.set->value, f.layer->record, f.layer->user ? "user" : f.set->status, ""};
    expl[k] = f;
  }
  // The layer of a resolved key, as run.json records it ("" without a value).
  std::string layer(std::string_view k) const {
    const auto it = e.find(k);
    return it == e.end() ? std::string() : it->second.layer;
  }
};

Entry default_entry(const Param& p) { return {p.value, "default", p.status ? name(*p.status) : "", ""}; }

// Whether a condition's value admits v: equality, or for a string list on a string key, one of its strings.
bool admits(const Value& want, const Value& v) {
  if (const auto* any = std::get_if<std::vector<std::string>>(&want); any && std::holds_alternative<std::string>(v)) {
    for (const std::string& s : *any) {
      if (s == std::get<std::string>(v)) return true;
    }
    return false;
  }
  return v == want;
}

// The first relevance condition of a row that fails (its key has another value), or nullptr: the row matters (also
// when every `also_when` condition holds). A condition on a key without a value holds (§1.1 `when`; e.g. the bundle's
// light mode before the bundle is loaded).
template <class Get>
const When* failing(const Param& p, const Get& get) {
  const auto first = [&](const std::vector<When>& ws) -> const When* {
    for (const When& w : ws) {
      const Value* v = get(w.key);
      if (v != nullptr && !admits(w.value, *v)) return &w;
    }
    return nullptr;
  };
  const When* w = first(p.when);
  return w != nullptr && !p.also_when.empty() && first(p.also_when) == nullptr ? nullptr : w;
}

// The explicit run values (every run, scene and dataset key a layer sets).
Table explicit_run(const Parsed& P) {
  Table t;
  const auto layers = chain(P, "");
  for (const Param& p : schema()) {
    if (p.scope == Scope::kCamera) continue;
    if (const Found f = lookup(layers, "", p.key); f.set != nullptr) t.set(p.key, f);
  }
  return t;
}

// A required key without a value (blender.md §3.11): throws `d` (kRequired or kFamilyMissing, with resolve's message,
// unchanged), or in collect mode records {key, hint, d} once and returns.
void required(Missing* collect, const std::string& key, const std::string& hint, Diagnostic d) {
  if (collect == nullptr) fail(std::move(d));
  for (const MissingKey& m : *collect) {
    if (m.key == key) return;
  }
  const size_t start = hint.find_first_not_of(" ;");
  collect->push_back({key, start == std::string::npos ? std::string() : hint.substr(start), std::move(d)});
}

// The diagnostic of a required key without a value (`key` the schema key, `camera` its camera or empty).
Diagnostic missing(const std::string& message, const std::string& key, const std::string& camera = {},
                   std::vector<std::string> keys = {}, DiagKind kind = DiagKind::kRequired) {
  return diag(kind, Severity::kError, message, key, camera, "", std::move(keys));
}

// What the photometry rules need of the scene source: an EVIMO2 sequence (its lab lights carry no units), a Blender
// exchange the run converts (not a supplied bundle: then its scene keys are inert) and whether its
// scene.blender.lm_per_w is unset.
struct SourceFacts {
  bool evimo2 = false, blender = false, blender_lm_unset = false;
};

std::string required_hint(const Param& p) {
  if (p.key == "photometry.illuminance_scale") {
    return " (relative photometry has no neutral scale); EVIMO2 replays: add profile evimo2/lab_lighting@v0";
  }
  if (p.key.rfind("scene.evimo2.", 0) == 0) {
    return std::string(!p.when.empty() ? " while scene.evimo2.lights is on" : " (baked into the bundle's materials)") +
           "; EVIMO2 replays: add profile evimo2/lab_lighting@v0";
  }
  return "";
}

// A camera's photocurrent scale photometry.illuminance_scale under its photometry.mode (D1, D13 item 2; SENSOR.md
// "Photometry"): relative, the explicit value; physical, the gain of the lens and the pixel on axis (DESIGN §7.1, the
// engine applies the cos^4 map), G = pi T QE kappa pitch^2 q / (4 N^2) in fA per unit of renderer radiance (cd/m2 with
// the scene's lights in cd, lx and cd/m2); reference, a placeholder 1 that the engine replaces by the solved scale at
// start (run.json cameras[].illuminance_scale). Warnings: a profile's scale is inert under physical and reference (a
// user's warns in soft_and_inert); physical on an EVIMO2 source (its lab lights carry no units) and RGB weights that do
// not sum to 1 (the renderer's value is a luminance only then: a white surface's band albedo is 1).
std::vector<double> numbers_of(const Value& v);

// The keys a photometry mode requires: reference, the reference current; physical on a Blender exchange,
// scene.blender.lm_per_w (blender.md §3.11: Blender watts have no luminous efficacy of their own). The camera stage
// reads the scene value from the run's layers, set before any camera resolves.
void photometry_required(const Table& t, const BundleCamera& bc, const SourceFacts& src, Missing* collect) {
  const std::string mode = std::get<std::string>(*t.get("photometry.mode"));
  const std::string where = "cameras." + bc.name + ".";
  if (mode == "reference" && std::holds_alternative<std::monostate>(*t.get("photometry.reference_current_fA"))) {
    required(collect, where + "photometry.reference_current_fA", "by photometry.mode = reference",
             missing(where + "photometry.reference_current_fA is required by photometry.mode = reference",
                     "photometry.reference_current_fA", bc.name, {"photometry.mode"}));
  }
  if (mode == "physical" && src.blender && src.blender_lm_unset) {
    const std::string hint = "by photometry.mode = physical (" + where.substr(0, where.size() - 1) + ")";
    required(collect, "scene.blender.lm_per_w", hint,
             missing("scene.blender.lm_per_w is required " + hint, "scene.blender.lm_per_w", "", {"photometry.mode"}));
  }
}

void photometry(Table& t, const BundleCamera& bc, const SourceFacts& src, Warnings& warn, Missing* collect) {
  const std::string mode = std::get<std::string>(*t.get("photometry.mode"));
  const std::string where = "cameras." + bc.name + ".";
  if (const auto it = t.expl.find("photometry.illuminance_scale");
      mode != "relative" && it != t.expl.end() && !it->second.layer->user) {
    warn(DiagKind::kInert,
         where + "photometry.illuminance_scale = " + format_value(it->second.set->value) + " (" +
             it->second.layer->label + ") is inert: photometry.mode = " + mode +
             (mode == "physical" ? " derives it" : " solves it at start"),
         "photometry.illuminance_scale", bc.name, it->second.layer->record, {"photometry.mode"});
  }
  photometry_required(t, bc, src, collect);
  const bool evimo2 = src.evimo2;
  if (mode == "reference") {
    t.e["photometry.illuminance_scale"] = {1.0, "derived", "derived", "reference photometry: solved at start"};
  } else if (mode == "physical") {
    if (!(bc.pixel_pitch_m > 0)) {
      fail(DiagKind::kCrossKey, where + "photometry.mode = physical: the bundle camera has no pixel_pitch_m",
           "photometry.mode", bc.name, t.layer("photometry.mode"));
    }
    const double n = t.num("optics.f_number"), pitch_um = bc.pixel_pitch_m * 1e6;
    const double g = kPi * t.num("optics.transmittance") / (4.0 * n * n) * t.num("sensor.qe") *
                     t.num("sensor.kappa") * (pitch_um * pitch_um) * (kElementaryCharge * 1e15);
    t.e["photometry.illuminance_scale"] = {g, "derived", "derived",
                                           "physical photometry: pi T QE kappa pitch^2 q / (4 N^2)"};
    if (evimo2) {
      warn(DiagKind::kCrossKey,
           where + "photometry.mode = physical on an EVIMO2 source, whose lab lights carry no units (cd, lx, cd/m2) "
                   "until their refit: the photocurrent is not physical; photometry.mode = reference states it",
           "photometry.mode", bc.name, t.layer("photometry.mode"));
    }
    double sum = 0;
    for (const double w : numbers_of(*t.get("sensor.rgb_weights"))) sum += w;
    if (std::abs(sum - 1.0) > 1e-6) {
      warn(DiagKind::kCrossKey,
           where + "sensor.rgb_weights sum to " + format_double(sum) + ": under photometry.mode = physical every "
                   "radiance scales by it (the lights' units assume a sum of 1)",
           "sensor.rgb_weights", bc.name, t.layer("sensor.rgb_weights"), {"photometry.mode"});
    }
  }
  // P2-F F2b: the lens's relative illumination, [] or [c2, c4] with G > 0 over the image (relative_illumination.h).
  const Value& ri = *t.get("optics.relative_illumination");
  if (const char* why = camera::relative_illumination_error(numbers_of(ri))) {
    fail(DiagKind::kHardRange, where + "optics.relative_illumination = " + format_value(ri) + ": " + why,
         "optics.relative_illumination", bc.name, t.layer("optics.relative_illumination"));
  }
}

// The lens PSF (D13 item 3, F076; RENDER.md "Lens PSF"): its core is optics.blur_sigma_um / the bundle camera's pitch
// (as photometry's) when set, a px value a layer states being then inert (warned), else optics.blur_sigma_px as resolved
// (phase1 and the v0 profiles: 0.15 px); optics.diffraction adds the Airy core's Gaussian equivalent 0.42 lambda N
// (DESIGN §8.1; Zhang, Zerubia & Olivo-Marin 2007, Appl. Opt. 46:1819: 0.21 lambda / NA, NA = 1 / 2N) in quadrature.
// The result is the camera's optics.blur_sigma_px, the renderer's one source.
void lens_blur(Table& t, const BundleCamera& bc, Warnings& warn) {
  const Value& um = *t.get("optics.blur_sigma_um");
  const bool from_um = !std::holds_alternative<std::monostate>(um);
  const bool diffraction = std::get<bool>(*t.get("optics.diffraction"));
  if (!from_um && !diffraction) return;
  const std::string where = "cameras." + bc.name + ".";
  if (!(bc.pixel_pitch_m > 0)) {
    const std::string key = from_um ? "optics.blur_sigma_um" : "optics.diffraction";
    fail(DiagKind::kCrossKey, where + key + ": the bundle camera has no pixel_pitch_m", key, bc.name, t.layer(key));
  }
  if (const auto it = t.expl.find("optics.blur_sigma_px"); from_um && it != t.expl.end()) {
    warn(DiagKind::kInert,
         where + "optics.blur_sigma_px = " + format_value(it->second.set->value) + " (" + it->second.layer->label +
             ") is inert: optics.blur_sigma_um = " + format_value(um) + " derives it",
         "optics.blur_sigma_px", bc.name, it->second.layer->record, {"optics.blur_sigma_um"});
  }
  const double pitch_um = bc.pixel_pitch_m * 1e6;
  Entry& e = t.e["optics.blur_sigma_px"];
  if (from_um) e = {t.num("optics.blur_sigma_um") / pitch_um, "derived", "derived",
                    "optics.blur_sigma_um / the bundle camera's pixel pitch"};
  if (diffraction) {  // the core's layer stays (a flagged default stays flagged)
    const double d = kAiryGaussian * t.num("optics.wavelength_nm") * 1e-3 * t.num("optics.f_number") / pitch_um;
    e.value = std::hypot(t.num("optics.blur_sigma_px"), d);
    e.status = "derived";
    e.note = (from_um ? e.note : std::string("optics.blur_sigma_px")) +
             " and the diffraction sigma 0.42 lambda N / pitch (" + fmt3(d) + " px) in quadrature";
  }
}

// The scene keys (resolve_scene and resolve): scene.evimo2.* only with an EVIMO2 source, scene.blender.* only with a
// Blender exchange (Inputs::source_kind; blender.md §3.11); unless the bundle is supplied, the lighting keys are
// required while they matter (the lights on; resolve also knows the bundle's light mode, `low_light`), and the albedos
// always.
void fill_scene(Table& run, const Inputs& in, bool supplied, std::optional<bool> low_light, Warnings& warn,
                Missing* collect = nullptr) {
  if (const auto it = run.expl.find("scene.trajectory.order");
      it != run.expl.end() && std::get<int64_t>(it->second.set->value) == 5) {  // the row's hard [4, 6] admits 5
    fail(DiagKind::kHardRange,
         it->second.layer->label + ": scene.trajectory.order = 5: the spline order is 4 or 6 (C1, DESIGN §5.3)",
         "scene.trajectory.order", "", it->second.layer->record);
  }
  const bool evimo2 = run.get("scene.evimo2.sequence") != nullptr;
  if (evimo2 && run.get("scene.path") != nullptr) {
    fail(DiagKind::kCrossKey, "scene.path and scene.evimo2.sequence are both set: one scene source per run",
         "scene.path", "", run.layer("scene.path"), {"scene.evimo2.sequence"});
  }
  if (const auto it = run.expl.find("scene.evimo2.low_light"); it != run.expl.end()) {
    fail(DiagKind::kWrongScope,
         it->second.layer->label + ": scene.evimo2.low_light is a fact of the sequence (its bundle's light mode), not "
                                   "a setting",
         "scene.evimo2.low_light", "", it->second.layer->record);
  }
  const bool blender = in.source_kind == "blender";
  const auto blender_key = [](std::string_view k) { return k.rfind("scene.blender.", 0) == 0; };
  for (const Param& p : schema()) {
    if (p.scope != Scope::kScene || run.get(p.key) != nullptr) continue;
    if (p.key.rfind("scene.evimo2.", 0) == 0 && !evimo2) continue;
    if (blender_key(p.key) && !blender) continue;
    if (p.def == Default::kLiteral) {
      run.e[p.key] = default_entry(p);
    } else if (p.def == Default::kDerived) {
      if (const auto it = in.facts.find(p.key); it != in.facts.end()) run.e[p.key] = {it->second, "derived", "derived", ""};
    }
  }
  // Scene values the configuration or the CLI sets in vain: every one when the bundle is supplied; without an EVIMO2
  // source the EVIMO2 keys and the trajectory keys (only the EVIMO2 adapter fits trajectories); with one the
  // scene.path sources' background keys (D13 item 1: an EVIMO2 source has scene.evimo2.background_albedo), which
  // then leave the values (no digest hashes them); without a Blender exchange the scene.blender.* keys, which leave
  // them too.
  const auto background = [](std::string_view k) { return k.rfind("scene.background_", 0) == 0; };
  for (const auto& [k, f] : run.expl) {
    if (!f.layer->user || k.rfind("scene.", 0) != 0) continue;
    const std::string layer = f.layer->record;
    if (supplied) {
      warn(DiagKind::kInert, k + " is set but inert: the run's bundle is supplied", k, "", layer);
    } else if (blender_key(k)) {
      if (!blender) warn(DiagKind::kInert, k + " is set but inert: the scene is not a Blender exchange", k, "", layer);
    } else if (evimo2 && background(k)) {
      warn(DiagKind::kInert, k + " is set but inert: an EVIMO2 sequence takes scene.evimo2.background_albedo", k, "",
           layer, {"scene.evimo2.background_albedo"});
    } else if (!evimo2 && !background(k) && k.rfind("scene.path", 0) != 0) {
      warn(DiagKind::kInert,
           k + " is set but inert: the scene is not an EVIMO2 sequence" +
               (k.rfind("scene.trajectory.", 0) == 0 ? " (only the EVIMO2 adapter fits trajectories)" : ""),
           k, "", layer);
    }
  }
  for (auto it = run.e.begin(); it != run.e.end();) {
    const bool unused = (evimo2 ? background(it->first) : it->first.rfind("scene.evimo2.", 0) == 0) ||
                        (blender_key(it->first) && !blender);
    it = unused ? run.e.erase(it) : std::next(it);
  }
  if (!evimo2) return;
  if (low_light) run.e["scene.evimo2.low_light"] = {*low_light, "bundle", "derived", ""};
  const bool lit = std::get<bool>(*run.get("scene.evimo2.lights"));
  const auto get = [&](std::string_view k) { return run.get(k); };
  for (const Param& p : schema()) {
    if (p.scope != Scope::kScene || p.def != Default::kRequired || run.get(p.key) != nullptr) continue;
    if (p.key == "scene.path" || p.key == "scene.evimo2.sequence") continue;
    if (!supplied && failing(p, get) == nullptr) {
      std::vector<std::string> keys;
      for (const When& w : p.when) keys.push_back(w.key);
      required(collect, p.key, required_hint(p), missing(p.key + " is required" + required_hint(p), p.key, "", keys));
    }
  }
  // The light groups relight builds, before any bundle is built (P0b2): the mains waveform's own checks (equal
  // lengths, at most 8 harmonics, sum |a_k| <= 1), and the strobe's rise within its width at half height (the C1
  // pulse's plateau >= 0). The strobe's other check, width plus rise within the period, needs the sequence's Vicon
  // period: relight's (the bundle is cached first).
  if (lit && !supplied) {
    const auto has = [&](std::initializer_list<const char*> keys) {
      return std::all_of(keys.begin(), keys.end(), [&](const char* k) { return run.get(k) != nullptr; });
    };
    if (has({"scene.evimo2.mains_f0_hz", "scene.evimo2.mains_a", "scene.evimo2.mains_phi_rad"})) {
      try {
        sensor::make_fourier_waveform(run.num("scene.evimo2.mains_f0_hz"),
                                      std::get<std::vector<double>>(*run.get("scene.evimo2.mains_a")),
                                      std::get<std::vector<double>>(*run.get("scene.evimo2.mains_phi_rad")));
      } catch (const std::invalid_argument& e) {
        fail(DiagKind::kCrossKey,
             std::string("scene.evimo2.mains_a and mains_phi_rad (the mains light group): ") + e.what(),
             "scene.evimo2.mains_a", "", run.layer("scene.evimo2.mains_a"),
             {"scene.evimo2.mains_phi_rad", "scene.evimo2.mains_f0_hz"});
      }
    }
    if (has({"scene.evimo2.strobe_rise_s", "scene.evimo2.strobe_width_s"}) &&
        !(run.num("scene.evimo2.strobe_rise_s") <= run.num("scene.evimo2.strobe_width_s"))) {
      fail(DiagKind::kCrossKey,
           "scene.evimo2.strobe_rise_s = " + format_value(*run.get("scene.evimo2.strobe_rise_s")) +
               " is above scene.evimo2.strobe_width_s = " + format_value(*run.get("scene.evimo2.strobe_width_s")) +
               " (the strobe light group): a pulse rises within its width at half height",
           "scene.evimo2.strobe_rise_s", "", run.layer("scene.evimo2.strobe_rise_s"), {"scene.evimo2.strobe_width_s"});
    }
  }
}

// What fidelity derives from the run's cameras (§1.3).
struct Fidelity {
  double theta_d = 0, n_step = 0, eps = kEpsRef;
  double sigma_min = std::numeric_limits<double>::infinity();  // sigma_v,min (infinite without noise)
  int64_t n = 16, min_dt = kMinDtFloorNs, q_min = 0;
};

int64_t pow2ceil(int64_t x) {
  int64_t p = 1;
  while (p < x) p <<= 1;
  return p;
}

// The camera's SensorParams from its camera-scope rows (the run-scope rows, some derived from these, stay unset), so
// that the sensor's own predicates (noise_on(), hot_pixels()) decide rather than copies of their rules.
sensor::SensorParams camera_sensor(const Table& cam) {
  sensor::SensorParams sp;
  for (const Param& r : sensor::param_rows()) {
    if (r.scope == Scope::kCamera) r.set(r, &sp, *cam.get(r.key));
  }
  return sp;
}

// PHASE2 F085 (P2-B B4a): the run's smallest likely pixel threshold, max(0.10, min over its cameras and polarities of
// theta - 3 sigma_theta): the frozen mismatch's 3-sigma tail (M3 A.19), hot pixels (the tails beyond it) excluded,
// floored at 0.10 log-e. fidelity's and balanced's design threshold, and the band's theta_band (logged; B5 reads it).
double theta_band(const std::vector<Table>& cams) {
  double theta = std::numeric_limits<double>::infinity();
  for (const Table& t : cams) {
    theta = std::min({theta, t.num("sensor.theta_on") - 3.0 * t.num("sensor.sigma_theta_on"),
                      t.num("sensor.theta_off") - 3.0 * t.num("sensor.sigma_theta_off")});
  }
  return std::max(kThetaBandFloor, theta);
}

bool ideal_reset(const std::vector<Table>& cams) {
  for (const Table& t : cams) {
    if (std::get<std::string>(*t.get("sensor.reset_mode")) == "ideal") return true;
  }
  return false;
}

Fidelity fidelity(const Table& run, const std::vector<Table>& cams) {
  Fidelity f;
  f.theta_d = run.num("accuracy.design_threshold");
  const double c = run.num("accuracy.design_contrast");
  f.n_step = std::expm1(c) / std::expm1(f.theta_d / 4);
  while (static_cast<double>(f.n) < f.n_step && f.n < kMaxFidelityN) f.n *= 4;
  double sigma_min = std::numeric_limits<double>::infinity();
  for (const Table& t : cams) {
    if (!camera_sensor(t).noise_on()) continue;
    sensor::NoiseModel m{};
    m.kq = t.num("sensor.k") * kElementaryCharge;
    m.a2 = 1.0 / t.num("sensor.tau_o");
    m.a3 = 1.0 / t.num("sensor.tau_sf");
    m.sigma_o = t.num("sensor.sigma_o");
    m.levels = static_cast<int>(run.num("accuracy.noise_leaf_level"));
    m.margin_z = run.num("accuracy.noise_margin_z");
    double p[3][3];
    sensor::noise_stationary_cov(m, kNoiseTopRate, p);
    sigma_min = std::min(sigma_min, std::sqrt(p[2][2]));
  }
  f.sigma_min = sigma_min;
  f.eps = kEpsRef * (std::isfinite(sigma_min) ? std::min(1.0, sigma_min / kSigmaRef) : 1.0);
  for (const Table& t : cams) {
    const int64_t q = std::llround(t.num("sensor.readout_quantum_s") * 1e9);
    if (q > 0 && (f.q_min == 0 || q < f.q_min)) f.q_min = q;
  }
  if (f.q_min > 0) f.min_dt = std::max(kMinDtFloorNs, pow2ceil(f.q_min));
  return f;
}

// Whether a camera takes sensor keys from an EVIMO2 profile fitted under the frozen noise diffusion (PHASE2 D-P2-2):
// v0 (the P1-M5 calibration) and v1 (round 12's minimal refit). That rule then stays theirs under every accuracy
// profile; a profile fitted later states nothing here and takes the profile's rule.
bool frozen_calibration(const Table& t) {
  const std::string_view prefix = "profile:evimo2/";
  for (const auto& [key, e] : t.e) {
    if (key.rfind("sensor.", 0) != 0 || e.layer.size() <= prefix.size() + 3 ||
        e.layer.compare(0, prefix.size(), prefix) != 0) {
      continue;
    }
    for (const std::string_view suffix : {"@v0", "@v1"}) {
      if (e.layer.compare(e.layer.size() - suffix.size(), suffix.size(), suffix) == 0) return true;
    }
  }
  return false;
}

// Whether any camera of the run does (the run-wide rules).
bool frozen_calibration(const std::vector<Table>& cams) {
  return std::any_of(cams.begin(), cams.end(), [](const Table& t) { return frozen_calibration(t); });
}

// P2-F F13-min (PHASE2 F079; p2f.md §4.2, §5): optics.vignetting_law, a derived camera key (Default::kDerived: the
// camera loop writes a cos4 placeholder; the cameras are resolved before fill_run decides the profile, so this pass
// follows it). fidelity and balanced derive jacobian, cos(theta) dOmega/dA of the camera model (cos^4 / det J_d for
// radtan: up to 15 % off cos^4 at the EVIMO2 corners), where the gain map is on (the row's relevance, so a run without
// a map keeps its digests) and the camera does not take its sensor keys from an EVIMO2 v0 or v1 profile (frozen: those
// runs keep Phase 1's cos^4 map bit for bit); phase1 and reference list cos4. An explicit value, from a profile (a v2
// states none), the configuration or the CLI, beats the derivation. Cameras of one rig are decided separately. Every
// camera that states nothing gets a layer-"derived" entry whose note names the reason, so config::explain (it shows a
// derived entry) and run.json say why the camera runs the law it runs.
void derive_vignetting_law(const Table& run, std::vector<Table>& cams) {
  const std::string& profile = std::get<std::string>(*run.get("accuracy.profile"));
  const bool tracks = profile == "fidelity" || profile == "balanced";
  const Param& row = *find("optics.vignetting_law");
  for (Table& t : cams) {
    if (t.expl.count(row.key) != 0) continue;
    std::string law = "cos4", note;
    if (!tracks) {
      note = "accuracy.profile = " + profile + " lists Phase 1's cos^4 law";
    } else if (frozen_calibration(t)) {
      note = "an EVIMO2 v0 or v1 sensor profile (frozen): Phase 1's cos^4 map";
    } else if (failing(row, [&](std::string_view k) { return t.get(k); }) != nullptr) {
      note = "the gain map is off (optics.vignetting, photometry.mode), so the law is inert: cos4";
    } else {
      law = "jacobian";
      note = "accuracy.profile = " + profile + ": cos(theta) dOmega/dA of the camera model (PHASE2 F079)";
    }
    t.e[row.key] = {std::move(law), "derived", "derived", std::move(note)};
  }
}

// P2-B B6c: the smallest pixel-response reach over the run's cameras (px), render::support_px's formula with the
// truncation at most kPixelBandTruncationMax: 0.5 aperture + min(n, 3) sigma_blur. Pooled escalation's premise, on
// which the per-pixel band rests, needs it at least kPixelBandSupportMin (fidelity keeps the tile map below it; an
// explicit pixel map is refused there).
double band_support(const Table& run, const std::vector<Table>& cams) {
  const double n_sigma = std::min(run.num("accuracy.psf_truncation_sigmas"), kPixelBandTruncationMax);
  double support = std::numeric_limits<double>::infinity();
  for (const Table& t : cams) {
    support = std::min(support, 0.5 * t.num("sensor.aperture_px") + n_sigma * t.num("optics.blur_sigma_px"));
  }
  return support;
}

// derived_from (ui.md §3.10): per accuracy profile, each key fill_run derives with every key its derivation reads,
// closed over the derived keys it reads (a derived key set explicitly replaces its derivation, so it stays an input).
// The derivations below are its source: an edit there edits this table, and T4 (tests/cpp/test_diagnostics.cpp)
// perturbs every row against it. Besides these keys, the derivations read whether a camera takes its sensor keys
// from an EVIMO2 v0 or v1 profile (frozen_calibration) and the bundle cameras' pixel pitch (lens_blur).
using Derivations = std::vector<std::pair<std::string, std::vector<std::string>>>;

Derivations derivation_table(std::string_view profile) {
  using Keys = std::vector<std::string>;
  const auto join = [](std::initializer_list<Keys> parts) {
    Keys out;
    for (const Keys& p : parts) {
      for (const std::string& k : p) {
        if (std::find(out.begin(), out.end(), k) == out.end()) out.push_back(k);
      }
    }
    return out;
  };
  // F085's theta_band (fidelity, balanced) or the smallest mean threshold (phase1, reference): design_threshold.
  const Keys band{"sensor.theta_on", "sensor.theta_off", "sensor.sigma_theta_on", "sensor.sigma_theta_off"};
  const Keys mean{"sensor.theta_on", "sensor.theta_off"};
  // fidelity's N: N_step from the design contrast at the design threshold.
  const Keys n_fid = join({{"accuracy.design_threshold", "accuracy.design_contrast"}, band});
  // sigma_v,min: the cameras with noise (noise_on(): noise and the cascade) and their noise rows (the stationary
  // covariance reads neither the noise tables' levels nor their margin).
  const Keys noise{"sensor.noise",  "sensor.photoreceptor", "sensor.k",
                   "sensor.tau_o",  "sensor.tau_sf",        "sensor.sigma_o"};
  const Keys quantum{"sensor.readout_quantum_s"};
  const Keys warmup{"accuracy.warmup"};
  if (profile == "phase1") return {{"accuracy.design_threshold", mean}};
  if (profile == "reference") {
    return {{"accuracy.design_threshold", mean}, {"accuracy.eps_L", noise},       {"accuracy.refine_tol_log", noise},
            {"accuracy.min_dt_ns", quantum},     {"accuracy.warmup_ns", warmup}, {"accuracy.noise_diffusion", {}}};
  }
  if (profile == "balanced") {
    // Its n_band_max is a literal null; an explicit band N raises a derived N_base below 64 to 64, which leaves d_px
    // at its cap, min(8 / sqrt(N), 1) = 1 for N = 16 and 64.
    const Keys n = join({n_fid, {"accuracy.n_band_max"}});
    return {{"accuracy.design_threshold", band},
            {"accuracy.n_subsamples", n},
            {"accuracy.d_px", join({{"accuracy.n_subsamples"}, n_fid})},
            {"accuracy.eps_L", noise},
            {"accuracy.refine_tol_log", noise},
            {"accuracy.eps_jump", noise},
            {"accuracy.min_dt_ns", quantum},
            {"accuracy.max_samples_per_window", {}},
            {"accuracy.warmup_ns", warmup},
            {"accuracy.noise_diffusion", {}},
            {"accuracy.pixel_integral", {}}};
  }
  // A derived key is not its own input (an explicit value replaces its derivation).
  const auto own = [](Derivations d) {
    for (auto& [key, inputs] : d) inputs.erase(std::remove(inputs.begin(), inputs.end(), key), inputs.end());
    return d;
  };
  if (profile == "fidelity") {
    // The band N: off for an ideal reset, at N_base >= 256, beside an explicit N_base that is not a power of two
    // >= 64 or an explicit phase1 integral; N_base raised to 64 beside it.
    const Keys n_band = join({{"sensor.reset_mode", "accuracy.pixel_integral", "accuracy.n_subsamples"}, n_fid});
    const Keys n = join({n_fid, {"accuracy.n_band_max"}, n_band});
    // The pixel response's reach: 0.5 aperture + min(truncation, 3) sigma_blur, sigma_blur from lens_blur.
    const Keys growth = join({{"accuracy.n_band_max"}, n_band,
                              {"accuracy.psf_truncation_sigmas", "sensor.aperture_px", "optics.blur_sigma_px",
                               "optics.blur_sigma_um", "optics.diffraction", "optics.wavelength_nm",
                               "optics.f_number"}});
    return own({{"accuracy.design_threshold", band},
                {"accuracy.n_band_max", n_band},
                {"accuracy.n_subsamples", n},
                {"accuracy.d_px", join({{"accuracy.n_subsamples"}, n})},
                {"accuracy.eps_L", noise},
                {"accuracy.refine_tol_log", noise},
                {"accuracy.min_dt_ns", quantum},
                {"accuracy.max_samples_per_window", {}},
                {"accuracy.warmup_ns", warmup},
                {"accuracy.noise_diffusion", {}},
                {"accuracy.pixel_integral", {}},
                {"accuracy.eps_jump", noise},
                {"accuracy.band_growth", growth},
                {"accuracy.band_reach", join({{"accuracy.band_growth"}, growth})}});
  }
  fail(DiagKind::kType, "accuracy.profile = '" + std::string(profile) + "': no such profile", "accuracy.profile");
}

// The run keys: the accuracy profile (with fidelity's derivations), the literals, the derived values; then the
// warnings of explicit accuracy values against the derived ones, and the band N's rules and facts (P2-B B4a).
void fill_run(Table& run, const std::vector<Table>& cams, const BundleInfo& b, Warnings& warn,
              std::vector<std::pair<std::string, double>>& facts, Missing* collect) {
  for (const char* k : {"accuracy.profile", "accuracy.design_contrast"}) {
    if (run.get(k) == nullptr) run.e[k] = default_entry(*find(k));
  }
  const std::string profile = std::get<std::string>(*run.get("accuracy.profile"));
  const bool frozen = frozen_calibration(cams);
  if (run.get("accuracy.design_threshold") == nullptr) {
    // F085 (P2-B B4a) where the design keys apply (fidelity, balanced), but not under an EVIMO2 v0 / v1 sensor profile;
    // elsewhere (inert under phase1 and reference) the smallest mean threshold, as before.
    double theta = std::numeric_limits<double>::infinity();
    for (const Table& t : cams) theta = std::min({theta, t.num("sensor.theta_on"), t.num("sensor.theta_off")});
    if ((profile == "fidelity" || profile == "balanced") && !frozen) theta = theta_band(cams);
    run.e["accuracy.design_threshold"] = {theta, "derived", "derived", ""};
  }
  const AccuracyProfile* ap = find_accuracy_profile(profile);
  if (ap == nullptr) {
    fail(DiagKind::kType, "accuracy.profile = '" + profile + "': no such profile", "accuracy.profile", "",
         run.layer("accuracy.profile"));
  }
  const std::string layer = "accuracy:" + profile;
  const auto derive = [&](const char* k, Value v) {
    if (run.get(k) != nullptr) return;
    try {
      check_value(*find(k), v, std::string(k) + " (derived by accuracy.profile " + profile + ")");
    } catch (const std::invalid_argument& e) {  // the derivation's inputs are the keys to change
      std::vector<std::string> keys;
      for (const auto& [dk, inputs] : derivation_table(profile)) {
        if (dk == k) keys = inputs;
      }
      fail(DiagKind::kHardRange, e.what(), k, "", layer, std::move(keys));
    }
    run.e[k] = {std::move(v), layer, "derived", ""};
  };
  if (profile == "phase1") {
    for (const AccuracyEntry& e : ap->entries) {
      if (run.get(e.key) == nullptr) run.e[e.key] = {e.value, layer, "derived", ""};
    }
  } else {
    // The literal entries first (D12; balanced: the ES switches; reference: every key it does not derive, the noise
    // tables' included), so that the derivations below read them and a literal beats a derivation.
    for (const AccuracyEntry& e : ap->entries) {
      if (e.formula.empty()) derive(e.key.c_str(), e.value);
    }
  }
  // Literals the derivations read (noise tables; the PSF truncation of fidelity's band map) before fidelity's keys.
  for (const char* k : {"accuracy.noise_leaf_level", "accuracy.noise_margin_z", "accuracy.psf_truncation_sigmas"}) {
    if (run.get(k) == nullptr) run.e[k] = default_entry(*find(k));
  }
  const Fidelity f = fidelity(run, cams);
  if (profile != "phase1") {
    // fidelity's derivations. balanced (ENGINE_CPP I7): its d_px scaled from fidelity's (with the cap) and the residual
    // gate at fidelity's tolerance. reference (PHASE2 P2-0a): its literal N and d_px, fidelity's tolerances.
    const bool economy = profile == "balanced";
    if (run.get("accuracy.n_subsamples") == nullptr && f.n_step > static_cast<double>(kMaxFidelityN)) {
      warn(DiagKind::kDerivedConflict,
           profile + ": N_step = " + format_double(f.n_step) + " at design threshold " + format_double(f.theta_d) +
               " exceeds 256; accuracy.n_subsamples capped at 256",
           "accuracy.n_subsamples", "", layer, {"accuracy.design_threshold", "accuracy.design_contrast"});
    }
    // PHASE2 D-P2-4 (P2-B B4a): fidelity's band N, G1's interim 256 above N_base; off (null) for an ideal-reset camera
    // (the oracles and the ideal family keep one N), under an EVIMO2 v0 / v1 sensor profile (their runs stay bitwise),
    // at N_base >= 256 and beside an explicit N_base the band cannot take (not a power of two >= 64) or an explicit
    // phase1 integral. With the band, a derived N_base of 16 becomes 64 (T1 needs N_base >= 64).
    const bool explicit_n = run.get("accuracy.n_subsamples") != nullptr;
    const int64_t n_base = explicit_n ? static_cast<int64_t>(run.num("accuracy.n_subsamples")) : f.n;
    if (profile == "fidelity") {
      const bool base_ok = !explicit_n || (n_base >= kBandBaseMin && (n_base & (n_base - 1)) == 0);
      const Value* integral = run.get("accuracy.pixel_integral");  // explicit (fidelity derives it below)
      const bool phase1 = integral != nullptr && std::get<std::string>(*integral) != "fixed64";
      derive("accuracy.n_band_max",
             ideal_reset(cams) || frozen || n_base >= kBandN || !base_ok || phase1 ? Value{} : Value{kBandN});
    }
    const Value* band = run.get("accuracy.n_band_max");
    if (!explicit_n && f.n < kBandBaseMin && band != nullptr && !std::holds_alternative<std::monostate>(*band)) {
      warn(DiagKind::kDerivedConflict,
           profile + ": N_step = " + format_double(f.n_step) + " gives accuracy.n_subsamples = " + std::to_string(f.n) +
               "; the band N (accuracy.n_band_max = " + format_value(*band) + ") takes N_base = " +
               std::to_string(kBandBaseMin),
           "accuracy.n_subsamples", "", layer, {"accuracy.n_band_max"});
      derive("accuracy.n_subsamples", kBandBaseMin);
    }
    derive("accuracy.n_subsamples", f.n);
    const double d_fid = 2.0 / std::sqrt(run.num("accuracy.n_subsamples"));
    derive("accuracy.d_px", economy ? std::min(kEconomyFactor * d_fid, kEconomyDpxCap) : d_fid);
    derive("accuracy.eps_L", f.eps);
    derive("accuracy.refine_tol_log", f.eps);
    if (economy) derive("accuracy.eps_jump", f.eps);
    derive("accuracy.min_dt_ns", f.min_dt);
    derive("accuracy.max_samples_per_window", Value{});
    const bool motion = std::get<std::string>(*run.get("accuracy.warmup")) == "motion";
    derive("accuracy.warmup_ns", (motion ? int64_t{2} : int64_t{6}) * time::kWindowNs);
    derive("accuracy.noise_diffusion", std::string(frozen_calibration(cams) ? "frozen" : "transient"));
    // PHASE2 D-P2-3 (P2-B B3): the fixed 64-position mean, but not for the EVIMO2 v0 / v1 calibrations' runs.
    derive("accuracy.pixel_integral", std::string(frozen_calibration(cams) ? "phase1" : "fixed64"));
    if (profile == "fidelity") {
      // P2-B's adoption (the user, 2026-10-12; ENGINE_CPP §0.1 "P2-B adoption notes"), held by the statistical gate
      // against uniform N_band (matched >= 99.9 %, p99 |dt| <= 1 us; BENCH.md) instead of T2 bit for bit: the jump
      // gate at fidelity's tolerance (R3; balanced's), but not under an EVIMO2 v0 / v1 sensor profile (their runs stay
      // bitwise); B6c's per-pixel band at reach scales (0.25, 0) with the band N where pooled escalation's premise
      // holds on every camera, else B4a's tile map at (1, 1) (also without the band, where both are inert).
      derive("accuracy.eps_jump", frozen ? Value{} : Value{f.eps});
      const bool band_on = !std::holds_alternative<std::monostate>(*run.get("accuracy.n_band_max"));
      derive("accuracy.band_growth",
             std::string(band_on && band_support(run, cams) >= kPixelBandSupportMin ? "pixel" : "tile"));
      const bool pixel_map = std::get<std::string>(*run.get("accuracy.band_growth")) == "pixel";
      derive("accuracy.band_reach", pixel_map ? std::vector<double>{0.25, 0.0} : std::vector<double>{1.0, 1.0});
    }
  }
  for (const Param& p : schema()) {
    if (p.scope != Scope::kRun || run.get(p.key) != nullptr) continue;
    if (p.def == Default::kLiteral) {
      run.e[p.key] = default_entry(p);
    } else if (p.def == Default::kRequired) {
      required(collect, p.key, required_hint(p), missing(p.key + " is required" + required_hint(p), p.key));
    }
  }
  if (collect != nullptr && !collect->empty()) return;  // the checks below read the run keys
  if (const Value* d = run.get("run.device"); std::get<std::string>(*d) == "auto") {
    run.e["run.device"] = {std::string(b.device_available ? "gpu" : "cpu"), "derived", "derived", ""};
  }
  for (const Param& p : schema()) {
    if (p.scope == Scope::kDataset && run.get(p.key) == nullptr) run.e[p.key] = default_entry(p);
  }
  // Derived against explicit (warnings).
  const auto exp = [&](const char* k) { return run.expl.count(k) != 0; };
  const double n = run.num("accuracy.n_subsamples");
  if (exp("accuracy.n_subsamples") && n < f.n_step) {
    warn(DiagKind::kDerivedConflict,
         "accuracy.n_subsamples = " + format_double(n) + " is below N_step = " + format_double(f.n_step) +
             " (design contrast at the design threshold " + format_double(f.theta_d) + ")",
         "accuracy.n_subsamples", "", run.layer("accuracy.n_subsamples"),
         {"accuracy.design_threshold", "accuracy.design_contrast"});
  }
  if (exp("accuracy.d_px") && run.num("accuracy.d_px") > 2.0 / std::sqrt(n)) {
    warn(DiagKind::kDerivedConflict,
         "accuracy.d_px = " + format_double(run.num("accuracy.d_px")) + " is above 2/sqrt(N) = " +
             format_double(2.0 / std::sqrt(n)),
         "accuracy.d_px", "", run.layer("accuracy.d_px"), {"accuracy.n_subsamples"});
  }
  // phase1's literals too (F026): its 0.002 is kappa sigma_v,ref, so a bright floor sigma_v,min < 0.02 takes kappa
  // above 0.1.
  for (const char* k : {"accuracy.eps_L", "accuracy.refine_tol_log"}) {
    const Value* v = run.get(k);
    if ((exp(k) || profile == "phase1") && !std::holds_alternative<std::monostate>(*v) &&
        std::get<double>(*v) > f.eps) {
      warn(DiagKind::kDerivedConflict,
           std::string(k) + " = " + format_value(*v) + (exp(k) ? "" : " (profile phase1)") +
               " is above the fidelity tolerance " + format_double(f.eps) + " for the run's quietest sensor" +
               (f.sigma_min < kSigmaRef ? " (sigma_v,min = " + fmt3(f.sigma_min) + " < 0.02)" : ""),
           k, "", run.layer(k));
    }
  }
  if (exp("accuracy.min_dt_ns") && f.q_min > 0 && run.num("accuracy.min_dt_ns") > static_cast<double>(pow2ceil(f.q_min))) {
    warn(DiagKind::kDerivedConflict,
         "accuracy.min_dt_ns = " + format_value(*run.get("accuracy.min_dt_ns")) +
             " is coarser than the readout quantum (pow2ceil = " + std::to_string(pow2ceil(f.q_min)) + " ns)",
         "accuracy.min_dt_ns", "", run.layer("accuracy.min_dt_ns"), {"sensor.readout_quantum_s"});
  }
  // The economy schedule's switches that another key turns off (ENGINE_CPP I7): set (configuration or CLI) but inert.
  const auto inert = [&](const char* k, const char* on) {
    const auto it = run.expl.find(k);
    if (it != run.expl.end() && it->second.layer->user) {
      warn(DiagKind::kInert, std::string(k) + " is set but inert: it applies when " + on, k, "",
           it->second.layer->record);
    }
  };
  const auto null = [&](const char* k) { return std::holds_alternative<std::monostate>(*run.get(k)); };
  if (null("accuracy.refine_tol_log")) {
    for (const char* k : {"accuracy.mid_levels", "accuracy.half_ratio"}) {
      inert(k, "accuracy.refine_tol_log is not null (the midpoint test is off)");
    }
  }
  if (!std::get<bool>(*run.get("accuracy.refine_signature"))) {
    for (const char* k : {"accuracy.tau_jump_ns", "accuracy.delta_ref", "accuracy.eps_jump"}) {
      inert(k, "accuracy.refine_signature is true (signature flags are off)");
    }
  } else if (null("accuracy.tau_jump_ns")) {
    inert("accuracy.delta_ref", "accuracy.tau_jump_ns is not null (no tau stop)");
  }
  // The band N's rules (PHASE2 D-P2-4, P2-B B4a): N_band = accuracy.n_band_max (no c_loc until EFC), a power of two
  // above N_base, itself a power of two >= 64; one N for an ideal reset (an explicit band there is an error, the lead's
  // decision 6); the fixed 64-position integral (T2). The clamp is allowed. Recorded (run.json "derived"): theta_band
  // (F085), c_cover (the edge contrast N_band keeps at kappa_q 0.35 at sigma = min(sigma_v,min, 0.02), F068; a warning
  // where shadow-casting lights or misses on a zero background may exceed it) and eps_switch, the engine's unit of the
  // switch steps (SwitchApply) and P2-0b's jumps: refine_tol_log / 8, kEpsRef / 8 without the midpoint test (D-P2-4's
  // eps_jump = eps_L / 8 where, as fidelity and balanced derive them, the two tolerances agree; an explicit eps_L
  // moves no schedule key, P0_PARAMS; never accuracy.eps_jump).
  // P2-B B6c: the band's map and reach scales apply with the band N, the scales under the pixel map only, and the pixel
  // map only where pooled escalation's premise holds (every camera's reach, render::support_px's formula with the
  // truncation at most kPixelBandTruncationMax, at least kPixelBandSupportMin), on which T2 then rests (a narrower
  // response keeps the tile map).
  const bool pixel_band = std::get<std::string>(*run.get("accuracy.band_growth")) == "pixel";
  if (null("accuracy.n_band_max")) {
    for (const char* k : {"accuracy.band_growth", "accuracy.band_reach"}) {
      inert(k, "accuracy.n_band_max is not null (one N)");
    }
    return;
  }
  if (!pixel_band) inert("accuracy.band_reach", "accuracy.band_growth is pixel");
  if (const double support = band_support(run, cams); pixel_band && support < kPixelBandSupportMin) {
    fail(DiagKind::kCrossKey,
         "accuracy.band_growth = pixel: a camera's pixel response reaches " + fmt3(support) +
             " px (0.5 sensor.aperture_px + min(accuracy.psf_truncation_sigmas, 3) optics.blur_sigma_px), below " +
             fmt3(kPixelBandSupportMin) + " px, where pooled escalation misses edges between pixels (P2-B B6); keep "
             "tile",
         "accuracy.band_growth", "", run.layer("accuracy.band_growth"),
         {"sensor.aperture_px", "accuracy.psf_truncation_sigmas", "optics.blur_sigma_px"});
  }
  const int64_t n_band = std::get<int64_t>(*run.get("accuracy.n_band_max"));
  const auto pow2 = [](int64_t v) { return v > 0 && (v & (v - 1)) == 0; };
  const std::string what = "accuracy.n_band_max = " + std::to_string(n_band);
  const std::string band_layer = run.layer("accuracy.n_band_max");
  const auto band_error = [&](DiagKind kind, const std::string& why, std::vector<std::string> keys) {
    fail(kind, what + why, "accuracy.n_band_max", "", band_layer, std::move(keys));
  };
  if (!pow2(n_band)) band_error(DiagKind::kHardRange, ": the band N is a power of two (64, 128, 256, 512 or 1024)", {});
  if (ideal_reset(cams)) {
    band_error(DiagKind::kCrossKey,
               ": a camera has sensor.reset_mode ideal, whose runs keep one N (PHASE2 D-P2-4); set it to null",
               {"sensor.reset_mode"});
  }
  const auto base = static_cast<int64_t>(n);
  if (!pow2(base) || base < kBandBaseMin) {
    band_error(DiagKind::kCrossKey,
               ": the band N needs accuracy.n_subsamples (N_base) to be a power of two >= 64, not " +
                   std::to_string(base) + " (P2-B B3: T1 holds bitwise there)",
               {"accuracy.n_subsamples"});
  }
  if (n_band <= base) {
    band_error(DiagKind::kCrossKey, ": the band N must exceed accuracy.n_subsamples = " + std::to_string(base),
               {"accuracy.n_subsamples"});
  }
  if (std::get<std::string>(*run.get("accuracy.pixel_integral")) != "fixed64") {
    band_error(DiagKind::kCrossKey,
               ": the band N needs accuracy.pixel_integral: fixed64 (PHASE2 D-P2-3; an EVIMO2 v0 / v1 sensor "
               "profile, phase1 and reference derive phase1)",
               {"accuracy.pixel_integral"});
  }
  const double sigma = std::min(f.sigma_min, kSigmaRef);
  const double c_cover = kKappaQ * sigma * std::pow(static_cast<double>(n_band), 0.75) / kKappaQRate;
  const Value* tol = run.get("accuracy.refine_tol_log");
  facts = {{"n_band", static_cast<double>(n_band)}, {"theta_band", theta_band(cams)}, {"c_cover", c_cover},
           {"eps_switch", (std::holds_alternative<std::monostate>(*tol) ? kEpsRef : std::get<double>(*tol)) / 8.0}};
  const bool dark = b.zero_background.value_or(false);
  if (b.shadow_lights > 0 || dark) {
    std::string why = b.shadow_lights > 0 ? std::to_string(b.shadow_lights) + " shadow-casting light(s)" : "";
    if (dark) why += std::string(why.empty() ? "" : " and ") + "misses on a zero background";
    warn(DiagKind::kRunLevel,
         what + " keeps the staircase at kappa_q <= 0.35 up to an edge contrast c_cover = " + fmt3(c_cover) +
             " log-e; the scene's " + why + " may give larger ones (PHASE2 G1)",
         "accuracy.n_band_max", "", band_layer);
  }
}

// The numbers of a value (a list's elements), for the soft ranges.
std::vector<double> numbers_of(const Value& v) {
  std::vector<double> xs;
  std::visit(
      [&](const auto& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, int64_t>) {
          xs.push_back(static_cast<double>(x));
        } else if constexpr (std::is_same_v<T, std::vector<double>> || std::is_same_v<T, std::vector<int64_t>>) {
          for (const auto e : x) xs.push_back(static_cast<double>(e));
        }
      },
      v);
  return xs;
}

// Soft ranges (values from profiles, the configuration or the CLI) and inert values (the configuration or the CLI);
// `camera` names t's camera (empty: the run's table).
void soft_and_inert(const Table& t, const Table& run, const std::string& camera, Warnings& warn) {
  const std::string prefix = camera.empty() ? std::string() : "cameras." + camera + ".";
  for (const auto& [key, f] : t.expl) {
    const Param& p = *find(key);
    const Value& v = f.set->value;
    for (const double x : numbers_of(v)) {
      if (p.soft.unset() || p.soft.contains(x)) continue;
      const bool below = (p.soft.lo_open ? x <= p.soft.lo : x < p.soft.lo);
      warn(DiagKind::kSoftRange,
           prefix + key + " = " + format_value(v) + " (" + f.layer->label + "): " + (below ? "below" : "above") +
               " the physical range " + format_double(p.soft.lo) + "-" + format_double(p.soft.hi) + " (" + p.source +
               ")",
           key, camera, f.layer->record);
      break;
    }
    if (f.layer->user) {
      const Table& scope = p.scope == Scope::kCamera ? t : run;
      if (const When* w = failing(p, [&](std::string_view k) { return scope.get(k); })) {
        const bool any = std::holds_alternative<std::vector<std::string>>(w->value);
        std::string also;
        for (const When& a : p.also_when) also += (also.empty() ? " or when " : " and ") + a.key + " is " +
                                                  format_value(a.value);
        std::vector<std::string> keys{w->key};
        for (const When& a : p.also_when) keys.push_back(a.key);
        warn(DiagKind::kInert,
             prefix + key + " is set but inert: it applies when " + w->key + (any ? " is one of " : " is ") +
                 format_value(w->value) + " (here " + format_value(*scope.get(w->key)) + ")" + also,
             key, camera, f.layer->record, std::move(keys));
      }
    }
  }
}

// Flagged defaults, one warning per kind and run naming every relevant key a camera takes from its family prior or
// the schema; a profile or the run configuration that states the value silences it:
// - values that come from EVIMO2 data or its setup rather than a neutral prior (P0c; the flagged family priors, D4:
//   the hot-pixel rates; the RGB weights, under the illuminant assumed for EVIMO2; literals such as the 0.15 px blur,
//   a prior for EVIMO2's lens);
// - features a family prior turns off for want of a literature value (D4c, P0e: hot pixels).
void flagged_defaults(const std::vector<Table>& cams, const BundleInfo& b, const std::vector<size_t>& sel,
                      Warnings& warn) {
  std::string evimo2, off;
  std::vector<std::string> evimo2_keys, off_keys;  // the schema keys, each once
  for (size_t j = 0; j < cams.size(); ++j) {
    const Table& t = cams[j];
    const std::string family = std::get<std::string>(*t.get("sensor.family"));
    const bool hot = camera_sensor(t).hot_pixels();
    for (const Param& p : schema()) {
      const auto it = t.e.find(p.key);
      if (it == t.e.end()) continue;
      const Entry& e = it->second;
      bool flagged = false, no_literature = false;
      if (e.layer == "default") {
        flagged = p.evimo2_derived;
      } else if (e.layer == "prior:" + family) {
        const sensor::Prior* v = sensor::find_prior(family, p.key);
        flagged = v->evimo2_derived;
        no_literature = v->no_literature;
      }
      if (!(flagged || no_literature) || failing(p, [&](std::string_view k) { return t.get(k); }) != nullptr) continue;
      if (!hot && sensor::hot_pixel_key(p.key)) continue;  // inert without hot pixels
      std::string& keys = flagged ? evimo2 : off;
      keys += (keys.empty() ? "" : ", ") + ("cameras." + b.cameras[sel[j]].name + "." + p.key);
      std::vector<std::string>& list = flagged ? evimo2_keys : off_keys;
      if (std::find(list.begin(), list.end(), p.key) == list.end()) list.push_back(p.key);
    }
  }
  if (!evimo2.empty()) {
    warn(DiagKind::kFlaggedDefault,
         "defaults derived from EVIMO2 data or its setup, not neutral priors (D4): " + evimo2 +
             "; set them, or use the EVIMO2 profiles for an EVIMO2 replay",
         "", "", "", std::move(evimo2_keys));
  }
  if (!off.empty()) {
    warn(DiagKind::kFlaggedDefault,
         "features off in the family prior for want of a literature value (D4c): " + off +
             "; set them to simulate the feature (the EVIMO2 profiles carry measured values)",
         "", "", "", std::move(off_keys));
  }
}

// validate_run() (§2.4): the span and its whole windows, the crop on every camera, the selection, min_dt_ns, the
// fixed-rate period, a static warm-up's minimum and the device. Sets the window span.
void validate_run(const Table& run, const BundleInfo& b, const std::vector<size_t>& sel, Resolved& out) {
  Warnings warn{out.warnings, out.diagnostics};
  const auto refuse = [&](const std::string& message, const std::string& key, std::vector<std::string> keys = {},
                          const std::string& camera = {}) {
    fail(DiagKind::kRunLevel, message, key, camera, run.layer(key), std::move(keys));
  };
  const Value* t0v = run.get("run.t_start_ns");
  const Value* t1v = run.get("run.t_end_ns");
  const bool has0 = !std::holds_alternative<std::monostate>(*t0v), has1 = !std::holds_alternative<std::monostate>(*t1v);
  if (has0 && has1 && std::get<int64_t>(*t1v) <= std::get<int64_t>(*t0v)) {
    refuse("run.t_end_ns must be after run.t_start_ns", "run.t_end_ns", {"run.t_start_ns"});
  }
  const int64_t t0 = has0 ? std::max(b.t_start_ns, std::get<int64_t>(*t0v)) : b.t_start_ns;
  const int64_t t1 = has1 ? std::min(b.t_end_ns, std::get<int64_t>(*t1v)) : b.t_end_ns;
  out.t_s = -time::floor_div(-t0, time::kWindowNs) * time::kWindowNs;
  out.t_e = time::floor_div(t1, time::kWindowNs) * time::kWindowNs;
  if (out.t_e <= out.t_s) {
    refuse("run span [" + std::to_string(t0) + ", " + std::to_string(t1) + "] ns holds no whole window of " +
               std::to_string(time::kWindowNs) + " ns aligned to t = 0",
           "run.t_start_ns", {"run.t_end_ns"});
  }
  if (const auto* c = std::get_if<std::vector<int64_t>>(run.get("run.crop_px"))) {
    for (const size_t i : sel) {
      const BundleCamera& cam = b.cameras[i];
      if (!((*c)[0] < (*c)[2] && (*c)[2] <= cam.width && (*c)[1] < (*c)[3] && (*c)[3] <= cam.height)) {
        refuse("run.crop_px = " + format_value(*run.get("run.crop_px")) + ": empty or outside camera " + cam.name +
                   " (" + std::to_string(cam.width) + " x " + std::to_string(cam.height) + ")",
               "run.crop_px", {}, cam.name);
      }
    }
  }
  const int64_t min_dt = std::get<int64_t>(*run.get("accuracy.min_dt_ns"));
  if ((min_dt & (min_dt - 1)) != 0) {
    refuse("accuracy.min_dt_ns = " + std::to_string(min_dt) + ": not a power of two", "accuracy.min_dt_ns");
  }
  if (const auto* hz = std::get_if<double>(run.get("run.fixed_rate_hz"))) {
    // On the ns grid to 1e-6 ns, or to the rounding of 1e9 / hz above 1 s periods; the negated <= refuses an infinite
    // period, and the upper bound keeps llround in int64.
    const double period = 1e9 / *hz, p = std::round(period);
    if (!(std::fabs(period - p) <= std::max(1e-6, 1e-15 * period)) || p < 1 || p > 0x1p62) {
      refuse("run.fixed_rate_hz = " + format_double(*hz) + ": not an integral period in ns (e.g. 1e6 Hz -> 1000 ns)",
             "run.fixed_rate_hz");
    }
  }
  const int64_t warmup = std::get<int64_t>(*run.get("accuracy.warmup_ns"));
  if (std::get<std::string>(*run.get("accuracy.warmup")) == "static" && warmup < kMinStaticWarmupNs) {
    refuse("accuracy.warmup_ns = " + std::to_string(warmup) + ": a static warm-up needs at least 50 ms (DESIGN §8.4); "
           "accuracy.warmup: motion takes whole windows from one",
           "accuracy.warmup_ns", {"accuracy.warmup"});
  }
  const auto asked = run.expl.find("accuracy.warmup");
  if (std::get<std::string>(*run.get("accuracy.warmup")) == "motion" && asked != run.expl.end() &&
      asked->second.layer->user) {
    // An explicit motion warm-up the bundle cannot cover in full (fidelity's default degrades as quietly, recorded
    // as run.json's warmup_windows): the engine renders the warm-up's whole windows the bundle covers before t_s and
    // completes them by a static warm-up of at least 50 ms in all (CameraEngine::start).
    const int64_t k = (warmup + time::kWindowNs - 1) / time::kWindowNs;
    const int64_t first = -time::floor_div(-b.t_start_ns, time::kWindowNs) * time::kWindowNs;
    const int64_t covered = std::max<int64_t>((out.t_s - first) / time::kWindowNs, 0);
    if (covered < k) {
      warn(DiagKind::kRunLevel,
           "accuracy.warmup = motion: the bundle covers " + std::to_string(covered) + " of the " + std::to_string(k) +
               " warm-up windows before the first window; the rest is a static warm-up on the first covered frame, "
               "to 50 ms in all at least (DESIGN §8.4)",
           "accuracy.warmup", "", asked->second.layer->record, {"accuracy.warmup_ns"});
    }
  }
  if (std::get<std::string>(*run.get("run.device")) == "gpu" && !b.device_available) {
    refuse("run.device = 'gpu': no usable device here (the device engine needs the CUDA and OptiX build, a CUDA "
           "device and OptiX); run.device: cpu runs the host engine",
           "run.device");
  }
}

// Canonical text of a value: hex floats, so equal text is equal bits.
struct Canonical {
  std::string operator()(std::monostate) const { return "null"; }
  std::string operator()(bool b) const { return b ? "true" : "false"; }
  std::string operator()(int64_t i) const { return std::to_string(i); }
  std::string operator()(double d) const {
    char buf[40];
    const auto r = std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::hex);
    return std::string(buf, r.ptr);
  }
  std::string operator()(const std::string& s) const { return "\"" + s + "\""; }
  template <class T>
  std::string operator()(const std::vector<T>& l) const {
    std::string out = "[";
    for (size_t i = 0; i < l.size(); ++i) out += (i ? "," : "") + (*this)(l[i]);
    return out + "]";
  }
};

std::string canonical(const Value& v) { return std::visit(Canonical{}, v); }

// digest(stage, camera) (§1.2): stage scene is the bundle digest; each later stage chains the previous digest with the
// canonical values of its keys (run keys, then the camera's), and the render stage also with the engine build. Scene
// keys the bundle does not use enter no digest: every one next to a supplied bundle (its content digest is the root),
// and those whose relevance fails (the lights off; the other light mode), of which relight reads none.
std::array<std::string, 5> digests(const Resolved& r, const Entries& cam, const BundleInfo& b) {
  std::map<std::string_view, const Value*, std::less<>> run;
  for (const auto& [k, e] : r.run) run.emplace(k, &e.value);
  const auto get = [&](std::string_view k) -> const Value* {
    const auto it = run.find(k);
    return it == run.end() ? nullptr : it->second;
  };
  // P2-H H4b (D-P3-11's rule, without its flag): the rate limiter's rows enter no digest while it is off, so a run
  // with the limiter off keeps the digests it had before the rows existed.
  bool limiter_off = true;
  for (const auto& [k, e] : cam) {
    const auto* mode = std::get_if<std::string>(&e.value);
    if (k == "sensor.limiter" && mode != nullptr) limiter_off = *mode == "off";
  }
  // P3-A A3 (D-P3-11, outputs.md §4.10): the IMU's rows enter no digest while the camera's imu.enabled is false, so a run
  // without an IMU keeps the digests it had before the rows existed; with it the output digest moves, no earlier one.
  bool imu_off = true;
  for (const auto& [k, e] : cam) {
    if (k == "imu.enabled") imu_off = !std::get<bool>(e.value);
  }
  std::array<std::string, 5> d;
  d[0] = b.bundle_digest;
  for (int s = 1; s < 5; ++s) {
    std::string text = d[s - 1] + "\n" + kStages[s] + "\n";
    if (s == 1) text += "engine " + b.engine_digest + "\n";
    for (const Entries* list : {&r.run, &cam}) {
      for (const auto& [k, e] : *list) {
        const Param& p = *find(k);
        if (static_cast<int>(p.stage) != s) continue;
        if (p.scope == Scope::kScene && (b.supplied || failing(p, get) != nullptr)) continue;
        if (k == "scene.baked_scale" && e.value == Value{1.0}) continue;  // D-P3-11: neutral at its default
        if (list == &cam && limiter_off && k.starts_with("sensor.limiter")) continue;
        if (list == &cam && imu_off && k.starts_with("imu.")) continue;
        // P2-D D3a: accuracy.localization shadow (P&V beside bisection, counters only) changes no output: hashed as
        // bisect, so its runs share bisect's digests and caches.
        const bool shadow = k == "accuracy.localization" && std::holds_alternative<std::string>(e.value) &&
                            std::get<std::string>(e.value) == "shadow";
        text += k + "=" + (shadow ? canonical(Value{std::string("bisect")}) : canonical(e.value)) + "\n";
      }
      text += "--\n";
    }
    d[s] = sha256_hex(text);
  }
  return d;
}

Entries ordered(const Table& t, bool camera) {
  Entries out;
  for (const Param& p : schema()) {
    if ((p.scope == Scope::kCamera) != camera) continue;
    if (const auto it = t.e.find(p.key); it != t.e.end()) out.emplace_back(p.key, it->second);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// The IMU's rows (P3-A A3; outputs.md §4.11, §6)
// ---------------------------------------------------------------------------------------------

// The imu.* values of a camera's table.
imu::Values imu_values(const Table& t) {
  imu::Values v;
  for (const auto& [k, e] : t.e) {
    if (k.starts_with("imu.")) v.emplace(k, e.value);
  }
  return v;
}

// The schema keys a message names (tokens "imu.<...>" of letters, digits, '_' and '.'), in order of appearance.
std::vector<std::string> imu_keys_in(const std::string& message) {
  std::vector<std::string> keys;
  const auto word = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.'; };
  for (size_t i = 0; i < message.size();) {
    if (!word(message[i])) {
      ++i;
      continue;
    }
    size_t j = i;
    while (j < message.size() && word(message[j])) ++j;
    std::string token = message.substr(i, j - i);
    while (!token.empty() && token.back() == '.') token.pop_back();
    if (token.starts_with("imu.") && find(token) != nullptr && std::find(keys.begin(), keys.end(), token) == keys.end()) {
      keys.push_back(token);
    }
    i = j;
  }
  return keys;
}

// imu::validate (§6) on an enabled camera's resolved imu.* values: its errors fail (kCrossKey, naming the keys the first
// one names), its warnings are collected. Returns the camera's ImuParams.
imu::ImuParams check_imu(const Table& t, const std::string& camera, Warnings& warn) {
  imu::ImuParams p;
  try {
    p = imu::from_values(imu_values(t));
  } catch (const imu::Refused& e) {
    fail(DiagKind::kType, "cameras." + camera + ": " + e.what(), "", camera);
  }
  const imu::Check c = imu::validate(p);
  if (!c.errors.empty()) {
    std::string all;
    for (const std::string& e : c.errors) all += (all.empty() ? "" : "; ") + e;
    std::vector<std::string> keys = imu_keys_in(c.errors.front());
    const std::string key = keys.empty() ? std::string() : keys.front();
    fail(DiagKind::kCrossKey, "cameras." + camera + ": " + all, key, camera, key.empty() ? std::string() : t.layer(key),
         std::move(keys));
  }
  for (const std::string& w : c.warnings) {
    std::vector<std::string> keys = imu_keys_in(w);
    const std::string key = keys.empty() ? std::string() : keys.front();
    warn(DiagKind::kCrossKey, "cameras." + camera + ": " + w, key, camera, key.empty() ? std::string() : t.layer(key),
         std::move(keys));
  }
  return p;
}

// Cameras of one rig share a world, so imu.gravity_mps2 should agree between the enabled cameras (null is one value: the
// bundle's gravity_world_mps2, else (0, 0, -9.80665)); a warning naming each camera that differs from the first.
void check_imu_gravity(const std::vector<std::pair<std::string, const Table*>>& enabled, Warnings& warn) {
  for (size_t i = 1; i < enabled.size(); ++i) {
    const Value& a = *enabled[0].second->get("imu.gravity_mps2");
    const Value& b = *enabled[i].second->get("imu.gravity_mps2");
    if (a == b) continue;
    const std::string& cam = enabled[i].first;
    warn(DiagKind::kCrossKey,
         "imu.gravity_mps2 differs between cameras." + enabled[0].first + " (" + format_value(a) + ") and cameras." + cam +
             " (" + format_value(b) + "): cameras of one rig share a world",
         "imu.gravity_mps2", cam, enabled[i].second->layer("imu.gravity_mps2"), {"imu.gravity_mps2"});
  }
}

// The full resolver's IMU checks (outputs.md §4.11): validate() of each camera with imu.enabled, then the gravity.
void check_imu_cameras(const std::vector<Table>& cams, const BundleInfo& b, const std::vector<size_t>& sel,
                       Warnings& warn) {
  std::vector<std::pair<std::string, const Table*>> enabled;
  for (size_t j = 0; j < cams.size(); ++j) {
    const Value* on = cams[j].get("imu.enabled");
    if (on == nullptr || !std::get<bool>(*on)) continue;
    check_imu(cams[j], b.cameras[sel[j]].name, warn);
    enabled.emplace_back(b.cameras[sel[j]].name, &cams[j]);
  }
  check_imu_gravity(enabled, warn);
}

}  // namespace

std::string scene_digest(const ValueMap& values, const std::map<std::string, std::string, std::less<>>& facts) {
  for (const auto& [k, v] : values) {
    if (find(k) == nullptr) fail(DiagKind::kUnknownKey, "scene_digest: unknown key " + k, k);
  }
  std::string text = "scene\n";
  for (const Param& p : schema()) {
    if (p.stage != Stage::kScene) continue;
    if (const auto it = values.find(p.key); it != values.end()) text += p.key + "=" + canonical(it->second) + "\n";
  }
  text += "--\n";
  for (const auto& [k, v] : facts) text += k + "=" + v + "\n";
  return sha256_hex(text);
}

std::vector<std::string> profiles_needed(const Inputs& in) {
  const Parsed P = parse(in, false);
  std::vector<std::string> out;
  const auto add = [&](const std::string& id) {
    if (std::find(out.begin(), out.end(), id) == out.end()) out.push_back(id);
  };
  for (const std::string& id : P.root_ids) add(id);
  for (const auto& [k, v] : in.config.map) {  // a mapping: parse() checked it
    if (k != "cameras") continue;
    for (const auto& [block, node] : v.map) {  // document order
      if (const auto it = P.block_ids.find(block); it != P.block_ids.end()) {
        for (const std::string& id : it->second) add(id);
      }
    }
  }
  return out;
}

Resolved resolve_scene(const Inputs& in) {
  const Parsed P = parse(in, true);
  Resolved out;
  Warnings warn{out.warnings, out.diagnostics};
  Table run = explicit_run(P);
  fill_scene(run, in, false, std::nullopt, warn);
  for (const char* k : {"output.dir", "output.overwrite", "execution.scene_workers"}) {
    const Param& p = *find(k);
    if (run.get(k) != nullptr) continue;
    if (p.def == Default::kRequired) fail(DiagKind::kRequired, std::string(k) + " is required", k);
    run.e[k] = default_entry(p);
  }
  for (const Param& p : schema()) {
    if (p.scope == Scope::kScene || p.key == "output.dir" || p.key == "output.overwrite" ||
        p.key == "execution.scene_workers") {
      if (const auto it = run.e.find(p.key); it != run.e.end()) out.run.emplace_back(p.key, it->second);
    }
  }
  return out;
}

std::string canonical_text(const Value& v) { return canonical(v); }

Entries resolve_section(const Inputs& in, const BundleInfo& b, std::string_view camera, std::string_view section,
                        std::vector<Diagnostic>* warnings) {
  const Parsed P = parse(in, true);
  const std::string cam(camera);
  const BundleCamera* bc = nullptr;
  std::string names;
  for (const BundleCamera& c : b.cameras) {
    names += (names.empty() ? "" : ", ") + c.name;
    if (c.name == cam) bc = &c;
  }
  if (bc == nullptr) {
    fail(DiagKind::kUnknownCamera, "cameras." + cam + ": not a camera of the bundle (" + names + ")", "", cam);
  }
  const std::string prefix = std::string(section) + ".";
  std::vector<const Param*> rows;
  for (const Param& p : schema()) {
    if (p.scope == Scope::kCamera && p.key.starts_with(prefix)) rows.push_back(&p);
  }
  if (rows.empty()) throw std::invalid_argument("resolve_section: no camera section '" + std::string(section) + "'");
  // The layers resolve() reads for this camera (root profiles, the camera's, the configuration, the CLI), the same
  // lookup, and the row's default where none sets it: literal ones, and the bundle camera's derived ones.
  const auto layers = chain(P, cam);
  Table t;
  for (const Param* p : rows) {
    if (const Found f = lookup(layers, cam, p->key); f.set != nullptr) {
      t.set(p->key, f);
    } else if (p->def == Default::kLiteral) {
      t.e[p->key] = default_entry(*p);
    } else if (p->def == Default::kDerived && p->key == "optics.f_number") {
      t.e[p->key] = {bc->f_number, "bundle", "derived", ""};
    } else if (p->def == Default::kDerived && p->key == "sensor.clock_offset_ns") {
      t.e[p->key] = {bc->clock_offset_ns, "bundle", "derived", ""};
    } else {
      throw std::logic_error("resolve_section: " + p->key + " has a " + name(p->def) +
                             " default; resolve the camera in full");
    }
  }
  if (warnings != nullptr) {
    std::vector<std::string> text;
    Warnings warn{text, *warnings};
    soft_and_inert(t, Table{}, cam, warn);
  }
  Entries out;
  for (const Param* p : rows) out.emplace_back(p->key, t.e.at(p->key));
  return out;
}

SectionResult resolve_section(const SectionInputs& in) {
  SectionResult out;
  Warnings warn{out.warnings, out.diagnostics};
  std::string names;
  for (const SectionCamera& c : in.cameras) names += (names.empty() ? "" : ", ") + c.name;
  const auto known = [&](const std::string& cam) {
    return std::any_of(in.cameras.begin(), in.cameras.end(), [&](const SectionCamera& c) { return c.name == cam; });
  };
  // The new layers, parsed and checked (types, hard ranges, mounts) as the full resolver's are.
  Inputs profile_inputs;
  profile_inputs.profiles = in.profile_docs;
  const Layer cli = parse_cli(in.sets);
  std::map<std::string, Layer, std::less<>> loaded;  // "r:<id>" root-mounted, "c:<id>" camera-mounted
  const auto load = [&](const std::string& id, bool camera_mount, const char* option) {
    check_id(id, option);
    const std::string key = (camera_mount ? "c:" : "r:") + id;
    if (loaded.count(key) == 0) loaded.emplace(key, parse_profile(id, profile_inputs, camera_mount));
    return &loaded.at(key);
  };
  std::vector<const Layer*> root;
  for (const std::string& id : in.profiles) root.push_back(load(id, false, "--profile"));
  std::map<std::string, std::vector<const Layer*>, std::less<>> by_camera;
  for (const auto& [cam, ids] : in.camera_profiles) {
    if (!known(cam)) {
      fail(DiagKind::kUnknownCamera, "--camera-profile " + cam + ": not a camera of the run (" + names + ")", "", cam);
    }
    for (const std::string& id : ids) by_camera[cam].push_back(load(id, true, "--camera-profile"));
  }
  // ballasim imu changes the imu.* keys only.
  const auto imu_only = [&](const Layer& L) {
    const auto refuse = [&](const std::string& where) {
      fail(DiagKind::kWrongScope,
           "ballasim imu changes imu.* keys only; re-run for others: " + L.label + " sets " + where, "", "", L.record);
    };
    for (const auto& kv : L.run) refuse(kv.second.where);
    for (const auto& block : L.cams) {
      for (const auto& kv : block.second) {
        if (!kv.first.starts_with("imu.")) refuse(kv.second.where);
      }
    }
  };
  imu_only(cli);
  for (const auto& [key, L] : loaded) imu_only(L);
  for (const auto& [block, values] : cli.cams) {
    if (block != "default" && !known(block)) {
      fail(DiagKind::kUnknownCamera, "cameras." + block + ": not a camera of the run (" + names + ")", "", block,
           cli.record);
    }
  }
  // A root profile may serve several rigs: its blocks of other cameras are unused, with a warning (as resolve()).
  for (const auto& [key, L] : loaded) {
    if (key.front() != 'r') continue;
    for (const auto& [block, values] : L.cams) {
      if (block != "default" && !known(block)) {
        warn(DiagKind::kUnknownCamera,
             "profile " + key.substr(2) + ": cameras." + block + " is not a camera of the run (" + names +
                 "); its values are unused",
             "", block, L.record);
      }
    }
  }
  std::vector<const Param*> rows;
  for (const Param& p : schema()) {
    if (p.scope == Scope::kCamera && p.key.starts_with("imu.")) rows.push_back(&p);
  }
  std::vector<Table> tables(in.cameras.size());
  std::vector<std::map<std::string, std::string, std::less<>>> sources(in.cameras.size());
  for (size_t j = 0; j < in.cameras.size(); ++j) {
    const SectionCamera& sc = in.cameras[j];
    Table& t = tables[j];
    auto& source = sources[j];
    for (const Param* p : rows) {
      t.e[p->key] = default_entry(*p);
      source[p->key] = p->source;
    }
    for (const auto& [key, e] : sc.recorded) {  // layer 2: what the run recorded
      const Param* p = find(key);
      const std::string where = "run.json cameras." + sc.name + "." + key;
      if (p == nullptr || !key.starts_with("imu.") || p->scope != Scope::kCamera) {
        fail(DiagKind::kUnknownKey, where + ": not an imu.* key of this version", key, sc.name);
      }
      try {
        t.e[key] = {typed(*p, e.value, where), e.layer, e.status, ""};
      } catch (const std::invalid_argument& err) {
        fail(value_kind(*p, e.value), err.what(), key, sc.name, e.layer);
      }
      source[key] = "run.json (" + e.layer + ")";
    }
    std::vector<const Layer*> layers = root;  // layers 3-5: root profiles, camera profiles, --set
    if (const auto it = by_camera.find(sc.name); it != by_camera.end()) {
      layers.insert(layers.end(), it->second.begin(), it->second.end());
    }
    layers.push_back(&cli);
    for (const Param* p : rows) {
      const Found f = lookup(layers, sc.name, p->key);
      if (f.set == nullptr) continue;
      t.set(p->key, f);
      source[p->key] = f.layer->source + (f.set->source.empty() ? std::string() : ": " + f.set->source);
    }
    soft_and_inert(t, Table{}, sc.name, warn);
  }
  std::vector<std::pair<std::string, const Table*>> enabled;
  for (size_t j = 0; j < in.cameras.size(); ++j) {
    const SectionCamera& sc = in.cameras[j];
    const Table& t = tables[j];
    double change = 0;
    if (std::get<bool>(*t.get("imu.enabled"))) {
      change = imu::rotation_change(check_imu(t, sc.name, warn));
      enabled.emplace_back(sc.name, &t);
    }
    SectionEntries entries;
    std::string text = sc.sensor_digest + "\nimu\n";  // the digest of the IMU's inputs: the sensor stage and the values
    for (const Param* p : rows) {
      const Entry& e = t.e.at(p->key);
      entries.push_back({p->key, {e.value, e.layer, e.status, sources[j].at(p->key)}});
      text += p->key + "=" + canonical(e.value) + "\n";
    }
    out.cameras.emplace_back(sc.name, std::move(entries));
    out.imu_digests.push_back(sha256_hex(text));
    out.rotation_change.push_back(change);
  }
  check_imu_gravity(enabled, warn);
  return out;
}

Resolved resolve(const Inputs& in, const BundleInfo& b, Missing* collect) {
  const Parsed P = parse(in, true);
  Resolved out;
  Warnings warn{out.warnings, out.diagnostics};
  std::string names;
  for (const BundleCamera& c : b.cameras) names += (names.empty() ? "" : ", ") + c.name;
  const auto index = [&](const std::string& cam) -> int {
    for (size_t i = 0; i < b.cameras.size(); ++i) {
      if (b.cameras[i].name == cam) return static_cast<int>(i);
    }
    return -1;
  };
  for (const Layer* L : {&P.config, &P.cli}) {
    for (const auto& [block, values] : L->cams) {
      if (block != "default" && index(block) < 0) {
        fail(DiagKind::kUnknownCamera, "cameras." + block + ": not a camera of the bundle (" + names + ")", "", block,
             L->record);
      }
    }
  }
  for (const auto& [block, ids] : P.block_ids) {
    if (block != "default" && index(block) < 0) {
      fail(DiagKind::kUnknownCamera, "cameras." + block + ": not a camera of the bundle (" + names + ")", "", block,
           P.config.record);
    }
  }
  // A root profile may serve several rigs: its blocks of other cameras are unused, with a warning.
  for (const auto& [id, L] : P.profiles) {
    if (std::find(P.root_ids.begin(), P.root_ids.end(), id) == P.root_ids.end()) continue;
    for (const auto& [block, values] : L.cams) {
      if (block != "default" && index(block) < 0) {
        warn(DiagKind::kUnknownCamera,
             "profile " + id + ": cameras." + block + " is not a camera of the bundle (" + names +
                 "); its values are unused",
             "", block, L.record);
      }
    }
  }
  Table run = explicit_run(P);
  // The selection: run.select, or every camera of the bundle.
  std::vector<size_t> sel;
  if (const auto* s = run.get("run.select") ? std::get_if<std::vector<std::string>>(run.get("run.select")) : nullptr) {
    for (const std::string& cam : *s) {
      const int i = index(cam);
      if (i < 0) {
        fail(DiagKind::kUnknownCamera, "run.select: " + cam + " is not a camera of the bundle (" + names + ")",
             "run.select", cam, run.layer("run.select"));
      }
      if (std::find(sel.begin(), sel.end(), static_cast<size_t>(i)) != sel.end()) {
        fail(DiagKind::kRunLevel, "run.select names " + cam + " twice", "run.select", cam, run.layer("run.select"));
      }
      sel.push_back(static_cast<size_t>(i));
    }
    std::sort(sel.begin(), sel.end());
  } else {
    for (size_t i = 0; i < b.cameras.size(); ++i) sel.push_back(i);
  }
  if (sel.empty()) {
    fail(DiagKind::kRunLevel, "run.select: no camera selected", "run.select", "", run.layer("run.select"));
  }
  if (run.get("run.select") == nullptr) run.e["run.select"] = default_entry(*find("run.select"));
  // The cameras: the family first (D2), then every camera key.
  std::string families;
  for (const sensor::Family& f : sensor::families()) families += (families.empty() ? "" : ", ") + f.name;
  std::vector<Table> cams(sel.size());
  // blender.md §3.11: physical photometry on a Blender exchange needs scene.blender.lm_per_w, known from the run's
  // layers before any camera resolves (its default is null).
  const Value* lm_per_w = run.get("scene.blender.lm_per_w");
  const SourceFacts src{run.get("scene.evimo2.sequence") != nullptr || b.low_light.has_value(),
                        in.source_kind == "blender" && !b.supplied,
                        lm_per_w == nullptr || std::holds_alternative<std::monostate>(*lm_per_w)};
  for (size_t j = 0; j < sel.size(); ++j) {
    const BundleCamera& bc = b.cameras[sel[j]];
    const auto layers = chain(P, bc.name);
    const Found fam = lookup(layers, bc.name, "sensor.family");
    if (fam.set == nullptr) {
      const std::string hint = "(" + families + "); the bundle camera says " + (bc.sensor.empty() ? "nothing" : bc.sensor);
      required(collect, "cameras." + bc.name + ".sensor.family", hint,
               missing("cameras." + bc.name + ": sensor.family is required " + hint, "sensor.family", bc.name, {},
                       DiagKind::kFamilyMissing));
    }
    // Collect mode without a family: the family defaults stay unset; the camera's other rows and its photometry's
    // required keys are still gathered.
    const std::string family = fam.set != nullptr ? std::get<std::string>(fam.set->value) : std::string();
    Table& t = cams[j];
    std::vector<const Param*> deferred;
    for (const Param& p : schema()) {
      if (p.scope != Scope::kCamera) continue;
      if (const Found f = lookup(layers, bc.name, p.key); f.set != nullptr) {
        t.set(p.key, f);
        continue;
      }
      switch (p.def) {
        case Default::kFamily: {
          if (family.empty()) break;
          bool inherited = false;
          const sensor::Prior* v = sensor::find_prior(family, p.key, &inherited);
          if (v == nullptr) throw std::logic_error("sensor family " + family + " has no prior for " + p.key);
          t.e[p.key] = {v->value, "prior:" + family, name(v->status),
                        inherited ? "inherited from " + sensor::find_family(family)->base + ", unvalidated for " + family
                                  : ""};
          break;
        }
        case Default::kLiteral:
          t.e[p.key] = default_entry(p);
          break;
        case Default::kDerived:  // the bundle camera's optics.f_number and sensor.clock_offset_ns; the law's cos4
          if (p.key == "optics.f_number") {
            t.e[p.key] = {bc.f_number, "bundle", "derived", ""};
          } else if (p.key == "sensor.clock_offset_ns") {
            t.e[p.key] = {bc.clock_offset_ns, "bundle", "derived", ""};
          } else if (p.key == "optics.vignetting_law") {  // derive_vignetting_law decides it after fill_run (F13-min)
            t.e[p.key] = {std::string("cos4"), "derived", "derived", ""};
          } else {
            throw std::logic_error("resolve: no derivation for the camera row " + p.key);
          }
          break;
        case Default::kRequired:
          if (!p.when.empty()) {  // required while it matters (its condition reads keys resolved after it)
            deferred.push_back(&p);
            break;
          }
          if (p.key == "sensor.family") break;  // collect mode: recorded above
          required(collect, "cameras." + bc.name + "." + p.key, required_hint(p),
                   missing("cameras." + bc.name + "." + p.key + " is required" + required_hint(p), p.key, bc.name));
      }
    }
    for (const Param* p : deferred) {
      if (failing(*p, [&](std::string_view k) { return t.get(k); }) == nullptr) {
        std::vector<std::string> keys;
        for (const When& w : p->when) keys.push_back(w.key);
        required(collect, "cameras." + bc.name + "." + p->key, required_hint(*p),
                 missing("cameras." + bc.name + "." + p->key + " is required" + required_hint(*p), p->key, bc.name,
                         std::move(keys)));
      }
    }
    if (family.empty()) {
      photometry_required(t, bc, src, collect);
      continue;
    }
    photometry(t, bc, src, warn, collect);
    lens_blur(t, bc, warn);
  }
  // Collect mode: the scene's required keys too, then the list (the checks below read the unset values).
  if (collect != nullptr && !collect->empty()) {
    fill_scene(run, in, b.supplied, b.low_light, warn, collect);
    return out;
  }
  // Camera keys with a run-wide target, one SceneSettings per renderer (§2.2 rule 6): the camera-scope SceneSettings rows.
  for (const Param* p : rows_of(Target::kSceneSettings)) {
    if (p->scope != Scope::kCamera) continue;
    const std::string& k = p->key;
    for (size_t j = 1; j < cams.size(); ++j) {
      if (*cams[j].get(k) != *cams[0].get(k)) {
        fail(DiagKind::kRunWideDisagree,
             "cameras disagree on " + k + " (" + b.cameras[sel[0]].name + " " + format_value(*cams[0].get(k)) + ", " +
                 b.cameras[sel[j]].name + " " + format_value(*cams[j].get(k)) +
                 "): one pixel response per run until per-camera renderer settings",
             k, b.cameras[sel[j]].name, cams[j].layer(k));
      }
    }
  }
  fill_scene(run, in, b.supplied, b.low_light, warn, collect);
  if (!b.supplied && run.get("scene.path") == nullptr && run.get("scene.evimo2.sequence") == nullptr) {
    fail(DiagKind::kRequired, "scene.path or scene.evimo2.sequence is required (or a bundle given to the run)",
         "scene.path", "", "", {"scene.evimo2.sequence"});
  }
  fill_run(run, cams, b, warn, out.facts, collect);  // reads no scene key
  if (collect != nullptr && !collect->empty()) return out;
  // Scanned rooms (scans.md §3.4): reference photometry solves away a common scale when every light is a proxy.
  if (b.baked.has_baked && b.baked.all_lights_proxy && !b.baked.lit_emissive) {
    for (size_t j = 0; j < cams.size(); ++j) {
      if (std::get<std::string>(*cams[j].get("photometry.mode")) == "reference") {
        warn(DiagKind::kInert, "scene.baked_scale inert under reference photometry: every light is a proxy and no lit "
             "material emits; the solved photocurrent scale cancels the common multiplier", "scene.baked_scale",
             b.cameras[sel[j]].name, run.layer("scene.baked_scale"), {"photometry.mode"});
      }
    }
  }
  if (b.baked.has_baked && b.baked.has_proxies && std::get<bool>(*run.get("accuracy.held_world"))) {
    warn(DiagKind::kRunLevel, "accuracy.held_world: held movers stay separate trace geometries in this scene "
         "(baked receivers and proxy lights)", "accuracy.held_world", "", run.layer("accuracy.held_world"));
  }
  derive_vignetting_law(run, cams);  // after fill_run (above; P2-F F13-min)
  // Noise leaves against fast poles (§2.4).
  for (size_t j = 0; j < cams.size(); ++j) {
    const double fast = std::min(cams[j].num("sensor.tau_o"), cams[j].num("sensor.tau_sf"));
    if (camera_sensor(cams[j]).noise_on() && fast < kFastPoleS) {
      warn(DiagKind::kDerivedConflict,
           "accuracy.noise_leaf_level = " + format_value(*run.get("accuracy.noise_leaf_level")) + ": camera " +
               b.cameras[sel[j]].name + " has min(tau_o, tau_sf) = " + format_double(fast) +
               " s < 25 us, short against the noise-tree leaves",
           "accuracy.noise_leaf_level", b.cameras[sel[j]].name, run.layer("accuracy.noise_leaf_level"),
           {"sensor.tau_o", "sensor.tau_sf"});
    }
  }
  // The cascade's output node against the photoreceptor loop (F042; SENSOR.md "Cascade validity"): above
  // k I tau_o = 0.25 the coupled loop's bandwidth and bright noise differ from the cascade's. I is the photocurrent the
  // resolver knows: reference photometry's median, relative photometry's scale (at unit renderer radiance); physical
  // photometry's luminance is known only at render.
  for (size_t j = 0; j < cams.size(); ++j) {
    const Table& t = cams[j];
    const std::string mode = std::get<std::string>(*t.get("photometry.mode"));
    if (camera_sensor(t).photoreceptor != sensor::Photoreceptor::kCascade || mode == "physical") continue;
    const bool ref = mode == "reference";
    const double i_fA = t.num(ref ? "photometry.reference_current_fA" : "photometry.illuminance_scale");
    const double k_tau = t.num("sensor.k") * t.num("sensor.tau_o") * 1e-15;  // per fA
    if (i_fA * k_tau > 0.25) {
      const std::string current = ref ? "photometry.reference_current_fA" : "photometry.illuminance_scale";
      warn(DiagKind::kDerivedConflict,
           "cameras." + b.cameras[sel[j]].name + ": k I tau_o = " + fmt3(i_fA * k_tau) + " > 0.25 at I = " +
               fmt3(i_fA) + " fA (" + (ref ? current : current + ", unit radiance") + "): above " +
               fmt3(0.25 / k_tau) + " fA the cascade's bandwidth and bright noise are approximate (tau_o sits inside "
               "the photoreceptor loop; sigma_o is a phenomenological floor)",
           "sensor.tau_o", b.cameras[sel[j]].name, t.layer("sensor.tau_o"), {"sensor.k", current});
    }
  }
  soft_and_inert(run, run, "", warn);
  for (size_t j = 0; j < cams.size(); ++j) soft_and_inert(cams[j], run, b.cameras[sel[j]].name, warn);
  flagged_defaults(cams, b, sel, warn);
  // The rate limiter (sensor/limiter.h, SENSOR.md "Rate limiter"): the driver caps the whole frame's stream.
  const auto* crop = std::get_if<std::vector<int64_t>>(run.get("run.crop_px"));
  for (size_t j = 0; j < cams.size(); ++j) {
    if (!camera_sensor(cams[j]).limiter_on()) continue;
    const BundleCamera& bc = b.cameras[sel[j]];
    const std::string where = "cameras." + bc.name + ".sensor.limiter = epoch";
    if (crop && !((*crop)[0] == 0 && (*crop)[1] == 0 && (*crop)[2] == bc.width && (*crop)[3] == bc.height)) {
      warn(DiagKind::kCrossKey,
           where + " under run.crop_px = " + format_value(*run.get("run.crop_px")) +
               ": it caps the crop's stream, which lacks the rest of the frame's load (the driver caps the whole "
               "frame's)",
           "sensor.limiter", bc.name, cams[j].layer("sensor.limiter"), {"run.crop_px"});
    }
  }
  // Every camera's SensorParams (§2.4: the sensor cross-key rules), then the run.
  for (size_t j = 0; j < cams.size(); ++j) {
    sensor::SensorParams sp = camera_sensor(cams[j]);
    for (const Param& r : sensor::param_rows()) {
      if (r.scope != Scope::kCamera) r.set(r, &sp, *run.get(r.key));
    }
    const std::string& cam = b.cameras[sel[j]].name;
    try {
      sp.validate();
    } catch (const DiagnosticError& e) {  // a cross-key rule naming its keys (sensor/params.cpp)
      Diagnostic d = e.diag;
      d.message = "cameras." + cam + ": " + e.what();
      d.camera = cam;
      d.layer = cams[j].layer(d.key);
      fail(std::move(d));
    } catch (const std::invalid_argument& e) {
      fail(DiagKind::kHardRange, "cameras." + cam + ": " + e.what(), "", cam);
    }
  }
  check_imu_cameras(cams, b, sel, warn);
  validate_run(run, b, sel, out);
  out.run = ordered(run, false);
  for (size_t j = 0; j < cams.size(); ++j) out.cameras.emplace_back(b.cameras[sel[j]].name, ordered(cams[j], true));
  for (const auto& cam : out.cameras) out.digests.push_back(digests(out, cam.second, b));
  return out;
}

namespace {

// The row of `key` for explain and relevant: unknown keys, a camera key without its camera, a run key with one (unless
// `run_in_camera`: relevant reads a run row in a camera's context) and a camera the record does not hold are refused.
const Param& row_for(std::string_view key, const Resolved& r, std::string_view camera, const char* what,
                     bool run_in_camera = false) {
  const Param* p = find(key);
  const std::string k(key), cam(camera);
  if (p == nullptr) {
    const Param* near = nearest(key, !camera.empty());
    Diagnostic d = diag(DiagKind::kUnknownKey, Severity::kError,
                        std::string(what) + ": unknown key " + k + suggest(near), k, cam, "", {});
    if (near != nullptr) d.nearest = near->key;
    fail(std::move(d));
  }
  if ((p->scope == Scope::kCamera) == camera.empty() && !(run_in_camera && !camera.empty())) {
    fail(DiagKind::kWrongScope,
         std::string(what) + ": " + k + (camera.empty() ? " is a camera key: name the camera" : " is not a camera key"),
         k, cam);
  }
  const auto named = [&](const auto& c) { return c.first == cam; };
  if (!camera.empty() && std::none_of(r.cameras.begin(), r.cameras.end(), named)) {
    fail(DiagKind::kUnknownCamera, std::string(what) + ": " + cam + " is not a resolved camera", k, cam);
  }
  return *p;
}

}  // namespace

std::vector<LayerEntry> explain(const Inputs& in, const Resolved& r, std::string_view key, std::string_view camera) {
  const Param& p = row_for(key, r, camera, "explain");
  const std::string cam(camera);
  const Entry* resolved = r.find(key, camera);
  std::vector<LayerEntry> out;
  if (p.def == Default::kLiteral) {
    out.push_back({"default", p.value, p.status ? name(*p.status) : "", p.source});
  } else if (p.def == Default::kFamily) {
    const Entry* fam = r.find("sensor.family", camera);
    if (fam != nullptr && std::holds_alternative<std::string>(fam->value)) {
      const std::string& family = std::get<std::string>(fam->value);
      if (const sensor::Prior* v = sensor::find_prior(family, p.key)) {
        out.push_back({"prior:" + family, v->value, name(v->status), v->source});
      }
    }
  }
  if (p.scope != Scope::kCamera) {
    const Entry* prof = r.find("accuracy.profile");
    const AccuracyProfile* ap =
        prof != nullptr && std::holds_alternative<std::string>(prof->value)
            ? find_accuracy_profile(std::get<std::string>(prof->value))
            : nullptr;
    for (const AccuracyEntry& e : ap != nullptr ? ap->entries : std::vector<AccuracyEntry>{}) {
      if (e.key != p.key) continue;
      const std::string layer = "accuracy:" + ap->name;
      // A derivation's value is the record's when it won; a literal entry is its value.
      const bool won = resolved != nullptr && resolved->layer == layer;
      Value v = e.formula.empty() ? e.value : (won ? resolved->value : Value{});
      out.push_back({layer, std::move(v), "derived", e.formula});
    }
  }
  if (resolved != nullptr && (resolved->layer == "derived" || resolved->layer == "bundle")) {
    out.push_back({resolved->layer, resolved->value, resolved->status, resolved->note});
  }
  const Parsed P = parse(in, true);
  for (const Layer* L : chain(P, cam)) {
    const Found f = lookup({L}, cam, p.key);
    if (f.set == nullptr) continue;
    out.push_back({L->record, f.set->value, L->user ? "user" : f.set->status,
                   f.set->source.empty() ? L->source : f.set->source});
  }
  return out;
}

bool relevant(std::string_view key, const Resolved& r, std::string_view camera) {
  const Param& p = row_for(key, r, camera, "relevant", true);
  const auto get = [&](std::string_view k) -> const Value* {
    if (!camera.empty()) {
      if (const Entry* e = r.find(k, camera)) return &e->value;
    }
    const Entry* e = r.find(k);
    return e != nullptr ? &e->value : nullptr;
  };
  return failing(p, get) == nullptr;
}

std::vector<std::pair<std::string, std::vector<std::string>>> derived_from(std::string_view profile) {
  return derivation_table(profile);
}

}  // namespace balla::config
