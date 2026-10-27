// config::resolve_frames (docs/design/phase3/frames.md §3.2): the frames values of the cameras of a run.
#include "config/resolve_frames.h"

#include <algorithm>
#include <set>
#include <string_view>
#include <variant>

#include "config/schema.h"

namespace balla::config {

namespace {

[[noreturn]] void refuse(const char* code, const std::string& message) { throw FramesRefusal(code, message); }

bool starts_with(std::string_view s, std::string_view prefix) { return s.substr(0, prefix.size()) == prefix; }

Entry* find_entry(Entries& list, std::string_view key) {
  for (auto& [k, e] : list) {
    if (k == key) return &e;
  }
  return nullptr;
}
const Entry* find_entry(const Entries& list, std::string_view key) {
  for (const auto& [k, e] : list) {
    if (k == key) return &e;
  }
  return nullptr;
}

bool is_frames_key(std::string_view key) {
  return starts_with(key, "frames.") || starts_with(key, "execution.frames_") || key == "run.frames_select";
}

std::string bundle_names(const BundleInfo& b) {
  std::string out;
  for (const BundleCamera& c : b.cameras) out += (out.empty() ? "" : ", ") + c.name;
  return out;
}

size_t edit_distance(std::string_view a, std::string_view b) {
  std::vector<size_t> d(b.size() + 1);
  for (size_t j = 0; j <= b.size(); ++j) d[j] = j;
  for (size_t i = 1; i <= a.size(); ++i) {
    size_t diag = d[0];
    d[0] = i;
    for (size_t j = 1; j <= b.size(); ++j) {
      const size_t up = d[j];
      d[j] = std::min({d[j] + 1, d[j - 1] + 1, diag + (a[i - 1] == b[j - 1] ? 0 : 1)});
      diag = up;
    }
  }
  return d[b.size()];
}

int bundle_index(const BundleInfo& b, std::string_view name) {
  for (size_t i = 0; i < b.cameras.size(); ++i) {
    if (b.cameras[i].name == name) return static_cast<int>(i);
  }
  return -1;
}

// The --set forms of the frames commands (§3.9): the five accepted shapes, and the cameras they name.
void check_sets(const Node& sets, const BundleInfo& b) {
  if (!sets.is_map) {
    if (std::holds_alternative<std::monostate>(sets.leaf)) return;
    refuse("FRM-SET-SCOPE", "--set items must be a mapping of full dotted keys to values");
  }
  const auto forms = [](const std::string& key) {
    return "--set " + key +
           ": the frames commands take cameras.<camera>.frames.<key>, cameras.default.frames.<key>, "
           "execution.frames_device, execution.frames_texture_mb and run.frames_select only";
  };
  for (const auto& [key, value] : sets.map) {
    if (key == "execution.frames_device" || key == "execution.frames_texture_mb" || key == "run.frames_select") {
      continue;
    }
    if (!starts_with(key, "cameras.")) refuse("FRM-SET-SCOPE", forms(key));
    const size_t dot = key.find('.', 8);
    if (dot == std::string::npos) refuse("FRM-SET-SCOPE", forms(key));
    const std::string camera = key.substr(8, dot - 8), rest = key.substr(dot + 1);
    const Param* row = starts_with(rest, "frames.") ? find(rest) : nullptr;
    if (row == nullptr || row->scope != Scope::kCamera) {
      std::string message = forms(key);
      if (starts_with(rest, "frames.")) {  // a key of the section that does not exist: the nearest row, if close
        const Param* best = nullptr;
        size_t best_d = 3;
        for (const Param& p : schema()) {
          if (p.scope == Scope::kCamera && starts_with(p.key, "frames.") && edit_distance(rest, p.key) < best_d) {
            best = &p;
            best_d = edit_distance(rest, p.key);
          }
        }
        message += "; there is no frames key " + rest + (best != nullptr ? " (did you mean " + best->key + "?)" : "");
      }
      refuse("FRM-SET-SCOPE", message);
    }
    if (camera != "default" && bundle_index(b, camera) < 0) {
      refuse("FRM-CAMERA",
             "--set " + key + ": cameras." + camera + ": not a camera of the bundle (" + bundle_names(b) + ")");
    }
  }
}

// The run's layers with the frames command's --set items as a layer above the run's own CLI items (later wins). The
// resolver keeps the camera's block above the default block within one layer (D10), so that an item of the top layer
// naming cameras.default.<key> must also beat a recorded cameras.<camera>.<key> of the layer below: those recorded
// items are dropped (an item of these sets for the same key replaces them again, below).
Inputs with_sets(const Inputs& in, const Node& sets) {
  Inputs out = in;
  if (!sets.is_map || sets.map.empty()) return out;
  const bool null_cli = !out.cli.is_map && std::holds_alternative<std::monostate>(out.cli.leaf);
  if (null_cli) {
    out.cli = Node{};
    out.cli.is_map = true;
  }
  if (!out.cli.is_map) return out;  // a malformed CLI layer: the resolver refuses it
  constexpr std::string_view kDefault = "cameras.default.";
  for (const auto& [key, value] : sets.map) {
    if (!starts_with(key, kDefault)) continue;
    const std::string_view rest = std::string_view(key).substr(kDefault.size());
    std::erase_if(out.cli.map, [&](const auto& kv) {
      const std::string_view k = kv.first;
      if (!starts_with(k, "cameras.") || starts_with(k, kDefault)) return false;
      const size_t dot = k.find('.', 8);
      return dot != std::string_view::npos && k.substr(dot + 1) == rest;
    });
  }
  for (const auto& [key, value] : sets.map) {
    const auto it =
        std::find_if(out.cli.map.begin(), out.cli.map.end(), [&](const auto& kv) { return kv.first == key; });
    if (it != out.cli.map.end()) it->second = value;
    else out.cli.map.emplace_back(key, value);
  }
  return out;
}

Entries frames_rows(const Entries& all) {
  Entries out;
  for (const auto& [k, e] : all) {
    if (starts_with(k, "frames.")) out.emplace_back(k, e);
  }
  return out;
}

double number(const Entries& list, std::string_view key) {
  const Entry* e = find_entry(list, key);
  if (e == nullptr) throw std::logic_error("resolve_frames: " + std::string(key) + " is not resolved");
  if (const auto* d = std::get_if<double>(&e->value)) return *d;
  return static_cast<double>(std::get<int64_t>(e->value));
}

std::string text_of(const Entries& list, std::string_view key) {
  const Entry* e = find_entry(list, key);
  const auto* text = e != nullptr ? std::get_if<std::string>(&e->value) : nullptr;
  return text != nullptr ? *text : std::string();
}

// The checks no row can express (FRM-VALUE), the ROI's resolution and the source rule, on one camera with frames on.
void check_camera(FramesCamera& cam, const BundleCamera& bc, const Entries& run_values, const Inputs& in) {
  const std::string where = "cameras." + cam.name + ".";
  const std::string mode = text_of(cam.values, "frames.mode");
  const auto value = [&](const char* key) -> const Value& { return find_entry(cam.values, key)->value; };
  // A null ROI is run.crop_px when set, else the full frame; always a definite region downstream.
  Entry* roi = find_entry(cam.values, "frames.roi_px");
  if (std::holds_alternative<std::monostate>(roi->value)) {
    const Entry* crop = find_entry(run_values, "run.crop_px");
    if (crop != nullptr && !std::holds_alternative<std::monostate>(crop->value)) {
      *roi = {crop->value, "derived", "derived", "run.crop_px"};
    } else {
      *roi = {std::vector<int64_t>{0, 0, bc.width, bc.height}, "derived", "derived", "the full frame"};
    }
  }
  if (mode == "off") return;
  if (mode == "matched" || mode == "full") {
    if (!in.source_kind.empty() && in.source_kind != "blender") {
      refuse("FRM-MODE-SOURCE", where + "frames.mode = '" + mode + "' needs a Blender exchange as the scene source "
                                "(scene.path naming an exchange directory); this run's source is '" + in.source_kind +
                                "': frames.mode = 'reference' is BallaSim's own frames for any source");
    }
  }
  const auto* ss = std::get_if<int64_t>(&value("frames.supersample"));
  if (ss != nullptr && *ss != 2 && *ss != 4 && *ss != 8) {
    refuse("FRM-VALUE", where + "frames.supersample = " + std::to_string(*ss) + ": expected 2, 4 or 8");
  }
  if (const auto* outs = std::get_if<std::vector<std::string>>(&value("frames.outputs"))) {
    static const std::set<std::string, std::less<>> kMembers{"rgb", "mono", "groups", "reference", "aps"};
    for (const std::string& o : *outs) {
      if (kMembers.count(o) == 0) {
        refuse("FRM-VALUE", where + "frames.outputs: '" + o + "' is not a dataset (rgb, mono, groups, reference, aps)");
      }
      if (mode == "reference" && (o == "rgb" || o == "groups")) {
        const char* why = o == "rgb" ? "(BallaSim renders mono; rgb is Blender's)"
                                     : "(its per-group means are waveform-weighted already; groups are the Blender "
                                       "modes' unweighted sub-renders)";
        refuse("FRM-VALUE", where + "frames.outputs: '" + o + "' is not an output of reference mode " + why);
      }
    }
  }
  const auto* box = std::get_if<std::vector<int64_t>>(&roi->value);
  if (box == nullptr || box->size() != 4 || !((*box)[0] < (*box)[2] && (*box)[1] < (*box)[3] && (*box)[0] >= 0 &&
                                              (*box)[1] >= 0 && (*box)[2] <= bc.width && (*box)[3] <= bc.height)) {
    refuse("FRM-VALUE", where + "frames.roi_px = " + format_value(roi->value) +
                            ": expected [x0, y0, x1, y1] with x0 < x1 and y0 < y1 inside the " +
                            std::to_string(bc.width) + " x " + std::to_string(bc.height) + " image");
  }
}

}  // namespace

std::vector<FramesCamera> resolve_frames(const Inputs& run_inputs, const BundleInfo& b, const Node& sets,
                                         FramesRun* run, bool check) {
  check_sets(sets, b);
  const Inputs in = with_sets(run_inputs, sets);
  const Resolved r = [&] {
    try {
      return resolve(in, b);
    } catch (const DiagnosticError& e) {  // a frames value of the wrong type or outside its row's range
      if ((e.diag.kind == DiagKind::kType || e.diag.kind == DiagKind::kHardRange) && is_frames_key(e.diag.key)) {
        refuse("FRM-VALUE", e.diag.message);
      }
      throw;
    }
  }();
  std::vector<FramesCamera> out;
  std::vector<Diagnostic> section_warnings;

  // The events cameras: the full resolver's tables (what run.json records), the frames rows of each.
  std::optional<double> smallest;
  for (const auto& [name, entries] : r.cameras) {
    FramesCamera c;
    c.name = name;
    c.bundle_index = bundle_index(b, name);
    c.events = true;
    c.values = frames_rows(entries);
    c.clock_offset_ns = std::get<int64_t>(find_entry(entries, "sensor.clock_offset_ns")->value);
    c.clock_drift_ppm = number(entries, "sensor.clock_drift_ppm");
    const double theta = std::min(number(entries, "sensor.theta_on"), number(entries, "sensor.theta_off"));
    smallest = smallest ? std::min(*smallest, theta) : theta;
    c.contrast = theta;
    out.push_back(std::move(c));
  }

  // The frames-only cameras: run.frames_select, outside run.select and in the bundle (FRM-CAMERA).
  std::vector<std::string> only;
  if (const Entry* e = r.find("run.frames_select"); e != nullptr && check) {
    if (const auto* names = std::get_if<std::vector<std::string>>(&e->value)) only = *names;
  }
  for (size_t i = 0; i < only.size(); ++i) {
    const std::string& name = only[i];
    if (bundle_index(b, name) < 0) {
      refuse("FRM-CAMERA", "run.frames_select: " + name + " is not a camera of the bundle (" + bundle_names(b) + ")");
    }
    if (std::any_of(r.cameras.begin(), r.cameras.end(), [&](const auto& cam) { return cam.first == name; })) {
      refuse("FRM-CAMERA", "run.frames_select: " + name +
                               " is simulated for events (run.select); name the cameras that get frames alone");
    }
    const auto earlier_end = only.begin() + static_cast<std::ptrdiff_t>(i);
    if (std::find(only.begin(), earlier_end, name) != earlier_end) {
      refuse("FRM-CAMERA", "run.frames_select names " + name + " twice");
    }
    FramesCamera c;
    c.name = name;
    c.bundle_index = bundle_index(b, name);
    c.events = false;
    c.values = resolve_section(in, b, name, "frames", &section_warnings);
    c.clock_offset_ns = b.cameras[static_cast<size_t>(c.bundle_index)].clock_offset_ns;
    c.clock_drift_ppm = 0.0;
    c.contrast = smallest;
    out.push_back(std::move(c));
  }

  // The run-scope rows, the contrast override, the checks and the digest lines.
  Entries run_values;
  for (const auto& [k, e] : r.run) {
    if (k == "run.crop_px" || is_frames_key(k)) run_values.emplace_back(k, e);
  }
  Entries frames_run;
  for (const auto& [k, e] : run_values) {
    if (k != "run.crop_px") frames_run.emplace_back(k, e);
  }
  for (FramesCamera& c : out) {
    const BundleCamera& bc = b.cameras[static_cast<size_t>(c.bundle_index)];
    if (check) check_camera(c, bc, run_values, in);
    if (const Entry* ref = find_entry(c.values, "frames.full.contrast_ref");
        ref != nullptr && std::holds_alternative<double>(ref->value)) {
      c.contrast = std::get<double>(ref->value);
    }
    std::vector<std::pair<std::string, std::string>> lines;
    for (const auto& [k, e] : c.values) {
      if (!starts_with(k, "frames.aps.")) lines.emplace_back(k, canonical_text(e.value));
    }
    for (const auto& [k, e] : frames_run) {
      if (starts_with(k, "execution.frames_")) lines.emplace_back(k, canonical_text(e.value));
    }
    std::sort(lines.begin(), lines.end());
    for (const auto& [k, text] : lines) c.digest_values += k + "=" + text + "\n";
  }
  if (run != nullptr) {
    run->values = std::move(frames_run);
    run->warnings.clear();
    for (const Diagnostic& d : r.diagnostics) {
      if (is_frames_key(d.key)) run->warnings.push_back(d);
    }
    for (Diagnostic& d : section_warnings) run->warnings.push_back(std::move(d));
  }
  return out;
}

}  // namespace balla::config
