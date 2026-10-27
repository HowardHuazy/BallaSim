// The frames' rows and config::resolve_frames (docs/design/phase3/frames.md §3.2, §8.1 U11; P3 FR-1b): 43 rows of
// stage none that move no digest, the camera section `frames`, resolve_section by name, and the frames values of events
// and frames-only cameras with their refusals. The Python half of U11 (run.json, profiles, the commands) is
// tests/python/test_frames_resolve.py.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "config/diagnostics.h"
#include "config/resolve.h"
#include "config/resolve_frames.h"
#include "config/schema.h"
#include "core/time/time.h"
#include "frames/aps_frames.h"

using namespace balla;
using config::Value;

namespace {

using Map = std::vector<std::pair<std::string, config::Node>>;
using Ints = std::vector<int64_t>;
// Not inlined: GCC 14 reports a false -Wfree-nonheap-object for the variant's destructor when these are.
[[gnu::noinline]] config::Node leaf(Value v) {
  config::Node n;
  n.leaf = std::move(v);
  return n;
}
[[gnu::noinline]] config::Node mapping(Map m) {
  config::Node n;
  n.is_map = true;
  n.map = std::move(m);
  return n;
}
[[gnu::noinline]] config::Node str(const char* s) { return leaf(std::string(s)); }
[[gnu::noinline]] config::Node strings(std::vector<std::string> v) { return leaf(std::move(v)); }
config::Node integer(int64_t i) { return leaf(i); }
config::Node ints(Ints v) { return leaf(std::move(v)); }
config::Node sets(Map m) { return mapping(std::move(m)); }
const config::Node kNoSets;

// A run configuration of two prophesee_gen3 cameras: `defaults` are the keys of cameras.default, `blocks` extra keys
// per camera (a second block of the same name merges), `extra` more root sections. Without `both` the camera right
// has no block at all, so no sensor family.
config::Inputs run_inputs(Map defaults = {}, std::vector<std::pair<std::string, Map>> blocks = {}, Map extra = {},
                          bool both = true) {
  const Map sensor{{"sensor.family", str("prophesee_gen3")}, {"photometry.illuminance_scale", leaf(200.0)}};
  Map cams{{"default", mapping(std::move(defaults))}, {"left", mapping(sensor)}};
  if (both) cams.emplace_back("right", mapping(sensor));
  for (auto& [name, keys] : blocks) cams.emplace_back(name, mapping(std::move(keys)));
  Map doc{{"format", str("ballasim.run/2")}, {"output", mapping({{"dir", str("out")}})},
          {"cameras", mapping(std::move(cams))}};
  for (auto& e : extra) doc.push_back(std::move(e));
  config::Inputs in;
  in.config = mapping(std::move(doc));
  return in;
}

// Frames on in reference mode for both cameras, with `more` keys of cameras.default.
Map reference(Map more = {}) {
  Map m{{"frames.mode", str("reference")}};
  for (auto& e : more) m.push_back(std::move(e));
  return m;
}

// Left is simulated for events, right (no sensor family) only gets frames.
config::Inputs split_inputs(Map defaults = {}, std::vector<std::pair<std::string, Map>> blocks = {}) {
  return run_inputs(std::move(defaults), std::move(blocks),
                    {{"run", mapping({{"select", strings({"left"})}, {"frames_select", strings({"right"})}})}}, false);
}

// A supplied bundle with 64 x 48 cameras left (offset 7 ns) and right (offset 9 ns).
config::BundleInfo bundle() {
  config::BundleInfo b;
  b.cameras = {{"left", "prophesee_gen3", 64, 48, 7, 15e-6, 2.0}, {"right", "prophesee_gen3", 64, 48, 9, 15e-6, 2.0}};
  b.t_end_ns = 10 * time::kWindowNs;
  b.supplied = true;
  b.bundle_digest = std::string(64, 'b');
  b.engine_digest = std::string(64, 'e');
  return b;
}

// By value: a reference into a temporary list would dangle (and GCC warns even where it does not).
config::Entry entry(const config::Entries& list, const std::string& key) {
  for (const auto& [k, e] : list) {
    if (k == key) return e;
  }
  FAIL("no entry " << key);
  return {};
}

// The message and code of the FramesRefusal that f throws.
struct Refusal {
  std::string message, code;
};
Refusal refusal(const std::function<void()>& f) {
  try {
    f();
  } catch (const config::FramesRefusal& e) {
    return {e.what(), e.code};
  }
  FAIL("no FramesRefusal");
  return {};
}
Refusal refuses_inputs(const config::Inputs& in, const config::Node& set_items = kNoSets) {
  return refusal([&] { config::resolve_frames(in, bundle(), set_items); });
}
Refusal refuses_set(const std::string& key, config::Node value, Map defaults = reference()) {
  return refusal([&] { config::resolve_frames(run_inputs(defaults), bundle(), sets({{key, std::move(value)}})); });
}

bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

}  // namespace

TEST_CASE("the frames rows: 40 camera rows and 3 run rows, stage none, no target, relevant while frames are on") {
  std::vector<const config::Param*> camera, run;
  for (const config::Param& p : config::schema()) {
    if (p.key.rfind("frames.", 0) == 0) camera.push_back(&p);
    if (p.key == "execution.frames_device" || p.key == "execution.frames_texture_mb" || p.key == "run.frames_select") {
      run.push_back(&p);
    }
  }
  CHECK(camera.size() == 40);
  CHECK(run.size() == 3);
  for (const config::Param* p : camera) {
    INFO(p->key);
    CHECK(p->scope == config::Scope::kCamera);
    CHECK(p->stage == config::Stage::kNone);
    CHECK(p->target == config::Target::kNone);
    if (p->key == "frames.mode") {
      CHECK(p->when.empty());
    } else {
      REQUIRE(p->when.size() == 1);
      CHECK(p->when[0].key == "frames.mode");
    }
    if (p->key.rfind("frames.aps.", 0) == 0) CHECK(p->cls == config::Class::kTSensor);
  }
  for (const config::Param* p : run) {
    CHECK(p->scope == config::Scope::kRun);
    CHECK(p->stage == config::Stage::kNone);
    CHECK(p->when.empty());
  }
  CHECK(config::find("frames.mode")->choices == std::vector<std::string>{"off", "reference", "matched", "full"});
  CHECK(std::get<std::string>(config::find("frames.mode")->value) == "off");
  CHECK(std::get<int64_t>(config::find("execution.frames_texture_mb")->value) == 4096);
  CHECK(std::get<std::vector<std::string>>(config::find("run.frames_select")->value).empty());
  // The APS rows repeat frames::ApsSettings (aps_frames.h): the defaults of the two cannot drift apart.
  const frames::ApsSettings aps;
  const auto value = [](const char* key) { return config::find(key)->value; };
  CHECK(std::get<int64_t>(value("frames.aps.bits")) == aps.bits);
  CHECK(std::get<double>(value("frames.aps.full_well_e")) == aps.full_well_e);
  CHECK(std::get<double>(value("frames.aps.read_noise_e")) == aps.read_noise_e);
  CHECK(std::get<double>(value("frames.aps.dark_current_e_s")) == aps.dark_current_e_s);
  CHECK(std::get<double>(value("frames.aps.prnu")) == aps.prnu);
  CHECK(std::get<double>(value("frames.aps.dsnu_e")) == aps.dsnu_e);
  CHECK(std::get<double>(value("frames.aps.offset_dn")) == aps.offset_dn);
  CHECK(std::holds_alternative<std::monostate>(value("frames.aps.gain_dn_per_e")) == !aps.gain_dn_per_e.has_value());
  CHECK(std::holds_alternative<std::monostate>(value("frames.aps.seed")) == !aps.seed.has_value());
}

TEST_CASE("frames in a configuration: camera blocks accept the section, the root refuses it, no digest moves") {
  const config::Resolved base = config::resolve(run_inputs(), bundle());
  CHECK(std::get<std::string>(base.find("frames.mode", "left")->value) == "off");
  CHECK(std::get<double>(base.find("frames.fps", "right")->value) == 30.0);
  // Everything set at once, the nested form as in a YAML file, moves no digest of any stage (stage none).
  const config::Node all = mapping({{"mode", str("reference")}, {"fps", leaf(60.0)}, {"shutter", str("rolling")},
                                    {"row_dt_ns", integer(20000)}, {"aps", mapping({{"bits", integer(12)}})},
                                    {"ref", mapping({{"k_max", integer(64)}})}});
  const config::Resolved on = config::resolve(run_inputs({{"frames", all}}), bundle());
  CHECK(on.digests == base.digests);
  CHECK(std::get<std::string>(on.find("frames.mode", "right")->value) == "reference");
  CHECK(std::get<int64_t>(on.find("frames.aps.bits", "left")->value) == 12);
  CHECK(std::get<int64_t>(on.find("frames.ref.k_max", "left")->value) == 64);
  const config::Resolved exec =
      config::resolve(run_inputs({}, {}, {{"execution", mapping({{"frames_device", str("cpu")}})}}), bundle());
  CHECK(exec.digests == base.digests);
  CHECK(std::get<std::string>(exec.find("execution.frames_device")->value) == "cpu");
  // A root-level frames section is refused like sensor:.
  const config::Inputs root = run_inputs({}, {}, {{"frames", mapping({{"mode", str("reference")}})}});
  try {
    config::resolve(root, bundle());
    FAIL("no error");
  } catch (const config::DiagnosticError& e) {
    CHECK(e.diag.kind == config::DiagKind::kWrongScope);
    CHECK(has(e.what(), "frames: camera sections go under cameras.default or cameras.<camera>"));
  }
  // Values set under mode off are inert, with a warning; hard ranges refuse.
  const config::Resolved inert = config::resolve(run_inputs({{"frames.fps", leaf(60.0)}}), bundle());
  CHECK(std::any_of(inert.warnings.begin(), inert.warnings.end(),
                    [](const std::string& w) { return has(w, "frames.fps is set but inert"); }));
  CHECK_THROWS_AS(config::resolve(run_inputs({{"frames.stride", integer(0)}}), bundle()), config::DiagnosticError);
}

TEST_CASE("resolve_section: a section of any bundle camera, as resolve() gives it") {
  const config::Inputs in = run_inputs({{"frames.mode", str("reference")}}, {{"right", {{"frames.fps", leaf(24.0)}}}});
  const config::Resolved full = config::resolve(in, bundle());
  for (const char* camera : {"left", "right"}) {
    const config::Entries section = config::resolve_section(in, bundle(), camera, "frames");
    REQUIRE(section.size() == 40);
    size_t i = 0;
    for (const auto& [key, e] : full.cameras[camera == std::string("left") ? 0 : 1].second) {
      if (key.rfind("frames.", 0) != 0) continue;
      CHECK(section[i].first == key);
      CHECK(section[i].second.value == e.value);
      CHECK(section[i].second.layer == e.layer);
      CHECK(section[i].second.status == e.status);
      ++i;
    }
    CHECK(i == 40);
  }
  CHECK(entry(config::resolve_section(in, bundle(), "right", "frames"), "frames.fps").layer == "config");
  // The camera block beats cameras.default (D10); a CLI item beats both.
  config::Inputs cli = in;
  cli.cli = mapping({{"cameras.right.frames.fps", leaf(48.0)}});
  const config::Entry& fps = entry(config::resolve_section(cli, bundle(), "right", "frames"), "frames.fps");
  CHECK(std::get<double>(fps.value) == 48.0);
  CHECK(fps.layer == "cli");
  // Warnings of the explicit values; an unknown camera; a section whose rows need the camera in full.
  std::vector<config::Diagnostic> warnings;
  config::resolve_section(run_inputs({{"frames.fps", leaf(5000.0)}}), bundle(), "left", "frames", &warnings);
  REQUIRE(!warnings.empty());
  CHECK(warnings[0].kind == config::DiagKind::kSoftRange);
  CHECK(warnings[0].camera == "left");
  CHECK_THROWS_AS(config::resolve_section(in, bundle(), "nope", "frames"), config::DiagnosticError);
  CHECK_THROWS_AS(config::resolve_section(in, bundle(), "left", "sensor"), std::logic_error);
  CHECK_THROWS_AS(config::resolve_section(in, bundle(), "left", "scene"), std::invalid_argument);
}

TEST_CASE("resolve_frames: events cameras equal the full resolver, the run rows, the ROI and the clock") {
  const config::Inputs in = run_inputs(reference(), {{"left", {{"frames.exposure_ns", integer(5000000)}}}});
  config::FramesRun run;
  const std::vector<config::FramesCamera> cams = config::resolve_frames(in, bundle(), kNoSets, &run);
  REQUIRE(cams.size() == 2);
  const config::Resolved full = config::resolve(in, bundle());
  for (size_t i = 0; i < 2; ++i) {
    CHECK(cams[i].events);
    CHECK(cams[i].bundle_index == static_cast<int32_t>(i));
    CHECK(cams[i].clock_offset_ns == (i == 0 ? 7 : 9));
    CHECK(cams[i].clock_drift_ppm == 0.0);
    REQUIRE(cams[i].values.size() == 40);
    for (const auto& [key, e] : cams[i].values) {
      if (key == "frames.roi_px") continue;
      const config::Entry* r = full.find(key, cams[i].name);
      REQUIRE(r != nullptr);
      CHECK(e.value == r->value);
      CHECK(e.layer == r->layer);
    }
    // The verdict contrast is the smaller threshold of the resolved sensor.
    const auto theta = [&](const char* key) { return std::get<double>(full.find(key, cams[i].name)->value); };
    REQUIRE(cams[i].contrast.has_value());
    CHECK(*cams[i].contrast == doctest::Approx(std::min(theta("sensor.theta_on"), theta("sensor.theta_off"))));
  }
  CHECK(std::get<int64_t>(entry(cams[0].values, "frames.exposure_ns").value) == 5000000);
  CHECK(std::holds_alternative<std::monostate>(entry(cams[1].values, "frames.exposure_ns").value));
  // ROI null: the full frame, derived; with run.crop_px, the crop; an explicit one is kept.
  const config::Entry& roi = entry(cams[0].values, "frames.roi_px");
  CHECK(std::get<Ints>(roi.value) == Ints{0, 0, 64, 48});
  CHECK(roi.layer == "derived");
  const auto crop = config::resolve_frames(
      run_inputs(reference(), {}, {{"run", mapping({{"crop_px", ints({8, 8, 40, 40})}})}}), bundle(), kNoSets);
  CHECK(std::get<Ints>(entry(crop[0].values, "frames.roi_px").value) == Ints{8, 8, 40, 40});
  CHECK(entry(crop[0].values, "frames.roi_px").note == "run.crop_px");
  const auto explicit_roi =
      config::resolve_frames(run_inputs(reference({{"frames.roi_px", ints({1, 2, 3, 4})}})), bundle(), kNoSets);
  CHECK(entry(explicit_roi[1].values, "frames.roi_px").layer == "config");
  // The run rows are at their defaults.
  CHECK(std::get<std::string>(entry(run.values, "execution.frames_device").value) == "auto");
  CHECK(std::get<int64_t>(entry(run.values, "execution.frames_texture_mb").value) == 4096);
  CHECK(std::get<std::vector<std::string>>(entry(run.values, "run.frames_select").value).empty());
  CHECK(run.values.size() == 3);
  // A contrast override replaces the sensor's.
  const auto c =
      config::resolve_frames(run_inputs(reference({{"frames.full.contrast_ref", leaf(0.4)}})), bundle(), kNoSets);
  CHECK(*c[0].contrast == 0.4);
}

TEST_CASE("resolve_frames: --set items are the top layer, in five forms") {
  const config::Inputs in = run_inputs(reference());
  const auto cams = config::resolve_frames(
      in, bundle(), sets({{"cameras.default.frames.fps", leaf(60.0)}, {"cameras.right.frames.fps", leaf(24.0)},
                          {"cameras.left.frames.aps.bits", integer(12)}, {"execution.frames_device", str("cpu")},
                          {"execution.frames_texture_mb", integer(0)}}));
  CHECK(std::get<double>(entry(cams[0].values, "frames.fps").value) == 60.0);
  CHECK(std::get<double>(entry(cams[1].values, "frames.fps").value) == 24.0);
  CHECK(entry(cams[1].values, "frames.fps").layer == "cli");
  CHECK(std::get<int64_t>(entry(cams[0].values, "frames.aps.bits").value) == 12);
  CHECK(std::get<int64_t>(entry(cams[1].values, "frames.aps.bits").value) == 10);
  // A set replaces the run's own CLI item of the same key (later wins); the run's other items stay.
  config::Inputs recorded = in;
  recorded.cli = mapping({{"cameras.default.frames.fps", leaf(15.0)}, {"cameras.default.frames.stride", integer(3)}});
  const auto over = config::resolve_frames(recorded, bundle(), sets({{"cameras.default.frames.fps", leaf(60.0)}}));
  CHECK(std::get<double>(entry(over[0].values, "frames.fps").value) == 60.0);
  CHECK(std::get<int64_t>(entry(over[0].values, "frames.stride").value) == 3);
  // A top-layer cameras.default item beats a recorded camera-specific item of the same key (the later layer wins, the
  // camera's block beats the default block only within one layer); another camera's and another key's items stay.
  config::Inputs camera_recorded = in;
  camera_recorded.cli = mapping({{"cameras.left.frames.fps", leaf(15.0)}, {"cameras.right.frames.fps", leaf(20.0)},
                                 {"cameras.left.frames.stride", integer(3)}});
  const auto top =
      config::resolve_frames(camera_recorded, bundle(), sets({{"cameras.default.frames.fps", leaf(60.0)}}));
  CHECK(std::get<double>(entry(top[0].values, "frames.fps").value) == 60.0);
  CHECK(std::get<double>(entry(top[1].values, "frames.fps").value) == 60.0);
  CHECK(std::get<int64_t>(entry(top[0].values, "frames.stride").value) == 3);
  // Within the top layer the camera's own item still beats the default one.
  const auto both = config::resolve_frames(camera_recorded, bundle(),
                                            sets({{"cameras.default.frames.fps", leaf(60.0)},
                                                  {"cameras.left.frames.fps", leaf(24.0)}}));
  CHECK(std::get<double>(entry(both[0].values, "frames.fps").value) == 24.0);
  CHECK(std::get<double>(entry(both[1].values, "frames.fps").value) == 60.0);
  // Without an item for the key the recorded camera-specific ones stand.
  const auto kept =
      config::resolve_frames(camera_recorded, bundle(), sets({{"cameras.default.frames.stride", integer(5)}}));
  CHECK(std::get<double>(entry(kept[0].values, "frames.fps").value) == 15.0);
  CHECK(std::get<double>(entry(kept[1].values, "frames.fps").value) == 20.0);
  CHECK(std::get<int64_t>(entry(kept[0].values, "frames.stride").value) == 5);
  // run.frames_select is a run row.
  const auto split = config::resolve_frames(split_inputs(reference()), bundle(),
                                            sets({{"run.frames_select", strings({"right"})}}));
  REQUIRE(split.size() == 2);
  CHECK(!split[1].events);
  // Refusals: every other form is FRM-SET-SCOPE, a camera the bundle lacks FRM-CAMERA.
  for (const char* bad : {"run.render_seed", "accuracy.n_subsamples", "cameras.left.sensor.theta_on",
                          "cameras.left.frames.aps", "cameras.left.frames", "cameras.left", "execution.render_threads",
                          "frames.mode"}) {
    INFO(bad);
    const Refusal r = refuses_set(bad, integer(1));
    CHECK(r.code == "FRM-SET-SCOPE");
    CHECK(has(r.message, std::string("--set ") + bad));
  }
  const Refusal near = refuses_set("cameras.left.frames.fpz", leaf(1.0));
  CHECK(near.code == "FRM-SET-SCOPE");
  CHECK(has(near.message, "did you mean frames.fps?"));
  const Refusal lft = refuses_set("cameras.lft.frames.fps", leaf(1.0));
  CHECK(lft.code == "FRM-CAMERA");
  CHECK(has(lft.message, "cameras.lft: not a camera of the bundle (left, right)"));
  // A frames value outside its row's range, or of the wrong type, is FRM-VALUE with the resolver's message.
  const Refusal range = refuses_set("cameras.left.frames.stride", integer(0));
  CHECK(range.code == "FRM-VALUE");
  CHECK(has(range.message, "cameras.left.frames.stride = 0: outside [1, inf)"));
  CHECK(refuses_set("cameras.left.frames.aps.bits", integer(7)).code == "FRM-VALUE");
  CHECK(refuses_set("cameras.left.frames.mode", integer(7)).code == "FRM-VALUE");
  // Another key's error stays the resolver's.
  CHECK_THROWS_AS(config::resolve_frames(run_inputs(reference({{"sensor.theta_on", leaf(-1.0)}})), bundle(), kNoSets),
                  config::DiagnosticError);
}

TEST_CASE("resolve_frames: FRM-VALUE for what no row expresses, FRM-MODE-SOURCE for Blender modes") {
  const auto refused = [](Map more) { return refuses_inputs(run_inputs(reference(std::move(more)))); };
  const auto accepted = [](Map more) {
    CHECK_NOTHROW(config::resolve_frames(run_inputs(reference(std::move(more))), bundle(), kNoSets));
  };
  const Refusal ss = refused({{"frames.supersample", integer(3)}});
  CHECK(ss.code == "FRM-VALUE");
  CHECK(has(ss.message, "cameras.left.frames.supersample = 3: expected 2, 4 or 8"));
  for (const int64_t ok : {2, 4, 8}) accepted({{"frames.supersample", integer(ok)}});
  const Refusal foo = refused({{"frames.outputs", strings({"mono", "foo"})}});
  CHECK(foo.code == "FRM-VALUE");
  CHECK(has(foo.message, "'foo' is not a dataset"));
  const Refusal rgb = refused({{"frames.outputs", strings({"rgb", "mono"})}});
  CHECK(rgb.code == "FRM-VALUE");
  CHECK(has(rgb.message, "'rgb' is not an output of reference mode"));
  CHECK(refused({{"frames.outputs", strings({"groups"})}}).code == "FRM-VALUE");
  accepted({{"frames.outputs", strings({"mono", "reference", "aps"})}});
  const std::vector<Ints> bad_rois{{8, 4, 8, 20}, {8, 20, 40, 4}, {0, 0, 65, 48}, {0, 0, 64, 49}};
  for (const Ints& roi : bad_rois) {
    const Refusal bad = refused({{"frames.roi_px", ints(roi)}});
    CHECK(bad.code == "FRM-VALUE");
    CHECK(has(bad.message, "frames.roi_px"));
  }
  accepted({{"frames.roi_px", ints({0, 0, 64, 48})}});
  // Mode off checks nothing of the section (it is inert).
  CHECK_NOTHROW(config::resolve_frames(
      run_inputs({{"frames.supersample", integer(3)}, {"frames.outputs", strings({"foo"})}}), bundle(), kNoSets));
  // matched and full need a Blender exchange: another known source kind refuses; an unknown one passes.
  config::Inputs in = run_inputs({{"frames.mode", str("matched")}});
  CHECK_NOTHROW(config::resolve_frames(in, bundle(), kNoSets));
  in.source_kind = "blender";
  CHECK_NOTHROW(config::resolve_frames(in, bundle(), kNoSets));
  for (const char* kind : {"evimo2", "yaml", "bundle"}) {
    in.source_kind = kind;
    const Refusal m = refuses_inputs(in);
    CHECK(m.code == "FRM-MODE-SOURCE");
    CHECK(has(m.message, std::string("this run's source is '") + kind + "'"));
  }
  in.config = run_inputs(reference()).config;
  in.source_kind = "evimo2";
  CHECK_NOTHROW(config::resolve_frames(in, bundle(), kNoSets));
}

TEST_CASE("resolve_frames: check false is the run's record, which a --set item can still repair") {
  // Values no row refuses but a check does: the supersample and an output of the wrong mode.
  config::Inputs in = run_inputs(reference({{"frames.supersample", integer(3)}, {"frames.outputs", strings({"rgb"})}}));
  CHECK(refuses_inputs(in).code == "FRM-VALUE");
  config::FramesRun run;
  const auto record = config::resolve_frames(in, bundle(), kNoSets, &run, false);
  REQUIRE(record.size() == 2);                                                    // left and right: both selected
  CHECK(std::get<int64_t>(entry(record[0].values, "frames.supersample").value) == 3);
  CHECK(std::holds_alternative<std::monostate>(entry(record[0].values, "frames.roi_px").value));   // no ROI resolution
  // The same layers with the repairing items pass the checks.
  CHECK_NOTHROW(config::resolve_frames(in, bundle(),
                                       sets({{"cameras.default.frames.supersample", integer(4)},
                                             {"cameras.default.frames.outputs", strings({"mono"})}})));
  // A recorded run.frames_select the checks refuse (a camera of run.select) is the record, not a refusal.
  config::Inputs split = run_inputs(reference(), {},
                                    {{"run", mapping({{"select", strings({"left"})},
                                                      {"frames_select", strings({"left", "nope"})}})}});
  CHECK(refuses_inputs(split).code == "FRM-CAMERA");
  const auto rec = config::resolve_frames(split, bundle(), kNoSets, &run, false);
  CHECK(rec.size() == 1);
  CHECK(std::get<std::vector<std::string>>(entry(run.values, "run.frames_select").value).size() == 2);
  CHECK(config::resolve_frames(split, bundle(), sets({{"run.frames_select", strings({"right"})}})).size() == 2);
  // The scope of the --set items is checked either way.
  CHECK(refusal([&] { config::resolve_frames(in, bundle(), sets({{"run.render_seed", integer(1)}}), nullptr, false); })
            .code == "FRM-SET-SCOPE");
}

TEST_CASE("resolve_frames: a frames-only camera needs no sensor family, and run.frames_select is checked") {
  config::FramesRun run;
  const auto cams = config::resolve_frames(split_inputs(reference(), {{"right", {{"frames.fps", leaf(24.0)}}}}),
                                           bundle(), kNoSets, &run);
  REQUIRE(cams.size() == 2);
  CHECK(cams[0].name == "left");
  CHECK(cams[0].events);
  CHECK(cams[1].name == "right");
  CHECK(!cams[1].events);
  CHECK(cams[1].bundle_index == 1);
  CHECK(cams[1].values.size() == 40);
  CHECK(std::get<double>(entry(cams[1].values, "frames.fps").value) == 24.0);
  CHECK(std::get<std::string>(entry(cams[1].values, "frames.mode").value) == "reference");  // cameras.default.frames
  CHECK(cams[1].clock_offset_ns == 9);  // the bundle camera's; no drift without a sensor
  CHECK(cams[1].clock_drift_ppm == 0.0);
  CHECK(cams[1].contrast == cams[0].contrast);  // the smallest C of the events cameras
  const auto& selected = entry(run.values, "run.frames_select").value;
  CHECK(std::get<std::vector<std::string>>(selected) == std::vector<std::string>{"right"});
  // The checks apply to it too.
  const Refusal ss = refuses_inputs(split_inputs(reference(), {{"right", {{"frames.supersample", integer(5)}}}}));
  CHECK(ss.code == "FRM-VALUE");
  // run.frames_select: a camera outside the bundle, a selected one, a duplicate.
  const auto with_select = [](std::vector<std::string> names) {
    config::Inputs in = split_inputs(reference());
    for (auto& [k, v] : in.config.map) {
      if (k != "run") continue;
      for (auto& [rk, rv] : v.map) {
        if (rk == "frames_select") rv = strings(names);
      }
    }
    return in;
  };
  const Refusal nope = refuses_inputs(with_select({"nope"}));
  CHECK(nope.code == "FRM-CAMERA");
  CHECK(has(nope.message, "run.frames_select: nope is not a camera of the bundle (left, right)"));
  const Refusal sel = refuses_inputs(with_select({"left"}));
  CHECK(sel.code == "FRM-CAMERA");
  CHECK(has(sel.message, "left is simulated for events"));
  const Refusal twice = refuses_inputs(with_select({"right", "right"}));
  CHECK(twice.code == "FRM-CAMERA");
  CHECK(has(twice.message, "names right twice"));
  // Without run.select every bundle camera is selected, so nothing is left for frames alone.
  const Refusal all =
      refuses_inputs(run_inputs(reference(), {}, {{"run", mapping({{"frames_select", strings({"right"})}})}}));
  CHECK(all.code == "FRM-CAMERA");
}

TEST_CASE("resolve_frames: digest_values are the canonical frames and execution lines without the APS") {
  const config::Inputs in = run_inputs(reference());
  const auto base = config::resolve_frames(in, bundle(), kNoSets);
  const std::string& text = base[0].digest_values;
  CHECK(text.rfind("execution.frames_device=\"auto\"\n", 0) == 0);  // sorted by key: execution... before frames...
  CHECK(has(text, "\nframes.mode=\"reference\"\n"));
  CHECK(has(text, "\nframes.roi_px=[0,0,64,48]\n"));  // the ROI is resolved
  CHECK(!has(text, "frames.aps."));
  CHECK(!has(text, "run.frames_select"));
  CHECK(text.back() == '\n');
  std::vector<std::string> keys;
  for (size_t pos = 0; pos < text.size(); pos = text.find('\n', pos) + 1) {
    keys.push_back(text.substr(pos, text.find('=', pos) - pos));
  }
  CHECK(std::is_sorted(keys.begin(), keys.end()));
  CHECK(keys.size() == 2 + 40 - 9);  // execution.frames_device and _texture_mb, the 31 non-APS frames rows
  // The APS values do not move it; any other frames value, and an execution one, does; the camera does not.
  const auto digest = [&](Map more) {
    return config::resolve_frames(run_inputs(reference(std::move(more))), bundle(), kNoSets)[0].digest_values;
  };
  CHECK(digest({{"frames.aps.bits", integer(12)}}) == text);
  CHECK(digest({{"frames.fps", leaf(60.0)}}) != text);
  CHECK(config::resolve_frames(in, bundle(), sets({{"execution.frames_device", str("cpu")}}))[0].digest_values != text);
  CHECK(base[1].digest_values == text);
}
