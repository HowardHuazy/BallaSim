// The resolver (docs/dev/P0_PARAMS.md §2): a run is a stack of layers (schema literals, the family prior of each
// camera and the accuracy profile, profile files, the run configuration, the CLI) that resolve to one value per key
// and camera. Python reads the YAML files, expands path values and hands the documents over as plain values; this
// code parses and checks them (§2.4), resolves, derives (§1.3), validates every camera and the run, and records
// where each value comes from, with the per-camera stage digests (§1.2).
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "config/diagnostics.h"
#include "core/config/param.h"

namespace balla::config {

// A document as Python hands it over: a mapping (keys in document order) or a leaf (null, bool, int, float, string,
// or a list of numbers or of strings; an empty list is an empty float list until a row types it).
struct Node {
  bool is_map = false;
  Value leaf;
  std::vector<std::pair<std::string, Node>> map;
};

// What a run resolves from (§2.3).
struct Inputs {
  std::string name = "<dict>";            // the run configuration's file, for messages
  Node config;                            // the run configuration (format 2)
  Node cli;                               // the --set items: {full dotted key: value}; null: none; else refused
  std::vector<std::string> cli_profiles;  // the --profile ids, appended to the root list
  std::map<std::string, Node, std::less<>> profiles;  // the profile files the layers name, by id
  std::map<std::string, Value, std::less<>> facts;    // derived scene values from the environment (Python)
  // The scene source (blender.md §3.11): evimo2, bundle, yaml, blender or empty (unknown). Not a schema row, which
  // fill_scene would add to every run: config.load sets it from the resolved scene keys and resolves the scene again
  // for a Blender exchange, whose scene.blender.* rows are filled and whose physical photometry needs lm_per_w.
  std::string source_kind;
};

// A required key without a value, in collect mode (blender.md §3.11): its full key (cameras.<camera>.<key> for a camera
// key), the rest of the message resolve() would throw and the diagnostic it would carry (kRequired or kFamilyMissing;
// ui.md §3.10).
struct MissingKey {
  std::string key, hint;
  Diagnostic diag;
};
using Missing = std::vector<MissingKey>;

// The facts of the scene bundle that resolution needs.
struct BundleCamera {
  std::string name, sensor;  // sensor: the C1 camera's sensor hint (named by the D2 message only)
  int64_t width = 0, height = 0, clock_offset_ns = 0;
  double pixel_pitch_m = 0, f_number = 2.0;  // C1 (physical photometry; f_number: optics.f_number's default)
};
// Scanned-room facts of the rendered bundle (scans.md §3.4), all false for a pre-scan scene.
struct BakedFacts {
  bool has_baked = false, has_proxies = false, all_lights_proxy = false, lit_emissive = false;
};
struct BundleInfo {
  std::vector<BundleCamera> cameras;
  int64_t t_start_ns = 0, t_end_ns = 0;
  bool device_available = false;  // the device engine can run here (run.device auto)
  bool supplied = false;          // the bundle was given to the run: the scene keys are not needed
  std::string bundle_digest;      // digest(scene): the bundle digest, or an EVIMO2 source's scene_digest()
  std::string engine_digest;      // SHA-256 of the _engine module, chained in from the render stage
  // EVIMO2 sources: the bundle's light mode (its relight sidecar's low_light), resolved as scene.evimo2.low_light; it
  // decides which lighting keys matter (inert warnings, digests).
  std::optional<bool> low_light;
  // The band N's coverage warning (P2-B B4a): the bundle's shadow-casting point, spot and directional lights, and
  // whether rays that miss see no background light (unknown: none assumed).
  int64_t shadow_lights = 0;
  std::optional<bool> zero_background;
  BakedFacts baked;
};

// One resolved value: its layer (default, prior:<family>, accuracy:<profile>, derived, bundle, profile:<id>, config
// or cli) and the status of that source (the row's or the prior's status, a profile entry's, derived or user).
struct Entry {
  Value value;
  std::string layer, status, note;  // note: an inherited prior names its base
};
using Entries = std::vector<std::pair<std::string, Entry>>;  // schema order

struct Resolved {
  Entries run;                                           // run, scene and dataset keys
  std::vector<std::pair<std::string, Entries>> cameras;  // the selected cameras, in bundle order
  std::vector<std::string> warnings;
  std::vector<Diagnostic> diagnostics;  // the warnings as diagnostics (ui.md §3.10), one per warning, in order
  // Per selected camera, the digests of the stages scene, render, schedule, sensor and output (resolve only).
  std::vector<std::array<std::string, 5>> digests;
  int64_t t_s = 0, t_e = 0;  // the whole windows of the run's span (resolve only)
  // Derived values no stage reads, recorded in run.json ("derived"): with the band N (P2-B B4a) n_band, theta_band,
  // c_cover and eps_switch (resolve.cpp fill_run); empty otherwise.
  std::vector<std::pair<std::string, double>> facts;
  const Entry* find(std::string_view key, std::string_view camera = {}) const;
};

// ---------------------------------------------------------------------------------------------------------------
// `ballasim imu` (docs/design/phase3/outputs.md §4.11): the imu.* keys of a finished run, resolved again.
// ---------------------------------------------------------------------------------------------------------------

// What a run recorded for one camera: its imu.* entries (run.json resolved.cameras[c], empty for a run older than the
// IMU rows) and its sensor-stage digest (run.json stage_digests[c]["sensor"]).
struct SectionCamera {
  std::string name;
  Entries recorded;
  std::string sensor_digest;
};

// The overlay: the recorded values of `cameras` (the run's cameras), under the new layers, lowest first: the root
// profiles, the camera profiles (CAMERA=ID, mount camera, in order) and the --set items (full dotted keys
// cameras.<camera>.imu.* or cameras.default.imu.*).
struct SectionInputs {
  std::vector<SectionCamera> cameras;
  std::vector<std::string> profiles;                                            // root-mounted ids (--profile)
  std::vector<std::pair<std::string, std::vector<std::string>>> camera_profiles;  // camera -> ids (--camera-profile)
  Node sets;                                                                    // null: none
  std::map<std::string, Node, std::less<>> profile_docs;                        // the profile files, by id
};

// One resolved imu.* value: its layer as run.json records it (the recorded value keeps its layer; a new layer is
// profile:<id> or cli), the status of that source and where it comes from (the row's source, run.json, a profile
// entry's source, "--set").
struct SectionEntry {
  Value value;
  std::string layer, status, source;
};
using SectionEntries = std::vector<std::pair<std::string, SectionEntry>>;  // schema order

struct SectionResult {
  std::vector<std::pair<std::string, SectionEntries>> cameras;  // as SectionInputs::cameras
  std::vector<std::string> warnings;
  std::vector<Diagnostic> diagnostics;  // the warnings as diagnostics, in order
  std::vector<std::string> imu_digests;  // per camera: sha256(sensor digest + "\nimu\n" + the canonical imu.* lines)
  // Per camera: the largest change of imu.T_cam_imu's rotation by the polar re-orthonormalization (0 for a rigid one).
  std::vector<double> rotation_change;
};

// Layers below, in order: schema literals, the recorded values, root profiles, camera profiles, --set. Refused
// (DiagnosticError): a key outside imu.* ("ballasim imu changes imu.* keys only; re-run for others"), a mount mismatch,
// an unknown camera or profile, a type or hard-range error, and (for a camera with imu.enabled) imu::validate's errors.
// Warnings: soft ranges and imu keys set but inert, blocks of root profiles for cameras the run lacks, validate's
// warnings and a gravity that differs between enabled cameras.
SectionResult resolve_section(const SectionInputs& in);

// ---------------------------------------------------------------------------------------------------------------
// Frames (docs/design/phase3/frames.md §3.2, P3 FR-1b): one section of a bundle camera, resolved from the layers.
// ---------------------------------------------------------------------------------------------------------------

// One camera section of any bundle camera, as resolve() would give it, with the same layer chain (root profiles, the
// camera's profiles, the configuration, the CLI; the camera's block, then cameras.default) and lookup, so that the
// layering cannot drift: every camera-scope row of `section` ("frames": the keys frames.*) in schema order, from the
// layers or its default (a literal, or the bundle camera's optics.f_number and sensor.clock_offset_ns). It needs no
// sensor family: a camera the run does not simulate resolves too. Rows with a family or required default are
// refused (std::logic_error: resolve the camera in full). The inputs must have resolved as a whole (resolve() checks
// the unknown-camera blocks). `warnings`, if given, receives the soft-range and inert warnings of the section's
// explicit values. An unknown camera is a DiagnosticError.
Entries resolve_section(const Inputs&, const BundleInfo&, std::string_view camera, std::string_view section,
                        std::vector<Diagnostic>* warnings = nullptr);

// The text a value has in a stage digest (hex floats, so equal text is equal bits); frames_digest_base's value lines.
std::string canonical_text(const Value& v);

// Errors (ui.md §3.10, UI-4a): every function below throws DiagnosticError (a std::invalid_argument) for a
// configuration that does not resolve, with the message it always had.

// The profile ids the layers name: the root list (the configuration's, then --profile), then the camera blocks'
// lists in document order, each once. Parses and checks the configuration and the CLI layer.
std::vector<std::string> profiles_needed(const Inputs& in);
// The scene keys, and output.dir, output.overwrite and execution.scene_workers: what loading or building the bundle
// and checking the run directory need. Every layer is parsed and checked; camera names are not (no bundle yet).
Resolved resolve_scene(const Inputs& in);
// Every value of the run on this bundle: the selected cameras' keys, the run keys with the accuracy profile and its
// derivations, the checks of §2.4 (errors throw std::invalid_argument; warnings are collected), the window span and
// the stage digests.
// Collect mode (`collect` non-null; blender.md §3.11, the add-on's Check): the required-key sites record each missing
// key in *collect and continue (a missing sensor.family leaves that camera's family defaults unset); once the cameras
// and the scene keys are gathered, a non-empty list returns before the checks that read the unset values. Every other
// failure still throws. With `collect` null the path and every output are unchanged.
Resolved resolve(const Inputs& in, const BundleInfo& bundle, Missing* collect = nullptr);
// digest(scene) of a source whose bundle is built from the scene keys (§1.2, §3.4: the EVIMO2 cache key): the SHA-256
// of the canonical values of the scene-stage keys among `values` (schema order; typed by their rows) and of `facts`,
// the inputs of the build besides them (sequence directory, importer version, source and core digests), as
// name=value. Keys of the later stages (the lighting, which relight applies at load) and of stage none (workers)
// enter no digest(scene); an unknown key is refused.
std::string scene_digest(const ValueMap& values, const std::map<std::string, std::string, std::less<>>& facts);

// One source of a key's value (ui.md §3.10, the form's provenance): its layer as run.json records it, its value, its
// status and where it comes from (the row's or the prior's source, the accuracy profile's formula, a profile entry's
// source, the configuration's file, "--set", the derivation's note).
struct LayerEntry {
  std::string layer;
  Value value;
  std::string status, source;
};
// Every source that offers `key` a value for `camera` (empty: a run key), lowest precedence first: the row's default
// (its literal, or the camera's family prior), the derivation, bundle value or accuracy profile entry that replaces
// it, then each profile, the configuration and the CLI that set it. The resolved record's layer names the winner.
std::vector<LayerEntry> explain(const Inputs& in, const Resolved& r, std::string_view key, std::string_view camera);
// Whether a row matters on resolved values (ui.md §4.9, the form's visibility): its `when` (or `also_when`) holds,
// reading the camera's values, then the run's; a condition on a key without a value holds (as the inert rule).
bool relevant(std::string_view key, const Resolved& r, std::string_view camera);
// Per accuracy profile: each accuracy key the profile derives (rather than lists as a literal) with every schema key
// its derivation reads (fill_run), closed over the derived keys it reads: a key set explicitly (in any layer) replaces
// its derivation and stays an input of the others. Beside these keys, the derivations read only whether a camera takes
// its sensor keys from an EVIMO2 v0 or v1 profile, and the bundle cameras' pixel pitch. T4 checks the table by
// perturbation (tests/cpp/test_diagnostics.cpp). Unknown profile: DiagnosticError.
std::vector<std::pair<std::string, std::vector<std::string>>> derived_from(std::string_view profile);

}  // namespace balla::config
