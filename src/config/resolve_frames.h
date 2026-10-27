// The frames' values of a run (docs/design/phase3/frames.md §3.2, P3 FR-1b): config::resolve_frames resolves, for every
// camera that simulates events (run.select) and every camera that only gets frames (run.frames_select), its `frames`
// section, from the run's own layers (rebuilt from run.json by Python) and the frames command's --set items on top.
// Host-only (balla_config); the frames steps bind it in ballasim._frames.
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "config/resolve.h"

namespace balla::config {

// A refusal with its frames.md §6 code (FRM-SET-SCOPE, FRM-VALUE, FRM-CAMERA, FRM-MODE-SOURCE); the frames step exits 2
// on it. ballasim._frames raises it as FramesError.
struct FramesRefusal : std::invalid_argument {
  FramesRefusal(std::string code_, const std::string& message)
      : std::invalid_argument(message), code(std::move(code_)) {}
  std::string code;
};

struct FramesCamera {
  std::string name;
  int32_t bundle_index = -1;  // the camera's index in the bundle (the draws' camera key)
  bool events = false;        // simulated for events (run.select), else frames only (run.frames_select)
  Entries values;             // the camera's frames.* rows in schema order; frames.roi_px resolved (below)
  int64_t clock_offset_ns = 0;     // the sensor clock: an events camera's resolved sensor.clock_offset_ns and
  double clock_drift_ppm = 0.0;    // sensor.clock_drift_ppm; a frames-only camera's bundle offset and drift 0
  // The C of the full-mode verdicts (§4.9.3): frames.full.contrast_ref when set, else min(sensor.theta_on,
  // sensor.theta_off) of an events camera, else the smallest such C over the events cameras.
  std::optional<double> contrast;
  // frames_digest_base's value lines (§3.6): the canonical key=value lines, sorted by key, of this camera's frames.*
  // values other than frames.aps.* and of the run's execution.frames_* values.
  std::string digest_values;
};

// The run-scope rows: execution.frames_device, execution.frames_texture_mb and run.frames_select, and the warnings
// (soft range, inert) that name a frames key, for the frames-only cameras' sections and the run's.
struct FramesRun {
  Entries values;
  std::vector<Diagnostic> warnings;
};

// The frames values of every camera of run.select, then of every camera of run.frames_select (in its order), on the
// bundle `b`. `run_inputs` are the layers the run resolved from (config.py's inputs: the recorded configuration, CLI
// items and profiles); `sets` is a Node mapping of --set items, applied as a layer above the run's CLI items (an item
// naming cameras.default.<key> also beats a recorded cameras.<camera>.<key>). The accepted forms
// are exactly cameras.<camera>.frames.<key> and cameras.default.frames.<key> (a key of the section),
// execution.frames_device, execution.frames_texture_mb and run.frames_select (FRM-SET-SCOPE otherwise; FRM-CAMERA for
// a camera the bundle lacks). The events cameras come from the full resolver (the same values as run.json's record);
// a frames-only camera needs no sensor.family and goes through resolve_section.
//
// After the resolution, for every camera whose frames.mode is not off (FRM-VALUE): frames.supersample in {2, 4, 8},
// frames.outputs members in {rgb, mono, groups, reference, aps} with no rgb or groups in reference mode, an explicit
// frames.roi_px with x0 < x1, y0 < y1 inside the image; matched and full need a Blender exchange source
// (FRM-MODE-SOURCE; an unknown source kind passes). A null frames.roi_px becomes run.crop_px when set, else the full
// frame (layer derived). run.frames_select must name bundle cameras outside run.select, each once (FRM-CAMERA).
// Resolver errors (types, hard ranges, unknown keys) are DiagnosticErrors.
//
// `check` false is the run's record: only the cameras of run.select and the run rows, resolved from the layers alone,
// with none of the checks of the paragraph above (their ROI stays as resolved, null included) and no run.frames_select
// camera. Python compares those values with run.json's, which must come first so that a value a check refuses can still
// be repaired by a --set item (the repaired layers are resolved again with the checks).
std::vector<FramesCamera> resolve_frames(const Inputs& run_inputs, const BundleInfo& bundle, const Node& sets,
                                         FramesRun* run = nullptr, bool check = true);

}  // namespace balla::config
