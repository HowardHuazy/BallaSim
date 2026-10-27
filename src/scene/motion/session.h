// The Blender motion session (docs/design/phase3/blender.md §3.5, §3.6, §3.8, §4.6, §4.7, §5.1; BL-9): everything the
// engine does between the add-on's export and the C1 bundle, with the oracle (Blender, over the pipe, or a recorded log)
// as the only source of poses.
//
//   1. convert the exchange's statics (a refusal ends the session before any oracle call), take the time map
//   2. name the oracle's body columns (bodies, light carriers, simulated cameras, the parts of an `inherit` body) and
//      announce them to the add-on
//   3. survey: the export instant (every column present: BRG-STATE, no unknown instance there) and a few dozen times
//      across the span, decoded; rigidity probes (GEO-DEFORM) with Simplify off; the Simplify check (Simplify stays on
//      only when the survey's answers are bitwise the same with it on); the scene's box (light carriers' bounds);
//      camera rigs
//   4. a pass over the frames when some body has no animation source: a static candidate that varies is MOT-SOURCE
//      (refused under strict, fitted by detection under lenient)
//   5. one BodyPipeline per moving column (rig followers ride their leader), driven in rounds (pipeline.h drive) with the
//      rig, inherit and static-candidate checks on every batch
//   6. origin_world, the trajectories (D-P3-12 span) and the camera rigs, the report, session.json, the done summary
//
// Provenance (PHASE3.md §0 rule 4): the protocol and the rules of §3.5 and §4.6-§4.7 are the design's; the Blender
// facts they rest on (frame_set and float32 times, MirrorState, Simplify not moving matrices, object_instances) are the
// headless oracles' (AX, DS d1, d5, d8, RI2); no Blender source was used.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "render/bundle.h"
#include "scene/blender/convert.h"
#include "scene/motion/checks.h"
#include "scene/motion/pipeline.h"
#include "scene/tree.h"

namespace balla::scene::motion {

inline constexpr std::array<int, 2> kOracleProtocol{1, 0};
inline constexpr const char* kOracleProtocolName = "ballasim.motion-oracle";

// --- the mesh oracle (§3.5 mesh_probe / meshes) ---
// Evaluated, object-space vertices of the given exchange objects at (F, s): `table` has one row per (object, sample)
// answered: (object index, sample i, V, offset of the first float in `co`).
struct RawMeshes {
  std::vector<int64_t> table;
  std::vector<float> co;
};
using MeshOracle = std::function<RawMeshes(std::span<const int32_t> objects, std::span<const int32_t> F,
                                           std::span<const float> s)>;

// What the session asked and what was answered (oracle_log.npz, §3.6): replays bitwise without Blender.
struct MeshCall {
  std::vector<int32_t> objects, F;
  std::vector<float> s;
  RawMeshes meshes;
};
struct SessionLog {
  std::vector<int32_t> bodies;   // the exchange objects[] indices announced (the columns)
  OracleLog poses;               // every `sample` message in order
  std::vector<MeshCall> meshes;  // every `mesh_probe` message in order
};

// --- options ---
struct SessionOptions {
  blender::ConvertOptions convert;
  SampleOptions sample;   // sample.eps_px is scene.blender.motion_tol_px; t0_ns / t1_ns are the session's (D-P3-12)
  DetectOptions detect;
  FitOptions fit;
  bool fps_base_snap = false;          // scene.blender.fps_base_snap
  std::vector<std::string> cameras;    // C1 names of cameras to carry besides those with simulate set
  int workers = 1;                     // the pipelines' steps run on a pool of this many threads (results are bitwise
                                       // independent of it)
  int span_margin_windows = 0;         // ceil(m / W): the frames margin (PHASE3.md D-P3-12, frames.md §3.11)
  int survey_samples = 33;             // times across [0, t(frame_end)] (the export instant is the first)
  int simplify_samples = 8;            // the Simplify check's times (the survey's first ones)
  int rigidity_samples = 8;            // times of the rigidity probes
  int confirm_max_samples = 4096;      // static candidates are looked at once per frame (or this many across the span)
  double box_widen = 0.25;             // the scene box (light carriers' bounds), widened by this fraction of its diagonal
};

// What the session needs from the other end. oracle is required; the rest may be empty. A replay supplies the log's
// (replaying_io).
struct SessionIO {
  std::function<void(std::span<const int32_t>)> bodies;   // the columns' exchange objects[] indices, once
  std::function<void(bool)> simplify;                     // `set {simplify}`
  Oracle oracle;
  MeshOracle meshes;
  std::function<void(double, std::string_view)> progress;
};

// Answers from a log (which must outlive the io): BRG-REPLAY at the first request that differs.
SessionIO replaying_io(const SessionLog& log);

// --- the result ---
struct SessionTrajectory {
  render::Trajectory traj;
  std::vector<std::pair<int64_t, int64_t>> holds;   // [begin, end) of the hold pieces
};

struct SessionResult {
  bool refused = false;                               // done.status "refused": the report says why
  std::vector<blender::ReportItem> report;            // the session's items (the statics' are statics.report)
  blender::Statics statics;                           // convert_statics' (about the Blender origin, not placed)
  std::array<double, 3> origin_world{};
  std::vector<SessionTrajectory> trajectories;        // bodies, then rigs
  std::vector<int32_t> body_trajectory;               // per statics.bodies: trajectory index, -1: static
  std::vector<int32_t> camera_trajectory;             // per statics.cameras: its rig's trajectory, -1: not carried
  std::vector<std::array<double, 16>> camera_t_rig_cam;   // per statics.cameras: the C1 camera in its rig's frame
  Tree session = Tree::dict();                        // session.json without engine, options and motion_key
  Tree summary = Tree::dict();                        // done.summary: timeline, spans, preview, report_summary
  Tree perf = Tree::dict();                           // session_perf.json
  SessionLog log;
  std::optional<TimeMap> time;
  bool simplify = false;                              // Simplify was left on
};

// Runs the session. MotionError (a refusal code of §6.1) is turned into a refusal item; BRG-* codes, std::invalid_argument
// (BRG-PROTO: a malformed exchange or answer) and an exception thrown by the oracle propagate.
SessionResult run_session(const blender::ExchangeIn& ex, const SessionOptions& opt, const SessionIO& io);

// --- pieces (unit-tested directly) ---
// TimeSettings of the exchange's time, span and render records.
TimeSettings time_settings(const blender::ExchangeIn& ex, bool fps_base_snap);
// The exchange's animation dump as KeyScene (declared keys, §4.7.4): objects' parents, own and data animations and
// constraint targets by name; the F-curves' key tables from ExchangeIn::anim_keys.
KeyScene key_scene(const blender::ExchangeIn& ex);

}  // namespace balla::scene::motion
