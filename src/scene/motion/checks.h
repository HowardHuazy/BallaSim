// The motion session's body-level checks (docs/design/phase3/blender.md §4.7.1, §4.7.2, §4.7.9, §6.1; BL-8): the
// mode of a body by provenance, the camera rigs, and the scale checks of a fitted body (float32 positions and times).
//
// Provenance (PHASE3.md §0 rule 4): which Blender settings move an object is re-derived from the Blender manual
// (Physics > Rigid Body: an Active body is simulated unless Animated, i.e. kinematic, is on; a baked point cache
// replays it; Constraints > Transform Cache; Object > Relations: parents and bone parents; Animation: actions, NLA,
// drivers) and from the headless oracles of blender.md §2.3 (AX exp04*, MF a3, R1: rigid-body frame states; AX exp06:
// transform caches), not from Blender's sources.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "scene/blender/convert.h"
#include "scene/motion/pipeline.h"

namespace balla::scene::motion {

// --- mode by provenance (§4.7.1): never by the shape of the motion (chord-linearity cannot tell keys from a bake) ---
enum class Source : uint8_t {
  kNone,        // nothing in the chain moves it: a static candidate (the session confirms it from its samples)
  kKeyed,       // F-curves, NLA strips, drivers, constraints or a bone parent in its chain
  kFrameState,  // an ACTIVE, non-kinematic rigid body or a TRANSFORM_CACHE constraint in its chain (BL-10)
};
struct ModeChoice {
  Source source = Source::kNone;
  int32_t frame_state_object = -1;  // the chain member holding the rigid body or the transform cache
  bool cache = false;               // a transform cache (else a rigid body)
  bool kinematic_split = false;     // its ACTIVE rigid body's kinematic flag is animated: keyed while kinematic,
                                    // frame-state otherwise, split at the switches (BL-10)
  std::string code, message;        // a refusal (PHY-MIXED, PHY-UNBAKED, PHY-REMAP), else empty
};
// The chain of `object`: itself and, transitively, its parent, instancer and instanced object (convert.cpp's static
// candidacy walks the same links). PHY-MIXED: a frame-state source and any keyed source in one chain (an animated
// kinematic flag on the rigid body itself excepted); PHY-UNBAKED: a rigid body without a baked, current cache of an
// enabled rigid-body world; PHY-REMAP: a rigid body with frame_map (framelen != 1). Each object is visited once (the
// statics conversion refuses link cycles); std::invalid_argument for an index out of range.
ModeChoice choose_mode(const blender::ExchangeIn& ex, int32_t object, double framelen = 1.0);
// MOT-SOURCE (§4.7.1): a body with no source whose samples vary; refused under strict, a warning under lenient (it is
// then fitted in keyed mode with detection only).
std::optional<blender::ReportItem> motion_source(const ModeChoice& mode, bool samples_vary, blender::Policy policy);

// --- rigs (§4.7.9) ---
// Cameras whose relative pose is constant within kRigTol (rotation angle, rad; translation relative to the larger
// distance of the two from the world origin, the float32 scale of matrix_world: 16x its 6e-8 rounding, MF E6) share
// one rig: the first camera of the group, in the order given. Each camera's pose in its rig is that constant.
inline constexpr double kRigTol = 1e-6;
struct RigGroups {
  std::vector<int32_t> cameras;     // the camera body columns, as given
  std::vector<int32_t> rig;         // per camera: the body column of its rig (its group's first camera)
  std::vector<rot::Q> q_rig_cam;    // R_rig^T R_cam
  std::vector<rot::V> p_rig_cam;    // R_rig^T (p_cam - p_rig), metres
  std::vector<double> dev_rad, dev_m;  // the largest deviations from those seen so far
};
// Groups from the samples of one batch (every camera column present at every sample). hints: per camera a group the
// scene declares (-1 none): cameras with the same hint must share a rig, else CAM-RIG.
RigGroups group_rigs(const PoseBatch& batch, std::span<const int32_t> cameras, std::span<const int32_t> hints = {});
// CAM-RIG when a grouped camera's pose in its rig departs from the constant at a sample of `batch` (drive's on_batch).
void check_rigs(const PoseBatch& batch, RigGroups& groups);

// --- a fitted body's report (§4.7.2, §4.2, §6.1; the scale checks) ---
// POS-RES (float32 positions: floor_px > eps/8 at an accepted sample, bound eps/2 + 2 floor_px), TIME-RES (float32
// time: dv q(F) > eps/4 at a breakpoint), MOT-FAR (a sample farther than kFarM from Blender's origin), LGT-MOVE-NEAR
// and the aliasing warning (0.6 pi .. 0.9 pi between samples). Every item is a warning with a measured bound; `object`
// is left for the session to name.
inline constexpr double kFarM = 64.0;
std::vector<blender::ReportItem> body_report(const SampleSet& set, const TimeMap& time, double eps_px);

}  // namespace balla::scene::motion
