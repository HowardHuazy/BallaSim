// The assembly of a static Blender exchange (docs/design/phase3/blender.md §3.7, §4.1, §4.2, §4.7.1, §4.7.9): when
// every body, simulated camera and light of an exchange is static, its statics are the whole scene and the C1 bundle
// is assembled from them without a motion session. This is what makes BL-4b's static job end to end run before the
// session (BL-9); the session takes over every exchange that moves, and with it the full timeline record.
//
// Provenance (PHASE3.md §0 rule 4): the timeline is re-derived from the Blender manual (Output Properties: Frame Rate
// as fps / Base, the frame range) and the headless oracles of blender.md §2.2-§2.3 (AX exp02/exp10, MF e1, DS d1:
// fps_base is a float32, a frame F is evaluated at the float32 F, which is exact for every integer frame of a span);
// it equals the motion session's TimeMap (BL-6) at the span's integer frames. No Blender source was used.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "render/bundle.h"
#include "scene/blender/convert.h"
#include "scene/geometry.h"
#include "scene/tree.h"

namespace balla::scene::blender {

// The span of an exchange's timeline (§4.2): the span's first frame F_ref is t = 0; frame f is at
// round_half_even((f - F_ref) 1e9 B_num / (fps B_den)) ns with B = B_num / B_den the exact value of the float32
// fps_base; the bundle covers [-2W, t_end) with t_end = (floor(t(frame_end) / W) + 1) W, so the run's default span
// [0, t_end) always holds frame_end (W = 2^23 ns, the engine's window).
struct StaticTimeline {
  int32_t fps = 0;
  uint32_t fps_base_bits = 0;
  int64_t base_num = 1, base_den = 1;
  int32_t frame_start = 0, frame_end = 0;
  int64_t t_frame_end_ns = 0, t_start_ns = 0, t_end_ns = 0;
};
// From the exchange's `time` (fps, fps_base_bits) and `span` (the add-on's frames; `time`'s range without one).
// std::invalid_argument "BRG-PROTO: ..." for a malformed record (fps outside [1, 32767], fps_base not finite and > 0,
// frame_end < frame_start).
StaticTimeline static_timeline(const ExchangeIn& ex);
// {fps, fps_base_bits, base_num, base_den, frame_ref, t_ref_ns, frame_start, frame_end, t_frame_end_ns, bundle_span,
// run_span, run_span_extends_ns}: the part of the session's timeline record (§4.2) a static scene has.
Tree timeline_record(const StaticTimeline& t);

// Why an exchange needs the motion session (empty: its statics are the whole scene). A converted scene needs it when
// a body is not a static candidate or is a light carrier, a simulated camera is not static, a light rides a body, a
// body's data is animated or its geometry may change over the span (may_change_shape: deform modifiers, animated
// shape keys or modifiers, geometry nodes, cloth, soft body, a modifier that follows the frame or reads other
// objects), or when anything up the parent, instancer and instanced-object links of a body, a simulated camera or a
// world-frame light can move or reshape it without a key on its own transform: animated data (a curve's eval_time
// moves its children along the path), a shape that may change (vertex parents and instances follow the evaluated
// mesh, whether or not the instancer is rendered), particles or geometry nodes (they move or replace instances), or an
// armature (bone constraints are not exported). Static candidacy (§4.7.1) has already looked at the keys,
// constraints and rigid bodies up those links. Each reason names its object, and its ancestor.
std::vector<std::string> motion_needed(const ExchangeIn& ex, const Statics& s);

// The trajectory of a static rig (§4.7.9): K = 6 equal controls R_world_rig, p - origin, knot spacing h = 2^30 ns
// from t0, as many as cover [t0, t1], with span [t0, t1]; the bundle holds it over the span.
render::Trajectory static_rig(const geo::M4& T_world_rig, const std::array<double, 3>& origin, int64_t t0, int64_t t1);
inline constexpr int64_t kStaticKnotNs = int64_t{1} << 30;

}  // namespace balla::scene::blender
