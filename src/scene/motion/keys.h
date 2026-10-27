// Declared keys (docs/design/phase3/blender.md §4.7.4): the times at which a body's own animation, its parent chain
// and its constraint targets can break its motion, read from the exchange's animation dump (§3.4: F-curve code
// tables, F-modifiers, NLA tracks), with a continuity hint and Blender's key plateau. They seed the sampler's base
// grid (§4.7.3) and detection's candidates (§4.7.5), and detected breakpoints snap to them.
//
// Provenance (PHASE3.md §0 rule 4): the rules are re-derived from the Blender manual (F-curve interpolation, easing,
// handle types and extrapolation; the Stepped, Cycles and Envelope F-modifiers and their restricted range; NLA strip
// timing, repeat, reverse, extrapolation, blending and auto blend; Robert Penner's easing equations, which the manual
// names, for BOUNCE's breaks at 1/2.75, 2/2.75 and 2.5/2.75) and from headless oracles: MF E7/E8 (the 1e-4-frame key
// plateau of every interpolation mode; the E7 key classes), R2 c1 (framelen scales the plateau) and the BL-7 probes
// `ballasim_out/phase3_research/blender/bl7/probes/p1_keys.py`, `p2_strip_ends.py`, `p3_strip_props.py`,
// `p4_animated_controls.py`, `p5_pushed_down.py` and `p6_strip_drivers.py` (4.2.1 and 4.5.3, identical results, kept
// with their JSON): BOUNCE with AUTO easing breaks as EASE_OUT does; EASE_IN mirrors the fractions and EASE_IN_OUT
// halves both; the NLA mapping f = frame_start + (a - action_frame_start) scale + r (action_frame_end -
// action_frame_start) scale, mirrored about the strip's middle when reversed; a key's plateau under a strip of scale 4
// is 4e-4 frames; a strip shows the underlying value before its start unless HOLD and after its end under NOTHING
// (HOLD_FORWARD holds the end only); a STEPPED modifier steps exactly at offset + k step with no plateau; a Cycles
// modifier repeats its curve with the period of its key range; a strip's blend_in, blend_out, frame_start, scale and
// repeat can be keyed through its owner (F-curves "animation_data.nla_tracks[...].strips[...].<property>" in the
// owner's active action) and each moves the motion (p3; blend_in keyed down from 8 to 0, so the ramp binds); such a
// curve pushed down into a strip of another track, above or below, moves it the same with no active action left, and
// muting that track keeps the last written value while muting that strip writes another (frame_start: neither the
// static nor the animated motion) (p5); a driver with a simple expression on such a property is valid without
// auto-run, sits in the owner's drivers under the same path and moves the motion (p6); an animated influence replaces
// the blend ramps, and an animated strip time no longer wraps at the repeats (p4). No Blender source was used.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "scene/motion/time_map.h"
#include "scene/tree.h"

namespace balla::scene::motion {

enum class Continuity : int8_t { kStep = -1, kC0 = 0, kC1 = 1 };

// A declared key: its time, class hint and plateau half-width in output frames (1e-4 N_s / framelen; Blender returns
// the key value within it). plateau is false for times that are no F-curve key (strip ends and blends, modifier
// steps, BOUNCE breaks): those snap within their time resolution only.
struct DeclaredKey {
  int64_t t_ns = 0;
  Continuity hint = Continuity::kC0;
  bool plateau = true;
  double plateau_frames = 1e-4;
};

// --- the exchange's animation dump (blender.md §3.4 animations[]; BL-2a's ExchangeIn adapts to it) ---

// Code tables (§3.4, Blender's enum orders, DS d6).
enum class Interp : int8_t {
  kConstant, kLinear, kBezier, kSine, kQuad, kCubic, kQuart, kQuint, kExpo, kCirc, kBack, kBounce, kElastic
};
enum class Easing : int8_t { kAuto, kEaseIn, kEaseOut, kEaseInOut };
enum class Handle : int8_t { kFree, kAligned, kVector, kAuto, kAutoClamped };

struct KeyModifier {               // F-modifier (§3.4 modifiers[])
  std::string type;                // STEPPED | CYCLES | NOISE | GENERATOR | FNGENERATOR | LIMITS | ENVELOPE
  bool mute = false, use_restricted_range = false;
  double frame_start = 0, frame_end = 0, blend_in = 0, blend_out = 0;
  Tree fields = Tree::dict();      // <type fields>: STEPPED frame_step, frame_offset, use_frame_start, frame_start,
                                   // use_frame_end, frame_end; CYCLES mode_before, mode_after, cycles_before,
                                   // cycles_after (modes NONE | REPEAT | REPEAT_OFFSET | MIRROR)
};

struct KeyCurve {                  // F-curve (§3.4 fcurves[]; keys: anim.<key>.keys)
  std::string data_path;
  int32_t index = 0;
  int32_t extrapolation = 0;       // 0 CONSTANT, 1 LINEAR
  std::vector<double> keys;        // n x 7: frame, value, interpolation, easing, left handle, right handle, collinear
  std::vector<KeyModifier> modifiers;
};

struct KeyStrip {                  // NLA strip (§3.4 strips[])
  std::string action, slot;        // resolved against KeyScene::animations by (action, slot)
  double frame_start = 0, frame_end = 0, action_frame_start = 0, action_frame_end = 0, scale = 1, repeat = 1,
         blend_in = 0, blend_out = 0;
  int32_t extrapolation = 1;       // 0 NOTHING, 1 HOLD, 2 HOLD_FORWARD
  bool use_reverse = false, use_animated_influence = false, use_animated_time = false, mute = false;
};

struct KeyTrack {
  bool mute = false, is_solo = false;
  std::vector<KeyStrip> strips;
};

struct KeyAnimation {              // one animation data block (§3.4 animations[])
  std::string action, slot;        // the active action (evaluated in scene time on top of the NLA)
  std::vector<KeyCurve> fcurves;   // the active action's F-curves
  std::vector<KeyTrack> nla;
  std::vector<std::string> driver_paths;  // drivers[].data_path: only drivers on strip properties are read
};

struct KeyObject {                 // the objects[] fields the key chain needs
  int32_t parent = -1;             // objects[] index (an armature for a bone parent)
  std::vector<int32_t> animations;        // the object's own animation data: animations[] indices
  std::vector<int32_t> data_animations;   // its data block's (a curve's eval_time, ...)
  std::vector<int32_t> constraint_targets;  // objects[] indices of its unmuted constraints' targets
};

struct KeyScene {
  std::vector<KeyObject> objects;
  std::vector<KeyAnimation> animations;
};

// A break of one F-curve in its own (action) time.
struct KeyFeature {
  double frame;
  Continuity hint;
  bool plateau;  // an F-curve key (Blender snaps to it within 1e-4 frames of the curve's time)
};

// The breaks of one F-curve within [lo, hi] of its own time (§4.7.4 table): CONSTANT at key i a step at key i+1 (and
// C0 at key i); LINEAR C0 at both ends; BEZIER C0 at a key with a VECTOR or FREE handle or non-collinear handles on
// that side, else C1; easing C0 at both ends and BOUNCE's three breaks; an end key C1 when LINEAR extrapolation
// continues a LINEAR or BEZIER segment (its slope or handle), else C0; STEPPED steps at offset + k step; CYCLES
// repeats the curve's breaks at the period of its key range (mirrored for MIRROR) with a step or C0 at each junction;
// a restricted range steps (or, with blends, kinks) at its ends. ENVELOPE (and an unknown modifier type) is MOT-SOURCE
// unless `lenient`; malformed code tables are BRG-PROTO. Ascending, merged (the lowest class and any plateau win).
std::vector<KeyFeature> curve_features(const KeyCurve& curve, double lo, double hi, bool lenient = false);

// Whether an object-level F-curve moves the object: its transform and delta transform, its constraints, the pose
// bones of an armature. A data block's curves (a constraint target's eval_time, ...) are all kept.
bool moves_object(const std::string& data_path);

// The declared keys of `object` within [t0, t1] (§4.7.4): its own animation, its parent chain's (an armature parent's
// pose bones included) and the animation of every chain member's constraint targets, one level deep, with their data
// blocks'. The active action runs in scene time; each unmuted strip (the solo tracks only, when one is solo) maps its
// action's breaks through its timing, plateau 1e-4 scale; its start and end step, or kink when blended or held, its
// blend ramps and repeat boundaries kink (only the repeats that start within the strip; more than 4M within the span
// is MOT-SOURCE). Detection finds what is not mapped: an animated influence leaves the strip's ends and repeats but
// neither its blend ramps nor its keys; an animated strip time leaves its ends and blend ramps only; strip properties
// animated through the owner (a curve "animation_data.nla_tracks[...]" in its active action or in any of its strips'
// actions, muted or not, or a driver on such a path) leave nothing of that animation's strips. Scene time c maps to
// the output frame c / framelen and to ns through the TimeMap; the plateau is 1e-4 N_s / framelen output frames.
// Ascending; equal times merged (the lowest class and the widest plateau win).
std::vector<DeclaredKey> declared_keys(const KeyScene& scene, int32_t object, const TimeMap& time, int64_t t0,
                                       int64_t t1, bool lenient = false);

}  // namespace balla::scene::motion
