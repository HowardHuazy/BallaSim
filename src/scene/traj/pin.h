// M10-PIN (P2-H; the user's M10-HOLD decision of 2026-10-15): a body whose Vicon track is scene-static is frozen, so a
// sliding sequence's static render moves nothing. Opt-in (scene.evimo2.pin_static_px_s); the hold test of fit_smoothing
// (hold_rule, hold_max_disp_m) is untouched.
//
// Scene-static is the bench's rule for M10 (a GT image speed under 60 px/s) read from the body's own Vicon samples:
// per kept sample the image rate of the next lag: the diameter of the window (the furthest apart any two of its samples
// are; sphere_displacement's bound with the surface's RMS radius, times f over the centre's depth along the camera) over
// the window's length. A body is quiet where that rate is under the limit; the union of its quiet windows [t_i, t_i + lag]
// is what to freeze (a body that slides for a second and rests before and after freezes in both rests, at two poses; a gap
// of no more than 25 ms is bridged). Every control point under an interval is set to the robust central
// pose of the kept samples there (rot::robust_mean_pose, the tabletop rule's pose), so the trajectory is exactly still
// over the interval and the engine reads it as a hold (Trajectory.holds). It differs from the unfrozen fit only there and
// within (order - 1) knots (25 ms) of the ends, and moves nothing where the body moves. One interval over the whole run
// makes the body static.
//
// The blend: a hold's still interval has (order - 1) outer controls at each end, all set to the rest pose, and the curve
// blends from the rest pose to the fitted controls over the 25 ms outside it, on the motion side. The interval is NOT
// pulled in to put the blend inside the quiet window: the quiet windows overshoot the scene-static edges of the bench's
// GT by about that reach (sliding_01's box_02: 24 ms and 31 ms), so the blend lies just outside the GT-static time;
// pulling in left it 0.8 ms inside at one edge and 6 ms outside at the other. The price is a snap at the border of a
// creeping body (docs/dev/SCENE.md).
//
// The lag: Vicon jitter of 0.5 mm at 0.5 m reads f sqrt(2) 0.5 mm / (0.5 m 0.1 s) = 8 px/s over 0.1 s (160 px/s over
// the 5 ms of a sample pair), against the 60 px/s limit; a body at the limit has moved 6 px. The diameter, not the
// first-to-last displacement, is what is read: a body that vibrates near 1 / lag (10 Hz) is back where it started a
// lag later, and a window that starts at the middle of the stroke sees half of it. A frozen window therefore holds
// every sample within limit * lag of every other (a vibration of under half of that, 3 px, freezes at any frequency).
// Measured on sliding_00 and sliding_01 (docs/dev/SCENE.md): at rest the bodies read about 5-30 px/s (p99), the sliding
// boxes 1000 px/s and more.
#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "scene/rotation.h"
#include "scene/traj/fit_smooth.h"

namespace balla::scene::traj {

struct PinOptions {
  double limit_px_s = 60.0;           // quiet below this image speed
  int64_t lag_ns = 100'000'000;       // the time over which a displacement is read as a speed (not a knob: see above)
  int64_t min_ns = 100'000'000;       // the shortest interval frozen (a whole-run interval needs no minimum)
  double focal_px = 556.0;            // the camera's focal length in pixels
  int64_t bridge_ns = 0;              // quiet intervals closer than this merge: a hold's reach, (order - 1) h of the fit
};

// One body's kept Vicon samples (t strictly increasing) with the depth of the body centre along the camera at each.
struct PinSamples {
  std::span<const int64_t> t_ns;
  std::span<const rot::Q> q;
  std::span<const rot::V> p;
  std::span<const double> depth_m;
  rot::V center{0, 0, 0};             // the body-frame point the displacement is read at
  double radius_m = 0;                // the surface's RMS distance from it
};

struct PinResult {
  std::vector<std::pair<int64_t, int64_t>> intervals_ns;   // quiet, bridged, at least min_ns (or the whole run)
  bool whole = false;                 // one interval covers the kept samples: the body is static
  int64_t pairs = 0;                  // samples with a partner a lag later (the speeds below)
  double speed_p99 = 0, speed_max = 0;   // the window rates in px/s over those; NaN without a pair
};

// The body's quiet intervals. Samples without a partner a lag later (the last lag of the run) belong to the window of
// the sample before them: a body quiet to the end is quiet to its last sample.
PinResult pin_intervals(const PinSamples& s, const PinOptions& o);

// Freezes `fit` under `intervals_ns` (and its existing holds, which merge with them): the controls of every merged hold
// group are set to the central pose of the kept samples inside its still interval, `holds` and `diag.holds_ns` follow,
// and `whole` sets diag.is_static (every control equal, the span one hold). Returns the largest image excursion (px)
// of a kept sample inside a frozen interval from its central pose, 0 when nothing froze.
double pin_apply(FitResult& fit, const PinSamples& s, const PinOptions& o,
                 const std::vector<std::pair<int64_t, int64_t>>& intervals_ns, bool whole);

}  // namespace balla::scene::traj
