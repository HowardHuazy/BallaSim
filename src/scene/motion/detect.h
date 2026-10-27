// Breakpoint detection's pure pieces (docs/design/phase3/blender.md §4.7.5; a port of the research prototype's
// proto_motion.py candidates / node_scan / classify and proto_detect.py confirm_step / refine_kink, MF E7, t_pipeline):
// the candidates, their classification on the samples, the pose distance of step bisection, one kink burst, and the
// merge and snap of the breakpoints. BodyPipeline (pipeline.h) runs them between oracle rounds.
//
// The signal of a body near a candidate is its pose in pixels, y = (w_pos (p - p_ref), w_rot Log(q_ref^T q)) with the
// error model's weights at the candidate (§4.7.2), so a jump of |dy| px moves the image by about that much.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "scene/motion/pipeline.h"

namespace balla::scene::motion {

// Candidate times (§4.7.5), ascending and unique: the midpoints of the unresolved intervals; spacing spikes (an
// interval shorter than 1/8 of the median of the three intervals on either side, and no longer than its neighbours);
// the node scan (one-sided quadratic slopes at a node from it and its two left, and its two right neighbours: a slope
// change times the shorter adjacent spacing above eps/4, largest within two nodes); declared step and C0 keys.
std::vector<int64_t> candidates(const SampleSet& set, std::span<const DeclaredKey> keys, double eps_px);

// The widest bracket (a, c] of fit samples the sampler can leave around a value step (§4.7.3-§4.7.5): 2 h_min; or one
// step of Blender's float32 time grid where that is coarser (2 q(F) at the bracket's ends + 2 ns of rounding: from
// |F| >= 2^13 at 24 fps), which refinement cannot split; plus 2 w for each declared plateau whose samples were dropped
// inside (a, c) (a step at a declared key then spans the plateau, w = 1e-4 N_s / framelen frames: 33 us at 3 fps,
// framelen 1/8 or an NLA scale of 8). A null time or plateaus leaves its term out.
struct StepBracket {
  int64_t h_min_ns = 16384;
  const TimeMap* time = nullptr;
  const Plateaus* plateaus = nullptr;
  int64_t limit_ns(int64_t a, int64_t c) const;
};

// A candidate classified on the fit samples (one-sided cubics on 6 samples per side, valid when their rms is at most
// eps/2 + the float32 floor; the standard error from their residuals, at least max(eps/100, floor)):
// - kStep: at a split t of a bracket (a, t] of at most bracket.limit_ns(a, t) whose fits' values differ by more than
//   max(eps/4 + 2 floor, k_sig SE); v_pxps is the larger one-sided speed there (w_pos |dp/dt| + w_rot |dphi/dt|);
// - kKink: the fits' slopes differ by more than k_sig SE; t where the one-sided cubics cross (Gauss-Newton);
// - kSmooth: valid fits without a significant slope change; kUnclear: no valid pair of fits.
// The splits tried are the first sample at or after the candidate, the one after it and the one before it.
struct Classified {
  enum class Kind { kStep, kKink, kSmooth, kUnclear } kind = Kind::kUnclear;
  int64_t t_ns = 0;
  size_t a = 0, c = 0;         // kStep: the bracket's sample indices in fit (c = a + 1)
  double jump_px = 0.0, dv_pxps = 0.0, v_pxps = 0.0;
  double w_pos = 0.0, w_rot = 0.0, f1 = 0.0;  // the signal's weights and the largest float32 floor of the samples used
};
Classified classify(const Track& fit, int64_t t_candidate, double eps_px, const StepBracket& bracket, double k_sig);

// The error model's distance between two poses of a body (px): w_pos |dp| + w_rot |Log(qa^T qb)|.
double pose_px(double w_pos, double w_rot, const rot::Q& qa, const rot::V& pa, const rot::Q& qb, const rot::V& pb);

// One burst's request times around t: n each in [t - W, t - W/16] and [t + W/16, t + W], evenly spaced, rounded to ns
// (ties to even). Ascending, unique.
std::vector<int64_t> burst_requests(int64_t t_ns, int64_t w_ns, int n);

// One burst's estimate: one-sided quadratics through the samples left of t and at or right of t (at least 3 each),
// crossed by Gauss-Newton from t, within the window; dv_pxps is the slope change and jump_px the gap there.
struct Crossing {
  bool ok = false;
  int64_t t_ns = 0;
  double dv_pxps = 0.0, jump_px = 0.0;
};
Crossing burst_crossing(const Track& samples, int64_t t_ns, int64_t w_ns, double w_pos, double w_rot);

// Breakpoints closer than merge_ns are merged (§4.7.5): a step wins over a kink, else the larger jump (steps) or slope
// change (kinks). Ascending.
std::vector<Breakpoint> merge_breakpoints(std::vector<Breakpoint> bps, int64_t merge_ns);

// Snaps each breakpoint to the nearest declared key within the key's plateau + 1 us (§4.7.5): its time becomes the
// key's, exact up to q(F) (dt_bound_ns), and it is `declared` and `snapped`. Returns how many snapped. The keys must be
// ascending in time (declared_keys' order; std::invalid_argument otherwise); each breakpoint visits only the keys
// within the widest snap window of it.
int snap_breakpoints(std::vector<Breakpoint>& bps, std::span<const DeclaredKey> keys, const TimeMap& time);

}  // namespace balla::scene::motion
