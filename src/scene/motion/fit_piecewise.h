// The piecewise fit of a Blender body's samples (docs/design/phase3/blender.md §4.7.6; a port of the research
// prototype MF/scripts/proto_fit.py and the validation of DS d14): per piece a uniform K = 6 split SO(3) x R^3
// cumulative B-spline (core/traj/spline.h) whose knots start at the piece's begin, fitted by weighted least squares in
// the error model's units (§4.7.2) with a small jerk penalty:
//
//   sum_n w_pos,n^2 |p(t_n) - p_n|^2 + w_rot,n^2 |Log(R_n^T R(t_n))|^2 + lambda jerk,
//   lambda = lambda_rel mean(w) n / N_ctrl (positions and rotations each with their own weights),
//
// positions by one banded solve, rotations by Gauss-Newton on SO(3) (right perturbations R_c <- R_c Exp(eps_c), at
// most gn_iters steps, stopping below 1e-13). Pieces are fitted left to right: a piece joined C0 to its left neighbour
// starts at that piece's end pose (a pseudo-sample at its begin with weight join_weight x the largest weight, the
// measured method of MF a7 and DS d14) and is then moved rigidly onto it (a constant shift of its positions and a
// constant left rotation of its rotations keep its shape; the move is the pseudo-sample's residual, at most about
// 1e-6 eps px), so the engine's join check (core/traj/pieces.h kJoinTolM, kJoinTolRad) holds to rounding.
//
// Holds (§4.7.6): a run of at least 3 bitwise-identical samples. Tying the controls under a hold (TieMap) would
// still the spline up to the enclosing knots, while Blender's body moves right after the run, so a hold becomes a
// piece of its own instead: K equal controls (knot spacing 2^30 ns, like a static rig, blender::static_rig), its pose
// bitwise constant over the piece, which the bundle declares as a hold. It is joined C0 on both sides unless a step
// bounds it (a CONSTANT key).
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "core/traj/spline.h"
#include "render/bundle.h"
#include "scene/motion/track.h"

namespace balla::scene::motion {

// §4.7.6's options (the scene.blender.fit_lambda_rel, validation_probes and max_halvings rows of §3.11 once BL-3b binds
// them; join_weight and accept_frac are documented constants).
struct FitOptions {
  bool enabled = true;               // false: the pipeline ends after detection (BL-6 / BL-7 behaviour)
  int64_t h_cap_ns = 0;              // the knot spacing cap; 0: 2 h0_ns() (detection's h_cap)
  int64_t h_floor_ns = 1000000;      // the smallest spacing the 2 x median rule picks (1 ms)
  double lambda_rel = 1e-10;         // fit_lambda_rel: relative jerk regularization
  double join_weight = 1e6;          // a C0 join's pseudo-sample weight, times the largest sample weight
  double accept_frac = 0.4;          // validation accepts e_px <= accept_frac eps + floor_px (DS d14: 0.4)
  int halvings = 5;                  // max_halvings: knot-spacing halvings per piece before MOT-FIT
  int probes = 32;                   // validation_probes: seeded probes per piece
  uint64_t probe_seed = 0x6d6f74696f6e3031ull;
  bool breakpoint_probes = true;     // the probes at b and b- of every join (DS d14; off only to reproduce it)
  int end_fill_div = 8;              // the fill's gaps within one knot spacing of a piece's end: h / end_fill_div
                                     // (elsewhere h / 2); 0: h / 2 throughout, the prototype's fill (DS d14)
  int gn_iters = 8;
};

inline constexpr int kFitOrder = 6;
inline constexpr int64_t kHoldKnotNs = int64_t{1} << 30;  // a hold piece's knot spacing (blender::kStaticKnotNs)

struct Pose {
  rot::Q q{1, 0, 0, 0};
  rot::V p{0, 0, 0};
};

// One fitted piece: its spline (K = 6, knots t0 + s h with t0 = begin), world metres.
struct FitPiece {
  int64_t begin_ns = 0, end_ns = 0;     // serves [begin, end) (the last piece [begin, end])
  int64_t t0_ns = 0, h_ns = 0;
  Continuity join = Continuity::kC0;    // the join at begin (piece 0: kC0)
  bool hold = false;
  std::vector<rot::Q> rot;
  std::vector<rot::V> pos;
  int halvings = 0;                     // knot-spacing halvings spent on it
  bool split = false;                   // made by the split at declared keys (or already split)
  double join_px = 0.0;                 // its C0 pseudo-sample's residual before the rigid move (px)
  double max_px = 0.0;                  // its validation maximum (px)
  double floor_px = 0.0;                // the largest float32 floor (g = 1) of its validation samples
  int64_t validated = 0;                // its validation samples
  int gn_iters = 0;
};

// Evaluates a fitted piece (the core's fp64 spline, as the renderer and the planner build it from the controls).
class PieceEval {
 public:
  explicit PieceEval(const FitPiece& piece);
  // view_ points into nodes_: a copy would alias the source's buffer and dangle once the source dies. A move keeps the
  // buffer (the vector hands it over), so view_ stays valid.
  PieceEval(const PieceEval&) = delete;
  PieceEval& operator=(const PieceEval&) = delete;
  PieceEval(PieceEval&&) noexcept = default;
  PieceEval& operator=(PieceEval&&) noexcept = default;
  Pose operator()(int64_t t_ns) const;  // std::invalid_argument outside the piece's support

 private:
  std::vector<::balla::traj::SplineNode<double>> nodes_;
  ::balla::traj::SplineView<double> view_{};  // points into nodes_
  int64_t t0_ = 0;
};

// Fits one piece to `samples` (ascending; those in its interval) with knot spacing h over [begin, end]; `start`: the
// pose it must start at (a C0 join). A hold piece takes K equal controls at `start`, or at the first sample's pose.
FitPiece fit_piece(const Track& samples, int64_t begin, int64_t end, int64_t h, const std::optional<Pose>& start,
                   bool hold, const FitOptions& opt);

// e_px of the error model (§4.7.2) between the piece and each sample: w_pos |dp| + w_rot |dtheta|.
std::vector<double> piece_errors(const FitPiece& piece, const Track& samples);

// The largest angle between consecutive control rotations (C1's pi/2 rule, bundle.py _check_increments).
double max_increment_rad(const FitPiece& piece);

// Holds (§4.7.6): the runs of at least `min_run` consecutive samples of `track` (ascending in time) with bitwise
// identical poses, as [first, last] sample times.
std::vector<std::pair<int64_t, int64_t>> find_holds(const Track& track, size_t min_run = 3);

// The pieces of a body's span [t0, t1] (§4.7.6): a breakpoint begins a piece of its class; a hold run [a, b] becomes a
// hold piece that begins at the last breakpoint in (the sample before a, a] (else at a) and ends at the first
// breakpoint in [b, the sample after b] (else at b; a CONSTANT key's step lies at the key's own sample, after the
// pre-key sample that ends the run), joined C0 on both sides unless that breakpoint says otherwise; breakpoints inside
// a hold piece are dropped. `times`: every sample time, ascending (the holds' neighbours).
struct PiecePlan {
  int64_t begin_ns = 0, end_ns = 0;
  Continuity join = Continuity::kC0;
  bool hold = false;
};
std::vector<PiecePlan> plan_pieces(int64_t t0, int64_t t1, std::span<const Breakpoint> breakpoints,
                                   std::span<const std::pair<int64_t, int64_t>> holds, std::span<const int64_t> times);

// The knot spacing of a piece (§4.7.6): clamp(2 x the median spacing of its samples, h_floor, h_cap), its own length
// when shorter (at least 1 ns).
int64_t knot_spacing(std::span<const int64_t> times, int64_t begin, int64_t end, int64_t h_floor, int64_t h_cap);

// A body's fitted motion (§3.6 session.json, §3.8): its pieces, holds and breakpoints, the validation maximum and the
// float32 floor of the validation samples; cv_px and gravity_ratio are frame-state mode's (BL-10).
struct PiecewiseFit {
  int64_t span_begin_ns = 0, span_end_ns = 0;
  std::vector<FitPiece> pieces;
  std::vector<std::pair<int64_t, int64_t>> holds;  // [begin, end) of the hold pieces
  std::vector<Breakpoint> breakpoints;
  double max_validation_px = 0.0, floor_px = 0.0, cv_px = 0.0, gravity_ratio = 0.0;
  int64_t validation_samples = 0;
  double max_join_px = 0.0;     // the largest C0 pseudo-sample residual before the rigid moves
  double max_reach_m = 0.0;     // the largest |p| of the samples (MOT-FAR, blender.md §6.1)
};

// The C1 trajectory (CONTRACTS.md C1; positions relative to origin_world): one spline when there is one piece, else
// the pieces table with each piece's control rows; span [span_begin, span_end].
render::Trajectory to_trajectory(const PiecewiseFit& fit, const std::array<double, 3>& origin_world);

}  // namespace balla::scene::motion
