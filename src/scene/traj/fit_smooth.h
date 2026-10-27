// Robust smoothing fit of measured poses (ballasim.traj.fit.fit_smoothing; DESIGN §5.4, M2 §C.0-C.6), bitwise as
// the Python: gating G-a, round 0 with lambda by GCV, IRLS rounds, gating G-b, holds, the C1 trajectory and the
// diagnostics. Only the EVIMO2 adapter's path (no fixed controls): the authored path's interpolation stays in Python.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "scene/rotation.h"
#include "scene/traj/gating.h"
#include "scene/traj/holds.h"

namespace balla::scene::traj {

// SmoothingOptions, every field given (the Python dataclass's defaults are the schema's and stay there).
struct SmoothingOptions {
  int order = 6;
  std::optional<int64_t> h_ns;
  std::optional<std::pair<int64_t, int64_t>> span_ns;
  double loss_c2 = 0;
  bool gate = true;
  double gate_sigma_k = 0, gate_rot_floor_rad = 0, gate_pos_floor_m = 0, mahalanobis_max = 0;
  std::optional<double> lambda_pos, lambda_rot;
  double gcv_lo = 0, gcv_hi = 0;
  bool detect_holds = true;
  double hold_max_disp_m = 0;
  int64_t hold_min_ns = 0;
  double hold_moving_mps = 0;
  std::string hold_rule = "sphere";
  double hold_max_disp_px = 0;
  std::optional<HoldView> hold_view;
  double branch_jump_rad = 0, branch_rate_factor = 0;
  rot::V center_body_m{0, 0, 0};
  double radius_m = 0.1;
  double sigma_floor_pos_m = 1e-6, sigma_floor_rot_rad = 1e-6;
  int max_irls = 60;
};

struct GcvInfo {
  bool searched = false;   // false: a fixed lambda (_fixed_lambda_info)
  std::vector<double> log10_grid, gcv;
  bool at_grid_edge = false;
  double edf = 0, rss = 0;
};

struct FitDiagnostics {
  int64_t num_samples = 0;
  bool gated = false;
  GateResult gate;
  std::vector<int64_t> gate_rejected_t_ns;
  int64_t rotation_alias_warnings = 0;
  int order = 0;
  int64_t h_ns = 0, t0_ns = 0, num_ctrl = 0;
  std::pair<int64_t, int64_t> span_ns;
  double lambda_pos_u = 0, lambda_rot_u = 0, lambda_pos_si = 0, lambda_rot_si = 0, q_pos = 0, q_rot = 0;
  double cutoff_hz_pos = 0, cutoff_hz_rot = 0, edf_pos = 0, edf_rot = 0;
  GcvInfo gcv_pos, gcv_rot;
  double sigma_pos_m = 0, sigma_rot_rad = 0, rms_pos_m = 0, rms_rot_rad = 0, max_pos_m = 0, max_rot_rad = 0;
  std::vector<int64_t> irls_iterations;
  double min_weight = 0, mean_weight = 0;
  int64_t num_kept = 0;
  std::vector<int64_t> gb_rejected_t_ns;
  std::string hold_rule;
  std::vector<std::pair<int64_t, int64_t>> holds_ns, presence_holds_ns, gaps_ns;
  std::vector<StaticInterval> static_intervals;
  std::vector<RejectedCandidate> rejected_static_candidates;
  double still_max_displacement_m = 0, max_increment_rad = 0;
  bool is_static = false;
  // The presence fallback (fit_smoothing): the first attempt's LinAlgError and the presence groups (control ranges)
  // tied from round 0 on the second. Reported only when taken.
  bool presence_fallback = false;
  std::string presence_fallback_error;
  std::vector<std::pair<int64_t, int64_t>> presence_fallback_groups;
};

struct FitResult {
  int order = 6;
  int64_t t0_ns = 0, h_ns = 0;
  std::vector<rot::Q> rot;
  std::vector<rot::V> pos;
  std::vector<std::pair<int64_t, int64_t>> holds;
  std::pair<int64_t, int64_t> span_ns;
  std::vector<uint8_t> keep;
  std::vector<double> weights;
  FitDiagnostics diag;
};

// fit_smoothing(samples, options) for samples as PoseSamples holds them: t strictly increasing, q normalized once
// (pose_samples_q), finite; check_pose_samples refuses what PoseSamples refuses (it does not normalize again: that
// would move last bits). Throws TrajectoryError, LinAlgError (errors.h) or std::invalid_argument where the Python
// raises. Runs its BLAS calls on one OpenBLAS thread (blas::OneThread) and refuses a result computed while another
// caller changed that count.
FitResult fit_smoothing(std::span<const int64_t> t_ns, std::span<const rot::Q> q, std::span<const rot::V> p,
                        const SmoothingOptions& opts);
void check_pose_samples(std::span<const int64_t> t_ns, std::span<const rot::Q> q, std::span<const rot::V> p);
// PoseSamples.__post_init__'s q / np.linalg.norm(q, axis=1, keepdims=True).
std::vector<rot::Q> pose_samples_q(std::span<const rot::Q> q);
// The fitted trajectory's pose at t (ballasim.traj.evaluate through the core's fp64 spline: _core.spline_eval_f64).
std::pair<rot::Q, rot::V> evaluate_pose(const FitResult& fit, int64_t t_ns);

}  // namespace balla::scene::traj
