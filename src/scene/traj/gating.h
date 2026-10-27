// Model-free outlier gating of pose samples, stage G-a (ballasim.traj.gating; M2 §C.2): leave-one-out predictions
// from robust (Tukey IRLS) quadratics in time, MAD thresholds, a second pass from unflagged neighbours and the
// minority orientation branches of persistent flips, bitwise as the Python.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "scene/rotation.h"

namespace balla::scene::traj {

struct GateOptions {
  double period_ns = 0;
  double sigma_k = 8.0;
  double rot_floor_rad = 0;
  double pos_floor_m = 0;
  double branch_jump_rad = 0;
  double branch_rate_factor = 0;
};

struct GateResult {
  std::vector<uint8_t> keep;
  std::vector<double> err_rot, err_pos;
  double threshold_rot = 0, threshold_pos = 0;
  int64_t half_window = 0;
  double sigma_rot = 0, sigma_pos = 0;
  int64_t num_branch_rejected = 0;
};

GateResult gate_poses(std::span<const int64_t> t_ns, std::span<const rot::Q> q, std::span<const rot::V> p,
                      const GateOptions& opts);

// Kept samples outside the largest orientation branch (gating.minority_branches).
std::vector<uint8_t> minority_branches(std::span<const int64_t> t_ns, std::span<const rot::Q> q,
                                       std::span<const uint8_t> keep, double jump_rad, double rate_factor);

}  // namespace balla::scene::traj
