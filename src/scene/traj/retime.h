// Re-timing of Vicon stamps onto their uniform frame grid (ballasim.traj.retime; G1 rec. 2, M2 §C.1): frame indices
// from the rounded stamp differences, t = a + b n by Huber IRLS (np.linalg.lstsq), bitwise as the Python.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace balla::scene::traj {

struct Retiming {
  std::vector<int64_t> t_raw_ns;
  std::vector<int64_t> frame_index;
  std::vector<int64_t> t_ns;
  double a_ns = 0, b_ns = 0, residual_std_ns = 0;
  int64_t num_gaps = 0;
};

Retiming retime_stamps(std::span<const int64_t> t_raw_ns, int64_t nominal_period_ns = 5'000'000,
                       double huber_k = 1.345, int iterations = 20);

}  // namespace balla::scene::traj
