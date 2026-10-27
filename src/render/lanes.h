// Fixed reduction order shared by the CPU executor and the CUDA cluster kernel (DESIGN §6.9:
// "one warp per pixel-sample reduces its sub-sample records in fixed lane order"; §7.9: CPU and GPU
// shading bitwise equal).
//
// The sub-samples of a cluster are dealt to 32 lanes by index k: lane k mod 32 sums its values in
// increasing k, starting from +0. The 32 lane sums are then combined by the xor butterfly of
// __shfl_xor_sync with offsets 16, 8, 4, 2, 1, in which lane l adds v[l] + v[l ^ off]; IEEE addition
// is commutative, so every lane ends with the same total. butterfly_sum runs the combination on one
// CPU thread; the warp version lives in the CUDA kernel (render/cuda/device_executor.cu).
#pragma once

#include <stdint.h>

#include "core/hd.h"

namespace balla::render {

inline constexpr uint32_t kLanes = 32;

// Butterfly over 32 lane values (in place); returns the total every lane would hold.
BALLA_HD inline float butterfly_sum(float* v) {
  for (uint32_t off = kLanes / 2; off > 0; off /= 2) {
    float next[kLanes];
    for (uint32_t l = 0; l < kLanes; ++l) next[l] = v[l] + v[l ^ off];
    for (uint32_t l = 0; l < kLanes; ++l) v[l] = next[l];
  }
  return v[0];
}

}  // namespace balla::render
