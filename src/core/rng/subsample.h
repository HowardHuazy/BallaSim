// Sensor-fixed sub-sample positions of a pixel (DESIGN §6.1).
//
// Sub-sample k of pixel p on camera c sits at u_k = T_h(xi_k), where xi_k is point k of the 2-D
// Owen-scrambled Sobol sequence seeded by hash(c, p) and T_h is the separable warp to the pixel
// response. The pattern never changes with time (common random numbers), and the first 2^m points
// are a (0, m, 2)-net, so N in {16, 64, 256, 1024} are nested prefixes of one sequence.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/vec.h"
#include "core/rng/purpose.h"
#include "core/rng/sobol.h"
#include "core/rng/warp.h"

namespace balla::rng {

// A 32-bit seed: by the birthday bound about n^2 / 2^33 pixel pairs of an n-pixel camera share a pattern (11 at
// VGA), scattered pixels whose sampling errors are then equal rather than independent.
BALLA_HD inline uint32_t subsample_pattern_seed(uint64_t run_seed, uint32_t camera, uint32_t pixel) {
  return draw(run_seed, Purpose::kSubsamplePattern, camera, pixel, 0).v[0];
}

// Offset of sub-sample k from the pixel centre, in pixels (x right, y down).
template <class T>
BALLA_HD Vec2<T> subsample_offset(const WarpTableView<T>& warp_table, uint32_t pattern_seed,
                                  uint32_t k) {
  uint32_t x, y;
  owen_sobol_2d(k, pattern_seed, &x, &y);
  return warp(warp_table, Vec2<T>{to_unit_interval<T>(x), to_unit_interval<T>(y)});
}

}  // namespace balla::rng
