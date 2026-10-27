// A body's decoded samples and its breakpoints (docs/design/phase3/blender.md §3.6, §3.8): what the sampler, the
// detection (pipeline.h, detect.h) and the piecewise fit (fit_piecewise.h) pass between them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "scene/motion/keys.h"
#include "scene/motion/poses.h"

namespace balla::scene::motion {

// A breakpoint (§3.6 session.json): its time (snapped: a declared key's, exact up to q(F)), class, the jump at it and
// the slope change (px, px/s), and its timing bound: q(F) when snapped, (eps/8)/dv for a kink, the bisection's last
// bracket for an unsnapped step.
struct Breakpoint {
  int64_t t_ns = 0;
  Continuity cls = Continuity::kC0;
  double jump_px = 0.0, dv_pxps = 0.0;
  bool declared = false, snapped = false;
  double dt_bound_ns = 0.0;
};

// One body's samples, ascending in time unless stated.
struct Track {
  std::vector<int64_t> t_ns;
  std::vector<rot::Q> q;
  std::vector<rot::V> p;
  std::vector<double> w_pos, w_rot, floor1;  // PoseBatch's error weights and float32 floor (g = 1)
  bool light_near = false;                   // LGT-MOVE-NEAR at one of the samples

  size_t size() const { return t_ns.size(); }
  void push(const Track& o, size_t i);
  void push(const PoseBatch& b, size_t i, int32_t body);
};

}  // namespace balla::scene::motion
