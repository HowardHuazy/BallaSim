// The sensor census (PHASE2 P2-E E0; docs/dev/SENSOR.md "Sensor census"): counters of the cell classes and bridge draws
// that decide P2-E's increments, compiled only into a CPU-only study build (CMake BALLASIM_SENSOR_CENSUS=ON, which
// refuses CUDA). In every other build BALLA_SENSOR_CENSUS is 0 and pixel.h compiles none of it: KernelStats keeps its
// fields and the kernel its code. The census only counts: a census build's CPU fp32 replay equals the GPU's bit for bit.
//
// MB-2 (the orbit cell's eligibility, E3): every whole cell that advance_slice starts is classed by the orbit entry's
// checks in the plan's order (p2e.md §3 "Entry"), by the first that fails: a sample inside the cell or a ramp (held
// values not constant), glare that varies over the call, no live modulated group (and not quiet), more than one, a
// pulse train, blind at the cell's start, the high-pass filter. A cell that passes them ("cheap") is settled by proxy
// when the input has been held for at least ln(Delta / (eps_h / 4)) / min(k Ibar, a2, a3) since its last change Delta
// (the total variation of ln Ibar over the change, Ibar the mean photocurrent; init counts as a change of the
// modulation depth s |a|_1), and eligible when also settled. within_radius is the entry's second, coefficient-free
// check on the state itself: |z Ibar - 1| + |s2 - xbar| + |s3 - xbar| <= 3 d / (1 - d) + eps_h / 4 (d = s |a|_1, xbar =
// ln(Ibar / i_scale)). An eligible cell is reachable by the lean pass while every earlier cell of the call was quiet or
// eligible (the GPU's quiet pass runs a window from its start; a call is a window), else only by the general pass.
//
// MB-1 (transient diffusion, E1): lattice draws with rho != 1, diffusion_weight's pieces, and the bridges drawn by the
// walk and by n3_at, with the chol factorizations an E1a per-walk level cache or an E1b per-(pixel, cell) level cache
// would still need (distinct rho != 1 levels per walk, per cell) and the distinct nodes per cell (levels <= 6).
#pragma once

#include <stdint.h>

#if defined(BALLASIM_SENSOR_CENSUS) && BALLASIM_SENSOR_CENSUS
#define BALLA_SENSOR_CENSUS 1
#include <algorithm>
#include <cmath>
#else
#define BALLA_SENSOR_CENSUS 0
#endif

namespace balla::sensor {

// Per pixel, across calls: the start of the input's current hold and the change that ended before it (log-e).
struct CensusPixel {
  int64_t t_hold = INT64_MIN;  // unknown (a loaded state): held for ever
  float delta = 0.0f;
};

// The counters, by index (SensorCensus::n; names: kCensusNames, the keys of stats()'s "census" dict).
enum CensusCounter : int {
  // MB-2: whole cells started by advance_slice, by class (each cell counts once).
  kCnCells,
  kCnQuiet,
  kCnRejNoise,        // noise off or leaf level 0 (no quiet or orbit cell)
  kCnRejSample,       // a sample inside the cell, or a ramp through it
  kCnRejGlare,        // glare varies over the call
  kCnRejStatic,       // no live modulated group, and not quiet
  kCnRejMulti,        // more than one live modulated group
  kCnRejPulse,        // the modulated group is a pulse train
  kCnRejBlind,
  kCnRejHpf,
  kCnCheap,           // passed the cheap rejections
  kCnRejUnsettled,    // cheap, not settled by proxy
  kCnEligibleLean,    // cheap and settled, reachable by the lean pass
  kCnEligibleGeneral, // cheap and settled, after a general cell of the call
  kCnEligibleFired,   // eligible cells in which an event fired (the orbit cell hands them back)
  kCnWithinRadius,    // cheap cells within the coefficient-free orbit radius
  kCnEligibleWithin,  // eligible cells within it
  // MB-1.
  kCnCellsRho,        // lattice draws with rho != 1 (split_step)
  kCnDwConst,         // diffusion_weight on constant pieces
  kCnDwRamp,          // ... on ramps
  kCnWalks,
  kCnBridgesWalk,
  kCnBridgesWalkRho,
  kCnN3Calls,
  kCnBridgesN3,
  kCnBridgesN3Rho,
  kCnFactorWalk,      // distinct rho != 1 levels per walk (E1a's factorizations)
  kCnFactorCell,      // distinct rho != 1 levels per (pixel, cell) over its walks and n3_at (E1b's)
  kCnNodesCell,       // distinct bridge nodes per (pixel, cell) (leaf level <= 6)
  kCensusCounters
};

inline constexpr const char* kCensusNames[kCensusCounters] = {
    "cells",          "quiet",          "rej_noise",        "rej_sample",     "rej_glare",
    "rej_static",     "rej_multi",      "rej_pulse",        "rej_blind",      "rej_hpf",
    "cheap",          "rej_unsettled",  "eligible_lean",    "eligible_general", "eligible_fired",
    "within_radius",  "eligible_within", "cells_rho",       "dw_const",       "dw_ramp",
    "walks",          "bridges_walk",   "bridges_walk_rho", "n3_calls",       "bridges_n3",
    "bridges_n3_rho", "factor_walk",    "factor_cell",      "nodes_cell"};

struct SensorCensus {
  uint64_t n[kCensusCounters] = {};
  CensusPixel* px = nullptr;  // SensorModel's per-pixel side state (not summed)

  void add(const SensorCensus& o) {
    for (int i = 0; i < kCensusCounters; ++i) n[i] += o.n[i];
  }
};

// The bridge draws of one walk (pixel.h Found).
struct CensusDraws {
  uint32_t rho_levels = 0;  // levels with a rho != 1 bridge
  uint64_t nodes = 0;       // nodes drawn, bit (t_mid - t_c0) >> (16 - leaf level)
};

}  // namespace balla::sensor
