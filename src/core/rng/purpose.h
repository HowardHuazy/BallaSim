// Purpose-tag registry and counter layout for every random draw (DESIGN §12.1, S1 §5.2).
//
// A draw is keyed by the run seed and addressed by (purpose, camera, pixel, physical index),
// never by launch, thread, window or render time, so results do not depend on scheduling.
// The registry is append-only: never renumber or reuse a tag, because golden first outputs
// (tests/cpp/test_philox.cpp) pin every stream.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/rng/philox.h"

namespace balla::rng {

enum class Purpose : uint32_t {
  kSubsamplePattern = 0x0001,  // per-pixel Owen-Sobol scramble seed of the sub-sample pattern (§6.1)
  // Pixel model (DESIGN §8, docs/dev/SENSOR.md). S1's sensor-map block 0x0001-0x0006 collides with
  // kSubsamplePattern, so the frozen per-pixel maps share one tag and the index selects the map.
  kSensorMap = 0x0002,       // frozen per-pixel maps; index: map block
  kNoise = 0x0010,           // noise-lattice steps and bridge midpoints; index: node time (ns)
  kNoiseInit = 0x0011,       // initial stationary noise state; index: first lattice node time (ns)
  kStationaryProbe = 0x0012,  // stationary initial state: probe choice (v0), snapshot draw (v1); index: init time (ns)
  kLimiterDrop = 0x0021,     // rate limiter (sensor/limiter.h): an event's selection key; pixel, index: t_cross (ns)
  kReadoutPhase = 0x0022,    // per-camera phase of the readout counter; pixel 0, index 0
  kLimiterPhase = 0x0023,    // rate limiter: per-camera phase of its epoch when none is given; pixel 0, index 0
  // Area lights (P2-F F8a-1; plan p2f §4.6; render/scene_data.cpp, render/shade/lights.h).
  kLightSampleSet = 0x0031,  // scramble seed of a light's fixed point set Y_l (S1's "fixed light-sample sets"); camera 0,
                             // pixel: the light's index in the bundle, index 0
  kLightPadding = 0x0033,    // scramble seed of the padded pair that assigns sub-sample k of a pixel a point of Y_l;
                             // camera, pixel, index: the light's index in the bundle (0x0032 stays S1's pupil)
  // APS frames (Phase 3, docs/design/phase3/frames.md §3.8, §4.10; sensor/aps.h). Pixel y W + x of the full image.
  kApsMap = 0x0060,          // PRNU and DSNU maps; index: map block (0 PRNU, 1 DSNU)
  kApsShot = 0x0061,         // shot noise; index: (t_soe + 2^56) 64 + j, j < 64, t_soe the row's (true clock)
  kApsRead = 0x0062,         // read noise; index as kApsShot, j = 0
  // The IMU (Phase 3 outputs, docs/design/phase3/outputs.md §4.2, §4.4; S1 §5.2 moved them from 0x0030). Keyed by
  // imu.seed and the bundle camera; pixel s = 0 (gyroscope) or 1 (accelerometer) unless noted.
  kImuNoise = 0x0070,        // white noise and the shaped noise's steps; pixel s, index: sample k
  kImuBiasWalk = 0x0071,     // bias random walk (pixel s) and bias-instability steps (pixel 2 + s); index: sample k
  kImuBiasInit = 0x0072,     // turn-on bias (pixel s), bias instability (2 + s), shaped noise (4 + s); index 0
  kImuClock = 0x0073,        // sample clock: start phase (pixel 0, index 0), wander steps (pixel 1, index k) and
                             // stationary start (pixel 2, index 0)
};

// Counter layout: v0 = purpose << 8 | camera (camera < 256, purpose < 2^24), v1 = pixel,
// v2:v3 = 64-bit physical index (e.g. a noise-lattice node number). Key = the 64-bit run seed.
BALLA_HD inline U32x4 draw(uint64_t run_seed, Purpose purpose, uint32_t camera, uint32_t pixel,
                           uint64_t index) {
  const PhiloxKey key{static_cast<uint32_t>(run_seed), static_cast<uint32_t>(run_seed >> 32)};
  const U32x4 counter{{(static_cast<uint32_t>(purpose) << 8) | (camera & 0xFFu), pixel,
                       static_cast<uint32_t>(index), static_cast<uint32_t>(index >> 32)}};
  return philox4x32_10(counter, key);
}

}  // namespace balla::rng
