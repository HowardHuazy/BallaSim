// GPU launch control of the sensor kernel (sensor_cuda.cu): the policy that bounds one launch and the
// counters that report what the launches took. Host-only plain data, visible to CPU-only builds so
// that SensorModel's interface does not depend on CUDA.
#pragma once

#include <cstdint>

namespace balla::sensor {

// Per window (at most 2^23 ns of a call) the quiet pass runs in pixel batches and the general pass in
// time slices, both sized from running cost estimates so that a launch takes about target_ms: the GPU
// also drives the desktop, and DESIGN §4.2 caps every kernel at 50 ms; a slice may take a few times
// its plan where a moving edge concentrates work (docs/dev/SENSOR.md "GPU launch control"). The
// bounds below exist for tests (forcing short windows, small batches or event-buffer overflows);
// batching and slicing never change a result.
struct GpuLaunchPolicy {
  double target_ms = 15.0;          // planned duration of one launch
  int64_t max_slice_ns = 0;         // upper bound of a window (a slice of the call), 0 = 2^23 ns
  uint32_t max_pixels = 0;          // upper bound of a general-pass launch's pixels, 0 = none
  uint64_t event_capacity = 1 << 18;  // initial device event buffer; it grows on overflow
  // PHASE2 P2-E E5, the per-slice sort: before each general-pass launch of a call with samples, the launch's pixels
  // are ordered by their own work in the time slice (slice_order.h slice_key) instead of the window's. Thread order
  // only: every result is the same either way. On since its replay A/B (docs/dev/BENCH.md "P2-E E5b"). The environment
  // variable BALLA_SENSOR_SLICE_SORT (0 or 1), when set, overrides it for every kernel object, at construction and at
  // each set_policy (docs/dev/SENSOR.md "GPU launch control").
  bool slice_sort = true;
};

struct GpuLaunchStats {
  uint64_t launches = 0;  // kernel launches, re-runs included
  uint64_t reruns = 0;    // launches repeated after an event-buffer overflow
  double total_ms = 0;    // summed launch durations
  double max_ms = 0;      // longest launch
  uint64_t slice_sorts = 0;  // launches ordered by the per-slice sort (GpuLaunchPolicy::slice_sort; in their time)
};

}  // namespace balla::sensor
