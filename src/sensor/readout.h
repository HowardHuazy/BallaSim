// Readout of a threshold crossing (M3 F.1-F.2), shared by the host and the device: the report is
// delayed by d0, floored to the free-running counter quantum q (whose phase is drawn per camera)
// and expressed on the sensor clock t + offset + drift t.
#pragma once

#include <math.h>
#include <stdint.h>

#include "core/hd.h"
#include "core/time/time.h"

namespace balla::sensor {

struct Readout {
  int64_t quantum_ns;  // 0: no quantization
  int64_t delay_ns;
  int64_t phase_ns;    // counter phase in [0, quantum)
  int64_t clock_offset_ns;
  double clock_drift_ppm;
};

BALLA_HD inline int64_t reported_time(const Readout& r, int64_t t_cross_ns) {
  int64_t t = t_cross_ns + r.delay_ns;
  if (r.quantum_ns > 0) t = r.phase_ns + r.quantum_ns * time::floor_div(t - r.phase_ns, r.quantum_ns);
  return t + r.clock_offset_ns + llround(r.clock_drift_ppm * 1e-6 * static_cast<double>(t));
}

}  // namespace balla::sensor
