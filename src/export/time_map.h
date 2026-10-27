// The export time rules (docs/design/phase3/outputs.md §4.7) and the writers' input views (§3.3). Every time a writer
// receives is on the camera's sensor clock: the events' t_reported, the IMU's C5 /t_ns. A writer maps it by its
// format family's rule:
//   T1a (us, file-relative)  RAW EVT 2.0/2.1/3.0, DAT, the IMU CSV sidecar, the EVT 3.0 words of a ROS EventPacket:
//                            u = floor_div(t_ns, 1000) - origin_us
//   T2 (ns)                  ROS1 (A6) and MCAP (A7) stamps: t_ns - 1000 origin_us + epoch_ns
//   T1b (us, epoch)          AEDAT4 (A8) timestamps: floor_div(t_ns, 1000) - origin_us + epoch_ns / 1000
// epoch_ns is a multiple of 1000 for every format, so every ns stamp shares its sub-us part with t_reported.
// Header-only.
#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

#include "core/time/time.h"
#include "export/errors.h"

namespace balla::exporter {

// One batch of a camera's events in output order (t_reported non-decreasing), structure of arrays.
struct Events {
  const uint16_t* x = nullptr;
  const uint16_t* y = nullptr;
  const uint8_t* p = nullptr;   // 1 = ON
  const int64_t* t_ns = nullptr;  // t_reported, ns on the camera's sensor clock
  size_t n = 0;
};

// One batch of a camera's IMU samples (C5): stamps on the sensor clock, published values N x 3 row-major, and the raw
// counts when that sensor quantizes (null otherwise).
struct ImuView {
  const int64_t* t_ns = nullptr;
  const double* gyro = nullptr;
  const double* accel = nullptr;
  const int32_t* counts_gyro = nullptr;
  const int32_t* counts_accel = nullptr;
  size_t n = 0;
};

struct TimeMap {
  int64_t origin_us = 0;  // the export origin: min over the cameras of floor(clock_c(t0) / 1000), else min /t_offset
  int64_t epoch_ns = 0;   // added by T1b and T2; recorded by T1a
};

// Refused unless epoch_ns is a multiple of 1000 and epoch_ns + 1000 origin_us (the header date's instant) fits int64.
inline void check(const TimeMap& m) {
  if (m.epoch_ns % 1000 != 0) {
    throw Refused("epoch_ns = " + std::to_string(m.epoch_ns) + ": must be a multiple of 1000 (every format, §4.7)");
  }
  int64_t origin_ns = 0, at = 0;
  if (__builtin_mul_overflow(m.origin_us, int64_t{1000}, &origin_ns) ||
      __builtin_add_overflow(origin_ns, m.epoch_ns, &at)) {
    throw Refused("epoch_ns + 1000 origin_us overflows int64");
  }
}

// T1a: microseconds after the origin, the sub-us part of t_ns dropped.
inline int64_t t1a_us(const TimeMap& m, int64_t t_ns) { return time::floor_div(t_ns, int64_t{1000}) - m.origin_us; }

// T1b: AEDAT4's int64 microseconds, the epoch added. Under check(), |origin_us| < 2^54, so |T1a| < 2^55 and
// |epoch_ns / 1000| < 2^54: it cannot overflow.
inline int64_t t1b_us(const TimeMap& m, int64_t t_ns) { return t1a_us(m, t_ns) + m.epoch_ns / 1000; }

// T2: the exact ns stamp of the ROS formats; false when it overflows int64 (the ROS writers refuse it, naming it).
inline bool t2_ns(const TimeMap& m, int64_t t_ns, int64_t* out) {
  int64_t origin_ns = 0, r = 0;
  return !__builtin_mul_overflow(m.origin_us, int64_t{1000}, &origin_ns) &&
         !__builtin_sub_overflow(t_ns, origin_ns, &r) && !__builtin_add_overflow(r, m.epoch_ns, out);
}

// "YYYY-MM-DD HH:MM:SS" (UTC, proleptic Gregorian) of the second that holds ns since the Unix epoch.
inline std::string utc_date(int64_t ns) {
  namespace c = std::chrono;
  const int64_t s = time::floor_div(ns, int64_t{1000000000});
  const int64_t days = time::floor_div(s, int64_t{86400});
  const int64_t sec = s - 86400 * days;
  const c::year_month_day d{c::sys_days{c::days{days}}};
  char buf[64];
  const int n =
      std::snprintf(buf, sizeof buf, "%04d-%02u-%02u %02d:%02d:%02d", static_cast<int>(d.year()),
                    static_cast<unsigned>(d.month()), static_cast<unsigned>(d.day()), static_cast<int>(sec / 3600),
                    static_cast<int>(sec / 60 % 60), static_cast<int>(sec % 60));
  return std::string(buf, static_cast<size_t>(n));
}

// The RAW and DAT header date (§3.7): derived from the time rule, not the wall clock, so the bytes are reproducible.
inline std::string header_date(const TimeMap& m) { return utc_date(m.epoch_ns + 1000 * m.origin_us); }

// clock_c(t) = t + offset + llround(drift 1e-6 t): the camera's sensor clock, sensor/readout.h's formula without the
// delay and the quantum (the engine's sensor_clock_ns, the clock of the event file). Refused on int64 overflow.
inline int64_t sensor_clock_ns(int64_t clock_offset_ns, double clock_drift_ppm, int64_t t_ns) {
  if (!(std::abs(clock_drift_ppm) < 1e4)) {  // the sensor row's hard range; then llround below cannot overflow
    throw Refused("sensor.clock_drift_ppm must be finite in (-1e4, 1e4)");
  }
  int64_t r = 0;
  const double drift = clock_drift_ppm * 1e-6 * static_cast<double>(t_ns);
  if (__builtin_add_overflow(t_ns, clock_offset_ns, &r) || __builtin_add_overflow(r, std::llround(drift), &r)) {
    throw Refused("the sensor clock at t = " + std::to_string(t_ns) + " ns overflows int64");
  }
  return r;
}

// A camera's export window on its sensor clock (§4.7, C6 clock.cameras): [clock_c(t0), clock_c(t1)) with an open end
// where t0 or t1 is absent, and the event writer's close bound clock_c(t_e + d0 + q) (engine.py), d0 and q rounded
// to ns as the sensor rounds them (llround).
struct CameraTimes {
  std::optional<int64_t> lo_ns, hi_ns;
  int64_t file_end_bound_ns = 0;
};

inline CameraTimes camera_times(int64_t clock_offset_ns, double clock_drift_ppm, double readout_delay_s,
                                double readout_quantum_s, int64_t t_end_ns, std::optional<int64_t> t0_ns,
                                std::optional<int64_t> t1_ns) {
  if (!(readout_delay_s >= 0 && readout_delay_s < 1) || !(readout_quantum_s >= 0 && readout_quantum_s < 1)) {
    throw Refused("sensor.readout_delay_s and sensor.readout_quantum_s must be in [0, 1) s");
  }
  CameraTimes c;
  if (t0_ns) c.lo_ns = sensor_clock_ns(clock_offset_ns, clock_drift_ppm, *t0_ns);
  if (t1_ns) c.hi_ns = sensor_clock_ns(clock_offset_ns, clock_drift_ppm, *t1_ns);
  int64_t t = 0;
  if (__builtin_add_overflow(t_end_ns, std::llround(readout_delay_s * 1e9) + std::llround(readout_quantum_s * 1e9),
                             &t)) {
    throw Refused("the run end plus the readout delay and quantum overflows int64");
  }
  c.file_end_bound_ns = sensor_clock_ns(clock_offset_ns, clock_drift_ppm, t);
  return c;
}

}  // namespace balla::exporter
