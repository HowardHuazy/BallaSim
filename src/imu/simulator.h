// The IMU simulator (docs/design/phase3/outputs.md §3.3, §4.1-4.6): samples one camera's IMU from its rig trajectory
// in blocks, bitwise independent of how the span is cut into blocks. Host, fp64, single-threaded.
//
// Per sample k: the instant tau_k of the sample clock (processes.h, §4.2); the truth of truth.h at tau_k; per sensor
// the filter input u = clip_pre(M x + b0) through the first-order filter of filter.h (§4.3), plus the bias processes
// and the noise (§4.4); quantization and saturation (§4.5); the stamp through the camera's readout counter, dropped
// below the event file's t_offset (§4.6).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "imu/filter.h"
#include "imu/params.h"
#include "imu/processes.h"
#include "imu/truth.h"
#include "render/pose.h"
#include "sensor/readout.h"

namespace balla::imu {

// The camera's readout counter without its delay (§4.6): t_ns = reported_time(clock, delay, t_true).
struct CameraClock {
  int64_t quantum_ns = 0, phase_ns = 0, offset_ns = 0;
  double drift_ppm = 0;
  friend bool operator==(const CameraClock&, const CameraClock&) = default;
};

// The counter of camera `camera_index` as SensorModel builds it (sensor_model.cpp): q = llround(readout_quantum_s 1e9)
// and, when q > 0, phase = Philox(sensor_seed, kReadoutPhase, camera, 0, 0).v[0] mod q; offset and drift as given.
// Refused outside the sensor rows' hard ranges (readout_quantum_s in [0, 1), clock_drift_ppm in (-1e4, 1e4)) and, as
// SensorModel does, for camera_index > 255 (purpose.h keeps the camera's low 8 bits).
CameraClock camera_clock(double readout_quantum_s, int64_t clock_offset_ns, double clock_drift_ppm,
                         uint64_t sensor_seed, uint32_t camera_index);

// Quantum 0 and phase 0: the run's sensor clock clock_c (offset and drift only), which maps true times onto the events'
// clock without the counter (frame times, §4.7).
CameraClock sensor_clock_of(const CameraClock&);

// Refused (naming `what`) unless quantum_ns >= 0, 0 <= phase_ns < quantum_ns (0 without a quantum) and drift_ppm is
// finite in (-1e4, 1e4), the sensor row's hard range: then reported_time and each of its steps are non-decreasing in t.
void check_clock(const CameraClock&, const char* what);

// readout.h's reported_time through `clock` with the delay `delay_ns`: (t + delay) floored to the counter, then
// + offset + llround(drift 1e-6 t). False when a step overflows int64 (out untouched).
bool reported_time_checked(const CameraClock& clock, int64_t delay_ns, int64_t t_true_ns, int64_t* out);
// As reported_time_checked; Refused on overflow.
int64_t reported_time(const CameraClock& clock, int64_t delay_ns, int64_t t_true_ns);

struct ImuInput {
  // Nodes (and pieces) owned by the caller (the binding owns them). A piecewise rig (num_pieces > 1, P3-E2; §4.3.4)
  // brings the join class of each piece with it: joins[p] in {-1 step, 0 C0, 1 C1} (C1 Trajectory.pieces[p].join;
  // piece 0: 0), checked by pieces_error. A single-spline rig has no joins.
  render::TrajectoryView rig;
  std::vector<int8_t> joins;
  render::RigidPose T_rig_cam;
  CameraClock clock;              // the camera's counter (camera_clock); the stamps' delay is imu.stamp_offset_ns
  Vec3d g_W;                      // world gravity, resolved by the caller (§4.1)
  uint32_t camera_index = 0;      // the Philox camera of the IMU's draws (keyed by imu.seed); < 256 (purpose.h)
  int64_t t_begin_ns = 0, t_end_ns = 0;  // the run span, true clock: samples have t_true in [t_begin, t_end)
  int64_t t_offset_us = 0;        // the event file's /t_offset; samples stamped below 1000 t_offset_us are dropped
};

// /flags bits (C5, §3.5).
inline constexpr uint8_t kFlagRailGyro = 0x07;   // bits 0-2: gyro x/y/z clamped at a rail (§4.5)
inline constexpr uint8_t kFlagRailAccel = 0x38;  // bits 3-5: accel x/y/z clamped at a rail
inline constexpr uint8_t kFlagWarmup = 0x40;     // bit 6: incomplete filter history (late rig start; after a step join)
inline constexpr uint8_t kFlagImpact = 0x80;     // bit 7: t_k - t_b in [0, W) of a join t_b of any class (A11)

// N samples, structure of arrays, appended in time order; vectors of 3 (or 4) per sample, row-major.
struct ImuBlock {
  std::vector<int64_t> t_true_ns;  // the instant, rounded to ns (true clock)
  std::vector<int64_t> t_ns;       // the stamp on the camera's sensor clock (§4.6)
  std::vector<double> gyro, accel;  // published values (rad/s, m/s^2): count lsb, or the unquantized value
  std::vector<int32_t> counts_gyro, counts_accel;  // raw counts when that sensor's lsb is set, else empty
  std::vector<uint8_t> flags;
  std::vector<double> omega_true, f_true;          // the truth at the exact instant, before filter and errors
  std::vector<double> bias_gyro, bias_accel;       // b0 + b_rw + b_gm at the sample
  std::vector<double> q_W_I, p_W_I;                // pose of the IMU frame: (w, x, y, z); p relative to origin_world
  size_t size() const { return t_true_ns.size(); }
};

// Counters over the samples returned so far (dropped_before_t_offset: over the samples not returned; step_resets: over
// every sample the filters stepped to, returned or not, since a reset is the filter's event: one per step join, counted
// when the first filtered sensor re-initializes).
struct ImuStats {
  int64_t dropped_before_t_offset = 0, warmup_samples = 0, impact_samples = 0, step_resets = 0;
  std::array<int64_t, 6> pre_clip_engaged{};  // samples whose filter input was clipped (clip_stage both), per axis
};

class ImuSimulator {
 public:
  // validate() and the input checks; errors throw Refused naming the key or the input.
  ImuSimulator(const ImuParams&, const ImuInput&);
  // The samples with t_true in [the previous call's end, min(t_end_ns, the run's t_end)), less those stamped below
  // the event file's t_offset; bitwise independent of the chunking. A t_end_ns at or before the previous end returns
  // an empty block.
  ImuBlock advance_to(int64_t t_end_ns);
  const ImuStats& stats() const { return stats_; }
  // Whether blocks carry sensor s's counts (its lsb is set): 0 gyroscope, 1 accelerometer.
  bool has_counts(int s) const { return ch_[static_cast<size_t>(s)].quantized; }

 private:
  struct Channel {
    bool filtered = false;
    double tau_ns = 0, d_ns = 0;  // tau_c and the pure delay d
    Vec3d y{0, 0, 0};             // the filter output at `prev`
    double prev = 0;              // the end of the last integrated interval, ns relative to t_begin
    double warm_until = -1;       // WARMUP before this (late rig start, §4.3.3); -1: never
    size_t next_join = 0;         // the first join whose delayed time t_b + d the filter has not reached (A11)
    // The error model (§4.4, §4.5).
    bool affine = false;          // M != I or b0 != 0
    Mat3d M{};
    Vec3d b0{0, 0, 0};
    double rail = 0;              // the pre-clip bound 2^(bits-1) lsb (clip_stage both with an lsb); 0: none
    bool quantized = false;
    double lsb = 0;
    SensorNoise noise;
  };
  // A join of the piecewise rig (p >= 1; §4.3.4), t_b = begin_p in ns relative to t_begin.
  struct Join {
    double t_b = 0;
    int8_t cls = 0;      // kJoinStep, kJoinC0 or kJoinC1
    Vec3d impulse{0, 0, 0};  // C0: J_I, the specific-force impulse (m/s, IMU axes); else 0
    bool reset_counted = false;
  };
  Truth truth_rel(double t_rel_ns) const;
  // Brings channel s (filtered) to tau: the step resets and C0 impulses of the joins its delayed time t_b + d passes,
  // and the integral of [prev, tau] cut at the knots and joins (§4.3.2, §4.3.4).
  void step_filter(int s, double tau, uint8_t* clipped);
  // The join flags of a sample at tau (kFlagImpact, and kFlagWarmup after a step join).
  uint8_t join_flags(double tau);
  // u = clip_pre(M x + b0) of sensor s from the truth; ORs the clipped axes into *clipped.
  Vec3d filter_input(int s, const Truth& t, uint8_t* clipped) const;

  ImuParams params_;
  ImuInput in_;
  Mount mount_;
  int64_t span_ns_ = 0;  // t_end - t_begin of the run
  SampleClock clock_;
  double prev_tau_ = 0;   // tau_{k-1} (ns relative to t_begin)
  std::array<Channel, 2> ch_{};  // gyro, accel
  int64_t count_lo_ = 0, count_hi_ = 0;  // the ADC range -2^(bits-1), 2^(bits-1) - 1
  sensor::Readout readout_{};
  std::vector<Join> joins_;       // the rig's joins, ascending in t_b
  double join_window_ = 0;        // W (§3.5): max over filtered sensors of 20 tau_c + d; one period when none
  bool join_warmup_ = false;      // a filtered sensor exists: a step join warms up
  size_t flag_from_ = 0;          // the first join whose window [t_b, t_b + W) may still hold the next sample
  bool drop_all_ = false;          // 1000 t_offset_us is above INT64_MAX: every stamp is below it
  int64_t drop_below_ = 0;         // 1000 t_offset_us (when representable)
  ImuStats stats_{};
  // Scratch buffers of one interval.
  Panels panels_;
  std::vector<double> cuts_;
  std::vector<Vec3d> u_;
};

}  // namespace balla::imu
