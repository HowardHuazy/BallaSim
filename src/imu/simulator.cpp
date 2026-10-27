// The IMU simulator's loop (docs/design/phase3/outputs.md §4.2-4.6); see simulator.h.
#include "imu/simulator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>
#include <vector>

#include "core/math/so3.h"
#include "core/pmath/explog.h"
#include "core/rng/purpose.h"
#include "core/time/time.h"

namespace balla::imu {

namespace {

std::string join(const std::vector<std::string>& parts, const char* sep) {
  std::string out;
  for (const std::string& p : parts) out += (out.empty() ? "" : sep) + p;
  return out;
}

void push3(std::vector<double>& out, const Vec3d& v) { out.insert(out.end(), {v.x, v.y, v.z}); }

// M = Exp(misalign_deg pi/180) (I + diag(scale) + Sym(nonorth)), Sym = [[0, xy, xz], [xy, 0, yz], [xz, yz, 0]]: the
// polar form R P with P acting in IMU axes (§4.4).
Mat3d error_matrix(const SensorErrors& e) {
  constexpr double kDeg = std::numbers::pi / 180;
  const Vec3d phi{e.misalign_deg[0] * kDeg, e.misalign_deg[1] * kDeg, e.misalign_deg[2] * kDeg};
  const Mat3d r = to_matrix(so3::exp(phi));
  const auto& s = e.scale;
  const auto& n = e.nonorth;
  const Mat3d p{{{1 + s[0], n[0], n[1]}, {n[0], 1 + s[1], n[2]}, {n[1], n[2], 1 + s[2]}}};
  return render::operator*(r, p);
}

bool identity(const Mat3d& m) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (m.m[i][j] != (i == j ? 1.0 : 0.0)) return false;
    }
  }
  return true;
}

}  // namespace

void check_clock(const CameraClock& c, const char* what) {
  const std::string w(what);
  if (c.quantum_ns < 0) throw Refused(w + ".quantum_ns must be >= 0");
  if (c.quantum_ns > 0 ? !(c.phase_ns >= 0 && c.phase_ns < c.quantum_ns) : c.phase_ns != 0) {
    throw Refused(w + ".phase_ns must lie in [0, quantum_ns) (0 without a quantum)");
  }
  if (!(std::abs(c.drift_ppm) < 1e4)) throw Refused(w + ".drift_ppm must be finite in (-1e4, 1e4)");
}

CameraClock camera_clock(double readout_quantum_s, int64_t clock_offset_ns, double clock_drift_ppm,
                         uint64_t sensor_seed, uint32_t camera_index) {
  if (!(readout_quantum_s >= 0 && readout_quantum_s < 1)) {
    throw Refused("sensor.readout_quantum_s = " + config::format_double(readout_quantum_s) + ": expected [0, 1) s");
  }
  if (!(std::abs(clock_drift_ppm) < 1e4)) {
    throw Refused("sensor.clock_drift_ppm = " + config::format_double(clock_drift_ppm) + ": expected (-1e4, 1e4)");
  }
  if (camera_index > 255) throw Refused("camera_index must be < 256 (purpose.h layout)");  // as SensorModel
  CameraClock c;
  c.quantum_ns = std::llround(readout_quantum_s * 1e9);
  if (c.quantum_ns > 0) {  // sensor_model.cpp: the per-camera phase of the free-running counter
    c.phase_ns = static_cast<int64_t>(rng::draw(sensor_seed, rng::Purpose::kReadoutPhase, camera_index, 0, 0).v[0] %
                                      static_cast<uint64_t>(c.quantum_ns));
  }
  c.offset_ns = clock_offset_ns;
  c.drift_ppm = clock_drift_ppm;
  return c;
}

CameraClock sensor_clock_of(const CameraClock& c) { return {0, 0, c.offset_ns, c.drift_ppm}; }

bool reported_time_checked(const CameraClock& c, int64_t delay_ns, int64_t t_true_ns, int64_t* out) {
  // The steps of sensor/readout.h, each checked: t + delay, the floor to the counter, + offset, + the drift term.
  int64_t t = 0;
  if (__builtin_add_overflow(t_true_ns, delay_ns, &t)) return false;
  if (c.quantum_ns > 0) {
    int64_t rel = 0, floored = 0;
    if (__builtin_sub_overflow(t, c.phase_ns, &rel)) return false;
    if (__builtin_mul_overflow(c.quantum_ns, time::floor_div(rel, c.quantum_ns), &floored)) return false;
    if (__builtin_add_overflow(c.phase_ns, floored, &t)) return false;
  }
  int64_t r = 0;
  const double drift = c.drift_ppm * 1e-6 * static_cast<double>(t);  // |drift| < 1e-2 |t|: llround cannot overflow
  if (__builtin_add_overflow(t, c.offset_ns, &r) || __builtin_add_overflow(r, std::llround(drift), &r)) return false;
  *out = r;
  return true;
}

int64_t reported_time(const CameraClock& c, int64_t delay_ns, int64_t t_true_ns) {
  check_clock(c, "clock");
  int64_t out = 0;
  if (!reported_time_checked(c, delay_ns, t_true_ns, &out)) {
    throw Refused("reported_time: the stamp of t = " + std::to_string(t_true_ns) + " ns overflows int64");
  }
  return out;
}

ImuSimulator::ImuSimulator(const ImuParams& params, const ImuInput& input) : params_(params), in_(input) {
  const Check check = validate(params_);
  if (!check.errors.empty()) throw Refused(join(check.errors, "; "));
  const render::TrajectoryView& rig = in_.rig;
  if (rig.order != 4 && rig.order != 6) throw Refused("rig trajectory: order must be 4 or 6");
  if (rig.spline.nodes == nullptr || rig.spline.num_segments < 1 || rig.spline.h_ns <= 0) {
    throw Refused("rig trajectory: needs at least one segment");
  }
  // A piecewise rig (P3-E2; §4.3.4): the pieces and their joins, and every join inside the span (the bundle's rule:
  // begin_0 is the span's begin and the last begin precedes its end).
  if (const std::string why = pieces_error(rig, in_.joins); !why.empty()) throw Refused("rig trajectory: " + why);
  for (int32_t p = 1; p < rig.num_pieces; ++p) {
    if (!(rig.span_begin_ns < rig.pieces[p].begin_ns && rig.pieces[p].begin_ns < rig.span_end_ns)) {
      throw Refused("rig trajectory: piece " + std::to_string(p) + " begins at " +
                    std::to_string(rig.pieces[p].begin_ns) + " ns, not inside the span (" +
                    std::to_string(rig.span_begin_ns) + ", " + std::to_string(rig.span_end_ns) + ") ns");
    }
  }
  if (!(in_.t_begin_ns < in_.t_end_ns)) throw Refused("run span: t_begin_ns must be below t_end_ns");
  if (in_.t_begin_ns < 0 && in_.t_end_ns > INT64_MAX + in_.t_begin_ns) {
    throw Refused("run span: t_end_ns - t_begin_ns overflows int64");
  }
  span_ns_ = in_.t_end_ns - in_.t_begin_ns;
  if (in_.t_begin_ns < rig.span_begin_ns || in_.t_end_ns > rig.span_end_ns) {
    throw Refused("run span [" + std::to_string(in_.t_begin_ns) + ", " + std::to_string(in_.t_end_ns) +
                  ") ns is not inside the rig trajectory's span [" + std::to_string(rig.span_begin_ns) + ", " +
                  std::to_string(rig.span_end_ns) + "] ns");
  }
  if (!(std::isfinite(in_.g_W.x) && std::isfinite(in_.g_W.y) && std::isfinite(in_.g_W.z))) {
    throw Refused("g_W must be finite");
  }
  // Philox keeps the camera's low 8 bits (purpose.h): camera 256 + c would reuse camera c's draws.
  if (in_.camera_index > 255) throw Refused("camera_index must be < 256 (purpose.h layout)");
  // §4.6: stamps through the camera's counter with the delay imu.stamp_offset_ns. Each step of reported_time is
  // non-decreasing in t (check_clock), so stamps that fit at both ends of the run span fit throughout.
  check_clock(in_.clock, "clock");
  int64_t first = 0, last = 0;
  if (!reported_time_checked(in_.clock, params_.stamp_offset_ns, in_.t_begin_ns, &first) ||
      !reported_time_checked(in_.clock, params_.stamp_offset_ns, in_.t_end_ns - 1, &last)) {
    throw Refused("clock: the stamps of the run span overflow int64");
  }
  readout_ = {in_.clock.quantum_ns, params_.stamp_offset_ns, in_.clock.phase_ns, in_.clock.offset_ns,
              in_.clock.drift_ppm};
  if (__builtin_mul_overflow(in_.t_offset_us, int64_t{1000}, &drop_below_)) {  // beyond int64: all or none dropped
    drop_all_ = in_.t_offset_us > 0;
    drop_below_ = INT64_MIN;
  }
  mount_ = mount_of(in_.T_rig_cam, params_);
  // The joins (p >= 1): t_b relative to t_begin and, for C0, the specific-force impulse (§4.3.4).
  for (int32_t p = 1; p < rig.num_pieces; ++p) {
    Join j;
    j.cls = in_.joins[static_cast<size_t>(p)];
    int64_t rel = 0;
    if (__builtin_sub_overflow(rig.pieces[p].begin_ns, in_.t_begin_ns, &rel)) {  // far from the run: never reached
      rel = rig.pieces[p].begin_ns > in_.t_begin_ns ? INT64_MAX : INT64_MIN;
    }
    j.t_b = static_cast<double>(rel);
    if (j.cls == kJoinC0) j.impulse = join_impulse(rig, mount_, rig.pieces[p].begin_ns);
    joins_.push_back(j);
  }
  const Stream stream{params_.seed, in_.camera_index};
  clock_ = SampleClock(stream, params_.rate_hz, params_.start_phase, params_.clock_adev);
  count_hi_ = (int64_t{1} << (params_.adc_bits - 1)) - 1;
  count_lo_ = -(int64_t{1} << (params_.adc_bits - 1));

  // §4.3.3: each filter starts at t_f = max(t_0 - 20 tau_c, span_begin + d) with y(t_f) = u(t_f), and integrates
  // [t_f, t_0] as its first interval. A rig that starts too late leaves samples before t_f + 20 tau_c with an
  // incomplete history (WARMUP); they are exact while the input is constant, by the deviation form. When even
  // t_f > t_0 (a rig span that begins at the run's start, as the engine tests' card bundle does), the filter rests at
  // u(t_f) before t_f, as for an input held before the span: samples at or before t_f give u(t_f).
  const double t0 = clock_.tau_ns();
  const double span_begin = static_cast<double>(rig.span_begin_ns - in_.t_begin_ns);
  const SensorErrors* sensors[2] = {&params_.gyro, &params_.accel};
  for (int s = 0; s < 2; ++s) {
    Channel& ch = ch_[s];
    const SensorErrors& e = *sensors[s];
    ch.noise = SensorNoise(stream, s, e, params_.noise_mode);
    ch.M = error_matrix(e);
    ch.b0 = ch.noise.b0();
    ch.affine = !identity(ch.M) || ch.b0.x != 0 || ch.b0.y != 0 || ch.b0.z != 0;
    if (e.lsb) {
      ch.quantized = true;
      ch.lsb = *e.lsb;
      if (params_.clip_stage == ClipStage::kBoth) ch.rail = std::ldexp(ch.lsb, params_.adc_bits - 1);
    }
    if (e.filter != Filter::kFirstOrder) continue;
    const double tau_s = tau_c_s(e);
    ch.filtered = true;
    ch.tau_ns = tau_s * 1e9;
    ch.d_ns = (e.group_delay_s - tau_s) * 1e9;  // >= 0: validate() refused group_delay_s < tau_c
    const double settled = t0 - 20 * ch.tau_ns, earliest = span_begin + ch.d_ns;
    const double t_f = std::max(settled, earliest);
    if (earliest > settled) ch.warm_until = t_f + 20 * ch.tau_ns;
    // max(): t_f - d can round below the span's begin when t_f = span_begin + d.
    uint8_t clipped = 0;
    ch.y = filter_input(s, truth_rel(std::max(t_f - ch.d_ns, span_begin)), &clipped);
    ch.prev = t_f;
    // The joins whose delayed time t_b + d is at or before t_f are history: the filter starts from u(t_f) after them.
    while (ch.next_join < joins_.size() && joins_[ch.next_join].t_b + ch.d_ns <= t_f) ++ch.next_join;
  }
  // §3.5: IMPACT marks t_k - t_b in [0, W), W the longest filter memory (20 tau_c + d) of the filtered sensors, or one
  // sample period when both are ideal point samples.
  for (const Channel& ch : ch_) {
    if (!ch.filtered) continue;
    join_warmup_ = true;
    join_window_ = std::max(join_window_, 20 * ch.tau_ns + ch.d_ns);
  }
  if (!join_warmup_) join_window_ = clock_.period_s() * 1e9;
}

Truth ImuSimulator::truth_rel(double t_rel_ns) const {
  const double n = std::floor(t_rel_ns);
  return truth_at(in_.rig, mount_, in_.g_W, where_right(in_.rig, in_.t_begin_ns + static_cast<int64_t>(n), t_rel_ns - n));
}

Vec3d ImuSimulator::filter_input(int s, const Truth& t, uint8_t* clipped) const {
  const Channel& ch = ch_[s];
  const Vec3d x = s == 0 ? t.omega_I : t.f_I;
  Vec3d u = ch.affine ? ch.M * x + ch.b0 : x;
  if (ch.rail > 0) {  // clip_stage both: the rails inside the integrand (§4.5)
    for (int i = 0; i < 3; ++i) {
      double& v = detail::at(u, i);
      if (v > ch.rail || v < -ch.rail) {
        v = v > 0 ? ch.rail : -ch.rail;
        *clipped |= static_cast<uint8_t>(1u << i);
      }
    }
  }
  return u;
}

void ImuSimulator::step_filter(int s, double tau, uint8_t* clipped) {
  Channel& ch = ch_[s];
  // The joins the filter's delayed time t_b + d passes in (prev, tau]: [first, last).
  const size_t first = ch.next_join;
  size_t last = first, reset = first;
  while (last < joins_.size() && joins_[last].t_b + ch.d_ns <= tau) {
    if (joins_[last].cls == kJoinStep) reset = last + 1;  // one past the last step join
    ++last;
  }
  ch.next_join = last;
  if (reset > first) {
    // A step (the pose teleports; §4.3.4): re-initialize at t_b + d as at start, y = u(t_b + d) from the right piece.
    // The latest step wins; what came before it, impulses included, left no trace. Noise and bias go on.
    const Join& jn = joins_[reset - 1];
    uint8_t unused = 0;
    ch.y = filter_input(s, truth_rel(jn.t_b), &unused);
    ch.prev = jn.t_b + ch.d_ns;
    for (size_t j = first; j < reset; ++j) {
      if (joins_[j].cls != kJoinStep || joins_[j].reset_counted) continue;
      joins_[j].reset_counted = true;
      ++stats_.step_resets;
    }
  }
  if (tau > ch.prev) {  // else tau <= t_f (or the reset is at tau): the filter rests at u
    knot_cuts(in_.rig, in_.t_begin_ns, ch.prev, tau, ch.d_ns, cuts_);
    panel_nodes(ch.prev, tau, ch.tau_ns, cuts_, panels_);
    u_.resize(panels_.s.size());
    for (size_t i = 0; i < u_.size(); ++i) {
      u_[i] = filter_input(s, truth_rel((tau + panels_.s[i]) - ch.d_ns), clipped);
    }
    ch.y = filter_step(ch.y, panels_, u_, ch.tau_ns, tau - ch.prev);
    ch.prev = tau;
  }
  if (s != 1) return;  // only the accelerometer's input has an impulse; the gyro's steps are cut above
  // C0 joins (§4.3.4): the specific force carries J_I delta(t - t_b), which the first-order filter turns into
  // (1/tau_c) e^{-(tau - t_b - d)/tau_c} M J_I at the first sample at or after t_b + d. It bypasses clip_pre.
  for (size_t j = std::max(first, reset); j < last; ++j) {
    const Join& jn = joins_[j];
    if (jn.cls != kJoinC0) continue;
    const Vec3d mj = ch.affine ? ch.M * jn.impulse : jn.impulse;
    const double gain = (1e9 / ch.tau_ns) * pmath::exp(-(tau - (jn.t_b + ch.d_ns)) / ch.tau_ns);
    ch.y += gain * mj;
  }
}

uint8_t ImuSimulator::join_flags(double tau) {
  uint8_t flags = 0;
  while (flag_from_ < joins_.size() && joins_[flag_from_].t_b + join_window_ <= tau) ++flag_from_;
  for (size_t j = flag_from_; j < joins_.size() && joins_[j].t_b <= tau; ++j) {
    flags |= kFlagImpact;  // t_k - t_b in [0, W) for a join of any class
    if (joins_[j].cls == kJoinStep && join_warmup_) flags |= kFlagWarmup;  // a filter was re-initialized
  }
  return flags;
}

ImuBlock ImuSimulator::advance_to(int64_t t_end_ns) {
  ImuBlock b;
  const int64_t bound = std::min(t_end_ns, in_.t_end_ns);
  for (;; clock_.next()) {
    const double tau = clock_.tau_ns();
    const uint64_t k = clock_.index();
    if (k > 0 && !(tau > prev_tau_)) {  // a clock_adev within its range never gets near this (|y| << 1)
      throw Refused("imu.clock_adev: the sample clock stepped back at sample " + std::to_string(k));
    }
    // tau >= 2^63 > span_ns_ is past the run's end, as is any tau that rounds to span_ns_ or later; both are tested
    // before an int64 can overflow (a valid rate_hz near 0 gives periods beyond INT64_MAX ns).
    if (!(tau < 0x1p63)) break;
    const int64_t rel = std::llround(tau);
    if (rel >= span_ns_) break;
    const int64_t t_true = in_.t_begin_ns + rel;
    if (t_true >= bound) break;
    // The processes take dt_k = t_k - t_{k-1} in seconds, and dt_0 = T (§4.4).
    const double dt_s = k == 0 ? clock_.period_s() : (tau - prev_tau_) / 1e9;
    prev_tau_ = tau;
    const Truth truth = truth_rel(tau);
    Vec3d out[2], bias[2];
    int32_t counts[2][3] = {};
    uint8_t flags = join_flags(tau), clipped[2] = {0, 0};
    for (int s = 0; s < 2; ++s) {
      Channel& ch = ch_[s];
      Vec3d y;
      if (!ch.filtered) {  // an ideal point sample: no delay; M, b0 and the pre-clip still apply (§4.3.3)
        y = filter_input(s, truth, &clipped[s]);
      } else {
        step_filter(s, tau, &clipped[s]);
        y = ch.y;
        if (tau < ch.warm_until) flags |= kFlagWarmup;
      }
      // §4.4: z = y + b_rw + b_gm + n; the processes step at every sample, emitted or dropped.
      const SensorNoise::Sample e = ch.noise.next(dt_s);
      Vec3d z = ch.noise.on() ? (y + e.drift) + e.noise : y;
      bias[s] = ch.b0 + e.drift;
      if (ch.quantized) {  // §4.5: counts clamped to the ADC range; a clamped axis sets its rail bit
        for (int i = 0; i < 3; ++i) {
          const double r = detail::at(z, i) / ch.lsb;
          // llround(r) is above the range exactly when r >= hi + 0.5 (both exact in fp64 for adc_bits <= 32).
          const bool above = !(r < static_cast<double>(count_hi_) + 0.5);
          const bool below = r <= static_cast<double>(count_lo_) - 0.5;
          const int64_t c = above ? count_hi_ : (below ? count_lo_ : std::llround(r));
          if (above || below) flags |= static_cast<uint8_t>(1u << (3 * s + i));
          counts[s][i] = static_cast<int32_t>(c);
          detail::at(z, i) = static_cast<double>(c) * ch.lsb;
        }
      }
      out[s] = z;
    }
    const int64_t stamp = sensor::reported_time(readout_, t_true);
    if (drop_all_ || stamp < drop_below_) {  // §4.6: below the event file's t_offset
      ++stats_.dropped_before_t_offset;
      continue;
    }
    if (flags & kFlagWarmup) ++stats_.warmup_samples;
    if (flags & kFlagImpact) ++stats_.impact_samples;
    for (int s = 0; s < 2; ++s) {
      for (int i = 0; i < 3; ++i) stats_.pre_clip_engaged[3 * s + i] += (clipped[s] >> i) & 1;
    }
    b.t_true_ns.push_back(t_true);
    b.t_ns.push_back(stamp);
    push3(b.gyro, out[0]);
    push3(b.accel, out[1]);
    if (ch_[0].quantized) b.counts_gyro.insert(b.counts_gyro.end(), counts[0], counts[0] + 3);
    if (ch_[1].quantized) b.counts_accel.insert(b.counts_accel.end(), counts[1], counts[1] + 3);
    b.flags.push_back(flags);
    push3(b.omega_true, truth.omega_I);
    push3(b.f_true, truth.f_I);
    push3(b.bias_gyro, bias[0]);
    push3(b.bias_accel, bias[1]);
    b.q_W_I.insert(b.q_W_I.end(), {truth.q_W_I.w, truth.q_W_I.x, truth.q_W_I.y, truth.q_W_I.z});
    push3(b.p_W_I, truth.p_W_I);
  }
  return b;
}

}  // namespace balla::imu
