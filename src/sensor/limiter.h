// The driver's event-rate limiter (DESIGN D13-5, F043; docs/dev/EVIMO2_LIMITER.md §5): a token bucket on one camera's
// time-sorted output stream, on the sensor clock, after the pixel loop (a dropped event has already reset its pixel,
// DESIGN §8.7). Host only, header only; sensor.limiter and its rows (params.h) configure it.
//
// Model. The cap R(phi) = R0 - (R0 - R1) phi / P falls linearly over each epoch of P and steps back to R0 at every
// reset t = phase + k P of the sensor clock. The bucket holds at most B events and fills at R; a bin u with n_u events
// lets a_u = min(n_u, floor(tokens)) of them through and spends a_u tokens. The a_u kept are those with the smallest
// selection keys Philox(seed, 0x0021, camera, pixel, t_cross) (ties by x, y, p, t_cross, then input order): blind to
// polarity and position, as the EVIMO2 drops are (EVIMO2_LIMITER §3 finding 5). The first bin finds the bucket full.
//
// Bins. With a readout quantum q > 0 a bin is one stamp (one t_reported value on the counter grid, M3 F.1), refilled
// to its end t + q; with q = 0 the 1 us bins [1000 j, 1000 (j + 1)) ns of the sensor clock.
//
// Arithmetic: integer and exact. Tokens are int64 Q32 events (B <= 1e9 keeps B 2^32 below 2^62); the rates are Q48
// events per ns, rounded once (by at most 2^-49 events per ns, 1.8e-6 Hz: a relative 1.8e-6 at the rows' 1 Hz floor).
// The refill over (a, b] is F(b) - F(a) of the floored cumulative integral F(t) = k E + G(phi) in Q32 events, with
// t - phase = k P + phi, G(phi) = floor(phi (2 P r0 - (r0 - r1) phi) / (2^17 P)) and E = G(P), in 128 bits (at the
// rows' hard bounds, P <= 100 s and R <= 1e10 Hz, phi 2 P r0 <= 1e11 x 2e11 x 10 x 2^48 = 5.6e37 < 2^127). F
// telescopes, so a refill does not depend on where empty bins or call boundaries fall, and the bucket's min(B, .)
// taken at the non-empty bins equals it taken at every bin (nothing is spent in between). Every decision therefore
// depends only on the bins in time order and on integer state: the output does not depend on how the stream is cut
// into calls, provided each call says up to where the stream is complete (the bin that later events may still join is
// held for the next call).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "core/rng/purpose.h"
#include "core/time/time.h"
#include "sensor/params.h"

namespace balla::sensor {

__extension__ typedef __int128 LimiterI128;

// The limiter of one camera in integer units (limiter_config).
struct LimiterConfig {
  bool on = false;          // sensor.limiter epoch; off, apply() leaves every stream as it is
  int64_t period_ns = 0;    // the epoch P (>= 1 us)
  int64_t phase_ns = 0;     // a reset time mod P, in [0, P)
  int64_t rate0_q48 = 0;    // R0, Q48 events per ns of the sensor clock
  int64_t rate1_q48 = 0;    // R1 <= R0
  int64_t depth_q32 = 0;    // B 2^32
  int64_t quantum_ns = 0;   // the readout quantum (0: 1 us bins)
  uint64_t seed = 0;        // the run's sensor seed (the Philox key)
  uint32_t camera = 0;      // the camera's index (the Philox counter, as the sensor's draws)
  uint32_t width = 0;       // pixel = y width + x
};

// The limiter's state between calls: the bucket and the counters (the held events live in Limiter).
struct LimiterState {
  int64_t tokens_q32 = 0;    // after the last decided bin
  int64_t last_end_ns = 0;   // that bin's end: the next refill's lower bound (valid once started)
  bool started = false;      // a bin was decided (the first finds the bucket full)
  uint64_t kept = 0, dropped = 0;
  uint64_t capped_bins = 0;  // bins that dropped events (microseconds at the Prophesee quantum of 1 us)
};

// The configuration of a camera's limiter: params (validated here) with the run's sensor seed, the camera's index and
// its width. A null limiter_phase_s draws the phase uniformly in [0, P) from tag 0x0023 (pixel 0, index 0, as the
// readout phase's 0x0022).
inline LimiterConfig limiter_config(const SensorParams& p, uint64_t seed, uint32_t camera, uint32_t width) {
  p.validate();
  LimiterConfig c;
  c.on = p.limiter_on();
  c.seed = seed;
  c.camera = camera;
  c.width = width;
  if (!c.on) return c;
  c.period_ns = std::llround(p.limiter_period_s * 1e9);
  c.rate0_q48 = std::llround(p.limiter_rate0_hz * 0x1p48 / 1e9);
  c.rate1_q48 = std::llround(p.limiter_rate1_hz * 0x1p48 / 1e9);
  c.depth_q32 = p.limiter_depth * (int64_t{1} << 32);
  c.quantum_ns = std::llround(p.readout_quantum_s * 1e9);
  if (p.limiter_phase_s) {
    c.phase_ns = time::floor_mod(std::llround(*p.limiter_phase_s * 1e9), c.period_ns);
  } else {
    const rng::U32x4 d = rng::draw(seed, rng::Purpose::kLimiterPhase, camera, 0, 0);
    c.phase_ns = static_cast<int64_t>(((static_cast<uint64_t>(d.v[1]) << 32) | d.v[0]) %
                                      static_cast<uint64_t>(c.period_ns));
  }
  return c;
}

// The cap's position at t: t - phase = k P + phi, phi in [0, P), and G(phi) = floor of the integral of the cap over
// [phase + k P, t] in Q32 events (phi in [0, P]; G(P) is the whole epoch's E). The floored cumulative integral is
// F(t) = k E + G(phi).
struct LimiterClock {
  int64_t k;
  LimiterI128 g;
};
inline LimiterI128 limiter_epoch_part_q32(const LimiterConfig& c, int64_t phi) {
  const LimiterI128 big_p = c.period_ns, f = phi;
  return f * (2 * big_p * c.rate0_q48 - static_cast<LimiterI128>(c.rate0_q48 - c.rate1_q48) * f) / (big_p << 17);
}
inline LimiterClock limiter_clock(const LimiterConfig& c, int64_t t) {
  const int64_t k = time::floor_div(t - c.phase_ns, c.period_ns);
  return {k, limiter_epoch_part_q32(c, t - c.phase_ns - k * c.period_ns)};
}
// F(b) - F(a) from the positions of a <= b: the bucket's refill over (a, b] in Q32 events.
inline LimiterI128 limiter_refill_q32(const LimiterConfig& c, const LimiterClock& a, const LimiterClock& b) {
  return static_cast<LimiterI128>(b.k - a.k) * limiter_epoch_part_q32(c, c.period_ns) + b.g - a.g;
}
inline LimiterI128 limiter_refill_q32(const LimiterConfig& c, int64_t a, int64_t b) {
  return limiter_refill_q32(c, limiter_clock(c, a), limiter_clock(c, b));
}

// The end of the bin of an event reported at t: t + q (one stamp), or the next whole microsecond (q = 0).
inline int64_t limiter_bin_end(const LimiterConfig& c, int64_t t) {
  return c.quantum_ns > 0 ? t + c.quantum_ns : (time::floor_div(t, int64_t{1000}) + 1) * 1000;
}

// No event reported at or after complete_ns can join the bin of an event reported at t.
inline bool limiter_bin_closed(const LimiterConfig& c, int64_t t, int64_t complete_ns) {
  return c.quantum_ns > 0 ? t < complete_ns : limiter_bin_end(c, t) <= complete_ns;
}

// An event's selection key: keyed by the pixel and the true crossing time, so it does not depend on the readout, the
// clock or the stream around it.
inline uint64_t limiter_key(const LimiterConfig& c, uint32_t pixel, int64_t t_cross_ns) {
  const rng::U32x4 d =
      rng::draw(c.seed, rng::Purpose::kLimiterDrop, c.camera, pixel, static_cast<uint64_t>(t_cross_ns));
  return (static_cast<uint64_t>(d.v[0]) << 32) | d.v[1];
}

// One camera's limiter over a stream of events Ev with members t_reported and t_cross (int64 ns), x, y and p
// (engine::OutEvent), given in calls in time order.
template <class Ev>
class Limiter {
 public:
  Limiter() = default;
  explicit Limiter(const LimiterConfig& c) : c_(c), epoch_(c.on ? limiter_epoch_part_q32(c, c.period_ns) : 0) {}

  const LimiterConfig& config() const { return c_; }
  const LimiterState& state() const { return s_; }
  const std::vector<Ev>& held() const { return held_; }

  // events: the call's events, sorted by t_reported (ties in any fixed order, the engine's x, y), none before an event
  // of an earlier call; every event of the stream reported before complete_ns is in this call or an earlier one.
  // Replaces events by the kept events of every bin decided now, the held events of the previous call first, in their
  // order, and holds the events from the first bin that later events may still join. Off: leaves events unchanged.
  // std::invalid_argument for events out of time order or in a bin already decided.
  void apply(std::vector<Ev>& events, int64_t complete_ns) {
    if (!c_.on) return;
    std::vector<Ev> in;
    if (held_.empty()) {
      in.swap(events);
    } else {
      in.swap(held_);
      in.insert(in.end(), events.begin(), events.end());
    }
    events.clear();
    for (size_t k = 1; k < in.size(); ++k) {
      if (in[k].t_reported < in[k - 1].t_reported) {
        throw std::invalid_argument("limiter: events out of time order (t_reported " +
                                    std::to_string(in[k].t_reported) + " after " +
                                    std::to_string(in[k - 1].t_reported) + ")");
      }
    }
    // The kept events are compacted in place (w <= i): a subsequence, in order.
    size_t i = 0, w = 0;
    while (i < in.size() && limiter_bin_closed(c_, in[i].t_reported, complete_ns)) {
      const int64_t end = limiter_bin_end(c_, in[i].t_reported);
      if (s_.started && end <= s_.last_end_ns) {
        throw std::invalid_argument("limiter: an event reported at " + std::to_string(in[i].t_reported) +
                                    " ns falls in a bin already decided (the stream was declared complete before it)");
      }
      size_t j = i + 1;
      while (j < in.size() && limiter_bin_end(c_, in[j].t_reported) == end) ++j;
      w = decide(in, i, j, end, w);
      i = j;
    }
    held_.assign(in.begin() + static_cast<std::ptrdiff_t>(i), in.end());
    in.resize(w);
    events.swap(in);
  }

  // The stream's end: apply() with nothing left to come (every held bin is decided).
  void flush(std::vector<Ev>& events) { apply(events, std::numeric_limits<int64_t>::max()); }

 private:
  using Rank = std::tuple<uint64_t, uint32_t, uint32_t, uint32_t, int64_t, size_t>;  // key, x, y, p, t_cross, index

  // Decides the bin in[i, j) ending at `end` and moves its kept events to in[w, ...); returns the next w.
  size_t decide(std::vector<Ev>& in, size_t i, size_t j, int64_t end, size_t w) {
    // The refill since the last bin's end, F(end) - F(last end), with that end's position kept: one 128-bit division
    // per bin (limiter_refill_q32's value).
    const LimiterI128 depth = c_.depth_q32;
    const LimiterClock now = limiter_clock(c_, end);
    LimiterI128 tokens = depth;
    if (s_.started) {
      tokens = std::min(depth, s_.tokens_q32 + static_cast<LimiterI128>(now.k - last_.k) * epoch_ + now.g - last_.g);
    }
    s_.started = true;
    s_.last_end_ns = end;
    last_ = now;
    const auto n = static_cast<uint64_t>(j - i);
    const uint64_t a = std::min(n, static_cast<uint64_t>(tokens >> 32));
    s_.tokens_q32 = static_cast<int64_t>(tokens - (static_cast<LimiterI128>(a) << 32));
    s_.kept += a;
    if (a == n) {
      if (w != i) std::copy(in.begin() + static_cast<std::ptrdiff_t>(i), in.begin() + static_cast<std::ptrdiff_t>(j),
                            in.begin() + static_cast<std::ptrdiff_t>(w));
      return w + (j - i);
    }
    s_.dropped += n - a;
    ++s_.capped_bins;
    if (a == 0) return w;
    rank_.clear();
    for (size_t k = i; k < j; ++k) {
      const Ev& e = in[k];
      const auto pixel = static_cast<uint32_t>(e.y) * c_.width + static_cast<uint32_t>(e.x);
      rank_.emplace_back(limiter_key(c_, pixel, e.t_cross), static_cast<uint32_t>(e.x), static_cast<uint32_t>(e.y),
                         static_cast<uint32_t>(e.p), static_cast<int64_t>(e.t_cross), k);
    }
    std::nth_element(rank_.begin(), rank_.begin() + static_cast<std::ptrdiff_t>(a), rank_.end());
    keep_.assign(j - i, 0);
    for (size_t r = 0; r < a; ++r) keep_[std::get<5>(rank_[r]) - i] = 1;
    for (size_t k = i; k < j; ++k) {
      if (keep_[k - i]) in[w++] = in[k];
    }
    return w;
  }

  LimiterConfig c_;
  LimiterI128 epoch_ = 0;     // E, the refill of a whole epoch
  LimiterState s_;
  LimiterClock last_{0, 0};   // the position of s_.last_end_ns (limiter_clock), once started
  std::vector<Ev> held_;
  std::vector<Rank> rank_;
  std::vector<uint8_t> keep_;
};

}  // namespace balla::sensor
