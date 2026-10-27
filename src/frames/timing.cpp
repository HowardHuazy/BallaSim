// Frame schedules and clocks (docs/design/phase3/frames.md §4.1).
#include "frames/timing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace balla::frames {

namespace {

__extension__ typedef __int128 i128;

constexpr i128 kInt64Max = std::numeric_limits<int64_t>::max();

// floor(a / b) and ceil(a / b) for b > 0.
i128 floor_div(i128 a, i128 b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }
i128 ceil_div(i128 a, i128 b) { return -floor_div(-a, b); }

// R + T, the length of a frame's [frame_soe, frame_eoe]; check_timing keeps it inside int64.
i128 frame_length(const RowTiming& t) {
  const i128 r = t.shutter == Shutter::kRolling ? static_cast<i128>(t.height - 1) * t.row_dt_ns : 0;
  return r + t.exposure_ns;
}

void check_timing(const RowTiming& t) {
  if (t.height < 1) throw std::invalid_argument("frames schedule: height must be >= 1");
  if (t.exposure_ns < 0) throw std::invalid_argument("frames schedule: exposure_ns must be >= 0");
  if (t.row_dt_ns < 0) throw std::invalid_argument("frames schedule: row_dt_ns must be >= 0");
  if (frame_length(t) > kInt64Max) {
    throw std::invalid_argument("frames schedule: the readout span plus the exposure exceed 2^63 ns");
  }
}

void check_support(Span support) {
  if (support.end_ns < support.begin_ns) throw std::invalid_argument("frames schedule: support end < begin");
}

// [a, a + R + T] inside the support, exact for any int64 anchor.
bool inside(i128 anchor, i128 length, Span support) {
  return anchor >= support.begin_ns && anchor + length <= support.end_ns;
}

std::string span_text(Span support) {
  return "[" + std::to_string(support.begin_ns) + ", " + std::to_string(support.end_ns) + "] ns";
}

[[noreturn]] void refuse_empty(int64_t n_candidates, Span support) {
  throw FramesError("FRM-SPAN", "frames: none of the " + std::to_string(n_candidates) +
                                    " frames lies inside the bundle span " + span_text(support));
}

std::string ns(int64_t t) { return std::to_string(t) + " ns"; }

// FRM-EXPOSURE (§4.1): T > P, or R + T > P, for the frame spacing P.
void check_exposure(const Schedule& s, int64_t period_ns, std::vector<Warning>* out) {
  const int64_t t = s.timing().exposure_ns, r = s.readout_span_ns();
  if (t > period_ns) {
    out->push_back({"FRM-EXPOSURE", "frames: exposure " + ns(t) + " exceeds the frame period " + ns(period_ns)});
  } else if (r + t > period_ns) {
    out->push_back({"FRM-EXPOSURE", "frames: readout span " + ns(r) + " plus exposure " + ns(t) +
                                        " exceed the frame period " + ns(period_ns)});
  }
}

}  // namespace

const char* exposure_source_name(ExposureSource s) {
  switch (s) {
    case ExposureSource::kSet: return "set";
    case ExposureSource::kBlenderShutter: return "blender_shutter";
    default: return "default";
  }
}

Exposure resolve_exposure(std::optional<int64_t> set_ns, const BlenderShutter* blender, const FrameRate* rate) {
  if (set_ns) {
    if (*set_ns < 0) throw std::invalid_argument("frames.exposure_ns must be >= 0");
    return {*set_ns, ExposureSource::kSet};
  }
  if (blender == nullptr) return {kDefaultExposureNs, ExposureSource::kDefault};
  if (!blender->use) return {0, ExposureSource::kBlenderShutter};
  if (rate == nullptr || rate->fps < 1 || rate->fps > 32767 || rate->base_num < 1 || rate->base_den < 1) {
    throw std::invalid_argument("frames: the Blender frame rate needs fps in [1, 32767] and base_num, base_den >= 1");
  }
  const float sigma = blender->shutter;
  if (!(sigma >= 0.0f && sigma <= 0x1p20f)) {
    throw std::invalid_argument("frames: Blender's motion_blur_shutter must lie in [0, 2^20] frames");
  }
  // sigma = mant 2^e exactly (24 significant bits), so sigma P = mant 1e9 base_num / (fps base_den 2^-e): one
  // round half to even of an exact rational. num = mant 1e9 base_num < 2^24 2^30 2^63 = 2^117 and
  // fps base_den < 2^78 for any int64 base (a float32 fps_base down to Blender's 1e-5 has base_den up to 2^40), and
  // e <= -3 for sigma <= 2^20. A denominator >= 2^125 puts sigma P below 2^-8: it rounds to 0.
  int e = 0;
  const double frac = std::frexp(static_cast<double>(sigma), &e);
  const i128 mant = static_cast<i128>(std::ldexp(frac, 24));
  e -= 24;
  const i128 num = mant * 1'000'000'000 * rate->base_num;
  const i128 den0 = static_cast<i128>(rate->fps) * rate->base_den;
  const int shift = -e;
  int den_bits = 0;
  for (i128 x = den0; x != 0; x >>= 1) ++den_bits;
  if (num == 0 || den_bits + shift >= 126) {
    return {0, ExposureSource::kBlenderShutter};  // sigma P < 2^117 / 2^125: rounds to 0
  }
  const i128 den = den0 << shift;
  i128 q = num / den;
  const i128 r2 = 2 * (num % den);
  if (r2 > den || (r2 == den && (q & 1) != 0)) ++q;
  if (q > std::numeric_limits<int64_t>::max() / 4) throw std::invalid_argument("frames: Blender exposure overflows");
  return {static_cast<int64_t>(q), ExposureSource::kBlenderShutter};
}

void Schedule::finish(int64_t n_candidates, Span support) {
  if (anchor_.empty()) refuse_empty(n_candidates, support);
  if (!dropped_.empty()) {
    warnings_.push_back({"FRM-SPAN", "frames: " + std::to_string(dropped_.size()) +
                                         " frames outside the bundle span " + span_text(support) + " were dropped"});
  }
}

Schedule Schedule::uniform(const RowTiming& timing, int64_t t0_ns, int64_t period_ns, std::optional<int32_t> count,
                           Span support) {
  check_timing(timing);
  check_support(support);
  if (period_ns <= 0) throw std::invalid_argument("frames schedule: period_ns must be > 0");
  if (count && *count < 1) throw std::invalid_argument("frames schedule: count must be >= 1");
  // The anchors t0 + k P increase with k, so the kept candidates are one run k_lo <= k <= k_hi: k_lo the first with
  // t0 + k P >= begin, k_fit the last with t0 + k P + R + T <= end. Exact in __int128 (|t0 + k P| < 2^95), and no
  // anchor outside the run is formed.
  const i128 t0 = t0_ns, p = period_ns, length = frame_length(timing);
  const i128 k_lo = std::max<i128>(0, ceil_div(support.begin_ns - t0, p));
  const i128 k_fit = floor_div(support.end_ns - length - t0, p);
  i128 n = 0;  // candidates k = 0 .. n - 1
  if (count) {
    n = *count;
  } else {
    n = k_fit < 0 ? 0 : k_fit + 1;  // every k >= 0 with t0 + k P + R + T <= end
    if (n > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument("frames schedule: more than 2^31 frames; set frames.count or a longer period");
    }
  }
  const i128 k_hi = std::min(k_fit, n - 1);
  if (k_hi < k_lo) refuse_empty(static_cast<int64_t>(n), support);
  const i128 n_dropped = n - (k_hi - k_lo + 1);
  if (n_dropped > kMaxDroppedFrames) {
    throw FramesError("FRM-SPAN", "frames: " + std::to_string(static_cast<int64_t>(n_dropped)) + " of the " +
                                      std::to_string(static_cast<int64_t>(n)) + " frames lie outside the bundle span " +
                                      span_text(support) + ", more than the " + std::to_string(kMaxDroppedFrames) +
                                      " a schedule may drop; set frames.t0_ns or frames.count to fit the span");
  }
  Schedule s(timing);
  s.anchor_.reserve(static_cast<size_t>(k_hi - k_lo + 1));
  s.candidate_.reserve(s.anchor_.capacity());
  s.dropped_.reserve(static_cast<size_t>(n_dropped));
  for (i128 k = 0; k < k_lo; ++k) s.dropped_.push_back(static_cast<int32_t>(k));
  for (i128 k = k_lo; k <= k_hi; ++k) {
    s.anchor_.push_back(static_cast<int64_t>(t0 + k * p));
    s.candidate_.push_back(static_cast<int32_t>(k));
  }
  for (i128 k = k_hi + 1; k < n; ++k) s.dropped_.push_back(static_cast<int32_t>(k));
  s.finish(static_cast<int64_t>(n), support);
  check_exposure(s, period_ns, &s.warnings_);
  return s;
}

Schedule Schedule::blender(const RowTiming& timing, std::span<const int64_t> frame_ns, Position position,
                           Span support) {
  check_timing(timing);
  check_support(support);
  if (frame_ns.empty()) throw std::invalid_argument("frames schedule: the blender timeline has no frame");
  if (frame_ns.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("frames schedule: more than 2^31 blender frames");
  }
  const i128 length = frame_length(timing);
  const i128 back = position == Position::kCenter ? length / 2  // length >= 0: floor
                    : position == Position::kEnd  ? length
                                                  : 0;
  Schedule s(timing);
  i128 spacing = kInt64Max;
  for (size_t k = 0; k < frame_ns.size(); ++k) {
    if (k > 0) {
      if (frame_ns[k] <= frame_ns[k - 1]) {
        throw std::invalid_argument("frames schedule: blender frame times must increase strictly");
      }
      spacing = std::min(spacing, static_cast<i128>(frame_ns[k]) - frame_ns[k - 1]);
    }
    const i128 a = frame_ns[k] - back;  // exact: an anchor below int64 lies before the support and is dropped
    if (inside(a, length, support)) {
      s.anchor_.push_back(static_cast<int64_t>(a));
      s.candidate_.push_back(static_cast<int32_t>(k));
    } else {
      s.dropped_.push_back(static_cast<int32_t>(k));
    }
  }
  s.finish(static_cast<int64_t>(frame_ns.size()), support);
  if (frame_ns.size() > 1) check_exposure(s, static_cast<int64_t>(spacing), &s.warnings_);
  return s;
}

// floor((soe + eoe) / 2) = soe + floor((R + T) / 2), as soe + eoe = 2 soe + (R + T) with R + T >= 0. The result lies
// in [soe, eoe], so it is exact for every frame whose eoe is; the sum soe + eoe would overflow for anchors near 2^63.
int64_t Schedule::frame_mid(int32_t k) const { return frame_soe(k) + (readout_span_ns() + timing_.exposure_ns) / 2; }

Schedule Schedule::shifted(std::span<const int64_t> shift_ns) const {
  if (shift_ns.size() != anchor_.size()) throw std::invalid_argument("Schedule::shifted: one shift per frame");
  Schedule s = *this;
  for (size_t k = 0; k < anchor_.size(); ++k) s.anchor_[k] += shift_ns[k];
  return s;
}

int64_t sensor_time(const sensor::Readout& clock, int64_t t_true) {
  sensor::Readout c = clock;
  c.delay_ns = 0;
  c.quantum_ns = 0;
  return sensor::reported_time(c, t_true);
}

}  // namespace balla::frames
