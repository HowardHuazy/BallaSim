// Light-group waveforms m_g(t) applied inside the pixel model (CONTRACTS C1/C3, DESIGN §7.5, §7.8).
//
// The photocurrent is I(t) = sum_g m_g(t) I_g(t) + I_dk (M3 A.1). Waveforms are mean-normalized:
//   constant: m = 1;
//   fourier:  m = 1 + sum_k a_k cos(2 pi k f0 t + phi_k);
//   pulse:    a trapezoid pulse train of frequency f, rise = fall time, duty D = (top + rise) f,
//             floor b and peak A = b + (1 - b) / D, so <m> = 1 (the convention of M3 V7 X3).
// Phases are evaluated in fp64 from int64 ns, which keeps the phase error below 1e-9 cycles for
// f t <= 1e7 (DESIGN §7.5). The pixel integrator approximates I(t) piecewise-linearly with the
// chord bound of M3 B.18-B.19, using the guaranteed sup bounds of |m'| and |m''| over each candidate
// piece computed here. Pulse corners are snapped to the integer-ns grid, so the simulated pulse is
// exactly piecewise linear on the ns grid (each corner moves by < 0.5 ns) and its PL
// representation carries no approximation error.
//
// waveform_mean and WaveformPrefix (end of file) give exact interval means of that same waveform, for
// frames (docs/design/phase3/frames.md §4.5).
#pragma once

#include <stdint.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/pmath/trig.h"

namespace balla::sensor {

inline constexpr int kMaxHarmonics = 8;
inline constexpr int kMaxLightGroups = 8;
inline constexpr int64_t kNoBreakpoint = INT64_MAX;

enum class WaveformType : int32_t { kConstant = 0, kFourier = 1, kPulse = 2 };

// Plain data, copied to the device as is.
struct Waveform {
  WaveformType type = WaveformType::kConstant;
  int32_t n_harmonics = 0;
  double f0_hz = 0.0;
  double a[kMaxHarmonics] = {};
  double phi[kMaxHarmonics] = {};
  // Pulse train, times in ns.
  double period_ns = 0.0;
  double phase_ns = 0.0;
  double rise_ns = 0.0;
  double top_ns = 0.0;
  double floor_level = 0.0;
  double peak_level = 0.0;
};

template <class T>
struct WaveBounds {
  T m_min;    // lower bound of m on the piece
  T dm_max;   // upper bound of |m'| (1/s)
  T d2m_max;  // upper bound of |m''| (1/s^2)
};

namespace detail {

inline constexpr double kTwoPi = 0x1.921fb54442d18p+2;

// Phase of harmonic k at t in [0, 2 pi): 2 pi frac(k f0 t) + phi_k, reduced in fp64.
BALLA_HD inline double harmonic_phase(const Waveform& w, int k, int64_t t_ns) {
  double cycles = w.f0_hz * 1e-9 * static_cast<double>(t_ns);
  cycles = cycles - floor(cycles);
  double x = static_cast<double>(k + 1) * cycles;
  x = x - floor(x);
  double ph = kTwoPi * x + w.phi[k];
  ph = ph - kTwoPi * floor(ph / kTwoPi);
  return ph;
}

// Corner j (0..4) of pulse period n, snapped to integer ns: rise start, rise end, fall start,
// fall end, next rise start.
BALLA_HD inline int64_t pulse_corner(const Waveform& w, int64_t n, int j) {
  const double start = w.phase_ns + static_cast<double>(n) * w.period_ns;
  double off = 0.0;
  switch (j) {
    case 0: off = 0.0; break;
    case 1: off = w.rise_ns; break;
    case 2: off = w.rise_ns + w.top_ns; break;
    case 3: off = 2.0 * w.rise_ns + w.top_ns; break;
    default: off = w.period_ns; break;
  }
  return static_cast<int64_t>(floor(start + off + 0.5));
}

// Index of the pulse period whose snapped corners [corner 0, corner 4) contain t.
BALLA_HD inline int64_t pulse_period(const Waveform& w, int64_t t_ns) {
  int64_t n = static_cast<int64_t>(floor((static_cast<double>(t_ns) - w.phase_ns) / w.period_ns));
  if (t_ns < pulse_corner(w, n, 0)) --n;
  if (t_ns >= pulse_corner(w, n, 4)) ++n;
  return n;
}

// Is some angle x0 + 2 pi n inside [lo, hi] (hi - lo < 2 pi, lo in [0, 2 pi))?
BALLA_HD inline bool contains_angle(double lo, double hi, double x0) {
  return (lo <= x0 && x0 <= hi) || (lo <= x0 + kTwoPi && x0 + kTwoPi <= hi);
}

}  // namespace detail

// m(t) at an integer nanosecond.
template <class T>
BALLA_HD T waveform_value(const Waveform& w, int64_t t_ns) {
  switch (w.type) {
    case WaveformType::kFourier: {
      T m = T(1);
      for (int k = 0; k < w.n_harmonics; ++k) {
        const T c = pmath::cos(static_cast<T>(detail::harmonic_phase(w, k, t_ns)));
        m = m + static_cast<T>(w.a[k]) * c;
      }
      return m;
    }
    case WaveformType::kPulse: {
      const int64_t n = detail::pulse_period(w, t_ns);
      int64_t c[5];
      for (int j = 0; j < 5; ++j) c[j] = detail::pulse_corner(w, n, j);
      const T lo = static_cast<T>(w.floor_level), hi = static_cast<T>(w.peak_level);
      if (t_ns >= c[3]) return lo;
      if (t_ns >= c[2]) return hi + (lo - hi) * static_cast<T>(t_ns - c[2]) / static_cast<T>(c[3] - c[2]);
      if (t_ns >= c[1]) return hi;
      return lo + (hi - lo) * static_cast<T>(t_ns - c[0]) / static_cast<T>(c[1] - c[0]);
    }
    default:
      return T(1);
  }
}

// The first breakpoint of m strictly after t (kNoBreakpoint for smooth waveforms).
BALLA_HD inline int64_t waveform_next_breakpoint(const Waveform& w, int64_t t_ns) {
  if (w.type != WaveformType::kPulse) return kNoBreakpoint;
  const int64_t n = detail::pulse_period(w, t_ns);
  for (int j = 1; j < 5; ++j) {
    const int64_t c = detail::pulse_corner(w, n, j);
    if (c > t_ns) return c;
  }
  return detail::pulse_corner(w, n + 1, 1);
}

// Guaranteed bounds of m, |m'| and |m''| on [t_a, t_b]. For the pulse the piece must not contain a
// breakpoint in its interior (the piece generator splits at them), so m is linear on it.
template <class T>
BALLA_HD WaveBounds<T> waveform_bounds(const Waveform& w, int64_t t_a, int64_t t_b) {
  switch (w.type) {
    case WaveformType::kFourier: {
      T m_min = T(1), dm = T(0), d2m = T(0);
      const double span_s = 1e-9 * static_cast<double>(t_b - t_a);
      for (int k = 0; k < w.n_harmonics; ++k) {
        const double omega = detail::kTwoPi * static_cast<double>(k + 1) * w.f0_hz;
        const double lo = detail::harmonic_phase(w, k, t_a);
        const double hi = lo + omega * span_s;
        T c_min = T(-1), c_max = T(1), s_abs = T(1), c_abs = T(1);
        if (hi - lo < detail::kTwoPi) {
          T s0, c0, s1, c1;
          pmath::sincos(static_cast<T>(lo), &s0, &c0);
          pmath::sincos(static_cast<T>(hi - detail::kTwoPi * floor(hi / detail::kTwoPi)), &s1, &c1);
          constexpr double kPi = detail::kTwoPi / 2;
          c_max = detail::contains_angle(lo, hi, 0.0) ? T(1) : max(c0, c1);
          c_min = detail::contains_angle(lo, hi, kPi) ? T(-1) : min(c0, c1);
          s_abs = detail::contains_angle(lo, hi, 0.5 * kPi) || detail::contains_angle(lo, hi, 1.5 * kPi)
                      ? T(1)
                      : max(abs(s0), abs(s1));
          c_abs = detail::contains_angle(lo, hi, 0.0) || detail::contains_angle(lo, hi, kPi)
                      ? T(1)
                      : max(abs(c0), abs(c1));
        }
        const T a = static_cast<T>(w.a[k]);
        m_min = m_min + (a >= T(0) ? a * c_min : a * c_max);
        dm = dm + abs(a) * static_cast<T>(omega) * s_abs;
        d2m = d2m + abs(a) * static_cast<T>(omega * omega) * c_abs;
      }
      return {m_min, dm, d2m};
    }
    case WaveformType::kPulse: {
      const T m0 = waveform_value<T>(w, t_a), m1 = waveform_value<T>(w, t_b);
      const T slope = abs(m1 - m0) / (static_cast<T>(t_b - t_a) * T(1e-9));
      return {min(m0, m1), slope, T(0)};
    }
    default:
      return {T(1), T(0), T(0)};
  }
}

// ---- Exact interval means (frames.md §4.5) ------------------------------------------------------------------------

namespace detail {

// sin(pi h) / (pi h) for h >= 0 cycles. The argument is reduced in cycles (h - rint(h) is exact by Sterbenz), so the
// value stays accurate for windows of many periods; below pi h = 1e-4 the series 1 - x^2/6 (error < 1e-18).
BALLA_HD inline double sinc_cycles(double h) {
  constexpr double kPi = kTwoPi / 2;
  const double x = kPi * h;
  if (x < 1e-4) return 1.0 - x * x / 6.0;
  const double n = rint(h);
  double s = pmath::sin(kPi * (h - n));
  if (n - 2.0 * floor(0.5 * n) != 0.0) s = -s;  // sin(pi h) = (-1)^n sin(pi (h - n))
  return s / x;
}

// The pulse as floor + (peak - floor) u(t), with u the unit trapezoid of waveform_value: (t - c0)/(c1 - c0) on the
// rise, 1 on the top, (c3 - t)/(c3 - c2) on the fall, 0 on the floor. This adds the integral of u over
// [x, y] n [c0, c3] of one period (corners c0..c3) in two parts: twice the integral of the pieces [x, y] covers
// whole, an exact integer (rise and fall c1 - c0 and c3 - c2, the top 2 (c2 - c1)), into *u2, and the integral of
// the ramps [x, y] cuts into *part. The floor (u = 0) adds nothing. Snapping can end a period's fall 1 ns past the
// next period's c0 (c3(n) = c0(n + 1) + 1), so a window's integral sums every period whose [c0, c3] meets it, both
// pieces counting on that overlap.
BALLA_HD inline void pulse_unit_integral(const int64_t c[4], int64_t x, int64_t y, int64_t* u2, double* part) {
  for (int p = 0; p < 3; ++p) {
    const int64_t lo = max(x, c[p]), hi = min(y, c[p + 1]);
    if (hi <= lo) continue;
    if (p == 1) {
      *u2 += 2 * (hi - lo);
    } else if (lo == c[p] && hi == c[p + 1]) {
      *u2 += hi - lo;
    } else {
      // Over [lo, hi], (t - c0)/R integrates to (hi - lo)(lo + hi - 2 c0)/(2R) and (c3 - t)/F to
      // (hi - lo)(2 c3 - lo - hi)/(2F).
      const int64_t s = p == 0 ? lo + hi - 2 * c[0] : 2 * c[3] - lo - hi;
      *part += static_cast<double>(hi - lo) * static_cast<double>(s) / (2.0 * static_cast<double>(c[p + 1] - c[p]));
    }
  }
}

BALLA_HD inline void pulse_corners4(const Waveform& w, int64_t n, int64_t c[4]) {
  for (int j = 0; j < 4; ++j) c[j] = pulse_corner(w, n, j);
}

BALLA_HD inline double pulse_mean(const Waveform& w, int64_t u2, double part, int64_t length_ns) {
  const double u = 0.5 * static_cast<double>(u2) + part;
  return w.floor_level + (w.peak_level - w.floor_level) * (u / static_cast<double>(length_ns));
}

}  // namespace detail

// (1/(t_b - t_a)) * integral of m over [t_a, t_b], in fp64 from int64 ns (frames.md §4.5); m(t_a) when t_b <= t_a
// (an instantaneous window).
//   fourier: 1 + sum_k a_k cos(phi_k(t_mid)) sinc(dphi_k / 2). The midpoint is m + r/2 with m = t_a + d/2 and
//            r = d & 1 on the positive difference d = t_b - t_a, exact for negative and zero-straddling windows (a
//            C-style (t_a + t_b) % 2 is not, F14); the phase error is harmonic_phase's at m.
//   pulse:   the exact integral of the snapped trapezoids, with the whole pieces summed in integers (no
//            "1 per period" shortcut: snapped periods differ from 1 by up to 8.9e-5, F10); relative error ~1e-16.
template <class T>
BALLA_HD T waveform_mean(const Waveform& w, int64_t t_a, int64_t t_b) {
  if (t_b <= t_a) return waveform_value<T>(w, t_a);
  switch (w.type) {
    case WaveformType::kFourier: {
      constexpr double kPi = detail::kTwoPi / 2;
      const int64_t d = t_b - t_a;
      const int64_t m = t_a + d / 2;
      const double r = static_cast<double>(d & 1);
      double mean = 1.0;
      for (int k = 0; k < w.n_harmonics; ++k) {
        const double cycles_per_ns = static_cast<double>(k + 1) * (w.f0_hz * 1e-9);
        const double phase = detail::harmonic_phase(w, k, m) + kPi * cycles_per_ns * r;
        mean += w.a[k] * pmath::cos(phase) * detail::sinc_cycles(cycles_per_ns * static_cast<double>(d));
      }
      return static_cast<T>(mean);
    }
    case WaveformType::kPulse: {
      int64_t u2 = 0;
      double part = 0.0;
      for (int64_t n = detail::pulse_period(w, t_a) - 1;; ++n) {
        int64_t c[4];
        detail::pulse_corners4(w, n, c);
        if (c[0] >= t_b) break;
        detail::pulse_unit_integral(c, t_a, t_b, &u2, &part);
      }
      return static_cast<T>(detail::pulse_mean(w, u2, part, t_b - t_a));
    }
    default:
      return T(1);
  }
}

// Exact means of many sub-intervals of one span (host only; frames.md §4.5). A pulse train's periods overlapping
// [span_a, span_b] are integrated once and prefix-summed in integers, so mean() costs O(log periods) plus four boundary
// periods and equals waveform_mean bitwise. Fourier and constant waveforms forward to waveform_mean.
class WaveformPrefix {
 public:
  WaveformPrefix(const Waveform& w, int64_t span_a, int64_t span_b) : w_(w), span_a_(span_a), span_b_(span_b) {
    if (span_b < span_a) throw std::invalid_argument("WaveformPrefix: span_b < span_a");
    if (w.type != WaveformType::kPulse) return;
    prefix_.push_back(0);
    for (int64_t n = detail::pulse_period(w, span_a) - 1;; ++n) {
      std::array<int64_t, 4> c;
      detail::pulse_corners4(w, n, c.data());
      if (c[0] >= span_b && !corners_.empty()) break;
      corners_.push_back(c);
      prefix_.push_back(prefix_.back() + (c[1] - c[0]) + 2 * (c[2] - c[1]) + (c[3] - c[2]));
    }
  }

  // The mean over [t0, t1], span_a <= t0 <= t1 <= span_b; m(t0) when t1 == t0.
  double mean(int64_t t0, int64_t t1) const {
    if (t0 < span_a_ || t1 > span_b_ || t1 < t0) {
      throw std::out_of_range("WaveformPrefix::mean: window outside the span");
    }
    if (w_.type != WaveformType::kPulse || t1 == t0) return waveform_mean<double>(w_, t0, t1);
    // Periods i0 - 1 .. i1 meet [t0, t1], in waveform_mean's order: i0 - 1 (its fall can reach 1 ns past c0(i0)),
    // i0, the run [i0 + 1, i1 - 1) that [t0, t1] covers whole (integers only, so its order is free), then i1 - 1
    // (explicit, so no run period's fall passes c0(i1)) and i1. Only boundary periods add to part, in increasing
    // order, as in waveform_mean.
    const size_t i0 = period_of(t0), i1 = period_of(t1 - 1);
    const size_t run_a = i0 + 1, run_b = std::max(run_a, i1 > 0 ? i1 - 1 : size_t{0});
    int64_t u2 = 0;
    double part = 0.0;
    for (size_t i = i0 > 0 ? i0 - 1 : 0; i < run_a; ++i) {
      detail::pulse_unit_integral(corners_[i].data(), t0, t1, &u2, &part);
    }
    u2 += prefix_[run_b] - prefix_[run_a];
    for (size_t i = run_b; i <= i1; ++i) detail::pulse_unit_integral(corners_[i].data(), t0, t1, &u2, &part);
    return detail::pulse_mean(w_, u2, part, t1 - t0);
  }

 private:
  // The last period whose corner 0 is <= t (t inside the span).
  size_t period_of(int64_t t) const {
    size_t lo = 0, hi = corners_.size();
    while (hi - lo > 1) {
      const size_t mid = lo + (hi - lo) / 2;
      (corners_[mid][0] <= t ? lo : hi) = mid;
    }
    return lo;
  }

  Waveform w_;
  int64_t span_a_, span_b_;
  std::vector<std::array<int64_t, 4>> corners_;  // corners 0..3 of the periods from pulse_period(span_a) - 1 on
  std::vector<int64_t> prefix_;                  // prefix_[i]: twice the unit integral of corners_[0..i), whole
};

}  // namespace balla::sensor
