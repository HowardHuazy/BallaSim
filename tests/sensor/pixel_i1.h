// FROZEN ORACLE (tests only): the per-pixel sensor kernel as of I1 (commit 4011ca1, src/sensor/pixel.h), kept
// verbatim in namespace balla::sensor::i1 so that the lean kernel of ENGINE_CPP I2 can be checked against
// the kernel it replaced, bit for bit (tests/sensor/test_lean.cpp). Do not edit; the shared maths it
// includes (substep.h, expcubic.h, noise.h, waveform.h, ...) are the production headers.
// The per-pixel sensor kernel (DESIGN §8.2-8.6, M3 §B-E), shared by the CPU threads and the GPU.
//
// One call advances one pixel from the model time to t_end through a single streaming loop:
//   1. Input: the photocurrent I(t) = sum_g m_g(t) I_g(t) + I_dk with I_g linear between the
//      pixel's samples and held after the last one, cut into linear pieces at sample times,
//      waveform breakpoints and the chord bound of M3 B.18-B.19 (eps_PL = eps_L / (8 (1 + V))).
//   2. Deterministic pass: adaptive sub-steps of the cascade (substep.h), never crossing a noise
//      cell boundary when noise is on (or a 2^23 ns window boundary when it is off).
//   3. Noise lattice: on entering cell k the frozen rate a1 is the exact cell average of k u
//      (M3 D.1), from a z-only look-ahead over the cell, and n(t_{k+1}) is drawn (M3 D.4).
//   4. Comparator pass, per chunk of sub-steps: a depth-first, time-ordered walk of the cell's
//      bridge tree that refines only where the exclusion test with margins cannot rule out a
//      crossing (M3 §D.6), and exact first-crossing search on the leaves (expcubic.h).
//   5. Events: physical reset at release or ideal (ESIM-compatible) reset, increments and labels
//      from the exact split Delta_s + Delta_n + Delta_l = Delta (M3 E.1-E.2).
// The deterministic sub-step sequence does not depend on the chunk capacity, and the comparator's
// tree values are keyed by absolute node times, so results do not depend on chunking either.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/pmath/explog.h"
#include "sensor/expcubic.h"
#include "sensor/maps.h"
#include "sensor/noise.h"
#include "sensor/noise_tables.h"
#include "sensor/substep.h"
#include "sensor/waveform.h"

namespace balla::sensor::i1 {

// Event labels (CONTRACTS C3) and flag bits (S1 §2.3 /gt/flags).
inline constexpr uint8_t kLabelSignal = 0;
inline constexpr uint8_t kLabelNoise = 1;
inline constexpr uint8_t kLabelLeak = 2;
inline constexpr uint8_t kLabelHot = 3;
inline constexpr uint16_t kFlagHotPixel = 1u << 5;
inline constexpr uint16_t kFlagQuarantined = 1u << 9;
inline constexpr uint16_t kFlagSolverFallback = 1u << 10;

inline constexpr int64_t kNotBlind = INT64_MIN;
// Without noise, sub-steps stop at 2^23 ns window boundaries: local fp32 time then keeps an ulp
// below 1 ns (M3 §B.9).
inline constexpr int kPlainSegmentLog2 = 23;
inline constexpr int64_t kPlainSegmentNs = int64_t{1} << kPlainSegmentLog2;

// The kernel ends a segment of its loop at every multiple of this grain (noise cells with noise,
// plain segments without), and carries nothing but the persistent state across segments. A call
// cut into time slices at multiples of the grain (PixelKernel::advance_slice) therefore gives
// bitwise the result of one call, and so do consecutive calls whose boundaries are multiples of it
// and carry the sample at the boundary over (every 2^23 ns window boundary qualifies).
BALLA_HD constexpr int64_t slice_grain(bool noise) { return noise ? kNoiseCellNs : kPlainSegmentNs; }

// Largest photocurrent one irradiance sample may contribute (fA): 1 uA, far above what a pixel
// delivers in direct sunlight (tens of nA). It is a numerical guard that keeps every product of the
// closed forms (k I h over a window, I^2 in the boundary-layer bound) finite in fp32; larger samples
// are clamped to it.
inline constexpr double kMaxSamplePhotocurrent_fA = 1e9;

// Input guard on one irradiance sample (M3 §B.6 guard 3): an overshooting reconstruction below 0
// becomes 0 and a sample above v_max becomes v_max. NaN maps to 0, although SensorModel refuses
// non-finite input before it reaches the kernel.
BALLA_HD inline float clamp_irradiance(float v, float v_max) { return v >= 0.0f ? (v <= v_max ? v : v_max) : 0.0f; }

// Persistent per-pixel state, stored in fp64 whatever the kernel precision.
struct PixelState {
  double z, x1, s2, s3;           // deterministic cascade; logs relative to ln(i_scale) (substep.h)
  double i_scale;                 // log offset of the pixel (fA): its mean photocurrent at init
  double tv;                      // memory-weighted total variation V of x1 (DESIGN §8.3)
  double nk[3], nk1[3];           // noise at the lattice nodes t_k and t_{k+1}
  double ln_a1;                   // ln of the frozen rate of cell k (valid when have_k1)
  int64_t cell;                   // k with t_k <= model time < t_{k+1}
  int64_t t_ref;                  // time the active reference was set
  double s3_ref, n3_ref, dr_ref;  // s3 and n3 at t_ref, reference offset (M3 E.1)
  int64_t blind_until;            // physical refractory: blind on (t_c, blind_until]
  int64_t t_search;               // first integer ns not yet searched for a crossing
  float held[kMaxLightGroups];    // last input sample (irradiance units)
  uint16_t flags;                 // event flags since the last event (or init), for the next one
  uint8_t have_k1;
  uint8_t reserved[5];
};

struct RawEvent {
  uint32_t pixel;
  uint8_t polarity;  // 1 = ON, 0 = OFF
  uint8_t label;
  uint16_t flags;
  int64_t t_cross;
  float ds, dn, dl;  // signal, noise and leak (+ reference offset) increments since t_ref
};

struct KernelStats {
  uint64_t substeps = 0;          // accepted deterministic sub-steps (pixel-steps)
  uint64_t substep_attempts = 0;  // including rejected halvings
  uint64_t settled_substeps = 0;  // constant sub-steps of settled pixels (no transcendental)
  uint64_t cells = 0;             // noise lattice steps
  uint64_t refinements = 0;       // bridge draws below the lattice
  uint64_t leaf_intervals = 0;    // elementary intervals that reached the exact solver
  uint64_t fallbacks = 0;         // sub-steps accepted at the 1 ns floor
  uint64_t input_clamps = 0;      // irradiance samples clamped by clamp_irradiance
  uint64_t quarantines = 0;       // non-finite states re-initialized

  BALLA_HD void add(const KernelStats& o) {
    substeps += o.substeps;
    substep_attempts += o.substep_attempts;
    settled_substeps += o.settled_substeps;
    cells += o.cells;
    refinements += o.refinements;
    leaf_intervals += o.leaf_intervals;
    fallbacks += o.fallbacks;
    input_clamps += o.input_clamps;
    quarantines += o.quarantines;
  }
};

// The samples of one pixel for one advance call.
struct InputRun {
  const int64_t* t;  // strictly increasing, t[0] >= model time, all <= t_end
  const float* irr;  // n x G irradiance (renderer units)
  int32_t n;
};

template <class T>
struct KernelConstants {
  CascadeConstants<T> cc;
  T eps_pl_base;  // eps_L / 8
  T scale;        // illuminance_scale, fA per irradiance unit
  float irr_max;  // kMaxSamplePhotocurrent_fA / scale: largest accepted irradiance sample
  int32_t groups;
  bool modulated;  // some light group has a non-constant waveform
  Waveform waves[kMaxLightGroups];
  bool noise;
  bool static_fast_path;  // PixelKernel::quiet_cell; off only to test its equivalence
  int32_t levels;
  T margin_z;
  NoiseTableView<T> tables;
  bool ideal_reset;
  int64_t refractory_ns;
  T eps_ci;
  uint64_t seed;
  uint32_t camera;
};

template <class T>
BALLA_HD bool is_finite(T x) {
  return x - x == T(0);
}

template <class T, class Sink>
class PixelKernel {
 public:
  BALLA_HD PixelKernel(const KernelConstants<T>& kc, const PixelMap& map, uint32_t pixel,
                       PixelState& ps, int64_t t_state, const InputRun& in, SubStep<T>* scratch,
                       int scratch_cap, Sink& sink, KernelStats& stats)
      : kc_(kc), pixel_(pixel), ps_(ps), t_state_(t_state), in_(in), scratch_(scratch),
        cap_(scratch_cap), sink_(sink), stats_(stats) {
    theta_on_ = static_cast<T>(map.theta_on);
    theta_off_ = static_cast<T>(map.theta_off);
    leak_ = static_cast<T>(map.leak);
    dark_ = static_cast<T>(map.dark_fA);
    phi0_ = static_cast<T>(map.phi0);
    hot_ = map.hot != HotType::kNone;
    skip_ = in_.n > 0 && in_.t[0] == t_state_ ? 1 : 0;
    load();
  }

  // Steady state at t0 with the input held at irr0 (M3 A.20). The deterministic state starts at
  // the period mean of the photocurrent (waveforms are mean-normalized), which is the periodic
  // orbit of the slow, dim pixels that would not settle during the warm-up.
  BALLA_HD void init(const float* irr0) {
    const int64_t t0 = t_state_;
    T mean_I = dark_;
    for (int g = 0; g < kc_.groups; ++g) {
      const float v = clamp_irradiance(irr0[g], kc_.irr_max);
      if (!(v == irr0[g])) ++stats_.input_clamps;
      ps_.held[g] = v;
      mean_I = mean_I + kc_.scale * static_cast<T>(v);
    }
    i_scale_ = mean_I;
    st_.z = T(1) / mean_I;
    x1_ = T(0);
    st_.s2 = st_.s3 = x1_;
    tv_ = T(0);
    T n3_0 = T(0);
    if (kc_.noise) {
      cell_ = t0 >> kNoiseCellLog2;
      const int64_t tk = cell_ * kNoiseCellNs;
      T xi[3], chol[6];
      normals3(kc_.seed, rng::Purpose::kNoiseInit, kc_.camera, pixel_, static_cast<uint64_t>(tk), xi);
      noise_lookup(kc_.tables, noise_pos(kc_.tables, pmath::log(kc_.cc.k * mean_I)),
                   NoiseLayout::kPinfChol, 6, chol);
      lower3_mul(chol, xi, nk_);
      have_k1_ = false;
      if (t0 != tk) {
        // The part of the cell before t0 is the steady state extended backwards.
        enter_cell(t0, kc_.cc.k * mean_I * static_cast<T>(t0 - tk) * T(1e-9));
        n3_0 = n3_at(t0);
      } else {
        n3_0 = nk_[2];
      }
    }
    set_reference(t0, st_.s3, n3_0, phi0_);
    blind_until_ = kNotBlind;
    t_search_ = t0 + 1;
    flags_ = 0;
    store();
  }

  // Advances the call that started at the model time (constructor) to t_end.
  BALLA_HD void advance(int64_t t_end) { advance_slice(t_state_, t_end, true); }

  // One time slice [t_from, t_to) of the call that started at the model time with the samples `in`.
  // A call may be run as consecutive slices, each by a fresh kernel object on the same state, cut
  // at multiples of slice_grain(kc.noise) (the GPU bounds its launches this way): the result is
  // bitwise that of advance(t_end). ends_call marks the last slice, after which the last sample is
  // held.
  BALLA_HD void advance_slice(int64_t t_from, int64_t t_to, bool ends_call) {
    int64_t t = t_from;
    while (t < t_to) {
      if (kc_.noise && kc_.static_fast_path && !have_k1_ && t + kNoiseCellNs <= t_to && quiet_cell(t)) {
        t += kNoiseCellNs;
        continue;
      }
      int64_t boundary;
      if (kc_.noise) {
        if (!have_k1_) enter_cell(t, T(0));
        boundary = (cell_ + 1) * kNoiseCellNs;
      } else {
        boundary = ((t >> kPlainSegmentLog2) + 1) * kPlainSegmentNs;  // floor division (arithmetic shift)
      }
      const int64_t seg_end = min(t_to, boundary);
      run_segment(t, seg_end);
      if (kc_.noise && seg_end == boundary) {
        for (int i = 0; i < 3; ++i) nk_[i] = nk1_[i];
        have_k1_ = false;
        ++cell_;
      }
      t = seg_end;
    }
    // The last sample is held until the next call.
    if (ends_call && in_.n > 0) {
      for (int g = 0; g < kc_.groups; ++g) ps_.held[g] = sample_value(n_aug() - 1, g);
    }
    store();
  }

 private:
  struct Node {
    int level;
    int64_t t;
    T xl[3], xr[3];
  };

  // A crossing found by search(). Model times may be negative (the run epoch is arbitrary, and the
  // warm-up of a run starting at t = 0 lies before it), so no time value can mean "none".
  struct Hit {
    bool found;
    bool on;
    int64_t t;
  };

  // ------------------------------------------------------------------------------------------
  // State load / store
  // ------------------------------------------------------------------------------------------
  BALLA_HD void load() {
    st_.z = static_cast<T>(ps_.z);
    i_scale_ = static_cast<T>(ps_.i_scale);
    x1_ = static_cast<T>(ps_.x1);
    st_.s2 = static_cast<T>(ps_.s2);
    st_.s3 = static_cast<T>(ps_.s3);
    tv_ = static_cast<T>(ps_.tv);
    for (int i = 0; i < 3; ++i) {
      nk_[i] = static_cast<T>(ps_.nk[i]);
      nk1_[i] = static_cast<T>(ps_.nk1[i]);
    }
    ln_a1_ = static_cast<T>(ps_.ln_a1);
    cell_ = ps_.cell;
    have_k1_ = ps_.have_k1 != 0;
    t_ref_ = ps_.t_ref;
    s3_ref_ = static_cast<T>(ps_.s3_ref);
    n3_ref_ = static_cast<T>(ps_.n3_ref);
    dr_ref_ = static_cast<T>(ps_.dr_ref);
    blind_until_ = ps_.blind_until;
    t_search_ = ps_.t_search;
    flags_ = ps_.flags;
    seg_ = 0;
  }

  BALLA_HD void store() {
    ps_.z = static_cast<double>(st_.z);
    ps_.i_scale = static_cast<double>(i_scale_);
    ps_.x1 = static_cast<double>(x1_);
    ps_.s2 = static_cast<double>(st_.s2);
    ps_.s3 = static_cast<double>(st_.s3);
    ps_.tv = static_cast<double>(tv_);
    for (int i = 0; i < 3; ++i) {
      ps_.nk[i] = static_cast<double>(nk_[i]);
      ps_.nk1[i] = static_cast<double>(nk1_[i]);
    }
    ps_.ln_a1 = static_cast<double>(ln_a1_);
    ps_.cell = cell_;
    ps_.have_k1 = have_k1_ ? 1 : 0;
    ps_.t_ref = t_ref_;
    ps_.s3_ref = static_cast<double>(s3_ref_);
    ps_.n3_ref = static_cast<double>(n3_ref_);
    ps_.dr_ref = static_cast<double>(dr_ref_);
    ps_.blind_until = blind_until_;
    ps_.t_search = t_search_;
    ps_.flags = flags_;
  }

  // ------------------------------------------------------------------------------------------
  // Input: samples augmented by the held value at the model time
  // ------------------------------------------------------------------------------------------
  BALLA_HD int n_aug() const { return 1 + in_.n - skip_; }

  BALLA_HD int64_t sample_time(int i) const { return i == 0 ? t_state_ : in_.t[i - 1 + skip_]; }

  BALLA_HD float sample_value(int i, int g) const {
    if (i == 0 && skip_ == 0) return ps_.held[g];
    return clamp_irradiance(in_.irr[static_cast<int64_t>(i - 1 + skip_) * kc_.groups + g], kc_.irr_max);
  }

  // Segment of the sample list containing t: [time(seg), time(seg + 1)), the last one is held.
  BALLA_HD int segment_of(int seg, int64_t t) const {
    const int last = n_aug() - 1;
    while (seg < last && sample_time(seg + 1) <= t) ++seg;
    return seg;
  }

  BALLA_HD int64_t segment_end(int seg) const {
    return seg + 1 < n_aug() ? sample_time(seg + 1) : kNoBreakpoint;
  }

  // Irradiance of every group at t in segment seg (linear between samples, held after the last).
  BALLA_HD void group_values(int seg, int64_t t, T* out) const {
    if (seg + 1 >= n_aug()) {
      for (int g = 0; g < kc_.groups; ++g) out[g] = static_cast<T>(sample_value(seg, g));
      return;
    }
    const int64_t t0 = sample_time(seg), t1 = sample_time(seg + 1);
    const T f = static_cast<T>(t - t0) / static_cast<T>(t1 - t0);
    for (int g = 0; g < kc_.groups; ++g) {
      const T v0 = static_cast<T>(sample_value(seg, g)), v1 = static_cast<T>(sample_value(seg + 1, g));
      out[g] = v0 + (v1 - v0) * f;
    }
  }

  BALLA_HD T photocurrent(int64_t t, const T* values) const {
    T I = dark_;
    for (int g = 0; g < kc_.groups; ++g) {
      const T m = kc_.waves[g].type == WaveformType::kConstant ? T(1) : waveform_value<T>(kc_.waves[g], t);
      I = I + m * kc_.scale * values[g];
    }
    return I;
  }

  // The longest linear piece [t, t_b], t_b <= limit, within one sample segment and between waveform
  // breakpoints, whose chord meets |ln I - ln chord| <= eps_pl (M3 B.18-B.19): dyadic halving of
  // the candidate until h^2 max|I''| <= 8 eps_pl I_min.
  BALLA_HD LinearPiece<T> next_piece(int& seg, int64_t t, int64_t limit, T eps_pl, bool& fallback) const {
    seg = segment_of(seg, t);
    int64_t tb = min(limit, segment_end(seg));
    T va[kMaxLightGroups], vb[kMaxLightGroups];
    group_values(seg, t, va);
    if (kc_.modulated) {
      for (int g = 0; g < kc_.groups; ++g) tb = min(tb, waveform_next_breakpoint(kc_.waves[g], t));
      while (true) {
        group_values(seg, tb, vb);
        const T h = static_cast<T>(tb - t) * T(1e-9);
        T i_min = dark_, i2 = T(0);
        for (int g = 0; g < kc_.groups; ++g) {
          const T lo = min(va[g], vb[g]), hi = max(va[g], vb[g]);
          if (kc_.waves[g].type == WaveformType::kConstant) {
            i_min = i_min + kc_.scale * lo;
            continue;
          }
          const WaveBounds<T> b = waveform_bounds<T>(kc_.waves[g], t, tb);
          i_min = i_min + kc_.scale * max(b.m_min, T(0)) * lo;
          i2 = i2 + kc_.scale * (b.d2m_max * hi + T(2) * b.dm_max * abs(vb[g] - va[g]) / h);
        }
        if (h * h * i2 <= T(8) * eps_pl * i_min) break;
        if (tb - t <= 1) {
          fallback = true;
          break;
        }
        tb = t + (tb - t) / 2;
      }
    } else {
      group_values(seg, tb, vb);
    }
    return {t, tb, photocurrent(t, va), photocurrent(tb, vb)};
  }

  BALLA_HD T eps_pl() const { return kc_.eps_pl_base / (T(1) + tv_); }

  // ------------------------------------------------------------------------------------------
  // Noise lattice
  // ------------------------------------------------------------------------------------------
  // Frozen rate of the cell containing t from a z-only look-ahead to its end (input held beyond
  // the last sample), then the lattice draw of n(t_{k+1}). `pre` is int k u before t (init only).
  BALLA_HD void enter_cell(int64_t t, T pre) {
    cell_ = t >> kNoiseCellLog2;  // floor division by the cell length (arithmetic shift)
    const int64_t tc1 = (cell_ + 1) * kNoiseCellNs;
    const T h_cell = static_cast<T>(kNoiseCellNs) * T(1e-9);
    int seg = seg_;
    bool fb = false;
    LinearPiece<T> pc = next_piece(seg, t, tc1, eps_pl(), fb);
    if (pc.t_b == tc1 && pc.I_a == pc.I_b && cascade_settled(kc_.cc, st_, x1_, pc.I_a)) {
      // Settled under constant input for the whole cell: a1 = k u = k / z, as in the main pass.
      if (pre == T(0) && st_.z == settled_z_) {
        ln_a1_ = settled_ln_a1_;
      } else {
        const T int_ku = pre + kc_.cc.k * static_cast<T>(tc1 - t) * T(1e-9) / st_.z;
        ln_a1_ = pmath::log(int_ku / h_cell);
        if (pre == T(0)) {
          settled_z_ = st_.z;
          settled_ln_a1_ = ln_a1_;
        }
      }
    } else {
      T z = st_.z, int_kI = T(0);
      while (true) {
        const T h = static_cast<T>(pc.t_b - pc.t_a) * T(1e-9);
        if (!(pc.I_a == pc.I_b && abs(z * pc.I_a - T(1)) <= kc_.cc.eps_drop)) {
          z = z_step(z, pc.I_a, pc.I_b, h, kc_.cc.k);
        }
        int_kI = int_kI + integrated_pole(pc.I_a, pc.I_b, h, kc_.cc.k);
        if (pc.t_b >= tc1) break;
        pc = next_piece(seg, pc.t_b, tc1, eps_pl(), fb);
      }
      const T int_ku = pre + pmath::log(z / st_.z) + int_kI;  // M3 D.1 via A.6
      ln_a1_ = pmath::log(max(int_ku / h_cell, T(1e-30)));
    }
    T xi[3];
    normals3(kc_.seed, rng::Purpose::kNoise, kc_.camera, pixel_, static_cast<uint64_t>(tc1), xi);
    const NoiseBlocks& nb = noise_blocks();
    lattice_step(nb.step, nb.step + 6, nk_, xi, nk1_);
    have_k1_ = true;
    ++stats_.cells;
  }

  // Interpolated table blocks of the current cell's rate, recomputed only when ln a1 changes (a
  // static pixel keeps one rate for every cell of a call).
  struct NoiseBlocks {
    T ln_a1;
    NoisePos<T> pos;
    T step[12];    // Phi (6) and chol Q (6) of the lattice step
    T margin0[23]; // level-0 exclusion margin
    bool valid;
  };

  BALLA_HD const NoiseBlocks& noise_blocks() {
    if (!blocks_.valid || blocks_.ln_a1 != ln_a1_) {
      blocks_.ln_a1 = ln_a1_;
      blocks_.pos = noise_pos(kc_.tables, ln_a1_);
      noise_lookup(kc_.tables, blocks_.pos, NoiseLayout::kStepPhi, 12, blocks_.step);
      if (kc_.levels > 0) noise_lookup(kc_.tables, blocks_.pos, NoiseLayout{kc_.levels}.margin(0), 23, blocks_.margin0);
      blocks_.valid = true;
    }
    return blocks_;
  }

  // Fast path for one whole cell starting at t, covered by a single constant input piece, of a
  // settled pixel: the common case of static pixels and of pixels between edges. It performs exactly
  // the operations of the general path (enter_cell's settled branch, the constant sub-step, the
  // level-0 exclusion of the tree walk) without its bookkeeping, so results are bitwise identical
  // (tests/sensor/test_pixel.cpp). Returns false when the cell needs the general path; the cell's
  // lattice draw is then already done.
  BALLA_HD bool quiet_cell(int64_t t) {
    if (kc_.modulated || blind_until_ != kNotBlind || kc_.levels == 0) return false;
    const int64_t tc1 = t + kNoiseCellNs;
    bool fallback = false;
    const LinearPiece<T> pc = next_piece(seg_, t, tc1, eps_pl(), fallback);
    if (pc.t_b != tc1 || pc.I_a != pc.I_b || !cascade_settled(kc_.cc, st_, x1_, pc.I_a)) return false;
    enter_cell(t, T(0));
    SubStep<T>& sub = scratch_[0];
    sub.t_a = t;
    sub.t_b = tc1;
    sub.s3 = ExpCubic<T>{{st_.s3, T(0), T(0), T(0)}, {T(0), T(0)}, {kc_.cc.a2, kc_.cc.a3}, 0};
    sub.int_ku = kc_.cc.k * (static_cast<T>(tc1 - t) * T(1e-9)) / st_.z;
    const int64_t lo = max(t_search_, t);
    if (lo < tc1) {
      T hc[4];
      noise_hermite(nk_, nk1_, kc_.cc.a3, static_cast<T>(kNoiseCellNs) * T(1e-9), hc);
      const T margin = noise_margin(noise_blocks().margin0, kc_.margin_z, nk_, nk1_);
      if (may_cross(delta_fn(sub, lo, hc, t), lo, tc1, lo, margin)) return false;
    }
    ++stats_.substeps;
    ++stats_.settled_substeps;
    if (tv_ != T(0)) tv_ = tv_ * pmath::exp(-sub.int_ku);
    t_search_ = max(t_search_, tc1);
    for (int i = 0; i < 3; ++i) nk_[i] = nk1_[i];
    have_k1_ = false;
    ++cell_;
    return true;
  }

  BALLA_HD int64_t level_length(int level) const { return kNoiseCellNs >> level; }

  BALLA_HD void bridge_child(int level, int64_t t_mid, const T* xl, const T* xr, T* xm) {
    T xi[3], blk[24];
    normals3(kc_.seed, rng::Purpose::kNoise, kc_.camera, pixel_, static_cast<uint64_t>(t_mid), xi);
    noise_lookup(kc_.tables, noise_blocks().pos, NoiseLayout{kc_.levels}.bridge(level), 24, blk);
    bridge_midpoint(blk, xl, xr, xi, xm);
    ++stats_.refinements;
  }

  // n3(t) in the current cell by descending the keyed tree to the leaf containing t.
  BALLA_HD T n3_at(int64_t t) {
    const int64_t tc0 = cell_ * kNoiseCellNs;
    if (t == tc0) return nk_[2];
    if (t == tc0 + kNoiseCellNs) return nk1_[2];
    T xl[3], xr[3], xm[3];
    for (int i = 0; i < 3; ++i) {
      xl[i] = nk_[i];
      xr[i] = nk1_[i];
    }
    int64_t node_t = tc0;
    for (int l = 1; l <= kc_.levels; ++l) {
      const int64_t mid = node_t + level_length(l);
      bridge_child(l, mid, xl, xr, xm);
      if (t == mid) return xm[2];
      if (t < mid) {
        for (int i = 0; i < 3; ++i) xr[i] = xm[i];
      } else {
        for (int i = 0; i < 3; ++i) xl[i] = xm[i];
        node_t = mid;
      }
    }
    T hc[4];
    noise_hermite(xl, xr, kc_.cc.a3, static_cast<T>(level_length(kc_.levels)) * T(1e-9), hc);
    const T u = static_cast<T>(t - node_t) * T(1e-9);
    return hc[0] + u * (hc[1] + u * (hc[2] + u * hc[3]));
  }

  // ------------------------------------------------------------------------------------------
  // Deterministic pass
  // ------------------------------------------------------------------------------------------
  BALLA_HD void run_segment(int64_t t0, int64_t t1) {
    int n_sub = 0;
    int64_t chunk_start = t0;
    int64_t t = t0;
    while (t < t1) {
      bool piece_fallback = false;
      const LinearPiece<T> pc = next_piece(seg_, t, t1, eps_pl(), piece_fallback);
      if (piece_fallback) flags_ |= kFlagSolverFallback;
      int64_t ta = pc.t_a;
      int64_t h_hint = 0;
      while (ta < pc.t_b) {
        const int64_t tb = make_substep(pc, ta, h_hint, scratch_[n_sub]);
        ta = tb;
        if (++n_sub == cap_) {
          comparator(n_sub, chunk_start, tb);
          n_sub = 0;
          chunk_start = tb;
        }
      }
      t = pc.t_b;
    }
    if (chunk_start < t1) comparator(n_sub, chunk_start, t1);
  }

  // One accepted sub-step starting at ta inside piece pc; returns its end.
  BALLA_HD int64_t make_substep(const LinearPiece<T>& pc, int64_t ta, int64_t& h_hint, SubStep<T>& sub) {
    CascadeState<T> next{};
    SubStepInfo info{};
    const int64_t tb = next_substep(kc_.cc, st_, i_scale_, x1_, pc, ta, pc.at(ta), h_hint, sub, next, info);
    ++stats_.substeps;
    stats_.substep_attempts += static_cast<uint64_t>(info.attempts);
    if (info.settled) {
      ++stats_.settled_substeps;
      if (tv_ != T(0)) tv_ = tv_ * pmath::exp(-sub.int_ku);
      return tb;
    }
    if (sub.fallback) {
      flags_ |= kFlagSolverFallback;
      ++stats_.fallbacks;
    }
    if (!(is_finite(next.z) && next.z > T(0) && is_finite(next.s2) && is_finite(next.s3))) {
      quarantine(pc.at(tb), sub);
    } else {
      st_ = next;
      x1_ = sub.x1_b;
      // V <- V e^{-int k u} + |dx1|: total variation of x1 over the stage-1 memory (DESIGN §8.3).
      tv_ = tv_ * pmath::exp(-sub.int_ku) + abs(sub.x1_b - sub.x1_a);
    }
    return tb;
  }

  // Numerical guard (DESIGN §8.3, S1 §7.2): re-initialize a non-finite cascade to steady state at
  // the current input; the next event carries QUARANTINED (also across a blind release).
  BALLA_HD void quarantine(T I, SubStep<T>& sub) {
    st_.z = T(1) / I;
    x1_ = -pmath::log(st_.z * i_scale_);
    st_.s2 = st_.s3 = x1_;
    tv_ = T(0);
    sub.s3 = ExpCubic<T>{{st_.s3, T(0), T(0), T(0)}, {T(0), T(0)}, {kc_.cc.a2, kc_.cc.a3}, 0};
    sub.x1_a = sub.x1_b = x1_;
    flags_ |= kFlagQuarantined;
    ++stats_.quarantines;
  }

  // ------------------------------------------------------------------------------------------
  // Comparator pass
  // ------------------------------------------------------------------------------------------
  BALLA_HD void set_reference(int64_t t, T s3, T n3, T offset) {
    t_ref_ = t;
    s3_ref_ = s3;
    n3_ref_ = n3;
    dr_ref_ = offset;
  }

  // Index of the sub-step containing t (t_a <= t < t_b, or the last one at its end).
  BALLA_HD int substep_at(int n_sub, int64_t t) const {
    int lo = 0, hi = n_sub - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) / 2;
      if (scratch_[mid].t_a <= t) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  BALLA_HD T s3_at(int n_sub, int64_t t) const {
    const SubStep<T>& sub = scratch_[substep_at(n_sub, t)];
    return sub.s3(static_cast<T>(t - sub.t_a) * T(1e-9));
  }

  // Delta on [e0, ...) of sub-step sub in local u = (t - e0) s, with the noise Hermite hc of the node
  // starting at node_t (or no noise): det part, noise, leak and the reference offset.
  BALLA_HD ExpCubic<T> delta_fn(const SubStep<T>& sub, int64_t e0, const T* hc, int64_t node_t) const {
    ExpCubic<T> g = e0 == sub.t_a ? sub.s3 : shifted(sub.s3, static_cast<T>(e0 - sub.t_a) * T(1e-9));
    T c0 = g.c[0] - s3_ref_;
    if (hc != nullptr) {
      const T d = static_cast<T>(e0 - node_t) * T(1e-9);
      c0 = c0 + ((hc[0] + d * (hc[1] + d * (hc[2] + d * hc[3]))) - n3_ref_);
      g.c[1] = g.c[1] + (hc[1] + d * (T(2) * hc[2] + d * (T(3) * hc[3])));
      g.c[2] = g.c[2] + (hc[2] + d * (T(3) * hc[3]));
      g.c[3] = g.c[3] + hc[3];
    }
    const T since_ref = static_cast<T>(static_cast<double>(e0 - t_ref_) * 1e-9);
    g.c[0] = c0 + (leak_ * since_ref + dr_ref_);
    g.c[1] = g.c[1] + leak_;
    return g;
  }

  // Exclusion test of a non-leaf tree node (M3 C.3 with the margin of D.8): can Delta, widened by
  // `margin`, reach either level on the integers [max(lo, e0), e1)?
  BALLA_HD bool may_cross(const ExpCubic<T>& delta, int64_t e0, int64_t e1, int64_t lo, T margin) const {
    const int64_t k_lo = max(lo, e0) - e0, k_hi = e1 - 1 - e0;
    if (k_lo > k_hi) return false;
    const Range<T> r = range(delta, static_cast<T>(k_lo) * T(1e-9), static_cast<T>(k_hi) * T(1e-9));
    return r.hi + margin >= theta_on_ || r.lo - margin <= -theta_off_;
  }

  // First crossing of Delta on the integers [max(lo, e0), e1); found = false if there is none.
  // first_firing works on offsets k >= 0 from e0, where -1 is a safe "none".
  BALLA_HD Hit search(const ExpCubic<T>& delta, int64_t e0, int64_t e1, int64_t lo) const {
    constexpr Hit kNone{false, false, 0};
    const int64_t k_lo = max(lo, e0) - e0, k_hi = e1 - 1 - e0;
    if (k_lo > k_hi) return kNone;
    const T u_lo = static_cast<T>(k_lo) * T(1e-9), u_hi = static_cast<T>(k_hi) * T(1e-9);
    const Range<T> r = range(delta, u_lo, u_hi);
    const bool on = r.hi >= theta_on_, off = r.lo <= -theta_off_;
    if (!(on || off)) return kNone;
    ExpCubic<T> gon = delta, goff = delta;
    gon.c[0] = gon.c[0] - theta_on_;
    for (int m = 0; m < 4; ++m) goff.c[m] = -goff.c[m];
    goff.C[0] = -goff.C[0];
    goff.C[1] = -goff.C[1];
    goff.c[0] = goff.c[0] - theta_off_;
    // Both firing functions share Delta's critical points (Polya-Rolle isolation).
    RootList<T> crit;
    if (u_hi > u_lo) crit = critical_points(delta, u_lo, u_hi);
    const int64_t k_on = on ? first_firing(gon, crit, k_lo, k_hi) : -1;
    const int64_t k_off = off ? first_firing(goff, crit, k_lo, k_hi) : -1;
    if (k_on < 0 && k_off < 0) return kNone;
    if (k_off < 0 || (k_on >= 0 && k_on <= k_off)) return {true, true, e0 + k_on};
    return {true, false, e0 + k_off};
  }

  // Emits the event at t with polarity `on` and updates the reference (M3 A.16-A.17, E.1-E.5).
  BALLA_HD void fire(int64_t t, bool on, T s3, T n3) {
    const T ds = s3 - s3_ref_;
    const T dn = n3 - n3_ref_;
    const T dl = leak_ * static_cast<T>(static_cast<double>(t - t_ref_) * 1e-9) + dr_ref_;
    const T D = ds + dn + dl;  // = Delta(t) >= theta_on or <= -theta_off
    // Shares p Delta_x / theta sum to one; Phase 1 labels by the increment split alone (DESIGN
    // §9.2 simplified: the shared-history counterfactual is Phase 2).
    const T a_s = ds / D, a_l = dl / D;
    uint8_t label = kLabelNoise;
    if (a_s >= T(0.5) && a_s >= a_l) label = kLabelSignal;
    else if (a_l >= T(0.5) && a_l > a_s) label = kLabelLeak;
    if (hot_ && label != kLabelSignal) label = kLabelHot;
    RawEvent ev;
    ev.pixel = pixel_;
    ev.polarity = on ? 1 : 0;
    ev.label = label;
    ev.flags = static_cast<uint16_t>(flags_ | (hot_ ? kFlagHotPixel : 0));
    ev.t_cross = t;
    ev.ds = static_cast<float>(ds);
    ev.dn = static_cast<float>(dn);
    ev.dl = static_cast<float>(dl);
    sink_.push(ev);
    flags_ = 0;
    if (kc_.ideal_reset) {
      // r += p theta: the overshoot past the level is carried as the new reference offset.
      set_reference(t, s3, n3, D - (on ? theta_on_ : -theta_off_));
      t_search_ = t + 1 + kc_.refractory_ns;
    } else if (kc_.refractory_ns == 0) {
      set_reference(t, s3, n3, -kc_.eps_ci);
      t_search_ = t + 1;
    } else {
      blind_until_ = t + kc_.refractory_ns;
      t_search_ = blind_until_ + 1;
    }
  }

  // Release of the physical refractory inside the chunk: r = x3(t_rel) + eps_ci (M3 A.16).
  BALLA_HD void release(int n_sub) {
    const int64_t t = blind_until_;
    set_reference(t, s3_at(n_sub, t), kc_.noise ? n3_at(t) : T(0), -kc_.eps_ci);
    blind_until_ = kNotBlind;
  }

  BALLA_HD void comparator(int n_sub, int64_t c0, int64_t c1) {
    if (kc_.noise) comparator_tree(n_sub, c0, c1);
    else comparator_plain(n_sub, c0, c1);
  }

  // Returns false while the physical refractory covers the rest of the chunk.
  BALLA_HD bool handle_blind(int n_sub, int64_t c1) {
    if (blind_until_ == kNotBlind) return true;
    if (blind_until_ >= c1) return false;
    release(n_sub);
    return true;
  }

  BALLA_HD void comparator_plain(int n_sub, int64_t c0, int64_t c1) {
    while (true) {
      if (!handle_blind(n_sub, c1)) return;
      const int64_t lo = max(t_search_, c0);
      if (lo >= c1) return;
      bool fired = false;
      for (int i = substep_at(n_sub, lo); i < n_sub && !fired; ++i) {
        const SubStep<T>& sub = scratch_[i];
        const int64_t e0 = max(sub.t_a, lo), e1 = min(sub.t_b, c1);
        if (e0 >= e1) continue;
        const ExpCubic<T> d = delta_fn(sub, e0, nullptr, 0);
        ++stats_.leaf_intervals;
        const Hit hit = search(d, e0, e1, lo);
        if (hit.found) {
          fire(hit.t, hit.on, sub.s3(static_cast<T>(hit.t - sub.t_a) * T(1e-9)), T(0));
          fired = true;
        }
      }
      if (!fired) {
        t_search_ = max(t_search_, c1);
        return;
      }
    }
  }

  // Depth-first, time-ordered walk of the cell's bridge tree over the chunk [c0, c1) (M3 §D.6).
  BALLA_HD void comparator_tree(int n_sub, int64_t c0, int64_t c1) {
    const int64_t tc0 = cell_ * kNoiseCellNs;
    while (true) {
      if (!handle_blind(n_sub, c1)) return;
      const int64_t lo = max(t_search_, c0);
      if (lo >= c1) return;
      Node stack[12];
      int sp = 0;
      stack[sp].level = 0;
      stack[sp].t = tc0;
      for (int i = 0; i < 3; ++i) {
        stack[sp].xl[i] = nk_[i];
        stack[sp].xr[i] = nk1_[i];
      }
      ++sp;
      bool fired = false;
      while (sp > 0 && !fired) {
        const Node node = stack[--sp];
        const int64_t node_end = node.t + level_length(node.level);
        const int64_t span_lo = max(node.t, lo), span_hi = min(node_end, c1);
        if (span_lo >= span_hi) continue;
        T hc[4];
        noise_hermite(node.xl, node.xr, kc_.cc.a3, static_cast<T>(level_length(node.level)) * T(1e-9), hc);
        if (node.level < kc_.levels) {
          T mblk[23];
          const T* mrow = mblk;
          if (node.level == 0) {
            mrow = noise_blocks().margin0;
          } else {
            noise_lookup(kc_.tables, noise_blocks().pos, NoiseLayout{kc_.levels}.margin(node.level), 23, mblk);
          }
          const T margin = noise_margin(mrow, kc_.margin_z, node.xl, node.xr);
          bool may_fire = false;
          for (int i = substep_at(n_sub, span_lo); i < n_sub && !may_fire; ++i) {
            const SubStep<T>& sub = scratch_[i];
            const int64_t e0 = max(sub.t_a, span_lo), e1 = min(sub.t_b, span_hi);
            if (e0 >= e1) {
              if (sub.t_a >= span_hi) break;
              continue;
            }
            may_fire = may_cross(delta_fn(sub, e0, hc, node.t), e0, e1, lo, margin);
          }
          if (!may_fire) continue;
          const int64_t mid = node.t + level_length(node.level + 1);
          Node right, left;
          right.level = left.level = node.level + 1;
          left.t = node.t;
          right.t = mid;
          T xm[3];
          bridge_child(node.level + 1, mid, node.xl, node.xr, xm);
          for (int i = 0; i < 3; ++i) {
            left.xl[i] = node.xl[i];
            left.xr[i] = xm[i];
            right.xl[i] = xm[i];
            right.xr[i] = node.xr[i];
          }
          stack[sp++] = right;
          stack[sp++] = left;
          continue;
        }
        for (int i = substep_at(n_sub, span_lo); i < n_sub && !fired; ++i) {
          const SubStep<T>& sub = scratch_[i];
          const int64_t e0 = max(sub.t_a, span_lo), e1 = min(sub.t_b, span_hi);
          if (e0 >= e1) {
            if (sub.t_a >= span_hi) break;
            continue;
          }
          ++stats_.leaf_intervals;
          const Hit hit = search(delta_fn(sub, e0, hc, node.t), e0, e1, lo);
          if (hit.found) {
            const T u = static_cast<T>(hit.t - node.t) * T(1e-9);
            fire(hit.t, hit.on, sub.s3(static_cast<T>(hit.t - sub.t_a) * T(1e-9)),
                 hc[0] + u * (hc[1] + u * (hc[2] + u * hc[3])));
            fired = true;
          }
        }
      }
      if (!fired) {
        t_search_ = max(t_search_, c1);
        return;
      }
    }
  }

  // ------------------------------------------------------------------------------------------
  const KernelConstants<T>& kc_;
  uint32_t pixel_;
  PixelState& ps_;
  int64_t t_state_;
  InputRun in_;
  SubStep<T>* scratch_;
  int cap_;
  Sink& sink_;
  KernelStats& stats_;
  int skip_ = 0;
  int seg_ = 0;

  T theta_on_, theta_off_, leak_, dark_, phi0_;
  bool hot_;

  NoiseBlocks blocks_{};
  T settled_z_ = T(-1), settled_ln_a1_ = T(0);  // rate of a settled cell, reused while z is unchanged
  CascadeState<T> st_{};
  T i_scale_ = T(1);
  T x1_ = T(0), tv_ = T(0);
  T nk_[3] = {}, nk1_[3] = {};
  T ln_a1_ = T(0);
  int64_t cell_ = 0;
  bool have_k1_ = false;
  int64_t t_ref_ = 0;
  T s3_ref_ = T(0), n3_ref_ = T(0), dr_ref_ = T(0);
  int64_t blind_until_ = kNotBlind;
  int64_t t_search_ = 0;
  uint16_t flags_ = 0;
};

}  // namespace balla::sensor::i1
