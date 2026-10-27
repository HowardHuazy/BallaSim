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
//   4. Comparator, streamed: each sub-step, as soon as it is accepted, is tested against the
//      exclusion hull of the cell (level 0 of the bridge tree, kept per cell); only where that
//      cannot rule out a crossing does a depth-first, time-ordered walk refine the tree (M3 §D.6),
//      with exact first-crossing search on the leaves (expcubic.h).
//   5. Events: physical reset at release or ideal (ESIM-compatible) reset, increments and labels
//      from the exact split Delta_s + Delta_n + Delta_l = Delta (M3 E.1-E.2).
// Tree values are keyed by absolute node times, so the result does not depend on how many sub-steps
// one comparator call sees (until I2 the comparator ran per chunk of up to 16 or 256 sub-steps).
//
// Lean form (ENGINE_CPP §1.5 item 2, I2). Everything below performs the operations of the I1 kernel
// (tests/sensor/pixel_i1.h, the frozen oracle) in the same order, so states and events are bitwise
// unchanged (tests/sensor/test_lean.cpp) except in two ways: a solver-fallback or quarantine flag goes
// to the first event after its own sub-step (I1 also put it on earlier events of its 16-sub-step
// chunk), and the per-sub-step walk refines only the nodes that fail for that sub-step, so the two
// could differ where the margin bound (z_m, a ~1e-9 miss) excludes a leaf that crosses, as I1's CPU
// and GPU chunk sizes could. What changed is what is kept and re-evaluated:
//   - the comparator runs per sub-step (above) instead of per chunk of sub-steps;
//   - groups whose irradiance is zero over the whole call ("dead") are skipped: they add exactly 0;
//   - a pixel whose live groups are all constant takes the quiet-cell fast path even when some
//     group of the sensor is modulated (pulse breakpoints still cut its pieces);
//   - the quiet cell resolves a non-excluded level-0 hull by a walk that fires nothing, and leaves
//     the cell to the general path only when that walk finds a crossing;
//   - the level-0 hull of a cell (Hermite and margin) is computed once per cell, and the cell's first
//     piece and the look-ahead's z over it are reused by the main pass (CellMemo, ZMemo);
//   - waveform values and next breakpoints at multiples of 2^10 ns and bounds over whole cells may
//     come from a per-window table (WaveTable) filled by the same functions;
//   - advance_quiet() runs only quiet cells and stops at the first other one, so that the GPU can
//     run quiet pixels in a lean pass and resume the rest in the general pass at a cell boundary.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/pmath/explog.h"
#include "sensor/census.h"
#include "sensor/expcubic.h"
#include "sensor/hpf.h"
#include "sensor/maps.h"
#include "sensor/noise.h"
#include "sensor/noise_tables.h"
#include "sensor/substep.h"
#include "sensor/waveform.h"

// The kernel's hot path (per cell and sub-step) is forced inline, so that the per-pixel state stays in
// registers on the GPU; rare and large paths (the tree walk below level 0, the exact crossing search,
// the n3 descent at a release, direct waveform evaluations) are pure functions kept out of line, one
// copy each. (A kernel's register allocation still covers them: docs/dev/SENSOR.md "GPU execution".)
#if defined(__CUDACC__)
#define BALLA_SENSOR_INLINE __forceinline__
#define BALLA_SENSOR_COLD __noinline__
#else
#define BALLA_SENSOR_INLINE inline
#define BALLA_SENSOR_COLD
#endif

namespace balla::sensor {

// Event labels (CONTRACTS C3) and flag bits (S1 §2.3 /gt/flags).
inline constexpr uint8_t kLabelSignal = 0;
inline constexpr uint8_t kLabelNoise = 1;
inline constexpr uint8_t kLabelLeak = 2;
inline constexpr uint8_t kLabelHot = 3;
inline constexpr uint16_t kFlagHotPixel = 1u << 5;
inline constexpr uint16_t kFlagQuarantined = 1u << 9;
inline constexpr uint16_t kFlagSolverFallback = 1u << 10;
// The pixel's first event after init (F034): its label leaves out the reference offset, which then holds Delta(t0).
inline constexpr uint16_t kFlagFirstAfterInit = 1u << 12;

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
  uint16_t flags;                 // event flags accumulated since the last event (FIRST_AFTER_INIT from init)
  uint8_t have_k1;
  uint8_t reserved;
  float rho_m1;                   // rho - 1 of cell k (PHASE2 D-P2-2; valid when have_k1): 0 under frozen diffusion
};

struct RawEvent {
  uint32_t pixel;
  uint8_t polarity;  // 1 = ON, 0 = OFF
  uint8_t label;
  uint16_t flags;
  int64_t t_cross;
  float ds, dn, dl;  // signal, noise and leak (+ reference offset) increments since t_ref
  // The leak since t0 that first_label took (less the high-pass filter's ml_leak) for a FIRST_AFTER_INIT event, 0
  // otherwise: the E tier relabels such an event from it (rederive.h, phase3/ui.md §3.10). It fills the struct's
  // padding.
  float l_first;
};
static_assert(sizeof(RawEvent) == 32, "RawEvent: l_first fills the padding (the device event buffers keep their size)");

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
  uint64_t pieces = 0;            // linear input pieces of the main pass (a quiet cell is one; P2-0b)
  uint64_t hpf_splits = 0;        // comparator intervals halved for the HPF's b-mode bound (hpf.h; CPU only)
#if BALLA_SENSOR_CENSUS
  SensorCensus census;  // the census build's counters (census.h)
#endif

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
    pieces += o.pieces;
    hpf_splits += o.hpf_splits;
#if BALLA_SENSOR_CENSUS
    census.add(o.census);
#endif
  }
};

// The same counters in 32 bits, for one pixel and call on the GPU (KernelStats sums them; the device does not report
// hpf_splits).
struct KernelStats32 {
  uint32_t substeps = 0, substep_attempts = 0, settled_substeps = 0, cells = 0, refinements = 0, leaf_intervals = 0,
           fallbacks = 0, input_clamps = 0, quarantines = 0, pieces = 0, hpf_splits = 0;
};

// The samples of one pixel for one advance call.
struct InputRun {
  const int64_t* t;  // strictly increasing, t[0] >= model time, all <= t_end
  const float* irr;  // n x G irradiance (renderer units)
  int32_t n;
};

// Memo of the waveforms for one window of a call, filled with the functions of waveform.h themselves,
// so that a lookup returns the bits of a direct evaluation: m_g and its next breakpoint at every
// multiple of 2^10 ns (the engine's sample times: lattice and refinement midpoints at min_dt >= 2^10),
// and the bounds of waveform_bounds over whole noise cells. Entries of constant groups, and
// breakpoints of groups that are not pulse trains, are neither filled nor read. n_grid = 0: no table
// (every evaluation is direct).
inline constexpr int kWaveGridLog2 = 10;
inline constexpr int64_t kWaveGridNs = int64_t{1} << kWaveGridLog2;

template <class T>
struct WaveTable {
  const T* value = nullptr;             // [g * (n_grid + 1) + j]: m_g(t0 + j 2^10)
  const int64_t* breakpoint = nullptr;  // [g * (n_grid + 1) + j]: waveform_next_breakpoint at t0 + j 2^10
  const T* bounds = nullptr;            // [(g * n_cells + c) * 3 + i]: m_min, dm_max, d2m_max over cell k0 + c
  int64_t t0 = 0;                       // a multiple of 2^10
  int32_t n_grid = 0;
  int64_t k0 = 0;
  int32_t n_cells = 0;
};

// Veiling glare over one call (D13 item 3; SensorModel::set_glare): per light group the uniform glare
// irradiance g x the frame-mean image irradiance, a at t_a (the call's start) and b at t_b (its end), linear in between
// (renderer units; a pixel takes it over its gain, so its photocurrent gains scale x glare). Off (g = 0): no term.
template <class T>
struct GlareRamp {
  bool on;
  int64_t t_a, t_b;
  T a[kMaxLightGroups], b[kMaxLightGroups];
};

template <class T>
struct KernelConstants {
  CascadeConstants<T> cc;
  T eps_pl_base;  // eps_L / 8
  T scale;        // illuminance_scale, fA per irradiance unit (a pixel's own: scale x PixelMap::gain)
  float irr_max;  // kMaxSamplePhotocurrent_fA / (scale x the largest gain): largest accepted irradiance sample
  int32_t groups;
  bool modulated;        // some light group has a non-constant waveform
  bool breakpoints;      // some light group is a pulse train (its breakpoints cut every pixel's pieces)
  uint32_t wave_mask;    // bit g: group g is not constant
  Waveform waves[kMaxLightGroups];
  WaveTable<T> table;
  GlareRamp<T> glare;    // the call's (set by SensorModel before each init and advance)
  bool noise;
  bool static_fast_path;  // PixelKernel::quiet_cell; off only to test its equivalence
  bool instant;           // instantaneous log response (D3; PixelKernel::advance_instant), noise off
  int32_t levels;
  T margin_z;
  NoiseTableView<T> tables;
  bool transient;  // accuracy.noise_diffusion transient: a cell's diffusion ratio rho (PHASE2 D-P2-2; split tables)
  bool ideal_reset;
  int64_t refractory_ns;
  T eps_ci;
  T label_cut;  // SensorParams::label_share_cut (0.5 is exact in fp32 and fp64)
  uint64_t seed;
  uint32_t camera;
  // The high-pass filter (hpf.h; SensorModel dispatches the kHpf kernels while it is on): its rate b = 1/tau_hp
  // (s^-1; 0 only under the b = 0 test hook), or per pixel from hpf_b_map (tau_hp_sigma_ln > 0; device memory on the
  // GPU), and the bound eps_hp = eps_L / 32 of its b-mode in the comparator.
  bool hpf;
  T hpf_b;
  const float* hpf_b_map;
  T eps_hp;
  // A region-of-interest model (SensorModel's roi; phase3/ui.md §3.10): the global id y * W + x of each of its pixels,
  // which keys every random draw and names the events, so that they are the full frame's; null: the index is the id.
  // Maps, states and hpf_b_map stay indexed by the model's own pixel index.
  const uint32_t* pixel_ids;
};

template <class T>
BALLA_HD bool is_finite(T x) {
  return x - x == T(0);
}

namespace detail {

// Direct waveform evaluations where the call's table does not hold the value (pieces cut at sample
// times): out of line, since they are rare and large (fp64 phase reduction, trig, pulse corners).
template <class T>
BALLA_HD BALLA_SENSOR_COLD T wave_value_direct(const Waveform& w, int64_t t) {
  return waveform_value<T>(w, t);
}
template <class T>
BALLA_HD BALLA_SENSOR_COLD WaveBounds<T> wave_bounds_direct(const Waveform& w, int64_t t_a, int64_t t_b) {
  return waveform_bounds<T>(w, t_a, t_b);
}
BALLA_HD BALLA_SENSOR_COLD inline int64_t next_breakpoint_direct(const Waveform& w, int64_t t) {
  return waveform_next_breakpoint(w, t);
}

// The call's veiling glare of group g at t over the pixel's gain (PixelKernel::group_value; out of line, off the paths
// of glare-free runs): linear in t, exact at both ends (a constant ramp is constant), so that consecutive calls join
// without a step. Linear pieces stay linear: no extra cut points.
template <class T>
BALLA_HD BALLA_SENSOR_COLD T glare_value(const GlareRamp<T>& r, int64_t t, int g, float gain) {
  T v = r.a[g];
  if (r.b[g] != v) {
    const T f = static_cast<T>(t - r.t_a) / static_cast<T>(r.t_b - r.t_a);
    v = (T(1) - f) * r.a[g] + f * r.b[g];
  }
  return v / static_cast<T>(gain);
}

// Phase 1 labels by the increment split alone (DESIGN §9.2 simplified: the shared-history counterfactual is Phase 2),
// at the label cut (D14a): with shares a_x = d_x / D, SIGNAL if a_s >= cut and a_s >= a_l, LEAK if a_l >= cut and
// a_l > a_s, NOISE otherwise.
template <class T>
BALLA_HD BALLA_SENSOR_INLINE uint8_t split_label(T ds, T dl, T D, T cut) {
  const T a_s = ds / D, a_l = dl / D;
  if (a_s >= cut && a_s >= a_l) return kLabelSignal;
  if (a_l >= cut && a_l > a_s) return kLabelLeak;
  return kLabelNoise;
}

// The label of a pixel's first event after init (F034): init's reference is the pixel's own s3 and n3 at t0 and its
// offset Delta(t0) (phi0, or the stationary state's), so the split of the increments without the offset, ds, dn and
// the leak since t0, l, is that of its own increments since t0. Their total is the event's level less Delta(t0), on
// the polarity's side unless Delta(t0) lay within rounding of the level: NOISE then.
template <class T>
BALLA_HD BALLA_SENSOR_COLD uint8_t first_label(bool on, T ds, T dn, T l, T cut) {
  const T D = ds + dn + l;
  return (on ? D > T(0) : D < T(0)) ? split_label(ds, l, D, cut) : kLabelNoise;
}

// What the comparator compares against (M3 E.1): the reference set at t_ref, the leak and the levels.
template <class T>
struct Comparand {
  int64_t t_ref;
  T s3_ref, n3_ref, dr_ref;
  T leak, theta_on, theta_off;
};

template <class T>
struct Triple {
  T v[3];
};

// Delta on [e0, ...) of sub-step sub in local u = (t - e0) s, with the noise Hermite hc of the node
// starting at node_t (or no noise): det part, noise, leak and the reference offset.
template <class T>
BALLA_HD BALLA_SENSOR_INLINE ExpCubic<T> delta_fn(const SubStep<T>& sub, int64_t e0, const T* hc, int64_t node_t,
                                                  const Comparand<T>& c) {
  ExpCubic<T> g = e0 == sub.t_a ? sub.s3 : shifted(sub.s3, static_cast<T>(e0 - sub.t_a) * T(1e-9));
  T c0 = g.c[0] - c.s3_ref;
  if (hc != nullptr) {
    const T d = static_cast<T>(e0 - node_t) * T(1e-9);
    c0 = c0 + ((hc[0] + d * (hc[1] + d * (hc[2] + d * hc[3]))) - c.n3_ref);
    g.c[1] = g.c[1] + (hc[1] + d * (T(2) * hc[2] + d * (T(3) * hc[3])));
    g.c[2] = g.c[2] + (hc[2] + d * (T(3) * hc[3]));
    g.c[3] = g.c[3] + hc[3];
  }
  const T since_ref = static_cast<T>(static_cast<double>(e0 - c.t_ref) * 1e-9);
  g.c[0] = c0 + (c.leak * since_ref + c.dr_ref);
  g.c[1] = g.c[1] + c.leak;
  return g;
}

// Exclusion test of a non-leaf tree node (M3 C.3 with the margin of D.8): can Delta, widened by
// `margin`, reach either level on the integers [max(lo, e0), e1)?
template <class T>
BALLA_HD BALLA_SENSOR_INLINE bool may_cross(const ExpCubic<T>& delta, int64_t e0, int64_t e1, int64_t lo, T margin,
                                            const Comparand<T>& c) {
  const int64_t k_lo = max(lo, e0) - e0, k_hi = e1 - 1 - e0;
  if (k_lo > k_hi) return false;
  const Range<T> r = range(delta, static_cast<T>(k_lo) * T(1e-9), static_cast<T>(k_hi) * T(1e-9));
  return r.hi + margin >= c.theta_on || r.lo - margin <= -c.theta_off;
}

// A crossing. Model times may be negative (the run epoch is arbitrary, and the warm-up of a run
// starting at t = 0 lies before it), so no time value can mean "none".
struct Hit {
  bool found;
  bool on;
  int64_t t;
};

// First crossing of Delta on the integers [max(lo, e0), e1); found = false if there is none.
// first_firing works on offsets k >= 0 from e0, where -1 is a safe "none".
template <class T>
BALLA_HD BALLA_SENSOR_COLD Hit search(const ExpCubic<T> delta, int64_t e0, int64_t e1, int64_t lo, T theta_on,
                                      T theta_off) {
  constexpr Hit kNone{false, false, 0};
  const int64_t k_lo = max(lo, e0) - e0, k_hi = e1 - 1 - e0;
  if (k_lo > k_hi) return kNone;
  const T u_lo = static_cast<T>(k_lo) * T(1e-9), u_hi = static_cast<T>(k_hi) * T(1e-9);
  const Range<T> r = range(delta, u_lo, u_hi);
  const bool on = r.hi >= theta_on, off = r.lo <= -theta_off;
  if (!(on || off)) return kNone;
  ExpCubic<T> gon = delta, goff = delta;
  gon.c[0] = gon.c[0] - theta_on;
  for (int m = 0; m < 4; ++m) goff.c[m] = -goff.c[m];
  goff.C[0] = -goff.C[0];
  goff.C[1] = -goff.C[1];
  goff.c[0] = goff.c[0] - theta_off;
  // Both firing functions share Delta's critical points (Polya-Rolle isolation).
  RootList<T> crit;
  if (u_hi > u_lo) crit = critical_points(delta, u_lo, u_hi);
  const int64_t k_on = on ? first_firing(gon, crit, k_lo, k_hi) : -1;
  const int64_t k_off = off ? first_firing(goff, crit, k_lo, k_hi) : -1;
  if (k_on < 0 && k_off < 0) return kNone;
  if (k_off < 0 || (k_on >= 0 && k_on <= k_off)) return {true, true, e0 + k_on};
  return {true, false, e0 + k_off};
}

BALLA_HD inline int64_t level_length(int level) { return kNoiseCellNs >> level; }

// The lattice step of a cell with diffusion ratio rho != 1 (PHASE2 D-P2-2): Phi from the step block, the factor of
// rho Q1 + Q2 from the split tables. Out of line, by value.
template <class T>
BALLA_HD BALLA_SENSOR_COLD Triple<T> split_lattice_step(const KernelConstants<T>& kc, NoisePos<T> pos, T rho,
                                                       Triple<T> nk, Triple<T> xi) {
  T phi[6], q12[12], chol[6];
  noise_lookup(kc.tables, pos, NoiseLayout::kStepPhi, 6, phi);
  noise_lookup(kc.tables, pos, NoiseLayout{kc.levels}.split_step(), 12, q12);
  split_chol(q12, rho, chol);
  Triple<T> out;
  lattice_step(phi, chol, nk.v, xi.v, out.v);
  return out;
}

// The bridge of a cell with diffusion ratio rho != 1 (PHASE2 D-P2-2), formed from the split tables (noise.h). Out of
// line, so that the frames of the walk and of n3_at stay those of the tables' bridge.
// By value, so that the callers' arrays stay in registers (pointers would put them in local memory on the GPU).
template <class T>
BALLA_HD BALLA_SENSOR_COLD Triple<T> bridge_split(const KernelConstants<T>& kc, NoisePos<T> pos, T rho, int level,
                                                  Triple<T> xl, Triple<T> xr, Triple<T> xi) {
  T blk[18];
  noise_lookup(kc.tables, pos, NoiseLayout{kc.levels}.split_level(level), 18, blk);
  Triple<T> xm;
  bridge_midpoint_split(blk, rho, xl.v, xr.v, xi.v, xm.v);
  return xm;
}

// Midpoint of a tree node at `level` (its time t_mid) from its endpoints (M3 D.6), keyed by t_mid. A cell with
// diffusion ratio rho != 1 (PHASE2 D-P2-2) takes bridge_split, out of line. kRho false compiles that branch out (rho is
// then 1): the walk and n3_at of PixelKernel's kP2 = false instantiation (P2-E E4), one kernel each, unlike the two
// walks of one kernel that P2-A measured slower on the GPU's frozen path than the runtime branch (+4 % against +2 %).
template <class T, bool kRho = true>
BALLA_HD BALLA_SENSOR_INLINE void bridge_child(const KernelConstants<T>& kc, uint32_t pixel, NoisePos<T> pos, T rho,
                                               int level, int64_t t_mid, const T* xl, const T* xr, T* xm) {
  T xi[3], blk[24];
  normals3(kc.seed, rng::Purpose::kNoise, kc.camera, pixel, static_cast<uint64_t>(t_mid), xi);
  if (kRho && rho != T(1)) {
    const Triple<T> m = bridge_split(kc, pos, rho, level, Triple<T>{{xl[0], xl[1], xl[2]}},
                                     Triple<T>{{xr[0], xr[1], xr[2]}}, Triple<T>{{xi[0], xi[1], xi[2]}});
    for (int i = 0; i < 3; ++i) xm[i] = m.v[i];
    return;
  }
  noise_lookup(kc.tables, pos, NoiseLayout{kc.levels}.bridge(level), 24, blk);
  bridge_midpoint(blk, xl, xr, xi, xm);
}

// The exclusion margin m_l(e) = z sigma_br + g |W e| (M3 D.8) of a node in a cell with diffusion ratio rho (PHASE2
// D-P2-2). With the rho = 1 predictor held fixed, the residual variance is linear in the two noise intensities, so
// sigma_br(rho)^2 <= sigma_br(1)^2 (1 + (rho - 1) f_l), f_l = max over the node of the input node's share of it
// (split tables); for rho < 1 the factor is 1. The small mean-mismatch term scales with sqrt(max(1, rho)) (SENSOR.md
// "Transient diffusion"). m: the level's margin block; f_l from the split tables.
template <class T>
BALLA_HD BALLA_SENSOR_INLINE T noise_margin_rho(const T* m, T z, const T* xl, const T* xr, T rho, T f_l) {
  const T e[6] = {xl[0], xl[1], xl[2], xr[0], xr[1], xr[2]};
  const T* W = m + 2;
  T norm2 = T(0);
  int idx = 0;
  for (int i = 0; i < 6; ++i) {
    T s = T(0);
    for (int j = 0; j <= i; ++j) s = s + W[idx++] * e[j];
    norm2 = norm2 + s * s;
  }
  const T big = rho > T(1) ? rho : T(1);
  const T s_br = rho > T(1) ? sqrt(T(1) + (rho - T(1)) * f_l) : T(1);
  return z * m[0] * s_br + m[1] * sqrt(norm2) * sqrt(big);
}

// The transient diffusion of a cell (accuracy.noise_diffusion transient; PHASE2 D-P2-2). The LNA's n1 obeys
// dn1 = -k u n1 dt + sqrt(k^2 q (I + u)) dW; over a cell [ts, te] its variance gains k q J,
//   J = int e^{-2 int_s^te k u} k (I + u) ds,
// while the frozen generator (rate a1, the exact average of k u; diffusion D1) gains D1 (1 - e^{-2 a1 h}) / (2 a1). So
// D1 = rho 2 k q a1 with rho = J / (1 - e^{-2 a1 h}) gives n1 at the lattice nodes the LNA's law, whatever the
// transient (exactly on piecewise-constant input; ramps below). As a1 h -> 0 it is D-P2-2's cell average
// k q (a_I + a1); as a1 h grows the weight moves to the cell's end, and a cell whose transient settled before its end
// gets rho -> 1 (the frozen value). One input piece's step of J, accumulated over the look-ahead's pieces:
// J <- J e^{-2 dL} + J_piece, dL = int_piece k u = ln(z_b / z_a) + int k I. A constant piece in closed form
// (z = 1/I + (z_a - 1/I) e^{-k I s}, so the integrand is a sum of exponentials; r = z_a I - 1):
//   J_piece = [(1 - e^{-2P}) + 3 r (e^{-P} - e^{-2P}) + P r^2 e^{-2P}] / (1 + r e^{-P})^2,  P = k I h;
// a ramp split into n = min(32, ceil(int k (I + u) / kSubPiece)) sub-pieces (z at the sub-nodes by the closed form; a
// piece longer than the n1 memory resolves only its tail), each by diffusion_ramp (a whole cell as one piece with
// k (I + u) uniform erred by up to 38 %). Out of line: only transient cells under transient diffusion take it. h (s)
// and k: the piece's length and the photoreceptor constant (ramps only).
template <class T>
BALLA_HD T diffusion_ramp(T jw, T ikI, T dl, T dkI) {
  // J_piece = int e^{-2 (L_b - L(s))} k (I + u) ds, L = int k u, dl = L_b - L_a. The u part is exact on any path,
  // int e^{-2 (L_b - L)} dL = (1 - e^{-a}) / 2 = dl g0 (a = 2 dl, g0 = (1 - e^{-a}) / a); the input part, k I linear
  // with integral ikI and dkI = k h (I_b - I_a), takes the weight with L linear in time: ikI g0 + dkI (g1 - g0 / 2),
  // g1 - g0 / 2 = 1/a - g0 (1/a + 1/2) = a/12 - a^2/24 + a^3/80 - ... (exact when k u is constant over the piece; the
  // error is second order in dl).
  const T a = T(2) * dl;
  const T em = pmath::expm1(-a);
  T g0, c;
  if (a > T(0.05)) {
    g0 = -em / a;
    c = T(1) / a - g0 * (T(1) / a + T(0.5));
  } else {
    g0 = T(1) - a * (T(0.5) - a * (T(1) / T(6)));
    c = a * (T(1) / T(12) - a * (T(1) / T(24) - a * (T(1) / T(80))));
  }
  return jw * (T(1) + em) + (ikI + dl) * g0 + dkI * c;
}

inline constexpr double kSubPiece = 0.25;   // largest int k (I + u) of a ramp's sub-piece
inline constexpr double kTailMemory = 12.0;  // int k u from the cell's end beyond which weights are below e^{-24}

template <class T>
BALLA_HD BALLA_SENSOR_COLD T diffusion_weight(T jw, T z_a, T z_b, T I_a, T I_b, T ikI, T h, T k) {
  if (I_a == I_b) {
    const T em = pmath::expm1(-ikI);  // e^{-P} - 1
    const T e1 = T(1) + em, e2 = e1 * e1;
    const T r = z_a * I_a - T(1);
    const T d = T(1) + r * e1;
    const T zr = z_b / z_a;
    const T jp = ((-em) * (T(2) + em) + T(3) * r * (-em) * e1 + ikI * (r * r) * e2) / (d * d);
    return jw * (e2 / (zr * zr)) + jp;
  }
  T z = z_a, I0 = I_a, hr = h, ik = ikI;
  T dl = pmath::log(z_b / z_a) + ikI;
  if (dl > T(kTailMemory)) {
    // Only the piece's last kTailMemory units of int k u reach the cell's end with weight above e^{-2 kTailMemory}:
    // the head [0, h - tail], tail = kTailMemory / (k u_b), takes one rule, the tail is resolved.
    const T tail = T(kTailMemory) * z_b / k;
    if (tail < h) {
      const T hh = h - tail;
      const T Im = I_a + (I_b - I_a) * (hh / h);
      const T zm = z_step(z_a, I_a, Im, hh, k);
      const T ih = integrated_pole(I_a, Im, hh, k);
      jw = diffusion_ramp(jw, ih, pmath::log(zm / z_a) + ih, hh * k * (Im - I_a));
      z = zm;
      I0 = Im;
      hr = tail;
      ik = integrated_pole(I0, I_b, hr, k);
      dl = pmath::log(z_b / z) + ik;
    }
  }
  const T total = ik + dl;  // int k (I + u) over what is left
  int n = static_cast<int>(total * T(1.0 / kSubPiece)) + 1;
  n = n < 32 ? n : 32;
  if (n == 1) return diffusion_ramp(jw, ik, dl, hr * k * (I_b - I0));
  const T hs = hr / static_cast<T>(n), dI = (I_b - I0) / static_cast<T>(n), Is = I0;
  for (int i = 0; i < n; ++i) {
    const T I1 = i + 1 == n ? I_b : Is + dI * static_cast<T>(i + 1);
    const T z1 = i + 1 == n ? z_b : z_step(z, I0, I1, hs, k);
    const T ik1 = integrated_pole(I0, I1, hs, k);
    jw = diffusion_ramp(jw, ik1, pmath::log(z1 / z) + ik1, hs * k * (I1 - I0));
    z = z1;
    I0 = I1;
  }
  return jw;
}

// rho = J / (1 - e^{-2 a1 h}) of diffusion_weight's J, >= 1/2 (k (I + u) >= k u). Capped at 1e6 and rounded to float,
// so that rho - 1 is exact in float and a saved state (PixelState::rho_m1, 0 at rho = 1) resumes it bit for bit in
// either precision.
template <class T>
BALLA_HD BALLA_SENSOR_COLD T diffusion_ratio(T jw, T int_ku) {
  const T den = -pmath::expm1(T(-2) * int_ku);
  if (!(den > T(0)) || !is_finite(jw)) return T(1);
  const T rho = jw / den;
  return static_cast<T>(static_cast<float>(rho < T(1e6) ? (rho > T(0.5) ? rho : T(0.5)) : T(1e6)));
}

// The margin of a level-l node in a cell with rho != 1: out of line and by value (it looks its margin block up itself),
// so that the frozen path's arrays and code stay as they were.
template <class T>
BALLA_HD BALLA_SENSOR_COLD T margin_rho(const KernelConstants<T>& kc, NoisePos<T> pos, int level, T rho, Triple<T> xl,
                                        Triple<T> xr) {
  T m[23], f_l;
  noise_lookup(kc.tables, pos, NoiseLayout{kc.levels}.margin(level), 23, m);
  noise_lookup(kc.tables, pos, NoiseLayout{kc.levels}.split_margin(level), 1, &f_l);
  return noise_margin_rho(m, kc.margin_z, xl.v, xr.v, rho, f_l);
}

// n3(t) in cell tc0 (nodes nk, nk1, rate ln_a1, diffusion ratio rho; 1 when kRho is false) by descending the keyed
// tree to the leaf holding t; refinements counts the bridge draws.
template <class T, bool kRho = true>
BALLA_HD BALLA_SENSOR_COLD T n3_at(const KernelConstants<T>& kc, uint32_t pixel, T ln_a1, T rho, Triple<T> nk,
                                   Triple<T> nk1, int64_t tc0, int64_t t, uint32_t& refinements) {
  if (t == tc0) return nk.v[2];
  if (t == tc0 + kNoiseCellNs) return nk1.v[2];
  const NoisePos<T> pos = noise_pos(kc.tables, ln_a1);
  T xl[3], xr[3], xm[3];
  for (int i = 0; i < 3; ++i) {
    xl[i] = nk.v[i];
    xr[i] = nk1.v[i];
  }
  int64_t node_t = tc0;
  for (int l = 1; l <= kc.levels; ++l) {
    const int64_t mid = node_t + level_length(l);
    bridge_child<T, kRho>(kc, pixel, pos, rho, l, mid, xl, xr, xm);
    ++refinements;
    if (t == mid) return xm[2];
    if (t < mid) {
      for (int i = 0; i < 3; ++i) xr[i] = xm[i];
    } else {
      for (int i = 0; i < 3; ++i) xl[i] = xm[i];
      node_t = mid;
    }
  }
  T hc[4];
  noise_hermite(xl, xr, kc.cc.a3, static_cast<T>(level_length(kc.levels)) * T(1e-9), hc);
  const T u = static_cast<T>(t - node_t) * T(1e-9);
  return hc[0] + u * (hc[1] + u * (hc[2] + u * hc[3]));
}

// The first crossing of a sub-step on the leaves of its cell's bridge tree, and Delta's parts there.
template <class T>
struct Found {
  bool found, on;
  int64_t t;
  T s3, n3;  // s3 and n3 at t
  uint32_t refinements, leaves;
#if BALLA_SENSOR_CENSUS
  CensusDraws c;
#endif
};

// The high-pass filter's interval for the walk (hpf.h): the search function g from the anchor t_x (with the cell's
// level-0 noise Hermite hc0 in it), and the bound of its b-mode, added to every margin.
template <class T>
struct HpfCtx {
  ExpCubic<T> g;
  int64_t t_x, tc0;
  T hc0[4];
  T bound;
};

// Delta on [e0, ...) of a walk node with Hermite hc starting at node_t: delta_fn's, or under the high-pass filter
// (kHpf) the interval's search function at e0 with the node's noise in place of the level-0 Hermite's.
template <class T, bool kHpf>
BALLA_HD BALLA_SENSOR_INLINE ExpCubic<T> node_delta(const SubStep<T>& sub, int64_t e0, const T* hc, int64_t node_t,
                                                    const Comparand<T>& c, const HpfCtx<T>& h) {
  if constexpr (kHpf) {
    ExpCubic<T> g = e0 == h.t_x ? h.g : shifted(h.g, static_cast<T>(e0 - h.t_x) * T(1e-9));
    const T d = static_cast<T>(e0 - node_t) * T(1e-9), d0 = static_cast<T>(e0 - h.tc0) * T(1e-9);
    const T* z = h.hc0;
    g.c[0] = g.c[0] + ((hc[0] + d * (hc[1] + d * (hc[2] + d * hc[3]))) - (z[0] + d0 * (z[1] + d0 * (z[2] + d0 * z[3]))));
    g.c[1] = g.c[1] + ((hc[1] + d * (T(2) * hc[2] + d * (T(3) * hc[3]))) -
                       (z[1] + d0 * (T(2) * z[2] + d0 * (T(3) * z[3]))));
    g.c[2] = g.c[2] + ((hc[2] + d * (T(3) * hc[3])) - (z[2] + d0 * (T(3) * z[3])));
    g.c[3] = g.c[3] + (hc[3] - z[3]);
    return g;
  } else {
    (void)h;
    return delta_fn(sub, e0, hc, node_t, c);
  }
}

// The depth-first, time-ordered walk of the cell [tc0, tc0 + 2^16) (nodes nk, nk1, rate ln_a1, diffusion
// ratio rho) for the sub-step `sub` from lo (M3 §D.6), below a level-0 node that may fire (with leaf level 0, the
// cell itself is the leaf). Pure: it draws the bridges it needs and changes no state. Under the high-pass filter
// (kHpf) Delta is the interval's (h: node_delta) and every margin carries its b-mode bound. Out of line through walk
// and walk_hpf below, so that the filter-free walk keeps its signature and code. kRho false: rho is 1 (bridge_child).
template <class T, bool kHpf, bool kRho>
BALLA_HD BALLA_SENSOR_INLINE Found<T> walk_impl(const KernelConstants<T>& kc, uint32_t pixel, T ln_a1, T rho,
                                                Triple<T> nk, Triple<T> nk1, int64_t tc0, const SubStep<T>& sub,
                                                int64_t lo, const Comparand<T>& c, const HpfCtx<T>& h) {
  struct Node {
    int level;
    int64_t t;
    T xl[3], xr[3];
  };
#if BALLA_SENSOR_CENSUS
  Found<T> out{false, false, 0, T(0), T(0), 0, 0, CensusDraws{}};
#else
  Found<T> out{false, false, 0, T(0), T(0), 0, 0};
#endif
  const NoisePos<T> pos = noise_pos(kc.tables, ln_a1);
  Node stack[12];
  int sp = 0;
  Node& root = stack[sp++];
  root.level = 0;
  root.t = tc0;
  for (int i = 0; i < 3; ++i) {
    root.xl[i] = nk.v[i];
    root.xr[i] = nk1.v[i];
  }
  bool top = kc.levels > 0;  // the level-0 node is known to be non-excluded
  while (sp > 0) {
    const Node node = stack[--sp];
    const int64_t length = level_length(node.level);
    const int64_t span_lo = max(node.t, lo), span_hi = min(node.t + length, sub.t_b);
    if (span_lo >= span_hi) continue;
    if (node.level < kc.levels) {
      if (!top) {
        T hc[4];
        noise_hermite(node.xl, node.xr, kc.cc.a3, static_cast<T>(length) * T(1e-9), hc);
        T margin;
        if (kRho && rho != T(1)) {
          margin = margin_rho(kc, pos, node.level, rho, Triple<T>{{node.xl[0], node.xl[1], node.xl[2]}},
                              Triple<T>{{node.xr[0], node.xr[1], node.xr[2]}});
        } else {
          T mblk[23];
          noise_lookup(kc.tables, pos, NoiseLayout{kc.levels}.margin(node.level), 23, mblk);
          margin = noise_margin(mblk, kc.margin_z, node.xl, node.xr);
        }
        const int64_t e0 = max(sub.t_a, span_lo), e1 = min(sub.t_b, span_hi);
        if constexpr (kHpf) margin = margin + h.bound;
        if (e0 >= e1 || !may_cross(node_delta<T, kHpf>(sub, e0, hc, node.t, c, h), e0, e1, lo, margin, c)) continue;
      }
      top = false;
      const int64_t mid = node.t + level_length(node.level + 1);
      T xm[3];
      bridge_child<T, kRho>(kc, pixel, pos, rho, node.level + 1, mid, node.xl, node.xr, xm);
      ++out.refinements;
#if BALLA_SENSOR_CENSUS
      if (rho != T(1)) out.c.rho_levels |= 1u << (node.level + 1);
      if (kc.levels <= 6) out.c.nodes |= uint64_t{1} << ((mid - tc0) >> (kNoiseCellLog2 - kc.levels));
#endif
      Node& right = stack[sp++];
      right.level = node.level + 1;
      right.t = mid;
      Node& left = stack[sp++];
      left.level = node.level + 1;
      left.t = node.t;
      for (int i = 0; i < 3; ++i) {
        left.xl[i] = node.xl[i];
        left.xr[i] = xm[i];
        right.xl[i] = xm[i];
        right.xr[i] = node.xr[i];
      }
      continue;
    }
    T hc[4];
    noise_hermite(node.xl, node.xr, kc.cc.a3, static_cast<T>(length) * T(1e-9), hc);
    const int64_t e0 = max(sub.t_a, span_lo), e1 = min(sub.t_b, span_hi);
    if (e0 >= e1) continue;
    ++out.leaves;
    const Hit hit = search(node_delta<T, kHpf>(sub, e0, hc, node.t, c, h), e0, e1, lo, c.theta_on, c.theta_off);
    if (!hit.found) continue;
    const T u = static_cast<T>(hit.t - node.t) * T(1e-9);
    out.found = true;
    out.on = hit.on;
    out.t = hit.t;
    out.s3 = sub.s3(static_cast<T>(hit.t - sub.t_a) * T(1e-9));
    out.n3 = hc[0] + u * (hc[1] + u * (hc[2] + u * hc[3]));
    return out;
  }
  return out;
}

template <class T, bool kRho = true>
BALLA_HD BALLA_SENSOR_COLD Found<T> walk(const KernelConstants<T>& kc, uint32_t pixel, T ln_a1, T rho, Triple<T> nk,
                                          Triple<T> nk1, int64_t tc0, const SubStep<T> sub, int64_t lo,
                                          Comparand<T> c) {
  return walk_impl<T, false, kRho>(kc, pixel, ln_a1, rho, nk, nk1, tc0, sub, lo, c, HpfCtx<T>{});
}

// The walk of an interval under the high-pass filter: sub ends at the interval's end; h by value (registers).
template <class T, bool kRho = true>
BALLA_HD BALLA_SENSOR_COLD Found<T> walk_hpf(const KernelConstants<T>& kc, uint32_t pixel, T ln_a1, T rho,
                                              Triple<T> nk, Triple<T> nk1, int64_t tc0, const SubStep<T> sub,
                                              int64_t lo, Comparand<T> c, HpfCtx<T> h) {
  return walk_impl<T, true, kRho>(kc, pixel, ln_a1, rho, nk, nk1, tc0, sub, lo, c, h);
}

}  // namespace detail

// kCacheBlocks keeps the interpolated noise-table blocks of the last rate across cells (a settled
// pixel draws every cell at one rate); the GPU's general pass, where the rate changes every cell,
// re-interpolates instead of holding them. Either way the values are the same. kGlare false compiles the veiling
// glare out (kc.glare must then be off): the hot loops of glare-free calls keep their code (SensorModel dispatches).
// kHpf compiles the high-pass filter in (hpf.h; SensorModel dispatches it, with kGlare true, while kc.hpf is on): its
// state is the HpfState passed to the constructor; without it the kernel's code is as before the filter. kP2 false
// compiles Phase 2's per-cell paths out (P2-E E4): the transient diffusion ratio rho (kc.transient must then be off, so
// rho = 1 in every cell and PixelState::rho_m1 stays 0). The GPU's frozen kernels take it (sensor_cuda.cu); the CPU's
// and the filter's keep kP2 true, the same bits. Its tests read `kP2 && ...` (and kRho's `kRho && ...`), which the
// front end folds, so that the kP2 true kernels' PTX is that of before E4: accessors with `if constexpr` reordered
// their blocks (+1.5 % on transient replays).
template <class T, class Sink, bool kCacheBlocks = true, class Stats = KernelStats, bool kGlare = true,
          bool kHpf = false, bool kP2 = true>
class PixelKernel {
 public:
  // live: the call's live-group mask when already known (live_mask() of an earlier kernel object on
  // the same call), or negative to scan the input for it. hpf: the pixel's filter state (kHpf only).
  BALLA_HD PixelKernel(const KernelConstants<T>& kc, const PixelMap& map, uint32_t pixel, PixelState& ps,
                       int64_t t_state, const InputRun& in, Sink& sink, Stats& stats, int32_t live = -1,
                       HpfState* hpf = nullptr)
      : kc_(kc), map_(map), pixel_(kc.pixel_ids != nullptr ? kc.pixel_ids[pixel] : pixel), ps_(ps), t_state_(t_state),
        in_(in), sink_(sink), stats_(stats), hs_(hpf) {
    theta_on_ = static_cast<T>(map.theta_on);
    theta_off_ = static_cast<T>(map.theta_off);
    leak_ = static_cast<T>(map.leak);
    dark_ = static_cast<T>(map.dark_fA);
    scale_ = kc.scale * static_cast<T>(map.gain);  // the pixel's photocurrent per renderer unit (gain 1: kc.scale)
    if constexpr (kHpf) b_ = kc.hpf_b_map != nullptr ? static_cast<T>(kc.hpf_b_map[pixel]) : kc.hpf_b;
    skip_ = in_.n > 0 && in_.t[0] == t_state_ ? 1 : 0;
    load();
    if (live < 0) {
      scan_input();
    } else {
      live_ = static_cast<uint32_t>(live);
      wave_live_ = (live_ & kc_.wave_mask) != 0;
    }
  }

  BALLA_HD uint32_t live_mask() const { return live_; }

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
      mean_I = mean_I + scale_ * (glare_on() ? static_cast<T>(v) + glare_at(t0, g) : static_cast<T>(v));
    }
    scan_input();
#if BALLA_SENSOR_CENSUS
    if constexpr (kCensus) census_init(t0);
#endif
    i_scale_ = mean_I;
    st_.z = T(1) / mean_I;
    x1_ = T(0);
    st_.s2 = st_.s3 = x1_;
    tv_ = T(0);
    if (kc_.instant) {
      // The textbook DVS's reference is ln I(t0) itself (the waveforms at t0 included), offset by phi0.
      set_reference(t0, log_ratio(photocurrent(0, t0)), T(0), static_cast<T>(map_.phi0));
      blind_until_ = kNotBlind;
      t_search_ = t0 + 1;
      flags_ = kFlagFirstAfterInit;
      store();
      return;
    }
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
        CellMemo m;
        enter_cell(t0, kc_.cc.k * mean_I * static_cast<T>(t0 - tk) * T(1e-9), m);
        uint32_t refinements = 0;
        n3_0 = n3_here(t0, refinements);
        stats_.refinements += refinements;
      } else {
        n3_0 = nk_[2];
      }
    }
    set_reference(t0, st_.s3, n3_0, static_cast<T>(map_.phi0));
    blind_until_ = kNotBlind;
    t_search_ = t0 + 1;
    flags_ = kFlagFirstAfterInit;
    store();
  }

  // A state that ran its own history up to the model time t0 (the stationary probe) starts as init's do (F034):
  // FIRST_AFTER_INIT, the history's flags dropped, and the reference re-based on its own s3 and n3 at t0 with
  // Delta(t0) as its offset (unchanged Delta: under the high-pass filter less its memory, which the new reference
  // takes in and sets to 0). A blind state keeps its reference: the release sets an own one.
  BALLA_HD void rebase_history() {
    flags_ = kFlagFirstAfterInit;
    if (blind_until_ == kNotBlind) {
      uint32_t refinements = 0;  // not counted: the history's statistics are dropped
      const T n3 = kc_.noise ? n3_here(t_state_, refinements) : T(0);
      const T dl = leak_ * static_cast<T>(static_cast<double>(t_state_ - t_ref_) * 1e-9) + dr_ref_;
      T delta = (st_.s3 - s3_ref_) + (n3 - n3_ref_) + dl;
      if constexpr (kHpf) delta = delta - static_cast<T>(hm_);
      set_reference(t_state_, st_.s3, n3, delta);
    }
    store();
  }

  // Advances the call that started at the model time (constructor) to t_end.
  BALLA_HD void advance(int64_t t_end) {
    if (kc_.instant) advance_instant(t_state_, t_end, true);
    else advance_slice(t_state_, t_end, true);
  }

  // The instantaneous log response (D3, the textbook DVS; kc.instant): Delta(t) = ln I(t) - the reference, with the
  // photocurrent I(t) = I_dk + sum_g m_g(t) s I_g(t) of the input model (I_g linear between samples, held after the
  // last; the waveforms applied) and the leak and reset offsets of the cascade path; no cascade and no noise. Events
  // fire at the first integer ns where the closed condition holds, as on the cascade path, found by a time-ordered
  // bisection that drops a node where bounds of I keep Delta inside the levels (instant_may_fire). Like advance_slice
  // it may run as consecutive slices cut at multiples of 2^23 ns (pieces never cross them).
  BALLA_HD void advance_instant(int64_t t_from, int64_t t_to, bool ends_call) {
    int seg = segment_at(t_from);
    int64_t t = t_from;
    while (t < t_to) {
      seg = segment_of(seg, t);
      int64_t e = min(min(t_to, segment_end(seg)), ((t >> kPlainSegmentLog2) + 1) * kPlainSegmentNs);
      if (kc_.breakpoints) {
        for (int g = 0; g < kc_.groups; ++g) {
          if (live(g)) e = min(e, next_breakpoint(g, t));
        }
      }
      ++stats_.substeps;
      ++stats_.pieces;
      while (true) {
        if (blind_until_ != kNotBlind) {  // physical refractory: the release at blind_until_ (M3 A.16)
          if (blind_until_ >= e) break;
          set_reference(blind_until_, log_ratio(photocurrent(seg, blind_until_)), T(0), -kc_.eps_ci);
          blind_until_ = kNotBlind;
        }
        const int64_t lo = max(t_search_, t);
        if (lo >= e) break;
        const InstantHit hit = instant_search(seg, lo, e);
        if (!hit.found) {
          t_search_ = max(t_search_, e);
          break;
        }
        fire(hit.t, hit.on, hit.x, T(0));
      }
      t = e;
    }
    finish(ends_call);
  }

  // One time slice [t_from, t_to) of the call that started at the model time with the samples `in`.
  // A call may be run as consecutive slices, each by a fresh kernel object on the same state, cut
  // at multiples of slice_grain(kc.noise) or where advance_quiet stopped: the result is bitwise that
  // of advance(t_end). ends_call marks the last slice, after which the last sample is held.
  BALLA_HD BALLA_SENSOR_INLINE void advance_slice(int64_t t_from, int64_t t_to, bool ends_call) {
    seg_ = segment_at(t_from);
    int64_t t = t_from;
    while (t < t_to) {
      CellMemo m;
#if BALLA_SENSOR_CENSUS
      CensusClass census{};
      if constexpr (kCensus) census = census_classify(t, t_to);
#endif
      if (kc_.noise && kc_.static_fast_path && !have_k1_ && t + kNoiseCellNs <= t_to && quiet_cell(t, m)) {
#if BALLA_SENSOR_CENSUS
        if constexpr (kCensus) census_count(census, true, t);
#endif
        t += kNoiseCellNs;
        continue;
      }
#if BALLA_SENSOR_CENSUS
      if constexpr (kCensus) census_count(census, false, t);
#endif
      int64_t boundary;
      if (kc_.noise) {
        if (!have_k1_) enter_cell(t, T(0), m);
        boundary = (cell_ + 1) * kNoiseCellNs;
      } else {
        boundary = ((t >> kPlainSegmentLog2) + 1) * kPlainSegmentNs;  // floor division (arithmetic shift)
      }
      const int64_t seg_end = min(t_to, boundary);
      run_segment(t, seg_end, m);
      if (kc_.noise && seg_end == boundary) next_cell();
      t = seg_end;
    }
    finish(ends_call);
  }

  // The quiet pass of a slice: whole quiet cells (quiet_cell) from t_from while they last. Returns
  // the time reached, where the general pass (advance_slice from there) resumes the slice; a pixel
  // that reaches t_to has finished it. A quiet cell never fires, so the pass emits no events.
  BALLA_HD BALLA_SENSOR_INLINE int64_t advance_quiet(int64_t t_from, int64_t t_to, bool ends_call) {
    seg_ = segment_at(t_from);
    int64_t t = t_from;
    if (kc_.noise && kc_.static_fast_path) {
      while (t < t_to && !have_k1_ && t + kNoiseCellNs <= t_to) {
        CellMemo m;
        if (!quiet_cell(t, m)) break;
        t += kNoiseCellNs;
      }
    }
    if (t == t_to) {
      finish(ends_call);
    } else {
      store();
    }
    return t;
  }

  // False when no cell of the call can be quiet: some live group is modulated (cheap test, for the
  // GPU's quiet pass to drop such pixels at once).
  BALLA_HD bool may_be_quiet() const { return kc_.noise && kc_.static_fast_path && !wave_live_; }

 private:
  // ------------------------------------------------------------------------------------------
  // State load / store
  // ------------------------------------------------------------------------------------------
  BALLA_HD BALLA_SENSOR_INLINE void load() {
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
    if constexpr (kP2) rho_ = T(1) + static_cast<T>(ps_.rho_m1);  // exact: rho_ is a float value (diffusion_ratio)
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
    if constexpr (kHpf) {
      hm_ = hs_->m;
      hmn_ = hs_->mn;
    }
    if (kc_.noise && have_k1_) {
      // A call (or slice) resumed inside a cell: its level-0 hull, as draw_lattice left it.
      if (kc_.levels > 0) {
        T m0[23];
        noise_lookup(kc_.tables, noise_pos(kc_.tables, ln_a1_), NoiseLayout{kc_.levels}.margin(0), 23, m0);
        level0(m0);
      } else {
        level0(nullptr);
      }
    }
  }

  BALLA_HD BALLA_SENSOR_INLINE void store() {
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
    ps_.rho_m1 = kP2 ? static_cast<float>(rho_ - T(1)) : 0.0f;
    ps_.cell = cell_;
    ps_.have_k1 = have_k1_ ? 1 : 0;
    ps_.t_ref = t_ref_;
    ps_.s3_ref = static_cast<double>(s3_ref_);
    ps_.n3_ref = static_cast<double>(n3_ref_);
    ps_.dr_ref = static_cast<double>(dr_ref_);
    ps_.blind_until = blind_until_;
    ps_.t_search = t_search_;
    ps_.flags = flags_;
    if constexpr (kHpf) {
      hs_->m = hm_;
      hs_->mn = hmn_;
    }
  }

  // The end of a slice: after the call's last slice the last sample is held until the next call.
  BALLA_HD BALLA_SENSOR_INLINE void finish(bool ends_call) {
#if BALLA_SENSOR_CENSUS
    if constexpr (kCensus) {
      if (ends_call) census_end_call();
    }
#endif
    if (ends_call && in_.n > 0) {
      for (int g = 0; g < kc_.groups; ++g) ps_.held[g] = sample_value(n_aug() - 1, g);
    }
    store();
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

  // Live groups: those with a non-zero value somewhere in the call's augmented samples. A dead group
  // contributes m * scale * (+-0) = +-0 to every sum, which leaves it bit for bit unchanged.
  BALLA_HD BALLA_SENSOR_INLINE void scan_input() {
    const int G = kc_.groups;
    const uint32_t all = (1u << G) - 1u;
    uint32_t live = 0;
    if (skip_ == 0) {
      for (int g = 0; g < G; ++g) {
        if (ps_.held[g] != 0.0f) live |= 1u << g;
      }
    }
    for (int i = 0; i < in_.n && live != all; ++i) {
      for (int g = 0; g < G; ++g) {
        if (clamp_irradiance(in_.irr[static_cast<int64_t>(i) * G + g], kc_.irr_max) != 0.0f) live |= 1u << g;
      }
    }
    if (glare_on()) {  // glare lights every pixel of a group the frame sees
      for (int g = 0; g < G; ++g) {
        if (kc_.glare.a[g] != T(0) || kc_.glare.b[g] != T(0)) live |= 1u << g;
      }
    }
    live_ = live;
    wave_live_ = (live & kc_.wave_mask) != 0;
  }

  BALLA_HD bool live(int g) const { return ((live_ >> g) & 1u) != 0; }

  // The segment containing t, by bisection (a slice may start late in a long run; segment_of's walk
  // from any earlier segment ends at the same one).
  BALLA_HD int segment_at(int64_t t) const {
    int lo = 0, hi = n_aug() - 1;  // the last segment with sample_time(seg) <= t (0 if none)
    while (lo < hi) {
      const int mid = (lo + hi + 1) / 2;
      if (sample_time(mid) <= t) lo = mid;
      else hi = mid - 1;
    }
    return lo;
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

  // Irradiance of group g at t in segment seg (linear between samples, held after the last), plus the call's glare.
  BALLA_HD T group_value(int seg, int64_t t, int g) const {
    T v;
    if (seg + 1 >= n_aug()) {
      v = static_cast<T>(sample_value(seg, g));
    } else {
      const int64_t t0 = sample_time(seg), t1 = sample_time(seg + 1);
      const T f = static_cast<T>(t - t0) / static_cast<T>(t1 - t0);
      const T v0 = static_cast<T>(sample_value(seg, g)), v1 = static_cast<T>(sample_value(seg + 1, g));
      v = v0 + (v1 - v0) * f;
    }
    return glare_on() ? v + glare_at(t, g) : v;
  }

  BALLA_HD bool glare_on() const {
    if constexpr (kGlare) return kc_.glare.on;
    return false;
  }
  BALLA_HD T glare_at(int64_t t, int g) const { return detail::glare_value(kc_.glare, t, g, map_.gain); }

  // Waveform evaluations through the call's table where it holds them (bitwise the direct value).
  BALLA_HD int64_t grid_index(int64_t t) const {
    const WaveTable<T>& w = kc_.table;
    const int64_t d = t - w.t0;
    return w.n_grid > 0 && (d & (kWaveGridNs - 1)) == 0 && d >= 0 && (d >> kWaveGridLog2) <= w.n_grid
               ? d >> kWaveGridLog2
               : -1;
  }

  BALLA_HD T wave_value(int g, int64_t t) const {
    const int64_t j = grid_index(t);
    if (j >= 0) return kc_.table.value[static_cast<int64_t>(g) * (kc_.table.n_grid + 1) + j];
    return detail::wave_value_direct<T>(kc_.waves[g], t);
  }

  BALLA_HD WaveBounds<T> wave_bounds(int g, int64_t t_a, int64_t t_b) const {
    const WaveTable<T>& w = kc_.table;
    const int64_t c = (t_a >> kNoiseCellLog2) - w.k0;
    if (w.n_cells > 0 && (t_a & (kNoiseCellNs - 1)) == 0 && t_b - t_a == kNoiseCellNs && c >= 0 && c < w.n_cells) {
      const T* b = w.bounds + (static_cast<int64_t>(g) * w.n_cells + c) * 3;
      return {b[0], b[1], b[2]};
    }
    return detail::wave_bounds_direct<T>(kc_.waves[g], t_a, t_b);
  }

  // The bounds of next_piece's chord test. In fp32 a candidate piece inside one noise cell takes the bounds of its
  // whole cell (ENGINE_CPP I6', a numerical tolerance item; Fourier groups): they hold on every sub-interval and are
  // looser, so pieces can only get shorter and the chord bound still holds (results move within eps_PL). It spares
  // the direct fp64 phase reduction of pieces cut at samples inside cells. The cell's bounds are the table's where it
  // holds the cell, else waveform_bounds over the cell (the function that fills the table), so CPU fp32 and GPU agree
  // bit for bit. Pulse bounds are cheap and stay exact; the fp64 kernel (the oracle) keeps the piece's own bounds.
  BALLA_HD WaveBounds<T> piece_wave_bounds(int g, int64_t t_a, int64_t t_b) const {
    if (sizeof(T) == sizeof(float) && kc_.waves[g].type == WaveformType::kFourier) {
      const int64_t k = t_a >> kNoiseCellLog2;  // floor division (arithmetic shift)
      if (((t_b - 1) >> kNoiseCellLog2) == k) return wave_bounds(g, k * kNoiseCellNs, (k + 1) * kNoiseCellNs);
    }
    return wave_bounds(g, t_a, t_b);
  }

  // The first breakpoint after t (tabulated at the grid points themselves only: snapped pulse corners
  // need not agree to the ns across a period boundary, so a later t is evaluated directly).
  BALLA_HD int64_t next_breakpoint(int g, int64_t t) const {
    if (kc_.waves[g].type != WaveformType::kPulse) return kNoBreakpoint;  // as waveform_next_breakpoint
    const int64_t j = grid_index(t);
    if (j >= 0) return kc_.table.breakpoint[static_cast<int64_t>(g) * (kc_.table.n_grid + 1) + j];
    return detail::next_breakpoint_direct(kc_.waves[g], t);
  }

  // The longest linear piece [t, t_b], t_b <= limit, within one sample segment and between waveform
  // breakpoints, whose chord meets |ln I - ln chord| <= eps_pl (M3 B.18-B.19): dyadic halving of
  // the candidate until h^2 max|I''| <= 8 eps_pl I_min. Dead groups are skipped (they add 0 to
  // I_min, I'' and I); without a live modulated group the chord test holds at once (I'' = 0).
  BALLA_HD BALLA_SENSOR_INLINE LinearPiece<T> next_piece(int& seg, int64_t t, int64_t limit, T eps_pl, bool& fallback) const {
    seg = segment_of(seg, t);
    int64_t tb = min(limit, segment_end(seg));
    if (kc_.modulated) {
      if (kc_.breakpoints) {
        for (int g = 0; g < kc_.groups; ++g) tb = min(tb, next_breakpoint(g, t));
      }
      while (wave_live_) {
        const T h = static_cast<T>(tb - t) * T(1e-9);
        T i_min = dark_, i2 = T(0);
        for (int g = 0; g < kc_.groups; ++g) {
          if (!live(g)) continue;
          const T va = group_value(seg, t, g), vb = group_value(seg, tb, g);
          const T lo = min(va, vb), hi = max(va, vb);
          if (kc_.waves[g].type == WaveformType::kConstant) {
            i_min = i_min + scale_ * lo;
            continue;
          }
          const WaveBounds<T> b = piece_wave_bounds(g, t, tb);
          i_min = i_min + scale_ * max(b.m_min, T(0)) * lo;
          i2 = i2 + scale_ * (b.d2m_max * hi + T(2) * b.dm_max * abs(vb - va) / h);
        }
        if (h * h * i2 <= T(8) * eps_pl * i_min) break;
        if (tb - t <= 1) {
          fallback = true;
          break;
        }
        tb = t + (tb - t) / 2;
      }
    }
    T I_a = dark_, I_b = dark_;
    for (int g = 0; g < kc_.groups; ++g) {
      if (!live(g)) continue;
      const T va = group_value(seg, t, g), vb = group_value(seg, tb, g);
      const bool flat = kc_.waves[g].type == WaveformType::kConstant;
      const T ma = flat ? T(1) : wave_value(g, t), mb = flat ? T(1) : wave_value(g, tb);
      I_a = I_a + ma * scale_ * va;
      I_b = I_b + mb * scale_ * vb;
    }
    return {t, tb, I_a, I_b};
  }

  BALLA_HD T eps_pl() const { return kc_.eps_pl_base / (T(1) + tv_); }

  // What one iteration of the cell loop computes once and reuses: quiet_cell, enter_cell and
  // run_segment all start with the piece from the cell's start (next_piece is a pure function of t,
  // the limit and eps_pl, and eps_pl does not change before the first sub-step), and the look-ahead's
  // z over the first piece is the first sub-step's z_b when that spans the piece.
  struct CellMemo {
    bool valid = false;
    bool fallback = false;
    int seg = 0;
    int64_t limit = 0;
    T eps = T(0);
    LinearPiece<T> pc{};
    ZMemo<T> z{};
  };

  BALLA_HD BALLA_SENSOR_INLINE LinearPiece<T> first_piece(int& seg, int64_t t, int64_t limit, bool& fallback,
                                                         CellMemo& m) const {
    const T eps = eps_pl();
    if (m.valid && m.pc.t_a == t && m.limit == limit && m.eps == eps) {
      seg = m.seg;
      if (m.fallback) fallback = true;
      return m.pc;
    }
    bool fb = false;
    const LinearPiece<T> pc = next_piece(seg, t, limit, eps, fb);
    if (fb) fallback = true;
    m.valid = true;
    m.fallback = fb;
    m.seg = seg;
    m.limit = limit;
    m.eps = eps;
    m.pc = pc;
    return pc;
  }

  // ------------------------------------------------------------------------------------------
  // Noise lattice
  // ------------------------------------------------------------------------------------------
  // Frozen rate of the cell containing t from a z-only look-ahead to its end (input held beyond
  // the last sample), then the lattice draw of n(t_{k+1}). `pre` is int k u before t (init only).
  BALLA_HD BALLA_SENSOR_INLINE void enter_cell(int64_t t, T pre, CellMemo& m) {
    cell_ = t >> kNoiseCellLog2;  // floor division by the cell length (arithmetic shift)
    const int64_t tc1 = (cell_ + 1) * kNoiseCellNs;
    int seg = seg_;
    bool fb = false;
    LinearPiece<T> pc = first_piece(seg, t, tc1, fb, m);
    if (pc.t_b == tc1 && pc.I_a == pc.I_b && cascade_settled(kc_.cc, st_, x1_, pc.I_a)) {
      settled_rate(t, pre, tc1);
    } else {
      const T h_cell = static_cast<T>(kNoiseCellNs) * T(1e-9);
      T z = st_.z, int_kI = T(0);
      // Transient diffusion: J = int over the cell of e^{-2 int_s^te k u} k (I + u) ds. Before t0 (init) the pixel is
      // settled (k u = k I), which contributes 1 - e^{-2 pre}.
      T jw = T(0);
      if (kP2 && kc_.transient && pre > T(0)) {
        jw = detail::diffusion_weight(jw, T(1), T(1), T(1), T(1), pre, T(0), T(0));
      }
#if BALLA_SENSOR_CENSUS
      if constexpr (kCensus) {
        if (kP2 && kc_.transient && pre > T(0)) ++stats_.census.n[kCnDwConst];
      }
#endif
      while (true) {
        const T h = static_cast<T>(pc.t_b - pc.t_a) * T(1e-9);
        const T z_a = z;
        if (!(pc.I_a == pc.I_b && abs(z * pc.I_a - T(1)) <= kc_.cc.eps_drop)) {
          z = z_step(z, pc.I_a, pc.I_b, h, kc_.cc.k);
          if (pc.t_a == t) m.z = {z_a, pc.I_a, pc.I_b, h, z};  // the main pass's first sub-step starts the same way
        }
        const T ikI = integrated_pole(pc.I_a, pc.I_b, h, kc_.cc.k);
        int_kI = int_kI + ikI;
        if (kP2 && kc_.transient) jw = detail::diffusion_weight(jw, z_a, z, pc.I_a, pc.I_b, ikI, h, kc_.cc.k);
#if BALLA_SENSOR_CENSUS
        if constexpr (kCensus) {
          if (kP2 && kc_.transient) ++stats_.census.n[pc.I_a == pc.I_b ? kCnDwConst : kCnDwRamp];
        }
#endif
        if (pc.t_b >= tc1) break;
        pc = next_piece(seg, pc.t_b, tc1, eps_pl(), fb);
      }
      const T int_ku = pre + pmath::log(z / st_.z) + int_kI;  // M3 D.1 via A.6
      ln_a1_ = pmath::log(max(int_ku / h_cell, T(1e-30)));
      rho_ = kP2 && kc_.transient ? detail::diffusion_ratio(jw, int_ku) : T(1);
    }
    draw_lattice(tc1);
  }

  // Settled under constant input for the whole cell: a1 = k u = k / z, as in the main pass.
  BALLA_HD BALLA_SENSOR_INLINE void settled_rate(int64_t t, T pre, int64_t tc1) {
    rho_ = T(1);  // settled: u = I, so the transient diffusion is the frozen one
    if (pre == T(0) && st_.z == settled_z_) {
      ln_a1_ = settled_ln_a1_;
      return;
    }
    const T h_cell = static_cast<T>(kNoiseCellNs) * T(1e-9);
    const T int_ku = pre + kc_.cc.k * static_cast<T>(tc1 - t) * T(1e-9) / st_.z;
    ln_a1_ = pmath::log(int_ku / h_cell);
    if (pre == T(0)) {
      settled_z_ = st_.z;
      settled_ln_a1_ = ln_a1_;
    }
  }

  // The lattice draw of n(t_{k+1}) at the cell's rate, and the cell's level-0 hull.
  BALLA_HD BALLA_SENSOR_INLINE void draw_lattice(int64_t tc1) {
    T xi[3];
    normals3(kc_.seed, rng::Purpose::kNoise, kc_.camera, pixel_, static_cast<uint64_t>(tc1), xi);
    const NoisePos<T> pos = noise_pos(kc_.tables, ln_a1_);
    const int stride_m0 = NoiseLayout{kc_.levels}.margin(0);
    if constexpr (kCacheBlocks) {
      if (!cache_.valid || cache_.ln_a1 != ln_a1_) {
        cache_.ln_a1 = ln_a1_;
        noise_lookup(kc_.tables, pos, NoiseLayout::kStepPhi, 12, cache_.step);
        if (kc_.levels > 0) noise_lookup(kc_.tables, pos, stride_m0, 23, cache_.margin0);
        cache_.valid = true;
      }
      if (kP2 && rho_ != T(1)) {
        split_step(pos, xi);
      } else {
        lattice_step(cache_.step, cache_.step + 6, nk_, xi, nk1_);
      }
      level0(cache_.margin0);
    } else {
      T step[12];
      noise_lookup(kc_.tables, pos, NoiseLayout::kStepPhi, 12, step);
      if (kP2 && rho_ != T(1)) {
        split_step(pos, xi);
      } else {
        lattice_step(step, step + 6, nk_, xi, nk1_);
      }
      if (kc_.levels > 0) {
        T m0[23];
        noise_lookup(kc_.tables, pos, stride_m0, 23, m0);
        level0(m0);
      } else {
        level0(nullptr);
      }
    }
    have_k1_ = true;
    ++stats_.cells;
  }

  // The lattice draw of a cell with diffusion ratio rho != 1 (PHASE2 D-P2-2), by a free function out of line: a call
  // to a member function would put the kernel object in local memory on the GPU (the frozen rule's hot path).
  BALLA_HD BALLA_SENSOR_INLINE void split_step(NoisePos<T> pos, const T* xi) {
#if BALLA_SENSOR_CENSUS
    if constexpr (kCensus) ++stats_.census.n[kCnCellsRho];
#endif
    const detail::Triple<T> x =
        detail::split_lattice_step(kc_, pos, rho_, triple(nk_), detail::Triple<T>{{xi[0], xi[1], xi[2]}});
    for (int i = 0; i < 3; ++i) nk1_[i] = x.v[i];
  }

  // Hermite of n3 over the whole cell and its level-0 exclusion margin (m0: the margin block).
  BALLA_HD BALLA_SENSOR_INLINE void level0(const T* m0) {
    noise_hermite(nk_, nk1_, kc_.cc.a3, static_cast<T>(kNoiseCellNs) * T(1e-9), hc0_);
    if (kc_.levels > 0) {
      if (kP2 && rho_ != T(1)) {
        margin0_ = detail::margin_rho(kc_, noise_pos(kc_.tables, ln_a1_), 0, rho_, triple(nk_), triple(nk1_));
      } else {
        margin0_ = noise_margin(m0, kc_.margin_z, nk_, nk1_);
      }
    }
  }

  BALLA_HD void next_cell() {
    for (int i = 0; i < 3; ++i) nk_[i] = nk1_[i];
    have_k1_ = false;
    ++cell_;
  }

  // Fast path for one whole cell starting at t, covered by a single constant input piece, of a
  // settled pixel whose live groups are constant: the common case of static or dark pixels and of
  // pixels between edges. It performs exactly the operations of the general path (enter_cell's
  // settled branch, the constant sub-step, the tree walk) without its bookkeeping, so results are
  // bitwise identical (tests/sensor/test_pixel.cpp, test_lean.cpp). Returns false when the cell needs
  // the general path (also when its walk finds a crossing, which the general path then fires); the
  // cell's lattice draw may then already be done.
  BALLA_HD BALLA_SENSOR_INLINE bool quiet_cell(int64_t t, CellMemo& m) {
    if (wave_live_ || blind_until_ != kNotBlind || kc_.levels == 0) return false;
    const int64_t tc1 = t + kNoiseCellNs;
    bool fallback = false;
    const LinearPiece<T> pc = first_piece(seg_, t, tc1, fallback, m);
    if (pc.t_b != tc1 || pc.I_a != pc.I_b || !cascade_settled(kc_.cc, st_, x1_, pc.I_a)) return false;
    cell_ = t >> kNoiseCellLog2;
    settled_rate(t, T(0), tc1);
    draw_lattice(tc1);
    SubStep<T> sub;
    sub.t_a = t;
    sub.t_b = tc1;
    sub.s3 = ExpCubic<T>{{st_.s3, T(0), T(0), T(0)}, {T(0), T(0)}, {kc_.cc.a2, kc_.cc.a3}, 0};
    sub.int_ku = kc_.cc.k * (static_cast<T>(tc1 - t) * T(1e-9)) / st_.z;
    if constexpr (kHpf) {
      // The general path's scan of this sub-step (comparator_hpf's first interval), declined where it would split.
      int n_split = 0;
      const HpfScan s = hpf_scan(sub, t, n_split, true);
      if (s.declined || s.found) return false;
      hm_ = s.m;
      hmn_ = s.mn;
    } else {
      const int64_t lo = max(t_search_, t);
      // Where the level-0 hull cannot exclude a crossing, the walk below it decides (it fires nothing):
      // with no crossing the cell is quiet; otherwise the general path redoes the same walk and fires.
      if (lo < tc1) {
        const detail::Comparand<T> c = comparand();
        if (detail::may_cross(detail::delta_fn(sub, lo, hc0_, t, c), lo, tc1, lo, margin0_, c)) {
          const detail::Found<T> f = walk_here(t, sub, lo, c);
          stats_.refinements += f.refinements;
          stats_.leaf_intervals += f.leaves;
          if (f.found) return false;
        }
      }
    }
    ++stats_.substeps;
    ++stats_.settled_substeps;
    ++stats_.pieces;
    if (tv_ != T(0)) tv_ = tv_ * pmath::exp(-sub.int_ku);
    t_search_ = max(t_search_, tc1);
    next_cell();
    return true;
  }

  // ------------------------------------------------------------------------------------------
  // Deterministic pass
  // ------------------------------------------------------------------------------------------
  BALLA_HD BALLA_SENSOR_INLINE void run_segment(int64_t t0, int64_t t1, CellMemo& m) {
    int64_t t = t0;
    while (t < t1) {
      bool piece_fallback = false;
      const LinearPiece<T> pc = t == t0 ? first_piece(seg_, t, t1, piece_fallback, m) : next_piece(seg_, t, t1, eps_pl(), piece_fallback);
      if (piece_fallback) flags_ |= kFlagSolverFallback;
      ++stats_.pieces;
      int64_t ta = pc.t_a;
      int64_t h_hint = 0;
      while (ta < pc.t_b) {
        SubStep<T> sub;
        ta = make_substep(pc, ta, h_hint, sub, m.z);
        comparator(sub);
      }
      t = pc.t_b;
    }
  }

  // One accepted sub-step starting at ta inside piece pc; returns its end.
  BALLA_HD BALLA_SENSOR_INLINE int64_t make_substep(const LinearPiece<T>& pc, int64_t ta, int64_t& h_hint, SubStep<T>& sub,
                                                    const ZMemo<T>& zm) {
    CascadeState<T> next{};
    SubStepInfo info{};
    const int64_t tb = next_substep(kc_.cc, st_, i_scale_, x1_, pc, ta, pc.at(ta), h_hint, sub, next, info, zm);
    ++stats_.substeps;
    stats_.substep_attempts += static_cast<uint32_t>(info.attempts);
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
    if constexpr (kHpf) {
      hm_ = 0.0;  // the filter's memory restarts with every reference (hpf.h)
      hmn_ = 0.0;
    }
  }

  // Emits the event at t with polarity `on` and updates the reference (M3 A.16-A.17, E.1-E.5). Under the high-pass
  // filter m and mn are its memory and the noise's part of it at t (hpf.h): each increment loses its own part, the
  // leak's and the offset's in closed form, the signal's the rest, so that they still sum to Delta(t).
  BALLA_HD void fire(int64_t t, bool on, T s3, T n3, T m = T(0), T mn = T(0)) {
#if BALLA_SENSOR_CENSUS
    if constexpr (kCensus) {
      if ((t >> kNoiseCellLog2) == c_elig_cell_ && !c_elig_fired_) {
        ++stats_.census.n[kCnEligibleFired];
        c_elig_fired_ = true;
      }
    }
#endif
    T ds = s3 - s3_ref_;
    T dn = n3 - n3_ref_;
    T l = leak_ * static_cast<T>(static_cast<double>(t - t_ref_) * 1e-9);
    T dl = l + dr_ref_;
    if constexpr (kHpf) {
      T ml_leak;
      const T ml = detail::hpf_leak_memory(leak_, dr_ref_, b_, static_cast<T>(static_cast<double>(t - t_ref_) * 1e-9),
                                           ml_leak);
      ds = ds - ((m - mn) - ml);
      dn = dn - mn;
      dl = dl - ml;
      l = l - ml_leak;  // the first event's leak since t0 (its offset's decay stays out with the offset)
    } else {
      (void)m;
      (void)mn;
    }
    const T D = ds + dn + dl;  // = Delta(t) >= theta_on or <= -theta_off
    // Shares p Delta_x / theta sum to one; the first event after init leaves the offset, Delta(t0), out (F034).
    uint8_t label = (flags_ & kFlagFirstAfterInit) != 0 ? detail::first_label(on, ds, dn, l, kc_.label_cut)
                                                        : detail::split_label(ds, dl, D, kc_.label_cut);
    const bool hot = map_.hot != HotType::kNone;
    if (hot && label != kLabelSignal) label = kLabelHot;
    RawEvent ev;
    ev.pixel = pixel_;
    ev.polarity = on ? 1 : 0;
    ev.label = label;
    ev.flags = static_cast<uint16_t>(flags_ | (hot ? kFlagHotPixel : 0));
    ev.t_cross = t;
    ev.ds = static_cast<float>(ds);
    ev.dn = static_cast<float>(dn);
    ev.dl = static_cast<float>(dl);
    ev.l_first = (flags_ & kFlagFirstAfterInit) != 0 ? static_cast<float>(l) : 0.0f;
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

  BALLA_HD BALLA_SENSOR_INLINE void comparator(const SubStep<T>& sub) {
    if constexpr (kHpf) {
      comparator_hpf(sub);
    } else {
      if (kc_.noise) comparator_tree(sub);
      else comparator_plain(sub);
    }
  }

  // Returns false while the physical refractory covers the rest of the sub-step; at the release
  // inside it, r = x3(t_rel) + eps_ci (M3 A.16).
  BALLA_HD BALLA_SENSOR_INLINE bool handle_blind(const SubStep<T>& sub) {
    if (blind_until_ == kNotBlind) return true;
    if (blind_until_ >= sub.t_b) return false;
    const int64_t t = blind_until_;
    uint32_t refinements = 0;
    const T n3 = kc_.noise ? n3_here(t, refinements) : T(0);
    stats_.refinements += refinements;
    set_reference(t, sub.s3(static_cast<T>(t - sub.t_a) * T(1e-9)), n3, -kc_.eps_ci);
    blind_until_ = kNotBlind;
    return true;
  }

  BALLA_HD BALLA_SENSOR_INLINE void comparator_plain(const SubStep<T>& sub) {
    while (true) {
      if (!handle_blind(sub)) return;
      const int64_t lo = max(t_search_, sub.t_a);
      if (lo >= sub.t_b) return;
      const detail::Comparand<T> c = comparand();
      ++stats_.leaf_intervals;
      const detail::Hit hit = detail::search(detail::delta_fn(sub, lo, static_cast<const T*>(nullptr), 0, c), lo, sub.t_b, lo, theta_on_,
                                             theta_off_);
      if (!hit.found) {
        t_search_ = max(t_search_, sub.t_b);
        return;
      }
      fire(hit.t, hit.on, sub.s3(static_cast<T>(hit.t - sub.t_a) * T(1e-9)), T(0));
    }
  }

  // The sub-step against the cell's bridge tree (M3 §D.6): the level-0 hull first, then, where it
  // cannot exclude a crossing, the walk below it (out of line), which returns the first crossing.
  BALLA_HD BALLA_SENSOR_INLINE void comparator_tree(const SubStep<T>& sub) {
    const int64_t tc0 = cell_ * kNoiseCellNs;
    while (true) {
      if (!handle_blind(sub)) return;
      const int64_t lo = max(t_search_, sub.t_a);
      if (lo >= sub.t_b) return;
      const detail::Comparand<T> c = comparand();
      if (kc_.levels > 0 && !detail::may_cross(detail::delta_fn(sub, lo, hc0_, tc0, c), lo, sub.t_b, lo, margin0_, c)) {
        t_search_ = max(t_search_, sub.t_b);
        return;
      }
      const detail::Found<T> f = walk_here(tc0, sub, lo, c);
      stats_.refinements += f.refinements;
      stats_.leaf_intervals += f.leaves;
      if (!f.found) {
        t_search_ = max(t_search_, sub.t_b);
        return;
      }
      fire(f.t, f.on, f.s3, f.n3);
    }
  }

  // ------------------------------------------------------------------------------------------
  // Comparator under the high-pass filter (kHpf; hpf.h, SENSOR.md "High-pass filter")
  // ------------------------------------------------------------------------------------------
  struct HpfScan {
    bool declined;  // quiet_cell's scan: the interval would need a split (the general path takes the cell)
    bool found, on;
    int64_t t;  // the crossing, else the interval's end
    T s3, n3;   // at the crossing
    double m, mn;  // the memory and its noise part at t (fp64: hpf_increment)
  };

  // One interval of the sub-step from the anchor t_x, where the memory is hm_ (hmn_): the filter's forms there, the
  // longest halving [t_x, t_e) of [t_x, sub.t_b) whose b-mode bound is within eps_hp (n_split halvings; `quiet`
  // declines any), searched from max(t_search, t_x) as comparator_tree (or comparator_plain) searches a sub-step, with
  // the bound in every margin; the memory at the first crossing, else at t_e. Changes no state but the counters. The
  // memory at t_e is formed before the search, so that the forms need not live across the walk's call; a crossing
  // forms them again (the same bits).
  BALLA_HD BALLA_SENSOR_INLINE HpfScan hpf_scan(const SubStep<T>& sub, int64_t t_x, int& n_split, bool quiet) {
    const int64_t tc0 = cell_ * kNoiseCellNs;
    const detail::Comparand<T> c = comparand();
    const detail::HpfForm<T> f = detail::hpf_form(hpf_delta(sub, t_x, tc0, c), static_cast<T>(hm_), b_);
    // With noise, intervals end at the next lattice node (a sub-step never crosses one: b u <= 0.066, hpf.h); without
    // it the sub-step (up to a 2^23 ns piece) is one interval unless the bound halves it.
    int64_t t_e = kc_.noise ? min(sub.t_b, ((t_x >> kNoiseCellLog2) + 1) * kNoiseCellNs) : sub.t_b;
    T bound = detail::hpf_bound(f, static_cast<T>(t_e - t_x) * T(1e-9));
    n_split = 0;
    while (bound > kc_.eps_hp && t_e - t_x > 1) {
      t_e = t_x + (t_e - t_x) / 2;
      bound = detail::hpf_bound(f, static_cast<T>(t_e - t_x) * T(1e-9));
      ++n_split;
    }
    HpfScan out{quiet && n_split > 0, false, false, t_e, T(0), T(0), 0.0, 0.0};
    if (out.declined) return out;
    hpf_memory_at(f, t_x, tc0, static_cast<T>(t_e - t_x) * T(1e-9), out);
    const int64_t lo = max(t_search_, t_x);
    if (lo < t_e) {
      const ExpCubic<T> g0 = lo == t_x ? f.g : shifted(f.g, static_cast<T>(lo - t_x) * T(1e-9));
      if (kc_.noise) {
        if (kc_.levels == 0 || detail::may_cross(g0, lo, t_e, lo, margin0_ + bound, c)) {
          SubStep<T> part = sub;
          part.t_b = t_e;
          const detail::HpfCtx<T> h{f.g, t_x, tc0, {hc0_[0], hc0_[1], hc0_[2], hc0_[3]}, bound};
          const detail::Found<T> w =
              detail::walk_hpf<T, kP2>(kc_, pixel_, ln_a1_, rho_, triple(nk_), triple(nk1_), tc0, part, lo, c, h);
#if BALLA_SENSOR_CENSUS
          if constexpr (kCensus) census_walk(w.refinements, w.c, tc0);
#endif
          stats_.refinements += w.refinements;
          stats_.leaf_intervals += w.leaves;
          if (w.found) {
            out.found = true;
            out.on = w.on;
            out.t = w.t;
            out.s3 = w.s3;
            out.n3 = w.n3;
          }
        }
      } else {
        ++stats_.leaf_intervals;
        const detail::Hit hit = detail::search(g0, lo, t_e, lo, theta_on_, theta_off_);
        if (hit.found) {
          out.found = true;
          out.on = hit.on;
          out.t = hit.t;
          out.s3 = sub.s3(static_cast<T>(hit.t - sub.t_a) * T(1e-9));
        }
      }
    }
    if (out.found) {
      const detail::HpfForm<T> fx = detail::hpf_form(hpf_delta(sub, t_x, tc0, c), static_cast<T>(hm_), b_);
      hpf_memory_at(fx, t_x, tc0, static_cast<T>(out.t - t_x) * T(1e-9), out);
    }
    return out;
  }

  // Delta~ from t_x: delta_fn with the cell's level-0 Hermite (or no noise). Two calls rather than one through a
  // selected pointer, which would make the device code take hc0_'s address (local memory for every use of it).
  BALLA_HD BALLA_SENSOR_INLINE ExpCubic<T> hpf_delta(const SubStep<T>& sub, int64_t t_x, int64_t tc0,
                                                     const detail::Comparand<T>& c) const {
    if (kc_.noise) return detail::delta_fn(sub, t_x, hc0_, tc0, c);
    return detail::delta_fn(sub, t_x, static_cast<const T*>(nullptr), tc0, c);
  }

  // The memory and its noise part u seconds after the anchor t_x of an interval with forms f (in the cell from tc0).
  BALLA_HD BALLA_SENSOR_INLINE void hpf_memory_at(const detail::HpfForm<T>& f, int64_t t_x, int64_t tc0, T u,
                                                  HpfScan& out) const {
    out.m = hm_ + static_cast<double>(detail::hpf_increment(f, u));
    out.mn = hmn_;
    if (kc_.noise) {
      // The noise's part of the memory: driven by the level-0 Hermite less the reference's n3, a cubic.
      const T d0 = static_cast<T>(t_x - tc0) * T(1e-9);
      const T* z = hc0_;
      const ExpCubic<T> n{{(z[0] + d0 * (z[1] + d0 * (z[2] + d0 * z[3]))) - n3_ref_,
                           z[1] + d0 * (T(2) * z[2] + d0 * (T(3) * z[3])), z[2] + d0 * (T(3) * z[3]), z[3]},
                          {T(0), T(0)},
                          {kc_.cc.a2, kc_.cc.a3},
                          0};
      out.mn = hmn_ + static_cast<double>(detail::hpf_increment(detail::hpf_form(n, static_cast<T>(hmn_), b_), u));
    }
  }

  // The sub-step under the high-pass filter: intervals from the anchor (the sub-step's start, a release or an event
  // inside it, where the memory restarts at 0) to the sub-step's end; the memory advances exactly over each. While
  // blind past the sub-step's end the memory is not advanced: the release resets it.
  BALLA_HD BALLA_SENSOR_INLINE void comparator_hpf(const SubStep<T>& sub) {
    int64_t t_x = sub.t_a;
    while (true) {
      if (!handle_blind(sub)) return;
      if (t_ref_ > t_x) t_x = t_ref_;
      if (t_x >= sub.t_b) return;
      int n_split = 0;
      const HpfScan s = hpf_scan(sub, t_x, n_split, false);
      stats_.hpf_splits += static_cast<uint32_t>(n_split);
      if (s.found) {
        fire(s.t, s.on, s.s3, s.n3, static_cast<T>(s.m), static_cast<T>(s.mn));
        continue;
      }
      hm_ = s.m;
      hmn_ = s.mn;
      t_search_ = max(t_search_, s.t);
      t_x = s.t;
    }
  }

  // ------------------------------------------------------------------------------------------
  // Instantaneous log response (advance_instant)
  // ------------------------------------------------------------------------------------------
  // I(t) at an integer t of sample segment seg: the same expression as next_piece's endpoint values.
  BALLA_HD T photocurrent(int seg, int64_t t) const {
    T I = dark_;
    for (int g = 0; g < kc_.groups; ++g) {
      if (!live(g)) continue;
      const T m = kc_.waves[g].type == WaveformType::kConstant ? T(1) : wave_value(g, t);
      I = I + m * scale_ * group_value(seg, t, g);
    }
    return I;
  }

  // ln I relative to the pixel's log offset (fp32 keeps the ratio's relative precision).
  BALLA_HD T log_ratio(T I) const { return pmath::log(I / i_scale_); }

  // Delta at t for x = log_ratio(I(t)): the expression fire() evaluates with s3 = x, n3 = 0, so a crossing found here
  // is the one fire() books.
  BALLA_HD T instant_delta(int64_t t, T x) const {
    const T ds = x - s3_ref_;
    const T dn = T(0) - n3_ref_;
    const T dl = leak_ * static_cast<T>(static_cast<double>(t - t_ref_) * 1e-9) + dr_ref_;
    return ds + dn + dl;
  }

  // Bounds of the photocurrent over a piece [a, c] (one sample segment, no breakpoint inside), valid on every
  // sub-interval: I is linear in t for constant groups, and |I''| <= curv = sum_g s (max|m''| v_hi + 2 max|m'| |v'|) for
  // modulated ones (v linear); i_lo <= I <= i_hi from each group's range of v and the global range of m (Fourier
  // 1 -+ sum |a_k|, pulse floor and peak), the tighter bound on long intervals; mod = sum_g s v_hi over the modulated
  // groups (their rounding scale). Without a live modulated group I is linear (curv 0), so its endpoint values bound
  // it on every node and no range caps them: i_hi is +inf (a cap at I_dk dropped every rising Delta under leak while
  // I fell).
  struct Curvature {
    T curv, mod, i_lo, i_hi;
  };
  BALLA_HD Curvature instant_curvature(int seg, int64_t a, int64_t c) const {
    if (!wave_live_ || c <= a) return Curvature{T(0), T(0), dark_, T(INFINITY)};
    Curvature k{T(0), T(0), dark_, dark_};
    const T h = static_cast<T>(c - a) * T(1e-9);
    for (int g = 0; g < kc_.groups; ++g) {
      if (!live(g)) continue;
      const T va = group_value(seg, a, g), vc = group_value(seg, c, g);
      const T v_lo = min(va, vc), v_hi = max(va, vc);
      const Waveform& w = kc_.waves[g];
      T m_lo = T(1), m_hi = T(1);
      if (w.type == WaveformType::kFourier) {
        T sum = T(0);
        for (int j = 0; j < w.n_harmonics; ++j) sum = sum + abs(static_cast<T>(w.a[j]));
        m_lo = max(T(1) - sum, T(0));
        m_hi = T(1) + sum;
      } else if (w.type == WaveformType::kPulse) {
        m_lo = static_cast<T>(w.floor_level);
        m_hi = static_cast<T>(w.peak_level);
      }
      k.i_lo = k.i_lo + scale_ * m_lo * v_lo;
      k.i_hi = k.i_hi + scale_ * m_hi * v_hi;
      if (w.type == WaveformType::kConstant) continue;
      const WaveBounds<T> b = wave_bounds(g, a, c);
      k.curv = k.curv + scale_ * (b.d2m_max * v_hi + T(2) * b.dm_max * abs(vc - va) / h);
      k.mod = k.mod + scale_ * v_hi;
    }
    return k;
  }

  // Can Delta reach a level at an integer of (a, c] (inside one piece; the condition does not hold at a), given
  // Ia = I(a), Ic = I(c)? With |I''| <= C on [a, c] (h = c - a), a rise |Ic - Ia| >= C h^2 / 2 puts the extremes of I
  // at the endpoints (an interior maximum needs a smaller rise: the parabola of curvature -C through both values), so
  // without leak Delta's value at c decides; with leak (Delta = ln I + l t, l >= 0) only a non-decreasing I does
  // (|Ic - Ia| > C h^2 with the rounding margin: the mean value theorem). Linear I (no live modulated group) is the
  // case C = 0, with no range cap. Elsewhere I lies within the chord bound h^2/8 C of its endpoint values (M3 B.18),
  // and within its range over the piece, widened by margins for the rounding of the evaluated I (m to a few ulp per
  // harmonic) and of the logarithm and sums. The endpoint tests take the computed values as they are: like the cascade path's monotone
  // pieces, a crossing whose overshoot stays within the rounding of I may be missed; in exchange a flat or near-tangent
  // Delta just below a level costs a few tests per bisection level, not a scan.
  BALLA_HD bool instant_may_fire(int64_t a, int64_t c, T Ia, T Ic, const Curvature& k) const {
    constexpr T eps = FloatTraits<T>::kEpsilon;
    T hi = max(Ia, Ic), lo = min(Ia, Ic);
    const T h = static_cast<T>(c - a) * T(1e-9);
    const T hh = h * h * k.curv;
    const T margin = T(64) * eps * (hi + T(2) * k.mod);
    if (leak_ == T(0) && hi - lo >= T(0.5) * hh) {  // extremes at the endpoints
      const T D = instant_delta(c, log_ratio(Ic));
      return D >= theta_on_ || D <= -theta_off_;
    }
    if (Ic >= Ia && (k.curv == T(0) || hi - lo > hh + T(2) * margin)) {  // non-decreasing: Delta too (leak >= 0)
      return instant_delta(c, log_ratio(Ic)) >= theta_on_;
    }
    hi = min(hi + T(0.125) * hh, k.i_hi) + margin;
    lo = max(lo - T(0.125) * hh, k.i_lo) - margin;
    const T dl_a = leak_ * static_cast<T>(static_cast<double>(a - t_ref_) * 1e-9) + dr_ref_;
    const T dl_c = leak_ * static_cast<T>(static_cast<double>(c - t_ref_) * 1e-9) + dr_ref_;
    const T base = T(0) - n3_ref_ - s3_ref_;
    const T x_hi = log_ratio(hi);
    if (x_hi + base + dl_c + T(16) * eps * (T(1) + abs(x_hi) + abs(s3_ref_) + abs(dl_c)) >= theta_on_) return true;
    if (!(lo > T(0))) return true;
    const T x_lo = log_ratio(lo);
    return x_lo + base + dl_a - T(16) * eps * (T(1) + abs(x_lo) + abs(s3_ref_) + abs(dl_a)) <= -theta_off_;
  }

  struct InstantHit {
    bool found, on;
    int64_t t;
    T x;  // log_ratio(I(t))
  };

  // The first integer t in [lo, hi) of a piece (one sample segment, no breakpoint inside) where Delta fires: a time-
  // ordered, depth-first bisection of (a, c] (a already checked) with the exclusion test above; leaves are evaluated.
  BALLA_HD InstantHit instant_search(int seg, int64_t lo, int64_t hi) {
    struct Node {
      int64_t a, c;
      T Ia, Ic;
    };
    const T I_lo = photocurrent(seg, lo);
    InstantHit out{false, false, lo, log_ratio(I_lo)};
    if (instant_fires(lo, out)) return out;
    if (hi - lo == 1) return out;
    const Curvature k = instant_curvature(seg, lo, hi - 1);
    Node stack[32];  // a piece spans at most 2^23 ns: 23 levels, one pending right sibling each
    int sp = 0;
    stack[sp++] = {lo, hi - 1, I_lo, photocurrent(seg, hi - 1)};
    while (sp > 0) {
      const Node n = stack[--sp];
      if (n.c - n.a == 1) {
        out.t = n.c;
        out.x = log_ratio(n.Ic);
        if (instant_fires(n.c, out)) return out;
        continue;
      }
      ++stats_.leaf_intervals;
      if (!instant_may_fire(n.a, n.c, n.Ia, n.Ic, k)) continue;
      const int64_t m = n.a + (n.c - n.a) / 2;
      const T Im = photocurrent(seg, m);
      stack[sp++] = {m, n.c, Im, n.Ic};
      stack[sp++] = {n.a, m, n.Ia, Im};
    }
    out.found = false;
    return out;
  }

  BALLA_HD bool instant_fires(int64_t t, InstantHit& h) const {
    const T D = instant_delta(t, h.x);
    h.found = D >= theta_on_ || D <= -theta_off_;
    h.on = D >= theta_on_;
    h.t = t;
    return h.found;
  }

  // n3 at t and the walk of the current cell.
  BALLA_HD BALLA_SENSOR_INLINE T n3_here(int64_t t, uint32_t& refinements) const {
#if BALLA_SENSOR_CENSUS
    const uint32_t r0 = refinements;
    const T v = detail::n3_at<T, kP2>(kc_, pixel_, ln_a1_, rho_, triple(nk_), triple(nk1_), cell_ * kNoiseCellNs, t,
                                      refinements);
    if constexpr (kCensus) census_n3(t, refinements - r0);
    return v;
#else
    return detail::n3_at<T, kP2>(kc_, pixel_, ln_a1_, rho_, triple(nk_), triple(nk1_), cell_ * kNoiseCellNs, t,
                                 refinements);
#endif
  }
  BALLA_HD BALLA_SENSOR_INLINE detail::Found<T> walk_here(int64_t tc0, const SubStep<T>& sub, int64_t lo,
                                                          const detail::Comparand<T>& c) const {
#if BALLA_SENSOR_CENSUS
    const detail::Found<T> f =
        detail::walk<T, kP2>(kc_, pixel_, ln_a1_, rho_, triple(nk_), triple(nk1_), tc0, sub, lo, c);
    if constexpr (kCensus) census_walk(f.refinements, f.c, tc0);
    return f;
#else
    return detail::walk<T, kP2>(kc_, pixel_, ln_a1_, rho_, triple(nk_), triple(nk1_), tc0, sub, lo, c);
#endif
  }

  BALLA_HD detail::Comparand<T> comparand() const {
    return {t_ref_, s3_ref_, n3_ref_, dr_ref_, leak_, theta_on_, theta_off_};
  }
  BALLA_HD static detail::Triple<T> triple(const T* x) { return {{x[0], x[1], x[2]}}; }

#if BALLA_SENSOR_CENSUS
  // ------------------------------------------------------------------------------------------
  // The sensor census (census.h; BALLASIM_SENSOR_CENSUS builds only): counts, never changes a result
  // ------------------------------------------------------------------------------------------
  static constexpr bool kCensus = requires(Stats& s) { s.census; };

  struct CensusClass {
    int cls = -1;  // the first failing check's counter (kCnRej*), -1 when eligible or not a whole fresh cell
    bool counted = false, cheap = false, eligible = false, within = false;
  };

  // The mean photocurrent of the augmented sample i's live groups (waveforms mean-normalized; no glare), fp64.
  BALLA_HD double census_ibar(int i) const {
    double I = static_cast<double>(dark_);
    for (int g = 0; g < kc_.groups; ++g) {
      if (live(g)) I += static_cast<double>(scale_) * static_cast<double>(sample_value(i, g));
    }
    return I;
  }
  BALLA_HD double census_ibar_held() const {
    double I = static_cast<double>(dark_);
    for (int g = 0; g < kc_.groups; ++g) {
      if (live(g)) I += static_cast<double>(scale_) * static_cast<double>(ps_.held[g]);
    }
    return I;
  }
  BALLA_HD bool census_same(int i, int j) const {
    for (int g = 0; g < kc_.groups; ++g) {
      if (live(g) && !(sample_value(i, g) == sample_value(j, g))) return false;
    }
    return true;
  }
  BALLA_HD bool census_same_held() const {  // sample 0 against the input held before the call
    for (int g = 0; g < kc_.groups; ++g) {
      if (live(g) && !(sample_value(0, g) == ps_.held[g])) return false;
    }
    return true;
  }
  // The modulation depth s |a|_1 of the live modulated groups at (seg, t), over the mean photocurrent ibar.
  BALLA_HD double census_depth(int seg, int64_t t, double ibar) const {
    double d = 0.0;
    for (int g = 0; g < kc_.groups; ++g) {
      const Waveform& w = kc_.waves[g];
      if (!live(g) || w.type == WaveformType::kConstant) continue;
      double amp = 0.0;
      if (w.type == WaveformType::kFourier) {
        for (int j = 0; j < w.n_harmonics; ++j) amp += w.a[j] < 0.0 ? -w.a[j] : w.a[j];
      } else {
        amp = w.peak_level - 1.0 > 1.0 - w.floor_level ? w.peak_level - 1.0 : 1.0 - w.floor_level;
      }
      d += static_cast<double>(scale_) * static_cast<double>(group_value(seg, t, g)) / ibar * amp;
    }
    return d;
  }
  // The hold that sample j ends a change into: its start and the change's total variation of ln Ibar (the run of
  // differing samples before j, within the call); {INT64_MIN, 0} when nothing changed from j back to the call's start.
  BALLA_HD void census_last_change(int j, int64_t& t_hold, double& delta) const {
    while (j > 0 && census_same(j - 1, j)) --j;
    if (j > 0) {
      t_hold = sample_time(j);
      delta = 0.0;
      for (int i = j; i > 0 && !census_same(i - 1, i); --i) {
        const double r = census_ibar(i) / census_ibar(i - 1);
        delta += r > 1.0 ? std::log(r) : -std::log(r);
      }
      return;
    }
    if (skip_ == 1 && !census_same_held()) {
      t_hold = t_state_;
      const double r = census_ibar(0) / census_ibar_held();
      delta = r > 1.0 ? std::log(r) : -std::log(r);
      return;
    }
    t_hold = INT64_MIN;
    delta = 0.0;
  }

  void census_init(int64_t t0) {
    if (stats_.census.px == nullptr) return;
    const double ibar = census_ibar_held();
    stats_.census.px[pixel_] = CensusPixel{t0, static_cast<float>(census_depth(0, t0, ibar))};
  }

  void census_end_call() {
    if (stats_.census.px == nullptr) return;
    int64_t t_hold;
    double delta;
    census_last_change(n_aug() - 1, t_hold, delta);
    if (t_hold != INT64_MIN) stats_.census.px[pixel_] = CensusPixel{t_hold, static_cast<float>(delta)};
  }

  // The class of the whole fresh cell [t, t + 2^16) by the orbit entry's checks (census.h), from the state at its start.
  CensusClass census_classify(int64_t t, int64_t t_to) const {
    CensusClass c;
    if (!kc_.noise || have_k1_ || t + kNoiseCellNs > t_to || (t & (kNoiseCellNs - 1)) != 0) return c;
    c.counted = true;
    const int64_t tc1 = t + kNoiseCellNs;
    if (kc_.levels == 0) {
      c.cls = kCnRejNoise;
      return c;
    }
    const int seg = segment_of(seg_, t);
    if (segment_end(seg) < tc1 || (seg + 1 < n_aug() && !census_same(seg, seg + 1))) {
      c.cls = kCnRejSample;
      return c;
    }
    int n_mod = 0, g_mod = -1;
    for (int g = 0; g < kc_.groups; ++g) {
      if (!live(g)) continue;
      if (glare_on() && !(kc_.glare.a[g] == kc_.glare.b[g])) {
        c.cls = kCnRejGlare;
        return c;
      }
      if (kc_.waves[g].type != WaveformType::kConstant) {
        ++n_mod;
        g_mod = g;
      }
    }
    if (n_mod == 0 || n_mod > 1 || kc_.waves[g_mod].type == WaveformType::kPulse) {
      c.cls = n_mod == 0 ? kCnRejStatic : (n_mod > 1 ? kCnRejMulti : kCnRejPulse);
      return c;
    }
    if (blind_until_ != kNotBlind) {
      c.cls = kCnRejBlind;
      return c;
    }
    if constexpr (kHpf) {
      c.cls = kCnRejHpf;
      return c;
    }
    c.cheap = true;
    double ibar = static_cast<double>(dark_);
    for (int g = 0; g < kc_.groups; ++g) {
      if (live(g)) ibar += static_cast<double>(scale_) * static_cast<double>(group_value(seg, t, g));
    }
    const double eps4 = 1.5 * static_cast<double>(kc_.cc.eps_accept) / 4.0;  // eps_h / 4
    const double d = census_depth(seg, t, ibar);
    const double xbar = std::log(ibar / static_cast<double>(i_scale_));
    const double dev = std::abs(static_cast<double>(st_.z) * ibar - 1.0) + std::abs(static_cast<double>(st_.s2) - xbar) +
                       std::abs(static_cast<double>(st_.s3) - xbar);
    c.within = d >= 1.0 || dev <= 3.0 * d / (1.0 - d) + eps4;
    int64_t t_hold;
    double delta;
    census_last_change(seg, t_hold, delta);
    if (t_hold == INT64_MIN && stats_.census.px != nullptr) {
      t_hold = stats_.census.px[pixel_].t_hold;
      delta = static_cast<double>(stats_.census.px[pixel_].delta);
    }
    bool settled = true;
    if (t_hold != INT64_MIN && delta > eps4) {
      double a_min = static_cast<double>(kc_.cc.k) * ibar;
      a_min = std::min(a_min, std::min(static_cast<double>(kc_.cc.a2), static_cast<double>(kc_.cc.a3)));
      settled = static_cast<double>(t - t_hold) * 1e-9 >= std::log(delta / eps4) / a_min;
    }
    c.eligible = settled;
    if (!settled) c.cls = kCnRejUnsettled;
    return c;
  }

  void census_count(const CensusClass& c, bool quiet, int64_t t) {
    if (!c.counted) {
      if (!quiet) c_lean_ = false;  // a resumed or partial cell: the general pass
      return;
    }
    uint64_t* n = stats_.census.n;
    ++n[kCnCells];
    if (quiet) {
      ++n[kCnQuiet];
      return;
    }
    if (c.cheap) {
      ++n[kCnCheap];
      if (c.within) ++n[kCnWithinRadius];
    }
    if (c.eligible) {
      ++n[c_lean_ ? kCnEligibleLean : kCnEligibleGeneral];
      if (c.within) ++n[kCnEligibleWithin];
      c_elig_cell_ = t >> kNoiseCellLog2;
      c_elig_fired_ = false;
      return;
    }
    ++n[c.cls];
    c_lean_ = false;
  }

  // The draws of a walk or an n3 descent in the cell from tc0: rho != 1 levels and nodes, new ones for the cell.
  void census_cell_draws(int64_t tc0, uint32_t rho_levels, uint64_t nodes) const {
    if (c_cell_ != tc0) {
      c_cell_ = tc0;
      c_cell_levels_ = 0;
      c_cell_nodes_ = 0;
    }
    uint64_t* n = stats_.census.n;
    n[kCnFactorCell] += static_cast<uint64_t>(__builtin_popcount(rho_levels & ~c_cell_levels_));
    n[kCnNodesCell] += static_cast<uint64_t>(__builtin_popcountll(nodes & ~c_cell_nodes_));
    c_cell_levels_ |= rho_levels;
    c_cell_nodes_ |= nodes;
  }
  void census_walk(uint32_t refinements, const CensusDraws& d, int64_t tc0) const {
    uint64_t* n = stats_.census.n;
    ++n[kCnWalks];
    n[kCnBridgesWalk] += refinements;
    if (rho_ != T(1)) n[kCnBridgesWalkRho] += refinements;
    n[kCnFactorWalk] += static_cast<uint64_t>(__builtin_popcount(d.rho_levels));
    census_cell_draws(tc0, d.rho_levels, d.nodes);
  }
  // n3_at's descent to t drew levels 1..r (the same path again, for its nodes).
  void census_n3(int64_t t, uint32_t r) const {
    uint64_t* n = stats_.census.n;
    ++n[kCnN3Calls];
    n[kCnBridgesN3] += r;
    const int64_t tc0 = cell_ * kNoiseCellNs;
    uint32_t levels = 0;
    uint64_t nodes = 0;
    int64_t node_t = tc0;
    for (uint32_t l = 1; l <= r; ++l) {
      const int64_t mid = node_t + detail::level_length(static_cast<int>(l));
      if (rho_ != T(1)) levels |= 1u << l;
      if (kc_.levels <= 6) nodes |= uint64_t{1} << ((mid - tc0) >> (kNoiseCellLog2 - kc_.levels));
      if (t > mid) node_t = mid;
    }
    if (rho_ != T(1)) n[kCnBridgesN3Rho] += r;
    census_cell_draws(tc0, levels, nodes);
  }

  bool c_lean_ = true;                    // every cell of the call so far quiet or eligible
  int64_t c_elig_cell_ = INT64_MIN;       // the last eligible cell
  bool c_elig_fired_ = false;
  mutable int64_t c_cell_ = INT64_MIN;    // the cell of c_cell_levels_ and c_cell_nodes_
  mutable uint32_t c_cell_levels_ = 0;
  mutable uint64_t c_cell_nodes_ = 0;
#endif

  // ------------------------------------------------------------------------------------------
  struct BlockCache {
    T ln_a1;
    T step[12];     // Phi (6) and chol Q (6) of the lattice step
    T margin0[23];  // level-0 exclusion margin
    bool valid;
  };

  const KernelConstants<T>& kc_;
  const PixelMap& map_;
  uint32_t pixel_;
  PixelState& ps_;
  int64_t t_state_;
  InputRun in_;
  Sink& sink_;
  Stats& stats_;
  HpfState* hs_;          // the high-pass filter's state (kHpf only)
  T b_ = T(0);            // its rate 1/tau_hp
  // Its memory m and m_n at the anchor (the model time, or the reference's time), fp64 in every precision: each
  // interval adds its change (hpf_increment, in T), so that rounding does not accumulate over tau_hp.
  double hm_ = 0.0, hmn_ = 0.0;
  int skip_ = 0;
  int seg_ = 0;
  uint32_t live_ = 0;
  bool wave_live_ = false;

  T theta_on_, theta_off_, leak_, dark_, scale_;

  BlockCache cache_{};  // used when kCacheBlocks

  T settled_z_ = T(-1), settled_ln_a1_ = T(0);  // rate of a settled cell, reused while z is unchanged
  T hc0_[4] = {}, margin0_ = T(0);               // the cell's level-0 Hermite of n3 and margin
  CascadeState<T> st_{};
  T i_scale_ = T(1);
  T x1_ = T(0), tv_ = T(0);
  T nk_[3] = {}, nk1_[3] = {};
  T ln_a1_ = T(0);
  T rho_ = T(1);  // the cell's diffusion ratio (PHASE2 D-P2-2; 1 under frozen diffusion and in settled cells)
  int64_t cell_ = 0;
  bool have_k1_ = false;
  int64_t t_ref_ = 0;
  T s3_ref_ = T(0), n3_ref_ = T(0), dr_ref_ = T(0);
  int64_t blind_until_ = kNotBlind;
  int64_t t_search_ = 0;
  uint16_t flags_ = 0;
};

}  // namespace balla::sensor
