// P2-D D3a (plan ballasim_out/p2_plans/p2d.md §3.1-§3.4, §5.3, §6 row D3a): predict-and-verify (P&V) localization of
// shadow flips, in shadow mode (accuracy.localization shadow): it runs beside bisection on round 0's signature-selected
// intervals and only counts (WindowStats::pv), so every output stays bit for bit. BALLA_HD functors on one pixel batch
// (sampling.h's layout), so the host and the device run the same arithmetic; every decision is a pure function of the
// pixel's own nodes, codes, tables, probes and plan data, so the counts do not depend on the batching.
//
// Per batch, after Round0Flags (window_impl.h WindowEngine::localize):
// - PvSlots: the candidates, the intervals (i - 1, i) of eligible pixels (N = PvParams::n, the band's N_band) that the
//   signature flag selects and bisection splits (signature_flag's splits; the jump gate's selection is kept); R0, the
//   structure rule (pv_shadow's view): the ends' geometry hashes differ (a coverage change, kRtCover), a table overflow
//   or a node without codes (a fifth cluster, a carried node without them: R0c), both ends uniform, an unresolvable
//   uniform end; each candidate's end-table step J (plan §3.2; E(b) - E(a) where the tables do not give it).
// - PvThin: R2's a-priori width (the sandwich rule, plan §3.4) at every node that a candidate's four-node window needs.
// - PvPixel: per pixel, the segmented scan of J, R6 (a grazing receiver, the plan bits), R2 on the four nodes, R5 (the
//   step-free midpoint test: the eta_L triples on E - J, with the cause of a flag) and the flip counts (Step D).
// - PvFill: the flip list in (pixel, light, direction, interval, k) order, so that each group (pixel, light,
//   direction and a run of consecutive intervals: one arrival-time field) is contiguous; the bracket cell tau per
//   (interval, light) (plan §3.1).
// - PvSeeds: up to three seeds per group, farthest-point in u; the rounds: PvEmit (each open flip's probe cells),
//   PvRequests, one probe launch, PvConsume (brackets, R1: a third state), PvFit (the group's affine fit, D3a's
//   predictor: t = alpha + beta . (u - u_bar) over its localized flips, n_loc >= 3; plan §3.1's ladder, D3b adds the
//   local, projective and quadratic stages).
// - PvTally per flip, and PvCount per interval after the assembly (cost: the refinement samples bisection rendered in
//   it, the assembly's counts).
// Shadow mode runs P&V on every candidate that R0 leaves (R2-, R4-, R5- and R6-routed ones too), so the full guard set
// (R0, R6, R4, R2, R5, R1, R3) and the posterior one (without R2) are both measured on the same run.
//
// P2-D D3b (plan §3.1 Step F, §3.4 R3 / R4, §5.3, §6 row D3b):
// - The predictor ladder (PvFitGroups, pv_predict): per group the affine stage (n_loc >= 3), the local affine stage
//   over the 6 localized flips nearest the query (n_loc >= 6), F020's projective form t (1 + b.v) = a0 + a.v (n_loc >=
//   6) and the quadratic (n_loc >= 10), in centred and scaled coordinates; the next round predicts with the stage of
//   the smallest leave-one-out RMS over (at most kPvLoo of) the group's localized flips. Waves: a waiting flip is
//   promoted to verification when its rank among the group's non-seeds is below kPvWave times the group's resolved
//   flips, so later flips are predicted from earlier ones (the CPU model's sequential order; k is the renderer's
//   Owen-scrambled Sobol order, well spread in u).
// - R4 (pre): the plan bit kPvThin (D1's thin-moving-shadow width, PvPlanPass) routes the pixel's candidates.
// - R3 (after the rounds, PvR3): an interval with localized flips of both directions of one (key, light) routes when
//   an opposite pair lies within 2 D_bound + pitch along the edge normal n (the gradient of the local affine fit of
//   the flip's own group, 6 nearest) and along its tangent (Euclidean where the fit is degenerate); PvR3Prop routes the
//   neighbouring intervals that hold flips of that (key, light) (the arrival and departure intervals).
// - The state check at bisection's midpoints (G-D2, PvCheck): the brackets (PvBracket) stay through the refinement
//   rounds, whose band calls write the midpoints' codes; each sub-sample's rendered state in a P&V-run interval is
//   compared with what the brackets predict: an unseen transient (equal end states, other state at the midpoint), an
//   unseen flip pair beside a localized flip (multi: outside the bracket at a time no probe observed), a third state
//   (a key equal to neither end's), and a contradiction (the rendered state at an observed bracket end differs from
//   the observed one, or the flip is missing: a bug, zero allowed).
// - The would-be P&V nodes per interval (PvNodeKeys / PvNodeCount, the sensor-piece estimate): the distinct inner
//   boundaries of the occupied tau-cells.
#pragma once

#include <math.h>
#include <stdint.h>

#include "core/hd.h"
#include "core/pmath/explog.h"
#include "core/rng/subsample.h"
#include "core/rng/warp.h"
#include "engine/par.h"
#include "engine/plan.h"
#include "engine/sampling.h"
#include "render/shade/codes.h"
#include "render/shade/shade.h"

namespace balla::engine {

using render::pv_get_code;

// What P&V reads besides the batch (SampleConfig::pv; plan §3.1). n: the eligible N (0: P&V off); lights: the scene's
// shadow-casting lights (<= kPvMaxLights) and light_group their light groups (shadow bit order); seed and warp: the
// sub-sample positions (render/ray.h subsample_direction's, warp in the engine's space); t_q: the bracket floor's quantum
// max(q_s / 2, 512) ns (DESIGN §6.2); vhat: the image-speed bound (px/s) of pixels without a plan's tile v_hat (a
// WindowEngine caller without a plan, the tests'; the Camera runs P&V in planned windows only, so its v_hat is the
// plan's); obliquity: the plan's rates carry F021's factors (no cone gain on D_bound).
struct PvParams {
  uint32_t n = 0;
  int32_t lights = 0;
  int8_t light_group[render::kPvMaxLights] = {};
  uint64_t seed = 0;
  rng::WarpTableView<float> warp{};
  uint32_t t_q = 512;
  float vhat = 0.0f;
  bool obliquity = false;
  // P2-D D3b, a test hook: the ladder's stages above the affine one (kStLocal, kStProj, kStQuad bits 1 << stage; the
  // engine runs them all).
  uint32_t stages = 0xFFu;
};

// An interval's route bits (per hi slot i of (i - 1, i)). kRtCand: a candidate; kRtCover: R0 by a coverage change;
// kRtR0: R0 by another cause (kRtOverflow, kRtNoTable, kRtCarried, kRtKey: equal hashes, unequal keys); R6, R2, R5
// (kRtR5Cause: 1 + the cause, plan §3.3), R1 (a-posteriori); D3b: R4 (pre), R3 and its propagation R3p (a-posteriori).
inline constexpr uint32_t kRtCand = 1u, kRtCover = 2u, kRtR0 = 4u, kRtR6 = 8u, kRtR2 = 16u, kRtR5 = 32u, kRtR1 = 64u,
                          kRtR4 = 128u, kRtOverflow = 1u << 8, kRtNoTable = 1u << 9, kRtCarried = 1u << 10,
                          kRtKey = 1u << 11, kRtR5Shift = 12, kRtR3 = 1u << 14, kRtR3p = 1u << 15;
// The guard sets' pre-routing rules (decided before any probe: P&V runs on the candidates they leave), and their
// routes with R1 and R3 (a-posteriori: after P&V's probes).
inline constexpr uint32_t kRtPreFull = kRtCover | kRtR0 | kRtR6 | kRtR4 | kRtR2 | kRtR5;
inline constexpr uint32_t kRtPrePosterior = kRtPreFull & ~kRtR2;  // `posterior`: without R2
inline constexpr uint32_t kRtFull = kRtPreFull | kRtR1 | kRtR3 | kRtR3p;
inline constexpr uint32_t kRtPosterior = kRtPrePosterior | kRtR1 | kRtR3 | kRtR3p;

// The counters (WindowStats::pv). Three classes of five words (intervals, refinement cost, flips, localized flips,
// probes): every candidate (kPvCand; flips and probes of those P&V ran on), those the full set accepts (kPvFull) and
// those the posterior set accepts (kPvPost). Per rule (Cover, R0, R6, R2, R5, R1) its intervals and their cost (a rule
// counts every interval it fires on, alone or not); R0's causes; R5's causes (the flagging triple's other interval a
// candidate, a routed candidate, or none); probes by kind (seed, bisection without a prediction, verify, gallop and
// bisection after a prediction); flips localized as seeds, from a prediction, verify hits (localized by their first
// verify), dropped (R1) and unfinished (the round cap); probe rounds (host); the eligible pixels' refinement cost; the
// probes of the candidates each guard set runs P&V on (full, posterior: its pre-routing rules leave them; R1-routed
// ones included, plan §3.11's c_pv: P&V's cost is spent before R1 routes).
// D3b: per rule (R4, R3, R3's propagation) its intervals and their cost; per stage of the ladder (affine, local,
// projective, quadratic) the localized flips it predicted, their probes and verify hits; the state check at
// bisection's midpoints per class (all, full, posterior; kPvChk* words), its contradictions (zero allowed), unchecked
// midpoints (no codes: an overflow, an unresolvable uniform one) and uniform midpoints whose geometry hash differs
// from the ends' (a third state); the would-be P&V nodes per class.
inline constexpr int kPvCand = 0, kPvFull = 5, kPvPost = 10, kPvRule = 15, kPvR0Cause = 27, kPvR5Cause = 31,
                     kPvProbeKind = 34, kPvSeeds = 38, kPvPredicted = 39, kPvVerifyHits = 40, kPvDropped = 41,
                     kPvUnfinished = 42, kPvRounds = 43, kPvEligibleCost = 44, kPvRunProbes = 45, kPvRuleB = 47,
                     kPvStage = 53, kPvCheck = 65, kPvCheckBug = 86, kPvNodes = 89, kPvWords = 92;
// The state check's words per class: midpoints checked, (sub-sample, light) states compared, unseen transients,
// multi (an unseen pair beside a localized flip), third-state sub-samples, states inside their bracket (skipped),
// states without a bracket (the flip dropped or unfinished).
inline constexpr int kPvChkMid = 0, kPvChkCompared = 1, kPvChkTransient = 2, kPvChkMulti = 3, kPvChkThird = 4,
                     kPvChkInBracket = 5, kPvChkUnpredicted = 6, kPvChkWords = 7;
inline constexpr int kPvRoundsMax = 160;  // rounds per batch: seeds, waves and gallops (D3a: <= 3 log2(2^19) + 4)
// The ladder's stages (PvFit::stage, PvFlip::stage), the leave-one-out sample per group and the waves' growth.
inline constexpr uint8_t kStNone = 0, kStAffine = 1, kStLocal = 2, kStProj = 3, kStQuad = 4;
inline constexpr uint32_t kPvLoo = 16, kPvLocalK = 6, kPvWave = 2;

BALLA_HD inline void atomic_or(uint32_t* p, uint32_t v) {
#if defined(__CUDA_ARCH__)
  atomicOr(p, v);
#else
  *p |= v;
#endif
}
BALLA_HD inline uint32_t pv_popcount(uint32_t v) {
#if defined(__CUDA_ARCH__)
  return static_cast<uint32_t>(__popc(v));
#else
  return static_cast<uint32_t>(__builtin_popcount(v));
#endif
}
BALLA_HD inline void atomic_add(uint32_t* p, uint32_t v) {
#if defined(__CUDA_ARCH__)
  atomicAdd(p, v);
#else
  *p += v;
#endif
}

// The sub-sample offset of k (px; render/ray.h subsample_direction's) of camera cam's pixel.
BALLA_HD inline Vec2<float> pv_position(const PvParams& p, uint16_t cam, uint32_t pixel, uint32_t k) {
  return rng::subsample_offset(p.warp, rng::subsample_pattern_seed(p.seed, cam, pixel), k);
}

// The bracket cell (plan §3.1): tau = min(max(2^floor(log2(2 clamp(tol_t, t_q, 10 us))), tau_min), len / 16), at
// least 2^shift (a pose slot of the window clock), tol_t = kappa sigma_v,ref w_e / (rho(c) c v_hat) with kappa 0.1,
// sigma_v,ref 0.02, w_e 1 px, rho(c) 1.06 / 1.15 / 1.39 at c = 0.5 / 1 / ln 10 (linear between, clamped), c the edge
// contrast (log), v_hat (px/s); tau_min = max(2^shift, 2^floor(log2(2 t_q))).
BALLA_HD inline uint32_t pv_tau(uint32_t len, int32_t shift, uint32_t t_q, double c, double vhat) {
  const double rho = c <= 0.5 ? 1.06 : c <= 1.0 ? 1.06 + (c - 0.5) * 0.18 : c <= 2.302585092994046
                                                                                ? 1.15 + (c - 1.0) * (0.24 / 1.302585092994046)
                                                                                : 1.39;
  const double denom = rho * c * vhat;
  double tol = denom > 0.0 ? 0.002 * 1e9 / denom : 1e4;  // ns
  tol = tol < static_cast<double>(t_q) ? static_cast<double>(t_q) : tol > 1e4 ? 1e4 : tol;
  uint32_t tau = 1u;
  while (static_cast<double>(tau) * 2.0 <= 2.0 * tol) tau *= 2u;
  uint32_t floor_q = 1u;
  while (floor_q * 2u <= 2u * t_q) floor_q *= 2u;
  const uint32_t slot = 1u << shift;
  if (tau < floor_q) tau = floor_q;
  if (tau < slot) tau = slot;
  if (tau > len / 16u) tau = len / 16u;
  return tau < slot ? slot : tau;
}

// A node's codes and table (render/shade/codes.h). Table row r: instance, surface, n_s, L_s,g (groups), per shadow light
// a, F, R (float bits).
struct PvNodes {
  const uint32_t* slot;           // per batch sample: its entry (render::kPvNone, kPvOverflow)
  const uint32_t* codes;          // the batch's entries
  const uint32_t* table;
  const uint32_t* carried_codes;  // by carried row (rows as carried_class)
  const uint32_t* carried_table;  // header kPvNone: none
  const uint32_t* carried_at;     // the batch's carried rows (null: the batch-offset rows, pointers offset)
  uint32_t code_words, table_words, row_words, bits;
  int32_t groups, lights;

  BALLA_HD bool get(uint32_t s, uint32_t a, uint32_t j, const uint32_t*& c, const uint32_t*& t) const {
    if (j == 0) {
      const size_t r = carried_at != nullptr ? carried_at[a] : a;
      t = carried_table + r * table_words;
      c = carried_codes + r * code_words;
      return t[0] != render::kPvNone;
    }
    const uint32_t e = slot[s];
    if (e >= render::kPvOverflow) return false;
    c = codes + static_cast<size_t>(e) * code_words;
    t = table + static_cast<size_t>(e) * table_words;
    return true;
  }
  BALLA_HD const uint32_t* row(const uint32_t* t, uint32_t r) const { return t + 1 + r * row_words; }
  BALLA_HD uint32_t rows(const uint32_t* t) const { return t[0] & 7u; }
  BALLA_HD float light_word(const uint32_t* t, uint32_t r, int32_t l, int32_t w) const {
    return bits_to_float(row(t, r)[3 + groups + 3 * l + w]);
  }
};

// A uniform node's shadow mask (one cluster, a uniform mask: no edge bit) with key (inst, surf), from its class word's
// shadowed pairs (n popcount(mask) mod 2^12: kClassShadowed's 12 bits since P2-F F6, 13 before; from 4096 pairs masks
// alias, e.g. N = 1024 with 4 lights, and the signature loop below resolves them at O(masks n)) and, where that leaves
// several masks, from its signature (sum_k geometry_term + shadow_term; shade.h). False: none fits.
BALLA_HD inline bool pv_uniform_mask(uint32_t cls, uint64_t sig, uint32_t inst, uint32_t surf, uint32_t n,
                                     int32_t lights, uint32_t& mask) {
  const uint32_t shadowed = cls & render::kClassShadowed;
  uint32_t count = 0;
  for (uint32_t m = 0; m < (1u << lights); ++m) {
    if (((n * pv_popcount(m)) & render::kClassShadowed) != shadowed) continue;
    ++count;
    mask = m;
  }
  if (count <= 1) return count == 1;
  uint64_t geo = 0;
  for (uint32_t k = 0; k < n; ++k) geo += render::geometry_term(k, inst, surf);
  for (uint32_t m = 0; m < (1u << lights); ++m) {
    if (((n * pv_popcount(m)) & render::kClassShadowed) != shadowed) continue;
    uint64_t s = geo;
    for (uint32_t k = 0; k < n; ++k) s += render::shadow_term(k, m);
    if (s == sig) {
      mask = m;
      return true;
    }
  }
  return false;
}

// The two ends of a candidate as P&V reads them (pv_shadow: equal geometry hashes): codes and tables, or for one
// uniform end its key (the other end's single row) and mask. resolve() returns 0 or R0's bits.
struct PvEnds {
  const uint32_t *ca = nullptr, *ta = nullptr, *cb = nullptr, *tb = nullptr;
  uint32_t u_inst = 0, u_surf = 0, u_mask = 0, u_row = 0;
  bool ua = false, ub = false;  // end a / b uniform

  BALLA_HD uint32_t resolve(const PvNodes& v, const uint32_t* cls, const uint64_t* sig, uint32_t n, uint32_t lo,
                            uint32_t hi, uint32_t a, uint32_t j) {
    if ((cls[lo] >> 16) != (cls[hi] >> 16)) return kRtCover;
    const bool ha = v.get(lo, a, j - 1, ca, ta), hb = v.get(hi, a, j, cb, tb);
    ua = !ha && (cls[lo] & kBandEdge) == 0;
    ub = !hb && (cls[hi] & kBandEdge) == 0;
    if ((!ha && !ua) || (!hb && !ub)) return kRtR0 | (!ha && !ua && j == 1 ? kRtCarried : kRtOverflow);
    if ((ha && (ta[0] & render::kPvTableOverflow) != 0) || (hb && (tb[0] & render::kPvTableOverflow) != 0)) {
      return kRtR0 | kRtOverflow;
    }
    if (ua && ub) return kRtR0 | kRtNoTable;
    if (!ua && !ub) return 0;
    // One uniform end: the tabled end has one row on every sub-sample (equal hashes), the uniform end's key.
    const uint32_t* c = ua ? cb : ca;
    const uint32_t* t = ua ? tb : ta;
    const uint32_t r = pv_get_code(c, 0, v.bits) & 7u;
    if (r >= v.rows(t)) return kRtR0 | kRtKey;
    for (uint32_t k = 1; k < n; ++k) {
      if ((pv_get_code(c, k, v.bits) & 7u) != r) return kRtR0 | kRtKey;
    }
    u_row = r;
    u_inst = v.row(t, r)[0];
    u_surf = v.row(t, r)[1];
    const uint32_t s = ua ? lo : hi;
    if (!pv_uniform_mask(cls[s], sig[s], u_inst, u_surf, n, v.lights, u_mask)) return kRtR0 | kRtNoTable;
    return 0;
  }
  // Sub-sample k's key (instance, surface; the environment and overflow codes as 0xFFFFFFFF rows) and mask at end b.
  BALLA_HD void state(const PvNodes& v, bool end_b, uint32_t k, uint32_t& inst, uint32_t& surf, uint32_t& mask) const {
    const bool u = end_b ? ub : ua;
    if (u) {
      inst = u_inst;
      surf = u_surf;
      mask = u_mask;
      return;
    }
    const uint32_t code = pv_get_code(end_b ? cb : ca, k, v.bits);
    const uint32_t r = code & 7u;
    mask = code >> 3;
    if (r < render::kPvRows) {
      const uint32_t* row = v.row(end_b ? tb : ta, r);
      inst = row[0];
      surf = row[1];
    } else {
      inst = r == render::kPvCodeEnv ? render::kInstanceNone : 0xFFFFFFFFu;
      surf = r;
    }
  }
  // Light l's a, R of the row with key (inst, surf) at an end (uniform: R from the mask, a from the other end; no row:
  // false).
  BALLA_HD bool light(const PvNodes& v, bool end_b, uint32_t inst, uint32_t surf, int32_t l, float& a_l,
                      float& r_l) const {
    const bool u = end_b ? ub : ua;
    const uint32_t* t = u ? (end_b ? ta : tb) : (end_b ? tb : ta);
    for (uint32_t r = 0; r < v.rows(t); ++r) {
      if (v.row(t, r)[0] != inst || v.row(t, r)[1] != surf) continue;
      a_l = v.light_word(t, r, l, 0);
      r_l = u ? (((u_mask >> l) & 1u) != 0 ? 0.0f : 1.0f) : v.light_word(t, r, l, 2);
      return true;
    }
    return false;
  }
};

// The edge contrast of light l at an interval (plan §3.1): the largest over the tabled end's rows of
// ln((E + a (1 - R)) / (E - a R)), E the end's value in the light's group plus the dark term (0: no row).
BALLA_HD inline double pv_contrast(const PvNodes& v, const PvEnds& e, const float* irr_a, const float* irr_b, float dark,
                                   const PvParams& p, int32_t l) {
  const bool use_b = e.ua;
  const uint32_t* t = use_b ? e.tb : e.ta;
  const double E = static_cast<double>((use_b ? irr_b : irr_a)[p.light_group[l]]) + static_cast<double>(dark);
  double c = 0.0;
  for (uint32_t r = 0; r < v.rows(t); ++r) {
    const double a = v.light_word(t, r, l, 0), R = v.light_word(t, r, l, 2);
    const double lit = E + a * (1.0 - R), dark_v = E - a * R;
    const double x = dark_v > 0.0 && lit > 0.0 ? pmath::log(lit / dark_v) : 10.0;  // host = device (pmath)
    c = x > c ? x : c;
  }
  return c;
}

// The pixel's shadow-edge displacement bound per second and P&V's plan data (plan §3.4: D_bound = v_hat (b - a), times
// max(1, g_max) on cone pixels under the as-built rates).
struct PvPixelData {
  const uint32_t* pixel;      // the batch's pixel ids
  const uint8_t* level;       // the batch's lattice levels
  const uint8_t* pv_plan;     // by pixel id (null: none)
  const uint32_t* pixel_tile;  // by pixel id (null: no plan v_hat)
  const float* tile_vhat;
  BALLA_HD double vhat(const PvParams& p, uint32_t a) const {
    const uint32_t id = pixel[a];
    double v = tile_vhat != nullptr ? static_cast<double>(tile_vhat[pixel_tile[id]]) : static_cast<double>(p.vhat);
    if (pv_plan != nullptr && !p.obliquity && (pv_plan[id] & kPvCone) != 0) {
      v *= 0.5 * static_cast<double>((pv_plan[id] & kPvGain) + 1);
    }
    return v;
  }
  BALLA_HD bool beyond(uint32_t a) const { return pv_plan != nullptr && (pv_plan[pixel[a]] & kPvBeyond) != 0; }
  // D3b: R4's plan bit (a moving shadow narrower than 2 m D_bound + pitch; PvPlanPass).
  BALLA_HD bool thin(uint32_t a) const { return pv_plan != nullptr && (pv_plan[pixel[a]] & kPvThin) != 0; }
  // D_bound of the pixel's lattice interval (px).
  BALLA_HD double d_bound(const PvParams& p, uint32_t a) const {
    return vhat(p, a) * (static_cast<double>(time::kWindowNs >> level[a]) * 1e-9);
  }
};

// Common inputs of the functors.
struct PvBatch {
  SampleRules rules;
  PvParams p;
  PvNodes v;
  PvPixelData px;
  uint16_t cam;
  const uint32_t* off0;
  const uint32_t* owner;
  const uint8_t* n_log2;  // the batch's log2 N (null: every pixel at the call's N, eligible = SampleConfig::subsamples)
  uint32_t subsamples;     // SampleConfig::subsamples (without n_log2)
  const uint32_t* dt;
  const float* irr;
  const uint64_t* sig;
  const uint32_t* cls;
  int32_t groups;
  uint32_t* route;   // per slot
  float* jstep;      // per slot x groups: J, then (PvPixel) its scan over the pixel's slots
  uint8_t* thin;     // per slot: R2's thin node

  BALLA_HD bool eligible(uint32_t a) const {
    return (n_log2 != nullptr ? (1u << n_log2[a]) : subsamples) == p.n;
  }
  BALLA_HD float dark(uint32_t a) const { return rules.mid.dark_of(px.pixel[a]); }
};

// Step B, R0, J: per slot i of the batch's lattice (sampling.h layout).
struct PvSlots {
  PvBatch b;
  BALLA_HD void operator()(size_t si) const {
    const auto i = static_cast<uint32_t>(si);
    const uint32_t a = b.owner[i], j = i - b.off0[a];
    float* J = b.jstep + static_cast<size_t>(i) * b.groups;
    for (int32_t g = 0; g < b.groups; ++g) J[g] = 0.0f;
    b.route[i] = 0;
    b.thin[i] = 0;
    if (j == 0 || !b.eligible(a)) return;
    const float* ea = b.irr + static_cast<size_t>(i - 1) * b.groups;
    const float* eb = b.irr + static_cast<size_t>(i) * b.groups;
    const uint32_t len = b.dt[i] - b.dt[i - 1];
    bool s = false, splits = false;
    signature_flag(b.rules, b.dark(a), b.rules.refine_signature && b.sig[i] != b.sig[i - 1], len, ea, eb, s, splits);
    if (!(splits && len >= b.rules.min_split && b.rules.max_refine_levels > 0)) return;
    PvEnds e;
    const uint32_t r0 = e.resolve(b.v, b.cls, b.sig, b.p.n, i - 1, i, a, j);
    b.route[i] = kRtCand | r0;
    if (r0 != 0) {
      for (int32_t g = 0; g < b.groups; ++g) J[g] = eb[g] - ea[g];
      return;
    }
    // A shadow candidate: every key's members are equal at both ends, so only the shadow ratios move (plan §3.2):
    // J_g = sum over rows and lights of group g of the ends' mean a times the change of R.
    const uint32_t* t = e.ua ? e.tb : e.ta;
    for (uint32_t r = 0; r < b.v.rows(t); ++r) {
      const uint32_t inst = b.v.row(t, r)[0], surf = b.v.row(t, r)[1];
      for (int32_t l = 0; l < b.p.lights; ++l) {
        float aa = 0, ra = 0, ab = 0, rb = 0;
        if (!e.light(b.v, false, inst, surf, l, aa, ra) || !e.light(b.v, true, inst, surf, l, ab, rb)) continue;
        J[b.p.light_group[l]] += 0.5f * (aa + ab) * (rb - ra);
      }
    }
  }
};

// R2 (plan §3.4): is node s thin? For each light and key present with both bit states, a minority member k with its
// nearest other-state member j1 (distance d1, direction e) and another other-state member within 20 degrees of -e at
// d2, inside the sub-samples' bounding box along -e, with d1 + d2 < D_bound + 1.2 pitch. Nodes past 256 sub-samples
// are thin (the positions held per thread; conservative: they route).
inline constexpr uint32_t kPvThinMaxN = 256;
inline constexpr double kPvConeCos = 0.93969262078590838;  // cos 20 degrees

struct PvThin {
  PvBatch b;
  BALLA_HD void operator()(size_t si) const {
    const auto s = static_cast<uint32_t>(si);
    const uint32_t a = b.owner[s], j = s - b.off0[a], end = b.off0[a + 1];
    if (!b.eligible(a)) return;
    bool need = false;
    for (uint32_t q = s + 1 > 1 ? s - 1 : s; q <= s + 2 && q < end; ++q) {  // the intervals ending at s - 1 ... s + 2
      if (q > b.off0[a] && (b.route[q] & (kRtCand | kRtCover | kRtR0)) == kRtCand) need = true;
    }
    const uint32_t* c = nullptr;
    const uint32_t* t = nullptr;
    if (!need || !b.v.get(s, a, j, c, t)) return;
    b.thin[s] = thin(a, c, t) ? 1 : 0;
  }

  BALLA_HD bool thin(uint32_t a, const uint32_t* c, const uint32_t* t) const {
    const uint32_t n = b.p.n;
    if (n > kPvThinMaxN) return true;
    const double limit = b.px.d_bound(b.p, a) + 1.2 / sqrt(static_cast<double>(n));
    float ux[kPvThinMaxN], uy[kPvThinMaxN];
    float x0 = 1e30f, x1 = -1e30f, y0 = 1e30f, y1 = -1e30f;
    for (uint32_t k = 0; k < n; ++k) {
      const Vec2<float> u = pv_position(b.p, b.cam, b.px.pixel[a], k);
      ux[k] = u.x;
      uy[k] = u.y;
      x0 = u.x < x0 ? u.x : x0;
      x1 = u.x > x1 ? u.x : x1;
      y0 = u.y < y0 ? u.y : y0;
      y1 = u.y > y1 ? u.y : y1;
    }
    for (uint32_t r = 0; r < b.v.rows(t); ++r) {
      for (int32_t l = 0; l < b.p.lights; ++l) {
        uint32_t ones = 0, zeros = 0;
        for (uint32_t k = 0; k < n; ++k) {
          const uint32_t code = pv_get_code(c, k, b.v.bits);
          if ((code & 7u) != r) continue;
          ((code >> (3 + l)) & 1u) != 0 ? ++ones : ++zeros;
        }
        if (ones == 0 || zeros == 0) continue;
        const uint32_t minority = ones <= zeros ? 1u : 0u;
        for (uint32_t k = 0; k < n; ++k) {
          const uint32_t ck = pv_get_code(c, k, b.v.bits);
          if ((ck & 7u) != r || ((ck >> (3 + l)) & 1u) != minority) continue;
          // Its nearest other-state member.
          double d1 = 1e30, ex = 0, ey = 0;
          for (uint32_t q = 0; q < n; ++q) {
            const uint32_t cq = pv_get_code(c, q, b.v.bits);
            if ((cq & 7u) != r || ((cq >> (3 + l)) & 1u) == minority) continue;
            const double dx = static_cast<double>(ux[q] - ux[k]), dy = static_cast<double>(uy[q] - uy[k]);
            const double d = sqrt(dx * dx + dy * dy);
            if (d < d1) {
              d1 = d;
              ex = dx;
              ey = dy;
            }
          }
          if (!(d1 > 0.0) || d1 >= limit) continue;
          ex /= d1;
          ey /= d1;
          // The bounding box's exit along -e.
          double exit = 1e30;
          if (ex > 0.0) exit = static_cast<double>(ux[k] - x0) / ex;
          if (ex < 0.0) exit = static_cast<double>(x1 - ux[k]) / -ex;
          if (ey > 0.0) exit = fmin(exit, static_cast<double>(uy[k] - y0) / ey);
          if (ey < 0.0) exit = fmin(exit, static_cast<double>(y1 - uy[k]) / -ey);
          for (uint32_t q = 0; q < n; ++q) {
            const uint32_t cq = pv_get_code(c, q, b.v.bits);
            if ((cq & 7u) != r || ((cq >> (3 + l)) & 1u) == minority) continue;
            const double wx = static_cast<double>(ux[q] - ux[k]), wy = static_cast<double>(uy[q] - uy[k]);
            const double d2 = sqrt(wx * wx + wy * wy);
            if (!(d2 > 0.0) || d2 >= exit || d1 + d2 >= limit) continue;
            if (-(wx * ex + wy * ey) >= kPvConeCos * d2) return true;
          }
        }
      }
    }
    return false;
  }
};

// Per pixel: J's scan, R6, R2, R5 and the flip counts (into flips[a]; per candidate into slot_flips).
struct PvPixel {
  PvBatch b;
  uint32_t* flips;       // nb + 1 (flips[nb] = 0)
  uint32_t* slot_flips;  // per slot
  size_t nb;

  BALLA_HD void operator()(size_t ai) const {
    const auto a = static_cast<uint32_t>(ai);
    if (a == nb) {
      flips[a] = 0;
      return;
    }
    flips[a] = 0;
    if (!b.eligible(a)) return;
    const uint32_t first = b.off0[a], end = b.off0[a + 1];
    const int32_t G = b.groups;
    for (uint32_t i = first + 1; i < end; ++i) {  // J at each node: the steps of the intervals up to it
      for (int32_t g = 0; g < G; ++g) b.jstep[static_cast<size_t>(i) * G + g] += b.jstep[static_cast<size_t>(i - 1) * G + g];
    }
    // The pre-routing rules in order: R6, R4 and R2 (plan bits, thin nodes), R0 per sub-sample with the flip counts,
    // then R5, whose cause reads the other interval's routes complete.
    for (uint32_t i = first + 1; i < end; ++i) {
      uint32_t rt = b.route[i];
      if ((rt & kRtCand) == 0) continue;
      if (b.px.beyond(a)) rt |= kRtR6;
      if (b.px.thin(a)) rt |= kRtR4;
      for (uint32_t q = i >= first + 2 ? i - 2 : first; q <= i + 1 && q < end; ++q) {
        if (b.thin[q] != 0) rt |= kRtR2;
      }
      b.route[i] = rt;
    }
    uint32_t total = 0;
    for (uint32_t i = first + 1; i < end; ++i) {
      uint32_t rt = b.route[i];
      if ((rt & kRtCand) == 0) continue;
      uint32_t n_flips = 0;
      if ((rt & (kRtCover | kRtR0)) == 0) {
        PvEnds e;
        e.resolve(b.v, b.cls, b.sig, b.p.n, i - 1, i, a, i - first);
        for (uint32_t k = 0; k < b.p.n; ++k) {
          uint32_t ia, sa, ma, ib, sb, mb;
          e.state(b.v, false, k, ia, sa, ma);
          e.state(b.v, true, k, ib, sb, mb);
          if (ia == 0xFFFFFFFFu || ib == 0xFFFFFFFFu) {  // a fifth cluster's code
            rt |= kRtR0 | kRtOverflow;
            break;
          }
          if (ia != ib || sa != sb) {  // equal hashes, unequal keys: a 2^-16 collision
            rt |= kRtR0 | kRtKey;
            break;
          }
          n_flips += pv_popcount((ma ^ mb) & ((1u << b.p.lights) - 1u));
        }
        if ((rt & kRtR0) != 0) n_flips = 0;
      }
      b.route[i] = rt;
      slot_flips[i] = n_flips;
      total += n_flips;
    }
    flips[a] = total;
    if (!(b.rules.midpoint && b.rules.mid_levels > 0)) return;
    // R5: a midpoint triple holding the candidate exceeds eta_L on E - J; its cause, the triple's other interval: an
    // unrouted candidate (1), a routed one (2: R0, R6, R4 or R2), none (3: not a candidate). Shadow mode keeps the
    // flips.
    const float dark = b.dark(a);
    for (uint32_t i = first + 1; i < end; ++i) {
      const uint32_t rt = b.route[i];
      if ((rt & kRtCand) == 0) continue;
      const bool right = i + 1 < end && exceeds(dark, i - 1, i, i + 1);
      const bool left = i >= first + 2 && exceeds(dark, i - 2, i - 1, i);
      if (!right && !left) continue;
      const uint32_t o = b.route[right ? i + 1 : i - 1] & ~(kRtR5 | (3u << kRtR5Shift));
      const uint32_t cause = (o & kRtCand) == 0 ? 3u : (o & (kRtCover | kRtR0 | kRtR6 | kRtR4 | kRtR2)) != 0 ? 2u : 1u;
      b.route[i] = rt | kRtR5 | (cause << kRtR5Shift);
    }
  }

  // eta_L's triple (lo, m, hi) on E - J (sampling.h midpoint_exceeds).
  BALLA_HD bool exceeds(float dark, uint32_t lo, uint32_t m, uint32_t hi) const {
    float x[3][kMaxGroups];
    const uint32_t s[3] = {lo, m, hi};
    for (int32_t q = 0; q < 3; ++q) {
      for (int32_t g = 0; g < b.groups; ++g) {
        const size_t w = static_cast<size_t>(s[q]) * b.groups + g;
        x[q][g] = b.irr[w] - b.jstep[w];
      }
    }
    return midpoint_exceeds(b.rules.mid, dark, x[0], x[1], x[2]);
  }
};

// One shadow flip (plan §3.1 Steps D-F): sub-sample k, light l, before-state bit `before` at the interval's start, its
// bracket [lo, hi] in cells of tau from t0 (the interval's start, ns from t_w; state `before` at lo, the other at hi),
// its group's flips [gb, ge), the probe cells of the round (kPvNoCell: none) and the cell a verify predicted. D3b: its
// group's index grp (PvFit), its rank among the group's non-seeds (the waves), the stage that predicted it, and loc,
// whether it was localized at the group's last fit (PvFitGroups writes it; PvEmit's local stage reads it).
inline constexpr uint32_t kPvNoCell = 0xFFFFFFFFu;
inline constexpr uint8_t kPhSeed = 0, kPhWait = 1, kPhVerify = 2, kPhDown = 3, kPhUp = 4, kPhBisect = 5, kPhDone = 6,
                         kPhDrop = 7;
inline constexpr uint8_t kFlSeed = 1, kFlPredicted = 2, kFlHit = 4, kFlVerified = 8, kFlR3 = 16;

struct PvFlip {
  uint32_t slot, a, t0, tau, lo, hi, p0, p1, step, gb, ge, grp, inst, surf;
  float ux, uy;
  uint16_t k, probes, rank;
  uint8_t l, before, phase, flags, stage, loc;
};

// A group's fits (plan §3.1's ladder), from its localized flips at their cells' centres: u centred at (cx, cy) and
// scaled by s, t (ns from t_w) at tc and by ts; the affine stage's coefficients aff (on [1, vx, vy]); the selected
// stage and, for the projective and quadratic ones, its coefficients c (on [1, vx, vy] over [1, bx, by] (t' (1 + b.v)
// = a0 + a.v), or [1, vx, vy, vx^2, vx vy, vy^2]); n_loc localized flips, n_done resolved ones (localized or dropped;
// the waves' measure), pending seeds (D3a).
struct PvFit {
  double cx, cy, s, tc, ts;
  double aff[3];
  double c[6];
  uint32_t n_loc, n_done, pending;
  uint8_t stage, pad[3];
};

// The P2-D D3b ladder's arithmetic: fp64 normal equations of at most six unknowns (Cholesky; host = device, no
// contraction), the leave-one-out residuals from the hat matrix.
struct PvNormal {
  double a[6][6];
  double r[6];
  int p;
  BALLA_HD void reset(int n) {
    p = n;
    for (int i = 0; i < 6; ++i) {
      r[i] = 0.0;
      for (int j = 0; j < 6; ++j) a[i][j] = 0.0;
    }
  }
  BALLA_HD void add(const double* x, double y) {
    for (int i = 0; i < p; ++i) {
      r[i] += x[i] * y;
      for (int j = 0; j <= i; ++j) a[i][j] += x[i] * x[j];
    }
  }
  // In place: the lower Cholesky factor of a (false: a pivot below 1e-10 of the largest diagonal, rank-deficient).
  BALLA_HD bool factor() {
    double big = 0.0;
    for (int i = 0; i < p; ++i) big = a[i][i] > big ? a[i][i] : big;
    if (!(big > 0.0)) return false;
    for (int j = 0; j < p; ++j) {
      double d = a[j][j];
      for (int k = 0; k < j; ++k) d -= a[j][k] * a[j][k];
      if (!(d > 1e-10 * big)) return false;
      a[j][j] = sqrt(d);
      for (int i = j + 1; i < p; ++i) {
        double s = a[i][j];
        for (int k = 0; k < j; ++k) s -= a[i][k] * a[j][k];
        a[i][j] = s / a[j][j];
      }
    }
    return true;
  }
  // x = A^-1 b with the factor.
  BALLA_HD void solve(const double* b, double* x) const {
    double y[6] = {};
    for (int i = 0; i < p; ++i) {
      double s = b[i];
      for (int k = 0; k < i; ++k) s -= a[i][k] * y[k];
      y[i] = s / a[i][i];
    }
    for (int i = p - 1; i >= 0; --i) {  // in place in y (the back substitution reads the solved entries above i)
      double s = y[i];
      for (int k = i + 1; k < p; ++k) s -= a[k][i] * y[k];
      y[i] = s / a[i][i];
    }
    for (int i = 0; i < p; ++i) x[i] = y[i];
  }
};

// A stage's regressors at scaled position (vx, vy) (the projective one's last two need the scaled time y).
BALLA_HD inline int pv_regressors(uint8_t stage, double vx, double vy, double y, double* x) {
  x[0] = 1.0;
  x[1] = vx;
  x[2] = vy;
  if (stage == kStProj) {
    x[3] = -y * vx;
    x[4] = -y * vy;
    return 5;
  }
  if (stage == kStQuad) {
    x[3] = vx * vx;
    x[4] = vx * vy;
    x[5] = vy * vy;
    return 6;
  }
  return 3;
}
// A global stage's scaled prediction at (vx, vy) from coefficients c (false: the projective denominator below 1e-3).
BALLA_HD inline bool pv_global(uint8_t stage, const double* c, double vx, double vy, double& y) {
  double x[6] = {};
  const int p = pv_regressors(stage, vx, vy, 0.0, x);
  if (stage == kStProj) {
    const double den = 1.0 + c[3] * vx + c[4] * vy;
    if (!(fabs(den) >= 1e-3)) return false;
    y = (c[0] + c[1] * vx + c[2] * vy) / den;
    return true;
  }
  y = 0.0;
  for (int i = 0; i < p; ++i) y += c[i] * x[i];
  return true;
}
// A localized flip's time: its cell's centre (ns from t_w).
BALLA_HD inline double pv_time(const PvFlip& x) {
  return static_cast<double>(x.t0) + (static_cast<double>(x.lo) + 0.5) * static_cast<double>(x.tau);
}

// The local stage at position (ux, uy): the affine fit t = a + g . (u_q - u) over the kPvLocalK localized flips of
// [gb, ge) nearest u (loc set; `skip` excluded; ties to the lower index), returning a (ns) and the gradient g (ns/px).
// False: fewer than 3 neighbours or a degenerate fit.
BALLA_HD inline bool pv_local(const PvFlip* flips, uint32_t gb, uint32_t ge, float ux, float uy, uint32_t skip,
                              double& t, double& gx, double& gy) {
  uint32_t nn[kPvLocalK];
  double dd[kPvLocalK];
  uint32_t m = 0;
  for (uint32_t q = gb; q < ge; ++q) {
    if (flips[q].loc == 0 || q == skip) continue;
    const double dx = static_cast<double>(flips[q].ux) - static_cast<double>(ux);
    const double dy = static_cast<double>(flips[q].uy) - static_cast<double>(uy);
    const double d = dx * dx + dy * dy;
    if (m == kPvLocalK && !(d < dd[m - 1])) continue;
    uint32_t j = m < kPvLocalK ? m++ : m - 1;
    while (j > 0 && d < dd[j - 1]) {
      dd[j] = dd[j - 1];
      nn[j] = nn[j - 1];
      --j;
    }
    dd[j] = d;
    nn[j] = q;
  }
  if (m < 3) return false;
  // Centred at u, times relative to the first neighbour's (conditioning), unscaled px.
  const double t_ref = pv_time(flips[nn[0]]);
  PvNormal ne;
  ne.reset(3);
  for (uint32_t j = 0; j < m; ++j) {
    const PvFlip& q = flips[nn[j]];
    const double x[3] = {1.0, static_cast<double>(q.ux) - static_cast<double>(ux),
                         static_cast<double>(q.uy) - static_cast<double>(uy)};
    ne.add(x, pv_time(q) - t_ref);
  }
  if (!ne.factor()) return false;
  double c[3];
  ne.solve(ne.r, c);
  t = t_ref + c[0];
  gx = c[1];
  gy = c[2];
  return true;
}

// Step D's list: per pixel, its flips from offset off[a], in (light, direction, interval, k) order. A group (one
// arrival-time field, Step E) is a run of consecutive lattice intervals with flips of one (light, direction): a gap
// starts another passage (a second edge, or the same edge back), which one affine field would not hold.
struct PvFill {
  PvBatch b;
  const uint32_t* off;
  const uint32_t* slot_flips;
  PvFlip* flips;
  BALLA_HD void operator()(size_t ai) const {
    const auto a = static_cast<uint32_t>(ai);
    if (off[a + 1] == off[a]) return;
    const uint32_t first = b.off0[a], end = b.off0[a + 1];
    const double vhat = b.px.vhat(b.p, a);
    uint32_t f = off[a];
    for (int32_t l = 0; l < b.p.lights; ++l) {
      for (uint32_t dir = 0; dir < 2; ++dir) {
        uint32_t gb = f, last = 0;  // the group's first flip, the interval of its last
        for (uint32_t i = first + 1; i < end; ++i) {
          if (slot_flips[i] == 0) continue;
          PvEnds e;
          e.resolve(b.v, b.cls, b.sig, b.p.n, i - 1, i, a, i - first);
          const float* ea = b.irr + static_cast<size_t>(i - 1) * b.groups;
          const float* eb = b.irr + static_cast<size_t>(i) * b.groups;
          const uint32_t len = b.dt[i] - b.dt[i - 1];
          const uint32_t tau = pv_tau(len, b.rules.shift, b.p.t_q, pv_contrast(b.v, e, ea, eb, b.dark(a), b.p, l), vhat);
          for (uint32_t k = 0; k < b.p.n; ++k) {
            uint32_t ia, sa, ma, ib, sb, mb;
            e.state(b.v, false, k, ia, sa, ma);
            e.state(b.v, true, k, ib, sb, mb);
            const uint32_t bit_a = (ma >> l) & 1u, bit_b = (mb >> l) & 1u;
            if (bit_a == bit_b || bit_a != dir) continue;
            if (f > gb && last + 1 < i) {  // a run of intervals ends: the next passage is a group of its own
              for (uint32_t q = gb; q < f; ++q) flips[q].ge = f;
              gb = f;
            }
            last = i;
            const Vec2<float> u = pv_position(b.p, b.cam, b.px.pixel[a], k);
            PvFlip& x = flips[f++];
            x = PvFlip{i,   a,   b.dt[i - 1], tau, 0u, len / tau, kPvNoCell, kPvNoCell, 1u, gb, 0u, 0u, ia, sa,
                       u.x, u.y, static_cast<uint16_t>(k), 0, 0, static_cast<uint8_t>(l), static_cast<uint8_t>(dir),
                       kPhWait, 0, kStNone, 0};
          }
        }
        for (uint32_t q = gb; q < f; ++q) flips[q].ge = f;
      }
    }
  }
};

// Step E: up to three seeds per group, farthest-point in u (the farthest from the centroid, then the farthest from it,
// then the largest smallest distance to both; ties to the lower index); a group of three or fewer is all seeds. Runs
// on each group's first flip, which also writes the non-seeds' ranks in group order (D3b's waves) and marks itself
// (first[f]: 1 on a group's first flip, else 0; first[nf] = 0) for the group indices' scan.
struct PvSeeds {
  PvFlip* flips;
  uint32_t* first;
  size_t nf;
  BALLA_HD void operator()(size_t fi) const {
    const auto f = static_cast<uint32_t>(fi);
    if (fi == nf) {
      first[fi] = 0;
      return;
    }
    first[fi] = flips[f].gb == f ? 1u : 0u;
    if (flips[f].gb != f) return;
    pick(f);
    uint32_t r = 0;
    for (uint32_t q = f; q < flips[f].ge; ++q) {
      if ((flips[q].flags & kFlSeed) == 0) flips[q].rank = static_cast<uint16_t>(r < 0xFFFFu ? r++ : r);
    }
  }
  BALLA_HD void pick(uint32_t f) const {
    const uint32_t gb = f, ge = flips[f].ge;
    if (ge - gb <= 3) {
      for (uint32_t q = gb; q < ge; ++q) mark(q);
      return;
    }
    double cx = 0, cy = 0;
    for (uint32_t q = gb; q < ge; ++q) {
      cx += static_cast<double>(flips[q].ux);
      cy += static_cast<double>(flips[q].uy);
    }
    cx /= static_cast<double>(ge - gb);
    cy /= static_cast<double>(ge - gb);
    uint32_t s[3] = {gb, gb, gb};
    for (int32_t n = 0; n < 3; ++n) {
      double best = -1.0;
      for (uint32_t q = gb; q < ge; ++q) {
        if ((n > 0 && q == s[0]) || (n > 1 && q == s[1])) continue;
        double d = 1e300;
        if (n == 0) d = dist2(q, cx, cy);
        if (n >= 1) d = dist2(q, static_cast<double>(flips[s[0]].ux), static_cast<double>(flips[s[0]].uy));
        if (n == 2) d = fmin(d, dist2(q, static_cast<double>(flips[s[1]].ux), static_cast<double>(flips[s[1]].uy)));
        if (d > best) {
          best = d;
          s[n] = q;
        }
      }
      mark(s[n]);
    }
  }
  BALLA_HD double dist2(uint32_t q, double x, double y) const {
    const double dx = static_cast<double>(flips[q].ux) - x, dy = static_cast<double>(flips[q].uy) - y;
    return dx * dx + dy * dy;
  }
  BALLA_HD void mark(uint32_t q) const {
    flips[q].phase = kPhSeed;
    flips[q].flags |= kFlSeed;
  }
};

// D3b: each flip's group index (first: PvSeeds' marks scanned; the group's first flip holds its index).
struct PvGroupIds {
  PvFlip* flips;
  const uint32_t* first;
  BALLA_HD void operator()(size_t f) const { flips[f].grp = first[flips[f].gb]; }
};

// The prediction of flip x's time (ns from t_w) by its group's selected stage (plan §3.1's ladder), and the stage that
// gave it: the projective form falls back to the local stage where its denominator vanishes (|1 + b.v| < 1e-3), the
// local stage to the affine one where its neighbours are degenerate.
BALLA_HD inline double pv_predict(const PvFit& g, const PvFlip* flips, const PvFlip& x, uint8_t& stage) {
  const double vx = (static_cast<double>(x.ux) - g.cx) / g.s, vy = (static_cast<double>(x.uy) - g.cy) / g.s;
  stage = g.stage;
  double y = 0.0;
  if ((stage == kStProj || stage == kStQuad) && pv_global(stage, g.c, vx, vy, y)) return g.tc + g.ts * y;
  if (stage == kStProj || stage == kStQuad) stage = g.n_loc >= kPvLocalK ? kStLocal : kStAffine;
  if (stage == kStLocal) {
    double t, gx, gy;
    if (pv_local(flips, x.gb, x.ge, x.ux, x.uy, kPvNoCell, t, gx, gy)) return t;
    stage = kStAffine;
  }
  return g.tc + g.ts * (g.aff[0] + g.aff[1] * vx + g.aff[2] * vy);
}

// The rounds (plan §3.1 Step F). PvEmit: each open flip's probe cells this round (nreq[f]: 0-2; nreq[nf] = 0). D3b: a
// waiting flip is promoted to verification once its group has a fit and its rank among the group's non-seeds is below
// kPvWave times the group's resolved flips (the waves), predicted once, by the selected stage of the ladder.
struct PvEmit {
  PvFlip* flips;
  const PvFit* fits;
  const uint32_t* route;
  uint32_t* nreq;
  size_t nf;
  BALLA_HD void operator()(size_t fi) const {
    if (fi == nf) {
      nreq[fi] = 0;
      return;
    }
    PvFlip& x = flips[fi];
    uint32_t n = 0;
    x.p0 = x.p1 = kPvNoCell;
    if (x.phase < kPhDone && (route[x.slot] & kRtR1) != 0) x.phase = kPhDrop;
    if (x.phase == kPhWait) {
      const PvFit& g = fits[x.grp];
      if (g.stage != kStNone) {
        if (static_cast<uint32_t>(x.rank) < kPvWave * g.n_done) {
          x.phase = kPhVerify;
          x.flags |= kFlPredicted;
        }
      } else if (g.pending == 0) {
        x.phase = kPhBisect;
      }
    }
    if (x.phase == kPhVerify) {
      const double t = pv_predict(fits[x.grp], flips, x, x.stage);
      double c = floor((t - static_cast<double>(x.t0)) / static_cast<double>(x.tau));
      c = c >= static_cast<double>(x.lo) ? c : static_cast<double>(x.lo);  // NaN (a degenerate fit): lo
      c = c > static_cast<double>(x.hi - 1) ? static_cast<double>(x.hi - 1) : c;
      x.step = static_cast<uint32_t>(c);  // the verified cell
      if (x.step > x.lo) x.p0 = x.step;
      if (x.step + 1 < x.hi) x.p1 = x.step + 1;
    } else if (x.phase == kPhDown || x.phase == kPhUp) {
      const uint32_t p = x.phase == kPhDown ? (x.hi - x.lo > x.step ? x.hi - x.step : x.lo)
                                            : (x.hi - x.lo > x.step ? x.lo + x.step : x.hi);
      if (p == x.lo || p == x.hi) {
        x.phase = kPhBisect;
      } else {
        x.p0 = p;
      }
    }
    if (x.phase == kPhSeed || x.phase == kPhBisect) {
      if (x.hi - x.lo <= 1) {
        x.phase = kPhDone;
      } else {
        x.p0 = x.lo + (x.hi - x.lo) / 2;
      }
    }
    n = (x.p0 != kPvNoCell ? 1u : 0u) + (x.p1 != kPvNoCell ? 1u : 0u);
    nreq[fi] = n;
  }
};

// The probe requests of the round (one sub-sample at a pose slot of the window clock).
struct PvRequests {
  const PvFlip* flips;
  const uint32_t* nreq;
  const uint32_t* off;
  const uint32_t* pixel;  // the batch's pixel ids
  uint16_t cam;
  int32_t shift;
  SampleRequest* requests;
  BALLA_HD void operator()(size_t fi) const {
    if (nreq[fi] == 0) return;
    const PvFlip& x = flips[fi];
    uint32_t o = off[fi];
    const uint32_t cells[2] = {x.p0, x.p1};
    for (const uint32_t p : cells) {
      if (p == kPvNoCell) continue;
      const uint32_t t = x.t0 + p * x.tau;
      requests[o++] = {cam, static_cast<uint16_t>(static_cast<uint32_t>(backend::RequestKind::kProbe) |
                                                  (static_cast<uint32_t>(x.l) << render::kProbeLightShift)),
                       pixel[x.a], x.k, 1u, static_cast<int32_t>(t), t >> shift};
    }
  }
};

// The round's results: brackets, gallops, R1 (a probe's key equal to neither end's, or after-state at the verified
// cell's start and before-state at its end: three or more flips), and the probe counts by kind.
struct PvConsume {
  PvFlip* flips;
  const uint32_t* nreq;
  const uint32_t* off;
  const render::ProbeOut* out;
  uint32_t* route;
  unsigned long long* counts;
  BALLA_HD void operator()(size_t fi) const {
    const uint32_t n = nreq[fi];
    if (n == 0) return;
    PvFlip& x = flips[fi];
    const render::ProbeOut* r = out + off[fi];
    const int kind = x.phase == kPhSeed ? 0
                     : x.phase == kPhVerify ? 2
                     : (x.flags & kFlPredicted) != 0 ? 3 : 1;
    // One counter address per call (add_count aggregates over the coalesced warp into rank 0's address).
    for (int q = 0; q < 4; ++q) add_count(&counts[kPvProbeKind + q], q == kind ? n : 0u);
    x.probes = static_cast<uint16_t>(x.probes + n);
    for (uint32_t q = 0; q < n; ++q) {
      if (r[q].instance != x.inst || r[q].surface != x.surf) {
        atomic_or(&route[x.slot], kRtR1);
        x.phase = kPhDrop;
        return;
      }
    }
    const uint32_t before = x.before, after = before ^ 1u;
    const auto bit = [&](uint32_t q) { return (r[q].mask >> x.l) & 1u; };
    if (x.phase == kPhSeed || x.phase == kPhBisect) {
      (bit(0) == before ? x.lo : x.hi) = x.p0;
    } else if (x.phase == kPhVerify) {
      const uint32_t c = x.step;
      const uint32_t s0 = x.p0 != kPvNoCell ? bit(0) : before;
      const uint32_t s1 = x.p1 != kPvNoCell ? bit(x.p0 != kPvNoCell ? 1 : 0) : after;
      const bool first = (x.flags & kFlVerified) == 0;
      x.flags |= kFlVerified;
      if (s0 == before && s1 == after) {
        x.lo = c;
        x.hi = c + 1;
        if (first) x.flags |= kFlHit;
      } else if (s0 == after && s1 == before) {
        atomic_or(&route[x.slot], kRtR1);
        x.phase = kPhDrop;
        return;
      } else if (s0 == after) {
        x.hi = c;
        x.phase = kPhDown;
        x.step = 1;
      } else {
        x.lo = c + 1;
        x.phase = kPhUp;
        x.step = 1;
      }
    } else if (x.phase == kPhDown) {
      if (bit(0) == after) {
        x.hi = x.p0;
        x.step *= 2;
      } else {
        x.lo = x.p0;
        x.phase = kPhBisect;
      }
    } else if (x.phase == kPhUp) {
      if (bit(0) == before) {
        x.lo = x.p0;
        x.step *= 2;
      } else {
        x.hi = x.p0;
        x.phase = kPhBisect;
      }
    }
    if (x.hi - x.lo <= 1) x.phase = kPhDone;
  }
};

// The group's fits over its localized flips (each at its cell's centre), on its first flip (plan §3.1's ladder): the
// affine stage once n_loc >= 3 and well conditioned (else no fit), the local and projective stages once n_loc >= 6,
// the quadratic once n_loc >= 10; the selected stage is the one of the smallest leave-one-out RMS over at most kPvLoo
// localized flips spread over the group (the hat matrix for the global stages, Sherman-Morrison for the projective
// ratio, the six nearest others for the local one; ties to the simpler stage). Also: each flip's loc, the resolved
// flips (the waves) and the pending seeds: a seed not yet localized or dropped, a seed whose interval R1 routed this
// round counting as dropped (PvEmit drops it next), so that the group's waiting flips do not wait on it.
struct PvFitGroups {
  PvFlip* flips;
  const uint32_t* route;
  PvFit* fits;
  uint32_t stages;  // PvParams::stages
  BALLA_HD void operator()(size_t fi) const {
    const auto f = static_cast<uint32_t>(fi);
    if (flips[f].gb != f) return;
    const uint32_t ge = flips[f].ge;
    PvFit g{};
    uint32_t n = 0, dropped = 0;
    double cx = 0, cy = 0, tc = 0;
    for (uint32_t q = f; q < ge; ++q) {
      PvFlip& x = flips[q];
      const bool r1 = (route[x.slot] & kRtR1) != 0;
      if ((x.flags & kFlSeed) != 0 && x.phase < kPhDone && !r1) ++g.pending;
      dropped += x.phase == kPhDrop || (x.phase < kPhDone && r1) ? 1u : 0u;
      x.loc = x.phase == kPhDone ? 1 : 0;
      if (x.phase != kPhDone) continue;
      ++n;
      cx += static_cast<double>(x.ux);
      cy += static_cast<double>(x.uy);
      tc += pv_time(x);
    }
    g.n_loc = n;
    g.n_done = n + dropped;
    if (n >= 3) fit(f, ge, n, cx / n, cy / n, tc / n, g);
    fits[flips[f].grp] = g;
  }

  BALLA_HD void fit(uint32_t f, uint32_t ge, uint32_t n, double cx, double cy, double tc, PvFit& g) const {
    double s2 = 0, t2 = 0;
    for (uint32_t q = f; q < ge; ++q) {
      if (flips[q].loc == 0) continue;
      const double dx = static_cast<double>(flips[q].ux) - cx, dy = static_cast<double>(flips[q].uy) - cy;
      const double dt = pv_time(flips[q]) - tc;
      s2 += dx * dx + dy * dy;
      t2 += dt * dt;
    }
    g.cx = cx;
    g.cy = cy;
    g.tc = tc;
    g.s = sqrt(s2 / n);
    g.ts = t2 > 0.0 ? sqrt(t2 / n) : static_cast<double>(flips[f].tau);
    if (!(g.s > 0.0)) return;
    PvNormal aff;
    if (!solve_stage(f, ge, kStAffine, g, aff, g.aff)) return;
    g.stage = kStAffine;
    // The candidates and the leave-one-out sample: every step-th localized flip, at most kPvLoo.
    const bool local = n >= kPvLocalK && (stages & (1u << kStLocal)) != 0;
    PvNormal proj, quad;
    double c_proj[6] = {}, c_quad[6] = {};
    const bool pr = n >= 6 && (stages & (1u << kStProj)) != 0 && solve_stage(f, ge, kStProj, g, proj, c_proj) &&
                    proj_regular(f, ge, g, c_proj);
    const bool qu = n >= 10 && (stages & (1u << kStQuad)) != 0 && solve_stage(f, ge, kStQuad, g, quad, c_quad);
    if (!local && !pr && !qu) return;
    const uint32_t step = (n + kPvLoo - 1) / kPvLoo;
    double sse[5] = {0, 0, 0, 0, 0};
    uint32_t j = 0;
    for (uint32_t q = f; q < ge; ++q) {
      if (flips[q].loc == 0) continue;
      if ((j++ % step) != 0) continue;
      const PvFlip& x = flips[q];
      const double vx = (static_cast<double>(x.ux) - g.cx) / g.s, vy = (static_cast<double>(x.uy) - g.cy) / g.s;
      const double y = (pv_time(x) - g.tc) / g.ts;
      sse[kStAffine] += loo(kStAffine, aff, g.aff, vx, vy, y);
      if (pr) sse[kStProj] += loo(kStProj, proj, c_proj, vx, vy, y);
      if (qu) sse[kStQuad] += loo(kStQuad, quad, c_quad, vx, vy, y);
      if (local) {
        double t, gx, gy;
        const double e = pv_local(flips, f, ge, x.ux, x.uy, q, t, gx, gy) ? (pv_time(x) - t) / g.ts : 1e150;
        sse[kStLocal] += e * e;
      }
    }
    const bool on[5] = {false, true, local, pr, qu};
    for (uint8_t st = kStLocal; st <= kStQuad; ++st) {
      if (on[st] && sse[st] < sse[g.stage]) g.stage = st;
    }
    const double* c = g.stage == kStProj ? c_proj : c_quad;
    for (int i = 0; i < 6; ++i) g.c[i] = c[i];
  }

  // A global stage's normal equations over the localized flips (scaled), solved into c (false: rank-deficient).
  BALLA_HD bool solve_stage(uint32_t f, uint32_t ge, uint8_t stage, const PvFit& g, PvNormal& ne, double* c) const {
    double x[6] = {};
    ne.reset(pv_regressors(stage, 0.0, 0.0, 0.0, x));
    for (uint32_t q = f; q < ge; ++q) {
      if (flips[q].loc == 0) continue;
      const double vx = (static_cast<double>(flips[q].ux) - g.cx) / g.s;
      const double vy = (static_cast<double>(flips[q].uy) - g.cy) / g.s;
      const double y = (pv_time(flips[q]) - g.tc) / g.ts;
      pv_regressors(stage, vx, vy, y, x);
      ne.add(x, y);
    }
    if (!ne.factor()) return false;
    ne.solve(ne.r, c);
    return true;
  }
  // F020's form needs 1 + b.v away from 0 at every localized flip (plan §3.5).
  BALLA_HD bool proj_regular(uint32_t f, uint32_t ge, const PvFit& g, const double* c) const {
    for (uint32_t q = f; q < ge; ++q) {
      if (flips[q].loc == 0) continue;
      const double vx = (static_cast<double>(flips[q].ux) - g.cx) / g.s;
      const double vy = (static_cast<double>(flips[q].uy) - g.cy) / g.s;
      if (!(fabs(1.0 + c[3] * vx + c[4] * vy) >= 1e-3)) return false;
    }
    return true;
  }
  // The squared leave-one-out residual of the flip at (vx, vy, y) under a global stage (factored ne, coefficients c):
  // the linear stages' r / (1 - h); the projective one's ratio with the coefficients downdated by Sherman-Morrison.
  BALLA_HD static double loo(uint8_t stage, const PvNormal& ne, const double* c, double vx, double vy, double y) {
    double x[6] = {}, z[6] = {};
    const int p = pv_regressors(stage, vx, vy, y, x);
    ne.solve(x, z);
    double h = 0.0, fit = 0.0;
    for (int i = 0; i < p; ++i) {
      h += x[i] * z[i];
      fit += x[i] * c[i];
    }
    if (!(1.0 - h > 1e-9)) return 1e300;
    const double r = (y - fit) / (1.0 - h);
    if (stage != kStProj) return r * r;
    double c1[6];
    for (int i = 0; i < p; ++i) c1[i] = c[i] - z[i] * r;
    double yp = 0.0;
    if (!pv_global(kStProj, c1, vx, vy, yp)) return 1e300;
    return (y - yp) * (y - yp);
  }
};

// The flips of pixel range [lo, hi) are sorted by (light, direction, interval, k) (PvFill): the first index at or
// after (l, dir, slot, k), by bisection.
template <class F>
BALLA_HD inline uint32_t pv_lower(const F* flips, uint32_t lo, uint32_t hi, uint32_t l, uint32_t dir, uint32_t slot,
                                  uint32_t k) {
  const uint64_t key = (static_cast<uint64_t>(l) << 61) | (static_cast<uint64_t>(dir) << 60) |
                       (static_cast<uint64_t>(slot) << 16) | k;
  while (lo < hi) {
    const uint32_t m = lo + (hi - lo) / 2;
    const F& x = flips[m];
    const uint64_t km = (static_cast<uint64_t>(x.l) << 61) | (static_cast<uint64_t>(x.before & 1u) << 60) |
                        (static_cast<uint64_t>(x.slot) << 16) | x.k;
    if (km < key) {
      lo = m + 1;
    } else {
      hi = m;
    }
  }
  return lo;
}

// R3 (plan §3.4; after the rounds, per localized flip x of interval I): with W = 2 D_bound(I) + pitch, D_bound(I) =
// v_hat (b - a), it fires when a localized flip of the other direction of x's (key, light) in I lies within W along
// x's edge normal n and within W along the tangent: n the unit gradient of the local affine fit of x's group's
// localized flip times around u_x (its six nearest, x included), the Euclidean distance where that fit is degenerate.
// Each direction's flips test with their own normal (a thin lit gap is a thin shadow with the roles swapped). A
// firing flip is marked (kFlR3) for the propagation.
struct PvR3 {
  PvBatch b;
  PvFlip* flips;
  const uint32_t* off;  // per batch pixel: its flips' offsets
  BALLA_HD void operator()(size_t fi) const {
    PvFlip& x = flips[fi];
    if (x.phase != kPhDone) return;
    const double len = static_cast<double>(b.dt[x.slot] - b.dt[x.slot - 1]);
    const double w = 2.0 * b.px.vhat(b.p, x.a) * len * 1e-9 + 1.0 / sqrt(static_cast<double>(b.p.n));
    double t, gx, gy;
    bool frame = pv_local(flips, x.gb, x.ge, x.ux, x.uy, kPvNoCell, t, gx, gy);
    const double gn = frame ? sqrt(gx * gx + gy * gy) : 0.0;
    frame = frame && gn > 0.0;
    const double nx = frame ? gx / gn : 0.0, ny = frame ? gy / gn : 0.0;
    const uint32_t end = off[x.a + 1];
    for (uint32_t q = pv_lower(flips, off[x.a], end, x.l, x.before ^ 1u, x.slot, 0); q < end; ++q) {
      const PvFlip& o = flips[q];
      if (o.l != x.l || o.before == x.before || o.slot != x.slot) break;
      if (o.phase != kPhDone || o.inst != x.inst || o.surf != x.surf) continue;
      const double dx = static_cast<double>(o.ux) - static_cast<double>(x.ux);
      const double dy = static_cast<double>(o.uy) - static_cast<double>(x.uy);
      const bool near = frame ? fabs(nx * dx + ny * dy) < w && fabs(-ny * dx + nx * dy) < w : dx * dx + dy * dy < w * w;
      if (!near) continue;
      x.flags |= kFlR3;
      atomic_or(&b.route[x.slot], kRtR3);
      return;
    }
  }
};

// R3's propagation: the intervals next to an R3-firing one (I - 1, I + 1 of the pixel) that hold flips of a firing
// flip's (key, light) route too (R3p): a feature's arrival and departure intervals hold flips of one direction only.
struct PvR3Prop {
  const PvFlip* flips;
  const uint32_t* off;
  const uint32_t* off0;  // per batch pixel: its lattice slots
  uint32_t* route;
  BALLA_HD void operator()(size_t fi) const {
    const PvFlip& x = flips[fi];
    const uint32_t lo = off[x.a], hi = off[x.a + 1];
    const uint32_t nbr[2] = {x.slot - 1, x.slot + 1};
    for (const uint32_t s : nbr) {
      if (s <= off0[x.a] || s >= off0[x.a + 1]) continue;
      for (uint32_t dir = 0; dir < 2; ++dir) {
        for (uint32_t q = pv_lower(flips, lo, hi, x.l, dir, s, 0); q < hi; ++q) {
          const PvFlip& o = flips[q];
          if (o.l != x.l || o.before != dir || o.slot != s) break;
          if ((o.flags & kFlR3) == 0 || o.inst != x.inst || o.surf != x.surf) continue;
          atomic_or(&route[x.slot], kRtR3p);
          return;
        }
      }
    }
  }
};

// After the rounds: each flip's outcome into its interval's counts (slot_stats: per slot flips, localized, probes and
// the would-be nodes, PvNodeCount) and the outcome counters; D3b: by the stage that predicted it.
struct PvTally {
  const PvFlip* flips;
  uint32_t* slot_stats;
  unsigned long long* counts;
  BALLA_HD void operator()(size_t fi) const {
    const PvFlip& x = flips[fi];
    const bool done = x.phase == kPhDone;
    uint32_t* w = slot_stats + 4 * static_cast<size_t>(x.slot);
    atomic_add(&w[0], 1u);
    atomic_add(&w[1], done ? 1u : 0u);
    atomic_add(&w[2], x.probes);
    count_if(&counts[kPvSeeds], done && (x.flags & kFlSeed) != 0);
    count_if(&counts[kPvPredicted], done && (x.flags & kFlPredicted) != 0);
    count_if(&counts[kPvVerifyHits], done && (x.flags & kFlHit) != 0);
    count_if(&counts[kPvDropped], x.phase == kPhDrop);
    count_if(&counts[kPvUnfinished], x.phase < kPhDone);
    const bool pred = done && (x.flags & kFlPredicted) != 0;
    for (uint8_t st = kStAffine; st <= kStQuad; ++st) {  // one address per call (add_count aggregates over a warp)
      const bool mine = pred && x.stage == st;
      count_if(&counts[kPvStage + st - 1], mine);
      add_count(&counts[kPvStage + 4 + st - 1], mine ? x.probes : 0u);
      count_if(&counts[kPvStage + 8 + st - 1], mine && (x.flags & kFlHit) != 0);
    }
  }
};

// D3b: what the state check reads of a flip, kept through the refinement rounds (16 B; the flips' order): its key in
// the pixel's list, state bit 0 its before-state, bit 1 localized, and its bracket's ends (ns from t_w).
struct PvBracket {
  uint32_t slot;
  uint16_t k;
  uint8_t l, before;  // before: bit 0 the before-state, bit 1 localized
  uint32_t t_lo, t_hi;
};
static_assert(sizeof(PvBracket) == 16);

struct PvBrackets {
  const PvFlip* flips;
  PvBracket* out;
  BALLA_HD void operator()(size_t fi) const {
    const PvFlip& x = flips[fi];
    out[fi] = PvBracket{x.slot, x.k, x.l, static_cast<uint8_t>(x.before | (x.phase == kPhDone ? 2u : 0u)),
                        x.t0 + x.lo * x.tau, x.t0 + x.hi * x.tau};
  }
};

// The would-be P&V nodes (plan §3.3, §5.3's sensor-piece estimate): each occupied tau-cell emits its two ends, shared
// ends once, none at the interval's ends: per localized flip its cell's inner ends as keys (slot << 24 | time from the
// interval's start; none: ~0), sorted, then counted once per distinct key into slot_stats' fourth word.
inline constexpr int kPvNodeKeyBits = 24;  // interval lengths <= 2^23 ns
struct PvNodeKeys {
  const PvFlip* flips;
  const uint32_t* dt;
  uint64_t* keys;   // 2 per flip
  uint32_t* vals;   // the sort's values (unused)
  BALLA_HD void operator()(size_t fi) const {
    const PvFlip& x = flips[fi];
    const bool done = x.phase == kPhDone;
    const uint32_t a = x.t0, lo = x.t0 + x.lo * x.tau, hi = x.t0 + x.hi * x.tau;
    const uint64_t s = static_cast<uint64_t>(x.slot) << kPvNodeKeyBits;
    keys[2 * fi] = done && lo > a ? s | (lo - a) : ~uint64_t{0};
    keys[2 * fi + 1] = done && hi < dt[x.slot] ? s | (hi - a) : ~uint64_t{0};
    vals[2 * fi] = vals[2 * fi + 1] = 0;
  }
};
// The nodes' sort keys' bits for a batch of `slots` lattice slots.
inline int pv_node_bits(size_t slots) {
  int b = 1;
  while ((size_t{1} << b) < slots + 1) ++b;
  return kPvNodeKeyBits + b;
}
struct PvNodeCount {
  const uint64_t* keys;  // sorted
  uint32_t* slot_stats;
  BALLA_HD void operator()(size_t i) const {
    const uint64_t k = keys[i];
    if (k == ~uint64_t{0} || (i > 0 && keys[i - 1] == k)) return;
    atomic_add(&slot_stats[4 * static_cast<size_t>(k >> kPvNodeKeyBits) + 3], 1u);
  }
};

// The state check at bisection's midpoints (G-D2, plan §5.3), per refinement sample g of the round [first, first + m)
// in an interval where P&V ran (a candidate R0 left), at the pixel's eligible N: each sub-sample's rendered key and
// mask (the round's band call wrote the codes: mid slot / codes / table; a uniform midpoint's from its class word and
// signature) against what the ends and the brackets predict, per light: equal end states and the other state at the
// midpoint, an unseen transient; unequal ones, the bracket's state outside it (inside: skipped), and otherwise a
// contradiction at an observed time (the bracket's ends were probed or are the interval's) or an unseen pair beside
// the localized flip (multi) elsewhere; a key equal to neither end's, a third state (no light compared). Counted per
// class (every P&V-run interval, those the full set accepts, those the posterior set accepts).
struct PvCheck {
  PvBatch b;
  uint32_t first, s0;
  const uint32_t* before;     // per refinement sample g - s0: its lattice interval's upper slot
  const PvBracket* brackets;  // the flips' order
  const uint32_t* off;        // per batch pixel: its flips' offsets
  const uint32_t* mid_slot;   // per sample: its mid entry (render::kPvNone, kPvOverflow)
  const uint32_t* mid_codes;
  const uint32_t* mid_table;
  unsigned long long* counts;
  BALLA_HD void operator()(size_t mi) const {
    const uint32_t g = first + static_cast<uint32_t>(mi);
    const uint32_t a = b.owner[g];
    if (!b.eligible(a)) return;
    const uint32_t i = before[g - s0], rt = b.route[i];
    if ((rt & kRtCand) == 0 || (rt & (kRtCover | kRtR0)) != 0) return;
    const bool cls[3] = {true, (rt & kRtFull) == 0, (rt & kRtPosterior) == 0};
    PvEnds e;
    e.resolve(b.v, b.cls, b.sig, b.p.n, i - 1, i, a, i - b.off0[a]);
    // The midpoint's state: tabled (its entry), or uniform (one key: the ends', if its geometry hash is theirs).
    const uint32_t me = mid_slot[g];
    uint32_t c[kPvChkWords] = {1, 0, 0, 0, 0, 0, 0};
    uint32_t bug = 0, unchecked = 0, third_uniform = 0;
    const uint32_t* mc = nullptr;
    const uint32_t* mt = nullptr;
    uint32_t u_inst = 0, u_surf = 0, u_mask = 0;
    if (me == render::kPvOverflow) {
      unchecked = 1;
    } else if (me != render::kPvNone) {
      mc = mid_codes + static_cast<size_t>(me) * b.v.code_words;
      mt = mid_table + static_cast<size_t>(me) * b.v.table_words;
      if ((mt[0] & render::kPvTableOverflow) != 0) unchecked = 1;
    } else if ((b.cls[g] >> 16) != (b.cls[i] >> 16)) {
      third_uniform = 1;
    } else {
      uint32_t i0, s_0, m0;
      e.state(b.v, false, 0, i0, s_0, m0);
      bool one = true;
      for (uint32_t k = 1; k < b.p.n && one; ++k) {
        uint32_t ik, sk, mk;
        e.state(b.v, false, k, ik, sk, mk);
        one = ik == i0 && sk == s_0;
      }
      u_inst = i0;
      u_surf = s_0;
      if (!one || !pv_uniform_mask(b.cls[g], b.sig[g], i0, s_0, b.p.n, b.p.lights, u_mask)) unchecked = 1;
    }
    if (unchecked == 0 && third_uniform == 0) {
      const uint32_t t = b.dt[g];
      for (uint32_t k = 0; k < b.p.n; ++k) {
        uint32_t ia, sa, ma, ib, sb, mb, im = u_inst, sm = u_surf, mm = u_mask;
        e.state(b.v, false, k, ia, sa, ma);
        e.state(b.v, true, k, ib, sb, mb);
        if (mc != nullptr) {
          const uint32_t code = pv_get_code(mc, k, b.v.bits), r = code & 7u;
          mm = code >> 3;
          if (r < b.v.rows(mt)) {
            im = b.v.row(mt, r)[0];
            sm = b.v.row(mt, r)[1];
          } else {
            im = r == render::kPvCodeEnv ? render::kInstanceNone : 0xFFFFFFFFu;
            sm = r;
          }
        }
        if (im != ia || sm != sa) {
          ++c[kPvChkThird];
          continue;
        }
        for (int32_t l = 0; l < b.p.lights; ++l) {
          const uint32_t ba = (ma >> l) & 1u, bb = (mb >> l) & 1u, bm = (mm >> l) & 1u;
          if (ba == bb) {
            ++c[kPvChkCompared];
            c[kPvChkTransient] += bm != ba ? 1u : 0u;
            continue;
          }
          const uint32_t q = pv_lower(brackets, off[a], off[a + 1], static_cast<uint32_t>(l), ba, i, k);
          if (q >= off[a + 1] || brackets[q].slot != i || brackets[q].k != k || brackets[q].l != l ||
              (brackets[q].before & 1u) != ba) {
            ++bug;  // a flip the lists do not hold
            continue;
          }
          const PvBracket& x = brackets[q];
          if ((x.before & 2u) == 0) {
            ++c[kPvChkUnpredicted];
          } else if (t > x.t_lo && t < x.t_hi) {
            ++c[kPvChkInBracket];
          } else {
            ++c[kPvChkCompared];
            if (bm != (t <= x.t_lo ? ba : bb)) {
              if (t == x.t_lo || t == x.t_hi) {
                ++bug;
              } else {
                ++c[kPvChkMulti];
              }
            }
          }
        }
      }
    }
    for (int q = 0; q < 3; ++q) {
      for (int w = 0; w < kPvChkWords; ++w) {
        const uint32_t v = unchecked == 0 && third_uniform == 0 && cls[q] ? c[w] : 0u;
        add_count(&counts[kPvCheck + kPvChkWords * q + w], v);
      }
    }
    add_count(&counts[kPvCheckBug], bug);
    add_count(&counts[kPvCheckBug + 1], unchecked);
    add_count(&counts[kPvCheckBug + 2], third_uniform);
  }
};

// Per slot, after the assembly: the interval's class and rule counts with its refinement cost (cost[i]: the samples
// bisection rendered in (i - 1, i); null: none), the eligible pixels' cost, and its probes in each guard set's run
// probes when that set's pre-routing rules leave it.
struct PvCount {
  const uint32_t* route;
  const uint32_t* slot_stats;
  const uint32_t* cost;
  const uint32_t* owner;
  const uint32_t* off0;
  const uint8_t* n_log2;
  uint32_t subsamples, n;
  unsigned long long* counts;
  BALLA_HD void operator()(size_t si) const {
    const auto i = static_cast<uint32_t>(si);
    const uint32_t a = owner[i];
    const bool elig = i > off0[a] && (n_log2 != nullptr ? (1u << n_log2[a]) : subsamples) == n;
    const uint32_t rt = route[i], c = cost != nullptr ? cost[i] : 0u;
    add_count(&counts[kPvEligibleCost], elig ? c : 0u);
    const bool cand = (rt & kRtCand) != 0;
    const uint32_t* w = slot_stats + 4 * static_cast<size_t>(i);
    const bool cls[3] = {cand, cand && (rt & kRtFull) == 0, cand && (rt & kRtPosterior) == 0};
    for (int q = 0; q < 3; ++q) {
      unsigned long long* k = counts + 5 * q;
      count_if(&k[0], cls[q]);
      add_count(&k[1], cls[q] ? c : 0u);
      add_count(&k[2], cls[q] ? w[0] : 0u);
      add_count(&k[3], cls[q] ? w[1] : 0u);
      add_count(&k[4], cls[q] ? w[2] : 0u);
      add_count(&counts[kPvNodes + q], cls[q] ? w[3] : 0u);  // D3b: the would-be P&V nodes
    }
    const uint32_t rules[9] = {kRtCover, kRtR0, kRtR6, kRtR2, kRtR5, kRtR1, kRtR4, kRtR3, kRtR3p};
    for (int q = 0; q < 9; ++q) {
      const bool fires = cand && (rt & rules[q]) != 0;
      const int at = q < 6 ? kPvRule + 2 * q : kPvRuleB + 2 * (q - 6);
      count_if(&counts[at], fires);
      add_count(&counts[at + 1], fires ? c : 0u);
    }
    const uint32_t causes[4] = {kRtOverflow, kRtNoTable, kRtCarried, kRtKey};
    for (int q = 0; q < 4; ++q) count_if(&counts[kPvR0Cause + q], cand && (rt & causes[q]) != 0);
    const uint32_t r5 = (rt >> kRtR5Shift) & 3u;
    for (uint32_t q = 0; q < 3; ++q) count_if(&counts[kPvR5Cause + q], cand && r5 == q + 1);
    add_count(&counts[kPvRunProbes], cand && (rt & kRtPreFull) == 0 ? w[2] : 0u);
    add_count(&counts[kPvRunProbes + 1], cand && (rt & kRtPrePosterior) == 0 ? w[2] : 0u);
  }
};

// The carried rows' codes and tables (PixelRuns' last samples; plan §3.1 Step A): entry e's words, or none (header
// kPvNone).
BALLA_HD inline void pv_copy_entry(uint32_t* dst_codes, uint32_t* dst_table, const uint32_t* codes, const uint32_t* table,
                                   uint32_t e, uint32_t code_words, uint32_t table_words) {
  if (e >= render::kPvOverflow) {
    dst_table[0] = render::kPvNone;
    return;
  }
  for (uint32_t w = 0; w < code_words; ++w) dst_codes[w] = codes[static_cast<size_t>(e) * code_words + w];
  for (uint32_t w = 0; w < table_words; ++w) dst_table[w] = table[static_cast<size_t>(e) * table_words + w];
}

struct PvCarry {
  const uint32_t* off0;
  const uint32_t* carried_at;  // null: a
  const uint32_t* slot;
  const uint32_t* codes;
  const uint32_t* table;
  const uint8_t* n_log2;
  uint32_t subsamples, n, code_words, table_words;
  uint32_t* carried_codes;
  uint32_t* carried_table;
  BALLA_HD void operator()(size_t a) const {
    const size_t r = carried_at != nullptr ? carried_at[a] : a;
    const bool elig = (n_log2 != nullptr ? (1u << n_log2[a]) : subsamples) == n;
    pv_copy_entry(carried_codes + r * code_words, carried_table + r * table_words, codes, table,
                  elig ? slot[off0[a + 1] - 1] : render::kPvNone, code_words, table_words);
  }
};

}  // namespace balla::engine
