// Window sampling functors (ENGINE_CPP §1.2, §1.3 K5-K13): the dyadic lattice and the fixed-rate
// grid, request building, the round-0 and refinement flags (signature and P3's eta_L midpoint test),
// midpoints, and the run assembly. BALLA_HD, so the host and the device run the same arithmetic; they
// reproduce the NumPy sampler of engine python (deleted at I5; the frozen bench references hold its bits)
// bit for bit.
//
// Layout of one pixel batch (nb active pixels, local index a): sample g < s0 is a lattice slot,
// pixel-major from off0[a] (slot j = 0 is the carried sample at t_w, never rendered); refinement
// samples follow from g = s0, each with the round-0 slot it precedes (`before`). Times are offsets
// dt = t - t_w in [0, 2^23]. An interval is a pair (lo << 32) | hi of sample indices.
#pragma once

#include <math.h>
#include <stdint.h>

#include "core/hd.h"
#include "core/time/time.h"
#include "engine/par.h"
#include "render/types.h"

namespace balla::engine {

using render::SampleRequest;
inline constexpr int kMaxGroups = render::kMaxLightGroups;

// NumPy's float minimum and maximum: NaN propagates from either operand.
BALLA_HD inline float np_min(float a, float b) { return (a <= b || a != a) ? a : b; }
BALLA_HD inline float np_max(float a, float b) { return (a >= b || a != a) ? a : b; }

// P3 §6.3's eta_L in sampling.py's fp32 operation order (MidpointTest.eta): per group in order,
// dev = |(a + b) * -0.5 + m|, num += m_max dev, den += m_min min(a, m, b), and with modulated groups
// the mediant max(dev / low, 0 where dev == 0); eta = num / den, or min(eta, mediant). Values are
// floats of the Python doubles; the comparison is eta > float(tol) (NEP 50).
// dark is the dark term in renderer units, dark current / (scale gain); with a per-pixel gain map (natural vignetting,
// D1) dark_px holds it per pixel id (Space memory) and dark the on-axis value (null: dark for every pixel).
struct MidpointParams {
  float tol;
  float dark;
  int32_t groups;
  bool mediant;  // m_min != m_max
  float m_min[kMaxGroups], m_max[kMaxGroups];
  const float* dark_px = nullptr;
  BALLA_HD float dark_of(uint32_t pixel) const { return dark_px != nullptr ? dark_px[pixel] : dark; }
};

BALLA_HD inline float midpoint_eta(const MidpointParams& p, float dark, const float* a, const float* m,
                                   const float* b) {
  float num = 0.0f, den = dark, mediant = 0.0f;
  for (int32_t g = 0; g < p.groups; ++g) {
    const float dev = fabsf((a[g] + b[g]) * -0.5f + m[g]);
    const float low = np_min(np_min(a[g], m[g]), b[g]);
    num += p.m_max[g] * dev;
    den += p.m_min[g] * low;
    if (p.mediant) mediant = np_max(mediant, dev == 0.0f ? 0.0f : dev / low);
  }
  const float eta = num / den;
  return p.mediant ? np_min(eta, mediant) : eta;
}
BALLA_HD inline float midpoint_eta(const MidpointParams& p, const float* a, const float* m, const float* b) {
  return midpoint_eta(p, p.dark, a, m, b);
}

BALLA_HD inline bool midpoint_exceeds(const MidpointParams& p, float dark, const float* a, const float* m,
                                      const float* b) {
  return midpoint_eta(p, dark, a, m, b) > p.tol;
}
BALLA_HD inline bool midpoint_exceeds(const MidpointParams& p, const float* a, const float* m, const float* b) {
  return midpoint_exceeds(p, p.dark, a, m, b);
}

// The endpoint jump eta_J of an interval (samples study §3.4), eta_L's analogue on the ends a, b: per group in order,
// dev = |b - a|, num += m_max dev, den += m_min min(a, b), with modulated groups the mediant max(dev / low); fp32.
BALLA_HD inline float endpoint_eta(const MidpointParams& p, float dark, const float* a, const float* b) {
  float num = 0.0f, den = dark, mediant = 0.0f;
  for (int32_t g = 0; g < p.groups; ++g) {
    const float dev = fabsf(b[g] - a[g]);
    const float low = np_min(a[g], b[g]);
    num += p.m_max[g] * dev;
    den += p.m_min[g] * low;
    if (p.mediant) mediant = np_max(mediant, dev == 0.0f ? 0.0f : dev / low);
  }
  const float eta = num / den;
  return p.mediant ? np_min(eta, mediant) : eta;
}
BALLA_HD inline float endpoint_eta(const MidpointParams& p, const float* a, const float* b) {
  return endpoint_eta(p, p.dark, a, b);
}

// What decides a window's samples (sampling.sample_window / fixed_rate_window arguments).
struct SampleRules {
  bool fixed = false;         // fixed-rate grid: no flags, no refinement
  int64_t period_ns = 0;      // fixed-rate period
  uint32_t m = 0;             // fixed-rate samples per pixel and window, the window end included
  bool refine_signature = true;
  bool midpoint = false;
  MidpointParams mid{};       // eta_L's parameters; eta_J's too (the jump rules)
  int32_t max_refine_levels = 4;
  uint32_t min_split = 2048;  // 2 min_dt_ns: shorter intervals are not split
  int32_t shift = 10;         // log2 min_dt_ns: asynchronous pose slot = dt >> shift
  // The economy schedule's switches (ENGINE_CPP I7, samples study §3.3-§3.5); the defaults are Phase 1's schedule.
  // Midpoint flags apply to intervals fewer than mid_levels levels below the lattice. A signature flag counts where
  // eta_J > eps_jump (gate), and splits an interval of at least tau2 max(1, delta_ref / eta_J) ns (tau2 = 2 tau; 0:
  // no stop). After a split a midpoint-flagged half refines only where its eta_J >= half_ratio x the larger of the
  // two halves' (0: both). max_refine_levels still stops every flag.
  int32_t mid_levels = INT32_MAX;
  bool gate = false;
  float eps_jump = 0.0f;
  float tau2 = 0.0f;
  float delta_ref = 0.0f;
  float half_ratio = 0.0f;
  // P2-C C2 (execution.jump_counters): count what the jump rules would change (kJump*, kSib* below); output-neutral.
  bool counters = false;

  BALLA_HD bool jump_rules() const { return gate || tau2 > 0.0f; }
  // A gated signature flag of an interval of len ns with endpoint jump eta: does the tau stop let it split?
  BALLA_HD bool tau_allows(uint32_t len, float eta) const {
    if (!(tau2 > 0.0f)) return true;
    float need = tau2;
    if (delta_ref > 0.0f && !(eta >= delta_ref)) need = need * (delta_ref / eta);  // eta 0: no split
    return static_cast<float>(len) >= need;
  }
};

// The signature flag of an interval (a, b its end values, len its length; dark its pixel's dark term) under the jump
// rules: s (the gated flag, counted) and whether it may split the interval (the tau stop).
BALLA_HD inline void signature_flag(const SampleRules& r, float dark, bool flip, uint32_t len, const float* a,
                                    const float* b, bool& s, bool& splits) {
  s = flip;
  splits = flip;
  if (!flip || !r.jump_rules()) return;
  const float eta = endpoint_eta(r.mid, dark, a, b);
  s = !r.gate || eta > r.eps_jump;
  splits = s && r.tau_allows(len, eta);
}

// P2-0b counters of a window (PHASE2 P2-0, F010, F022; ENGINE_CPP §0.1), from the renderer's class word
// (render/types.h kClass*) of every rendered sample and of the ends of every terminal signature interval (one that
// holds a flip and is not split further): rendered pixel-samples that are textured, on a shadow boundary or on a
// silhouette (two or more clusters); shadow jumps (equal geometry hashes at the ends), their sub-sample flips (the net
// change of shadowed pairs, at least one) and coverage jumps (the hashes differ); and the shadow jumps by eta_J per
// flip, in units of eps_L / 8 (refine_tol_log / 8; DESIGN's eps_jump), in the bins [0, 1), [1, 2), [2, 4), ...,
// [64, inf). Counting changes no sample.
inline constexpr int kDiagTextured = 0, kDiagShadowEdge = 1, kDiagSilhouette = 2, kDiagShadowJumps = 3,
                     kDiagShadowFlips = 4, kDiagCoverageJumps = 5, kDiagHist = 6, kDiagBins = 8,
                     kDiagWords = kDiagHist + kDiagBins;

// The class word bits of a sample that the band N renders at N_band (P2-B B4a: a carried seed of the plan, plan.h; a
// window's edge pixels, WindowEngine::Input::edge, which B4b escalates at N_base): two or more clusters (a silhouette
// or a material boundary) or a shadow boundary.
inline constexpr uint32_t kBandEdge = render::kClassMulti | render::kClassShadowEdge;

BALLA_HD inline void count_sample(unsigned long long* d, uint32_t c, bool rendered) {
  count_if(&d[kDiagTextured], rendered && (c & render::kClassTextured) != 0);
  count_if(&d[kDiagShadowEdge], rendered && (c & render::kClassShadowEdge) != 0);
  count_if(&d[kDiagSilhouette], rendered && (c & render::kClassMulti) != 0);
}

// A terminal signature interval (`terminal`; ends a, b with class words ca, cb).
BALLA_HD inline void count_jump(const SampleRules& r, float dark, unsigned long long* d, bool terminal, uint32_t ca,
                                uint32_t cb, const float* a, const float* b) {
  const bool shadow = terminal && (ca >> 16) == (cb >> 16);
  uint32_t flips = 0;
  int32_t bin = -1;
  if (shadow) {
    const uint32_t m = (cb - ca) & render::kClassShadowed;  // the net change as a signed 12-bit difference (F6)
    const uint32_t net = m > render::kClassShadowed / 2 ? render::kClassShadowed + 1 - m : m;
    flips = net > 0 ? net : 1;
    const float unit = (r.mid.tol > 0.0f ? 0.125f * r.mid.tol : 2.5e-4f) * static_cast<float>(flips);
    const float x = endpoint_eta(r.mid, dark, a, b) / unit;
    bin = 0;
    for (float edge = 1.0f; bin < kDiagBins - 1 && x >= edge; edge *= 2.0f) ++bin;
  }
  count_if(&d[kDiagShadowJumps], shadow);
  add_count(&d[kDiagShadowFlips], flips);
  count_if(&d[kDiagCoverageJumps], terminal && !shadow);
  for (int32_t k = 0; k < kDiagBins; ++k) count_if(&d[kDiagHist + k], bin == k);
}

// P2-C C2's counters (SampleRules::counters; execution.jump_counters; plan p2c §3.2-§3.3, §11.4). They change no flag,
// sample or run. A signature interval (refine_signature, differing end signatures) is counted where the flag pass
// evaluates it, so every interval of a window's tree once (round 0's lattice intervals, then each split's two halves);
// the thresholds are e_k = eps_L / 8, eps_L / 2, eps_L (eps_L = refine_tol_log, 0.002 without a midpoint test).
// - eligible, transit: both outer intervals exist and have equal end signatures (the two-sided rule applies), or not
//   (the rule falls back to the endpoint eta_J). Round 0's outers are the lattice neighbours; a half's are its sibling
//   and, on the other side, its parent's outer (inherited: list entries carry lo_out and hi_out, kNoOuter for none).
// - eta_delta[8]: eligible intervals by the two-sided eta_Delta (two_sided_eta) in units of eps_L / 8, bins [0, 1),
//   [1, 2), [2, 4), ..., [64, inf).
// - misses[3]: endpoint misses at e_k, terminal intervals that could still split (above the level cap and at least
//   min_split long) with eta_J <= e_k < eta_Delta: what the endpoint gate leaves un-localized and the node rule would
//   localize. miss_splits: at eps_L, the splits each would take down to the cap (a localized flip costs one sample a
//   level).
// - savings[3]: node savings at e_k, intervals with eta_Delta <= e_k < eta_J (any level). saved_samples: the samples
//   rendered in the subtrees of signature-only splits with eta_Delta <= eps_L, which the node rule at eps_L does not
//   render (in a run gated at eps_L every such split is a node saving).
// - cancel_coverage, cancel_shadow (with class words): terminal intervals that both rules gate at eps_L (eta_J and
//   eta_Delta <= eps_L) whose ends' geometry hashes differ (coverage flips with a small net step), or are equal with an
//   equal shadowed-pair count (shadow flips that cancel).
// - bracket[14]: terminal localized intervals (their signature flag counted) by length, bins [0, 2^11), [2^11, 2^12),
//   ..., [2^22, 2^23), [2^23, inf) ns.
// - sibling_flags: midpoint flags of an interval without a signature flag beside one with it (in the triple that
//   flags it), where the split happens; sibling_skips: of which F060's outer-node rule would not split it (eta of the
//   outer node against the chord from the far end, times lambda / (1 - lambda), at most the tolerance; at round 0 no
//   jump-free covering triple exceeds); sibling_forced: of which no eligible outer node (or jump-free covering
//   triple) is left, so the incoming-cluster rule splits it once; sibling_skipped_samples: the samples rendered in the
//   subtrees of the skipped splits and not in a node saving's (counted there), so that the two subtree counts add.
// - terminal_eligible, terminal_transit: the terminal signature intervals (one per localized or gated flip cluster,
//   P2-0b's shadow_jumps + coverage_jumps without class words) with both outers eligible, or not: the denominators of
//   the endpoint misses (eligible / transit count an interval at every level it is evaluated on).
// - sibling_added, sibling_added_forced: smooth intervals beside a jump interval that the run does not split (no
//   midpoint flag) and the node rules would split: by the outer-node rule (its eta lambda / (1 - lambda) above the
//   tolerance) or forced (no eligible outer node; at round 0 no jump-free covering triple), where a midpoint test
//   would run (rules.midpoint, the depth below mid_levels and the cap, at least min_split long). One sample each, the
//   split's midpoint: a lower bound of the samples the node rules add there (the halves' own splits are not counted).
inline constexpr int kJumpEligible = 0, kJumpTransit = 1, kJumpEta = 2, kJumpMiss = kJumpEta + kDiagBins,
                     kJumpSave = kJumpMiss + 3, kJumpMissSplits = kJumpSave + 3, kJumpSavedSamples = kJumpMissSplits + 1,
                     kJumpCancelCoverage = kJumpSavedSamples + 1, kJumpCancelShadow = kJumpCancelCoverage + 1,
                     kJumpBracket = kJumpCancelShadow + 1, kJumpBracketBins = 14,
                     kSibFlags = kJumpBracket + kJumpBracketBins, kSibSkips = kSibFlags + 1, kSibForced = kSibSkips + 1,
                     kSibSkippedSamples = kSibForced + 1, kJumpTermEligible = kSibSkippedSamples + 1,
                     kJumpTermTransit = kJumpTermEligible + 1, kSibAdded = kJumpTermTransit + 1,
                     kSibAddedForced = kSibAdded + 1, kJumpWords = kSibAddedForced + 1;
inline constexpr uint32_t kNoOuter = 0xFFFFFFFFu;
// A listed interval's subtree tags (the counters' only state): in the subtree of a node saving, of a skipped sibling.
// A flag pass carries a selected interval's tags in its select flag's upper bits (1 | tags << 1: nonzero either way).
inline constexpr uint8_t kTagSaved = 1, kTagSkipped = 2;

// The two-sided eta_Delta of an interval (a, b) of length h with outer intervals (ao, a) of length h_l and (b, bo) of
// length h_r (plan p2c §3.2): per group the node step less each outer interval's trend over h, (b - a) - (a - ao) h /
// h_l and (b - a) - (bo - b) h / h_r, the larger in magnitude, weighted as endpoint_eta (m_max over m_min min(a, b)
// plus dark, the mediant with modulated groups); fp32.
BALLA_HD inline float two_sided_eta(const MidpointParams& p, float dark, const float* ao, const float* a, const float* b,
                                    const float* bo, float h_l, float h, float h_r) {
  float num = 0.0f, den = dark, mediant = 0.0f;
  const float rl = h / h_l, rr = h / h_r;
  for (int32_t g = 0; g < p.groups; ++g) {
    const float step = b[g] - a[g];
    const float dev = np_max(fabsf(step - (a[g] - ao[g]) * rl), fabsf(step - (bo[g] - b[g]) * rr));
    const float low = np_min(a[g], b[g]);
    num += p.m_max[g] * dev;
    den += p.m_min[g] * low;
    if (p.mediant) mediant = np_max(mediant, dev == 0.0f ? 0.0f : dev / low);
  }
  const float eta = num / den;
  return p.mediant ? np_min(eta, mediant) : eta;
}

// eta_L's deviation of node m from the chord (a, b) at m's fraction lam of it, (t_m - t_a) / (t_b - t_a); eta_L's
// weighting (midpoint_eta's at lam 1/2, up to rounding).
BALLA_HD inline float chord_eta(const MidpointParams& p, float dark, const float* a, const float* m, const float* b,
                                float lam) {
  float num = 0.0f, den = dark, mediant = 0.0f;
  for (int32_t g = 0; g < p.groups; ++g) {
    const float dev = fabsf(m[g] - (a[g] + lam * (b[g] - a[g])));
    const float low = np_min(np_min(a[g], m[g]), b[g]);
    num += p.m_max[g] * dev;
    den += p.m_min[g] * low;
    if (p.mediant) mediant = np_max(mediant, dev == 0.0f ? 0.0f : dev / low);
  }
  const float eta = num / den;
  return p.mediant ? np_min(eta, mediant) : eta;
}

// F060's outer-node test of a smooth half of length h (plan p2c §3.3 (ii)): its end m away from the jump half against
// the chord from its other end a (the node it shares with the jump half) to the outer node b, h_out beyond m; m lies
// at lam = h / (h + h_out) of that chord, and the half flags iff eta lam / (1 - lam) > tol (for a quadratic of
// curvature k: k h^2 / 2 at every lam, the half's own midpoint test).
BALLA_HD inline bool outer_node_flags(const MidpointParams& p, float dark, const float* a, const float* m, const float* b,
                                      float h, float h_out) {
  const float lam = h / (h + h_out);
  return chord_eta(p, dark, a, m, b, lam) * (lam / (1.0f - lam)) > p.tol;
}

// Bin of x in [0, 1), [1, 2), [2, 4), ..., [64, inf) (kDiagBins; NaN in the first).
BALLA_HD inline int32_t unit_bin(float x) {
  int32_t bin = 0;
  for (float edge = 1.0f; bin < kDiagBins - 1 && x >= edge; edge *= 2.0f) ++bin;
  return bin;
}

// The splits an interval of len ns at `depth` levels below the lattice can still take (the cap, min_split).
BALLA_HD inline uint32_t splits_left(const SampleRules& r, int32_t depth, uint32_t len) {
  uint32_t n = 0;
  for (int32_t d = depth; d < r.max_refine_levels && len >= r.min_split; ++d, len >>= 1) ++n;
  return n;
}

// The signature flag s of interval (a, b) (signature_flag's, without the tau stop): is it a jump interval of the run?
BALLA_HD inline bool jump_of(const SampleRules& r, float dark, const uint32_t* dt, const float* irr, const uint64_t* sig,
                             int32_t groups, uint32_t a, uint32_t b) {
  bool s = false, splits = false;
  signature_flag(r, dark, r.refine_signature && sig[a] != sig[b], dt[b] - dt[a], irr + static_cast<size_t>(a) * groups,
                 irr + static_cast<size_t>(b) * groups, s, splits);
  return s;
}

// C2's counts of one signature interval (a, b) with outer nodes ao, bo (kNoOuter: none) at `depth` levels below the
// lattice: s its signature flag, terminal (not split further), sig_split (split by its signature flag alone); cls the
// class words (null: no cancellation counts). Returns kTagSaved for a signature-only split with eta_Delta <= eps_L.
BALLA_HD inline uint8_t count_signature(const SampleRules& r, float dark, unsigned long long* c, const uint32_t* dt,
                                        const float* irr, const uint64_t* sig, int32_t groups, uint32_t a, uint32_t b,
                                        uint32_t ao, uint32_t bo, int32_t depth, bool s, bool terminal, bool sig_split,
                                        const uint32_t* cls) {
  const float* ea = irr + static_cast<size_t>(a) * groups;
  const float* eb = irr + static_cast<size_t>(b) * groups;
  const uint32_t len = dt[b] - dt[a];
  const float eta_j = endpoint_eta(r.mid, dark, ea, eb);
  const bool eligible = ao != kNoOuter && bo != kNoOuter && sig[ao] == sig[a] && sig[b] == sig[bo];
  const float eta_d = eligible ? two_sided_eta(r.mid, dark, irr + static_cast<size_t>(ao) * groups, ea, eb,
                                               irr + static_cast<size_t>(bo) * groups,
                                               static_cast<float>(dt[a] - dt[ao]), static_cast<float>(len),
                                               static_cast<float>(dt[bo] - dt[b]))
                               : eta_j;
  const float unit = r.mid.tol > 0.0f ? 0.125f * r.mid.tol : 2.5e-4f;
  count_if(&c[kJumpEligible], eligible);
  count_if(&c[kJumpTransit], !eligible);
  count_if(&c[kJumpTermEligible], terminal && eligible);
  count_if(&c[kJumpTermTransit], terminal && !eligible);
  const int32_t bin = eligible ? unit_bin(eta_d / unit) : -1;
  for (int32_t k = 0; k < kDiagBins; ++k) count_if(&c[kJumpEta + k], bin == k);
  const uint32_t left = terminal ? splits_left(r, depth, len) : 0;
  for (int32_t k = 0; k < 3; ++k) {
    const float e = unit * static_cast<float>(k == 0 ? 1 : k == 1 ? 4 : 8);
    count_if(&c[kJumpMiss + k], left > 0 && eta_j <= e && eta_d > e);
    count_if(&c[kJumpSave + k], eta_j > e && eta_d <= e);
  }
  const float eps = 8.0f * unit;
  add_count(&c[kJumpMissSplits], left > 0 && eta_j <= eps && eta_d > eps ? left : 0u);
  bool coverage = false, shadow = false;
  if (cls != nullptr && terminal && eta_j <= eps && eta_d <= eps) {
    coverage = (cls[a] >> 16) != (cls[b] >> 16);
    shadow = !coverage && ((cls[b] - cls[a]) & render::kClassShadowed) == 0;
  }
  count_if(&c[kJumpCancelCoverage], coverage);
  count_if(&c[kJumpCancelShadow], shadow);
  int32_t bracket = -1;
  if (s && terminal) {
    bracket = 0;
    while (bracket < kJumpBracketBins - 1 && (len >> (bracket + 11)) != 0) ++bracket;
  }
  for (int32_t k = 0; k < kJumpBracketBins; ++k) count_if(&c[kJumpBracket + k], bracket == k);
  return sig_split && eta_d <= eps ? kTagSaved : 0;
}

// Offset of fixed-rate slot j of the window at t_w: the absolute multiples of the period inside
// (t_w, t_w + W), then t_w + W itself (render/pose.h slot_time, minus t_w).
BALLA_HD inline uint32_t fixed_dt(int64_t t_w, int64_t period, uint32_t m, uint32_t j) {
  if (j == 0) return 0;
  if (j >= m) return static_cast<uint32_t>(time::kWindowNs);
  return static_cast<uint32_t>((time::floor_div(t_w, period) + j) * period - t_w);
}

BALLA_HD inline uint64_t interval(uint32_t lo, uint32_t hi) { return (static_cast<uint64_t>(lo) << 32) | hi; }

// K4: lattice samples of every active pixel (2^level + 1, or m + 1 on the fixed-rate grid), counts[n] = 0.
struct LatticeCounts {
  const uint8_t* level;
  bool fixed;
  uint32_t m;
  size_t n;
  uint64_t* counts;
  BALLA_HD void operator()(size_t a) const {
    counts[a] = a == n ? 0 : fixed ? m + 1 : (uint64_t{1} << level[a]) + 1;
  }
};

// A batch's slot offsets relative to its first pixel.
struct LocalOffsets {
  const uint64_t* off64;
  uint64_t base;
  uint32_t* off0;
  BALLA_HD void operator()(size_t i) const { off0[i] = static_cast<uint32_t>(off64[i] - base); }
};

// K5, owners: marks the last slot of every pixel but the batch's last, so that the exclusive scan of
// the marks is each slot's pixel (O(s0), against a binary search of off0 per slot).
struct MarkPixelEnds {
  const uint32_t* off0;
  uint32_t* marks;  // s0, zeroed
  BALLA_HD void operator()(size_t a) const { marks[off0[a + 1] - 1] = 1; }
};

// K5: every lattice slot's time and (slot 0) carried value, given its owner; the other slots become
// the round's samples to render, in slot order (out_index).
struct LatticeSlots {
  int64_t t_w;
  SampleRules rules;
  const uint32_t* off0;  // nb + 1
  const uint32_t* owner;
  const uint8_t* level;
  const float* carried_irr;
  const uint64_t* carried_sig;
  const uint32_t* carried_at;  // carried row of active pixel a (a when null)
  int32_t groups;
  uint32_t* dt;
  float* irr;
  uint64_t* sig;
  uint32_t* out_index;
  const uint32_t* carried_class = nullptr;  // the counters' class words (null: none)
  uint32_t* cls = nullptr;

  BALLA_HD void operator()(size_t si) const {
    const auto s = static_cast<uint32_t>(si);
    const uint32_t a = owner[s], j = s - off0[a];
    dt[s] = rules.fixed ? fixed_dt(t_w, rules.period_ns, rules.m, j) : j << (time::kWindowLog2 - level[a]);
    if (j == 0) {
      const size_t c = carried_at != nullptr ? carried_at[a] : a;
      for (int32_t g = 0; g < groups; ++g) irr[static_cast<size_t>(s) * groups + g] = carried_irr[c * groups + g];
      sig[s] = carried_sig[c];
      if (cls != nullptr) cls[s] = carried_class[c];
      return;
    }
    out_index[s - a - 1] = s;  // every pixel before a has one unrendered slot
  }
};

// K7: the requests of the samples out_index[i]; pose_slot holds the time slot of the window clock. k_count is
// subsamples, or 2^n_log2[a] with the band's per-pixel N (P2-B B2).
struct BuildRequests {
  const uint32_t* out_index;
  const uint32_t* dt;
  const uint32_t* owner;
  const uint32_t* off0;
  const uint32_t* pixel;
  uint16_t cam;
  uint32_t subsamples;
  bool fixed;
  int32_t shift;
  SampleRequest* requests;
  const uint8_t* n_log2 = nullptr;  // the batch's active pixels' log2 N (null: subsamples)

  BALLA_HD void operator()(size_t i) const {
    const uint32_t g = out_index[i], a = owner[g], d = dt[g];
    const uint32_t slot = fixed ? g - off0[a] : d >> shift;
    const uint32_t n = n_log2 != nullptr ? 1u << n_log2[a] : subsamples;
    requests[i] = {cam, 0, pixel[a], 0, n, static_cast<int32_t>(d), slot};
  }
};

// P2-B B2, the two N classes of a round's samples out_index[i] (i < n): band[i] = 1 for an owner at the band's
// log2 N, and band[n] = 0, the scan input of a stable partition.
struct BandFlags {
  const uint32_t* out_index;
  const uint32_t* owner;
  const uint8_t* n_log2;
  uint8_t band_log2;
  size_t n;
  uint32_t* band;
  BALLA_HD void operator()(size_t i) const { band[i] = i < n && n_log2[owner[out_index[i]]] == band_log2 ? 1 : 0; }
};

// The stable partition: with before[i] the band samples before i (BandFlags' exclusive scan) and base the base
// class's count, sample out_index[i] goes to out[i - before[i]] (the base class first) or out[base + before[i]].
struct BandPartition {
  const uint32_t* out_index;
  const uint32_t* owner;
  const uint8_t* n_log2;
  uint8_t band_log2;
  const uint32_t* before;
  uint32_t base;
  uint32_t* out;
  BALLA_HD void operator()(size_t i) const {
    const uint32_t g = out_index[i];
    out[n_log2[owner[g]] == band_log2 ? base + before[i] : static_cast<uint32_t>(i) - before[i]] = g;
  }
};

// K8: flags of the round-0 interval (i - 1, i) of every slot i that is not a pixel's first. Node k
// tests the triple (k - 1, k, k + 1) of one pixel and flags its two intervals; sel marks the flagged
// intervals that can be split. counters: [0] signature flags, [1] midpoint-only flags, [2] flags
// that cannot be split (min_dt).
struct Round0Flags {
  SampleRules rules;
  const uint32_t* off0;
  const uint32_t* owner;
  const uint32_t* pixel;  // the batch's pixel ids (rules.mid.dark_of)
  const uint32_t* dt;
  const float* irr;
  const uint64_t* sig;
  int32_t groups;
  uint8_t* sel;
  unsigned long long* counters;
  const uint32_t* cls = nullptr;     // class words and the P2-0b counters (null: none)
  unsigned long long* diag = nullptr;
  uint8_t* edge = nullptr;           // P2-B B4a, with cls: 1 for the batch's pixels with a sample of edge bits
  unsigned long long* jump = nullptr;  // P2-C C2's counters (kJump*, kSib*; null: none)

  BALLA_HD void operator()(size_t si) const { pass<false>(si); }
  // P2-C C2: with kCounters (Counting<Round0Flags>) the counters' code too; the plain pass has none of it.
  template <bool kCounters>
  BALLA_HD void pass(size_t si) const {
    const auto i = static_cast<uint32_t>(si);
    const uint32_t a = owner[i], j = i - off0[a], count = off0[a + 1] - off0[a];
    bool s = false, s_splits = false, m = false, want = false, split = false;
    float dark = 0.0f;
    if (j >= 1) {
      const float* e = irr + static_cast<size_t>(i) * groups;
      const uint32_t len = dt[i] - dt[i - 1];
      dark = rules.mid.dark_of(pixel[a]);
      signature_flag(rules, dark, rules.refine_signature && sig[i] != sig[i - 1], len, e - groups, e, s, s_splits);
      if (rules.midpoint && rules.mid_levels > 0) {
        m = (j + 1 < count && midpoint_exceeds(rules.mid, dark, e - groups, e, e + groups)) ||
            (j >= 2 && midpoint_exceeds(rules.mid, dark, e - 2 * groups, e - groups, e));
      }
      want = s_splits || m;
      split = want && len >= rules.min_split;
    }
    uint8_t tags = 0;
    if constexpr (kCounters) {
      if (j >= 1) tags = count_c2(i, j, count, dark, s, s_splits, m, split);
    }
    sel[i] = split ? static_cast<uint8_t>(1u | (static_cast<uint32_t>(tags) << 1)) : 0;
    count_if(&counters[0], s);
    count_if(&counters[1], m && !s);
    count_if(&counters[2], want && !split);
    if (cls != nullptr) {  // the interval is split only below the refinement cap
      const bool terminal = j >= 1 && sig[i] != sig[i - 1] && !(split && rules.max_refine_levels > 0);
      const size_t lo = j >= 1 ? i - 1 : i;
      count_sample(diag, cls[i], j >= 1);
      count_jump(rules, dark, diag, terminal, cls[lo], cls[i], irr + lo * groups,
                 irr + static_cast<size_t>(i) * groups);
      if (edge != nullptr && (cls[i] & kBandEdge) != 0) edge[a] = 1;  // the carried sample (j = 0) included
    }
  }

  // P2-C C2: the counts of interval (i - 1, i), slot j of count, and the tags of its subtree. Its outers are the
  // lattice neighbours; a midpoint flag without a signature flag is a smooth sibling where a covering triple that
  // exceeds, (i - 1, i, i + 1) or (i - 2, i - 1, i), has a jump half: the node rules test only the jump-free ones, and
  // split it once when none is left.
  BALLA_HD uint8_t count_c2(uint32_t i, uint32_t j, uint32_t count, float dark, bool s, bool s_splits, bool m,
                            bool split) const {
    const bool real = split && rules.max_refine_levels > 0;  // the split happens
    const bool flip = rules.refine_signature && sig[i] != sig[i - 1];
    uint8_t tags = 0;
    if (flip) {
      tags |= count_signature(rules, dark, jump, dt, irr, sig, groups, i - 1, i, j >= 2 ? i - 2 : kNoOuter,
                              j + 1 < count ? i + 1 : kNoOuter, 0, s, !real, real && s_splits && !m, cls);
    }
    if (m && !s && real) {
      const float* e = irr + static_cast<size_t>(i) * groups;
      const bool has_r = j + 1 < count, has_l = j >= 2;
      const bool jr = has_r && jump_of(rules, dark, dt, irr, sig, groups, i, i + 1);
      const bool jl = has_l && jump_of(rules, dark, dt, irr, sig, groups, i - 2, i - 1);
      const bool xr = has_r && midpoint_exceeds(rules.mid, dark, e - groups, e, e + groups);
      const bool xl = has_l && midpoint_exceeds(rules.mid, dark, e - 2 * groups, e - groups, e);
      const bool sibling = (xr && jr) || (xl && jl);
      const bool forced = sibling && !(has_r && !jr) && !(has_l && !jl);
      const bool skip = sibling && !forced && !((xr && !jr) || (xl && !jl));
      count_if(&jump[kSibFlags], sibling);
      count_if(&jump[kSibForced], forced);
      count_if(&jump[kSibSkips], skip);
      if (skip) tags |= kTagSkipped;
    }
    // A smooth interval the run leaves whole beside a jump interval: with no jump-free covering triple the
    // incoming-cluster rule splits it once (an addition). A jump-free covering triple is the run's own test, which
    // does not exceed here (no midpoint flag), so the node rules add nothing else at round 0.
    if (!s && !m && rules.midpoint && rules.mid_levels > 0 && rules.max_refine_levels > 0 &&
        dt[i] - dt[i - 1] >= rules.min_split) {
      const bool has_r = j + 1 < count, has_l = j >= 2;
      const bool jr = has_r && jump_of(rules, dark, dt, irr, sig, groups, i, i + 1);
      const bool jl = has_l && jump_of(rules, dark, dt, irr, sig, groups, i - 2, i - 1);
      const bool added = (jr || jl) && !(has_r && !jr) && !(has_l && !jl);
      count_if(&jump[kSibAdded], added);
      count_if(&jump[kSibAddedForced], added);
    }
    return tags;
  }
};

// The selected round-0 intervals (i - 1, i); with P2-C C2's counters (outer not null) their outer nodes, the lattice
// neighbours (kNoOuter beyond the pixel's slots), and their tags (the select flags' upper bits).
struct Round0List {
  const uint32_t* index;
  uint64_t* list;
  const uint32_t* off0 = nullptr;
  const uint32_t* owner = nullptr;
  const uint8_t* sel = nullptr;
  uint64_t* outer = nullptr;
  uint8_t* tag = nullptr;
  BALLA_HD void operator()(size_t k) const {
    const uint32_t i = index[k];
    list[k] = interval(i - 1, i);
    if (outer != nullptr) {
      const uint32_t a = owner[i], j = i - off0[a], count = off0[a + 1] - off0[a];
      outer[k] = interval(j >= 2 ? i - 2 : kNoOuter, j + 1 < count ? i + 1 : kNoOuter);
      tag[k] = static_cast<uint8_t>(sel[i] >> 1);
    }
  }
};

// K9: the midpoint of every listed interval becomes sample first + k (rendered in list order).
struct Midpoints {
  const uint64_t* list;
  uint32_t first, s0;
  uint32_t* dt;
  uint32_t* owner;
  uint32_t* before;  // indexed by g - s0
  uint32_t* out_index;

  BALLA_HD void operator()(size_t k) const {
    const uint32_t lo = static_cast<uint32_t>(list[k] >> 32), hi = static_cast<uint32_t>(list[k]);
    const uint32_t g = first + static_cast<uint32_t>(k), d = (dt[lo] + dt[hi]) >> 1, a = owner[hi];
    dt[g] = d;
    owner[g] = a;
    before[g - s0] = hi < s0 ? hi : before[hi - s0];
    out_index[k] = g;
  }
};

// K12: flags of the halves (lo, g) and (g, hi) of every split interval, `depth` levels below the lattice (round
// k + 1); the triple (lo, g, hi) flags both for the midpoint test (the half rule may keep one). Candidates 2k and
// 2k + 1; counters as in Round0Flags.
struct HalfFlags {
  SampleRules rules;
  const uint64_t* list;
  uint32_t first;
  int32_t depth;
  const uint32_t* owner;
  const uint32_t* pixel;  // the batch's pixel ids (rules.mid.dark_of)
  const uint32_t* dt;
  const float* irr;
  const uint64_t* sig;
  int32_t groups;
  uint64_t* candidates;
  uint8_t* sel;
  unsigned long long* counters;
  const uint32_t* cls = nullptr;     // class words and the P2-0b counters (null: none)
  unsigned long long* diag = nullptr;
  uint8_t* edge = nullptr;           // as Round0Flags::edge
  // P2-C C2 (jump null: none): the counters, the listed intervals' outer nodes and tags, and the candidates' outer
  // nodes (a half inherits its parent's outer on its far side; its sibling is the other).
  unsigned long long* jump = nullptr;
  const uint64_t* outer = nullptr;
  const uint8_t* tag = nullptr;
  uint64_t* cand_outer = nullptr;

  BALLA_HD void operator()(size_t k) const { pass<false>(k); }
  template <bool kCounters>  // as Round0Flags::pass
  BALLA_HD void pass(size_t k) const {
    const uint32_t lo = static_cast<uint32_t>(list[k] >> 32), hi = static_cast<uint32_t>(list[k]);
    const uint32_t g = first + static_cast<uint32_t>(k);
    const float* el = irr + static_cast<size_t>(lo) * groups;
    const float* eg = irr + static_cast<size_t>(g) * groups;
    const float* eh = irr + static_cast<size_t>(hi) * groups;
    const uint32_t len1 = dt[g] - dt[lo], len2 = dt[hi] - dt[g];
    const float dark = rules.mid.dark_of(pixel[owner[hi]]);
    bool s1, s2, t1, t2;
    signature_flag(rules, dark, rules.refine_signature && sig[lo] != sig[g], len1, el, eg, s1, t1);
    signature_flag(rules, dark, rules.refine_signature && sig[g] != sig[hi], len2, eg, eh, s2, t2);
    const bool bad = rules.midpoint && depth < rules.mid_levels && midpoint_exceeds(rules.mid, dark, el, eg, eh);
    bool m1 = bad, m2 = bad;
    if (bad && rules.half_ratio > 0.0f) {  // step-aware halving (samples study §3.5)
      const float j1 = endpoint_eta(rules.mid, dark, el, eg), j2 = endpoint_eta(rules.mid, dark, eg, eh);
      const float cut = rules.half_ratio * (j1 >= j2 ? j1 : j2);
      m1 = j1 >= cut;
      m2 = j2 >= cut;
    }
    const bool f1 = t1 || m1, f2 = t2 || m2;
    const bool p1 = f1 && len1 >= rules.min_split, p2 = f2 && len2 >= rules.min_split;
    candidates[2 * k] = interval(lo, g);
    candidates[2 * k + 1] = interval(g, hi);
    uint32_t tag1 = 0, tag2 = 0;
    if constexpr (kCounters) count_c2(k, lo, g, hi, dark, s1, s2, t1, t2, m1, m2, p1, p2, tag1, tag2);
    sel[2 * k] = p1 ? static_cast<uint8_t>(1u | (tag1 << 1)) : 0;
    sel[2 * k + 1] = p2 ? static_cast<uint8_t>(1u | (tag2 << 1)) : 0;
    count_if(&counters[0], s1);
    count_if(&counters[0], s2);
    count_if(&counters[1], m1 && !s1);
    count_if(&counters[1], m2 && !s2);
    count_if(&counters[2], f1 && !p1);
    count_if(&counters[2], f2 && !p2);
    if (cls != nullptr) {  // a selected half is split only below the refinement cap
      const bool deeper = depth < rules.max_refine_levels;
      count_sample(diag, cls[g], true);
      count_jump(rules, dark, diag, sig[lo] != sig[g] && !(p1 && deeper), cls[lo], cls[g], el, eg);
      count_jump(rules, dark, diag, sig[g] != sig[hi] && !(p2 && deeper), cls[g], cls[hi], eg, eh);
      if (edge != nullptr && (cls[g] & kBandEdge) != 0) edge[owner[hi]] = 1;
    }
  }

  // P2-C C2: this round's sample g in a tagged subtree, the counts of the halves (lo, g) and (g, hi) (outers: lo_out
  // and the sibling, the sibling and hi_out), their smooth-sibling tests, the halves' tags and outer nodes.
  BALLA_HD void count_c2(size_t k, uint32_t lo, uint32_t g, uint32_t hi, float dark, bool s1, bool s2, bool t1, bool t2,
                         bool m1, bool m2, bool p1, bool p2, uint32_t& tag1, uint32_t& tag2) const {
    const uint32_t lo_out = static_cast<uint32_t>(outer[k] >> 32), hi_out = static_cast<uint32_t>(outer[k]);
    const uint8_t inherited = tag[k];
    count_if(&jump[kJumpSavedSamples], (inherited & kTagSaved) != 0);
    count_if(&jump[kSibSkippedSamples], inherited == kTagSkipped);  // below both: a node saving's (the two add)
    const bool deeper = depth < rules.max_refine_levels;
    const bool r1 = p1 && deeper, r2 = p2 && deeper;  // the splits happen
    tag1 = tag2 = inherited;
    if (rules.refine_signature && sig[lo] != sig[g]) {
      tag1 |= count_signature(rules, dark, jump, dt, irr, sig, groups, lo, g, lo_out, hi, depth, s1, !r1,
                              r1 && t1 && !m1, cls);
    }
    if (rules.refine_signature && sig[g] != sig[hi]) {
      tag2 |= count_signature(rules, dark, jump, dt, irr, sig, groups, g, hi, lo, hi_out, depth, s2, !r2,
                              r2 && t2 && !m2, cls);
    }
    if (m1 && !s1 && s2 && r1) tag1 |= sibling(dark, g, lo, lo_out, false, true);
    if (m2 && !s2 && s1 && r2) tag2 |= sibling(dark, g, hi, hi_out, true, true);
    // The smooth halves the run leaves whole (no midpoint flag) where a midpoint test runs: the node rules' additions.
    const bool tested = rules.midpoint && depth < rules.mid_levels && deeper;
    if (tested && !m1 && !s1 && s2 && dt[g] - dt[lo] >= rules.min_split) sibling(dark, g, lo, lo_out, false, false);
    if (tested && !m2 && !s2 && s1 && dt[hi] - dt[g] >= rules.min_split) sibling(dark, g, hi, hi_out, true, false);
    cand_outer[2 * k] = interval(lo_out, hi);
    cand_outer[2 * k + 1] = interval(lo, hi_out);
  }

  // A smooth half (a, m) or (m, a) (right: m after a) beside a jump half that shares a: the outer-node rule on m's
  // outer node b, or the forced split when b is missing or (m, b) is a jump interval. split: the run splits the half
  // (its midpoint flag; counted as a sibling flag, skip or forced split, tagged where skipped); else the node rules'
  // split of it, where they would split, is an addition.
  BALLA_HD uint8_t sibling(float dark, uint32_t a, uint32_t m, uint32_t b, bool right, bool split) const {
    const bool forced = b == kNoOuter || (right ? jump_of(rules, dark, dt, irr, sig, groups, m, b)
                                                : jump_of(rules, dark, dt, irr, sig, groups, b, m));
    bool flags = true;
    if (!forced) {
      const float h = static_cast<float>(right ? dt[m] - dt[a] : dt[a] - dt[m]);
      const float h_out = static_cast<float>(right ? dt[b] - dt[m] : dt[m] - dt[b]);
      flags = outer_node_flags(rules.mid, dark, irr + static_cast<size_t>(a) * groups,
                               irr + static_cast<size_t>(m) * groups, irr + static_cast<size_t>(b) * groups, h, h_out);
    }
    if (!split) {
      count_if(&jump[kSibAdded], flags);
      count_if(&jump[kSibAddedForced], forced);
      return 0;
    }
    count_if(&jump[kSibFlags], true);
    count_if(&jump[kSibForced], forced);
    count_if(&jump[kSibSkips], !flags);
    return flags ? 0 : kTagSkipped;
  }
};

// The selected candidates; with P2-C C2's counters (outer not null) their outer nodes and tags (the select flags'
// upper bits).
struct GatherList {
  const uint32_t* index;
  const uint64_t* candidates;
  uint64_t* list;
  const uint8_t* sel = nullptr;
  const uint64_t* cand_outer = nullptr;
  uint64_t* outer = nullptr;
  uint8_t* tag = nullptr;
  BALLA_HD void operator()(size_t k) const {
    list[k] = candidates[index[k]];
    if (outer != nullptr) {
      outer[k] = cand_outer[index[k]];
      tag[k] = static_cast<uint8_t>(sel[index[k]] >> 1);
    }
  }
};

// P2-C C2: a flag functor's pass with the counters compiled in (Round0Flags, HalfFlags: their jump and outer buffers
// set), so that the plain pass, the functor itself, carries none of their code.
template <class F>
struct Counting {
  F f;
  BALLA_HD void operator()(size_t i) const { f.template pass<true>(i); }
};

// K13 (assembly, the NumPy engine's WindowSamples.runs without sorting the lattice): refinement sample e is
// keyed (before, dt) for a stable sort and counted against the round-0 slot it precedes.
struct PoolKeys {
  uint32_t s0;
  const uint32_t* dt;
  const uint32_t* before;
  uint64_t* keys;
  uint32_t* values;
  uint32_t* counts;  // s0 + 1, zeroed

  BALLA_HD void operator()(size_t e) const {
    keys[e] = (static_cast<uint64_t>(before[e]) << (time::kWindowLog2 + 1)) | dt[s0 + e];
    values[e] = static_cast<uint32_t>(e);
    atomic_increment(&counts[before[e]]);
  }
};

// With cum[i] = refinement samples before or at slot i (= ex[i + 1], ex the exclusive scan of the
// counts; null when there are none), slot i goes to i + cum[i] and sorted refinement sample e to
// before + e.
struct ScatterSlots {
  int64_t t_w;
  const uint32_t* ex;  // null: no refinement samples
  const uint32_t* dt;
  const float* irr;
  int32_t groups;
  int64_t* t_out;
  float* irr_out;

  BALLA_HD void operator()(size_t s) const {
    const size_t f = s + (ex != nullptr ? ex[s + 1] : 0);
    t_out[f] = t_w + dt[s];
    for (int32_t g = 0; g < groups; ++g) irr_out[f * groups + g] = irr[s * groups + g];
  }
};

struct ScatterPool {
  int64_t t_w;
  uint32_t s0;
  const uint32_t* sorted;  // refinement samples by (before, dt)
  const uint32_t* before;
  const uint32_t* dt;
  const float* irr;
  int32_t groups;
  int64_t* t_out;
  float* irr_out;

  BALLA_HD void operator()(size_t e) const {
    const uint32_t p = sorted[e];
    const size_t f = before[p] + e, g = static_cast<size_t>(s0) + p;
    t_out[f] = t_w + dt[g];
    for (int32_t k = 0; k < groups; ++k) irr_out[f * groups + k] = irr[g * groups + k];
  }
};

// Each pixel's run offset in the window's runs, and its last sample (t_w + W) carried over.
struct PixelRuns {
  int64_t base;
  const uint32_t* off0;
  const uint32_t* ex;  // null: no refinement samples
  const float* irr;
  const uint64_t* sig;
  int32_t groups;
  const uint32_t* carried_at;
  float* carried_irr;
  uint64_t* carried_sig;
  int64_t* offsets;
  const uint32_t* cls = nullptr;  // class words (null: none)
  uint32_t* carried_class = nullptr;

  BALLA_HD void operator()(size_t a) const {
    offsets[a] = base + off0[a] + (ex != nullptr ? ex[off0[a] + 1] : 0);
    const size_t last = off0[a + 1] - 1, c = carried_at != nullptr ? carried_at[a] : a;
    for (int32_t g = 0; g < groups; ++g) carried_irr[c * groups + g] = irr[last * groups + g];
    carried_sig[c] = sig[last];
    if (cls != nullptr) carried_class[c] = cls[last];
  }
};

}  // namespace balla::engine
