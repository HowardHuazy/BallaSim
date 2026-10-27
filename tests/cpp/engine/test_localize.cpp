// P2-D D3a (plan p2d.md §3.1-§3.4, §6 row D3a): P&V's shadow mode (engine/localize.h) on the host WindowEngine with
// the analytic stand-in of fake_pv.h: the runs stay bit for bit, the counts do not depend on batching, every localized
// flip lies in its bracket, and the rules fire where they should (R0c, R2 on thin lines, rarely on straight edges, R5
// on curvature the steps do not explain).
#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "fake_pv.h"

using namespace balla::engine;
using namespace balla::engine::test;

namespace {

constexpr size_t kAll = std::numeric_limits<size_t>::max();

void check_same_runs(const FakeRuns& a, const FakeRuns& b) {
  CHECK(a.offsets == b.offsets);
  CHECK(a.t == b.t);
  CHECK(a.irr == b.irr);
  CHECK(a.last_irr == b.last_irr);
  CHECK(a.last_sig == b.last_sig);
  CHECK(a.stats.samples_by_round == b.stats.samples_by_round);
  CHECK(a.stats.flagged_signature == b.stats.flagged_signature);
  CHECK(a.stats.flagged_midpoint == b.stats.flagged_midpoint);
  CHECK(a.stats.diag == b.stats.diag);
}

// The counters but the rounds (a per-batch host count).
std::vector<uint64_t> counts(const WindowStats& st) {
  std::vector<uint64_t> v(st.pv.begin(), st.pv.end());
  v[kPvRounds] = 0;
  return v;
}

// The guard sets' probes nest: those of the intervals each accepts within those of the candidates it runs P&V on
// (R1-routed ones included), full within posterior, all within every candidate's.
void check_run_probes(const std::array<uint64_t, kPvWords>& pv) {
  CHECK(pv[kPvFull + 4] <= pv[kPvRunProbes]);
  CHECK(pv[kPvPost + 4] <= pv[kPvRunProbes + 1]);
  CHECK(pv[kPvRunProbes] <= pv[kPvRunProbes + 1]);
  CHECK(pv[kPvRunProbes + 1] <= pv[kPvCand + 4]);
}

// The time (ns) at which sub-sample offset (ux, uy) of `pixel` crosses the boundary (across = level), by bisection of
// the scene's own function (monotone in t).
int64_t crossing(const PvFakeScene& sc, uint32_t pixel, float ux, float uy, double level) {
  int64_t lo = sc.t_w - kW, hi = sc.t_w + 2 * kW;
  while (hi - lo > 1) {
    const int64_t m = lo + (hi - lo) / 2;
    (sc.across(pixel, static_cast<double>(ux), static_cast<double>(uy), m) > level ? hi : lo) = m;
  }
  return hi;  // the first ns past `level`
}

}  // namespace

TEST_CASE("P2-D D3a: P&V's shadow mode leaves every run bit for bit, batched or not, and its counts do not move") {
  for (const uint32_t kind : {0u, 1u, 2u, 3u}) {
    INFO("kind " << kind);
    PvFakeSetup on = pv_fake_setup(kind + 3, 90, kind);
    PvFakeSetup off = on;
    off.pv = false;
    const PvFakeRuns a = pv_fake_window<Space::kHost>(off, kAll), b = pv_fake_window<Space::kHost>(on, kAll);
    check_same_runs(a.runs, b.runs);
    CHECK_FALSE(a.runs.stats.pv_on);
    REQUIRE(b.runs.stats.pv_on);
    CHECK(b.runs.stats.pv[kPvCand] > 0);
    CHECK(b.pv_calls >= 1);  // round 0's call at N 256 (one N), and D3b's refinement calls for the state check
    CHECK(b.runs.rendered == a.runs.rendered);
    // A second run is the same, and so is a batched one (the counts are per pixel).
    const PvFakeRuns again = pv_fake_window<Space::kHost>(on, kAll);
    CHECK(counts(again.runs.stats) == counts(b.runs.stats));
    CHECK(again.runs.stats.pv[kPvRounds] == b.runs.stats.pv[kPvRounds]);
    const PvFakeRuns parts = pv_fake_window<Space::kHost>(on, 2'500'000);
    REQUIRE(parts.runs.stats.batches > 1);
    check_same_runs(a.runs, parts.runs);
    CHECK(counts(parts.runs.stats) == counts(b.runs.stats));
    CHECK(parts.runs.peak <= 2'500'000);  // the P&V buffers in footprint (the stage's freed once it ends)
    check_run_probes(b.runs.stats.pv);
  }
}

TEST_CASE("P2-D D3a: straight and wiggly edges: every flip localized within its bracket, few probes per flip") {
  for (const uint32_t kind : {0u, 2u}) {
    INFO("kind " << kind);
    const PvFakeRuns r = pv_fake_window<Space::kHost>(pv_fake_setup(7 + kind, 120, kind), kAll);
    const auto& pv = r.runs.stats.pv;
    REQUIRE(!r.flips.empty());
    CHECK(pv[kPvCand + 2] == r.flips.size());
    size_t done = 0, bad = 0, before = 0;
    const PvFakeSetup s = pv_fake_setup(7 + kind, 120, kind);  // the pixel ids (active pixel a)
    for (const PvFlip& x : r.flips) {
      before += x.before;  // the shadow only grows over a sub-sample (lit -> shadowed)
      if (x.phase != kPhDone) continue;
      ++done;
      const int64_t t = crossing(r.scene, s.base.pixel[x.a], x.ux, x.uy, 0.0);
      const int64_t lo = kTw + x.t0 + static_cast<int64_t>(x.lo) * x.tau, hi = kTw + x.t0 + static_cast<int64_t>(x.hi) * x.tau;
      bad += !(x.hi == x.lo + 1 && t > lo && t <= hi);
    }
    CHECK(bad == 0);
    CHECK(before == 0);
    CHECK(done == pv[kPvCand + 3]);
    CHECK(pv[kPvDropped] == 0);     // one flip per sub-sample: no third state
    CHECK(pv[kPvUnfinished] == 0);
    CHECK(pv[kPvCand + 3] == pv[kPvCand + 2]);  // every flip localized
    const double per_flip = static_cast<double>(pv[kPvCand + 4]) / static_cast<double>(pv[kPvCand + 3]);
    MESSAGE("kind " << kind << ": " << r.flips.size() << " flips, " << per_flip << " probes per flip, verify hits "
                    << pv[kPvVerifyHits] << " of " << pv[kPvPredicted] << " predicted, " << pv[kPvSeeds]
                    << " seeds, R2 " << pv[kPvRule + 6] << " / R5 " << pv[kPvRule + 8] << " of " << pv[kPvCand]
                    << " candidates");
    CHECK(pv[kPvPredicted] > pv[kPvSeeds]);
    // R5 never fires here: the tables' steps J are the irradiance's steps exactly (one surface, a = 0.5), so E - J is
    // flat at every node (the segmented scan of J over the pixel's slots).
    CHECK(pv[kPvRule + 8] == 0);
    if (kind == 0) {
      // The affine predictor is exact on a straight edge up to the localized flips' half-cell placement: mostly verify
      // hits (two probes), a gallop step otherwise.
      CHECK(per_flip < 3.0);
      CHECK(static_cast<double>(pv[kPvVerifyHits]) >= 0.75 * static_cast<double>(pv[kPvPredicted]));
      CHECK(static_cast<double>(pv[kPvRule + 6]) <= 0.3 * static_cast<double>(pv[kPvCand]));  // R2: seldom here
    }
  }
}

TEST_CASE("P2-D D3a: thin lines fire R2 at their nodes; a carried node without codes routes interval 1 (R0c)") {
  PvFakeSetup thin = pv_fake_setup(5, 120, 1);
  thin.width = 0.1;
  const PvFakeRuns r = pv_fake_window<Space::kHost>(thin, kAll);
  const auto& pv = r.runs.stats.pv;
  CHECK(pv[kPvCand] > 0);
  CHECK(pv[kPvRule + 6] > 0);  // R2
  CHECK(static_cast<double>(pv[kPvRule + 6]) >= 0.5 * static_cast<double>(pv[kPvCand] - pv[kPvRule + 2]));
  // Every flip of an interval with both directions: localized or dropped (R1), none unfinished; a localized flip lies
  // in its bracket (entry at across 0, exit at across = width).
  CHECK(pv[kPvUnfinished] == 0);
  check_run_probes(pv);
  MESSAGE("thin line: R1 " << pv[kPvRule + 10] << " intervals, dropped flips " << pv[kPvDropped] << "; probes: full "
                           << pv[kPvFull + 4] << " (run " << pv[kPvRunProbes] << "), posterior " << pv[kPvPost + 4]
                           << " (run " << pv[kPvRunProbes + 1] << "), all " << pv[kPvCand + 4]);
  size_t bad = 0;
  for (const PvFlip& x : r.flips) {
    if (x.phase != kPhDone) continue;
    const int64_t t = crossing(r.scene, thin.base.pixel[x.a], x.ux, x.uy, x.before == 0 ? 0.0 : thin.width);
    const int64_t lo = kTw + x.t0 + static_cast<int64_t>(x.lo) * x.tau, hi = kTw + x.t0 + static_cast<int64_t>(x.hi) * x.tau;
    bad += !(t > lo && t <= hi);
  }
  CHECK(bad == 0);
  // Slow edges, so that many pixels are on an edge at t_w: without carried codes their interval 1 is R0c; with them,
  // none is.
  PvFakeSetup slow = pv_fake_setup(6, 120, 0);
  slow.speed = 100.0;
  const PvFakeRuns with = pv_fake_window<Space::kHost>(slow, kAll);
  slow.carried_codes = false;
  const PvFakeRuns without = pv_fake_window<Space::kHost>(slow, kAll);
  CHECK(with.runs.stats.pv[kPvR0Cause + 2] == 0);
  CHECK(without.runs.stats.pv[kPvR0Cause + 2] > 0);
  check_same_runs(with.runs, without.runs);
  CHECK(without.runs.stats.pv[kPvRule + 2] >= without.runs.stats.pv[kPvR0Cause + 2]);  // R0 counts them
}

TEST_CASE("P2-D D3a: R5 (the step-free midpoint test on E - J) fires on curvature the steps do not explain") {
  // A smooth term curve (dt / W)^2 in group 0: no flip, so J leaves it in E - J, and eta_L's triples see its second
  // difference (curve h^2 at a lattice step h W, against tol 0.002 at E ~ 0.5-1: the coarse levels fire).
  PvFakeSetup s = pv_fake_setup(9, 120, 0);
  s.curve = 2.0;
  const PvFakeRuns on = pv_fake_window<Space::kHost>(s, kAll);
  PvFakeSetup off = s;
  off.pv = false;
  check_same_runs(pv_fake_window<Space::kHost>(off, kAll).runs, on.runs);
  const auto& pv = on.runs.stats.pv;
  const uint64_t r5 = pv[kPvRule + 8];
  MESSAGE("R5 " << r5 << " of " << pv[kPvCand] << " candidates; causes [candidate, routed, none] "
                << pv[kPvR5Cause] << " / " << pv[kPvR5Cause + 1] << " / " << pv[kPvR5Cause + 2]);
  CHECK(r5 > 0);
  CHECK(r5 < pv[kPvCand]);  // the fine levels' triples stay below tol
  CHECK(pv[kPvR5Cause] + pv[kPvR5Cause + 1] + pv[kPvR5Cause + 2] == r5);
  CHECK(pv[kPvFull] + r5 <= pv[kPvCand]);  // the full set routes them
  check_run_probes(pv);
  // The flips of R5-routed intervals are still localized (shadow mode measures both guard sets on one run).
  CHECK(pv[kPvCand + 3] == pv[kPvCand + 2]);
}

TEST_CASE("P2-D D3a: the bracket cell and a uniform node's mask") {
  // tau: 2 clamp(tol_t, t_q, 10 us) rounded down to a power of two, at least the floors, at most len / 16.
  CHECK(pv_tau(1u << 20, 10, 512, 1.0, 1000.0) == 2048);  // tol_t = 0.002 / (1.15 * 1000) s = 1.74 us (DESIGN's example)
  CHECK(pv_tau(1u << 20, 10, 512, 0.0, 1000.0) == 16384);  // no contrast: the 10 us cap
  CHECK(pv_tau(1u << 20, 10, 512, 5.0, 1e6) == 1024);       // fast and steep: the floors 2 t_q and 2^shift
  CHECK(pv_tau(1u << 20, 12, 512, 5.0, 1e6) == 4096);       // the window clock's slot
  CHECK(pv_tau(1u << 14, 10, 512, 0.0, 1000.0) == 1024);    // len / 16
  CHECK(pv_tau(1u << 11, 10, 512, 0.0, 1000.0) == 1024);    // never below the slot
  // Three lights, N 256: the class word's shadowed pairs leave three masks of popcount 1; the signature picks one.
  for (const uint32_t mask : {0u, 1u, 2u, 4u, 3u, 6u, 7u}) {
    uint64_t sig = 0;
    for (uint32_t k = 0; k < 256; ++k) sig += balla::render::signature_term(k, 3, 2, mask);
    const uint32_t cls = (256u * static_cast<uint32_t>(__builtin_popcount(mask))) & balla::render::kClassShadowed;
    uint32_t got = 99;
    REQUIRE(pv_uniform_mask(cls, sig, 3, 2, 256, 3, got));
    CHECK(got == mask);
    const int pop = __builtin_popcount(mask);
    CHECK(pv_uniform_mask(cls, sig + 1, 3, 2, 256, 3, got) == (pop == 0 || pop == 3));  // unique by popcount alone
  }
}

TEST_CASE("P2-D D3a: a seed whose interval R1 routes does not hold its group's waiting flips") {
  // One group of five flips over intervals 1 and 2 (slots 1, 2): seeds 0 and 1 localized in slot 1, seed 2 still
  // bisecting in slot 2, which another flip's probe (another light, the other direction) routed this round (R1);
  // flips 3 and 4 wait in slot 1. The fit has two localized flips (no prediction), and the seed in slot 2 is dropped
  // at the next emit: the waiting flips bisect in that same round instead of waiting on it (no probe would be emitted,
  // and the rounds would end with them unfinished).
  const auto group = [] {
    std::vector<PvFlip> f(5);
    for (uint32_t q = 0; q < 5; ++q) {
      f[q] = PvFlip{};
      f[q].slot = q == 2 ? 2u : 1u;
      f[q].gb = 0;
      f[q].ge = 5;
      f[q].tau = 1024;
      f[q].hi = 64;
      f[q].p0 = f[q].p1 = kPvNoCell;
      f[q].ux = 0.1f * static_cast<float>(q);
      f[q].uy = 0.05f * static_cast<float>(q * q);
      f[q].phase = q < 2 ? kPhDone : q == 2 ? kPhSeed : kPhWait;
      f[q].flags = q < 3 ? kFlSeed : 0;
    }
    f[0].lo = 3;
    f[0].hi = 4;
    f[1].lo = 7;
    f[1].hi = 8;
    return f;
  };
  for (const bool r1 : {true, false}) {
    INFO("slot 2 routed by R1: " << r1);
    std::vector<PvFlip> f = group();
    const std::vector<uint32_t> route = {0u, kRtCand, kRtCand | (r1 ? kRtR1 : 0u)};
    std::vector<PvFit> fits(5);
    std::vector<uint32_t> nreq(6, 99);
    for (size_t q = 0; q < 5; ++q) PvFitGroups{f.data(), route.data(), fits.data(), 0xFFu}(q);
    CHECK(fits[0].stage == kStNone);
    CHECK(fits[0].pending == (r1 ? 0u : 1u));
    for (size_t q = 0; q <= 5; ++q) PvEmit{f.data(), fits.data(), route.data(), nreq.data(), 5}(q);
    CHECK(f[2].phase == (r1 ? kPhDrop : kPhSeed));
    CHECK(nreq[2] == (r1 ? 0u : 1u));
    for (const size_t q : {size_t{3}, size_t{4}}) {
      CHECK(f[q].phase == (r1 ? kPhBisect : kPhWait));
      CHECK(nreq[q] == (r1 ? 1u : 0u));
      if (r1) CHECK(f[q].p0 == 32u);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// P2-D D3b (plan p2d.md §3.1 Step F, §3.4 R3 / R4, §5.3, §6 row D3b): the predictor ladder with its waves, R3 in the
// edge frame and its propagation, R4 from the plan bits, the state check at bisection's midpoints, the would-be nodes.

namespace {

// The state check's words of class q (0 all, 1 full, 2 posterior).
uint64_t chk(const std::array<uint64_t, kPvWords>& pv, int q, int w) { return pv[kPvCheck + kPvChkWords * q + w]; }

double per_flip(const std::array<uint64_t, kPvWords>& pv) {
  return static_cast<double>(pv[kPvCand + 4]) / static_cast<double>(pv[kPvCand + 3]);
}

}  // namespace

TEST_CASE("P2-D D3b: the ladder: the projective stage is exact on a slanted caster, the local one leads on relief") {
  // Kind 4: each sub-sample's arrival time is a ratio of affine forms in u (F020); the leave-one-out RMS picks the
  // projective stage, whose predictions hit their tau-cell, against the affine stage alone (D3a's predictor).
  PvFakeSetup s = pv_fake_setup(21, 120, 4);
  const PvFakeRuns ladder = pv_fake_window<Space::kHost>(s, kAll);
  s.stages = 0;  // the affine stage only
  const PvFakeRuns affine = pv_fake_window<Space::kHost>(s, kAll);
  const auto& pv = ladder.runs.stats.pv;
  check_same_runs(ladder.runs, affine.runs);
  REQUIRE(pv[kPvPredicted] > 0);
  const uint64_t proj = pv[kPvStage + kStProj - 1];
  MESSAGE("slanted caster: " << per_flip(pv) << " probes per flip with the ladder (stages: affine / local / projective "
                             << "/ quadratic " << pv[kPvStage] << " / " << pv[kPvStage + 1] << " / " << proj << " / "
                             << pv[kPvStage + 3] << ", hits " << pv[kPvStage + 8 + kStProj - 1] << " of " << proj
                             << "), " << per_flip(affine.runs.stats.pv) << " with the affine stage alone");
  CHECK(pv[kPvCand + 3] == pv[kPvCand + 2]);  // every flip localized
  CHECK(pv[kPvUnfinished] == 0);
  // The fits read each localized flip at its cell's centre (+-tau / 2), and the first waves fit few flips: most, not
  // all, predictions hit (the CPU model's 2.2 probes per flip read exact flip times).
  CHECK(proj >= pv[kPvPredicted] / 2);
  CHECK(static_cast<double>(pv[kPvStage + 8 + kStProj - 1]) >= 0.85 * static_cast<double>(proj));
  CHECK(per_flip(pv) < 2.6);
  CHECK(per_flip(pv) < 0.5 * per_flip(affine.runs.stats.pv));
  CHECK(pv[kPvRounds] < static_cast<uint64_t>(kPvRoundsMax));
  CHECK(affine.runs.stats.pv[kPvStage + kStProj - 1] == 0);
  // Every localized flip in its bracket (the state at its cell's ends).
  size_t bad = 0;
  for (const PvFlip& x : ladder.flips) {
    if (x.phase != kPhDone) continue;
    const int64_t t = crossing(ladder.scene, s.base.pixel[x.a], x.ux, x.uy, 0.0);
    bad += !(t > kTw + x.t0 + static_cast<int64_t>(x.lo) * x.tau && t <= kTw + x.t0 + static_cast<int64_t>(x.hi) * x.tau);
  }
  CHECK(bad == 0);
  // Relief-like wiggles of 0.03 px at a 1 px wavelength, 0.06 px at 0.3 px (kind 2's default): fewer probes than the
  // affine stage alone on the first; on the second (slope 1.3, 60 us of arrival-time wiggle against 2 us cells) no
  // stage predicts and the selection costs at most a few percent.
  for (const double amp : {0.03, 0.06}) {
    INFO("wiggle " << amp);
    PvFakeSetup w = pv_fake_setup(8, 120, 2);
    w.wiggle = amp;
    w.wiggle_k = amp == 0.03 ? 6.283185307179586 : 20.94395;
    const PvFakeRuns wl = pv_fake_window<Space::kHost>(w, kAll);
    w.stages = 0;
    const PvFakeRuns wa = pv_fake_window<Space::kHost>(w, kAll);
    const auto& q = wl.runs.stats.pv;
    MESSAGE("wiggle " << amp << " px: " << per_flip(q) << " probes per flip with the ladder (affine / local / "
                      << "projective / quadratic " << q[kPvStage] << " / " << q[kPvStage + 1] << " / "
                      << q[kPvStage + 2] << " / " << q[kPvStage + 3] << ", hits " << q[kPvVerifyHits] << " of "
                      << q[kPvPredicted] << "), " << per_flip(wa.runs.stats.pv) << " with the affine stage alone");
    if (amp == 0.03) {
      CHECK(per_flip(q) < per_flip(wa.runs.stats.pv));
    } else {
      CHECK(per_flip(q) < 1.05 * per_flip(wa.runs.stats.pv));
    }
    // The stages' counts cover the predicted flips, their probes and hits.
    uint64_t flips = 0, probes = 0, hits = 0;
    for (int st = 0; st < 4; ++st) {
      flips += q[kPvStage + st];
      probes += q[kPvStage + 4 + st];
      hits += q[kPvStage + 8 + st];
    }
    CHECK(flips == q[kPvPredicted]);
    CHECK(hits == q[kPvVerifyHits]);
    CHECK(probes <= q[kPvCand + 4]);
  }
}

TEST_CASE("P2-D D3b: R3 fires on a thin line and its neighbours (propagation), never on a half-plane") {
  for (const uint32_t kind : {0u, 2u, 4u}) {  // half-planes: straight, wiggly, slanted
    INFO("kind " << kind);
    const PvFakeRuns r = pv_fake_window<Space::kHost>(pv_fake_setup(30 + kind, 120, kind), kAll);
    CHECK(r.runs.stats.pv[kPvCand + 3] > 0);
    CHECK(r.runs.stats.pv[kPvRuleB + 2] == 0);  // R3
    CHECK(r.runs.stats.pv[kPvRuleB + 4] == 0);  // R3's propagation
  }
  for (const double width : {0.05, 0.1, 0.2}) {
    INFO("thin line, width " << width);
    PvFakeSetup s = pv_fake_setup(33, 120, 1);
    s.width = width;
    const PvFakeRuns r = pv_fake_window<Space::kHost>(s, kAll);
    const auto& pv = r.runs.stats.pv;
    MESSAGE("thin line " << width << " px: R3 " << pv[kPvRuleB + 2] << ", propagated " << pv[kPvRuleB + 4] << " of "
                         << pv[kPvCand] << " candidates; R2 " << pv[kPvRule + 6] << "; posterior accepts "
                         << pv[kPvPost] << " intervals");
    CHECK(pv[kPvRuleB + 2] > 0);
    CHECK(pv[kPvRuleB + 4] > 0);
    CHECK(pv[kPvRuleB + 3] <= pv[kPvCand + 1]);  // their cost within the candidates'
    // A firing flip (kFlR3) lies in an interval with localized flips of both directions.
    size_t fired = 0, lone = 0;
    for (const PvFlip& x : r.flips) {
      if ((x.flags & kFlR3) == 0) continue;
      ++fired;
      bool other = false;
      for (const PvFlip& o : r.flips) other = other || (o.slot == x.slot && o.before != x.before && o.phase == kPhDone);
      lone += other ? 0 : 1;
    }
    CHECK(fired > 0);
    CHECK(lone == 0);
  }
}

TEST_CASE("P2-D D3b: R4 routes the candidates of the pixels the plan bits mark thin, in both guard sets") {
  PvFakeSetup s = pv_fake_setup(41, 90, 0);
  const PvFakeRuns none = pv_fake_window<Space::kHost>(s, kAll);
  s.plan.assign(3 * 90, 0);
  for (size_t a = 0; a < 90; ++a) s.plan[s.base.pixel[a]] = kPvThin;
  const PvFakeRuns all = pv_fake_window<Space::kHost>(s, kAll);
  check_same_runs(none.runs, all.runs);
  CHECK(none.runs.stats.pv[kPvRuleB] == 0);
  CHECK(all.runs.stats.pv[kPvRuleB] == all.runs.stats.pv[kPvCand]);
  CHECK(all.runs.stats.pv[kPvRuleB + 1] == all.runs.stats.pv[kPvCand + 1]);
  CHECK(all.runs.stats.pv[kPvFull] == 0);
  CHECK(all.runs.stats.pv[kPvPost] == 0);
  CHECK(none.runs.stats.pv[kPvPost] > 0);
  // Pre-routing: P&V's probes are spent only on the candidates a set's pre-routing leaves.
  CHECK(all.runs.stats.pv[kPvRunProbes] == 0);
  CHECK(all.runs.stats.pv[kPvRunProbes + 1] == 0);
  CHECK(all.runs.stats.pv[kPvCand + 4] == none.runs.stats.pv[kPvCand + 4]);  // shadow mode still localizes them
}

TEST_CASE("P2-D D3b: the state check at bisection's midpoints: transients, multi, no contradiction; a lying probe") {
  for (const uint32_t kind : {0u, 2u, 4u}) {  // one flip per sub-sample: every midpoint state is predicted
    INFO("kind " << kind);
    const PvFakeRuns r = pv_fake_window<Space::kHost>(pv_fake_setup(50 + kind, 120, kind), kAll);
    const auto& pv = r.runs.stats.pv;
    CHECK(chk(pv, 0, kPvChkMid) > 0);
    CHECK(chk(pv, 0, kPvChkCompared) > 0);
    CHECK(chk(pv, 0, kPvChkTransient) == 0);
    CHECK(chk(pv, 0, kPvChkMulti) == 0);
    CHECK(chk(pv, 0, kPvChkThird) == 0);
    CHECK(chk(pv, 0, kPvChkUnpredicted) == 0);
    CHECK(pv[kPvCheckBug] == 0);
    CHECK(pv[kPvCheckBug + 1] == 0);
    CHECK(pv[kPvCheckBug + 2] == 0);
    for (int w = 0; w < kPvChkWords; ++w) {  // the classes nest
      CHECK(chk(pv, 1, w) <= chk(pv, 2, w));
      CHECK(chk(pv, 2, w) <= chk(pv, 0, w));
    }
    // The would-be nodes: at most two per localized flip, at least one per interval with one.
    CHECK(pv[kPvNodes] > 0);
    CHECK(pv[kPvNodes] <= 2 * pv[kPvCand + 3]);
    CHECK(pv[kPvNodes + 1] <= pv[kPvNodes + 2]);
    CHECK(pv[kPvNodes + 2] <= pv[kPvNodes]);
  }
  // The thin line: transients (equal end states, the line at the midpoint), none in an interval R3 or R2 leaves
  // unrouted that bisection's midpoints see as a contradiction; the double edge: an unseen pair beside a flip (multi).
  PvFakeSetup thin = pv_fake_setup(55, 120, 1);
  thin.width = 0.1;
  const PvFakeRuns t = pv_fake_window<Space::kHost>(thin, kAll);
  PvFakeSetup dbl = pv_fake_setup(56, 120, 5);
  dbl.width = 0.1;
  const PvFakeRuns d = pv_fake_window<Space::kHost>(dbl, kAll);
  MESSAGE("thin line: midpoints " << chk(t.runs.stats.pv, 0, kPvChkMid) << ", transients "
                                  << chk(t.runs.stats.pv, 0, kPvChkTransient) << " (full "
                                  << chk(t.runs.stats.pv, 1, kPvChkTransient) << ", posterior "
                                  << chk(t.runs.stats.pv, 2, kPvChkTransient) << ") of "
                                  << chk(t.runs.stats.pv, 0, kPvChkCompared) << " compared; double edge: multi "
                                  << chk(d.runs.stats.pv, 0, kPvChkMulti) << ", transients "
                                  << chk(d.runs.stats.pv, 0, kPvChkTransient));
  CHECK(chk(t.runs.stats.pv, 0, kPvChkTransient) > 0);
  CHECK(t.runs.stats.pv[kPvCheckBug] == 0);
  CHECK(chk(d.runs.stats.pv, 0, kPvChkMulti) > 0);
  CHECK(d.runs.stats.pv[kPvCheckBug] == 0);
  // A probe that reports the state 6 us late: brackets the render contradicts where a midpoint lands on their ends.
  PvFakeSetup lie = pv_fake_setup(57, 120, 0);
  lie.lie = 6000;
  const PvFakeRuns l = pv_fake_window<Space::kHost>(lie, kAll);
  MESSAGE("lying probe: contradictions " << l.runs.stats.pv[kPvCheckBug] << ", multi "
                                         << chk(l.runs.stats.pv, 0, kPvChkMulti));
  CHECK(l.runs.stats.pv[kPvCheckBug] > 0);
}
