// The host WindowEngine (ENGINE_CPP §3 I1 tests): eta_L against NumPy's float32 evaluation, the run
// assembly against a brute-force sort, batched = unbatched, and the typed budget error.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fake.h"

using namespace balla::engine;
using namespace balla::engine::test;

TEST_CASE("eta_L equals NumPy's float32 evaluation, including at the tolerance") {
  // sampling.MidpointTest.for_groups(0.002, 1e-3, [constant, 120 Hz a1 = 0.2, pulse duty 0.06 floor 0]).eta
  const MidpointParams p = midpoint_params(0.002, 1e-3, {{1.0, 1.0}, {0x1.999999999999ap-1, 0x1.3333333333333p+0},
                                                         {0.0, 0x1.0aaaaaaaaaaabp+4}});
  CHECK(p.mediant);
  struct Golden {
    float a[3], m[3], b[3], eta;
  };
  const Golden goldens[] = {
      {{0x1.f43aaap-3f, 0x1.71cad0p-1f, 0x1.ddec1cp-2f}, {0x1.c9dc50p-3f, 0x1.2fba02p-1f, 0x1.24679ap-2f},
       {0x1.ed42bap-3f, 0x1.ef3ab4p-1f, 0x1.4710c0p-7f}, 0x1.9586e8p+0f},
      {{0x1.7647b2p-7f, 0x1.bb1aa0p-6f, 0x1.554d3ap-1f}, {0x1.937f74p-8f, 0x1.c4b81ap-4f, 0x1.a33d66p-2f},
       {0x1.0d810ap-8f, 0x1.d6aff2p-3f, 0x1.63e072p-14f}, 0x1.821838p+5f},
      {{0x1.397920p-23f, 0x1.1b9230p-1f, 0x1.034610p-1f}, {0x1.808382p-16f, 0x1.ab1524p-2f, 0x1.513cfep-2f},
       {0x1.7d0b26p-15f, 0x1.178e96p-3f, 0x1.9f0b22p-4f}, 0x1.d57b64p-1f},
      {{0x1.a3ad9ep-4f, 0x1.c7eb26p-6f, 0x1.61a0f2p-6f}, {0x1.0031dap-1f, 0x1.ac2dd2p-4f, 0x1.4a77c2p-4f},
       {0x1.8b047ap-1f, 0x1.fe3636p-3f, 0x1.164d80p-3f}, 0x1.15fa50p+0f},
      {{0x1.0f40aep-6f, 0x1.69213ep-4f, 0x1.070cc4p-3f}, {0x1.1e76bep-4f, 0x1.02c514p-4f, 0x1.e25002p-5f},
       {0x1.f67484p-4f, 0x1.f0e20ap-7f, 0x1.b860e2p-20f}, 0x1.bd1fa8p+1f},
      {{0x1.5b4708p-3f, 0x1.f91e96p-1f, 0.0f}, {0x1.7f2e8ep-4f, 0x1.f05ac8p-2f, 0x1.e35164p-3f},
       {0x1.d2c6f8p-8f, 0x1.535fc6p-2f, 0x1.088978p-7f}, 0x1.ddd334p+3f},
  };
  for (const Golden& g : goldens) CHECK(midpoint_eta(p, g.a, g.m, g.b) == g.eta);
  // eta == float(tol) does not exceed: the double tolerance is compared as float32 (NEP 50), and
  // 0.001999974249793457 rounds to 16777 * 2^-23.
  const MidpointParams q = midpoint_params(0.001999974249793457, 0.0, {{1.0, 1.0}});
  const float one[kMaxGroups] = {1.0f}, at[kMaxGroups] = {1.0f + 16777.0f * 0x1p-23f},
              above[kMaxGroups] = {1.0f + 16778.0f * 0x1p-23f};
  CHECK(midpoint_eta(q, one, at, one) == 16777.0f * 0x1p-23f);
  CHECK_FALSE(midpoint_exceeds(q, one, at, one));
  CHECK(midpoint_exceeds(q, one, above, one));
  // A pixel that sees nothing never refines (0 / 0 is NaN, and NaN propagates through the mediant).
  const MidpointParams z = midpoint_params(0.002, 0.0, {{1.0, 1.0}, {0.0, 2.0}});
  const float zero[kMaxGroups] = {};
  CHECK(std::isnan(midpoint_eta(z, zero, zero, zero)));
  CHECK_FALSE(midpoint_exceeds(z, zero, zero, zero));
}

TEST_CASE("the run assembly equals a brute-force sort of the samples") {
  for (uint32_t seed = 0; seed < 4; ++seed) {
    for (const int mode : {0, 1, 2}) {  // the lattice, the fixed-rate grid, the economy schedule (I7)
      const bool fixed = mode == 1;
      INFO("seed " << seed << ", mode " << mode);
      const FakeSetup s = mode == 2 ? economy(fake_setup(seed, 60, false)) : fake_setup(seed, fixed ? 12 : 60, fixed);
      const FakeRuns r = fake_window<Space::kHost>(s, std::numeric_limits<size_t>::max());
      const size_t n = s.pixel.size();
      CHECK(r.stats.batches == 1);
      CHECK(r.rendered == r.t.size() - n);  // every sample but the carried ones was rendered once
      REQUIRE(r.offsets.front() == 0);
      REQUIRE(static_cast<size_t>(r.offsets.back()) == r.t.size());
      // Brute force: every (pixel, t) of the runs, sorted by (pixel, t), with the renderer's values.
      std::vector<std::tuple<size_t, int64_t, float, float>> all, sorted;
      for (size_t a = 0; a < n; ++a) {
        for (int64_t i = r.offsets[a]; i < r.offsets[a + 1]; ++i) {
          all.emplace_back(a, r.t[i], r.irr[2 * i], r.irr[2 * i + 1]);
        }
      }
      sorted = all;
      std::sort(sorted.begin(), sorted.end());
      CHECK(all == sorted);
      for (size_t a = 0; a < n; ++a) {
        const int64_t first = r.offsets[a], last = r.offsets[a + 1] - 1;
        CHECK(r.t[first] == kTw);
        CHECK(r.t[last] == kTw + kW);
        CHECK(r.irr[2 * first] == s.carried_irr[2 * a]);
        for (int64_t i = first + 1; i <= last; ++i) {
          CHECK(r.t[i] > r.t[i - 1]);
          float e[2];
          uint64_t sig;
          s.render.eval(s.pixel[a], r.t[i], e, &sig);
          CHECK(std::memcmp(e, &r.irr[2 * i], sizeof(e)) == 0);
          if (i == last) {
            CHECK(std::memcmp(e, &r.last_irr[2 * a], sizeof(e)) == 0);
            CHECK(sig == r.last_sig[a]);
          }
        }
      }
      if (!fixed) CHECK(r.stats.samples_by_round.size() > 2);  // refinement ran
      // The P2-0b counters: every adjacent pair of the runs is a terminal interval (the fake's flips are shadow-only),
      // every sample after a pixel's first was rendered; the fixed-rate grid counts nothing.
      uint64_t jumps = 0, textured = 0;
      for (size_t a = 0; a < n; ++a) {
        uint64_t prev = s.carried_sig[a];
        for (int64_t i = r.offsets[a] + 1; i < r.offsets[a + 1]; ++i) {
          float e[2];
          uint64_t sig;
          s.render.eval(s.pixel[a], r.t[i], e, &sig);
          jumps += sig != prev ? 1 : 0;
          textured += sig != 0 ? 1 : 0;
          prev = sig;
        }
      }
      const auto& d = r.stats.diag;
      uint64_t hist = 0;
      for (int k = 0; k < kDiagBins; ++k) hist += d[kDiagHist + k];
      CHECK(d[kDiagShadowJumps] == (fixed ? 0 : jumps));
      CHECK(d[kDiagTextured] == (fixed ? 0 : textured));
      CHECK(hist == d[kDiagShadowJumps]);
      CHECK(d[kDiagShadowFlips] >= d[kDiagShadowJumps]);
      CHECK(d[kDiagCoverageJumps] == 0);
      if (!fixed) CHECK(jumps > 0);
    }
  }
}

void check_same_runs(const FakeRuns& a, const FakeRuns& b) {
  CHECK(a.offsets == b.offsets);
  CHECK(a.t == b.t);
  CHECK(a.irr == b.irr);
  CHECK(a.last_irr == b.last_irr);
  CHECK(a.last_sig == b.last_sig);
  CHECK(a.stats.samples_by_round == b.stats.samples_by_round);
  CHECK(a.stats.flagged_signature == b.stats.flagged_signature);
  CHECK(a.stats.flagged_midpoint == b.stats.flagged_midpoint);
  CHECK(a.stats.refine_cap_hits == b.stats.refine_cap_hits);
  CHECK(a.stats.diag == b.stats.diag);
}

TEST_CASE("pixel batches give the unbatched runs within the budget; a one-pixel overflow raises MemoryBudgetError") {
  const FakeSetup s = fake_setup(11, 80, false);
  const FakeRuns whole = fake_window<Space::kHost>(s, std::numeric_limits<size_t>::max());
  // The fake renderer refines heavily (3.8 samples per lattice sample, against <= 2 on the bench), so
  // the runs outgrow their reservation and are held twice while they grow: the budget is about twice
  // the runs (10648 samples, 255 kB) and a third below the unbatched peak (854 kB).
  const size_t budget = 600'000;
  const FakeRuns parts = fake_window<Space::kHost>(s, budget);
  CHECK(whole.stats.batches == 1);
  CHECK(parts.stats.batches > 1);
  CHECK(whole.peak > budget);
  CHECK(parts.peak <= budget);
  check_same_runs(parts, whole);
  // P2-C C2: call_round has an entry per call made, kRedoneCall for those of the batches that overflowed after
  // rendering (redone), and render_calls counts the others.
  const auto redone = [](const FakeRuns& r) {
    return static_cast<size_t>(std::count(r.stats.call_round.begin(), r.stats.call_round.end(), kRedoneCall));
  };
  for (const FakeRuns* r : {&whole, &parts}) {
    CHECK(r->stats.call_round.size() == r->calls);
    CHECK(r->stats.call_round.size() - redone(*r) == r->stats.render_calls);
  }
  CHECK(redone(whole) == 0);
  CHECK(redone(parts) > 0);  // a batch that fits its lattice and overflows in a refinement round
  CHECK_THROWS_AS(fake_window<Space::kHost>(s, 60'000), MemoryBudgetError);  // one pixel next to the runs
  CHECK_THROWS_AS(fake_window<Space::kHost>(s, 2'000), MemoryBudgetError);   // the runs of the lattice
  CHECK_THROWS_AS(fake_window<Space::kHost>(s, 1'000), MemoryBudgetError);   // the pixel offsets
}

TEST_CASE("an engine keeps its windows within the budget across windows (held capacity is released)") {
  // A larger window, then smaller ones, on one engine with one budget: the capacity the first leaves
  // is counted, and released when a batch of a later window would not fit next to it.
  const size_t budget = 700'000;
  WindowEngine<Space::kHost> engine;
  size_t batched = 0;
  for (const auto& [seed, n] : {std::pair<uint32_t, size_t>{3, 90}, {4, 60}, {5, 90}, {6, 20}}) {
    INFO("seed " << seed);
    const FakeSetup s = fake_setup(seed, n, false);
    const FakeRuns ref = fake_window<Space::kHost>(s, std::numeric_limits<size_t>::max());
    const FakeRuns r = fake_window(engine, s, budget);
    CHECK((r.stats.batches > 1) == (ref.peak > budget));  // batched exactly when the unbatched peak exceeds the budget
    batched += r.stats.batches > 1 ? 1 : 0;
    CHECK(r.peak <= budget);
    CHECK(engine.held_bytes() <= budget);
    check_same_runs(r, ref);
  }
  CHECK(batched == 2);  // windows 1 and 3 (peaks of 946 and 871 kB unbatched)
}

TEST_CASE("level_cap and the fixed-rate sample count follow sampling.py") {
  const int64_t levels[] = {6, 6, 5, 2, 1, 1};
  const int64_t full = 65 + 65 + 33 + 5 + 3 + 3;
  CHECK_FALSE(level_cap(levels, 6, full, 1).has_value());
  CHECK(level_cap(levels, 6, full - 1, 1) == 5);
  CHECK_THROWS_AS(level_cap(levels, 6, 17, 1), std::invalid_argument);
  CHECK(fixed_rate_samples(0, 1000) == 8389);              // 1000 .. 8388000, then 2^23
  CHECK(fixed_rate_samples(kW, 1 << 20) == 8);             // 2^23 + k 2^20 for k = 1..7, then the end
  CHECK(fixed_rate_samples(37 * kW, 1000) == 8390);        // 310379000 .. 318767000, then the end
}

TEST_CASE("eta_L's per-pixel dark term (a gain map, D1) is each pixel's own: a pixel's runs are those of its dark") {
  // Odd pixels take a dark term that silences eta_L (100 against samples near 1), even ones the setup's 1e-3; every
  // pixel's runs equal those of a window with its dark term for all, on the lattice and under the economy schedule
  // (eta_J's gate and tau stop read it too).
  for (const bool es : {false, true}) {
    INFO("economy " << es);
    const FakeSetup lo = es ? economy(fake_setup(5, 60, false)) : fake_setup(5, 60, false);
    FakeSetup hi = lo, map = lo;
    hi.cfg.rules.mid.dark = 100.0f;
    std::vector<float> dark(lo.pixel.back() + 1, -1.0f);
    for (size_t a = 0; a < lo.pixel.size(); ++a) dark[lo.pixel[a]] = a % 2 ? 100.0f : lo.cfg.rules.mid.dark;
    map.cfg.rules.mid.dark_px = dark.data();
    const size_t all = std::numeric_limits<size_t>::max();
    const FakeRuns rl = fake_window<Space::kHost>(lo, all), rh = fake_window<Space::kHost>(hi, all),
                   rm = fake_window<Space::kHost>(map, all);
    CHECK(rl.stats.flagged_midpoint > rh.stats.flagged_midpoint);  // the dark term decides here
    for (size_t a = 0; a < lo.pixel.size(); ++a) {
      const FakeRuns& r = a % 2 ? rh : rl;
      const auto t = [](const FakeRuns& x, size_t i) {
        return std::vector<int64_t>(x.t.begin() + x.offsets[i], x.t.begin() + x.offsets[i + 1]);
      };
      const auto irr = [](const FakeRuns& x, size_t i) {
        return std::vector<float>(x.irr.begin() + 2 * x.offsets[i], x.irr.begin() + 2 * x.offsets[i + 1]);
      };
      CHECK(t(rm, a) == t(r, a));
      CHECK(irr(rm, a) == irr(r, a));
    }
  }
}

// The runs of active pixel a.
std::vector<int64_t> pixel_t(const FakeRuns& x, size_t a) {
  return std::vector<int64_t>(x.t.begin() + x.offsets[a], x.t.begin() + x.offsets[a + 1]);
}
std::vector<float> pixel_irr(const FakeRuns& x, size_t a) {
  return std::vector<float>(x.irr.begin() + 2 * x.offsets[a], x.irr.begin() + 2 * x.offsets[a + 1]);
}

// P2-B B2: pixels at N_base = 64 or N_band = 256 (seeded), the window's two classes rendered in calls of their own.
FakeSetup mixed(FakeSetup s, uint32_t seed) {
  std::mt19937 rng(seed);
  s.cfg.band_subsamples = 256;
  for (size_t a = 0; a < s.pixel.size(); ++a) s.n_log2.push_back(rng() % 3 == 0 ? 8 : 6);
  return s;
}

TEST_CASE("per-pixel N (P2-B B2): each pixel's runs are those of a uniform window at its N, batched or not") {
  for (uint32_t seed = 0; seed < 3; ++seed) {
    for (const int mode : {0, 1, 2}) {  // the lattice, the fixed-rate grid, the economy schedule (I7)
      const bool fixed = mode == 1;
      INFO("seed " << seed << ", mode " << mode);
      const FakeSetup s = mode == 2 ? economy(fake_setup(seed, 80, false)) : fake_setup(seed, fixed ? 12 : 80, fixed);
      FakeSetup hi = s;
      hi.cfg.subsamples = 256;
      const FakeSetup mix = mixed(s, seed + 100);
      const size_t all = std::numeric_limits<size_t>::max();
      const FakeRuns rl = fake_window<Space::kHost>(s, all), rh = fake_window<Space::kHost>(hi, all),
                     rm = fake_window<Space::kHost>(mix, all);
      uint64_t band = 0, samples = 0;
      for (size_t a = 0; a < s.pixel.size(); ++a) {
        const bool b = mix.n_log2[a] == 8;
        const FakeRuns& r = b ? rh : rl;
        CHECK(pixel_t(rm, a) == pixel_t(r, a));
        CHECK(pixel_irr(rm, a) == pixel_irr(r, a));
        CHECK(std::memcmp(&rm.last_irr[2 * a], &r.last_irr[2 * a], 2 * sizeof(float)) == 0);
        CHECK(rm.last_sig[a] == r.last_sig[a]);
        const auto rendered = static_cast<uint64_t>(rm.offsets[a + 1] - rm.offsets[a] - 1);
        band += b ? rendered : 0;
        samples += rendered;
      }
      CHECK(rh.irr != rl.irr);  // the N reaches the renderer
      CHECK(band > 0);
      CHECK(band < samples);
      CHECK(rm.stats.band_samples == band);
      CHECK(rm.band == band);  // every band sample in a call at 256 (and every call uniform: no NaN above)
      CHECK(rl.stats.band_samples == 0);
      // Batches split the classes anywhere: the same runs.
      const FakeRuns parts = fake_window<Space::kHost>(mix, fixed ? 4'000'000 : 300'000);
      CHECK(parts.stats.batches > 1);
      check_same_runs(parts, rm);
      CHECK(parts.stats.band_samples == band);
    }
  }
  // One class only: the uniform window, counters included; the classes need two distinct powers of two.
  FakeSetup s = fake_setup(7, 40, false), hi = s, band = mixed(s, 3);
  hi.cfg.subsamples = 256;
  band.n_log2.assign(s.pixel.size(), 8);
  const size_t all = std::numeric_limits<size_t>::max();
  check_same_runs(fake_window<Space::kHost>(band, all), fake_window<Space::kHost>(hi, all));
  band.n_log2.assign(s.pixel.size(), 6);
  check_same_runs(fake_window<Space::kHost>(band, all), fake_window<Space::kHost>(s, all));
  band.cfg.band_subsamples = 64;
  CHECK_THROWS_AS(fake_window<Space::kHost>(band, all), std::invalid_argument);
  band.cfg.band_subsamples = 96;
  CHECK_THROWS_AS(fake_window<Space::kHost>(band, all), std::invalid_argument);
}

TEST_CASE("a window checks its cancel token at each pixel batch and refinement round (BL-0b)") {
  const FakeSetup s = fake_setup(11, 80, false);
  const size_t all = std::numeric_limits<size_t>::max(), budget = 600'000;  // budget: batches, as above
  const FakeRuns whole = fake_window<Space::kHost>(s, all), parts = fake_window<Space::kHost>(s, budget);
  REQUIRE(parts.stats.batches > 1);
  // An unset token changes no run.
  CancelToken unset;
  FakeSetup u = s;
  u.cfg.cancel = &unset;
  check_same_runs(fake_window<Space::kHost>(u, all), whole);
  check_same_runs(fake_window<Space::kHost>(u, budget), parts);
  // Set before the window: no render call. Set during any call but the last: the next check point throws before
  // any further call (a refinement round of the batch, or the next batch's start), in batches or not.
  for (const auto& [bytes, calls] : {std::pair<size_t, size_t>{all, whole.calls}, {budget, parts.calls}}) {
    INFO("budget " << bytes);
    const auto out = cancel_outcomes(s, calls, [bytes = bytes](const FakeSetup& c) {
      fake_window<Space::kHost>(c, bytes);
    });
    size_t rounds = 0, batches = 0;
    for (size_t at = 0; at < calls; ++at) {
      INFO("token set at call " << at);
      CHECK(out[at].first == at);
      rounds += out[at].second == "window: refinement round: cancelled" ? 1 : 0;
      batches += out[at].second == "window: batch: cancelled" ? 1 : 0;
    }
    CHECK(out[0].second == "window: batch: cancelled");
    CHECK(rounds + batches == calls);
    CHECK(rounds > 0);
    CHECK((batches > 1) == (bytes == budget));         // between batches too
    CHECK(out[calls] == std::pair<size_t, std::string>{calls, ""});  // set at the last call: nothing left to stop
  }
}
