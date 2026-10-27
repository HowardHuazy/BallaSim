// Host = device: the primitives on random data, eta_L, and the WindowEngine with the analytic
// renderer (batched and not). Skipped without a CUDA device.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "device_helpers.h"
#include "engine/band_map.h"
#include "scenes.h"

using namespace balla::engine;
using namespace balla::engine::test;

namespace {
using H = Par<Space::kHost>;
}

TEST_CASE("device primitives equal the serial host primitives bit for bit") {
  if (!device_available()) {
    MESSAGE("no CUDA device: skipped");
    return;
  }
  std::mt19937_64 rng(5);
  for (const size_t n : {size_t{1}, size_t{1000}, size_t{300'001}}) {
    INFO("n = " << n);
    std::vector<uint32_t> a32(n);
    std::vector<uint64_t> a64(n);
    for (size_t i = 0; i < n; ++i) {
      a32[i] = static_cast<uint32_t>(rng() % 1000);
      a64[i] = rng() % 100000;
    }
    std::vector<uint32_t> h32(n);
    std::vector<uint64_t> h64(n);
    Temp<Space::kHost> temp;
    H::exclusive_scan(a32.data(), h32.data(), n, temp, nullptr);
    H::exclusive_scan(a64.data(), h64.data(), n, temp, nullptr);
    CHECK(device_exclusive_scan(a32) == h32);
    CHECK(device_exclusive_scan(a64) == h64);
    // Stability: many equal keys (above end_bit the key bits are ignored).
    std::vector<uint64_t> keys(n), hk(n);
    std::vector<uint32_t> values(n), hv(n);
    for (size_t i = 0; i < n; ++i) {
      keys[i] = (rng() % 64) | (rng() << 40);
      values[i] = static_cast<uint32_t>(i);
    }
    std::vector<uint64_t> k0 = keys;
    std::vector<uint32_t> v0 = values;
    REQUIRE(H::sort_pairs(k0.data(), hk.data(), v0.data(), hv.data(), n, 20, temp, nullptr));
    const auto [dk, dv] = device_sort_pairs(keys, values, 20);
    CHECK(dk == hk);
    CHECK(dv == hv);
    for (size_t i = 1; i < n; ++i) {
      REQUIRE(((hk[i - 1] & 0xFFFFF) < (hk[i] & 0xFFFFF) || ((hk[i - 1] & 0xFFFFF) == (hk[i] & 0xFFFFF) && hv[i - 1] < hv[i])));
    }
    std::vector<uint8_t> flags(n);
    for (size_t i = 0; i < n; ++i) flags[i] = static_cast<uint8_t>(rng() % 3 == 0);
    std::vector<uint32_t> sel(n);
    unsigned long long count = 0;
    H::select_flagged(flags.data(), sel.data(), n, &count, temp, nullptr);
    sel.resize(count);
    CHECK(device_select(flags) == sel);
    CHECK(device_count(flags) == host_count(flags));
    CHECK(host_count(flags) == static_cast<unsigned long long>(std::count(flags.begin(), flags.end(), 1)));
  }
}

TEST_CASE("eta_L on the device equals the host") {
  if (!device_available()) return;
  const MidpointParams p = midpoint_params(0.002, 1e-3, {{1.0, 1.0}, {0.8, 1.2}, {0.0, 16.7}});
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const size_t n = 50'000;
  std::vector<float> a(3 * n), m(3 * n), b(3 * n), h(n);
  for (size_t i = 0; i < 3 * n; ++i) {
    a[i] = u(rng) * u(rng) * u(rng);
    b[i] = u(rng) * u(rng);
    m[i] = 0.5f * (a[i] + b[i]) * (0.7f + 0.6f * u(rng));
    if (i % 97 == 0) a[i] = 0.0f;
  }
  for (size_t i = 0; i < n; ++i) h[i] = midpoint_eta(p, &a[3 * i], &m[3 * i], &b[3 * i]);
  const std::vector<float> d = device_eta(p, a, m, b);
  CHECK(std::memcmp(d.data(), h.data(), n * sizeof(float)) == 0);
}

TEST_CASE("the device WindowEngine equals the host one, batched or not") {
  if (!device_available()) return;
  for (uint32_t seed = 0; seed < 3; ++seed) {
    for (const int mode : {0, 1, 2}) {  // the lattice, the fixed-rate grid, the economy schedule (I7)
      const bool fixed = mode == 1;
      INFO("seed " << seed << ", mode " << mode);
      const FakeSetup s = mode == 2 ? economy(fake_setup(seed, 120, false)) : fake_setup(seed, fixed ? 16 : 120, fixed);
      const FakeRuns host = fake_window<Space::kHost>(s, std::numeric_limits<size_t>::max());
      for (const size_t budget : {std::numeric_limits<size_t>::max(), size_t{fixed ? 4'000'000u : 1'000'000u}}) {
        const FakeRuns dev = device_fake_window(s, budget);
        if (budget < std::numeric_limits<size_t>::max()) {
          CHECK(dev.stats.batches > 1);
          CHECK(dev.peak <= budget);  // buffers, CUB's temporary storage and growth copies included
        }
        CHECK(dev.offsets == host.offsets);
        CHECK(dev.t == host.t);
        CHECK(dev.irr == host.irr);
        CHECK(dev.last_irr == host.last_irr);
        CHECK(dev.last_sig == host.last_sig);
        CHECK(dev.stats.samples_by_round == host.stats.samples_by_round);
        CHECK(dev.stats.flagged_signature == host.stats.flagged_signature);
        CHECK(dev.stats.flagged_midpoint == host.stats.flagged_midpoint);
        CHECK(dev.stats.refine_cap_hits == host.stats.refine_cap_hits);
      }
    }
  }
}

TEST_CASE("the device WindowEngine checks its cancel token where the host one does (BL-0b)") {
  if (!device_available()) return;
  const FakeSetup s = fake_setup(0, 120, false);  // batched at 1 MB (the test above)
  // Unbatched: the host's outcomes. In batches (the device's own split): the token set at call k stops the window
  // with k calls made, the last call aside.
  const size_t all = std::numeric_limits<size_t>::max();
  const size_t calls = fake_window<Space::kHost>(s, all).calls;
  const auto host = cancel_outcomes(s, calls, [all](const FakeSetup& c) { fake_window<Space::kHost>(c, all); });
  CHECK(cancel_outcomes(s, calls, [all](const FakeSetup& c) { device_fake_window(c, all); }) == host);
  const FakeRuns parts = device_fake_window(s, 1'000'000);
  REQUIRE(parts.stats.batches > 1);
  const auto dev = cancel_outcomes(s, parts.calls, [](const FakeSetup& c) { device_fake_window(c, 1'000'000); });
  for (size_t at = 0; at < parts.calls; ++at) {
    INFO("token set at call " << at);
    CHECK(dev[at].first == at);
    CHECK_FALSE(dev[at].second.empty());
  }
  CHECK(dev.back() == std::pair<size_t, std::string>{parts.calls, ""});
}

TEST_CASE("a device engine keeps its windows within the budget across windows") {
  if (!device_available()) return;
  const size_t budget = 800'000;
  std::vector<FakeSetup> setups;
  for (const auto& [seed, n] : {std::pair<uint32_t, size_t>{3, 90}, {4, 60}, {5, 90}}) {
    setups.push_back(fake_setup(seed, n, false));
  }
  const std::vector<FakeRuns> runs = device_fake_windows(setups, budget);
  for (size_t i = 0; i < setups.size(); ++i) {
    INFO("window " << i);
    const FakeRuns host = fake_window<Space::kHost>(setups[i], std::numeric_limits<size_t>::max());
    CHECK((runs[i].stats.batches > 1) == (i != 1));  // the second fits unbatched (547 kB)
    CHECK(runs[i].peak <= budget);
    CHECK(runs[i].t == host.t);
    CHECK(runs[i].irr == host.irr);
    CHECK(runs[i].offsets == host.offsets);
    CHECK(runs[i].last_irr == host.last_irr);
    CHECK(runs[i].stats.samples_by_round == host.stats.samples_by_round);
  }
}

TEST_CASE("per-pixel N (P2-B B2): the device WindowEngine equals the host one, batched or not") {
  if (!device_available()) return;
  for (uint32_t seed = 0; seed < 2; ++seed) {
    for (const int mode : {0, 1, 2}) {  // the lattice, the fixed-rate grid, the economy schedule (I7)
      const bool fixed = mode == 1;
      INFO("seed " << seed << ", mode " << mode);
      FakeSetup s = mode == 2 ? economy(fake_setup(seed, 120, false)) : fake_setup(seed, fixed ? 16 : 120, fixed);
      std::mt19937 rng(seed + 40);
      s.cfg.band_subsamples = 256;
      for (size_t a = 0; a < s.pixel.size(); ++a) s.n_log2.push_back(rng() % 3 == 0 ? 8 : 6);
      const FakeRuns host = fake_window<Space::kHost>(s, std::numeric_limits<size_t>::max());
      REQUIRE(host.stats.band_samples > 0);
      for (const size_t budget : {std::numeric_limits<size_t>::max(), size_t{fixed ? 4'000'000u : 1'000'000u}}) {
        const FakeRuns dev = device_fake_window(s, budget);
        if (budget < std::numeric_limits<size_t>::max()) {
          CHECK(dev.stats.batches > 1);
          CHECK(dev.peak <= budget);  // the partition's buffers and scan storage included
        }
        CHECK(dev.offsets == host.offsets);
        CHECK(dev.t == host.t);
        CHECK(dev.irr == host.irr);
        CHECK(dev.last_irr == host.last_irr);
        CHECK(dev.last_sig == host.last_sig);
        CHECK(dev.stats.samples_by_round == host.stats.samples_by_round);
        CHECK(dev.stats.band_samples == host.stats.band_samples);
        if (dev.stats.batches == 1) CHECK(dev.band == host.band);  // a batch redone renders twice
        CHECK(dev.stats.diag == host.stats.diag);
      }
    }
  }
}

TEST_CASE("the per-pixel band's radii and cascade on the device equal the host's (P2-B B6c)") {
  // BandRadius on random seeds (plan, carried, both; motions with NaN and inf, which take the cap; tile floors) and the
  // cascade on its radii: host = device bit for bit, and the cascade's band each seed's L-inf ball.
  if (!device_available()) return;
  std::mt19937_64 rng(23);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (const auto& [w, h] : {std::pair<int32_t, int32_t>{96, 72}, {640, 480}, {37, 5}}) {
    INFO(w << " x " << h);
    const size_t n = static_cast<size_t>(w) * h;
    const int32_t tiles_x = (w + 7) / 8, cap = std::max(w, h);
    std::vector<double> seed_m(n, 0.0), tile_floor(static_cast<size_t>(tiles_x) * ((h + 7) / 8), 0.0);
    std::vector<uint8_t> kind(n, 0);
    std::vector<uint32_t> pixel_tile(n);
    for (size_t i = 0; i < n; ++i) {
      pixel_tile[i] = static_cast<uint32_t>((i / w) / 8 * tiles_x + (i % w) / 8);
      if (u(rng) < 0.04) {
        kind[i] = static_cast<uint8_t>(1 + rng() % 3);
        seed_m[i] = u(rng) < 0.01 ? std::numeric_limits<double>::quiet_NaN() : 30.0 * u(rng) * u(rng);
      }
    }
    for (double& f : tile_floor) f = u(rng) < 0.02 ? std::numeric_limits<double>::infinity() : 12.0 * u(rng) * u(rng);
    std::vector<int32_t> radius(n);
    std::vector<uint32_t> hist(static_cast<size_t>(cap) + 1, 0);
    H::for_each(n, BandRadius{seed_m.data(), kind.data(), tile_floor.data(), pixel_tile.data(), 0.95, cap,
                              radius.data(), hist.data()}, nullptr);
    const auto [d_radius, d_hist] = device_band_radius(seed_m, kind, tile_floor, pixel_tile, 0.95, cap);
    CHECK(d_radius == radius);
    CHECK(d_hist == hist);
    if (n > 10'000) CHECK(hist[static_cast<size_t>(cap)] > 0);  // NaN and inf take the cap
    const std::vector<int32_t> levels = band_levels(hist);
    std::vector<uint8_t> band(n), tmp(n);
    for (const BandDomain& dom : {BandDomain{0, 0, w, h}, band_domain(w / 4, h / 3, w / 2 + 1, h / 2 + 1, 7, w, h)}) {
      INFO("domain [" << dom.x0 << ", " << dom.x1 << ") x [" << dom.y0 << ", " << dom.y1 << ")");
      band_cascade<Space::kHost>(radius.data(), levels, w, h, dom, band.data(), tmp.data(), nullptr);
      CHECK(device_band_cascade(radius, levels, w, h, dom) == band);
    }
  }
}

TEST_CASE("the pixel stage with the obliquity factors and the plan bits: host = device bit for bit (P2-D D1)") {
  // plan_cases.h's random receivers, with and without I8's band and the seeds, with the factors and with the plan bits
  // alone: the rates, reaches, flags, plan bits, R4's widths, k_base, the tile totals and the counts of PvPlanPass
  // equal on both memory spaces.
  if (!device_available()) return;
  const balla::render::Camera cam = balla::render::test::pinhole(96, 72, 80.0, 0);
  const CameraGeometry geo(cam, 0.95);
  const size_t n = geo.n_pixels, tiles = static_cast<size_t>(geo.tiles_x) * static_cast<size_t>(geo.tiles_y);
  for (const bool band : {false, true}) {
    for (const uint32_t obliquity : {1u, 0u}) {
      INFO("band and seeds " << band << ", obliquity " << obliquity);
      PixelStageInput in = pixel_stage_input(geo, band, 31);
      in.st.params.obliquity = obliquity;
      const PixelStageRun h = pixel_stage_run<Space::kHost>(in), d = device_pixel_stage(in);
      CHECK(std::memcmp(h.k.data(), d.k.data(), n * sizeof(double)) == 0);
      CHECK(h.k_base.size() == (obliquity != 0 ? n : 0));
      CHECK(h.k_base.size() == d.k_base.size());
      CHECK((h.k_base.empty() || std::memcmp(h.k_base.data(), d.k_base.data(), n * sizeof(double)) == 0));
      CHECK(h.reach == d.reach);
      CHECK(h.flags == d.flags);
      CHECK(h.pv_plan == d.pv_plan);
      CHECK(std::memcmp(h.width.data(), d.width.data(), n * sizeof(float)) == 0);
      CHECK(h.counts == d.counts);
      CHECK(std::memcmp(h.tiles.k.data(), d.tiles.k.data(), tiles * sizeof(double)) == 0);
      CHECK(h.tiles.k_base.size() == d.tiles.k_base.size());
      CHECK((h.tiles.k_base.empty() ||
             std::memcmp(h.tiles.k_base.data(), d.tiles.k_base.data(), tiles * sizeof(double)) == 0));
      CHECK(h.tiles.reach == d.tiles.reach);
      CHECK(h.tiles.depend == d.tiles.depend);
      CHECK(h.tiles.seed == d.tiles.seed);
      if (band) {
        CHECK(std::memcmp(h.tiles.kcone.data(), d.tiles.kcone.data(), tiles * sizeof(double)) == 0);
        CHECK(std::memcmp(h.tiles.ksource.data(), d.tiles.ksource.data(), tiles * sizeof(double)) == 0);
      }
      // P2-D D1f: the band's tile results as built, under the factors only.
      CHECK(h.tiles.kcone0.size() == (band && obliquity != 0 ? tiles : 0));
      CHECK(h.tiles.kcone0.size() == d.tiles.kcone0.size());
      if (!h.tiles.kcone0.empty() && h.tiles.kcone0.size() == d.tiles.kcone0.size()) {
        CHECK(std::memcmp(h.tiles.kcone0.data(), d.tiles.kcone0.data(), tiles * sizeof(double)) == 0);
        CHECK(std::memcmp(h.tiles.ksource0.data(), d.tiles.ksource0.data(), tiles * sizeof(double)) == 0);
        CHECK(h.tiles.sreach0 == d.tiles.sreach0);
      }
    }
  }
}

TEST_CASE("P2-D D3a, D3b: P&V's shadow mode on the device equals the host: runs, counters and flips, batched or not") {
  if (!device_available()) {
    MESSAGE("no CUDA device: skipped");
    return;
  }
  // 4: kind 0 with curvature (R5 fires); D3b: 5, the slanted caster (the projective stage), 6, the double edge (the
  // state check's multi), 7, a lying probe (its contradictions), 8, R4's plan bits on every other pixel.
  for (const uint32_t kind : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
    for (const size_t budget : {std::numeric_limits<size_t>::max(), size_t{2'500'000}}) {
      INFO("kind " << kind << ", budget " << budget);
      const uint32_t scene = kind == 5 ? 4u : kind == 6 ? 5u : kind >= 7 ? 0u : kind % 4;
      PvFakeSetup s = pv_fake_setup(kind + 3, 90, scene);
      if (kind == 4) s.curve = 2.0;
      if (kind == 7) s.lie = 6000;
      if (kind == 8) {
        s.plan.assign(3 * 90, 0);
        for (size_t a = 0; a < 90; a += 2) s.plan[s.base.pixel[a]] = kPvThin;
      }
      const PvFakeRuns host = pv_fake_window<Space::kHost>(s, budget);
      const PvFakeRuns dev = device_pv_fake_window(s, budget);
      CHECK(dev.runs.offsets == host.runs.offsets);
      CHECK(dev.runs.t == host.runs.t);
      CHECK(dev.runs.irr == host.runs.irr);
      CHECK(dev.runs.last_irr == host.runs.last_irr);
      CHECK(dev.runs.last_sig == host.runs.last_sig);
      CHECK(dev.runs.stats.batches == host.runs.stats.batches);
      CHECK(dev.runs.stats.pv == host.runs.stats.pv);
      REQUIRE(dev.flips.size() == host.flips.size());
      size_t differ = 0;
      for (size_t f = 0; f < dev.flips.size(); ++f) {
        const PvFlip &a = dev.flips[f], &b = host.flips[f];
        differ += a.slot != b.slot || a.k != b.k || a.l != b.l || a.lo != b.lo || a.hi != b.hi || a.tau != b.tau ||
                  a.phase != b.phase || a.flags != b.flags || a.probes != b.probes || a.grp != b.grp ||
                  a.rank != b.rank || a.stage != b.stage || a.loc != b.loc;
      }
      CHECK(differ == 0);
      CHECK(host.runs.stats.pv[kPvCand] > 0);
    }
  }
}

TEST_CASE("P2-F F9b: the pixel stage with AO receivers: host = device bit for bit") {
  // plan_cases.h's random receivers with two moving bodies' AO reach, with and without I8's band and the seeds: the
  // rates, k_base, reaches, flags (kPlanAo among them), plan bits, the tile totals and the AO receivers' count equal on
  // both memory spaces.
  if (!device_available()) return;
  const balla::render::Camera cam = balla::render::test::pinhole(96, 72, 80.0, 0);
  const CameraGeometry geo(cam, 0.95);
  const size_t n = geo.n_pixels, tiles = static_cast<size_t>(geo.tiles_x) * static_cast<size_t>(geo.tiles_y);
  for (const bool band : {false, true}) {
    INFO("band and seeds " << band);
    const PixelStageInput in = pixel_stage_input(geo, band, 31, true);
    const PixelStageRun h = pixel_stage_run<Space::kHost>(in), d = device_pixel_stage(in);
    CHECK(std::memcmp(h.k.data(), d.k.data(), n * sizeof(double)) == 0);
    CHECK(std::memcmp(h.k_base.data(), d.k_base.data(), n * sizeof(double)) == 0);
    CHECK(h.reach == d.reach);
    CHECK(h.flags == d.flags);
    CHECK(h.pv_plan == d.pv_plan);
    CHECK(h.counts == d.counts);
    CHECK(std::memcmp(h.tiles.k.data(), d.tiles.k.data(), tiles * sizeof(double)) == 0);
    CHECK(h.tiles.depend == d.tiles.depend);
    CHECK(h.tiles.seed == d.tiles.seed);
    CHECK(h.tiles.ao_pixels == d.tiles.ao_pixels);
    CHECK(h.tiles.ao_pixels > 0);
  }
}
