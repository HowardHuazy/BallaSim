// The pixel kernel end to end through SensorModel: closed-form oracles (leak, ESIM ramp O13),
// invariances (render schedule, chunking, threads, resume) and the event bookkeeping.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sensor/sensor_model.h"

using namespace balla::sensor;

namespace {

SensorParams quiet(const std::string& preset_name) {
  SensorParams p = preset(preset_name);
  p.noise = false;
  p.hot_fraction = 0;
  p.leak_rate_hz = 0;
  return p;
}

// Samples of one shared PL irradiance function f at the given times, for `pixels` pixels.
struct Runs {
  std::vector<uint32_t> ids;
  std::vector<int64_t> t;
  std::vector<float> irr;
  size_t m = 0;
};

template <class F>
Runs make_runs(uint32_t pixels, const std::vector<int64_t>& times, F f) {
  Runs r;
  r.m = times.size();
  for (uint32_t p = 0; p < pixels; ++p) {
    r.ids.push_back(p);
    for (int64_t t : times) {
      r.t.push_back(t);
      r.irr.push_back(static_cast<float>(f(t)));
    }
  }
  return r;
}

Events advance(SensorModel& s, const Runs& r, int64_t t_end) {
  return s.advance(r.ids.data(), r.ids.size(), r.t.data(), r.irr.data(), r.m, nullptr, t_end);
}

std::vector<int64_t> linspace_ns(int64_t a, int64_t b, int n) {
  std::vector<int64_t> v;
  for (int i = 0; i <= n; ++i) v.push_back(a + (b - a) * i / n);
  return v;
}

void append(Events& dst, const Events& src) {
  auto cat = [](auto& d, const auto& s) { d.insert(d.end(), s.begin(), s.end()); };
  cat(dst.x, src.x);
  cat(dst.y, src.y);
  cat(dst.p, src.p);
  cat(dst.t_cross_ns, src.t_cross_ns);
  cat(dst.t_reported_ns, src.t_reported_ns);
  cat(dst.label, src.label);
  cat(dst.flags, src.flags);
  cat(dst.ds, src.ds);
  cat(dst.dn, src.dn);
  cat(dst.dl, src.dl);
}

// Every field of two event lists is equal.
void check_same(const Events& a, const Events& b) {
  CHECK(a.x == b.x);
  CHECK(a.y == b.y);
  CHECK(a.p == b.p);
  CHECK(a.t_cross_ns == b.t_cross_ns);
  CHECK(a.t_reported_ns == b.t_reported_ns);
  CHECK(a.label == b.label);
  CHECK(a.flags == b.flags);
  CHECK(a.ds == b.ds);
  CHECK(a.dn == b.dn);
  CHECK(a.dl == b.dl);
}

}  // namespace

TEST_CASE("a static pixel without noise or leak never fires (O1 at the sensor)") {
  for (const char* name : {"prophesee_gen3", "ideal"}) {
    SensorModel s(quiet(name), 16, 8, 3, {make_constant_waveform()}, 0, 2);
    std::vector<float> irr0(16 * 8);
    for (size_t i = 0; i < irr0.size(); ++i) irr0[i] = 1.0f + 300.0f * static_cast<float>(i % 7);
    s.init(0, irr0.data());
    size_t events = 0;
    for (int w = 1; w <= 20; ++w) {
      events += s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, w * (int64_t{1} << 23)).size();
    }
    CHECK(events == 0);
  }
}

TEST_CASE("leak-only events follow the closed form to the nanosecond") {
  SensorParams p = quiet("prophesee_gen3");
  p.leak_rate_hz = 40.0;
  p.leak_sigma_log10 = 0.0;
  p.readout_quantum_s = 0;
  p.readout_delay_s = 0;
  SensorModel s(p, 8, 4, 11, {make_constant_waveform()}, 0, 2);
  std::vector<float> irr0(32, 500.0f);
  const int64_t t0 = 12345;
  s.init(t0, irr0.data());
  const Events ev = s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, 300'000'000);
  const int64_t T_r = std::llround(p.refractory_s * 1e9);
  int checked = 0;
  for (uint32_t px = 0; px < 32; ++px) {
    const PixelMap& m = s.maps().pixels[px];
    // Delta(t) = phi0 + l (t - t0) reaches theta_on first at t0 + ceil((theta - phi0) / l); after a
    // release the reference restarts from Delta = 0, so the period is T_r + theta / l.
    int64_t expect = t0 + static_cast<int64_t>(std::ceil((m.theta_on - m.phi0) / m.leak * 1e9));
    for (size_t i = 0; i < ev.size(); ++i) {
      if (ev.y[i] * 8u + ev.x[i] != px) continue;
      CHECK(ev.p[i] == 1);
      CHECK(ev.label[i] == kLabelLeak);
      CHECK(std::llabs(ev.t_cross_ns[i] - expect) <= 1);
      CHECK(ev.dl[i] == doctest::Approx(m.theta_on).epsilon(1e-6));  // float32 field
      expect = ev.t_cross_ns[i] + T_r + static_cast<int64_t>(std::ceil(m.theta_on / m.leak * 1e9));
      ++checked;
    }
  }
  CHECK(checked > 64);
}

TEST_CASE("ESIM-bug regression O13: a 5-log ramp gives floor((5 + phi0) / theta) events per pixel") {
  // The ideal family on its fast cascade (also with the poles in reversed order, tau_o > tau_sf: the split of
  // substep.h depends only on which pole is faster) and on the instantaneous log response (D3).
  struct Case {
    double sigma;
    bool reversed;
    Photoreceptor pr;
  };
  for (const Case cs : {Case{0.0, false, Photoreceptor::kCascade}, Case{0.021, false, Photoreceptor::kCascade},
                       Case{0.0, true, Photoreceptor::kCascade}, Case{0.0, false, Photoreceptor::kInstantaneous},
                       Case{0.021, false, Photoreceptor::kInstantaneous}}) {
    const double sigma = cs.sigma;
    const bool reversed = cs.reversed;
    SensorParams p = preset("ideal");
    p.photoreceptor = cs.pr;
    p.sigma_theta_on = p.sigma_theta_off = sigma;
    if (reversed) std::swap(p.tau_o, p.tau_sf);
    const uint32_t W = 32, H = 32;
    SensorModel s(p, W, H, 99, {make_constant_waveform()}, 0, 4);
    const double I0 = 1000.0;
    std::vector<float> irr0(W * H, static_cast<float>(I0));
    s.init(0, irr0.data());
    // Log-linear ramp over 10 ms sampled every 10 us (PL in linear irradiance), then a hold.
    const int64_t T = 10'000'000;
    const Runs r = make_runs(W * H, linspace_ns(10'000, T, 999),
                             [&](int64_t t) { return I0 * std::exp(5.0 * static_cast<double>(t) / T); });
    Events ev = advance(s, r, T);
    const Events tail = s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, T + 20'000'000);
    std::vector<int> count(W * H, 0);
    for (size_t i = 0; i < ev.size(); ++i) {
      CHECK(ev.p[i] == 1);
      ++count[ev.y[i] * W + ev.x[i]];
    }
    for (size_t i = 0; i < tail.size(); ++i) ++count[tail.y[i] * W + tail.x[i]];
    const double d = p.dark_current_fA;
    const double swing = std::log((I0 * std::exp(5.0) + d) / (I0 + d));
    int exact = 0, skipped = 0, total = 0;
    for (uint32_t px = 0; px < W * H; ++px) {
      const PixelMap& m = s.maps().pixels[px];
      const double levels = (swing + m.phi0) / m.theta_on;
      if (std::fabs(levels - std::round(levels)) < 1e-6) {
        ++skipped;  // the last level sits at the asymptote: ambiguous by construction
        continue;
      }
      exact += count[px] == static_cast<int>(std::floor(levels));
      total += count[px];
    }
    MESSAGE("sigma_theta " << sigma << std::string(reversed ? " (reversed poles)" : "")
                           << std::string(cs.pr == Photoreceptor::kCascade ? " cascade" : " instantaneous")
                           << ": mean events/pixel "
                           << static_cast<double>(total) / (W * H - skipped));
    CHECK(exact == static_cast<int>(W * H) - skipped);
    if (sigma == 0.0) {
      CHECK(skipped == 0);
      for (int c : count) CHECK(c == 25);
    }
  }
}

TEST_CASE("events and labels are invariant to the render schedule (M3 V5 L4)") {
  // PL steps of +-1.5 log at 0.2 pA (M3 L4), sampled at its 7 breakpoints or with 400 extra points
  // on the same function. The lattice noise is identical by construction; the deterministic
  // outputs of the two schedules differ only within the integration tolerance, which bounds the
  // timing differences by (eps_h + eps_PL (1 + V)) / |dDelta/dt| (M3 B.17).
  const uint32_t W = 16, H = 8;
  const int64_t T = 50'000'000;
  const std::vector<int64_t> brk = {0, 4'000'000, 12'000'000, 20'000'000, 30'000'000, 38'000'000, T};
  const double I0 = 200.0;
  const std::vector<double> lev = {I0, I0, I0 * std::exp(1.5), I0 * std::exp(1.5), I0 * std::exp(0.4),
                                   I0 * std::exp(0.4), I0 * std::exp(1.0)};
  auto f = [&](int64_t t) {
    for (size_t i = 0; i + 1 < brk.size(); ++i)
      if (t <= brk[i + 1]) {
        const double frac = static_cast<double>(t - brk[i]) / static_cast<double>(brk[i + 1] - brk[i]);
        return lev[i] + (lev[i + 1] - lev[i]) * frac;
      }
    return lev.back();
  };
  std::vector<int64_t> dense(brk.begin() + 1, brk.end());
  std::mt19937_64 rng(5);
  std::uniform_int_distribution<int64_t> ut(1, T - 1);
  for (int i = 0; i < 400; ++i) dense.push_back(ut(rng));
  std::sort(dense.begin(), dense.end());
  dense.erase(std::unique(dense.begin(), dense.end()), dense.end());
  const std::vector<int64_t> sparse(brk.begin() + 1, brk.end());
  for (const double eps_L : {0.002, 2e-5}) {
    SensorParams p = preset("prophesee_gen3");
    p.theta_on = p.theta_off = 0.2;
    p.sigma_theta_on = p.sigma_theta_off = 0;
    p.hot_fraction = 0;
    p.eps_L = eps_L;
    Events out[2];
    for (int k = 0; k < 2; ++k) {
      SensorModel s(p, W, H, 2024, {make_constant_waveform()}, 0, 4);
      std::vector<float> irr0(W * H, static_cast<float>(I0));
      s.init(0, irr0.data());
      out[k] = advance(s, make_runs(W * H, k == 0 ? sparse : dense, f), T);
    }
    REQUIRE(out[0].size() == out[1].size());
    CHECK(out[0].size() > 200);
    // Per pixel in time order: near-simultaneous events of different pixels may swap in the output order.
    std::vector<size_t> ord[2];
    for (int k = 0; k < 2; ++k) {
      ord[k].resize(out[k].size());
      for (size_t i = 0; i < ord[k].size(); ++i) ord[k][i] = i;
      const Events& e = out[k];
      std::stable_sort(ord[k].begin(), ord[k].end(), [&](size_t a, size_t b) {
        return std::tie(e.y[a], e.x[a], e.t_cross_ns[a]) < std::tie(e.y[b], e.x[b], e.t_cross_ns[b]);
      });
    }
    std::vector<int64_t> dt;
    for (size_t j = 0; j < out[0].size(); ++j) {
      const size_t i = ord[0][j], m = ord[1][j];
      CHECK(out[0].x[i] == out[1].x[m]);
      CHECK(out[0].y[i] == out[1].y[m]);
      CHECK(out[0].p[i] == out[1].p[m]);
      CHECK(out[0].label[i] == out[1].label[m]);
      dt.push_back(std::llabs(out[0].t_cross_ns[i] - out[1].t_cross_ns[m]));
    }
    std::sort(dt.begin(), dt.end());
    const int64_t median = dt[dt.size() / 2], worst = dt.back();
    MESSAGE("eps_L " << eps_L << ": " << out[0].size() << " events, median |dt| = " << median
                     << " ns, max |dt| = " << worst << " ns");
    // Near-tangential noise crossings amplify the tolerance most (9 ns measured at both tolerances).
    CHECK(worst <= (eps_L > 1e-3 ? 100 : 20));
    CHECK(median <= (eps_L > 1e-3 ? 20 : 1));
  }
}

TEST_CASE("splitting a run into calls at the slice grain is bitwise invariant") {
  // The integrator calls advance once per 2^23 ns window and carries the window's last sample over
  // as the next call's first (docs/dev/ENGINE.md). With noise on, any 2^16 ns lattice boundary
  // qualifies as well (pixel.h, slice_grain). Events, labels, increments and states must equal those
  // of one call over the whole run, with noise, hot pixels and a 120 Hz light group, also for runs
  // that start before t = 0 and are cut at t = 0 (times are signed; floor division on the grains). The ideal family
  // runs both photoreceptors: the noise-free cascade and the instantaneous response (D3).
  struct Config {
    const char* preset;
    bool noise;
    int64_t t0;
    Photoreceptor photoreceptor;
  };
  const uint32_t W = 16, H = 16;
  const int64_t T = 2 * (int64_t{1} << 23);
  auto f = [](int64_t t, uint32_t px) {
    return 50.0 + 400.0 * (1.0 + std::sin(2e-7 * static_cast<double>(t) * (1 + px % 3) + 0.3 * px));
  };
  const Photoreceptor cascade = Photoreceptor::kCascade;
  for (const Config cfg : {Config{"prophesee_gen3", true, 0, cascade}, Config{"prophesee_gen3", true, -T / 2, cascade},
                           Config{"prophesee_gen3", false, 0, cascade}, Config{"ideal", false, -T / 2, cascade},
                           Config{"ideal", false, -T / 2, Photoreceptor::kInstantaneous}}) {
    const int64_t t0 = cfg.t0;
    for (const bool flicker : {false, true}) {
      for (const Precision prec : {Precision::kFp64, Precision::kFp32}) {
        SensorParams p = preset(cfg.preset);
        p.noise = cfg.noise;
        p.photoreceptor = cfg.photoreceptor;
        if (p.noise) p.hot_fraction = 0.05;
        std::vector<Waveform> groups = {make_constant_waveform()};
        if (flicker) groups.push_back(make_fourier_waveform(120.0, {0.2}, {0.3}));
        const int G = static_cast<int>(groups.size());
        std::vector<int64_t> splits = {t0 + T / 2};
        if (p.noise) splits = {t0 + 37 * kNoiseCellNs, t0 + T / 2, t0 + T / 2 + 5 * kNoiseCellNs};
        // Samples every ~0.5 ms (jittered) plus one at every split point.
        std::vector<int64_t> times = splits;
        for (int64_t t = 1; t <= T; t += 524'289) times.push_back(t0 + t + (t * 7919) % 100'000);
        times.push_back(t0 + T);
        std::sort(times.begin(), times.end());
        times.erase(std::unique(times.begin(), times.end()), times.end());
        while (times.back() > t0 + T) times.pop_back();
        auto model = [&] {
          auto s = std::make_unique<SensorModel>(p, W, H, 11, groups, 0, 3, prec);
          std::vector<float> irr0(W * H * G, 40.0f);
          for (uint32_t px = 0; px < W * H; ++px) irr0[px * G] = static_cast<float>(f(t0, px));
          s->init(t0, irr0.data());
          return s;
        };
        // Runs of the samples in [a, b] for every pixel.
        auto call = [&](SensorModel& s, int64_t a, int64_t b) {
          std::vector<uint32_t> ids;
          std::vector<int64_t> t;
          std::vector<float> irr;
          std::vector<int64_t> sel;
          for (int64_t ti : times)
            if (ti >= a && ti <= b && ti >= s.time_ns()) sel.push_back(ti);
          for (uint32_t px = 0; px < W * H; ++px) {
            ids.push_back(px);
            for (int64_t ti : sel) {
              t.push_back(ti);
              irr.push_back(static_cast<float>(f(ti, px)));
              for (int g = 1; g < G; ++g) irr.push_back(40.0f);
            }
          }
          return s.advance(ids.data(), ids.size(), t.data(), irr.data(), sel.size(), nullptr, b);
        };
        const auto single = model();
        const Events one = call(*single, t0, t0 + T);
        const auto split = model();
        Events parts;
        int64_t a = t0;
        splits.push_back(t0 + T);
        for (int64_t b : splits) {
          append(parts, call(*split, a, b));  // the sample at a is carried over (the model time)
          a = b;
        }
        INFO(cfg.preset << " noise " << cfg.noise << " t0 " << t0 << " flicker " << flicker << " fp32 "
                        << (prec == Precision::kFp32));
        CHECK(one.size() > 100);
        REQUIRE(parts.size() == one.size());
        CHECK(parts.t_cross_ns == one.t_cross_ns);
        CHECK(parts.t_reported_ns == one.t_reported_ns);
        CHECK(parts.x == one.x);
        CHECK(parts.y == one.y);
        CHECK(parts.p == one.p);
        CHECK(parts.label == one.label);
        CHECK(parts.flags == one.flags);
        CHECK(parts.ds == one.ds);
        CHECK(parts.dn == one.dn);
        CHECK(parts.dl == one.dl);
        const std::vector<PixelState> sa = single->states(), sb = split->states();
        CHECK(std::memcmp(sa.data(), sb.data(), sa.size() * sizeof(PixelState)) == 0);
      }
    }
  }
}

TEST_CASE("a run started before t = 0 gives the events of a later start, shifted (times are signed)") {
  // Regression: search() returned -1 for "no crossing" and the comparators tested t >= 0, so every
  // crossing at a negative model time was dropped, and a run initialized before t = 0 (the warm-up of
  // a run starting at t_s = 0) fired its backlog at the first t >= 0. Without noise the kernel is
  // invariant to a shift D on every grain it uses: D is a multiple of 2^23 ns (plain segments,
  // windows), 2^16 ns (noise cells) and 1 us (readout quantum). The light is constant (a waveform's
  // phase is absolute time), and both runs see the same function of t - t0, through calls that end
  // at the same window boundaries relative to t0 + D, starting off every grain before t = 0.
  const int64_t W23 = int64_t{1} << 23;
  const int64_t D = 125 * W23;  // 1.048576 s
  const uint32_t W = 16, H = 8;
  const int64_t t0 = -20'000'000 - 12'345;
  const std::vector<int64_t> ends = {-W23, 0, W23, 2 * W23};
  auto f = [](int64_t dt, uint32_t px) {
    return 300.0 * (1.0 + 0.8 * std::sin(2 * std::numbers::pi * 150e-9 * static_cast<double>(dt) + 0.4 * px));
  };
  std::vector<int64_t> rel;  // sample times relative to t0: every 97 us and at every call end
  for (int64_t d = 97'000; t0 + d < ends.back(); d += 97'000) rel.push_back(d);
  for (int64_t e : ends) rel.push_back(e - t0);
  std::sort(rel.begin(), rel.end());
  rel.erase(std::unique(rel.begin(), rel.end()), rel.end());
  for (const char* name : {"prophesee_gen3", "ideal"}) {
    for (const Precision prec : {Precision::kFp64, Precision::kFp32}) {
      SensorParams p = preset(name);
      p.noise = false;  // deterministic
      auto run = [&](int64_t shift, std::vector<PixelState>& states) {
        SensorModel s(p, W, H, 9, {make_constant_waveform()}, 0, 2, prec);
        std::vector<float> irr0(W * H);
        for (uint32_t px = 0; px < W * H; ++px) irr0[px] = static_cast<float>(f(0, px));
        s.init(t0 + shift, irr0.data());
        Events all;
        for (int64_t e : ends) {
          std::vector<int64_t> sel;
          for (int64_t d : rel)
            if (t0 + shift + d >= s.time_ns() && t0 + d <= e) sel.push_back(t0 + shift + d);
          Runs r;
          r.m = sel.size();
          for (uint32_t px = 0; px < W * H; ++px) {
            r.ids.push_back(px);
            for (int64_t t : sel) {
              r.t.push_back(t);
              r.irr.push_back(static_cast<float>(f(t - t0 - shift, px)));
            }
          }
          append(all, advance(s, r, e + shift));
        }
        states = s.states();
        return all;
      };
      std::vector<PixelState> sa, sb;
      const Events a = run(0, sa), b = run(D, sb);
      INFO(name << " fp32 " << (prec == Precision::kFp32));
      const auto negative = std::count_if(a.t_cross_ns.begin(), a.t_cross_ns.end(), [](int64_t t) { return t < 0; });
      CHECK(negative > 100);
      REQUIRE(a.size() == b.size());
      std::vector<int64_t> shifted(a.size()), reported(a.size());
      for (size_t i = 0; i < a.size(); ++i) {
        shifted[i] = a.t_cross_ns[i] + D;
        reported[i] = a.t_reported_ns[i] + D;
      }
      CHECK(b.t_cross_ns == shifted);
      CHECK(b.t_reported_ns == reported);
      CHECK(b.x == a.x);
      CHECK(b.y == a.y);
      CHECK(b.p == a.p);
      CHECK(b.label == a.label);
      CHECK(b.flags == a.flags);
      CHECK(b.ds == a.ds);
      CHECK(b.dn == a.dn);
      CHECK(b.dl == a.dl);
      for (PixelState& st : sb) {  // the shifted states moved back by D, bit for bit
        st.t_ref -= D;
        st.t_search -= D;
        if (st.blind_until != kNotBlind) st.blind_until -= D;
      }
      CHECK(std::memcmp(sa.data(), sb.data(), sa.size() * sizeof(PixelState)) == 0);
    }
  }
}

TEST_CASE("noise on: a warm-up before t = 0 matches one at a later time in distribution (no start-up burst)") {
  // With noise the realizations cannot align: every draw is keyed by its absolute node time (noise.h),
  // so a run shifted by D sees another, equally valid realization. The comparison is statistical,
  // on the review's reproduction (prophesee_gen3, constant 60 fA, the engine's 50.3 ms warm-up) with
  // t_s = 0 against t_s = D. Before the fix the warm-up before t = 0 gave no events, and the first
  // 10 ms after t_s about 2,100 on 80 x 60 pixels (500 of them exactly at t_s) against 140.
  const int64_t D = 125 * (int64_t{1} << 23);
  const int64_t warmup = 50'300'000, ms = 1'000'000;
  const uint32_t W = 160, H = 120;
  struct Counts {
    double warm = 0, warm_on = 0, first_ms = 0, rest = 0, at_ts = 0;
  };
  auto run = [&](int64_t t_s) {
    SensorModel s(preset("prophesee_gen3"), W, H, 1, {make_constant_waveform()}, 0, 4);
    std::vector<float> irr0(W * H, 60.0f);
    s.init(t_s - warmup, irr0.data());
    Counts c;
    const Events warm = s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, t_s);
    c.warm = static_cast<double>(warm.size());
    c.warm_on = static_cast<double>(std::count(warm.p.begin(), warm.p.end(), 1));
    const Events first = s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, t_s + ms);
    c.first_ms = static_cast<double>(first.size());
    c.at_ts = static_cast<double>(std::count(first.t_cross_ns.begin(), first.t_cross_ns.end(), t_s));
    c.rest = static_cast<double>(s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, t_s + 20 * ms).size());
    return c;
  };
  const Counts a = run(0), b = run(D);
  MESSAGE("t_s = 0 | D: warm-up " << a.warm << " | " << b.warm << " (ON " << a.warm_on << " | " << b.warm_on
                                  << "), first ms " << a.first_ms << " | " << b.first_ms << ", next 19 ms "
                                  << a.rest << " | " << b.rest << ", at t_s " << a.at_ts << " | " << b.at_ts);
  // Counts of two independent realizations: within 5 sigma of their Poisson difference.
  auto close = [](double x, double y) { return std::fabs(x - y) <= 5.0 * std::sqrt(x + y + 1.0); };
  CHECK(b.warm > 5000);
  CHECK(close(a.warm, b.warm));
  CHECK(close(a.warm_on, b.warm_on));
  CHECK(close(a.first_ms, b.first_ms));
  CHECK(close(a.rest, b.rest));
  CHECK(a.at_ts <= 2);
  CHECK(b.at_ts <= 2);
}

TEST_CASE("leak-type hot pixels fire at their target rate despite the refractory") {
  // Noise off makes every hot pixel leak-type and its events deterministic: after the first event
  // the period is T_r + theta / l, which the map sets to 1 / R for the drawn target R in
  // [10, 2000] Hz, so a pixel fires R T events in T, to within one.
  SensorParams p = preset("prophesee_gen3");
  p.noise = false;
  p.hot_fraction = 1.0;
  const uint32_t W = 16, H = 8;
  SensorModel s(p, W, H, 21, {make_constant_waveform()}, 0, 2);
  std::vector<float> irr0(W * H, 300.0f);
  s.init(0, irr0.data());
  const double T = 0.5;
  const Events ev = s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, static_cast<int64_t>(T * 1e9));
  std::vector<int> count(W * H, 0);
  for (size_t i = 0; i < ev.size(); ++i) {
    CHECK(ev.label[i] == kLabelHot);
    ++count[ev.y[i] * W + ev.x[i]];
  }
  double r_max = 0;
  for (uint32_t px = 0; px < W * H; ++px) {
    const double target = s.maps().hot_target_hz[px];
    REQUIRE(s.maps().pixels[px].hot == HotType::kLeak);
    CHECK(target >= p.hot_rate_min_hz);
    CHECK(target <= p.hot_rate_max_hz);
    CHECK(std::fabs(count[px] - target * T) <= 1.0 + 1e-6 * target * T);
    r_max = std::max(r_max, target);
  }
  CHECK(r_max > 1000.0);  // the refractory matters most at the high rates
}

TEST_CASE("results do not depend on the thread count or a save/load resume") {
  SensorParams p = preset("prophesee_gen3");
  p.hot_fraction = 0.05;
  const uint32_t W = 12, H = 10;
  const int64_t T = 30'000'000;
  auto f = [](int64_t t) { return 300.0 * (1.0 + 0.9 * std::sin(2e-7 * static_cast<double>(t))); };
  const Runs r = make_runs(W * H, linspace_ns(1000, T, 300), f);
  auto run = [&](int threads, bool resume) {
    SensorModel s(p, W, H, 77, {make_constant_waveform()}, 0, threads);
    std::vector<float> irr0(W * H, static_cast<float>(f(0)));
    s.init(0, irr0.data());
    Runs first = r, second = r;
    // Split the samples at T/2 into two calls.
    first.t.clear(); first.irr.clear(); second.t.clear(); second.irr.clear();
    size_t half = 0;
    while (r.t[half] <= T / 2) ++half;
    first.m = half;
    second.m = r.m - half;
    for (uint32_t px = 0; px < W * H; ++px) {
      for (size_t j = 0; j < r.m; ++j) {
        Runs& dst = j < half ? first : second;
        dst.t.push_back(r.t[px * r.m + j]);
        dst.irr.push_back(r.irr[px * r.m + j]);
      }
    }
    Events a = advance(s, first, T / 2);
    if (resume) {
      const std::vector<uint8_t> blob = s.save_state();
      SensorModel s2(p, W, H, 77, {make_constant_waveform()}, 0, threads);
      s2.load_state(blob.data(), blob.size());
      Events b = advance(s2, second, T);
      return std::make_tuple(a, b, s2.save_state());
    }
    Events b = advance(s, second, T);
    return std::make_tuple(a, b, s.save_state());
  };
  const auto [ref_a, ref_b, ref_state] = run(1, false);
  CHECK(ref_a.size() + ref_b.size() > 100);
  for (const auto& [a, b, state] : {run(5, false), run(3, true)}) {
    check_same(a, ref_a);
    check_same(b, ref_b);
    CHECK(state == ref_state);
  }
  // The kernel driven directly, one call per pixel, equals the model.
  SensorModel s(p, W, H, 77, {make_constant_waveform()}, 0, 1);
  std::vector<float> irr0(W * H, static_cast<float>(f(0)));
  s.init(0, irr0.data());
  const Events full = advance(s, r, T);
  SensorModel s0(p, W, H, 77, {make_constant_waveform()}, 0, 1);
  s0.init(0, irr0.data());
  std::vector<PixelState> states = s0.states();
  struct Sink {
    std::vector<RawEvent> ev;
    void push(const RawEvent& e) { ev.push_back(e); }
  } sink;
  KernelStats stats;
  for (uint32_t px = 0; px < W * H; ++px) {
    const InputRun in{r.t.data() + px * r.m, r.irr.data() + px * r.m, static_cast<int32_t>(r.m)};
    PixelKernel<double, Sink> k(s0.constants64(), s0.maps().pixels[px], px, states[px], 0, in, sink, stats);
    k.advance(T);
  }
  std::sort(sink.ev.begin(), sink.ev.end(), [](const RawEvent& a, const RawEvent& b) {
    return a.t_cross != b.t_cross ? a.t_cross < b.t_cross : a.pixel < b.pixel;
  });
  REQUIRE(sink.ev.size() == full.size());
  for (size_t i = 0; i < full.size(); ++i) {
    const RawEvent& e = sink.ev[i];
    CHECK(e.t_cross == full.t_cross_ns[i]);
    CHECK(e.pixel == full.y[i] * W + full.x[i]);
    CHECK(e.polarity == full.p[i]);
    CHECK(e.label == full.label[i]);
    CHECK(e.flags == full.flags[i]);
    CHECK(e.ds == full.ds[i]);
    CHECK(e.dn == full.dn[i]);
    CHECK(e.dl == full.dl[i]);
  }
  const std::vector<PixelState> model_states = s.states();
  CHECK(std::memcmp(states.data(), model_states.data(), states.size() * sizeof(PixelState)) == 0);
}

TEST_CASE("the stationary initial state and the advance do not depend on the thread count") {
  // 72 x 64 pixels: two blocks of the feature loop (4096 pixels), five of the transfer (1024) and 72
  // of the advance, over a graded frame so that the transfer takes several probes.
  SensorParams p = preset("prophesee_gen3");
  p.initial_state = InitialState::kStationary;
  p.stationary_burnin_ns = 250'000'000;
  p.stationary_probe_pixels = 8;
  const uint32_t W = 72, H = 64;
  std::vector<float> irr0(W * H);
  for (size_t i = 0; i < irr0.size(); ++i) irr0[i] = 5.0f + 0.05f * static_cast<float>(i);
  const int64_t t0 = 1000 * (int64_t{1} << 23);
  auto run = [&](int threads) {
    SensorModel s(p, W, H, 31, {make_fourier_waveform(120, {0.02}, {0})}, 0, threads, Precision::kFp32);
    s.init(t0, irr0.data());
    REQUIRE(s.stationary_probes() == 8);
    std::vector<uint8_t> at_init = s.save_state();
    Events ev = s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, t0 + 5'000'000);
    return std::make_tuple(at_init, ev, s.save_state());
  };
  const auto [init1, ev1, end1] = run(1);
  const auto [init4, ev4, end4] = run(4);
  MESSAGE(ev1.size() << " events in 5 ms");
  CHECK(init4 == init1);
  check_same(ev4, ev1);
  CHECK(end4 == end1);
}

TEST_CASE("event bookkeeping: identity, refractory and readout") {
  SensorParams p = preset("prophesee_gen3");
  p.hot_fraction = 0;
  const uint32_t W = 16, H = 16;
  const int64_t T = 100'000'000;
  SensorModel s(p, W, H, 5, {make_constant_waveform()}, 0, 4);
  std::vector<float> irr0(W * H, 40.0f);  // noise-dominated photocurrent near the BA peak
  s.init(0, irr0.data());
  const Events ev = s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, T);
  REQUIRE(ev.size() > 50);
  const int64_t T_r = std::llround(p.refractory_s * 1e9);
  const int64_t q = 1000, d0 = std::llround(p.readout_delay_s * 1e9);
  std::vector<int64_t> last(W * H, INT64_MIN / 2);
  for (size_t i = 0; i < ev.size(); ++i) {
    const uint32_t px = ev.y[i] * W + ev.x[i];
    const PixelMap& m = s.maps().pixels[px];
    const double D = static_cast<double>(ev.ds[i]) + static_cast<double>(ev.dn[i]) + static_cast<double>(ev.dl[i]);
    if (ev.p[i]) CHECK(D >= m.theta_on - 1e-5);
    else CHECK(D <= -m.theta_off + 1e-5);
    CHECK(std::fabs(D) <= (ev.p[i] ? m.theta_on : m.theta_off) + 1e-3);
    CHECK(ev.t_cross_ns[i] - last[px] > T_r);
    last[px] = ev.t_cross_ns[i];
    const int64_t delay = ev.t_reported_ns[i] - ev.t_cross_ns[i];
    CHECK(delay <= d0);
    CHECK(delay > d0 - q);
    CHECK(((ev.t_reported_ns[i] - s.readout_phase_ns()) % q + q) % q == 0);
    CHECK(ev.label[i] <= kLabelLeak);
  }
}

TEST_CASE_TEMPLATE("the first event after init leaves the reference offset Delta(t0) out of its label (F034)", T,
                   float, double) {
  const T cut = T(0.5);
  // The phase law, phi0 = 0.625 theta_on (values exact in fp32), dl = phi0 + l: an ON edge supplies the rest of the
  // level. The split since the reference books it LEAK (a_l 0.625); the own increments since t0 are the edge's: SIGNAL
  // (a_s 0.96).
  T ds = T(0.09), dn = T(0.00374), l = T(0.00001);
  CHECK(detail::split_label(ds, T(0.15625) + l, T(0.25), cut) == kLabelLeak);
  CHECK(detail::first_label(true, ds, dn, l, cut) == kLabelSignal);
  // The leak alone from phi0 to theta_on (as in the closed-form leak test): LEAK either way.
  CHECK(detail::first_label(true, T(0), T(0), T(0.09375), cut) == kLabelLeak);
  // Noise from phi0 to the level against a small opposing signal: NOISE.
  CHECK(detail::first_label(true, T(-0.01), T(0.10375), T(0), cut) == kLabelNoise);
  // OFF: phi0 opposes the edge, which supplies theta_off + phi0: SIGNAL either way.
  CHECK(detail::split_label(T(-0.40625), T(0.15625), T(-0.25), cut) == kLabelSignal);
  CHECK(detail::first_label(false, T(-0.40625), T(0), T(0), cut) == kLabelSignal);
  // A stationary state's Delta(t0) = 0.1875 (its aged leak and transferred offset, re-based into the offset): an ON
  // edge supplies 0.05 since t0, noise 0.0125 less the leak's 1e-5. Since the reference LEAK (a_l 0.75); own: SIGNAL.
  ds = T(0.05), dn = T(0.01249);
  CHECK(detail::split_label(ds, T(0.1875) + l, T(0.25), cut) == kLabelLeak);
  CHECK(detail::first_label(true, ds, dn, l, cut) == kLabelSignal);
  // No own increment (Delta(t0) on the level, which init never sets): NOISE, not a share of a zero total.
  CHECK(detail::first_label(true, T(0), T(0), T(0), cut) == kLabelNoise);
  CHECK(detail::first_label(false, T(0), T(0), T(0), cut) == kLabelNoise);
}

TEST_CASE("the static-cell fast path is bitwise equivalent to the general path") {
  SensorParams p = preset("prophesee_gen3");
  p.hot_fraction = 0.05;
  const uint32_t W = 16, H = 12;
  SensorModel s(p, W, H, 8, {make_constant_waveform()}, 0, 1);
  std::vector<float> irr0(W * H);
  for (size_t i = 0; i < irr0.size(); ++i) irr0[i] = 5.0f + 20.0f * static_cast<float>(i % 9);  // BA-rich currents
  s.init(0, irr0.data());
  KernelConstants<double> slow = s.constants64();
  slow.static_fast_path = false;
  std::vector<PixelState> fast_states = s.states(), slow_states = s.states();
  struct Sink {
    std::vector<RawEvent> ev;
    void push(const RawEvent& e) { ev.push_back(e); }
  } fast_sink, slow_sink, probe_sink;
  KernelStats fast_stats, slow_stats, probe_stats;
  // Odd pixels get a brightening ramp in the second window; the rest stay static throughout.
  const int64_t W23 = int64_t{1} << 23;
  std::vector<int64_t> t = {W23 + 3'000'000, W23 + 6'000'000};
  std::vector<float> ramp = {400.0f, 900.0f};
  int64_t quiet_cells = 0, total_cells = 0;
  for (int w = 0; w < 6; ++w) {
    for (uint32_t px = 0; px < W * H; ++px) {
      const bool moving = w == 1 && px % 2 == 1;
      const InputRun in = moving ? InputRun{t.data(), ramp.data(), 2} : InputRun{nullptr, nullptr, 0};
      // The leading quiet cells, by the fast path's own predicate on a copy of the state.
      PixelState copy = fast_states[px];
      const int64_t reached = PixelKernel<double, Sink>(s.constants64(), s.maps().pixels[px], px, copy, w * W23, in,
                                                        probe_sink, probe_stats)
                                  .advance_quiet(w * W23, (w + 1) * W23, true);
      quiet_cells += (reached - w * W23) / kNoiseCellNs;
      total_cells += W23 / kNoiseCellNs;
      PixelKernel<double, Sink> a(s.constants64(), s.maps().pixels[px], px, fast_states[px], w * W23, in, fast_sink,
                                  fast_stats);
      a.advance((w + 1) * W23);
      PixelKernel<double, Sink> b(slow, s.maps().pixels[px], px, slow_states[px], w * W23, in, slow_sink,
                                  slow_stats);
      b.advance((w + 1) * W23);
    }
  }
  REQUIRE(fast_sink.ev.size() == slow_sink.ev.size());
  CHECK(fast_sink.ev.size() > 100);
  for (size_t i = 0; i < fast_sink.ev.size(); ++i) {
    CHECK(fast_sink.ev[i].t_cross == slow_sink.ev[i].t_cross);
    CHECK(fast_sink.ev[i].dn == slow_sink.ev[i].dn);
  }
  CHECK(std::memcmp(fast_states.data(), slow_states.data(), fast_states.size() * sizeof(PixelState)) == 0);
  CHECK(fast_stats.cells == slow_stats.cells);
  MESSAGE(quiet_cells << " of " << total_cells << " cells quiet; refinements " << fast_stats.refinements
                      << " fast, " << slow_stats.refinements << " general");
  CHECK(quiet_cells * 2 > total_cells);  // the fast path did run
  // The fast path walks a non-excluded cell's tree without firing first, and the general path walks it
  // again when there is a crossing: bridge draws are counted twice there, so advance() took it.
  CHECK(fast_stats.refinements > slow_stats.refinements);
  CHECK(fast_stats.substeps == slow_stats.substeps);
}

TEST_CASE("transient diffusion (D-P2-2): static pixels keep the frozen bits; a mid-cell state resumes bit for bit") {
  // Pixels from 5 fA to 50 pA (a1 h from 0.003 to 33: the exact, blended and frozen regimes of the ratio rho), a
  // x10 step up or down on every pixel half a window in, at its own lattice phase. Static pixels never leave rho = 1,
  // so the two rules agree bit for bit there. After the steps a call ends inside a cell during the transients: the
  // saved state carries the cell's rho (PixelState::rho_m1), and a fresh model continues bit for bit, on the CPU in
  // both precisions and on the GPU, which also equals the CPU fp32 kernel.
  const uint32_t W = 32, H = 16, N = W * H;
  const std::vector<Waveform> groups{make_constant_waveform()};
  const int64_t kW = int64_t{1} << 23;
  std::vector<float> irr0(N);
  for (uint32_t px = 0; px < N; ++px) irr0[px] = static_cast<float>(5.0 * std::pow(10.0, 4.0 * (px % 64) / 63.0));
  SensorParams frozen = preset("prophesee_gen3");
  frozen.illuminance_scale = 1.0;
  SensorParams transient = frozen;
  transient.noise_diffusion = NoiseDiffusion::kTransient;
  const auto same_states = [](const SensorModel& a, const SensorModel& b) {
    const std::vector<PixelState> sa = a.states(), sb = b.states();
    return sa.size() == sb.size() && std::memcmp(sa.data(), sb.data(), sa.size() * sizeof(PixelState)) == 0;
  };
  const auto same_events = [](const Events& a, const Events& b) {
    CHECK(a.t_cross_ns == b.t_cross_ns);
    CHECK(a.x == b.x);
    CHECK(a.y == b.y);
    CHECK(a.p == b.p);
    CHECK(a.label == b.label);
    CHECK(a.ds == b.ds);
    CHECK(a.dn == b.dn);
  };
  {
    SensorModel a(frozen, W, H, 3, groups, 0, 2, Precision::kFp64, Device::kCpu);
    SensorModel b(transient, W, H, 3, groups, 0, 2, Precision::kFp64, Device::kCpu);
    a.init(0, irr0.data());
    b.init(0, irr0.data());
    const Events ea = a.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, 2 * kW);
    const Events eb = b.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, 2 * kW);
    REQUIRE(ea.size() > 0);
    REQUIRE(ea.size() == eb.size());
    same_events(ea, eb);
    CHECK(same_states(a, b));
  }
  std::vector<uint32_t> ids(N);
  std::vector<int64_t> t(2 * N);
  std::vector<float> irr(2 * N);
  for (uint32_t px = 0; px < N; ++px) {
    ids[px] = px;
    t[2 * px] = kW / 2 + 3001 * (px % 17);
    t[2 * px + 1] = t[2 * px] + 1;
    irr[2 * px] = irr0[px];
    irr[2 * px + 1] = irr0[px] * (px % 2 ? 10.0f : 0.1f);
  }
  const int64_t t_mid = kW / 2 + 3 * kNoiseCellNs + 20000;
  std::vector<Events> cpu32;
  std::vector<PixelState> cpu32_states;
  struct Arm {
    Device dev;
    Precision prec;
  };
  for (const Arm arm : {Arm{Device::kCpu, Precision::kFp64}, Arm{Device::kCpu, Precision::kFp32},
                        Arm{Device::kCuda, Precision::kFp32}}) {
    if (arm.dev == Device::kCuda && !sensor_cuda_available()) continue;
    INFO("device " << static_cast<int>(arm.dev) << ", precision " << static_cast<int>(arm.prec));
    SensorModel a(transient, W, H, 3, groups, 0, 2, arm.prec, arm.dev);
    SensorModel fresh(transient, W, H, 3, groups, 0, 2, arm.prec, arm.dev);
    SensorModel old(frozen, W, H, 3, groups, 0, 2, arm.prec, arm.dev);
    a.init(0, irr0.data());
    old.init(0, irr0.data());
    const Events e1 = a.advance(ids.data(), N, t.data(), irr.data(), 2, nullptr, t_mid);
    old.advance(ids.data(), N, t.data(), irr.data(), 2, nullptr, t_mid);
    const std::vector<PixelState> mid = a.states();
    int inside = 0, ratio = 0;
    for (const PixelState& s : mid) {
      inside += s.have_k1;
      ratio += s.have_k1 && s.rho_m1 != 0.0f;
    }
    CHECK(inside == static_cast<int>(N));
    CHECK(ratio > static_cast<int>(N) / 2);  // the transients' cells (rho = 1 only past a1 h = 10)
    CHECK(!same_states(a, old));              // the rules differ in transient cells
    const std::vector<uint8_t> state = a.save_state();
    fresh.load_state(state.data(), state.size());
    const Events e2 = a.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, 2 * kW);
    const Events f2 = fresh.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, 2 * kW);
    REQUIRE(e2.size() > 0);
    REQUIRE(e2.size() == f2.size());
    same_events(e2, f2);
    CHECK(same_states(a, fresh));
    if (arm.prec == Precision::kFp32) {
      if (arm.dev == Device::kCpu) {
        cpu32 = {e1, e2};
        cpu32_states = a.states();
      } else {
        REQUIRE(cpu32.size() == 2);
        REQUIRE(e1.size() == cpu32[0].size());
        REQUIRE(e2.size() == cpu32[1].size());
        same_events(e1, cpu32[0]);
        same_events(e2, cpu32[1]);
        const std::vector<PixelState> g = a.states();
        CHECK(std::memcmp(g.data(), cpu32_states.data(), g.size() * sizeof(PixelState)) == 0);
      }
    }
  }
}

namespace {

struct RawSink {
  std::vector<RawEvent> ev;
  void push(const RawEvent& e) { ev.push_back(e); }
};

// Six calls of 5 ms (they end inside noise cells) of PixelKernel<T, ..., kCache, ..., kP2> from s's states; odd pixels
// ramp in the second call.
template <class T, bool kCache, bool kP2>
std::tuple<std::vector<RawEvent>, std::vector<PixelState>, KernelStats> run_p2(const SensorModel& s,
                                                                               const KernelConstants<T>& kc) {
  std::vector<PixelState> states = s.states();
  RawSink sink;
  KernelStats stats;
  const int64_t call = 5'000'000;
  std::vector<int64_t> t = {7'000'000, 9'000'000};
  std::vector<float> ramp = {400.0f, 30.0f, 900.0f, 0.0f};
  for (int w = 0; w < 6; ++w) {
    for (uint32_t px = 0; px < states.size(); ++px) {
      const InputRun in = w == 1 && px % 2 == 1 ? InputRun{t.data(), ramp.data(), 2} : InputRun{nullptr, nullptr, 0};
      PixelKernel<T, RawSink, kCache, KernelStats, true, false, kP2>(kc, s.maps().pixels[px], px, states[px], w * call,
                                                                     in, sink, stats)
          .advance((w + 1) * call);
    }
  }
  return {sink.ev, states, stats};
}

template <class T, bool kCache>
void check_p2_off(const SensorModel& s, const KernelConstants<T>& kc) {
  const auto [ea, sa, ta] = run_p2<T, kCache, true>(s, kc);
  const auto [eb, sb, tb] = run_p2<T, kCache, false>(s, kc);
  REQUIRE(ea.size() == eb.size());
  CHECK(ea.size() > 100);
  for (size_t i = 0; i < ea.size(); ++i) {
    CHECK(ea[i].pixel == eb[i].pixel);
    CHECK(ea[i].polarity == eb[i].polarity);
    CHECK(ea[i].label == eb[i].label);
    CHECK(ea[i].flags == eb[i].flags);
    CHECK(ea[i].t_cross == eb[i].t_cross);
    CHECK(ea[i].ds == eb[i].ds);
    CHECK(ea[i].dn == eb[i].dn);
    CHECK(ea[i].dl == eb[i].dl);
  }
  CHECK(std::memcmp(sa.data(), sb.data(), sa.size() * sizeof(PixelState)) == 0);
  CHECK(ta.cells == tb.cells);
  CHECK(ta.substeps == tb.substeps);
  CHECK(ta.refinements == tb.refinements);
  CHECK(ta.leaf_intervals == tb.leaf_intervals);
  CHECK(ta.refinements > 0);
}

}  // namespace

TEST_CASE("frozen diffusion: the kernel with Phase 2's paths compiled out (kP2 false, P2-E E4) is bitwise the same") {
  // The GPU's frozen kernels take PixelKernel<..., kP2 = false>; the CPU's keep kP2 true. Under frozen diffusion rho is 1
  // in every cell, so the two agree bit for bit: events, states and counters, fp64 and fp32, with the block cache on and
  // off (the GPU's quiet and general passes), on 120 Hz flicker with ramps and calls that end inside cells.
  const uint32_t W = 16, H = 12, N = W * H;
  SensorParams p = preset("prophesee_gen3");
  p.hot_fraction = 0.05;
  REQUIRE(p.noise_diffusion == NoiseDiffusion::kFrozen);
  SensorModel s(p, W, H, 8, {make_constant_waveform(), make_fourier_waveform(120.0, {0.3}, {0.5})}, 0, 1);
  std::vector<float> irr0(N * 2);
  for (uint32_t i = 0; i < N; ++i) {
    irr0[2 * i] = 20.0f + 5.0f * static_cast<float>(i % 41);
    irr0[2 * i + 1] = 30.0f * static_cast<float>(i % 3);
  }
  s.init(0, irr0.data());
  check_p2_off<double, true>(s, s.constants64());
  check_p2_off<double, false>(s, s.constants64());
  check_p2_off<float, true>(s, s.constants32());
  check_p2_off<float, false>(s, s.constants32());
}

TEST_CASE("transient diffusion (D-P2-2): the cell's n1-marginal weight J and ratio rho against quadrature") {
  // J = int e^{-2 int_s^te k u} k (I + u) ds over a cell made of input pieces (diffusion_weight's recursion), against a
  // fine trapezoid of the same integral on the exact stage-1 path; rho = J / (1 - e^{-2 int k u}) tends to the cell
  // average (a_I + a1) / (2 a1) as a1 h -> 0 and to 1 for a cell whose transient settled long before its end.
  const double k = 1e16 * 1e-15;  // 1 / (fA s)
  const auto z_of = [&](double z0, double Ia, double Ib, double h) { return z_step(z0, Ia, Ib, h, k); };
  struct Piece {
    double Ia, Ib, h;
  };
  const auto exact = [&](double z0, const std::vector<Piece>& ps) {
    // Fine trapezoid in s on each piece (u from the closed form at each node), weights from the cumulative int k u.
    std::vector<double> s_t, f, u;
    double z = z0, t0 = 0;
    for (const Piece& p : ps) {
      const int n = 200000;
      for (int i = (s_t.empty() ? 0 : 1); i <= n; ++i) {
        const double s = p.h * i / n;
        const double I = p.Ia + (p.Ib - p.Ia) * s / p.h;
        const double zs = i == 0 ? z : z_of(z, p.Ia, I, s);
        s_t.push_back(t0 + s);
        f.push_back(I + 1.0 / zs);  // I + u (fA)
        u.push_back(1.0 / zs);
      }
      z = z_of(z, p.Ia, p.Ib, p.h);
      t0 += p.h;
    }
    // Cumulative int k u from the start (trapezoid), then J.
    std::vector<double> cum(s_t.size(), 0.0);
    for (size_t i = 1; i < s_t.size(); ++i) cum[i] = cum[i - 1] + 0.5 * k * (u[i] + u[i - 1]) * (s_t[i] - s_t[i - 1]);
    double J = 0;
    for (size_t i = 1; i < s_t.size(); ++i) {
      const double wa = std::exp(-2 * (cum.back() - cum[i - 1])), wb = std::exp(-2 * (cum.back() - cum[i]));
      J += 0.5 * k * (wa * f[i - 1] + wb * f[i]) * (s_t[i] - s_t[i - 1]);
    }
    return std::pair<double, double>{J, cum.back()};
  };
  const auto kernel = [&](auto zero, double z0, const std::vector<Piece>& ps) {
    using T = decltype(zero);
    T jw = T(0), z = static_cast<T>(z0), lam = T(0);
    for (const Piece& p : ps) {
      const T za = z;
      z = z_step(z, static_cast<T>(p.Ia), static_cast<T>(p.Ib), static_cast<T>(p.h), static_cast<T>(k));
      const T ikI = integrated_pole(static_cast<T>(p.Ia), static_cast<T>(p.Ib), static_cast<T>(p.h), static_cast<T>(k));
      lam = lam + (balla::pmath::log(z / za) + ikI);
      jw = detail::diffusion_weight(jw, za, z, static_cast<T>(p.Ia), static_cast<T>(p.Ib), ikI, static_cast<T>(p.h),
                                    static_cast<T>(k));
    }
    return std::pair<double, double>{static_cast<double>(jw), static_cast<double>(detail::diffusion_ratio(jw, lam))};
  };
  const double h = 65.536e-6;
  // Steps inside a cell at 10 fA ... 100 pA (a1 h 0.0066 ... 66), up and down by 10, the step at a third of the cell.
  for (const double I1 : {10.0, 1e3, 1e5}) {
    for (const double r : {10.0, 0.1}) {
      const double I0 = I1 / r;
      const std::vector<Piece> ps{{I0, I0, h / 3}, {I0, I1, 1e-9}, {I1, I1, 2 * h / 3 - 1e-9}};
      const auto [J, L] = exact(1.0 / I0, ps);
      const auto [j64, rho64] = kernel(0.0, 1.0 / I0, ps);
      const auto [j32, rho32] = kernel(0.0f, 1.0 / I0, ps);
      INFO("I1 " << I1 << " fA, x" << r << ": J " << J << " vs " << j64 << " / " << j32 << ", rho " << rho64);
      CHECK(std::fabs(j64 / J - 1) < 2e-4);  // the 1 ns ramp under the uniform rule; the quadrature's own error
      CHECK(std::fabs(j32 / J - 1) < 1e-3);
      CHECK(std::fabs(rho64 - J / -std::expm1(-2 * L)) < 2e-4 * rho64);
      if (I1 == 10.0) {  // a1 h << 1: the cell average (a_I + a1) / (2 a1)
        const double aI = k * (I0 * h / 3 + I1 * 2 * h / 3) / h, a1 = L / h;
        CHECK(rho64 == doctest::Approx((aI + a1) / (2 * a1)).epsilon(0.01));
      }
    }
  }
  // Whole-cell ramps (the renderer's input is linear between samples): x10, /10, x100 and /100 from a settled pixel
  // at k I0 h = 0.1 ... 30, one piece over the cell and over 98 us and 1 ms (the cell's first 65.5 us, the rest outside
  // it). In sub-pieces (diffusion_ramp) they stay within 0.3 % (one uniform piece: up to 38 %, the round-16 review).
  double worst_ramp[2] = {0, 0};
  for (const double kI0h : {0.1, 0.3, 1.0, 3.0, 10.0, 30.0}) {
    const double I0 = kI0h / (k * h);
    for (const double r : {10.0, 0.1, 100.0, 0.01}) {
      for (const double len : {h, 98e-6, 1e-3}) {
        const double I1 = I0 * r;
        const double Ie = len == h ? I1 : I0 + (I1 - I0) * h / len;  // the input at the cell's end
        const std::vector<Piece> ps{{I0, Ie, h}};
        const auto [J, L] = exact(1.0 / I0, ps);
        const double rho = J / -std::expm1(-2 * L);
        const auto [j64, rho64] = kernel(0.0, 1.0 / I0, ps);
        const auto [j32, rho32] = kernel(0.0f, 1.0 / I0, ps);
        INFO("ramp x" << r << " over " << len * 1e6 << " us at k I0 h " << kI0h << ": rho " << rho << " vs " << rho64
                      << " / " << rho32);
        worst_ramp[0] = std::max(worst_ramp[0], std::fabs(rho64 / rho - 1));
        worst_ramp[1] = std::max(worst_ramp[1], std::fabs(rho32 / rho - 1));
        (void)j64;
        (void)j32;
      }
    }
  }
  MESSAGE("whole-cell ramps: worst |rho / rho_exact - 1| " << worst_ramp[0] << " (fp64), " << worst_ramp[1]
                                                          << " (fp32)");
  CHECK(worst_ramp[0] < 0.005);
  CHECK(worst_ramp[1] < 0.005);
  // A settled cell long after a step (a1 h = 66): rho = 1.
  {
    const auto [j, rho] = kernel(0.0, 1.0 / 1e5, std::vector<Piece>{{1e5, 1e5, h}});
    CHECK(rho == doctest::Approx(1.0).epsilon(1e-6));
    (void)j;
  }
}
