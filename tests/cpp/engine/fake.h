// A WindowEngine with an analytic stand-in for the renderer, run in either memory space: two groups
// of triangle waves (exactly rounded arithmetic only, so the host and the device agree bit for bit)
// and one signature step per pixel, whose class word (render/types.h) is textured with one shadowed pair after the step
// and 0 before it. The engine tests compare runs of both spaces, batched and not.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "engine/par.h"
#include "engine/window.h"

namespace balla::engine::test {

inline constexpr int64_t kW = int64_t{1} << 23;
inline constexpr int64_t kTw = 37 * kW;

struct FakeRender {
  int64_t t_w;
  double rate;  // cycles per second

  // n: the sub-samples (P2-B B2's N classes): an offset (n - 64) 2^-12 in the first group, none at 64.
  BALLA_HD void eval(uint32_t pixel, int64_t t, float* irr, uint64_t* sig, uint32_t n = 64) const {
    double x = 0.37 * pixel + rate * 1e-9 * static_cast<double>(t - t_w);
    x -= static_cast<double>(static_cast<int64_t>(x));
    const double tri = 4.0 * (x < 0.5 ? x : 1.0 - x) - 1.0;  // in [-1, 1]
    irr[0] = static_cast<float>(1.0 + 0.5 * tri + (static_cast<double>(n) - 64.0) * 0x1p-12);
    irr[1] = static_cast<float>(0.25 + 0.1 * tri * tri);
    const int64_t step = t_w + static_cast<int64_t>((pixel * 2654435761u) % static_cast<uint32_t>(kW));
    *sig = t >= step ? 0x9E3779B97F4A7C15ull : 0ull;
  }
};

// Request i at the call's N; a request of another N (a call that is not uniform) renders NaN.
struct FakeKernel {
  FakeRender f;
  const SampleRequest* requests;
  const uint32_t* out_index;
  float* irr;
  uint64_t* sig;
  uint32_t* klass;
  uint32_t subsamples;
  BALLA_HD void operator()(size_t i) const {
    const uint32_t row = out_index[i];
    f.eval(requests[i].pixel, f.t_w + requests[i].t_off_ns, irr + 2 * static_cast<size_t>(row), sig + row, subsamples);
    if (requests[i].k_count != subsamples) irr[2 * static_cast<size_t>(row)] = NAN;
    if (klass != nullptr) klass[row] = sig[row] != 0 ? render::kClassTextured | 1u : 0u;
  }
};

template <Space S>
class FakeHook final : public RenderHook {
 public:
  explicit FakeHook(FakeRender f) : f_(f) {}
  void render(SampleRequest* requests, size_t n, const uint32_t* out_index, float* irr, uint64_t* sig,
              uint32_t* klass, uint32_t subsamples) override {
    Par<S>::for_each(n, FakeKernel{f_, requests, out_index, irr, sig, klass, subsamples}, nullptr);
    ++calls;
    rendered += n;
    if (subsamples != 64) band += n;
    if (calls_seen != nullptr) *calls_seen = calls;
    if (cancel_set != nullptr && calls == cancel_at) cancel_set->set();
  }
  size_t calls = 0, rendered = 0, band = 0;  // band: requests of calls at another N than 64
  // BL-0b: sets *cancel_set at the cancel_at-th call (1-based) and counts the calls in *calls_seen, also when the
  // window then throws (null: neither).
  CancelToken* cancel_set = nullptr;
  size_t cancel_at = 0;
  size_t* calls_seen = nullptr;

 private:
  FakeRender f_;
};

struct FakeSetup {
  std::vector<uint32_t> pixel;
  std::vector<uint8_t> level;
  std::vector<float> carried_irr;  // n x 2
  std::vector<uint64_t> carried_sig;
  std::vector<uint8_t> n_log2;  // P2-B B2: each pixel's log2 N (empty: cfg.subsamples for all)
  SampleConfig cfg;
  FakeRender render{kTw, 0.0};
  CancelToken* cancel_set = nullptr;  // BL-0b: FakeHook's
  size_t cancel_at = 0;
  size_t* calls_seen = nullptr;
};

struct FakeRuns {
  std::vector<int64_t> offsets, t;
  std::vector<float> irr, last_irr;
  std::vector<uint64_t> last_sig;
  WindowStats stats;
  size_t rendered = 0, band = 0;  // band: rendered at another N than 64 (FakeHook::band)
  size_t calls = 0;  // FakeHook::calls: every render call, a redone batch's included
  size_t peak = 0;  // the engine's peak bytes during the window
};

inline FakeSetup fake_setup(uint32_t seed, size_t n, bool fixed) {
  std::mt19937 rng(seed);
  FakeSetup s;
  for (size_t a = 0; a < n; ++a) {
    s.pixel.push_back(static_cast<uint32_t>(3 * a + 1));
    s.level.push_back(static_cast<uint8_t>(1 + rng() % 7));
    s.carried_irr.push_back(1.0f + 0.001f * static_cast<float>(rng() % 100));
    s.carried_irr.push_back(0.25f);
    s.carried_sig.push_back(rng() % 2);
  }
  s.cfg.groups = 2;
  SampleRules& r = s.cfg.rules;
  r.min_split = 2048;
  r.shift = 10;
  r.max_refine_levels = 4;
  r.midpoint = true;
  r.mid = midpoint_params(0.002, 1e-3, {{1.0, 1.0}, {0.8, 1.2}});
  if (fixed) {
    r.fixed = true;
    r.period_ns = 1000;
    r.m = fixed_rate_samples(kTw, 1000);
  }
  s.render.rate = 300.0 + 40.0 * (seed % 5);
  return s;
}

// The samples study's economy schedule ES (ENGINE_CPP I7; every switch on) on a setup: 3 midpoint levels, the tau
// stop at 2^14 ns scaled by the step (delta_ref 0.04), the half rule and the residual gate; 13 levels as the safety cap.
inline FakeSetup economy(FakeSetup s) {
  s.cfg.rules.max_refine_levels = 13;
  economy_rules(s.cfg.rules, 3, int64_t{1} << 14, 0.04, 0.25, 0.002);
  return s;
}

// One window in space S on `engine` (budget in bytes).
template <Space S>
FakeRuns fake_window(WindowEngine<S>& engine, const FakeSetup& s, size_t budget) {
  using P = Par<S>;
  const size_t n = s.pixel.size(), g = 2;
  Buffer<uint32_t, S> pixel;
  Buffer<uint8_t, S> level;
  Buffer<float, S> irr;
  Buffer<uint64_t, S> sig;
  Buffer<uint32_t, S> klass;  // the carried samples' class words: 0
  Buffer<uint8_t, S> n_log2;
  if (!s.n_log2.empty()) P::upload(n_log2.ensure(n), s.n_log2.data(), n, nullptr);
  P::upload(pixel.ensure(n), s.pixel.data(), n * 4, nullptr);
  P::zero(klass.ensure(n), n * 4, nullptr);
  P::upload(level.ensure(n), s.level.data(), n, nullptr);
  P::upload(irr.ensure(n * g), s.carried_irr.data(), n * g * 4, nullptr);
  P::upload(sig.ensure(n), s.carried_sig.data(), n * 8, nullptr);
  FakeHook<S> hook(s.render);
  hook.cancel_set = s.cancel_set;
  hook.cancel_at = s.cancel_at;
  hook.calls_seen = s.calls_seen;
  const typename WindowEngine<S>::Input in{kTw,         pixel.data(), level.data(),
                                           n,           irr.data(),   sig.data(),
                                           nullptr,     klass.data(), s.n_log2.empty() ? nullptr : n_log2.data()};
  FakeRuns out;
  const size_t others = BufferMeter<S>::live.load() - engine.held_bytes();
  BufferMeter<S>::reset_peak();
  out.stats = engine.run(in, s.cfg, hook, budget, nullptr);
  out.peak = BufferMeter<S>::peak.load() - others;
  out.rendered = hook.rendered;
  out.band = hook.band;
  out.calls = hook.calls;
  const size_t samples = engine.samples();
  out.offsets.resize(n + 1);
  out.t.resize(samples);
  out.irr.resize(samples * g);
  out.last_irr.resize(n * g);
  out.last_sig.resize(n);
  P::download(out.offsets.data(), engine.offsets(), (n + 1) * 8, nullptr);
  P::download(out.t.data(), engine.t(), samples * 8, nullptr);
  P::download(out.irr.data(), engine.irr(), samples * g * 4, nullptr);
  P::download(out.last_irr.data(), irr.data(), n * g * 4, nullptr);
  P::download(out.last_sig.data(), sig.data(), n * 8, nullptr);
  return out;
}

// BL-0b: the window of s with a cancel token set before it (at 0) or during its render call at = 1..calls, by
// run(setup) (a fake_window in a memory space): per at, the calls made and the check point's message ("" when the
// window completed).
template <class Run>
std::vector<std::pair<size_t, std::string>> cancel_outcomes(const FakeSetup& s, size_t calls, Run run) {
  std::vector<std::pair<size_t, std::string>> out;
  for (size_t at = 0; at <= calls; ++at) {
    CancelToken token;
    size_t seen = 0;
    if (at == 0) token.set();
    FakeSetup c = s;
    c.cfg.cancel = &token;
    c.cancel_set = &token;
    c.cancel_at = at;
    c.calls_seen = &seen;
    std::string what;
    try {
      run(c);
    } catch (const Cancelled& e) {
      what = e.what();
    }
    out.emplace_back(seen, what);
  }
  return out;
}

// One window on a new engine.
template <Space S>
FakeRuns fake_window(const FakeSetup& s, size_t budget) {
  WindowEngine<S> engine;
  return fake_window(engine, s, budget);
}

}  // namespace balla::engine::test
