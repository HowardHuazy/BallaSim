// The re-simulation's sensor side (docs/design/phase3/ui.md §8.1 W8, S1-S5): the output order and cut against the
// engine's own output stage (engine/output.h's functors with the host's stable passes), the ROI model against the full
// model, rederive against fresh runs, RawEvent::l_first, the stationary probe split, and ResimRun's held buffer against
// the engine's window outputs and flush. Small synthetic frames: the fast tier.
#include <doctest/doctest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/output.h"
#include "sensor/maps.h"
#include "sensor/output_order.h"
#include "sensor/rederive.h"
#include "sensor/resim.h"
#include "sensor/sensor_model.h"

using namespace balla::sensor;
namespace engine = balla::engine;

namespace {

constexpr int64_t kW = int64_t{1} << 23;

// The engine's output stage on the host as Camera::output and flush run it (the crop, BuildEvents, the held events
// first, three OutKeys passes with Host::sort_pairs' stable sort, GatherEvents at the cut), with the pass widths and
// the cut written as engine_impl.h wrote them before output_order.h: the reference of W8 and S5.
struct EngineOutput {
  uint32_t width = 0;
  int y_bits = 1;
  Roi crop{};
  Readout readout{};
  std::vector<engine::OutEvent> held;

  static void passes(const std::vector<engine::OutEvent>& ev, int y_bits, std::vector<uint32_t>& order) {
    const size_t n = ev.size();
    order.resize(n);
    std::iota(order.begin(), order.end(), 0u);
    std::vector<uint64_t> keys(n);
    for (int pass = 0; pass < 3; ++pass) {
      for (size_t i = 0; i < n; ++i) engine::OutKeys{ev.data(), order.data(), pass, y_bits, keys.data()}(i);
      const int bits = pass == 1 ? 16 + y_bits : 64;
      const uint64_t mask = bits >= 64 ? ~uint64_t{0} : (uint64_t{1} << bits) - 1;
      std::vector<uint32_t> idx(n);
      std::iota(idx.begin(), idx.end(), 0u);
      std::stable_sort(idx.begin(), idx.end(),
                       [&](uint32_t a, uint32_t b) { return (keys[a] & mask) < (keys[b] & mask); });
      std::vector<uint32_t> next(n);
      for (size_t i = 0; i < n; ++i) next[i] = order[idx[i]];
      order.swap(next);
    }
  }

  static int64_t cut(Readout r, int64_t t_end) {
    r.delay_ns = 0;
    r.quantum_ns = 0;
    return reported_time(r, t_end);
  }

  std::vector<engine::OutEvent> window(const std::vector<RawEvent>& raw, int64_t t_end) {
    std::vector<uint8_t> flags(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
      engine::CropFlags{raw.data(), width, crop.x0, crop.y0, crop.x1, crop.y1, flags.data()}(i);
    }
    std::vector<uint32_t> sel;
    for (size_t i = 0; i < raw.size(); ++i) {
      if (flags[i] != 0) sel.push_back(static_cast<uint32_t>(i));
    }
    const size_t n_held = held.size(), total = n_held + sel.size();
    held.resize(total);
    for (size_t i = 0; i < sel.size(); ++i) {
      engine::BuildEvents{raw.data(), sel.data(), width, readout, held.data() + n_held, nullptr}(i);
    }
    std::vector<uint32_t> order;
    passes(held, y_bits, order);
    std::vector<engine::OutEvent> sorted(total);
    unsigned long long ready = 0;
    const int64_t limit = cut(readout, t_end);
    for (size_t i = 0; i < total; ++i) engine::GatherEvents{held.data(), order.data(), sorted.data(), &ready, limit}(i);
    std::vector<engine::OutEvent> out(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(ready));
    held.assign(sorted.begin() + static_cast<std::ptrdiff_t>(ready), sorted.end());
    return out;
  }

  std::vector<engine::OutEvent> flush() {
    std::vector<engine::OutEvent> out;
    out.swap(held);
    return out;
  }
};

// Rows of `got` that differ from `want` in the event file's columns (x, y, p, t, label, t_cross, flags).
size_t mismatches(const RawEvents& got, const std::vector<engine::OutEvent>& want, uint32_t width) {
  if (got.size() != want.size()) return std::max(got.size(), want.size());
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const RawEvent& e = got.ev[i];
    const engine::OutEvent& o = want[i];
    bad += !(e.pixel % width == o.x && e.pixel / width == o.y && e.polarity == o.p && e.t_cross == o.t_cross &&
             got.t_reported[i] == o.t_reported && got.label_s1[i] == o.label && e.flags == o.flags);
  }
  return bad;
}

// A synthetic camera: a graded frame (many strata of ln I) of two light groups, the second modulated at 120 Hz, with a
// bright bar moving at 1 px/ms; the engine's call sequence of a motion warm-up: the hold t_init -> t_m (off the window
// grid at t_init), two warm-up windows [t_m, t_s) and three run windows [t_s, t_e).
struct Scene {
  uint32_t W = 48, H = 40;
  Roi roi{8, 6, 40, 38};  // 32 x 32
  int64_t t_m = 102 * kW, t_init = 100 * kW - 1'234'567, t_s = 104 * kW, t_e = 107 * kW;
  uint32_t camera = 3;
  uint64_t seed = 77;
  std::vector<Waveform> groups{make_constant_waveform(), make_fourier_waveform(120, {0.3}, {0.4})};

  float value(uint32_t px, int g, int64_t t) const {
    const double x = px % W, y = px / W;
    const double base = 18.0 + 0.7 * x + 0.45 * y;
    if (g == 1) return static_cast<float>(0.25 * base);
    const double xc = 2.0 + static_cast<double>(t - t_m) * 1e-6;
    return static_cast<float>(base * (std::fabs(x - xc) < 2.5 ? 3.0 : 1.0));
  }
  std::vector<float> frame(int64_t t) const {
    std::vector<float> f(static_cast<size_t>(W) * H * 2);
    for (uint32_t p = 0; p < W * H; ++p) {
      for (int g = 0; g < 2; ++g) f[p * 2 + g] = value(p, g, t);
    }
    return f;
  }
  // Call k's runs over the whole frame: pixels listed unless p % 7 == k % 7, samples every W / (4 + p % 5) from t_w
  // (the carried sample) to the window end.
  CacheCall call(size_t k) const {
    CacheCall c;
    if (k == 0) {
      c.kind = CallKind::kHold;
      c.t_w_ns = t_init;
      c.t_end_ns = t_m;
      return c;
    }
    c.t_w_ns = t_m + static_cast<int64_t>(k - 1) * kW;
    c.t_end_ns = c.t_w_ns + kW;
    c.kind = c.t_w_ns < t_s ? CallKind::kWarmup : CallKind::kWindow;
    for (uint32_t p = 0; p < W * H; ++p) {
      if (p % 7 == k % 7) continue;
      const int64_t m = 4 + p % 5, step = kW / m;
      c.ids.push_back(p);
      c.count.push_back(static_cast<int32_t>(m + 1));
      for (int64_t j = 0; j <= m; ++j) {
        const int64_t t = c.t_w_ns + j * step;
        c.t.push_back(t);
        for (int g = 0; g < 2; ++g) c.irr.push_back(value(p, g, t));
      }
    }
    return c;
  }
  size_t n_calls() const { return 1 + static_cast<size_t>((t_e - t_m) / kW); }
};

CacheCall restricted(const CacheCall& c, const Roi& r, uint32_t W) {
  CacheCall out;
  out.kind = c.kind;
  out.t_w_ns = c.t_w_ns;
  out.t_end_ns = c.t_end_ns;
  size_t at = 0;
  for (size_t i = 0; i < c.ids.size(); ++i) {
    const uint32_t x = c.ids[i] % W, y = c.ids[i] / W;
    const size_t n = static_cast<size_t>(c.count[i]);
    if (static_cast<int32_t>(x) >= r.x0 && static_cast<int32_t>(x) < r.x1 && static_cast<int32_t>(y) >= r.y0 &&
        static_cast<int32_t>(y) < r.y1) {
      out.ids.push_back(c.ids[i]);
      out.count.push_back(c.count[i]);
      out.t.insert(out.t.end(), c.t.begin() + static_cast<std::ptrdiff_t>(at),
                   c.t.begin() + static_cast<std::ptrdiff_t>(at + n));
      out.irr.insert(out.irr.end(), c.irr.begin() + static_cast<std::ptrdiff_t>(2 * at),
                     c.irr.begin() + static_cast<std::ptrdiff_t>(2 * (at + n)));
    }
    at += n;
  }
  return out;
}

// The scene's cache of the ROI without a writer run: the calls are the scene's own, and without veiling glare no frame
// means are needed (S2, whose references are fresh replays).
CacheView scene_cache(const Scene& sc) {
  CacheView c;
  c.width = sc.W;
  c.height = sc.H;
  c.camera = sc.camera;
  c.roi = sc.roi;
  c.t_init_ns = sc.t_init;
  c.init_irr = sc.frame(sc.t_m);
  for (size_t k = 0; k < sc.n_calls(); ++k) c.calls.push_back(restricted(sc.call(k), sc.roi, sc.W));
  return c;
}

// The writer run: the full model with every pixel listed, as the engine runs a camera (the frame at t_m from t_init),
// its frame means tracked at every call (the signal tap's), the ROI's calls and states, and each run window's raw
// events.
struct Writer {
  CacheView cache;
  std::vector<std::vector<RawEvent>> raw;          // per run window, the full frame's
  std::vector<std::vector<PixelState>> states;     // the ROI pixels' after init and after each call
  std::vector<std::vector<HpfState>> hpf;
  Readout readout{};
  size_t probes = 0;
};

Writer write(const Scene& sc, const SensorParams& p, Precision prec, double scale = 1.0,
             const std::vector<float>& gain = {}) {
  Writer w;
  SensorParams p0 = p;
  p0.illuminance_scale = 1.0;  // the resolver's placeholder; the engine sets the solved scale and the gains
  SensorModel full(p0, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 4, prec);
  full.set_photometry(scale, gain);
  full.set_track_frame_means(true);
  w.readout = full.readout();
  CacheView& c = w.cache;
  c.width = sc.W;
  c.height = sc.H;
  c.camera = sc.camera;
  c.roi = sc.roi;
  c.t_init_ns = sc.t_init;
  c.init_irr = sc.frame(sc.t_m);
  c.gain = gain;
  const auto roi_states = [&] {
    const std::vector<PixelState> all = full.states();
    const std::vector<HpfState> h = full.hpf_states();
    std::vector<PixelState> s;
    std::vector<HpfState> hs;
    for (int32_t y = sc.roi.y0; y < sc.roi.y1; ++y) {
      for (int32_t x = sc.roi.x0; x < sc.roi.x1; ++x) {
        s.push_back(all[static_cast<size_t>(y) * sc.W + x]);
        if (!h.empty()) hs.push_back(h[static_cast<size_t>(y) * sc.W + x]);
      }
    }
    w.states.push_back(s);
    w.hpf.push_back(hs);
  };
  full.init(sc.t_init, c.init_irr.data());
  w.probes = full.stationary_probes();
  c.frame_mean = full.glare_frame_mean();
  roi_states();
  for (size_t k = 0; k < sc.n_calls(); ++k) {
    const CacheCall call = sc.call(k);
    std::vector<int64_t> start;
    int64_t at = 0;
    for (const int32_t n : call.count) {
      start.push_back(at);
      at += n;
    }
    std::vector<RawEvent> raw = full.advance_runs_raw(call.ids.data(), call.ids.size(), call.t.data(), call.irr.data(),
                                                      call.t.size(), start.data(), call.count.data(), call.t_end_ns);
    const std::vector<double>& m = full.glare_frame_mean();
    c.frame_mean.insert(c.frame_mean.end(), m.begin(), m.end());
    roi_states();
    if (call.kind == CallKind::kWindow) w.raw.push_back(std::move(raw));
    c.calls.push_back(restricted(call, sc.roi, sc.W));
  }
  return w;
}

// The engine's per-window outputs and flush of a writer, cropped to r, over its first `windows` run windows.
std::vector<std::vector<engine::OutEvent>> engine_outputs(const Scene& sc, const Writer& w, const Roi& r,
                                                          size_t windows) {
  EngineOutput e;
  e.width = sc.W;
  e.y_bits = order_y_bits(sc.H);
  e.crop = r;
  e.readout = w.readout;
  std::vector<std::vector<engine::OutEvent>> out;
  for (size_t i = 0; i < windows; ++i) out.push_back(e.window(w.raw[i], sc.t_s + static_cast<int64_t>(i + 1) * kW));
  out.push_back(e.flush());
  return out;
}

SensorSpec spec_of(const Scene& sc, SensorParams p, Precision prec, double scale = 1.0) {
  p.illuminance_scale = scale;
  return {p, sc.seed, sc.groups, prec};
}

void append(RawEvents& all, const RawEvents& out) {
  all.ev.insert(all.ev.end(), out.ev.begin(), out.ev.end());
  all.t_reported.insert(all.t_reported.end(), out.t_reported.begin(), out.t_reported.end());
  all.label_s1.insert(all.label_s1.end(), out.label_s1.begin(), out.label_s1.end());
}

// A run's whole session stream: every step's events, then finish's.
RawEvents collect(ResimRun& run) {
  RawEvents all, out;
  while (run.step(&out)) append(all, out);
  run.finish(&out);
  append(all, out);
  return all;
}

// A ResimRun's released batches per run window, then finish's, checked against `want` (engine_outputs) and its model's
// states against the writer's after every call (when the ROI is the cache's). Returns the whole session stream.
RawEvents replay(const Scene& sc, const Writer& w, const SensorSpec& spec, const Roi& roi, int64_t t1, StartMode mode,
                 const ProbeResult* probe, const std::vector<std::vector<engine::OutEvent>>& want,
                 Device device = Device::kCpu) {
  ResimRun run(w.cache, spec, roi, t1, mode, probe, device, 4);
  const bool same_roi = std::memcmp(&roi, &sc.roi, sizeof(Roi)) == 0;
  const auto states_equal = [&](size_t k) {
    const std::vector<PixelState> s = run.model().states();
    const std::vector<HpfState> h = run.model().hpf_states();
    size_t first_bad = s.size();
    for (size_t i = 0; i < s.size() && first_bad == s.size(); ++i) {
      if (std::memcmp(&s[i], &w.states[k][i], sizeof(PixelState)) != 0) first_bad = i;
    }
    if (first_bad < s.size()) {
      const auto* a = reinterpret_cast<const unsigned char*>(&s[first_bad]);
      const auto* b = reinterpret_cast<const unsigned char*>(&w.states[k][first_bad]);
      size_t off = 0;
      while (a[off] == b[off]) ++off;
      MESSAGE("call " << k << ": pixel " << first_bad << " differs at byte " << off << " z " << s[first_bad].z
                      << " vs " << w.states[k][first_bad].z << " i_scale " << s[first_bad].i_scale << " vs "
                      << w.states[k][first_bad].i_scale);
    }
    CHECK(std::memcmp(s.data(), w.states[k].data(), s.size() * sizeof(PixelState)) == 0);
    CHECK(h.size() == w.hpf[k].size());
    if (h.size() == w.hpf[k].size()) CHECK(std::memcmp(h.data(), w.hpf[k].data(), h.size() * sizeof(HpfState)) == 0);
  };
  if (same_roi) states_equal(0);
  RawEvents all, out;
  size_t window = 0;
  while (run.step(&out)) {
    const CacheCall& call = w.cache.calls[run.calls_done() - 1];
    if (same_roi) states_equal(run.calls_done());
    if (call.kind != CallKind::kWindow) {
      CHECK(out.size() == 0);
      continue;
    }
    REQUIRE(window + 1 < want.size());
    INFO("run window " << window);
    CHECK(mismatches(out, want[window], sc.W) == 0);
    ++window;
    append(all, out);
  }
  CHECK(window + 1 == want.size());
  run.finish(&out);
  CHECK(mismatches(out, want.back(), sc.W) == 0);
  append(all, out);
  return all;
}

size_t events_in(const std::vector<std::vector<engine::OutEvent>>& v) {
  size_t n = 0;
  for (const auto& b : v) n += b.size();
  return n;
}

SensorParams base_params() {
  SensorParams p = preset("prophesee_gen3");
  p.stationary_burnin_ns = 250'000'000;
  p.stationary_probe_pixels = 8;
  return p;
}

ProbeResult probe_for(const Scene& sc, const Writer& w, const SensorSpec& spec) {
  SensorParams p = spec.params;
  p.initial_state = InitialState::kStationary;
  SensorModel m(p, sc.W, sc.H, spec.seed, spec.groups, sc.camera, 4, spec.precision, Device::kCpu, sc.roi);
  if (!w.cache.gain.empty()) m.set_photometry(p.illuminance_scale, w.cache.gain);
  if (!w.cache.frame_mean.empty()) m.set_frame_means(w.cache.frame_mean);
  return m.stationary_probe(w.cache.t_init_ns, w.cache.init_irr.data());
}

}  // namespace

TEST_CASE("W8 output_order and output_cut equal the engine's three passes and cut") {
  std::mt19937_64 rng(5);
  for (const uint32_t height : {1u, 2u, 40u, 480u, 1080u}) {
    for (const size_t n : {size_t{0}, size_t{1}, size_t{7}, size_t{600}, size_t{5000}}) {
      INFO("height " << height << ", " << n << " events");
      const int y_bits = order_y_bits(height);
      // Few pixels and stamps, so that (t_reported, x, y) ties are common; one crossing per pixel and ns.
      std::vector<engine::OutEvent> ev;
      std::vector<std::pair<uint32_t, int64_t>> used;
      while (ev.size() < n) {
        engine::OutEvent o{};
        o.x = static_cast<uint16_t>(rng() % 5);
        o.y = static_cast<uint16_t>(rng() % std::min<uint32_t>(height, 4));
        o.t_cross = -3'000 + static_cast<int64_t>(rng() % 40'000);
        const uint32_t px = o.y * 5u + o.x;
        if (std::find(used.begin(), used.end(), std::make_pair(px, o.t_cross)) != used.end()) continue;
        used.emplace_back(px, o.t_cross);
        o.t_reported = (o.t_cross + 2'500) / 4'000 * 4'000 - (rng() % 3 == 0 ? int64_t{1} << 40 : 0);
        ev.push_back(o);
      }
      std::vector<uint32_t> want;
      EngineOutput::passes(ev, y_bits, want);
      std::vector<int64_t> tc(n), tr(n);
      std::vector<uint16_t> x(n), y(n);
      for (size_t i = 0; i < n; ++i) tc[i] = ev[i].t_cross, tr[i] = ev[i].t_reported, x[i] = ev[i].x, y[i] = ev[i].y;
      std::vector<uint32_t> got;
      output_order(tc, x, y, tr, y_bits, got);
      CHECK(got == want);
    }
  }
  for (int i = 0; i < 20000; ++i) {
    const int64_t q = std::array<int64_t, 4>{0, 1'000, 1'024, 4'000}[rng() % 4];
    Readout r{q, q + static_cast<int64_t>(rng() % 300'000), q > 0 ? static_cast<int64_t>(rng() % q) : 0,
              static_cast<int64_t>(rng() % 2'000'000'000) - 1'000'000'000,
              std::array<double, 4>{-200, 0, 37.5, 200}[rng() % 4]};
    const int64_t t = static_cast<int64_t>(rng() % (int64_t{1} << 42));
    CHECK(output_cut(r, t) == EngineOutput::cut(r, t));
  }
  CHECK(order_key_bits(1, 9) == 25);
  CHECK(order_key_bits(0, 9) == 64);
  for (uint8_t c3 = 0; c3 < 4; ++c3) CHECK(label_s1(c3) == std::array<uint8_t, 4>{0, 2, 3, 4}[c3]);
}

TEST_CASE("S1 S5 the ROI model and ResimRun reproduce the writer run in the ROI") {
  const Scene sc;
  struct Variant {
    const char* name;
    StartMode mode;
    double glare;
    double scale;
    bool gain, hot, hpf;
    Precision prec;
  };
  const Variant variants[] = {
      {"phase", StartMode::kPhase, 0, 1, false, false, false, Precision::kFp32},
      {"stationary, hot pixels", StartMode::kStationary, 0, 1, false, true, false, Precision::kFp32},
      {"phase, glare", StartMode::kPhase, 0.01, 1, false, true, false, Precision::kFp32},
      {"stationary, glare, reference scale and gains, tau_hp spread", StartMode::kStationary, 0.01, 37.5, true, true,
       true, Precision::kFp32},
      {"phase, fp64, tau_hp spread", StartMode::kPhase, 0, 1, false, true, true, Precision::kFp64},
  };
  std::vector<float> gain(static_cast<size_t>(sc.W) * sc.H);
  for (uint32_t p = 0; p < sc.W * sc.H; ++p) {
    const double dx = (p % sc.W - 23.5) / 30.0, dy = (p / sc.W - 19.5) / 30.0;
    gain[p] = static_cast<float>(1.0 - 0.4 * (dx * dx + dy * dy));
  }
  for (const Variant& v : variants) {
    INFO(std::string(v.name));
    SensorParams p = base_params();
    p.initial_state = v.mode == StartMode::kPhase ? InitialState::kPhase : InitialState::kStationary;
    p.veiling_glare = v.glare;
    p.hot_fraction = v.hot ? 0.02 : 0.0;
    if (v.hpf) {
      p.tau_hp = 0.05;
      p.tau_hp_sigma_ln = 0.3;
    }
    const Writer w = write(sc, p, v.prec, v.scale, v.gain ? gain : std::vector<float>{});
    if (v.mode == StartMode::kStationary) CHECK(w.probes == 8);
    const auto want = engine_outputs(sc, w, sc.roi, 3);
    MESSAGE(std::string(v.name) << ": " << events_in(want) << " events in the ROI");
    CHECK(events_in(want) > 200);
    const SensorSpec spec = spec_of(sc, p, v.prec, v.scale);
    replay(sc, w, spec, sc.roi, sc.t_e, v.mode, nullptr, want);
    if (v.glare > 0) {  // a sub-ROI: the writer's events restricted to it; a shorter t1: its windows and the held rest
      const Roi sub{12, 10, 28, 30};
      replay(sc, w, spec, sub, sc.t_e, v.mode, nullptr, engine_outputs(sc, w, sub, 3));
      replay(sc, w, spec, sc.roi, sc.t_s + 2 * kW, v.mode, nullptr, engine_outputs(sc, w, sc.roi, 2));
    }
    if (v.mode == StartMode::kStationary) {  // S4: a probe made apart gives the same run
      const ProbeResult probe = probe_for(sc, w, spec);
      replay(sc, w, spec, sc.roi, sc.t_e, v.mode, &probe, want);
    }
  }
  // A glare edit on a cache written at g = 0 (its frame means tracked): the writer run at g = 0.01, bit for bit.
  SensorParams p = base_params();
  const Writer w0 = write(sc, p, Precision::kFp32);
  p.veiling_glare = 0.01;
  const Writer w1 = write(sc, p, Precision::kFp32);
  CHECK(w0.cache.frame_mean == w1.cache.frame_mean);
  Writer edit = w0;  // the g = 0 cache, against the g = 0.01 run's states
  edit.states = w1.states;
  edit.hpf = w1.hpf;
  replay(sc, edit, spec_of(sc, p, Precision::kFp32), sc.roi, sc.t_e, StartMode::kPhase, nullptr,
         engine_outputs(sc, w1, sc.roi, 3));
}

TEST_CASE("S1 the ROI model's own contract: global ids, refusals") {
  const Scene sc;
  SensorParams p = base_params();
  p.hot_fraction = 0.05;
  SensorModel full(p, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, Precision::kFp32);
  SensorModel roi(p, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, Precision::kFp32, Device::kCpu, sc.roi);
  REQUIRE(roi.n_pixels() == 32 * 32);
  CHECK(roi.width() == sc.W);
  CHECK(roi.pixel_ids().front() == static_cast<uint32_t>(sc.roi.y0) * sc.W + sc.roi.x0);
  for (size_t i = 0; i < roi.n_pixels(); ++i) {
    const PixelMap& a = roi.maps().pixels[i];
    const PixelMap& b = full.maps().pixels[roi.pixel_ids()[i]];
    CHECK(std::memcmp(&a, &b, sizeof(PixelMap)) == 0);
  }
  CHECK_THROWS_AS(SensorModel(p, sc.W, sc.H, 1, sc.groups, 0, 1, Precision::kFp32, Device::kCpu, Roi{0, 0, 49, 4}),
                  std::invalid_argument);
  CHECK_THROWS_AS(SensorModel(p, sc.W, sc.H, 1, sc.groups, 0, 1, Precision::kFp32, Device::kCpu, Roi{4, 4, 4, 8}),
                  std::invalid_argument);
  CHECK_THROWS_AS(roi.set_photometry(2.0, std::vector<float>(roi.n_pixels(), 1.0f)), std::invalid_argument);
  CHECK_THROWS_AS(full.set_frame_means({1.0, 1.0}), std::logic_error);
  CHECK_THROWS_AS(roi.set_track_frame_means(true), std::logic_error);
  CHECK_THROWS_AS(roi.save_state(), std::logic_error);
  const std::vector<float> f = sc.frame(sc.t_m);
  roi.init(sc.t_init, f.data());
  const uint32_t outside = 0, inside = static_cast<uint32_t>(sc.roi.y0 + 1) * sc.W + sc.roi.x0 + 1;
  const int64_t t[2] = {sc.t_init, sc.t_init + 1000};
  const float irr[4] = {30, 5, 30, 5};
  const int64_t start = 0;
  const int32_t count = 2;
  CHECK_THROWS_AS(roi.advance_runs(&outside, 1, t, irr, 2, &start, &count, sc.t_m), std::invalid_argument);
  CHECK_NOTHROW(roi.advance_runs(&inside, 1, t, irr, 2, &start, &count, sc.t_m));
  CHECK_THROWS_AS(roi.advance_device(DeviceRuns{}, sc.t_m + kW, nullptr), std::logic_error);
  // Glare without the frame means: an ROI model cannot compute them past init.
  SensorParams g = p;
  g.veiling_glare = 0.01;
  SensorModel roi_g(g, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, Precision::kFp32, Device::kCpu, sc.roi);
  roi_g.init(sc.t_init, f.data());
  CHECK_THROWS_AS(roi_g.advance_runs(&inside, 1, t, irr, 2, &start, &count, sc.t_m), std::logic_error);
}

TEST_CASE("S1 an ROI model refuses the dark currents the full model refuses (the floor over the whole frame)") {
  const Scene sc;
  SensorParams d = base_params();
  d.dark_sigma_ln = 3.0;
  const uint32_t n = sc.W * sc.H;
  const SensorMaps all = draw_sensor_maps(d, sc.seed, sc.camera, n);
  double low = std::numeric_limits<double>::infinity();
  uint32_t at = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (all.pixels[i].dark_fA < low) low = all.pixels[i].dark_fA, at = i;
  }
  // lowest_dark_fA is draw_sensor_maps's minimum bit for bit, over any split of the frame.
  CHECK(lowest_dark_fA(d, sc.seed, sc.camera, 0, n) == low);
  CHECK(std::min(lowest_dark_fA(d, sc.seed, sc.camera, 0, at), lowest_dark_fA(d, sc.seed, sc.camera, at, n)) == low);
  CHECK(lowest_dark_fA(d, sc.seed, sc.camera, 5, 5) == std::numeric_limits<double>::infinity());
  SensorParams flat = d;
  flat.dark_sigma_ln = 0;
  const double flat_dark = draw_sensor_maps(flat, sc.seed, sc.camera, 1).pixels[0].dark_fA;
  CHECK(lowest_dark_fA(flat, sc.seed, sc.camera, 0, n) == flat_dark);
  // A half of the frame without the lowest pixel; the floor between its lowest and the frame's.
  const int32_t w = static_cast<int32_t>(sc.W), h = static_cast<int32_t>(sc.H);
  const Roi half = static_cast<int32_t>(at % sc.W) < w / 2 ? Roi{w / 2, 0, w, h} : Roi{0, 0, w / 2, h};
  double low_half = std::numeric_limits<double>::infinity();
  for (int32_t y = half.y0; y < half.y1; ++y) {
    for (int32_t x = half.x0; x < half.x1; ++x) low_half = std::min(low_half, all.pixels[y * sc.W + x].dark_fA);
  }
  REQUIRE(low < low_half);
  const double floor_fA = std::sqrt(double{FLT_MIN} / (d.k * 1e-15 * 1e-9));
  d.dark_current_fA *= floor_fA / std::sqrt(low * low_half);
  const auto build = [&](Precision prec, std::optional<Roi> roi) {
    SensorModel m(d, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, prec, Device::kCpu, roi);
    return m.n_pixels();
  };
  CHECK_THROWS_WITH_AS(build(Precision::kFp32, std::nullopt), doctest::Contains("fp32 kernel's floor"),
                       std::invalid_argument);
  CHECK_THROWS_WITH_AS(build(Precision::kFp32, half), doctest::Contains("fp32 kernel's floor"), std::invalid_argument);
  CHECK(build(Precision::kFp64, half) == static_cast<size_t>(sc.W / 2) * sc.H);
  d.dark_current_fA *= 2 * std::sqrt(low_half / low);  // the whole frame above the floor
  CHECK(build(Precision::kFp32, half) == static_cast<size_t>(sc.W / 2) * sc.H);
}

TEST_CASE("S2 rederive equals fresh runs: t_reported, S1 labels and row order") {
  const Scene sc;
  const Roi kS2Roi{12, 10, 36, 34};  // 24 x 24 of the cache's 32 x 32
  for (const bool stationary : {false, true}) {
    for (const bool hpf : {false, true}) {
      INFO(std::string(stationary ? "stationary" : "phase") << std::string(hpf ? ", tau_hp > 0" : ", tau_hp = 0"));
      SensorParams p = base_params();
      p.initial_state = stationary ? InitialState::kStationary : InitialState::kPhase;
      p.hot_fraction = 0.02;
      if (hpf) {
        p.tau_hp = 0.05;
        p.tau_hp_sigma_ln = 0.3;
      }
      const StartMode mode = stationary ? StartMode::kStationary : StartMode::kPhase;
      Writer w;
      w.cache = scene_cache(sc);
      const SensorSpec base = spec_of(sc, p, Precision::kFp32);
      // One probe serves every edit: readout and label cut are outside its key.
      const ProbeResult probe = stationary ? probe_for(sc, w, base) : ProbeResult{};
      const ProbeResult* pr = stationary ? &probe : nullptr;
      const auto session = [&](const SensorSpec& s) {
        ResimRun run(w.cache, s, kS2Roi, sc.t_e, mode, pr, Device::kCpu, 4);
        return collect(run);
      };
      const RawEvents a = session(base);
      size_t first = 0;
      for (const RawEvent& e : a.ev) first += (e.flags & kFlagFirstAfterInit) != 0;
      MESSAGE(a.size() << " events, " << first << " first after init");
      REQUIRE(first > 40);
      // The identity edit reproduces the stream itself.
      struct Edit {
        double quantum_s, delay_s, drift_ppm, cut;
        int64_t offset_ns;
      };
      const double q0 = p.readout_quantum_s, d0 = p.readout_delay_s;
      const Edit edits[] = {
          {q0, d0, 0, 0.5, 0},          {q0, d0 + 1e-6, 0, 0.5, 12'345}, {0, d0, 0, 0.5, 0},
          {4e-6, d0 + 4e-6, 200, 0.5, 0}, {q0, d0, 0, 0.6, 0},             {4e-6, d0 + 7e-6, -200, 0.35, -98'765},
      };
      for (const Edit& e : edits) {
        INFO("q " << e.quantum_s << ", d0 " << e.delay_s << ", drift " << e.drift_ppm << ", cut " << e.cut
                  << ", offset " << e.offset_ns);
        SensorSpec s = base;
        s.params.readout_quantum_s = e.quantum_s;
        s.params.readout_delay_s = e.delay_s;
        s.params.clock_drift_ppm = e.drift_ppm;
        s.params.label_share_cut = e.cut;
        s.params.clock_offset_ns = e.offset_ns;
        const RawEvents b = session(s);
        const Readout r = readout_of(s.params, sc.seed, sc.camera);
        const Rederived d = rederive(a.ev, r, e.cut, sc.seed, sc.camera, sc.W, order_y_bits(sc.H), true);
        REQUIRE(d.order.size() == b.size());
        size_t bad = 0;
        for (size_t i = 0; i < b.size(); ++i) {
          const uint32_t j = d.order[i];
          const RawEvent &x = a.ev[j], &y = b.ev[i];
          bad += !(x.pixel == y.pixel && x.t_cross == y.t_cross && x.polarity == y.polarity && x.flags == y.flags &&
                   x.ds == y.ds && x.dn == y.dn && x.dl == y.dl && x.l_first == y.l_first &&
                   d.t_reported[j] == b.t_reported[i] && d.label_s1[j] == b.label_s1[i]);
        }
        CHECK(bad == 0);
      }
    }
  }
  // fp64: readout edits re-derive with the labels copied.
  SensorParams p = base_params();
  p.hot_fraction = 0.02;
  const CacheView cache = scene_cache(sc);
  const auto stream = [&](const SensorParams& q) {
    ResimRun run(cache, spec_of(sc, q, Precision::kFp64), kS2Roi, sc.t_e, StartMode::kPhase, nullptr, Device::kCpu, 4);
    return collect(run);
  };
  const RawEvents a = stream(p);
  SensorParams q = p;
  q.readout_quantum_s = 4e-6;
  q.readout_delay_s = p.readout_delay_s + 5e-6;
  q.clock_drift_ppm = 200;
  q.clock_offset_ns = 4321;
  const RawEvents b = stream(q);
  const Rederived d = rederive(a.ev, readout_of(q, sc.seed, sc.camera), q.label_share_cut, sc.seed, sc.camera, sc.W,
                               order_y_bits(sc.H), false);
  REQUIRE(d.order.size() == b.size());
  size_t bad = 0;
  for (size_t i = 0; i < b.size(); ++i) {
    const uint32_t j = d.order[i];
    bad += !(a.ev[j].pixel == b.ev[i].pixel && a.ev[j].t_cross == b.ev[i].t_cross &&
             d.t_reported[j] == b.t_reported[i] && d.label_s1[j] == b.label_s1[i]);
  }
  CHECK(bad == 0);
}

TEST_CASE("S3 l_first: the leak since t0 of a first event after init, 0 otherwise") {
  CHECK(sizeof(RawEvent) == 32);
  const Scene sc;
  SensorParams p = base_params();
  const Writer w = write(sc, p, Precision::kFp32);
  ResimRun run(w.cache, spec_of(sc, p, Precision::kFp32), sc.roi, sc.t_e, StartMode::kPhase, nullptr);
  const RawEvents all = collect(run);
  size_t first = 0, bad = 0;
  const float cut = static_cast<float>(p.label_share_cut);
  for (const RawEvent& e : all.ev) {
    if ((e.flags & kFlagFirstAfterInit) == 0) {
      bad += e.l_first != 0.0f;
      continue;
    }
    ++first;
    // Phase start, no filter: l = leak (t - t0), in the kernel's fp32 (pixel.h fire()); the label is first_label's.
    const PixelMap& m = run.model().maps().pixels[static_cast<size_t>((e.pixel / sc.W - sc.roi.y0) * 32 +
                                                                      (e.pixel % sc.W - sc.roi.x0))];
    const float l = static_cast<float>(m.leak) * static_cast<float>(static_cast<double>(e.t_cross - sc.t_init) * 1e-9);
    uint8_t c3 = detail::first_label<float>(e.polarity != 0, e.ds, e.dn, e.l_first, cut);
    if ((e.flags & kFlagHotPixel) != 0 && c3 != kLabelSignal) c3 = kLabelHot;
    bad += !(e.l_first == l && c3 == e.label);
  }
  CHECK(first > 50);
  CHECK(bad == 0);
}

TEST_CASE("S4 the stationary probe split equals the monolithic init") {
  const Scene sc;
  for (const double glare : {0.0, 0.01}) {
    for (const bool hpf : {false, true}) {
      INFO("glare " << glare << std::string(hpf ? ", tau_hp spread" : ""));
      SensorParams p = base_params();
      p.initial_state = InitialState::kStationary;
      p.veiling_glare = glare;
      p.hot_fraction = 0.02;
      if (hpf) {
        p.tau_hp = 0.05;
        p.tau_hp_sigma_ln = 0.3;
      }
      const std::vector<float> f = sc.frame(sc.t_m);
      SensorModel mono(p, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 3, Precision::kFp32);
      mono.init(sc.t_init, f.data());
      REQUIRE(mono.stationary_probes() == 8);
      SensorModel split(p, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, Precision::kFp32);
      const ProbeResult probe = split.stationary_probe(sc.t_init, f.data());
      CHECK_FALSE(split.initialized());  // the probe leaves the model untouched
      split.stationary_init(sc.t_init, f.data(), probe);
      const std::vector<PixelState> a = mono.states(), b = split.states();
      CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(PixelState)) == 0);
      CHECK(mono.hpf_states().size() == split.hpf_states().size());
      CHECK(split.stationary_probes() == 8);
      CHECK(split.stationary_phase_kept() == mono.stationary_phase_kept());
      // An ROI model with the full model's probe, and with its own: the full model's states in the ROI.
      const std::vector<double>& m0 = mono.glare_frame_mean();
      std::vector<double> means = m0;
      for (int i = 0; i < 8; ++i) means.insert(means.end(), m0.begin(), m0.end());
      for (const bool given : {true, false}) {
        SensorModel roi(p, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, Precision::kFp32, Device::kCpu, sc.roi);
        if (glare > 0) roi.set_frame_means(means);
        if (given) roi.stationary_init(sc.t_init, f.data(), probe);
        else roi.init(sc.t_init, f.data());
        const std::vector<PixelState> r = roi.states();
        size_t bad = 0;
        for (size_t i = 0; i < r.size(); ++i) {
          bad += std::memcmp(&r[i], &a[roi.pixel_ids()[i]], sizeof(PixelState)) != 0;
        }
        CHECK(bad == 0);
        // The ROI model's own probe serves the full model (its strata are the frame's).
        if (!given) {
          const ProbeResult rp = roi.stationary_probe(sc.t_init, f.data());
          SensorModel back(p, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, Precision::kFp32);
          back.stationary_init(sc.t_init, f.data(), rp);
          const std::vector<PixelState> bs = back.states();
          CHECK(std::memcmp(a.data(), bs.data(), a.size() * sizeof(PixelState)) == 0);
        }
      }
      // Readout and label-cut edits keep the probe; other inputs refuse it.
      SensorParams e = p;
      e.readout_quantum_s = 4e-6;
      e.readout_delay_s = p.readout_delay_s + 4e-6;
      e.clock_offset_ns = 999;
      e.clock_drift_ppm = 100;
      e.label_share_cut = 0.6;
      SensorModel ok(e, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, Precision::kFp32);
      CHECK_NOTHROW(ok.stationary_init(sc.t_init, f.data(), probe));
      const std::vector<PixelState> c = ok.states();
      CHECK(std::memcmp(a.data(), c.data(), a.size() * sizeof(PixelState)) == 0);
      SensorParams th = p;
      th.theta_on *= 1.01;
      SensorModel other(th, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, Precision::kFp32);
      CHECK_THROWS_AS(other.stationary_init(sc.t_init, f.data(), probe), std::invalid_argument);
      SensorModel seed2(p, sc.W, sc.H, sc.seed + 1, sc.groups, sc.camera, 2, Precision::kFp32);
      CHECK_THROWS_AS(seed2.stationary_init(sc.t_init, f.data(), probe), std::invalid_argument);
      CHECK_THROWS_AS(split.stationary_init(sc.t_init + 1, f.data(), probe), std::invalid_argument);
      std::vector<float> g = f;
      g[17] += 1.0f;
      CHECK_THROWS_AS(split.stationary_init(sc.t_init, g.data(), probe), std::invalid_argument);
      SensorModel fp64(p, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, Precision::kFp64);
      CHECK_THROWS_AS(fp64.stationary_init(sc.t_init, f.data(), probe), std::invalid_argument);
      ProbeResult cut = probe;
      cut.blob.pop_back();
      CHECK_THROWS_AS(split.stationary_init(sc.t_init, f.data(), cut), std::invalid_argument);
      SensorParams ph = p;
      ph.initial_state = InitialState::kPhase;
      SensorModel phase(ph, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, Precision::kFp32);
      CHECK_THROWS_AS(phase.stationary_probe(sc.t_init, f.data()), std::logic_error);
    }
  }
  // An empty probe under the filter (noise off: the phase law; every pixel hot: none eligible) still serves its own
  // model, full or ROI, and gives init's states.
  for (const bool noise : {false, true}) {
    INFO(std::string(noise ? "every pixel hot" : "noise off") << ", tau_hp > 0");
    SensorParams p = base_params();
    p.initial_state = InitialState::kStationary;
    p.tau_hp = 0.05;
    p.noise = noise;
    p.hot_fraction = noise ? 1.0 : 0.02;
    const std::vector<float> f = sc.frame(sc.t_m);
    SensorModel mono(p, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, Precision::kFp32);
    mono.init(sc.t_init, f.data());
    CHECK(mono.stationary_probes() == 0);
    const std::vector<PixelState> a = mono.states();
    const std::vector<HpfState> ah = mono.hpf_states();
    REQUIRE(ah.size() == a.size());
    for (const bool roi : {false, true}) {
      SensorModel m(p, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, Precision::kFp32, Device::kCpu,
                    roi ? std::optional<Roi>(sc.roi) : std::nullopt);
      const ProbeResult probe = m.stationary_probe(sc.t_init, f.data());
      REQUIRE_NOTHROW(m.stationary_init(sc.t_init, f.data(), probe));
      const std::vector<PixelState> b = m.states();
      const std::vector<HpfState> bh = m.hpf_states();
      REQUIRE(bh.size() == b.size());
      size_t bad = 0;
      for (size_t i = 0; i < b.size(); ++i) {
        const size_t j = roi ? m.pixel_ids()[i] : i;
        bad += std::memcmp(&b[i], &a[j], sizeof(PixelState)) != 0 ||
               std::memcmp(&bh[i], &ah[j], sizeof(HpfState)) != 0;
      }
      CHECK(bad == 0);
    }
    // Without the filter the same probe is refused.
    SensorParams q = p;
    q.tau_hp = 0;
    SensorModel full(p, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, Precision::kFp32);
    const ProbeResult probe = full.stationary_probe(sc.t_init, f.data());
    SensorModel off(q, sc.W, sc.H, sc.seed, sc.groups, sc.camera, 2, Precision::kFp32);
    CHECK_THROWS_AS(off.stationary_init(sc.t_init, f.data(), probe), std::invalid_argument);
  }
}

TEST_CASE("S5 ResimRun refuses an inconsistent replay") {
  const Scene sc;
  SensorParams p = base_params();
  const Writer w = write(sc, p, Precision::kFp32);
  const SensorSpec spec = spec_of(sc, p, Precision::kFp32);
  CHECK_THROWS_AS(ResimRun(w.cache, spec, Roi{4, 6, 40, 38}, sc.t_e, StartMode::kPhase), std::invalid_argument);
  CHECK_THROWS_AS(ResimRun(w.cache, spec, sc.roi, sc.t_s + kW + 1, StartMode::kPhase), std::invalid_argument);
  // t1 at a warm-up's end
  CHECK_THROWS_AS(ResimRun(w.cache, spec, sc.roi, sc.t_s, StartMode::kPhase), std::invalid_argument);
  CacheView bad = w.cache;
  bad.calls[2].count[0] += 1;
  CHECK_THROWS_AS(ResimRun(bad, spec, sc.roi, sc.t_e, StartMode::kPhase), std::invalid_argument);
  // Veiling glare on a cache without frame means: refused before the model is built.
  SensorSpec glare = spec;
  glare.params.veiling_glare = 0.01;
  CHECK_NOTHROW(ResimRun(w.cache, glare, sc.roi, sc.t_e, StartMode::kPhase));
  CacheView no_means = w.cache;
  no_means.frame_mean.clear();
  CHECK_NOTHROW(ResimRun(no_means, spec, sc.roi, sc.t_e, StartMode::kPhase));
  CHECK_THROWS_WITH_AS(ResimRun(no_means, glare, sc.roi, sc.t_e, StartMode::kPhase),
                       doctest::Contains("frame means"), std::invalid_argument);
  // The camera-wide rate limiter: no ROI replay of it (tier R).
  SensorSpec limited = spec;
  limited.params.limiter = LimiterMode::kEpoch;
  limited.params.limiter_rate0_hz = 1e6;
  limited.params.limiter_rate1_hz = 5e5;
  limited.params.limiter_period_s = 0.01;
  limited.params.limiter_depth = 64;
  CHECK_NOTHROW(limited.params.validate());
  CHECK_THROWS_WITH_AS(ResimRun(w.cache, limited, sc.roi, sc.t_e, StartMode::kPhase),
                       doctest::Contains("sensor.limiter"), std::invalid_argument);
  ResimRun run(w.cache, spec, sc.roi, sc.t_e, StartMode::kPhase);
  RawEvents out;
  CHECK(run.step(&out));
  run.cancel();
  CHECK_FALSE(run.step(&out));
  CHECK(run.calls_done() == 1);
}

TEST_SUITE("cuda") {
  TEST_CASE("S1 the ROI model on the CUDA device equals the CPU fp32 replay and the writer run") {
    const Scene sc;
    for (const bool stationary : {false, true}) {
      INFO(std::string(stationary ? "stationary" : "phase"));
      SensorParams p = base_params();
      p.initial_state = stationary ? InitialState::kStationary : InitialState::kPhase;
      p.hot_fraction = 0.02;
      p.veiling_glare = 0.01;
      p.tau_hp = 0.05;
      p.tau_hp_sigma_ln = 0.3;
      const Writer w = write(sc, p, Precision::kFp32);
      const auto want = engine_outputs(sc, w, sc.roi, 3);
      const StartMode mode = stationary ? StartMode::kStationary : StartMode::kPhase;
      replay(sc, w, spec_of(sc, p, Precision::kFp32), sc.roi, sc.t_e, mode, nullptr, want, Device::kCuda);
    }
  }
}
