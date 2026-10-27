// The lean sensor kernel of ENGINE_CPP I2 against the kernel it replaced, bit for bit: every scenario
// runs the frozen I1 kernel (pixel_i1.h) on the CPU in fp32 with the I1 GPU's comparator chunk (16),
// the new SensorModel on the CPU in fp32 and, when available, on the GPU (quiet and general passes,
// under the default launch policy and optionally a forcing one), and requires the same events (every
// field) and the same states after every call (F034 aside: first events' labels and FIRST_AFTER_INIT, which I1 did
// not have; check_same and check_states). The one deliberate difference, where a solver-fallback
// or quarantine flag goes (pixel.h "Lean form"), is pinned against the I1 kernel at chunk 1, whose
// comparator runs after every sub-step. Also: GPU launches on small images, where I1's kernel needed
// 150-180 ms, are reported over DESIGN §4.2's 50 ms cap and fail over 100 ms, also when a few pixels
// carry far more work per unit than the planner's work units count.
//
// The oracle shares the production maths headers (substep.h, expcubic.h, noise.h, waveform.h, stage1.h), so
// these tests cannot see a change there; the frozen bench references and the sensor replay can. I6' changed fp32
// results on purpose, within tolerance: the small-slope stage-1 branch (stage1.h, shared, so invisible here) and the
// in-cell waveform bounds of the lean kernel (pixel.h; the oracle keeps the piece's own), which never move a piece in
// these scenarios, so they stay bitwise; the replay tier guards both (ENGINE_CPP I6p notes).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "pixel_i1.h"
#include "sensor/sensor_model.h"

using namespace balla::sensor;

namespace {

constexpr int64_t kW = int64_t{1} << 23;

static_assert(sizeof(i1::PixelState) == sizeof(PixelState), "the I1 state layout is the saved-state format");

i1::KernelConstants<float> old_constants(const KernelConstants<float>& k) {
  i1::KernelConstants<float> o{};
  o.cc = k.cc;
  o.eps_pl_base = k.eps_pl_base;
  o.scale = k.scale;
  o.irr_max = k.irr_max;
  o.groups = k.groups;
  o.modulated = k.modulated;
  for (int g = 0; g < kMaxLightGroups; ++g) o.waves[g] = k.waves[g];
  o.noise = k.noise;
  o.static_fast_path = k.static_fast_path;
  o.levels = k.levels;
  o.margin_z = k.margin_z;
  o.tables = k.tables;
  o.ideal_reset = k.ideal_reset;
  o.refractory_ns = k.refractory_ns;
  o.eps_ci = k.eps_ci;
  o.seed = k.seed;
  o.camera = k.camera;
  return o;
}

// One call's packed runs for the listed pixels.
struct Window {
  std::vector<uint32_t> ids;
  std::vector<int64_t> start;
  std::vector<int32_t> count;
  std::vector<int64_t> t;
  std::vector<float> irr;
};

struct Scenario {
  std::string name;
  SensorParams params;
  std::vector<Waveform> groups;
  uint32_t width = 40, height = 32;
  int64_t t_init = 3 * kW;
  // Call ends (absolute times, non-decreasing); empty: `windows` calls of one 2^23 ns window each.
  std::vector<int64_t> ends = {};
  int windows = 3;
  std::optional<GpuLaunchPolicy> policy = {};  // the GPU model's launch policy (default when empty)
  std::function<float(uint32_t px, int g)> irr0 = {};
  // Samples of pixel px in call c over [t0, t1]: appends (t, irr[G]) or nothing (unlisted).
  std::function<void(uint32_t px, int c, int64_t t0, int64_t t1, int groups, Window&)> samples = {};

  std::vector<int64_t> call_ends() const {
    if (!ends.empty()) return ends;
    std::vector<int64_t> e;
    for (int w = 0; w < windows; ++w) e.push_back(t_init + (w + 1) * kW);
    return e;
  }
};

Scenario make(std::string name, const SensorParams& p, std::vector<Waveform> groups) {
  Scenario sc;
  sc.name = std::move(name);
  sc.params = p;
  sc.groups = std::move(groups);
  return sc;
}

struct OldSink {
  std::vector<i1::RawEvent>* out;
  void push(const i1::RawEvent& e) { out->push_back(e); }
};

// The I1 kernel's init on every pixel, on its own states.
std::vector<i1::PixelState> old_init(const i1::KernelConstants<float>& okc, const SensorMaps& maps, uint32_t n,
                                     int64_t t_init, const std::vector<float>& irr0, int G) {
  std::vector<i1::PixelState> states(n);
  std::vector<i1::RawEvent> none;
  OldSink sink{&none};
  i1::KernelStats stats;
  SubStep<float> scratch[1];
  const i1::InputRun in{nullptr, nullptr, 0};
  for (uint32_t px = 0; px < n; ++px) {
    states[px] = i1::PixelState{};
    i1::PixelKernel<float, OldSink> k(okc, maps.pixels[px], px, states[px], t_init, in, scratch, 1, sink, stats);
    k.init(irr0.data() + static_cast<size_t>(px) * G);
  }
  return states;
}

// One call [t0, t1] of the I1 kernel with comparator chunks of `chunk` sub-steps on every pixel; the
// events in the model's order (t_cross, then pixel).
std::vector<i1::RawEvent> old_call(const i1::KernelConstants<float>& okc, const SensorMaps& maps,
                                   std::vector<i1::PixelState>& states, const Window& win, int G, int64_t t0,
                                   int64_t t1, int chunk) {
  const uint32_t n = static_cast<uint32_t>(states.size());
  std::vector<i1::RawEvent> events;
  OldSink sink{&events};
  i1::KernelStats stats;
  std::vector<SubStep<float>> scratch(static_cast<size_t>(chunk));
  std::vector<int32_t> row_of(n, -1);
  for (size_t i = 0; i < win.ids.size(); ++i) row_of[win.ids[i]] = static_cast<int32_t>(i);
  for (uint32_t px = 0; px < n; ++px) {
    i1::InputRun in{nullptr, nullptr, 0};
    if (row_of[px] >= 0) {
      const size_t r = static_cast<size_t>(row_of[px]);
      in = {win.t.data() + win.start[r], win.irr.data() + win.start[r] * G, win.count[r]};
    }
    i1::PixelKernel<float, OldSink> k(okc, maps.pixels[px], px, states[px], t0, in, scratch.data(), chunk, sink, stats);
    k.advance(t1);
  }
  std::sort(events.begin(), events.end(), [](const i1::RawEvent& a, const i1::RawEvent& b) {
    return a.t_cross != b.t_cross ? a.t_cross < b.t_cross : a.pixel < b.pixel;
  });
  return events;
}

void check_same(const std::vector<i1::RawEvent>& old_events, const Events& ev, const SensorModel& model,
                const std::string& what) {
  CAPTURE(what);
  CHECK_MESSAGE(old_events.size() == ev.size(), what << ": " << old_events.size() << " I1 events, " << ev.size() << " now");
  if (old_events.size() != ev.size()) return;
  for (size_t i = 0; i < ev.size(); ++i) {
    const i1::RawEvent& e = old_events[i];
    CHECK(e.t_cross == ev.t_cross_ns[i]);
    CHECK(e.pixel % model.width() == ev.x[i]);
    CHECK(e.pixel / model.width() == ev.y[i]);
    CHECK(e.polarity == ev.p[i]);
    // F034: a pixel's first event after init carries FIRST_AFTER_INIT and its label leaves out the reference offset
    // (phi0 here), which I1 booked as leak; every other event's label is I1's.
    if ((ev.flags[i] & kFlagFirstAfterInit) == 0) CHECK(e.label == ev.label[i]);
    CHECK(e.flags == (ev.flags[i] & ~kFlagFirstAfterInit));
    CHECK(e.ds == ev.ds[i]);
    CHECK(e.dn == ev.dn[i]);
    CHECK(e.dl == ev.dl[i]);
    CHECK(model.reported_time(e.t_cross) == ev.t_reported_ns[i]);
  }
}

void check_states(const std::vector<i1::PixelState>& old_states, const SensorModel& model, const std::string& what) {
  INFO(what);
  const std::vector<PixelState> st = model.states();
  REQUIRE(st.size() == old_states.size());
  std::vector<PixelState> same = st;  // but for FIRST_AFTER_INIT (F034), which I1 did not have
  for (PixelState& s : same) s.flags = static_cast<uint16_t>(s.flags & ~kFlagFirstAfterInit);
  CHECK(std::memcmp(same.data(), old_states.data(), same.size() * sizeof(PixelState)) == 0);
}

struct Outcome {
  size_t events = 0;   // all calls
  size_t flagged = 0;  // events with a solver-fallback or quarantine flag
};

constexpr uint16_t kRareFlags = kFlagSolverFallback | kFlagQuarantined;

// Runs a scenario through the I1 kernel (comparator chunk `chunk`), the new CPU fp32 path and (when
// available) the lean GPU path; returns what fired, so that scenarios can assert that they exercise
// firing.
Outcome run_scenario(const Scenario& sc, int chunk = 16) {
  const uint32_t n = sc.width * sc.height;
  const int G = static_cast<int>(sc.groups.size());
  std::optional<SensorModel> gpu;
  if (sensor_cuda_available()) {
    gpu.emplace(sc.params, sc.width, sc.height, 7, sc.groups, 0, 1, Precision::kFp32, Device::kCuda);
    if (sc.policy) gpu->set_gpu_launch_policy(*sc.policy);
  }
  SensorModel cpu(sc.params, sc.width, sc.height, 7, sc.groups, 0, 4, Precision::kFp32, Device::kCpu);
  std::vector<float> irr0(static_cast<size_t>(n) * G);
  for (uint32_t px = 0; px < n; ++px) {
    for (int g = 0; g < G; ++g) irr0[static_cast<size_t>(px) * G + g] = sc.irr0(px, g);
  }
  if (gpu) gpu->init(sc.t_init, irr0.data());
  cpu.init(sc.t_init, irr0.data());
  const i1::KernelConstants<float> okc = old_constants(cpu.constants32());
  std::vector<i1::PixelState> old_states = old_init(okc, cpu.maps(), n, sc.t_init, irr0, G);
  check_states(old_states, cpu, sc.name + ": init (CPU)");
  if (gpu) check_states(old_states, *gpu, sc.name + ": init (GPU)");
  Outcome out;
  const std::vector<int64_t> ends = sc.call_ends();
  int64_t t0 = sc.t_init;
  for (size_t c = 0; c < ends.size(); ++c) {
    const int64_t t1 = ends[c];
    Window win;
    for (uint32_t px = 0; px < n; ++px) {
      const size_t before = win.t.size();
      sc.samples(px, static_cast<int>(c), t0, t1, G, win);
      if (win.t.size() > before) {
        win.ids.push_back(px);
        win.start.push_back(static_cast<int64_t>(before));
        win.count.push_back(static_cast<int32_t>(win.t.size() - before));
      }
    }
    const std::vector<i1::RawEvent> old_events = old_call(okc, cpu.maps(), old_states, win, G, t0, t1, chunk);
    const Events ec = cpu.advance_runs(win.ids.data(), win.ids.size(), win.t.data(), win.irr.data(), win.t.size(),
                                       win.start.data(), win.count.data(), t1);
    const std::string what = sc.name + ": call " + std::to_string(c);
    check_same(old_events, ec, cpu, what + " (CPU)");
    check_states(old_states, cpu, what + " (CPU)");
    if (gpu) {
      const Events eg = gpu->advance_runs(win.ids.data(), win.ids.size(), win.t.data(), win.irr.data(),
                                          win.t.size(), win.start.data(), win.count.data(), t1);
      check_same(old_events, eg, *gpu, what + " (GPU)");
      check_states(old_states, *gpu, what + " (GPU)");
    }
    out.events += old_events.size();
    for (const i1::RawEvent& e : old_events) out.flagged += (e.flags & kRareFlags) != 0;
    t0 = t1;
  }
  return out;
}

// Lattice-like runs: samples every `step` ns from t0 up to t1 (t1 included).
void lattice(uint32_t px, int64_t t0, int64_t t1, int64_t step, int G, Window& win,
             const std::function<float(uint32_t, int, int64_t)>& f) {
  for (int64_t t = t0;; t = std::min(t + step, t1)) {
    win.t.push_back(t);
    for (int g = 0; g < G; ++g) win.irr.push_back(f(px, g, t));
    if (t == t1) break;
  }
}

const Waveform kFlicker = make_fourier_waveform(120.0, {0.02}, {0.0});

// A bright edge moving across the image under 120 Hz light (group 1; group 0 is a dark constant).
float edge(uint32_t width, uint32_t q, int g, int64_t t) {
  const double x = static_cast<double>(q % width) - 2e-7 * static_cast<double>(t - 3 * kW);
  const float v = static_cast<float>(40.0 + 400.0 / (1.0 + std::exp(-2.0 * (x - 0.5 * width))));
  return g == 1 ? v : 0.0f;
}

// The edge, sampled every 2^17 ns on most pixels and every 37 x 2^10 ns on every fifth (refined).
Scenario edges_scenario(std::string name, const SensorParams& p, uint32_t width, uint32_t height) {
  Scenario sc = make(std::move(name), p, {make_constant_waveform(), kFlicker});
  sc.width = width;
  sc.height = height;
  sc.irr0 = [width](uint32_t px, int g) { return edge(width, px, g, 3 * kW); };
  sc.samples = [width](uint32_t px, int, int64_t t0, int64_t t1, int G, Window& win) {
    if (t1 == t0) return;
    lattice(px, t0, t1, px % 5 == 0 ? (int64_t{1} << 10) * 37 : int64_t{1} << 17, G, win,
            [width](uint32_t q, int g, int64_t t) { return edge(width, q, g, t); });
  };
  return sc;
}

// GPU launch policies that force short windows, small batches, event-buffer overflows and re-runs.
GpuLaunchPolicy forcing_policy() {
  GpuLaunchPolicy pol;
  pol.max_slice_ns = 3 * (int64_t{1} << 16) + 777;
  pol.max_pixels = 37;
  pol.event_capacity = 1;
  return pol;
}

GpuLaunchPolicy tiny_target_policy() {
  GpuLaunchPolicy pol;
  pol.target_ms = 0.001;
  pol.max_pixels = 5;
  return pol;
}

}  // namespace

TEST_CASE("lean kernel = I1 kernel: static scenes (constant light, 120 Hz, dark pixels under 120 Hz)") {
  SensorParams p = preset("prophesee_gen3");
  p.hot_fraction = 0;
  // Static, constant light: every cell is quiet unless the noise nears a level.
  Scenario constant = make("static constant light", p, {make_constant_waveform()});
  constant.irr0 = [](uint32_t px, int) { return 5.0f + 20.0f * static_cast<float>(px % 9); };
  constant.samples = [](uint32_t, int, int64_t, int64_t, int, Window&) {};
  CHECK(run_scenario(constant).events > 0);
  // Static, 120 Hz: every lit cell takes the general path.
  Scenario flicker = make("static 120 Hz", p, {make_constant_waveform(), kFlicker});
  flicker.irr0 = [](uint32_t px, int g) { return g == 1 ? 50.0f + 10.0f * static_cast<float>(px % 13) : 0.0f; };
  flicker.samples = [](uint32_t, int, int64_t, int64_t, int, Window&) {};
  CHECK(run_scenario(flicker).events > 0);
  // Dark pixels (zero irradiance in every group) under a modulated group: quiet cells (I1 took the
  // general path for them), mixed with lit pixels and listed with constant lattice samples.
  Scenario dark = make("dark pixels under 120 Hz", p, {make_constant_waveform(), kFlicker});
  dark.irr0 = [](uint32_t px, int g) { return g == 1 && px % 3 == 0 ? 80.0f : 0.0f; };
  dark.samples = [](uint32_t px, int, int64_t t0, int64_t t1, int G, Window& win) {
    if (px % 2 == 0) lattice(px, t0, t1, kW / 2, G, win, [](uint32_t q, int g, int64_t) {
      return g == 1 && q % 3 == 0 ? 80.0f : 0.0f;
    });
  };
  CHECK(run_scenario(dark).events > 0);
  // Leak-only dark pixels (dark current alone, every group dead) with a charge-injection offset: the
  // quiet pass fires nothing, so each leak event leaves its cell to the general path.
  SensorParams pl = p;
  pl.leak_rate_hz = 20.0;
  pl.eps_ci = 0.004;
  Scenario leak = make("leak-only dark pixels, eps_ci", pl, {make_constant_waveform(), kFlicker});
  leak.irr0 = [](uint32_t, int) { return 0.0f; };
  leak.samples = [](uint32_t px, int, int64_t t0, int64_t t1, int G, Window& win) {
    if (px % 4 == 0) lattice(px, t0, t1, kW / 2, G, win, [](uint32_t, int, int64_t) { return 0.0f; });
  };
  CHECK(run_scenario(leak).events > 0);
}

TEST_CASE("lean kernel = I1 kernel: edges, hot pixels, bright jumps, a pulse group, noise off") {
  // A moving edge crossing the image: lattice samples at 2^17 ns plus refinement samples near the edge
  // (on the 2^10 grid), 120 Hz light, hot pixels; also under the forcing GPU policies.
  SensorParams p = preset("prophesee_gen3");
  p.hot_fraction = 0.03;
  Scenario edges = edges_scenario("edges and hot pixels", p, 40, 32);
  CHECK(run_scenario(edges).events > 50);
  edges.name = "edges and hot pixels, forced GPU slices, batches and overflows";
  edges.policy = forcing_policy();
  CHECK(run_scenario(edges).events > 50);
  edges.name = "edges and hot pixels, GPU target 1 us, 5-pixel batches";
  edges.policy = tiny_target_policy();
  CHECK(run_scenario(edges).events > 50);
  // Bright jumps: steps of 2-3x in a few microseconds on bright pixels (Dawson/erfcx branches).
  SensorParams pb = preset("prophesee_gen3");
  pb.hot_fraction = 0;
  Scenario jumps = make("bright jumps", pb, {make_constant_waveform()});
  jumps.width = 24;
  jumps.height = 16;
  jumps.irr0 = [](uint32_t px, int) { return 2000.0f + 100.0f * static_cast<float>(px % 7); };
  jumps.samples = [](uint32_t px, int w, int64_t t0, int64_t t1, int, Window& win) {
    const float lo = 2000.0f + 100.0f * static_cast<float>(px % 7), hi = lo * (2.0f + static_cast<float>(px % 3) * 0.5f);
    const int64_t tj = t0 + (int64_t{1} << 10) * (1000 + 37 * static_cast<int64_t>(px % 11));
    const float a = w % 2 == 0 ? lo : hi, b = w % 2 == 0 ? hi : lo;
    for (const auto [t, v] : {std::pair<int64_t, float>{t0, a}, {tj, a}, {tj + 4096, b}, {t1, b}}) {
      win.t.push_back(t);
      win.irr.push_back(v);
    }
  };
  CHECK(run_scenario(jumps).events > 50);
  // A pulse group nobody's light uses (the lab scene's strobe): its breakpoints still cut every
  // pixel's pieces, also in quiet cells.
  Scenario pulse = make("dead pulse group", p, {make_constant_waveform(), kFlicker, make_pulse_waveform(200.0, 0.06, 5e-5, 0.0, 0.0)});
  pulse.irr0 = [](uint32_t px, int g) { return g == 0 ? 30.0f + static_cast<float>(px % 17) : (g == 1 && px % 4 == 0 ? 60.0f : 0.0f); };
  pulse.samples = [](uint32_t px, int, int64_t t0, int64_t t1, int G, Window& win) {
    if (px % 3 == 0) lattice(px, t0, t1, kW / 4, G, win, [](uint32_t q, int g, int64_t t) {
      return g == 0 ? 30.0f + static_cast<float>(q % 17) + 1e-6f * static_cast<float>(t % 1000) : (g == 1 && q % 4 == 0 ? 60.0f : 0.0f);
    });
  };
  CHECK(run_scenario(pulse).events > 0);
  // Noise off: no quiet pass, whole windows per launch.
  SensorParams pq = preset("prophesee_gen3");
  pq.noise = false;
  pq.hot_fraction = 0.02;
  Scenario quiet = edges_scenario("edges, noise off", pq, 40, 32);
  CHECK(run_scenario(quiet).events > 50);
}

TEST_CASE("lean kernel = I1 kernel: negative, off-grid, mid-cell, zero-length and multi-window calls") {
  SensorParams p = preset("prophesee_gen3");
  p.hot_fraction = 0.05;
  Scenario sc = edges_scenario("odd call boundaries before t = 0", p, 24, 16);
  // From -2 W - 4321 ns: a call ending mid-cell, a zero-length call, one ending off the grid before
  // t = 0, one spanning several windows across t = 0 and ending mid-cell, and a last one on the grid.
  sc.t_init = -2 * kW - 4321;
  sc.ends = {sc.t_init + 54321, sc.t_init + 54321, -kW + 777, 2 * kW + 3 * 65536 + 99, 3 * kW};
  CHECK(run_scenario(sc).events > 50);
  sc.name = "odd call boundaries before t = 0, forced GPU slices, batches and overflows";
  sc.policy = forcing_policy();
  CHECK(run_scenario(sc).events > 50);
}

TEST_CASE("lean kernel = I1 kernel: a physical refractory longer than half a window") {
  // A 4.5 ms blind time spans cells, slices and window boundaries; hot pixels (<= 200 Hz, so that
  // rate x T_r < 1) and brightness steps every window keep pixels firing.
  SensorParams p = preset("prophesee_gen3");
  p.refractory_s = 4.5e-3;
  p.hot_fraction = 0.05;
  p.hot_rate_max_hz = 200.0;
  p.theta_on = p.theta_off = 0.15;
  Scenario sc = make("refractory 4.5 ms", p, {make_constant_waveform(), kFlicker});
  sc.width = 24;
  sc.height = 16;
  sc.windows = 4;
  sc.irr0 = [](uint32_t px, int g) { return g == 1 ? 100.0f + 10.0f * static_cast<float>(px % 5) : 0.0f; };
  sc.samples = [](uint32_t px, int w, int64_t t0, int64_t t1, int G, Window& win) {
    const float base = 100.0f + 10.0f * static_cast<float>(px % 5);
    const float before = base * (w % 2 == 0 ? 1.0f : 2.5f), after = base * (w % 2 == 0 ? 2.5f : 1.0f);
    const int64_t tj = t0 + (int64_t{1} << 10) * (2000 + 211 * static_cast<int64_t>(px % 13));
    for (const auto [t, v] : {std::pair<int64_t, float>{t0, before}, {tj, before}, {tj + 2048, after}, {t1, after}}) {
      win.t.push_back(t);
      for (int g = 0; g < G; ++g) win.irr.push_back(g == 1 ? v : 0.0f);
    }
  };
  CHECK(run_scenario(sc).events > 50);
}

TEST_CASE("lean kernel = I1 kernel: ideal reset with noise at leaf levels 0 and 10, tiny images") {
  SensorParams p = preset("prophesee_gen3");
  p.hot_fraction = 0.05;
  p.reset_mode = ResetMode::kIdeal;
  p.refractory_s = 0.0;
  for (const int leaf : {0, 10}) {
    p.noise_leaf_level = leaf;
    CHECK(run_scenario(edges_scenario("ideal reset, leaf level " + std::to_string(leaf), p, 16, 12)).events > 50);
  }
  // Below a warp and below the quiet pass's 1024-pixel batch floor.
  SensorParams pt = preset("prophesee_gen3");
  pt.hot_fraction = 0.2;
  for (const auto [w, h] : {std::pair<uint32_t, uint32_t>{5, 3}, {1, 1}}) {
    Scenario sc = edges_scenario("tiny image " + std::to_string(w) + " x " + std::to_string(h), pt, w, h);
    sc.windows = 6;
    CHECK(run_scenario(sc).events > 0);
  }
}

TEST_CASE("solver-fallback flags go to the first event after their sub-step (I1 kernel at chunk 1)") {
  // Steps of 3x at ~10 nA (illuminance x 1e4): the cascade is too stiff for the sub-step test above
  // the 1 ns floor, so the sub-steps at each step are accepted as fallbacks; low thresholds make events
  // follow within a few sub-steps. Flags accumulate from their sub-step to the next event. I1 made a
  // chunk's 16 sub-steps before comparing them, so an event in a chunk also carried a flag raised
  // later in that chunk; at chunk 1 its comparator runs after every sub-step, as the lean kernel's
  // does, and the two agree in every field.
  SensorParams p = preset("prophesee_gen3");
  p.hot_fraction = 0;
  p.illuminance_scale = 1e4;
  p.theta_on = p.theta_off = 0.05;
  p.refractory_s = 2e-6;
  static constexpr float kLo = 3e4f;
  Scenario sc = make("stiff steps with fallback sub-steps", p, {make_constant_waveform()});
  sc.width = 8;
  sc.height = 4;
  sc.windows = 2;
  sc.irr0 = [](uint32_t, int) { return kLo; };
  sc.samples = [](uint32_t px, int, int64_t t0, int64_t t1, int, Window& win) {
    // A 3x step up or down every 41 us, offset per pixel.
    float v = kLo;
    win.t.push_back(t0);
    win.irr.push_back(v);
    for (int64_t t = t0 + 20000 + 997 * static_cast<int64_t>(px); t + 1 < t1; t += 41000) {
      win.t.push_back(t);
      win.irr.push_back(v);
      v = v > kLo ? kLo : 3.0f * kLo;
      win.t.push_back(t + 1);
      win.irr.push_back(v);
    }
    win.t.push_back(t1);
    win.irr.push_back(v);
  };
  const Outcome o = run_scenario(sc, 1);
  CHECK(o.events > 100);
  CHECK(o.flagged > 0);
  // The scenario sees the deviation: at chunk 16, the I1 kernel flags some event before its fallback
  // sub-step, and differs from chunk 1 in nothing but flags.
  const uint32_t n = sc.width * sc.height;
  SensorModel model(p, sc.width, sc.height, 7, sc.groups, 0, 1, Precision::kFp32, Device::kCpu);
  const i1::KernelConstants<float> okc = old_constants(model.constants32());
  const std::vector<float> irr0(n, kLo);
  std::vector<i1::PixelState> s1 = old_init(okc, model.maps(), n, sc.t_init, irr0, 1), s16 = s1;
  size_t earlier = 0;
  for (int w = 0; w < sc.windows; ++w) {
    const int64_t t0 = sc.t_init + w * kW, t1 = t0 + kW;
    Window win;
    for (uint32_t px = 0; px < n; ++px) {
      const size_t before = win.t.size();
      sc.samples(px, w, t0, t1, 1, win);
      win.ids.push_back(px);
      win.start.push_back(static_cast<int64_t>(before));
      win.count.push_back(static_cast<int32_t>(win.t.size() - before));
    }
    const std::vector<i1::RawEvent> e1 = old_call(okc, model.maps(), s1, win, 1, t0, t1, 1);
    const std::vector<i1::RawEvent> e16 = old_call(okc, model.maps(), s16, win, 1, t0, t1, 16);
    REQUIRE(e1.size() == e16.size());
    for (size_t i = 0; i < e1.size(); ++i) {
      CHECK(e1[i].t_cross == e16[i].t_cross);
      CHECK(e1[i].pixel == e16[i].pixel);
      CHECK(e1[i].ds == e16[i].ds);
      CHECK(e1[i].dn == e16[i].dn);
      CHECK(e1[i].dl == e16[i].dl);
      // At chunk 16 the first event of a chunk takes the flags of every sub-step in it, also of later
      // ones, which an event after them at chunk 1 (and in the lean kernel) carries instead.
      earlier += (e16[i].flags & ~e1[i].flags & kRareFlags) != 0;
    }
  }
  MESSAGE(o.events << " events, " << o.flagged << " flagged; " << earlier
                   << " flagged early by the I1 kernel at chunk 16");
  CHECK(earlier > 0);
}

TEST_CASE("GPU launches stay under 50 ms on small images: bright moving edge, noise on and off; heavy pixels" *
          doctest::test_suite("cuda")) {
  // The engine's card scene in sensor terms: 4-30 pA (illuminance 2e4), an edge at 1000 px/s, 9 samples
  // per window; I1's kernel took 150-180 ms per launch on 64 x 48 without noise.
  for (const uint32_t H : {48u, 64u}) {
    for (const bool noise : {false, true}) {
      SensorParams p = preset("prophesee_gen3");
      p.noise = noise;
      p.illuminance_scale = 2e4;
      const uint32_t W = 64;
      SensorModel s(p, W, H, 1, {make_constant_waveform()}, 0, 1, Precision::kFp32, Device::kCuda);
      const auto value = [&](uint32_t px, int64_t t) {
        const double x = static_cast<double>(px % W) - 10.0 - 1e-6 * static_cast<double>(t);  // 1000 px/s
        return static_cast<float>(x < 0 ? 1.5 : 0.2);
      };
      std::vector<float> irr0(W * H);
      for (uint32_t px = 0; px < W * H; ++px) irr0[px] = value(px, 0);
      s.init(-50'000'000, irr0.data());
      s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, 0);
      s.reset_stats();
      for (int w = 0; w < 4; ++w) {
        std::vector<uint32_t> ids;
        std::vector<int64_t> t;
        std::vector<float> irr;
        for (uint32_t px = 0; px < W * H; ++px) {
          ids.push_back(px);
          for (int j = 0; j <= 8; ++j) {
            const int64_t tj = w * kW + kW * j / 8;
            t.push_back(tj);
            irr.push_back(value(px, tj));
          }
        }
        s.advance(ids.data(), ids.size(), t.data(), irr.data(), 9, nullptr, (w + 1) * kW);
      }
      const GpuLaunchStats& ls = s.gpu_launch_stats();
      MESSAGE("64 x " << H << ", noise " << noise << ": " << ls.launches << " launches, max " << ls.max_ms << " ms");
      WARN(ls.max_ms < 50.0);   // DESIGN §4.2's cap, reported: the desktop and other jobs share the GPU
      CHECK(ls.max_ms < 100.0);  // twice the cap: a planner that stops slicing takes 106 ms or more
    }
  }
  // A 24 x 20 image whose pixels each take far more time per work unit than the planner's units count
  // (cells + 16 x samples): 100 samples per window off the 2^10 grid, each a 1.5-3x step, under a strong
  // 120 Hz group, low thresholds, 25 % hot pixels up to 3 kHz, and a constant group lighting up in
  // the middle of the third window. Every launch here is smaller than a full wave, so only the
  // learnt latency of the heaviest pixel bounds it; the I2 planner before review ran each window in
  // one launch (~160 ms).
  SensorParams p = preset("prophesee_gen3");
  p.theta_on = 0.12;
  p.theta_off = 0.15;
  p.hot_fraction = 0.25;
  p.hot_rate_max_hz = 3000;
  p.illuminance_scale = 100;
  const uint32_t W = 24, H = 20, n = W * H;
  SensorModel s(p, W, H, 3, {make_constant_waveform(), make_fourier_waveform(120.0, {0.3}, {0.0})}, 0, 1,
                Precision::kFp32, Device::kCuda);
  std::vector<float> irr0(2 * n);
  for (uint32_t px = 0; px < n; ++px) {
    irr0[2 * px] = 0.0f;
    irr0[2 * px + 1] = 1.0f + 0.1f * static_cast<float>(px % 7);
  }
  s.init(0, irr0.data());
  size_t events = 0;
  for (int w = 0; w < 4; ++w) {
    std::vector<uint32_t> ids;
    std::vector<int64_t> t, start;
    std::vector<int32_t> count;
    std::vector<float> irr;
    const int m = 100;
    for (uint32_t px = 0; px < n; ++px) {
      ids.push_back(px);
      start.push_back(static_cast<int64_t>(t.size()));
      const float base = 1.0f + 0.1f * static_cast<float>(px % 7);
      for (int j = 0; j <= m; ++j) {
        const int64_t tj = j == m ? (w + 1) * kW : w * kW + (kW / m) * j + (j > 0 ? 37 * static_cast<int64_t>(px % 23) : 0);
        t.push_back(tj);
        irr.push_back(w > 2 || (w == 2 && 2 * j > m) ? 50.0f : 0.0f);
        irr.push_back(base * (1.0f + 0.5f * static_cast<float>((j * 7 + static_cast<int>(px)) % 5)));
      }
      count.push_back(m + 1);
    }
    events += s.advance_runs(ids.data(), ids.size(), t.data(), irr.data(), t.size(), start.data(), count.data(),
                             (w + 1) * kW)
                  .size();
  }
  const GpuLaunchStats& ls = s.gpu_launch_stats();
  MESSAGE("24 x 20, heavy pixels: " << events << " events, " << ls.launches << " launches, max " << ls.max_ms << " ms");
  CHECK(events > 1000);
  WARN(ls.max_ms < 50.0);
  CHECK(ls.max_ms < 100.0);
}
