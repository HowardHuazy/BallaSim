// The engine on the host (ENGINE_CPP §1.7, I4): Embree, the CPU sensor and the serial primitives through
// make_engine, in every preset (linux-cpu included). Its end-to-end checks are in tests/python/test_engine_run.py
// (both device pairs) and the bench's IDENTICAL rows (BENCH.md).
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "engine/band_map.h"
#include "engine/engine.h"
#include "scenes.h"

using namespace balla;
using namespace balla::engine;
using namespace balla::render::test;

namespace {

constexpr int64_t kW = int64_t{1} << 23, kT0 = 120 * kW;

// A box turning and sliding in front of a wall, under a shadow-casting point light.
render::SceneBundle box_scene() {
  render::SceneBundle s;
  s.light_groups = {"default"};
  s.trajectories.push_back(static_trajectory(kT0 - kW, kT0 + 8 * kW, quat_identity<double>(), {0.0, 0.0, 0.0}));
  s.trajectories.push_back(moving_trajectory(kT0 - kW, kT0 + 8 * kW, quat_identity<double>(), {-0.2, 0.0, 1.0},
                                             {0.6, 0.0, 0.0}, {0.0, 0.0, 2.0}));
  s.cameras.push_back(pinhole(40, 30, 40.0, 0));
  render::Body wall = make_body(1, plane_mesh(2.0, 2), {0.3f, 0.3f, 0.3f});
  wall.T_world_body = translation(0.0, 0.0, 2.0);
  s.bodies.push_back(wall);
  render::Body box = make_body(2, box_mesh(0.1f, 0.1f, 0.1f), {0.9f, 0.9f, 0.9f});
  box.trajectory = 1;
  s.bodies.push_back(box);
  s.lights.push_back(point_light({0.3, -0.4, 0.0}, 2.0f, true));
  s.lights.push_back(ambient_light(0.3f));
  s.t_start_ns = kT0 - kW;  // the span the trajectories cover (the motion warm-up's)
  s.t_end_ns = kT0 + 8 * kW;
  return s;
}

// Changes to the test engine's configuration and sensor.
using Edit = std::function<void(EngineConfig&, SensorSpec&)>;

std::unique_ptr<Engine> engine_on(render::BackendKind backend, double budget_mb, const Edit& edit = {},
                                  render::SceneBundle bundle = box_scene()) {
  EngineConfig c;
  c.render = options(backend, 4);
  c.render.scene.n_subsamples = 16;
  c.light_groups = {sensor::make_constant_waveform()};
  c.device_budget_mb = budget_mb;
  SensorSpec s;
  s.params = sensor::preset("prophesee_gen3");
  s.params.illuminance_scale = 200.0;
  s.precision = sensor::Precision::kFp32;
  s.num_threads = 4;
  if (edit) edit(c, s);
  const std::vector<std::optional<SensorSpec>> sensors(bundle.cameras.size(), s);
  return make_engine(std::move(bundle), c, sensors);
}

std::unique_ptr<Engine> host_engine(double budget_mb, const Edit& edit = {}) {
  return engine_on(render::BackendKind::kEmbree, budget_mb, edit);
}

struct Run {
  std::vector<OutEvent> events;
  uint32_t batches = 0;     // the most pixel batches of a window
  uint64_t samples = 0;     // the most pixel-samples of a window
  bool capped = false;      // a window's levels were clamped
  std::vector<size_t> split;  // the events released by each step and by the flush
};

// start, three steps and flush.
Run run(double budget_mb, const Edit& edit = {}) {
  const std::unique_ptr<Engine> e = host_engine(budget_mb, edit);
  CHECK_FALSE(e->on_device());
  CameraEngine& cam = e->camera(0);
  CHECK(cam.sensor().device() == sensor::Device::kCpu);
  cam.start(kT0, 6 * kW, false);
  Run out;
  const auto append = [&](const std::vector<OutEvent>& ready) {
    out.split.push_back(ready.size());
    out.events.insert(out.events.end(), ready.begin(), ready.end());
  };
  for (int w = 0; w < 3; ++w) {
    StepResult r = cam.step(kT0 + w * kW);
    CHECK(r.planned);
    out.batches = std::max(out.batches, r.window.stats.batches);
    out.samples = std::max(out.samples, r.window.pixel_samples);
    out.capped |= r.level_cap.has_value();
    append(r.ready);
  }
  append(cam.flush());
  return out;
}

// A run with the most bytes every Buffer of the host space held during it (the engine's buffers: BufferMeter).
std::pair<Run, size_t> metered(double budget_mb, const Edit& edit = {}) {
  using Meter = BufferMeter<Space::kHost>;
  const size_t base = Meter::live.load();
  Meter::reset_peak();
  Run r = run(budget_mb, edit);
  return {std::move(r), Meter::peak.load() - base};
}

// Output order: t_reported, then (x, y).
bool ordered(const std::vector<OutEvent>& a) {
  bool ok = true;
  for (size_t i = 1; i < a.size(); ++i) {
    ok &= std::make_tuple(a[i - 1].t_reported, a[i - 1].x, a[i - 1].y) <= std::make_tuple(a[i].t_reported, a[i].x, a[i].y);
  }
  return ok;
}

bool same(const std::vector<OutEvent>& a, const std::vector<OutEvent>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    const OutEvent &x = a[i], &y = b[i];
    if (x.t_reported != y.t_reported || x.t_cross != y.t_cross || x.x != y.x || x.y != y.y || x.p != y.p ||
        x.label != y.label || x.flags != y.flags || x.instance != y.instance || x.semantic != y.semantic ||
        std::memcmp(&x.depth, &y.depth, 3 * sizeof(float)) != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

TEST_CASE("the host engine runs, reruns bit for bit and batches exactly") {
  const Run whole = run(0.0);
  const std::vector<OutEvent>& a = whole.events;
  CHECK(whole.batches == 1);
  REQUIRE(a.size() > 50);
  bool box = false;
  for (const OutEvent& ev : a) box |= ev.instance == 2;
  CHECK(ordered(a));
  CHECK(box);
  const auto [again, peak] = metered(0.0);
  CHECK(same(again.events, a));
  // Half the unbatched run's peak: room for the windows' runs, not for their whole footprint (pixel batches). The budget
  // bounds every buffer of the engine, the planner's and the output stage's included (ENGINE_CPP I3 open item (d)).
  const size_t budget = peak / 2;
  const auto [parts, parts_peak] = metered(static_cast<double>(budget) / 1048576.0);
  CHECK(parts.batches > 1);
  CHECK(parts_peak <= budget);
  CHECK(same(parts.events, a));
  CHECK_THROWS_AS(run(0.0005), MemoryBudgetError);
}

TEST_CASE("the output stage's growth counts in the budget: within it, or MemoryBudgetError before allocating") {
  // Every pixel a leak-type hot pixel at 300 Hz: about 2.5 events per pixel and window, so that the output stage's
  // buffers (O(events)) take most of the peak. At every budget the run writes the unbatched events within it
  // (the WindowEngine's buffers are released first when the stage's growth does not fit beside them) or raises.
  const Edit hot = [](EngineConfig&, SensorSpec& s) {
    s.params.hot_fraction = 1.0;
    s.params.hot_leak_fraction = 1.0;
    s.params.hot_rate_min_hz = s.params.hot_rate_max_hz = 300.0;
  };
  const auto [whole, peak] = metered(0.0, hot);
  REQUIRE(whole.events.size() > 5000);
  int completed = 0, raised = 0;
  for (const double f : {0.9, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2}) {
    const auto budget = static_cast<size_t>(f * static_cast<double>(peak));
    INFO("budget " << f << " of the peak");
    try {
      const auto [r, p] = metered(static_cast<double>(budget) / 1048576.0, hot);
      CHECK(p <= budget);
      CHECK(same(r.events, whole.events));
      ++completed;
    } catch (const MemoryBudgetError&) {
      ++raised;
    }
  }
  CHECK(completed > 0);
  CHECK(raised > 0);
}

TEST_CASE("the host engine refuses misuse and the device renderer") {
  const std::unique_ptr<Engine> e = host_engine(0.0);
  CHECK_THROWS_AS(e->camera(0).step(kT0), std::logic_error);
  CHECK_THROWS_AS(e->camera(1), std::invalid_argument);
  EngineConfig c;
  c.render = options(render::BackendKind::kOptix, 4);
  c.light_groups = {sensor::make_constant_waveform()};
  if (!BALLA_ENGINE_HAVE_DEVICE) CHECK_THROWS_AS(make_engine(box_scene(), c, {}), std::invalid_argument);
  // An explicit budget beyond size_t is no limit (its conversion was undefined: 0 bytes, every window refused).
  CHECK(host_engine(1.0)->budget_bytes() == 1048576);
  CHECK(host_engine(1e14)->budget_bytes() == std::numeric_limits<size_t>::max());
}

TEST_CASE("the host engine: a window after a failed one reports only its own render times and calls") {
  // 0.265 MB: about 0.095 MiB for the samples beside the engine's 0.158 MiB (with P2-D D1's plan-pass normals and plan
  // bits); one pixel alone needs about 0.105 MiB once batches have rendered (0.245 to 0.29 MB do).
  const std::unique_ptr<Engine> e = host_engine(0.265);
  CameraEngine& cam = e->camera(0);
  cam.start(kT0, 6 * kW, false);
  CHECK_THROWS_WITH_AS(cam.step(kT0), doctest::Contains("alone needs"), MemoryBudgetError);
  const WindowResult w = cam.window(kT0, nullptr, nullptr, nullptr, 0);
  CHECK(w.stats.render_calls == 0);
  CHECK(w.render_ms_by_call.empty());  // one entry per render call
  CHECK(w.render.trace_ms == 0.0);
  CHECK(w.render.shade_ms == 0.0);
  CHECK(w.render.host_ms == 0.0);
}

TEST_CASE("the host engine checks its cancel token in start and step before their work (BL-0b)") {
  const Run plain = run(0.0);
  // An unset token changes no event.
  CancelToken unset, set;
  set.set();
  const std::unique_ptr<Engine> a = host_engine(0.0), b = host_engine(0.0);
  CameraEngine& cam = a->camera(0);
  cam.set_cancel(&unset);
  cam.start(kT0, 6 * kW, false);
  std::vector<OutEvent> events;
  for (int w = 0; w < 3; ++w) {
    const StepResult r = cam.step(kT0 + w * kW);
    events.insert(events.end(), r.ready.begin(), r.ready.end());
  }
  const std::vector<OutEvent> held = cam.flush();
  events.insert(events.end(), held.begin(), held.end());
  CHECK(same(events, plain.events));
  // A set token: start and step throw Cancelled at their entry, the sensor where it was.
  CameraEngine& c = b->camera(0);
  c.set_cancel(&set);
  CHECK_THROWS_WITH_AS(c.start(kT0, 6 * kW, false), "start: cancelled", Cancelled);
  c.set_cancel(nullptr);
  c.start(kT0, 6 * kW, false);
  c.set_cancel(&set);
  CHECK_THROWS_WITH_AS(c.step(kT0), "step: cancelled", Cancelled);
  CHECK(c.sensor().time_ns() == kT0);
}

// A motion warm-up the bundle covers only in part (one of its three windows) is completed by a static one: the
// carried samples at t_s and the runs from t_s on are those of a static start. Full and no coverage, on both device
// pairs: tests/python/test_engine_run.py.
TEST_CASE("the host engine's motion warm-up: the windows the bundle covers, then a static start's runs") {
  const std::unique_ptr<Engine> still = host_engine(0.0), moved = host_engine(0.0);
  CameraEngine &a = still->camera(0), &b = moved->camera(0);
  CHECK(a.start(kT0, 6 * kW, false).motion_windows == 0);
  CHECK(b.start(kT0, 3 * kW, true).motion_windows == 1);  // [kT0 - kW, kT0); the static part to 50 ms in all
  CHECK(b.sensor().time_ns() == kT0);
  CHECK(a.runs_digest().first == b.runs_digest().first);
  a.step(kT0);
  b.step(kT0);
  CHECK(a.runs_digest().second == b.runs_digest().second);
}

TEST_CASE("the host engine: no level clamp (D9) and the output cut on a drifting sensor clock") {
  const Run base = run(0.0);
  REQUIRE(base.events.size() > 50);
  CHECK_FALSE(base.capped);  // the default clamp (2^25) is far above this scene's windows
  // Without a clamp the level cap is skipped: the windows of a clamp never reached.
  const Run unclamped = run(0.0, [](EngineConfig& c, SensorSpec&) { c.max_samples_per_window.reset(); });
  CHECK(same(unclamped.events, base.events));
  CHECK_FALSE(unclamped.capped);
  CHECK_THROWS_AS(run(0.0, [](EngineConfig& c, SensorSpec&) { c.max_samples_per_window = 1; }), std::invalid_argument);

  // A slow sensor clock (-0.5 %: about -5 ms at t = 1 s, beyond d0 = 300 us). The cut is the window end on the
  // sensor clock, so each step releases the events of the drift-free run (a cut at t_w + W + offset released every
  // event at once, also those a later window's events may tie with), in output order, with the same crossings.
  const Run slow = run(0.0, [](EngineConfig&, SensorSpec& s) { s.params.clock_drift_ppm = -5000.0; });
  CHECK(slow.split == base.split);
  CHECK(base.split.back() > 0);  // the held events of the last window
  CHECK(ordered(slow.events));
  const auto crossings = [](const std::vector<OutEvent>& ev) {
    std::vector<std::tuple<int64_t, uint16_t, uint16_t, uint8_t>> out;
    for (const OutEvent& e : ev) out.emplace_back(e.t_cross, e.x, e.y, e.p);
    std::sort(out.begin(), out.end());
    return out;
  };
  CHECK(crossings(slow.events) == crossings(base.events));
}

// P2-B B2, the band's per-pixel N: N_base 16 and N_band 64 on the box scene, the windows' N forced per pixel.
namespace {

std::vector<render::BackendKind> engine_backends() {
  std::vector<render::BackendKind> b{render::BackendKind::kEmbree};
  if (device_available()) b.push_back(render::BackendKind::kOptix);
  return b;
}

Edit at_n(uint32_t n_sub, uint32_t n_band, const Edit& more = {}) {
  return [=](EngineConfig& c, SensorSpec& s) {
    c.render.scene.n_subsamples = n_sub;
    c.n_band = n_band;
    if (more) more(c, s);
  };
}

// A seeded map of log2 N per pixel of the 40 x 30 camera: log2 16 or log2 64.
std::vector<uint8_t> random_map(uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> m(40 * 30);
  for (uint8_t& v : m) v = rng() % 2 ? 6 : 4;
  return m;
}

// The box scene with a box that stops at t_stop (its controls equal from there on: held from its first window with
// equal controls, I3b) and a small box that moves on across the big box's final position.
render::SceneBundle stopping_scene(int64_t t_stop) {
  render::SceneBundle s = box_scene();
  const int64_t t_begin = kT0 - kW, t_end = kT0 + 8 * kW;
  const double stop_s = 1e-9 * static_cast<double>(t_stop - (t_begin - 3 * kKnotNs));  // make_trajectory's t0
  s.trajectories[1] = make_trajectory(t_begin, t_end, [&](double sec, Quat<double>* q, Vec3<double>* p) {
    const double x = std::min(sec, stop_s);
    *q = so3::exp(x * Vec3<double>{0.0, 0.0, 2.0}) * quat_identity<double>();
    *p = Vec3<double>{-0.2, 0.0, 1.0} + x * Vec3<double>{0.6, 0.0, 0.0};
  });
  s.trajectories.push_back(moving_trajectory(t_begin, t_end, quat_identity<double>(), {-0.25, 0.03, 0.9},
                                             {1.5, 0.0, 0.0}, {0.0, 0.0, 0.0}));
  render::Body small = make_body(3, box_mesh(0.03f, 0.03f, 0.03f), {0.6f, 0.6f, 0.6f});
  small.trajectory = 2;
  s.bodies.push_back(small);
  return s;
}

// Active pixel i's times and irradiance in a copy of a window's runs.
std::vector<int64_t> run_t(const WindowRuns& r, size_t i) {
  return std::vector<int64_t>(r.t.begin() + r.offsets[i], r.t.begin() + r.offsets[i + 1]);
}
std::vector<float> run_irr(const WindowRuns& r, size_t i) {
  const size_t g = r.t.empty() ? 1 : r.irr.size() / r.t.size();
  return std::vector<float>(r.irr.begin() + static_cast<std::ptrdiff_t>(g * static_cast<size_t>(r.offsets[i])),
                            r.irr.begin() + static_cast<std::ptrdiff_t>(g * static_cast<size_t>(r.offsets[i + 1])));
}
bool same_runs(const WindowRuns& a, const WindowRuns& b) {
  return a.pixel == b.pixel && a.offsets == b.offsets && a.t == b.t && a.irr == b.irr && a.n_log2 == b.n_log2;
}

// start, four steps with the maps of seeds 0-3 forced, flush: each window's runs and the events.
struct BandRun {
  std::vector<WindowRuns> runs;
  std::vector<StepResult> steps;
  std::vector<OutEvent> events;
  size_t peak = 0;  // the most bytes the host space's Buffers held (BufferMeter)
};
BandRun band_run(render::BackendKind backend, double budget_mb, const Edit& edit) {
  using Meter = BufferMeter<Space::kHost>;
  const size_t live = Meter::live.load();
  Meter::reset_peak();
  const std::unique_ptr<Engine> e = engine_on(backend, budget_mb, at_n(16, 64, edit));
  CameraEngine& cam = e->camera(0);
  cam.start(kT0, 6 * kW, false);
  BandRun out;
  for (uint32_t w = 0; w < 4; ++w) {
    cam.force_n(random_map(w));
    out.steps.push_back(cam.step(kT0 + w * kW));
    out.events.insert(out.events.end(), out.steps.back().ready.begin(), out.steps.back().ready.end());
    out.runs.push_back(cam.last_runs());
  }
  const std::vector<OutEvent> rest = cam.flush();
  out.events.insert(out.events.end(), rest.begin(), rest.end());
  out.peak = Meter::peak.load() - live;
  return out;
}

}  // namespace

TEST_CASE("the band's per-pixel N (P2-B B2): every pixel's runs are a uniform run's at its N, on both engines") {
  // A pixel's window renders at its N; a pixel whose carried sample has the other N (the start frame is at N_band)
  // re-renders its t_w node first, so its runs are those of the uniform run at its N, window by window. P2-B B4b: a
  // pixel at N_base with an edge sample is escalated, its window sampled again at N_band (and its runs then the
  // uniform N_band run's).
  for (const render::BackendKind backend : engine_backends()) {
    INFO((backend == render::BackendKind::kOptix ? "device" : "host"));
    const BandRun mixed = band_run(backend, 0.0, {});
    const std::unique_ptr<Engine> lo = engine_on(backend, 0.0, at_n(16, 0)), hi = engine_on(backend, 0.0, at_n(64, 0));
    CameraEngine &a = lo->camera(0), &b = hi->camera(0);
    a.start(kT0, 6 * kW, false);
    b.start(kT0, 6 * kW, false);
    uint64_t switches = 0, up = 0, escalations = 0;
    for (uint32_t w = 0; w < 4; ++w) {
      INFO("window " << w);
      const std::vector<uint8_t> map = random_map(w);
      const StepResult sa = a.step(kT0 + w * kW);
      CHECK_FALSE(sa.window.band.on);
      CHECK(sa.window.stats.render_calls == sa.window.render_ms_by_call.size());
      b.step(kT0 + w * kW);
      const WindowRuns &m = mixed.runs[w], ra = a.last_runs(), rb = b.last_runs();
      REQUIRE(m.pixel.size() > 100);
      REQUIRE(m.pixel == ra.pixel);  // the plan does not depend on N
      REQUIRE(m.pixel == rb.pixel);
      REQUIRE(m.n_log2.size() == m.pixel.size());
      CHECK(ra.n_log2.empty());
      uint64_t band_pixels = 0, band_samples = 0, escalated = 0, escalated_samples = 0;
      for (size_t i = 0; i < m.pixel.size(); ++i) {
        const uint8_t forced = map[m.pixel[i]], n = m.n_log2[i];
        CHECK((n == forced || (forced == 4 && n == 6)));
        const WindowRuns& r = n == 6 ? rb : ra;
        CHECK(run_t(m, i) == run_t(r, i));
        CHECK(run_irr(m, i) == run_irr(r, i));
        const auto len = static_cast<uint64_t>(m.offsets[i + 1] - m.offsets[i]);
        band_pixels += forced == 6 ? 1 : 0;
        band_samples += forced == 6 ? len - 1 : 0;
        escalated += n != forced ? 1 : 0;
        escalated_samples += n != forced ? len : 0;  // the t_w node and the pass's samples
      }
      const WindowResult& s = mixed.steps[w].window;
      CHECK(s.band.on);
      // A call per N class of each round, the switches' before them, the escalation's after: each timed
      // (render_ms_by_call).
      CHECK(s.stats.render_calls == s.render_ms_by_call.size());
      CHECK(s.stats.render_calls > s.stats.samples_by_round.size());
      // P2-C C2 (round_times): the calls split into the switches', the pass's rounds and the escalation's.
      const RoundTimes rt = round_times(s);
      CHECK(rt.ok);
      CHECK(rt.pass.size() == s.stats.samples_by_round.size());
      CHECK(rt.escalation.size() == s.escalation.samples_by_round.size());
      CHECK(s.escalation.samples_by_round.empty() == (s.band.escalated == 0));
      double all = 0.0, split = rt.switches_ms + rt.escalation_nodes_ms + rt.redone_ms;
      for (const double v : s.render_ms_by_call) all += v;
      for (const double v : rt.pass) split += v;
      for (const double v : rt.escalation) split += v;
      CHECK(split == doctest::Approx(all));
      CHECK(rt.redone_ms == 0.0);
      CHECK(s.stats.call_round.size() + s.escalation.call_round.size() + (s.band.escalated > 0 ? 1 : 0) <=
            s.render_ms_by_call.size());
      CHECK(s.band.band_pixels == band_pixels);
      CHECK(s.stats.band_samples == band_samples);
      CHECK(s.band.escalated == escalated);
      // Every edge pixel at N_base and (P2-B B6a, pooled) every N_base pixel beside an edge pixel; the forced map
      // overrides the border ring.
      CHECK(s.band.escalated == s.band.base_edge + s.band.escalated_pooled);
      CHECK(s.band.border == 0);
      CHECK(s.band.escalated_samples == escalated_samples);
      CHECK(band_pixels > 0);
      CHECK(band_pixels < m.pixel.size());
      uint64_t binned = 0;
      for (const uint64_t v : s.band.eta) binned += v;
      CHECK(binned == s.band.switches);
      CHECK(s.band.unquiescent <= s.band.switches - s.band.eta[0]);
      switches += s.band.switches;
      up += s.band.up;
      escalations += escalated;
    }
    CHECK(up > 0);            // back up to N_band
    CHECK(switches > up);     // down from the start frame's N_band
    CHECK(escalations > 0);   // the box's silhouettes and shadow edges at N_base
  }
}

TEST_CASE("the band's per-pixel N: pixel batches, launch sizes, render threads and window cuts give the same runs") {
  for (const render::BackendKind backend : engine_backends()) {
    INFO((backend == render::BackendKind::kOptix ? "device" : "host"));
    const bool host = backend == render::BackendKind::kEmbree;
    const BandRun base = band_run(backend, 0.0, {});
    REQUIRE(base.events.size() > 50);
    std::vector<BandRun> splits;
    // Half the unbatched host run's peak (the windows' runs fit, their footprint does not).
    if (host) splits.push_back(band_run(backend, static_cast<double>(base.peak / 2) / 1048576.0, {}));
    splits.push_back(band_run(backend, 0.0, [](EngineConfig& c, SensorSpec&) { c.render.max_rays_per_launch = 1 << 12; }));
    if (host) splits.push_back(band_run(backend, 0.0, [](EngineConfig& c, SensorSpec&) { c.render.num_threads = 1; }));
    for (size_t k = 0; k < splits.size(); ++k) {
      INFO("split " << k);
      const BandRun& r = splits[k];
      for (size_t w = 0; w < 4; ++w) {
        CHECK_FALSE(r.runs[w].pixel.empty());
        CHECK(same_runs(r.runs[w], base.runs[w]));
      }
      CHECK(same(r.events, base.events));
      if (host && k == 0) {
        uint32_t batches = 0;
        for (const StepResult& st : r.steps) batches = std::max(batches, st.window.stats.batches);
        CHECK(batches > 1);
      }
    }
    // window(): a third of the pixels, in reverse rank order, have the runs they have among all of them at the same N.
    // P2-B B6a: pooled escalation reads the active 8-neighbours' edges, so a pixel that the whole escalates by a
    // neighbour outside the third stays at N_base in the third (never the converse: its own and its neighbours' edges
    // are the whole's too).
    const std::unique_ptr<Engine> whole = engine_on(backend, 0.0, at_n(16, 64)), cut = engine_on(backend, 0.0, at_n(16, 64));
    std::vector<uint32_t> pixel(40 * 30), sub;
    std::vector<uint8_t> level(pixel.size()), sub_level;
    std::vector<int64_t> rank(pixel.size()), sub_rank;
    std::mt19937 rng(77);
    for (uint32_t p = 0; p < pixel.size(); ++p) {
      pixel[p] = p;
      level[p] = static_cast<uint8_t>(1 + rng() % 5);
      rank[p] = p;
      if (p % 3 == 1) {
        sub.push_back(p);
        sub_level.push_back(level[p]);
        sub_rank.push_back(static_cast<int64_t>(pixel.size() - 1 - p));
      }
    }
    CameraEngine &cw = whole->camera(0), &cc = cut->camera(0);
    for (CameraEngine* c : {&cw, &cc}) {
      c->force_n(random_map(5));
      c->start(kT0, 6 * kW, false);
    }
    cw.window(kT0, pixel.data(), level.data(), rank.data(), pixel.size());
    const WindowResult rc = cc.window(kT0, sub.data(), sub_level.data(), sub_rank.data(), sub.size());
    CHECK(rc.band.switches > 0);
    const WindowRuns xw = cw.last_runs(), xc = cc.last_runs();
    REQUIRE(xw.pixel == pixel);
    REQUIRE(xc.pixel.size() == sub.size());
    size_t pooled_outside = 0;
    for (size_t i = 0; i < xc.pixel.size(); ++i) {
      const uint32_t p = xc.pixel[i];
      CHECK(p == sub[sub.size() - 1 - i]);  // rank order
      if (xc.n_log2[i] != xw.n_log2[p]) {
        CHECK(xc.n_log2[i] == 4);
        CHECK(xw.n_log2[p] == 6);
        ++pooled_outside;
        continue;
      }
      CHECK(run_t(xc, i) == run_t(xw, p));
      CHECK(run_irr(xc, i) == run_irr(xw, p));
    }
    CHECK(pooled_outside < xc.pixel.size() / 8);
  }
}

TEST_CASE("the band's per-pixel N: a motion warm-up leaves a static start's frame and runs") {
  // The motion window switches pixels to N_base; the frame at t_s is at N_band again, as a static start's.
  const std::unique_ptr<Engine> still = host_engine(0.0, at_n(16, 64)), moved = host_engine(0.0, at_n(16, 64));
  CameraEngine &a = still->camera(0), &b = moved->camera(0);
  for (CameraEngine* c : {&a, &b}) c->force_n(random_map(11));
  CHECK(a.start(kT0, 6 * kW, false).motion_windows == 0);
  CHECK(b.start(kT0, 3 * kW, true).motion_windows == 1);
  CHECK(a.runs_digest().first == b.runs_digest().first);
  const StepResult ra = a.step(kT0), rb = b.step(kT0);
  CHECK(a.runs_digest().second == b.runs_digest().second);
  CHECK(ra.window.band.switches == rb.window.band.switches);
  CHECK(ra.window.band.switches > 0);
}

TEST_CASE("the band's per-pixel N: the digests hash the carried N; the hooks and the engine refuse misuse") {
  // All pixels forced to N_band = 64: the runs of a uniform run at 64, but digests of their own (the carried N).
  const std::unique_ptr<Engine> band = host_engine(0.0, at_n(16, 64)), again = host_engine(0.0, at_n(16, 64)),
                                uniform = host_engine(0.0, at_n(64, 0));
  CameraEngine &m = band->camera(0), &m2 = again->camera(0), &u = uniform->camera(0);
  for (CameraEngine* c : {&m, &m2}) c->force_n(std::vector<uint8_t>(40 * 30, 6));
  for (CameraEngine* c : {&m, &m2, &u}) c->start(kT0, 6 * kW, false);
  const StepResult sm = m.step(kT0);
  m2.step(kT0);
  u.step(kT0);
  WindowRuns all_band = m.last_runs();
  CHECK(all_band.n_log2 == std::vector<uint8_t>(all_band.pixel.size(), 6));
  all_band.n_log2.clear();  // the uniform run has one N
  CHECK(same_runs(all_band, u.last_runs()));
  CHECK(sm.window.band.switches == 0);
  CHECK(sm.window.band.band_pixels == m.last_runs().pixel.size());
  CHECK(m.runs_digest() == m2.runs_digest());
  CHECK(m.runs_digest().first != u.runs_digest().first);
  CHECK(m.runs_digest().second != u.runs_digest().second);
  // Back to the band's own choice (P2-B B4a's tile map): its band tiles' pixels at N_band, the rest at N_base.
  m.force_n(random_map(3));
  CHECK(m.step(kT0 + kW).window.band.switches > 0);
  m.force_n({});
  const StepResult back = m.step(kT0 + 2 * kW);
  CHECK(back.plan.seeds);
  CHECK(back.plan.band_tiles > 0);
  CHECK(back.window.band.band_pixels > 0);
  CHECK(back.window.band.band_pixels <= m.last_runs().pixel.size());
  // Misuse.
  CHECK_THROWS_AS(u.force_n(random_map(1)), std::logic_error);
  CHECK_THROWS_AS(m.force_n(std::vector<uint8_t>(10, 6)), std::invalid_argument);
  std::vector<uint8_t> five(40 * 30, 4);
  five[7] = 5;
  CHECK_THROWS_AS(m.force_n(five), std::invalid_argument);
  for (const auto& [n_sub, n_band] : {std::pair<uint32_t, uint32_t>{16, 48}, {16, 16}, {64, 32}, {24, 64}, {16, 2048}}) {
    INFO(n_sub << " / " << n_band);
    CHECK_THROWS_AS(host_engine(0.0, at_n(n_sub, n_band)), std::invalid_argument);
  }
}

TEST_CASE("the band's per-pixel N: a switch differs from the uniform run only in a window whose held set was rebuilt") {
  // The big box stops while the small one moves on. In the window that rebuilds the held set (the big box joins the
  // world, I3b) a switch re-renders its t_w node under the new set, while the uniform runs carry nodes rendered under
  // the previous one, and a held body's hits differ from a moving one's in the last bits: only switched pixels of
  // that window may differ (ENGINE_CPP §0.1 "P2-B B2 notes"), and escalated ones (P2-B B4b: their t_w nodes are
  // re-rendered so too). Without the held world every window is exact.
  for (const render::BackendKind backend : engine_backends()) {
    for (const bool held : {false, true}) {
      INFO((backend == render::BackendKind::kOptix ? "device" : "host") << ", held_world " << held);
      const auto world = [held](EngineConfig& c, SensorSpec&) { c.held_world = held; };
      const auto engine = [&](uint32_t n_sub, uint32_t n_band) {
        return engine_on(backend, 0.0, at_n(n_sub, n_band, world), stopping_scene(kT0 + kW));
      };
      const std::unique_ptr<Engine> mixed = engine(16, 64), lo = engine(16, 0), hi = engine(64, 0);
      CameraEngine &m = mixed->camera(0), &a = lo->camera(0), &b = hi->camera(0);
      for (CameraEngine* c : {&m, &a, &b}) c->start(kT0, 6 * kW, false);
      std::vector<uint8_t> carried(40 * 30, 6);  // the start frame's N
      size_t rebuilds = 0, differ_rebuilt = 0;
      for (uint32_t w = 0; w < 5; ++w) {
        INFO("window " << w);
        const std::vector<uint8_t> map = random_map(w);
        m.force_n(map);
        const StepResult sm = m.step(kT0 + w * kW), sa = a.step(kT0 + w * kW);
        b.step(kT0 + w * kW);
        const bool rebuilt = sm.window.held.rebuilt;
        CHECK(sa.window.held.rebuilt == rebuilt);
        rebuilds += rebuilt ? 1 : 0;
        const WindowRuns rm = m.last_runs(), ra = a.last_runs(), rb = b.last_runs();
        REQUIRE(rm.pixel.size() > 100);
        REQUIRE(rm.pixel == ra.pixel);
        REQUIRE(rm.pixel == rb.pixel);
        size_t differ = 0, switched = 0;
        for (size_t i = 0; i < rm.pixel.size(); ++i) {
          const uint32_t p = rm.pixel[i];
          const uint8_t n = rm.n_log2[i];  // map[p], or 6 for an escalated pixel
          const WindowRuns& r = n == 6 ? rb : ra;
          if (run_t(rm, i) != run_t(r, i) || run_irr(rm, i) != run_irr(r, i)) {
            ++differ;
            switched += carried[p] != map[p] || n != map[p] ? 1 : 0;
          }
          carried[p] = n;
        }
        CHECK(switched == differ);
        if (!rebuilt) CHECK(differ == 0);
        differ_rebuilt += rebuilt ? differ : 0;
      }
      CHECK(rebuilds == (held ? 1u : 0u));
      if (held) MESSAGE("switched pixels that differ at the rebuild: " << differ_rebuilt);
    }
  }
}

TEST_CASE("the band's per-pixel N: last_runs and runs_digest give a camera's own window, or none") {
  // Two cameras share the engine's sampling buffers: a camera's hooks see its runs until a window of either begins.
  render::SceneBundle two = box_scene();
  two.cameras.push_back(pinhole(20, 15, 20.0, 0));
  const std::unique_ptr<Engine> e = engine_on(render::BackendKind::kEmbree, 0.0, at_n(16, 64), two);
  CameraEngine &a = e->camera(0), &d = e->camera(1);
  CHECK_NOTHROW(a.runs_digest());  // before start(): no carried N to hash yet
  CHECK(a.last_runs().pixel.empty());
  for (CameraEngine* c : {&a, &d}) c->start(kT0, 6 * kW, false);
  a.step(kT0);
  const WindowRuns own = a.last_runs();
  const std::pair<uint64_t, uint64_t> digest = a.runs_digest();
  REQUIRE(own.pixel.size() > 100);
  CHECK(own.offsets.back() == static_cast<int64_t>(own.t.size()));
  d.step(kT0);
  const WindowRuns other = d.last_runs();
  REQUIRE_FALSE(other.pixel.empty());
  CHECK(other.offsets.back() == static_cast<int64_t>(other.t.size()));
  CHECK(a.last_runs().pixel.empty());
  CHECK(a.runs_digest().first == digest.first);
  CHECK(a.runs_digest().second != digest.second);
  a.step(kT0 + kW);
  CHECK(d.last_runs().pixel.empty());
  CHECK_FALSE(a.last_runs().pixel.empty());
}

TEST_CASE("the band's prediction (P2-B B4a): tile levels unchanged, its tiles at N_band, edge pixels counted") {
  // The seeds and their tile map are written beside the planner's rates: with the band the plan (levels, static tiles,
  // v_hat, the shadow pixels) and the active pixels are the band-off run's, window by window. The band's pixels are
  // those of its tiles (on this 40 x 30 image every dynamic tile: the box's and its shadow's reach); its edge
  // counters split the window's edge pixels by N, the same on both engines.
  std::vector<std::vector<uint64_t>> by_backend;
  for (const render::BackendKind backend : engine_backends()) {
    INFO((backend == render::BackendKind::kOptix ? "device" : "host"));
    const std::unique_ptr<Engine> band = engine_on(backend, 0.0, at_n(16, 64));
    const std::unique_ptr<Engine> off = engine_on(backend, 0.0, at_n(16, 0));
    CameraEngine &b = band->camera(0), &o = off->camera(0);
    b.start(kT0, 6 * kW, false);
    o.start(kT0, 6 * kW, false);
    std::vector<uint64_t> counts;
    uint64_t edges = 0, band_edges = 0, overfill = 0;
    for (uint32_t w = 0; w < 4; ++w) {
      INFO("window " << w);
      const StepResult sb = b.step(kT0 + w * kW), so = o.step(kT0 + w * kW);
      CHECK(sb.plan.seeds);
      CHECK_FALSE(so.plan.seeds);
      CHECK(sb.plan.plan_pass);
      CHECK(sb.plan.levels == so.plan.levels);
      CHECK(sb.plan.tiles_static == so.plan.tiles_static);
      CHECK(sb.plan.tiles_dynamic == so.plan.tiles_dynamic);
      CHECK(sb.plan.shadow_pixels == so.plan.shadow_pixels);
      CHECK(sb.plan.v_hat_max == so.plan.v_hat_max);
      const WindowRuns rb = b.last_runs(), ro = o.last_runs();
      REQUIRE(rb.pixel.size() > 100);
      CHECK(rb.pixel == ro.pixel);
      const SwitchStats& s = sb.window.band;
      CHECK(sb.plan.band_tiles > 0);
      CHECK(sb.plan.band_tiles <= sb.plan.tiles_dynamic);
      CHECK(sb.plan.seed_pixels > 0);
      CHECK(s.band_pixels > 0);
      CHECK(s.band_pixels <= 64 * sb.plan.band_tiles);
      CHECK(s.band_edge <= s.band_pixels);
      CHECK(s.base_edge <= rb.pixel.size() - s.band_pixels);
      edges += s.band_edge + s.base_edge;
      band_edges += s.band_edge;
      overfill += s.band_pixels - s.band_edge;
      // The plan's and the class words' counts (the renderers agree on those; refinement, and so band_samples, may
      // differ in a sample where their values differ in the last bits).
      counts.insert(counts.end(), {s.band_pixels, s.band_edge, s.base_edge, s.switches, s.up, sb.plan.band_tiles,
                                   sb.plan.seed_pixels, sb.plan.carried_seeds});
    }
    CHECK(edges > 0);
    CHECK(band_edges == edges);  // the box's silhouettes and shadow edges are all predicted (none left to escalate)
    MESSAGE("edge pixels " << edges << ", at N_band " << band_edges << "; overfill " << overfill);
    by_backend.push_back(std::move(counts));
  }
  if (by_backend.size() == 2) CHECK(by_backend[0] == by_backend[1]);  // host = device
}

TEST_CASE("the band's edge counters split a window's edge pixels by N") {
  // Every active pixel forced to one N: its edge pixels all count at that N.
  for (const uint8_t n : {uint8_t{4}, uint8_t{6}}) {
    const std::unique_ptr<Engine> e = host_engine(0.0, at_n(16, 64));
    CameraEngine& c = e->camera(0);
    c.force_n(std::vector<uint8_t>(40 * 30, n));
    c.start(kT0, 6 * kW, false);
    const StepResult r = c.step(kT0);
    const SwitchStats& s = r.window.band;
    CHECK((n == 6 ? s.band_edge : s.base_edge) > 0);
    CHECK((n == 6 ? s.base_edge : s.band_edge) == 0);
    CHECK(s.band_pixels == (n == 6 ? c.last_runs().pixel.size() : 0u));
  }
}

TEST_CASE("the band's seeds inside a crop: the start frame's carried words outside it do not seed") {
  // The box scene cropped to its right part (x >= 24), away from the box and its shadow: outside the crop the carried
  // samples stay the start frame's, so neither they nor the plan-pass seeds there count, while the uncropped run
  // counts the box's (both engines).
  for (const render::BackendKind backend : engine_backends()) {
    INFO((backend == render::BackendKind::kOptix ? "device" : "host"));
    const std::unique_ptr<Engine> cut =
        engine_on(backend, 0.0, at_n(16, 64, [](EngineConfig& c, SensorSpec&) { c.crop = Crop{24, 0, 40, 30}; }));
    const std::unique_ptr<Engine> whole = engine_on(backend, 0.0, at_n(16, 64));
    CameraEngine &a = cut->camera(0), &b = whole->camera(0);
    a.start(kT0, 6 * kW, false);
    b.start(kT0, 6 * kW, false);
    for (uint32_t w = 0; w < 3; ++w) {
      INFO("window " << w);
      const StepResult ra = a.step(kT0 + w * kW), rb = b.step(kT0 + w * kW);
      CHECK(ra.plan.seeds);
      CHECK(ra.plan.seed_pixels == 0);
      CHECK(ra.plan.carried_seeds == 0);
      CHECK(rb.plan.seed_pixels > 0);
      CHECK(rb.plan.carried_seeds > 0);
    }
  }
}

// P2-B B4b, T2: the band (N_base 64, N_band 256, its own prediction and escalation) against the uniform N_band run.
namespace {

// The scene on a 96 x 72 camera of the same field of view (f = 96 px), whose tiles (12 x 9) the band's seeds do not all
// reach.
render::SceneBundle wide(render::SceneBundle s) {
  s.cameras[0] = pinhole(96, 72, 96.0, 0);
  return s;
}

// The box scene with its box (half-size 0.05 m) entering the image from the left at 12 m/s (about 1150 px/s at 1 m) in
// the run's first window (at x = -0.6 m at kT0, the image's edge at -0.5 m): no plan pass sees it at that window's
// start, so nothing predicts its pixels, which escalate. y: the box centre's height (m).
render::SceneBundle birth_scene(double y = -0.1) {
  render::SceneBundle s = wide(box_scene());
  const double at_t0 = 1e-9 * static_cast<double>(kW + 3 * kKnotNs);  // kT0 on the trajectory's clock (knot 0 at 0)
  s.trajectories[1] = moving_trajectory(kT0 - kW, kT0 + 8 * kW, quat_identity<double>(),
                                        {-0.6 - 12.0 * at_t0, y, 1.0}, {12.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
  s.bodies[1].mesh = box_mesh(0.05f, 0.05f, 0.05f);
  return s;
}

// P2-B B6b: the box held throughout (a static trajectory) and a small box moving far from its shadow, under a camera
// that turns slowly (every tile dynamic), on the 96 x 72 camera.
render::SceneBundle held_scene() {
  render::SceneBundle s = wide(box_scene());
  const int64_t t_begin = kT0 - kW, t_end = kT0 + 8 * kW;
  s.trajectories[0] = moving_trajectory(t_begin, t_end, quat_identity<double>(), {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0},
                                        {0.0, 0.05, 0.0});
  s.trajectories[1] = static_trajectory(t_begin, t_end, quat_identity<double>(), {-0.2, 0.0, 1.0});
  s.trajectories.push_back(moving_trajectory(t_begin, t_end, quat_identity<double>(), {0.4, -0.3, 1.0},
                                             {0.0, 0.3, 0.0}, {0.0, 0.0, 0.0}));
  render::Body small = make_body(3, box_mesh(0.03f, 0.03f, 0.03f), {0.6f, 0.6f, 0.6f});
  small.trajectory = 2;
  s.bodies.push_back(small);
  return s;
}

// The box scene under 18 shadow-casting point lights (more than the class word's 16 shadow bits) on a 64 x 48 camera.
render::SceneBundle many_lights_scene() {
  render::SceneBundle s = box_scene();
  s.cameras[0] = pinhole(64, 48, 64.0, 0);
  s.lights.clear();
  for (int i = 0; i < 18; ++i) {
    const double a = 0.35 * i;
    s.lights.push_back(point_light({0.5 * std::cos(a), 0.4 * std::sin(a), 0.05 * i}, 0.15f, true));
  }
  s.lights.push_back(ambient_light(0.2f));
  return s;
}

struct T2Run {
  size_t windows = 0, active = 0;
  uint64_t band_pixels = 0, escalated = 0, switches = 0, changed = 0, unquiescent = 0;
  // P2-B B6a: the escalations by pooling alone, the border ring's pixels outside the plan's band, the active pixels on
  // the ring (image or crop) and those of them at N_base.
  uint64_t pooled = 0, border = 0;
  size_t ring = 0, ring_base = 0;
  uint64_t plan = 0;  // P2-B B6c: the active pixels the plan's band (tile or pixel map) puts at N_band
  uint64_t esc_changed = 0, esc_unquiescent = 0;  // the escalated pixels' t_w steps (SwitchStats::escalated_*)
  bool capped = false;
  // Pixel-windows whose runs differ from the uniform run's, and those of them at N_base and not escalated: T2's gap (an
  // unpredicted edge that reaches only sub-samples k >= N_base of the pixel in the window), and those on the ring (P2-B
  // B6a: at N_base only under force_n); whether the events agree.
  size_t differ = 0, gap = 0, differ_ring = 0;
  // Gap nodes: windows whose carried t_w node differs from the uniform run's (a gap run's end node), at N_base (seen)
  // or at N_band after a switch up or an escalation (re-rendered, so counted by changed or esc_changed).
  size_t gap_nodes = 0, gap_nodes_rerendered = 0;
  bool same_events = false;
};

// start, `windows` steps and flush of both engines (fixed64, held_world false: the B2 held-set exception), comparing
// each window's runs and every event; base: every pixel forced to N_base (force_n), so that the band's pixels are the
// escalated ones. Each window's detector counts (changed + esc_changed; esc_changed alone when forced, where the
// switches are down onto the escalated pixels' N_band nodes) equal its re-rendered gap nodes.
T2Run t2_run(render::BackendKind backend, const render::SceneBundle& bundle, uint32_t windows, const Edit& more = {},
             bool base = false, std::optional<Crop> crop = {}) {
  const auto with = [&](uint32_t n_sub, uint32_t n_band) {
    return at_n(n_sub, n_band, [more, crop](EngineConfig& c, SensorSpec& s) {
      c.render.scene.pixel_integral = render::PixelIntegral::kFixed64;
      c.held_world = false;
      c.crop = crop;
      if (more) more(c, s);
    });
  };
  const int32_t width = bundle.cameras[0].width;
  const Crop rect = crop.value_or(Crop{0, 0, width, bundle.cameras[0].height});
  const std::unique_ptr<Engine> band = engine_on(backend, 0.0, with(64, 256), bundle),
                                uniform = engine_on(backend, 0.0, with(256, 0), bundle);
  CameraEngine &b = band->camera(0), &u = uniform->camera(0);
  const size_t n_pixels = static_cast<size_t>(bundle.cameras[0].width) * bundle.cameras[0].height;
  if (base) b.force_n(std::vector<uint8_t>(n_pixels, 6));
  b.start(kT0, 6 * kW, false);
  u.start(kT0, 6 * kW, false);
  T2Run out;
  std::vector<OutEvent> eb, eu;
  std::vector<uint8_t> gap_node(n_pixels, 0);  // the pixel's carried node differs from the uniform run's
  std::vector<uint8_t> carried_n(n_pixels, 8);  // log2 of its N (the start frame's: N_band)
  for (uint32_t w = 0; w < windows; ++w) {
    INFO("window " << w);
    const StepResult sb = b.step(kT0 + w * kW), su = u.step(kT0 + w * kW);
    eb.insert(eb.end(), sb.ready.begin(), sb.ready.end());
    eu.insert(eu.end(), su.ready.begin(), su.ready.end());
    CHECK(sb.plan.levels == su.plan.levels);  // the band leaves the plan alone
    CHECK(sb.level_cap == su.level_cap);
    WindowRuns rb = b.last_runs();
    const WindowRuns ru = u.last_runs();
    REQUIRE(rb.pixel == ru.pixel);
    REQUIRE(rb.n_log2.size() == rb.pixel.size());
    const SwitchStats& s = sb.window.band;
    uint64_t rerendered = 0, escalated_up = 0;
    for (size_t i = 0; i < rb.pixel.size(); ++i) {
      const uint32_t p = rb.pixel[i];
      if (gap_node[p] != 0) {
        ++out.gap_nodes;
        rerendered += rb.n_log2[i] == 8 ? 1 : 0;
      }
      escalated_up += base && rb.n_log2[i] == 8 && carried_n[p] != 8 ? 1 : 0;  // forced: escalated, not switched down
      carried_n[p] = rb.n_log2[i];
      const auto x = static_cast<int32_t>(p % static_cast<uint32_t>(width)),
                 y = static_cast<int32_t>(p / static_cast<uint32_t>(width));
      const bool ring = x == rect.x0 || x + 1 == rect.x1 || y == rect.y0 || y + 1 == rect.y1;
      if (ring) {
        ++out.ring;
        out.ring_base += rb.n_log2[i] == 6 ? 1 : 0;
      }
      const std::vector<float> xb = run_irr(rb, i), xu = run_irr(ru, i);
      const auto g = static_cast<std::ptrdiff_t>(xb.size() / run_t(rb, i).size());
      gap_node[p] = std::equal(xb.end() - g, xb.end(), xu.end() - g) ? 0 : 1;  // the end node, the next carried node
      if (run_t(rb, i) == run_t(ru, i) && xb == xu) continue;
      ++out.differ;
      out.gap += rb.n_log2[i] == 6 ? 1 : 0;  // log2 N_base: neither in the band nor escalated
      out.differ_ring += ring ? 1 : 0;
    }
    out.gap_nodes_rerendered += rerendered;
    CHECK(s.escalated_changed + (base ? 0 : s.changed) == rerendered);
    CHECK(s.escalated_unquiescent <= s.escalated_changed);
    uint64_t esc_bins = 0;  // a step for every escalated pixel but those switched down in the window
    for (const uint64_t v : s.escalated_eta) esc_bins += v;
    CHECK(esc_bins <= s.escalated);
    if (base) CHECK(esc_bins == escalated_up);
    CHECK(s.escalated == s.base_edge + s.escalated_pooled);
    CHECK(s.border <= s.band_pixels);
    if (base) CHECK(s.border == 0);  // force_n overrides the ring
    if (!base) CHECK(s.plan + s.border == s.band_pixels);  // P2-B B6c: N_band is the plan's band or the ring
    out.windows += 1;
    out.active += rb.pixel.size();
    out.band_pixels += s.band_pixels;
    out.escalated += s.escalated;
    out.switches += s.switches;
    out.changed += s.changed;
    out.unquiescent += s.unquiescent;
    out.esc_changed += s.escalated_changed;
    out.esc_unquiescent += s.escalated_unquiescent;
    out.pooled += s.escalated_pooled;
    out.border += s.border;
    out.plan += s.plan;
    out.capped = out.capped || sb.level_cap.has_value();
  }
  const std::vector<OutEvent> rb = b.flush(), ru = u.flush();
  eb.insert(eb.end(), rb.begin(), rb.end());
  eu.insert(eu.end(), ru.begin(), ru.end());
  CHECK(eb.size() > 20);
  out.same_events = same(eb, eu);
  return out;
}

}  // namespace

TEST_CASE("the band equals the uniform N_band run bit for bit (P2-B B4b, T2): movers, 18 lights, clamp, crop, birth") {
  // T2 (ENGINE_CPP §0.1 "P2-B B4b notes"): with fixed64, N_base >= 64 and N_band powers of two, the start frame at
  // N_band, the switches' t_w re-renders and the escalation of every N_base pixel with an edge sample, the band-on run
  // is the uniform N_band run at the same d_px, wherever the band's prediction or the escalation sees the window's
  // edges; the prediction affects speed only. Its two gaps leave an N_base pixel unescalated with an N_band twin that
  // differs: (1) an edge that no seed predicted and that reaches only sub-samples k >= N_base of the pixel (the
  // footprint's tail) in a window, here the entering box's shadow at the end of a window, which P2-B B6a's pooled
  // escalation closes (the edge lies inside a neighbour's 64-point hull, so the neighbour shows it, while those hulls
  // reach past +-0.5 px: the default pixel response, not a narrow one, below) but for slivers, corner grazes and B3's
  // terminators, with the crop's border ring at N_band; (2) a shading-normal terminator inside the pixel (a lit light's
  // f 0 on the 64-prefix, positive later; B3's T1 gap), which these flat boxes do not have.
  // A gap run's end node changes when the next window switches the pixel up (SwitchStats::changed) or escalates it
  // (escalated_changed); t2_run checks both counts window by window against the gap nodes.
  for (const render::BackendKind backend : engine_backends()) {
    INFO((backend == render::BackendKind::kOptix ? "device" : "host"));
    const auto exact = [](const T2Run& r) {
      CHECK(r.differ == 0);
      CHECK(r.same_events);
      CHECK(r.changed == 0);
      CHECK(r.unquiescent == 0);
      CHECK(r.esc_changed == 0);
      CHECK(r.esc_unquiescent == 0);
      CHECK(r.gap_nodes == 0);
      CHECK(r.ring_base == 0);  // P2-B B6a: the border ring at N_band
    };
    {
      INFO("box");
      const T2Run r = t2_run(backend, wide(box_scene()), 4);
      exact(r);
      CHECK(r.band_pixels < r.active);  // N_base pixels too
      CHECK(r.switches > 0);
      CHECK(r.border > 0);
      MESSAGE("box: active " << r.active << ", band " << r.band_pixels << ", escalated " << r.escalated << " (pooled "
                             << r.pooled << "), border " << r.border << ", switches " << r.switches);
    }
    {
      INFO("crop");  // the ring is the crop's
      const T2Run r = t2_run(backend, wide(box_scene()), 3, {}, false, Crop{20, 12, 76, 60});
      exact(r);
      CHECK(r.ring > 0);
      CHECK(r.border > 0);
      MESSAGE("crop: active " << r.active << ", band " << r.band_pixels << ", escalated " << r.escalated
                              << " (pooled " << r.pooled << "), border " << r.border);
    }
    {
      // P2-B B6b: the held box's shadow seeds a finite reach (held_scene), so its receivers are N_base pixels too,
      // exact through the escalation and pooling.
      INFO("held body");
      const T2Run r = t2_run(backend, held_scene(), 3);
      exact(r);
      CHECK(r.band_pixels < r.active);
      MESSAGE("held body: active " << r.active << ", band " << r.band_pixels << ", escalated " << r.escalated
                                   << " (pooled " << r.pooled << "), border " << r.border << ", switches "
                                   << r.switches);
    }
    {
      INFO("18 shadow lights");
      const T2Run r = t2_run(backend, many_lights_scene(), 3);
      exact(r);
      MESSAGE("18 lights: active " << r.active << ", band " << r.band_pixels << ", escalated " << r.escalated);
    }
    {
      INFO("clamp");
      const T2Run r = t2_run(backend, wide(box_scene()), 3,
                             [](EngineConfig& c, SensorSpec&) { c.max_samples_per_window = 96 * 72 * 3; });
      exact(r);
      CHECK(r.capped);
    }
    {
      INFO("birth");
      // The entering box's pixels, which no seed predicted, escalate; pooled escalation (P2-B B6a) closes the gap that
      // B4b left here (an edge of the box's shadow on the tail sub-samples of an N_base pixel at a window's end).
      const T2Run r = t2_run(backend, birth_scene(), 4);
      exact(r);
      CHECK(r.escalated > 0);
      CHECK(r.pooled > 0);
      MESSAGE("birth: active " << r.active << ", band " << r.band_pixels << ", escalated " << r.escalated
                               << " (pooled " << r.pooled << "), border " << r.border << ", switches " << r.switches
                               << ", gap pixel-windows " << r.gap << ", changed " << r.changed
                               << ", escalated changed " << r.esc_changed << ", gap nodes " << r.gap_nodes
                               << ", events " << (r.same_events ? "equal" : "differ"));
    }
    {
      // Every pixel forced to N_base (force_n overrides the border ring): the box's edges escalate window after
      // window. B4b left gap nodes here, which its moving edges left at a window's end on N_base pixels' tail
      // sub-samples (9, each escalated in the next window and counted by escalated_changed); pooled escalation
      // (P2-B B6a) escalates those pixels in the window itself, so none is left.
      INFO("birth, forced N_base");
      const T2Run r = t2_run(backend, birth_scene(), 4, {}, true);
      CHECK(r.band_pixels == 0);
      CHECK(r.escalated > 0);
      CHECK(r.pooled > 0);
      CHECK(r.differ == 0);
      CHECK(r.gap_nodes == 0);
      CHECK(r.esc_changed == 0);
      CHECK(r.same_events);
      MESSAGE("birth, forced N_base: active " << r.active << ", escalated " << r.escalated << " (pooled " << r.pooled
                                              << "), gap pixel-windows " << r.gap << ", gap nodes " << r.gap_nodes
                                              << " (re-rendered " << r.gap_nodes_rerendered << "), escalated changed "
                                              << r.esc_changed << " (unquiescent " << r.esc_unquiescent << ")");
    }
  }
}

TEST_CASE("pooled escalation's premise: a narrow pixel response leaves T2's tail gap open (P2-B B6a)") {
  // Pooling closes T2's tail gap where a neighbour's 64 points' hull reaches the edge, which needs the hulls to reach
  // past +-0.5 px (ENGINE_CPP §0.1 "P2-B B6 notes"; the default response's support is 0.95 px). At aperture 0.5 px and
  // blur 0.1 px (support 0.55 px) a strip between neighbouring hulls is seen by no 64-prefix. The birth scene's box,
  // 0.6 m wide here, slides along its top silhouette edge, which lies 0.41 px below row 20's centres (y 20.41, in that
  // strip): under a forced N_base map the row's pixels that no vertical edge crosses in a window differ from the
  // uniform N_band run's, unescalated. This records the limit. At the default response the same run differs only on
  // the image's border, where force_n overrides the ring (the box's trailing edge entering the left column's tail at a
  // window's end, which no neighbour sees).
  for (const render::BackendKind backend : engine_backends()) {
    INFO((backend == render::BackendKind::kOptix ? "device" : "host"));
    render::SceneBundle wide_box = birth_scene(-0.0993);  // top edge 35.5 - 96 (0.0993 + 0.05) / 0.95 px
    wide_box.bodies[1].mesh = box_mesh(0.3f, 0.05f, 0.05f);
    {
      INFO("default response");
      const T2Run r = t2_run(backend, wide_box, 4, {}, true);
      CHECK(r.pooled > 0);
      CHECK(r.differ == r.gap);
      CHECK(r.differ == r.differ_ring);
      MESSAGE("default response, forced N_base: active " << r.active << ", escalated " << r.escalated << " (pooled "
                                                         << r.pooled << "), gap pixel-windows " << r.gap
                                                         << " (on the border " << r.differ_ring << ")");
    }
    const T2Run r = t2_run(
        backend, wide_box, 4,
        [](EngineConfig& c, SensorSpec&) {
          c.render.scene.aperture_px = 0.5;
          c.render.scene.blur_sigma_px = 0.1;
        },
        true);
    CHECK(r.escalated > 0);
    CHECK(r.pooled > 0);
    CHECK(r.differ == r.gap);      // every difference an unescalated N_base pixel-window
    CHECK(r.gap > r.differ_ring);  // the limit: gaps inside the image
    MESSAGE("narrow response, forced N_base: active " << r.active << ", escalated " << r.escalated << " (pooled "
                                                      << r.pooled << "), gap pixel-windows " << r.gap
                                                      << " (on the border " << r.differ_ring << "), gap nodes "
                                                      << r.gap_nodes << ", events "
                                                      << (r.same_events ? "equal" : "differ"));
  }
}

TEST_CASE("a held body's shadow seeds a finite reach without the held world (P2-B B6b)") {
  // accuracy.held_world false traces the held box as a mover whose id is not among the window's moving bodies; no
  // moving pair shadows its shadow's receivers (k_pairs 0), where B4a's seed rate was unbounded (every tile in the
  // band). Its shadow's edges move with the static content, so its seeds now reach as under the held world.
  for (const render::BackendKind backend : engine_backends()) {
    INFO((backend == render::BackendKind::kOptix ? "device" : "host"));
    std::vector<PlanStats> plans[2];
    for (const bool held : {false, true}) {
      const std::unique_ptr<Engine> e =
          engine_on(backend, 0.0, at_n(16, 64, [held](EngineConfig& c, SensorSpec&) { c.held_world = held; }),
                    held_scene());
      CameraEngine& c = e->camera(0);
      c.start(kT0, 6 * kW, false);
      for (uint32_t w = 0; w < 3; ++w) plans[held ? 1 : 0].push_back(c.step(kT0 + w * kW).plan);
    }
    for (size_t w = 0; w < 3; ++w) {
      INFO("window " << w);
      const PlanStats &off = plans[0][w], &on = plans[1][w];
      CHECK(off.moving_bodies == 1);
      CHECK(off.tiles_dynamic == 12 * 9);
      CHECK(off.seed_pixels > 0);
      CHECK(off.seed_pixels == on.seed_pixels);
      CHECK(off.band_tiles < off.tiles_dynamic);  // a finite reach
      CHECK(off.band_tiles == on.band_tiles);
      MESSAGE("held box: band tiles " << off.band_tiles << " of " << off.tiles_dynamic << " (held world: "
                                      << on.band_tiles << ")");
    }
  }
}

// P2-B B6c: the per-pixel band (EngineConfig::band_growth pixel; band_map.h).
namespace {

// The band of seeds at radius[q] >= 0 by brute force: each seed's L-inf ball, clipped to the image.
std::vector<uint8_t> splats(const std::vector<int32_t>& radius, int32_t w, int32_t h) {
  std::vector<uint8_t> out(radius.size(), 0);
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {
      const int32_t r = radius[static_cast<size_t>(y) * w + x];
      if (r < 0) continue;
      for (int32_t v = std::max(0, y - r); v <= std::min(h - 1, y + r); ++v) {
        for (int32_t u = std::max(0, x - r); u <= std::min(w - 1, x + r); ++u) out[static_cast<size_t>(v) * w + u] = 1;
      }
    }
  }
  return out;
}

// The same restricted to a domain: the balls of the seeds inside it, clipped to it (0 outside it).
std::vector<uint8_t> splats_in(const std::vector<int32_t>& radius, int32_t w, int32_t h, BandDomain dom) {
  std::vector<int32_t> inside(radius.size(), -1);
  for (int32_t y = dom.y0; y < dom.y1; ++y) {
    for (int32_t x = dom.x0; x < dom.x1; ++x) {
      inside[static_cast<size_t>(y) * w + x] = radius[static_cast<size_t>(y) * w + x];
    }
  }
  std::vector<uint8_t> out = splats(inside, w, h);
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {
      if (x < dom.x0 || x >= dom.x1 || y < dom.y0 || y >= dom.y1) out[static_cast<size_t>(y) * w + x] = 0;
    }
  }
  return out;
}

Edit pixel_band(double sigma_cam, double sigma_mov, const Edit& more = {}) {
  return [=](EngineConfig& c, SensorSpec& s) {
    c.band_growth = BandGrowth::kPixels;
    c.band_reach = {sigma_cam, sigma_mov};
    if (more) more(c, s);
  };
}

}  // namespace

TEST_CASE("the per-pixel band's cascade is the exact union of the seeds' L-inf balls (P2-B B6c)") {
  // Random radius maps (seeds at a density, radii 0 to beyond the image, which the cap clips) on images, lines and
  // columns: the cascade of separable sliding-window ORs equals the seeds' splats, and its levels are the distinct
  // radii, descending. On a random crop's domain (the crop grown by the largest radius, band_domain; the review's
  // fix): on the crop the full image's splats, on the domain the splats of its seeds clipped to it, 0 outside it.
  std::mt19937 rng(17);
  using Case = std::tuple<int32_t, int32_t, double, int32_t>;  // width, height, seed density, largest radius
  for (const auto& [w, h, density, r_max] : {Case{40, 30, 0.05, 6}, Case{96, 72, 0.02, 40}, Case{96, 72, 0.3, 3},
                                             Case{1, 50, 0.1, 5}, Case{50, 1, 0.1, 60}, Case{33, 17, 0.003, 33},
                                             Case{16, 16, 0.0, 3}, Case{64, 48, 0.01, 64}}) {
    INFO(w << " x " << h << ", density " << density << ", radii <= " << r_max);
    for (int trial = 0; trial < 4; ++trial) {
      const int32_t cap = std::max(w, h);
      std::vector<int32_t> radius(static_cast<size_t>(w) * h, -1);
      std::vector<uint32_t> hist(static_cast<size_t>(cap) + 1, 0);
      std::uniform_real_distribution<double> u(0.0, 1.0);
      for (int32_t& r : radius) {
        if (u(rng) >= density) continue;
        r = std::min(static_cast<int32_t>(rng() % static_cast<uint32_t>(r_max + 1)), cap);
        ++hist[static_cast<size_t>(r)];
      }
      const std::vector<int32_t> levels = band_levels(hist);
      CHECK(std::is_sorted(levels.rbegin(), levels.rend()));
      CHECK(std::adjacent_find(levels.begin(), levels.end()) == levels.end());
      std::vector<uint8_t> band(radius.size(), 7), tmp(radius.size());
      band_cascade<Space::kHost>(radius.data(), levels, w, h, band.data(), tmp.data(), nullptr);
      const std::vector<uint8_t> full = splats(radius, w, h);
      CHECK(band == full);
      const int32_t cx0 = static_cast<int32_t>(rng() % static_cast<uint32_t>(w)),
                    cy0 = static_cast<int32_t>(rng() % static_cast<uint32_t>(h));
      const int32_t cx1 = cx0 + 1 + static_cast<int32_t>(rng() % static_cast<uint32_t>(w - cx0)),
                    cy1 = cy0 + 1 + static_cast<int32_t>(rng() % static_cast<uint32_t>(h - cy0));
      const BandDomain dom = band_domain(cx0, cy0, cx1, cy1, levels.empty() ? -1 : levels.front(), w, h);
      INFO("crop [" << cx0 << ", " << cx1 << ") x [" << cy0 << ", " << cy1 << "), domain [" << dom.x0 << ", "
                    << dom.x1 << ") x [" << dom.y0 << ", " << dom.y1 << ")");
      CHECK((dom.x0 <= cx0 && dom.y0 <= cy0 && dom.x1 >= cx1 && dom.y1 >= cy1));
      std::fill(band.begin(), band.end(), 7);
      std::fill(tmp.begin(), tmp.end(), 7);
      band_cascade<Space::kHost>(radius.data(), levels, w, h, dom, band.data(), tmp.data(), nullptr);
      CHECK(band == splats_in(radius, w, h, dom));
      bool crop_exact = true;
      for (int32_t y = cy0; y < cy1; ++y) {
        for (int32_t x = cx0; x < cx1; ++x) {
          crop_exact = crop_exact && band[static_cast<size_t>(y) * w + x] == full[static_cast<size_t>(y) * w + x];
        }
      }
      CHECK(crop_exact);
    }
  }
}

TEST_CASE("the per-pixel band's radii and seed motions by hand (P2-B B6c)") {
  // BandRadius: r = floor(max(rho_plan, rho_carried) + s + M), rho 1 for a plan seed and s for a carried one, a carried
  // seed's M at least its tile's floor (sigma_mov times the touches() reach), NaN and inf and beyond max(w, h) at the
  // cap. At s 0.95, cap 40: values worked out by hand.
  const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
  const std::vector<double> tile_floor{3.2, 0.0, inf};
  struct Seed {
    uint8_t kind;
    double m;
    uint32_t tile;
    int32_t r;
  };
  const std::vector<Seed> seeds{
      {kSeedPlan, 2.3, 1, 4},                   // floor(1 + 0.95 + 2.3)
      {kSeedCarried, 0.5, 0, 5},                // floor(0.95 + 0.95 + max(0.5, 3.2))
      {kSeedPlan | kSeedCarried, 0.0, 1, 1},    // floor(max(1, 0.95) + 0.95)
      {kSeedPlan | kSeedCarried, 0.5, 0, 5},    // floor(1 + 0.95 + max(0.5, 3.2))
      {kSeedPlan, 0.5, 0, 2},                   // a plan seed takes no tile floor: floor(1 + 0.95 + 0.5)
      {kSeedCarried, 7.0, 0, 8},                // floor(0.95 + 0.95 + 7)
      {kSeedPlan, 1.1, 1, 3},                   // floor(1 + 0.95 + 1.1)
      {kSeedCarried, 0.0, 1, 1},                // floor(0.95 + 0.95)
      {kSeedPlan, nan, 1, 40},                  // NaN: the cap
      {kSeedPlan, 100.0, 1, 40},                // beyond max(w, h): the cap
      {kSeedCarried, 0.0, 2, 40},               // an unbounded tile floor: the cap
      {0, 5.0, 0, -1},                          // no seed
  };
  const size_t n = seeds.size();
  std::vector<double> seed_m(n);
  std::vector<uint8_t> kind(n);
  std::vector<uint32_t> tile(n), hist(41, 0);
  std::vector<int32_t> radius(n, -7);
  for (size_t i = 0; i < n; ++i) {
    seed_m[i] = seeds[i].m;
    kind[i] = seeds[i].kind;
    tile[i] = seeds[i].tile;
  }
  Par<Space::kHost>::for_each(
      n, BandRadius{seed_m.data(), kind.data(), tile_floor.data(), tile.data(), 0.95, 40, radius.data(), hist.data()},
      nullptr);
  std::vector<uint32_t> want_hist(41, 0);
  for (size_t i = 0; i < n; ++i) {
    INFO("seed " << i);
    CHECK(radius[i] == seeds[i].r);
    if (seeds[i].r >= 0) ++want_hist[static_cast<size_t>(seeds[i].r)];
  }
  CHECK(hist == want_hist);
  CHECK(band_levels(hist) == std::vector<int32_t>{40, 8, 5, 4, 3, 2, 1});

  // PixelPlan::seed_motion: max(sigma_cam M_static, sigma_mov M_mov), M = jmax rate W (px), a part dropped at scale 0
  // (also an unbounded or NaN one), NaN kept at a positive scale. jmax 1.5, k_static 40 rad/s, movers 100 rad/s.
  const double w_s = 8388608e-9;  // the window, 2^23 ns
  const double ms = 1.5 * 40.0 * w_s, mm = 1.5 * 100.0 * w_s;  // 0.5033 and 1.2583 px
  PixelPlan pp{};
  pp.p.jmax = 1.5;
  const auto motion = [&pp](double cam, double mov, double k_static, double k_mov) {
    pp.p.sigma_cam = cam;
    pp.p.sigma_mov = mov;
    return pp.seed_motion(k_static, k_mov);
  };
  CHECK(motion(1.0, 1.0, 40.0, 100.0) == doctest::Approx(mm));
  CHECK(motion(1.0, 1.0, 100.0, 40.0) == doctest::Approx(mm));
  CHECK(motion(0.25, 0.0, 40.0, 100.0) == doctest::Approx(0.25 * ms));  // the static part only
  CHECK(motion(0.25, 0.0, 40.0, inf) == doctest::Approx(0.25 * ms));
  CHECK(motion(0.25, 0.0, 40.0, nan) == doctest::Approx(0.25 * ms));
  CHECK(motion(0.0, 1.0, 40.0, 100.0) == doctest::Approx(mm));  // the movers' part only
  CHECK(motion(0.0, 1.0, 400.0, 100.0) == doctest::Approx(mm));
  CHECK(motion(0.0, 1.0, inf, 100.0) == doctest::Approx(mm));
  CHECK(motion(0.0, 0.5, 400.0, 100.0) == doctest::Approx(0.5 * mm));
  CHECK(motion(1.0, 0.0, 40.0, 100.0) == doctest::Approx(ms));
  CHECK(motion(0.0, 0.0, inf, nan) == 0.0);
  CHECK(std::isnan(motion(1.0, 0.0, nan, 100.0)));
  CHECK(std::isinf(motion(0.0, 1.0, 40.0, inf)));
}

TEST_CASE("the per-pixel band's reach scales act on their own parts: a still camera (P2-B B6c)") {
  // The box scene's camera is still, so its seeds' static motion is 0: at sigma (1, 0) every radius is the footprint's
  // (floor(rho + s) = 1) and the map is sigma (0, 0)'s, while sigma (0, 1) reaches farther with the moving box. A swap
  // of the two scales (in seed_motion or its caller) or of the carried seeds' floor scale fails here.
  for (const render::BackendKind backend : engine_backends()) {
    INFO((backend == render::BackendKind::kOptix ? "device" : "host"));
    const auto at = [backend](double cam, double mov) {
      return engine_on(backend, 0.0, at_n(16, 64, pixel_band(cam, mov)), wide(box_scene()));
    };
    const std::unique_ptr<Engine> cam_only = at(1.0, 0.0), none = at(0.0, 0.0), mov = at(0.0, 1.0);
    CameraEngine &a = cam_only->camera(0), &z = none->camera(0), &m = mov->camera(0);
    for (CameraEngine* c : {&a, &z, &m}) c->start(kT0, 6 * kW, false);
    int32_t mov_max = -1;
    for (uint32_t w = 0; w < 3; ++w) {
      INFO("window " << w);
      const StepResult ra = a.step(kT0 + w * kW), rz = z.step(kT0 + w * kW), rm = m.step(kT0 + w * kW);
      CHECK(ra.plan.seed_pixels > 0);
      CHECK(ra.plan.radius_max == 1);
      CHECK(rz.plan.radius_max == 1);
      CHECK((ra.plan.radius_levels == 1 && rz.plan.radius_levels == 1));
      CHECK(a.last_band() == z.last_band());
      mov_max = std::max(mov_max, rm.plan.radius_max);
      MESSAGE("still camera, window " << w << ": radius max " << ra.plan.radius_max << " at (1, 0), "
                                      << rm.plan.radius_max << " at (0, 1) (" << rm.plan.radius_levels << " levels)");
    }
    CHECK(mov_max > 1);
  }
}

TEST_CASE("the per-pixel band equals the uniform N_band run bit for bit (P2-B B6c, T2) at reach scales 1 and 0") {
  // T2 rests on the escalation and pooling (memo Appendix A), not on the band's reach: at sigma (1, 1), each seed's
  // reach in pixels, and at (0, 0), the seeds and their footprint alone, the band-on run is the uniform N_band run on
  // both engines, for the box scene (N_base pixels and switches), a crop, 18 shadow lights, the clamp and the
  // unpredicted entering box. At (0, 0) the band is smaller and the escalations more.
  for (const render::BackendKind backend : engine_backends()) {
    INFO((backend == render::BackendKind::kOptix ? "device" : "host"));
    const auto exact = [](const T2Run& r) {
      CHECK(r.differ == 0);
      CHECK(r.same_events);
      CHECK(r.changed == 0);
      CHECK(r.unquiescent == 0);
      CHECK(r.esc_changed == 0);
      CHECK(r.esc_unquiescent == 0);
      CHECK(r.gap_nodes == 0);
      CHECK(r.ring_base == 0);
    };
    std::vector<T2Run> box;
    for (const auto& [cam, mov] : {std::pair{1.0, 1.0}, std::pair{0.0, 0.0}}) {
      INFO("sigma (" << cam << ", " << mov << ")");
      {
        INFO("box");
        box.push_back(t2_run(backend, wide(box_scene()), 4, pixel_band(cam, mov)));
        const T2Run& r = box.back();
        exact(r);
        CHECK(r.band_pixels < r.active);
        CHECK(r.plan > 0);
        MESSAGE("pixel band (" << cam << ", " << mov << "), box: active " << r.active << ", band " << r.band_pixels
                               << " (plan " << r.plan << ", border " << r.border << "), escalated " << r.escalated
                               << " (pooled " << r.pooled << "), switches " << r.switches);
      }
      {
        INFO("crop");
        const T2Run r = t2_run(backend, wide(box_scene()), 3, pixel_band(cam, mov), false, Crop{20, 12, 76, 60});
        exact(r);
        CHECK(r.ring > 0);
      }
      {
        INFO("18 shadow lights");
        const T2Run r = t2_run(backend, many_lights_scene(), 2, pixel_band(cam, mov));
        exact(r);
        MESSAGE("pixel band (" << cam << ", " << mov << "), 18 lights: active " << r.active << ", band "
                               << r.band_pixels << ", escalated " << r.escalated);
      }
      {
        INFO("clamp");
        const T2Run r = t2_run(backend, wide(box_scene()), 3, pixel_band(cam, mov, [](EngineConfig& c, SensorSpec&) {
                                 c.max_samples_per_window = 96 * 72 * 3;
                               }));
        exact(r);
        CHECK(r.capped);
      }
      {
        INFO("birth");
        const T2Run r = t2_run(backend, birth_scene(), 4, pixel_band(cam, mov));
        exact(r);
        CHECK(r.escalated > 0);
        MESSAGE("pixel band (" << cam << ", " << mov << "), birth: active " << r.active << ", band " << r.band_pixels
                               << ", escalated " << r.escalated << " (pooled " << r.pooled << ")");
      }
    }
    REQUIRE(box.size() == 2);
    CHECK(box[1].band_pixels < box[0].band_pixels);
    CHECK(box[1].escalated >= box[0].escalated);
  }
}

TEST_CASE("the per-pixel band: host = device with 0, 3 and 18 shadow lights and crops; the plan unchanged (P2-B B6c)") {
  // The band's map (CameraEngine::last_band) and its counters, window by window, on both engines (the plan pass's
  // seeds, BandRadius and the cascade): equal. The tile levels and the active pixels are the tile band's (the map
  // moves N only); the plan's band pixels are counted (band_plan_pixels, with the ring band_pixels); a smaller reach
  // scale gives a sub-band of the same window's seeds (the first window: the start frame's carried words).
  render::SceneBundle none = wide(box_scene()), three = wide(box_scene());
  none.lights = {point_light({0.3, -0.4, 0.0}, 2.0f, false), ambient_light(0.3f)};
  three.lights = {point_light({0.3, -0.4, 0.0}, 1.0f, true), point_light({-0.4, 0.3, 0.2}, 0.7f, true),
                  point_light({0.0, 0.5, -0.3}, 0.5f, true), ambient_light(0.3f)};
  using Case = std::tuple<std::string, render::SceneBundle, std::optional<Crop>>;
  for (const auto& [name, bundle, crop] :
       {Case{"0 lights", none, {}}, Case{"3 lights", three, {}}, Case{"3 lights, crop", three, Crop{20, 12, 76, 60}},
        Case{"18 lights", many_lights_scene(), {}}, Case{"18 lights, crop", many_lights_scene(), Crop{8, 6, 50, 40}}}) {
    INFO(name);
    std::vector<std::vector<uint64_t>> counts;
    std::vector<std::vector<std::vector<uint8_t>>> maps;
    for (const render::BackendKind backend : engine_backends()) {
      INFO((backend == render::BackendKind::kOptix ? "device" : "host"));
      const Edit cut = [crop](EngineConfig& c, SensorSpec&) { c.crop = crop; };
      const std::unique_ptr<Engine> pix = engine_on(backend, 0.0, at_n(16, 64, pixel_band(0.25, 0.0, cut)), bundle);
      const std::unique_ptr<Engine> one = engine_on(backend, 0.0, at_n(16, 64, pixel_band(1.0, 1.0, cut)), bundle);
      const std::unique_ptr<Engine> tile = engine_on(backend, 0.0, at_n(16, 64, cut), bundle);
      CameraEngine &a = pix->camera(0), &b = one->camera(0), &t = tile->camera(0);
      for (CameraEngine* c : {&a, &b, &t}) c->start(kT0, 6 * kW, false);
      std::vector<uint64_t> c_counts;
      std::vector<std::vector<uint8_t>> c_maps;
      for (uint32_t w = 0; w < 3; ++w) {
        INFO("window " << w);
        const StepResult ra = a.step(kT0 + w * kW), rb = b.step(kT0 + w * kW), rt = t.step(kT0 + w * kW);
        CHECK(ra.plan.pixel);
        CHECK_FALSE(rt.plan.pixel);
        CHECK(ra.plan.levels == rt.plan.levels);
        CHECK(ra.plan.tiles_dynamic == rt.plan.tiles_dynamic);
        CHECK(ra.plan.band_tiles == 0);
        CHECK(a.last_runs().pixel == t.last_runs().pixel);
        const std::vector<uint8_t> ma = a.last_band(), mb = b.last_band();
        REQUIRE(ma.size() == static_cast<size_t>(bundle.cameras[0].width) * bundle.cameras[0].height);
        CHECK(t.last_band().size() == ma.size());
        const uint64_t in_a = static_cast<uint64_t>(std::count(ma.begin(), ma.end(), 1));
        CHECK(in_a > 0);
        CHECK(in_a < ma.size());
        if (w == 0) {
          bool sub = true;
          for (size_t p = 0; p < ma.size(); ++p) sub = sub && (ma[p] == 0 || mb[p] == 1);
          CHECK(sub);
        }
        CHECK(ra.plan.seed_pixels > 0);
        CHECK(ra.plan.radius_max >= 1);
        CHECK(ra.plan.radius_levels >= 1);
        const SwitchStats& s = ra.window.band;
        CHECK(s.plan + s.border == s.band_pixels);
        CHECK(s.plan <= in_a);
        c_counts.insert(c_counts.end(), {in_a, static_cast<uint64_t>(ra.plan.radius_max), ra.plan.radius_levels,
                                         ra.plan.seed_pixels, ra.plan.carried_seeds, s.plan, s.border,
                                         static_cast<uint64_t>(rb.plan.radius_max), rb.plan.radius_levels});
        c_maps.push_back(ma);
        c_maps.push_back(mb);
        MESSAGE(name << ", window " << w << ": band " << in_a << " (sigma 1: "
                     << std::count(mb.begin(), mb.end(), 1) << ") of " << ma.size() << " pixels, radius max "
                     << ra.plan.radius_max << " (" << rb.plan.radius_max << "), levels " << ra.plan.radius_levels
                     << " (" << rb.plan.radius_levels << ")");
      }
      counts.push_back(std::move(c_counts));
      maps.push_back(std::move(c_maps));
    }
    if (counts.size() == 2) {  // host = device
      CHECK(counts[0] == counts[1]);
      CHECK(maps[0] == maps[1]);
    }
  }
}

// P2-D D3b (plan p2d.md §5.3, §6 row D3b; G-D2): accuracy.localization shadow at a uniform N 256 on the host engine's
// renderer: the box scene, a box born in the run's first window, eight shadow-casting lights (the codes' widest mask)
// and eighteen (more than P&V's eight: off). The events are bisect's bit for bit, and the state check at bisection's
// midpoints finds no contradiction between P&V's brackets and the rendered sub-sample states.
TEST_CASE("P2-D D3b: the state check on the box, birth and many-light scenes: no contradiction, events unchanged") {
  struct Scene {
    const char* name;
    render::SceneBundle bundle;
    bool pv;
  };
  render::SceneBundle eight = many_lights_scene();
  eight.lights.erase(eight.lights.begin() + 8, eight.lights.begin() + 18);  // 8 shadow lights and the ambient
  for (Scene& sc : std::vector<Scene>{{"box", box_scene(), true}, {"birth", birth_scene(), true},
                                       {"8 lights", eight, true}, {"18 lights", many_lights_scene(), false}}) {
    INFO(sc.name);
    std::vector<OutEvent> ev[2];
    std::array<uint64_t, kPvWords> pv{};
    bool on = false;
    for (const int shadow : {0, 1}) {
      const Edit edit = [shadow](EngineConfig& c, SensorSpec&) {
        c.render.scene.n_subsamples = 256;
        c.localization = shadow != 0 ? Localization::kShadow : Localization::kBisect;
      };
      const std::unique_ptr<Engine> e = engine_on(render::BackendKind::kEmbree, 0.0, edit, sc.bundle);
      CameraEngine& cam = e->camera(0);
      cam.start(kT0, 6 * kW, false);
      for (int w = 0; w < 2; ++w) {
        StepResult r = cam.step(kT0 + w * kW);
        ev[shadow].insert(ev[shadow].end(), r.ready.begin(), r.ready.end());
        if (shadow == 0) continue;
        on = on || r.window.stats.pv_on;
        for (const WindowStats* st : {&r.window.stats, &r.window.escalation}) {
          if (!st->pv_on) continue;
          for (int q = 0; q < kPvWords; ++q) pv[static_cast<size_t>(q)] += st->pv[static_cast<size_t>(q)];
        }
      }
      const std::vector<OutEvent> rest = cam.flush();
      ev[shadow].insert(ev[shadow].end(), rest.begin(), rest.end());
    }
    CHECK(same(ev[0], ev[1]));
    CHECK(!ev[0].empty());
    CHECK(on == sc.pv);
    if (!sc.pv) continue;
    MESSAGE(std::string(sc.name) << ": " << pv[kPvCand] << " candidates, " << pv[kPvCand + 2] << " flips, "
                    << static_cast<double>(pv[kPvCand + 4]) / static_cast<double>(std::max<uint64_t>(pv[kPvCand + 3], 1))
                    << " probes per flip; check: " << pv[kPvCheck + kPvChkMid] << " midpoints, "
                    << pv[kPvCheck + kPvChkCompared] << " compared, " << pv[kPvCheck + kPvChkTransient]
                    << " transients, " << pv[kPvCheck + kPvChkMulti] << " multi, " << pv[kPvCheck + kPvChkThird]
                    << " third-state, " << pv[kPvCheckBug + 1] << " unchecked; R3 " << pv[kPvRuleB + 2] << ", R4 "
                    << pv[kPvRuleB]);
    CHECK(pv[kPvCand] > 0);
    CHECK(pv[kPvCand + 3] > 0);
    CHECK(pv[kPvCheck + kPvChkMid] > 0);
    CHECK(pv[kPvCheck + kPvChkCompared] > 0);
    CHECK(pv[kPvCheckBug] == 0);  // G-D2: no bracket contradiction
    CHECK(pv[kPvUnfinished] == 0);
  }
}

// P2-H H9 (accuracy.level_policy; F004, DESIGN §13.5): global puts the window's level L_w, the largest level of the
// per-tile plan's dynamic tiles (l_min when none is dynamic), on every active tile, and global_certified on the
// dynamic tiles alone, the static tiles certified as under tile; so the lattice holds the active pixels x 2^L_w
// samples. Under a held camera with the box and its shadow moving, the per-tile plan has static tiles and tiles below
// L_w, so the three policies differ.
TEST_CASE("the level policies (P2-H H9): the window's largest tile level on every active tile or on the dynamic ones") {
  const auto steps = [](LevelPolicy policy, std::optional<Crop> crop, render::SceneBundle bundle) {
    const std::unique_ptr<Engine> e = engine_on(
        render::BackendKind::kEmbree, 0.0,
        [&](EngineConfig& c, SensorSpec&) {
          c.level_policy = policy;
          c.crop = crop;
        },
        std::move(bundle));
    CameraEngine& cam = e->camera(0);
    cam.start(kT0, 6 * kW, false);
    std::vector<StepResult> out;
    for (int w = 0; w < 3; ++w) out.push_back(cam.step(kT0 + w * kW));
    return out;
  };
  const auto area = [](const std::optional<Crop>& c) {
    return c ? static_cast<uint64_t>(c->x1 - c->x0) * static_cast<uint64_t>(c->y1 - c->y0) : uint64_t{40 * 30};
  };
  const auto tiles_in = [](const std::optional<Crop>& c) {  // 8 px tiles of the 40 x 30 image that the crop touches
    const Crop k = c.value_or(Crop{0, 0, 40, 30});
    return static_cast<uint64_t>((k.x1 - 1) / 8 - k.x0 / 8 + 1) * static_cast<uint64_t>((k.y1 - 1) / 8 - k.y0 / 8 + 1);
  };
  const int32_t l_min = 1;  // the midpoint test is on (EngineConfig::refine_tol_log)
  bool static_seen = false, below_seen = false;
  for (const std::optional<Crop>& crop : {std::optional<Crop>{}, std::optional<Crop>{Crop{4, 2, 29, 21}}}) {
    CAPTURE(crop.has_value());
    const std::vector<StepResult> tile = steps(LevelPolicy::kPerTile, crop, box_scene());
    const std::vector<StepResult> global = steps(LevelPolicy::kGlobal, crop, box_scene());
    const std::vector<StepResult> certified = steps(LevelPolicy::kGlobalCertified, crop, box_scene());
    for (size_t w = 0; w < tile.size(); ++w) {
      CAPTURE(w);
      const StepResult &t = tile[w], &g = global[w], &c = certified[w];
      REQUIRE((t.planned && g.planned && c.planned));
      CHECK_FALSE((t.level_cap || g.level_cap || c.level_cap));
      int32_t l_w = -1;
      uint64_t dynamic = 0;
      for (size_t l = 0; l < t.plan.levels.size(); ++l) {
        if (t.plan.levels[l] > 0) l_w = static_cast<int32_t>(l);
        dynamic += t.plan.levels[l];
      }
      REQUIRE(l_w >= 0);  // the box moves in every window
      CHECK(t.plan.level_w == -1);
      CHECK(dynamic == t.plan.tiles_dynamic);
      static_seen = static_seen || t.plan.tiles_static > 0;
      below_seen = below_seen || t.plan.levels[static_cast<size_t>(l_w)] < dynamic;
      // global: every active tile at L_w, none static; the lattice is the crop's pixels x 2^L_w.
      CHECK(g.plan.level_w == l_w);
      CHECK(g.plan.tiles_static == 0);
      CHECK(g.plan.tiles_dynamic == tiles_in(crop));
      CHECK(g.plan.levels[static_cast<size_t>(l_w)] == tiles_in(crop));
      CHECK(g.window.stats.pixels == area(crop));
      CHECK(g.window.stats.samples_by_round.at(0) == (area(crop) << l_w));
      // global_certified: the per-tile plan's static set; its dynamic pixels at L_w.
      CHECK(c.plan.level_w == l_w);
      CHECK(c.plan.tiles_static == t.plan.tiles_static);
      CHECK(c.plan.tiles_dynamic == t.plan.tiles_dynamic);
      CHECK(c.plan.levels[static_cast<size_t>(l_w)] == t.plan.tiles_dynamic);
      CHECK(c.window.stats.pixels == t.window.stats.pixels);
      CHECK(c.window.stats.samples_by_round.at(0) == (t.window.stats.pixels << l_w));
      CHECK(c.window.stats.samples_by_round.at(0) >= t.window.stats.samples_by_round.at(0));
      // The plan's own bound is the per-tile one under every policy.
      CHECK(g.plan.v_hat_max == t.plan.v_hat_max);
      CHECK(c.plan.v_hat_max == t.plan.v_hat_max);
    }
  }
  CHECK(static_seen);  // the policies differ on this scene: static tiles, and dynamic tiles below L_w
  CHECK(below_seen);
  // No tile dynamic (the box held: no plan pass): global samples every active pixel at l_min, the others none.
  render::SceneBundle still = box_scene();
  still.trajectories[1] = static_trajectory(kT0 - kW, kT0 + 8 * kW, quat_identity<double>(), {-0.2, 0.0, 1.0});
  const std::vector<StepResult> t = steps(LevelPolicy::kPerTile, std::nullopt, still);
  const std::vector<StepResult> g = steps(LevelPolicy::kGlobal, std::nullopt, still);
  const std::vector<StepResult> c = steps(LevelPolicy::kGlobalCertified, std::nullopt, still);
  for (size_t w = 0; w < t.size(); ++w) {
    CAPTURE(w);
    CHECK_FALSE((t[w].plan.plan_pass || g[w].plan.plan_pass || c[w].plan.plan_pass));
    CHECK((t[w].window.stats.pixels == 0 && c[w].window.stats.pixels == 0));
    CHECK(g[w].plan.level_w == l_min);
    CHECK(g[w].plan.tiles_dynamic == tiles_in(std::nullopt));
    CHECK(g[w].window.stats.samples_by_round.at(0) == (area(std::nullopt) << l_min));
    CHECK(c[w].plan.level_w == l_min);
    CHECK(c[w].plan.tiles_dynamic == 0);
  }
  // The fixed-rate grid has no plan: the policy is inert there.
  const auto fixed = [](LevelPolicy policy) {
    const std::unique_ptr<Engine> e = host_engine(0.0, [policy](EngineConfig& cfg, SensorSpec&) {
      cfg.fixed_period_ns = 100'000;
      cfg.level_policy = policy;
    });
    CameraEngine& cam = e->camera(0);
    cam.start(kT0, 6 * kW, false);
    std::vector<OutEvent> out;
    for (int w = 0; w < 3; ++w) {
      const StepResult r = cam.step(kT0 + w * kW);
      CHECK_FALSE(r.planned);
      out.insert(out.end(), r.ready.begin(), r.ready.end());
    }
    const std::vector<OutEvent> rest = cam.flush();
    out.insert(out.end(), rest.begin(), rest.end());
    return out;
  };
  const std::vector<OutEvent> grid = fixed(LevelPolicy::kPerTile);
  CHECK(grid.size() > 50);
  CHECK(same(fixed(LevelPolicy::kGlobal), grid));
}

TEST_CASE("baked proxies: no window holds a mover, so held_world changes nothing (scans.md §4.2)") {
  // stopping_scene's wall a baked scan under its point light as a proxy: the big box's hold (I3b) would merge it into
  // the scan's world, where the box could not cast its proxy shadow. Renderer::holds_movers is false there, so no
  // window holds it, and held_world true runs as held_world false, event for event. The control, without the proxy:
  // the held set is rebuilt once.
  const auto scene = [](bool proxy) {
    render::SceneBundle s = stopping_scene(kT0 + kW);
    render::Material& m = s.bodies[0].materials[0];
    m.shading = render::Shading::kBaked;
    m.baked_scale = 0.3f;
    s.bodies[0].scan = true;
    s.lights[0].proxy = proxy;
    return s;
  };
  for (const render::BackendKind backend : engine_backends()) {
    std::vector<OutEvent> kept[2];  // the held_world true runs with and without the proxy
    for (const bool proxy : {true, false}) {
      std::vector<OutEvent> events[2];
      uint32_t held_windows[2] = {}, rebuilds[2] = {};
      for (const bool held : {false, true}) {
        INFO(std::string(backend == render::BackendKind::kOptix ? "device" : "host") << ", proxy " << proxy
                                                                                     << ", held_world " << held);
        const auto world = [held](EngineConfig& c, SensorSpec&) { c.held_world = held; };
        const std::unique_ptr<Engine> e = engine_on(backend, 0.0, world, scene(proxy));
        CameraEngine& cam = e->camera(0);
        const StartResult started = cam.start(kT0, 6 * kW, false);
        // S3b's start diagnostic (scans.md §4.2): per proxy shadow light the share of its rays from baked receivers that
        // the scan blocks, and the split static bodies; empty and 0 without a proxy. The renderer's own value on the same
        // bundle is the oracle (the engine's start frame is t_s).
        if (proxy) {
          REQUIRE(started.proxy_blocked.size() == 1);
          render::RendererOptions ro;
          ro.backend = backend;
          render::Renderer direct(scene(true), ro);
          const std::vector<float> want = direct.proxy_visibility(0, kT0);
          REQUIRE(want.size() == 1);
          CHECK(std::abs(started.proxy_blocked[0] - want[0]) <= 1e-3f);
          CHECK((started.proxy_blocked[0] >= 0.0f && started.proxy_blocked[0] <= 1.0f));
          CHECK(started.split_geometries == direct.split_geometries());
        } else {
          CHECK(started.proxy_blocked.empty());
          CHECK(started.split_geometries == 0);
        }
        std::vector<OutEvent>& out = events[held ? 1 : 0];
        for (int w = 0; w < 5; ++w) {
          const StepResult r = cam.step(kT0 + w * kW);
          held_windows[held ? 1 : 0] += r.window.held.bodies > 0 ? 1u : 0u;
          rebuilds[held ? 1 : 0] += r.window.held.rebuilt ? 1u : 0u;
          out.insert(out.end(), r.ready.begin(), r.ready.end());
        }
        const std::vector<OutEvent> rest = cam.flush();
        out.insert(out.end(), rest.begin(), rest.end());
        CHECK(out.size() > 100);
      }
      const std::string what = std::string(proxy ? "baked proxies" : "no proxy") + " on the " +
                               (backend == render::BackendKind::kOptix ? "device" : "host");
      MESSAGE(what << ": held windows " << held_windows[1] << ", rebuilds " << rebuilds[1] << ", events "
                   << events[1].size());
      CHECK(held_windows[0] == 0);
      if (proxy) {
        CHECK(held_windows[1] == 0);
        CHECK(rebuilds[1] == 0);
        CHECK(same(events[0], events[1]));
      } else {
        CHECK(held_windows[1] > 0);
        CHECK(rebuilds[1] == 1);
      }
      kept[proxy ? 0 : 1] = events[1];
    }
    // Since S3a-2 the proxy lamp's shadow of the box darkens the scan (U2): the proxy run's events are not the control's.
    CHECK_FALSE(same(kept[0], kept[1]));
  }
}
