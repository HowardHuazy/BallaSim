// SensorModel configuration and input validation (contract C3).
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <numbers>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sensor/sensor_model.h"

using namespace balla::sensor;

// Out of memory inside the kernel: while armed on this thread, the allocation of one RawEvent (the first push_back
// of a worker's event vector) throws std::bad_alloc. Otherwise operator new is malloc's.
namespace {
thread_local bool g_fail_event_alloc = false;
}  // namespace

void* operator new(std::size_t size) {
  if (g_fail_event_alloc && size == sizeof(RawEvent)) throw std::bad_alloc();
  if (void* p = std::malloc(size ? size : 1)) return p;
  throw std::bad_alloc();
}
// The replaced operator new is malloc's, so free() is its pair; GCC flags free() once it inlines both into one test
// (a false -Wmismatched-new-delete that comes and goes with what else it inlines there).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
#pragma GCC diagnostic pop

TEST_CASE("presets validate and invalid parameters are refused with the field name") {
  for (const std::string& name : preset_names()) CHECK_NOTHROW(preset(name).validate());
  CHECK_THROWS_AS(preset("davis"), std::invalid_argument);
  SensorParams p = preset("prophesee_gen3");
  p.readout_delay_s = 0.5e-6;  // d0 < quantum breaks causality (DESIGN §8.7)
  CHECK_THROWS_WITH_AS(p.validate(), doctest::Contains("readout_delay_s"), std::invalid_argument);
  // The poles may come in either order (M3 Table A.1 ranges overlap) but must differ by 10 %.
  for (double ratio : {1.0, 1.05, 1.0 / 1.05}) {
    p = preset("prophesee_gen3");
    p.tau_sf = p.tau_o * ratio;
    CHECK_THROWS_WITH_AS(p.validate(), doctest::Contains("tau_sf"), std::invalid_argument);
  }
  p = preset("prophesee_gen3");
  p.tau_o = 300e-6;
  p.tau_sf = 100e-6;
  CHECK_NOTHROW(p.validate());
  p = preset("prophesee_gen3");
  p.hot_fraction = 4e-4;  // off in the prior (D4c)
  p.refractory_s = 1e-3;  // hot pixels up to 2 kHz cannot fire once per millisecond
  CHECK_THROWS_WITH_AS(p.validate(), doctest::Contains("hot_rate_max_hz"), std::invalid_argument);
  p.hot_fraction = 0;
  CHECK_NOTHROW(p.validate());
  p = preset("prophesee_gen3");
  p.eps_ci = 0.01;
  CHECK_THROWS_WITH_AS(p.validate(), doctest::Contains("eps_ci"), std::invalid_argument);
  p = preset("prophesee_gen3");
  p.dark_current_fA = 0;
  CHECK_THROWS_AS(p.validate(), std::invalid_argument);
  CHECK_THROWS_AS(make_fourier_waveform(120, {0.7, 0.4}, {0, 0}), std::invalid_argument);
  CHECK_THROWS_AS(make_pulse_waveform(200, 0.06, 0, 0, 0.2), std::invalid_argument);
  CHECK_NOTHROW(make_pulse_waveform(200, 0.06, 20e-6, 0, 0.2));
}

TEST_CASE("pulse waveforms are mean-normalized and exactly piecewise linear on the ns grid") {
  const Waveform w = make_pulse_waveform(200, 0.06, 20e-6, 1e-3, 0.2);
  double sum = 0;
  for (int64_t t = 0; t < 5'000'000; ++t) sum += waveform_value<double>(w, t);
  CHECK(sum / 5'000'000 == doctest::Approx(1.0).epsilon(1e-6));
  // Between consecutive breakpoints m is linear: the midpoint equals the mean of the ends.
  int64_t t = 0;
  for (int i = 0; i < 12; ++i) {
    const int64_t b = waveform_next_breakpoint(w, t);
    const int64_t mid = t + (b - t) / 2;
    if ((b - t) % 2 == 0) {
      CHECK(waveform_value<double>(w, mid) ==
            doctest::Approx(0.5 * (waveform_value<double>(w, t) + waveform_value<double>(w, b))).epsilon(1e-12));
    }
    t = b;
  }
}

TEST_CASE("fourier waveforms follow their formula and their bounds hold") {
  const double a[3] = {0.3, -0.2, 0.1}, phi[3] = {0.5, -1.0, 2.5};
  const Waveform w = make_fourier_waveform(120, {a[0], a[1], a[2]}, {phi[0], phi[1], phi[2]});
  // m = 1 + sum_k a_k cos(2 pi (k + 1) 120 t / 1e9 + phi_k), the cycles reduced exactly in integers.
  const auto ref = [&](int64_t t) {
    double m = 1;
    for (int k = 0; k < 3; ++k) {
      const int64_t num = ((k + 1) * 120 * t) % 1'000'000'000;
      m += a[k] * std::cos(2 * std::numbers::pi * static_cast<double>(num) / 1e9 + phi[k]);
    }
    return m;
  };
  std::mt19937_64 rng(11);
  // Values up to t = 1e13 ns (f t ~ 1.2e6 cycles: DESIGN §7.5's phase error, ~5e-10 cycles per harmonic).
  for (int i = 0; i < 2000; ++i) {
    const int64_t t = std::uniform_int_distribution<int64_t>(0, 9'999'999'999'999)(rng);
    CHECK(std::fabs(waveform_value<double>(w, t) - ref(t)) < 1e-8);
    CHECK(std::fabs(static_cast<double>(waveform_value<float>(w, t)) - ref(t)) < 2e-6);
  }
  // Bounds of pieces of 1 us to 10 ms, against the reference at 65 points: the minimum, the slopes, and the second
  // differences at steps of 10 us or more (slack for the reference's rounding).
  for (int i = 0; i < 200; ++i) {
    const int64_t t_a = std::uniform_int_distribution<int64_t>(0, 9'999'999'999)(rng);
    const int64_t h = std::uniform_int_distribution<int64_t>(1'000, 10'000'000)(rng) / 64;
    const WaveBounds<double> b = waveform_bounds<double>(w, t_a, t_a + 64 * h);
    const double h_s = static_cast<double>(h) * 1e-9;
    double m[65];
    for (int j = 0; j <= 64; ++j) m[j] = ref(t_a + j * h);
    for (int j = 0; j <= 64; ++j) CHECK(m[j] >= b.m_min - 1e-11);
    for (int j = 0; j < 64; ++j) CHECK(std::fabs(m[j + 1] - m[j]) / h_s <= b.dm_max * (1 + 1e-9) + 1e-14 / h_s);
    if (h < 10'000) continue;
    for (int j = 1; j < 64; ++j) {
      CHECK(std::fabs(m[j + 1] - 2 * m[j] + m[j - 1]) / (h_s * h_s) <= b.d2m_max * (1 + 1e-6) + 1e-14 / (h_s * h_s));
    }
  }
}

TEST_CASE("waveforms outside the factories' rules are refused at construction (M24)") {
  const SensorParams p = preset("prophesee_gen3");
  const auto build = [&](const Waveform& w) { SensorModel s(p, 2, 2, 1, {make_constant_waveform(), w}, 0, 1); };
  CHECK_NOTHROW(build(make_fourier_waveform(120, {0.3, -0.2}, {0.5, 1.0})));
  CHECK_NOTHROW(build(make_pulse_waveform(200, 0.06, 20e-6, 1e-3, 0.2)));
  CHECK_NOTHROW(build(make_pulse_waveform(1e-9, 0.5, 1e-6, 0, 0.2)));  // a period of 1e18 ns
  Waveform w = make_fourier_waveform(120, {0.1}, {0.0});
  w.n_harmonics = 9;  // the kernel would read past a[kMaxHarmonics]
  CHECK_THROWS_WITH_AS(build(w), doctest::Contains("n_harmonics"), std::invalid_argument);
  w.n_harmonics = 2;
  w.a[0] = 0.8;
  w.a[1] = 0.5;  // m < 0
  CHECK_THROWS_WITH_AS(build(w), doctest::Contains("sum |a_k|"), std::invalid_argument);
  w = make_pulse_waveform(200, 0.06, 20e-6, 0, 0.2);
  w.period_ns = 0;  // a division by zero in pulse_period
  CHECK_THROWS_WITH_AS(build(w), doctest::Contains("period_ns"), std::invalid_argument);
  w.type = static_cast<WaveformType>(7);
  CHECK_THROWS_WITH_AS(build(w), doctest::Contains("type"), std::invalid_argument);
  // The factory accepts it, but its period index and corners would leave int64 (the kernel did not terminate).
  CHECK_THROWS_WITH_AS(build(make_pulse_waveform(1e-11, 0.5, 1e-6, 0, 0.2)), doctest::Contains("period_ns"),
                       std::invalid_argument);
}

TEST_CASE("advance validates its inputs") {
  SensorParams p = preset("ideal");
  SensorModel s(p, 4, 3, 1, {make_constant_waveform()}, 0, 1);
  const uint32_t ids[2] = {1, 5};
  const int64_t t_ok[4] = {10, 20, 10, 30};
  const float irr[4] = {1, 2, 3, 4};
  CHECK_THROWS_AS(s.advance(ids, 2, t_ok, irr, 2, nullptr, 100), std::logic_error);  // before init
  std::vector<float> irr0(12, 1.0f);
  s.init(0, irr0.data());
  const uint32_t bad_id[2] = {1, 12};
  CHECK_THROWS_AS(s.advance(bad_id, 2, t_ok, irr, 2, nullptr, 100), std::invalid_argument);
  const uint32_t dup[2] = {3, 3};
  CHECK_THROWS_AS(s.advance(dup, 2, t_ok, irr, 2, nullptr, 100), std::invalid_argument);
  const int64_t t_bad[4] = {10, 10, 10, 30};
  CHECK_THROWS_AS(s.advance(ids, 2, t_bad, irr, 2, nullptr, 100), std::invalid_argument);
  CHECK_THROWS_AS(s.advance(ids, 2, t_ok, irr, 2, nullptr, 25), std::invalid_argument);  // sample after t_end
  const int32_t counts[2] = {2, 3};
  CHECK_THROWS_AS(s.advance(ids, 2, t_ok, irr, 2, counts, 100), std::invalid_argument);
  const int32_t ragged[2] = {1, 2};
  CHECK_NOTHROW(s.advance(ids, 2, t_ok, irr, 2, ragged, 100));
  CHECK(s.time_ns() == 100);
  const int64_t t_past[4] = {50, 60, 70, 80};
  CHECK_THROWS_AS(s.advance(ids, 2, t_past, irr, 2, nullptr, 200), std::invalid_argument);  // before model time
  CHECK_THROWS_AS(s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, 50), std::invalid_argument);
  // A first sample at exactly the model time replaces the held value.
  const int64_t t_now[4] = {100, 150, 100, 160};
  CHECK_NOTHROW(s.advance(ids, 2, t_now, irr, 2, nullptr, 200));
}

TEST_CASE("ragged sample runs give the same bits as the padded rectangle") {
  // Runs of 1-9 samples of a moving sinusoid on every other pixel, as a padded n x 9 rectangle with
  // counts and as one packed array with row starts in reverse pixel order (rows need not follow the ids);
  // the packed runs also through advance_device on the CPU (the engine's host path, ENGINE_CPP §1.7).
  for (const Precision prec : {Precision::kFp64, Precision::kFp32}) {
    const uint32_t W = 16, H = 12;
    const int64_t t_end = int64_t{1} << 23;
    SensorParams p = preset("prophesee_gen3");
    p.hot_fraction = 0.05;
    SensorModel rect(p, W, H, 9, {make_constant_waveform()}, 0, 3, prec);
    SensorModel runs(p, W, H, 9, {make_constant_waveform()}, 0, 3, prec);
    SensorModel host(p, W, H, 9, {make_constant_waveform()}, 0, 2, prec);
    std::vector<float> irr0(W * H);
    for (size_t i = 0; i < irr0.size(); ++i) irr0[i] = 50.0f + static_cast<float>(i % 7);
    rect.init(0, irr0.data());
    runs.init(0, irr0.data());
    host.init(0, irr0.data());
    const size_t m = 9;
    std::vector<uint32_t> ids;
    std::vector<int32_t> counts;
    std::vector<int64_t> t_rect;
    std::vector<float> irr_rect;
    for (uint32_t px = 0; px < W * H; px += 2) {
      ids.push_back(px);
      const auto c = static_cast<int32_t>(1 + px % m);
      counts.push_back(c);
      for (size_t j = 0; j < m; ++j) {
        const int64_t t = j < static_cast<size_t>(c) ? t_end * static_cast<int64_t>(j + 1) / c : 0;
        t_rect.push_back(t);
        irr_rect.push_back(static_cast<float>(60.0 * (1.5 + std::sin(0.3 * px + 2e-6 * static_cast<double>(t)))));
      }
    }
    const size_t n = ids.size();
    std::vector<int64_t> start(n);
    std::vector<int64_t> t_packed;
    std::vector<float> irr_packed;
    for (size_t k = n; k-- > 0;) {
      start[k] = static_cast<int64_t>(t_packed.size());
      for (int32_t j = 0; j < counts[k]; ++j) {
        t_packed.push_back(t_rect[k * m + static_cast<size_t>(j)]);
        irr_packed.push_back(irr_rect[k * m + static_cast<size_t>(j)]);
      }
    }
    const Events a = rect.advance(ids.data(), n, t_rect.data(), irr_rect.data(), m, counts.data(), t_end);
    const Events b = runs.advance_runs(ids.data(), n, t_packed.data(), irr_packed.data(), t_packed.size(),
                                       start.data(), counts.data(), t_end);
    REQUIRE(a.size() > 50);
    CHECK(a.t_cross_ns == b.t_cross_ns);
    CHECK(a.x == b.x);
    CHECK(a.p == b.p);
    CHECK(a.label == b.label);
    CHECK(a.ds == b.ds);
    const std::vector<PixelState> sa = rect.states(), sb = runs.states();
    CHECK(std::memcmp(sa.data(), sb.data(), sa.size() * sizeof(PixelState)) == 0);
    const DeviceEvents raw = host.advance_device(
        {ids.data(), start.data(), counts.data(), n, t_packed.data(), irr_packed.data(), t_packed.size()}, t_end, nullptr);
    CHECK(raw.count == a.size());
    const Events c = host.download_events(raw);
    CHECK(a.t_cross_ns == c.t_cross_ns);
    CHECK(a.t_reported_ns == c.t_reported_ns);
    CHECK(a.x == c.x);
    CHECK(a.y == c.y);
    CHECK(a.p == c.p);
    CHECK(a.label == c.label);
    CHECK(a.flags == c.flags);
    CHECK(a.ds == c.ds);
    const std::vector<PixelState> sc = host.states();
    CHECK(std::memcmp(sa.data(), sc.data(), sa.size() * sizeof(PixelState)) == 0);
    CHECK(host.time_ns() == t_end);
    // A run reaching past the packed samples is refused before any state changes.
    start[0] = static_cast<int64_t>(t_packed.size());
    CHECK_THROWS_AS(runs.advance_runs(ids.data(), n, t_packed.data(), irr_packed.data(), t_packed.size(),
                                      start.data(), counts.data(), 2 * t_end),
                    std::invalid_argument);
    CHECK(runs.time_ns() == t_end);
  }
}

TEST_CASE("load_state refuses a state saved by a different model") {
  SensorParams p = preset("prophesee_gen3");
  SensorModel a(p, 4, 4, 1, {make_constant_waveform()}, 0);
  std::vector<float> irr0(16, 10.0f);
  a.init(0, irr0.data());
  const std::vector<uint8_t> blob = a.save_state();
  CHECK(blob.size() == a.state_bytes());
  SensorModel b(p, 4, 4, 2, {make_constant_waveform()}, 0);
  CHECK_THROWS_AS(b.load_state(blob.data(), blob.size()), std::invalid_argument);
  SensorModel e(p, 4, 4, 1, {make_constant_waveform()}, 1);  // another camera of the rig
  CHECK_THROWS_AS(e.load_state(blob.data(), blob.size()), std::invalid_argument);
  p.theta_on = 0.3;
  SensorModel c(p, 4, 4, 1, {make_constant_waveform()}, 0);
  CHECK_THROWS_AS(c.load_state(blob.data(), blob.size()), std::invalid_argument);
  SensorModel d(preset("prophesee_gen3"), 4, 4, 1, {make_constant_waveform()}, 0);
  CHECK_NOTHROW(d.load_state(blob.data(), blob.size()));
  CHECK(d.initialized());
  // A gain map (set_photometry, D1 vignetting) enters the fingerprint; a unit map leaves it as without one.
  SensorModel g(preset("prophesee_gen3"), 4, 4, 1, {make_constant_waveform()}, 0);
  g.set_photometry(g.params().illuminance_scale, std::vector<float>(16, 1.0f));
  CHECK_NOTHROW(g.load_state(blob.data(), blob.size()));
  std::vector<float> vignette(16, 1.0f);
  vignette[5] = 0.5f;
  g.set_photometry(g.params().illuminance_scale, vignette);
  CHECK_THROWS_AS(g.load_state(blob.data(), blob.size()), std::invalid_argument);
  const std::vector<uint8_t> vignetted = g.save_state();
  CHECK_THROWS_AS(d.load_state(vignetted.data(), vignetted.size()), std::invalid_argument);
  vignette[5] = 0.25f;
  d.set_photometry(d.params().illuminance_scale, vignette);
  CHECK_THROWS_AS(d.load_state(vignetted.data(), vignetted.size()), std::invalid_argument);
  vignette[5] = 0.5f;
  d.set_photometry(d.params().illuminance_scale, vignette);
  CHECK_NOTHROW(d.load_state(vignetted.data(), vignetted.size()));
}

TEST_CASE("the cameras of a rig draw independent maps and noise") {
  const SensorParams p = preset("prophesee_gen3");
  SensorModel left(p, 32, 16, 5, {make_constant_waveform()}, 0, 2), right(p, 32, 16, 5, {make_constant_waveform()}, 1, 2);
  int same_theta = 0;
  for (size_t i = 0; i < left.maps().pixels.size(); ++i) {
    same_theta += left.maps().pixels[i].theta_on == right.maps().pixels[i].theta_on;
  }
  CHECK(same_theta == 0);
  std::vector<float> irr0(32 * 16, 40.0f);  // noise-dominated photocurrent
  left.init(0, irr0.data());
  right.init(0, irr0.data());
  const Events a = left.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, 50'000'000);
  const Events b = right.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, 50'000'000);
  REQUIRE(a.size() > 20);
  CHECK(a.t_cross_ns != b.t_cross_ns);
}

TEST_CASE("non-finite irradiance is refused and out-of-range samples are clamped (DESIGN §8.3 guards)") {
  for (const Precision prec : {Precision::kFp64, Precision::kFp32}) {
    SensorModel s(preset("prophesee_gen3"), 2, 2, 3, {make_constant_waveform()}, 0, 1, prec);
    std::vector<float> irr0(4, 100.0f);
    irr0[2] = std::numeric_limits<float>::infinity();
    CHECK_THROWS_WITH_AS(s.init(0, irr0.data()), doctest::Contains("non-finite"), std::invalid_argument);
    CHECK_FALSE(s.initialized());
    irr0[2] = 1e38f;  // finite but far above irr_max: clamped to it
    s.init(0, irr0.data());
    CHECK(s.stats().input_clamps == 1);
    const uint32_t ids[2] = {0, 3};
    const int64_t t[4] = {1'000'000, 2'000'000, 1'000'000, 2'000'000};
    for (const float bad : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
      const float irr[4] = {bad, 100.0f, 100.0f, 100.0f};
      CHECK_THROWS_WITH_AS(s.advance(ids, 2, t, irr, 2, nullptr, 5'000'000), doctest::Contains("non-finite"),
                           std::invalid_argument);
    }
    CHECK(s.time_ns() == 0);
    CHECK(s.initialized());  // a refused input changes nothing
    // A sample of 1e38 (it overflowed the fp32 kernel before the clamp) and a negative one: the run
    // completes with finite states, no quarantine and no fallback sub-step.
    const float irr[4] = {1e38f, 100.0f, -5.0f, 100.0f};
    const Events ev = s.advance(ids, 2, t, irr, 2, nullptr, 20'000'000);
    CHECK(s.stats().input_clamps == 3);
    CHECK(s.stats().quarantines == 0);
    CHECK(s.stats().fallbacks == 0);
    CHECK(ev.size() > 10);  // the 1e7-fold flash and its decay cross many levels
    for (const PixelState& ps : s.states()) {
      CHECK(std::isfinite(ps.z));
      CHECK(std::isfinite(ps.s3));
      CHECK(std::isfinite(ps.ln_a1));
    }
  }
}

TEST_CASE("an advance that fails part-way leaves the model uninitialized until load_state (H9)") {
  // One thread, one block of 64 pixels: the quiet ones (reference at their initial state, no leak) advance first,
  // then the last pixel's step fires, and its first event fails to allocate. The pixels are part-way, so a retry of
  // the same call would integrate them twice: refused until load_state.
  SensorParams p = preset("prophesee_gen3");
  p.leak_rate_hz = 0;
  p.initial_phase = InitialPhase::kZero;
  SensorModel s(p, 8, 8, 4, {make_constant_waveform()}, 0, 1);
  std::vector<float> irr0(64, 1e5f);
  irr0[63] = 2.0f;
  s.init(0, irr0.data());
  const std::vector<uint8_t> saved = s.save_state();
  const uint32_t id = 63;
  const int64_t t = 1'000'000;
  const float step = 1e5f;
  bool threw = false;
  g_fail_event_alloc = true;
  try {
    s.advance(&id, 1, &t, &step, 1, nullptr, 20'000'000);
  } catch (const std::bad_alloc&) {
    threw = true;
  }
  g_fail_event_alloc = false;
  REQUIRE(threw);
  const std::vector<PixelState> part = s.states();
  const size_t payload = part.size() * sizeof(PixelState);
  CHECK(std::memcmp(saved.data() + saved.size() - payload, part.data(), payload) != 0);  // failed mid-kernel
  CHECK_FALSE(s.initialized());
  CHECK_THROWS_AS(s.advance(&id, 1, &t, &step, 1, nullptr, 20'000'000), std::logic_error);
  s.load_state(saved.data(), saved.size());
  CHECK(s.initialized());
  CHECK(s.time_ns() == 0);
  const Events ev = s.advance(&id, 1, &t, &step, 1, nullptr, 20'000'000);
  REQUIRE(ev.size() > 0);
  CHECK(std::all_of(ev.x.begin(), ev.x.end(), [](uint16_t x) { return x == 7; }));  // the step alone fires
  CHECK(std::all_of(ev.y.begin(), ev.y.end(), [](uint16_t y) { return y == 7; }));
}

TEST_CASE("the noise-table position of a NaN rate is the first node (no out-of-bounds index)") {
  const double data[4] = {0, 0, 0, 0};
  const NoiseTableView<double> v{data, 4, 1, 0, -4.6, 64.0};
  CHECK(noise_pos(v, std::numeric_limits<double>::quiet_NaN()).i == 0);
  CHECK(noise_pos(v, -std::numeric_limits<double>::infinity()).i == 0);
  CHECK(noise_pos(v, std::numeric_limits<double>::infinity()).i == 2);
  CHECK(noise_pos(v, std::numeric_limits<double>::infinity()).f == 1.0);
}

TEST_CASE("a pixel gain multiplies the scale wherever the input forms (set_photometry; D1 vignetting)") {
  // Gain 1/2 at scale 400 is scale 200 exactly (a power of two), so the events and states are those of scale 200 bit
  // for bit if and only if every term of the input (init, pieces, chord and curvature bounds, the instantaneous
  // response) takes the pixel's scale; on both photoreceptors, with a 120 Hz group, on the CPU and the CUDA device.
  const uint32_t W = 12, H = 8;
  const int64_t t_end = int64_t{1} << 23;
  const std::vector<Waveform> groups{make_constant_waveform(), make_fourier_waveform(120.0, {0.2}, {0.0})};
  std::vector<float> irr0(W * H * 2);
  for (size_t i = 0; i < irr0.size(); ++i) irr0[i] = 0.05f + 0.01f * static_cast<float>(i % 5);
  std::vector<uint32_t> ids;
  std::vector<int64_t> t;
  std::vector<float> irr;
  const size_t m = 6;
  for (uint32_t px = 0; px < W * H; ++px) {
    ids.push_back(px);
    for (size_t j = 0; j < m; ++j) {
      t.push_back(t_end * static_cast<int64_t>(j + 1) / static_cast<int64_t>(m));
      irr.push_back(static_cast<float>(0.1 * (1.2 + std::sin(0.4 * px + 0.9 * static_cast<double>(j)))));
      irr.push_back(0.03f);
    }
  }
  for (const Device dev : {Device::kCpu, Device::kCuda}) {
    if (dev == Device::kCuda && !sensor_cuda_available()) continue;
    for (const Photoreceptor pr : {Photoreceptor::kCascade, Photoreceptor::kInstantaneous}) {
      INFO("device " << static_cast<int>(dev) << ", photoreceptor " << static_cast<int>(pr));
      const Precision prec = dev == Device::kCuda ? Precision::kFp32 : Precision::kFp64;
      SensorParams p = preset("prophesee_gen3");
      p.photoreceptor = pr;
      p.illuminance_scale = 200;
      SensorModel ref(p, W, H, 3, groups, 0, 2, prec, dev);
      p.illuminance_scale = 400;
      SensorModel half(p, W, H, 3, groups, 0, 2, prec, dev);
      half.set_photometry(400, std::vector<float>(W * H, 0.5f));
      CHECK(half.params().illuminance_scale == 400);
      CHECK(half.irr_max() == ref.irr_max());
      ref.init(0, irr0.data());
      half.init(0, irr0.data());
      const Events a = ref.advance(ids.data(), ids.size(), t.data(), irr.data(), m, nullptr, t_end);
      const Events b = half.advance(ids.data(), ids.size(), t.data(), irr.data(), m, nullptr, t_end);
      REQUIRE(a.size() > 20);
      CHECK(a.t_cross_ns == b.t_cross_ns);
      CHECK(a.x == b.x);
      CHECK(a.p == b.p);
      CHECK(a.ds == b.ds);
      const std::vector<PixelState> sa = ref.states(), sb = half.states();
      CHECK(std::memcmp(sa.data(), sb.data(), sa.size() * sizeof(PixelState)) == 0);
    }
  }
  SensorModel m0(preset("prophesee_gen3"), W, H, 3, groups, 0, 1);
  CHECK_THROWS_AS(m0.set_photometry(1.0, std::vector<float>(3, 1.0f)), std::invalid_argument);
  CHECK_THROWS_AS(m0.set_photometry(1.0, std::vector<float>(W * H, 0.0f)), std::invalid_argument);
  CHECK_THROWS_AS(m0.set_photometry(-1.0, {}), std::invalid_argument);
}

TEST_CASE("veiling glare adds g E_g(t) per group, linear over each call and continuous across calls (D13 item 3)") {
  // Model A has glare g = 1/8; model B has none and is given A's glare in its samples: every pixel listed, each sample
  // plus g E_g(t) / gain, E_g linear over each call from the frame mean of gain x the inputs A's pixels hold at its
  // start (init: the frame's own) to that at its end (each listed pixel's last sample, the others held), and a
  // sample at the call's end where A holds. Dyadic values (gains 1/2 and 1, samples on 2^20 ns, irradiances on 1/64)
  // make every input term exact in fp64, so the events are equal bit for bit if and only if each of A's input
  // evaluations is gain E_p + g E_g(t) (held pixels and the held tail of a run included). Two calls; a group dark in
  // some pixels' own samples is lit by the glare; cascade and instantaneous; a 120 Hz group modulates its glare.
  const uint32_t W = 16, H = 8, N = W * H;
  const int G = 2;
  const int64_t kW = int64_t{1} << 23, kStep = int64_t{1} << 20;
  const double g = 0.125;
  const std::vector<Waveform> groups{make_constant_waveform(), make_fourier_waveform(120.0, {0.25}, {0.0})};
  std::vector<float> gain(N), irr0(N * G);
  for (uint32_t p = 0; p < N; ++p) {
    gain[p] = p % 2 ? 0.5f : 1.0f;
    irr0[p * G] = static_cast<float>(1 + (p * 7) % 17) / 64.0f;
    irr0[p * G + 1] = p % 5 == 0 ? 0.0f : static_cast<float>(1 + (p * 3) % 11) / 64.0f;
  }
  double e0[2] = {0, 0};
  for (uint32_t p = 0; p < N; ++p) {
    for (int c = 0; c < G; ++c) e0[c] += static_cast<double>(gain[p]) * static_cast<double>(irr0[p * G + c]);
  }
  for (double& e : e0) e /= N;
  struct Run {
    std::vector<uint32_t> ids;
    std::vector<int64_t> t, start;
    std::vector<int32_t> count;
    std::vector<float> irr;
    void add(uint32_t p, const std::vector<std::pair<int64_t, std::array<float, 2>>>& s) {
      ids.push_back(p);
      start.push_back(static_cast<int64_t>(t.size()));
      count.push_back(static_cast<int32_t>(s.size()));
      for (const auto& [ti, v] : s) {
        t.push_back(ti);
        irr.insert(irr.end(), v.begin(), v.end());
      }
    }
  };
  for (const Photoreceptor pr : {Photoreceptor::kCascade, Photoreceptor::kInstantaneous}) {
    INFO("photoreceptor " << static_cast<int>(pr));
    SensorParams pa = preset("prophesee_gen3");
    pa.photoreceptor = pr;
    pa.illuminance_scale = 400;
    SensorParams pb = pa;
    pa.veiling_glare = g;
    SensorModel a(pa, W, H, 7, groups, 0, 2), b(pb, W, H, 7, groups, 0, 2);
    a.set_photometry(400, gain);
    b.set_photometry(400, gain);
    std::vector<float> irr0b(irr0);
    for (uint32_t p = 0; p < N; ++p) {
      for (int c = 0; c < G; ++c) irr0b[p * G + c] += static_cast<float>(g * e0[c] / static_cast<double>(gain[p]));
    }
    a.init(0, irr0.data());
    b.init(0, irr0b.data());
    CHECK(a.glare_frame_mean() == std::vector<double>{e0[0], e0[1]});
    CHECK(b.glare_frame_mean().empty());
    std::vector<float> held(irr0);  // A's raw held input
    std::vector<double> from{e0[0], e0[1]};
    size_t total = 0;
    for (int call = 0; call < 2; ++call) {
      const int64_t t0 = call * kW, t1 = t0 + kW;
      // A's runs: listed pixels every step, or every other one (a held tail); then the means of what A holds at t1.
      Run ra, rb;
      std::vector<std::vector<std::pair<int64_t, std::array<float, 2>>>> sa(N);
      for (uint32_t p = 0; p < N; ++p) {
        if (p % 3 == static_cast<uint32_t>(call)) continue;
        for (int64_t j = 1; j <= 8; j += (p % 4 == 0 ? 2 : 1)) {
          const std::array<float, 2> v{static_cast<float>(1 + (p + 3 * j) % 9) / 16.0f,
                                       p % 5 == 0 ? 0.0f : static_cast<float>(1 + (p + 5 * j) % 7) / 32.0f};
          sa[p].push_back({t0 + j * kStep, v});
        }
        ra.add(p, sa[p]);
      }
      std::vector<float> end_held(held);
      for (uint32_t p = 0; p < N; ++p) {
        if (!sa[p].empty()) std::copy_n(sa[p].back().second.begin(), G, end_held.begin() + p * G);
      }
      std::vector<double> end{0, 0};
      for (uint32_t p = 0; p < N; ++p) {
        for (int c = 0; c < G; ++c) end[c] += static_cast<double>(gain[p]) * static_cast<double>(end_held[p * G + c]);
      }
      for (double& e : end) e /= N;
      const auto glare = [&](int64_t t, int c, uint32_t p) {
        const double f = static_cast<double>(t - t0) / static_cast<double>(kW);
        return static_cast<float>(((1 - f) * g * from[c] + f * g * end[c]) / static_cast<double>(gain[p]));
      };
      for (uint32_t p = 0; p < N; ++p) {
        std::vector<std::pair<int64_t, std::array<float, 2>>> sb;
        for (const auto& [ti, v] : sa[p]) sb.push_back({ti, {v[0] + glare(ti, 0, p), v[1] + glare(ti, 1, p)}});
        if (sb.empty() || sb.back().first < t1) {
          sb.push_back({t1, {end_held[p * G] + glare(t1, 0, p), end_held[p * G + 1] + glare(t1, 1, p)}});
        }
        rb.add(p, sb);
      }
      held = end_held;
      CHECK(end != from);
      const Events ea = a.advance_runs(ra.ids.data(), ra.ids.size(), ra.t.data(), ra.irr.data(), ra.t.size(),
                                       ra.start.data(), ra.count.data(), t1);
      const Events eb = b.advance_runs(rb.ids.data(), rb.ids.size(), rb.t.data(), rb.irr.data(), rb.t.size(),
                                       rb.start.data(), rb.count.data(), t1);
      total += ea.size();
      CHECK(ea.t_cross_ns == eb.t_cross_ns);
      CHECK(ea.x == eb.x);
      CHECK(ea.y == eb.y);
      CHECK(ea.p == eb.p);
      CHECK(ea.label == eb.label);
      CHECK(ea.ds == eb.ds);
      CHECK(ea.dn == eb.dn);
      CHECK(ea.dl == eb.dl);
      CHECK(a.glare_frame_mean() == end);  // the next call starts here: no step at the boundary
      from = end;
    }
    CHECK(total > 50);
  }
  // The frame mean is gain x irradiance over the pixels; none without glare.
  SensorModel m(preset("prophesee_gen3"), W, H, 7, groups, 0, 1);
  m.set_photometry(1.0, gain);
  CHECK(m.frame_mean(irr0.data()) == std::vector<double>{e0[0], e0[1]});
  m.init(0, irr0.data());
  CHECK(m.glare_frame_mean().empty());
}

TEST_CASE("veiling glare: a state loaded into a fresh model advances as the uninterrupted model does") {
  // The frame means at the model time are those of the held inputs, so a saved state carries them: a model that was
  // never initialized (sensor-only re-simulation from a checkpoint) and one initialized on another frame both continue
  // bit for bit; on each device.
  const uint32_t W = 16, H = 12, N = W * H;
  const std::vector<Waveform> groups{make_constant_waveform(), make_fourier_waveform(120.0, {0.2}, {0.0})};
  std::vector<float> irr0(N * 2), other(N * 2, 0.7f);
  for (size_t i = 0; i < irr0.size(); ++i) irr0[i] = i % 7 == 0 ? 0.0f : 0.02f + 0.01f * static_cast<float>(i % 5);
  const int64_t kW = int64_t{1} << 23;
  const size_t m = 4;
  const auto runs = [&](int call, std::vector<uint32_t>& ids, std::vector<int64_t>& t, std::vector<float>& irr) {
    for (uint32_t px = static_cast<uint32_t>(call); px < N; px += 3) {  // a third of the pixels listed, the rest held
      ids.push_back(px);
      for (size_t j = 0; j < m; ++j) {
        t.push_back(call * kW + kW * static_cast<int64_t>(j + 1) / static_cast<int64_t>(m));
        irr.push_back(static_cast<float>(0.1 * (1.2 + std::sin(0.4 * px + 0.9 * static_cast<double>(j + call)))));
        irr.push_back(0.03f + 0.01f * static_cast<float>(call));
      }
    }
  };
  SensorParams p = preset("prophesee_gen3");
  p.illuminance_scale = 400;
  p.veiling_glare = 0.05;
  for (const Device dev : {Device::kCpu, Device::kCuda}) {
    if (dev == Device::kCuda && !sensor_cuda_available()) continue;
    INFO("device " << static_cast<int>(dev));
    const Precision prec = dev == Device::kCuda ? Precision::kFp32 : Precision::kFp64;
    SensorModel a(p, W, H, 5, groups, 0, 2, prec, dev), fresh(p, W, H, 5, groups, 0, 2, prec, dev),
        other_init(p, W, H, 5, groups, 0, 2, prec, dev);
    a.init(0, irr0.data());
    other_init.init(0, other.data());
    std::vector<uint32_t> ids;
    std::vector<int64_t> t;
    std::vector<float> irr;
    runs(0, ids, t, irr);
    a.advance(ids.data(), ids.size(), t.data(), irr.data(), m, nullptr, kW);
    const std::vector<uint8_t> state = a.save_state();
    fresh.load_state(state.data(), state.size());
    other_init.load_state(state.data(), state.size());
    CHECK(fresh.glare_frame_mean() == a.glare_frame_mean());
    CHECK(other_init.glare_frame_mean() == a.glare_frame_mean());
    ids.clear(), t.clear(), irr.clear();
    runs(1, ids, t, irr);
    const Events ea = a.advance(ids.data(), ids.size(), t.data(), irr.data(), m, nullptr, 2 * kW);
    REQUIRE(ea.size() > 20);
    for (SensorModel* b : {&fresh, &other_init}) {
      const Events eb = b->advance(ids.data(), ids.size(), t.data(), irr.data(), m, nullptr, 2 * kW);
      CHECK(ea.t_cross_ns == eb.t_cross_ns);
      CHECK(ea.x == eb.x);
      CHECK(ea.p == eb.p);
      CHECK(ea.ds == eb.ds);
      CHECK(b->glare_frame_mean() == a.glare_frame_mean());
    }
  }
}

TEST_CASE("veiling glare: the CUDA device equals the CPU fp32 kernel bit for bit") {
  if (!sensor_cuda_available()) return;
  const uint32_t W = 24, H = 16, N = W * H;
  const std::vector<Waveform> groups{make_constant_waveform(), make_fourier_waveform(120.0, {0.2}, {0.0})};
  std::vector<float> irr0(N * 2);
  for (size_t i = 0; i < irr0.size(); ++i) irr0[i] = i % 7 == 0 ? 0.0f : 0.02f + 0.01f * static_cast<float>(i % 5);
  std::vector<uint32_t> ids;
  std::vector<int64_t> t;
  std::vector<float> irr;
  const int64_t t_end = int64_t{1} << 23;
  const size_t m = 4;
  for (uint32_t px = 0; px < N; px += 2) {  // half the pixels listed, the rest held
    ids.push_back(px);
    for (size_t j = 0; j < m; ++j) {
      t.push_back(t_end * static_cast<int64_t>(j + 1) / static_cast<int64_t>(m));
      irr.push_back(static_cast<float>(0.1 * (1.2 + std::sin(0.4 * px + 0.9 * static_cast<double>(j)))));
      irr.push_back(0.03f);
    }
  }
  // Under both noise diffusions: the GPU's frozen glare kernel compiles the transient paths out (pixel.h kP2, P2-E E4),
  // its transient glare kernel keeps them, as the CPU's kernels do.
  std::vector<int64_t> frozen_t;
  for (const NoiseDiffusion diffusion : {NoiseDiffusion::kFrozen, NoiseDiffusion::kTransient}) {
    INFO("noise_diffusion " << static_cast<int>(diffusion));
    SensorParams p = preset("prophesee_gen3");
    p.illuminance_scale = 400;
    p.veiling_glare = 0.02;
    p.noise_diffusion = diffusion;
    SensorModel cpu(p, W, H, 3, groups, 0, 2, Precision::kFp32, Device::kCpu);
    SensorModel gpu(p, W, H, 3, groups, 0, 2, Precision::kFp32, Device::kCuda);
    Events e[2];
    for (int k = 0; k < 2; ++k) {
      SensorModel& s = k ? gpu : cpu;
      s.init(0, irr0.data());
      e[k] = s.advance(ids.data(), ids.size(), t.data(), irr.data(), m, nullptr, t_end);
    }
    REQUIRE(e[0].size() > 20);
    CHECK(e[0].t_cross_ns == e[1].t_cross_ns);
    CHECK(e[0].x == e[1].x);
    CHECK(e[0].p == e[1].p);
    CHECK(e[0].ds == e[1].ds);
    const std::vector<PixelState> a = cpu.states(), b = gpu.states();
    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(PixelState)) == 0);
    CHECK(cpu.glare_frame_mean() == gpu.glare_frame_mean());  // the device's held inputs and run ends: the same means
    if (diffusion == NoiseDiffusion::kFrozen)
      frozen_t = e[0].t_cross_ns;
    else
      CHECK(e[0].t_cross_ns != frozen_t);  // the transient run took the rho path
  }
}

TEST_CASE("pixel dark currents below the kernel precision's floor are refused (M27)") {
  // The floor sqrt(min normal / (k 1 ns)): 1.1e-15 fA in fp32 and 1.5e-150 fA in fp64 at k = 1e16. Below it fp32 gave
  // z = inf and a quarantine per sub-step (0/0 in z_step's small-slope branch), fp64 a solver fallback per sub-step.
  const auto build = [](double dark_fA, double temperature_K, Precision precision) {
    SensorParams p = preset("prophesee_gen3");
    p.dark_current_fA = dark_fA;
    p.temperature_K = temperature_K;
    SensorModel s(p, 4, 4, 1, {make_constant_waveform()}, 0, 1, precision);
  };
  CHECK_NOTHROW(build(preset("prophesee_gen3").dark_current_fA, preset("prophesee_gen3").temperature_K, Precision::kFp32));
  for (const auto [dark, temperature] : {std::pair(1e-50, 298.15), std::pair(0.21, 70.0)}) {
    CHECK_THROWS_WITH_AS(build(dark, temperature, Precision::kFp32), doctest::Contains("fp32 kernel's floor"),
                         std::invalid_argument);
    CHECK_NOTHROW(build(dark, temperature, Precision::kFp64));
  }
  CHECK_THROWS_WITH_AS(build(0.21, 10.0, Precision::kFp64), doctest::Contains("fp64 kernel's floor"),
                       std::invalid_argument);
}

#if BALLA_SENSOR_CENSUS
TEST_CASE("the sensor census classes every whole cell once and counts the draws consistently (census.h)") {
  // A census build only (BALLASIM_SENSOR_CENSUS): half the pixels lit by a constant group (quiet), half by a 120 Hz one
  // (the orbit cell's case), held, then a step on the flicker pixels inside a window. The counters do not depend on
  // the thread count, every whole cell has one class, and the step's cells fail by sample and then by the hold proxy.
  const uint32_t W = 16, H = 8, n = W * H;
  const int64_t win = int64_t{1} << 23;
  const std::vector<Waveform> groups{make_constant_waveform(), make_fourier_waveform(120.0, {0.02}, {0.0})};
  std::vector<float> irr0(n * 2, 0.0f);
  for (uint32_t p = 0; p < n; ++p) irr0[p * 2 + (p % 2)] = 0.5f;
  std::vector<uint32_t> ids;
  std::vector<int64_t> t;
  std::vector<float> irr;
  for (uint32_t p = 1; p < n; p += 2) {
    ids.push_back(p);
    t.push_back(9 * win + win / 2);
    irr.push_back(0.0f);
    irr.push_back(0.8f);
  }
  SensorParams prm = preset("prophesee_gen3");
  prm.noise_diffusion = NoiseDiffusion::kTransient;
  prm.illuminance_scale = 400;  // 200 fA: k Ibar = 2000 / s, so a hold settles within a few ms
  std::array<std::array<uint64_t, kCensusCounters>, 3> got{};
  for (const int threads : {1, 4}) {
    INFO("threads " << threads);
    SensorModel s(prm, W, H, 5, groups, 0, threads, Precision::kFp32);
    s.init(0, irr0.data());
    s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, 9 * win);  // 75 ms held
    for (int k = 0; k < 3; ++k) {
      s.reset_stats();
      if (k == 0) s.advance(ids.data(), ids.size(), t.data(), irr.data(), 1, nullptr, 10 * win);
      else s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, (10 + k) * win);
      const uint64_t* c = s.stats().census.n;
      if (threads == 1) {
        for (int i = 0; i < kCensusCounters; ++i) got[k][i] = c[i];
      } else {
        for (int i = 0; i < kCensusCounters; ++i) CHECK(got[k][i] == c[i]);
      }
      uint64_t classes = c[kCnQuiet] + c[kCnEligibleLean] + c[kCnEligibleGeneral] + c[kCnRejUnsettled];
      for (int i = kCnRejNoise; i <= kCnRejHpf; ++i) classes += c[i];
      CHECK(c[kCnCells] == n * 128);
      CHECK(classes == c[kCnCells]);
      CHECK(c[kCnCheap] == c[kCnEligibleLean] + c[kCnEligibleGeneral] + c[kCnRejUnsettled]);
      CHECK(c[kCnQuiet] >= n / 2 * 128 * 9 / 10);  // the constant pixels, settled (less cells with a noise event)
      CHECK(c[kCnRejStatic] + c[kCnQuiet] == n / 2 * 128);
      CHECK(c[kCnEligibleWithin] <= c[kCnWithinRadius]);
      CHECK(c[kCnFactorWalk] <= c[kCnBridgesWalkRho]);
      CHECK(c[kCnFactorCell] <= c[kCnBridgesWalkRho] + c[kCnBridgesN3Rho]);
      CHECK(c[kCnNodesCell] <= c[kCnBridgesWalk] + c[kCnBridgesN3]);
      if (k == 0) {
        CHECK(c[kCnRejSample] == n / 2 * 64);  // the ramp from the held input to the sample at mid-window
        CHECK(c[kCnRejUnsettled] > 0);         // then a short hold
        CHECK(c[kCnEligibleLean] == 0);        // no cell of the call is reachable by the lean pass after the ramp
        CHECK(c[kCnCellsRho] > 0);             // the ramp's transient cells
        CHECK(c[kCnDwRamp] > 0);
      } else if (k == 2) {
        CHECK(c[kCnRejSample] == 0);
        CHECK(c[kCnEligibleLean] > 0);         // a held call: lean from its start while the cells are eligible
      }
    }
  }
}
#endif
