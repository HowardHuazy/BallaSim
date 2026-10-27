// The GPU path runs the fp32 kernel of the CPU: states and events must agree bit for bit
// (DESIGN §12.1 L1), however the launches batch pixels, slice time or overflow the event buffer. A
// launch over the 50 ms cap of a GPU that also drives the desktop (DESIGN §4.2) is reported (the
// desktop and other jobs share the GPU) and one over 100 ms fails (a planner that stops slicing takes
// 106 ms or more). The GPU cases form the test suite "cuda", skipped without CUDA or without a device
// (main.cpp).
#include <doctest/doctest.h>
#if BALLASIM_SENSOR_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "sensor/sensor_model.h"

using namespace balla::sensor;

namespace {

constexpr int64_t kW23 = int64_t{1} << 23;

// Ragged moving-grating runs for every other pixel over [t0, t1): group 0 moves, group 1 (when
// present) is a constant second light.
struct Runs {
  std::vector<uint32_t> ids;
  std::vector<int64_t> t;
  std::vector<float> irr;
  std::vector<int32_t> counts;
  size_t m = 9;
};

Runs grating_runs(uint32_t n_pixels, int groups, int64_t t0, int64_t t1) {
  Runs r;
  for (uint32_t px = 0; px < n_pixels; px += 2) {
    r.ids.push_back(px);
    const int32_t c = 1 + static_cast<int32_t>(px % r.m);
    r.counts.push_back(c);
    for (size_t j = 0; j < r.m; ++j) {
      r.t.push_back(j < static_cast<size_t>(c) ? t0 + (t1 - t0) * static_cast<int64_t>(j + 1) / c : 0);
      const double phase = 0.05 * px + 1e-7 * static_cast<double>(r.t.back());
      r.irr.push_back(static_cast<float>(200.0 * (1.2 + std::sin(phase))));
      for (int g = 1; g < groups; ++g) r.irr.push_back(static_cast<float>(10.0 * (px % 5) * g));
    }
  }
  return r;
}

Events advance(SensorModel& s, const Runs& r, int64_t t_end) {
  return s.advance(r.ids.data(), r.ids.size(), r.t.data(), r.irr.data(), r.m, r.counts.data(), t_end);
}

void check_equal(const Events& a, const Events& b) {
  REQUIRE(a.size() == b.size());
  CHECK(a.t_cross_ns == b.t_cross_ns);
  CHECK(a.t_reported_ns == b.t_reported_ns);
  CHECK(a.x == b.x);
  CHECK(a.y == b.y);
  CHECK(a.p == b.p);
  CHECK(a.label == b.label);
  CHECK(a.flags == b.flags);
  CHECK(a.ds == b.ds);
  CHECK(a.dn == b.dn);
  CHECK(a.dl == b.dl);
}

void check_same_states(const SensorModel& a, const SensorModel& b) {
  const std::vector<PixelState> sa = a.states(), sb = b.states();
  REQUIRE(sa.size() == sb.size());
  CHECK(std::memcmp(sa.data(), sb.data(), sa.size() * sizeof(PixelState)) == 0);
  CHECK(a.stats().cells == b.stats().cells);
  CHECK(a.stats().substeps == b.stats().substeps);
}

#if BALLASIM_SENSOR_CUDA
template <class T>
T* to_device(const std::vector<T>& v) {
  T* p = nullptr;
  REQUIRE(cudaMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T)) == cudaSuccess);
  if (!v.empty()) REQUIRE(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
  return p;
}

void free_device(std::initializer_list<void*> ptrs) {
  for (void* q : ptrs) cudaFree(q);
}
#endif

}  // namespace

TEST_CASE("zero-length runs equal unlisted pixels on the CPU and the GPU") {
  // Every third listed pixel takes no sample (one of them from row_start = n_samples): it advances
  // like a pixel the call does not list.
  const uint32_t W = 32, H = 24;
  const std::vector<Waveform> groups = {make_constant_waveform(), make_fourier_waveform(120.0, {0.3}, {0.5})};
  SensorParams p = preset("prophesee_gen3");
  p.hot_fraction = 0.02;
  SensorModel with(p, W, H, 23, groups, 0, 4, Precision::kFp32, Device::kCpu);
  SensorModel without(p, W, H, 23, groups, 0, 4, Precision::kFp32, Device::kCpu);
  std::optional<SensorModel> gpu, dev;
  std::vector<float> irr0(W * H * 2);
  for (size_t i = 0; i < W * H; ++i) {
    irr0[2 * i] = 20.0f + 5.0f * static_cast<float>(i % 41);
    irr0[2 * i + 1] = 30.0f * static_cast<float>(i % 3);
  }
  with.init(0, irr0.data());
  without.init(0, irr0.data());
  if (sensor_cuda_available()) {
    for (auto* m : {&gpu, &dev}) {
      m->emplace(p, W, H, 23, groups, 0, 1, Precision::kFp32, Device::kCuda);
      (*m)->init(0, irr0.data());
    }
  }
  for (int w = 0; w < 2; ++w) {
    const Runs r = grating_runs(W * H, 2, w * kW23, (w + 1) * kW23);
    const int64_t t_end = (w + 1) * kW23;
    std::vector<int64_t> start(r.ids.size()), start_b;
    std::vector<int32_t> counts = r.counts, counts_b;
    std::vector<uint32_t> ids_b;
    for (size_t k = 0; k < r.ids.size(); ++k) {
      start[k] = static_cast<int64_t>(k * r.m);
      if (k % 3 == 0) {
        counts[k] = 0;
        if (k == 3) start[k] = static_cast<int64_t>(r.t.size());
      } else {
        ids_b.push_back(r.ids[k]);
        start_b.push_back(start[k]);
        counts_b.push_back(counts[k]);
      }
    }
    const Events a = with.advance_runs(r.ids.data(), r.ids.size(), r.t.data(), r.irr.data(), r.t.size(), start.data(),
                                       counts.data(), t_end);
    CHECK(a.size() > 100);
    check_equal(a, without.advance_runs(ids_b.data(), ids_b.size(), r.t.data(), r.irr.data(), r.t.size(),
                                        start_b.data(), counts_b.data(), t_end));
    check_same_states(with, without);
    if (gpu) {
      check_equal(a, gpu->advance_runs(r.ids.data(), r.ids.size(), r.t.data(), r.irr.data(), r.t.size(), start.data(),
                                       counts.data(), t_end));
      check_same_states(with, *gpu);
    }
#if BALLASIM_SENSOR_CUDA
    if (dev) {
      uint32_t* d_ids = to_device(r.ids);
      int64_t* d_start = to_device(start);
      int32_t* d_count = to_device(counts);
      int64_t* d_t = to_device(r.t);
      float* d_irr = to_device(r.irr);
      REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
      const DeviceRuns runs{d_ids, d_start, d_count, r.ids.size(), d_t, d_irr, r.t.size()};
      check_equal(a, dev->download_events(dev->advance_device(runs, t_end, nullptr)));
      check_same_states(with, *dev);
      free_device({d_ids, d_start, d_count, d_t, d_irr});
    }
#endif
  }
}

// The GPU cases (main.cpp excludes them without a CUDA sensor kernel or a device).
TEST_SUITE_BEGIN("cuda");

TEST_CASE("CUDA advance is bitwise identical to the CPU fp32 kernel") {
  // Under both noise diffusions: the GPU's frozen kernels compile the transient paths out (pixel.h kP2, P2-E E4), the
  // CPU's keep them.
  for (const NoiseDiffusion diffusion : {NoiseDiffusion::kFrozen, NoiseDiffusion::kTransient}) {
    INFO("noise_diffusion " << static_cast<int>(diffusion));
    SensorParams p = preset("prophesee_gen3");
    p.hot_fraction = 0.02;
    p.noise_diffusion = diffusion;
    const uint32_t W = 48, H = 40;
    const std::vector<Waveform> groups = {make_constant_waveform(), make_fourier_waveform(120.0, {0.3}, {0.5})};
    SensorModel cpu(p, W, H, 31, groups, 0, 4, Precision::kFp32, Device::kCpu);
    SensorModel gpu(p, W, H, 31, groups, 0, 1, Precision::kFp32, Device::kCuda);
    std::vector<float> irr0(W * H * 2);
    for (size_t i = 0; i < W * H; ++i) {
      irr0[2 * i] = 20.0f + 5.0f * static_cast<float>(i % 41);
      irr0[2 * i + 1] = 30.0f * static_cast<float>(i % 3);
    }
    cpu.init(0, irr0.data());
    gpu.init(0, irr0.data());
    // A moving grating on half of the pixels, ragged runs, two windows.
    for (int w = 0; w < 2; ++w) {
      const Runs r = grating_runs(W * H, 2, w * kW23, (w + 1) * kW23);
      const Events a = advance(cpu, r, (w + 1) * kW23);
      CHECK(a.size() > 100);
      check_equal(a, advance(gpu, r, (w + 1) * kW23));
    }
    check_same_states(cpu, gpu);
  }
}

TEST_CASE("CUDA ragged runs (advance_runs) are bitwise identical to the CPU fp32 kernel") {
  const uint32_t W = 40, H = 32;
  const std::vector<Waveform> groups = {make_constant_waveform(), make_fourier_waveform(120.0, {0.3}, {0.5})};
  SensorModel cpu(preset("prophesee_gen3"), W, H, 17, groups, 0, 4, Precision::kFp32, Device::kCpu);
  SensorModel gpu(preset("prophesee_gen3"), W, H, 17, groups, 0, 1, Precision::kFp32, Device::kCuda);
  std::vector<float> irr0(W * H * 2, 60.0f);
  cpu.init(0, irr0.data());
  gpu.init(0, irr0.data());
  const Runs r = grating_runs(W * H, 2, 0, kW23);
  // Pack the padded rows: row k keeps its counts[k] samples.
  std::vector<int64_t> start(r.ids.size()), t;
  std::vector<float> irr;
  for (size_t k = 0; k < r.ids.size(); ++k) {
    start[k] = static_cast<int64_t>(t.size());
    for (int32_t j = 0; j < r.counts[k]; ++j) {
      const size_t at = k * r.m + static_cast<size_t>(j);
      t.push_back(r.t[at]);
      irr.insert(irr.end(), r.irr.begin() + static_cast<std::ptrdiff_t>(2 * at),
                 r.irr.begin() + static_cast<std::ptrdiff_t>(2 * at + 2));
    }
  }
  const Events a = advance(cpu, r, kW23);
  const Events b = gpu.advance_runs(r.ids.data(), r.ids.size(), t.data(), irr.data(), t.size(), start.data(),
                                    r.counts.data(), kW23);
  CHECK(a.size() > 100);
  check_equal(a, b);
  check_same_states(cpu, gpu);
}

TEST_CASE("pixel batches, time slices and event-buffer overflows leave the GPU result bitwise unchanged") {
  // Noise on: slices on the 2^16 ns lattice; noise off: on 2^23 ns segments (pixel.h, slice_grain).
  for (const bool noise : {true, false}) {
    SensorParams p = preset("prophesee_gen3");
    p.hot_fraction = 0.02;
    p.noise = noise;
    const uint32_t W = 48, H = 40;
    const std::vector<Waveform> groups = {make_constant_waveform(), make_fourier_waveform(120.0, {0.3}, {0.5})};
    SensorModel cpu(p, W, H, 32, groups, 0, 4, Precision::kFp32, Device::kCpu);
    SensorModel gpu(p, W, H, 32, groups, 0, 1, Precision::kFp32, Device::kCuda);
    GpuLaunchPolicy policy;
    policy.max_slice_ns = noise ? 3 * kNoiseCellNs : 1;  // at least one grain step per slice
    policy.max_pixels = 300;                             // 7 strided batches of the 1920 pixels
    policy.event_capacity = 16;                          // forces overflows and re-runs
    // A capacity whose byte size overflows, and one that cannot be allocated: refused, and the buffer,
    // its capacity and the policy stay (without a stale CUDA error at the next launch).
    GpuLaunchPolicy bad = policy;
    bad.event_capacity = (uint64_t{1} << 59) + 1;
    CHECK_THROWS_AS(gpu.set_gpu_launch_policy(bad), std::runtime_error);
    gpu.set_gpu_launch_policy(policy);
    bad.event_capacity = uint64_t{1} << 50;
    CHECK_THROWS_AS(gpu.set_gpu_launch_policy(bad), std::runtime_error);
    std::vector<float> irr0(W * H * 2);
    for (size_t i = 0; i < W * H; ++i) {
      irr0[2 * i] = 20.0f + 5.0f * static_cast<float>(i % 41);
      irr0[2 * i + 1] = 30.0f * static_cast<float>(i % 3);
    }
    cpu.init(1234, irr0.data());  // a start off the lattice: the first slice is a partial cell
    gpu.init(1234, irr0.data());
    // One call over three windows, then a zero-length call that only carries a sample over.
    const Runs r = grating_runs(W * H, 2, 1234, 3 * kW23);
    const Events a = advance(cpu, r, 3 * kW23);
    CHECK(a.size() > 300);
    check_equal(a, advance(gpu, r, 3 * kW23));
    Runs now;
    now.m = 1;
    for (uint32_t px = 0; px < W * H; px += 3) {
      now.ids.push_back(px);
      now.counts.push_back(1);
      now.t.push_back(3 * kW23);
      now.irr.push_back(150.0f + static_cast<float>(px % 7));
      now.irr.push_back(5.0f);
    }
    CHECK(advance(cpu, now, 3 * kW23).size() == 0);
    CHECK(advance(gpu, now, 3 * kW23).size() == 0);
    check_same_states(cpu, gpu);
    const GpuLaunchStats& ls = gpu.gpu_launch_stats();
    const uint64_t slices = noise ? (3 * kW23 + 3 * kNoiseCellNs - 1) / (3 * kNoiseCellNs) : 3;
    MESSAGE("noise " << noise << ": " << ls.launches << " launches, " << ls.reruns << " re-runs");
    CHECK(ls.launches >= 7 * slices + ls.reruns);
    CHECK(ls.reruns > 0);
  }
}

TEST_CASE("the instantaneous response (D3) is bitwise identical on the GPU, also under batches and overflows") {
  // The ideal family with every branch of advance_instant: mismatch, leak, leak-type hot pixels, the ideal reset and
  // the physical one with a refractory period, a 120 Hz group and a pulse group (breakpoints).
  for (const bool physical : {false, true}) {
    SensorParams p = preset("ideal");
    p.sigma_theta_on = p.sigma_theta_off = 0.02;
    p.leak_rate_hz = 2.0;
    p.leak_sigma_log10 = 0.1;
    p.hot_fraction = 0.02;
    if (physical) {
      p.reset_mode = ResetMode::kPhysical;
      p.refractory_s = 50e-6;
    }
    const uint32_t W = 48, H = 40;
    const std::vector<Waveform> groups = {make_constant_waveform(), make_fourier_waveform(120.0, {0.3}, {0.5}),
                                          make_pulse_waveform(200.0, 0.06, 50e-6, 1e-3, 0.1)};
    SensorModel cpu(p, W, H, 33, groups, 0, 4, Precision::kFp32, Device::kCpu);
    SensorModel gpu(p, W, H, 33, groups, 0, 1, Precision::kFp32, Device::kCuda);
    GpuLaunchPolicy policy;
    policy.max_slice_ns = 1;     // whole windows (the grain without noise)
    policy.max_pixels = 300;     // 7 strided batches of the 1920 pixels
    policy.event_capacity = 16;  // forces overflows and re-runs
    gpu.set_gpu_launch_policy(policy);
    std::vector<float> irr0(W * H * 3);
    for (size_t i = 0; i < W * H; ++i) {
      irr0[3 * i] = 20.0f + 5.0f * static_cast<float>(i % 41);
      irr0[3 * i + 1] = 30.0f * static_cast<float>(i % 3);
      irr0[3 * i + 2] = 15.0f * static_cast<float>(i % 2);
    }
    cpu.init(1234, irr0.data());
    gpu.init(1234, irr0.data());
    const Runs r = grating_runs(W * H, 3, 1234, 3 * kW23);
    const Events a = advance(cpu, r, 3 * kW23);
    CHECK(a.size() > 1000);
    check_equal(a, advance(gpu, r, 3 * kW23));
    check_same_states(cpu, gpu);
    CHECK(gpu.gpu_launch_stats().reruns > 0);
  }
}

TEST_CASE("a run started before t = 0 is bitwise identical on the GPU, sliced on the lattice") {
  // Negative model times: slice bounds by floor division, lattice cells by arithmetic shift, Philox
  // indices of negative node times, and crossings before t = 0 (pixel.h search()).
  for (const bool noise : {true, false}) {
    SensorParams p = preset("prophesee_gen3");
    p.hot_fraction = 0.02;
    p.noise = noise;
    const uint32_t W = 32, H = 24;
    const std::vector<Waveform> groups = {make_constant_waveform(), make_fourier_waveform(120.0, {0.3}, {0.5})};
    SensorModel cpu(p, W, H, 33, groups, 0, 4, Precision::kFp32, Device::kCpu);
    SensorModel gpu(p, W, H, 33, groups, 0, 1, Precision::kFp32, Device::kCuda);
    GpuLaunchPolicy policy;
    policy.max_slice_ns = noise ? 5 * kNoiseCellNs : 1;
    policy.max_pixels = 200;
    gpu.set_gpu_launch_policy(policy);
    std::vector<float> irr0(W * H * 2);
    for (size_t i = 0; i < W * H; ++i) {
      irr0[2 * i] = 20.0f + 5.0f * static_cast<float>(i % 41);
      irr0[2 * i + 1] = 30.0f * static_cast<float>(i % 3);
    }
    const int64_t t0 = -2 * kW23 - 4321;  // off the lattice, before t = 0
    cpu.init(t0, irr0.data());
    gpu.init(t0, irr0.data());
    const Runs r = grating_runs(W * H, 2, t0, kW23);
    const Events a = advance(cpu, r, kW23);
    CHECK(std::count_if(a.t_cross_ns.begin(), a.t_cross_ns.end(), [](int64_t t) { return t < 0; }) > 100);
    check_equal(a, advance(gpu, r, kW23));
    check_same_states(cpu, gpu);
  }
}

TEST_CASE("CUDA clamps out-of-range irradiance exactly like the CPU and refuses non-finite input") {
  const SensorParams p = preset("prophesee_gen3");
  SensorModel cpu(p, 2, 2, 3, {make_constant_waveform()}, 0, 1, Precision::kFp32, Device::kCpu);
  SensorModel gpu(p, 2, 2, 3, {make_constant_waveform()}, 0, 1, Precision::kFp32, Device::kCuda);
  const std::vector<float> irr0 = {100.0f, 1e38f, 100.0f, 100.0f};
  cpu.init(0, irr0.data());
  gpu.init(0, irr0.data());
  const uint32_t ids[2] = {0, 3};
  const int64_t t[4] = {1'000'000, 2'000'000, 1'000'000, 2'000'000};
  const float inf[4] = {std::numeric_limits<float>::infinity(), 100.0f, 100.0f, 100.0f};
  CHECK_THROWS_AS(gpu.advance(ids, 2, t, inf, 2, nullptr, 5'000'000), std::invalid_argument);
  const float irr[4] = {1e38f, 100.0f, -5.0f, 100.0f};
  check_equal(cpu.advance(ids, 2, t, irr, 2, nullptr, 20'000'000), gpu.advance(ids, 2, t, irr, 2, nullptr, 20'000'000));
  check_same_states(cpu, gpu);
  CHECK(gpu.stats().quarantines == 0);
  CHECK(gpu.stats().fallbacks == 0);
}

TEST_CASE("GPU launches stay under 50 ms on a VGA sensor: mixed scene, warm-up and long calls") {
  // Static rows on top, a moving grating below (busy and quiet pixels in contiguous regions), the
  // Prophesee preset with noise on, at the P1 runs' values (P0e moved them to literature priors, hot pixels off):
  // ~123 hot pixels of up to 2 kHz, the busiest. The cap is DESIGN §4.2's; the default policy plans 15 ms per launch.
  constexpr uint32_t W = 640, H = 480;
  SensorParams p = preset("prophesee_gen3");
  p.refractory_s = 0.24e-3, p.leak_rate_hz = 0.005, p.hot_fraction = 4e-4;
  SensorModel s(p, W, H, 1, {make_constant_waveform()}, 0, 6, Precision::kFp32, Device::kCuda);
  std::vector<float> irr0(W * H);
  for (uint32_t i = 0; i < W * H; ++i) irr0[i] = static_cast<float>(30.0 * std::pow(100.0, i % 97 / 96.0));
  s.init(0, irr0.data());
  auto report = [&](const std::string& what, double sim_ms) {
    const GpuLaunchStats& ls = s.gpu_launch_stats();
    MESSAGE(what << " (" << sim_ms << " ms simulated): " << ls.launches << " launches, max " << ls.max_ms
                 << " ms, mean " << ls.total_ms / static_cast<double>(std::max<uint64_t>(ls.launches, 1)) << " ms");
    WARN(ls.max_ms < 50.0);   // DESIGN §4.2's cap, reported: the desktop and other jobs share the GPU
    CHECK(ls.max_ms < 100.0);  // twice the cap: a planner that stops slicing takes 106 ms or more
    s.reset_stats();
  };
  // The 50 ms warm-up after init as one call (docs/dev/SENSOR.md, integrator notes).
  s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, 50'000'000);
  report("warm-up call", 50);
  // Per-window calls: the bottom half follows a grating moving at 500 px/s (9 samples per window).
  const int64_t t_w = s.time_ns();
  for (int w = 0; w < 2; ++w) {
    const int64_t t0 = t_w + w * kW23, t1 = t0 + kW23;
    std::vector<uint32_t> ids;
    std::vector<int64_t> t;
    std::vector<float> irr;
    for (uint32_t px = W * H / 2; px < W * H; ++px) {
      ids.push_back(px);
      for (int j = 0; j <= 8; ++j) {
        const int64_t tj = t0 + kW23 * j / 8;
        t.push_back(tj);
        const double phase = 2 * 3.141592653589793 * ((px % W) / 64.0 - 500.0 / 64.0 * 1e-9 * static_cast<double>(tj));
        irr.push_back(static_cast<float>(300.0 * (1.0 + 0.8 * std::sin(phase))));
      }
    }
    s.advance(ids.data(), ids.size(), t.data(), irr.data(), 9, nullptr, t1);
  }
  report("mixed per-window calls", 2 * kW23 * 1e-6);
  // A long call: everything static again, 0.1 s at once.
  s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, s.time_ns() + 100'000'000);
  report("long static call", 100);
}

#if BALLASIM_SENSOR_CUDA
TEST_CASE("advance_device on device-resident runs equals advance; non-finite input fails before any state changes") {
  const uint32_t W = 40, H = 32;
  const std::vector<Waveform> groups = {make_constant_waveform(), make_fourier_waveform(120.0, {0.3}, {0.5})};
  SensorParams p = preset("prophesee_gen3");
  p.hot_fraction = 0.02;
  SensorModel host(p, W, H, 19, groups, 0, 1, Precision::kFp32, Device::kCuda);
  SensorModel dev(p, W, H, 19, groups, 0, 1, Precision::kFp32, Device::kCuda);
  dev.set_gpu_launch_policy({20.0, 0, 0, 64});  // overflows of a small event buffer keep earlier launches' events
  std::vector<float> irr0(W * H * 2, 60.0f);
  host.init(0, irr0.data());
  dev.init(0, irr0.data());
  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  for (int w = 0; w < 2; ++w) {
    const Runs r = grating_runs(W * H, 2, w * kW23, (w + 1) * kW23);
    std::vector<int64_t> start(r.ids.size());
    for (size_t k = 0; k < r.ids.size(); ++k) start[k] = static_cast<int64_t>(k * r.m);
    std::vector<float> irr = r.irr;
    irr[1] = -5.0f;  // row 0's first sample (group 1): clamped, and counted on both paths
    irr[5] = 3e9f;   // padding (row 0 takes one sample), ignored on both paths
    irr[7] = std::numeric_limits<float>::quiet_NaN();
    const uint64_t host_clamps0 = host.stats().input_clamps;
    const Events a = host.advance_runs(r.ids.data(), r.ids.size(), r.t.data(), irr.data(), r.t.size(), start.data(),
                                       r.counts.data(), (w + 1) * kW23);
    CHECK(host.stats().input_clamps == host_clamps0 + 1);
    uint32_t* d_ids = to_device(r.ids);
    int64_t* d_start = to_device(start);
    int32_t* d_count = to_device(r.counts);
    int64_t* d_t = to_device(r.t);
    float* d_irr = to_device(irr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);  // pageable copies may land after cudaMemcpy returns
    const DeviceRuns runs{d_ids, d_start, d_count, r.ids.size(), d_t, d_irr, r.t.size()};
    const uint64_t clamps0 = dev.stats().input_clamps;
    const DeviceEvents ev = dev.advance_device(runs, (w + 1) * kW23, stream);
    CHECK(dev.stats().input_clamps == clamps0 + 1);
    const Events b = dev.download_events(ev);
    CHECK(a.size() > 100);
    check_equal(a, b);
    check_same_states(host, dev);
    // A NaN sample that a run takes: refused, the state and the model time stay.
    irr[1] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(cudaMemcpy(d_irr, irr.data(), irr.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const std::vector<PixelState> before = dev.states();
    CHECK_THROWS_AS(dev.advance_device(runs, (w + 2) * kW23, stream), std::invalid_argument);
    CHECK(dev.time_ns() == (w + 1) * kW23);
    const std::vector<PixelState> after = dev.states();
    CHECK(std::memcmp(before.data(), after.data(), before.size() * sizeof(PixelState)) == 0);
    free_device({d_ids, d_start, d_count, d_t, d_irr});
  }
  cudaStreamDestroy(stream);
}

TEST_CASE("advance_device refuses ids out of range, duplicate ids and runs out of range, on the CPU and the GPU") {
  const SensorParams p = preset("prophesee_gen3");
  SensorModel cpu(p, 8, 4, 5, {make_constant_waveform()}, 0, 1, Precision::kFp32, Device::kCpu);
  SensorModel gpu(p, 8, 4, 5, {make_constant_waveform()}, 0, 1, Precision::kFp32, Device::kCuda);
  const std::vector<float> irr0(32, 100.0f);
  cpu.init(0, irr0.data());
  gpu.init(0, irr0.data());
  const std::vector<int64_t> t = {1'000'000, 2'000'000};
  const std::vector<float> irr = {300.0f, 30.0f};
  int64_t* d_t = to_device(t);
  float* d_irr = to_device(irr);
  struct Call {
    std::vector<uint32_t> ids;
    std::vector<int64_t> start;
    std::vector<int32_t> count;
    const char* error;  // nullptr: valid
  };
  const Call calls[] = {{{1, 32}, {0, 1}, {1, 1}, "SensorModel.advance: pixel id out of range"},
                        {{3, 3}, {0, 1}, {1, 1}, "SensorModel.advance: duplicate pixel id"},
                        {{1, 2}, {0, 2}, {1, 1}, "SensorModel.advance: sample run out of range"},
                        {{1, 2}, {0, 1}, {-1, 1}, "SensorModel.advance: sample run out of range"},
                        {{1, 2}, {0, 1}, {1, 1}, nullptr}};
  const std::vector<PixelState> before = gpu.states();
  for (const Call& call : calls) {
    const DeviceRuns host_runs{call.ids.data(), call.start.data(), call.count.data(), 2, t.data(), irr.data(), 2};
    uint32_t* d_ids = to_device(call.ids);
    int64_t* d_start = to_device(call.start);
    int32_t* d_count = to_device(call.count);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const DeviceRuns runs{d_ids, d_start, d_count, 2, d_t, d_irr, 2};
    if (call.error) {
      CHECK_THROWS_WITH_AS(cpu.advance_device(host_runs, 5'000'000, nullptr), call.error, std::invalid_argument);
      CHECK_THROWS_WITH_AS(gpu.advance_device(runs, 5'000'000, nullptr), call.error, std::invalid_argument);
      CHECK(gpu.time_ns() == 0);
      const std::vector<PixelState> after = gpu.states();
      CHECK(std::memcmp(before.data(), after.data(), before.size() * sizeof(PixelState)) == 0);
    } else {
      // A valid call afterwards still equals the CPU.
      const Events a = cpu.download_events(cpu.advance_device(host_runs, 5'000'000, nullptr));
      CHECK(a.size() > 0);
      check_equal(a, gpu.download_events(gpu.advance_device(runs, 5'000'000, nullptr)));
      check_same_states(cpu, gpu);
    }
    free_device({d_ids, d_start, d_count});
  }
  free_device({d_t, d_irr});
}
#endif  // BALLASIM_SENSOR_CUDA

TEST_SUITE_END();
