// Throughput of the sensor kernel on a VGA sensor, CPU and GPU (P1-M3 performance report).
//
//   balla_sensor_bench [threads] [windows] [cpu|cuda|both]   (timed 8.39 ms windows after a 50 ms warm-up)
//
// Scenes (Prophesee Gen3 preset at the P1 runs' values, illuminance_scale 1, photocurrent in fA):
//   static      every pixel static at 30..3000 fA (noise lattice only);
//   stripes     a square-wave grating (period 64 px, contrast ln 4, 300 fA mean) moving at 500 px/s
//               over box pixels: exact piecewise-linear coverage samples at the window ends and at
//               the instants an edge enters or leaves each pixel, as the integrator would deliver;
// each with noise on and off. Reports wall time per simulated second, pixel-steps/s (accepted
// deterministic sub-steps), lattice cells/s and events/s.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "sensor/sensor_model.h"

using namespace balla::sensor;

namespace {

constexpr uint32_t kW = 640, kH = 480;
constexpr int64_t kWindow = int64_t{1} << 23;

// Irradiance of column x at time t for a grating moving right at v px/s (box pixel [x-0.5, x+0.5]).
struct Grating {
  double period = 64, v = 500, lo = 150, hi = 600;
  // Fraction of [a, b] covered by bright bars of the pattern shifted by s = v t.
  double bright(double a, double b) const {
    auto integral = [&](double u) {  // measure of bright on [0, u] for bars [0, P/2) mod P
      const double P = period, h = P / 2;
      const double n = std::floor(u / P);
      const double r = u - n * P;
      return n * h + std::min(r, h);
    };
    return (integral(b) - integral(a)) / (b - a);
  }
  double value(uint32_t x, int64_t t_ns) const {
    const double s = v * 1e-9 * static_cast<double>(t_ns);
    const double a = x - 0.5 - s + 1e6 * period, b = a + 1.0;
    const double f = bright(a, b);
    return lo + (hi - lo) * f;
  }
  // Times in (t0, t1) when an edge passes a pixel boundary of column x (coverage kinks).
  void kinks(uint32_t x, int64_t t0, int64_t t1, std::vector<int64_t>& out) const {
    const double h = period / 2;
    for (double boundary : {x - 0.5, x + 0.5}) {
      // Edges sit at s + k h; boundary = s + k h  ->  t = (boundary - k h) / v.
      const double s0 = v * 1e-9 * static_cast<double>(t0), s1 = v * 1e-9 * static_cast<double>(t1);
      const double k_lo = std::ceil((boundary - s1) / h), k_hi = std::floor((boundary - s0) / h);
      for (double k = k_lo; k <= k_hi; k += 1) {
        const int64_t t = static_cast<int64_t>(std::llround((boundary - k * h) / v * 1e9));
        if (t > t0 && t < t1) out.push_back(t);
      }
    }
  }
};

void run(const char* scene, bool noise, int threads, int windows, bool cuda) {
  SensorParams p = preset("prophesee_gen3");
  // The preset as the P1 runs had it (SENSOR.md "Performance"; P0e moved these to literature priors, hot pixels off):
  // the rows stay comparable, with ~123 hot pixels of up to 2 kHz.
  p.refractory_s = 0.24e-3, p.leak_rate_hz = 0.005, p.hot_fraction = 4e-4;
  p.noise = noise;
  SensorModel s(p, kW, kH, 1, {make_constant_waveform()}, 0, threads, cuda ? Precision::kFp32 : Precision::kFp64,
                cuda ? Device::kCuda : Device::kCpu);
  const bool stripes = scene[0] == 's' && scene[1] == 't' && scene[2] == 'r';
  const Grating g;
  std::vector<float> irr0(kW * kH);
  for (uint32_t y = 0; y < kH; ++y)
    for (uint32_t x = 0; x < kW; ++x)
      irr0[y * kW + x] = stripes ? static_cast<float>(g.value(x, 0))
                                 : static_cast<float>(30.0 * std::pow(100.0, (x + kW * y) % 97 / 96.0));
  s.init(0, irr0.data());
  // Warm-up as in a run (DESIGN §8.4: 50 ms): settles the initial ON burst of pixels whose phase
  // phi0 starts within a few sigma_v of theta_on. Not timed.
  constexpr int kWarmup = 6;
  size_t events = 0;
  double wall = 0;
  for (int w = 0; w < kWarmup + windows; ++w) {
    if (w == kWarmup) {
      s.reset_stats();
      events = 0;
      wall = 0;
    }
    const int64_t t0 = w * kWindow, t1 = t0 + kWindow;
    std::vector<uint32_t> ids;
    std::vector<int64_t> t;
    std::vector<float> irr;
    size_t m = 0;
    std::vector<int32_t> counts;
    if (stripes) {
      // Rows share the pattern; build one row of runs and replicate it for every row.
      std::vector<std::vector<int64_t>> col_times(kW);
      for (uint32_t x = 0; x < kW; ++x) {
        std::vector<int64_t>& ts = col_times[x];
        g.kinks(x, t0, t1, ts);
        ts.push_back(t1);
        std::sort(ts.begin(), ts.end());
        m = std::max(m, ts.size());
      }
      for (uint32_t y = 0; y < kH; ++y)
        for (uint32_t x = 0; x < kW; ++x) {
          ids.push_back(y * kW + x);
          counts.push_back(static_cast<int32_t>(col_times[x].size()));
        }
      t.assign(ids.size() * m, t1);
      irr.assign(ids.size() * m, 0.0f);
      for (size_t i = 0; i < ids.size(); ++i) {
        const uint32_t x = ids[i] % kW;
        for (size_t j = 0; j < col_times[x].size(); ++j) {
          t[i * m + j] = col_times[x][j];
          irr[i * m + j] = static_cast<float>(g.value(x, col_times[x][j]));
        }
      }
    }
    const auto a = std::chrono::steady_clock::now();
    const int32_t* cnt = counts.empty() ? nullptr : counts.data();
    const Events ev = s.advance(ids.data(), ids.size(), t.data(), irr.data(), m, cnt, t1);
    wall += std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
    events += ev.size();
  }
  const double sim = static_cast<double>(windows) * kWindow * 1e-9;
  const KernelStats& st = s.stats();
  const GpuLaunchStats& ls = s.gpu_launch_stats();
  std::printf("%-8s noise=%-3s %-14s %7.3f s wall per sim s | pixel-steps %8.3g/s (%5.1f%% settled) | "
              "cells %8.3g/s | leaf %8.3g/s | events %8.3g/s (%.3g ev per sim s)",
              scene, noise ? "on" : "off", cuda ? "cuda fp32" : (std::to_string(s.num_threads()) + " thr fp64").c_str(),
              wall / sim, st.substeps / wall,
              100.0 * st.settled_substeps / std::max<uint64_t>(1, st.substeps), st.cells / wall,
              st.leaf_intervals / wall, events / wall, events / sim);
  if (cuda) std::printf(" | %llu launches, max %.1f ms", static_cast<unsigned long long>(ls.launches), ls.max_ms);
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  const int threads = argc > 1 ? std::atoi(argv[1]) : 6;
  const int windows = argc > 2 ? std::atoi(argv[2]) : 4;
  const std::string which = argc > 3 ? argv[3] : "both";
  for (const char* scene : {"static", "stripes"})
    for (bool noise : {true, false}) {
      if (which != "cuda") run(scene, noise, threads, windows, false);
      if (which != "cpu" && sensor_cuda_available()) run(scene, noise, threads, windows, true);
    }
  return 0;
}
