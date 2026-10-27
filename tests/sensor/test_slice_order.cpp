// The general pass's work estimate and key over a time range (slice_order.h; PHASE2 P2-E E5): the key equals the
// quiet kernel's window key (sensor_cuda.cu quiet_kernel, restated below as it is written there), a window-long range
// gives the quiet kernel's work, and the sample counts equal brute force on random runs. The per-slice sort
// (GpuLaunchPolicy::slice_sort): slice_key equals the dry launches' key, and the per-slice order of every launch of a
// window, made here as sensor_cuda.cu's sort_slice makes it (a stable sort of the launch's positions by slice_key), is
// a permutation of the launch's pixels with the heavy ones first and the window order's busy pixels; on the CUDA device
// (test suite "cuda"), the sort leaves events, states and work counters bitwise unchanged under batches, one-cell
// slices and overflow re-runs, and BALLA_SENSOR_SLICE_SORT switches it. The keys order threads only, so no output
// depends on them.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "sensor/sensor_model.h"
#include "sensor/slice_order.h"

using namespace balla::sensor;

namespace {

// sensor_cuda.cu quiet_kernel's key of work u, as written there (kDoneKey 0xFFFF).
uint16_t quiet_kernel_key(uint32_t u) {
  constexpr uint16_t kDoneKey = 0xFFFF;
  const uint32_t q = u < 32768u ? u : std::min(32768u + (u - 32768u) / 64u, static_cast<uint32_t>(kDoneKey - 1));
  return static_cast<uint16_t>(kDoneKey - 1 - q);
}

// quiet_kernel's work of a pixel resumed at t over a window ending at t_to, with n samples in the call.
uint32_t quiet_kernel_work(int64_t t, int64_t t_to, int32_t n) {
  return static_cast<uint32_t>((t_to - t) >> kNoiseCellLog2) + 1u + 16u * static_cast<uint32_t>(n);
}

int32_t brute_count(const std::vector<int64_t>& t, int64_t t0, int64_t te) {
  return static_cast<int32_t>(std::count_if(t.begin(), t.end(), [&](int64_t x) { return x >= t0 && x < te; }));
}

}  // namespace

TEST_CASE("work_key is the quiet kernel's key: equal values, descending work, the done key apart") {
  CHECK(kWorkPerSample == 16u);
  CHECK(kWorkDoneKey == 0xFFFF);
  for (uint32_t u = 0; u <= 200000; ++u) REQUIRE(work_key(u) == quiet_kernel_key(u));
  std::mt19937_64 rng(5);
  for (int i = 0; i < 100000; ++i) {
    const uint32_t u = static_cast<uint32_t>(rng());
    REQUIRE(work_key(u) == quiet_kernel_key(u));
  }
  for (uint32_t u : {0xFFFFFFFEu, 0xFFFFFFFFu}) CHECK(work_key(u) == quiet_kernel_key(u));
  // Monotone: more work never sorts after less (ascending key); exact below 2^15 units; never the done key.
  uint16_t prev = work_key(0);
  for (uint32_t u = 1; u <= 4200000; u += (u < 70000 ? 1u : 97u)) {
    const uint16_t k = work_key(u);
    REQUIRE(k <= prev);
    REQUIRE(k < kWorkDoneKey);
    if (u < 32768u) REQUIRE(k == static_cast<uint16_t>(0xFFFE - u));
    prev = k;
  }
  CHECK(work_key(0xFFFFFFFFu) == 0);
}

TEST_CASE("slice_work over a pixel's whole window is the quiet kernel's work") {
  constexpr int64_t kWindow = int64_t{1} << 23;
  std::mt19937_64 rng(11);
  for (int trial = 0; trial < 2000; ++trial) {
    const int64_t w0 = (static_cast<int64_t>(rng() % 4096) - 2048) * kWindow;  // windows before and after t = 0
    const int64_t t_to = w0 + kWindow;
    // The resume time: the window's start, a cell boundary inside it (where the quiet pass stopped), or any time.
    int64_t t = w0;
    if (trial % 3 == 1) t = w0 + static_cast<int64_t>(rng() % 128) * kNoiseCellNs;
    if (trial % 3 == 2) t = w0 + static_cast<int64_t>(rng() % static_cast<uint64_t>(kWindow));
    const int32_t n = static_cast<int32_t>(rng() % 200);
    std::vector<int64_t> s(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) s[static_cast<size_t>(i)] = t + static_cast<int64_t>(i) * ((t_to - t) / (n + 1));
    REQUIRE(slice_work(s.data(), n, t, t_to) == quiet_kernel_work(t, t_to, n));
    REQUIRE(work_key(slice_work(s.data(), n, t, t_to)) == quiet_kernel_key(quiet_kernel_work(t, t_to, n)));
  }
}

TEST_CASE("samples_in and slice_work count the samples of [t0, te) as brute force does") {
  std::mt19937_64 rng(23);
  for (int trial = 0; trial < 3000; ++trial) {
    // A strictly increasing run on the 2^10 ns grid and off it, with gaps of 1 ns to several cells.
    const int32_t n = static_cast<int32_t>(rng() % 64);
    std::vector<int64_t> t;
    int64_t x = (static_cast<int64_t>(rng() % 1000000) - 500000) * 1024;
    for (int32_t i = 0; i < n; ++i) {
      x += trial % 2 == 0 ? 1024 * static_cast<int64_t>(1 + rng() % 200) : static_cast<int64_t>(1 + rng() % 300000);
      t.push_back(x);
    }
    for (int q = 0; q < 20; ++q) {
      const int64_t lo = t.empty() ? 0 : t.front() - 100000, hi = t.empty() ? 1 : t.back() + 100000;
      int64_t t0 = lo + static_cast<int64_t>(rng() % static_cast<uint64_t>(hi - lo));
      int64_t te = lo + static_cast<int64_t>(rng() % static_cast<uint64_t>(hi - lo));
      if (q % 5 == 0 && !t.empty()) t0 = t[rng() % t.size()];  // a sample exactly at either end
      if (q % 5 == 1 && !t.empty()) te = t[rng() % t.size()];
      const int32_t expect = te > t0 ? brute_count(t, t0, te) : 0;
      REQUIRE(samples_in(t.data(), n, t0, te) == expect);
      if (te > t0) {
        const uint64_t u =
            static_cast<uint64_t>((te - t0) >> kNoiseCellLog2) + 1u + 16u * static_cast<uint64_t>(expect);
        REQUIRE(slice_work(t.data(), n, t0, te) == u);
      }
    }
  }
  const std::vector<int64_t> none;
  CHECK(samples_in(none.data(), 0, 0, 100) == 0);
  CHECK(slice_work(none.data(), 0, 0, kNoiseCellNs) == 2u);
  CHECK(slice_work(none.data(), 0, 0, kNoiseCellNs - 1) == 1u);
}

TEST_CASE("slices of a window: per-slice work and keys sort heavy slices first, ties in the incoming order") {
  // Three pixels over one window cut into two slices: A's samples in the first half, B's in the second, C's spread.
  constexpr int64_t kWindow = int64_t{1} << 23, kHalf = kWindow / 2;
  std::vector<int64_t> a, b, c;
  for (int i = 0; i < 40; ++i) {
    a.push_back(1000 + i * 1024);
    b.push_back(kHalf + 1000 + i * 1024);
    c.push_back(i * (kWindow / 40));
  }
  const std::vector<const std::vector<int64_t>*> px{&a, &b, &c};
  // The window's order ranks all three alike (40 samples each); each slice ranks its own heavy pixel first.
  std::vector<uint16_t> win;
  for (const auto* p : px) {
    win.push_back(work_key(slice_work(p->data(), static_cast<int32_t>(p->size()), 0, kWindow)));
  }
  CHECK(win[0] == win[1]);
  CHECK(win[1] == win[2]);
  for (int s = 0; s < 2; ++s) {
    const int64_t t0 = s * kHalf, te = t0 + kHalf;
    std::vector<int> order{0, 1, 2};
    std::vector<uint16_t> key;
    for (const auto* p : px) {
      key.push_back(work_key(slice_work(p->data(), static_cast<int32_t>(p->size()), t0, te)));
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](int x, int y) { return key[static_cast<size_t>(x)] < key[static_cast<size_t>(y)]; });
    CHECK(order.front() == s);  // A in the first slice, B in the second
    CHECK(order.back() == 1 - s);
  }
}

namespace {

// sensor_cuda.cu dry_slice_key_kernel's key (the E5 study build's), as written there.
uint16_t dry_launch_key(const int64_t* t, int32_t n, int64_t t_resume, int64_t ts, int64_t te, bool last) {
  const int64_t t0 = std::max(t_resume, ts);
  if (!(t0 < te || last)) return kWorkDoneKey;
  return work_key(t0 < te ? slice_work(t, n, t0, te) : 1u);
}

// A pixel left for the general pass: its samples and where the quiet pass left it.
struct Busy {
  std::vector<int64_t> t;
  int64_t t_resume = 0;
};

// The per-slice order of one launch as sensor_cuda.cu's sort_slice makes it: the pixels at the positions [first,
// first + count) of the window's order, stably sorted by slice_key (CUB's radix sort is stable).
std::vector<uint32_t> slice_order(const std::vector<Busy>& px, const std::vector<uint32_t>& window_order, size_t first,
                                  size_t count, int64_t ts, int64_t te, bool last) {
  std::vector<uint32_t> ids(window_order.begin() + static_cast<std::ptrdiff_t>(first),
                            window_order.begin() + static_cast<std::ptrdiff_t>(first + count));
  std::vector<uint16_t> key(px.size());
  for (uint32_t p : ids) {
    const Busy& b = px[p];
    key[p] = slice_key(b.t.data(), static_cast<int32_t>(b.t.size()), b.t_resume, ts, te, last);
  }
  std::stable_sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) { return key[a] < key[b]; });
  return ids;
}

}  // namespace

TEST_CASE("slice_key is the dry launches' key: work_key over the slice's part, pixels without work in it last") {
  std::mt19937_64 rng(31);
  for (int trial = 0; trial < 20000; ++trial) {
    const int64_t w0 = (static_cast<int64_t>(rng() % 64) - 32) * (int64_t{1} << 23);
    const int32_t n = static_cast<int32_t>(rng() % 40);
    std::vector<int64_t> t;
    int64_t x = w0;
    for (int32_t i = 0; i < n; ++i) t.push_back(x += 1 + static_cast<int64_t>(rng() % 400000));
    const int64_t t_resume = w0 + (trial % 2 == 0 ? static_cast<int64_t>(rng() % 128) * kNoiseCellNs
                                                  : static_cast<int64_t>(rng() % (int64_t{1} << 23)));
    const int64_t ts = w0 + static_cast<int64_t>(rng() % 128) * kNoiseCellNs;
    const int64_t te = std::min(w0 + (int64_t{1} << 23), ts + static_cast<int64_t>(1 + rng() % 16) * kNoiseCellNs);
    for (const bool last : {false, true}) {
      const uint16_t k = slice_key(t.data(), n, t_resume, ts, te, last);
      REQUIRE(k == dry_launch_key(t.data(), n, t_resume, ts, te, last));
      const int64_t t0 = std::max(t_resume, ts);
      if (t0 < te) REQUIRE(k == work_key(slice_work(t.data(), n, t0, te)));
      else REQUIRE(k == (last ? work_key(1u) : kWorkDoneKey));
      REQUIRE((k == kWorkDoneKey) == (t0 >= te && !last));  // only a pixel without work takes the done key
    }
  }
}

TEST_CASE("per-slice order: each launch's pixels permuted, heavy first, ties in the window's order, busy pixels kept") {
  constexpr int64_t kWindow = int64_t{1} << 23;
  std::mt19937_64 rng(47);
  for (int trial = 0; trial < 40; ++trial) {
    const int64_t w0 = (static_cast<int64_t>(rng() % 16) - 8) * kWindow, t_to = w0 + kWindow;
    // Busy pixels: runs clustered anywhere in the window (or none), resumed at the window's start, at a cell boundary
    // inside it (where the quiet pass stopped) or anywhere.
    std::vector<Busy> px(300 + rng() % 400);
    for (Busy& b : px) {
      const int64_t start = w0 + static_cast<int64_t>(rng() % static_cast<uint64_t>(kWindow - kWindow / 8));
      const int n = static_cast<int>(rng() % 3 == 0 ? 0 : rng() % 60);
      for (int i = 0; i < n; ++i) b.t.push_back(start + static_cast<int64_t>(i + 1) * (kWindow / 8 / (n + 1)));
      const int mode = static_cast<int>(rng() % 3);
      b.t_resume = mode == 0 ? w0
                   : mode == 1 ? w0 + static_cast<int64_t>(rng() % 128) * kNoiseCellNs
                               : w0 + static_cast<int64_t>(rng() % static_cast<uint64_t>(kWindow));
    }
    // The window's order: the quiet kernel's keys over the window (every sample of the call), stably sorted.
    std::vector<uint32_t> window(px.size());
    std::iota(window.begin(), window.end(), 0u);
    std::vector<uint16_t> wkey(px.size());
    for (size_t p = 0; p < px.size(); ++p) {
      wkey[p] = quiet_kernel_key(quiet_kernel_work(px[p].t_resume, t_to, static_cast<int32_t>(px[p].t.size())));
    }
    std::stable_sort(window.begin(), window.end(), [&](uint32_t a, uint32_t b) { return wkey[a] < wkey[b]; });
    // Slices of whole cells (one cell up to a quarter window) and batches of up to 37 to all pixels.
    const size_t batch = trial % 2 == 0 ? px.size() : 1 + rng() % 37;
    std::vector<uint64_t> sample_work(px.size(), 0);
    int64_t ts = w0;
    while (true) {
      const int64_t te = std::min(t_to, ts + static_cast<int64_t>(1 + rng() % 32) * kNoiseCellNs);
      const bool last = te == t_to;
      std::vector<int> seen(px.size(), 0);
      for (size_t first = 0; first < px.size(); first += batch) {
        const size_t count = std::min(batch, px.size() - first);
        const std::vector<uint32_t> order = slice_order(px, window, first, count, ts, te, last);
        // The same pixels as the window order's positions, so each pixel runs once per slice.
        std::vector<uint32_t> a(order), b(window.begin() + static_cast<std::ptrdiff_t>(first),
                                          window.begin() + static_cast<std::ptrdiff_t>(first + count));
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        REQUIRE(a == b);
        size_t busy_window = 0, busy_slice = 0;
        for (size_t i = 0; i < count; ++i) {
          const Busy& q = px[order[i]];
          const int64_t t0 = std::max(q.t_resume, ts);
          const bool busy = t0 < te || last;
          const uint16_t k = slice_key(q.t.data(), static_cast<int32_t>(q.t.size()), q.t_resume, ts, te, last);
          REQUIRE(busy == (k != kWorkDoneKey));
          if (busy) {
            REQUIRE(i == busy_slice);  // every busy pixel before the first idle one
            ++busy_slice;
          }
          if (i > 0) {
            const Busy& r = px[order[i - 1]];
            const uint16_t kp = slice_key(r.t.data(), static_cast<int32_t>(r.t.size()), r.t_resume, ts, te, last);
            REQUIRE(kp <= k);  // descending work
            if (kp == k) {     // ties in the window's order
              const auto pos = [&](uint32_t p) { return std::find(window.begin(), window.end(), p) - window.begin(); };
              REQUIRE(pos(order[i - 1]) < pos(order[i]));
            }
          }
          if (t0 < te) {
            const uint32_t u = slice_work(q.t.data(), static_cast<int32_t>(q.t.size()), t0, te);
            sample_work[order[i]] += u - 1u - static_cast<uint32_t>((te - t0) >> kNoiseCellLog2);
          }
          ++seen[order[i]];
        }
        for (size_t i = first; i < first + count; ++i) {
          const Busy& q = px[window[i]];
          busy_window += std::max(q.t_resume, ts) < te || last;
        }
        REQUIRE(busy_window == busy_slice);
      }
      REQUIRE(std::all_of(seen.begin(), seen.end(), [](int k) { return k == 1; }));
      if (last) break;
      ts = te;
    }
    // The slices' sample work adds up to the window's from the resume time: the slices partition [t_resume, t_to).
    for (size_t p = 0; p < px.size(); ++p) {
      const Busy& q = px[p];
      const auto n = static_cast<int32_t>(q.t.size());
      REQUIRE(sample_work[p] == uint64_t{kWorkPerSample} * static_cast<uint64_t>(samples_in(q.t.data(), n, q.t_resume,
                                                                                            t_to)));
    }
  }
}

namespace {

constexpr int64_t kW23 = int64_t{1} << 23;

// BALLA_SENSOR_SLICE_SORT for the scope of a test (unset by default), restored afterwards.
class SliceSortEnv {
 public:
  explicit SliceSortEnv(const char* value) {
    if (const char* e = std::getenv(kName)) saved_ = std::string(e);
    set(value);
  }
  ~SliceSortEnv() { set(saved_ ? saved_->c_str() : nullptr); }
  SliceSortEnv(const SliceSortEnv&) = delete;
  SliceSortEnv& operator=(const SliceSortEnv&) = delete;
  static void set(const char* value) {
    if (value != nullptr) ::setenv(kName, value, 1);
    else ::unsetenv(kName);
  }

 private:
  static constexpr const char* kName = "BALLA_SENSOR_SLICE_SORT";
  std::optional<std::string> saved_;
};

// Padded runs (SensorModel::advance's layout, m per pixel) over (t0, t1] for every pixel but one in five: each takes
// 1-9 samples clustered in an eighth of the span that moves from pixel to pixel, so that the slices of a window rank
// the pixels differently from the window. Group 0 a moving grating, group 1 (a 120 Hz light) a constant level.
struct Clustered {
  std::vector<uint32_t> ids;
  std::vector<int64_t> t;
  std::vector<float> irr;
  std::vector<int32_t> counts;
  size_t m = 9;
};

Clustered clustered_runs(uint32_t n_pixels, int64_t t0, int64_t t1) {
  Clustered r;
  const int64_t span = t1 - t0, len = span / 8;
  for (uint32_t px = 0; px < n_pixels; ++px) {
    if (px % 5 == 4) continue;
    r.ids.push_back(px);
    const int32_t c = 1 + static_cast<int32_t>((px * 7) % r.m);
    r.counts.push_back(c);
    const int64_t start = t0 + (span - len) * static_cast<int64_t>((px * 37) % 101) / 100;
    for (size_t j = 0; j < r.m; ++j) {
      r.t.push_back(j < static_cast<size_t>(c) ? start + len * static_cast<int64_t>(j + 1) / c : 0);
      const double phase = 0.05 * px + 1e-7 * static_cast<double>(r.t.back());
      r.irr.push_back(static_cast<float>(200.0 * (1.2 + std::sin(phase))));
      r.irr.push_back(static_cast<float>(10.0 * (px % 5)));
    }
  }
  return r;
}

Events advance(SensorModel& s, const Clustered& r, int64_t t_end) {
  return s.advance(r.ids.data(), r.ids.size(), r.t.data(), r.irr.data(), r.m, r.counts.data(), t_end);
}

void check_events(const Events& a, const Events& b) {
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

void check_counters(const KernelStats& a, const KernelStats& b) {
  CHECK(a.substeps == b.substeps);
  CHECK(a.substep_attempts == b.substep_attempts);
  CHECK(a.settled_substeps == b.settled_substeps);
  CHECK(a.cells == b.cells);
  CHECK(a.refinements == b.refinements);
  CHECK(a.leaf_intervals == b.leaf_intervals);
  CHECK(a.fallbacks == b.fallbacks);
  CHECK(a.input_clamps == b.input_clamps);
  CHECK(a.quarantines == b.quarantines);
  CHECK(a.pieces == b.pieces);
  CHECK(a.hpf_splits == b.hpf_splits);
}

}  // namespace

TEST_SUITE_BEGIN("cuda");

TEST_CASE("the per-slice sort leaves the GPU's events, states and counters bitwise unchanged") {
  const SliceSortEnv env(nullptr);  // the policies below decide
  // The cascade kernels it orders: frozen and transient diffusion, veiling glare, the high-pass filter, no noise.
  enum Variant { kFrozen, kTransient, kGlare, kHpf, kNoNoise };
  for (const Variant v : {kFrozen, kTransient, kGlare, kHpf, kNoNoise}) {
    // Policies forcing one-cell slices (a tiny target) or 3-cell windows, both with batches and overflow re-runs.
    for (const bool tiny : {true, false}) {
      INFO("variant " << static_cast<int>(v) << ", tiny target " << tiny);
      SensorParams p = preset("prophesee_gen3");
      p.hot_fraction = 0.02;
      p.noise = v != kNoNoise;
      if (v == kTransient) p.noise_diffusion = NoiseDiffusion::kTransient;
      if (v == kGlare) p.veiling_glare = 0.05;
      if (v == kHpf) p.tau_hp = 0.05;
      const uint32_t W = 48, H = 40;
      const std::vector<Waveform> groups = {make_constant_waveform(), make_fourier_waveform(120.0, {0.3}, {0.5})};
      SensorModel cpu(p, W, H, 41, groups, 0, 4, Precision::kFp32, Device::kCpu);
      SensorModel off(p, W, H, 41, groups, 0, 1, Precision::kFp32, Device::kCuda);
      SensorModel on(p, W, H, 41, groups, 0, 1, Precision::kFp32, Device::kCuda);
      GpuLaunchPolicy policy;
      if (tiny) policy.target_ms = 0.001;
      else policy.max_slice_ns = p.noise ? 3 * kNoiseCellNs : 1;
      policy.max_pixels = 300;
      policy.event_capacity = 16;
      policy.slice_sort = false;
      off.set_gpu_launch_policy(policy);
      policy.slice_sort = true;
      on.set_gpu_launch_policy(policy);
      std::vector<float> irr0(W * H * 2);
      for (size_t i = 0; i < W * H; ++i) {
        irr0[2 * i] = 20.0f + 5.0f * static_cast<float>(i % 41);
        irr0[2 * i + 1] = 30.0f * static_cast<float>(i % 3);
      }
      for (SensorModel* m : {&cpu, &off, &on}) m->init(1234, irr0.data());
      // One call over three windows (off the lattice at its start), then two one-window calls.
      int64_t t = 1234;
      for (const int64_t t_end : {3 * kW23, 4 * kW23, 5 * kW23}) {
        const Clustered r = clustered_runs(W * H, t, t_end);
        const Events a = advance(cpu, r, t_end);
        CHECK(a.size() > 100);
        check_events(a, advance(off, r, t_end));
        check_events(a, advance(on, r, t_end));
        t = t_end;
      }
      CHECK(on.save_state() == cpu.save_state());
      CHECK(on.save_state() == off.save_state());
      check_counters(on.stats(), off.stats());
      const GpuLaunchStats &ls_on = on.gpu_launch_stats(), &ls_off = off.gpu_launch_stats();
      MESSAGE("variant " << static_cast<int>(v) << ", tiny " << tiny << ": " << ls_on.launches << " launches, "
                         << ls_on.slice_sorts << " sorted, " << ls_on.reruns << " re-runs");
      CHECK(ls_on.slice_sorts > 0);
      CHECK(ls_off.slice_sorts == 0);
      CHECK(ls_on.reruns > 0);
    }
  }
}

TEST_CASE("BALLA_SENSOR_SLICE_SORT switches the per-slice sort for every kernel object; other values are refused") {
  SensorParams p = preset("prophesee_gen3");
  p.hot_fraction = 0.02;
  const uint32_t W = 32, H = 24;
  const std::vector<Waveform> groups = {make_constant_waveform(), make_fourier_waveform(120.0, {0.3}, {0.5})};
  const std::vector<float> irr0(W * H * 2, 50.0f);
  const Clustered r = clustered_runs(W * H, 1'000'000, kW23);  // after the sample-free call below
  // The sorted launches of a call on a model made under the variable's value, after the policy (if any) is set.
  const auto sorted = [&](const char* value, std::optional<bool> policy) {
    const SliceSortEnv env(value);
    SensorModel gpu(p, W, H, 7, groups, 0, 1, Precision::kFp32, Device::kCuda);
    if (policy) {
      GpuLaunchPolicy pol;
      pol.slice_sort = *policy;
      gpu.set_gpu_launch_policy(pol);
    }
    gpu.init(0, irr0.data());
    gpu.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, 1'000'000);  // a sample-free call keeps the window order
    CHECK(gpu.gpu_launch_stats().slice_sorts == 0);
    advance(gpu, r, kW23);
    return gpu.gpu_launch_stats().slice_sorts;
  };
  CHECK(sorted(nullptr, std::nullopt) > 0);  // on by default (since E5b's replay A/B)
  CHECK(sorted("", std::nullopt) > 0);
  CHECK(sorted(nullptr, false) == 0);
  CHECK(sorted(nullptr, true) > 0);
  CHECK(sorted("1", std::nullopt) > 0);
  CHECK(sorted("1", false) > 0);  // the variable overrides the policy
  CHECK(sorted("0", true) == 0);
  for (const char* bad : {"yes", "2", " 1"}) {
    const SliceSortEnv env(bad);
    CHECK_THROWS_AS(SensorModel(p, W, H, 7, groups, 0, 1, Precision::kFp32, Device::kCuda), std::invalid_argument);
  }
}

TEST_SUITE_END();
