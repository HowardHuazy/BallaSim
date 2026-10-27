#include "sensor/sensor_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cub/device/device_radix_sort.cuh>
#include <stdexcept>
#include <string>

#include "core/time/time.h"
#include "kernels/device_array.h"
#include "sensor/slice_order.h"

// PHASE2 P2-E E5's study build (CMake BALLASIM_SENSOR_DRY, CUDA only; docs/dev/SENSOR.md "Dry launches"): timing-only
// re-launches of every general-pass slice in other thread orders. Every other build compiles none of it.
#if defined(BALLASIM_SENSOR_DRY) && BALLASIM_SENSOR_DRY
#define BALLA_SENSOR_DRY 1
#include <cstdio>
#include <memory>
#include <numeric>
#include <vector>
#else
#define BALLA_SENSOR_DRY 0
#endif

namespace balla::sensor {

// The quiet pass never fires: its kernel object has a sink whose push is never instantiated.
struct QuietSink {
  __device__ void push(const RawEvent&);
};

namespace {

using kernels::check_cuda;
using kernels::DeviceArray;

constexpr int kQuietBlock = 128;
constexpr int kActiveBlock = 64;
// The general pass at 6 resident blocks per SM caps it at 168 registers: its hot path fits them, and
// only the rare out-of-line walk and search (which would take 255) spill. 5-15 % faster than
// uncapped on the recorded windows; 146 registers spill the hot path and are slower.
constexpr int kActiveMinBlocks = 6;
constexpr int kStatCount = 10;
// Sort keys of the general pass: active pixels by descending work (see kSampleWork), finished pixels
// last.
constexpr uint16_t kDoneKey = 0xFFFF;
constexpr int kKeyBits = 16;
// First batch of the quiet pass, in pixels (it adapts to the launch times).
constexpr uint32_t kInitialQuietPixels = 1u << 16;
// The general pass's work estimate per pixel: one unit per cell left plus kSampleWork per sample. On
// the recorded scene13 and sfm windows a sample on a noise-cell boundary costs about one lit cell and
// one inside a cell about fifty (it cuts pieces whose waveform bounds are evaluated directly, and the
// warps of the heaviest pixels serialize their lanes' different pieces); 16 fits their mix in the
// window totals to about 15 %.
constexpr uint32_t kSampleWork = 16;
// slice_order.h restates the window key over any time range (the per-slice sort, GpuLaunchPolicy::slice_sort, and the
// E5 study build's dry launches): it must keep these constants, and quiet_kernel's quantization below (no code is
// generated for the check).
static_assert(kSampleWork == kWorkPerSample && kDoneKey == kWorkDoneKey,
              "slice_order.h's work estimate and key must follow the general pass's (kSampleWork, kDoneKey)");
// Cost of a work unit before any launch of a full wave has been timed (ms): about twice the measured
// cost.
constexpr double kInitialUnitMs = 3e-6;
// The heaviest pixel's cost per work unit before any launch of less than a full wave has been timed
// (ms). A pixel's chain of sub-steps runs on one thread, so its latency bounds a launch from below (on
// the RTX 2080 Ti 5-9 us per unit on the recorded windows, where scene13's heaviest pixels hold
// ~13,500 units per window); heavy pixels therefore get time slices. Units do not count events, tree
// walks or sub-step halvings, which can make a pixel several times slower per unit, so the cost is
// learnt.
constexpr double kInitialLatencyMs = 7.5e-3;
// Cells and grid intervals covered by one window's waveform table (one more for a window that
// starts off the grid).
constexpr int kTableCells = static_cast<int>(kPlainSegmentNs / kNoiseCellNs) + 1;
constexpr int kTableGrid = static_cast<int>(kPlainSegmentNs / kWaveGridNs) + 1;

// Device counters of a call (one array, reset with memsets).
enum Counter : int {
  kEvents = 0,                       // events in the buffer (this call)
  kLaunchStats = 1,                  // [1, 11): work counters of the current general-pass launch
  kNonFinite = kLaunchStats + kStatCount,
  kClamped,
  kBadRuns,                          // bits: 1 pixel id out of range, 2 duplicate id, 4 run out of range (this call)
  kActive,                           // pixels left for the general pass (this window)
  kWork,                             // their work, and the heaviest one's (this window)
  kHeaviest,
  kQuietStats,                       // [kQuietStats, + 10): work counters of the quiet pass (this window)
  kCounters = kQuietStats + kStatCount
};

struct DeviceSink {
  RawEvent* buf;
  unsigned long long* count;
  unsigned long long capacity;
  __device__ void push(const RawEvent& e) const {
    const unsigned long long i = atomicAdd(count, 1ull);
    if (i < capacity) buf[i] = e;  // an overflowing launch is detected and re-run by the host
  }
};

// Device-resident runs of one call, as kernels read them.
struct Runs {
  const int32_t* row_of;  // pixel -> run, or -1
  const int64_t* t;
  const float* irr;
  const int64_t* row_start;
  const int32_t* row_count;
};

__device__ InputRun input_of(const Runs& r, uint32_t p, int groups) {
  InputRun in{nullptr, nullptr, 0};
  const int32_t row = r.row_of[p];
  if (row >= 0) {
    in.t = r.t + r.row_start[row];
    in.irr = r.irr + static_cast<size_t>(r.row_start[row]) * static_cast<size_t>(groups);
    in.n = r.row_count[row];
  }
  return in;
}

// Adds a thread's work counters to out[0, 10): block reduction in shared memory, one atomic per
// block and counter.
__device__ void add_stats(const KernelStats32& st, unsigned long long* out) {
  __shared__ unsigned long long block_stats[kStatCount];
  if (threadIdx.x < kStatCount) block_stats[threadIdx.x] = 0;
  __syncthreads();
  const uint32_t v[kStatCount] = {st.substeps, st.substep_attempts, st.settled_substeps,
                                  st.cells,    st.refinements,      st.leaf_intervals,
                                  st.fallbacks, st.input_clamps,    st.quarantines, st.pieces};
  for (int k = 0; k < kStatCount; ++k) {
    if (v[k] != 0) atomicAdd(&block_stats[k], static_cast<unsigned long long>(v[k]));
  }
  __syncthreads();
  if (threadIdx.x < kStatCount && block_stats[threadIdx.x] != 0) atomicAdd(&out[threadIdx.x], block_stats[threadIdx.x]);
}

// The quiet pass of one window over the pixels [first, first + count), in image order: quiet cells
// while they last (PixelKernel::advance_quiet). A pixel that does not finish the window is left for
// the general pass at t_resume[p], with its sort key; key[p] = kDoneKey otherwise. Does nothing when
// the call's input check found a non-finite sample or a bad run (no state changes). kHpf: the high-pass filter's
// instantiation (with kGlare and kP2 true), on the side states hpf. kP2 false: Phase 2's per-cell paths compiled out
// (pixel.h; frozen diffusion; cascade_kernels).
template <bool kGlare, bool kHpf, bool kP2>
__global__ void __launch_bounds__(kQuietBlock) quiet_kernel(const __grid_constant__ KernelConstants<float> kc,
                                                              const PixelMap* maps, PixelState* states, Runs runs,
                                                              int64_t t_call, int64_t t_from, int64_t t_to,
                                                              bool ends_call, uint32_t first, uint32_t count,
                                                              int64_t* t_resume, uint16_t* key, uint8_t* live,
                                                              unsigned long long* counters, HpfState* hpf) {
  if (counters[kNonFinite] != 0 || counters[kBadRuns] != 0) return;
  KernelStats32 st;
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t p = first + i;
  bool active = false;
  unsigned long long work = 0;
  if (i < count) {
    const InputRun in = input_of(runs, p, kc.groups);
    QuietSink sink;
    PixelKernel<float, QuietSink, true, KernelStats32, kGlare, kHpf, kP2> k(kc, maps[p], p, states[p], t_call, in, sink,
                                                                            st, -1, kHpf ? hpf + p : nullptr);
    int64_t t = t_from;
    bool done = false;
    if (k.may_be_quiet()) {
      t = k.advance_quiet(t_from, t_to, ends_call);
      done = t == t_to;
    }
    if (done) {
      key[p] = kDoneKey;
    } else {
      // Sorted by descending work: pixels with equal work mostly share one lattice level, i.e. the same
      // sample times, so a warp's threads walk the same pieces (thread mapping never changes a result).
      // The key is exact up to 2^15 units and coarser above (slice_order.h's work_key restates it: change both).
      const uint32_t u = static_cast<uint32_t>((t_to - t) >> kNoiseCellLog2) + 1u + kSampleWork * static_cast<uint32_t>(in.n);
      const uint32_t q = u < 32768u ? u : min(32768u + (u - 32768u) / 64u, static_cast<uint32_t>(kDoneKey - 1));
      key[p] = static_cast<uint16_t>(kDoneKey - 1 - q);
      t_resume[p] = t;
      live[p] = static_cast<uint8_t>(k.live_mask());
      active = true;
      work = u;
    }
  }
  // One atomic per warp and counter.
  const unsigned mask = __ballot_sync(0xFFFFFFFFu, active);
  unsigned long long sum = work, top = work;
  for (int off = 16; off > 0; off /= 2) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, off);
    top = max(top, __shfl_down_sync(0xFFFFFFFFu, top, off));
  }
  if ((threadIdx.x & 31u) == 0 && mask != 0) {
    atomicAdd(&counters[kActive], static_cast<unsigned long long>(__popc(mask)));
    atomicAdd(&counters[kWork], sum);
    atomicMax(&counters[kHeaviest], top);
  }
  add_stats(st, counters + kQuietStats);
}

// The general pass over sorted positions [first, first + count) for the time slice [ts, te) of the
// window: each pixel advances from max(t_resume, ts); the window's last slice ends its call when
// ends_call. Each thread saves its pixel's state to backup[position] first (or, when the host re-runs
// a launch whose events overflowed the buffer, restores it from there). kHpf: the high-pass filter's instantiation
// (with kGlare and kP2 true), on the side states hpf, backed up to hpf_backup alike. kP2 as quiet_kernel's.
template <bool kGlare, bool kHpf, bool kP2>
__global__ void __launch_bounds__(kActiveBlock, kActiveMinBlocks)
active_kernel(const __grid_constant__ KernelConstants<float> kc, const PixelMap* maps, PixelState* states,
              PixelState* backup, const uint32_t* order, uint32_t first, uint32_t count, bool restore, Runs runs,
              int64_t t_call, int64_t ts, int64_t te, bool last, bool ends_call, const int64_t* t_resume,
              const uint8_t* live, const __grid_constant__ DeviceSink sink, unsigned long long* stats_out,
              HpfState* hpf, HpfState* hpf_backup) {
  KernelStats32 st;
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const uint32_t pos = first + i;
    const uint32_t p = order[pos];
    const int64_t t0 = max(t_resume[p], ts);
    if (t0 < te || last) {
      if (restore) states[p] = backup[pos];
      else backup[pos] = states[p];
      if constexpr (kHpf) {
        if (restore) hpf[p] = hpf_backup[pos];
        else hpf_backup[pos] = hpf[p];
      }
      const InputRun in = input_of(runs, p, kc.groups);
      PixelKernel<float, const DeviceSink, false, KernelStats32, kGlare, kHpf, kP2> k(
          kc, maps[p], p, states[p], t_call, in, sink, st, live[p], kHpf ? hpf + p : nullptr);
      k.advance_slice(t0, te, last && ends_call);
    }
  }
  add_stats(st, stats_out);
}

// The general pass of an instantaneous-response sensor (D3; PixelKernel::advance_instant), with active_kernel's
// arguments and duties: its own kernel, so that the cascade kernels' code is not touched.
template <bool kGlare>
__global__ void __launch_bounds__(kActiveBlock)
instant_kernel(const __grid_constant__ KernelConstants<float> kc, const PixelMap* maps, PixelState* states,
               PixelState* backup, const uint32_t* order, uint32_t first, uint32_t count, bool restore, Runs runs,
               int64_t t_call, int64_t ts, int64_t te, bool last, bool ends_call, const int64_t* t_resume,
               const uint8_t* live, const __grid_constant__ DeviceSink sink, unsigned long long* stats_out) {
  KernelStats32 st;
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const uint32_t pos = first + i;
    const uint32_t p = order[pos];
    const int64_t t0 = max(t_resume[p], ts);
    if (t0 < te || last) {
      if (restore) states[p] = backup[pos];
      else backup[pos] = states[p];
      const InputRun in = input_of(runs, p, kc.groups);
      PixelKernel<float, const DeviceSink, false, KernelStats32, kGlare> k(kc, maps[p], p, states[p], t_call, in, sink,
                                                                            st, live[p]);
      k.advance_instant(t0, te, last && ends_call);
    }
  }
  add_stats(st, stats_out);
}

// The cascade kernels of a window (quiet and general pass): the high-pass filter's instantiation has glare and Phase 2's
// paths compiled in (pixel.h kHpf); otherwise glare (kGlare) and the transient diffusion (kP2, P2-E E4) are compiled in
// only for calls that have them, so that a frozen, glare-free call runs the code of before both. Ten kernels.
struct CascadeKernels {
  decltype(&quiet_kernel<false, false, false>) quiet;
  decltype(&active_kernel<false, false, false>) active;
};

template <bool kGlare, bool kHpf, bool kP2>
CascadeKernels cascade_instance() {
  return {quiet_kernel<kGlare, kHpf, kP2>, active_kernel<kGlare, kHpf, kP2>};
}

CascadeKernels cascade_kernels(const KernelConstants<float>& kc) {
  if (kc.hpf) return cascade_instance<true, true, true>();
  if (kc.transient) return kc.glare.on ? cascade_instance<true, false, true>() : cascade_instance<false, false, true>();
  return kc.glare.on ? cascade_instance<true, false, false>() : cascade_instance<false, false, false>();
}

// The per-slice sort's keys (PHASE2 P2-E E5; GpuLaunchPolicy::slice_sort): for each position first + i of a
// general-pass launch in the window's order, its pixel (ids[i]) and the pixel's slice_key over the slice [ts, te)
// (key[i]).
__global__ void slice_key_kernel(const uint32_t* order, uint32_t first, uint32_t count, const int64_t* t_resume,
                                 Runs runs, int groups, int64_t ts, int64_t te, bool last, uint16_t* key,
                                 uint32_t* ids) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint32_t p = order[first + i];
  const InputRun in = input_of(runs, p, groups);
  key[i] = slice_key(in.t, in.n, t_resume[p], ts, te, last);
  ids[i] = p;
}

// GpuLaunchPolicy::slice_sort as the environment variable BALLA_SENSOR_SLICE_SORT sets it, when a kernel object is made
// and at every set_policy (so that it also reaches the tests' policies): 1 on, 0 off, unset or empty the policy's own
// value; any other value is refused.
bool slice_sort_from_env(bool fallback) {
  const char* e = std::getenv("BALLA_SENSOR_SLICE_SORT");
  if (e == nullptr || e[0] == '\0') return fallback;
  if (std::strcmp(e, "1") == 0) return true;
  if (std::strcmp(e, "0") == 0) return false;
  throw std::invalid_argument(std::string("BALLA_SENSOR_SLICE_SORT must be 0 or 1, not '") + e + "'");
}

#if BALLA_SENSOR_DRY
// The dry launches' slice keys (P2-E E5, slice_order.h): per position i of [first, first + count) of the window's
// order, the pixel (ids[i]) and its key over [max(t_resume, ts), te) (kWorkDoneKey without work in the slice);
// busy[0] += the pixels with work, busy[1] += their work units (one atomic per warp and counter).
__global__ void dry_slice_key_kernel(const uint32_t* order, uint32_t first, uint32_t count, const int64_t* t_resume,
                                     Runs runs, int groups, int64_t ts, int64_t te, bool last, uint16_t* key,
                                     uint32_t* ids, unsigned long long* busy) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long units = 0;
  bool has_work = false;
  if (i < count) {
    const uint32_t p = order[first + i];
    const int64_t t0 = max(t_resume[p], ts);
    uint16_t k = kWorkDoneKey;
    if (t0 < te || last) {
      const InputRun in = input_of(runs, p, groups);
      const uint32_t u = t0 < te ? slice_work(in.t, in.n, t0, te) : 1u;
      k = work_key(u);
      units = u;
      has_work = true;
    }
    key[i] = k;
    ids[i] = p;
  }
  const unsigned mask = __ballot_sync(0xFFFFFFFFu, has_work);
  for (int off = 16; off > 0; off /= 2) units += __shfl_down_sync(0xFFFFFFFFu, units, off);
  if ((threadIdx.x & 31u) == 0 && mask != 0) {
    atomicAdd(&busy[0], static_cast<unsigned long long>(__popc(mask)));
    atomicAdd(&busy[1], units);
  }
}

// active_kernel's work (never re-run, never restoring) with each thread's counted work over its pixel's slice in
// work[i] (0 without work): the hindsight order's profile. The count is the thread's own KernelStats32 (cells, pieces,
// sub-step attempts, refinements, leaf intervals, filter splits), unweighted: deterministic and free of the warp's
// other lanes, which a clock read after advance_slice is not (the lanes that finish early wait at the reconvergence
// point for the warp's heaviest, so every lane would read about the warp's time).
template <bool kGlare, bool kHpf, bool kP2>
__global__ void __launch_bounds__(kActiveBlock, kActiveMinBlocks)
dry_prof_kernel(const __grid_constant__ KernelConstants<float> kc, const PixelMap* maps, PixelState* states,
                PixelState* backup, const uint32_t* order, uint32_t first, uint32_t count, Runs runs, int64_t t_call,
                int64_t ts, int64_t te, bool last, bool ends_call, const int64_t* t_resume, const uint8_t* live,
                const __grid_constant__ DeviceSink sink, unsigned long long* stats_out, HpfState* hpf,
                HpfState* hpf_backup, uint32_t* work) {
  KernelStats32 st;
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const uint32_t pos = first + i;
    const uint32_t p = order[pos];
    const int64_t t0 = max(t_resume[p], ts);
    if (t0 < te || last) {
      backup[pos] = states[p];
      if constexpr (kHpf) hpf_backup[pos] = hpf[p];
      const InputRun in = input_of(runs, p, kc.groups);
      PixelKernel<float, const DeviceSink, false, KernelStats32, kGlare, kHpf, kP2> k(
          kc, maps[p], p, states[p], t_call, in, sink, st, live[p], kHpf ? hpf + p : nullptr);
      k.advance_slice(t0, te, last && ends_call);
    }
    const unsigned long long w = static_cast<unsigned long long>(st.cells) + st.pieces + st.substep_attempts +
                                 st.refinements + st.leaf_intervals + st.hpf_splits;
    work[i] = w > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t>(w);
  }
  add_stats(st, stats_out);
}

using DryProfKernel = decltype(&dry_prof_kernel<false, false, false>);

// The profiling kernel of a call: cascade_kernels' instantiation.
DryProfKernel dry_prof_for(const KernelConstants<float>& kc) {
  if (kc.hpf) return dry_prof_kernel<true, true, true>;
  if (kc.transient) return kc.glare.on ? dry_prof_kernel<true, false, true> : dry_prof_kernel<false, false, true>;
  return kc.glare.on ? dry_prof_kernel<true, false, false> : dry_prof_kernel<false, false, false>;
}
#endif

// row_of[pixel[i]] = i for the listed pixels (row_of is -1 elsewhere); *bad |= 1 for an id out of
// range (not written), 2 for a duplicate id.
__global__ void scatter_rows_kernel(const uint32_t* pixel, uint32_t n, uint32_t n_pixels, int32_t* row_of,
                                    unsigned long long* bad) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint32_t p = pixel[i];
  if (p >= n_pixels) atomicOr(bad, 1ull);
  else if (atomicExch(&row_of[p], static_cast<int32_t>(i)) != -1) atomicOr(bad, 2ull);
}

// The input each pixel holds after the call: its run's last sample (row_of: its run, or -1), else its held input
// (also for a run outside the n_samples samples, which advance_device then refuses).
__global__ void end_frame_kernel(const PixelState* states, const int32_t* row_of, const int64_t* row_start,
                                 const int32_t* row_count, const float* irr, size_t n_samples, uint32_t n, int groups,
                                 float* out) {
  const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  const int32_t row = row_of[p];
  const float* v = states[p].held;
  if (row >= 0 && row_count[row] > 0 && row_start[row] >= 0 &&
      static_cast<size_t>(row_start[row]) + static_cast<size_t>(row_count[row]) <= n_samples) {
    v = irr + static_cast<size_t>(row_start[row] + row_count[row] - 1) * static_cast<size_t>(groups);
  }
  for (int g = 0; g < groups; ++g) out[static_cast<size_t>(p) * groups + g] = v[g];
}

__global__ void iota_kernel(uint32_t n, uint32_t* out) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = i;
}

// The values the runs take, one warp per run (blocks of whole warps), as advance_runs checks them:
// counters[kNonFinite] += non-finite values, counters[kClamped] += values that clamp_irradiance
// changes (one atomic per warp and counter); a run outside the n_samples samples is not read and sets
// counters[kBadRuns] |= 4.
__global__ void check_inputs_kernel(const float* irr, const int64_t* row_start, const int32_t* row_count, size_t n_runs,
                                    size_t n_samples, int groups, float irr_max, unsigned long long* counters) {
  const size_t run = (blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x) >> 5;
  if (run >= n_runs) return;  // whole warps
  const unsigned lane = threadIdx.x & 31u;
  const int64_t s = row_start[run];
  const int32_t c = row_count[run];
  if (s < 0 || c < 0 || static_cast<size_t>(s) + static_cast<size_t>(c) > n_samples) {
    if (lane == 0) atomicOr(&counters[kBadRuns], 4ull);
    return;
  }
  const float* v = irr + static_cast<size_t>(s) * static_cast<size_t>(groups);
  const size_t n = static_cast<size_t>(c) * static_cast<size_t>(groups);
  unsigned long long n_bad = 0, n_clamped = 0;
  for (size_t j = lane; j - lane < n; j += 32) {  // the same trip count on every lane: full-mask ballots
    const float x = j < n ? v[j] : 0.0f;
    const bool bad = j < n && !isfinite(x);
    const bool clamped = j < n && !bad && clamp_irradiance(x, irr_max) != x;
    n_bad += __popc(__ballot_sync(0xFFFFFFFFu, bad));
    n_clamped += __popc(__ballot_sync(0xFFFFFFFFu, clamped));
  }
  if (lane == 0) {
    if (n_bad != 0) atomicAdd(&counters[kNonFinite], n_bad);
    if (n_clamped != 0) atomicAdd(&counters[kClamped], n_clamped);
  }
}

// The waveform table of a window (pixel.h, WaveTable): values and next breakpoints of every
// non-constant group at the grid points, and its bounds over whole cells.
__global__ void wave_table_kernel(const __grid_constant__ KernelConstants<float> kc, WaveTable<float> w, float* value,
                                  int64_t* breakpoint, float* bounds) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int g = static_cast<int>(blockIdx.y);
  if (kc.waves[g].type == WaveformType::kConstant) return;
  const Waveform& wave = kc.waves[g];
  if (j <= w.n_grid) {
    const int64_t t = w.t0 + j * kWaveGridNs;
    value[g * (w.n_grid + 1) + j] = waveform_value<float>(wave, t);
    breakpoint[g * (w.n_grid + 1) + j] = waveform_next_breakpoint(wave, t);
  }
  if (j < w.n_cells) {
    const int64_t t = (w.k0 + j) * kNoiseCellNs;
    const WaveBounds<float> b = waveform_bounds<float>(wave, t, t + kNoiseCellNs);
    float* out = bounds + (static_cast<int64_t>(g) * w.n_cells + j) * 3;
    out[0] = b.m_min;
    out[1] = b.dm_max;
    out[2] = b.d2m_max;
  }
}

unsigned blocks_for(uint64_t n, int block) { return static_cast<unsigned>((n + block - 1) / block); }

// The kernel object's pinned counter mirror and timing events, released also when Impl's constructor throws.
struct HostResources {
  unsigned long long* host = nullptr;  // pinned mirror of the counters
  cudaEvent_t start = nullptr, stop = nullptr;
  HostResources() = default;
  HostResources(const HostResources&) = delete;
  HostResources& operator=(const HostResources&) = delete;
  ~HostResources() {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (host != nullptr) cudaFreeHost(host);
  }
};

// The per-slice sort's buffers (GpuLaunchPolicy::slice_sort), made before the first call that sorts: the keys and
// pixels of a launch's positions (from 0), their sorted keys, and the launch's pixels in slice order (at the launch's
// own positions, as active_kernel reads them and its backup is indexed).
struct SliceSort {
  DeviceArray<uint16_t> key, key_sorted;
  DeviceArray<uint32_t> ids, order;
  explicit SliceSort(size_t n) : key(n), key_sorted(n), ids(n), order(n) {}
};

#if BALLA_SENSOR_DRY
// The dry launches of P2-E E5's study build (MB-6; docs/dev/SENSOR.md "Dry launches"). On with the environment
// variable BALLA_SENSOR_DRY=1 when the kernel object is made. Before every general-pass launch of a call with samples,
// on scratch copies of the states (and filter states), never on the call's own:
//   1. the slice's own order (E5's candidate): dry_slice_key_kernel and a 16-bit radix sort of the launch's positions,
//      timed together (E5's cost per slice: sort_ms);
//   2. a profiling launch in the window's order (dry_prof_kernel: each thread's counted work in the slice), then the
//      hindsight order (the positions by descending counted work: a reference order, not a bound on any order);
//   3. the slice in the window's order (today's), the slice's order and the hindsight order, each from the same
//      scratch states, in an order that rotates with the slice (window_ms, slice_ms, hindsight_ms).
// The real launch follows unchanged. One JSON line per call ("call") and per launch ("slice", with the real launch's
// ms and events, every dry launch's events, the busy pixels and work units, and the warp-max sums of the profiled
// work under the three orders) goes to the file BALLA_SENSOR_DRY_OUT (appended), else to stderr.
struct DryHook {
  static constexpr unsigned long long kDryEvents = 1ull << 20;
  static constexpr int kBusy = 1 + kStatCount;  // counters: [0] events, [1, 1 + kStatCount) stats, then busy[2]
  size_t n;
  DeviceArray<PixelState> states, backup;
  std::unique_ptr<DeviceArray<HpfState>> hpf, hpf_backup;
  DeviceArray<RawEvent> events;
  DeviceArray<unsigned long long> counters;
  DeviceArray<uint16_t> key, key_sorted;
  DeviceArray<uint32_t> ids, order_slice, order_hind, work, work_sorted;
  std::unique_ptr<DeviceArray<uint8_t>> temp;
  cudaEvent_t a = nullptr, b = nullptr;
  FILE* out = stderr;
  bool own_out = false;
  unsigned long long calls = 0, slices = 0;
  bool call_on = false;

  static bool enabled() {
    const char* e = std::getenv("BALLA_SENSOR_DRY");
    return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
  }

  explicit DryHook(size_t n_pixels)
      : n(n_pixels), states(n), backup(n), events(kDryEvents), counters(kBusy + 2), key(n), key_sorted(n), ids(n),
        order_slice(n), order_hind(n), work(n), work_sorted(n) {
    size_t b1 = 0, b2 = 0;
    check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, b1, key.get(), key_sorted.get(), ids.get(), order_slice.get(),
                                               static_cast<int>(n), 0, kKeyBits),
               "dry sort size");
    check_cuda(cub::DeviceRadixSort::SortPairsDescending(nullptr, b2, work.get(), work_sorted.get(), ids.get(),
                                                         order_hind.get(), static_cast<int>(n), 0, 32),
               "dry hindsight sort size");
    temp = std::make_unique<DeviceArray<uint8_t>>(std::max<size_t>({b1, b2, 1}));
    check_cuda(cudaEventCreate(&a), "cudaEventCreate");
    check_cuda(cudaEventCreate(&b), "cudaEventCreate");
    if (const char* path = std::getenv("BALLA_SENSOR_DRY_OUT"); path != nullptr && path[0] != '\0') {
      FILE* f = std::fopen(path, "a");
      if (f == nullptr) throw std::runtime_error(std::string("BALLA_SENSOR_DRY_OUT: cannot open ") + path);
      out = f;
      own_out = true;
    }
  }
  DryHook(const DryHook&) = delete;
  DryHook& operator=(const DryHook&) = delete;
  ~DryHook() {
    if (a != nullptr) cudaEventDestroy(a);
    if (b != nullptr) cudaEventDestroy(b);
    if (own_out) std::fclose(out);
  }

  // Duration (ms) of what f enqueues on the stream.
  template <class F>
  float timed(cudaStream_t stream, F&& f) {
    check_cuda(cudaEventRecord(a, stream), "dry event");
    f();
    check_cuda(cudaGetLastError(), "dry launch");
    check_cuda(cudaEventRecord(b, stream), "dry event");
    check_cuda(cudaEventSynchronize(b), "dry sync");
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime");
    return ms;
  }

  // The scratch states from the call's (and the filter's), and zeroed dry counters.
  void scratch(const PixelState* from, const HpfState* hpf_from, cudaStream_t stream) {
    check_cuda(cudaMemcpyAsync(states.get(), from, n * sizeof(PixelState), cudaMemcpyDeviceToDevice, stream), "dry");
    if (hpf_from != nullptr) {
      if (!hpf) {
        hpf = std::make_unique<DeviceArray<HpfState>>(n);
        hpf_backup = std::make_unique<DeviceArray<HpfState>>(n);
      }
      check_cuda(cudaMemcpyAsync(hpf->get(), hpf_from, n * sizeof(HpfState), cudaMemcpyDeviceToDevice, stream), "dry");
    }
    check_cuda(cudaMemsetAsync(counters.get(), 0, kBusy * sizeof(unsigned long long), stream), "dry");
  }
  DeviceSink sink() const { return {events.get(), counters.get(), kDryEvents}; }
  unsigned long long read(int i, cudaStream_t stream) const {
    unsigned long long v = 0;
    check_cuda(cudaMemcpyAsync(&v, counters.get() + i, sizeof(v), cudaMemcpyDeviceToHost, stream), "dry counter");
    check_cuda(cudaStreamSynchronize(stream), "dry counter");
    return v;
  }

  void begin_call(int64_t t_state, int64_t t_end, size_t n_runs) {
    call_on = n_runs > 0;  // the sample-free calls (a replay's warm-up) are not measured
    std::fprintf(out, "{\"kind\": \"call\", \"call\": %llu, \"t_state\": %lld, \"t_end\": %lld, \"runs\": %zu}\n",
                 ++calls, static_cast<long long>(t_state), static_cast<long long>(t_end), n_runs);
    std::fflush(out);
  }
};

// What the dry launches of one general-pass launch measured (DryHook).
struct DryRecord {
  bool on = false;
  int64_t ts = 0, te = 0;
  uint32_t first = 0, count = 0;
  bool last = false;
  int rot = 0;
  float sort_ms = 0, prof_ms = 0, ms[3] = {0, 0, 0};  // ms: window, slice, hindsight order
  unsigned long long busy = 0, units = 0, events[3] = {0, 0, 0};
  double work_sum = 0, work_max = 0, warp_max[3] = {0, 0, 0};  // the profile's counted work
};

// Sum over warps (32 consecutive threads of a launch) of the largest counted work of their lanes, when thread i runs
// the position perm[i] of the profile (window order).
double warp_max_sum(const std::vector<uint32_t>& work, const std::vector<uint32_t>& perm) {
  double sum = 0;
  for (size_t w = 0; w < perm.size(); w += 32) {
    uint32_t m = 0;
    for (size_t j = w; j < std::min(perm.size(), w + 32); ++j) m = std::max(m, work[perm[j]]);
    sum += m;
  }
  return sum;
}
#endif

}  // namespace

int sensor_cuda_device_count() {
  // Any error (no device, no driver or a broken one) is no usable device, as in optix_device_available().
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();  // clear the error, so that later checks do not see it
    return 0;
  }
  return count;
}

struct CudaSensorKernel::Impl : HostResources {
  KernelConstants<float> kc;
  DeviceArray<PixelMap> maps;
  DeviceArray<float> tables;
  DeviceArray<PixelState> states;
  DeviceArray<PixelState> backup;            // the general pass's states, for re-running a launch after an overflow
  DeviceArray<unsigned long long> counters;  // Counter
  DeviceArray<int32_t> row_of;               // pixel -> its run, or -1
  DeviceArray<int64_t> t_resume;             // where the general pass resumes each pixel
  DeviceArray<uint8_t> live;                 // and its live-group mask (pixel.h)
  DeviceArray<uint16_t> key, key_sorted;     // sort keys of the general pass
  DeviceArray<uint32_t> iota, order;         // pixel ids, and the pixels in general-pass order
  DeviceArray<float> wave_value, wave_bounds;
  DeviceArray<int64_t> wave_breakpoint;
  std::unique_ptr<DeviceArray<float>> frame;  // end_frame's (veiling glare only)
  std::unique_ptr<DeviceArray<uint8_t>> sort_temp;  // the window sort's (and the per-slice sort's) temporary storage
  std::unique_ptr<SliceSort> slice;                  // the per-slice sort's buffers, once a call sorts
  // The high-pass filter (enable_hpf): its side states, their backup for overflow re-runs, its per-pixel rates.
  std::unique_ptr<DeviceArray<HpfState>> hpf, hpf_backup;
  std::unique_ptr<DeviceArray<float>> hpf_b;
  std::unique_ptr<DeviceArray<uint32_t>> pixel_ids;  // an ROI model's frame ids (set_pixel_ids)
  size_t n_pixels;
  GpuLaunchPolicy policy;
  unsigned long long capacity;
  std::unique_ptr<DeviceArray<RawEvent>> events;
  uint32_t full_wave = 1;                       // threads of one full wave of the general pass
  uint32_t quiet_pixels = kInitialQuietPixels;  // the quiet pass's batch
  double unit_ms = kInitialUnitMs;              // running estimate of a work unit's cost in the general pass
  double latency_ms = kInitialLatencyMs;        // and of the heaviest pixel's cost per unit of its own work
  bool timed = false;                           // a general-pass launch has been timed
#if BALLA_SENSOR_DRY
  std::unique_ptr<DryHook> dry;  // P2-E E5's dry launches, when BALLA_SENSOR_DRY=1
#endif

  Impl(const KernelConstants<float>& base, const std::vector<PixelMap>& m, const std::vector<float>& tab)
      : kc(base), maps(std::span<const PixelMap>(m)), tables(std::span<const float>(tab)), states(m.size()),
        backup(m.size()), counters(kCounters), row_of(m.size()), t_resume(m.size()), live(m.size()), key(m.size()),
        key_sorted(m.size()), iota(m.size()), order(m.size()),
        wave_value(static_cast<size_t>(kMaxLightGroups) * (kTableGrid + 1)),
        wave_bounds(static_cast<size_t>(kMaxLightGroups) * kTableCells * 3),
        wave_breakpoint(static_cast<size_t>(kMaxLightGroups) * (kTableGrid + 1)), n_pixels(m.size()),
        capacity(policy.event_capacity), events(std::make_unique<DeviceArray<RawEvent>>(capacity)) {
    kc.tables.data = tables.size() > 0 ? tables.get() : nullptr;
    policy.slice_sort = slice_sort_from_env(policy.slice_sort);
    kc.hpf = false;  // until enable_hpf (its rates map is a device array)
    kc.hpf_b_map = nullptr;
    kc.pixel_ids = nullptr;  // until set_pixel_ids (a device array)
    if (n_pixels > 0) {
      iota_kernel<<<blocks_for(n_pixels, 256), 256>>>(static_cast<uint32_t>(n_pixels), iota.get());
      check_cuda(cudaGetLastError(), "iota launch");
      size_t bytes = 0;
      check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, bytes, key.get(), key_sorted.get(), iota.get(), order.get(),
                                                 static_cast<int>(n_pixels), 0, kKeyBits),
                 "sensor sort size");
      sort_temp = std::make_unique<DeviceArray<uint8_t>>(std::max<size_t>(bytes, 1));
    }
    check_cuda(cudaMallocHost(&host, kCounters * sizeof(unsigned long long)), "cudaMallocHost");
    int device = 0, sms = 0, blocks_per_sm = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "cudaDeviceGetAttribute");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, active_kernel<false, false, false>,
                                                             kActiveBlock, 0),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    full_wave = static_cast<uint32_t>(std::max(sms * blocks_per_sm, 1) * kActiveBlock);
    check_cuda(cudaEventCreate(&start), "cudaEventCreate");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate");
#if BALLA_SENSOR_DRY
    if (n_pixels > 0 && DryHook::enabled()) dry = std::make_unique<DryHook>(n_pixels);
#endif
    landed();
  }
  // cudaMemcpy from pageable memory may return before its DMA lands, and advance_device may run on a
  // non-blocking stream: uploads wait here until they are complete.
  static void landed() { check_cuda(cudaStreamSynchronize(nullptr), "upload"); }

  // Counters [lo, hi) to the pinned mirror, then waits for the stream.
  void read(int lo, int hi, cudaStream_t stream) {
    check_cuda(cudaMemcpyAsync(host + lo, counters.get() + lo, static_cast<size_t>(hi - lo) * sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, stream),
               "sensor counters");
    check_cuda(cudaStreamSynchronize(stream), "sensor");
  }

  // Time of the launch between the start and stop events (ms), added to the launch counters.
  double time_launch(GpuLaunchStats& ls, bool rerun) const {
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    ++ls.launches;
    if (rerun) ++ls.reruns;
    ls.total_ms += static_cast<double>(ms);
    ls.max_ms = std::max(ls.max_ms, static_cast<double>(ms));
    return static_cast<double>(ms);
  }

  static KernelStats stats_at(const unsigned long long* h) {
    KernelStats k;
    k.substeps = h[0];
    k.substep_attempts = h[1];
    k.settled_substeps = h[2];
    k.cells = h[3];
    k.refinements = h[4];
    k.leaf_intervals = h[5];
    k.fallbacks = h[6];
    k.input_clamps = h[7];
    k.quarantines = h[8];
    k.pieces = h[9];
    return k;
  }

  // The per-slice order of the general-pass launch over the positions [first, first + count) of the window's order and
  // the slice [ts, te) (GpuLaunchPolicy::slice_sort; PHASE2 P2-E E5): the same pixels at the same positions of
  // slice->order, by ascending slice_key (descending work in the slice, those without work last), ties in the window's
  // order (the radix sort is stable). Enqueued on the stream.
  void sort_slice(const Runs& r, int groups, uint32_t first, uint32_t count, int64_t ts, int64_t te, bool last,
                  cudaStream_t stream) {
    slice_key_kernel<<<blocks_for(count, 256), 256, 0, stream>>>(order.get(), first, count, t_resume.get(), r, groups,
                                                                 ts, te, last, slice->key.get(), slice->ids.get());
    check_cuda(cudaGetLastError(), "slice key launch");
    size_t bytes = sort_temp->size();  // sized for all pixels (the window sort's), so for any launch of them
    check_cuda(cub::DeviceRadixSort::SortPairs(sort_temp->get(), bytes, slice->key.get(), slice->key_sorted.get(),
                                               slice->ids.get(), slice->order.get() + first, static_cast<int>(count), 0,
                                               kKeyBits, stream),
               "slice sort");
  }

  // One launch of the general pass with its overflow re-runs; returns its duration (ms). sorted: in the per-slice order
  // (sort_slice, timed with the launch; its re-runs keep that order), else in the window's.
  double general_launch(const KernelConstants<float>& kc_window, const Runs& r, uint32_t first, uint32_t count,
                        int64_t t_call, int64_t ts, int64_t te, bool last, bool ends_call, bool sorted,
                        unsigned long long& total, cudaStream_t stream, KernelStats& stats,
                        GpuLaunchStats& launch_stats) {
    unsigned long long* c = counters.get();
    const uint32_t* const pixels = sorted ? slice->order.get() : order.get();
    bool restore = false;
    while (true) {
      host[kEvents] = total;
      std::fill(host + kLaunchStats, host + kLaunchStats + kStatCount, 0ull);
      check_cuda(cudaMemcpyAsync(c, host, (1 + kStatCount) * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream),
                 "counter reset");
      const DeviceSink sink{events->get(), c + kEvents, capacity};
      check_cuda(cudaEventRecord(start, stream), "cudaEventRecord");
      if (sorted && !restore) {
        sort_slice(r, kc_window.groups, first, count, ts, te, last, stream);
        ++launch_stats.slice_sorts;
      }
      const bool glare = kc_window.glare.on;  // the glare-free instantiations keep their code (pixel.h kGlare)
      if (kc_window.instant) {
        (glare ? instant_kernel<true> : instant_kernel<false>)<<<blocks_for(count, kActiveBlock), kActiveBlock, 0,
                                                                 stream>>>(
            kc_window, maps.get(), states.get(), backup.get(), pixels, first, count, restore, r, t_call, ts, te, last,
            ends_call, t_resume.get(), live.get(), sink, c + kLaunchStats);
      } else {
        const auto active = cascade_kernels(kc_window).active;
        active<<<blocks_for(count, kActiveBlock), kActiveBlock, 0, stream>>>(
            kc_window, maps.get(), states.get(), backup.get(), pixels, first, count, restore, r, t_call, ts, te, last,
            ends_call, t_resume.get(), live.get(), sink, c + kLaunchStats, hpf ? hpf->get() : nullptr,
            hpf_backup ? hpf_backup->get() : nullptr);
      }
      check_cuda(cudaGetLastError(), "sensor general launch");
      check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord");
      read(kEvents, kLaunchStats + kStatCount, stream);
      const double ms = time_launch(launch_stats, restore);
      if (host[kEvents] <= capacity) {
        total = host[kEvents];
        stats.add(stats_at(host + kLaunchStats));
        return ms;
      }
      // Overflow: grow the buffer, keeping the events of earlier launches, and re-run this one.
      const unsigned long long grown_capacity = host[kEvents] + host[kEvents] / 4;
      auto grown = std::make_unique<DeviceArray<RawEvent>>(grown_capacity);
      if (total > 0) {
        check_cuda(cudaMemcpyAsync(grown->get(), events->get(), total * sizeof(RawEvent), cudaMemcpyDeviceToDevice,
                                   stream),
                   "event buffer growth");
        check_cuda(cudaStreamSynchronize(stream), "event buffer growth");
      }
      events = std::move(grown);  // the buffer and its capacity change together, once both steps succeeded
      capacity = grown_capacity;
      restore = true;
    }
  }

#if BALLA_SENSOR_DRY
  // DryHook's measurement of the general-pass launch over positions [first, first + count) and the slice [ts, te),
  // made before the launch itself (the call's states are only read).
  DryRecord dry_measure(const KernelConstants<float>& kc_window, const Runs& r, uint32_t first, uint32_t count,
                        int64_t t_call, int64_t ts, int64_t te, bool last, bool ends_call, cudaStream_t stream) {
    DryRecord rec;
    if (!dry || !dry->call_on || kc_window.instant || count == 0) return rec;
    DryHook& d = *dry;
    rec.on = true;
    rec.ts = ts;
    rec.te = te;
    rec.first = first;
    rec.count = count;
    rec.last = last;
    rec.rot = static_cast<int>(d.slices % 3);
    ++d.slices;
    // 1. The slice's own order: its keys and their sort, as E5 would run them before the launch.
    check_cuda(cudaMemsetAsync(d.counters.get() + DryHook::kBusy, 0, 2 * sizeof(unsigned long long), stream), "dry");
    rec.sort_ms = d.timed(stream, [&] {
      dry_slice_key_kernel<<<blocks_for(count, 256), 256, 0, stream>>>(order.get(), first, count, t_resume.get(), r,
                                                                       kc_window.groups, ts, te, last, d.key.get(),
                                                                       d.ids.get(), d.counters.get() + DryHook::kBusy);
      size_t bytes = d.temp->size();
      check_cuda(cub::DeviceRadixSort::SortPairs(d.temp->get(), bytes, d.key.get(), d.key_sorted.get(), d.ids.get(),
                                                 d.order_slice.get() + first, static_cast<int>(count), 0, kKeyBits,
                                                 stream),
                 "dry sort");
    });
    rec.busy = d.read(DryHook::kBusy, stream);
    rec.units = d.read(DryHook::kBusy + 1, stream);
    // 2. The profile in the window's order, and the hindsight order from it.
    HpfState* const hpf_from = hpf ? hpf->get() : nullptr;
    d.scratch(states.get(), hpf_from, stream);
    const DryProfKernel prof = dry_prof_for(kc_window);
    rec.prof_ms = d.timed(stream, [&] {
      prof<<<blocks_for(count, kActiveBlock), kActiveBlock, 0, stream>>>(
          kc_window, maps.get(), d.states.get(), d.backup.get(), order.get(), first, count, r, t_call, ts, te, last,
          ends_call, t_resume.get(), live.get(), d.sink(), d.counters.get() + 1, hpf_from ? d.hpf->get() : nullptr,
          hpf_from ? d.hpf_backup->get() : nullptr, d.work.get());
    });
    size_t bytes = d.temp->size();
    check_cuda(cub::DeviceRadixSort::SortPairsDescending(d.temp->get(), bytes, d.work.get(), d.work_sorted.get(),
                                                         d.ids.get(), d.order_hind.get() + first,
                                                         static_cast<int>(count), 0, 32, stream),
               "dry hindsight sort");
    // 3. The slice in the three orders, from the same scratch states, the first order rotating with the slice.
    const auto active = cascade_kernels(kc_window).active;
    const uint32_t* orders[3] = {order.get(), d.order_slice.get(), d.order_hind.get()};
    for (int j = 0; j < 3; ++j) {
      const int v = (rec.rot + j) % 3;
      d.scratch(states.get(), hpf_from, stream);
      rec.ms[v] = d.timed(stream, [&] {
        active<<<blocks_for(count, kActiveBlock), kActiveBlock, 0, stream>>>(
            kc_window, maps.get(), d.states.get(), d.backup.get(), orders[v], first, count, false, r, t_call, ts, te,
            last, ends_call, t_resume.get(), live.get(), d.sink(), d.counters.get() + 1,
            hpf_from ? d.hpf->get() : nullptr, hpf_from ? d.hpf_backup->get() : nullptr);
      });
      rec.events[v] = d.read(0, stream);
    }
    // The warp-max sums of the profile's counted work under the three orders (host side; positions relative to
    // first): the CPU census's model with the kernel's own counts in place of its work estimate.
    std::vector<uint32_t> work(count), perm(count);
    std::vector<uint16_t> keys(count);
    check_cuda(cudaMemcpyAsync(work.data(), d.work.get(), count * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream),
               "dry work");
    check_cuda(cudaMemcpyAsync(keys.data(), d.key.get(), count * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream),
               "dry keys");
    check_cuda(cudaStreamSynchronize(stream), "dry work");
    for (uint32_t i = 0; i < count; ++i) {
      perm[i] = i;
      rec.work_sum += work[i];
      rec.work_max = std::max(rec.work_max, static_cast<double>(work[i]));
    }
    rec.warp_max[0] = warp_max_sum(work, perm);
    std::stable_sort(perm.begin(), perm.end(), [&](uint32_t x, uint32_t y) { return keys[x] < keys[y]; });
    rec.warp_max[1] = warp_max_sum(work, perm);
    std::iota(perm.begin(), perm.end(), 0u);
    std::stable_sort(perm.begin(), perm.end(), [&](uint32_t x, uint32_t y) { return work[x] > work[y]; });
    rec.warp_max[2] = warp_max_sum(work, perm);
    return rec;
  }

  // One JSON line for a measured launch, with the real launch's duration and events.
  void dry_emit(const DryRecord& rec, double real_ms, unsigned long long real_events) const {
    if (!rec.on) return;
    std::fprintf(dry->out,
                 "{\"kind\": \"slice\", \"call\": %llu, \"ts\": %lld, \"te\": %lld, \"first\": %u, \"count\": %u, "
                 "\"last\": %d, \"rot\": %d, \"busy\": %llu, \"units\": %llu, \"real_ms\": %.4f, \"window_ms\": %.4f, "
                 "\"slice_ms\": %.4f, \"hindsight_ms\": %.4f, \"prof_ms\": %.4f, \"sort_ms\": %.4f, "
                 "\"events_real\": %llu, \"events\": [%llu, %llu, %llu], \"work_sum\": %.0f, \"work_max\": %.0f, "
                 "\"warp_max\": [%.0f, %.0f, %.0f]}\n",
                 dry->calls, static_cast<long long>(rec.ts), static_cast<long long>(rec.te), rec.first, rec.count,
                 rec.last ? 1 : 0, rec.rot, rec.busy, rec.units, real_ms, static_cast<double>(rec.ms[0]),
                 static_cast<double>(rec.ms[1]), static_cast<double>(rec.ms[2]), static_cast<double>(rec.prof_ms),
                 static_cast<double>(rec.sort_ms), real_events, rec.events[0], rec.events[1], rec.events[2],
                 rec.work_sum, rec.work_max, rec.warp_max[0], rec.warp_max[1], rec.warp_max[2]);
    std::fflush(dry->out);
  }
#endif
};

CudaSensorKernel::CudaSensorKernel(const KernelConstants<float>& kc, const std::vector<PixelMap>& maps,
                                   const std::vector<float>& tables) {
  if (sensor_cuda_device_count() == 0) throw std::runtime_error("no CUDA device available");
  impl_ = std::make_unique<Impl>(kc, maps, tables);
}

CudaSensorKernel::~CudaSensorKernel() = default;

void CudaSensorKernel::upload_states(const std::vector<PixelState>& states, const std::vector<HpfState>& hpf) {
  if (states.size() != impl_->n_pixels) throw std::invalid_argument("upload_states: wrong number of pixels");
  if (!hpf.empty() && (!impl_->hpf || hpf.size() != impl_->n_pixels)) {
    throw std::invalid_argument("upload_states: filter states without the filter's kernels, or of a wrong size");
  }
  impl_->states.upload(states.data());
  if (!hpf.empty()) impl_->hpf->upload(hpf.data());
  Impl::landed();
}

void CudaSensorKernel::download_states(std::vector<PixelState>& states) const {
  states.resize(impl_->n_pixels);
  impl_->states.download(states.data());
}

void CudaSensorKernel::download_hpf(std::vector<HpfState>& hpf) const {
  hpf.assign(impl_->hpf ? impl_->n_pixels : 0, HpfState{});
  if (impl_->hpf) impl_->hpf->download(hpf.data());
}

void CudaSensorKernel::enable_hpf(const KernelConstants<float>& kc, const std::vector<float>& hpf_b) {
  Impl& s = *impl_;
  if (!hpf_b.empty() && hpf_b.size() != s.n_pixels) throw std::invalid_argument("enable_hpf: wrong number of rates");
  auto states = std::make_unique<DeviceArray<HpfState>>(s.n_pixels);
  auto backup = std::make_unique<DeviceArray<HpfState>>(s.n_pixels);
  std::unique_ptr<DeviceArray<float>> rates;
  if (!hpf_b.empty()) rates = std::make_unique<DeviceArray<float>>(std::span<const float>(hpf_b));
  if (s.n_pixels > 0) check_cuda(cudaMemset(states->get(), 0, s.n_pixels * sizeof(HpfState)), "filter states");
  Impl::landed();
  s.hpf = std::move(states);
  s.hpf_backup = std::move(backup);
  s.hpf_b = std::move(rates);
  s.kc.hpf = kc.hpf;
  s.kc.hpf_b = kc.hpf_b;
  s.kc.eps_hp = kc.eps_hp;
  s.kc.hpf_b_map = s.hpf_b ? s.hpf_b->get() : nullptr;
}

void CudaSensorKernel::set_pixel_ids(const std::vector<uint32_t>& ids) {
  Impl& s = *impl_;
  if (ids.size() != s.n_pixels) throw std::invalid_argument("set_pixel_ids: one id per pixel");
  s.pixel_ids = std::make_unique<DeviceArray<uint32_t>>(std::span<const uint32_t>(ids));
  Impl::landed();
  s.kc.pixel_ids = s.pixel_ids->get();
}

void CudaSensorKernel::set_policy(const GpuLaunchPolicy& policy) {
  if (!(policy.target_ms > 0) || policy.max_slice_ns < 0 || policy.event_capacity == 0) {
    throw std::invalid_argument("GpuLaunchPolicy: need target_ms > 0, max_slice_ns >= 0 and event_capacity >= 1");
  }
  const bool slice_sort = slice_sort_from_env(policy.slice_sort);
  auto events = std::make_unique<DeviceArray<RawEvent>>(policy.event_capacity);  // a failure changes nothing
  impl_->policy = policy;
  impl_->policy.slice_sort = slice_sort;
  impl_->capacity = policy.event_capacity;
  impl_->events = std::move(events);
}

void CudaSensorKernel::set_photometry(const KernelConstants<float>& kc, const std::vector<PixelMap>& maps) {
  if (maps.size() != impl_->n_pixels) throw std::invalid_argument("set_photometry: wrong number of pixels");
  impl_->kc.scale = kc.scale;
  impl_->kc.irr_max = kc.irr_max;
  impl_->maps.upload(maps.data());
}

void CudaSensorKernel::set_glare(const GlareRamp<float>& glare) { impl_->kc.glare = glare; }

std::vector<float> CudaSensorKernel::end_frame(const DeviceRuns& runs, void* stream_ptr) {
  Impl& s = *impl_;
  const auto stream = static_cast<cudaStream_t>(stream_ptr);
  const size_t groups = static_cast<size_t>(s.kc.groups);
  std::vector<float> out(s.n_pixels * groups);
  if (out.empty()) return out;
  if (!s.frame) s.frame = std::make_unique<DeviceArray<float>>(out.size());
  check_cuda(cudaMemsetAsync(s.row_of.get(), 0xFF, s.n_pixels * sizeof(int32_t), stream), "row map");
  if (runs.n > 0) {
    // The flags of bad ids are dropped: advance_device resets them and checks the ids again.
    scatter_rows_kernel<<<blocks_for(runs.n, 256), 256, 0, stream>>>(runs.pixel, static_cast<uint32_t>(runs.n),
                                                                     static_cast<uint32_t>(s.n_pixels), s.row_of.get(),
                                                                     s.counters.get() + kBadRuns);
    check_cuda(cudaGetLastError(), "row map launch");
  }
  end_frame_kernel<<<blocks_for(s.n_pixels, 256), 256, 0, stream>>>(
      s.states.get(), s.row_of.get(), runs.row_start, runs.row_count, runs.irr, runs.n_samples,
      static_cast<uint32_t>(s.n_pixels), s.kc.groups, s.frame->get());
  check_cuda(cudaGetLastError(), "end frame launch");
  check_cuda(cudaMemcpyAsync(out.data(), s.frame->get(), out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
             "end frame");
  check_cuda(cudaStreamSynchronize(stream), "end frame");
  return out;
}

std::vector<RawEvent> CudaSensorKernel::advance(const uint32_t* pixel_ids, size_t n, const int64_t* t, const float* irr,
                                                size_t n_samples, const int64_t* row_start, const int32_t* row_count,
                                                int64_t t_state, int64_t t_end, KernelStats& stats,
                                                GpuLaunchStats& launch_stats) {
  const size_t groups = static_cast<size_t>(impl_->kc.groups);
  const DeviceArray<uint32_t> d_pixel{std::span<const uint32_t>(pixel_ids, n)};
  const DeviceArray<int64_t> d_t{std::span<const int64_t>(t, n_samples)};
  const DeviceArray<float> d_irr{std::span<const float>(irr, n_samples * groups)};
  const DeviceArray<int64_t> d_start{std::span<const int64_t>(row_start, n)};
  const DeviceArray<int32_t> d_count{std::span<const int32_t>(row_count, n)};
  const DeviceRuns runs{d_pixel.get(), d_start.get(), d_count.get(), n, d_t.get(), d_irr.get(), n_samples};
  return download(advance_device(runs, t_state, t_end, nullptr, false, stats, launch_stats));
}

std::vector<RawEvent> CudaSensorKernel::download(const DeviceEvents& events) const {
  std::vector<RawEvent> out(events.count);
  if (events.count > 0) {
    check_cuda(cudaMemcpy(out.data(), events.data, events.count * sizeof(RawEvent), cudaMemcpyDeviceToHost),
               "event download");
  }
  return out;
}

DeviceEvents CudaSensorKernel::advance_device(const DeviceRuns& runs, int64_t t_state, int64_t t_end, void* stream_ptr,
                                              bool check_inputs, KernelStats& stats, GpuLaunchStats& launch_stats) {
  Impl& s = *impl_;
  const auto stream = static_cast<cudaStream_t>(stream_ptr);
  const size_t groups = static_cast<size_t>(s.kc.groups);
  const uint32_t n = static_cast<uint32_t>(s.n_pixels);
  unsigned long long* c = s.counters.get();
  check_cuda(cudaMemsetAsync(c, 0, kCounters * sizeof(unsigned long long), stream), "sensor counters");
  // The call's input: the pixel -> run map, and the check of the pixel ids, the runs and the samples
  // they take (a bad one stops the passes before any state changes; it is reported at the first sync).
  check_cuda(cudaMemsetAsync(s.row_of.get(), 0xFF, s.n_pixels * sizeof(int32_t), stream), "row map");
  if (runs.n > 0) {
    scatter_rows_kernel<<<blocks_for(runs.n, 256), 256, 0, stream>>>(runs.pixel, static_cast<uint32_t>(runs.n), n,
                                                                     s.row_of.get(), c + kBadRuns);
    check_cuda(cudaGetLastError(), "row map launch");
  }
  if (check_inputs && runs.n > 0) {
    check_inputs_kernel<<<blocks_for(runs.n, 8), 256, 0, stream>>>(runs.irr, runs.row_start, runs.row_count, runs.n,
                                                                   runs.n_samples, s.kc.groups, s.kc.irr_max, c);
    check_cuda(cudaGetLastError(), "input check launch");
  }
  const Runs r{s.row_of.get(), runs.t, runs.irr, runs.row_start, runs.row_count};
  const int64_t grain = slice_grain(s.kc.noise);
  // The per-slice sort (GpuLaunchPolicy::slice_sort; PHASE2 P2-E E5): the cascade's general pass of a call with
  // samples, where MB-6 measured it (the sample-free calls, such as a warm-up, and the instantaneous response keep the
  // window's order). Its buffers are made here, outside any launch's timed span.
  const bool sort_slices = s.policy.slice_sort && runs.n > 0 && !s.kc.instant && n > 0;
  if (sort_slices && !s.slice) s.slice = std::make_unique<SliceSort>(s.n_pixels);
#if BALLA_SENSOR_DRY
  if (s.dry) s.dry->begin_call(t_state, t_end, runs.n);
#endif

  // Windows: slices of the call at most one 2^23 ns window long (window boundaries are multiples of
  // every grain), shorter under policy.max_slice_ns. Per window the quiet pass and the sort, then the
  // general pass in time slices of whole grains (adapted to the launch times), each over the sorted
  // active pixels in batches of at most policy.max_pixels.
  unsigned long long total = 0;  // events so far in this call
  int64_t t_from = t_state;
  bool first_window = true;
  while (first_window || t_from < t_end) {
    int64_t t_to = std::min(t_end, (time::floor_div(t_from, kPlainSegmentNs) + 1) * kPlainSegmentNs);
    if (s.policy.max_slice_ns > 0 && t_to - t_from > s.policy.max_slice_ns) {
      const int64_t base = time::floor_div(t_from, grain) * grain;
      t_to = std::max(base + (t_from - base + s.policy.max_slice_ns) / grain * grain, base + grain);
      t_to = std::min(t_to, t_end);
    }
    const bool ends_call = t_to == t_end;
    KernelConstants<float> kc = s.kc;
    if (kc.modulated && t_to > t_from) {
      WaveTable<float>& w = kc.table;
      w.t0 = time::floor_div(t_from, kWaveGridNs) * kWaveGridNs;
      w.n_grid = static_cast<int32_t>((t_to - w.t0 + kWaveGridNs - 1) / kWaveGridNs);
      w.k0 = time::floor_div(t_from, kNoiseCellNs);
      w.n_cells = static_cast<int32_t>(time::floor_div(t_to - 1, kNoiseCellNs) - w.k0 + 1);
      w.value = s.wave_value.get();
      w.breakpoint = s.wave_breakpoint.get();
      w.bounds = s.wave_bounds.get();
      wave_table_kernel<<<dim3(blocks_for(static_cast<uint64_t>(w.n_grid) + 1, 128), static_cast<unsigned>(groups)), 128,
                          0, stream>>>(kc, w, s.wave_value.get(), s.wave_breakpoint.get(), s.wave_bounds.get());
      check_cuda(cudaGetLastError(), "wave table launch");
    }

    check_cuda(cudaMemsetAsync(c + kActive, 0, (kCounters - kActive) * sizeof(unsigned long long), stream),
               "sensor counters");
    // Quiet pass in pixel batches (a batch that took longer than the target shortens the next ones at
    // once, a shorter one lengthens them at most twofold), then the sort of the remaining pixels.
    for (uint32_t first = 0; first < n;) {
      const uint32_t count = std::min(s.quiet_pixels, n - first);
      check_cuda(cudaEventRecord(s.start, stream), "cudaEventRecord");
      const auto quiet = cascade_kernels(kc).quiet;
      quiet<<<blocks_for(count, kQuietBlock), kQuietBlock, 0, stream>>>(
          kc, s.maps.get(), s.states.get(), r, t_state, t_from, t_to, ends_call, first, count, s.t_resume.get(),
          s.key.get(), s.live.get(), c, s.hpf ? s.hpf->get() : nullptr);
      check_cuda(cudaGetLastError(), "sensor quiet launch");
      check_cuda(cudaEventRecord(s.stop, stream), "cudaEventRecord");
      s.read(kNonFinite, kCounters, stream);
      if (first_window && s.host[kBadRuns] != 0) {  // the lowest bit first, with the messages of advance_runs
        const unsigned long long bad = s.host[kBadRuns];
        throw std::invalid_argument((bad & 1) != 0   ? "SensorModel.advance: pixel id out of range"
                                    : (bad & 2) != 0 ? "SensorModel.advance: duplicate pixel id"
                                                     : "SensorModel.advance: sample run out of range");
      }
      if (first_window && s.host[kNonFinite] > 0) {
        throw std::invalid_argument("SensorModel.advance: " + std::to_string(s.host[kNonFinite]) +
                                    " non-finite irradiance samples");
      }
      const double ms = s.time_launch(launch_stats, false);
      first += count;
      const double want = static_cast<double>(count) * s.policy.target_ms / std::max(ms, 1e-3);
      s.quiet_pixels = static_cast<uint32_t>(
          std::clamp(std::min(want, 2.0 * count), std::min(1024.0, static_cast<double>(n)), static_cast<double>(n)));
    }
    // The remaining pixels sorted by descending work, and cut into batches of about the target time.
    const uint32_t n_active = static_cast<uint32_t>(s.host[kActive]);
    if (first_window) stats.input_clamps += s.host[kClamped];
    stats.add(Impl::stats_at(s.host + kQuietStats));
    const double total_work = static_cast<double>(s.host[kWork]), heaviest = static_cast<double>(s.host[kHeaviest]);
    if (n_active > 0) {
      size_t bytes = s.sort_temp->size();
      check_cuda(cub::DeviceRadixSort::SortPairs(s.sort_temp->get(), bytes, s.key.get(), s.key_sorted.get(),
                                                 s.iota.get(), s.order.get(), static_cast<int>(n), 0, kKeyBits, stream),
                 "sensor sort");
    }

    // General pass over the sorted active pixels (in batches of at most policy.max_pixels), in time
    // slices on the grain (bitwise invariant, pixel.h slice_grain; without noise only whole windows).
    // Before each slice the rest of the window is planned again: enough slices for the estimated time
    // (work x cost per unit) and for the heaviest pixel's own latency (its work x its cost per unit),
    // the window's first slice four times shorter, also when one slice would do (the content, and so
    // the cost, may differ from the last window's), and the kernel object's very first slice a single
    // grain (nothing is measured yet). Both costs follow the launches (a higher one at once, a lower
    // one at most halving it): the cost per unit of all work from launches of a full wave or more,
    // which are bound by throughput, and the heaviest pixel's from the first batch of smaller
    // launches, which are bound by their pixels' latency (a full launch would credit its throughput
    // time to the heaviest pixel, many times its latency).
    const uint32_t batch = s.policy.max_pixels > 0 ? s.policy.max_pixels : std::max(n_active, 1u);
    const double window_ns = static_cast<double>(std::max<int64_t>(t_to - t_from, 1));
    int64_t ts = t_from;
    bool first_slice = true;
    while (n_active > 0) {
      int64_t te = t_to;
      const double frac = static_cast<double>(t_to - ts) / window_ns;
      if (s.kc.noise) {
        const int64_t base = time::floor_div(ts, grain) * grain;
        const int64_t left = (t_to - base + grain - 1) / grain;  // grains left
        double k = std::max({1.0, std::ceil(total_work * frac * s.unit_ms / s.policy.target_ms),
                             std::ceil(heaviest * frac * s.latency_ms / s.policy.target_ms)});
        if (first_slice) k *= 4.0;
        if (!s.timed) k = static_cast<double>(left);
        const int64_t per = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(static_cast<double>(left) / k)));
        te = std::min(t_to, base + per * grain);
      }
      const bool last = te == t_to;
      const double share = static_cast<double>(te - ts) / window_ns;  // the slice's share of the window's work
      for (uint32_t first = 0; first < n_active; first += batch) {
        const uint32_t count = std::min(batch, n_active - first);
#if BALLA_SENSOR_DRY
        const DryRecord dry_rec = s.dry_measure(kc, r, first, count, t_state, ts, te, last, ends_call, stream);
        const unsigned long long total_before = total;
#endif
        const double ms = s.general_launch(kc, r, first, count, t_state, ts, te, last, ends_call, sort_slices, total,
                                           stream, stats, launch_stats);
#if BALLA_SENSOR_DRY
        s.dry_emit(dry_rec, ms, total - total_before);
#endif
        s.timed = true;
        const double units = total_work * share * count / n_active;
        if (count >= s.full_wave) {
          if (units >= 1.0) s.unit_ms = std::max(ms / units, 0.5 * s.unit_ms);
        } else if (first == 0 && heaviest * share >= 1.0) {
          s.latency_ms = std::max(ms / (heaviest * share), 0.5 * s.latency_ms);
        }
      }
      if (last) break;
      ts = te;
      first_slice = false;
    }
    first_window = false;
    t_from = t_to;
  }
  return {s.events->get(), static_cast<size_t>(total)};
}

}  // namespace balla::sensor
