// The general pass's work estimate and sort key (sensor_cuda.cu), as functions of a pixel's samples and a time
// range. The GPU's quiet kernel computes the window's key inline (quiet_kernel: cells left + 1 + kSampleWork x the
// call's samples, quantized to 16 bits); this header states the same arithmetic for any range [t0, te) of a window,
// so that a time slice can be ordered by its own work: PHASE2 P2-E E5, the per-slice sort (p2e.md §5). Its users are
// the per-slice sort of sensor_cuda.cu (slice_key, GpuLaunchPolicy::slice_sort; docs/dev/SENSOR.md "GPU launch
// control") and the E5 study build's dry launches (CMake BALLASIM_SENSOR_DRY; SENSOR.md "Dry launches").
// sensor_cuda.cu static_asserts kWorkPerSample and kWorkDoneKey against its kSampleWork and kDoneKey, and
// tests/sensor/test_slice_order.cpp pins work_key to the quiet kernel's quantization, the counts to brute force and
// the per-slice order to a reference sort. Thread mapping never changes a result (sensor_cuda.cu), so no key here can
// change an output.
#pragma once

#include <cstdint>

#include "core/hd.h"
#include "sensor/noise_tables.h"

namespace balla::sensor {

// Key of a pixel without work (sorted last), and the work units of one sample (sensor_cuda.cu kDoneKey, kSampleWork;
// a static_assert there keeps them equal).
inline constexpr uint16_t kWorkDoneKey = 0xFFFF;
inline constexpr uint32_t kWorkPerSample = 16;

// Index of the first t[i] >= x in the strictly increasing t[0, n) (n when none).
BALLA_HD inline int32_t first_at_or_after(const int64_t* t, int32_t n, int64_t x) {
  int32_t lo = 0, hi = n;
  while (lo < hi) {
    const int32_t mid = lo + (hi - lo) / 2;
    if (t[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// Samples of t[0, n) in [t0, te) (0 for an empty range).
BALLA_HD inline int32_t samples_in(const int64_t* t, int32_t n, int64_t t0, int64_t te) {
  if (n <= 0 || te <= t0) return 0;
  return first_at_or_after(t, n, te) - first_at_or_after(t, n, t0);
}

// Work units of a pixel over [t0, te) (te > t0): the noise cells left plus one, plus kWorkPerSample per sample in the
// range: the quiet kernel's estimate (t0 its resume time, te the window's end, every sample of the call in the window)
// restricted to the range. Saturates at 2^32 - 1.
BALLA_HD inline uint32_t slice_work(const int64_t* t, int32_t n, int64_t t0, int64_t te) {
  const uint64_t cells = static_cast<uint64_t>((te - t0) >> kNoiseCellLog2) + 1u;
  const uint64_t u = cells + uint64_t{kWorkPerSample} * static_cast<uint64_t>(samples_in(t, n, t0, te));
  return u < 0xFFFFFFFFull ? static_cast<uint32_t>(u) : 0xFFFFFFFFu;
}

// The 16-bit ascending sort key of work u (descending work; kWorkDoneKey is left for pixels without work): exact up
// to 2^15 units and coarser above, the quiet kernel's quantization.
BALLA_HD inline uint16_t work_key(uint32_t u) {
  const uint32_t top = static_cast<uint32_t>(kWorkDoneKey - 1);
  uint32_t q = u;
  if (u >= 32768u) {
    q = 32768u + (u - 32768u) / 64u;
    if (q > top) q = top;
  }
  return static_cast<uint16_t>(top - q);
}

// The per-slice sort key of a pixel left for the general pass at t_resume, for the slice [ts, te) of its window (last:
// the window's last slice, which runs every such pixel): work_key of its work over [max(t_resume, ts), te), one unit
// when that range is empty in the last slice, and kWorkDoneKey when the pixel has no work in the slice (active_kernel
// leaves it alone), so that it sorts last. Ascending keys order the slice by descending work.
BALLA_HD inline uint16_t slice_key(const int64_t* t, int32_t n, int64_t t_resume, int64_t ts, int64_t te, bool last) {
  const int64_t t0 = t_resume > ts ? t_resume : ts;
  if (t0 < te) return work_key(slice_work(t, n, t0, te));
  return last ? work_key(1u) : kWorkDoneKey;
}

}  // namespace balla::sensor
