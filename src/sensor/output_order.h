// The engine's output order and window cut (ENGINE_CPP §1.3 K15; engine/output.h), shared with the re-simulation
// (phase3/ui.md §3.10): the events of a window and the held ones are sorted by three stable LSD passes, by t_cross,
// by (x, y) x-major and by t_reported, which orders them by (t_reported, x, y, t_cross) (two events tie in
// (t_reported, x, y) only at one pixel, where the earlier t_cross comes first; a pixel never fires twice in one ns);
// a window releases the events reported before its end on the sensor clock. It depends on sensor/readout.h only, so
// that the re-simulation reuses it without render or engine headers.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/hd.h"
#include "sensor/readout.h"

namespace balla::sensor {

// S1 /gt/label of a sensor label (C3: 0 signal, 1 noise, 2 leak, 3 hot -> S1 0 signal, 2 noise, 3 leak, 4 hot).
BALLA_HD inline uint8_t label_s1(uint8_t c3) {
  constexpr uint8_t kLabelC3ToS1[4] = {0, 2, 3, 4};
  return kLabelC3ToS1[c3 & 3];
}

// Sort key of pass 0 (t_cross), 1 ((x, y), x-major: x << y_bits | y, with y < 2^y_bits) or 2 (t_reported); signed
// times are biased to unsigned order.
BALLA_HD inline uint64_t order_key(int pass, int64_t t_cross, uint16_t x, uint16_t y, int64_t t_reported, int y_bits) {
  constexpr uint64_t kBias = uint64_t{1} << 63;
  return pass == 0   ? static_cast<uint64_t>(t_cross) ^ kBias
         : pass == 1 ? (static_cast<uint64_t>(x) << y_bits) | y
                     : static_cast<uint64_t>(t_reported) ^ kBias;
}

// Bits of pass 1's key: 16 for x and y_bits for y.
BALLA_HD inline int order_key_bits(int pass, int y_bits) { return pass == 1 ? 16 + y_bits : 64; }

// The y_bits of a camera of this height: the smallest b >= 1 with 2^b >= height (the engine's Camera::y_bits).
inline int order_y_bits(uint32_t height) {
  int b = 1;
  while ((uint64_t{1} << b) < height) ++b;
  return b;
}

// The window cut: events reported before the window end on the sensor clock are final, since a later window's are
// reported at or after it (d0 >= q). The readout's own formula with a zero delay and no quantum (offset and drift);
// non-decreasing in t for |drift| < 1e6 ppm. With no drift it is t_window_end + offset.
BALLA_HD inline int64_t output_cut(const Readout& r, int64_t t_window_end_ns) {
  Readout clock = r;
  clock.delay_ns = 0;
  clock.quantum_ns = 0;
  return reported_time(clock, t_window_end_ns);
}

namespace detail {

// vals reordered stably by keys (whose entry i belongs to vals[i]); keys is left in that order too. LSD radix over
// 16-bit digits, skipping a digit that every key shares; small inputs take a stable comparison sort (the same order).
inline void stable_sort_by_key(std::vector<uint64_t>& keys, std::vector<uint32_t>& vals, int bits) {
  const size_t n = keys.size();
  if (n < 2) return;
  if (n < 1024) {
    std::vector<uint32_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = static_cast<uint32_t>(i);
    std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
    std::vector<uint64_t> k2(n);
    std::vector<uint32_t> v2(n);
    for (size_t i = 0; i < n; ++i) {
      k2[i] = keys[idx[i]];
      v2[i] = vals[idx[i]];
    }
    keys.swap(k2);
    vals.swap(v2);
    return;
  }
  std::vector<uint64_t> k2(n);
  std::vector<uint32_t> v2(n);
  std::vector<uint32_t> count(65537);
  for (int shift = 0; shift < bits; shift += 16) {
    std::fill(count.begin(), count.end(), 0u);
    for (size_t i = 0; i < n; ++i) ++count[((keys[i] >> shift) & 0xFFFF) + 1];
    if (count[((keys[0] >> shift) & 0xFFFF) + 1] == n) continue;  // one digit value: the pass changes nothing
    for (size_t d = 1; d < count.size(); ++d) count[d] += count[d - 1];
    for (size_t i = 0; i < n; ++i) {
      const uint32_t at = count[(keys[i] >> shift) & 0xFFFF]++;
      k2[at] = keys[i];
      v2[at] = vals[i];
    }
    keys.swap(k2);
    vals.swap(v2);
  }
}

}  // namespace detail

// Host: the permutation that puts n events into output order (perm[i] = the index of the i-th event in that order),
// by the engine's three stable passes over order_key (engine/output.h OutKeys).
inline void output_order(std::span<const int64_t> t_cross, std::span<const uint16_t> x, std::span<const uint16_t> y,
                         std::span<const int64_t> t_reported, int y_bits, std::vector<uint32_t>& perm) {
  const size_t n = t_cross.size();
  if (x.size() != n || y.size() != n || t_reported.size() != n) {
    throw std::invalid_argument("output_order: the columns must have one entry per event");
  }
  if (n > UINT32_MAX) throw std::invalid_argument("output_order: at most 2^32 - 1 events");
  perm.resize(n);
  for (size_t i = 0; i < n; ++i) perm[i] = static_cast<uint32_t>(i);
  std::vector<uint64_t> keys(n);
  for (int pass = 0; pass < 3; ++pass) {
    for (size_t i = 0; i < n; ++i) {
      const uint32_t e = perm[i];
      keys[i] = order_key(pass, t_cross[e], x[e], y[e], t_reported[e], y_bits);
    }
    detail::stable_sort_by_key(keys, perm, order_key_bits(pass, y_bits));
  }
}

}  // namespace balla::sensor
