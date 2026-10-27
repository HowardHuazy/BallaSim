// Integer-nanosecond time (DESIGN §5.1, §6.10; M2 §A.4).
//
// All times are int64 ns on the simulation ("true") clock, relative to the run epoch, and may be
// negative. The scheduler works in fixed windows of W = 2^23 ns on an absolute dyadic lattice:
// window w starts at t_w = w * W, and level-l nodes sit at t_w + j * 2^(23 - l). Device code
// receives int32 offsets from a per-work-item int64 base, never window- or launch-relative floats.
#pragma once

#include <stdint.h>

#include "core/hd.h"

namespace balla::time {

inline constexpr int kWindowLog2 = 23;
inline constexpr int64_t kWindowNs = int64_t{1} << kWindowLog2;  // 8.388608 ms
inline constexpr int kMaxSampleLevel = 13;                       // finest sample spacing 1.024 µs
inline constexpr int kMaxLevel = kWindowLog2;                    // probes may refine to 1 ns

// Floor division and the matching non-negative remainder; C++ '/' truncates towards zero.
// Precondition: b > 0.
BALLA_HD constexpr int64_t floor_div(int64_t a, int64_t b) {
  const int64_t q = a / b;
  return (a % b < 0) ? q - 1 : q;
}
BALLA_HD constexpr int64_t floor_mod(int64_t a, int64_t b) {
  const int64_t r = a % b;
  return r < 0 ? r + b : r;
}
BALLA_HD constexpr int32_t floor_div(int32_t a, int32_t b) {
  const int32_t q = a / b;
  return (a % b < 0) ? q - 1 : q;
}
BALLA_HD constexpr int32_t floor_mod(int32_t a, int32_t b) {
  const int32_t r = a % b;
  return r < 0 ? r + b : r;
}

// Index of the window containing t. W is a power of two, so this is an arithmetic shift, which
// C++20 defines as floor division for negative operands.
BALLA_HD constexpr int64_t window_index(int64_t t_ns) { return t_ns >> kWindowLog2; }
BALLA_HD constexpr int64_t window_start(int64_t window) { return window * kWindowNs; }

// Spacing of level-l lattice nodes, 2^(23 - l) ns. Precondition: 0 <= level <= kMaxLevel.
BALLA_HD constexpr int64_t lattice_step(int level) { return int64_t{1} << (kWindowLog2 - level); }

// Time of node j at level l in window w. Precondition: 0 <= j <= 2^level.
BALLA_HD constexpr int64_t lattice_node(int64_t window, int level, int64_t j) {
  return window_start(window) + j * lattice_step(level);
}

// True if t lies on the level-l lattice (every level-l node is also a node of all finer levels).
BALLA_HD constexpr bool on_lattice(int64_t t_ns, int level) {
  return (t_ns & (lattice_step(level) - 1)) == 0;
}

// Coarsest level whose lattice contains t: 0 for window starts, kMaxLevel for odd nanoseconds.
BALLA_HD constexpr int coarsest_level(int64_t t_ns) {
  int level = 0;
  while (level < kMaxLevel && !on_lattice(t_ns, level)) ++level;
  return level;
}

// int32 offsets from an int64 base: |t - base| must stay below 2^31 ns (2.147 s). The distance is taken in
// uint64, exact for any two int64 times (t - base in int64 overflows for distances of 2^63 ns or more).
BALLA_HD constexpr bool offset_fits(int64_t t_ns, int64_t base_ns) {
  return t_ns >= base_ns ? static_cast<uint64_t>(t_ns) - static_cast<uint64_t>(base_ns) <= uint64_t{INT32_MAX}
                         : static_cast<uint64_t>(base_ns) - static_cast<uint64_t>(t_ns) <= uint64_t{1} << 31;
}
// Precondition: offset_fits(t_ns, base_ns).
BALLA_HD constexpr int32_t to_offset(int64_t t_ns, int64_t base_ns) {
  return static_cast<int32_t>(t_ns - base_ns);
}
BALLA_HD constexpr int64_t from_offset(int64_t base_ns, int32_t offset_ns) {
  return base_ns + offset_ns;
}

}  // namespace balla::time
