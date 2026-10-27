// The browser UI's event wire (docs/design/phase3/ui.md §3.5, §3.10, §4.1–4.2, §4.5): the level-of-detail (LOD)
// level of an event, the LOD order of a block, the BSEV `ev` words, GT columns in LOD order, the BSRT rate bins and
// the per-event test of a region (BSMK masks). Host-only integer code; Python (ballasim/ui/streams.py, ui/compare.py,
// viz/stats.py's MaskRegion) only reads files and frames messages around it.
//
// Blocks are the absolute time slices [k 2^28, (k + 1) 2^28) ns of a stream's own clock. Within a block, events are
// served in LOD order: level descending, file order within a level, so that every prefix [0, n_ge[j]) holds exactly
// the events of level >= j, a binomial thinning with P(level >= j) = 2^-j. The level is a function of the event alone
// (its time, pixel, polarity and the occurrence index k of a duplicate), so two streams that share an event give it
// the same level, and differences of sampled images cancel common events exactly (§4.1, check C1).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace balla::wire {

inline constexpr int kBlockShift = 28, kLevels = 8, kMaxCoord = 16383;
inline constexpr int64_t kBlockNs = int64_t{1} << kBlockShift;
// Folded into a closed file's stream id (ui.md §3.3), so that a change of event_key moves every cached URL.
inline constexpr uint32_t kLodKeyVersion = 1;
// S1 labels 0 SIGNAL ... 4 HOT; the BSEV tag is 1 + label (0 without GT; 6 and 7 are reserved).
inline constexpr uint32_t kMaxLabel = 4;

// SplitMix64's output function, its gamma step included (Steele, Lea, Flood 2014; check C1's `mix`).
constexpr uint64_t mix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// The pixel part of the key: x | y<<16 | p<<32 with p in {0, 1}.
constexpr uint64_t pixel_word(uint32_t x, uint32_t y, uint32_t p) {
  return uint64_t{x} | (uint64_t{y} << 16) | (uint64_t{p != 0} << 32);
}

// key = mix(mix(t_ns) ^ (x | y<<16 | p<<32 | k<<33)), §4.1. k: the occurrence index of (x, y, p) among the events
// with the same t_ns, in file order (0 for unique events). Any change of this function bumps kLodKeyVersion.
constexpr uint64_t event_key(int64_t t_ns, uint32_t x, uint32_t y, uint32_t p, uint32_t k) {
  return mix64(mix64(static_cast<uint64_t>(t_ns)) ^ pixel_word(x, y, p) ^ (uint64_t{k} << 33));
}

// min(8, clz32(key >> 32)): P(level >= j) = 2^-j, and the levels nest.
constexpr uint32_t lod_level(uint64_t key) {
  const uint32_t h = static_cast<uint32_t>(key >> 32);
  const uint32_t z = h == 0 ? 32u : static_cast<uint32_t>(__builtin_clz(h));
  return z < static_cast<uint32_t>(kLevels) ? z : static_cast<uint32_t>(kLevels);
}

struct BlockOrder {
  std::vector<uint32_t> perm;                      // LOD position -> file index within the block
  std::array<uint32_t, kLevels + 1> n_ge{};        // n_ge[j] = #events with level >= j (n_ge[0] = n)
  uint32_t duplicates = 0;                         // events with k > 0: equal (x, y, p, t_ns) to an earlier event
};

// The events of one block in file order (time-sorted; std::invalid_argument otherwise) -> level descending, file
// order within a level (a counting sort). k is counted per run of equal t_ns: by a scan for short runs, a small
// open-addressing table for long ones (real EVIMO2 streams decoded from float32 seconds, §4.1).
BlockOrder lod_order(std::span<const int64_t> t_ns, std::span<const uint16_t> x, std::span<const uint16_t> y,
                     std::span<const uint8_t> p);

// lod_order and pack_ev of a whole block fused (ui.md §4.2 step 2): writes the `ev` words of all n events in LOD
// order into out[2 n], equal to pack_ev(..., lod_order(...).perm, 0, n, t_base_ns, out), but scattered in file order
// (sequential reads), and returns the order without its perm. Refusals as both.
BlockOrder pack_block(std::span<const int64_t> t_ns, std::span<const uint16_t> x, std::span<const uint16_t> y,
                      std::span<const uint8_t> p, const uint8_t* label, int64_t t_base_ns, uint32_t* out);

// The `ev` words of positions [i0, i1) of an order (perm empty: the identity, time order) into out[2 (i1 - i0)]:
// word 0 = x | y<<14 | tag<<28 | p<<31 (tag = 1 + the S1 label, 0 without labels; p = 1 for ON), word 1 =
// t - t_base_ns, which must lie in [0, 2^32). std::invalid_argument on a coordinate above kMaxCoord, a label above
// kMaxLabel, a time outside that range or a perm entry outside the events; label, if given, has one entry per event.
void pack_ev(std::span<const int64_t> t_ns, std::span<const uint16_t> x, std::span<const uint16_t> y,
             std::span<const uint8_t> p, const uint8_t* label, std::span<const uint32_t> perm, uint32_t i0,
             uint32_t i1, int64_t t_base_ns, uint32_t* out);

// A GT column in LOD order: dst[i] = src[perm[i]] for elements of elem_bytes, the destination elements
// dst_stride_bytes apart (0: packed), so that several columns interleave (flow_x, flow_y). The caller checks that
// every perm entry addresses src.
void gather(std::span<const uint32_t> perm, const void* src, size_t elem_bytes, void* dst,
            size_t dst_stride_bytes = 0);

// BSRT bins: adds each event with t0_ns <= t < t0_ns + n bin_ns (n = total.size()) to total[(t - t0_ns) / bin_ns]
// and, when `on` is not empty (it then has n entries), its ON events to on[...]. Other events are skipped; the input
// need not be sorted. std::invalid_argument for bin_ns <= 0 or mismatched sizes.
void rate_counts(std::span<const int64_t> t_ns, std::span<const uint8_t> p, int64_t t0_ns, int64_t bin_ns,
                 std::span<uint32_t> total, std::span<uint32_t> on);

// Contract C4 times: out[i] = 1000 (t_offset_us + t_us[i]) + t_ns_rem[i] (t_ns_rem empty: 0), the exact ns of an
// event file's /events/{t, t_ns_rem} and /t_offset (engine/writer.py).
void c4_times(std::span<const int64_t> t_us, std::span<const uint16_t> t_ns_rem, int64_t t_offset_us,
              std::span<int64_t> out);

// Polarity as 0/1 with 1 = ON: (p != 0) xor invert.
void polarity01(std::span<const uint8_t> p, bool invert, std::span<uint8_t> out);

// A region (ui.md §3.5 BSMK, §4.5): mask m holds over the half-open interval [t_lo_ns[m], t_hi_ns[m]) of one clock;
// the intervals are ordered and disjoint (t_lo[m] <= t_hi[m] <= t_lo[m + 1]), and time outside them is outside the
// region. Pixel i = y W + x of mask m is bit (i & 31) of bits[m mask_words(W, H) + (i >> 5)].
struct RegionMasks {
  uint16_t width = 0, height = 0;
  std::span<const int64_t> t_lo_ns, t_hi_ns;
  std::span<const uint32_t> bits;
};

constexpr size_t mask_words(uint32_t width, uint32_t height) { return (size_t{width} * height + 31) / 32; }

// The per-event test of a region (viz.stats.MaskRegion's `select`): out[i] = 1 when t_ns[i] lies in a mask's
// interval and (x[i], y[i]) is inside the image and set in that mask, else 0. The events need not be sorted (a sorted
// run reuses the interval of the previous event). std::invalid_argument for mismatched sizes, an empty image, or
// intervals that are not ordered and disjoint.
void region_bits(std::span<const uint16_t> x, std::span<const uint16_t> y, std::span<const int64_t> t_ns,
                 const RegionMasks& r, uint8_t* out);

}  // namespace balla::wire
