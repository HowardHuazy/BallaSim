// The event wire (wire.h): LOD order, `ev` packing, gathers, rate bins and region bits.
#include "core/wire/wire.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace balla::wire {

namespace {

void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(what);
}

// Occurrence indices of a run of equal t_ns: k[i] = #{j < i in the run : word[j] == word[i]}. Short runs scan;
// long ones use an open-addressing table whose slots are invalidated by a generation number, never cleared.
class DuplicateIndex {
 public:
  static constexpr size_t kScanMax = 32;

  // words[0, n) -> k[0, n); returns the number of k > 0
  uint32_t index(const uint64_t* words, size_t n, uint32_t* k) {
    uint32_t dups = 0;
    if (n <= kScanMax) {
      for (size_t i = 0; i < n; ++i) {
        uint32_t c = 0;
        for (size_t j = 0; j < i; ++j) c += words[j] == words[i];
        k[i] = c;
        dups += c != 0;
      }
      return dups;
    }
    size_t size = 64;
    while (size < 2 * n) size <<= 1;
    if (slots_.size() < size) {
      slots_.assign(size, Slot{});
      generation_ = 0;
    }
    if (++generation_ == 0) {  // wrapped: every stale slot would look live
      for (Slot& s : slots_) s.generation = 0;
      generation_ = 1;
    }
    const size_t mask = size - 1;
    for (size_t i = 0; i < n; ++i) {
      size_t h = static_cast<size_t>(mix64(words[i])) & mask;
      while (slots_[h].generation == generation_ && slots_[h].word != words[i]) h = (h + 1) & mask;
      Slot& s = slots_[h];
      if (s.generation != generation_) {
        s = Slot{words[i], 1, generation_};
        k[i] = 0;
      } else {
        k[i] = s.count++;
        ++dups;
      }
    }
    return dups;
  }

 private:
  struct Slot {
    uint64_t word = 0;
    uint32_t count = 0;
    uint32_t generation = 0;
  };
  std::vector<Slot> slots_;
  uint32_t generation_ = 0;
};

template <size_t B>
void gather_fixed(std::span<const uint32_t> perm, const unsigned char* src, unsigned char* dst, size_t stride) {
  for (size_t i = 0; i < perm.size(); ++i) std::memcpy(dst + i * stride, src + size_t{perm[i]} * B, B);
}

}  // namespace

namespace {

// Pass 1 of the LOD order: each event's level (k counted per run of equal t_ns), the count per level and the
// number of duplicates.
uint32_t compute_levels(std::span<const int64_t> t_ns, std::span<const uint16_t> x, std::span<const uint16_t> y,
                        std::span<const uint8_t> p, uint8_t* level, std::array<uint32_t, kLevels + 1>& count) {
  const size_t n = t_ns.size();
  require(x.size() == n && y.size() == n && p.size() == n, "lod_order: t_ns, x, y and p need equal lengths");
  require(n <= std::numeric_limits<uint32_t>::max(), "lod_order: a block holds at most 2^32 - 1 events");
  DuplicateIndex dup;
  std::vector<uint64_t> words;
  std::vector<uint32_t> ks;
  uint32_t duplicates = 0;
  size_t i = 0;
  while (i < n) {
    size_t j = i + 1;
    while (j < n && t_ns[j] == t_ns[i]) ++j;
    require(j == n || t_ns[j] > t_ns[i], "lod_order: events must be sorted by time");
    const uint64_t th = mix64(static_cast<uint64_t>(t_ns[i]));
    if (j - i == 1) {
      const uint32_t l = lod_level(mix64(th ^ pixel_word(x[i], y[i], p[i])));
      level[i] = static_cast<uint8_t>(l);
      ++count[l];
    } else {
      const size_t run = j - i;
      words.resize(run);
      ks.resize(run);
      for (size_t r = 0; r < run; ++r) words[r] = pixel_word(x[i + r], y[i + r], p[i + r]);
      duplicates += dup.index(words.data(), run, ks.data());
      for (size_t r = 0; r < run; ++r) {
        const uint32_t l = lod_level(mix64(th ^ words[r] ^ (uint64_t{ks[r]} << 33)));
        level[i + r] = static_cast<uint8_t>(l);
        ++count[l];
      }
    }
    i = j;
  }
  return duplicates;
}

// n_ge from the counts, and the first LOD position of each level: level l starts after every event of a higher
// level, at n_ge[l + 1].
void order_tables(const std::array<uint32_t, kLevels + 1>& count, BlockOrder& out,
                  std::array<uint32_t, kLevels + 1>& next) {
  std::array<uint32_t, kLevels + 2> ge{};
  for (int l = kLevels; l >= 0; --l) ge[l] = ge[l + 1] + count[l];
  for (int l = 0; l <= kLevels; ++l) {
    out.n_ge[l] = ge[l];
    next[l] = ge[l + 1];
  }
}

// One `ev` word pair; bad_* accumulate the refusals, reported after the loop.
inline void pack_one(int64_t t, uint16_t x, uint16_t y, uint8_t p, const uint8_t* label, size_t s, int64_t t_base_ns,
                     uint32_t* w, uint32_t& bad_coord, uint32_t& bad_label, uint64_t& bad_time) {
  bad_coord |= static_cast<uint32_t>(x > kMaxCoord) | static_cast<uint32_t>(y > kMaxCoord);
  uint32_t tag = 0;
  if (label != nullptr) {
    bad_label |= static_cast<uint32_t>(label[s] > kMaxLabel);
    tag = 1u + label[s];
  }
  const uint64_t dt = static_cast<uint64_t>(t) - static_cast<uint64_t>(t_base_ns);
  bad_time |= dt >> 32;
  w[0] = (uint32_t{x} & 0x3FFFu) | ((uint32_t{y} & 0x3FFFu) << 14) | ((tag & 7u) << 28) | (uint32_t{p != 0} << 31);
  w[1] = static_cast<uint32_t>(dt);
}

void report(uint32_t bad_coord, uint32_t bad_label, uint64_t bad_time, const char* who) {
  const std::string w(who);
  require(!bad_coord, (w + ": x and y must be <= 16383 (BS-UI-TOO-WIDE)").c_str());
  require(!bad_label, (w + ": labels must be S1 classes 0..4").c_str());
  require(!bad_time, (w + ": t - t_base_ns must lie in [0, 2^32)").c_str());
}

}  // namespace

BlockOrder lod_order(std::span<const int64_t> t_ns, std::span<const uint16_t> x, std::span<const uint16_t> y,
                     std::span<const uint8_t> p) {
  const size_t n = t_ns.size();
  std::vector<uint8_t> level(n);
  std::array<uint32_t, kLevels + 1> count{}, next{};
  BlockOrder out;
  out.duplicates = compute_levels(t_ns, x, y, p, level.data(), count);
  order_tables(count, out, next);
  out.perm.resize(n);
  for (size_t e = 0; e < n; ++e) out.perm[next[level[e]]++] = static_cast<uint32_t>(e);
  return out;
}

BlockOrder pack_block(std::span<const int64_t> t_ns, std::span<const uint16_t> x, std::span<const uint16_t> y,
                      std::span<const uint8_t> p, const uint8_t* label, int64_t t_base_ns, uint32_t* out) {
  const size_t n = t_ns.size();
  std::vector<uint8_t> level(n);
  std::array<uint32_t, kLevels + 1> count{}, next{};
  BlockOrder order;
  order.duplicates = compute_levels(t_ns, x, y, p, level.data(), count);
  order_tables(count, order, next);
  uint32_t bad_coord = 0, bad_label = 0;
  uint64_t bad_time = 0;
  for (size_t e = 0; e < n; ++e) {
    pack_one(t_ns[e], x[e], y[e], p[e], label, e, t_base_ns, out + 2 * size_t{next[level[e]]++}, bad_coord, bad_label,
             bad_time);
  }
  report(bad_coord, bad_label, bad_time, "pack_block");
  return order;
}

void pack_ev(std::span<const int64_t> t_ns, std::span<const uint16_t> x, std::span<const uint16_t> y,
             std::span<const uint8_t> p, const uint8_t* label, std::span<const uint32_t> perm, uint32_t i0,
             uint32_t i1, int64_t t_base_ns, uint32_t* out) {
  const size_t n = t_ns.size();
  require(x.size() == n && y.size() == n && p.size() == n, "pack_ev: t_ns, x, y and p need equal lengths");
  const size_t order_n = perm.empty() ? n : perm.size();
  require(i0 <= i1 && i1 <= order_n, "pack_ev: need i0 <= i1 <= the number of ordered events");
  if (!perm.empty()) {
    for (uint32_t idx = i0; idx < i1; ++idx) require(perm[idx] < n, "pack_ev: a perm entry lies outside the events");
  }
  uint32_t bad_coord = 0, bad_label = 0;
  uint64_t bad_time = 0;
  for (uint32_t idx = i0; idx < i1; ++idx) {
    const size_t s = perm.empty() ? idx : perm[idx];
    pack_one(t_ns[s], x[s], y[s], p[s], label, s, t_base_ns, out + 2 * size_t{idx - i0}, bad_coord, bad_label,
             bad_time);
  }
  report(bad_coord, bad_label, bad_time, "pack_ev");
}

void gather(std::span<const uint32_t> perm, const void* src, size_t elem_bytes, void* dst, size_t dst_stride_bytes) {
  require(elem_bytes > 0, "gather: elem_bytes must be positive");
  const size_t stride = dst_stride_bytes == 0 ? elem_bytes : dst_stride_bytes;
  require(stride >= elem_bytes, "gather: the destination stride is shorter than an element");
  const auto* s = static_cast<const unsigned char*>(src);
  auto* d = static_cast<unsigned char*>(dst);
  switch (elem_bytes) {
    case 1: gather_fixed<1>(perm, s, d, stride); break;
    case 2: gather_fixed<2>(perm, s, d, stride); break;
    case 4: gather_fixed<4>(perm, s, d, stride); break;
    case 8: gather_fixed<8>(perm, s, d, stride); break;
    default:
      for (size_t i = 0; i < perm.size(); ++i) std::memcpy(d + i * stride, s + size_t{perm[i]} * elem_bytes, elem_bytes);
  }
}

void rate_counts(std::span<const int64_t> t_ns, std::span<const uint8_t> p, int64_t t0_ns, int64_t bin_ns,
                 std::span<uint32_t> total, std::span<uint32_t> on) {
  require(bin_ns > 0, "rate_counts: bin_ns must be positive");
  require(on.empty() || on.size() == total.size(), "rate_counts: on needs one entry per bin, or none");
  require(on.empty() || p.size() == t_ns.size(), "rate_counts: p needs one entry per event");
  const uint64_t n_bins = total.size(), ubin = static_cast<uint64_t>(bin_ns);
  // The bin [d_lo, d_hi) of offsets d = t - t0 (unsigned, exact for any int64 pair) of the previous event: sorted
  // input divides only where the bin changes. `beyond`: that bin lies past the last one.
  uint64_t b = 0, d_lo = 1, d_hi = 0;
  bool beyond = false;
  for (size_t i = 0; i < t_ns.size(); ++i) {
    if (t_ns[i] < t0_ns) continue;
    const uint64_t d = static_cast<uint64_t>(t_ns[i]) - static_cast<uint64_t>(t0_ns);
    if (d < d_lo || d >= d_hi) {
      b = d / ubin;
      d_lo = b * ubin;
      d_hi = d_lo <= std::numeric_limits<uint64_t>::max() - ubin ? d_lo + ubin : std::numeric_limits<uint64_t>::max();
      beyond = b >= n_bins;
    }
    if (beyond) continue;
    ++total[b];
    if (!on.empty()) on[b] += p[i] != 0;
  }
}

void c4_times(std::span<const int64_t> t_us, std::span<const uint16_t> t_ns_rem, int64_t t_offset_us,
              std::span<int64_t> out) {
  require(out.size() == t_us.size() && (t_ns_rem.empty() || t_ns_rem.size() == t_us.size()),
          "c4_times: t_us, t_ns_rem and the output need equal lengths");
  for (size_t i = 0; i < t_us.size(); ++i) {
    out[i] = 1000 * (t_offset_us + t_us[i]) + (t_ns_rem.empty() ? 0 : int64_t{t_ns_rem[i]});
  }
}

void polarity01(std::span<const uint8_t> p, bool invert, std::span<uint8_t> out) {
  require(out.size() == p.size(), "polarity01: p and the output need equal lengths");
  const uint8_t flip = invert ? 1 : 0;
  for (size_t i = 0; i < p.size(); ++i) out[i] = static_cast<uint8_t>((p[i] != 0) ^ flip);
}

void region_bits(std::span<const uint16_t> x, std::span<const uint16_t> y, std::span<const int64_t> t_ns,
                 const RegionMasks& r, uint8_t* out) {
  const size_t n = t_ns.size(), masks = r.t_lo_ns.size();
  require(x.size() == n && y.size() == n, "region_bits: x, y and t_ns need equal lengths");
  require(r.t_hi_ns.size() == masks, "region_bits: t_lo_ns and t_hi_ns need one entry per mask");
  require(r.width > 0 && r.height > 0, "region_bits: the image is empty");
  const size_t words = mask_words(r.width, r.height);
  require(r.bits.size() == masks * words, "region_bits: bits needs ceil(W H / 32) words per mask");
  for (size_t m = 0; m < masks; ++m) {
    require(r.t_lo_ns[m] <= r.t_hi_ns[m] && (m + 1 == masks || r.t_hi_ns[m] <= r.t_lo_ns[m + 1]),
            "region_bits: the intervals must be ordered and disjoint");
  }
  const int64_t* lo = r.t_lo_ns.data();
  const uint32_t w = r.width, h = r.height;
  constexpr size_t kNone = std::numeric_limits<size_t>::max();
  size_t m = kNone;                              // the last mask with t_lo <= t, kNone before the first
  for (size_t i = 0; i < n; ++i) {
    const int64_t t = t_ns[i];
    if (m == kNone ? (masks && t >= lo[0]) : (t < lo[m] || (m + 1 < masks && t >= lo[m + 1]))) {
      const int64_t* at = std::upper_bound(lo, lo + masks, t);
      m = at == lo ? kNone : static_cast<size_t>(at - lo) - 1;
    }
    const uint32_t xi = x[i], yi = y[i];
    bool in = false;
    if (m != kNone && t < r.t_hi_ns[m] && xi < w && yi < h) {
      const size_t pixel = size_t{yi} * w + xi;
      in = (r.bits[m * words + (pixel >> 5)] >> (pixel & 31)) & 1u;
    }
    out[i] = in ? 1 : 0;
  }
}

}  // namespace balla::wire
