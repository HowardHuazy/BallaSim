// CRC-32 of the zlib/PNG/MCAP family (ISO-HDLC: reflected polynomial 0xEDB88320, initial and final XOR 0xFFFFFFFF),
// with zlib's crc32(crc, buf, len) chaining: crc32(crc32(0, a), b) == crc32(0, a + b). Our own tables, since the
// environment has no zlib.h (docs/design/phase3/outputs.md §2.1); MCAP's chunk, data-section and summary CRCs use it.
// Eight bytes per step through eight 256-entry tables ("slicing by 8", the same CRC as the byte-wise table that A0
// shipped, about 5x faster), then the byte-wise table for the tail. Header-only.
#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace balla::exporter {

static_assert(std::endian::native == std::endian::little, "the slicing CRC reads little-endian words");

namespace detail {
// kCrc32Tables[0] is the byte-wise table; kCrc32Tables[k][i] is the CRC state after byte i and k zero bytes.
inline constexpr std::array<std::array<uint32_t, 256>, 8> kCrc32Tables = [] {
  std::array<std::array<uint32_t, 256>, 8> t{};
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    t[0][i] = c;
  }
  for (size_t k = 1; k < 8; ++k) {
    for (uint32_t i = 0; i < 256; ++i) t[k][i] = (t[k - 1][i] >> 8) ^ t[0][t[k - 1][i] & 0xFFu];
  }
  return t;
}();
}  // namespace detail

inline uint32_t crc32(uint32_t crc, const void* data, size_t n) {
  const auto& t = detail::kCrc32Tables;
  const auto* p = static_cast<const uint8_t*>(data);
  uint32_t c = ~crc;
  for (; n >= 8; n -= 8, p += 8) {
    uint32_t lo = 0, hi = 0;
    std::memcpy(&lo, p, 4);
    std::memcpy(&hi, p + 4, 4);
    lo ^= c;
    c = t[7][lo & 0xFFu] ^ t[6][(lo >> 8) & 0xFFu] ^ t[5][(lo >> 16) & 0xFFu] ^ t[4][lo >> 24] ^ t[3][hi & 0xFFu] ^
        t[2][(hi >> 8) & 0xFFu] ^ t[1][(hi >> 16) & 0xFFu] ^ t[0][hi >> 24];
  }
  for (; n > 0; --n, ++p) c = t[0][(c ^ *p) & 0xFFu] ^ (c >> 8);
  return ~c;
}

}  // namespace balla::exporter
