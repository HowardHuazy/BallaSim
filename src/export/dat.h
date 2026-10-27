// Prophesee DAT CD records (docs/design/phase3/outputs.md §4.8; docs.prophesee.ai file_formats dat, the layout of
// OpenEB's DAT writer as the R-FMT prototype scripts/evt.py writes it): after the '% ' header lines, the event type
// 0x0C (CD) and size 0x08, then per event u32 timestamp (us) and u32 x | y << 14 | p << 28, little-endian. The
// timestamp is absolute u32 us with no wrap handling in OpenEB's reader, so the writer refuses u >= 2^32 (71.6 min
// after the origin). Header-only.
#pragma once

#include <cstdint>

namespace balla::exporter {

inline constexpr uint8_t kDatTypeCd = 0x0C;
inline constexpr uint8_t kDatEventSize = 8;
inline constexpr int32_t kDatMaxDim = 1 << 14;  // x and y have 14 bits
inline constexpr int64_t kDatMaxU = (int64_t{1} << 32) - 1;

// Appends one record's two words to `out` (anything with push_back of uint32_t).
template <class Out>
void dat_record(int64_t u, uint32_t x, uint32_t y, uint32_t p, Out& out) {
  out.push_back(static_cast<uint32_t>(u));
  out.push_back(x | (y << 14) | (p << 28));
}

}  // namespace balla::exporter
