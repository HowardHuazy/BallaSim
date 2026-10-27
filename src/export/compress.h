// Block compression for the containers (docs/design/phase3/outputs.md §3.7, §4.8): one self-contained frame per call,
// over the environment's liblz4 (frame API), libzstd and libbz2. Every setting is fixed here, so the bytes depend only
// on (codec, LZ4 frame kind, input) and on the library versions pinned in tools/env/environment.yml.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace balla::exporter {

// none | lz4 (frame, level 0) | lz4-high (level 9) | zstd (level 3) | zstd-high (ZSTD_maxCLevel()) | bz2 (block 9, work
// factor 30). These are dv-processing's settings per codec (compression_support.hpp) and rosbag's bz2 settings.
enum class Codec : uint8_t { kNone, kLz4, kLz4High, kZstd, kZstdHigh, kBz2 };

// The LZ4 frame header an LZ4 codec writes (streaming API, so the declared block size is kept for small inputs):
//   kStandard  64 KB blocks, linked, no checksums: LZ4F's defaults, as dv-processing sets them for AEDAT4
//              (outputs.md §4.8); MCAP's lz4 chunks use the same frame; FLG 0x40, BD 0x40
//   kRoslz4    1 MB blocks, independent, content checksum: FLG 0x64, BD 0x60, the header ROS1's roslz4 accepts
//              (processHeader: FLG 0x64 only, block-max id 4-7)
enum class Lz4Frame : uint8_t { kStandard, kRoslz4 };

Codec parse_codec(std::string_view name);  // the names above; Refused otherwise, listing them
std::string_view codec_name(Codec);
int codec_level(Codec);                    // the library level (bz2: the block size in 100 kB); 0 for none

// Reusable per-thread compression state. LZ4 frames start from a fresh LZ4F context each call; the zstd context is
// reused, whose output zstd does not tie to earlier inputs (F7 checks the pool's bytes against fresh compress()).
class Compressor {
 public:
  explicit Compressor(Codec, Lz4Frame = Lz4Frame::kStandard);
  ~Compressor();
  Compressor(const Compressor&) = delete;
  Compressor& operator=(const Compressor&) = delete;

  void compress(const void* data, size_t n, std::vector<char>& out);  // out is replaced by one frame
  Codec codec() const { return codec_; }

 private:
  Codec codec_;
  Lz4Frame frame_;
  void* zstd_ = nullptr;  // ZSTD_CCtx
};

// One frame from fresh library state.
std::vector<char> compress(Codec, const void* data, size_t n, Lz4Frame = Lz4Frame::kStandard);
// The inverse: data must be exactly one frame (lz4, zstd) or stream (bz2) of raw_size bytes, the exact decompressed
// size the containers record. Corrupt or truncated input, bytes after the frame (a second frame included) and another
// size are Refused. The output grows as the data decodes, so a wrong raw_size costs no more memory than the data
// produces; a zstd frame whose header declares another size is refused before decoding.
std::vector<char> decompress(Codec, const void* data, size_t n, size_t raw_size);
// As decompress, for a container that does not record the decompressed size (AEDAT4's packets and table): exactly one
// frame or stream of any size up to max_size, whose output grows as it decodes; more is Refused.
std::vector<char> decompress_upto(Codec, const void* data, size_t n, size_t max_size);

}  // namespace balla::exporter
