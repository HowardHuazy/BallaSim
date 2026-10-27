#include "export/compress.h"

#include <bzlib.h>
#include <lz4frame.h>
#include <zstd.h>

#include <algorithm>
#include <climits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "export/errors.h"

namespace balla::exporter {

namespace {

constexpr int kBz2Block = 9, kBz2WorkFactor = 30;

LZ4F_preferences_t lz4_prefs(Codec c, Lz4Frame f) {
  LZ4F_preferences_t p{};
  p.frameInfo.frameType = LZ4F_frame;
  p.frameInfo.blockChecksumFlag = LZ4F_noBlockChecksum;
  p.frameInfo.contentSize = 0;
  p.frameInfo.dictID = 0;
  if (f == Lz4Frame::kRoslz4) {
    p.frameInfo.blockSizeID = LZ4F_max1MB;
    p.frameInfo.blockMode = LZ4F_blockIndependent;
    p.frameInfo.contentChecksumFlag = LZ4F_contentChecksumEnabled;
  } else {
    p.frameInfo.blockSizeID = LZ4F_max64KB;
    p.frameInfo.blockMode = LZ4F_blockLinked;
    p.frameInfo.contentChecksumFlag = LZ4F_noContentChecksum;
  }
  p.compressionLevel = codec_level(c);
  p.autoFlush = 1;
  return p;
}

void check_lz4(size_t r, const char* what) {
  if (LZ4F_isError(r)) throw std::runtime_error(std::string("lz4: ") + what + ": " + LZ4F_getErrorName(r));
}

struct Lz4Ctx {
  LZ4F_cctx* c = nullptr;
  Lz4Ctx() { check_lz4(LZ4F_createCompressionContext(&c, LZ4F_VERSION), "context"); }
  ~Lz4Ctx() { LZ4F_freeCompressionContext(c); }
};

struct Lz4DCtx {
  LZ4F_dctx* c = nullptr;
  Lz4DCtx() { check_lz4(LZ4F_createDecompressionContext(&c, LZ4F_VERSION), "context"); }
  ~Lz4DCtx() { LZ4F_freeDecompressionContext(c); }
};

void lz4_compress(Codec codec, Lz4Frame frame, const void* data, size_t n, std::vector<char>& out) {
  const LZ4F_preferences_t prefs = lz4_prefs(codec, frame);
  Lz4Ctx ctx;  // fresh per frame: LZ4's fast reset of a used context may pick other matches
  out.resize(LZ4F_HEADER_SIZE_MAX + LZ4F_compressBound(n, &prefs));
  size_t pos = LZ4F_compressBegin(ctx.c, out.data(), out.size(), &prefs);
  check_lz4(pos, "begin");
  const size_t body = LZ4F_compressUpdate(ctx.c, out.data() + pos, out.size() - pos, data, n, nullptr);
  check_lz4(body, "update");
  pos += body;
  const size_t end = LZ4F_compressEnd(ctx.c, out.data() + pos, out.size() - pos, nullptr);
  check_lz4(end, "end");
  out.resize(pos + end);
}

unsigned int bz2_length(size_t n) {
  if (n > UINT_MAX) throw Refused("bz2: a chunk of " + std::to_string(n) + " bytes exceeds 4 GiB");
  return static_cast<unsigned int>(n);
}

[[noreturn]] void corrupt(Codec c, const std::string& why) {
  throw Refused(std::string(codec_name(c)) + " data: " + why);
}

std::string sizes(size_t got, size_t raw_size) {
  return std::to_string(got) + " bytes, expected " + std::to_string(raw_size);
}

// A decoder's output. It grows as the data decodes (doubling, from a few times the input) and never past raw_size, so
// a wrong raw_size (a corrupt container header) costs no more memory than the data itself produces before the refusal.
class Output {
 public:
  Output(size_t raw_size, size_t first) : raw_size_(raw_size) { buf_.resize(std::min(raw_size, first)); }
  static size_t guess(size_t n) { return 4 * n + kMinGrow; }  // the first size for an input of n bytes

  // The free bytes at next(), growing the buffer when it is full; 0 once raw_size bytes are out.
  size_t room() {
    if (used_ == buf_.size() && used_ < raw_size_) {
      buf_.resize(std::min(raw_size_, std::max(2 * buf_.size(), kMinGrow)));
    }
    return buf_.size() - used_;
  }
  char* next() { return buf_.data() + used_; }
  void advance(size_t k) { used_ += k; }
  size_t used() const { return used_; }
  std::vector<char> take() {
    buf_.resize(used_);
    return std::move(buf_);
  }

 private:
  static constexpr size_t kMinGrow = size_t{64} << 10;
  size_t raw_size_, used_ = 0;
  std::vector<char> buf_;
};

// exact: raw_size is the size; otherwise the most bytes the frame may give (decompress_upto).
std::vector<char> lz4_decompress(Codec codec, const char* src, size_t n, size_t raw_size, bool exact) {
  Lz4DCtx ctx;
  Output out(raw_size, Output::guess(n));
  size_t in = 0, hint = 1;
  while (hint != 0) {
    if (in == n) corrupt(codec, "a truncated frame, or more than " + std::to_string(raw_size) + " bytes");
    const size_t room = out.room();
    size_t dn = room, sn = n - in;
    char scratch = 0;  // a valid pointer when the output is full: a frame running past raw_size fails below
    hint = LZ4F_decompress(ctx.c, room > 0 ? out.next() : &scratch, &dn, src + in, &sn, nullptr);
    if (LZ4F_isError(hint)) corrupt(codec, LZ4F_getErrorName(hint));
    out.advance(dn);
    in += sn;
    if (dn == 0 && sn == 0 && hint != 0) corrupt(codec, "more than " + std::to_string(raw_size) + " bytes");
  }
  if (in != n) corrupt(codec, std::to_string(n - in) + " bytes after the frame");
  if (exact && out.used() != raw_size) corrupt(codec, sizes(out.used(), raw_size));
  return out.take();
}

struct ZstdDCtx {
  ZSTD_DCtx* c = ZSTD_createDCtx();
  ZstdDCtx() {
    if (c == nullptr) throw std::bad_alloc();
  }
  ~ZstdDCtx() { ZSTD_freeDCtx(c); }
};

std::vector<char> zstd_decompress(Codec codec, const void* data, size_t n, size_t raw_size, bool exact) {
  // The frame header's content size (our frames always carry it) is checked before any allocation.
  const unsigned long long declared = ZSTD_getFrameContentSize(data, n);
  if (declared == ZSTD_CONTENTSIZE_ERROR) corrupt(codec, "no zstd frame header");
  if (declared != ZSTD_CONTENTSIZE_UNKNOWN && (exact ? declared != raw_size : declared > raw_size)) {
    corrupt(codec, exact ? "the frame declares " + sizes(declared, raw_size)
                         : "the frame declares " + std::to_string(declared) + " bytes, more than " +
                               std::to_string(raw_size));
  }
  ZstdDCtx ctx;
  // A header that agrees with raw_size may still lie, so the output grows as for lz4 and bz2 rather than being sized
  // from it. When raw_size fits the first buffer (ratios up to about 4), zstd decodes in one pass, with no window
  // buffer; beyond that its streaming decoder fills the growing buffer, and a lying size fails zstd's own check.
  Output out(raw_size, Output::guess(n));
  ZSTD_inBuffer in{data, n, 0};
  for (;;) {
    const size_t room = out.room(), in_before = in.pos;
    char scratch = 0;
    ZSTD_outBuffer o{room > 0 ? out.next() : &scratch, room, 0};
    const size_t r = ZSTD_decompressStream(ctx.c, &o, &in);
    if (ZSTD_isError(r)) corrupt(codec, ZSTD_getErrorName(r));
    out.advance(o.pos);
    if (r == 0) break;  // the frame is decoded and flushed
    if (o.pos == 0 && in.pos == in_before) {
      corrupt(codec, room == 0 ? "more than " + std::to_string(raw_size) + " bytes" : "a truncated frame");
    }
  }
  if (in.pos != n) corrupt(codec, std::to_string(n - in.pos) + " bytes after the frame");
  if (exact && out.used() != raw_size) corrupt(codec, sizes(out.used(), raw_size));
  return out.take();
}

struct Bz2DStream {
  bz_stream s{};
  Bz2DStream() {
    const int r = BZ2_bzDecompressInit(&s, 0, 0);
    if (r == BZ_MEM_ERROR) throw std::bad_alloc();
    if (r != BZ_OK) throw std::runtime_error("bz2: BZ2_bzDecompressInit returned " + std::to_string(r));
  }
  ~Bz2DStream() { BZ2_bzDecompressEnd(&s); }
};

// One bzip2 stream with nothing after it (BZ2_bzBuffToBuffDecompress would ignore trailing bytes and further streams).
// The streaming API takes the input and output in pieces of at most 4 GiB.
std::vector<char> bz2_decompress(Codec codec, const char* src, size_t n, size_t raw_size, bool exact) {
  Bz2DStream d;
  Output out(raw_size, Output::guess(n));
  size_t in = 0;
  for (;;) {
    const size_t room = out.room();
    char scratch_in = 0, scratch_out = 0;  // libbz2 needs avail_out >= 1: a byte in scratch_out is one past raw_size
    const unsigned int give_in = static_cast<unsigned int>(std::min<size_t>(n - in, UINT_MAX));
    const unsigned int give_out = room > 0 ? static_cast<unsigned int>(std::min<size_t>(room, UINT_MAX)) : 1;
    d.s.next_in = n > in ? const_cast<char*>(src + in) : &scratch_in;
    d.s.avail_in = give_in;
    d.s.next_out = room > 0 ? out.next() : &scratch_out;
    d.s.avail_out = give_out;
    const int r = BZ2_bzDecompress(&d.s);
    if (r == BZ_MEM_ERROR) throw std::bad_alloc();
    const size_t took = give_in - d.s.avail_in, made = give_out - d.s.avail_out;
    in += took;
    if (room == 0 && made > 0) corrupt(codec, "more than " + std::to_string(raw_size) + " bytes");
    out.advance(made);
    if (r == BZ_STREAM_END) break;
    if (r == BZ_DATA_ERROR_MAGIC) corrupt(codec, "no bzip2 stream header");
    if (r != BZ_OK) corrupt(codec, "BZ2_bzDecompress returned " + std::to_string(r));
    if (took == 0 && made == 0) corrupt(codec, "a truncated stream");
  }
  if (in != n) corrupt(codec, std::to_string(n - in) + " bytes after the stream");
  if (exact && out.used() != raw_size) corrupt(codec, sizes(out.used(), raw_size));
  return out.take();
}

}  // namespace

Codec parse_codec(std::string_view name) {
  for (const Codec c : {Codec::kNone, Codec::kLz4, Codec::kLz4High, Codec::kZstd, Codec::kZstdHigh, Codec::kBz2}) {
    if (codec_name(c) == name) return c;
  }
  throw Refused("unknown compression '" + std::string(name) + "': none, lz4, lz4-high, zstd, zstd-high or bz2");
}

std::string_view codec_name(Codec c) {
  switch (c) {
    case Codec::kNone: return "none";
    case Codec::kLz4: return "lz4";
    case Codec::kLz4High: return "lz4-high";
    case Codec::kZstd: return "zstd";
    case Codec::kZstdHigh: return "zstd-high";
    case Codec::kBz2: return "bz2";
  }
  throw std::logic_error("codec_name: invalid Codec");
}

int codec_level(Codec c) {
  switch (c) {
    case Codec::kNone: return 0;
    case Codec::kLz4: return 0;
    case Codec::kLz4High: return 9;
    case Codec::kZstd: return 3;
    case Codec::kZstdHigh: return ZSTD_maxCLevel();
    case Codec::kBz2: return kBz2Block;
  }
  throw std::logic_error("codec_level: invalid Codec");
}

Compressor::Compressor(Codec codec, Lz4Frame frame) : codec_(codec), frame_(frame) {
  (void)codec_name(codec);  // validates the enum
  if (codec == Codec::kZstd || codec == Codec::kZstdHigh) {
    zstd_ = ZSTD_createCCtx();
    if (zstd_ == nullptr) throw std::bad_alloc();
  }
}

Compressor::~Compressor() { ZSTD_freeCCtx(static_cast<ZSTD_CCtx*>(zstd_)); }

void Compressor::compress(const void* data, size_t n, std::vector<char>& out) {
  switch (codec_) {
    case Codec::kNone: {
      const auto* p = static_cast<const char*>(data);
      out.assign(p, p + n);
      return;
    }
    case Codec::kLz4:
    case Codec::kLz4High: lz4_compress(codec_, frame_, data, n, out); return;
    case Codec::kZstd:
    case Codec::kZstdHigh: {
      out.resize(ZSTD_compressBound(n));
      const size_t r = ZSTD_compressCCtx(static_cast<ZSTD_CCtx*>(zstd_), out.data(), out.size(), data, n,
                                         codec_level(codec_));
      if (ZSTD_isError(r)) throw std::runtime_error(std::string("zstd: ") + ZSTD_getErrorName(r));
      out.resize(r);
      return;
    }
    case Codec::kBz2: {
      const unsigned int len = bz2_length(n);
      unsigned int cap = bz2_length(n + n / 100 + 600);  // bzip2's documented bound: 1 % plus 600 bytes
      out.resize(cap);
      char empty = 0;  // libbz2 refuses a null source, even of length 0
      char* src = n > 0 ? const_cast<char*>(static_cast<const char*>(data)) : &empty;
      const int r = BZ2_bzBuffToBuffCompress(out.data(), &cap, src, len, kBz2Block, 0, kBz2WorkFactor);
      if (r != BZ_OK) throw std::runtime_error("bz2: BZ2_bzBuffToBuffCompress returned " + std::to_string(r));
      out.resize(cap);
      return;
    }
  }
}

std::vector<char> compress(Codec codec, const void* data, size_t n, Lz4Frame frame) {
  std::vector<char> out;
  Compressor(codec, frame).compress(data, n, out);
  return out;
}

namespace {

std::vector<char> decode(Codec codec, const void* data, size_t n, size_t raw_size, bool exact) {
  const auto* src = static_cast<const char*>(data);
  switch (codec) {
    case Codec::kNone:
      if (exact ? n != raw_size : n > raw_size) {
        corrupt(codec,
                exact ? sizes(n, raw_size) : std::to_string(n) + " bytes, more than " + std::to_string(raw_size));
      }
      return std::vector<char>(src, src + n);
    case Codec::kLz4:
    case Codec::kLz4High: return lz4_decompress(codec, src, n, raw_size, exact);
    case Codec::kZstd:
    case Codec::kZstdHigh: return zstd_decompress(codec, data, n, raw_size, exact);
    case Codec::kBz2: return bz2_decompress(codec, src, n, raw_size, exact);
  }
  throw std::logic_error("decompress: invalid Codec");
}

}  // namespace

std::vector<char> decompress(Codec codec, const void* data, size_t n, size_t raw_size) {
  return decode(codec, data, n, raw_size, true);
}

std::vector<char> decompress_upto(Codec codec, const void* data, size_t n, size_t max_size) {
  return decode(codec, data, n, max_size, false);
}

}  // namespace balla::exporter
