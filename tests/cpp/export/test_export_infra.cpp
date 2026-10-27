// The exporters' infrastructure (docs/design/phase3/outputs.md §8.2): F9 known answers (CRC-32, SHA-256 incremental
// and one-shot, codec round trips and frame headers) and F7 for the compression pool and the sink (bytes independent
// of the thread count and of the run).
#include <doctest/doctest.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "config/sha256.h"
#include "export/chunk_pool.h"
#include "export/compress.h"
#include "export/crc32.h"
#include "export/errors.h"
#include "export/sink.h"

using namespace balla;
using exporter::Codec;
using exporter::Lz4Frame;

namespace {

const Codec kAllCodecs[] = {Codec::kNone, Codec::kLz4, Codec::kLz4High, Codec::kZstd, Codec::kZstdHigh, Codec::kBz2};

// Compressible bytes: a sorted-timestamp-like ramp with noise, as event packets look (deterministic).
std::vector<char> sample(size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<char> v(n);
  uint32_t t = 0;
  for (size_t i = 0; i < n; i += 8) {
    t += static_cast<uint32_t>(rng() % 7);
    const uint32_t xy = static_cast<uint32_t>(rng() % 4096);
    for (size_t k = 0; k < 8 && i + k < n; ++k) {
      v[i + k] = static_cast<char>(k < 4 ? (t >> (8 * k)) & 0xFF : (xy >> (8 * (k - 4))) & 0xFF);
    }
  }
  return v;
}

std::string temp_path(const std::string& name) {
  const auto dir = std::filesystem::temp_directory_path() / ("balla_export_tests_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  return (dir / name).string();
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

uint8_t byte(const std::vector<char>& v, size_t i) { return static_cast<uint8_t>(v.at(i)); }

}  // namespace

TEST_CASE("F9 CRC-32: the check value, the empty message and chaining") {
  const std::string m = "123456789";
  CHECK(exporter::crc32(0, m.data(), m.size()) == 0xCBF43926u);
  CHECK(exporter::crc32(0, nullptr, 0) == 0u);
  const std::string q = "The quick brown fox jumps over the lazy dog";
  CHECK(exporter::crc32(0, q.data(), q.size()) == 0x414FA339u);
  for (size_t cut = 0; cut <= q.size(); ++cut) {
    CHECK(exporter::crc32(exporter::crc32(0, q.data(), cut), q.data() + cut, q.size() - cut) == 0x414FA339u);
  }
}

TEST_CASE("F9 SHA-256: FIPS 180-4 vectors through Sha256 and sha256_hex, every split point") {
  const std::string abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
  CHECK(config::sha256_hex("abc") == abc);
  config::Sha256 s;
  s.update("a", 1);
  s.update("bc", 2);
  CHECK(s.hex_final() == abc);
  CHECK_THROWS_AS(s.update("x", 1), std::logic_error);
  CHECK(config::sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  const std::string two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";  // 448 bits: two blocks
  const std::string two_d = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
  CHECK(config::sha256_hex(two) == two_d);
  for (size_t cut = 0; cut <= two.size(); ++cut) {
    config::Sha256 h;
    h.update(two.data(), cut);
    h.update(two.data() + cut, two.size() - cut);
    CHECK(h.hex_final() == two_d);
  }
  // One million 'a', fed in uneven pieces (FIPS 180-4 long message).
  const std::string chunk(997, 'a');
  config::Sha256 h;
  size_t left = 1000000;
  for (size_t i = 0; left > 0; ++i) {
    const size_t n = std::min<size_t>(left, 1 + (i * 131) % chunk.size());
    h.update(chunk.data(), n);
    left -= n;
  }
  CHECK(h.hex_final() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  // The padding boundaries against hashlib's answers (independent of this implementation): with 55 bytes (mod 64) the
  // 0x80 byte and the length fit one block; from 56 they need a second.
  std::string pattern(300, '\0');
  for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = static_cast<char>((i * 131 + 17) & 0xFF);
  const std::pair<size_t, std::string> edges[] = {
      {55, "59aaae80b8e7958f5757b4ac9274f1fd57ec0dc7aa8349319102316781b92006"},
      {56, "dca902d31487ffab357ce36cc5abc6947fbb69127ec04c468ba5687268c4bf7c"},
      {63, "b8e655e9e7ad413b96f0a371eb1d6db71dcca27f4eee35892517cbe4bfaa6f9a"},
      {64, "e5146be62accc56709594cb45c651b361df94f622cbb09b91ea3ca7a2060bd53"},
      {119, "3244509f8746f6fb511ca8a72a292a4d7d12e456cce283c03f169dbbcb0d512e"},
      {120, "c5236e50172da3b69c903dcd3b3bb8ce855182f79364d1b59ed36367aa25e468"}};
  for (const auto& [n, hex] : edges) {
    CAPTURE(n);
    CHECK(config::sha256_hex(std::string_view(pattern.data(), n)) == hex);
  }
  CHECK(config::sha256_hex(std::string(55, 'a')) == "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
  // Every length 0..300, byte-at-a-time against one-shot (the update() buffering), and all of them against hashlib
  // through one constant: hashlib.sha256("".join(hashlib.sha256(pattern[:n]).hexdigest() for n in range(301))).
  std::string all;
  for (size_t n = 0; n <= pattern.size(); ++n) {
    config::Sha256 b;
    for (size_t i = 0; i < n; ++i) b.update(&pattern[i], 1);
    const std::string one = config::sha256_hex(std::string_view(pattern.data(), n));
    REQUIRE(b.hex_final() == one);
    all += one;
  }
  CHECK(config::sha256_hex(all) == "84cbf5c50953feecf651ec99df2f8d25a74e737acc6b9e062db2ee2c9a61e68f");
}

TEST_CASE("F9 codecs: names, levels and round trips of every codec and frame kind") {
  for (const Codec c : kAllCodecs) CHECK(exporter::parse_codec(exporter::codec_name(c)) == c);
  CHECK_THROWS_AS(exporter::parse_codec("gzip"), exporter::Refused);
  CHECK(exporter::codec_level(Codec::kZstd) == 3);
  CHECK(exporter::codec_level(Codec::kZstdHigh) >= 19);
  CHECK(exporter::codec_level(Codec::kLz4High) == 9);
  const size_t big = size_t{1} << 21;  // several LZ4 blocks of either size
  for (const size_t n : {size_t{0}, size_t{1}, size_t{100}, size_t{65536}, big}) {
    const std::vector<char> raw = sample(n, n + 1);
    for (const Codec c : kAllCodecs) {
      if (c == Codec::kZstdHigh && n == big) continue;  // level 22: seconds per MB
      const bool lz4 = c == Codec::kLz4 || c == Codec::kLz4High;
      for (const Lz4Frame f : {Lz4Frame::kStandard, Lz4Frame::kRoslz4}) {
        if (!lz4 && f == Lz4Frame::kRoslz4) continue;  // the frame kind is LZ4's only
        CAPTURE(n);
        CAPTURE(exporter::codec_name(c));
        const std::vector<char> z = exporter::compress(c, raw.data(), raw.size(), f);
        CHECK(exporter::decompress(c, z.data(), z.size(), raw.size()) == raw);
        CHECK(exporter::compress(c, raw.data(), raw.size(), f) == z);  // deterministic
        if (n == big && c != Codec::kNone) CHECK(z.size() < raw.size() - raw.size() / 20);  // it compresses
      }
    }
  }
}

TEST_CASE("F9 frame headers: LZ4 standard and roslz4 (FLG 0x64, BD 0x60), zstd and bz2 magic") {
  const std::vector<char> raw = sample(1000, 7);
  for (const Codec c : {Codec::kLz4, Codec::kLz4High}) {
    const std::vector<char> std_frame = exporter::compress(c, raw.data(), raw.size(), Lz4Frame::kStandard);
    const std::vector<char> ros = exporter::compress(c, raw.data(), raw.size(), Lz4Frame::kRoslz4);
    for (const auto* z : {&std_frame, &ros}) {
      CHECK(byte(*z, 0) == 0x04);
      CHECK(byte(*z, 1) == 0x22);
      CHECK(byte(*z, 2) == 0x4D);
      CHECK(byte(*z, 3) == 0x18);
    }
    CHECK(byte(std_frame, 4) == 0x40);  // version 01, linked blocks, no checksums
    CHECK(byte(std_frame, 5) == 0x40);  // 64 KB blocks
    CHECK(byte(ros, 4) == 0x64);        // version 01, independent blocks, content checksum
    CHECK(byte(ros, 5) == 0x60);        // 1 MB blocks, kept for a 1000-byte input
  }
  const std::vector<char> zs = exporter::compress(Codec::kZstd, raw.data(), raw.size());
  CHECK((byte(zs, 0) == 0x28 && byte(zs, 1) == 0xB5 && byte(zs, 2) == 0x2F && byte(zs, 3) == 0xFD));
  const std::vector<char> bz = exporter::compress(Codec::kBz2, raw.data(), raw.size());
  CHECK(std::string(bz.data(), 4) == "BZh9");
}

TEST_CASE("F9 decompress refuses corrupt, truncated and mis-sized input") {
  const std::vector<char> raw = sample(70000, 11);
  for (const Codec c : {Codec::kLz4, Codec::kZstd, Codec::kBz2}) {
    CAPTURE(exporter::codec_name(c));
    std::vector<char> z = exporter::compress(c, raw.data(), raw.size());
    CHECK_THROWS_AS(exporter::decompress(c, z.data(), z.size(), raw.size() - 1), exporter::Refused);
    CHECK_THROWS_AS(exporter::decompress(c, z.data(), z.size(), raw.size() + 1), exporter::Refused);
    CHECK_THROWS_AS(exporter::decompress(c, z.data(), z.size() / 2, raw.size()), exporter::Refused);
    // Bytes after the frame or stream: junk, or a second frame (bz2's one-shot API would ignore both).
    std::vector<char> tail = z;
    tail.insert(tail.end(), {'j', 'u', 'n', 'k'});
    CHECK_THROWS_WITH_AS(exporter::decompress(c, tail.data(), tail.size(), raw.size()),
                         doctest::Contains("4 bytes after the"), exporter::Refused);
    std::vector<char> twice = z;
    twice.insert(twice.end(), z.begin(), z.end());
    CHECK_THROWS_AS(exporter::decompress(c, twice.data(), twice.size(), raw.size()), exporter::Refused);
    CHECK_THROWS_AS(exporter::decompress(c, twice.data(), twice.size(), 2 * raw.size()), exporter::Refused);
    // A corrupt size from a container header is refused without allocating it (the output grows as data decodes).
    CHECK_THROWS_AS(exporter::decompress(c, z.data(), z.size(), size_t{1} << 46), exporter::Refused);
    CHECK_THROWS_AS(exporter::decompress(c, z.data(), z.size(), 0), exporter::Refused);
    CHECK_THROWS_AS(exporter::decompress(c, nullptr, 0, raw.size()), exporter::Refused);
    z[1] = static_cast<char>(z[1] ^ 0x5A);  // the magic
    CHECK_THROWS_AS(exporter::decompress(c, z.data(), z.size(), raw.size()), exporter::Refused);
  }
  // A frame larger than the first output guess grows the output to the exact size.
  const std::vector<char> zeros(size_t{3} << 20, '\0');
  for (const Codec c : {Codec::kLz4, Codec::kZstd, Codec::kBz2}) {
    CAPTURE(exporter::codec_name(c));
    const std::vector<char> z = exporter::compress(c, zeros.data(), zeros.size());
    CHECK(z.size() * 4 + 65536 < zeros.size());
    CHECK(exporter::decompress(c, z.data(), z.size(), zeros.size()) == zeros);
  }
  // A zstd frame without its content size, as streaming writers make it: the same blocks behind a header without the
  // size field (zstd format, RFC 8878 §3.1.1.1). Its output grows as it decodes, and a wrong size is still refused.
  const std::vector<char> z = exporter::compress(Codec::kZstd, zeros.data(), zeros.size());
  const uint8_t fhd = byte(z, 4);
  REQUIRE((fhd & 0x23) == 0);  // not single-segment (3 MiB > the 2 MiB window of level 3): a window byte; no dictionary
  std::vector<char> unsized(z.begin(), z.begin() + 4);
  unsized.push_back(static_cast<char>(fhd & 0x04));  // keep the checksum flag; frame content size flag 0: no size
  unsized.push_back(z[5]);                           // the window descriptor
  unsized.insert(unsized.end(), z.begin() + 6 + (fhd >> 6 == 0 ? 0 : size_t{1} << (fhd >> 6)), z.end());
  CHECK(exporter::decompress(Codec::kZstd, unsized.data(), unsized.size(), zeros.size()) == zeros);
  CHECK_THROWS_WITH_AS(exporter::decompress(Codec::kZstd, unsized.data(), unsized.size(), zeros.size() - 1),
                       doctest::Contains("more than"), exporter::Refused);
  CHECK_THROWS_AS(exporter::decompress(Codec::kZstd, unsized.data(), unsized.size(), size_t{1} << 46),
                  exporter::Refused);
  CHECK_THROWS_AS(exporter::decompress(Codec::kZstd, unsized.data(), unsized.size() - 3, zeros.size()),
                  exporter::Refused);
  const std::vector<char> two = sample(10, 1);
  CHECK_THROWS_AS(exporter::decompress(Codec::kNone, two.data(), two.size(), 9), exporter::Refused);
}

TEST_CASE("F7 ChunkPool: emitted in submission order, bytes independent of the thread count") {
  // 40 chunks of uneven sizes, as rosbag1 chunks and AEDAT4 packets are.
  std::vector<std::vector<char>> chunks;
  for (size_t i = 0; i < 40; ++i) chunks.push_back(sample(1000 + (i * 7919) % 200000, i));
  for (const Codec c : {Codec::kNone, Codec::kLz4, Codec::kLz4High, Codec::kZstd, Codec::kBz2}) {  // zstd-high: slow
    CAPTURE(exporter::codec_name(c));
    std::vector<std::vector<char>> fresh;  // one compress() each, from fresh library state
    for (const auto& raw : chunks) fresh.push_back(exporter::compress(c, raw.data(), raw.size(), Lz4Frame::kRoslz4));
    for (const int threads : {1, 2, 4, 8}) {
      CAPTURE(threads);
      exporter::ChunkPool pool(c, threads, Lz4Frame::kRoslz4);
      std::vector<std::vector<char>> got;
      std::vector<size_t> order;
      for (size_t i = 0; i < chunks.size(); ++i) {
        pool.submit(chunks[i], [&got, &order, i](std::vector<char>&& z) {
          order.push_back(i);
          got.push_back(std::move(z));
        });
        if (i == 20) pool.drain();  // a drain in the middle changes nothing
      }
      pool.drain();
      REQUIRE(order.size() == chunks.size());
      for (size_t i = 0; i < order.size(); ++i) CHECK(order[i] == i);
      CHECK(got == fresh);
    }
  }
  CHECK_THROWS_AS(exporter::ChunkPool(Codec::kZstd, 0), exporter::Refused);
  CHECK_THROWS_AS(exporter::ChunkPool(Codec::kZstd, exporter::ChunkPool::kMaxThreads + 1), exporter::Refused);
}

TEST_CASE("F7 ChunkPool: an emit error surfaces on the calling thread; the pool stops cleanly") {
  // The error comes out of submit() or drain(), whichever reaches chunk 3's turn (the workers' timing decides), and
  // always after exactly the 3 chunks before it.
  exporter::ChunkPool pool(Codec::kZstd, 4);
  const std::vector<char> raw = sample(50000, 5);
  int emitted = 0, thrown = 0, emitted_at_throw = -1;
  const auto count = [&emitted](std::vector<char>&&) { ++emitted; };
  const auto fail = [](std::vector<char>&&) { throw std::runtime_error("disk full"); };
  const auto caught = [&](const std::runtime_error& e) {
    CHECK(std::string(e.what()) == "disk full");
    ++thrown;
    emitted_at_throw = emitted;
  };
  for (int i = 0; i < 7; ++i) {
    try {
      if (i == 3) {
        pool.submit(raw, fail);
      } else {
        pool.submit(raw, count);
      }
    } catch (const std::runtime_error& e) {
      caught(e);
    }
  }
  try {
    pool.drain();
  } catch (const std::runtime_error& e) {
    caught(e);
  }
  CHECK(thrown == 1);
  CHECK(emitted_at_throw == 3);
  pool.drain();  // the rest still come out, in order
  CHECK(emitted == 6);
}

TEST_CASE("ByteSink: buffered writes, patches on disk and in the buffer, the SHA-256 of the closed file") {
  const std::string path = temp_path("sink.bin");
  std::string expect;
  {
    exporter::ByteSink sink(path);
    const std::vector<char> big = sample(exporter::ByteSink::kBufferBytes + 12345, 9);  // bypasses the buffer
    const std::string head(16, '\0');
    sink.write(head.data(), head.size());
    sink.write(big.data(), big.size());
    for (int i = 0; i < 1000; ++i) sink.write("0123456789", 10);
    CHECK(sink.tell() == 16 + big.size() + 10000);
    // A patch across the flushed/buffered boundary is impossible here (big flushed everything); patch the head on
    // disk and the tail in the buffer.
    sink.patch(0, "HEADER--HEADER--", 16);
    sink.patch(sink.tell() - 4, "TAIL", 4);
    CHECK_THROWS_AS(sink.patch(sink.tell() - 2, "xyz", 3), std::out_of_range);
    expect = "HEADER--HEADER--" + std::string(big.data(), big.size());
    for (int i = 0; i < 1000; ++i) expect += "0123456789";
    expect.replace(expect.size() - 4, 4, "TAIL");
    const exporter::FileInfo info = sink.close();
    CHECK(info.path == path);
    CHECK(info.bytes == expect.size());
    CHECK(info.sha256 == config::sha256_hex(expect));
    CHECK_THROWS_AS(sink.write("x", 1), std::logic_error);
  }
  CHECK(read_file(path) == expect);
  {
    // A patch straddling the flushed part and the buffer.
    exporter::ByteSink sink(path);
    const std::vector<char> a = sample(exporter::ByteSink::kBufferBytes - 3, 4);
    sink.write(a.data(), a.size());
    sink.write("abcdef", 6);  // flushes a, buffers abcdef
    sink.patch(a.size() - 2, "WXYZ", 4);
    const exporter::FileInfo info = sink.close();
    std::string e(a.data(), a.size());
    e += "abcdef";
    e.replace(a.size() - 2, 4, "WXYZ");
    CHECK(read_file(path) == e);
    CHECK(info.sha256 == config::sha256_hex(e));
  }
  CHECK_THROWS_AS(exporter::ByteSink(temp_path("no/such/dir/x.bin")), std::system_error);
  std::filesystem::remove_all(std::filesystem::path(path).parent_path());
}

TEST_CASE("F7 sink + pool: two runs and 1, 4 or 8 threads give one SHA-256") {
  std::vector<std::vector<char>> chunks;
  for (size_t i = 0; i < 24; ++i) chunks.push_back(sample(20000 + 3000 * i, 100 + i));
  for (const Codec c : {Codec::kLz4, Codec::kZstd, Codec::kBz2}) {
    std::string first;
    for (const int threads : {1, 4, 8, 4}) {
      const std::string path = temp_path("pool.bin");
      exporter::ByteSink sink(path);
      exporter::ChunkPool pool(c, threads);
      for (const auto& raw : chunks) {
        pool.submit(raw, [&sink](std::vector<char>&& z) {
          const uint32_t n = static_cast<uint32_t>(z.size());
          sink.write(&n, sizeof n);
          sink.write(z.data(), z.size());
        });
      }
      pool.drain();
      const std::string sha = sink.close().sha256;
      if (first.empty()) first = sha;
      CHECK(sha == first);
    }
  }
  std::filesystem::remove_all(std::filesystem::path(temp_path("x")).parent_path());
}
