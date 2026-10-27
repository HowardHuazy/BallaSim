// The MCAP writer (docs/design/phase3/outputs.md §8.2): F1 R-FMT's golden CDR Imu, its CDR EventPacket regenerated
// with the EVT 3.0 words without vector words, and its tiny.mcap (out/t3_vectors.json: 850 bytes, SHA-256
// 1e7313ae...) through the container; the file re-read record by record (the data-section, chunk and summary CRCs,
// the schemas and channels before their first message, the message and chunk indexes, the statistics, the summary
// offsets and the footer); F7 lz4 and zstd chunks hold none's records, the bytes the same at 1-8 threads; F8 one
// append against many; the CDR layouts; the refusals of §6; the packet capacity; the slicing CRC-32 against the
// byte-wise one. The strict Python oracle (F6), the round trips (F3) and the driver are tests/python/test_export.py's.
#include <doctest/doctest.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

#include "config/sha256.h"
#include "export/cdr.h"
#include "export/compress.h"
#include "export/crc32.h"
#include "export/errors.h"
#include "export/mcap.h"
#include "export/msgdefs.h"
#include "export/time_map.h"

using namespace balla;
using exporter::Codec;
using exporter::MsgType;

namespace {

struct TempDir {
  std::filesystem::path dir =
      std::filesystem::temp_directory_path() / ("balla_export_mcap_tests_" + std::to_string(::getpid()));
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};
const TempDir kTemp;

std::string temp_path(const std::string& name) {
  std::filesystem::create_directories(kTemp.dir);
  return (kTemp.dir / name).string();
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::string hex(const std::vector<char>& v) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (const char c : v) {
    s.push_back(d[(static_cast<unsigned char>(c) >> 4) & 15]);
    s.push_back(d[static_cast<unsigned char>(c) & 15]);
  }
  return s;
}

std::vector<char> unhex(const std::string& h) {
  std::vector<char> out;
  for (size_t i = 0; i + 1 < h.size(); i += 2) out.push_back(static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16)));
  return out;
}

template <class T>
T get(const std::string& b, size_t at) {
  T v{};
  REQUIRE(at + sizeof(T) <= b.size());
  std::memcpy(&v, b.data() + at, sizeof(T));
  return v;
}

uint32_t crc(const std::string& b, size_t from, size_t to) { return exporter::crc32(0, b.data() + from, to - from); }

struct Columns {
  std::vector<uint16_t> x, y;
  std::vector<uint8_t> p;
  std::vector<int64_t> t;
  void add(int64_t t_ns, uint16_t xi, uint16_t yi, uint8_t pi) {
    t.push_back(t_ns);
    x.push_back(xi);
    y.push_back(yi);
    p.push_back(pi);
  }
  exporter::Events view(size_t a = 0, size_t b = SIZE_MAX) const {
    b = std::min(b, t.size());
    return {x.data() + a, y.data() + a, p.data() + a, t.data() + a, b - a};
  }
};

// R-FMT's seven golden events (out/t3_vectors.json) at t = 1000 t_us + 419 ns.
Columns golden_events() {
  const int64_t t_us[] = {0, 0, 0, 1, 64, 4096, 4097};
  const uint16_t x[] = {10, 11, 12, 639, 3, 3, 1279}, y[] = {5, 5, 5, 479, 0, 0, 719};
  const uint8_t p[] = {1, 1, 1, 0, 1, 0, 1};
  Columns c;
  for (size_t i = 0; i < 7; ++i) c.add(1000 * t_us[i] + 419, x[i], y[i], p[i]);
  return c;
}

constexpr int64_t kEpoch = int64_t{1'760'000'000} * 1'000'000'000;

// A stream with same-microsecond groups, idle gaps up to 30 ms and row runs, from 3 s + 123 ns.
Columns dense_stream(size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  Columns c;
  int64_t t = 3'000'000'123;
  for (size_t i = 0; i < n; ++i) {
    const uint64_t r = rng();
    if (r % 200 == 0) t += static_cast<int64_t>(r % 30'000'000);
    else if (r % 3 != 0) t += static_cast<int64_t>(r % 3000);
    c.add(t, static_cast<uint16_t>((r >> 32) % 1280), static_cast<uint16_t>((r >> 20) % 4 == 0 ? (r >> 24) % 720 : 100),
          static_cast<uint8_t>((r >> 40) & 1));
  }
  return c;
}

struct Imu {
  std::vector<int64_t> t;
  std::vector<double> g, a;
  exporter::ImuView view(size_t i0 = 0, size_t i1 = SIZE_MAX) const {
    i1 = std::min(i1, t.size());
    return {t.data() + i0, g.data() + 3 * i0, a.data() + 3 * i0, nullptr, nullptr, i1 - i0};
  }
};

Imu imu_samples(size_t n, int64_t t0, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> d(0, 3);
  Imu s;
  for (size_t i = 0; i < n; ++i) {
    s.t.push_back(t0 + 1'000'000 * static_cast<int64_t>(i) + 417);
    for (int k = 0; k < 3; ++k) {
      s.g.push_back(d(rng));
      s.a.push_back(9.8 + d(rng));
    }
  }
  return s;
}

// A file of one camera's events (in batches of `batch`, 0: one batch), each batch followed by the IMU samples up to
// its last event (the driver's interleave, by time); a second event channel, declared but never written, between
// them.
std::string write_mcap(Codec codec, int threads, const Columns& ev, const Imu& imu, size_t batch,
                       uint64_t chunk_bytes = exporter::mcap::kChunkBytes,
                       const exporter::TimeMap& m = {3'000'000, 1'000'000'000}, int64_t packet_ns = 1'000'000) {
  const std::string path = temp_path("file_" + std::to_string(static_cast<int>(codec)) + ".mcap");
  exporter::McapWriter w(path, codec, chunk_bytes, m, threads, "BallaSim test");
  const uint16_t ce = w.channel("/cam/events", MsgType::kEventPacketEvt3);
  (void)w.channel("/unused/events", MsgType::kEventPacketEvt3);
  const uint16_t ci = w.channel("/cam/imu", MsgType::kImu);
  const size_t step = batch == 0 ? ev.t.size() : batch;
  size_t j = 0;
  for (size_t a = 0; a < ev.t.size(); a += step) {
    const exporter::Events e = ev.view(a, a + step);
    w.event_packets_evt3(ce, "cam", 1280, 720, e, packet_ns);
    const int64_t last = e.t_ns[e.n - 1];
    size_t k = j;
    while (k < imu.t.size() && imu.t[k] <= last) ++k;
    if (k > j) w.imu(ci, "cam_imu", imu.view(j, k), exporter::CovarianceMode::kModel, {1e-6, 2e-3});
    j = k;
  }
  if (j < imu.t.size()) w.imu(ci, "cam_imu", imu.view(j), exporter::CovarianceMode::kModel, {1e-6, 2e-3});
  const exporter::FileInfo info = w.close();
  const std::string bytes = read_file(path);
  CHECK(info.bytes == bytes.size());
  CHECK(info.sha256 == config::sha256_hex(bytes));
  return bytes;
}

// One record: opcode, position, content.
struct Rec {
  uint8_t op = 0;
  size_t pos = 0;
  std::string c;
};

std::vector<Rec> records(const std::string& b, size_t at, size_t end) {
  std::vector<Rec> out;
  while (at < end) {
    Rec r;
    r.op = get<uint8_t>(b, at);
    r.pos = at;
    const auto n = get<uint64_t>(b, at + 1);
    REQUIRE(at + 9 + n <= end);
    r.c = b.substr(at + 9, n);
    at += 9 + n;
    out.push_back(std::move(r));
  }
  CHECK(at == end);
  return out;
}

std::string str(const std::string& c, size_t& at) {
  const auto n = get<uint32_t>(c, at);
  REQUIRE(at + 4 + n <= c.size());
  at += 4 + n;
  return c.substr(at - n, n);
}

struct Message {
  uint16_t channel;
  uint32_t sequence;
  uint64_t log, publish;
  std::string data;
};

// A file read strictly, record by record.
struct Parsed {
  std::string profile, library;
  std::map<uint16_t, std::string> schemas, channels;     // record contents by id, from the data section
  std::map<uint16_t, std::string> topics;                // channel id -> topic
  std::vector<uint16_t> channel_order;                   // channel ids in file order
  std::vector<std::string> chunk_compression;
  std::vector<std::string> raw;                          // each chunk's records, decompressed
  std::vector<Message> messages;                         // in file order
  uint64_t stat_messages = 0, stat_start = 0, stat_end = 0;
  uint16_t stat_schemas = 0;
  uint32_t stat_channels = 0, stat_chunks = 0;
  std::map<uint16_t, uint64_t> stat_counts;
};

Parsed parse(const std::string& b) {
  Parsed p;
  const std::string magic(exporter::mcap::kMagic);
  REQUIRE(b.size() >= 16 + 9 + 20);
  REQUIRE(b.compare(0, 8, magic) == 0);
  REQUIRE(b.compare(b.size() - 8, 8, magic) == 0);
  const size_t footer = b.size() - 8 - 29;
  REQUIRE(get<uint8_t>(b, footer) == 0x02);
  REQUIRE(get<uint64_t>(b, footer + 1) == 20);
  const auto summary_start = get<uint64_t>(b, footer + 9);
  const auto offset_start = get<uint64_t>(b, footer + 17);
  CHECK(get<uint32_t>(b, footer + 25) == crc(b, summary_start, footer + 25));  // summary through the offsets
  // The data section.
  const std::vector<Rec> data = records(b, 8, summary_start);
  REQUIRE(!data.empty());
  REQUIRE(data.front().op == 0x01);
  size_t at = 0;
  p.profile = str(data.front().c, at);
  p.library = str(data.front().c, at);
  REQUIRE(data.back().op == 0x0F);
  CHECK(get<uint32_t>(data.back().c, 0) == crc(b, 0, data.back().pos));  // magic and header included
  struct ChunkSeen {
    size_t pos, length;
    uint64_t start, end, raw_size, z_size;
    std::string compression;
    std::map<uint16_t, std::vector<std::pair<uint64_t, uint64_t>>> index;
    std::map<uint16_t, uint64_t> mi_pos;
    uint64_t mi_length = 0;
  };
  std::vector<ChunkSeen> chunks;
  for (size_t i = 1; i + 1 < data.size(); ++i) {
    const Rec& r = data[i];
    if (r.op == 0x03) {
      const auto id = get<uint16_t>(r.c, 0);
      CHECK(id == p.schemas.size() + 1);  // ids from 1 in first-use order
      p.schemas[id] = r.c;
    } else if (r.op == 0x04) {
      const auto id = get<uint16_t>(r.c, 0);
      CHECK(id == p.channels.size() + 1);
      CHECK(p.schemas.count(get<uint16_t>(r.c, 2)) == 1);  // its schema first
      p.channels[id] = r.c;
      size_t k = 4;
      p.topics[id] = str(r.c, k);
      p.channel_order.push_back(id);
    } else if (r.op == 0x06) {
      ChunkSeen c{r.pos, 9 + r.c.size(), get<uint64_t>(r.c, 0), get<uint64_t>(r.c, 8), get<uint64_t>(r.c, 16), 0, {},
                  {}, {}, 0};
      const auto raw_crc = get<uint32_t>(r.c, 24);
      size_t k = 28;
      c.compression = str(r.c, k);
      c.z_size = get<uint64_t>(r.c, k);
      REQUIRE(k + 8 + c.z_size == r.c.size());
      const std::string z = r.c.substr(k + 8);
      const Codec codec = c.compression.empty() ? Codec::kNone : c.compression == "lz4" ? Codec::kLz4 : Codec::kZstd;
      REQUIRE((c.compression.empty() || c.compression == "lz4" || c.compression == "zstd"));
      const std::vector<char> raw = codec == Codec::kNone ? std::vector<char>(z.begin(), z.end())
                                                          : exporter::decompress(codec, z.data(), z.size(), c.raw_size);
      const std::string rs(raw.begin(), raw.end());
      REQUIRE(rs.size() == c.raw_size);
      CHECK(crc(rs, 0, rs.size()) == raw_crc);
      uint64_t lo = std::numeric_limits<uint64_t>::max(), hi = 0;
      for (const Rec& m : records(rs, 0, rs.size())) {
        REQUIRE(m.op == 0x05);  // only messages in the chunks
        Message msg{get<uint16_t>(m.c, 0), get<uint32_t>(m.c, 2), get<uint64_t>(m.c, 6), get<uint64_t>(m.c, 14),
                    m.c.substr(22)};
        REQUIRE(p.channels.count(msg.channel) == 1);  // its channel before it
        c.index[msg.channel].emplace_back(msg.log, m.pos);
        lo = std::min(lo, msg.log);
        hi = std::max(hi, msg.log);
        p.messages.push_back(std::move(msg));
      }
      CHECK(c.start == lo);
      CHECK(c.end == hi);
      p.raw.push_back(rs);
      p.chunk_compression.push_back(c.compression);
      chunks.push_back(std::move(c));
    } else if (r.op == 0x07) {
      REQUIRE(!chunks.empty());
      ChunkSeen& c = chunks.back();
      const auto id = get<uint16_t>(r.c, 0);
      const auto n = get<uint32_t>(r.c, 2);
      REQUIRE(6 + n == r.c.size());
      CHECK((c.mi_pos.empty() || c.mi_pos.rbegin()->first < id));  // id order
      c.mi_pos[id] = r.pos;
      c.mi_length += 9 + r.c.size();
      std::vector<std::pair<uint64_t, uint64_t>> got;
      for (size_t k = 0; k < n; k += 16) got.emplace_back(get<uint64_t>(r.c, 6 + k), get<uint64_t>(r.c, 14 + k));
      CHECK(got == c.index.at(id));  // every message of the chunk on that channel, in order
      CHECK(data[i - 1].op == (c.mi_pos.size() == 1 ? 0x06 : 0x07));  // right after its chunk
    } else {
      FAIL("unexpected record in the data section");
    }
  }
  for (const ChunkSeen& c : chunks) CHECK(c.mi_pos.size() == c.index.size());
  // The summary: groups of one opcode, then their offsets.
  const std::vector<Rec> summary = records(b, summary_start, offset_start);
  const std::vector<Rec> offsets = records(b, offset_start, footer);
  std::vector<std::tuple<uint8_t, uint64_t, uint64_t>> groups;
  for (const Rec& r : summary) {
    if (groups.empty() || std::get<0>(groups.back()) != r.op) groups.emplace_back(r.op, r.pos, 0);
    std::get<2>(groups.back()) = r.pos + 9 + r.c.size() - std::get<1>(groups.back());
  }
  REQUIRE(offsets.size() == groups.size());
  for (size_t g = 0; g < groups.size(); ++g) {
    CHECK(offsets[g].op == 0x0E);
    CHECK(get<uint8_t>(offsets[g].c, 0) == std::get<0>(groups[g]));
    CHECK(get<uint64_t>(offsets[g].c, 1) == std::get<1>(groups[g]));
    CHECK(get<uint64_t>(offsets[g].c, 9) == std::get<2>(groups[g]));
  }
  std::vector<std::string> sum_schemas, sum_channels;
  size_t chunk_index = 0, stats = 0;
  for (const Rec& r : summary) {
    if (r.op == 0x03) {
      sum_schemas.push_back(r.c);
    } else if (r.op == 0x04) {
      sum_channels.push_back(r.c);
    } else if (r.op == 0x0B) {
      ++stats;
      p.stat_messages = get<uint64_t>(r.c, 0);
      p.stat_schemas = get<uint16_t>(r.c, 8);
      p.stat_channels = get<uint32_t>(r.c, 10);
      CHECK(get<uint32_t>(r.c, 14) == 0);  // attachments
      CHECK(get<uint32_t>(r.c, 18) == 0);  // metadata
      p.stat_chunks = get<uint32_t>(r.c, 22);
      p.stat_start = get<uint64_t>(r.c, 26);
      p.stat_end = get<uint64_t>(r.c, 34);
      const auto n = get<uint32_t>(r.c, 42);
      REQUIRE(46 + n == r.c.size());
      for (size_t k = 0; k < n; k += 10) p.stat_counts[get<uint16_t>(r.c, 46 + k)] = get<uint64_t>(r.c, 48 + k);
    } else if (r.op == 0x08) {
      REQUIRE(chunk_index < chunks.size());
      const ChunkSeen& c = chunks[chunk_index++];
      CHECK(get<uint64_t>(r.c, 0) == c.start);
      CHECK(get<uint64_t>(r.c, 8) == c.end);
      CHECK(get<uint64_t>(r.c, 16) == c.pos);
      CHECK(get<uint64_t>(r.c, 24) == c.length);
      const auto n = get<uint32_t>(r.c, 32);
      std::map<uint16_t, uint64_t> mi;
      for (size_t k = 0; k < n; k += 10) mi[get<uint16_t>(r.c, 36 + k)] = get<uint64_t>(r.c, 38 + k);
      CHECK(mi == c.mi_pos);
      size_t k = 36 + n;
      CHECK(get<uint64_t>(r.c, k) == c.mi_length);
      k += 8;
      CHECK(str(r.c, k) == c.compression);
      CHECK(get<uint64_t>(r.c, k) == c.z_size);
      CHECK(get<uint64_t>(r.c, k + 8) == c.raw_size);
      CHECK(k + 16 == r.c.size());
    } else {
      FAIL("unexpected record in the summary");
    }
  }
  CHECK(stats == 1);
  CHECK(chunk_index == chunks.size());
  std::vector<std::string> data_schemas, data_channels;
  for (const auto& [id, c] : p.schemas) data_schemas.push_back(c);
  for (const auto& [id, c] : p.channels) data_channels.push_back(c);
  CHECK(sum_schemas == data_schemas);  // the summary repeats them byte for byte, in id order
  CHECK(sum_channels == data_channels);
  CHECK(p.stat_messages == p.messages.size());
  CHECK(p.stat_schemas == p.schemas.size());
  CHECK(p.stat_channels == p.channels.size());
  CHECK(p.stat_chunks == chunks.size());
  std::map<uint16_t, uint64_t> counts;
  uint64_t lo = 0, hi = 0;
  for (size_t i = 0; i < p.messages.size(); ++i) {
    ++counts[p.messages[i].channel];
    lo = i == 0 ? p.messages[i].log : std::min(lo, p.messages[i].log);
    hi = i == 0 ? p.messages[i].log : std::max(hi, p.messages[i].log);
  }
  CHECK(p.stat_counts == counts);
  CHECK(p.stat_start == lo);
  CHECK(p.stat_end == hi);
  return p;
}

// A CDR reader of one message: aligned primitives after the 4-byte encapsulation header.
struct CdrReader {
  const std::string& b;
  size_t o = 4;
  template <class T>
  T next() {
    o += (sizeof(T) - (o - 4) % sizeof(T)) % sizeof(T);
    const T v = get<T>(b, o);
    o += sizeof(T);
    return v;
  }
  std::string string() {
    const auto n = next<uint32_t>();
    REQUIRE(n >= 1);
    REQUIRE(b[o + n - 1] == '\0');
    o += n;
    return b.substr(o - n, n - 1);
  }
};

uint32_t byte_crc(uint32_t c, const unsigned char* p, size_t n) {  // the plain bitwise CRC-32, as a reference
  c = ~c;
  for (size_t i = 0; i < n; ++i) {
    c ^= p[i];
    for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
  }
  return ~c;
}

}  // namespace

TEST_CASE("CRC-32: slicing by 8 equals the bitwise CRC at every length, alignment and split") {
  std::mt19937_64 rng(7);
  std::vector<unsigned char> buf(1100);
  for (auto& v : buf) v = static_cast<unsigned char>(rng());
  for (size_t off = 0; off < 9; ++off) {
    for (size_t n = 0; n + off <= 300; ++n) {
      CHECK(exporter::crc32(0, buf.data() + off, n) == byte_crc(0, buf.data() + off, n));
    }
  }
  for (size_t cut = 0; cut <= 1100; cut += 37) {
    CHECK(exporter::crc32(exporter::crc32(0, buf.data(), cut), buf.data() + cut, 1100 - cut) ==
          byte_crc(0, buf.data(), 1100));
  }
}

TEST_CASE("F1 CDR: R-FMT's golden Imu byte for byte; its EventPacket with EVT 3.0 words without vector words") {
  // R-FMT's cdr_imu_hex: the ROS 2 Quaternion's default orientation (0, 0, 0, 1), orientation_covariance[0] = -1 and
  // zero covariances (kZero).
  std::vector<char> msg;
  const double g[3] = {0.1, 0.2, 0.3}, a[3] = {0.0, 0.0, 9.81};
  exporter::cdr::put_imu(msg, kEpoch, "imu0", g, a, {}, {});
  CHECK(hex(msg) ==
        "000100000078e7680000000005000000696d75300000000000000000000000000000000000000000000000000000000000000000"
        "000000000000f03f000000000000f0bf000000000000000000000000000000000000000000000000000000000000000000000000"
        "000000000000000000000000000000000000000000000000000000009a9999999999b93f9a9999999999c93f333333333333d33f"
        "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000000000000000000001f85eb51b89e23400000000000000000"
        "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        "000000000000000000000000");
  // R-FMT's cdr_eventpacket_hex regenerated by its own serializer on the EVT 3.0 words without vector words (A5):
  // header stamp 1760000000 s, frame cam0, 1280 x 720, seq 0, time_base 0, "evt3", little-endian, 36 payload bytes.
  const Columns c = golden_events();
  msg.clear();
  exporter::cdr::put_event_packet_evt3(msg, 0, kEpoch, "cam0", 1280, 720, c.view(), {0, kEpoch});
  CHECK(hex(msg) ==
        "000100000078e768000000000500000063616d3000000000d0020000000500000000000000000000000000000000000000000000"
        "050000006576743300000000240000000080006005000a280b280c280160df017f224060000003280180006003200160cf02ff2c");
}

TEST_CASE("F1 MCAP: R-FMT's tiny.mcap byte for byte through the container") {
  // Its one message: R-FMT's CDR EventPacket (with its vector words) on /cam0/events, logged and published at the
  // last event's microsecond; library 'BallaSim proto'; schema text 'uint8[] events\n'; no compression.
  const std::vector<char> data = unhex(
      "000100000078e768000000000500000063616d3000000000d0020000000500000000000000000000000000000000000000000000"
      "050000006576743300000000220000000080006005000a3807400160df017f224060000003280180006003200160cf02ff2c");
  REQUIRE(data.size() == 102);
  const std::string path = temp_path("tiny.mcap");
  exporter::mcap::McapFile f(path, Codec::kNone, exporter::mcap::kChunkBytes, 1, "ros2", "BallaSim proto");
  const uint32_t s = f.add_schema("event_camera_msgs/msg/EventPacket", "ros2msg", "uint8[] events\n");
  const uint32_t ch = f.add_channel(s, "/cam0/events", "cdr", {{"offered_qos_profiles", ""}});
  f.write(ch, 0, kEpoch + 4'097'000, kEpoch + 4'097'000, data.data(), data.size());
  const exporter::FileInfo info = f.close();
  CHECK(info.bytes == 850);
  CHECK(info.sha256 == "1e7313aeba4e764247ae088eb2aa5dd41f39f1154c27c66bbb2233149e930fc2");
  const Parsed p = parse(read_file(path));
  CHECK(p.profile == "ros2");
  CHECK(p.messages.size() == 1);
}

TEST_CASE("Layout: schemas and channels before their first message, indexes, statistics, summary and footer") {
  const Columns ev = dense_stream(60'000, 3);
  const Imu imu = imu_samples(400, 3'000'000'000, 5);
  const std::string b = write_mcap(Codec::kNone, 1, ev, imu, 4096, 64 * 1024);
  const Parsed p = parse(b);
  CHECK(p.profile == "ros2");
  CHECK(p.library == "BallaSim test");
  CHECK(p.raw.size() > 3);
  // The two used channels, ids in first-use order; the declared but unused one stays out, and so does nothing else.
  REQUIRE(p.channels.size() == 2);
  CHECK(p.topics.at(1) == "/cam/events");
  CHECK(p.topics.at(2) == "/cam/imu");
  for (const auto& [id, c] : p.channels) {
    size_t k = 4;
    (void)str(c, k);
    CHECK(str(c, k) == "cdr");
    CHECK(get<uint32_t>(c, k) == 4 + 20 + 4);  // the map: "offered_qos_profiles" -> ""
    k += 4;
    CHECK(str(c, k) == "offered_qos_profiles");
    CHECK(str(c, k).empty());
    CHECK(k == c.size());
  }
  REQUIRE(p.schemas.size() == 2);
  const std::pair<const char*, MsgType> types[] = {{"event_camera_msgs/msg/EventPacket", MsgType::kEventPacketEvt3},
                                                   {"sensor_msgs/msg/Imu", MsgType::kImu}};
  for (uint16_t id = 1; id <= 2; ++id) {
    const std::string& c = p.schemas.at(id);
    size_t k = 2;
    CHECK(str(c, k) == types[id - 1].first);
    CHECK(str(c, k) == "ros2msg");
    CHECK(str(c, k) == exporter::ros2_msgdef(types[id - 1].second).definition);
    CHECK(get<uint16_t>(p.channels.at(id), 2) == id);
  }
  // Messages: per channel, sequence = the message index, log time = publish time = the CDR header stamp.
  std::map<uint16_t, uint32_t> seq;
  size_t events = 0, samples = 0;
  for (const Message& m : p.messages) {
    CHECK(m.sequence == seq[m.channel]++);
    CHECK(m.log == m.publish);
    CdrReader r{m.data};
    const auto sec = r.next<int32_t>();
    const auto nsec = r.next<uint32_t>();
    CHECK(m.log == static_cast<uint64_t>(sec) * 1'000'000'000 + nsec);
    if (m.channel == 1) {
      CHECK(r.string() == "cam");
      (void)r.next<uint32_t>();
      (void)r.next<uint32_t>();
      CHECK(r.next<uint64_t>() == seq[1] - 1);  // EventPacket.seq
      ++events;
    } else {
      CHECK(r.string() == "cam_imu");
      ++samples;
    }
  }
  CHECK(samples == imu.t.size());
  CHECK(events > 100);
}

TEST_CASE("F7 lz4 and zstd: the records of none's chunks, the same bytes at 1, 2, 4 and 8 threads") {
  const Columns ev = dense_stream(80'000, 11);
  const Imu imu = imu_samples(600, 3'000'000'000, 2);
  const Parsed none = parse(write_mcap(Codec::kNone, 1, ev, imu, 0, 128 * 1024));
  for (const Codec codec : {Codec::kLz4, Codec::kZstd}) {
    const std::string one = write_mcap(codec, 1, ev, imu, 0, 128 * 1024);
    const Parsed p = parse(one);
    CHECK(p.raw == none.raw);
    for (const std::string& c : p.chunk_compression) CHECK(c == (codec == Codec::kLz4 ? "lz4" : "zstd"));
    for (const int threads : {2, 4, 8}) CHECK(write_mcap(codec, threads, ev, imu, 0, 128 * 1024) == one);
  }
  for (const std::string& c : none.chunk_compression) CHECK(c.empty());
}

TEST_CASE("F8 streaming: one append and appends of 1, 7, 997 or 2^17 events give the same bytes") {
  const Columns ev = dense_stream(30'000, 4);
  const std::string one = write_mcap(Codec::kZstd, 2, ev, Imu{}, 0);
  for (const size_t batch : {size_t{7}, size_t{997}, size_t{1} << 17}) {
    CHECK(write_mcap(Codec::kZstd, 2, ev, Imu{}, batch) == one);
  }
  Columns small;
  for (size_t i = 0; i < 2000; ++i) small.add(ev.t[i], ev.x[i], ev.y[i], ev.p[i]);
  CHECK(write_mcap(Codec::kNone, 1, small, Imu{}, 1) == write_mcap(Codec::kNone, 1, small, Imu{}, 0));
}

TEST_CASE("EventPacket and Imu messages: the T1a packets, CDR layout, stamps and covariance modes") {
  // Origin 3 s; packets of 1 ms; events at u = 0, 999 (same packet), 1000, 1000, 5000 (next packets), and t_ns_rem 419.
  Columns c;
  for (const int64_t u : {0, 999, 1000, 1000, 5000, 20'004'096}) c.add(1000 * (3'000'000 + u) + 419, 7, 9, 1);
  const Parsed p = parse(write_mcap(Codec::kNone, 1, c, Imu{}, 2));
  REQUIRE(p.messages.size() == 4);
  const int64_t first_u[] = {0, 1000, 5000, 20'004'096};
  for (size_t k = 0; k < 4; ++k) {
    const Message& m = p.messages[k];
    const int64_t stamp = 1'000'000'000 + 1000 * first_u[k] + 419;  // T2 of the first event
    CHECK(m.log == static_cast<uint64_t>(stamp));
    CdrReader r{m.data};
    CHECK(m.data.compare(0, 4, std::string("\x00\x01\x00\x00", 4)) == 0);
    CHECK(r.next<int32_t>() == stamp / 1'000'000'000);
    CHECK(r.next<uint32_t>() == stamp % 1'000'000'000);
    CHECK(r.string() == "cam");
    CHECK(r.next<uint32_t>() == 720);
    CHECK(r.next<uint32_t>() == 1280);
    CHECK(r.next<uint64_t>() == k);  // seq
    CHECK(r.next<uint64_t>() == 0);  // time_base
    CHECK(r.string() == "evt3");
    CHECK(r.next<uint8_t>() == 0);  // is_bigendian
    const auto n = r.next<uint32_t>();
    CHECK(r.o + n == m.data.size());  // nothing after the words
    // The words start with the first event's TIME_HIGH and TIME_LOW (u mod 2^24), so the packet decodes alone.
    const auto u0 = static_cast<uint32_t>(first_u[k] & 0xFFFFFF);
    CHECK(get<uint16_t>(m.data, r.o) == (0x8000 | (u0 >> 12)));
    CHECK(get<uint16_t>(m.data, r.o + 2) == (0x6000 | (u0 & 0xFFF)));
  }
  // Imu: orientation (0, 0, 0, 1) with orientation_covariance[0] = -1; values bit-exact; kModel diag(white_var).
  const Imu imu = imu_samples(3, 3'000'000'000, 1);
  for (const auto mode : {exporter::CovarianceMode::kModel, exporter::CovarianceMode::kZero}) {
    const std::string path = temp_path("imu.mcap");
    exporter::McapWriter w(path, Codec::kNone, exporter::mcap::kChunkBytes, {3'000'000, 0}, 1, "BallaSim");
    const uint16_t ch = w.channel("/cam/imu", MsgType::kImu);
    w.imu(ch, "cam_imu", imu.view(), mode, {4e-6, 9e-4});
    const exporter::FileInfo info = w.close();
    REQUIRE(info.streams.size() == 1);
    CHECK(info.streams[0].kind == "imu");
    CHECK(info.streams[0].id == "/cam/imu");
    CHECK(info.streams[0].count == 3);
    CHECK(info.streams[0].t_first == imu.t[0] - 3'000'000'000);
    CHECK(info.streams[0].t_last == imu.t[2] - 3'000'000'000);
    const Parsed q = parse(read_file(path));
    REQUIRE(q.messages.size() == 3);
    for (size_t i = 0; i < 3; ++i) {
      const std::string& d = q.messages[i].data;
      REQUIRE(d.size() == 4 + 24 + 37 * 8);  // header (frame "cam_imu", padded to 8), then 37 doubles
      const double var = mode == exporter::CovarianceMode::kModel ? 4e-6 : 0.0;
      const size_t o = 28;
      CHECK(get<double>(d, o) == 0.0);
      CHECK(get<double>(d, o + 24) == 1.0);  // w
      CHECK(get<double>(d, o + 32) == -1.0);
      CHECK(std::memcmp(d.data() + o + 104, imu.g.data() + 3 * i, 24) == 0);
      CHECK(get<double>(d, o + 128) == var);
      CHECK(get<double>(d, o + 128 + 32) == var);
      CHECK(get<double>(d, o + 128 + 8) == 0.0);
      CHECK(std::memcmp(d.data() + o + 200, imu.a.data() + 3 * i, 24) == 0);
      CHECK(get<double>(d, o + 224) == (mode == exporter::CovarianceMode::kModel ? 9e-4 : 0.0));
    }
  }
}

TEST_CASE("A file without messages: the magic, the header, statistics of zero, no chunk") {
  const std::string path = temp_path("empty.mcap");
  exporter::McapWriter w(path, Codec::kZstd, exporter::mcap::kChunkBytes, {0, 1'000'000'000}, 4, "BallaSim");
  (void)w.channel("/cam/events", MsgType::kEventPacketEvt3);
  const exporter::FileInfo info = w.close();
  CHECK(info.streams.at(0).count == 0);
  const Parsed p = parse(read_file(path));
  CHECK(p.messages.empty());
  CHECK(p.channels.empty());
  CHECK(p.stat_chunks == 0);
  CHECK(p.stat_start == 0);
}

TEST_CASE("Refusals: codec, threads, chunk size, topics, types, settings, order, origin, stamps, after close") {
  using exporter::Refused;
  const std::string path = temp_path("refused.mcap");
  for (const Codec bad : {Codec::kBz2, Codec::kLz4High, Codec::kZstdHigh}) {
    CHECK_THROWS_AS(exporter::McapWriter(path, bad, 1024, {0, 0}, 1, "B"), Refused);
  }
  CHECK_THROWS_AS(exporter::McapWriter(path, Codec::kNone, 1024, {0, 0}, 0, "B"), Refused);
  CHECK_THROWS_AS(exporter::McapWriter(path, Codec::kNone, 1024, {0, 0}, 257, "B"), Refused);
  CHECK_THROWS_AS(exporter::McapWriter(path, Codec::kNone, 0, {0, 0}, 1, "B"), Refused);
  CHECK_THROWS_AS(exporter::McapWriter(path, Codec::kNone, (uint64_t{1} << 32) + 1, {0, 0}, 1, "B"), Refused);
  CHECK_THROWS_AS(exporter::McapWriter(path, Codec::kNone, 1024, {0, 999}, 1, "B"), Refused);
  exporter::McapWriter w(path, Codec::kNone, 1024, {3'000'000, 1'000'000'000}, 1, "B");
  // ROS 2's topic name rule.
  for (const char* bad : {"", "/", "1cam", "/1cam/events", "/cam events", "/cam-1", "/cam//events", "/cam/", "~cam",
                          "/cam__left/events", "{ns}/events", "cam\n"}) {
    CAPTURE(bad);
    CHECK_THROWS_AS(w.channel(bad, MsgType::kImu), Refused);
  }
  CHECK_THROWS_WITH_AS(w.channel("/cam/events", MsgType::kDvsEventArray), doctest::Contains("no ROS 2 type"), Refused);
  const uint16_t ce = w.channel("/cam/events", MsgType::kEventPacketEvt3);
  const uint16_t ci = w.channel("~/cam_imu", MsgType::kImu);
  (void)w.channel("cam/left_1/imu", MsgType::kImu);
  CHECK_THROWS_AS(w.channel("/cam/events", MsgType::kImu), Refused);  // each topic once
  Columns c;
  for (int i = 0; i < 5; ++i) c.add(3'000'000'000 + 1000 * i, static_cast<uint16_t>(i), 2, 1);
  CHECK_THROWS_AS(w.imu(ce, "cam", Imu{}.view(), exporter::CovarianceMode::kModel, {0, 0}), Refused);  // type
  CHECK_THROWS_AS(w.event_packets_evt3(ci, "cam", 64, 64, c.view(), 1'000'000), Refused);
  CHECK_THROWS_AS(w.event_packets_evt3(ce, "cam", 4096, 64, c.view(), 1'000'000), Refused);  // 11-bit
  CHECK_THROWS_AS(w.event_packets_evt3(ce, "cam", 64, 64, c.view(), 1500), Refused);  // not on the us grid
  CHECK_THROWS_AS(w.event_packets_evt3(ce, "cam", 64, 64, c.view(), 1000 * (int64_t{1} << 24) + 1000), Refused);
  // A refused first call leaves the channel unconfigured.
  Columns bad = c;
  bad.p[3] = 2;
  CHECK_THROWS_AS(w.event_packets_evt3(ce, "x", 32, 32, bad.view(), 2'000'000), Refused);
  w.event_packets_evt3(ce, "cam", 64, 64, c.view(0, 2), 1'000'000);
  CHECK_THROWS_AS(w.event_packets_evt3(ce, "cam2", 64, 64, c.view(2), 1'000'000), Refused);  // fixed by the first call
  CHECK_THROWS_AS(w.event_packets_evt3(ce, "cam", 64, 64, c.view(0, 1), 1'000'000), Refused);  // before the last
  w.event_packets_evt3(ce, "cam", 64, 64, c.view(2), 1'000'000);
  Imu imu = imu_samples(3, 3'000'000'000, 4);
  CHECK_THROWS_AS(w.imu(ci, "cam_imu", imu.view(), exporter::CovarianceMode::kModel, {-1, 0}), Refused);
  std::swap(imu.t[0], imu.t[1]);
  CHECK_THROWS_AS(w.imu(ci, "cam_imu", imu.view(), exporter::CovarianceMode::kModel, {0, 0}), Refused);
  Imu early = imu_samples(1, 2'000'000'000, 4);  // before the origin
  CHECK_THROWS_AS(w.imu(ci, "cam_imu", early.view(), exporter::CovarianceMode::kZero, {0, 0}), Refused);
  const exporter::FileInfo info = w.close();
  REQUIRE(info.streams.size() == 3);
  CHECK(info.streams[0].count == 5);
  CHECK(info.streams[0].t_first == 1'000'000'000);  // T2: t - 1000 origin_us + epoch_ns
  CHECK(info.streams[1].count == 0);
  CHECK_THROWS_AS(w.event_packets_evt3(ce, "cam", 64, 64, c.view(), 1'000'000), Refused);  // after close
  CHECK_THROWS_AS(w.channel("/late", MsgType::kImu), Refused);
  CHECK_THROWS_AS(w.close(), Refused);
  CHECK(parse(read_file(path)).channels.size() == 1);  // the unused channels stay out of the file
  // ROS 2 time and MCAP log times: a stamp below 0 or at 2^31 s is refused, naming the range; nothing is written.
  exporter::McapWriter neg(temp_path("neg.mcap"), Codec::kNone, 1024, {3'000'000, -1'000'000'000}, 1, "B");
  const uint16_t cn = neg.channel("/n", MsgType::kEventPacketEvt3);
  Columns one;
  one.add(3'000'000'000, 1, 1, 1);
  CHECK_THROWS_WITH_AS(neg.event_packets_evt3(cn, "n", 8, 8, one.view(), 1'000'000),
                       doctest::Contains("outside ROS 2's [0, 2^31 s)"), Refused);
  CHECK(neg.close().streams.at(0).count == 0);
  const int64_t last_epoch = (int64_t{1} << 31) * 1'000'000'000 - 1000;
  exporter::McapWriter late(temp_path("late.mcap"), Codec::kNone, 1024, {0, last_epoch}, 1, "B");
  const uint16_t cl = late.channel("/l", MsgType::kImu);
  const Imu at = imu_samples(2, 0, 1);  // the second sample lies past 2^31 s
  CHECK_THROWS_AS(late.imu(cl, "l", at.view(), exporter::CovarianceMode::kZero, {0, 0}), Refused);
  late.imu(cl, "l", at.view(0, 1), exporter::CovarianceMode::kZero, {0, 0});  // the last stamp ROS 2 time holds
  CHECK(late.close().streams.at(0).t_last == last_epoch + 417);
  CHECK(exporter::ros2_msgdef(MsgType::kImu).type == "sensor_msgs/msg/Imu");
  CHECK_THROWS_AS(exporter::ros2_msgdef(MsgType::kDvsEventArray), Refused);
  CHECK(exporter::ros2_msgdef(MsgType::kImage).type == "sensor_msgs/msg/Image");  // since A9
}

TEST_CASE("Packet capacity: the events whose words fit CDR's u32 byte count") {
  constexpr uint64_t kU32Max = 0xFFFFFFFFu;
  for (const int64_t us : {int64_t{1}, int64_t{1000}, int64_t{4096}, int64_t{1} << 24}) {
    CAPTURE(us);
    const uint64_t n = exporter::McapWriter::max_packet_events(us);
    const uint64_t time_high = 2 * (static_cast<uint64_t>(us) / 4096 + 2);
    CHECK(6 * n + time_high <= kU32Max);  // the most bytes n events can take fit
    CHECK(6 * (n + 1) + time_high > kU32Max);
  }
  // The bound is the encoder's: a packet's words are at most 3 per event plus its TIME_HIGH words.
  const Columns ev = dense_stream(20'000, 6);
  for (const int64_t us : {int64_t{1000}, int64_t{33'000}, int64_t{1} << 24}) {
    const Parsed p = parse(write_mcap(Codec::kNone, 1, ev, Imu{}, 0, exporter::mcap::kChunkBytes,
                                      {3'000'000, 1'000'000'000}, 1000 * us));
    for (const Message& m : p.messages) {
      CdrReader r{m.data};
      (void)r.next<int32_t>();
      (void)r.next<uint32_t>();
      (void)r.string();
      (void)r.next<uint32_t>();
      (void)r.next<uint32_t>();
      (void)r.next<uint64_t>();
      (void)r.next<uint64_t>();
      (void)r.string();
      (void)r.next<uint8_t>();
      const auto bytes = r.next<uint32_t>();
      size_t events = 0;
      for (size_t k = 0; k < bytes; k += 2) events += (get<uint16_t>(m.data, r.o + k) >> 12) == 0x2;  // ADDR_X
      CHECK(bytes <= 6 * events + 2 * (static_cast<uint64_t>(us) / 4096 + 2));
    }
  }
}
