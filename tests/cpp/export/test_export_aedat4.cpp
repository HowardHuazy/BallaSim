// The FlatBuffers builder and the AEDAT4 writer (docs/design/phase3/outputs.md §8.2): F1 R-FMT's golden IOHeader,
// EVTS packet, FileDataTable and infoNode, and its tiny.aedat4 (out/t3_vectors.json: 902 bytes, SHA-256 06589e06...)
// through the writer; dv-processing 2.0.4's own IMUS packet and FileDataTable (tests/data/export/t3_vectors.json)
// byte for byte; the builder's alignment, strings, shared vtables and forced defaults; every file re-read (the
// version, the IOHeader and its table position, the packets up to the table, each decompressed to its stream's
// size-prefixed FlatBuffer, the table's entries, the T1b times, dv's IMU units and the T1a packet grid); F7 every codec
// holds none's packets, the bytes the same at 1-8 threads; F8 one append against many (across streams, each
// stream's packets; the interleave follows the calls); the refusals of §6; decompress_upto. The strict Python oracle,
// the round trips (F3), dv-processing's reading and the driver are tests/python/test_export.py's.
#include <doctest/doctest.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include "config/sha256.h"
#include "export/aedat4.h"
#include "export/compress.h"
#include "export/errors.h"
#include "export/flatbuf_lite.h"
#include "export/time_map.h"

using namespace balla;
using exporter::Aedat4Writer;
using exporter::Codec;
namespace aedat4 = exporter::aedat4;

namespace {

struct TempDir {
  std::filesystem::path dir =
      std::filesystem::temp_directory_path() / ("balla_export_aedat4_tests_" + std::to_string(::getpid()));
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

template <class V>
std::string hex(const V& v) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (const char c : v) {
    s.push_back(d[(static_cast<unsigned char>(c) >> 4) & 15]);
    s.push_back(d[static_cast<unsigned char>(c) & 15]);
  }
  return s;
}

template <class T>
T get(const std::string& b, size_t at) {
  T v{};
  REQUIRE(at + sizeof(T) <= b.size());
  std::memcpy(&v, b.data() + at, sizeof(T));
  return v;
}

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

Columns head(const Columns& c, size_t n) {
  Columns out;
  for (size_t i = 0; i < n; ++i) out.add(c.t[i], c.x[i], c.y[i], c.p[i]);
  return out;
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

constexpr exporter::TimeMap kMap{3'000'000, 0};

// A file of one camera's events (in batches of `batch`, 0: one batch), each batch followed by the IMU samples up to
// its last event (the driver's interleave, by time).
std::string write_aedat4(Codec codec, int threads, const Columns& ev, const Imu& imu, size_t batch,
                         const exporter::TimeMap& m = kMap, int64_t packet_us = Aedat4Writer::kDefaultPacketUs,
                         exporter::FileInfo* info_out = nullptr) {
  const std::string path = temp_path("file_" + std::to_string(static_cast<int>(codec)) + ".aedat4");
  Aedat4Writer w(path, codec, m, threads, packet_us);
  const int32_t se = w.add_events_stream("cam", 1280, 720);
  const int32_t si = w.add_imu_stream("cam");
  const size_t step = batch == 0 ? std::max<size_t>(ev.t.size(), 1) : batch;
  size_t j = 0;
  for (size_t a = 0; a < ev.t.size(); a += step) {
    const exporter::Events e = ev.view(a, a + step);
    w.events(se, e);
    const int64_t last = e.t_ns[e.n - 1];
    size_t k = j;
    while (k < imu.t.size() && imu.t[k] <= last) ++k;
    if (k > j) w.imu(si, imu.view(j, k));
    j = k;
  }
  if (j < imu.t.size()) w.imu(si, imu.view(j));
  const exporter::FileInfo info = w.close();
  const std::string bytes = read_file(path);
  CHECK(info.bytes == bytes.size());
  CHECK(info.sha256 == config::sha256_hex(bytes));
  if (info_out != nullptr) *info_out = info;
  return bytes;
}

// A FlatBuffers table at pos: each vtable field's absolute position, or nullopt.
std::vector<std::optional<size_t>> table(const std::string& b, size_t pos) {
  const size_t vt = pos - static_cast<size_t>(static_cast<int64_t>(get<int32_t>(b, pos)));
  const auto vsize = get<uint16_t>(b, vt), tsize = get<uint16_t>(b, vt + 2);
  REQUIRE(vsize >= 4);
  REQUIRE(pos + tsize <= b.size());
  std::vector<std::optional<size_t>> out;
  for (size_t i = 0; i < (vsize - 4u) / 2; ++i) {
    const auto o = get<uint16_t>(b, vt + 4 + 2 * i);
    REQUIRE(o < tsize);
    out.push_back(o == 0 ? std::nullopt : std::optional<size_t>(pos + o));
  }
  return out;
}

size_t deref(const std::string& b, size_t pos) { return pos + get<uint32_t>(b, pos); }

template <class T>
T field(const std::string& b, const std::vector<std::optional<size_t>>& f, size_t i, T def = T{}) {
  return i < f.size() && f[i] ? get<T>(b, *f[i]) : def;
}

// A size-prefixed FlatBuffer: its body (after the prefix) with the identifier checked.
std::string body_of(const std::string& raw, const char* identifier) {
  REQUIRE(raw.size() >= 12);
  CHECK(get<uint32_t>(raw, 0) == raw.size() - 4);
  CHECK(raw.compare(8, 4, identifier) == 0);
  return raw.substr(4);
}

struct Event {
  int64_t t;
  int16_t x, y;
  bool p;
};
struct Sample {
  int64_t t;
  float v[10];  // temperature, accelerometer xyz, gyroscope xyz, magnetometer xyz
};
struct Packet {
  int32_t stream = 0;
  size_t offset = 0, size = 0;
  std::string raw;  // decompressed
};
struct Entry {
  int64_t offset, elements, t_start, t_end;
  int32_t stream, size;
};
struct File {
  int32_t compression = -1;
  int64_t position = -1;
  std::string info;
  size_t header_size = 0;
  std::vector<Packet> packets;
  std::vector<Entry> table;
};

std::vector<Event> events_of(const std::string& raw) {
  const std::string b = body_of(raw, "EVTS");
  const auto root = table(b, deref(b, 0));
  REQUIRE(root.size() == 1);
  const size_t vec = deref(b, *root[0]);
  const auto n = get<uint32_t>(b, vec);
  CHECK((vec + 4 + 4) % 8 == 0);  // 8-aligned in the size-prefixed buffer
  std::vector<Event> out;
  for (size_t i = 0; i < n; ++i) {
    const size_t at = vec + 4 + 16 * i;
    CHECK((get<uint8_t>(b, at + 12) <= 1));
    CHECK(b.compare(at + 13, 3, std::string(3, '\0')) == 0);
    out.push_back(
        {get<int64_t>(b, at), get<int16_t>(b, at + 8), get<int16_t>(b, at + 10), get<uint8_t>(b, at + 12) == 1});
  }
  return out;
}

std::vector<Sample> samples_of(const std::string& raw) {
  const std::string b = body_of(raw, "IMUS");
  const auto root = table(b, deref(b, 0));
  const size_t vec = deref(b, *root[0]);
  std::vector<Sample> out;
  for (size_t i = 0; i < get<uint32_t>(b, vec); ++i) {
    const auto f = table(b, deref(b, vec + 4 + 4 * i));
    Sample s{field<int64_t>(b, f, 0), {}};
    for (size_t k = 0; k < 10; ++k) s.v[k] = field<float>(b, f, k + 1, 0.0f);
    out.push_back(s);
  }
  return out;
}

// A file read strictly: the version, the IOHeader, the packets up to the table position, the table to the end.
File parse(const std::string& b) {
  File p;
  REQUIRE(b.compare(0, 14, aedat4::kVersion) == 0);
  const auto hsize = get<uint32_t>(b, 14);
  p.header_size = hsize + 4;
  const std::string h = body_of(b.substr(14, 4 + hsize), "IOHE");
  const auto f = table(h, deref(h, 0));
  REQUIRE(f.size() == 3);
  REQUIRE((f[0] && f[1] && f[2]));  // every field (ForceDefaults)
  p.compression = get<int32_t>(h, *f[0]);
  p.position = get<int64_t>(h, *f[1]);
  const size_t s = deref(h, *f[2]);
  p.info = h.substr(s + 4, get<uint32_t>(h, s));
  CHECK(h[s + 4 + p.info.size()] == '\0');
  const Codec codecs[] = {Codec::kNone, Codec::kLz4, Codec::kLz4High, Codec::kZstd, Codec::kZstdHigh};
  REQUIRE((p.compression >= 0 && p.compression < 5));
  const Codec c = codecs[p.compression];
  size_t at = 18 + hsize;
  REQUIRE(p.position > 0);
  const auto end = static_cast<size_t>(p.position);
  while (at < end) {
    Packet k;
    k.stream = get<int32_t>(b, at);
    const auto size = get<int32_t>(b, at + 4);
    REQUIRE(size > 0);
    k.offset = at + 8;
    k.size = static_cast<size_t>(size);
    REQUIRE(k.offset + k.size <= end);
    const std::vector<char> raw = exporter::decompress_upto(c, b.data() + k.offset, k.size, size_t{1} << 31);
    k.raw.assign(raw.begin(), raw.end());
    at = k.offset + k.size;
    p.packets.push_back(std::move(k));
  }
  CHECK(at == end);
  const std::vector<char> z = exporter::decompress_upto(c, b.data() + end, b.size() - end, size_t{1} << 31);
  const std::string t = body_of(std::string(z.begin(), z.end()), "FTAB");
  const auto root = table(t, deref(t, 0));
  const size_t vec = deref(t, *root[0]);
  for (size_t i = 0; i < get<uint32_t>(t, vec); ++i) {
    const auto e = table(t, deref(t, vec + 4 + 4 * i));
    REQUIRE(e.size() >= 2);  // fields equal to 0 are left out, the PacketHeader struct never
    REQUIRE(e[1]);
    p.table.push_back({field<int64_t>(t, e, 0), field<int64_t>(t, e, 2), field<int64_t>(t, e, 3),
                       field<int64_t>(t, e, 4), get<int32_t>(t, *e[1]), get<int32_t>(t, *e[1] + 4)});
  }
  // One entry per packet, in file order, describing it.
  REQUIRE(p.table.size() == p.packets.size());
  for (size_t i = 0; i < p.packets.size(); ++i) {
    const Packet& k = p.packets[i];
    const Entry& e = p.table[i];
    CHECK(e.offset == static_cast<int64_t>(k.offset));
    CHECK(e.stream == k.stream);
    CHECK(e.size == static_cast<int32_t>(k.size));
    if (k.raw.compare(8, 4, "EVTS") == 0) {
      const auto ev = events_of(k.raw);
      CHECK(e.elements == static_cast<int64_t>(ev.size()));
      CHECK(e.t_start == ev.front().t);
      CHECK(e.t_end == ev.back().t);
    } else {
      const auto im = samples_of(k.raw);
      CHECK(e.elements == static_cast<int64_t>(im.size()));
      CHECK(e.t_start == im.front().t);
      CHECK(e.t_end == im.back().t);
    }
  }
  return p;
}

// Every event and sample of a parsed file against the input: T1b times, input order, dv's units, the T1a grid.
void check_contents(const File& f, const Columns& ev, const Imu& imu, const exporter::TimeMap& m, int64_t packet_us) {
  std::vector<Event> got;
  std::vector<Sample> samples;
  std::vector<size_t> ev_sizes, imu_sizes;
  for (const Packet& k : f.packets) {
    if (k.stream == 0) {
      const auto e = events_of(k.raw);
      got.insert(got.end(), e.begin(), e.end());
      ev_sizes.push_back(e.size());
    } else {
      REQUIRE(k.stream == 1);
      const auto s = samples_of(k.raw);
      samples.insert(samples.end(), s.begin(), s.end());
      imu_sizes.push_back(s.size());
    }
  }
  REQUIRE(got.size() == ev.t.size());
  for (size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i].t == exporter::t1b_us(m, ev.t[i]));
    CHECK(got[i].x == static_cast<int16_t>(ev.x[i]));
    CHECK(got[i].y == static_cast<int16_t>(ev.y[i]));
    CHECK(got[i].p == (ev.p[i] == 1));
  }
  REQUIRE(samples.size() == imu.t.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    CHECK(samples[i].t == exporter::t1b_us(m, imu.t[i]));
    CHECK(samples[i].v[0] == 0.0f);
    for (size_t a = 0; a < 3; ++a) {
      CHECK(samples[i].v[1 + a] == static_cast<float>(imu.a[3 * i + a] / aedat4::kEarthG));
      CHECK(samples[i].v[4 + a] == static_cast<float>(imu.g[3 * i + a] * 180.0 / std::numbers::pi));
      CHECK(samples[i].v[7 + a] == 0.0f);
    }
  }
  // One packet per occupied cell of the T1a grid, per stream.
  auto cells = [&](const std::vector<int64_t>& t) {
    std::vector<size_t> sizes;
    int64_t cell = 0;
    for (size_t i = 0; i < t.size(); ++i) {
      const int64_t k = time::floor_div(exporter::t1a_us(m, t[i]), packet_us);
      if (i == 0 || k != cell) sizes.push_back(0);
      ++sizes.back();
      cell = k;
    }
    return sizes;
  };
  CHECK(ev_sizes == cells(ev.t));
  CHECK(imu_sizes == cells(imu.t));
}

}  // namespace

TEST_CASE("F1 AEDAT4: R-FMT's IOHeader, EVTS packet, FileDataTable and infoNode byte for byte") {
  const std::string xml =
      "<dv version=\"2.0\"><node name=\"outInfo\" path=\"/outInfo/\"><node name=\"0\" path=\"/outInfo/0/\"><attr "
      "key=\"compression\" type=\"string\">NONE</attr><attr key=\"originalModuleName\" "
      "type=\"string\">BallaSim</attr><attr key=\"originalOutputName\" type=\"string\">events</attr><attr "
      "key=\"typeDescription\" type=\"string\">Array of events (polarity ON/OFF).</attr><attr key=\"typeIdentifier\" "
      "type=\"string\">EVTS</attr><node name=\"info\" path=\"/outInfo/0/info/\"><attr key=\"sizeX\" "
      "type=\"int\">1280</attr><attr key=\"sizeY\" type=\"int\">720</attr><attr key=\"source\" "
      "type=\"string\">BallaSim_cam0</attr></node></node></node></dv>";
  const std::string info = aedat4::info_node(Codec::kNone, {{0, "cam0", "events", 1280, 720}});
  CHECK(info == xml);
  const std::vector<char> h = aedat4::io_header(Codec::kNone, -1, info);
  CHECK(hex(h) ==
        "8402000018000000494f48450000000000000a00140004000c0008000a000000000000000c000000ffffffffffffffff50020000" +
            hex(xml) + "00000000");
  const std::vector<char> pk = aedat4::event_packet(golden_events().view(), {0, 0});
  CHECK(hex(pk) ==
        "8c0000001000000045565453000006000800040006000000040000000700000000000000000000000a00050001000000000000000000"
        "00000b0005000100000000000000000000000c0005000100000001000000000000007f02df01000000004000000000000000030000"
        "0001000000001000000000000003000000000000000110000000000000ff04cf0201000000");
  const int64_t offset = static_cast<int64_t>(aedat4::kVersion.size() + h.size() + 8);
  CHECK(hex(aedat4::file_data_table({{offset, 0, static_cast<int32_t>(pk.size()), 7, 0, 4097}})) ==
        "54000000100000004654414200000600080004000600000004000000010000001400000000000e0024000c000400140000001c000e00"
        "000000000000900000009e0200000000000007000000000000000110000000000000");
}

TEST_CASE("F1 AEDAT4: R-FMT's tiny.aedat4 byte for byte through the writer") {
  const std::string path = temp_path("tiny.aedat4");
  Aedat4Writer w(path, Codec::kNone, {0, 0}, 1);
  const int32_t s = w.add_events_stream("cam0", 1280, 720);
  CHECK(s == 0);
  w.events(s, golden_events().view());
  const exporter::FileInfo info = w.close();
  CHECK(info.bytes == 902);
  CHECK(info.sha256 == "06589e06aae53ff137ad767f8da0551ba2d1458144a405933e19e2618af54deb");
  REQUIRE(info.streams.size() == 1);
  CHECK(info.streams[0].camera == "cam0");
  CHECK(info.streams[0].kind == "events");
  CHECK(info.streams[0].id == "0");
  CHECK(info.streams[0].count == 7);
  CHECK(info.streams[0].t_first == 0);
  CHECK(info.streams[0].t_last == 4097);
  const File f = parse(read_file(path));
  CHECK(f.compression == 0);
  CHECK(f.position == 902 - 88);
}

TEST_CASE("F1 AEDAT4: dv-processing 2.0.4's IMUS packet and FileDataTable byte for byte") {
  // tests/data/export/t3_vectors.json aedat4_dv_imus_hex: samples at 100 us (accelerometer (0.5, -0.25, 1) g,
  // gyroscope (10, 20, 30) deg/s) and 200 us (accelerometer (0, 0, 1) g); zero fields left out, as FlatBuffers does.
  const std::string dv_imus =
      "7c00000010000000494d555300000600080004000600000004000000020000003c0000001400000000000e0010000800000000000000"
      "04000e0000000000803fc800000000000000140024001c000000040008000c00100014001800140000000000003f000080be0000803f"
      "000020410000a0410000f0416400000000000000";
  aedat4::ImuRecord a{100, 0, {0.5f, -0.25f, 1.0f}, {10.0f, 20.0f, 30.0f}, {}}, b{200, 0, {0, 0, 1.0f}, {}, {}};
  CHECK(hex(aedat4::imu_packet({a, b})) == dv_imus);
  // The same packet through the writer, from published m/s^2 and rad/s.
  const std::string path = temp_path("dv_imu.aedat4");
  {
    Aedat4Writer w(path, Codec::kNone, {0, 0}, 1);
    (void)w.add_events_stream("camL", 1280, 720);
    const int32_t si = w.add_imu_stream("camL");
    Imu imu;
    imu.t = {100'419, 200'000};
    imu.a = {0.5 * aedat4::kEarthG, -0.25 * aedat4::kEarthG, aedat4::kEarthG, 0, 0, aedat4::kEarthG};
    for (const double deg : {10.0, 20.0, 30.0, 0.0, 0.0, 0.0}) imu.g.push_back(deg * std::numbers::pi / 180.0);
    w.imu(si, imu.view());
    w.close();
  }
  const File f = parse(read_file(path));
  REQUIRE(f.packets.size() == 1);
  CHECK(hex(f.packets[0].raw) == dv_imus);
  // aedat4_dv_ftab_hex from its entries (aedat4_dv_ftab_entries); the second shares the first's vtable layout but
  // not its padding, so each has its own.
  CHECK(hex(aedat4::file_data_table({{1390, 0, 144, 7, 0, 4097}, {1542, 1, 128, 2, 100, 200}})) ==
        "9c0000001400000046544142000000000000060008000400060000000400000002000000580000001400000000000e0032000c0004"
        "0014001c0024000e0000000100000080000000060600000000000002000000000000006400000000000000c80000000000000000"
        "00000000000e0024000c000400140000001c000e00000000000000900000006e0500000000000007000000000000000110000000"
        "000000");
}

TEST_CASE("FlatBuffers builder: alignment, strings, shared vtables, forced defaults, growth") {
  exporter::fb::Builder b;
  const uint32_t s = b.create_string("abc");
  uint32_t t[2];
  for (uint32_t& off : t) {  // two tables of one layout and size (a 12-byte vtable keeps the next one 4-aligned, so
                             // no padding enters it): the second shares the first's vtable
    const uint32_t start = b.start_table();
    b.add_scalar<int32_t>(1, 7, 0);
    b.add_scalar<int32_t>(0, 5, 0);
    b.add_scalar<int32_t>(2, 0, 0);  // the default: left out
    b.add_offset(3, s);
    off = b.end_table(start);
  }
  const uint32_t vec = b.create_offset_vector({t[0], t[1]});
  const uint32_t start = b.start_table();
  b.add_offset(0, vec);
  const std::vector<char> raw = b.finish(b.end_table(start), "TEST", true);
  const std::string buf(raw.begin(), raw.end());
  CHECK(buf.size() % 4 == 0);  // aligned to its largest scalar
  const std::string body = body_of(buf, "TEST");
  const auto root = table(body, deref(body, 0));
  const size_t v = deref(body, *root[0]);
  REQUIRE(get<uint32_t>(body, v) == 2);
  std::vector<size_t> vtables;
  for (size_t i = 0; i < 2; ++i) {
    const size_t pos = deref(body, v + 4 + 4 * i);
    const auto f = table(body, pos);
    REQUIRE(f.size() == 4);
    CHECK(get<int32_t>(body, *f[0]) == 5);
    CHECK(get<int32_t>(body, *f[1]) == 7);
    CHECK(!f[2]);
    const size_t str = deref(body, *f[3]);
    CHECK(body.substr(str, 8) == std::string("\x03\x00\x00\x00" "abc\0", 8));
    vtables.push_back(pos - static_cast<size_t>(static_cast<int64_t>(get<int32_t>(body, pos))));
  }
  CHECK(vtables[0] == vtables[1]);
  // An int64 field: 8-aligned in the size-prefixed buffer, its padding inside the table, so tables at other
  // alignments have other vtables.
  exporter::fb::Builder b8;
  std::vector<uint32_t> tables;
  for (int i = 0; i < 3; ++i) {
    const uint32_t start8 = b8.start_table();
    b8.add_scalar<int64_t>(0, 7 + i, 0);
    tables.push_back(b8.end_table(start8));
    (void)b8.push<int16_t>(1);  // shifts the next table's alignment
  }
  const uint32_t v8 = b8.create_offset_vector(tables);
  const uint32_t r8 = b8.start_table();
  b8.add_offset(0, v8);
  const std::vector<char> raw8 = b8.finish(b8.end_table(r8), "TST8", true);
  CHECK(raw8.size() % 8 == 0);
  const std::string body8 = body_of(std::string(raw8.begin(), raw8.end()), "TST8");
  const size_t vec8 = deref(body8, *table(body8, deref(body8, 0))[0]);
  for (size_t i = 0; i < 3; ++i) {
    const auto f8 = table(body8, deref(body8, vec8 + 4 + 4 * i));
    CHECK(get<int64_t>(body8, *f8[0]) == static_cast<int64_t>(7 + i));
    CHECK(*f8[0] % 8 == 4);
  }
  // ForceDefaults writes the default too.
  exporter::fb::Builder forced(true);
  const uint32_t fs = forced.start_table();
  forced.add_scalar<int32_t>(0, 0, 0);
  const std::vector<char> f = forced.finish(forced.end_table(fs), "", false);
  const std::string fb_str(f.begin(), f.end());
  CHECK(table(fb_str, deref(fb_str, 0))[0].has_value());
  // A struct vector far past the first buffer: the bytes survive the growth.
  Columns c;
  for (int i = 0; i < 100'000; ++i) c.add(1000 * i, static_cast<uint16_t>(i % 1280), 3, static_cast<uint8_t>(i & 1));
  const std::vector<char> pk = aedat4::event_packet(c.view(), {0, 0});
  const auto ev = events_of(std::string(pk.begin(), pk.end()));
  REQUIRE(ev.size() == c.t.size());
  for (size_t i = 0; i < ev.size(); i += 997) {
    CHECK(ev[i].t == c.t[i] / 1000);
    CHECK(ev[i].x == static_cast<int16_t>(c.x[i]));
  }
}

TEST_CASE("File layout: header, packets up to the table, the table's entries, T1b times, dv's units, the grid") {
  const Columns ev = dense_stream(60'000, 3);
  const Imu imu = imu_samples(400, 3'000'000'000, 5);
  for (const int64_t packet_us : {int64_t{10'000}, int64_t{1}, int64_t{33'000}}) {
    CAPTURE(packet_us);
    const exporter::TimeMap m{3'000'000, packet_us == 1 ? 0 : 5'000'000'000};
    Columns use = ev;
    if (packet_us == 1) use = head(ev, 3000);  // a packet per occupied microsecond
    exporter::FileInfo info;
    const std::string b = write_aedat4(Codec::kNone, 1, use, imu, 4096, m, packet_us, &info);
    const File f = parse(b);
    CHECK(f.compression == 0);
    CHECK(f.info == aedat4::info_node(Codec::kNone, {{0, "cam", "events", 1280, 720}, {1, "cam", "imu", 0, 0}}));
    check_contents(f, use, imu, m, packet_us);
    REQUIRE(info.streams.size() == 2);
    CHECK(info.streams[0].count == static_cast<int64_t>(use.t.size()));
    CHECK(info.streams[0].t_first == exporter::t1b_us(m, use.t.front()));
    CHECK(info.streams[0].t_last == exporter::t1b_us(m, use.t.back()));
    CHECK(info.streams[1].kind == "imu");
    CHECK(info.streams[1].id == "1");
    CHECK(info.streams[1].count == 400);
    CHECK(info.streams[1].t_last == exporter::t1b_us(m, imu.t.back()));
  }
  // The infoNode of an IMU stream and of a camera name that needs escaping.
  CHECK(aedat4::info_node(Codec::kZstd, {{1, "a&<\"'>", "imu", 0, 0}}) ==
        "<dv version=\"2.0\"><node name=\"outInfo\" path=\"/outInfo/\"><node name=\"1\" path=\"/outInfo/1/\"><attr "
        "key=\"compression\" type=\"string\">ZSTD</attr><attr key=\"originalModuleName\" type=\"string\">BallaSim"
        "</attr><attr key=\"originalOutputName\" type=\"string\">imu</attr><attr key=\"typeDescription\" "
        "type=\"string\">Inertial Measurement Unit data samples.</attr><attr key=\"typeIdentifier\" "
        "type=\"string\">IMUS</attr><node name=\"info\" path=\"/outInfo/1/info/\"><attr key=\"accelerometerUnitMps2\" "
        "type=\"double\">9.81007</attr><attr key=\"source\" type=\"string\">BallaSim_a&amp;&lt;&quot;&apos;&gt;"
        "</attr></node></node></node></dv>");
  // Stream nodes in the ConfigTree's name order: "10" before "2".
  std::vector<aedat4::StreamDecl> many;
  for (int32_t i = 0; i < 12; ++i) many.push_back({i, "c" + std::to_string(i), "imu", 0, 0});
  const std::string x = aedat4::info_node(Codec::kNone, many);
  CHECK(x.find("name=\"10\"") < x.find("name=\"2\""));
  // A file without items: the header, no packet, an empty table.
  const std::string path = temp_path("empty.aedat4");
  Aedat4Writer w(path, Codec::kLz4, kMap, 2);
  (void)w.add_imu_stream("cam");
  CHECK(w.close().streams.at(0).count == 0);
  const File e = parse(read_file(path));
  CHECK(e.packets.empty());
  CHECK(e.compression == 1);
}

TEST_CASE("F7: every codec holds none's packets; the same bytes at 1, 2, 4 and 8 threads") {
  const Columns ev = dense_stream(80'000, 6);
  const Imu imu = imu_samples(500, 3'000'000'000, 7);
  const File none = parse(write_aedat4(Codec::kNone, 1, ev, imu, 0));
  for (const Codec c : {Codec::kLz4, Codec::kLz4High, Codec::kZstd, Codec::kZstdHigh}) {
    CAPTURE(static_cast<int>(c));
    const std::string one = write_aedat4(c, 1, ev, imu, 0);
    const File f = parse(one);
    CHECK(f.compression == aedat4::compression_type(c));
    REQUIRE(f.packets.size() == none.packets.size());
    for (size_t i = 0; i < f.packets.size(); ++i) {
      CHECK(f.packets[i].stream == none.packets[i].stream);
      CHECK(f.packets[i].raw == none.packets[i].raw);
      CHECK(std::vector<char>(one.begin() + static_cast<std::ptrdiff_t>(f.packets[i].offset),
                              one.begin() + static_cast<std::ptrdiff_t>(f.packets[i].offset + f.packets[i].size)) ==
            exporter::compress(c, none.packets[i].raw.data(), none.packets[i].raw.size()));
    }
    if (c == Codec::kLz4 || c == Codec::kZstd) {
      for (const int threads : {2, 4, 8}) CHECK(write_aedat4(c, threads, ev, imu, 0) == one);
    }
  }
}

TEST_CASE("F8 streaming: one append and appends of 1, 7, 997 or 2^17 events give the same bytes; two streams") {
  const Columns ev = dense_stream(30'000, 4);
  const std::string one = write_aedat4(Codec::kZstd, 2, ev, Imu{}, 0);
  for (const size_t batch : {size_t{7}, size_t{997}, size_t{1} << 17}) {
    CHECK(write_aedat4(Codec::kZstd, 2, ev, Imu{}, batch) == one);
  }
  const Columns small = head(ev, 2000);
  CHECK(write_aedat4(Codec::kNone, 1, small, Imu{}, 1) == write_aedat4(Codec::kNone, 1, small, Imu{}, 0));
  // IMU batches too, the events fixed.
  const Imu imu = imu_samples(300, 3'000'000'000, 9);
  const std::string path = temp_path("imu_batches.aedat4");
  std::vector<std::string> files;
  for (const size_t step : {size_t{300}, size_t{1}, size_t{7}}) {
    Aedat4Writer w(path, Codec::kNone, kMap, 1);
    const int32_t si = w.add_imu_stream("cam");
    for (size_t a = 0; a < imu.t.size(); a += step) w.imu(si, imu.view(a, a + step));
    w.close();
    files.push_back(read_file(path));
  }
  CHECK(files[1] == files[0]);
  CHECK(files[2] == files[0]);
  // Across streams only each stream's packets are invariant: packets enter the file as they complete, so the
  // interleave follows the order of the calls (the driver's rounds).
  const Columns a = dense_stream(6000, 11), b = dense_stream(6000, 12);
  const auto two = [&](size_t cut) {
    const std::string p = temp_path("two_streams.aedat4");
    Aedat4Writer w(p, Codec::kNone, kMap, 2);
    const int32_t sa = w.add_events_stream("a", 1280, 720), sb = w.add_events_stream("b", 1280, 720);
    w.events(sa, a.view(0, cut));
    w.events(sb, b.view(0, cut));
    if (cut < a.t.size()) w.events(sa, a.view(cut));
    if (cut < b.t.size()) w.events(sb, b.view(cut));
    w.close();
    return read_file(p);
  };
  const File whole = parse(two(SIZE_MAX)), split = parse(two(1500));
  const auto packets_of = [](const File& f, int32_t stream) {
    std::vector<std::string> out;
    for (const Packet& k : f.packets) {
      if (k.stream == stream) out.push_back(k.raw);
    }
    return out;
  };
  const auto order = [](const File& f) {
    std::vector<int32_t> out;
    for (const Packet& k : f.packets) out.push_back(k.stream);
    return out;
  };
  for (const int32_t s : {0, 1}) CHECK(packets_of(split, s) == packets_of(whole, s));
  CHECK(order(split) != order(whole));
}

TEST_CASE("Refusals: codec, threads, packet, epoch, sizes, names, streams, order, origin, T1b, values, close") {
  using exporter::Refused;
  const std::string path = temp_path("refused.aedat4");
  std::filesystem::remove(path);
  CHECK_THROWS_AS(Aedat4Writer(path, Codec::kBz2, {0, 0}, 1), Refused);
  CHECK(!std::filesystem::exists(path));  // refused before the file is created
  CHECK_THROWS_AS(Aedat4Writer(path, Codec::kNone, {0, 0}, 0), Refused);
  CHECK_THROWS_AS(Aedat4Writer(path, Codec::kNone, {0, 0}, 257), Refused);
  CHECK_THROWS_AS(Aedat4Writer(path, Codec::kNone, {0, 999}, 1), Refused);
  CHECK_THROWS_AS(Aedat4Writer(path, Codec::kNone, {0, 0}, 1, 0), Refused);
  CHECK_THROWS_AS(Aedat4Writer(path, Codec::kNone, {0, 0}, 1, (int64_t{1} << 24) + 1), Refused);
  CHECK_THROWS_AS(aedat4::compression_type(Codec::kBz2), Refused);
  Aedat4Writer w(path, Codec::kNone, {3'000'000, -2'000'000'000}, 1);
  CHECK_THROWS_AS(w.add_events_stream("cam", 32768, 10), Refused);
  CHECK_THROWS_AS(w.add_events_stream("cam", 10, 0), Refused);
  CHECK_THROWS_AS(w.add_events_stream("", 10, 10), Refused);
  CHECK_THROWS_AS(w.add_events_stream("c\nam", 10, 10), Refused);
  const int32_t se = w.add_events_stream("cam", 32767, 64);
  CHECK_THROWS_AS(w.add_events_stream("cam", 10, 10), Refused);  // each kind once per camera
  const int32_t si = w.add_imu_stream("cam");
  CHECK_THROWS_AS(w.add_imu_stream("cam"), Refused);
  Columns c;
  for (int i = 0; i < 5; ++i) c.add(5'000'000'000 + 1000 * i, static_cast<uint16_t>(i), 2, 1);
  CHECK_THROWS_AS(w.events(si, c.view()), Refused);  // an imu stream
  CHECK_THROWS_AS(w.imu(se, Imu{{5'000'000'000}, {0, 0, 0}, {0, 0, 0}}.view()), Refused);
  CHECK_THROWS_AS(w.events(7, c.view()), Refused);
  CHECK_THROWS_AS(w.events(-1, c.view()), Refused);
  Columns bad = c;
  bad.p[3] = 2;
  CHECK_THROWS_AS(w.events(se, bad.view()), Refused);
  bad = c;
  bad.y[1] = 64;
  CHECK_THROWS_AS(w.events(se, bad.view()), Refused);
  bad = c;
  std::swap(bad.t[1], bad.t[2]);
  CHECK_THROWS_AS(w.events(se, bad.view()), Refused);
  Columns early;
  early.add(2'999'999'999, 1, 1, 1);  // before the origin
  CHECK_THROWS_AS(w.events(se, early.view()), Refused);
  Columns negative;
  negative.add(4'999'999'000, 1, 1, 1);  // T1b = 1999999 - 2000000 < 0
  CHECK_THROWS_WITH_AS(w.events(se, negative.view()), doctest::Contains("< 0"), Refused);
  // Nothing refused was held: the first accepted items declare the streams; then no more streams.
  w.events(se, c.view(0, 3));
  CHECK_THROWS_AS(w.add_events_stream("late", 10, 10), Refused);
  CHECK_THROWS_AS(w.events(se, c.view(0, 1)), Refused);  // before the last
  w.events(se, c.view(3));
  const exporter::FileInfo info = w.close();
  CHECK(info.streams.at(0).count == 5);
  CHECK(info.streams.at(0).t_first == 0);
  CHECK(info.streams.at(1).count == 0);
  CHECK_THROWS_AS(w.events(se, c.view()), Refused);
  CHECK_THROWS_AS(w.close(), Refused);
  const File f = parse(read_file(path));
  REQUIRE(f.packets.size() == 1);
  CHECK(events_of(f.packets[0].raw).size() == 5);
}

TEST_CASE("Capacity: the items a packet holds within its int32 size, the packets a table holds") {
  CHECK(Aedat4Writer::max_packet_events() == (aedat4::kMaxPacketBytes - 64) / 16);
  CHECK(Aedat4Writer::max_packet_samples() == (aedat4::kMaxPacketBytes - 64) / 96);
  CHECK(Aedat4Writer::max_packets() == (exporter::fb::kMaxBytes - 64) / 72);
  // The bound per sample: a table with every field (timestamp and ten floats), its own vtable, padding and slot.
  std::vector<aedat4::ImuRecord> rows;
  for (int i = 0; i < 64; ++i) {
    aedat4::ImuRecord r{i + 1, 1, {1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    r.accelerometer[i % 3] = static_cast<float>(i);  // other layouts too
    rows.push_back(r);
  }
  CHECK(aedat4::imu_packet(rows).size() <= 96 * rows.size() + 64);
  std::vector<aedat4::TableEntry> entries;
  for (int i = 0; i < 64; ++i) entries.push_back({1000 + i, i % 3, 100, i, i % 2 == 0 ? 0 : i, 2 * i});
  CHECK(aedat4::file_data_table(entries).size() <= 72 * entries.size() + 64);
}

TEST_CASE("decompress_upto: one frame of any size up to the limit") {
  std::vector<char> data(200'000);
  for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<char>((i * 7) % 251);
  for (const Codec c : {Codec::kNone, Codec::kLz4, Codec::kLz4High, Codec::kZstd, Codec::kZstdHigh, Codec::kBz2}) {
    CAPTURE(static_cast<int>(c));
    const std::vector<char> z = exporter::compress(c, data.data(), data.size());
    CHECK(exporter::decompress_upto(c, z.data(), z.size(), data.size()) == data);
    CHECK(exporter::decompress_upto(c, z.data(), z.size(), size_t{1} << 31) == data);
    CHECK_THROWS_AS(exporter::decompress_upto(c, z.data(), z.size(), data.size() - 1), exporter::Refused);
    std::vector<char> trailing = z;
    trailing.push_back(0);
    if (c != Codec::kNone) {
      CHECK_THROWS_AS(exporter::decompress_upto(c, trailing.data(), trailing.size(), size_t{1} << 31),
                      exporter::Refused);
      CHECK_THROWS_AS(exporter::decompress_upto(c, z.data(), z.size() - 3, size_t{1} << 31), exporter::Refused);
    }
  }
}
