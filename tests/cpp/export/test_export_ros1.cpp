// The rosbag1 writer (docs/design/phase3/outputs.md §8.2): F1 R-FMT's golden dvs_msgs/EventArray bytes and its tiny
// bag (out/t3_vectors.json: 7000 bytes, SHA-256 cb0d9604...) through the serializer and the container; the record
// layout (bag header padded to 4096 bytes and rewritten at close, chunks, index data, connections and chunk infos
// after index_pos, header fields in name order); the roslz4 chunk frame; F7 the bytes at 1, 2, 4 and 8 threads; F8
// one append against many; the packets on the T1a grid; the EventPacket evt3 payload and the Imu layout; the refusals
// of §6. The round trips through independent readers (export_oracles/ros1.py, tools/evimo2_bench/bag.py) and the MD5s
// (F2) are tests/python/test_export.py's.
#include <doctest/doctest.h>

#include <unistd.h>

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
#include <vector>

#include "config/sha256.h"
#include "export/compress.h"
#include "export/errors.h"
#include "export/msgdefs.h"
#include "export/ros1.h"
#include "export/time_map.h"

using namespace balla;
using exporter::Codec;
using exporter::MsgType;

namespace {

struct TempDir {
  std::filesystem::path dir =
      std::filesystem::temp_directory_path() / ("balla_export_ros1_tests_" + std::to_string(::getpid()));
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

template <class T>
T get(const std::string& b, size_t at) {
  T v{};
  REQUIRE(at + sizeof(T) <= b.size());
  std::memcpy(&v, b.data() + at, sizeof(T));
  return v;
}

// One record of a bag: its header fields and its data.
struct Record {
  std::map<std::string, std::string> fields;
  std::string data;
  size_t pos = 0;
};

// The records of b from `at` to the end (or `count` of them).
std::vector<Record> records(const std::string& b, size_t at, size_t count = SIZE_MAX) {
  std::vector<Record> out;
  while (at < b.size() && out.size() < count) {
    Record r;
    r.pos = at;
    const auto hlen = get<uint32_t>(b, at);
    size_t i = at + 4;
    std::string prev;
    while (i < at + 4 + hlen) {
      const auto n = get<uint32_t>(b, i);
      const std::string kv = b.substr(i + 4, n);
      const size_t eq = kv.find('=');
      REQUIRE(eq != std::string::npos);
      const std::string key = kv.substr(0, eq);
      CHECK(prev < key);  // name order, as ros::Header writes them
      prev = key;
      r.fields[key] = kv.substr(eq + 1);
      i += 4 + n;
    }
    const auto dlen = get<uint32_t>(b, i);
    r.data = b.substr(i + 4, dlen);
    at = i + 4 + dlen;
    out.push_back(std::move(r));
  }
  return out;
}

template <class T>
T field(const Record& r, const std::string& key) {
  REQUIRE(r.fields.count(key) == 1);
  return get<T>(r.fields.at(key), 0);
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

// A bag of one camera's events (dvs or evt3) and IMU, the events in batches of `batch` (0: one batch), each
// followed by the IMU samples up to its last event (the driver's interleave, by time).
std::string write_bag(Codec codec, int threads, const Columns& ev, const Imu& imu, MsgType ev_type, size_t batch,
                      const exporter::TimeMap& m = {3'000'000, 1'000'000'000}, uint32_t threshold = 768 * 1024,
                      int64_t packet_ns = 1'000'000) {
  const std::string path = temp_path("bag_" + std::to_string(static_cast<int>(codec)));
  exporter::Rosbag1Writer w(path, codec, threshold, m, threads);
  const uint32_t ce = w.connection("/cam/events", ev_type);
  const uint32_t ci = imu.t.empty() ? 0 : w.connection("/cam/imu", MsgType::kImu);
  const size_t step = batch == 0 ? ev.t.size() : batch;
  size_t j = 0;
  for (size_t a = 0; a < ev.t.size(); a += step) {
    const exporter::Events e = ev.view(a, a + step);
    if (ev_type == MsgType::kDvsEventArray) w.event_arrays(ce, "cam", 1280, 720, e, packet_ns);
    else w.event_packets_evt3(ce, "cam", 1280, 720, e, packet_ns);
    const int64_t last = e.t_ns[e.n - 1];
    size_t k = j;
    while (k < imu.t.size() && imu.t[k] <= last) ++k;
    if (k > j && !imu.t.empty()) w.imu(ci, "cam_imu", imu.view(j, k), exporter::CovarianceMode::kModel, {1e-6, 2e-3});
    j = k;
  }
  if (j < imu.t.size()) w.imu(ci, "cam_imu", imu.view(j), exporter::CovarianceMode::kModel, {1e-6, 2e-3});
  const exporter::FileInfo info = w.close();
  const std::string bytes = read_file(path);
  CHECK(info.bytes == bytes.size());
  CHECK(info.sha256 == config::sha256_hex(bytes));
  return bytes;
}

// The bag's chunks, decompressed, in file order, and the index section's records.
struct Parsed {
  std::vector<Record> chunks, index_data, connections, chunk_infos;
  std::vector<std::string> raw;  // each chunk's records, decompressed
  uint64_t index_pos = 0;
  uint32_t conn_count = 0, chunk_count = 0;
};

Parsed parse(const std::string& b) {
  Parsed p;
  REQUIRE(b.compare(0, 13, "#ROSBAG V2.0\n") == 0);
  const Record h = records(b, 13, 1).at(0);
  CHECK(h.fields.at("op") == std::string(1, '\x03'));
  CHECK(4 + (get<uint32_t>(b, 13)) + 4 + h.data.size() == 4096 + 8);  // fields and padding make 4096 bytes
  CHECK(h.data == std::string(h.data.size(), ' '));
  p.index_pos = field<uint64_t>(h, "index_pos");
  p.conn_count = field<uint32_t>(h, "conn_count");
  p.chunk_count = field<uint32_t>(h, "chunk_count");
  for (const Record& r : records(b, 13 + 4096 + 8)) {
    if (r.pos >= p.index_pos) {
      const char op = r.fields.at("op").at(0);
      if (op == 0x07) p.connections.push_back(r);
      else if (op == 0x06) p.chunk_infos.push_back(r);
      else FAIL("unexpected record after index_pos");
      continue;
    }
    const char op = r.fields.at("op").at(0);
    if (op == 0x05) {
      const std::string c = r.fields.at("compression");
      const Codec codec = c == "none" ? Codec::kNone : c == "bz2" ? Codec::kBz2 : Codec::kLz4;
      const std::vector<char> raw =
          exporter::decompress(codec, r.data.data(), r.data.size(), field<uint32_t>(r, "size"));
      p.raw.emplace_back(raw.begin(), raw.end());
      p.chunks.push_back(r);
    } else if (op == 0x04) {
      p.index_data.push_back(r);
    } else {
      FAIL("unexpected record before index_pos");
    }
  }
  return p;
}

}  // namespace

TEST_CASE("F1 ROS1: R-FMT's golden dvs_msgs/EventArray and tiny bag, byte for byte") {
  const Columns c = golden_events();
  std::vector<char> msg;
  exporter::ros1::put_event_array(msg, 0, kEpoch, "cam0", 1280, 720, c.view(), exporter::TimeMap{0, kEpoch});
  CHECK(hex(msg) ==
        "000000000078e768000000000400000063616d30d002000000050000070000000a0005000078e768a3010000010b00050000"
        "78e768a3010000010c0005000078e768a3010000017f02df010078e7688b05000000030000000078e768a3fb000001030000"
        "000078e768a3813e0000ff04cf020078e7688b853e0001");
  // R-FMT's tiny.bag: one dvs_msgs/EventArray connection without callerid, that message received at the last event's
  // time, no compression.
  const std::string path = temp_path("tiny.bag");
  exporter::ros1::BagFile bag(path, Codec::kNone, exporter::ros1::kChunkThreshold, 1);
  const uint32_t h = bag.add_connection("/cam0/events", MsgType::kDvsEventArray);
  bag.write(h, kEpoch + 4'097'419, msg.data(), msg.size());
  const exporter::FileInfo info = bag.close();
  CHECK(info.bytes == 7000);
  CHECK(info.sha256 == "cb0d9604dbe347273b3bc8c62729e1ac1227fb0f3e697e0a64d19a95d2ed5de3");
}

TEST_CASE("Bag layout: header, chunks with first-use connections, index data, then connections and chunk infos") {
  const Columns ev = dense_stream(60'000, 3);
  const Imu imu = imu_samples(400, 3'000'000'000, 5);
  const std::string b = write_bag(Codec::kNone, 1, ev, imu, MsgType::kDvsEventArray, 4096, {3'000'000, 1'000'000'000},
                                  64 * 1024);
  const Parsed p = parse(b);
  CHECK(p.chunks.size() > 3);
  CHECK(p.chunk_count == p.chunks.size());
  CHECK(p.chunk_infos.size() == p.chunks.size());
  CHECK(p.conn_count == 2);
  REQUIRE(p.connections.size() == 2);
  // Connection ids in first-use order; the connection header holds md5sum, message_definition, topic and type.
  const Record& c0 = p.connections[0];
  CHECK(field<uint32_t>(c0, "conn") == 0);
  CHECK(c0.fields.at("topic") == "/cam/events");
  CHECK(c0.data.find("type=dvs_msgs/EventArray") != std::string::npos);
  CHECK(c0.data.find("md5sum=5e8beee5a6c107e504c2e78903c224b8") != std::string::npos);
  CHECK(p.connections[1].fields.at("topic") == "/cam/imu");
  CHECK(p.connections[1].data.find("md5sum=6a62c6daae103f4ff57a132d6f95cec2") != std::string::npos);
  // Each connection record appears once in the chunks, in the chunk of its first message; each chunk's index data
  // follows it, one record per connection in id order, offsets pointing at that connection's message records.
  size_t conn_records = 0, messages = 0, ii = 0;
  int64_t prev_time[2] = {-1, -1};
  for (size_t k = 0; k < p.chunks.size(); ++k) {
    const std::vector<Record> rs = records(p.raw[k], 0);
    std::map<uint32_t, std::vector<std::pair<int64_t, uint32_t>>> seen;
    int64_t lo = INT64_MAX, hi = INT64_MIN;
    for (const Record& r : rs) {
      if (r.fields.at("op") == std::string(1, '\x07')) {
        ++conn_records;
        continue;
      }
      REQUIRE(r.fields.at("op") == std::string(1, '\x02'));
      const uint32_t id = field<uint32_t>(r, "conn");
      const int64_t t = int64_t{field<uint32_t>(r, "time")} * 1'000'000'000 + get<uint32_t>(r.fields.at("time"), 4);
      seen[id].emplace_back(t, static_cast<uint32_t>(r.pos));
      lo = std::min(lo, t);
      hi = std::max(hi, t);
      CHECK(t >= prev_time[id]);  // record times in order per connection
      prev_time[id] = t;
      // The record time is header.stamp (seq u32, then the stamp).
      CHECK(get<uint32_t>(r.data, 4) == field<uint32_t>(r, "time"));
      ++messages;
    }
    for (const auto& [id, entries] : seen) {
      REQUIRE(ii < p.index_data.size());
      const Record& ix = p.index_data[ii++];
      CHECK(field<uint32_t>(ix, "conn") == id);
      CHECK(field<uint32_t>(ix, "ver") == 1);
      REQUIRE(field<uint32_t>(ix, "count") == entries.size());
      for (size_t e = 0; e < entries.size(); ++e) {
        const int64_t t = int64_t{get<uint32_t>(ix.data, 12 * e)} * 1'000'000'000 + get<uint32_t>(ix.data, 12 * e + 4);
        CHECK(t == entries[e].first);
        CHECK(get<uint32_t>(ix.data, 12 * e + 8) == entries[e].second);
      }
    }
    const Record& ci = p.chunk_infos[k];
    CHECK(field<uint64_t>(ci, "chunk_pos") == p.chunks[k].pos);
    CHECK(field<uint32_t>(ci, "count") == seen.size());
    const int64_t s = int64_t{field<uint32_t>(ci, "start_time")} * 1'000'000'000 +
                      get<uint32_t>(ci.fields.at("start_time"), 4);
    const int64_t e = int64_t{field<uint32_t>(ci, "end_time")} * 1'000'000'000 +
                      get<uint32_t>(ci.fields.at("end_time"), 4);
    CHECK(s == lo);
    CHECK(e == hi);
    CHECK(p.raw[k].size() > 0);
    if (k + 1 < p.chunks.size()) CHECK(p.raw[k].size() > 64 * 1024);  // closed once past the threshold
  }
  CHECK(ii == p.index_data.size());
  CHECK(conn_records == 2);
  CHECK(messages > 400);
}

TEST_CASE("F7 chunk compression: bz2 and lz4 (the roslz4 frame) give the bytes of none's chunks at 1-8 threads") {
  const Columns ev = dense_stream(120'000, 9);
  const Imu imu = imu_samples(2000, 3'000'000'000, 2);
  const Parsed none = parse(write_bag(Codec::kNone, 1, ev, imu, MsgType::kDvsEventArray, 1 << 17));
  for (const Codec c : {Codec::kBz2, Codec::kLz4}) {
    const std::string one = write_bag(c, 1, ev, imu, MsgType::kDvsEventArray, 1 << 17);
    for (const int threads : {2, 4, 8}) CHECK(write_bag(c, threads, ev, imu, MsgType::kDvsEventArray, 1 << 17) == one);
    const Parsed z = parse(one);
    CHECK(z.raw == none.raw);  // the same records, chunked at the same points
    for (const Record& r : z.chunks) {
      if (c == Codec::kLz4) {
        CHECK(r.fields.at("compression") == "lz4");
        CHECK(static_cast<unsigned char>(r.data[4]) == 0x64);  // FLG and BD as roslz4 requires
        CHECK(static_cast<unsigned char>(r.data[5]) == 0x60);
      } else {
        CHECK(r.fields.at("compression") == "bz2");
        CHECK(r.data.compare(0, 4, "BZh9") == 0);
      }
    }
  }
}

TEST_CASE("F8 streaming: one append and appends of 1, 7, 997 or 2^17 events give the same bytes") {
  const Columns ev = dense_stream(40'000, 11);
  for (const MsgType t : {MsgType::kDvsEventArray, MsgType::kEventPacketEvt3}) {
    const std::string one = write_bag(Codec::kNone, 1, ev, Imu{}, t, 0);
    for (const size_t b : {size_t{1}, size_t{7}, size_t{997}, size_t{1} << 17}) {
      CHECK(write_bag(Codec::kNone, 1, ev, Imu{}, t, b) == one);
    }
  }
}

TEST_CASE("Packets: one message per occupied T1a grid cell, stamped by its first event; EventPacket evt3 payload") {
  // Origin 3 s; packets of 1 ms; events at u = 0, 999 (same packet), 1000, 1000, 5000 (next packets), and t_ns_rem 419.
  Columns c;
  for (const int64_t u : {0, 999, 1000, 1000, 5000, 20'004'096}) c.add(1000 * (3'000'000 + u) + 419, 7, 9, 1);
  const exporter::TimeMap m{3'000'000, 1'000'000'000};
  for (const MsgType t : {MsgType::kDvsEventArray, MsgType::kEventPacketEvt3}) {
    const Parsed p = parse(write_bag(Codec::kNone, 1, c, Imu{}, t, 2, m));
    std::vector<Record> msgs;
    for (const Record& r : records(p.raw.at(0), 0)) {
      if (r.fields.at("op") == std::string(1, '\x02')) msgs.push_back(r);
    }
    REQUIRE(msgs.size() == 4);
    const int64_t first_u[] = {0, 1000, 5000, 20'004'096};
    const size_t sizes[] = {2, 2, 1, 1};
    for (size_t k = 0; k < 4; ++k) {
      const std::string& d = msgs[k].data;
      CHECK(get<uint32_t>(d, 0) == k);  // header.seq: the connection's message index
      const int64_t stamp = 1'000'000'000 + 1000 * first_u[k] + 419;  // T2 of the first event
      CHECK(get<uint32_t>(d, 4) == stamp / 1'000'000'000);
      CHECK(get<uint32_t>(d, 8) == stamp % 1'000'000'000);
      CHECK(get<uint32_t>(d, 12) == 3);
      CHECK(d.compare(16, 3, "cam") == 0);
      CHECK(get<uint32_t>(d, 19) == 720);
      CHECK(get<uint32_t>(d, 23) == 1280);
      if (t == MsgType::kDvsEventArray) {
        CHECK(get<uint32_t>(d, 27) == sizes[k]);
        CHECK(d.size() == 31 + 13 * sizes[k]);
        continue;
      }
      CHECK(get<uint64_t>(d, 27) == k);  // seq
      CHECK(get<uint64_t>(d, 35) == 0);  // time_base
      CHECK(get<uint32_t>(d, 43) == 4);
      CHECK(d.compare(47, 4, "evt3") == 0);
      CHECK(d[51] == 0);  // is_bigendian
      const uint32_t n = get<uint32_t>(d, 52);
      CHECK(d.size() == 56 + n);
      // The words start with the first event's TIME_HIGH and TIME_LOW (u mod 2^24), so the packet decodes alone.
      const auto u0 = static_cast<uint32_t>(first_u[k] & 0xFFFFFF);
      CHECK(get<uint16_t>(d, 56) == (0x8000 | (u0 >> 12)));
      CHECK(get<uint16_t>(d, 58) == (0x6000 | (u0 & 0xFFF)));
    }
  }
}

TEST_CASE("Imu messages: layout, orientation unknown, the covariance modes, float64 bit-exact") {
  std::vector<char> msg;
  const double g[3] = {0.1, -0.0, 5e-324}, a[3] = {0, 0, 9.81};
  std::array<double, 9> cg{}, ca{};
  cg[0] = cg[4] = cg[8] = 1.5e-5;
  ca[0] = ca[4] = ca[8] = 2.5e-3;
  exporter::ros1::put_imu(msg, 3, 5'000'000'417, "cam_imu", g, a, cg, ca);
  const std::string d(msg.begin(), msg.end());
  REQUIRE(d.size() == 4 + 8 + 4 + 7 + 4 * 8 + 9 * 8 + 3 * 8 + 9 * 8 + 3 * 8 + 9 * 8);
  CHECK(get<uint32_t>(d, 0) == 3);
  CHECK(get<uint32_t>(d, 4) == 5);
  CHECK(get<uint32_t>(d, 8) == 417);
  const size_t o = 23;
  for (int i = 0; i < 4; ++i) CHECK(get<double>(d, o + 8 * i) == 0.0);  // orientation zeros
  CHECK(get<double>(d, o + 32) == -1.0);                                // orientation_covariance[0] = -1
  const size_t w = o + 32 + 72;
  CHECK(std::memcmp(d.data() + w, g, 24) == 0);
  CHECK(get<double>(d, w + 24) == 1.5e-5);
  CHECK(get<double>(d, w + 24 + 32) == 1.5e-5);
  CHECK(get<double>(d, w + 24 + 8) == 0.0);
  CHECK(std::memcmp(d.data() + w + 24 + 72, a, 24) == 0);
  CHECK(get<double>(d, w + 24 + 72 + 24 + 64) == 2.5e-3);
  // Through the writer: kZero writes zero covariances; kModel diag(white_var).
  const Imu imu = imu_samples(3, 3'000'000'000, 1);
  for (const auto mode : {exporter::CovarianceMode::kModel, exporter::CovarianceMode::kZero}) {
    const std::string path = temp_path("imu.bag");
    exporter::Rosbag1Writer wr(path, Codec::kNone, 768 * 1024, {3'000'000, 0}, 1);
    const uint32_t c = wr.connection("/cam/imu", MsgType::kImu);
    wr.imu(c, "cam_imu", imu.view(), mode, {4e-6, 9e-4});
    const exporter::FileInfo info = wr.close();
    REQUIRE(info.streams.size() == 1);
    CHECK(info.streams[0].kind == "imu");
    CHECK(info.streams[0].id == "/cam/imu");
    CHECK(info.streams[0].count == 3);
    CHECK(info.streams[0].t_first == imu.t[0] - 3'000'000'000);
    CHECK(info.streams[0].t_last == imu.t[2] - 3'000'000'000);
    const Parsed p = parse(read_file(path));
    const std::vector<Record> rs = records(p.raw.at(0), 0);
    REQUIRE(rs.size() == 4);  // the connection, then the three samples
    const std::string& m = rs[1].data;
    const double var = mode == exporter::CovarianceMode::kModel ? 4e-6 : 0.0;
    CHECK(get<double>(m, w + 24) == var);
    CHECK(get<double>(m, w + 24 + 8) == 0.0);
    CHECK(std::memcmp(m.data() + w, imu.g.data(), 24) == 0);
  }
}

TEST_CASE("Refusals: codec, threads, topics, types, fixed connection settings, order, origin, stamps, after close") {
  using exporter::Refused;
  const std::string path = temp_path("refused.bag");
  CHECK_THROWS_AS(exporter::Rosbag1Writer(path, Codec::kZstd, 1024, {0, 0}, 1), Refused);
  CHECK_THROWS_AS(exporter::Rosbag1Writer(path, Codec::kNone, 1024, {0, 0}, 0), Refused);
  CHECK_THROWS_AS(exporter::Rosbag1Writer(path, Codec::kNone, 1024, {0, 0}, 257), Refused);
  CHECK_THROWS_AS(exporter::Rosbag1Writer(path, Codec::kNone, 1024, {0, 999}, 1), Refused);
  exporter::Rosbag1Writer w(path, Codec::kNone, 1024, {3'000'000, 1'000'000'000}, 1);
  for (const char* bad : {"", "1cam", "/cam events", "/cam-1", "cam\n"}) {
    CHECK_THROWS_AS(w.connection(bad, MsgType::kImu), Refused);
  }
  const uint32_t ce = w.connection("/cam/events", MsgType::kDvsEventArray);
  const uint32_t c3 = w.connection("~cam/evt3", MsgType::kEventPacketEvt3);
  const uint32_t ci = w.connection("cam_imu", MsgType::kImu);
  CHECK_THROWS_AS(w.connection("/cam/events", MsgType::kImu), Refused);  // each topic once
  Columns c;
  for (int i = 0; i < 5; ++i) c.add(3'000'000'000 + 1000 * i, static_cast<uint16_t>(i), 2, 1);
  CHECK_THROWS_AS(w.imu(ce, "cam", Imu{}.view(), exporter::CovarianceMode::kModel, {0, 0}), Refused);  // type
  CHECK_THROWS_AS(w.event_packets_evt3(ce, "cam", 64, 64, c.view(), 1'000'000), Refused);
  CHECK_THROWS_AS(w.event_arrays(ce, "cam", 70'000, 64, c.view(), 1'000'000), Refused);  // u16 coordinates
  CHECK_THROWS_AS(w.event_packets_evt3(c3, "cam", 4096, 64, c.view(), 1'000'000), Refused);  // 11-bit
  CHECK_THROWS_AS(w.event_arrays(ce, "cam", 64, 64, c.view(), 1500), Refused);  // not on the us grid
  CHECK_THROWS_AS(w.event_arrays(ce, "cam", 64, 64, c.view(), 0), Refused);
  // A refused first call leaves the connection unconfigured: a polarity of 2 refuses the batch, and the next call
  // may then fix other settings.
  Columns bad = c;
  bad.p[3] = 2;
  CHECK_THROWS_AS(w.event_arrays(ce, "x", 32, 32, bad.view(), 2'000'000), Refused);
  w.event_arrays(ce, "cam", 64, 64, c.view(0, 2), 1'000'000);
  CHECK_THROWS_AS(w.event_arrays(ce, "cam2", 64, 64, c.view(2), 1'000'000), Refused);  // fixed by the first call
  CHECK_THROWS_AS(w.event_arrays(ce, "cam", 64, 32, c.view(2), 1'000'000), Refused);
  CHECK_THROWS_AS(w.event_arrays(ce, "cam", 64, 64, c.view(2), 2'000'000), Refused);
  CHECK_THROWS_AS(w.event_arrays(ce, "cam", 64, 64, c.view(0, 1), 1'000'000), Refused);  // before the last event
  Columns out = c;
  out.x[4] = 64;
  CHECK_THROWS_AS(w.event_arrays(ce, "cam", 64, 64, out.view(2), 1'000'000), Refused);  // outside the sensor
  w.event_arrays(ce, "cam", 64, 64, c.view(2), 1'000'000);
  Columns early;
  early.add(2'999'999'000, 1, 1, 1);  // before the origin
  CHECK_THROWS_AS(w.event_packets_evt3(c3, "cam", 64, 64, early.view(), 1'000'000), Refused);
  Imu imu = imu_samples(3, 3'000'000'000, 4);
  CHECK_THROWS_AS(w.imu(ci, "cam_imu", imu.view(), exporter::CovarianceMode::kModel, {-1, 0}), Refused);
  CHECK_THROWS_AS(w.imu(ci, "cam_imu", imu.view(), exporter::CovarianceMode::kModel, {0, std::nan("")}), Refused);
  std::swap(imu.t[0], imu.t[1]);
  CHECK_THROWS_AS(w.imu(ci, "cam_imu", imu.view(), exporter::CovarianceMode::kModel, {0, 0}), Refused);
  const exporter::FileInfo info = w.close();
  REQUIRE(info.streams.size() == 3);
  CHECK(info.streams[0].count == 5);
  CHECK(info.streams[0].t_first == 1'000'000'000);  // T2: t - 1000 origin_us + epoch_ns
  CHECK(info.streams[0].t_last == 1'000'004'000);
  CHECK(info.streams[1].count == 0);
  CHECK_THROWS_AS(w.event_arrays(ce, "cam", 64, 64, c.view(), 1'000'000), Refused);  // after close
  CHECK_THROWS_AS(w.close(), Refused);
  const Parsed p = parse(read_file(path));
  CHECK(p.conn_count == 1);  // the unused connections stay out of the bag
  // ROS1 time is u32 seconds: a stamp below 0 or at 2^32 s is refused, naming the event; nothing of it is written.
  exporter::Rosbag1Writer neg(temp_path("neg.bag"), Codec::kNone, 1024, {3'000'000, -1'000'000'000}, 1);
  const uint32_t cn = neg.connection("/n", MsgType::kDvsEventArray);
  Columns one;
  one.add(3'000'000'000, 1, 1, 1);
  CHECK_THROWS_WITH_AS(neg.event_arrays(cn, "n", 8, 8, one.view(), 1'000'000),
                       doctest::Contains("outside ROS1's [0, 2^32 s)"), Refused);
  const int64_t last_epoch = (int64_t{1} << 32) * 1'000'000'000 - 1000;
  exporter::Rosbag1Writer late(temp_path("late.bag"), Codec::kNone, 1024, {0, last_epoch}, 1);
  const uint32_t cl = late.connection("/l", MsgType::kImu);
  Imu at = imu_samples(2, 0, 1);  // the second sample lies past 2^32 s
  CHECK_THROWS_AS(late.imu(cl, "l", at.view(), exporter::CovarianceMode::kZero, {0, 0}), Refused);
  CHECK(late.close().streams.at(0).count == 0);
  CHECK(exporter::ros1_msgdef(MsgType::kImu).type == "sensor_msgs/Imu");
  CHECK(exporter::ros1_msgdef(MsgType::kImage).type == "sensor_msgs/Image");  // since A9
}

TEST_CASE("Packet limits: at most 2^24 us; a packet whose record could pass 4 GiB is refused before it is held") {
  using exporter::Refused;
  constexpr uint64_t kU32Max = 0xFFFFFFFFu;
  constexpr int64_t kMaxPacketNs = 1000 * exporter::Rosbag1Writer::kMaxPacketUs;
  static_assert(exporter::Rosbag1Writer::kMaxPacketUs == int64_t{1} << 24);  // EventPacket evt3 carries u mod 2^24
  {
    exporter::Rosbag1Writer w(temp_path("long.bag"), Codec::kNone, 1024, {0, 0}, 1);
    const uint32_t ce = w.connection("/e", MsgType::kDvsEventArray);
    const uint32_t c3 = w.connection("/p", MsgType::kEventPacketEvt3);
    Columns c;
    c.add(1000, 1, 1, 1);
    CHECK_THROWS_WITH_AS(w.event_arrays(ce, "e", 8, 8, c.view(), kMaxPacketNs + 1000), doctest::Contains("2^24"),
                         Refused);
    CHECK_THROWS_AS(w.event_packets_evt3(c3, "p", 8, 8, c.view(), kMaxPacketNs + 1000), Refused);
    w.event_arrays(ce, "e", 8, 8, c.view(), kMaxPacketNs);
    w.event_packets_evt3(c3, "p", 8, 8, c.view(), kMaxPacketNs);
    const exporter::FileInfo info = w.close();
    CHECK(info.streams.at(0).count == 1);
    CHECK(info.streams.at(1).count == 1);
  }
  // The last 2^24 us cell of int64 ends past INT64_MAX: its end saturates, so t_ns = INT64_MAX stays in it.
  {
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    const int64_t start = kMaxPacketNs * (kMax / kMaxPacketNs);  // the last cell (origin 0)
    REQUIRE(start / 1000 + exporter::Rosbag1Writer::kMaxPacketUs > kMax / 1000);
    const std::string path = temp_path("end.bag");
    exporter::Rosbag1Writer w(path, Codec::kNone, 1 << 20, {0, 1'000'000'000 - start}, 1);  // stamps from 1 s
    const uint32_t ce = w.connection("/e", MsgType::kDvsEventArray);
    Columns c;
    for (const int64_t t : {start - 1000, start, kMax - 1, kMax}) c.add(t, 1, 1, 1);
    w.event_arrays(ce, "e", 8, 8, c.view(0, 2), kMaxPacketNs);
    w.event_arrays(ce, "e", 8, 8, c.view(2), kMaxPacketNs);
    CHECK(w.close().streams.at(0).t_last == kMax - start + 1'000'000'000);
    const Parsed p = parse(read_file(path));
    const std::vector<Record> recs = records(p.raw.at(0), 0);
    REQUIRE(recs.size() == 3);  // the connection, then one message per cell
    const size_t count_at = 4 + 8 + (4 + 1) + 4 + 4;  // the array length after header("e"), height, width
    CHECK(get<uint32_t>(recs[1].data, count_at) == 1);
    CHECK(get<uint32_t>(recs[2].data, count_at) == 3);
  }
  // A chunk threshold 6000 bytes short of 2^32 - 1 leaves a few hundred events of room in a message (rosbag's 768 KiB
  // leaves 4 GiB less it and the connection record): one event per call, each at its own us and row (evt3's worst
  // case of 3 words), fills the 1 s packet up to its capacity; the call past it is refused whole, holding nothing,
  // and so is a batch whose first event is past it, even though its second starts the next packet.
  for (const MsgType type : {MsgType::kDvsEventArray, MsgType::kEventPacketEvt3}) {
    CAPTURE(static_cast<int>(type));
    const bool dvs = type == MsgType::kDvsEventArray;
    const auto threshold = static_cast<uint32_t>(kU32Max - 6000);
    const int64_t packet_ns = 1'000'000'000;
    const std::string path = temp_path("full.bag");
    exporter::Rosbag1Writer w(path, Codec::kNone, threshold, {0, 0}, 1);
    const uint32_t ce = w.connection("/cam/events", type);
    auto send = [&](const Columns& c) {
      if (dvs) w.event_arrays(ce, "cam", 64, 64, c.view(), packet_ns);
      else w.event_packets_evt3(ce, "cam", 64, 64, c.view(), packet_ns);
    };
    size_t accepted = 0;
    std::string refusal;
    for (; accepted < 100'000; ++accepted) {
      Columns one;
      one.add(1000 * static_cast<int64_t>(accepted), 1, static_cast<uint16_t>(accepted % 2), 1);
      try {
        send(one);
      } catch (const Refused& e) {
        refusal = e.what();
        break;
      }
    }
    CHECK(refusal.find("use shorter packets (--packet-us)") != std::string::npos);
    Columns both;
    both.add(1000 * static_cast<int64_t>(accepted), 1, 0, 1);
    both.add(packet_ns, 2, 3, 0);  // the next packet
    CHECK_THROWS_AS(send(both), Refused);
    Columns next;
    next.add(packet_ns, 2, 3, 0);
    send(next);
    const exporter::FileInfo info = w.close();
    CHECK(info.streams.at(0).count == static_cast<int64_t>(accepted) + 1);
    // The capacity is the room write() guarantees (2^32 - 1 less the threshold, the connection record and the
    // message record's 46 header bytes) less the message's fixed bytes, over the most bytes an event can take.
    const Parsed p = parse(read_file(path));
    REQUIRE(p.raw.size() == 1);
    const std::vector<Record> recs = records(p.raw[0], 0);
    REQUIRE(recs.size() == 3);  // the connection, then the two messages
    CHECK(recs[0].fields.at("op") == std::string(1, '\x07'));
    const uint64_t conn_record = recs[1].pos - recs[0].pos;
    const uint64_t room = kU32Max - threshold - conn_record - 46;
    const uint64_t header = 4 + 8 + 4 + 3;  // seq, stamp, "cam"
    const uint64_t fixed = header + (dvs ? 12 : 37 + 2 * static_cast<uint64_t>(packet_ns / 1000 / 4096 + 2));
    CHECK(accepted == (room - fixed) / (dvs ? 13 : 6));
    CHECK(recs[1].data.size() <= room);  // it fits whatever the open chunk held
    if (dvs) {
      CHECK(recs[1].data.size() == fixed + 13 * accepted);
      CHECK(get<uint32_t>(recs[1].data, header + 8) == accepted);  // the array length
      CHECK(recs[1].data.size() + 13 > room);                       // one more event might not have fit
    } else {
      CHECK(recs[1].data.size() == header + 37 + 2 * (3 * accepted + 1));  // 3 words per event, one TIME_HIGH
    }
    CHECK(get<uint32_t>(recs[2].data, 0) == 1);  // header.seq of the next packet's message
  }
}
