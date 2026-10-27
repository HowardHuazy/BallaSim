// ROS1 serializers, the bag container and the rosbag1 writer (ros1.h; docs/design/phase3/outputs.md §3.7, §4.7-4.8).
#include "export/ros1.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "export/chunk_pool.h"
#include "export/evt.h"
#include "export/ros_streams.h"

namespace balla::exporter {

static_assert(std::endian::native == std::endian::little, "ROS1 serialization is little-endian, written in host order");

namespace ros1 {

namespace {

constexpr std::string_view kMagic = "#ROSBAG V2.0\n";
constexpr char kOpMessage = 0x02, kOpFileHeader = 0x03, kOpIndex = 0x04, kOpChunk = 0x05, kOpChunkInfo = 0x06,
               kOpConnection = 0x07;
constexpr uint32_t kIndexVersion = 1, kChunkInfoVersion = 1;
constexpr uint64_t kU32Max = std::numeric_limits<uint32_t>::max();
// A message record's bytes besides its data: the header length, the conn, op and time fields, the data length.
constexpr uint64_t kMessageRecordOverhead = 4 + (4 + 5 + 4) + (4 + 3 + 1) + (4 + 5 + 8) + 4;

template <class T>
void put(std::vector<char>& out, T v) {
  const size_t at = out.size();
  out.resize(at + sizeof(T));
  std::memcpy(out.data() + at, &v, sizeof(T));
}

void put_bytes(std::vector<char>& out, const void* p, size_t n) {
  const auto* c = static_cast<const char*>(p);
  out.insert(out.end(), c, c + n);
}

void put_string(std::vector<char>& out, std::string_view s) {
  put<uint32_t>(out, static_cast<uint32_t>(s.size()));
  put_bytes(out, s.data(), s.size());
}

void put_time(std::vector<char>& out, int64_t ns) {
  const Time t = time_of(ns);
  put(out, t.sec);
  put(out, t.nsec);
}

// A field value's bytes: a little-endian integer or a ROS1 time.
template <class T>
std::string bytes_of(T v) {
  std::string s(sizeof(T), '\0');
  std::memcpy(s.data(), &v, sizeof(T));
  return s;
}
std::string time_bytes(int64_t ns) {
  const Time t = time_of(ns);
  return bytes_of(t.sec) + bytes_of(t.nsec);
}

struct Field {
  std::string_view name, value;
};

// A record header (ros::Header): u32 length, then u32-length-prefixed "name=value" fields, in name order.
void put_fields(std::vector<char>& out, std::initializer_list<Field> fields) {
  size_t total = 0;
  std::string_view prev;
  for (const Field& f : fields) {
    if (!prev.empty() && !(prev < f.name)) throw std::logic_error("ros1: record header fields out of name order");
    prev = f.name;
    total += 4 + f.name.size() + 1 + f.value.size();
  }
  put<uint32_t>(out, static_cast<uint32_t>(total));
  for (const Field& f : fields) {
    put<uint32_t>(out, static_cast<uint32_t>(f.name.size() + 1 + f.value.size()));
    put_bytes(out, f.name.data(), f.name.size());
    out.push_back('=');
    put_bytes(out, f.value.data(), f.value.size());
  }
}

void put_record(std::vector<char>& out, std::initializer_list<Field> fields, const char* data, size_t n) {
  put_fields(out, fields);
  put<uint32_t>(out, static_cast<uint32_t>(n));
  put_bytes(out, data, n);
}

// The bag header record (op 0x03), padded with spaces so that its fields and data make kFileHeaderLength bytes; its
// size does not depend on the values, so close() rewrites it in place.
std::vector<char> file_header(uint64_t index_pos, uint32_t conn_count, uint32_t chunk_count) {
  std::vector<char> out;
  const char op = kOpFileHeader;
  put_fields(out, {{"chunk_count", bytes_of(chunk_count)},
                   {"conn_count", bytes_of(conn_count)},
                   {"index_pos", bytes_of(index_pos)},
                   {"op", std::string_view(&op, 1)}});
  const size_t header_len = out.size() - 4;
  const size_t pad = header_len < kFileHeaderLength ? kFileHeaderLength - header_len : 0;
  put<uint32_t>(out, static_cast<uint32_t>(pad));
  out.insert(out.end(), pad, ' ');
  return out;
}

}  // namespace

void put_header(std::vector<char>& out, uint32_t seq, int64_t stamp_ns, std::string_view frame_id) {
  put(out, seq);
  put_time(out, stamp_ns);
  put_string(out, frame_id);
}

void put_event_array(std::vector<char>& out, uint32_t seq, int64_t stamp_ns, std::string_view frame_id, uint32_t width,
                     uint32_t height, const Events& e, const TimeMap& m) {
  put_header(out, seq, stamp_ns, frame_id);
  put(out, height);
  put(out, width);
  put<uint32_t>(out, static_cast<uint32_t>(e.n));
  // T2 = (t - 1000 origin_us) + epoch_ns, in t2_ns's order: the writer checked the first and last stamps, which
  // bound every partial sum between them (check(m) bounds 1000 origin_us).
  const int64_t origin_ns = 1000 * m.origin_us;
  const size_t at = out.size();
  out.resize(at + 13 * e.n);
  char* w = out.data() + at;
  for (size_t i = 0; i < e.n; ++i, w += 13) {
    const Time t = time_of((e.t_ns[i] - origin_ns) + m.epoch_ns);
    const uint8_t p = e.p[i];
    std::memcpy(w, &e.x[i], 2);
    std::memcpy(w + 2, &e.y[i], 2);
    std::memcpy(w + 4, &t.sec, 4);
    std::memcpy(w + 8, &t.nsec, 4);
    std::memcpy(w + 12, &p, 1);
  }
}

void put_event_packet_evt3(std::vector<char>& out, uint64_t seq, int64_t stamp_ns, std::string_view frame_id,
                           uint32_t width, uint32_t height, const Events& e, const TimeMap& m) {
  std::vector<uint16_t> words;
  words.reserve(e.n + e.n / 2 + 8);
  Evt3Encoder enc(Evt3Encoder::Start::kFirstEvent);
  for (size_t i = 0; i < e.n; ++i) enc.event(t1a_us(m, e.t_ns[i]), e.x[i], e.y[i], e.p[i], words);
  put_header(out, static_cast<uint32_t>(seq), stamp_ns, frame_id);
  put(out, height);
  put(out, width);
  put(out, seq);
  put<uint64_t>(out, 0);  // time_base: unused for evt3 (§4.7)
  put_string(out, "evt3");
  put<uint8_t>(out, 0);  // is_bigendian
  put<uint32_t>(out, static_cast<uint32_t>(2 * words.size()));
  put_bytes(out, words.data(), 2 * words.size());
}

void put_image(std::vector<char>& out, uint32_t seq, int64_t stamp_ns, std::string_view frame_id,
               const ros::Image& im) {
  put_header(out, seq, stamp_ns, frame_id);
  put(out, im.height);
  put(out, im.width);
  put_string(out, im.encoding);
  put<uint8_t>(out, 0);  // is_bigendian
  put(out, im.step);
  put<uint32_t>(out, static_cast<uint32_t>(im.n));
  put_bytes(out, im.data, im.n);
}

void put_imu(std::vector<char>& out, uint32_t seq, int64_t stamp_ns, std::string_view frame_id, const double* gyro,
             const double* accel, const std::array<double, 9>& cov_gyro, const std::array<double, 9>& cov_accel) {
  put_header(out, seq, stamp_ns, frame_id);
  const double orientation[4] = {0, 0, 0, 0};
  const std::array<double, 9> no_orientation = {-1, 0, 0, 0, 0, 0, 0, 0, 0};
  put_bytes(out, orientation, sizeof orientation);
  put_bytes(out, no_orientation.data(), 9 * sizeof(double));
  put_bytes(out, gyro, 3 * sizeof(double));
  put_bytes(out, cov_gyro.data(), 9 * sizeof(double));
  put_bytes(out, accel, 3 * sizeof(double));
  put_bytes(out, cov_accel.data(), 9 * sizeof(double));
}

struct BagFile::Impl {
  struct Connection {
    std::string topic;
    MsgType type;
    int64_t id = -1;  // assigned at the first message
  };
  struct Entry {
    int64_t time_ns;
    uint32_t offset;
  };
  struct Chunk {  // what a chunk's emit needs besides its bytes
    int64_t start_ns = 0, end_ns = 0;
    std::map<uint32_t, std::vector<Entry>> index;  // by connection id, as rosbag's std::map
    size_t raw_size = 0;
  };
  struct ChunkInfo {
    uint64_t pos;
    int64_t start_ns, end_ns;
    std::vector<std::pair<uint32_t, uint32_t>> counts;
  };

  Impl(const std::string& path, Codec c, uint32_t chunk_threshold, int threads)
      : codec(c), threshold(chunk_threshold), sink(path), pool(c, threads, Lz4Frame::kRoslz4) {
    sink.write(kMagic.data(), kMagic.size());
    const std::vector<char> h = file_header(0, 0, 0);
    sink.write(h.data(), h.size());
  }

  // The connection record (op 0x07): {conn, op, topic}, then the connection header.
  std::vector<char> connection_record(const Connection& c) const {
    const Ros1MsgDef& d = ros1_msgdef(c.type);
    std::vector<char> data;
    put_fields(data, {{"md5sum", d.md5}, {"message_definition", d.definition}, {"topic", c.topic}, {"type", d.type}});
    data.erase(data.begin(), data.begin() + 4);  // the connection header is the record's data: fields only
    std::vector<char> out;
    const char op = kOpConnection;
    put_record(out,
               {{"conn", bytes_of(static_cast<uint32_t>(c.id))}, {"op", std::string_view(&op, 1)}, {"topic", c.topic}},
               data.data(), data.size());
    return out;
  }

  void submit_chunk() {
    if (buf.empty()) return;
    open.raw_size = buf.size();
    auto meta = std::make_shared<Chunk>(std::move(open));
    open = Chunk{};
    std::vector<char> raw = std::move(buf);
    buf = {};
    pool.submit(std::move(raw), [this, meta](std::vector<char>&& z) { emit_chunk(*meta, z); });
  }

  // On the calling thread, in chunk order: the chunk record, then its index data records.
  void emit_chunk(const Chunk& c, const std::vector<char>& z) {
    if (z.size() > kU32Max) throw Refused("rosbag1: a compressed chunk exceeds 4 GiB");
    ChunkInfo info{sink.tell(), c.start_ns, c.end_ns, {}};
    std::vector<char> out;
    const char op = kOpChunk;
    put_fields(out, {{"compression", codec_name(codec)},
                     {"op", std::string_view(&op, 1)},
                     {"size", bytes_of(static_cast<uint32_t>(c.raw_size))}});
    put<uint32_t>(out, static_cast<uint32_t>(z.size()));
    sink.write(out.data(), out.size());
    sink.write(z.data(), z.size());
    for (const auto& [id, entries] : c.index) {
      std::vector<char> data;
      data.reserve(12 * entries.size());
      for (const Entry& e : entries) {
        put_time(data, e.time_ns);
        put(data, e.offset);
      }
      out.clear();
      const char op_index = kOpIndex;
      put_record(out,
                 {{"conn", bytes_of(id)},
                  {"count", bytes_of(static_cast<uint32_t>(entries.size()))},
                  {"op", std::string_view(&op_index, 1)},
                  {"ver", bytes_of(kIndexVersion)}},
                 data.data(), data.size());
      sink.write(out.data(), out.size());
      info.counts.emplace_back(id, static_cast<uint32_t>(entries.size()));
    }
    infos.push_back(std::move(info));
  }

  Codec codec;
  uint32_t threshold;
  ByteSink sink;
  ChunkPool pool;  // after the sink: destroyed first, so no emit outlives it
  std::vector<Connection> conns;
  std::vector<uint32_t> by_id;  // handles in id order
  std::vector<char> buf;        // the open chunk's records
  Chunk open;
  std::vector<ChunkInfo> infos;
  bool closed = false;
};

BagFile::BagFile(const std::string& path, Codec c, uint32_t chunk_threshold, int threads) {
  if (c != Codec::kNone && c != Codec::kBz2 && c != Codec::kLz4) {
    throw Refused("rosbag1: chunk compression must be none, bz2 or lz4, got " + std::string(codec_name(c)));
  }
  if (threads < 1 || threads > ChunkPool::kMaxThreads) {  // before the file is created
    throw Refused("threads must be in [1, " + std::to_string(ChunkPool::kMaxThreads) + "], got " +
                  std::to_string(threads));
  }
  impl_ = std::make_unique<Impl>(path, c, chunk_threshold, threads);
}

BagFile::~BagFile() = default;

uint32_t BagFile::add_connection(const std::string& topic, MsgType type) {
  Impl& s = *impl_;
  if (s.closed) throw Refused("rosbag1: connection after close");
  (void)ros1_msgdef(type);  // refuses the types this version does not write
  s.conns.push_back({topic, type, -1});
  return static_cast<uint32_t>(s.conns.size() - 1);
}

void BagFile::write(uint32_t handle, int64_t time_ns, const char* data, size_t n) {
  Impl& s = *impl_;
  if (s.closed) throw Refused("rosbag1: write after close");
  if (handle >= s.conns.size()) throw std::out_of_range("rosbag1: unknown connection handle");
  if (!in_range(time_ns)) {
    throw Refused("rosbag1: time " + std::to_string(time_ns) + " ns is outside ROS1's [0, 2^32 s)");
  }
  Impl::Connection& c = s.conns[handle];
  std::vector<char> conn_record;
  if (c.id < 0) {
    c.id = static_cast<int64_t>(s.by_id.size());
    conn_record = s.connection_record(c);
  }
  const uint64_t record = kMessageRecordOverhead + uint64_t{n};
  if (n > kU32Max || s.buf.size() + conn_record.size() + record > kU32Max) {
    if (conn_record.size() > 0) c.id = -1;
    throw Refused("rosbag1: a message of " + std::to_string(n) + " bytes would take its chunk past 4 GiB (u32 sizes "
                  "and offsets): use shorter packets (--packet-us)");
  }
  if (!conn_record.empty()) {
    s.by_id.push_back(handle);
    s.buf.insert(s.buf.end(), conn_record.begin(), conn_record.end());
  }
  const auto id = static_cast<uint32_t>(c.id);
  if (s.open.index.empty()) {
    s.open.start_ns = s.open.end_ns = time_ns;
  } else {
    s.open.start_ns = std::min(s.open.start_ns, time_ns);
    s.open.end_ns = std::max(s.open.end_ns, time_ns);
  }
  s.open.index[id].push_back({time_ns, static_cast<uint32_t>(s.buf.size())});
  const char op = kOpMessage;
  put_record(s.buf, {{"conn", bytes_of(id)}, {"op", std::string_view(&op, 1)}, {"time", time_bytes(time_ns)}}, data,
             n);
  if (s.buf.size() > s.threshold) s.submit_chunk();
}

uint64_t BagFile::max_message(uint32_t handle) const {
  const Impl& s = *impl_;
  if (handle >= s.conns.size()) throw std::out_of_range("rosbag1: unknown connection handle");
  // write() is entered with at most the threshold in the open chunk (it submits the chunk once past it).
  const uint64_t used =
      uint64_t{s.threshold} + s.connection_record(s.conns[handle]).size() + kMessageRecordOverhead;
  return used < kU32Max ? kU32Max - used : 0;
}

FileInfo BagFile::close() {
  Impl& s = *impl_;
  if (s.closed) throw Refused("rosbag1: closed twice");
  s.closed = true;
  s.submit_chunk();
  s.pool.drain();
  const uint64_t index_pos = s.sink.tell();
  for (const uint32_t h : s.by_id) {
    const std::vector<char> r = s.connection_record(s.conns[h]);
    s.sink.write(r.data(), r.size());
  }
  for (const Impl::ChunkInfo& info : s.infos) {
    std::vector<char> data, out;
    for (const auto& [id, count] : info.counts) {
      put(data, id);
      put(data, count);
    }
    const char op = kOpChunkInfo;
    put_record(out,
               {{"chunk_pos", bytes_of(info.pos)},
                {"count", bytes_of(static_cast<uint32_t>(info.counts.size()))},
                {"end_time", time_bytes(info.end_ns)},
                {"op", std::string_view(&op, 1)},
                {"start_time", time_bytes(info.start_ns)},
                {"ver", bytes_of(kChunkInfoVersion)}},
               data.data(), data.size());
    s.sink.write(out.data(), out.size());
  }
  const std::vector<char> h =
      file_header(index_pos, static_cast<uint32_t>(s.by_id.size()), static_cast<uint32_t>(s.infos.size()));
  s.sink.patch(kMagic.size(), h.data(), h.size());
  return s.sink.close();
}

}  // namespace ros1

namespace {

// ROS's graph resource name rule (wiki.ros.org/Names): a letter, '/' or '~' first, then letters, digits, '_' or '/'.
bool ros_name(std::string_view s) {
  if (s.empty()) return false;
  auto alpha = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
  if (!alpha(s[0]) && s[0] != '/' && s[0] != '~') return false;
  return std::all_of(s.begin() + 1, s.end(),
                     [&](char c) { return alpha(c) || (c >= '0' && c <= '9') || c == '_' || c == '/'; });
}

// The rosbag1 side of the ROS writers' streams (ros_streams.h): ROS1 serialization into one bag.
struct Ros1Format {
  static constexpr std::string_view kName = "rosbag1", kStream = "connection";
  static constexpr std::string_view kRange = "ROS1's [0, 2^32 s) (u32 seconds)";
  static constexpr std::string_view kCapacity = "which could take its record past 4 GiB (u32 sizes and offsets)";
  static bool in_range(int64_t ns) { return ros1::in_range(ns); }
  static std::string_view type_name(MsgType t) { return ros1_msgdef(t).type; }

  // The most events one packet may hold: its message at the most bytes they can need (EventArray: 13 per event;
  // EventPacket evt3: 6 per event and the packet's TIME_HIGH words) within what the bag accepts whatever its open
  // chunk holds, so a held packet always fits its record.
  uint64_t packet_capacity(const ros::Stream& s) const {
    const uint64_t header = 4 + 8 + 4 + s.frame_id.size();  // std_msgs/Header: seq, stamp, frame_id
    const bool dvs = s.type == MsgType::kDvsEventArray;
    // EventArray: height, width, the array length. EventPacket: height, width, seq, time_base, encoding "evt3",
    // is_bigendian, the payload length, and the TIME_HIGH words.
    const uint64_t fixed = header + (dvs ? 12 : 37 + ros::evt3_time_high_bytes(s.packet_us));
    const uint64_t room = bag.max_message(s.handle);
    return room > fixed ? (room - fixed) / (dvs ? 13 : 6) : 0;
  }

  void write_events(const ros::Stream& s, int64_t at, const Events& e, const TimeMap& m) {
    msg.clear();
    if (s.type == MsgType::kDvsEventArray) {
      ros1::put_event_array(msg, static_cast<uint32_t>(s.messages), at, s.frame_id, static_cast<uint32_t>(s.width),
                            static_cast<uint32_t>(s.height), e, m);
    } else {
      ros1::put_event_packet_evt3(msg, s.messages, at, s.frame_id, static_cast<uint32_t>(s.width),
                                  static_cast<uint32_t>(s.height), e, m);
    }
    bag.write(s.handle, at, msg.data(), msg.size());
  }

  void write_imu(const ros::Stream& s, int64_t at, const double* gyro, const double* accel,
                 const std::array<double, 9>& cov_g, const std::array<double, 9>& cov_a) {
    msg.clear();
    ros1::put_imu(msg, static_cast<uint32_t>(s.messages), at, s.frame_id, gyro, accel, cov_g, cov_a);
    bag.write(s.handle, at, msg.data(), msg.size());
  }

  // The most data bytes an Image may hold within what the bag accepts: its header (seq, stamp, frame_id), height,
  // width, encoding, is_bigendian, step and the data length besides.
  uint64_t image_capacity(const ros::Stream& s, size_t encoding_bytes) const {
    const uint64_t fixed = (4 + 8 + 4 + s.frame_id.size()) + 8 + (4 + encoding_bytes) + 1 + 4 + 4;
    const uint64_t room = bag.max_message(s.handle);
    return room > fixed ? room - fixed : 0;
  }

  void write_image(const ros::Stream& s, int64_t at, const ros::Image& im) {
    msg.clear();
    ros1::put_image(msg, static_cast<uint32_t>(s.messages), at, s.frame_id, im);
    bag.write(s.handle, at, msg.data(), msg.size());
  }

  ros1::BagFile& bag;
  std::vector<char> msg;  // one message's bytes, reused
};

}  // namespace

struct Rosbag1Writer::Impl {
  Impl(const std::string& path, Codec c, uint32_t chunk_bytes, const TimeMap& m, int threads)
      : bag(path, c, chunk_bytes, threads), format{bag, {}}, streams(m, format) {}

  ros1::BagFile bag;
  Ros1Format format;
  ros::Streams<Ros1Format> streams;
};

Rosbag1Writer::Rosbag1Writer(const std::string& path, Codec c, uint32_t chunk_bytes, TimeMap m, int threads) {
  check(m);
  impl_ = std::make_unique<Impl>(path, c, chunk_bytes, m, threads);
}

Rosbag1Writer::~Rosbag1Writer() = default;

uint32_t Rosbag1Writer::connection(const std::string& topic, MsgType type) {
  Impl& s = *impl_;
  if (s.streams.closed()) throw Refused("rosbag1: connection after close");
  if (type != MsgType::kDvsEventArray && type != MsgType::kEventPacketEvt3 && type != MsgType::kImu &&
      type != MsgType::kImage) {
    throw Refused("rosbag1: unknown message type");
  }
  if (!ros_name(topic)) {
    throw Refused("rosbag1: topic '" + topic + "' breaks ROS's name rule (a letter, '/' or '~' first, then letters, "
                  "digits, '_' or '/'): name it with --topic STREAM:CAMERA=NAME");
  }
  s.streams.check_new(topic);
  return s.streams.add(topic, type, s.bag.add_connection(topic, type));
}

void Rosbag1Writer::event_arrays(uint32_t conn, const std::string& frame_id, int32_t w, int32_t h, const Events& e,
                                 int64_t packet_ns) {
  impl_->streams.events(conn, MsgType::kDvsEventArray, frame_id, w, h, e, packet_ns);
}

void Rosbag1Writer::event_packets_evt3(uint32_t conn, const std::string& frame_id, int32_t w, int32_t h,
                                       const Events& e, int64_t packet_ns) {
  impl_->streams.events(conn, MsgType::kEventPacketEvt3, frame_id, w, h, e, packet_ns);
}

void Rosbag1Writer::imu(uint32_t conn, const std::string& frame_id, const ImuView& v, CovarianceMode mode,
                        std::array<double, 2> white_var) {
  impl_->streams.imu(conn, frame_id, v, mode, white_var);
}

void Rosbag1Writer::images(uint32_t conn, const std::string& frame_id, const FrameView& v, const FrameOptions& o) {
  impl_->streams.images(conn, frame_id, v, o);
}

FileInfo Rosbag1Writer::close() {
  Impl& s = *impl_;
  std::vector<StreamInfo> streams = s.streams.close();  // writes the held packets
  FileInfo info = s.bag.close();
  info.streams = std::move(streams);
  return info;
}

}  // namespace balla::exporter
