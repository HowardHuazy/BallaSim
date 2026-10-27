// The MCAP container and writer (mcap.h; docs/design/phase3/outputs.md §3.7, §4.7-4.8).
#include "export/mcap.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "export/cdr.h"
#include "export/chunk_pool.h"
#include "export/crc32.h"

namespace balla::exporter {

static_assert(std::endian::native == std::endian::little, "MCAP records are little-endian, written in host order");

namespace mcap {

namespace {

constexpr uint64_t kU16Ids = std::numeric_limits<uint16_t>::max();  // ids 1..65535 (schema id 0 means none)

// Record content: little-endian fields, strings u32-length-prefixed.
struct Bytes {
  std::vector<char> b;
  template <class T>
  Bytes& put(T v) {
    const size_t at = b.size();
    b.resize(at + sizeof(T));
    std::memcpy(b.data() + at, &v, sizeof(T));
    return *this;
  }
  Bytes& raw(const void* p, size_t n) {
    const auto* c = static_cast<const char*>(p);
    b.insert(b.end(), c, c + n);
    return *this;
  }
  Bytes& str(std::string_view s) {
    put(static_cast<uint32_t>(s.size()));
    return raw(s.data(), s.size());
  }
};

// A record: opcode, u64 content length, content.
std::vector<char> record(Op op, const std::vector<char>& content) {
  Bytes r;
  r.put(static_cast<uint8_t>(op)).put(static_cast<uint64_t>(content.size())).raw(content.data(), content.size());
  return std::move(r.b);
}

std::string_view compression_of(Codec c) {
  switch (c) {
    case Codec::kNone: return "";
    case Codec::kLz4: return "lz4";
    case Codec::kZstd: return "zstd";
    default: break;
  }
  throw Refused("mcap: chunk compression must be none, lz4 or zstd, got " + std::string(codec_name(c)));
}

}  // namespace

struct McapFile::Impl {
  struct Schema {
    std::string name, encoding, data;
    uint16_t id = 0;  // 0 until written
  };
  struct Channel {
    uint32_t schema;
    std::string topic, encoding;
    std::vector<std::pair<std::string, std::string>> metadata;  // key order
    uint16_t id = 0;  // 0 until written
    uint64_t count = 0;
  };
  struct Entry {
    uint64_t time, offset;
  };
  struct Chunk {  // what a chunk's emit needs besides its bytes
    uint64_t start = 0, end = 0, raw_size = 0;
    uint32_t crc = 0;
    std::map<uint16_t, std::vector<Entry>> index;  // by channel id
  };

  Impl(const std::string& path, Codec c, uint64_t chunk_bytes, int threads, std::string_view profile,
       std::string_view library)
      : compression(compression_of(c)), threshold(chunk_bytes), sink(path), pool(c, threads) {
    data(kMagic.data(), kMagic.size());
    Bytes h;
    h.str(profile).str(library);
    data(record(kHeader, h.b));
  }

  // Bytes of the data section (from the magic on), with their running CRC-32.
  void data(const void* p, size_t n) {
    sink.write(p, n);
    data_crc = crc32(data_crc, p, n);
  }
  void data(const std::vector<char>& v) { data(v.data(), v.size()); }

  std::vector<char> schema_record(const Schema& s) const {
    Bytes c;
    c.put(s.id).str(s.name).str(s.encoding).str(s.data);
    return record(kSchema, c.b);
  }

  std::vector<char> channel_record(const Channel& ch) const {
    Bytes m;
    for (const auto& [k, v] : ch.metadata) m.str(k).str(v);
    Bytes c;
    c.put(ch.id).put(schemas[ch.schema].id).str(ch.topic).str(ch.encoding);
    c.put(static_cast<uint32_t>(m.b.size())).raw(m.b.data(), m.b.size());
    return record(kChannel, c.b);
  }

  // A channel's first message: its schema's record (unless written) and its own, outside the chunks, after the chunks
  // submitted so far (drained, so they keep their order) and before the open one.
  void first_use(Channel& ch) {
    Schema& s = schemas[ch.schema];
    if ((s.id == 0 && written_schemas.size() == kU16Ids) || written_channels.size() == kU16Ids) {
      throw Refused("mcap: more than 65535 schemas or channels");
    }
    pool.drain();
    if (s.id == 0) {
      s.id = static_cast<uint16_t>(written_schemas.size() + 1);
      written_schemas.push_back(ch.schema);
      data(schema_record(s));
    }
    ch.id = static_cast<uint16_t>(written_channels.size() + 1);
    written_channels.push_back(static_cast<uint32_t>(&ch - channels.data()));
    data(channel_record(ch));
  }

  void submit_chunk() {
    if (buf.empty()) return;
    open.raw_size = buf.size();
    open.crc = crc32(0, buf.data(), buf.size());
    auto meta = std::make_shared<Chunk>(std::move(open));
    open = Chunk{};
    std::vector<char> raw = std::move(buf);
    buf = {};
    pool.submit(std::move(raw), [this, meta](std::vector<char>&& z) { emit_chunk(*meta, z); });
  }

  // On the calling thread, in chunk order: the Chunk record, then a Message Index per channel (id order); the Chunk
  // Index for the summary.
  void emit_chunk(Chunk& c, const std::vector<char>& z) {
    const uint64_t chunk_start = sink.tell();
    Bytes head;
    head.put(c.start).put(c.end).put(c.raw_size).put(c.crc).str(compression).put(static_cast<uint64_t>(z.size()));
    Bytes rec;
    rec.put(static_cast<uint8_t>(kChunk)).put(static_cast<uint64_t>(head.b.size() + z.size()));
    data(rec.b);
    data(head.b);
    data(z.data(), z.size());
    const uint64_t chunk_length = sink.tell() - chunk_start;
    Bytes offsets;
    const uint64_t index_start = sink.tell();
    for (auto& [id, entries] : c.index) {
      std::stable_sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.time < b.time; });
      offsets.put(id).put(sink.tell());
      Bytes mi;
      mi.put(id).put(static_cast<uint32_t>(16 * entries.size()));
      for (const Entry& e : entries) mi.put(e.time).put(e.offset);
      data(record(kMessageIndex, mi.b));
    }
    Bytes ci;
    ci.put(c.start).put(c.end).put(chunk_start).put(chunk_length);
    ci.put(static_cast<uint32_t>(offsets.b.size())).raw(offsets.b.data(), offsets.b.size());
    ci.put(sink.tell() - index_start).str(compression).put(static_cast<uint64_t>(z.size())).put(c.raw_size);
    chunk_indexes.push_back(record(kChunkIndex, ci.b));
  }

  std::string compression;
  uint64_t threshold;
  ByteSink sink;
  ChunkPool pool;  // after the sink: destroyed first, so no emit outlives it
  uint32_t data_crc = 0;
  std::vector<Schema> schemas;
  std::vector<Channel> channels;
  std::vector<uint32_t> written_schemas, written_channels;  // handles in id order
  std::vector<char> buf;                                    // the open chunk's records
  Chunk open;
  std::vector<std::vector<char>> chunk_indexes;
  uint64_t messages = 0, t_min = 0, t_max = 0;
  bool closed = false;
};

McapFile::McapFile(const std::string& path, Codec c, uint64_t chunk_bytes, int threads, std::string_view profile,
                   std::string_view library) {
  (void)compression_of(c);  // before the file is created
  if (threads < 1 || threads > ChunkPool::kMaxThreads) {
    throw Refused("threads must be in [1, " + std::to_string(ChunkPool::kMaxThreads) + "], got " +
                  std::to_string(threads));
  }
  // At most 2^32 bytes before a chunk closes, so a chunk's Message Index lengths (u32, 16 bytes per message of at
  // least 31) cannot overflow.
  if (chunk_bytes == 0 || chunk_bytes > (uint64_t{1} << 32)) throw Refused("mcap: the chunk size must be in [1, 2^32]");
  impl_ = std::make_unique<Impl>(path, c, chunk_bytes, threads, profile, library);
}

McapFile::~McapFile() = default;

uint32_t McapFile::add_schema(std::string_view name, std::string_view encoding, std::string_view data) {
  Impl& s = *impl_;
  if (s.closed) throw Refused("mcap: schema after close");
  s.schemas.push_back({std::string(name), std::string(encoding), std::string(data), 0});
  return static_cast<uint32_t>(s.schemas.size() - 1);
}

uint32_t McapFile::add_channel(uint32_t schema, std::string_view topic, std::string_view message_encoding,
                               std::vector<std::pair<std::string, std::string>> metadata) {
  Impl& s = *impl_;
  if (s.closed) throw Refused("mcap: channel after close");
  if (schema >= s.schemas.size()) throw std::out_of_range("mcap: unknown schema handle");
  std::sort(metadata.begin(), metadata.end());
  for (size_t i = 1; i < metadata.size(); ++i) {
    if (metadata[i].first == metadata[i - 1].first) throw Refused("mcap: metadata key " + metadata[i].first + " twice");
  }
  s.channels.push_back({schema, std::string(topic), std::string(message_encoding), std::move(metadata), 0, 0});
  return static_cast<uint32_t>(s.channels.size() - 1);
}

void McapFile::write(uint32_t channel, uint32_t sequence, uint64_t log_time, uint64_t publish_time, const char* data,
                     size_t n) {
  Impl& s = *impl_;
  if (s.closed) throw Refused("mcap: write after close");
  if (channel >= s.channels.size()) throw std::out_of_range("mcap: unknown channel handle");
  Impl::Channel& ch = s.channels[channel];
  if (ch.id == 0) s.first_use(ch);
  if (s.open.index.empty()) {
    s.open.start = s.open.end = log_time;
  } else {
    s.open.start = std::min(s.open.start, log_time);
    s.open.end = std::max(s.open.end, log_time);
  }
  s.open.index[ch.id].push_back({log_time, s.buf.size()});
  Bytes h;
  h.put(static_cast<uint8_t>(kMessage)).put(static_cast<uint64_t>(2 + 4 + 8 + 8 + n));
  h.put(ch.id).put(sequence).put(log_time).put(publish_time);
  s.buf.insert(s.buf.end(), h.b.begin(), h.b.end());
  s.buf.insert(s.buf.end(), data, data + n);
  s.t_min = s.messages == 0 ? log_time : std::min(s.t_min, log_time);
  s.t_max = s.messages == 0 ? log_time : std::max(s.t_max, log_time);
  ++s.messages;
  ++ch.count;
  if (s.buf.size() >= s.threshold) s.submit_chunk();
}

FileInfo McapFile::close() {
  Impl& s = *impl_;
  if (s.closed) throw Refused("mcap: closed twice");
  s.closed = true;
  s.submit_chunk();
  s.pool.drain();
  Bytes end;
  end.put(s.data_crc);  // every byte before the Data End record
  s.data(record(kDataEnd, end.b));
  // The summary: one group per record kind, each followed by its Summary Offset; the CRC runs from the summary
  // through the Footer's two offsets.
  const uint64_t summary_start = s.sink.tell();
  uint32_t crc = 0;
  auto put = [&](const std::vector<char>& r) {
    s.sink.write(r.data(), r.size());
    crc = crc32(crc, r.data(), r.size());
  };
  std::vector<std::vector<char>> offsets;
  auto group = [&](Op op, const std::vector<std::vector<char>>& records) {
    if (records.empty()) return;
    const uint64_t at = s.sink.tell();
    for (const std::vector<char>& r : records) put(r);
    Bytes o;
    o.put(static_cast<uint8_t>(op)).put(at).put(s.sink.tell() - at);
    offsets.push_back(record(kSummaryOffset, o.b));
  };
  std::vector<std::vector<char>> schemas, channels;
  for (const uint32_t h : s.written_schemas) schemas.push_back(s.schema_record(s.schemas[h]));
  for (const uint32_t h : s.written_channels) channels.push_back(s.channel_record(s.channels[h]));
  group(kSchema, schemas);
  group(kChannel, channels);
  Bytes counts;
  for (const uint32_t h : s.written_channels) counts.put(s.channels[h].id).put(s.channels[h].count);
  Bytes st;
  st.put(s.messages).put(static_cast<uint16_t>(s.written_schemas.size()));
  st.put(static_cast<uint32_t>(s.written_channels.size())).put(uint32_t{0}).put(uint32_t{0});  // attachments, metadata
  st.put(static_cast<uint32_t>(s.chunk_indexes.size())).put(s.t_min).put(s.t_max);
  st.put(static_cast<uint32_t>(counts.b.size())).raw(counts.b.data(), counts.b.size());
  group(kStatistics, {record(kStatistics, st.b)});
  group(kChunkIndex, s.chunk_indexes);
  const uint64_t summary_offset_start = s.sink.tell();
  for (const std::vector<char>& r : offsets) put(r);
  Bytes footer;
  footer.put(static_cast<uint8_t>(kFooter)).put(uint64_t{20}).put(summary_start).put(summary_offset_start);
  put(footer.b);
  Bytes tail;
  tail.put(crc).raw(kMagic.data(), kMagic.size());
  s.sink.write(tail.b.data(), tail.b.size());
  return s.sink.close();
}

}  // namespace mcap

namespace {

// ROS 2's topic name rule (design.ros2.org, "Topic and Service name mapping to DDS" and "Topic and Service names"),
// without substitutions: an optional '/' or "~/", then tokens of letters, digits and '_' separated by single '/', no
// token starting with a digit, no repeated '_', no trailing '/'.
bool ros2_topic(std::string_view s) {
  if (s.substr(0, 2) == "~/") s.remove_prefix(2);
  else if (!s.empty() && s[0] == '/') s.remove_prefix(1);
  if (s.empty() || s.find("__") != std::string_view::npos) return false;
  auto word = [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
  };
  size_t start = 0;
  for (;;) {
    const size_t end = s.find('/', start);
    const std::string_view token = s.substr(start, end == std::string_view::npos ? end : end - start);
    if (token.empty() || (token[0] >= '0' && token[0] <= '9') || !std::all_of(token.begin(), token.end(), word)) {
      return false;
    }
    if (end == std::string_view::npos) return true;
    start = end + 1;
  }
}

// The MCAP side of the ROS writers' streams (ros_streams.h): CDR messages into one MCAP file.
struct McapFormat {
  static constexpr std::string_view kName = "mcap", kStream = "channel";
  static constexpr std::string_view kRange = "ROS 2's [0, 2^31 s) (int32 seconds; MCAP log times are unsigned)";
  static constexpr std::string_view kCapacity = "whose words could pass CDR's u32 byte count (4 GiB)";
  static bool in_range(int64_t ns) { return mcap::in_range(ns); }
  static std::string_view type_name(MsgType t) { return ros2_msgdef(t).type; }

  uint64_t packet_capacity(const ros::Stream& s) const { return McapWriter::max_packet_events(s.packet_us); }

  void write_events(const ros::Stream& s, int64_t at, const Events& e, const TimeMap& m) {
    msg.clear();
    cdr::put_event_packet_evt3(msg, s.messages, at, s.frame_id, static_cast<uint32_t>(s.width),
                               static_cast<uint32_t>(s.height), e, m);
    file.write(s.handle, static_cast<uint32_t>(s.messages), static_cast<uint64_t>(at), static_cast<uint64_t>(at),
               msg.data(), msg.size());
  }

  void write_imu(const ros::Stream& s, int64_t at, const double* gyro, const double* accel,
                 const std::array<double, 9>& cov_g, const std::array<double, 9>& cov_a) {
    msg.clear();
    cdr::put_imu(msg, at, s.frame_id, gyro, accel, cov_g, cov_a);
    file.write(s.handle, static_cast<uint32_t>(s.messages), static_cast<uint64_t>(at), static_cast<uint64_t>(at),
               msg.data(), msg.size());
  }

  // An Image's data is a CDR u8 sequence, whose count is a u32.
  uint64_t image_capacity(const ros::Stream&, size_t) const { return std::numeric_limits<uint32_t>::max(); }

  void write_image(const ros::Stream& s, int64_t at, const ros::Image& im) {
    msg.clear();
    cdr::put_image(msg, at, s.frame_id, im);
    file.write(s.handle, static_cast<uint32_t>(s.messages), static_cast<uint64_t>(at), static_cast<uint64_t>(at),
               msg.data(), msg.size());
  }

  mcap::McapFile& file;
  std::vector<char> msg;  // one message's bytes, reused
};

}  // namespace

struct McapWriter::Impl {
  Impl(const std::string& path, Codec c, uint64_t chunk_bytes, const TimeMap& m, int threads,
       const std::string& library)
      : file(path, c, chunk_bytes, threads, "ros2", library), format{file, {}}, streams(m, format) {}

  mcap::McapFile file;
  McapFormat format;
  ros::Streams<McapFormat> streams;
  std::map<MsgType, uint32_t> schemas;  // the file's schema handle per type
  size_t channels = 0;
};

uint64_t McapWriter::max_packet_events(int64_t packet_us) {
  const uint64_t room = std::numeric_limits<uint32_t>::max(), fixed = ros::evt3_time_high_bytes(packet_us);
  return room > fixed ? (room - fixed) / 6 : 0;
}

McapWriter::McapWriter(const std::string& path, Codec c, uint64_t chunk_bytes, TimeMap m, int threads,
                       const std::string& library) {
  check(m);
  impl_ = std::make_unique<Impl>(path, c, chunk_bytes, m, threads, library);
}

McapWriter::~McapWriter() = default;

uint16_t McapWriter::channel(const std::string& topic, MsgType type) {
  Impl& s = *impl_;
  if (s.streams.closed()) throw Refused("mcap: channel after close");
  if (type == MsgType::kDvsEventArray) {
    throw Refused("mcap: dvs_msgs/EventArray has no ROS 2 type: use EventPacket evt3 (--event-msg evt3)");
  }
  const Ros2MsgDef& def = ros2_msgdef(type);
  if (!ros2_topic(topic)) {
    throw Refused("mcap: topic '" + topic + "' breaks ROS 2's topic name rule (an optional '/' or '~/', then tokens "
                  "of letters, digits and '_' separated by single '/', none starting with a digit, no '__', no "
                  "trailing '/'): name it with --topic STREAM:CAMERA=NAME");
  }
  s.streams.check_new(topic);
  if (s.channels == mcap::kU16Ids) throw Refused("mcap: more than 65535 channels");
  auto it = s.schemas.find(type);
  if (it == s.schemas.end()) it = s.schemas.emplace(type, s.file.add_schema(def.type, "ros2msg", def.definition)).first;
  const uint32_t handle = s.file.add_channel(it->second, topic, "cdr", {{"offered_qos_profiles", ""}});
  ++s.channels;
  return static_cast<uint16_t>(s.streams.add(topic, type, handle));
}

void McapWriter::event_packets_evt3(uint16_t ch, const std::string& frame_id, int32_t w, int32_t h, const Events& e,
                                    int64_t packet_ns) {
  impl_->streams.events(ch, MsgType::kEventPacketEvt3, frame_id, w, h, e, packet_ns);
}

void McapWriter::imu(uint16_t ch, const std::string& frame_id, const ImuView& v, CovarianceMode mode,
                     std::array<double, 2> white_var) {
  impl_->streams.imu(ch, frame_id, v, mode, white_var);
}

void McapWriter::images(uint16_t ch, const std::string& frame_id, const FrameView& v, const FrameOptions& o) {
  impl_->streams.images(ch, frame_id, v, o);
}

FileInfo McapWriter::close() {
  Impl& s = *impl_;
  std::vector<StreamInfo> streams = s.streams.close();  // writes the held packets
  FileInfo info = s.file.close();
  info.streams = std::move(streams);
  return info;
}

}  // namespace balla::exporter
