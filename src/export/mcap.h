// MCAP files of a run's events and IMU as ROS 2 messages (docs/design/phase3/outputs.md §3.3, §3.7, §4.7-4.8),
// written from the MCAP specification (mcap.dev/spec, format version 0, and its ros2 profile) and the R-FMT prototype
// (phase3_research/outputs/formats/scripts/mcap2.py, whose golden tiny.mcap tests/cpp/export reproduces); no MCAP or
// ROS code is used.
//   mcap::McapFile  the container: the magic; the Header record (profile, library); in the data section the Schema
//                   and Channel records, each placed outside the chunks before the first message that uses it, and
//                   the chunks (op 0x06, records compressed whole: "" none, "lz4" an LZ4 frame, "zstd" level 3, with
//                   the CRC-32 of the uncompressed records), each closed once its records reach the chunk size and
//                   followed by one Message Index per channel (id order); the Data End record with the CRC-32 of every
//                   byte before it (the magic and the Header included, as the reference readers check it); the
//                   summary: the Schema, Channel, Statistics and Chunk Index groups, one Summary Offset per group, and
//                   the Footer with the CRC-32 of the summary through the Footer's offsets; the magic. Chunks are
//                   compressed on a ChunkPool, so the bytes do not depend on the thread count.
//   McapWriter      one file of a run's cameras (§3.7) through the ROS writers' shared streams (ros_streams.h):
//                   event_camera_msgs/msg/EventPacket evt3 per packet on the T1a grid, sensor_msgs/msg/Imu per sample,
//                   sensor_msgs/msg/Image per frame at its exposure midpoint (A9),
//                   CDR (cdr.h); every stamp T2 (exact ns: t_ns - 1000 origin_us + epoch_ns, 0 <= sec < 2^31, as ROS 2
//                   time has int32 seconds and MCAP log times are unsigned), log time = publish time = header.stamp,
//                   the sequence the channel's message index. Profile ros2, schema encoding ros2msg (msgdefs.h),
//                   message encoding cdr, channel metadata offered_qos_profiles "" and no topic_type_hash or rosbag2
//                   Metadata record (outputs.md Q11): rosbag2 opens the file alone, deriving its metadata from the
//                   Statistics record and the channels (§2.5).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "export/compress.h"
#include "export/msgdefs.h"
#include "export/ros_streams.h"
#include "export/sink.h"
#include "export/time_map.h"

namespace balla::exporter {

namespace mcap {

inline constexpr std::string_view kMagic{"\x89MCAP0\r\n", 8};
inline constexpr uint64_t kChunkBytes = uint64_t{1} << 20;  // §4.8: chunks of 1 MiB
inline constexpr int64_t kNsPerSec = 1'000'000'000;
inline constexpr int64_t kMaxStampNs = (int64_t{1} << 31) * kNsPerSec;  // ROS 2 time: int32 seconds
// A T2 stamp ROS 2 time and an MCAP log time both hold: [0, 2^31 s).
inline bool in_range(int64_t ns) { return ns >= 0 && ns < kMaxStampNs; }

// The record opcodes this writer uses (mcap.dev/spec).
enum Op : uint8_t {
  kHeader = 0x01, kFooter = 0x02, kSchema = 0x03, kChannel = 0x04, kMessage = 0x05, kChunk = 0x06,
  kMessageIndex = 0x07, kChunkIndex = 0x08, kStatistics = 0x0B, kSummaryOffset = 0x0E, kDataEnd = 0x0F
};

class McapFile {
 public:
  // codec: none, lz4 (the LZ4 frame, 64 KB linked blocks, level 0) or zstd (level 3), else Refused; a chunk closes once
  // its records reach chunk_bytes (in [1, 2^32]); threads: the compression pool's, in [1, 256]. Writes the magic and
  // the Header.
  McapFile(const std::string& path, Codec, uint64_t chunk_bytes, int threads, std::string_view profile,
           std::string_view library);
  ~McapFile();
  McapFile(const McapFile&) = delete;
  McapFile& operator=(const McapFile&) = delete;

  // A schema, and a channel on one: registered here and written to the data section before the first message that
  // uses them, with ids from 1 in first-use order, so a schema or channel never used stays out of the file. Each
  // returns a handle. metadata: the channel's string map, written in key order.
  uint32_t add_schema(std::string_view name, std::string_view encoding, std::string_view data);
  uint32_t add_channel(uint32_t schema, std::string_view topic, std::string_view message_encoding,
                       std::vector<std::pair<std::string, std::string>> metadata);
  // One Message record (sequence, log and publish time in ns, data) on a channel, into the open chunk.
  void write(uint32_t channel, uint32_t sequence, uint64_t log_time, uint64_t publish_time, const char* data,
             size_t n);
  // Flushes the open chunk, writes the Data End record, the summary, the summary offsets and the footer, and hashes
  // the file.
  FileInfo close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mcap

class McapWriter {
 public:
  static constexpr int64_t kMaxPacketUs = ros::kMaxPacketUs;
  // The most events an EventPacket of packet_us may hold: their words at 6 bytes each (TIME_LOW, ADDR_Y, ADDR_X) and
  // the packet's TIME_HIGH words within the u32 byte count of CDR's events sequence (MCAP's own lengths are u64).
  static uint64_t max_packet_events(int64_t packet_us);

  // codec none, lz4 or zstd; chunk_bytes: the chunk size (1 MiB by default, §4.8); the TimeMap's epoch a multiple of
  // 1000 (T2); threads in [1, 256]; library: the Header's library ("BallaSim <version>", §3.8).
  McapWriter(const std::string& path, Codec, uint64_t chunk_bytes, TimeMap, int threads, const std::string& library);
  ~McapWriter();
  McapWriter(const McapWriter&) = delete;
  McapWriter& operator=(const McapWriter&) = delete;

  // A channel: topic by ROS 2's topic name rule (an optional '/' or "~/", then tokens of letters, digits and '_'
  // separated by single '/', none starting with a digit, no "__", no trailing '/'), each topic once; the type
  // kEventPacketEvt3, kImu or kImage (no ROS 2 dvs_msgs, §1.3). Returns its handle; at most 65535 channels.
  uint16_t channel(const std::string& topic, MsgType);
  // As Rosbag1Writer's (ros1.h), on the stamps of ROS 2 time: a channel's events in output order cut into
  // event_camera_msgs/msg/EventPacket evt3 messages of packet_ns on the T1a grid (w and h in [1, 2048]); a packet holds
  // at most the events whose words fit the CDR byte count (u32); a batch that breaks a rule is refused whole.
  void event_packets_evt3(uint16_t ch, const std::string& frame_id, int32_t w, int32_t h, const Events&,
                          int64_t packet_ns);
  // A channel's IMU samples in stamp order, one sensor_msgs/msg/Imu each; covariances as Rosbag1Writer::imu.
  void imu(uint16_t ch, const std::string& frame_id, const ImuView&, CovarianceMode, std::array<double, 2> white_var);
  // A channel's frames, one sensor_msgs/msg/Image each at its midpoint's stamp (ros_streams.h images).
  void images(uint16_t ch, const std::string& frame_id, const FrameView&, const FrameOptions&);
  // Writes the held packets (channel order), closes the file; streams: one per channel, {kind events, imu or frames,
  // id the topic, count, t_first, t_last (T2 ns)}, camera left to the caller.
  FileInfo close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace balla::exporter
