// ROS1 bags (rosbag format 2.0) of a run's events and IMU (docs/design/phase3/outputs.md §3.3, §3.7, §4.7-4.8),
// written from the rosbag 2.0 format description (wiki.ros.org/Bags/Format/2.0), ros_comm's rosbag_storage behaviour
// as re-read at 30483a9 (R-DOC out/upstream_pins.json: Bag::init's 768 KiB chunk threshold, stopWriting's index
// section, the file header padded to 4096 bytes and rewritten at close, roslz4's frame header) and the R-FMT prototype
// (phase3_research/outputs/formats/scripts/ros1.py, whose golden bag tests/cpp/export reproduces); no ROS code is used.
//   ros1::put_*     the ROS1 serialization (little-endian; strings and variable arrays u32-length-prefixed, fixed
//                   arrays bare) of dvs_msgs/EventArray, event_camera_msgs/EventPacket (evt3), sensor_msgs/Imu and
//                   sensor_msgs/Image (A9)
//   ros1::BagFile   the container: '#ROSBAG V2.0\n'; the bag header record (op 0x03); chunks (0x05) of message
//                   records (0x02), each holding the connection records (0x07) of the connections first used in it
//                   and closed once its data passes the threshold, then one index data record (0x04) per connection
//                   of the chunk; after index_pos the connection records, then the chunk infos (0x06). Header fields
//                   are in name order, as ros::Header writes its std::map. Chunks are compressed on a ChunkPool
//                   (none, bz2 block 9, or lz4 in the roslz4 frame: FLG 0x64, BD 0x60), so the bytes do not depend
//                   on the thread count.
//   Rosbag1Writer   one bag of a run's cameras (§3.7) through the ROS writers' shared streams (ros_streams.h):
//                   event messages per packet of packet_ns on the T1a us grid (k = floor(u / packet_us)), each held
//                   back until an event of a later packet arrives or close(), so the bytes do not depend on how the
//                   events are batched (T3); IMU messages per sample; every stamp T2 (exact ns: t_ns - 1000 origin_us
//                   + epoch_ns, 0 <= sec < 2^32), the record time equal to header.stamp, header.stamp of an event
//                   message its first event's (as the Prophesee driver), of an image its exposure midpoint's.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "export/compress.h"
#include "export/msgdefs.h"
#include "export/ros_streams.h"
#include "export/sink.h"
#include "export/time_map.h"

namespace balla::exporter {

namespace ros1 {

inline constexpr uint32_t kChunkThreshold = 768 * 1024;  // rosbag's Bag::init()
inline constexpr size_t kFileHeaderLength = 4096;        // rosbag's FILE_HEADER_LENGTH
inline constexpr int64_t kNsPerSec = 1'000'000'000;
inline constexpr int64_t kMaxStampNs = (int64_t{1} << 32) * kNsPerSec;  // ROS1 time: u32 seconds

// A ROS1 time (u32 sec, u32 nsec) of ns in [0, kMaxStampNs); the caller has checked the range.
struct Time {
  uint32_t sec = 0, nsec = 0;
};
inline Time time_of(int64_t ns) {
  return {static_cast<uint32_t>(ns / kNsPerSec), static_cast<uint32_t>(ns % kNsPerSec)};
}
inline bool in_range(int64_t ns) { return ns >= 0 && ns < kMaxStampNs; }

// std_msgs/Header: seq, stamp, frame_id.
void put_header(std::vector<char>& out, uint32_t seq, int64_t stamp_ns, std::string_view frame_id);
// dvs_msgs/EventArray: header, height, width, events[] of (x u16, y u16, ts, polarity) in 13 bytes, each ts the T2
// stamp of its t_ns by m (the caller has checked them).
void put_event_array(std::vector<char>& out, uint32_t seq, int64_t stamp_ns, std::string_view frame_id, uint32_t width,
                     uint32_t height, const Events& e, const TimeMap& m);
// event_camera_msgs/EventPacket: header (seq), height, width, seq (u64), time_base 0, encoding "evt3", is_bigendian 0,
// events: the EVT 3.0 words (evt.h, Start::kFirstEvent) of the T1a times u >= 0, little-endian.
void put_event_packet_evt3(std::vector<char>& out, uint64_t seq, int64_t stamp_ns, std::string_view frame_id,
                           uint32_t width, uint32_t height, const Events& e, const TimeMap& m);
// sensor_msgs/Imu: header, orientation (0, 0, 0, 0) with orientation_covariance[0] = -1 (no estimate), the angular
// velocity and linear acceleration (float64, the published values) and their row-major covariances.
void put_imu(std::vector<char>& out, uint32_t seq, int64_t stamp_ns, std::string_view frame_id, const double* gyro,
             const double* accel, const std::array<double, 9>& cov_gyro, const std::array<double, 9>& cov_accel);
// sensor_msgs/Image: header, height, width, encoding, is_bigendian 0, step, data (u32 length, the bytes).
void put_image(std::vector<char>& out, uint32_t seq, int64_t stamp_ns, std::string_view frame_id, const ros::Image&);

class BagFile {
 public:
  // codec: none, bz2 or lz4 (the roslz4 frame), else Refused; a chunk closes once its data exceeds chunk_threshold
  // bytes; threads: the compression pool's, in [1, 256].
  BagFile(const std::string& path, Codec, uint32_t chunk_threshold, int threads);
  ~BagFile();
  BagFile(const BagFile&) = delete;
  BagFile& operator=(const BagFile&) = delete;

  // A connection for one topic and type: its connection header holds md5sum, message_definition, topic and type
  // (as rospy's Bag.write; no callerid). Its id is assigned at its first message in first-use order, as rosbag
  // assigns them, so a connection never written stays out of the bag. Returns a handle.
  uint32_t add_connection(const std::string& topic, MsgType);
  // One message record of n bytes on a connection, received at time_ns in [0, 2^32 s) (else Refused). A record or
  // chunk beyond 4 GiB is Refused before anything of it is written.
  void write(uint32_t handle, int64_t time_ns, const char* data, size_t n);
  // The largest message write() accepts on a connection whatever the open chunk holds (it holds at most the chunk
  // threshold): 2^32 - 1 less the threshold, the connection's record and the message record's header; 0 if none.
  uint64_t max_message(uint32_t handle) const;
  // Flushes the open chunk, writes the index section, rewrites the bag header and hashes the file.
  FileInfo close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ros1

class Rosbag1Writer {
 public:
  // The longest packet (ros_streams.h): an EventPacket evt3 payload carries u mod 2^24, which its reader recovers only
  // within a packet spanning less than 2^24 us (16.78 s, §4.7); EventArray packets keep the same bound.
  static constexpr int64_t kMaxPacketUs = ros::kMaxPacketUs;

  // codec none, bz2 or lz4; chunk_bytes: the chunk threshold (rosbag's 768 KiB by default); the TimeMap's epoch a
  // multiple of 1000 (T2); threads in [1, 256].
  Rosbag1Writer(const std::string& path, Codec, uint32_t chunk_bytes, TimeMap, int threads);
  ~Rosbag1Writer();
  Rosbag1Writer(const Rosbag1Writer&) = delete;
  Rosbag1Writer& operator=(const Rosbag1Writer&) = delete;

  // A connection: topic by ROS's name rule (first character a letter, '/' or '~'; then letters, digits, '_' or '/'),
  // each topic once; the type kDvsEventArray, kEventPacketEvt3, kImu or kImage. Returns its handle.
  uint32_t connection(const std::string& topic, MsgType);
  // A connection's events in output order (t_ns non-decreasing across calls, each >= 1000 origin_us, x < w, y < h,
  // p in {0, 1}, every T2 stamp in [0, 2^32 s)), cut into messages of packet_ns (a multiple of 1000 in
  // [1000, 1000 kMaxPacketUs]) on the T1a grid. frame_id, w, h and packet_ns are fixed by a connection's first call.
  // w and h: [1, 65536] for EventArray (u16 coordinates), [1, 2048] for EventPacket evt3 (11-bit addresses). A packet
  // may hold as many events as its message can take at the most bytes they can need (EventArray 13 each; evt3 6
  // each plus packet_us / 4096 + 2 TIME_HIGH words) within BagFile::max_message, so a held packet always fits its
  // record. A batch that breaks any rule, this one included, is refused whole, before any of it is held.
  void event_arrays(uint32_t conn, const std::string& frame_id, int32_t w, int32_t h, const Events&, int64_t packet_ns);
  void event_packets_evt3(uint32_t conn, const std::string& frame_id, int32_t w, int32_t h, const Events&,
                          int64_t packet_ns);
  // A connection's IMU samples in stamp order, one sensor_msgs/Imu each; covariances diag(white_var) (gyro, accel:
  // finite, >= 0) for kModel, zero for kZero. frame_id is fixed by the first call.
  void imu(uint32_t conn, const std::string& frame_id, const ImuView&, CovarianceMode, std::array<double, 2> white_var);
  // A connection's frames (ros_streams.h images): one sensor_msgs/Image each, stamped at the exposure midpoint.
  void images(uint32_t conn, const std::string& frame_id, const FrameView&, const FrameOptions&);
  // Writes the held packets (connection order), closes the bag; streams: one per connection, {kind events, imu or
  // frames, id the topic, count, t_first, t_last (their T2 stamps, ns)}, camera left to the caller.
  FileInfo close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace balla::exporter
