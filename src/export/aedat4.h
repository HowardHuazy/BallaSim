// AEDAT 4.0 files of a run's events and IMU (docs/design/phase3/outputs.md §3.3, §3.7, §4.7-4.8; EXPORT.md).
// Provenance: an independent implementation from dv-processing's public FlatBuffers schemas and its writer's
// behaviour (iniVation, Apache-2.0: IOHeader, FileDataTable, Event and IMU tables, the I/O layout, the IMU units), as
// R-FMT's prototype (phase3_research/outputs/formats/scripts/aedat4.py) wrote them down, and checked against
// dv-processing 2.0.4's reader and writer (the pin, EXPORT.md); no dv-processing code is used
// (THIRD_PARTY_NOTICES.md, "Ported code").
//   File            "#!AER-DAT4.0\r\n"; the IOHeader, a size-prefixed FlatBuffer "IOHE" (compression,
//                   dataTablePosition, infoNode XML) written with every field (ForceDefaults) and patched at close
//                   with the table's position; the packets, each {int32 stream, int32 size} then one frame (the
//                   file's codec) of a size-prefixed FlatBuffer: "EVTS" (a vector of 16-byte Event structs) or
//                   "IMUS" (a vector of IMU tables); last, without a packet header and up to the end of the file,
//                   the FileDataTable "FTAB" (one entry per packet: offset after its header, stream, size, element
//                   count, first and last timestamp), size-prefixed and compressed with the file's codec as one frame.
//   Codecs          none, lz4 (frame, 64 KB linked blocks, level 0), lz4-high (level 9), zstd (level 3), zstd-high
//                   (ZSTD_maxCLevel()): dv-processing's settings per CompressionType, byte for byte.
//   Aedat4Writer    one file of a run's cameras (§3.7): per camera an events stream (EVTS, output "events")
//                   and, when it has one, an IMU stream (IMUS, output "imu"), source "BallaSim_<camera>", ids
//                   0..n-1 in the order added; each stream's items cut into packets of packet_us (default 10 ms) on
//                   the T1a grid, held until an item of a later packet arrives or close() (T3), and compressed on a
//                   ChunkPool: a stream's packets depend neither on how its items are batched nor on the thread
//                   count, and packets enter the file as they complete, so the interleave of the streams follows the
//                   order of the calls across them (the driver fixes it with its 100 ms rounds). Times T1b:
//                   floor(t_ns / 1000) - origin_us + epoch_ns / 1000 (int64 us). IMU samples in dv's units:
//                   accelerometer float32(m/s^2 / 9.81007) in g (dv's getAccelerations() multiplies by 9.81007f),
//                   gyroscope float32(rad/s * 180 / pi) in deg/s, temperature and magnetometer 0 (outputs.md Q12).
//                   Frames (A9): per camera a frames stream (FRME, output "frames", sizeX and sizeY the full image),
//                   one Frame per packet as dv-processing writes them, 8-bit only (OPENCV_8U_C1 gray, OPENCV_8U_C3
//                   BGR; PHASE3.md §5.3): gray16 and f32 frames go through an 8-bit map (frame_map.h). Its times
//                   (frames.md §3.11, T1b): timestamp the exposure midpoint, start of frame and of exposure the first
//                   read row's exposure start, end of exposure the last read row's, end of frame that plus one row
//                   time; `exposure` the per-row exposure in us (floored); position the ROI's top-left pixel; source
//                   SYNTHETIC; the packet's table entry at the exposure start, the time dv-processing 2.0.4 gives
//                   the frame (its time queries go by the table). dv-processing 2.0.4's own FRME packets pin the
//                   table layout (tests/cpp/export).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "export/compress.h"
#include "export/frame_map.h"
#include "export/sink.h"
#include "export/time_map.h"

namespace balla::exporter {

namespace aedat4 {

inline constexpr std::string_view kVersion{"#!AER-DAT4.0\r\n"};
inline constexpr double kEarthG = 9.81007;  // m/s^2 per g in dv's IMU (imu_base.hpp getAccelerations: earthG)
inline constexpr int32_t kMaxSize = 32767;  // Event x and y are int16
// The largest packet FlatBuffer: its frame, whatever the codec's worst-case growth, keeps an int32 packet size.
inline constexpr size_t kMaxPacketBytes = (size_t{1} << 31) - (size_t{1} << 24);

// dv's CompressionType (NONE 0, LZ4 1, LZ4_HIGH 2, ZSTD 3, ZSTD_HIGH 4) and its name; Refused for bz2.
int32_t compression_type(Codec);
std::string_view compression_name(Codec);

// One stream of the infoNode: kind "events" (EVTS, with width and height), "imu" (IMUS) or "frames" (FRME, with the
// full image's width and height).
struct StreamDecl {
  int32_t id = 0;
  std::string camera, kind;
  int32_t width = 0, height = 0;
};
// The infoNode XML (dv's ConfigTree layout, compact): <dv version="2.0"><node name="outInfo" path="/outInfo/">, per
// stream (ids ordered as decimal strings) its compression, originalModuleName BallaSim, originalOutputName (the
// kind), typeDescription and typeIdentifier, and an info node with sizeX and sizeY (events),
// accelerometerUnitMps2 9.81007 (imu) and source BallaSim_<camera>, attributes in key order, text XML-escaped.
std::string info_node(Codec, const std::vector<StreamDecl>&);
// The IOHeader FlatBuffer (size-prefixed, "IOHE", every field written).
std::vector<char> io_header(Codec, int64_t data_table_position, std::string_view info);
// An EVTS packet FlatBuffer of events (T1b timestamps under the map).
std::vector<char> event_packet(const Events&, const TimeMap&);
// One IMU table's fields in dv's units.
struct ImuRecord {
  int64_t timestamp = 0;  // us
  float temperature = 0, accelerometer[3] = {}, gyroscope[3] = {}, magnetometer[3] = {};
};
std::vector<char> imu_packet(const std::vector<ImuRecord>&);
// One Frame table's fields (dv's frame.fbs: times in us, format an OpenCV type code, sizes and position int16, pixels
// 8-bit rows top-down, source a FrameSource).
inline constexpr int8_t kOpenCv8UC1 = 0, kOpenCv8UC3 = 16;  // FrameFormat OPENCV_8U_C1, OPENCV_8U_C3
inline constexpr int8_t kSourceSynthetic = 4;               // FrameSource SYNTHETIC
struct FrameRecord {
  int64_t timestamp = 0, start_of_frame = 0, end_of_frame = 0, start_of_exposure = 0, end_of_exposure = 0;
  int64_t exposure = 0;
  int8_t format = kOpenCv8UC1, source = kSourceSynthetic;
  int16_t size_x = 0, size_y = 0, position_x = 0, position_y = 0;
  const uint8_t* pixels = nullptr;
  size_t pixel_bytes = 0;
};
// An FRME packet FlatBuffer of one frame, fields in dv's CreateFrame order, defaults left out as FlatBuffers does.
std::vector<char> frame_packet(const FrameRecord&);
// One FileDataTable entry.
struct TableEntry {
  int64_t byte_offset = 0;  // of the packet's data, after its header
  int32_t stream = 0, size = 0;
  int64_t elements = 0, t_start = 0, t_end = 0;
};
std::vector<char> file_data_table(const std::vector<TableEntry>&);

}  // namespace aedat4

class Aedat4Writer {
 public:
  static constexpr int64_t kDefaultPacketUs = 10'000;   // §3.7: packets every 10 ms per stream
  static constexpr int64_t kMaxPacketUs = int64_t{1} << 24;  // the CLI's --packet-us range
  static constexpr int kMaxStreams = 1 << 16;
  // The most items one packet may hold within kMaxPacketBytes: 16 bytes per event; at most 96 bytes per IMU sample
  // (its table, vtable, padding and vector slot).
  static uint64_t max_packet_events();
  static uint64_t max_packet_samples();
  // The most packets a file may hold, so its FileDataTable (at most 72 bytes per entry) stays a FlatBuffer.
  static uint64_t max_packets();

  // codec none, lz4, lz4-high, zstd or zstd-high (else Refused, before the file is created); the TimeMap's epoch a
  // multiple of 1000; threads in [1, 256]; packet_us in [1, 2^24].
  Aedat4Writer(const std::string& path, Codec, TimeMap, int threads, int64_t packet_us = kDefaultPacketUs);
  ~Aedat4Writer();
  Aedat4Writer(const Aedat4Writer&) = delete;
  Aedat4Writer& operator=(const Aedat4Writer&) = delete;

  // A camera's events stream (width and height in [1, 32767]) or IMU stream, each once per camera; camera: a non-empty
  // name without control characters. Returns the stream id. Every stream is declared before the first item, since the
  // IOHeader written then lists them.
  int32_t add_events_stream(const std::string& camera, int32_t w, int32_t h);
  int32_t add_imu_stream(const std::string& camera);
  // A camera's frames stream (A9), the full image width and height in [1, 32767]; returns the stream id.
  int32_t add_frame_stream(const std::string& camera, int32_t w, int32_t h);
  // A stream's events in output order (t_ns non-decreasing across calls, each >= 1000 origin_us with a T1b timestamp
  // >= 0, x < w, y < h, p in {0, 1}); a batch that breaks a rule, or would fill a packet past max_packet_events() or
  // the file past max_packets(), is refused whole.
  void events(int32_t stream, const Events&);
  // A stream's IMU samples in stamp order (C5 published values, rad/s and m/s^2); refused whole as events().
  void imu(int32_t stream, const ImuView&);
  // A frames stream's frames (times on the sensor clock, midpoints and exposure starts non-decreasing across calls,
  // each frame's start >= 1000 origin_us with T1b times >= 0, the ROI inside the stream's image), one FRME packet
  // each, written at once, its table entry at the exposure start.
  // 8-bit frames are stored as they are under FrameMap::kNone (rgb8 as BGR); gray16 and f32 frames need linear8 or
  // srgb8. A batch that breaks a rule is refused whole.
  void frames(int32_t stream, const FrameView&, const FrameOptions&);
  // Writes the held packets (stream order), the FileDataTable and the IOHeader's table position; streams: one per
  // stream {camera, kind events, imu or frames, id the stream id, count, t_first, t_last (T1b us; frames: their
  // midpoints)}.
  FileInfo close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace balla::exporter
