// Prophesee RAW (EVT 2.0, 2.1, 3.0) and DAT files of one camera's events (docs/design/phase3/outputs.md §3.3, §3.7,
// §4.7-4.8): a '% key value' header, final when written (the driver pre-scans the window's t_ns_rem), then the words
// of evt.h or the records of dat.h under the T1a rule. append() takes the events in output order, any batch sizes
// (one batch and many give the same bytes, T3); close() returns the file's size, SHA-256 and its one events stream
// (count, first and last u). The RAW formats have no IMU word: the IMU goes to the CSV sidecar of imu_csv.h.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "export/evt.h"
#include "export/sink.h"
#include "export/time_map.h"

namespace balla::exporter {

enum class RawFormat : uint8_t { kEvt2, kEvt21, kEvt3 };

struct RawOptions {
  Evt2TimeHigh evt2_time_high = Evt2TimeHigh::kOpenEb;  // EVT 2.0 only; EVT 2.1 is always on change (evt.h)
  bool evt21_legacy_endianness = false;  // refused: the legacy word layout has no verified source here (EXPORT.md)
  // The dropped sub-us parts (t_reported mod 1000) of the window's events, pre-scanned by the driver: the header says
  // `ballasim_t_ns_rem <n>` when min = max, else `varies`; -1, -1 when the window holds no event (`none`). An event
  // outside [min, max] is refused, so the header never misstates the file.
  int32_t t_ns_rem_min = 0, t_ns_rem_max = 0;
  std::string version;  // "BallaSim <version>" in the RAW header (plugin_version)
};

std::string_view raw_format_name(RawFormat);  // the header's format: EVT2, EVT21, EVT3

class RawWriter {
 public:
  // Refused: width or height outside [1, 2048] (11-bit addresses), an epoch that is not a multiple of 1000, a
  // t_ns_rem range outside [0, 999] (or not -1, -1), a version with control characters, legacy EVT 2.1.
  RawWriter(const std::string& path, RawFormat, int32_t width, int32_t height, TimeMap, const RawOptions&);
  ~RawWriter();
  RawWriter(const RawWriter&) = delete;
  RawWriter& operator=(const RawWriter&) = delete;

  // Events in output order: t_ns non-decreasing within and across calls, >= 1000 origin_us, x < width, y < height,
  // p in {0, 1}, t_ns mod 1000 in the declared range, and for EVT 2.0 and 2.1 a first event at u < 2^34 us (their
  // first TIME_HIGH is read as absolute; later events may cross the wrap); a batch that breaks any is refused whole
  // (nothing written). The words go out in 256 KiB blocks, so the memory does not grow with the batch.
  void append(const Events&);
  FileInfo close();  // flushes EVT 2.1's open word; streams = {kind "events", count, t_first, t_last} in u (us)

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class DatWriter {
 public:
  // As RawWriter, with width and height in [1, 16384] (14-bit addresses).
  DatWriter(const std::string& path, int32_t width, int32_t height, TimeMap, const RawOptions&);
  ~DatWriter();
  DatWriter(const DatWriter&) = delete;
  DatWriter& operator=(const DatWriter&) = delete;

  void append(const Events&);  // as RawWriter::append; also refused when u >= 2^32 (DAT's u32 microseconds)
  FileInfo close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace balla::exporter
