// The IMU sidecar of the RAW and DAT exports, <cam>.imu.csv (docs/design/phase3/outputs.md §3.7): the public RAW
// formats have no IMU word, so a camera's C5 samples go to a CSV on the events' us base (T1a). One header row, then
// one row per sample:
//   t_us[,cgx,cgy,cgz][,cax,cay,caz],gx,gy,gz,ax,ay,az
// t_us = floor_div(t_ns, 1000) - origin_us; the integer counts of each sensor whose lsb is set; the published values
// (rad/s, m/s^2) as %.17g, which round-trips float64 (std::to_chars: the C locale whatever the process's). Header-only.
#pragma once

#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>

#include "export/errors.h"
#include "export/sink.h"
#include "export/time_map.h"

namespace balla::exporter {

class ImuCsvWriter {
 public:
  // counts_gyro, counts_accel: whether the rows carry that sensor's counts (its lsb is set); every append must then
  // give them, and no others.
  ImuCsvWriter(const std::string& path, TimeMap m, bool counts_gyro, bool counts_accel)
      : map_((check(m), m)), counts_gyro_(counts_gyro), counts_accel_(counts_accel), sink_(path) {
    std::string h = "t_us";
    if (counts_gyro_) h += ",cgx,cgy,cgz";
    if (counts_accel_) h += ",cax,cay,caz";
    h += ",gx,gy,gz,ax,ay,az\n";
    sink_.write(h.data(), h.size());
  }

  // Samples in stamp order (non-decreasing within and across calls), each >= 1000 origin_us; a batch that breaks
  // either, or whose counts do not match the constructor's, is refused whole (nothing written).
  void append(const ImuView& v) {
    if (closed_) throw Refused("ImuCsvWriter: append after close");
    if ((v.counts_gyro != nullptr) != counts_gyro_ || (v.counts_accel != nullptr) != counts_accel_) {
      throw Refused("ImuCsvWriter: the counts given do not match the file's columns");
    }
    if (v.n == 0) return;
    if (v.t_ns == nullptr || v.gyro == nullptr || v.accel == nullptr) {
      throw std::invalid_argument("ImuCsvWriter: null columns");
    }
    int64_t prev = have_last_ ? last_t_ : std::numeric_limits<int64_t>::min();
    for (size_t i = 0; i < v.n; ++i) {
      if (v.t_ns[i] < prev) {
        throw Refused("ImuCsvWriter: IMU samples must be in stamp order: t_ns " + std::to_string(v.t_ns[i]) +
                      " follows " + std::to_string(prev));
      }
      prev = v.t_ns[i];
    }
    if (t1a_us(map_, v.t_ns[0]) < 0) {
      throw Refused("ImuCsvWriter: t_ns " + std::to_string(v.t_ns[0]) + " lies before the export origin " +
                    std::to_string(map_.origin_us) + " us");
    }
    buf_.clear();
    for (size_t i = 0; i < v.n; ++i) {
      put(t1a_us(map_, v.t_ns[i]));
      if (counts_gyro_) {
        for (size_t k = 0; k < 3; ++k) put(',', v.counts_gyro[3 * i + k]);
      }
      if (counts_accel_) {
        for (size_t k = 0; k < 3; ++k) put(',', v.counts_accel[3 * i + k]);
      }
      for (size_t k = 0; k < 3; ++k) put(',', v.gyro[3 * i + k]);
      for (size_t k = 0; k < 3; ++k) put(',', v.accel[3 * i + k]);
      buf_.push_back('\n');
    }
    sink_.write(buf_.data(), buf_.size());
    if (count_ == 0) t_first_ = t1a_us(map_, v.t_ns[0]);
    t_last_ = t1a_us(map_, v.t_ns[v.n - 1]);
    last_t_ = v.t_ns[v.n - 1];
    have_last_ = true;
    count_ += static_cast<int64_t>(v.n);
  }

  FileInfo close() {
    if (closed_) throw Refused("ImuCsvWriter: closed twice");
    closed_ = true;
    FileInfo info = sink_.close();
    StreamInfo s;
    s.kind = "imu";
    s.count = count_;
    s.t_first = t_first_;
    s.t_last = t_last_;
    info.streams = {s};
    return info;
  }

 private:
  template <class T>
  void put(T value) {
    char tmp[40];
    std::to_chars_result r{};
    if constexpr (std::is_floating_point_v<T>) {
      r = std::to_chars(tmp, tmp + sizeof tmp, value, std::chars_format::general, 17);  // printf's %.17g
    } else {
      r = std::to_chars(tmp, tmp + sizeof tmp, value);
    }
    if (r.ec != std::errc{}) throw std::logic_error("ImuCsvWriter: to_chars failed");
    buf_.append(tmp, r.ptr);
  }
  template <class T>
  void put(char sep, T value) {
    buf_.push_back(sep);
    put(value);
  }

  TimeMap map_;
  bool counts_gyro_, counts_accel_;
  ByteSink sink_;
  std::string buf_;
  bool closed_ = false, have_last_ = false;
  int64_t last_t_ = 0, count_ = 0, t_first_ = 0, t_last_ = 0;
};

}  // namespace balla::exporter
