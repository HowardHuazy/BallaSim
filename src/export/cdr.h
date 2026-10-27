// CDR serialization of the ROS 2 messages the MCAP writer stores (docs/design/phase3/outputs.md §4.8), written from
// the encoding rules as R-FMT's prototype states and applies them (phase3_research/outputs/formats/scripts/mcap2.py,
// class Cdr; Fast-CDR Cdr.h:329-333, Cdr.cpp:268-275 and 858-879 at 03cb3bf, R-DOC out/upstream_pins.json); no ROS or
// Fast-CDR code is used. XCDR version 1 ("plain CDR") little-endian, as ROS 2's rmw serializes:
//   - the encapsulation header 00 01 00 00 (CDR_LE, options 0);
//   - each primitive aligned to its size (at most 8) relative to the byte after that header, padded with zeros;
//   - a string as a u32 length that counts its terminating NUL, the bytes, then the NUL;
//   - a sequence as a u32 element count, then the elements; a fixed array as its elements alone;
//   - bool as one byte; no padding after the last field.
// Messages (§3.7): std_msgs/msg/Header (builtin_interfaces/msg/Time: int32 sec, u32 nanosec; frame_id; no seq in
// ROS 2), event_camera_msgs/msg/EventPacket (evt3), sensor_msgs/msg/Imu and sensor_msgs/msg/Image (A9). Header-only.
#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#include "export/evt.h"
#include "export/ros_streams.h"
#include "export/time_map.h"

namespace balla::exporter::cdr {

static_assert(std::endian::native == std::endian::little, "CDR_LE is written in host order");

inline constexpr int64_t kNsPerSec = 1'000'000'000;

// Appends one CDR_LE message to a byte vector: the encapsulation header at construction, then the fields.
class Writer {
 public:
  explicit Writer(std::vector<char>& out) : out_(out) {
    static constexpr char kEncapsulation[4] = {0x00, 0x01, 0x00, 0x00};
    out_.insert(out_.end(), kEncapsulation, kEncapsulation + 4);
    base_ = out_.size();
  }

  template <class T>
  void put(T v) {
    align(sizeof(T));
    raw(&v, sizeof(T));
  }
  void string(std::string_view s) {
    put(static_cast<uint32_t>(s.size() + 1));
    raw(s.data(), s.size());
    out_.push_back('\0');
  }
  // n values of T as a fixed array (no count).
  template <class T>
  void array(const T* v, size_t n) {
    align(sizeof(T));
    raw(v, n * sizeof(T));
  }
  void align(size_t n) { out_.resize(out_.size() + (n - (out_.size() - base_) % n) % n, '\0'); }
  void raw(const void* p, size_t n) {
    const auto* c = static_cast<const char*>(p);
    out_.insert(out_.end(), c, c + n);
  }
  std::vector<char>& bytes() { return out_; }

 private:
  std::vector<char>& out_;
  size_t base_ = 0;
};

// std_msgs/msg/Header of a stamp in [0, 2^31 s) (the caller has checked it): int32 sec, u32 nanosec, frame_id.
inline void put_header(Writer& w, int64_t stamp_ns, std::string_view frame_id) {
  w.put(static_cast<int32_t>(stamp_ns / kNsPerSec));
  w.put(static_cast<uint32_t>(stamp_ns % kNsPerSec));
  w.string(frame_id);
}

// event_camera_msgs/msg/EventPacket: header, height, width, seq (u64), time_base 0, encoding "evt3", is_bigendian 0,
// events: a u8 sequence of the EVT 3.0 words (evt.h, Start::kFirstEvent) of the T1a times u >= 0, little-endian, so
// the packet decodes alone (§4.7). The caller bounds the word count (its byte count is a u32).
inline void put_event_packet_evt3(std::vector<char>& out, uint64_t seq, int64_t stamp_ns, std::string_view frame_id,
                                  uint32_t width, uint32_t height, const Events& e, const TimeMap& m) {
  Writer w(out);
  put_header(w, stamp_ns, frame_id);
  w.put(height);
  w.put(width);
  w.put(seq);
  w.put(uint64_t{0});  // time_base: unused for evt3 (§4.7)
  w.string("evt3");
  w.put(uint8_t{0});  // is_bigendian
  w.put(uint32_t{0});  // the byte count, set below
  const size_t at = out.size();
  // At most TIME_LOW, ADDR_Y and ADDR_X per event, and one TIME_HIGH per 4096 us step the events span (their times
  // are non-decreasing), written in place.
  const int64_t steps = e.n == 0 ? 0 : (t1a_us(m, e.t_ns[e.n - 1]) >> 12) - (t1a_us(m, e.t_ns[0]) >> 12) + 1;
  out.resize(at + 2 * (3 * e.n + static_cast<size_t>(steps)));
  struct Words {
    char* p;
    void push_back(uint16_t v) {
      std::memcpy(p, &v, 2);
      p += 2;
    }
  } words{out.data() + at};
  Evt3Encoder enc(Evt3Encoder::Start::kFirstEvent);
  for (size_t i = 0; i < e.n; ++i) enc.event(t1a_us(m, e.t_ns[i]), e.x[i], e.y[i], e.p[i], words);
  const auto n = static_cast<uint32_t>(words.p - (out.data() + at));
  out.resize(at + n);
  std::memcpy(out.data() + at - 4, &n, 4);
}

// sensor_msgs/msg/Imu: header; orientation (0, 0, 0, 1), the ROS 2 Quaternion's default, with orientation_covariance[0]
// = -1 (no estimate); the angular velocity and linear acceleration (float64, the published values) and their
// row-major covariances.
inline void put_imu(std::vector<char>& out, int64_t stamp_ns, std::string_view frame_id, const double* gyro,
                    const double* accel, const std::array<double, 9>& cov_gyro,
                    const std::array<double, 9>& cov_accel) {
  Writer w(out);
  put_header(w, stamp_ns, frame_id);
  static constexpr double kOrientation[4] = {0, 0, 0, 1};
  static constexpr double kNoOrientation[9] = {-1, 0, 0, 0, 0, 0, 0, 0, 0};
  w.array(kOrientation, 4);
  w.array(kNoOrientation, 9);
  w.array(gyro, 3);
  w.array(cov_gyro.data(), 9);
  w.array(accel, 3);
  w.array(cov_accel.data(), 9);
}

// sensor_msgs/msg/Image: header, height, width, encoding, is_bigendian 0, step, data (a u8 sequence). The caller
// bounds the data (its count is a u32).
inline void put_image(std::vector<char>& out, int64_t stamp_ns, std::string_view frame_id, const ros::Image& im) {
  Writer w(out);
  put_header(w, stamp_ns, frame_id);
  w.put(im.height);
  w.put(im.width);
  w.string(im.encoding);
  w.put(uint8_t{0});  // is_bigendian
  w.put(im.step);
  w.put(static_cast<uint32_t>(im.n));
  w.raw(im.data, im.n);
}

}  // namespace balla::exporter::cdr
