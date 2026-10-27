// The message streams of the ROS writers (docs/design/phase3/outputs.md §3.7, §4.7-4.8), shared by rosbag1 (ros1.cpp,
// A6) and MCAP (mcap.cpp, A7); moved from ros1.cpp in A7 unchanged in behaviour. Per stream (a rosbag1 connection or
// an MCAP channel): its events cut into packets of packet_ns on the T1a us grid (k = floor(u / packet_us)), each held
// back until an event of a later packet arrives or close(), so the bytes do not depend on how the events are batched
// (T3); its IMU samples one message each; its frames (A9) one sensor_msgs/Image each, stamped at the exposure
// midpoint (outputs.md Q5); every stamp T2 (exact ns: t_ns - 1000 origin_us + epoch_ns, within the format's range),
// header.stamp of an event message its first event's (as the Prophesee driver), the record or log time equal to
// header.stamp. A batch that breaks a rule is refused whole, before any of it is held.
//
// The Format policy serializes and stores the messages:
//   static constexpr std::string_view kName, kStream;  the refusals' prefix ("rosbag1") and stream word ("connection")
//   static constexpr std::string_view kRange;          the stamp range, as named in refusals
//   static constexpr std::string_view kCapacity;       why a packet holds at most packet_capacity() events
//   static bool in_range(int64_t stamp_ns);
//   static std::string_view type_name(MsgType);
//   uint64_t packet_capacity(const Stream&) const;     the most events a packet's message can take (0: none)
//   void write_events(const Stream&, int64_t stamp_ns, const Events&, const TimeMap&);   message s.messages
//   void write_imu(const Stream&, int64_t stamp_ns, const double* gyro, const double* accel,
//                  const std::array<double, 9>& cov_gyro, const std::array<double, 9>& cov_accel);
//   uint64_t image_capacity(const Stream&, size_t encoding_bytes) const;   the most data bytes an Image can take
//   void write_image(const Stream&, int64_t stamp_ns, const Image&);
// Header-only.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/time/time.h"
#include "export/errors.h"
#include "export/frame_map.h"
#include "export/msgdefs.h"
#include "export/sink.h"
#include "export/time_map.h"

namespace balla::exporter::ros {

// The longest packet: an EventPacket evt3 payload carries u mod 2^24, which its reader recovers only within a packet
// spanning less than 2^24 us (16.78 s, §4.7); EventArray packets keep the same bound.
inline constexpr int64_t kMaxPacketUs = int64_t{1} << 24;

// The most events one packet may hold: the room a message has for its events (Format::packet_capacity), over the most
// bytes an event can take. EventPacket evt3 words: at most TIME_LOW, ADDR_Y and ADDR_X per event (6 bytes), plus the
// packet's TIME_HIGH words, at most packet_us / 4096 + 2 as its times span less than packet_us.
inline uint64_t evt3_time_high_bytes(int64_t packet_us) { return 2 * (static_cast<uint64_t>(packet_us) / 4096 + 2); }

inline bool is_events(MsgType t) { return t == MsgType::kDvsEventArray || t == MsgType::kEventPacketEvt3; }

// One sensor_msgs/Image's fields after its header: is_bigendian is 0 (the data is little-endian).
struct Image {
  uint32_t height = 0, width = 0;
  std::string_view encoding;
  uint32_t step = 0;  // bytes per row
  const uint8_t* data = nullptr;
  size_t n = 0;  // height * step
};

// One stream and its held packet.
struct Stream {
  std::string topic;
  MsgType type;
  uint32_t handle;  // the container's
  bool configured = false;
  std::string frame_id;
  int32_t width = 0, height = 0;
  int64_t packet_us = 0;
  uint64_t capacity = 0;  // the most events one packet may hold (Format::packet_capacity)
  // The open packet (events of one T1a grid cell), held until an event of a later packet or close(), and the first
  // t_ns past it (packet_of).
  int64_t packet = 0, packet_end = 0;
  std::vector<uint16_t> x, y;
  std::vector<uint8_t> p;
  std::vector<int64_t> t;
  uint64_t messages = 0;  // the messages written: the next one's header.seq, EventPacket seq or MCAP sequence
  bool have_last = false;
  int64_t last_t = 0, count = 0, t_first = 0, t_last = 0;  // the items and their T2 stamps
};

template <class Format>
class Streams {
 public:
  Streams(const TimeMap& m, Format& f) : map_(m), format_(f) {}

  // Refused after close or for a topic already used (the writer checks the name rule first).
  void check_new(const std::string& topic) const {
    if (closed_) throw Refused(prefix() + std::string(Format::kStream) + " after close");
    for (const Stream& t : streams_) {
      if (t.topic == topic) throw Refused(prefix() + "topic " + topic + " is used twice");
    }
  }

  // A stream on a container handle (after check_new); returns its index.
  uint32_t add(const std::string& topic, MsgType type, uint32_t handle) {
    check_new(topic);
    Stream st;
    st.topic = topic;
    st.type = type;
    st.handle = handle;
    streams_.push_back(std::move(st));
    return static_cast<uint32_t>(streams_.size() - 1);
  }

  bool closed() const { return closed_; }

  // A stream's events in output order (t_ns non-decreasing across calls, each >= 1000 origin_us, x < w, y < h, p in
  // {0, 1}, every T2 stamp in the format's range), cut into messages of packet_ns (a multiple of 1000 in [1000, 1000
  // kMaxPacketUs]) on the T1a grid. frame_id, w, h and packet_ns are fixed by a stream's first call. w and h: [1,
  // 65536] for EventArray (u16 coordinates), [1, 2048] for EventPacket evt3 (11-bit addresses). A packet holds at most
  // its capacity, so a held packet always fits its message.
  void events(uint32_t conn, MsgType type, const std::string& frame_id, int32_t w, int32_t h, const Events& e,
              int64_t packet_ns) {
    Stream& s = stream(conn, type, "events");
    checked(s, frame_id, w, h, packet_ns, [&] {
      if (e.n == 0) return;
      if (e.x == nullptr || e.y == nullptr || e.p == nullptr || e.t_ns == nullptr) {
        throw std::invalid_argument(prefix() + "null event columns");
      }
      for (size_t i = 0; i < e.n; ++i) {
        if (e.x[i] >= s.width || e.y[i] >= s.height) {
          throw Refused(prefix() + s.topic + ": event (" + std::to_string(e.x[i]) + ", " + std::to_string(e.y[i]) +
                        ") outside the " + std::to_string(s.width) + " x " + std::to_string(s.height) + " sensor");
        }
        if (e.p[i] > 1) throw Refused(prefix() + "polarity must be 0 or 1, got " + std::to_string(e.p[i]));
      }
      check_times(s, e.t_ns, e.n, "event");
      // The packets the batch fills (the held one first) against their capacity, before any of it is held.
      uint64_t held = s.t.size();
      int64_t cell = s.packet, end = s.packet_end;
      for (size_t i = 0; i < e.n; ++i) {
        if (held == 0 || e.t_ns[i] >= end) {
          int64_t k = 0;
          packet_of(s, e.t_ns[i], &k, &end);
          if (k != cell) held = 0;
          cell = k;
        }
        if (++held > s.capacity) {
          throw Refused(prefix() + s.topic + ": the packet of t_ns " + std::to_string(e.t_ns[i]) +
                        " would hold more than " + std::to_string(s.capacity) + " events, " +
                        std::string(Format::kCapacity) + ": use shorter packets (--packet-us)");
        }
      }
    });
    if (e.n == 0) return;
    for (size_t i = 0; i < e.n; ++i) {
      if (s.t.empty() || e.t_ns[i] >= s.packet_end) {
        int64_t k = 0, end = 0;
        packet_of(s, e.t_ns[i], &k, &end);
        if (!s.t.empty() && k != s.packet) flush(s);
        s.packet = k;
        s.packet_end = end;
      }
      s.x.push_back(e.x[i]);
      s.y.push_back(e.y[i]);
      s.p.push_back(e.p[i]);
      s.t.push_back(e.t_ns[i]);
    }
    commit(s, e.t_ns, e.n);
  }

  // A stream's IMU samples in stamp order, one message each; covariances diag(white_var) (gyro, accel: finite, >= 0)
  // for kModel, zero for kZero. frame_id is fixed by the first call.
  void imu(uint32_t conn, const std::string& frame_id, const ImuView& v, CovarianceMode mode,
           std::array<double, 2> white_var) {
    Stream& st = stream(conn, MsgType::kImu, "imu");
    if (mode != CovarianceMode::kModel && mode != CovarianceMode::kZero) {
      throw Refused(prefix() + "unknown covariance mode");
    }
    for (const double w : white_var) {
      if (!(std::isfinite(w) && w >= 0)) throw Refused(prefix() + "white_var must be finite and >= 0");
    }
    checked(st, frame_id, 0, 0, 0, [&] {
      if (v.n == 0) return;
      if (v.t_ns == nullptr || v.gyro == nullptr || v.accel == nullptr) {
        throw std::invalid_argument(prefix() + "null IMU columns");
      }
      check_times(st, v.t_ns, v.n, "IMU sample");
    });
    if (v.n == 0) return;
    std::array<double, 9> cov_g{}, cov_a{};
    if (mode == CovarianceMode::kModel) {
      for (size_t k = 0; k < 9; k += 4) {
        cov_g[k] = white_var[0];
        cov_a[k] = white_var[1];
      }
    }
    for (size_t i = 0; i < v.n; ++i) {
      format_.write_imu(st, stamp(v.t_ns[i], "IMU sample"), v.gyro + 3 * i, v.accel + 3 * i, cov_g, cov_a);
      ++st.messages;
    }
    commit(st, v.t_ns, v.n);
  }

  // A stream's frames in time order (midpoints non-decreasing across calls, each frame's start >= 1000 origin_us,
  // every midpoint's T2 stamp in the format's range), one sensor_msgs/Image each at its midpoint's stamp: the pixels
  // as stored, in their own encoding (mono8, rgb8, mono16, 32FC1, 32FC3), under FrameMap::kNone, else mono8 or rgb8
  // through the map (frame_map.h). frame_id is fixed by the first call.
  void images(uint32_t conn, const std::string& frame_id, const FrameView& v, const FrameOptions& o) {
    Stream& st = stream(conn, MsgType::kImage, "images");
    const bool mapped = o.map != FrameMap::kNone;
    const std::string_view encoding =
        !mapped ? ros_encoding(v.format) : channels(v.format) == 3 ? std::string_view("rgb8") : "mono8";
    const uint64_t step = mapped ? static_cast<uint64_t>(v.width) * static_cast<uint64_t>(channels(v.format))
                                 : static_cast<uint64_t>(row_bytes(v));
    checked(st, frame_id, 0, 0, 0, [&] {
      exporter::check(v, o, prefix() + st.topic);
      if (v.n == 0) return;
      check_times(st, v.t_mid_ns, v.n, "frame");
      for (size_t k = 0; k < v.n; ++k) {
        if (t1a_us(map_, v.t_frame_start_ns[k]) < 0) {
          throw Refused(prefix() + st.topic + ": the frame start " + std::to_string(v.t_frame_start_ns[k]) +
                        " ns lies before the export origin " + std::to_string(map_.origin_us) + " us");
        }
      }
      const uint64_t cap = format_.image_capacity(st, encoding.size());
      if (step > std::numeric_limits<uint32_t>::max() || step * static_cast<uint64_t>(v.height) > cap) {
        throw Refused(prefix() + st.topic + ": a " + std::to_string(v.width) + " x " + std::to_string(v.height) +
                      " " + std::string(encoding) + " image is larger than a message holds");
      }
    });
    if (v.n == 0) return;
    std::vector<uint8_t> px;
    for (size_t k = 0; k < v.n; ++k) {
      if (mapped) {
        frame8(v, k, o, /*bgr=*/false, px);
      } else {
        frame_packed(v, k, px);
      }
      const Image im{static_cast<uint32_t>(v.height), static_cast<uint32_t>(v.width), encoding,
                     static_cast<uint32_t>(step), px.data(), px.size()};
      format_.write_image(st, stamp(v.t_mid_ns[k], "frame"), im);
      ++st.messages;
    }
    commit(st, v.t_mid_ns, v.n);
  }

  // Writes the held packets (stream order) and returns one StreamInfo per stream: {kind events, imu or frames, id the
  // topic, count (events, samples or frames), t_first, t_last (their T2 stamps, ns)}, camera left to the caller.
  // Nothing may follow.
  std::vector<StreamInfo> close() {
    if (closed_) throw Refused(prefix() + "closed twice");
    for (Stream& st : streams_) flush(st);
    closed_ = true;
    std::vector<StreamInfo> out;
    for (const Stream& st : streams_) {
      StreamInfo si;
      si.kind = st.type == MsgType::kImu ? "imu" : st.type == MsgType::kImage ? "frames" : "events";
      si.id = st.topic;
      si.count = st.count;
      si.t_first = st.t_first;
      si.t_last = st.t_last;
      out.push_back(std::move(si));
    }
    return out;
  }

 private:
  static std::string prefix() { return std::string(Format::kName) + ": "; }

  Stream& stream(uint32_t conn, MsgType type, const char* what) {
    if (closed_) throw Refused(prefix() + what + " after close");
    if (conn >= streams_.size()) {
      throw Refused(prefix() + "unknown " + std::string(Format::kStream) + " " + std::to_string(conn));
    }
    Stream& s = streams_[conn];
    if (s.type != type) {
      throw Refused(prefix() + std::string(Format::kStream) + " " + s.topic + " is " +
                    std::string(Format::type_name(s.type)) + ", not " + std::string(Format::type_name(type)));
    }
    return s;
  }

  // A stream's frame_id, size and packet length are fixed by its first call.
  void configure(Stream& s, const std::string& frame_id, int32_t w, int32_t h, int64_t packet_ns) {
    if (s.configured) {
      if (frame_id != s.frame_id || w != s.width || h != s.height || packet_ns != 1000 * s.packet_us) {
        throw Refused(prefix() + std::string(Format::kStream) + " " + s.topic +
                      ": frame_id, width, height and packet length are fixed by its first call");
      }
      return;
    }
    if (frame_id.size() > (size_t{1} << 20)) throw Refused(prefix() + "frame_id longer than 1 MiB");
    if (is_events(s.type)) {
      const int32_t max = s.type == MsgType::kEventPacketEvt3 ? 2048 : 65536;
      if (w < 1 || h < 1 || w > max || h > max) {
        throw Refused(prefix() + std::string(Format::type_name(s.type)) + " width and height must be in [1, " +
                      std::to_string(max) + "], got " + std::to_string(w) + " x " + std::to_string(h));
      }
      if (packet_ns <= 0 || packet_ns % 1000 != 0 || packet_ns / 1000 > kMaxPacketUs) {
        throw Refused(prefix() + "packet_ns must be a multiple of 1000 (the T1a us grid) in [1000, 1000 * 2^24] "
                      "(2^24 us = 16.78 s, the EventPacket evt3 bound), got " + std::to_string(packet_ns));
      }
      s.width = w;
      s.height = h;
      s.packet_us = packet_ns / 1000;
    }
    s.frame_id = frame_id;
    if (is_events(s.type)) s.capacity = format_.packet_capacity(s);
    s.configured = true;
  }

  // The T2 stamp of t_ns, Refused (naming it) unless the format's range holds it.
  int64_t stamp(int64_t t_ns, const char* what) const {
    int64_t ns = 0;
    if (!t2_ns(map_, t_ns, &ns) || !Format::in_range(ns)) {
      throw Refused(prefix() + "the " + std::string(what) + " at t_ns " + std::to_string(t_ns) +
                    " has the stamp t_ns - 1000 origin_us + epoch_ns outside " + std::string(Format::kRange) +
                    ": choose --epoch-ns or --t0/--t1");
    }
    return ns;
  }

  // Order and origin of a batch of times against the stream's last one, and the T2 range of its ends (T2 is monotone
  // in t, so the ends bound every stamp).
  void check_times(const Stream& s, const int64_t* t, size_t n, const char* what) const {
    int64_t prev = s.have_last ? s.last_t : std::numeric_limits<int64_t>::min();
    for (size_t i = 0; i < n; ++i) {
      if (t[i] < prev) {
        throw Refused(prefix() + s.topic + ": " + what + "s must be in time order: t_ns " + std::to_string(t[i]) +
                      " follows " + std::to_string(prev));
      }
      prev = t[i];
    }
    if (t1a_us(map_, t[0]) < 0) {
      throw Refused(prefix() + s.topic + ": t_ns " + std::to_string(t[0]) + " lies before the export origin " +
                    std::to_string(map_.origin_us) + " us");
    }
    (void)stamp(t[0], what);
    (void)stamp(t[n - 1], what);
  }

  void commit(Stream& s, const int64_t* t, size_t n) {
    if (s.count == 0) s.t_first = stamp(t[0], "item");
    s.t_last = stamp(t[n - 1], "item");
    s.last_t = t[n - 1];
    s.have_last = true;
    s.count += static_cast<int64_t>(n);
  }

  // Runs a call's checks after configuring the stream; a refused first call leaves it unconfigured, so a refused
  // batch changes nothing.
  template <class F>
  void checked(Stream& s, const std::string& frame_id, int32_t w, int32_t h, int64_t packet_ns, F&& checks) {
    const bool fresh = !s.configured;
    configure(s, frame_id, w, h, packet_ns);
    try {
      checks();
    } catch (...) {
      if (fresh) s.configured = false;
      throw;
    }
  }

  // The packet of t_ns (>= 1000 origin_us), k = floor(u / packet_us) on the T1a grid, and the first t_ns of the next
  // one, 1000 (origin_us + (k + 1) packet_us), saturated at INT64_MAX (only when the exact end passes it): the events
  // come in time order, so the callers divide only when an event reaches the end.
  void packet_of(const Stream& s, int64_t t_ns, int64_t* k, int64_t* end_ns) const {
    *k = time::floor_div(t1a_us(map_, t_ns), s.packet_us);
    int64_t u = 0;
    if (__builtin_mul_overflow(*k + 1, s.packet_us, &u) || __builtin_add_overflow(u, map_.origin_us, &u) ||
        __builtin_mul_overflow(u, int64_t{1000}, end_ns)) {
      *end_ns = std::numeric_limits<int64_t>::max();
    }
  }

  // Writes the open packet as one message (within its capacity, so the format accepts it).
  void flush(Stream& s) {
    if (s.t.empty()) return;
    const Events e{s.x.data(), s.y.data(), s.p.data(), s.t.data(), s.t.size()};
    format_.write_events(s, stamp(s.t.front(), "event"), e, map_);
    ++s.messages;
    s.x.clear();
    s.y.clear();
    s.p.clear();
    s.t.clear();
  }

  TimeMap map_;
  Format& format_;
  std::vector<Stream> streams_;
  bool closed_ = false;
};

}  // namespace balla::exporter::ros
