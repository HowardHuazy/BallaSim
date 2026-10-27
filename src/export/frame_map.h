// Frame streams of the exporters (docs/design/phase3/outputs.md §3.3, §4.7-4.8 "Frames"; P3-A A9): the view of a
// camera's frames (contract C7, frames.md §3.6) that the AEDAT4 (FRME) and ROS (sensor_msgs/Image) writers take, and
// the optional 8-bit map of their pixels.
//   FrameView     n frames of one PixelFormat, rows top-down with their strides, the ROI's position in the full
//                 image, and five times per frame on the camera's sensor clock: the exposure midpoint and the start
//                 and end of the frame and of its exposure (the driver maps the frames file's true-clock times
//                 through clock_c, §4.7). gray16 frames carry their significant bits (an APS's frames.aps.bits).
//   FrameMap      none (pixels as stored), linear8 or srgb8, per channel (§4.8): normalize (gray8 / 255, gray16 /
//                 (2^bits - 1), f32 as is), v' = clip(scale v, 0, 1) (NaN, a pixel with no lens preimage, gives 0),
//                 for srgb8 the sRGB OETF of IEC 61966-2-1, then round half to even of 255 v'. srgb8 is Mitsuba 3's
//                 Bitmap::convert to UInt8 with its sRGB ramp and pbrt-v4's PNG quantization (QuantizePixelsToU256,
//                 LinearToSRGB8) without its dither, i.e. Blender's "Standard" view transform on an sRGB display at
//                 exposure log2(scale); no tone mapping.
// gray16 normalizes by its significant bits, so an APS's 10-bit DN fills the 8-bit range (PHASE3.md §5.3: aps_dn
// exports to AEDAT4 through linear8). Header-only.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "export/errors.h"

namespace balla::exporter {

enum class PixelFormat : uint8_t { kGray8, kRgb8, kGray16, kGrayF32, kRgbF32 };  // F32: linear scene-referred
enum class FrameMap : uint8_t { kNone, kLinear8, kSrgb8 };

struct FrameOptions {
  FrameMap map = FrameMap::kNone;
  double scale = 1.0;  // finite, > 0; used by the map only
};

struct FrameView {
  const int64_t* t_mid_ns = nullptr;  // exposure midpoint (AEDAT4 timestamp, ROS header.stamp)
  const int64_t* t_frame_start_ns = nullptr;
  const int64_t* t_frame_end_ns = nullptr;
  const int64_t* t_exp_start_ns = nullptr;
  const int64_t* t_exp_end_ns = nullptr;
  PixelFormat format = PixelFormat::kGray8;
  const uint8_t* pixels = nullptr;  // frame k, row r at pixels + k frame_stride_bytes + r row_stride_bytes
  int64_t row_stride_bytes = 0, frame_stride_bytes = 0;
  int32_t width = 0, height = 0;
  size_t n = 0;
  int32_t bits = 0;            // gray16: the significant bits, 1-16 (0: 16)
  int32_t x0 = 0, y0 = 0;      // the ROI's top-left pixel in the full image
  int64_t exposure_ns = -1;    // the per-row exposure (AEDAT4 `exposure`); -1: t_exp_end - t_exp_start
};

inline int channels(PixelFormat f) { return f == PixelFormat::kRgb8 || f == PixelFormat::kRgbF32 ? 3 : 1; }
inline int sample_bytes(PixelFormat f) {
  return f == PixelFormat::kGray8 || f == PixelFormat::kRgb8 ? 1 : f == PixelFormat::kGray16 ? 2 : 4;
}
inline bool is_8bit(PixelFormat f) { return f == PixelFormat::kGray8 || f == PixelFormat::kRgb8; }
inline std::string_view format_name(PixelFormat f) {
  static constexpr std::string_view kNames[] = {"gray8", "rgb8", "gray16", "gray_f32", "rgb_f32"};
  return kNames[static_cast<size_t>(f)];
}
// The ROS sensor_msgs/Image encoding of a format as stored (image_encodings.h).
inline std::string_view ros_encoding(PixelFormat f) {
  static constexpr std::string_view kNames[] = {"mono8", "rgb8", "mono16", "32FC1", "32FC3"};
  return kNames[static_cast<size_t>(f)];
}
inline FrameMap parse_frame_map(std::string_view s) {
  if (s == "none") return FrameMap::kNone;
  if (s == "linear8") return FrameMap::kLinear8;
  if (s == "srgb8") return FrameMap::kSrgb8;
  throw Refused("frame map must be none, linear8 or srgb8, got '" + std::string(s) + "'");
}

// The bytes of one packed row of `width` pixels.
inline size_t row_bytes(const FrameView& v) {
  return static_cast<size_t>(v.width) * static_cast<size_t>(channels(v.format) * sample_bytes(v.format));
}

// Refused unless the view is well formed: a size of at least 1 x 1, strides that hold a row and a frame, gray16 bits
// in [0, 16], the options' scale finite and > 0, and per frame its times in order (frame start <= exposure start <=
// midpoint <= exposure end <= frame end) and its midpoint not before the previous frame's. `what` prefixes refusals.
inline void check(const FrameView& v, const FrameOptions& o, std::string_view what) {
  const std::string w(what);
  if (v.width < 1 || v.height < 1) {
    throw Refused(w + ": frames must be at least 1 x 1, got " + std::to_string(v.width) + " x " +
                  std::to_string(v.height));
  }
  if (v.row_stride_bytes < static_cast<int64_t>(row_bytes(v)) ||
      v.frame_stride_bytes < (v.height - 1) * v.row_stride_bytes + static_cast<int64_t>(row_bytes(v))) {
    throw Refused(w + ": frame strides too short for the frame size");
  }
  if (v.bits < 0 || v.bits > 16) throw Refused(w + ": gray16 bits must be in [1, 16] (0: 16)");
  if (!(std::isfinite(o.scale) && o.scale > 0)) throw Refused(w + ": the frame scale must be finite and > 0");
  if (v.n == 0) return;
  if (v.pixels == nullptr || v.t_mid_ns == nullptr || v.t_frame_start_ns == nullptr || v.t_frame_end_ns == nullptr ||
      v.t_exp_start_ns == nullptr || v.t_exp_end_ns == nullptr) {
    throw Refused(w + ": null frame columns");
  }
  for (size_t k = 0; k < v.n; ++k) {
    if (!(v.t_frame_start_ns[k] <= v.t_exp_start_ns[k] && v.t_exp_start_ns[k] <= v.t_mid_ns[k] &&
          v.t_mid_ns[k] <= v.t_exp_end_ns[k] && v.t_exp_end_ns[k] <= v.t_frame_end_ns[k])) {
      throw Refused(w + ": frame " + std::to_string(k) + "'s times are out of order (frame start <= exposure start "
                    "<= midpoint <= exposure end <= frame end)");
    }
    if (k > 0 && v.t_mid_ns[k] < v.t_mid_ns[k - 1]) {
      throw Refused(w + ": frames must be in time order: midpoint " + std::to_string(v.t_mid_ns[k]) + " follows " +
                    std::to_string(v.t_mid_ns[k - 1]));
    }
  }
}

namespace detail {

// One channel value in [0, 1] (before the scale): gray8 / 255, gray16 / (2^bits - 1), f32 as stored.
inline double normalized(const FrameView& v, const uint8_t* p) {
  switch (v.format) {
    case PixelFormat::kGray8:
    case PixelFormat::kRgb8: return *p / 255.0;
    case PixelFormat::kGray16: {
      uint16_t x;
      std::memcpy(&x, p, 2);
      return x / (std::ldexp(1.0, v.bits == 0 ? 16 : v.bits) - 1.0);
    }
    default: {
      float x;
      std::memcpy(&x, p, 4);
      return x;
    }
  }
}

}  // namespace detail

// §4.8: the 8-bit value of a normalized channel value under the map (linear8 or srgb8) and scale.
inline uint8_t map8(double v, FrameMap m, double scale) {
  double x = scale * v;
  if (!(x > 0)) x = 0;  // NaN (no lens preimage) and negatives
  if (x > 1) x = 1;
  if (m == FrameMap::kSrgb8) x = x <= 0.0031308 ? 12.92 * x : 1.055 * std::pow(x, 1.0 / 2.4) - 0.055;
  return static_cast<uint8_t>(std::nearbyint(255.0 * x));  // the default rounding mode: half to even
}

// Frame k as packed 8-bit pixels, rows top-down (`out` resized): gray to one channel, rgb to three in RGB order, or
// BGR when bgr (OpenCV's order, AEDAT4). With FrameMap::kNone the format must be gray8 or rgb8 (copied).
inline void frame8(const FrameView& v, size_t k, const FrameOptions& o, bool bgr, std::vector<uint8_t>& out) {
  const int c = channels(v.format), sb = sample_bytes(v.format);
  if (o.map == FrameMap::kNone && !is_8bit(v.format)) {
    throw std::logic_error("frame8: an unmapped frame that is not 8-bit");
  }
  out.resize(static_cast<size_t>(v.width) * static_cast<size_t>(v.height) * static_cast<size_t>(c));
  uint8_t* w = out.data();
  for (int32_t r = 0; r < v.height; ++r) {
    const uint8_t* row = v.pixels + static_cast<int64_t>(k) * v.frame_stride_bytes + r * v.row_stride_bytes;
    for (int32_t x = 0; x < v.width; ++x, w += c) {
      const uint8_t* px = row + static_cast<size_t>(x) * static_cast<size_t>(c * sb);
      for (int ch = 0; ch < c; ++ch) {
        const uint8_t* s = px + (bgr && c == 3 ? 2 - ch : ch) * sb;
        w[ch] = o.map == FrameMap::kNone ? *s : map8(detail::normalized(v, s), o.map, o.scale);
      }
    }
  }
}

// Frame k packed as stored (rows top-down, no padding, little-endian samples): the data of a ROS image with the
// format's own encoding.
inline void frame_packed(const FrameView& v, size_t k, std::vector<uint8_t>& out) {
  const size_t rb = row_bytes(v);
  out.resize(rb * static_cast<size_t>(v.height));
  for (int32_t r = 0; r < v.height; ++r) {
    std::memcpy(out.data() + static_cast<size_t>(r) * rb,
                v.pixels + static_cast<int64_t>(k) * v.frame_stride_bytes + r * v.row_stride_bytes, rb);
  }
}

}  // namespace balla::exporter
