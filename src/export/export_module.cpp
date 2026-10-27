// ballasim._export: the format exporters (docs/dev/EXPORT.md; docs/design/phase3/outputs.md §3.3). Optional: built
// when liblz4, libzstd and libbz2 are all found; build_info()['export'] reports it. A0 bound the block codecs, the
// CRC-32 (also the Python oracles' reference) and the refusal translator; A5 the RAW and DAT writers, the IMU CSV
// sidecar and the cameras' export windows; A6 the rosbag1 writer and the ROS1 message definitions; A7 the MCAP writer
// and the ROS 2 schemas; A8 the AEDAT4 writer and decompress_upto; A9 the frame streams of the AEDAT4, rosbag1 and MCAP
// writers, the 8-bit frame map and the Kalibr files.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "bindings/export_errors.h"
#include "bindings/imu_values.h"
#include "bindings/ndarray_util.h"
#include "export/aedat4.h"
#include "export/compress.h"
#include "export/crc32.h"
#include "export/errors.h"
#include "export/frame_map.h"
#include "export/imu_csv.h"
#include "export/kalibr.h"
#include "export/mcap.h"
#include "export/msgdefs.h"
#include "export/raw_writer.h"
#include "export/ros1.h"
#include "export/time_map.h"

namespace nb = nanobind;
using namespace balla;
using exporter::Codec;

namespace {

// Writer inputs: C-contiguous CPU arrays of the exact dtype (bound with noconvert), so a wrong dtype is a TypeError,
// never a silent cast (h5py's reads of the C4 and C5 columns have these dtypes).
template <class T>
using Column = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <class T>
using Rows3 = nb::ndarray<const T, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

exporter::Lz4Frame lz4_frame(const std::string& name) {
  if (name == "standard") return exporter::Lz4Frame::kStandard;
  if (name == "roslz4") return exporter::Lz4Frame::kRoslz4;
  throw exporter::Refused("unknown lz4 frame '" + name + "': standard or roslz4");
}

exporter::RawFormat raw_format(const std::string& name) {
  if (name == "evt2") return exporter::RawFormat::kEvt2;
  if (name == "evt21") return exporter::RawFormat::kEvt21;
  if (name == "evt3") return exporter::RawFormat::kEvt3;
  throw exporter::Refused("unknown RAW format '" + name + "': evt2, evt21 or evt3");
}

exporter::Evt2TimeHigh evt2_time_high(const std::string& name) {
  if (name == "openeb") return exporter::Evt2TimeHigh::kOpenEb;
  if (name == "change") return exporter::Evt2TimeHigh::kOnChange;
  throw exporter::Refused("unknown evt2_time_high '" + name + "': openeb or change");
}

exporter::RawOptions raw_options(const std::string& time_high, int32_t rem_min, int32_t rem_max,
                                 const std::string& version) {
  exporter::RawOptions o;
  o.evt2_time_high = evt2_time_high(time_high);
  o.t_ns_rem_min = rem_min;
  o.t_ns_rem_max = rem_max;
  o.version = version;
  return o;
}

exporter::MsgType msg_type(const std::string& name) {
  if (name == "dvs") return exporter::MsgType::kDvsEventArray;
  if (name == "evt3") return exporter::MsgType::kEventPacketEvt3;
  if (name == "imu") return exporter::MsgType::kImu;
  if (name == "image") return exporter::MsgType::kImage;
  throw exporter::Refused("unknown message type '" + name + "': dvs, evt3, imu or image");
}

exporter::CovarianceMode covariance_mode(const std::string& name) {
  if (name == "model") return exporter::CovarianceMode::kModel;
  if (name == "zero") return exporter::CovarianceMode::kZero;
  throw exporter::Refused("unknown covariance '" + name + "': model or zero");
}

nb::bytes to_bytes(const std::vector<char>& v) { return nb::bytes(v.data(), v.size()); }

nb::dict info_dict(const exporter::FileInfo& f) {
  nb::dict d;
  d["path"] = f.path;
  d["bytes"] = f.bytes;
  d["sha256"] = f.sha256;
  nb::list streams;
  for (const exporter::StreamInfo& s : f.streams) {
    nb::dict sd;
    sd["camera"] = s.camera;
    sd["kind"] = s.kind;
    sd["id"] = s.id;
    sd["count"] = s.count;
    sd["t_first"] = s.t_first;
    sd["t_last"] = s.t_last;
    streams.append(sd);
  }
  d["streams"] = streams;
  return d;
}

exporter::Events events_of(const Column<uint16_t>& x, const Column<uint16_t>& y, const Column<uint8_t>& p,
                            const Column<int64_t>& t_ns) {
  const size_t n = t_ns.shape(0);
  if (x.shape(0) != n || y.shape(0) != n || p.shape(0) != n) {
    throw exporter::Refused("append: x, y, p and t_ns must have one length");
  }
  return {x.data(), y.data(), p.data(), t_ns.data(), n};
}

// A writer behind a mutex. Its calls release the GIL first and then take the lock, so a thread that waits for the
// writer never holds the GIL (as the sensor and IMU bindings do); the arrays stay referenced by the call's arguments.
template <class W>
struct Guarded {
  template <class... A>
  explicit Guarded(std::in_place_t, A&&... a) : w(std::forward<A>(a)...) {}
  Guarded(const Guarded&) = delete;
  Guarded& operator=(const Guarded&) = delete;
  template <class F>
  auto locked(F&& f) {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex);
    return f(w);
  }
  std::mutex mutex;
  W w;
};
using PyRawWriter = Guarded<exporter::RawWriter>;
using PyDatWriter = Guarded<exporter::DatWriter>;
using PyImuCsvWriter = Guarded<exporter::ImuCsvWriter>;
using PyRosbag1Writer = Guarded<exporter::Rosbag1Writer>;
using PyMcapWriter = Guarded<exporter::McapWriter>;
using PyAedat4Writer = Guarded<exporter::Aedat4Writer>;

exporter::ImuView imu_view(const Column<int64_t>& t_ns, const Rows3<double>& gyro, const Rows3<double>& accel,
                           const std::optional<Rows3<int32_t>>& counts_gyro,
                           const std::optional<Rows3<int32_t>>& counts_accel) {
  const size_t n = t_ns.shape(0);
  if (gyro.shape(0) != n || accel.shape(0) != n || (counts_gyro && counts_gyro->shape(0) != n) ||
      (counts_accel && counts_accel->shape(0) != n)) {
    throw exporter::Refused("append: t_ns, gyro, accel and the counts must have one length");
  }
  return {t_ns.data(), gyro.data(), accel.data(), counts_gyro ? counts_gyro->data() : nullptr,
          counts_accel ? counts_accel->data() : nullptr, n};
}

// Any C-contiguous CPU array: the frames' pixels, typed by frame_call.
using Pixels = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;

// One call's frames (A9): five int64 time columns of n entries on the sensor clock, and the pixels, (n, H, W) uint8,
// uint16 or float32 (gray8, gray16, gray f32) or (n, H, W, 3) uint8 or float32 (rgb8, rgb f32), rows top-down.
struct FrameCall {
  exporter::FrameView view;
  exporter::FrameOptions options;
};

// The view's pixels, format, size, strides and count from the array.
void set_pixels(exporter::FrameView& v, const Pixels& px) {
  using exporter::PixelFormat;
  const bool rgb = px.ndim() == 4;
  if (!(px.ndim() == 3 || (rgb && px.shape(3) == 3))) {
    throw exporter::Refused("frames: pixels must be (n, H, W) or (n, H, W, 3)");
  }
  PixelFormat f;
  if (px.dtype() == nb::dtype<uint8_t>()) {
    f = rgb ? PixelFormat::kRgb8 : PixelFormat::kGray8;
  } else if (px.dtype() == nb::dtype<uint16_t>() && !rgb) {
    f = PixelFormat::kGray16;
  } else if (px.dtype() == nb::dtype<float>()) {
    f = rgb ? PixelFormat::kRgbF32 : PixelFormat::kGrayF32;
  } else {
    throw exporter::Refused("frames: pixels must be uint8, uint16 (gray only) or float32");
  }
  if (px.shape(1) > INT32_MAX || px.shape(2) > INT32_MAX) throw exporter::Refused("frames: frames too large");
  v.format = f;
  v.pixels = static_cast<const uint8_t*>(px.data());
  v.width = static_cast<int32_t>(px.shape(2));
  v.height = static_cast<int32_t>(px.shape(1));
  v.row_stride_bytes = static_cast<int64_t>(exporter::row_bytes(v));
  v.frame_stride_bytes = v.row_stride_bytes * v.height;
  v.n = px.shape(0);
}

FrameCall frame_call(const Column<int64_t>& t_mid, const Column<int64_t>& t_frame_start,
                     const Column<int64_t>& t_frame_end, const Column<int64_t>& t_exp_start,
                     const Column<int64_t>& t_exp_end, const Pixels& px, int32_t bits, int32_t x0, int32_t y0,
                     int64_t exposure_ns, const std::string& frame_map, double scale) {
  FrameCall c;
  exporter::FrameView& v = c.view;
  set_pixels(v, px);
  const size_t n = v.n;
  if (t_mid.shape(0) != n || t_frame_start.shape(0) != n || t_frame_end.shape(0) != n ||
      t_exp_start.shape(0) != n || t_exp_end.shape(0) != n) {
    throw exporter::Refused("frames: the five time columns and the pixels must have one count of frames");
  }
  v.t_mid_ns = t_mid.data();
  v.t_frame_start_ns = t_frame_start.data();
  v.t_frame_end_ns = t_frame_end.data();
  v.t_exp_start_ns = t_exp_start.data();
  v.t_exp_end_ns = t_exp_end.data();
  v.bits = bits;
  v.x0 = x0;
  v.y0 = y0;
  v.exposure_ns = exposure_ns;
  c.options = {exporter::parse_frame_map(frame_map), scale};
  return c;
}

#define BALLA_FRAME_ARGS                                                                                         \
  nb::arg("t_mid_ns").noconvert(), nb::arg("t_frame_start_ns").noconvert(), nb::arg("t_frame_end_ns").noconvert(), \
      nb::arg("t_exp_start_ns").noconvert(), nb::arg("t_exp_end_ns").noconvert(), nb::arg("pixels").noconvert(),   \
      nb::kw_only(), nb::arg("bits") = 0, nb::arg("x0") = 0, nb::arg("y0") = 0, nb::arg("exposure_ns") = -1,      \
      nb::arg("frame_map") = "none", nb::arg("scale") = 1.0

constexpr const char* kFramesDoc =
    "Frames in time order (A9): int64 t_mid_ns, t_frame_start_ns, t_frame_end_ns, t_exp_start_ns and t_exp_end_ns "
    "on the camera's sensor clock (the exposure midpoint, the start and end of the frame and of the exposure), each "
    "frame's start >= 1000 origin_us; pixels (n, H, W) uint8 (gray8), uint16 (gray16 of `bits` significant bits, 0: "
    "16) or float32 (linear gray), or (n, H, W, 3) uint8 or float32 (rgb), rows top-down; x0, y0: the ROI's top-left "
    "pixel; exposure_ns: the per-row exposure (-1: t_exp_end - t_exp_start); frame_map none, linear8 or srgb8 with "
    "scale (outputs.md §4.8). A batch that breaks a rule is refused whole (ExportError).";

constexpr const char* kEventsDoc =
    "Events in output order: uint16 x, uint16 y, uint8 p (0/1) and int64 t_ns (t_reported on the camera's sensor "
    "clock), 1-D C-contiguous arrays of these exact dtypes; t_ns non-decreasing within and across calls and >= "
    "1000 origin_us, x < width, y < height, t_ns mod 1000 in the declared t_ns_rem range. A batch that breaks any "
    "is refused whole (ExportError); the GIL is released while it encodes.";
constexpr const char* kCloseDoc =
    "Flushes, closes and hashes the file: {path, bytes, sha256, streams: [{camera, kind, id, count, t_first, "
    "t_last}]} with t_first and t_last in u (us after the origin, T1a); camera and id are the caller's to fill.";

}  // namespace

NB_MODULE(_export, m) {
  m.doc() = "BallaSim format exporters (docs/dev/EXPORT.md): block codecs and CRC-32, the RAW (EVT 2.0, 2.1, 3.0) and "
            "DAT writers, the IMU CSV sidecar, the rosbag1, MCAP and AEDAT4 writers with their frame streams, the "
            "Kalibr files; refusals raise ballasim.export.errors.ExportError";
  bindings::register_export_error<exporter::Refused>();
  bindings::register_export_error<imu::Refused>();  // the Kalibr values' IMU parameters
  m.attr("CODECS") = std::vector<std::string>{"none", "lz4", "lz4-high", "zstd", "zstd-high", "bz2"};
  m.def(
      "compress",
      [](nb::bytes data, const std::string& codec, const std::string& frame) {
        const Codec c = exporter::parse_codec(codec);
        const exporter::Lz4Frame f = lz4_frame(frame);
        const char* p = data.c_str();  // C-API calls: read with the GIL held
        const size_t n = data.size();
        std::vector<char> out;
        {
          nb::gil_scoped_release release;  // bytes are immutable and held by the caller
          out = exporter::compress(c, p, n, f);
        }
        return to_bytes(out);
      },
      nb::arg("data"), nb::arg("codec"), nb::arg("lz4_frame") = "standard",
      "One self-contained frame of data with a codec of CODECS (lz4: frame level 0, lz4-high 9, zstd level 3, "
      "zstd-high the maximum, bz2 block 9); lz4_frame 'standard' (64 KB linked blocks) or 'roslz4' (1 MB independent "
      "blocks with a content checksum, FLG 0x64 BD 0x60). Fresh library state per call.");
  m.def(
      "decompress",
      [](nb::bytes data, const std::string& codec, size_t size) {
        const Codec c = exporter::parse_codec(codec);
        const char* p = data.c_str();  // C-API calls: read with the GIL held
        const size_t n = data.size();
        std::vector<char> out;
        {
          nb::gil_scoped_release release;  // bytes are immutable and held by the caller
          out = exporter::decompress(c, p, n, size);
        }
        return to_bytes(out);
      },
      nb::arg("data"), nb::arg("codec"), nb::arg("size"),
      "The inverse of compress: size is the exact decompressed size; ExportError on corrupt input or another size.");
  m.def(
      "decompress_upto",
      [](nb::bytes data, const std::string& codec, size_t max_size) {
        const Codec c = exporter::parse_codec(codec);
        const char* p = data.c_str();  // C-API calls: read with the GIL held
        const size_t n = data.size();
        std::vector<char> out;
        {
          nb::gil_scoped_release release;  // bytes are immutable and held by the caller
          out = exporter::decompress_upto(c, p, n, max_size);
        }
        return to_bytes(out);
      },
      nb::arg("data"), nb::arg("codec"), nb::arg("max_size"),
      "As decompress, for containers that do not record the decompressed size (AEDAT4): exactly one frame of any size "
      "up to max_size; ExportError on corrupt input or more.");
  m.def(
      "crc32", [](nb::bytes data, uint32_t crc) { return exporter::crc32(crc, data.c_str(), data.size()); },
      nb::arg("data"), nb::arg("crc") = 0,
      "CRC-32 (zlib's, as MCAP uses it) of data, continuing crc: crc32(b, crc32(a)) == crc32(a + b).");

  m.def(
      "camera_times",
      [](int64_t clock_offset_ns, double clock_drift_ppm, double readout_delay_s, double readout_quantum_s,
         int64_t t_end_ns, std::optional<int64_t> t0_ns, std::optional<int64_t> t1_ns) {
        const exporter::CameraTimes c = exporter::camera_times(clock_offset_ns, clock_drift_ppm, readout_delay_s,
                                                               readout_quantum_s, t_end_ns, t0_ns, t1_ns);
        nb::dict d;
        nb::list window;
        window.append(c.lo_ns ? nb::cast(*c.lo_ns) : nb::none());
        window.append(c.hi_ns ? nb::cast(*c.hi_ns) : nb::none());
        d["window_reported_ns"] = window;
        d["file_end_bound_ns"] = c.file_end_bound_ns;
        return d;
      },
      nb::arg("clock_offset_ns"), nb::arg("clock_drift_ppm"), nb::arg("readout_delay_s"), nb::arg("readout_quantum_s"),
      nb::arg("t_end_ns"), nb::arg("t0_ns") = nb::none(), nb::arg("t1_ns") = nb::none(),
      "A camera's export window on its sensor clock clock_c(t) = t + offset + llround(drift 1e-6 t) (outputs.md "
      "§4.7) from its recorded sensor values: {window_reported_ns: [clock_c(t0) or None, clock_c(t1) or None], "
      "file_end_bound_ns: clock_c(t_end + d0 + q)}, the bound below which the run's event writer closed the file "
      "(d0, q: the readout delay and quantum rounded to ns as the sensor rounds them).");

  nb::class_<PyRawWriter>(m, "RawWriter",
                          "A Prophesee RAW file of one camera's events (EVT 2.0, 2.1 or 3.0; outputs.md §3.7, §4.8), "
                          "T1a times: u = floor(t_ns / 1000) - origin_us.")
      .def(
          "__init__",
          [](PyRawWriter* self, const std::string& path, const std::string& format, int32_t width, int32_t height,
             int64_t origin_us, int64_t epoch_ns, const std::string& time_high, int32_t rem_min, int32_t rem_max,
             const std::string& version) {
            const exporter::TimeMap time_map{origin_us, epoch_ns};
            new (self) PyRawWriter(std::in_place, path, raw_format(format), width, height, time_map,
                                   raw_options(time_high, rem_min, rem_max, version));
          },
          nb::arg("path"), nb::arg("format"), nb::arg("width"), nb::arg("height"), nb::arg("origin_us"),
          nb::arg("epoch_ns"), nb::arg("evt2_time_high") = "openeb", nb::arg("t_ns_rem_min") = 0,
          nb::arg("t_ns_rem_max") = 0, nb::arg("version") = "",
          "format: evt2, evt21 or evt3; width and height <= 2048; epoch_ns a multiple of 1000 (recorded in the "
          "header with the origin); evt2_time_high: openeb (a TIME_HIGH every 16 us, OpenEB's encoder) or change "
          "(EVT 2.0 only; EVT 2.1 is always on change); [t_ns_rem_min, t_ns_rem_max]: the window's pre-scanned "
          "sub-us parts (-1, -1: no events); version: the header's 'plugin_version BallaSim <version>'.")
      .def(
          "append",
          [](PyRawWriter& s, const Column<uint16_t>& x, const Column<uint16_t>& y, const Column<uint8_t>& p,
             const Column<int64_t>& t_ns) {
            const exporter::Events e = events_of(x, y, p, t_ns);
            s.locked([&](exporter::RawWriter& w) { w.append(e); });
          },
          nb::arg("x").noconvert(), nb::arg("y").noconvert(), nb::arg("p").noconvert(), nb::arg("t_ns").noconvert(),
          kEventsDoc)
      .def(
          "close", [](PyRawWriter& s) { return info_dict(s.locked([](exporter::RawWriter& w) { return w.close(); })); },
          kCloseDoc);

  nb::class_<PyDatWriter>(m, "DatWriter",
                          "A Prophesee DAT file of one camera's events (outputs.md §3.7, §4.8): u32 us under T1a, "
                          "refused from u = 2^32 us on.")
      .def(
          "__init__",
          [](PyDatWriter* self, const std::string& path, int32_t width, int32_t height, int64_t origin_us,
             int64_t epoch_ns, int32_t rem_min, int32_t rem_max, const std::string& version) {
            new (self) PyDatWriter(std::in_place, path, width, height, exporter::TimeMap{origin_us, epoch_ns},
                                   raw_options("openeb", rem_min, rem_max, version));
          },
          nb::arg("path"), nb::arg("width"), nb::arg("height"), nb::arg("origin_us"), nb::arg("epoch_ns"),
          nb::arg("t_ns_rem_min") = 0, nb::arg("t_ns_rem_max") = 0, nb::arg("version") = "",
          "As RawWriter, with width and height <= 16384; the version is not written (DAT has no plugin line).")
      .def(
          "append",
          [](PyDatWriter& s, const Column<uint16_t>& x, const Column<uint16_t>& y, const Column<uint8_t>& p,
             const Column<int64_t>& t_ns) {
            const exporter::Events e = events_of(x, y, p, t_ns);
            s.locked([&](exporter::DatWriter& w) { w.append(e); });
          },
          nb::arg("x").noconvert(), nb::arg("y").noconvert(), nb::arg("p").noconvert(), nb::arg("t_ns").noconvert(),
          kEventsDoc)
      .def(
          "close", [](PyDatWriter& s) { return info_dict(s.locked([](exporter::DatWriter& w) { return w.close(); })); },
          kCloseDoc);

  nb::class_<PyImuCsvWriter>(m, "ImuCsvWriter",
                             "The IMU sidecar <cam>.imu.csv of the RAW and DAT exports (outputs.md §3.7): t_us (T1a), "
                             "the counts of the quantized sensors, then the published values as %.17g.")
      .def(
          "__init__",
          [](PyImuCsvWriter* self, const std::string& path, int64_t origin_us, int64_t epoch_ns, bool counts_gyro,
             bool counts_accel) {
            new (self)
                PyImuCsvWriter(std::in_place, path, exporter::TimeMap{origin_us, epoch_ns}, counts_gyro, counts_accel);
          },
          nb::arg("path"), nb::arg("origin_us"), nb::arg("epoch_ns"), nb::arg("counts_gyro"), nb::arg("counts_accel"),
          "counts_gyro, counts_accel: whether the rows carry that sensor's counts (its lsb is set).")
      .def(
          "append",
          [](PyImuCsvWriter& s, const Column<int64_t>& t_ns, const Rows3<double>& gyro, const Rows3<double>& accel,
             std::optional<Rows3<int32_t>> counts_gyro, std::optional<Rows3<int32_t>> counts_accel) {
            const exporter::ImuView v = imu_view(t_ns, gyro, accel, counts_gyro, counts_accel);
            s.locked([&](exporter::ImuCsvWriter& w) { w.append(v); });
          },
          nb::arg("t_ns").noconvert(), nb::arg("gyro").noconvert(), nb::arg("accel").noconvert(),
          nb::arg("counts_gyro").noconvert() = nb::none(), nb::arg("counts_accel").noconvert() = nb::none(),
          "C5 samples in stamp order: int64 t_ns (>= 1000 origin_us), float64 gyro and accel (N, 3), int32 counts "
          "(N, 3) exactly for the sensors the file has counts for; refused whole otherwise; GIL released.")
      .def(
          "close",
          [](PyImuCsvWriter& s) { return info_dict(s.locked([](exporter::ImuCsvWriter& w) { return w.close(); })); },
          kCloseDoc);
  m.def(
      "ros1_msgdefs",
      []() {
        nb::dict d;
        using Named = std::pair<const char*, exporter::MsgType>;
        for (const auto& [name, t] : {Named{"dvs", exporter::MsgType::kDvsEventArray},
                                      Named{"evt3", exporter::MsgType::kEventPacketEvt3},
                                      Named{"imu", exporter::MsgType::kImu},
                                      Named{"image", exporter::MsgType::kImage}}) {
          const exporter::Ros1MsgDef& def = exporter::ros1_msgdef(t);
          nb::dict e;
          e["type"] = std::string(def.type);
          e["md5"] = std::string(def.md5);
          e["definition"] = std::string(def.definition);
          d[name] = e;
        }
        return d;
      },
      "The ROS1 message types the rosbag1 writer uses, by its message names (dvs, evt3, imu, image): {type, md5, "
      "definition}, the definition being genmsg's full text (src/export/msgdefs.h).");

  nb::class_<PyRosbag1Writer>(m, "Rosbag1Writer",
                              "One ROS1 bag (rosbag format 2.0) of a run's cameras (outputs.md §3.7, §4.7-4.8): "
                              "event messages per packet on the T1a us grid, IMU messages per sample, T2 stamps (ns: "
                              "t_ns - 1000 origin_us + epoch_ns, u32 seconds), record time = header.stamp.")
      .def(
          "__init__",
          [](PyRosbag1Writer* self, const std::string& path, const std::string& compression, int64_t origin_us,
             int64_t epoch_ns, int threads, uint32_t chunk_bytes) {
            new (self) PyRosbag1Writer(std::in_place, path, exporter::parse_codec(compression), chunk_bytes,
                                       exporter::TimeMap{origin_us, epoch_ns}, threads);
          },
          nb::arg("path"), nb::arg("compression"), nb::arg("origin_us"), nb::arg("epoch_ns"), nb::arg("threads") = 4,
          nb::arg("chunk_bytes") = exporter::ros1::kChunkThreshold,
          "compression: none, bz2 or lz4 (the roslz4 frame); epoch_ns a multiple of 1000; threads (1-256) compress "
          "whole chunks, the bytes not depending on them; chunk_bytes: rosbag's chunk threshold (768 KiB).")
      .def(
          "connection",
          [](PyRosbag1Writer& s, const std::string& topic, const std::string& msg) {
            const exporter::MsgType t = msg_type(msg);
            return s.locked([&](exporter::Rosbag1Writer& w) { return w.connection(topic, t); });
          },
          nb::arg("topic"), nb::arg("msg"),
          "A connection: msg dvs (dvs_msgs/EventArray), evt3 (event_camera_msgs/EventPacket, evt3), imu "
          "(sensor_msgs/Imu) or image (sensor_msgs/Image); topic by ROS's name rule, each once. Returns its handle; "
          "its bag id is given at its first message (a connection never written stays out of the bag).")
      .def(
          "event_arrays",
          [](PyRosbag1Writer& s, uint32_t conn, const std::string& frame_id, int32_t width, int32_t height,
             const Column<uint16_t>& x, const Column<uint16_t>& y, const Column<uint8_t>& p,
             const Column<int64_t>& t_ns, int64_t packet_ns) {
            const exporter::Events e = events_of(x, y, p, t_ns);
            s.locked([&](exporter::Rosbag1Writer& w) { w.event_arrays(conn, frame_id, width, height, e, packet_ns); });
          },
          nb::arg("conn"), nb::arg("frame_id"), nb::arg("width"), nb::arg("height"), nb::arg("x").noconvert(),
          nb::arg("y").noconvert(), nb::arg("p").noconvert(), nb::arg("t_ns").noconvert(), nb::arg("packet_ns"),
          "A dvs connection's events (as RawWriter.append's, width and height <= 65536), one dvs_msgs/EventArray per "
          "packet_ns (a multiple of 1000, at most 1000 * 2^24) on the T1a grid, held back until the packet is "
          "complete; every event ts exact (T2). frame_id, width, height and packet_ns are fixed by the first call. A "
          "batch that would fill a packet past what its message record can take (4 GiB) is refused whole.")
      .def(
          "event_packets_evt3",
          [](PyRosbag1Writer& s, uint32_t conn, const std::string& frame_id, int32_t width, int32_t height,
             const Column<uint16_t>& x, const Column<uint16_t>& y, const Column<uint8_t>& p,
             const Column<int64_t>& t_ns, int64_t packet_ns) {
            const exporter::Events e = events_of(x, y, p, t_ns);
            s.locked(
                [&](exporter::Rosbag1Writer& w) { w.event_packets_evt3(conn, frame_id, width, height, e, packet_ns); });
          },
          nb::arg("conn"), nb::arg("frame_id"), nb::arg("width"), nb::arg("height"), nb::arg("x").noconvert(),
          nb::arg("y").noconvert(), nb::arg("p").noconvert(), nb::arg("t_ns").noconvert(), nb::arg("packet_ns"),
          "As event_arrays for an evt3 connection (width and height <= 2048): one event_camera_msgs/EventPacket per "
          "packet whose EVT 3.0 words start at its first event (u = T1a mod 2^24) and whose header.stamp is that "
          "event's T2 stamp; event k's time floored to the us is stamp - (stamp mod 1000) + 1000 (u_k - u_first).")
      .def(
          "imu",
          [](PyRosbag1Writer& s, uint32_t conn, const std::string& frame_id, const Column<int64_t>& t_ns,
             const Rows3<double>& gyro, const Rows3<double>& accel, const std::string& covariance,
             std::array<double, 2> white_var) {
            const exporter::ImuView v = imu_view(t_ns, gyro, accel, std::nullopt, std::nullopt);
            const exporter::CovarianceMode mode = covariance_mode(covariance);
            s.locked([&](exporter::Rosbag1Writer& w) { w.imu(conn, frame_id, v, mode, white_var); });
          },
          nb::arg("conn"), nb::arg("frame_id"), nb::arg("t_ns").noconvert(), nb::arg("gyro").noconvert(),
          nb::arg("accel").noconvert(), nb::arg("covariance") = "model", nb::arg("white_var") = std::array<double, 2>{},
          "An imu connection's C5 samples in stamp order: one sensor_msgs/Imu each (float64 values bit-exact, "
          "orientation zeros with orientation_covariance[0] = -1); covariance model: diag(white_var) (gyro, accel "
          "per-sample white variance, _imu.white_variance), zero: all zero (unknown).")
      .def(
          "images",
          [](PyRosbag1Writer& s, uint32_t conn, const std::string& frame_id, const Column<int64_t>& t_mid,
             const Column<int64_t>& t_fs, const Column<int64_t>& t_fe, const Column<int64_t>& t_es,
             const Column<int64_t>& t_ee, const Pixels& px, int32_t bits, int32_t x0, int32_t y0, int64_t exposure_ns,
             const std::string& frame_map, double scale) {
            const FrameCall c = frame_call(t_mid, t_fs, t_fe, t_es, t_ee, px, bits, x0, y0, exposure_ns, frame_map,
                                           scale);
            s.locked([&](exporter::Rosbag1Writer& w) { w.images(conn, frame_id, c.view, c.options); });
          },
          nb::arg("conn"), nb::arg("frame_id"), BALLA_FRAME_ARGS,
          "An image connection's frames: one sensor_msgs/Image each, header.stamp the midpoint's T2 stamp, the pixels "
          "as stored (mono8, rgb8, mono16, 32FC1, 32FC3) or, with a frame_map, mono8 or rgb8.")
      .def(
          "close",
          [](PyRosbag1Writer& s) { return info_dict(s.locked([](exporter::Rosbag1Writer& w) { return w.close(); })); },
          "Writes the held packets, the index section and the bag header; {path, bytes, sha256, streams: one per "
          "connection {kind events or imu, id the topic, count (events or samples), t_first, t_last (T2 ns)}}, "
          "camera left to the caller.");
  m.def(
      "ros2_msgdefs",
      []() {
        nb::dict d;
        using Named = std::pair<const char*, exporter::MsgType>;
        for (const auto& [name, t] : {Named{"evt3", exporter::MsgType::kEventPacketEvt3},
                                      Named{"imu", exporter::MsgType::kImu},
                                      Named{"image", exporter::MsgType::kImage}}) {
          const exporter::Ros2MsgDef& def = exporter::ros2_msgdef(t);
          nb::dict e;
          e["type"] = std::string(def.type);
          e["definition"] = std::string(def.definition);
          d[name] = e;
        }
        return d;
      },
      "The ROS 2 message types the MCAP writer uses, by its message names (evt3, imu, image): {type, definition}, "
      "the definition being the ros2msg schema text (src/export/msgdefs.h).");

  nb::class_<PyMcapWriter>(m, "McapWriter",
                           "One MCAP file (profile ros2, CDR messages) of a run's cameras (outputs.md §3.7, §4.7-4.8): "
                           "EventPacket evt3 messages per packet on the T1a us grid, sensor_msgs/msg/Imu per sample, "
                           "T2 stamps (ns: t_ns - 1000 origin_us + epoch_ns, [0, 2^31 s)), log time = publish time = "
                           "header.stamp.")
      .def(
          "__init__",
          [](PyMcapWriter* self, const std::string& path, const std::string& compression, int64_t origin_us,
             int64_t epoch_ns, int threads, uint64_t chunk_bytes, const std::string& library) {
            new (self) PyMcapWriter(std::in_place, path, exporter::parse_codec(compression), chunk_bytes,
                                    exporter::TimeMap{origin_us, epoch_ns}, threads, library);
          },
          nb::arg("path"), nb::arg("compression"), nb::arg("origin_us"), nb::arg("epoch_ns"), nb::arg("threads") = 4,
          nb::arg("chunk_bytes") = exporter::mcap::kChunkBytes, nb::arg("library") = "BallaSim",
          "compression: none, lz4 (LZ4 frame) or zstd (level 3); epoch_ns a multiple of 1000; threads (1-256) "
          "compress whole chunks, the bytes not depending on them; chunk_bytes: the chunk size (1 MiB; at most 2^32); "
          "library: the Header's ('BallaSim <version>').")
      .def(
          "channel",
          [](PyMcapWriter& s, const std::string& topic, const std::string& msg) {
            const exporter::MsgType t = msg_type(msg);
            return s.locked([&](exporter::McapWriter& w) { return w.channel(topic, t); });
          },
          nb::arg("topic"), nb::arg("msg"),
          "A channel: msg evt3 (event_camera_msgs/msg/EventPacket, evt3), imu (sensor_msgs/msg/Imu) or image "
          "(sensor_msgs/msg/Image); topic by ROS 2's topic name rule, each once. Returns its handle; its schema and "
          "channel records are written before its first message (a channel never written stays out of the file).")
      .def(
          "event_packets_evt3",
          [](PyMcapWriter& s, uint16_t ch, const std::string& frame_id, int32_t width, int32_t height,
             const Column<uint16_t>& x, const Column<uint16_t>& y, const Column<uint8_t>& p,
             const Column<int64_t>& t_ns, int64_t packet_ns) {
            const exporter::Events e = events_of(x, y, p, t_ns);
            s.locked([&](exporter::McapWriter& w) { w.event_packets_evt3(ch, frame_id, width, height, e, packet_ns); });
          },
          nb::arg("ch"), nb::arg("frame_id"), nb::arg("width"), nb::arg("height"), nb::arg("x").noconvert(),
          nb::arg("y").noconvert(), nb::arg("p").noconvert(), nb::arg("t_ns").noconvert(), nb::arg("packet_ns"),
          "As Rosbag1Writer.event_packets_evt3, in CDR: one EventPacket per packet_ns on the T1a grid whose EVT 3.0 "
          "words start at its first event and whose header.stamp is that event's T2 stamp.")
      .def(
          "imu",
          [](PyMcapWriter& s, uint16_t ch, const std::string& frame_id, const Column<int64_t>& t_ns,
             const Rows3<double>& gyro, const Rows3<double>& accel, const std::string& covariance,
             std::array<double, 2> white_var) {
            const exporter::ImuView v = imu_view(t_ns, gyro, accel, std::nullopt, std::nullopt);
            const exporter::CovarianceMode mode = covariance_mode(covariance);
            s.locked([&](exporter::McapWriter& w) { w.imu(ch, frame_id, v, mode, white_var); });
          },
          nb::arg("ch"), nb::arg("frame_id"), nb::arg("t_ns").noconvert(), nb::arg("gyro").noconvert(),
          nb::arg("accel").noconvert(), nb::arg("covariance") = "model", nb::arg("white_var") = std::array<double, 2>{},
          "An imu channel's C5 samples in stamp order: one sensor_msgs/msg/Imu each (float64 values bit-exact, "
          "orientation (0, 0, 0, 1) with orientation_covariance[0] = -1); covariances as Rosbag1Writer.imu.")
      .def(
          "images",
          [](PyMcapWriter& s, uint16_t ch, const std::string& frame_id, const Column<int64_t>& t_mid,
             const Column<int64_t>& t_fs, const Column<int64_t>& t_fe, const Column<int64_t>& t_es,
             const Column<int64_t>& t_ee, const Pixels& px, int32_t bits, int32_t x0, int32_t y0, int64_t exposure_ns,
             const std::string& frame_map, double scale) {
            const FrameCall c = frame_call(t_mid, t_fs, t_fe, t_es, t_ee, px, bits, x0, y0, exposure_ns, frame_map,
                                           scale);
            s.locked([&](exporter::McapWriter& w) { w.images(ch, frame_id, c.view, c.options); });
          },
          nb::arg("ch"), nb::arg("frame_id"), BALLA_FRAME_ARGS,
          "As Rosbag1Writer.images, in CDR: one sensor_msgs/msg/Image per frame at its midpoint's stamp.")
      .def(
          "close",
          [](PyMcapWriter& s) { return info_dict(s.locked([](exporter::McapWriter& w) { return w.close(); })); },
          "Writes the held packets, the Data End record, the summary and the footer; {path, bytes, sha256, streams: "
          "one per channel {kind events or imu, id the topic, count, t_first, t_last (T2 ns)}}, camera left to the "
          "caller.");
  m.attr("AEDAT4_EARTH_G") = exporter::aedat4::kEarthG;
  nb::class_<PyAedat4Writer>(m, "Aedat4Writer",
                             "One AEDAT 4.0 file of a run's cameras (outputs.md §3.7, §4.7-4.8): per camera an events "
                             "stream (EVTS) and an IMU stream (IMUS, dv's units: g and deg/s), packets of packet_us on "
                             "the T1a us grid, T1b timestamps (us: floor(t_ns / 1000) - origin_us + epoch_ns / 1000), "
                             "the FileDataTable at the end.")
      .def(
          "__init__",
          [](PyAedat4Writer* self, const std::string& path, const std::string& compression, int64_t origin_us,
             int64_t epoch_ns, int threads, int64_t packet_us) {
            new (self) PyAedat4Writer(std::in_place, path, exporter::parse_codec(compression),
                                      exporter::TimeMap{origin_us, epoch_ns}, threads, packet_us);
          },
          nb::arg("path"), nb::arg("compression"), nb::arg("origin_us"), nb::arg("epoch_ns"), nb::arg("threads") = 4,
          nb::arg("packet_us") = exporter::Aedat4Writer::kDefaultPacketUs,
          "compression: none, lz4, lz4-high, zstd or zstd-high (dv-processing's settings); epoch_ns a multiple of "
          "1000; threads (1-256) compress whole packets, the bytes not depending on them; packet_us: the packet "
          "length, 1 to 2^24 (10 ms).")
      .def(
          "add_events_stream",
          [](PyAedat4Writer& s, const std::string& camera, int32_t width, int32_t height) {
            return s.locked([&](exporter::Aedat4Writer& w) { return w.add_events_stream(camera, width, height); });
          },
          nb::arg("camera"), nb::arg("width"), nb::arg("height"),
          "A camera's events stream (width and height <= 32767), source BallaSim_<camera>; returns its stream id. "
          "Every stream is declared before the first item.")
      .def(
          "add_imu_stream",
          [](PyAedat4Writer& s, const std::string& camera) {
            return s.locked([&](exporter::Aedat4Writer& w) { return w.add_imu_stream(camera); });
          },
          nb::arg("camera"), "A camera's IMU stream; returns its stream id.")
      .def(
          "add_frame_stream",
          [](PyAedat4Writer& s, const std::string& camera, int32_t width, int32_t height) {
            return s.locked([&](exporter::Aedat4Writer& w) { return w.add_frame_stream(camera, width, height); });
          },
          nb::arg("camera"), nb::arg("width"), nb::arg("height"),
          "A camera's frames stream (FRME, width and height the full image's, <= 32767); returns its stream id.")
      .def(
          "frames",
          [](PyAedat4Writer& s, int32_t stream, const Column<int64_t>& t_mid, const Column<int64_t>& t_fs,
             const Column<int64_t>& t_fe, const Column<int64_t>& t_es, const Column<int64_t>& t_ee, const Pixels& px,
             int32_t bits, int32_t x0, int32_t y0, int64_t exposure_ns, const std::string& frame_map, double scale) {
            const FrameCall c = frame_call(t_mid, t_fs, t_fe, t_es, t_ee, px, bits, x0, y0, exposure_ns, frame_map,
                                           scale);
            s.locked([&](exporter::Aedat4Writer& w) { w.frames(stream, c.view, c.options); });
          },
          nb::arg("stream"), BALLA_FRAME_ARGS,
          "A frames stream's frames, one FRME packet each (T1b times: timestamp the midpoint, the table entry at the "
          "exposure start, dv-processing's frame time; 8-bit, OPENCV_8U_C1 or OPENCV_8U_C3 in BGR order; gray16 and "
          "float32 frames need frame_map linear8 or srgb8).")
      .def(
          "events",
          [](PyAedat4Writer& s, int32_t stream, const Column<uint16_t>& x, const Column<uint16_t>& y,
             const Column<uint8_t>& p, const Column<int64_t>& t_ns) {
            const exporter::Events e = events_of(x, y, p, t_ns);
            s.locked([&](exporter::Aedat4Writer& w) { w.events(stream, e); });
          },
          nb::arg("stream"), nb::arg("x").noconvert(), nb::arg("y").noconvert(), nb::arg("p").noconvert(),
          nb::arg("t_ns").noconvert(),
          "An events stream's events (as RawWriter.append's), one EVTS packet per packet_us on the T1a grid, held "
          "back until the packet is complete.")
      .def(
          "imu",
          [](PyAedat4Writer& s, int32_t stream, const Column<int64_t>& t_ns, const Rows3<double>& gyro,
             const Rows3<double>& accel) {
            const exporter::ImuView v = imu_view(t_ns, gyro, accel, std::nullopt, std::nullopt);
            s.locked([&](exporter::Aedat4Writer& w) { w.imu(stream, v); });
          },
          nb::arg("stream"), nb::arg("t_ns").noconvert(), nb::arg("gyro").noconvert(), nb::arg("accel").noconvert(),
          "An IMU stream's C5 samples in stamp order: float32 accelerometer = m/s^2 / 9.81007 (g), gyroscope = rad/s "
          "* 180 / pi (deg/s), temperature and magnetometer 0; one IMUS packet per packet_us.")
      .def(
          "close",
          [](PyAedat4Writer& s) { return info_dict(s.locked([](exporter::Aedat4Writer& w) { return w.close(); })); },
          "Writes the held packets, the FileDataTable and the IOHeader's table position; {path, bytes, sha256, "
          "streams: one per stream {camera, kind events, imu or frames, id the stream id, count, t_first, t_last "
          "(T1b us)}}.");

  m.def(
      "frame_map8",
      [](const Pixels& px, int32_t bits, const std::string& frame_map, double scale, bool bgr) {
        exporter::FrameView v;
        set_pixels(v, px);
        v.bits = bits;
        const std::vector<int64_t> zero(v.n, 0);  // times play no part in the map
        v.t_mid_ns = v.t_frame_start_ns = v.t_frame_end_ns = v.t_exp_start_ns = v.t_exp_end_ns = zero.data();
        const exporter::FrameOptions o{exporter::parse_frame_map(frame_map), scale};
        exporter::check(v, o, "frame_map8");
        if (o.map == exporter::FrameMap::kNone && !exporter::is_8bit(v.format)) {
          throw exporter::Refused("frame_map8: gray16 and float32 frames need linear8 or srgb8");
        }
        const size_t ch = static_cast<size_t>(exporter::channels(v.format));
        const size_t h = static_cast<size_t>(v.height), w = static_cast<size_t>(v.width), per = h * w * ch;
        std::vector<uint8_t> out(v.n * per), one;
        for (size_t k = 0; k < v.n; ++k) {
          exporter::frame8(v, k, o, bgr, one);
          std::copy(one.begin(), one.end(), out.begin() + static_cast<std::ptrdiff_t>(k * per));
        }
        return ch == 3 ? bindings::to_numpy(std::move(out), {v.n, h, w, 3})
                       : bindings::to_numpy(std::move(out), {v.n, h, w});
      },
      nb::arg("pixels").noconvert(), nb::arg("bits") = 0, nb::arg("frame_map") = "none", nb::arg("scale") = 1.0,
      nb::arg("bgr") = false,
      "The writers' 8-bit frames (outputs.md §4.8, src/export/frame_map.h) of pixels as the frame writers take them: "
      "(n, H, W[, 3]) uint8 copied (none) or mapped, uint16 (bits) and float32 through linear8 or srgb8 with scale; "
      "rgb in BGR order when bgr (AEDAT4).");

  m.def(
      "write_kalibr",
      [](const std::string& imu_path, const std::string& camchain_path, nb::handle imu, double readout_delay_s,
         double inflate, const std::string& model, const std::vector<double>& dist, double fx, double fy, double cx,
         double cy, int32_t width, int32_t height, const std::string& camera, const std::string& imu_topic,
         const std::string& image_topic, const std::string& version) {
        const imu::ImuParams p = bindings::imu_params_of(imu);
        const exporter::KalibrValues k = exporter::kalibr_values(p, readout_delay_s, inflate);
        const exporter::KalibrCamera c = exporter::kalibr_camera(model, dist, fx, fy, cx, cy, width, height);
        const std::string imu_text = exporter::kalibr_imu_yaml(k, inflate, camera, imu_topic, version);
        const std::string chain_text = exporter::kalibr_camchain_yaml(k, c, camera, image_topic, version);
        exporter::FileInfo a, b;
        {
          nb::gil_scoped_release release;
          a = exporter::write_text_file(imu_path, imu_text);
          b = exporter::write_text_file(camchain_path, chain_text);
        }
        nb::dict v;
        v["update_rate"] = k.update_rate;
        v["gyroscope_noise_density"] = k.gyro_nd;
        v["gyroscope_random_walk"] = k.gyro_rw;
        v["accelerometer_noise_density"] = k.accel_nd;
        v["accelerometer_random_walk"] = k.accel_rw;
        v["timeshift_cam_imu"] = k.timeshift_cam_imu_s;
        v["T_cam_imu"] = std::vector<double>(k.T_cam_imu.begin(), k.T_cam_imu.end());
        v["distortion_model"] = c.distortion_model;
        nb::dict d;
        d["values"] = v;
        d["files"] = nb::make_tuple(info_dict(a), info_dict(b));
        return d;
      },
      nb::arg("imu_path"), nb::arg("camchain_path"), nb::arg("imu"), nb::arg("readout_delay_s"), nb::arg("inflate"),
      nb::arg("model"), nb::arg("dist"), nb::arg("fx"), nb::arg("fy"), nb::arg("cx"), nb::arg("cy"), nb::arg("width"),
      nb::arg("height"), nb::arg("camera"), nb::arg("imu_topic"), nb::arg("image_topic"), nb::arg("version"),
      "Kalibr's imu-<cam>.yaml and camchain-imucam-<cam>.yaml (outputs.md §4.9, src/export/kalibr.h) of a camera: imu "
      "its resolved imu.* values (C5 `params`), readout_delay_s its sensor.readout_delay_s, inflate the densities' "
      "factor, the bundle camera's model (radtan: dist k1, k2, p1, p2, k3, refused when k3 != 0; kb: k1..k4), "
      "intrinsics and resolution, the IMU and image topics and 'BallaSim <version>'. Returns {values, files: (imu, "
      "camchain) {path, bytes, sha256, streams []}}.");
}
