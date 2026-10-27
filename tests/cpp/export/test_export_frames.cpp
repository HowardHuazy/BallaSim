// Frame streams and Kalibr (docs/design/phase3/outputs.md §4.8 "Frames", §4.9, §8.2 F10; P3-A A9): the 8-bit maps
// (gray8 and rgb8 copied, rgb8 as BGR for AEDAT4, f32 through the sRGB OETF, gray16 by its significant bits);
// dv-processing 2.0.4's own FRME packets byte for byte; the AEDAT4 frames stream (infoNode, one packet per frame with
// the T1b times, the BGR pixels, the position and exposure; the table entry) and its refusals; the ROS1 and CDR
// sensor_msgs/Image layouts and the ROS writers' image streams; the Kalibr values, YAML numbers and texts. The
// driver, the order-sensitive oracles and the validators are tests/python/test_export_frames.py's.
#include <doctest/doctest.h>

#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

#include "export/aedat4.h"
#include "export/cdr.h"
#include "export/errors.h"
#include "export/frame_map.h"
#include "export/kalibr.h"
#include "export/mcap.h"
#include "export/ros1.h"
#include "imu/params.h"

using namespace balla;
using exporter::FrameMap;
using exporter::FrameOptions;
using exporter::FrameView;
using exporter::PixelFormat;
using exporter::Refused;

namespace {

struct TempDir {
  std::filesystem::path dir =
      std::filesystem::temp_directory_path() / ("balla_export_frames_tests_" + std::to_string(::getpid()));
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};
const TempDir kTemp;

std::string temp_path(const std::string& name) {
  std::filesystem::create_directories(kTemp.dir);
  return (kTemp.dir / name).string();
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

template <class V>
std::string hex(const V& v) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (const auto c : v) {
    s.push_back(d[(static_cast<unsigned char>(c) >> 4) & 15]);
    s.push_back(d[static_cast<unsigned char>(c) & 15]);
  }
  return s;
}

// n frames of one format and size with their five times (sensor clock, ns).
struct Frames {
  std::vector<int64_t> mid, fs, fe, es, ee;
  std::vector<uint8_t> px;
  PixelFormat format = PixelFormat::kGray8;
  int32_t w = 0, h = 0;
  int32_t bits = 0, x0 = 0, y0 = 0;
  int64_t exposure_ns = -1;

  // A frame exposed over [start, start + exposure] with one row time `row` after its end.
  void add(int64_t start, int64_t exposure, int64_t row, const std::vector<uint8_t>& pixels) {
    es.push_back(start);
    fs.push_back(start);
    ee.push_back(start + exposure);
    fe.push_back(start + exposure + row);
    mid.push_back(start + exposure / 2);
    px.insert(px.end(), pixels.begin(), pixels.end());
  }
  FrameView view() const {
    FrameView v;
    v.t_mid_ns = mid.data();
    v.t_frame_start_ns = fs.data();
    v.t_frame_end_ns = fe.data();
    v.t_exp_start_ns = es.data();
    v.t_exp_end_ns = ee.data();
    v.format = format;
    v.pixels = px.data();
    v.width = w;
    v.height = h;
    v.row_stride_bytes = static_cast<int64_t>(exporter::row_bytes(v));
    v.frame_stride_bytes = v.row_stride_bytes * h;
    v.n = mid.size();
    v.bits = bits;
    v.x0 = x0;
    v.y0 = y0;
    v.exposure_ns = exposure_ns;
    return v;
  }
};

template <class T>
std::vector<uint8_t> bytes_of(const std::vector<T>& v) {
  std::vector<uint8_t> b(v.size() * sizeof(T));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

std::vector<uint8_t> map_one(PixelFormat f, const std::vector<uint8_t>& px, int32_t w, int32_t h, FrameOptions o,
                             bool bgr = false, int32_t bits = 0) {
  Frames fr;
  fr.format = f;
  fr.w = w;
  fr.h = h;
  fr.bits = bits;
  fr.add(0, 0, 0, px);
  std::vector<uint8_t> out;
  exporter::frame8(fr.view(), 0, o, bgr, out);
  return out;
}

// The sRGB OETF of IEC 61966-2-1 and round half to even, in long double: the §4.8 formula, independently written.
uint8_t srgb8_reference(long double v) {
  v = v < 0 ? 0 : v > 1 ? 1 : v;
  const long double e = v <= 0.0031308L ? 12.92L * v : 1.055L * std::pow(v, 1.0L / 2.4L) - 0.055L;
  const long double x = 255.0L * e, f = std::floor(x);
  const long double r = x - f > 0.5L ? f + 1 : x - f < 0.5L ? f : (std::fmod(f, 2.0L) == 0 ? f : f + 1);
  return static_cast<uint8_t>(r);
}

}  // namespace

TEST_CASE("F10 frame maps: 8-bit copied (rgb8 as BGR), f32 through linear8 and srgb8, gray16 by its bits") {
  const std::vector<uint8_t> gray = {0, 1, 127, 128, 254, 255};
  CHECK(map_one(PixelFormat::kGray8, gray, 3, 2, {}) == gray);
  CHECK(map_one(PixelFormat::kGray8, gray, 3, 2, {FrameMap::kLinear8, 1.0}) == gray);
  const std::vector<uint8_t> rgb = {10, 20, 30, 40, 50, 60};
  CHECK(map_one(PixelFormat::kRgb8, rgb, 2, 1, {}) == rgb);
  CHECK(map_one(PixelFormat::kRgb8, rgb, 2, 1, {}, /*bgr=*/true) == std::vector<uint8_t>{30, 20, 10, 60, 50, 40});

  // f32 with srgb8 on 256 ramp values, and the formula's corners: NaN, negatives and values above 1.
  std::vector<float> ramp;
  for (int k = 0; k < 256; ++k) ramp.push_back(static_cast<float>(k) / 255.0f);
  const std::vector<uint8_t> s8 = map_one(PixelFormat::kGrayF32, bytes_of(ramp), 256, 1, {FrameMap::kSrgb8, 1.0});
  int diffs = 0;
  for (int k = 0; k < 256; ++k) diffs += s8[static_cast<size_t>(k)] != srgb8_reference(ramp[static_cast<size_t>(k)]);
  CHECK(diffs == 0);
  CHECK(s8.front() == 0);
  CHECK(s8.back() == 255);
  const std::vector<float> corners = {std::numeric_limits<float>::quiet_NaN(), -1.0f, 2.0f, 0.5f, 0.25f};
  CHECK(map_one(PixelFormat::kGrayF32, bytes_of(corners), 5, 1, {FrameMap::kLinear8, 1.0}) ==
        std::vector<uint8_t>{0, 0, 255, 128, 64});  // 127.5 rounds to the even 128, 63.75 to 64
  CHECK(map_one(PixelFormat::kGrayF32, bytes_of(corners), 5, 1, {FrameMap::kLinear8, 2.0}) ==
        std::vector<uint8_t>{0, 0, 255, 255, 128});
  CHECK(map_one(PixelFormat::kGrayF32, bytes_of(corners), 5, 1, {FrameMap::kSrgb8, 1.0})[3] == 188);  // 187.52
  // rgb f32 to BGR.
  const std::vector<float> rgbf = {1.0f, 0.0f, 0.5f};
  CHECK(map_one(PixelFormat::kRgbF32, bytes_of(rgbf), 1, 1, {FrameMap::kLinear8, 1.0}, true) ==
        std::vector<uint8_t>{128, 0, 255});

  // gray16 with linear8: round_half_even(255 v / 65535) = v / 257 exactly at multiples of 257.
  const std::vector<uint16_t> g16 = {0, 128, 129, 32896, 65535, 257 * 77};
  CHECK(map_one(PixelFormat::kGray16, bytes_of(g16), 6, 1, {FrameMap::kLinear8, 1.0}) ==
        std::vector<uint8_t>{0, 0, 1, 128, 255, 77});
  // An APS's 10-bit DN fills the range: 1023 -> 255, 512 -> 255 * 512 / 1023 = 127.6 -> 128.
  const std::vector<uint16_t> dn = {0, 512, 1023, 4000};
  CHECK(map_one(PixelFormat::kGray16, bytes_of(dn), 4, 1, {FrameMap::kLinear8, 1.0}, false, 10) ==
        std::vector<uint8_t>{0, 128, 255, 255});
}

TEST_CASE("F1 AEDAT4: dv-processing 2.0.4's own FRME packets byte for byte") {
  // tests/data/export/t3_vectors.json aedat4_dv_frme_hex, from dv-processing 2.0.4's MonoCameraWriter (compression
  // NONE): a 5 x 4 gray frame at 1234 us, exposure 500 us, position (16, 120), SYNTHETIC (dv sets the start and end
  // of exposure from the timestamp and exposure, leaves the frame's start and end 0); a 3 x 2 BGR frame at 2000 us,
  // exposure 0, UNDEFINED; a 3 x 2 BGR frame at 5000 us, exposure 777 us, position (7, 9), SYNTHETIC.
  std::vector<uint8_t> gray;
  for (int i = 0; i < 20; ++i) gray.push_back(static_cast<uint8_t>(3 * i));
  std::vector<uint8_t> bgr(18);
  for (size_t i = 0; i < bgr.size(); ++i) bgr[i] = static_cast<uint8_t>(10 * (i % 3 + 1));
  exporter::aedat4::FrameRecord a{1234, 0, 0, 1234, 1734, 500, 0, 4, 5, 4, 16, 120, gray.data(), gray.size()};
  CHECK(hex(exporter::aedat4::frame_packet(a)) ==
        "740000002800000046524d4500001e0034001400000000001c002400000008000a000c000e0010002c0007001e0000000000000405"
        "0004001000780024000000d204000000000000d204000000000000c606000000000000f4010000000000001400000000030609"
        "0c0f1215181b1e2124272a2d30333639");
  exporter::aedat4::FrameRecord b{2000, 0, 0, 2000, 2000, 0, 16, 0, 3, 2, 0, 0, bgr.data(), bgr.size()};
  CHECK(hex(exporter::aedat4::frame_packet(b)) ==
        "640000002400000046524d4500001a00280010000000000018002000070008000a00000000000c001a00000000000010030002001c"
        "000000d007000000000000d007000000000000d007000000000000120000000a141e0a141e0a141e0a141e0a141e0a141e0000");
  exporter::aedat4::FrameRecord c{5000, 0, 0, 5000, 5777, 777, 16, 4, 3, 2, 7, 9, bgr.data(), bgr.size()};
  CHECK(hex(exporter::aedat4::frame_packet(c)) ==
        "740000002800000046524d4500001e0034001400000000001c002400060008000a000c000e0010002c0007001e0000000000100403"
        "00020007000900240000008813000000000000881300000000000091160000000000000903000000000000120000000a141e0a141e"
        "0a141e0a141e0a141e0a141e0000");
}

TEST_CASE("AEDAT4 frames stream: infoNode, one FRME packet per frame (T1b times, BGR, position, exposure), table") {
  const std::string path = temp_path("frames.aedat4");
  Frames fr;
  fr.format = PixelFormat::kRgb8;
  fr.w = 2;
  fr.h = 1;
  fr.x0 = 3;
  fr.y0 = 4;
  fr.exposure_ns = 4'000'999;  // the per-row exposure, floored to 4000 us
  fr.add(3'000'000'417, 5'000'000, 31'000, {1, 2, 3, 4, 5, 6});
  fr.add(3'033'000'417, 5'000'000, 31'000, {7, 8, 9, 10, 11, 12});
  exporter::FileInfo info;
  {
    exporter::Aedat4Writer w(path, exporter::Codec::kNone, {3'000'000, 0}, 2);
    CHECK(w.add_events_stream("cam", 64, 48) == 0);
    CHECK(w.add_frame_stream("cam", 64, 48) == 1);
    CHECK_THROWS_AS(w.add_frame_stream("cam", 64, 48), Refused);  // once per camera
    w.frames(1, fr.view(), {});
    info = w.close();
  }
  const std::string file = read_file(path);
  CHECK(file.find("<attr key=\"typeIdentifier\" type=\"string\">FRME</attr>") != std::string::npos);
  CHECK(file.find("<attr key=\"originalOutputName\" type=\"string\">frames</attr>") != std::string::npos);
  const std::vector<uint8_t> bgr0 = {3, 2, 1, 6, 5, 4}, bgr1 = {9, 8, 7, 12, 11, 10};
  // T1b at origin 3 s: start 417 ns -> 0 us; end of exposure 5000 us; end of frame 5031; midpoint 2500.
  const exporter::aedat4::FrameRecord r0{2500, 0, 5031, 0, 5000, 4000, 16, 4, 2, 1, 3, 4, bgr0.data(), 6};
  const exporter::aedat4::FrameRecord r1{35500, 33000, 38031, 33000, 38000, 4000, 16, 4, 2, 1, 3, 4, bgr1.data(), 6};
  std::vector<exporter::aedat4::TableEntry> table;
  for (const auto& r : {r0, r1}) {
    const std::vector<char> packet = exporter::aedat4::frame_packet(r);
    const std::string want(packet.begin(), packet.end());
    const size_t at = file.find(want);
    REQUIRE(at != std::string::npos);
    int32_t header[2];
    std::memcpy(header, file.data() + at - 8, 8);
    CHECK(header[0] == 1);
    CHECK(header[1] == static_cast<int32_t>(packet.size()));
    // Its table entry: one element at the exposure start (dv-processing 2.0.4's frame time), not the midpoint.
    table.push_back({static_cast<int64_t>(at), 1, header[1], 1, r.start_of_exposure, r.start_of_exposure});
  }
  const std::vector<char> ftab = exporter::aedat4::file_data_table(table);  // codec none: stored as built
  REQUIRE(file.size() > ftab.size());
  CHECK(file.compare(file.size() - ftab.size(), ftab.size(), std::string(ftab.begin(), ftab.end())) == 0);
  REQUIRE(info.streams.size() == 2);
  CHECK(info.streams[1].kind == "frames");
  CHECK(info.streams[1].count == 2);
  CHECK(info.streams[1].t_first == 2500);
  CHECK(info.streams[1].t_last == 35500);
  CHECK(info.streams[0].count == 0);
}

TEST_CASE("AEDAT4 frames refusals: 8-bit only, the ROI in the image, times in order, nothing before the origin") {
  const std::string path = temp_path("refused.aedat4");
  exporter::Aedat4Writer w(path, exporter::Codec::kNone, {1'000, 0}, 1);
  const int32_t s = w.add_frame_stream("cam", 8, 8);
  Frames f32;
  f32.format = PixelFormat::kGrayF32;
  f32.w = f32.h = 1;
  const std::vector<float> one = {0.5f};
  f32.add(2'000'000, 1000, 0, bytes_of(one));
  CHECK_THROWS_WITH_AS(w.frames(s, f32.view(), {}), doctest::Contains("AEDAT4 frames are 8-bit"), Refused);
  w.frames(s, f32.view(), {FrameMap::kSrgb8, 1.0});  // mapped: accepted

  Frames g;
  g.w = g.h = 2;
  g.x0 = 7;  // 7 + 2 > 8
  g.add(3'000'000, 1000, 0, {1, 2, 3, 4});
  CHECK_THROWS_WITH_AS(w.frames(s, g.view(), {}), doctest::Contains("outside the 8 x 8 image"), Refused);
  g.x0 = 6;
  Frames early = g;
  early.es[0] = early.fs[0] = 999'999;  // the frame starts before the origin (1000 us)
  CHECK_THROWS_WITH_AS(w.frames(s, early.view(), {}), doctest::Contains("before the export origin"), Refused);
  Frames back = g;
  back.mid[0] = back.es[0] = back.fs[0] = 1'900'000;  // before the previous call's midpoint
  CHECK_THROWS_WITH_AS(w.frames(s, back.view(), {}), doctest::Contains("time order"), Refused);
  Frames long_exp = g;  // a later midpoint, but an exposure start before the previous call's (the table key)
  long_exp.es[0] = long_exp.fs[0] = 1'999'000;
  CHECK_THROWS_WITH_AS(w.frames(s, long_exp.view(), {}), doctest::Contains("exposure start 1999000 follows"),
                       Refused);
  Frames bad = g;
  bad.mid[0] = bad.ee[0] + 1;  // the midpoint after the exposure's end
  CHECK_THROWS_WITH_AS(w.frames(s, bad.view(), {}), doctest::Contains("out of order"), Refused);
  CHECK_THROWS_AS(w.frames(s, g.view(), {FrameMap::kLinear8, 0.0}), Refused);  // scale > 0
  w.frames(s, g.view(), {});
  CHECK_THROWS_AS(w.events(s, {}), Refused);  // a frames stream takes no events
  const exporter::FileInfo info = w.close();
  CHECK(info.streams[0].count == 2);

  exporter::Aedat4Writer neg(temp_path("neg.aedat4"), exporter::Codec::kNone, {1'000, -5'000'000}, 1);
  const int32_t sn = neg.add_frame_stream("cam", 8, 8);
  CHECK_THROWS_WITH_AS(neg.frames(sn, g.view(), {}), doctest::Contains("timestamp < 0"), Refused);
}

TEST_CASE("ROS1 and CDR sensor_msgs/Image: layouts; the ROS writers' image streams at the midpoint's T2 stamp") {
  const uint8_t data[6] = {1, 2, 3, 4, 5, 6};
  const exporter::ros::Image im{1, 2, "rgb8", 6, data, 6};
  std::vector<char> m;
  exporter::ros1::put_image(m, 7, 1'000'000'123, "cam", im);
  std::vector<char> want;
  auto u32 = [&](uint32_t v) {
    const char* p = reinterpret_cast<const char*>(&v);
    want.insert(want.end(), p, p + 4);
  };
  u32(7), u32(1), u32(123), u32(3);
  want.insert(want.end(), {'c', 'a', 'm'});
  u32(1), u32(2), u32(4);
  want.insert(want.end(), {'r', 'g', 'b', '8', 0});
  u32(6), u32(6);
  want.insert(want.end(), data, data + 6);
  CHECK(m == want);

  std::vector<char> c;
  exporter::cdr::put_image(c, 1'000'000'123, "cam", im);
  // 00 01 00 00 | sec 1, nanosec 123, "cam\0" (4 + 4) | height, width | "rgb8\0" (4 + 5) | is_bigendian, 2 padding
  // bytes so step is 4-aligned | step | data count, data.
  std::vector<char> w2 = {0, 1, 0, 0};
  auto c32 = [&](uint32_t v) {
    const char* p = reinterpret_cast<const char*>(&v);
    w2.insert(w2.end(), p, p + 4);
  };
  c32(1), c32(123), c32(4);
  w2.insert(w2.end(), {'c', 'a', 'm', 0});
  c32(1), c32(2), c32(5);
  w2.insert(w2.end(), {'r', 'g', 'b', '8', 0, 0, 0, 0});
  c32(6), c32(6);
  w2.insert(w2.end(), data, data + 6);
  CHECK(c == w2);

  Frames fr;
  fr.format = PixelFormat::kGray16;
  fr.w = 2;
  fr.h = 1;
  fr.bits = 10;
  const std::vector<uint16_t> px = {1023, 512};
  fr.add(2'000'000'417, 1'000'000, 0, bytes_of(px));
  {
    exporter::Rosbag1Writer w(temp_path("img.bag"), exporter::Codec::kNone, exporter::ros1::kChunkThreshold,
                              {2'000'000, 1'000'000'000}, 1);
    const uint32_t k = w.connection("/cam/image_raw", exporter::MsgType::kImage);
    w.images(k, "cam", fr.view(), {});
    const exporter::FileInfo info = w.close();
    REQUIRE(info.streams.size() == 1);
    CHECK(info.streams[0].kind == "frames");
    CHECK(info.streams[0].t_first == 1'000'000'000 + 500'417);  // the midpoint's T2 stamp
  }
  {
    exporter::McapWriter w(temp_path("img.mcap"), exporter::Codec::kNone, exporter::mcap::kChunkBytes,
                           {2'000'000, 1'000'000'000}, 1, "BallaSim test");
    const uint16_t k = w.channel("/cam/image_raw", exporter::MsgType::kImage);
    w.images(k, "cam", fr.view(), {FrameMap::kLinear8, 1.0});
    Frames early = fr;
    early.fs[0] = early.es[0] = 1'999'999'999;  // before the origin
    CHECK_THROWS_WITH_AS(w.images(k, "cam", early.view(), {}), doctest::Contains("before the export origin"),
                         Refused);
    const exporter::FileInfo info = w.close();
    CHECK(info.streams[0].count == 1);
  }
  CHECK(exporter::ros1_msgdef(exporter::MsgType::kImage).md5 == "060021388200f6f0f447d0fcd9c64743");
  CHECK(exporter::ros2_msgdef(exporter::MsgType::kImage).type == "sensor_msgs/msg/Image");
}

TEST_CASE("Kalibr: YAML floats, the values (densities, rate, timeshift, T_cam_imu), the camera and the texts") {
  using exporter::yaml_float;
  CHECK(yaml_float(1e-5) == "1.0e-05");
  CHECK(yaml_float(1000) == "1000.0");
  CHECK(yaml_float(0.0039) == "0.0039");
  CHECK(yaml_float(-0.0) == "-0.0");
  CHECK(yaml_float(1.5e20) == "1.5e+20");
  CHECK(yaml_float(0.1 + 0.2) == "0.30000000000000004");  // round-trips
  CHECK_THROWS_AS(yaml_float(std::nan("")), Refused);

  imu::ImuParams p;
  p.rate_hz = 200;
  p.gyro.noise_density = 1e-4;
  p.gyro.random_walk = 2.4e-5;
  p.accel.noise_density = 1.2e-3;
  p.accel.random_walk = 5e-3;
  p.gyro.filter = imu::Filter::kFirstOrder;  // tau_g = 3.9 ms
  p.stamp_offset_ns = 250'000;
  p.T_cam_imu = {-1, 0, 0, 0.001, 0, -1, 0, 0.002, 0, 0, 1, -0.04, 0, 0, 0, 1};
  const exporter::KalibrValues k = exporter::kalibr_values(p, 3e-4, 10);
  CHECK(k.update_rate == 200);
  CHECK(k.gyro_nd == 1e-4 * 10);
  CHECK(k.accel_rw == 5e-3 * 10);
  CHECK(k.timeshift_cam_imu_s == (3.9e-3 + 250'000 * 1e-9) - 3e-4);
  CHECK(k.T_cam_imu[0] == -1);
  CHECK(k.T_cam_imu[3] == 0.001);
  CHECK(k.T_cam_imu[11] == -0.04);
  CHECK(k.T_cam_imu[15] == 1);
  p.gyro.filter = imu::Filter::kNone;  // tau_g 0
  CHECK(exporter::kalibr_values(p, 3e-4, 1).timeshift_cam_imu_s == 250'000 * 1e-9 - 3e-4);
  CHECK_THROWS_AS(exporter::kalibr_values(p, 3e-4, 0), Refused);
  imu::ImuParams bad = p;
  bad.noise_mode = imu::NoiseMode::kShaped;  // shaped needs both filters
  CHECK_THROWS_WITH_AS(exporter::kalibr_values(bad, 3e-4, 1), doctest::Contains("shaped"), Refused);

  const double radtan[5] = {0.1, -0.2, 0.001, 0.002, 0.0};
  const exporter::KalibrCamera cam = exporter::kalibr_camera("radtan", radtan, 500, 501, 320, 240, 640, 480);
  CHECK(cam.distortion_model == "radtan");
  CHECK(cam.distortion_coeffs == std::vector<double>{0.1, -0.2, 0.001, 0.002});
  const double k3[5] = {0.1, -0.2, 0.001, 0.002, 0.01};
  CHECK_THROWS_WITH_AS(exporter::kalibr_camera("radtan", k3, 500, 501, 320, 240, 640, 480),
                       doctest::Contains("k3"), Refused);
  const double kb[4] = {0.01, 0.02, 0.03, 0.04};
  CHECK(exporter::kalibr_camera("kb", kb, 500, 501, 320, 240, 640, 480).distortion_model == "equidistant");
  CHECK_THROWS_AS(exporter::kalibr_camera("ds", kb, 500, 501, 320, 240, 640, 480), Refused);

  const exporter::KalibrValues k1 = exporter::kalibr_values(p, 3e-4, 1);
  CHECK(exporter::kalibr_imu_yaml(k1, 1, "left", "/left/imu", "BallaSim 0.1.0") ==
        "# BallaSim 0.1.0: Kalibr IMU parameters of camera left (docs/dev/EXPORT.md):\n"
        "# the simulator's own noise densities (Kalibr's continuous-time convention) times 1.0.\n"
        "accelerometer_noise_density: 0.0012\n"
        "accelerometer_random_walk: 0.005\n"
        "gyroscope_noise_density: 1.0e-04\n"
        "gyroscope_random_walk: 2.4e-05\n"
        "rostopic: /left/imu\n"
        "update_rate: 200.0\n");
  const std::string chain = exporter::kalibr_camchain_yaml(k1, cam, "left", "/left/image_raw", "BallaSim 0.1.0");
  CHECK(chain.find("cam0:\n  T_cam_imu:\n  - [-1.0, 0.0, 0.0, 0.001]\n  - [0.0, -1.0, 0.0, 0.002]\n"
                   "  - [0.0, 0.0, 1.0, -0.04]\n  - [0.0, 0.0, 0.0, 1.0]\n  camera_model: pinhole\n"
                   "  distortion_coeffs: [0.1, -0.2, 0.001, 0.002]\n  distortion_model: radtan\n"
                   "  intrinsics: [500.0, 501.0, 320.0, 240.0]\n  resolution: [640, 480]\n"
                   "  rostopic: /left/image_raw\n  timeshift_cam_imu: " +
                   yaml_float(250'000 * 1e-9 - 3e-4) + "\n") != std::string::npos);
  CHECK_THROWS_AS(exporter::kalibr_imu_yaml(k1, 1, "left", "left imu", "BallaSim"), Refused);  // a topic with a space
  CHECK_THROWS_AS(exporter::kalibr_imu_yaml(k1, 1, "left\nx: 1", "/left/imu", "BallaSim"), Refused);
  const exporter::FileInfo f = exporter::write_text_file(temp_path("imu.yaml"), "a: 1.0\n");
  CHECK(f.bytes == 7);
  CHECK(read_file(f.path) == "a: 1.0\n");
}
