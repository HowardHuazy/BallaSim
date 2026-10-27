// Kalibr's camera-IMU calibration inputs (kalibr.h; docs/design/phase3/outputs.md §4.9).
#include "export/kalibr.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <string>

#include "export/errors.h"

namespace balla::exporter {

namespace {

// ROS's graph resource name rule (wiki.ros.org/Names), as the rosbag1 writer checks topics: then the name is a plain
// YAML scalar.
bool ros_name(std::string_view s) {
  if (s.empty()) return false;
  auto alpha = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
  if (!alpha(s[0]) && s[0] != '/' && s[0] != '~') return false;
  return std::all_of(s.begin() + 1, s.end(),
                     [&](char c) { return alpha(c) || (c >= '0' && c <= '9') || c == '_' || c == '/'; });
}

void check_comment(std::string_view s, const char* what) {
  if (std::any_of(s.begin(), s.end(), [](char c) { return static_cast<unsigned char>(c) < 0x20 || c == 0x7F; })) {
    throw Refused(std::string("kalibr: the ") + what + " must be free of control characters");
  }
}

void check_topic(std::string_view topic) {
  if (!ros_name(topic)) {
    throw Refused("kalibr: rostopic '" + std::string(topic) + "' breaks ROS's name rule (a letter, '/' or '~' "
                  "first, then letters, digits, '_' or '/')");
  }
}

std::string list(const double* v, size_t n) {
  std::string s = "[";
  for (size_t i = 0; i < n; ++i) s += (i ? ", " : "") + yaml_float(v[i]);
  return s + "]";
}

}  // namespace

std::string yaml_float(double v) {
  if (!std::isfinite(v)) throw Refused("kalibr: a value is not finite");
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof buf, v);
  const std::string s(buf, r.ptr);
  const size_t e = s.find('e');
  std::string mantissa = s.substr(0, e);
  if (mantissa.find('.') == std::string::npos) mantissa += ".0";
  return e == std::string::npos ? mantissa : mantissa + s.substr(e);
}

KalibrValues kalibr_values(const imu::ImuParams& p, double readout_delay_s, double inflate) {
  const imu::Check c = imu::validate(p);
  if (!c.errors.empty()) throw Refused("kalibr: " + c.errors.front());
  if (!(std::isfinite(readout_delay_s) && readout_delay_s >= 0)) {
    throw Refused("kalibr: sensor.readout_delay_s must be finite and >= 0");
  }
  if (!(std::isfinite(inflate) && inflate > 0)) throw Refused("kalibr: inflate must be finite and > 0");
  KalibrValues k;
  k.update_rate = p.rate_hz;
  k.gyro_nd = p.gyro.noise_density * inflate;
  k.gyro_rw = p.gyro.random_walk * inflate;
  k.accel_nd = p.accel.noise_density * inflate;
  k.accel_rw = p.accel.random_walk * inflate;
  const double tau_g = p.gyro.filter == imu::Filter::kFirstOrder ? p.gyro.group_delay_s : 0.0;
  k.timeshift_cam_imu_s = (tau_g + static_cast<double>(p.stamp_offset_ns) * 1e-9) - readout_delay_s;
  const imu::Mat3d r = imu::rotation_of(p);
  const imu::Vec3d t = imu::translation_of(p);
  const double tt[3] = {t.x, t.y, t.z};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) k.T_cam_imu[static_cast<size_t>(4 * i + j)] = r.m[i][j];
    k.T_cam_imu[static_cast<size_t>(4 * i + 3)] = tt[i];
  }
  k.T_cam_imu[15] = 1.0;
  return k;
}

KalibrCamera kalibr_camera(std::string_view model, std::span<const double> dist, double fx, double fy, double cx,
                           double cy, int32_t width, int32_t height) {
  KalibrCamera c;
  if (model == "radtan") {
    if (dist.size() != 5) throw Refused("kalibr: a radtan camera has 5 coefficients (k1, k2, p1, p2, k3)");
    if (dist[4] != 0.0) {
      throw Refused("kalibr: the camera's radtan k3 = " + yaml_float(dist[4]) +
                    " is not 0, and Kalibr's radtan has four coefficients (k1, k2, p1, p2)");
    }
    c.distortion_model = "radtan";
    c.distortion_coeffs.assign(dist.begin(), dist.begin() + 4);
  } else if (model == "kb") {
    if (dist.size() != 4) throw Refused("kalibr: a kb camera has 4 coefficients (k1..k4)");
    c.distortion_model = "equidistant";
    c.distortion_coeffs.assign(dist.begin(), dist.end());
  } else {
    throw Refused("kalibr: camera model '" + std::string(model) + "': radtan or kb");
  }
  for (const double v : {fx, fy, cx, cy}) (void)yaml_float(v);  // finite
  for (const double v : c.distortion_coeffs) (void)yaml_float(v);
  if (width < 1 || height < 1) throw Refused("kalibr: the camera's resolution must be positive");
  c.intrinsics = {fx, fy, cx, cy};
  c.resolution = {width, height};
  return c;
}

std::string kalibr_imu_yaml(const KalibrValues& k, double inflate, std::string_view camera, std::string_view rostopic,
                            std::string_view version) {
  check_comment(camera, "camera name");
  check_comment(version, "version");
  check_topic(rostopic);
  std::string s;
  s += "# " + std::string(version) + ": Kalibr IMU parameters of camera " + std::string(camera) +
       " (docs/dev/EXPORT.md):\n";
  s += "# the simulator's own noise densities (Kalibr's continuous-time convention) times " + yaml_float(inflate) +
       ".\n";
  s += "accelerometer_noise_density: " + yaml_float(k.accel_nd) + "\n";
  s += "accelerometer_random_walk: " + yaml_float(k.accel_rw) + "\n";
  s += "gyroscope_noise_density: " + yaml_float(k.gyro_nd) + "\n";
  s += "gyroscope_random_walk: " + yaml_float(k.gyro_rw) + "\n";
  s += "rostopic: " + std::string(rostopic) + "\n";
  s += "update_rate: " + yaml_float(k.update_rate) + "\n";
  return s;
}

std::string kalibr_camchain_yaml(const KalibrValues& k, const KalibrCamera& c, std::string_view camera,
                                 std::string_view rostopic, std::string_view version) {
  check_comment(camera, "camera name");
  check_comment(version, "version");
  check_topic(rostopic);
  std::string s;
  s += "# " + std::string(version) + ": Kalibr camera-IMU chain of camera " + std::string(camera) +
       " (docs/dev/EXPORT.md).\n";
  s += "# timeshift_cam_imu (t_imu = t_cam + shift) = tau_g + imu.stamp_offset_ns - sensor.readout_delay_s; it\n";
  s += "# excludes the photoreceptor latency, which depends on the illumination (EVIMO2: kappa = -1.0 ms in normal\n";
  s += "# light, +6 to +11 ms in low light).\n";
  s += "cam0:\n";
  s += "  T_cam_imu:\n";
  for (int i = 0; i < 4; ++i) s += "  - " + list(k.T_cam_imu.data() + 4 * i, 4) + "\n";
  s += "  camera_model: pinhole\n";
  s += "  distortion_coeffs: " + list(c.distortion_coeffs.data(), c.distortion_coeffs.size()) + "\n";
  s += "  distortion_model: " + c.distortion_model + "\n";
  s += "  intrinsics: " + list(c.intrinsics.data(), 4) + "\n";
  s += "  resolution: [" + std::to_string(c.resolution[0]) + ", " + std::to_string(c.resolution[1]) + "]\n";
  s += "  rostopic: " + std::string(rostopic) + "\n";
  s += "  timeshift_cam_imu: " + yaml_float(k.timeshift_cam_imu_s) + "\n";
  return s;
}

FileInfo write_text_file(const std::string& path, std::string_view text) {
  ByteSink sink(path);
  sink.write(text.data(), text.size());
  return sink.close();
}

}  // namespace balla::exporter
