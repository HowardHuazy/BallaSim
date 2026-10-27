// Kalibr's camera-IMU calibration inputs of a camera with an IMU (docs/design/phase3/outputs.md §3.3, §4.9; P3-A A9),
// written from Kalibr's wiki ("yaml formats", "IMU Noise Model"; github.com/ethz-asl/kalibr/wiki): no Kalibr code is
// used.
//   imu-<cam>.yaml              accelerometer_noise_density, accelerometer_random_walk, gyroscope_noise_density,
//                               gyroscope_random_walk (the simulator's own Kalibr densities, imu.<s>.noise_density and
//                               random_walk, times `inflate`, default 1: the wiki's >= 10x is for real MEMS values used
//                               in estimation, offered but never applied), rostopic (the IMU topic), update_rate
//                               (imu.rate_hz, the true-clock rate)
//   camchain-imucam-<cam>.yaml  cam0: T_cam_imu (imu.T_cam_imu as the simulator uses it, its rotation
//                               re-orthonormalized, X_cam = T X_imu), camera_model pinhole, distortion_coeffs and
//                               distortion_model (radtan [k1, k2, p1, p2], refused when k3 != 0, since Kalibr's radtan
//                               has four; equidistant [k1..k4] for kb), intrinsics [fx, fy, cx, cy] and resolution
//                               [W, H] of the bundle camera, rostopic (the image topic), timeshift_cam_imu (s; Kalibr:
//                               t_imu = t_cam + shift) = tau_g + imu.stamp_offset_ns - sensor.readout_delay_s, tau_g
//                               the gyroscope's group delay (imu.gyro.group_delay_s under a first-order filter, else
//                               0), with a comment that it excludes the illumination-dependent photoreceptor latency.
// Numbers are written as YAML 1.1 floats that round-trip (shortest form, always with a '.', so PyYAML, which Kalibr
// reads them with, takes them as floats).
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "export/sink.h"
#include "imu/params.h"

namespace balla::exporter {

struct KalibrValues {
  double update_rate = 0, gyro_nd = 0, gyro_rw = 0, accel_nd = 0, accel_rw = 0, timeshift_cam_imu_s = 0;
  std::array<double, 16> T_cam_imu{};  // row-major, last row (0, 0, 0, 1)
};

// The IMU's Kalibr values; Refused when validate() refuses the parameters, readout_delay_s is not finite and >= 0, or
// inflate is not finite and > 0.
KalibrValues kalibr_values(const imu::ImuParams&, double readout_delay_s, double inflate);

struct KalibrCamera {
  std::string distortion_model;          // radtan or equidistant
  std::vector<double> distortion_coeffs;  // 4 each
  std::array<double, 4> intrinsics{};     // fx, fy, cx, cy
  std::array<int32_t, 2> resolution{};    // width, height
};

// A bundle camera (model radtan: dist k1, k2, p1, p2, k3; kb: k1..k4) as Kalibr's pinhole camera; Refused for radtan
// with k3 != 0, an unknown model or a wrong coefficient count.
KalibrCamera kalibr_camera(std::string_view model, std::span<const double> dist, double fx, double fy, double cx,
                           double cy, int32_t width, int32_t height);

// The shortest round-trip form of a finite double with a '.' before any exponent ("1.0e-05", "1000.0").
std::string yaml_float(double);

// The two files' texts; camera names the BallaSim camera in a comment, version is "BallaSim <version>".
std::string kalibr_imu_yaml(const KalibrValues&, double inflate, std::string_view camera, std::string_view rostopic,
                            std::string_view version);
std::string kalibr_camchain_yaml(const KalibrValues&, const KalibrCamera&, std::string_view camera,
                                 std::string_view rostopic, std::string_view version);

// A text file through the exporters' sink: {path, bytes, sha256, no streams}.
FileInfo write_text_file(const std::string& path, std::string_view text);

}  // namespace balla::exporter
