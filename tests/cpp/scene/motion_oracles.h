// Synthetic oracles for the motion session's tests (blender.md §8.1 T6, T7; a port of the research prototype's
// t_synthetic.py with Blender's key plateau): analytic motions answered the way Blender answers frame_set(F, s), as
// the headless oracles measured it (blender.md §2.3): evaluation at the realized float32 frame f32(f32(F) + s) times
// framelen, a key's value within 1e-4 frames of the key (MF E8), and matrix_world rounded to float32.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "core/math/quat.h"
#include "core/math/so3.h"
#include "scene/motion/pipeline.h"

namespace balla::scene::motion::testing {

struct Pose {
  rot::Q q{1, 0, 0, 0};
  rot::V p{0, 0, 0};
};

struct SyntheticBody {
  std::function<Pose(double frame)> motion;  // the pose at an evaluated (scene-time) frame
  std::vector<double> keys;                  // frames whose value Blender returns within `plateau` frames of them
  double scale = 1.0;                        // a uniform object scale (S = scale I)
  double plateau = 1e-4;                     // 1e-4 N_s: a key under an NLA strip of scale N_s holds N_s times longer
};

// The 3x4 float32 matrix_world of a pose (with a uniform scale).
inline void to_matrix34(const Pose& pose, double scale, float* m) {
  const Mat3<double> r = to_matrix(pose.q);
  const double t[3] = {pose.p.x, pose.p.y, pose.p.z};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) m[4 * i + j] = static_cast<float>(r.m[i][j] * scale);
    m[4 * i + 3] = static_cast<float>(t[i]);
  }
}

// Every body present, no unknown instances; *calls counts the samples answered.
inline Oracle synthetic_oracle(const TimeMap& time, std::vector<SyntheticBody> bodies,
                               std::shared_ptr<int64_t> calls = nullptr) {
  return [time, bodies = std::move(bodies), calls](std::span<const int32_t> F, std::span<const float> s) {
    RawPoses r;
    r.n = static_cast<int32_t>(F.size());
    r.bodies = static_cast<int32_t>(bodies.size());
    r.m.resize(F.size() * bodies.size() * 12);
    r.present.assign(F.size() * bodies.size(), 1);
    r.unknown.assign(F.size(), 0);
    for (size_t i = 0; i < F.size(); ++i) {
      const float c = TimeMap::realized(F[i], s[i]);
      const float cf = time.reported(c);
      r.c_final.push_back(static_cast<double>(cf));
      for (size_t b = 0; b < bodies.size(); ++b) {
        double frame = static_cast<double>(cf);
        for (const double k : bodies[b].keys) {
          if (std::fabs(frame - k) < bodies[b].plateau) frame = k;
        }
        to_matrix34(bodies[b].motion(frame), bodies[b].scale, &r.m[(i * bodies.size() + b) * 12]);
      }
    }
    if (calls) *calls += r.n;
    return r;
  };
}

// --- the prototype's analytic motions (t_synthetic.py), at 24 fps with frame_start 1: t = (frame - 1) / 24 s ---
inline constexpr double kFps = 24.0, kF0 = 1.0, kG = 9.81;

inline rot::Q qexp(rot::V v) { return so3::exp(v); }

// A ball (rho 0.05 m) dropped from 0.5 m, restitution 0.8, 0.5 m/s across; its spin rate changes at each impact.
// impacts: the impact times (s).
inline Pose bounce(double frame, std::vector<double>* impacts = nullptr) {
  const double t = (frame - kF0) / kFps;
  const double rho = 0.05, z0 = 0.5, e = 0.8, vx = 0.5;
  std::vector<double> imp{std::sqrt(2 * (z0 - rho) / kG)}, vs;
  double v = kG * imp[0];
  while (imp.back() < 3.0 && v * e > 0.5) {
    v *= e;
    vs.push_back(v);
    imp.push_back(imp.back() + 2 * v / kG);
  }
  vs.push_back(0.0);
  if (impacts) *impacts = imp;
  size_t arc = 0;
  while (arc < imp.size() && t >= imp[arc]) ++arc;
  double z;
  if (arc == 0) {
    z = z0 - 0.5 * kG * t * t;
  } else if (arc < imp.size()) {
    const double dt = t - imp[arc - 1];
    z = rho + vs[arc - 1] * dt - 0.5 * kG * dt * dt;
  } else {
    z = rho;
  }
  std::vector<double> rates{0.0}, edges{0.0};
  for (size_t k = 1; k <= imp.size(); ++k) rates.push_back((k % 2 ? -1.0 : 1.0) * (6.0 + static_cast<double>(k)));
  edges.insert(edges.end(), imp.begin(), imp.end());
  double cum = 0.0;
  for (size_t k = 1; k <= arc; ++k) cum += rates[k - 1] * (edges[k] - edges[k - 1]);
  const double ang = cum + rates[arc] * (t - edges[arc]);
  const double an = std::sqrt(0.09 + 1.0 + 0.04);
  return {qexp({ang * 0.3 / an, ang * 1.0 / an, ang * 0.2 / an}), {vx * t - 0.4, 0.0, z}};
}

// Spin-up to 2 rev/s with a C-infinity ramp, a 3-axis shake and a smooth translation: no breakpoint.
inline Pose smooth(double frame) {
  const double t = (frame - kF0) / kFps, pi = 3.141592653589793;
  const double w = 2 * pi * 2.0, T = 0.2;
  const double ang = w * (t - T * (1 - std::exp(-t / T)));
  const rot::Q shake = qexp({0.07 * std::sin(2 * pi * 6.3 * t), 0.06 * std::sin(2 * pi * 5.1 * t),
                             0.04 * std::sin(2 * pi * 3.7 * t)});
  return {shake * qexp({0.0, 0.0, ang}), {0.2 * std::sin(1.3 * t), 0.1 * std::cos(0.7 * t), 0.05 * std::sin(2.1 * t)}};
}

// LINEAR keys every 6 frames (0.25 s) from frame 1 to 73 (velocity kinks), constant before and after; slow rotation.
inline double linear_key_value(int k, int axis) {
  const uint32_t h = static_cast<uint32_t>(k * 2654435761u + static_cast<uint32_t>(axis) * 40503u + 12345u);
  return -0.15 + 0.3 * static_cast<double>(h % 10007u) / 10006.0;
}
inline std::vector<double> linear_keys() {
  std::vector<double> k;
  for (int i = 0; i <= 12; ++i) k.push_back(kF0 + 6.0 * i);
  return k;
}
inline Pose linear(double frame) {
  const double t = (frame - kF0) / kFps;
  const double u = std::clamp((frame - kF0) / 6.0, 0.0, 12.0);
  const int k = std::min(static_cast<int>(std::floor(u)), 11);
  const double a = u - k;
  rot::V p{(1 - a) * linear_key_value(k, 0) + a * linear_key_value(k + 1, 0),
           (1 - a) * linear_key_value(k, 1) + a * linear_key_value(k + 1, 1),
           (1 - a) * linear_key_value(k, 2) + a * linear_key_value(k + 1, 2)};
  return {qexp({0.0, 0.0, 0.3 * t}), p};
}

// CONSTANT keys every 12 frames (0.5 s): the position steps at each key (Blender: 1e-4 frames early); slow rotation.
inline std::vector<double> step_keys() {
  std::vector<double> k;
  for (int i = 0; i <= 5; ++i) k.push_back(kF0 + 12.0 * i);
  return k;
}
inline Pose step(double frame) {
  const double t = (frame - kF0) / kFps;
  const int k = std::clamp(static_cast<int>(std::floor((frame - kF0) / 12.0)), 0, 5);
  return {qexp({0.5 * std::sin(t), 0.0, 0.0}), {0.02 * k, 0.0, 0.01 * (k % 2)}};
}

// A static camera 1 m from the bodies (identity rotation; the prototype's CAM).
inline Pose camera(double) { return {{1, 0, 0, 0}, {0.0, -1.0, 0.2}}; }

// A Bezier rig with collinear (AUTO-like) handles: a C1 piecewise cubic (Catmull-Rom tangents, flat at the end keys)
// through keys every 12 frames, its curvature jumping at each key; a slow turn about z. No breakpoint: declared C1
// keys.
inline std::vector<double> rig_keys() { return {1, 13, 25, 37, 49, 61, 73}; }
inline Pose rig(double frame) {
  const std::vector<double> k = rig_keys();
  auto val = [](size_t i, int axis) { return 0.1 * std::sin(1.7 * static_cast<double>(i) + 0.9 * axis); };
  auto tan = [&](size_t i, int axis) {  // dv/dframe
    if (i == 0 || i + 1 == k.size()) return 0.0;
    return (val(i + 1, axis) - val(i - 1, axis)) / (k[i + 1] - k[i - 1]);
  };
  const double f = std::clamp(frame, k.front(), k.back());
  size_t i = std::min(static_cast<size_t>((f - k.front()) / 12.0), k.size() - 2);
  const double h = k[i + 1] - k[i], u = (f - k[i]) / h;
  const double h00 = 2 * u * u * u - 3 * u * u + 1, h10 = u * u * u - 2 * u * u + u, h01 = -2 * u * u * u + 3 * u * u,
               h11 = u * u * u - u * u;
  double p[3];
  for (int a = 0; a < 3; ++a) {
    p[a] = h00 * val(i, a) + h10 * h * tan(i, a) + h01 * val(i + 1, a) + h11 * h * tan(i + 1, a);
  }
  return {qexp({0.0, 0.0, 0.02 * (frame - kF0)}), {p[0], p[1], p[2]}};
}

// --- the tests' common setup ---
constexpr double kEps = 0.01;
constexpr int64_t kHmin = 16384;

inline TimeMap time24(int32_t frame_end) {
  TimeSettings s;
  s.fps = 24;
  s.frame_start = 1;
  s.frame_end = frame_end;
  return TimeMap(s);
}

// Bodies (columns 0 .. n-1, rho 0.05 m, unit scale at the export frame) watched by a static camera rig (column n) 1 m
// away, at G = 640 px/rad (MF).
inline MotionScene scene_of(size_t n) {
  MotionScene scene;
  scene.bodies.resize(n + 1);
  for (size_t b = 0; b < n; ++b) {
    scene.bodies[b].rho_m = 0.05;
    scene.bodies[b].S_ref = std::array<double, 9>{1, 0, 0, 0, 1, 0, 0, 0, 1};
  }
  scene.bodies[n].role = Role::kRig;
  scene.cameras.push_back({static_cast<int32_t>(n), {0, 0, 0}, 640.0, 0.0});
  return scene;
}

struct Run {
  std::vector<SampleSet> sets;
  RoundStats stats;
};

// Sampling and detection only, as BL-6 and BL-7 tested them (BL-8's fit tests pass FitOptions{}).
inline FitOptions no_fit() {
  FitOptions f;
  f.enabled = false;
  return f;
}

inline Run run(const Oracle& oracle, const TimeMap& time, const MotionScene& scene,
               const std::vector<std::vector<DeclaredKey>>& keys, int workers, const SampleOptions& opt = {},
               const DetectOptions& det = {}, const FitOptions& fit = no_fit()) {
  const ErrorModel model(scene);
  std::vector<BodyPipeline> pipes;
  for (size_t b = 0; b < scene.bodies.size(); ++b) {
    const std::vector<DeclaredKey> k = b < keys.size() ? keys[b] : std::vector<DeclaredKey>{};
    pipes.emplace_back(static_cast<int32_t>(b), Mode::kKeyed, time, k, opt, det, fit);
  }
  render::ThreadPool pool(workers);
  Run r;
  r.stats = drive(oracle, time, model, pipes, pool);
  for (auto& p : pipes) r.sets.push_back(std::move(p).take());
  return r;
}

inline std::vector<DeclaredKey> declared(const TimeMap& time, const std::vector<double>& frames, Continuity hint) {
  std::vector<DeclaredKey> out;
  for (const double f : frames) out.push_back({time.to_ns(static_cast<float>(f)), hint, true, 1e-4});
  return out;
}

}  // namespace balla::scene::motion::testing
