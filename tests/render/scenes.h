// Scene bundles for the renderer tests: planes, boxes, analytic trajectories and pinhole cameras.
#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/math/quat.h"
#include "core/math/so3.h"
#include "core/math/vec.h"
#include "render/bundle.h"
#include "render/renderer.h"

namespace balla::render::test {

inline constexpr int64_t kKnotNs = 5'000'000;  // 5 ms knots, as fitted EVIMO2 trajectories

// Trajectory over [t_begin, t_end] with control points c = 0..n-1 at times t0 + c h, where the
// control pose of knot c is pose(c * h_seconds). With the K = 6 cumulative basis, controls linear
// in c (positions) or with constant increments about one axis (rotations) give exactly constant
// velocity and angular velocity (linear precision of uniform B-splines).
template <class PoseFn>
Trajectory make_trajectory(int64_t t_begin, int64_t t_end, PoseFn pose) {
  Trajectory t;
  t.order = 6;
  t.h_ns = kKnotNs;
  t.t0_ns = t_begin - 3 * kKnotNs;
  const int64_t segments = (t_end - t.t0_ns) / kKnotNs + 3;
  const int64_t n = segments + t.order - 1;
  for (int64_t c = 0; c < n; ++c) {
    Quat<double> q;
    Vec3<double> p;
    pose(static_cast<double>(c) * 1e-9 * kKnotNs, &q, &p);
    t.rot.insert(t.rot.end(), {q.w, q.x, q.y, q.z});
    t.pos.insert(t.pos.end(), {p.x, p.y, p.z});
  }
  t.span_begin_ns = t_begin;
  t.span_end_ns = t_end;
  return t;
}

inline Trajectory static_trajectory(int64_t t_begin, int64_t t_end, Quat<double> q, Vec3<double> p) {
  return make_trajectory(t_begin, t_end, [&](double, Quat<double>* qo, Vec3<double>* po) {
    *qo = q;
    *po = p;
  });
}

// Constant velocity v (m/s) and angular velocity w (rad/s, world frame) through pose (q0, p0) at
// knot 0.
inline Trajectory moving_trajectory(int64_t t_begin, int64_t t_end, Quat<double> q0, Vec3<double> p0,
                                    Vec3<double> v, Vec3<double> w) {
  return make_trajectory(t_begin, t_end, [&](double s, Quat<double>* qo, Vec3<double>* po) {
    *qo = so3::exp(s * w) * q0;
    *po = p0 + s * v;
  });
}

// One piece of piecewise_trajectory: it serves [begin, the next begin), its knots start at t0 <= begin with spacing h,
// and its controls follow moving_trajectory's motion through (identity, p0) at its first knot with velocity v and
// angular velocity w, so pieces move differently (the wrong piece gives other values at any time).
struct PieceSpec {
  int64_t begin, t0, h;
  Vec3<double> p0, v, w;
  int8_t join = 0;
};

// A piecewise trajectory (C1 "pieces", docs/design/phase3/blender.md §3.7) of order K over [spec[0].begin, t_end]:
// each piece gets the fewest rows whose support covers its interval.
inline Trajectory piecewise_trajectory(int64_t t_end, const std::vector<PieceSpec>& spec, int32_t order = 6) {
  Trajectory t;
  t.order = order;
  for (size_t k = 0; k < spec.size(); ++k) {
    const PieceSpec& s = spec[k];
    const int64_t end = k + 1 < spec.size() ? spec[k + 1].begin : t_end;
    const int64_t rows = (end - s.t0 + s.h - 1) / s.h + order - 1;
    TrajectoryPiece piece{s.begin, s.t0, s.h, static_cast<int32_t>(t.pos.size() / 3), 0, s.join};
    for (int64_t c = 0; c < rows; ++c) {
      const double secs = static_cast<double>(c * s.h) * 1e-9;
      const Quat<double> q = so3::exp(secs * s.w);
      const Vec3<double> p = s.p0 + secs * s.v;
      t.rot.insert(t.rot.end(), {q.w, q.x, q.y, q.z});
      t.pos.insert(t.pos.end(), {p.x, p.y, p.z});
    }
    piece.ctrl_end = static_cast<int32_t>(t.pos.size() / 3);
    t.pieces.push_back(piece);
  }
  t.t0_ns = spec[0].t0;
  t.h_ns = spec[0].h;
  t.span_begin_ns = spec[0].begin;
  t.span_end_ns = t_end;
  return t;
}

// Pinhole camera (zero distortion) looking along +z of its rig.
inline Camera pinhole(int width, int height, double f, int rig_trajectory) {
  Camera c;
  c.name = "cam";
  c.model = CameraModelKind::kRadtan;
  c.width = width;
  c.height = height;
  c.fx = c.fy = f;
  c.cx = 0.5 * (width - 1);
  c.cy = 0.5 * (height - 1);
  c.rig_trajectory = rig_trajectory;
  return c;
}

// Square plane of half-size `half` at z = 0 of its body frame, normal +z, split into n x n quads;
// UVs map [-half, half]^2 to [0, 1]^2 with v down the image (y down).
inline Mesh plane_mesh(double half, int n = 1) {
  Mesh m;
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      const double u = static_cast<double>(i) / n, v = static_cast<double>(j) / n;
      m.positions.insert(m.positions.end(), {static_cast<float>(-half + 2 * half * u),
                                             static_cast<float>(-half + 2 * half * v), 0.0f});
      m.normals.insert(m.normals.end(), {0.0f, 0.0f, 1.0f});
      m.uvs.insert(m.uvs.end(), {static_cast<float>(u), static_cast<float>(v)});
    }
  }
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const uint32_t a = static_cast<uint32_t>(j * (n + 1) + i), b = a + 1, c = a + static_cast<uint32_t>(n) + 1, d = c + 1;
      m.indices.insert(m.indices.end(), {a, b, d, a, d, c});
    }
  }
  return m;
}

// Axis-aligned box centred at the body origin with half-extents (hx, hy, hz).
inline Mesh box_mesh(float hx, float hy, float hz) {
  Mesh m;
  m.positions = {-hx, -hy, -hz, hx, -hy, -hz, hx, hy, -hz, -hx, hy, -hz,
                 -hx, -hy, hz,  hx, -hy, hz,  hx, hy, hz,  -hx, hy, hz};
  m.indices = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
               2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7};
  return m;
}

inline std::array<double, 16> translation(double x, double y, double z) {
  return {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1};
}

inline Body make_body(uint32_t id, Mesh mesh, std::array<float, 3> color = {1.0f, 1.0f, 1.0f}) {
  Body b;
  b.id = id;
  b.name = "body" + std::to_string(id);
  b.semantic = id + 100;
  b.mesh = std::move(mesh);
  Material m;
  m.base_color = color;
  b.materials.push_back(m);
  return b;
}

inline Light ambient_light(float intensity, int group = 0) {
  Light l;
  l.kind = BundleLightKind::kAmbient;
  l.group = group;
  l.intensity = intensity;
  return l;
}

// A background radiance for the rays that miss (D13 item 1); casts_shadows stays at its default, which is ignored.
inline Light background_light(float radiance, int group = 0) {
  Light l;
  l.kind = BundleLightKind::kBackground;
  l.group = group;
  l.intensity = radiance;
  return l;
}

inline Light point_light(std::array<double, 3> position, float intensity, bool shadows = true, int group = 0) {
  Light l;
  l.kind = BundleLightKind::kPoint;
  l.group = group;
  l.intensity = intensity;
  l.position = position;
  l.casts_shadows = shadows;
  return l;
}

inline RendererOptions options(BackendKind backend, int threads = 4) {
  RendererOptions o;
  o.backend = backend;
  o.num_threads = threads;
  o.scene.sensor_rgb_weights = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
  return o;
}

// Backends usable in this process.
inline std::vector<BackendKind> backends() {
  std::vector<BackendKind> b{BackendKind::kEmbree};
  if (optix_available()) b.push_back(BackendKind::kOptix);
  return b;
}

inline std::string backend_name(BackendKind b) { return b == BackendKind::kOptix ? "optix" : "embree"; }

}  // namespace balla::render::test
