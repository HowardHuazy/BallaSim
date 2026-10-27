// Ray generation, frame changes and the triangle re-intersection shared by every backend and by
// the shading and ground-truth kernels (DESIGN §4.3: request-level interface, rays are never
// materialized). The tracer and the cluster kernel regenerate the same sub-sample ray with this
// code, so a hit record (instance, prim) is all the shading needs.
#pragma once

#include <stdint.h>

#include "core/camera/ray_lut.h"
#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"
#include "core/rng/subsample.h"
#include "render/scene_view.h"

namespace balla::render {

using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;

// Direction (camera frame) of sub-sample k of a pixel: the warped Owen-Sobol offset (DESIGN §6.1,
// core/rng/subsample.h) through the camera's ray LUT. Rays are not normalized: for radtan cameras
// the ray parameter is the camera-frame depth z.
BALLA_HD inline Vec3f subsample_direction(const SceneView& s, const CameraView& cam, uint32_t pixel, uint32_t k) {
  const uint32_t w = static_cast<uint32_t>(cam.width);
  const uint32_t x = pixel % w, y = pixel / w;
  const Vec2f off = rng::subsample_offset(s.warp, cam.pattern_seeds[pixel], k);
  return camera::lut_ray(cam.lut, Vec2f{static_cast<float>(x) + off.x, static_cast<float>(y) + off.y});
}

// Direction of the ray through a pixel position (centre ray of DESIGN §9.1).
BALLA_HD inline Vec3f centre_direction(const CameraView& cam, float x, float y) {
  return camera::lut_ray(cam.lut, Vec2f{x, y});
}

struct Ray {
  Vec3f o, d;
};

BALLA_HD inline Vec3f rotate(const PoseRow& row, Vec3f v) {
  const Float4 a = row.m[0], b = row.m[1], c = row.m[2];
  return {a.x * v.x + a.y * v.y + a.z * v.z, b.x * v.x + b.y * v.y + b.z * v.z, c.x * v.x + c.y * v.y + c.z * v.z};
}

BALLA_HD inline Vec3f rotate_inverse(const PoseRow& row, Vec3f v) {
  const Float4 a = row.m[0], b = row.m[1], c = row.m[2];
  return {a.x * v.x + b.x * v.y + c.x * v.z, a.y * v.x + b.y * v.y + c.y * v.z, a.z * v.x + b.z * v.y + c.z * v.z};
}

// Camera origin in the geometry frame.
BALLA_HD inline Vec3f row_origin(const PoseRow& row) { return {row.m[0].w, row.m[1].w, row.m[2].w}; }

// Camera-frame ray -> geometry frame. Rigid, so the ray parameter t is preserved.
BALLA_HD inline Ray to_geometry(const PoseRow& row, Vec3f o_cam, Vec3f d_cam) {
  return {rotate(row, o_cam) + row_origin(row), rotate(row, d_cam)};
}

// Primary rays start at the camera centre, whose geometry-frame position is the row's p.
BALLA_HD inline Ray primary_to_geometry(const PoseRow& row, Vec3f d_cam) {
  return {row_origin(row), rotate(row, d_cam)};
}

BALLA_HD inline Vec3f point_to_camera(const PoseRow& row, Vec3f x_g) {
  return rotate_inverse(row, x_g - row_origin(row));
}

// Conservative test whether the camera-frame segment o + t d, t in [0, tmax], meets the bounding
// sphere of a mover (G6 §2.1 sphere culling). The radius is inflated by 1e-4 relative + 1e-5 m, so that
// fp32 rounding culls no ray the exact test keeps for spheres of radius >= 1 mm within 300 m (measured).
// Farther, a ray up to about 4e-8 x distance inside a small sphere may be culled (4e-5 m, 4e-8 rad at 1 km):
// far below a pixel, like the fp32 rounding of the row's own translation there. The discriminant uses the
// closest-approach form, which does not cancel for distant spheres.
BALLA_HD inline bool sphere_overlaps(const PoseRow& row, Vec3f o, Vec3f d, float tmax) {
  const Float4 sphere = row.sphere;
  const float r = sphere.w * 1.0001f + 1e-5f;
  const Vec3f oc = o - Vec3f{sphere.x, sphere.y, sphere.z};
  const float a = dot(d, d);
  const float b = dot(oc, d);
  const Vec3f closest = oc - (b / a) * d;
  const float disc = r * r - dot(closest, closest);
  if (disc < 0.0f) return false;
  const float s = sqrt(a * disc);
  return (-b - s) <= tmax * a && (-b + s) >= 0.0f;
}

struct Triangle {
  Vec3f v0, v1, v2;
  uint32_t i0, i1, i2;
};

BALLA_HD inline Vec3f load3(const float* a, uint32_t i) {
  return {a[3 * static_cast<uint64_t>(i)], a[3 * static_cast<uint64_t>(i) + 1],
          a[3 * static_cast<uint64_t>(i) + 2]};
}

BALLA_HD inline Vec2f load2(const float* a, uint32_t i) {
  return {a[2 * static_cast<uint64_t>(i)], a[2 * static_cast<uint64_t>(i) + 1]};
}

BALLA_HD inline Triangle fetch_triangle(const GeometryView& g, uint32_t prim) {
  const uint64_t b = 3 * static_cast<uint64_t>(prim);
  Triangle t;
  t.i0 = g.indices[b];
  t.i1 = g.indices[b + 1];
  t.i2 = g.indices[b + 2];
  t.v0 = load3(g.positions, t.i0);
  t.v1 = load3(g.positions, t.i1);
  t.v2 = load3(g.positions, t.i2);
  return t;
}

// fp32 re-intersection of a ray with the triangle a backend reported (DESIGN §4.3: hit records
// carry only hint barycentrics). Moeller-Trumbore gives the plane distance t and the barycentrics
// (b1, b2) of vertices 1 and 2; the barycentrics are clamped onto the triangle, because a backend
// may report a hit a rounding error outside the edge. n is the unnormalized geometric normal
// e1 x e2 in the geometry frame.
struct SurfaceHit {
  float t;
  float b1, b2;
  Vec3f n;
};

BALLA_HD inline SurfaceHit reintersect(const Triangle& tri, const Ray& ray) {
  const Vec3f e1 = tri.v1 - tri.v0, e2 = tri.v2 - tri.v0;
  const Vec3f pvec = cross(ray.d, e2);
  const float det = dot(e1, pvec);
  const Vec3f tvec = ray.o - tri.v0;
  const Vec3f qvec = cross(tvec, e1);
  SurfaceHit h;
  h.n = cross(e1, e2);
  if (det == 0.0f) {  // ray in the triangle's plane: degenerate grazing hit
    h.t = 0.0f;
    h.b1 = h.b2 = 1.0f / 3.0f;
    return h;
  }
  const float inv = 1.0f / det;
  h.t = dot(e2, qvec) * inv;
  float b1 = clamp(dot(tvec, pvec) * inv, 0.0f, 1.0f);
  float b2 = clamp(dot(ray.d, qvec) * inv, 0.0f, 1.0f);
  const float s = b1 + b2;
  if (s > 1.0f) {
    b1 = b1 / s;
    b2 = b2 / s;
  }
  h.b1 = b1;
  h.b2 = b2;
  return h;
}

BALLA_HD inline Vec3f normalize(Vec3f v) {
  const float n2 = dot(v, v);
  return n2 > 0.0f ? (1.0f / sqrt(n2)) * v : v;
}

// Origin of a secondary ray leaving a surface point p along the unit normal n, offset by a few
// ulps of p in integer space so it cannot re-hit the surface (Waechter & Binder, "A Fast and
// Robust Method for Avoiding Self-Intersection", Ray Tracing Gems, 2019). Integer offsets are
// exact, so host and device agree bitwise.
BALLA_HD inline Vec3f offset_origin(Vec3f p, Vec3f n) {
  constexpr float kOrigin = 1.0f / 32.0f;
  constexpr float kFloatScale = 1.0f / 65536.0f;
  constexpr float kIntScale = 256.0f;
  auto axis = [&](float pc, float nc) {
    const int32_t of = static_cast<int32_t>(kIntScale * nc);
    const int32_t bits = static_cast<int32_t>(float_to_bits(pc));
    const float pi = bits_to_float(static_cast<uint32_t>(pc < 0.0f ? bits - of : bits + of));
    return abs(pc) < kOrigin ? pc + kFloatScale * nc : pi;
  };
  return {axis(p.x, n.x), axis(p.y, n.y), axis(p.z, n.z)};
}

}  // namespace balla::render
