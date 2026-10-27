// Surface attributes of one sub-sample hit, recomputed from its hit record (DESIGN §4.3, §7.3):
// the cluster kernel regenerates the ray, fetches the triangle and re-intersects it in fp32, so the
// attributes are smooth in time even on large triangles and identical for every backend that
// reports the same (instance, prim).
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/vec.h"
#include "render/ray.h"
#include "render/scene_view.h"

namespace balla::render {

struct SurfaceSample {
  Vec3f p;               // hit point, camera frame
  Vec3f n_geo;           // unit geometric normal, camera frame, facing the camera
  Vec3f n_shade;         // unit shading normal, camera frame, on the camera side of n_geo
  Vec2f uv;              // texture coordinates (zero without UVs)
  Vec3f grad_u, grad_v;  // camera-frame gradients of u and v along the triangle's plane
};

// Gradient of a function linear on the triangle with vertex differences (f1 - f0, f2 - f0), for
// edges e1, e2 and normal n = e1 x e2: (df1 (e2 x n) + df2 (n x e1)) / |n|^2, which lies in the
// plane and satisfies grad . e1 = df1, grad . e2 = df2.
BALLA_HD inline Vec3f plane_gradient(Vec3f e1, Vec3f e2, Vec3f n, float df1, float df2) {
  const float n2 = dot(n, n);
  if (!(n2 > 0.0f)) return {0.0f, 0.0f, 0.0f};
  return (1.0f / n2) * (df1 * cross(e2, n) + df2 * cross(n, e1));
}

// Surface of sub-sample ray d_cam (from the camera centre) on triangle prim of geometry g.
// with_uv selects the UV interpolation and gradients, which only textured materials need.
BALLA_HD inline SurfaceSample surface_sample(const GeometryView& g, const PoseRow& row, uint32_t prim,
                                             Vec3f d_cam, bool with_uv) {
  const Triangle tri = fetch_triangle(g, prim);
  const SurfaceHit h = reintersect(tri, primary_to_geometry(row, d_cam));
  SurfaceSample s;
  s.p = h.t * d_cam;
  Vec3f n_geo = rotate_inverse(row, normalize(h.n));
  if (dot(n_geo, d_cam) > 0.0f) n_geo = -n_geo;
  s.n_geo = n_geo;
  const float b0 = (1.0f - h.b1) - h.b2;
  s.n_shade = n_geo;
  if (g.normals != nullptr) {
    const Vec3f n = (b0 * load3(g.normals, tri.i0) + h.b1 * load3(g.normals, tri.i1)) +
                    h.b2 * load3(g.normals, tri.i2);
    if (dot(n, n) > 1e-12f) {  // zero where the source mesh had no normals: keep n_geo
      Vec3f ns = rotate_inverse(row, normalize(n));
      if (dot(ns, n_geo) < 0.0f) ns = -ns;  // two-sided shading: follow the visible side
      s.n_shade = ns;
    }
  }
  s.uv = {0.0f, 0.0f};
  s.grad_u = s.grad_v = {0.0f, 0.0f, 0.0f};
  if (with_uv && g.uvs != nullptr) {
    const Vec2f t0 = load2(g.uvs, tri.i0), t1 = load2(g.uvs, tri.i1), t2 = load2(g.uvs, tri.i2);
    s.uv = (b0 * t0 + h.b1 * t1) + h.b2 * t2;
    const Vec3f e1 = tri.v1 - tri.v0, e2 = tri.v2 - tri.v0;
    s.grad_u = rotate_inverse(row, plane_gradient(e1, e2, h.n, t1.x - t0.x, t2.x - t0.x));
    s.grad_v = rotate_inverse(row, plane_gradient(e1, e2, h.n, t1.y - t0.y, t2.y - t0.y));
  }
  return s;
}


}  // namespace balla::render
