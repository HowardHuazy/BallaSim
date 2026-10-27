// Centre-ray ground truth (DESIGN §9.1): instance, semantic, depth and the instantaneous motion
// field of the surface point seen through a pixel position, from the exact pose and the analytic
// relative twist of the hit body (M2 D3, D8). Shared by the CPU executor and the CUDA kernel, so
// per-event GT and dense queries are bit-identical on one backend. P2-D D1: when normal is not null, also the hit
// triangle's unit geometric normal in the camera frame, facing the camera (surface.h's n_geo; NaN on a miss), 3 floats.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"
#include "render/ray.h"
#include "render/scene_view.h"
#include "render/shade/shade.h"
#include "render/types.h"

namespace balla::render {

BALLA_HD inline GtSample gt_eval(const SceneView& s, const PoseTableView& poses, const GtRequest& req,
                                 const GtHit& hit, float* normal = nullptr) {
  const float nan = quiet_nan<float>();
  GtSample out{kInstanceNone, 0, nan, nan, nan};
  if (normal != nullptr) normal[0] = normal[1] = normal[2] = nan;
  if (hit.instance == kInstanceNone) return out;
  const BodyView* body = find_body(s, hit.instance);
  if (body == nullptr) return out;
  const CameraView& cam = s.cameras[req.cam];
  const PoseRow& row = pose_row(poses, req.pose_slot, body->geometry);
  const Vec3f d = centre_direction(cam, req.x, req.y);
  const SurfaceHit h =
      reintersect(fetch_triangle(s.geometries[body->geometry], hit.prim), primary_to_geometry(row, d));
  const Vec3f x = h.t * d;
  if (normal != nullptr) {
    Vec3f n = rotate_inverse(row, normalize(h.n));
    if (dot(n, d) > 0.0f) n = -n;
    normal[0] = n.x;
    normal[1] = n.y;
    normal[2] = n.z;
  }
  out.instance = hit.instance;
  out.semantic = body->semantic;
  out.depth = x.z;
  // X_C' = omega x X_C + v (M2 D3); pixel velocity = J_proj(X_C) X_C' (M2 D8).
  const Vec3f xdot = cross(Vec3f{row.omega.x, row.omega.y, row.omega.z}, x) + Vec3f{row.v.x, row.v.y, row.v.z};
  camera::Mat23<float> j;
  if (camera_jacobian(cam, x, &j)) {
    out.flow_x = (j.m[0][0] * xdot.x + j.m[0][1] * xdot.y) + j.m[0][2] * xdot.z;
    out.flow_y = (j.m[1][0] * xdot.x + j.m[1][1] * xdot.y) + j.m[1][2] * xdot.z;
  }
  return out;
}

}  // namespace balla::render
