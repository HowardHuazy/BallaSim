// The P2-B B0 census (ENGINE_CPP §0.1 "P2-B B0 notes"; PHASE2 D-P2-4 as amended for B1): of a pixel-sample's
// sub-samples k >= 16, those that B1's exact reuse rule would take from the 16 probes without a traversal, and of
// those the ones whose traced hit differs from the probes' (the rule's misses: occluders thinner than the probe
// spacing, coincident surfaces). The rule: the probes k < 16 all hit one (instance, prim) of the static world; no
// mover's bounding sphere meets the sub-sample's ray before that triangle (the tracers' culling test); and the ray's
// unclamped barycentrics on it are all >= 2^-10. A counter only (execution.census): no output depends on it.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "render/ray.h"
#include "render/scene_view.h"

namespace balla::render {

inline constexpr uint32_t kCensusProbes = 16;
inline constexpr float kCensusMargin = 0x1p-10f;

struct CensusCounts {
  uint32_t reusable = 0, mismatched = 0;
};

// One pixel-sample's census from its req.k_count hit records.
BALLA_HD inline CensusCounts census_sample(const SceneView& s, const PoseTableView& poses, const SampleRequest& req,
                                           const HitRecord* hits) {
  CensusCounts c;
  const uint32_t n = req.k_count, inst = hits[0].instance, prim = hits[0].prim;
  if (n <= kCensusProbes || inst == kInstanceNone) return c;
  for (uint32_t k = 1; k < kCensusProbes; ++k) {
    if (hits[k].instance != inst || hits[k].prim != prim) return c;
  }
  const BodyView* body = find_body(s, inst);
  if (body == nullptr || body->geometry != 0) return c;  // a mover's triangle: traced
  const Triangle tri = fetch_triangle(s.geometries[0], prim);
  const PoseRow& row = pose_row(poses, req.pose_slot, 0);
  const Vec3f e1 = tri.v1 - tri.v0, e2 = tri.v2 - tri.v0;
  const Vec3f tvec = row_origin(row) - tri.v0;  // primary rays start at the camera centre (ray.h)
  const Vec3f qvec = cross(tvec, e1);
  const CameraView& cam = s.cameras[req.cam];
  for (uint32_t k = kCensusProbes; k < n; ++k) {
    const Vec3f d_cam = subsample_direction(s, cam, req.pixel, req.k_begin + k);
    const Vec3f d = rotate(row, d_cam);  // Moeller-Trumbore as reintersect, without the clamp
    const Vec3f pvec = cross(d, e2);
    const float det = dot(e1, pvec);
    if (det == 0.0f) continue;
    const float inv = 1.0f / det;
    const float t = dot(e2, qvec) * inv, b1 = dot(tvec, pvec) * inv, b2 = dot(d, qvec) * inv;
    if (!(t > 0.0f && b1 >= kCensusMargin && b2 >= kCensusMargin && (1.0f - b1) - b2 >= kCensusMargin)) continue;
    bool mover = false;
    for (int32_t g = 1; g < s.num_geometries && !mover; ++g) {
      mover = sphere_overlaps(pose_row(poses, req.pose_slot, g), Vec3f{0.0f, 0.0f, 0.0f}, d_cam, t);
    }
    if (mover) continue;
    ++c.reusable;
    if (hits[k].instance != inst || hits[k].prim != prim) ++c.mismatched;
  }
  return c;
}

// Adds a pixel-sample's census to counters v (kCensusWords), by its class word (none: the rest).
BALLA_HD inline void census_add(uint64_t* v, const SampleRequest& req, CensusCounts c, const uint32_t* klass) {
  const bool band = klass != nullptr && (*klass & (kClassMulti | kClassShadowEdge)) != 0;
  uint64_t* w = v + (band ? kCensusBand : kCensusRest);
  w[kCensusSubsamples] += req.k_count > kCensusProbes ? req.k_count - kCensusProbes : 0u;
  w[kCensusReusable] += c.reusable;
  w[kCensusMismatched] += c.mismatched;
}

}  // namespace balla::render
