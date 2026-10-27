// P2-D D2 (plan ballasim_out/p2_plans/p2d.md §3.1 Step F, §3.6): the state of one probed sub-sample (render/shade/
// codes.h ProbeOut) from its traced hit and shadow mask, as the shade kernels see that sub-sample: its cluster key
// (instance, material), or (instance, patch) under accuracy.shading_patches crease (P2-F F5: shade.h hit_key), its
// mask as the signature takes it (accuracy.unlit_mask applied: shade.h lit_occlusion) and f_l of the requested shadow
// light at its surface point (accumulate's term, bit for bit). A probe is a request with
// k_count 1 (sub-sample k_begin) at any pose slot, so at a lattice time it reproduces render_into's state of that
// sub-sample on the same backend.
// In a scene with baked proxies (scans.md §4.2) the mask of a baked receiver is U2's occlusion word (inserted objects
// only, minus the scan's own shadow: lights.h shadow_mask), as render_into's; the static word, which render_into also
// adds to the signature, is not part of a probe's state, and a baked cluster's f is 0 (below).
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "render/ray.h"
#include "render/scene_view.h"
#include "render/shade/codes.h"
#include "render/shade/lights.h"
#include "render/shade/shade.h"
#include "render/surface.h"

namespace balla::render {

// req: the probe (k_count 1; its shadow light req.flags >> kProbeLightShift; f is 0 for a light out of range).
BALLA_HD inline ProbeOut probe_state(const SceneView& s, const PoseTableView& poses, const SampleRequest& req,
                                     const HitRecord& h, uint32_t occluded) {
  ProbeOut o{h.instance, 0u, occluded, 0.0f};
  const BodyView* body = h.instance != kInstanceNone ? find_body(s, h.instance) : nullptr;
  if (body == nullptr) return o;  // the environment: no surface terms (shadow_mask gave it 0)
  const GeometryView& geo = s.geometries[body->geometry];
  o.surface = hit_key(geo, h.prim);  // the cluster key: the material, or the patch under crease (P2-F F5)
  const SurfaceSample ss =
      surface_sample(geo, pose_row(poses, req.pose_slot, body->geometry), h.prim,
                     subsample_direction(s, s.cameras[req.cam], req.pixel, req.k_begin), false);
  if (s.unlit_mask != 0 && occluded != 0) o.mask = lit_occlusion(s, poses, req.pose_slot, ss.p, ss.n_shade, occluded);
  const uint32_t b = static_cast<uint32_t>(req.flags) >> kProbeLightShift;
  // A baked receiver (scans.md §4.1): accumulate adds no f there (light_terms), so f stays 0. The test reads the hit
  // triangle's own material (lights.h baked_receiver), not the cluster key o.surface.
  if (b < static_cast<uint32_t>(s.num_shadow_lights) && !baked_receiver(s, geo, h.prim)) {
    const int32_t j = s.shadow_light[b];
    o.f = lambert_irradiance(s.lights[j], light_row(poses, req.pose_slot, j), ss.p, ss.n_shade);
  }
  return o;
}

}  // namespace balla::render
