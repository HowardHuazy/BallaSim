// Launch parameters shared by optix_tracer.cpp (host) and programs.cu (OptiX device programs).
// One struct serves every pipeline; each raygen reads only its own fields.
#pragma once

#include <optix_types.h>
#include <stdint.h>

#include "render/scene_view.h"
#include "render/types.h"

namespace balla::render::optix {

struct LaunchParams {
  SceneView scene;                          // device views: cameras, warp, geometries, bodies, lights
  PoseTableView poses;
  const OptixTraversableHandle* handles;    // per trace geometry; 0 for geometries without triangles
  // sample pipelines
  const SampleRequest* requests;
  uint32_t num_requests;
  uint32_t subsamples;
  HitRecord* hits;
  uint32_t* occluded;                       // sample_shadow only
  uint32_t* static_occluded;                // sample_shadow_baked only: the static word (scans.md §4.2)
  // centre pipelines
  const GtRequest* gt_requests;
  GtHit* gt_hits;
  uint32_t num_gt;
  uint32_t* shadow_states;                  // centre_shadow: (state, t) per centre ray and shadow light
  uint32_t proxy_visibility;                // centre_shadow_baked only: the start diagnostic's states instead (lights.h)
};

}  // namespace balla::render::optix
