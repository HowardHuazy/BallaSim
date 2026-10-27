// Backend interface of the renderer (DESIGN §4.3): the only backend-specific surface. This is the
// completed form of the draft in backend/api.h, whose SampleRequest and HitRecord it uses.
//
// Motion is never delegated to a backend: every tracer traces static acceleration structures (the
// merged static world in world coordinates, one per mover in its body frame, and in scenes with baked proxies one
// per split static body in world coordinates, scans.md §4.2) with the camera-relative poses of the pose table, and
// generates its rays with the shared code of render/ray.h. Hence Embree and OptiX trace bitwise-identical rays and
// differ only in triangle intersection arithmetic (DESIGN §12.1 L2).
//
// All pointers handed to a tracer live in its memory space (host for Embree, device for OptiX);
// device tracers run asynchronously on the stream of their executor.
#pragma once

#include <cstdint>

#include "render/scene_view.h"
#include "render/types.h"

namespace balla::render {

class SceneData;
struct TraceSet;

enum class MemorySpace { kHost, kDevice };

// One launch worth of pixel-samples. Every request of a launch renders the same N sub-samples, so it
// has k_count == subsamples and hit/shadow record r * subsamples + k belongs to sub-sample
// k_begin + k of request r. (The engine's per-pixel N, ENGINE_CPP P2-B B2, renders each N class in a
// render_into call of its own; a variable k_count within a launch would need a group descriptor array.)
struct TraceBatch {
  const SampleRequest* requests;
  uint32_t num_requests;
  uint32_t subsamples;
  PoseTableView poses;
  void* stream;   // the executor slot's cudaStream_t (device tracers); nullptr on the host
  uint32_t slot;  // executor slot (device tracers keep per-slot launch parameters)
};

// Centre rays of one launch.
struct GtBatch {
  const GtRequest* requests;
  uint32_t count;
  PoseTableView poses;
  void* stream;
  uint32_t slot;
  // The start diagnostic of scenes with baked proxies (Renderer::proxy_visibility; lights.h proxy_visibility_states):
  // `shadow` then receives, instead of the plan pass's states, per proxy shadow light 0 (no proxy ray from this hit),
  // 1 (traced, the scan does not block it) or 2 (blocked by the scan), and a second word 0.
  bool proxy_visibility = false;
};

class Tracer {
 public:
  virtual ~Tracer() = default;

  virtual MemorySpace memory_space() const = 0;

  // Builds the static-world structure and one per mover from the host geometry of `scene`; `view`
  // is the scene in the tracer's memory space (ray generation reads cameras and the warp from it).
  virtual void build(const SceneData& scene, const SceneView& view) = 0;

  // Traces the geometries of a held set from now on (nullptr: the scene's own; ENGINE_CPP I3b): builds the
  // structure of its merged world, keeps the movers' ones. `view` is the executor's view of the same set,
  // after its use_traces. nullptr needs no memory; after a failure, the caller restores with it (hold does).
  virtual void use_traces(const TraceSet* traces, const SceneView& view) = 0;
  // Device memory of the held set's structure (0 on the host).
  virtual size_t held_device_bytes() const { return 0; }

  // Closest hit of every sub-sample ray: the sphere-culled movers in ascending instance id, then the
  // static world, each with tmax = the closest hit so far (the shared order fixes tie-breaking);
  // hits[r * subsamples + k]. No atomics, fixed slots. When occluded is
  // not null, the same launch also traces the hit's shadow rays (render/shade/lights.h: shadow_mask;
  // movers before the world) and writes each sub-sample's mask; a separate shadow launch would
  // regenerate every ray and refetch every triangle. When static_occluded is not null (scenes with baked proxies,
  // SceneView::baked_proxies, and occluded not null; the renderer passes it then), the launch also writes each
  // sub-sample's static word (scans.md §4.2: per shadow bit, a proxy whose ray the scan blocks, in a light group with
  // an inserted blocker) and a baked receiver's occlusion bits follow the inserted / scan rule (lights.h shadow_mask);
  // 0 for lit receivers and misses. Without occluded, the words are 0.
  virtual void sample(const TraceBatch& batch, HitRecord* hits, uint32_t* occluded,
                      uint32_t* static_occluded = nullptr) = 0;

  // Closest hits of centre rays (ground truth and the dense API). When shadow is not null, it receives
  // (state, blocker t) per ray and shadow light (lights.h: shadow_states; the plan pass, ENGINE_CPP I8; in a scene with
  // baked proxies a baked receiver's proxies follow scans.md §4.2's plan-pass rule), or with GtBatch::proxy_visibility
  // the start diagnostic's words.
  virtual void centre(const GtBatch& batch, GtHit* hits, uint32_t* shadow) = 0;
};

}  // namespace balla::render
