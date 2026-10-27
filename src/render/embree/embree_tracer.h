// Embree 4 tracer: the reference and CPU backend (DESIGN §4.3). Robust mode, high-quality builds,
// one scene for the static world and one per mover, exact per-ray transforms from the pose table
// and rtcIntersect1/rtcOccluded1 per geometry, multithreaded over fixed request chunks.
#pragma once

#include <memory>

#include "render/thread_pool.h"
#include "render/tracer.h"

namespace balla::render {

class EmbreeTracer final : public Tracer {
 public:
  explicit EmbreeTracer(ThreadPool& pool);
  ~EmbreeTracer() override;

  MemorySpace memory_space() const override { return MemorySpace::kHost; }
  void build(const SceneData& scene, const SceneView& view) override;
  void use_traces(const TraceSet* traces, const SceneView& view) override;
  void sample(const TraceBatch& batch, HitRecord* hits, uint32_t* occluded,
              uint32_t* static_occluded = nullptr) override;
  void centre(const GtBatch& batch, GtHit* hits, uint32_t* shadow) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace balla::render
