// OptiX 9.1 tracer: the production GPU backend (DESIGN §4.3, §6.9; G6). A direct GAS for the static
// world and one per mover, three compile-time-specialized pipelines (sample, shadow, centre) from
// one embedded OptiX-IR module, launches on the stream of the executor slot of each batch.
#pragma once

#include <memory>

#include "render/cuda/device_executor.h"
#include "render/tracer.h"

namespace balla::render {

class OptixTracer final : public Tracer {
 public:
  // The executor owns the device copies of the geometry and the streams; it must outlive the tracer.
  explicit OptixTracer(DeviceExecutor& executor);
  ~OptixTracer() override;

  MemorySpace memory_space() const override { return MemorySpace::kDevice; }
  // Once, on a fresh tracer (Renderer::Impl::build): the GAS of the executor's geometries in use; no held
  // state is reset. A failure leaves the GAS set partly freed, so the tracer must then be discarded.
  void build(const SceneData& scene, const SceneView& view) override;
  void use_traces(const TraceSet* traces, const SceneView& view) override;
  size_t held_device_bytes() const override;
  void sample(const TraceBatch& batch, HitRecord* hits, uint32_t* occluded,
              uint32_t* static_occluded = nullptr) override;
  void centre(const GtBatch& batch, GtHit* hits, uint32_t* shadow) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace balla::render
