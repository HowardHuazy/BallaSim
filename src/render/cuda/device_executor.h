// Device executor: the scene, the pose-table mirror, the window pose table and the batch buffers in CUDA
// memory, and the shared cluster-shading and ground-truth code (render/shade/shade.h, render/gt.h) as
// CUDA kernels. Two batch slots, each with its own stream (shared with the OptiX tracer) and buffers,
// let one chunk be shaded while the next is traced. All pointers are device pointers, and nothing
// synchronizes the host unless stated.
#pragma once

#include <memory>
#include <vector>

#include "render/executor.h"

namespace balla::render {

class DeviceExecutor final : public Executor {
 public:
  DeviceExecutor();  // throws std::runtime_error without a CUDA device
  ~DeviceExecutor() override;
  DeviceExecutor(const DeviceExecutor&) = delete;
  DeviceExecutor& operator=(const DeviceExecutor&) = delete;

  // Trace geometries in use with device pointers, as host structs (the tracer builds its GAS from them).
  const std::vector<GeometryView>& device_geometries() const;

  MemorySpace memory_space() const override { return MemorySpace::kDevice; }
  uint32_t batch_slots() const override;
  void* stream(uint32_t slot) const override;
  void upload_scene(const SceneData& scene) override;
  const SceneView& scene_view() const override;
  void use_traces(const TraceSet* traces) override;
  size_t held_device_bytes() const override;
  PoseTableView sync_poses(PoseTable& table, uint32_t slot) override;
  // The window pose table's rows of newly used slots: flagged and scanned on the device, then evaluated
  // after one host sync that sizes them.
  void begin_window(uint16_t cam, const WindowClock& clock, void* stream) override;
  void assign_pose_rows(SampleRequest* requests, uint32_t n, void* stream) override;
  PoseTableView window_poses() const override;
  HitRecord* hit_buffer(uint32_t slot, size_t count) override;
  uint32_t* occlusion_buffer(uint32_t slot, size_t count) override;
  uint32_t* static_occlusion_buffer(uint32_t slot, size_t count) override;
  GtHit* gt_hit_buffer(uint32_t slot, size_t count) override;
  void shade_into(uint32_t slot, const TraceBatch& batch, const HitRecord* hits, const uint32_t* occluded, float* irr,
                  uint64_t* sig, uint32_t* sub, const uint32_t* out_index, uint32_t* klass,
                  const PvOut* pv = nullptr, const uint32_t* static_occluded = nullptr,
                  float* irr_unshadowed = nullptr) override;
  void probe_into(uint32_t slot, const TraceBatch& batch, const HitRecord* hits, const uint32_t* occluded,
                  ProbeOut* out) override;
  PvOut pv_scratch(size_t n, size_t code_words, size_t table_words, void* stream) override;
  void pv_scratch_add(uint64_t entry_bytes, void* stream) override;
  void pv_scratch_totals(uint64_t* totals) override;
  void census_into(uint32_t slot, const TraceBatch& batch, const HitRecord* hits, const uint32_t* out_index,
                   const uint32_t* klass) override;
  void census_totals(uint64_t* totals) override;
  const GtRequest* gt_query_rows(uint32_t slot, uint16_t cam, const GtQuery* queries, uint32_t n,
                                 PoseTableView* poses) override;
  void evaluate_gt_into(uint32_t slot, const GtRequest* requests, const GtHit* hits, uint32_t count,
                        const PoseTableView& poses, GtSample* out, float* normal) override;
  void join(void* waiter, void* signaller) override;
  void wait(uint32_t slot) override;
  void start_timing(void* stream) override;
  uint32_t timing_mark(uint32_t slot) override;
  double timing_ms(uint32_t index) override;
  void reset_timing() override;
  void* stage(StageBuffer id, void* host, size_t bytes, bool upload, void* stream) override;
  void unstage(void* host, const void* staged, size_t bytes, void* stream) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace balla::render
