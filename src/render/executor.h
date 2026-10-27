// Memory-space-specific storage and execution of the shared, backend-independent kernels: the
// cluster shading of render/shade/shade.h and the ground truth of render/gt.h. The host executor
// runs them on the thread pool (Embree path); the device executor runs the same BALLA_HD code as
// CUDA kernels (OptiX path), so both give bitwise-equal results from equal hits.
//
// Batches go through numbered slots. The device executor has two, each with its own CUDA stream
// and buffers, so the GPU traces chunk i + 1 while chunk i is shaded; the host executor has one and
// works synchronously.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

#include "render/census.h"
#include "render/pose_table.h"
#include "render/scene_view.h"
#include "render/shade/codes.h"
#include "render/tracer.h"
#include "render/types.h"

namespace balla::render {

class SceneData;
class ThreadPool;
struct TraceSet;

// Buffers of the host API's staging (Executor::stage): render() and gt() as wrappers of the entry points.
enum StageBuffer : uint32_t { kStageRequests, kStageIrradiance, kStageSignature, kStageSubInstance, kStageQueries,
                              kStageGt, kStageClass, kStagePvSlot, kStagePvCodes, kStagePvTable, kStagePvCount,
                              kStageProbes, kStageIrrUnshadowed, kStageShadowStates, kStageBuffers };

// The renderer's chunk loop and GT path (Renderer::render_into, gt_into, and render() / gt() as their
// wrappers) run on either executor through this interface; pointers are in executor memory.
class Executor {
 public:
  virtual ~Executor() = default;

  virtual MemorySpace memory_space() const = 0;
  virtual uint32_t batch_slots() const = 0;
  // Execution stream of a slot (cudaStream_t; nullptr on the host).
  virtual void* stream(uint32_t slot) const = 0;

  // Copies the scene into executor memory; scene_view() is valid afterwards.
  virtual void upload_scene(const SceneData& scene) = 0;
  virtual const SceneView& scene_view() const = 0;
  // Switches to the trace geometries of a held set (nullptr: the scene's own; ENGINE_CPP I3b): keeps the
  // static world's arrays and appends the set's tail after them; scene_view() and the pose rows follow. The
  // set is copied (it may go after the call). No batch may be pending. nullptr needs no memory, and a
  // failure leaves the scene's own geometries in use. Ends the window pose table (its rows have the old
  // stride): begin_window follows before assign_pose_rows and window_poses.
  virtual void use_traces(const TraceSet* traces) = 0;
  // Device memory held for held sets (the merged world's tail and tables; 0 on the host).
  virtual size_t held_device_bytes() const { return 0; }
  // Evaluates the table's new slots (fp64, render/pose.h: compute_slot) for work queued on the slot
  // and returns the table's view in executor memory (render()'s pose cache). The device executor
  // evaluates them on the GPU directly into its mirror; the host executor fills the table's own rows.
  virtual PoseTableView sync_poses(PoseTable& table, uint32_t slot) = 0;

  // Window pose table (ENGINE_CPP §1.4 item 2): rows of one camera's window, only for the slots in use.
  // begin_window drops the rows; assign_pose_rows replaces each request's pose_slot (a slot of the window
  // clock) by its row, first evaluating the rows of slots used for the first time (the device syncs once,
  // to size them). Rows come from the same compute_slot as every other pose row, so they are bitwise
  // equal. assign_pose_rows and window_poses throw std::logic_error without a window (before the first
  // begin_window, and after use_traces).
  virtual void begin_window(uint16_t cam, const WindowClock& clock, void* stream) = 0;
  virtual void assign_pose_rows(SampleRequest* requests, uint32_t n, void* stream) = 0;
  virtual PoseTableView window_poses() const = 0;

  // Slot buffers in executor memory, valid until the slot's next batch.
  virtual HitRecord* hit_buffer(uint32_t slot, size_t count) = 0;
  virtual uint32_t* occlusion_buffer(uint32_t slot, size_t count) = 0;
  // The static words of a chunk (Tracer::sample's static_occluded; scenes with baked proxies, scans.md §4.2).
  virtual uint32_t* static_occlusion_buffer(uint32_t slot, size_t count) = 0;
  virtual GtHit* gt_hit_buffer(uint32_t slot, size_t count) = 0;

  // Queues the shading of one chunk on its slot (occluded may be nullptr without shadow lights): request i
  // writes row out_index[i] of irr / sig (row i when out_index is null), of klass (shade.h sample_class) when it is
  // not null, and, when sub is not null (only without out_index), its sub-sample instances at row i; when pv is not null
  // (at most kPvMaxLights shadow lights), its P&V outputs (render/shade/codes.h; PvOut::slot's row as irr's).
  // static_occluded: the chunk's static words (Tracer::sample) in a scene with baked proxies: they enter the
  // signatures, and with them a baked cluster's radiance takes U2's shadow factor (scans.md §4.2); nullptr: baked
  // clusters render U0 (F = 1).
  // irr_unshadowed (optional; the dense channel shadow_factor, scans.md §3.7; rows as irr's): the irradiance with every
  // proxy factor set to 1. Only chunks with static words have anything to write: the device executor's baked kernels
  // write it, its other kernels ignore it (the renderer passes it for such chunks only, and a scene without baked proxies
  // has shadow_factor 1 everywhere); the host executor writes it whenever it is given.
  virtual void shade_into(uint32_t slot, const TraceBatch& batch, const HitRecord* hits, const uint32_t* occluded,
                          float* irr, uint64_t* sig, uint32_t* sub, const uint32_t* out_index, uint32_t* klass,
                          const PvOut* pv = nullptr, const uint32_t* static_occluded = nullptr,
                          float* irr_unshadowed = nullptr) = 0;
  // P2-D D2: the state of each probe of a chunk traced at one sub-sample per request (batch.subsamples 1;
  // render/shade/probe.h), request i into out[i], queued on its slot.
  virtual void probe_into(uint32_t slot, const TraceBatch& batch, const HitRecord* hits, const uint32_t* occluded,
                          ProbeOut* out) = 0;
  // execution.pv_outputs (P2-D D2's cost measurement; RendererOptions::pv_outputs): P&V outputs for one render_into
  // call of n requests in grow-only executor buffers whose contents are dropped (no slot words), the count zeroed on
  // `stream`; pv_scratch_add (after the call's work, on slot 0's stream) adds its entries and their bytes to running
  // totals, which pv_scratch_totals reads (entries, bytes) once every slot's work is done.
  virtual PvOut pv_scratch(size_t n, size_t code_words, size_t table_words, void* stream) = 0;
  virtual void pv_scratch_add(uint64_t entry_bytes, void* stream) = 0;
  virtual void pv_scratch_totals(uint64_t* totals) = 0;
  // The P2-B B0 census of one chunk (render/census.h), queued on its slot after its shading: request i's counts go to
  // the class of its word klass[out_index[i]] (klass[i] without out_index; the rest without klass) and are added to
  // the executor's running totals (kCensusWords), which census_totals reads once every slot's work is done.
  virtual void census_into(uint32_t slot, const TraceBatch& batch, const HitRecord* hits, const uint32_t* out_index,
                           const uint32_t* klass) = 0;
  virtual void census_totals(uint64_t* totals) = 0;
  // Ground truth on a slot's stream: one exact pose row per run of equal consecutive query times, the
  // centre requests that use them, then (after the tracer) the ground truth written to out, and when normal is not
  // null each hit's camera-frame geometric normal (gt.h; 3 floats per request, P2-D D1).
  virtual const GtRequest* gt_query_rows(uint32_t slot, uint16_t cam, const GtQuery* queries, uint32_t n,
                                         PoseTableView* poses) = 0;
  virtual void evaluate_gt_into(uint32_t slot, const GtRequest* requests, const GtHit* hits, uint32_t count,
                                const PoseTableView& poses, GtSample* out, float* normal) = 0;

  // `waiter` waits for the work queued so far on `signaller` (streams; nullptr is the legacy stream);
  // wait(slot) blocks until the slot's queued work is done. Both are no-ops on the synchronous host.
  virtual void join(void* waiter, void* signaller) = 0;
  virtual void wait(uint32_t slot) = 0;
  // Stage timers (RenderStats): start_timing records the clock origin on a stream, timing_mark a time on a
  // slot's stream (a CUDA event; the host clock on the host) and returns its index, timing_ms reads one once
  // its work is done, reset_timing drops them.
  virtual void start_timing(void* stream) = 0;
  virtual uint32_t timing_mark(uint32_t slot) = 0;
  virtual double timing_ms(uint32_t index) = 0;
  virtual void reset_timing() = 0;

  // The host API's staging: executor memory holding `bytes` for host memory `host` (host itself on the host
  // executor; on the device the grow-only buffer `id`, filled from host on `stream` when upload is set).
  // unstage copies such memory back to host and waits for it (nothing on the host executor).
  virtual void* stage(StageBuffer id, void* host, size_t bytes, bool upload, void* stream) = 0;
  virtual void unstage(void* host, const void* staged, size_t bytes, void* stream) = 0;
};

class HostExecutor final : public Executor {
 public:
  explicit HostExecutor(ThreadPool& pool) : pool_(pool) {}

  MemorySpace memory_space() const override { return MemorySpace::kHost; }
  uint32_t batch_slots() const override { return 1; }
  void* stream(uint32_t) const override { return nullptr; }
  void upload_scene(const SceneData& scene) override;
  const SceneView& scene_view() const override { return view_; }
  void use_traces(const TraceSet* traces) override;
  PoseTableView sync_poses(PoseTable& table, uint32_t slot) override;
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
  void join(void*, void*) override {}
  void wait(uint32_t) override {}
  void start_timing(void*) override {
    marks_.clear();
    epoch_ = Clock::now();
  }
  uint32_t timing_mark(uint32_t) override {
    marks_.push_back(Clock::now());
    return static_cast<uint32_t>(marks_.size() - 1);
  }
  double timing_ms(uint32_t i) override { return std::chrono::duration<double, std::milli>(marks_[i] - epoch_).count(); }
  void reset_timing() override { marks_.clear(); }
  void* stage(StageBuffer, void* host, size_t, bool, void*) override { return host; }
  void unstage(void*, const void*, size_t, void*) override {}

 private:
  using Clock = std::chrono::steady_clock;
  // Rows of `keys` (camera, time) evaluated into rows / light_rows (resized), on the pool.
  void evaluate_rows(const std::vector<std::pair<uint16_t, int64_t>>& keys, size_t first, std::vector<PoseRow>& rows,
                     std::vector<LightRow>& light_rows);

  ThreadPool& pool_;
  Clock::time_point epoch_{};
  std::vector<Clock::time_point> marks_;
  const SceneData* scene_ = nullptr;
  SceneView view_{};
  PoseSceneView pose_view_{};
  // A held set's trace geometries (use_traces): the merged world's arrays (the static world's, copied at the
  // first hold, then the set's tail) and the set's views, bodies and pose geometries.
  std::vector<float> positions_, normals_, uvs_;
  std::vector<uint32_t> indices_, materials_, instances_, patches_;
  std::vector<GeometryView> views_;
  std::vector<BodyView> bodies_;
  std::vector<PoseGeometryDef> pose_geometries_;
  std::vector<AoBodyView> ao_bodies_;  // accuracy.mover_ao proxy: the set's posing of the movers' spheres
  std::vector<AoSphere> ao_spheres_;
  std::vector<HitRecord> hits_;
  std::vector<uint32_t> occluded_;
  std::vector<uint32_t> static_occluded_;
  std::vector<GtHit> gt_hits_;
  std::array<std::atomic<uint64_t>, kCensusWords> census_{};  // census_into's running totals
  std::vector<uint32_t> pv_codes_, pv_table_;                    // pv_scratch's outputs, its count and totals
  uint32_t pv_count_ = 0;
  uint64_t pv_totals_[2] = {0, 0};
  // The window pose table: per slot its row (kNoRow before first use), the slots' keys and rows.
  static constexpr uint32_t kNoRow = 0xFFFFFFFFu;
  WindowClock clock_{};
  uint16_t window_cam_ = 0;
  uint32_t window_slots_ = 0;
  std::vector<uint32_t> window_index_;
  std::vector<std::pair<uint16_t, int64_t>> window_keys_;
  std::vector<PoseRow> window_rows_;
  std::vector<LightRow> window_light_rows_;
  // gt_query_rows: the rows of one chunk and its centre requests.
  std::vector<std::pair<uint16_t, int64_t>> gt_keys_;
  std::vector<PoseRow> gt_rows_;
  std::vector<LightRow> gt_light_rows_;
  std::vector<GtRequest> gt_requests_;
};

}  // namespace balla::render
