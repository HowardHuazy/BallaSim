// Lazily filled pose table of one window (contract C2 set_window; DESIGN §6.9).
//
// A pose slot is one (camera, time) pair. Its rows (one per trace geometry, one per light) are
// evaluated in fp64 the first time a request references the slot and kept until the table is
// reset, so the integrator's repeated rounds over the same lattice nodes reuse them. Slot ids are
// an implementation detail: row contents depend only on (camera, time), so results never depend on
// which requests share a batch.
#pragma once

#include <cstdint>
#include <vector>

#include "render/pose.h"
#include "render/scene_view.h"
#include "render/thread_pool.h"

namespace balla::render {

class PoseTable {
 public:
  struct Key {
    uint16_t cam;
    int64_t t_ns;
    bool operator==(const Key& o) const { return cam == o.cam && t_ns == o.t_ns; }
  };

  // capacity: the most slots the table holds between resets.
  PoseTable(int32_t num_geometries, int32_t num_lights, size_t capacity);

  // Drops every slot and sets the window base of the int32 time offsets.
  void reset(int64_t base_ns);
  // Drops every slot and sets the number of trace geometries per slot (a held set, ENGINE_CPP I3b).
  void reshape(int32_t num_geometries) {
    num_geometries_ = num_geometries;
    reset(base_ns_);
  }
  int64_t base() const { return base_ns_; }
  // Incremented by every reset, so device mirrors know when to start over.
  uint64_t generation() const { return generation_; }

  // Slot of (cam, t_ns), allocated (unevaluated) on first use; false if that would exceed the
  // capacity.
  bool slot(uint16_t cam, int64_t t_ns, uint32_t* slot);

  // Evaluates every slot allocated since the last call on the host (fp64, thread pool). Throws
  // std::invalid_argument naming the first time outside the span of a trajectory it needs. Device
  // executors evaluate the same rows on the GPU from keys() instead.
  void evaluate(const PoseSceneView& scene, ThreadPool& pool);
  const std::vector<Key>& keys() const { return keys_; }

  size_t num_slots() const { return keys_.size(); }
  PoseTableView view() const {
    return {rows_.data(), light_rows_.data(), num_geometries_, num_lights_};
  }
  const std::vector<PoseRow>& rows() const { return rows_; }
  const std::vector<LightRow>& light_rows() const { return light_rows_; }
  int32_t num_geometries() const { return num_geometries_; }
  int32_t num_lights() const { return num_lights_; }

 private:
  static constexpr uint32_t kEmpty = 0xFFFFFFFFu;

  int32_t num_geometries_, num_lights_;
  size_t capacity_;
  int64_t base_ns_ = 0;
  uint64_t generation_ = 0;
  // Open addressing with linear probing over a power-of-two table at most half full; cells hold slot
  // ids (kEmpty when free). A batch adds up to 2^16 keys, so this beats a node-based map severalfold.
  std::vector<uint32_t> cells_;
  uint64_t mask_ = 0;
  std::vector<Key> keys_;
  size_t evaluated_ = 0;
  Key last_{0xFFFF, 0};  // fast path: consecutive requests usually share their time
  uint32_t last_slot_ = 0;
  std::vector<PoseRow> rows_;
  std::vector<LightRow> light_rows_;
};

}  // namespace balla::render
