#include <stdexcept>
#include <string>

#include "render/executor.h"
#include "render/gt.h"
#include "render/scene_data.h"
#include "render/shade/probe.h"
#include "render/shade/shade.h"
#include "render/thread_pool.h"

namespace balla::render {

void HostExecutor::upload_scene(const SceneData& scene) {
  scene_ = &scene;
  use_traces(nullptr);
}

void HostExecutor::use_traces(const TraceSet* t) {
  view_ = scene_->host_view();
  pose_view_ = scene_->pose_view();
  window_slots_ = 0;  // the window's rows have the old stride
  if (t == nullptr) return;
  const GeometryData& world = scene_->geometries()[0];
  const GeometryData& tail = t->tail;
  // keep values of the world (copied at first use; zeros where it has none), then the tail's.
  const auto merge = [](auto& dst, const auto& world_values, size_t keep, const auto& tail_values) {
    if (dst.size() < keep) dst.assign(world_values.begin(), world_values.end());
    dst.resize(keep);
    dst.insert(dst.end(), tail_values.begin(), tail_values.end());
    return dst.data();
  };
  const size_t nv = world.num_vertices(), nt = world.num_triangles();
  const GeometryView merged{merge(positions_, world.positions, 3 * nv, tail.positions),
                            merge(indices_, world.indices, 3 * nt, tail.indices),
                            tail.has_normals ? merge(normals_, world.normals, 3 * nv, tail.normals) : nullptr,
                            tail.has_uvs ? merge(uvs_, world.uvs, 2 * nv, tail.uvs) : nullptr,
                            merge(materials_, world.materials, nt, tail.materials),
                            merge(instances_, world.instances, nt, tail.instances),
                            0,
                            static_cast<uint32_t>(nv) + tail.num_vertices(),
                            static_cast<uint32_t>(nt) + tail.num_triangles(),
                            tail.single_material,
                            world.radius,
                            tail.has_patches ? merge(patches_, world.patches, nt, tail.patches) : nullptr, 0};
  views_ = t->select(view_.geometries, merged);
  bodies_ = t->bodies;
  pose_geometries_ = t->pose_geometries;
  view_.geometries = views_.data();
  view_.num_geometries = pose_view_.num_geometries = static_cast<int32_t>(views_.size());
  view_.bodies = bodies_.data();
  pose_view_.geometries = pose_geometries_.data();
  ao_bodies_ = t->ao_bodies;
  ao_spheres_ = t->ao_spheres;
  view_.ao_bodies = ao_bodies_.empty() ? nullptr : ao_bodies_.data();
  view_.ao_spheres = ao_spheres_.empty() ? nullptr : ao_spheres_.data();
}

PoseTableView HostExecutor::sync_poses(PoseTable& table, uint32_t) {
  table.evaluate(pose_view_, pool_);
  return table.view();
}

void HostExecutor::evaluate_rows(const std::vector<std::pair<uint16_t, int64_t>>& keys, size_t first,
                                 std::vector<PoseRow>& rows, std::vector<LightRow>& light_rows) {
  const auto geoms = static_cast<size_t>(pose_view_.num_geometries), lights = static_cast<size_t>(pose_view_.num_lights);
  rows.resize(keys.size() * geoms);
  light_rows.resize(keys.size() * lights);
  // Times were checked against the trajectory spans before: the window's slots by begin_window, gt_into's
  // queries by the caller (check_gt). As in the device kernels, a failure is not reported.
  pool_.parallel_for(keys.size() - first, 256, [&](size_t begin, size_t end) {
    for (size_t k = first + begin; k < first + end; ++k) {
      compute_slot(pose_view_, keys[k].first, keys[k].second, rows.data() + k * geoms, light_rows.data() + k * lights);
    }
  });
}

void HostExecutor::begin_window(uint16_t cam, const WindowClock& clock, void*) {
  clock_ = clock;
  window_cam_ = cam;
  window_slots_ = num_slots(clock);
  window_index_.assign(window_slots_, kNoRow);
  window_keys_.clear();
}

void HostExecutor::assign_pose_rows(SampleRequest* requests, uint32_t n, void*) {
  if (n == 0) return;
  if (window_slots_ == 0) throw std::logic_error("render: assign_pose_rows needs begin_window first (also after hold)");
  for (uint32_t i = 0; i < n; ++i) {  // before any change: a refused call leaves the table and requests as they were
    const uint32_t slot = requests[i].pose_slot;
    if (slot >= window_slots_) throw std::invalid_argument("render: pose slot " + std::to_string(slot) + " out of range");
  }
  const size_t first = window_keys_.size();
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t slot = requests[i].pose_slot;
    if (window_index_[slot] == kNoRow) {
      window_index_[slot] = static_cast<uint32_t>(window_keys_.size());
      window_keys_.emplace_back(window_cam_, slot_time(clock_, slot));
    }
    requests[i].pose_slot = window_index_[slot];
  }
  if (window_keys_.size() > first) evaluate_rows(window_keys_, first, window_rows_, window_light_rows_);
}

PoseTableView HostExecutor::window_poses() const {
  if (window_slots_ == 0) throw std::logic_error("render: render_into needs begin_window first (also after hold)");
  return {window_rows_.data(), window_light_rows_.data(), pose_view_.num_geometries, pose_view_.num_lights};
}

HitRecord* HostExecutor::hit_buffer(uint32_t, size_t count) {
  if (hits_.size() < count) hits_.resize(count);
  return hits_.data();
}

uint32_t* HostExecutor::occlusion_buffer(uint32_t, size_t count) {
  if (occluded_.size() < count) occluded_.resize(count);
  return occluded_.data();
}

uint32_t* HostExecutor::static_occlusion_buffer(uint32_t, size_t count) {
  if (static_occluded_.size() < count) static_occluded_.resize(count);
  return static_occluded_.data();
}

GtHit* HostExecutor::gt_hit_buffer(uint32_t, size_t count) {
  if (gt_hits_.size() < count) gt_hits_.resize(count);
  return gt_hits_.data();
}

void HostExecutor::shade_into(uint32_t, const TraceBatch& b, const HitRecord* hits, const uint32_t* occluded,
                              float* irr, uint64_t* sig, uint32_t* sub, const uint32_t* out_index, uint32_t* klass,
                              const PvOut* pv, const uint32_t* static_occluded, float* irr_unshadowed) {
  const size_t groups = static_cast<size_t>(view_.num_groups);
  pool_.parallel_for(b.num_requests, 64, [&](size_t begin, size_t end) {
    for (size_t r = begin; r < end; ++r) {
      const size_t base = r * b.subsamples, row = out_index != nullptr ? out_index[r] : r;
      const PvSample pv_sample{pv, row, b.subsamples};
      shade_pixel_sample(view_, b.poses, b.requests[r], hits + base, occluded != nullptr ? occluded + base : nullptr,
                         irr + row * groups, sig + row, sub != nullptr ? sub + base : nullptr,
                         klass != nullptr ? klass + row : nullptr, pv != nullptr ? &pv_sample : nullptr,
                         static_occluded != nullptr ? static_occluded + base : nullptr,
                         irr_unshadowed != nullptr ? irr_unshadowed + row * groups : nullptr);
    }
  });
}

void HostExecutor::probe_into(uint32_t, const TraceBatch& b, const HitRecord* hits, const uint32_t* occluded,
                              ProbeOut* out) {
  pool_.parallel_for(b.num_requests, 1024, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      out[i] = probe_state(view_, b.poses, b.requests[i], hits[i], occluded != nullptr ? occluded[i] : 0u);
    }
  });
}

PvOut HostExecutor::pv_scratch(size_t n, size_t code_words, size_t table_words, void*) {
  if (pv_codes_.size() < n * code_words) pv_codes_.resize(n * code_words);
  if (pv_table_.size() < n * table_words) pv_table_.resize(n * table_words);
  pv_count_ = 0;
  PvOut o;
  o.codes = pv_codes_.data();
  o.table = pv_table_.data();
  o.count = &pv_count_;
  o.capacity = static_cast<uint32_t>(n);
  return o;
}

void HostExecutor::pv_scratch_add(uint64_t entry_bytes, void*) {
  pv_totals_[0] += pv_count_;
  pv_totals_[1] += pv_count_ * entry_bytes;
}

void HostExecutor::pv_scratch_totals(uint64_t* totals) {
  totals[0] = pv_totals_[0];
  totals[1] = pv_totals_[1];
}

void HostExecutor::census_into(uint32_t, const TraceBatch& b, const HitRecord* hits, const uint32_t* out_index,
                               const uint32_t* klass) {
  pool_.parallel_for(b.num_requests, 64, [&](size_t begin, size_t end) {
    uint64_t v[kCensusWords] = {};
    for (size_t r = begin; r < end; ++r) {
      const size_t row = out_index != nullptr ? out_index[r] : r;
      census_add(v, b.requests[r], census_sample(view_, b.poses, b.requests[r], hits + r * b.subsamples),
                 klass != nullptr ? klass + row : nullptr);
    }
    for (int i = 0; i < kCensusWords; ++i) census_[static_cast<size_t>(i)].fetch_add(v[i], std::memory_order_relaxed);
  });
}

void HostExecutor::census_totals(uint64_t* totals) {
  for (int i = 0; i < kCensusWords; ++i) totals[i] = census_[static_cast<size_t>(i)].load(std::memory_order_relaxed);
}

const GtRequest* HostExecutor::gt_query_rows(uint32_t, uint16_t cam, const GtQuery* q, uint32_t n,
                                             PoseTableView* poses) {
  gt_keys_.clear();
  gt_requests_.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    if (i == 0 || q[i].t_ns != q[i - 1].t_ns) gt_keys_.emplace_back(cam, q[i].t_ns);
    gt_requests_[i] = {cam, 0, q[i].x, q[i].y, static_cast<uint32_t>(gt_keys_.size() - 1)};
  }
  evaluate_rows(gt_keys_, 0, gt_rows_, gt_light_rows_);
  *poses = {gt_rows_.data(), gt_light_rows_.data(), pose_view_.num_geometries, pose_view_.num_lights};
  return gt_requests_.data();
}

void HostExecutor::evaluate_gt_into(uint32_t, const GtRequest* requests, const GtHit* hits, uint32_t count,
                                    const PoseTableView& poses, GtSample* out, float* normal) {
  pool_.parallel_for(count, 1024, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      out[i] = gt_eval(view_, poses, requests[i], hits[i], normal != nullptr ? normal + 3 * i : nullptr);
    }
  });
}

}  // namespace balla::render
