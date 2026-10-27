#include "render/pose_table.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <string>

namespace balla::render {
namespace {

uint64_t hash_key(uint16_t cam, int64_t t_ns) {
  uint64_t z = static_cast<uint64_t>(t_ns) ^ (static_cast<uint64_t>(cam) << 48);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

}  // namespace

PoseTable::PoseTable(int32_t num_geometries, int32_t num_lights, size_t capacity)
    : num_geometries_(num_geometries), num_lights_(num_lights), capacity_(capacity) {
  size_t cells = 16;
  while (cells < 2 * capacity) cells *= 2;
  cells_.assign(cells, kEmpty);
  mask_ = cells - 1;
}

void PoseTable::reset(int64_t base_ns) {
  base_ns_ = base_ns;
  ++generation_;
  for (const Key& k : keys_) {  // clear only the used cells: cheaper than the whole table
    uint64_t h = hash_key(k.cam, k.t_ns) & mask_;
    while (cells_[h] != kEmpty) {
      cells_[h] = kEmpty;
      h = (h + 1) & mask_;
    }
  }
  keys_.clear();
  rows_.clear();
  light_rows_.clear();
  evaluated_ = 0;
  last_ = {0xFFFF, 0};
}

bool PoseTable::slot(uint16_t cam, int64_t t_ns, uint32_t* slot) {
  const Key key{cam, t_ns};
  if (key == last_) {
    *slot = last_slot_;
    return true;
  }
  uint64_t h = hash_key(cam, t_ns) & mask_;
  while (cells_[h] != kEmpty) {
    if (keys_[cells_[h]] == key) {
      last_ = key;
      last_slot_ = *slot = cells_[h];
      return true;
    }
    h = (h + 1) & mask_;
  }
  if (keys_.size() >= capacity_) return false;
  cells_[h] = static_cast<uint32_t>(keys_.size());
  keys_.push_back(key);
  last_ = key;
  last_slot_ = *slot = cells_[h];
  return true;
}

void PoseTable::evaluate(const PoseSceneView& scene, ThreadPool& pool) {
  const size_t first = evaluated_, n = keys_.size() - first;
  if (n == 0) return;
  rows_.resize(keys_.size() * static_cast<size_t>(num_geometries_));
  light_rows_.resize(keys_.size() * static_cast<size_t>(num_lights_));
  std::atomic<size_t> bad{SIZE_MAX};
  pool.parallel_for(n, 256, [&](size_t begin, size_t end) {
    for (size_t i = first + begin; i < first + end; ++i) {
      if (!compute_slot(scene, keys_[i].cam, keys_[i].t_ns, rows_.data() + i * static_cast<size_t>(num_geometries_),
                        light_rows_.data() + i * static_cast<size_t>(num_lights_))) {
        size_t expected = bad.load();
        while (i < expected && !bad.compare_exchange_weak(expected, i)) {
        }
      }
    }
  });
  if (bad.load() != SIZE_MAX) {
    const Key k = keys_[bad.load()];
    // Forget the unevaluated slots so the table stays consistent for the next batch.
    for (size_t i = keys_.size(); i-- > first;) {
      uint64_t h = hash_key(keys_[i].cam, keys_[i].t_ns) & mask_;
      while (cells_[h] != i) h = (h + 1) & mask_;
      // Backward-shift deletion keeps every probe chain intact.
      uint64_t hole = h;
      for (uint64_t j = (h + 1) & mask_; cells_[j] != kEmpty; j = (j + 1) & mask_) {
        const uint64_t home = hash_key(keys_[cells_[j]].cam, keys_[cells_[j]].t_ns) & mask_;
        if (((j - home) & mask_) >= ((j - hole) & mask_)) {
          cells_[hole] = cells_[j];
          hole = j;
        }
      }
      cells_[hole] = kEmpty;
    }
    keys_.resize(first);
    rows_.resize(first * static_cast<size_t>(num_geometries_));
    light_rows_.resize(first * static_cast<size_t>(num_lights_));
    last_ = {0xFFFF, 0};
    throw std::invalid_argument("render: time " + std::to_string(k.t_ns) + " ns (camera " + std::to_string(k.cam) +
                                ") lies outside the span of a trajectory the scene needs");
  }
  evaluated_ = keys_.size();
}

}  // namespace balla::render
