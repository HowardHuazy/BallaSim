// Single-mesh ray queries shared by the Embree and OptiX smoke paths: one static triangle mesh,
// one closest-hit ray per entry, no poses. They prove the toolchains end to end and give the
// ray-by-ray conformance tests (§12.1 L2) a common result format.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace balla::backend {

// Primitive index of a miss in MeshHits. A prim-only sentinel: hit records mark misses by
// instance (kInstanceNone in api.h), never by prim.
inline constexpr uint32_t kNoPrim = 0xFFFFFFFFu;

struct TriangleMeshView {
  const float* vertices;   // 3 floats per vertex
  size_t num_vertices;
  const uint32_t* indices;  // 3 vertex indices per triangle
  size_t num_triangles;
};

struct RayBatchView {
  const float* origins;     // 3 floats per ray
  const float* directions;  // 3 floats per ray, need not be unit length
  size_t count;
};

// Closest hits along origin + t * direction, t in [0, inf). Barycentrics (u, v) weight vertices
// 1 and 2 of the triangle (both Embree and OptiX use this convention). Misses: t = +inf,
// prim = kNoPrim, u = v = 0. An empty mesh gives all misses; an index >= num_vertices throws
// std::invalid_argument before any device work (check_indices).
struct MeshHits {
  std::vector<float> t;
  std::vector<uint32_t> prim;
  std::vector<float> u, v;
};

inline MeshHits all_misses(size_t n) {
  MeshHits hits;
  hits.t.assign(n, std::numeric_limits<float>::infinity());
  hits.prim.assign(n, kNoPrim);
  hits.u.assign(n, 0.0f);
  hits.v.assign(n, 0.0f);
  return hits;
}

inline void check_indices(const TriangleMeshView& mesh) {
  for (size_t k = 0; k < 3 * mesh.num_triangles; ++k) {
    if (mesh.indices[k] >= mesh.num_vertices) throw std::invalid_argument("vertex index out of range");
  }
}

}  // namespace balla::backend
