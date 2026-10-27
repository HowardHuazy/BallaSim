#include "scene/assets/mesh_stats.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace balla::scene::assets {

MeshStats mesh_stats(const geo::Mesh& mesh) {
  MeshStats s;
  const size_t nv = mesh.num_vertices(), nt = mesh.num_triangles();
  s.triangles = static_cast<int64_t>(nt);
  for (const uint32_t i : mesh.indices) {
    if (i >= nv) throw std::invalid_argument("mesh_stats: a triangle index is out of range");
  }
  if (nv > 0) {
    for (size_t a = 0; a < 3; ++a) {
      double lo = static_cast<double>(mesh.positions[a]), hi = lo;
      for (size_t v = 1; v < nv; ++v) {
        const auto x = static_cast<double>(mesh.positions[3 * v + a]);
        lo = std::min(lo, x);
        hi = std::max(hi, x);
      }
      s.bbox_m[a] = hi - lo;
    }
  }

  // Welding: vertices sorted by their position bits; equal bits share the first vertex's id.
  const auto bits = [&](size_t v) {
    return std::make_tuple(std::bit_cast<uint32_t>(mesh.positions[3 * v]),
                           std::bit_cast<uint32_t>(mesh.positions[3 * v + 1]),
                           std::bit_cast<uint32_t>(mesh.positions[3 * v + 2]));
  };
  std::vector<uint32_t> order(nv);
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    const auto ka = bits(a), kb = bits(b);
    return ka != kb ? ka < kb : a < b;
  });
  std::vector<uint32_t> weld(nv);
  for (size_t i = 0; i < nv; ++i) {
    const bool same = i > 0 && bits(order[i]) == bits(order[i - 1]);
    weld[order[i]] = same ? weld[order[i - 1]] : order[i];
    s.welded_vertices += same ? 0 : 1;
  }

  // Undirected edges of the triangles with three distinct welded corners, as (low id << 32 | high id).
  std::vector<uint64_t> edges;
  edges.reserve(3 * nt);
  for (size_t t = 0; t < nt; ++t) {
    const uint32_t c[3] = {weld[mesh.indices[3 * t]], weld[mesh.indices[3 * t + 1]], weld[mesh.indices[3 * t + 2]]};
    if (c[0] == c[1] || c[1] == c[2] || c[2] == c[0]) {
      ++s.degenerate_triangles;
      continue;
    }
    for (int k = 0; k < 3; ++k) {
      const uint32_t a = c[k], b = c[(k + 1) % 3];
      edges.push_back(static_cast<uint64_t>(std::min(a, b)) << 32 | std::max(a, b));
    }
  }
  std::sort(edges.begin(), edges.end());
  for (size_t i = 0; i < edges.size();) {
    size_t j = i + 1;
    while (j < edges.size() && edges[j] == edges[i]) ++j;
    if (j - i == 1) {
      const auto a = static_cast<size_t>(edges[i] >> 32), b = static_cast<size_t>(edges[i] & 0xFFFFFFFFu);
      const auto d = [&](size_t k) {
        return static_cast<double>(mesh.positions[3 * a + k]) - static_cast<double>(mesh.positions[3 * b + k]);
      };
      const double dx = d(0), dy = d(1), dz = d(2);
      s.open_boundary_m += std::sqrt((dx * dx + dy * dy) + dz * dz);
    } else if (j - i > 2) {
      ++s.nonmanifold_edges;
    }
    i = j;
  }
  return s;
}

}  // namespace balla::scene::assets
