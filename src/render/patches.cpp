#include "render/patches.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>

namespace balla::render {
namespace {

struct D3 {
  double x, y, z;
};

D3 sub(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
D3 cross(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
double dot(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// The angle between two vectors (0 when either is zero), atan2 of the cross product's length and the dot product.
double angle(D3 a, D3 b) { return std::atan2(std::sqrt(dot(cross(a, b), cross(a, b))), dot(a, b)); }

D3 vec(std::span<const float> v, uint32_t i) {
  return {static_cast<double>(v[3 * i]), static_cast<double>(v[3 * i + 1]), static_cast<double>(v[3 * i + 2])};
}

// Union-find over triangles: union by smaller root index, path halving (the components, not the roots, matter).
uint32_t find(std::vector<uint32_t>& parent, uint32_t i) {
  while (parent[i] != i) {
    parent[i] = parent[parent[i]];
    i = parent[i];
  }
  return i;
}

}  // namespace

Patches build_patches(const PatchMesh& mesh, std::span<const uint32_t> materials, std::span<const uint8_t> specular) {
  const size_t nv = mesh.positions.size() / 3, nt = mesh.indices.size() / 3;
  if (materials.size() != nt) throw std::invalid_argument("build_patches: one material per triangle");
  if (!mesh.normals.empty() && mesh.normals.size() != 3 * nv) {
    throw std::invalid_argument("build_patches: one normal per vertex");
  }
  for (uint32_t m : materials) {
    if (!specular.empty() && m >= specular.size()) throw std::invalid_argument("build_patches: material out of range");
  }
  // Welded positions: vertices sorted by coordinates (by value: -0 and +0 weld), ties by index.
  std::vector<uint32_t> order(nv), weld(nv);
  std::iota(order.begin(), order.end(), 0u);
  const auto key = [&](uint32_t v) {
    return std::array<float, 3>{mesh.positions[3 * v], mesh.positions[3 * v + 1], mesh.positions[3 * v + 2]};
  };
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    const auto ka = key(a), kb = key(b);
    return ka < kb || (!(kb < ka) && a < b);
  });
  for (size_t i = 0, group = 0; i < nv; ++i) {
    if (i > 0 && key(order[i - 1]) < key(order[i])) group = i;
    weld[order[i]] = static_cast<uint32_t>(group);
  }
  // Edge records keyed by their welded ends (low, high), with the vertices at those ends and the triangle.
  struct Edge {
    uint64_t key;
    uint32_t tri, v_lo, v_hi;
  };
  std::vector<Edge> edges;
  edges.reserve(3 * nt);
  std::vector<D3> face(nt);
  for (uint32_t t = 0; t < nt; ++t) {
    const uint32_t* c = &mesh.indices[3 * t];
    for (int e = 0; e < 3; ++e) {
      if (c[e] >= nv) throw std::invalid_argument("build_patches: index out of range");
    }
    face[t] = cross(sub(vec(mesh.positions, c[1]), vec(mesh.positions, c[0])),
                    sub(vec(mesh.positions, c[2]), vec(mesh.positions, c[0])));
    for (int e = 0; e < 3; ++e) {
      uint32_t a = c[e], b = c[(e + 1) % 3];
      if (weld[a] == weld[b]) continue;  // a collapsed edge joins nothing
      if (weld[a] > weld[b]) std::swap(a, b);
      edges.push_back({(static_cast<uint64_t>(weld[a]) << 32) | weld[b], t, a, b});
    }
  }
  std::sort(edges.begin(), edges.end(),
            [](const Edge& x, const Edge& y) { return x.key < y.key || (x.key == y.key && x.tri < y.tri); });

  const bool has_normals = !mesh.normals.empty();
  const auto hard = [&](uint32_t u, uint32_t v) {
    return u != v && angle(vec(mesh.normals, u), vec(mesh.normals, v)) > kPatchHardEdgeRad;
  };
  const auto boundary = [&](const Edge& x, const Edge& y) {
    const uint32_t mx = materials[x.tri], my = materials[y.tri];
    if (mx != my) return true;                                                  // (a)
    const bool crease = angle(face[x.tri], face[y.tri]) > kPatchCreaseRad;
    if (!has_normals) return crease;                                            // (c)
    const bool glossy = !specular.empty() && (specular[mx] != 0 || specular[my] != 0);
    return (hard(x.v_lo, y.v_lo) || hard(x.v_hi, y.v_hi)) && (crease || glossy);  // (b)
  };
  std::vector<uint32_t> parent(nt);
  std::iota(parent.begin(), parent.end(), 0u);
  for (size_t i = 0; i < edges.size();) {
    size_t j = i;
    while (j < edges.size() && edges[j].key == edges[i].key) ++j;
    for (size_t a = i; a < j; ++a) {  // every pair of triangles on the edge (non-manifold edges included)
      for (size_t b = a + 1; b < j; ++b) {
        if (edges[a].tri == edges[b].tri || boundary(edges[a], edges[b])) continue;
        const uint32_t ra = find(parent, edges[a].tri), rb = find(parent, edges[b].tri);
        if (ra != rb) parent[std::max(ra, rb)] = std::min(ra, rb);
      }
    }
    i = j;
  }
  Patches out;
  out.of_triangle.resize(nt);
  std::vector<uint32_t> label(nt, UINT32_MAX);
  for (uint32_t t = 0; t < nt; ++t) {
    const uint32_t r = find(parent, t);
    if (label[r] == UINT32_MAX) label[r] = out.count++;
    out.of_triangle[t] = label[r];
  }
  return out;
}

}  // namespace balla::render
