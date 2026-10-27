// Rigid transforms and mesh geometry as the EVIMO2 adapter computes them (ballasim.scene.transforms,
// ballasim.scene.assets.geometry; docs/dev/CPP_PORT.md item 5, S4), in NumPy's order of operations: matmul through
// npla's BLAS dispatch, reductions as NumPy sums them, float32 storage where the Python casts.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "scene/rotation.h"

namespace balla::scene::geo {

using rot::M;
using rot::Q;
using rot::V;

// A 4 x 4 transform, row-major (T_A_B maps B coordinates to A).
using M4 = std::array<double, 16>;

M4 identity4();
M4 make_transform(const M& r, V t);
// transforms.pose_to_transform: Rotation.from_quat(q).as_matrix() and p.
M4 pose_to_transform(Q q, V p);
// transforms.transform_to_pose: (canonical_quat(Rotation.from_matrix(R).as_quat()), t).
std::pair<Q, V> transform_to_pose(const M4& t);
// transforms.invert_transform: (R^T, -einsum("...ij,...j->...i", R^T, t)).
M4 invert_transform(const M4& t);
// np.matmul of two 4 x 4 matrices (cblas_dgemm).
M4 matmul(const M4& a, const M4& b);
M rotation_of(const M4& t);
V translation_of(const M4& t);

// bundle.Mesh: float32 positions (V, 3), uint32 indices (T, 3), optional float32 normals (V, 3), float32 uvs (V, 2)
// and uint16 material ids (T).
struct Mesh {
  std::vector<float> positions;
  std::vector<uint32_t> indices;
  std::vector<float> normals, uvs;
  std::vector<uint16_t> material_ids;
  bool has_normals = false, has_uvs = false, has_material_ids = false;
  size_t num_vertices() const { return positions.size() / 3; }
  size_t num_triangles() const { return indices.size() / 3; }
};

// geometry.transform_mesh (normals by the inverse transpose, winding flipped for a mirror).
Mesh transform_mesh(const Mesh& mesh, const M4& t);
// geometry.merge_meshes: parts with their material counts; ids offset per part, None for a single material.
Mesh merge_meshes(const std::vector<std::pair<const Mesh*, int>>& parts);
// geometry.triangle_areas (float64, from the float32 positions).
std::vector<double> triangle_areas(const Mesh& mesh);
// geometry.area_centroid.
V area_centroid(const Mesh& mesh);
// geometry.bounding_sphere of the float32 positions: AABB centre and the largest distance to it.
std::pair<V, double> bounding_sphere(const Mesh& mesh);
// geometry.icosphere(level, radius).
Mesh icosphere(int level, double radius);
// adapter._dominant_normal: the unit normal of the mesh's largest planar area (face normals binned at ~2 degrees).
V dominant_normal(const Mesh& mesh);

// transforms.kabsch(a, b): R a_i + t ~ b_i, and the per-point residual norms.
struct Kabsch {
  M r;
  V t;
  std::vector<double> residuals;
};
Kabsch kabsch(std::span<const V> a, std::span<const V> b);

// --- mesh validation (geometry.validate_mesh and its helpers), on float64 positions (float32 ones convert exactly) ---

// geometry.vertex_normals: angle-weighted (Thuermer & Wuethrich) vertex normals, float32, unit or 0.
std::vector<float> vertex_normals(std::span<const double> positions, std::span<const int64_t> indices);
// geometry.kept_faces: the triangles validation keeps (not degenerate, the first of each duplicate vertex set), and
// the degenerate and duplicate counts.
std::vector<uint8_t> kept_faces(std::span<const double> positions, std::span<const int64_t> indices, double rel_eps,
                                int64_t* degenerate, int64_t* duplicate);
// geometry.welded_normals: vertex_normals over the kept triangles.
std::vector<float> welded_normals(std::span<const double> positions, std::span<const int64_t> indices,
                                  double rel_eps = 1e-7);
// geometry.validate_mesh (compute_normals=True): refuses NaN/Inf (AssetError), drops degenerate and duplicate
// triangles and unreferenced vertices, computes missing normals or renormalizes given ones (float32, as NumPy does);
// the report's entries in the Python's order (name, value) for the caller's dict.
struct ValidationReport {
  int64_t input_vertices = 0, input_triangles = 0, degenerate_removed = 0, duplicate_removed = 0,
          unreferenced_removed = 0, vertices = 0, triangles = 0;
  std::string normals;
};
Mesh validate_mesh(const Mesh& mesh, ValidationReport& report, double rel_eps = 1e-7);

}  // namespace balla::scene::geo
