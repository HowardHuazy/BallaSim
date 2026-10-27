// Mesh statistics of `ballasim scan report` (docs/design/phase3/scans.md §3.5, §3.6, §4.8; S2c): the size of an
// imported scan and its holes, reported as open boundary.
#pragma once

#include <array>
#include <cstdint>

#include "scene/geometry.h"

namespace balla::scene::assets {

struct MeshStats {
  int64_t triangles = 0;
  std::array<double, 3> bbox_m{};  // the extent of the positions per axis (max - min); 0 without vertices
  // Vertices with bit-equal float32 positions are welded first, so a mesh split at UV or normal seams counts as one
  // surface. An edge is a pair of welded vertices; a triangle whose welded corners repeat has zero area and adds no
  // edge (counted in degenerate_triangles).
  double open_boundary_m = 0.0;     // total length of the edges with one face (fp64, in ascending edge order)
  int64_t nonmanifold_edges = 0;    // edges with more than two faces
  int64_t welded_vertices = 0;      // distinct positions
  int64_t degenerate_triangles = 0;
};

// Throws std::invalid_argument when an index is out of range.
MeshStats mesh_stats(const geo::Mesh& mesh);

}  // namespace balla::scene::assets
