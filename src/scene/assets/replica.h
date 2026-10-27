// Replica scene import (docs/design/phase3/scans.md §4.5, S5b-1): a Replica scene directory (`mesh.ply`, the Ptex chunks
// `textures/<c>-color-ptex.hdr`, `habitat/sorted_faces.bin`) becomes one baked scan mesh and one binary16 tile atlas.
//
// Mapping (verified by D1, scans.md §4.5.1): chunk c holds the faces sorted_faces.bin lists for it; its atlas is raw RGB
// binary16 with side dim = sqrt(bytes / 6), tiles of tileSize (16) in rows of W = dim / 16; the i-th face of the chunk
// owns the tile at ((i mod W) * 16, (i div W) * 16), and quad corners in[0..3] sit at uv (0,0), (1,0), (1,1), (0,1) of
// it, x a column and y a file row, no flip.
//
// Each source tile is box-meaned (in fp64, to binary16) to T x T, T in {4, 8, 16}, framed by a gutter of G = T / 2
// texels on every side (P = T + 2 G, a power of two), and packed in sorted_faces order into a square atlas of side
// P * ceil(sqrt(quads)). The material is baked, `texture_tile` = P, so the renderer reads each face's tile alone (S5a).
// Shared manifold edges copy neighbour interiors with the edge orientation preserved; boundary/non-manifold
// edges clamp. Corners extend the top/bottom gutter row. S5b-2 adds the deterministic self-check and threading;
// semantic parts follow in S6.
//
// Replica's licence binds every derived output (scans.md §3.7): the caller keeps what this returns restricted.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "scene/assets/ply.h"
#include "scene/geometry.h"
#include "scene/tree.h"

namespace balla::scene::assets {

// textures/parameters.json (read by the Python): the tile side and the SplitMesh cell size.
struct ReplicaInputs {
  int32_t tile_size = 16;   // must be 16
  double split_size = 3.0;  // provenance only
};

struct ReplicaOptions {
  int32_t tile_texels = 0;  // T: 0 auto (8 up to 2.5 M quads, else 4), or 4, 8, 16
  int32_t workers = 1;
  double self_check_ratio = 2.0;
};

struct ReplicaPart {
  int32_t object_id = -1;  // -1: not split
  geo::Mesh mesh;          // 4 vertices per quad (positions and normals copied), two triangles per quad (0,1,2), (0,2,3)
  Material material;       // baked, texture_linear = the atlas of this part's padded tiles, texture_tile = P, K = 1
};

struct ReplicaResult {
  std::vector<ReplicaPart> parts;
  Tree info;  // format, quads, chunks, chunk_quads, chunk_atlas_dims, tile_texels, gutter_texels, tile_pitch, atlas_side
};

// Refuses (AssetError) a missing mesh.ply, sorted_faces.bin or chunk atlas, a sorted_faces.bin that is not a permutation
// of the faces, a non-quad face, a vertex index out of range, tile_size != 16, an atlas that is not square, not a
// multiple of the tile size or too small for its chunk, a negative or non-finite source texel, an atlas of more than 2^32
// texels or one whose side breaks (|uv| + 32) * side <= 2^30 (the renderer's EWA index bound).
ReplicaResult import_replica(const std::string& scene_dir, const ReplicaInputs& in, const ReplicaOptions& opt);

// The exact value of IEEE binary16 bits (65504 is the largest finite one).
double binary16_value(uint16_t bits);

// JSON side-file geometry: reflector areas and gravity-direction diagnostics (no SDK source used).
Tree replica_reports(const std::vector<std::vector<std::array<double, 3>>>& polygons,
                     const std::optional<std::array<double, 3>>& gravity);

}  // namespace balla::scene::assets
