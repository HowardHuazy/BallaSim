// Shading patches (accuracy.shading_patches crease; plan p2f §4.4, P2-F F5; DESIGN §5.2): a body's triangles split
// into the patches that the renderer's clusters and the visibility signature key by (instance, patch) instead of
// (instance, material) (render/shade/shade.h hit_key). Within one body, an edge between two triangles is a patch
// boundary when
//   (a) their materials differ;
//   (b) the mesh declares a hard edge there (the two triangles' vertices at either end of the edge carry normals that
//       differ by more than kPatchHardEdgeRad) and either the dihedral angle exceeds kPatchCreaseRad or either
//       triangle's material has a specular lobe;
//   (c) the mesh has no normals and the dihedral angle exceeds kPatchCreaseRad.
// Edges join triangles through welded positions (vertices with equal coordinates), so UV seams, which duplicate
// vertices with equal normals, do not cut; disconnected shells are separate patches. The dihedral angle is the angle
// between the two triangles' winding normals (a zero-area triangle has none and joins its neighbours). Patches are the
// connected components of the remaining edges (union-find), numbered in order of their first triangle, so the result
// depends only on the mesh. Flat-shaded Lambert facets below 30 degrees stay one patch: the unnormalized mean normal of
// a cluster is exact for unshadowed Lambert (shade.h cluster_radiance), so they need no cut. Glossy hard edges are cut
// at any angle (a highlight is discontinuous there). No material has a specular lobe before P2-F's GGX (F4, F6), so
// rule (b)'s specular branch is reached only through `specular` here.
//
// Both thresholds are the plan's maths-bounded constants (§4.11), not parameters.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace balla::render {

inline constexpr double kPatchCreaseRad = 0.52359877559829887;  // 30 degrees (DESIGN §5.2)
inline constexpr double kPatchHardEdgeRad = 1e-3;               // welded vertices' normals: a declared hard edge

struct PatchMesh {
  std::span<const float> positions;   // V x 3
  std::span<const float> normals;     // V x 3, or empty (rule (c))
  std::span<const uint32_t> indices;  // T x 3
};

struct Patches {
  std::vector<uint32_t> of_triangle;  // T: the patch of each triangle, 0 .. count - 1
  uint32_t count = 0;
};

// The patches of one body's mesh. materials: T material indices (any numbering); specular: per material index whether
// it has a specular lobe (empty: none has). Host code, O(T log T).
Patches build_patches(const PatchMesh& mesh, std::span<const uint32_t> materials, std::span<const uint8_t> specular);

}  // namespace balla::render
