// Embree 4 single-mesh ray query (robust mode, high-quality build).
#pragma once

#include "backend/mesh_query.h"

namespace balla::backend {

// Throws std::invalid_argument for an out-of-range vertex index, std::runtime_error with Embree's
// error text on failure.
MeshHits embree_intersect(const TriangleMeshView& mesh, const RayBatchView& rays);

}  // namespace balla::backend
