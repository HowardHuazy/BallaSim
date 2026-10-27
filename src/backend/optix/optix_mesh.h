// OptiX 9.1 single-mesh ray query: one GAS, a raygen / closest-hit / miss pipeline compiled to
// OptiX-IR at build time and embedded in the library.
#pragma once

#include "backend/mesh_query.h"

namespace balla::backend {

// Throws std::invalid_argument for an out-of-range vertex index or more than 2^30 rays, and
// std::runtime_error on any CUDA or OptiX error (no device, driver without OptiX, ...).
MeshHits optix_intersect(const TriangleMeshView& mesh, const RayBatchView& rays);

}  // namespace balla::backend
