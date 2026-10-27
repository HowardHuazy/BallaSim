// Launch parameters shared by optix_mesh.cpp (host) and mesh_programs.cu (device).
#pragma once

#include <optix_types.h>

namespace balla::optix {

struct MeshLaunchParams {
  OptixTraversableHandle handle;
  const float* origins;     // 3 floats per ray
  const float* directions;  // 3 floats per ray
  float* t;                 // +inf on a miss
  unsigned int* prim;       // backend::kNoPrim on a miss
  float* u;
  float* v;
  unsigned int count;
};

}  // namespace balla::optix
