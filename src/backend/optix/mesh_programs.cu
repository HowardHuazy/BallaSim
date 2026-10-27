// OptiX device programs of the single-mesh ray query (compiled to OptiX-IR).
#include <optix_device.h>

#include "backend/mesh_query.h"
#include "backend/optix/mesh_programs.h"

extern "C" {
__constant__ balla::optix::MeshLaunchParams params;
}

extern "C" __global__ void __raygen__mesh() {
  const unsigned int i = optixGetLaunchIndex().x;
  if (i >= params.count) return;
  const float3 origin =
      make_float3(params.origins[3 * i], params.origins[3 * i + 1], params.origins[3 * i + 2]);
  const float3 direction = make_float3(params.directions[3 * i], params.directions[3 * i + 1],
                                       params.directions[3 * i + 2]);
  const float inf = __int_as_float(0x7f800000);  // t in [0, inf), as Embree (mesh_query.h)
  unsigned int t_bits = __float_as_uint(inf);
  unsigned int prim = balla::backend::kNoPrim;
  unsigned int u_bits = 0, v_bits = 0;
  optixTrace(params.handle, origin, direction, 0.0f, inf, 0.0f, OptixVisibilityMask(255),
             OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, t_bits, prim, u_bits, v_bits);
  params.t[i] = __uint_as_float(t_bits);
  params.prim[i] = prim;
  params.u[i] = __uint_as_float(u_bits);
  params.v[i] = __uint_as_float(v_bits);
}

extern "C" __global__ void __closesthit__mesh() {
  const float2 b = optixGetTriangleBarycentrics();
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetPrimitiveIndex());
  optixSetPayload_2(__float_as_uint(b.x));
  optixSetPayload_3(__float_as_uint(b.y));
}

// A miss leaves the payload at its initial (miss) values.
extern "C" __global__ void __miss__mesh() {}
