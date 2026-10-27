#include "backend/embree/embree_mesh.h"

#include <embree4/rtcore.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace balla::backend {
namespace {

struct DeviceDeleter {
  void operator()(RTCDevice d) const { rtcReleaseDevice(d); }
};
struct SceneDeleter {
  void operator()(RTCScene s) const { rtcReleaseScene(s); }
};
using DevicePtr = std::unique_ptr<RTCDeviceTy, DeviceDeleter>;
using ScenePtr = std::unique_ptr<RTCSceneTy, SceneDeleter>;

void check(RTCDevice device, const char* what) {
  const RTCError err = rtcGetDeviceError(device);
  if (err != RTC_ERROR_NONE) {
    throw std::runtime_error(std::string("Embree error in ") + what + ": " +
                             rtcGetErrorString(err));
  }
}

}  // namespace

MeshHits embree_intersect(const TriangleMeshView& mesh, const RayBatchView& rays) {
  check_indices(mesh);
  DevicePtr device(rtcNewDevice(nullptr));
  if (!device) {
    throw std::runtime_error(std::string("Embree error in rtcNewDevice: ") +
                             rtcGetErrorString(rtcGetDeviceError(nullptr)));
  }
  ScenePtr scene(rtcNewScene(device.get()));
  rtcSetSceneFlags(scene.get(), RTC_SCENE_FLAG_ROBUST);
  rtcSetSceneBuildQuality(scene.get(), RTC_BUILD_QUALITY_HIGH);

  RTCGeometry geom = rtcNewGeometry(device.get(), RTC_GEOMETRY_TYPE_TRIANGLE);
  auto* vb = static_cast<float*>(rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
                                                         RTC_FORMAT_FLOAT3, 3 * sizeof(float),
                                                         mesh.num_vertices));
  auto* ib = static_cast<uint32_t*>(rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
                                                            RTC_FORMAT_UINT3, 3 * sizeof(uint32_t),
                                                            mesh.num_triangles));
  check(device.get(), "rtcSetNewGeometryBuffer");
  std::copy_n(mesh.vertices, 3 * mesh.num_vertices, vb);
  std::copy_n(mesh.indices, 3 * mesh.num_triangles, ib);
  rtcCommitGeometry(geom);
  rtcAttachGeometry(scene.get(), geom);
  rtcReleaseGeometry(geom);
  rtcCommitScene(scene.get());
  check(device.get(), "rtcCommitScene");

  MeshHits hits = all_misses(rays.count);
  for (size_t i = 0; i < rays.count; ++i) {
    RTCRayHit rh{};
    rh.ray.org_x = rays.origins[3 * i];
    rh.ray.org_y = rays.origins[3 * i + 1];
    rh.ray.org_z = rays.origins[3 * i + 2];
    rh.ray.dir_x = rays.directions[3 * i];
    rh.ray.dir_y = rays.directions[3 * i + 1];
    rh.ray.dir_z = rays.directions[3 * i + 2];
    rh.ray.tnear = 0.0f;
    rh.ray.tfar = std::numeric_limits<float>::infinity();
    rh.ray.mask = 0xFFFFFFFFu;
    rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcIntersect1(scene.get(), &rh);
    if (rh.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
      hits.t[i] = rh.ray.tfar;
      hits.prim[i] = rh.hit.primID;
      hits.u[i] = rh.hit.u;
      hits.v[i] = rh.hit.v;
    }
  }
  check(device.get(), "rtcIntersect1");
  return hits;
}

}  // namespace balla::backend
