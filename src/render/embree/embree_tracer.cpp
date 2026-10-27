#include "render/embree/embree_tracer.h"

#include <embree4/rtcore.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "render/ray.h"
#include "render/scene_data.h"
#include "render/shade/ao_traced.h"
#include "render/shade/lights.h"

namespace balla::render {
namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Requests (or centre rays) per parallel chunk; any value gives identical results.
constexpr size_t kRequestGrain = 16;
constexpr size_t kCentreGrain = 1024;

void check(RTCDevice device, const char* what) {
  const RTCError err = rtcGetDeviceError(device);
  if (err != RTC_ERROR_NONE) {
    throw std::runtime_error(std::string("Embree error in ") + what + ": " + rtcGetErrorString(err));
  }
}

uint16_t to_unorm16(float x) { return static_cast<uint16_t>(std::lround(std::clamp(x, 0.0f, 1.0f) * 65535.0f)); }

void set_ray(RTCRay& ray, const Ray& r, float tmax) {
  ray.org_x = r.o.x;
  ray.org_y = r.o.y;
  ray.org_z = r.o.z;
  ray.dir_x = r.d.x;
  ray.dir_y = r.d.y;
  ray.dir_z = r.d.z;
  ray.tnear = 0.0f;
  ray.tfar = tmax;
  ray.time = 0.0f;
  ray.mask = 0xFFFFFFFFu;
  ray.id = 0;
  ray.flags = 0;
}

}  // namespace

struct EmbreeTracer::Impl {
  explicit Impl(ThreadPool& p) : pool(p) {}
  ~Impl() {
    for (RTCScene s : scenes) {
      if (s != nullptr) rtcReleaseScene(s);
    }
    if (held != nullptr) rtcReleaseScene(held);
    if (device != nullptr) rtcReleaseDevice(device);
  }

  ThreadPool& pool;
  RTCDevice device = nullptr;
  std::vector<RTCScene> scenes;  // per geometry of the scene; nullptr when it has no triangles
  RTCScene held = nullptr;       // the merged world of a held set (use_traces)
  std::vector<RTCScene> active;  // per trace geometry in use
  SceneView view{};

  RTCScene make_scene(const GeometryView& g) const {
    if (g.num_triangles == 0) return nullptr;
    RTCScene rs = rtcNewScene(device);
    RTCGeometry geom = nullptr;
    try {
      rtcSetSceneFlags(rs, RTC_SCENE_FLAG_ROBUST);
      rtcSetSceneBuildQuality(rs, RTC_BUILD_QUALITY_HIGH);
      geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
      // Embree-allocated buffers carry the padding its SIMD loads need past the last vertex.
      auto* vb = static_cast<float*>(rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                                             3 * sizeof(float), g.num_vertices));
      auto* ib = static_cast<uint32_t*>(rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                                                                3 * sizeof(uint32_t), g.num_triangles));
      check(device, "rtcSetNewGeometryBuffer");
      std::copy(g.positions, g.positions + 3 * size_t{g.num_vertices}, vb);
      std::copy(g.indices, g.indices + 3 * size_t{g.num_triangles}, ib);
      rtcCommitGeometry(geom);
      rtcAttachGeometry(rs, geom);
      rtcReleaseGeometry(std::exchange(geom, nullptr));  // the scene owns it now
      rtcCommitScene(rs);
      check(device, "rtcCommitScene");
    } catch (...) {
      // A failed check leaks no handle (a null one is not released: that would leave a device error).
      if (geom != nullptr) rtcReleaseGeometry(geom);
      if (rs != nullptr) rtcReleaseScene(rs);
      throw;
    }
    return rs;
  }

  // Closest hit of the camera-centre ray d_cam (DESIGN §6.9 per-ray work), in the OptiX tracer's
  // order: sphere-culled movers by ascending id, then the static world.
  HitRecord closest(const PoseTableView& poses, uint32_t slot, Vec3f d_cam) const {
    HitRecord best{kInstanceNone, 0, 0, 0, kInf};
    for (int32_t i = 0; i < view.num_geometries; ++i) {
      const int32_t g = i + 1 < view.num_geometries ? i + 1 : 0;
      if (active[static_cast<size_t>(g)] == nullptr) continue;
      const PoseRow& row = pose_row(poses, slot, g);
      if (g > 0 && !sphere_overlaps(row, Vec3f{0.0f, 0.0f, 0.0f}, d_cam, best.t_hit)) continue;
      RTCRayHit rh;
      set_ray(rh.ray, primary_to_geometry(row, d_cam), best.t_hit);
      rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
      rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
      rtcIntersect1(active[static_cast<size_t>(g)], &rh);
      if (rh.hit.geomID == RTC_INVALID_GEOMETRY_ID) continue;
      const GeometryView& geo = view.geometries[g];
      best = {geo.instances != nullptr ? geo.instances[rh.hit.primID] : geo.instance, rh.hit.primID,
              to_unorm16(rh.hit.u), to_unorm16(rh.hit.v), rh.ray.tfar};
    }
    return best;
  }

  bool occluded(const RTCScene scene, const Ray& r, float tmax) const {
    RTCRay ray;
    set_ray(ray, r, tmax);
    rtcOccluded1(scene, &ray);
    return ray.tfar < 0.0f;  // Embree sets tfar = -inf on occlusion
  }

  // Any hit of a shadow ray leaving geometry g_hit: movers first (sphere-culled), then the world; of the geometries
  // of kWhich (lights.h ShadowWhich: kInserted the first pass without g_hit, kWorld the second).
  template <ShadowWhich kWhich>
  bool shadow_blocked(const PoseTableView& poses, uint32_t slot, int32_t g_hit, const ShadowRay& sr) const {
    const int32_t first_pass = kWhich == ShadowWhich::kWorld ? 1 : 0;
    const int32_t end_pass = kWhich == ShadowWhich::kInserted ? 1 : 2;
    for (int32_t pass = first_pass; pass < end_pass; ++pass) {
      const int32_t g_begin = pass == 0 ? 1 : 0, g_end = pass == 0 ? view.num_geometries : 1;
      for (int32_t g = g_begin; g < g_end; ++g) {
        if (active[static_cast<size_t>(g)] == nullptr || (kWhich == ShadowWhich::kInserted && g == g_hit)) continue;
        const PoseRow& row = pose_row(poses, slot, g);
        if (g > 0 && !sphere_overlaps(row, sr.origin_cam, sr.dir_cam, sr.tmax)) continue;
        Ray r = to_geometry(row, sr.origin_cam, sr.dir_cam);
        if (g == g_hit) r.o = sr.origin_hit;  // exact offset origin against self-intersection
        if (occluded(active[static_cast<size_t>(g)], r, sr.tmax)) return true;
      }
    }
    return false;
  }

  // accuracy.mover_ao traced (P2-F F9b; render/shade/ao_traced.h): any hit of an ambient ray leaving geometry g_hit on
  // the other movers, each up to its reach (sphere-culled, by ascending id).
  bool ao_blocked(const PoseTableView& poses, uint32_t slot, int32_t g_hit, const AoRay& ar) const {
    for (int32_t g = 1; g < view.num_geometries; ++g) {
      const float tmax = view.ao_reach[g];
      if (g == g_hit || !(tmax > 0.0f) || active[static_cast<size_t>(g)] == nullptr) continue;
      const PoseRow& row = pose_row(poses, slot, g);
      if (!sphere_overlaps(row, ar.origin_cam, ar.dir_cam, tmax)) continue;
      if (occluded(active[static_cast<size_t>(g)], to_geometry(row, ar.origin_cam, ar.dir_cam), tmax)) return true;
    }
    return false;
  }

  // The plan pass (ENGINE_CPP I8): the first geometry in shadow_blocked's order (of the geometries of kWhich) that a
  // shadow ray hits, and the ray parameter of its closest hit there (*t); -1 when nothing blocks. When that geometry
  // is a mover, *multi tells whether a later moving geometry (GeometryView::dynamic: not a static body split from the
  // world, scans.md §4.2) blocks the ray too (the I8 review).
  template <ShadowWhich kWhich>
  int32_t shadow_blocker_t(const PoseTableView& poses, uint32_t slot, int32_t g_hit, const ShadowRay& sr, float* t,
                           bool* multi) const {
    int32_t first = -1;
    const int32_t first_pass = kWhich == ShadowWhich::kWorld ? 1 : 0;
    const int32_t end_pass = kWhich == ShadowWhich::kInserted ? 1 : 2;
    for (int32_t pass = first_pass; pass < end_pass && first <= 0; ++pass) {
      const int32_t g_begin = pass == 0 ? 1 : 0, g_end = pass == 0 ? view.num_geometries : 1;
      for (int32_t g = g_begin; g < g_end; ++g) {
        if (active[static_cast<size_t>(g)] == nullptr || (kWhich == ShadowWhich::kInserted && g == g_hit)) continue;
        if (first > 0 && view.geometries[g].dynamic == 0) continue;  // a split static body is no moving shadow
        const PoseRow& row = pose_row(poses, slot, g);
        if (g > 0 && !sphere_overlaps(row, sr.origin_cam, sr.dir_cam, sr.tmax)) continue;
        Ray r = to_geometry(row, sr.origin_cam, sr.dir_cam);
        if (g == g_hit) r.o = sr.origin_hit;
        if (first > 0) {
          if (occluded(active[static_cast<size_t>(g)], r, sr.tmax)) {
            *multi = true;
            return first;
          }
          continue;
        }
        RTCRayHit rh;
        set_ray(rh.ray, r, sr.tmax);
        rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        rtcIntersect1(active[static_cast<size_t>(g)], &rh);
        if (rh.hit.geomID == RTC_INVALID_GEOMETRY_ID) continue;
        *t = rh.ray.tfar;
        first = g;
        if (g == 0) return first;
      }
    }
    return first;
  }
};

EmbreeTracer::EmbreeTracer(ThreadPool& pool) : impl_(std::make_unique<Impl>(pool)) {}
EmbreeTracer::~EmbreeTracer() = default;

void EmbreeTracer::build(const SceneData&, const SceneView& view) {
  Impl& s = *impl_;
  s.device = rtcNewDevice(nullptr);
  if (s.device == nullptr) {
    throw std::runtime_error(std::string("Embree error in rtcNewDevice: ") +
                             rtcGetErrorString(rtcGetDeviceError(nullptr)));
  }
  s.view = view;
  for (int32_t g = 0; g < view.num_geometries; ++g) s.scenes.push_back(s.make_scene(view.geometries[g]));
  s.active = s.scenes;
}

void EmbreeTracer::use_traces(const TraceSet* t, const SceneView& view) {
  Impl& s = *impl_;
  s.active = s.scenes;  // the scene's own structures first (needs no memory)
  s.view = view;
  if (s.held != nullptr) rtcReleaseScene(s.held);
  s.held = nullptr;
  if (t == nullptr) return;
  s.held = s.make_scene(view.geometries[0]);  // the executor's merged world
  s.active = t->select(s.scenes.data(), s.held);
}

void EmbreeTracer::sample(const TraceBatch& b, HitRecord* hits, uint32_t* occluded, uint32_t* static_occluded) {
  const Impl& s = *impl_;
  // One instantiation per batch (the baked-proxy rule of lights.h shadow_mask only with a static-word buffer).
  const auto run = [&](auto baked) {
    constexpr bool kBaked = decltype(baked)::value;
    s.pool.parallel_for(b.num_requests, kRequestGrain, [&](size_t begin, size_t end) {
      for (size_t r = begin; r < end; ++r) {
        const SampleRequest& req = b.requests[r];
        const CameraView& cam = s.view.cameras[req.cam];
        for (uint32_t k = 0; k < b.subsamples; ++k) {
          const size_t i = r * b.subsamples + k;
          const Vec3f d = subsample_direction(s.view, cam, req.pixel, req.k_begin + k);
          hits[i] = s.closest(b.poses, req.pose_slot, d);
          if (occluded != nullptr) {
            const auto blocked = [&](int32_t g_hit, const ShadowRay& sr, auto which) {
              return s.shadow_blocked<decltype(which)::value>(b.poses, req.pose_slot, g_hit, sr);
            };
            occluded[i] = shadow_mask<kBaked>(s.view, b.poses, req.pose_slot, hits[i], d, req.cam, req.pixel,
                                              req.k_begin + k, blocked, kBaked ? static_occluded + i : nullptr);
            if (s.view.mover_ao == kMoverAoTraced && s.view.ao_reach != nullptr) {  // P2-F F9b: the AO bit
              occluded[i] |= ao_traced_mask(s.view, b.poses, req.pose_slot, cam, req.pixel, req.k_begin + k, hits[i], d,
                                            [&](int32_t g_hit, const AoRay& ar) {
                                              return s.ao_blocked(b.poses, req.pose_slot, g_hit, ar);
                                            });
            }
          } else if (static_occluded != nullptr) {
            static_occluded[i] = 0;  // no shadow rays, no static word
          }
        }
      }
    });
  };
  if (static_occluded != nullptr && occluded != nullptr) {
    run(std::true_type{});
  } else {
    run(std::false_type{});
  }
}

void EmbreeTracer::centre(const GtBatch& b, GtHit* out, uint32_t* shadow) {
  const Impl& s = *impl_;
  const bool baked = s.view.baked_proxies != 0;  // one instantiation per batch, as sample()
  s.pool.parallel_for(b.count, kCentreGrain, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      const GtRequest& q = b.requests[i];
      const Vec3f d = centre_direction(s.view.cameras[q.cam], q.x, q.y);
      const HitRecord h = s.closest(b.poses, q.pose_slot, d);
      out[i] = {h.instance, h.prim, h.t_hit, 0};
      if (shadow == nullptr) continue;
      uint32_t* words = shadow + 2 * i * static_cast<size_t>(s.view.num_shadow_lights);
      const auto blocker = [&](int32_t g_hit, const ShadowRay& sr, float* t, bool* multi, auto which) {
        return s.shadow_blocker_t<decltype(which)::value>(b.poses, q.pose_slot, g_hit, sr, t, multi);
      };
      if (b.proxy_visibility) {
        proxy_visibility_states(s.view, b.poses, q.pose_slot, h, d,
                                [&](int32_t g_hit, const ShadowRay& sr, auto which) {
                                  return s.shadow_blocked<decltype(which)::value>(b.poses, q.pose_slot, g_hit, sr);
                                },
                                words);
      } else if (baked) {
        shadow_states<true>(s.view, b.poses, q.pose_slot, h, d, blocker, words);
      } else {
        shadow_states<false>(s.view, b.poses, q.pose_slot, h, d, blocker, words);
      }
    }
  });
}

}  // namespace balla::render
