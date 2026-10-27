// OptiX device programs of the renderer (compiled to OptiX-IR and embedded).
//
// G6 design rules: one compile-time-specialized pipeline per variant (an "uber" raygen with a
// runtime mode switch measured 2.5-7x slower), one thread per sub-sample ray with consecutive
// threads on the sub-samples of one pixel-sample (warp-uniform pose loads, coherent rays), poses
// read from the pose table rather than evaluated per ray, the static world traced as a direct GAS,
// movers sphere-culled and traced in their body frames at the ray's exact pose (multi-trace), and
// minimal live state across traversals. Ray generation, frame changes, culling and shadow-ray
// set-up are the shared BALLA_HD code that the Embree tracer runs too.
//
// Rays are traced with optixTraverse and read back through hit-object queries: no closest-hit or
// miss program runs (the pipelines link null ones), and no payload is passed. Against optixTrace with
// a closest-hit program that wrote the same three values, this is bitwise equal and takes 2-5 % less
// GPU time on the replayed bench windows (ENGINE_CPP §0.1, I1b notes).
#include <optix_device.h>

#include "render/optix/launch_params.h"
#include "render/ray.h"
#include "render/shade/ao_traced.h"
#include "render/shade/lights.h"

extern "C" {
__constant__ balla::render::optix::LaunchParams params;
}

namespace {

using namespace balla;
using namespace balla::render;

__forceinline__ __device__ float3 f3(Vec3f v) { return make_float3(v.x, v.y, v.z); }

// Barycentrics of the traversed hit, quantized to 16 bits each (HitRecord::b1, b2).
__forceinline__ __device__ uint16_t quantize_barycentric(float b) {
  return static_cast<uint16_t>(__float2uint_rn(fminf(fmaxf(b, 0.0f), 1.0f) * 65535.0f));
}

// Closest hit of the camera-centre ray d_cam: the sphere-culled movers in ascending instance id,
// then the static world, each traced with tmax = the best distance so far (movers first: on
// mover-heavy pixels the large world GAS is then traversed with a short tmax, +10 % measured).
__forceinline__ __device__ HitRecord closest(uint32_t slot, Vec3f d_cam) {
  const SceneView& s = params.scene;
  HitRecord best{kInstanceNone, 0, 0, 0, infinity<float>()};
  for (int32_t i = 0; i < s.num_geometries; ++i) {
    const int32_t g = i + 1 < s.num_geometries ? i + 1 : 0;  // movers first, then the world
    const OptixTraversableHandle handle = params.handles[g];
    if (handle == 0) continue;
    const PoseRow& row = pose_row(params.poses, slot, g);
    if (g > 0 && !sphere_overlaps(row, Vec3f{0.0f, 0.0f, 0.0f}, d_cam, best.t_hit)) continue;
    const Ray r = primary_to_geometry(row, d_cam);
    optixTraverse(handle, f3(r.o), f3(r.d), 0.0f, best.t_hit, 0.0f, OptixVisibilityMask(255),
                  OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0);
    if (!optixHitObjectIsHit()) continue;
    const unsigned prim = optixHitObjectGetPrimitiveIndex();
    const float2 b = optixHitObjectGetTriangleBarycentrics();
    const GeometryView& geo = s.geometries[g];
    best = {geo.instances != nullptr ? geo.instances[prim] : geo.instance, prim, quantize_barycentric(b.x),
            quantize_barycentric(b.y), optixHitObjectGetRayTmax()};
  }
  return best;
}

// Any hit along a shadow ray leaving geometry g_hit: movers first (sphere-culled), then the world; of the geometries
// of kWhich (lights.h ShadowWhich: kInserted the movers without g_hit, kWorld the world), a compile-time constant.
template <ShadowWhich kWhich>
__device__ bool shadow_blocked(uint32_t slot, int32_t g_hit, const ShadowRay& sr) {
  const int32_t n = params.scene.num_geometries;
  const int32_t i_begin = kWhich == ShadowWhich::kWorld ? n - 1 : 0;   // kWorld: g = 0 alone
  const int32_t i_end = kWhich == ShadowWhich::kInserted ? n - 1 : n;  // kInserted: g = 1, ..., n - 1
  for (int32_t i = i_begin; i < i_end; ++i) {
    const int32_t g = i + 1 < n ? i + 1 : 0;  // 1, 2, ..., n - 1, then 0
    const OptixTraversableHandle handle = params.handles[g];
    if (handle == 0 || (kWhich == ShadowWhich::kInserted && g == g_hit)) continue;
    const PoseRow& row = pose_row(params.poses, slot, g);
    if (g > 0 && !sphere_overlaps(row, sr.origin_cam, sr.dir_cam, sr.tmax)) continue;
    Ray r = to_geometry(row, sr.origin_cam, sr.dir_cam);
    if (g == g_hit) r.o = sr.origin_hit;  // exact offset origin against self-intersection
    optixTraverse(handle, f3(r.o), f3(r.d), 0.0f, sr.tmax, 0.0f, OptixVisibilityMask(255),
                  OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0, 1, 0);
    if (optixHitObjectIsHit()) return true;
  }
  return false;
}

// accuracy.mover_ao traced (P2-F F9b; render/shade/ao_traced.h): any hit of an ambient ray leaving geometry g_hit on
// the other movers, each up to its reach (sphere-culled, by ascending id), as the Embree tracer.
__device__ bool ao_blocked(uint32_t slot, int32_t g_hit, const AoRay& ar) {
  const SceneView& s = params.scene;
  for (int32_t g = 1; g < s.num_geometries; ++g) {
    const float tmax = s.ao_reach[g];
    const OptixTraversableHandle handle = params.handles[g];
    if (g == g_hit || !(tmax > 0.0f) || handle == 0) continue;
    const PoseRow& row = pose_row(params.poses, slot, g);
    if (!sphere_overlaps(row, ar.origin_cam, ar.dir_cam, tmax)) continue;
    const Ray r = to_geometry(row, ar.origin_cam, ar.dir_cam);
    optixTraverse(handle, f3(r.o), f3(r.d), 0.0f, tmax, 0.0f, OptixVisibilityMask(255),
                  OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0, 1, 0);
    if (optixHitObjectIsHit()) return true;
  }
  return false;
}

// The plan pass (ENGINE_CPP I8): the first geometry in shadow_blocked's order (of the geometries of kWhich) that a
// shadow ray hits, and the ray parameter of its closest hit there (*t); -1 when nothing blocks. When that geometry is
// a mover, *multi tells whether a later moving geometry (GeometryView::dynamic: not a static body split from the world,
// scans.md §4.2) blocks the ray too (an any-hit query each; the I8 review).
template <ShadowWhich kWhich>
__device__ int32_t shadow_blocker_t(uint32_t slot, int32_t g_hit, const ShadowRay& sr, float* t, bool* multi) {
  const int32_t n = params.scene.num_geometries;
  const int32_t i_begin = kWhich == ShadowWhich::kWorld ? n - 1 : 0;
  const int32_t i_end = kWhich == ShadowWhich::kInserted ? n - 1 : n;
  int32_t first = -1;
  for (int32_t i = i_begin; i < i_end; ++i) {
    const int32_t g = i + 1 < n ? i + 1 : 0;
    if (first > 0 && g == 0) break;  // the world after a mover: no other mover
    const OptixTraversableHandle handle = params.handles[g];
    if (handle == 0 || (kWhich == ShadowWhich::kInserted && g == g_hit)) continue;
    if (first > 0 && params.scene.geometries[g].dynamic == 0) continue;  // a split static body is no moving shadow
    const PoseRow& row = pose_row(params.poses, slot, g);
    if (g > 0 && !sphere_overlaps(row, sr.origin_cam, sr.dir_cam, sr.tmax)) continue;
    Ray r = to_geometry(row, sr.origin_cam, sr.dir_cam);
    if (g == g_hit) r.o = sr.origin_hit;
    const unsigned flags = first < 0 ? OPTIX_RAY_FLAG_DISABLE_ANYHIT
                                     : OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
    optixTraverse(handle, f3(r.o), f3(r.d), 0.0f, sr.tmax, 0.0f, OptixVisibilityMask(255), flags, 0, 1, 0);
    if (!optixHitObjectIsHit()) continue;
    if (first >= 0) {
      *multi = true;
      break;
    }
    *t = optixHitObjectGetRayTmax();
    first = g;
  }
  return first;
}

}  // namespace

// Closest hit of sub-sample idx of the launch; returns its camera-frame direction and request too.
__forceinline__ __device__ bool trace_subsample(unsigned idx, SampleRequest* req, Vec3f* d) {
  const uint32_t r = idx / params.subsamples;
  if (r >= params.num_requests) return false;
  *req = params.requests[r];
  const uint32_t k = idx - r * params.subsamples;
  *d = subsample_direction(params.scene, params.scene.cameras[req->cam], req->pixel, req->k_begin + k);
  params.hits[idx] = closest(req->pose_slot, *d);
  return true;
}

// Visibility only (scenes without shadow-casting lights; the probe variant of DESIGN §6.9).
extern "C" __global__ void __raygen__sample() {
  SampleRequest req;
  Vec3f d;
  trace_subsample(optixGetLaunchIndex().x, &req, &d);
}

// Visibility plus the shadow rays of the hit, fused so the ray and the triangle are not refetched.
extern "C" __global__ void __raygen__sample_shadow() {
  const unsigned idx = optixGetLaunchIndex().x;
  SampleRequest req;
  Vec3f d;
  if (!trace_subsample(idx, &req, &d)) return;
  params.occluded[idx] = shadow_mask<false>(params.scene, params.poses, req.pose_slot, params.hits[idx], d, req.cam,
                                            req.pixel,
                                            req.k_begin + idx % params.subsamples,  // the sub-sample's place in its request
                                            [&](int32_t g_hit, const ShadowRay& sr, auto which) {
                                              return shadow_blocked<decltype(which)::value>(req.pose_slot, g_hit, sr);
                                            });
}

// The same in a scene with baked proxies (SceneView::baked_proxies; scans.md §4.2): a baked receiver's proxies follow
// the inserted / scan rule of lights.h shadow_mask, and the sub-sample's static word is written too. Its own pipeline
// (G6): the lit raygens above compile none of this.
extern "C" __global__ void __raygen__sample_shadow_baked() {
  const unsigned idx = optixGetLaunchIndex().x;
  SampleRequest req;
  Vec3f d;
  if (!trace_subsample(idx, &req, &d)) return;
  uint32_t statics = 0;
  params.occluded[idx] = shadow_mask<true>(params.scene, params.poses, req.pose_slot, params.hits[idx], d, req.cam,
                                           req.pixel, req.k_begin + idx % params.subsamples,
                                           [&](int32_t g_hit, const ShadowRay& sr, auto which) {
                                             return shadow_blocked<decltype(which)::value>(req.pose_slot, g_hit, sr);
                                           },
                                           &statics);
  params.static_occluded[idx] = statics;
}

// accuracy.mover_ao traced (P2-F F9b; render/shade/ao_traced.h): the above plus the hit's ambient ray, its AO bit in
// the occlusion word (a pipeline of its own, so the other variants keep their code).
extern "C" __global__ void __raygen__sample_shadow_ao() {
  const unsigned idx = optixGetLaunchIndex().x;
  SampleRequest req;
  Vec3f d;
  if (!trace_subsample(idx, &req, &d)) return;
  const uint32_t k = req.k_begin + (idx - (idx / params.subsamples) * params.subsamples);
  const uint32_t mask =  // P2-F F8a-1: shadow_mask takes (camera, pixel, k), the area lights' points
      shadow_mask<false>(params.scene, params.poses, req.pose_slot, params.hits[idx], d, req.cam, req.pixel, k,
                         [&](int32_t g_hit, const ShadowRay& sr, auto which) {
                           return shadow_blocked<decltype(which)::value>(req.pose_slot, g_hit, sr);
                         });
  params.occluded[idx] = mask | ao_traced_mask(params.scene, params.poses, req.pose_slot,
                                               params.scene.cameras[req.cam], req.pixel, k, params.hits[idx], d,
                                               [&](int32_t g_hit, const AoRay& ar) {
                                                 return ao_blocked(req.pose_slot, g_hit, ar);
                                               });
}

extern "C" __global__ void __raygen__centre() {
  const unsigned idx = optixGetLaunchIndex().x;
  if (idx >= params.num_gt) return;
  const GtRequest q = params.gt_requests[idx];
  const HitRecord h = closest(q.pose_slot, centre_direction(params.scene.cameras[q.cam], q.x, q.y));
  params.gt_hits[idx] = {h.instance, h.prim, h.t_hit, 0};
}

// The plan pass's centre rays with the shadow states of their hits (ENGINE_CPP I8).
extern "C" __global__ void __raygen__centre_shadow() {
  const unsigned idx = optixGetLaunchIndex().x;
  if (idx >= params.num_gt) return;
  const GtRequest q = params.gt_requests[idx];
  const Vec3f d = centre_direction(params.scene.cameras[q.cam], q.x, q.y);
  const HitRecord h = closest(q.pose_slot, d);
  params.gt_hits[idx] = {h.instance, h.prim, h.t_hit, 0};
  shadow_states<false>(params.scene, params.poses, q.pose_slot, h, d,
                       [&](int32_t g_hit, const ShadowRay& sr, float* t, bool* multi, auto which) {
                         return shadow_blocker_t<decltype(which)::value>(q.pose_slot, g_hit, sr, t, multi);
                       },
                       params.shadow_states +
                           2 * static_cast<size_t>(idx) * static_cast<size_t>(params.scene.num_shadow_lights));
}

// The centre-shadow pipeline of scenes with baked proxies (scans.md §4.2, §7): the plan pass's states with the baked
// receivers' rule (lights.h shadow_states<true>), or, with LaunchParams::proxy_visibility (Renderer::proxy_visibility,
// the start diagnostic), the scan's visibility of every proxy from the baked hits (lights.h proxy_visibility_states).
// Its own pipeline (G6): the lit raygen above is the one it was.
extern "C" __global__ void __raygen__centre_shadow_baked() {
  const unsigned idx = optixGetLaunchIndex().x;
  if (idx >= params.num_gt) return;
  const GtRequest q = params.gt_requests[idx];
  const Vec3f d = centre_direction(params.scene.cameras[q.cam], q.x, q.y);
  const HitRecord h = closest(q.pose_slot, d);
  params.gt_hits[idx] = {h.instance, h.prim, h.t_hit, 0};
  uint32_t* out = params.shadow_states +
                  2 * static_cast<size_t>(idx) * static_cast<size_t>(params.scene.num_shadow_lights);
  if (params.proxy_visibility != 0) {
    proxy_visibility_states(params.scene, params.poses, q.pose_slot, h, d,
                            [&](int32_t g_hit, const ShadowRay& sr, auto which) {
                              return shadow_blocked<decltype(which)::value>(q.pose_slot, g_hit, sr);
                            },
                            out);
    return;
  }
  shadow_states<true>(params.scene, params.poses, q.pose_slot, h, d,
                      [&](int32_t g_hit, const ShadowRay& sr, float* t, bool* multi, auto which) {
                        return shadow_blocker_t<decltype(which)::value>(q.pose_slot, g_hit, sr, t, multi);
                      },
                      out);
}
