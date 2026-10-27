// Mover AO, the traced form (accuracy.mover_ao traced; plan p2f §4.7 and §13.3's alternative to the proxies, P2-F F9b;
// DESIGN §7.7's M_amb ambient directions; PHASE2 F108). The proxies (render/shade/ao.h) miss O-AO's bias gate on every
// EVIMO2 mover (p99 |dO| 0.12-0.29 against 0.03, F9b's study), so this form traces the occlusion instead:
//
//   each hit sub-sample k traces one ambient ray from its hit point along w_k, cosine-distributed about the hit
//   triangle's geometric normal (the side the camera sees), against the other movers only (static geometry occludes no
//   ambient, and a body none of its own, as under proxy), each mover up to twice its window start d0 (the proxy fit's,
//   render/ao_proxy.h: the same region in which proxy's AO is not exactly nothing); its blocked bit is bit
//   kAoTracedBit of the sub-sample's occlusion word. A cluster's ambient is A_g times its open share
//     1 - O_s = (n_s - b_s) / n_s,  n_s its members, b_s those whose ambient ray is blocked
//   (the pooled ratio of shade.h with uniform weights: DESIGN's "same conditional pooled ratio"), exactly 1 where no
//   member is blocked, so those clusters keep off's value bit for bit.
//
// Directions: w_k in the hit triangle's frame (Duff et al. 2017, "Building an Orthonormal Basis, Revisited", as
// pbrt-v4's CoordinateSystem) from point k of the pixel's 2-D Owen-scrambled Sobol sequence (core/rng/sobol.h; the
// scramble seed a hash of the pixel's sub-sample pattern seed, so directions and positions are independent), through
// Shirley and Chiu's concentric disk and Malley's lift (pbrt-v4 SampleCosineHemisphere; Cycles' AO node samples the
// same cosine hemisphere). The first 2^m points are a (0, m, 2)-net, so a pixel-sample of N = 2^m sub-samples has N
// stratified directions (M_amb = N: 64 at fidelity's N_base, 256 in its band) and the sets nest across N. A static
// receiver's directions are fixed in its geometry's frame: they change neither with time nor with the camera.
//
// Events: the bit is in the signature (shade.h shadow_term takes the whole word), so a flip is a J2 visibility event,
// localized like a shadow edge; a flip moves the cluster's ambient by A_g rho / n_s. F108's step rule bounds that step:
// ln(1 + a / N) <= theta_d / 4 at ambient share a <= 1 needs N >= n_s = ceil(1 / (exp(theta_d / 4) - 1)) (33 / 25 / 16
// at theta_d 0.12 / 0.16 / 0.25), which the engine checks at start-up. The class word counts the bit among the shadowed
// pairs but not among the shadow-edge bits (class_masks masks both of its halves to the shadow lights), so AO flips do
// not escalate the band. Held sets (ENGINE_CPP I3b) would merge still movers into the static world, where no AO ray
// looks: the engine holds none under traced, and Renderer::hold refuses a non-empty set.
//
// Measured (F9b's study, docs/design/phase2-checks/p2_f_ao_bias.py, floor receivers of the 24 EVIMO2 movers, these
// directions): the estimator's |dO| p99 against 4096 directions is 0.046-0.070 at 64 directions, 0.016-0.031 at 256
// and 0.006-0.013 at 1024; unbiased (mean signed error within 0.001).
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"
#include "core/pmath/trig.h"
#include "core/rng/sobol.h"
#include "render/ray.h"
#include "render/scene_view.h"

namespace balla::render {

// SceneView::mover_ao's values (render/scene_data.h MoverAo).
inline constexpr int32_t kMoverAoProxy = 1, kMoverAoTraced = 2;
// The occlusion word's AO bit (shadow bits are 0 .. num_shadow_lights - 1, at most 31 under traced).
inline constexpr uint32_t kAoTracedBit = 31, kAoTracedMask = 1u << kAoTracedBit;

// The direction-scramble seed of a pixel: a hash of its sub-sample pattern seed.
BALLA_HD inline uint32_t ao_direction_seed(uint32_t pattern_seed) {
  return rng::detail::hash_u32(pattern_seed ^ 0x414F5444u);
}

// Direction k of a pixel, cosine-distributed about +z: Sobol point k (scrambled by seed) through the concentric disk
// (Shirley and Chiu 1997) and Malley's lift.
BALLA_HD inline Vec3f ao_local_direction(uint32_t seed, uint32_t k) {
  uint32_t ux, uy;
  rng::owen_sobol_2d(k, seed, &ux, &uy);
  const float a = 2.0f * rng::to_unit_interval<float>(ux) - 1.0f;
  const float b = 2.0f * rng::to_unit_interval<float>(uy) - 1.0f;
  if (a == 0.0f && b == 0.0f) return {0.0f, 0.0f, 1.0f};
  constexpr float kPiOver4 = 0.785398163397448310f, kPiOver2 = 1.57079632679489662f;
  const bool wide = abs(a) > abs(b);
  const float r = wide ? a : b;
  const float phi = wide ? kPiOver4 * (b / a) : kPiOver2 - kPiOver4 * (a / b);
  float s, c;
  pmath::sincos(phi, &s, &c);
  const float x = r * c, y = r * s;
  return {x, y, sqrt(max(0.0f, 1.0f - x * x - y * y))};
}

// An orthonormal frame (t, b, n) about the unit n (Duff et al. 2017; pbrt-v4 CoordinateSystem).
BALLA_HD inline void ao_frame(Vec3f n, Vec3f* t, Vec3f* b) {
  const float sign = n.z >= 0.0f ? 1.0f : -1.0f;
  const float a = -1.0f / (sign + n.z);
  const float c = n.x * n.y * a;
  *t = {1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x};
  *b = {c, sign + n.y * n.y * a, -n.y};
}

// The ambient ray of one sub-sample hit, in the camera frame: from the offset hit point along the unit w_k.
struct AoRay {
  Vec3f origin_cam, dir_cam;
};

// The AO bit (kAoTracedMask or 0) of sub-sample k (the pixel-sample's global index, k_begin + k) of `pixel` whose hit
// is `hit` along d_cam; blocked(g_hit, ray) is the tracer's any-hit query against the movers other than g_hit, each up
// to its SceneView::ao_reach. A miss, or a hit on a body without trace geometry, has none.
template <class Blocked>
BALLA_HD uint32_t ao_traced_mask(const SceneView& s, const PoseTableView& poses, uint32_t slot, const CameraView& cam,
                                 uint32_t pixel, uint32_t k, const HitRecord& hit, Vec3f d_cam, Blocked&& blocked) {
  const BodyView* body = hit.instance != kInstanceNone ? find_body(s, hit.instance) : nullptr;
  if (body == nullptr || body->geometry < 0) return 0u;
  const PoseRow& row = pose_row(poses, slot, body->geometry);
  const Ray ray = primary_to_geometry(row, d_cam);
  const SurfaceHit sh = reintersect(fetch_triangle(s.geometries[body->geometry], hit.prim), ray);
  Vec3f n = normalize(sh.n);
  if (dot(n, ray.d) > 0.0f) n = -n;  // the side the camera sees
  Vec3f t, b;
  ao_frame(n, &t, &b);
  const Vec3f l = ao_local_direction(ao_direction_seed(cam.pattern_seeds[pixel]), k);
  const Vec3f w = l.x * t + l.y * b + l.z * n;
  const AoRay ar{point_to_camera(row, offset_origin(ray.o + sh.t * ray.d, n)), rotate_inverse(row, w)};
  return blocked(body->geometry, ar) ? kAoTracedMask : 0u;
}

// A cluster's open share under traced: its n members, b of them blocked.
BALLA_HD inline float ao_traced_visibility(uint32_t n, uint32_t b) {
  return b == 0 ? 1.0f : static_cast<float>(n - b) / static_cast<float>(n);
}

}  // namespace balla::render
