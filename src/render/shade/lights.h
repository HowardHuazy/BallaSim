// Analytic point, spot and directional lights (DESIGN §7.5-7.6, S2 §5.1), the sample points of area lights, and the
// shadow-ray set-up shared by every backend.
//
// Units are the relative ones of contract C1: point and spot intensity is W/sr-equivalent, a
// directional light's intensity is the irradiance on a surface facing it. The spot cone falls off
// with the C2 smootherstep between the outer and inner cone angles, so the spot edge is smooth
// signal, not a visibility jump (DESIGN §7.10). An area light (P2-F F8a-1) is a one-sided emitter of radiance L: its
// shadow rays go to one point of its fixed set per sub-sample, and the shading that uses them is F8a-2's.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/vec.h"
#include "core/pmath/smootherstep.h"
#include "core/rng/sobol.h"
#include "render/ray.h"
#include "render/scene_view.h"

namespace balla::render {

// Incident light at a camera-frame point x: unit direction towards the light and the irradiance
// it would deliver to a surface facing it (0 outside a spot cone).
struct Incidence {
  Vec3f omega;
  float e;
};

BALLA_HD inline Incidence incidence(const LightView& l, const LightRow& row, Vec3f x) {
  const Vec3f dir{row.direction[0], row.direction[1], row.direction[2]};
  if (l.kind == LightKind::kDirectional) return {-dir, l.intensity};
  const Vec3f to = Vec3f{row.position[0], row.position[1], row.position[2]} - x;
  const float d2 = dot(to, to);
  if (!(d2 > 0.0f)) return {{0.0f, 0.0f, 1.0f}, 0.0f};
  const Vec3f omega = (1.0f / sqrt(d2)) * to;
  float e = l.intensity / d2;
  if (l.kind == LightKind::kSpot) {
    const float c = -dot(omega, dir);
    e *= pmath::smootherstep((c - l.cos_outer) / (l.cos_inner - l.cos_outer));
  }
  return {omega, e};
}

// Lambert irradiance from light l at camera-frame point x for a (possibly unnormalized) normal n:
// max(0, n . F) with the vector irradiance F = e omega. Linear in n before the clip, so at one point
// the unnormalized mean normal of a cluster gives the footprint average while no normal faces away
// (S2 MAT-13).
BALLA_HD inline float lambert_irradiance(const LightView& l, const LightRow& row, Vec3f x, Vec3f n) {
  const Incidence in = incidence(l, row, x);
  return max(0.0f, in.e * dot(n, in.omega));
}

// Shadow ray of one sub-sample hit towards one light (DESIGN §7.6). The occlusion bit of a
// sub-sample is 1 when the light's contribution is blocked: by other geometry (traced), or by the
// surface itself when the light lies behind its geometric plane (no ray needed). Lights that
// deliver nothing to the point (outside a spot cone) get bit 0 and no ray.
enum class ShadowState : int32_t { kNoContribution = 0, kSelfOccluded = 1, kTrace = 2 };

struct ShadowRay {
  ShadowState state;
  Vec3f origin_hit;  // origin in the frame of the hit's own geometry (offset against self-hits)
  Vec3f origin_cam;  // the same origin in the camera frame, for every other geometry
  Vec3f dir_cam;     // towards the light; unnormalized for point/spot lights (light at t = 1)
  float tmax;
};

// d_cam: the primary ray's camera-frame direction; hit_ray: the same ray in the hit geometry's
// frame; sh: its re-intersection. The hit point is t d_cam in the camera frame, as in the cluster
// kernel, so both evaluate the light at the same point.
BALLA_HD inline ShadowRay shadow_ray(const LightView& l, const LightRow& lrow, const PoseRow& row,
                                     Vec3f d_cam, const Ray& hit_ray, const SurfaceHit& sh) {
  ShadowRay r;
  const Vec3f p_g = hit_ray.o + sh.t * hit_ray.d;
  const Incidence in = incidence(l, lrow, sh.t * d_cam);
  if (!(in.e > 0.0f)) {
    r.state = ShadowState::kNoContribution;
    return r;
  }
  Vec3f n_g = normalize(sh.n);
  if (dot(n_g, hit_ray.d) > 0.0f) n_g = -n_g;  // the side the camera sees
  if (!(dot(rotate(row, in.omega), n_g) > 0.0f)) {
    r.state = ShadowState::kSelfOccluded;
    return r;
  }
  r.state = ShadowState::kTrace;
  r.origin_hit = offset_origin(p_g, n_g);
  r.origin_cam = point_to_camera(row, r.origin_hit);
  if (l.kind == LightKind::kDirectional) {
    r.dir_cam = in.omega;
    r.tmax = infinity<float>();
  } else {
    r.dir_cam = Vec3f{lrow.position[0], lrow.position[1], lrow.position[2]} - r.origin_cam;
    r.tmax = 1.0f;
  }
  return r;
}

// The trace geometries a shadow ray is tested against (scans.md §4.2): kAll, every one (lit receivers; the only kind
// before baked proxies); kInserted, the geometries g >= 1 but the receiver's own (I, the inserted objects that cast
// proxy shadows on a baked receiver); kWorld, geometry 0 alone (W, the scan). The tracers' Blocked callables get it as
// a type, Which<...>, so that each sample pipeline compiles only its own loop (programs.cu's G6 rule).
enum class ShadowWhich : int32_t { kAll = 0, kInserted = 1, kWorld = 2 };
template <ShadowWhich W>
struct Which {
  static constexpr ShadowWhich value = W;
};

// Whether primitive prim of trace geometry g carries a baked material (scans.md §4.1). A baked receiver's radiance
// holds its capture's lighting: only proxy lights reach it (through their shadows, S3a), so the other shadow lights
// keep bit 0 there and trace no ray (§4.1 "Shadow rays"). False in scenes without baked materials, read-free.
BALLA_HD inline bool baked_receiver(const SceneView& s, const GeometryView& g, uint32_t prim) {
  return s.has_baked != 0 && s.materials[g.materials[g.single_material ? 0u : prim]].baked >= 0;
}

// Cell of the kLightGrid x kLightGrid grid over the unit square that holds the fixed-point pair (zx, zy): the top five
// bits of each coordinate, y selecting the row.
static_assert(kLightGrid == 32, "light_cell takes the top five bits of each coordinate");
BALLA_HD inline uint32_t light_cell(uint32_t zx, uint32_t zy) { return ((zy >> 27) << 5) | (zx >> 27); }

// The point of area light l that sub-sample k of the pixel aims at (plan p2f §4.6), in the camera frame: the padded
// pair z_k = owen_sobol_2d(k, seed of (camera, pixel, light)) picks the point of the light's set Y_l in its cell, and
// the point lies at c + a h_u u + b h_v v with the set's local (a, b), u the row's axis and v = u x n, so (u, v, -n) is
// right-handed as Blender's local frame of a light that emits along its -Z. The first N pairs fall in N distinct cells
// for every N <= kLightPoints: the pixel's sub-samples see N distinct points. k is the sub-sample's index in the
// pixel's sequence (SampleRequest::k_begin + its place in the request).
BALLA_HD inline Vec3f area_light_point(const SceneView& s, const LightView& l, const LightRow& row, uint32_t cam,
                                       uint32_t pixel, uint32_t k) {
  const uint32_t seed = s.cameras[cam].light_seeds[static_cast<uint64_t>(pixel) * static_cast<uint64_t>(s.num_area_lights) +
                                                   static_cast<uint64_t>(l.area_index)];
  uint32_t zx, zy;
  rng::owen_sobol_2d(k, seed, &zx, &zy);
  const float* ab = s.light_points + 2 * (static_cast<int64_t>(l.point_offset) + static_cast<int64_t>(light_cell(zx, zy)));
  const Vec3f u{row.axis_u[0], row.axis_u[1], row.axis_u[2]};
  const Vec3f n{row.direction[0], row.direction[1], row.direction[2]};
  const Vec3f v = cross(u, n);
  const Vec3f c{row.position[0], row.position[1], row.position[2]};
  return (c + (l.half_u * ab[0]) * u) + (l.half_v * ab[1]) * v;
}

// Shadow rays to an area light's points end at the point itself, a fraction 1e-4 short of it (the emitter is no
// geometry; a lamp body around it should not shadow the point it hides). Conventions (RENDER.md, "F8a-1: conventions"):
// pbrt-v4 shortens its shadow segments to 1 - ShadowEpsilon, ShadowEpsilon = 1e-4 (`Integrator::Unoccluded`); Mitsuba 3
// to 1 - ShadowEpsilon of `spawn_ray_to`, ShadowEpsilon = 8.9e-4 (v3.9.1, measured); a Cycles area lamp is no geometry,
// as here. The value is pbrt-v4's, a ninth of Mitsuba's.
inline constexpr float kAreaShadowTmax = 1.0f - 1e-4f;

// Shadow ray of one sub-sample hit towards the point y_cam of an area light (kRect, kDisk), the shadow_ray of this kind.
// The emitter is one-sided: a point that sees its back side delivers nothing (kNoContribution). Conventions (RENDER.md,
// "F8a-1: conventions"): a Cycles area lamp emits into the half space of its local -Z only (kernel/light/area.h; its
// `spread` defaults to pi, the whole half space, which is the test here), pbrt-v4's `diffuse` area light has `twosided`
// false and a Mitsuba 3 `area` emitter on a shape emits on the side of the shape's normal. Self-occlusion is no
// bit here (point lights keep theirs): a point behind the hit's geometric plane is kNoContribution too, since the
// analytic form factor clips the emitter at the cluster's plane and so keeps the surface's own horizon smooth and
// out of every signature (plan p2f §4.6); the three renderers have no such state either, the receiver's cosine factor
// is zero there. Otherwise the ray is traced to y_cam.
BALLA_HD inline ShadowRay area_shadow_ray(const LightRow& lrow, const PoseRow& row, Vec3f d_cam, const Ray& hit_ray,
                                          const SurfaceHit& sh, Vec3f y_cam) {
  ShadowRay r;
  const Vec3f p_g = hit_ray.o + sh.t * hit_ray.d;
  const Vec3f w = y_cam - sh.t * d_cam;  // from the hit to the point
  const Vec3f n_l{lrow.direction[0], lrow.direction[1], lrow.direction[2]};
  if (!(dot(n_l, w) < 0.0f)) {
    r.state = ShadowState::kNoContribution;
    return r;
  }
  Vec3f n_g = normalize(sh.n);
  if (dot(n_g, hit_ray.d) > 0.0f) n_g = -n_g;  // the side the camera sees
  if (!(dot(rotate(row, w), n_g) > 0.0f)) {
    r.state = ShadowState::kNoContribution;
    return r;
  }
  r.state = ShadowState::kTrace;
  r.origin_hit = offset_origin(p_g, n_g);
  r.origin_cam = point_to_camera(row, r.origin_hit);
  r.dir_cam = y_cam - r.origin_cam;
  r.tmax = kAreaShadowTmax;
  return r;
}

// Shadow mask of one sub-sample hit (bit LightView::shadow_bit set when that light is blocked),
// shared by the tracers; blocked(g_hit, ray, Which<w>{}) is the tracer's any-hit query along a shadow ray that
// leaves trace geometry g_hit, against the geometries of w. cam, pixel and k (the sub-sample's index in the pixel's
// sequence) name the sub-sample for the points of area lights, whose bit is set only by a blocker (area_shadow_ray).
//
// kBakedProxies false (every scene but the baked-proxy pipelines): a lit receiver's light is blocked by any geometry
// (Which<kAll>). A baked receiver sees its proxies only, by the same rule (U0's, scans.md §4.1: other lights trace no
// ray there).
//
// kBakedProxies true (SceneView::baked_proxies; the tracers' baked pipelines), lit receivers as above and their static
// word 0; a baked receiver follows the visibility rule of U2 (scans.md §4.2), for each proxy light l whose shadow ray
// state is kTrace (the other lights: bit 0 in both words and no ray; kNoContribution: 0 in both):
//   I_l = a geometry g >= 1 but the receiver's own blocks (Which<kInserted>: the inserted objects),
//   W_l = the scan (geometry 0) blocks (Which<kWorld>), or l is self-occluded (no ray);
// in each light group g with some I_l among its proxies P_g, W_l is evaluated for every proxy of P_g (a ray each, in
// light order) and the bits are
//   occlusion  b_l = I_l and not W_l   (the returned mask: the existing mask's bit),
//   static     s_l = W_l               (*static_out; 0 in groups without an inserted blocker).
// Two groups' proxies never mix, and the scan's own shadow (W) stays out of the occlusion bit: the texture already
// holds it. static_out (null: the word is dropped) receives s for every hit, 0 for lit receivers and misses. A proxy
// area light is traced to its centre, one ray as in the plan pass (shadow_ray), not to a sampled point (P2-F F8a-1).
template <bool kBakedProxies, class Blocked>
BALLA_HD uint32_t shadow_mask(const SceneView& s, const PoseTableView& poses, uint32_t slot, const HitRecord& hit,
                              Vec3f d_cam, uint32_t cam, uint32_t pixel, uint32_t k, Blocked&& blocked,
                              uint32_t* static_out = nullptr) {
  if constexpr (kBakedProxies) {
    if (static_out != nullptr) *static_out = 0u;
  }
  const BodyView* body = hit.instance != kInstanceNone ? find_body(s, hit.instance) : nullptr;
  if (body == nullptr) return 0;
  const PoseRow& row = pose_row(poses, slot, body->geometry);
  const Ray ray = primary_to_geometry(row, d_cam);
  const GeometryView& geo = s.geometries[body->geometry];
  const SurfaceHit sh = reintersect(fetch_triangle(geo, hit.prim), ray);
  const bool baked = baked_receiver(s, geo, hit.prim);
  if constexpr (kBakedProxies) {
    if (baked) {
      // Pass 1: the proxies' states and the inserted blockers I (shadow-bit masks).
      uint32_t inserted = 0, traced = 0, self = 0;
      for (int32_t j = 0; j < s.num_lights; ++j) {
        const LightView& l = s.lights[j];
        if (l.shadow_bit < 0 || l.proxy == 0) continue;
        const ShadowRay sr = shadow_ray(l, light_row(poses, slot, j), row, d_cam, ray, sh);
        const uint32_t bit = 1u << l.shadow_bit;
        if (sr.state == ShadowState::kSelfOccluded) {
          self |= bit;
        } else if (sr.state == ShadowState::kTrace) {
          traced |= bit;
          if (blocked(body->geometry, sr, Which<ShadowWhich::kInserted>{})) inserted |= bit;
        }
      }
      // Pass 2: the scan's W for every proxy of a group with an inserted blocker (the same rays as pass 1's).
      uint32_t mask = 0, statics = 0;
      if (inserted != 0) {
        for (int32_t j = 0; j < s.num_lights; ++j) {
          const LightView& l = s.lights[j];
          if (l.shadow_bit < 0 || l.proxy == 0 || (inserted & s.proxy_mask[l.group]) == 0) continue;
          const uint32_t bit = 1u << l.shadow_bit;
          if ((self & bit) != 0) {
            statics |= bit;
          } else if ((traced & bit) != 0) {
            const ShadowRay sr = shadow_ray(l, light_row(poses, slot, j), row, d_cam, ray, sh);
            if (blocked(body->geometry, sr, Which<ShadowWhich::kWorld>{})) {
              statics |= bit;
            } else if ((inserted & bit) != 0) {
              mask |= bit;
            }
          }
        }
      }
      if (static_out != nullptr) *static_out = statics;
      return mask;
    }
  }
  uint32_t mask = 0;
  for (int32_t j = 0; j < s.num_lights; ++j) {
    const LightView& l = s.lights[j];
    if (l.shadow_bit < 0 || (baked && l.proxy == 0)) continue;
    const LightRow& lrow = light_row(poses, slot, j);
    if (is_area(l.kind)) {
      const ShadowRay sr = area_shadow_ray(lrow, row, d_cam, ray, sh, area_light_point(s, l, lrow, cam, pixel, k));
      if (sr.state == ShadowState::kTrace && blocked(body->geometry, sr, Which<ShadowWhich::kAll>{})) {
        mask |= 1u << l.shadow_bit;
      }
      continue;
    }
    const ShadowRay sr = shadow_ray(l, lrow, row, d_cam, ray, sh);
    if (sr.state == ShadowState::kSelfOccluded ||
        (sr.state == ShadowState::kTrace && blocked(body->geometry, sr, Which<ShadowWhich::kAll>{}))) {
      mask |= 1u << l.shadow_bit;
    }
  }
  return mask;
}

// The plan pass's shadow states (ENGINE_CPP I8, the moving-shadow band): per shadow bit of a centre-ray hit, a state,
// kShadowLit (traced, unblocked), kShadowStatic (no hit, no contribution, self-occluded, or blocked by the static
// world) or kShadowMover + id (blocked by the moving body of instance id, the first in the shadow rays' order:
// movers by id, then the world), with kShadowMulti set where another moving body blocks the ray too (the I8 review:
// the band cannot see that body's edges inside the first one's shadow), and the first body's ray parameter t (its
// closest hit; point and spot lights: the fraction of the way to the light (an area light's: to its centre),
// directional: metres).
// blocker(g_hit, ray, &t, &multi, Which<w>{}) returns, among the trace geometries of w, the first that blocks (0 for the
// world, -1 when nothing does); *multi is set when a later moving (GeometryView::dynamic) geometry blocks the ray too.
// out: (state, t's bits) per shadow bit. A body whose id does not fit below kShadowMulti (ids from kShadowMover's
// complement on) is kShadowUnknown: the band takes its cone rate. A static body split from the world (GeometryView::
// dynamic 0, scans.md §4.2) blocks as the world does: kShadowStatic.
//
// Baked receivers (scans.md §4.2 "Plan pass"; kBakedProxies, the baked-proxy pipelines): the lights other than proxies
// trace no ray and stay kShadowStatic (shadow_mask's rule); a proxy whose ray is traced is
//   kShadowLit      when no inserted geometry blocks it (Which<kInserted>: the geometries g >= 1 but the receiver's own),
//   kShadowStatic   when one does and the scan blocks too (Which<kWorld>: the texture already holds that shadow),
//   otherwise the first inserted blocker's: kShadowMover + id for a moving body, kShadowStatic for a split static one.
// The static word is not read: the plan pass marks moving-shadow sources only. Lit receivers take kAll as before.
inline constexpr uint32_t kShadowLit = 0, kShadowStatic = 1, kShadowMover = 2, kShadowMulti = 1u << 31,
                          kShadowUnknown = kShadowMulti - 1;
template <bool kBakedProxies, class Blocker>
BALLA_HD void shadow_states(const SceneView& s, const PoseTableView& poses, uint32_t slot, const HitRecord& hit,
                            Vec3f d_cam, Blocker&& blocker, uint32_t* out) {
  for (int32_t b = 0; b < s.num_shadow_lights; ++b) {
    out[2 * b] = kShadowStatic;
    out[2 * b + 1] = 0;
  }
  const BodyView* body = hit.instance != kInstanceNone ? find_body(s, hit.instance) : nullptr;
  if (body == nullptr) return;
  const PoseRow& row = pose_row(poses, slot, body->geometry);
  const Ray ray = primary_to_geometry(row, d_cam);
  const GeometryView& geo = s.geometries[body->geometry];
  const SurfaceHit sh = reintersect(fetch_triangle(geo, hit.prim), ray);
  const bool baked = baked_receiver(s, geo, hit.prim);
  for (int32_t j = 0; j < s.num_lights; ++j) {
    const LightView& l = s.lights[j];
    if (l.shadow_bit < 0 || (baked && l.proxy == 0)) continue;
    const LightRow& lrow = light_row(poses, slot, j);
    // An area light is traced to its centre (plan p2f §4.6): one ray per bit, as for a point light.
    const ShadowRay sr = is_area(l.kind) ? area_shadow_ray(lrow, row, d_cam, ray, sh,
                                                           Vec3f{lrow.position[0], lrow.position[1], lrow.position[2]})
                                         : shadow_ray(l, lrow, row, d_cam, ray, sh);
    if (sr.state != ShadowState::kTrace) continue;
    float t = 0.0f;
    bool multi = false;
    int32_t g = -1;
    if constexpr (kBakedProxies) {
      if (baked) {
        g = blocker(body->geometry, sr, &t, &multi, Which<ShadowWhich::kInserted>{});
        if (g >= 0) {
          float t_world = 0.0f;
          bool multi_world = false;
          if (blocker(body->geometry, sr, &t_world, &multi_world, Which<ShadowWhich::kWorld>{}) == 0) {
            g = 0;
            t = t_world;
            multi = false;
          }
        }
      } else {
        g = blocker(body->geometry, sr, &t, &multi, Which<ShadowWhich::kAll>{});
      }
    } else {
      g = blocker(body->geometry, sr, &t, &multi, Which<ShadowWhich::kAll>{});
    }
    const uint32_t id = g > 0 ? s.geometries[g].instance : 0;
    const uint32_t mover = id < kShadowUnknown - kShadowMover ? kShadowMover + id : kShadowUnknown;
    out[2 * l.shadow_bit] = g < 0                                    ? kShadowLit
                            : g == 0 || s.geometries[g].dynamic == 0 ? kShadowStatic
                                                                     : mover | (multi ? kShadowMulti : 0u);
    out[2 * l.shadow_bit + 1] = float_to_bits(t);
  }
}

// The start diagnostic's trace (scans.md §4.2 "Start diagnostic"; Renderer::proxy_visibility): per shadow bit of a
// centre-ray hit, 0 where no proxy ray from a baked receiver is traced (a lit receiver, a light other than a proxy,
// a miss, a proxy behind the surface or outside its cone), 1 where the proxy's ray is traced and the scan (geometry 0
// alone, Which<kWorld>) does not block it, 2 where it does; the second word 0. blocked(g_hit, ray, Which<w>{}) is
// shadow_mask's callable.
template <class Blocked>
BALLA_HD void proxy_visibility_states(const SceneView& s, const PoseTableView& poses, uint32_t slot, const HitRecord& hit,
                                      Vec3f d_cam, Blocked&& blocked, uint32_t* out) {
  for (int32_t b = 0; b < s.num_shadow_lights; ++b) {
    out[2 * b] = 0;
    out[2 * b + 1] = 0;
  }
  const BodyView* body = hit.instance != kInstanceNone ? find_body(s, hit.instance) : nullptr;
  if (body == nullptr) return;
  const PoseRow& row = pose_row(poses, slot, body->geometry);
  const Ray ray = primary_to_geometry(row, d_cam);
  const GeometryView& geo = s.geometries[body->geometry];
  if (!baked_receiver(s, geo, hit.prim)) return;
  const SurfaceHit sh = reintersect(fetch_triangle(geo, hit.prim), ray);
  for (int32_t j = 0; j < s.num_lights; ++j) {
    const LightView& l = s.lights[j];
    if (l.shadow_bit < 0 || l.proxy == 0) continue;
    const ShadowRay sr = shadow_ray(l, light_row(poses, slot, j), row, d_cam, ray, sh);
    if (sr.state != ShadowState::kTrace) continue;
    out[2 * l.shadow_bit] = blocked(body->geometry, sr, Which<ShadowWhich::kWorld>{}) ? 2u : 1u;
  }
}

}  // namespace balla::render
