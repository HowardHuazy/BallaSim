// Mover AO by sphere-set proxies (accuracy.mover_ao proxy; plan p2f §4.7, P2-F F9a; PHASE2 F108; DESIGN §7.7): the
// ambient of every light group at a cluster is A_g (1 - O), O the cosine-weighted share of its hemisphere that the
// other moving bodies' proxies (render/ao_proxy.h) block:
//   1 - O = prod_{bodies b != own} prod_{spheres i of b} (1 - w_i F_i),
// with, for sphere i (centre c_i, radius R_i) seen from the cluster's mean point x and unit mean shading normal n,
// d = |c_i - x|, H = d / R_i and mu = cos(phi) = n . (c_i - x) / d:
//   contact floor  H_e = 1 + delta q((H - 1) / delta), delta = kAoContactDelta (bsdf.h floor_q: monotone, C2,
//                  H_e >= 1 + delta / 2 for every H, so a receiver inside a proxy sees contact, F(1.0005, 1) = 0.999)
//   mu >= 1/H_e    F = mu / H_e^2                                                      (the whole sphere in view)
//   |mu| < 1/H_e   F = 1/2 - asin(sqrt(H_e^2 - 1) / (H_e sin phi)) / pi
//                      + (mu acos(-sqrt(H_e^2 - 1) cot phi) - sqrt(H_e^2 - 1) sqrt(1 - H_e^2 mu^2)) / (pi H_e^2)
//   mu <= -1/H_e   F = 0                                                               (below the horizon)
// Howell's catalogue (differential element to a sphere, element tilted), checked against quadrature to 9.4e-8 (plan
// §1.2, f9_sphere_ff); C2 across the branches (the horizon-clipped area vanishes as h^(5/2)). The inverse trigonometry
// goes through pmath::atan2, so host and device agree bitwise, with the exact cosine sides of both angles: with
// x = sqrt(H_e^2 - 1) = sqrt(e (2 + e)) (e = H_e - 1) and r = sqrt(1 - H_e^2 mu^2) = sqrt((1 - H_e mu)(1 + H_e mu)),
// asin(x / (H_e sin phi)) = atan2(x, r) and acos(-x cot phi) = atan2(r, -x mu) (H_e^2 sin^2 phi - x^2 = r^2 =
// sin^2 phi - x^2 mu^2). No rounded ratio enters asin or acos where their slope is unbounded, so the forms keep their
// accuracy at contact and at both horizon joins (fp32 within 3e-7 of fp64 at every ulp near mu = +-1/H_e; a
// rounded-ratio form erred by 1.4e-4 there, the F9a review).
//   window         w_i = 1 - smootherstep((d - d0) / d0): 1 below d0 = R_b / sqrt(eps_AO), 0 from 2 d0 on, C2
// Per body the omitted occlusion is <= sum_i (R_i / d0)^2 = eps_AO (F <= (R / d)^2; measured worst 0.29 eps_AO, plan
// §1.2), eps_AO = 0.002 fixed (render/ao_proxy.h kAoEps: eps_L's family prior, not the run's eps_L), and beyond
// 2 d0 = 44.7 R_b a sphere's factor is exactly 1, so a receiver outside every mover's window keeps its value bit for
// bit. AO is C2 in (x, n) and enters no signature: no jumps, no step rule (§4.7). Bounce light off
// the movers stays a declared model error (DESIGN §7.7); static geometry occludes no ambient, as before.
//
// Conventions (p2f.md §3): Cycles' AO node (Blender 5.2.2, kernel/svm/ao.h svm_ao) traces num_samples rays per
// shading point, cosine-weighted (a uniform disk lifted to the hemisphere), up to its Distance (a hard cutoff), against
// every object or the object itself only (Only Local), and returns the unoccluded share; pbrt-v4's AOIntegrator traces
// cosine-weighted rays up to maxDistance (default infinite; from memory, not re-read); Mitsuba 3 (3.9.1, the installed
// plugins) has no AO integrator (its `aov` outputs geometric fields only). The share is the same cosine-weighted
// quantity; the proxies' analytic form (plan §14 departure 5) replaces the rays, the C2 window replaces Cycles' hard
// Distance (which would be a jump in time), and only other movers occlude (static geometry occludes no ambient, as
// before). The traced form (Embree, 4096 cosine directions) is O-AO's CPU reference for non-spherical movers (F9b).
//
// Evaluation: one sphere per lane, at most 32 per pass, the factors multiplied by the xor butterfly
// (butterfly_product, as lanes.h butterfly_sum), so the CPU (shade.h ao_visibility) and the warp kernel
// (device_executor.cu) agree bitwise.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"
#include "core/pmath/smootherstep.h"
#include "core/pmath/trig.h"
#include "render/lanes.h"
#include "render/ray.h"
#include "render/scene_view.h"
#include "render/shade/bsdf.h"

namespace balla::render {

// Maths-bounded constants (plan p2f §4.7, §4.11), documented and not knobs.
// The window's eps_AO (0.002, eps_L's family prior) and d0 are the fit's (render/ao_proxy.h kAoEps, AoProxy::d0).
inline constexpr double kAoContactDelta = 1e-3;  // delta of the contact floor

// The form factor F of a sphere from a differential element (module comment): e = H_e - 1 > 0 (the floored
// distance), mu the cosine between the element's normal and the sphere's centre. The clipped branch's rounding near
// the lower horizon (F -> 0) is clamped at 0.
template <class T>
BALLA_HD T sphere_form_factor(T e, T mu) {
  const T he = T(1) + e;
  const T he_mu = he * mu;
  if (he_mu >= T(1)) return mu / (he * he);
  if (he_mu <= T(-1)) return T(0);
  const T x = sqrt(e * (T(2) + e));                    // sqrt(H_e^2 - 1)
  const T r = sqrt((T(1) - he_mu) * (T(1) + he_mu));   // sqrt(1 - H_e^2 mu^2) > 0 here (|H_e mu| < 1)
  const T a = pmath::atan2(x, r);                      // asin(x / (H_e sin phi))
  const T c = pmath::atan2(r, -(x * mu));              // acos(-x cot phi)
  return max(T(0), (T(0.5) - a * bsdf::kInvPi<T>) + (mu * c - x * r) * bsdf::kInvPi<T> / (he * he));
}

// The floored H_e - 1 of a centre at distance d from a sphere of radius r: delta q((d - r) / (r delta)).
template <class T>
BALLA_HD T ao_contact(T d, T r) {
  return bsdf::smooth_floor((d - r) / r, T(kAoContactDelta));
}

// The window w(d) of a body whose window starts at d0.
template <class T>
BALLA_HD T ao_window(T d, T d0) {
  return T(1) - pmath::smootherstep((d - d0) / d0);
}

// 1 - w_i F_i of the scene's AO sphere i (SceneView::ao_spheres) at a cluster of body `own` with mean point x and unit
// normal n (camera frame), at pose slot `slot`; exactly 1 for an index past the spheres, a sphere of the cluster's own
// body, and a sphere at or beyond twice its body's d0.
BALLA_HD inline float ao_sphere_factor(const SceneView& s, const PoseTableView& poses, uint32_t slot, uint32_t own,
                                       Vec3f x, Vec3f n, int32_t i) {
  if (i >= s.num_ao_spheres) return 1.0f;
  const AoSphere sp = s.ao_spheres[i];
  const AoBodyView b = s.ao_bodies[sp.body];
  if (b.instance == own) return 1.0f;
  const Vec3f c = point_to_camera(pose_row(poses, slot, b.geometry), Vec3f{sp.centre[0], sp.centre[1], sp.centre[2]});
  const Vec3f to = c - x;
  const float d = sqrt(dot(to, to));
  if (!(d < 2.0f * b.d0)) return 1.0f;
  const float mu = d > 0.0f ? clamp(dot(n, to) / d, -1.0f, 1.0f) : 1.0f;
  return 1.0f - ao_window(d, b.d0) * sphere_form_factor(ao_contact(d, sp.radius), mu);
}

// Butterfly over 32 lane values (in place) with multiplication: the product every lane holds after the warp's
// __shfl_xor_sync rounds with offsets 16, 8, 4, 2, 1 (IEEE multiplication is commutative, so all lanes agree).
BALLA_HD inline float butterfly_product(float* v) {
  for (uint32_t off = kLanes / 2; off > 0; off /= 2) {
    float next[kLanes];
    for (uint32_t l = 0; l < kLanes; ++l) next[l] = v[l] * v[l ^ off];
    for (uint32_t l = 0; l < kLanes; ++l) v[l] = next[l];
  }
  return v[0];
}

// 1 - O at one cluster on one CPU thread, in the warp kernel's schedule: pass p takes spheres 32 p + lane, and its
// butterfly product multiplies the running value (from 1) in pass order. n is unit (shade.h ao_receiver).
inline float ao_visibility_host(const SceneView& s, const PoseTableView& poses, uint32_t slot, uint32_t own, Vec3f x,
                                Vec3f n) {
  float vis = 1.0f;
  for (int32_t base = 0; base < s.num_ao_spheres; base += static_cast<int32_t>(kLanes)) {
    float v[kLanes];
    for (uint32_t l = 0; l < kLanes; ++l) {
      v[l] = ao_sphere_factor(s, poses, slot, own, x, n, base + static_cast<int32_t>(l));
    }
    vis = vis * butterfly_product(v);
  }
  return vis;
}

}  // namespace balla::render
