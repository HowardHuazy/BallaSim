// The per-pixel part of the window planner (ENGINE_CPP §1.1, §1.3 K2; planner.py), shared by the
// host and the device: each pixel's image-speed rate k from the plan pass (static content under
// camera motion, moving shadows of static lights, moving lights), its reach in tiles, and the tile
// maxima that the host's tile stages take (motion.h).
//
// Arithmetic. The NumPy planner (deleted at I5) was the oracle through I4, and the frozen bench references
// hold its bits, so every fp64 expression here and in motion.cpp follows NumPy's evaluation on the
// reference build (NumPy 2.5 with OpenBLAS, x86-64, measured): 1-D dots, norms and transposed 3 x 3
// products sum left to right; einsum "ij,ij->i" sums (p0 + p2) + p1; (n, 3) @ (3, 3) and 3 x 3 @ 3 x 3
// accumulate with fused multiply-adds (dot_fma_a); (m, 3) @ (3,) runs OpenBLAS's dgemv_t kernel, whose
// rows of 4 use dot_fma_b and whose tail rows depend on m (gemv3). Square roots and divisions are
// correctly rounded on both sides, so the results equal NumPy's bit for bit; a BLAS with other kernels
// differs in the last ulp.
#pragma once

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"
#include "core/time/time.h"
#include "engine/par.h"
#include "engine/sampling.h"

namespace balla::engine {

using Vec3d = Vec3<double>;

// The planner's tile edge (px), exposed as _engine.TILE. The sub-sample reach around a pixel centre is the
// run's render::support_px (the single PSF source, P0_PARAMS §3.3), held by CameraGeometry.
inline constexpr int32_t kTile = 8;
// The window in seconds (planner.py W_S).
inline constexpr double kWindowS = static_cast<double>(time::kWindowNs) * 1e-9;

namespace np {

BALLA_HD inline double dot_seq(Vec3d a, Vec3d b) { return (a.x * b.x + a.y * b.y) + a.z * b.z; }
BALLA_HD inline double dot_ein(Vec3d a, Vec3d b) { return (a.x * b.x + a.z * b.z) + a.y * b.y; }
BALLA_HD inline double dot_fma_a(Vec3d a, Vec3d b) { return fma(a.z, b.z, fma(a.y, b.y, a.x * b.x)); }
BALLA_HD inline double dot_fma_b(Vec3d a, Vec3d b) { return fma(a.z, b.z, fma(a.x, b.x, a.y * b.y)); }
BALLA_HD inline double norm_seq(Vec3d a) { return sqrt(dot_seq(a, a)); }
// Row i of (m, 3) @ (3,).
BALLA_HD inline double gemv3(Vec3d a, Vec3d b, size_t i, size_t m) {
  if (m == 1) return dot_seq(a, b);
  const size_t q = m & ~size_t{3};
  return (i >= q && m - q >= 2 && i < q + 2) ? dot_fma_a(a, b) : dot_fma_b(a, b);
}
// np.maximum: NaN propagates.
BALLA_HD inline double maximum(double a, double b) { return a != a ? a : (b != b ? b : (a < b ? b : a)); }

}  // namespace np

// Row-major 3 x 3.
struct Mat3d {
  double m[3][3];
  BALLA_HD Vec3d row(int i) const { return {m[i][0], m[i][1], m[i][2]}; }
  BALLA_HD Vec3d col(int j) const { return {m[0][j], m[1][j], m[2][j]}; }
};

// A static shadow-casting light of the plan pass (planner.py _Receivers).
struct ShadowLight {
  uint32_t directional;  // 0: point or spot at `pos`; 1: directional, pos = -direction (towards the light)
  uint32_t first, count;  // its ShadowBody terms
  Vec3d pos;
  uint32_t bit;          // its shadow bit (the plan pass's shadow states, I8)
};

// One (static shadow light, moving body) pair: _Receivers.shadowed_by.
struct ShadowBody {
  enum Mode : uint32_t { kAll, kBeyond, kCone, kCylinder };
  uint32_t mode;
  double near;      // point: the light's distance to the body over the window
  double cos_half;  // kCone: receivers with dir . axis >= cos_half
  Vec3d axis;       // kCone: the cone's axis in the world
  Vec3d centre;     // kCylinder: swept sphere
  double radius, c_dot_l, c_dot_c;
  double speed;     // the body's point speed bound
  // The moving-shadow band (I8): whether the pair keeps the cone bound at every receiver (kKeep: (c) a spot light whose
  // outer cone the body may enter, or a light that may touch the body) or casts no shadow (kDead: a body outside a
  // spot light's outer cone throughout), and (d) its shadow's width bound: the body's thinnest part (the smallest
  // extent of its bounding box; under the band, of its mesh's thin parts, MeshDistance::thickness) and the light's
  // largest distance to the body.
  enum Keep : uint32_t { kBand, kKeep, kDead };
  uint32_t keep;
  double extent, far;
};

// A moving body of the band's sources (I8): its instance id, origin at mid-window (relative to origin_world) and speed
// bounds over the window.
struct MoverBody {
  uint32_t id;
  Vec3d centre;
  double v, omega, half_s;
};

// The plan pass's shadow states (render/shade/lights.h): blocked by a moving geometry from kStateMover on, by more
// than one where kStateMulti is set (the I8 review), by a body whose id does not fit at kStateUnknown.
inline constexpr uint32_t kStateMover = 2, kStateMulti = 1u << 31, kStateUnknown = kStateMulti - 1;
// Neighbouring centre rays whose depths differ by more than this fraction of the nearer are a depth discontinuity
// (the band's retained region (a)): an occlusion edge, a step, or a surface at grazing incidence (about
// 1 / (f tan(angle)) per pixel, so below about 2.3 degrees at f = 500 px).
inline constexpr double kDepthStep = 0.05;
// The band's region (d): a pair whose shadow may be narrower than this (px) keeps the cone bound.
inline constexpr double kNarrowPx = 2.0;

// A light on a moving carrier (moving_light_edge_speed).
struct MovingLight {
  enum Mode : uint32_t { kNone, kPoint, kDirectional };
  uint32_t mode;
  uint32_t first, count;  // kDirectional: body spheres
  Vec3d pos;
  double v, kappa, omega;
};

struct BodySphere {
  Vec3d centre;
  double swept_radius, point_speed;
};

// P2-F F9b (accuracy.mover_ao proxy or traced; plan p2f §4.7 "Planner"): a moving body's AO receivers over the window.
// Its AO is exactly nothing beyond twice its window start d0 from every proxy sphere (render/shade/ao.h; the traced
// form's rays stop at 2 d0, render/shade/ao_traced.h), and its spheres lie within its mesh radius of its origin, so a
// receiver x with |x - centre| > reach = 2 d0 (1 + 1e-6) + the body's swept radius sees none of it in the window. A
// receiver within reach is an AO receiver: its tile is not STATIC, at rate receiver_rate(p, r_min, speed), the body's
// point speed bound (the AO field moves with the occluder: magnification <= 1, plan §4.7).
struct AoMover {
  Vec3d centre;  // the body's origin at mid-window (relative to origin_world)
  double reach, speed;
};

// What the pixel stage of one window needs (motion.cpp Planner::begin).
struct PlanParams {
  double omega = 0, v = 0;  // camera window bounds
  double jmax = 0;          // CameraGeometry::speed_factor_max
  double support = 0;       // CameraGeometry::support_px
  int32_t big = 0;          // max(tiles_x, tiles_y)
  uint32_t lights = 0;      // shadow or moving lights: receivers are needed
  uint32_t n_shadow = 0, n_moving = 0;
  Mat3d r_wc{};             // camera pose at t_w
  Vec3d c_c{};
  // I8: the moving-shadow band (needs the shadow states, n_bits lights per pixel, and the instances), with the
  // n_movers moving bodies; it reads the image's neighbours.
  uint32_t band = 0, n_bits = 0, n_movers = 0;
  int32_t width = 0, height = 0;
  double jmin = 0;          // CameraGeometry::speed_factor_min (the band's region (d))
  // P2-B B4a: the band N's seeds (PixelPlan::seed; PHASE2 D-P2-4), written beside the rates and never into them, so the
  // tile levels do not change. They read the instances, the shadow states of all n_bits and the n_movers moving
  // bodies; moving_bits marks the shadow bits of the moving lights. The crop [x0, x1) x [y0, y1): the pixels whose
  // carried samples are current (those outside keep the start frame's) and whose seeds are counted.
  uint32_t seeds = 0, moving_bits = 0;
  int32_t crop_x0 = 0, crop_y0 = 0, crop_x1 = 0x7fffffff, crop_y1 = 0x7fffffff;
  // P2-B B6c (accuracy.band_growth pixel, with the seeds): each seed's motion in px (PixelPlan::seed_m), its static
  // (camera) part scaled by sigma_cam and its mover part by sigma_mov (accuracy.band_reach), and its kind (band_map.h).
  uint32_t pixel = 0;
  double sigma_cam = 1.0, sigma_mov = 1.0;
  // P2-D D1 (F021): obliquity (accuracy.shadow_obliquity projected) scales the shadow-edge rates by the receiver's
  // obliquity factors (obliquity()), moving receivers composed, and takes the band's region (d) foreshortening from
  // the normal. bits: the pixel stage reads the plan pass's normals (PixelPlan::normal) and writes P&V's plan bits
  // (pv_plan) without moving a rate, a level or a band decision (the engine in every window with moving-shadow pairs
  // or moving lights, so the as-built rate's non-bounds are counted; obliquity and pv imply it). pv adds R4's
  // thin-moving-shadow width (PixelPlan::pv_width) to the plan bits, at the sub-sample pitch (px).
  uint32_t obliquity = 0, bits = 0, pv = 0;
  double pitch = 0;
  // P2-F F9b: the n_ao moving bodies whose AO receivers the pixel stage marks (PixelPlan::ao; kPlanAo).
  uint32_t n_ao = 0;
};

// Pixel i (row-major over p.width) inside p's crop.
BALLA_HD inline bool in_crop(const PlanParams& p, size_t i) {
  const auto x = static_cast<int32_t>(i % static_cast<size_t>(p.width));
  const auto y = static_cast<int32_t>(i / static_cast<size_t>(p.width));
  return x >= p.crop_x0 && x < p.crop_x1 && y >= p.crop_y0 && y < p.crop_y1;
}

// The reach (tiles) of content at `rate` (rad/s) over a window, with the sub-sample support on both sides, at most
// the grid (PixelPlan::reach_of; Planner::finish's carried seeds).
BALLA_HD inline int32_t reach_tiles(const PlanParams& p, double rate) {
  const double r = floor(((p.jmax * rate) * kWindowS + 2.0 * p.support + (kTile - 1)) / kTile);
  return isfinite(r) ? static_cast<int32_t>(r < p.big ? r : p.big) : p.big;
}

// The receiver test of one static light (light, receiver x = row i of n).
struct Receiver {
  double dist, along, x2;
  Vec3d dir;
};

BALLA_HD inline Receiver make_receiver(const ShadowLight& l, Vec3d x, size_t i, size_t n) {
  Receiver r{};
  if (l.directional != 0) {
    r.along = np::gemv3(x, l.pos, i, n);
    r.x2 = np::dot_ein(x, x);
  } else {
    const Vec3d rel = x - l.pos;
    r.dist = sqrt(np::dot_ein(rel, rel));
    r.dir = {rel.x / r.dist, rel.y / r.dist, rel.z / r.dist};
  }
  return r;
}

// Whether the receiver's path to the light may meet the body during the window.
BALLA_HD inline bool shadowed(const ShadowBody& b, const Receiver& r, Vec3d x, size_t i, size_t n) {
  switch (b.mode) {
    case ShadowBody::kAll:
      return true;
    case ShadowBody::kCylinder: {
      const double s = b.c_dot_l - r.along;
      const double w2 = (r.x2 - 2.0 * np::gemv3(x, b.centre, i, n)) + b.c_dot_c;
      return (s > -b.radius) && (w2 - s * s <= b.radius * b.radius);
    }
    default:
      return r.dist >= b.near && (b.mode == ShadowBody::kBeyond || np::gemv3(r.dir, b.axis, i, n) >= b.cos_half);
  }
}

// lambda: the receiver's distance to the light over the light's distance to the body (1 for
// directional lights, inf for a light that may touch the body).
BALLA_HD inline double shadow_lambda(const ShadowBody& b, const Receiver& r) {
  return b.mode == ShadowBody::kAll ? infinity<double>() : (b.mode == ShadowBody::kCylinder ? 1.0 : r.dist / b.near);
}

// Bound on the speed of the edges a moving light sweeps over the receiver x (0 where it makes none).
BALLA_HD inline double moving_edge(const MovingLight& l, const BodySphere* spheres, Vec3d x) {
  if (l.mode == MovingLight::kPoint) {
    return l.v + l.kappa * (np::norm_seq(x - l.pos) + l.v * kWindowS / 2.0);
  }
  double speed = 0.0;
  if (l.mode == MovingLight::kDirectional) {
    for (uint32_t b = l.first; b < l.first + l.count; ++b) {
      const double s = np::norm_seq(x - spheres[b].centre) + spheres[b].swept_radius;
      speed = np::maximum(speed, spheres[b].point_speed + l.omega * s);
    }
  }
  return speed;
}

// Rate k of a feature moving at edge (m/s) over a receiver at range >= r_min (inf where r_min <= 0).
BALLA_HD inline double receiver_rate(const PlanParams& p, double r_min, double edge) {
  return r_min > 0 ? p.omega + (p.v + edge) / r_min : infinity<double>();
}

// The band's region (d): the pair's shadow at the receiver (range rng, foreshortened by cos_view) may be narrower than
// kNarrowPx, judged by the body's smallest thickness (ShadowBody::extent: its bounding box's smallest extent or, under
// the band, the mesh's thinnest part; the I8 review) times the shadow's smallest magnification, at the image's
// smallest scale.
BALLA_HD inline bool narrow_shadow(const PlanParams& p, const ShadowBody& b, const Receiver& r, double rng,
                                   double cos_view) {
  const double lambda = b.mode == ShadowBody::kCylinder ? 1.0 : r.dist / b.far;
  return b.extent * lambda * p.jmin * cos_view < kNarrowPx * rng;
}

// Per-pixel flags of the pixel stage: a hit, a moving-shadow rate (the cone bound, or the band's), and a plan-pass
// discontinuity with a 4-neighbour (I8: hit, instance or depth; computed with the band); P2-F F9b: an AO receiver.
inline constexpr uint8_t kPlanHit = 1, kPlanShadow = 2, kPlanDisc = 4, kPlanAo = 8;

// P2-D D1 (F021; DESIGN §6.7): the obliquity factors of a shadow edge over a receiver plane. The shadow X = L + lambda
// (P - L) of an occluder point P on the plane through X with unit normal n moves at
//   dX/dt = w_perp - d [n . w_perp - n . dQ/dt] / (n . d),  w = lambda dP/dt + (1 - lambda) dL/dt,
// d the unit direction from the light to X, w_perp its part normal to d and dQ/dt the plane's material velocity at X;
// the camera sees P_r dX/dt (P_r = I - r r^T, r the unit view ray), so |P_r dX/dt| <= g |w_perp| + h |n . dQ/dt| with
//   g = sigma_max(P_r (I - d n^T / (n . d)) on d_perp) <= 1 / cos psi,  h = |P_r d| / cos psi,  cos psi = |n . d|.
// g < 1 where the view is grazing too. The cap (g, h <= cap, which cos psi >= 1 / cap ensures) is a design choice, not
// a maths bound: beyond it the factors take the cap and `beyond` is set (P&V's R6); a normal that is not finite is
// beyond it. The bound holds for receivers planar at the normal's scale (the hit triangle's: a facet of a scan).
// g_beyond: g alone is beyond the cap (h enters only on a moving receiver; D1f: a static receiver is beyond only then).
//
// P2-D D1f, a self-shadow (P on the receiver's own body B: occluder and receiver share one rigid motion): in B's frame
// P and the plane are still and the light moves at dL/dt - dQ_B(L)/dt, so w = (1 - lambda) (dL/dt - dQ_B(L)/dt) there
// and |P_r dX/dt| <= |dQ_B(X)/dt| + g (lambda - 1) |dL/dt - dQ_B(L)/dt|: the receiver's own speed s_r plus g times the
// edge's motion over the body. With a static light that is s_r + g (lambda - 1) rigid_speed(B, L) (point lights, lambda -
// 1 = |X - P| / |P - L|), s_r + g |X - P| omega_B with |X - P| <= 2 R_B (directional: the direction turns at omega_B in
// B's frame). The world-frame form, g lambda s_P + h s_r, counts B's motion twice and multiplies it by the grazing
// factors; both are bounds, so the pixel stage takes the lesser (self_edge).
inline constexpr double kObliquityCap = 8.0;
// P&V's roughness (R4's relief margin): a receiver whose geometric normal is more than 10 degrees from a 4-neighbour's
// on the same instance; such pixels take the margin kRoughMargin.
inline constexpr double kRoughCos = 0.98480775301220806, kRoughMargin = 4.0;
// The plan bits (PixelPlan::pv_plan): bits 0-3 q = ceil(2 max(1, g_max)) - 1, g_max the largest factor of the pixel's
// moving-shadow pairs and moving lights (g; h too on a moving receiver), so (q + 1) / 2 >= g_max; R6 (a pair's
// receiver beyond the cap), rough, R4 (a moving shadow narrower than 2 m D_bound + pitch; PvPlanPass), and cone (the
// receiver lies in a moving-shadow cone or under a moving light's edges).
inline constexpr uint8_t kPvGain = 15, kPvBeyond = 16, kPvRough = 32, kPvThin = 64, kPvCone = 128;

struct Obliquity {
  double g, h;
  bool beyond, g_beyond;
};

BALLA_HD inline Obliquity obliquity(Vec3d n, Vec3d r, Vec3d d, double cap) {
  const double nd = dot(n, d), cos_psi = fabs(nd);
  if (!(cos_psi > 0) || !isfinite(cos_psi + dot(r, r))) return {cap, cap, true, true};
  // e1, e2: an orthonormal basis of d_perp (e1 normal to d and to the axis of d's smallest component).
  const double ax = fabs(d.x), ay = fabs(d.y), az = fabs(d.z);
  const Vec3d axis = ax <= ay && ax <= az ? Vec3d{1, 0, 0} : (ay <= az ? Vec3d{0, 1, 0} : Vec3d{0, 0, 1});
  Vec3d e1 = cross(d, axis);
  e1 = (1.0 / norm(e1)) * e1;
  const Vec3d e2 = cross(d, e1);
  const auto view = [&](Vec3d v) { return v - dot(r, v) * r; };
  const Vec3d a1 = view(e1 - (dot(n, e1) / nd) * d), a2 = view(e2 - (dot(n, e2) / nd) * d);
  const double g11 = dot(a1, a1), g22 = dot(a2, a2), g12 = dot(a1, a2);
  const double g = sqrt(0.5 * (g11 + g22) + sqrt(0.25 * (g11 - g22) * (g11 - g22) + g12 * g12));
  const double h = norm(view(d)) / cos_psi;
  return {g < cap ? g : cap, h < cap ? h : cap, !(g <= cap && h <= cap), !(g <= cap)};
}

// P2-B B6c, the seed kinds (PixelPlan::seed_kind; band_map.h): a plan seed (a plan-pass discontinuity, a shadow bit
// that differs from a neighbour's, I8's retained regions) and a carried seed (a carried sample with edge bits).
inline constexpr uint8_t kSeedPlan = 1, kSeedCarried = 2;

// The rigid-body speed bound of a moving body's point at x (relative to origin_world) over the window, with the slack
// for the points within that speed times W of x (I8: the band's shadow edges).
BALLA_HD inline double rigid_speed(const MoverBody& m, Vec3d x) {
  const double rho = np::norm_seq(x - m.centre) + m.v * m.half_s;
  return (m.v + m.omega * rho) * (1.0 + m.omega * kWindowS);
}

// K2: one pixel's rate k (rad/s), reach (tiles) and flags, from its plan-pass depth (camera z, NaN
// on a miss; float, as the renderer's ground truth returns it).
struct PixelPlan {
  PlanParams p;
  const float* depth;
  size_t depth_stride;  // in floats; also the stride of instance
  const double* rays;   // n x 3 unit rays
  size_t n;
  const ShadowLight* shadow_lights;
  const ShadowBody* shadow_bodies;
  const MovingLight* moving;
  const BodySphere* spheres;
  double* k;
  int32_t* reach;
  uint8_t* flags;
  // I8 (null when off): the plan pass's instances (GtSample::instance, as uint32 at depth's stride), its shadow states
  // ((state, blocker t's bits) at state[2 (i * p.n_bits + bit)]; render/shade/lights.h), and the moving bodies. The
  // band's outputs: the cone bound's rate of the pairs it may drop (0 where none), and a band source's rate (0 where
  // none) and reach (tiles; -1 where none).
  const uint32_t* instance = nullptr;
  const uint32_t* state = nullptr;
  const MoverBody* movers = nullptr;
  double* kcone = nullptr;
  double* ksource = nullptr;
  int32_t* sreach = nullptr;
  // P2-B B4a (p.seeds): each pixel's seed reach in tiles (-1: not a seed of the band N), and the class words of the
  // carried samples by pixel id (null: none), whose edge bits (kBandEdge) make a pixel a seed too.
  int32_t* seed = nullptr;
  const uint32_t* carried = nullptr;
  // P2-B B6c (p.pixel): each pixel's seed motion M in px, max(sigma_cam M_static, sigma_mov M_mov) (0 where it is no
  // seed), and its kind (band_map.h kSeedPlan, kSeedCarried; 0: no seed).
  double* seed_m = nullptr;
  uint8_t* seed_kind = nullptr;
  // P2-D D1 (null when p.bits is 0): the plan pass's geometric normals (3 floats per pixel, camera
  // frame, facing the camera; NaN on a miss), and the outputs: the plan bits (kPv*), under p.obliquity the rate with
  // g = h = 1 (k_base: the tiles whose level the factors move, Planner::finish), under p.pv R4's moving-shadow width
  // (px; inf where none).
  const float* normal = nullptr;
  uint8_t* pv_plan = nullptr;
  double* k_base = nullptr;
  float* pv_width = nullptr;
  // P2-F F9b (p.n_ao > 0): the moving bodies whose AO receivers this stage marks.
  const AoMover* ao = nullptr;
  // P2-D D1f (null unless p.obliquity with the band): the band's outputs as built (kcone, ksource, sreach with g = h =
  // 1 and region (d) at the as-built foreshortening), so that Planner::finish's levels without the factors are the
  // as-built ones under the band too.
  double* kcone0 = nullptr;
  double* ksource0 = nullptr;
  int32_t* sreach0 = nullptr;

  BALLA_HD int32_t reach_of(double rate) const { return reach_tiles(p, rate); }
  BALLA_HD bool hit_at(size_t i) const { return isfinite(depth[i * depth_stride]); }
  // A hit/miss, instance or depth discontinuity between the centre rays of pixels i and j.
  BALLA_HD bool discontinuous(size_t i, size_t j) const {
    const bool a = hit_at(i), b = hit_at(j);
    if (a != b) return true;
    if (!a) return false;
    if (instance[i * depth_stride] != instance[j * depth_stride]) return true;
    const double zi = depth[i * depth_stride], zj = depth[j * depth_stride];
    return fabs(zi - zj) > kDepthStep * (zi < zj ? zi : zj);
  }
  BALLA_HD uint32_t state_of(size_t i, uint32_t l) const { return state[2 * (i * p.n_bits + shadow_lights[l].bit)]; }
  BALLA_HD const MoverBody* mover(uint32_t id) const {
    for (uint32_t m = 0; m < p.n_movers; ++m) {
      if (movers[m].id == id) return &movers[m];
    }
    return nullptr;
  }
  // P2-D D1: a receiver's world normal nw and view ray rw, its own mover's speed bound s_r (0 when static), whether
  // the rates take the factors (on); D1f: its own mover (null when static; movers[m] is the pair term sl.first + m of
  // every static light sl, Planner::terms).
  struct Oblique {
    bool on;
    Vec3d nw, rw;
    double s_r;
    const MoverBody* own;
  };
  // The factors of static light l at the receiver r (d: from the light to the receiver; directional: -sl.pos).
  BALLA_HD static Obliquity light_obliquity(const ShadowLight& sl, const Receiver& r, const Oblique& o) {
    const Vec3d d = sl.directional != 0 ? (-1.0 / norm(sl.pos)) * sl.pos : r.dir;
    return obliquity(o.nw, o.rw, d, kObliquityCap);
  }
  // An edge speed (m/s) with the factors: g edge + h s_r (as built: edge).
  BALLA_HD static double oblique(const Oblique* o, const Obliquity& ob, double edge) {
    return o != nullptr && o->on ? ob.g * edge + ob.h * o->s_r : edge;
  }
  // P2-D D1f: a self-shadow's edge speed with the factors (obliquity's comment): the lesser of the body frame's, s_r + g
  // rel (rel: the edge's speed over the body before the factors, inf where unbounded), and the world frame's, which
  // is no bound where h alone is beyond the cap; as built (factors off), edge.
  BALLA_HD static double self_edge(const Oblique* o, const Obliquity& ob, double edge, double rel) {
    if (o == nullptr || !o->on) return edge;
    const double body = isfinite(rel) ? o->s_r + ob.g * rel : infinity<double>();
    return ob.beyond && !ob.g_beyond ? body : fmin(body, ob.g * edge + ob.h * o->s_r);
  }
  // The edge's speed over the body B of a self-shadow (self_edge), at distance |X - P| = blocker t times |to_light| (or,
  // for a pair, lambda - 1 at most lambda_max - 1, |X - P| at most 2 R_B), for static light sl.
  BALLA_HD static double self_rel(const ShadowLight& sl, const MoverBody& m, double lambda, double dist_xp) {
    return sl.directional != 0 ? dist_xp * m.omega : (lambda - 1.0) * rigid_speed(m, sl.pos);
  }

  BALLA_HD void operator()(size_t i) const {
    const Vec3d ray{rays[3 * i], rays[3 * i + 1], rays[3 * i + 2]};
    const double rng = static_cast<double>(depth[i * depth_stride]) / ray.z;
    const bool hit = isfinite(rng);
    const double r_min = hit ? rng - p.v * kWindowS : infinity<double>();
    const double k_static = hit ? (r_min > 0 ? p.omega + p.v / r_min : infinity<double>()) : p.omega;
    double kk = k_static;
    double kk0 = k_static;  // P2-D D1: kk with g = h = 1 (k_base)
    uint8_t f = hit ? kPlanHit : 0;
    // I8: plan-pass discontinuities with the 4-neighbours, border tiles, and per static light whether the pixel lies
    // in a mover's shadow at a boundary of it (some neighbour's state differs); and the receiver's foreshortening for
    // region (d) (the I8 review): |cos| of the angle between the centre ray and the plan-pass surface's normal, from
    // the camera-frame points of a continuous neighbour on each axis (1 where an axis has none: a discontinuity, which
    // is a source at its cone rate). P2-B B4a's seeds take the same discontinuities and, per shadow bit, whether its
    // state differs from a 4-neighbour's (lit, static or a mover, of any light).
    bool disc = false, border = false;
    uint32_t boundary = 0;  // bit l: static light l
    uint32_t differs = 0;   // bit b: shadow bit b (the seeds)
    double cos_view = 1.0;
    if (p.band != 0 || p.seeds != 0) {
      const auto w = static_cast<size_t>(p.width), h = static_cast<size_t>(p.height);
      const size_t px = i % w, py = i / w;
      const size_t nb[4] = {px > 0 ? i - 1 : i, px + 1 < w ? i + 1 : i, py > 0 ? i - w : i, py + 1 < h ? i + w : i};
      Vec3d tangent[2] = {};
      bool have[2] = {false, false};
      for (size_t a = 0; a < 4; ++a) {
        const size_t j = nb[a];
        if (j == i) continue;
        const bool dj = discontinuous(i, j);
        disc = disc || dj;
        for (uint32_t b = 0; p.seeds != 0 && b < p.n_bits; ++b) {
          differs |= state[2 * (i * p.n_bits + b)] != state[2 * (j * p.n_bits + b)] ? 1u << b : 0u;
        }
        if (p.band == 0) continue;
        if (!dj && hit && !have[a / 2]) {
          const Vec3d rj{rays[3 * j], rays[3 * j + 1], rays[3 * j + 2]};
          const double sj = static_cast<double>(depth[j * depth_stride]) / rj.z;
          tangent[a / 2] = Vec3d{rj.x * sj, rj.y * sj, rj.z * sj} - Vec3d{ray.x * rng, ray.y * rng, ray.z * rng};
          have[a / 2] = true;
        }
        for (uint32_t l = 0; l < p.n_shadow; ++l) {
          boundary |= state_of(i, l) >= kStateMover && state_of(i, l) != state_of(j, l) ? 1u << l : 0u;
        }
      }
      if (p.band != 0) {
        if (have[0] && have[1]) {
          const Vec3d nrm = cross(tangent[0], tangent[1]);
          const double len = np::norm_seq(nrm);
          if (len > 0) cos_view = fabs(np::dot_seq(nrm, ray)) / len;
        }
        const size_t tx = px / kTile, ty = py / kTile;
        border = tx == 0 || ty == 0 || tx + 1 == (w + kTile - 1) / kTile || ty + 1 == (h + kTile - 1) / kTile;
        f |= disc ? kPlanDisc : 0;
      }
    }
    // The seeds' rates beside the band's: the largest rate of the (static light, moving body) pairs that may shadow the
    // receiver (k_pairs) and of the moving lights' edges over it (k_move).
    double k_src = 0.0, k_pairs = 0.0, k_move = 0.0;
    Vec3d x{};
    if (p.lights != 0 || p.seeds != 0 || p.n_ao != 0) {
      const double s = hit ? rng : 0.0;
      x = Vec3d{p.c_c.x + np::dot_fma_a(ray, p.r_wc.row(0)) * s, p.c_c.y + np::dot_fma_a(ray, p.r_wc.row(1)) * s,
                p.c_c.z + np::dot_fma_a(ray, p.r_wc.row(2)) * s};
    }
    // P2-D D1 (F021): with the plan pass's normals, the receiver's foreshortening from its normal (cos_n: R4's widths;
    // the band's region (d) takes it only under the factors, so the plan bits alone leave the band's decisions as
    // built), its world normal and view ray, its own mover's speed bound (a moving receiver's term, composed), and for
    // the plan bits the largest factor of its moving-shadow pairs and moving lights, whether one is beyond the cap, and
    // the narrowest shadow.
    Oblique ob_rx{p.obliquity != 0, {}, {}, 0.0, nullptr};
    const double cos_built = cos_view;  // D1f: region (d)'s foreshortening as built (k_base, kcone0)
    double g_max = 1.0, width = infinity<double>(), cos_n = cos_view;
    bool cone = false, beyond = false;
    if (normal != nullptr && hit) {
      const Vec3d nc{normal[3 * i], normal[3 * i + 1], normal[3 * i + 2]};
      if (isfinite(dot(nc, nc))) cos_n = fabs(dot(nc, ray));
      if (p.obliquity != 0) cos_view = cos_n;
      const Vec3d nw{dot(nc, p.r_wc.row(0)), dot(nc, p.r_wc.row(1)), dot(nc, p.r_wc.row(2))};
      ob_rx.nw = (1.0 / norm(nw)) * nw;  // unit in fp64 (the renderer's is fp32)
      ob_rx.rw = Vec3d{dot(ray, p.r_wc.row(0)), dot(ray, p.r_wc.row(1)), dot(ray, p.r_wc.row(2))};
      ob_rx.own = instance != nullptr ? mover(instance[i * depth_stride]) : nullptr;
      if (ob_rx.own != nullptr) ob_rx.s_r = rigid_speed(*ob_rx.own, x);
    }
    const Oblique* obp = normal != nullptr ? &ob_rx : nullptr;
    // D1f: h enters only on a moving receiver, and not a self-shadow's body-frame bound: elsewhere g alone decides R6.
    const auto note = [&](const Obliquity& o, bool self) {
      cone = true;
      beyond = beyond || (ob_rx.s_r > 0 && !self ? o.beyond : o.g_beyond);
      g_max = np::maximum(g_max, ob_rx.s_r > 0 && o.h > o.g ? o.h : o.g);
    };
    if (p.lights != 0) {
      // The band: the cone rate of the pairs it drops here (k_cone) and of the pairs it may drop at some receiver
      // (k_any: k_cone and the pairs region (d) keeps here only), a source's edge rate, and whether the pixel is a
      // source at the cone rate (cone_src).
      double k_cone = 0.0, k_any = 0.0, k_cone0 = 0.0;
      double k_any0 = 0.0, k_src0 = 0.0;  // D1f: k_any and k_src as built (kcone0, ksource0)
      bool cone_src = disc;
      for (uint32_t l = 0; l < p.n_shadow; ++l) {
        const ShadowLight& sl = shadow_lights[l];
        const Receiver r = make_receiver(sl, x, i, n);
        const Obliquity ob =
            obp != nullptr && hit ? light_obliquity(sl, r, ob_rx) : Obliquity{1.0, 1.0, false, false};
        for (uint32_t b = sl.first; b < sl.first + sl.count; ++b) {
          if (!(hit && shadowed(shadow_bodies[b], r, x, i, n))) continue;
          if (p.band != 0 && shadow_bodies[b].keep == ShadowBody::kDead) continue;
          const ShadowBody& sb = shadow_bodies[b];
          const double lambda = shadow_lambda(sb, r), edge = lambda * sb.speed;
          // D1f: the pair of the receiver's own body (a self-shadow) in the body's frame.
          const bool self = ob_rx.own != nullptr && static_cast<uint32_t>(ob_rx.own - movers) == b - sl.first;
          const double rate = receiver_rate(
              p, r_min, self ? self_edge(obp, ob, edge, self_rel(sl, *ob_rx.own, lambda, 2.0 * sb.radius))
                             : oblique(obp, ob, edge));
          const double rate0 = p.obliquity != 0 ? receiver_rate(p, r_min, edge) : rate;
          if (obp != nullptr) {
            note(ob, self);
            if (p.pv != 0) width = fmin(width, shadow_width(p, shadow_bodies[b], r, rng, cos_n));
          }
          k_pairs = np::maximum(k_pairs, rate);
          const bool keep_all = p.band == 0 || shadow_bodies[b].keep == ShadowBody::kKeep;
          if (!keep_all) {
            k_any = np::maximum(k_any, rate);
            k_any0 = np::maximum(k_any0, rate0);
          }
          const bool keep = keep_all || narrow_shadow(p, shadow_bodies[b], r, rng, cos_view);
          if (keep) {
            kk = np::maximum(kk, rate);
            f |= kPlanShadow;
          } else {
            k_cone = np::maximum(k_cone, rate);
          }
          // D1f: the as-built decision (region (d) at the as-built foreshortening; the factors' cos_n may differ).
          const bool keep0 =
              keep_all || (p.obliquity != 0 ? narrow_shadow(p, shadow_bodies[b], r, rng, cos_built) : keep);
          if (keep0) {
            kk0 = np::maximum(kk0, rate0);
          } else {
            k_cone0 = np::maximum(k_cone0, rate0);
          }
        }
        // A band source on a mover's shadow boundary: the edge there moves at the blocking point P's rigid speed bound
        // times its magnification |X - L| / |P - L| (point lights; 1 for directional ones). An unknown body, or more
        // than one mover blocking (the second one's edges inside the first one's shadow; the I8 review): the cone.
        if (((boundary >> l) & 1u) == 0) continue;
        const uint32_t st = state_of(i, l);
        const MoverBody* m = (st & kStateMulti) != 0 || st == kStateUnknown ? nullptr : mover(st - kStateMover);
        const double t = bits_to_float(state[2 * (i * p.n_bits + sl.bit) + 1]);
        const Vec3d to_light = sl.directional != 0 ? sl.pos : sl.pos - x;
        const double lambda = sl.directional != 0 ? 1.0 : 1.0 / (1.0 - t);
        if (m == nullptr || !(lambda > 0)) {
          cone_src = true;
        } else {
          const double edge = lambda * rigid_speed(*m, x + t * to_light);
          const double e = m == ob_rx.own ? self_edge(obp, ob, edge, self_rel(sl, *m, lambda, t * norm(to_light)))
                                          : oblique(obp, ob, edge);
          k_src = np::maximum(k_src, receiver_rate(p, r_min, e));
          k_src0 = np::maximum(k_src0, receiver_rate(p, r_min, edge));
        }
      }
      // The band (I8): the cone bound of the other pairs only in the tiles within a source's reach (Planner::finish),
      // each at its pixels' cone rate and at most the source's: shadow boundaries at the edge's rate; the retained
      // region (a), plan-pass discontinuities, and the boundaries above at k_any (a pair (d) keeps here may be dropped
      // at the receivers a shadow emerging here reaches; the I8 review). (b) Border tiles keep the cone bound; (c), (d)
      // above.
      if (p.band != 0) {
        if (cone_src) {
          k_src = np::maximum(k_src, k_any);
          k_src0 = np::maximum(k_src0, k_any0);
        }
        if (k_cone > 0 && border) {
          kk = np::maximum(kk, k_cone);
          f |= kPlanShadow;
        }
        if (k_cone0 > 0 && border) kk0 = np::maximum(kk0, k_cone0);
        kcone[i] = k_cone;
        ksource[i] = k_src;
        sreach[i] = k_src > 0 ? reach_of(k_src) : -1;
        if (kcone0 != nullptr) {
          kcone0[i] = k_cone0;
          ksource0[i] = k_src0;
          sreach0[i] = k_src0 > 0 ? reach_of(k_src0) : -1;
        }
      }
      if (hit) {
        for (uint32_t l = 0; l < p.n_moving; ++l) {
          const double edge = moving_edge(moving[l], spheres, x);
          double rate = receiver_rate(p, r_min, edge);
          kk0 = np::maximum(kk0, rate);
          if (obp != nullptr && moving[l].mode != MovingLight::kNone) {
            // d: from the light to the receiver (directional: the propagation direction, MovingLight::pos).
            const Vec3d to_x = moving[l].mode == MovingLight::kPoint ? x - moving[l].pos : moving[l].pos;
            const Obliquity ob = obliquity(ob_rx.nw, ob_rx.rw, (1.0 / norm(to_x)) * to_x, kObliquityCap);
            note(ob, false);
            rate = receiver_rate(p, r_min, oblique(obp, ob, edge));
          }
          kk = np::maximum(kk, rate);
          k_move = np::maximum(k_move, rate);
        }
      }
    }
    // P2-B B4a: the band N's seeds (PHASE2 D-P2-4), beside the rates: a plan-pass discontinuity (hit/miss, instance or
    // depth: silhouettes and occlusion edges), a shadow bit whose state differs from a neighbour's (shadow boundaries,
    // static ones included), a carried sample with edge bits (a material boundary or an edge the centre rays missed
    // at t_w), and with I8's band its retained regions (band sources, and the pixels it keeps at the cone bound:
    // shadows that may be born or be thin there). Each reaches the tiles its content can cross in the window.
    if (p.seeds != 0) {
      // A carried sample is current only inside the crop: outside it the start frame's stays (ActiveFlags).
      const bool carried_edge = carried != nullptr && in_crop(p, i) && (carried[i] & kBandEdge) != 0;
      const bool retained = p.band != 0 && (k_src > 0 || (f & kPlanShadow) != 0);
      const bool plan_seed = disc || differs != 0 || retained;
      int32_t reach_seed = -1;
      double m = 0.0;
      if (plan_seed || carried_edge) {
        // The seed's rate: the static content's (k_static) and the movers' (P2-B B6c: the two parts of its reach).
        double mov = seed_rate(i, x, r_min, differs, k_pairs, k_move, carried_edge, obp);
        if (retained) mov = np::maximum(mov, np::maximum(kk, k_src));
        reach_seed = reach_of(np::maximum(k_static, mov));
        if (p.pixel != 0) m = seed_motion(k_static, mov);
      }
      seed[i] = reach_seed;
      if (p.pixel != 0) {
        seed_m[i] = m;
        seed_kind[i] = static_cast<uint8_t>((plan_seed ? kSeedPlan : 0u) | (carried_edge ? kSeedCarried : 0u));
      }
    }
    // P2-F F9b: an AO receiver of a moving body (AoMover) is not STATIC and takes the body's rate (k and k_base alike;
    // after the seeds, whose reach the AO field, no edge, does not move).
    if (hit && p.n_ao != 0) {
      for (uint32_t a = 0; a < p.n_ao; ++a) {
        const Vec3d rel = x - ao[a].centre;
        if (!(np::dot_seq(rel, rel) <= ao[a].reach * ao[a].reach)) continue;
        const double rate = receiver_rate(p, r_min, ao[a].speed);
        kk = np::maximum(kk, rate);
        kk0 = np::maximum(kk0, rate);
        f |= kPlanAo;
      }
    }
    k[i] = kk;
    reach[i] = reach_of(k_static);
    flags[i] = f;
    if (k_base != nullptr) k_base[i] = kk0;
    if (pv_plan != nullptr) {
      const double q = ceil(2.0 * g_max) - 1.0;  // g_max in [1, kObliquityCap]
      pv_plan[i] = static_cast<uint8_t>((q < 15.0 ? static_cast<uint8_t>(q) : kPvGain) | (cone ? kPvCone : 0) |
                                        (cone && beyond ? kPvBeyond : 0) | (rough(i) ? kPvRough : 0));
    }
    if (pv_width != nullptr) pv_width[i] = static_cast<float>(width);
  }

  // P2-D D1: R4's width of a pair's shadow at the receiver (px; region (d)'s, narrow_shadow).
  BALLA_HD double shadow_width(const PlanParams& q, const ShadowBody& b, const Receiver& r, double rng,
                               double cos_view) const {
    const double lambda = b.mode == ShadowBody::kCylinder ? 1.0 : r.dist / b.far;
    return b.extent * lambda * q.jmin * cos_view / rng;
  }

  // P2-D D1: R4's roughness: pixel i's geometric normal is more than 10 degrees from a 4-neighbour's on its instance.
  BALLA_HD bool rough(size_t i) const {
    if (instance == nullptr || !hit_at(i)) return false;
    const auto w = static_cast<size_t>(p.width), h = static_cast<size_t>(p.height);
    const size_t px = i % w, py = i / w;
    const size_t nb[4] = {px > 0 ? i - 1 : i, px + 1 < w ? i + 1 : i, py > 0 ? i - w : i, py + 1 < h ? i + w : i};
    const Vec3d ni{normal[3 * i], normal[3 * i + 1], normal[3 * i + 2]};
    for (const size_t j : nb) {
      if (j == i || !hit_at(j) || instance[j * depth_stride] != instance[i * depth_stride]) continue;
      if (dot(ni, Vec3d{normal[3 * j], normal[3 * j + 1], normal[3 * j + 2]}) < kRoughCos) return true;
    }
    return false;
  }

  // P2-B B6c: a seed's motion over the window in px beyond rho + s (band_map.h), the static (camera) part scaled by
  // sigma_cam and the movers' part by sigma_mov; a scale of 0 drops its part (also an unbounded one). NaN propagates
  // (BandRadius: the cap).
  BALLA_HD double seed_motion(double k_static, double mov) const {
    const double ms = (p.jmax * k_static) * kWindowS, mm = (p.jmax * mov) * kWindowS;
    return np::maximum(p.sigma_cam > 0 ? p.sigma_cam * ms : 0.0, p.sigma_mov > 0 ? p.sigma_mov * mm : 0.0);
  }

  // P2-B B4a: a seed's rate (rad/s), the input of its reach: the static content's (k_static: camera motion; the
  // caller's) and the movers' part, which this returns (from 0; P2-B B6c scales the two parts apart). On a
  // moving body's hit, its rigid speed bound there (its silhouettes, its texture and the shadows it carries). Per
  // shadow bit whose state differs from a neighbour's (bits): a moving light's edges (k_move); on the shadowed side of
  // a moving body's shadow, the edge's rate as I8's band sources take it (the blocking point's rigid speed bound times
  // its magnification), or where the body is unknown, not alone or not found, the largest pair rate here (k_pairs;
  // without one the blockers are held, P2-B B6b). The lit side of a moving shadow and a static shadow's edges move
  // with the static content: the neighbour on the shadowed side carries the edge's rate. A carried sample's edge
  // (carried_edge) may be any of them, so it takes k_pairs and k_move, or a moving body's silhouette that the centre
  // ray missed (a thin or sub-pixel part), which Planner::finish adds per tile (the bodies' rates of touches()).
  BALLA_HD double seed_rate(size_t i, Vec3d x, double r_min, uint32_t bits, double k_pairs, double k_move,
                            bool carried_edge, const Oblique* obp = nullptr) const {
    double rate = 0.0;
    if (hit_at(i)) {
      if (const MoverBody* own = mover(instance[i * depth_stride])) {
        rate = np::maximum(rate, receiver_rate(p, r_min, rigid_speed(*own, x)));
      }
    }
    if (carried_edge) rate = np::maximum(rate, np::maximum(k_pairs, k_move));
    for (uint32_t b = 0; b < p.n_bits; ++b) {
      if (((bits >> b) & 1u) == 0) continue;
      if (((p.moving_bits >> b) & 1u) != 0) {
        rate = np::maximum(rate, k_move);
        continue;
      }
      const uint32_t st = state[2 * (i * p.n_bits + b)];
      if (st < kStateMover) continue;
      const MoverBody* m = (st & kStateMulti) != 0 || st == kStateUnknown ? nullptr : mover(st - kStateMover);
      uint32_t l = 0;
      while (l < p.n_shadow && shadow_lights[l].bit != b) ++l;
      const double t = bits_to_float(state[2 * (i * p.n_bits + b) + 1]);
      const bool directional = l < p.n_shadow && shadow_lights[l].directional != 0;
      const double lambda = directional ? 1.0 : 1.0 / (1.0 - t);
      if (m != nullptr && l < p.n_shadow && lambda > 0) {
        const Vec3d to_light = directional ? shadow_lights[l].pos : shadow_lights[l].pos - x;
        const double edge = lambda * rigid_speed(*m, x + t * to_light);
        const Obliquity ob = obp != nullptr && obp->on
                                 ? light_obliquity(shadow_lights[l], make_receiver(shadow_lights[l], x, i, n), *obp)
                                 : Obliquity{1.0, 1.0, false, false};
        // D1f: a self-shadow (the blocker is the receiver's own body) in the body's frame.
        const double e = obp != nullptr && m == obp->own
                             ? self_edge(obp, ob, edge, self_rel(shadow_lights[l], *m, lambda, t * norm(to_light)))
                             : oblique(obp, ob, edge);
        rate = np::maximum(rate, receiver_rate(p, r_min, e));
      } else {
        // P2-B B6b: k_pairs, also when it is 0 (it was unbounded then). Sound: held means exactly still over the
        // window (motion.h WindowMotion::held); whenever a body moves, shadow_ holds every static shadow light
        // (Planner::begin), and a moving body that blocks a light here at t_w has its pair's cone over this receiver,
        // so k_pairs > 0. k_pairs = 0 thus means no moving pair can shadow the receiver (I8's band skips kDead pairs
        // only, bodies outside a spot light's outer cone throughout, which cast none of its light's shadows), and the
        // blocker is held: a body the held world traces as static (accuracy.held_world false traces it as a mover,
        // whose id is not among the window's movers), whose shadow of a static light moves with the static content,
        // k_static as under the held world. Moving lights' bits took k_move above.
        rate = np::maximum(rate, k_pairs);
      }
    }
    return rate;
  }
};

// The tile maxima of the pixel stage (from 0, as planner.py's np.maximum.at), the tile's dependence on
// moving shadows or lights (P2-F F9b: or a moving body's AO receivers), and its shadow and hit counts; with the band (kcone not null; I8), the largest droppable
// cone rate, band-source rate and reach (-1: none), and the pixels with a droppable cone rate and no kept one; with
// P2-B B4a's seeds (seed not null), the largest seed reach (-1: none), and inside the crop the seeds and those whose
// carried sample has edge bits (PixelPlan's carried seeds).
struct TileReduce {
  const double* k;
  const int32_t* reach;
  const uint8_t* flags;
  int32_t width, height, tiles_x;
  bool moving_lights;
  double* tile_k;
  int32_t* tile_reach;
  uint8_t* tile_depend;
  uint32_t* tile_shadow;
  uint32_t* tile_hit;
  const double* kcone = nullptr;       // I8's band (PixelPlan)
  const double* ksource = nullptr;
  const int32_t* sreach = nullptr;
  double* tile_kcone = nullptr;
  double* tile_ksource = nullptr;
  int32_t* tile_sreach = nullptr;
  uint32_t* tile_cone = nullptr;
  const int32_t* seed = nullptr;  // P2-B B4a's seeds (PixelPlan)
  const uint32_t* carried = nullptr;
  int32_t* tile_seed = nullptr;
  uint32_t* tile_seeds = nullptr;
  uint32_t* tile_carried = nullptr;
  int32_t crop_x0 = 0, crop_y0 = 0, crop_x1 = 0x7fffffff, crop_y1 = 0x7fffffff;  // PlanParams' crop
  const double* k_base = nullptr;  // P2-D D1 (PixelPlan::k_base): its tile maxima
  double* tile_k_base = nullptr;
  uint32_t* tile_ao = nullptr;  // P2-F F9b: the tile's AO receivers (kPlanAo), when counted
  const double* kcone0 = nullptr;  // P2-D D1f (PixelPlan::kcone0, ...): the band's as built, as kcone, ...
  const double* ksource0 = nullptr;
  const int32_t* sreach0 = nullptr;
  double* tile_kcone0 = nullptr;
  double* tile_ksource0 = nullptr;
  int32_t* tile_sreach0 = nullptr;

  BALLA_HD void operator()(size_t t) const {
    const int32_t ty = static_cast<int32_t>(t) / tiles_x, tx = static_cast<int32_t>(t) % tiles_x;
    double km = 0.0, kb = 0.0;
    int32_t rm = 0;
    uint32_t shadow = 0, hit = 0, cone = 0, ao = 0;
    double kc = 0.0, ks = 0.0, kc0 = 0.0, ks0 = 0.0;
    int32_t sr = -1, sd = -1, sr0 = -1;
    uint32_t seeds = 0, carried_edge = 0;
    for (int32_t y = ty * kTile; y < min(height, (ty + 1) * kTile); ++y) {
      for (int32_t x = tx * kTile; x < min(width, (tx + 1) * kTile); ++x) {
        const size_t i = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
        km = np::maximum(km, k[i]);
        if (k_base != nullptr) kb = np::maximum(kb, k_base[i]);
        rm = max(rm, reach[i]);
        shadow += (flags[i] & kPlanShadow) != 0 ? 1 : 0;
        hit += (flags[i] & kPlanHit) != 0 ? 1 : 0;
        ao += (flags[i] & kPlanAo) != 0 ? 1 : 0;
        if (kcone != nullptr) {
          kc = np::maximum(kc, kcone[i]);
          ks = np::maximum(ks, ksource[i]);
          sr = max(sr, sreach[i]);
          cone += kcone[i] > 0 && (flags[i] & kPlanShadow) == 0 ? 1 : 0;
        }
        if (kcone0 != nullptr) {
          kc0 = np::maximum(kc0, kcone0[i]);
          ks0 = np::maximum(ks0, ksource0[i]);
          sr0 = max(sr0, sreach0[i]);
        }
        if (seed != nullptr) {
          sd = max(sd, seed[i]);
          const bool in = x >= crop_x0 && x < crop_x1 && y >= crop_y0 && y < crop_y1;
          seeds += in && seed[i] >= 0 ? 1 : 0;
          carried_edge += in && carried != nullptr && (carried[i] & kBandEdge) != 0 ? 1 : 0;
        }
      }
    }
    tile_k[t] = km;
    if (tile_k_base != nullptr) tile_k_base[t] = kb;
    tile_reach[t] = rm;
    tile_depend[t] = (shadow > 0 || ao > 0 || (moving_lights && hit > 0)) ? 1 : 0;
    tile_shadow[t] = shadow;
    tile_hit[t] = hit;
    if (tile_ao != nullptr) tile_ao[t] = ao;
    if (tile_kcone != nullptr) {
      tile_kcone[t] = kc;
      tile_ksource[t] = ks;
      tile_sreach[t] = sr;
      tile_cone[t] = cone;
    }
    if (tile_kcone0 != nullptr) {
      tile_kcone0[t] = kc0;
      tile_ksource0[t] = ks0;
      tile_sreach0[t] = sr0;
    }
    if (tile_seed != nullptr) {
      tile_seed[t] = sd;
      tile_seeds[t] = seeds;
      tile_carried[t] = carried_edge;
    }
  }
};

// P2-D D1, after Planner::finish: R4 into the plan bits (pv_width not null), a pixel whose narrowest moving shadow is
// below 2 m D_bound + pitch, D_bound = v_hat W / 2^level of its tile (the level capped at cap; the lattice interval's
// bound) times (q + 1) / 2 on a cone pixel when the rates are as built (obliquity 0: the factor they lack; under the
// factors v_hat already holds it), m = kRoughMargin on rough receivers, else 1; and in the crop the counts of the plan
// bits: [0] cone pixels with g_max > 1 (where the as-built rate is no bound), [1] R6, [2] rough, [3] R4 pixels.
struct PvPlanPass {
  uint8_t* pv_plan;
  const float* pv_width;  // null: no R4
  const uint32_t* pixel_tile;
  const int8_t* level;
  const float* v_hat;
  int32_t cap;
  double pitch;
  int32_t width, crop_x0, crop_y0, crop_x1, crop_y1;
  unsigned long long* counts;
  uint32_t obliquity;  // PlanParams::obliquity

  BALLA_HD void operator()(size_t i) const {
    uint8_t b = pv_plan[i];
    const uint32_t t = pixel_tile[i];
    if (pv_width != nullptr && level[t] >= 0) {
      const int32_t lv = level[t] < cap ? level[t] : cap;
      const double gain = obliquity == 0 && (b & kPvCone) != 0 ? 0.5 * static_cast<double>((b & kPvGain) + 1) : 1.0;
      const double d_bound = static_cast<double>(v_hat[t]) * (kWindowS / static_cast<double>(1u << lv)) * gain;
      const double m = (b & kPvRough) != 0 ? kRoughMargin : 1.0;
      if (static_cast<double>(pv_width[i]) < 2.0 * m * d_bound + pitch) b |= kPvThin;
      pv_plan[i] = b;
    }
    const auto x = static_cast<int32_t>(i % static_cast<size_t>(width));
    const auto y = static_cast<int32_t>(i / static_cast<size_t>(width));
    const bool in = x >= crop_x0 && x < crop_x1 && y >= crop_y0 && y < crop_y1;
    count_if(&counts[0], in && (b & kPvCone) != 0 && (b & kPvGain) >= 2);
    count_if(&counts[1], in && (b & kPvBeyond) != 0);
    count_if(&counts[2], in && (b & kPvRough) != 0);
    count_if(&counts[3], in && (b & kPvThin) != 0);
  }
};

// The tile results of the pixel stage on the host.
struct TileTotals {
  std::vector<double> k;
  std::vector<int32_t> reach;
  std::vector<uint8_t> depend;
  std::vector<double> kcone, ksource;  // I8: with the band (TileReduce::tile_kcone, ...), else empty
  std::vector<int32_t> sreach;
  std::vector<uint32_t> cone;
  // P2-B B4a: with the seeds (TileReduce::tile_seed, tile_carried), else empty: the largest seed reach and the carried
  // seeds per tile.
  std::vector<int32_t> seed;
  std::vector<uint32_t> carried;
  std::vector<double> k_base;  // P2-D D1: under the obliquity factors (TileReduce::tile_k_base), else empty
  // P2-D D1f: under the obliquity factors with the band, the band's tile results as built (TileReduce::tile_kcone0,
  // ...), else empty.
  std::vector<double> kcone0, ksource0;
  std::vector<int32_t> sreach0;
  uint64_t shadow_pixels = 0, hit_pixels = 0;
  uint64_t ao_pixels = 0;  // P2-F F9b: the AO receivers (TileReduce::tile_ao), when counted
  uint64_t seed_pixels = 0, carried_seeds = 0;  // the seeds' totals in the crop (TileReduce::tile_seeds, tile_carried)
};

}  // namespace balla::engine
