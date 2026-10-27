// Plain views of the preprocessed render scene, valid in one memory space (host for the Embree
// path, device for the OptiX/CUDA path). Every BALLA_HD function of the renderer reads the scene
// through these structs, so the CPU and the GPU run the same code on the same values.
#pragma once

#include <stdint.h>

#include "core/camera/kannala_brandt.h"
#include "core/camera/radtan.h"
#include "core/camera/ray_lut.h"
#include "core/rng/warp.h"
#include "render/shade/bsdf.h"
#include "render/types.h"

namespace balla::render {

// One trace geometry: the merged static world (index 0), one rigid mover (DESIGN §6.9: direct
// world GAS plus one GAS per dynamic body), or a static body split from the world (scans.md §4.2).
struct GeometryView {
  const float* positions;     // 3 per vertex: world frame for the static world and split static bodies, body frame
                              // for movers
  const uint32_t* indices;    // 3 per triangle
  const float* normals;       // 3 per vertex, or nullptr; zero vectors where a source mesh had none
  const float* uvs;           // 2 per vertex, or nullptr; zero where a source mesh had none
  const uint32_t* materials;  // global material index per triangle
  const uint32_t* instances;  // body id per triangle (static world); nullptr for movers and split static bodies
  uint32_t instance;          // body id of a mover or split static body; 0 for the static world
  uint32_t num_vertices;
  uint32_t num_triangles;
  uint32_t single_material;   // 1 if every triangle has the same material (skips per-prim lookups)
  float sphere_radius;        // bounding sphere about the pose table's centre; +inf for the world
  // accuracy.shading_patches crease (P2-F F5; render/patches.h): the global patch id per triangle, which keys the
  // clusters and the signature instead of the material (render/shade/shade.h hit_key); nullptr under material.
  const uint32_t* patches;
  // 1: a mover (a trajectory; body frame); 0: the static world, or a static body split from it into its own geometry
  // (world frame; scans.md §4.2, SceneView::baked_proxies), whose blocking is static to the plan pass.
  int32_t dynamic;
};

// Body ids sorted ascending, with the trace geometry that holds each body's triangles.
struct BodyView {
  uint32_t id;
  uint32_t semantic;
  int32_t geometry;
};

// One mip level inside the shared texel array. scale_x/y are level-0 texels per level texel.
struct TextureLevel {
  uint32_t offset;
  int32_t width, height;
  float scale_x, scale_y;
};

// Linear, sensor-weighted (mono) texture with the material's base colour folded in, plus its box
// mip pyramid down to 1 x 1 (DESIGN §7.2, S2 MAT-06: filtering and weighting commute).
// P2-F F11a (p2f.md §4.9): channels 2 interleaves (visible, near-infrared) albedo in every texel (level offsets and
// strides count floats), for scenes where the cameras' nir_fraction differs between (camera, light group) pairs.
struct TextureView {
  const TextureLevel* levels;
  int32_t num_levels;
  int32_t channels = 1;
  // C1 Material.texture_tile (scans.md §4.5.6, S5a): 0, or the pitch P of a per-face tile atlas at level 0. Its pyramid
  // stops at log2(P) + 1 levels, so level l holds the cells of side P >> l and the top level one mean per cell; a
  // lookup clamps to the cell its uv lies in (render/shade/ewa.h). Only a mono texture is tiled (channels 1).
  int32_t tile_pitch = 0;
};

struct MaterialView {
  float albedo;    // sensor-weighted base colour, used when the material has no texture
  float emissive;  // sensor-weighted emitted radiance (relative units), light group 0
  int32_t texture; // index into the texture table or -1
  // C1 feature "baked" (docs/design/phase3/scans.md §3.2): the offset of the material's num_groups scales
  // S_g = fl32(K a_g) in SceneView::baked_scale, or -1 for a lit material. A baked material's albedo and texture hold
  // the radiance in texel units (T), not a reflectance.
  int32_t baked;
  // C1 feature "pbr" (P2-F F4, F6; render/shade/bsdf.h): the GGX + Lambert BSDF of a lit material, in the mono
  // parameters of bsdf::BsdfParams. `lobe` is Material::has_lobe (s > 0 or m > 0); a material without one shades as
  // Lambert, bit for bit, whatever the other fields hold (they are 0 then, alpha2 included).
  float alpha2 = 0.0f;    // alpha^2 = r^4 for the roughness r (alpha = r^2), rounded once from fp64
  float metallic = 0.0f;  // m
  float specular = 0.0f;  // s, the dielectric F0
  int32_t lobe = 0;
  // P2-F F11a: the near-infrared albedo of an untextured material (two-channel scenes, SceneView::nir_two_channel; else 0).
  float nir_albedo = 0.0f;
};

enum class CameraModel : int32_t { kRadtan = 0, kKannalaBrandt = 1 };

struct CameraView {
  camera::RayLutView<float> lut;   // rays (x_n, y_n, 1) for radtan, unit rays for Kannala-Brandt
  const uint32_t* pattern_seeds;   // Owen-Sobol scramble seed per pixel (DESIGN §6.1)
  // Area lights (P2-F F8a-1): scramble seed of the padded pair that gives each sub-sample a point of the light's set,
  // at pixel * SceneView::num_area_lights + LightView::area_index; nullptr without area lights.
  const uint32_t* light_seeds;
  int32_t width, height;
  CameraModel model;
  camera::RadtanCamera<float> radtan;       // for the flow Jacobian and the texture footprint
  camera::KannalaBrandtCamera<float> kb;
  // P2-F F11a: the fraction f of this camera's photo-electrons from the near infrared, per light group, where
  // SceneView::nir_two_channel mixes the albedo channels per (camera, group): rho = (1 - f) rho_vis + f rho_nir. Zero
  // otherwise (a mono scene folded f at level 0).
  float nir_fraction[kMaxLightGroups];
};

// kRect and kDisk are the one-sided area lights of the C1 feature "area_light" (P2-F, plan p2f §4.6): `intensity` is
// their radiance L, `direction` the emitting side's normal.
enum class LightKind : int32_t { kPoint = 0, kSpot = 1, kDirectional = 2, kRect = 3, kDisk = 4 };

BALLA_HD inline bool is_area(LightKind k) { return k == LightKind::kRect || k == LightKind::kDisk; }

// A point, spot, directional or area light (ambient and background terms are folded into SceneView::ambient and
// SceneView::background). Its position and direction (and an area light's u axis) in the camera frame live in the
// pose table, per pose slot.
struct LightView {
  LightKind kind;
  int32_t group;
  float intensity;     // intensity * (w . color) / sum(w): white light delivers its intensity
  float cos_outer;     // spot cone: smootherstep falloff from cos_outer (0) to cos_inner (1)
  float cos_inner;
  int32_t shadow_bit;  // bit in the per-sub-sample shadow mask, or -1 if it casts no shadows
  int32_t proxy;       // 1: stands for a scan's captured lighting (C1 Light.proxy); the only light baked receivers see
  // Area lights only (zero otherwise; P2-F F8a-1). The emitter is the rectangle, or the ellipse inscribed in it, of
  // half sizes half_u, half_v along the unit axes u and v = u x n of the pose row (n the row's direction).
  float half_u = 0.0f, half_v = 0.0f;
  float area_gain = 0.0f;    // c_L (plan p2f §4.6): 1 for a rect, pi / (16 sin(pi / 16)) for the disk's inscribed 32-gon
  int32_t point_offset = 0;  // first point of the light's set Y_l in SceneView::light_points (2 floats per point)
  int32_t area_index = 0;    // the light among the area lights: its column of CameraView::light_seeds
};

// accuracy.mover_ao proxy (P2-F F9a; render/ao_proxy.h, render/shade/ao.h): a moving body's proxy spheres, posed by
// the pose row of `geometry`: the mover's own trace geometry (spheres in its body frame) or, while it is held in the
// static world (SceneData::trace_set), the world (spheres in the world frame at the held pose).
struct AoBodyView {
  uint32_t instance;  // the mover's body id: clusters of that body skip its spheres
  int32_t geometry;   // trace geometry whose pose row poses the spheres
  float d0;           // the window's start R_b / sqrt(eps_AO) (m)
};
struct AoSphere {
  float centre[3];  // in its body's geometry frame
  float radius;
  int32_t body;     // index into SceneView::ao_bodies
};

struct SceneView {
  const GeometryView* geometries;  // [0] = static world (possibly empty), then movers (and split static bodies) by id
  int32_t num_geometries;
  const BodyView* bodies;
  int32_t num_bodies;
  const MaterialView* materials;
  const TextureView* textures;
  const float* texels;
  const CameraView* cameras;
  int32_t num_cameras;
  const LightView* lights;
  int32_t num_lights;
  // Area lights (P2-F F8a-1): the sets Y_l of all of them, kLightPoints points each in the order of their 32 x 32 cell
  // (render/shade/lights.h light_cell), stored as the point's local coordinates (a, b) on the emitter in units of the
  // half sizes: a rect's (2u - 1, 2v - 1), a disk's concentric (Shirley-Chiu) image of the same square.
  const float* light_points;
  int32_t num_area_lights;  // the stride of CameraView::light_seeds
  int32_t num_shadow_lights;
  int32_t shadow_light[kMaxShadowLights];  // light index of each shadow bit
  int32_t num_groups;
  float ambient[kMaxLightGroups];  // ambient radiance per light group (colour weighted as for lights)
  // Radiance of the rays that miss every geometry, per light group (D13 item 1; colour weighted as for lights), and
  // whether any is non-zero (without one, misses contribute nothing: the Phase 1 environment).
  float background[kMaxLightGroups];
  int32_t has_background;
  // accuracy.pixel_integral (PHASE2 D-P2-3): 1 under fixed64, whose per-member terms come from the sub-samples k < 64
  // (render/shade/shade.h); 0 under phase1.
  int32_t fixed64;
  // accuracy.unlit_mask (P2-C C2): 1 when the signature and the class word take shadow bits whose light's direct term
  // is 0 at the sub-sample as lit (render/shade/shade.h lit_occlusion); 0: as traced.
  int32_t unlit_mask;
  // accuracy.texture_shadow (P2-F F1): 1 when a textured cluster's shadow sums weight each member's f by its own
  // texture albedo (render/shade/shade.h accumulate); 0: pooled.
  int32_t texture_shadow;
  // accuracy.shading_patches crease (P2-F F5): the material of each global patch id (render/shade/shade.h
  // key_material); nullptr under material, whose cluster keys are the materials themselves.
  const uint32_t* patch_material;
  // P2-F F6: 1 when any material has a lobe (MaterialView::lobe): the executors then run the kV1 shading path, and
  // `bsdf` holds the coefficient tables it reads (host arrays in SceneData::host_view, device copies in the device
  // executor; all nullptr when no material has a lobe).
  int32_t specular;
  bsdf::BsdfTables bsdf;
  // accuracy.mover_ao (P2-F F9a; render/shade/ao.h): 1 under proxy, whose clusters take the ambient times 1 - O from
  // the movers' proxy spheres (body-major; none under off, or without movers).
  int32_t mover_ao;
  int32_t num_ao_spheres;
  const AoBodyView* ao_bodies;
  const AoSphere* ao_spheres;
  // P2-F F9b: 2 under traced (render/shade/ao_traced.h), whose sub-samples trace an ambient ray against the other
  // movers, each up to its reach (2 d0) per trace geometry ([0] = 0, the world; 0 for a mover without a proxy);
  // nullptr otherwise.
  const float* ao_reach;
  // P2-F F11a (p2f.md §4.9): 1 when the cameras' nir_fraction is not the same for every (camera, light group): textures
  // are two-channel (TextureView::channels) and cluster_radiance mixes the channels per (camera, group); 0: one albedo
  // (the NIR folded in at level 0, or absent while every f is 0: Phase 1's path bit for bit).
  int32_t nir_two_channel;
  rng::WarpTableView<float> warp;  // sub-sample warp to the pixel response
  // C1 feature "baked" (scans.md §3.2, §4.1): num_groups fp32 scales per baked material (MaterialView::baked), and
  // whether any material is baked (0: every baked branch is skipped, the Phase 1 renderer).
  const float* baked_scale;
  int32_t has_baked;
  // has_baked and some proxy light (scans.md §3.2, §4.2): the static world (geometry 0) holds the scan bodies only,
  // every other static body has a trace geometry of its own (GeometryView::dynamic 0), and no mover is held in it.
  int32_t baked_proxies;
  // The proxy shadow factor of U2 (scans.md §3.2, §4.2), per light group g: fill[g] = A^f_g, the sum of the group's
  // proxy ambient intensities (already part of ambient[g]); proxy_mask[g] = M_g, the shadow bits of the group's proxy
  // lights P_g; proxy_slot[g] = b_g, the first of those bits (the cluster sums keep the group's F sum in the slot pair
  // of shade.h ClusterSums at b_g), or -1 for a group without a positional proxy (its baked radiance has no factor).
  float fill[kMaxLightGroups];
  uint32_t proxy_mask[kMaxLightGroups];
  int32_t proxy_slot[kMaxLightGroups];
};

struct alignas(16) Float4 {
  float x, y, z, w;
};

// Camera-relative pose of one trace geometry at one pose slot (DESIGN §6.9): evaluated in fp64,
// stored in fp32 with translations that are camera-object distances. 96 B (DESIGN's 88 B padded to
// 16-byte members, so the device reads a row with 128-bit loads).
struct alignas(16) PoseRow {
  Float4 m[3];    // row i = (R_i0, R_i1, R_i2, p_i) of T_g<-cam: X_g = R X_cam + p
  Float4 sphere;  // bounding-sphere centre in the camera frame (movers) and radius (+inf: world)
  Float4 omega;   // relative twist in the camera frame: X_C' = omega x X_C + v (M2 D3); w unused
  Float4 v;
};
static_assert(sizeof(PoseRow) == 96);

// Light position (point, spot, area centre) and propagation direction (spot, directional; an area light's emitting
// normal) in the camera frame, and an area light's unit u axis (zeros for the other kinds).
struct LightRow {
  float position[3];
  float direction[3];
  float axis_u[3];
};
static_assert(sizeof(LightRow) == 36);

struct PoseTableView {
  const PoseRow* rows;         // slot * num_geometries + geometry
  const LightRow* light_rows;  // slot * num_lights + light
  int32_t num_geometries;
  int32_t num_lights;
};

BALLA_HD inline const PoseRow& pose_row(const PoseTableView& t, uint32_t slot, int32_t geometry) {
  return t.rows[static_cast<uint64_t>(slot) * static_cast<uint64_t>(t.num_geometries) +
                static_cast<uint64_t>(geometry)];
}

BALLA_HD inline const LightRow& light_row(const PoseTableView& t, uint32_t slot, int32_t light) {
  return t.light_rows[static_cast<uint64_t>(slot) * static_cast<uint64_t>(t.num_lights) +
                      static_cast<uint64_t>(light)];
}

// Body of an instance id by binary search over the sorted ids; nullptr if unknown.
BALLA_HD inline const BodyView* find_body(const SceneView& s, uint32_t id) {
  int32_t lo = 0, hi = s.num_bodies;
  while (lo < hi) {
    const int32_t mid = (lo + hi) / 2;
    if (s.bodies[mid].id < id) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return (lo < s.num_bodies && s.bodies[lo].id == id) ? &s.bodies[lo] : nullptr;
}

}  // namespace balla::render
