// C++ mirror of the scene bundle (contract C1, docs/dev/CONTRACTS.md): the only input of the
// renderer. Arrays are flattened row-major; units SI; positions relative to origin_world.
//
// Fields of C1 that the renderer does not read (waveforms, holds, clock offsets, sensor presets,
// pixel pitch, f-number, meta) are carried by the integrator and the sensor; they are omitted here.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace balla::render {

struct Mesh {
  std::vector<float> positions;          // V x 3, body frame
  std::vector<uint32_t> indices;         // T x 3
  std::vector<float> normals;            // V x 3 or empty
  std::vector<float> uvs;                // V x 2 or empty; v points down the image (row = v H)
  std::vector<uint16_t> material_ids;    // T or empty (all 0)
};

// How a material's texture (times base_color) is read (docs/design/phase3/scans.md §4.1): kLit, an albedo under the
// lights; kBaked, radiance already lit at capture (glTF KHR_materials_unlit), scaled by baked_scale per light group.
enum class Shading : uint8_t { kLit = 0, kBaked = 1 };

// C1 feature "nir" (P2-F, p2f.md §4.9): the forms of a material's near-infrared albedo (bundle.py NIR_FORMS).
enum class NirForm : uint8_t { kAssumed = 0, kScale, kAffine, kScalar, kTexture };

struct Material {
  std::array<float, 3> base_color{1.0f, 1.0f, 1.0f};  // linear RGB multiplier
  std::vector<uint8_t> texture;                       // H x W x C sRGB-encoded, or empty
  // The size of `texture`, or of `texture_linear` (then C = 3).
  int32_t texture_width = 0, texture_height = 0, texture_channels = 0;  // C = 3 or 4 (alpha ignored)
  std::array<float, 3> emissive{0.0f, 0.0f, 0.0f};    // linear RGB radiance
  // C1 feature "baked" (PHASE3.md D-P3-10; scans.md §3.1). The defaults are the Phase 1 material.
  Shading shading = Shading::kLit;
  float baked_scale = 1.0f;                // K: renderer radiance per linear texel unit (baked only)
  std::vector<float> baked_groups;         // shares a_g per light group, sum 1; empty: all in group 0 (baked only)
  std::vector<uint16_t> texture_linear;    // H x W x 3 linear RGB as IEEE binary16 bits, or empty (baked only)
  int32_t texture_tile = 0;                // 0, or the tile pitch P (a power of two >= 4) of a per-face tile atlas
  // C1 feature "pbr" (P2-F, p2f.md §4.3): the BSDF's mono parameters in [0, 1], roughness r (alpha = r^2), metallic m
  // and specular s (the dielectric F0). s = m = 0 has no lobe: the Lambert material (lit only).
  float roughness = 1.0f, metallic = 0.0f, specular = 0.0f;
  // C1 feature "nir" (P2-F, p2f.md §4.9; lit only): rendered since F11a (sensor.nir_fraction; f = 0 renders the visible albedo).
  NirForm nir_form = NirForm::kAssumed;
  std::vector<double> nir_params;        // the form's 2 parameters, or empty (the default [1, 0])
  std::vector<uint8_t> nir_texture;      // H x W linear NIR albedo (form kTexture only), or empty
  int32_t nir_texture_width = 0, nir_texture_height = 0;

  bool has_lobe() const { return specular > 0.0f || metallic > 0.0f; }
};

struct Body {
  uint32_t id = 0;  // instance id >= 1, unique
  std::string name;
  uint32_t semantic = 0;
  Mesh mesh;
  std::vector<Material> materials;
  std::optional<int32_t> trajectory;  // index into trajectories; nullopt = static
  std::array<double, 16> T_world_body{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  bool scan = false;  // feature "baked": captured scan geometry, static, its shadows in the baked textures
  double normal_prefilter_m = 0.0;  // feature "pbr": r_f >= 0 of the specular normals' prefilter (P2-F F7); 0: none
};

// One piece of a piecewise trajectory (C1 version-2 feature "pieces"; docs/design/phase3/blender.md §3.7, §4.7.7): the
// spline of the control rows [ctrl_begin, ctrl_end) with knots t0_ns + s h_ns, serving [begin_ns, the next begin).
// join (piece p > 0): -1 a step, 0 C0, 1 C1; piece 0: 0.
struct TrajectoryPiece {
  int64_t begin_ns = 0, t0_ns = 0, h_ns = 0;
  int32_t ctrl_begin = 0, ctrl_end = 0;
  int8_t join = 0;
};

struct Trajectory {
  int32_t order = 6;  // K = 4 or 6
  int64_t t0_ns = 0, h_ns = 0;  // piece 0's of a piecewise trajectory
  std::vector<double> rot;  // N x 4, (w, x, y, z), R_world_body
  std::vector<double> pos;  // N x 3, relative to origin_world
  int64_t span_begin_ns = 0, span_end_ns = 0;
  std::vector<TrajectoryPiece> pieces;  // empty: one piece (the spline above); else >= 2, right-continuous
};

enum class CameraModelKind { kRadtan, kKannalaBrandt };

struct Camera {
  std::string name;
  CameraModelKind model = CameraModelKind::kRadtan;
  int32_t width = 0, height = 0;
  double fx = 0, fy = 0, cx = 0, cy = 0;
  std::vector<double> dist;  // radtan: k1, k2, p1, p2[, k3]; kb: k1..k4
  int32_t rig_trajectory = 0;
  std::array<double, 16> T_rig_cam{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};

// kBackground (D13 item 1): the radiance seen by rays that miss every geometry, in the units of shaded surface
// radiance (a surface of albedo rho under an ambient A alone shows rho A); it lights nothing.
// kRect and kDisk (C1 feature "area_light", P2-F p2f.md §4.6): one-sided emitters of radiance `intensity`.
enum class BundleLightKind { kPoint, kSpot, kDirectional, kAmbient, kBackground, kRect, kDisk };

// Point, spot and directional lights have pose-table rows and may cast shadows; ambient and background terms fold
// into per-group constants (scene_data.cpp) and their casts_shadows is ignored.
inline bool positional(BundleLightKind k) { return k != BundleLightKind::kAmbient && k != BundleLightKind::kBackground; }

struct Light {
  BundleLightKind kind = BundleLightKind::kPoint;
  int32_t group = 0;
  std::array<float, 3> color{1.0f, 1.0f, 1.0f};
  float intensity = 0.0f;
  std::optional<int32_t> body;  // index into bodies of the body carrying the light (docs/dev/SCENE.md); nullopt = world
  std::array<double, 3> position{0, 0, 0};
  std::array<double, 3> direction{0, 0, 1};  // propagation direction (spot axis, directional)
  float spot_inner_rad = 0.0f, spot_outer_rad = 0.0f;
  bool casts_shadows = true;
  bool proxy = false;  // feature "baked": stands for the scan's captured, static lighting (scans.md §4.2)
  // Feature "area_light" (kRect, kDisk; `direction` is the emitting side's unit normal): [s_u, s_v] in metres (a
  // disk's diameters), both > 0, and the unit in-plane u axis; other kinds keep these defaults.
  std::array<double, 2> size{0.0, 0.0};
  std::array<double, 3> axis{1.0, 0.0, 0.0};
};

inline bool area_light(BundleLightKind k) { return k == BundleLightKind::kRect || k == BundleLightKind::kDisk; }

struct SceneBundle {
  std::string name;
  int64_t t_start_ns = 0, t_end_ns = 0;
  std::array<double, 3> origin_world{0, 0, 0};
  std::vector<Body> bodies;
  std::vector<Trajectory> trajectories;
  std::vector<Camera> cameras;
  std::vector<std::string> light_groups;  // names; index 0 is the constant group "default"
  std::vector<Light> lights;
};

// Throws std::invalid_argument naming the first inconsistency (index ranges, shapes, ids, finite
// values, spline orders, camera models, and the rules of the features "baked", "pieces", "pbr", "area_light" and "nir"
// as bundle.py states them).
// An empty mesh passes here (only shapes are checked): C1 admits one only for a light's carrier on a trajectory, which
// bundle.py checks (blender.md §3.7). The renderer never hits a meshless body and the planner leaves it out.
void validate(const SceneBundle& bundle);

}  // namespace balla::render
