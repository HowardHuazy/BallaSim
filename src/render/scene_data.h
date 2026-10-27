// Host-side preprocessed render scene, built once from a validated bundle (Renderer::build):
// trace geometries (merged static world + one per mover), the global material table, linear mono
// mip pyramids, camera LUTs and pattern seeds, lights, and the fp64 trajectory data of the pose
// table. host_view() and pose_view() expose it to the shared BALLA_HD code; executors copy the same
// arrays to the device.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/camera/ray_lut.h"
#include "core/rng/warp_table.h"
#include "core/traj/spline.h"
#include "render/bundle.h"
#include "render/pose.h"
#include "render/scene_view.h"

namespace balla::render {

// Renderer settings that shape the preprocessed scene (contract C2 constructor arguments); their parameter
// rows are in render/param_fields.h.
// The pixel integral above N = 64 (accuracy.pixel_integral; PHASE2 D-P2-3, render/shade/shade.h): phase1, every
// term over all N sub-samples; fixed64, the per-member terms over the 64-position prefix (equal at N <= 64).
enum class PixelIntegral : int32_t { kPhase1 = 0, kFixed64 = 1 };
// The shadow ratio over textured clusters (accuracy.texture_shadow; plan p2f §4.1, P2-F F1; render/shade/shade.h):
// pooled, sum f V / sum f; weighted, each member's own texture albedo weighting its f in both sums.
enum class TextureShadow : int32_t { kPooled = 0, kWeighted = 1 };
// The renderer's cluster key (accuracy.shading_patches; plan p2f §4.4, P2-F F5; render/patches.h): material, a
// body's (instance, material); crease, its (instance, patch), patches cut at material changes, creases above 30
// degrees and hard edges on specular bodies.
enum class ShadingPatches : int32_t { kMaterial = 0, kCrease = 1 };
// Mover AO (accuracy.mover_ao; plan p2f §4.7, P2-F F9a; render/ao_proxy.h, render/shade/ao.h): off, the ambient
// unoccluded; proxy, each moving body's sphere-set proxy occludes the ambient of every other body's clusters; traced
// (P2-F F9b, plan §13.3's alternative; render/shade/ao_traced.h), one ambient ray per sub-sample against the other
// movers, within the proxies' windows.
enum class MoverAo : int32_t { kOff = 0, kProxy = 1, kTraced = 2 };

struct SceneSettings {
  uint32_t n_subsamples = 64;
  // Gen3.1 under white LED (DESIGN §7.2). Materials (albedo, texture mips, emission) take w . rgb and
  // lights (background included) intensity w . colour / sum(w) (RENDER.md "Spectral model"), so sum(w)
  // is a gain on all surface radiance, not on a miss's background; the fitted sets sum to 1.
  std::array<double, 3> sensor_rgb_weights{0.52, 0.37, 0.11};
  uint64_t seed = 0;                // run seed of the sub-sample patterns (DESIGN §6.1)
  double aperture_px = 1.0;         // pixel response box(w) * Gaussian(sigma), truncated at
  double blur_sigma_px = 0.15;      // w/2 + n_sigma sigma (DESIGN §6.1)
  double truncation_sigmas = 3.0;
  int32_t warp_cells = 256;
  int32_t lut_samples_per_px = 2;   // ray LUT resolution (DESIGN §8.1)
  PixelIntegral pixel_integral = PixelIntegral::kPhase1;
  // accuracy.unlit_mask (P2-C C2; render/shade/shade.h lit_occlusion): a sub-sample's shadow bit of a light whose
  // direct term at it is 0 counts as lit in the signature and the class word; the irradiance is unchanged.
  bool unlit_mask = false;
  // accuracy.texture_shadow (P2-F F1; render/shade/shade.h accumulate): weighted takes each member's own texture albedo
  // as the weight of its f in a textured cluster's shadow ratio; pooled (v0) does not.
  TextureShadow texture_shadow = TextureShadow::kPooled;
  // accuracy.shading_patches (P2-F F5; render/patches.h, render/shade/shade.h hit_key): crease keys the clusters and
  // the signature by shading patch, so a patch change is a jump; material (v0) by material.
  ShadingPatches shading_patches = ShadingPatches::kMaterial;
  // accuracy.mover_ao (P2-F F9a; render/shade/ao.h): proxy multiplies every cluster's ambient by 1 - O, the share of
  // its hemisphere that other movers' proxy spheres block; traced (P2-F F9b; render/shade/ao_traced.h) by its members'
  // open share of one ambient ray each; off (v0) leaves the ambient unoccluded.
  MoverAo mover_ao = MoverAo::kOff;
};

// The single PSF source (P0_PARAMS §3.3): the sub-sample reach around a pixel centre (px), w/2 + n sigma in
// the operation order of rng::PixelResponse::x_max(). The renderer's warp table, its GT bounds, the planner's
// reach and the cameras' injective domains all take it from the run's settings (0.95 at the defaults).
constexpr double support_px(const SceneSettings& s) {
  return 0.5 * s.aperture_px + s.truncation_sigmas * s.blur_sigma_px;
}

struct GeometryData {
  std::vector<float> positions, normals, uvs;
  std::vector<uint32_t> indices, materials, instances;
  std::vector<uint32_t> patches;  // global patch id per triangle under accuracy.shading_patches crease (has_patches)
  uint32_t instance = 0;     // mover (or split static body) id; 0 for the static world
  int32_t trajectory = -1;   // mover trajectory; -1 for the static world and split static bodies
  Vec3d centre{0, 0, 0};     // bounding sphere in the body frame (movers), the world frame (split static bodies)
  float radius = 0.0f;
  bool has_normals = false;
  bool has_uvs = false;
  bool has_patches = false;  // crease: every geometry, so a merged world carries the world's and the tail's
  uint32_t single_material = 0;
  uint32_t num_vertices() const { return static_cast<uint32_t>(positions.size() / 3); }
  uint32_t num_triangles() const { return static_cast<uint32_t>(indices.size() / 3); }
};

struct CameraData {
  CameraView view;  // pointers into lut/seeds below (host)
  camera::RayLut<float> lut;
  std::vector<uint32_t> seeds;
  std::vector<uint32_t> light_seeds;  // CameraView::light_seeds: pixel * area lights + area light (empty without any)
};

// Trace geometries of a held set (ENGINE_CPP §0.1 I3b; SceneData::trace_set): movers that are still are
// traced as part of the static world at their poses. The merged world [0] is the static world's arrays
// followed by `tail`, the held movers' triangles (ascending id) at their poses, as SceneData builds a world
// of static bodies; the other movers follow. Executors keep the world's arrays and append only the tail.
struct TraceSet {
  // World-frame arrays; indices count from the static world's first vertex. has_normals, has_uvs and
  // single_material are the merged world's (its normals and UVs are zero where a source mesh has none).
  GeometryData tail;
  std::vector<int32_t> geometry;  // per trace geometry, its index in SceneData::geometries() (0: the world)
  std::vector<BodyView> bodies;   // SceneData::bodies() with the trace geometries of this set
  std::vector<PoseGeometryDef> pose_geometries;
  // accuracy.mover_ao proxy: SceneData::ao_bodies() and ao_spheres() with each held mover's spheres in the world frame
  // at its held pose (trace geometry 0) and the others' trace geometries of this set.
  std::vector<AoBodyView> ao_bodies;
  std::vector<AoSphere> ao_spheres;

  // Per trace geometry: `world`, then own[geometry[i]] (a backend's per-geometry handles or views).
  template <class T>
  std::vector<T> select(const T* own, T world) const {
    std::vector<T> out{world};
    for (size_t i = 1; i < geometry.size(); ++i) out.push_back(own[geometry[i]]);
    return out;
  }
};

class SceneData {
 public:
  // nir_fraction (P2-F F11a, p2f.md §4.9): per camera (bundle order; missing cameras and empty rows are 0), the fraction
  // f of its photo-electrons from the near infrared per light group. Every f equal: the NIR is folded into the
  // albedos at level 0 (none at f = 0: Phase 1's textures); otherwise textures have two channels and the cameras mix
  // them per light group (SceneView::nir_two_channel). std::invalid_argument for a row of the wrong length or an f
  // outside [0, 1].
  SceneData(const SceneBundle& bundle, const SceneSettings& settings,
            const std::vector<std::vector<double>>& nir_fraction = {});
  SceneData(const SceneData&) = delete;
  SceneData& operator=(const SceneData&) = delete;

  // Whether nir_fraction needs two-channel textures (the f differ between (camera, light group) pairs), decided from the
  // bundle's cameras and light groups alone: the constructor's rule without building the scene (same errors).
  static bool needs_two_channels(const SceneBundle& bundle, const std::vector<std::vector<double>>& nir_fraction);

  // geometries[0] is the static world (possibly empty); movers follow in ascending body id. With baked_proxies() the
  // world holds the scan bodies only, and every other static body with triangles follows among the movers in id order
  // (trajectory -1, world-frame triangles and bounding sphere; scans.md §4.2).
  const std::vector<GeometryData>& geometries() const { return geometries_; }
  const std::vector<BodyView>& bodies() const { return bodies_; }
  const std::vector<MaterialView>& materials() const { return materials_; }
  const std::vector<std::vector<TextureLevel>>& texture_levels() const { return texture_levels_; }
  const std::vector<int32_t>& texture_channels() const { return texture_channels_; }  // per texture: 1 or 2
  // TextureView::tile_pitch per texture (0, or the tile pitch P of an atlas whose pyramid stops at log2(P) + 1 levels).
  const std::vector<int32_t>& texture_tiles() const { return texture_tiles_; }
  const std::vector<float>& texels() const { return texels_; }
  // SceneView::baked_scale: num_groups() scales S_g = fl32(double(K) double(a_g)) per baked material, in material order.
  const std::vector<float>& baked_scale() const { return baked_scale_; }
  const std::vector<std::unique_ptr<CameraData>>& cameras() const { return cameras_; }
  const std::vector<LightView>& lights() const { return lights_; }
  // Area lights (P2-F F8a-1): the sets Y_l (SceneView::light_points) and their number.
  const std::vector<float>& light_points() const { return light_points_; }
  int32_t num_area_lights() const { return num_area_lights_; }
  const rng::WarpTable<float>& warp() const { return warp_; }
  int32_t num_groups() const { return num_groups_; }
  int32_t num_shadow_lights() const { return num_shadow_lights_; }
  // The material of each global patch id under accuracy.shading_patches crease (empty under material).
  const std::vector<uint32_t>& patch_material() const { return patch_material_; }
  // accuracy.mover_ao proxy: the movers' proxies (render/ao_proxy.h), each body's geometry its index in geometries(),
  // spheres in its body frame (empty under off).
  const std::vector<AoBodyView>& ao_bodies() const { return ao_bodies_; }
  const std::vector<AoSphere>& ao_spheres() const { return ao_spheres_; }
  // accuracy.mover_ao traced (P2-F F9b): each trace geometry's AO reach, 2 d0 of its proxy (SceneView::ao_reach; empty
  // otherwise), and whether a sub-sample traces an ambient ray (a mover with a proxy), which needs occlusion words.
  const std::vector<float>& ao_reach() const { return ao_reach_; }
  bool traces_ao() const { return traces_ao_; }
  // P2-F F6: whether any material has a specular lobe (SceneView::specular; the kV1 shading path).
  bool has_specular() const { return specular_; }
  bool nir_two_channel() const { return nir_two_channel_ != 0; }  // SceneView::nir_two_channel
  // SceneView::baked_proxies: some material is baked and some light is a proxy (scans.md §4.2).
  bool baked_proxies() const { return baked_proxies_ != 0; }

  // The trace geometries with the movers `held` (geometry indices >= 1, ascending, unique) merged into the
  // static world at their world poses: O(held triangles) of work.
  std::unique_ptr<TraceSet> trace_set(std::span<const int32_t> held, std::span<const RigidPose> poses) const;

  // Views over the host arrays. GeometryView and TextureView arrays live inside this object.
  SceneView host_view() const;
  PoseSceneView pose_view() const;
  // Pose-table inputs, for device copies (TrajectoryView::spline.nodes point into spline_nodes()). A piecewise
  // trajectory's nodes are its pieces' node arrays one after the other (each built from the piece's own rows), and
  // piece_views()[t] (empty for one piece; TrajectoryView::pieces points at it) points into them.
  const std::vector<std::vector<traj::SplineNode<double>>>& spline_nodes() const { return spline_nodes_; }
  const std::vector<std::vector<PieceView>>& piece_views() const { return piece_views_; }
  const std::vector<TrajectoryView>& trajectories() const { return trajectories_; }
  const std::vector<PoseCameraDef>& pose_cameras() const { return pose_cameras_; }
  const std::vector<PoseGeometryDef>& pose_geometries() const { return pose_geometries_; }
  const std::vector<PoseLightDef>& pose_lights() const { return pose_lights_; }

 private:
  std::vector<GeometryData> geometries_;
  std::vector<BodyView> bodies_;
  std::vector<MaterialView> materials_;
  std::vector<std::vector<TextureLevel>> texture_levels_;
  std::vector<int32_t> texture_channels_;
  std::vector<int32_t> texture_tiles_;
  std::vector<float> texels_;
  std::vector<float> baked_scale_;
  std::vector<std::unique_ptr<CameraData>> cameras_;
  std::vector<LightView> lights_;
  std::array<float, kMaxLightGroups> ambient_{};
  std::array<float, kMaxLightGroups> background_{};
  // SceneView::fill, proxy_mask, proxy_slot (the proxy lights' per-group tables of U2, scans.md §3.2).
  std::array<float, kMaxLightGroups> fill_{};
  std::array<uint32_t, kMaxLightGroups> proxy_mask_{};
  std::array<int32_t, kMaxLightGroups> proxy_slot_{-1, -1, -1, -1, -1, -1, -1, -1};
  int32_t num_groups_ = 1;
  int32_t num_shadow_lights_ = 0;
  int32_t num_area_lights_ = 0;
  std::vector<float> light_points_;
  int32_t fixed64_ = 0;  // SceneView::fixed64
  int32_t unlit_mask_ = 0;  // SceneView::unlit_mask
  int32_t texture_shadow_ = 0;  // SceneView::texture_shadow
  std::vector<uint32_t> patch_material_;  // SceneView::patch_material (crease)
  int32_t mover_ao_ = 0;  // SceneView::mover_ao
  std::vector<AoBodyView> ao_bodies_;
  std::vector<AoSphere> ao_spheres_;
  std::vector<Vec3d> ao_centres_;  // the spheres' body-frame centres in fp64 (held sets pose them)
  std::vector<float> ao_reach_;    // traced: per trace geometry (SceneView::ao_reach)
  bool traces_ao_ = false;
  bool specular_ = false;  // SceneView::specular
  int32_t nir_two_channel_ = 0;  // SceneView::nir_two_channel
  int32_t baked_proxies_ = 0;   // SceneView::baked_proxies
  rng::WarpTable<float> warp_;
  // Pose-table inputs (fp64).
  std::vector<std::vector<traj::SplineNode<double>>> spline_nodes_;
  std::vector<std::vector<PieceView>> piece_views_;
  std::vector<TrajectoryView> trajectories_;
  std::vector<PoseCameraDef> pose_cameras_;
  std::vector<PoseGeometryDef> pose_geometries_;
  std::vector<PoseLightDef> pose_lights_;
  // Storage behind host_view().
  std::vector<GeometryView> geometry_views_;
  std::vector<TextureView> texture_views_;
  std::vector<CameraView> camera_views_;
};

}  // namespace balla::render
