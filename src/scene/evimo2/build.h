// The EVIMO2 adapter's bundle build in C++ (ballasim.evimo2.adapter.build_bundle, lit=False; docs/dev/CPP_PORT.md
// item 5, S4), bitwise as the Python: re-timing, world frame, recomposition at identical Vicon rows, object meshes
// (Kabsch registration, body frame at the area centroid, marker spheres), hold views, the per-body fits on a thread
// pool, static bodies, cameras and clocks, the light anchor and relight sidecar, and the meta.
//
// Python keeps what is file reading (U4: the sequence's pickled npz, the catalogue's text files and the asset cache
// until S5) and the lights (light_rig / relight run at every load, until S6): it passes typed arrays in and assembles
// the SceneBundle from the result.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "scene/geometry.h"
#include "scene/rotation.h"
#include "scene/traj/fit_smooth.h"
#include "scene/traj/holds.h"
#include "scene/traj/pin.h"
#include "scene/traj/retime.h"
#include "scene/tree.h"

namespace balla::scene::evimo2 {

using rot::Q;
using rot::V;

// Samples of one key of full_trajectory (dataset.PoseTrack without the raw stamps): rows into it, q (w, x, y, z) as
// recorded, p in metres.
struct Track {
  std::vector<int64_t> rows;
  std::vector<Q> q;
  std::vector<V> p;
};

// One camera folder of the sequence (the first is the bundle's camera, then scene.evimo2.extra_cameras).
struct CameraInput {
  std::string camera, directory, category, recording, name;
  int64_t width = 0, height = 0;
  double fx = 0, fy = 0, cx = 0, cy = 0;
  std::array<double, 5> dist{};          // k1, k2, p1, p2, k3 (Sequence.radtan)
  geo::M4 T_rig_cam{};
  bool g1 = false;                       // a G1 table entry (clocks.g1_delta_ms)
  double g1_delta_ms = 0;
  std::optional<double> g1_se_ms;
  std::optional<double> raw_line3_s;     // clocks.raw_line3_s
};

// A catalogue object (dataset.EvimoObject) with its imported asset when the build uses it.
struct CatalogueObject {
  std::string name;
  double unit_scale = 1;
  std::vector<V> config_markers, vsk_markers_mm;
  double marker_radius_mm = 0;
  bool has_asset = false;
  geo::Mesh mesh;                        // the asset's mesh (asset frame, metres)
  int num_materials = 0;
  bool placeholder = false;              // every asset material untextured (AST-23)
};

struct Options {
  bool markers = true;
  int marker_subdivisions = 2;
  std::string clock_reference = "npz";
  int64_t warmup_ns = 0;
  std::optional<std::vector<int64_t>> objects;
  int64_t min_samples = 10;
  bool static_tabletop = true;
  double static_tabletop_fraction = 0.5;
  bool lights = true;
  double marker_albedo = 0, untextured_albedo = 0;
  std::optional<double> marker_radius_m;
  // M10-PIN (scene.evimo2.pin_static_px_s; off when empty): every object body is frozen where its Vicon image speed
  // stays under this many px/s (traj/pin.h), and is static when that holds for the whole run.
  std::optional<double> pin_static_px_s;
  int workers = 1;
  traj::SmoothingOptions smoothing;      // the scene.trajectory values; h, span, radius, centre and view per fit
};

struct Inputs {
  std::vector<int64_t> t_raw_ns;         // Sequence.full_trajectory_t_raw_ns
  Track cam;                             // Sequence.track("cam"): T_C0<-C
  std::vector<int64_t> object_ids;       // Sequence.object_ids
  std::map<int64_t, Track> tracks;       // the selected objects' tracks and the tabletop's (T_C<-O)
  std::map<int64_t, CatalogueObject> catalogue;
  std::vector<CameraInput> cameras;
};

// One body of the bundle. Materials: the asset's (or one untextured placeholder of untextured_albedo), then one of
// marker_albedo for the marker spheres.
struct Body {
  int64_t id = 0;
  std::string name;
  int64_t semantic = 0;
  geo::Mesh mesh;
  bool placeholder = false;
  int asset_materials = 0;
  bool markers = false;
  std::optional<int64_t> trajectory;
  geo::M4 T_world_body{};
};

struct Camera {
  std::string name;
  int64_t width = 0, height = 0;
  double fx = 0, fy = 0, cx = 0, cy = 0;
  std::array<double, 5> dist{};
  geo::M4 T_rig_cam{};
  int64_t clock_offset_ns = 0;
  std::string sensor;
  double pixel_pitch_m = 0;
};

struct Build {
  int64_t t_start_ns = 0, t_end_ns = 0;
  std::vector<Body> bodies;
  std::vector<size_t> trajectories;      // the bundle's trajectories, as indices into fits (0: the rig)
  std::vector<std::string> fit_names;    // "rig", then each body's name
  std::vector<traj::FitResult> fits;
  std::vector<Camera> cameras;
  traj::Retiming retiming;
  geo::M4 T_world_c0{};
  std::vector<std::pair<int64_t, geo::M4>> T_body_subject;
  std::vector<std::string> warnings;
  Tree sidecar;                          // relight_sidecar()
  Tree meta;                             // the unlit meta: "lights" and each body's "asset" are null for Python
};

// The image-space hold rule's views (adapter._hold_views): the rig's and one per body (mesh in its body frame,
// normalized samples, their Vicon rows).
struct HoldBody {
  const geo::Mesh* mesh;
  const std::vector<Q>* q;
  const std::vector<V>* p;
  const std::vector<int64_t>* rows;
};
std::pair<traj::HoldView, std::vector<traj::HoldView>> hold_views(const std::vector<CameraInput>& cameras,
                                                                  const std::vector<geo::M4>& t_w_rig,
                                                                  const std::vector<HoldBody>& bodies);
// clocks.clock_offset: the camera's clock offset on the "npz" (G1's residual delta) or "raw" (L3 - delta) clock,
// its meta entry (ClockOffset.as_meta) and warnings.
struct ClockOffset {
  int64_t offset_ns = 0;
  Tree meta;
  std::vector<std::string> warnings;
};
ClockOffset clock_offset(const CameraInput& camera, const std::string& reference);
// adapter._tabletop_static: the robust mean pose of the kept samples when at least `fraction` of them lie within
// max_disp_m of it (the bistable tabletop), the inlier fraction and the largest displacement.
struct TabletopStatic {
  std::optional<geo::M4> pose;
  double inlier_fraction = 0, max_displacement_m = 0;
};
TabletopStatic tabletop_static(std::span<const Q> q, std::span<const V> p, std::span<const uint8_t> keep, V centre,
                               double radius, double max_disp_m, double fraction);
// adapter._light_anchor: the dominant plane normal of the tabletop (else the largest body) at t, oriented toward the
// mean camera position, its centre, and the source string; world +z without bodies.
struct Anchor {
  V up, centre;
  std::string source;
};
Anchor light_anchor(const std::vector<Body>& bodies, const std::vector<const traj::FitResult*>& trajectories,
                    std::span<const V> cam_positions, int64_t t_ns);
// adapter._radtan_field: (largest singular value of the pixel Jacobian, largest r^2) over the field.
std::pair<double, double> radtan_field(const CameraInput& c);

Build build(const Inputs& in, const Options& opts);

}  // namespace balla::scene::evimo2
