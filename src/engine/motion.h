// Window motion of a scene bundle and the host stages of the tile planner (ENGINE_CPP §1.1, §1.3 H0
// and H1): ports of the NumPy engine's kinematics.py (trajectory bounds and holds, mesh distances and cones,
// body and light windows), camera.py (tiles, pixel rays, image-speed factors, tile cones, the Morton order)
// and planner.py's tile stages (reach dilation, the touches of moving bodies, levels, STATIC); those modules are
// thin wrappers of this code since I5. The per-pixel stage between Planner::terms and Planner::finish is
// plan.h, run by enqueue_pixel_stage and collect_pixel_stage on either memory space. Every fp64 expression
// follows the NumPy oracle's evaluation (plan.h).
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/math/quat.h"
#include "core/time/time.h"
#include "core/traj/pieces.h"
#include "core/traj/spline.h"
#include "engine/plan.h"
#include "render/bundle.h"

namespace balla::engine {

struct WindowMotion {
  double v = 0, omega = 0;  // speed bounds of the origin (m/s) and angular speed (rad/s)
  bool held = true;         // exactly still over the window
};

using Holds = std::vector<std::pair<int64_t, int64_t>>;  // sorted, disjoint [begin, end) intervals

// R_world_body of a unit quaternion (w, x, y, z), as scipy's Rotation.from_quat(...).as_matrix().
Mat3d quat_to_matrix(const Quat<double>& q);

// How far apart the two one-sided poses at the breakpoint b of a C0 or C1 join (C1 pieces join 0 or 1) may be: the
// left piece's pose at b (its support is closed there) and the right piece's, in position (m) and rotation angle (rad)
// (core/traj/pieces.h). The planner's window bound takes the largest one-sided speed at b, which covers the motion only
// while the pose is continuous there: a larger jump is a step (-1), refused until BL-12. 1e-8 lies about 10^6 above the
// fp64 rounding of a pose within the 16 m reach; BL-8's fitter moves each right piece rigidly onto its left piece's end
// pose (a fitted C0 join's residual, at most 1e-6 eps px, blender.md §4.7.6, can exceed 1e-8 rad for a small far body),
// so its joins keep the pose to rounding. The image displacement 1e-8 can hide is about 1e-3 px in the error model's
// sharpest case (§4.7.2: G = 2667 px at the 5 cm distance floor, a 1 m body radius), far below accuracy.d_px.
using traj::kJoinTolM;
using traj::kJoinTolRad;

// A join labelled continuous (C0 or C1) whose one-sided poses at its breakpoint differ by more than kJoinTol*.
struct JoinJump {
  int32_t piece = 0;  // the right piece p >= 1 (the join is its own)
  int8_t join = 0;
  int64_t at_ns = 0;  // its begin_ns, the breakpoint
  double m = 0.0, rad = 0.0;
};

// Window queries on one C1 trajectory (TrajectoryMotion of the NumPy engine, deleted at I5). A piecewise trajectory
// (C1 "pieces", blender.md §4.7.7) keeps per piece its first time, knot origin, spacing, nodes (from its own rows) and
// segment bounds; a single spline is one piece.
class TrajectoryMotion {
 public:
  TrajectoryMotion(const render::Trajectory& t, Holds holds);
  // Bounds over the closed interval [t0, t1]: every segment that touches it, in every piece serving part of it (the
  // largest of both one-sided bounds at a C0 or C1 join, where the pose is continuous to kJoinTol*). Held: inside a
  // declared hold, or with bounds exactly 0 in one piece (bitwise constant there); zero bounds on both sides of a join
  // do not make it held, since the two pieces' constants may differ (a hold across a join is declared, with one
  // control point under every piece it overlaps, C1).
  WindowMotion window(int64_t t0_ns, int64_t t1_ns) const;
  // (R_world_body, position relative to origin_world) at t_ns, fp64, from the piece serving t_ns (the right one at a
  // breakpoint, as render::body_state).
  std::pair<Mat3d, Vec3d> pose(int64_t t_ns) const;
  const Holds& holds() const { return holds_; }
  // A step join (-1) of a piecewise trajectory: its pose jumps, which no speed bound covers (BL-12 handles jumps).
  bool has_step() const { return has_step_; }
  // The first join labelled C0 or C1 whose pose jumps by more than kJoinTolM or kJoinTolRad: a mislabelled step.
  const std::optional<JoinJump>& jump() const { return jump_; }

 private:
  struct Piece {
    int64_t begin_ns, t0_ns;
    int32_t h_ns, segments;
    size_t node0, seg0;  // its first node in nodes_ (its rows' offset) and its first segment in v_ and omega_
  };
  // Piece pc's own spline at t_ns (within its support, closed at both ends).
  traj::Kinematics<double> evaluate(const Piece& pc, int64_t t_ns) const;
  int order_;
  std::vector<Piece> pieces_;
  std::vector<traj::SplineNode<double>> nodes_;
  std::vector<double> v_, omega_;  // per segment, the pieces' one after the other
  Holds holds_;
  bool has_step_ = false;
  std::optional<JoinJump> jump_;
};

// The caller's scratch of MeshDistance::cone (one per thread of queries, e.g. per Planner): the
// vertices already evaluated in the current query (stamp == epoch), for any mesh.
struct ConeScratch {
  std::vector<uint32_t> stamp;
  uint32_t epoch = 0;
};

// Distance bounds from points to one mesh and cones that hold it, in the body frame
// (kinematics.py MeshDistance): Morton-ordered clusters of kCluster triangles with bounding spheres,
// built by the constructor. Queries do not modify the object (concurrent queries are safe). The constructor
// refuses non-finite positions and vertex indices out of range, and stores a triangle (a, b, c) whose a and b
// coincide as (b, c, a), whose distance is the exact (finite) distance to the segment it reduces to.
class MeshDistance {
 public:
  static constexpr size_t kCluster = 128;
  static constexpr double kRel = 0.01;
  explicit MeshDistance(const render::Mesh& mesh);
  // Lower bound on the distance from q to the mesh surface, within 2 kRel of the exact distance.
  double distance(Vec3d q) const;
  // Unit axis from q towards the mesh's box centre and the largest angle to a vertex; none when q is
  // the box centre.
  std::optional<std::pair<Vec3d, double>> cone(Vec3d q, ConeScratch& scratch) const;
  // The thickness of the mesh's thinnest parts, at most cap (ENGINE_CPP I8 review: the band's region (d) per part):
  // per triangle (at most kThicknessSamples of them: every ceil(nt / kThicknessSamples)-th in Morton order), the
  // distance from its centroid along the inward normal (the winding's, oriented by the signed volume) to the nearest
  // other triangle; the smallest value that kThinArea of the sampled surface area reaches; 0 for a mesh that encloses
  // no volume (open or flat). A thin part of a thick body (a rod on a box, a blade on its handle) is thin here, though
  // the bounding box is not; a part with less than kThinArea of the surface is not judged.
  static constexpr size_t kThicknessSamples = 1024;
  static constexpr double kThinArea = 0.02;
  double thickness(double cap) const;

 private:
  double triangles_distance(Vec3d q, size_t cluster, double run) const;
  std::vector<Vec3d> v_;
  std::vector<std::array<uint32_t, 3>> tri_;  // in Morton order of their centroids
  std::vector<Vec3d> centre_;                  // bounding spheres of the clusters
  std::vector<double> radius_;
  std::vector<Vec3d> sub_centre_;              // bounding spheres of groups of kSub consecutive triangles
  std::vector<double> sub_radius_;
  Vec3d box_centre_{};
};

// Distance from p to each triangle (a, b, c) by Voronoi region (the NumPy engine's _point_triangle_distance).
double point_triangle_distance(Vec3d p, Vec3d a, Vec3d b, Vec3d c);

struct BodyWindow {
  int32_t index = 0;   // into bundle.bodies
  bool held = true;
  double v = 0, omega = 0, radius = 0;
  Vec3d centre{};      // body origin at the window midpoint (relative to origin_world)
  Mat3d rotation{};    // R_world_body at the window midpoint
  double half_s = 0;   // half the window (s)
  const MeshDistance* mesh = nullptr;  // null for a body the planner never queries (SceneMotion)
  double extent = 0;   // the smallest extent of the mesh's bounding box in the body frame (I8's band, region (d))
  // (Planner::terms takes the smaller MeshDistance::thickness under the band)

  double point_speed() const { return v + omega * radius; }
  double swept_radius() const { return radius + v * half_s; }
  Vec3d to_body(Vec3d p) const;
  // Lower bound over the window on the distance between the body's surface and a point at p at the
  // window midpoint that moves at most p_speed.
  double distance(Vec3d p, double p_speed = 0.0) const;
};

struct LightWindow {
  bool directional = false;  // else point or spot
  Vec3d position{};          // point: mid-window position; directional: unit propagation direction
  bool spot = false, casts_shadows = true;
  int32_t body = -1;         // carrier, or -1
  bool moving = false;
  double v = 0, omega = 0;
  int32_t shadow_bit = -1;   // the renderer's shadow bit (casts_shadows), else -1
  Vec3d axis{};              // spot: unit axis at mid-window
  double outer = 0;          // spot: outer cone half-angle (rad)
};

// The bodies' hold intervals for S1 MOVING (host or device arrays): ids sorted ascending, each body's
// holds [hold_off[b], hold_off[b + 1]) of holds (begin, end pairs), and whether it has a trajectory.
struct MovingTable {
  const uint32_t* id;
  const uint8_t* has_trajectory;
  const uint32_t* hold_off;
  const int64_t* holds;
  uint32_t n;

  // The instance is outside its hold intervals at t (false for static bodies and unknown ids).
  BALLA_HD bool moving(uint32_t instance, int64_t t) const {
    if (instance == 0) return false;
    uint32_t lo = 0, hi = n;  // the first id >= instance
    while (lo < hi) {
      const uint32_t mid = (lo + hi) / 2;
      if (id[mid] < instance) lo = mid + 1;
      else hi = mid;
    }
    if (lo == n || id[lo] != instance || has_trajectory[lo] == 0) return false;
    for (uint32_t h = hold_off[lo]; h < hold_off[lo + 1]; ++h) {
      if (holds[2 * h] <= t && t < holds[2 * h + 1]) return false;
    }
    return true;
  }
};

// The host arrays of a MovingTable.
struct MovingHolds {
  std::vector<uint32_t> id, hold_off;
  std::vector<uint8_t> has_trajectory;
  std::vector<int64_t> holds;
  MovingTable table() const {
    return {id.data(), has_trajectory.data(), hold_off.data(), holds.data(), static_cast<uint32_t>(id.size())};
  }
};

class SceneMotion {
 public:
  // holds[i]: the hold intervals of trajectory i (C1; the renderer's bundle does not carry them).
  // planner_meshes: build the MeshDistance of every body the planner may query (bodies on
  // trajectories; every other body too when a light on a trajectory casts point shadows); the other
  // bodies' BodyWindow::mesh is null.
  SceneMotion(const render::SceneBundle& bundle, std::vector<Holds> holds, bool planner_meshes);
  WindowMotion camera_window(size_t cam, int64_t t0_ns, int64_t t1_ns) const;
  std::pair<Mat3d, Vec3d> camera_pose(size_t cam, int64_t t_ns) const;  // (R_world_cam, centre)
  std::vector<BodyWindow> body_windows(int64_t t0_ns, int64_t t1_ns) const;
  std::vector<LightWindow> light_windows(int64_t t0_ns, int64_t t1_ns) const;
  bool body_moving_at(size_t body, int64_t t_ns) const;
  // Ids of the bodies on trajectories that are held over [t0, t1] (ENGINE_CPP I3b: the renderer's held set).
  std::vector<uint32_t> held_bodies(int64_t t0_ns, int64_t t1_ns) const;
  const MovingHolds& moving_holds() const { return moving_; }
  const render::SceneBundle& bundle() const { return bundle_; }
  const MeshDistance* mesh(size_t body) const { return meshes_[body] ? &*meshes_[body] : nullptr; }  // as BodyWindow
  double extent(size_t body) const { return extent_[body]; }                                         // as BodyWindow

 private:
  struct Carrier {
    Mat3d r;
    Vec3d p;
    WindowMotion m;
  };
  Carrier carrier(size_t body, int64_t t0_ns, int64_t t1_ns) const;
  const render::SceneBundle& bundle_;
  std::vector<TrajectoryMotion> trajectories_;
  std::vector<double> radius_, extent_;
  std::vector<std::optional<MeshDistance>> meshes_;
  MovingHolds moving_;
};

// Fixed per-camera data of the planner and the request order (camera.py CameraGeometry).
struct CameraGeometry {
  // support_px: the sub-sample reach around a pixel centre (render::support_px of the run's settings).
  CameraGeometry(const render::Camera& cam, double support_px);
  double support_px;
  int32_t width, height, tiles_x, tiles_y;
  size_t n_pixels;
  std::vector<double> rays;            // unit ray of every pixel centre, n x 3
  std::vector<uint32_t> pixel_tile;    // tile of every pixel
  std::vector<double> speed_factor;    // per tile: max ||J(d)|| over the tile's support and its neighbours
  double speed_factor_max;             // over the image
  double speed_factor_min;             // the smallest singular value of J over the image (I8's band, region (d))
  std::vector<Vec3d> tile_ray;         // tile cones: centre ray and angular radius
  std::vector<double> tile_radius;
  std::vector<uint32_t> pixel_by_rank; // (tile Morton code, pixel) order
};

struct Crop {
  int32_t x0, y0, x1, y1;  // x0 <= x < x1, y0 <= y < y1
};

// P2-H H9 (accuracy.level_policy; F004, DESIGN §13.5): the lattice level of each tile. tile: its own (Phase 1's);
// global: the window's largest tile level L_w on every active tile, static ones included; global_certified: L_w on the
// dynamic tiles, the static tiles certified as under tile. L_w is the largest level of the per-tile plan's dynamic
// tiles (same speed bound and dyadic rounding), l_min when no tile is dynamic.
enum class LevelPolicy : int32_t { kPerTile = 0, kGlobal = 1, kGlobalCertified = 2 };

struct PlanSchedule {
  double d_px = 0.25;
  int32_t l_min = 1, l_max = 13;
  LevelPolicy level_policy = LevelPolicy::kPerTile;  // P2-H H9: applied after the per-tile levels (Planner::finish)
  // I8 (accuracy.shadow_band): the moving-shadow boundary band, from the plan pass's shadow states and instances
  // (plan.h PixelPlan).
  bool band = false;
  // P2-B B4a (EngineConfig::n_band): the band N's seeds and tile map (TilePlan::band), from the same plan pass, which
  // then runs in every window; the tile levels do not change.
  bool seeds = false;
  // P2-B B6c (accuracy.band_growth pixel, with the seeds): a per-pixel band (band_map.h) instead of the tile map, each
  // seed's motion scaled by sigma_cam (its static part) and sigma_mov (its movers' part; accuracy.band_reach).
  bool pixel = false;
  double sigma_cam = 1.0, sigma_mov = 1.0;
  // P2-D D1 (F021): obliquity (accuracy.shadow_obliquity projected) scales the shadow-edge rates by the receivers'
  // obliquity factors from the plan pass's geometric normals (plan.h obliquity), moving receivers composed; bits writes
  // P&V's plan bits (PixelPlan::pv_plan) and their counts in every window with moving-shadow pairs or moving lights,
  // rates as they are (the engine sets it: under off, plan_oblique_pixels counts where the as-built rate is no bound);
  // pv adds R4 (the narrowest moving shadow at the mesh's thickness, against 2 m D_bound + pitch, pitch the sub-sample
  // pitch in px). Each needs the plan pass's normals (Planner::plan_bits); obliquity and pv write the bits in every
  // window.
  bool obliquity = false, bits = false, pv = false;
  double pitch = 0;
  // P2-F F9b (accuracy.mover_ao proxy or traced): the moving bodies' AO receivers (plan.h AoMover), which need the plan
  // pass (its depths) in every window with a moving body that has a proxy.
  bool ao = false;
};

// The planner's schedule of a run's accuracy keys: l_min 1 with the midpoint test (refine_tol_log set), else 0, and at
// most l_max (min_dt_ns = 2^23 leaves level 0 alone); l_max the level of spacing min_dt_ns, 23 - log2(min_dt_ns).
// min_dt_ns must be a power of two in [1, 2^23] and d_px > 0 (the engine's checks; std::invalid_argument).
inline PlanSchedule plan_schedule(double d_px, std::optional<double> refine_tol_log, int64_t min_dt_ns) {
  if (min_dt_ns < 1 || min_dt_ns > time::kWindowNs || (min_dt_ns & (min_dt_ns - 1)) != 0) {
    throw std::invalid_argument("min_dt_ns must be a power of two in [1, 2^23]");
  }
  if (!(d_px > 0)) throw std::invalid_argument("d_px must be > 0");
  int32_t shift = 0;
  while ((int64_t{1} << shift) < min_dt_ns) ++shift;
  const int32_t l_max = time::kWindowLog2 - shift;
  return {d_px, std::min(refine_tol_log.has_value() ? 1 : 0, l_max), l_max};
}

struct PlanStats {
  bool camera_held = true;
  double camera_v = 0, camera_omega = 0;
  uint32_t moving_bodies = 0, moving_lights = 0;
  bool plan_pass = false;
  uint64_t shadow_pixels = 0, light_pixels = 0, tiles_static = 0, tiles_dynamic = 0;
  std::vector<uint64_t> levels;  // tiles per level (l_max + 1)
  double v_hat_max = 0;
  // P2-H H9 (PlanSchedule::level_policy global or global_certified): the window's level L_w (-1 under tile).
  int32_t level_w = -1;
  // P2-B B4a (PlanSchedule::seeds): the band N's dynamic tiles (TilePlan::band), its seed pixels and those of them
  // whose carried sample has edge bits.
  bool seeds = false;
  uint64_t band_tiles = 0, seed_pixels = 0, carried_seeds = 0;
  // P2-B B6c (PlanSchedule::pixel; band_tiles 0): the per-pixel band's largest seed radius (px; -1: no seed) and its
  // distinct radii (the cascade's levels), set by the camera's plan (Camera::plan_window), not by the Planner.
  bool pixel = false;
  int32_t radius_max = -1;
  uint64_t radius_levels = 0;
  // P2-D D1 (PlanSchedule::obliquity): the dynamic tiles whose level the obliquity factors raise and lower (against g =
  // h = 1 under the same plan); with the plan bits (obliquity or pv; set by the camera's plan, PvPlanPass) in the crop:
  // the cone pixels whose factor exceeds 1 (where the as-built rate is no bound), R6's, the rough and R4's pixels.
  bool obliquity = false, pv_bits = false;
  uint64_t obliquity_tiles_up = 0, obliquity_tiles_down = 0;
  // P2-D D1f (Planner::set_obliquity_budget): the tiles whose raised level the budget's ceiling lowered, and the
  // ceiling (-1: none needed).
  uint64_t obliquity_tiles_capped = 0;
  int32_t obliquity_ceiling = -1;
  uint64_t oblique_pixels = 0, beyond_pixels = 0, rough_pixels = 0, thin_pixels = 0;
  // P2-F F9b (PlanSchedule::ao): the moving bodies with AO receivers and the receivers (kPlanAo pixels).
  bool ao = false;
  uint32_t ao_movers = 0;
  uint64_t ao_pixels = 0;
};

struct TilePlan {
  std::vector<uint8_t> is_static;  // no samples in this window
  std::vector<int8_t> level;       // -1 where static
  std::vector<double> v_hat;       // image-speed bound (px/s)
  // P2-B B4a (PlanSchedule::seeds; else empty): 1 where the tile's pixels sample at N_band, the tiles within a seed's
  // reach (dilate_by_reach), else at N_base.
  std::vector<uint8_t> band;
  // P2-B B6c (PlanSchedule::pixel, band empty; empty without a plan pass, where no seed exists): per tile the floor of
  // its carried seeds' motion, sigma_mov times the reach (px) of the moving bodies that may touch it (touches();
  // NaN or inf: unbounded), for band_map.h BandRadius.
  std::vector<double> band_floor;
  PlanStats stats;
};

// out[T] = max of value[S] over tiles S within Chebyshev distance radius[S] of T (planner.py), for
// value >= +0 and radius >= 0 (the planner's; NaN propagates). O(tiles x distinct radii), any radius.
std::vector<double> dilate_by_reach(const std::vector<double>& value, const std::vector<int32_t>& radius,
                                    int32_t tiles_y, int32_t tiles_x);

// The pixel stage's terms of a light (and a body) over a window: a static shadow-casting light whose
// body terms are [first, first + count), one (light, moving body) pair, a moving light (directional
// ones refer to body spheres [0, every.size())), and a body's swept sphere.
ShadowLight shadow_light(const LightWindow& l, uint32_t first, uint32_t count);
ShadowBody shadow_body(const LightWindow& l, const BodyWindow& body, ConeScratch& scratch);
MoverBody mover_body(const BodyWindow& b, uint32_t id);
MovingLight moving_light(const LightWindow& l, const std::vector<BodyWindow>& every);
BodySphere body_sphere(const BodyWindow& b);

// The pixel stage of a window (Planner::begin).
struct PixelStage {
  bool needed = false;  // the plan pass runs
  PlanParams params;
  std::vector<ShadowLight> shadow_lights;
  std::vector<ShadowBody> shadow_bodies;
  std::vector<MovingLight> moving;
  std::vector<BodySphere> spheres;
  std::vector<MoverBody> movers;  // I8: the moving bodies of the band (params.n_movers)
  std::vector<AoMover> ao;        // P2-F F9b: the moving bodies' AO reach (params.n_ao)
};

// Tile plans of one camera (planner.py Planner). A window takes begin(t_w) (motion; whether the plan
// pass is needed), terms() (the pixel stage's terms) and touches() (moving bodies' tiles), the pixel
// stage on the plan pass's depths (enqueue_pixel_stage), and finish() with its tile results (none
// when the pass is not needed). touches() does not need the pixel stage, so the device engine runs it
// while the GPU runs the plan pass and the pixel kernels; plan() runs the sequence with a pixel-stage
// callback. The SceneMotion must have been built with planner_meshes.
class Planner {
 public:
  Planner(const SceneMotion& motion, const CameraGeometry& geometry, uint16_t cam, PlanSchedule schedule,
          std::optional<Crop> crop);
  void begin(int64_t t_w);
  bool plan_pass() const { return stage_.needed; }
  const PixelStage& terms();
  void touches();
  TilePlan finish(const TileTotals* tiles);
  TilePlan plan(int64_t t_w, const std::function<TileTotals(const PixelStage&)>& pixel_stage);
  const std::vector<uint32_t>& crop_pixels_per_tile() const { return crop_per_tile_; }
  const PlanSchedule& schedule() const { return schedule_; }
  // P2-D D1 (after begin()): whether this window's pixel stage reads the plan pass's normals and writes the plan bits
  // (PlanParams::bits).
  bool plan_bits() const {
    const bool lights = !shadow_.empty() || !moving_.empty();
    return stage_.needed && (schedule_.obliquity || schedule_.pv || (schedule_.bits && lights));
  }
  const PixelStage& stage() const { return stage_; }  // the window's terms (after terms())
  const Crop& crop() const { return crop_; }
  // P2-D D1f: under the obliquity factors, the window's lattice budget (samples; 0: none) for the next finish(): where
  // the factors raise the lattice above it, the raised tiles' levels take the highest ceiling that fits, never below
  // the level without the factors, the as-built one (PlanStats::obliquity_tiles_capped). The engine sets it to
  // obliquity_budget() (Camera::plan_window).
  void set_obliquity_budget(uint64_t samples) { obliquity_budget_ = samples; }

 private:
  const SceneMotion& motion_;
  const CameraGeometry& geo_;
  uint16_t cam_;
  PlanSchedule schedule_;
  uint64_t obliquity_budget_ = 0;  // P2-D D1f
  std::vector<uint8_t> active_tiles_;
  std::vector<uint32_t> crop_per_tile_;
  Crop crop_{};  // the crop (the whole image without one): P2-B B4a's seeds (PlanParams::crop_*)
  // The window's state.
  int64_t t_w_ = 0;
  WindowMotion cm_;
  std::vector<BodyWindow> every_, bodies_;  // all bodies; those not held
  std::vector<LightWindow> moving_, shadow_;
  PixelStage stage_;
  PlanStats stats_;
  std::vector<double> rate_;
  std::vector<uint8_t> dynamic_;
  ConeScratch cone_;
  // I8's band, region (d): a body's min(extent, MeshDistance::thickness) (NaN: not yet computed; the constructor
  // computes those of the bodies on trajectories).
  std::vector<double> thickness_;
  double thickness(const BodyWindow& body);
  // P2-F F9b (schedule_.ao): per body of the bundle its AO window start d0 (render/ao_proxy.h, the renderer's fit of
  // its mesh; 0: no proxy, no AO), for the bodies on trajectories.
  std::vector<double> ao_d0_;
};

// P2-D D1f: the obliquity factors' lattice budget per window (samples; Planner::set_obliquity_budget), from the run's
// configuration alone, so that the plan and the events do not depend on the device (its budget only batches, output-
// invariant): the factors' runs at twice the lattice (the reservation with refinement room, WindowEngine::run; about
// 8 + 4 G bytes per sample, G light groups) within kObliquityRunBytes, and within accuracy.max_samples_per_window
// where set, so that its clamp does not cut the as-built levels for the factors' sake. 3 GiB is a design choice, not a
// maths bound: about half of a 2080 Ti's default device budget (6.4 GiB, 70 % of the memory free after the build).
inline constexpr uint64_t kObliquityRunBytes = uint64_t{3} << 30;
inline uint64_t obliquity_budget(int groups, std::optional<int64_t> max_samples_per_window) {
  uint64_t samples = kObliquityRunBytes / (2 * (8 + 4 * static_cast<uint64_t>(std::max(groups, 1))));
  if (max_samples_per_window) {
    samples = std::min(samples, static_cast<uint64_t>(std::max<int64_t>(*max_samples_per_window, 1)));
  }
  return samples;
}

// Buffers of the pixel stage in one memory space. The tile results share one buffer (TileSlots), so
// that collect_pixel_stage downloads them with one copy.
// P2-B B6c (PlanSchedule::pixel): the seeds' motion and kinds (PixelPlan::seed_m, seed_kind) and the per-pixel band's
// scratch (band_map.h: the radii, their histogram, the cascade's line buffer, the tiles' floors).
template <Space S>
struct PlanBuffers {
  Buffer<double, S> rays, k, kcone, ksource;
  Buffer<int32_t, S> reach, sreach, seed;
  Buffer<uint8_t, S> flags, tiles;
  Buffer<ShadowLight, S> shadow_lights;
  Buffer<ShadowBody, S> shadow_bodies;
  Buffer<MovingLight, S> moving;
  Buffer<BodySphere, S> spheres;
  Buffer<MoverBody, S> movers;
  Buffer<double, S> seed_m, band_floor;
  Buffer<uint8_t, S> seed_kind, band_tmp;
  Buffer<int32_t, S> band_radius;
  Buffer<uint32_t, S> band_levels;
  // P2-D D1: the rates with g = h = 1, the plan bits, R4's widths and the plan bits' counts (PvPlanPass).
  Buffer<double, S> k_base;
  Buffer<uint8_t, S> pv_plan;
  Buffer<float, S> pv_width;
  Buffer<unsigned long long, S> pv_counts;
  Buffer<AoMover, S> ao;  // P2-F F9b
  // P2-D D1f: under the factors with the band, the band's per-pixel outputs as built (PixelPlan::kcone0, ...).
  Buffer<double, S> kcone0, ksource0;
  Buffer<int32_t, S> sreach0;
  bool rays_ready = false;
  size_t bytes() const {
    return rays.bytes() + k.bytes() + kcone.bytes() + ksource.bytes() + reach.bytes() + sreach.bytes() + seed.bytes() +
           flags.bytes() + tiles.bytes() + shadow_lights.bytes() + shadow_bodies.bytes() + moving.bytes() +
           spheres.bytes() + movers.bytes() + seed_m.bytes() + band_floor.bytes() + seed_kind.bytes() +
           band_tmp.bytes() + band_radius.bytes() + band_levels.bytes() + k_base.bytes() + pv_plan.bytes() +
           pv_width.bytes() + pv_counts.bytes() + ao.bytes() + kcone0.bytes() + ksource0.bytes() +
           sreach0.bytes();
  }
};

// The tile results' places in PlanBuffers::tiles: k and I8's band's kcone and ksource (double), under P2-D D1's
// obliquity factors (`base`) k_base and D1f's band as built, kcone0 and ksource0 (double) and sreach0 (int32), reach
// and the band's sreach (int32), shadow and hit counts and the band's cone count (uint32), with P2-B B4a's seeds
// (`seeds`) the seed reach (int32) and the seed and carried-edge counts (uint32), with P2-F F9b's AO receivers (`ao`)
// their count (uint32), and depend (uint8), each over the tiles in this order.
struct TileSlots {
  size_t tiles;
  bool seeds = false, base = false, ao = false;
  size_t seed_slots() const { return seeds ? tiles * (sizeof(int32_t) + 2 * sizeof(uint32_t)) : 0; }
  size_t base_slots() const { return base ? tiles * (3 * sizeof(double) + sizeof(int32_t)) : 0; }
  size_t ao_slots() const { return ao ? tiles * sizeof(uint32_t) : 0; }
  size_t bytes() const {
    return tiles * (3 * sizeof(double) + 2 * sizeof(int32_t) + 3 * sizeof(uint32_t) + 1) + seed_slots() + base_slots() +
           ao_slots();
  }
  size_t kcone() const { return tiles * sizeof(double); }
  size_t ksource() const { return kcone() + tiles * sizeof(double); }
  size_t k_base() const { return ksource() + tiles * sizeof(double); }
  size_t kcone0() const { return k_base() + tiles * sizeof(double); }
  size_t ksource0() const { return kcone0() + tiles * sizeof(double); }
  size_t sreach0() const { return ksource0() + tiles * sizeof(double); }
  size_t reach() const { return k_base() + base_slots(); }
  size_t sreach() const { return reach() + tiles * sizeof(int32_t); }
  size_t shadow() const { return sreach() + tiles * sizeof(int32_t); }
  size_t hit() const { return shadow() + tiles * sizeof(uint32_t); }
  size_t cone() const { return hit() + tiles * sizeof(uint32_t); }
  size_t seed() const { return cone() + tiles * sizeof(uint32_t); }
  size_t seeds_count() const { return seed() + tiles * sizeof(int32_t); }
  size_t carried() const { return seeds_count() + tiles * sizeof(uint32_t); }
  size_t ao_count() const { return cone() + tiles * sizeof(uint32_t) + seed_slots(); }
  size_t depend() const { return ao_count() + ao_slots(); }
};

// K2 on the plan pass's depths (Space memory; depth[i * stride] is pixel i's) and the tile maxima,
// queued; collect_pixel_stage downloads them. I8's band and P2-B B4a's seeds: the plan pass's instances (at depth's
// stride) and shadow states (st.params.n_bits per pixel); the seeds also take the carried samples' class words by pixel
// id (null: none). P2-D D1's obliquity and plan bits: the instances and the plan pass's normals (3 floats per pixel),
// and they write the plan bits (b.pv_plan; PvPlanPass completes them after Planner::finish).
template <Space S>
void enqueue_pixel_stage(const PixelStage& st, const CameraGeometry& geo, const float* depth, size_t stride,
                         PlanBuffers<S>& b, Stream stream, const uint32_t* instance = nullptr,
                         const uint32_t* state = nullptr, const uint32_t* carried = nullptr,
                         const float* normal = nullptr) {
  const size_t n = geo.n_pixels;
  const bool obl = st.params.obliquity != 0, bits = st.params.bits != 0, ao = st.params.n_ao != 0;
  const bool seeds = st.params.seeds != 0, pass = st.params.band != 0 || seeds || bits;
  const bool band0 = obl && st.params.band != 0;  // P2-D D1f: the band as built
  const TileSlots ts{static_cast<size_t>(geo.tiles_x) * static_cast<size_t>(geo.tiles_y), seeds, obl, ao};
  if (bits && (instance == nullptr || normal == nullptr)) {
    throw std::logic_error("enqueue_pixel_stage: the obliquity and plan bits need the plan pass's instances and "
                           "normals");
  }
  if (st.params.band != 0 && (instance == nullptr || state == nullptr)) {
    throw std::logic_error("enqueue_pixel_stage: the band needs the plan pass's instances and shadow states");
  }
  if (seeds && (instance == nullptr || (st.params.n_bits > 0 && state == nullptr))) {
    throw std::logic_error("enqueue_pixel_stage: the band N's seeds need the plan pass's instances and shadow states");
  }
  if (st.params.pixel != 0 && !seeds) throw std::logic_error("enqueue_pixel_stage: the per-pixel band needs the seeds");
  if (!b.rays_ready) {
    Par<S>::upload(b.rays.ensure(3 * n), geo.rays.data(), 3 * n * sizeof(double), stream);
    b.rays_ready = true;
  }
  const PixelPlan pp{st.params,
                     depth,
                     stride,
                     b.rays.data(),
                     n,
                     upload_vector(b.shadow_lights, st.shadow_lights, stream),
                     upload_vector(b.shadow_bodies, st.shadow_bodies, stream),
                     upload_vector(b.moving, st.moving, stream),
                     upload_vector(b.spheres, st.spheres, stream),
                     b.k.ensure(n),
                     b.reach.ensure(n),
                     b.flags.ensure(n),
                     pass ? instance : nullptr,
                     pass ? state : nullptr,
                     upload_vector(b.movers, st.movers, stream),
                     st.params.band != 0 ? b.kcone.ensure(n) : nullptr,
                     st.params.band != 0 ? b.ksource.ensure(n) : nullptr,
                     st.params.band != 0 ? b.sreach.ensure(n) : nullptr,
                     seeds ? b.seed.ensure(n) : nullptr,
                     seeds ? carried : nullptr,
                     st.params.pixel != 0 ? b.seed_m.ensure(n) : nullptr,
                     st.params.pixel != 0 ? b.seed_kind.ensure(n) : nullptr,
                     bits ? normal : nullptr,
                     bits ? b.pv_plan.ensure(n) : nullptr,
                     obl ? b.k_base.ensure(n) : nullptr,
                     st.params.pv != 0 ? b.pv_width.ensure(n) : nullptr,
                     ao ? upload_vector(b.ao, st.ao, stream) : nullptr,
                     band0 ? b.kcone0.ensure(n) : nullptr,
                     band0 ? b.ksource0.ensure(n) : nullptr,
                     band0 ? b.sreach0.ensure(n) : nullptr};
  Par<S>::for_each(n, pp, stream);
  uint8_t* t = b.tiles.ensure(ts.bytes());
  const bool band = st.params.band != 0;
  Par<S>::for_each(ts.tiles,
                   TileReduce{b.k.data(), b.reach.data(), b.flags.data(), geo.width, geo.height, geo.tiles_x,
                              st.params.n_moving > 0, reinterpret_cast<double*>(t),
                              reinterpret_cast<int32_t*>(t + ts.reach()), t + ts.depend(),
                              reinterpret_cast<uint32_t*>(t + ts.shadow()), reinterpret_cast<uint32_t*>(t + ts.hit()),
                              band ? b.kcone.data() : nullptr, band ? b.ksource.data() : nullptr,
                              band ? b.sreach.data() : nullptr,
                              band ? reinterpret_cast<double*>(t + ts.kcone()) : nullptr,
                              band ? reinterpret_cast<double*>(t + ts.ksource()) : nullptr,
                              band ? reinterpret_cast<int32_t*>(t + ts.sreach()) : nullptr,
                              band ? reinterpret_cast<uint32_t*>(t + ts.cone()) : nullptr,
                              seeds ? b.seed.data() : nullptr,
                              seeds ? carried : nullptr,
                              seeds ? reinterpret_cast<int32_t*>(t + ts.seed()) : nullptr,
                              seeds ? reinterpret_cast<uint32_t*>(t + ts.seeds_count()) : nullptr,
                              seeds ? reinterpret_cast<uint32_t*>(t + ts.carried()) : nullptr,
                              st.params.crop_x0,
                              st.params.crop_y0,
                              st.params.crop_x1,
                              st.params.crop_y1,
                              obl ? b.k_base.data() : nullptr,
                              obl ? reinterpret_cast<double*>(t + ts.k_base()) : nullptr,
                              ao ? reinterpret_cast<uint32_t*>(t + ts.ao_count()) : nullptr,
                              band0 ? b.kcone0.data() : nullptr,
                              band0 ? b.ksource0.data() : nullptr,
                              band0 ? b.sreach0.data() : nullptr,
                              band0 ? reinterpret_cast<double*>(t + ts.kcone0()) : nullptr,
                              band0 ? reinterpret_cast<double*>(t + ts.ksource0()) : nullptr,
                              band0 ? reinterpret_cast<int32_t*>(t + ts.sreach0()) : nullptr},
                   stream);
}

// The tile results of enqueue_pixel_stage (one download, one sync); ao: P2-F F9b's AO receivers (params.n_ao > 0).
template <Space S>
TileTotals collect_pixel_stage(const CameraGeometry& geo, PlanBuffers<S>& b, Stream stream, bool band = false,
                               bool seeds = false, bool obliquity = false, bool ao = false) {
  const TileSlots ts{static_cast<size_t>(geo.tiles_x) * static_cast<size_t>(geo.tiles_y), seeds, obliquity, ao};
  std::vector<uint8_t> host(ts.bytes());
  Par<S>::download(host.data(), b.tiles.data(), host.size(), stream);
  TileTotals out;
  out.k.resize(ts.tiles);
  out.reach.resize(ts.tiles);
  out.depend.assign(host.begin() + static_cast<std::ptrdiff_t>(ts.depend()), host.end());
  std::memcpy(out.k.data(), host.data(), ts.tiles * sizeof(double));
  std::memcpy(out.reach.data(), host.data() + ts.reach(), ts.tiles * sizeof(int32_t));
  if (band) {
    out.kcone.resize(ts.tiles);
    out.ksource.resize(ts.tiles);
    out.sreach.resize(ts.tiles);
    out.cone.resize(ts.tiles);
    std::memcpy(out.kcone.data(), host.data() + ts.kcone(), ts.tiles * sizeof(double));
    std::memcpy(out.ksource.data(), host.data() + ts.ksource(), ts.tiles * sizeof(double));
    std::memcpy(out.sreach.data(), host.data() + ts.sreach(), ts.tiles * sizeof(int32_t));
    std::memcpy(out.cone.data(), host.data() + ts.cone(), ts.tiles * sizeof(uint32_t));
  }
  for (size_t i = 0; i < ts.tiles; ++i) {
    uint32_t shadow = 0, hit = 0;
    std::memcpy(&shadow, host.data() + ts.shadow() + i * sizeof(uint32_t), sizeof(uint32_t));
    std::memcpy(&hit, host.data() + ts.hit() + i * sizeof(uint32_t), sizeof(uint32_t));
    out.shadow_pixels += shadow;
    out.hit_pixels += hit;
  }
  if (obliquity) {
    out.k_base.resize(ts.tiles);
    std::memcpy(out.k_base.data(), host.data() + ts.k_base(), ts.tiles * sizeof(double));
  }
  for (size_t i = 0; ao && i < ts.tiles; ++i) {
    uint32_t count = 0;
    std::memcpy(&count, host.data() + ts.ao_count() + i * sizeof(uint32_t), sizeof(uint32_t));
    out.ao_pixels += count;
  }
  if (obliquity && band) {  // P2-D D1f
    out.kcone0.resize(ts.tiles);
    out.ksource0.resize(ts.tiles);
    out.sreach0.resize(ts.tiles);
    std::memcpy(out.kcone0.data(), host.data() + ts.kcone0(), ts.tiles * sizeof(double));
    std::memcpy(out.ksource0.data(), host.data() + ts.ksource0(), ts.tiles * sizeof(double));
    std::memcpy(out.sreach0.data(), host.data() + ts.sreach0(), ts.tiles * sizeof(int32_t));
  }
  if (seeds) {
    out.seed.resize(ts.tiles);
    out.carried.resize(ts.tiles);
    std::memcpy(out.seed.data(), host.data() + ts.seed(), ts.tiles * sizeof(int32_t));
    std::memcpy(out.carried.data(), host.data() + ts.carried(), ts.tiles * sizeof(uint32_t));
    for (size_t i = 0; i < ts.tiles; ++i) {
      uint32_t count = 0;
      std::memcpy(&count, host.data() + ts.seeds_count() + i * sizeof(uint32_t), sizeof(uint32_t));
      out.seed_pixels += count;
      out.carried_seeds += out.carried[i];
    }
  }
  return out;
}

}  // namespace balla::engine
