// Oracle answers and their decoding (docs/design/phase3/blender.md §3.5, §3.8, §4.1, §4.7.2): the raw float32
// matrices Blender returns, the scene facts the motion session needs about the sampled bodies, and the decoded,
// time-stamped poses with their float32 rounding bounds.
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "scene/motion/time_map.h"
#include "scene/rotation.h"

namespace balla::scene::motion {

// One `poses` answer (§3.5): n samples of B bodies, as they crossed the pipe.
struct RawPoses {
  int32_t n = 0, bodies = 0;
  std::vector<int32_t> F;          // n, as requested
  std::vector<float> s;            // n, as requested
  std::vector<double> c_final;     // n: frame_current_final
  std::vector<float> m;            // n x B x 12: row-major 3x4 matrix_world (Blender units)
  std::vector<uint8_t> present;    // n x B
  std::vector<uint32_t> unknown;   // n: render-visible instances whose key is not in the exchange
  std::string unknown_keys;        // the add-on's description of the first such instances (<= 4 keys), for VIS-ANIM
};

// What the motion session knows about each sampled body column (filled from the converted statics, blender.md §3.8
// BodyStatic / CameraStatic; BL-3 and BL-9 build it).
enum class Role : uint8_t { kBody, kRig, kCarrier };

struct MotionLight {
  rot::V offset_m{0, 0, 0};  // light position in the carrier's frame (relative to its centre), metres
  bool spot = false, directional = false, casts_shadows = true;
};

struct MotionBody {
  Role role = Role::kBody;   // kBody: a mesh (it may also carry lights); kRig: a camera rig; kCarrier: meshless
  double rho_m = 0.0;        // mesh radius about the centre (kBody)
  rot::V centre_m{0, 0, 0};  // c (§4.1): the centre of S co in metres; the trajectory follows t + R c
  // S at the export frame (row-major; BodyStatic::S, §3.8); GEO-SCALEANIM when S(t) departs. Required for kBody and
  // kCarrier (decode and ErrorModel throw std::invalid_argument without it, so a missing S never skips the check);
  // optional for kRig, whose scale moves no ray (checked only when set).
  std::optional<std::array<double, 9>> S_ref;
  std::vector<MotionLight> lights;
  // A static body: its triangles are in MotionScene::static_vertices, whose exact distance the error model already takes
  // (StaticDistance), so the bounds of §4.7.2 that name the nearest moving body (d_scene, d_occ) skip its sphere.
  bool still = false;
};

struct MotionCamera {
  int32_t rig_body = 0;               // the body column of its rig (Role::kRig)
  rot::V t_rig_cam_m{0, 0, 0};        // camera centre in the rig frame, metres
  double gain_px_per_rad = 0.0;       // G_j (error_model.h angular_gain)
  double clip_start_m = 0.0;
};

struct MotionScene {
  double scale_length = 1.0;           // metres per Blender unit
  std::vector<MotionBody> bodies;      // the pipe's body columns, in order
  std::vector<MotionCamera> cameras;   // the simulated cameras
  std::vector<double> static_vertices;   // static geometry, world frame, metres (V x 3)
  std::vector<uint32_t> static_indices;  // T x 3
  std::array<double, 6> scene_box{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
  // the scene AABB (lo xyz, hi xyz; metres): static geometry and the bodies' motion; lo > hi: empty
};

// Decoded samples (sample-major, n x B for the per-body fields).
struct PoseBatch {
  int32_t bodies = 0;
  std::vector<int64_t> t_ns;             // n: to_ns(realized c'), never the target
  std::vector<float> c;                  // n: realized c'
  std::vector<rot::Q> q;                 // R_world_body (polar factor; w >= 0)
  std::vector<rot::V> p;                 // t + R c in metres
  std::vector<std::array<double, 9>> S;  // R^T A (row-major)
  std::vector<uint8_t> present;
  std::vector<double> dp_round;          // |1/2 ulp32(t)| scale_length + |c| dtheta_round (m)
  std::vector<double> dtheta_round;      // 3 * 1/2 ulp32(max |A_jk|) / sigma_min(S) (rad)
  // ErrorModel::weigh: e_px = w_pos |dp| + w_rot |dtheta| for this body at this sample (§4.7.2), its value at the
  // rounding bounds (floor_px with gain 1), and whether one of its lights came within 5 cm of a body (LGT-MOVE-NEAR)
  std::vector<double> w_pos, w_rot, floor1;
  std::vector<uint8_t> light_near;

  size_t size() const { return t_ns.size(); }
  size_t at(size_t i, int32_t b) const { return i * static_cast<size_t>(bodies) + static_cast<size_t>(b); }
  void append(const PoseBatch& o);
};

// c' = realized(F, s); t_ns = to_ns(c'); BRG-TIME unless c_final == reported(c') exactly; BRG-PROTO for a non-finite
// matrix entry; the polar decomposition A = R S (det A < 0: R = polar(-A), S = R^T A), GEO-SCALEANIM for a singular A
// or when |S - S_ref|_F > 1e-6 |S_ref|_F, metres, the recentred position and the rounding bounds. Absent bodies get the
// identity pose and no checks. std::invalid_argument for a kBody or kCarrier column without S_ref
// (check_scale_refs).
PoseBatch decode(const RawPoses& raw, const TimeMap& time, const MotionScene& scene);

// std::invalid_argument naming the first kBody or kCarrier column of `scene` without S_ref, or with a non-finite or
// zero one (GEO-SCALEANIM must hold for every mesh body and light carrier, §4.1).
void check_scale_refs(const MotionScene& scene);

}  // namespace balla::scene::motion
