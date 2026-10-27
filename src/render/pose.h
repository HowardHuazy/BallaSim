// Exact poses of one pose slot (camera, time) in fp64, stored camera-relative in fp32 (DESIGN §5.3,
// §6.9; G6 recommendation 1). All pixels sampled at one time share their poses, so the K = 6 spline
// is evaluated once per (slot, body) instead of once per ray, and the fp32 rows hold camera-object
// distances (<= a few metres) rather than world coordinates (residual <= 2e-4 px, M2 §E.1).
//
// BALLA_HD so the same code can fill the table on the host or the device (bitwise equal under the
// FP policy, as tests/gpu/test_spline_device.cpp shows for the spline itself).
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/math/quat.h"
#include "core/math/vec.h"
#include "core/time/time.h"
#include "core/traj/pieces.h"
#include "core/traj/spline.h"
#include "render/scene_view.h"

namespace balla::render {

using Vec3d = Vec3<double>;
using Mat3d = Mat3<double>;

// X_A = r X_B + p for T_A_B.
struct RigidPose {
  Mat3d r;
  Vec3d p;
};

// One piece of a piecewise trajectory (C1 "pieces", blender.md §4.7.7): the first time it serves, its knot origin and
// the spline of its own control rows.
using PieceView = traj::SplinePiece<double>;

struct TrajectoryView {
  traj::SplineView<double> spline;  // piece 0's when piecewise
  int64_t t0_ns;
  int32_t order;  // 4 or 6
  int64_t span_begin_ns, span_end_ns;  // valid query interval [begin, end]
  // num_pieces <= 1: the single spline above (today's path); else the pieces, right-continuous, by traj::piece_index.
  const PieceView* pieces = nullptr;
  int32_t num_pieces = 0;
};

// Body state at one time: world pose, world angular velocity, world velocity of the body origin.
struct BodyState {
  RigidPose pose;
  Vec3d omega;
  Vec3d v;
};

struct PoseCameraDef {
  int32_t rig_trajectory;
  RigidPose rig_cam;  // T_rig_cam
};

struct PoseGeometryDef {
  int32_t trajectory;  // -1 for the static world and split static bodies (scans.md §4.2)
  Vec3d centre;        // bounding-sphere centre in the body frame (movers), the world frame (split static bodies)
  float radius;        // +inf for the static world
};

struct PoseLightDef {
  int32_t trajectory;  // body trajectory, or -1 for a light fixed in the world
  RigidPose attach;    // T_world_body of a static body (identity for world lights)
  Vec3d position;      // in the attachment frame
  Vec3d direction;     // unit propagation direction in the attachment frame
  Vec3d axis{0.0, 0.0, 0.0};  // area lights (P2-F F8a-1): the unit in-plane u axis in the attachment frame; zero otherwise
};

struct PoseSceneView {
  const TrajectoryView* trajectories;
  const PoseCameraDef* cameras;
  const PoseGeometryDef* geometries;
  int32_t num_geometries;
  const PoseLightDef* lights;
  int32_t num_lights;
};

BALLA_HD inline Mat3d transpose(const Mat3d& a) {
  return {{{a.m[0][0], a.m[1][0], a.m[2][0]}, {a.m[0][1], a.m[1][1], a.m[2][1]},
           {a.m[0][2], a.m[1][2], a.m[2][2]}}};
}

BALLA_HD inline Mat3d operator*(const Mat3d& a, const Mat3d& b) {
  Mat3d c;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c.m[i][j] = (a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j]) + a.m[i][2] * b.m[2][j];
    }
  }
  return c;
}

// False outside the trajectory's span. A piecewise trajectory evaluates the piece serving t_ns (the last whose begin is
// <= t_ns: a time at a breakpoint takes the right piece) with the same int64 search on host and device (L1).
BALLA_HD inline bool body_state(const TrajectoryView& tv, int64_t t_ns, BodyState* out) {
  if (t_ns < tv.span_begin_ns || t_ns > tv.span_end_ns) return false;
  traj::SegmentTime st{};
  const traj::SplineView<double>* spline = &tv.spline;
  if (tv.num_pieces > 1) {
    const PieceView& piece = tv.pieces[traj::piece_index(tv.pieces, tv.num_pieces, t_ns)];
    spline = &piece.spline;
    if (!traj::locate(t_ns, piece.t0_ns, spline->h_ns, spline->num_segments, &st)) return false;
  } else if (!traj::locate(t_ns, tv.t0_ns, tv.spline.h_ns, tv.spline.num_segments, &st)) {
    return false;
  }
  const traj::Kinematics<double> k = tv.order == 4 ? traj::evaluate<4>(*spline, st) : traj::evaluate<6>(*spline, st);
  out->pose.r = to_matrix(k.q);
  out->pose.p = k.p;
  out->omega = out->pose.r * k.omega;  // body-frame omega -> world
  out->v = k.v;
  return true;
}

BALLA_HD inline void store3(float* dst, Vec3d v) {
  dst[0] = static_cast<float>(v.x);
  dst[1] = static_cast<float>(v.y);
  dst[2] = static_cast<float>(v.z);
}

BALLA_HD inline Float4 to_float4(Vec3d v, float w) {
  return {static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), w};
}

// Time slots of a window pose table (ENGINE_CPP §1.4 item 2). Asynchronous slots lie at
// t_w + (slot << shift); fixed-rate slots (period_ns > 0): slot 0 at t_w, slots 0 < slot < m at the
// absolute multiples (floor(t_w / period) + slot) period inside (t_w, t_w + W), and slot m at t_w + W
// (engine/sampling.h fixed_dt).
struct WindowClock {
  int64_t t_w;
  int32_t shift;
  uint32_t m;
  int64_t period_ns;
};

BALLA_HD inline int64_t slot_time(const WindowClock& c, uint32_t slot) {
  if (c.period_ns <= 0) return c.t_w + (static_cast<int64_t>(slot) << c.shift);
  if (slot == 0) return c.t_w;
  return slot >= c.m ? c.t_w + time::kWindowNs : (time::floor_div(c.t_w, c.period_ns) + slot) * c.period_ns;
}

BALLA_HD inline uint32_t num_slots(const WindowClock& c) {
  return c.period_ns > 0 ? c.m + 1 : static_cast<uint32_t>((time::kWindowNs >> c.shift) + 1);
}

// Fills the num_geometries pose rows and num_lights light rows of camera `cam` at t_ns. False if
// t_ns lies outside the span of the camera rig's or any mover's trajectory.
BALLA_HD inline bool compute_slot(const PoseSceneView& s, uint32_t cam, int64_t t_ns, PoseRow* rows,
                                  LightRow* light_rows) {
  const PoseCameraDef& c = s.cameras[cam];
  BodyState rig;
  if (!body_state(s.trajectories[c.rig_trajectory], t_ns, &rig)) return false;
  // T_W_C = T_W_rig T_rig_C; camera velocity v_rig + omega x (R_W_rig p_rig_C) (M2 C12, B10).
  const Mat3d r_wc = rig.pose.r * c.rig_cam.r;
  const Vec3d lever = rig.pose.r * c.rig_cam.p;
  const Vec3d p_wc = rig.pose.p + lever;
  const Vec3d v_c = rig.v + cross(rig.omega, lever);
  const Mat3d r_cw = transpose(r_wc);
  const Vec3d omega_c = r_cw * rig.omega;  // camera angular velocity in the camera frame

  for (int32_t g = 0; g < s.num_geometries; ++g) {
    const PoseGeometryDef& def = s.geometries[g];
    PoseRow& row = rows[g];
    Mat3d r_gc;
    Vec3d p_gc, omega_rel, v_rel, centre;
    if (def.trajectory < 0) {
      // Static world: T_world_cam itself; static points move as X' = -omega_C x X - R_CW v_C (M2 D3). A split static
      // body (finite radius, scans.md §4.2) is in the world frame too, its sphere centred at R_CW (c_W - p_WC).
      r_gc = r_wc;
      p_gc = p_wc;
      omega_rel = -omega_c;
      v_rel = -(r_cw * v_c);
      centre = def.radius < infinity<float>() ? r_cw * (def.centre - p_wc) : Vec3d{0.0, 0.0, 0.0};
    } else {
      BodyState b;
      if (!body_state(s.trajectories[def.trajectory], t_ns, &b)) return false;
      const Mat3d r_bw = transpose(b.pose.r);
      r_gc = r_bw * r_wc;
      p_gc = r_bw * (p_wc - b.pose.p);
      // Relative twist (M2 D3): omega_rel = R_CW (omega_B - omega_C), p_CO = R_CW (p_WB - p_WC),
      // p_CO' = R_CW (v_B - v_C) - omega_C x p_CO, v_rel = p_CO' - omega_rel x p_CO.
      omega_rel = r_cw * (b.omega - rig.omega);
      const Vec3d p_co = r_cw * (b.pose.p - p_wc);
      const Vec3d p_co_dot = r_cw * (b.v - v_c) - cross(omega_c, p_co);
      v_rel = p_co_dot - cross(omega_rel, p_co);
      centre = r_cw * ((b.pose.r * def.centre + b.pose.p) - p_wc);
    }
    const double p[3] = {p_gc.x, p_gc.y, p_gc.z};
    for (int i = 0; i < 3; ++i) {
      row.m[i] = {static_cast<float>(r_gc.m[i][0]), static_cast<float>(r_gc.m[i][1]), static_cast<float>(r_gc.m[i][2]),
                  static_cast<float>(p[i])};
    }
    row.sphere = to_float4(centre, def.radius);
    row.omega = to_float4(omega_rel, 0.0f);
    row.v = to_float4(v_rel, 0.0f);
  }

  for (int32_t j = 0; j < s.num_lights; ++j) {
    const PoseLightDef& def = s.lights[j];
    RigidPose wb = def.attach;
    if (def.trajectory >= 0) {
      BodyState b;
      if (!body_state(s.trajectories[def.trajectory], t_ns, &b)) return false;
      wb = b.pose;
    }
    store3(light_rows[j].position, r_cw * ((wb.r * def.position + wb.p) - p_wc));
    store3(light_rows[j].direction, r_cw * (wb.r * def.direction));
    if (def.axis.x != 0.0 || def.axis.y != 0.0 || def.axis.z != 0.0) {
      store3(light_rows[j].axis_u, r_cw * (wb.r * def.axis));
    } else {
      light_rows[j].axis_u[0] = light_rows[j].axis_u[1] = light_rows[j].axis_u[2] = 0.0f;  // point, spot, directional
    }
  }
  return true;
}

}  // namespace balla::render
