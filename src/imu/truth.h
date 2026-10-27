// The IMU truth (docs/design/phase3/outputs.md §4.1, goal G1): the pure kinematics of the IMU frame I on the camera's
// rig body B, before any filter or device error. Header-only, host, fp64.
//
//   omega_I = R_BI^T k.omega
//   f_I     = R_BI^T [ R_WB(k.q)^T (k.a - g_W) + k.omega_dot x p_B + k.omega x (k.omega x p_B) ]
//   q_W_I   = k.q (x) q(R_BI),   p_W_I = k.p + R_WB p_B
//
// with k the rig spline's Kinematics (omega and omega_dot body-frame, v and a world-frame; core/traj/spline.h) and
// T_B_I = T_rig_cam T_cam_imu. The body-frame omega is used directly: render::body_state turns it into the world
// frame for rendering, which would cost 8.4 deg/s on EVIMO2 (§2.1, §2.4).
//
// A piecewise rig (C1 Trajectory.pieces, P3-E2; A11) is evaluated on the piece serving the time, right-continuous as
// render::body_state picks it (traj::piece_index); where_left gives the left limit at a join (§4.3.4).
#pragma once

#include <cmath>
#include <cstdint>
#include <span>
#include <string>

#include "core/math/quat.h"
#include "core/math/vec.h"
#include "core/traj/pieces.h"
#include "core/traj/spline.h"
#include "imu/params.h"
#include "render/pose.h"

namespace balla::imu {

// T_B_I: the IMU's rotation R_BI and origin p_B in the rig body, and q(R_BI).
struct Mount {
  Mat3d R_BI;
  Vec3d p_B;
  Quat<double> q_BI;
};

struct Truth {
  Vec3d omega_I, f_I;
  Quat<double> q_W_I;
  Vec3d p_W_I;
};

// A place on the rig trajectory: piece (0 for a single spline), segment of that piece's spline and u.
struct Where {
  int32_t piece, segment;
  double u;
};

// R_BI^T v: a body-frame vector in IMU axes.
inline Vec3d to_imu(const Mount& m, Vec3d v) {
  const auto& r = m.R_BI.m;
  return {(r[0][0] * v.x + r[1][0] * v.y) + r[2][0] * v.z, (r[0][1] * v.x + r[1][1] * v.y) + r[2][1] * v.z,
          (r[0][2] * v.x + r[1][2] * v.y) + r[2][2] * v.z};
}

// T_B_I = T_rig_cam T_cam_imu, with R_cam_imu re-orthonormalized (rotation_of). Refused when T_cam_imu fails the
// §6 rigidity rule or T_rig_cam is not finite.
inline Mount mount_of(const render::RigidPose& T_rig_cam, const ImuParams& p) {
  if (const std::string why = detail::rigidity_error(p.T_cam_imu); !why.empty()) {
    throw Refused("imu.T_cam_imu " + why);
  }
  for (int i = 0; i < 3; ++i) {
    const double row[4] = {T_rig_cam.r.m[i][0], T_rig_cam.r.m[i][1], T_rig_cam.r.m[i][2],
                           i == 0 ? T_rig_cam.p.x : (i == 1 ? T_rig_cam.p.y : T_rig_cam.p.z)};
    for (double x : row) {
      if (!std::isfinite(x)) throw Refused("T_rig_cam has a non-finite entry");
    }
  }
  const Mat3d r_ci = rotation_of(p);
  const Vec3d p_ci = translation_of(p);
  Mount m;
  m.R_BI = render::operator*(T_rig_cam.r, r_ci);
  m.p_B = T_rig_cam.r * p_ci + T_rig_cam.p;
  m.q_BI = from_matrix(m.R_BI);
  return m;
}

// Joins of a piecewise rig (C1 Trajectory.pieces[p].join, p > 0): a step (the pose jumps), C0 (the velocity jumps)
// and C1 (the acceleration jumps); piece 0 has 0.
inline constexpr int8_t kJoinStep = -1, kJoinC0 = 0, kJoinC1 = 1;

// The spline of piece `piece` (the single spline when the rig has no pieces).
inline const traj::SplineView<double>& spline_of(const render::TrajectoryView& rig, int32_t piece) {
  return rig.num_pieces > 1 ? rig.pieces[piece].spline : rig.spline;
}

// What evaluation relies on in a piecewise rig (num_pieces > 1) and its joins (one per piece), as a message naming
// the first failure ("" when none): every piece's spline has nodes, segments and h >= 1; the begins strictly increase;
// each piece's support [t0_p, t0_p + segments h_p] covers its interval [begin_p, begin_{p+1}] (the last one's up to
// the span's end), so the left limit at a join lies inside it (R-BL :662); joins in {-1, 0, 1}, piece 0's 0. A rig
// without pieces takes no joins or the one 0.
inline std::string pieces_error(const render::TrajectoryView& rig, std::span<const int8_t> joins) {
  if (rig.num_pieces <= 1) return joins.size() <= 1 && (joins.empty() || joins[0] == 0) ? "" : "joins without pieces";
  if (rig.pieces == nullptr) return "num_pieces > 1 without pieces";
  if (joins.size() != static_cast<size_t>(rig.num_pieces)) return "joins needs one entry per piece";
  for (int32_t p = 0; p < rig.num_pieces; ++p) {
    const render::PieceView& pc = rig.pieces[p];
    const std::string at = "piece " + std::to_string(p) + ": ";
    if (pc.spline.nodes == nullptr || pc.spline.num_segments < 1 || pc.spline.h_ns < 1) {
      return at + "needs at least one segment";
    }
    if (p > 0 && pc.begin_ns <= rig.pieces[p - 1].begin_ns) return at + "begin_ns must exceed the previous piece's";
    if (p == 0 ? joins[0] != 0 : (joins[p] < kJoinStep || joins[p] > kJoinC1)) {
      return at + "join must be -1, 0 or 1 (piece 0: 0)";
    }
    const int64_t end = p + 1 < rig.num_pieces ? rig.pieces[p + 1].begin_ns : rig.span_end_ns;
    __extension__ typedef __int128 i128;  // t0 + segments h may leave int64 for a far t0
    const i128 hi = static_cast<i128>(pc.t0_ns) + static_cast<i128>(pc.spline.num_segments) * pc.spline.h_ns;
    if (!(pc.t0_ns <= pc.begin_ns && static_cast<i128>(end) <= hi)) {
      return at + "its spline's support does not cover [begin_ns, " + std::to_string(end) + "]";
    }
  }
  return "";
}

// Right-continuous place of the instant t_ns + frac_ns, frac in [0, 1): the piece serving t_ns (a node exactly at a
// join takes the right piece; traj::piece_index, R-BL :1250), the segment by traj::locate on that piece's spline and
// u = (r + frac) / h. Refused outside the trajectory's span [span_begin_ns, span_end_ns].
inline Where where_right(const render::TrajectoryView& rig, int64_t t_ns, double frac_ns) {
  traj::SegmentTime st{};
  const bool inside = t_ns >= rig.span_begin_ns && (t_ns < rig.span_end_ns || (t_ns == rig.span_end_ns && frac_ns == 0));
  int32_t piece = 0;
  int64_t t0 = rig.t0_ns;
  if (inside && rig.num_pieces > 1) {
    piece = traj::piece_index(rig.pieces, rig.num_pieces, t_ns);
    t0 = rig.pieces[piece].t0_ns;
  }
  const traj::SplineView<double>& spline = spline_of(rig, piece);
  if (!inside || !traj::locate(t_ns, t0, spline.h_ns, spline.num_segments, &st)) {
    throw Refused("rig trajectory: time " + std::to_string(t_ns) + " ns + " + config::format_double(frac_ns) +
                  " ns is outside its span [" + std::to_string(rig.span_begin_ns) + ", " +
                  std::to_string(rig.span_end_ns) + "] ns");
  }
  return {piece, st.segment, (static_cast<double>(st.r_ns) + frac_ns) / static_cast<double>(spline.h_ns)};
}

// The left limit at the join t_ns = begin_p (p >= 1) of a piecewise rig: piece p - 1 at its last segment covering
// t_ns, u in (0, 1] (a knot ends the segment before it: u = 1). The piece's support is closed and covers begin_p
// (pieces_error; R-BL :662). Refused unless t_ns is a piece's begin after the first.
inline Where where_left(const render::TrajectoryView& rig, int64_t t_ns) {
  const int32_t p = rig.num_pieces > 1 ? traj::piece_index(rig.pieces, rig.num_pieces, t_ns) : 0;
  if (p < 1 || rig.pieces[p].begin_ns != t_ns) {
    throw Refused("rig trajectory: " + std::to_string(t_ns) + " ns is not a join (the begin of a piece after the first)");
  }
  const render::PieceView& left = rig.pieces[p - 1];
  traj::SegmentTime st{};
  if (!traj::locate(t_ns, left.t0_ns, left.spline.h_ns, left.spline.num_segments, &st)) {
    throw Refused("rig trajectory: piece " + std::to_string(p - 1) + "'s support does not reach its end " +
                  std::to_string(t_ns) + " ns");
  }
  // t_ns > begin_{p-1} >= t0_{p-1}, so a knot (r = 0) is never the support's first.
  if (st.r_ns == 0) return {p - 1, st.segment - 1, 1.0};
  return {p - 1, st.segment, static_cast<double>(st.r_ns) / static_cast<double>(left.spline.h_ns)};
}

// The rig's kinematics at a place (evaluate_u<6> or <4> by rig.order on that piece's spline).
inline traj::Kinematics<double> kinematics_at(const render::TrajectoryView& rig, const Where& w) {
  const traj::SplineView<double>& spline = spline_of(rig, w.piece);
  return rig.order == 4 ? traj::evaluate_u<4>(spline, w.segment, w.u) : traj::evaluate_u<6>(spline, w.segment, w.u);
}

// §4.1 at a place of the rig trajectory; g_W is the world gravity in m/s^2 (§4.1's priority, resolved by the caller).
inline Truth truth_at(const render::TrajectoryView& rig, const Mount& m, const Vec3d& g_W, const Where& w) {
  const traj::Kinematics<double> k = kinematics_at(rig, w);
  // Inside holds omega = omega_dot = a = 0 exactly, so both lever terms are exact zeros and f_I is
  // R_BI^T (R_WB^T (-g_W)) bitwise (U3).
  const Vec3d body = (rotate_inverse(k.q, k.a - g_W) + cross(k.omega_dot, m.p_B)) + cross(k.omega, cross(k.omega, m.p_B));
  return {to_imu(m, k.omega), to_imu(m, body), k.q * m.q_BI, k.p + rotate(k.q, m.p_B)};
}

// §4.3.4: the specific-force impulse of a C0 join at t_b = begin_p (m/s, IMU axes),
//   J_I = R_BI^T [ R_WB(t_b)^T dv_W + dw_B x p_B ],
// with dv_W and dw_B the jumps of the rig's world velocity and body-frame rate (right limit minus left limit): the
// jump of the IMU origin's velocity v + R_WB (omega x p_B), in IMU axes. R_WB is the right piece's (continuous at C0).
inline Vec3d join_impulse(const render::TrajectoryView& rig, const Mount& m, int64_t t_b_ns) {
  const traj::Kinematics<double> l = kinematics_at(rig, where_left(rig, t_b_ns));
  const traj::Kinematics<double> r = kinematics_at(rig, where_right(rig, t_b_ns, 0.0));
  return to_imu(m, rotate_inverse(r.q, r.v - l.v) + cross(r.omega - l.omega, m.p_B));
}

}  // namespace balla::imu
