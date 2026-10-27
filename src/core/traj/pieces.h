// Piecewise trajectories (C1 Trajectory.pieces, version-2 feature "pieces"; docs/design/phase3/blender.md §3.7,
// §4.7.7): P uniform splines of one order, piece p serving [begin_p, begin_{p+1}) and the last one the rest of the
// span (closed at its end). Pieces are right-continuous: a time exactly at a breakpoint b takes the right piece, so
// ground truth at b is the right-hand limit.
//
// BALLA_HD, so host and device pick a time's piece with the same int64 search (bitwise equal, DESIGN §12.1 L1). Its
// users: _core.spline_eval_pieces_f64 (ballasim.traj.evaluate), the renderer's body_state (render/pose.h, host and
// device) and the planner's TrajectoryMotion (engine/motion.cpp), since Phase 3 BL-5b.
#pragma once

#include <stdint.h>

#include "core/hd.h"
#include "core/traj/spline.h"

namespace balla::traj {

// How far apart the two one-sided poses at the breakpoint b of a join labelled C0 or C1 may be: the left piece's pose
// at b (its support is closed there) and the right piece's, in position (m) and rotation angle (rad). 1e-8 lies about
// 10^6 above the fp64 rounding of a pose within the 16 m reach. The engine refuses a labelled join that jumps by more
// (engine/motion.h, a step in fact); the Blender motion fitter (scene/motion/fit_piecewise.h) moves each right piece
// rigidly onto its left piece's end pose, so its joins keep the pose to rounding (blender.md §4.7.6).
inline constexpr double kJoinTolM = 1e-8, kJoinTolRad = 1e-8;

template <class T>
struct SplinePiece {
  int64_t begin_ns;      // first time the piece serves
  int64_t t0_ns;         // its knot origin (<= begin_ns)
  SplineView<T> spline;  // nodes built from the piece's own control rows (its first node has d = dp = 0)
};

// The piece serving t_ns: the last p with pieces[p].begin_ns <= t_ns, by binary search over the strictly increasing
// begins; 0 for a time before every begin (piece 0 then serves its own support, as a single spline does). `Piece` is
// any type with an int64 `begin_ns` (SplinePiece, the renderer's piece view). num_pieces >= 1.
template <class Piece>
BALLA_HD inline int32_t piece_index(const Piece* pieces, int32_t num_pieces, int64_t t_ns) {
  int32_t lo = 0, hi = num_pieces;  // lo == 0 or pieces[lo].begin_ns <= t_ns; every p >= hi begins after t_ns
  while (hi - lo > 1) {
    const int32_t mid = lo + (hi - lo) / 2;
    if (pieces[mid].begin_ns <= t_ns) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// The piece serving t_ns and the time's segment in it; false outside that piece's spline support.
template <class T>
BALLA_HD inline bool locate_piece(const SplinePiece<T>* pieces, int32_t num_pieces, int64_t t_ns, int32_t* piece,
                                  SegmentTime* out) {
  const int32_t p = piece_index(pieces, num_pieces, t_ns);
  *piece = p;
  return locate(t_ns, pieces[p].t0_ns, pieces[p].spline.h_ns, pieces[p].spline.num_segments, out);
}

}  // namespace balla::traj
