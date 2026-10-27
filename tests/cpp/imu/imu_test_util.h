// Helpers of the IMU unit tests (tests/cpp/imu): owned rig trajectories and ideal-device parameters.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "core/math/quat.h"
#include "core/math/so3.h"
#include "core/math/vec.h"
#include "core/traj/spline_build.h"
#include "imu/params.h"
#include "imu/simulator.h"
#include "render/pose.h"

namespace balla::imu::test {

// A rig trajectory with its nodes; view() points into them (keep the Rig alive and unmoved while it is used).
struct Rig {
  std::vector<traj::SplineNode<double>> nodes;
  int64_t t0_ns = 0;
  int32_t h_ns = 5'000'000;
  int32_t order = 6;
  int64_t span_begin_ns = 0, span_end_ns = 0;  // 0, 0: the whole support

  render::TrajectoryView view() const {
    const int32_t segments = static_cast<int32_t>(nodes.size()) - order + 1;
    const int64_t end = t0_ns + int64_t{segments} * h_ns;
    return {{nodes.data(), segments, h_ns, 1e9 / h_ns}, t0_ns, order,
            span_begin_ns == 0 && span_end_ns == 0 ? t0_ns : span_begin_ns,
            span_begin_ns == 0 && span_end_ns == 0 ? end : span_end_ns};
  }
};

inline Rig make_rig(const std::vector<Quat<double>>& q, const std::vector<Vec3d>& p, int32_t order, int32_t h_ns,
                    int64_t t0_ns = 0) {
  Rig r;
  r.nodes = traj::build_spline_nodes(q, p, {0, 0, 0});
  r.order = order;
  r.h_ns = h_ns;
  r.t0_ns = t0_ns;
  return r;
}

// Smooth random controls in the style of R-DOC C3: a random walk of rotation vectors (scale_w per knot) and a
// doubly integrated random walk of positions (scale_p), plus an offset so positions are not near 0.
inline Rig random_rig(uint64_t seed, int n, double scale_w, double scale_p, int32_t order, int32_t h_ns) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g;
  std::vector<Quat<double>> q;
  std::vector<Vec3d> p;
  Vec3d rv{0, 0, 0}, v{0, 0, 0}, x{1.5, -0.7, 1.2};
  for (int i = 0; i < n; ++i) {
    rv += scale_w * Vec3d{g(rng), g(rng), g(rng)};
    Quat<double> qi = so3::exp(rv);
    if (!q.empty() && (qi.w * q.back().w + qi.x * q.back().x + qi.y * q.back().y + qi.z * q.back().z) < 0) {
      qi = {-qi.w, -qi.x, -qi.y, -qi.z};
    }
    q.push_back(qi);
    v += scale_p * Vec3d{g(rng), g(rng), g(rng)};
    x += v;
    p.push_back(x);
  }
  return make_rig(q, p, order, h_ns);
}

// A piecewise rig (C1 Trajectory.pieces, A11): each piece's nodes from its own control rows, one after the other (as
// SceneData builds them); view() and the joins point into the object (do not copy it).
struct PieceSpec {
  int64_t begin_ns, t0_ns;
  int32_t h_ns;
  int8_t join;                  // -1 step, 0 C0, 1 C1 (piece 0: 0)
  std::vector<Quat<double>> q;  // the piece's control rows
  std::vector<Vec3d> p;
};

struct PiecewiseRig {
  std::vector<traj::SplineNode<double>> nodes;
  std::vector<render::PieceView> pieces;
  std::vector<int8_t> joins;
  int32_t order = 6;
  int64_t span_begin_ns = 0, span_end_ns = 0;
  PiecewiseRig() = default;
  PiecewiseRig(const PiecewiseRig&) = delete;
  PiecewiseRig& operator=(const PiecewiseRig&) = delete;

  render::TrajectoryView view() const {
    render::TrajectoryView v{pieces[0].spline, pieces[0].t0_ns, order, span_begin_ns, span_end_ns};
    v.pieces = pieces.data();
    v.num_pieces = static_cast<int32_t>(pieces.size());
    return v;
  }
};

inline void make_piecewise(PiecewiseRig& r, const std::vector<PieceSpec>& specs, int32_t order, int64_t span_begin_ns,
                           int64_t span_end_ns) {
  r.order = order;
  r.span_begin_ns = span_begin_ns;
  r.span_end_ns = span_end_ns;
  size_t rows = 0;
  for (const PieceSpec& s : specs) rows += s.q.size();
  r.nodes.reserve(rows);
  std::vector<size_t> first;
  for (const PieceSpec& s : specs) {
    first.push_back(r.nodes.size());
    const auto part = traj::build_spline_nodes(s.q, s.p, {0, 0, 0});
    r.nodes.insert(r.nodes.end(), part.begin(), part.end());
    r.joins.push_back(s.join);
  }
  for (size_t k = 0; k < specs.size(); ++k) {
    const int32_t segments = static_cast<int32_t>(specs[k].q.size()) - order + 1;
    r.pieces.push_back({specs[k].begin_ns, specs[k].t0_ns,
                        {r.nodes.data() + first[k], segments, specs[k].h_ns, 1e9 / specs[k].h_ns}});
  }
}

// An ideal device sampled at rate_hz from t_begin (start_phase zero), with both first-order filters on or off.
inline ImuParams ideal(double rate_hz, bool filters) {
  ImuParams p;
  p.enabled = true;
  p.rate_hz = rate_hz;
  p.start_phase = StartPhase::kZero;
  if (filters) p.gyro.filter = p.accel.filter = Filter::kFirstOrder;
  return p;
}

// T_cam_imu = [R | t], row-major.
inline std::array<double, 16> transform(const Mat3d& r, const Vec3d& t) {
  return {r.m[0][0], r.m[0][1], r.m[0][2], t.x, r.m[1][0], r.m[1][1], r.m[1][2], t.y,
          r.m[2][0], r.m[2][1], r.m[2][2], t.z, 0,         0,         0,         1};
}

inline ImuInput input(const Rig& rig, int64_t t_begin_ns, int64_t t_end_ns,
                      render::RigidPose T_rig_cam = {{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {0, 0, 0}},
                      Vec3d g_W = {0, 0, -9.80665}) {
  ImuInput in;
  in.rig = rig.view();
  in.T_rig_cam = T_rig_cam;
  in.g_W = g_W;
  in.t_begin_ns = t_begin_ns;
  in.t_end_ns = t_end_ns;
  in.t_offset_us = INT64_MIN / 1000;  // no event file: no stamp is dropped
  return in;
}

// As above, for a piecewise rig: its joins come with it.
inline ImuInput input(const PiecewiseRig& rig, int64_t t_begin_ns, int64_t t_end_ns,
                      render::RigidPose T_rig_cam = {{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {0, 0, 0}},
                      Vec3d g_W = {0, 0, -9.80665}) {
  ImuInput in;
  in.rig = rig.view();
  in.joins = rig.joins;
  in.T_rig_cam = T_rig_cam;
  in.g_W = g_W;
  in.t_begin_ns = t_begin_ns;
  in.t_end_ns = t_end_ns;
  in.t_offset_us = INT64_MIN / 1000;
  return in;
}

}  // namespace balla::imu::test
