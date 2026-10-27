// Host-side construction of spline nodes from fp64 control poses (M2 §A.3).
#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/math/quat.h"
#include "core/math/so3.h"
#include "core/math/vec.h"
#include "core/traj/spline.h"

namespace balla::traj {

namespace detail {

// a == b or a == -b componentwise: the same rotation, stored identically up to the sign.
inline bool same_rotation_exactly(const Quat<double>& a, const Quat<double>& b) {
  return (a.w == b.w && a.x == b.x && a.y == b.y && a.z == b.z) ||
         (a.w == -b.w && a.x == -b.x && a.y == -b.y && a.z == -b.z);
}

}  // namespace detail

// q: control rotations (any sign, normalized here), p: control positions; both fp64. Positions are
// stored relative to `origin`, increments are differenced in fp64 before any later cast.
//
// Equal consecutive controls give exactly zero increments, so a spline is exactly still wherever
// its K controls are equal (holds, M2 §D.5): dp_c = p_c - p_{c-1} is 0 by IEEE subtraction, and
// d_c is set to 0 when the normalized rotations are equal (either sign). Log(R_{c-1}^T R_c)
// itself would leave ~1e-16 rad: the vector part of the quaternion product q_{c-1}* q_c does not
// cancel exactly, because its components sum the products in different orders.
inline std::vector<SplineNode<double>> build_spline_nodes(std::span<const Quat<double>> q,
                                                          std::span<const Vec3<double>> p,
                                                          Vec3<double> origin) {
  if (q.size() != p.size()) throw std::invalid_argument("spline: rotation and position counts differ");
  std::vector<SplineNode<double>> nodes(q.size());
  for (size_t c = 0; c < q.size(); ++c) {
    SplineNode<double>& node = nodes[c];
    node.q = normalized(q[c]);
    node.p = p[c] - origin;
    if (c == 0) {
      node.d = {0.0, 0.0, 0.0};
      node.dp = {0.0, 0.0, 0.0};
    } else {
      const Quat<double>& prev = nodes[c - 1].q;
      node.d = detail::same_rotation_exactly(prev, node.q) ? Vec3<double>{0.0, 0.0, 0.0}
                                                           : so3::log(conjugate(prev) * node.q);
      node.dp = p[c] - p[c - 1];
    }
  }
  return nodes;
}

template <class T>
std::vector<SplineNode<T>> convert_nodes(const std::vector<SplineNode<double>>& nodes) {
  std::vector<SplineNode<T>> out(nodes.size());
  for (size_t c = 0; c < nodes.size(); ++c) {
    out[c] = {cast<T>(nodes[c].q), cast<T>(nodes[c].d), cast<T>(nodes[c].p), cast<T>(nodes[c].dp)};
  }
  return out;
}

}  // namespace balla::traj
