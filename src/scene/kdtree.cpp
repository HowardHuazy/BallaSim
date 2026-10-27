// Contains C++ translations of SciPy (cKDTree's squared-distance summation order), under its BSD licence:
// see THIRD_PARTY_NOTICES.md, "Ported code".
#include "scene/kdtree.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace balla::scene {

namespace {

constexpr int32_t kLeaf = 16;

// SciPy's sqeuclidean_distance_double for m = 3: ((0 + d0^2) + d1^2) + d2^2.
double sqdist(const double* p, const double q[3]) {
  double s = 0.0;
  for (int k = 0; k < 3; ++k) {
    const double d = p[k] - q[k];
    s += d * d;
  }
  return s;
}

}  // namespace

KdTree::KdTree(std::vector<double> points) : pts_(std::move(points)) {
  if (pts_.size() % 3 != 0) throw std::invalid_argument("KdTree: points must be (n, 3)");
  const size_t n = pts_.size() / 3;
  if (n > static_cast<size_t>(std::numeric_limits<int32_t>::max())) throw std::invalid_argument("KdTree: too many points");
  order_.resize(n);
  std::iota(order_.begin(), order_.end(), 0);
  if (n > 0) build(0, static_cast<int32_t>(n), 0);
}

int32_t KdTree::build(int32_t begin, int32_t end, int depth) {
  const auto id = static_cast<int32_t>(nodes_.size());
  Node node{begin, end, -1, -1, {}, {}};
  for (int k = 0; k < 3; ++k) {
    node.lo[k] = std::numeric_limits<double>::infinity();
    node.hi[k] = -node.lo[k];
  }
  for (int32_t i = begin; i < end; ++i) {
    const double* p = &pts_[3 * static_cast<size_t>(order_[static_cast<size_t>(i)])];
    for (int k = 0; k < 3; ++k) {
      node.lo[k] = std::min(node.lo[k], p[k]);
      node.hi[k] = std::max(node.hi[k], p[k]);
    }
  }
  nodes_.push_back(node);
  if (end - begin <= kLeaf) return id;
  // Split the widest extent at its median point.
  int axis = 0;
  for (int k = 1; k < 3; ++k) {
    if (node.hi[k] - node.lo[k] > node.hi[axis] - node.lo[axis]) axis = k;
  }
  if (!(node.hi[axis] > node.lo[axis])) return id;   // every point equal (or NaN): one leaf
  const int32_t mid = begin + (end - begin) / 2;
  std::nth_element(order_.begin() + begin, order_.begin() + mid, order_.begin() + end, [this, axis](int32_t a, int32_t b) {
    return pts_[3 * static_cast<size_t>(a) + static_cast<size_t>(axis)] <
           pts_[3 * static_cast<size_t>(b) + static_cast<size_t>(axis)];
  });
  const int32_t left = build(begin, mid, depth + 1);
  const int32_t right = build(mid, end, depth + 1);
  nodes_[static_cast<size_t>(id)].left = left;
  nodes_[static_cast<size_t>(id)].right = right;
  return id;
}

// The squared distance from q to the node's bounding box, summed as sqdist: every point in the box has a computed
// distance at least this (each |q_k - p_k| >= the gap, and rounding is monotone), so a box farther than the best so far
// cannot hold a nearer point.
double KdTree::box_sq(int32_t node, const double q[3]) const {
  const Node& nd = nodes_[static_cast<size_t>(node)];
  double s = 0.0;
  for (int k = 0; k < 3; ++k) {
    const double g = q[k] < nd.lo[k] ? nd.lo[k] - q[k] : q[k] > nd.hi[k] ? q[k] - nd.hi[k] : 0.0;
    s += g * g;
  }
  return s;
}

void KdTree::search(int32_t node, const double q[3], double& best) const {
  const Node& nd = nodes_[static_cast<size_t>(node)];
  if (nd.left < 0) {
    for (int32_t i = nd.begin; i < nd.end; ++i) {
      const double d = sqdist(&pts_[3 * static_cast<size_t>(order_[static_cast<size_t>(i)])], q);
      if (d < best) best = d;
    }
    return;
  }
  const double dl = box_sq(nd.left, q), dr = box_sq(nd.right, q);
  const bool left_first = dl <= dr;
  const int32_t a = left_first ? nd.left : nd.right, b = left_first ? nd.right : nd.left;
  const double da = left_first ? dl : dr, db = left_first ? dr : dl;
  if (!(da > best)) search(a, q, best);
  if (!(db > best)) search(b, q, best);
}

double KdTree::nearest(rot::V x) const {
  double best = std::numeric_limits<double>::infinity();
  if (nodes_.empty()) return best;
  const double q[3] = {x.x, x.y, x.z};
  search(0, q, best);
  return std::sqrt(best);
}

}  // namespace balla::scene
