// Exact nearest-neighbour distances in 3-D, as scipy.spatial.cKDTree(points).query(x)[0] returns them: the squared
// distance summed ((0 + dx^2) + dy^2) + dz^2 (SciPy's sqeuclidean_distance_double for m = 3), its minimum over the
// points, then sqrt. Any exact search gives that minimum; this one is a median-split k-d tree pruned by the distance
// to each node's bounding box, which is safe in floating point (a pruned point's computed distance is never below the
// bound it was pruned on).
#pragma once

#include <cstdint>
#include <vector>

#include "scene/rotation.h"

namespace balla::scene {

class KdTree {
 public:
  // points: (n, 3) float64, row-major (cKDTree converts float32 data to float64).
  explicit KdTree(std::vector<double> points);
  // The distance from x to its nearest point (sqrt of the minimum squared distance); inf without points.
  double nearest(rot::V x) const;

 private:
  struct Node {
    int32_t begin, end;      // the point range in order_
    int32_t left, right;     // children (-1 for a leaf)
    double lo[3], hi[3];     // the bounding box of the node's points
  };
  int32_t build(int32_t begin, int32_t end, int depth);
  double box_sq(int32_t node, const double q[3]) const;
  void search(int32_t node, const double q[3], double& best) const;

  std::vector<double> pts_;
  std::vector<int32_t> order_;
  std::vector<Node> nodes_;
};

}  // namespace balla::scene
