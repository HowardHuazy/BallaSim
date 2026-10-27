// The exact distance from a point to the static geometry (blender.md §4.7.2 d_scene): a camera in a room or above a
// large ground plane gets its true clearance instead of the 5 cm floor.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "scene/rotation.h"

namespace balla::scene::motion {

// Exact point-to-triangle distance (closest point by Voronoi region; Ericson, "Real-Time Collision Detection" §5.1.5).
double point_triangle_distance(const rot::V& p, const rot::V& a, const rot::V& b, const rot::V& c);

// Triangles in Morton order of their centroids, grouped in leaves of 16 and leaves in groups of 16, each with a
// bounding sphere; a query visits groups and leaves in order of their lower bound |x - centre| - radius and stops when
// that bound exceeds the running minimum (as the planner's MeshDistance prunes, src/engine/motion.h). The result is
// the minimum over every triangle. Concurrent queries are safe.
class StaticDistance {
 public:
  StaticDistance() = default;
  StaticDistance(std::span<const double> vertices, std::span<const uint32_t> indices);  // world frame, metres
  double operator()(const rot::V& x) const;  // +inf without geometry
  bool empty() const { return tri_.empty(); }

 private:
  struct Sphere {
    rot::V c;
    double r;
  };
  static constexpr size_t kLeaf = 16;
  std::vector<std::array<rot::V, 3>> tri_;
  std::vector<Sphere> leaf_, group_;
};

}  // namespace balla::scene::motion
