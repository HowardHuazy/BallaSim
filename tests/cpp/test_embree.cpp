#include <doctest/doctest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "backend/embree/embree_mesh.h"

using namespace balla::backend;

TEST_CASE("Embree: rays against a unit quad at z = 2") {
  const std::vector<float> vertices = {-1, -1, 2, 1, -1, 2, 1, 1, 2, -1, 1, 2};
  const std::vector<uint32_t> indices = {0, 1, 2, 0, 2, 3};
  const std::vector<float> origins = {0, 0, 0, 0.25f, -0.5f, 0, 0, 0, 0, 0, 0, 5};
  const std::vector<float> dirs = {0, 0, 1, 0, 0, 2, 3, 0, 1, 0, 0, 1};
  const MeshHits hits = embree_intersect({vertices.data(), 4, indices.data(), 2},
                                         {origins.data(), dirs.data(), 4});
  REQUIRE(hits.t.size() == 4);
  CHECK(hits.t[0] == doctest::Approx(2.0f).epsilon(1e-6));
  CHECK(hits.t[1] == doctest::Approx(1.0f).epsilon(1e-6));  // direction length 2
  CHECK(hits.prim[1] == 0u);
  // Hit point from barycentrics: (1-u-v) v0 + u v1 + v v2 = (0.25, -0.5, 2) on triangle 0.
  const float x = (1 - hits.u[1] - hits.v[1]) * -1 + hits.u[1] * 1 + hits.v[1] * 1;
  const float y = (1 - hits.u[1] - hits.v[1]) * -1 + hits.u[1] * -1 + hits.v[1] * 1;
  CHECK(x == doctest::Approx(0.25f).epsilon(1e-6));
  CHECK(y == doctest::Approx(-0.5f).epsilon(1e-6));
  CHECK(std::isinf(hits.t[2]));  // passes beside the quad
  CHECK(hits.prim[2] == kNoPrim);
  CHECK(std::isinf(hits.t[3]));  // starts behind it, points away
}

TEST_CASE("Embree: an out-of-range vertex index throws std::invalid_argument") {
  const std::vector<float> vertices = {-1, -1, 2, 1, -1, 2, 1, 1, 2, -1, 1, 2};
  const std::vector<uint32_t> indices = {0, 1, 7};
  const std::vector<float> origin = {0, 0, 0}, dir = {0, 0, 1};
  CHECK_THROWS_AS(embree_intersect({vertices.data(), 4, indices.data(), 1},
                                   {origin.data(), dir.data(), 1}),
                  std::invalid_argument);
}
