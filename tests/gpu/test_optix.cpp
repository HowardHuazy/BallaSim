#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "backend/embree/embree_mesh.h"
#include "backend/optix/optix_mesh.h"

using namespace balla::backend;

TEST_CASE("OptiX closest hits agree with Embree (DESIGN §12.1 L2)") {
  // A 32 x 32 grid of quads, bent into a wave so rays hit it at varying angles and depths.
  constexpr int kN = 32;
  std::vector<float> vertices;
  for (int j = 0; j <= kN; ++j) {
    for (int i = 0; i <= kN; ++i) {
      const float x = -2.0f + 4.0f * static_cast<float>(i) / kN;
      const float y = -2.0f + 4.0f * static_cast<float>(j) / kN;
      vertices.insert(vertices.end(), {x, y, 3.0f + 0.4f * std::sin(2.0f * x) * std::cos(1.5f * y)});
    }
  }
  std::vector<uint32_t> indices;
  for (uint32_t j = 0; j < kN; ++j) {
    for (uint32_t i = 0; i < kN; ++i) {
      const uint32_t a = j * (kN + 1) + i, b = a + 1, c = a + kN + 1, d = c + 1;
      indices.insert(indices.end(), {a, b, d, a, d, c});
    }
  }
  std::mt19937_64 rng(9);
  std::uniform_real_distribution<float> dir(-0.9f, 0.9f);
  std::vector<float> origins, directions;
  for (int i = 0; i < 100000; ++i) {
    origins.insert(origins.end(), {0.0f, 0.0f, 0.0f});
    directions.insert(directions.end(), {dir(rng), dir(rng), 1.0f});
  }
  const TriangleMeshView mesh{vertices.data(), vertices.size() / 3, indices.data(), indices.size() / 3};
  const RayBatchView rays{origins.data(), directions.data(), origins.size() / 3};
  const MeshHits ox = optix_intersect(mesh, rays);
  const MeshHits em = embree_intersect(mesh, rays);
  // A hit on an edge of its triangle: every edge of these triangles has one barycentric at 0.
  const auto on_edge = [](float u, float v) { return std::min({u, v, 1.0f - u - v}) <= 1e-4f; };
  size_t ties = 0, hits = 0;
  for (size_t i = 0; i < rays.count; ++i) {
    if (ox.prim[i] != em.prim[i]) {  // a tie on a shared edge: same t, both hits on an edge
      ++ties;
      CHECK(ox.prim[i] != kNoPrim);
      CHECK(em.prim[i] != kNoPrim);
      CHECK(std::abs(ox.t[i] - em.t[i]) <= 1e-4f * em.t[i]);
      CHECK(on_edge(ox.u[i], ox.v[i]));
      CHECK(on_edge(em.u[i], em.v[i]));
      continue;
    }
    if (em.prim[i] == kNoPrim) {
      CHECK(std::isinf(ox.t[i]));
      continue;
    }
    ++hits;
    CHECK(std::abs(ox.t[i] - em.t[i]) <= 1e-4f * em.t[i]);
    CHECK(std::abs(ox.u[i] - em.u[i]) <= 1e-4f);
    CHECK(std::abs(ox.v[i] - em.v[i]) <= 1e-4f);
  }
  CHECK(ties <= rays.count / 10000);
  CHECK(hits > rays.count / 2);
  CHECK(hits < rays.count);  // misses beside the grid as well
}

namespace {

// The unit quad of test_embree.cpp: two triangles at z = 2 sharing the diagonal (-1,-1)-(1,1).
const std::vector<float> kQuadVertices = {-1, -1, 2, 1, -1, 2, 1, 1, 2, -1, 1, 2};
const std::vector<uint32_t> kQuadIndices = {0, 1, 2, 0, 2, 3};
const TriangleMeshView kQuad{kQuadVertices.data(), 4, kQuadIndices.data(), 2};

}  // namespace

TEST_CASE("OptiX: hits beyond t = 1e30 agree with Embree (t in [0, inf))") {
  // Origins off the shared diagonal, so each ray has one closest triangle; |d| = 1e-30 puts the
  // hit at t = 2e30. Embree's barycentrics collapse at this scale, so only t and prim are compared.
  const std::vector<float> origins = {0.25f, -0.5f, 0, -0.5f, 0.25f, 0};
  const std::vector<float> dirs = {0, 0, 1e-30f, 0, 0, 1e-30f};
  const RayBatchView rays{origins.data(), dirs.data(), 2};
  const MeshHits ox = optix_intersect(kQuad, rays);
  const MeshHits em = embree_intersect(kQuad, rays);
  for (size_t i = 0; i < rays.count; ++i) {
    CHECK(em.prim[i] != kNoPrim);
    CHECK(ox.prim[i] == em.prim[i]);
    CHECK(std::isfinite(ox.t[i]));
    CHECK(ox.t[i] == doctest::Approx(em.t[i]).epsilon(1e-6));
  }
}

TEST_CASE("OptiX: an empty mesh gives all misses, as Embree") {
  const std::vector<float> origins = {0, 0, 0, 0.25f, -0.5f, 0};
  const std::vector<float> dirs = {0, 0, 1, 0, 0, 1};
  const TriangleMeshView empty{nullptr, 0, nullptr, 0};
  const RayBatchView rays{origins.data(), dirs.data(), 2};
  const MeshHits ox = optix_intersect(empty, rays);
  const MeshHits em = embree_intersect(empty, rays);
  REQUIRE(ox.t.size() == 2);
  for (size_t i = 0; i < rays.count; ++i) {
    CHECK(std::isinf(ox.t[i]));
    CHECK(ox.prim[i] == kNoPrim);
    CHECK(ox.u[i] == 0.0f);
    CHECK(ox.v[i] == 0.0f);
    CHECK(ox.t[i] == em.t[i]);
    CHECK(ox.prim[i] == em.prim[i]);
  }
}

TEST_CASE("OptiX: invalid arguments throw std::invalid_argument before any device work") {
  const std::vector<uint32_t> bad = {0, 1, 7};
  const std::vector<float> origin = {0, 0, 0}, dir = {0, 0, 1};
  CHECK_THROWS_AS(optix_intersect({kQuadVertices.data(), 4, bad.data(), 1},
                                  {origin.data(), dir.data(), 1}),
                  std::invalid_argument);
  // Beyond the OptiX launch limit; rejected before any buffer is read or allocated.
  CHECK_THROWS_AS(optix_intersect(kQuad, RayBatchView{nullptr, nullptr, (size_t{1} << 30) + 1}),
                  std::invalid_argument);
}
