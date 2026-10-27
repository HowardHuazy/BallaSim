// balla_scene: SciPy's sparse structure (SMMP order, canonicalization), OpenBLAS's one-thread pinning in pool
// threads, the scan report's mesh statistics (scans.md §3.5, T7) and the binary16 decoder of the Replica import
// (S5b-1). The bitwise comparisons with NumPy and SciPy themselves are Python tests
// (tests/python/test_scene_cpp.py).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "render/thread_pool.h"
#include "scene/assets/mesh_stats.h"
#include "scene/assets/replica.h"
#include "scene/blas.h"
#include "scene/npo.h"
#include "scene/sparse.h"

namespace sp = balla::scene::sparse;
namespace blas = balla::scene::blas;
namespace npo = balla::scene::npo;

namespace {

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

}  // namespace

TEST_CASE("csr_matmat lists each output row in its linked-list order and drops exact zeros") {
  // A = [[1, 2], [0, 3]], B = [[1, -1, 0], [0, 0.5, 1]]: row 0 of A B touches columns 0, 1 (from B row 0), then 2.
  const sp::Matrix a = sp::csr(2, 2, {0, 2, 3}, {0, 1, 1}, {1.0, 2.0, 3.0});
  const sp::Matrix b = sp::csr(2, 3, {0, 2, 4}, {0, 1, 1, 2}, {1.0, -1.0, 0.5, 1.0});
  const sp::Matrix c = sp::matmul(a, b);
  // Row 0: sums[0] = 1, sums[1] = -1 + 1 = 0 (dropped), sums[2] = 2; head order 2, 1, 0.
  CHECK(c.indptr == std::vector<int64_t>{0, 2, 4});
  CHECK(c.indices == std::vector<int64_t>{2, 0, 2, 1});
  CHECK(c.data == std::vector<double>{2.0, 1.0, 3.0, 1.5});
  CHECK_FALSE(sp::has_canonical_format(c));
}

TEST_CASE("coo_to_csr sums duplicates in their order of appearance after a stable sort of short rows") {
  const std::vector<int64_t> r{0, 0, 0, 1}, c{2, 0, 2, 1};
  const std::vector<double> v{1e16, 1.0, -1e16, 4.0};
  const sp::Matrix m = sp::coo_to_csr(2, 3, r, c, v);
  CHECK(m.indices == std::vector<int64_t>{0, 2, 1});
  // (1e16 + -1e16) in appearance order is 0, kept as an explicit entry (sum_duplicates does not drop zeros).
  CHECK(m.data == std::vector<double>{1.0, 0.0, 4.0});
}

TEST_CASE("csc @ csr goes through csc_matmat on the converted right operand") {
  const sp::Matrix a = sp::csr(2, 2, {0, 2, 3}, {0, 1, 1}, {1.0, 2.0, 3.0});
  const sp::Matrix at = sp::transpose(a);
  const sp::Matrix p = sp::matmul(at, a);   // A^T A as CSC
  CHECK(p.format == sp::Format::kCsc);
  const sp::Matrix d = sp::to_csr(p);
  // A^T A = [[1, 2], [2, 13]].
  CHECK(d.indptr == std::vector<int64_t>{0, 2, 4});
  std::vector<double> dense(4, 0.0);
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t jj = d.indptr[static_cast<size_t>(i)]; jj < d.indptr[static_cast<size_t>(i + 1)]; ++jj) {
      dense[static_cast<size_t>(2 * i + d.indices[static_cast<size_t>(jj)])] += d.data[static_cast<size_t>(jj)];
    }
  }
  CHECK(dense == std::vector<double>{1.0, 2.0, 2.0, 13.0});
}

TEST_CASE("pairwise_sum splits at multiples of 8 above 128 elements") {
  std::vector<double> a(300);
  for (size_t i = 0; i < a.size(); ++i) a[i] = 1.0 / static_cast<double>(i + 1);
  const double whole = npo::pairwise_sum(a.data(), 300);
  const double halves = npo::pairwise_sum(a.data(), 144) + npo::pairwise_sum(a.data() + 144, 156);
  CHECK(same_bits(whole, halves));
}

TEST_CASE("matmul3 is bitwise cblas_dgemm (inline after the self-check, or the library) for every transpose pair") {
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<double> u(-0.5, 0.5), e(-30.0, 30.0);
  int mismatches = 0;
  for (int trial = 0; trial < 100'000; ++trial) {
    double a[9], b[9], c1[9], c2[9];
    for (int i = 0; i < 9; ++i) {
      a[i] = trial % 17 == i ? -0.0 : u(rng) * std::exp(e(rng));
      b[i] = trial % 19 == i ? 0.0 : u(rng) * std::exp(e(rng));
    }
    const bool ta = trial % 2, tb = (trial / 2) % 2;
    cblas_dgemm(blas::kRowMajor, ta ? blas::kTrans : blas::kNoTrans, tb ? blas::kTrans : blas::kNoTrans, 3, 3, 3, 1.0,
                a, 3, b, 3, 0.0, c1, 3);
    blas::matmul3(a, ta, b, tb, c2);
    mismatches += std::memcmp(c1, c2, sizeof c1) != 0;
  }
  CHECK(mismatches == 0);
  const std::string how = blas::matmul3_is_inline() ? "matmul3: inline" : "matmul3: the library (self-check failed)";
  MESSAGE(how);
}

TEST_CASE("openblas_set_num_threads_local is process-wide in this build (so OneThread pins the process count)") {
  const int before = openblas_get_num_threads();
  openblas_set_num_threads(16);
  const int prev = openblas_set_num_threads_local(1);
  int other = 0;
  std::thread t([&other] { other = openblas_get_num_threads(); });
  t.join();
  openblas_set_num_threads_local(prev);
  // Another thread sees the "local" setting: it is the process-wide count. If a later OpenBLAS makes it thread-local,
  // this fails and OneThread can go back to it (docs: CPP_PORT item 5, S1).
  CHECK(other == 1);
  openblas_set_num_threads(before);
}

TEST_CASE("ddot of 45k elements on 8 pool threads, each pinned to one OpenBLAS thread, equals the one-thread value") {
  const int n = 45'000;
  std::mt19937_64 rng(45'000);
  std::uniform_real_distribution<double> u(-0.5, 0.5), s(0.0, 4.0);
  std::vector<double> x(n), y(n);
  for (int i = 0; i < n; ++i) {
    x[static_cast<size_t>(i)] = u(rng) * std::exp(s(rng));
    y[static_cast<size_t>(i)] = u(rng) * std::exp(s(rng));
  }
  const int threads_before = openblas_get_num_threads();
  openblas_set_num_threads(1);
  const double one = blas::dot(n, x.data(), y.data());
  openblas_set_num_threads(16);
  const double sixteen = blas::dot(n, x.data(), y.data());
  // The premise: OpenBLAS splits this sum by its thread count (n > 10000), so 16 threads round differently.
  CHECK_FALSE(same_bits(one, sixteen));
  std::vector<double> got(64, 0.0);
  {
    balla::render::ThreadPool pool(8);
    pool.parallel_for(got.size(), 1, [&](size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) {
        const blas::OneThread pin;
        got[i] = blas::dot(n, x.data(), y.data());
      }
    });
  }
  for (const double v : got) CHECK(same_bits(v, one));
  // The last pin ends with the process-wide count restored.
  CHECK(openblas_get_num_threads() == 16);
  CHECK(same_bits(blas::dot(n, x.data(), y.data()), sixteen));
  openblas_set_num_threads(threads_before);
  CHECK(blas::config().find("OpenBLAS") != std::string::npos);
}

namespace {

// An axis-aligned box of size (a, b, c) at the origin, each face its own 4 vertices (split as a UV or normal seam would
// split it), two triangles per face; `faces` picks the faces in the order -z, +z, -y, +y, -x, +x.
balla::scene::geo::Mesh seam_box(float a, float b, float c, std::vector<int> faces) {
  const float p[8][3] = {{0, 0, 0}, {a, 0, 0}, {a, b, 0}, {0, b, 0}, {0, 0, c}, {a, 0, c}, {a, b, c}, {0, b, c}};
  const int quads[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {2, 3, 7, 6}, {0, 4, 7, 3}, {1, 2, 6, 5}};
  balla::scene::geo::Mesh m;
  for (const int f : faces) {
    const auto base = static_cast<uint32_t>(m.num_vertices());
    for (const int v : quads[f]) m.positions.insert(m.positions.end(), p[v], p[v] + 3);
    for (const uint32_t i : {0u, 1u, 2u, 0u, 2u, 3u}) m.indices.push_back(base + i);
  }
  return m;
}

}  // namespace

TEST_CASE("mesh_stats: an open box's boundary, a closed box, a fin and a degenerate triangle") {
  namespace as = balla::scene::assets;
  // Open top (+z missing): the boundary is the top rim, 2 (a + b); the 24 seam vertices weld to 8.
  const as::MeshStats open = as::mesh_stats(seam_box(2.0f, 3.0f, 0.5f, {0, 2, 3, 4, 5}));
  CHECK(open.triangles == 10);
  CHECK(open.bbox_m[0] == 2.0);
  CHECK(open.bbox_m[1] == 3.0);
  CHECK(open.bbox_m[2] == 0.5);
  CHECK(open.open_boundary_m == 10.0);
  CHECK(open.nonmanifold_edges == 0);
  CHECK(open.welded_vertices == 8);
  CHECK(open.degenerate_triangles == 0);
  const as::MeshStats closed = as::mesh_stats(seam_box(1.0f, 1.0f, 1.0f, {0, 1, 2, 3, 4, 5}));
  CHECK(closed.open_boundary_m == 0.0);
  CHECK(closed.nonmanifold_edges == 0);
  // A fin on the closed box's edge (0,0,0)-(1,0,0) gives that edge three faces; its two other edges are boundary.
  balla::scene::geo::Mesh fin = seam_box(1.0f, 1.0f, 1.0f, {0, 1, 2, 3, 4, 5});
  const auto base = static_cast<uint32_t>(fin.num_vertices());
  for (const float v : {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.5f, -1.0f, 0.0f}) fin.positions.push_back(v);
  for (const uint32_t i : {0u, 1u, 2u}) fin.indices.push_back(base + i);
  // And a triangle with two welded-equal corners adds no edge.
  for (const uint32_t i : {0u, 1u, 0u}) fin.indices.push_back(base + i);
  const as::MeshStats f = as::mesh_stats(fin);
  CHECK(f.triangles == 14);
  CHECK(f.nonmanifold_edges == 1);
  CHECK(f.degenerate_triangles == 1);
  CHECK(f.open_boundary_m == doctest::Approx(2.0 * std::sqrt(1.25)).epsilon(1e-15));
  // An index past the vertices is refused.
  fin.indices.back() = 1000;
  CHECK_THROWS_AS(as::mesh_stats(fin), std::invalid_argument);
}

TEST_CASE("binary16_value decodes IEEE binary16 exactly: every finite pattern round-trips through binary16_bits") {
  namespace as = balla::scene::assets;
  int bad = 0;
  for (uint32_t b = 0; b < 65536; ++b) {
    const auto bits = static_cast<uint16_t>(b);
    const double v = as::binary16_value(bits);
    if (((b >> 10) & 31) == 31) {                      // infinity (zero mantissa) and NaN
      bad += (b & 1023) == 0 ? !std::isinf(v) : !std::isnan(v);
    } else {
      bad += as::binary16_bits(v) != bits;
    }
  }
  CHECK(bad == 0);
  CHECK(as::binary16_value(0x3c00) == 1.0);
  CHECK(as::binary16_value(0x7bff) == 65504.0);
  CHECK(as::binary16_value(0x0001) == 0x1p-24);       // the smallest subnormal
  CHECK(as::binary16_value(0x0400) == 0x1p-14);       // the smallest normal
  CHECK(as::binary16_value(0xc000) == -2.0);
  CHECK(as::binary16_value(0x8000) == 0.0);
  CHECK(std::signbit(as::binary16_value(0x8000)));
  CHECK(as::binary16_value(0x7c00) == INFINITY);
}
