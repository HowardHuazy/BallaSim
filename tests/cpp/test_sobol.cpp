#include <doctest/doctest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "core/rng/sobol.h"

using namespace balla::rng;

namespace {

// True if every elementary box of volume 2^-m holds exactly one of the 2^m points.
bool is_0m2_net(const std::vector<uint32_t>& x, const std::vector<uint32_t>& y, int m) {
  const size_t n = size_t{1} << m;
  for (int a = 0; a <= m; ++a) {
    std::vector<int> count(n, 0);
    for (size_t i = 0; i < n; ++i) {
      const uint64_t bx = static_cast<uint64_t>(x[i]) >> (32 - a);
      const uint64_t by = static_cast<uint64_t>(y[i]) >> (32 - (m - a));
      ++count[static_cast<size_t>((bx << (m - a)) | by)];
    }
    for (int c : count) {
      if (c != 1) return false;
    }
  }
  return true;
}

// Dimension 1 by the Joe-Kuo recurrence (s = 1, m_1 = 1): V_1 = 2^31, V_k = V_{k-1} ^ (V_{k-1} >> 1).
uint32_t sobol_dim1_recurrence(uint32_t index) {
  uint32_t v = 0x80000000u, result = 0;
  for (; index != 0; index >>= 1, v ^= v >> 1) {
    if (index & 1u) result ^= v;
  }
  return result;
}

}  // namespace

TEST_CASE("bit reversal") {
  CHECK(reverse_bits(0u) == 0u);
  CHECK(reverse_bits(1u) == 0x80000000u);
  CHECK(reverse_bits(0x80000000u) == 1u);
  CHECK(reverse_bits(0x12345678u) == 0x1e6a2c48u);
  CHECK(reverse_bits(reverse_bits(0xdeadbeefu)) == 0xdeadbeefu);
}

TEST_CASE("unscrambled Sobol: van der Corput and the first Joe-Kuo dimensions") {
  for (uint32_t i = 0; i < 1000; ++i) CHECK(sobol(i, 0) == reverse_bits(i));
  CHECK(sobol(0, 1) == 0u);
  CHECK(sobol(1, 1) == 0x80000000u);
  CHECK(sobol(2, 1) == 0xC0000000u);
  CHECK(sobol(3, 1) == 0x40000000u);
  CHECK(sobol(4, 2) == 0x60000000u);  // V3 = V1 ^ (V1 >> 2) ^ V2 for (s, a, m) = (2, 1, {1, 3})
  // Every dimension is a (0, 1)-sequence: each aligned block of 2^m indices is stratified.
  for (uint32_t d = 0; d < kSobolDimensions; ++d) {
    std::vector<int> seen(64, 0);
    for (uint32_t i = 64; i < 128; ++i) ++seen[sobol(i, d) >> 26];
    for (int s : seen) CHECK(s == 1);
  }
}

TEST_CASE("closed-form Sobol dimension 1 equals the direction-number recurrence") {
  for (uint32_t i = 0; i < (1u << 20); ++i) REQUIRE(sobol_dim1(i) == sobol_dim1_recurrence(i));
  std::mt19937 rng(12);
  for (int i = 0; i < 2000000; ++i) {
    const uint32_t x = rng();
    REQUIRE(sobol_dim1(x) == sobol_dim1_recurrence(x));
  }
  CHECK(sobol_dim1(0xFFFFFFFFu) == sobol_dim1_recurrence(0xFFFFFFFFu));
  CHECK(sobol(12345u, 1) == sobol_dim1(12345u));
}

TEST_CASE("owen_sobol_2d equals owen_sobol in dimensions 0 and 1") {
  std::mt19937 rng(13);
  for (int s = 0; s < 2000; ++s) {
    const uint32_t seed = rng();
    for (uint32_t k = 0; k < 256; ++k) {
      uint32_t x, y;
      owen_sobol_2d(k, seed, &x, &y);
      REQUIRE(x == owen_sobol(k, 0, seed));
      REQUIRE(y == owen_sobol(k, 1, seed));
    }
  }
}

TEST_CASE("Owen-scrambled Sobol prefixes are (0, m, 2)-nets for N = 16, 64, 256") {
  for (uint32_t seed : {0u, 1u, 12345u, 0xdeadbeefu, 0x9e3779b9u}) {
    std::vector<uint32_t> x(1024), y(1024);
    for (uint32_t i = 0; i < 1024; ++i) {
      x[i] = owen_sobol(i, 0, seed);
      y[i] = owen_sobol(i, 1, seed);
    }
    for (int m : {4, 6, 8, 10}) CHECK(is_0m2_net(x, y, m));
  }
}

TEST_CASE("Owen scrambling decorrelates seeds and dimensions") {
  int equal_seed = 0, equal_dim = 0;
  for (uint32_t i = 0; i < 256; ++i) {
    equal_seed += owen_sobol(i, 0, 1u) == owen_sobol(i, 0, 2u);
    equal_dim += owen_sobol(i, 0, 1u) == owen_sobol(i, 1, 1u);
  }
  CHECK(equal_seed == 0);
  CHECK(equal_dim == 0);
  CHECK(to_unit_interval<float>(0xFFFFFFFFu) < 1.0f);
  CHECK(to_unit_interval<double>(0xFFFFFFFFu) < 1.0);
  CHECK(to_unit_interval<double>(0x80000000u) == 0.5);
}
