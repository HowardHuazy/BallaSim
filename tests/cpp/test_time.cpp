#include <doctest/doctest.h>

#include <cstdint>
#include <initializer_list>
#include <limits>

#include "core/time/time.h"

using namespace balla::time;

TEST_CASE("floor_div and floor_mod round towards negative infinity") {
  CHECK(floor_div(int64_t{7}, int64_t{2}) == 3);
  CHECK(floor_div(int64_t{-7}, int64_t{2}) == -4);
  CHECK(floor_div(int64_t{-8}, int64_t{2}) == -4);
  CHECK(floor_mod(int64_t{-7}, int64_t{2}) == 1);
  CHECK(floor_mod(int64_t{-8}, int64_t{2}) == 0);
  CHECK(floor_div(int32_t{-1}, int32_t{5000000}) == -1);
  CHECK(floor_mod(int32_t{-1}, int32_t{5000000}) == 4999999);
  for (int64_t a = -1000; a <= 1000; ++a) {
    for (int64_t b : {1, 3, 7, 1000}) {
      const int64_t q = floor_div(a, b), r = floor_mod(a, b);
      CHECK(q * b + r == a);
      CHECK(r >= 0);
      CHECK(r < b);
    }
  }
}

TEST_CASE("floor_div at the extremes of int64") {
  constexpr int64_t lo = std::numeric_limits<int64_t>::min();
  constexpr int64_t hi = std::numeric_limits<int64_t>::max();
  CHECK(floor_div(lo, kWindowNs) == -(int64_t{1} << 40));
  CHECK(floor_mod(lo, kWindowNs) == 0);
  CHECK(floor_div(lo + 1, kWindowNs) == -(int64_t{1} << 40));
  CHECK(floor_mod(lo + 1, kWindowNs) == 1);
  CHECK(floor_div(hi, kWindowNs) == (int64_t{1} << 40) - 1);
  CHECK(floor_mod(hi, kWindowNs) == kWindowNs - 1);
}

TEST_CASE("windows of 2^23 ns, including negative times") {
  CHECK(kWindowNs == 8388608);
  CHECK(window_index(0) == 0);
  CHECK(window_index(kWindowNs - 1) == 0);
  CHECK(window_index(kWindowNs) == 1);
  CHECK(window_index(-1) == -1);
  CHECK(window_index(-kWindowNs) == -1);
  CHECK(window_index(-kWindowNs - 1) == -2);
  for (int64_t t : {int64_t{-123456789012}, int64_t{-1}, int64_t{0}, int64_t{987654321098765}}) {
    const int64_t w = window_index(t);
    CHECK(w == floor_div(t, kWindowNs));
    CHECK(window_start(w) <= t);
    CHECK(t < window_start(w + 1));
  }
}

TEST_CASE("dyadic lattice nodes") {
  CHECK(lattice_step(0) == kWindowNs);
  CHECK(lattice_step(kMaxSampleLevel) == 1024);
  CHECK(lattice_step(kMaxLevel) == 1);
  CHECK(lattice_node(-3, 0, 0) == -3 * kWindowNs);
  CHECK(lattice_node(2, 1, 1) == 2 * kWindowNs + kWindowNs / 2);
  CHECK(lattice_node(5, 13, 8192) == lattice_node(6, 0, 0));
  // Nodes nest: every level-l node is a node of every finer level.
  for (int level = 0; level <= kMaxSampleLevel; ++level) {
    const int64_t t = lattice_node(-7, level, (int64_t{1} << level) - 1);
    for (int finer = level; finer <= kMaxLevel; ++finer) CHECK(on_lattice(t, finer));
  }
  CHECK(coarsest_level(lattice_node(-4, 0, 0)) == 0);
  CHECK(coarsest_level(lattice_node(-4, 5, 3)) == 5);
  CHECK(coarsest_level(lattice_node(9, 13, 1)) == 13);
  CHECK(coarsest_level(-1) == kMaxLevel);
}

TEST_CASE("int32 offsets from int64 bases") {
  const int64_t base = 1'700'000'000'000'000'000;
  CHECK(offset_fits(base + INT32_MAX, base));
  CHECK_FALSE(offset_fits(base + INT32_MAX + int64_t{1}, base));
  CHECK(offset_fits(base + INT32_MIN, base));
  CHECK_FALSE(offset_fits(base + INT32_MIN - int64_t{1}, base));
  for (int64_t d : {int64_t{0}, int64_t{-5}, int64_t{123456789}, int64_t{INT32_MIN}}) {
    CHECK(from_offset(base, to_offset(base + d, base)) == base + d);
  }
  // Distances of 2^63 ns or more: t - base overflowed int64 (undefined; INT64_MAX - (-1) wrapped to INT64_MIN).
  static_assert(!offset_fits(INT64_MAX, -1));
  static_assert(!offset_fits(INT64_MIN, 1));
  static_assert(!offset_fits(INT64_MAX, INT64_MIN));
  static_assert(offset_fits(INT64_MIN, INT64_MIN + INT32_MAX));
  static_assert(offset_fits(INT64_MAX, INT64_MAX - INT32_MAX));
}
