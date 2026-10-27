// Hash-based Owen-scrambled Sobol points (Burley 2020, JCGT 9(4)) with Joe-Kuo direction numbers
// (new-joe-kuo-6.21201; see THIRD_PARTY_NOTICES.md), generated on the fly so the same code runs
// on the device without lookup tables.
//
// owen_sobol(i, d, seed) is dimension d of point i of a shuffled, Owen-scrambled Sobol sequence:
// the index is permuted by a nested uniform scramble and each dimension is scrambled with its own
// seed. Both are bijections that map elementary intervals onto elementary intervals, so the first
// 2^m points of dimensions (0, 1) form a (0, m, 2)-net for every m, and prefixes nest (§6.1).
// Dimensions 0 and 1, the sub-sample pattern, have closed forms (bit reversal and sobol_dim1), so
// owen_sobol_2d evaluates a pattern point without the direction-number recurrence.
#pragma once

#include <stdint.h>

#include "core/hd.h"

namespace balla::rng {

inline constexpr uint32_t kSobolDimensions = 8;

BALLA_HD inline uint32_t reverse_bits(uint32_t x) {
#if defined(__CUDA_ARCH__)
  return __brev(x);
#else
  x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
  x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
  x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
  x = ((x >> 8) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8);
  return (x >> 16) | (x << 16);
#endif
}

namespace detail {

// Primitive polynomial (degree s, coefficients a) and initial direction numbers m_1..m_s of
// Joe-Kuo dimensions 3..8, which are our dimensions 2..7 (dimension 0 is van der Corput and
// dimension 1, Joe-Kuo dimension 2, has the closed form sobol_dim1).
struct SobolPolynomial {
  uint32_t s, a;
  uint32_t m[5];
};

BALLA_HD inline SobolPolynomial joe_kuo(uint32_t dim) {
  switch (dim) {
    case 2: return {2, 1, {1, 3, 0, 0, 0}};
    case 3: return {3, 1, {1, 3, 1, 0, 0}};
    case 4: return {3, 2, {1, 1, 1, 0, 0}};
    case 5: return {4, 1, {1, 1, 3, 3, 0}};
    case 6: return {4, 4, {1, 3, 5, 13, 0}};
    default: return {5, 2, {1, 1, 5, 5, 17}};
  }
}

// Laine-Karras style permutation with Vegdahl's improved constants (2021). Every step maps bit b
// to itself xor a function of lower bits, so in bit-reversed order it is a nested uniform scramble.
BALLA_HD inline uint32_t lk_permutation(uint32_t x, uint32_t seed) {
  x ^= x * 0x3d20adeau;
  x += seed;
  x *= (seed >> 16) | 1u;
  x ^= x * 0x05526c56u;
  x ^= x * 0x53a22864u;
  return x;
}

// "lowbias32" integer finalizer (C. Wellons), used to derive per-dimension seeds.
BALLA_HD inline uint32_t hash_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// Seed of the scramble of dimension dim.
BALLA_HD inline uint32_t dimension_seed(uint32_t seed, uint32_t dim) {
  return hash_u32(seed ^ (0x9E3779B9u * (dim + 1u)));
}

}  // namespace detail

// Sobol dimension 1 (Joe-Kuo dimension 2: s = 1, m_1 = 1) in closed form. Its direction numbers
// V_k = V_{k-1} ^ (V_{k-1} >> 1) form the Pascal matrix mod 2, so bit t (from the top) of the point
// is the XOR of the index bits m with C(m, t) odd, i.e. t a bit-subset of m (Lucas): the
// superset-XOR transform of the index, 5 masked shift-XOR steps instead of a loop over 32 bits.
BALLA_HD inline uint32_t sobol_dim1(uint32_t index) {
  uint32_t y = index;
  y ^= (y >> 1) & 0x55555555u;
  y ^= (y >> 2) & 0x33333333u;
  y ^= (y >> 4) & 0x0F0F0F0Fu;
  y ^= (y >> 8) & 0x00FF00FFu;
  y ^= (y >> 16) & 0x0000FFFFu;
  return reverse_bits(y);
}

// Unscrambled Sobol point, 32-bit fixed point. Precondition: dim < kSobolDimensions.
BALLA_HD inline uint32_t sobol(uint32_t index, uint32_t dim) {
  if (dim == 0) return reverse_bits(index);
  if (dim == 1) return sobol_dim1(index);
  const detail::SobolPolynomial p = detail::joe_kuo(dim);
  // Direction numbers V_k (k = 1..32) by the Joe-Kuo recurrence; h1..h5 hold V_{k-1}..V_{k-5}.
  uint32_t h1 = 0, h2 = 0, h3 = 0, h4 = 0, h5 = 0;
  uint32_t result = 0;
  for (uint32_t k = 1; index != 0; ++k, index >>= 1) {
    uint32_t v;
    if (k <= p.s) {
      v = p.m[k - 1] << (32 - k);
    } else {
      const uint32_t vs = p.s == 2 ? h2 : p.s == 3 ? h3 : p.s == 4 ? h4 : h5;
      v = vs ^ (vs >> p.s);
      if ((p.a >> (p.s - 2)) & 1u) v ^= h1;
      if (p.s > 2 && ((p.a >> (p.s - 3)) & 1u)) v ^= h2;
      if (p.s > 3 && ((p.a >> (p.s - 4)) & 1u)) v ^= h3;
      if (p.s > 4 && ((p.a >> (p.s - 5)) & 1u)) v ^= h4;
    }
    h5 = h4;
    h4 = h3;
    h3 = h2;
    h2 = h1;
    h1 = v;
    if (index & 1u) result ^= v;
  }
  return result;
}

// Nested uniform (Owen) scramble of a 32-bit fixed-point coordinate.
BALLA_HD inline uint32_t owen_scramble(uint32_t x, uint32_t seed) {
  return reverse_bits(detail::lk_permutation(reverse_bits(x), seed));
}

BALLA_HD inline uint32_t owen_sobol(uint32_t index, uint32_t dim, uint32_t seed) {
  const uint32_t shuffled = owen_scramble(index, seed);
  return owen_scramble(sobol(shuffled, dim), detail::dimension_seed(seed, dim));
}

// Dimensions 0 and 1 of one point: owen_sobol(index, 0, seed) and owen_sobol(index, 1, seed),
// sharing the index shuffle.
BALLA_HD inline void owen_sobol_2d(uint32_t index, uint32_t seed, uint32_t* x, uint32_t* y) {
  const uint32_t shuffled = owen_scramble(index, seed);
  *x = owen_scramble(reverse_bits(shuffled), detail::dimension_seed(seed, 0));
  *y = owen_scramble(sobol_dim1(shuffled), detail::dimension_seed(seed, 1));
}

// Fixed point to [0, 1): float keeps the top 24 bits (exact, never rounds up to 1).
template <class T>
BALLA_HD T to_unit_interval(uint32_t x);
template <>
BALLA_HD inline float to_unit_interval<float>(uint32_t x) {
  return static_cast<float>(x >> 8) * 0x1p-24f;
}
template <>
BALLA_HD inline double to_unit_interval<double>(uint32_t x) {
  return static_cast<double>(x) * 0x1p-32;
}

}  // namespace balla::rng
