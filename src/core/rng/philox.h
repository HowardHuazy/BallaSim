// Philox4x32-10 counter-based generator (Salmon et al., SC'11), checked against the Random123
// known-answer vectors in tests/cpp/test_philox.cpp. Counter-based draws are order-independent
// and bitwise identical on host and device (DESIGN §12.1, S1 §5.2).
#pragma once

#include <stdint.h>

#include "core/hd.h"

namespace balla::rng {

struct U32x4 {
  uint32_t v[4];
};

struct PhiloxKey {
  uint32_t k0, k1;
};

namespace detail {

inline constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
inline constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
inline constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;  // golden ratio
inline constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;  // sqrt(3) - 1

BALLA_HD inline uint32_t mulhi32(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__)
  return __umulhi(a, b);
#else
  return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
#endif
}

BALLA_HD inline U32x4 philox_round(U32x4 c, PhiloxKey k) {
  const uint32_t hi0 = mulhi32(kPhiloxM0, c.v[0]);
  const uint32_t lo0 = kPhiloxM0 * c.v[0];
  const uint32_t hi1 = mulhi32(kPhiloxM1, c.v[2]);
  const uint32_t lo1 = kPhiloxM1 * c.v[2];
  return {{hi1 ^ c.v[1] ^ k.k0, lo1, hi0 ^ c.v[3] ^ k.k1, lo0}};
}

}  // namespace detail

BALLA_HD inline U32x4 philox4x32_10(U32x4 counter, PhiloxKey key) {
  counter = detail::philox_round(counter, key);
  for (int round = 1; round < 10; ++round) {
    key.k0 += detail::kPhiloxW0;
    key.k1 += detail::kPhiloxW1;
    counter = detail::philox_round(counter, key);
  }
  return counter;
}

}  // namespace balla::rng
