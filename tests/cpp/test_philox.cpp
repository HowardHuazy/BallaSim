#include <doctest/doctest.h>

#include <cstdint>

#include "core/rng/philox.h"
#include "core/rng/purpose.h"
#include "core/rng/subsample.h"

using namespace balla::rng;

TEST_CASE("Philox4x32-10 matches the Random123 known-answer vectors") {
  struct Kat {
    U32x4 ctr;
    PhiloxKey key;
    U32x4 expected;
  };
  const Kat kats[] = {
      {{{0u, 0u, 0u, 0u}}, {0u, 0u}, {{0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}}},
      {{{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}},
       {0xffffffffu, 0xffffffffu},
       {{0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}}},
      {{{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}},
       {0xa4093822u, 0x299f31d0u},
       {{0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}}},
  };
  for (const Kat& k : kats) {
    const U32x4 out = philox4x32_10(k.ctr, k.key);
    for (int i = 0; i < 4; ++i) CHECK(out.v[i] == k.expected.v[i]);
  }
}

TEST_CASE("purpose-tag streams are pinned by golden first outputs") {
  // Append-only registry: these values may never change (DESIGN §12.1).
  CHECK(subsample_pattern_seed(0, 0, 0) == 0xc2972a5au);
  CHECK(subsample_pattern_seed(0x0123456789abcdefull, 2, 307199) == 0x8d4933a3u);
  // Distinct cameras, pixels and purposes address distinct counters.
  CHECK(subsample_pattern_seed(7, 0, 1) != subsample_pattern_seed(7, 1, 1));
  CHECK(subsample_pattern_seed(7, 0, 1) != subsample_pattern_seed(7, 0, 2));
  CHECK(draw(7, Purpose::kSubsamplePattern, 0, 1, 0).v[0] !=
        draw(7, Purpose::kSubsamplePattern, 0, 1, 1).v[0]);
  // The pixel-model tags (registered for stream P), first word at the origin and one word at an
  // address with every counter field non-zero.
  struct Pin {
    Purpose purpose;
    uint32_t origin, word2;
  };
  const Pin pins[] = {{Purpose::kSensorMap, 0xcce173ffu, 0xa3c44c71u},
                      {Purpose::kNoise, 0x49a23b22u, 0xcaf3d5e0u},
                      {Purpose::kNoiseInit, 0xe1acca2du, 0x36d4ec5eu},
                      {Purpose::kStationaryProbe, 0xee952b28u, 0xdbbb6c27u},
                      {Purpose::kReadoutPhase, 0x1f376e45u, 0xe68cd7b4u},
                      // The rate limiter's (P2-H H4a; sensor/limiter.h): selection keys and the epoch phase.
                      {Purpose::kLimiterDrop, 0xef1865b9u, 0x61023ca7u},
                      {Purpose::kLimiterPhase, 0x9a85b788u, 0x56ea58aau},
                      // Area lights (P2-F F8a-1; S1 §5.2 0x0031 and the appended 0x0033): a light's point set and the
                      // padded pairs that assign its points to sub-samples.
                      {Purpose::kLightSampleSet, 0x6f823f51u, 0x229a58d1u},
                      {Purpose::kLightPadding, 0x4fa72b3cu, 0xc68b638au},
                      // APS frames (Phase 3 FR-2, S1 §5.2 0x0060-0x0062).
                      {Purpose::kApsMap, 0x373b6309u, 0x284237bcu},
                      {Purpose::kApsShot, 0xd35a12a0u, 0xdc98feccu},
                      {Purpose::kApsRead, 0xe63835eeu, 0x57d000f3u},
                      // The IMU tags (outputs.md §4.4).
                      {Purpose::kImuNoise, 0xd5610e08u, 0x4d1824d4u},
                      {Purpose::kImuBiasWalk, 0x366b6f63u, 0x6f642875u},
                      {Purpose::kImuBiasInit, 0x247e448du, 0x5a22a38eu},
                      {Purpose::kImuClock, 0xdf95160cu, 0x9f28c15eu}};
  for (const Pin& p : pins) {
    CHECK(draw(0, p.purpose, 0, 0, 0).v[0] == p.origin);
    CHECK(draw(0x0123456789abcdefull, p.purpose, 2, 307199, 65536).v[2] == p.word2);
  }
}
