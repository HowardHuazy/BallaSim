#include <doctest/doctest.h>

#include <type_traits>

#include "backend/api.h"

using namespace balla::backend;

TEST_CASE("backend interface layout") {
  static_assert(sizeof(HitRecord) == 16);
  static_assert(sizeof(SampleRequest) == 24);
  static_assert(std::is_trivially_copyable_v<SampleRequest>);
  static_assert(std::is_trivially_copyable_v<HitRecord>);
  SampleRequest r{};
  r.flags = static_cast<uint16_t>(RequestKind::kGt);
  CHECK((r.flags & kRequestKindMask) == static_cast<uint16_t>(RequestKind::kGt));
}

TEST_CASE("instance sentinels follow DESIGN §5.1") {
  static_assert(kInstanceNone == 0u);
  static_assert(kInstanceInvalid == 0xFFFFFFFFu);
  const HitRecord miss{};  // a zero-initialized record reads as a miss
  CHECK(miss.instance == kInstanceNone);
}
