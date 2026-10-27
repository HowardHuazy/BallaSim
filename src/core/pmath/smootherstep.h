// C2 smootherstep with identical results on host and device (DESIGN §7.3, §7.10).
#pragma once

#include "core/hd.h"
#include "core/math/scalar.h"

namespace balla::pmath {

// 6x^5 - 15x^4 + 10x^3 on [0, 1] (clamped outside): value, first and second derivatives vanish
// at both ends, which C2 blends need (the renderer's texture LOD blend and spot-light cone).
// Evaluated as 1 - S(1 - x) above 1/2 (S(1/2) = 1/2 exactly), so it is exactly antisymmetric
// about 1/2 and the rounding error near x = 1 is as small as near x = 0.
template <class T>
BALLA_HD T smootherstep(T x) {
  const bool upper = x > T(0.5);
  const T y = clamp(upper ? T(1) - x : x, T(0), T(1));
  const T s = y * y * y * (y * (y * T(6) - T(15)) + T(10));
  return upper ? T(1) - s : s;
}

}  // namespace balla::pmath
