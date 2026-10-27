// Small matrix types shared by the camera models.
#pragma once

namespace balla::camera {

// Symmetric 2x2 matrix [[xx, xy], [xy, yy]] (the radtan distortion Jacobian is symmetric, M2 D7).
template <class T>
struct Sym2 {
  T xx, xy, yy;
};

// 2x3 Jacobian d(pixel)/d(X_C), row-major.
template <class T>
struct Mat23 {
  T m[2][3];
};

}  // namespace balla::camera
