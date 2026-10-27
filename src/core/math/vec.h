// Small fixed-size vectors and matrices for the shared maths, templated on float or double.
//
// Every operation spells out its evaluation order. With contraction disabled (FP policy), host and
// device then round identically, which the bitwise CPU/GPU parity contract (DESIGN §12.1 L1) needs.
#pragma once

#include "core/hd.h"
#include "core/math/scalar.h"

namespace balla {

template <class T>
struct Vec2 {
  T x, y;
};

template <class T>
struct Vec3 {
  T x, y, z;
};

template <class T>
BALLA_HD constexpr Vec2<T> operator+(Vec2<T> a, Vec2<T> b) {
  return {a.x + b.x, a.y + b.y};
}
template <class T>
BALLA_HD constexpr Vec2<T> operator-(Vec2<T> a, Vec2<T> b) {
  return {a.x - b.x, a.y - b.y};
}
template <class T>
BALLA_HD constexpr Vec2<T> operator*(T s, Vec2<T> a) {
  return {s * a.x, s * a.y};
}
template <class T>
BALLA_HD constexpr T dot(Vec2<T> a, Vec2<T> b) {
  return a.x * b.x + a.y * b.y;
}

template <class T>
BALLA_HD constexpr Vec3<T> operator+(Vec3<T> a, Vec3<T> b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
template <class T>
BALLA_HD constexpr Vec3<T> operator-(Vec3<T> a, Vec3<T> b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
template <class T>
BALLA_HD constexpr Vec3<T> operator-(Vec3<T> a) {
  return {-a.x, -a.y, -a.z};
}
template <class T>
BALLA_HD constexpr Vec3<T> operator*(T s, Vec3<T> a) {
  return {s * a.x, s * a.y, s * a.z};
}
template <class T>
BALLA_HD constexpr Vec3<T>& operator+=(Vec3<T>& a, Vec3<T> b) {
  a = a + b;
  return a;
}
template <class T>
BALLA_HD constexpr T dot(Vec3<T> a, Vec3<T> b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <class T>
BALLA_HD constexpr Vec3<T> cross(Vec3<T> a, Vec3<T> b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
template <class T>
BALLA_HD T norm(Vec3<T> a) {
  return sqrt(dot(a, a));
}

template <class U, class T>
BALLA_HD constexpr Vec3<U> cast(Vec3<T> a) {
  return {static_cast<U>(a.x), static_cast<U>(a.y), static_cast<U>(a.z)};
}
template <class U, class T>
BALLA_HD constexpr Vec2<U> cast(Vec2<T> a) {
  return {static_cast<U>(a.x), static_cast<U>(a.y)};
}

// Row-major 3x3 matrix: m[r][c].
template <class T>
struct Mat3 {
  T m[3][3];
};

template <class T>
BALLA_HD constexpr Vec3<T> operator*(const Mat3<T>& a, Vec3<T> v) {
  return {a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
          a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
          a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z};
}

}  // namespace balla
