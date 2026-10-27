// Unit quaternions for rotations: Hamilton convention, active, stored as (w, x, y, z)
// (DESIGN §5.1; never JPL). q represents R(q) with R(q) v = q (0, v) q*.
#pragma once

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"

namespace balla {

template <class T>
struct Quat {
  T w, x, y, z;
};

template <class T>
BALLA_HD constexpr Quat<T> quat_identity() {
  return {T(1), T(0), T(0), T(0)};
}

// Hamilton product: R(a * b) = R(a) R(b).
template <class T>
BALLA_HD constexpr Quat<T> operator*(Quat<T> a, Quat<T> b) {
  return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
          a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
          a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

template <class T>
BALLA_HD constexpr Quat<T> conjugate(Quat<T> q) {
  return {q.w, -q.x, -q.y, -q.z};
}

template <class T>
BALLA_HD constexpr T norm2(Quat<T> q) {
  return q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
}

template <class T>
BALLA_HD Quat<T> normalized(Quat<T> q) {
  const T s = T(1) / sqrt(norm2(q));
  return {s * q.w, s * q.x, s * q.y, s * q.z};
}

// R(q) v for unit q, as v + w t + u x t with t = 2 u x v (u = vector part).
template <class T>
BALLA_HD constexpr Vec3<T> rotate(Quat<T> q, Vec3<T> v) {
  const Vec3<T> u{q.x, q.y, q.z};
  const Vec3<T> t = T(2) * cross(u, v);
  return v + q.w * t + cross(u, t);
}

// R(q)^T v for unit q.
template <class T>
BALLA_HD constexpr Vec3<T> rotate_inverse(Quat<T> q, Vec3<T> v) {
  return rotate(conjugate(q), v);
}

template <class T>
BALLA_HD constexpr Mat3<T> to_matrix(Quat<T> q) {
  const T xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
  const T xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
  const T wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
  return {{{T(1) - T(2) * (yy + zz), T(2) * (xy - wz), T(2) * (xz + wy)},
           {T(2) * (xy + wz), T(1) - T(2) * (xx + zz), T(2) * (yz - wx)},
           {T(2) * (xz - wy), T(2) * (yz + wx), T(1) - T(2) * (xx + yy)}}};
}

// Shepperd's method: branch on the largest of (trace, R00, R11, R22) for a well-conditioned
// square root. Returns the representative with w >= 0, normalized.
template <class T>
BALLA_HD Quat<T> from_matrix(const Mat3<T>& r) {
  const T tr = r.m[0][0] + r.m[1][1] + r.m[2][2];
  Quat<T> q;
  if (tr >= r.m[0][0] && tr >= r.m[1][1] && tr >= r.m[2][2]) {
    const T s = T(2) * sqrt(T(1) + tr);
    q = {T(0.25) * s, (r.m[2][1] - r.m[1][2]) / s, (r.m[0][2] - r.m[2][0]) / s,
         (r.m[1][0] - r.m[0][1]) / s};
  } else if (r.m[0][0] >= r.m[1][1] && r.m[0][0] >= r.m[2][2]) {
    const T s = T(2) * sqrt(T(1) + r.m[0][0] - r.m[1][1] - r.m[2][2]);
    q = {(r.m[2][1] - r.m[1][2]) / s, T(0.25) * s, (r.m[0][1] + r.m[1][0]) / s,
         (r.m[0][2] + r.m[2][0]) / s};
  } else if (r.m[1][1] >= r.m[2][2]) {
    const T s = T(2) * sqrt(T(1) + r.m[1][1] - r.m[0][0] - r.m[2][2]);
    q = {(r.m[0][2] - r.m[2][0]) / s, (r.m[0][1] + r.m[1][0]) / s, T(0.25) * s,
         (r.m[1][2] + r.m[2][1]) / s};
  } else {
    const T s = T(2) * sqrt(T(1) + r.m[2][2] - r.m[0][0] - r.m[1][1]);
    q = {(r.m[1][0] - r.m[0][1]) / s, (r.m[0][2] + r.m[2][0]) / s, (r.m[1][2] + r.m[2][1]) / s,
         T(0.25) * s};
  }
  if (q.w < T(0)) q = {-q.w, -q.x, -q.y, -q.z};
  return normalized(q);
}

template <class U, class T>
BALLA_HD constexpr Quat<U> cast(Quat<T> q) {
  return {static_cast<U>(q.w), static_cast<U>(q.x), static_cast<U>(q.y), static_cast<U>(q.z)};
}

}  // namespace balla
