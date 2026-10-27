// Pinhole camera with OpenCV radial-tangential distortion (k1, k2, p1, p2, k3), M2 §D.2-D.3.
//
// Forward projection is valid only on the injective domain D = {Z > z_min, |x_n| <= r_lim}
// (DESIGN §8.1; M2 Theorem D1). Outside D the radtan polynomial folds back, and out-of-view
// geometry would appear inside the image as ghosts. Pinhole is the special case with zero
// coefficients. Pixel centres sit at integer coordinates (DESIGN §5.1).
#pragma once

#include "core/camera/types.h"
#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"

namespace balla::camera {

template <class T>
struct RadtanCamera {
  T fx, fy, cx, cy;
  T k1, k2, p1, p2, k3;
  T r_lim;  // injective-domain radius in normalized coordinates (injective_domain.h)
  T z_min;  // points with Z <= z_min are not projected
};

template <class T>
BALLA_HD Vec2<T> distort(const RadtanCamera<T>& c, Vec2<T> xn) {
  const T x = xn.x, y = xn.y;
  const T rho = x * x + y * y;
  const T f = T(1) + rho * (c.k1 + rho * (c.k2 + rho * c.k3));
  const T xy = x * y;
  return {x * f + T(2) * c.p1 * xy + c.p2 * (rho + T(2) * x * x),
          y * f + c.p1 * (rho + T(2) * y * y) + T(2) * c.p2 * xy};
}

// J_d = d x_d / d x_n (M2 D7); symmetric because radtan is the gradient of a potential.
template <class T>
BALLA_HD Sym2<T> distort_jacobian(const RadtanCamera<T>& c, Vec2<T> xn) {
  const T x = xn.x, y = xn.y;
  const T rho = x * x + y * y;
  const T f = T(1) + rho * (c.k1 + rho * (c.k2 + rho * c.k3));
  const T fp = c.k1 + rho * (T(2) * c.k2 + T(3) * c.k3 * rho);
  return {f + T(2) * x * x * fp + T(2) * c.p1 * y + T(6) * c.p2 * x,
          T(2) * x * y * fp + T(2) * c.p1 * x + T(2) * c.p2 * y,
          f + T(2) * y * y * fp + T(6) * c.p1 * y + T(2) * c.p2 * x};
}

template <class T>
BALLA_HD bool in_domain(const RadtanCamera<T>& c, Vec3<T> X) {
  if (!(X.z > c.z_min)) return false;
  const T x = X.x / X.z, y = X.y / X.z;
  return x * x + y * y <= c.r_lim * c.r_lim;
}

// Camera-frame point -> distorted pixel. False outside the injective domain.
template <class T>
BALLA_HD bool project(const RadtanCamera<T>& c, Vec3<T> X, Vec2<T>* pixel) {
  if (!in_domain(c, X)) return false;
  const Vec2<T> xd = distort(c, Vec2<T>{X.x / X.z, X.y / X.z});
  *pixel = {c.fx * xd.x + c.cx, c.fy * xd.y + c.cy};
  return true;
}

// d pixel / d X_C = K_a J_d J_n (M2 D5-D8). False outside the injective domain.
template <class T>
BALLA_HD bool project_jacobian(const RadtanCamera<T>& c, Vec3<T> X, Mat23<T>* jac) {
  if (!in_domain(c, X)) return false;
  const T iz = T(1) / X.z;
  const T x = X.x * iz, y = X.y * iz;
  const Sym2<T> jd = distort_jacobian(c, Vec2<T>{x, y});
  // J_n = iz [[1, 0, -x], [0, 1, -y]]
  const T a00 = c.fx * jd.xx, a01 = c.fx * jd.xy;
  const T a10 = c.fy * jd.xy, a11 = c.fy * jd.yy;
  *jac = {{{a00 * iz, a01 * iz, -(a00 * x + a01 * y) * iz},
           {a10 * iz, a11 * iz, -(a10 * x + a11 * y) * iz}}};
  return true;
}

// Distorted pixel -> ray (x_n, y_n, 1) by safeguarded Newton on the monotone branch: every step
// is backtracked until it stays inside |x_n| <= r_lim and reduces the residual. J_d is positive
// definite there, so the preimage is unique (M2 §D.3). False if the pixel has no preimage in D.
template <class T>
BALLA_HD bool unproject(const RadtanCamera<T>& c, Vec2<T> pixel, Vec3<T>* ray) {
  const Vec2<T> xd{(pixel.x - c.cx) / c.fx, (pixel.y - c.cy) / c.fy};
  const T r2_lim = c.r_lim * c.r_lim;
  Vec2<T> x = xd;
  if (dot(x, x) > r2_lim) x = (T(0.5) * c.r_lim / sqrt(dot(x, x))) * x;
  Vec2<T> f = distort(c, x) - xd;
  T e = dot(f, f);
  // Newton converges quadratically; iterate until a step no longer reduces the residual, which
  // leaves it at rounding level, then accept if it is within a few ulp of |x_d|.
  const T scale = max(T(1), sqrt(dot(xd, xd)));
  const T floor_tol = FloatTraits<T>::kEpsilon * scale;
  for (int it = 0; it < 64 && e > floor_tol * floor_tol; ++it) {
    const Sym2<T> j = distort_jacobian(c, x);
    const T det = j.xx * j.yy - j.xy * j.xy;
    if (!(det > T(0))) return false;
    const Vec2<T> step{-(j.yy * f.x - j.xy * f.y) / det, -(j.xx * f.y - j.xy * f.x) / det};
    T lambda = T(1);
    bool accepted = false;
    for (int ls = 0; ls < 40 && !accepted; ++ls, lambda *= T(0.5)) {
      const Vec2<T> xn = x + lambda * step;
      if (dot(xn, xn) > r2_lim) continue;
      const Vec2<T> fn = distort(c, xn) - xd;
      const T en = dot(fn, fn);
      if (en < e) {
        x = xn;
        f = fn;
        e = en;
        accepted = true;
      }
    }
    if (!accepted) break;
  }
  const T accept_tol = T(8) * FloatTraits<T>::kEpsilon * scale;
  if (!(e <= accept_tol * accept_tol)) return false;
  *ray = {x.x, x.y, T(1)};
  return true;
}

}  // namespace balla::camera
