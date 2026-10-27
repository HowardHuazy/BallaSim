// Kannala-Brandt equidistant fisheye model: theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8),
// x_d = theta_d (X, Y) / rho, rho = |(X, Y)|, theta = atan2(rho, Z) (M2 §D.2), valid beyond 90°.
// The injective domain is the cone theta <= theta_lim on which theta_d' > 0 (M2 §D.3).
#pragma once

#include "core/camera/types.h"
#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"
#include "core/pmath/trig.h"

namespace balla::camera {

template <class T>
struct KannalaBrandtCamera {
  T fx, fy, cx, cy;
  T k1, k2, k3, k4;
  T theta_lim;  // injective-domain half-angle (injective_domain.h)
};

template <class T>
BALLA_HD T theta_distorted(const KannalaBrandtCamera<T>& c, T theta) {
  const T t2 = theta * theta;
  return theta * (T(1) + t2 * (c.k1 + t2 * (c.k2 + t2 * (c.k3 + t2 * c.k4))));
}

template <class T>
BALLA_HD T theta_distorted_derivative(const KannalaBrandtCamera<T>& c, T theta) {
  const T t2 = theta * theta;
  return T(1) + t2 * (T(3) * c.k1 + t2 * (T(5) * c.k2 + t2 * (T(7) * c.k3 + t2 * (T(9) * c.k4))));
}

template <class T>
BALLA_HD bool project(const KannalaBrandtCamera<T>& c, Vec3<T> X, Vec2<T>* pixel) {
  const T rho = sqrt(X.x * X.x + X.y * X.y);
  if (!(rho > T(0) || X.z > T(0))) return false;  // origin or the backward axis
  const T theta = pmath::atan2(rho, X.z);
  if (!(theta <= c.theta_lim)) return false;
  const T s = rho > T(0) ? theta_distorted(c, theta) / rho : T(0);
  *pixel = {c.fx * (s * X.x) + c.cx, c.fy * (s * X.y) + c.cy};
  return true;
}

// d pixel / d X_C (M2 D10). Near the optical axis (rho <= eps Z) the model equals the pinhole to
// O(theta^2) = O(eps^2), so the pinhole Jacobian is used there instead of the 0/0 form.
template <class T>
BALLA_HD bool project_jacobian(const KannalaBrandtCamera<T>& c, Vec3<T> X, Mat23<T>* jac) {
  const T rho2 = X.x * X.x + X.y * X.y;
  const T rho = sqrt(rho2);
  if (!(rho > T(0) || X.z > T(0))) return false;
  const T theta = pmath::atan2(rho, X.z);
  if (!(theta <= c.theta_lim)) return false;
  T j[2][3];
  if (rho <= FloatTraits<T>::kEpsilon * X.z) {
    const T iz = T(1) / X.z;
    j[0][0] = iz;   j[0][1] = T(0); j[0][2] = -X.x * iz * iz;
    j[1][0] = T(0); j[1][1] = iz;   j[1][2] = -X.y * iz * iz;
  } else {
    const T mx = X.x / rho, my = X.y / rho;
    const T n2 = rho2 + X.z * X.z;
    const T gx = X.z * mx / n2, gy = X.z * my / n2, gz = -rho / n2;  // grad theta
    const T dthd = theta_distorted_derivative(c, theta);
    const T s = theta_distorted(c, theta) / rho;
    j[0][0] = dthd * mx * gx + s * (T(1) - mx * mx);
    j[0][1] = dthd * mx * gy - s * (mx * my);
    j[0][2] = dthd * mx * gz;
    j[1][0] = dthd * my * gx - s * (mx * my);
    j[1][1] = dthd * my * gy + s * (T(1) - my * my);
    j[1][2] = dthd * my * gz;
  }
  *jac = {{{c.fx * j[0][0], c.fx * j[0][1], c.fx * j[0][2]},
           {c.fy * j[1][0], c.fy * j[1][1], c.fy * j[1][2]}}};
  return true;
}

// Distorted pixel -> unit ray. theta_d is strictly increasing on [0, theta_lim], so Newton with
// bisection safeguarding inside the bracket always converges. False if the pixel lies beyond
// theta_d(theta_lim).
template <class T>
BALLA_HD bool unproject(const KannalaBrandtCamera<T>& c, Vec2<T> pixel, Vec3<T>* ray) {
  const Vec2<T> xd{(pixel.x - c.cx) / c.fx, (pixel.y - c.cy) / c.fy};
  const T rd = sqrt(dot(xd, xd));
  if (rd == T(0)) {
    *ray = {T(0), T(0), T(1)};
    return true;
  }
  if (!(rd <= theta_distorted(c, c.theta_lim))) return false;
  T lo = T(0), hi = c.theta_lim;
  T theta = min(rd, hi);
  const T tol = T(4) * FloatTraits<T>::kEpsilon * rd;
  for (int it = 0; it < 100; ++it) {
    const T f = theta_distorted(c, theta) - rd;
    if (abs(f) <= tol) break;
    if (f > T(0)) {
      hi = theta;
    } else {
      lo = theta;
    }
    T next = theta - f / theta_distorted_derivative(c, theta);
    if (!(next > lo && next < hi)) next = T(0.5) * (lo + hi);
    if (next == theta) break;
    theta = next;
  }
  T s, co;
  pmath::sincos(theta, &s, &co);
  *ray = {s * (xd.x / rd), s * (xd.y / rd), co};
  return true;
}

}  // namespace balla::camera
