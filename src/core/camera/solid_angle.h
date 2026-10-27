// Natural vignetting of a camera model: the solid angle per image area at a ray, and the law built on it (P2-F F2,
// PHASE2 F079; SENSOR.md "Photometry"; optics.vignetting_law jacobian).
//
// A small fixed pupil at the projection centre, its normal the optical axis z, passes from a Lambertian patch of
// radiance L seen in direction w the flux L a cos(theta) dOmega into the image area dA that w's cone maps to (a the
// pupil's area, theta the angle of w to z), so the image irradiance is E ~ L cos(theta) dOmega/dA. dOmega/dA is the
// camera model's own. With J the 2x3 projection Jacobian (rows j0, j1; radtan.h, kannala_brandt.h) at the unit ray w
// and T = [t1 t2] a right-handed orthonormal basis of w's tangent plane, dA/dOmega = |det(J T)| px^2 per sr, and by
// Binet-Cauchy det(J T) = (j0 x j1).(t1 x t2) = (j0 x j1).w, so no basis is built:
//
//   rho(w) = 1 / |(j0 x j1).w|   (sr per px^2)          g(w) = cos(theta) rho(w) / rho(z)   (1 on the axis)
//
// Closed forms (tests/cpp/test_camera.cpp, V-LAW): pinhole cos^4(theta); radtan cos^4(theta) / det J_d (J_d = d x_d /
// d x_n at the undistorted point, the identity on the axis); Kannala-Brandt cos(theta) sin(theta) / (theta_d theta_d').
//
// Conventions. Blender Cycles and Eevee render no natural vignetting (compositor only). pbrt-v4's RealisticCamera
// weights each ray traced through the lens system by cos^4(theta) times the exit pupil's area, so a distortion's area
// magnification enters its image; its PerspectiveCamera weighs 1. Mitsuba 3's perspective and thinlens sensors weigh 1
// (sample_ray, checked on 3.9.1 scalar_rgb: P2-F F2 notes). This law is RealisticCamera's for a fixed small pupil, on
// the calibrated model instead of a lens prescription; cos^4 alone (optics.vignetting_law cos4) is the pinhole's.
// Not modelled: a pupil that moves or shrinks off axis (mechanical vignetting, pupil aberration).
#pragma once

#include "core/camera/types.h"
#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"

namespace balla::camera {

// Solid angle per pixel area (sr per px^2) at the unit ray w, |w| = 1, of a model with project_jacobian (RadtanCamera,
// KannalaBrandtCamera). False where the model does not project w (outside its injective domain) or J is singular.
template <class T, template <class> class Camera>
BALLA_HD bool solid_angle_density(const Camera<T>& cam, Vec3<T> w, T* density) {
  Mat23<T> j;
  if (!project_jacobian(cam, w, &j)) return false;
  const Vec3<T> j0{j.m[0][0], j.m[0][1], j.m[0][2]};
  const Vec3<T> j1{j.m[1][0], j.m[1][1], j.m[1][2]};
  const T area = abs(dot(cross(j0, j1), w));  // px^2 per sr
  if (!(area > T(0))) return false;
  *density = T(1) / area;
  return true;
}

// The natural-vignetting gain g(w) = cos(theta) rho(w) / rho(z) at the unit ray w. False as solid_angle_density, or
// where w looks 90 degrees or more off axis (the pupil seen edge-on or from behind).
template <class T, template <class> class Camera>
BALLA_HD bool vignetting_gain(const Camera<T>& cam, Vec3<T> w, T* gain) {
  T rho = T(0), rho_axis = T(0);
  if (!(w.z > T(0)) || !solid_angle_density(cam, w, &rho) ||
      !solid_angle_density(cam, Vec3<T>{T(0), T(0), T(1)}, &rho_axis)) {
    return false;
  }
  *gain = w.z * (rho / rho_axis);
  return true;
}

}  // namespace balla::camera
