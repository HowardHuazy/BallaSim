// Per-camera ray LUT (DESIGN §8.1): rays unprojected in fp64 on a grid in distorted pixel
// coordinates and interpolated bilinearly, so ray generation costs one lookup for every camera
// model. The grid covers every sub-sample position: pixel centres +- max_offset (the warp's x_max).
//
// Rays are (x_n, y_n, 1) for radtan cameras (depth = ray parameter) and unit vectors for
// Kannala-Brandt (which reaches beyond 90 degrees). Every node must have a preimage in the
// camera's injective domain: the grid is the smallest node-aligned cover of the sub-sample
// support, so a node outside the domain would put NaN into rays of in-image sub-samples.
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

#include "core/hd.h"
#include "core/math/scalar.h"
#include "core/math/vec.h"

namespace balla::camera {

template <class T>
struct RayLutView {
  const Vec3<T>* rays;  // node (i, j) at (u0 + i / samples_per_px, v0 + j / samples_per_px)
  int32_t nx, ny;
  T u0, v0;
  T samples_per_px;
};

// Bilinear interpolation at a distorted pixel position; positions outside the grid extrapolate
// linearly from the border cell.
template <class T>
BALLA_HD Vec3<T> lut_ray(const RayLutView<T>& lut, Vec2<T> pixel) {
  const T fx = (pixel.x - lut.u0) * lut.samples_per_px;
  const T fy = (pixel.y - lut.v0) * lut.samples_per_px;
  // Clamped in T: converting floor(fx) beyond int32 is undefined (the host gives INT32_MIN, the
  // device saturates). max(T(0), NaN) is 0, and T(nx - 2) is exact (RayLut caps the grid at 2^24
  // nodes per axis).
  const T cx = min(max(T(0), floor(fx)), T(lut.nx - 2));
  const T cy = min(max(T(0), floor(fy)), T(lut.ny - 2));
  const int32_t i = static_cast<int32_t>(cx), j = static_cast<int32_t>(cy);
  const T tx = fx - cx, ty = fy - cy;
  const Vec3<T>* r0 = lut.rays + static_cast<int64_t>(j) * lut.nx + i;
  const Vec3<T>* r1 = r0 + lut.nx;
  const Vec3<T> bottom = (T(1) - tx) * r0[0] + tx * r0[1];
  const Vec3<T> top = (T(1) - tx) * r1[0] + tx * r1[1];
  return (T(1) - ty) * bottom + ty * top;
}

template <class T>
class RayLut {
 public:
  // Camera is RadtanCamera<double> or KannalaBrandtCamera<double>.
  template <class Camera>
  RayLut(const Camera& cam, int width, int height, double max_offset_px, int samples_per_px)
      : samples_per_px_(samples_per_px) {
    if (width <= 0 || height <= 0 || samples_per_px <= 0 || !(max_offset_px >= 0.0)) {
      throw std::invalid_argument("RayLut: invalid image size, offset or resolution");
    }
    const double margin =
        std::max(0.0, std::ceil((max_offset_px - 0.5) * samples_per_px) / samples_per_px);
    u0_ = -0.5 - margin;
    v0_ = -0.5 - margin;
    const double gx = (width + 2.0 * margin) * samples_per_px;
    const double gy = (height + 2.0 * margin) * samples_per_px;
    if (!(gx < 0x1p24 && gy < 0x1p24)) {
      throw std::invalid_argument(
          "RayLut: the node grid exceeds 2^24 nodes per axis; reduce samples_per_px or max_offset_px");
    }
    nx_ = static_cast<int32_t>(std::lround(gx)) + 1;
    ny_ = static_cast<int32_t>(std::lround(gy)) + 1;
    rays_.resize(static_cast<size_t>(nx_) * static_cast<size_t>(ny_));
    for (int32_t j = 0; j < ny_; ++j) {
      for (int32_t i = 0; i < nx_; ++i) {
        const Vec2<double> pixel{u0_ + static_cast<double>(i) / samples_per_px,
                                 v0_ + static_cast<double>(j) / samples_per_px};
        Vec3<double> ray;
        if (!unproject(cam, pixel, &ray)) {
          throw std::domain_error("RayLut: node (" + std::to_string(pixel.x) + ", " +
                                  std::to_string(pixel.y) +
                                  ") px lies outside the camera's injective domain; the camera "
                                  "is too close to its fold for this sub-sample support");
        }
        rays_[static_cast<size_t>(j) * static_cast<size_t>(nx_) + static_cast<size_t>(i)] =
            cast<T>(ray);
      }
    }
  }

  RayLutView<T> view() const {
    return {rays_.data(), nx_, ny_, static_cast<T>(u0_), static_cast<T>(v0_),
            static_cast<T>(samples_per_px_)};
  }
  const std::vector<Vec3<T>>& rays() const { return rays_; }

 private:
  int samples_per_px_;
  double u0_ = 0.0, v0_ = 0.0;
  int32_t nx_ = 0, ny_ = 0;
  std::vector<Vec3<T>> rays_;
};

}  // namespace balla::camera
