// Camera models in batch form (fp64): projection, Jacobians, unprojection, injective domains and
// the per-camera ray LUT.
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "bindings/core_bindings.h"
#include "bindings/ndarray_util.h"
#include "core/camera/injective_domain.h"
#include "core/camera/kannala_brandt.h"
#include "core/camera/radtan.h"
#include "core/camera/ray_lut.h"

namespace nb = nanobind;

namespace balla::bindings {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// The scene bundle's rule (render::validate): a negative or zero focal length would mirror or collapse the image.
void check_intrinsics(double fx, double fy, double cx, double cy) {
  if (!(std::isfinite(fx) && std::isfinite(fy) && fx > 0 && fy > 0 && std::isfinite(cx) && std::isfinite(cy))) {
    throw std::invalid_argument("camera: invalid intrinsics (fx, fy must be finite and > 0; cx, cy finite)");
  }
}

struct Radtan {
  camera::RadtanCamera<double> cam;
  camera::RadtanDomain domain;
  int width, height;
  double support_px;
};

struct KannalaBrandt {
  camera::KannalaBrandtCamera<double> cam;
  camera::KannalaBrandtDomain domain;
  int width, height;
  double support_px;
};

template <class Cam>
std::tuple<Out<double>, Out<bool>> project_batch(const Cam& cam, const In<double, -1, 3>& X) {
  const size_t n = X.shape(0);
  std::vector<double> uv(2 * n, kNaN);
  std::vector<uint8_t> valid(n, 0);
  for (size_t i = 0; i < n; ++i) {
    Vec2<double> pix;
    if (camera::project(cam, Vec3<double>{X(i, 0), X(i, 1), X(i, 2)}, &pix)) {
      uv[2 * i] = pix.x;
      uv[2 * i + 1] = pix.y;
      valid[i] = 1;
    }
  }
  return {to_numpy(std::move(uv), {n, 2}), to_numpy_bool(std::move(valid))};
}

template <class Cam>
std::tuple<Out<double>, Out<bool>> jacobian_batch(const Cam& cam, const In<double, -1, 3>& X) {
  const size_t n = X.shape(0);
  std::vector<double> jac(6 * n, kNaN);
  std::vector<uint8_t> valid(n, 0);
  for (size_t i = 0; i < n; ++i) {
    camera::Mat23<double> j;
    if (camera::project_jacobian(cam, Vec3<double>{X(i, 0), X(i, 1), X(i, 2)}, &j)) {
      for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c) jac[6 * i + 3 * static_cast<size_t>(r) + static_cast<size_t>(c)] = j.m[r][c];
      valid[i] = 1;
    }
  }
  return {to_numpy(std::move(jac), {n, 2, 3}), to_numpy_bool(std::move(valid))};
}

template <class Cam>
std::tuple<Out<double>, Out<bool>> unproject_batch(const Cam& cam, const In<double, -1, 2>& uv) {
  const size_t n = uv.shape(0);
  std::vector<double> rays(3 * n, kNaN);
  std::vector<uint8_t> valid(n, 0);
  for (size_t i = 0; i < n; ++i) {
    Vec3<double> ray;
    if (camera::unproject(cam, Vec2<double>{uv(i, 0), uv(i, 1)}, &ray)) {
      rays[3 * i] = ray.x;
      rays[3 * i + 1] = ray.y;
      rays[3 * i + 2] = ray.z;
      valid[i] = 1;
    }
  }
  return {to_numpy(std::move(rays), {n, 3}), to_numpy_bool(std::move(valid))};
}

Out<double> distort_jacobian_batch(const Radtan& r, const In<double, -1, 2>& xn) {
  const size_t n = xn.shape(0);
  std::vector<double> jac(4 * n);
  for (size_t i = 0; i < n; ++i) {
    const camera::Sym2<double> j = camera::distort_jacobian(r.cam, Vec2<double>{xn(i, 0), xn(i, 1)});
    jac[4 * i] = j.xx;
    jac[4 * i + 1] = j.xy;
    jac[4 * i + 2] = j.xy;
    jac[4 * i + 3] = j.yy;
  }
  return to_numpy(std::move(jac), {n, 2, 2});
}

Out<double> lut_lookup(const camera::RayLut<double>& lut, const In<double, -1, 2>& uv) {
  const size_t n = uv.shape(0);
  const camera::RayLutView<double> view = lut.view();
  std::vector<double> rays(3 * n);
  for (size_t i = 0; i < n; ++i) {
    const Vec3<double> r = camera::lut_ray(view, Vec2<double>{uv(i, 0), uv(i, 1)});
    rays[3 * i] = r.x;
    rays[3 * i + 1] = r.y;
    rays[3 * i + 2] = r.z;
  }
  return to_numpy(std::move(rays), {n, 3});
}

}  // namespace

void bind_camera(nb::module_& m) {
  nb::class_<camera::RayLut<double>>(m, "RayLut",
                                     "Per-camera ray LUT with bilinear lookup (fp64 nodes).")
      .def("lookup", &lut_lookup, nb::arg("uv"),
           "Rays (n, 3) at distorted pixel positions (n, 2).")
      .def_prop_ro("origin", [](const camera::RayLut<double>& l) {
        const auto v = l.view();
        return std::make_tuple(v.u0, v.v0);
      }, "Pixel position of node (0, 0).")
      .def_prop_ro("shape", [](const camera::RayLut<double>& l) {
        const auto v = l.view();
        return std::make_tuple(v.ny, v.nx);
      }, "(rows, columns) of the node grid.");

  nb::class_<Radtan>(m, "RadtanCamera",
                     "Pinhole + radial-tangential distortion (OpenCV k1, k2, p1, p2, k3), fp64.\n"
                     "The injective domain r_lim is computed at construction (M2 Theorem D1) and\n"
                     "covers pixel centres +- support_px; a camera whose support reaches beyond\n"
                     "0.99 r_mono raises ValueError.")
      .def("__init__",
           [](Radtan* self, double fx, double fy, double cx, double cy, double k1, double k2,
              double p1, double p2, double k3, int width, int height, double z_min,
              double support_px) {
             check_intrinsics(fx, fy, cx, cy);
             const camera::RadtanCamera<double> c{fx, fy, cx, cy, k1, k2, p1, p2, k3, 0.0, z_min};
             const camera::RadtanDomain d =
                 camera::radtan_injective_domain(c, width, height, support_px);
             new (self) Radtan{c, d, width, height, support_px};
             self->cam.r_lim = d.r_lim;
           },
           nb::arg("fx"), nb::arg("fy"), nb::arg("cx"), nb::arg("cy"), nb::arg("k1") = 0.0,
           nb::arg("k2") = 0.0, nb::arg("p1") = 0.0, nb::arg("p2") = 0.0, nb::arg("k3") = 0.0,
           nb::kw_only(), nb::arg("width"), nb::arg("height"), nb::arg("z_min") = 0.0,
           nb::arg("support_px") = kDefaultSupportPx)
      .def_ro("support_px", &Radtan::support_px,
              "Sub-sample reach around pixel centres (px) that the domain covers.")
      .def_prop_ro("r_mono", [](const Radtan& r) { return r.domain.r_mono; })
      .def_prop_ro("r_img", [](const Radtan& r) { return r.domain.r_img; })
      .def_prop_ro("r_lim", [](const Radtan& r) { return r.domain.r_lim; })
      .def("project", [](const Radtan& r, const In<double, -1, 3>& X) { return project_batch(r.cam, X); },
           nb::arg("X"), "Camera-frame points (n, 3) -> (pixels (n, 2), valid (n,)).")
      .def("project_jacobian",
           [](const Radtan& r, const In<double, -1, 3>& X) { return jacobian_batch(r.cam, X); },
           nb::arg("X"), "d pixel / d X_C (n, 2, 3) and validity.")
      .def("distort_jacobian", &distort_jacobian_batch, nb::arg("xn"),
           "J_d = d x_d / d x_n (n, 2, 2) at normalized coordinates.")
      .def("unproject",
           [](const Radtan& r, const In<double, -1, 2>& uv) { return unproject_batch(r.cam, uv); },
           nb::arg("uv"), "Pixels (n, 2) -> (rays (x_n, y_n, 1) (n, 3), valid (n,)).")
      .def("ray_lut",
           [](const Radtan& r, double max_offset_px, int samples_per_px) {
             return camera::RayLut<double>(r.cam, r.width, r.height, max_offset_px, samples_per_px);
           },
           nb::arg("max_offset_px"), nb::arg("samples_per_px") = 2,
           "Ray LUT covering pixel centres +- max_offset_px; raises ValueError if a node lies\n"
           "outside the injective domain.");

  nb::class_<KannalaBrandt>(m, "KannalaBrandtCamera",
                            "Kannala-Brandt equidistant fisheye (k1..k4), fp64.")
      .def("__init__",
           [](KannalaBrandt* self, double fx, double fy, double cx, double cy, double k1,
              double k2, double k3, double k4, int width, int height, double support_px) {
             check_intrinsics(fx, fy, cx, cy);
             const camera::KannalaBrandtCamera<double> c{fx, fy, cx, cy, k1, k2, k3, k4, 0.0};
             const camera::KannalaBrandtDomain d =
                 camera::kannala_brandt_injective_domain(c, width, height, support_px);
             new (self) KannalaBrandt{c, d, width, height, support_px};
             self->cam.theta_lim = d.theta_lim;
           },
           nb::arg("fx"), nb::arg("fy"), nb::arg("cx"), nb::arg("cy"), nb::arg("k1") = 0.0,
           nb::arg("k2") = 0.0, nb::arg("k3") = 0.0, nb::arg("k4") = 0.0, nb::kw_only(),
           nb::arg("width"), nb::arg("height"), nb::arg("support_px") = kDefaultSupportPx)
      .def_ro("support_px", &KannalaBrandt::support_px,
              "Sub-sample reach around pixel centres (px) that the domain covers.")
      .def_prop_ro("theta_mono", [](const KannalaBrandt& k) { return k.domain.theta_mono; })
      .def_prop_ro("theta_img", [](const KannalaBrandt& k) { return k.domain.theta_img; })
      .def_prop_ro("theta_lim", [](const KannalaBrandt& k) { return k.domain.theta_lim; })
      .def("project",
           [](const KannalaBrandt& k, const In<double, -1, 3>& X) { return project_batch(k.cam, X); },
           nb::arg("X"))
      .def("project_jacobian",
           [](const KannalaBrandt& k, const In<double, -1, 3>& X) { return jacobian_batch(k.cam, X); },
           nb::arg("X"))
      .def("unproject",
           [](const KannalaBrandt& k, const In<double, -1, 2>& uv) { return unproject_batch(k.cam, uv); },
           nb::arg("uv"), "Pixels (n, 2) -> (unit rays (n, 3), valid (n,)).")
      .def("ray_lut",
           [](const KannalaBrandt& k, double max_offset_px, int samples_per_px) {
             return camera::RayLut<double>(k.cam, k.width, k.height, max_offset_px, samples_per_px);
           },
           nb::arg("max_offset_px"), nb::arg("samples_per_px") = 2);
}

}  // namespace balla::bindings
