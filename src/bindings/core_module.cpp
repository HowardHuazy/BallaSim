// ballasim._core: the CPU core (shared maths in batch form, Embree mesh queries).
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <stdexcept>
#include <vector>

#include "backend/embree/embree_mesh.h"
#include "bindings/core_bindings.h"
#include "bindings/ndarray_util.h"
#include "core/math/so3.h"
#include "core/time/time.h"

namespace nb = nanobind;
using namespace balla;
using balla::bindings::ExactIn;
using balla::bindings::In;
using balla::bindings::to_numpy;

namespace {

// DESIGN §4.6: the core is compiled for x86-64-v3; refuse to run on older CPUs.
void check_cpu() {
#if defined(__x86_64__) && defined(__AVX2__)
  __builtin_cpu_init();
  if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
    throw std::runtime_error("ballasim._core was built for x86-64-v3 (AVX2 + FMA3), which this "
                             "CPU lacks; rebuild with -DBALLASIM_CPU_ARCH=x86-64");
  }
#endif
}

// Which optional extension modules (ballasim._<name>) this build produced, from the build
// configuration: nothing is imported, so no CUDA context or OptiX library is touched.
nb::dict build_info() {
  nb::dict d;
  d["version"] = BALLASIM_VERSION;
  d["cuda"] = static_cast<bool>(BALLASIM_BUILT_WITH_CUDA);
  d["optix"] = static_cast<bool>(BALLASIM_BUILT_WITH_OPTIX);
  d["render"] = static_cast<bool>(BALLASIM_BUILT_WITH_RENDER);
  d["sensor"] = static_cast<bool>(BALLASIM_BUILT_WITH_SENSOR);
  d["engine"] = static_cast<bool>(BALLASIM_BUILT_WITH_ENGINE);
  d["scene"] = static_cast<bool>(BALLASIM_BUILT_WITH_SCENE);
  d["imu"] = static_cast<bool>(BALLASIM_BUILT_WITH_IMU);
  d["export"] = static_cast<bool>(BALLASIM_BUILT_WITH_EXPORT);
  d["frames"] = static_cast<bool>(BALLASIM_BUILT_WITH_FRAMES);
  d["compiler"] = std::string(__VERSION__);
  return d;
}

bindings::Out<double> so3_exp(const In<double, -1, 3>& phi) {
  const size_t n = phi.shape(0);
  std::vector<double> q(4 * n);
  for (size_t i = 0; i < n; ++i) {
    const Vec3<double> v{phi(i, 0), phi(i, 1), phi(i, 2)};
    if (!(dot(v, v) <= 4e12)) throw std::invalid_argument("so3_exp: need finite |phi| <= 2e6 rad (the sincos domain)");
    const Quat<double> r = so3::exp(v);
    q[4 * i] = r.w;
    q[4 * i + 1] = r.x;
    q[4 * i + 2] = r.y;
    q[4 * i + 3] = r.z;
  }
  return to_numpy(std::move(q), {n, 4});
}

bindings::Out<double> so3_log(const In<double, -1, 4>& q) {
  const size_t n = q.shape(0);
  std::vector<double> phi(3 * n);
  for (size_t i = 0; i < n; ++i) {
    const Vec3<double> r = so3::log(Quat<double>{q(i, 0), q(i, 1), q(i, 2), q(i, 3)});
    phi[3 * i] = r.x;
    phi[3 * i + 1] = r.y;
    phi[3 * i + 2] = r.z;
  }
  return to_numpy(std::move(phi), {n, 3});
}

nb::dict embree_intersect(const In<float, -1, 3>& vertices, const ExactIn<uint32_t, -1, 3>& indices,
                          const In<float, -1, 3>& origins, const In<float, -1, 3>& directions) {
  if (origins.shape(0) != directions.shape(0)) {
    throw std::invalid_argument("origins and directions need the same number of rows");
  }
  const std::vector<uint32_t> idx = bindings::checked_indices(indices, vertices.shape(0));
  backend::MeshHits hits;
  {
    nb::gil_scoped_release release;
    hits = backend::embree_intersect({vertices.data(), vertices.shape(0), idx.data(), indices.shape(0)},
                                     {origins.data(), directions.data(), origins.shape(0)});
  }
  const size_t n = hits.t.size();
  nb::dict d;
  d["t"] = to_numpy(std::move(hits.t), {n});
  d["prim"] = to_numpy(std::move(hits.prim), {n});
  d["u"] = to_numpy(std::move(hits.u), {n});
  d["v"] = to_numpy(std::move(hits.v), {n});
  return d;
}

}  // namespace

NB_MODULE(_core, m) {
  check_cpu();
  m.doc() = "BallaSim CPU core: shared BALLA_HD maths in batch form and the Embree mesh query";
  m.def("build_info", &build_info,
        "Version, compiler and the optional modules of this build: cuda, optix, render, sensor, engine, scene, imu, "
        "export and frames are True when ballasim._<name> was built (nothing is imported to find out).");
  m.def("so3_exp", &so3_exp, nb::arg("phi"),
        "Rotation vectors (n, 3) -> unit quaternions (n, 4) as (w, x, y, z), fp64.");
  m.def("so3_log", &so3_log, nb::arg("q"),
        "Quaternions (n, 4) (w, x, y, z) -> rotation vectors (n, 3) with |phi| <= pi, fp64.");
  m.def("embree_intersect", &embree_intersect, nb::arg("vertices"), nb::arg("indices").noconvert(),
        nb::arg("origins"), nb::arg("directions"),
        "Closest hits of rays against one triangle mesh with Embree; indices (T, 3) uint32 (other dtypes "
        "raise TypeError). Returns a dict of t (inf on a miss), prim (0xFFFFFFFF on a miss) and barycentrics u, v.");
  bindings::bind_traj(m);
  bindings::bind_camera(m);
  bindings::bind_rng(m);
  // The window length W = 2^23 ns (DESIGN §6.10); windows are aligned to the simulation epoch t = 0.
  m.attr("WINDOW_NS") = balla::time::kWindowNs;
  bindings::bind_config(m);
  bindings::bind_wire(m);
}
