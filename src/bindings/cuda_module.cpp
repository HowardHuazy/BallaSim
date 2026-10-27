// ballasim._cuda: CUDA kernels of the shared core. Static cudart; importing works without a GPU
// and calls raise RuntimeError instead.
#include <nanobind/nanobind.h>

#include <vector>

#include "bindings/spline_batch.h"
#include "kernels/spline_eval.h"

namespace nb = nanobind;
using namespace balla;
using namespace balla::bindings;

namespace {

template <class T>
nb::dict evaluate(const SplineInput& s, std::vector<traj::SplineNode<T>> nodes,
                  const TimeArray& t_ns) {
  const OffsetBatch batch = make_offsets(s, t_ns);
  std::vector<traj::Kinematics<T>> out(batch.tau_ns.size());
  std::vector<uint8_t> valid(batch.tau_ns.size());
  {
    nb::gil_scoped_release release;
    kernels::spline_evaluate_device<T>(s.order, nodes, s.h_ns, static_cast<T>(1e9 / s.h_ns),
                                       batch.base_segment, batch.tau_ns, out, valid);
  }
  for (size_t i = 0; i < valid.size(); ++i) {
    if (!valid[i]) throw_out_of_span(t_ns(i));
  }
  return kinematics_to_dict(out);
}

nb::dict spline_eval_f32(int order, int64_t t0_ns, int64_t h_ns, const In<double, -1, 4>& q,
                         const In<double, -1, 3>& p, const In<double, 3>& origin,
                         const TimeArray& t_ns) {
  const SplineInput s = make_spline_input(order, t0_ns, h_ns, q, p, to_vec3(origin));
  return evaluate(s, traj::convert_nodes<float>(s.nodes), t_ns);
}

nb::dict spline_eval_f64(int order, int64_t t0_ns, int64_t h_ns, const In<double, -1, 4>& q,
                         const In<double, -1, 3>& p, const TimeArray& t_ns) {
  const SplineInput s = make_spline_input(order, t0_ns, h_ns, q, p, {0.0, 0.0, 0.0});
  return evaluate(s, s.nodes, t_ns);
}

}  // namespace

NB_MODULE(_cuda, m) {
  m.doc() = "BallaSim CUDA kernels";
  m.def("device_count", &kernels::cuda_device_count,
        "Number of visible CUDA devices (0 without a device or a usable driver).");
  m.def("spline_eval_f32", &spline_eval_f32, nb::arg("order"), nb::arg("t0_ns"), nb::arg("h_ns"),
        nb::arg("q"), nb::arg("p"), nb::arg("origin"), nb::arg("t_ns").noconvert(),
        "Device version of ballasim._core.spline_eval_f32 (same arguments, bitwise equal results).");
  m.def("spline_eval_f64", &spline_eval_f64, nb::arg("order"), nb::arg("t0_ns"), nb::arg("h_ns"),
        nb::arg("q"), nb::arg("p"), nb::arg("t_ns").noconvert(),
        "Device version of ballasim._core.spline_eval_f64, evaluated at int32 offsets from one\n"
        "knot-aligned base (the fp64 pose-table path); bitwise equal to the host results.");
}
