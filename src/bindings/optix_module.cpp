// ballasim._optix: the OptiX backend (built only when OPTIX_ROOT points to the user's SDK).
#include <nanobind/nanobind.h>

#include <stdexcept>
#include <vector>

#include "backend/optix/optix_mesh.h"
#include "bindings/ndarray_util.h"

namespace nb = nanobind;
using namespace balla;
using namespace balla::bindings;

namespace {

nb::dict intersect(const In<float, -1, 3>& vertices, const ExactIn<uint32_t, -1, 3>& indices,
                   const In<float, -1, 3>& origins, const In<float, -1, 3>& directions) {
  if (origins.shape(0) != directions.shape(0)) {
    throw std::invalid_argument("origins and directions need the same number of rows");
  }
  const std::vector<uint32_t> idx = checked_indices(indices, vertices.shape(0));
  backend::MeshHits hits;
  {
    nb::gil_scoped_release release;
    hits = backend::optix_intersect({vertices.data(), vertices.shape(0), idx.data(), indices.shape(0)},
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

NB_MODULE(_optix, m) {
  m.doc() = "BallaSim OptiX backend";
  m.def("intersect", &intersect, nb::arg("vertices"), nb::arg("indices").noconvert(), nb::arg("origins"),
        nb::arg("directions"),
        "Closest hits of rays against one triangle mesh with OptiX; indices (T, 3) uint32 only (same "
        "arguments and result format as ballasim._core.embree_intersect).");
}
