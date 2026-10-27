// Philox, Owen-scrambled Sobol and the pixel-response warp in batch form.
#include <nanobind/nanobind.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "bindings/core_bindings.h"
#include "bindings/ndarray_util.h"
#include "core/rng/philox.h"
#include "core/rng/sobol.h"
#include "core/rng/subsample.h"
#include "core/rng/warp.h"
#include "core/rng/warp_table.h"

namespace nb = nanobind;

namespace balla::bindings {
namespace {

// fp64 and fp32 tables of the same pixel response; the fp32 one is what the device uses.
struct Warp {
  rng::PixelResponse response;
  rng::WarpTable<double> table64;
  rng::WarpTable<float> table32;
};

Out<uint32_t> philox_batch(const ExactIn<uint32_t, -1, 4>& counter,
                           const ExactIn<uint32_t, -1, 2>& key) {
  const size_t n = counter.shape(0);
  if (key.shape(0) != n) throw std::invalid_argument("counter and key need the same number of rows");
  std::vector<uint32_t> out(4 * n);
  for (size_t i = 0; i < n; ++i) {
    const rng::U32x4 r = rng::philox4x32_10({{counter(i, 0), counter(i, 1), counter(i, 2), counter(i, 3)}},
                                            {key(i, 0), key(i, 1)});
    for (size_t k = 0; k < 4; ++k) out[4 * i + k] = r.v[k];
  }
  return to_numpy(std::move(out), {n, 4});
}

void check_dim(uint32_t dim) {
  if (dim >= rng::kSobolDimensions) throw std::invalid_argument("Sobol dimension must be < 8");
}

Out<uint32_t> sobol_batch(const ExactIn<uint32_t, -1>& index, uint32_t dim) {
  check_dim(dim);
  std::vector<uint32_t> out(index.shape(0));
  for (size_t i = 0; i < out.size(); ++i) out[i] = rng::sobol(index(i), dim);
  return to_numpy(std::move(out), {out.size()});
}

Out<uint32_t> owen_sobol_batch(const ExactIn<uint32_t, -1>& index, uint32_t dim,
                               uint32_t seed) {
  check_dim(dim);
  std::vector<uint32_t> out(index.shape(0));
  for (size_t i = 0; i < out.size(); ++i) out[i] = rng::owen_sobol(index(i), dim, seed);
  return to_numpy(std::move(out), {out.size()});
}

template <class T>
Out<T> inverse_cdf_batch(const rng::WarpTable<T>& table, const In<T, -1>& xi) {
  const rng::WarpTableView<T> view = table.view();
  std::vector<T> out(xi.shape(0));
  for (size_t i = 0; i < out.size(); ++i) {
    if (!(xi(i) >= T(0) && xi(i) < T(1))) throw std::invalid_argument("xi must lie in [0, 1)");
    out[i] = rng::warp_inverse_cdf(view, xi(i));
  }
  return to_numpy(std::move(out), {out.size()});
}

template <class T>
Out<T> offsets_batch(const rng::WarpTable<T>& table, uint32_t pattern_seed, uint32_t count) {
  const rng::WarpTableView<T> view = table.view();
  std::vector<T> out(2 * static_cast<size_t>(count));
  for (uint32_t k = 0; k < count; ++k) {
    const Vec2<T> o = rng::subsample_offset(view, pattern_seed, k);
    out[2 * k] = o.x;
    out[2 * k + 1] = o.y;
  }
  return to_numpy(std::move(out), {static_cast<size_t>(count), 2});
}

}  // namespace

void bind_rng(nb::module_& m) {
  m.def("philox4x32_10", &philox_batch, nb::arg("counter").noconvert(),
        nb::arg("key").noconvert(),
        "Philox4x32-10: counters (n, 4) uint32 and keys (n, 2) uint32 -> (n, 4) uint32.\n"
        "Inputs of any other dtype raise TypeError instead of wrapping.");
  m.def("sobol", &sobol_batch, nb::arg("index").noconvert(), nb::arg("dim"),
        "Unscrambled Sobol points (Joe-Kuo directions) as 32-bit fixed point, dim < 8.");
  m.def("owen_sobol", &owen_sobol_batch, nb::arg("index").noconvert(), nb::arg("dim"),
        nb::arg("seed"),
        "Shuffled Owen-scrambled Sobol points (Burley 2020) as 32-bit fixed point, dim < 8.");
  m.def("subsample_pattern_seed",
        [](uint64_t run_seed, uint32_t camera, uint32_t pixel) {
          if (camera > 0xFFu) throw std::invalid_argument("camera must be < 256 (the RNG counter keeps 8 camera bits)");
          return rng::subsample_pattern_seed(run_seed, camera, pixel);
        },
        nb::arg("run_seed"), nb::arg("camera"), nb::arg("pixel"),
        "Owen-Sobol scramble seed of a pixel's sub-sample pattern (Philox, purpose kSubsamplePattern), camera < 256.");

  nb::class_<Warp>(m, "WarpTable",
                   "Separable inverse CDF of box(w) * Gaussian(sigma), truncated at\n"
                   "w/2 + n_sigma sigma (DESIGN §6.1), tabulated in fp64 and fp32.")
      .def("__init__",
           [](Warp* self, double w, double sigma, double n_sigma, int32_t cells) {
             const rng::PixelResponse h(w, sigma, n_sigma);
             new (self) Warp{h, rng::WarpTable<double>(h, cells), rng::WarpTable<float>(h, cells)};
           },
           nb::arg("w") = kDefaultSettings.aperture_px, nb::arg("sigma") = kDefaultSettings.blur_sigma_px,
           nb::arg("n_sigma") = kDefaultSettings.truncation_sigmas, nb::arg("cells") = kDefaultSettings.warp_cells)
      .def_prop_ro("x_max", [](const Warp& w) { return w.response.x_max(); })
      .def("inverse_cdf", [](const Warp& w, const In<double, -1>& xi) { return inverse_cdf_batch(w.table64, xi); },
           nb::arg("xi"), "xi in [0, 1) -> offset (px), fp64 table.")
      .def("inverse_cdf_f32", [](const Warp& w, const In<float, -1>& xi) { return inverse_cdf_batch(w.table32, xi); },
           nb::arg("xi"), "xi in [0, 1) -> offset (px), fp32 table (the device path).")
      .def("subsample_offsets",
           [](const Warp& w, uint32_t seed, uint32_t count) { return offsets_batch(w.table64, seed, count); },
           nb::arg("pattern_seed"), nb::arg("count"),
           "Offsets (count, 2) of sub-samples 0..count-1 from the pixel centre (px), fp64.")
      .def("subsample_offsets_f32",
           [](const Warp& w, uint32_t seed, uint32_t count) { return offsets_batch(w.table32, seed, count); },
           nb::arg("pattern_seed"), nb::arg("count"), "As subsample_offsets, fp32.");
}

}  // namespace balla::bindings
