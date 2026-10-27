// _render.bsdf_eval, bsdf_alpha_eff and bsdf_tables: render/shade/bsdf.h over NumPy arrays, for the conformance
// gates of DESIGN §7.11 (p2f.md §6.3, F3b) and the Mitsuba 3 plugin tools/mitsuba/ballasim_bsdf.py, which reads the
// tables here and evaluates its own expressions over them. Test tooling: no renderer path goes through these.
//
// Each call evaluates bsdf.h in fp64 (dtype "float64", the conformance side) or fp32 ("float32", the executors'
// precision: the inputs are rounded to fp32 first); results are returned as fp64 either way (exact for fp32).
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings/ndarray_util.h"
#include "render/shade/bsdf.h"
#include "render/shade/bsdf_tables.h"
#include "render/shade/bspline_table.h"

namespace balla::render::bindings {

namespace bsdf_detail {

namespace nb = nanobind;
using balla::bindings::In;

using Dirs = In<double, -1, 3>;
using Values = In<double, -1>;

// Element i of a per-row array that may also hold one value for every row.
inline double at(const Values& a, size_t i) { return a.shape(0) == 1 ? a(0) : a(i); }

inline size_t rows_of(std::initializer_list<const Dirs*> dirs, std::initializer_list<const Values*> values) {
  const size_t n = (*dirs.begin())->shape(0);
  for (const Dirs* d : dirs) {
    if (d->shape(0) != n) throw std::invalid_argument("bsdf: n, wv and wl need the same number of rows");
  }
  for (const Values* v : values) {
    if (v->shape(0) != n && v->shape(0) != 1) {
      throw std::invalid_argument("bsdf: each parameter needs one value per row, or one value");
    }
  }
  return n;
}

inline bool is_fp32(const std::string& dtype) {
  if (dtype == "float32") return true;
  if (dtype == "float64") return false;
  throw std::invalid_argument("bsdf: dtype must be 'float32' or 'float64'");
}

template <class T>
Vec3<T> row(const Dirs& a, size_t i) {
  return {static_cast<T>(a(i, 0)), static_cast<T>(a(i, 1)), static_cast<T>(a(i, 2))};
}

// f, and with parts the specular albedo E_s(mu_v), the lobe D V F and the tables' view coordinate t.
template <class T>
void eval_rows(const Dirs& n, const Dirs& wv, const Dirs& wl, const Values& rho, const Values& metallic,
               const Values& alpha2, const Values& s, size_t count, bool parts, std::vector<double>* out) {
  const bsdf::BsdfTables tables = bsdf::host_tables();
  for (size_t i = 0; i < count; ++i) {
    const Vec3<T> nn = row<T>(n, i), v = row<T>(wv, i), l = row<T>(wl, i);
    const bsdf::BsdfParams<T> p{static_cast<T>(at(rho, i)), static_cast<T>(at(metallic, i)),
                                static_cast<T>(at(alpha2, i)), static_cast<T>(at(s, i))};
    out[0][i] = static_cast<double>(bsdf::bsdf_eval(tables, p, nn, v, l));
    if (!parts) continue;
    const bsdf::ViewTerms<T> vt = bsdf::view_terms(nn, v);
    const bsdf::Fresnel<T> fr = bsdf::fresnel_params(p.rho, p.metallic, p.s);
    out[1][i] = static_cast<double>(bsdf::spec_albedo(fr, bsdf::spec_ab(tables, vt.t, bsdf::table_r(p.alpha2))));
    out[2][i] = static_cast<double>(bsdf::lobe(p.alpha2, fr, nn, v, vt, l));
    out[3][i] = static_cast<double>(vt.t);
  }
}

inline nb::object bsdf_eval(const Dirs& n, const Dirs& wv, const Dirs& wl, const Values& rho, const Values& metallic,
                            const Values& alpha2, const Values& s, const std::string& dtype, bool parts) {
  const bool fp32 = is_fp32(dtype);
  const size_t count = rows_of({&n, &wv, &wl}, {&rho, &metallic, &alpha2, &s});
  std::vector<double> out[4];
  for (int k = 0; k < (parts ? 4 : 1); ++k) out[k].resize(count);
  {
    nb::gil_scoped_release release;
    if (fp32) {
      eval_rows<float>(n, wv, wl, rho, metallic, alpha2, s, count, parts, out);
    } else {
      eval_rows<double>(n, wv, wl, rho, metallic, alpha2, s, count, parts, out);
    }
  }
  if (!parts) return nb::cast(balla::bindings::to_numpy(std::move(out[0]), {count}));
  nb::dict d;
  d["f"] = balla::bindings::to_numpy(std::move(out[0]), {count});
  d["e_s"] = balla::bindings::to_numpy(std::move(out[1]), {count});
  d["lobe"] = balla::bindings::to_numpy(std::move(out[2]), {count});
  d["t"] = balla::bindings::to_numpy(std::move(out[3]), {count});
  return d;
}

inline nb::object bsdf_alpha_eff(const Values& alpha2, const Values& alpha2_add_mean, const Values& r_s,
                                 const std::string& dtype) {
  const bool fp32 = is_fp32(dtype);
  size_t count = alpha2.shape(0);
  for (const Values* v : {&alpha2_add_mean, &r_s}) {
    if (count == 1) count = v->shape(0);
  }
  for (const Values* v : {&alpha2, &alpha2_add_mean, &r_s}) {
    if (v->shape(0) != count && v->shape(0) != 1) {
      throw std::invalid_argument("bsdf_alpha_eff: the arrays need the same length, or one value");
    }
  }
  std::vector<double> out(count);
  for (size_t i = 0; i < count; ++i) {
    out[i] = fp32 ? static_cast<double>(bsdf::alpha_eff(static_cast<float>(at(alpha2, i)),
                                                        static_cast<float>(at(alpha2_add_mean, i)),
                                                        static_cast<float>(at(r_s, i))))
                  : bsdf::alpha_eff(at(alpha2, i), at(alpha2_add_mean, i), at(r_s, i));
  }
  return nb::cast(balla::bindings::to_numpy(std::move(out), {count}));
}

inline nb::dict bsdf_tables() {
  const bsdf::BsdfTables t = bsdf::host_tables();
  const auto stride = static_cast<size_t>(kTableStride);
  nb::dict d;
  const std::pair<const char*, const float*> tables[] = {{"spec_a", t.spec_a},   {"spec_b", t.spec_b},
                                                          {"ltc_m00", t.ltc_m00}, {"ltc_m02", t.ltc_m02},
                                                          {"ltc_m20", t.ltc_m20}, {"ltc_m22", t.ltc_m22}};
  for (const auto& [name, data] : tables) {
    d[name] = balla::bindings::to_numpy(std::vector<float>(data, data + kTableSize), {stride, stride});
  }
  d["nodes"] = kTableNodes;
  return d;
}

}  // namespace bsdf_detail

inline void bind_bsdf(nanobind::module_& m) {
  namespace nb = nanobind;
  using namespace nb::literals;
  m.def("bsdf_eval", &bsdf_detail::bsdf_eval, "n"_a, "wv"_a, "wl"_a, "rho"_a, "metallic"_a, "alpha2"_a, "s"_a,
        nb::kw_only(), "dtype"_a = "float64", "parts"_a = false,
        "render/shade/bsdf.h's bsdf_eval per row: f(wv, wl) = (1 - m)(1 - E_s(mu_v)) rho/pi + D V F (1/sr), 0 for\n"
        "n . wl <= 0. n, wv (towards the camera) and wl (towards the light) are (N, 3) unit vectors; rho, metallic,\n"
        "alpha2 (alpha^2 = r^4, or alpha_eff^2 for a cluster) and s (the dielectric F0) are (N,) or one value.\n"
        "dtype 'float64' (the conformance side) or 'float32' (the executors' precision; inputs rounded to fp32\n"
        "first). Returns f (N,) float64; with parts=True a dict of f, e_s (E_s at the floored mu_v), lobe (D V F)\n"
        "and t (the tables' view coordinate sqrt(1 - sqrt(mu_v,e))). Test tooling (p2f.md §6.3).");
  m.def("bsdf_alpha_eff", &bsdf_detail::bsdf_alpha_eff, "alpha2"_a, "alpha2_add_mean"_a, "r_s"_a, nb::kw_only(),
        "dtype"_a = "float64",
        "render/shade/bsdf.h's alpha_eff: the floored and capped alpha_eff^2 of a cluster from alpha^2, the mean\n"
        "Toksvig alpha^2_add and the normal resultant r_s (arrays of one length, or one value each).");
  m.def("bsdf_tables", &bsdf_detail::bsdf_tables,
        "The BSDF's C2 cubic B-spline coefficient tables (render/shade/bspline_table.h): spec_a, spec_b (E_s = F0 A\n"
        "+ F90 B) and the LTC inverse entries ltc_m00, ltc_m02, ltc_m20, ltc_m22, each (66, 66) float32 stored\n"
        "[r][x] over (x, r) = (sqrt(1 - sqrt(mu_v)), sqrt(alpha)), the 64 nodes per axis at i / 63 with one ghost\n"
        "coefficient on each side (node i at index i + 1); 'nodes' = 64. Copies.");
}

}  // namespace balla::render::bindings
