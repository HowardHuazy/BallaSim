// ballasim._scene: the scene side in C++ (docs/dev/CPP_PORT.md item 5): build_evimo2 (the EVIMO2 adapter's build, S4),
// import_ply / load_ply and evimo2_catalogue (S5), each bitwise as the Python it replaced, the test bindings in
// `testing` (one piece each, checked against NumPy, SciPy, the remaining Python and the frozen goldens), and the
// Blender motion session's TimeMap (docs/design/phase3/blender.md §4.2).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings/ndarray_util.h"
#include "scene/assets/mesh_stats.h"
#include "scene/assets/replica.h"
#include "scene/assets/ply.h"
#include "scene/blas.h"
#include "scene/errors.h"
#include "scene/evimo2/build.h"
#include "scene/evimo2/catalogue.h"
#include "scene/geometry.h"
#include "scene/kdtree.h"
#include "scene/motion/time_map.h"
#include "scene/npla.h"
#include "scene/npo.h"
#include "scene/pyfmt.h"
#include "scene/rotation.h"
#include "scene/sparse.h"
#include "scene/traj/banded.h"
#include "scene/traj/basis.h"
#include "scene/traj/fit_smooth.h"
#include "scene/traj/gating.h"
#include "scene/traj/holds.h"
#include "scene/traj/retime.h"
#include "scene/tree.h"

namespace nb = nanobind;
using namespace nb::literals;
using balla::bindings::ExactIn;
using balla::bindings::In;
using balla::bindings::Out;
using balla::bindings::to_numpy;
namespace sc = balla::scene;
namespace st = balla::scene::traj;
namespace ev = balla::scene::evimo2;
namespace geo = balla::scene::geo;
using sc::rot::M;
using sc::rot::Q;
using sc::rot::V;

namespace {

using F1 = In<double, -1>;
using F3 = In<double, -1, 3>;
using F4 = In<double, -1, 4>;
using I1 = ExactIn<int64_t, -1>;

std::vector<double> vec(const F1& a) { return std::vector<double>(a.data(), a.data() + a.shape(0)); }
std::vector<int64_t> ivec(const I1& a) {
  std::vector<int64_t> out(a.shape(0));
  for (size_t i = 0; i < out.size(); ++i) out[i] = a(i);
  return out;
}
std::vector<Q> quats(const F4& a) {
  std::vector<Q> out(a.shape(0));
  for (size_t i = 0; i < out.size(); ++i) out[i] = {a(i, 0), a(i, 1), a(i, 2), a(i, 3)};
  return out;
}
std::vector<V> vecs(const F3& a) {
  std::vector<V> out(a.shape(0));
  for (size_t i = 0; i < out.size(); ++i) out[i] = {a(i, 0), a(i, 1), a(i, 2)};
  return out;
}
V vec3(const In<double, 3>& a) { return {a(0), a(1), a(2)}; }

// Row-wise bindings take arrays of one length (NumPy would refuse to broadcast them).
void same_rows(const char* fn, size_t n, std::initializer_list<size_t> others) {
  for (const size_t m : others) {
    if (m != n) throw std::invalid_argument(std::string(fn) + ": arrays differ in length");
  }
}

Out<double> np1(std::vector<double> v) {
  const size_t n = v.size();
  return to_numpy(std::move(v), {n});
}
Out<int64_t> npi(std::vector<int64_t> v) {
  const size_t n = v.size();
  return to_numpy(std::move(v), {n});
}
Out<double> npq(const std::vector<Q>& q) {
  std::vector<double> out(4 * q.size());
  for (size_t i = 0; i < q.size(); ++i) {
    out[4 * i] = q[i].w;
    out[4 * i + 1] = q[i].x;
    out[4 * i + 2] = q[i].y;
    out[4 * i + 3] = q[i].z;
  }
  return to_numpy(std::move(out), {q.size(), 4});
}
Out<double> npv(const std::vector<V>& v) {
  std::vector<double> out(3 * v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    out[3 * i] = v[i].x;
    out[3 * i + 1] = v[i].y;
    out[3 * i + 2] = v[i].z;
  }
  return to_numpy(std::move(out), {v.size(), 3});
}
Out<double> npm(const std::vector<M>& m) {
  std::vector<double> out(9 * m.size());
  for (size_t i = 0; i < m.size(); ++i) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) out[9 * i + static_cast<size_t>(3 * r + c)] = m[i].m[r][c];
    }
  }
  return to_numpy(std::move(out), {m.size(), 3, 3});
}
nb::object npbool(const std::vector<uint8_t>& v) {
  return nb::module_::import_("numpy").attr("asarray")(npi(std::vector<int64_t>(v.begin(), v.end()))).attr("astype")("bool");
}
nb::list pairs(const std::vector<std::pair<int64_t, int64_t>>& v) {
  nb::list out;
  for (const auto& [a, b] : v) {
    nb::list p;
    p.append(a);
    p.append(b);
    out.append(p);
  }
  return out;
}
nb::list ilist(const std::vector<int64_t>& v) {
  nb::list out;
  for (const int64_t x : v) out.append(x);
  return out;
}
nb::list flist(const std::vector<double>& v) {
  nb::list out;
  for (const double x : v) out.append(x);
  return out;
}

// (data, indices, indptr, (rows, cols), (has_sorted_indices, has_canonical_format)) of a CSR/CSC matrix.
nb::tuple sparse_tuple(const sc::sparse::Matrix& m) {
  bool sorted = true;
  for (int64_t i = 0; i < m.major(); ++i) {
    for (int64_t jj = m.indptr[static_cast<size_t>(i)]; jj + 1 < m.indptr[static_cast<size_t>(i + 1)]; ++jj) {
      if (m.indices[static_cast<size_t>(jj)] > m.indices[static_cast<size_t>(jj + 1)]) sorted = false;
    }
  }
  return nb::make_tuple(np1(m.data), npi(m.indices), npi(m.indptr), nb::make_tuple(m.rows, m.cols),
                        nb::make_tuple(sorted, sc::sparse::has_canonical_format(m)),
                        m.format == sc::sparse::Format::kCsr ? "csr" : "csc");
}

sc::sparse::Matrix csr_in(const F1& data, const I1& indices, const I1& indptr, std::pair<int64_t, int64_t> shape) {
  return sc::sparse::csr(shape.first, shape.second, ivec(indptr), ivec(indices), vec(data));
}

nb::list intervals_list(const std::vector<st::StaticInterval>& v) {
  nb::list out;
  for (const auto& s : v) {
    nb::dict d;
    d["i0"] = s.i0;
    d["i1"] = s.i1;
    d["t_begin_ns"] = s.t_begin_ns;
    d["t_end_ns"] = s.t_end_ns;
    d["max_displacement"] = s.max_displacement;
    out.append(d);
  }
  return out;
}

nb::list rejected_list(const std::vector<st::RejectedCandidate>& v) {
  nb::list out;
  for (const auto& r : v) {
    nb::dict d;
    d["t_begin_ns"] = r.t_begin_ns;
    d["t_end_ns"] = r.t_end_ns;
    d["max_displacement"] = r.max_displacement;
    d["reason"] = r.reason;
    out.append(d);
  }
  return out;
}

std::optional<st::HoldView> hold_view(nb::handle view) {
  if (view.is_none()) return std::nullopt;
  nb::dict d = nb::cast<nb::dict>(view);
  st::HoldView v;
  v.focal_px = vec(nb::cast<F1>(d["focal_px"]));
  v.corner_r2 = vec(nb::cast<F1>(d["corner_r2"]));
  v.depth_m = vec(nb::cast<F1>(d["depth_m"]));
  if (d.contains("cameras_body_m") && !d["cameras_body_m"].is_none()) {
    v.cameras_body = vecs(nb::cast<F3>(d["cameras_body_m"]));
    v.has_cameras = true;
  }
  return v;
}

st::SmoothingOptions smoothing_options(nb::dict o, const In<double, 3>& center, nb::handle view) {
  st::SmoothingOptions s;
  auto get = [&o](const char* k) { return nb::handle(o[k]); };
  s.order = nb::cast<int>(get("order"));
  if (!get("h_ns").is_none()) s.h_ns = nb::cast<int64_t>(get("h_ns"));
  if (!get("span_ns").is_none()) {
    nb::sequence sp = nb::cast<nb::sequence>(get("span_ns"));
    s.span_ns = std::pair<int64_t, int64_t>{nb::cast<int64_t>(sp[0]), nb::cast<int64_t>(sp[1])};
  }
  s.loss_c2 = nb::cast<double>(get("loss_c2"));
  s.gate = nb::cast<bool>(get("gate"));
  s.gate_sigma_k = nb::cast<double>(get("gate_sigma_k"));
  s.gate_rot_floor_rad = nb::cast<double>(get("gate_rot_floor_rad"));
  s.gate_pos_floor_m = nb::cast<double>(get("gate_pos_floor_m"));
  s.mahalanobis_max = nb::cast<double>(get("mahalanobis_max"));
  if (!get("lambda_pos").is_none()) s.lambda_pos = nb::cast<double>(get("lambda_pos"));
  if (!get("lambda_rot").is_none()) s.lambda_rot = nb::cast<double>(get("lambda_rot"));
  nb::sequence g = nb::cast<nb::sequence>(get("gcv_log10_range"));
  s.gcv_lo = nb::cast<double>(g[0]);
  s.gcv_hi = nb::cast<double>(g[1]);
  s.detect_holds = nb::cast<bool>(get("detect_holds"));
  s.hold_max_disp_m = nb::cast<double>(get("hold_max_disp_m"));
  s.hold_min_ns = nb::cast<int64_t>(get("hold_min_ns"));
  s.hold_moving_mps = nb::cast<double>(get("hold_moving_mps"));
  s.hold_rule = nb::cast<std::string>(get("hold_rule"));
  s.hold_max_disp_px = nb::cast<double>(get("hold_max_disp_px"));
  s.branch_jump_rad = nb::cast<double>(get("branch_jump_rad"));
  s.branch_rate_factor = nb::cast<double>(get("branch_rate_factor"));
  s.radius_m = nb::cast<double>(get("radius_m"));
  s.sigma_floor_pos_m = nb::cast<double>(get("sigma_floor_pos_m"));
  s.sigma_floor_rot_rad = nb::cast<double>(get("sigma_floor_rot_rad"));
  s.max_irls = nb::cast<int>(get("max_irls"));
  s.center_body_m = vec3(center);
  s.hold_view = hold_view(view);
  return s;
}

nb::dict gcv_dict(const st::GcvInfo& g) {
  nb::dict d;
  if (g.searched) {
    d["log10_grid"] = flist(g.log10_grid);
    d["gcv"] = flist(g.gcv);
  }
  d["at_grid_edge"] = g.at_grid_edge;
  return d;
}

nb::dict diagnostics(const st::FitResult& r) {
  const st::FitDiagnostics& g = r.diag;
  nb::dict d;
  d["mode"] = "smoothing";
  d["num_samples"] = g.num_samples;
  if (g.gated) {
    nb::dict gt;
    int64_t rejected = 0;
    for (const uint8_t k : g.gate.keep) rejected += k ? 0 : 1;
    gt["num_rejected"] = rejected;
    gt["threshold_rot_rad"] = g.gate.threshold_rot;
    gt["threshold_pos_m"] = g.gate.threshold_pos;
    gt["half_window"] = g.gate.half_window;
    gt["sigma_rot_rad"] = g.gate.sigma_rot;
    gt["sigma_pos_m"] = g.gate.sigma_pos;
    gt["num_branch_rejected"] = g.gate.num_branch_rejected;
    gt["rejected_t_ns"] = ilist(g.gate_rejected_t_ns);
    d["gating"] = gt;
  }
  d["rotation_alias_warnings"] = g.rotation_alias_warnings;
  d["order"] = g.order;
  d["h_ns"] = g.h_ns;
  d["t0_ns"] = g.t0_ns;
  d["num_ctrl"] = g.num_ctrl;
  d["span_ns"] = ilist({g.span_ns.first, g.span_ns.second});
  d["lambda_pos_u"] = g.lambda_pos_u;
  d["lambda_rot_u"] = g.lambda_rot_u;
  d["lambda_pos_si"] = g.lambda_pos_si;
  d["lambda_rot_si"] = g.lambda_rot_si;
  d["q_pos"] = g.q_pos;
  d["q_rot"] = g.q_rot;
  d["cutoff_hz_pos"] = g.cutoff_hz_pos;
  d["cutoff_hz_rot"] = g.cutoff_hz_rot;
  d["edf_pos"] = g.edf_pos;
  d["edf_rot"] = g.edf_rot;
  d["gcv_pos"] = gcv_dict(g.gcv_pos);
  d["gcv_rot"] = gcv_dict(g.gcv_rot);
  d["sigma_pos_m"] = g.sigma_pos_m;
  d["sigma_rot_rad"] = g.sigma_rot_rad;
  d["rms_pos_m"] = g.rms_pos_m;
  d["rms_rot_rad"] = g.rms_rot_rad;
  d["max_pos_m"] = g.max_pos_m;
  d["max_rot_rad"] = g.max_rot_rad;
  d["irls_iterations"] = ilist(g.irls_iterations);
  d["min_weight"] = g.min_weight;
  d["mean_weight"] = g.mean_weight;
  d["num_kept"] = g.num_kept;
  d["gb_rejected_t_ns"] = ilist(g.gb_rejected_t_ns);
  d["hold_rule"] = g.hold_rule;
  d["holds_ns"] = pairs(g.holds_ns);
  d["static_intervals"] = intervals_list(g.static_intervals);
  d["rejected_static_candidates"] = rejected_list(g.rejected_static_candidates);
  d["presence_holds_ns"] = pairs(g.presence_holds_ns);
  d["still_max_displacement_m"] = g.still_max_displacement_m;
  d["max_increment_rad"] = g.max_increment_rad;
  d["gaps_ns"] = pairs(g.gaps_ns);
  d["static"] = g.is_static;
  if (g.presence_fallback) {
    nb::dict f;
    f["error"] = g.presence_fallback_error;
    f["groups"] = pairs(g.presence_fallback_groups);
    d["presence_fallback"] = f;
  }
  return d;
}

nb::dict fit_dict(const st::FitResult& r) {
  nb::dict d;
  d["order"] = r.order;
  d["t0_ns"] = r.t0_ns;
  d["h_ns"] = r.h_ns;
  d["rot"] = npq(r.rot);
  d["pos"] = npv(r.pos);
  std::vector<int64_t> holds;
  for (const auto& [a, b] : r.holds) {
    holds.push_back(a);
    holds.push_back(b);
  }
  d["holds"] = to_numpy(std::move(holds), {r.holds.size(), 2});
  d["span_ns"] = nb::make_tuple(r.span_ns.first, r.span_ns.second);
  d["keep"] = npbool(r.keep);
  d["weights"] = np1(r.weights);
  d["diagnostics"] = diagnostics(r);
  return d;
}

nb::dict fit_smoothing(const I1& t, const F4& q, const F3& p, nb::dict options, const In<double, 3>& center,
                       nb::handle view) {
  const std::vector<int64_t> tv = ivec(t);
  const std::vector<Q> qv = quats(q);
  const std::vector<V> pv = vecs(p);
  const st::SmoothingOptions o = smoothing_options(options, center, view);
  st::FitResult r;
  {
    nb::gil_scoped_release release;
    r = st::fit_smoothing(tv, qv, pv, o);
  }
  return fit_dict(r);
}

// --- the EVIMO2 build (ballasim.evimo2.adapter.build_bundle) -----------------------------------------------------

nb::object tree_to_py(const sc::Tree& t) {
  using K = sc::Tree::Kind;
  switch (t.kind()) {
    case K::kNull:
      return nb::none();
    case K::kBool:
      return nb::bool_(t.as_bool());
    case K::kInt:
      return nb::int_(t.as_int());
    case K::kFloat:
      return nb::float_(t.as_float());
    case K::kString:
      return nb::str(t.as_string().c_str(), t.as_string().size());
    case K::kList: {
      nb::list l;
      for (const sc::Tree& x : t.items()) l.append(tree_to_py(x));
      return l;
    }
    case K::kDict: {
      nb::dict d;
      for (const auto& [k, v] : t.entries()) d[nb::str(k.c_str(), k.size())] = tree_to_py(v);
      return d;
    }
  }
  return nb::none();
}

using F32x3 = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using F32x2 = nb::ndarray<const float, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu>;
using U32x3 = nb::ndarray<const uint32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using U16 = nb::ndarray<const uint16_t, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using M44 = nb::ndarray<const double, nb::shape<4, 4>, nb::c_contig, nb::device::cpu>;

// An array argument of exactly this dtype and shape (no conversion: a float64 mesh or an int32 row index is a
// caller error, not something to cast).
template <class A>
A exact(nb::handle h, const char* what) {
  A a;
  if (!nb::try_cast(h, a, false)) {
    throw std::invalid_argument(std::string("build_evimo2: ") + what + " has the wrong dtype, shape or layout");
  }
  return a;
}

geo::M4 m4_in(nb::handle h) {
  const auto a = exact<M44>(h, "a 4x4 transform");
  geo::M4 t;
  for (size_t i = 0; i < 16; ++i) t[i] = a.data()[i];
  return t;
}

Out<double> m4_out(const geo::M4& t) { return to_numpy(std::vector<double>(t.begin(), t.end()), {4, 4}); }

geo::Mesh mesh_in(nb::handle h) {
  const nb::tuple t = nb::cast<nb::tuple>(h);
  geo::Mesh m;
  const auto pos = exact<F32x3>(t[0], "mesh positions");
  m.positions.assign(pos.data(), pos.data() + 3 * pos.shape(0));
  const auto idx = exact<U32x3>(t[1], "mesh indices");
  m.indices.assign(idx.data(), idx.data() + 3 * idx.shape(0));
  for (const uint32_t i : m.indices) {
    if (i >= pos.shape(0)) throw std::invalid_argument("build_evimo2: mesh vertex index out of range");
  }
  if (!t[2].is_none()) {
    const auto n = exact<F32x3>(t[2], "mesh normals");
    if (n.shape(0) != pos.shape(0)) throw std::invalid_argument("build_evimo2: normals and positions differ in length");
    m.normals.assign(n.data(), n.data() + 3 * n.shape(0));
    m.has_normals = true;
  }
  if (!t[3].is_none()) {
    const auto u = exact<F32x2>(t[3], "mesh uvs");
    if (u.shape(0) != pos.shape(0)) throw std::invalid_argument("build_evimo2: uvs and positions differ in length");
    m.uvs.assign(u.data(), u.data() + 2 * u.shape(0));
    m.has_uvs = true;
  }
  if (!t[4].is_none()) {
    const auto mi = exact<U16>(t[4], "mesh material ids");
    if (mi.shape(0) != idx.shape(0)) {
      throw std::invalid_argument("build_evimo2: material ids and triangles differ in length");
    }
    m.material_ids.assign(mi.data(), mi.data() + mi.shape(0));
    m.has_material_ids = true;
  }
  return m;
}

nb::tuple mesh_out(geo::Mesh&& m) {
  const size_t nv = m.num_vertices(), nt = m.num_triangles();
  nb::object normals = nb::none(), uvs = nb::none(), mids = nb::none();
  if (m.has_normals) normals = nb::cast(to_numpy(std::move(m.normals), {nv, 3}));
  if (m.has_uvs) uvs = nb::cast(to_numpy(std::move(m.uvs), {nv, 2}));
  if (m.has_material_ids) mids = nb::cast(to_numpy(std::move(m.material_ids), {nt}));
  return nb::make_tuple(to_numpy(std::move(m.positions), {nv, 3}), to_numpy(std::move(m.indices), {nt, 3}), normals,
                        uvs, mids);
}

ev::Track track_in(nb::handle h) {
  const nb::tuple t = nb::cast<nb::tuple>(h);
  ev::Track tr;
  tr.rows = ivec(exact<I1>(t[0], "track rows"));
  tr.q = quats(nb::cast<F4>(t[1]));
  tr.p = vecs(nb::cast<F3>(t[2]));
  if (tr.q.size() != tr.rows.size() || tr.p.size() != tr.rows.size()) {
    throw std::invalid_argument("build_evimo2: a track's rows, q and p differ in length");
  }
  return tr;
}

ev::Inputs evimo2_inputs(nb::dict d) {
  ev::Inputs in;
  in.t_raw_ns = ivec(exact<I1>(d["t_raw_ns"], "t_raw_ns"));
  in.cam = track_in(d["cam"]);
  in.object_ids = nb::cast<std::vector<int64_t>>(d["object_ids"]);
  for (const auto& [k, v] : nb::cast<nb::dict>(d["tracks"])) in.tracks[nb::cast<int64_t>(k)] = track_in(v);
  for (const auto& [k, v] : nb::cast<nb::dict>(d["catalogue"])) {
    const nb::dict o = nb::cast<nb::dict>(v);
    ev::CatalogueObject c;
    c.name = nb::cast<std::string>(o["name"]);
    c.unit_scale = nb::cast<double>(o["unit_scale"]);
    c.config_markers = vecs(nb::cast<F3>(o["config_markers"]));
    c.vsk_markers_mm = vecs(nb::cast<F3>(o["vsk_markers_mm"]));
    c.marker_radius_mm = nb::cast<double>(o["marker_radius_mm"]);
    if (!o["mesh"].is_none()) {
      c.has_asset = true;
      c.mesh = mesh_in(o["mesh"]);
      c.num_materials = nb::cast<int>(o["num_materials"]);
      c.placeholder = nb::cast<bool>(o["placeholder"]);
    }
    in.catalogue[nb::cast<int64_t>(k)] = std::move(c);
  }
  for (nb::handle h : nb::cast<nb::list>(d["cameras"])) {
    const nb::dict c = nb::cast<nb::dict>(h);
    ev::CameraInput ci;
    ci.camera = nb::cast<std::string>(c["camera"]);
    ci.directory = nb::cast<std::string>(c["directory"]);
    ci.category = nb::cast<std::string>(c["category"]);
    ci.recording = nb::cast<std::string>(c["recording"]);
    ci.name = nb::cast<std::string>(c["name"]);
    ci.width = nb::cast<int64_t>(c["width"]);
    ci.height = nb::cast<int64_t>(c["height"]);
    ci.fx = nb::cast<double>(c["fx"]);
    ci.fy = nb::cast<double>(c["fy"]);
    ci.cx = nb::cast<double>(c["cx"]);
    ci.cy = nb::cast<double>(c["cy"]);
    const auto dist = nb::cast<In<double, 5>>(c["dist"]);
    for (size_t i = 0; i < 5; ++i) ci.dist[i] = dist(i);
    ci.T_rig_cam = m4_in(c["T_rig_cam"]);
    if (!c["g1"].is_none()) {
      const nb::tuple g = nb::cast<nb::tuple>(c["g1"]);
      ci.g1 = true;
      ci.g1_delta_ms = nb::cast<double>(g[0]);
      if (!g[1].is_none()) ci.g1_se_ms = nb::cast<double>(g[1]);
    }
    if (!c["raw_line3_s"].is_none()) ci.raw_line3_s = nb::cast<double>(c["raw_line3_s"]);
    in.cameras.push_back(std::move(ci));
  }
  return in;
}

ev::Options evimo2_options(nb::dict o) {
  ev::Options opts;
  opts.markers = nb::cast<bool>(o["markers"]);
  opts.marker_subdivisions = nb::cast<int>(o["marker_subdivisions"]);
  opts.clock_reference = nb::cast<std::string>(o["clock_reference"]);
  opts.warmup_ns = nb::cast<int64_t>(o["warmup_ns"]);
  if (!o["objects"].is_none()) opts.objects = nb::cast<std::vector<int64_t>>(o["objects"]);
  opts.min_samples = nb::cast<int64_t>(o["min_samples"]);
  opts.static_tabletop = nb::cast<bool>(o["static_tabletop"]);
  opts.static_tabletop_fraction = nb::cast<double>(o["static_tabletop_fraction"]);
  opts.lights = nb::cast<bool>(o["lights"]);
  opts.marker_albedo = nb::cast<double>(o["marker_albedo"]);
  opts.untextured_albedo = nb::cast<double>(o["untextured_albedo"]);
  if (!o["marker_radius_m"].is_none()) opts.marker_radius_m = nb::cast<double>(o["marker_radius_m"]);
  if (!o["pin_static_px_s"].is_none()) opts.pin_static_px_s = nb::cast<double>(o["pin_static_px_s"]);
  opts.workers = nb::cast<int>(o["workers"]);
  const nb::dict sm = nb::cast<nb::dict>(o["smoothing"]);
  opts.smoothing = smoothing_options(sm, nb::cast<In<double, 3>>(sm["center_body_m"]), nb::none());
  return opts;
}

nb::dict build_evimo2(nb::dict inputs, nb::dict options) {
  const ev::Inputs in = evimo2_inputs(inputs);
  const ev::Options opts = evimo2_options(options);
  ev::Build b;
  {
    nb::gil_scoped_release release;
    b = ev::build(in, opts);
  }
  nb::dict d;
  d["t_start_ns"] = b.t_start_ns;
  d["t_end_ns"] = b.t_end_ns;
  nb::list bodies;
  for (ev::Body& x : b.bodies) {
    nb::dict bd;
    bd["id"] = x.id;
    bd["name"] = x.name;
    bd["semantic"] = x.semantic;
    bd["mesh"] = mesh_out(std::move(x.mesh));
    bd["placeholder"] = x.placeholder;
    bd["asset_materials"] = x.asset_materials;
    bd["markers"] = x.markers;
    bd["trajectory"] = x.trajectory ? nb::object(nb::int_(*x.trajectory)) : nb::none();
    bd["T_world_body"] = m4_out(x.T_world_body);
    bodies.append(bd);
  }
  d["bodies"] = bodies;
  nb::list trajs;
  for (const size_t i : b.trajectories) trajs.append(i);
  d["trajectories"] = trajs;
  nb::list fits;
  for (size_t i = 0; i < b.fits.size(); ++i) fits.append(nb::make_tuple(b.fit_names[i], fit_dict(b.fits[i])));
  d["fits"] = fits;
  nb::list cams;
  for (const ev::Camera& c : b.cameras) {
    nb::dict cd;
    cd["name"] = c.name;
    cd["width"] = c.width;
    cd["height"] = c.height;
    cd["fx"] = c.fx;
    cd["fy"] = c.fy;
    cd["cx"] = c.cx;
    cd["cy"] = c.cy;
    cd["dist"] = np1(std::vector<double>(c.dist.begin(), c.dist.end()));
    cd["T_rig_cam"] = m4_out(c.T_rig_cam);
    cd["clock_offset_ns"] = c.clock_offset_ns;
    cd["sensor"] = c.sensor;
    cd["pixel_pitch_m"] = c.pixel_pitch_m;
    cams.append(cd);
  }
  d["cameras"] = cams;
  nb::dict rt;
  rt["t_raw_ns"] = npi(b.retiming.t_raw_ns);
  rt["frame_index"] = npi(b.retiming.frame_index);
  rt["t_ns"] = npi(b.retiming.t_ns);
  rt["a_ns"] = b.retiming.a_ns;
  rt["b_ns"] = b.retiming.b_ns;
  rt["residual_std_ns"] = b.retiming.residual_std_ns;
  rt["num_gaps"] = b.retiming.num_gaps;
  d["retiming"] = rt;
  d["T_world_c0"] = m4_out(b.T_world_c0);
  nb::dict tbs;
  for (const auto& [id, t] : b.T_body_subject) tbs[nb::int_(id)] = m4_out(t);
  d["T_body_subject"] = tbs;
  nb::list warnings;
  for (const std::string& w : b.warnings) warnings.append(w);
  d["warnings"] = warnings;
  d["sidecar"] = tree_to_py(b.sidecar);
  d["meta"] = tree_to_py(b.meta);
  return d;
}

// --- S5: PLY import and the EVIMO object catalogue ----------------------------------------------------------------

// The materials as keyword dicts of ballasim.scene.bundle.Material (Material(**m), scans.md §3.5): base_color,
// texture (uint8 or None), emissive, shading ("lit" or "baked"), texture_linear (float16 or None) and, away from 0,
// texture_tile.
nb::tuple asset_out(sc::assets::Asset&& a) {
  nb::list mats;
  nb::object float16;
  for (sc::assets::Material& m : a.materials) {
    nb::object tex = nb::none(), lin = nb::none();
    if (m.texture) {
      const size_t h = static_cast<size_t>(m.texture->height), w = static_cast<size_t>(m.texture->width);
      tex = nb::cast(to_numpy(std::move(m.texture->rgb), {h, w, 3}));
    }
    if (m.texture_linear) {
      // binary16 bits as uint16, viewed (not cast) as float16: the bits are the values.
      const size_t h = static_cast<size_t>(m.texture_linear->height), w = static_cast<size_t>(m.texture_linear->width);
      if (!float16.is_valid()) float16 = nb::module_::import_("numpy").attr("float16");
      lin = nb::cast(to_numpy(std::move(m.texture_linear->bits), {h, w, 3})).attr("view")(float16);
    }
    nb::dict d;
    d["base_color"] = to_numpy(std::vector<float>(m.base_color.begin(), m.base_color.end()), {3});
    d["texture"] = tex;
    d["emissive"] = to_numpy(std::vector<float>(m.emissive.begin(), m.emissive.end()), {3});
    d["shading"] = m.baked ? "baked" : "lit";
    d["texture_linear"] = lin;
    if (m.texture_tile != 0) d["texture_tile"] = m.texture_tile;
    mats.append(d);
  }
  return nb::make_tuple(mesh_out(std::move(a.mesh)), mats, tree_to_py(a.info));
}

nb::list catalogue_out(const std::vector<ev::CatalogueEntry>& entries) {
  nb::list out;
  for (const ev::CatalogueEntry& e : entries) {
    nb::dict d;
    d["id"] = e.id;
    d["name"] = e.name;
    d["folder"] = e.folder;
    d["unit_scale"] = e.unit_scale;
    d["mesh_path"] = e.mesh_path;
    d["config_markers"] = npv(e.config_markers);
    d["vsk_markers_mm"] = npv(e.vsk_markers_mm);
    nb::list names;
    for (const std::string& n : e.vsk_names) names.append(n);
    d["vsk_names"] = names;
    d["marker_radius_mm"] = e.marker_radius_mm;
    nb::dict s;
    for (const auto& [k, v] : e.settings) s[nb::str(k.c_str(), k.size())] = nb::str(v.c_str(), v.size());
    d["settings"] = s;
    out.append(d);
  }
  return out;
}

void bind_assets(nb::module_& m) {
  m.def("load_ply", [](const std::string& path, double scale, const std::string& uv_origin) {
    sc::assets::Asset a;
    {
      nb::gil_scoped_release release;
      a = sc::assets::load_ply(path, scale, uv_origin);
    }
    return asset_out(std::move(a));
  }, "path"_a, "scale"_a, "uv_origin"_a = "bottom",
        "ballasim.scene.assets.ply.load_ply in C++: (mesh tuple, [Material keyword dicts], info).");
  m.def("import_ply", [](const std::string& path, nb::handle transform, const std::string& uv_origin,
                         int64_t max_texture_size, bool baked) {
    const geo::M4 t = m4_in(transform);
    sc::assets::Asset a;
    {
      nb::gil_scoped_release release;
      sc::blas::OneThread one;
      a = sc::assets::import_ply(path, t, uv_origin, max_texture_size, baked);
    }
    return asset_out(std::move(a));
  }, "path"_a, "transform"_a, "uv_origin"_a, "max_texture_size"_a, "baked"_a = false,
        "ballasim.scene.assets._import of a PLY in C++: load_ply at scale 1, the units/axes transform (4x4, skipped\n"
        "when np.allclose to the identity), validate_mesh and texture limiting; (mesh tuple, [Material keyword dicts],\n"
        "info). baked: every material baked, textures halved to binary16 linear RGB (scans.md 3.5).");
  m.def("import_replica", [](const std::string& dir, int32_t tile_size, double split_size, int32_t tile_texels,
                             int32_t workers, double self_check_ratio) {
    sc::assets::ReplicaResult r;
    {
      nb::gil_scoped_release release;
      r = sc::assets::import_replica(dir, {tile_size, split_size}, {tile_texels, workers, self_check_ratio});
    }
    sc::assets::Asset a;                            // one part until S6 splits by object
    a.mesh = std::move(r.parts.at(0).mesh);
    a.materials.push_back(std::move(r.parts.at(0).material));
    a.info = std::move(r.info);
    return asset_out(std::move(a));
  }, "dir"_a, "tile_size"_a, "split_size"_a, "tile_texels"_a = 0, "workers"_a = 1, "self_check_ratio"_a = 2.0,
        "S5b-1 (docs/design/phase3/scans.md §4.5): a Replica scene directory as one baked scan mesh and one binary16 tile\n"
        "atlas (texture_tile = P); (mesh tuple, [Material keyword dicts], info). tile_size and split_size are\n"
        "textures/parameters.json's tileSize (must be 16) and splitSize; tile_texels is T: 0 auto, or 4, 8, 16.");
  m.def("replica_reports", [](const std::vector<std::vector<std::array<double, 3>>>& polygons,
                                const std::optional<std::array<double, 3>>& gravity) {
    return tree_to_py(sc::assets::replica_reports(polygons, gravity));
  }, "polygons"_a, "gravity"_a = nb::none());
  m.def("mesh_stats", [](nb::handle positions, nb::handle indices) {
    F32x3 pos;
    U32x3 idx;
    if (!nb::try_cast(positions, pos, false) || !nb::try_cast(indices, idx, false)) {
      throw std::invalid_argument(
          "mesh_stats: positions must be float32 (V, 3) and indices uint32 (T, 3), C-contiguous");
    }
    geo::Mesh mesh;
    mesh.positions.assign(pos.data(), pos.data() + 3 * pos.shape(0));
    mesh.indices.assign(idx.data(), idx.data() + 3 * idx.shape(0));
    sc::assets::MeshStats s;
    {
      nb::gil_scoped_release release;
      s = sc::assets::mesh_stats(mesh);
    }
    nb::dict d;
    d["triangles"] = s.triangles;
    d["bbox_m"] = std::vector<double>(s.bbox_m.begin(), s.bbox_m.end());
    d["open_boundary_m"] = s.open_boundary_m;
    d["nonmanifold_edges"] = s.nonmanifold_edges;
    d["welded_vertices"] = s.welded_vertices;
    d["degenerate_triangles"] = s.degenerate_triangles;
    return d;
  }, "positions"_a, "indices"_a,
        "S2c (docs/design/phase3/scans.md §3.5): an asset mesh's statistics for `ballasim scan report`, from float32\n"
        "positions (V, 3) and uint32 indices (T, 3): triangles, bbox_m (extent per axis), open_boundary_m (length of\n"
        "the edges with one face after welding bit-equal positions), nonmanifold_edges (edges with more than two\n"
        "faces), welded_vertices and degenerate_triangles (repeated welded corners: no edges).");
  m.def("evimo2_catalogue", [](const std::string& base) { return catalogue_out(ev::catalogue(base)); }, "base"_a,
        "ballasim.evimo2.dataset.catalogue(objects=base) in C++: one dict per object, in folder order.");
  m.def("evimo2_catalogue_files", &ev::catalogue_files, "base"_a,
        "ballasim.evimo2.dataset.catalogue_files(base) in C++.");
}

void bind_testing(nb::module_& t) {
  t.doc() = "Test bindings: each mirrors one Python function of the scene side bit for bit (no run path uses them).";
  // NumPy order.
  t.def("np_sum", [](const F1& a) { return sc::npo::sum(vec(a)); });
  // NumPy's float64 -> float16 cast (.view(np.uint16)): the rounding of baked texture_linear (scans.md 3.5).
  t.def("binary16_bits", [](const F1& a) {
    std::vector<uint16_t> out(a.shape(0));
    for (size_t i = 0; i < out.size(); ++i) out[i] = sc::assets::binary16_bits(a(i));
    const size_t n = out.size();
    return to_numpy(std::move(out), {n});
  });
  t.def("np_median", [](const F1& a) { return sc::npo::median(vec(a)); });
  t.def("np_median_i64", [](const I1& a) { const auto v = ivec(a); return sc::npo::median(std::span<const int64_t>(v)); },
        "a"_a.noconvert());
  t.def("np_percentile", [](const F1& a, double q) { return sc::npo::percentile(vec(a), q); });
  t.def("np_std", [](const F1& a) { return sc::npo::std0(vec(a)); });
  t.def("py_pow", &sc::npo::py_pow);
  t.def("np_arange", [](double a, double b, double s) { return np1(sc::npo::arange(a, b, s)); });
  t.def("np_polyfit2", [](const F1& x, const F1& y) {
    if (x.shape(0) != 3 || y.shape(0) != 3) throw std::invalid_argument("three points");
    return np1(sc::npo::polyfit2(x.data(), y.data()));
  });
  t.def("np_lstsq", [](const In<double, -1, -1>& a, const F1& b, double rcond) {
    std::vector<double> av(a.data(), a.data() + a.shape(0) * a.shape(1));
    return np1(sc::npo::lstsq(av, static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)), vec(b), rcond));
  });
  // NumPy's matmul and linalg (any strides: the operand as NumPy holds it, a view included).
  using A2 = nb::ndarray<const double, nb::ndim<2>, nb::device::cpu>;
  auto view2 = [](const A2& a) {
    return sc::npo::View2{a.data(), static_cast<int64_t>(a.shape(0)), static_cast<int64_t>(a.shape(1)),
                          static_cast<int64_t>(a.stride(0)), static_cast<int64_t>(a.stride(1))};
  };
  t.def("np_matmul", [view2](const A2& a, const A2& b) {
    sc::blas::OneThread one;
    const auto va = view2(a), vb = view2(b);
    auto out = sc::npo::matmul(va, vb);
    return to_numpy(std::move(out), {static_cast<size_t>(va.rows), static_cast<size_t>(vb.cols)});
  }, "a"_a.noconvert(), "b"_a.noconvert());
  t.def("np_inv", [](const In<double, -1, -1>& a) {
    const int n = static_cast<int>(a.shape(0));
    if (a.shape(1) != a.shape(0)) throw std::invalid_argument("square");
    auto out = sc::npo::inv(a.data(), n);
    return to_numpy(std::move(out), {static_cast<size_t>(n), static_cast<size_t>(n)});
  });
  t.def("np_det", [](const In<double, -1, -1>& a) {
    if (a.shape(1) != a.shape(0)) throw std::invalid_argument("square");
    return sc::npo::det(a.data(), static_cast<int>(a.shape(0)));
  });
  t.def("np_svd", [](const In<double, -1, -1>& a, bool compute_uv) {
    const int m = static_cast<int>(a.shape(0)), n = static_cast<int>(a.shape(1)), k = std::min(m, n);
    std::vector<double> u(static_cast<size_t>(m * m)), s(static_cast<size_t>(k)), vt(static_cast<size_t>(n * n));
    if (!compute_uv) {
      sc::npo::svd_values(a.data(), m, n, s.data());
      return nb::make_tuple(np1(std::move(s)));
    }
    sc::npo::svd(a.data(), m, n, u.data(), s.data(), vt.data());
    return nb::make_tuple(to_numpy(std::move(u), {static_cast<size_t>(m), static_cast<size_t>(m)}), np1(std::move(s)),
                          to_numpy(std::move(vt), {static_cast<size_t>(n), static_cast<size_t>(n)}));
  });
  t.def("np_linspace", [](double a, double b, int64_t num) { return np1(sc::npo::linspace(a, b, num)); });
  t.def("blas_dot", [](const F1& a, const F1& b) {
    if (a.shape(0) != b.shape(0)) throw std::invalid_argument("lengths differ");
    sc::blas::OneThread one;
    return sc::blas::dot(static_cast<int>(a.shape(0)), a.data(), b.data());
  });
  // Rotations (row-wise over arrays).
  t.def("rotation_angle", [](const F4& a, const F4& b) {
    same_rows("rotation_angle", a.shape(0), {b.shape(0)});
    const auto qa = quats(a), qb = quats(b);
    std::vector<double> out(qa.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = sc::rot::rotation_angle(qa[i], qb[i]);
    return np1(out);
  });
  t.def("quat_rotate", [](const F4& q, const F3& v) {
    same_rows("quat_rotate", q.shape(0), {v.shape(0)});
    const auto qs = quats(q);
    const auto vs = vecs(v);
    std::vector<V> out(qs.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = sc::rot::quat_rotate(qs[i], vs[i]);
    return npv(out);
  });
  t.def("relative_rotvec", [](const F4& a, const F4& b) {
    same_rows("relative_rotvec", a.shape(0), {b.shape(0)});
    const auto qa = quats(a), qb = quats(b);
    std::vector<V> out(qa.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = sc::rot::relative_rotvec(qa[i], qb[i]);
    return npv(out);
  });
  t.def("canonical_quat", [](const F4& q) {
    auto qs = quats(q);
    for (auto& x : qs) x = sc::rot::canonical_quat(x);
    return npq(qs);
  });
  t.def("quat_to_matrix", [](const F4& q) {
    const auto qs = quats(q);
    std::vector<M> out(qs.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = sc::rot::quat_to_matrix(qs[i]);
    return npm(out);
  });
  t.def("matrix_to_quat", [](const In<double, -1, 3, 3>& r) {
    std::vector<Q> out(r.shape(0));
    for (size_t i = 0; i < out.size(); ++i) {
      M m;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) m.m[a][b] = r(i, a, b);
      }
      out[i] = sc::rot::canonical_quat(sc::rot::sp_from_matrix(m));
    }
    return npq(out);
  });
  t.def("right_jacobian", [](const F3& phi, bool inverse) {
    const auto vs = vecs(phi);
    std::vector<M> out(vs.size());
    sc::blas::OneThread one;
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = inverse ? sc::rot::right_jacobian_inv(vs[i]) : sc::rot::right_jacobian(vs[i]);
    }
    return npm(out);
  });
  t.def("robust_mean_pose", [](const F4& q, const F3& p) {
    sc::blas::OneThread one;
    const auto qs = quats(q);
    const auto ps = vecs(p);
    const auto [qm, pm] = sc::rot::robust_mean_pose(qs, ps);
    return nb::make_tuple(npq({qm}), npv({pm}));
  });
  t.def("rotation_mean", [](const F4& q) {
    sc::blas::OneThread one;
    std::vector<Q> qs = quats(q);
    for (auto& x : qs) x = sc::rot::sp_normalize(x);
    return npq({sc::rot::sp_mean(qs)});
  });
  t.def("slerp_init", [](const I1& t_ns, const F4& q, const I1& at) {
    return npq(sc::rot::slerp_init(ivec(t_ns), quats(q), ivec(at)));
  }, "t_ns"_a.noconvert(), "q"_a, "at_ns"_a.noconvert());
  // Bases and penalties.
  t.def("basis", [](const F1& u, int order, bool cumulative) {
    sc::blas::OneThread one;
    const auto uv = vec(u);
    std::vector<double> b = cumulative ? st::cumulative_basis(uv, order) : st::basis(uv, order);
    const size_t cols = static_cast<size_t>(cumulative ? order - 1 : order);
    return to_numpy(std::move(b), {uv.size(), cols});
  });
  t.def("segment_gram", [](int order, int m) {
    const auto g = st::segment_gram(order, m);
    const size_t k = static_cast<size_t>(order - m);
    return to_numpy(std::vector<double>(g.begin(), g.end()), {k, k});
  });
  t.def("design_matrix", [](int64_t t0, int64_t h, int order, int64_t n, const I1& t_ns) {
    sc::blas::OneThread one;
    return sparse_tuple(st::KnotGrid(t0, h, order, n).design_matrix(ivec(t_ns)));
  }, "t0_ns"_a, "h_ns"_a, "order"_a, "num_ctrl"_a, "t_ns"_a.noconvert());
  t.def("roughness_matrix", [](int64_t n, int k, int m) { return sparse_tuple(st::roughness_matrix(n, k, m)); });
  t.def("difference_gram", [](int64_t n, int k, int m) { return sparse_tuple(st::difference_gram(n, k, m)); });
  t.def("difference_matrix", [](int64_t n, int order) { return sparse_tuple(st::difference_matrix(n, order)); });
  t.def("kron_identity3", [](const F1& data, const I1& indices, const I1& indptr, std::pair<int64_t, int64_t> shape) {
    return sparse_tuple(sc::sparse::kron_identity3(csr_in(data, indices, indptr, shape)));
  }, "data"_a, "indices"_a.noconvert(), "indptr"_a.noconvert(), "shape"_a);
  t.def("sparse_matmul", [](const F1& ad, const I1& ai, const I1& ap, std::pair<int64_t, int64_t> ash, bool at,
                            const F1& bd, const I1& bi, const I1& bp, std::pair<int64_t, int64_t> bsh, bool bt) {
    sc::sparse::Matrix a = csr_in(ad, ai, ap, ash), b = csr_in(bd, bi, bp, bsh);
    if (at) a = sc::sparse::transpose(a);
    if (bt) b = sc::sparse::transpose(b);
    return sparse_tuple(sc::sparse::matmul(a, b));
  }, "a_data"_a, "a_indices"_a.noconvert(), "a_indptr"_a.noconvert(), "a_shape"_a, "a_transpose"_a, "b_data"_a,
     "b_indices"_a.noconvert(), "b_indptr"_a.noconvert(), "b_shape"_a, "b_transpose"_a);
  t.def("greville_ns", [](int64_t t0, int64_t h, int order, int64_t n) {
    return npi(st::KnotGrid(t0, h, order, n).greville_ns());
  });
  t.def("knot_grid_covering", [](int64_t a, int64_t b, int64_t h, int order) {
    const auto g = st::KnotGrid::covering(a, b, h, order);
    return nb::make_tuple(g.t0_ns, g.h_ns, g.order, g.num_ctrl);
  });
  // Re-timing, gating, holds.
  t.def("retime_stamps", [](const I1& t_raw) {
    sc::blas::OneThread one;
    const st::Retiming r = st::retime_stamps(ivec(t_raw));
    nb::dict d;
    d["t_ns"] = npi(r.t_ns);
    d["frame_index"] = npi(r.frame_index);
    d["a_ns"] = r.a_ns;
    d["b_ns"] = r.b_ns;
    d["residual_std_ns"] = r.residual_std_ns;
    d["num_gaps"] = r.num_gaps;
    return d;
  }, "t_raw"_a.noconvert());
  t.def("gate_poses", [](const I1& t_ns, const F4& q, const F3& p, double period_ns, double sigma_k, double rot_floor,
                         double pos_floor, double branch_jump, double branch_rate) {
    sc::blas::OneThread one;
    const auto tv = ivec(t_ns);
    const auto qv = quats(q);
    const auto pv = vecs(p);
    const st::GateResult g =
        st::gate_poses(tv, qv, pv, {period_ns, sigma_k, rot_floor, pos_floor, branch_jump, branch_rate});
    nb::dict d;
    d["keep"] = npbool(g.keep);
    d["err_rot"] = np1(g.err_rot);
    d["err_pos"] = np1(g.err_pos);
    d["scalars"] = np1({g.threshold_rot, g.threshold_pos, static_cast<double>(g.half_window), g.sigma_rot, g.sigma_pos,
                        static_cast<double>(g.num_branch_rejected)});
    return d;
  }, "t_ns"_a.noconvert(), "q"_a, "p"_a, "period_ns"_a, "sigma_k"_a, "rot_floor_rad"_a, "pos_floor_m"_a,
     "branch_jump_rad"_a, "branch_rate_factor"_a);
  t.def("sphere_speed", [](const F3& v, const F3& omega, const F4& q, const In<double, 3>& center, double radius) {
    same_rows("sphere_speed", v.shape(0), {omega.shape(0), q.shape(0)});
    const auto vs = vecs(v), ws = vecs(omega);
    const auto qs = quats(q);
    std::vector<double> out(vs.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = st::sphere_speed(vs[i], ws[i], qs[i], vec3(center), radius);
    return np1(out);
  });
  t.def("detect_static", [](const I1& t_ns, const F4& q, const F3& p, const F1& speed, const In<double, 3>& center,
                            double radius, double max_disp, int64_t min_duration_ns, double moving_speed) {
    sc::blas::OneThread one;
    st::StaticOptions o;
    o.max_disp = max_disp;
    o.min_duration_ns = min_duration_ns;
    o.moving_speed = moving_speed;
    const auto tv = ivec(t_ns);
    const auto qv = quats(q);
    const auto pv = vecs(p);
    const auto sv = vec(speed);
    const st::HoldResult h = st::detect_static(tv, qv, pv, sv, vec3(center), radius, o);
    return nb::make_tuple(intervals_list(h.accepted), rejected_list(h.rejected));
  }, "t_ns"_a.noconvert(), "q"_a, "p"_a, "speed"_a, "center_body_m"_a, "radius_m"_a, "max_disp"_a,
     "min_duration_ns"_a, "moving_speed"_a);
  t.def("detect_static_image", [](const I1& t_ns, const F4& q, const F3& p, const F3& v, const F3& omega, nb::dict view,
                                  const I1& idx, const In<double, 3>& center, double radius, int64_t min_duration_ns,
                                  double max_disp_px, int64_t min_piece_ns) {
    sc::blas::OneThread one;
    const auto hv = hold_view(view);
    const auto tv = ivec(t_ns);
    const auto qv = quats(q);
    const auto pv = vecs(p), vv = vecs(v), wv = vecs(omega);
    const auto iv = ivec(idx);
    const st::HoldResult h = st::detect_static_image(tv, qv, pv, vv, wv, *hv, iv, vec3(center), radius, min_duration_ns,
                                                     max_disp_px, min_piece_ns);
    return nb::make_tuple(intervals_list(h.accepted), rejected_list(h.rejected));
  }, "t_ns"_a.noconvert(), "q"_a, "p"_a, "v"_a, "omega"_a, "hold_view"_a, "idx"_a.noconvert(), "center_body_m"_a,
     "radius_m"_a, "min_duration_ns"_a, "max_disp_px"_a, "min_piece_ns"_a);
  t.def("pin_intervals", [](const I1& t_ns, const F4& q, const F3& p, const F1& depth_m, const In<double, 3>& center,
                            double radius, double limit_px_s, int64_t lag_ns, int64_t min_ns, double focal_px,
                            int64_t bridge_ns) {
    sc::blas::OneThread one;
    const auto tv = ivec(t_ns);
    const auto qv = quats(q);
    const auto pv = vecs(p);
    const auto zv = vec(depth_m);
    const st::PinSamples s{tv, qv, pv, zv, vec3(center), radius};
    const st::PinResult r = st::pin_intervals(s, {limit_px_s, lag_ns, min_ns, focal_px, bridge_ns});
    nb::dict d;
    d["intervals_ns"] = pairs(r.intervals_ns);
    d["whole"] = r.whole;
    d["pairs"] = r.pairs;
    d["speed_p99"] = r.speed_p99;
    d["speed_max"] = r.speed_max;
    return d;
  }, "t_ns"_a.noconvert(), "q"_a, "p"_a, "depth_m"_a, "center_body_m"_a, "radius_m"_a, "limit_px_s"_a, "lag_ns"_a,
     "min_ns"_a, "focal_px"_a, "bridge_ns"_a = 0);
  t.def("hold_groups", [](int64_t t0, int64_t h, int order, int64_t n, std::vector<std::pair<int64_t, int64_t>> iv) {
    const auto [groups, still] = st::hold_groups(st::KnotGrid(t0, h, order, n), std::move(iv));
    return nb::make_tuple(pairs(groups), pairs(still));
  });
  t.def("tie_map", [](int64_t n, std::vector<std::pair<int64_t, int64_t>> groups) {
    const st::TieMap tie(n, std::move(groups));
    return nb::make_tuple(npi(tie.free_index), sparse_tuple(tie.matrix(1)), sparse_tuple(tie.matrix(3)));
  });
  // Banded algebra.
  t.def("band_factor", [](const F1& data, const I1& indices, const I1& indptr, std::pair<int64_t, int64_t> shape,
                          int64_t u) {
    sc::blas::OneThread one;
    const st::BandedSPD c(csr_in(data, indices, indptr, shape), u);
    std::vector<double> f = c.factor();
    return to_numpy(std::move(f), {static_cast<size_t>(c.u() + 1), static_cast<size_t>(c.n())});
  }, "data"_a, "indices"_a.noconvert(), "indptr"_a.noconvert(), "shape"_a, "u"_a);
  t.def("selected_inverse_band", [](const In<double, -1, -1, -1>& factors) {
    const size_t lc = factors.shape(0), ub = factors.shape(1), n = factors.shape(2);
    if (ub == 0) throw std::invalid_argument("selected_inverse_band: factors need u + 1 >= 1 rows");
    std::vector<std::vector<double>> fs(lc);
    std::vector<const std::vector<double>*> ptrs;
    for (size_t l = 0; l < lc; ++l) {
      fs[l].assign(factors.data() + l * ub * n, factors.data() + (l + 1) * ub * n);
      ptrs.push_back(&fs[l]);
    }
    std::vector<double> s = st::selected_inverse_band(ptrs, static_cast<int64_t>(ub) - 1, static_cast<int64_t>(n));
    s.resize(lc * ub * n);   // empty when lc or n is 0, as the Python's (0, u + 1, n) result
    return to_numpy(std::move(s), {lc, ub, n});
  });
  t.def("trace_of_product", [](const In<double, -1, -1, -1>& s, const F1& data, const I1& indices, const I1& indptr,
                               std::pair<int64_t, int64_t> shape) {
    const size_t lc = s.shape(0), ub = s.shape(1), n = s.shape(2);
    std::vector<double> sv(s.data(), s.data() + lc * ub * n);
    if (static_cast<size_t>(shape.first) != n) throw std::invalid_argument("trace_of_product: shapes differ");
    return np1(st::trace_of_product(sv, static_cast<int64_t>(lc),
                                    st::to_upper_band(csr_in(data, indices, indptr, shape), static_cast<int64_t>(ub) - 1)));
  }, "s_band"_a, "data"_a, "indices"_a.noconvert(), "indptr"_a.noconvert(), "shape"_a);
  // The fitter.
  t.def("fit_smoothing", &fit_smoothing, "t_ns"_a.noconvert(), "q"_a, "p"_a, "options"_a, "center_body_m"_a,
        "hold_view"_a = nb::none(),
        "fit_smoothing(samples, SmoothingOptions(**options, center_body_m, hold_view)) in C++ for samples as\n"
        "PoseSamples holds them (q already normalized: PoseSamples(t, q, p).q, or pose_samples_q): a dict of the\n"
        "trajectory (order, t0_ns, h_ns, rot, pos, holds, span_ns), keep, weights and diagnostics (floats as float).\n"
        "options: the scalar SmoothingOptions fields; hold_view: dict(focal_px, corner_r2, depth_m, cameras_body_m)\n"
        "or None.");
  t.def("pose_samples_q", [](const F4& q) { return npq(st::pose_samples_q(quats(q))); },
        "PoseSamples' normalization of q (n, 4).");
  // S4: the adapter's geometry, k-d tree, hold views and Python formatting.
  t.def("geo_transform_mesh", [](nb::handle mesh, nb::handle tr) {
    sc::blas::OneThread one;
    return mesh_out(geo::transform_mesh(mesh_in(mesh), m4_in(tr)));
  });
  t.def("geo_merge_meshes", [](nb::list parts) {
    std::vector<geo::Mesh> meshes;
    std::vector<int> counts;
    for (nb::handle h : parts) {
      const nb::tuple p = nb::cast<nb::tuple>(h);
      meshes.push_back(mesh_in(p[0]));
      counts.push_back(nb::cast<int>(p[1]));
    }
    std::vector<std::pair<const geo::Mesh*, int>> refs;
    for (size_t i = 0; i < meshes.size(); ++i) refs.emplace_back(&meshes[i], counts[i]);
    return mesh_out(geo::merge_meshes(refs));
  });
  t.def("geo_icosphere", [](int level, double radius) { return mesh_out(geo::icosphere(level, radius)); });
  t.def("geo_area_centroid", [](nb::handle mesh) {
    const V c = geo::area_centroid(mesh_in(mesh));
    return np1({c.x, c.y, c.z});
  });
  t.def("geo_bounding_sphere", [](nb::handle mesh) {
    const auto [c, r] = geo::bounding_sphere(mesh_in(mesh));
    return nb::make_tuple(np1({c.x, c.y, c.z}), r);
  });
  t.def("geo_triangle_areas", [](nb::handle mesh) { return np1(geo::triangle_areas(mesh_in(mesh))); });
  t.def("geo_dominant_normal", [](nb::handle mesh) {
    sc::blas::OneThread one;
    const V n = geo::dominant_normal(mesh_in(mesh));
    return np1({n.x, n.y, n.z});
  });
  t.def("geo_kabsch", [](const F3& a, const F3& b) {
    sc::blas::OneThread one;
    same_rows("geo_kabsch", a.shape(0), {b.shape(0)});
    const geo::Kabsch k = geo::kabsch(vecs(a), vecs(b));
    return nb::make_tuple(npm({k.r}), np1({k.t.x, k.t.y, k.t.z}), np1(k.residuals));
  });
  t.def("geo_invert_transform", [](nb::handle tr) { return m4_out(geo::invert_transform(m4_in(tr))); });
  t.def("geo_pose_to_transform", [](const F4& q, const F3& p) {
    same_rows("geo_pose_to_transform", q.shape(0), {p.shape(0)});
    const auto qs = quats(q);
    const auto ps = vecs(p);
    std::vector<double> out;
    for (size_t i = 0; i < qs.size(); ++i) {
      const geo::M4 m = geo::pose_to_transform(qs[i], ps[i]);
      out.insert(out.end(), m.begin(), m.end());
    }
    return to_numpy(std::move(out), {qs.size(), 4, 4});
  });
  t.def("kdtree_nearest", [](const F3& points, const F3& queries) {
    const auto pv = vecs(points);
    std::vector<double> flat;
    for (const V& p : pv) flat.insert(flat.end(), {p.x, p.y, p.z});
    const sc::KdTree tree(std::move(flat));
    const auto qv = vecs(queries);
    std::vector<double> out(qv.size());
    for (size_t i = 0; i < qv.size(); ++i) out[i] = tree.nearest(qv[i]);
    return np1(std::move(out));
  });
  t.def("radtan_field", [](double fx, double fy, double cx, double cy, const In<double, 5>& dist, int64_t w, int64_t h) {
    sc::blas::OneThread one;
    ev::CameraInput c;
    c.fx = fx, c.fy = fy, c.cx = cx, c.cy = cy, c.width = w, c.height = h;
    for (size_t i = 0; i < 5; ++i) c.dist[i] = dist(i);
    const auto [f, r2] = ev::radtan_field(c);
    return nb::make_tuple(f, r2);
  });
  t.def("hold_views", [](nb::list cameras, const In<double, -1, 4, 4>& t_w_rig, nb::list bodies) {
    sc::blas::OneThread one;
    std::vector<ev::CameraInput> cams;
    for (nb::handle h : cameras) {
      const nb::dict c = nb::cast<nb::dict>(h);
      ev::CameraInput ci;
      ci.fx = nb::cast<double>(c["fx"]), ci.fy = nb::cast<double>(c["fy"]);
      ci.cx = nb::cast<double>(c["cx"]), ci.cy = nb::cast<double>(c["cy"]);
      const auto dist = nb::cast<In<double, 5>>(c["dist"]);
      for (size_t i = 0; i < 5; ++i) ci.dist[i] = dist(i);
      ci.width = nb::cast<int64_t>(c["width"]), ci.height = nb::cast<int64_t>(c["height"]);
      ci.T_rig_cam = m4_in(c["T_rig_cam"]);
      cams.push_back(std::move(ci));
    }
    std::vector<geo::M4> tw(t_w_rig.shape(0));
    for (size_t n = 0; n < tw.size(); ++n) {
      for (size_t i = 0; i < 16; ++i) tw[n][i] = t_w_rig.data()[16 * n + i];
    }
    std::vector<geo::Mesh> meshes;
    std::vector<std::vector<Q>> qs;
    std::vector<std::vector<V>> ps;
    std::vector<std::vector<int64_t>> rows;
    for (nb::handle h : bodies) {
      const nb::tuple b = nb::cast<nb::tuple>(h);
      meshes.push_back(mesh_in(b[0]));
      qs.push_back(quats(nb::cast<F4>(b[1])));
      ps.push_back(vecs(nb::cast<F3>(b[2])));
      rows.push_back(ivec(exact<I1>(b[3], "rows")));
      same_rows("hold_views", qs.back().size(), {ps.back().size()});
    }
    std::vector<ev::HoldBody> hb;
    for (size_t i = 0; i < meshes.size(); ++i) hb.push_back({&meshes[i], &qs[i], &ps[i], &rows[i]});
    const auto [rig, views] = ev::hold_views(cams, tw, hb);
    auto view_dict = [](const st::HoldView& v) {
      nb::dict d;
      d["focal_px"] = np1(v.focal_px);
      d["corner_r2"] = np1(v.corner_r2);
      d["depth_m"] = np1(v.depth_m);
      d["cameras_body_m"] = v.has_cameras ? nb::cast(npv(v.cameras_body)) : nb::none();
      return d;
    };
    nb::list vl;
    for (const auto& v : views) vl.append(view_dict(v));
    return nb::make_tuple(view_dict(rig), vl);
  });
  t.def("tabletop_static", [](const F4& q, const F3& p, nb::handle keep, const In<double, 3>& centre, double radius,
                              double max_disp_m, double fraction) {
    sc::blas::OneThread one;
    same_rows("tabletop_static", q.shape(0), {p.shape(0)});
    const auto k = exact<nb::ndarray<const bool, nb::shape<-1>, nb::device::cpu>>(keep, "keep");
    if (k.shape(0) != q.shape(0)) throw std::invalid_argument("tabletop_static: arrays differ in length");
    std::vector<uint8_t> kv(k.shape(0));
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = k(i) ? 1 : 0;
    const ev::TabletopStatic ts =
        ev::tabletop_static(quats(q), vecs(p), kv, vec3(centre), radius, max_disp_m, fraction);
    nb::dict info;
    info["inlier_fraction"] = ts.inlier_fraction;
    info["max_displacement_m"] = ts.max_displacement_m;
    return nb::make_tuple(ts.pose ? nb::cast(m4_out(*ts.pose)) : nb::none(), info);
  });
  t.def("light_anchor", [](nb::list bodies, nb::list trajectories, const F3& cam_positions, int64_t t_ns) {
    sc::blas::OneThread one;
    std::vector<ev::Body> bs;
    for (nb::handle h : bodies) {
      const nb::dict d = nb::cast<nb::dict>(h);
      ev::Body b;
      b.id = nb::cast<int64_t>(d["id"]);
      b.name = nb::cast<std::string>(d["name"]);
      b.mesh = mesh_in(d["mesh"]);
      if (!d["trajectory"].is_none()) b.trajectory = nb::cast<int64_t>(d["trajectory"]);
      b.T_world_body = m4_in(d["T_world_body"]);
      bs.push_back(std::move(b));
    }
    std::vector<st::FitResult> fits;
    for (nb::handle h : trajectories) {
      const nb::dict d = nb::cast<nb::dict>(h);
      st::FitResult f;
      f.order = nb::cast<int>(d["order"]);
      f.t0_ns = nb::cast<int64_t>(d["t0_ns"]);
      f.h_ns = nb::cast<int64_t>(d["h_ns"]);
      f.rot = quats(nb::cast<F4>(d["rot"]));
      f.pos = vecs(nb::cast<F3>(d["pos"]));
      fits.push_back(std::move(f));
    }
    std::vector<const st::FitResult*> ptrs;
    for (const st::FitResult& f : fits) ptrs.push_back(&f);
    const ev::Anchor a = ev::light_anchor(bs, ptrs, vecs(cam_positions), t_ns);
    return nb::make_tuple(np1({a.up.x, a.up.y, a.up.z}), np1({a.centre.x, a.centre.y, a.centre.z}), a.source);
  });
  t.def("clock_offset", [](const std::string& camera, const std::string& category, const std::string& name,
                           nb::handle g1, std::optional<double> raw_line3_s, const std::string& reference) {
    ev::CameraInput c;
    c.camera = camera, c.category = category, c.name = name, c.directory = "<" + name + ">";
    if (!g1.is_none()) {
      const nb::tuple g = nb::cast<nb::tuple>(g1);
      c.g1 = true;
      c.g1_delta_ms = nb::cast<double>(g[0]);
      if (!g[1].is_none()) c.g1_se_ms = nb::cast<double>(g[1]);
    }
    c.raw_line3_s = raw_line3_s;
    const ev::ClockOffset k = ev::clock_offset(c, reference);
    return tree_to_py(k.meta);
  }, "camera"_a, "category"_a, "name"_a, "g1"_a.none(), "raw_line3_s"_a.none(), "reference"_a = "npz");
  t.def("py_repr", &sc::pyfmt::repr);
  t.def("py_hex", &sc::pyfmt::hex);
  t.def("py_fixed", &sc::pyfmt::fixed);
  t.def("py_signed_fixed", &sc::pyfmt::signed_fixed);
  t.def("parse_vsk", [](const std::string& text, const std::string& path) {
    std::vector<V> markers;
    std::vector<std::string> names;
    double radius = 0;
    ev::parse_vsk(text, path, markers, names, radius);
    return nb::make_tuple(npv(markers), names, radius);
  }, "text"_a, "path"_a, "dataset._read_vsk on a vicon.vsk's text: (markers (n, 3) mm, names, median radius).");
  t.def("read_png", [](const std::string& path) {
    sc::assets::Image im = sc::assets::read_png_rgb(path);
    const size_t h = static_cast<size_t>(im.height), w = static_cast<size_t>(im.width);
    return to_numpy(std::move(im.rgb), {h, w, 3});
  }, "path"_a, "The RGB pixels of an 8-bit PNG (what PIL's Image.open(path).convert('RGB') gives).");
  t.def("py_percent", &sc::pyfmt::percent);
}

}  // namespace

// The Blender exchange's statics (blender_bindings.cpp).
void bind_blender(nb::module_& m);
// --- the Blender motion session's clock (docs/design/phase3/blender.md §4.2) ------------------------------------

namespace {

sc::Tree py_to_tree(nb::handle o) {
  if (o.is_none()) return sc::Tree::null();
  if (nb::isinstance<nb::bool_>(o)) return sc::Tree::boolean(nb::cast<bool>(o));
  if (nb::isinstance<nb::int_>(o)) return sc::Tree::integer(nb::cast<int64_t>(o));
  if (nb::isinstance<nb::float_>(o)) return sc::Tree::real(nb::cast<double>(o));
  if (nb::isinstance<nb::str>(o)) return sc::Tree::string(nb::cast<std::string>(o));
  if (nb::isinstance<nb::dict>(o)) {
    sc::Tree t = sc::Tree::dict();
    for (const auto& [k, v] : nb::borrow<nb::dict>(o)) t[nb::cast<std::string>(k)] = py_to_tree(v);
    return t;
  }
  if (nb::isinstance<nb::list>(o) || nb::isinstance<nb::tuple>(o)) {
    sc::Tree t = sc::Tree::list();
    for (nb::handle v : o) t.push_back(py_to_tree(v));
    return t;
  }
  throw std::invalid_argument("timeline record: unsupported value type");
}

// A frame value as the float32 it must be (to_ns and reported take Blender's float32 frames).
float frame_f32(double c) {
  const auto f = static_cast<float>(c);
  if (!(static_cast<double>(f) == c)) throw std::invalid_argument("TimeMap: the frame value is not a float32");
  return f;
}

void bind_motion(nb::module_& m) {
  namespace mo = balla::scene::motion;
  nb::class_<mo::TimeMap>(m, "TimeMap",
                          "The Blender motion session's clock (docs/design/phase3/blender.md §4.2): exact int64 ns\n"
                          "of the float32 frames Blender evaluates; the span's frame_start is t = 0.")
      .def("__init__",
           [](mo::TimeMap* self, int32_t fps, uint32_t fps_base_bits, int32_t frame_map_old, int32_t frame_map_new,
              int32_t frame_start, int32_t frame_end, bool motion_blur, double motion_blur_shutter,
              const std::string& motion_blur_position, bool fps_base_snap) {
             mo::TimeSettings s;
             s.fps = fps;
             s.fps_base_bits = fps_base_bits;
             s.frame_map_old = frame_map_old;
             s.frame_map_new = frame_map_new;
             s.frame_start = frame_start;
             s.frame_end = frame_end;
             s.motion_blur = motion_blur;
             s.motion_blur_shutter = frame_f32(motion_blur_shutter);
             s.motion_blur_position = motion_blur_position;
             s.fps_base_snap = fps_base_snap;
             new (self) mo::TimeMap(s);
           },
           nb::kw_only(), "fps"_a, "fps_base_bits"_a = 0x3f800000u, "frame_map_old"_a = 100, "frame_map_new"_a = 100,
           "frame_start"_a, "frame_end"_a, "motion_blur"_a = false, "motion_blur_shutter"_a = 0.5,
           "motion_blur_position"_a = "CENTER", "fps_base_snap"_a = false)
      .def_static("from_record", [](nb::dict d) { return mo::TimeMap::from_record(py_to_tree(d)); }, "record"_a)
      .def("record", [](const mo::TimeMap& t) { return tree_to_py(t.record()); })
      .def_static(
          "realized", [](int32_t F, double s) { return static_cast<double>(mo::TimeMap::realized(F, frame_f32(s))); },
          "F"_a, "s"_a, "f32(f32(F) + s): the frame Blender evaluates for frame_set(F, subframe=s).")
      .def("reported", [](const mo::TimeMap& t, double c) { return static_cast<double>(t.reported(frame_f32(c))); },
           "c"_a, "f32(c * framelen): the frame_current_final Blender reports at a realized frame c.")
      .def("to_ns", [](const mo::TimeMap& t, double c) { return t.to_ns(frame_f32(c)); }, "c"_a,
           "The exact time of a float32 frame c, rounded half to even.")
      .def("frames_to_ns", &mo::TimeMap::frames_to_ns, "num"_a, "den"_a,
           "The exact time of the frame frame_start + num / den, rounded half to even.")
      .def("request",
           [](const mo::TimeMap& t, int64_t t_ns) {
             int32_t F = 0;
             float s = 0.0f;
             t.request(t_ns, &F, &s);
             return std::make_pair(F, static_cast<double>(s));
           },
           "t_ns"_a, "(F, s) of the float32 frame nearest a target time (s in {0} or [2^-30, 1)).")
      .def("resolution_ns", &mo::TimeMap::resolution_ns, "F"_a, "q(F) = 1/2 ulp32(max(|F|, 1)) P in ns.")
      .def("t_end_ns", &mo::TimeMap::t_end_ns)
      .def("period_ns", &mo::TimeMap::period_ns)
      .def_prop_ro("fps", &mo::TimeMap::fps)
      .def_prop_ro("base_num", &mo::TimeMap::base_num)
      .def_prop_ro("base_den", &mo::TimeMap::base_den)
      .def_prop_ro("frame_ref", &mo::TimeMap::frame_ref)
      .def_prop_ro("t_ref_ns", &mo::TimeMap::t_ref_ns)
      .def_prop_ro("framelen_bits", &mo::TimeMap::framelen_bits);
}

}  // namespace

// LinAlgError, TrajectoryError and ZeroDivisionError as numpy.linalg.LinAlgError, ballasim.traj.fit.TrajectoryError and
// ZeroDivisionError.
static void translate(const std::exception_ptr& e, void*) {
  try {
    std::rethrow_exception(e);
  } catch (const sc::LinAlgError& x) {
    const nb::object type = nb::module_::import_("numpy.linalg").attr("LinAlgError");
    PyErr_SetString(type.ptr(), x.what());
  } catch (const sc::TrajectoryError& x) {
    const nb::object type = nb::module_::import_("ballasim.traj.fit").attr("TrajectoryError");
    PyErr_SetString(type.ptr(), x.what());
  } catch (const sc::ZeroDivisionError& x) {
    PyErr_SetString(PyExc_ZeroDivisionError, x.what());
  } catch (const sc::AssetError& x) {
    const nb::object type = nb::module_::import_("ballasim.scene.assets.geometry").attr("AssetError");
    PyErr_SetString(type.ptr(), x.what());
  } catch (const ev::CatalogueMissing& x) {
    PyErr_SetString(PyExc_FileNotFoundError, x.what());
  }
}

NB_MODULE(_scene, m) {
  nb::register_exception_translator(&translate);
  m.doc() = "BallaSim scene side in C++ (docs/dev/CPP_PORT.md item 5): the EVIMO2 trajectory fitter, bitwise as the "
            "Python";
  m.def("openblas_config", &sc::blas::config,
        "openblas_get_config() of the OpenBLAS this module calls (version and DYNAMIC_ARCH kernel).");
  m.def("matmul3_inline", &sc::blas::matmul3_is_inline,
        "Whether the 3x3 products run inline (bitwise the library's, checked on this machine) or call cblas_dgemm.");
  m.def("build_evimo2", &build_evimo2, "inputs"_a, "options"_a,
        "The EVIMO2 adapter's unlit build (ballasim.evimo2.adapter.build_bundle) from the sequence's typed arrays,\n"
        "bitwise as the Python: bodies (meshes, material layout, trajectory index or static pose), the fits, cameras,\n"
        "re-timing, T_world_c0, T_body_subject, warnings, the relight sidecar and the unlit meta (lights and each\n"
        "body's asset left None for the caller). Runs with the GIL released; the fits on `workers` threads.");
  bind_assets(m);
  bind_blender(m);
  bind_motion(m);
  nb::module_ testing = m.def_submodule("testing");
  bind_testing(testing);
}
