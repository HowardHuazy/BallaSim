// ballasim._imu: the IMU simulation (docs/dev/IMU.md; docs/design/phase3/outputs.md §3.3). Always built with the
// Python bindings, since it needs no compressor; build_info()['imu'] reports it. Refusals (imu::Refused) raise
// ballasim.export.errors.ExportError. A0 bound the formats; A1 the parameters' defaults, the truth (imu_truth) and
// the simulator; A2 the device model, the sample clock and the stamps (camera_clock, sensor_clock_of,
// reported_time); A6 the white-noise variance of the ROS Imu covariances (white_variance); A11 the piecewise rig
// (trajectory `pieces`); A3 the parameter rows' overlay of `ballasim imu` (resolve_imu).
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings/config_value.h"
#include "bindings/export_errors.h"
#include "bindings/imu_values.h"
#include "bindings/ndarray_util.h"
#include "bindings/spline_batch.h"
#include "config/resolve.h"
#include "imu/format.h"
#include "imu/params.h"
#include "imu/simulator.h"
#include "imu/truth.h"

namespace nb = nanobind;
using namespace balla;
using bindings::ExactIn;
using bindings::In;
using bindings::to_numpy;

namespace {

// A rig trajectory with the spline nodes it owns: the view points into `nodes` (and `pieces`), whose buffers a move
// keeps. A piecewise rig (pieces, joins) keeps every piece's nodes one after the other at the rows' offsets.
struct Rig {
  std::vector<traj::SplineNode<double>> nodes;
  std::vector<render::PieceView> pieces;  // empty: one spline
  std::vector<int8_t> joins;              // one per piece (C1 pieces[p].join; piece 0: 0); empty: one spline
  render::TrajectoryView view{};
};

nb::object item(const nb::dict& d, const char* key, const char* what) {
  if (!d.contains(key)) throw imu::Refused(std::string(what) + ": missing '" + key + "'");
  return d[key];
}

int64_t integer(nb::handle x, const std::string& what) {
  int64_t i = 0;
  if (bindings::is_bool(x) || PyFloat_Check(x.ptr()) != 0 || !nb::try_cast<int64_t>(x, i)) {
    throw imu::Refused(what + ": expected an integer");
  }
  return i;
}

// The pieces of a piecewise rig (P3-E2, C1 Trajectory.pieces: int64[P, 6] = begin_ns, t0_ns, h_ns, ctrl_begin,
// ctrl_end, join; P >= 2): each piece's nodes from its own control rows (its first node has d = dp = 0, as
// _core.spline_eval_pieces_f64 and SceneData build them), the views and the joins. The table's rules are bundle.py's
// (consecutive row ranges that cover rot and pos, each holding at least K rows; h in [1, 2^31); joins in {-1, 0, 1},
// piece 0's 0); imu::pieces_error checks the rest once the span is known.
void parse_pieces(Rig& rig, nb::handle table_obj, const In<double, -1, 4>& rot, const In<double, -1, 3>& pos,
                  int order, int64_t t0, int64_t h_ns) {
  using bindings::ExactIn;
  ExactIn<int64_t, -1, 6> table;
  try {
    table = nb::cast<ExactIn<int64_t, -1, 6>>(table_obj, /*convert=*/false);
  } catch (const nb::cast_error&) {
    throw imu::Refused("trajectory.pieces: expected an int64 array of shape (P, 6)");
  }
  const size_t num = table.shape(0), rows = rot.shape(0);
  if (num < 2) throw imu::Refused("trajectory.pieces: needs at least 2 pieces (None for one), has " + std::to_string(num));
  if (pos.shape(0) != rows) throw imu::Refused("trajectory: rot and pos need the same number of rows");
  std::vector<Quat<double>> q(rows);
  std::vector<Vec3<double>> p(rows);
  for (size_t c = 0; c < rows; ++c) {
    q[c] = {rot(c, 0), rot(c, 1), rot(c, 2), rot(c, 3)};
    p[c] = {pos(c, 0), pos(c, 1), pos(c, 2)};
    const double qq = q[c].w * q[c].w + q[c].x * q[c].x + q[c].y * q[c].y + q[c].z * q[c].z;  // as make_spline_input
    if (!(std::isfinite(qq) && qq > 0 && std::isfinite(p[c].x) && std::isfinite(p[c].y) && std::isfinite(p[c].z))) {
      throw imu::Refused("trajectory: control " + std::to_string(c) + ": need finite p and a finite, non-zero q");
    }
  }
  int64_t next_row = 0;
  rig.nodes.reserve(rows);
  std::vector<std::pair<int64_t, int64_t>> row_ranges;
  for (size_t k = 0; k < num; ++k) {
    const std::string at = "trajectory.pieces[" + std::to_string(k) + "]";
    const int64_t begin = table(k, 0), t0_k = table(k, 1), h = table(k, 2), c0 = table(k, 3), c1 = table(k, 4),
                  join = table(k, 5);
    if (!(h >= 1 && h <= INT32_MAX)) throw imu::Refused(at + ": h_ns " + std::to_string(h) + " outside [1, 2^31)");
    if (c0 != next_row || c1 > static_cast<int64_t>(rows) || c1 < c0 + order) {
      throw imu::Refused(at + ": control rows [" + std::to_string(c0) + ", " + std::to_string(c1) + ") must start at row " +
                         std::to_string(next_row) + ", lie within the " + std::to_string(rows) + " rows and hold at least K = " +
                         std::to_string(order));
    }
    if (join < imu::kJoinStep || join > imu::kJoinC1 || (k == 0 && join != 0)) {
      throw imu::Refused(at + ": join " + std::to_string(join) + " must be -1, 0 or 1 (piece 0: 0)");
    }
    next_row = c1;
    const auto first = static_cast<size_t>(c0), count = static_cast<size_t>(c1 - c0);
    const auto part = traj::build_spline_nodes(std::span<const Quat<double>>(q).subspan(first, count),
                                               std::span<const Vec3<double>>(p).subspan(first, count), {0.0, 0.0, 0.0});
    rig.nodes.insert(rig.nodes.end(), part.begin(), part.end());
    row_ranges.emplace_back(c0, c1);
    rig.joins.push_back(static_cast<int8_t>(join));
    rig.pieces.push_back({begin, t0_k, {nullptr, static_cast<int32_t>(count) - order + 1, static_cast<int32_t>(h), 1e9 / static_cast<double>(h)}});
  }
  if (next_row != static_cast<int64_t>(rows)) {
    throw imu::Refused("trajectory.pieces: the pieces' control rows cover " + std::to_string(next_row) + " of the " +
                       std::to_string(rows) + " rows");
  }
  for (size_t k = 0; k < num; ++k) rig.pieces[k].spline.nodes = rig.nodes.data() + row_ranges[k].first;
  if (rig.pieces[0].t0_ns != t0 || rig.pieces[0].spline.h_ns != h_ns) {
    throw imu::Refused("trajectory: t0_ns and h_ns must equal piece 0's");
  }
  rig.view.spline = rig.pieces[0].spline;
  rig.view.pieces = rig.pieces.data();
  rig.view.num_pieces = static_cast<int32_t>(num);
}

// bundle.py's Trajectory as a dict: K, t0_ns, h_ns, rot f64[N,4] (w, x, y, z), pos f64[N,3] (relative to
// origin_world), span_ns (begin, end); other keys (holds) are not read. `pieces` (int64[P, 6], P >= 2) makes it a
// piecewise rig (P3-E2; A11 evaluates its joins, §4.3.4); None or absent is one spline.
Rig parse_trajectory(nb::handle h) {
  if (!nb::isinstance<nb::dict>(h)) {
    throw imu::Refused("trajectory: expected a dict with K, t0_ns, h_ns, rot, pos and span_ns");
  }
  const auto d = nb::borrow<nb::dict>(h);
  const bool piecewise = d.contains("pieces") && !d["pieces"].is_none();
  const int64_t order = integer(item(d, "K", "trajectory"), "trajectory.K");
  // Checked before the narrowing to int, which would wrap 2^32 + 6 to 6.
  if (order != 4 && order != 6) throw imu::Refused("trajectory.K: spline order must be 4 or 6");
  const int64_t t0 = integer(item(d, "t0_ns", "trajectory"), "trajectory.t0_ns");
  const int64_t h_ns = integer(item(d, "h_ns", "trajectory"), "trajectory.h_ns");
  Rig rig;
  try {
    const auto rot = nb::cast<In<double, -1, 4>>(item(d, "rot", "trajectory"));
    const auto pos = nb::cast<In<double, -1, 3>>(item(d, "pos", "trajectory"));
    if (piecewise) {
      if (!(h_ns >= 1 && h_ns <= INT32_MAX)) throw std::invalid_argument("h_ns must be in [1, 2^31)");
      rig.view.t0_ns = t0;
      rig.view.order = static_cast<int32_t>(order);
      parse_pieces(rig, d["pieces"], rot, pos, static_cast<int>(order), t0, h_ns);
    } else {
      bindings::SplineInput s = bindings::make_spline_input(static_cast<int>(order), t0, h_ns, rot, pos,
                                                            {0.0, 0.0, 0.0}, /*finite_controls=*/true);
      rig.nodes = std::move(s.nodes);
      rig.view.spline = {rig.nodes.data(), s.num_segments, s.h_ns, 1e9 / s.h_ns};
      rig.view.t0_ns = t0;
      rig.view.order = s.order;
    }
  } catch (const std::invalid_argument& e) {
    throw imu::Refused(std::string("trajectory: ") + e.what());
  } catch (const nb::cast_error&) {
    throw imu::Refused("trajectory: rot and pos must be float64 arrays of shape (N, 4) and (N, 3)");
  }
  // A list, a tuple or a 1-D array of two integers; anything else (an int, a dict, bytes, a 0-d array) is a Refused,
  // not the TypeError, KeyError or IndexError that len() and [] would raise.
  const nb::object span = item(d, "span_ns", "trajectory");
  PyObject* const sp = span.ptr();
  const bool bytes = PyBytes_Check(sp) != 0 || PyByteArray_Check(sp) != 0 || PyMemoryView_Check(sp) != 0;
  const bool sequence = PyList_Check(sp) != 0 || PyTuple_Check(sp) != 0 || (!bytes && nb::ndarray_check(span));
  nb::object first, last;
  try {
    if (!sequence || nb::len(span) != 2) throw imu::Refused("trajectory.span_ns: expected (begin, end)");
    first = span[0];
    last = span[1];
  } catch (const nb::python_error&) {
    throw imu::Refused("trajectory.span_ns: expected (begin, end)");
  }
  rig.view.span_begin_ns = integer(first, "trajectory.span_ns[0]");
  rig.view.span_end_ns = integer(last, "trajectory.span_ns[1]");
  if (piecewise) {
    // bundle.py: piece 0 begins at the span's begin and the last piece before its end; each piece's support covers its
    // interval, the last one's up to the span's end (imu::pieces_error).
    if (!(rig.view.span_begin_ns < rig.view.span_end_ns)) {
      throw imu::Refused("trajectory.span_ns: needs begin < end");
    }
    if (rig.pieces.front().begin_ns != rig.view.span_begin_ns || rig.pieces.back().begin_ns >= rig.view.span_end_ns) {
      throw imu::Refused("trajectory.pieces: piece 0 must begin at span_ns[0] = " + std::to_string(rig.view.span_begin_ns) +
                         " and the last piece before span_ns[1] = " + std::to_string(rig.view.span_end_ns));
    }
    if (const std::string why = imu::pieces_error(rig.view, rig.joins); !why.empty()) {
      throw imu::Refused("trajectory.pieces: " + why);
    }
    return rig;
  }
  const int64_t support = int64_t{rig.view.spline.num_segments} * rig.view.spline.h_ns;  // < 2^62
  if (t0 > INT64_MAX - support) throw imu::Refused("trajectory: t0_ns + num_segments h_ns overflows int64");
  const int64_t end = t0 + support;
  if (!(t0 <= rig.view.span_begin_ns && rig.view.span_begin_ns < rig.view.span_end_ns && rig.view.span_end_ns <= end)) {
    throw imu::Refused("trajectory.span_ns: needs t0_ns <= begin < end <= t0_ns + num_segments h_ns = " +
                       std::to_string(end));
  }
  return rig;
}

// A row-major 4 x 4 rigid transform's rotation and translation; Refused unless finite with the last row (0, 0, 0, 1).
render::RigidPose pose_of(const In<double, 4, 4>& t, const char* what) {
  render::RigidPose p{};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      if (!std::isfinite(t(i, j))) throw imu::Refused(std::string(what) + " has a non-finite entry");
    }
  }
  if (!(t(3, 0) == 0 && t(3, 1) == 0 && t(3, 2) == 0 && t(3, 3) == 1)) {
    throw imu::Refused(std::string(what) + ": last row is not exactly (0, 0, 0, 1)");
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) p.r.m[i][j] = t(i, j);
  }
  p.p = {t(0, 3), t(1, 3), t(2, 3)};
  return p;
}

imu::Vec3d vec3(const In<double, 3>& v) { return {v(0), v(1), v(2)}; }

// One camera's resolved imu.* values ({"imu.rate_hz": 1000.0, ...}, every key) as ImuParams.
imu::ImuParams params_of(nb::handle values) { return bindings::imu_params_of(values); }

double real(nb::handle x, const std::string& what) {
  double v = 0;
  if (bindings::is_bool(x) || nb::isinstance<nb::str>(x) || !nb::try_cast<double>(x, v) || !std::isfinite(v)) {
    throw imu::Refused(what + ": expected a finite number");
  }
  return v;
}

// {quantum_ns, phase_ns, offset_ns, drift_ppm} (camera_clock's dict), checked by imu::check_clock.
imu::CameraClock clock_of(nb::handle h) {
  if (!nb::isinstance<nb::dict>(h)) throw imu::Refused("clock: expected a dict of quantum_ns, phase_ns, offset_ns, drift_ppm");
  const auto d = nb::borrow<nb::dict>(h);
  imu::CameraClock c;
  c.quantum_ns = integer(item(d, "quantum_ns", "clock"), "clock.quantum_ns");
  c.phase_ns = integer(item(d, "phase_ns", "clock"), "clock.phase_ns");
  c.offset_ns = integer(item(d, "offset_ns", "clock"), "clock.offset_ns");
  c.drift_ppm = real(item(d, "drift_ppm", "clock"), "clock.drift_ppm");
  imu::check_clock(c, "clock");
  return c;
}

nb::dict clock_dict(const imu::CameraClock& c) {
  nb::dict d;
  d["quantum_ns"] = c.quantum_ns;
  d["phase_ns"] = c.phase_ns;
  d["offset_ns"] = c.offset_ns;
  d["drift_ppm"] = c.drift_ppm;
  return d;
}

// The block's arrays; counts_gyro and counts_accel when that sensor has an lsb (also in an empty block).
nb::dict block_dict(imu::ImuBlock&& b, bool counts_gyro, bool counts_accel) {
  const size_t n = b.size();
  nb::dict d;
  d["t_true_ns"] = to_numpy(std::move(b.t_true_ns), {n});
  d["t_ns"] = to_numpy(std::move(b.t_ns), {n});
  d["gyro"] = to_numpy(std::move(b.gyro), {n, 3});
  d["accel"] = to_numpy(std::move(b.accel), {n, 3});
  if (counts_gyro) d["counts_gyro"] = to_numpy(std::move(b.counts_gyro), {n, 3});
  if (counts_accel) d["counts_accel"] = to_numpy(std::move(b.counts_accel), {n, 3});
  d["flags"] = to_numpy(std::move(b.flags), {n});
  d["omega_true"] = to_numpy(std::move(b.omega_true), {n, 3});
  d["f_true"] = to_numpy(std::move(b.f_true), {n, 3});
  d["bias_gyro"] = to_numpy(std::move(b.bias_gyro), {n, 3});
  d["bias_accel"] = to_numpy(std::move(b.bias_accel), {n, 3});
  d["q_W_I"] = to_numpy(std::move(b.q_W_I), {n, 4});
  d["p_W_I"] = to_numpy(std::move(b.p_W_I), {n, 3});
  return d;
}

// The simulator with the nodes its trajectory view points into.
struct PySimulator {
  PySimulator(Rig&& r, std::unique_ptr<imu::ImuSimulator>&& s) : rig(std::move(r)), sim(std::move(s)) {}

  // Runs f(sim) without the GIL, holding the simulator's lock: calls from several Python threads are serialized.
  // As sensor_module.cpp's locked(): the lock is taken after the GIL is released and released before the GIL is
  // taken again, so no thread holds one while waiting for the other.
  template <class F>
  auto locked(F&& f) {
    nb::gil_scoped_release release;
    const std::lock_guard<std::mutex> lock(mu);
    return f(*sim);
  }

  Rig rig;
  std::unique_ptr<imu::ImuSimulator> sim;
  std::mutex mu;
};

// A dict item as a str, Refused (naming `what`) otherwise.
std::string text(nb::handle h, const std::string& what) {
  if (!nb::isinstance<nb::str>(h)) throw imu::Refused(what + ": expected a string");
  return nb::cast<std::string>(h);
}

// resolve_imu's `recorded`: {"cameras": {camera: {key: {value, layer, status}}}, "digests": {camera: sensor digest}}
// for the run's `cameras`.
config::SectionInputs section_inputs(nb::handle recorded, const std::vector<std::string>& cameras) {
  if (!nb::isinstance<nb::dict>(recorded)) throw imu::Refused("recorded: expected {cameras: {...}, digests: {...}}");
  const auto rec = nb::borrow<nb::dict>(recorded);
  const nb::object rcams = item(rec, "cameras", "recorded"), rdigests = item(rec, "digests", "recorded");
  if (!nb::isinstance<nb::dict>(rcams) || !nb::isinstance<nb::dict>(rdigests)) {
    throw imu::Refused("recorded: cameras and digests must be dicts");
  }
  config::SectionInputs in;
  for (const std::string& name : cameras) {
    config::SectionCamera sc;
    sc.name = name;
    const auto digests = nb::borrow<nb::dict>(rdigests);
    if (!digests.contains(name.c_str())) throw imu::Refused("recorded: no sensor-stage digest of camera " + name);
    const nb::object digest = digests[name.c_str()];
    sc.sensor_digest = text(digest, "recorded.digests." + name);
    const auto cams = nb::borrow<nb::dict>(rcams);
    if (cams.contains(name.c_str())) {
      const nb::object block = cams[name.c_str()];
      if (!nb::isinstance<nb::dict>(block)) throw imu::Refused("recorded.cameras." + name + ": expected a dict");
      for (auto [k, e] : nb::borrow<nb::dict>(block)) {
        const std::string key = text(k, "recorded.cameras." + name + ": key");
        const std::string where = "recorded.cameras." + name + "." + key;
        if (!nb::isinstance<nb::dict>(e)) throw imu::Refused(where + ": expected {value, layer, status}");
        const auto entry = nb::borrow<nb::dict>(e);
        const config::Node v = bindings::node_from_py(item(entry, "value", where.c_str()), where);
        if (v.is_map) throw imu::Refused(where + ": expected a value, not a mapping");
        sc.recorded.emplace_back(key, config::Entry{v.leaf, text(item(entry, "layer", where.c_str()), where + ".layer"),
                                                    text(item(entry, "status", where.c_str()), where + ".status"), ""});
      }
    }
    in.cameras.push_back(std::move(sc));
  }
  return in;
}

nb::dict section_dict(const config::SectionResult& r) {
  nb::dict cams, digests, change;
  for (size_t i = 0; i < r.cameras.size(); ++i) {
    nb::dict entries;
    for (const auto& [key, e] : r.cameras[i].second) {
      nb::dict d;
      d["value"] = bindings::to_py(e.value);
      d["status"] = e.status;
      d["source"] = e.source;
      d["layer"] = e.layer;
      entries[key.c_str()] = d;
    }
    cams[r.cameras[i].first.c_str()] = entries;
    digests[r.cameras[i].first.c_str()] = r.imu_digests[i];
    change[r.cameras[i].first.c_str()] = r.rotation_change[i];
  }
  nb::dict out;
  out["cameras"] = cams;
  out["warnings"] = r.warnings;
  out["digests"] = digests;
  out["rotation_change"] = change;
  return out;
}

}  // namespace

NB_MODULE(_imu, m) {
  m.doc() = "BallaSim IMU simulation (docs/dev/IMU.md): C5 formats, the truth and the simulator; refusals raise "
            "ballasim.export.errors.ExportError";
  bindings::register_export_error<imu::Refused>();
  m.attr("FILE_FORMAT") = std::string(imu::kFileFormat);
  m.attr("RUN_FORMAT") = std::string(imu::kRunFormat);
  m.def("check_format", &imu::check_format, nb::arg("value"), nb::arg("expected") = std::string(imu::kFileFormat),
        "Checks a file's `format` against `expected` (FILE_FORMAT by default): the same name and major version, "
        "with or without a minor; ExportError otherwise.");

  m.def(
      "default_values",
      []() {
        nb::dict d;
        for (const auto& [k, v] : imu::default_values()) d[k.c_str()] = bindings::to_py(v);
        return d;
      },
      "The defaults of the 37 imu.* keys (ImuParams{}): an ideal IMU, off. The schema rows take them in A3.");
  m.def(
      "white_variance",
      [](nb::handle values) {
        const imu::ImuParams p = params_of(values);
        const imu::Check c = imu::validate(p);
        if (!c.errors.empty()) throw imu::Refused(c.errors.front());
        const std::array<double, 2> v = imu::white_variance(p);
        return nb::make_tuple(v[0], v[1]);
      },
      nb::arg("imu"),
      "(gyro, accel): the per-sample variance of each sensor's additive white noise from one camera's resolved "
      "imu.* values, the ROS Imu covariances (outputs.md §4.8): sigma^2 rate_hz in white mode, sigma^2 / (2 tau_c) "
      "in shaped mode; ExportError on values validate() refuses.");

  m.def(
      "resolve_imu",
      [](nb::handle recorded, const std::vector<std::string>& root_profiles, nb::handle camera_profiles,
         nb::handle sets, nb::handle profile_docs, const std::vector<std::string>& cameras) {
        try {
          config::SectionInputs in = section_inputs(recorded, cameras);
          in.profiles = root_profiles;
          if (!camera_profiles.is_none()) {
            if (!nb::isinstance<nb::dict>(camera_profiles)) {
              throw imu::Refused("camera_profiles: expected {camera: [profile ids]}");
            }
            for (auto [cam, ids] : nb::borrow<nb::dict>(camera_profiles)) {
              in.camera_profiles.emplace_back(text(cam, "camera_profiles: camera"),
                                              nb::cast<std::vector<std::string>>(ids));
            }
          }
          if (!sets.is_none()) in.sets = bindings::node_from_py(sets, "sets");
          if (!profile_docs.is_none()) {
            if (!nb::isinstance<nb::dict>(profile_docs)) throw imu::Refused("profile_docs: expected {id: document}");
            for (auto [id, doc] : nb::borrow<nb::dict>(profile_docs)) {
              const std::string key = text(id, "profile_docs: id");
              in.profile_docs.emplace(key, bindings::node_from_py(doc, "profile " + key));
            }
          }
          return section_dict(config::resolve_section(in));
        } catch (const imu::Refused&) {
          throw;
        } catch (const std::invalid_argument& e) {  // a DiagnosticError (the overlay's refusals) or a value error
          throw imu::Refused(e.what());
        } catch (const nb::cast_error& e) {
          throw imu::Refused(std::string("resolve_imu: ") + e.what());
        }
      },
      nb::arg("recorded").none(), nb::arg("root_profiles"), nb::arg("camera_profiles").none(), nb::arg("sets").none(),
      nb::arg("profile_docs").none(), nb::arg("cameras"),
      "The imu.* keys of a finished run resolved again (outputs.md §4.11): the run's recorded values "
      "(recorded = {cameras: {camera: {key: {value, layer, status}}}, digests: {camera: the recorded sensor-stage "
      "digest}}, for `cameras`, the run's cameras) under the new layers: root_profiles (ids; mount root), "
      "camera_profiles ({camera: [ids]}; mount camera), sets ({full key: value}: cameras.<camera>.imu.* or "
      "cameras.default.imu.*) and profile_docs ({id: the profile file's document}). Returns {cameras: {camera: "
      "{key: {value, status, source, layer}}}, warnings, digests: {camera: imu_digest}, rotation_change: {camera: "
      "the largest change of T_cam_imu's rotation by its re-orthonormalization}}. ExportError for a key outside imu.*, "
      "a mount mismatch, an unknown camera, a type or range error and imu::validate's refusals.");
  m.def(
      "camera_clock",
      [](nb::handle readout_quantum_s, nb::handle clock_offset_ns, nb::handle clock_drift_ppm, nb::handle sensor_seed,
         nb::handle camera_index) {
        const int64_t seed = integer(sensor_seed, "sensor_seed");
        const int64_t camera = integer(camera_index, "camera_index");
        if (seed < 0) throw imu::Refused("sensor_seed: expected an integer >= 0");
        if (camera < 0 || camera > 255) throw imu::Refused("camera_index: expected an integer in [0, 256)");
        return clock_dict(imu::camera_clock(real(readout_quantum_s, "readout_quantum_s"),
                                            integer(clock_offset_ns, "clock_offset_ns"),
                                            real(clock_drift_ppm, "clock_drift_ppm"), static_cast<uint64_t>(seed),
                                            static_cast<uint32_t>(camera)));
      },
      nb::arg("readout_quantum_s"), nb::arg("clock_offset_ns"), nb::arg("clock_drift_ppm"), nb::arg("sensor_seed"),
      nb::arg("camera_index"),
      "The camera's readout counter (outputs.md §4.6) from the recorded sensor.readout_quantum_s, "
      "sensor.clock_offset_ns, sensor.clock_drift_ppm, run.sensor_seed and the bundle camera index, as the sensor "
      "builds it: {quantum_ns, phase_ns, offset_ns, drift_ppm}, phase_ns = Philox(sensor_seed, kReadoutPhase, "
      "camera, 0, 0) mod quantum_ns.");
  m.def(
      "sensor_clock_of", [](nb::handle clock) { return clock_dict(imu::sensor_clock_of(clock_of(clock))); },
      nb::arg("clock"),
      "The clock without its counter (quantum 0, phase 0): the run's sensor clock clock_c, offset and drift only, "
      "which maps true times onto the events' clock (frame times, outputs.md §4.7).");
  m.def(
      "reported_time",
      [](nb::handle clock, nb::handle delay_ns, const bindings::TimeArray& t_true_ns) {
        const imu::CameraClock c = clock_of(clock);
        const int64_t delay = integer(delay_ns, "delay_ns");
        const size_t n = t_true_ns.shape(0);
        std::vector<int64_t> out(n);
        for (size_t i = 0; i < n; ++i) {
          if (!imu::reported_time_checked(c, delay, t_true_ns(i), &out[i])) {
            throw imu::Refused("reported_time: the stamp of t = " + std::to_string(t_true_ns(i)) + " ns overflows int64");
          }
        }
        return to_numpy(std::move(out), {n});
      },
      nb::arg("clock"), nb::arg("delay_ns"), nb::arg("t_true_ns").noconvert(),
      "sensor/readout.h's reported time of int64 true times through `clock` with the delay delay_ns: (t + delay) "
      "floored to the counter's quantum and phase, then + offset + llround(drift 1e-6 t). IMU stamps take "
      "delay = imu.stamp_offset_ns; frame times take sensor_clock_of(clock) and 0. A stamp that overflows int64 is "
      "refused.");

  m.def(
      "imu_truth",
      [](nb::handle trajectory, const In<double, 4, 4>& T_rig_cam, const In<double, 4, 4>& T_cam_imu,
         const In<double, 3>& g_W, const bindings::TimeArray& t_ns) {
        const Rig rig = parse_trajectory(trajectory);
        imu::ImuParams p;
        for (size_t i = 0; i < 16; ++i) p.T_cam_imu[i] = T_cam_imu(i / 4, i % 4);
        const imu::Mount mount = imu::mount_of(pose_of(T_rig_cam, "T_rig_cam"), p);
        const imu::Vec3d g = vec3(g_W);
        const size_t n = t_ns.shape(0);
        std::vector<int64_t> times(n);
        for (size_t i = 0; i < n; ++i) times[i] = t_ns(i);
        std::vector<double> w(3 * n), f(3 * n), q(4 * n), pos(3 * n);
        {
          nb::gil_scoped_release release;
          for (size_t i = 0; i < n; ++i) {
            const imu::Truth tr = imu::truth_at(rig.view, mount, g, imu::where_right(rig.view, times[i], 0.0));
            const double row[13] = {tr.omega_I.x, tr.omega_I.y, tr.omega_I.z, tr.f_I.x,   tr.f_I.y,   tr.f_I.z,  tr.q_W_I.w,
                                    tr.q_W_I.x,   tr.q_W_I.y,   tr.q_W_I.z,   tr.p_W_I.x, tr.p_W_I.y, tr.p_W_I.z};
            std::copy(row, row + 3, w.begin() + 3 * i);
            std::copy(row + 3, row + 6, f.begin() + 3 * i);
            std::copy(row + 6, row + 10, q.begin() + 4 * i);
            std::copy(row + 10, row + 13, pos.begin() + 3 * i);
          }
        }
        nb::dict d;
        d["omega"] = to_numpy(std::move(w), {n, 3});
        d["f"] = to_numpy(std::move(f), {n, 3});
        d["q"] = to_numpy(std::move(q), {n, 4});
        d["p"] = to_numpy(std::move(pos), {n, 3});
        return d;
      },
      nb::arg("trajectory"), nb::arg("T_rig_cam"), nb::arg("T_cam_imu"), nb::arg("g_W"), nb::arg("t_ns").noconvert(),
      "The IMU truth (outputs.md §4.1) at int64 ns times: omega (body-frame rate in IMU axes, rad/s), f (specific "
      "force in IMU axes, m/s^2), q (w, x, y, z) and p of the IMU frame in the world (p relative to origin_world). "
      "trajectory: bundle.py's Trajectory as a dict (K, t0_ns, h_ns, rot, pos, span_ns, and `pieces`, int64[P, 6], "
      "for a piecewise rig: evaluated on the piece serving each time, right-continuous, a time at a join taking the "
      "right piece); T_rig_cam and T_cam_imu: 4 x 4 (T_cam_imu by the imu.T_cam_imu rigidity rule, then re-orthonormalized); "
      "g_W: world gravity (m/s^2).");

  nb::class_<PySimulator>(m, "ImuSimulator",
                          "One camera's IMU (outputs.md §3.3): advance_to(t) returns the samples with t_true in "
                          "[the previous end, t), bitwise independent of the chunking.")
      .def(
          "__init__",
          [](PySimulator* self, nb::handle trajectory, const In<double, 4, 4>& T_rig_cam, nb::handle imu,
             nb::handle clock, const In<double, 3>& g_W, uint32_t camera_index, int64_t t_begin_ns, int64_t t_end_ns,
             int64_t t_offset_us) {
            Rig rig = parse_trajectory(trajectory);
            imu::ImuInput in;
            in.rig = rig.view;
            in.joins = rig.joins;
            in.T_rig_cam = pose_of(T_rig_cam, "T_rig_cam");
            in.clock = clock_of(clock);
            in.g_W = vec3(g_W);
            in.camera_index = camera_index;
            in.t_begin_ns = t_begin_ns;
            in.t_end_ns = t_end_ns;
            in.t_offset_us = t_offset_us;
            auto sim = std::make_unique<imu::ImuSimulator>(params_of(imu), in);
            new (self) PySimulator(std::move(rig), std::move(sim));
          },
          nb::arg("trajectory"), nb::arg("T_rig_cam"), nb::arg("imu"), nb::arg("clock"), nb::arg("g_W"),
          nb::arg("camera_index"), nb::arg("t_begin_ns"), nb::arg("t_end_ns"), nb::arg("t_offset_us"),
          "trajectory: the rig's bundle Trajectory as a dict (as imu_truth; a piecewise rig's joins are the C0 "
          "impulses, C1 cuts and step resets of outputs.md §4.3.4); T_rig_cam: 4 x 4; imu: the camera's resolved imu.* values (every key, as default_values()); clock: the camera's "
          "readout counter (camera_clock); g_W: world gravity (m/s^2); camera_index: the bundle camera (< 256), "
          "which keys the IMU's draws with imu.seed; [t_begin_ns, t_end_ns): the run span (true clock); "
          "t_offset_us: the event file's /t_offset (samples stamped below 1000 t_offset_us are dropped and "
          "counted).")
      .def(
          "advance_to",
          [](PySimulator& s, int64_t t_end_ns) {
            imu::ImuBlock b = s.locked([&](imu::ImuSimulator& sim) { return sim.advance_to(t_end_ns); });
            return block_dict(std::move(b), s.sim->has_counts(0), s.sim->has_counts(1));
          },
          nb::arg("t_end_ns"),
          "The next samples, as a dict of arrays: t_true_ns and t_ns (int64: the instant on the true clock and the "
          "stamp on the camera's sensor clock), gyro and accel (published, (N, 3)), counts_gyro and counts_accel "
          "(int32 (N, 3), each when that sensor's lsb is set), flags (uint8; bits 0-5 rails, bit 6 WARMUP, bit 7 IMPACT), "
          "omega_true and f_true (the truth at the instant), bias_gyro and bias_accel (b0 + b_rw + b_gm), q_W_I "
          "((N, 4), w, x, y, z) and p_W_I.")
      .def(
          "stats",
          [](PySimulator& s) {
            const imu::ImuStats st = s.locked([](imu::ImuSimulator& sim) { return sim.stats(); });  // a copy
            nb::dict d;
            d["dropped_before_t_offset"] = st.dropped_before_t_offset;
            d["warmup_samples"] = st.warmup_samples;
            d["impact_samples"] = st.impact_samples;
            d["step_resets"] = st.step_resets;
            nb::list pre;
            for (int64_t x : st.pre_clip_engaged) pre.append(x);
            d["pre_clip_engaged"] = pre;
            return d;
          },
          "Counters so far: dropped_before_t_offset, warmup_samples, impact_samples, step_resets, pre_clip_engaged "
          "(samples whose filter input was clipped under clip_stage both; gyro x, y, z, accel x, y, z).");
}
