// Spline evaluation and motion bounds in batch form (fp64 reference path and fp32 device path).
#include <nanobind/nanobind.h>

#include <string>
#include <vector>

#include "bindings/core_bindings.h"
#include "bindings/spline_batch.h"
#include "core/traj/bounds.h"
#include "core/traj/pieces.h"
#include "core/traj/spline.h"

namespace nb = nanobind;

namespace balla::bindings {
namespace {

template <int K>
std::vector<traj::Kinematics<double>> eval_f64(const SplineInput& s, const TimeArray& t_ns) {
  const traj::SplineView<double> view{s.nodes.data(), s.num_segments, s.h_ns, 1e9 / s.h_ns};
  std::vector<traj::Kinematics<double>> out(t_ns.shape(0));
  for (size_t i = 0; i < out.size(); ++i) {
    traj::SegmentTime st;
    if (!traj::locate(t_ns(i), s.t0_ns, s.h_ns, s.num_segments, &st)) throw_out_of_span(t_ns(i));
    out[i] = traj::evaluate<K>(view, st);
  }
  return out;
}

template <int K>
std::vector<traj::Kinematics<float>> eval_f32(const SplineInput& s, const TimeArray& t_ns) {
  const std::vector<traj::SplineNode<float>> nodes = traj::convert_nodes<float>(s.nodes);
  const traj::SplineView<float> view{nodes.data(), s.num_segments, s.h_ns,
                                     static_cast<float>(1e9 / s.h_ns)};
  const OffsetBatch batch = make_offsets(s, t_ns);
  std::vector<traj::Kinematics<float>> out(batch.tau_ns.size());
  for (size_t i = 0; i < out.size(); ++i) {
    traj::SegmentTime st;
    if (!traj::locate_offset(batch.base_segment, batch.tau_ns[i], s.h_ns, s.num_segments, &st)) {
      throw_out_of_span(t_ns(i));
    }
    out[i] = traj::evaluate<K>(view, st);
  }
  return out;
}

nb::dict spline_eval_f64(int order, int64_t t0_ns, int64_t h_ns, const In<double, -1, 4>& q,
                         const In<double, -1, 3>& p, const TimeArray& t_ns) {
  const SplineInput s = make_spline_input(order, t0_ns, h_ns, q, p, {0.0, 0.0, 0.0});
  return kinematics_to_dict(order == 4 ? eval_f64<4>(s, t_ns) : eval_f64<6>(s, t_ns));
}

// A piecewise trajectory (C1 Trajectory.pieces): per piece, the nodes of its own control rows, built exactly as
// make_spline_input builds a single spline's (origin 0), and its SplinePiece view of them.
struct PiecesInput {
  std::vector<std::vector<traj::SplineNode<double>>> nodes;
  std::vector<traj::SplinePiece<double>> pieces;
};

// pieces: int64 (P, 6) rows (begin_ns, t0_ns, h_ns, ctrl_begin, ctrl_end, join) as C1 stores them; the join column is
// not read (evaluation needs none). Checks what evaluation relies on: begins strictly increasing, t0_ns <= begin_ns,
// h in [1, 2^31), and control rows that follow each other, cover q and p and give each piece at least `order` rows.
PiecesInput make_pieces_input(int order, const ExactIn<int64_t, -1, 6>& table, const In<double, -1, 4>& q,
                              const In<double, -1, 3>& p) {
  if (order != 4 && order != 6) throw std::invalid_argument("spline order must be 4 or 6");
  const size_t num_pieces = table.shape(0), n = q.shape(0);
  if (num_pieces == 0 || num_pieces > static_cast<size_t>(INT32_MAX)) {
    throw std::invalid_argument("pieces needs between 1 and 2^31 - 1 rows");
  }
  if (p.shape(0) != n) throw std::invalid_argument("q and p need the same number of rows");
  PiecesInput in{std::vector<std::vector<traj::SplineNode<double>>>(num_pieces),
                 std::vector<traj::SplinePiece<double>>(num_pieces)};
  int64_t next_row = 0;
  for (size_t k = 0; k < num_pieces; ++k) {
    const std::string where = "piece " + std::to_string(k) + ": ";
    const int64_t begin = table(k, 0), t0 = table(k, 1), h = table(k, 2), c0 = table(k, 3), c1 = table(k, 4);
    if (k > 0 && begin <= table(k - 1, 0)) throw std::invalid_argument(where + "begin_ns must exceed the last piece's");
    if (t0 > begin) throw std::invalid_argument(where + "t0_ns lies after begin_ns");
    if (h <= 0 || h > INT32_MAX) throw std::invalid_argument(where + "h_ns must be in [1, 2^31)");
    if (c0 != next_row || c1 > static_cast<int64_t>(n) || c1 - c0 < order) {
      throw std::invalid_argument(where + "control rows [" + std::to_string(c0) + ", " + std::to_string(c1) +
                                  ") must start at row " + std::to_string(next_row) + ", lie within the " +
                                  std::to_string(n) + " rows and hold at least `order` rows");
    }
    next_row = c1;
    std::vector<Quat<double>> qs(static_cast<size_t>(c1 - c0));
    std::vector<Vec3<double>> ps(qs.size());
    for (size_t c = 0; c < qs.size(); ++c) {
      const size_t r = static_cast<size_t>(c0) + c;
      qs[c] = {q(r, 0), q(r, 1), q(r, 2), q(r, 3)};
      ps[c] = {p(r, 0), p(r, 1), p(r, 2)};
    }
    in.nodes[k] = traj::build_spline_nodes(qs, ps, {0.0, 0.0, 0.0});
    in.pieces[k] = {begin, t0,
                    {in.nodes[k].data(), static_cast<int32_t>(c1 - c0) - order + 1, static_cast<int32_t>(h),
                     1e9 / static_cast<double>(h)}};
  }
  if (next_row != static_cast<int64_t>(n)) throw std::invalid_argument("the pieces' control rows must cover q and p");
  return in;
}

template <int K>
std::vector<traj::Kinematics<double>> eval_pieces_f64(const PiecesInput& in, const TimeArray& t_ns) {
  const auto num_pieces = static_cast<int32_t>(in.pieces.size());
  std::vector<traj::Kinematics<double>> out(t_ns.shape(0));
  for (size_t i = 0; i < out.size(); ++i) {
    int32_t piece = 0;
    traj::SegmentTime st;
    if (!traj::locate_piece(in.pieces.data(), num_pieces, t_ns(i), &piece, &st)) throw_out_of_span(t_ns(i));
    out[i] = traj::evaluate<K>(in.pieces[static_cast<size_t>(piece)].spline, st);
  }
  return out;
}

nb::dict spline_eval_pieces_f64(int order, const ExactIn<int64_t, -1, 6>& pieces, const In<double, -1, 4>& q,
                                const In<double, -1, 3>& p, const TimeArray& t_ns) {
  const PiecesInput in = make_pieces_input(order, pieces, q, p);
  return kinematics_to_dict(order == 4 ? eval_pieces_f64<4>(in, t_ns) : eval_pieces_f64<6>(in, t_ns));
}

nb::dict spline_eval_f32(int order, int64_t t0_ns, int64_t h_ns, const In<double, -1, 4>& q,
                         const In<double, -1, 3>& p, const In<double, 3>& origin,
                         const TimeArray& t_ns) {
  const SplineInput s = make_spline_input(order, t0_ns, h_ns, q, p, to_vec3(origin));
  return kinematics_to_dict(order == 4 ? eval_f32<4>(s, t_ns) : eval_f32<6>(s, t_ns));
}

template <int K>
nb::dict bounds_f64(const SplineInput& s) {
  const traj::SplineView<double> view{s.nodes.data(), s.num_segments, s.h_ns, 1e9 / s.h_ns};
  const size_t n = static_cast<size_t>(s.num_segments);
  std::vector<double> w(n), wd(n), v(n), a(n);
  for (size_t i = 0; i < n; ++i) {
    const traj::MotionBounds<double> b = traj::segment_bounds<K>(view, static_cast<int32_t>(i));
    w[i] = b.omega;
    wd[i] = b.omega_dot;
    v[i] = b.v;
    a[i] = b.a;
  }
  nb::dict d;
  d["omega"] = to_numpy(std::move(w), {n});
  d["omega_dot"] = to_numpy(std::move(wd), {n});
  d["v"] = to_numpy(std::move(v), {n});
  d["a"] = to_numpy(std::move(a), {n});
  return d;
}

nb::dict spline_segment_bounds(int order, int64_t h_ns, const In<double, -1, 4>& q,
                               const In<double, -1, 3>& p) {
  const SplineInput s = make_spline_input(order, 0, h_ns, q, p, {0.0, 0.0, 0.0}, true);
  return order == 4 ? bounds_f64<4>(s) : bounds_f64<6>(s);
}

}  // namespace

void bind_traj(nb::module_& m) {
  m.def("spline_eval_f64", &spline_eval_f64, nb::arg("order"), nb::arg("t0_ns"), nb::arg("h_ns"),
        nb::arg("q"), nb::arg("p"), nb::arg("t_ns").noconvert(),
        "Evaluate the cumulative SO(3) x R^3 spline (order 4 or 6) in fp64 at int64 ns times\n"
        "(t_ns of any other dtype raises TypeError instead of being truncated).\n"
        "q: (N, 4) control quaternions (w, x, y, z); p: (N, 3) control positions; segment s covers\n"
        "[t0 + s h, t0 + (s + 1) h). Returns a dict of q, p, omega, omega_dot (body frame), v, a.");
  m.def("spline_eval_pieces_f64", &spline_eval_pieces_f64, nb::arg("order"), nb::arg("pieces").noconvert(),
        nb::arg("q"), nb::arg("p"), nb::arg("t_ns").noconvert(),
        "As spline_eval_f64, for a piecewise trajectory (C1 Trajectory.pieces): pieces is int64 (P, 6), rows\n"
        "(begin_ns, t0_ns, h_ns, ctrl_begin, ctrl_end, join); piece k is the spline of the control rows\n"
        "[ctrl_begin, ctrl_end) of q and p with knots t0_ns + s h_ns. A time takes the last piece whose begin_ns\n"
        "is <= it (a time at a breakpoint takes the right piece; one before every begin, piece 0) and raises\n"
        "outside that piece's support.");
  m.def("spline_eval_f32", &spline_eval_f32, nb::arg("order"), nb::arg("t0_ns"), nb::arg("h_ns"),
        nb::arg("q"), nb::arg("p"), nb::arg("origin"), nb::arg("t_ns").noconvert(),
        "As spline_eval_f64, on the fp32 device path: nodes derived in fp64 relative to `origin`\n"
        "and cast, times as int32 offsets from one knot-aligned base. Positions are relative to\n"
        "`origin`. Bitwise equal to ballasim._cuda.spline_eval_f32.");
  m.def("spline_segment_bounds", &spline_segment_bounds, nb::arg("order"), nb::arg("h_ns"),
        nb::arg("q"), nb::arg("p"),
        "Per-segment bounds on |omega|, |omega_dot|, |v| and |a| (M2 Theorem B3), fp64.");
}

}  // namespace balla::bindings
