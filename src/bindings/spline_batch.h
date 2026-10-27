// Spline construction and query batching shared by the _core and _cuda modules.
#pragma once

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "bindings/ndarray_util.h"
#include "core/math/quat.h"
#include "core/math/vec.h"
#include "core/time/time.h"
#include "core/traj/spline.h"
#include "core/traj/spline_build.h"

namespace balla::bindings {

struct SplineInput {
  int order;
  int64_t t0_ns;
  int32_t h_ns;
  int32_t num_segments;
  std::vector<traj::SplineNode<double>> nodes;
};

// finite_controls: refuse non-finite controls and zero quaternions (the bounds need them; evaluation propagates a
// NaN into the poses it reaches, and the trajectory fitter evaluates rotations with NaN placeholder positions).
inline SplineInput make_spline_input(int order, int64_t t0_ns, int64_t h_ns,
                                     const In<double, -1, 4>& q, const In<double, -1, 3>& p,
                                     Vec3<double> origin, bool finite_controls = false) {
  if (order != 4 && order != 6) throw std::invalid_argument("spline order must be 4 or 6");
  if (h_ns <= 0 || h_ns > INT32_MAX) throw std::invalid_argument("h_ns must be in [1, 2^31)");
  const size_t n = q.shape(0);
  if (p.shape(0) != n) throw std::invalid_argument("q and p need the same number of rows");
  if (n < static_cast<size_t>(order)) throw std::invalid_argument("need at least `order` control points");
  std::vector<Quat<double>> qs(n);
  std::vector<Vec3<double>> ps(n);
  for (size_t c = 0; c < n; ++c) {
    qs[c] = {q(c, 0), q(c, 1), q(c, 2), q(c, 3)};
    ps[c] = {p(c, 0), p(c, 1), p(c, 2)};
    // As render::validate: a NaN or a zero quaternion made the nodes NaN, which the bounds' max then dropped.
    const double qq = qs[c].w * qs[c].w + qs[c].x * qs[c].x + qs[c].y * qs[c].y + qs[c].z * qs[c].z;
    if (finite_controls &&
        !(std::isfinite(qq) && qq > 0 && std::isfinite(ps[c].x) && std::isfinite(ps[c].y) && std::isfinite(ps[c].z))) {
      throw std::invalid_argument("control " + std::to_string(c) + ": need finite p and a finite, non-zero q");
    }
  }
  return {order, t0_ns, static_cast<int32_t>(h_ns), static_cast<int32_t>(n) - order + 1,
          traj::build_spline_nodes(qs, ps, origin)};
}

// int32 offsets of all queries from one knot-aligned base, as on the device (M2 A10).
struct OffsetBatch {
  int32_t base_segment;
  std::vector<int32_t> tau_ns;
};

// Query times: int64 ns of any layout; other dtypes are rejected, not converted.
using TimeArray = ExactIn<int64_t, -1>;

inline OffsetBatch make_offsets(const SplineInput& s, const TimeArray& t_ns) {
  const size_t n = t_ns.shape(0);
  OffsetBatch b{0, std::vector<int32_t>(n)};
  if (n == 0) return b;
  int64_t t_min = t_ns(0);
  for (size_t i = 1; i < n; ++i) t_min = std::min(t_min, t_ns(i));
  b.base_segment = static_cast<int32_t>(std::clamp<int64_t>(
      time::floor_div(t_min - s.t0_ns, int64_t{s.h_ns}), 0, s.num_segments - 1));
  const int64_t base_ns = s.t0_ns + int64_t{b.base_segment} * s.h_ns;
  for (size_t i = 0; i < n; ++i) {
    if (!time::offset_fits(t_ns(i), base_ns)) {
      throw std::invalid_argument("query times span more than 2^31 ns; split the batch");
    }
    b.tau_ns[i] = time::to_offset(t_ns(i), base_ns);
  }
  return b;
}

[[noreturn]] inline void throw_out_of_span(int64_t t_ns) {
  throw std::invalid_argument("query time " + std::to_string(t_ns) + " ns is outside the spline span");
}

template <class T>
nb::dict kinematics_to_dict(const std::vector<traj::Kinematics<T>>& k) {
  const size_t n = k.size();
  std::vector<T> q(4 * n), p(3 * n), w(3 * n), wd(3 * n), v(3 * n), a(3 * n);
  auto put3 = [](std::vector<T>& dst, size_t i, Vec3<T> x) {
    dst[3 * i] = x.x;
    dst[3 * i + 1] = x.y;
    dst[3 * i + 2] = x.z;
  };
  for (size_t i = 0; i < n; ++i) {
    q[4 * i] = k[i].q.w;
    q[4 * i + 1] = k[i].q.x;
    q[4 * i + 2] = k[i].q.y;
    q[4 * i + 3] = k[i].q.z;
    put3(p, i, k[i].p);
    put3(w, i, k[i].omega);
    put3(wd, i, k[i].omega_dot);
    put3(v, i, k[i].v);
    put3(a, i, k[i].a);
  }
  nb::dict d;
  d["q"] = to_numpy(std::move(q), {n, 4});
  d["p"] = to_numpy(std::move(p), {n, 3});
  d["omega"] = to_numpy(std::move(w), {n, 3});
  d["omega_dot"] = to_numpy(std::move(wd), {n, 3});
  d["v"] = to_numpy(std::move(v), {n, 3});
  d["a"] = to_numpy(std::move(a), {n, 3});
  return d;
}

inline Vec3<double> to_vec3(const In<double, 3>& x) { return {x(0), x(1), x(2)}; }

}  // namespace balla::bindings
