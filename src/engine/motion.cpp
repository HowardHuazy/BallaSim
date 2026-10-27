// Window motion, camera geometry and the host stages of the tile planner (motion.h).
#include "engine/motion.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>

#include "core/camera/injective_domain.h"
#include "core/camera/kannala_brandt.h"
#include "core/camera/radtan.h"
#include "core/traj/bounds.h"
#include "core/traj/pieces.h"
#include "core/traj/spline_build.h"
#include "render/ao_proxy.h"

namespace balla::engine {

namespace {

using np::dot_ein;
using np::dot_seq;
using np::norm_seq;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// R @ v of a C-ordered 3 x 3 (dgemv_t: rows 0, 1 and 2 of m = 3).
Vec3d mat_vec(const Mat3d& r, Vec3d v) {
  return {np::gemv3(r.row(0), v, 0, 3), np::gemv3(r.row(1), v, 1, 3), np::gemv3(r.row(2), v, 2, 3)};
}
// R.T @ v.
Vec3d mat_t_vec(const Mat3d& r, Vec3d v) { return {dot_seq(r.col(0), v), dot_seq(r.col(1), v), dot_seq(r.col(2), v)}; }
Mat3d mat_mul(const Mat3d& a, const Mat3d& b) {
  Mat3d out{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) out.m[i][j] = np::dot_fma_a(a.row(i), b.col(j));
  return out;
}
Mat3d top_left(const std::array<double, 16>& t) { return {{{t[0], t[1], t[2]}, {t[4], t[5], t[6]}, {t[8], t[9], t[10]}}}; }
Vec3d translation(const std::array<double, 16>& t) { return {t[3], t[7], t[11]}; }
Vec3d vec(const std::array<double, 3>& a) { return {a[0], a[1], a[2]}; }
Vec3d scaled(Vec3d a, double d) { return {a.x / d, a.y / d, a.z / d}; }
Vec3d vertex(const std::vector<float>& pos, size_t i) {
  return {static_cast<double>(pos[3 * i]), static_cast<double>(pos[3 * i + 1]), static_cast<double>(pos[3 * i + 2])};
}
// Python's min / max of two floats.
double py_min(double a, double b) { return b < a ? b : a; }
double py_max(double a, double b) { return b > a ? b : a; }
double np_minimum(double a, double b) { return a != a ? a : (b != b ? b : (b < a ? b : a)); }
double clip1(double x) { return x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x); }

}  // namespace

Mat3d quat_to_matrix(const Quat<double>& q) {
  const double n = std::sqrt(((q.x * q.x + q.y * q.y) + q.z * q.z) + q.w * q.w);
  const double x = q.x / n, y = q.y / n, z = q.z / n, w = q.w / n;
  const double x2 = x * x, y2 = y * y, z2 = z * z, w2 = w * w;
  const double xy = x * y, zw = z * w, xz = x * z, yw = y * w, yz = y * z, xw = x * w;
  return {{{((x2 - y2) - z2) + w2, 2 * (xy - zw), 2 * (xz + yw)},
           {2 * (xy + zw), ((-x2 + y2) - z2) + w2, 2 * (yz - xw)},
           {2 * (xz - yw), 2 * (yz + xw), ((-x2 - y2) + z2) + w2}}};
}

// ---------------------------------------------------------------------------------------------
// Trajectories
// ---------------------------------------------------------------------------------------------

TrajectoryMotion::TrajectoryMotion(const render::Trajectory& t, Holds holds)
    : order_(t.order), holds_(std::move(holds)) {
  const size_t n = t.rot.size() / 4;
  std::vector<Quat<double>> q(n);
  std::vector<Vec3d> p(n);
  for (size_t c = 0; c < n; ++c) {
    q[c] = {t.rot[4 * c], t.rot[4 * c + 1], t.rot[4 * c + 2], t.rot[4 * c + 3]};
    p[c] = {t.pos[3 * c], t.pos[3 * c + 1], t.pos[3 * c + 2]};
  }
  // One piece for a single spline (rows [0, n)); a piecewise trajectory's pieces each from their own rows.
  std::vector<render::TrajectoryPiece> table = t.pieces;
  if (table.empty()) {
    table.push_back({t.span_begin_ns, t.t0_ns, t.h_ns, 0, static_cast<int32_t>(n), 0});
  }
  nodes_.reserve(n);
  for (const render::TrajectoryPiece& r : table) {
    const auto c0 = static_cast<size_t>(r.ctrl_begin), rows = static_cast<size_t>(r.ctrl_end) - c0;
    const auto part = traj::build_spline_nodes(std::span<const Quat<double>>(q).subspan(c0, rows),
                                               std::span<const Vec3d>(p).subspan(c0, rows), {0.0, 0.0, 0.0});
    nodes_.insert(nodes_.end(), part.begin(), part.end());
    const Piece piece{r.begin_ns, r.t0_ns, static_cast<int32_t>(r.h_ns), static_cast<int32_t>(rows) - order_ + 1, c0,
                      v_.size()};
    pieces_.push_back(piece);
    has_step_ = has_step_ || r.join < 0;
    const traj::SplineView<double> view{nodes_.data() + c0, piece.segments, piece.h_ns, 1e9 / piece.h_ns};
    for (int32_t s = 0; s < piece.segments; ++s) {
      const traj::MotionBounds<double> b = order_ == 4 ? traj::segment_bounds<4>(view, s) : traj::segment_bounds<6>(view, s);
      v_.push_back(b.v);
      omega_.push_back(b.omega);
    }
  }
  // A join labelled C0 or C1 must keep the pose (kJoinTol*): compare the left piece's pose at b with the right one's.
  for (size_t k = 1; k < table.size() && !jump_; ++k) {
    if (table[k].join < 0) continue;
    const int64_t b = table[k].begin_ns;
    const traj::Kinematics<double> l = evaluate(pieces_[k - 1], b), r = evaluate(pieces_[k], b);
    const Quat<double> d = conjugate(l.q) * r.q;  // the turn from the left pose to the right one
    const double m = norm_seq(r.p - l.p);
    const double rad = 2.0 * std::atan2(std::sqrt((d.x * d.x + d.y * d.y) + d.z * d.z), std::fabs(d.w));
    if (!(m <= kJoinTolM && rad <= kJoinTolRad)) jump_ = JoinJump{static_cast<int32_t>(k), table[k].join, b, m, rad};
  }
}

traj::Kinematics<double> TrajectoryMotion::evaluate(const Piece& pc, int64_t t_ns) const {
  traj::SegmentTime st{};
  if (!traj::locate(t_ns, pc.t0_ns, pc.h_ns, pc.segments, &st)) {
    throw std::invalid_argument("query time " + std::to_string(t_ns) + " ns is outside the spline span");
  }
  const traj::SplineView<double> view{nodes_.data() + pc.node0, pc.segments, pc.h_ns, 1e9 / pc.h_ns};
  return order_ == 4 ? traj::evaluate<4>(view, st) : traj::evaluate<6>(view, st);
}

WindowMotion TrajectoryMotion::window(int64_t t0_ns, int64_t t1_ns) const {
  // The pieces serving [t0, t1] and, in each, the segments its part of the interval touches (clamped to the piece's
  // spline, as one spline clamps them); the bounds' maximum over all of them, in time order.
  const auto num = static_cast<int32_t>(pieces_.size());
  const int32_t p0 = traj::piece_index(pieces_.data(), num, t0_ns), p1 = traj::piece_index(pieces_.data(), num, t1_ns);
  bool first = true;
  double v = 0.0, omega = 0.0;
  for (int32_t k = p0; k <= p1; ++k) {
    const Piece& pc = pieces_[static_cast<size_t>(k)];
    const auto seg = [&](int64_t t) {
      return std::min<int64_t>(std::max<int64_t>(time::floor_div(t - pc.t0_ns, int64_t{pc.h_ns}), 0), pc.segments - 1);
    };
    const int64_t a = k == p0 ? t0_ns : pc.begin_ns;
    const int64_t b = k == p1 ? t1_ns : pieces_[static_cast<size_t>(k) + 1].begin_ns - 1;
    for (int64_t s = seg(a); s <= seg(b); ++s) {
      const size_t i = pc.seg0 + static_cast<size_t>(s);
      v = first ? v_[i] : np::maximum(v, v_[i]);
      omega = first ? omega_[i] : np::maximum(omega, omega_[i]);
      first = false;
    }
  }
  if (first) {  // t1 < t0: the segment of t0, as a single spline gives it
    const Piece& pc = pieces_[static_cast<size_t>(p0)];
    const int64_t s = std::min<int64_t>(std::max<int64_t>(time::floor_div(t0_ns - pc.t0_ns, int64_t{pc.h_ns}), 0),
                                        pc.segments - 1);
    v = v_[pc.seg0 + static_cast<size_t>(s)];
    omega = omega_[pc.seg0 + static_cast<size_t>(s)];
  }
  bool in_hold = false;
  for (const auto& [b, e] : holds_) in_hold = in_hold || (b <= t0_ns && t1_ns < e);
  // Zero bounds make the pose bitwise constant within one piece only: across a join the two pieces' constants may
  // differ (by up to kJoinTol*), so there only a declared hold holds (a single spline is always one piece).
  return {v, omega, in_hold || (v == 0.0 && omega == 0.0 && p0 == p1)};
}

std::pair<Mat3d, Vec3d> TrajectoryMotion::pose(int64_t t_ns) const {
  const Piece& pc = pieces_[static_cast<size_t>(traj::piece_index(pieces_.data(), static_cast<int32_t>(pieces_.size()),
                                                                  t_ns))];
  const traj::Kinematics<double> k = evaluate(pc, t_ns);
  return {quat_to_matrix(k.q), k.p};
}

// ---------------------------------------------------------------------------------------------
// Mesh distances
// ---------------------------------------------------------------------------------------------

namespace {

constexpr size_t kSub = 16;  // triangles per sub-group (kCluster / 8)

// The bounding sphere (box centre, largest vertex distance) of triangles [t0, t1).
std::pair<Vec3d, double> sphere_of(const std::vector<Vec3d>& v, const std::vector<std::array<uint32_t, 3>>& tri,
                                   size_t t0, size_t t1) {
  Vec3d lo = v[tri[t0][0]], hi = lo;
  for (size_t t = t0; t < t1; ++t) {
    for (const uint32_t k : tri[t]) {
      lo = {np_minimum(lo.x, v[k].x), np_minimum(lo.y, v[k].y), np_minimum(lo.z, v[k].z)};
      hi = {np::maximum(hi.x, v[k].x), np::maximum(hi.y, v[k].y), np::maximum(hi.z, v[k].z)};
    }
  }
  const Vec3d centre{0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y), 0.5 * (lo.z + hi.z)};
  double radius = 0.0;
  for (size_t t = t0; t < t1; ++t) {
    double reach = norm_seq(v[tri[t][0]] - centre);
    reach = np::maximum(reach, norm_seq(v[tri[t][1]] - centre));
    reach = np::maximum(reach, norm_seq(v[tri[t][2]] - centre));
    radius = t == t0 ? reach : np::maximum(radius, reach);
  }
  return {centre, radius};
}

// Margins of the pruning bounds: far above the rounding of the values they bound (relative 1e-15).
constexpr double kPruneAbs = 1e-7, kPruneRel = 1e-9;

}  // namespace

MeshDistance::MeshDistance(const render::Mesh& mesh) {
  const size_t nv = mesh.positions.size() / 3, nt = mesh.indices.size() / 3;
  if (nv == 0 || nt == 0) throw std::invalid_argument("MeshDistance: the mesh has no triangle");
  v_.resize(nv);
  for (size_t i = 0; i < nv; ++i) {
    v_[i] = vertex(mesh.positions, i);
    if (!std::isfinite(v_[i].x) || !std::isfinite(v_[i].y) || !std::isfinite(v_[i].z)) {
      throw std::invalid_argument("MeshDistance: positions are not finite");
    }
  }
  Vec3d lo = v_[0], hi = v_[0];
  for (const Vec3d& p : v_) {
    lo = {np_minimum(lo.x, p.x), np_minimum(lo.y, p.y), np_minimum(lo.z, p.z)};
    hi = {np::maximum(hi.x, p.x), np::maximum(hi.y, p.y), np::maximum(hi.z, p.z)};
  }
  const Vec3d ext = hi - lo;
  const double span = py_max(np::maximum(np::maximum(ext.x, ext.y), ext.z), 1e-30);
  std::vector<uint64_t> code(nt);
  std::vector<std::array<uint32_t, 3>> tri(nt);
  for (size_t t = 0; t < nt; ++t) {
    tri[t] = {mesh.indices[3 * t], mesh.indices[3 * t + 1], mesh.indices[3 * t + 2]};
    if (tri[t][0] >= nv || tri[t][1] >= nv || tri[t][2] >= nv) {
      throw std::invalid_argument("MeshDistance: vertex index out of range");
    }
    const Vec3d a = v_[tri[t][0]], b = v_[tri[t][1]], d = v_[tri[t][2]];
    const double cen[3] = {((a.x + b.x) + d.x) / 3.0, ((a.y + b.y) + d.y) / 3.0, ((a.z + b.z) + d.z) / 3.0};
    const double lo3[3] = {lo.x, lo.y, lo.z};
    for (int axis = 0; axis < 3; ++axis) {
      const auto cell = static_cast<int64_t>(np_minimum((cen[axis] - lo3[axis]) / span * 1024, 1023.0));
      for (int bit = 0; bit < 10; ++bit) code[t] |= static_cast<uint64_t>((cell >> bit) & 1) << (3 * bit + axis);
    }
    // point_triangle_distance's edge-ab case divides by |ab|^2, 0 when a == b: (b, c, a) puts the repeated
    // vertex in a and c, where every query ends in its vertex-a, vertex-b or edge-ab case (the exact distance
    // to the segment).
    if (a.x == b.x && a.y == b.y && a.z == b.z) tri[t] = {tri[t][1], tri[t][2], tri[t][0]};
  }
  std::vector<size_t> order(nt);
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return code[a] < code[b]; });
  tri_.resize(nt);
  for (size_t t = 0; t < nt; ++t) tri_[t] = tri[order[t]];
  for (size_t s0 = 0; s0 < nt; s0 += kCluster) {
    const auto [centre, radius] = sphere_of(v_, tri_, s0, std::min(nt, s0 + kCluster));
    centre_.push_back(centre);
    radius_.push_back(radius);
  }
  for (size_t s0 = 0; s0 < nt; s0 += kSub) {
    const auto [centre, radius] = sphere_of(v_, tri_, s0, std::min(nt, s0 + kSub));
    sub_centre_.push_back(centre);
    sub_radius_.push_back(radius);
  }
  box_centre_ = {0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y), 0.5 * (lo.z + hi.z)};
}

namespace {

// Whether the segment o + t d, t in [0, len] (d a unit vector), meets the sphere (c, r).
bool segment_meets(Vec3d o, Vec3d d, double len, Vec3d c, double r) {
  const Vec3d oc = c - o;
  const double t = std::clamp(dot_seq(oc, d), 0.0, len);
  const Vec3d e = oc - t * d;
  return dot_seq(e, e) <= r * r * (1.0 + 1e-9) + 1e-300;
}

// The ray parameter of o + t d on triangle (a, b, c) (Moller-Trumbore), inf where it misses.
double ray_triangle(Vec3d o, Vec3d d, Vec3d a, Vec3d b, Vec3d c) {
  const Vec3d e1 = b - a, e2 = c - a, p = cross(d, e2);
  const double det = dot_seq(e1, p);
  if (!(std::fabs(det) > 0.0)) return kInf;
  const double inv = 1.0 / det;
  const Vec3d s = o - a;
  const double u = dot_seq(s, p) * inv;
  if (u < 0.0 || u > 1.0) return kInf;
  const Vec3d q = cross(s, e1);
  const double v = dot_seq(d, q) * inv;
  if (v < 0.0 || u + v > 1.0) return kInf;
  return dot_seq(e2, q) * inv;
}

}  // namespace

double MeshDistance::thickness(double cap) const {
  Vec3d lo = v_[0], hi = v_[0];
  for (const Vec3d& p : v_) {
    lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
    hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
  }
  const double span = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
  double volume6 = 0.0;
  for (const auto& t : tri_) volume6 += dot_seq(v_[t[0]] - lo, cross(v_[t[1]] - lo, v_[t[2]] - lo));
  if (!(std::fabs(volume6) > 1e-9 * span * span * span)) return 0.0;
  const double inward = volume6 > 0 ? -1.0 : 1.0;  // a positive volume: the winding's normals point outwards
  const double eps = 1e-7 * span;                  // hits nearer than this: the triangle's own plane
  const size_t nt = tri_.size(), stride = (nt + kThicknessSamples - 1) / kThicknessSamples;
  std::vector<std::pair<double, double>> samples;  // (thickness at the triangle, its area)
  double area = 0.0;
  for (size_t k = 0; k < nt; k += stride) {
    const Vec3d a = v_[tri_[k][0]], b = v_[tri_[k][1]], c = v_[tri_[k][2]];
    const Vec3d nrm = cross(b - a, c - a);
    const double len = norm_seq(nrm);
    if (!(len > 0.0)) continue;
    const Vec3d d = (inward / len) * nrm;
    const Vec3d o{((a.x + b.x) + c.x) / 3.0, ((a.y + b.y) + c.y) / 3.0, ((a.z + b.z) + c.z) / 3.0};
    double best = cap;
    for (size_t ci = 0; ci < centre_.size(); ++ci) {
      if (!segment_meets(o, d, best, centre_[ci], radius_[ci])) continue;
      const size_t t_end = std::min(nt, (ci + 1) * kCluster);
      for (size_t g = ci * kCluster / kSub; g * kSub < t_end; ++g) {
        if (!segment_meets(o, d, best, sub_centre_[g], sub_radius_[g])) continue;
        for (size_t u = g * kSub; u < std::min(t_end, (g + 1) * kSub); ++u) {
          const double t = u == k ? kInf : ray_triangle(o, d, v_[tri_[u][0]], v_[tri_[u][1]], v_[tri_[u][2]]);
          if (t > eps && t < best) best = t;
        }
      }
    }
    samples.emplace_back(best, 0.5 * len);
    area += 0.5 * len;
  }
  // The thinnest part that holds kThinArea of the (sampled) surface: wrinkles, slivers and duplicate faces of scanned
  // meshes give single triangles nanometre-to-micrometre hits (ENGINE_CPP "I8 review resolution").
  std::sort(samples.begin(), samples.end());
  double run = 0.0;
  for (const auto& [t, w] : samples) {
    run += w;
    if (run >= kThinArea * area) return std::min(t, cap);
  }
  return cap;
}

double point_triangle_distance(Vec3d p, Vec3d a, Vec3d b, Vec3d c) {
  const Vec3d ab = b - a, ac = c - a;
  const double d1 = dot_ein(ab, p - a), d2 = dot_ein(ac, p - a);
  const double d3 = dot_ein(ab, p - b), d4 = dot_ein(ac, p - b);
  const double d5 = dot_ein(ab, p - c), d6 = dot_ein(ac, p - c);
  const double va = d3 * d6 - d5 * d4, vb = d5 * d2 - d1 * d6, vc = d1 * d4 - d3 * d2;
  double v = 0, w = 0;
  if (d1 <= 0 && d2 <= 0) {
  } else if (d3 >= 0 && d4 <= d3) {
    v = 1.0;
  } else if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    v = d1 / (d1 - d3);
  } else if (d6 >= 0 && d5 <= d6) {
    w = 1.0;
  } else if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    w = d2 / (d2 - d6);
  } else if (va <= 0 && d4 >= d3 && d5 >= d6) {
    const double e = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    v = 1.0 - e;
    w = e;
  } else {
    const double inner = 1.0 / ((va + vb) + vc);
    v = vb * inner;
    w = vc * inner;
  }
  const Vec3d closest{(a.x + ab.x * v) + ac.x * w, (a.y + ab.y * v) + ac.y * w, (a.z + ab.z * v) + ac.z * w};
  return norm_seq(closest - p);
}

// min(run, exact distances of cluster i's triangles); groups whose sphere is farther than run (plus
// rounding) are skipped, since their exact distances cannot be below it.
double MeshDistance::triangles_distance(Vec3d q, size_t i, double run) const {
  const size_t t_end = std::min(tri_.size(), (i + 1) * kCluster);
  for (size_t g = i * kCluster / kSub; g * kSub < t_end; ++g) {
    const double lower = norm_seq(sub_centre_[g] - q) - sub_radius_[g];
    if (lower > run + kPruneRel * (1.0 + std::fabs(lower) + sub_radius_[g])) continue;
    for (size_t t = g * kSub; t < std::min(t_end, (g + 1) * kSub); ++t) {
      run = np_minimum(run, point_triangle_distance(q, v_[tri_[t][0]], v_[tri_[t][1]], v_[tri_[t][2]]));
    }
  }
  return run;
}

double MeshDistance::distance(Vec3d q) const {
  const size_t m = centre_.size();
  std::vector<double> lower(m);
  size_t arg = 0;
  for (size_t i = 0; i < m; ++i) {
    lower[i] = norm_seq(centre_[i] - q) - radius_[i];
    if (lower[i] < lower[arg]) arg = i;
  }
  // Python: min(best, min of the tight near clusters' bounds, min of the loose near clusters' exact
  // distances), near = lower < best. The loose clusters are taken in order of their bound, and those
  // whose bound exceeds the running minimum by more than rounding are skipped: their exact distance
  // cannot be below it, so the minimum is the same.
  const double best = triangles_distance(q, arg, kInf);
  double run = best;
  std::vector<std::pair<double, size_t>> loose;
  for (size_t i = 0; i < m; ++i) {
    if (!(lower[i] < best)) continue;
    if (radius_[i] > kRel * lower[i]) {
      loose.emplace_back(lower[i], i);
    } else {
      run = py_min(run, lower[i]);
    }
  }
  std::sort(loose.begin(), loose.end());
  for (const auto& [lo, i] : loose) {
    if (lo > run + kPruneRel * (1.0 + std::fabs(lo) + radius_[i])) break;
    run = py_min(run, triangles_distance(q, i, run));
  }
  return run;
}

// Python evaluates acos / asin for every cluster and every vertex of the selected clusters. Here bounds
// skip what cannot change the result: a cluster's angle + spread is at most tan(angle) + tan(spread)
// (both below pi / 2), and every vertex of a cluster lies within its reach of the axis. Both bounds get
// margins (1e-7 absolute, 1e-9 relative) far above the rounding of the evaluated values, including
// acos near 1 (sqrt(2 * 1e-15) < 1e-7), so the floor, the selection and the minimum cosine are Python's.
std::optional<std::pair<Vec3d, double>> MeshDistance::cone(Vec3d q, ConeScratch& scratch) const {
  Vec3d axis = box_centre_ - q;
  const double n = norm_seq(axis);
  if (n == 0.0) return std::nullopt;
  axis = scaled(axis, n);
  const size_t m = centre_.size();
  // Cheap pass: outside flags, bounds of angle + spread (upper) and angle - spread (lower).
  std::vector<double> upper(m), reach(m, kInf);
  std::vector<uint8_t> state(m, 0);  // 0 inside, 1 outside and skipped, 2 outside and evaluated
  size_t seed = m;
  double seed_lo = -kInf;
  for (size_t i = 0; i < m; ++i) {
    const Vec3d rel = centre_[i] - q;
    const double dist = norm_seq(rel);
    if (!(dist > radius_[i])) continue;
    state[i] = 1;
    const double cs = np::gemv3(rel, axis, i, m) / dist, sn = norm_seq(cross(rel, axis)) / dist;
    const double s = radius_[i] / dist, ts = s / std::sqrt(1.0 - s * s);
    upper[i] = cs > 0 ? (sn / cs + ts) * (1 + kPruneRel) + kPruneAbs : kInf;
    if (sn - ts > seed_lo) {
      seed_lo = sn - ts;
      seed = i;
    }
  }
  // The floor: max(angle - spread) over outside clusters, evaluated where the bound reaches it.
  double floor = 0.0;
  bool any = false;
  const auto evaluate = [&](size_t i) {
    const Vec3d rel = centre_[i] - q;
    const double dist = norm_seq(rel);
    const double angle = std::acos(clip1(np::gemv3(rel, axis, i, m) / dist));
    const double spread = std::asin(np_minimum(radius_[i] / dist, 1.0));
    floor = any ? np::maximum(floor, angle - spread) : angle - spread;
    any = true;
    reach[i] = angle + spread;
    state[i] = 2;
  };
  if (seed < m) evaluate(seed);
  for (size_t i = 0; i < m; ++i) {
    if (state[i] == 1 && !(upper[i] < floor)) evaluate(i);
  }
  // Rows of (pts @ axis) / |pts| over the selected clusters (reach >= floor) in cluster order; the last
  // rows (BLAS tail, row >= q4) are evaluated in their own order, the others in dgemv_t's block order.
  std::vector<size_t> sel;
  size_t rows = 0;
  for (size_t i = 0; i < m; ++i) {
    if (state[i] == 1 || !(reach[i] >= floor)) continue;
    sel.push_back(i);
    rows += 3 * (std::min(tri_.size(), (i + 1) * kCluster) - i * kCluster);
  }
  const size_t q4 = rows & ~size_t{3};
  std::vector<uint32_t>& stamp = scratch.stamp;
  if (stamp.size() < v_.size()) stamp.resize(v_.size(), 0u);
  if (++scratch.epoch == 0) {
    std::fill(stamp.begin(), stamp.end(), 0u);
    scratch.epoch = 1;
  }
  const uint32_t epoch = scratch.epoch;
  double cos_min = kInf;
  bool first = true;
  const auto take = [&](double cs) {
    cos_min = first ? cs : np_minimum(cos_min, cs);
    first = false;
  };
  // Tail rows.
  if (q4 < rows) {
    const size_t i = sel.back(), t_end = std::min(tri_.size(), (i + 1) * kCluster);
    for (size_t r = q4; r < rows; ++r) {
      const size_t k = rows - 1 - r;  // rows counted back from the cluster's last vertex
      const Vec3d p = v_[tri_[t_end - 1 - k / 3][2 - k % 3]] - q;
      take(np::gemv3(p, axis, r, rows) / norm_seq(p));
    }
  }
  // Block rows, by groups of kSub triangles: a vertex of a group lies within theta + sigma of the axis
  // (theta its sphere's angle, sigma = asin(r / d)), so its cosine is at least cos(theta + sigma) =
  // cos(theta) cos(sigma) - sin(theta) sin(sigma) (both below pi / 2); groups whose bound exceeds the
  // running minimum are skipped. The group of the smallest bound goes first.
  struct Group {
    size_t g, row0;
    double bound;
  };
  std::vector<Group> groups;
  for (size_t j = 0, r = 0; j < sel.size(); ++j) {
    const size_t i = sel[j], t_end = std::min(tri_.size(), (i + 1) * kCluster);
    for (size_t g = i * kCluster / kSub; g * kSub < t_end; ++g) {
      const Vec3d rel = sub_centre_[g] - q;
      const double d = norm_seq(rel), sg = sub_radius_[g] / d;
      const double cs = dot_seq(rel, axis) / d, sn = norm_seq(cross(rel, axis)) / d;
      const double bound = (d > sub_radius_[g] && cs > 0) ? cs * std::sqrt(1.0 - sg * sg) - sn * sg - kPruneRel : -kInf;
      groups.push_back({g, r, bound});
      r += 3 * (std::min(t_end, (g + 1) * kSub) - g * kSub);
    }
  }
  const auto group_rows = [&](const Group& gr) {
    size_t r = gr.row0;
    for (size_t t = gr.g * kSub; t < std::min(tri_.size(), (gr.g + 1) * kSub) && r < q4; ++t) {
      for (const uint32_t k : tri_[t]) {
        if (r < q4 && stamp[k] != epoch) {
          stamp[k] = epoch;
          const Vec3d p = v_[k] - q;
          take(np::dot_fma_b(p, axis) / norm_seq(p));
        }
        ++r;
      }
    }
  };
  if (!groups.empty()) {
    const auto low = std::min_element(groups.begin(), groups.end(),
                                      [](const Group& x, const Group& y) { return x.bound < y.bound; });
    group_rows(*low);
    for (const Group& gr : groups) {
      if (&gr == &*low || gr.bound > cos_min) continue;
      group_rows(gr);
    }
  }
  return std::make_pair(axis, std::acos(clip1(cos_min)));
}

Vec3d BodyWindow::to_body(Vec3d p) const { return mat_t_vec(rotation, p - centre); }

double BodyWindow::distance(Vec3d p, double p_speed) const {
  if (mesh == nullptr) throw std::logic_error("BodyWindow::distance: the body's mesh distance was not built");
  return mesh->distance(to_body(p)) - (point_speed() + p_speed) * half_s;
}

// ---------------------------------------------------------------------------------------------
// Scene motion
// ---------------------------------------------------------------------------------------------

SceneMotion::SceneMotion(const render::SceneBundle& bundle, std::vector<Holds> holds, bool planner_meshes)
    : bundle_(bundle) {
  holds.resize(std::max(holds.size(), bundle.trajectories.size()));
  for (size_t i = 0; i < bundle.trajectories.size(); ++i) {
    trajectories_.emplace_back(bundle.trajectories[i], std::move(holds[i]));
    // A step join makes the pose jump: no window bound covers it, and the engine has no forced nodes at the step
    // (blender.md §9 BL-12), so it is refused rather than sampled as if the motion were continuous.
    if (trajectories_.back().has_step()) {
      throw std::invalid_argument("engine: trajectory " + std::to_string(i) +
                                  " has a step join (pieces join -1), which the planner does not bound (Phase 3 BL-12)");
    }
    // A join labelled C0 or C1 whose pose jumps is a step in fact: the window bounds would not cover it either.
    if (const std::optional<JoinJump>& j = trajectories_.back().jump()) {
      char what[384];
      std::snprintf(what, sizeof what,
                    "engine: trajectory %zu piece %d is joined C%d (pieces join %d) but its pose jumps by %.3g m and "
                    "%.3g rad at %lld ns (a C0 or C1 join keeps it within %g m and %g rad; a jump is a step join, -1, "
                    "which the planner does not bound until Phase 3 BL-12)",
                    i, static_cast<int>(j->piece), static_cast<int>(j->join), static_cast<int>(j->join), j->m, j->rad,
                    static_cast<long long>(j->at_ns), kJoinTolM, kJoinTolRad);
      throw std::invalid_argument(what);
    }
  }
  // The planner queries the meshes of moving bodies (touches, shadows of static lights) and, for a
  // moving point or spot light that casts shadows, of every body but its carrier (moving_light).
  const size_t nb = bundle.bodies.size();
  std::vector<int32_t> moving_shadow_carriers;
  for (const render::Light& l : bundle.lights) {
    if (l.body && bundle.bodies.at(static_cast<size_t>(*l.body)).trajectory && l.casts_shadows &&
        (l.kind == render::BundleLightKind::kPoint || l.kind == render::BundleLightKind::kSpot)) {
      moving_shadow_carriers.push_back(*l.body);
    }
  }
  meshes_.resize(nb);
  for (size_t i = 0; i < nb; ++i) {
    const render::Body& b = bundle.bodies[i];
    double r = 0.0;
    Vec3d lo{kInf, kInf, kInf}, hi{-kInf, -kInf, -kInf};
    for (size_t k = 0; k < b.mesh.positions.size() / 3; ++k) {
      const Vec3d x = vertex(b.mesh.positions, k);
      const double d = norm_seq(x);
      r = k == 0 ? d : np::maximum(r, d);
      lo = {std::min(lo.x, x.x), std::min(lo.y, x.y), std::min(lo.z, x.z)};
      hi = {std::max(hi.x, x.x), std::max(hi.y, x.y), std::max(hi.z, x.z)};
    }
    radius_.push_back(r);
    extent_.push_back(b.mesh.positions.empty() ? 0.0 : std::min({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z}));
    const bool queried = b.trajectory.has_value() ||
                         std::any_of(moving_shadow_carriers.begin(), moving_shadow_carriers.end(),
                                     [&](int32_t c) { return static_cast<size_t>(c) != i; });
    // A meshless body (C1: a light's carrier, blender.md §3.7) has no distance to build: body_windows omits it.
    if (planner_meshes && queried && !b.mesh.indices.empty()) meshes_[i].emplace(b.mesh);
  }
  // MOVING: bodies by instance id with their trajectory's holds.
  std::vector<uint32_t> order(nb);
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) { return bundle.bodies[x].id < bundle.bodies[y].id; });
  moving_.hold_off.push_back(0);
  for (const uint32_t i : order) {
    const render::Body& b = bundle.bodies[i];
    moving_.id.push_back(b.id);
    moving_.has_trajectory.push_back(b.trajectory ? 1 : 0);
    if (b.trajectory) {
      for (const auto& [hb, he] : trajectories_.at(static_cast<size_t>(*b.trajectory)).holds()) {
        moving_.holds.push_back(hb);
        moving_.holds.push_back(he);
      }
    }
    moving_.hold_off.push_back(static_cast<uint32_t>(moving_.holds.size() / 2));
  }
}

WindowMotion SceneMotion::camera_window(size_t cam, int64_t t0_ns, int64_t t1_ns) const {
  const render::Camera& c = bundle_.cameras.at(cam);
  const WindowMotion m = trajectories_.at(static_cast<size_t>(c.rig_trajectory)).window(t0_ns, t1_ns);
  const double lever = norm_seq(translation(c.T_rig_cam));
  return {m.v + m.omega * lever, m.omega, m.held};
}

std::pair<Mat3d, Vec3d> SceneMotion::camera_pose(size_t cam, int64_t t_ns) const {
  const render::Camera& c = bundle_.cameras.at(cam);
  const auto [r, p] = trajectories_.at(static_cast<size_t>(c.rig_trajectory)).pose(t_ns);
  return {mat_mul(r, top_left(c.T_rig_cam)), p + mat_vec(r, translation(c.T_rig_cam))};
}

SceneMotion::Carrier SceneMotion::carrier(size_t body, int64_t t0_ns, int64_t t1_ns) const {
  const render::Body& b = bundle_.bodies.at(body);
  if (!b.trajectory) return {top_left(b.T_world_body), translation(b.T_world_body), {}};
  const TrajectoryMotion& tm = trajectories_.at(static_cast<size_t>(*b.trajectory));
  const WindowMotion m = tm.window(t0_ns, t1_ns);
  const auto [r, p] = tm.pose(t0_ns + (t1_ns - t0_ns) / 2);
  return {r, p, m.held ? WindowMotion{} : m};
}

std::vector<BodyWindow> SceneMotion::body_windows(int64_t t0_ns, int64_t t1_ns) const {
  const double half_s = static_cast<double>(t1_ns - t0_ns) * 0.5e-9;
  std::vector<BodyWindow> out;
  for (size_t i = 0; i < bundle_.bodies.size(); ++i) {
    // Nothing hits, occludes or touches a meshless body (a light's carrier): leaving it out of every list the planner
    // builds from these windows (shadow bodies, touches, movers, spheres, the moving lights' kappa) is exact. Its
    // lights still ride it (light_windows, carrier()).
    if (bundle_.bodies[i].mesh.indices.empty()) continue;
    const Carrier c = carrier(i, t0_ns, t1_ns);
    out.push_back({static_cast<int32_t>(i), c.m.held, c.m.v, c.m.omega, radius_[i], c.p, c.r, half_s,
                   meshes_[i] ? &*meshes_[i] : nullptr, extent_[i]});
  }
  return out;
}

std::vector<LightWindow> SceneMotion::light_windows(int64_t t0_ns, int64_t t1_ns) const {
  std::vector<LightWindow> out;
  int32_t bits = 0;  // the renderer's shadow bits: the positional lights that cast shadows, in order
  for (const render::Light& light : bundle_.lights) {
    if (!render::positional(light.kind)) continue;
    Carrier c{{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {0, 0, 0}, {}};
    if (light.body) c = carrier(static_cast<size_t>(*light.body), t0_ns, t1_ns);
    LightWindow lw;
    lw.directional = light.kind == render::BundleLightKind::kDirectional;
    double lever = 0.0;
    if (lw.directional) {
      const Vec3d d = mat_vec(c.r, vec(light.direction));
      lw.position = scaled(d, norm_seq(d));
    } else {
      lw.position = c.p + mat_vec(c.r, vec(light.position));
      lever = norm_seq(vec(light.position));
    }
    lw.spot = light.kind == render::BundleLightKind::kSpot;
    if (lw.spot) {
      const Vec3d a = mat_vec(c.r, vec(light.direction));
      lw.axis = scaled(a, norm_seq(a));
      lw.outer = light.spot_outer_rad;
    }
    lw.casts_shadows = light.casts_shadows;
    lw.shadow_bit = light.casts_shadows ? bits++ : -1;
    lw.body = light.body ? *light.body : -1;
    lw.moving = !c.m.held;
    lw.v = c.m.v + c.m.omega * lever;
    lw.omega = c.m.omega;
    out.push_back(lw);
  }
  return out;
}

bool SceneMotion::body_moving_at(size_t body, int64_t t_ns) const {
  return moving_.table().moving(bundle_.bodies.at(body).id, t_ns);
}

std::vector<uint32_t> SceneMotion::held_bodies(int64_t t0_ns, int64_t t1_ns) const {
  std::vector<uint32_t> out;
  for (const render::Body& b : bundle_.bodies) {
    if (b.mesh.indices.empty()) continue;  // a meshless carrier has no geometry to join the world
    if (b.trajectory && trajectories_[static_cast<size_t>(*b.trajectory)].window(t0_ns, t1_ns).held) out.push_back(b.id);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Camera geometry
// ---------------------------------------------------------------------------------------------

namespace {

uint32_t spread2(uint32_t v) {
  v &= 0xFFFF;
  v = (v | (v << 8)) & 0x00FF00FF;
  v = (v | (v << 4)) & 0x0F0F0F0F;
  v = (v | (v << 2)) & 0x33333333;
  return (v | (v << 1)) & 0x55555555;
}

template <class Cam>
void fill_geometry(CameraGeometry& g, const Cam& model) {
  const double s = g.support_px;
  const auto unit_ray = [&](double u, double v) {
    Vec3d r{};
    if (!camera::unproject(model, Vec2<double>{u, v}, &r)) {
      throw std::invalid_argument("pixel positions outside the camera's injective domain");
    }
    return scaled(r, norm_seq(r));
  };
  const auto w = static_cast<size_t>(g.width), h = static_cast<size_t>(g.height);
  g.rays.resize(3 * g.n_pixels);
  g.pixel_tile.resize(g.n_pixels);
  for (size_t p = 0; p < g.n_pixels; ++p) {
    const Vec3d r = unit_ray(static_cast<double>(p % w), static_cast<double>(p / w));
    g.rays[3 * p] = r.x;
    g.rays[3 * p + 1] = r.y;
    g.rays[3 * p + 2] = r.z;
    g.pixel_tile[p] = static_cast<uint32_t>((p / w / kTile) * static_cast<size_t>(g.tiles_x) + (p % w) / kTile);
  }
  // ||J(d)|| on a 1-px lattice over the sub-sample support (grid row k is pixel row k - 1).
  const size_t gw = w + 2, gh = h + 2;
  const auto grid = [&](size_t i, size_t n) {
    return i == 0 ? -s : (i == n - 1 ? static_cast<double>(n - 3) + s : static_cast<double>(i - 1));
  };
  std::vector<double> norm(gw * gh);
  g.speed_factor_max = 0.0;
  g.speed_factor_min = kInf;
  for (size_t r = 0; r < gh; ++r) {
    for (size_t c = 0; c < gw; ++c) {
      camera::Mat23<double> j{};
      const Vec3d d = unit_ray(grid(c, gw), grid(r, gh));
      if (!camera::project_jacobian(model, d, &j)) {
        throw std::invalid_argument("camera: Jacobian undefined inside the sub-sample support");
      }
      const Vec3d j0{j.m[0][0], j.m[0][1], j.m[0][2]}, j1{j.m[1][0], j.m[1][1], j.m[1][2]};
      const double a = dot_ein(j0, j0), cc = dot_ein(j1, j1), b = dot_ein(j0, j1);
      const double v = std::sqrt(0.5 * (a + cc) + std::sqrt(0.25 * ((a - cc) * (a - cc)) + b * b));
      norm[r * gw + c] = v;
      g.speed_factor_max = (r == 0 && c == 0) ? v : np::maximum(g.speed_factor_max, v);
      // The smaller singular value (on a unit ray, the image scale across it); I8's band only.
      const double lo = 0.5 * (a + cc) - std::sqrt(0.25 * ((a - cc) * (a - cc)) + b * b);
      g.speed_factor_min = std::min(g.speed_factor_min, std::sqrt(std::max(lo, 0.0)));
    }
  }
  const auto tx_n = static_cast<size_t>(g.tiles_x), ty_n = static_cast<size_t>(g.tiles_y);
  std::vector<double> tile_j(tx_n * ty_n);
  for (size_t ty = 0; ty < ty_n; ++ty) {
    for (size_t tx = 0; tx < tx_n; ++tx) {
      double m = norm[ty * kTile * gw + tx * kTile];
      for (size_t r = ty * kTile; r < std::min((ty + 1) * kTile, h) + 2; ++r)
        for (size_t c = tx * kTile; c < std::min((tx + 1) * kTile, w) + 2; ++c) m = np::maximum(m, norm[r * gw + c]);
      tile_j[ty * tx_n + tx] = m;
    }
  }
  g.speed_factor.resize(tile_j.size());
  for (size_t ty = 0; ty < ty_n; ++ty) {
    for (size_t tx = 0; tx < tx_n; ++tx) {
      double m = tile_j[ty * tx_n + tx];
      for (size_t y = ty == 0 ? 0 : ty - 1; y <= std::min(ty + 1, ty_n - 1); ++y)
        for (size_t x = tx == 0 ? 0 : tx - 1; x <= std::min(tx + 1, tx_n - 1); ++x) m = np::maximum(m, tile_j[y * tx_n + x]);
      g.speed_factor[ty * tx_n + tx] = m;
    }
  }
  // Tile cones: the centre ray and the largest angle to the boundary of the tile's support.
  constexpr double kS[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
  for (size_t ty = 0; ty < ty_n; ++ty) {
    for (size_t tx = 0; tx < tx_n; ++tx) {
      const double y0 = static_cast<double>(ty * kTile) - s;
      const double y1 = static_cast<double>(std::min((ty + 1) * kTile, h) - 1) + s;
      const double x0 = static_cast<double>(tx * kTile) - s;
      const double x1 = static_cast<double>(std::min((tx + 1) * kTile, w) - 1) + s;
      const Vec3d centre = unit_ray((x0 + x1) / 2, (y0 + y1) / 2);
      Vec2<double> edge[20];
      for (int k = 0; k < 5; ++k) {
        edge[k] = {x0 + (x1 - x0) * kS[k], y0};
        edge[5 + k] = {x0 + (x1 - x0) * kS[k], y1};
        edge[10 + k] = {x0, y0 + (y1 - y0) * kS[k]};
        edge[15 + k] = {x1, y0 + (y1 - y0) * kS[k]};
      }
      double radius = 0.0;
      for (size_t k = 0; k < 20; ++k) {
        const double a = std::acos(clip1(np::gemv3(unit_ray(edge[k].x, edge[k].y), centre, k, 20)));
        radius = k == 0 ? a : np::maximum(radius, a);
      }
      g.tile_ray.push_back(centre);
      g.tile_radius.push_back(radius);
    }
  }
  // Request order (DESIGN §6.9): tile Morton code, then pixel id.
  std::vector<uint32_t> code(g.n_pixels);
  for (size_t p = 0; p < g.n_pixels; ++p) {
    const uint32_t t = g.pixel_tile[p];
    code[p] = spread2(t % static_cast<uint32_t>(g.tiles_x)) | (spread2(t / static_cast<uint32_t>(g.tiles_x)) << 1);
  }
  g.pixel_by_rank.resize(g.n_pixels);
  std::iota(g.pixel_by_rank.begin(), g.pixel_by_rank.end(), uint32_t{0});
  std::stable_sort(g.pixel_by_rank.begin(), g.pixel_by_rank.end(), [&](uint32_t a, uint32_t b) { return code[a] < code[b]; });
}

}  // namespace

CameraGeometry::CameraGeometry(const render::Camera& cam, double support)
    : support_px(support),
      width(cam.width),
      height(cam.height),
      tiles_x((cam.width + kTile - 1) / kTile),
      tiles_y((cam.height + kTile - 1) / kTile),
      n_pixels(static_cast<size_t>(cam.width) * static_cast<size_t>(cam.height)),
      speed_factor_max(0.0) {
  std::vector<double> k = cam.dist;
  if (cam.model == render::CameraModelKind::kRadtan) {
    k.resize(5, 0.0);
    camera::RadtanCamera<double> c{cam.fx, cam.fy, cam.cx, cam.cy, k[0], k[1], k[2], k[3], k[4], 0.0, 0.0};
    c.r_lim = camera::radtan_injective_domain(c, width, height, support_px).r_lim;
    fill_geometry(*this, c);
  } else {
    k.resize(4, 0.0);
    camera::KannalaBrandtCamera<double> c{cam.fx, cam.fy, cam.cx, cam.cy, k[0], k[1], k[2], k[3], 0.0};
    c.theta_lim = camera::kannala_brandt_injective_domain(c, width, height, support_px).theta_lim;
    fill_geometry(*this, c);
  }
}

// ---------------------------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------------------------

namespace {

// Scratch of the dilation: the zero-padded input and its prefix and suffix maxima.
struct DilateScratch {
  std::vector<double> p, g, h;
};

// The (2d + 1)-window maximum (zero outside, floor +0) along one axis of n elements of v contiguous
// values each (a[i * v + j]: element i, lane j), in place: van Herk / Gil-Werman, prefix and suffix
// maxima in blocks of 2d + 1 over the zero-padded axis, O(n v) for any d. mx is np.maximum.
template <class Max>
void dilate_axis(double* a, size_t n, size_t v, size_t d, DilateScratch& s, Max mx) {
  const size_t w = 2 * d + 1, len = n + 2 * d;
  s.p.assign(len * v, 0.0);
  std::copy(a, a + n * v, s.p.begin() + static_cast<std::ptrdiff_t>(d * v));
  s.g.resize(len * v);
  s.h.resize(len * v);
  const double* p = s.p.data();
  double* g = s.g.data();
  double* h = s.h.data();
  for (size_t b = 0; b < len; b += w) {
    const size_t e = std::min(b + w, len);
    for (size_t j = 0; j < v; ++j) g[b * v + j] = p[b * v + j];
    for (size_t k = b + 1; k < e; ++k)
      for (size_t j = 0; j < v; ++j) g[k * v + j] = mx(g[(k - 1) * v + j], p[k * v + j]);
    for (size_t j = 0; j < v; ++j) h[(e - 1) * v + j] = p[(e - 1) * v + j];
    for (size_t k = e - 1; k-- > b;)
      for (size_t j = 0; j < v; ++j) h[k * v + j] = mx(p[k * v + j], h[(k + 1) * v + j]);
  }
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < v; ++j) a[i * v + j] = mx(0.0, mx(h[i * v + j], g[(i + 2 * d) * v + j]));
}

// The cascade of dilate_by_reach: acc = D_{e' - e}(acc) max (the tiles of group e) from the largest
// group radius e down, then D_{e_min}(acc); D is the (2d + 1)^2 maximum filter (zero outside),
// separable: along each row, then down the columns (all rows as lanes).
// The tiles of group e are tile[first[e], first[e + 1]).
template <class Max>
std::vector<double> reach_cascade(const std::vector<double>& value, const std::vector<uint32_t>& first,
                                  const std::vector<uint32_t>& tile, int32_t tiles_y, int32_t tiles_x, Max mx) {
  const auto ty = static_cast<size_t>(tiles_y), tx = static_cast<size_t>(tiles_x);
  std::vector<double> acc(value.size(), 0.0);
  DilateScratch s;
  const auto dilate = [&](size_t d) {
    for (size_t y = 0; y < ty; ++y) dilate_axis(acc.data() + y * tx, tx, 1, d, s, mx);
    dilate_axis(acc.data(), ty, tx, d, s, mx);
  };
  size_t prev = 0;
  for (size_t e = first.size() - 2; e > 0; --e) {
    if (first[e] == first[e + 1]) continue;
    if (prev > 0) dilate(prev - e);
    for (uint32_t i = first[e]; i < first[e + 1]; ++i) acc[tile[i]] = mx(acc[tile[i]], value[tile[i]]);
    prev = e;
  }
  if (prev > 0) dilate(prev);
  return acc;
}

}  // namespace

// planner.py: for each radius r of the tiles with value > 0 (clamped to big), the tiles of radius r
// (whatever their value) dilated by a (2r + 1)^2 maximum filter, and the maximum with value. Here one
// cascade from the largest radius down (reach_cascade), O(tiles) per distinct radius whatever its
// size. It is exact: max is order-free, D distributes over max, and D_a D_b = D_{a + b} on the grid
// (between tiles at Chebyshev distance <= a + b, a tile within a of the first and b of the second lies
// between them, so inside the grid). A radius >= big - 1 covers the grid, and values >= +0 make the
// zero padding and the filter's +0 floor change nothing. A NaN value propagates as through np.maximum
// (scipy's filter does not propagate it; the planner's rates are never NaN). Without NaN, np.maximum is
// a < b ? b : a, which vectorizes: about 1 ms on a VGA tile grid with all 80 radii, 0.1 ms with 5.
std::vector<double> dilate_by_reach(const std::vector<double>& value, const std::vector<int32_t>& radius,
                                    int32_t tiles_y, int32_t tiles_x) {
  const int32_t big = std::max(tiles_y, tiles_x);
  const size_t n = value.size();
  if (n == 0) return value;
  std::vector<uint8_t> used(static_cast<size_t>(big) + 1, 0);  // the radii of tiles with value > 0
  for (size_t t = 0; t < n; ++t) {
    if (value[t] > 0) used[static_cast<size_t>(std::clamp(radius[t], 0, big))] = 1;
  }
  // The tiles by dilation radius (0: none), counting sort.
  std::vector<uint32_t> group(n), first(static_cast<size_t>(big) + 1, 0), tile(n);
  bool nan = false;
  for (size_t t = 0; t < n; ++t) {
    const int32_t r = std::clamp(radius[t], 0, big);
    group[t] = (r > 0 && used[static_cast<size_t>(r)] != 0) ? static_cast<uint32_t>(std::min(r, big - 1)) : 0;
    ++first[group[t] + 1];
    nan = nan || value[t] != value[t];
  }
  std::partial_sum(first.begin(), first.end(), first.begin());
  std::vector<uint32_t> next(first.begin(), first.end() - 1);
  for (size_t t = 0; t < n; ++t) tile[next[group[t]]++] = static_cast<uint32_t>(t);
  const std::vector<double> acc =
      nan ? reach_cascade(value, first, tile, tiles_y, tiles_x, [](double a, double b) { return np::maximum(a, b); })
          : reach_cascade(value, first, tile, tiles_y, tiles_x, [](double a, double b) { return a < b ? b : a; });
  std::vector<double> out = value;
  for (size_t t = 0; t < n; ++t) out[t] = np::maximum(out[t], acc[t]);
  return out;
}

namespace {

// P2-H H9 (accuracy.level_policy; F004, DESIGN §13.5): the per-tile plan `out` at the window's level L_w, the largest
// level of its dynamic tiles (the per-tile speed bound and dyadic rounding; l_min when none is dynamic), on every
// active tile (global) or on its dynamic tiles alone, the static ones certified as the per-tile plan certifies them
// (global_certified). v_hat stays the per-tile bound (0 on a static tile); the stats count the new levels.
void apply_level_policy(LevelPolicy policy, int32_t l_min, const std::vector<uint8_t>& active,
                        const std::vector<uint8_t>& band_map, TilePlan& out, PlanStats& stats) {
  int32_t l_w = -1;
  for (const int8_t l : out.level) l_w = std::max(l_w, static_cast<int32_t>(l));
  if (l_w < 0) l_w = l_min;
  stats.level_w = l_w;
  std::fill(stats.levels.begin(), stats.levels.end(), uint64_t{0});
  stats.tiles_static = stats.tiles_dynamic = stats.band_tiles = 0;
  for (size_t t = 0; t < out.level.size(); ++t) {
    const bool st = active[t] == 0 || (policy == LevelPolicy::kGlobalCertified && out.is_static[t] != 0);
    out.is_static[t] = st ? 1 : 0;
    out.level[t] = st ? int8_t{-1} : static_cast<int8_t>(l_w);
    stats.tiles_static += (st && active[t] != 0) ? 1 : 0;
    stats.tiles_dynamic += st ? 0 : 1;
    stats.band_tiles += !st && !band_map.empty() && band_map[t] != 0 ? 1 : 0;
    if (!st) ++stats.levels[static_cast<size_t>(l_w)];
  }
}

}  // namespace

Planner::Planner(const SceneMotion& motion, const CameraGeometry& geometry, uint16_t cam, PlanSchedule schedule,
                 std::optional<Crop> crop)
    : motion_(motion), geo_(geometry), cam_(cam), schedule_(schedule) {
  const size_t tiles = static_cast<size_t>(geo_.tiles_x) * static_cast<size_t>(geo_.tiles_y);
  active_tiles_.assign(tiles, 0);
  crop_per_tile_.assign(tiles, 0);
  crop_ = crop.value_or(Crop{0, 0, geo_.width, geo_.height});
  for (size_t p = 0; p < geo_.n_pixels; ++p) {
    const auto x = static_cast<int32_t>(p % static_cast<size_t>(geo_.width));
    const auto y = static_cast<int32_t>(p / static_cast<size_t>(geo_.width));
    if (crop && !(x >= crop->x0 && x < crop->x1 && y >= crop->y0 && y < crop->y1)) continue;
    active_tiles_[geo_.pixel_tile[p]] = 1;
    ++crop_per_tile_[geo_.pixel_tile[p]];
  }
  // I8's band, region (d), and P2-D D1's R4: the thickness of every body on a trajectory, here at start-up rather than
  // in the window in which it first moves (4-11 ms per 64k-triangle mesh, 10-23 ms for EVIMO2's 80-320k; thickness()
  // computes any other body on first use).
  if (schedule_.band || schedule_.pv) {
    const auto& bodies = motion_.bundle().bodies;
    thickness_.assign(bodies.size(), kNaN);
    for (size_t i = 0; i < bodies.size(); ++i) {
      const MeshDistance* mesh = motion_.mesh(i);
      if (bodies[i].trajectory && mesh != nullptr) thickness_[i] = mesh->thickness(motion_.extent(i));
    }
  }
  // P2-F F9b: the AO window of every body on a trajectory, from the renderer's own fit of its mesh (deterministic, so
  // the d0 its shading takes; a few ms per mesh at start-up).
  if (schedule_.ao) {
    const auto& bodies = motion_.bundle().bodies;
    ao_d0_.assign(bodies.size(), 0.0);
    for (size_t i = 0; i < bodies.size(); ++i) {
      if (bodies[i].trajectory) ao_d0_[i] = render::fit_ao_proxy(bodies[i].mesh.positions, bodies[i].mesh.indices).d0;
    }
  }
}

double Planner::thickness(const BodyWindow& body) {
  const auto i = static_cast<size_t>(body.index);
  if (thickness_.size() <= i) thickness_.resize(motion_.bundle().bodies.size(), kNaN);
  if (std::isnan(thickness_[i])) thickness_[i] = body.mesh != nullptr ? body.mesh->thickness(body.extent) : 0.0;
  return thickness_[i];
}

ShadowLight shadow_light(const LightWindow& l, uint32_t first, uint32_t count) {
  return {l.directional ? 1u : 0u, first, count, l.directional ? -l.position : l.position,
          static_cast<uint32_t>(std::max(l.shadow_bit, 0))};
}

ShadowBody shadow_body(const LightWindow& l, const BodyWindow& body, ConeScratch& scratch) {
  ShadowBody sb{};
  sb.speed = body.point_speed();
  sb.extent = body.extent;
  if (l.directional) {
    sb.mode = ShadowBody::kCylinder;
    sb.centre = body.centre;
    sb.radius = body.swept_radius();
    sb.c_dot_l = dot_seq(body.centre, -l.position);
    sb.c_dot_c = dot_seq(body.centre, body.centre);
    return sb;
  }
  if (body.mesh == nullptr) throw std::logic_error("shadow_body: the body's mesh distance was not built");
  const Vec3d q = body.to_body(l.position);
  const double d_mid = body.mesh->distance(q);
  const double drift = body.point_speed() * body.half_s;
  sb.near = d_mid - drift;  // the light's distance to the body over the window
  sb.mode = ShadowBody::kBeyond;
  if (sb.near <= 0.0) {
    sb.mode = ShadowBody::kAll;
  } else if (const auto cone = body.mesh->cone(q, scratch)) {
    // A point drifting by `drift` turns by at most asin(drift / d) as seen from the light.
    const double half = cone->second + std::asin(drift / d_mid);
    if (!(half >= 0.5 * std::numbers::pi)) {
      sb.mode = ShadowBody::kCone;
      sb.axis = mat_vec(body.rotation, cone->first);
      sb.cos_half = std::cos(half);
    }
  }
  // The band (I8): (d)'s largest light-to-body distance; the cone bound everywhere for a light that may touch the
  // body, and (c) for a spot light where a shadow may be born: the body's cone from the light crosses the outer cone,
  // and the direction of its box centre may lie outside it at t_w (a body partly inside casts a shadow whose
  // boundaries the band sees, and it grows from them). A body outside the outer cone throughout casts no shadow of a
  // spot light: the band skips the pair.
  sb.far = d_mid + drift + 2.0 * body.radius;
  bool born = sb.mode != ShadowBody::kCone;
  if (l.spot && !born) {
    const double alpha = std::acos(clip1(dot_seq(sb.axis, l.axis))), half = std::acos(clip1(sb.cos_half));
    if (alpha - half >= l.outer) sb.keep = ShadowBody::kDead;
    born = alpha + half > l.outer && alpha + std::asin(drift / d_mid) >= l.outer;
  }
  if (sb.mode == ShadowBody::kAll || (l.spot && born)) sb.keep = ShadowBody::kKeep;
  return sb;
}

MoverBody mover_body(const BodyWindow& b, uint32_t id) { return {id, b.centre, b.v, b.omega, b.half_s}; }

MovingLight moving_light(const LightWindow& l, const std::vector<BodyWindow>& every) {
  MovingLight ml{MovingLight::kNone, 0, 0, l.position, l.v, l.omega, l.omega};
  if (l.directional) {
    if (l.casts_shadows) {
      ml.mode = MovingLight::kDirectional;  // occluders up to |X - c_O| + r_O upstream
      ml.count = static_cast<uint32_t>(every.size());
    }
  } else if (l.casts_shadows || l.spot) {
    ml.mode = MovingLight::kPoint;  // kappa: the carrier's turn, or the light's motion relative to an occluder
    if (l.casts_shadows) {
      for (const BodyWindow& body : every) {
        if (body.index == l.body) continue;
        const double d = body.distance(l.position, l.v);
        ml.kappa = py_max(ml.kappa, d > 0 ? (l.v + body.point_speed()) / d : kInf);
      }
    }
  }
  return ml;
}

BodySphere body_sphere(const BodyWindow& b) { return {b.centre, b.swept_radius(), b.point_speed()}; }

void Planner::begin(int64_t t_w) {
  const int64_t t1 = t_w + time::kWindowNs;
  t_w_ = t_w;
  cm_ = motion_.camera_window(cam_, t_w, t1);
  every_ = motion_.body_windows(t_w, t1);
  bodies_.clear();
  for (const BodyWindow& b : every_) {
    if (!b.held) bodies_.push_back(b);
  }
  moving_.clear();
  shadow_.clear();
  for (const LightWindow& l : motion_.light_windows(t_w, t1)) {
    if (l.moving) moving_.push_back(l);
    if (!bodies_.empty() && l.casts_shadows && !l.moving) shadow_.push_back(l);
  }
  stats_ = {};
  stats_.camera_held = cm_.held;
  stats_.camera_v = cm_.v;
  stats_.camera_omega = cm_.omega;
  stats_.moving_bodies = static_cast<uint32_t>(bodies_.size());
  stats_.moving_lights = static_cast<uint32_t>(moving_.size());
  stage_ = {};
  // P2-B B4a: the band N's seeds read the plan pass (K1 forced on with the band; under 1 % of render) in every window
  // where a tile can be dynamic. A held camera with no moving body or light leaves every tile static: no pixel samples.
  stage_.needed = !cm_.held || !shadow_.empty() || !moving_.empty() || (schedule_.seeds && !bodies_.empty());
  // P2-F F9b: a moving body with a proxy has AO receivers, which the plan pass finds (a held one casts a still AO).
  stats_.ao = schedule_.ao;
  for (const BodyWindow& b : bodies_) {
    if (schedule_.ao && ao_d0_[static_cast<size_t>(b.index)] > 0) ++stats_.ao_movers;
  }
  stage_.needed = stage_.needed || stats_.ao_movers > 0;
  stats_.plan_pass = stage_.needed;
  const size_t n = static_cast<size_t>(geo_.tiles_x) * static_cast<size_t>(geo_.tiles_y);
  rate_.assign(n, 0.0);
  dynamic_.assign(n, 0);
}

const PixelStage& Planner::terms() {
  if (!stage_.needed) return stage_;
  PlanParams& p = stage_.params;
  p.omega = cm_.omega;
  p.v = cm_.v;
  p.jmax = geo_.speed_factor_max;
  p.support = geo_.support_px;
  p.big = std::max(geo_.tiles_x, geo_.tiles_y);
  p.lights = !shadow_.empty() || !moving_.empty();
  p.n_shadow = static_cast<uint32_t>(shadow_.size());
  p.n_moving = static_cast<uint32_t>(moving_.size());
  // I8: the band over the static lights' moving shadows (its sources take the blocking body's rigid speed bound).
  p.band = schedule_.band && !shadow_.empty() ? 1 : 0;
  // P2-B B4a: the band N's seeds, with the moving bodies' rigid speed bounds and the moving lights' shadow bits.
  p.seeds = schedule_.seeds ? 1 : 0;
  // P2-B B6c: the per-pixel band's seed motions (PixelPlan::seed_m) at the reach scales.
  p.pixel = schedule_.seeds && schedule_.pixel ? 1 : 0;
  p.sigma_cam = schedule_.sigma_cam;
  p.sigma_mov = schedule_.sigma_mov;
  // P2-D D1: the obliquity factors in the shadow-edge rates (their moving receivers take the movers' speed bounds) and
  // P&V's plan bits.
  p.obliquity = schedule_.obliquity ? 1 : 0;
  p.bits = plan_bits() ? 1 : 0;
  p.pv = schedule_.pv ? 1 : 0;
  p.pitch = schedule_.pitch;
  p.n_movers = p.band != 0 || p.seeds != 0 || p.bits != 0 ? static_cast<uint32_t>(bodies_.size()) : 0;
  // P2-F F9b: the moving bodies' AO reach, 2 d0 (with a relative margin over the renderer's fp32 test) and the swept
  // radius around the mid-window origin, at their point speed bound.
  for (const BodyWindow& b : bodies_) {
    const double d0 = schedule_.ao ? ao_d0_[static_cast<size_t>(b.index)] : 0.0;
    if (d0 > 0) stage_.ao.push_back({b.centre, 2.0 * d0 * (1.0 + 1e-6) + b.swept_radius(), b.point_speed()});
  }
  p.n_ao = static_cast<uint32_t>(stage_.ao.size());
  for (const LightWindow& l : moving_) {
    if (l.shadow_bit >= 0) p.moving_bits |= 1u << static_cast<uint32_t>(l.shadow_bit);
  }
  p.width = geo_.width;
  p.height = geo_.height;
  p.jmin = geo_.speed_factor_min;
  p.crop_x0 = crop_.x0;
  p.crop_y0 = crop_.y0;
  p.crop_x1 = crop_.x1;
  p.crop_y1 = crop_.y1;
  for (const render::Light& l : motion_.bundle().lights) {
    p.n_bits += render::positional(l.kind) && l.casts_shadows ? 1 : 0;
  }
  if (p.lights != 0 || p.seeds != 0 || p.n_ao != 0) std::tie(p.r_wc, p.c_c) = motion_.camera_pose(cam_, t_w_);
  for (const LightWindow& l : shadow_) {
    stage_.shadow_lights.push_back(shadow_light(l, static_cast<uint32_t>(stage_.shadow_bodies.size()),
                                                static_cast<uint32_t>(bodies_.size())));
    for (const BodyWindow& body : bodies_) {
      stage_.shadow_bodies.push_back(shadow_body(l, body, cone_));
      if (p.band != 0 || p.pv != 0) stage_.shadow_bodies.back().extent = thickness(body);
    }
  }
  for (uint32_t m = 0; m < p.n_movers; ++m) {
    stage_.movers.push_back(mover_body(bodies_[m], motion_.bundle().bodies[static_cast<size_t>(bodies_[m].index)].id));
  }
  for (const LightWindow& l : moving_) stage_.moving.push_back(moving_light(l, every_));
  if (!moving_.empty()) {
    for (const BodyWindow& b : every_) stage_.spheres.push_back(body_sphere(b));
  }
  return stage_;
}

// Moving bodies: the tiles their swept sphere may touch, seen from the mid-window camera. A tile is
// touched when acos(tile_ray . u) <= asin(r_eff / dist) + tile_radius + omega W / 2; acos is correctly
// rounded and decreasing, so cosines clearly beyond the cosines of the extreme thresholds decide
// without it (|d acos / dc| >= 1, margin 1e-12 >> the rounding of cos and acos).
void Planner::touches() {
  const size_t n = rate_.size();
  if (bodies_.empty()) return;
  const auto [r_wc, c_c] = motion_.camera_pose(cam_, t_w_ + time::kWindowNs / 2);
  const auto [rad_lo, rad_hi] = std::minmax_element(geo_.tile_radius.begin(), geo_.tile_radius.end());
  for (const BodyWindow& body : bodies_) {
    const double r_eff = body.swept_radius() + cm_.v * kWindowS / 2;
    const Vec3d c = mat_t_vec(r_wc, body.centre - c_c);
    const double dist = norm_seq(c);
    const double range_min = py_max(dist - r_eff, body.distance(c_c, cm_.v));
    const double k_body = range_min > 0 ? cm_.omega + (cm_.v + body.point_speed()) / range_min : kInf;
    const double grow = cm_.omega * kWindowS / 2;
    const bool all = dist <= r_eff;
    const double a = all ? 0.0 : std::asin(r_eff / dist);
    const Vec3d u = scaled(c, dist);
    const double th_lo = (a + *rad_lo) + grow, th_hi = (a + *rad_hi) + grow;
    const double c_in = th_lo <= std::numbers::pi ? std::cos(th_lo) + 1e-12 : kInf;    // above: touched
    const double c_out = th_hi <= std::numbers::pi ? std::cos(th_hi) - 1e-12 : -kInf;  // below: not touched
    for (size_t t = 0; t < n; ++t) {
      bool touched = all;
      if (!touched) {
        const size_t tx = t % static_cast<size_t>(geo_.tiles_x);
        const double cs = clip1(np::gemv3(geo_.tile_ray[t], u, tx, static_cast<size_t>(geo_.tiles_x)));
        if (cs < c_out) continue;
        touched = cs > c_in || std::acos(cs) <= (a + geo_.tile_radius[t]) + grow;
      }
      if (touched) {
        rate_[t] = np::maximum(rate_[t], k_body);
        dynamic_[t] = 1;
      }
    }
  }
}

TilePlan Planner::finish(const TileTotals* tiles) {
  const size_t n = rate_.size();
  // P2-B B4a: touches()' rates, per tile the largest of the moving bodies that may touch it at their nearest range
  // (before the pixel stage's rates join them), for the carried seeds below.
  const std::vector<double> body_rate = schedule_.seeds && stage_.needed ? rate_ : std::vector<double>{};
  // P2-D D1: under the obliquity factors, the tile rates with g = h = 1 (the pixel stage's k_base; D1f: under the band
  // with the band's tile results as built, TileTotals::kcone0, ...), for the levels they would give: the as-built ones.
  std::vector<double> base_rate;
  if (stage_.needed && tiles != nullptr && !tiles->k_base.empty()) base_rate = rate_;
  if (stage_.needed) {
    if (tiles == nullptr) throw std::logic_error("Planner::finish: the pixel stage's tiles are needed");
    // I8's band: the tiles within a band source's reach take their pixels' droppable cone rate, at most the largest
    // rate of the sources that reach them. Every tile with a droppable cone rate stays dynamic, reached or not (the
    // I8 review): the band chooses its rate only, so under a held camera a moving shadow it does not see (one entering
    // the image, hidden in another mover's shadow, or too thin for the centre rays) is sampled at the lattice's
    // coarsest level and found by the signature flags, as under a moving camera, where every tile is dynamic anyway.
    const std::vector<double>* k = &tiles->k;
    const std::vector<uint8_t>* depend = &tiles->depend;
    std::vector<double> band_k;
    std::vector<uint8_t> band_depend;
    uint64_t band_pixels = 0;
    if (stage_.params.band != 0) {
      if (tiles->kcone.size() != n) throw std::logic_error("Planner::finish: the band needs its tile results");
      std::vector<int32_t> radius(n);
      for (size_t t = 0; t < n; ++t) radius[t] = std::max(tiles->sreach[t], 0);
      const std::vector<double> reached = dilate_by_reach(tiles->ksource, radius, geo_.tiles_y, geo_.tiles_x);
      band_k = tiles->k;
      band_depend = tiles->depend;
      for (size_t t = 0; t < n; ++t) {
        if (tiles->kcone[t] > 0) band_depend[t] = 1;
        if (!(reached[t] > 0 && tiles->kcone[t] > 0)) continue;
        band_k[t] = np::maximum(band_k[t], std::min(reached[t], tiles->kcone[t]));
        band_pixels += tiles->cone[t];
      }
      k = &band_k;
      depend = &band_depend;
    }
    const std::vector<double> d = dilate_by_reach(*k, tiles->reach, geo_.tiles_y, geo_.tiles_x);
    for (size_t t = 0; t < n; ++t) rate_[t] = np::maximum(rate_[t], d[t]);
    if (!base_rate.empty()) {
      std::vector<double> kb = tiles->k_base;
      if (stage_.params.band != 0) {  // D1f: the band's additions as built (as band_k's above)
        if (tiles->kcone0.size() != n) throw std::logic_error("Planner::finish: the base needs the band as built");
        std::vector<int32_t> radius(n);
        for (size_t t = 0; t < n; ++t) radius[t] = std::max(tiles->sreach0[t], 0);
        const std::vector<double> reached = dilate_by_reach(tiles->ksource0, radius, geo_.tiles_y, geo_.tiles_x);
        for (size_t t = 0; t < n; ++t) {
          if (reached[t] > 0 && tiles->kcone0[t] > 0) {
            kb[t] = np::maximum(kb[t], std::min(reached[t], tiles->kcone0[t]));
          }
        }
      }
      const std::vector<double> db = dilate_by_reach(kb, tiles->reach, geo_.tiles_y, geo_.tiles_x);
      for (size_t t = 0; t < n; ++t) base_rate[t] = np::maximum(base_rate[t], db[t]);
    }
    if (!cm_.held) {
      dynamic_.assign(n, 1);
    } else if (std::any_of(depend->begin(), depend->end(), [](uint8_t x) { return x != 0; })) {
      const std::vector<double> dep(depend->begin(), depend->end());
      const std::vector<double> dd = dilate_by_reach(dep, tiles->reach, geo_.tiles_y, geo_.tiles_x);
      for (size_t t = 0; t < n; ++t) dynamic_[t] = dynamic_[t] | (dd[t] > 0 ? 1 : 0);
    }
    if (stage_.params.lights != 0) {
      stats_.shadow_pixels = tiles->shadow_pixels + band_pixels;
      stats_.light_pixels = stats_.moving_lights > 0 ? tiles->hit_pixels : 0;
    }
    stats_.ao_pixels = tiles->ao_pixels;  // P2-F F9b
  }
  // P2-B B4a: the band N's tile map, every tile within a seed's reach (without a plan pass no tile is dynamic: none).
  // A carried seed may be a moving body's silhouette that no centre ray hit (a thin or sub-pixel part): its tile
  // reaches at least as far as the bodies that may touch it move (body_rate).
  std::vector<uint8_t> band_map;
  std::vector<double> band_floor;
  if (schedule_.seeds && schedule_.pixel) {
    // P2-B B6c: the per-pixel band (band_map.h, the camera's), whose carried seeds take their tile's floor here: the
    // reach (px) of the moving bodies that may touch the tile, times sigma_mov (0 drops it, also an unbounded one).
    stats_.seeds = true;
    stats_.pixel = true;
    if (stage_.needed) {
      if (tiles->seed.size() != n || tiles->carried.size() != n) {
        throw std::logic_error("Planner::finish: the band N needs its seeds");
      }
      band_floor.assign(n, 0.0);
      const double sigma = schedule_.sigma_mov, jmax = stage_.params.jmax;
      for (size_t t = 0; t < n; ++t) {
        if (sigma > 0 && !(body_rate[t] <= 0)) band_floor[t] = sigma * ((jmax * body_rate[t]) * kWindowS);  // NaN: NaN
      }
      stats_.seed_pixels = tiles->seed_pixels;
      stats_.carried_seeds = tiles->carried_seeds;
    }
  } else if (schedule_.seeds) {
    band_map.assign(n, 0);
    stats_.seeds = true;
    if (stage_.needed) {
      if (tiles->seed.size() != n || tiles->carried.size() != n) {
        throw std::logic_error("Planner::finish: the band N needs its seeds");
      }
      std::vector<double> value(n);
      std::vector<int32_t> radius(n);
      for (size_t t = 0; t < n; ++t) {
        value[t] = tiles->seed[t] >= 0 ? 1.0 : 0.0;
        radius[t] = std::max(tiles->seed[t], 0);
        if (tiles->carried[t] > 0 && !(body_rate[t] <= 0)) {  // NaN: unbounded (reach_tiles: the grid)
          radius[t] = std::max(radius[t], reach_tiles(stage_.params, body_rate[t]));
        }
      }
      const std::vector<double> reached = dilate_by_reach(value, radius, geo_.tiles_y, geo_.tiles_x);
      for (size_t t = 0; t < n; ++t) band_map[t] = reached[t] > 0 ? 1 : 0;
      stats_.seed_pixels = tiles->seed_pixels;
      stats_.carried_seeds = tiles->carried_seeds;
    }
  }
  // Levels: l = clamp(ceil(log2(v_hat W / d_px)), l_min, l_max); STATIC tiles get none.
  TilePlan out;
  out.is_static.resize(n);
  out.level.resize(n);
  out.v_hat.resize(n);
  stats_.levels.assign(static_cast<size_t>(schedule_.l_max) + 1, 0);
  stats_.obliquity = schedule_.obliquity;
  const auto level_of = [&](double v_hat) {
    const double lv = std::ceil(std::log2(v_hat * kWindowS / schedule_.d_px));
    return std::isnan(lv) ? schedule_.l_min
                          : std::clamp(lv, static_cast<double>(schedule_.l_min), static_cast<double>(schedule_.l_max));
  };
  // P2-D D1f: the levels without the factors (base) and the budget's ceiling over the tiles the factors raise: the
  // highest c >= l_min with sum over the dynamic tiles of the crop's pixels x (2^l + 1) <= the budget (level_cap_counts'
  // count), l = max(base, min(level, c)) on the raised tiles; none fits: c = l_min, the levels without the factors.
  // Under P2-H H9's global policies (the merge note, ENGINE_CPP "P2-D D1f notes") the sum is the lattice the policy
  // plans from those capped levels: their window level L_w(c) (apply_level_policy's: the largest, l_min when no tile
  // is dynamic) on every active tile (global) or on the dynamic tiles (global_certified).
  std::vector<double> lv(n), lb(n);
  for (size_t t = 0; t < n; ++t) {
    const bool st = dynamic_[t] == 0 || active_tiles_[t] == 0;
    lv[t] = st ? -1.0 : level_of(geo_.speed_factor[t] * rate_[t]);
    lb[t] = st || base_rate.empty() ? lv[t] : level_of(geo_.speed_factor[t] * base_rate[t]);
  }
  const LevelPolicy policy = schedule_.level_policy;
  const auto capped_level = [&](size_t t, double c) {
    return lv[t] > lb[t] ? std::max(lb[t], std::min(lv[t], c)) : lv[t];
  };
  const auto window_level = [&](const auto& level) {  // L_w of per-tile levels (-1 on static tiles)
    double l_w = -1.0;
    for (size_t t = 0; t < n; ++t) {
      if (lv[t] >= 0) l_w = std::max(l_w, level(t));
    }
    return l_w < 0 ? static_cast<double>(schedule_.l_min) : l_w;
  };
  double ceiling = schedule_.l_max;
  if (!base_rate.empty() && obliquity_budget_ > 0) {
    const auto lattice = [&](double c) {
      double sum = 0;
      if (policy == LevelPolicy::kPerTile) {
        for (size_t t = 0; t < n; ++t) {
          if (lv[t] < 0) continue;
          const double l = capped_level(t, c);
          sum += static_cast<double>(crop_per_tile_[t]) * (std::ldexp(1.0, static_cast<int>(l)) + 1.0);
        }
        return sum;
      }
      const double l_w = window_level([&](size_t t) { return capped_level(t, c); });
      for (size_t t = 0; t < n; ++t) {
        const bool planned = policy == LevelPolicy::kGlobal ? active_tiles_[t] != 0 : lv[t] >= 0;
        if (planned) sum += static_cast<double>(crop_per_tile_[t]) * (std::ldexp(1.0, static_cast<int>(l_w)) + 1.0);
      }
      return sum;
    };
    const auto budget = static_cast<double>(obliquity_budget_);
    if (lattice(ceiling) > budget) {
      // At l_min every raised tile is at its level without the factors (>= l_min): the search stops there.
      while (ceiling > schedule_.l_min && lattice(ceiling) > budget) ceiling -= 1.0;
      stats_.obliquity_ceiling = static_cast<int32_t>(ceiling);
    }
  }
  for (size_t t = 0; t < n; ++t) {
    const bool st = dynamic_[t] == 0 || active_tiles_[t] == 0;
    const double v_hat = st ? 0.0 : geo_.speed_factor[t] * rate_[t];
    double clamped = level_of(v_hat);
    if (!st && !base_rate.empty()) {
      const double base = lb[t];
      if (clamped > base && clamped > ceiling) {
        clamped = std::max(base, ceiling);
        ++stats_.obliquity_tiles_capped;
      }
      stats_.obliquity_tiles_up += clamped > base ? 1 : 0;
      stats_.obliquity_tiles_down += clamped < base ? 1 : 0;
    }
    out.is_static[t] = st ? 1 : 0;
    out.level[t] = st ? int8_t{-1} : static_cast<int8_t>(clamped);
    out.v_hat[t] = v_hat;
    stats_.tiles_static += (st && active_tiles_[t] != 0) ? 1 : 0;
    stats_.tiles_dynamic += st ? 0 : 1;
    stats_.band_tiles += !st && !band_map.empty() && band_map[t] != 0 ? 1 : 0;
    if (!st) ++stats_.levels[static_cast<size_t>(out.level[t])];
    stats_.v_hat_max = t == 0 ? v_hat : np::maximum(stats_.v_hat_max, v_hat);
  }
  if (policy != LevelPolicy::kPerTile) {  // P2-H H9: the per-tile levels at the window's L_w
    apply_level_policy(policy, schedule_.l_min, active_tiles_, band_map, out, stats_);
    if (!base_rate.empty()) {
      // P2-D D1f's counters on the levels the policy plans: per planned tile its level L_w against the policy's level
      // without the factors (L_w of the base levels: up, down) and without the ceiling (L_w of the factored levels:
      // capped).
      const double l_b = window_level([&](size_t t) { return lb[t]; });
      const double l_u = window_level([&](size_t t) { return lv[t]; });
      const auto l_w = static_cast<double>(stats_.level_w);
      stats_.obliquity_tiles_up = stats_.obliquity_tiles_down = stats_.obliquity_tiles_capped = 0;
      for (size_t t = 0; t < n; ++t) {
        if (out.is_static[t] != 0) continue;
        stats_.obliquity_tiles_up += l_w > l_b ? 1 : 0;
        stats_.obliquity_tiles_down += l_w < l_b ? 1 : 0;
        stats_.obliquity_tiles_capped += l_w < l_u ? 1 : 0;
      }
    }
  }
  out.band = std::move(band_map);
  out.band_floor = std::move(band_floor);
  out.stats = stats_;
  return out;
}

TilePlan Planner::plan(int64_t t_w, const std::function<TileTotals(const PixelStage&)>& pixel_stage) {
  begin(t_w);
  const PixelStage& st = terms();
  touches();
  TileTotals tiles;
  if (st.needed) tiles = pixel_stage(st);
  return finish(st.needed ? &tiles : nullptr);
}

}  // namespace balla::engine
