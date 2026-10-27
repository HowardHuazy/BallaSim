// The piecewise fit (fit_piecewise.h; blender.md §4.7.6).
#include "scene/motion/fit_piecewise.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "core/math/so3.h"
#include "core/traj/basis.h"
#include "core/traj/pieces.h"
#include "core/traj/spline_build.h"
#include "scene/rotation.h"
#include "scene/sparse.h"
#include "scene/traj/banded.h"
#include "scene/traj/basis.h"

namespace balla::scene::motion {

namespace {

constexpr int K = kFitOrder;
using rot::M;
using rot::Q;
using rot::V;

double angle(const Q& a, const Q& b) { return norm(so3::log(conjugate(a) * b)); }

M mul(const M& a, const M& b) {
  M c{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) c.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
  }
  return c;
}
M tr(const M& a) {
  M t{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) t.m[i][j] = a.m[j][i];
  }
  return t;
}
M sub(const M& a, const M& b) {
  M c{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) c.m[i][j] = a.m[i][j] - b.m[i][j];
  }
  return c;
}
M scale(const M& a, double s) {
  M c{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) c.m[i][j] = a.m[i][j] * s;
  }
  return c;
}
M identity() {
  M e{};
  for (int i = 0; i < 3; ++i) e.m[i][i] = 1.0;
  return e;
}
V mulv(const M& a, const V& v) {
  return {a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z, a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
          a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z};
}
double comp(const V& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : v.z; }

// Adds w a^T b to the 3 x 3 block (ci, cj) of the upper band (ci <= cj; scalar indices 3 ci + r, 3 cj + c).
void add_block(traj::Band& h, int64_t ci, int64_t cj, const M& a, const M& b, double w) {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      const int64_t i = 3 * ci + r, j = 3 * cj + c;
      if (i > j) continue;
      const double v = a.m[0][r] * b.m[0][c] + a.m[1][r] * b.m[1][c] + a.m[2][r] * b.m[2][c];
      h.at(i, j) += w * v;
    }
  }
}

// segment and cumulative basis of t on the grid [t0, t0 + nseg h] (closed on the right)
struct Loc {
  int64_t s;
  ::balla::traj::CumulativeBasis<double, K> b;
};
Loc locate(int64_t t, int64_t t0, int64_t h, int64_t nseg) {
  ::balla::traj::SegmentTime st{};
  if (!::balla::traj::locate(t, t0, static_cast<int32_t>(h), static_cast<int32_t>(nseg), &st)) {
    throw std::invalid_argument("fit_piece: a sample at " + std::to_string(t) + " ns lies outside the piece's knots");
  }
  return {st.segment, ::balla::traj::cumulative_basis<K>(static_cast<double>(st.r_ns) / static_cast<double>(h))};
}

double mean(const std::vector<double>& v) {
  double s = 0.0;
  for (const double x : v) s += x;
  return v.empty() ? 0.0 : s / static_cast<double>(v.size());
}

}  // namespace

PieceEval::PieceEval(const FitPiece& piece)
    : nodes_(::balla::traj::build_spline_nodes(piece.rot, piece.pos, V{0.0, 0.0, 0.0})), t0_(piece.t0_ns) {
  if (piece.rot.size() < static_cast<size_t>(K) || piece.h_ns < 1 || piece.h_ns > INT32_MAX) {
    throw std::invalid_argument("PieceEval: not a fitted piece");
  }
  const auto h = static_cast<int32_t>(piece.h_ns);
  view_ = {nodes_.data(), static_cast<int32_t>(piece.rot.size()) - K + 1, h, 1e9 / static_cast<double>(h)};
}

Pose PieceEval::operator()(int64_t t_ns) const {
  ::balla::traj::SegmentTime st{};
  if (!::balla::traj::locate(t_ns, t0_, view_.h_ns, view_.num_segments, &st)) {
    throw std::invalid_argument("PieceEval: " + std::to_string(t_ns) + " ns lies outside the piece's support");
  }
  const ::balla::traj::Kinematics<double> k = ::balla::traj::evaluate<K>(view_, st);
  return {k.q, k.p};
}

FitPiece fit_piece(const Track& s, int64_t begin, int64_t end, int64_t h, const std::optional<Pose>& start, bool hold,
                   const FitOptions& opt) {
  if (!(begin < end)) throw std::invalid_argument("fit_piece: empty interval");
  FitPiece pc;
  pc.begin_ns = begin;
  pc.end_ns = end;
  pc.t0_ns = begin;
  pc.hold = hold;
  if (hold) {  // K equal controls: bitwise constant over [begin, begin + nseg 2^30]
    if (!start && s.size() == 0) throw std::invalid_argument("fit_piece: a hold piece needs a pose");
    const Pose p0 = start ? *start : Pose{s.q[0], s.p[0]};
    pc.h_ns = kHoldKnotNs;
    const int64_t nseg = std::max<int64_t>(1, (end - begin + kHoldKnotNs - 1) / kHoldKnotNs);
    pc.rot.assign(static_cast<size_t>(nseg + K - 1), normalized(p0.q));
    pc.pos.assign(static_cast<size_t>(nseg + K - 1), p0.p);
    return pc;
  }
  if (h < 1 || h > INT32_MAX) throw std::invalid_argument("fit_piece: knot spacing outside [1, 2^31)");
  const traj::KnotGrid grid = traj::KnotGrid::covering(begin, end, h, K);
  const int64_t n = grid.num_ctrl, nseg = grid.num_segments();
  pc.h_ns = h;

  // the data: the samples, and a C0 start's pseudo-sample at begin (weight join_weight x the largest)
  std::vector<int64_t> t(s.t_ns.begin(), s.t_ns.end());
  std::vector<Q> q(s.q.begin(), s.q.end());
  std::vector<V> p(s.p.begin(), s.p.end());
  std::vector<double> wp(s.size()), wr(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    wp[i] = s.w_pos[i] * s.w_pos[i];
    wr[i] = s.w_rot[i] * s.w_rot[i];
  }
  // a part the error model does not see (w = 0: a point carrier's rotation, a directional light's position) is still
  // fitted, with unit weights, so the trajectory stays well-posed
  const double wp_max0 = wp.empty() ? 0.0 : *std::max_element(wp.begin(), wp.end());
  const double wr_max0 = wr.empty() ? 0.0 : *std::max_element(wr.begin(), wr.end());
  if (!(wp_max0 > 0.0)) std::fill(wp.begin(), wp.end(), 1.0);
  if (!(wr_max0 > 0.0)) std::fill(wr.begin(), wr.end(), 1.0);
  const double wp_max = wp.empty() ? 1.0 : *std::max_element(wp.begin(), wp.end());
  const double wr_max = wr.empty() ? 1.0 : *std::max_element(wr.begin(), wr.end());
  if (start) {
    t.insert(t.begin(), begin);
    q.insert(q.begin(), start->q);
    p.insert(p.begin(), start->p);
    wp.insert(wp.begin(), opt.join_weight * wp_max);
    wr.insert(wr.begin(), opt.join_weight * wr_max);
  }
  if (t.empty()) {
    throw std::invalid_argument("fit_piece: no samples in [" + std::to_string(begin) + ", " + std::to_string(end) + "]");
  }
  std::vector<Loc> loc;
  loc.reserve(t.size());
  for (const int64_t ti : t) loc.push_back(locate(ti, begin, h, nseg));
  const double lam_p = opt.lambda_rel * mean(wp) * static_cast<double>(t.size()) / static_cast<double>(n);
  const double lam_r = opt.lambda_rel * mean(wr) * static_cast<double>(t.size()) / static_cast<double>(n);

  // positions: (B^T W B + lam P) x = B^T W y, one band Cholesky for the three coordinates
  {
    traj::Band a(n, K - 1);
    std::vector<double> rhs(static_cast<size_t>(3 * n), 0.0);
    for (size_t i = 0; i < t.size(); ++i) {
      const auto& lb = loc[i].b.lambda;
      double bv[K];
      bv[0] = 1.0 - lb[0];
      for (int j = 1; j < K - 1; ++j) bv[j] = lb[j - 1] - lb[j];
      bv[K - 1] = lb[K - 2];
      const int64_t s0 = loc[i].s;
      for (int x = 0; x < K; ++x) {
        for (int y = x; y < K; ++y) a.at(s0 + x, s0 + y) += wp[i] * bv[x] * bv[y];
        for (int c = 0; c < 3; ++c) rhs[static_cast<size_t>(3 * (s0 + x) + c)] += wp[i] * bv[x] * comp(p[i], c);
      }
    }
    const traj::Band pen = traj::to_upper_band(traj::roughness_matrix(n, K, K / 2), K - 1);
    for (size_t k = 0; k < a.ab.size(); ++k) a.ab[k] += lam_p * pen.ab[k];
    const std::vector<double> x = traj::BandedSPD(a).solve(rhs, 3);
    pc.pos.resize(static_cast<size_t>(n));
    for (int64_t c = 0; c < n; ++c) {
      pc.pos[static_cast<size_t>(c)] = {x[static_cast<size_t>(3 * c)], x[static_cast<size_t>(3 * c + 1)],
                                        x[static_cast<size_t>(3 * c + 2)]};
    }
  }

  // rotations: Gauss-Newton with the jerk penalty on the increments, e = D_2 d weighted by the segment Gram
  {
    std::vector<int64_t> ts(s.t_ns.begin(), s.t_ns.end());
    std::vector<Q> qs(s.q.begin(), s.q.end());
    if (ts.empty()) {
      ts.push_back(begin);
      qs.push_back(start->q);
    }
    pc.rot = rot::slerp_init(ts, qs, grid.greville_ns());
    const sparse::Matrix gram = traj::difference_gram(n, K, K / 2);  // (n - 3) x (n - 3), CSR
    constexpr double kCoef[3] = {1.0, -2.0, 1.0};
    for (int it = 0; it < opt.gn_iters; ++it) {
      std::vector<V> d(static_cast<size_t>(n));
      std::vector<M> jinv(static_cast<size_t>(n));
      for (int64_t c = 1; c < n; ++c) {
        d[static_cast<size_t>(c)] = so3::log(conjugate(pc.rot[static_cast<size_t>(c - 1)]) * pc.rot[static_cast<size_t>(c)]);
        jinv[static_cast<size_t>(c)] = rot::right_jacobian_inv(d[static_cast<size_t>(c)]);
      }
      traj::Band hb(3 * n, 3 * K - 1);
      std::vector<double> g(static_cast<size_t>(3 * n), 0.0);
      for (size_t i = 0; i < t.size(); ++i) {
        const int64_t s0 = loc[i].s;
        M a[K], m[K], qq[K + 1], blk[K];
        Q r_t = pc.rot[static_cast<size_t>(s0)];
        for (int j = 1; j < K; ++j) {
          const double l = loc[i].b.lambda[j - 1];
          const V ld = l * d[static_cast<size_t>(s0 + j)];
          const Q e = so3::exp(ld);
          a[j] = rot::quat_to_matrix(e);
          m[j] = scale(rot::right_jacobian(ld), l);
          r_t = r_t * e;
        }
        const V r = so3::log(conjugate(q[i]) * r_t);
        const M jr = rot::right_jacobian_inv(r);
        qq[K] = identity();
        for (int j = K - 1; j >= 1; --j) qq[j] = mul(a[j], qq[j + 1]);
        blk[0] = sub(tr(qq[1]), mul(mul(tr(qq[2]), m[1]), tr(jinv[static_cast<size_t>(s0 + 1)])));
        for (int j = 1; j < K - 1; ++j) {
          blk[j] = sub(mul(mul(tr(qq[j + 1]), m[j]), jinv[static_cast<size_t>(s0 + j)]),
                       mul(mul(tr(qq[j + 2]), m[j + 1]), tr(jinv[static_cast<size_t>(s0 + j + 1)])));
        }
        blk[K - 1] = mul(m[K - 1], jinv[static_cast<size_t>(s0 + K - 1)]);
        for (int j = 0; j < K; ++j) blk[j] = mul(jr, blk[j]);
        for (int x = 0; x < K; ++x) {
          for (int y = x; y < K; ++y) add_block(hb, s0 + x, s0 + y, blk[x], blk[y], wr[i]);
          const V gx = mulv(tr(blk[x]), r);
          for (int c = 0; c < 3; ++c) g[static_cast<size_t>(3 * (s0 + x) + c)] += wr[i] * comp(gx, c);
        }
      }
      // penalty: e_k = d_{k+1} - 2 d_{k+2} + d_{k+3}; d e_k / d eps_{k+j} = E_k[j] (j = 0..3) from
      // d d_c = J_r^-1(d_c) eps_c - J_r^-1(d_c)^T eps_{c-1}
      const int64_t ne = n - 3;
      std::vector<V> e(static_cast<size_t>(ne));
      std::vector<std::array<M, 4>> ek(static_cast<size_t>(ne));
      for (int64_t k = 0; k < ne; ++k) {
        V acc{0, 0, 0};
        std::array<M, 4> blocks{};  // zero
        for (int mm = 0; mm < 3; ++mm) {
          const size_t c = static_cast<size_t>(k + 1 + mm);
          acc = acc + kCoef[mm] * d[c];
          blocks[static_cast<size_t>(mm + 1)] = sub(blocks[static_cast<size_t>(mm + 1)], scale(jinv[c], -kCoef[mm]));
          blocks[static_cast<size_t>(mm)] = sub(blocks[static_cast<size_t>(mm)], scale(tr(jinv[c]), kCoef[mm]));
        }
        e[static_cast<size_t>(k)] = acc;
        ek[static_cast<size_t>(k)] = blocks;
      }
      for (int64_t row = 0; row < gram.rows; ++row) {
        for (int64_t z = gram.indptr[static_cast<size_t>(row)]; z < gram.indptr[static_cast<size_t>(row + 1)]; ++z) {
          const int64_t col = gram.indices[static_cast<size_t>(z)];
          const double gv = lam_r * gram.data[static_cast<size_t>(z)];
          const auto& er = ek[static_cast<size_t>(row)];
          const auto& ec = ek[static_cast<size_t>(col)];
          for (int x = 0; x < 4; ++x) {
            const V gx = mulv(tr(er[static_cast<size_t>(x)]), e[static_cast<size_t>(col)]);
            for (int c = 0; c < 3; ++c) g[static_cast<size_t>(3 * (row + x) + c)] += gv * comp(gx, c);
            for (int y = 0; y < 4; ++y) {
              if (row + x <= col + y) add_block(hb, row + x, col + y, er[static_cast<size_t>(x)], ec[static_cast<size_t>(y)], gv);
            }
          }
        }
      }
      for (double& v : g) v = -v;
      const std::vector<double> step = traj::BandedSPD(hb).solve(g, 1);
      double big = 0.0;
      for (int64_t c = 0; c < n; ++c) {
        const V dc{step[static_cast<size_t>(3 * c)], step[static_cast<size_t>(3 * c + 1)],
                   step[static_cast<size_t>(3 * c + 2)]};
        big = std::max({big, std::fabs(dc.x), std::fabs(dc.y), std::fabs(dc.z)});
        pc.rot[static_cast<size_t>(c)] = normalized(pc.rot[static_cast<size_t>(c)] * so3::exp(dc));
      }
      rot::align_hemisphere(pc.rot);
      pc.gn_iters = it + 1;
      if (big < 1e-13) break;
    }
  }

  // a C0 start: move the piece rigidly onto the start pose (a constant shift and a constant left rotation keep the
  // spline's shape and its increments), so the join keeps the pose to rounding (core/traj/pieces.h kJoinTol*)
  if (start) {
    const Pose at = PieceEval(pc)(begin);
    const V dp = start->p - at.p;
    const Q dq = normalized(start->q * conjugate(at.q));
    pc.join_px = std::sqrt(wp_max) * norm(dp) + std::sqrt(wr_max) * norm(so3::log(dq));
    for (V& c : pc.pos) c = c + dp;
    for (Q& c : pc.rot) c = normalized(dq * c);
    rot::align_hemisphere(pc.rot);
    const Pose moved = PieceEval(pc)(begin);
    // the join holds to rounding: 1e-3 of the engine's tolerance plus the fp64 rounding of world positions (an action
    // far from Blender's origin, MOT-FAR, has ulp(1e5 m) = 1.5e-11 m > 1e-11; the engine's check runs on positions
    // relative to origin_world, where that rounding is the whole difference)
    const double ulps = 16.0 * std::numeric_limits<double>::epsilon() * std::max(norm(moved.p), norm(start->p));
    if (!(norm(moved.p - start->p) <= 1e-3 * ::balla::traj::kJoinTolM + ulps &&
          angle(moved.q, start->q) <= 1e-3 * ::balla::traj::kJoinTolRad)) {
      throw std::logic_error("fit_piece: the rigid move left the join apart");
    }
  }
  return pc;
}

std::vector<double> piece_errors(const FitPiece& piece, const Track& s) {
  const PieceEval eval(piece);
  std::vector<double> e(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    const Pose f = eval(s.t_ns[i]);
    e[i] = s.w_pos[i] * norm(f.p - s.p[i]) + s.w_rot[i] * angle(s.q[i], f.q);
  }
  return e;
}

double max_increment_rad(const FitPiece& piece) {
  double m = 0.0;
  for (size_t c = 1; c < piece.rot.size(); ++c) m = std::max(m, angle(piece.rot[c - 1], piece.rot[c]));
  return m;
}

std::vector<std::pair<int64_t, int64_t>> find_holds(const Track& tr_, size_t min_run) {
  std::vector<std::pair<int64_t, int64_t>> out;
  auto same = [&](size_t a, size_t b) {
    return std::memcmp(&tr_.q[a], &tr_.q[b], sizeof(Q)) == 0 && std::memcmp(&tr_.p[a], &tr_.p[b], sizeof(V)) == 0;
  };
  size_t i = 0;
  while (i < tr_.size()) {
    size_t j = i + 1;
    while (j < tr_.size() && same(i, j)) ++j;
    if (j - i >= std::max<size_t>(min_run, 2)) out.emplace_back(tr_.t_ns[i], tr_.t_ns[j - 1]);
    i = j;
  }
  return out;
}

std::vector<PiecePlan> plan_pieces(int64_t t0, int64_t t1, std::span<const Breakpoint> breakpoints,
                                   std::span<const std::pair<int64_t, int64_t>> holds, std::span<const int64_t> times) {
  if (!(t0 < t1)) throw std::invalid_argument("plan_pieces: empty span");
  std::vector<Breakpoint> bps;
  for (const Breakpoint& b : breakpoints) {
    if (b.t_ns > t0 && b.t_ns < t1) bps.push_back(b);
  }
  std::sort(bps.begin(), bps.end(), [](const Breakpoint& a, const Breakpoint& b) { return a.t_ns < b.t_ns; });
  // hold pieces [hb, he)
  std::vector<std::pair<int64_t, int64_t>> hp;
  for (const auto& [a, b] : holds) {
    const auto ia = std::lower_bound(times.begin(), times.end(), a);
    const auto ib = std::upper_bound(times.begin(), times.end(), b);
    const int64_t prev = ia == times.begin() ? t0 - 1 : *(ia - 1);
    const int64_t next = ib == times.end() ? t1 + 1 : *ib;
    // a run from the first sample holds from the span's start, one to the last sample to its end (the realized times
    // of the span's ends may lie a few ns inside it)
    int64_t hb = a <= t0 || ia == times.begin() ? t0 : a, he = b >= t1 || ib == times.end() ? t1 : b;
    for (const Breakpoint& bp : bps) {
      if (bp.t_ns > prev && bp.t_ns <= a) hb = bp.t_ns;  // the last one in (prev, a]
    }
    for (auto it = bps.rbegin(); it != bps.rend(); ++it) {
      if (it->t_ns >= b && it->t_ns <= next) he = it->t_ns;  // the first one in [b, next]: a CONSTANT key's step is
    }                                                        // its own sample's time, after the pre-key sample
    if (hb < he && (hp.empty() || hb >= hp.back().second)) hp.emplace_back(hb, he);
  }
  // boundaries: the holds' ends and the breakpoints outside every hold piece
  std::vector<std::pair<int64_t, Continuity>> cuts;
  auto inside_hold = [&](int64_t t) {
    return std::any_of(hp.begin(), hp.end(), [t](const auto& h) { return t > h.first && t < h.second; });
  };
  for (const Breakpoint& b : bps) {
    if (!inside_hold(b.t_ns)) cuts.emplace_back(b.t_ns, b.cls);
  }
  for (const auto& [hb, he] : hp) {
    for (const int64_t c : {hb, he}) {
      if (c > t0 && c < t1) cuts.emplace_back(c, Continuity::kC0);
    }
  }
  // one cut per time: a breakpoint's class wins over a hold end's C0 (the lowest class)
  std::sort(cuts.begin(), cuts.end(), [](const auto& a, const auto& b) {
    return a.first != b.first ? a.first < b.first : static_cast<int>(a.second) < static_cast<int>(b.second);
  });
  cuts.erase(std::unique(cuts.begin(), cuts.end(), [](const auto& a, const auto& b) { return a.first == b.first; }),
             cuts.end());
  std::vector<PiecePlan> out;
  int64_t begin = t0;
  Continuity join = Continuity::kC0;
  auto push = [&](int64_t end) {
    const bool hold = std::any_of(hp.begin(), hp.end(), [&](const auto& h) { return h.first == begin && h.second == end; });
    out.push_back({begin, end, out.empty() ? Continuity::kC0 : join, hold});
  };
  for (const auto& [c, cls] : cuts) {
    push(c);
    begin = c;
    join = cls;
  }
  push(t1);
  return out;
}

int64_t knot_spacing(std::span<const int64_t> times, int64_t begin, int64_t end, int64_t h_floor, int64_t h_cap) {
  const int64_t len = end - begin;
  if (len < 1) throw std::invalid_argument("knot_spacing: empty piece");
  // the median spacing of the samples inside the piece's middle 80 % (the prototype's rule: refinement near the ends
  // would otherwise pull it down)
  const int64_t lo = begin + len / 10, hi = end - len / 10;
  std::vector<int64_t> in;
  for (const int64_t t : times) {
    if (t > lo && t < hi) in.push_back(t);
  }
  double med = static_cast<double>(len) / 2.0;
  if (in.size() >= 3) {
    std::vector<int64_t> dt;
    for (size_t i = 1; i < in.size(); ++i) dt.push_back(in[i] - in[i - 1]);
    std::sort(dt.begin(), dt.end());
    const size_t m = dt.size() / 2;
    med = dt.size() % 2 ? static_cast<double>(dt[m]) : 0.5 * static_cast<double>(dt[m - 1] + dt[m]);
  }
  const double h = std::min({std::max(2.0 * med, static_cast<double>(h_floor)), static_cast<double>(h_cap),
                             static_cast<double>(len)});
  return std::max<int64_t>(1, static_cast<int64_t>(h));
}

render::Trajectory to_trajectory(const PiecewiseFit& fit, const std::array<double, 3>& o) {
  if (fit.pieces.empty()) throw std::invalid_argument("to_trajectory: no pieces");
  render::Trajectory tr_;
  tr_.order = K;
  tr_.t0_ns = fit.pieces[0].t0_ns;
  tr_.h_ns = fit.pieces[0].h_ns;
  tr_.span_begin_ns = fit.span_begin_ns;
  tr_.span_end_ns = fit.span_end_ns;
  for (const FitPiece& pc : fit.pieces) {
    const auto c0 = static_cast<int32_t>(tr_.rot.size() / 4);
    for (size_t c = 0; c < pc.rot.size(); ++c) {
      const Q qn = normalized(pc.rot[c]);
      tr_.rot.insert(tr_.rot.end(), {qn.w, qn.x, qn.y, qn.z});
      tr_.pos.insert(tr_.pos.end(), {pc.pos[c].x - o[0], pc.pos[c].y - o[1], pc.pos[c].z - o[2]});
    }
    if (fit.pieces.size() > 1) {
      tr_.pieces.push_back({pc.begin_ns, pc.t0_ns, pc.h_ns, c0, static_cast<int32_t>(tr_.rot.size() / 4),
                            static_cast<int8_t>(&pc == &fit.pieces[0] ? 0 : static_cast<int>(pc.join))});
    }
  }
  return tr_;
}

}  // namespace balla::scene::motion
