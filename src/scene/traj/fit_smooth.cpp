#include "scene/traj/fit_smooth.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/traj/spline.h"
#include "core/traj/spline_build.h"
#include "scene/blas.h"
#include "scene/errors.h"
#include "scene/npo.h"
#include "scene/sparse.h"
#include "scene/traj/banded.h"
#include "scene/traj/basis.h"

namespace balla::scene::traj {

namespace {

using rot::M;
using rot::Q;
using rot::V;
namespace sp = sparse;

double py_max(double a, double b) { return b > a ? b : a; }
using npo::maximum;
using npo::max;
using npo::min;
double sq(double x) { return x * x; }
// np.sum(x ** 2, axis=1) of an (n, 3) row.
double rowsum_sq(V x) { return 0.0 + (((0.0 + x.x * x.x) + x.y * x.y) + x.z * x.z); }

// ballasim.traj.evaluate.evaluate_controls through the core's fp64 spline (what _core.spline_eval_f64 runs).
struct Kin {
  std::vector<Q> q;
  std::vector<V> p, omega, v;
};

Kin eval_controls(int k, int64_t t0, int64_t h, const std::vector<Q>& rot, const std::vector<V>& pos,
                  std::span<const int64_t> t) {
  if (k != 4 && k != 6) throw std::invalid_argument("spline order must be 4 or 6");
  if (h <= 0 || h > INT32_MAX) throw std::invalid_argument("h_ns must be in [1, 2^31)");
  if (rot.size() != pos.size()) throw std::invalid_argument("q and p need the same number of rows");
  if (rot.size() < static_cast<size_t>(k)) throw std::invalid_argument("need at least `order` control points");
  const std::vector<::balla::traj::SplineNode<double>> nodes =
      ::balla::traj::build_spline_nodes(rot, pos, V{0.0, 0.0, 0.0});
  const auto h32 = static_cast<int32_t>(h);
  const int32_t nseg = static_cast<int32_t>(rot.size()) - k + 1;
  const ::balla::traj::SplineView<double> view{nodes.data(), nseg, h32, 1e9 / h32};
  Kin out;
  out.q.resize(t.size());
  out.p.resize(t.size());
  out.omega.resize(t.size());
  out.v.resize(t.size());
  for (size_t i = 0; i < t.size(); ++i) {
    ::balla::traj::SegmentTime st;
    if (!::balla::traj::locate(t[i], t0, h32, nseg, &st)) {
      throw std::invalid_argument("query time " + std::to_string(t[i]) + " ns is outside the spline span");
    }
    const ::balla::traj::Kinematics<double> kin =
        k == 4 ? ::balla::traj::evaluate<4>(view, st) : ::balla::traj::evaluate<6>(view, st);
    out.q[i] = kin.q;
    out.p[i] = kin.p;
    out.omega[i] = kin.omega;
    out.v[i] = kin.v;
  }
  return out;
}

M transposed(const M& a) {
  M t;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) t.m[r][c] = a.m[c][r];
  }
  return t;
}

// One np.matmul of two 3 x 3 blocks; ta / tb: the operand is a swapaxes view of the stored block.
M mm(const M& a, bool ta, const M& b, bool tb) {
  M c;
  blas::matmul3(&a.m[0][0], ta, &b.m[0][0], tb, &c.m[0][0]);
  return c;
}

// Control points on a knot grid with tied groups, plus the fitter's linearizations (fit._Spline, no fixed controls).
class Spline {
 public:
  Spline(const KnotGrid& g, std::vector<Q> rot0, std::vector<V> pos0, TieMap t)
      : grid(g), k(g.order), m(g.order / 2), n(g.num_ctrl), tie(std::move(t)), rot(std::move(rot0)),
        pos(std::move(pos0)) {
    rot::align_hemisphere(rot);
    project();
    g1 = tie.matrix(1);
    g3 = tie.matrix(3);
    penalty_pos = roughness_matrix(n, k, m);
    d_rot = sp::kron_identity3(difference_matrix(n - 1, m - 1));
    gamma3 = sp::kron_identity3(difference_gram(n, k, m));
    gamma3_csc = sp::to_csc(gamma3);
  }

  void project() {
    std::vector<int64_t> first(static_cast<size_t>(tie.num_free), 0);
    for (int64_t c = n - 1; c >= 0; --c) first[static_cast<size_t>(tie.free_index[static_cast<size_t>(c)])] = c;
    std::vector<Q> r(static_cast<size_t>(n));
    std::vector<V> p(static_cast<size_t>(n));
    for (int64_t c = 0; c < n; ++c) {
      const auto src = static_cast<size_t>(first[static_cast<size_t>(tie.free_index[static_cast<size_t>(c)])]);
      r[static_cast<size_t>(c)] = rot[src];
      p[static_cast<size_t>(c)] = pos[src];
    }
    rot = std::move(r);
    pos = std::move(p);
  }

  Kin evaluate(std::span<const int64_t> t) const { return eval_controls(k, grid.t0_ns, grid.h_ns, rot, pos, t); }

  std::vector<V> increments() const {
    std::vector<V> d(static_cast<size_t>(n - 1));
    for (size_t c = 1; c < rot.size(); ++c) d[c - 1] = rot::relative_rotvec(rot[c - 1], rot[c]);
    return d;
  }

  static std::vector<double> ravel(const std::vector<V>& v) {
    std::vector<double> out(3 * v.size());
    for (size_t i = 0; i < v.size(); ++i) {
      out[3 * i] = v[i].x;
      out[3 * i + 1] = v[i].y;
      out[3 * i + 2] = v[i].z;
    }
    return out;
  }

  double rotation_penalty_value() const {
    const std::vector<double> e = sp::matvec(d_rot, ravel(increments()));
    const std::vector<double> ge = sp::matvec(gamma3, e);
    return blas::dot(static_cast<int>(e.size()), e.data(), ge.data());
  }

  // The increments d_c = Log(R_{c-1}^T R_c) (c = 1..n-1, at index c - 1) and J_r^-1(d_c): what rotation_penalty and
  // rotation_jacobian both take from the current controls (pure functions of them, computed once).
  struct Increments {
    std::vector<V> d;
    std::vector<M> jri;
  };

  Increments linearization() const {
    Increments inc{increments(), {}};
    inc.jri.resize(inc.d.size());
    for (size_t i = 0; i < inc.d.size(); ++i) inc.jri[i] = rot::right_jacobian_inv(inc.d[i]);
    return inc;
  }

  // rotation_penalty(): e = D_{m-1} d and de0 = D_{m-1} lin, lin = d d / d eps (rows 3 (c-1) + a: -J_r^-1(d_c)^T at
  // control c - 1, J_r^-1(d_c) at control c). de0 as SciPy's d_rot @ lin stores it: each row's columns descending (its
  // linked list over first touches, which ascend), exact zeros dropped.
  std::pair<std::vector<double>, sp::Matrix> rotation_penalty(const Increments& inc) const {
    std::vector<double> e = sp::matvec(d_rot, ravel(inc.d));
    const int64_t rows = d_rot.rows, w = 3 * (m + 1);   // m difference terms span m + 1 controls
    std::vector<double> coef(static_cast<size_t>(m));
    for (int64_t j = 0; j < m; ++j) coef[static_cast<size_t>(j)] = d_rot.data[static_cast<size_t>(j)];
    sp::Matrix de0{sp::Format::kCsr, rows, 3 * n, std::vector<int64_t>(static_cast<size_t>(rows + 1), 0), {}, {}};
    de0.indices.reserve(static_cast<size_t>(rows * w));
    de0.data.reserve(static_cast<size_t>(rows * w));
    std::vector<double> sums(static_cast<size_t>(w));
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t kk = row / 3, a = row % 3;
      std::fill(sums.begin(), sums.end(), 0.0);
      for (int64_t dl = 0; dl < m; ++dl) {
        const double cv = coef[static_cast<size_t>(dl)];
        const M& jri = inc.jri[static_cast<size_t>(kk + dl)];
        for (int b = 0; b < 3; ++b) sums[static_cast<size_t>(3 * dl + b)] += cv * -jri.m[b][a];
        for (int b = 0; b < 3; ++b) sums[static_cast<size_t>(3 * (dl + 1) + b)] += cv * jri.m[a][b];
      }
      for (int64_t c = w - 1; c >= 0; --c) {
        if (sums[static_cast<size_t>(c)] != 0) {
          de0.indices.push_back(3 * kk + c);
          de0.data.push_back(sums[static_cast<size_t>(c)]);
        }
      }
      de0.indptr[static_cast<size_t>(row + 1)] = static_cast<int64_t>(de0.data.size());
    }
    return {std::move(e), std::move(de0)};
  }

  // d r_n / d eps_c for r_n = Log(R~_n^T R(t_n)) and R_c <- R_c Exp(eps_c) (3 n_t x 3 N), as the canonical CSR
  // sp.csr_matrix((blk, (rows, cols))) builds (each row's 3 K columns ascending).
  sp::Matrix rotation_jacobian(std::span<const int64_t> t, const std::vector<V>& r_res, const Increments& inc) const {
    std::vector<int64_t> s;
    std::vector<double> u;
    grid.locate(t, s, u);
    const size_t nt = t.size(), km1 = static_cast<size_t>(k - 1), kk = static_cast<size_t>(k);
    const std::vector<double> lam = cumulative_basis(u, k);
    sp::Matrix out{sp::Format::kCsr, 3 * static_cast<int64_t>(nt), 3 * n, std::vector<int64_t>(3 * nt + 1), {}, {}};
    out.indices.resize(nt * kk * 9);
    out.data.resize(nt * kk * 9);
    for (size_t r = 0; r <= 3 * nt; ++r) out.indptr[r] = static_cast<int64_t>(r * 3 * kk);
    std::vector<M> a(km1), q(kk), mj(km1), blk(kk);
    M eye{};
    for (int i = 0; i < 3; ++i) eye.m[i][i] = 1.0;
    for (size_t i = 0; i < nt; ++i) {
      const auto si = static_cast<size_t>(s[i]);
      for (size_t j = 0; j < km1; ++j) {
        const V& d = inc.d[si + j];
        const double l = lam[i * km1 + j];
        const V ld{l * d.x, l * d.y, l * d.z};
        a[j] = rot::quat_to_matrix(rot::so3_exp(ld));
        const M jr = rot::right_jacobian(ld);
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c) mj[j].m[r][c] = l * jr.m[r][c];
        }
      }
      const M* jri = &inc.jri[si];
      q[kk - 1] = eye;
      for (size_t j = kk - 1; j-- > 0;) q[j] = mm(a[j], false, q[j + 1], false);
      const M x0 = mm(mm(q[1], true, mj[0], false), false, jri[0], true);
      const M qt0 = transposed(q[0]);
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) blk[0].m[r][c] = qt0.m[r][c] - x0.m[r][c];
      }
      for (size_t j = 1; j + 1 < kk; ++j) {
        const M left = mm(mm(q[j], true, mj[j - 1], false), false, jri[j - 1], false);
        const M right = mm(mm(q[j + 1], true, mj[j], false), false, jri[j], true);
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c) blk[j].m[r][c] = left.m[r][c] - right.m[r][c];
        }
      }
      blk[kk - 1] = mm(mj[kk - 2], false, jri[kk - 2], false);
      const M jr_res = rot::right_jacobian_inv(r_res[i]);
      for (size_t j = 0; j < kk; ++j) {
        const M b = mm(jr_res, false, blk[j], false);
        for (size_t r = 0; r < 3; ++r) {
          for (size_t c = 0; c < 3; ++c) {
            const size_t e = (3 * i + r) * 3 * kk + 3 * j + c;
            out.indices[e] = 3 * (s[i] + static_cast<int64_t>(j)) + static_cast<int64_t>(c);
            out.data[e] = b.m[r][c];
          }
        }
      }
    }
    return out;
  }

  // a @ g3 (or g1): with no tied groups the tie matrix is the identity and SciPy's product only lists each row's
  // entries in reverse (its linked list) and drops exact zeros; with ties, the general product.
  sp::Matrix times_tie(const sp::Matrix& a, const sp::Matrix& g) const {
    if (!tie.groups.empty()) return sp::matmul(a, g);
    sp::Matrix out{sp::Format::kCsr, a.rows, g.cols, std::vector<int64_t>(static_cast<size_t>(a.rows + 1), 0), {}, {}};
    out.indices.reserve(a.indices.size());
    out.data.reserve(a.data.size());
    for (int64_t r = 0; r < a.rows; ++r) {
      for (int64_t jj = a.indptr[static_cast<size_t>(r + 1)] - 1; jj >= a.indptr[static_cast<size_t>(r)]; --jj) {
        const double v = 0.0 + a.data[static_cast<size_t>(jj)] * 1.0;
        if (v != 0) {
          out.indices.push_back(a.indices[static_cast<size_t>(jj)]);
          out.data.push_back(v);
        }
      }
      out.indptr[static_cast<size_t>(r + 1)] = static_cast<int64_t>(out.data.size());
    }
    return out;
  }

  void apply_rotation_step(const std::vector<double>& delta_var) {
    const std::vector<double> full = sp::matvec(g3, delta_var);
    for (size_t c = 0; c < rot.size(); ++c) {
      const Q e = rot::so3_exp(V{full[3 * c], full[3 * c + 1], full[3 * c + 2]});
      rot[c] = rot::canonical_quat(rot::quat_multiply(rot[c], e));
    }
    rot::align_hemisphere(rot);
    project();
  }

  void set_positions(const std::vector<double>& x_var) {
    const std::vector<double> full = sp::matvecs(g1, x_var, 3);
    for (size_t c = 0; c < pos.size(); ++c) pos[c] = {full[3 * c], full[3 * c + 1], full[3 * c + 2]};
    project();
  }

  KnotGrid grid;
  int k, m;
  int64_t n;
  TieMap tie;
  std::vector<Q> rot;
  std::vector<V> pos;
  sp::Matrix g1, g3, penalty_pos, d_rot, gamma3, gamma3_csc;
};

// _solve_and_trace: A_l x_l = rhs_l and tr(A_l^-1 M), one band Cholesky each and one batched selected inversion; the
// matrices as their upper bands (what to_upper_band gives the Python).
std::pair<std::vector<std::vector<double>>, std::vector<double>> solve_and_trace(
    const std::vector<Band>& mats, const std::vector<std::vector<double>>& rhs, int64_t nrhs, const Band& m) {
  std::vector<BandedSPD> chol;
  chol.reserve(mats.size());
  for (const Band& a : mats) chol.emplace_back(a);
  std::vector<std::vector<double>> sols;
  for (size_t i = 0; i < chol.size(); ++i) sols.push_back(chol[i].solve(rhs[i], nrhs));
  std::vector<const std::vector<double>*> factors;
  for (const BandedSPD& c : chol) factors.push_back(&c.factor());
  const std::vector<double> s = selected_inverse_band(factors, m.u, m.n);
  return {std::move(sols), trace_of_product(s, static_cast<int64_t>(chol.size()), m)};
}

// a + lam b on two upper bands: (rs.jtwj + lam * rs.dtgd) of sparse matrices without -0 entries, entry by entry.
Band band_add_scaled(const Band& a, const Band& b, double lam) {
  Band out(a.n, a.u);
  for (size_t k = 0; k < out.ab.size(); ++k) out.ab[k] = a.ab[k] + b.ab[k] * lam;
  return out;
}

// The rotation block's Gauss-Newton pieces at the current controls. jtwj and dtgd are kept as the upper bands
// (u = 3 K - 1) the Python's to_upper_band takes from them, summed in SciPy's SMMP order: every entry accumulates
// its products in the order of the reduced index (residual rows for J^T W J; penalty rows, then the Gram's rows
// for D^T Gamma D), which the sparse structure does not change.
struct RotSystem {
  std::vector<V> rr;
  sp::Matrix jac;
  Band jtwj, dtgd;
  std::vector<double> grad_data, grad_pen;
  double penalty;
};

class Smoother {
 public:
  Smoother(std::span<const int64_t> t_all, std::span<const Q> q_all, std::span<const V> p_all,
           const std::vector<uint8_t>& keep0, const SmoothingOptions& o, const KnotGrid& g)
      : opts(o), ts(t_all), qs(q_all), ps(p_all), keep(keep0), grid(g),
        spline(make_spline(t_all, q_all, keep0, g)) {
    prepare();
  }

  static Spline make_spline(std::span<const int64_t> t_all, std::span<const Q> q_all, const std::vector<uint8_t>& keep,
                            const KnotGrid& g) {
    std::vector<int64_t> tk;
    std::vector<Q> qk;
    for (size_t i = 0; i < keep.size(); ++i) {
      if (keep[i]) {
        tk.push_back(t_all[i]);
        qk.push_back(q_all[i]);
      }
    }
    std::vector<Q> rot0 = rot::slerp_init(tk, qk, g.greville_ns());
    return Spline(g, std::move(rot0), std::vector<V>(static_cast<size_t>(g.num_ctrl), V{0.0, 0.0, 0.0}),
                  TieMap(g.num_ctrl, {}));
  }

  void prepare() {
    idx.clear();
    t.clear();
    qm.clear();
    pm.clear();
    for (size_t i = 0; i < keep.size(); ++i) {
      if (!keep[i]) continue;
      idx.push_back(static_cast<int64_t>(i));
      t.push_back(ts[i]);
      qm.push_back(qs[i]);
      pm.push_back(ps[i]);
    }
    b = grid.design_matrix(t);
  }

  void set_ties(const std::vector<std::pair<int64_t, int64_t>>& groups) {
    spline = Spline(grid, spline.rot, spline.pos, TieMap(grid.num_ctrl, groups));
  }

  void drop(const std::vector<uint8_t>& mask_kept) {
    for (size_t k = 0; k < mask_kept.size(); ++k) {
      if (mask_kept[k]) keep[static_cast<size_t>(idx[k])] = 0;
    }
    prepare();
  }

  std::pair<std::vector<V>, std::vector<V>> residuals() const {
    const Kin ev = spline.evaluate(t);
    std::vector<V> rr(t.size()), rp(t.size());
    for (size_t i = 0; i < t.size(); ++i) {
      rr[i] = rot::relative_rotvec(qm[i], ev.q[i]);
      rp[i] = ev.p[i] - pm[i];
    }
    return {rr, rp};
  }

  std::pair<std::vector<double>, std::vector<double>> weights(const std::vector<V>& rr, const std::vector<V>& rp) const {
    const double sr2 = npo::py_pow(sigma_r, 2.0), sp2 = npo::py_pow(sigma_p, 2.0);
    std::vector<double> w(rr.size()), s(rr.size());
    for (size_t i = 0; i < rr.size(); ++i) {
      s[i] = rowsum_sq(rr[i]) / sr2 + rowsum_sq(rp[i]) / sp2;
      w[i] = 1.0 / (1.0 + s[i] / opts.loss_c2);
    }
    return {w, s};
  }

  // --- position block -----------------------------------------------------------------------------------------
  struct PosSystem {
    sp::Matrix btwb, pen;
    std::vector<double> rhs;
  };

  PosSystem pos_system(const std::vector<double>& w) const {
    const sp::Matrix& g = spline.g1;
    const sp::Matrix bw = sp::multiply_rows(b, w);
    const sp::Matrix gt = sp::transpose(g);
    PosSystem out;
    out.btwb = sp::to_csr(sp::matmul(sp::matmul(gt, sp::matmul(sp::transpose(b), bw)), g));
    out.pen = sp::to_csr(sp::matmul(sp::matmul(gt, spline.penalty_pos), g));
    std::vector<double> pmf(3 * pm.size());
    for (size_t i = 0; i < pm.size(); ++i) {
      pmf[3 * i] = pm[i].x;
      pmf[3 * i + 1] = pm[i].y;
      pmf[3 * i + 2] = pm[i].z;
    }
    out.rhs = sp::matvecs(gt, sp::matvecs(sp::transpose(bw), pmf, 3), 3);
    return out;
  }

  void solve_positions(double lam, const std::vector<double>& w) {
    const PosSystem ps_ = pos_system(w);
    spline.set_positions(BandedSPD(sp::add(ps_.btwb, sp::scaled(ps_.pen, lam))).solve(ps_.rhs, 3));
  }

  double pos_rss(const std::vector<double>& x, const std::vector<double>& w) const {
    const std::vector<double> fit = sp::matvecs(b, sp::matvecs(spline.g1, x, 3), 3);
    std::vector<double> terms(fit.size());
    for (size_t i = 0; i < pm.size(); ++i) {
      const double r[3] = {fit[3 * i] - pm[i].x, fit[3 * i + 1] - pm[i].y, fit[3 * i + 2] - pm[i].z};
      for (int c = 0; c < 3; ++c) terms[3 * i + static_cast<size_t>(c)] = w[i] * (r[c] * r[c]);
    }
    return npo::sum(terms);
  }

  // GCV over log10(lambda): whole decades, quarter decades, then a parabola (fit._gcv_search).
  template <class Eval>
  std::pair<double, GcvInfo> gcv_search(Eval evaluate) const {
    const double lo = opts.gcv_lo, hi = opts.gcv_hi;
    const std::vector<double> coarse = npo::arange(lo, hi + 1e-9, 1.0);
    if (coarse.empty()) throw std::invalid_argument("need at least one array to stack (gcv_log10_range is empty)");
    std::vector<double> lams(coarse.size());
    for (size_t i = 0; i < coarse.size(); ++i) lams[i] = npo::py_pow(10.0, coarse[i]);
    const std::vector<double> vc = evaluate(lams);
    const auto i = static_cast<size_t>(npo::nanargmin(vc));
    const double f0 = py_max(lo, coarse[i] - 1.0), f1 = (hi < coarse[i] + 1.0 ? hi : coarse[i] + 1.0) + 1e-9;
    const std::vector<double> fine = npo::arange(f0, f1, 0.25);
    std::vector<double> lf(fine.size());
    for (size_t k = 0; k < fine.size(); ++k) lf[k] = npo::py_pow(10.0, fine[k]);
    const std::vector<double> vf = evaluate(lf);
    const auto j = static_cast<size_t>(npo::nanargmin(vf));
    double best = fine[j];
    if (0 < j && j < fine.size() - 1) {
      const double x[3] = {fine[j - 1], fine[j], fine[j + 1]};
      const double y[3] = {vf[j - 1], vf[j], vf[j + 1]};
      const std::vector<double> a = npo::polyfit2(x, y);
      if (a[0] > 0) best = std::clamp(-a[1] / (2 * a[0]), x[0], x[2]);
    }
    GcvInfo info;
    info.searched = true;
    info.at_grid_edge = best <= lo + 1e-9 || best >= hi - 1e-9;
    for (const double c : coarse) info.log10_grid.push_back(npo::round_decimals(c, 4));
    for (const double c : fine) info.log10_grid.push_back(npo::round_decimals(c, 4));
    info.gcv = vc;
    info.gcv.insert(info.gcv.end(), vf.begin(), vf.end());
    return {npo::py_pow(10.0, best), info};
  }

  static std::vector<double> gcv_values(double nd, const std::vector<double>& rss, const std::vector<double>& tr) {
    std::vector<double> out(rss.size());
    for (size_t i = 0; i < rss.size(); ++i) out[i] = (nd * rss[i]) / sq(maximum(nd - tr[i], 1e-9));
    return out;
  }

  std::pair<std::vector<double>, std::vector<double>> pos_rss_edf(const PosSystem& sys, const std::vector<double>& w,
                                                                  const std::vector<double>& lams) const {
    const int64_t u = grid.order - 1;
    std::vector<Band> mats;
    for (const double lam : lams) mats.push_back(to_upper_band(sp::add(sys.btwb, sp::scaled(sys.pen, lam)), u));
    const auto [sols, tr] = solve_and_trace(mats, std::vector<std::vector<double>>(lams.size(), sys.rhs), 3,
                                            to_upper_band(sys.btwb, u));
    std::vector<double> rss;
    for (const auto& x : sols) rss.push_back(pos_rss(x, w));
    return {rss, tr};
  }

  std::pair<double, GcvInfo> gcv_positions(const std::vector<double>& w) const {
    const PosSystem sys = pos_system(w);
    const auto nd = static_cast<double>(t.size());
    auto gcv = [&](const std::vector<double>& lams) {
      const auto [rss, tr] = pos_rss_edf(sys, w, lams);
      return gcv_values(nd, rss, tr);
    };
    auto [lam, info] = gcv_search(gcv);
    const auto [rss, tr] = pos_rss_edf(sys, w, {lam});
    info.edf = tr[0];
    info.rss = rss[0];
    return {lam, info};
  }

  // --- rotation block -----------------------------------------------------------------------------------------
  RotSystem rot_system(const std::vector<double>& w) const {
    const Kin ev = spline.evaluate(t);
    RotSystem rs;
    rs.rr.resize(t.size());
    for (size_t i = 0; i < t.size(); ++i) rs.rr[i] = rot::relative_rotvec(qm[i], ev.q[i]);
    const Spline::Increments inc = spline.linearization();
    rs.jac = spline.times_tie(spline.rotation_jacobian(t, rs.rr, inc), spline.g3);
    const int64_t u = 3 * grid.order - 1, nf = rs.jac.cols, ub = u + 1;
    // Upper-band accumulators by row (acc[f * ub + (g - f)]): each entry sums its terms in the order of the reduced
    // index, as SciPy's product does; converted to LAPACK's band layout at the end. Every row of J and of D spans a
    // contiguous range of free parameters (the tie map is monotone), so each is expanded to a dense window: the
    // entries SciPy dropped as exact zeros only add +-0 products, which leave a sum started at +0 unchanged.
    std::vector<double> acc(static_cast<size_t>(nf * ub), 0.0);
    auto to_band = [&](Band& band) {
      band = Band(nf, u);
      for (int64_t f = 0; f < nf; ++f) {
        for (int64_t dd = 0; dd <= u && f + dd < nf; ++dd) band.at(f, f + dd) = acc[static_cast<size_t>(f * ub + dd)];
      }
    };
    auto window = [](const sp::Matrix& a, int64_t row, std::vector<double>& x) {
      const int64_t b0 = a.indptr[static_cast<size_t>(row)], b1 = a.indptr[static_cast<size_t>(row + 1)];
      int64_t lo = INT64_MAX, hi = -1;
      for (int64_t jj = b0; jj < b1; ++jj) {
        lo = std::min(lo, a.indices[static_cast<size_t>(jj)]);
        hi = std::max(hi, a.indices[static_cast<size_t>(jj)]);
      }
      if (hi < 0) return std::pair<int64_t, int64_t>{0, 0};
      x.assign(static_cast<size_t>(hi - lo + 1), 0.0);
      for (int64_t jj = b0; jj < b1; ++jj) {
        x[static_cast<size_t>(a.indices[static_cast<size_t>(jj)] - lo)] = a.data[static_cast<size_t>(jj)];
      }
      return std::pair<int64_t, int64_t>{lo, hi - lo + 1};
    };
    // J^T W J (jtw @ jac) and J^T W r (jtw @ rr): jtw's rows list the residual rows in order.
    rs.grad_data.assign(static_cast<size_t>(nf), 0.0);
    const std::vector<double> rrf = Spline::ravel(rs.rr);
    std::vector<double> x, xw;
    for (int64_t r = 0; r < rs.jac.rows; ++r) {
      const auto [f0, len] = window(rs.jac, r, x);
      const double wr = w[static_cast<size_t>(r / 3)], rv = rrf[static_cast<size_t>(r)];
      xw.resize(x.size());
      for (int64_t p = 0; p < len; ++p) {
        xw[static_cast<size_t>(p)] = x[static_cast<size_t>(p)] * wr;
        rs.grad_data[static_cast<size_t>(f0 + p)] += xw[static_cast<size_t>(p)] * rv;
      }
      for (int64_t p = 0; p < len; ++p) {
        double* row = &acc[static_cast<size_t>((f0 + p) * ub)];
        const double a = xw[static_cast<size_t>(p)];
        const int64_t qmax = std::min(len - 1, p + u);
        for (int64_t q2 = p; q2 <= qmax; ++q2) row[q2 - p] += a * x[static_cast<size_t>(q2)];
      }
    }
    to_band(rs.jtwj);
    // D^T Gamma D ((de.T @ gamma) @ de) and D^T (Gamma e).
    auto [e, de0] = spline.rotation_penalty(inc);
    const sp::Matrix de = spline.tie.groups.empty() ? std::move(de0) : sp::matmul(de0, spline.g3);
    const sp::Matrix& gcsc = spline.gamma3_csc;
    std::fill(acc.begin(), acc.end(), 0.0);
    // de's rows as dense windows of a common width (flat storage).
    std::vector<int64_t> de_lo(static_cast<size_t>(de.rows)), de_len(static_cast<size_t>(de.rows));
    int64_t width = 0;
    for (int64_t p = 0; p < de.rows; ++p) {
      int64_t lo = INT64_MAX, hi = -1;
      for (int64_t jj = de.indptr[static_cast<size_t>(p)]; jj < de.indptr[static_cast<size_t>(p + 1)]; ++jj) {
        lo = std::min(lo, de.indices[static_cast<size_t>(jj)]);
        hi = std::max(hi, de.indices[static_cast<size_t>(jj)]);
      }
      de_lo[static_cast<size_t>(p)] = hi < 0 ? 0 : lo;
      de_len[static_cast<size_t>(p)] = hi < 0 ? 0 : hi - lo + 1;
      width = std::max(width, de_len[static_cast<size_t>(p)]);
    }
    std::vector<double> de_flat(static_cast<size_t>(de.rows * width), 0.0);
    for (int64_t p = 0; p < de.rows; ++p) {
      for (int64_t jj = de.indptr[static_cast<size_t>(p)]; jj < de.indptr[static_cast<size_t>(p + 1)]; ++jj) {
        de_flat[static_cast<size_t>(p * width + de.indices[static_cast<size_t>(jj)] - de_lo[static_cast<size_t>(p)])] =
            de.data[static_cast<size_t>(jj)];
      }
    }
    std::vector<double> t1;
    for (int64_t q = 0; q < de.rows; ++q) {
      // Row q of (de.T @ gamma)^T: over the Gram's column q (rows ascending), gamma[p, q] de[p, :].
      int64_t lo = INT64_MAX, hi = -1;
      for (int64_t pp = gcsc.indptr[static_cast<size_t>(q)]; pp < gcsc.indptr[static_cast<size_t>(q + 1)]; ++pp) {
        const auto p = static_cast<size_t>(gcsc.indices[static_cast<size_t>(pp)]);
        if (de_len[p] == 0) continue;
        lo = std::min(lo, de_lo[p]);
        hi = std::max(hi, de_lo[p] + de_len[p] - 1);
      }
      const int64_t qlen = de_len[static_cast<size_t>(q)];
      if (hi < 0 || qlen == 0) continue;
      t1.assign(static_cast<size_t>(hi - lo + 1), 0.0);
      for (int64_t pp = gcsc.indptr[static_cast<size_t>(q)]; pp < gcsc.indptr[static_cast<size_t>(q + 1)]; ++pp) {
        const auto p = static_cast<size_t>(gcsc.indices[static_cast<size_t>(pp)]);
        const double gv = gcsc.data[static_cast<size_t>(pp)];
        const double* dp = &de_flat[p * static_cast<size_t>(width)];
        double* tt = &t1[static_cast<size_t>(de_lo[p] - lo)];
        for (int64_t k = 0; k < de_len[p]; ++k) tt[k] += gv * dp[k];
      }
      const double* dq = &de_flat[static_cast<size_t>(q * width)];
      const int64_t g0 = de_lo[static_cast<size_t>(q)];
      for (int64_t f = lo; f <= hi; ++f) {
        const double tf = t1[static_cast<size_t>(f - lo)];
        double* row = &acc[static_cast<size_t>(f * ub)];
        const int64_t gb = std::max(g0, f), ge_ = std::min(g0 + qlen - 1, f + u);
        for (int64_t g = gb; g <= ge_; ++g) row[g - f] += dq[g - g0] * tf;
      }
    }
    to_band(rs.dtgd);
    const std::vector<double> ge = sp::matvec(spline.gamma3, e);
    rs.grad_pen.assign(static_cast<size_t>(nf), 0.0);
    for (int64_t p = 0; p < de.rows; ++p) {
      for (int64_t k = de.indptr[static_cast<size_t>(p)]; k < de.indptr[static_cast<size_t>(p + 1)]; ++k) {
        rs.grad_pen[static_cast<size_t>(de.indices[static_cast<size_t>(k)])] +=
            de.data[static_cast<size_t>(k)] * ge[static_cast<size_t>(p)];
      }
    }
    rs.penalty = blas::dot(static_cast<int>(e.size()), e.data(), ge.data());
    return rs;
  }

  double weighted_rr(const std::vector<double>& w, const std::vector<V>& rr) const {
    std::vector<double> terms(rr.size());
    for (size_t i = 0; i < rr.size(); ++i) terms[i] = w[i] * rowsum_sq(rr[i]);
    return npo::sum(terms);
  }

  double rot_objective(const std::vector<double>& w, double lam) const {
    const std::vector<V> rr = residuals().first;
    return weighted_rr(w, rr) + lam * spline.rotation_penalty_value();
  }

  double rotation_steps(double lam, const std::vector<double>& w, int steps) {
    double last = 0.0;
    for (int st = 0; st < steps; ++st) {
      const RotSystem rs = rot_system(w);
      const Band h = band_add_scaled(rs.jtwj, rs.dtgd, lam);
      std::vector<double> g(rs.grad_data.size()), mg(rs.grad_data.size());
      for (size_t i = 0; i < g.size(); ++i) {
        g[i] = rs.grad_data[i] + lam * rs.grad_pen[i];
        mg[i] = -g[i];
      }
      const double f0 = weighted_rr(w, rs.rr) + lam * rs.penalty;
      const std::vector<Q> rot_saved = spline.rot;
      bool accepted = false;
      std::vector<double> delta;
      for (int tr = 0; tr < 12; ++tr) {
        // h + mu sp.diags(h.diagonal()) on the band.
        Band hm = h;
        for (int64_t i = 0; i < h.n; ++i) hm.at(i, i) = h.at(i, i) + h.at(i, i) * mu;
        delta = BandedSPD(hm).solve(mg, 1);
        spline.apply_rotation_step(delta);
        const double f1 = rot_objective(w, lam);
        if (f1 <= f0 * (1 + 1e-12) + 1e-300) {
          mu = py_max(mu / 3.0, 1e-12);
          accepted = true;
          break;
        }
        spline.rot = rot_saved;
        mu *= 10.0;
      }
      if (accepted) {
        std::vector<double> ad(delta.size());
        for (size_t i = 0; i < delta.size(); ++i) ad[i] = std::abs(delta[i]);
        last = max(ad);
      } else {
        last = 0.0;
      }
      if (!accepted || last < 1e-10) break;
    }
    return last;
  }

  std::pair<std::vector<double>, std::vector<double>> rot_rss_edf(const RotSystem& rs, const std::vector<double>& w3,
                                                                  const std::vector<double>& lams) const {
    std::vector<Band> mats;
    std::vector<std::vector<double>> rhs;
    for (const double lam : lams) {
      mats.push_back(band_add_scaled(rs.jtwj, rs.dtgd, lam));
      std::vector<double> r(rs.grad_data.size());
      for (size_t i = 0; i < r.size(); ++i) r[i] = -(rs.grad_data[i] + lam * rs.grad_pen[i]);
      rhs.push_back(std::move(r));
    }
    const auto [sols, tr] = solve_and_trace(mats, rhs, 1, rs.jtwj);
    const std::vector<double> rrf = Spline::ravel(rs.rr);
    std::vector<double> rss;
    for (const auto& delta : sols) {
      const std::vector<double> jd = sp::matvec(rs.jac, delta);
      std::vector<double> terms(rrf.size());
      for (size_t i = 0; i < rrf.size(); ++i) terms[i] = w3[i] * sq(rrf[i] + jd[i]);
      rss.push_back(npo::sum(terms));
    }
    return {rss, tr};
  }

  std::pair<double, GcvInfo> gcv_rotations(const std::vector<double>& w) const {
    const RotSystem rs = rot_system(w);
    const auto nd = static_cast<double>(3 * t.size());
    std::vector<double> w3(3 * w.size());
    for (size_t i = 0; i < w.size(); ++i) w3[3 * i] = w3[3 * i + 1] = w3[3 * i + 2] = w[i];
    auto gcv = [&](const std::vector<double>& lams) {
      const auto [rss, tr] = rot_rss_edf(rs, w3, lams);
      return gcv_values(nd, rss, tr);
    };
    auto [lam, info] = gcv_search(gcv);
    const auto [rss, tr] = rot_rss_edf(rs, w3, {lam});
    info.edf = tr[0];
    info.rss = rss[0];
    return {lam, info};
  }

  // RSS and edf at a user-fixed lambda (_fixed_lambda_info).
  GcvInfo fixed_lambda_info(double lam, const std::vector<double>& w, bool positions) const {
    GcvInfo info;
    if (positions) {
      const PosSystem sys = pos_system(w);
      const int64_t u = grid.order - 1;
      const auto [sols, tr] = solve_and_trace({to_upper_band(sp::add(sys.btwb, sp::scaled(sys.pen, lam)), u)}, {sys.rhs},
                                              3, to_upper_band(sys.btwb, u));
      info.edf = tr[0];
      info.rss = pos_rss(sols[0], w);
      return info;
    }
    const RotSystem rs = rot_system(w);
    std::vector<double> r(rs.grad_data.size());
    for (size_t i = 0; i < r.size(); ++i) r[i] = -(rs.grad_data[i] + lam * rs.grad_pen[i]);
    const auto [sols, tr] = solve_and_trace({band_add_scaled(rs.jtwj, rs.dtgd, lam)}, {r}, 1, rs.jtwj);
    info.edf = tr[0];
    info.rss = weighted_rr(w, rs.rr);
    return info;
  }

  // --- IRLS ---------------------------------------------------------------------------------------------------
  int64_t irls(double lam_p, double lam_r) {
    const double tol_p = py_max(1e-2 * sigma_p, 1e-9), tol_r = py_max(1e-2 * sigma_r, 1e-9);
    for (int it = 0; it < opts.max_irls; ++it) {
      const auto [rr, rp] = residuals();
      const std::vector<double> w = weights(rr, rp).first;
      const std::vector<V> pos_old = spline.pos;
      solve_positions(lam_p, w);
      const double step = rotation_steps(lam_r, w, 1);
      if (it && step < tol_r) {
        std::vector<double> dif(3 * pos_old.size());
        for (size_t c = 0; c < pos_old.size(); ++c) {
          const V dd = spline.pos[c] - pos_old[c];
          dif[3 * c] = std::abs(dd.x);
          dif[3 * c + 1] = std::abs(dd.y);
          dif[3 * c + 2] = std::abs(dd.z);
        }
        if (max(dif) < tol_p) return it + 1;
      }
    }
    return opts.max_irls;
  }

  void update_sigmas(const GcvInfo& ip, const GcvInfo& ir) {
    const auto nn = static_cast<double>(t.size());
    sigma_p = py_max(std::sqrt(ip.rss / py_max(3 * (nn - ip.edf), 1.0)), opts.sigma_floor_pos_m);
    sigma_r = py_max(std::sqrt(ir.rss / py_max(3 * nn - ir.edf, 1.0)), opts.sigma_floor_rot_rad);
  }

  std::vector<double> current_weights() const {
    const auto [rr, rp] = residuals();
    return weights(rr, rp).first;
  }

  const SmoothingOptions& opts;
  std::span<const int64_t> ts;
  std::span<const Q> qs;
  std::span<const V> ps;
  std::vector<uint8_t> keep;
  KnotGrid grid;
  double sigma_p = 1e-3, sigma_r = 1e-3, mu = 1e-8;
  Spline spline;
  std::vector<int64_t> idx, t;
  std::vector<Q> qm;
  std::vector<V> pm;
  sp::Matrix b;
};

int64_t check_rotation_aliasing(const std::vector<Q>& q) {
  if (q.size() < 2) return 0;
  std::vector<double> steps(q.size() - 1);
  for (size_t i = 0; i + 1 < q.size(); ++i) steps[i] = rot::rotation_angle(q[i], q[i + 1]);
  if (max(steps) > 0.9 * std::numbers::pi) {
    throw TrajectoryError("consecutive samples differ by more than 162 deg: the rotation direction is ambiguous; "
                                "add intermediate samples");
  }
  const double warn = 0.6 * std::numbers::pi;
  return std::count_if(steps.begin(), steps.end(), [warn](double s) { return s > warn; });
}

double cutoff_hz(double lam_si, double period_s, int m) {
  return npo::py_pow(1.0 / (lam_si * period_s), 1.0 / (2 * m)) / (2 * std::numbers::pi);
}

std::vector<double> linspace64(int64_t a, int64_t b) {
  const double delta = static_cast<double>(b) - static_cast<double>(a);
  const double step = delta / 63;
  std::vector<double> y(64);
  for (int i = 0; i < 64; ++i) y[static_cast<size_t>(i)] = static_cast<double>(i) * step + static_cast<double>(a);
  y[63] = static_cast<double>(b);
  return y;
}

}  // namespace

FitResult fit_smoothing(std::span<const int64_t> t_ns, std::span<const Q> q, std::span<const V> p,
                        const SmoothingOptions& opts) {
  const blas::OneThread one_thread;
  const size_t n = t_ns.size();
  check_pose_samples(t_ns, q, p);
  if (n < 2) throw TrajectoryError("smoothing needs at least two samples");
  if ((opts.hold_rule != "sphere" && opts.hold_rule != "image") || (opts.hold_rule == "image" && !opts.hold_view)) {
    throw std::invalid_argument("hold_rule " + opts.hold_rule + ": sphere, or image with a HoldView (the cameras)");
  }
  if (opts.hold_rule == "image" && !(opts.hold_min_ns > 0 && opts.hold_max_disp_px > 0)) {
    throw std::invalid_argument("hold_rule image needs scene.trajectory.hold_min_ns > 0 and hold_max_disp_px > 0");
  }
  FitResult res;
  FitDiagnostics& diag = res.diag;
  diag.num_samples = static_cast<int64_t>(n);
  std::vector<int64_t> dts(n - 1);
  for (size_t i = 0; i + 1 < n; ++i) dts[i] = t_ns[i + 1] - t_ns[i];
  const double period = npo::median(dts);
  std::vector<uint8_t> keep(n, 1);
  if (opts.gate) {
    GateOptions go{period, opts.gate_sigma_k, opts.gate_rot_floor_rad, opts.gate_pos_floor_m, opts.branch_jump_rad,
                   opts.branch_rate_factor};
    diag.gate = gate_poses(t_ns, q, p, go);
    diag.gated = true;
    keep = diag.gate.keep;
    for (size_t i = 0; i < n; ++i) {
      if (!keep[i]) diag.gate_rejected_t_ns.push_back(t_ns[i]);
    }
  }
  std::vector<size_t> kept;
  for (size_t i = 0; i < n; ++i) {
    if (keep[i]) kept.push_back(i);
  }
  if (kept.size() < 2) throw TrajectoryError("fewer than two samples survive gating");
  std::vector<Q> qk;
  for (const size_t i : kept) qk.push_back(q[i]);
  diag.rotation_alias_warnings = check_rotation_aliasing(qk);

  const int64_t h = opts.h_ns ? *opts.h_ns : static_cast<int64_t>(std::nearbyint(period));
  const int64_t t_first = t_ns[kept.front()], t_last = t_ns[kept.back()];
  const std::pair<int64_t, int64_t> span = opts.span_ns ? *opts.span_ns : std::pair<int64_t, int64_t>{t_first, t_last};
  if (span.first > t_first || span.second < t_last) throw std::invalid_argument("span_ns must contain every kept sample");
  const KnotGrid grid = KnotGrid::covering(span.first, span.second, h, opts.order);
  // The robust part of the fit (round 0, the robust rounds, G-b, holds) on this grid; early: tie groups held from
  // round 0 (the presence fallback below), none on the normal path.
  std::optional<Smoother> fit_slot;
  double lam_p = 0, lam_r = 0;
  GcvInfo info_p, info_r;
  HoldResult holds;
  bool whole = false;
  std::vector<std::pair<int64_t, int64_t>> presence, still;
  auto robust = [&](const std::vector<std::pair<int64_t, int64_t>>& early) {
    diag.irls_iterations.clear();
    diag.gb_rejected_t_ns.clear();
    holds = HoldResult{};
    presence.clear();
    Smoother& fit = fit_slot.emplace(t_ns, q, p, keep, opts, grid);
    if (!early.empty()) fit.set_ties(early);

    // Round 0: unweighted fit with lambda by GCV, then sigma from its residuals.
    const std::vector<double> w1(fit.t.size(), 1.0);
    bool have_p = false;
    if (opts.lambda_pos) {
      lam_p = *opts.lambda_pos;
    } else {
      std::tie(lam_p, info_p) = fit.gcv_positions(w1);
      have_p = true;
    }
    fit.solve_positions(lam_p, w1);
    lam_r = opts.lambda_rot ? *opts.lambda_rot : lam_p;
    fit.rotation_steps(lam_r, w1, 20);
    if (!opts.lambda_rot) {
      std::tie(lam_r, info_r) = fit.gcv_rotations(w1);
      fit.rotation_steps(lam_r, w1, 20);
    }
    if (!have_p) info_p = fit.fixed_lambda_info(lam_p, w1, true);
    if (opts.lambda_rot) info_r = fit.fixed_lambda_info(lam_r, w1, false);
    fit.update_sigmas(info_p, info_r);

    // Robust rounds: IRLS at fixed lambda, then GCV with the converged weights, while lambda moves > 1/4 decade.
    for (int round = 0; round < 3; ++round) {
      diag.irls_iterations.push_back(fit.irls(lam_p, lam_r));
      const std::vector<double> w = fit.current_weights();
      double new_p, new_r;
      if (opts.lambda_pos) {
        new_p = lam_p;
        info_p = fit.fixed_lambda_info(lam_p, w, true);
      } else {
        std::tie(new_p, info_p) = fit.gcv_positions(w);
      }
      if (opts.lambda_rot) {
        new_r = lam_r;
        info_r = fit.fixed_lambda_info(lam_r, w, false);
      } else {
        std::tie(new_r, info_r) = fit.gcv_rotations(w);
      }
      fit.update_sigmas(info_p, info_r);
      const bool moved = std::abs(std::log10(new_p / lam_p)) > 0.25 || std::abs(std::log10(new_r / lam_r)) > 0.25;
      lam_p = new_p;
      lam_r = new_r;
      if (!moved) break;
    }
    diag.irls_iterations.push_back(fit.irls(lam_p, lam_r));

    // Stage G-b on the unconstrained robust fit.
    {
      const auto [rr, rp] = fit.residuals();
      const double sr2 = npo::py_pow(fit.sigma_r, 2.0), sp2 = npo::py_pow(fit.sigma_p, 2.0);
      std::vector<uint8_t> bad(rr.size());
      size_t nbad = 0;
      for (size_t i = 0; i < rr.size(); ++i) {
        const double maha = std::sqrt(rowsum_sq(rr[i]) / sr2 + rowsum_sq(rp[i]) / sp2);
        bad[i] = maha > opts.mahalanobis_max ? 1 : 0;
        if (bad[i]) {
          diag.gb_rejected_t_ns.push_back(fit.t[i]);
          ++nbad;
        }
      }
      if (nbad > 0 && rr.size() - nbad >= 2) {
        fit.drop(bad);
        diag.irls_iterations.push_back(fit.irls(lam_p, lam_r));
      }
    }

    // Holds: static stretches on the gated samples, plus presence holds outside the data.
    if (opts.detect_holds && opts.hold_rule == "image") {
      const Kin ev = fit.spline.evaluate(fit.t);
      holds = detect_static_image(fit.t, ev.q, ev.p, ev.v, ev.omega, *opts.hold_view, fit.idx, opts.center_body_m,
                                  opts.radius_m, opts.hold_min_ns, opts.hold_max_disp_px,
                                  2 * (grid.order - 1) * grid.h_ns);
    } else if (opts.detect_holds) {
      const Kin ev = fit.spline.evaluate(fit.t);
      std::vector<double> speed(fit.t.size());
      for (size_t i = 0; i < speed.size(); ++i) {
        speed[i] = sphere_speed(ev.v[i], ev.omega[i], ev.q[i], opts.center_body_m, opts.radius_m);
      }
      StaticOptions so;
      so.max_disp = opts.hold_max_disp_m;
      so.min_duration_ns = opts.hold_min_ns;
      so.moving_speed = opts.hold_moving_mps;
      holds = detect_static(fit.t, fit.qm, fit.pm, speed, opts.center_body_m, opts.radius_m, so);
    }
    std::vector<std::pair<int64_t, int64_t>> intervals;
    for (const StaticInterval& s : holds.accepted) intervals.emplace_back(s.t_begin_ns, s.t_end_ns);
    whole = holds.accepted.size() == 1 && holds.accepted[0].i0 == 0 &&
            holds.accepted[0].i1 == static_cast<int64_t>(fit.t.size()) - 1;
    if (whole) {
      intervals = {span};
    } else {
      if (span.first < t_first) presence.emplace_back(span.first, t_first);
      if (span.second > t_last) presence.emplace_back(t_last, span.second);
    }
    std::vector<std::pair<int64_t, int64_t>> all = intervals;
    all.insert(all.end(), presence.begin(), presence.end());
    auto [groups, still_] = hold_groups(grid, all);
    still = std::move(still_);
    if (!groups.empty()) {
      fit.set_ties(groups);
      diag.irls_iterations.push_back(fit.irls(lam_p, lam_r));
    }
  };
  // Presence fallback. Outside its kept samples a body's controls are set by the jerk penalty alone until the
  // presence holds tie them at the end. On a data-free stretch of L controls the system there is lambda P, and the
  // smallest eigenvalue of P's leading minors falls as L^-6 (L^-2m, m = 3); past about 700 controls (3.5 s at
  // 200 Hz) it is below the rounding of P's float64 entries, a leading minor is indefinite and the band Cholesky
  // fails (P alone on 1500 controls: minor 861). A7's drone_01, a rejected orientation branch, leaves 668 free. Such
  // a fit refits once with the presence groups tied from round 0, the hold its result gets anyway; a fit that
  // factorizes keeps its path bit for bit.
  try {
    robust({});
  } catch (const LinAlgError& e) {
    std::vector<std::pair<int64_t, int64_t>> outside;
    if (span.first < t_first) outside.emplace_back(span.first, t_first);
    if (span.second > t_last) outside.emplace_back(t_last, span.second);
    const std::vector<std::pair<int64_t, int64_t>> early = hold_groups(grid, outside).first;
    if (early.empty()) throw;
    diag.presence_fallback = true;
    diag.presence_fallback_error = e.what();
    diag.presence_fallback_groups = early;
    robust(early);
  }
  Smoother& fit = *fit_slot;

  res.order = grid.order;
  res.t0_ns = grid.t0_ns;
  res.h_ns = grid.h_ns;
  res.rot = fit.spline.rot;
  res.pos = fit.spline.pos;
  res.holds = still;
  std::sort(res.holds.begin(), res.holds.end());
  res.span_ns = span;
  std::vector<double> incr(res.rot.size() - 1);
  for (size_t c = 0; c + 1 < res.rot.size(); ++c) incr[c] = rot::rotation_angle(res.rot[c], res.rot[c + 1]);
  const double worst = max(incr);
  if (worst > std::numbers::pi / 2) {
    throw TrajectoryError("per-knot rotation increment exceeds 90 deg; halve the knot spacing (M2 §C.5)");
  }

  const std::vector<double> w_final = fit.current_weights();
  res.weights.assign(n, 0.0);
  for (size_t k = 0; k < fit.idx.size(); ++k) res.weights[static_cast<size_t>(fit.idx[k])] = w_final[k];
  res.keep = fit.keep;
  const auto [rr, rp] = fit.residuals();
  double disp_still = 0.0;
  if (!still.empty()) {
    for (const auto& [a, b] : res.holds) {
      std::vector<int64_t> ts;
      for (const double y : linspace64(a, b)) {
        const auto ti = static_cast<int64_t>(y);
        if (ti < b) ts.push_back(ti);
      }
      const Kin k = eval_controls(res.order, res.t0_ns, res.h_ns, res.rot, res.pos, ts);
      std::vector<double> d(ts.size());
      for (size_t i = 0; i < ts.size(); ++i) {
        d[i] = sphere_displacement(k.q[0], k.p[0], k.q[i], k.p[i], opts.center_body_m, opts.radius_m);
      }
      disp_still = py_max(disp_still, max(d));
    }
  }
  const int m = grid.order / 2;
  const double h_s = static_cast<double>(h) * 1e-9;
  const double hs_pow = npo::py_pow(h_s, static_cast<double>(2 * m - 1));
  diag.order = grid.order;
  diag.h_ns = h;
  diag.t0_ns = grid.t0_ns;
  diag.num_ctrl = grid.num_ctrl;
  diag.span_ns = span;
  diag.lambda_pos_u = lam_p;
  diag.lambda_rot_u = lam_r;
  diag.lambda_pos_si = lam_p * hs_pow;
  diag.lambda_rot_si = lam_r * hs_pow;
  diag.q_pos = npo::py_pow(fit.sigma_p, 2.0) / (lam_p * hs_pow);
  diag.q_rot = npo::py_pow(fit.sigma_r, 2.0) / (lam_r * hs_pow);
  diag.cutoff_hz_pos = cutoff_hz(lam_p * hs_pow, period * 1e-9, m);
  diag.cutoff_hz_rot = cutoff_hz(lam_r * hs_pow, period * 1e-9, m);
  diag.edf_pos = info_p.edf;
  diag.edf_rot = info_r.edf;
  diag.gcv_pos = info_p;
  diag.gcv_rot = info_r;
  diag.sigma_pos_m = fit.sigma_p;
  diag.sigma_rot_rad = fit.sigma_r;
  std::vector<double> ssp(rp.size()), ssr(rr.size()), np_(rp.size()), nr(rr.size());
  for (size_t i = 0; i < rp.size(); ++i) {
    ssp[i] = rowsum_sq(rp[i]);
    ssr[i] = rowsum_sq(rr[i]);
    np_[i] = std::sqrt(ssp[i]);
    nr[i] = std::sqrt(ssr[i]);
  }
  diag.rms_pos_m = std::sqrt(npo::mean(ssp));
  diag.rms_rot_rad = std::sqrt(npo::mean(ssr));
  diag.max_pos_m = max(np_);
  diag.max_rot_rad = max(nr);
  diag.min_weight = min(w_final);
  diag.mean_weight = npo::mean(w_final);
  diag.num_kept = static_cast<int64_t>(fit.t.size());
  diag.hold_rule = opts.hold_rule;
  diag.holds_ns = res.holds;
  diag.static_intervals = holds.accepted;
  diag.rejected_static_candidates = holds.rejected;
  diag.presence_holds_ns = presence;
  diag.still_max_displacement_m = disp_still;
  diag.max_increment_rad = worst;
  for (size_t i = 0; i + 1 < fit.t.size(); ++i) {
    if (static_cast<double>(fit.t[i + 1] - fit.t[i]) > 1.5 * period) diag.gaps_ns.emplace_back(fit.t[i], fit.t[i + 1]);
  }
  diag.is_static = whole;
  blas::OneThread::verify();
  return res;
}

std::pair<Q, V> evaluate_pose(const FitResult& fit, int64_t t_ns) {
  const int64_t t[1] = {t_ns};
  const Kin k = eval_controls(fit.order, fit.t0_ns, fit.h_ns, fit.rot, fit.pos, t);
  return {k.q[0], k.p[0]};
}

void check_pose_samples(std::span<const int64_t> t_ns, std::span<const Q> q, std::span<const V> p) {
  const size_t n = t_ns.size();
  if (q.size() != n || p.size() != n) throw std::invalid_argument("PoseSamples needs t_ns (n,), q (n, 4), p (n, 3)");
  for (size_t i = 1; i < n; ++i) {
    if (t_ns[i] <= t_ns[i - 1]) throw std::invalid_argument("sample times must be strictly increasing");
  }
  for (size_t i = 0; i < n; ++i) {
    const Q& a = q[i];
    const V& b = p[i];
    if (!(std::isfinite(a.w) && std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z) && std::isfinite(b.x) &&
          std::isfinite(b.y) && std::isfinite(b.z))) {
      throw std::invalid_argument("samples contain NaN or Inf");
    }
    const double norm = std::sqrt(0.0 + ((((0.0 + a.w * a.w) + a.x * a.x) + a.y * a.y) + a.z * a.z));
    if (std::abs(norm - 1.0) > 1e-3) throw std::invalid_argument("sample quaternions must have unit norm");
  }
}

std::vector<Q> pose_samples_q(std::span<const Q> q) {
  std::vector<Q> out(q.size());
  for (size_t i = 0; i < q.size(); ++i) out[i] = rot::normalize_quat(q[i]);
  return out;
}

}  // namespace balla::scene::traj
