// fp64 references of the LTC tables (p2f.md §4.3, §6.1 T-TAB; F3c), shared by the generator (gen_tables.cpp) and
// tests/render/test_bsdf.cpp: the lobe the LTC approximates, the LTC itself, Heitz et al.'s (2016) fit and the fit's
// error measures. Host only and deterministic like tables_ref.h (+, -, *, /, sqrt and pmath).
//
// The lobe. The single-scatter GGX lobe of bsdf.h with F = 1, normalized: f^(l) = D V mu_l / E1, E1 = A + B =
// int D V mu_l dw_l. The Fresnel factor and the magnitude enter through E_s (bsdf.h spec_albedo), as fitLTC's
// separate magnitude and Fresnel tables do, so L_spec = c_L L E_s I_ltc / pi (p2f.md §4.6). Its mean direction
// w_bar = int f^ l dw / |.| (in the plane of v and n) is the fit's frame, as in fitLTC.
//
// The LTC. M = [X Y Z] K with Z = w_bar turned by psi about Y, X = (Z_z, 0, -Z_x), Y = (0, 1, 0) and K = [[m11, 0,
// m13], [0, m22, 0], [0, 0, 1]] (fitLTC's parameters, with psi = 0); tabulated as M^-1 / (M^-1)_11 in Heitz et al.'s
// layout, [[m00, 0, m02], [0, 1, 0], [m20, 0, m22]] (Entries), in the frame (t, b, n) of bsdf.h ltc_frame: t towards
// the view's projection. Its density is D(w) = max(0, w'_z) det(M^-1) / (pi |w'|^4), w' = M^-1 w, invariant under
// scaling of M^-1.
//
// The fit. fitLTC's procedure with one more free parameter: Nelder-Mead on (ln m11, ln m22, m13 / m11_seed, psi)
// minimizing int |f^ - D|^3 dw, where psi turns the frame's axis Z away from w_bar about Y (fitLTC fixes Z = w_bar and
// fits three; the fourth is the last free entry of the layout: freeing it lowers the objective 1.6-13x off normal view
// and lifts the full-hemisphere mass at grazing views from 0.79-0.95 to >= 0.987, though cap25 below rises at some
// grazing views for r 0.2-0.3). The error is estimated with the balance heuristic over a kFitGrid^2 midpoint
// grid drawn through both the LTC and the lobe's visible normals (Heitz 2018): fixed points, so it is a deterministic
// function of the parameters. Each roughness row runs on one thread: its normal-view node is the isotropic fit
// (m11 = m22 = m, m13 = psi = 0, w_bar = n: M^-1 ~ diag(1, 1, m), golden section in ln m), and every next view starts
// from the previous one. fitLTC seeds the normal-view node from the next rougher row instead, which would serialize the
// rows; the isotropic fit is one-dimensional and needs no seed. (fitLTC's details here are from memory of its source,
// which has no local copy; the paper's layout and objective are as cited.)
#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/math/scalar.h"
#include "core/math/vec.h"
#include "core/pmath/explog.h"
#include "core/pmath/trig.h"
#include "tables_ref.h"

namespace balla::bsdf_ref::ltc {

// The r = 0 row holds the alpha -> 0 limit, fitted at r = 1e-3 (alpha = 1e-6; m20 and m22 ~ 2 alpha vanish there).
// bsdf.h's floor alpha = 1e-4 is table coordinate r = 0.01, between this node and the next (1/63).
inline constexpr double kRMin = 1e-3;
inline constexpr int kFitGrid = 40;          // midpoints per axis and sampling strategy
inline constexpr double kMomentTol = 1e-9;   // absolute quadrature tolerance of E1 and the mean direction
inline constexpr double kFitXTol = 1e-8;     // Nelder-Mead stops when the simplex is this small in every parameter
inline constexpr int kFitMaxEvals = 3000;    // and never takes more evaluations than this

// M^-1 / (M^-1)_11 = [[m00, 0, m02], [0, 1, 0], [m20, 0, m22]].
struct Entries {
  double m00, m02, m20, m22;
};

inline double det(const Entries& e) { return e.m00 * e.m22 - e.m02 * e.m20; }

// The LTC's density at a unit direction l.
inline double ltc_density(const Entries& e, double det_e, Vec3<double> l) {
  const double z = e.m20 * l.x + e.m22 * l.z;
  if (!(z > 0.0)) return 0.0;
  const double x = e.m00 * l.x + e.m02 * l.z;
  const double len2 = x * x + l.y * l.y + z * z;
  return z * det_e / (kPi * (len2 * len2));
}

// int_{mu > 0} D dw, exactly: the plane mu = 0 maps to the plane through the origin with normal (-m20, 0, m00) in the
// cosine's space, and a clamped cosine holds (1 + cos gamma) / 2 of its mass on one side of a plane tilted by gamma
// (the sky view factor of a tilted surface). It is the full-hemisphere emitter's I_ltc / pi.
inline double upper_mass(const Entries& e) {
  return 0.5 * (1.0 + e.m00 / sqrt(e.m00 * e.m00 + e.m20 * e.m20));
}

// --- The lobe ---------------------------------------------------------------------------------------------------

struct Lobe {
  SpecGeometry g;
  double g_v;          // sqrt(mu_v^2 + alpha^2 s_v^2), the view's factor of V
  double e1;           // int D V mu_l dw_l with F = 1 (= A + B)
  Vec3<double> mean;   // the unit mean direction (y = 0)
};

// (E1, int f l_x, int f l_z) at disk point (rho, phi) of the half disk, times measure.
inline Vals<3> moment_integrand(const SpecGeometry& g, double rho, double measure, double phi) {
  double sp, cp;
  pmath::sincos(phi, &sp, &cp);
  VndfReflection r;
  if (!vndf_reflect(g, rho * cp, rho * sp, &r) || !(r.l.z > 0.0)) return {0.0, 0.0, 0.0};
  const double w = measure * vndf_weight(g, r.l);
  return {w, w * r.l.x, w * r.l.z};
}

inline Lobe make_lobe(double t, double r) {
  Lobe lb;
  lb.g = spec_geometry(t, max(r, kRMin));
  const double a = lb.g.alpha;
  lb.g_v = sqrt(lb.g.mu_v * lb.g.mu_v + (a * a) * (lb.g.v.x * lb.g.v.x));
  const Vals<3> m = vndf_disk_integral<3>(lb.g, kMomentTol, moment_integrand);
  lb.e1 = m[0];
  // At normal view the half disk is not split across the plane of v and n (tables_ref.h), but the lobe is
  // rotation-invariant: its mean is n.
  if (t > 0.0) {
    const double len = sqrt(m[1] * m[1] + m[2] * m[2]);
    lb.mean = {m[1] / len, 0.0, m[2] / len};
  } else {
    lb.mean = {0.0, 0.0, 1.0};
  }
  return lb;
}

// GGX D of the half vector h = v + l (unnormalized; the tangent form of bsdf.h ggx_d) for h_z > 0.
inline double lobe_d(const Lobe& lb, Vec3<double> h) {
  const double a2 = lb.g.alpha * lb.g.alpha;
  const double c2 = h.z * h.z, s2 = h.x * h.x + h.y * h.y;
  const double den = a2 * c2 + s2;
  if (!(den > 0.0)) return 0.0;
  const double q = (c2 + s2) / den;
  return a2 * (q * q) / kPi;
}

// f^(l) = D V mu_l / E1 at a unit direction l.
inline double lobe_density(const Lobe& lb, Vec3<double> l) {
  const double mu_l = l.z;
  if (!(mu_l > 0.0)) return 0.0;
  const double a2 = lb.g.alpha * lb.g.alpha;
  const double g_l = sqrt(mu_l * mu_l + a2 * (l.x * l.x + l.y * l.y));
  return lobe_d(lb, lb.g.v + l) * mu_l / (2.0 * (mu_l * lb.g_v + lb.g.mu_v * g_l) * lb.e1);
}

// The density of lobe_sample over the sphere: D(h) G1(v) / (4 mu_v) = D(h) / (2 (mu_v + g_v)), finite at mu_v = 0.
inline double lobe_pdf(const Lobe& lb, Vec3<double> l) {
  const Vec3<double> h = lb.g.v + l;
  if (!(h.z > 0.0)) return 0.0;
  return lobe_d(lb, h) / (2.0 * (lb.g.mu_v + lb.g_v));
}

// The reflection of a visible normal (Heitz 2018) at (u1, u2) in [0, 1]^2; the zero vector when degenerate.
inline Vec3<double> lobe_sample(const Lobe& lb, double u1, double u2) {
  double sp, cp;
  pmath::sincos(2.0 * kPi * u2, &sp, &cp);
  const double rho = sqrt(u1);
  VndfReflection r;
  if (!vndf_reflect(lb.g, rho * cp, rho * sp, &r)) return {0.0, 0.0, 0.0};
  return r.l;
}

// --- The LTC and its fit ----------------------------------------------------------------------------------------

struct Params {
  double m11, m22, m13, psi;
};

struct Ltc {
  Entries e;
  double det_e;
  double m00, m02, m11, m20, m22;  // M = [X Y Z] K: [[m00, 0, m02], [0, m11, 0], [m20, 0, m22]]
};

// The LTC of parameters p about the mean direction w_bar (unit, y = 0): Z = w_bar turned by p.psi about Y.
inline Ltc make_ltc(const Params& p, Vec3<double> mean) {
  double sp, cp;
  pmath::sincos(p.psi, &sp, &cp);
  const double lx = cp * mean.x + sp * mean.z, lz = cp * mean.z - sp * mean.x;
  Ltc l;
  l.e = {p.m22 * (lz - p.m13 * lx) / p.m11, -p.m22 * (lx + p.m13 * lz) / p.m11, p.m22 * lx, p.m22 * lz};
  l.det_e = det(l.e);
  l.m00 = p.m11 * lz;
  l.m02 = p.m13 * lz + lx;
  l.m11 = p.m22;
  l.m20 = -p.m11 * lx;
  l.m22 = lz - p.m13 * lx;
  return l;
}

// A cosine-distributed direction at (u1, u2) mapped through M.
inline Vec3<double> ltc_sample(const Ltc& l, double u1, double u2) {
  double sp, cp;
  pmath::sincos(2.0 * kPi * u2, &sp, &cp);
  const double rho = sqrt(u1), z = sqrt(max(0.0, 1.0 - u1));
  const Vec3<double> w{l.m00 * (rho * cp) + l.m02 * z, l.m11 * (rho * sp), l.m20 * (rho * cp) + l.m22 * z};
  return (1.0 / norm(w)) * w;
}

// The lobe at one node with its fixed visible-normal samples.
struct Target {
  Lobe lobe;
  std::vector<Vec3<double>> l;
  std::vector<double> f, p;  // f^ and lobe_pdf at l
};

inline Target make_target(double t, double r) {
  Target tg;
  tg.lobe = make_lobe(t, r);
  const int n = kFitGrid;
  tg.l.resize(static_cast<size_t>(n) * n);
  tg.f.resize(tg.l.size());
  tg.p.resize(tg.l.size());
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const size_t k = static_cast<size_t>(j) * n + i;
      tg.l[k] = lobe_sample(tg.lobe, (i + 0.5) / n, (j + 0.5) / n);
      tg.f[k] = lobe_density(tg.lobe, tg.l[k]);
      tg.p[k] = lobe_pdf(tg.lobe, tg.l[k]);
    }
  }
  return tg;
}

// fitLTC's error: int |f^ - D|^3 dw by the balance heuristic over both strategies' midpoint grids.
inline double fit_error(const Target& tg, const Ltc& ltc) {
  const int n = kFitGrid;
  double err = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const Vec3<double> l = ltc_sample(ltc, (i + 0.5) / n, (j + 0.5) / n);
      const double d = ltc_density(ltc.e, ltc.det_e, l), p = lobe_pdf(tg.lobe, l);
      if (d + p > 0.0) {
        const double e = abs(lobe_density(tg.lobe, l) - d);
        err += e * e * e / (d + p);
      }
      const size_t k = static_cast<size_t>(j) * n + i;
      const double d2 = ltc_density(ltc.e, ltc.det_e, tg.l[k]);
      if (d2 + tg.p[k] > 0.0) {
        const double e = abs(tg.f[k] - d2);
        err += e * e * e / (d2 + tg.p[k]);
      }
    }
  }
  return err / (static_cast<double>(n) * n);
}

// Nelder-Mead (reflection 1, expansion 2, contractions 1/2, shrink 1/2; Lagarias et al. 1998's acceptance rules) from
// x0 with initial steps `step`: deterministic (ties keep the earlier vertex). Stops when every vertex is within xtol of
// the best in every coordinate, or after max_evals evaluations.
template <int N, class F>
std::array<double, N> nelder_mead(const F& f, const std::array<double, N>& x0, const std::array<double, N>& step,
                                  double xtol, int max_evals, int* evals, double* f_best) {
  using X = std::array<double, N>;
  std::array<X, N + 1> x;
  std::array<double, N + 1> fx;
  x[0] = x0;
  fx[0] = f(x0);
  int ne = 1;
  for (int i = 0; i < N; ++i) {
    x[i + 1] = x0;
    x[i + 1][i] += step[i];
    fx[i + 1] = f(x[i + 1]);
    ++ne;
  }
  const auto order = [&] {  // insertion sort by value, stable
    for (int i = 1; i <= N; ++i) {
      for (int k = i; k > 0 && fx[k] < fx[k - 1]; --k) {
        std::swap(fx[k], fx[k - 1]);
        std::swap(x[k], x[k - 1]);
      }
    }
  };
  const auto along = [&](const X& c, const X& w, double s) {  // c + s (w - c)
    X y;
    for (int k = 0; k < N; ++k) y[k] = c[k] + s * (w[k] - c[k]);
    return y;
  };
  for (;;) {
    order();
    double size = 0.0;
    for (int i = 1; i <= N; ++i) {
      for (int k = 0; k < N; ++k) size = max(size, abs(x[i][k] - x[0][k]));
    }
    if (size <= xtol || ne >= max_evals) break;
    X c{};
    for (int i = 0; i < N; ++i) {
      for (int k = 0; k < N; ++k) c[k] += x[i][k];
    }
    for (int k = 0; k < N; ++k) c[k] /= N;
    const X xr = along(c, x[N], -1.0);
    const double fr = f(xr);
    ++ne;
    if (fr < fx[0]) {
      const X xe = along(c, x[N], -2.0);
      const double fe = f(xe);
      ++ne;
      if (fe < fr) {
        x[N] = xe;
        fx[N] = fe;
      } else {
        x[N] = xr;
        fx[N] = fr;
      }
      continue;
    }
    if (fr < fx[N - 1]) {
      x[N] = xr;
      fx[N] = fr;
      continue;
    }
    const bool outside = fr < fx[N];
    const X xc = outside ? along(c, xr, 0.5) : along(c, x[N], 0.5);
    const double fc = f(xc);
    ++ne;
    if (outside ? fc <= fr : fc < fx[N]) {
      x[N] = xc;
      fx[N] = fc;
      continue;
    }
    for (int i = 1; i <= N; ++i) {
      x[i] = along(x[0], x[i], 0.5);
      fx[i] = f(x[i]);
      ++ne;
    }
  }
  *evals = ne;
  *f_best = fx[0];
  return x[0];
}

// Golden-section minimum of f on [a, b] to width tol (f assumed unimodal there).
template <class F>
double golden_section(const F& f, double a, double b, double tol, int* evals) {
  const double g = 0.618033988749894848204586834365638118;
  double c = b - g * (b - a), d = a + g * (b - a);
  double fc = f(c), fd = f(d);
  int ne = 2;
  while (b - a > tol) {
    if (fc <= fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - g * (b - a);
      fc = f(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + g * (b - a);
      fd = f(d);
    }
    ++ne;
  }
  *evals = ne;
  return 0.5 * (a + b);
}

struct NodeFit {
  Entries e;
  Params p;
  double err;       // fit_error at the result
  int evals;        // objective evaluations
  bool at_bracket;  // the isotropic fit ended at its bracket's edge (a failure)
};

// The isotropic fit at normal view: m11 = m22 = m, m13 = psi = 0 and w_bar = n, by golden section in ln m over
// ln alpha -+ 3. Its entries are exactly diag(1, 1, m): M^-1 = diag(1/m, 1/m, 1) normalized by its middle entry.
inline NodeFit fit_isotropic(const Target& tg) {
  NodeFit nf;
  const double la = pmath::log(tg.lobe.g.alpha), lo = la - 3.0, hi = la + 3.0;
  const auto iso = [&](double y) {
    const double m = pmath::exp(y);
    return fit_error(tg, make_ltc({m, m, 0.0, 0.0}, {0.0, 0.0, 1.0}));
  };
  const double y = golden_section(iso, lo, hi, kFitXTol, &nf.evals);
  nf.at_bracket = y - lo < 1e-3 || hi - y < 1e-3;
  const double m = pmath::exp(y);
  nf.p = {m, m, 0.0, 0.0};
  nf.e = {1.0, 0.0, 0.0, m};
  nf.err = iso(y);
  return nf;
}

// The fitted nodes of roughness row r at views t_i = i / (n - 1), i = 0 .. n - 1: the isotropic fit at t = 0, then
// Nelder-Mead from the previous node's parameters with steps 0.05.
inline std::vector<NodeFit> fit_row(double r, int n) {
  std::vector<NodeFit> row(n);
  for (int i = 0; i < n; ++i) {
    const Target tg = make_target(static_cast<double>(i) / (n - 1), r);
    if (i == 0) {
      row[0] = fit_isotropic(tg);
      continue;
    }
    NodeFit& nf = row[i];
    nf.at_bracket = false;
    const Params seed = row[i - 1].p;
    const double scale = seed.m11;
    const auto params = [&](const std::array<double, 4>& x) {
      return Params{pmath::exp(x[0]), pmath::exp(x[1]), x[2] * scale, x[3]};
    };
    const auto objective = [&](const std::array<double, 4>& x) {
      return fit_error(tg, make_ltc(params(x), tg.lobe.mean));
    };
    const std::array<double, 4> x =
        nelder_mead<4>(objective, {pmath::log(seed.m11), pmath::log(seed.m22), seed.m13 / scale, seed.psi},
                       {0.05, 0.05, 0.05, 0.05}, kFitXTol, kFitMaxEvals, &nf.evals, &nf.err);
    nf.p = params(x);
    nf.e = make_ltc(nf.p, tg.lobe.mean).e;
  }
  return row;
}

// --- Error measures (T-TAB) ---------------------------------------------------------------------------------------

// Error measures of tabulated entries e at the lobe lb, by the balance heuristic over grid^2 midpoints drawn through
// both the lobe's visible normals and the LTC (every integrand below is bounded by its MIS weight's denominator, and
// both strategies follow the lobes, however narrow). Caps are about the lobe's mean direction w_bar, with edges
// w0 sqrt(2)^k (w0 = alpha / 4) up to pi, clipped to mu > 0 (an emitter about w_bar; the last cap is the whole
// hemisphere):
// - cap25 / cap10: max |C_ltc / C_lobe - 1| over the caps that hold >= 25 % / 10 % of the lobe: the relative error of
//   L_spec for such an emitter;
// - tv: 1/2 int |D - f^| dw, the largest absolute error of I_ltc / pi over all emitters;
// - upper: int_{mu > 0} D dw of the plain LTC (exact), and upper_quad its estimate here;
// - kink: with `other` (the bilinear table), max |ln(C_other / C_e)| over the caps holding >= 25 %;
// - lobe_total: the estimate of int f^ dw = 1.
struct Metrics {
  double cap25, cap10, tv, upper, upper_quad, kink, lobe_total;
};

inline Metrics fit_metrics(const Lobe& lb, const Entries& e, const Entries* other = nullptr, int grid = 256) {
  const double de = det(e), dother = other ? det(*other) : 0.0;
  Ltc ltc;  // sampling through e: M = (M^-1)^-1 up to scale
  ltc.e = e;
  ltc.det_e = de;
  ltc.m00 = e.m22;
  ltc.m02 = -e.m02;
  ltc.m11 = de;
  ltc.m20 = -e.m20;
  ltc.m22 = e.m00;
  std::vector<double> edges;
  for (double w = 0.25 * lb.g.alpha; w < kPi; w *= 1.4142135623730951) edges.push_back(w);
  edges.push_back(kPi);
  const size_t nb = edges.size();
  std::vector<double> b_lobe(nb, 0.0), b_ltc(nb, 0.0), b_other(nb, 0.0);
  double tv = 0.0, upper = 0.0, total = 0.0;
  const double inv_n = 1.0 / (static_cast<double>(grid) * grid);
  for (int s = 0; s < 2; ++s) {
    for (int j = 0; j < grid; ++j) {
      for (int i = 0; i < grid; ++i) {
        const double u1 = (i + 0.5) / grid, u2 = (j + 0.5) / grid;
        const Vec3<double> l = s == 0 ? lobe_sample(lb, u1, u2) : ltc_sample(ltc, u1, u2);
        if (!(dot(l, l) > 0.0)) continue;
        const double d = ltc_density(e, de, l), p = lobe_pdf(lb, l);
        if (!(d + p > 0.0)) continue;
        const double w = inv_n / (d + p);
        const double f = lobe_density(lb, l);
        total += w * f;
        tv += 0.5 * w * abs(d - f);
        if (!(l.z > 0.0)) continue;
        upper += w * d;
        const double psi = pmath::acos(clamp(dot(l, lb.mean), -1.0, 1.0));
        size_t k = 0;
        while (k + 1 < nb && edges[k] < psi) ++k;
        b_lobe[k] += w * f;
        b_ltc[k] += w * d;
        if (other) b_other[k] += w * ltc_density(*other, dother, l);
      }
    }
  }
  Metrics mt{0.0, 0.0, tv, upper_mass(e), upper, 0.0, total};
  double cl = 0.0, cd = 0.0, co = 0.0, all = 0.0;
  for (size_t k = 0; k < nb; ++k) all += b_lobe[k];
  for (size_t k = 0; k < nb; ++k) {
    cl += b_lobe[k];
    cd += b_ltc[k];
    co += b_other[k];
    if (!(cl > 0.0)) continue;
    const double share = cl / all, rel = abs(cd / cl - 1.0);
    if (share >= 0.10) mt.cap10 = max(mt.cap10, rel);
    if (share >= 0.25) {
      mt.cap25 = max(mt.cap25, rel);
      if (other) mt.kink = max(mt.kink, abs(pmath::log(co / cd)));
    }
  }
  return mt;
}

}  // namespace balla::bsdf_ref::ltc
