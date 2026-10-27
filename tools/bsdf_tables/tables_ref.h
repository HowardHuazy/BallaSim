// fp64 references of the BSDF tables (p2f.md §4.3, §6.1 T-TAB), shared by the generator (gen_tables.cpp) and
// tests/render/test_bsdf.cpp: the specular albedo split A, B by adaptive Gauss-Kronrod quadrature, and the
// interpolating cubic B-spline prefilter (Unser 1999). Host only and deterministic: +, -, *, /, sqrt and pmath under
// the FP policy, so a regeneration is bitwise equal on any IEEE host.
//
// Quadrature. A = int D V (1 - (1 - u)^5) mu_l dw_l and B = int D V (1 - u)^5 mu_l dw_l (u = w_v . h). With the
// visible-normal density D_v(h) = G1(v) (v . h)^+ D(h) / mu_v and dw_l = 4 (v . h) dw_h, D V mu_l dw_l =
// D_v(h) (G2 / G1(v)) dw_h, so A and B are the averages of (G2 / G1(v)) (1 - (1 - u)^5) and (G2 / G1(v)) (1 - u)^5 over
// visible normals. Heitz's (2018) exact VNDF map takes the unit disk (area density 1/pi) to D_v, so the integrand is
// bounded by 1 and smooth up to the horizon kink of w_l, at every alpha: the lobe's width never has to be resolved,
// which a quadrature over (theta_l, phi_l) would need at alpha ~ 1e-4 (a departure from p2f.md §4.3's wording, same
// integral). The map is symmetric in the disk coordinate across the plane of v and n, so the half disk is integrated
// and doubled. Nested globally adaptive G7-K15 (QUADPACK qk15/qag) over the disk's polar angle (radius sin theta) and
// azimuth.
#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "core/math/scalar.h"
#include "core/math/vec.h"
#include "core/pmath/trig.h"

namespace balla::bsdf_ref {

template <int N>
using Vals = std::array<double, N>;

namespace detail {

// QUADPACK qk15: Kronrod abscissae (xgk[1], xgk[3], xgk[5] and the centre are the 7-point Gauss nodes) and weights.
inline constexpr double kXgk[8] = {0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                                   0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                                   0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
                                   0.207784955007898467600689403773245, 0.0};
inline constexpr double kWgk[8] = {0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                                   0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                                   0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                                   0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
inline constexpr double kWg[4] = {0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
                                  0.381830050505118944950369775488975, 0.417959183673469387755102040816327};

// One G7-K15 panel with QUADPACK's (qk15) error estimate per component, resasc min(1, (200 |K - G| / resasc)^1.5),
// and the maximum over the components.
template <int N, class F>
Vals<N> gk15(const F& f, double a, double b, double* err) {
  const double c = 0.5 * (a + b), h = 0.5 * (b - a);
  Vals<N> fv[15];
  fv[7] = f(c);
  for (int j = 0; j < 7; ++j) {
    const double dx = h * kXgk[j];
    fv[j] = f(c - dx);
    fv[14 - j] = f(c + dx);
  }
  Vals<N> rk{};
  double e = 0.0;
  for (int k = 0; k < N; ++k) {
    double resk = kWgk[7] * fv[7][k], resg = kWg[3] * fv[7][k];
    for (int j = 0; j < 7; ++j) {
      const double s = fv[j][k] + fv[14 - j][k];
      resk += kWgk[j] * s;
      if (j % 2 == 1) resg += kWg[j / 2] * s;
    }
    const double mean = 0.5 * resk;
    double resasc = kWgk[7] * abs(fv[7][k] - mean);
    for (int j = 0; j < 7; ++j) resasc += kWgk[j] * (abs(fv[j][k] - mean) + abs(fv[14 - j][k] - mean));
    resasc *= abs(h);
    double ek = abs((resk - resg) * h);
    if (resasc > 0.0 && ek > 0.0) {
      const double q = min(1.0, 200.0 * ek / resasc);
      ek = resasc * (q * sqrt(q));
    }
    e = max(e, ek);
    rk[k] = resk * h;
  }
  *err = e;
  return rk;
}

}  // namespace detail

// Globally adaptive G7-K15 (QUADPACK qag): the panel with the largest error estimate is bisected until the summed
// estimate is <= tol or max_panels panels exist. Deterministic: ties go to the first panel, sums run in panel order.
template <int N, class F>
Vals<N> integrate(const F& f, double a, double b, double tol, int max_panels = 200) {
  struct Panel {
    double a, b, err;
    Vals<N> val;
  };
  std::vector<Panel> panels;
  panels.reserve(max_panels);
  Panel p0{a, b, 0.0, {}};
  p0.val = detail::gk15<N>(f, a, b, &p0.err);
  panels.push_back(p0);
  for (;;) {
    double total = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < panels.size(); ++i) {
      total += panels[i].err;
      if (panels[i].err > panels[worst].err) worst = i;
    }
    if (total <= tol || static_cast<int>(panels.size()) >= max_panels) break;
    const Panel w = panels[worst];
    const double m = 0.5 * (w.a + w.b);
    Panel l{w.a, m, 0.0, {}}, r{m, w.b, 0.0, {}};
    l.val = detail::gk15<N>(f, l.a, l.b, &l.err);
    r.val = detail::gk15<N>(f, r.a, r.b, &r.err);
    panels[worst] = l;
    panels.push_back(r);
  }
  Vals<N> s{};
  for (const Panel& p : panels) {
    for (int k = 0; k < N; ++k) s[k] += p.val[k];
  }
  return s;
}

inline constexpr double kPi = 3.14159265358979323846264338327950288;

// The view of table coordinate t = sqrt(1 - sqrt(mu_v)) (bsdf.h) at roughness coordinate r = sqrt(alpha), n = +z,
// w_v in the xz plane, and Heitz's (2018) VNDF frame for it.
struct SpecGeometry {
  double alpha, mu_v;
  Vec3<double> v, vh, t1, t2;
  double s;    // the warp's blend (1 + vh.z) / 2
  double g1v;  // G1(v), the masking of the view
};

inline SpecGeometry spec_geometry(double t, double r) {
  SpecGeometry g;
  g.alpha = r * r;
  const double sq = (1.0 - t) * (1.0 + t);  // sqrt(mu_v)
  g.mu_v = sq * sq;
  const double one_minus_mu = (t * t) * (2.0 - t * t);
  const double s_v = sqrt(one_minus_mu * (1.0 + g.mu_v));  // sqrt(1 - mu_v^2) without cancellation
  g.v = {s_v, 0.0, g.mu_v};
  const Vec3<double> vs{g.alpha * s_v, 0.0, g.mu_v};
  g.vh = (1.0 / norm(vs)) * vs;
  if (g.vh.x > 0.0) {
    g.t1 = {0.0, 1.0, 0.0};
    g.t2 = {-g.vh.z, 0.0, g.vh.x};  // vh x t1
  } else {
    g.t1 = {1.0, 0.0, 0.0};
    g.t2 = {0.0, 1.0, 0.0};
  }
  g.s = 0.5 * (1.0 + g.vh.z);
  const double a2s2 = g.alpha * g.alpha * (s_v * s_v);
  g.g1v = 2.0 * g.mu_v / (g.mu_v + sqrt(g.mu_v * g.mu_v + a2s2));
  return g;
}

// The reflection through Heitz's (2018) visible-normal map at disk point (t1, t2) of the unit disk: u = w_v . h
// (clamped to 1) and l = 2 u h - w_v. False for a degenerate normal or u <= 0; l may lie below the horizon.
struct VndfReflection {
  Vec3<double> l;
  double u;
};

inline bool vndf_reflect(const SpecGeometry& g, double t1, double t2, VndfReflection* out) {
  const double t2w = (1.0 - g.s) * sqrt(max(0.0, (1.0 - t1) * (1.0 + t1))) + g.s * t2;
  const double nz = sqrt(max(0.0, 1.0 - t1 * t1 - t2w * t2w));
  const Vec3<double> nh = t1 * g.t1 + t2w * g.t2 + nz * g.vh;
  const Vec3<double> hs{g.alpha * nh.x, g.alpha * nh.y, max(0.0, nh.z)};
  const double hn = norm(hs);
  if (!(hn > 0.0)) return false;
  const Vec3<double> h = (1.0 / hn) * hs;
  out->u = min(1.0, dot(g.v, h));
  if (!(out->u > 0.0)) return false;
  out->l = (2.0 * out->u) * h - g.v;
  return true;
}

// G2 / G1(v) at a reflected direction l above the horizon (mu_l > 0): the integrand's weight, <= 1.
inline double vndf_weight(const SpecGeometry& g, Vec3<double> l) {
  const double mu_l = l.z;
  const double a2s2 = g.alpha * g.alpha * (l.x * l.x + l.y * l.y);
  const double lambda_l = a2s2 / (2.0 * mu_l * (sqrt(mu_l * mu_l + a2s2) + mu_l));
  return 1.0 / (1.0 + lambda_l * g.g1v);
}

// (A, B) integrand at disk point (rho, phi) of the half disk t1 >= 0, times `measure` (the area element's factor).
inline Vals<2> spec_integrand(const SpecGeometry& g, double rho, double measure, double phi) {
  double sp, cp;
  pmath::sincos(phi, &sp, &cp);
  VndfReflection r;
  if (!vndf_reflect(g, rho * cp, rho * sp, &r) || !(r.l.z > 0.0)) return {0.0, 0.0};
  const double w = vndf_weight(g, r.l);  // G2 / G1(v)
  const double gu = 1.0 - r.u, gu2 = gu * gu, w5 = gu2 * gu2 * gu;
  return {measure * w * (1.0 - w5), measure * w * w5};
}

// The half-disk integral of integrand(g, rho, measure, phi) -> Vals<N>, doubled: rho = sin(theta). Near the rim, where
// the visible normal turns away from v (nz -> 0), the integrand falls to 0 over a band of width ~alpha in theta but
// ~alpha^2 in rho; a breakpoint at pi/2 - 4 alpha gives that band its own panels. Area density 1/pi on the disk, times
// 2 for the mirror half: rho drho / pi = sin cos dtheta / pi. The halves are mirror images across the plane of v and n
// only for t > 0 (g.t1 = y); at normal view they are so for rotation-invariant integrands.
template <int N, class F>
Vals<N> vndf_disk_integral(const SpecGeometry& g, double tol, const F& integrand) {
  const double inner_tol = 0.1 * tol;
  const auto ring = [&](double theta) {
    double st, ct;
    pmath::sincos(theta, &st, &ct);
    const double measure = 2.0 * st * ct / kPi;
    return integrate<N>([&](double phi) { return integrand(g, st, measure, phi); }, -0.5 * kPi, 0.5 * kPi,
                        inner_tol);
  };
  const double theta_b = 0.5 * kPi - min(0.25 * kPi, 4.0 * g.alpha);
  const Vals<N> lo = integrate<N>(ring, 0.0, theta_b, 0.5 * tol);
  const Vals<N> hi = integrate<N>(ring, theta_b, 0.5 * kPi, 0.5 * tol);
  Vals<N> s;
  for (int k = 0; k < N; ++k) s[k] = lo[k] + hi[k];
  return s;
}

// A and B at table coordinates (t, r) to absolute tolerance tol. r = 0 is the mirror limit: A = 1 - (1 - mu_v)^5,
// B = (1 - mu_v)^5.
inline Vals<2> spec_ab(double t, double r, double tol = 1e-8) {
  const SpecGeometry g = spec_geometry(t, r);
  if (!(g.alpha > 0.0)) {
    const double gm = 1.0 - g.mu_v, gm2 = gm * gm, w5 = gm2 * gm2 * gm;
    return {1.0 - w5, w5};
  }
  return vndf_disk_integral<2>(g, tol, spec_integrand);
}

// --- Interpolating cubic B-spline prefilter -----------------------------------------------------------------------

enum class Extension { kEven, kOdd, kNatural };

// Coefficients c[-1 .. n] (n + 2 values, stored ghost-shifted at 0 .. n + 1) of the uniform cubic B-spline that
// interpolates f[0 .. n - 1]: (c[i-1] + 4 c[i] + c[i+1]) / 6 = f[i], with condition `start` at node 0 (even:
// c[-1] = c[1]; odd: c[-1] = -c[1], so S(0) = 0 needs f[0] = 0; natural: S''(0) = 0) and natural at node n - 1.
inline void fit_axis(const double* f, int n, Extension start, double* c) {
  std::vector<double> lo(n, 1.0), di(n, 4.0), up(n, 1.0), rhs(n);
  for (int i = 0; i < n; ++i) rhs[i] = 6.0 * f[i];
  lo[0] = 0.0;
  up[0] = start == Extension::kEven ? 2.0 : 0.0;
  di[0] = start == Extension::kNatural ? 6.0 : 4.0;
  lo[n - 1] = 0.0;
  up[n - 1] = 0.0;
  di[n - 1] = 6.0;
  std::vector<double> cp(n), dp(n);  // Thomas algorithm (diagonally dominant)
  cp[0] = up[0] / di[0];
  dp[0] = rhs[0] / di[0];
  for (int i = 1; i < n; ++i) {
    const double m = di[i] - lo[i] * cp[i - 1];
    cp[i] = up[i] / m;
    dp[i] = (rhs[i] - lo[i] * dp[i - 1]) / m;
  }
  double* x = c + 1;
  x[n - 1] = dp[n - 1];
  for (int i = n - 2; i >= 0; --i) x[i] = dp[i] - cp[i] * x[i + 1];
  switch (start) {
    case Extension::kEven: c[0] = x[1]; break;
    case Extension::kOdd: c[0] = -x[1]; break;
    case Extension::kNatural: c[0] = 2.0 * x[0] - x[1]; break;
  }
  c[n + 1] = 2.0 * x[n - 1] - x[n - 2];
}

// The (n + 2)^2 coefficients, [r][x] ghost-shifted, of the tensor spline through nodes[j * n + i] at (x_i, r_j):
// condition x_start at x = 0, natural at x = 1, r = 0 and r = 1. The 1-D conditions are linear, so filtering the rows
// and then every column (ghosts included) keeps both.
inline std::vector<double> fit_table(const std::vector<double>& nodes, int n, Extension x_start) {
  const int m = n + 2;
  std::vector<double> rows(static_cast<size_t>(n) * m), out(static_cast<size_t>(m) * m), col(n), cc(m);
  for (int j = 0; j < n; ++j) {
    fit_axis(&nodes[static_cast<size_t>(j) * n], n, x_start, &rows[static_cast<size_t>(j) * m]);
  }
  for (int k = 0; k < m; ++k) {
    for (int j = 0; j < n; ++j) col[j] = rows[static_cast<size_t>(j) * m + k];
    fit_axis(col.data(), n, Extension::kNatural, cc.data());
    for (int j = 0; j < m; ++j) out[static_cast<size_t>(j) * m + k] = cc[j];
  }
  return out;
}

}  // namespace balla::bsdf_ref
