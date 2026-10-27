// The BSDF library of radiance v1 (p2f.md §4.3, F3a, F3c): the C2 B-spline tables, the A/B tables against fp64
// quadrature (T-TAB, §6.1), the lobe's forms, reciprocity and energy, the floors and caps, the fp32 accuracy gate
// (§6.3), and the LTC tables (T-TAB's LTC rows and view sweep). No renderer code path uses bsdf.h yet; host = device
// parity is in tests/gpu/test_parity.cpp.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "render/shade/bsdf.h"
#include "render/shade/bsdf_tables.h"
#include "render/shade/bspline_table.h"
#include "ltc_ref.h"
#include "tables_ref.h"

using namespace balla;
using namespace balla::render;
namespace bs = balla::render::bsdf;
namespace ref = balla::bsdf_ref;
namespace ltc = balla::bsdf_ref::ltc;

namespace {

constexpr int kN = kTableNodes;

double node(int i) { return static_cast<double>(i) / (kN - 1); }

double spline(const float* c, double x, double r) { return bspline_eval(c, bspline_tap(x), bspline_tap(r)); }

// The fp32 coefficients of the tensor spline through f at the nodes, as the generator makes them.
std::vector<float> fit(const std::function<double(double, double)>& f, ref::Extension x_start) {
  std::vector<double> nodes(static_cast<size_t>(kN) * kN);
  for (int j = 0; j < kN; ++j) {
    for (int i = 0; i < kN; ++i) nodes[j * kN + i] = f(node(i), node(j));
  }
  const std::vector<double> c = ref::fit_table(nodes, kN, x_start);
  return {c.begin(), c.end()};
}

Vec3<double> unit(Vec3<double> a) { return (1.0 / norm(a)) * a; }

// A unit vector at polar cosine mu and azimuth phi about n (any orthonormal frame of n).
Vec3<double> about(Vec3<double> n, double mu, double phi) {
  const Vec3<double> a = std::abs(n.x) < 0.9 ? Vec3<double>{1.0, 0.0, 0.0} : Vec3<double>{0.0, 1.0, 0.0};
  const Vec3<double> t = unit(cross(n, a)), b = cross(n, t);
  const double s = std::sqrt(std::max(0.0, 1.0 - mu * mu));
  return unit((s * std::cos(phi)) * t + (s * std::sin(phi)) * b + mu * n);
}

template <class T>
Vec3<T> to(Vec3<double> a) {
  return {T(a.x), T(a.y), T(a.z)};
}

}  // namespace

TEST_CASE("B-spline tables interpolate their nodes with even, odd and natural boundaries, C2 between cells") {
  const auto f_even = [](double x, double r) { return std::cos(2.5 * x) * (1.0 + 0.3 * r + 0.2 * r * r); };
  const auto f_odd = [](double x, double r) { return std::sin(2.5 * x) * (1.0 + 0.3 * r); };
  const std::vector<float> ce = fit(f_even, ref::Extension::kEven), co = fit(f_odd, ref::Extension::kOdd);
  REQUIRE(ce.size() == static_cast<size_t>(kTableSize));

  double node_err = 0.0, inner_err = 0.0, edge_err = 0.0;
  for (int j = 0; j < kN; ++j) {
    for (int i = 0; i < kN; ++i) {
      node_err = std::max(node_err, std::abs(spline(ce.data(), node(i), node(j)) - f_even(node(i), node(j))));
      node_err = std::max(node_err, std::abs(spline(co.data(), node(i), node(j)) - f_odd(node(i), node(j))));
      if (i == kN - 1 || j == kN - 1) continue;
      const double x = node(i) + 0.5 / (kN - 1), r = node(j) + 0.5 / (kN - 1);
      const double e = std::max(std::abs(spline(ce.data(), x, r) - f_even(x, r)),
                                std::abs(spline(co.data(), x, r) - f_odd(x, r)));
      // The even and odd conditions are exact for these functions, so only the natural ends (x = 1, r = 0, 1) carry
      // an O(h^2) error, which decays by ~(2 - sqrt 3) per cell away from them.
      double& err = x < 0.75 && r > 0.25 && r < 0.75 ? inner_err : edge_err;
      err = std::max(err, e);
    }
  }
  INFO("node " << node_err << ", inner cells " << inner_err << ", cells near natural ends " << edge_err);
  CHECK(node_err <= 1e-6);   // fp32 coefficients
  CHECK(inner_err <= 1e-6);  // includes x -> 0: the even and odd extensions are right
  CHECK(edge_err <= 1e-3);

  const double h = 1e-3, r = 0.4;
  // Even at x = 0: zero slope, so S(h) - S(0) = O(h^2).
  CHECK(std::abs(spline(ce.data(), h, r) - spline(ce.data(), 0.0, r)) <= 1e-5);
  // Odd at x = 0: S(0) = 0 and S''(0) = 0.
  CHECK(std::abs(spline(co.data(), 0.0, r)) <= 1e-12);
  CHECK(std::abs(spline(co.data(), 2 * h, r) - 2 * spline(co.data(), h, r) + spline(co.data(), 0.0, r)) <= 1e-7);
  // Natural at x = 1: S''(1) = 0 (f'' = 5 there, so a second difference of f would be 5e-6).
  CHECK(std::abs(spline(ce.data(), 1.0, r) - 2 * spline(ce.data(), 1.0 - h, r) + spline(ce.data(), 1.0 - 2 * h, r)) <=
        5e-7);

  // C2 at every interior knot along x: one-sided second derivatives (in the cell fraction) from both cells agree.
  const BsplineTap<double> tr = bspline_tap(r);
  const auto s_at = [&](int cell, double f) { return bspline_eval(ce.data(), bspline_tap_at(cell, f), tr); };
  const double d = 1e-4;
  double c0 = 0.0, c1 = 0.0, c2 = 0.0;
  for (int k = 1; k <= kN - 2; ++k) {
    const double l0 = s_at(k - 1, 1.0), r0 = s_at(k, 0.0);
    const double l1 = (l0 - s_at(k - 1, 1.0 - d)) / d, r1 = (s_at(k, d) - r0) / d;
    const double l2 = (l0 - 2 * s_at(k - 1, 1.0 - d) + s_at(k - 1, 1.0 - 2 * d)) / (d * d);
    const double r2 = (s_at(k, 2 * d) - 2 * s_at(k, d) + r0) / (d * d);
    c0 = std::max(c0, std::abs(l0 - r0));
    c1 = std::max(c1, std::abs(l1 - r1));
    c2 = std::max(c2, std::abs(l2 - r2));
  }
  INFO("knot jumps: value " << c0 << ", slope " << c1 << ", curvature " << c2 << " (per cell fraction)");
  CHECK(c0 <= 1e-12);
  CHECK(c1 <= 1e-6);  // one-sided difference error ~ d S''
  CHECK(c2 <= 1e-5);  // ~ d S'''
}

TEST_CASE("T-TAB: the A and B tables against fp64 quadrature") {
  const bs::BsdfTables t = bs::host_tables();
  // Nodes on a 10 x 10 subset (the generator's report checks all of them): fit and fp32 rounding only.
  double node_err = 0.0;
  for (int j = 0; j < kN; j += 7) {
    for (int i = 0; i < kN; i += 7) {
      const ref::Vals<2> q = ref::spec_ab(node(i), node(j));
      node_err = std::max(node_err, std::abs(spline(t.spec_a, node(i), node(j)) - q[0]));
      node_err = std::max(node_err, std::abs(spline(t.spec_b, node(i), node(j)) - q[1]));
    }
  }
  INFO("nodes: max |spline - quadrature| " << node_err);
  CHECK(node_err <= 1e-6);

  // Cell centres in the reflection policy's region r >= 0.2, denser towards grazing (x -> 1), where A and B are
  // steepest: within 1e-3 absolute (T-TAB, gated for r >= 0.2: a departure, see below). The full map is
  // gen_tables --report's.
  double cell_err = 0.0;
  for (int j : {13, 20, 30, 40, 50, 62}) {
    for (int i : {0, 10, 20, 30, 40, 50, 55, 58, 60, 61, 62}) {
      const double x = node(i) + 0.5 / (kN - 1), r = node(j) + 0.5 / (kN - 1);
      const ref::Vals<2> q = ref::spec_ab(x, r);
      cell_err = std::max(cell_err, std::abs(spline(t.spec_a, x, r) - q[0]));
      cell_err = std::max(cell_err, std::abs(spline(t.spec_b, x, r) - q[1]));
    }
  }
  INFO("cell centres, r >= 0.2: max |spline - quadrature| " << cell_err);
  CHECK(cell_err <= 1e-3);

  // Near-mirror rows r < 0.2, which only scene.allow_missing_reflections admits (O22's plane has r = 0.1): A and B vary
  // with mu_v / alpha at grazing, on a scale no 64-node axis resolves when alpha is small, so T-TAB's 1e-3 is not met
  // there (departure 2, reported to the lead). Tracked so these rows cannot get worse silently: the grazing cell
  // centres (gen_tables --report: 2.8e-2 at r 0.04), and the views a shading point can reach (mu_v,e >= delta_mu / 2,
  // so t <= t_max): within 1e-3 from r 0.09 up (O22's r 0.1 included), up to 5.5e-2 below r 0.085 (a dense sweep).
  double mirror_cells = 0.0;
  for (int j = 0; j < 13; ++j) {
    for (int i = 55; i < kN - 1; ++i) {
      const double x = node(i) + 0.5 / (kN - 1), r = node(j) + 0.5 / (kN - 1);
      const ref::Vals<2> q = ref::spec_ab(x, r);
      mirror_cells = std::max(mirror_cells, std::max(std::abs(spline(t.spec_a, x, r) - q[0]),
                                                     std::abs(spline(t.spec_b, x, r) - q[1])));
    }
  }
  const double t_max = std::sqrt(1.0 - std::sqrt(0.5 * bs::kDeltaMu));
  double reach_lo = 0.0, reach_hi = 0.0;  // r < 0.09 and r in [0.09, 0.2)
  for (double r : {0.015, 0.02, 0.04, 0.05, 0.065, 0.08, 0.09, 0.1, 0.12, 0.15, 0.19}) {
    for (double x : {0.95, 0.97, 0.975, 0.977, 0.98, 0.982, 0.985, 0.987, t_max}) {
      const ref::Vals<2> q = ref::spec_ab(x, r);
      double& worst = r < 0.09 ? reach_lo : reach_hi;
      worst = std::max(worst, std::abs(spline(t.spec_a, x, r) - q[0]));
      worst = std::max(worst, std::abs(spline(t.spec_b, x, r) - q[1]));
    }
  }
  INFO("near-mirror rows: grazing cell centres " << mirror_cells << "; reachable views " << reach_lo << " (r < 0.09), "
                                                 << reach_hi << " (0.09 <= r < 0.2)");
  CHECK(mirror_cells <= 3e-2);
  CHECK(reach_lo <= 6e-2);
  CHECK(reach_hi <= 1e-3);

  // The mirror row r = 0: B = (1 - mu_v)^5, with mu_v = (1 - t^2)^2.
  for (int i = 0; i < kN; ++i) {
    const double g = node(i) * node(i) * (2.0 - node(i) * node(i));  // 1 - mu_v
    CHECK(spline(t.spec_b, node(i), 0.0) == doctest::Approx(g * g * g * g * g).epsilon(1e-6).scale(1e-7));
  }

  // A, B >= 0 and A + B <= 1 (single scattering loses energy), including between the nodes, for r >= 0.04
  // (alpha >= 0.0016). Near-mirror rows overshoot near grazing, where A + B dips over a width ~alpha in mu_v that no
  // node resolves (max 1.003 at r < 0.032, mu_v ~ 0.006); spec_albedo's cap keeps E_s below 1 there. In the policy
  // region, at every reachable view (t <= t_max), A + B stays below the cap's start 1 - delta_E, so the cap is the
  // identity for every material there.
  double lo = 1.0, hi = 0.0, hi_mirror = 0.0, hi_policy = 0.0, capped_max = 0.0;
  for (int j = 0; j <= 256; ++j) {
    for (int i = 0; i <= 256; ++i) {
      const double x = i / 256.0, r = j / 256.0;
      const double a = spline(t.spec_a, x, r), b = spline(t.spec_b, x, r);
      lo = std::min(lo, std::min(a, b));
      double& top = r >= 0.04 ? hi : hi_mirror;
      top = std::max(top, a + b);
      if (r >= 0.2 && x <= t_max) hi_policy = std::max(hi_policy, a + b);
      capped_max = std::max(capped_max, bs::spec_albedo(bs::Fresnel<double>{1.0, 1.0}, bs::SpecAB<double>{a, b}));
    }
  }
  INFO("min(A, B) " << lo << ", max(A + B) " << hi << " (r >= 0.04), " << hi_mirror << " (r < 0.04), " << hi_policy
                    << " (r >= 0.2, reachable views); capped E_s of a white metal " << capped_max);
  CHECK(lo >= -1e-6);
  CHECK(hi <= 1.0 + 1e-6);
  CHECK(hi_mirror <= 1.005);
  CHECK(hi_policy <= 1.0 - bs::kDeltaE);
  CHECK(capped_max <= 1.0 - 0.5 * bs::kDeltaE);
}

TEST_CASE("GGX lobe: D normalization, the tangent-form visibility, reciprocity and the s = m = 0 limit") {
  // Projected area: 2 pi int_0^1 D(mu) mu dmu = 1.
  for (double alpha2 : {0.0016, 0.04, 0.25, 1.0}) {
    const auto f = [&](double mu) { return ref::Vals<1>{2.0 * ref::kPi * bs::ggx_d(alpha2, mu, 1.0 - mu * mu) * mu}; };
    CHECK(ref::integrate<1>(f, 0.0, 1.0, 1e-12)[0] == doctest::Approx(1.0).epsilon(1e-9));
  }

  std::mt19937_64 rng(23);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  // V in the tangent form equals G2 / (4 mu_l mu_v) with Cycles' Lambda = (sqrt(1 + alpha^2 tan^2) - 1) / 2.
  double v_err = 0.0;
  for (int i = 0; i < 10000; ++i) {
    const double mu_v = 0.01 + 0.99 * u01(rng), mu_l = 0.01 + 0.99 * u01(rng), alpha2 = std::pow(u01(rng), 2.0);
    const auto lambda = [&](double mu) { return 0.5 * (std::sqrt(1.0 + alpha2 * (1.0 / (mu * mu) - 1.0)) - 1.0); };
    const double g2 = 1.0 / (1.0 + lambda(mu_v) + lambda(mu_l));
    const double v = bs::smith_v(alpha2, mu_v, 1.0 - mu_v * mu_v, mu_l, 1.0 - mu_l * mu_l);
    v_err = std::max(v_err, std::abs(v / (g2 / (4.0 * mu_l * mu_v)) - 1.0));
  }
  CHECK(v_err <= 1e-12);

  // Reciprocity of the lobe, bitwise in both precisions, for mu_v, mu_l >= delta_mu.
  int asymmetric[2] = {};
  for (int i = 0; i < 20000; ++i) {
    const Vec3<double> n = unit({u01(rng) - 0.5, u01(rng) - 0.5, u01(rng) - 0.5});
    const Vec3<double> v = about(n, 0.002 + 0.998 * u01(rng), 2 * ref::kPi * u01(rng));
    const Vec3<double> l = about(n, 0.002 + 0.998 * u01(rng), 2 * ref::kPi * u01(rng));
    const double alpha2 = std::pow(0.2 + 0.8 * u01(rng), 4.0);
    const bs::Fresnel<double> fd = bs::fresnel_params(u01(rng), u01(rng), 0.1 * u01(rng));
    const double fwd = bs::lobe(alpha2, fd, n, v, bs::view_terms(n, v), l);
    const double bwd = bs::lobe(alpha2, fd, n, l, bs::view_terms(n, l), v);
    asymmetric[0] += fwd != bwd;
    const Vec3<float> nf = to<float>(n), vf = to<float>(v), lf = to<float>(l);
    const bs::Fresnel<float> ff{float(fd.f0), float(fd.f90)};
    asymmetric[1] += bs::lobe(float(alpha2), ff, nf, vf, bs::view_terms(nf, vf), lf) !=
                     bs::lobe(float(alpha2), ff, nf, lf, bs::view_terms(nf, lf), vf);
  }
  CHECK(asymmetric[0] == 0);
  CHECK(asymmetric[1] == 0);

  // s = m = 0: F0 = F90 = 0, so E_s and the lobe are exactly 0 and f is rho / pi (the Lambert path).
  const bs::BsdfTables t = bs::host_tables();
  const Vec3<float> n{0.0f, 0.0f, 1.0f};
  const Vec3<float> v = to<float>(about({0.0, 0.0, 1.0}, 0.7, 0.3)), l = to<float>(about({0.0, 0.0, 1.0}, 0.6, 2.9));
  for (float alpha2 : {0.0016f, 0.25f, 1.0f}) {
    const bs::BsdfParams<float> p{0.37f, 0.0f, alpha2, 0.0f};
    const bs::Fresnel<float> fr = bs::fresnel_params(p.rho, p.metallic, p.s);
    CHECK(fr.f0 == 0.0f);
    CHECK(fr.f90 == 0.0f);
    CHECK(bs::spec_albedo(fr, bs::spec_ab(t, bs::view_terms(n, v).t, bs::table_r(alpha2))) == 0.0f);
    CHECK(bs::lobe(alpha2, fr, n, v, bs::view_terms(n, v), l) == 0.0f);
    CHECK(bs::bsdf_eval(t, p, n, v, l) == p.rho * bs::kInvPi<float>);
  }
  // A back-facing shading normal with the light exactly opposite the view: h = 0, D = 0, everything finite.
  const Vec3<double> nz{0.0, 0.0, 1.0}, back = about(nz, -0.3, 0.4);
  CHECK(bs::lobe(0.25, bs::fresnel_params(0.5, 0.0, 0.04), nz, back, bs::view_terms(nz, back), -back) == 0.0);
  CHECK(std::isfinite(bs::bsdf_eval(t, bs::BsdfParams<double>{0.5, 0.0, 0.25, 0.04}, nz, back, -back)));
  // Fresnel: F(1) = F0, F(0) = F90; F90 = min(1, 50 s) for a dielectric, 1 for a metal; continuous to s = 0.
  const bs::Fresnel<double> diel = bs::fresnel_params(0.5, 0.0, 0.004), metal = bs::fresnel_params(0.9, 1.0, 0.04);
  CHECK(bs::schlick(diel, 1.0) == 0.004);
  CHECK(bs::schlick(diel, 0.0) == doctest::Approx(0.2));
  CHECK(bs::fresnel_params(0.5, 0.0, 0.04).f90 == 1.0);
  CHECK(metal.f0 == 0.9);
  CHECK(metal.f90 == 1.0);
}

TEST_CASE("energy: the table albedo equals the lobe's hemispherical integral, and the BSDF's albedo is at most 1") {
  const bs::BsdfTables t = bs::host_tables();
  const Vec3<double> n{0.0, 0.0, 1.0};
  struct Case {
    double mu_v, r, rho, metallic, s;
  };
  for (const Case& c : {Case{0.2, 0.3, 0.8, 0.0, 0.04}, Case{0.6, 0.5, 0.5, 1.0, 0.04}, Case{0.95, 1.0, 0.9, 0.3, 0.08},
                        Case{0.05, 0.6, 0.7, 0.0, 0.5}, Case{0.4, 0.22, 1.0, 0.0, 0.04}}) {
    const double alpha2 = std::pow(c.r, 4.0);
    const Vec3<double> v = about(n, c.mu_v, 0.0);
    const bs::ViewTerms<double> vt = bs::view_terms(n, v);
    const bs::Fresnel<double> fr = bs::fresnel_params(c.rho, c.metallic, c.s);
    // int lobe mu_l dw_l over the hemisphere, as (mu_l, phi), phi in [0, pi] doubled (the plane of v and n).
    const auto ring = [&](double mu_l) {
      return ref::integrate<1>(
          [&](double phi) { return ref::Vals<1>{2.0 * bs::lobe(alpha2, fr, n, v, vt, about(n, mu_l, phi)) * mu_l}; },
          0.0, ref::kPi, 1e-9);
    };
    const double e_quad = ref::integrate<1>(ring, 0.0, 1.0, 1e-8)[0];
    const double e_tab = bs::spec_albedo(fr, bs::spec_ab(t, vt.t, bs::table_r(alpha2)));
    INFO("mu_v " << c.mu_v << " r " << c.r << ": quadrature " << e_quad << ", table " << e_tab);
    CHECK(std::abs(e_tab - e_quad) <= 1e-3);
    CHECK((1.0 - c.metallic) * (1.0 - e_quad) * c.rho + e_quad <= 1.0);
  }
  // The table form (1 - m)(1 - E_s) rho + E_s <= 1, and a diffuse weight >= 0, over random materials and views at
  // every roughness: the near-mirror rows' overshoot (A + B up to 1.003) is capped in spec_albedo. Every fourth
  // material is a white metal at a near-mirror roughness, where the overshoot is.
  std::mt19937_64 rng(29);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  double worst = 0.0, min_weight = 1.0;
  for (int i = 0; i < 100000; ++i) {
    const bool white = i % 4 == 0;
    const double rho = white ? 1.0 : u01(rng), m = white ? 1.0 : u01(rng), s = u01(rng);
    const double r = white ? 0.04 * u01(rng) : u01(rng);
    const bs::Fresnel<double> fr = bs::fresnel_params(rho, m, s);
    const double e_s = bs::spec_albedo(fr, bs::spec_ab(t, u01(rng), r));
    worst = std::max(worst, bs::diffuse_weight(m, e_s) * rho + e_s);
    min_weight = std::min(min_weight, bs::diffuse_weight(0.0, e_s));
  }
  INFO("max (1 - m)(1 - E_s) rho + E_s " << worst << ", min (1 - E_s) " << min_weight);
  CHECK(worst <= 1.0);
  CHECK(min_weight >= 0.5 * bs::kDeltaE - 1e-15);  // 1 - (1 - delta_E / 2), rounded
  // A near-mirror (r 0.02) dielectric and white metal at mu_v ~ 0.006, where the spline overshoots most.
  for (const double metallic : {0.0, 1.0}) {
    const bs::Fresnel<double> fr = bs::fresnel_params(1.0, metallic, 0.04);
    const Vec3<double> v = about(n, 0.006, 0.0);
    const double e_s = bs::spec_albedo(fr, bs::spec_ab(t, bs::view_terms(n, v).t, 0.02));
    CHECK(e_s <= 1.0 - 0.5 * bs::kDeltaE);
    CHECK(bs::diffuse_weight(metallic, e_s) >= 0.0);
  }
}

TEST_CASE("floors, caps and the effective roughness are monotone and C2") {
  double prev = bs::floor_q(-0.5);
  bool monotone = true;
  for (int i = 1; i <= 20000; ++i) {
    const double q = bs::floor_q(-0.5 + 2.0 * i / 20000.0);
    monotone &= q >= prev;
    prev = q;
  }
  CHECK(monotone);
  const double h = 1e-4;
  // At u = 0: the constant 1/2 joins with zero slope and curvature; at u = 1: the identity, slope 1, curvature 0.
  CHECK(bs::floor_q(0.0) == 0.5);
  CHECK(std::abs((bs::floor_q(2 * h) - 2 * bs::floor_q(h) + bs::floor_q(0.0)) / (h * h)) <= 1e-3);
  CHECK(bs::floor_q(1.0) == 1.0);
  CHECK((bs::floor_q(1.0) - bs::floor_q(1.0 - h)) / h == doctest::Approx(1.0).epsilon(1e-6));
  CHECK(std::abs((bs::floor_q(1.0) - 2 * bs::floor_q(1.0 - h) + bs::floor_q(1.0 - 2 * h)) / (h * h)) <= 1e-3);

  for (double x : {1e-3, 0.002, 0.5, 1.0}) CHECK(bs::smooth_floor(x, 1e-3) == x);
  for (double x : {-1.0, 0.0, 5e-4, 9.99e-4}) {
    CHECK(bs::smooth_floor(x, 1e-3) >= 5e-4);
    CHECK(bs::smooth_floor(x, 1e-3) <= 1e-3);
  }
  for (double a : {0.0, 0.5, 0.99}) CHECK(bs::smooth_cap1(a, 1e-2) == a);
  CHECK(bs::smooth_cap1(1.0, 1e-2) == doctest::Approx(0.995));
  CHECK(bs::smooth_cap1(7.0, 1e-2) == doctest::Approx(0.995));

  // alpha_eff: r_s = 1 adds nothing; a spread of normals widens the lobe (kappa(0.9) = 0.38 / 1.971); the cap and
  // the floor hold.
  CHECK(bs::alpha_eff(0.25, 0.0, 1.0) == 0.25);
  CHECK(bs::alpha_eff(0.25, 0.01, 1.0) == doctest::Approx(0.26));
  CHECK(bs::alpha_eff(0.04, 0.0, 0.9) == doctest::Approx(0.04 + 0.38 / 1.971).epsilon(1e-12));
  double prev_a = 0.0;
  bool widening = true;
  for (int i = 1000; i >= 0; --i) {
    const double a = bs::alpha_eff(0.04, 0.0, i / 1000.0);
    widening &= a >= prev_a;
    prev_a = a;
  }
  CHECK(widening);
  CHECK(bs::alpha_eff(0.04, 0.0, 0.0) <= 1.0 - 0.5 * bs::kDeltaA);
  // The floor's minimum is Mitsuba's alpha >= 1e-4: alpha^2 = kAlpha2Min / 2 = 1e-8 exactly, and >= 1e-8 below it.
  CHECK(bs::alpha_eff(0.0, 0.0, 1.0) == 1e-8);
  CHECK(bs::alpha_eff(0.0f, 0.0f, 1.0f) == 1e-8f);
  for (double a2 : {1e-12, 5e-9, 1e-8, 1.5e-8}) CHECK(bs::alpha_eff(a2, 0.0, 1.0) >= 1e-8);
  CHECK(bs::alpha_eff(2e-8, 0.0, 1.0) == 2e-8);

  // The cap of E_s: identity up to 1 - delta_E, at most 1 - delta_E / 2, monotone; s = m = 0 stays exactly 0.
  const bs::Fresnel<double> white{1.0, 1.0};
  const auto e_of = [&](double sum) { return bs::spec_albedo(white, bs::SpecAB<double>{sum, 0.0}); };
  for (double a : {0.0, 0.3, 0.999}) CHECK(e_of(a) == a);
  CHECK(e_of(1.0) == doctest::Approx(1.0 - 0.5 * bs::kDeltaE));
  CHECK(e_of(1.003) == doctest::Approx(1.0 - 0.5 * bs::kDeltaE));
  bool capped_monotone = true;
  for (int i = 1; i <= 4000; ++i) capped_monotone &= e_of(0.998 + 1e-6 * i) >= e_of(0.998 + 1e-6 * (i - 1));
  CHECK(capped_monotone);
  CHECK(bs::spec_albedo(bs::Fresnel<double>{0.0, 0.0}, bs::SpecAB<double>{0.7, 0.31}) == 0.0);
}

TEST_CASE("view terms: t = sqrt(1 - sqrt(mu_v)) through normal view, and the floor of mu_v") {
  const Vec3<double> n{0.0, 0.0, 1.0};
  double err_d = 0.0, err_f = 0.0;
  for (int i = -1000; i <= 1000; ++i) {
    const double th = 1e-4 * i;
    const Vec3<double> v{std::sin(th), 0.0, std::cos(th)};
    // 1 - sqrt(cos th) = 2 sin^2(th / 2) / (1 + sqrt(cos th)), without cancellation.
    const double t = std::sqrt(2.0) * std::abs(std::sin(0.5 * th)) / std::sqrt(1.0 + std::sqrt(std::cos(th)));
    if (t == 0.0) {
      CHECK(bs::view_terms(n, v).t == 0.0);
      continue;
    }
    err_d = std::max(err_d, std::abs(bs::view_terms(n, v).t / t - 1.0));
    err_f = std::max(err_f, std::abs(double(bs::view_terms(to<float>(n), to<float>(v)).t) / t - 1.0));
  }
  INFO("relative error of t near normal view: fp64 " << err_d << ", fp32 " << err_f);
  CHECK(err_d <= 1e-14);
  CHECK(err_f <= 1e-6);
  // Both branches meet at mu_v = 1/2 within rounding, and t reaches 1 at grazing.
  const Vec3<double> lo = about(n, 0.5 - 1e-12, 0.0), hi = about(n, 0.5 + 1e-12, 0.0);
  CHECK(std::abs(bs::view_terms(n, lo).t - bs::view_terms(n, hi).t) <= 1e-11);
  CHECK(bs::view_terms(n, about(n, 0.3, 0.0)).t == doctest::Approx(std::sqrt(1.0 - std::sqrt(0.3))).epsilon(1e-14));
  // A shading normal facing away: mu_v is floored, positive and finite.
  const bs::ViewTerms<double> away = bs::view_terms(n, about(n, -0.3, 1.0));
  CHECK(away.mu >= 0.5 * bs::kDeltaMu);
  CHECK(away.t <= 1.0);
}

TEST_CASE("fp32 accuracy gate: bsdf_eval<float> against bsdf_eval<double> (DESIGN §7.11, p2f.md §6.3)") {
  // 10^6 random (n, v, l, r in [0.2, 1], s, rho, m) with mu_v, mu_l >= 0.01 and f >= 1e-4 f_max, f_max the value at
  // the mirror direction; both precisions see the same fp32 inputs and the same tables.
  const bs::BsdfTables t = bs::host_tables();
  std::mt19937_64 rng(31);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  double worst = 0.0;
  int64_t excluded = 0, kept = 0;
  for (int i = 0; i < 1000000; ++i) {
    const Vec3<float> nf = to<float>(unit({u01(rng) - 0.5, u01(rng) - 0.5, u01(rng) - 0.5}));
    const Vec3<double> n = cast<double>(nf);
    const Vec3<float> vf = to<float>(about(n, 0.01 + 0.99 * u01(rng), 2 * ref::kPi * u01(rng)));
    const Vec3<float> lf = to<float>(about(n, 0.01 + 0.99 * u01(rng), 2 * ref::kPi * u01(rng)));
    const float r = float(0.2 + 0.8 * u01(rng));
    const bs::BsdfParams<float> pf{float(u01(rng)), float(u01(rng)), r * r * r * r,
                                   float(i % 2 == 0 ? 0.1 * u01(rng) : u01(rng))};
    const bs::BsdfParams<double> pd{pf.rho, pf.metallic, pf.alpha2, pf.s};
    const Vec3<double> v = cast<double>(vf), l = cast<double>(lf);
    if (!(dot(n, v) >= 0.01 && dot(n, l) >= 0.01)) {
      ++excluded;
      continue;
    }
    const double fd = bs::bsdf_eval(t, pd, n, v, l);
    const double f_max = bs::bsdf_eval(t, pd, n, v, (2.0 * dot(n, v)) * n - v);
    if (!(fd >= 1e-4 * f_max)) {
      ++excluded;
      continue;
    }
    ++kept;
    worst = std::max(worst, std::abs(double(bs::bsdf_eval(t, pf, nf, vf, lf)) / fd - 1.0));
  }
  INFO("max |f32 / f64 - 1| = " << worst << " over " << kept << " configurations (" << excluded << " excluded)");
  CHECK(kept >= 900000);
  CHECK(worst <= 2e-5);
}

namespace {

// I_ltc of a polygon (vertices relative to the shading point): mapped by ltc_apply, clipped to z >= 0, summed as
// sum_i 1/2 theta_i (p_i x p_i+1)_z / |p_i x p_i+1| with theta_i = atan2(|p_i x p_i+1|, p_i . p_i+1) (p2f.md §4.6);
// pi times the LTC's share of the polygon. fp64, for the tests only (F8b builds the renderer's form).
double ltc_polygon(const bs::LtcInverse<double>& m, const bs::LtcFrame<double>& f, Vec3<double> n,
                   const std::vector<Vec3<double>>& verts) {
  std::vector<Vec3<double>> p, q;
  for (const Vec3<double>& v : verts) p.push_back(bs::ltc_apply(m, f, n, v));
  for (size_t i = 0; i < p.size(); ++i) {
    const Vec3<double> a = p[i], b = p[(i + 1) % p.size()];
    if (a.z >= 0.0) q.push_back(a);
    if ((a.z >= 0.0) != (b.z >= 0.0)) q.push_back(a + (a.z / (a.z - b.z)) * (b - a));
  }
  double sum = 0.0;
  for (size_t i = 0; i < q.size(); ++i) {
    const Vec3<double> a = unit(q[i]), b = unit(q[(i + 1) % q.size()]);
    const Vec3<double> c = cross(a, b);
    const double s = norm(c);
    if (s > 0.0) sum += 0.5 * std::atan2(s, dot(a, b)) * c.z / s;
  }
  return std::abs(sum);
}

}  // namespace

TEST_CASE("LTC tables: the normal-view nodes are the isotropic fit, and the off-diagonal entries are odd") {
  const bs::BsdfTables t = bs::host_tables();
  // At t = 0, for every roughness (nodes and between): m02 = m20 = 0 (odd splines through node values 0), m00 = 1.
  double odd = 0.0, iso = 0.0;
  for (int j = 0; j <= 2 * (kN - 1); ++j) {
    const bs::LtcInverse<double> m = bs::ltc_inverse(t, 0.0, j / (2.0 * (kN - 1)));
    odd = std::max(odd, std::max(std::abs(m.m02), std::abs(m.m20)));
    iso = std::max(iso, std::abs(m.m00 - 1.0));
  }
  INFO("t = 0: |m02|, |m20| <= " << odd << ", |m00 - 1| <= " << iso);
  CHECK(odd <= 1e-12);
  CHECK(iso <= 1e-6);
  // T-TAB's nodes: the isotropic fit, redone here without a seed, is what the splines hold at t = 0 (the generator
  // checks every node of every table on each run, in the ctest bsdf_tables_regen).
  for (int j : {0, 13, 32, 63}) {
    const ltc::NodeFit nf = ltc::fit_isotropic(ltc::make_target(0.0, node(j)));
    CHECK_FALSE(nf.at_bracket);
    CHECK(bs::ltc_inverse(t, 0.0, node(j)).m22 == doctest::Approx(nf.e.m22).epsilon(1e-6));
  }
  // The lobe the LTC approximates is bsdf.h's lobe at F = 1 times mu_l, normalized by A + B.
  std::mt19937_64 rng(43);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  const Vec3<double> n{0.0, 0.0, 1.0};
  const bs::Fresnel<double> one{1.0, 1.0};
  double lobe_err = 0.0, e1_err = 0.0;
  for (int k = 0; k < 20; ++k) {
    const double x = 0.95 * u01(rng), r = 0.2 + 0.8 * u01(rng);
    const ltc::Lobe lb = ltc::make_lobe(x, r);
    const ref::Vals<2> ab = ref::spec_ab(x, r);
    e1_err = std::max(e1_err, std::abs(lb.e1 - (ab[0] + ab[1])));
    const bs::ViewTerms<double> vt = bs::view_terms(n, lb.g.v);
    for (int q = 0; q < 50; ++q) {
      const Vec3<double> l = about(n, 0.01 + 0.99 * u01(rng), 2 * ref::kPi * u01(rng));
      const double want = bs::lobe(std::pow(r, 4.0), one, n, lb.g.v, vt, l) * l.z;  // D V F mu_l
      lobe_err = std::max(lobe_err, std::abs(ltc::lobe_density(lb, l) * lb.e1 / want - 1.0));
    }
  }
  INFO("lobe against bsdf.h " << lobe_err << ", E1 against A + B " << e1_err);
  CHECK(lobe_err <= 1e-12);
  CHECK(e1_err <= 2e-8);
}

TEST_CASE("T-TAB (LTC): the fit through the splines at cell centres, r >= 0.2") {
  // ltc_ref.h fit_metrics at cell centres through the fp32 splines. B4's "LTC fit <= 3 %" is read as cap25: the
  // relative error of L_spec for a disk emitter about the lobe's mean direction that receives >= 25 % of the lobe, the
  // whole hemisphere included. Gated from normal view to mu_v = 0.6 at every r >= 0.2 (gen_tables --report: 2.45e-2
  // over all 1500 such cells; every row holds 3 % from mu_v >= 0.5625). Towards grazing views the L^3 fit does not
  // hold 3 % (a departure: RENDER.md "P2-F notes", F3c; there no four-parameter LTC keeps both cap25 and tv small), so
  // those views are tracked at the reachable views (t <= t_max): cap25 <= 0.45 (0.434), tv <= 0.47 (0.453), and the
  // full-hemisphere emitter's I_ltc / pi >= 0.985 (0.9867).
  const bs::BsdfTables t = bs::host_tables();
  const double t_max = std::sqrt(1.0 - std::sqrt(0.5 * bs::kDeltaMu));
  double gated = 0.0, cap = 0.0, tv = 0.0, upper = 1.0, estimate = 0.0;
  int n_gated = 0;
  for (int j : {13, 16, 19, 22, 26, 30, 32, 38, 45, 52, 62}) {
    for (int i : {0, 6, 12, 18, 24, 28, 32, 36, 40, 46, 52, 58, 61}) {
      const double x = node(i) + 0.5 / (kN - 1), r = node(j) + 0.5 / (kN - 1);
      if (x > t_max) continue;
      const bs::LtcInverse<double> m = bs::ltc_inverse(t, x, r);
      const ltc::Lobe lb = ltc::make_lobe(x, r);
      const ltc::Metrics mt = ltc::fit_metrics(lb, {m.m00, m.m02, m.m20, m.m22}, nullptr, 128);
      if (lb.g.mu_v >= 0.6) {
        gated = std::max(gated, mt.cap25);
        ++n_gated;
      }
      cap = std::max(cap, mt.cap25);
      tv = std::max(tv, mt.tv);
      upper = std::min(upper, mt.upper);
      // The estimator's own checks: the lobe's total is 1, and the LTC's upper mass matches the exact form.
      estimate = std::max({estimate, std::abs(mt.lobe_total - 1.0), std::abs(mt.upper_quad - mt.upper)});
    }
  }
  INFO("cap25 at mu_v >= 0.6: " << gated << " (" << n_gated << " cells); reachable views: cap25 " << cap << ", tv "
                                << tv << ", full-hemisphere I_ltc / pi >= " << upper << "; estimator " << estimate);
  CHECK(n_gated >= 50);
  CHECK(gated <= 0.03);
  CHECK(cap <= 0.45);
  CHECK(tv <= 0.47);
  CHECK(upper >= 0.985);
  CHECK(estimate <= 5e-3);
  // Near-mirror rows (r < 0.2, only under scene.allow_missing_reflections) at mu_v >= 0.6, tracked: the r = 0 node is
  // the alpha -> 0 limit, and bsdf.h's floor alpha = 1e-4 is r = 0.01, where the natural end of the spline in r
  // leaves cap25 0.230 (0.115 at r 0.012; <= 0.040 from r 0.014; RENDER.md "P2-F notes", F3c).
  const double t6 = std::sqrt(1.0 - std::sqrt(0.6));
  double floor_cap = 0.0, near_cap = 0.0;
  for (const double r : {0.01, 0.014, 0.02, 0.05, 0.1}) {
    double& worst_r = r < 0.011 ? floor_cap : near_cap;
    for (int i = 0; i <= 8; ++i) {
      const double x = t6 * i / 8.0;
      const bs::LtcInverse<double> m = bs::ltc_inverse(t, x, r);
      const ltc::Metrics mt = ltc::fit_metrics(ltc::make_lobe(x, r), {m.m00, m.m02, m.m20, m.m22}, nullptr, 128);
      worst_r = std::max(worst_r, mt.cap25);
    }
  }
  INFO("near-mirror rows at mu_v >= 0.6: cap25 " << floor_cap << " at the floor r = 0.01, " << near_cap
                                                 << " at r 0.014 - 0.1");
  CHECK(floor_cap <= 0.25);
  CHECK(near_cap <= 0.045);
}

TEST_CASE("LTC end to end: ltc_frame, ltc_inverse and ltc_apply against the lobe over a rect (F8b's path)") {
  // F8b's map: emitter vertices through ltc_apply in the frame of ltc_frame, with the entries of ltc_inverse. (1) Its
  // density at any direction is ltc_ref.h's ltc_density of the same entries in that frame: the renderer's map and
  // the fit's share one convention (a transposed off-diagonal in either breaks this). (2) For a 1.2 x 0.8 m rect 2 m
  // away, centred on the mirror direction and turned at random about it, I_ltc / pi matches the brute-force integral
  // of the normalized lobe D V mu_l / (A + B) over the rect (a 500^2 midpoint grid) within 4 % at mu_v >= 0.6 and
  // r in [0.2, 1] (3.0 % over these 32 configurations; a transposed ltc_apply errs by up to 100 %).
  const bs::BsdfTables t = bs::host_tables();
  std::mt19937_64 rng(47);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  const bs::Fresnel<double> one{1.0, 1.0};
  constexpr int kG = 500;
  double worst = 0.0, density = 0.0;
  for (int k = 0; k < 32; ++k) {
    const Vec3<double> n = unit({u01(rng) - 0.5, u01(rng) - 0.5, u01(rng) - 0.5});
    const Vec3<double> wv = about(n, 0.6 + 0.4 * u01(rng), 2 * ref::kPi * u01(rng));
    const double r = 0.2 + 0.8 * u01(rng), alpha2 = r * r * r * r;
    const bs::ViewTerms<double> vt = bs::view_terms(n, wv);
    const bs::LtcFrame<double> f = bs::ltc_frame(n, wv);
    const bs::LtcInverse<double> m = bs::ltc_inverse(t, vt.t, bs::table_r(alpha2));
    const ref::Vals<2> ab = ref::spec_ab(vt.t, r);
    const Vec3<double> dir = (2.0 * dot(n, wv)) * n - wv, c = 2.0 * dir;
    const Vec3<double> ua = unit(cross(dir, about(dir, 0.0, 2 * ref::kPi * u01(rng)))), va = cross(dir, ua);
    const double hw = 0.6, hh = 0.4;
    const std::vector<Vec3<double>> rect{c - hw * ua - hh * va, c + hw * ua - hh * va, c + hw * ua + hh * va,
                                         c - hw * ua + hh * va};
    const double i_ltc = ltc_polygon(m, f, n, rect) / ref::kPi;
    double brute = 0.0;
    for (int j = 0; j < kG; ++j) {
      for (int i = 0; i < kG; ++i) {
        const Vec3<double> p = c + (hw * (2.0 * (i + 0.5) / kG - 1.0)) * ua + (hh * (2.0 * (j + 0.5) / kG - 1.0)) * va;
        const double d2 = dot(p, p);
        const Vec3<double> l = (1.0 / std::sqrt(d2)) * p;
        const double mu_l = dot(n, l);
        if (mu_l > 0.0) brute += bs::lobe(alpha2, one, n, wv, vt, l) * mu_l * dot(l, dir) / d2;
      }
    }
    brute *= 4.0 * hw * hh / (static_cast<double>(kG) * kG * (ab[0] + ab[1]));
    worst = std::max(worst, std::abs(i_ltc / brute - 1.0));
    const double det = m.m00 * m.m22 - m.m02 * m.m20;
    for (int q = 0; q < 100; ++q) {
      const Vec3<double> l = about(n, 2.0 * u01(rng) - 1.0, 2 * ref::kPi * u01(rng));
      const Vec3<double> w = bs::ltc_apply(m, f, n, l);
      const double len2 = dot(w, w);
      const double d_map = w.z > 0.0 ? w.z * det / (ref::kPi * (len2 * len2)) : 0.0;
      const double d_fit = ltc::ltc_density({m.m00, m.m02, m.m20, m.m22}, det, {dot(f.t, l), dot(f.b, l), dot(n, l)});
      density = std::max(density, std::abs(d_map - d_fit) / std::max(d_fit, 1e-300));
    }
  }
  INFO("I_ltc / pi against brute force: max relative error " << worst << "; density of ltc_apply against ltc_ref.h "
                                                               << density);
  CHECK(density <= 1e-12);
  CHECK(worst <= 0.04);
}

TEST_CASE("LTC frame: right-handed, towards the view's projection, Duff et al.'s basis at normal view; fp32") {
  std::mt19937_64 rng(41);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  const bs::LtcInverse<double> identity{1.0, 0.0, 0.0, 1.0};
  double ortho = 0.0, plane = 0.0, toward = 1.0;
  const auto check_frame = [&](Vec3<double> n, const bs::LtcFrame<double>& f) {
    ortho = std::max({ortho, std::abs(dot(f.t, n)), std::abs(norm(f.t) - 1.0), norm(cross(f.t, f.b) - n)});
  };
  for (int i = 0; i < 10000; ++i) {
    const Vec3<double> n = unit({u01(rng) - 0.5, u01(rng) - 0.5, u01(rng) - 0.5});
    const Vec3<double> v = about(n, -0.5 + 1.499 * u01(rng), 2 * ref::kPi * u01(rng));
    const bs::LtcFrame<double> f = bs::ltc_frame(n, v);
    check_frame(n, f);
    // The view in the LTC's space is (s_v, 0, mu_v) with s_v >= 0: the tables' fitting frame.
    const Vec3<double> local = bs::ltc_apply(identity, f, n, v);
    plane = std::max(plane, std::abs(local.y));
    toward = std::min(toward, local.x / std::sqrt(std::max(1e-300, 1.0 - local.z * local.z)));
  }
  for (const Vec3<double> n : {Vec3<double>{0.0, 0.0, 1.0}, Vec3<double>{0.0, 0.0, -1.0}, unit({0.3, -0.5, 0.2}),
                               unit({-0.2, 0.1, -0.9})}) {
    check_frame(n, bs::ltc_frame(n, n));  // normal view: Duff et al.'s basis
  }
  INFO("frame orthonormality " << ortho << ", |b . w_v| " << plane << ", min t . w_v / s_v " << toward);
  CHECK(ortho <= 1e-14);
  CHECK(plane <= 1e-14);
  CHECK(toward >= 1.0 - 1e-12);
  // fp32 entries against fp64 (the same fp32 coefficients): relative to the largest entry.
  const bs::BsdfTables t = bs::host_tables();
  double rel = 0.0;
  for (int i = 0; i < 100000; ++i) {
    const float x = float(u01(rng)), r = float(u01(rng));
    const bs::LtcInverse<float> mf = bs::ltc_inverse(t, x, r);
    const bs::LtcInverse<double> md = bs::ltc_inverse(t, double(x), double(r));
    const double scale = std::max({1.0, std::abs(md.m00), std::abs(md.m02), std::abs(md.m20), std::abs(md.m22)});
    rel = std::max({rel, std::abs(double(mf.m00) - md.m00) / scale, std::abs(double(mf.m02) - md.m02) / scale,
                    std::abs(double(mf.m20) - md.m20) / scale, std::abs(double(mf.m22) - md.m22) / scale});
  }
  INFO("fp32 entries: max |f32 - f64| / max(1, |M^-1|) " << rel);
  CHECK(rel <= 1e-6);
}

TEST_CASE("T-TAB view sweep: L_spec of a rect light has no jump or kink through normal view") {
  // A 0.4 x 0.2 m rect light 0.8 m above the shading point, off to one side and turned, and a dielectric at r 0.5.
  // The view sweeps through the normal in a vertical plane at azimuth 0.7: L_spec = E_s I_ltc / pi (p2f.md §4.6) is
  // C2 there, so its second difference at the crossing is the size of its neighbours'. The counterfactual keeps the
  // frame from turning (as an even extension of the off-diagonal entries would): a kink, ~h times the slope jump.
  const bs::BsdfTables t = bs::host_tables();
  const Vec3<double> n{0.0, 0.0, 1.0}, c{0.3, 0.15, 0.8}, u{std::cos(0.35), std::sin(0.35), 0.0};
  const Vec3<double> v{-u.y, u.x, 0.0};
  std::vector<Vec3<double>> rect;
  for (const auto& [a, b] : {std::pair{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}}) {
    rect.push_back(c + (0.2 * a) * u + (0.1 * b) * v);
  }
  const double phi0 = 0.7, rough = 0.5;
  const bs::Fresnel<double> fr = bs::fresnel_params(0.5, 0.0, 0.04);
  const auto l_spec = [&](double th, bool turning) {
    const Vec3<double> wv{std::sin(th) * std::cos(phi0), std::sin(th) * std::sin(phi0), std::cos(th)};
    const bs::ViewTerms<double> vt = bs::view_terms(n, wv);
    const double e_s = bs::spec_albedo(fr, bs::spec_ab(t, vt.t, rough));
    const bs::LtcFrame<double> f =
        turning ? bs::ltc_frame(n, wv)
                : bs::LtcFrame<double>{{std::cos(phi0), std::sin(phi0), 0.0}, {-std::sin(phi0), std::cos(phi0), 0.0}};
    return e_s * ltc_polygon(bs::ltc_inverse(t, vt.t, rough), f, n, rect) / ref::kPi;
  };
  const double h = 2e-3;
  for (const bool turning : {true, false}) {
    double near = 0.0, far = 0.0, lo = 1e300, hi = 0.0;
    for (int k = -39; k <= 39; ++k) {
      const double l0 = l_spec(k * h, turning);
      const double d2 = std::abs(l_spec((k + 1) * h, turning) - 2.0 * l0 + l_spec((k - 1) * h, turning));
      if (std::abs(k) <= 1) {
        near = std::max(near, d2);
      } else if (std::abs(k) >= 5) {
        far = std::max(far, d2);
      }
      lo = std::min(lo, l0);
      hi = std::max(hi, l0);
    }
    const std::string label = turning ? "frame turning" : "counterfactual, fixed frame";
    INFO(label << ": |second difference| at the crossing " << near << ", elsewhere <= " << far << "; L_spec in ["
               << lo << ", " << hi << "]");
    CHECK(hi - lo >= 0.01 * hi);  // the sweep moves the highlight across the light: L_spec is not flat
    if (turning) {
      CHECK(near <= 2.0 * far);
    } else {
      CHECK(near >= 10.0 * far);
    }
  }
}
