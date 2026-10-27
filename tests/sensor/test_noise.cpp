// In-path noise: exact transitions, stationary law, tabulation, bridges and keyed draws
// (src/sensor/noise_tables.*, noise.h; M3 V3, V7b).
#include <doctest/doctest.h>

#include <cmath>
#include <random>

#include "sensor/noise.h"
#include "sensor/noise_tables.h"

using namespace balla::sensor;
using balla::rng::Purpose;

namespace {

NoiseModel prophesee_like(int levels = 5) {
  NoiseModel m{};
  m.kq = 1e16 * 1.602176634e-19;
  m.a2 = 1.0 / 25e-6;
  m.a3 = 1.0 / 100e-6;
  m.sigma_o = 0.06;
  m.levels = levels;
  m.margin_z = 6.0;
  return m;
}

// Modal closed form of M3 D.2-D.3 for distinct rates (the reference the tables replace).
void modal(const NoiseModel& m, double a1, double h, double phi[3][3], double q[3][3]) {
  const double a[3] = {a1, m.a2, m.a3};
  // A = V diag(-a) V^-1 with eigenvectors of the lower bidiagonal matrix.
  double V[3][3] = {}, Vi[3][3] = {};
  const double A[3][3] = {{-a1, 0, 0}, {m.a2, -m.a2, 0}, {0, m.a3, -m.a3}};
  for (int j = 0; j < 3; ++j) {
    // (A + a_j I) v = 0 solved top-down with v_j = 1.
    double v[3] = {0, 0, 0};
    v[j] = 1;
    for (int i = j + 1; i < 3; ++i) v[i] = A[i][i - 1] * v[i - 1] / (-a[j] - A[i][i]);
    for (int i = 0; i < 3; ++i) V[i][j] = v[i];
  }
  // Invert the unit lower-triangular V.
  for (int j = 0; j < 3; ++j) {
    Vi[j][j] = 1;
    for (int i = j + 1; i < 3; ++i) {
      double s = 0;
      for (int k = j; k < i; ++k) s -= V[i][k] * Vi[k][j];
      Vi[i][j] = s;
    }
  }
  const double GG[3] = {2 * m.kq * a1, 2 * m.sigma_o * m.sigma_o * m.a2, 0};
  double M[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0;
      for (int k = 0; k < 3; ++k) s += Vi[i][k] * GG[k] * Vi[j][k];
      M[i][j] = s * (-std::expm1(-(a[i] + a[j]) * h)) / (a[i] + a[j]);
    }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double p = 0, s = 0;
      for (int k = 0; k < 3; ++k) {
        p += V[i][k] * std::exp(-a[k] * h) * Vi[k][j];
        for (int l = 0; l < 3; ++l) s += V[i][k] * M[k][l] * V[j][l];
      }
      phi[i][j] = p;
      q[i][j] = s;
    }
}

}  // namespace

TEST_CASE("scaling-and-squaring transitions match the modal closed form (M3 D.2-D.3)") {
  const NoiseModel m = prophesee_like();
  double worst_phi = 0, worst_q = 0;
  for (double a1 : {0.3, 30.0, 3e3, 2.7e4, 3e5}) {
    for (double h : {2.048e-6, 16.384e-6, 65.536e-6}) {
      const NoiseMatrices t = noise_transition(m, a1, h);
      double phi[3][3], q[3][3];
      modal(m, a1, h, phi, q);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          worst_phi = std::max(worst_phi, std::fabs(t.phi[i][j] - phi[i][j]));
          worst_q = std::max(worst_q, std::fabs(t.q[i][j] - q[i][j]) / std::sqrt(q[i][i] * q[j][j]));
        }
    }
  }
  MESSAGE("max |dPhi| " << worst_phi << ", max rel |dQ| " << worst_q);
  CHECK(worst_phi < 1e-12);
  CHECK(worst_q < 1e-9);
}

TEST_CASE("transitions compose, confluent rates included, and Q tends to P_inf") {
  const NoiseModel m = prophesee_like();
  for (double a1 : {1e-2, 4e4 /* = a2 */, 1e4 /* = a3 */, 1e9}) {
    const double h = 32.768e-6;
    const NoiseMatrices one = noise_transition(m, a1, h), two = noise_transition(m, a1, 2 * h);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        double pp = 0, qq = one.q[i][j];
        for (int k = 0; k < 3; ++k) {
          pp += one.phi[i][k] * one.phi[k][j];
          for (int l = 0; l < 3; ++l) qq += one.phi[i][k] * one.q[k][l] * one.phi[j][l];
        }
        CHECK(std::fabs(pp - two.phi[i][j]) <= 1e-13 + 1e-12 * std::fabs(pp));
        CHECK(std::fabs(qq - two.q[i][j]) <= 1e-12 * std::sqrt(two.q[i][i] * two.q[j][j]));
      }
    double P[3][3];
    noise_stationary_cov(m, a1, P);
    const NoiseMatrices inf = noise_transition(m, a1, 60.0 / std::min(a1, m.a3));  // e^-60: stationary
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) CHECK(std::fabs(inf.q[i][j] - P[i][j]) <= 1e-10 * std::sqrt(P[i][i] * P[j][j]));
  }
}

TEST_CASE("comparator-referred variance equals M3 A.12") {
  const NoiseModel m = prophesee_like();
  for (double a1 : {0.1, 100.0, 1e4, 1e6}) {
    double P[3][3];
    noise_stationary_cov(m, a1, P);
    const double a2 = m.a2, a3 = m.a3;
    const double expect = m.kq * a2 * a3 * (a1 + a2 + a3) / ((a1 + a2) * (a1 + a3) * (a2 + a3)) +
                          m.sigma_o * m.sigma_o * a3 / (a2 + a3);
    CHECK(P[2][2] == doctest::Approx(expect).epsilon(1e-14));
  }
  // Prophesee-like values quoted in DESIGN §8.6: 0.048 (dim) and 0.027 (bright floor).
  double P[3][3];
  noise_stationary_cov(m, 1e16 * 1e-15, P);
  CHECK(std::sqrt(P[2][2]) == doctest::Approx(0.048).epsilon(0.02));
  noise_stationary_cov(m, 1e16 * 1e-9, P);
  CHECK(std::sqrt(P[2][2]) == doctest::Approx(0.0268).epsilon(0.02));
}

TEST_CASE("tables interpolate within 2e-4 relative and the bridge reproduces the exact law (M3 V7b)") {
  const NoiseModel m = prophesee_like();
  const NoiseTables tab(m);
  const NoiseLayout lay = tab.layout();
  NoiseTableView<double> v{tab.data64().data(), tab.n_nodes(), lay.stride(), m.levels, tab.ln_a1_min(), tab.inv_dln()};
  // Bridge of level l at a table position, against the exact law at a1: with x_L = 0, x_m = B x_R + L xi
  // and x_R ~ N(0, Q_{l-1}), so Cov(x_m) must be Q_l and Cov(x_m, x_R) must be Q_l Phi_l^T; the mean
  // of x_m given both ends must satisfy A + B Phi_{l-1} = Phi_l.
  struct BridgeErr {
    double law = 0, cross = 0, mean = 0;
  };
  const auto bridge_err = [&](NoisePos<double> pos, double a1, int l, BridgeErr& e) {
    double b[24];
    noise_lookup(v, pos, lay.bridge(l), 24, b);
    const NoiseMatrices tc = noise_transition(m, a1, 65.536e-6 / (1 << l));
    const NoiseMatrices tp = noise_transition(m, a1, 65.536e-6 / (1 << (l - 1)));
    const double* Lb = b + 18;
    const double Lrow[3][3] = {{Lb[0], 0, 0}, {Lb[1], Lb[2], 0}, {Lb[3], Lb[4], Lb[5]}};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        double cov = 0, cross = 0, want_cross = 0, mean = b[3 * i + j];
        for (int k = 0; k < 3; ++k) {
          for (int q = 0; q < 3; ++q) cov += b[9 + 3 * i + k] * tp.q[k][q] * b[9 + 3 * j + q];
          cov += Lrow[i][k] * Lrow[j][k];
          cross += b[9 + 3 * i + k] * tp.q[k][j];
          want_cross += tc.q[i][k] * tc.phi[j][k];
          mean += b[9 + 3 * i + k] * tp.phi[k][j];
        }
        e.law = std::max(e.law, std::fabs(cov - tc.q[i][j]) / std::sqrt(tc.q[i][i] * tc.q[j][j]));
        e.cross = std::max(e.cross, std::fabs(cross - want_cross) / std::sqrt(tc.q[i][i] * tp.q[j][j]));
        e.mean = std::max(e.mean, std::fabs(mean - tc.phi[i][j]));
      }
  };
  double worst_q = 0;
  BridgeErr at_node, between;
  for (int node = 0; node + 1 < tab.n_nodes(); node += 11) {
    const double ln_a1 = tab.ln_a1_min() + (node + 0.5) / tab.inv_dln();  // midpoint between nodes
    const double a1 = std::exp(ln_a1);
    const NoisePos<double> pos = noise_pos(v, ln_a1);
    double L[6];
    noise_lookup(v, pos, NoiseLayout::kStepChol, 6, L);
    const double LL[3][3] = {{L[0] * L[0], L[0] * L[1], L[0] * L[3]},
                             {L[0] * L[1], L[1] * L[1] + L[2] * L[2], L[1] * L[3] + L[2] * L[4]},
                             {L[0] * L[3], L[1] * L[3] + L[2] * L[4], L[3] * L[3] + L[4] * L[4] + L[5] * L[5]}};
    const NoiseMatrices ex = noise_transition(m, a1, 65.536e-6);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        worst_q = std::max(worst_q, std::fabs(LL[i][j] - ex.q[i][j]) / std::sqrt(ex.q[i][i] * ex.q[j][j]));
    // Every level, exactly at the node and interpolated between nodes.
    const double a1n = std::exp(tab.ln_a1_min() + node / tab.inv_dln());
    for (int l = 1; l <= m.levels; ++l) {
      bridge_err({node, 0.0}, a1n, l, at_node);
      bridge_err(pos, a1, l, between);
    }
  }
  MESSAGE("interpolated chol Q rel err " << worst_q << "; bridge law, cross and |A + B Phi_p - Phi_c| at the nodes "
                                         << at_node.law << ", " << at_node.cross << ", " << at_node.mean
                                         << ", between them " << between.law << ", " << between.cross << ", "
                                         << between.mean);
  CHECK(worst_q < 2e-4);
  CHECK(at_node.law < 1e-6);
  CHECK(at_node.cross < 1e-6);
  CHECK(at_node.mean < 1e-6);
  CHECK(between.law < 2e-4);
  CHECK(between.cross < 2e-4);
  CHECK(between.mean < 2e-4);
}

TEST_CASE("split tables: rho Q1 + Q2 is the law of a cell with diffusion rho D1, and the kernel forms it (D-P2-2)") {
  // A cell whose diffusion is rho D1 is the cell of a model with k q scaled by rho (D1 = 2 k q a1): its exact law comes
  // from noise_transition of that model. The kernel's lattice factor (split_chol) and bridge (bridge_midpoint_split)
  // from the split tables of the base model must reproduce it, in fp64 and fp32, and the base margins scaled by
  // sqrt(max(1, rho)) must bound the scaled model's bridge residual.
  NoiseModel m = prophesee_like();
  m.split = true;
  const NoiseTables tab(m);
  const NoiseLayout lay = tab.layout();
  CHECK(lay.stride() == 18 + 47 * m.levels + 12 + 19 * m.levels);
  const NoiseTableView<double> v64{tab.data64().data(), tab.n_nodes(), lay.stride(), m.levels, tab.ln_a1_min(),
                                   tab.inv_dln()};
  const NoiseTableView<float> v32{tab.data32().data(), tab.n_nodes(), lay.stride(), m.levels,
                                  static_cast<float>(tab.ln_a1_min()), static_cast<float>(tab.inv_dln())};
  const auto scaled = [](const double a[3][3], int i, int j, double x) {
    return std::fabs(x - a[i][j]) / std::sqrt(a[i][i] * a[j][j]);
  };
  // Sym packed -> full.
  const auto full = [](const auto* p, double out[3][3]) {
    const int idx[3][3] = {{0, 1, 3}, {1, 2, 4}, {3, 4, 5}};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) out[i][j] = static_cast<double>(p[idx[i][j]]);
  };
  const auto lower = [](const auto* p, double out[3][3]) {
    const int idx[3][3] = {{0, -1, -1}, {1, 2, -1}, {3, 4, 5}};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) out[i][j] = idx[i][j] < 0 ? 0.0 : static_cast<double>(p[idx[i][j]]);
  };
  double parts = 0, phis = 0;
  double fac[2] = {0, 0}, law[2] = {0, 0}, cross[2] = {0, 0}, mean[2] = {0, 0}, meanz[2] = {0, 0};
  double margin = 0, tight = 0;
  for (int node = 0; node < tab.n_nodes(); node += 23) {
    const double a1 = std::exp(tab.ln_a1_min() + node / tab.inv_dln());
    const double* row = tab.data64().data() + static_cast<size_t>(node) * lay.stride();
    for (int l = 0; l <= m.levels; ++l) {
      const NoiseMatrices ex = noise_transition(m, a1, 65.536e-6 / (1 << l));
      const double* b = row + (l == 0 ? lay.split_step() : lay.split_level(l));
      double q1[3][3], q2[3][3], ph[3][3];
      full(b + (l == 0 ? 0 : 6), q1);
      full(b + (l == 0 ? 6 : 12), q2);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) parts = std::max(parts, scaled(ex.q, i, j, q1[i][j] + q2[i][j]));
      if (l > 0) {
        lower(b, ph);
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) phis = std::max(phis, std::fabs(ph[i][j] - ex.phi[i][j]));
      }
    }
    for (const double rho : {0.5, 0.71, 1.9, 10.0, 120.0}) {
      NoiseModel mr = prophesee_like();
      mr.kq *= rho;
      NoiseModel mrt = mr;
      mrt.levels = m.levels;
      const NoiseTables tr(mrt);
      const double* rr = tr.data64().data() + static_cast<size_t>(node) * tr.layout().stride();
      const auto run = [&](auto zero, const auto& v, int prec) {
        using T = decltype(zero);
        const NoisePos<T> pos{node, T(0)};
        // The lattice step's factor.
        T q12[12], c[6];
        noise_lookup(v, pos, lay.split_step(), 12, q12);
        split_chol(q12, static_cast<T>(rho), c);
        double C[3][3];
        lower(c, C);
        const NoiseMatrices e0 = noise_transition(mr, a1, 65.536e-6);
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) {
            double cc = 0;
            for (int k = 0; k < 3; ++k) cc += C[i][k] * C[j][k];
            fac[prec] = std::max(fac[prec], scaled(e0.q, i, j, cc));
          }
        // The bridges: x_m = A x_L + B x_R + L xi, read off with unit inputs.
        for (int l = 1; l <= m.levels; ++l) {
          T blk[18];
          noise_lookup(v, pos, lay.split_level(l), 18, blk);
          double A[3][3], B[3][3], L[3][3];
          for (int j = 0; j < 3; ++j) {
            T e[3] = {T(0), T(0), T(0)}, z[3] = {T(0), T(0), T(0)}, out[3];
            e[j] = T(1);
            bridge_midpoint_split(blk, static_cast<T>(rho), e, z, z, out);
            for (int i = 0; i < 3; ++i) A[i][j] = static_cast<double>(out[i]);
            bridge_midpoint_split(blk, static_cast<T>(rho), z, e, z, out);
            for (int i = 0; i < 3; ++i) B[i][j] = static_cast<double>(out[i]);
            bridge_midpoint_split(blk, static_cast<T>(rho), z, z, e, out);
            for (int i = 0; i < 3; ++i) L[i][j] = static_cast<double>(out[i]);
          }
          const NoiseMatrices tc = noise_transition(mr, a1, 65.536e-6 / (1 << l));
          const NoiseMatrices tp = noise_transition(mr, a1, 65.536e-6 / (1 << (l - 1)));
          // The mean's error in units of the conditional std, for endpoints of the stationary law: with the exact
          // weights (Ae, Be) of the long-double tables of the scaled model, d = dA x_L + dB x_R has covariance
          // dA P dA^T + dB P dB^T + dA P Phi_p^T dB^T + (.)^T, against Sigma_m = Le Le^T.
          {
            const double* eb = rr + tr.layout().bridge(l);
            double dA[3][3], dB[3][3], P[3][3], CLR[3][3], S[3][3];
            for (int i = 0; i < 3; ++i)
              for (int j = 0; j < 3; ++j) {
                dA[i][j] = A[i][j] - eb[3 * i + j];
                dB[i][j] = B[i][j] - eb[9 + 3 * i + j];
              }
            noise_stationary_cov(mr, a1, P);
            for (int i = 0; i < 3; ++i)
              for (int j = 0; j < 3; ++j) {
                CLR[i][j] = 0;
                for (int k = 0; k < 3; ++k) CLR[i][j] += P[i][k] * tp.phi[j][k];  // Cov(x_L, x_R)
              }
            double Le[3][3];
            lower(eb + 18, Le);
            for (int i = 0; i < 3; ++i) {
              S[i][i] = 0;
              for (int k = 0; k < 3; ++k) S[i][i] += Le[i][k] * Le[i][k];
              double var = 0;
              for (int a = 0; a < 3; ++a)
                for (int b2 = 0; b2 < 3; ++b2)
                  var += dA[i][a] * P[a][b2] * dA[i][b2] + dB[i][a] * P[a][b2] * dB[i][b2] +
                         2 * dA[i][a] * CLR[a][b2] * dB[i][b2];
              meanz[prec] = std::max(meanz[prec], std::sqrt(std::max(var, 0.0) / S[i][i]));
            }
          }
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
              double cov = 0, cr = 0, want_cr = 0, mn = A[i][j];
              for (int k = 0; k < 3; ++k) {
                for (int q = 0; q < 3; ++q) cov += B[i][k] * tp.q[k][q] * B[j][q];
                cov += L[i][k] * L[j][k];
                cr += B[i][k] * tp.q[k][j];
                want_cr += tc.q[i][k] * tc.phi[j][k];
                mn += B[i][k] * tp.phi[k][j];
              }
              law[prec] = std::max(law[prec], scaled(tc.q, i, j, cov));
              cross[prec] = std::max(cross[prec], std::fabs(cr - want_cr) / std::sqrt(tc.q[i][i] * tp.q[j][j]));
              mean[prec] = std::max(mean[prec], std::fabs(mn - tc.phi[i][j]));
            }
        }
      };
      run(0.0, v64, 0);
      run(0.0f, v32, 1);
      // Margins: the scaled model's bridge residual within the kernel's bound, sigma_br(1) sqrt(1 + (rho - 1) f_l) for
      // rho > 1 (f_l the input node's share, in [0, 1]) and sigma_br(1) below, every level.
      for (int l = 0; l < m.levels; ++l) {
        const double base = row[lay.margin(l)], got = rr[tr.layout().margin(l)], f_l = row[lay.split_margin(l)];
        CHECK(f_l >= 0.0);
        CHECK(f_l <= 1.0);
        const double bound = base * std::sqrt(rho > 1 ? 1 + (rho - 1) * f_l : 1.0);
        margin = std::max(margin, got / bound);
        tight = std::max(tight, std::sqrt(std::max(1.0, rho)) * base / bound);
      }
    }
  }
  MESSAGE("split parts " << parts << ", Phi_l " << phis << "; lattice factor fp64 / fp32 " << fac[0] << " / " << fac[1]
                         << "; bridge law " << law[0] << " / " << law[1] << ", cross " << cross[0] << " / "
                         << cross[1] << ", mean " << mean[0] << " / " << mean[1] << " (in conditional sigma "
                         << meanz[0] << " / " << meanz[1] << ")"
                         << "; worst sigma_br(rho) / its bound " << margin << " (the bound is up to " << tight
                         << " x tighter than sqrt(max(1, rho)))");
  CHECK(parts < 1e-12);
  CHECK(phis < 1e-12);
  CHECK(fac[0] < 1e-12);
  CHECK(fac[1] < 1e-5);
  CHECK(law[0] < 1e-9);
  CHECK(cross[0] < 1e-9);
  CHECK(mean[0] < 1e-9);
  CHECK(law[1] < 1e-4);
  CHECK(cross[1] < 1e-4);
  CHECK(meanz[0] < 1e-6);
  CHECK(meanz[1] < 1e-2);
  CHECK(margin <= 1.0 + 1e-9);
}

TEST_CASE("exclusion margins whiten the endpoint pair and bound the bridge residual (M3 D.8)") {
  // Per node and level l (cells of h = 65.536 us / 2^l): W C_ee W^T = I for the stationary endpoint
  // covariance C_ee of (x_L, x_R); sigma_br, the largest bridge residual std of n3 over the cell, is
  // attained at the midpoint, where the level-(l+1) bridge gives it exactly; and g is at least, and
  // close to, the midpoint's mean mismatch sqrt(dw^T C_ee dw), dw = row 2 of [A | B] minus the Hermite
  // weights of n3 at u = 1/2 (M3 D.7).
  const NoiseModel m = prophesee_like();
  const NoiseTables tab(m);
  const NoiseLayout lay = tab.layout();
  std::mt19937_64 rng(4);
  std::normal_distribution<double> normal;
  double worst_w = 0, sig_lo = 2, sig_hi = 0, g_lo = 2, g_hi = 0, worst_reader = 0;
  for (int node = 0; node < tab.n_nodes(); node += 7) {
    const double a1 = std::exp(tab.ln_a1_min() + node / tab.inv_dln());
    const double* row = tab.data64().data() + static_cast<size_t>(node) * lay.stride();
    double P[3][3];
    noise_stationary_cov(m, a1, P);
    for (int l = 0; l < m.levels; ++l) {
      const double h = 65.536e-6 / (1 << l);
      const double* mg = row + lay.margin(l);
      const NoiseMatrices t = noise_transition(m, a1, h);
      double C[6][6], W[6][6] = {};
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          double phiP = 0;  // Cov(x_R, x_L) = Phi P
          for (int k = 0; k < 3; ++k) phiP += t.phi[i][k] * P[k][j];
          C[i][j] = C[i + 3][j + 3] = P[i][j];
          C[i + 3][j] = C[j][i + 3] = phiP;
        }
      for (int i = 0, idx = 2; i < 6; ++i)  // row-major lower-packed (NoiseLayout)
        for (int j = 0; j <= i; ++j) W[i][j] = mg[idx++];
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
          double s = 0;
          for (int k = 0; k < 6; ++k)
            for (int q = 0; q < 6; ++q) s += W[i][k] * C[k][q] * W[j][q];
          worst_w = std::max(worst_w, std::fabs(s - (i == j ? 1.0 : 0.0)));
        }
      const double* b = row + lay.bridge(l + 1);
      const double r = mg[0] / std::sqrt(b[21] * b[21] + b[22] * b[22] + b[23] * b[23]);
      sig_lo = std::min(sig_lo, r);
      sig_hi = std::max(sig_hi, r);
      const double e = h * m.a3 / 8;  // Hermite at u = 1/2: H00 = H01 = 1/2, H10 = -H11 = 1/8
      const double wh[6] = {0, e, 0.5 - e, 0, -e, 0.5 + e};
      double dw[6], g2 = 0;
      for (int j = 0; j < 3; ++j) {
        dw[j] = b[6 + j] - wh[j];
        dw[j + 3] = b[15 + j] - wh[j + 3];
      }
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) g2 += dw[i] * C[i][j] * dw[j];
      const double ratio = mg[1] / std::sqrt(g2);
      g_lo = std::min(g_lo, ratio);
      g_hi = std::max(g_hi, ratio);
      // The kernel's reader follows the same packing.
      double ep[6], norm2 = 0;
      for (double& x : ep) x = normal(rng) * std::sqrt(P[2][2]);
      for (int i = 0; i < 6; ++i) {
        double s = 0;
        for (int j = 0; j <= i; ++j) s += W[i][j] * ep[j];
        norm2 += s * s;
      }
      const double want = m.margin_z * mg[0] + mg[1] * std::sqrt(norm2);
      worst_reader = std::max(worst_reader, std::fabs(noise_margin<double>(mg, m.margin_z, ep, ep + 3) - want) / want);
    }
  }
  MESSAGE("max |W C W^T - I| " << worst_w << "; sigma_br / midpoint std in [" << sig_lo << ", " << sig_hi
                               << "]; g / midpoint g in [" << g_lo << ", " << g_hi << "]; reader rel err "
                               << worst_reader);
  CHECK(worst_w < 1e-7);
  CHECK(sig_lo >= 1 - 1e-9);
  CHECK(sig_hi <= 1 + 1e-3);
  CHECK(g_lo >= 1 - 1e-6);
  CHECK(g_hi <= 1.05);
  CHECK(worst_reader < 1e-12);
}

TEST_CASE("keyed normals are standard, order-independent and pinned") {
  // Golden first outputs pin the sensor purpose tags and the counter layout of core/rng/purpose.h
  // (append-only registry: these values may never change).
  double xi[3];
  normals3<double>(0, Purpose::kNoise, 0, 0, 0, xi);
  CHECK(xi[0] == 0x1.9314987575cccp+0);
  CHECK(xi[2] == 0x1.907c8fda1f566p-3);
  normals3<double>(0x0123456789abcdefull, Purpose::kNoise, 1, 307199, 65536, xi);
  CHECK(xi[2] == -0x1.65145c21ebbb9p+1);
  normals3<double>(0x0123456789abcdefull, Purpose::kNoiseInit, 1, 307199, 65536, xi);
  CHECK(xi[0] == -0x1.5de1b9dc29e54p-1);
  CHECK(balla::rng::draw(0x0123456789abcdefull, Purpose::kSensorMap, 2, 307199, 1).v[3] == 0x48fca0b4u);
  CHECK(balla::rng::draw(0x0123456789abcdefull, Purpose::kReadoutPhase, 2, 0, 0).v[0] == 0x0e79efaeu);
  // Distinct node times, pixels and purposes address distinct draws.
  double a[3], b[3];
  normals3<double>(9, Purpose::kNoise, 0, 5, 65536, a);
  normals3<double>(9, Purpose::kNoise, 0, 5, 131072, b);
  CHECK(a[0] != b[0]);
  normals3<double>(9, Purpose::kNoiseInit, 0, 5, 65536, b);
  CHECK(a[0] != b[0]);
  double s = 0, s2 = 0;
  const int n = 200000;
  for (int i = 0; i < n; ++i) {
    normals3<double>(42, Purpose::kNoise, 0, static_cast<uint32_t>(i % 1000), static_cast<uint64_t>(i / 1000) << 16,
                     xi);
    for (double x : xi) {
      s += x;
      s2 += x * x;
    }
  }
  CHECK(std::fabs(s / (3 * n)) < 0.01);
  CHECK(std::fabs(s2 / (3 * n) - 1) < 0.01);
  // fp32 normals use 23-bit uniforms: the same law, bounded at 5.8 sigma.
  double f1 = 0, f2 = 0, fmax = 0;
  float xf[3];
  for (int i = 0; i < n; ++i) {
    normals3<float>(42, Purpose::kNoise, 0, static_cast<uint32_t>(i % 1000), static_cast<uint64_t>(i / 1000) << 16,
                    xf);
    for (float x : xf) {
      f1 += static_cast<double>(x);
      f2 += static_cast<double>(x) * static_cast<double>(x);
      fmax = std::max(fmax, std::fabs(static_cast<double>(x)));
    }
  }
  CHECK(std::fabs(f1 / (3 * n)) < 0.01);
  CHECK(std::fabs(f2 / (3 * n) - 1) < 0.01);
  CHECK(fmax < 5.8);
}
