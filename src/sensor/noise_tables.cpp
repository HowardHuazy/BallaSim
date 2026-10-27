#include "sensor/noise_tables.h"

#include <cmath>
#include <stdexcept>

#include "core/pmath/explog.h"

namespace balla::sensor {

namespace {

using LD = long double;

struct M3 {
  LD m[3][3] = {};
};

M3 mul(const M3& a, const M3& b) {
  M3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      LD s = 0;
      for (int k = 0; k < 3; ++k) s += a.m[i][k] * b.m[k][j];
      r.m[i][j] = s;
    }
  return r;
}

M3 transpose(const M3& a) {
  M3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r.m[i][j] = a.m[j][i];
  return r;
}

M3 add(const M3& a, const M3& b) {
  M3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r.m[i][j] = a.m[i][j] + b.m[i][j];
  return r;
}

M3 scaled(const M3& a, LD s) {
  M3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r.m[i][j] = a.m[i][j] * s;
  return r;
}

struct System {
  M3 A, GG;
};

// The noise sources of a system: both (the model), or only the input node's (D1) or the output node's (D2), whose
// covariances Q1 and Q2 sum to Q (PHASE2 D-P2-2: a cell with diffusion rho D1 has rho Q1 + Q2).
enum class Sources { kBoth, kInput, kOutput };

System make_system(const NoiseModel& m, LD a1, Sources src = Sources::kBoth) {
  System s;
  s.A.m[0][0] = -a1;
  s.A.m[1][0] = m.a2;
  s.A.m[1][1] = -m.a2;
  s.A.m[2][1] = m.a3;
  s.A.m[2][2] = -m.a3;
  // D1 = 2 k q a1 keeps Var(n1) = k q (M3 D.1).
  if (src != Sources::kOutput) s.GG.m[0][0] = 2 * static_cast<LD>(m.kq) * a1;
  if (src != Sources::kInput) s.GG.m[1][1] = 2 * static_cast<LD>(m.sigma_o) * m.sigma_o * m.a2;
  return s;
}

// Phi(h) = e^{A h} and Q(h) = int_0^h e^{A s} G G^T e^{A^T s} ds by scaling and squaring.
void transition_ld(const System& sys, LD h, M3& phi, M3& q) {
  LD norm = 0;
  for (int i = 0; i < 3; ++i) {
    LD row = 0;
    for (int j = 0; j < 3; ++j) row += std::fabs(sys.A.m[i][j]);
    norm = std::max(norm, row);
  }
  int s = 0;
  LD delta = h;
  while (norm * delta > LD(1) / 64 && s < 200) {
    delta /= 2;
    ++s;
  }
  // Taylor series on the scaled step; ||A delta|| <= 1/64 makes 16 terms exact to long double.
  const M3 Ad = scaled(sys.A, delta);
  M3 term;
  for (int i = 0; i < 3; ++i) term.m[i][i] = 1;
  phi = term;
  for (int n = 1; n <= 16; ++n) {
    term = scaled(mul(term, Ad), LD(1) / n);
    phi = add(phi, term);
  }
  // Q(delta) = sum_n M_n delta^(n+1) / (n+1)!, M_0 = G G^T, M_n = A M_{n-1} + M_{n-1} A^T.
  M3 Mn = sys.GG;
  LD coef = delta;
  q = scaled(Mn, coef);
  const M3 At = transpose(sys.A);
  for (int n = 1; n <= 16; ++n) {
    Mn = add(mul(sys.A, Mn), mul(Mn, At));
    coef = coef * delta / (n + 1);
    q = add(q, scaled(Mn, coef));
  }
  for (int k = 0; k < s; ++k) {
    q = add(mul(mul(phi, q), transpose(phi)), q);  // Q(2h) = Phi(h) Q(h) Phi(h)^T + Q(h)
    phi = mul(phi, phi);
  }
}

// Closed-form stationary covariance (lower bidiagonal Lyapunov equation, solved row by row; it
// reproduces M3 A.12 for P33).
M3 stationary_ld(const NoiseModel& m, LD a1) {
  const LD a2 = m.a2, a3 = m.a3;
  M3 p;
  p.m[0][0] = m.kq;
  p.m[1][0] = a2 * p.m[0][0] / (a1 + a2);
  p.m[1][1] = p.m[1][0] + static_cast<LD>(m.sigma_o) * m.sigma_o;
  p.m[2][0] = a3 * p.m[1][0] / (a1 + a3);
  p.m[2][1] = (a3 * p.m[1][1] + a2 * p.m[2][0]) / (a2 + a3);
  p.m[2][2] = p.m[2][1];
  p.m[0][1] = p.m[1][0];
  p.m[0][2] = p.m[2][0];
  p.m[1][2] = p.m[2][1];
  return p;
}

// Cholesky factor of an n x n symmetric positive semidefinite matrix (row-major), with pivots
// clamped at a tiny fraction of the trace so a numerically singular matrix still factors.
template <int N>
void cholesky(const LD (&a)[N][N], LD (&l)[N][N]) {
  LD trace = 0;
  for (int i = 0; i < N; ++i) trace += std::fabs(a[i][i]);
  const LD tiny = trace * LD(1e-36) + LD(1e-300);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) l[i][j] = 0;
  for (int j = 0; j < N; ++j) {
    LD d = a[j][j];
    for (int k = 0; k < j; ++k) d -= l[j][k] * l[j][k];
    d = d > tiny ? d : 0;
    l[j][j] = std::sqrt(d);
    for (int i = j + 1; i < N; ++i) {
      LD s = a[i][j];
      for (int k = 0; k < j; ++k) s -= l[i][k] * l[j][k];
      l[i][j] = l[j][j] > 0 ? s / l[j][j] : 0;
    }
  }
}

// Solves (L L^T) x = b for a Cholesky factor L.
template <int N>
void chol_solve(const LD (&l)[N][N], const LD (&b)[N], LD (&x)[N]) {
  LD y[N];
  for (int i = 0; i < N; ++i) {
    LD s = b[i];
    for (int k = 0; k < i; ++k) s -= l[i][k] * y[k];
    y[i] = l[i][i] > 0 ? s / l[i][i] : 0;
  }
  for (int i = N - 1; i >= 0; --i) {
    LD s = y[i];
    for (int k = i + 1; k < N; ++k) s -= l[k][i] * x[k];
    x[i] = l[i][i] > 0 ? s / l[i][i] : 0;
  }
}

void put_lower3(const M3& m, double* out) {
  out[0] = static_cast<double>(m.m[0][0]);
  out[1] = static_cast<double>(m.m[1][0]);
  out[2] = static_cast<double>(m.m[1][1]);
  out[3] = static_cast<double>(m.m[2][0]);
  out[4] = static_cast<double>(m.m[2][1]);
  out[5] = static_cast<double>(m.m[2][2]);
}

M3 chol3(const M3& a) {
  LD in[3][3], l[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) in[i][j] = 0.5L * (a.m[i][j] + a.m[j][i]);
  cholesky<3>(in, l);
  M3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r.m[i][j] = l[i][j];
  return r;
}

M3 inverse_spd(const M3& a) {
  LD in[3][3], l[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) in[i][j] = 0.5L * (a.m[i][j] + a.m[j][i]);
  cholesky<3>(in, l);
  M3 r;
  for (int j = 0; j < 3; ++j) {
    LD e[3] = {0, 0, 0}, x[3];
    e[j] = 1;
    chol_solve<3>(l, e, x);
    for (int i = 0; i < 3; ++i) r.m[i][j] = x[i];
  }
  return r;
}

// Margin quantities of a level-l node of length h (M3 D.8): max over the node of the bridge
// residual std of n3, of the endpoint-dependent mean mismatch g, and W = chol(C_ee)^-1.
void margin_ld(const NoiseModel& m, const System& sys, LD a1, LD h, double* out) {
  const M3 P = stationary_ld(m, a1);
  // Interior points u = i / 20: Phi(i h / 20) as powers of one transition.
  constexpr int kPoints = 20;
  M3 phi_step, q_unused;
  transition_ld(sys, h / kPoints, phi_step, q_unused);
  M3 powers[kPoints + 1];
  for (int i = 0; i < 3; ++i) powers[0].m[i][i] = 1;
  for (int i = 1; i <= kPoints; ++i) powers[i] = mul(powers[i - 1], phi_step);
  const M3& phi_h = powers[kPoints];
  LD Cee[6][6];
  const M3 PphiT = mul(P, transpose(phi_h));
  const M3 phiP = mul(phi_h, P);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      Cee[i][j] = P.m[i][j];
      Cee[i][j + 3] = PphiT.m[i][j];
      Cee[i + 3][j] = phiP.m[i][j];
      Cee[i + 3][j + 3] = P.m[i][j];
    }
  LD L6[6][6];
  cholesky<6>(Cee, L6);
  LD sigma_br = 0, g = 0;
  const LD a3 = m.a3;
  for (int iu = 1; iu < kPoints; ++iu) {
    const LD u = static_cast<LD>(iu) / kPoints;
    const M3 cov_tL = mul(powers[iu], P);                         // Cov(x(t), x_L)
    const M3 cov_tR = mul(P, transpose(powers[kPoints - iu]));    // Cov(x(t), x_R)
    LD c[6], w[6];
    for (int j = 0; j < 3; ++j) {
      c[j] = cov_tL.m[2][j];
      c[j + 3] = cov_tR.m[2][j];
    }
    chol_solve<6>(L6, c, w);
    LD var_br = P.m[2][2];
    for (int j = 0; j < 6; ++j) var_br -= c[j] * w[j];
    // Hermite weights of n3(t) on (x_L, x_R) with slopes n3' = a3 (n2 - n3) (M3 D.7).
    const LD H00 = 2 * u * u * u - 3 * u * u + 1, H10 = u * u * u - 2 * u * u + u;
    const LD H01 = -2 * u * u * u + 3 * u * u, H11 = u * u * u - u * u;
    const LD wh[6] = {0, H10 * h * a3, H00 - H10 * h * a3, 0, H11 * h * a3, H01 - H11 * h * a3};
    LD dw[6], g2 = 0;
    for (int j = 0; j < 6; ++j) dw[j] = w[j] - wh[j];
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) g2 += dw[i] * Cee[i][j] * dw[j];
    sigma_br = std::max(sigma_br, std::sqrt(std::max(var_br, LD(0))));
    g = std::max(g, std::sqrt(std::max(g2, LD(0))));
  }
  out[0] = static_cast<double>(sigma_br);
  out[1] = static_cast<double>(g);
  // W = L6^-1 (lower triangular), packed row-major.
  LD W[6][6] = {};
  for (int j = 0; j < 6; ++j) {
    W[j][j] = L6[j][j] > 0 ? 1 / L6[j][j] : 0;
    for (int i = j + 1; i < 6; ++i) {
      LD s = 0;
      for (int k = j; k < i; ++k) s -= L6[i][k] * W[k][j];
      W[i][j] = L6[i][i] > 0 ? s / L6[i][i] : 0;
    }
  }
  int idx = 2;
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j <= i; ++j) out[idx++] = static_cast<double>(W[i][j]);
}

// The input node's largest share of the bridge residual of n3 on a level-l node of length h (PHASE2 D-P2-2): with the
// rho = 1 predictor w (the optimal weights of margin_ld) held fixed, the residual n3(t) - w.(x_L, x_R) has variance
// V1 + V2, one part per noise source (the stationary law, the transitions and the source are linear in each); a cell
// whose input-node diffusion is rho D1 then has
//   sigma_br(rho)^2 <= max_t (rho V1 + V2) <= sigma_br(1)^2 (1 + (rho - 1) f),  f = max_t V1 / (V1 + V2).
LD margin_share_ld(const NoiseModel& m, const System& sys, LD a1, LD h) {
  NoiseModel m1 = m, m2 = m;
  m1.sigma_o = 0;
  m2.kq = 0;
  const M3 P = stationary_ld(m, a1), P1 = stationary_ld(m1, a1), P2 = stationary_ld(m2, a1);
  constexpr int kPoints = 20;
  M3 phi_step, q_unused;
  transition_ld(sys, h / kPoints, phi_step, q_unused);
  M3 powers[kPoints + 1];
  for (int i = 0; i < 3; ++i) powers[0].m[i][i] = 1;
  for (int i = 1; i <= kPoints; ++i) powers[i] = mul(powers[i - 1], phi_step);
  const M3& phi_h = powers[kPoints];
  const auto cee = [&](const M3& Pk, LD (&C)[6][6]) {
    const M3 PphiT = mul(Pk, transpose(phi_h)), phiP = mul(phi_h, Pk);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        C[i][j] = Pk.m[i][j];
        C[i][j + 3] = PphiT.m[i][j];
        C[i + 3][j] = phiP.m[i][j];
        C[i + 3][j + 3] = Pk.m[i][j];
      }
  };
  LD C[6][6], C1[6][6], C2[6][6], L6[6][6];
  cee(P, C);
  cee(P1, C1);
  cee(P2, C2);
  cholesky<6>(C, L6);
  LD f = 0;
  for (int iu = 1; iu < kPoints; ++iu) {
    const auto c_of = [&](const M3& Pk, LD (&c)[6]) {
      const M3 cov_tL = mul(powers[iu], Pk), cov_tR = mul(Pk, transpose(powers[kPoints - iu]));
      for (int j = 0; j < 3; ++j) {
        c[j] = cov_tL.m[2][j];
        c[j + 3] = cov_tR.m[2][j];
      }
    };
    LD c[6], c1[6], c2[6], w[6];
    c_of(P, c);
    c_of(P1, c1);
    c_of(P2, c2);
    chol_solve<6>(L6, c, w);
    const auto resid = [&](const M3& Pk, const LD (&ck)[6], const LD (&Ck)[6][6]) {
      LD v = Pk.m[2][2];
      for (int i = 0; i < 6; ++i) {
        v -= 2 * w[i] * ck[i];
        for (int j = 0; j < 6; ++j) v += w[i] * Ck[i][j] * w[j];
      }
      return std::max(v, LD(0));
    };
    const LD v1 = resid(P1, c1, C1), v2 = resid(P2, c2, C2);
    if (v1 + v2 > 0) f = std::max(f, v1 / (v1 + v2));
  }
  return f;
}

}  // namespace

NoiseMatrices noise_transition(const NoiseModel& m, double a1, double h_s) {
  const System sys = make_system(m, a1);
  M3 phi, q;
  transition_ld(sys, h_s, phi, q);
  NoiseMatrices r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      r.phi[i][j] = static_cast<double>(phi.m[i][j]);
      r.q[i][j] = static_cast<double>(q.m[i][j]);
    }
  return r;
}

void noise_stationary_cov(const NoiseModel& m, double a1, double p[3][3]) {
  const M3 P = stationary_ld(m, a1);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) p[i][j] = static_cast<double>(P.m[i][j]);
}

NoiseTables::NoiseTables(const NoiseModel& model) : model_(model) {
  if (model.levels < 0 || model.levels > 10) throw std::invalid_argument("noise tables: levels must lie in [0, 10]");
  const double dln = pmath::log(10.0) / kNoiseNodesPerDecade;
  inv_dln_ = 1.0 / dln;
  n_nodes_ = static_cast<int>(std::lround((kNoiseLnA1Max - kNoiseLnA1Min) * inv_dln_)) + 1;
  const NoiseLayout lay = layout();
  const int stride = lay.stride();
  data64_.assign(static_cast<size_t>(n_nodes_) * stride, 0.0);
  const LD h0 = static_cast<LD>(kNoiseCellNs) * 1e-9L;
  for (int node = 0; node < n_nodes_; ++node) {
    const LD a1 = std::exp(static_cast<LD>(kNoiseLnA1Min) + static_cast<LD>(node) * static_cast<LD>(dln));
    double* row = data64_.data() + static_cast<size_t>(node) * stride;
    const System sys = make_system(model, a1);
    // Transitions at every level: Phi_l, Q_l for cells of h0 / 2^l.
    std::vector<M3> phis(model.levels + 1), qs(model.levels + 1);
    for (int l = 0; l <= model.levels; ++l) transition_ld(sys, h0 / static_cast<LD>(int64_t{1} << l), phis[l], qs[l]);
    put_lower3(phis[0], row + NoiseLayout::kStepPhi);
    put_lower3(chol3(qs[0]), row + NoiseLayout::kStepChol);
    put_lower3(chol3(stationary_ld(model, a1)), row + NoiseLayout::kPinfChol);
    for (int l = 1; l <= model.levels; ++l) {
      // Midpoint of a level-(l-1) node given both ends (M3 D.5): the parent's covariance is
      // Phi_l Q_l Phi_l^T + Q_l, exactly the doubling step of its own construction.
      const M3& ph = phis[l];
      const M3& qh = qs[l];
      const M3 parent = add(mul(mul(ph, qh), transpose(ph)), qh);
      const M3 K = mul(mul(qh, transpose(ph)), inverse_spd(parent));
      const M3 Am = add(ph, scaled(mul(mul(K, ph), ph), -1));
      M3 sig = add(qh, scaled(mul(mul(K, ph), qh), -1));
      double* b = row + lay.bridge(l);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          b[3 * i + j] = static_cast<double>(Am.m[i][j]);
          b[9 + 3 * i + j] = static_cast<double>(K.m[i][j]);
        }
      put_lower3(chol3(sig), b + 18);
    }
    for (int l = 0; l < model.levels; ++l) {
      margin_ld(model, sys, a1, h0 / static_cast<LD>(int64_t{1} << l), row + lay.margin(l));
    }
    if (model.split) {
      const System in = make_system(model, a1, Sources::kInput), out = make_system(model, a1, Sources::kOutput);
      for (int l = 0; l <= model.levels; ++l) {
        M3 phi, q1, q2;
        const LD h = h0 / static_cast<LD>(int64_t{1} << l);
        transition_ld(in, h, phi, q1);
        transition_ld(out, h, phi, q2);
        double* b = row + (l == 0 ? lay.split_step() : lay.split_level(l));
        if (l > 0) {
          put_lower3(phis[l], b);
          b += 6;
        }
        put_lower3(q1, b);
        put_lower3(q2, b + 6);
      }
      for (int l = 0; l < model.levels; ++l) {
        row[lay.split_margin(l)] =
            static_cast<double>(margin_share_ld(model, sys, a1, h0 / static_cast<LD>(int64_t{1} << l)));
      }
    }
  }
  data32_.resize(data64_.size());
  for (size_t i = 0; i < data64_.size(); ++i) data32_[i] = static_cast<float>(data64_[i]);
}

}  // namespace balla::sensor
