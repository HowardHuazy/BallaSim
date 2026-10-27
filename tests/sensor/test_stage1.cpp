// Stage 1 closed form (src/sensor/stage1.h) against an independent long-double quadrature of the
// variation-of-constants solution, and the cubic surrogate / stage 2-3 split (substep.h) against
// the exact cascade (M3 V1 T1, T4).
#include <doctest/doctest.h>

#include <cmath>
#include <random>

#include "sensor/stage1.h"
#include "sensor/substep.h"

using namespace balla::sensor;

namespace {

// z(h) = z0 e^{-P(h)} + k int_0^h e^{-k r (I_h - s r / 2)} dr (r = h - sigma), composite 8-point
// Gauss-Legendre in long double. The integrand is below e^{-k r I_min}, so r <= 90 / (k I_min).
long double z_ref(long double z0, long double I0, long double I1, long double h, long double k) {
  const long double s = (I1 - I0) / h;
  const long double P = k * h * 0.5L * (I0 + I1);
  const long double r_max = std::min(h, 90.0L / (k * std::min(I0, I1)));
  static const long double gx[4] = {0.18343464249564980493947614236018398L, 0.52553240991632898581773904918924635L,
                                    0.79666647741362673959155393647583044L, 0.96028985649753623168356086856947299L};
  static const long double gw[4] = {0.36268378337836198296515044927719561L, 0.31370664587788728733796220198660131L,
                                    0.22238103445337447054435599442624088L, 0.10122853629037625915253135430996219L};
  const int panels = 4000;
  const long double w = r_max / panels;
  long double sum = 0;
  for (int p = 0; p < panels; ++p) {
    const long double c = (p + 0.5L) * w;
    for (int i = 0; i < 4; ++i)
      for (int sgn = -1; sgn <= 1; sgn += 2) {
        const long double r = c + sgn * 0.5L * w * gx[i];
        sum += gw[i] * std::exp(-k * r * (I1 - 0.5L * s * r));
      }
  }
  return z0 * std::exp(-P) + k * sum * 0.5L * w;
}

}  // namespace

TEST_CASE("z closed form matches the long-double reference over P in [1e-7, 3e3] (M3 V1 T1)") {
  std::mt19937_64 rng(20261006);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double worst64[4] = {0, 0, 0, 0}, worst32 = 0;
  int count[4] = {0, 0, 0, 0};
  for (int trial = 0; trial < 1500; ++trial) {
    const double k = std::pow(10.0, 0.0 + 2.0 * u(rng));          // 1e15..1e17 A^-1 s^-1 in (fA s)^-1
    const double I0 = std::pow(10.0, -1.0 + 7.0 * u(rng));        // 0.1 fA .. 1 nA
    double I1 = I0 * std::exp(-3.0 + 6.0 * u(rng));
    double h = std::pow(10.0, -6.0 + 4.0 * u(rng));
    const double z0 = std::exp(-2.0 + 4.0 * u(rng)) / I0;
    if (trial % 10 == 0) I1 = I0 * (1.0 + 2e-7 * (u(rng) - 0.5));  // nearly flat
    double P = k * h * 0.5 * (I0 + I1);
    if (P > 3e3) {
      h *= 3e3 / P;
      P = 3e3;
    }
    const double ref = static_cast<double>(std::log(z_ref(z0, I0, I1, h, k)));
    const double err = std::fabs(std::log(z_step(z0, I0, I1, h, k)) - ref);
    const int bin = P < 0.5 ? 0 : (P < 10 ? 1 : (P < 1e3 ? 2 : 3));
    worst64[bin] = std::max(worst64[bin], err);
    ++count[bin];
    const float zf = z_step(static_cast<float>(z0), static_cast<float>(I0), static_cast<float>(I1),
                            static_cast<float>(h), static_cast<float>(k));
    worst32 = std::max(worst32, std::fabs(std::log(static_cast<double>(zf)) - ref));
  }
  MESSAGE("max |d ln z| fp64 by P bin [0,.5) [.5,10) [10,1e3) [1e3,3e3]: " << worst64[0] << " " << worst64[1] << " "
                                                                           << worst64[2] << " " << worst64[3]
                                                                           << "; fp32 " << worst32);
  for (int b = 0; b < 4; ++b) {
    CHECK(count[b] > 50);
    CHECK(worst64[b] < 1e-12);
  }
  CHECK(worst32 < 3e-6);
}

TEST_CASE("the fp32 small-slope branch stays within 2 ulp of the long-double reference (ENGINE_CPP I6p)") {
  // Its domain: P <= 1/2 and P |dI| <= 2.4e-4 I_m (flat inputs included), over the kernel's ranges of k, I and h.
  // The dropped terms are O((P dI / I_m)^2) < 6e-8 relative, below fp32 rounding; GL8 in fp32 is the comparison.
  std::mt19937_64 rng(20261008);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  int worst = 0, n = 0;
  for (int trial = 0; trial < 2500; ++trial) {
    const float k = static_cast<float>(std::pow(10.0, 2.0 * u(rng)));    // 1e15..1e17 A^-1 s^-1 in (fA s)^-1
    const float I0 = static_cast<float>(std::pow(10.0, -1.0 + 7.0 * u(rng)));
    const float P_want = static_cast<float>(std::pow(10.0, -7.0 + 6.7 * u(rng)));  // up to ~0.5
    const float h = P_want / (k * I0);
    const float rel = trial % 8 == 0 ? 0.0f : static_cast<float>((2.0 * u(rng) - 1.0) * 2.3e-4 / static_cast<double>(P_want));
    const float I1 = I0 * (1.0f + std::max(-0.5f, std::min(0.5f, rel)));
    const float z0 = static_cast<float>(std::exp(-2.0 + 4.0 * u(rng))) / I0;
    const float Im = 0.5f * (I0 + I1), P = k * h * Im;
    if (!(P > 0 && P <= 0.5f && P * std::fabs(I1 - I0) <= 2.4e-4f * Im)) continue;
    const float ref = static_cast<float>(z_ref(z0, I0, I1, h, k));
    const float got = z_step(z0, I0, I1, h, k);
    const float ulp = std::nextafter(ref, INFINITY) - ref;
    worst = std::max(worst, static_cast<int>(std::lround(std::fabs(got - ref) / ulp)));
    ++n;
  }
  MESSAGE("small-slope branch: " << n << " cases, worst " << worst << " ulp");
  CHECK(n > 2000);
  CHECK(worst <= 2);
}

TEST_CASE("z closed form composes as a semigroup up to P = 1e9") {
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double worst = 0;
  for (int trial = 0; trial < 200; ++trial) {
    const double k = 10.0;
    const double I0 = std::pow(10.0, 3.0 + 4.0 * u(rng));
    const double I1 = I0 * std::exp(-3.0 + 6.0 * u(rng));
    const double h = std::pow(10.0, -4.0 + 3.0 * u(rng));
    const double z0 = std::exp(-2.0 + 4.0 * u(rng)) / I0;
    const double one = z_step(z0, I0, I1, h, k);
    double z = z0;
    const int n = 4096;
    for (int j = 0; j < n; ++j) {
      const double a = I0 + (I1 - I0) * j / n, b = I0 + (I1 - I0) * (j + 1) / n;
      z = z_step(z, a, b, h / n, k);
    }
    worst = std::max(worst, std::fabs(std::log(one) - std::log(z)));
  }
  MESSAGE("semigroup max |d ln z| = " << worst);
  CHECK(worst < 1e-12);
}

TEST_CASE("branches of the closed form join continuously") {
  const double k = 10.0, z0 = 0.37;
  // Around P = 1/2 (Gauss-Legendre <-> Dawson / erfcx) and around the flat threshold.
  for (double ratio : {0.3, 0.999, 1.0, 1.001, 3.0}) {
    const double I0 = 2.0, I1 = I0 * ratio;
    double h = 0.5 / (k * 0.5 * (I0 + I1));
    while (integrated_pole(I0, I1, h, k) > 0.5) h = std::nextafter(h, 0.0);
    const double below = z_step(z0, I0, I1, h, k);  // last h on the Gauss-Legendre branch
    const double above = z_step(z0, I0, I1, std::nextafter(std::nextafter(h, 1.0), 1.0), k);
    CHECK(std::fabs(below - above) <= 1e-14 * std::fabs(below));
  }
  // Slopes on both sides of the zero-order-hold switch, compared with the reference.
  for (double rel : {0.0, 1e-17, 3e-16, 1e-12, 1e-9}) {
    const double I0 = 5.0, I1 = 5.0 * (1 + rel), h = 1e-2;
    const double ref = static_cast<double>(std::log(z_ref(z0, I0, I1, h, k)));
    CHECK(std::fabs(std::log(z_step(z0, I0, I1, h, k)) - ref) <= 1e-14);
  }
}

TEST_CASE("accepted sub-steps honour the surrogate tolerance (M3 V1 T4)") {
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double eps_h = 2.5e-4;
  CascadeConstants<double> cc{};
  cc.k = 10.0;
  cc.a2 = 1.0 / 25e-6;
  cc.a3 = 1.0 / 100e-6;
  cc.tau_o = 25e-6;
  cc.tau_sf = 100e-6;
  cc.eps_accept = eps_h / 1.5;
  cc.eps_drop = eps_h / 1024;
  double worst = 0;
  for (int trial = 0; trial < 800; ++trial) {
    cc.k = std::pow(10.0, 0.5 + u(rng));
    const double I0 = std::pow(10.0, 5.0 * u(rng));
    const double I1 = I0 * std::exp(-2.5 + 5.0 * u(rng));
    const double z0 = trial % 3 == 0 ? 1.0 / I0 : std::exp(-2.0 + 4.0 * u(rng)) / I0;
    const double H = std::pow(10.0, -5.0 + 3.5 * u(rng));
    CascadeState<double> st{z0, -std::log(z0 * I0), -std::log(z0 * I0)};
    SubStep<double> sub{};
    CascadeState<double> next{};
    double h = H;
    int halvings = 0;  // bounded: a sub-step that is never accepted (a NaN) would halve h to 0
    while (!try_substep(cc, st, I0, I0, I0 + (I1 - I0) * h / H, h, false, sub, next)) {
      REQUIRE(++halvings < 60);
      h *= 0.5;
    }
    // Sup error of the x1 surrogate over 257 points, from the exact z at each point.
    const double Ib = I0 + (I1 - I0) * h / H;
    const double x1a = -std::log(z0 * I0), x1b = -std::log(next.z * I0);
    const double da = cc.k * (I0 - 1 / z0), db = cc.k * (Ib - 1 / next.z);
    const double delta = (x1b - x1a) / h;
    const double c2 = (3 * delta - 2 * da - db) / h, c3 = (da + db - 2 * delta) / (h * h);
    for (int i = 1; i <= 256; ++i) {
      const double t = h * i / 256;
      const double exact = -std::log(z_step(z0, I0, I0 + (Ib - I0) * t / h, t, cc.k) * I0);
      const double herm = x1a + t * (da + t * (c2 + t * c3));
      worst = std::max(worst, std::fabs(herm - exact) / eps_h);
    }
  }
  MESSAGE("max sup error / eps_h on accepted sub-steps = " << worst);
  CHECK(worst <= 1.0);
}

TEST_CASE("stage 2-3 output matches the exact response in both splits, for either pole order") {
  // A curved x1 surrogate on a short step (residual split) and a linear one on a long step (plain
  // split): compare s3 with a fine exponential-integrator reference of tau y' + y = input.
  for (int form = 0; form < 4; ++form) {
    const bool reversed = form >= 2;  // tau_o > tau_sf
    CascadeConstants<double> cc{};
    cc.k = 10.0;
    cc.tau_o = reversed ? 100e-6 : 25e-6;
    cc.tau_sf = reversed ? 25e-6 : 100e-6;
    cc.a2 = 1.0 / cc.tau_o;
    cc.a3 = 1.0 / cc.tau_sf;
    cc.eps_accept = 1.0;  // accept whatever is offered: this test checks stages 2-3 only
    cc.eps_drop = 1e-12;
    const bool curved = form % 2 == 0;
    const double I0 = 50.0, I1 = curved ? 80.0 : 50.0;
    const double h = curved ? 10e-6 : 400e-6;
    const double z0 = curved ? 1.3 / I0 : 1.0 / I0;  // a boundary layer for the curved case
    CascadeState<double> st{z0, -std::log(z0 * I0) + 0.2, -std::log(z0 * I0) - 0.1};
    SubStep<double> sub{};
    CascadeState<double> next{};
    REQUIRE(try_substep(cc, st, I0, I0, I1, h, false, sub, next));
    CHECK(sub.s3.order == (curved ? 4 : 0));
    // Reference: integrate x2, x3 with the x1 surrogate as input on 2e5 steps (second order).
    const double x1a = -std::log(z0 * I0), x1b = -std::log(next.z * I0);
    const double da = cc.k * (I0 - 1 / z0), db = cc.k * (I1 - 1 / next.z);
    const double delta = (x1b - x1a) / h;
    const double c2 = curved ? (3 * delta - 2 * da - db) / h : 0.0;
    const double c3 = curved ? (da + db - 2 * delta) / (h * h) : 0.0;
    auto x1 = [&](double t) { return x1a + t * ((curved ? da : delta) + t * (c2 + t * c3)); };
    const int n = 200000;
    double x2 = st.s2, x3 = st.s3, worst = 0;
    for (int i = 0; i < n; ++i) {
      const double t = h * i / n, dt = h / n;
      // Heun step of x2' = a2 (x1 - x2), x3' = a3 (x2 - x3).
      const double k2a = cc.a2 * (x1(t) - x2), k3a = cc.a3 * (x2 - x3);
      const double x2p = x2 + dt * k2a, x3p = x3 + dt * k3a;
      const double k2b = cc.a2 * (x1(t + dt) - x2p), k3b = cc.a3 * (x2p - x3p);
      x2 += 0.5 * dt * (k2a + k2b);
      x3 += 0.5 * dt * (k3a + k3b);
      if ((i + 1) % 2000 == 0) worst = std::max(worst, std::fabs(sub.s3((i + 1) * dt) - x3));
    }
    CHECK(worst < 1e-9);
    CHECK(std::fabs(next.s2 - x2) < 1e-9);
    CHECK(std::fabs(next.s3 - x3) < 1e-9);
  }
}
