// The Prophesee high-pass filter (sensor.tau_hp; hpf.h, docs/dev/SENSOR.md "High-pass filter"): its interval forms
// against quadrature, the b-mode's bound and the first crossings it allows; the leak's closed forms and the laws of
// the maps; labels; the kernel's invariances with the filter on (fast path, call splits, threads, save/load and the
// state format, the CUDA device); the b = 0 hook against the filter-free kernel; fp32 drift of the memory.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "sensor/hpf.h"
#include "sensor/sensor_model.h"

using namespace balla::sensor;

namespace {

using LD = long double;

// R_j(x) in long double (j = 0: e^{-x}).
LD resid_ld(int j, LD x) {
  if (j == 0) return std::exp(-x);
  if (x > 0.5L) {
    LD s = 1, t = 1;
    for (int i = 1; i < j; ++i) {
      t *= -x / i;
      s += t;
    }
    return std::exp(-x) - s;
  }
  LD term = 1;
  for (int i = 1; i <= j; ++i) term *= -x / i;
  LD s = 0;
  for (int m = j; m < j + 40; ++m) {
    s += term;
    term *= -x / (m + 1);
  }
  return s;
}

// Delta~ of an interval in long double: a cubic plus two modes E_i = R_order(a_i u).
struct RefFn {
  LD c[4], C[2], a[2];
  int order;
  LD operator()(LD u) const {
    LD v = c[0] + u * (c[1] + u * (c[2] + u * c[3]));
    for (int i = 0; i < 2; ++i) v += C[i] * resid_ld(order, a[i] * u);
    return v;
  }
};

// M(u) = m e^{-b u} + b int_0^u e^{-b (u - s)} d(s) ds: 8-point Gauss-Legendre on 64 panels, long double.
LD memory_ref(const RefFn& d, LD m, LD b, LD u) {
  static const LD x[4] = {0.1834346424956498049394761L, 0.5255324099163289858177390L, 0.7966664774136267395915539L,
                          0.9602898564975362316835609L};
  static const LD w[4] = {0.3626837833783619829651504L, 0.3137066458778872873379622L, 0.2223810344533744705443560L,
                          0.1012285362903762591525314L};
  // At least 64 panels, and enough that each spans at most a quarter of the fastest time constant.
  const LD fastest = std::max(b, std::max(d.a[0], d.a[1]));
  const int panels = std::max(64, static_cast<int>(std::ceil(4 * fastest * u)));
  const LD h = u / panels;
  LD sum = 0;
  for (int k = 0; k < panels; ++k) {
    const LD mid = (k + 0.5L) * h, half = 0.5L * h;
    for (int i = 0; i < 4; ++i) {
      for (const LD sgn : {-1.0L, 1.0L}) {
        const LD s = mid + sgn * half * x[i];
        sum += w[i] * half * std::exp(-b * (u - s)) * d(s);
      }
    }
  }
  return m * std::exp(-b * u) + b * sum;
}

template <class T>
ExpCubic<T> to_expcubic(const RefFn& r) {
  ExpCubic<T> g{};
  for (int i = 0; i < 4; ++i) g.c[i] = static_cast<T>(r.c[i]);
  for (int i = 0; i < 2; ++i) {
    g.C[i] = static_cast<T>(r.C[i]);
    g.a[i] = static_cast<T>(r.a[i]);
  }
  g.order = r.order;
  return g;
}

// A random interval's Delta~ of the cascade's class: the Prophesee poles, the residual split (order 4) where
// max(a) h <= 1 and the plain one beyond (substep.h), coefficients of physical size.
RefFn random_fn(std::mt19937_64& rng, LD h) {
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  RefFn r{};
  r.a[0] = 1.0L / 25e-6L;
  r.a[1] = 1.0L / 100e-6L;
  r.order = r.a[0] * h <= 1 ? 4 : 0;
  r.c[0] = 0.3L * U(rng);
  r.c[1] = 3e3L * U(rng);
  r.c[2] = 1e7L * U(rng);
  r.c[3] = 1e11L * U(rng);
  for (int i = 0; i < 2; ++i) r.C[i] = (r.order == 4 ? 20.0L : 0.4L) * U(rng);
  if (U(rng) > 0.8) r.C[1] = 0;  // a single mode, or none
  if (U(rng) > 0.9) r.C[0] = 0;
  return r;
}

LD scale_of(const RefFn& r, LD h) {
  LD s = 1 + std::fabs(r.c[0]) + std::fabs(r.c[1]) * h + std::fabs(r.c[2]) * h * h + std::fabs(r.c[3]) * h * h * h;
  for (int i = 0; i < 2; ++i) s += std::fabs(r.C[i]) * std::fabs(resid_ld(r.order, r.a[i] * h));
  return s;
}

struct Sink {
  std::vector<RawEvent> ev;
  void push(const RawEvent& e) { ev.push_back(e); }
};

Events advance_held(SensorModel& s, int64_t t_end) { return s.advance(nullptr, 0, nullptr, nullptr, 0, nullptr, t_end); }

// Every field of two event lists is equal.
void check_same(const Events& a, const Events& b) {
  REQUIRE(a.size() == b.size());
  CHECK(a.x == b.x);
  CHECK(a.y == b.y);
  CHECK(a.p == b.p);
  CHECK(a.t_cross_ns == b.t_cross_ns);
  CHECK(a.t_reported_ns == b.t_reported_ns);
  CHECK(a.label == b.label);
  CHECK(a.flags == b.flags);
  CHECK(a.ds == b.ds);
  CHECK(a.dn == b.dn);
  CHECK(a.dl == b.dl);
}

void check_same_states(const SensorModel& a, const SensorModel& b) {
  const std::vector<PixelState> sa = a.states(), sb = b.states();
  REQUIRE(sa.size() == sb.size());
  CHECK(std::memcmp(sa.data(), sb.data(), sa.size() * sizeof(PixelState)) == 0);
  const std::vector<HpfState> ha = a.hpf_states(), hb = b.hpf_states();
  REQUIRE(ha.size() == hb.size());
  CHECK(std::memcmp(ha.data(), hb.data(), ha.size() * sizeof(HpfState)) == 0);
}

// A moving grating on every other pixel over [t0, t1), ragged (counts), plus a constant second group.
struct Grating {
  std::vector<uint32_t> ids;
  std::vector<int64_t> t;
  std::vector<float> irr;
  std::vector<int32_t> counts;
  size_t m = 9;
};

Grating grating(uint32_t n_pixels, int64_t t0, int64_t t1) {
  Grating r;
  for (uint32_t px = 0; px < n_pixels; px += 2) {
    r.ids.push_back(px);
    const int32_t c = 1 + static_cast<int32_t>(px % r.m);
    r.counts.push_back(c);
    for (size_t j = 0; j < r.m; ++j) {
      r.t.push_back(j < static_cast<size_t>(c) ? t0 + (t1 - t0) * static_cast<int64_t>(j + 1) / c : 0);
      const double phase = 0.05 * px + 1e-7 * static_cast<double>(r.t.back());
      r.irr.push_back(static_cast<float>(200.0 * (1.2 + std::sin(phase))));
      r.irr.push_back(static_cast<float>(10.0 * (px % 5)));
    }
  }
  return r;
}

Events advance(SensorModel& s, const Grating& r, int64_t t_end) {
  return s.advance(r.ids.data(), r.ids.size(), r.t.data(), r.irr.data(), r.m, r.counts.data(), t_end);
}

constexpr int64_t kW23 = int64_t{1} << 23;

}  // namespace

TEST_CASE("HPF forms: G and M exact against quadrature; the b-mode within its bound; fp32 to its rounding") {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<double> U01(0.0, 1.0);
  double worst_m64 = 0, worst_g64 = 0, worst_m32 = 0, worst_excess = 0;
  int above_eps = 0, n = 0;
  const double eps_hp = 0.002 / 32;
  for (int trial = 0; trial < 10000; ++trial) {
    const LD tau = trial % 2 ? 1e-3L : 10.0L, b = 1 / tau;
    const LD h = 1e-9L * std::pow(65536.0L, static_cast<LD>(U01(rng)));  // 1 ns to 65.5 us, log-uniform
    const RefFn r = random_fn(rng, h);
    const LD m_x = 0.1L * (2 * U01(rng) - 1);
    const LD scale = scale_of(r, h);
    const detail::HpfForm<double> f = detail::hpf_form(to_expcubic<double>(r), static_cast<double>(m_x),
                                                       static_cast<double>(b));
    const detail::HpfForm<float> f32 = detail::hpf_form(to_expcubic<float>(r), static_cast<float>(m_x),
                                                        static_cast<float>(b));
    const double bound = detail::hpf_bound(f, static_cast<double>(h));
    above_eps += bound > eps_hp;
    for (const LD frac : {0.0L, 0.1L, 1.0L / 3, 0.5L, 0.77L, 1.0L}) {
      const LD u = frac * h;
      const LD M = memory_ref(r, m_x, b, u), G = r(u) - M;
      const double ud = static_cast<double>(u);
      // M exactly (fp64 to rounding, relative to the interval's scale).
      worst_m64 = std::max(worst_m64, static_cast<double>(std::fabs(detail::hpf_memory(f, ud) - M) / scale));
      worst_m32 = std::max(worst_m32, static_cast<double>(std::fabs(detail::hpf_memory(f32, static_cast<float>(u)) - M) /
                                                          scale));
      // G = g + the b-mode, exactly; g within the bound of G.
      const LD u2 = u * u;
      const LD full = static_cast<LD>(f.g(ud)) + static_cast<LD>(f.w4) * u2 * u2 *
                                                       static_cast<LD>(detail::hpf_e4(static_cast<double>(b * u)));
      worst_g64 = std::max(worst_g64, static_cast<double>(std::fabs(full - G) / scale));
      worst_excess = std::max(worst_excess,
                              static_cast<double>((std::fabs(static_cast<LD>(f.g(ud)) - G) - bound) / scale));
      ++n;
    }
  }
  MESSAGE(n << " points: |M - M_ref| <= " << worst_m64 << " (fp64), " << worst_m32 << " (fp32); |G - G_ref| <= "
            << worst_g64 << "; |g - G| - bound <= " << worst_excess << " (relative to the interval's scale); "
            << above_eps << " of 10000 intervals above eps_hp");
  CHECK(worst_m64 < 1e-12);
  CHECK(worst_g64 < 1e-12);
  CHECK(worst_excess < 1e-12);  // the dropped b-mode never exceeds its bound
  CHECK(worst_m32 < 2e-6);
  CHECK(above_eps > 0);  // some intervals need the split (1 ms at 65 us), most do not
  CHECK(above_eps < 2000);
  // The b = 0 limit: G = d - m_x, M constant, exactly.
  const RefFn r = random_fn(rng, 20e-6L);
  const detail::HpfForm<double> z = detail::hpf_form(to_expcubic<double>(r), 0.0125, 0.0);
  CHECK(z.w4 == 0.0);
  CHECK(detail::hpf_memory(z, 1.3e-5) == 0.0125);
  for (int i = 0; i < 2; ++i) CHECK(z.g.C[i] == static_cast<double>(r.C[i]));
  CHECK(detail::hpf_e4(0.0) == 1.0);
  CHECK(std::fabs(detail::hpf_e4(0.066) - 24 * static_cast<double>(resid_ld(4, 0.066L)) / std::pow(0.066, 4)) < 1e-15);
  CHECK(std::fabs(detail::hpf_e4(0.3) - 24 * static_cast<double>(resid_ld(4, 0.3L)) / std::pow(0.3, 4)) < 1e-12);
}

TEST_CASE("HPF forms on noise-off intervals: whole sub-steps up to 2^23 ns, halved to the bound as the kernel does") {
  // Without noise an interval is the sub-step (up to a 2^23 ns piece) unless the bound halves it, so b u reaches
  // past E4's polynomial (its direct form). Delta~ of physical size over the interval: each term of the cubic up to
  // 0.3 at u = h0, the modes on the plain split (a h0 > 1).
  std::mt19937_64 rng(4242);
  std::uniform_real_distribution<double> U01(0.0, 1.0), U(-1.0, 1.0);
  double worst_m64 = 0, worst_g64 = 0, worst_m32 = 0, worst_excess = 0, max_bu = 0;
  int halved = 0, direct = 0, n = 0;
  const double eps_hp = 0.002 / 32;
  for (int trial = 0; trial < 1000; ++trial) {
    const LD tau = trial % 4 == 0 ? 1e-3L : trial % 4 == 1 ? 0.05L : trial % 4 == 2 ? 0.2L : 10.0L, b = 1 / tau;
    const LD h0 = 65536e-9L * std::pow(128.0L, static_cast<LD>(U01(rng)));  // 65.5 us to 8.39 ms
    RefFn r{};
    r.a[0] = 1.0L / 25e-6L;
    r.a[1] = 1.0L / 100e-6L;
    r.order = 0;
    for (int k = 0; k < 4; ++k) r.c[k] = 0.3L * U(rng) / std::pow(h0, static_cast<LD>(k));
    for (int i = 0; i < 2; ++i) r.C[i] = 0.4L * U(rng);
    const LD m_x = 0.1L * (2 * U01(rng) - 1);
    const detail::HpfForm<double> f = detail::hpf_form(to_expcubic<double>(r), static_cast<double>(m_x),
                                                       static_cast<double>(b));
    const detail::HpfForm<float> f32 = detail::hpf_form(to_expcubic<float>(r), static_cast<float>(m_x),
                                                        static_cast<float>(b));
    LD h = h0;
    while (detail::hpf_bound(f, static_cast<double>(h)) > eps_hp) h /= 2;  // hpf_scan's halving
    halved += h < h0;
    const double bound = detail::hpf_bound(f, static_cast<double>(h));
    const LD scale = scale_of(r, h);
    for (const LD frac : {0.1L, 1.0L / 3, 0.5L, 0.77L, 1.0L}) {
      const LD u = frac * h;
      const LD M = memory_ref(r, m_x, b, u), G = r(u) - M;
      const double ud = static_cast<double>(u);
      max_bu = std::max(max_bu, static_cast<double>(b * u));
      direct += b * u > 0.1L;
      worst_m64 = std::max(worst_m64, static_cast<double>(std::fabs(detail::hpf_memory(f, ud) - M) / scale));
      worst_m32 = std::max(worst_m32, static_cast<double>(std::fabs(detail::hpf_memory(f32, static_cast<float>(u)) - M) /
                                                          scale));
      const LD u2 = u * u;
      const LD full = static_cast<LD>(f.g(ud)) + static_cast<LD>(f.w4) * u2 * u2 *
                                                       static_cast<LD>(detail::hpf_e4(static_cast<double>(b * u)));
      worst_g64 = std::max(worst_g64, static_cast<double>(std::fabs(full - G) / scale));
      worst_excess = std::max(worst_excess,
                              static_cast<double>((std::fabs(static_cast<LD>(f.g(ud)) - G) - bound) / scale));
      ++n;
    }
  }
  MESSAGE(n << " points on 1000 intervals (" << halved << " halved, " << direct << " points at b u > 0.1, max b u "
            << max_bu << "): |M - M_ref| <= " << worst_m64 << " (fp64), " << worst_m32 << " (fp32); |G - G_ref| <= "
            << worst_g64 << "; |g - G| - bound <= " << worst_excess);
  CHECK(worst_m64 < 1e-12);
  CHECK(worst_g64 < 1e-12);
  CHECK(worst_excess < 1e-12);
  CHECK(worst_m32 < 2e-6);
  CHECK(halved > 0);
  CHECK(direct > 0);  // E4's direct form is exercised
}

TEST_CASE("HPF forms: the first crossing of g lies within bound / |G'| of G's (dense scans of 300 intervals)") {
  std::mt19937_64 rng(77);
  std::uniform_real_distribution<double> U01(0.0, 1.0);
  int checked = 0;
  double worst = 0;  // |k - k_ref| less the allowance, ns
  for (int trial = 0; trial < 300; ++trial) {
    const LD tau = trial % 3 == 0 ? 1e-3L : (trial % 3 == 1 ? 0.05L : 10.0L), b = 1 / tau;
    const int64_t h_ns = 1000 + static_cast<int64_t>(64536 * U01(rng));
    const LD h = h_ns * 1e-9L;
    const RefFn r = random_fn(rng, h);
    const LD m_x = 0.05L * (2 * U01(rng) - 1);
    const detail::HpfForm<double> f = detail::hpf_form(to_expcubic<double>(r), static_cast<double>(m_x),
                                                       static_cast<double>(b));
    const double bound = detail::hpf_bound(f, static_cast<double>(h));
    // G on the integer ns by the exact recurrence of the memory between grid points (long double).
    std::vector<LD> G(static_cast<size_t>(h_ns) + 1);
    LD M = m_x;
    const LD eb = std::exp(-b * 1e-9L);
    LD d_prev = r(0);
    G[0] = d_prev - M;
    for (int64_t k = 1; k <= h_ns; ++k) {
      const LD d = r(k * 1e-9L);
      // M' = b (d - M) with d linear over 1 ns (its curvature is below 1e-12 there).
      M = M * eb + (1 - eb) * d_prev + (d - d_prev) * (1 - (1 - eb) / (b * 1e-9L));
      G[static_cast<size_t>(k)] = d - M;
      d_prev = d;
    }
    // A level G reaches at a random interior time.
    const int64_t k_star = 1 + static_cast<int64_t>((h_ns - 2) * U01(rng));
    const LD theta = G[static_cast<size_t>(k_star)];
    if (std::fabs(theta) < 0.01L) continue;
    const bool on = theta > 0;
    int64_t k_ref = -1;
    for (int64_t k = 0; k <= h_ns && k_ref < 0; ++k) {
      if (on ? G[static_cast<size_t>(k)] >= theta : G[static_cast<size_t>(k)] <= theta) k_ref = k;
    }
    REQUIRE(k_ref >= 0);
    const double th = static_cast<double>(std::fabs(theta));
    const detail::Hit hit = detail::search(f.g, 0, h_ns + 1, 0, on ? th : 1e9, on ? 1e9 : th);
    const LD slope = std::fabs(G[static_cast<size_t>(std::min(k_ref + 1, h_ns))] - G[static_cast<size_t>(std::max<int64_t>(k_ref - 1, 0))]) /
                     (2e-9L);
    if (slope * 1e-9L < 1e-3L * bound) continue;  // grazing: the allowance would span the interval
    ++checked;
    const double allowance = static_cast<double>(bound / slope * 1e9L) + 1.0;
    REQUIRE(hit.found);
    worst = std::max(worst, std::fabs(static_cast<double>(hit.t - k_ref)) - allowance);
  }
  MESSAGE(checked << " crossings; worst |k - k_ref| - (bound / |G'| + 1 ns) = " << worst << " ns");
  CHECK(checked > 150);
  CHECK(worst <= 1.0);
}

TEST_CASE("HPF leak: periods tau ln((l tau + eps_ci) / (l tau - theta)) + T_r to the ns; none below l tau = theta") {
  for (const Precision prec : {Precision::kFp64, Precision::kFp32}) {
    for (const double tau : {0.05, 0.2, 1.0}) {
      for (const bool ideal : {false, true}) {
        SensorParams p = preset("prophesee_gen3");
        p.noise = false;
        p.sigma_theta_on = p.sigma_theta_off = 0;
        p.hot_fraction = 0;
        p.leak_sigma_log10 = 0;
        p.initial_phase = InitialPhase::kZero;
        p.tau_hp = tau;
        // Without noise an interval is a whole held piece unless its b-mode bound eps_L / 32 halves it, so the closed
        // form is met to that level error over the slope: eps_L 1e-9 (fp64) and fp32's floor 1.6e-4.
        p.eps_L = prec == Precision::kFp64 ? 1e-9 : 1.6e-4;
        if (ideal) p.reset_mode = ResetMode::kIdeal;
        // Two pixels' worth of leak per row: rows 0-1 above l tau = theta (4 and 1.6 x), row 2 below (0.8 x).
        p.leak_rate_hz = 4.0 / tau;  // l tau = theta_on R_L tau = 4 theta_on
        const uint32_t W = 4, H = 1;
        SensorModel s(p, W, H, 3, {make_constant_waveform()}, 0, 1, prec);
        std::vector<float> irr0(W * H, 100.0f);
        s.init(0, irr0.data());
        const double l = p.theta_on * p.leak_rate_hz, lt = l * tau;
        const double t_c = tau * std::log(lt / (lt - p.theta_on));
        const double T = (ideal ? 0.0 : p.refractory_s) + tau * std::log((lt + (ideal ? 0.0 : p.eps_ci)) / (lt - p.theta_on));
        const Events ev = advance_held(s, static_cast<int64_t>(std::min(20 * T, 5.0) * 1e9));
        INFO("tau " << tau << " ideal " << ideal << " fp32 " << (prec == Precision::kFp32));
        int64_t last = -1;
        int count = 0;
        double worst = 0;
        for (size_t i = 0; i < ev.size(); ++i) {
          if (ev.x[i] != 0) continue;
          CHECK(ev.p[i] == 1);
          CHECK(ev.label[i] == kLabelLeak);
          const double want = last < 0 ? t_c * 1e9 : static_cast<double>(last) + (ideal ? t_c : T) * 1e9;
          // The ideal reset carries the sub-ns overshoot, the physical one restarts at the release (an integer ns).
          worst = std::max(worst, std::fabs(static_cast<double>(ev.t_cross_ns[i]) - want));
          last = ev.t_cross_ns[i];
          ++count;
        }
        // fp32: the memory's and the local time's rounding, a few float epsilons of Delta's scale over the slope at
        // the level (the filter-free kernel's own fp32 leak times err alike; DESIGN §8.3 "Precision").
        const double slope = l - p.theta_on / tau;
        const double tol = 1.5 + p.eps_L / 32 / slope * 1e9 +
                           (prec == Precision::kFp64 ? 0.0 : 8 * 1.2e-7 * (p.theta_on + l * T) / slope * 1e9);
        MESSAGE("tau_hp " << tau << " s" << (ideal ? ", ideal" : ", physical") << (prec == Precision::kFp32 ? ", fp32" : ", fp64")
                          << ": " << count << " events, worst |t - t_closed| " << worst << " ns (tolerance " << tol << ")");
        CHECK(count >= static_cast<int>(std::min(20 * T, 5.0) / T) - 1);
        CHECK(worst <= tol);
      }
    }
  }
  // l tau <= theta: Delta settles at l tau below the level, and nothing fires.
  SensorParams p = preset("prophesee_gen3");
  p.noise = false;
  p.sigma_theta_on = p.sigma_theta_off = 0;
  p.hot_fraction = 0;
  p.leak_sigma_log10 = 0;
  p.initial_phase = InitialPhase::kZero;
  p.tau_hp = 0.2;
  p.leak_rate_hz = 0.95 / p.tau_hp;  // l tau = 0.95 theta_on
  SensorModel s(p, 2, 2, 3, {make_constant_waveform()}, 0, 1);
  std::vector<float> irr0(4, 100.0f);
  s.init(0, irr0.data());
  CHECK(advance_held(s, 3'000'000'000).size() == 0);
  const std::vector<HpfState> h = s.hpf_states();
  // Delta = l tau (1 - e^{-t / tau}) after 3 s = 15 tau: m = l t - Delta.
  const double l = p.theta_on * p.leak_rate_hz;
  CHECK(std::fabs(h[0].m - (l * 3.0 - l * p.tau_hp * (1 - std::exp(-3.0 / p.tau_hp)))) < 1e-9);
  CHECK(h[0].mn == 0.0);
}

TEST_CASE("HPF maps: the phase law and leak-type hot pixels at their target rate") {
  // Leak-type hot pixels (noise off: all of them), physical and ideal resets: R T events in T, to within one.
  for (const bool ideal : {false, true}) {
    SensorParams p = preset("prophesee_gen3");
    p.noise = false;
    p.hot_fraction = 1.0;
    p.tau_hp = 0.05;
    if (ideal) p.reset_mode = ResetMode::kIdeal;
    const uint32_t W = 16, H = 8;
    SensorModel s(p, W, H, 21, {make_constant_waveform()}, 0, 2);
    std::vector<float> irr0(W * H, 300.0f);
    s.init(0, irr0.data());
    const double T = 0.5;
    const Events ev = advance_held(s, static_cast<int64_t>(T * 1e9));
    std::vector<int> count(W * H, 0);
    for (size_t i = 0; i < ev.size(); ++i) ++count[ev.y[i] * W + ev.x[i]];
    int bad = 0;
    for (uint32_t px = 0; px < W * H; ++px) {
      const double target = s.maps().hot_target_hz[px];
      REQUIRE(s.maps().pixels[px].hot == HotType::kLeak);
      bad += std::fabs(count[px] - target * T) > 1.0 + 1e-6 * target * T;
    }
    INFO("ideal " << ideal);
    CHECK(bad == 0);
  }
  // The phase law: first events of a leak-driven pixel uniform over its cycle (mean T_c / 2), none after l tau.
  SensorParams p = preset("prophesee_gen3");
  p.noise = false;
  p.hot_fraction = 0;
  p.sigma_theta_on = p.sigma_theta_off = 0;
  p.leak_sigma_log10 = 0;
  p.tau_hp = 0.2;
  p.eps_L = 1e-9;  // the b-mode's level budget eps_L / 32 below the closed form's ns (noise-off intervals are long)
  p.leak_rate_hz = 2.0 / p.tau_hp;  // l tau = 2 theta_on
  const uint32_t W = 64, H = 64;
  SensorModel s(p, W, H, 9, {make_constant_waveform()}, 0, 4);
  std::vector<float> irr0(W * H, 100.0f);
  s.init(0, irr0.data());
  const double l = p.theta_on * p.leak_rate_hz, lt = l * p.tau_hp;
  const double t_c = p.tau_hp * std::log(lt / (lt - p.theta_on));
  const Events ev = advance_held(s, static_cast<int64_t>(t_c * 1e9) + 1000);
  std::vector<int64_t> first(W * H, -1);
  for (size_t i = 0; i < ev.size(); ++i) {
    const uint32_t px = ev.y[i] * W + ev.x[i];
    if (first[px] < 0) first[px] = ev.t_cross_ns[i];
  }
  double sum = 0, sum2 = 0;
  int missing = 0;
  for (uint32_t px = 0; px < W * H; ++px) {
    const double phi0 = s.maps().pixels[px].phi0;
    CHECK(phi0 >= 0);
    CHECK(phi0 < p.theta_on);
    if (first[px] < 0) {
      ++missing;
      continue;
    }
    const double u = static_cast<double>(first[px]) * 1e-9 / t_c;
    sum += u;
    sum2 += u * u;
    // Its own closed form from phi0: tau ln((l tau - phi0) / (l tau - theta)).
    CHECK(std::fabs(static_cast<double>(first[px]) - 1e9 * p.tau_hp * std::log((lt - phi0) / (lt - p.theta_on))) <= 1.5);
  }
  const int n = W * H - missing;
  const double mean = sum / n, var = sum2 / n - mean * mean;
  MESSAGE("phase law: mean first-event phase " << mean << ", variance " << var << " (uniform: 0.5, 0.0833)");
  CHECK(missing == 0);
  CHECK(std::fabs(mean - 0.5) < 3 * std::sqrt(1.0 / 12 / n));
  CHECK(std::fabs(var - 1.0 / 12) < 0.01);
}

TEST_CASE("HPF labels: increments sum to Delta at the event; static noise never books SIGNAL") {
  for (const double tau : {0.02, 0.2}) {
    SensorParams p = preset("prophesee_gen3");
    p.hot_fraction = 0;
    p.tau_hp = tau;
    p.leak_rate_hz = 0.5;
    p.theta_on = p.theta_off = 0.12;  // the filter takes the noise's variance from ~2 var_v toward var_v: low levels
    p.sigma_theta_on = p.sigma_theta_off = 0.01;
    const uint32_t W = 16, H = 16;
    SensorModel s(p, W, H, 5, {make_constant_waveform()}, 0, 4);
    std::vector<float> irr0(W * H, 40.0f);  // noise-dominated photocurrent near the BA peak
    s.init(0, irr0.data());
    const Events ev = advance_held(s, 400'000'000);
    REQUIRE(ev.size() > 50);
    int signal = 0;
    for (size_t i = 0; i < ev.size(); ++i) {
      const PixelMap& m = s.maps().pixels[ev.y[i] * W + ev.x[i]];
      const double D = static_cast<double>(ev.ds[i]) + static_cast<double>(ev.dn[i]) + static_cast<double>(ev.dl[i]);
      // The level less the b-mode's bound (eps_L / 32) and rounding; the overshoot of a ns.
      if (ev.p[i]) CHECK(D >= m.theta_on - 1e-4);
      else CHECK(D <= -m.theta_off + 1e-4);
      CHECK(std::fabs(D) <= (ev.p[i] ? m.theta_on : m.theta_off) + 1e-3);
      CHECK(std::fabs(ev.ds[i]) < 1e-5);  // the signal's memory of a static s3 is nothing
      signal += ev.label[i] == kLabelSignal;
    }
    INFO("tau " << tau);
    CHECK(signal == 0);
  }
}

TEST_CASE("HPF on: fast path = general path, thread counts and a save/load resume are bitwise") {
  SensorParams p = preset("prophesee_gen3");
  p.hot_fraction = 0.05;
  p.tau_hp = 0.05;
  p.tau_hp_sigma_ln = 0.5;  // the per-pixel map too
  const uint32_t W = 16, H = 12;
  // The fast path against the general path, the kernel driven directly.
  {
    SensorModel s(p, W, H, 8, {make_constant_waveform()}, 0, 1);
    std::vector<float> irr0(W * H);
    for (size_t i = 0; i < irr0.size(); ++i) irr0[i] = 5.0f + 20.0f * static_cast<float>(i % 9);
    s.init(0, irr0.data());
    KernelConstants<double> slow = s.constants64();
    slow.static_fast_path = false;
    std::vector<PixelState> fs = s.states(), ss = s.states();
    std::vector<HpfState> fh = s.hpf_states(), sh = s.hpf_states();
    Sink fast_sink, slow_sink;
    KernelStats fast_stats, slow_stats;
    std::vector<int64_t> t = {kW23 + 3'000'000, kW23 + 6'000'000};
    std::vector<float> ramp = {400.0f, 900.0f};
    using K = PixelKernel<double, Sink, true, KernelStats, true, true>;
    for (int w = 0; w < 6; ++w) {
      for (uint32_t px = 0; px < W * H; ++px) {
        const bool moving = w == 1 && px % 2 == 1;
        const InputRun in = moving ? InputRun{t.data(), ramp.data(), 2} : InputRun{nullptr, nullptr, 0};
        K(s.constants64(), s.maps().pixels[px], px, fs[px], w * kW23, in, fast_sink, fast_stats, -1, &fh[px])
            .advance((w + 1) * kW23);
        K(slow, s.maps().pixels[px], px, ss[px], w * kW23, in, slow_sink, slow_stats, -1, &sh[px])
            .advance((w + 1) * kW23);
      }
    }
    REQUIRE(fast_sink.ev.size() == slow_sink.ev.size());
    CHECK(fast_sink.ev.size() > 100);
    for (size_t i = 0; i < fast_sink.ev.size(); ++i) {
      CHECK(fast_sink.ev[i].t_cross == slow_sink.ev[i].t_cross);
      CHECK(fast_sink.ev[i].ds == slow_sink.ev[i].ds);
      CHECK(fast_sink.ev[i].dn == slow_sink.ev[i].dn);
      CHECK(fast_sink.ev[i].dl == slow_sink.ev[i].dl);
    }
    CHECK(std::memcmp(fs.data(), ss.data(), fs.size() * sizeof(PixelState)) == 0);
    CHECK(std::memcmp(fh.data(), sh.data(), fh.size() * sizeof(HpfState)) == 0);
    CHECK(fast_stats.substeps == slow_stats.substeps);
  }
  // Thread counts and a save/load resume; noise on and off; fp64 and fp32.
  for (const bool noise : {true, false}) {
    for (const Precision prec : {Precision::kFp64, Precision::kFp32}) {
      SensorParams q = p;
      q.noise = noise;
      const std::vector<Waveform> groups = {make_constant_waveform(), make_fourier_waveform(120.0, {0.3}, {0.5})};
      auto model = [&](int threads) {
        auto s = std::make_unique<SensorModel>(q, W, H, 31, groups, 0, threads, prec);
        std::vector<float> irr0(W * H * 2);
        for (size_t i = 0; i < W * H; ++i) {
          irr0[2 * i] = 20.0f + 5.0f * static_cast<float>(i % 41);
          irr0[2 * i + 1] = 30.0f * static_cast<float>(i % 3);
        }
        s->init(-1234, irr0.data());
        return s;
      };
      INFO("noise " << noise << " fp32 " << (prec == Precision::kFp32));
      // Thread counts.
      const Grating r = grating(W * H, -1234, 3 * kW23);
      const auto one = model(1);
      const Events all = advance(*one, r, 3 * kW23);
      CHECK(all.size() > 100);
      const auto two = model(3);
      check_same(advance(*two, r, 3 * kW23), all);
      check_same_states(*one, *two);
      // A save/load resume after the first window (its samples carried by the first call, then held), against the
      // uninterrupted model.
      Grating early = r;
      for (size_t k = 0; k < r.ids.size(); ++k) {
        int32_t c = 0;
        while (c < r.counts[k] && r.t[k * r.m + static_cast<size_t>(c)] <= kW23) ++c;
        early.counts[k] = c;
      }
      const auto three = model(2);
      const Events e1 = advance(*three, early, kW23);
      const std::vector<uint8_t> blob = three->save_state();
      CHECK(blob.size() == three->state_bytes());
      CHECK(blob.size() == 56 + W * H * (sizeof(PixelState) + sizeof(HpfState)));
      const auto four = model(4);
      four->load_state(blob.data(), blob.size());
      const Events e2 = advance_held(*four, 2 * kW23);
      const auto five = model(1);
      check_same(advance(*five, early, kW23), e1);
      check_same(advance_held(*five, 2 * kW23), e2);
      check_same_states(*four, *five);
    }
  }
}

TEST_CASE("HPF: calls split at the slice grain are bitwise invariant (noise on and off)") {
  // As test_pixel.cpp's split test: one call against calls cut at windows and lattice cells, the sample at the cut
  // carried over. Without noise the grain is 2^23 ns, as in test_pixel.cpp (an interval is the whole sub-step).
  const uint32_t W = 12, H = 12;
  const int64_t T = 2 * kW23;
  auto f = [](int64_t t, uint32_t px) {
    return 50.0 + 400.0 * (1.0 + std::sin(2e-7 * static_cast<double>(t) * (1 + px % 3) + 0.3 * px));
  };
  for (const bool noise : {true, false}) {
    for (const Precision prec : {Precision::kFp64, Precision::kFp32}) {
      SensorParams p = preset("prophesee_gen3");
      p.noise = noise;
      p.tau_hp = 0.02;
      if (noise) p.hot_fraction = 0.05;
      std::vector<int64_t> splits = {T / 2};
      if (noise) splits = {37 * kNoiseCellNs, T / 2, T / 2 + 5 * kNoiseCellNs};
      std::vector<int64_t> times = splits;
      for (int64_t t = 1; t <= T; t += 524'289) times.push_back(t + (t * 7919) % 100'000);
      times.push_back(T);
      std::sort(times.begin(), times.end());
      times.erase(std::unique(times.begin(), times.end()), times.end());
      while (times.back() > T) times.pop_back();
      auto model = [&] {
        auto s = std::make_unique<SensorModel>(p, W, H, 11, std::vector<Waveform>{make_constant_waveform()}, 0, 3, prec);
        std::vector<float> irr0(W * H);
        for (uint32_t px = 0; px < W * H; ++px) irr0[px] = static_cast<float>(f(0, px));
        s->init(0, irr0.data());
        return s;
      };
      auto call = [&](SensorModel& s, int64_t a, int64_t b) {
        std::vector<uint32_t> ids;
        std::vector<int64_t> t;
        std::vector<float> irr;
        std::vector<int64_t> sel;
        for (int64_t ti : times)
          if (ti >= a && ti <= b && ti >= s.time_ns()) sel.push_back(ti);
        for (uint32_t px = 0; px < W * H; ++px) {
          ids.push_back(px);
          for (int64_t ti : sel) {
            t.push_back(ti);
            irr.push_back(static_cast<float>(f(ti, px)));
          }
        }
        return s.advance(ids.data(), ids.size(), t.data(), irr.data(), sel.size(), nullptr, b);
      };
      const auto single = model();
      const Events one = call(*single, 0, T);
      const auto split = model();
      Events parts;
      int64_t a = 0;
      splits.push_back(T);
      std::vector<Events> pieces;
      for (int64_t b : splits) {
        pieces.push_back(call(*split, a, b));
        a = b;
      }
      size_t total = 0;
      for (const Events& e : pieces) total += e.size();
      INFO("noise " << noise << " fp32 " << (prec == Precision::kFp32));
      CHECK(one.size() > 100);
      REQUIRE(total == one.size());
      size_t i = 0;
      for (const Events& e : pieces) {
        for (size_t j = 0; j < e.size(); ++j, ++i) {
          CHECK(e.t_cross_ns[j] == one.t_cross_ns[i]);
          CHECK(e.x[j] == one.x[i]);
          CHECK(e.y[j] == one.y[i]);
          CHECK(e.label[j] == one.label[i]);
          CHECK(e.ds[j] == one.ds[i]);
          CHECK(e.dn[j] == one.dn[i]);
          CHECK(e.dl[j] == one.dl[i]);
        }
      }
      check_same_states(*single, *split);
    }
  }
}

TEST_CASE("HPF state format: version 6 with the filter's section; version 5 loads as filter-off") {
  SensorParams p = preset("prophesee_gen3");
  SensorModel off(p, 4, 4, 1, {make_constant_waveform()}, 0);
  std::vector<float> irr0(16, 10.0f);
  off.init(0, irr0.data());
  advance_held(off, 3'000'000);
  std::vector<uint8_t> blob = off.save_state();
  CHECK(blob.size() == 56 + 16 * sizeof(PixelState));  // no filter section while off
  uint32_t version = 0;
  std::memcpy(&version, blob.data() + 4, 4);
  CHECK(version == 6);
  // A version-5 state (the same bytes but the version) loads into a filter-off model.
  const uint32_t five = 5;
  std::memcpy(blob.data() + 4, &five, 4);
  SensorModel off2(p, 4, 4, 1, {make_constant_waveform()}, 0);
  CHECK_NOTHROW(off2.load_state(blob.data(), blob.size()));
  CHECK(std::memcmp(off2.save_state().data() + 8, off.save_state().data() + 8, blob.size() - 8) == 0);
  // Not into a filter-on model (its size and fingerprint differ), nor a version 4 state.
  SensorParams q = p;
  q.tau_hp = 0.1;
  SensorModel on(q, 4, 4, 1, {make_constant_waveform()}, 0);
  CHECK_THROWS_AS(on.load_state(blob.data(), blob.size()), std::invalid_argument);
  const uint32_t four = 4;
  std::memcpy(blob.data() + 4, &four, 4);
  CHECK_THROWS_AS(off2.load_state(blob.data(), blob.size()), std::invalid_argument);
  // With the filter: the section follows the pixel states, and another tau_hp is another model.
  on.init(0, irr0.data());
  advance_held(on, 3'000'000);
  const std::vector<uint8_t> with = on.save_state();
  CHECK(with.size() == 56 + 16 * (sizeof(PixelState) + sizeof(HpfState)));
  SensorModel on2(q, 4, 4, 1, {make_constant_waveform()}, 0);
  CHECK_NOTHROW(on2.load_state(with.data(), with.size()));
  CHECK(on2.save_state() == with);
  q.tau_hp = 0.2;
  SensorModel other(q, 4, 4, 1, {make_constant_waveform()}, 0);
  CHECK_THROWS_AS(other.load_state(with.data(), with.size()), std::invalid_argument);
  CHECK_THROWS_AS(off2.load_state(with.data(), with.size()), std::invalid_argument);
  // Off, the kernels and the states are those without the filter: PixelState keeps 200 bytes, no side states.
  CHECK(sizeof(PixelState) == 200);
  CHECK(off.hpf_states().empty());
  CHECK_FALSE(off.hpf_kernel());
  CHECK(on.hpf_kernel());
}

TEST_CASE("HPF at b = 0 (the test hook) reproduces the filter-free kernel: counts equal, |dt| <= 1 ns") {
  // fp64: the hook against the filter-free kernel. fp32: the filter's path re-expands Delta from its anchors (shifts
  // from the sub-step's start, the level-0 Hermite added and taken out at walk nodes), so its float rounding differs
  // from the filter-free kernel's; both against fp64, the hook may err no more than the filter-free fp32 kernel does.
  for (const bool noise : {true, false}) {
    SensorParams p = preset("prophesee_gen3");
    p.noise = noise;
    p.hot_fraction = 0.05;
    const uint32_t W = 24, H = 16;
    const std::vector<Waveform> groups = {make_constant_waveform(), make_fourier_waveform(120.0, {0.3}, {0.5})};
    auto run = [&](bool hook, Precision prec) {
      SensorModel s(p, W, H, 41, groups, 0, 4, prec);
      if (hook) s.force_hpf_kernel();
      CHECK(s.hpf_kernel() == hook);
      std::vector<float> irr0(W * H * 2);
      for (size_t i = 0; i < W * H; ++i) {
        irr0[2 * i] = 20.0f + 5.0f * static_cast<float>(i % 41);
        irr0[2 * i + 1] = 30.0f * static_cast<float>(i % 3);
      }
      s.init(0, irr0.data());
      const Grating r = grating(W * H, 0, 3 * kW23);
      Events ev = advance(s, r, 3 * kW23);
      if (hook) {
        for (const HpfState& h : s.hpf_states()) CHECK(h.m == 0.0);
      }
      return ev;
    };
    const Events off64 = run(false, Precision::kFp64), hook64 = run(true, Precision::kFp64);
    const Events off32 = run(false, Precision::kFp32), hook32 = run(true, Precision::kFp32);
    INFO("noise " << noise);
    CHECK(off64.size() > 100);
    REQUIRE(hook64.size() == off64.size());
    REQUIRE(off32.size() == off64.size());
    REQUIRE(hook32.size() == off64.size());
    int64_t worst64 = 0, worst_hook32 = 0, worst_off32 = 0;
    size_t beyond_hook32 = 0, beyond_off32 = 0;
    for (size_t i = 0; i < off64.size(); ++i) {
      CHECK(hook64.x[i] == off64.x[i]);
      CHECK(hook64.p[i] == off64.p[i]);
      worst64 = std::max<int64_t>(worst64, std::llabs(hook64.t_cross_ns[i] - off64.t_cross_ns[i]));
      const int64_t dh = std::llabs(hook32.t_cross_ns[i] - off64.t_cross_ns[i]);
      const int64_t d0 = std::llabs(off32.t_cross_ns[i] - off64.t_cross_ns[i]);
      worst_hook32 = std::max(worst_hook32, dh);
      worst_off32 = std::max(worst_off32, d0);
      beyond_hook32 += dh > 2;
      beyond_off32 += d0 > 2;
    }
    MESSAGE("b = 0 hook, noise " << noise << ": " << off64.size() << " events; fp64 max |dt| " << worst64
                                 << " ns; against fp64, fp32 hook max " << worst_hook32 << " ns (" << beyond_hook32
                                 << " beyond 2 ns), fp32 filter-free max " << worst_off32 << " ns (" << beyond_off32
                                 << ")");
    CHECK(worst64 <= 1);
    CHECK(worst_hook32 <= 2 * worst_off32 + 2);
    CHECK(beyond_hook32 <= 2 * beyond_off32 + 5);
  }
}

TEST_CASE("HPF fp32 memory: drift against fp64 over 10 tau_hp below eps_L / 8") {
  // A slow 3 Hz flicker under leak, no events (levels out of reach), noise off: the memory accumulates over 10 tau.
  for (const double tau : {0.05, 0.2}) {
    SensorParams p = preset("prophesee_gen3");
    p.noise = false;
    p.hot_fraction = 0;
    p.theta_on = p.theta_off = 5.0;
    p.tau_hp = tau;
    p.leak_rate_hz = 0.05;
    const uint32_t W = 8, H = 8;
    const std::vector<Waveform> groups = {make_fourier_waveform(3.0, {0.8}, {0.0})};
    double worst = 0;
    std::vector<HpfState> h[2];
    for (const Precision prec : {Precision::kFp64, Precision::kFp32}) {
      SensorModel s(p, W, H, 2, groups, 0, 2, prec);
      std::vector<float> irr0(W * H);
      for (size_t i = 0; i < irr0.size(); ++i) irr0[i] = 10.0f + 30.0f * static_cast<float>(i);
      s.init(0, irr0.data());
      CHECK(advance_held(s, static_cast<int64_t>(10 * tau * 1e9)).size() == 0);
      h[prec == Precision::kFp32] = s.hpf_states();
    }
    for (size_t i = 0; i < h[0].size(); ++i) worst = std::max(worst, std::fabs(h[0][i].m - h[1][i].m));
    MESSAGE("tau " << tau << ": max |m_fp32 - m_fp64| after 10 tau = " << worst);
    CHECK(worst <= 0.002 / 8);
  }
}

TEST_CASE("HPF: the stationary initial state with the filter, thread-invariant; its history follows tau_hp") {
  SensorParams p = preset("prophesee_gen3");
  p.initial_state = InitialState::kStationary;
  p.stationary_probe_pixels = 8;
  p.tau_hp = 0.05;
  CHECK(probe_history_ns(p) == (int64_t{1'000'000'000} / kNoiseCellNs) * kNoiseCellNs);  // max(1 s, 10 tau)
  p.tau_hp = 0.5;
  CHECK(probe_history_ns(p) == (int64_t{5'000'000'000} / kNoiseCellNs) * kNoiseCellNs);
  p.tau_hp = 0.05;
  const uint32_t W = 40, H = 32;
  std::vector<float> irr0(W * H);
  for (size_t i = 0; i < irr0.size(); ++i) irr0[i] = 5.0f + 0.05f * static_cast<float>(i);
  const int64_t t0 = 1000 * kW23;
  auto run = [&](int threads) {
    SensorModel s(p, W, H, 31, {make_fourier_waveform(120, {0.02}, {0})}, 0, threads, Precision::kFp32);
    s.init(t0, irr0.data());
    REQUIRE(s.stationary_probes() == 8);
    std::vector<uint8_t> at_init = s.save_state();
    // No pixel on or past a level at t0: Delta(t0) = dr_ref for every non-blind pixel (its reference is at t0).
    for (const PixelState& x : s.states()) {
      if (x.blind_until != kNotBlind || x.t_ref != t0) continue;
      CHECK(x.dr_ref < p.theta_on + 0.2);
    }
    Events ev = advance_held(s, t0 + 5'000'000);
    return std::make_tuple(at_init, ev, s.save_state());
  };
  const auto [init1, ev1, end1] = run(1);
  const auto [init4, ev4, end4] = run(4);
  CHECK(init4 == init1);
  check_same(ev4, ev1);
  CHECK(end4 == end1);
}

TEST_SUITE_BEGIN("cuda");

TEST_CASE("HPF on the CUDA device: bitwise the CPU fp32 kernel under 3-cell slices, batches and overflows") {
  for (const bool noise : {true, false}) {
    for (const bool spread : {false, true}) {
      SensorParams p = preset("prophesee_gen3");
      p.hot_fraction = 0.02;
      p.noise = noise;
      p.tau_hp = 0.05;
      if (spread) p.tau_hp_sigma_ln = 0.7;
      const uint32_t W = 48, H = 40;
      const std::vector<Waveform> groups = {make_constant_waveform(), make_fourier_waveform(120.0, {0.3}, {0.5})};
      SensorModel cpu(p, W, H, 32, groups, 0, 4, Precision::kFp32, Device::kCpu);
      SensorModel gpu(p, W, H, 32, groups, 0, 1, Precision::kFp32, Device::kCuda);
      GpuLaunchPolicy policy;
      policy.max_slice_ns = noise ? 3 * kNoiseCellNs : 1;
      policy.max_pixels = 300;
      policy.event_capacity = 16;
      gpu.set_gpu_launch_policy(policy);
      std::vector<float> irr0(W * H * 2);
      for (size_t i = 0; i < W * H; ++i) {
        irr0[2 * i] = 20.0f + 5.0f * static_cast<float>(i % 41);
        irr0[2 * i + 1] = 30.0f * static_cast<float>(i % 3);
      }
      cpu.init(1234, irr0.data());
      gpu.init(1234, irr0.data());
      const Grating r = grating(W * H, 1234, 3 * kW23);
      const Events a = advance(cpu, r, 3 * kW23);
      INFO("noise " << noise << " spread " << spread);
      CHECK(a.size() > 300);
      check_same(a, advance(gpu, r, 3 * kW23));
      check_same_states(cpu, gpu);
      CHECK(gpu.gpu_launch_stats().reruns > 0);
      // A state saved on the device resumes on the CPU bit for bit, and the reverse.
      const std::vector<uint8_t> blob = gpu.save_state();
      CHECK(blob == cpu.save_state());
      SensorModel cpu2(p, W, H, 32, groups, 0, 2, Precision::kFp32, Device::kCpu);
      SensorModel gpu2(p, W, H, 32, groups, 0, 1, Precision::kFp32, Device::kCuda);
      cpu2.load_state(blob.data(), blob.size());
      gpu2.load_state(blob.data(), blob.size());
      check_same(advance_held(cpu2, 4 * kW23), advance_held(gpu2, 4 * kW23));
      check_same_states(cpu2, gpu2);
    }
  }
  // The b = 0 hook on the device equals the CPU's.
  SensorParams p = preset("prophesee_gen3");
  const uint32_t W = 32, H = 24;
  SensorModel cpu(p, W, H, 5, {make_constant_waveform()}, 0, 2, Precision::kFp32, Device::kCpu);
  SensorModel gpu(p, W, H, 5, {make_constant_waveform()}, 0, 1, Precision::kFp32, Device::kCuda);
  cpu.force_hpf_kernel();
  gpu.force_hpf_kernel();
  std::vector<float> irr0(W * H, 40.0f);
  cpu.init(0, irr0.data());
  gpu.init(0, irr0.data());
  check_same(advance_held(cpu, 2 * kW23), advance_held(gpu, 2 * kW23));
  check_same_states(cpu, gpu);
}

TEST_SUITE_END();
