// U4 (docs/design/phase3/outputs.md §8.1) and the simulator of an ideal device: the panel rule, a constant input exact
// bitwise, the ramp response, the panel rule at 50 Hz against a 64-panel reference (the C7 low-rate case, which
// ImuSimulator refuses because 92 and 99 Hz are above that rate: §6), the WARMUP flag, chunking and refusals. The
// device model (A2) is in test_imu_device.cpp and test_imu_processes.cpp.
// U4(iii) on the pinned C7 splines through ballasim._imu is in tests/python/test_imu.py.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "imu/filter.h"
#include "imu/imu_test_util.h"
#include "imu/simulator.h"

using namespace balla;
using namespace balla::imu;
using balla::imu::test::ideal;
using balla::imu::test::input;
using balla::imu::test::make_rig;
using balla::imu::test::random_rig;
using balla::imu::test::Rig;

namespace {

std::string refusal(const ImuParams& p, const ImuInput& in) {
  try {
    ImuSimulator sim(p, in);
  } catch (const Refused& e) {
    return e.what();
  }
  return "";
}

bool same(const ImuBlock& a, const ImuBlock& b) {
  return a.t_true_ns == b.t_true_ns && a.gyro == b.gyro && a.accel == b.accel && a.flags == b.flags &&
         a.omega_true == b.omega_true && a.f_true == b.f_true && a.q_W_I == b.q_W_I && a.p_W_I == b.p_W_I &&
         a.bias_gyro == b.bias_gyro && a.bias_accel == b.bias_accel;
}

void append(ImuBlock& to, const ImuBlock& b) {
  const auto cat = [](auto& x, const auto& y) { x.insert(x.end(), y.begin(), y.end()); };
  cat(to.t_true_ns, b.t_true_ns);
  cat(to.gyro, b.gyro);
  cat(to.accel, b.accel);
  cat(to.flags, b.flags);
  cat(to.omega_true, b.omega_true);
  cat(to.f_true, b.f_true);
  cat(to.bias_gyro, b.bias_gyro);
  cat(to.bias_accel, b.bias_accel);
  cat(to.q_W_I, b.q_W_I);
  cat(to.p_W_I, b.p_W_I);
}

}  // namespace

TEST_CASE("panel_nodes: GL-4 panels of at most tau_c / 2, split at the cuts") {
  Panels p;
  panel_nodes(-3.0, 2.0, 2.0, {}, p);  // 5 / 1 = 5 panels
  CHECK(p.s.size() == 20);
  double sum = 0;
  for (size_t i = 0; i < p.s.size(); ++i) {
    CHECK(p.s[i] > -5.0);
    CHECK(p.s[i] < 0.0);
    sum += p.w[i];
  }
  CHECK(sum == doctest::Approx(5.0).epsilon(1e-15));
  // A cut at 0.5 (relative -1.5 to b): pieces of 3.5 and 1.5, 4 and 2 panels; no panel straddles the cut.
  const double cuts[] = {0.5};
  panel_nodes(-3.0, 2.0, 2.0, cuts, p);
  CHECK(p.s.size() == 24);
  double left = 0, right = 0;
  for (size_t i = 0; i < p.s.size(); ++i) (p.s[i] < -1.5 ? left : right) += p.w[i];
  CHECK(left == doctest::Approx(3.5).epsilon(1e-15));
  CHECK(right == doctest::Approx(1.5).epsilon(1e-15));
  // GL-4 is exact for cubics: the weights integrate s^3 over [-5, 0].
  double cubic = 0;
  for (size_t i = 0; i < p.s.size(); ++i) cubic += p.w[i] * p.s[i] * p.s[i] * p.s[i];
  CHECK(cubic == doctest::Approx(-625.0 / 4).epsilon(1e-14));
}

TEST_CASE("knot_cuts: the knots shifted by d strictly inside the interval") {
  Rig rig = random_rig(1, 20, 0.01, 1e-4, 6, 1'000'000);
  rig.t0_ns = 500;
  std::vector<double> c;
  // Relative to t_ref = 1500: knots at -1000 + j 1e6; shifted by d = 250.
  knot_cuts(rig.view(), 1500, 0.0, 3'000'000.0, 250.0, c);
  REQUIRE(c.size() == 3);
  CHECK(c[0] == 1'000'000 - 1000 + 250);
  CHECK(c[2] == 3'000'000 - 1000 + 250);
  knot_cuts(rig.view(), 1500, 999'250.0, 1'999'250.0, 250.0, c);  // both ends on knots: none strictly inside
  CHECK(c.empty());
}

TEST_CASE("U4(i): a constant input gives y_k == u bitwise, warm-up included") {
  const Quat<double> q0 = so3::exp(Vec3d{0.2, 0.9, -0.4});
  const Rig rig = make_rig(std::vector<Quat<double>>(40, q0), std::vector<Vec3d>(40, Vec3d{1, 2, 3}), 6, 5'000'000);
  ImuParams p = ideal(1000, true);
  p.T_cam_imu = test::transform(to_matrix(so3::exp(Vec3d{0.1, -2.0, 0.3})), {0.01, 0.02, -0.04});
  for (int64_t t_begin : {int64_t{60'000'000}, int64_t{0}}) {  // settled, then a rig that starts at t_begin
    INFO("t_begin " << t_begin);
    ImuSimulator sim(p, input(rig, t_begin, 175'000'000));
    const ImuBlock b = sim.advance_to(INT64_MAX);
    REQUIRE(b.size() == static_cast<size_t>((175'000'000 - t_begin) / 1'000'000));
    for (size_t i = 0; i < 3 * b.size(); ++i) {
      CHECK(b.gyro[i] == 0.0);
      CHECK(b.omega_true[i] == 0.0);
      CHECK(b.accel[i] == b.f_true[i]);
      CHECK(b.accel[i] == b.f_true[i % 3]);
    }
    // A rig that starts at t_begin: samples before span_begin + d + 20 tau_c carry WARMUP (gyro: 2.170 + 34.598 ms).
    const int64_t warm = t_begin == 0 ? 37 : 0;
    CHECK(sim.stats().warmup_samples == warm);
    for (size_t k = 0; k < b.size(); ++k) CHECK((b.flags[k] == kFlagWarmup) == (static_cast<int64_t>(k) < warm));
  }
}

TEST_CASE("U4(ii): a ramp a + b t is filtered to a + b (t - d - tau_c) after 20 tau_c") {
  // The filter core on an analytic input, in ns, at the gyro's MPU-9250 setting; with a cut at every 5 ms knot.
  ImuParams prm;
  const double tau = tau_c_s(prm.gyro) * 1e9, d = (prm.gyro.group_delay_s - tau_c_s(prm.gyro)) * 1e9;
  const Vec3d a{1.0, -2.0, 0.5}, slope{0.5e-9, -3e-9, 1e-9};  // per ns; no component crosses 0
  const auto u = [&](double t) { return a + t * slope; };
  for (double period : {1e6, 1e7, 2e7}) {  // 1 kHz, 100 Hz, 50 Hz
    INFO("period " << period);
    Panels pn;
    std::vector<double> cuts;
    std::vector<Vec3d> us;
    double prev = -20 * tau;
    Vec3d y = u(prev - d);
    double worst = 0;
    for (int k = 0; k <= 400; ++k) {
      const double t = k * period;
      cuts.clear();
      for (double c = std::ceil((prev - d) / 5e6) * 5e6 + d; c < t; c += 5e6) {
        if (c > prev) cuts.push_back(c);
      }
      panel_nodes(prev, t, tau, cuts, pn);
      us.resize(pn.s.size());
      for (size_t i = 0; i < us.size(); ++i) us[i] = u((t + pn.s[i]) - d);
      y = filter_step(y, pn, us, tau, t - prev);
      prev = t;
      const Vec3d expect = u(t - d - tau);
      worst = std::max({worst, std::abs(y.x - expect.x) / std::abs(expect.x), std::abs(y.y - expect.y) / std::abs(expect.y),
                        std::abs(y.z - expect.z) / std::abs(expect.z)});
    }
    MESSAGE("period " << period << " ns: worst relative error " << worst);
    CHECK(worst <= 1e-10);
  }
}

TEST_CASE("U4: the panel rule at 50 Hz against a 64-panel reference, on a C7-like spline") {
  // ImuSimulator refuses 50 Hz with the MPU-9250 cutoffs (cutoff >= rate, §6), so the 50 Hz case of R-DOC C7 runs
  // through the filter core here, on a K = 6 spline with C7's EVIMO2-like scales (h = 5 ms), both sensors.
  const Rig rig = random_rig(7, 400, 0.02, 2.5e-4, 6, 5'000'000);
  const render::TrajectoryView v = rig.view();
  ImuParams prm;
  prm.T_cam_imu = test::transform({{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {-1.3e-3, -0.5e-3, -40.9e-3});
  const Mount m = mount_of({{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {0, 0, 0}}, prm);
  const Vec3d g{0, 0, -9.80665};
  const int64_t t_b = 100'000'000;
  const auto x = [&](int s, double rel) {
    const double n = std::floor(rel);
    const Truth tr = truth_at(v, m, g, where_right(v, t_b + static_cast<int64_t>(n), rel - n));
    return s == 0 ? tr.omega_I : tr.f_I;
  };
  const double lsb[2] = {250.0 / 32768 * 3.14159265358979323846 / 180, 9.81 / 16384};
  for (int s = 0; s < 2; ++s) {
    const SensorErrors& e = s == 0 ? prm.gyro : prm.accel;
    const double tau = tau_c_s(e) * 1e9, d = (e.group_delay_s - tau_c_s(e)) * 1e9, period = 2e7;
    Panels pn, ref;
    std::vector<double> cuts;
    std::vector<Vec3d> us;
    double prev = -20 * tau;
    Vec3d y = x(s, prev - d), y_ref = y;
    double worst = 0;
    for (int k = 0; k < 90; ++k) {
      const double t = k * period;
      knot_cuts(v, t_b, prev, t, d, cuts);
      panel_nodes(prev, t, tau, cuts, pn);
      us.resize(pn.s.size());
      for (size_t i = 0; i < us.size(); ++i) us[i] = x(s, (t + pn.s[i]) - d);
      y = filter_step(y, pn, us, tau, t - prev);
      // Reference: 64 GL-4 panels per knot piece, the plain kernel sum and recursion y = phi y + I.
      ref.s.clear();
      ref.w.clear();
      double lo = prev;
      for (size_t c = 0; c <= cuts.size(); ++c) {
        const double hi = c < cuts.size() ? cuts[c] : t;
        Panels piece;
        panel_nodes(lo, hi, (hi - lo) / 32, {}, piece);  // (hi - lo) / (tau / 2) = 64 panels
        for (size_t i = 0; i < piece.s.size(); ++i) {
          ref.s.push_back(piece.s[i] + (hi - t));
          ref.w.push_back(piece.w[i]);
        }
        lo = hi;
      }
      Vec3d integral{0, 0, 0};
      for (size_t i = 0; i < ref.s.size(); ++i) {
        integral += (ref.w[i] * std::exp(ref.s[i] / tau) / tau) * x(s, (t + ref.s[i]) - d);
      }
      y_ref = std::exp(-(t - prev) / tau) * y_ref + integral;
      prev = t;
      const Vec3d e3 = y - y_ref;
      worst = std::max({worst, std::abs(e3.x), std::abs(e3.y), std::abs(e3.z)});
    }
    MESSAGE(std::string(s == 0 ? "gyro" : "accel") << " at 50 Hz: max |y - y_ref| = " << worst / lsb[s] << " LSB");
    CHECK(worst / lsb[s] <= 0.2);
  }
}

TEST_CASE("ImuSimulator: advance_to is bitwise independent of the chunking") {
  const Rig rig = random_rig(5, 200, 0.02, 2.5e-4, 6, 5'000'000);
  ImuParams p = ideal(1026.45, true);
  p.T_cam_imu = test::transform({{{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}}}, {-1.3e-3, -0.5e-3, -40.9e-3});
  const ImuInput in = input(rig, 60'000'000, 400'000'000);
  ImuSimulator one(p, in), many(p, in);
  const ImuBlock all = one.advance_to(INT64_MAX);
  CHECK(all.size() == 349);  // ceil(0.34 s * 1026.45 Hz)
  ImuBlock parts;
  for (int64_t t = 60'000'000; t <= 400'000'000; t += 341'077) append(parts, many.advance_to(t));
  append(parts, many.advance_to(INT64_MAX));
  CHECK(same(all, parts));
  CHECK(many.advance_to(0).size() == 0);
  // filter none: an ideal point sample of the truth, no delay.
  ImuSimulator point(ideal(1000, false), in);
  const ImuBlock b = point.advance_to(INT64_MAX);
  CHECK(b.gyro == b.omega_true);
  CHECK(b.accel == b.f_true);
  CHECK(b.t_true_ns.front() == 60'000'000);
  CHECK(b.t_true_ns.back() == 399'000'000);
}

TEST_CASE("ImuSimulator: refusals") {
  const Rig rig = random_rig(5, 40, 0.02, 2.5e-4, 6, 5'000'000);  // support [0, 175 ms]
  const ImuInput in = input(rig, 60'000'000, 170'000'000);
  // The defaults (start_phase random) and the device errors are simulated (A2); validate()'s refusals stand.
  ImuParams p = ideal(1000, true);
  p.start_phase = StartPhase::kRandom;
  p.gyro.noise_density = 1e-4;
  p.accel.scale = {0, 1e-3, 0};
  p.accel.lsb = 5.987548828125e-4;
  CHECK(refusal(p, in).empty());
  p = ideal(90, true);  // validate(): cutoff >= rate
  CHECK(refusal(p, in).find("imu.gyro.cutoff_hz = 92 is at or above imu.rate_hz = 90") != std::string::npos);
  CHECK(refusal(ideal(1000, true), in).empty());
  // The run span must lie in the rig span; the filters need their inputs from t_0 - d on.
  CHECK(refusal(ideal(1000, true), input(rig, 60'000'000, 180'000'000)).find("is not inside") != std::string::npos);
  CHECK(refusal(ideal(1000, true), input(rig, 70'000'000, 60'000'000)).find("t_begin_ns") != std::string::npos);
  Rig late = rig;
  late.span_begin_ns = 59'000'000;
  late.span_end_ns = 170'000'000;
  CHECK(refusal(ideal(1000, true), input(late, 59'000'000, 170'000'000)).empty());  // WARMUP, not refused
  CHECK(refusal(ideal(1000, true), input(late, 58'999'999, 170'000'000)).find("is not inside") != std::string::npos);
  ImuInput nan_g = in;
  nan_g.g_W.z = std::nan("");
  CHECK(refusal(ideal(1000, true), nan_g) == "g_W must be finite");
  Rig wide = rig;  // a run span longer than INT64_MAX ns
  wide.span_begin_ns = INT64_MIN;
  wide.span_end_ns = INT64_MAX;
  CHECK(refusal(ideal(1000, false), input(wide, -3, INT64_MAX - 3)).empty());  // t_end - t_begin = INT64_MAX
  CHECK(refusal(ideal(1000, false), input(wide, -4, INT64_MAX - 3)) == "run span: t_end_ns - t_begin_ns overflows int64");
}

TEST_CASE("ImuSimulator: a rate near 0 (periods beyond INT64_MAX ns) gives the sample at t_begin, then none") {
  for (const int64_t t0 : {int64_t{0}, int64_t{-1'000'000'000}}) {
    Rig rig = random_rig(5, 40, 0.02, 2.5e-4, 6, 5'000'000);  // support [t0, t0 + 175 ms]
    rig.t0_ns = t0;
    ImuSimulator sim(ideal(1e-10, false), input(rig, t0 + 60'000'000, t0 + 170'000'000));
    const ImuBlock b = sim.advance_to(INT64_MAX);
    REQUIRE(b.size() == 1);
    CHECK(b.t_true_ns[0] == t0 + 60'000'000);
    CHECK(sim.advance_to(INT64_MAX).size() == 0);
  }
}
