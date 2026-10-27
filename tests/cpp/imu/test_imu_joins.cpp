// U12 (docs/design/phase3/outputs.md §4.3.4, §8.1): trajectory joins of a piecewise rig (A11). Two- and three-piece
// synthetic rigs of constant spin and velocity, whose filtered response is analytic: a C0 bounce (the specific force
// carries an impulse M J_I), a C1 join (only the cut at t_b + d) and a step join (the filter is re-initialized at
// t_b + d, IMPACT and WARMUP flags); a sample exactly at t_b uses the right piece. Also where_left, pieces_error,
// the knot cuts of a piecewise rig, join_impulse, the simulator's refusals and the chunking.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core/math/quat.h"
#include "core/math/so3.h"
#include "imu/filter.h"
#include "imu/imu_test_util.h"
#include "imu/simulator.h"
#include "imu/truth.h"

using namespace balla;
using namespace balla::imu;
using balla::imu::test::ideal;
using balla::imu::test::input;
using balla::imu::test::make_piecewise;
using balla::imu::test::PieceSpec;
using balla::imu::test::PiecewiseRig;

namespace {

constexpr int32_t kH = 5'000'000;
constexpr int64_t kTb = 100'000'000;  // the first join: a knot of both pieces around it
constexpr double kG = 9.80665;

Quat<double> rz(double theta) { return so3::exp(Vec3d{0, 0, theta}); }

double norm_inf(const Vec3d& v) { return std::max({std::abs(v.x), std::abs(v.y), std::abs(v.z)}); }

// The control rows of a piece whose pose follows theta(s) (rotation about z) and pos(s), s in seconds since kTb: the
// control c stands for the time t0 + c h - (K - 2) h / 2 (Greville), where a K-th order B-spline of controls G(xi_c)
// reproduces any polynomial G of degree < K exactly.
PieceSpec piece(int order, int64_t begin, int64_t t0, int rows, int8_t join, const std::function<double(double)>& theta,
                const std::function<Vec3d(double)>& pos, int32_t h = kH) {
  PieceSpec s{begin, t0, h, join, {}, {}};
  for (int c = 0; c < rows; ++c) {
    const double ds = (static_cast<double>(t0 - kTb) + c * static_cast<double>(h) - (order - 2) * static_cast<double>(h) / 2) * 1e-9;
    s.q.push_back(rz(theta(ds)));
    s.p.push_back(pos(ds));
  }
  return s;
}

// Constant spin w about z and constant velocity v through (theta_b, p_b) at kTb (the reference time of every piece).
PieceSpec line_piece(int order, int64_t begin, int64_t t0, int rows, int8_t join, double theta_b, const Vec3d& p_b, double w,
                     const Vec3d& v, int32_t h = kH) {
  return piece(order, begin, t0, rows, join, [=](double s) { return theta_b + w * s; },
               [=](double s) { return p_b + s * v; }, h);
}

// A static pose (every control equal: omega = a = 0 exactly).
PieceSpec still_piece(int64_t begin, int64_t t0, int rows, int8_t join, const Quat<double>& q, const Vec3d& p) {
  return PieceSpec{begin, t0, kH, join, std::vector<Quat<double>>(static_cast<size_t>(rows), q),
                   std::vector<Vec3d>(static_cast<size_t>(rows), p)};
}

// A rotated camera, a rotated IMU and a lever, as the other truth tests: the IMU axes differ from the body's.
struct Fixture {
  render::RigidPose T_rig_cam{render::operator*(Mat3d{{{std::cos(0.4), -std::sin(0.4), 0}, {std::sin(0.4), std::cos(0.4), 0}, {0, 0, 1}}},
                                                 to_matrix(so3::exp(Vec3d{0.3, -0.2, 0.1}))),
                              {0.02, -0.01, 0.05}};
  Vec3d t_ci{-1.3e-3, -0.5e-3, -40.9e-3};
  Mat3d r_ci = to_matrix(so3::exp(Vec3d{-0.1, 0.25, 2.9}));
  Mount mount;
  explicit Fixture(const ImuParams& base = {}) {
    ImuParams p = base;
    p.T_cam_imu = test::transform(r_ci, t_ci);
    mount = mount_of(T_rig_cam, p);
  }
  ImuParams with_mount(ImuParams p) const {
    p.T_cam_imu = test::transform(r_ci, t_ci);
    return p;
  }
};

// y(t) of a first-order filter (tau, delay d, ns) with unit-gain input u_j between the joins and the impulses M J_j
// at the joins: u_last + sum_j (u_{j-1} - u_j) e^{-(t - r_j)/tau} + sum_j (1/tau) e^{-(t - r_j)/tau} M J_j, r_j = t_b_j + d,
// for t >= the last r_j that has passed; before r_1 it is u_0.
Vec3d analytic(double t, double tau, double d, const std::vector<double>& t_b, const std::vector<Vec3d>& u,
               const std::vector<Vec3d>& mj /* per join; (0,0,0) when none */) {
  size_t passed = 0;  // the joins whose delayed time t_b + d the output has reached
  while (passed < t_b.size() && t >= t_b[passed] + d) ++passed;
  Vec3d y = u[passed];
  for (size_t j = 0; j < passed; ++j) {
    const double e = std::exp(-(t - (t_b[j] + d)) / tau);
    y = y + e * (u[j] - u[j + 1]) + (1e9 / tau * e) * mj[j];
  }
  return y;
}

Vec3d vec_at(const std::vector<double>& flat, size_t k) { return {flat[3 * k], flat[3 * k + 1], flat[3 * k + 2]}; }

// The truth of the piece serving t (exclusive of joins: pick a time inside the piece).
Truth truth_of(const PiecewiseRig& rig, const Mount& m, int64_t t_ns) {
  const render::TrajectoryView v = rig.view();
  return truth_at(v, m, {0, 0, -kG}, where_right(v, t_ns, 0.0));
}

}  // namespace

TEST_CASE("pieces_error: what evaluation relies on") {
  PiecewiseRig rig;
  make_piecewise(rig,
                 {line_piece(6, 0, 0, 25, 0, 0.4, {0, 0, 1}, 1.0, {1, 0, 0}),
                  line_piece(6, kTb, kTb, 25, 0, 0.4, {0, 0, 1}, -1.0, {0, 1, 0})},
                 6, 0, 2 * kTb);
  const render::TrajectoryView v = rig.view();
  CHECK(pieces_error(v, rig.joins).empty());
  CHECK(pieces_error(v, std::vector<int8_t>{0, 1}).empty());
  CHECK(pieces_error(v, std::vector<int8_t>{0, -1}).empty());
  CHECK(pieces_error(v, std::vector<int8_t>{0}) == "joins needs one entry per piece");
  CHECK(pieces_error(v, std::vector<int8_t>{1, 0}).find("piece 0: join must be -1, 0 or 1 (piece 0: 0)") == 0);
  CHECK(pieces_error(v, std::vector<int8_t>{0, 2}).find("piece 1: join must be") == 0);
  // The begins strictly increase.
  PiecewiseRig same;
  make_piecewise(same,
                 {line_piece(6, 0, 0, 25, 0, 0.4, {0, 0, 1}, 1.0, {1, 0, 0}),
                  line_piece(6, 0, 0, 25, 0, 0.4, {0, 0, 1}, -1.0, {0, 1, 0})},
                 6, 0, 2 * kTb);
  CHECK(pieces_error(same.view(), same.joins).find("piece 1: begin_ns must exceed the previous piece's") == 0);
  // Each piece's support covers its interval (the last one's up to the span's end).
  PiecewiseRig short_support;
  make_piecewise(short_support,
                 {line_piece(6, 0, 0, 10, 0, 0.4, {0, 0, 1}, 1.0, {1, 0, 0}),  // support [0, 25 ms]
                  line_piece(6, kTb, kTb, 25, 0, 0.4, {0, 0, 1}, -1.0, {0, 1, 0})},
                 6, 0, 2 * kTb);
  CHECK(pieces_error(short_support.view(), short_support.joins).find("piece 0: its spline's support does not cover [begin_ns, " +
                                                                    std::to_string(kTb) + "]") == 0);
  PiecewiseRig late_origin;  // t0 after begin
  make_piecewise(late_origin,
                 {line_piece(6, 0, 0, 25, 0, 0.4, {0, 0, 1}, 1.0, {1, 0, 0}),
                  line_piece(6, kTb, kTb + 1, 25, 0, 0.4, {0, 0, 1}, -1.0, {0, 1, 0})},
                 6, 0, 2 * kTb);
  CHECK(pieces_error(late_origin.view(), late_origin.joins).find("piece 1: its spline's support does not cover") == 0);
  // A single spline takes no joins or the one 0.
  const render::TrajectoryView one{v.pieces[0].spline, 0, 6, 0, kTb};
  CHECK(pieces_error(one, {}).empty());
  CHECK(pieces_error(one, std::vector<int8_t>{0}).empty());
  CHECK(pieces_error(one, std::vector<int8_t>{1}) == "joins without pieces");
}

TEST_CASE("where_right and where_left at a join; join_impulse against the analytic bounce") {
  for (const int K : {6, 4}) {
    INFO("K = " << K);
    const Fixture f;
    const double theta_b = 0.4, w0 = 3.0, w1 = -2.0;
    const Vec3d p_b{0.3, 0.2, 1.0}, v0{1.0, -0.5, 0.2}, v1{-0.6, 0.3, 0.9};
    PiecewiseRig rig;
    // Piece 0 reaches 15 ms past the join (support [0, 115 ms]); piece 1 starts on its knot; a third piece begins 3 ms
    // after the second (inside its first segment).
    make_piecewise(rig,
                   {line_piece(K, 0, 0, 28 + K - 6, 0, theta_b, p_b, w0, v0),
                    line_piece(K, kTb, kTb, 25, 0, theta_b, p_b, w1, v1),
                    line_piece(K, kTb + 3'000'000, kTb + 3'000'000, 25, 0, theta_b + (w1 - w0) * 3e-3, p_b + 3e-3 * (v1 - v0), w0, v0)},
                   K, 0, 2 * kTb);
    const render::TrajectoryView v = rig.view();
    // Right-continuous: the join belongs to the right piece.
    CHECK(where_right(v, kTb - 1, 0.5).piece == 0);
    Where w = where_right(v, kTb, 0.0);
    CHECK(w.piece == 1);
    CHECK(w.segment == 0);
    CHECK(w.u == 0.0);
    // The left limit: piece 0 at its last segment covering t_b, a knot ending the segment before it (u = 1).
    w = where_left(v, kTb);
    CHECK(w.piece == 0);
    CHECK(w.segment == 19);
    CHECK(w.u == 1.0);
    // The second join falls inside piece 1's first segment: u = 3 ms / 5 ms.
    w = where_left(v, kTb + 3'000'000);
    CHECK(w.piece == 1);
    CHECK(w.segment == 0);
    CHECK(w.u == 0.6);
    // Not a join: the first piece's begin, an interior time and a time one ns off.
    CHECK_THROWS_AS(where_left(v, 0), Refused);
    CHECK_THROWS_AS(where_left(v, kTb + 1), Refused);
    CHECK_THROWS_AS(where_left(v, kTb - 1), Refused);
    const render::TrajectoryView one{rig.pieces[0].spline, 0, K, 0, kTb};
    CHECK_THROWS_AS(where_left(one, kTb), Refused);

    // Both sides at the join: continuous pose, the velocity and the rate jump.
    const Truth l = truth_at(v, f.mount, {0, 0, -kG}, where_left(v, kTb));
    const Truth r = truth_at(v, f.mount, {0, 0, -kG}, where_right(v, kTb, 0.0));
    CHECK(norm_inf(l.p_W_I - r.p_W_I) < 1e-12);
    CHECK(std::abs(l.q_W_I.w - r.q_W_I.w) + std::abs(l.q_W_I.z - r.q_W_I.z) < 1e-12);
    CHECK(norm_inf(r.omega_I - l.omega_I) > 1.0);
    // J_I = R_BI^T [R_WB^T dv + dw x p_B] with R_WB = Rz(theta_b) and dw = (w1 - w0) z.
    const Vec3d dv = v1 - v0;
    const Vec3d rotated = rotate_inverse(rz(theta_b), dv);
    const Vec3d lever = cross(Vec3d{0, 0, w1 - w0}, f.mount.p_B);
    const Vec3d expect = to_imu(f.mount, rotated + lever);
    const Vec3d got = join_impulse(v, f.mount, kTb);
    CHECK(norm_inf(got - expect) < 1e-9);
    CHECK(norm_inf(got) > 1.0);
    // The second join (piece 1 to 2): the left limit at u = 0.6 of piece 1's first segment.
    const Vec3d expect2 = to_imu(f.mount, rotate_inverse(rz(theta_b + w1 * 3e-3), v0 - v1) + cross(Vec3d{0, 0, w0 - w1}, f.mount.p_B));
    CHECK(norm_inf(join_impulse(v, f.mount, kTb + 3'000'000) - expect2) < 1e-9);
    CHECK_THROWS_AS(join_impulse(v, f.mount, kTb + 1), Refused);
  }
}

TEST_CASE("knot_cuts on a piecewise rig: the join, and the knots of the piece serving each part") {
  // Piece 0: knots every 5 ms from 0, serving [0, 100 ms) (its knots at and after 100 ms are not on the path).
  // Piece 1: begins at 100 ms with knots 98 + 3 j ms (101, 104, 107, 110, ... after the join).
  PiecewiseRig rig;
  make_piecewise(rig,
                 {line_piece(6, 0, 0, 30, 0, 0.4, {0, 0, 1}, 1.0, {1, 0, 0}),
                  line_piece(6, kTb, kTb - 2'000'000, 40, 0, 0.4, {0, 0, 1}, -1.0, {0, 1, 0}, 3'000'000)},
                 6, 0, 2 * kTb);
  const render::TrajectoryView v = rig.view();
  std::vector<double> c;
  knot_cuts(v, 0, 90e6, 110e6, 250.0, c);
  const std::vector<double> expect{90e6 + 250, 95e6 + 250, 100e6 + 250, 101e6 + 250, 104e6 + 250, 107e6 + 250};
  REQUIRE(c.size() == expect.size());
  for (size_t i = 0; i < c.size(); ++i) CHECK(c[i] == expect[i]);
  // Relative to another reference time the same cuts shift.
  knot_cuts(v, 7'000'000, 90e6 - 7e6, 110e6 - 7e6, 250.0, c);
  REQUIRE(c.size() == expect.size());
  for (size_t i = 0; i < c.size(); ++i) CHECK(c[i] == expect[i] - 7e6);
  // An interval that ends exactly on the join: the join is not strictly inside; one that starts on it: neither.
  knot_cuts(v, 0, 90e6, 100e6, 0.0, c);
  REQUIRE(c.size() == 1);
  CHECK(c[0] == 95e6);
  knot_cuts(v, 0, 100e6, 106e6, 0.0, c);
  REQUIRE(c.size() == 2);
  CHECK(c[0] == 101e6);
  CHECK(c[1] == 104e6);
  // Entirely inside piece 0 near the join: only piece 0's knots; entirely inside piece 1: only piece 1's.
  knot_cuts(v, 0, 61e6, 79e6, 0.0, c);
  REQUIRE(c.size() == 3);
  CHECK(c[0] == 65e6);
  CHECK(c[2] == 75e6);
  knot_cuts(v, 0, 130e6, 140e6, 0.0, c);
  REQUIRE(c.size() == 3);
  CHECK(c[0] == 131e6);
  CHECK(c[2] == 137e6);
  // A delay larger than the interval reaching back over the join.
  knot_cuts(v, 0, 101e6, 103e6, 5e6, c);  // x in (96, 98): piece 0's knots at ... none (95, 100), the 100 ms join is
  CHECK(c.empty());                       // outside (96, 98); 100 + 5 = 105 > 103
  knot_cuts(v, 0, 104e6, 106e6, 5e6, c);  // x in (99, 101): the join at 100 -> c = 105
  REQUIRE(c.size() == 1);
  CHECK(c[0] == 105e6);
}

namespace {

// One run of a 1 kHz ideal device (both MPU-9250 filters, errors optional) from t_begin = 50 ms to 190 ms over a rig
// whose span is [0, 200 ms]: 140 samples at k ms, so the sample k = 50 sits exactly on the join at 100 ms.
struct Run {
  ImuBlock b;
  ImuStats stats;
};

constexpr int64_t kBegin = 50'000'000, kStop = 190'000'000;

Run run(const PiecewiseRig& rig, const ImuParams& p, const Fixture& f, int64_t t_begin = kBegin, int64_t t_end = kStop) {
  ImuSimulator sim(p, input(rig, t_begin, t_end, f.T_rig_cam));
  Run r;
  r.b = sim.advance_to(INT64_MAX);
  r.stats = sim.stats();
  return r;
}

double tau_ns(const SensorErrors& e) { return tau_c_s(e) * 1e9; }
double delay_ns(const SensorErrors& e) { return (e.group_delay_s - tau_c_s(e)) * 1e9; }

}  // namespace

TEST_CASE("U12: a C0 bounce adds (1/tau_c) e^{-(t - t_b - d)/tau_c} M J_I to the accelerometer, a step to the gyro") {
  for (const int K : {6, 4}) {
    INFO("K = " << K);
    const double theta_b = 0.4, w0 = 3.0, w1 = -2.0;
    const Vec3d p_b{0.3, 0.2, 1.0}, v0{1.0, -0.5, 0.2}, v1{-0.6, 0.3, 0.9};
    PiecewiseRig rig;
    make_piecewise(rig,
                   {line_piece(K, 0, 0, 28 + K - 6, 0, theta_b, p_b, w0, v0), line_piece(K, kTb, kTb, 25, 0, theta_b, p_b, w1, v1)}, K, 0,
                   2 * kTb);
    // A non-trivial error matrix (scale and non-orthogonality; no bias) on the accelerometer.
    ImuParams p = ideal(1000, true);
    p.accel.scale = {0.01, -0.02, 0.03};
    p.accel.nonorth = {0.004, -0.003, 0.002};
    const Fixture f;
    p = f.with_mount(p);
    const Mat3d M{{{1.01, 0.004, -0.003}, {0.004, 0.98, 0.002}, {-0.003, 0.002, 1.03}}};
    const Run r = run(rig, p, f);
    REQUIRE(r.b.size() == 140);

    // The analytic inputs: omega and f of each piece (z-rotation: R_WB^T g is constant), and J_I.
    const Vec3d g_up{0, 0, kG};
    const auto f_of = [&](double w) {
      return to_imu(f.mount, g_up + (w * w) * Vec3d{-f.mount.p_B.x, -f.mount.p_B.y, 0.0});
    };
    const auto om_of = [&](double w) { return to_imu(f.mount, Vec3d{0, 0, w}); };
    const Vec3d J = to_imu(f.mount, rotate_inverse(rz(theta_b), v1 - v0) + cross(Vec3d{0, 0, w1 - w0}, f.mount.p_B));
    const SensorErrors& ga = p.gyro;
    const SensorErrors& ac = p.accel;
    const double t_b = static_cast<double>(kTb - kBegin);  // 50 ms into the run
    double worst_a = 0, worst_g = 0, peak_a = 0;
    for (size_t k = 0; k < r.b.size(); ++k) {
      const double t = static_cast<double>(k) * 1e6;
      const Vec3d ya = analytic(t, tau_ns(ac), delay_ns(ac), {t_b}, {M * f_of(w0), M * f_of(w1)}, {M * J});
      const Vec3d yg = analytic(t, tau_ns(ga), delay_ns(ga), {t_b}, {om_of(w0), om_of(w1)}, {Vec3d{0, 0, 0}});
      worst_a = std::max(worst_a, norm_inf(vec_at(r.b.accel, k) - ya));
      worst_g = std::max(worst_g, norm_inf(vec_at(r.b.gyro, k) - yg));
      peak_a = std::max(peak_a, norm_inf(ya));
    }
    MESSAGE("K = " << K << ": accel max error " << worst_a << " m/s^2 (peak " << peak_a << "), gyro " << worst_g << " rad/s");
    CHECK(worst_a <= 1e-9);
    CHECK(worst_g <= 1e-9);
    CHECK(peak_a > 100.0);  // the impulse is large: a missing one cannot hide

    // Before the delayed join the output is the steady value; the impulse appears at the first sample at or after
    // t_b + d (accel: 52 ms, d = 1.272 ms), once, and decays with tau_c.
    const Vec3d before = vec_at(r.b.accel, 51);
    CHECK(norm_inf(before - M * f_of(w0)) <= 1e-9);
    const Vec3d at = vec_at(r.b.accel, 52) - M * f_of(w1);
    CHECK(norm_inf(at) > 50.0);
    // Truth: a sample exactly at t_b (k = 50) uses the right piece, the one before it the left.
    CHECK(norm_inf(vec_at(r.b.omega_true, 50) - om_of(w1)) <= 1e-12);
    CHECK(norm_inf(vec_at(r.b.omega_true, 49) - om_of(w0)) <= 1e-12);
    CHECK(norm_inf(vec_at(r.b.f_true, 50) - f_of(w1)) <= 1e-9);
    CHECK(norm_inf(vec_at(r.b.f_true, 49) - f_of(w0)) <= 1e-9);
    // IMPACT: t_k - t_b in [0, W), W = max(20 tau_c + d) = 36.767 ms (gyro): samples 50..86; no WARMUP at a C0 join.
    for (size_t k = 0; k < r.b.size(); ++k) CHECK(r.b.flags[k] == (k >= 50 && k <= 86 ? kFlagImpact : 0));
    CHECK(r.stats.impact_samples == 37);
    CHECK(r.stats.warmup_samples == 0);
    CHECK(r.stats.step_resets == 0);
  }
}

TEST_CASE("U12: a C0 impulse bypasses the pre-clip, a filter-none sensor never sees it, a gyro has none") {
  const double theta_b = 0.4, w0 = 3.0, w1 = -2.0;
  PiecewiseRig rig;
  make_piecewise(rig,
                 {line_piece(6, 0, 0, 28, 0, theta_b, {0.3, 0.2, 1.0}, w0, {1.0, -0.5, 0.2}),
                  line_piece(6, kTb, kTb, 25, 0, theta_b, {0.3, 0.2, 1.0}, w1, {-0.6, 0.3, 0.9})},
                 6, 0, 2 * kTb);
  const Fixture f;
  // Both ideal point samples: y_k = u(t_k) (filter none), the impulse does not appear; the IMPACT window is one period.
  ImuParams none = f.with_mount(ideal(1000, false));
  const Run n = run(rig, none, f);
  REQUIRE(n.b.size() == 140);
  CHECK(n.b.accel == n.b.f_true);
  CHECK(n.b.gyro == n.b.omega_true);
  for (size_t k = 0; k < n.b.size(); ++k) CHECK(n.b.flags[k] == (k == 50 ? kFlagImpact : 0));
  CHECK(n.stats.impact_samples == 1);
  // Only the accelerometer filtered: the window is its 20 tau_c + d = 33.4 ms and the gyro stays a point sample.
  ImuParams acc_only = f.with_mount(ideal(1000, false));
  acc_only.accel.filter = Filter::kFirstOrder;
  const Run a = run(rig, acc_only, f);
  CHECK(a.b.gyro == a.b.omega_true);
  CHECK(a.stats.impact_samples == 34);  // 20 * 1.6077 + 1.2723 = 33.43 ms: samples 50..83
  // clip_stage both with the accelerometer's rail well below the impulse: the pre-clip bounds the smooth part only.
  ImuParams clipped = f.with_mount(ideal(1000, true));
  clipped.clip_stage = ClipStage::kBoth;
  clipped.accel.lsb = 16.0 / 32768;  // rail 16 m/s^2
  clipped.gyro.lsb = 250.0 / 32768 * 3.14159265358979323846 / 180;
  clipped.adc_bits = 16;
  const Run c = run(rig, clipped, f);
  const Vec3d J = to_imu(f.mount, rotate_inverse(rz(theta_b), Vec3d{-1.6, 0.8, 0.7}) + cross(Vec3d{0, 0, w1 - w0}, f.mount.p_B));
  const double tau_a = tau_ns(clipped.accel), d_a = delay_ns(clipped.accel);
  // The first sample after r carries the impulse's e^{-(52 - 51.272)/1.608}/tau_c |J| on top of the (unclipped) step.
  const double impulse = 1e9 / tau_a * std::exp(-(52e6 - (50e6 + d_a)) / tau_a) * norm_inf(J);
  MESSAGE("impulse at k = 52: " << impulse << " m/s^2 vs the 16 m/s^2 rail");
  CHECK(impulse > 16.0);
  CHECK(c.b.counts_accel.size() == 3 * c.b.size());
  // The output is clamped by the ADC at +-16 m/s^2 - 1 lsb, which the rail bits report.
  bool rail = false;
  for (size_t k = 52; k < 56; ++k) rail = rail || ((c.b.flags[k] & kFlagRailAccel) != 0);
  CHECK(rail);
}

TEST_CASE("U12: a C1 join has no impulse, only the cut at t_b + d") {
  for (const int K : {6, 4}) {
    INFO("K = " << K);
    // The velocity v0 is continuous; the acceleration jumps from 0 to a1 (pose continuous). Piece 1 is the quadratic
    // P(s) = p_b + v0 s + a1 s^2 / 2, exact in K = 6 and K = 4 splines of Greville controls.
    const Vec3d p_b{0.3, 0.2, 1.0}, v0{1.0, -0.5, 0.2}, a1{12.0, -6.0, 3.0};
    PiecewiseRig rig;
    make_piecewise(rig,
                   {line_piece(K, 0, 0, 28 + K - 6, 0, 0.4, p_b, 0.0, v0),
                    piece(K, kTb, kTb, 25, 1, [](double) { return 0.4; },
                          [=](double s) { return p_b + s * v0 + (0.5 * s * s) * a1; })},
                   K, 0, 2 * kTb);
    const Fixture f;
    const ImuParams p = f.with_mount(ideal(1000, true));
    const Run r = run(rig, p, f);
    REQUIRE(r.b.size() == 140);
    const Vec3d f0 = to_imu(f.mount, Vec3d{0, 0, kG});
    const Vec3d f1 = to_imu(f.mount, rotate_inverse(rz(0.4), a1) + Vec3d{0, 0, kG});
    CHECK(norm_inf(truth_of(rig, f.mount, kTb - 7'000'000).f_I - f0) < 1e-9);
    CHECK(norm_inf(truth_of(rig, f.mount, kTb + 7'000'000).f_I - f1) < 1e-9);
    const SensorErrors& ac = p.accel;
    const double t_b = static_cast<double>(kTb - kBegin);
    double worst = 0;
    for (size_t k = 0; k < r.b.size(); ++k) {
      const Vec3d y = analytic(static_cast<double>(k) * 1e6, tau_ns(ac), delay_ns(ac), {t_b}, {f0, f1}, {Vec3d{0, 0, 0}});
      worst = std::max(worst, norm_inf(vec_at(r.b.accel, k) - y));
    }
    MESSAGE("K = " << K << ": accel max error " << worst);
    CHECK(worst <= 1e-9);  // without the cut at t_b + d the quadrature straddles the jump: ~1e-2
    CHECK(r.stats.impact_samples == 37);  // IMPACT marks every join; no filter was re-initialized
    CHECK(r.stats.warmup_samples == 0);
    CHECK(r.stats.step_resets == 0);
    for (size_t k = 0; k < r.b.size(); ++k) CHECK(r.b.flags[k] == (k >= 50 && k <= 86 ? kFlagImpact : 0));
  }
}

TEST_CASE("U12: a step join re-initializes the filters at t_b + d, with IMPACT and WARMUP flags") {
  const Quat<double> qa = so3::exp(Vec3d{0.2, -0.1, 0.3}), qb = so3::exp(Vec3d{-0.5, 0.4, 1.2});
  // Piece 1 is a different pose and spins at 2 rad/s about its body z.
  PiecewiseRig step;
  PieceSpec spin{kTb, kTb, kH, -1, {}, {}};
  for (int c = 0; c < 25; ++c) {
    spin.q.push_back(qb * rz(c * 2.0 * kH * 1e-9));  // body-frame rate 2 rad/s about z of qb
    spin.p.push_back({-4.0, 0.5, 0.25});
  }
  make_piecewise(step, {still_piece(0, 0, 28, 0, qa, {1.0, 2.0, 3.0}), spin}, 6, 0, 2 * kTb);
  const Fixture f;
  ImuParams p = f.with_mount(ideal(1000, true));
  const Run r = run(step, p, f);
  REQUIRE(r.b.size() == 140);
  const Truth t0 = truth_of(step, f.mount, kTb - 7'000'000);
  const Truth t1 = truth_of(step, f.mount, kTb + 7'000'000);
  CHECK(norm_inf(t0.omega_I) == 0.0);
  CHECK(norm_inf(t1.omega_I) > 1.0);
  CHECK(norm_inf(t1.f_I - t0.f_I) > 1.0);
  // Accelerometer: u0 until the first sample at or after t_b + d = 51.272 ms (k = 52), then exactly u1 (a held input
  // after a re-initialization is bitwise u). The gyroscope: 0 until t_b + 2.170 ms (k = 53), then omega_1.
  // The filter restarts from the right piece's value at t_b itself (u(t_b)), so compare to the truth there.
  const Truth at_join = truth_of(step, f.mount, kTb);
  for (size_t k = 0; k < r.b.size(); ++k) {
    const Vec3d a = vec_at(r.b.accel, k), g = vec_at(r.b.gyro, k);
    if (k < 52) {
      CHECK(norm_inf(a - t0.f_I) <= 1e-12);
    } else {
      // After the reset the accelerometer follows the first-order response to the (slowly changing) right piece: it
      // starts at u(t_b) and tracks u(t - d - tau_c) within the variation of the input over the delay.
      CHECK(norm_inf(a - vec_at(r.b.f_true, k)) < 0.5);
    }
    if (k < 53) {
      CHECK(norm_inf(g) == 0.0);
    } else {
      CHECK(norm_inf(g - at_join.omega_I) <= 1e-9);
    }
  }
  // The first sample after the reset has integrated 0.73 ms of an input that turns at 2 rad/s from u(t_b).
  CHECK(norm_inf(vec_at(r.b.accel, 52) - at_join.f_I) < 0.02);
  // Flags: IMPACT and WARMUP in [t_b, t_b + W), W = 36.767 ms; one reset; the others clear.
  for (size_t k = 0; k < r.b.size(); ++k) {
    CHECK(r.b.flags[k] == (k >= 50 && k <= 86 ? (kFlagImpact | kFlagWarmup) : 0));
  }
  CHECK(r.stats.impact_samples == 37);
  CHECK(r.stats.warmup_samples == 37);
  CHECK(r.stats.step_resets == 1);
  // The truth at the join sample is the right piece's.
  CHECK(norm_inf(vec_at(r.b.f_true, 50) - at_join.f_I) < 1e-12);
  CHECK(norm_inf(vec_at(r.b.f_true, 49) - t0.f_I) < 1e-12);
}

TEST_CASE("U12: a static step makes the filtered output exactly the new input after the reset") {
  const Quat<double> qa = so3::exp(Vec3d{0.2, -0.1, 0.3}), qb = so3::exp(Vec3d{-0.5, 0.4, 1.2});
  PiecewiseRig rig;
  make_piecewise(rig, {still_piece(0, 0, 28, 0, qa, {1.0, 2.0, 3.0}), still_piece(kTb, kTb, 25, -1, qb, {-4.0, 0.5, 0.25})}, 6,
                 0, 2 * kTb);
  const Fixture f;
  const Run r = run(rig, f.with_mount(ideal(1000, true)), f);
  const Truth t0 = truth_of(rig, f.mount, kTb - 7'000'000), t1 = truth_of(rig, f.mount, kTb + 7'000'000);
  REQUIRE(norm_inf(t1.f_I - t0.f_I) > 1.0);
  for (size_t k = 0; k < r.b.size(); ++k) {
    // Held inputs: bitwise the truth of the piece the delayed time is in (accel switches at k = 52).
    const Vec3d a = vec_at(r.b.accel, k);
    const Truth& want = k < 52 ? t0 : t1;
    CHECK(a.x == want.f_I.x);
    CHECK(a.y == want.f_I.y);
    CHECK(a.z == want.f_I.z);
  }
  CHECK(r.stats.step_resets == 1);
  // A reset needs a filter: with both sensors ideal point samples the IMPACT flag marks one sample and nothing resets.
  const Run n = run(rig, f.with_mount(ideal(1000, false)), f);
  CHECK(n.stats.step_resets == 0);
  CHECK(n.stats.impact_samples == 1);
  CHECK(n.stats.warmup_samples == 0);
}

TEST_CASE("U12: two joins inside one sample interval of a 100 Hz IMU") {
  // 100 Hz: the filter steps over 10 ms; the joins are 3 ms apart, so one interval holds both delayed times.
  const double w[3] = {3.0, -2.0, 1.5};
  const Vec3d v[3] = {{1.0, -0.5, 0.2}, {-0.6, 0.3, 0.9}, {0.4, 0.8, -0.3}};
  const Vec3d p_b{0.3, 0.2, 1.0};
  const int64_t t_b2 = kTb + 3'000'000;
  for (const int8_t second : {int8_t{0}, int8_t{-1}}) {
    INFO("second join " << int{second});
    PiecewiseRig rig;
    PieceSpec third = line_piece(6, t_b2, t_b2, 25, second, 0.4 + (w[1] - w[2]) * 3e-3, p_b + 3e-3 * (v[1] - v[2]), w[2], v[2]);
    if (second == -1) third = line_piece(6, t_b2, t_b2, 25, second, 1.1, {5.0, 5.0, 5.0}, w[2], v[2]);
    make_piecewise(rig,
                   {line_piece(6, 0, 0, 28, 0, 0.4, p_b, w[0], v[0]), line_piece(6, kTb, kTb, 25, 0, 0.4, p_b, w[1], v[1]), third},
                   6, 0, 2 * kTb);
    const Fixture f;
    ImuParams p = f.with_mount(ideal(100, true));
    const Run r = run(rig, p, f, 50'000'000, 190'000'000);
    REQUIRE(r.b.size() == 14);
    const Vec3d g_up{0, 0, kG};
    const SensorErrors& ac = p.accel;
    const Vec3d u0 = to_imu(f.mount, g_up + (w[0] * w[0]) * Vec3d{-f.mount.p_B.x, -f.mount.p_B.y, 0.0});
    const Vec3d u1 = to_imu(f.mount, g_up + (w[1] * w[1]) * Vec3d{-f.mount.p_B.x, -f.mount.p_B.y, 0.0});
    const Truth right = truth_of(rig, f.mount, t_b2 + 7'000'000);
    const Vec3d u2 = right.f_I;
    // The accelerometer delay 1.272 ms: r1 = 51.272, r2 = 54.272 ms, both inside the interval (50, 60] ms of sample 1.
    const double t_b1_rel = 50e6, t_b2_rel = 53e6;
    const Vec3d J1 = to_imu(f.mount, rotate_inverse(rz(0.4), v[1] - v[0]) + cross(Vec3d{0, 0, w[1] - w[0]}, f.mount.p_B));
    const Vec3d J2 = second == 0 ? to_imu(f.mount, rotate_inverse(rz(0.4 + w[1] * 3e-3), v[2] - v[1]) +
                                                       cross(Vec3d{0, 0, w[2] - w[1]}, f.mount.p_B))
                                 : Vec3d{0, 0, 0};
    for (size_t k = 0; k < r.b.size(); ++k) {
      const double t = static_cast<double>(k) * 1e7;
      Vec3d y;
      if (second == 0) {
        y = analytic(t, tau_ns(ac), delay_ns(ac), {t_b1_rel, t_b2_rel}, {u0, u1, u2}, {J1, J2});
      } else {
        // The step wipes the history: after r2 the filter holds u(t_b2) = u2 at that instant (the right piece is
        // constant in f), and the first join's impulse and step are forgotten.
        y = t >= t_b2_rel + delay_ns(ac) ? u2 : analytic(t, tau_ns(ac), delay_ns(ac), {t_b1_rel}, {u0, u1}, {J1});
      }
      CHECK(norm_inf(vec_at(r.b.accel, k) - y) <= 1e-9 * std::max(1.0, norm_inf(y)));
    }
    CHECK(r.stats.step_resets == (second == -1 ? 1 : 0));
  }
}

TEST_CASE("U12: advance_to is bitwise independent of the chunking across joins") {
  const Quat<double> qb = so3::exp(Vec3d{-0.5, 0.4, 1.2});
  PiecewiseRig rig;
  make_piecewise(rig,
                 {line_piece(6, 0, 0, 28, 0, 0.4, {0.3, 0.2, 1.0}, 3.0, {1.0, -0.5, 0.2}),
                  line_piece(6, kTb, kTb, 25, 0, 0.4, {0.3, 0.2, 1.0}, -2.0, {-0.6, 0.3, 0.9}),
                  still_piece(kTb + 40'000'000, kTb + 40'000'000, 20, -1, qb, {0.0, 0.0, 0.0})},
                 6, 0, 2 * kTb);
  const Fixture f;
  ImuParams p = f.with_mount(ideal(1026.45, true));
  p.start_phase = StartPhase::kRandom;
  p.gyro.noise_density = 1e-4;
  p.accel.noise_density = 1e-3;
  p.noise_mode = NoiseMode::kWhite;
  ImuSimulator one(p, input(rig, kBegin, kStop, f.T_rig_cam)), many(p, input(rig, kBegin, kStop, f.T_rig_cam));
  const ImuBlock all = one.advance_to(INT64_MAX);
  ImuBlock parts;
  const auto cat = [](auto& x, const auto& y) { x.insert(x.end(), y.begin(), y.end()); };
  for (int64_t t = kBegin; t <= kStop; t += 341'077) {
    const ImuBlock b = many.advance_to(t);
    cat(parts.t_true_ns, b.t_true_ns);
    cat(parts.gyro, b.gyro);
    cat(parts.accel, b.accel);
    cat(parts.flags, b.flags);
  }
  const ImuBlock rest = many.advance_to(INT64_MAX);
  cat(parts.t_true_ns, rest.t_true_ns);
  cat(parts.gyro, rest.gyro);
  cat(parts.accel, rest.accel);
  cat(parts.flags, rest.flags);
  CHECK(all.t_true_ns == parts.t_true_ns);
  CHECK(all.gyro == parts.gyro);
  CHECK(all.accel == parts.accel);
  CHECK(all.flags == parts.flags);
  CHECK(one.stats().impact_samples == many.stats().impact_samples);
  CHECK(one.stats().step_resets == many.stats().step_resets);
  CHECK(one.stats().step_resets == 1);
  CHECK(one.stats().impact_samples > 0);
}

TEST_CASE("U12: refusals of a piecewise rig") {
  PiecewiseRig rig;
  make_piecewise(rig,
                 {line_piece(6, 0, 0, 28, 0, 0.4, {0, 0, 1}, 1.0, {1, 0, 0}), line_piece(6, kTb, kTb, 25, 0, 0.4, {0, 0, 1}, -1.0, {0, 1, 0})},
                 6, 0, 2 * kTb);
  const Fixture f;
  const ImuParams p = f.with_mount(ideal(1000, true));
  const auto refusal = [&](const ImuInput& in) {
    try {
      ImuSimulator sim(p, in);
    } catch (const Refused& e) {
      return std::string(e.what());
    }
    return std::string();
  };
  CHECK(refusal(input(rig, kBegin, kStop, f.T_rig_cam)).empty());
  ImuInput no_joins = input(rig, kBegin, kStop, f.T_rig_cam);
  no_joins.joins.clear();
  CHECK(refusal(no_joins) == "rig trajectory: joins needs one entry per piece");
  ImuInput bad_join = input(rig, kBegin, kStop, f.T_rig_cam);
  bad_join.joins[1] = 2;
  CHECK(refusal(bad_join).find("piece 1: join must be -1, 0 or 1") != std::string::npos);
  // A single spline takes no joins (a stale join list is refused, the one 0 is accepted).
  ImuInput stale = input(rig, kBegin, kStop, f.T_rig_cam);
  stale.rig.num_pieces = 0;
  stale.rig.pieces = nullptr;
  CHECK(refusal(stale) == "rig trajectory: joins without pieces");
  stale.joins = {0};
  CHECK(refusal(stale).empty());
  // A join must lie inside the span.
  PiecewiseRig at_end;
  make_piecewise(at_end,
                 {line_piece(6, 0, 0, 46, 0, 0.4, {0, 0, 1}, 1.0, {1, 0, 0}), line_piece(6, 2 * kTb, 2 * kTb, 25, 0, 0.4, {0, 0, 1}, -1.0, {0, 1, 0})},
                 6, 0, 2 * kTb);
  CHECK(refusal(input(at_end, kBegin, kStop, f.T_rig_cam)).find("piece 1 begins at 200000000 ns, not inside the span") !=
        std::string::npos);
}
