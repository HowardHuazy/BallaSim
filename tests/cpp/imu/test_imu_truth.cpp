// U1, U2b, U3 (docs/design/phase3/outputs.md §8.1): the IMU truth of §4.1 on a constant-rate spin, against finite
// differences of the IMU pose, and inside a hold; the mount and where_right.
#include <doctest/doctest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "core/math/quat.h"
#include "core/math/so3.h"
#include "imu/truth.h"
#include "imu/imu_test_util.h"

using namespace balla;
using namespace balla::imu;
using balla::imu::test::make_rig;
using balla::imu::test::random_rig;
using balla::imu::test::Rig;

namespace {

double norm_inf(const Vec3d& v) { return std::max({std::abs(v.x), std::abs(v.y), std::abs(v.z)}); }

Mat3d rz(double a) { return {{{std::cos(a), -std::sin(a), 0}, {std::sin(a), std::cos(a), 0}, {0, 0, 1}}}; }

Mat3d mul(const Mat3d& a, const Mat3d& b) { return render::operator*(a, b); }

// A mount with a rotated camera, a rotated IMU and a 4 cm lever.
struct Fixture {
  render::RigidPose T_rig_cam{mul(rz(0.4), to_matrix(so3::exp(Vec3d{0.3, -0.2, 0.1}))), {0.02, -0.01, 0.05}};
  ImuParams params = [] {
    ImuParams p;
    p.T_cam_imu = imu::test::transform(to_matrix(so3::exp(Vec3d{-0.1, 0.25, 2.9})), {-1.3e-3, -0.5e-3, -40.9e-3});
    return p;
  }();
  Mount mount = mount_of(T_rig_cam, params);
};

Truth at(const Rig& rig, const Mount& m, const Vec3d& g, int64_t t_ns, double frac = 0) {
  const render::TrajectoryView v = rig.view();
  return truth_at(v, m, g, where_right(v, t_ns, frac));
}

}  // namespace

TEST_CASE("mount_of: T_B_I = T_rig_cam T_cam_imu") {
  const Fixture f;
  const Mat3d r_ci = rotation_of(f.params);
  const Mat3d expect = mul(f.T_rig_cam.r, r_ci);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) CHECK(f.mount.R_BI.m[i][j] == expect.m[i][j]);
  }
  const Vec3d p = f.T_rig_cam.r * translation_of(f.params) + f.T_rig_cam.p;
  CHECK(f.mount.p_B.x == p.x);
  CHECK(f.mount.p_B.z == p.z);
  const Mat3d back = to_matrix(f.mount.q_BI);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) CHECK(std::abs(back.m[i][j] - f.mount.R_BI.m[i][j]) < 1e-15);
  }
  ImuParams bad = f.params;
  bad.T_cam_imu[15] = 1 + 1e-12;
  CHECK_THROWS_AS(mount_of(f.T_rig_cam, bad), Refused);
  render::RigidPose nan_pose = f.T_rig_cam;
  nan_pose.p.y = std::nan("");
  CHECK_THROWS_AS(mount_of(nan_pose, f.params), Refused);
}

TEST_CASE("where_right: right-continuous at knots, u = (r + frac) / h, refused outside the span") {
  Rig rig = random_rig(3, 12, 0.02, 2.5e-4, 6, 5'000'000);
  rig.t0_ns = -7;
  const render::TrajectoryView v = rig.view();  // 7 segments, support [-7, 35e6 - 7]
  Where w = where_right(v, -7 + 2 * 5'000'000, 0.0);
  CHECK(w.piece == 0);
  CHECK(w.segment == 2);
  CHECK(w.u == 0.0);
  w = where_right(v, -7 + 2 * 5'000'000 - 1, 0.25);
  CHECK(w.segment == 1);
  CHECK(w.u == (4'999'999 + 0.25) / 5e6);
  w = where_right(v, v.span_end_ns, 0.0);  // the closed right end: the last segment at u = 1
  CHECK(w.segment == 6);
  CHECK(w.u == 1.0);
  CHECK_THROWS_AS(where_right(v, v.span_end_ns, 0.5), Refused);
  CHECK_THROWS_AS(where_right(v, v.span_end_ns + 1, 0.0), Refused);
  CHECK_THROWS_AS(where_right(v, v.span_begin_ns - 1, 0.5), Refused);
  rig.span_begin_ns = 1000;
  rig.span_end_ns = 2000;
  CHECK_THROWS_AS(where_right(rig.view(), 999, 0.9), Refused);
  CHECK(where_right(rig.view(), 1000, 0.0).u == 1007 / 5e6);
}

TEST_CASE("U1: a constant-rate spin gives omega_I = R_BI^T w0 and the centripetal specific force") {
  const Fixture f;
  const Vec3d w0{0.3, -1.2, 2.0}, g{0.0, 0.0, -9.80665};
  for (int K : {6, 4}) {
    INFO("K = " << K);
    const int32_t h = 5'000'000;
    const double h_s = h * 1e-9;
    std::vector<Quat<double>> q;
    std::vector<Vec3d> p;
    for (int c = 0; c < 60; ++c) {
      q.push_back(so3::exp((c * h_s) * w0));
      p.push_back({1.0, 2.0, 3.0});
    }
    const Rig rig = make_rig(q, p, K, h);
    const Vec3d centripetal = cross(w0, cross(w0, f.mount.p_B));
    for (int64_t t : {int64_t{3}, int64_t{12'345'678}, int64_t{100'000'000}, int64_t{200'000'000} - 1}) {
      for (double frac : {0.0, 0.375}) {
        const Truth tr = at(rig, f.mount, g, t, frac);
        // The spline reproduces R(t) = Exp((t - t0 + (K - 2) h / 2) w0): the cumulative basis sums to u + (K - 2)/2.
        const Quat<double> r_wb = so3::exp(((t + frac) * 1e-9 + (K - 2) * h_s / 2) * w0);
        const Vec3d omega = to_imu(f.mount, w0);
        const Vec3d force = to_imu(f.mount, rotate_inverse(r_wb, Vec3d{0, 0, 0} - g) + centripetal);
        CHECK(norm_inf(tr.omega_I - omega) < 1e-12);
        CHECK(norm_inf(tr.f_I - force) < 1e-12);
        CHECK(norm_inf(tr.p_W_I - (Vec3d{1, 2, 3} + rotate(r_wb, f.mount.p_B))) < 1e-12);
      }
    }
  }
}

TEST_CASE("U2b: finite differences of the IMU pose agree with omega_I and f_I") {
  const Fixture f;
  const Vec3d g{0.0, 0.0, -9.80665};
  for (int K : {6, 4}) {
    INFO("K = " << K);
    // EVIMO2-like motion (R-DOC C3's scales): |omega| up to ~8 rad/s, |a| up to ~30 m/s^2.
    const Rig rig = random_rig(11, 120, 0.02, 2.5e-4, K, 5'000'000);
    const int64_t hd = 10'000;  // 1e-5 s
    const double h_s = 1e-5;
    double peak_w = 0, peak_f = 0, err_w = 0, err_f = 0;
    for (int s = 2; s < 110; s += 3) {
      const int64_t t = int64_t{s} * 5'000'000 + 2'500'000;  // mid-segment: >= 3 h from every knot
      Truth tr[5];
      for (int j = -2; j <= 2; ++j) tr[j + 2] = at(rig, f.mount, g, t + j * hd);
      const Truth& c = tr[2];
      const Vec3d pdd = (1.0 / (12 * h_s * h_s)) *
                        ((((-1.0) * tr[4].p_W_I + 16.0 * tr[3].p_W_I) - 30.0 * c.p_W_I) + (16.0 * tr[1].p_W_I - tr[0].p_W_I));
      const Vec3d f_fd = rotate_inverse(c.q_W_I, pdd - g);
      // [omega_I]x = R_WI^T dR_WI/dt by central differences.
      const Mat3d rp = to_matrix(tr[3].q_W_I), rm = to_matrix(tr[1].q_W_I), r0 = to_matrix(c.q_W_I);
      Mat3d om{};
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int k = 0; k < 3; ++k) om.m[i][j] += r0.m[k][i] * (rp.m[k][j] - rm.m[k][j]) / (2 * h_s);
        }
      }
      const Vec3d w_fd{0.5 * (om.m[2][1] - om.m[1][2]), 0.5 * (om.m[0][2] - om.m[2][0]), 0.5 * (om.m[1][0] - om.m[0][1])};
      peak_w = std::max(peak_w, norm_inf(c.omega_I));
      peak_f = std::max(peak_f, norm_inf(c.f_I));
      err_w = std::max(err_w, norm_inf(w_fd - c.omega_I));
      err_f = std::max(err_f, norm_inf(f_fd - c.f_I));
    }
    MESSAGE("K " << K << ": omega peak " << peak_w << " err " << err_w << "; f peak " << peak_f << " err " << err_f);
    CHECK(peak_w > 1);
    CHECK(peak_f > 10);
    CHECK(err_w <= 1e-4 * peak_w);
    CHECK(err_f <= 1e-4 * peak_f);
  }
}

TEST_CASE("U3: inside a hold omega_I and the lever terms are exactly zero and f_I = R_BI^T R_WB^T (-g)") {
  const Fixture f;
  const Vec3d g{0.1, -0.2, -9.80665};
  const Quat<double> q0 = so3::exp(Vec3d{0.7, -0.4, 1.9});
  for (int K : {6, 4}) {
    INFO("K = " << K);
    const std::vector<Quat<double>> q(20, q0);
    const std::vector<Vec3d> p(20, Vec3d{0.4, -2.0, 1.1});
    const Rig rig = make_rig(q, p, K, 5'000'000);
    const render::TrajectoryView v = rig.view();
    for (int64_t t : {int64_t{0}, int64_t{7'777'777}, int64_t{40'000'000}}) {
      const Where w = where_right(v, t, 0.125);
      const traj::Kinematics<double> k =
          K == 4 ? traj::evaluate_u<4>(v.spline, w.segment, w.u) : traj::evaluate_u<6>(v.spline, w.segment, w.u);
      CHECK(k.omega.x == 0.0);
      CHECK(k.omega.y == 0.0);
      CHECK(k.omega.z == 0.0);
      CHECK(norm_inf(k.omega_dot) == 0.0);
      CHECK(norm_inf(k.a) == 0.0);
      CHECK(norm_inf(cross(k.omega_dot, f.mount.p_B)) == 0.0);
      CHECK(norm_inf(cross(k.omega, cross(k.omega, f.mount.p_B))) == 0.0);
      const Truth tr = truth_at(v, f.mount, g, w);
      CHECK(tr.omega_I.x == 0.0);
      CHECK(tr.omega_I.y == 0.0);
      CHECK(tr.omega_I.z == 0.0);
      const Vec3d expect = to_imu(f.mount, rotate_inverse(k.q, k.a - g));
      CHECK(tr.f_I.x == expect.x);
      CHECK(tr.f_I.y == expect.y);
      CHECK(tr.f_I.z == expect.z);
      // f_I = -R_WI^T g up to the rounding of the two rotation paths.
      CHECK(norm(tr.f_I + rotate_inverse(tr.q_W_I, g)) <= 4 * DBL_EPSILON * norm(g));
    }
  }
}
