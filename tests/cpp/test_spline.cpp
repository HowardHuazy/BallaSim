#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "core/math/so3.h"
#include "core/time/time.h"
#include "core/traj/basis.h"
#include "core/traj/bounds.h"
#include "core/traj/pieces.h"
#include "core/traj/spline.h"
#include "core/traj/spline_build.h"
#include "test_util.h"

using namespace balla;
using namespace balla::traj;
using balla::test::angle_between;
using balla::test::distance;
using balla::test::random_unit;

namespace {

constexpr int32_t kH = 5'000'000;  // 5 ms knots

// Smooth random control poses: random-walk angular velocity (|w| <= w_max) and acceleration.
struct Controls {
  std::vector<Quat<double>> q;
  std::vector<Vec3<double>> p;
};

Controls random_controls(std::mt19937_64& rng, int n, double w_max, double a_max) {
  std::normal_distribution<double> g;
  const double h = kH * 1e-9;
  Controls c;
  Quat<double> q = quat_identity<double>();
  Vec3<double> w{0, 0, 0}, v{0, 0, 0}, a{0, 0, 0}, p{2.0, -1.5, 1.2};
  for (int i = 0; i < n; ++i) {
    c.q.push_back(q);
    c.p.push_back(p);
    w += (0.15 * w_max) * Vec3<double>{g(rng), g(rng), g(rng)};
    if (norm(w) > w_max) w = (w_max / norm(w)) * w;
    a += (0.15 * a_max) * Vec3<double>{g(rng), g(rng), g(rng)};
    if (norm(a) > a_max) a = (a_max / norm(a)) * a;
    q = q * so3::exp(h * w);
    v += h * a;
    if (norm(v) > 5.0) v = (5.0 / norm(v)) * v;  // M2 D6 conditions: |v| <= 5 m/s
    p += h * v;
  }
  return c;
}

template <int K, class T>
struct Spline {
  std::vector<SplineNode<T>> nodes;
  SplineView<T> view() const {
    return {nodes.data(), static_cast<int32_t>(nodes.size()) - K + 1, kH, static_cast<T>(1e9 / kH)};
  }
};

template <int K>
Spline<K, double> make_spline(const Controls& c, Vec3<double> origin = {0, 0, 0}) {
  return {build_spline_nodes(c.q, c.p, origin)};
}

// Evaluation at real-valued time (seconds from t0), for finite differences.
template <int K>
Kinematics<double> eval_at(const SplineView<double>& v, double t_seconds) {
  const double x = t_seconds / (kH * 1e-9);
  const double s = std::floor(x);
  return evaluate_u<K>(v, static_cast<int32_t>(s), x - s);
}

}  // namespace

TEST_CASE_TEMPLATE("cumulative basis: values, partition of unity, knot continuity", T, float, double) {
  const double eps = std::is_same_v<T, double> ? 1e-15 : 1e-6;
  const CumulativeBasis<T, 4> b4 = cumulative_basis<4>(T(0));
  CHECK(b4.lambda[0] == doctest::Approx(5.0 / 6).epsilon(eps));
  CHECK(b4.lambda[1] == doctest::Approx(1.0 / 6).epsilon(eps));
  CHECK(b4.lambda[2] == 0);
  const CumulativeBasis<T, 6> b6 = cumulative_basis<6>(T(0));
  CHECK(b6.lambda[0] == doctest::Approx(119.0 / 120).epsilon(eps));
  CHECK(b6.lambda[3] == doctest::Approx(1.0 / 120).epsilon(eps));
  for (int i = 0; i <= 64; ++i) {
    const T u = T(i) / T(64);
    const CumulativeBasis<T, 6> b = cumulative_basis<6>(u);
    double sum = 0;
    for (int j = 0; j < 5; ++j) {
      CHECK(b.dlambda[j] >= T(-eps));  // M2 Lemma B1: lambda'_j >= 0, sum = 1
      sum += static_cast<double>(b.dlambda[j]);
    }
    CHECK(sum == doctest::Approx(1.0).epsilon(10 * eps));
  }
  // lambda_j(1) of segment s equals lambda_{j-1}(0) of segment s + 1 (lambda_0 = 1).
  const CumulativeBasis<T, 6> e = cumulative_basis<6>(T(1));
  CHECK(e.lambda[0] == doctest::Approx(1.0).epsilon(eps));
  for (int j = 1; j < 5; ++j) {
    CHECK(e.lambda[j] == doctest::Approx(static_cast<double>(b6.lambda[j - 1])).epsilon(eps));
    CHECK(e.dlambda[j] == doctest::Approx(static_cast<double>(b6.dlambda[j - 1])).epsilon(eps));
  }
}

TEST_CASE("segment location: floor semantics, closed right end, out of span") {
  SegmentTime st{};
  const int64_t t0 = -1'000'000'000;  // negative epoch-relative start
  CHECK(locate(t0, t0, kH, 10, &st));
  CHECK(st.segment == 0);
  CHECK(st.r_ns == 0);
  CHECK(locate(t0 + 3 * int64_t{kH} + 7, t0, kH, 10, &st));
  CHECK(st.segment == 3);
  CHECK(st.r_ns == 7);
  CHECK(locate(t0 + 10 * int64_t{kH}, t0, kH, 10, &st));  // t_end maps to u = 1
  CHECK(st.segment == 9);
  CHECK(st.r_ns == kH);
  CHECK_FALSE(locate(t0 - 1, t0, kH, 10, &st));
  CHECK_FALSE(locate(t0 + 10 * int64_t{kH} + 1, t0, kH, 10, &st));
  // Distances of 2^63 ns or more (t - t0 overflowed int64; a wrapped one could land in the span).
  CHECK_FALSE(locate(INT64_MAX, t0, kH, 10, &st));
  // t - t0 wraps to 3 h: the old form placed this time, 2^64 - 3 h before the span, in segment 3.
  CHECK_FALSE(locate(INT64_MIN + 3 * int64_t{kH} - 11, INT64_MAX - 10, kH, 10, &st));
  CHECK(locate_offset(4, -1, kH, 10, &st));
  CHECK(st.segment == 3);
  CHECK(st.r_ns == kH - 1);
  CHECK(locate_offset(4, 6 * kH, kH, 10, &st));
  CHECK(st.segment == 9);
  CHECK(st.r_ns == kH);
  CHECK_FALSE(locate_offset(4, 6 * kH + 1, kH, 10, &st));
  // Offsets within h of INT32_MIN: the remainder must not overflow int32 (it did as tau - sl h).
  const int32_t sl_min = time::floor_div(INT32_MIN, kH);
  CHECK(locate_offset(-sl_min, INT32_MIN, kH, 10, &st));
  CHECK(st.segment == 0);
  CHECK(st.r_ns == static_cast<int32_t>(int64_t{INT32_MIN} - int64_t{sl_min} * kH));
  CHECK_FALSE(locate_offset(0, INT32_MIN, kH, 10, &st));
  CHECK(locate_offset(0, INT32_MAX, kH, 500, &st));
  CHECK(st.segment == INT32_MAX / kH);
  CHECK(st.r_ns == INT32_MAX % kH);
}

TEST_CASE("piece lookup: the last begin <= t, right-continuous at breakpoints, piece 0 before the first begin") {
  // C1 Trajectory.pieces (blender.md §4.7.7): int64 begins, one of them beyond 2^31 ns from the others.
  const int64_t b[4] = {-5'000'000'000, -7, 0, int64_t{3} << 40};
  std::vector<SplinePiece<double>> pieces;
  for (const int64_t begin : b) pieces.push_back({begin, begin, {nullptr, 1, kH, 1e9 / kH}});
  const auto n = static_cast<int32_t>(pieces.size());
  CHECK(piece_index(pieces.data(), n, INT64_MIN) == 0);
  CHECK(piece_index(pieces.data(), n, b[0] - 1) == 0);
  for (int32_t p = 0; p < n; ++p) {
    CHECK(piece_index(pieces.data(), n, b[p]) == p);  // a breakpoint takes the right piece
    if (p > 0) CHECK(piece_index(pieces.data(), n, b[p] - 1) == p - 1);
    if (p + 1 < n) CHECK(piece_index(pieces.data(), n, b[p + 1] - 1) == p);
  }
  CHECK(piece_index(pieces.data(), n, INT64_MAX) == n - 1);
  CHECK(piece_index(pieces.data(), 1, INT64_MAX) == 0);
  // locate_piece locates within the chosen piece's own knots and support: piece 1 spans [-7, -7 + h].
  int32_t piece = -1;
  SegmentTime st{};
  CHECK(locate_piece(pieces.data(), n, -2, &piece, &st));
  CHECK(piece == 1);
  CHECK(st.segment == 0);
  CHECK(st.r_ns == 5);
  CHECK_FALSE(locate_piece(pieces.data(), n, -8 - kH, &piece, &st));  // piece 0's support ends at b[0] + h
  CHECK(piece == 0);
}

TEST_CASE_TEMPLATE("constant increments give constant twists (bounds attained)", T, float, double) {
  const Vec3<double> d{0.03, -0.02, 0.01}, dp{0.01, 0.0, -0.005};
  Controls c;
  Quat<double> q = quat_identity<double>();
  Vec3<double> p{1, 2, 3};
  for (int i = 0; i < 12; ++i) {
    c.q.push_back(q);
    c.p.push_back(p);
    q = q * so3::exp(d);
    p += dp;
  }
  const Spline<6, T> s{convert_nodes<T>(build_spline_nodes(c.q, c.p, {0, 0, 0}))};
  const SplineView<T> view = s.view();
  const double tol = std::is_same_v<T, double> ? 1e-12 : 2e-5;
  for (int32_t r : {0, 1234567, kH / 2, kH}) {
    const Kinematics<T> k = evaluate<6>(view, SegmentTime{3, r});
    CHECK(distance(k.omega, cast<T>((1e9 / kH) * d)) <= tol * 1e9 / kH * norm(d));
    CHECK(distance(k.v, cast<T>((1e9 / kH) * dp)) <= tol * 1e9 / kH * norm(dp));
    CHECK(norm(cast<double>(k.omega_dot)) <= tol * 1e18 / kH / kH * norm(d));
    CHECK(norm(cast<double>(k.a)) <= 1e3 * tol);
    const MotionBounds<T> b = segment_bounds<6>(view, 3);
    CHECK(static_cast<double>(b.omega) >= norm(cast<double>(k.omega)));
    CHECK(static_cast<double>(b.omega) == doctest::Approx(norm(d) * 1e9 / kH).epsilon(tol));
  }
}

TEST_CASE_TEMPLATE("derivatives match finite differences", K_, std::integral_constant<int, 4>,
                   std::integral_constant<int, 6>) {
  constexpr int K = K_::value;
  std::mt19937_64 rng(11);
  const Controls c = random_controls(rng, 60, 41.0, 27.0);
  const Spline<K, double> s = make_spline<K>(c);
  const SplineView<double> v = s.view();
  const double h = kH * 1e-9;
  const double dt = 1e-6;
  std::uniform_real_distribution<double> ut(5 * h, 50 * h);
  for (int i = 0; i < 200; ++i) {
    const double t = ut(rng);
    const Kinematics<double> k = eval_at<K>(v, t), km = eval_at<K>(v, t - dt), kp = eval_at<K>(v, t + dt);
    // omega_B from R(t-dt)^T R(t+dt) = Exp(2 dt omega_B) + O(dt^3).
    const Vec3<double> w_fd = (0.5 / dt) * so3::log(conjugate(km.q) * kp.q);
    CHECK(distance(k.omega, w_fd) <= 1e-6 * (1.0 + norm(k.omega)));
    CHECK(distance(k.omega_dot, (0.5 / dt) * (kp.omega - km.omega)) <= 1e-5 * (1.0 + norm(k.omega_dot)));
    CHECK(distance(k.v, (0.5 / dt) * (kp.p - km.p)) <= 1e-6 * (1.0 + norm(k.v)));
    CHECK(distance(k.a, (0.5 / dt) * (kp.v - km.v)) <= 1e-5 * (1.0 + norm(k.a)));
  }
}

namespace {

// |v| without rounding or underflow (long double has 15 exponent bits): the bounds must cover the
// magnitude of the vector the evaluation returned, whatever its scale.
template <class T>
long double magnitude(Vec3<T> v) {
  const long double x = v.x, y = v.y, z = v.z;
  return std::sqrt(x * x + y * y + z * z);
}

// bounds.h claims its bounds for the floating-point evaluation of either precision, so both the
// fp64 path and the fp32 device path (nodes derived in fp64, then cast) are checked against the
// bounds computed in their own precision.
template <int K, class T>
void check_bounds_hold(const Controls& c) {
  Vec3<double> origin{0, 0, 0};
  for (const Vec3<double>& p : c.p) origin += (1.0 / static_cast<double>(c.p.size())) * p;
  const Spline<K, T> s{convert_nodes<T>(build_spline_nodes(c.q, c.p, origin))};
  const SplineView<T> v = s.view();
  for (int32_t seg = 0; seg < v.num_segments; ++seg) {
    const MotionBounds<T> b = segment_bounds<K>(v, seg);
    for (int i = 0; i <= 32; ++i) {
      const Kinematics<T> k = evaluate<K>(v, SegmentTime{seg, kH / 32 * i});
      CHECK(magnitude(k.omega) <= b.omega);
      CHECK(magnitude(k.omega_dot) <= b.omega_dot);
      CHECK(magnitude(k.v) <= b.v);
      CHECK(magnitude(k.a) <= b.a);
    }
  }
}

// Constant increments attain (B6) and (B7) exactly, so only the slack keeps the bound above the
// rounded evaluation; tiny and near-constant increments probe the absolute rounding allowance.
Controls constant_controls(Vec3<double> d, Vec3<double> dp, int n) {
  Controls c;
  Quat<double> q = quat_identity<double>();
  Vec3<double> p{1.5, -2.0, 0.7};
  for (int i = 0; i < n; ++i) {
    c.q.push_back(q);
    c.p.push_back(p);
    q = q * so3::exp(d);
    p += dp;
  }
  return c;
}

}  // namespace

TEST_CASE_TEMPLATE("motion bounds are never violated", K_, std::integral_constant<int, 4>,
                   std::integral_constant<int, 6>) {
  constexpr int K = K_::value;
  std::mt19937_64 rng(13);
  for (int trial = 0; trial < 20; ++trial) {
    const double w_max = trial % 2 ? 41.0 : 300.0;  // up to |d| = 1.5 rad per knot
    const Controls c = random_controls(rng, 40, w_max, 200.0);
    check_bounds_hold<K, double>(c);
    check_bounds_hold<K, float>(c);
  }
  std::uniform_real_distribution<double> mag(-3.0, 0.2);
  for (int trial = 0; trial < 40; ++trial) {
    const Vec3<double> d = std::pow(10.0, mag(rng)) * random_unit<double>(rng);   // up to 1.6 rad
    const Vec3<double> dp = std::pow(10.0, mag(rng)) * random_unit<double>(rng);  // up to 1.6 m
    const Controls c = constant_controls(d, dp, 16);
    check_bounds_hold<K, double>(c);
    check_bounds_hold<K, float>(c);
  }
}

TEST_CASE_TEMPLATE("motion bounds hold when the norms of the increments underflow", K_, std::integral_constant<int, 4>,
                   std::integral_constant<int, 6>) {
  // Below |x| ~ 1.5e-162 (fp64) and 5e-23 (fp32) the squares in norm() underflow, so a bound
  // without an absolute allowance is 0 while the evaluated velocity is not (hypothesis found
  // |dp| = 3.5e-163 with order 4, tests/python/test_spline.py). Scales reach the subnormal range of
  // either precision.
  constexpr int K = K_::value;
  std::mt19937_64 rng(19);
  std::uniform_real_distribution<double> spread(0.1, 1.0);
  for (const double scale : {1e-310, 1e-300, 1e-200, 1e-170, 2e-163, 1e-160, 1e-150, 1e-44, 1e-40, 1e-30, 1e-23}) {
    for (int trial = 0; trial < 4; ++trial) {
      Controls c;
      Quat<double> q = so3::exp(random_unit<double>(rng));
      Vec3<double> p{0.0, 0.0, 0.0};
      for (int i = 0; i < 16; ++i) {
        c.q.push_back(q);
        c.p.push_back(p);
        q = q * so3::exp((scale * spread(rng)) * random_unit<double>(rng));
        p += (scale * spread(rng)) * random_unit<double>(rng);
      }
      check_bounds_hold<K, double>(c);
      check_bounds_hold<K, float>(c);
    }
  }
}

TEST_CASE("equal control points make the spline exactly still, rotation included") {
  // A hold: controls 3..10 are one pose, unnormalized, the last one with the opposite sign.
  std::mt19937_64 rng(23);
  Controls c = random_controls(rng, 16, 41.0, 27.0);
  const Quat<double> held{0.31, -0.62, 0.17, 0.5};  // |q| != 1: normalized by build_spline_nodes
  for (int i = 3; i <= 10; ++i) {
    c.q[static_cast<size_t>(i)] = i < 10 ? held : Quat<double>{-held.w, -held.x, -held.y, -held.z};
    c.p[static_cast<size_t>(i)] = Vec3<double>{0.4, -1.1, 2.3};
  }
  const Spline<6, double> s64 = make_spline<6>(c);
  for (int i = 4; i <= 10; ++i) {
    const SplineNode<double>& n = s64.nodes[static_cast<size_t>(i)];
    CHECK((n.d.x == 0.0 && n.d.y == 0.0 && n.d.z == 0.0));
    CHECK((n.dp.x == 0.0 && n.dp.y == 0.0 && n.dp.z == 0.0));
  }
  CHECK(norm(s64.nodes[3].d) > 0.0);  // entering the hold is a real increment
  CHECK(norm(s64.nodes[11].d) > 0.0);
  // Segments 3..5 use only controls 3..10 (K = 6): the pose is bitwise constant and every
  // derivative exactly 0, in fp64 and in the fp32 device form.
  const Spline<6, float> s32{convert_nodes<float>(s64.nodes)};
  const Kinematics<double> first64 = evaluate<6>(s64.view(), SegmentTime{3, 0});
  const Kinematics<float> first32 = evaluate<6>(s32.view(), SegmentTime{3, 0});
  for (int32_t seg = 3; seg <= 5; ++seg) {
    for (int32_t r = 0; r <= kH; r += kH / 64) {
      const Kinematics<double> k = evaluate<6>(s64.view(), SegmentTime{seg, r});
      const Kinematics<float> kf = evaluate<6>(s32.view(), SegmentTime{seg, r});
      CHECK(std::memcmp(&k.q, &first64.q, sizeof k.q) == 0);
      CHECK(std::memcmp(&k.p, &first64.p, sizeof k.p) == 0);
      CHECK(std::memcmp(&kf.q, &first32.q, sizeof kf.q) == 0);
      CHECK(std::memcmp(&kf.p, &first32.p, sizeof kf.p) == 0);
      for (const Vec3<double>& z : {k.omega, k.omega_dot, k.v, k.a}) CHECK((z.x == 0 && z.y == 0 && z.z == 0));
      for (const Vec3<float>& z : {kf.omega, kf.omega_dot, kf.v, kf.a}) CHECK((z.x == 0 && z.y == 0 && z.z == 0));
    }
  }
}

TEST_CASE_TEMPLATE("fp32 device path stays within the M2 §E.1 error budget", K_, std::integral_constant<int, 4>,
                   std::integral_constant<int, 6>) {
  constexpr int K = K_::value;  // both orders are user-selectable (scene.trajectory.order)
  std::mt19937_64 rng(17);
  const Controls c = random_controls(rng, 400, 41.0, 27.0);
  Vec3<double> origin{0, 0, 0};
  for (const Vec3<double>& p : c.p) origin += (1.0 / static_cast<double>(c.p.size())) * p;
  const Spline<K, double> s64 = make_spline<K>(c, origin);
  const Spline<K, float> s32{convert_nodes<float>(s64.nodes)};
  const SplineView<double> v64 = s64.view();
  const SplineView<float> v32 = s32.view();
  std::uniform_int_distribution<int32_t> tau(0, 300 * kH);
  double rot = 0, pos = 0, w = 0, wmax = 0, wd = 0, wdmax = 0, vel = 0, acc = 0, amax = 0;
  for (int i = 0; i < 20000; ++i) {
    SegmentTime a{}, b{};
    const int32_t t = tau(rng);
    REQUIRE(locate_offset(20, t, kH, v32.num_segments, &a));
    REQUIRE(locate(int64_t{20} * kH + t, 0, kH, v64.num_segments, &b));
    const Kinematics<double> k64 = evaluate<K>(v64, b);
    const Kinematics<float> k32 = evaluate<K>(v32, a);
    rot = std::max(rot, angle_between(cast<double>(k32.q), k64.q));
    pos = std::max(pos, distance(cast<double>(k32.p), k64.p));  // reference kept in fp64
    w = std::max(w, distance(cast<double>(k32.omega), k64.omega));
    wmax = std::max(wmax, norm(k64.omega));
    wd = std::max(wd, distance(cast<double>(k32.omega_dot), k64.omega_dot));
    wdmax = std::max(wdmax, norm(k64.omega_dot));
    vel = std::max(vel, distance(cast<double>(k32.v), k64.v));
    acc = std::max(acc, distance(cast<double>(k32.a), k64.a));
    amax = std::max(amax, norm(k64.a));
  }
  MESSAGE("K = " << K << ": rot " << rot << ", pos " << pos << ", omega " << w / wmax << ", omega_dot " << wd / wdmax
                     << ", v " << vel << ", a " << acc / amax);
  CHECK(rot <= 6.4e-7);  // M2 §E.1 maxima
  CHECK(pos <= 1.3e-6);
  CHECK(w <= 3.1e-7 * wmax);
  CHECK(wd <= 7.2e-7 * wdmax);
  CHECK(vel <= 1.3e-6);
  CHECK(acc <= 1e-5 * amax);
}
