#include <doctest/doctest.h>

#include <cmath>
#include <numbers>
#include <random>

#include "core/math/quat.h"
#include "core/math/so3.h"
#include "test_util.h"

using namespace balla;
using balla::test::angle_between;
using balla::test::distance;
using balla::test::random_unit;

namespace {
template <class T>
constexpr double tol() {
  return std::is_same_v<T, double> ? 4e-15 : 4e-6;
}
}  // namespace

TEST_CASE_TEMPLATE("Hamilton product and rotation agree with the matrix form", T, float, double) {
  std::mt19937_64 rng(1);
  for (int i = 0; i < 1000; ++i) {
    const Quat<T> a = so3::exp(T(3) * random_unit<T>(rng));
    const Quat<T> b = so3::exp(T(2) * random_unit<T>(rng));
    const Vec3<T> v = random_unit<T>(rng);
    CHECK(distance(rotate(a * b, v), to_matrix(a) * (to_matrix(b) * v)) < tol<T>());
    CHECK(distance(rotate_inverse(a, rotate(a, v)), v) < tol<T>());
    CHECK(angle_between(from_matrix(to_matrix(a)), a) < 10 * tol<T>());
  }
  // Hamilton convention: i * j = k.
  const Quat<T> i{T(0), T(1), T(0), T(0)}, j{T(0), T(0), T(1), T(0)};
  const Quat<T> k = i * j;
  CHECK(k.w == T(0));
  CHECK(k.z == T(1));
}

TEST_CASE_TEMPLATE("SO(3) exp/log round trips, including near 0 and near pi", T, float, double) {
  std::mt19937_64 rng(2);
  const double pi = std::numbers::pi;
  const double angles[] = {0.0, 1e-12, 1e-7, 1e-4, 1e-2, 0.5, 1.0, 2.0, 3.0, pi - 1e-3, pi - 1e-6};
  for (double angle : angles) {
    for (int i = 0; i < 200; ++i) {
      const Vec3<T> phi = static_cast<T>(angle) * random_unit<T>(rng);
      const Quat<T> q = so3::exp(phi);
      CHECK(std::abs(std::sqrt(static_cast<double>(norm2(q))) - 1.0) < tol<T>());
      const Vec3<T> back = so3::log(q);
      // Near pi the axis is ill-conditioned in the quaternion, so compare rotations as well.
      if (angle < pi - 1e-3) {
        CHECK(distance(back, phi) <= tol<T>() * (1.0 + angle));
      }
      CHECK(angle_between(so3::exp(back), q) < 10 * tol<T>());
      CHECK(norm(cast<double>(back)) <= pi + 1e-6);
    }
  }
}

TEST_CASE("SO(3) log is sign invariant and returns the shortest rotation") {
  std::mt19937_64 rng(3);
  for (int i = 0; i < 1000; ++i) {
    const Quat<double> q = so3::exp(2.5 * random_unit<double>(rng));
    const Quat<double> neg{-q.w, -q.x, -q.y, -q.z};
    CHECK(distance(so3::log(q), so3::log(neg)) == 0.0);
  }
  // A rotation by 3pi/2 about z is the rotation by -pi/2.
  const Vec3<double> phi = so3::log(so3::exp(Vec3<double>{0.0, 0.0, 1.5 * std::numbers::pi}));
  CHECK(phi.z == doctest::Approx(-0.5 * std::numbers::pi).epsilon(1e-15));
  // Exactly pi: either sign of the axis is correct.
  const Vec3<double> half = so3::log(Quat<double>{0.0, 0.0, 1.0, 0.0});
  CHECK(std::abs(half.y) == doctest::Approx(std::numbers::pi).epsilon(1e-15));
}

TEST_CASE("small-angle branches agree with high-precision references at the switch-over") {
  // Just below and above the switch-over angle, both branches match long-double references.
  const Vec3<double> axis{0.6, 0.0, 0.8};
  for (double f : {1 - 1e-6, 1 + 1e-6}) {
    const long double theta = std::sqrt(static_cast<long double>(so3::detail::SmallAngle<double>::kTheta2 * f));
    const Quat<double> q = so3::exp(static_cast<double>(theta) * axis);
    const long double s = std::sin(theta / 2) / theta;
    CHECK(std::abs(q.w - static_cast<double>(std::cos(theta / 2))) <= 2e-16);
    CHECK(std::abs(q.x - static_cast<double>(s * 0.6L * theta)) <= 2e-16 * static_cast<double>(theta));
    const Vec3<double> back = so3::log(q);
    CHECK(std::abs(back.z - static_cast<double>(0.8L * theta)) <= 4e-16 * static_cast<double>(theta));
  }
  const Vec3<float> axisf{0.6f, 0.0f, 0.8f};
  for (double f : {1 - 1e-3, 1 + 1e-3}) {
    const double theta = std::sqrt(static_cast<double>(so3::detail::SmallAngle<float>::kTheta2) * f);
    const Quat<float> q = so3::exp(static_cast<float>(theta) * axisf);
    CHECK(std::abs(static_cast<double>(q.w) - std::cos(theta / 2)) <= 1.2e-7);
    CHECK(std::abs(static_cast<double>(q.z) - std::sin(theta / 2) * 0.8) <= 1.2e-7 * theta);
    const Vec3<float> back = so3::log(q);
    CHECK(std::abs(static_cast<double>(back.z) - 0.8 * theta) <= 4e-7 * theta);
  }
}
