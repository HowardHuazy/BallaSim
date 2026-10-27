// decode (poses.h): every quantity derived from the oracle's raw answers is computed here, in C++, so a recorded
// oracle log replays bitwise (blender.md §3.6).
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

#include "core/math/quat.h"
#include "scene/motion/poses.h"
#include "scene/pyfmt.h"

namespace balla::scene::motion {

namespace {

using M3 = std::array<std::array<double, 3>, 3>;

double det3(const M3& a) {
  return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
         a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

// A^-T = cofactor(A) / det A.
M3 inverse_transpose(const M3& a, double det) {
  M3 c;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const int i1 = (i + 1) % 3, i2 = (i + 2) % 3, j1 = (j + 1) % 3, j2 = (j + 2) % 3;
      c[i][j] = (a[i1][j1] * a[i2][j2] - a[i1][j2] * a[i2][j1]) / det;
    }
  }
  return c;
}

// The orthogonal polar factor of a matrix with det > 0, in place: Newton's iteration X <- (g X + X^-T / g) / 2 with
// the determinant scaling g = det(X)^(-1/3) (Higham, "Functions of Matrices", SIAM 2008, §8.6). False when an iterate
// is singular.
bool polar(M3& x) {
  for (int it = 0; it < 100; ++it) {
    const double d = det3(x);
    if (!(d > 0.0) || !std::isfinite(d)) return false;
    const M3 inv_t = inverse_transpose(x, d);
    const double g = it < 20 ? std::cbrt(1.0 / d) : 1.0;
    double diff = 0.0, norm = 0.0;
    M3 y;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        y[i][j] = 0.5 * (g * x[i][j] + inv_t[i][j] / g);
        diff += (y[i][j] - x[i][j]) * (y[i][j] - x[i][j]);
        norm += y[i][j] * y[i][j];
      }
    }
    x = y;
    if (diff <= 1e-30 * norm) break;
  }
  return true;
}

// The smallest |eigenvalue| of a symmetric 3x3 matrix (trigonometric solution of the characteristic cubic).
double min_abs_eigenvalue(const M3& a) {
  const double p1 = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
  if (p1 == 0.0) return std::min({std::fabs(a[0][0]), std::fabs(a[1][1]), std::fabs(a[2][2])});
  const double q = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
  const double p2 = (a[0][0] - q) * (a[0][0] - q) + (a[1][1] - q) * (a[1][1] - q) + (a[2][2] - q) * (a[2][2] - q) +
                    2.0 * p1;
  const double p = std::sqrt(p2 / 6.0);
  M3 b = a;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) b[i][j] = (a[i][j] - (i == j ? q : 0.0)) / p;
  }
  const double r = std::clamp(det3(b) / 2.0, -1.0, 1.0);
  const double phi = std::acos(r) / 3.0;
  const double e1 = q + 2.0 * p * std::cos(phi), e3 = q + 2.0 * p * std::cos(phi + 2.0 * std::numbers::pi / 3.0);
  const double e2 = 3.0 * q - e1 - e3;
  return std::min({std::fabs(e1), std::fabs(e2), std::fabs(e3)});
}

double half_ulp(float x) {
  x = std::fabs(x);
  return 0.5 * (static_cast<double>(std::nextafter(x, INFINITY)) - static_cast<double>(x));
}

}  // namespace

void PoseBatch::append(const PoseBatch& o) {
  if (bodies != o.bodies) throw std::invalid_argument("PoseBatch::append: body counts differ");
  auto cat = [](auto& a, const auto& b) { a.insert(a.end(), b.begin(), b.end()); };
  cat(t_ns, o.t_ns);
  cat(c, o.c);
  cat(q, o.q);
  cat(p, o.p);
  cat(S, o.S);
  cat(present, o.present);
  cat(dp_round, o.dp_round);
  cat(dtheta_round, o.dtheta_round);
  cat(w_pos, o.w_pos);
  cat(w_rot, o.w_rot);
  cat(floor1, o.floor1);
  cat(light_near, o.light_near);
}

void check_scale_refs(const MotionScene& scene) {
  for (size_t b = 0; b < scene.bodies.size(); ++b) {
    const MotionBody& body = scene.bodies[b];
    if (body.role == Role::kRig) continue;
    const char* what = body.role == Role::kBody ? "mesh body" : "light carrier";
    if (!body.S_ref) {
      throw std::invalid_argument("motion scene: body column " + std::to_string(b) + " (a " + what +
                                  ") has no S_ref, its scale at the export frame");
    }
    double ref = 0.0;
    for (const double x : *body.S_ref) ref += x * x;
    if (!(std::isfinite(ref) && ref > 0.0)) {
      throw std::invalid_argument("motion scene: body column " + std::to_string(b) + " (a " + what +
                                  ") has a non-finite or zero S_ref");
    }
  }
}

PoseBatch decode(const RawPoses& raw, const TimeMap& time, const MotionScene& scene) {
  check_scale_refs(scene);
  const auto n = static_cast<size_t>(raw.n), nb = static_cast<size_t>(raw.bodies);
  if (raw.n < 0 || raw.bodies != static_cast<int32_t>(scene.bodies.size()) || raw.F.size() != n ||
      raw.s.size() != n || raw.c_final.size() != n || raw.m.size() != n * nb * 12 || raw.present.size() != n * nb ||
      raw.unknown.size() != n) {
    throw MotionError("BRG-PROTO", "a poses answer has the wrong shape");
  }
  PoseBatch out;
  out.bodies = raw.bodies;
  out.t_ns.resize(n);
  out.c.resize(n);
  out.q.assign(n * nb, rot::Q{1, 0, 0, 0});
  out.p.assign(n * nb, rot::V{0, 0, 0});
  out.S.assign(n * nb, {1, 0, 0, 0, 1, 0, 0, 0, 1});
  out.present.assign(raw.present.begin(), raw.present.end());
  out.dp_round.assign(n * nb, 0.0);
  out.dtheta_round.assign(n * nb, 0.0);
  for (size_t i = 0; i < n; ++i) {
    const float c = TimeMap::realized(raw.F[i], raw.s[i]);
    if (!(raw.c_final[i] == static_cast<double>(time.reported(c)))) {  // shortest round-trip digits: 1 ulp shows
      throw MotionError("BRG-TIME", "frame_current_final " + pyfmt::repr(raw.c_final[i]) + " at F = " +
                                        std::to_string(raw.F[i]) + ", s = " +
                                        pyfmt::repr(static_cast<double>(raw.s[i])) + " differs from the expected " +
                                        pyfmt::repr(static_cast<double>(time.reported(c))));
    }
    out.c[i] = c;
    out.t_ns[i] = time.to_ns(c);
    for (size_t b = 0; b < nb; ++b) {
      const size_t k = i * nb + b;
      if (!raw.present[k]) continue;
      const float* m = &raw.m[k * 12];
      if (!std::all_of(m, m + 12, [](float x) { return std::isfinite(x); })) {
        throw MotionError("BRG-PROTO", "body column " + std::to_string(b) + " has a non-finite matrix_world at t = " +
                                           std::to_string(out.t_ns[i]) + " ns (F = " + std::to_string(raw.F[i]) +
                                           ", s = " + pyfmt::repr(static_cast<double>(raw.s[i])) + ")");
      }
      M3 a;
      float amax = 0.0f;
      for (int r = 0; r < 3; ++r) {
        for (int j = 0; j < 3; ++j) {
          a[r][j] = static_cast<double>(m[4 * r + j]);
          amax = std::max(amax, std::fabs(m[4 * r + j]));
        }
      }
      const double d = det3(a);
      M3 neg = a;
      if (d < 0.0) {
        for (auto& row : neg) {
          for (double& x : row) x = -x;
        }
      }
      M3 r = neg;
      if (!polar(r)) {
        throw MotionError("GEO-SCALEANIM", "body column " + std::to_string(b) + " has a singular matrix_world at t = " +
                                               std::to_string(out.t_ns[i]) + " ns");
      }
      M3 s{};
      for (int i2 = 0; i2 < 3; ++i2) {
        for (int j = 0; j < 3; ++j) {
          s[i2][j] = r[0][i2] * a[0][j] + r[1][i2] * a[1][j] + r[2][i2] * a[2][j];
        }
      }
      Mat3<double> rm;
      for (int i2 = 0; i2 < 3; ++i2) {
        for (int j = 0; j < 3; ++j) rm.m[i2][j] = r[i2][j];
      }
      const rot::Q q = from_matrix(rm);
      const MotionBody& body = scene.bodies[b];
      std::array<double, 9> sf;
      for (int i2 = 0; i2 < 3; ++i2) {
        for (int j = 0; j < 3; ++j) sf[static_cast<size_t>(3 * i2 + j)] = s[i2][j];
      }
      if (body.S_ref) {
        double diff = 0.0, ref = 0.0;
        for (size_t j = 0; j < 9; ++j) {
          diff += (sf[j] - (*body.S_ref)[j]) * (sf[j] - (*body.S_ref)[j]);
          ref += (*body.S_ref)[j] * (*body.S_ref)[j];
        }
        if (!(std::sqrt(diff) <= 1e-6 * std::sqrt(ref))) {
          throw MotionError("GEO-SCALEANIM", "body " + std::to_string(b) + " changes scale at t = " +
                                                 std::to_string(out.t_ns[i]) + " ns");
        }
      }
      const M3 sym = {{{s[0][0], 0.5 * (s[0][1] + s[1][0]), 0.5 * (s[0][2] + s[2][0])},
                       {0.5 * (s[0][1] + s[1][0]), s[1][1], 0.5 * (s[1][2] + s[2][1])},
                       {0.5 * (s[0][2] + s[2][0]), 0.5 * (s[1][2] + s[2][1]), s[2][2]}}};
      const double dth = 3.0 * half_ulp(amax) / min_abs_eigenvalue(sym);
      const rot::V t{static_cast<double>(m[3]), static_cast<double>(m[7]), static_cast<double>(m[11])};
      const rot::V dt{half_ulp(m[3]), half_ulp(m[7]), half_ulp(m[11])};
      out.q[k] = q;
      out.p[k] = scene.scale_length * t + rotate(q, body.centre_m);
      out.S[k] = sf;
      out.dtheta_round[k] = dth;
      out.dp_round[k] = scene.scale_length * norm(dt) + norm(body.centre_m) * dth;
    }
  }
  return out;
}

}  // namespace balla::scene::motion
