// Contains C++ translations of SciPy 1.18 (Rotation: _rotation_cy.pyx; Slerp: _rotation.py) and NumPy (its einsum
// kernels' order), under their BSD licences: see THIRD_PARTY_NOTICES.md, "Ported code".
#include "scene/rotation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "core/math/so3.h"
#include "scene/blas.h"
#include "scene/npo.h"

namespace balla::scene::rot {

namespace {

double sq(double x) { return x * x; }

}  // namespace

Q sp_normalize(Q q) {
  const double n = std::sqrt(((q.x * q.x + q.y * q.y) + q.z * q.z) + q.w * q.w);
  if (n == 0) throw std::invalid_argument("Found zero norm quaternions in `quat`.");
  return {q.w / n, q.x / n, q.y / n, q.z / n};
}

M sp_as_matrix(Q q) {
  const double x = q.x, y = q.y, z = q.z, w = q.w;
  const double x2 = x * x, y2 = y * y, z2 = z * z, w2 = w * w;
  const double xy = x * y, zw = z * w, xz = x * z, yw = y * w, yz = y * z, xw = x * w;
  M m;
  m.m[0][0] = ((x2 - y2) - z2) + w2;
  m.m[0][1] = 2 * (xy - zw);
  m.m[0][2] = 2 * (xz + yw);
  m.m[1][0] = 2 * (xy + zw);
  m.m[1][1] = ((-x2 + y2) - z2) + w2;
  m.m[1][2] = 2 * (yz - xw);
  m.m[2][0] = 2 * (xz - yw);
  m.m[2][1] = 2 * (yz + xw);
  m.m[2][2] = ((-x2 - y2) + z2) + w2;
  return m;
}

Q sp_compose(Q p, Q q) {
  const double c0 = p.y * q.z - p.z * q.y;
  const double c1 = p.z * q.x - p.x * q.z;
  const double c2 = p.x * q.y - p.y * q.x;
  const Q r{((p.w * q.w - p.x * q.x) - p.y * q.y) - p.z * q.z, (p.w * q.x + q.w * p.x) + c0,
            (p.w * q.y + q.w * p.y) + c1, (p.w * q.z + q.w * p.z) + c2};
  return sp_normalize(r);
}

V sp_as_rotvec(Q q) {
  const bool flip =
      q.w < 0 || (q.w == 0 && (q.x < 0 || (q.x == 0 && (q.y < 0 || (q.y == 0 && q.z < 0)))));
  if (flip) q = {-q.w, -q.x, -q.y, -q.z};
  const double an = std::sqrt((q.x * q.x + q.y * q.y) + q.z * q.z);
  const double angle = 2 * std::atan2(an, q.w);
  double scale;
  if (angle <= 1e-3) {
    const double a2 = angle * angle;
    scale = (2 + a2 / 12) + ((7 * a2) * a2) / 2880;
  } else {
    scale = angle / std::sin(angle / 2);
  }
  return {scale * q.x, scale * q.y, scale * q.z};
}

Q sp_from_rotvec(V v) {
  const double angle = std::sqrt((v.x * v.x + v.y * v.y) + v.z * v.z);
  double scale;
  if (angle <= 1e-3) {
    const double a2 = angle * angle;
    scale = (0.5 - a2 / 48) + (a2 * a2) / 3840;
  } else {
    scale = std::sin(angle / 2) / angle;
  }
  return {std::cos(angle / 2), scale * v.x, scale * v.y, scale * v.z};
}

Q sp_from_matrix(const M& m) {
  const auto& a = m.m;
  const double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                     a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                     a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  if (!(det > 0)) throw std::invalid_argument("Non-positive determinant (left-handed or null coordinate frame)");
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double g = (a[i][0] * a[j][0] + a[i][1] * a[j][1]) + a[i][2] * a[j][2];
      if (std::abs(g - (i == j ? 1.0 : 0.0)) > (i == j ? 1e-5 : 1e-12)) {
        throw std::invalid_argument("Rotation.from_matrix: not orthogonal (the SVD projection is not ported)");
      }
    }
  }
  const double tr = (a[0][0] + a[1][1]) + a[2][2];
  const double dec[4] = {a[0][0], a[1][1], a[2][2], tr};
  int choice = 0;
  for (int i = 1; i < 4; ++i) {
    if (dec[i] > dec[choice]) choice = i;
  }
  double qq[4];  // SciPy's (x, y, z, w)
  if (choice != 3) {
    const int i = choice, j = (i + 1) % 3, k = (j + 1) % 3;
    qq[i] = (1 - tr) + 2 * a[i][i];
    qq[j] = a[j][i] + a[i][j];
    qq[k] = a[k][i] + a[i][k];
    qq[3] = a[k][j] - a[j][k];
  } else {
    qq[0] = a[2][1] - a[1][2];
    qq[1] = a[0][2] - a[2][0];
    qq[2] = a[1][0] - a[0][1];
    qq[3] = 1 + tr;
  }
  return sp_normalize({qq[3], qq[0], qq[1], qq[2]});
}

Q sp_mean(std::span<const Q> q) {
  const int n = static_cast<int>(q.size());
  if (n == 0) throw std::invalid_argument("Mean of an empty rotation set is undefined.");
  std::vector<double> w(4 * q.size()), b(4 * q.size());
  for (size_t i = 0; i < q.size(); ++i) {
    const double row[4] = {q[i].x, q[i].y, q[i].z, q[i].w};
    for (int c = 0; c < 4; ++c) {
      w[4 * i + static_cast<size_t>(c)] = 1.0 * row[c];
      b[4 * i + static_cast<size_t>(c)] = row[c];
    }
  }
  double k[16], ev[4], v[16];
  cblas_dgemm(blas::kRowMajor, blas::kTrans, blas::kNoTrans, 4, 4, n, 1.0, w.data(), 4, b.data(), 4, 0.0, k, 4);
  npo::eigh4(k, ev, v);
  return {v[4 * 3 + 3], v[4 * 0 + 3], v[4 * 1 + 3], v[4 * 2 + 3]};
}

Q normalize_quat(Q q) {
  const double n = std::sqrt(0.0 + ((((0.0 + q.w * q.w) + q.x * q.x) + q.y * q.y) + q.z * q.z));
  return {q.w / n, q.x / n, q.y / n, q.z / n};
}

Q canonical_quat(Q q) {
  q = normalize_quat(q);
  return q.w < 0.0 ? Q{-q.w, -q.x, -q.y, -q.z} : q;
}

Q quat_multiply(Q a, Q b) {
  return {((a.w * b.w - a.x * b.x) - a.y * b.y) - a.z * b.z, ((a.w * b.x + a.x * b.w) + a.y * b.z) - a.z * b.y,
          ((a.w * b.y - a.x * b.z) + a.y * b.w) + a.z * b.x, ((a.w * b.z + a.x * b.y) - a.y * b.x) + a.z * b.w};
}

Q quat_conjugate(Q q) { return {q.w * 1.0, q.x * -1.0, q.y * -1.0, q.z * -1.0}; }

M quat_to_matrix(Q q) { return sp_as_matrix(sp_normalize(q)); }

V quat_rotate(Q q, V v) {
  const M r = quat_to_matrix(q);
  double out[3];
  for (int i = 0; i < 3; ++i) {
    double lane0 = r.m[i][0] * v.x + 0.0;
    double lane1 = r.m[i][1] * v.y + 0.0;
    lane0 = r.m[i][2] * v.z + lane0;
    lane1 = 0.0 * 0.0 + lane1;
    out[i] = 0.0 + (lane0 + lane1);
  }
  return {out[0], out[1], out[2]};
}

V relative_rotvec(Q a, Q b) { return so3_log(quat_multiply(quat_conjugate(a), b)); }

double rotation_angle(Q a, Q b) {
  const Q na = normalize_quat(a), nb = normalize_quat(b);
  const double dot =
      std::abs(0.0 + ((((0.0 + na.w * nb.w) + na.x * nb.x) + na.y * nb.y) + na.z * nb.z));
  const Q rel = quat_multiply(quat_conjugate(a), b);
  const double n = std::sqrt(0.0 + (((0.0 + rel.x * rel.x) + rel.y * rel.y) + rel.z * rel.z));
  return 2.0 * std::atan2(n, std::min(dot, 1.0));
}

Q so3_exp(V phi) {
  if (!(dot(phi, phi) <= 4e12)) throw std::invalid_argument("so3_exp: need finite |phi| <= 2e6 rad (the sincos domain)");
  return so3::exp(phi);
}

V so3_log(Q q) { return so3::log(q); }

V cross(V a, V b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }

double norm_axis(V v) { return std::sqrt(0.0 + (((0.0 + v.x * v.x) + v.y * v.y) + v.z * v.z)); }

double norm_dot(V v) {
  const double x[3] = {v.x, v.y, v.z};
  return std::sqrt(blas::dot(3, x, x));
}

namespace {

M hat(V v) {
  M k{};
  k.m[0][1] = -v.z;
  k.m[0][2] = v.y;
  k.m[1][0] = v.z;
  k.m[1][2] = -v.x;
  k.m[2][0] = -v.y;
  k.m[2][1] = v.x;
  return k;
}

M kk(const M& k) {
  M out;
  blas::matmul3(&k.m[0][0], false, &k.m[0][0], false, &out.m[0][0]);
  return out;
}

}  // namespace

M right_jacobian(V phi) {
  const double theta2 = 0.0 + (((0.0 + phi.x * phi.x) + phi.y * phi.y) + phi.z * phi.z);
  const bool small = theta2 < 1e-8;
  const double safe = small ? 1.0 : std::sqrt(theta2);
  const double a = small ? 0.5 - theta2 / 24.0 : (1.0 - std::cos(safe)) / sq(safe);
  const double b = small ? 1.0 / 6.0 - theta2 / 120.0 : (safe - std::sin(safe)) / npo::py_pow(safe, 3.0);
  const M k = hat(phi), k2 = kk(k);
  M j;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) j.m[r][c] = ((r == c ? 1.0 : 0.0) - a * k.m[r][c]) + b * k2.m[r][c];
  }
  return j;
}

M right_jacobian_inv(V phi) {
  const double theta2 = 0.0 + (((0.0 + phi.x * phi.x) + phi.y * phi.y) + phi.z * phi.z);
  const bool small = theta2 < 1e-8;
  const double safe = small ? 1.0 : std::sqrt(theta2);
  const double c = small ? 1.0 / 12.0 + theta2 / 720.0
                         : 1.0 / sq(safe) - (1.0 + std::cos(safe)) / ((2.0 * safe) * std::sin(safe));
  const M k = hat(phi), k2 = kk(k);
  M j;
  for (int r = 0; r < 3; ++r) {
    for (int cc = 0; cc < 3; ++cc) j.m[r][cc] = ((r == cc ? 1.0 : 0.0) + 0.5 * k.m[r][cc]) + c * k2.m[r][cc];
  }
  return j;
}

Q rotation_median(std::span<const Q> q_in, int iters, double tol) {
  std::vector<Q> q(q_in.size());
  for (size_t i = 0; i < q.size(); ++i) q[i] = canonical_quat(q_in[i]);
  std::vector<Q> sp(q.size());
  for (size_t i = 0; i < q.size(); ++i) sp[i] = sp_normalize(q[i]);
  Q m = canonical_quat(sp_from_matrix(sp_as_matrix(sp_mean(sp))));
  std::vector<V> v(q.size());
  std::vector<double> w(q.size());
  for (int it = 0; it < iters; ++it) {
    for (size_t i = 0; i < q.size(); ++i) {
      v[i] = relative_rotvec(m, q[i]);
      w[i] = 1.0 / std::max(norm_axis(v[i]), 1e-15);
    }
    double s[3] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < q.size(); ++i) {
      s[0] += w[i] * v[i].x;
      s[1] += w[i] * v[i].y;
      s[2] += w[i] * v[i].z;
    }
    const double ws = npo::sum(w);
    const V step{s[0] / ws, s[1] / ws, s[2] / ws};
    m = canonical_quat(quat_multiply(m, so3_exp(step)));
    if (norm_dot(step) < tol) break;
  }
  return m;
}

V geometric_median(std::span<const V> x, int iters, double tol) {
  const size_t n = x.size();
  std::vector<double> col(n);
  double med[3];
  for (int j = 0; j < 3; ++j) {
    for (size_t i = 0; i < n; ++i) col[i] = j == 0 ? x[i].x : j == 1 ? x[i].y : x[i].z;
    med[j] = npo::median(col);
  }
  V m{med[0], med[1], med[2]};
  std::vector<double> w(n);
  for (int it = 0; it < iters; ++it) {
    for (size_t i = 0; i < n; ++i) w[i] = 1.0 / std::max(norm_axis(x[i] - m), 1e-15);
    double s[3] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < n; ++i) {
      s[0] += w[i] * x[i].x;
      s[1] += w[i] * x[i].y;
      s[2] += w[i] * x[i].z;
    }
    const double ws = npo::sum(w);
    const V m_new{s[0] / ws, s[1] / ws, s[2] / ws};
    const double nm = norm_dot(m);
    if (norm_dot(m_new - m) <= tol * (nm > 1.0 ? nm : 1.0)) return m_new;
    m = m_new;
  }
  return m;
}

std::pair<Q, V> robust_mean_pose(std::span<const Q> q, std::span<const V> p) {
  return {rotation_median(q), geometric_median(p)};
}

void align_hemisphere(std::vector<Q>& q) {
  if (q.size() <= 1) return;
  std::vector<double> flips(q.size(), 1.0);
  for (size_t i = 1; i < q.size(); ++i) {
    const Q& a = q[i];
    const Q& b = q[i - 1];
    const double d = 0.0 + ((((0.0 + a.w * b.w) + a.x * b.x) + a.y * b.y) + a.z * b.z);
    flips[i] = d < 0 ? -1.0 : 1.0;
  }
  double c = 1.0;
  for (size_t i = 0; i < q.size(); ++i) {
    c = c * flips[i];
    q[i] = {q[i].w * c, q[i].x * c, q[i].y * c, q[i].z * c};
  }
}

std::vector<Q> slerp_init(std::span<const int64_t> t_ns, std::span<const Q> q, std::span<const int64_t> at_ns) {
  const size_t n = t_ns.size();
  if (q.size() != n) throw std::invalid_argument("slerp_init: t_ns and q differ in length");
  if (n == 0) throw std::invalid_argument("slerp_init: no samples");
  if (n == 1) return std::vector<Q>(at_ns.size(), q[0]);
  std::vector<Q> rot(n);
  std::vector<double> times(n), dt(n - 1);
  for (size_t i = 0; i < n; ++i) {
    rot[i] = sp_normalize(q[i]);
    times[i] = static_cast<double>(t_ns[i]);
  }
  std::vector<V> rotvecs(n - 1);
  for (size_t i = 0; i + 1 < n; ++i) {
    dt[i] = times[i + 1] - times[i];
    if (!(dt[i] > 0)) throw std::invalid_argument("Times must be in strictly increasing order.");
    const Q inv{rot[i].w, rot[i].x * -1.0, rot[i].y * -1.0, rot[i].z * -1.0};
    rotvecs[i] = sp_as_rotvec(sp_compose(inv, rot[i + 1]));
  }
  std::vector<Q> out(at_ns.size());
  for (size_t k = 0; k < at_ns.size(); ++k) {
    const int64_t tc_i = std::clamp(at_ns[k], t_ns[0], t_ns[n - 1]);
    const auto tc = static_cast<double>(tc_i);
    auto ind = static_cast<int64_t>(std::lower_bound(times.begin(), times.end(), tc) - times.begin()) - 1;
    if (tc == times[0]) ind = 0;
    if (ind < 0 || ind > static_cast<int64_t>(n) - 2) throw std::invalid_argument("Interpolation times out of range");
    const auto i = static_cast<size_t>(ind);
    const double alpha = (tc - times[i]) / dt[i];
    const V rv{rotvecs[i].x * alpha, rotvecs[i].y * alpha, rotvecs[i].z * alpha};
    out[k] = sp_compose(rot[i], sp_from_rotvec(rv));
  }
  align_hemisphere(out);
  return out;
}

}  // namespace balla::scene::rot
