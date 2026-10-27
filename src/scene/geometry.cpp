// Contains C++ translations of NumPy (the order of np.cross, np.unique(axis=0), np.bincount), under its BSD licence:
// see THIRD_PARTY_NOTICES.md, "Ported code".
#include "scene/geometry.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "scene/errors.h"
#include "scene/npla.h"
#include "scene/npo.h"

namespace balla::scene::geo {

namespace {

using npo::View2;

V row3(const std::vector<double>& a, size_t i) { return {a[3 * i], a[3 * i + 1], a[3 * i + 2]}; }

// NumPy's sum over axis 0 of an (n, 3) array: the rows added in order to a zeroed output.
V rowseq_sum(std::span<const V> x) {
  V s{0.0, 0.0, 0.0};
  for (const V& r : x) s = {s.x + r.x, s.y + r.y, s.z + r.z};
  return s;
}

std::vector<double> positions64(const Mesh& m) { return std::vector<double>(m.positions.begin(), m.positions.end()); }

// np.unique(rows, axis=0, return_inverse=True)'s inverse: each row's index among the sorted distinct rows
// (lexicographic, signed).
template <size_t K>
std::vector<int64_t> unique_rows_inverse(const std::vector<std::array<int64_t, K>>& rows, std::vector<size_t>* first) {
  std::vector<size_t> order(rows.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(), [&rows](size_t a, size_t b) { return rows[a] < rows[b]; });
  std::vector<int64_t> inv(rows.size());
  int64_t u = -1;
  for (size_t k = 0; k < order.size(); ++k) {
    if (k == 0 || rows[order[k]] != rows[order[k - 1]]) {
      ++u;
      if (first) first->push_back(order[k]);
    }
    inv[order[k]] = u;
  }
  return inv;
}

}  // namespace

M4 identity4() {
  M4 t{};
  t[0] = t[5] = t[10] = t[15] = 1.0;
  return t;
}

M4 make_transform(const M& r, V t) {
  M4 out{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) out[static_cast<size_t>(4 * i + j)] = r.m[i][j];
  }
  out[3] = t.x;
  out[7] = t.y;
  out[11] = t.z;
  out[15] = 1.0;
  return out;
}

M4 pose_to_transform(Q q, V p) { return make_transform(rot::quat_to_matrix(q), p); }

M rotation_of(const M4& t) {
  M r;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) r.m[i][j] = t[static_cast<size_t>(4 * i + j)];
  }
  return r;
}

V translation_of(const M4& t) { return {t[3], t[7], t[11]}; }

std::pair<Q, V> transform_to_pose(const M4& t) {
  return {rot::canonical_quat(rot::sp_from_matrix(rotation_of(t))), translation_of(t)};
}

M4 invert_transform(const M4& t) {
  M rt;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) rt.m[i][j] = t[static_cast<size_t>(4 * j + i)];
  }
  const V p = translation_of(t);
  double v[3];
  for (int i = 0; i < 3; ++i) {
    // einsum's strided outstride0 kernel: a sequential sum from 0, added to the zeroed output.
    double acc = 0.0;
    acc += rt.m[i][0] * p.x;
    acc += rt.m[i][1] * p.y;
    acc += rt.m[i][2] * p.z;
    v[i] = -(acc + 0.0);
  }
  return make_transform(rt, {v[0], v[1], v[2]});
}

M4 matmul(const M4& a, const M4& b) {
  M4 c;
  npo::matmul(npo::c_view(a.data(), 4, 4), npo::c_view(b.data(), 4, 4), c.data());
  return c;
}

Mesh transform_mesh(const Mesh& mesh, const M4& t) {
  const View2 a{t.data(), 3, 3, 4, 1};
  const size_t nv = mesh.num_vertices();
  const std::vector<double> p64 = positions64(mesh);
  const std::vector<double> r = npo::matmul(npo::c_view(p64.data(), static_cast<int64_t>(nv), 3), a.t());
  Mesh out;
  out.positions.resize(3 * nv);
  for (size_t i = 0; i < nv; ++i) {
    for (size_t j = 0; j < 3; ++j) out.positions[3 * i + j] = static_cast<float>(r[3 * i + j] + t[4 * j + 3]);
  }
  double a3[9];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) a3[3 * i + j] = t[static_cast<size_t>(4 * i + j)];
  }
  out.indices = mesh.indices;
  if (npo::det(a3, 3) < 0) {
    for (size_t k = 0; k < mesh.num_triangles(); ++k) std::swap(out.indices[3 * k + 1], out.indices[3 * k + 2]);
  }
  if (mesh.has_normals) {
    const std::vector<double> nt = npo::inv(a3, 3);   // inv(a).T.T: inv(a) as stored
    const std::vector<double> n64(mesh.normals.begin(), mesh.normals.end());
    std::vector<double> n = npo::matmul(npo::c_view(n64.data(), static_cast<int64_t>(nv), 3), npo::c_view(nt.data(), 3, 3));
    out.normals.resize(3 * nv);
    for (size_t i = 0; i < nv; ++i) {
      const double d = npo::maximum(rot::norm_axis(row3(n, i)), 1e-300);
      for (size_t j = 0; j < 3; ++j) out.normals[3 * i + j] = static_cast<float>(n[3 * i + j] / d);
    }
    out.has_normals = true;
  }
  out.uvs = mesh.uvs;
  out.has_uvs = mesh.has_uvs;
  out.material_ids = mesh.material_ids;
  out.has_material_ids = mesh.has_material_ids;
  return out;
}

Mesh merge_meshes(const std::vector<std::pair<const Mesh*, int>>& parts) {
  if (parts.empty()) throw std::invalid_argument("nothing to merge");
  bool has_normals = true, has_uvs = false;
  for (const auto& [m, n] : parts) {
    has_normals = has_normals && m->has_normals;
    has_uvs = has_uvs || m->has_uvs;
  }
  Mesh out;
  int64_t offset = 0, mats = 0;
  std::vector<int64_t> mids;
  for (const auto& [m, count] : parts) {
    out.positions.insert(out.positions.end(), m->positions.begin(), m->positions.end());
    for (const uint32_t i : m->indices) out.indices.push_back(static_cast<uint32_t>(static_cast<int64_t>(i) + offset));
    if (has_normals) out.normals.insert(out.normals.end(), m->normals.begin(), m->normals.end());
    if (has_uvs) {
      if (m->has_uvs) {
        out.uvs.insert(out.uvs.end(), m->uvs.begin(), m->uvs.end());
      } else {
        out.uvs.insert(out.uvs.end(), 2 * m->num_vertices(), 0.0f);
      }
    }
    for (size_t k = 0; k < m->num_triangles(); ++k) {
      mids.push_back((m->has_material_ids ? static_cast<int64_t>(m->material_ids[k]) : 0) + mats);
    }
    mats += count;
    offset += static_cast<int64_t>(m->num_vertices());
  }
  if (mats > 0xFFFF) throw std::invalid_argument("more than 65535 materials");
  out.has_normals = has_normals;
  out.has_uvs = has_uvs;
  if (mats != 1) {
    out.material_ids.resize(mids.size());
    for (size_t k = 0; k < mids.size(); ++k) out.material_ids[k] = static_cast<uint16_t>(mids[k]);
    out.has_material_ids = true;
  }
  return out;
}

std::vector<double> triangle_areas(const Mesh& mesh) {
  const std::vector<double> p = positions64(mesh);
  const size_t nt = mesh.num_triangles();
  std::vector<double> a(nt);
  for (size_t k = 0; k < nt; ++k) {
    const V p0 = row3(p, mesh.indices[3 * k]), p1 = row3(p, mesh.indices[3 * k + 1]), p2 = row3(p, mesh.indices[3 * k + 2]);
    a[k] = 0.5 * rot::norm_axis(rot::cross(p1 - p0, p2 - p0));
  }
  return a;
}

V area_centroid(const Mesh& mesh) {
  const std::vector<double> p = positions64(mesh);
  const std::vector<double> a = triangle_areas(mesh);
  std::vector<V> w(a.size());
  for (size_t k = 0; k < a.size(); ++k) {
    const V p0 = row3(p, mesh.indices[3 * k]), p1 = row3(p, mesh.indices[3 * k + 1]), p2 = row3(p, mesh.indices[3 * k + 2]);
    // p[indices].mean(axis=1): the three vertices summed from 0 in order, over 3.
    const V c{(((0.0 + p0.x) + p1.x) + p2.x) / 3.0, (((0.0 + p0.y) + p1.y) + p2.y) / 3.0,
              (((0.0 + p0.z) + p1.z) + p2.z) / 3.0};
    w[k] = {a[k] * c.x, a[k] * c.y, a[k] * c.z};
  }
  const V s = rowseq_sum(w);
  const double total = npo::sum(a);
  return {s.x / total, s.y / total, s.z / total};
}

std::pair<V, double> bounding_sphere(const Mesh& mesh) {
  const std::vector<double> p = positions64(mesh);
  const size_t nv = mesh.num_vertices();
  if (nv == 0) throw std::invalid_argument("zero-size array to reduction operation minimum which has no identity");
  double lo[3], hi[3];
  for (size_t j = 0; j < 3; ++j) {
    std::vector<double> col(nv);
    for (size_t i = 0; i < nv; ++i) col[i] = p[3 * i + j];
    lo[j] = npo::min(col);
    hi[j] = npo::max(col);
  }
  const V c{0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])};
  std::vector<double> d(nv);
  for (size_t i = 0; i < nv; ++i) d[i] = rot::norm_axis(row3(p, i) - c);
  return {c, npo::max(d)};
}

Mesh icosphere(int level, double radius) {
  const double g = (1.0 + npo::py_pow(5.0, 0.5)) / 2.0;
  std::vector<V> v = {{-1, g, 0}, {1, g, 0}, {-1, -g, 0}, {1, -g, 0}, {0, -1, g}, {0, 1, g},
                      {0, -1, -g}, {0, 1, -g}, {g, 0, -1}, {g, 0, 1}, {-g, 0, -1}, {-g, 0, 1}};
  std::vector<std::array<int64_t, 3>> f = {{0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
                                           {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                                           {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
                                           {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1}};
  auto normalized = [](V x) {
    const double n = rot::norm_axis(x);
    return V{x.x / n, x.y / n, x.z / n};
  };
  for (V& x : v) x = normalized(x);
  for (int l = 0; l < level; ++l) {
    const size_t nf = f.size();
    std::vector<std::array<int64_t, 2>> edges(3 * nf);
    for (size_t k = 0; k < nf; ++k) {
      const int64_t pairs[3][2] = {{f[k][0], f[k][1]}, {f[k][1], f[k][2]}, {f[k][2], f[k][0]}};
      for (int e = 0; e < 3; ++e) {
        edges[static_cast<size_t>(e) * nf + k] = {std::min(pairs[e][0], pairs[e][1]), std::max(pairs[e][0], pairs[e][1])};
      }
    }
    std::vector<size_t> first;
    const std::vector<int64_t> inv = unique_rows_inverse(edges, &first);
    const auto base = static_cast<int64_t>(v.size());
    std::vector<V> mid(first.size());
    for (size_t u = 0; u < first.size(); ++u) {
      const auto& e = edges[first[u]];
      mid[u] = normalized(v[static_cast<size_t>(e[0])] + v[static_cast<size_t>(e[1])]);
    }
    v.insert(v.end(), mid.begin(), mid.end());
    std::vector<std::array<int64_t, 3>> g4(4 * nf);
    for (size_t k = 0; k < nf; ++k) {
      const int64_t a = f[k][0], b = f[k][1], c = f[k][2];
      const int64_t m01 = base + inv[k], m12 = base + inv[nf + k], m20 = base + inv[2 * nf + k];
      g4[k] = {a, m01, m20};
      g4[nf + k] = {b, m12, m01};
      g4[2 * nf + k] = {c, m20, m12};
      g4[3 * nf + k] = {m01, m12, m20};
    }
    f = std::move(g4);
  }
  Mesh out;
  out.positions.resize(3 * v.size());
  out.normals.resize(3 * v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    const double xyz[3] = {v[i].x, v[i].y, v[i].z};
    for (size_t j = 0; j < 3; ++j) {
      out.positions[3 * i + j] = static_cast<float>(xyz[j] * radius);
      out.normals[3 * i + j] = static_cast<float>(xyz[j]);
    }
  }
  out.has_normals = true;
  out.indices.resize(3 * f.size());
  for (size_t k = 0; k < f.size(); ++k) {
    for (size_t j = 0; j < 3; ++j) out.indices[3 * k + j] = static_cast<uint32_t>(f[k][j]);
  }
  return out;
}

V dominant_normal(const Mesh& mesh) {
  const std::vector<double> p = positions64(mesh);
  const size_t nt = mesh.num_triangles();
  std::vector<V> n(nt);
  std::vector<double> area(nt);
  std::vector<std::array<int64_t, 3>> bins(nt);
  for (size_t k = 0; k < nt; ++k) {
    const V p0 = row3(p, mesh.indices[3 * k]), p1 = row3(p, mesh.indices[3 * k + 1]), p2 = row3(p, mesh.indices[3 * k + 2]);
    const V c = rot::cross(p1 - p0, p2 - p0);
    area[k] = rot::norm_axis(c);
    const double d = npo::maximum(area[k], 1e-300);
    n[k] = {c.x / d, c.y / d, c.z / d};
    bins[k] = {static_cast<int64_t>(std::nearbyint(n[k].x * 30.0)), static_cast<int64_t>(std::nearbyint(n[k].y * 30.0)),
               static_cast<int64_t>(std::nearbyint(n[k].z * 30.0))};
  }
  const std::vector<int64_t> inv = unique_rows_inverse(bins, nullptr);
  const int64_t nbins = inv.empty() ? 0 : *std::max_element(inv.begin(), inv.end()) + 1;
  std::vector<double> total(static_cast<size_t>(nbins), 0.0);
  for (size_t k = 0; k < nt; ++k) total[static_cast<size_t>(inv[k])] += area[k];
  const int64_t best = npo::argmax(total);
  std::vector<V> w;
  for (size_t k = 0; k < nt; ++k) {
    if (inv[k] == best) w.push_back({area[k] * n[k].x, area[k] * n[k].y, area[k] * n[k].z});
  }
  const V up = rowseq_sum(w);
  const double norm = rot::norm_dot(up);
  return {up.x / norm, up.y / norm, up.z / norm};
}

Kabsch kabsch(std::span<const V> a, std::span<const V> b) {
  const size_t n = a.size();
  if (b.size() != n || n < 3) throw std::invalid_argument("kabsch needs two (N >= 3, 3) arrays of equal shape");
  const V sa = rowseq_sum(a), sb = rowseq_sum(b);
  const auto nd = static_cast<double>(n);
  const V ca{sa.x / nd, sa.y / nd, sa.z / nd}, cb{sb.x / nd, sb.y / nd, sb.z / nd};
  std::vector<double> ac(3 * n), bc(3 * n);
  for (size_t i = 0; i < n; ++i) {
    const V x = a[i] - ca, y = b[i] - cb;
    ac[3 * i] = x.x, ac[3 * i + 1] = x.y, ac[3 * i + 2] = x.z;
    bc[3 * i] = y.x, bc[3 * i + 1] = y.y, bc[3 * i + 2] = y.z;
  }
  const auto ni = static_cast<int64_t>(n);
  const std::vector<double> h = npo::matmul(npo::c_view(ac.data(), ni, 3).t(), npo::c_view(bc.data(), ni, 3));
  double u[9], s[3], vt[9];
  npo::svd(h.data(), 3, 3, u, s, vt);
  const View2 vtt = npo::c_view(vt, 3, 3).t(), ut = npo::c_view(u, 3, 3).t();
  const std::vector<double> vu = npo::matmul(vtt, ut);
  const double dt = npo::det(vu.data(), 3);
  const double d = dt > 0 ? 1.0 : dt < 0 ? -1.0 : dt == 0 ? 0.0 : dt;   // np.sign (NaN stays NaN)
  const double diag[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, d};
  const std::vector<double> vd = npo::matmul(vtt, npo::c_view(diag, 3, 3));
  const std::vector<double> r = npo::matmul(npo::c_view(vd.data(), 3, 3), ut);
  const double cav[3] = {ca.x, ca.y, ca.z};
  const std::vector<double> rc = npo::matmul(npo::c_view(r.data(), 3, 3), View2{cav, 3, 1, 1, 1});
  Kabsch out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) out.r.m[i][j] = r[static_cast<size_t>(3 * i + j)];
  }
  out.t = {cb.x - rc[0], cb.y - rc[1], cb.z - rc[2]};
  std::vector<double> a64(3 * n);
  for (size_t i = 0; i < n; ++i) a64[3 * i] = a[i].x, a64[3 * i + 1] = a[i].y, a64[3 * i + 2] = a[i].z;
  const std::vector<double> ar = npo::matmul(npo::c_view(a64.data(), ni, 3), npo::c_view(r.data(), 3, 3).t());
  out.residuals.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const V e{(ar[3 * i] + out.t.x) - b[i].x, (ar[3 * i + 1] + out.t.y) - b[i].y, (ar[3 * i + 2] + out.t.z) - b[i].z};
    out.residuals[i] = rot::norm_axis(e);
  }
  return out;
}

// --- mesh validation ------------------------------------------------------------------------------------------------

namespace {

V at3(std::span<const double> p, int64_t i) {
  const auto k = static_cast<size_t>(3 * i);
  return {p[k], p[k + 1], p[k + 2]};
}

// np.sum(a * b, axis=1) of one row.
double rowdot(V a, V b) { return 0.0 + (((0.0 + a.x * b.x) + a.y * b.y) + a.z * b.z); }

// np.clip(x, -1, 1) (NaN stays NaN).
double clip1(double x) { return x < -1.0 ? -1.0 : x > 1.0 ? 1.0 : x; }

}  // namespace

std::vector<float> vertex_normals(std::span<const double> p, std::span<const int64_t> idx) {
  const size_t nv = p.size() / 3, nt = idx.size() / 3;
  std::vector<V> fn(nt);
  for (size_t t = 0; t < nt; ++t) {
    const V p0 = at3(p, idx[3 * t]), p1 = at3(p, idx[3 * t + 1]), p2 = at3(p, idx[3 * t + 2]);
    const V c = rot::cross(p1 - p0, p2 - p0);
    const double d = npo::maximum(rot::norm_axis(c), 1e-300);
    fn[t] = {c.x / d, c.y / d, c.z / d};
  }
  std::vector<double> out(3 * nv, 0.0);
  std::vector<double> bc(3 * nv);
  for (int k = 0; k < 3; ++k) {
    std::fill(bc.begin(), bc.end(), 0.0);
    for (size_t t = 0; t < nt; ++t) {
      const V pk = at3(p, idx[3 * t + static_cast<size_t>(k)]);
      const V a = at3(p, idx[3 * t + static_cast<size_t>((k + 1) % 3)]) - pk;
      const V b = at3(p, idx[3 * t + static_cast<size_t>((k + 2) % 3)]) - pk;
      const double cosang = rowdot(a, b) / npo::maximum(rot::norm_axis(a) * rot::norm_axis(b), 1e-300);
      const double w = std::acos(clip1(cosang));
      const auto v = static_cast<size_t>(idx[3 * t + static_cast<size_t>(k)]);
      bc[3 * v] += w * fn[t].x;
      bc[3 * v + 1] += w * fn[t].y;
      bc[3 * v + 2] += w * fn[t].z;
    }
    for (size_t i = 0; i < out.size(); ++i) out[i] = out[i] + bc[i];
  }
  std::vector<float> n(3 * nv);
  for (size_t i = 0; i < nv; ++i) {
    const V o{out[3 * i], out[3 * i + 1], out[3 * i + 2]};
    const double len = rot::norm_axis(o);
    const double d = npo::maximum(len, 1e-300);
    n[3 * i] = static_cast<float>(len > 0 ? o.x / d : 0.0);
    n[3 * i + 1] = static_cast<float>(len > 0 ? o.y / d : 0.0);
    n[3 * i + 2] = static_cast<float>(len > 0 ? o.z / d : 0.0);
  }
  return n;
}

std::vector<uint8_t> kept_faces(std::span<const double> p, std::span<const int64_t> idx, double rel_eps,
                                int64_t* degenerate, int64_t* duplicate) {
  const size_t nv = p.size() / 3, nt = idx.size() / 3;
  if (nv == 0) throw std::invalid_argument("zero-size array to reduction operation maximum which has no identity");
  double hi[3], lo[3];
  for (size_t j = 0; j < 3; ++j) {
    std::vector<double> col(nv);
    for (size_t i = 0; i < nv; ++i) col[i] = p[3 * i + j];
    hi[j] = npo::max(col);
    lo[j] = npo::min(col);
  }
  const double diag = rot::norm_dot({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
  const double limit = npo::py_pow(rel_eps * diag, 2.0);
  std::vector<uint8_t> keep(nt);
  std::vector<size_t> kept;
  for (size_t t = 0; t < nt; ++t) {
    const int64_t a = idx[3 * t], b = idx[3 * t + 1], c = idx[3 * t + 2];
    const bool repeated = a == b || b == c || a == c;
    const V p0 = at3(p, a), p1 = at3(p, b), p2 = at3(p, c);
    const double area = 0.5 * rot::norm_axis(rot::cross(p1 - p0, p2 - p0));
    keep[t] = !(repeated || area < limit);
    if (keep[t]) kept.push_back(t);
  }
  // np.unique(np.sort(idx[keep], axis=1), axis=0, return_index=True): the first of every vertex set (stable sort).
  std::vector<std::array<int64_t, 3>> rows(kept.size());
  for (size_t k = 0; k < kept.size(); ++k) {
    rows[k] = {idx[3 * kept[k]], idx[3 * kept[k] + 1], idx[3 * kept[k] + 2]};
    std::sort(rows[k].begin(), rows[k].end());
  }
  std::vector<size_t> order(kept.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(), [&rows](size_t x, size_t y) { return rows[x] < rows[y]; });
  std::vector<uint8_t> mask(nt, 0);
  int64_t unique = 0;
  for (size_t k = 0; k < order.size(); ++k) {
    if (k == 0 || rows[order[k]] != rows[order[k - 1]]) {
      mask[kept[order[k]]] = 1;
      ++unique;
    }
  }
  *degenerate = static_cast<int64_t>(nt - kept.size());
  *duplicate = static_cast<int64_t>(kept.size()) - unique;
  return mask;
}

std::vector<float> welded_normals(std::span<const double> p, std::span<const int64_t> idx, double rel_eps) {
  int64_t deg = 0, dup = 0;
  const std::vector<uint8_t> keep = kept_faces(p, idx, rel_eps, &deg, &dup);
  std::vector<int64_t> kept;
  for (size_t t = 0; t < keep.size(); ++t) {
    if (keep[t]) kept.insert(kept.end(), idx.begin() + static_cast<std::ptrdiff_t>(3 * t),
                             idx.begin() + static_cast<std::ptrdiff_t>(3 * t + 3));
  }
  return vertex_normals(p, kept);
}

Mesh validate_mesh(const Mesh& mesh, ValidationReport& report, double rel_eps) {
  auto finite = [](const std::vector<float>& a) {
    return std::all_of(a.begin(), a.end(), [](float x) { return std::isfinite(x); });
  };
  if (!finite(mesh.positions)) throw AssetError("AST-V01: mesh positions contain NaN or Inf");
  if (mesh.has_normals && !finite(mesh.normals)) throw AssetError("AST-V01: mesh normals contain NaN or Inf");
  if (mesh.has_uvs && !finite(mesh.uvs)) throw AssetError("AST-V01: mesh uvs contain NaN or Inf");
  const size_t nv = mesh.num_vertices(), nt = mesh.num_triangles();
  if (nt == 0) throw AssetError("mesh has no triangles");
  std::vector<int64_t> idx(mesh.indices.begin(), mesh.indices.end());
  for (const int64_t i : idx) {
    if (i < 0 || i >= static_cast<int64_t>(nv)) throw AssetError("triangle index out of range");
  }
  report = {};
  report.input_vertices = static_cast<int64_t>(nv);
  report.input_triangles = static_cast<int64_t>(nt);
  const std::vector<double> p64(mesh.positions.begin(), mesh.positions.end());
  const std::vector<uint8_t> keep = kept_faces(p64, idx, rel_eps, &report.degenerate_removed, &report.duplicate_removed);
  std::vector<uint8_t> used(nv, 0);
  for (size_t t = 0; t < nt; ++t) {
    if (keep[t]) {
      for (size_t k = 0; k < 3; ++k) used[static_cast<size_t>(idx[3 * t + k])] = 1;
    }
  }
  std::vector<int64_t> remap(nv);
  int64_t next = 0;
  for (size_t i = 0; i < nv; ++i) {
    remap[i] = next - 1 + used[i];
    next += used[i];
    if (!used[i]) ++report.unreferenced_removed;
  }
  Mesh out;
  for (size_t i = 0; i < nv; ++i) {
    if (!used[i]) continue;
    out.positions.insert(out.positions.end(), mesh.positions.begin() + static_cast<std::ptrdiff_t>(3 * i),
                         mesh.positions.begin() + static_cast<std::ptrdiff_t>(3 * i + 3));
    if (mesh.has_normals) {
      out.normals.insert(out.normals.end(), mesh.normals.begin() + static_cast<std::ptrdiff_t>(3 * i),
                         mesh.normals.begin() + static_cast<std::ptrdiff_t>(3 * i + 3));
    }
    if (mesh.has_uvs) {
      out.uvs.insert(out.uvs.end(), mesh.uvs.begin() + static_cast<std::ptrdiff_t>(2 * i),
                     mesh.uvs.begin() + static_cast<std::ptrdiff_t>(2 * i + 2));
    }
  }
  out.has_normals = mesh.has_normals;
  out.has_uvs = mesh.has_uvs;
  for (size_t t = 0; t < nt; ++t) {
    if (!keep[t]) continue;
    for (size_t k = 0; k < 3; ++k) out.indices.push_back(static_cast<uint32_t>(remap[static_cast<size_t>(idx[3 * t + k])]));
    if (mesh.has_material_ids) out.material_ids.push_back(mesh.material_ids[t]);
  }
  out.has_material_ids = mesh.has_material_ids;
  if (!out.has_normals) {
    const std::vector<double> q(out.positions.begin(), out.positions.end());
    const std::vector<int64_t> oi(out.indices.begin(), out.indices.end());
    out.normals = vertex_normals(q, oi);
    out.has_normals = true;
    report.normals = "computed (angle-weighted)";
  } else {
    // In float32, as NumPy computes np.linalg.norm of a float32 array and the division.
    for (size_t i = 0; i < out.num_vertices(); ++i) {
      float* n = &out.normals[3 * i];
      const float len = std::sqrt(((0.0f + n[0] * n[0]) + n[1] * n[1]) + n[2] * n[2]);
      const float d = (len >= 1e-30f || std::isnan(len)) ? len : 1e-30f;
      for (int k = 0; k < 3; ++k) n[k] = len > 0 ? n[k] / d : 0.0f;
    }
    report.normals = "given by the importer";
  }
  report.vertices = static_cast<int64_t>(out.num_vertices());
  report.triangles = static_cast<int64_t>(out.num_triangles());
  return out;
}

}  // namespace balla::scene::geo
