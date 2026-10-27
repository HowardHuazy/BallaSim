// origin_world, the exact static reach of the 16 m rule and the crop of static geometry (docs/design/phase3/blender.md
// §3.7, §4.1, §4.6; R2 c3: the AABB-corner check refused a disk whose vertices lie within 15.03 m).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "scene/blender/convert.h"

namespace balla::scene::blender {

namespace {

using P3 = std::array<double, 3>;

// R x + t of a row-major 4 x 4 transform.
P3 apply(const geo::M4& T, double x, double y, double z) {
  return {T[0] * x + T[1] * y + T[2] * z + T[3], T[4] * x + T[5] * y + T[6] * z + T[7],
          T[8] * x + T[9] * y + T[10] * z + T[11]};
}

double norm(const P3& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

geo::M4 shifted(geo::M4 T, const P3& origin) {
  T[3] -= origin[0];
  T[7] -= origin[1];
  T[11] -= origin[2];
  return T;
}

std::string metres(double x) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.3f", x);
  return buf;
}

}  // namespace

std::array<double, 3> origin_world(const Statics& s, std::span<const std::array<double, 3>> samples) {
  P3 lo{INFINITY, INFINITY, INFINITY}, hi{-INFINITY, -INFINITY, -INFINITY};
  const auto add = [&](const P3& p) {
    for (size_t r = 0; r < 3; ++r) {
      lo[r] = std::min(lo[r], p[r]);
      hi[r] = std::max(hi[r], p[r]);
    }
  };
  for (const P3& p : samples) add(p);
  if (samples.empty()) {
    for (const BodyStatic& b : s.bodies) {
      if (!b.T_world_static) continue;
      const std::vector<float>& x = b.mesh.positions;
      for (size_t i = 0; i < x.size(); i += 3) add(apply(*b.T_world_static, x[i], x[i + 1], x[i + 2]));
    }
  }
  if (!(lo[0] <= hi[0])) return {0.0, 0.0, 0.0};
  return {0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])};
}

std::vector<std::array<double, 3>> provisional_samples(const ExchangeIn& ex, const Statics& s) {
  std::vector<P3> out;
  const double k = ex.scale_length();
  for (const CameraIn& c : ex.cameras) {
    if (!c.simulate) continue;
    const auto& m = ex.objects.at(static_cast<size_t>(c.object)).matrix_world;
    out.push_back({k * m[3], k * m[7], k * m[11]});
  }
  for (const BodyStatic& b : s.bodies) {
    if (!b.T_world_static) out.push_back({b.T_world_export[3], b.T_world_export[7], b.T_world_export[11]});
  }
  return out;
}

double static_reach(const BodyStatic& b, const std::array<double, 3>& origin) {
  const geo::M4 T = shifted(*b.T_world_static, origin);
  double reach = 0.0;
  const std::vector<float>& x = b.mesh.positions;
  for (size_t i = 0; i < x.size(); i += 3) reach = std::max(reach, norm(apply(T, x[i], x[i + 1], x[i + 2])));
  return reach;
}

Cropped crop_mesh(const geo::Mesh& mesh, const geo::M4& T, double r) {
  std::vector<float> pos = mesh.positions, nrm = mesh.normals, uv = mesh.uvs;
  const bool has_n = mesh.has_normals, has_uv = mesh.has_uvs, has_m = mesh.has_material_ids;
  const auto vtx = [&](uint32_t i) { return apply(T, pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]); };
  const auto within = [&](uint32_t i) { return norm(vtx(i)) <= r; };
  std::unordered_map<uint64_t, uint32_t> mids;
  // The midpoint of an edge, once per edge: positions and uvs averaged and stored as float32, normals renormalized.
  const auto mid = [&](uint32_t a, uint32_t b) {
    const uint64_t key = (static_cast<uint64_t>(std::min(a, b)) << 32) | std::max(a, b);
    if (auto it = mids.find(key); it != mids.end()) return it->second;
    const auto i = static_cast<uint32_t>(pos.size() / 3);
    for (size_t k = 0; k < 3; ++k) {
      const double sum = static_cast<double>(pos[3 * a + k]) + static_cast<double>(pos[3 * b + k]);
      pos.push_back(static_cast<float>(0.5 * sum));
    }
    if (has_n) {
      double w[3], len = 0.0;
      for (size_t k = 0; k < 3; ++k) {
        w[k] = static_cast<double>(nrm[3 * a + k]) + static_cast<double>(nrm[3 * b + k]);
        len += w[k] * w[k];
      }
      len = std::sqrt(len);
      for (size_t k = 0; k < 3; ++k) nrm.push_back(len > 0 ? static_cast<float>(w[k] / len) : 0.0f);
    }
    if (has_uv) {
      for (size_t k = 0; k < 2; ++k) {
        const double sum = static_cast<double>(uv[2 * a + k]) + static_cast<double>(uv[2 * b + k]);
        uv.push_back(static_cast<float>(0.5 * sum));
      }
    }
    mids.emplace(key, i);
    return i;
  };
  const auto area = [&](uint32_t a, uint32_t b, uint32_t c) {
    const P3 p = vtx(a), q = vtx(b), s = vtx(c);
    const P3 u{q[0] - p[0], q[1] - p[1], q[2] - p[2]}, v{s[0] - p[0], s[1] - p[1], s[2] - p[2]};
    return 0.5 * norm({u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]});
  };
  const double min_edge = r / 64.0;
  struct Tri {
    uint32_t a, b, c;
    uint16_t m;
  };
  std::vector<Tri> stack, kept;
  Cropped out;
  for (size_t t = mesh.num_triangles(); t-- > 0;) {
    stack.push_back({mesh.indices[3 * t], mesh.indices[3 * t + 1], mesh.indices[3 * t + 2],
                     has_m ? mesh.material_ids[t] : uint16_t{0}});
  }
  while (!stack.empty()) {
    const Tri f = stack.back();
    stack.pop_back();
    if (within(f.a) && within(f.b) && within(f.c)) {
      kept.push_back(f);
      continue;
    }
    const P3 p = vtx(f.a), q = vtx(f.b), s = vtx(f.c);
    double d2 = 0.0, edge = 0.0;
    for (size_t k = 0; k < 3; ++k) {
      const double lo = std::min({p[k], q[k], s[k]}), hi = std::max({p[k], q[k], s[k]});
      const double gap = std::max({lo, -hi, 0.0});   // the AABB's distance from the origin along axis k
      d2 += gap * gap;
    }
    const P3* corners[4] = {&p, &q, &s, &p};
    for (size_t k = 0; k < 3; ++k) {
      const P3& u = *corners[k];
      const P3& w = *corners[k + 1];
      edge = std::max(edge, norm({u[0] - w[0], u[1] - w[1], u[2] - w[2]}));
    }
    if (std::sqrt(d2) > r || edge < min_edge) {
      out.dropped_area_m2 += area(f.a, f.b, f.c);
      continue;
    }
    const uint32_t ab = mid(f.a, f.b), bc = mid(f.b, f.c), ca = mid(f.c, f.a);
    stack.push_back({ab, bc, ca, f.m});
    stack.push_back({ca, bc, f.c, f.m});
    stack.push_back({ab, f.b, bc, f.m});
    stack.push_back({f.a, ab, ca, f.m});
  }
  // Compact: the vertices the kept triangles use, in vertex order.
  std::vector<int64_t> remap(pos.size() / 3, -1);
  for (const Tri& f : kept) remap[f.a] = remap[f.b] = remap[f.c] = 0;
  geo::Mesh& g = out.mesh;
  int64_t n = 0;
  for (size_t i = 0; i < remap.size(); ++i) {
    if (remap[i] < 0) continue;
    remap[i] = n++;
    g.positions.insert(g.positions.end(), pos.begin() + static_cast<std::ptrdiff_t>(3 * i),
                       pos.begin() + static_cast<std::ptrdiff_t>(3 * i + 3));
    if (has_n) {
      g.normals.insert(g.normals.end(), nrm.begin() + static_cast<std::ptrdiff_t>(3 * i),
                       nrm.begin() + static_cast<std::ptrdiff_t>(3 * i + 3));
    }
    if (has_uv) {
      g.uvs.insert(g.uvs.end(), uv.begin() + static_cast<std::ptrdiff_t>(2 * i),
                   uv.begin() + static_cast<std::ptrdiff_t>(2 * i + 2));
    }
  }
  for (const Tri& f : kept) {
    for (const uint32_t i : {f.a, f.b, f.c}) g.indices.push_back(static_cast<uint32_t>(remap[i]));
    if (has_m) g.material_ids.push_back(f.m);
  }
  g.has_normals = has_n;
  g.has_uvs = has_uv;
  g.has_material_ids = has_m;
  return out;
}

void place_statics(Statics& s, const std::array<double, 3>& origin, const ConvertOptions& o) {
  if (o.crop_radius_m && !(*o.crop_radius_m > 0.0 && *o.crop_radius_m <= 16.0)) {
    throw std::invalid_argument("crop_radius_m must lie in (0, 16] m");
  }
  std::vector<BodyStatic> kept;
  std::vector<int32_t> index_of(s.bodies.size(), -1);   // old body index -> kept index (lights refer to bodies)
  for (size_t bi = 0; bi < s.bodies.size(); ++bi) {
    BodyStatic& b = s.bodies[bi];
    index_of[bi] = static_cast<int32_t>(kept.size());
    if (!b.T_world_static || b.role == BodyStatic::Role::kCarrier) {
      kept.push_back(std::move(b));
      continue;
    }
    if (o.crop_radius_m) {
      Cropped c = crop_mesh(b.mesh, shifted(*b.T_world_static, origin), *o.crop_radius_m);
      if (c.dropped_area_m2 > 0.0 || c.mesh.num_triangles() != b.mesh.num_triangles()) {
        s.report.push_back({"CROP", "info", b.name,
                            "cropped to " + metres(*o.crop_radius_m) + " m about origin_world: " +
                                metres(c.dropped_area_m2) + " m^2 dropped" +
                                (c.mesh.num_triangles() == 0 ? " (the whole body)" : ""),
                            c.dropped_area_m2, BoundKind::kMeasured});
      }
      if (c.mesh.num_triangles() == 0) {
        index_of[bi] = -1;
        continue;
      }
      b.mesh = std::move(c.mesh);
      b.rho = 0.0;
      for (size_t i = 0; i < b.mesh.positions.size(); i += 3) {
        const auto& x = b.mesh.positions;
        b.rho = std::max(b.rho, norm({x[i], x[i + 1], x[i + 2]}));
      }
    } else if (const double reach = static_reach(b, origin); reach > 16.0) {
      s.report.push_back({"GEO-16M", "refuse", b.name,
                          "static geometry reaches " + metres(reach) +
                              " m from origin_world (limit 16 m, DESIGN §5.3); set scene.blender.crop_radius_m to "
                              "crop it",
                          reach, BoundKind::kMeasured});
    }
    kept.push_back(std::move(b));
  }
  s.bodies = std::move(kept);
  for (LightStatic& l : s.lights) {   // only moving bodies carry lights, and only static ones are cropped away
    if (l.light.body) l.light.body = index_of[static_cast<size_t>(*l.light.body)];
  }
}

}  // namespace balla::scene::blender
