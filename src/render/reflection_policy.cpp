#include "render/reflection_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

#include "render/texture.h"

namespace balla::render {
namespace {

using V3 = std::array<double, 3>;
constexpr double kInf = std::numeric_limits<double>::infinity();

V3 sub(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
V3 axpy(double s, const V3& x, const V3& y) { return {y[0] + s * x[0], y[1] + s * x[1], y[2] + s * x[2]}; }
double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double norm(const V3& a) { return std::sqrt(dot(a, a)); }

// T x for a row-major rigid T (C1's T_world_body).
V3 apply(const std::array<double, 16>& t, const V3& x) {
  return {t[0] * x[0] + t[1] * x[1] + t[2] * x[2] + t[3], t[4] * x[0] + t[5] * x[1] + t[6] * x[2] + t[7],
          t[8] * x[0] + t[9] * x[1] + t[10] * x[2] + t[11]};
}

// The band value of a linear RGB triple, in scene_data.cpp's order.
double band(const std::array<double, 3>& w, const std::array<double, 3>& c) {
  return (w[0] * c[0] + w[1] * c[1]) + w[2] * c[2];
}

std::array<double, 3> rgb(const std::array<float, 3>& c) {
  return {static_cast<double>(c[0]), static_cast<double>(c[1]), static_cast<double>(c[2])};
}

// IEEE binary16 bits as a double (render::validate refused non-finite texels).
double half_value(uint16_t h) {
  const int e = (h >> 10) & 0x1F, f = h & 0x3FF;
  const double m = e == 0 ? std::ldexp(f, -24) : std::ldexp(1024 + f, e - 25);
  return (h & 0x8000u) != 0 ? -m : m;
}

// The range of a material's visible band albedo w . (base_color x texel) over its texels; a texture on a mesh without
// UVs renders as its mean (scene_data.cpp), so its range is that mean.
std::array<double, 2> visible_range(const Material& m, const std::array<double, 3>& w, bool has_uvs) {
  const std::array<double, 3> base = rgb(m.base_color);
  const std::array<double, 3> wb{w[0] * base[0], w[1] * base[1], w[2] * base[2]};
  if (m.texture.empty() && m.texture_linear.empty()) {
    const double r = (wb[0] + wb[1]) + wb[2];
    return {r, r};
  }
  const size_t n = static_cast<size_t>(m.texture_width) * static_cast<size_t>(m.texture_height);
  const size_t ch = m.texture.empty() ? 3 : static_cast<size_t>(m.texture_channels);
  double lo = kInf, hi = -kInf, sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    std::array<double, 3> c{};
    for (size_t k = 0; k < 3; ++k) {
      c[k] = m.texture.empty() ? half_value(m.texture_linear[3 * i + k]) : srgb_to_linear(m.texture[ch * i + k]);
    }
    const double r = band(wb, c);
    lo = std::min(lo, r);
    hi = std::max(hi, r);
    sum += r;
  }
  if (!has_uvs) return {sum / static_cast<double>(n), sum / static_cast<double>(n)};
  return {lo, hi};
}

// The distinct NIR fractions of the cameras' rows (P2-F F11a; sensor.nir_fraction per camera and light group): an empty
// row, like no row at all, is f = 0. Throws std::invalid_argument for an f outside [0, 1].
std::vector<double> nir_fraction_set(const std::vector<std::vector<double>>& rows) {
  std::vector<double> fs;
  const auto add = [&](double f) {
    if (!(f >= 0.0 && f <= 1.0)) throw std::invalid_argument("reflection_policy: nir_fraction must lie in [0, 1]");
    if (std::find(fs.begin(), fs.end(), f) == fs.end()) fs.push_back(f);
  };
  if (rows.empty()) add(0.0);
  for (const std::vector<double>& row : rows) {
    if (row.empty()) add(0.0);
    for (double f : row) add(f);
  }
  return fs;
}

// The range of the albedo a camera's light group sees, (1 - f) rho_vis + f rho_NIR (render/shade/nir.h), over the
// material's texels and over the fractions f in `fs`: rho_lo and rho_hi bound the rendered radiance of every (camera,
// group) pair from below and above. The visible range alone (visible_range) where no f is above 0 or the material is
// baked (a baked radiance has no NIR albedo); band_images is the renderer's own level-0 albedo images.
std::array<double, 2> albedo_range(const Material& m, const std::array<double, 3>& w, bool has_uvs,
                                   const std::vector<double>& fs) {
  if (m.shading == Shading::kBaked || !std::any_of(fs.begin(), fs.end(), [](double f) { return f > 0.0; })) {
    return visible_range(m, w, has_uvs);
  }
  const std::array<double, 3> base = rgb(m.base_color);
  const BandImages im = band_images(m, {w[0] * base[0], w[1] * base[1], w[2] * base[2]});
  const bool image = im.width > 0;
  const double n = image ? static_cast<double>(im.vis.size()) : 1.0;
  double vis_mean = im.vis_const, nir_mean = im.nir_const;
  if (image) {
    vis_mean = nir_mean = 0.0;
    for (size_t i = 0; i < im.vis.size(); ++i) {
      vis_mean += im.vis[i];
      nir_mean += im.nir[i];
    }
    vis_mean /= n;
    nir_mean /= n;
  }
  double lo = kInf, hi = -kInf;
  for (double f : fs) {
    if (!image || !has_uvs) {  // a constant, or the mean of a texture that cannot be placed
      const double r = (1.0 - f) * vis_mean + f * nir_mean;
      lo = std::min(lo, r);
      hi = std::max(hi, r);
      continue;
    }
    for (size_t i = 0; i < im.vis.size(); ++i) {
      const double r = (1.0 - f) * im.vis[i] + f * im.nir[i];
      lo = std::min(lo, r);
      hi = std::max(hi, r);
    }
  }
  return {lo, hi};
}

// Squared distance from p to triangle abc (Ericson, Real-Time Collision Detection §5.1.5, closest point by regions).
double point_triangle_distance2(const V3& p, const V3& a, const V3& b, const V3& c) {
  const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const double d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) return dot(ap, ap);
  const V3 bp = sub(p, b);
  const double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) return dot(bp, bp);
  const auto at = [&](const V3& q) { return dot(sub(p, q), sub(p, q)); };
  const auto ratio = [](double num, double den) { return den > 0.0 ? num / den : 0.0; };
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return at(axpy(ratio(d1, d1 - d3), ab, a));
  const V3 cp = sub(p, c);
  const double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) return dot(cp, cp);
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return at(axpy(ratio(d2, d2 - d6), ac, a));
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0) {
    return at(axpy(ratio(d4 - d3, (d4 - d3) + (d5 - d6)), sub(c, b), b));
  }
  const double den = va + vb + vc;
  if (!(den > 0.0)) return std::min({dot(ap, ap), dot(bp, bp), dot(cp, cp)});  // degenerate: a vertex bounds it
  return at(axpy(vc / den, ac, axpy(vb / den, ab, a)));
}

// Where something can be over the run: a ball (radius 0: a fixed point).
struct Ball {
  V3 c{0.0, 0.0, 0.0};
  double r = 0.0;
};

// A ball holding every position of a trajectory's spline: the control positions lie in it, and a B-spline stays in
// their convex hull.
Ball trajectory_ball(const Trajectory& t) {
  const size_t n = t.pos.size() / 3;
  V3 lo{kInf, kInf, kInf}, hi{-kInf, -kInf, -kInf};
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < 3; ++k) {
      lo[k] = std::min(lo[k], t.pos[3 * i + k]);
      hi[k] = std::max(hi[k], t.pos[3 * i + k]);
    }
  }
  Ball b{{0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])}, 0.0};
  for (size_t i = 0; i < n; ++i) b.r = std::max(b.r, norm(sub({t.pos[3 * i], t.pos[3 * i + 1], t.pos[3 * i + 2]}, b.c)));
  return b;
}

// A body's geometry for the distance bound: a static body's world-frame vertices, a moving one's ball.
struct Geometry {
  const Body* body = nullptr;
  bool moving = false;
  Ball ball;
  std::vector<V3> world;
};

Geometry geometry(const SceneBundle& s, const Body& b) {
  Geometry g{&b, b.trajectory.has_value(), {}, {}};
  const size_t nv = b.mesh.positions.size() / 3;
  if (g.moving) {
    g.ball = trajectory_ball(s.trajectories[static_cast<size_t>(*b.trajectory)]);
    double r = 0.0;
    for (size_t i = 0; i < nv; ++i) {
      r = std::max(r, norm({b.mesh.positions[3 * i], b.mesh.positions[3 * i + 1], b.mesh.positions[3 * i + 2]}));
    }
    g.ball.r += r;
  } else {
    g.world.resize(nv);
    for (size_t i = 0; i < nv; ++i) {
      g.world[i] = apply(b.T_world_body, {b.mesh.positions[3 * i], b.mesh.positions[3 * i + 1], b.mesh.positions[3 * i + 2]});
    }
  }
  return g;
}

// The smallest distance between a light's ball and a body over the run (0 when they may touch).
double distance(const Ball& light, const Geometry& g) {
  if (g.moving) return std::max(0.0, norm(sub(light.c, g.ball.c)) - g.ball.r - light.r);
  double d2 = kInf;
  const std::vector<uint32_t>& idx = g.body->mesh.indices;
  for (size_t t = 0; t + 2 < idx.size(); t += 3) {
    d2 = std::min(d2, point_triangle_distance2(light.c, g.world[idx[t]], g.world[idx[t + 1]], g.world[idx[t + 2]]));
  }
  return std::max(0.0, std::sqrt(d2) - light.r);
}

// The light's place over the run: its world position, or the ball of a moving carrier widened by the offset.
Ball light_ball(const SceneBundle& s, const Light& l) {
  if (!l.body) return {l.position, 0.0};
  const Body& carrier = s.bodies[static_cast<size_t>(*l.body)];
  if (!carrier.trajectory) return {apply(carrier.T_world_body, l.position), 0.0};
  Ball b = trajectory_ball(s.trajectories[static_cast<size_t>(*carrier.trajectory)]);
  b.r += norm(l.position);
  return b;
}

std::vector<uint32_t> used_materials(const Body& b) {
  if (b.mesh.material_ids.empty()) return {0};
  std::vector<uint32_t> out(b.mesh.material_ids.begin(), b.mesh.material_ids.end());
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

}  // namespace

bool reflection_refused(const Material& m) {
  return m.has_lobe() &&
         (m.roughness < kReflectionRoughnessMin || (m.metallic > 0.0f && m.roughness < kReflectionMetalRoughnessMin));
}

std::vector<ReflectionReport> reflection_policy(const SceneBundle& s, const std::array<double, 3>& w,
                                                const std::vector<GroupRange>& groups,
                                                const std::vector<std::vector<double>>& nir_fraction) {
  validate(s);
  const std::vector<double> fs = nir_fraction_set(nir_fraction);
  if (!groups.empty() && groups.size() != s.light_groups.size()) {
    throw std::invalid_argument("reflection_policy: groups need one range per light group (" +
                                std::to_string(s.light_groups.size()) + "), got " + std::to_string(groups.size()));
  }
  for (const GroupRange& g : groups) {
    if (!(g.lo >= 0.0 && g.lo <= g.hi && std::isfinite(g.hi))) {
      throw std::invalid_argument("reflection_policy: a light group's range needs 0 <= lo <= hi < inf");
    }
  }
  const auto range = [&](const Light& l) {
    return groups.empty() ? GroupRange{} : groups[static_cast<size_t>(l.group)];
  };
  const Material fallback{};  // a body without materials renders with it (scene_data.cpp)
  const auto material = [&](const Body& b, uint32_t j) -> const Material& {
    return b.materials.empty() ? fallback : b.materials[j];
  };
  std::vector<ReflectionReport> out;
  for (size_t i = 0; i < s.bodies.size(); ++i) {
    const Body& b = s.bodies[i];
    if (b.mesh.indices.empty()) continue;
    for (uint32_t j : used_materials(b)) {
      const Material& m = material(b, j);
      if (m.has_lobe()) {
        out.push_back({static_cast<uint32_t>(i), j, m.roughness, m.metallic, m.specular, reflection_refused(m)});
      }
    }
  }
  if (out.empty()) return out;

  // The lights in band units (scene_data.cpp: intensity w . colour / sum w). The upper terms (A, B, E_max) take every
  // light at its group's peak and the lower one (the ambient of L_diff,min) at its group's trough.
  const double w_sum = (w[0] + w[1]) + w[2];
  const auto intensity = [&](const Light& l) {
    return static_cast<double>(l.intensity) * band(w, rgb(l.color)) / w_sum;
  };
  double ambient = 0.0, ambient_min = 0.0, background = 0.0, e_max = 0.0;
  std::vector<Geometry> geometries;
  for (const Light& l : s.lights) {
    const double v0 = intensity(l);
    if (!(v0 > 0.0)) continue;
    const GroupRange gr = range(l);
    const double v = v0 * gr.hi;
    switch (l.kind) {
      case BundleLightKind::kAmbient:
        ambient += v;
        ambient_min += v0 * gr.lo;
        break;
      case BundleLightKind::kBackground: background += v; break;
      case BundleLightKind::kDirectional: e_max += v; break;
      case BundleLightKind::kRect:
      case BundleLightKind::kDisk: e_max += std::numbers::pi * v; break;  // radiance L over a half-space: E <= pi L
      case BundleLightKind::kPoint:
      case BundleLightKind::kSpot: {
        if (geometries.empty()) {
          for (const Body& b : s.bodies) {
            if (!b.mesh.indices.empty()) geometries.push_back(geometry(s, b));
          }
        }
        const Ball place = light_ball(s, l);
        double d = kInf;
        for (const Geometry& g : geometries) d = std::min(d, distance(place, g));
        e_max += d > 0.0 ? v / (d * d) : kInf;
        break;
      }
    }
  }

  // A baked material's radiance K rho splits over the light groups by its shares a_g (all in group 0 when empty), each
  // following its waveform: its peak is K rho sum_g a_g m_g,max.
  const auto baked_gain = [&](const Material& m) {
    if (groups.empty()) return 1.0;
    if (m.baked_groups.empty()) return groups[0].hi;
    double gain = 0.0;
    for (size_t g = 0; g < m.baked_groups.size(); ++g) gain += static_cast<double>(m.baked_groups[g]) * groups[g].hi;
    return gain;
  };

  // L_max over every used material of every body with geometry.
  double l_max = background;
  for (const Body& b : s.bodies) {
    if (b.mesh.indices.empty()) continue;
    for (uint32_t j : used_materials(b)) {
      const Material& m = material(b, j);
      const double rho_hi = albedo_range(m, w, !b.mesh.uvs.empty(), fs)[1];
      const double l = m.shading == Shading::kBaked
                           ? static_cast<double>(m.baked_scale) * rho_hi * baked_gain(m)
                           : rho_hi * (e_max / std::numbers::pi + ambient) +
                                 band(w, rgb(m.emissive));
      l_max = std::max(l_max, l);
    }
  }
  for (ReflectionReport& r : out) {
    const Body& b = s.bodies[r.body];
    const Material& m = material(b, r.material);
    const double rho_lo = albedo_range(m, w, !b.mesh.uvs.empty(), fs)[0];
    r.l_max = l_max;
    r.l_diff_min = std::min(1.0, (1.0 - static_cast<double>(m.metallic)) * rho_lo) * ambient_min;
    r.bound = r.l_diff_min > 0.0 && std::isfinite(l_max) ? std::log1p(kReflectionFMax * l_max / r.l_diff_min) : kInf;
  }
  return out;
}

}  // namespace balla::render
