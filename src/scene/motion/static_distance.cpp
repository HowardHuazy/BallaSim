#include "scene/motion/static_distance.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace balla::scene::motion {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

double dist(const rot::V& a, const rot::V& b) { return norm(a - b); }

// A sphere holding the points of triangles [t0, t1): the box centre and the largest vertex distance, inflated by a
// relative 1e-12 so that its lower bound never exceeds an exact triangle distance after rounding.
template <class Tri>
std::pair<rot::V, double> sphere(const Tri& tri, size_t t0, size_t t1) {
  rot::V lo = tri[t0][0], hi = lo;
  for (size_t t = t0; t < t1; ++t) {
    for (const rot::V& v : tri[t]) {
      lo = {std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
      hi = {std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
    }
  }
  const rot::V c = 0.5 * (lo + hi);
  double r = 0.0;
  for (size_t t = t0; t < t1; ++t) {
    for (const rot::V& v : tri[t]) r = std::max(r, dist(c, v));
  }
  return {c, r * (1.0 + 1e-12) + 1e-300};
}

// Ericson's closest-point regions. Its edge-ab case divides by |ab|^2, 0 when a == b: point_triangle_distance passes
// such a triangle as (b, c, a), which puts the repeated vertex in a and c, where every query ends in the vertex-a,
// vertex-b or edge-ab case (the exact distance to the segment; the planner's MeshDistance does the same).
double closest(const rot::V& p, const rot::V& a, const rot::V& b, const rot::V& c) {
  const rot::V ab = b - a, ac = c - a, ap = p - a;
  const double d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) return dist(p, a);
  const rot::V bp = p - b;
  const double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) return dist(p, b);
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return dist(p, a + (d1 / (d1 - d3)) * ab);
  const rot::V cp = p - c;
  const double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) return dist(p, c);
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return dist(p, a + (d2 / (d2 - d6)) * ac);
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    return dist(p, b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b));
  }
  const double denom = (va + vb) + vc;
  if (!(denom != 0.0)) {  // degenerate triangle: the nearest of its edges
    auto segment = [&](const rot::V& o, const rot::V& e) {
      return dist(p, o + std::clamp(dot(e, p - o) / std::max(dot(e, e), 1e-300), 0.0, 1.0) * e);
    };
    return std::min({segment(a, ab), segment(a, ac), segment(b, c - b)});
  }
  const double v = vb / denom, w = vc / denom;
  return dist(p, a + v * ab + w * ac);
}

}  // namespace

double point_triangle_distance(const rot::V& p, const rot::V& a, const rot::V& b, const rot::V& c) {
  if (a.x == b.x && a.y == b.y && a.z == b.z) return closest(p, b, c, a);
  return closest(p, a, b, c);
}

StaticDistance::StaticDistance(std::span<const double> vertices, std::span<const uint32_t> indices) {
  if (vertices.size() % 3 != 0 || indices.size() % 3 != 0) throw std::invalid_argument("StaticDistance: bad shapes");
  const size_t nv = vertices.size() / 3, nt = indices.size() / 3;
  if (nt == 0) return;
  auto vert = [&](uint32_t i) {
    if (i >= nv) throw std::invalid_argument("StaticDistance: vertex index out of range");
    const rot::V v{vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]};
    if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
      throw std::invalid_argument("StaticDistance: vertices are not finite");
    }
    return v;
  };
  std::vector<std::array<rot::V, 3>> tri(nt);
  rot::V lo{kInf, kInf, kInf}, hi{-kInf, -kInf, -kInf};
  for (size_t t = 0; t < nt; ++t) {
    tri[t] = {vert(indices[3 * t]), vert(indices[3 * t + 1]), vert(indices[3 * t + 2])};
    for (const rot::V& v : tri[t]) {
      lo = {std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
      hi = {std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
    }
  }
  const double span = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 1e-30});
  std::vector<uint64_t> code(nt, 0);
  for (size_t t = 0; t < nt; ++t) {
    const rot::V cen = (1.0 / 3.0) * ((tri[t][0] + tri[t][1]) + tri[t][2]);
    const double rel[3] = {cen.x - lo.x, cen.y - lo.y, cen.z - lo.z};
    for (int axis = 0; axis < 3; ++axis) {
      const auto cell = static_cast<uint64_t>(std::clamp(rel[axis] / span * 1024.0, 0.0, 1023.0));
      for (int bit = 0; bit < 10; ++bit) code[t] |= ((cell >> bit) & 1u) << (3 * bit + axis);
    }
  }
  std::vector<size_t> order(nt);
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return code[a] < code[b]; });
  tri_.resize(nt);
  for (size_t t = 0; t < nt; ++t) tri_[t] = tri[order[t]];
  for (size_t t0 = 0; t0 < nt; t0 += kLeaf) {
    const auto [c, r] = sphere(tri_, t0, std::min(nt, t0 + kLeaf));
    leaf_.push_back({c, r});
  }
  for (size_t t0 = 0; t0 < nt; t0 += kLeaf * kLeaf) {
    const auto [c, r] = sphere(tri_, t0, std::min(nt, t0 + kLeaf * kLeaf));
    group_.push_back({c, r});
  }
}

double StaticDistance::operator()(const rot::V& x) const {
  if (tri_.empty()) return kInf;
  auto above = [](double lower, double best, double r) { return lower > best + 1e-12 * (1.0 + std::fabs(lower) + r); };
  std::vector<std::pair<double, size_t>> groups(group_.size());
  for (size_t g = 0; g < group_.size(); ++g) groups[g] = {dist(x, group_[g].c) - group_[g].r, g};
  std::sort(groups.begin(), groups.end());
  double best = kInf;
  for (const auto& [lower, g] : groups) {
    if (above(lower, best, group_[g].r)) break;
    const size_t l1 = std::min(leaf_.size(), (g + 1) * kLeaf);
    for (size_t l = g * kLeaf; l < l1; ++l) {
      if (above(dist(x, leaf_[l].c) - leaf_[l].r, best, leaf_[l].r)) continue;
      const size_t t1 = std::min(tri_.size(), (l + 1) * kLeaf);
      for (size_t t = l * kLeaf; t < t1; ++t) {
        best = std::min(best, point_triangle_distance(x, tri_[t][0], tri_[t][1], tri_[t][2]));
      }
    }
  }
  return best;
}

}  // namespace balla::scene::motion
