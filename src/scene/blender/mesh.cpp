// Blender meshes as C1 meshes (docs/design/phase3/blender.md §4.1, §4.6): the split by (vertex, normal, uv), the
// body frame and the rigidity probes. Inputs are the exchange's raw arrays (loop triangles = Cycles' corner
// triangles, corner or point normals, UVs with v up; CV E4, DS d5).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "scene/blender/convert.h"

namespace balla::scene::blender {

namespace {

bool finite(const std::vector<float>& a) {
  return std::all_of(a.begin(), a.end(), [](float x) { return std::isfinite(x); });
}

double det3(const std::array<double, 9>& m) {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

// The bit pattern of a float with -0 as +0, so that equal bits mean == for finite values.
uint32_t bits(float x) {
  if (x == 0.0f) x = 0.0f;
  uint32_t b;
  std::memcpy(&b, &x, sizeof b);
  return b;
}

using CornerKey = std::array<uint32_t, 6>;   // normal (corner domain) and uv bits, uv spec
struct CornerKeyHash {
  size_t operator()(const CornerKey& k) const {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (const uint32_t x : k) {
      uint64_t z = h ^ (x + 0x9E3779B97F4A7C15ull);
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
      h = z ^ (z >> 31);
    }
    return static_cast<size_t>(h);
  }
};

// M^-T = cofactor(M) / det M.
std::array<double, 9> inverse_transpose(const std::array<double, 9>& m) {
  const double d = det3(m);
  return {(m[4] * m[8] - m[5] * m[7]) / d, (m[5] * m[6] - m[3] * m[8]) / d, (m[3] * m[7] - m[4] * m[6]) / d,
          (m[2] * m[7] - m[1] * m[8]) / d, (m[0] * m[8] - m[2] * m[6]) / d, (m[1] * m[6] - m[0] * m[7]) / d,
          (m[1] * m[5] - m[2] * m[4]) / d, (m[2] * m[3] - m[0] * m[5]) / d, (m[0] * m[4] - m[1] * m[3]) / d};
}

// EXTEND and CLIP are exact only while the texture coordinates stay in [0, 1] (§4.5); this absorbs their rounding.
constexpr double kUnitTol = 1e-6;

}  // namespace

SplitMesh split_mesh(const MeshIn& m, const std::array<double, 9>& M, const std::array<double, 3>& o, int slots,
                     std::span<const SlotUv> slot_uv) {
  const size_t nv = m.num_vertices(), nt = m.num_triangles(), nc = m.corner_vert.size();
  const auto fail = [&m](const std::string& code, const std::string& what) {
    throw GeometryError(code, "mesh " + m.id + ": " + what);
  };
  if (m.co.size() != 3 * nv || m.tri_corners.size() != 3 * nt) fail("BRG-PROTO", "co or tri_corners is not (n, 3)");
  if (!m.tri_mat.empty() && m.tri_mat.size() != nt) fail("BRG-PROTO", "tri_mat does not have one entry per triangle");
  const bool point = m.domain == MeshIn::Normals::kPoint;
  if (m.normals.size() != 3 * (point ? nv : nc)) fail("BRG-PROTO", "the normals do not match the normals domain");
  const std::vector<float>* uv = nullptr;   // the default map: the active render one, else the only one
  if (auto it = m.uvs.find(m.active_render_uv); it != m.uvs.end()) {
    uv = &it->second;
  } else if (m.uvs.size() == 1) {
    uv = &m.uvs.begin()->second;
  }
  for (const auto& [name, map] : m.uvs) {
    if (map.size() != 2 * nc) fail("BRG-PROTO", "UV map " + name + " is not (corners, 2)");
  }
  for (const int32_t v : m.corner_vert) {
    if (v < 0 || static_cast<size_t>(v) >= nv) fail("BRG-PROTO", "corner_vert out of range");
  }
  for (const int32_t c : m.tri_corners) {
    if (c < 0 || static_cast<size_t>(c) >= nc) fail("BRG-PROTO", "tri_corners out of range");
  }
  if (!finite(m.co) || !finite(m.normals)) fail("AST-V01", "positions or normals contain NaN or Inf");
  for (const int16_t k : m.tri_mat) {
    if (k < 0 || (slots > 0 && k >= slots)) {
      fail("GEO-MATSLOT", "material index " + std::to_string(k) + " outside its " + std::to_string(slots) + " slots");
    }
  }
  // Textured slots (§4.5): each distinct (map, affine) is a spec; a corner keys on its raw uv in its spec's map and on
  // the spec, and is written as spec.A (u, v, 1). Spec 0 is the untextured default: the default map, (u, 1 - v).
  struct Spec {
    const std::vector<float>* map;
    const SlotUv* slot;   // null for spec 0
  };
  std::vector<Spec> specs{{uv, nullptr}};
  std::vector<uint32_t> spec_of_slot;
  for (const SlotUv& su : slot_uv) {
    uint32_t id = 0;
    if (su.textured) {
      const std::vector<float>* map = su.map.empty() ? uv : nullptr;
      if (!su.map.empty()) {
        if (auto it = m.uvs.find(su.map); it != m.uvs.end()) map = &it->second;
      }
      if (map == nullptr) {
        fail("MAT-BAKE", su.map.empty() ? "a textured material on a mesh without a (single or active render) UV map"
                                        : "the UV map " + su.map + " a textured material names is not on the mesh");
      }
      for (id = 1; id < specs.size(); ++id) {
        if (specs[id].map == map && specs[id].slot->A == su.A && specs[id].slot->unit == su.unit) break;
      }
      if (id == specs.size()) specs.push_back({map, &su});
    }
    spec_of_slot.push_back(id);
  }
  // Only the maps read (the default one and the textured slots') must be finite: Cycles never reads the others.
  std::set<const std::vector<float>*> checked;
  for (const Spec& sp : specs) {
    if (sp.map == nullptr || !checked.insert(sp.map).second || finite(*sp.map)) continue;
    for (const auto& [name, map] : m.uvs) {
      if (&map == sp.map) fail("AST-V01", "UV map " + name + " contains NaN or Inf");
    }
  }
  const bool textured = specs.size() > 1;
  const auto spec_at = [&](uint32_t tc) -> uint32_t {
    if (!textured) return 0;
    const size_t k = m.tri_mat.empty() ? 0 : static_cast<size_t>(m.tri_mat[tc / 3]);
    return k < spec_of_slot.size() ? spec_of_slot[k] : 0;
  };
  const auto uv_at = [&](uint32_t tc, uint32_t spec, size_t q) -> float {   // raw uv component q of a corner
    const std::vector<float>* map = specs[spec].map;
    return map == nullptr ? 0.0f : (*map)[2 * static_cast<size_t>(m.tri_corners[tc]) + q];
  };
  const bool has_uv = uv != nullptr || textured;
  SplitMesh out;
  const double det = det3(M);
  if (nt == 0 || !(std::abs(det) > 0.0)) return out;   // nothing, or a zero-scale part: Cycles renders nothing
  std::vector<double> p(3 * nv);
  for (size_t i = 0; i < nv; ++i) {
    const double x = m.co[3 * i], y = m.co[3 * i + 1], z = m.co[3 * i + 2];
    for (size_t r = 0; r < 3; ++r) p[3 * i + r] = M[3 * r] * x + M[3 * r + 1] * y + M[3 * r + 2] * z + o[r];
  }
  std::vector<int64_t> tv(3 * nt);
  for (size_t k = 0; k < 3 * nt; ++k) tv[k] = m.corner_vert[static_cast<size_t>(m.tri_corners[k])];
  const std::vector<uint8_t> keep = geo::kept_faces(p, tv, 1e-7, &out.degenerate, &out.duplicate);

  // Bucket the kept corners by vertex (stable: triangle order), then give each distinct (normal, uv) of a vertex one
  // output vertex in order of first use. Output vertices follow the input vertex order.
  std::vector<uint32_t> start(nv + 1, 0);
  for (size_t t = 0; t < nt; ++t) {
    if (keep[t]) {
      for (size_t k = 0; k < 3; ++k) ++start[static_cast<size_t>(tv[3 * t + k]) + 1];
    }
  }
  for (size_t i = 0; i < nv; ++i) start[i + 1] += start[i];
  std::vector<uint32_t> fill(start.begin(), start.end() - 1), slot_tc(start[nv]);
  for (size_t t = 0; t < nt; ++t) {
    if (keep[t]) {
      for (size_t k = 0; k < 3; ++k) {
        slot_tc[fill[static_cast<size_t>(tv[3 * t + k])]++] = static_cast<uint32_t>(3 * t + k);
      }
    }
  }
  const auto same = [&](uint32_t a, uint32_t b) {   // two triangle corners of one vertex
    const size_t ca = static_cast<size_t>(m.tri_corners[a]), cb = static_cast<size_t>(m.tri_corners[b]);
    if (!point) {
      for (size_t r = 0; r < 3; ++r) {
        if (m.normals[3 * ca + r] != m.normals[3 * cb + r]) return false;
      }
    }
    if (!textured) return uv == nullptr || ((*uv)[2 * ca] == (*uv)[2 * cb] && (*uv)[2 * ca + 1] == (*uv)[2 * cb + 1]);
    const uint32_t sa = spec_at(a);
    return sa == spec_at(b) && uv_at(a, sa, 0) == uv_at(b, sa, 0) && uv_at(a, sa, 1) == uv_at(b, sa, 1);
  };
  // A vertex's distinct (normal, uv) are found by a linear scan while it has few, then through a hash of their bit
  // patterns (the corners are finite and -0 hashes as +0, so equal keys mean `same`): first-use order either way, so
  // the output does not depend on which was used, and a high-valence vertex (a fan's apex) costs O(valence).
  const auto key_of = [&](uint32_t tc) {
    const size_t c = static_cast<size_t>(m.tri_corners[tc]);
    CornerKey k{};
    if (!point) {
      for (size_t r = 0; r < 3; ++r) k[r] = bits(m.normals[3 * c + r]);
    }
    const uint32_t sp = spec_at(tc);
    if (has_uv) {
      k[3] = bits(uv_at(tc, sp, 0));
      k[4] = bits(uv_at(tc, sp, 1));
    }
    k[5] = sp;
    return k;
  };
  constexpr size_t kScan = 8;
  std::unordered_map<CornerKey, uint32_t, CornerKeyHash> index;
  const std::array<double, 9> N = inverse_transpose(M);
  std::vector<uint32_t> out_of(3 * nt), reps;
  uint32_t next = 0;
  for (size_t v = 0; v < nv; ++v) {
    reps.clear();
    bool hashed = false;
    const uint32_t base = next;
    for (uint32_t s = start[v]; s < start[v + 1]; ++s) {
      const uint32_t tc = slot_tc[s];
      size_t r = 0;
      if (hashed) {
        r = index.emplace(key_of(tc), static_cast<uint32_t>(reps.size())).first->second;
      } else {
        while (r < reps.size() && !same(reps[r], tc)) ++r;
      }
      if (r == reps.size()) {
        reps.push_back(tc);
        ++next;
        if (!hashed && reps.size() == kScan) {
          index.clear();
          for (size_t q = 0; q < reps.size(); ++q) index.emplace(key_of(reps[q]), static_cast<uint32_t>(q));
          hashed = true;
        }
      }
      out_of[tc] = base + static_cast<uint32_t>(r);
    }
    for (const uint32_t tc : reps) {
      const size_t c = static_cast<size_t>(m.tri_corners[tc]);
      out.positions.insert(out.positions.end(), {p[3 * v], p[3 * v + 1], p[3 * v + 2]});
      const float* n = &m.normals[3 * (point ? v : c)];
      double w[3];
      const double n0 = n[0], n1 = n[1], n2 = n[2];
      for (size_t r = 0; r < 3; ++r) w[r] = N[3 * r] * n0 + N[3 * r + 1] * n1 + N[3 * r + 2] * n2;
      const double len = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
      for (size_t r = 0; r < 3; ++r) out.normals.push_back(len > 0 ? static_cast<float>(w[r] / len) : 0.0f);
      const uint32_t sp = spec_at(tc);
      if (sp == 0 && uv != nullptr) {
        out.uvs.push_back((*uv)[2 * c]);
        out.uvs.push_back(static_cast<float>(1.0 - static_cast<double>((*uv)[2 * c + 1])));
      } else if (sp == 0 && has_uv) {
        out.uvs.insert(out.uvs.end(), {0.0f, 1.0f});
      } else if (sp != 0) {
        const std::array<double, 6>& A = specs[sp].slot->A;
        const double ru = uv_at(tc, sp, 0), rv = uv_at(tc, sp, 1);
        const double uo = A[0] * ru + A[1] * rv + A[2], vo = A[3] * ru + A[4] * rv + A[5];
        if (specs[sp].slot->unit && !(uo >= -kUnitTol && uo <= 1.0 + kUnitTol && vo >= -kUnitTol &&
                                      vo <= 1.0 + kUnitTol)) {
          fail("MAT-BAKE", "EXTEND or CLIP texture coordinates leave [0, 1] (C1 textures repeat)");
        }
        out.uvs.push_back(static_cast<float>(uo));
        out.uvs.push_back(static_cast<float>(vo));
      }
    }
  }
  const bool flip = det < 0;
  for (size_t t = 0; t < nt; ++t) {
    if (!keep[t]) continue;
    const uint32_t* o3 = &out_of[3 * t];
    out.indices.insert(out.indices.end(), {o3[0], o3[flip ? 2 : 1], o3[flip ? 1 : 2]});
    out.material_ids.push_back(static_cast<uint16_t>(m.tri_mat.empty() || slots == 0 ? 0 : m.tri_mat[t]));
  }
  return out;
}

geo::Mesh finish_mesh(std::vector<SplitMesh>& parts, const std::vector<int>& bases, int total, std::array<double, 3>* c,
                      double* rho) {
  double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  bool any_uv = false;
  for (const SplitMesh& s : parts) {
    any_uv = any_uv || !s.uvs.empty();
    for (size_t i = 0; i < s.positions.size(); ++i) {
      lo[i % 3] = std::min(lo[i % 3], s.positions[i]);
      hi[i % 3] = std::max(hi[i % 3], s.positions[i]);
    }
  }
  *c = {0.0, 0.0, 0.0};
  if (lo[0] <= hi[0]) *c = {0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])};
  geo::Mesh g;
  *rho = 0.0;
  for (size_t j = 0; j < parts.size(); ++j) {
    const SplitMesh& s = parts[j];
    const auto v0 = static_cast<uint32_t>(g.positions.size() / 3);
    for (size_t i = 0; i < s.positions.size(); i += 3) {
      double r2 = 0.0;
      for (size_t r = 0; r < 3; ++r) {
        const float x = static_cast<float>(s.positions[i + r] - (*c)[r]);
        g.positions.push_back(x);
        r2 += static_cast<double>(x) * static_cast<double>(x);
      }
      *rho = std::max(*rho, std::sqrt(r2));
    }
    for (const uint32_t i : s.indices) g.indices.push_back(i + v0);
    g.normals.insert(g.normals.end(), s.normals.begin(), s.normals.end());
    if (any_uv) {
      if (s.uvs.empty()) g.uvs.resize(g.uvs.size() + 2 * (s.positions.size() / 3), 0.0f);
      g.uvs.insert(g.uvs.end(), s.uvs.begin(), s.uvs.end());
    }
    for (const uint16_t k : s.material_ids) g.material_ids.push_back(static_cast<uint16_t>(k + bases[j]));
    parts[j] = SplitMesh{};   // release as we go
  }
  g.has_normals = true;
  g.has_uvs = any_uv;
  g.has_material_ids = total > 1;
  if (!g.has_material_ids) g.material_ids.clear();
  return g;
}

namespace {

bool flag(const Tree& t, const char* k) {
  const Tree* v = t.kind() == Tree::Kind::kDict ? t.find(k) : nullptr;
  return v != nullptr && v->kind() == Tree::Kind::kBool && v->as_bool();
}

std::string text_of(const Tree& t, const char* k) {
  const Tree* v = t.kind() == Tree::Kind::kDict ? t.find(k) : nullptr;
  return v != nullptr && v->kind() == Tree::Kind::kString ? v->as_string() : std::string();
}

const std::vector<Tree>& list_of(const Tree& t, const char* k) {
  static const std::vector<Tree> none;
  const Tree* v = t.kind() == Tree::Kind::kDict ? t.find(k) : nullptr;
  return v != nullptr && v->kind() == Tree::Kind::kList ? v->items() : none;
}

// Whether an animation record keys or drives a `modifiers[...]` path, in its F-curves, its drivers or the F-curves of
// the (action, slot) records its NLA strips play.
bool animates_modifiers(const ExchangeIn& ex, int32_t a) {
  if (a < 0 || ex.animations.kind() != Tree::Kind::kList) return false;
  const std::vector<Tree>& anims = ex.animations.items();
  if (static_cast<size_t>(a) >= anims.size()) return false;
  const auto keyed = [](const Tree& rec) {
    for (const char* k : {"fcurves", "drivers"}) {
      for (const Tree& f : list_of(rec, k)) {
        if (text_of(f, "data_path").rfind("modifiers[", 0) == 0) return true;
      }
    }
    return false;
  };
  const Tree& an = anims[static_cast<size_t>(a)];
  if (keyed(an)) return true;
  for (const Tree& track : list_of(an, "nla")) {
    for (const Tree& strip : list_of(track, "strips")) {
      const std::string action = text_of(strip, "action"), slot = text_of(strip, "slot");
      if (action.empty()) continue;
      for (const Tree& rec : anims) {
        if (text_of(rec, "action") == action && text_of(rec, "slot") == slot && keyed(rec)) return true;
      }
    }
  }
  return false;
}

}  // namespace

bool steady_modifier(const std::string& type) {
  static const std::set<std::string> kSteady{"ARRAY", "BEVEL", "BOOLEAN", "DATA_TRANSFER", "DECIMATE", "EDGE_SPLIT",
                                             "MASK", "MIRROR", "MULTIRES", "NORMAL_EDIT", "REMESH", "SCREW", "SKIN",
                                             "SOLIDIFY", "SUBSURF", "TRIANGULATE", "UV_PROJECT", "UV_WARP",
                                             "VOLUME_TO_MESH", "WEIGHTED_NORMAL", "WELD", "WIREFRAME"};
  return kSteady.count(type) != 0;
}

bool may_change_shape(const ExchangeIn& ex, const ObjectIn& ob) {
  if (flag(ob.deform, "is_deform_modified") || flag(ob.deform, "shape_keys_animated") ||
      flag(ob.deform, "has_geometry_nodes")) {
    return true;
  }
  for (const Tree& md : list_of(ob.deform, "modifiers")) {
    const std::string type = md.kind() == Tree::Kind::kString ? md.as_string() : text_of(md, "type");
    if (type == "NODES" || type == "CLOTH" || type == "SOFT_BODY") return true;
    const Tree* shown = md.kind() == Tree::Kind::kDict ? md.find("show_render") : nullptr;
    if (shown != nullptr && shown->kind() == Tree::Kind::kBool && !shown->as_bool()) continue;
    if (!steady_modifier(type) || !list_of(md, "objects").empty()) return true;
  }
  return animates_modifiers(ex, ob.animation);
}

std::vector<int32_t> rigidity_probe_objects(const ExchangeIn& ex) {
  std::vector<int32_t> out;
  for (size_t i = 0; i < ex.objects.size(); ++i) {
    const ObjectIn& ob = ex.objects[i];
    if (ob.mesh >= 0 && may_change_shape(ex, ob)) out.push_back(static_cast<int32_t>(i));
  }
  return out;
}

std::optional<ReportItem> check_rigidity(const std::string& object, std::span<const float> co0,
                                         std::span<const std::vector<float>> probes, double tol_bu) {
  double worst = 0.0;
  for (const std::vector<float>& co : probes) {
    if (co.size() != co0.size()) {
      return ReportItem{"GEO-DEFORM", "refuse", object, "the evaluated topology changes over the span (Phase 4)", 0.0,
                        BoundKind::kNone};
    }
    for (size_t i = 0; i < co.size(); i += 3) {
      const double dx = static_cast<double>(co[i]) - static_cast<double>(co0[i]);
      const double dy = static_cast<double>(co[i + 1]) - static_cast<double>(co0[i + 1]);
      const double dz = static_cast<double>(co[i + 2]) - static_cast<double>(co0[i + 2]);
      worst = std::max(worst, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
  }
  if (worst <= tol_bu) return std::nullopt;
  return ReportItem{"GEO-DEFORM", "refuse", object,
                    "object-space vertices move up to " + std::to_string(worst) + " BU (> rigidity_tol_m): a deforming "
                    "mesh (Phase 4)", worst, BoundKind::kMeasured};
}

}  // namespace balla::scene::blender
