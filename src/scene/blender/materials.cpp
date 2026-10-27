// Blender materials and images in C1 terms (docs/design/phase3/blender.md §4.5, §6.1): the accepted node graphs
// (sockets by identifier), the exactness condition of C1's Lambert, the MAT-* and TEX-* codes, and the C1 textures.
//
// Provenance (PHASE3.md §0 rule 4): re-derived from the Blender manual (Material Output, Principled BSDF, Diffuse
// BSDF, Emission, Add and Mix Shader, Mix, Image Texture, Mapping, UV Map and Texture Coordinate nodes) and from
// headless oracles, not from Blender's sources: CV E5 and R2 c1 (Principled and Diffuse BSDF against Lambert: the
// sampled ratios below), CV E6 and E8 (byte pixels are raw codes; linear 8-bit codes re-encoded to sRGB err <= 2.63 %),
// RI r1_sockets (socket identifiers) and the BL-2b probe `ballasim_out/phase3_research/blender/bl2b/probes/
// m1_material_oracle.py` (4.2.1 and 4.5.3, identical results, kept with their JSON and the check a1_check_m1.py): the
// Mapping node gives POINT x' = R (s * x) + L, TEXTURE x' = (R^T (x - L)) / s and VECTOR x' = R (s * x) with
// R = Rz Ry Rx of its Euler angles; the Mix node gives MIX (1 - f) A + f B and MULTIPLY A (1 - f + f B), f clamped
// to [0, 1] by default; an Image Texture's MIRROR extension mirrors every other period in s and t; byte sRGB images
// decode by IEC 61966-2-1 and byte Linear Rec.709 ones read code / 255. tests/data/blender_m1_material_oracle.json
// keeps the renders the tests compare with. Probe `m2_output_alpha.py` (4.2.1 and 4.5.3, identical; check
// a3_check_m2.py): Cycles renders a Material Output whose target is CYCLES over every ALL one, active or not, and
// within one target the active output (the first in node order when none is); an Image Texture's Color output of a
// byte RGBA image is dec(floor(c a / 255) / 255) for sRGB and Linear Rec.709 images of alpha_mode STRAIGHT (and
// PREMUL as generated), within 2.7e-7, and the raw codes for CHANNEL_PACKED, NONE and every Non-Color image.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "scene/blender/convert.h"

namespace balla::scene::blender {

namespace {

enum class Kind {
  kOther, kOutput, kPrincipled, kDiffuse, kEmission, kAddShader, kMixShader, kMix, kTexImage, kMapping, kUvMap,
  kTexCoord, kReroute, kFrame
};

// A node's kind by its bl_idname or its RNA type (the exporter may write either; RI r1_sockets, probe m1 sockets).
Kind kind_of(const std::string& t) {
  static const std::map<std::string, Kind> kinds{
      {"ShaderNodeOutputMaterial", Kind::kOutput}, {"OUTPUT_MATERIAL", Kind::kOutput},
      {"ShaderNodeBsdfPrincipled", Kind::kPrincipled}, {"BSDF_PRINCIPLED", Kind::kPrincipled},
      {"ShaderNodeBsdfDiffuse", Kind::kDiffuse}, {"BSDF_DIFFUSE", Kind::kDiffuse},
      {"ShaderNodeEmission", Kind::kEmission}, {"EMISSION", Kind::kEmission},
      {"ShaderNodeAddShader", Kind::kAddShader}, {"ADD_SHADER", Kind::kAddShader},
      {"ShaderNodeMixShader", Kind::kMixShader}, {"MIX_SHADER", Kind::kMixShader},
      {"ShaderNodeMix", Kind::kMix}, {"MIX", Kind::kMix},
      {"ShaderNodeTexImage", Kind::kTexImage}, {"TEX_IMAGE", Kind::kTexImage},
      {"ShaderNodeMapping", Kind::kMapping}, {"MAPPING", Kind::kMapping},
      {"ShaderNodeUVMap", Kind::kUvMap}, {"UVMAP", Kind::kUvMap},
      {"ShaderNodeTexCoord", Kind::kTexCoord}, {"TEX_COORD", Kind::kTexCoord},
      {"NodeReroute", Kind::kReroute}, {"REROUTE", Kind::kReroute},
      {"NodeFrame", Kind::kFrame}, {"FRAME", Kind::kFrame}};
  const auto it = kinds.find(t);
  return it == kinds.end() ? Kind::kOther : it->second;
}

// CV E5 (4.2.1): the largest sampled Principled/Lambert ratio (Specular IOR Level 0.5, Metallic 0) at roughness 0.3,
// 0.5 and 1.0 over views 0/45/70 deg and suns 0/45 deg; quoted for the sampled roughness not above the material's.
constexpr double kSpecRatio[3][2] = {{0.3, 7.13}, {0.5, 1.74}, {1.0, 1.045}};
// R2 c1 (4.2.1, 4.5.3): the smallest sampled diffuse-roughness/Lambert ratio, up to roughness 0.5 and up to 1.0.
constexpr double kDiffRoughMin[2][2] = {{0.5, 0.58}, {1.0, 0.26}};
// CV E6: the largest relative error of a linear 8-bit code re-encoded to the nearest sRGB code and decoded.
constexpr double kLinearToSrgbErr = 0.02631;   // 0.0263028 measured, rounded up

// The Principled inputs §4.5 maps or checks by name; every other enabled input must keep its default (MAT-INPUT).
const std::set<std::string> kPrincipledKnown{
    "Base Color", "Metallic", "Roughness", "Specular IOR Level", "Alpha", "Normal", "Weight", "Diffuse Roughness",
    "Subsurface Weight", "Transmission Weight", "Coat Weight", "Sheen Weight", "Emission Color", "Emission Strength"};

struct Unsupported {   // a graph outside §4.5's set: one item ends the material's conversion
  std::string code, message;
};

using Affine = std::array<double, 12>;   // 3 x 4 row-major: x' = A x + c on (x, y, z, 1)

Affine compose(const std::array<double, 9>& A, const std::array<double, 3>& c, const Affine& m) {
  Affine out{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      double v = A[3 * i] * m[j] + A[3 * i + 1] * m[4 + j] + A[3 * i + 2] * m[8 + j];
      if (j == 3) v += c[i];
      out[4 * i + j] = v;
    }
  }
  return out;
}

// R = Rz Ry Rx (probe m1: the Mapping node's Euler XYZ rotation).
std::array<double, 9> euler_xyz(const std::array<double, 3>& r) {
  const double cx = std::cos(r[0]), sx = std::sin(r[0]), cy = std::cos(r[1]), sy = std::sin(r[1]);
  const double cz = std::cos(r[2]), sz = std::sin(r[2]);
  return {cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
          sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
          -sy,     cy * sx,                cy * cx};
}

struct Tex {   // one Image Texture lookup: its image, settings and the map from (u, v) to Blender's (s, t)
  int32_t image = -1;
  std::string extension, map;
  std::array<double, 6> st{1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  bool operator==(const Tex& o) const {
    return image == o.image && extension == o.extension && map == o.map && st == o.st;
  }
};

struct Col {   // a colour a + b T, with T the texture's linear colour (b = 0 without one)
  std::array<double, 3> a{}, b{};
  std::optional<Tex> tex;
};

bool zero(const std::array<double, 3>& v) { return v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0; }

// The inputs each converted node kind must carry (identifiers, as both versions name them: RI r1_sockets, probe m1).
const std::map<Kind, std::vector<std::string>>& required_inputs() {
  static const std::map<Kind, std::vector<std::string>> req{
      {Kind::kOutput, {"Surface", "Volume", "Displacement"}},
      {Kind::kPrincipled, {"Base Color", "Metallic", "Roughness", "Specular IOR Level", "Alpha", "Normal",
                           "Subsurface Weight", "Transmission Weight", "Coat Weight", "Sheen Weight", "Emission Color",
                           "Emission Strength"}},
      {Kind::kDiffuse, {"Color", "Roughness", "Normal"}},
      {Kind::kEmission, {"Color", "Strength"}},
      {Kind::kAddShader, {"Shader", "Shader_001"}},
      {Kind::kMixShader, {"Fac", "Shader", "Shader_001"}},
      {Kind::kMix, {"Factor_Float", "A_Color", "B_Color"}},
      {Kind::kTexImage, {"Vector"}},
      {Kind::kMapping, {"Vector", "Location", "Rotation", "Scale"}}};
  return req;
}

class Graph {
 public:
  Graph(const MaterialIn& m, const std::vector<ImageIn>& images) : m_(m), images_(images) {
    for (size_t i = 0; i < m.nodes.size(); ++i) {
      if (!node_of_.emplace(m.nodes[i].name, i).second) proto("two nodes named " + m.nodes[i].name);
      if (const auto r = required_inputs().find(kind(i)); r != required_inputs().end()) {
        for (const std::string& id : r->second) {
          if (input(i, id) == nullptr) proto(m.nodes[i].name + " (" + m.nodes[i].type + ") has no input " + id);
        }
      }
    }
    from_.resize(m.nodes.size());
    for (const LinkIn& l : m.links) {
      if (!node_of_.count(l.from_node) || !node_of_.count(l.to_node)) proto("a link names a missing node");
      if (!link_to_.emplace(std::make_pair(l.to_node, l.to_socket), &l).second) {
        proto("two links into " + l.to_node + "." + l.to_socket);
      }
      from_[node_of_.at(l.to_node)].push_back(node_of_.at(l.from_node));
    }
  }

  const NodeIn& node(size_t i) const { return m_.nodes[i]; }
  Kind kind(size_t i) const { return kind_of(m_.nodes[i].type); }

  const SocketIn* input(size_t i, const std::string& id) const {
    for (const SocketIn& s : m_.nodes[i].inputs) {
      if (s.identifier == id) return &s;
    }
    return nullptr;
  }

  // The node and output socket feeding input `id` of node i, through reroutes; nullopt when unlinked.
  std::optional<std::pair<size_t, std::string>> source(size_t i, const std::string& id) const {
    std::string node = m_.nodes[i].name, sock = id;
    for (size_t hops = 0; hops <= m_.nodes.size(); ++hops) {
      const auto it = link_to_.find({node, sock});
      if (it == link_to_.end()) return std::nullopt;
      const size_t j = node_of_.at(it->second->from_node);
      if (flag(j, "mute")) throw Unsupported{"MAT-BAKE", "the muted node " + m_.nodes[j].name + " is in use"};
      if (kind(j) != Kind::kReroute) return std::make_pair(j, it->second->from_socket);
      node = m_.nodes[j].name;
      sock = "Input";
    }
    proto("a reroute cycle");
  }
  bool linked(size_t i, const std::string& id) const { return source(i, id).has_value(); }

  std::string prop(size_t i, const char* key, const std::string& fallback) const {
    const Tree* v = m_.nodes[i].props.kind() == Tree::Kind::kDict ? m_.nodes[i].props.find(key) : nullptr;
    return v != nullptr && v->kind() == Tree::Kind::kString ? v->as_string() : fallback;
  }
  bool flag(size_t i, const char* key, bool fallback = false) const {
    const Tree* v = m_.nodes[i].props.kind() == Tree::Kind::kDict ? m_.nodes[i].props.find(key) : nullptr;
    return v != nullptr && v->kind() == Tree::Kind::kBool ? v->as_bool() : fallback;
  }

  // An unlinked input's value (component k; a scalar repeats), or `fallback` when the socket is absent.
  double value(size_t i, const std::string& id, size_t k = 0, double fallback = 0.0) const {
    const SocketIn* s = input(i, id);
    if (s == nullptr || s->n == 0) return fallback;
    return s->value[std::min(k, static_cast<size_t>(s->n - 1))];
  }
  std::array<double, 3> vec3(size_t i, const std::string& id, double fallback) const {
    const SocketIn* s = input(i, id);
    if (s == nullptr || !s->enabled) return {fallback, fallback, fallback};
    return {value(i, id, 0, fallback), value(i, id, 1, fallback), value(i, id, 2, fallback)};
  }

  [[noreturn]] void cycle() const { proto("a link cycle"); }
  size_t size() const { return m_.nodes.size(); }

  // The colour at input `id` of node i (§4.5: a constant, an Image Texture through optional Mappings, or a Mix of
  // MIX or MULTIPLY over those). Each node output is evaluated once (memo_), so a graph whose nodes feed one another
  // more than once costs linear time; the depth bound stops a cycle.
  Col color(size_t i, const std::string& id, size_t depth = 0) const {
    if (depth > m_.nodes.size()) cycle();
    const auto src = source(i, id);
    if (!src) {
      Col c;
      c.a = {value(i, id, 0), value(i, id, 1), value(i, id, 2)};
      return c;
    }
    if (const auto it = memo_.find(*src); it != memo_.end()) return it->second;
    Col c = output_color(src->first, src->second, depth);
    memo_.emplace(*src, c);
    return c;
  }

  // The colour at output `out` of node j.
  Col output_color(size_t j, const std::string& out, size_t depth) const {
    Col c;
    const NodeIn& n = m_.nodes[j];
    if (kind(j) == Kind::kTexImage && out == "Color") {
      c.b = {1.0, 1.0, 1.0};
      c.tex = texture(j);
      return c;
    }
    if (kind(j) == Kind::kMix && out == "Result_Color" && n.data_type == "RGBA") {
      if (n.blend_type != "MIX" && n.blend_type != "MULTIPLY") {
        throw Unsupported{"MAT-BAKE", "a Mix node blending by " + n.blend_type + " (MIX and MULTIPLY convert)"};
      }
      if (linked(j, "Factor_Float")) throw Unsupported{"MAT-BAKE", "a Mix node with a linked factor"};
      double f = value(j, "Factor_Float", 0, 0.5);
      if (flag(j, "clamp_factor", true)) f = std::clamp(f, 0.0, 1.0);
      const Col A = color(j, "A_Color", depth + 1), B = color(j, "B_Color", depth + 1);
      if (A.tex && B.tex && !(*A.tex == *B.tex)) throw Unsupported{"MAT-BAKE", "two textures in one colour"};
      c.tex = A.tex ? A.tex : B.tex;
      for (size_t k = 0; k < 3; ++k) {
        if (n.blend_type == "MIX") {   // (1 - f) A + f B
          c.a[k] = (1.0 - f) * A.a[k] + f * B.a[k];
          c.b[k] = (1.0 - f) * A.b[k] + f * B.b[k];
        } else {                       // A (1 - f + f B): exact while one of A, B has no texture term
          if (A.b[k] != 0.0 && B.b[k] != 0.0) throw Unsupported{"MAT-BAKE", "a texture multiplied by itself"};
          c.a[k] = A.a[k] * (1.0 - f + f * B.a[k]);
          c.b[k] = A.b[k] * (1.0 - f + f * B.a[k]) + f * A.a[k] * B.b[k];
        }
      }
      if (flag(j, "clamp_result")) {
        for (size_t k = 0; k < 3; ++k) {
          if (c.b[k] != 0.0 && (c.a[k] != 0.0 || c.b[k] > 1.0 || c.b[k] < 0.0)) {
            throw Unsupported{"MAT-BAKE", "a clamped Mix result that may clip a texture"};
          }
          if (c.b[k] == 0.0) c.a[k] = std::clamp(c.a[k], 0.0, 1.0);
        }
      }
      if (zero(c.b)) c.tex.reset();
      return c;
    }
    throw Unsupported{"MAT-BAKE", n.type + "." + out + " drives a colour (procedural or varying input; Cycles bakes "
                                                     "under lenient wait for BL-13)"};
  }

  // The UV maps ("": the default one) read by the Image Textures that reach node i (GEO-UV2, §4.6).
  std::vector<std::string> uv_maps(size_t i) const {
    std::set<std::string> maps;
    std::vector<size_t> todo{i};
    std::vector<bool> seen(m_.nodes.size(), false);
    seen[i] = true;
    while (!todo.empty()) {
      const size_t j = todo.back();
      todo.pop_back();
      if (kind(j) == Kind::kTexImage) {
        try {
          maps.insert(texture(j).map);
        } catch (const Unsupported&) {   // reported where it is used
        }
      }
      for (const size_t k : from_[j]) {
        if (!seen[k]) {
          seen[k] = true;
          todo.push_back(k);
        }
      }
    }
    return {maps.begin(), maps.end()};
  }

 private:
  [[noreturn]] void proto(const std::string& what) const {
    throw std::invalid_argument("BRG-PROTO: material " + m_.name + ": " + what);
  }

  Tex texture(size_t j) const {
    Tex t;
    const Tree& props = m_.nodes[j].props;
    const Tree* im = props.kind() == Tree::Kind::kDict ? props.find("image") : nullptr;
    if (im == nullptr || im->kind() == Tree::Kind::kNull) {
      throw Unsupported{"MAT-BAKE", "an Image Texture without an image"};
    }
    if (im->kind() != Tree::Kind::kInt || im->as_int() < 0 || im->as_int() >= static_cast<int64_t>(images_.size())) {
      proto("an Image Texture's image is not an images[] index");
    }
    t.image = static_cast<int32_t>(im->as_int());
    const ImageIn& img = images_[static_cast<size_t>(t.image)];
    t.extension = prop(j, "extension", img.extension.empty() ? "REPEAT" : img.extension);
    if (prop(j, "projection", "FLAT") != "FLAT") {
      throw Unsupported{"MAT-BAKE", "an Image Texture with " + prop(j, "projection", "") + " projection"};
    }
    if (t.extension != "REPEAT" && t.extension != "MIRROR" && t.extension != "EXTEND" && t.extension != "CLIP") {
      throw Unsupported{"MAT-BAKE", "an Image Texture with extension " + t.extension};
    }
    // The vector: the default UV map when unlinked; else Mappings over a UV Map or Texture Coordinate UV output.
    Affine m{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    std::vector<size_t> mappings;
    auto src = source(j, "Vector");
    while (src && kind(src->first) == Kind::kMapping) {
      if (mappings.size() > m_.nodes.size()) proto("a link cycle");
      mappings.push_back(src->first);
      src = source(src->first, "Vector");
    }
    if (src) {
      const auto [k, out] = *src;
      const bool uv = (kind(k) == Kind::kTexCoord || kind(k) == Kind::kUvMap) && out == "UV";
      if (!uv) throw Unsupported{"MAT-BAKE", "texture coordinates from " + m_.nodes[k].type + "." + out};
      if (flag(k, "from_instancer")) throw Unsupported{"MAT-BAKE", "texture coordinates from the instancer"};
      if (kind(k) == Kind::kUvMap) t.map = prop(k, "uv_map", "");
    } else if (!mappings.empty()) {
      throw Unsupported{"MAT-BAKE", "a Mapping node without texture coordinates"};
    }
    for (auto it = mappings.rbegin(); it != mappings.rend(); ++it) {   // innermost first
      const size_t k = *it;
      for (const char* in : {"Location", "Rotation", "Scale"}) {
        if (linked(k, in)) throw Unsupported{"MAT-BAKE", "a Mapping node with a linked " + std::string(in)};
      }
      const std::string type = prop(k, "vector_type", "POINT");
      const std::array<double, 3> L = vec3(k, "Location", 0.0), s = vec3(k, "Scale", 1.0);
      const std::array<double, 9> R = euler_xyz(vec3(k, "Rotation", 0.0));
      std::array<double, 9> A{};
      std::array<double, 3> c{};
      if (type == "POINT" || type == "VECTOR") {   // R (s * x) + L (VECTOR: no L)
        for (size_t r = 0; r < 3; ++r) {
          for (size_t q = 0; q < 3; ++q) A[3 * r + q] = R[3 * r + q] * s[q];
        }
        if (type == "POINT") c = L;
      } else if (type == "TEXTURE") {               // (R^T (x - L)) / s
        if (s[0] == 0.0 || s[1] == 0.0 || s[2] == 0.0) throw Unsupported{"MAT-BAKE", "a TEXTURE Mapping of scale 0"};
        for (size_t r = 0; r < 3; ++r) {
          for (size_t q = 0; q < 3; ++q) A[3 * r + q] = R[3 * q + r] / s[r];
          c[r] = -(A[3 * r] * L[0] + A[3 * r + 1] * L[1] + A[3 * r + 2] * L[2]);
        }
      } else {
        throw Unsupported{"MAT-BAKE", "a " + type + " Mapping (not affine)"};
      }
      m = compose(A, c, m);
    }
    t.st = {m[0], m[1], m[3], m[4], m[5], m[7]};   // (u, v, 0, 1): column z drops out
    return t;
  }

  const MaterialIn& m_;
  const std::vector<ImageIn>& images_;
  std::map<std::string, size_t> node_of_;
  std::map<std::pair<std::string, std::string>, const LinkIn*> link_to_;
  std::vector<std::vector<size_t>> from_;   // per node: the nodes linked into it
  mutable std::map<std::pair<size_t, std::string>, Col> memo_;
};

struct Leaf {
  size_t node;
  double w;
};

// The BSDF and Emission leaves under a shader socket with their weights (Add: 1 and 1; Mix Shader: 1 - f and f).
// It stops at the second BSDF or Emission, so a shader graph whose nodes feed one another more than once is refused
// after O(depth) visits instead of being walked as a tree.
void leaves(const Graph& g, size_t i, double w, std::vector<Leaf>& bsdf, std::vector<Leaf>& emit, size_t depth = 0) {
  if (depth > g.size()) g.cycle();
  switch (g.kind(i)) {
    case Kind::kPrincipled:
    case Kind::kDiffuse:
    case Kind::kEmission:
      (g.kind(i) == Kind::kEmission ? emit : bsdf).push_back({i, w});
      if (bsdf.size() > 1 || emit.size() > 1) throw Unsupported{"MAT-SHADER", "more than one BSDF or Emission"};
      return;
    case Kind::kAddShader:
    case Kind::kMixShader: {
      double f = 0.5;
      if (g.kind(i) == Kind::kMixShader) {
        if (g.linked(i, "Fac")) throw Unsupported{"MAT-BAKE", "a Mix Shader with a linked factor"};
        f = std::clamp(g.value(i, "Fac", 0, 0.5), 0.0, 1.0);
      }
      const auto a = g.source(i, "Shader"), b = g.source(i, "Shader_001");
      if (!a || !b) throw Unsupported{"MAT-SHADER", g.node(i).type + " with an unconnected shader input"};
      const bool add = g.kind(i) == Kind::kAddShader;
      leaves(g, a->first, w * (add ? 1.0 : 1.0 - f), bsdf, emit, depth + 1);
      leaves(g, b->first, w * (add ? 1.0 : f), bsdf, emit, depth + 1);
      return;
    }
    default:
      throw Unsupported{"MAT-SHADER", g.node(i).type + " (§4.5 converts Principled, Diffuse and Emission, combined by "
                                                       "Add or Mix Shader; glass, refraction, transparent, holdout and "
                                                       "volume are refused)"};
  }
}

std::string fmt(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.4g", v);
  return buf;
}

// A uint8 .npy (NumPy's format 1.0 to 3.0: the magic, a little-endian header length, the header dict {'descr',
// 'fortran_order', 'shape'}, then the data), read whole and in C order; the file is closed before it returns.
struct Npy {
  std::vector<uint8_t> data;
  std::vector<size_t> shape;
};

Npy read_npy_u8(const std::string& path, const std::string& where) {
  const auto bad = [&](const std::string& what) {
    return std::invalid_argument("BRG-PROTO: " + where + ": " + path + ": " + what);
  };
  std::ifstream f(path, std::ios::binary);
  if (!f) throw bad("cannot be opened");
  unsigned char pre[12];
  f.read(reinterpret_cast<char*>(pre), 10);
  if (f.gcount() != 10 || std::memcmp(pre, "\x93" "NUMPY", 6) != 0) throw bad("not a .npy file");
  size_t len = 0;
  if (pre[6] == 1) {
    len = pre[8] | static_cast<size_t>(pre[9]) << 8;
  } else if (pre[6] == 2 || pre[6] == 3) {
    f.read(reinterpret_cast<char*>(pre + 10), 2);
    if (f.gcount() != 2) throw bad("a truncated .npy header");
    len = pre[8] | static_cast<size_t>(pre[9]) << 8 | static_cast<size_t>(pre[10]) << 16 |
          static_cast<size_t>(pre[11]) << 24;
  } else {
    throw bad(".npy format version " + std::to_string(pre[6]) + " (1 to 3 are read)");
  }
  if (len > (size_t{1} << 20)) throw bad("a .npy header longer than 1 MiB");
  std::string h(len, '\0');
  f.read(h.data(), static_cast<std::streamsize>(len));
  if (static_cast<size_t>(f.gcount()) != len) throw bad("a truncated .npy header");
  const auto after = [&](const char* key) {   // the first non-blank character after 'key':
    for (const char q : {'\'', '"'}) {
      const std::string k = q + std::string(key) + q;
      if (size_t p = h.find(k); p != std::string::npos && (p = h.find(':', p + k.size())) != std::string::npos) {
        if ((p = h.find_first_not_of(" \t", p + 1)) != std::string::npos) return p;
      }
    }
    throw bad(std::string("a .npy header without ") + key);
  };
  size_t p = after("descr");
  const size_t e = h[p] == '\'' || h[p] == '"' ? h.find(h[p], p + 1) : std::string::npos;
  if (e == std::string::npos) throw bad("a .npy descr that is not a string");
  const std::string descr = h.substr(p + 1, e - p - 1);
  if (descr != "|u1" && descr != "<u1" && descr != ">u1" && descr != "=u1" && descr != "u1") {
    throw bad("pixels of dtype " + descr + ", not uint8");
  }
  p = after("fortran_order");
  const bool fortran = h.compare(p, 4, "True") == 0;
  if (!fortran && h.compare(p, 5, "False") != 0) throw bad("a .npy fortran_order that is not True or False");
  p = after("shape");
  const size_t close = h[p] == '(' ? h.find(')', p) : std::string::npos;
  if (close == std::string::npos) throw bad("a .npy shape that is not a tuple");
  Npy out;
  size_t n = 1;
  for (size_t q = h.find_first_not_of(" \t,", p + 1); q < close; q = h.find_first_not_of(" \t,", q)) {
    size_t d = 0;
    if (h[q] < '0' || h[q] > '9') throw bad("a .npy shape that is not a tuple of integers");
    for (; q < close && h[q] >= '0' && h[q] <= '9'; ++q) {
      d = 10 * d + static_cast<size_t>(h[q] - '0');
      if (d > (size_t{1} << 31)) throw bad("a .npy dimension above 2^31");
    }
    if (d != 0 && n > (size_t{1} << 36) / d) throw bad("more than 2^36 pixel bytes");
    out.shape.push_back(d);
    n *= d;
  }
  if (out.shape.size() != 2 && out.shape.size() != 3) throw bad("pixels are not (H, W) or (H, W, C)");
  std::vector<uint8_t> raw(n);
  f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(n));
  if (static_cast<size_t>(f.gcount()) != n) throw bad("a truncated .npy (fewer pixel bytes than its shape)");
  if (!fortran) {
    out.data = std::move(raw);
    return out;
  }
  const size_t H = out.shape[0], W = out.shape[1], C = out.shape.size() == 3 ? out.shape[2] : 1;
  out.data.resize(n);   // Fortran order: element (y, x, k) at y + H (x + W k)
  for (size_t y = 0; y < H; ++y) {
    for (size_t x = 0; x < W; ++x) {
      for (size_t k = 0; k < C; ++k) out.data[(y * W + x) * C + k] = raw[y + H * (x + W * k)];
    }
  }
  return out;
}

}  // namespace

MaterialOut convert_material(const MaterialIn& m, const std::vector<ImageIn>& images, const ConvertOptions& o) {
  MaterialOut out;
  const bool strict = o.policy == Policy::kStrict;
  // A code that is refused under strict and becomes `lenient` (approx or warn) under lenient.
  const auto item = [&](const char* code, const char* lenient, const std::string& msg, double bound = 0.0,
                        BoundKind kind = BoundKind::kNone) {
    out.items.push_back({code, strict || lenient == nullptr ? "refuse" : lenient, m.name, msg, bound, kind});
  };
  if (!m.has_graph) {
    item("MAT-SHADER", nullptr, "the material has no node tree");
    return out;
  }
  try {
    const Graph g(m, images);
    // The Cycles output (probe m2): the Material Outputs with target CYCLES if there are any, else those with ALL;
    // among them the active one, else the only one. Several inactive ones are refused: Cycles takes the first in node
    // order, which the exchange does not promise to keep.
    std::vector<size_t> outs;
    for (const char* tier : {"CYCLES", "ALL"}) {
      for (size_t i = 0; i < m.nodes.size(); ++i) {
        if (g.kind(i) == Kind::kOutput && g.prop(i, "target", "ALL") == tier) outs.push_back(i);
      }
      if (!outs.empty()) break;
    }
    const auto active = std::find_if(outs.begin(), outs.end(), [&](size_t i) { return g.flag(i, "is_active_output"); });
    if (outs.empty()) throw Unsupported{"MAT-SHADER", "no Material Output for Cycles (target CYCLES or ALL)"};
    if (active == outs.end() && outs.size() > 1) {
      throw Unsupported{"MAT-SHADER", "several Material Outputs for Cycles and none active"};
    }
    const size_t output = active != outs.end() ? *active : outs.front();
    if (g.linked(output, "Volume")) throw Unsupported{"MAT-SHADER", "a volume shader"};
    if (g.linked(output, "Displacement")) {
      throw Unsupported{"MAT-NORMAL", "displacement (refused under both policies: no bound)"};
    }
    out.uv_maps = g.uv_maps(output);
    const auto surface = g.source(output, "Surface");
    if (!surface) throw Unsupported{"MAT-SHADER", "nothing is connected to the Surface output"};
    std::vector<Leaf> bsdf, emit;
    leaves(g, surface->first, 1.0, bsdf, emit);
    if (bsdf.size() > 1 || emit.size() > 1) throw Unsupported{"MAT-SHADER", "more than one BSDF or Emission"};
    const double k = o.lm_per_w.value_or(1.0);
    std::array<double, 3> emissive{};
    const auto add_emission = [&](size_t i, const char* colour, const char* strength, double w) {
      if (g.linked(i, strength)) throw Unsupported{"MAT-BAKE", "a linked emission strength"};
      const double s = g.value(i, strength, 0, 0.0);
      if (s == 0.0) return;
      const Col c = g.color(i, colour);
      if (c.tex) {
        throw Unsupported{"MAT-BAKE", "a textured emission (C1 emissive is constant; baked shading after S1a, H4)"};
      }
      for (size_t q = 0; q < 3; ++q) emissive[q] += w * k * s * c.a[q];
    };
    Col base;
    if (!bsdf.empty()) {
      const size_t b = bsdf.front().node;
      const double w = bsdf.front().w;
      const bool principled = g.kind(b) == Kind::kPrincipled;
      base = g.color(b, principled ? "Base Color" : "Color");
      for (size_t q = 0; q < 3; ++q) {
        base.a[q] *= w;
        base.b[q] *= w;
      }
      if (g.linked(b, "Normal")) item("MAT-NORMAL", "warn", "a normal or bump map (no bound; ignored under lenient)");
      const char* rough_in = principled ? "Diffuse Roughness" : "Roughness";
      if (g.input(b, rough_in) != nullptr && (g.linked(b, rough_in) || g.value(b, rough_in) > 0.0)) {
        const bool lin = g.linked(b, rough_in);
        const double r = lin ? 1.0 : g.value(b, rough_in);
        const double lo = r <= kDiffRoughMin[0][0] ? kDiffRoughMin[0][1] : kDiffRoughMin[1][1];
        item("MAT-DIFFROUGH", "approx",
             std::string(rough_in) + (lin ? " is linked" : " " + fmt(r)) + " > 0: Lambert approximation, sampled "
             "ratio " + fmt(lo) + "-0.91 (R2 c1, roughness 0.5-1.0: 0.26-0.91)", lo, BoundKind::kSampled);
      }
      if (principled) {
        const bool m_lin = g.linked(b, "Metallic"), s_lin = g.linked(b, "Specular IOR Level");
        const double metal = g.value(b, "Metallic"), spec = g.value(b, "Specular IOR Level", 0, 0.5);
        const double rough = g.value(b, "Roughness", 0, 0.5);
        if (m_lin || metal > 0.0) {
          item("MAT-SPEC-UNB", "approx", "Metallic " + (m_lin ? std::string("linked") : fmt(metal)) +
                                             ": no measured bound (Lambert under lenient)");
        } else if (s_lin || (spec > 0.0 && (g.linked(b, "Roughness") || rough < kSpecRatio[0][0]))) {
          item("MAT-SPEC-UNB", "approx", "Specular IOR Level " + (s_lin ? std::string("linked") : fmt(spec)) +
                                             " with roughness below 0.3 or linked: no measured bound");
        } else if (spec > 0.0) {
          size_t band = 0;
          while (band + 1 < 3 && rough >= kSpecRatio[band + 1][0]) ++band;
          out.items.push_back({"MAT-SPEC", "warn", m.name,
                               "Specular IOR Level " + fmt(spec) + ", roughness " + fmt(rough) +
                                   ": largest sampled Principled/Lambert ratio " + fmt(kSpecRatio[band][1]) +
                                   " at roughness " + fmt(kSpecRatio[band][0]) +
                                   " (0.854-1.71 away from the mirror configurations; CV E5, Specular IOR Level 0.5)",
                               kSpecRatio[band][1], BoundKind::kSampled});
        }
        if (g.linked(b, "Alpha") || g.value(b, "Alpha", 0, 1.0) < 1.0) {
          item("MAT-ALPHA", "approx", "alpha below 1 or linked (opaque under lenient)");
        }
        std::string layers;
        for (const char* in : {"Subsurface Weight", "Transmission Weight", "Coat Weight", "Sheen Weight"}) {
          if (g.input(b, in) != nullptr && (g.linked(b, in) || g.value(b, in) > 0.0)) {
            layers += (layers.empty() ? "" : ", ") + std::string(in);
          }
        }
        if (!layers.empty()) item("MAT-LAYER", "warn", layers + " > 0 (ignored under lenient)");
        std::string other;
        for (const SocketIn& s : g.node(b).inputs) {
          if (!s.enabled || kPrincipledKnown.count(s.identifier)) continue;
          bool differs = g.linked(b, s.identifier) || s.n != s.n_def;
          for (int q = 0; !differs && q < s.n; ++q) differs = s.value[static_cast<size_t>(q)] != s.def[static_cast<size_t>(q)];
          if (differs) other += (other.empty() ? "" : ", ") + s.identifier;
        }
        if (!other.empty()) item("MAT-INPUT", "warn", other + " differ from their defaults (ignored under lenient)");
        add_emission(b, "Emission Color", "Emission Strength", w);
      } else {
        std::string other;
        for (const SocketIn& s : g.node(b).inputs) {
          if (!s.enabled || s.identifier == "Color" || s.identifier == "Roughness" || s.identifier == "Normal" ||
              s.identifier == "Weight") {
            continue;
          }
          bool differs = g.linked(b, s.identifier) || s.n != s.n_def;
          for (int q = 0; !differs && q < s.n; ++q) differs = s.value[static_cast<size_t>(q)] != s.def[static_cast<size_t>(q)];
          if (differs) other += (other.empty() ? "" : ", ") + s.identifier;
        }
        if (!other.empty()) item("MAT-INPUT", "warn", other + " differ from their defaults (ignored under lenient)");
      }
    }
    if (!emit.empty()) add_emission(emit.front().node, "Color", "Strength", emit.front().w);
    // base_color = constant x texture (§4.5): a + b T converts when a or b vanishes.
    if (base.tex && !zero(base.a)) throw Unsupported{"MAT-BAKE", "a texture mixed with a constant (not a product)"};
    const std::array<double, 3> bc = base.tex ? base.b : base.a;
    for (size_t q = 0; q < 3; ++q) {
      if (!(bc[q] >= 0.0) || !std::isfinite(bc[q]) || !(emissive[q] >= 0.0) || !std::isfinite(emissive[q])) {
        throw Unsupported{"MAT-INPUT", "a negative or non-finite colour"};
      }
      out.mat.base_color[q] = static_cast<float>(bc[q]);
      out.mat.emissive[q] = static_cast<float>(emissive[q]);
    }
    if (base.tex) {
      const Tex& t = *base.tex;
      out.image = t.image;
      out.mirror = t.extension == "MIRROR";
      out.unit_uv = t.extension == "EXTEND" || t.extension == "CLIP";
      out.uv_map = t.map;
      const double h = out.mirror ? 0.5 : 1.0;   // (u', v') = (s, 1 - t), halved on the MIRROR tile
      out.uv = {h * t.st[0], h * t.st[1], h * t.st[2], -h * t.st[3], -h * t.st[4], h * (1.0 - t.st[5])};
    }
  } catch (const Unsupported& u) {   // outside the accepted graphs: refused under both policies
    out.items.push_back({u.code, "refuse", m.name, u.message, 0.0, BoundKind::kNone});
  }
  return out;
}

TextureResult convert_texture(const ImageIn& im, int32_t index, bool mirror, Policy policy) {
  TextureResult r;
  const auto refuse = [&](const char* code, const std::string& msg) {
    r.items.push_back(ReportItem{code, "refuse", im.id, msg, 0.0, BoundKind::kNone});
    return r;
  };
  if (im.is_float) return refuse("TEX-CS", "a float image (refused until C1's texture_linear)");
  const bool srgb = im.colorspace == "sRGB", data = im.colorspace == "Non-Color";
  const bool linear = im.colorspace == "Linear Rec.709" || data;
  if (!srgb && !linear) {
    return refuse("TEX-CS", "colour space " + im.colorspace + " (sRGB, Linear Rec.709 and Non-Color convert)");
  }
  const std::string where = "images[" + std::to_string(index) + "]";
  if (im.path.empty()) {   // written without pixels (file null): a missing file, or a tiled, sequence or movie image
    return refuse("TEX-CS", "no pixels: a missing image file, or a tiled (UDIM), sequence or movie image");
  }
  const Npy px = read_npy_u8(im.path, where);
  const size_t h = px.shape[0], w = px.shape[1], c = px.shape.size() == 3 ? px.shape[2] : 1;
  if ((im.w >= 0 && static_cast<size_t>(im.w) != w) || (im.h >= 0 && static_cast<size_t>(im.h) != h) ||
      (im.c >= 0 && static_cast<size_t>(im.c) != c)) {
    throw std::invalid_argument("BRG-PROTO: " + where + ": pixels do not match width, height and channels (" +
                                im.id + ": " + std::to_string(h) + " x " + std::to_string(w) + " x " +
                                std::to_string(c) + ")");
  }
  if (w < 1 || h < 1 || c < 1 || c > 4) {
    throw std::invalid_argument("BRG-PROTO: " + where + ": pixels are not H x W x C with H, W >= 1 and C in 1..4");
  }
  // Alpha (probe m2): the Color output is alpha-associated in the byte domain for sRGB and Linear images of
  // alpha_mode STRAIGHT; raw for CHANNEL_PACKED, NONE and Non-Color. When some alpha is below 1, any other mode (or
  // none written) is MAT-ALPHA.
  bool associate = false;
  if ((c == 2 || c == 4) && !data) {
    bool partial = false;
    for (size_t i = c - 1; i < px.data.size() && !partial; i += c) partial = px.data[i] != 255;
    const std::string& mode = im.alpha_mode;
    if (partial && mode == "STRAIGHT") {
      associate = true;
    } else if (partial && mode != "CHANNEL_PACKED" && mode != "NONE") {
      const std::string msg = "texels with alpha < 1 and alpha_mode " + (mode.empty() ? "unwritten" : mode) +
                              ": Cycles' Color output is alpha-associated for STRAIGHT and raw for CHANNEL_PACKED and "
                              "NONE (probe m2)";
      if (policy == Policy::kStrict) return refuse("MAT-ALPHA", msg);
      r.items.push_back({"MAT-ALPHA", "approx", im.id, msg + "; associated as STRAIGHT, Blender's default", 0.0,
                         BoundKind::kNone});
      associate = true;
    }
  }
  uint8_t lut[256];
  for (int v = 0; v < 256; ++v) {   // linear code -> nearest sRGB code (IEC 61966-2-1 encoding)
    const double x = v / 255.0;
    const double e = x <= 0.0031308 ? 12.92 * x : 1.055 * std::pow(x, 1.0 / 2.4) - 0.055;
    lut[v] = srgb ? static_cast<uint8_t>(v) : static_cast<uint8_t>(std::lround(std::clamp(e, 0.0, 1.0) * 255.0));
  }
  if (linear) {
    r.items.push_back({"TEX-CS", "warn", im.id, im.colorspace + " byte image re-encoded to sRGB codes (<= 2.63 % in "
                                                "linear light, CV E6)", kLinearToSrgbErr, BoundKind::kMeasured});
  }
  TextureOut t;
  t.image = index;
  t.mirror = mirror;
  const size_t W = mirror ? 2 * w : w, H = mirror ? 2 * h : h;
  t.width = static_cast<int32_t>(W);
  t.height = static_cast<int32_t>(H);
  t.rgb.resize(W * H * 3);
  for (size_t y = 0; y < H; ++y) {
    const size_t ty = y % h, sy = (y / h) % 2 == 0 ? ty : h - 1 - ty;   // the lower half mirrors in t
    const uint8_t* row = px.data.data() + (h - 1 - sy) * w * c;          // bottom-up input, top-down output
    for (size_t x = 0; x < W; ++x) {
      const size_t tx = x % w, sx = (x / w) % 2 == 0 ? tx : w - 1 - tx;  // the right half mirrors in s
      const uint8_t* p = row + sx * c;
      uint8_t* q = &t.rgb[(y * W + x) * 3];
      for (size_t k = 0; k < 3; ++k) {
        const unsigned v = p[c >= 3 ? k : 0];
        q[k] = lut[associate ? v * p[c - 1] / 255u : v];   // floor(c a / 255)
      }
    }
  }
  r.tex = std::move(t);
  return r;
}

}  // namespace balla::scene::blender
