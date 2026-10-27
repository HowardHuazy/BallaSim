#include "scene/assets/ply.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "scene/errors.h"
#include "scene/npo.h"

namespace balla::scene::assets {

namespace {

using T = PlyType;

T type_of(const std::string& s) {
  static const std::map<std::string, T> types = {
      {"char", T::kI1},   {"int8", T::kI1},  {"uchar", T::kU1}, {"uint8", T::kU1},    {"short", T::kI2},
      {"int16", T::kI2},  {"ushort", T::kU2}, {"uint16", T::kU2}, {"int", T::kI4},     {"int32", T::kI4},
      {"uint", T::kU4},   {"uint32", T::kU4}, {"float", T::kF4},  {"float32", T::kF4}, {"double", T::kF8},
      {"float64", T::kF8}};
  const auto it = types.find(s);
  if (it == types.end()) throw std::invalid_argument("'" + s + "'");   // the Python's KeyError
  return it->second;
}

size_t size_of(T t) {
  switch (t) {
    case T::kI1:
    case T::kU1:
      return 1;
    case T::kI2:
    case T::kU2:
      return 2;
    case T::kI4:
    case T::kU4:
    case T::kF4:
      return 4;
    case T::kF8:
      return 8;
  }
  return 0;
}

bool is_float(T t) { return t == T::kF4 || t == T::kF8; }

struct Prop {
  std::string name;
  bool list = false;
  T count_type = T::kU1, type = T::kF4;
};

struct Element {
  std::string name;
  int64_t count = 0;
  std::vector<Prop> props;
};

using Column = PlyColumn;
using Elem = PlyElement;

// Python's str.split() whitespace for latin-1 text.
bool py_space(unsigned char c) {
  return c == ' ' || (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x1f) || c == 0x85 || c == 0xa0;
}

std::vector<std::string> split_words(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && py_space(static_cast<unsigned char>(s[i]))) ++i;
    const size_t j = i;
    while (i < s.size() && !py_space(static_cast<unsigned char>(s[i]))) ++i;
    if (i > j) out.push_back(s.substr(j, i - j));
  }
  return out;
}

// str.splitlines() of latin-1 text.
std::vector<std::string> split_lines(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0, start = 0;
  while (i < s.size()) {
    const auto c = static_cast<unsigned char>(s[i]);
    if (c == '\n' || c == '\r' || c == 0x0b || c == 0x0c || c == 0x1c || c == 0x1d || c == 0x1e || c == 0x85) {
      out.push_back(s.substr(start, i - start));
      i += (c == '\r' && i + 1 < s.size() && s[i + 1] == '\n') ? 2 : 1;
      start = i;
    } else {
      ++i;
    }
  }
  if (start < s.size()) out.push_back(s.substr(start));
  return out;
}

// NumPy's string to number cast for one PLY word: float types through Python's float() (a double, then rounded to
// float32 for 'float'), integer types through int().
double parse_word(const std::string& w, T t) {
  const char* b = w.data();
  const char* e = w.data() + w.size();
  if (b != e && *b == '+') ++b;
  if (is_float(t)) {
    double d = 0;
    const auto r = std::from_chars(b, e, d);
    if (r.ec != std::errc() || r.ptr != e) throw std::invalid_argument("could not convert string to float: '" + w + "'");
    return t == T::kF4 ? static_cast<double>(static_cast<float>(d)) : d;
  }
  int64_t i = 0;
  const auto r = std::from_chars(b, e, i);
  if (r.ec != std::errc() || r.ptr != e) throw std::invalid_argument("invalid literal for int() with base 10: '" + w + "'");
  return static_cast<double>(i);
}

template <class U>
U load_le(const unsigned char* p, bool big) {
  U v;
  std::memcpy(&v, p, sizeof(U));
  if (big) {
    unsigned char tmp[sizeof(U)];
    std::memcpy(tmp, &v, sizeof(U));
    std::reverse(tmp, tmp + sizeof(U));
    std::memcpy(&v, tmp, sizeof(U));
  }
  return v;
}

double read_value(const unsigned char* p, T t, bool big) {
  switch (t) {
    case T::kI1:
      return static_cast<double>(static_cast<int8_t>(p[0]));
    case T::kU1:
      return static_cast<double>(p[0]);
    case T::kI2:
      return static_cast<double>(load_le<int16_t>(p, big));
    case T::kU2:
      return static_cast<double>(load_le<uint16_t>(p, big));
    case T::kI4:
      return static_cast<double>(load_le<int32_t>(p, big));
    case T::kU4:
      return static_cast<double>(load_le<uint32_t>(p, big));
    case T::kF4:
      return static_cast<double>(load_le<float>(p, big));
    case T::kF8:
      return load_le<double>(p, big);
  }
  return 0;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::invalid_argument(path + ": cannot open");
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

}  // namespace

PlyElements read_ply(const std::string& path) {
  const std::string raw = read_file(path);
  size_t pos = 0;
  auto readline = [&raw, &pos]() {
    const size_t nl = raw.find('\n', pos);
    const size_t end = nl == std::string::npos ? raw.size() : nl + 1;
    std::string line = raw.substr(pos, end - pos);
    pos = end;
    return line;
  };
  {
    const std::vector<std::string> w = split_words(readline());
    if (w.size() != 1 || w[0] != "ply") throw AssetError(path + ": not a PLY file");
  }
  std::string fmt;
  std::vector<Element> elements;
  for (;;) {
    const std::string line = readline();
    if (line.empty()) throw AssetError(path + ": truncated PLY header");
    const std::vector<std::string> w = split_words(line);
    if (w.empty() || w[0] == "comment" || w[0] == "obj_info") continue;
    if (w[0] == "format") {
      fmt = w.at(1);
    } else if (w[0] == "element") {
      elements.push_back({w.at(1), std::stoll(w.at(2)), {}});
    } else if (w[0] == "property") {
      if (elements.empty()) throw std::invalid_argument(path + ": property before any element");
      const std::string& name = w.at(1) == "list" ? w.at(4) : w.at(2);
      for (const Prop& q : elements.back().props) {
        if (q.name == name) throw AssetError(path + ": property " + name + " repeated in element " + elements.back().name);
      }
      if (w.at(1) == "list") {
        elements.back().props.push_back({w.at(4), true, type_of(w.at(2)), type_of(w.at(3))});
      } else {
        elements.back().props.push_back({w.at(2), false, T::kU1, type_of(w.at(1))});
      }
    } else if (w[0] == "end_header") {
      break;
    }
  }
  std::map<std::string, Elem> out;
  if (fmt == "ascii") {
    std::string body = raw.substr(pos);
    std::vector<std::string> lines;
    for (std::string& ln : split_lines(body)) {
      if (!split_words(ln).empty()) lines.push_back(std::move(ln));
    }
    size_t row0 = 0;
    for (const Element& el : elements) {
      Elem cols;
      for (const Prop& p : el.props) cols[p.name] = {p.type, p.list, false, {}, p.list ? std::vector<size_t>{0} : std::vector<size_t>{}};
      for (int64_t r = 0; r < el.count; ++r) {
        if (row0 + static_cast<size_t>(r) >= lines.size()) throw std::invalid_argument(path + ": list index out of range");
        const std::vector<std::string> words = split_words(lines[row0 + static_cast<size_t>(r)]);
        size_t k = 0;
        for (const Prop& p : el.props) {
          Column& c = cols[p.name];
          if (p.list) {
            const double cnt = parse_word(words.at(k), T::kI4);
            if (!(cnt >= 0) || cnt > static_cast<double>(words.size())) throw AssetError(path + ": list count out of range");
            const auto n = static_cast<size_t>(cnt);
            for (size_t i = 0; i < n; ++i) c.values.push_back(parse_word(words.at(k + 1 + i), p.type));
            c.offsets.push_back(c.values.size());
            k += 1 + n;
          } else {
            c.values.push_back(parse_word(words.at(k), p.type));
            k += 1;
          }
        }
      }
      for (const Prop& p : el.props) {
        Column& c = cols[p.name];
        if (p.list && c.rows() > 0) {
          bool same = true;
          for (size_t r = 1; r < c.rows(); ++r) same = same && c.len(r) == c.len(0);
          c.stacked = same;
        }
      }
      row0 += static_cast<size_t>(el.count);
      out[el.name] = std::move(cols);
    }
    return out;
  }
  if (fmt != "binary_little_endian" && fmt != "binary_big_endian") {
    throw AssetError(path + ": unsupported PLY format '" + fmt + "'");
  }
  const bool big = fmt == "binary_big_endian";
  const auto* data = reinterpret_cast<const unsigned char*>(raw.data());
  size_t off = pos;
  auto need = [&](size_t n) {
    if (off + n > raw.size()) throw std::invalid_argument(path + ": buffer is smaller than requested size");
  };
  for (const Element& el : elements) {
    Elem cols;
    size_t nlists = 0;
    for (const Prop& p : el.props) {
      cols[p.name] = {p.type, p.list, false, {}, p.list ? std::vector<size_t>{0} : std::vector<size_t>{}};
      nlists += p.list ? 1 : 0;
    }
    for (int64_t r = 0; r < el.count; ++r) {
      for (const Prop& p : el.props) {
        Column& c = cols[p.name];
        if (p.list) {
          need(size_of(p.count_type));
          const double cnt = read_value(data + off, p.count_type, big);
          off += size_of(p.count_type);
          if (!(cnt >= 0) || cnt > static_cast<double>((raw.size() - off) / size_of(p.type))) {
            throw AssetError(path + ": list count out of range");
          }
          const auto n = static_cast<size_t>(cnt);
          for (size_t i = 0; i < n; ++i) c.values.push_back(read_value(data + off + i * size_of(p.type), p.type, big));
          off += n * size_of(p.type);
          c.offsets.push_back(c.values.size());
        } else {
          need(size_of(p.type));
          c.values.push_back(read_value(data + off, p.type, big));
          off += size_of(p.type);
        }
      }
    }
    // The Python reads an element as one structured block (a 2-D list array) when it has at most one list property
    // and every row's count equals the first's; otherwise row by row into a list.
    if (el.count > 0 && nlists <= 1) {
      for (const Prop& p : el.props) {
        if (!p.list) continue;
        Column& c = cols[p.name];
        bool same = true;
        for (size_t r = 1; r < c.rows(); ++r) same = same && c.len(r) == c.len(0);
        c.stacked = same;
      }
    }
    out[el.name] = std::move(cols);
  }
  if (off != raw.size()) {
    throw AssetError(path + ": " + std::to_string(raw.size() - off) + " unparsed bytes after the last element");
  }
  return out;
}

namespace {

const Column* first_of(const Elem& e, std::initializer_list<const char*> names) {
  for (const char* n : names) {
    const auto it = e.find(n);
    if (it != e.end()) return &it->second;
  }
  return nullptr;
}

// ply._triangulate: fans; a 2-D face array emits every face's first fan triangle, then every second one, ...; a list
// emits each face's fan in turn (faces of fewer than 3 vertices give none).
std::vector<int64_t> triangulate(const Column& faces) {
  std::vector<int64_t> out;
  const size_t n = faces.rows();
  if (faces.stacked) {
    const size_t k = n ? faces.len(0) : 0;
    if (k < 3) throw AssetError("PLY faces with fewer than 3 vertices");
    for (size_t fan = 1; fan + 1 < k; ++fan) {
      for (size_t f = 0; f < n; ++f) {
        const double* v = &faces.values[faces.offsets[f]];
        out.insert(out.end(), {static_cast<int64_t>(v[0]), static_cast<int64_t>(v[fan]), static_cast<int64_t>(v[fan + 1])});
      }
    }
    return out;
  }
  for (size_t f = 0; f < n; ++f) {
    const double* v = &faces.values[faces.offsets[f]];
    for (size_t i = 1; i + 1 < faces.len(f); ++i) {
      out.insert(out.end(), {static_cast<int64_t>(v[0]), static_cast<int64_t>(v[i]), static_cast<int64_t>(v[i + 1])});
    }
  }
  return out;
}


// geometry.srgb_to_linear / linear_to_srgb of one value (NumPy's float64 power is glibc's pow here).
double srgb_to_linear(double c) { return c <= 0.04045 ? c / 12.92 : npo::py_pow((c + 0.055) / 1.055, 2.4); }

double linear_to_srgb(double c) {
  c = c < 0.0 ? 0.0 : c > 1.0 ? 1.0 : c;
  return c <= 0.0031308 ? 12.92 * c : 1.055 * npo::py_pow(c, 1.0 / 2.4) - 0.055;
}

std::vector<double> col_values(const Elem& e, const std::string& name, const std::string& path) {
  const auto it = e.find(name);
  if (it == e.end()) throw AssetError(path + ": vertex element lacks '" + name + "'");
  if (it->second.list) throw AssetError(path + ": vertex property " + name + " must be a scalar property");
  return it->second.values;
}

// A scalar property holding exactly n values (the Python's np.stack refused anything else).
const Column& scalar_col(const Elem& e, const char* name, size_t n, const std::string& path) {
  const auto it = e.find(name);
  if (it == e.end()) throw AssetError(path + ": missing property " + name);
  if (it->second.list || it->second.values.size() != n) {
    throw AssetError(path + ": property " + name + " must be a scalar with one value per row");
  }
  return it->second;
}

Asset artec(const std::string& path, const std::map<std::string, Elem>& els, const std::vector<double>& pos,
            const Column& faces, const std::vector<float>* file_normals) {
  const Elem& tv = els.at("multi_texture_vertex");
  const Elem& tf = els.at("multi_texture_face");
  const Column& corner = tf.at("texture_vertex_indices");
  const size_t nf = faces.rows();
  if (!faces.stacked || (nf && faces.len(0) != 3) || !corner.stacked || corner.rows() != nf ||
      (nf && corner.len(0) != 3)) {
    throw AssetError(path + ": Artec faces must be triangles with one texture vertex per corner");
  }
  const size_t nv = pos.size() / 3;
  for (size_t i = 0; i < 3 * nf; ++i) {
    const double f = faces.values[i];
    if (!(f >= 0 && f < static_cast<double>(nv))) throw AssetError(path + ": face vertex index out of range");
  }
  const auto ftx_it = tf.find("tx");
  if (ftx_it == tf.end() || ftx_it->second.list) throw AssetError(path + ": multi_texture_face lacks a scalar tx");
  const Column& ftx = ftx_it->second;
  if (ftx.rows() != nf) throw AssetError(path + ": multi_texture_face count differs from the face count (AST-22)");
  const size_t ntv = tv.count("tx") ? tv.at("tx").values.size() : 0;
  const Column& vtx = scalar_col(tv, "tx", ntv, path);
  const Column& tu = scalar_col(tv, "u", ntv, path);
  const Column& tvv = scalar_col(tv, "v", ntv, path);
  for (const double t : corner.values) {
    if (!(t >= 0 && t < static_cast<double>(ntv))) throw AssetError(path + ": texture vertex index out of range");
  }
  if (std::any_of(ftx.values.begin(), ftx.values.end(), [](double x) { return x != 0; }) ||
      std::any_of(vtx.values.begin(), vtx.values.end(), [](double x) { return x != 0; })) {
    throw AssetError(path + ": more than one Artec atlas is not supported (AST-22)");
  }
  const Column& tn = scalar_col(tf, "tn", nf, path);
  std::vector<int64_t> seen(nf, 0);
  bool perm = tn.rows() == nf && nf > 0;
  for (size_t i = 0; i < tn.rows() && perm; ++i) {
    const auto t = static_cast<int64_t>(tn.values[i]);
    if (t < 0 || t >= static_cast<int64_t>(nf) || seen[static_cast<size_t>(t)]++) perm = false;
  }
  if (!perm) throw AssetError(path + ": multi_texture_face.tn is not a permutation of the face indices");
  std::vector<int64_t> corner_of_face(3 * nf);
  for (size_t i = 0; i < nf; ++i) {
    const auto f = static_cast<size_t>(tn.values[i]);
    for (size_t k = 0; k < 3; ++k) corner_of_face[3 * f + k] = static_cast<int64_t>(corner.values[3 * i + k]);
  }
  const std::string atlas = (std::filesystem::path(path).parent_path() / "surface_0.png").string();
  if (!std::filesystem::exists(atlas)) throw AssetError(path + ": Artec atlas surface_0.png is missing");
  // One output vertex per unique (position index, texture-vertex index) wedge, in np.unique's sorted order.
  std::vector<std::array<int64_t, 2>> pairs(3 * nf);
  std::vector<int64_t> fidx(3 * nf);
  for (size_t i = 0; i < 3 * nf; ++i) {
    fidx[i] = static_cast<int64_t>(faces.values[i]);
    pairs[i] = {fidx[i], corner_of_face[i]};
  }
  std::vector<size_t> order(pairs.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(), [&pairs](size_t a, size_t b) { return pairs[a] < pairs[b]; });
  std::vector<std::array<int64_t, 2>> uniq;
  std::vector<uint32_t> inv(pairs.size());
  for (size_t k = 0; k < order.size(); ++k) {
    if (k == 0 || pairs[order[k]] != pairs[order[k - 1]]) uniq.push_back(pairs[order[k]]);
    inv[order[k]] = static_cast<uint32_t>(uniq.size() - 1);
  }
  std::vector<float> normals;
  if (file_normals) {
    normals = *file_normals;
  } else {
    normals = geo::welded_normals(pos, fidx);
  }
  Asset a;
  geo::Mesh& m = a.mesh;
  for (const auto& [vi, ti] : uniq) {
    const auto v = static_cast<size_t>(vi), t = static_cast<size_t>(ti);
    for (size_t k = 0; k < 3; ++k) {
      m.positions.push_back(static_cast<float>(pos[3 * v + k]));
      m.normals.push_back(normals.at(3 * v + k));
    }
    m.uvs.push_back(static_cast<float>(tu.values[t]));
    m.uvs.push_back(static_cast<float>(tvv.values[t]));
  }
  m.indices = std::move(inv);
  m.has_normals = m.has_uvs = true;
  Material mat;
  mat.base_color = {1.0f, 1.0f, 1.0f};
  mat.texture = read_png_rgb(atlas);
  a.materials.push_back(std::move(mat));
  return a;
}

}  // namespace

Asset load_ply(const std::string& path, double scale, const std::string& uv_origin) {
  const PlyElements els = read_ply(path);
  if (!els.count("vertex") || !els.count("face")) {
    throw AssetError(path + ": needs vertex and face elements (point clouds are not renderable, DESIGN §5.2)");
  }
  const Elem& v = els.at("vertex");
  const std::vector<double> x = col_values(v, "x", path), y = col_values(v, "y", path), z = col_values(v, "z", path);
  const size_t nv = x.size();
  if (y.size() != nv || z.size() != nv) throw AssetError(path + ": vertex x, y and z differ in length");
  std::vector<double> pos(3 * nv);
  for (size_t i = 0; i < nv; ++i) {
    pos[3 * i] = x[i] * scale;
    pos[3 * i + 1] = y[i] * scale;
    pos[3 * i + 2] = z[i] * scale;
  }
  const Column* faces = first_of(els.at("face"), {"vertex_indices", "vertex_index"});
  if (!faces) throw AssetError(path + ": face element has no vertex_indices");
  if (!faces->list) throw AssetError(path + ": face vertex_indices must be a list property");
  std::vector<float> normals;
  const bool has_normals = v.count("nx") && v.count("ny") && v.count("nz");
  if (has_normals) {
    const Column &nx = scalar_col(v, "nx", nv, path), &ny = scalar_col(v, "ny", nv, path),
                 &nz = scalar_col(v, "nz", nv, path);
    normals.resize(3 * nv);
    for (size_t i = 0; i < nv; ++i) {
      normals[3 * i] = static_cast<float>(nx.values[i]);
      normals[3 * i + 1] = static_cast<float>(ny.values[i]);
      normals[3 * i + 2] = static_cast<float>(nz.values[i]);
    }
  }
  Tree info = Tree::dict();
  info["format"] = Tree::string("ply");
  info["scale"] = Tree::real(scale);
  if (els.count("multi_texture_face")) {
    info["layout"] = Tree::string("artec_multi_texture");
    info["normals"] = Tree::string(has_normals ? "file" : "angle-weighted on the welded mesh");
    Asset a = artec(path, els, pos, *faces, has_normals ? &normals : nullptr);
    a.info = std::move(info);
    return a;
  }
  Asset a;
  geo::Mesh& m = a.mesh;
  m.positions.resize(3 * nv);
  for (size_t i = 0; i < 3 * nv; ++i) m.positions[i] = static_cast<float>(pos[i]);
  for (const int64_t i : triangulate(*faces)) m.indices.push_back(static_cast<uint32_t>(i));
  if (has_normals) {
    m.normals = std::move(normals);
    m.has_normals = true;
  }
  const Column* u = first_of(v, {"u", "s", "texture_u"});
  const Column* vv = first_of(v, {"v", "t", "texture_v"});
  if (u && vv) {
    if (u->list || vv->list || u->values.size() != nv || vv->values.size() != nv) {
      throw AssetError(path + ": vertex UVs must be scalar properties with one value per vertex");
    }
    m.uvs.resize(2 * nv);
    for (size_t i = 0; i < nv; ++i) {
      m.uvs[2 * i] = static_cast<float>(u->values[i]);
      m.uvs[2 * i + 1] = static_cast<float>(uv_origin == "bottom" ? 1.0 - vv->values[i] : vv->values[i]);
    }
    m.has_uvs = true;
  }
  Material mat;
  if (v.count("red") && v.count("green") && v.count("blue")) {
    const Column* ch[3] = {&scalar_col(v, "red", nv, path), &scalar_col(v, "green", nv, path),
                           &scalar_col(v, "blue", nv, path)};
    // np.stack's result type: float64 with any double, or float32 with a 32-bit integer; float32 with float32 and
    // narrower integers (the clip, * 255, rint and / 255 then run in float32); an integer type otherwise.
    bool any_f4 = false, any_f8 = false, any_i32 = false;
    for (const Column* c : ch) {
      any_f4 = any_f4 || c->type == T::kF4;
      any_f8 = any_f8 || c->type == T::kF8;
      any_i32 = any_i32 || c->type == T::kI4 || c->type == T::kU4;
    }
    const bool fl = any_f4 || any_f8, f32 = any_f4 && !any_f8 && !any_i32;
    std::vector<std::array<double, 3>> rgb(nv);
    for (size_t i = 0; i < nv; ++i) {
      for (size_t k = 0; k < 3; ++k) {
        double c = ch[k]->values[i];
        if (f32) {
          float cf = static_cast<float>(c);
          cf = cf < 0.0f ? 0.0f : cf > 1.0f ? 1.0f : cf;
          c = static_cast<double>(std::nearbyint(cf * 255.0f));
        } else if (fl) {
          c = std::nearbyint((c < 0 ? 0.0 : c > 1 ? 1.0 : c) * 255.0);
        }
        rgb[i][k] = c;
      }
    }
    for (size_t i = 1; i < nv; ++i) {
      if (rgb[i] != rgb[0]) {
        std::vector<std::array<double, 3>> u2 = rgb;
        std::sort(u2.begin(), u2.end());
        const auto distinct = std::unique(u2.begin(), u2.end()) - u2.begin();
        throw AssetError(path + ": " + std::to_string(distinct) + " distinct vertex colours; C1 meshes carry no vertex "
                         "colours, so only a single colour throughout is supported (SCENE.md open issues)");
      }
    }
    if (nv == 0) throw std::invalid_argument("index 0 is out of bounds for axis 0 with size 0");
    for (size_t k = 0; k < 3; ++k) {
      if (std::isnan(rgb[0][k])) throw std::invalid_argument("cannot convert float NaN to integer");
    }
    Tree srgb = Tree::list();
    for (size_t k = 0; k < 3; ++k) srgb.push_back(Tree::integer(static_cast<int64_t>(rgb[0][k])));
    info["layout"] = Tree::string("uniform_vertex_color");
    info["vertex_color_srgb"] = std::move(srgb);
    for (size_t k = 0; k < 3; ++k) {
      const double c = f32 ? static_cast<double>(static_cast<float>(rgb[0][k]) / 255.0f) : rgb[0][k] / 255.0;
      mat.base_color[k] = static_cast<float>(srgb_to_linear(c));
    }
  } else {
    info["layout"] = Tree::string("plain");
    mat.base_color = {0.5f, 0.5f, 0.5f};
  }
  a.materials.push_back(std::move(mat));
  a.info = std::move(info);
  return a;
}

namespace {

std::vector<double> to_linear(const Image& tex) {
  std::vector<double> lin(tex.rgb.size());
  for (size_t i = 0; i < lin.size(); ++i) lin[i] = srgb_to_linear(static_cast<double>(tex.rgb[i]) / 255.0);
  return lin;
}

// One halving in linear light: np.pad(..., mode="edge") to even sides, then the 2 x 2 box ((tl + bl) + tr) + br,
// times 0.25 (assets._halve).
void halve(std::vector<double>& lin, int64_t& h, int64_t& w) {
  const int64_t h2 = (h + h % 2) / 2, w2 = (w + w % 2) / 2;
  auto px = [&](int64_t r, int64_t c, int64_t k) {
    return lin[static_cast<size_t>((std::min(r, h - 1) * w + std::min(c, w - 1)) * 3 + k)];
  };
  std::vector<double> out(static_cast<size_t>(h2 * w2 * 3));
  for (int64_t r = 0; r < h2; ++r) {
    for (int64_t c = 0; c < w2; ++c) {
      for (int64_t k = 0; k < 3; ++k) {
        out[static_cast<size_t>((r * w2 + c) * 3 + k)] =
            0.25 * (((px(2 * r, 2 * c, k) + px(2 * r + 1, 2 * c, k)) + px(2 * r, 2 * c + 1, k)) + px(2 * r + 1, 2 * c + 1, k));
      }
    }
  }
  lin = std::move(out);
  h = h2;
  w = w2;
}

}  // namespace

std::pair<Image, int> downsample(Image tex, int64_t max_size) {
  int levels = 0;
  std::vector<double> lin;
  int64_t h = tex.height, w = tex.width;
  while (std::max(tex.height, tex.width) > max_size) {
    if (lin.empty()) lin = to_linear(tex);
    halve(lin, h, w);
    tex.height = h;
    tex.width = w;
    tex.rgb.resize(lin.size());
    for (size_t i = 0; i < lin.size(); ++i) {
      tex.rgb[i] = static_cast<uint8_t>(std::nearbyint(linear_to_srgb(lin[i]) * 255.0));
    }
    ++levels;
  }
  return {std::move(tex), levels};
}

std::pair<std::optional<LinearImage>, int> downsample_linear(const Image& tex, int64_t max_size) {
  int64_t h = tex.height, w = tex.width;
  if (std::max(h, w) <= max_size) return {std::nullopt, 0};
  std::vector<double> lin = to_linear(tex);
  int levels = 0;
  while (std::max(h, w) > max_size) {
    halve(lin, h, w);
    ++levels;
  }
  LinearImage out;
  out.height = h;
  out.width = w;
  out.bits.resize(lin.size());
  for (size_t i = 0; i < lin.size(); ++i) out.bits[i] = binary16_bits(lin[i]);
  return {std::move(out), levels};
}

uint16_t binary16_bits(double x) {
  const unsigned sign = std::signbit(x) ? 0x8000u : 0u;
  const double a = std::fabs(x);
  if (std::isnan(x)) return static_cast<uint16_t>(sign | 0x7e00u);
  if (a >= 65520.0) return static_cast<uint16_t>(sign | 0x7c00u);  // 65504 + half an ulp, a tie, rounds to infinity
  if (a < 0x1p-14) {
    // Zero or subnormal, in units of 2^-24 (exact scaling, one rounding); 1024 units are the smallest normal's bits.
    return static_cast<uint16_t>(sign | static_cast<unsigned>(std::nearbyint(std::ldexp(a, 24))));
  }
  int e = 0;
  std::frexp(a, &e);                                     // a in [2^(e-1), 2^e)
  double m = std::nearbyint(std::ldexp(a, 11 - e));      // 11-bit significand in [1024, 2048], ties to even
  int exponent = e - 1;
  if (m == 2048.0) {
    m = 1024.0;
    ++exponent;
  }
  return static_cast<uint16_t>(sign | (static_cast<unsigned>(exponent + 15) << 10) |
                               (static_cast<unsigned>(m) - 1024u));
}

Asset import_ply(const std::string& path, const geo::M4& transform, const std::string& uv_origin,
                 int64_t max_texture_size, bool baked) {
  Asset a = load_ply(path, 1.0, uv_origin);
  bool identity = true;
  const geo::M4 eye = geo::identity4();
  for (size_t i = 0; i < 16; ++i) identity = identity && std::fabs(transform[i] - eye[i]) <= 1e-8 + 1e-5 * std::fabs(eye[i]);
  if (!identity) a.mesh = geo::transform_mesh(a.mesh, transform);
  geo::ValidationReport rep;
  a.mesh = geo::validate_mesh(a.mesh, rep);
  Tree report = Tree::dict();
  report["input_vertices"] = Tree::integer(rep.input_vertices);
  report["input_triangles"] = Tree::integer(rep.input_triangles);
  report["degenerate_removed"] = Tree::integer(rep.degenerate_removed);
  report["duplicate_removed"] = Tree::integer(rep.duplicate_removed);
  report["unreferenced_removed"] = Tree::integer(rep.unreferenced_removed);
  report["normals"] = Tree::string(rep.normals);
  report["vertices"] = Tree::integer(rep.vertices);
  report["triangles"] = Tree::integer(rep.triangles);
  Tree shrunk = Tree::list();
  for (size_t j = 0; j < a.materials.size(); ++j) {
    Material& m = a.materials[j];
    m.baked = baked;
    if (!m.texture) continue;
    int levels = 0;
    if (baked) {
      auto [lin, n] = downsample_linear(*m.texture, max_texture_size);
      levels = n;
      if (lin) {
        m.texture_linear = std::move(*lin);
        m.texture.reset();
      }
    } else {
      auto [tex, n] = downsample(std::move(*m.texture), max_texture_size);
      m.texture = std::move(tex);
      levels = n;
    }
    if (levels) {
      Tree e = Tree::dict();
      e["material"] = Tree::integer(static_cast<int64_t>(j));
      e["halvings"] = Tree::integer(levels);
      shrunk.push_back(std::move(e));
    }
  }
  a.info["validation"] = std::move(report);
  a.info["textures_downsampled"] = std::move(shrunk);
  if (baked) {
    Tree list = Tree::list();
    for (size_t j = 0; j < a.materials.size(); ++j) list.push_back(Tree::integer(static_cast<int64_t>(j)));
    a.info["baked_materials"] = std::move(list);
  }
  return a;
}

}  // namespace balla::scene::assets
