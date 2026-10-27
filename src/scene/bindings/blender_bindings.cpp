// ballasim._scene's Blender exchange bindings (docs/design/phase3/blender.md §3.8, §3.9): the ExchangeIn filled from
// the reader's checked exchange.json and arrays (python/ballasim/blender/exchange.py), the statics conversion, its
// report and the formats `ballasim probe` lists. Byte moving only: every decision is in src/scene/blender/.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include "bindings/ndarray_util.h"
#include "scene/blender/assemble.h"
#include "scene/blender/convert.h"
#include "scene/motion/session.h"
#include "scene/tree.h"

namespace nb = nanobind;
using namespace nb::literals;
using balla::bindings::Out;
using balla::bindings::to_numpy;
namespace sc = balla::scene;
namespace bl = balla::scene::blender;
namespace geo = balla::scene::geo;

namespace {

sc::Tree tree_of(nb::handle o) {
  if (o.is_none()) return sc::Tree::null();
  if (nb::isinstance<nb::bool_>(o)) return sc::Tree::boolean(nb::cast<bool>(o));
  if (nb::isinstance<nb::int_>(o)) return sc::Tree::integer(nb::cast<int64_t>(o));
  if (nb::isinstance<nb::float_>(o)) return sc::Tree::real(nb::cast<double>(o));
  if (nb::isinstance<nb::str>(o)) return sc::Tree::string(nb::cast<std::string>(o));
  if (nb::isinstance<nb::dict>(o)) {
    sc::Tree t = sc::Tree::dict();
    for (const auto& [k, v] : nb::borrow<nb::dict>(o)) t[nb::cast<std::string>(k)] = tree_of(v);
    return t;
  }
  if (nb::isinstance<nb::list>(o) || nb::isinstance<nb::tuple>(o)) {
    sc::Tree t = sc::Tree::list();
    for (nb::handle v : o) t.push_back(tree_of(v));
    return t;
  }
  throw std::invalid_argument("BRG-PROTO: an exchange value is not a JSON value");
}

nb::object py_of(const sc::Tree& t) {
  using K = sc::Tree::Kind;
  switch (t.kind()) {
    case K::kNull:
      return nb::none();
    case K::kBool:
      return nb::bool_(t.as_bool());
    case K::kInt:
      return nb::int_(t.as_int());
    case K::kFloat:
      return nb::float_(t.as_float());
    case K::kString:
      return nb::str(t.as_string().c_str(), t.as_string().size());
    case K::kList: {
      nb::list l;
      for (const sc::Tree& x : t.items()) l.append(py_of(x));
      return l;
    }
    case K::kDict: {
      nb::dict d;
      for (const auto& [k, v] : t.entries()) d[nb::str(k.c_str(), k.size())] = py_of(v);
      return d;
    }
  }
  return nb::none();
}

// The reader checks versions and keys, not value types: every value is type-checked here. A wrong type is a
// BRG-PROTO ValueError (convert_exchange names the record), never a TypeError or std::bad_cast.
struct WrongType : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

nb::dict record(nb::handle d) {
  if (!nb::isinstance<nb::dict>(d)) throw WrongType("is not an object");
  return nb::borrow<nb::dict>(d);
}
nb::list items(nb::handle l, const char* key) {
  if (!nb::isinstance<nb::list>(l)) throw WrongType(std::string(key) + " is not a list");
  return nb::borrow<nb::list>(l);
}

// d[key] when present and not None.
nb::handle get(nb::handle d, const char* key) {
  const nb::dict dd = record(d);
  if (!dd.contains(key)) return nb::handle();
  nb::handle v = dd[key];
  return v.is_none() ? nb::handle() : v;
}
nb::list list_at(nb::handle d, const char* key) {
  nb::handle v = get(d, key);
  return v.is_valid() ? items(v, key) : nb::list();
}
template <class T>
T value(nb::handle d, const char* key, T fallback) {
  nb::handle v = get(d, key);
  if (!v.is_valid()) return fallback;
  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
    if (nb::isinstance<nb::bool_>(v)) throw WrongType(std::string(key) + " is a boolean, not an integer");
  }
  T out{};
  if (!nb::try_cast(v, out)) throw WrongType(std::string(key) + " has the wrong type");
  return out;
}
int32_t index(nb::handle d, const char* key) { return value<int32_t>(d, key, -1); }
sc::Tree tree(nb::handle d, const char* key) {
  nb::handle v = get(d, key);
  return v.is_valid() ? tree_of(v) : sc::Tree::null();
}

// An exchange array of exactly this dtype (no cast: a float64 mesh is an exporter error), copied out, with its
// column count checked.
template <class T>
std::vector<T> array(nb::handle arrays, nb::handle key_obj, size_t cols, const std::string& what) {
  if (!key_obj.is_valid()) throw std::invalid_argument("BRG-PROTO: " + what + " is missing");
  if (!nb::isinstance<nb::str>(key_obj)) throw WrongType(what + ": an array key is not a string");
  const std::string key = nb::cast<std::string>(key_obj);
  const nb::dict a = nb::borrow<nb::dict>(arrays);
  if (!a.contains(key.c_str())) throw std::invalid_argument("BRG-PROTO: arrays.npz has no " + key);
  nb::ndarray<const T, nb::c_contig, nb::device::cpu> arr;
  if (!nb::try_cast(a[key.c_str()], arr, false) || arr.ndim() != (cols == 1 ? 1u : 2u) ||
      (cols > 1 && arr.shape(1) != cols)) {
    throw std::invalid_argument("BRG-PROTO: " + key + " has the wrong dtype, shape or layout for " + what);
  }
  return std::vector<T>(arr.data(), arr.data() + arr.size());
}

// A socket record {identifier, name, type, enabled, value, default, linked}; a value is a number, a boolean, a list of
// up to 4 numbers, or anything else (strings, IDs: no components).
int components(nb::handle v, std::array<double, 4>& out) {
  if (!v.is_valid() || v.is_none() || nb::isinstance<nb::str>(v)) return 0;
  if (nb::isinstance<nb::bool_>(v) || nb::isinstance<nb::int_>(v) || nb::isinstance<nb::float_>(v)) {
    out[0] = nb::cast<double>(v);
    return 1;
  }
  if (nb::isinstance<nb::list>(v)) {
    const nb::list l = nb::borrow<nb::list>(v);
    if (l.size() > 4) return 0;
    for (size_t k = 0; k < l.size(); ++k) out[k] = nb::cast<double>(l[k]);
    return static_cast<int>(l.size());
  }
  throw WrongType("a socket value is not a number or a list of numbers");
}

bl::SocketIn socket_in(nb::handle s) {
  bl::SocketIn out;
  out.identifier = value<std::string>(s, "identifier", "");
  if (out.identifier.empty()) throw WrongType("a socket has no identifier");
  out.name = value<std::string>(s, "name", "");
  out.type = value<std::string>(s, "type", "");
  out.enabled = value<bool>(s, "enabled", true);
  out.linked = value<bool>(s, "linked", false);
  out.n = components(get(s, "value"), out.value);
  out.n_def = components(get(s, "default"), out.def);
  return out;
}

// A node graph {nodes, links} (materials, lights, the world; §3.4) into its nodes and links.
void graph_in(nb::handle g, std::vector<bl::NodeIn>& nodes, std::vector<bl::LinkIn>& links) {
  for (nb::handle n : list_at(g, "nodes")) {
    bl::NodeIn node;
    node.name = value<std::string>(n, "name", "");
    node.type = value<std::string>(n, "type", "");
    node.data_type = value<std::string>(n, "data_type", "");
    node.blend_type = value<std::string>(n, "blend_type", "");
    if (nb::handle p = get(n, "props"); p.is_valid()) node.props = tree_of(record(p));
    for (nb::handle s : list_at(n, "inputs")) node.inputs.push_back(socket_in(s));
    for (nb::handle s : list_at(n, "outputs")) node.outputs.push_back(socket_in(s));
    nodes.push_back(std::move(node));
  }
  for (nb::handle l : list_at(g, "links")) {
    const nb::list four = items(l, "links[]");
    if (four.size() != 4) throw WrongType("a link is not [from_node, from_socket, to_node, to_socket]");
    std::array<std::string, 4> f;
    for (size_t k = 0; k < 4; ++k) {
      if (!nb::isinstance<nb::str>(four[k])) throw WrongType("a link holds a non-string");
      f[k] = nb::cast<std::string>(four[k]);
    }
    links.push_back({f[0], f[1], f[2], f[3]});
  }
}

bl::GraphIn graph_at(nb::handle d, const char* key) {
  bl::GraphIn g;
  nb::handle v = get(d, key);
  if (!v.is_valid()) return g;
  g.present = true;
  graph_in(v, g.nodes, g.links);
  return g;
}

bl::MaterialIn material_in(nb::handle m) {
  bl::MaterialIn out;
  out.id = nb::cast<std::string>(nb::str(record(m)["id"]));
  out.name = value<std::string>(m, "name", out.id);
  out.emission_sampling = value<std::string>(m, "emission_sampling", "");
  out.asset = tree(m, "asset");
  nb::handle g = get(m, "graph");
  if (!g.is_valid()) return out;
  out.has_graph = true;
  graph_in(g, out.nodes, out.links);
  return out;
}

// An images[] record and the path of its pixel file (the reader's texture paths, aligned with images[]; None for a
// float image). The converter reads the file only when a material uses the image.
bl::ImageIn image_in(nb::handle r, nb::handle path) {
  bl::ImageIn im;
  im.id = nb::cast<std::string>(nb::str(record(r)["id"]));
  im.colorspace = value<std::string>(r, "colorspace", "");
  im.extension = value<std::string>(r, "extension", "");
  im.alpha_mode = value<std::string>(r, "alpha_mode", "");
  im.is_float = value<std::string>(r, "kind", "byte") == "float";
  im.asset = tree(r, "asset");
  im.w = value<int32_t>(r, "width", -1);
  im.h = value<int32_t>(r, "height", -1);
  im.c = value<int32_t>(r, "channels", -1);
  if (path.is_valid() && !path.is_none()) {
    if (!nb::isinstance<nb::str>(path)) throw WrongType("a texture path is not a string");
    im.path = nb::cast<std::string>(path);
  }
  return im;
}

bl::ExchangeIn exchange_in(nb::dict doc, nb::dict arrays, const std::string& digest, std::string& where,
                           nb::handle textures = nb::handle(), nb::handle calibrations = nb::handle()) {
  bl::ExchangeIn ex;
  where = "version";
  const nb::list version = list_at(doc, "version");
  if (version.size() != 2) throw WrongType("must be [major, minor]");
  ex.version = {nb::cast<int32_t>(version[0]), nb::cast<int32_t>(version[1])};
  where = "exchange.json";
  ex.digest = digest;
  ex.units = tree(doc, "units");
  ex.color = tree(doc, "color");
  ex.time = tree(doc, "time");
  ex.span = tree(doc, "span");
  ex.render = tree(doc, "render");
  ex.rigidbody_world = tree(doc, "rigidbody_world");
  ex.cache_files = tree(doc, "cache_files");
  if (const nb::list g = list_at(doc, "gravity"); g.size() > 0) {
    if (g.size() != 3) throw WrongType("gravity needs 3 values");
    for (size_t k = 0; k < 3; ++k) ex.gravity[k] = nb::cast<double>(g[k]);
  }
  ex.use_gravity = value<bool>(doc, "use_gravity", true);
  const nb::object json_dumps = nb::module_::import_("json").attr("dumps");
  size_t at = 0;
  for (nb::handle m : list_at(doc, "meshes")) {
    where = "meshes[" + std::to_string(at++) + "]";
    bl::MeshIn mi;
    mi.id = nb::cast<std::string>(nb::str(record(m)["id"]));
    const std::string dom = value<std::string>(m, "normals_domain", "CORNER");
    mi.domain = dom == "POINT" ? bl::MeshIn::Normals::kPoint : dom == "FACE" ? bl::MeshIn::Normals::kFace
                                                                             : bl::MeshIn::Normals::kCorner;
    nb::handle a = get(m, "arrays");
    if (!a.is_valid()) throw WrongType("has no arrays");
    const std::string w = "mesh " + mi.id;
    mi.co = array<float>(arrays, get(a, "co"), 3, w + " co");
    mi.corner_vert = array<int32_t>(arrays, get(a, "corner_vert"), 1, w + " corner_vert");
    mi.tri_corners = array<int32_t>(arrays, get(a, "tri_corners"), 3, w + " tri_corners");
    if (get(a, "tri_mat").is_valid()) mi.tri_mat = array<int16_t>(arrays, get(a, "tri_mat"), 1, w + " tri_mat");
    const bool point = mi.domain == bl::MeshIn::Normals::kPoint;
    mi.normals = array<float>(arrays, get(a, point ? "vert_normal" : "corner_normal"), 3, w + " normals");
    if (nb::handle uv = get(a, "uv"); uv.is_valid()) {
      for (const auto& [k, v] : record(uv)) {
        mi.uvs[nb::cast<std::string>(k)] = array<float>(arrays, v, 2, w + " uv " + nb::cast<std::string>(k));
      }
    }
    mi.active_render_uv = value<std::string>(m, "active_render_uv", "");
    mi.asset = tree(m, "asset");
    ex.meshes.push_back(std::move(mi));
  }
  at = 0;
  for (nb::handle o : list_at(doc, "objects")) {
    where = "objects[" + std::to_string(at++) + "]";
    bl::ObjectIn ob;
    ob.key = nb::cast<std::string>(json_dumps(record(o)["key"], "separators"_a = nb::make_tuple(",", ":")));
    ob.name_full = value<std::string>(o, "name_full", "");
    ob.type = value<std::string>(o, "type", "");
    ob.is_instance = value<std::string>(o, "kind", "object") == "instance";
    ob.is_editable = value<bool>(o, "is_editable", true);
    ob.instancer = index(o, "instancer");
    ob.instance_object = index(o, "instance_object");
    ob.parent = index(o, "parent");
    ob.mesh = index(o, "mesh");
    ob.animation = index(o, "animation");
    ob.data_animation = index(o, "data_animation");
    if (const nb::list l = list_at(o, "persistent_id"); l.size() > 0) {
      for (size_t k = 0; k < 8 && k < l.size(); ++k) ob.persistent_id[k] = nb::cast<int64_t>(l[k]);
    }
    ob.parent_type = value<std::string>(o, "parent_type", "");
    ob.parent_bone = value<std::string>(o, "parent_bone", "");
    for (nb::handle v : list_at(o, "material_slots")) {
      if (!v.is_none() && nb::isinstance<nb::bool_>(v)) throw WrongType("material_slots holds a boolean");
      ob.material_slots.push_back(v.is_none() ? -1 : nb::cast<int32_t>(v));
      if (ob.material_slots.back() < -1) throw WrongType("material_slots holds a negative index");
    }
    const nb::list mw = list_at(o, "matrix_world");
    if (mw.size() != 16) throw WrongType("matrix_world needs 16 values");
    for (size_t k = 0; k < 16; ++k) ob.matrix_world[k] = nb::cast<double>(mw[k]);
    if (nb::handle v = get(o, "visibility"); v.is_valid()) ob.visibility = tree_of(v);
    if (nb::handle v = get(o, "deform"); v.is_valid()) ob.deform = tree_of(v);
    ob.rigid_body = tree(o, "rigid_body");
    ob.field = tree(o, "field");
    if (nb::handle v = get(o, "constraints"); v.is_valid()) ob.constraints = tree_of(v);
    ob.lightgroup = value<std::string>(o, "lightgroup", "");
    ob.pass_index = value<int32_t>(o, "pass_index", 0);
    ob.instance_id = value<int64_t>(o, "instance_id", 0);
    ob.instance_block = value<int64_t>(o, "instance_block", 0);
    ob.instancer_ids = value<std::string>(o, "instancer_ids", "per_instance");
    ob.semantic = value<std::string>(o, "semantic", "");
    for (nb::handle v : list_at(o, "collections")) ob.collections.push_back(nb::cast<std::string>(v));
    ob.asset = tree(o, "asset");
    ex.objects.push_back(std::move(ob));
  }
  // The reader's calibration records (aligned with cameras[]; python/ballasim/blender/calibration.py). Without
  // them a camera's calibration cannot be used: its record says so (CAM-CALIB).
  const nb::list cals = calibrations.is_valid() && !calibrations.is_none() ? items(calibrations, "calibrations")
                                                                           : nb::list();
  at = 0;
  for (nb::handle c : list_at(doc, "cameras")) {
    where = "cameras[" + std::to_string(at) + "]";
    bl::CameraIn ci;
    ci.object = index(c, "object");
    ci.c1_name = value<std::string>(c, "c1_name", "");
    if (nb::handle v = get(c, "props"); v.is_valid()) ci.props = tree_of(v);
    if (calibrations.is_valid() && !calibrations.is_none()) {
      if (at < cals.size() && !cals[at].is_none()) ci.calibration = tree_of(record(cals[at]));
    } else if (nb::handle raw = get(c, "calibration"); raw.is_valid()) {
      const std::string source = value<std::string>(raw, "source", "none");
      if (source != "none") {
        ci.calibration = sc::Tree::dict();
        ci.calibration["source"] = sc::Tree::string(source);
        ci.calibration["error"] = sc::Tree::string("not read (ballasim.blender.calibration reads it)");
      }
    }
    ci.sensor_family = value<std::string>(c, "sensor_family", "");
    ci.simulate = value<bool>(c, "simulate", true);
    ci.animation = index(c, "animation");
    ex.cameras.push_back(std::move(ci));
    ++at;
  }
  at = 0;
  for (nb::handle l : list_at(doc, "lights")) {
    where = "lights[" + std::to_string(at++) + "]";
    bl::LightIn li;
    li.object = index(l, "object");
    li.type = value<std::string>(l, "type", "");
    if (nb::handle v = get(l, "props"); v.is_valid()) li.props = tree_of(record(v));
    li.graph = graph_at(l, "node_graph");
    li.light_linking = tree(l, "light_linking");
    li.animation = index(l, "animation");
    li.graph_animation = index(l, "node_graph_animation");
    ex.lights.push_back(std::move(li));
  }
  at = 0;
  for (nb::handle m : list_at(doc, "materials")) {
    where = "materials[" + std::to_string(at++) + "]";
    ex.materials.push_back(material_in(m));
  }
  const nb::list tex = textures.is_valid() && !textures.is_none() ? items(textures, "textures") : nb::list();
  at = 0;
  for (nb::handle r : list_at(doc, "images")) {
    where = "images[" + std::to_string(at) + "]";
    ex.images.push_back(image_in(r, at < tex.size() ? nb::handle(tex[at]) : nb::handle()));
    ++at;
  }
  where = "exchange.json";
  for (const char* k : {"animations", "light_groups"}) {
    sc::Tree t = tree(doc, k);
    if (t.kind() == sc::Tree::Kind::kNull) t = sc::Tree::list();
    (k[0] == 'a' ? ex.animations : ex.light_groups) = t;
  }
  if (nb::handle w = get(doc, "world"); w.is_valid()) {
    where = "world";
    ex.world.present = true;
    ex.world.graph = graph_at(w, "graph");
    ex.world.color = tree(w, "color");
    ex.world.lightgroup = value<std::string>(w, "lightgroup", "");
    ex.world.animation = index(w, "animation");
    ex.world.graph_animation = index(w, "graph_animation");
    if (nb::handle v = get(w, "cycles_visibility"); v.is_valid()) ex.world.cycles_visibility = tree_of(record(v));
    ex.world.asset = tree(w, "asset");
  }
  if (nb::handle v = get(doc, "semantic_classes"); v.is_valid()) ex.semantic_classes = tree_of(v);
  if (nb::handle v = get(doc, "collection_semantics"); v.is_valid()) ex.collection_semantics = tree_of(v);
  at = 0;
  for (nb::handle it : list_at(doc, "addon_report")) {
    where = "addon_report[" + std::to_string(at++) + "]";
    ex.addon_report.push_back({value<std::string>(it, "code", ""), value<std::string>(it, "object", ""),
                               value<std::string>(it, "message", "")});
  }
  return ex;
}

// exchange_in with every type error as a BRG-PROTO ValueError naming the record.
bl::ExchangeIn convert_exchange(nb::dict doc, nb::dict arrays, const std::string& digest,
                              nb::handle textures = nb::handle(), nb::handle calibrations = nb::handle()) {
  std::string where = "exchange.json";
  const auto wrong = [&where](const std::string& what) {
    return std::invalid_argument("BRG-PROTO: " + where + ": " + what);
  };
  try {
    return exchange_in(doc, arrays, digest, where, textures, calibrations);
  } catch (const WrongType& e) {
    throw wrong(e.what());
  } catch (const std::bad_cast&) {   // nb::cast_error
    throw wrong("a value has the wrong type");
  } catch (const nb::python_error& e) {   // e.g. a record's key that is not JSON
    throw wrong(e.what());
  }
}

// The F-curves' key tables (anim.<key>.keys float64[n, 7]) that the motion session's declared keys read; the statics
// and the checks do not need them, so only a session asks (a missing table is the exporter's fault: BRG-PROTO).
void read_anim_keys(bl::ExchangeIn& ex, nb::dict arrays) {
  for (const sc::Tree& rec : ex.animations.items()) {
    const sc::Tree* fcurves = rec.kind() == sc::Tree::Kind::kDict ? rec.find("fcurves") : nullptr;
    if (fcurves == nullptr || fcurves->kind() != sc::Tree::Kind::kList) continue;
    for (const sc::Tree& f : fcurves->items()) {
      const sc::Tree* name = f.kind() == sc::Tree::Kind::kDict ? f.find("keyframes") : nullptr;
      if (name == nullptr || name->kind() != sc::Tree::Kind::kString) continue;
      try {
        ex.anim_keys[name->as_string()] = array<double>(arrays, nb::str(name->as_string().c_str()), 7,
                                                        "F-curve keys " + name->as_string());
      } catch (const WrongType& e) {
        throw std::invalid_argument(std::string("BRG-PROTO: animations: ") + e.what());
      }
    }
  }
}

bl::ConvertOptions options_in(nb::dict d) {
  bl::ConvertOptions o;
  for (const auto& [k, v] : d) {
    const std::string key = nb::cast<std::string>(k);
    const auto opt = [&v]() {
      return v.is_none() ? std::optional<double>() : std::optional<double>(nb::cast<double>(v));
    };
    if (key == "policy") {
      const std::string p = nb::cast<std::string>(v);
      if (p != "strict" && p != "lenient") throw std::invalid_argument("policy must be strict or lenient");
      o.policy = p == "strict" ? bl::Policy::kStrict : bl::Policy::kLenient;
    } else if (key == "lm_per_w") {
      o.lm_per_w = opt();
    } else if (key == "world_as_ambient") {
      o.world_as_ambient = nb::cast<bool>(v);
    } else if (key == "crop_radius_m") {
      o.crop_radius_m = opt();
    } else if (key == "max_triangles") {
      o.max_triangles = nb::cast<int64_t>(v);
    } else if (key == "calib_tol_px") {
      o.calib_tol_px = nb::cast<double>(v);
    } else if (key == "fisheye_fit_tol_px") {
      o.fisheye_fit_tol_px = nb::cast<double>(v);
    } else if (key == "rigidity_tol_m") {
      o.rigidity_tol_m = nb::cast<double>(v);
    } else {
      throw std::invalid_argument("unknown conversion option " + key);
    }
  }
  return o;
}

Out<double> m4_out(const geo::M4& t) { return to_numpy(std::vector<double>(t.begin(), t.end()), {4, 4}); }

nb::dict mesh_dict(geo::Mesh&& m) {
  const size_t nv = m.num_vertices(), nt = m.num_triangles();
  nb::dict d;
  d["positions"] = to_numpy(std::move(m.positions), {nv, 3});
  d["indices"] = to_numpy(std::move(m.indices), {nt, 3});
  d["normals"] = m.has_normals ? nb::cast(to_numpy(std::move(m.normals), {nv, 3})) : nb::none();
  d["uvs"] = m.has_uvs ? nb::cast(to_numpy(std::move(m.uvs), {nv, 2})) : nb::none();
  d["material_ids"] = m.has_material_ids ? nb::cast(to_numpy(std::move(m.material_ids), {nt})) : nb::none();
  return d;
}

// Every kind has its own name (F4 added rect and disk to the enum); an unknown value is a bug, not "background".
const char* light_kind(balla::render::BundleLightKind k) {
  using K = balla::render::BundleLightKind;
  switch (k) {
    case K::kPoint: return "point";
    case K::kSpot: return "spot";
    case K::kDirectional: return "directional";
    case K::kAmbient: return "ambient";
    case K::kBackground: return "background";
    case K::kRect: return "rect";
    case K::kDisk: return "disk";
  }
  throw std::logic_error("blender: unknown light kind");
}

nb::dict camera_dict(const balla::render::Camera& c) {
  nb::dict d;
  d["name"] = c.name;
  d["model"] = c.model == balla::render::CameraModelKind::kKannalaBrandt ? "kb" : "radtan";
  d["width"] = c.width;
  d["height"] = c.height;
  d["fx"] = c.fx;
  d["fy"] = c.fy;
  d["cx"] = c.cx;
  d["cy"] = c.cy;
  d["dist"] = c.dist;
  return d;
}

nb::dict statics(nb::dict doc, nb::dict arrays, const std::string& digest, nb::dict options, nb::handle origin,
                 nb::handle textures, nb::handle calibrations, const std::vector<int64_t>& dynamic_ids) {
  const bl::ExchangeIn ex = convert_exchange(doc, arrays, digest, textures, calibrations);
  const bl::ConvertOptions o = options_in(options);
  std::array<double, 3> org{};
  if (!origin.is_none()) {
    const std::vector<double> v = nb::cast<std::vector<double>>(origin);
    if (v.size() != 3) throw std::invalid_argument("origin needs 3 values");
    std::copy(v.begin(), v.end(), org.begin());
  }
  bl::Statics s;
  {
    nb::gil_scoped_release release;
    s = bl::convert_statics(ex, o);
  }
  std::optional<bl::StaticTimeline> timeline;
  std::string timeline_error;
  try {
    timeline = bl::static_timeline(ex);
  } catch (const std::invalid_argument& e) {   // BRG-PROTO: a malformed time record fails the assembly, not the check
    timeline_error = e.what();
  }
  // Bodies the motion session found moving although nothing in their chain says so (lenient MOT-SOURCE) are not static:
  // placed about the origin as moving bodies, neither cropped nor checked against the static reach.
  if (!dynamic_ids.empty()) {
    const std::set<int64_t> dyn(dynamic_ids.begin(), dynamic_ids.end());
    for (bl::BodyStatic& b : s.bodies) {
      if (dyn.count(b.id) != 0) b.T_world_static.reset();
    }
  }
  const std::vector<std::string> needed = bl::motion_needed(ex, s);
  nb::dict out;
  if (!s.refused()) {
    if (origin.is_none()) {
      const auto samples = bl::provisional_samples(ex, s);
      org = bl::origin_world(s, samples);
    }
    nb::gil_scoped_release release;
    bl::place_statics(s, org, o);
  }
  const bool placed = !s.refused() || !origin.is_none();
  out["origin_world"] = placed ? nb::cast(std::vector<double>(org.begin(), org.end())) : nb::none();
  nb::list bodies;
  for (bl::BodyStatic& b : s.bodies) {
    nb::dict d;
    d["object"] = b.object;
    d["id"] = b.id;
    d["semantic"] = b.semantic;
    d["name"] = b.name;
    d["role"] = b.role == bl::BodyStatic::Role::kCarrier ? "carrier" : "body";
    d["mesh"] = mesh_dict(std::move(b.mesh));
    nb::list mats;
    for (size_t q = 0; q < b.materials.size(); ++q) {
      const auto& m = b.materials[q];
      nb::dict md;
      md["base_color"] = nb::make_tuple(m.base_color[0], m.base_color[1], m.base_color[2]);
      md["emissive"] = nb::make_tuple(m.emissive[0], m.emissive[1], m.emissive[2]);
      md["texture"] = b.texture_of[q] >= 0 ? nb::cast(b.texture_of[q]) : nb::none();
      mats.append(md);
    }
    d["materials"] = mats;
    d["material_of"] = b.material_of;
    d["S"] = to_numpy(std::vector<double>(b.S.begin(), b.S.end()), {3, 3});
    d["c"] = std::vector<double>(b.c.begin(), b.c.end());
    d["rho"] = b.rho;
    d["T_world_export"] = m4_out(b.T_world_export);
    d["T_world_static"] = b.T_world_static ? nb::cast(m4_out(*b.T_world_static)) : nb::none();
    d["parts"] = b.parts;
    bodies.append(d);
  }
  out["bodies"] = bodies;
  nb::list texs;
  for (bl::TextureOut& t : s.textures) {
    nb::dict td;
    td["image"] = t.image;
    td["mirror"] = t.mirror;
    td["pixels"] = to_numpy(std::move(t.rgb), {static_cast<size_t>(t.height), static_cast<size_t>(t.width), 3});
    texs.append(td);
  }
  out["textures"] = texs;
  nb::list cams;
  for (const bl::CameraStatic& c : s.cameras) {
    nb::dict d = camera_dict(c.cam);
    d["index"] = c.index;
    d["object"] = c.object;
    d["sensor"] = c.sensor;
    d["pixel_pitch_m"] = c.pixel_pitch_m;
    d["f_number"] = c.f_number;
    d["blender_pinhole"] = std::vector<double>(c.blender.begin(), c.blender.end());
    d["rig_object"] = c.rig_object;
    d["T_rig_cam"] = m4_out(c.T_rig_cam);
    d["T_world_export"] = m4_out(c.T_world_export);
    d["T_world_static"] = c.T_world_static ? nb::cast(m4_out(*c.T_world_static)) : nb::none();
    d["theta_max"] = c.theta_max;
    d["clip_start_m"] = c.clip_start_m;
    d["calibrated"] = c.calibrated;
    d["simulate"] = c.simulate;
    cams.append(d);
  }
  out["cameras"] = cams;
  nb::list lights;
  for (const bl::LightStatic& l : s.lights) {
    const balla::render::Light& L = l.light;
    nb::dict d;
    d["object"] = l.object;
    d["name"] = l.name;
    d["kind"] = light_kind(L.kind);
    d["group"] = L.group;
    d["color"] = nb::make_tuple(L.color[0], L.color[1], L.color[2]);
    d["intensity"] = L.intensity;
    d["body"] = L.body ? nb::cast(*L.body) : nb::none();
    d["position"] = std::vector<double>(L.position.begin(), L.position.end());
    d["direction"] = std::vector<double>(L.direction.begin(), L.direction.end());
    d["spot_inner_rad"] = L.spot_inner_rad;
    d["spot_outer_rad"] = L.spot_outer_rad;
    d["casts_shadows"] = L.casts_shadows;
    lights.append(d);
  }
  out["lights"] = lights;
  nb::list groups;
  for (const bl::LightGroupOut& g : s.groups) {
    nb::dict d;
    d["name"] = g.name;
    d["waveform"] = py_of(g.waveform);
    groups.append(d);
  }
  out["light_groups"] = groups;
  out["assets"] = py_of(bl::assets_doc(s.assets));
  out["camera_names"] = s.camera_names;
  out["triangles"] = s.triangles;
  out["refused"] = s.refused();
  // The static assembly (assemble.h): the timeline's span, what needs the motion session, and, for a static scene
  // that converts, each simulated camera's constant rig trajectory (K = 6, h = 2^30 ns, held over the bundle span).
  if (timeline) {
    out["timeline"] = py_of(bl::timeline_record(*timeline));
  } else {
    nb::dict err;
    err["error"] = timeline_error;
    out["timeline"] = err;
  }
  out["session_needed"] = needed;
  nb::list rigs;
  if (!s.refused() && needed.empty() && timeline) {
    for (const bl::CameraStatic& c : s.cameras) {
      if (!c.simulate || !c.T_world_static) continue;
      const balla::render::Trajectory tr = bl::static_rig(*c.T_world_static, org, timeline->t_start_ns,
                                                          timeline->t_end_ns);
      nb::dict d;
      d["camera"] = c.cam.name;
      d["K"] = tr.order;
      d["t0_ns"] = tr.t0_ns;
      d["h_ns"] = tr.h_ns;
      const size_t nc = tr.rot.size() / 4;
      d["rot"] = to_numpy(std::vector<double>(tr.rot), {nc, 4});
      d["pos"] = to_numpy(std::vector<double>(tr.pos), {nc, 3});
      d["span_ns"] = nb::make_tuple(tr.span_begin_ns, tr.span_end_ns);
      rigs.append(d);
    }
  }
  out["rigs"] = rigs;
  out["report"] = py_of(bl::report_doc(s.report, o.policy, s.matched));
  out["options"] = py_of(bl::options_record(o));
  return out;
}


// ---- the motion session (docs/design/phase3/blender.md §3.5; src/scene/motion/session.h) -----------------------------

namespace mo = balla::scene::motion;

// A contiguous NumPy copy of any array-like, as the given dtype (no silent casts of the wrong kind: the dtype names the
// protocol's).
template <class T>
std::vector<T> vec_from(nb::handle o, const char* dtype, const char* what) {
  nb::object a = nb::module_::import_("numpy").attr("ascontiguousarray")(o, dtype);
  nb::ndarray<const T, nb::c_contig, nb::device::cpu> arr;
  if (!nb::try_cast(a, arr, false)) throw std::invalid_argument(std::string("BRG-PROTO: ") + what + " is not a " + dtype + " array");
  return std::vector<T>(arr.data(), arr.data() + arr.size());
}

template <class T>
Out<T> numpy_of(const std::vector<T>& v, std::initializer_list<size_t> shape) {
  return to_numpy(std::vector<T>(v), shape);
}

// A poses answer: (c_final, m, present, unknown[, unknown_keys]) as arrays; the shapes are checked by the session.
mo::RawPoses raw_poses(nb::handle answer, size_t n) {
  if (!nb::isinstance<nb::tuple>(answer) && !nb::isinstance<nb::list>(answer)) {
    throw std::invalid_argument("BRG-PROTO: a poses answer is not a tuple");
  }
  const nb::sequence t = nb::borrow<nb::sequence>(answer);
  if (nb::len(t) < 4) throw std::invalid_argument("BRG-PROTO: a poses answer needs (c_final, m, present, unknown)");
  mo::RawPoses r;
  r.n = static_cast<int32_t>(n);
  r.c_final = vec_from<double>(t[0], "float64", "c_final");
  r.m = vec_from<float>(t[1], "float32", "m");
  r.present = vec_from<uint8_t>(t[2], "uint8", "present");
  r.unknown = vec_from<uint32_t>(t[3], "uint32", "unknown");
  if (nb::len(t) > 4 && !t[4].is_none()) r.unknown_keys = nb::cast<std::string>(t[4]);
  if (r.c_final.size() != n || r.unknown.size() != n) throw std::invalid_argument("BRG-PROTO: a poses answer has the wrong length");
  r.bodies = n == 0 ? 0 : static_cast<int32_t>(r.present.size() / n);
  if (r.m.size() != n * static_cast<size_t>(r.bodies) * 12 || r.present.size() != n * static_cast<size_t>(r.bodies)) {
    throw std::invalid_argument("BRG-PROTO: a poses answer's matrices and presence disagree");
  }
  return r;
}

nb::dict poses_dict(const mo::RawPoses& r) {
  nb::dict d;
  const auto n = static_cast<size_t>(r.n), b = static_cast<size_t>(r.bodies);
  d["F"] = numpy_of(r.F, {r.F.size()});
  d["s"] = numpy_of(r.s, {r.s.size()});
  d["c_final"] = numpy_of(r.c_final, {n});
  d["m"] = numpy_of(r.m, {n, b, 12});
  d["present"] = numpy_of(r.present, {n, b});
  d["unknown"] = numpy_of(r.unknown, {n});
  d["unknown_keys"] = r.unknown_keys;
  return d;
}

mo::RawPoses poses_from(nb::handle h) {
  mo::RawPoses r;
  r.F = vec_from<int32_t>(h["F"], "int32", "F");
  r.s = vec_from<float>(h["s"], "float32", "s");
  r.n = static_cast<int32_t>(r.F.size());
  r.c_final = vec_from<double>(h["c_final"], "float64", "c_final");
  r.m = vec_from<float>(h["m"], "float32", "m");
  r.present = vec_from<uint8_t>(h["present"], "uint8", "present");
  r.unknown = vec_from<uint32_t>(h["unknown"], "uint32", "unknown");
  if (nb::handle k = h["unknown_keys"]; !k.is_none()) r.unknown_keys = nb::cast<std::string>(k);
  r.bodies = r.n == 0 ? 0 : static_cast<int32_t>(r.present.size() / static_cast<size_t>(r.n));
  return r;
}

nb::dict log_dict(const mo::SessionLog& log) {
  nb::dict d;
  d["bodies"] = numpy_of(log.bodies, {log.bodies.size()});
  nb::list calls;
  for (const mo::RawPoses& r : log.poses.calls) calls.append(poses_dict(r));
  d["calls"] = calls;
  nb::list meshes;
  for (const mo::MeshCall& c : log.meshes) {
    nb::dict m;
    m["objects"] = numpy_of(c.objects, {c.objects.size()});
    m["F"] = numpy_of(c.F, {c.F.size()});
    m["s"] = numpy_of(c.s, {c.s.size()});
    m["table"] = numpy_of(c.meshes.table, {c.meshes.table.size() / 4, 4});
    m["co"] = numpy_of(c.meshes.co, {c.meshes.co.size()});
    meshes.append(m);
  }
  d["mesh_calls"] = meshes;
  return d;
}

mo::SessionLog log_from(nb::handle d) {
  mo::SessionLog log;
  log.bodies = vec_from<int32_t>(d["bodies"], "int32", "bodies");
  for (nb::handle c : nb::borrow<nb::list>(d["calls"])) log.poses.calls.push_back(poses_from(c));
  for (nb::handle c : nb::borrow<nb::list>(d["mesh_calls"])) {
    mo::MeshCall m;
    m.objects = vec_from<int32_t>(c["objects"], "int32", "objects");
    m.F = vec_from<int32_t>(c["F"], "int32", "F");
    m.s = vec_from<float>(c["s"], "float32", "s");
    m.meshes.table = vec_from<int64_t>(c["table"], "int64", "table");
    m.meshes.co = vec_from<float>(c["co"], "float32", "co");
    log.meshes.push_back(std::move(m));
  }
  return log;
}

// SessionOptions from the dict of the Python side: motion_tol_px, fps_base_snap, cameras, workers, span_margin_windows,
// survey_samples and the heuristics' dicts sample / detect / fit (the field names of pipeline.h and fit_piecewise.h).
mo::SessionOptions session_options(nb::dict sess, const bl::ConvertOptions& convert) {
  mo::SessionOptions o;
  o.convert = convert;
  const auto has = [&sess](const char* k) { return sess.contains(k) && !sess[k].is_none(); };
  if (has("motion_tol_px")) o.sample.eps_px = nb::cast<double>(sess["motion_tol_px"]);
  if (has("fps_base_snap")) o.fps_base_snap = nb::cast<bool>(sess["fps_base_snap"]);
  if (has("cameras")) o.cameras = nb::cast<std::vector<std::string>>(sess["cameras"]);
  if (has("workers")) o.workers = nb::cast<int>(sess["workers"]);
  if (has("span_margin_windows")) o.span_margin_windows = nb::cast<int>(sess["span_margin_windows"]);
  if (has("survey_samples")) o.survey_samples = nb::cast<int>(sess["survey_samples"]);
  const auto section = [&](const char* name, const auto& apply) {
    if (!has(name)) return;
    for (const auto& [k, v] : nb::borrow<nb::dict>(sess[name])) {
      if (!apply(nb::cast<std::string>(k), v)) {
        throw std::invalid_argument(std::string("unknown ") + name + " option " + nb::cast<std::string>(k));
      }
    }
  };
  section("sample", [&o](const std::string& k, nb::handle v) {
    if (k == "h0_ns") o.sample.h0_ns = nb::cast<int64_t>(v);
    else if (k == "h_min_ns") o.sample.h_min_ns = nb::cast<int64_t>(v);
    else if (k == "max_rounds") o.sample.max_rounds = nb::cast<int>(v);
    else if (k == "max_per_message") o.sample.max_per_message = nb::cast<int>(v);
    else if (k == "alias_per_frame") o.sample.alias_per_frame = nb::cast<int>(v);
    else if (k == "max_base_samples") o.sample.max_base_samples = nb::cast<int64_t>(v);
    else return false;
    return true;
  });
  section("detect", [&o](const std::string& k, nb::handle v) {
    if (k == "k_sig") o.detect.k_sig = nb::cast<double>(v);
    else if (k == "unsplit_coeff") o.detect.unsplit_coeff = nb::cast<double>(v);
    else if (k == "burst_n") o.detect.burst_n = nb::cast<int>(v);
    else if (k == "burst_iters") o.detect.burst_iters = nb::cast<int>(v);
    else if (k == "burst_floor_ns") o.detect.burst_floor_ns = nb::cast<int64_t>(v);
    else if (k == "merge_ns") o.detect.merge_ns = nb::cast<int64_t>(v);
    else if (k == "h_cap_ns") o.detect.h_cap_ns = nb::cast<int64_t>(v);
    else if (k == "refuse_steps") o.detect.refuse_steps = nb::cast<bool>(v);
    else return false;
    return true;
  });
  section("fit", [&o](const std::string& k, nb::handle v) {
    if (k == "h_cap_ns") o.fit.h_cap_ns = nb::cast<int64_t>(v);
    else if (k == "h_floor_ns") o.fit.h_floor_ns = nb::cast<int64_t>(v);
    else if (k == "lambda_rel") o.fit.lambda_rel = nb::cast<double>(v);
    else if (k == "accept_frac") o.fit.accept_frac = nb::cast<double>(v);
    else if (k == "halvings") o.fit.halvings = nb::cast<int>(v);
    else if (k == "probes") o.fit.probes = nb::cast<int>(v);
    else if (k == "probe_seed") o.fit.probe_seed = nb::cast<uint64_t>(v);
    else if (k == "breakpoint_probes") o.fit.breakpoint_probes = nb::cast<bool>(v);
    else if (k == "end_fill_div") o.fit.end_fill_div = nb::cast<int>(v);
    else return false;
    return true;
  });
  return o;
}

nb::dict trajectory_dict(const mo::SessionTrajectory& t) {
  const balla::render::Trajectory& tr = t.traj;
  nb::dict d;
  d["K"] = tr.order;
  d["t0_ns"] = tr.t0_ns;
  d["h_ns"] = tr.h_ns;
  d["rot"] = numpy_of(tr.rot, {tr.rot.size() / 4, 4});
  d["pos"] = numpy_of(tr.pos, {tr.pos.size() / 3, 3});
  d["span_ns"] = nb::make_tuple(tr.span_begin_ns, tr.span_end_ns);
  std::vector<int64_t> pieces;
  for (const balla::render::TrajectoryPiece& p : tr.pieces) {
    pieces.insert(pieces.end(), {p.begin_ns, p.t0_ns, p.h_ns, p.ctrl_begin, p.ctrl_end, p.join});
  }
  d["pieces"] = tr.pieces.empty() ? nb::none() : nb::object(nb::cast(numpy_of(pieces, {tr.pieces.size(), 6})));
  std::vector<int64_t> holds;
  for (const auto& h : t.holds) holds.insert(holds.end(), {h.first, h.second});
  d["holds"] = numpy_of(holds, {t.holds.size(), 2});
  return d;
}

nb::dict motion_session(nb::dict doc, nb::dict arrays, const std::string& digest, nb::dict options, nb::dict sess,
                        nb::handle oracle, nb::handle mesh_oracle, nb::handle on_bodies, nb::handle on_simplify,
                        nb::handle on_progress, nb::handle textures, nb::handle calibrations, nb::handle log) {
  bl::ExchangeIn ex = convert_exchange(doc, arrays, digest, textures, calibrations);
  read_anim_keys(ex, arrays);
  const mo::SessionOptions opt = session_options(sess, options_in(options));
  std::optional<mo::SessionLog> replay;
  mo::SessionIO io;
  if (!log.is_none()) {
    replay = log_from(log);
    io = mo::replaying_io(*replay);
  } else {
    if (oracle.is_none()) throw std::invalid_argument("motion_session needs an oracle or a log");
    io.oracle = [oracle](std::span<const int32_t> F, std::span<const float> s) {
      nb::gil_scoped_acquire gil;
      const size_t n = F.size();
      nb::object r = oracle(to_numpy(std::vector<int32_t>(F.begin(), F.end()), {n}),
                            to_numpy(std::vector<float>(s.begin(), s.end()), {n}));
      return raw_poses(r, n);
    };
    if (!mesh_oracle.is_none()) {
      io.meshes = [mesh_oracle](std::span<const int32_t> objects, std::span<const int32_t> F, std::span<const float> s) {
        nb::gil_scoped_acquire gil;
        const size_t n = F.size();
        nb::object r = mesh_oracle(to_numpy(std::vector<int32_t>(objects.begin(), objects.end()), {objects.size()}),
                                   to_numpy(std::vector<int32_t>(F.begin(), F.end()), {n}),
                                   to_numpy(std::vector<float>(s.begin(), s.end()), {n}));
        const nb::sequence t = nb::borrow<nb::sequence>(r);
        mo::RawMeshes out;
        out.table = vec_from<int64_t>(t[0], "int64", "table");
        out.co = vec_from<float>(t[1], "float32", "co");
        return out;
      };
    }
    if (!on_bodies.is_none()) {
      io.bodies = [on_bodies](std::span<const int32_t> b) {
        nb::gil_scoped_acquire gil;
        on_bodies(to_numpy(std::vector<int32_t>(b.begin(), b.end()), {b.size()}));
      };
    }
    if (!on_simplify.is_none()) {
      io.simplify = [on_simplify](bool on) {
        nb::gil_scoped_acquire gil;
        on_simplify(on);
      };
    }
  }
  if (!on_progress.is_none()) {
    io.progress = [on_progress](double f, std::string_view stage) {
      nb::gil_scoped_acquire gil;
      on_progress(f, std::string(stage));
    };
  }
  mo::SessionResult r;
  {
    nb::gil_scoped_release release;
    r = mo::run_session(ex, opt, io);
  }
  nb::dict out;
  out["refused"] = r.refused;
  out["origin_world"] = std::vector<double>(r.origin_world.begin(), r.origin_world.end());
  out["simplify"] = r.simplify;
  nb::list report;
  for (const bl::ReportItem& it : r.report) {
    nb::dict d;
    d["code"] = it.code;
    d["severity"] = it.severity;
    d["object"] = it.object;
    d["message"] = it.message;
    d["bound"] = it.bound;
    d["bound_kind"] = bl::bound_kind_name(it.bound_kind);
    report.append(d);
  }
  out["report"] = report;
  {
    std::vector<bl::ReportItem> all = r.statics.report;
    all.insert(all.end(), r.report.begin(), r.report.end());
    out["conversion_report"] = py_of(bl::report_doc(all, opt.convert.policy, r.statics.matched));
  }
  nb::list trajs;
  for (const mo::SessionTrajectory& t : r.trajectories) trajs.append(trajectory_dict(t));
  out["trajectories"] = trajs;
  out["body_trajectory"] = r.body_trajectory;
  out["camera_trajectory"] = r.camera_trajectory;
  std::vector<double> rig;
  for (const auto& m : r.camera_t_rig_cam) rig.insert(rig.end(), m.begin(), m.end());
  out["camera_T_rig_cam"] = to_numpy(std::move(rig), {r.camera_t_rig_cam.size(), 4, 4});
  out["session"] = py_of(r.session);
  out["summary"] = py_of(r.summary);
  out["perf"] = py_of(r.perf);
  out["log"] = log_dict(r.log);
  return out;
}

}  // namespace

void bind_blender(nb::module_& m) {
  m.def(
      "blender_formats",
      [] {
        nb::dict d, ex;
        ex["ballasim.blender_exchange"] = nb::make_tuple(bl::kExchangeFormat[0], bl::kExchangeFormat[1]);
        d["exchange_formats"] = ex;
        nb::dict oracle;
        oracle[balla::scene::motion::kOracleProtocolName] =
            nb::make_tuple(balla::scene::motion::kOracleProtocol[0], balla::scene::motion::kOracleProtocol[1]);
        d["oracle_protocols"] = oracle;
        d["converter"] = bl::kConverterVersion;
        return d;
      },
      "The Blender formats this engine reads ({'exchange_formats': {name: (major, highest minor)}}, the statics\n"
      "converter's version), as `ballasim probe --json` lists them (blender.md §3.4, §3.9).");
  m.def("blender_statics", &statics, "doc"_a, "arrays"_a, "digest"_a, "options"_a, "origin"_a = nb::none(),
        "textures"_a = nb::none(), "calibrations"_a = nb::none(), "dynamic_ids"_a = std::vector<int64_t>{},
        "The statics of a checked exchange (exchange.json as a dict, its arrays by key, its digest; blender.md §3.8):\n"
        "bodies (C1 meshes in their body frame, materials (§4.5), ids, semantics, poses; light carriers), the\n"
        "textures they share, the cameras (§4.3: C1 intrinsics, sensor, pitch, f-number, rig pose), the lights and\n"
        "the world (§4.4: C1 lights, a carrier's in its body frame, else world frame, Blender origin, metres), the\n"
        "light groups, the cameras' C1 names, the assets they use (PHASE3.md H4), the conversion report (§3.12, with\n"
        "matched_overrides) and the options. `textures` lists each images[] record's pixel file (textures/\n"
        "<sha256>.npy; None for a float image) in order; a file is read only when a material uses its image.\n"
        "`calibrations` lists each cameras[] record's calibration as ballasim.blender.calibration reads it (None:\n"
        "none). Unless something is refused, the 16 m rule or the crop is applied about `origin` (default: the\n"
        "provisional origin of the simulated cameras and non-static bodies at the export frame). `timeline` is the\n"
        "span's part of the timeline record (§4.2), `session_needed` why the scene needs the motion session\n"
        "(empty: its statics are the whole scene) and `rigs`, for such a static scene, each simulated camera's\n"
        "constant rig trajectory over the bundle span [-2W, t_end] (§4.7.9).");
  m.def(
      "blender_camera_settings",
      [](nb::dict record, double sensor_width_mm) {
        const bl::Calibration cal = bl::calibration_camera(tree_of(record));   // invalid_argument -> ValueError
        const bl::BlenderCamera b = bl::camera_settings(cal.cam, sensor_width_mm);
        nb::dict d, set;
        set["sensor_fit"] = b.sensor_fit;
        set["lens"] = b.lens_mm;
        set["sensor_width"] = b.sensor_width_mm;
        set["shift_x"] = b.shift_x;
        set["shift_y"] = b.shift_y;
        set["resolution_x"] = b.res_x;
        set["resolution_y"] = b.res_y;
        set["resolution_percentage"] = 100;
        set["pixel_aspect_x"] = b.pixel_aspect_x;
        set["pixel_aspect_y"] = b.pixel_aspect_y;
        d["blender"] = set;
        d["calibration"] = camera_dict(cal.cam);
        d["sensor"] = cal.sensor;
        d["pixel_pitch_m"] = cal.pixel_pitch_m ? nb::cast(*cal.pixel_pitch_m) : nb::none();
        d["f_number"] = cal.f_number ? nb::cast(*cal.f_number) : nb::none();
        return d;
      },
      "record"_a, "sensor_width_mm"_a = 36.0,
      "The Blender camera settings that reproduce a calibration's pinhole part (blender.md §4.3 inverse; sensor fit\n"
      "HORIZONTAL; its distortion is the frames area's warp), and the calibration as a C1 camera. `record`: a\n"
      "calibration as ballasim.blender.calibration reads it. ValueError naming CAM-MODEL or CAM-CALIB otherwise.");
  m.def("blender_statics_options", [](nb::dict options) { return py_of(bl::options_record(options_in(options))); },
        "options"_a, "Every statics option (blender.md §3.11's conversion rows) with its default where unset.");
  m.def("blender_camera_names", &bl::camera_names, "name_fulls"_a,
        "The C1 names of cameras by their Blender name_full (blender.md §3.4; 'default' is reserved).");
  m.def("motion_session", &motion_session, "doc"_a, "arrays"_a, "digest"_a, "options"_a, "session"_a,
        "oracle"_a = nb::none(), "mesh_oracle"_a = nb::none(), "on_bodies"_a = nb::none(), "on_simplify"_a = nb::none(),
        "on_progress"_a = nb::none(), "textures"_a = nb::none(), "calibrations"_a = nb::none(), "log"_a = nb::none(),
        "The Blender motion session (blender.md §3.5, §5.1; src/scene/motion/session.h) of a checked exchange. `oracle`\n"
        "answers a `sample` request, oracle(F: int32[n], s: float32[n]) -> (c_final: float64[n], m: float32[n, B, 12],\n"
        "present: uint8[n, B], unknown: uint32[n][, unknown_keys: str]), `mesh_oracle(objects, F, s) -> (table:\n"
        "int64[rows, 4], co: float32[...])` a `mesh_probe`; `on_bodies(objects: int32[B])` and `on_simplify(bool)` are the\n"
        "`bodies` and `set` messages, `on_progress(fraction, stage)` the progress. With `log` (a previous result's)\n"
        "the session replays it instead (BRG-REPLAY when its requests differ). Returns the refusal flag and report, the\n"
        "origin_world, the trajectories (K, t0_ns, h_ns, rot, pos, span_ns, pieces, holds), their indices per statics\n"
        "body and camera, the cameras' T_rig_cam, session.json's record, done.summary, the perf record and the log.");
  nb::module_ t = m.def_submodule("blender_testing", "Pieces of the Blender statics conversion, for the tests.");
  t.def("polar", [](const std::vector<double>& a) {
    if (a.size() != 9) throw std::invalid_argument("polar: 9 values");
    std::array<double, 9> m9{};
    std::copy(a.begin(), a.end(), m9.begin());
    const bl::Polar p = bl::polar(m9);
    return nb::make_tuple(std::vector<double>(p.R.begin(), p.R.end()), std::vector<double>(p.S.begin(), p.S.end()),
                          p.mirror);
  });
  t.def("crop_mesh", [](nb::dict mesh, const std::vector<double>& T, double r) {
    geo::Mesh g;
    g.positions = nb::cast<std::vector<float>>(nb::module_::import_("numpy").attr("ravel")(mesh["positions"]));
    g.indices = nb::cast<std::vector<uint32_t>>(nb::module_::import_("numpy").attr("ravel")(mesh["indices"]));
    if (!mesh["normals"].is_none()) {
      g.normals = nb::cast<std::vector<float>>(nb::module_::import_("numpy").attr("ravel")(mesh["normals"]));
      g.has_normals = true;
    }
    if (!mesh["uvs"].is_none()) {
      g.uvs = nb::cast<std::vector<float>>(nb::module_::import_("numpy").attr("ravel")(mesh["uvs"]));
      g.has_uvs = true;
    }
    geo::M4 m4{};
    std::copy(T.begin(), T.end(), m4.begin());
    bl::Cropped c = bl::crop_mesh(g, m4, r);
    nb::dict d = mesh_dict(std::move(c.mesh));
    d["dropped_area_m2"] = c.dropped_area_m2;
    return d;
  });
  t.def("split_mesh", [](nb::dict mesh, nb::dict arrays, const std::vector<double>& M, const std::vector<double>& o,
                         int slots) {
    nb::dict doc;
    nb::list meshes;
    meshes.append(mesh);
    nb::list version;
    version.append(1);
    version.append(0);
    doc["version"] = version;
    doc["meshes"] = meshes;
    doc["objects"] = nb::list();
    doc["cameras"] = nb::list();
    bl::ExchangeIn ex = convert_exchange(doc, arrays, "");
    std::array<double, 9> m9{};
    std::array<double, 3> o3{};
    std::copy(M.begin(), M.end(), m9.begin());
    std::copy(o.begin(), o.end(), o3.begin());
    geo::Mesh g;
    double seconds = 0.0;
    {
      nb::gil_scoped_release release;
      const auto t0 = std::chrono::steady_clock::now();
      std::vector<bl::SplitMesh> parts{bl::split_mesh(ex.meshes.front(), m9, o3, slots)};
      std::array<double, 3> c{};
      double rho = 0.0;
      g = bl::finish_mesh(parts, std::vector<int>{0}, std::max(slots, 1), &c, &rho);
      seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    return nb::make_tuple(mesh_dict(std::move(g)), seconds);
  });
  t.def(
      "blender_camera",
      [](nb::dict props, nb::dict render, double fisheye_fit_tol_px) {
        bl::ExchangeIn ex;
        ex.render = tree_of(render);
        const bl::BlenderModel bm = bl::blender_camera(tree_of(props), bl::render_size(ex), fisheye_fit_tol_px,
                                                       "props");
        nb::dict d = camera_dict(bm.cam);
        d["pixel_pitch_m"] = bm.pitch_m;
        nb::list notes;
        for (const bl::ReportItem& it : bm.notes) notes.append(nb::make_tuple(it.code, it.severity, it.bound));
        d["notes"] = notes;
        d["refusal"] = bm.refusal ? nb::cast(bm.refusal->message) : nb::none();
        return d;
      },
      "props"_a, "render"_a, "fisheye_fit_tol_px"_a = 0.05);
  t.def("fit_lens_polynomial", [](const std::vector<double>& k, double r_max_mm, double theta_cap) {
    if (k.size() != 5) throw std::invalid_argument("fit_lens_polynomial: k0..k4");
    const bl::PolyFit f = bl::fit_lens_polynomial({k[0], k[1], k[2], k[3], k[4]}, r_max_mm, theta_cap);
    nb::dict d;
    d["f_mm"] = f.f_mm;
    d["k"] = std::vector<double>(f.k.begin(), f.k.end());
    d["max_err_mm"] = f.max_err_mm;
    d["theta_max"] = f.theta_max;
    d["samples"] = f.samples;
    return d;
  });
  t.def("rigidity_probe_objects", [](nb::dict doc, nb::dict arrays) {
    return bl::rigidity_probe_objects(convert_exchange(doc, arrays, ""));
  });
  t.def("check_rigidity", [](const std::string& object, const std::vector<float>& co0,
                             const std::vector<std::vector<float>>& probes, double tol) -> nb::object {
    const auto r = bl::check_rigidity(object, co0, probes, tol);
    if (!r) return nb::none();
    return nb::make_tuple(r->code, r->severity, r->bound);
  });
}
