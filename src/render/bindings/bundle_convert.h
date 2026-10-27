// Python objects -> renderer types, shared by the extension modules that take a scene bundle
// (ballasim._render, ballasim._engine): the C1 bundle as nested dicts of NumPy arrays or objects with
// the same attribute names, and ground-truth samples as a dict of arrays.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings/ndarray_util.h"
#include "render/bundle.h"
#include "render/types.h"

namespace balla::render::bindings {

namespace nb = nanobind;
using namespace nb::literals;
using balla::bindings::to_numpy;

// ---------------------------------------------------------------------------------------------
// Bundle conversion
// ---------------------------------------------------------------------------------------------

inline nb::object numpy() { return nb::module_::import_("numpy"); }

inline std::optional<nb::object> find_field(nb::handle obj, const char* key) {
  if (nb::isinstance<nb::dict>(obj)) {
    nb::dict d = nb::borrow<nb::dict>(obj);
    if (d.contains(key)) return nb::borrow<nb::object>(d[key]);
    return std::nullopt;
  }
  if (nb::hasattr(obj, key)) return nb::getattr(obj, key);
  return std::nullopt;
}

inline nb::object field(nb::handle obj, const char* key, const std::string& where) {
  auto f = find_field(obj, key);
  if (!f) throw std::invalid_argument("scene bundle: " + where + " has no field '" + key + "'");
  return *f;
}

// Optional field: missing or None -> nullopt.
inline std::optional<nb::object> opt_field(nb::handle obj, const char* key) {
  auto f = find_field(obj, key);
  if (!f || f->is_none()) return std::nullopt;
  return f;
}

// Integer array `arr` (NumPy) bound for `dtype` (T): refuses values outside T's range, which the unsafe cast to
// `dtype` would wrap. Dtypes NumPy casts safely (the same or a narrower one) need no pass over the values.
template <class T>
inline void check_int_range(nb::handle np, nb::handle arr, const char* dtype, const std::string& what) {
  if (nb::cast<bool>(np.attr("can_cast")(arr.attr("dtype"), dtype))) return;
  if (nb::cast<size_t>(arr.attr("size")) == 0) return;
  T lo{}, hi{};
  if (!nb::try_cast<T>(nb::int_(np.attr("min")(arr)), lo) || !nb::try_cast<T>(nb::int_(np.attr("max")(arr)), hi)) {
    throw std::invalid_argument(what + " has values outside the range of " + dtype);
  }
}

// Array-like -> contiguous NumPy array of `dtype` with the given shape (-1 = any). Integer outputs accept only
// integer inputs whose values fit `dtype`, so floats never truncate and values never wrap into indices or times.
template <class T>
inline std::vector<T> array_field(nb::handle value, const char* dtype, std::initializer_list<long> shape,
                           const std::string& what) {
  nb::object np = numpy();
  nb::object arr = np.attr("asarray")(value);
  constexpr bool want_int = std::numeric_limits<T>::is_integer;
  if (want_int && !nb::cast<bool>(np.attr("issubdtype")(arr.attr("dtype"), np.attr("integer")))) {
    throw std::invalid_argument("scene bundle: " + what + " must be an integer array");
  }
  const auto ndim = nb::cast<size_t>(arr.attr("ndim"));
  if (ndim != shape.size()) {
    throw std::invalid_argument("scene bundle: " + what + " must have " + std::to_string(shape.size()) + " dimensions");
  }
  nb::tuple s = nb::borrow<nb::tuple>(arr.attr("shape"));
  size_t i = 0;
  for (long expected : shape) {
    const long got = nb::cast<long>(s[i++]);
    if (expected >= 0 && got != expected) {
      throw std::invalid_argument("scene bundle: " + what + " has a wrong shape");
    }
  }
  if constexpr (want_int) check_int_range<T>(np, arr, dtype, "scene bundle: " + what);
  nb::object c = np.attr("ascontiguousarray")(arr, "dtype"_a = dtype);
  auto view = nb::cast<nb::ndarray<const T, nb::c_contig, nb::device::cpu>>(c);
  return std::vector<T>(view.data(), view.data() + view.size());
}

// A Python bool or a NumPy bool_, as bundle.py accepts; anything else is refused instead of read as truthy.
inline bool bool_field(nb::handle value, const std::string& what) {
  bool out = false;
  if (nb::try_cast<bool>(value, out, false)) return out;
  if (PyObject_IsInstance(value.ptr(), numpy().attr("bool_").ptr()) == 1) return PyObject_IsTrue(value.ptr()) == 1;
  throw std::invalid_argument("scene bundle: " + what + " must be a bool");
}

// A number as bundle.py takes baked_scale: a Python int or float or a NumPy integer or floating scalar, not a bool
// (an int subclass) and nothing else with __float__, within float32's finite range (checked on the double, as
// bundle.py does, before rounding to float32).
inline float float32_field(nb::handle value, const std::string& what) {
  nb::object np = numpy();
  PyObject* v = value.ptr();
  const bool number = !PyBool_Check(v) && (PyLong_Check(v) || PyFloat_Check(v) ||
                                          PyObject_IsInstance(v, np.attr("integer").ptr()) == 1 ||
                                          PyObject_IsInstance(v, np.attr("floating").ptr()) == 1);
  double d = 0.0;
  if (!number || !nb::try_cast<double>(value, d) ||
      !(d >= 0.0 && d <= static_cast<double>(std::numeric_limits<float>::max()))) {
    throw std::invalid_argument("scene bundle: " + what + " must be finite (as float32) and >= 0");
  }
  return static_cast<float>(d);
}

// A number as bundle.py's _number_in takes it (the C1 feature "pbr"): a Python int or float or a NumPy integer or
// floating scalar, not a bool, finite and in [0, hi], checked on the double.
inline double bounded_number(nb::handle value, double hi, const std::string& what, const char* expected) {
  nb::object np = numpy();
  PyObject* v = value.ptr();
  const bool number = !PyBool_Check(v) && (PyLong_Check(v) || PyFloat_Check(v) ||
                                          PyObject_IsInstance(v, np.attr("integer").ptr()) == 1 ||
                                          PyObject_IsInstance(v, np.attr("floating").ptr()) == 1);
  double d = 0.0;
  if (!number || !nb::try_cast<double>(value, d) || !(d >= 0.0 && d <= hi && std::isfinite(d))) {
    throw std::invalid_argument("scene bundle: " + what + " must be " + expected);
  }
  return d;
}

// An int as bundle.py's _is_int takes it: a Python int or a NumPy integer, not a bool; it must fit int32.
inline int32_t int32_field(nb::handle value, const std::string& what) {
  PyObject* v = value.ptr();
  int32_t out = 0;
  if (PyBool_Check(v) || !(PyLong_Check(v) || PyObject_IsInstance(v, numpy().attr("integer").ptr()) == 1) ||
      !nb::try_cast<int32_t>(value, out)) {
    throw std::invalid_argument("scene bundle: " + what + " must be an int (int32)");
  }
  return out;
}

// A feature array as bundle.py's _check_array takes it: a NumPy array of exactly `dtype` (no list, no cast).
inline nb::object exact_array(nb::handle value, const char* dtype, const std::string& what) {
  nb::object np = numpy();
  if (PyObject_IsInstance(value.ptr(), np.attr("ndarray").ptr()) != 1 ||
      !nb::cast<bool>(value.attr("dtype").attr("__eq__")(np.attr("dtype")(dtype)))) {
    throw std::invalid_argument("scene bundle: " + what + " must be a NumPy array of " + dtype);
  }
  return nb::borrow<nb::object>(value);
}

inline std::array<double, 16> matrix4(nb::handle value, const std::string& what) {
  const auto v = array_field<double>(value, "float64", {4, 4}, what);
  std::array<double, 16> m{};
  std::copy(v.begin(), v.end(), m.begin());
  return m;
}

template <size_t N, class T>
inline std::array<T, N> vecn(nb::handle value, const char* dtype, const std::string& what) {
  const auto v = array_field<T>(value, dtype, {static_cast<long>(N)}, what);
  std::array<T, N> a{};
  std::copy(v.begin(), v.end(), a.begin());
  return a;
}

inline Material to_material(nb::handle m, const std::string& where) {
  Material mat;
  mat.base_color = vecn<3, float>(field(m, "base_color", where), "float32", where + ".base_color");
  if (auto e = opt_field(m, "emissive")) mat.emissive = vecn<3, float>(*e, "float32", where + ".emissive");
  if (auto t = opt_field(m, "texture")) {
    nb::object np = numpy();
    nb::object arr = np.attr("asarray")(*t);
    if (!nb::cast<bool>(arr.attr("dtype").attr("__eq__")(np.attr("dtype")("uint8")))) {
      throw std::invalid_argument("scene bundle: " + where + ".texture must be uint8 (sRGB-encoded)");
    }
    nb::tuple s = nb::borrow<nb::tuple>(arr.attr("shape"));
    if (nb::cast<size_t>(arr.attr("ndim")) != 3) {
      throw std::invalid_argument("scene bundle: " + where + ".texture must be H x W x C");
    }
    mat.texture_height = nb::cast<int32_t>(s[0]);
    mat.texture_width = nb::cast<int32_t>(s[1]);
    mat.texture_channels = nb::cast<int32_t>(s[2]);
    mat.texture = array_field<uint8_t>(arr, "uint8", {-1, -1, -1}, where + ".texture");
  }
  // C1 feature "baked" (scans.md §3.1): an absent key (an older dict) is the field's default, and so is None for
  // baked_groups and texture_linear. The other fields refuse None and the types bundle.py refuses (a bool is not a
  // number), and an empty baked_groups is not None, so the binding accepts what SceneBundle.validate accepts.
  if (auto sh = find_field(m, "shading")) {
    std::string name;
    if (!nb::try_cast<std::string>(*sh, name) || (name != "lit" && name != "baked")) {
      throw std::invalid_argument("scene bundle: " + where + ".shading must be 'lit' or 'baked'" +
                                  (name.empty() ? std::string() : ", got '" + name + "'"));
    }
    mat.shading = name == "baked" ? Shading::kBaked : Shading::kLit;
  }
  if (auto k = find_field(m, "baked_scale")) mat.baked_scale = float32_field(*k, where + ".baked_scale");
  if (auto g = opt_field(m, "baked_groups")) {
    nb::object arr = exact_array(*g, "float32", where + ".baked_groups");
    mat.baked_groups = array_field<float>(arr, "float32", {-1}, where + ".baked_groups");
    // Material::baked_groups reads empty as None (all in group 0); an empty array is a list of no shares.
    if (mat.baked_groups.empty()) {
      throw std::invalid_argument("scene bundle: " + where + ".baked_groups needs one share per light group (it is "
                                  "empty; None puts all in group 0)");
    }
  }
  if (auto t = opt_field(m, "texture_linear")) {
    // Read as binary16 bits: a cast to uint16 would convert the values, and array_field refuses float inputs for
    // integer outputs.
    nb::object np = numpy();
    nb::object arr = exact_array(*t, "float16", where + ".texture_linear");
    nb::tuple s = nb::borrow<nb::tuple>(arr.attr("shape"));
    if (nb::cast<size_t>(arr.attr("ndim")) != 3 || nb::cast<long>(s[2]) != 3 || nb::cast<size_t>(arr.attr("size")) == 0) {
      throw std::invalid_argument("scene bundle: " + where + ".texture_linear must be a non-empty H x W x 3 array");
    }
    nb::object bits = np.attr("ascontiguousarray")(arr).attr("view")(np.attr("uint16"));
    mat.texture_linear = array_field<uint16_t>(bits, "uint16", {-1, -1, 3}, where + ".texture_linear");
    if (mat.texture.empty()) {
      mat.texture_height = nb::cast<int32_t>(s[0]);
      mat.texture_width = nb::cast<int32_t>(s[1]);
      mat.texture_channels = 3;
    }
  }
  if (auto p = find_field(m, "texture_tile")) mat.texture_tile = int32_field(*p, where + ".texture_tile");
  // C1 features "pbr" and "nir" (P2-F, p2f.md §4.11), as the "baked" fields above: an absent key is the default, None
  // only where the default is None, and the values bundle.py refuses are refused.
  for (auto [key, out] : {std::pair{"roughness", &mat.roughness}, std::pair{"metallic", &mat.metallic},
                          std::pair{"specular", &mat.specular}}) {
    if (auto v = find_field(m, key)) {
      *out = static_cast<float>(bounded_number(*v, 1.0, where + "." + key, "a number in [0, 1]"));
    }
  }
  if (auto f = find_field(m, "nir_form")) {
    static constexpr std::array<const char*, 5> kForms{"assumed", "scale", "affine", "scalar", "texture"};
    std::string name;
    const bool str = nb::try_cast<std::string>(*f, name);
    const auto it = std::find(kForms.begin(), kForms.end(), name);
    if (!str || it == kForms.end()) {
      throw std::invalid_argument("scene bundle: " + where + ".nir_form must be assumed, scale, affine, scalar or texture");
    }
    mat.nir_form = static_cast<NirForm>(it - kForms.begin());
  }
  if (auto p = opt_field(m, "nir_params")) {
    nb::object arr = exact_array(*p, "float64", where + ".nir_params");
    mat.nir_params = array_field<double>(arr, "float64", {2}, where + ".nir_params");
  }
  if (auto t = opt_field(m, "nir_texture")) {
    nb::object arr = exact_array(*t, "uint8", where + ".nir_texture");
    mat.nir_texture = array_field<uint8_t>(arr, "uint8", {-1, -1}, where + ".nir_texture");
    nb::tuple s = nb::borrow<nb::tuple>(arr.attr("shape"));
    mat.nir_texture_height = nb::cast<int32_t>(s[0]);
    mat.nir_texture_width = nb::cast<int32_t>(s[1]);
    if (mat.nir_texture.empty()) throw std::invalid_argument("scene bundle: " + where + ".nir_texture is empty");
  }
  return mat;
}

inline Body to_body(nb::handle b, size_t index) {
  const std::string where = "bodies[" + std::to_string(index) + "]";
  Body body;
  body.id = nb::cast<uint32_t>(field(b, "id", where));
  if (auto n = opt_field(b, "name")) body.name = nb::cast<std::string>(*n);
  if (auto s = opt_field(b, "semantic")) body.semantic = nb::cast<uint32_t>(*s);
  nb::object mesh = field(b, "mesh", where);
  const std::string mw = where + ".mesh";
  body.mesh.positions = array_field<float>(field(mesh, "positions", mw), "float32", {-1, 3}, mw + ".positions");
  body.mesh.indices = array_field<uint32_t>(field(mesh, "indices", mw), "uint32", {-1, 3}, mw + ".indices");
  if (auto n = opt_field(mesh, "normals")) body.mesh.normals = array_field<float>(*n, "float32", {-1, 3}, mw + ".normals");
  if (auto u = opt_field(mesh, "uvs")) body.mesh.uvs = array_field<float>(*u, "float32", {-1, 2}, mw + ".uvs");
  if (auto m = opt_field(mesh, "material_ids")) {
    body.mesh.material_ids = array_field<uint16_t>(*m, "uint16", {-1}, mw + ".material_ids");
  }
  if (auto mats = opt_field(b, "materials")) {
    size_t i = 0;
    for (nb::handle m : *mats) body.materials.push_back(to_material(m, where + ".materials[" + std::to_string(i++) + "]"));
  }
  if (auto t = opt_field(b, "trajectory")) body.trajectory = nb::cast<int32_t>(*t);
  if (auto t = opt_field(b, "T_world_body")) body.T_world_body = matrix4(*t, where + ".T_world_body");
  if (auto s = find_field(b, "scan")) body.scan = bool_field(*s, where + ".scan");
  if (auto r = find_field(b, "normal_prefilter_m")) {
    body.normal_prefilter_m = bounded_number(*r, std::numeric_limits<double>::infinity(), where + ".normal_prefilter_m",
                                             "a finite number >= 0 (metres)");
  }
  return body;
}

// C1 version 2's Trajectory.pieces (blender.md §3.7): an int64 (P, 6) NumPy array, as bundle.py's _check_array takes it
// (no list, no cast), with P >= 2 (None is one piece; an empty table is refused rather than read as None). Control rows
// must fit int32 and joins int8 before they narrow; render::validate checks the piece rules.
inline std::vector<TrajectoryPiece> to_pieces(nb::handle value, const std::string& where) {
  nb::object arr = exact_array(value, "int64", where);
  const auto v = array_field<int64_t>(arr, "int64", {-1, 6}, where);
  const size_t n = v.size() / 6;
  if (n < 2) throw std::invalid_argument("scene bundle: " + where + " needs at least 2 pieces (None for one)");
  std::vector<TrajectoryPiece> out(n);
  for (size_t p = 0; p < n; ++p) {
    const int64_t* r = v.data() + 6 * p;
    if (r[3] < 0 || r[3] > std::numeric_limits<int32_t>::max() || r[4] < 0 ||
        r[4] > std::numeric_limits<int32_t>::max() || r[5] < -1 || r[5] > 1) {
      throw std::invalid_argument("scene bundle: " + where + "[" + std::to_string(p) +
                                  "]: control rows must be in [0, 2^31) and join in {-1, 0, 1}");
    }
    out[p] = {r[0], r[1], r[2], static_cast<int32_t>(r[3]), static_cast<int32_t>(r[4]), static_cast<int8_t>(r[5])};
  }
  return out;
}

inline Trajectory to_trajectory(nb::handle t, size_t index) {
  const std::string where = "trajectories[" + std::to_string(index) + "]";
  Trajectory tr;
  tr.order = nb::cast<int32_t>(field(t, "K", where));
  tr.t0_ns = nb::cast<int64_t>(field(t, "t0_ns", where));
  tr.h_ns = nb::cast<int64_t>(field(t, "h_ns", where));
  tr.rot = array_field<double>(field(t, "rot", where), "float64", {-1, 4}, where + ".rot");
  tr.pos = array_field<double>(field(t, "pos", where), "float64", {-1, 3}, where + ".pos");
  const auto span = array_field<int64_t>(field(t, "span_ns", where), "int64", {2}, where + ".span_ns");
  tr.span_begin_ns = span[0];
  tr.span_end_ns = span[1];
  if (auto p = opt_field(t, "pieces")) tr.pieces = to_pieces(*p, where + ".pieces");
  return tr;
}

inline Camera to_camera(nb::handle c, size_t index) {
  const std::string where = "cameras[" + std::to_string(index) + "]";
  Camera cam;
  if (auto n = opt_field(c, "name")) cam.name = nb::cast<std::string>(*n);
  const auto model = nb::cast<std::string>(field(c, "model", where));
  if (model == "radtan") {
    cam.model = CameraModelKind::kRadtan;
  } else if (model == "kb") {
    cam.model = CameraModelKind::kKannalaBrandt;
  } else {
    throw std::invalid_argument("scene bundle: " + where + ".model must be 'radtan' or 'kb'");
  }
  cam.width = nb::cast<int32_t>(field(c, "width", where));
  cam.height = nb::cast<int32_t>(field(c, "height", where));
  cam.fx = nb::cast<double>(field(c, "fx", where));
  cam.fy = nb::cast<double>(field(c, "fy", where));
  cam.cx = nb::cast<double>(field(c, "cx", where));
  cam.cy = nb::cast<double>(field(c, "cy", where));
  if (auto d = opt_field(c, "dist")) cam.dist = array_field<double>(*d, "float64", {-1}, where + ".dist");
  cam.rig_trajectory = nb::cast<int32_t>(field(c, "rig_trajectory", where));
  if (auto t = opt_field(c, "T_rig_cam")) cam.T_rig_cam = matrix4(*t, where + ".T_rig_cam");
  return cam;
}

inline Light to_light(nb::handle l, size_t index) {
  const std::string where = "lights[" + std::to_string(index) + "]";
  Light light;
  const auto kind = nb::cast<std::string>(field(l, "kind", where));
  if (kind == "point") {
    light.kind = BundleLightKind::kPoint;
  } else if (kind == "spot") {
    light.kind = BundleLightKind::kSpot;
  } else if (kind == "directional") {
    light.kind = BundleLightKind::kDirectional;
  } else if (kind == "ambient") {
    light.kind = BundleLightKind::kAmbient;
  } else if (kind == "background") {
    light.kind = BundleLightKind::kBackground;
  } else if (kind == "rect") {
    light.kind = BundleLightKind::kRect;
  } else if (kind == "disk") {
    light.kind = BundleLightKind::kDisk;
  } else {
    throw std::invalid_argument("scene bundle: " + where +
                                ".kind must be point, spot, directional, ambient, background, rect or disk");
  }
  if (auto g = opt_field(l, "group")) light.group = nb::cast<int32_t>(*g);
  if (auto c = opt_field(l, "color")) light.color = vecn<3, float>(*c, "float32", where + ".color");
  light.intensity = nb::cast<float>(field(l, "intensity", where));
  if (auto b = opt_field(l, "body")) light.body = nb::cast<int32_t>(*b);
  if (auto p = opt_field(l, "position")) light.position = vecn<3, double>(*p, "float64", where + ".position");
  if (auto d = opt_field(l, "direction")) light.direction = vecn<3, double>(*d, "float64", where + ".direction");
  if (auto a = opt_field(l, "spot_inner_rad")) light.spot_inner_rad = nb::cast<float>(*a);
  if (auto a = opt_field(l, "spot_outer_rad")) light.spot_outer_rad = nb::cast<float>(*a);
  if (auto s = opt_field(l, "casts_shadows")) light.casts_shadows = nb::cast<bool>(*s);
  if (auto p = find_field(l, "proxy")) light.proxy = bool_field(*p, where + ".proxy");
  // C1 feature "area_light" (P2-F, p2f.md §4.6): None is the default, as in bundle.py.
  if (auto z = opt_field(l, "size")) {
    light.size = vecn<2, double>(exact_array(*z, "float64", where + ".size"), "float64", where + ".size");
  }
  if (auto a = opt_field(l, "axis")) {
    light.axis = vecn<3, double>(exact_array(*a, "float64", where + ".axis"), "float64", where + ".axis");
  }
  return light;
}

inline SceneBundle to_bundle(nb::handle b) {
  SceneBundle s;
  if (auto n = opt_field(b, "name")) s.name = nb::cast<std::string>(*n);
  s.t_start_ns = nb::cast<int64_t>(field(b, "t_start_ns", "bundle"));
  s.t_end_ns = nb::cast<int64_t>(field(b, "t_end_ns", "bundle"));
  if (auto o = opt_field(b, "origin_world")) s.origin_world = vecn<3, double>(*o, "float64", "origin_world");
  size_t i = 0;
  for (nb::handle body : field(b, "bodies", "bundle")) s.bodies.push_back(to_body(body, i++));
  i = 0;
  for (nb::handle t : field(b, "trajectories", "bundle")) s.trajectories.push_back(to_trajectory(t, i++));
  i = 0;
  for (nb::handle c : field(b, "cameras", "bundle")) s.cameras.push_back(to_camera(c, i++));
  if (auto groups = opt_field(b, "light_groups")) {
    for (nb::handle g : *groups) {
      auto name = find_field(g, "name");
      s.light_groups.push_back(name ? nb::cast<std::string>(*name) : nb::cast<std::string>(nb::str(g)));
    }
  }
  if (s.light_groups.empty()) s.light_groups.push_back("default");
  i = 0;
  if (auto lights = opt_field(b, "lights")) {
    for (nb::handle l : *lights) s.lights.push_back(to_light(l, i++));
  }
  return s;
}

// ---------------------------------------------------------------------------------------------
// Query arguments
// ---------------------------------------------------------------------------------------------

// 1-D integer array (or scalar broadcast to `n` when n > 0) -> vector<T>; floats are rejected.
template <class T>
inline std::vector<T> int_vector(nb::handle value, const char* dtype, const char* what, size_t n = 0) {
  nb::object np = numpy();
  nb::object arr = np.attr("asarray")(value);
  if (!nb::cast<bool>(np.attr("issubdtype")(arr.attr("dtype"), np.attr("integer")))) {
    throw std::invalid_argument(std::string(what) + " must be an integer array");
  }
  const auto ndim = nb::cast<size_t>(arr.attr("ndim"));
  if (ndim == 0 && n > 0) arr = np.attr("full")(n, arr);
  else if (ndim != 1) throw std::invalid_argument(std::string(what) + " must be one-dimensional");
  // Range check before the cast so that out-of-range values raise instead of wrapping.
  check_int_range<T>(np, arr, dtype, what);
  nb::object c = np.attr("ascontiguousarray")(arr, "dtype"_a = dtype);
  auto view = nb::cast<nb::ndarray<const T, nb::c_contig, nb::device::cpu>>(c);
  return std::vector<T>(view.data(), view.data() + view.size());
}

inline std::vector<float> float_vector(nb::handle value, const char* what) {
  nb::object np = numpy();
  nb::object arr = np.attr("asarray")(value);
  if (nb::cast<size_t>(arr.attr("ndim")) != 1) throw std::invalid_argument(std::string(what) + " must be one-dimensional");
  nb::object c = np.attr("ascontiguousarray")(arr, "dtype"_a = "float32");
  auto view = nb::cast<nb::ndarray<const float, nb::c_contig, nb::device::cpu>>(c);
  return std::vector<float>(view.data(), view.data() + view.size());
}

// ---------------------------------------------------------------------------------------------
// Ground truth
// ---------------------------------------------------------------------------------------------

inline nb::dict gt_dict(std::vector<GtSample>&& gt, std::initializer_list<size_t> shape) {
  const size_t n = gt.size();
  std::vector<uint32_t> instance(n);
  std::vector<uint16_t> semantic(n);
  std::vector<float> depth(n), fx(n), fy(n);
  for (size_t i = 0; i < n; ++i) {
    instance[i] = gt[i].instance;
    semantic[i] = static_cast<uint16_t>(gt[i].semantic);
    depth[i] = gt[i].depth;
    fx[i] = gt[i].flow_x;
    fy[i] = gt[i].flow_y;
  }
  nb::dict d;
  d["instance"] = to_numpy(std::move(instance), shape);
  d["semantic"] = to_numpy(std::move(semantic), shape);
  d["depth"] = to_numpy(std::move(depth), shape);
  d["flow_x"] = to_numpy(std::move(fx), shape);
  d["flow_y"] = to_numpy(std::move(fy), shape);
  return d;
}

}  // namespace balla::render::bindings
