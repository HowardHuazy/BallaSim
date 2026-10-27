// The statics of a Blender exchange (docs/design/phase3/blender.md §3.4, §4.1, §4.6, §4.8, §6.1): which objects
// become bodies (light carriers included), their ids and semantics, their meshes, the camera names and the report;
// the cameras (camera.cpp) and the lights (lights.cpp) last.
#include "scene/blender/convert.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "scene/npla.h"

namespace balla::scene::blender {

namespace {

using M9 = std::array<double, 9>;

M9 linear(const std::array<double, 16>& m) { return {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]}; }

M9 mul(const M9& a, const M9& b) {
  M9 c{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) c[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
  }
  return c;
}

M9 transpose(const M9& a) { return {a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]}; }

double det3(const M9& m) {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

// [R | t] as a row-major 4 x 4.
geo::M4 rigid(const M9& R, const std::array<double, 3>& t) {
  return {R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2], 0.0, 0.0, 0.0, 1.0};
}

const Tree* entry(const Tree& t, const char* key) { return t.kind() == Tree::Kind::kDict ? t.find(key) : nullptr; }

bool flag(const Tree& t, const char* key, bool fallback) {
  const Tree* v = entry(t, key);
  return v != nullptr && v->kind() == Tree::Kind::kBool ? v->as_bool() : fallback;
}

render::Material default_material() {
  render::Material m;
  m.base_color = {0.8f, 0.8f, 0.8f};
  return m;
}

bool geometry_type(const std::string& t) {
  return t == "MESH" || t == "CURVE" || t == "SURFACE" || t == "META" || t == "FONT";
}

// Object types Cycles renders as geometry, the unsupported ones (GEO-CURVES) included.
bool renders_geometry(const std::string& t) {
  return geometry_type(t) || t == "CURVES" || t == "POINTCLOUD" || t == "VOLUME";
}

// The name of a legacy particle system the object renders as strands (a PARTICLE_SYSTEM modifier shown for render
// whose settings' render_type is PATH: hair or paths, which object_instances does not list), or "".
std::string strand_system(const ObjectIn& ob) {
  const Tree* mods = ob.deform.kind() == Tree::Kind::kDict ? ob.deform.find("modifiers") : nullptr;
  if (mods == nullptr || mods->kind() != Tree::Kind::kList) return "";
  for (const Tree& md : mods->items()) {
    if (md.kind() != Tree::Kind::kDict) continue;
    const Tree* ty = md.find("type");
    const Tree* rt = md.find("render_type");
    const Tree* name = md.find("name");
    if (ty != nullptr && ty->kind() == Tree::Kind::kString && ty->as_string() == "PARTICLE_SYSTEM" &&
        flag(md, "show_render", true) && rt != nullptr && rt->kind() == Tree::Kind::kString &&
        rt->as_string() == "PATH") {
      return name != nullptr && name->kind() == Tree::Kind::kString ? name->as_string() : std::string("?");
    }
  }
  return "";
}

bool singular(const M9& a) {
  const double d = det3(a);
  return !(std::abs(d) > 0.0) || !std::isfinite(d);
}

}  // namespace

double ExchangeIn::scale_length() const {
  const Tree* v = entry(units, "scale_length");
  if (v == nullptr) return 1.0;
  const double s = v->kind() == Tree::Kind::kInt ? static_cast<double>(v->as_int()) : v->as_float();
  if (!(s > 0.0) || !std::isfinite(s)) throw std::invalid_argument("BRG-PROTO: units.scale_length must be > 0");
  return s;
}

bool Statics::refused() const {
  return std::any_of(report.begin(), report.end(), [](const ReportItem& r) { return r.severity == "refuse"; });
}

const char* bound_kind_name(BoundKind k) {
  return k == BoundKind::kMeasured ? "measured" : k == BoundKind::kSampled ? "sampled" : "none";
}

Tree options_record(const ConvertOptions& o) {
  Tree t = Tree::dict();
  const auto opt = [](const std::optional<double>& v) { return v ? Tree::real(*v) : Tree::null(); };
  t["policy"] = Tree::string(o.policy == Policy::kStrict ? "strict" : "lenient");
  t["lm_per_w"] = opt(o.lm_per_w);
  t["world_as_ambient"] = Tree::boolean(o.world_as_ambient);
  t["crop_radius_m"] = opt(o.crop_radius_m);
  t["max_triangles"] = Tree::integer(o.max_triangles);
  t["calib_tol_px"] = Tree::real(o.calib_tol_px);
  t["fisheye_fit_tol_px"] = Tree::real(o.fisheye_fit_tol_px);
  t["rigidity_tol_m"] = Tree::real(o.rigidity_tol_m);
  return t;
}

std::vector<std::string> camera_names(const std::vector<std::string>& name_fulls) {
  std::set<std::string> taken{"default"};
  std::vector<std::string> out;
  for (const std::string& full : name_fulls) {
    std::string base;
    for (const char ch : full) {
      if ((static_cast<unsigned char>(ch) & 0xC0) == 0x80) continue;   // UTF-8 continuation: one '_' per character
      const bool ok = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' ||
                      ch == '-';
      base.push_back(ok ? ch : '_');
    }
    if (base.empty()) base = "camera";
    std::string name = base;
    for (int k = 2; taken.count(name) != 0; ++k) name = base + "_" + std::to_string(k);
    taken.insert(name);
    out.push_back(name);
  }
  return out;
}

Polar polar(const M9& a) {
  if (singular(a)) throw std::invalid_argument("polar: singular matrix");
  const bool mirror = det3(a) < 0.0;
  M9 b = a;
  if (mirror) {
    for (double& x : b) x = -x;
  }
  M9 u{}, vt{};
  double sv[3];
  npo::svd(b.data(), 3, 3, u.data(), sv, vt.data());
  const M9 R = mul(u, vt);
  return {R, mul(transpose(R), a), mirror};
}

Tree report_doc(const std::vector<ReportItem>& items, Policy policy, const Tree& matched) {
  Tree doc = Tree::dict();
  doc["format"] = Tree::string("ballasim.blender_conversion_report");
  doc["version"] = Tree::ints({1, 0});
  doc["converter"] = Tree::integer(kConverterVersion);
  doc["policy"] = Tree::string(policy == Policy::kStrict ? "strict" : "lenient");
  Tree list = Tree::list();
  std::map<std::string, int64_t> count;
  for (const ReportItem& r : items) {
    Tree it = Tree::dict();
    it["code"] = Tree::string(r.code);
    it["severity"] = Tree::string(r.severity);
    it["object"] = Tree::string(r.object);
    it["message"] = Tree::string(r.message);
    it["bound"] = Tree::real(r.bound);
    it["bound_kind"] = Tree::string(bound_kind_name(r.bound_kind));
    list.push_back(std::move(it));
    ++count[r.severity];
  }
  doc["items"] = std::move(list);
  doc["matched_overrides"] = matched;
  Tree sum = Tree::dict();
  for (const auto& [key, sev] : {std::pair{"refused", "refuse"}, std::pair{"warnings", "warn"},
                                 std::pair{"exact", "exact"}, std::pair{"approx", "approx"}, std::pair{"info", "info"},
                                 std::pair{"errors", "error"}}) {
    sum[key] = Tree::integer(count[sev]);
  }
  doc["summary"] = std::move(sum);
  return doc;
}

Tree assets_doc(const std::vector<AssetUse>& assets) {
  Tree list = Tree::list();
  for (const AssetUse& a : assets) {
    Tree d = Tree::dict();
    d["kind"] = Tree::string(a.kind);
    d["index"] = Tree::integer(a.index);
    d["name"] = Tree::string(a.name);
    for (const char* key : {"license", "copyright", "author"}) {
      const Tree* v = a.record.find(key);
      d[key] = v != nullptr && v->kind() == Tree::Kind::kString ? *v : Tree::null();
    }
    Tree bodies = Tree::list();
    for (const std::string& n : a.bodies) bodies.push_back(Tree::string(n));
    d["bodies"] = std::move(bodies);
    list.push_back(std::move(d));
  }
  return list;
}

Statics convert_statics(const ExchangeIn& ex, const ConvertOptions& o) {
  Statics st;
  const auto n = static_cast<int32_t>(ex.objects.size());
  const auto proto = [](const std::string& what) { throw std::invalid_argument("BRG-PROTO: " + what); };
  const auto item = [&st](const char* code, const std::string& sev, const std::string& obj, const std::string& msg) {
    st.report.push_back({code, sev, obj, msg, 0.0, BoundKind::kNone});
  };
  const bool strict = o.policy == Policy::kStrict;

  // Indices and keys (the add-on's contract, §3.4).
  std::set<std::string> keys;
  for (int32_t i = 0; i < n; ++i) {
    const ObjectIn& ob = ex.objects[static_cast<size_t>(i)];
    for (const int32_t j : {ob.instancer, ob.parent, ob.instance_object}) {
      if (j < -1 || j >= n) proto("objects[" + std::to_string(i) + "] refers to object " + std::to_string(j));
    }
    const std::string where = "objects[" + std::to_string(i) + "]";
    if (ob.mesh < -1 || ob.mesh >= static_cast<int32_t>(ex.meshes.size())) proto(where + ".mesh out of range");
    if (ob.is_instance != (ob.instancer >= 0)) proto(where + ": an instance needs its instancer");
    for (const int32_t ms : ob.material_slots) {
      if (ms >= static_cast<int32_t>(ex.materials.size())) proto(where + ".material_slots out of range");
    }
    for (const int32_t a : {ob.animation, ob.data_animation}) {
      if (a < -1 || a >= static_cast<int32_t>(ex.animations.items().size())) proto(where + ": animation out of range");
    }
    if (!keys.insert(ob.key).second) proto("duplicate object key " + ob.key);
  }

  // Static candidates (§4.7.1): no animation, constraint or rigid body in the parent, instancer and instance-object
  // chains. One iterative walk over those links, so a long chain cannot exhaust the stack and a cycle of any length
  // is found where it closes.
  enum : int8_t { kUnseen, kOnWalk, kStatic, kMoving };
  std::vector<int8_t> state(static_cast<size_t>(n), kUnseen);
  {
    std::vector<std::pair<int32_t, int>> walk;   // (object, next link to follow)
    for (int32_t root = 0; root < n; ++root) {
      if (state[static_cast<size_t>(root)] != kUnseen) continue;
      state[static_cast<size_t>(root)] = kOnWalk;
      walk.push_back({root, 0});
      while (!walk.empty()) {
        const int32_t i = walk.back().first;
        const ObjectIn& ob = ex.objects[static_cast<size_t>(i)];
        const std::array<int32_t, 3> links{ob.parent, ob.instancer, ob.instance_object};
        if (const int e = walk.back().second++; e < 3) {
          const int32_t j = links[static_cast<size_t>(e)];
          if (j < 0) continue;
          if (state[static_cast<size_t>(j)] == kOnWalk) proto("a parent or instancer cycle through " + ob.name_full);
          if (state[static_cast<size_t>(j)] == kUnseen) {
            state[static_cast<size_t>(j)] = kOnWalk;
            walk.push_back({j, 0});
          }
          continue;
        }
        bool still = ob.animation < 0 && ob.constraints.items().empty() && ob.rigid_body.kind() == Tree::Kind::kNull;
        for (const int32_t j : links) still = still && (j < 0 || state[static_cast<size_t>(j)] == kStatic);
        state[static_cast<size_t>(i)] = still ? kStatic : kMoving;
        walk.pop_back();
      }
    }
  }
  const auto is_static = [&state](int32_t i) { return state[static_cast<size_t>(i)] == kStatic; };
  std::vector<bool> static_objects(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) static_objects[static_cast<size_t>(i)] = is_static(i);

  std::vector<std::string> cam_full;
  for (const CameraIn& c : ex.cameras) {
    if (c.object < 0 || c.object >= n) proto("cameras[].object out of range");
    cam_full.push_back(ex.objects[static_cast<size_t>(c.object)].name_full);
  }
  st.camera_names = camera_names(cam_full);
  for (size_t k = 0; k < ex.cameras.size(); ++k) {
    if (!ex.cameras[k].c1_name.empty() && ex.cameras[k].c1_name != st.camera_names[k]) {
      proto("camera " + cam_full[k] + ": c1_name " + ex.cameras[k].c1_name + " differs from the engine's " +
            st.camera_names[k]);
    }
  }
  for (const AddonItem& a : ex.addon_report) item(a.code.c_str(), "info", a.object, a.message);

  // Light carriers (§3.7, §4.4; BL-5b runs them): a static light needs none; a light that moves with its parent rides
  // the parent's body, a meshless one when the parent has no triangles (or renders none); a light that moves on its
  // own gets a meshless carrier of its own. Both take the light's id. Only lights that convert_lights can keep get
  // one: a light hidden from render, with animated render visibility or with visible_diffuse off (refused under
  // strict, dropped under lenient) never rides a body, so no carrier is left without its light (§3.7).
  std::vector<int32_t> light_child(static_cast<size_t>(n), -1);
  std::vector<bool> own_carrier(static_cast<size_t>(n), false);
  for (const LightIn& l : ex.lights) {
    if (l.object < 0 || l.object >= n) proto("lights[].object out of range");
    const ObjectIn& lo = ex.objects[static_cast<size_t>(l.object)];
    if (!flag(lo.visibility, "render", true) || flag(lo.visibility, "hide_render_animated", false) ||
        !flag(lo.visibility, "diffuse", true)) {
      continue;
    }
    const LightMotion m = light_motion(ex, l.object, static_objects);
    if (m == LightMotion::kOwn) own_carrier[static_cast<size_t>(l.object)] = true;
    if (m == LightMotion::kParent && light_child[static_cast<size_t>(lo.parent)] < 0) {
      light_child[static_cast<size_t>(lo.parent)] = l.object;
    }
  }

  // Instance ranks per instancer, by persistent_id (§4.8).
  std::vector<int64_t> rank(static_cast<size_t>(n), -1);
  {
    std::map<int32_t, std::vector<std::pair<std::array<int64_t, 8>, int32_t>>> by_instancer;
    for (int32_t i = 0; i < n; ++i) {
      const ObjectIn& ob = ex.objects[static_cast<size_t>(i)];
      if (ob.is_instance) by_instancer[ob.instancer].push_back({ob.persistent_id, i});
    }
    for (auto& [inst, v] : by_instancer) {
      std::sort(v.begin(), v.end());
      for (size_t r = 0; r < v.size(); ++r) rank[static_cast<size_t>(v[r].second)] = static_cast<int64_t>(r);
    }
  }

  // Which objects become bodies.
  struct Plan {
    int32_t object;               // the body's object (the instancer for `inherit`)
    std::vector<int32_t> parts;   // objects whose meshes it holds
    bool carrier = false;
    int64_t id = 0;
    std::string name;
  };
  std::vector<Plan> plans;
  std::map<int32_t, size_t> inherit_plan;
  for (int32_t i = 0; i < n; ++i) {
    const ObjectIn& ob = ex.objects[static_cast<size_t>(i)];
    const int32_t child = light_child[static_cast<size_t>(i)];
    // A light parent that renders no triangles is a meshless carrier with its light's id; only after the checks
    // that refuse geometry it may render later (VIS-ANIM) or cannot convert (GEO-CURVES).
    const auto parent_carrier = [&] {
      if (child >= 0) plans.push_back({i, {}, true, ex.objects[static_cast<size_t>(child)].instance_id, ob.name_full});
    };
    if (own_carrier[static_cast<size_t>(i)]) {   // a light object with motion of its own
      plans.push_back({i, {}, true, ob.instance_id, ob.name_full});
      continue;
    }
    // Animated hide_render first: a geometry object hidden at the export frame may render later in the span (§4.9).
    if (flag(ob.visibility, "hide_render_animated", false) && (ob.mesh >= 0 || renders_geometry(ob.type))) {
      item("VIS-ANIM", "refuse", ob.name_full, "animated render visibility (BL-12)");
      continue;
    }
    if (!flag(ob.visibility, "render", true)) {   // hidden from render for the whole span: excluded
      parent_carrier();
      continue;
    }
    if (ob.type == "CURVES" || ob.type == "POINTCLOUD" || ob.type == "VOLUME") {
      item("GEO-CURVES", "refuse", ob.name_full, ob.type + " objects (hair curves, point clouds, volumes) are not "
                                                  "supported");
      continue;
    }
    if (const std::string ps = strand_system(ob); !ps.empty()) {
      item("GEO-CURVES", "refuse", ob.name_full, "particle system " + ps + " renders strands (render_type PATH: hair "
                                                 "or paths), which are not supported");
      continue;
    }
    if (ob.mesh < 0) {
      if (child >= 0) {
        parent_carrier();
      } else if (geometry_type(ob.type)) {
        item("EMPTY-MESH", "info", ob.name_full, "the evaluated mesh has no triangles: not a body");
      }
      continue;
    }
    std::string vis;
    if (!flag(ob.visibility, "camera", true)) vis = "invisible to the camera";
    if (!flag(ob.visibility, "shadow", true)) vis = "casts no shadow (visible_shadow off)";
    if (flag(ob.visibility, "holdout", false)) vis = "a holdout";
    if (flag(ob.visibility, "shadow_catcher", false)) vis = "a shadow catcher";
    if (!vis.empty()) {
      item("VIS", "refuse", ob.name_full, "the object is " + vis);
      continue;
    }
    if (flag(ob.deform, "is_viewport_node", false)) {
      item("GEO-VIEWPORT", strict ? "refuse" : "warn", ob.name_full,
           "a geometry-nodes group reads Is Viewport: render and viewport geometry may differ (no bound)" +
               std::string(strict ? "" : "; exported from the viewport branch"));
      if (strict) continue;
    }
    if (ob.is_instance) {
      const ObjectIn& in = ex.objects[static_cast<size_t>(ob.instancer)];
      if (in.instancer_ids == "inherit") {
        auto [it, fresh] = inherit_plan.emplace(ob.instancer, plans.size());
        if (fresh) plans.push_back({ob.instancer, {}, false, in.instance_id, in.name_full});
        plans[it->second].parts.push_back(i);
        continue;
      }
      if (in.instancer_ids != "per_instance") proto(in.name_full + ": instancer_ids " + in.instancer_ids);
      const int64_t r = rank[static_cast<size_t>(i)];
      const std::string name = in.name_full + "/" +
                               (ob.instance_object >= 0 ? ex.objects[static_cast<size_t>(ob.instance_object)].name_full
                                                        : ob.name_full) +
                               "[" + std::to_string(r) + "]";
      if (in.instance_id < 1 || in.instance_block < 1) {
        item("ID-RANGE", "refuse", in.name_full, "the instancer has no id block (Assign ids)");
        continue;
      }
      if (r >= in.instance_block) {
        item("ID-RANGE", "refuse", in.name_full,
             "instance rank " + std::to_string(r) + " outside the id block of " + std::to_string(in.instance_block));
        continue;
      }
      plans.push_back({i, {i}, false, in.instance_id + r, name});
      continue;
    }
    if (ob.instancer_ids == "inherit") {   // the instancer's own geometry joins its instances' body (§4.8)
      auto [it, fresh] = inherit_plan.emplace(i, plans.size());
      if (fresh) plans.push_back({i, {}, false, ob.instance_id, ob.name_full});
      plans[it->second].parts.push_back(i);
      continue;
    }
    plans.push_back({i, {i}, false, ob.instance_id, ob.name_full});
  }

  // GEO-BUDGET before any mesh work.
  for (const Plan& p : plans) {
    for (const int32_t j : p.parts) {
      const MeshIn& mesh = ex.meshes[static_cast<size_t>(ex.objects[static_cast<size_t>(j)].mesh)];
      st.triangles += static_cast<int64_t>(mesh.num_triangles());
    }
  }
  if (st.triangles > o.max_triangles) {
    item("GEO-BUDGET", "refuse", "", std::to_string(st.triangles) + " triangles after instancing exceed "
                                     "scene.blender.max_triangles = " + std::to_string(o.max_triangles));
    st.report.back().bound = static_cast<double>(st.triangles);
    st.report.back().bound_kind = BoundKind::kMeasured;
    return st;
  }

  // Ids and semantics (§4.8).
  const auto semantic_of = [&](int32_t i, std::string* err) -> int64_t {
    const ObjectIn& ob = ex.objects[static_cast<size_t>(i)];
    std::string name = ob.semantic;
    if (name.empty() && ob.is_instance) name = ex.objects[static_cast<size_t>(ob.instancer)].semantic;
    for (size_t k = 0; name.empty() && k < ob.collections.size(); ++k) {
      if (const Tree* c = entry(ex.collection_semantics, ob.collections[k].c_str());
          c != nullptr && c->kind() == Tree::Kind::kString) {
        name = c->as_string();
      }
    }
    if (name.empty()) return 0;
    const Tree* id = entry(ex.semantic_classes, name.c_str());
    if (id == nullptr || id->kind() != Tree::Kind::kInt || id->as_int() < 0 || id->as_int() > 0xFFFE) {
      *err = "semantic class " + name + " is not in semantic_classes with an id in [0, 0xFFFE]";
      return -1;
    }
    return id->as_int();
  };
  std::map<int64_t, std::string> id_owner;
  const double k = ex.scale_length();

  // Materials (§4.5), each converted once when a body first uses it (its report items with it), and their textures,
  // one per (image, MIRROR) (their items once per image).
  std::map<int32_t, MaterialOut> mats;
  const auto material = [&](int32_t mi) -> const MaterialOut& {
    auto [it, fresh] = mats.try_emplace(mi);
    if (fresh) {
      it->second = convert_material(ex.materials[static_cast<size_t>(mi)], ex.images, o);
      st.report.insert(st.report.end(), it->second.items.begin(), it->second.items.end());
    }
    return it->second;
  };
  std::map<std::pair<int32_t, bool>, int32_t> texture_index;
  std::set<int32_t> image_reported;
  const auto texture = [&](const MaterialOut& mo) -> int32_t {
    if (mo.image < 0) return -1;
    auto [it, fresh] = texture_index.try_emplace({mo.image, mo.mirror}, -1);
    if (fresh) {
      TextureResult tr = convert_texture(ex.images[static_cast<size_t>(mo.image)], mo.image, mo.mirror, o.policy);
      if (image_reported.insert(mo.image).second) st.report.insert(st.report.end(), tr.items.begin(), tr.items.end());
      if (tr.tex) {
        it->second = static_cast<int32_t>(st.textures.size());
        st.textures.push_back(std::move(*tr.tex));
      }
    }
    return it->second;
  };
  const auto default_map = [](const MeshIn& mesh) -> std::string {
    if (mesh.uvs.count(mesh.active_render_uv)) return mesh.active_render_uv;
    return mesh.uvs.size() == 1 ? mesh.uvs.begin()->first : std::string();
  };

  // Per plan: id, semantic, pose and material slots; then the meshes, each unique (mesh, S, slot counts, UV specs of
  // the textured slots) split once.
  using MeshKey = std::tuple<int32_t, M9, std::vector<int>, std::string>;
  struct Ready {
    const Plan* plan;
    BodyStatic b;
    Polar pol;
    std::array<double, 3> t;
    std::vector<int> slots, bases;   // per part: its material slot count, its first material in the body
    std::vector<std::vector<SlotUv>> slot_uv;   // per part, per slot
    MeshKey key;
  };
  std::vector<Ready> ready;
  std::map<MeshKey, int64_t> uses;
  for (const Plan& p : plans) {
    const ObjectIn& ob = ex.objects[static_cast<size_t>(p.object)];
    if (p.id < 1 || p.id > 0xFFFFFFFE) {
      item("ID-RANGE", "refuse", p.name,
           "instance id " + std::to_string(p.id) + (p.id == 0 ? " (unassigned: Assign ids)" : "") +
               " outside [1, 0xFFFFFFFE]");
      continue;
    }
    if (auto [it, fresh] = id_owner.emplace(p.id, p.name); !fresh) {
      item("ID-RANGE", "refuse", p.name, "instance id " + std::to_string(p.id) + " is also " + it->second + "'s");
      continue;
    }
    std::string err;
    const int64_t sem = semantic_of(p.object, &err);
    if (sem < 0) {
      item("ID-RANGE", "refuse", p.name, err);
      continue;
    }
    if (!std::all_of(ob.matrix_world.begin(), ob.matrix_world.end(), [](double x) { return std::isfinite(x); })) {
      item("AST-V01", "refuse", p.name, "matrix_world contains NaN or Inf");
      continue;
    }
    Ready r{&p, {}, {}, {k * ob.matrix_world[3], k * ob.matrix_world[7], k * ob.matrix_world[11]}, {}, {}, {}, {}};
    try {
      r.pol = polar(linear(ob.matrix_world));
    } catch (const std::invalid_argument&) {
      if (is_static(p.object)) {
        item("EMPTY-MESH", "info", p.name, "matrix_world is singular (zero scale): renders nothing, not a body");
      } else {   // the scale may grow over the span (§4.1: S must be constant)
        item("GEO-SCALEANIM", "refuse", p.name, "zero scale at the export frame on an animated object");
      }
      continue;
    }
    BodyStatic& b = r.b;
    b.object = p.object;
    b.id = static_cast<uint32_t>(p.id);
    b.semantic = static_cast<uint32_t>(sem);
    b.name = p.name;
    b.parts = p.parts;
    b.S = r.pol.S;
    if (p.carrier) {
      b.role = BodyStatic::Role::kCarrier;
    } else {
      // Parts with the same material_slots share their materials (an `inherit` scatter of one object: one set); no
      // slots is the default material, as one empty slot is.
      std::map<std::vector<int32_t>, int> base_of;
      for (const int32_t j : p.parts) {
        const std::vector<int32_t>& ms = ex.objects[static_cast<size_t>(j)].material_slots;
        r.slots.push_back(static_cast<int>(ms.size()));
        auto [it, fresh] = base_of.emplace(ms.empty() ? std::vector<int32_t>{-1} : ms,
                                           static_cast<int>(b.material_of.size()));
        if (fresh) b.material_of.insert(b.material_of.end(), it->first.begin(), it->first.end());
        r.bases.push_back(it->second);
        if (b.material_of.size() > static_cast<size_t>(kMaxBodyMaterials)) break;
      }
      if (b.material_of.size() > static_cast<size_t>(kMaxBodyMaterials)) {
        item("GEO-MATSLOT", "refuse", p.name,
             "more than " + std::to_string(kMaxBodyMaterials) + " distinct material slots in one body");
        continue;
      }
      // The parts' textured slots (§4.5) and their UV maps: GEO-UV2 when one material reads two maps of a mesh;
      // MAT-EMITGRP when an emitter is outside the default light group.
      std::string uv_key, uv2;
      bool emitter_outside = false;
      for (const int32_t j : p.parts) {
        const ObjectIn& pj = ex.objects[static_cast<size_t>(j)];
        const MeshIn& mesh = ex.meshes[static_cast<size_t>(pj.mesh)];
        std::vector<SlotUv>& su = r.slot_uv.emplace_back(pj.material_slots.size());
        for (size_t q = 0; q < pj.material_slots.size(); ++q) {
          if (pj.material_slots[q] < 0) continue;
          const MaterialOut& mo = material(pj.material_slots[q]);
          std::set<std::string> maps;
          for (const std::string& name : mo.uv_maps) maps.insert(name.empty() ? default_map(mesh) : name);
          if (maps.size() > 1 && uv2.empty()) uv2 = ex.materials[static_cast<size_t>(pj.material_slots[q])].name;
          if (mo.emitter() && !pj.lightgroup.empty() && pj.lightgroup != "default") emitter_outside = true;
          if (mo.image < 0) continue;
          su[q] = {true, mo.unit_uv, mo.uv_map, mo.uv};
          uv_key += std::to_string(q) + ":" + mo.uv_map + (mo.unit_uv ? "|1|" : "|0|");
          uv_key.append(reinterpret_cast<const char*>(mo.uv.data()), sizeof(double) * mo.uv.size());
        }
        uv_key += ";";
      }
      if (!uv2.empty()) {
        item("GEO-UV2", "refuse", p.name, "material " + uv2 + " reads two UV maps of the mesh");
        continue;
      }
      if (emitter_outside) {
        item("MAT-EMITGRP", "refuse", p.name, "an emissive material outside the default light group (C1 emitters "
                                              "are in group 0)");
        continue;
      }
      const bool single = p.parts.size() == 1 && p.parts.front() == p.object;
      r.key = MeshKey(single ? ob.mesh : -1 - p.object, r.pol.S, r.slots, uv_key);
      ++uses[r.key];
    }
    ready.push_back(std::move(r));
  }

  std::map<std::pair<std::string, int32_t>, size_t> asset_of;
  std::map<MeshKey, std::tuple<geo::Mesh, std::array<double, 3>, double>> mesh_cache;
  std::map<MeshKey, ReportItem> failed;
  for (Ready& r : ready) {
    const Plan& p = *r.plan;
    BodyStatic& b = r.b;
    if (!p.carrier) {
      const int64_t left = --uses[r.key];
      if (auto f = failed.find(r.key); f != failed.end()) {
        st.report.push_back(f->second);
        st.report.back().object = p.name;
        continue;
      }
      auto it = mesh_cache.find(r.key);
      if (it == mesh_cache.end()) {
        std::vector<SplitMesh> parts;
        try {
          for (const int32_t j : p.parts) {
            const ObjectIn& pj = ex.objects[static_cast<size_t>(j)];
            M9 M{};
            std::array<double, 3> off{0.0, 0.0, 0.0};
            if (j == p.object) {
              for (size_t q = 0; q < 9; ++q) M[q] = k * r.pol.S[q];
            } else {   // `inherit`: the part in the instancer's rigid frame
              const M9 Rt = transpose(r.pol.R);
              M = mul(Rt, linear(pj.matrix_world));
              for (double& x : M) x *= k;
              const std::array<double, 3> d{k * pj.matrix_world[3] - r.t[0], k * pj.matrix_world[7] - r.t[1],
                                            k * pj.matrix_world[11] - r.t[2]};
              for (size_t q = 0; q < 3; ++q) off[q] = Rt[3 * q] * d[0] + Rt[3 * q + 1] * d[1] + Rt[3 * q + 2] * d[2];
              if (!std::all_of(pj.matrix_world.begin(), pj.matrix_world.end(),
                               [](double x) { return std::isfinite(x); })) {
                throw GeometryError("AST-V01", pj.name_full + ": matrix_world contains NaN or Inf");
              }
              if (singular(M)) {
                if (!is_static(j)) {
                  throw GeometryError("GEO-SCALEANIM", pj.name_full + ": zero scale at the export frame on an "
                                                                      "animated instance");
                }
                item("EMPTY-MESH", "info", pj.name_full, "matrix_world is singular (zero scale): renders nothing");
              }
            }
            const size_t part = parts.size();
            parts.push_back(split_mesh(ex.meshes[static_cast<size_t>(pj.mesh)], M, off, r.slots[part],
                                       r.slot_uv[part]));
            if (parts.back().degenerate > 0) {
              item("AST-V02", "info", pj.name_full,
                   std::to_string(parts.back().degenerate) + " degenerate triangles dropped");
            }
            if (parts.back().duplicate > 0) {
              item("AST-V04", "info", pj.name_full,
                   std::to_string(parts.back().duplicate) + " duplicate triangles dropped");
            }
          }
        } catch (const GeometryError& e) {
          item(e.code.c_str(), "refuse", p.name, e.what());
          failed.emplace(r.key, st.report.back());
          continue;
        }
        std::array<double, 3> c{};
        double rho = 0.0;
        geo::Mesh g = finish_mesh(parts, r.bases, static_cast<int>(b.material_of.size()), &c, &rho);
        it = mesh_cache.emplace(r.key, std::make_tuple(std::move(g), c, rho)).first;
      }
      if (left == 0) {   // its last body: move the mesh out instead of holding two copies
        std::tie(b.mesh, b.c, b.rho) = std::move(it->second);
        mesh_cache.erase(it);
      } else {
        std::tie(b.mesh, b.c, b.rho) = it->second;
      }
      if (b.mesh.num_triangles() == 0) {
        if (light_child[static_cast<size_t>(p.object)] < 0) {
          item("EMPTY-MESH", "info", p.name, "every triangle is degenerate: not a body");
          continue;
        }
        b.role = BodyStatic::Role::kCarrier;   // a light's parent whose triangles are all degenerate: a carrier
        b.mesh = geo::Mesh{};
        b.material_of.clear();
        b.c = {0.0, 0.0, 0.0};
        b.rho = 0.0;
        b.parts.clear();
      }
      for (const int32_t mi : b.material_of) {
        if (mi < 0) {
          b.materials.push_back(default_material());
          b.texture_of.push_back(-1);
          continue;
        }
        const MaterialOut& mo = material(mi);
        b.materials.push_back(mo.mat);
        b.texture_of.push_back(texture(mo));
        if (b.texture_of.back() >= 0) {
          const TextureOut& tx = st.textures[static_cast<size_t>(b.texture_of.back())];
          b.materials.back().texture_width = tx.width;
          b.materials.back().texture_height = tx.height;
          b.materials.back().texture_channels = 3;
        }
      }
    }
    if (b.materials.empty()) {
      b.materials.push_back(default_material());
      b.texture_of.assign(1, -1);
      b.material_of.assign(1, -1);
    }
    // Provenance (PHASE3.md H4): the IDs the body uses that carry an `asset` record.
    const auto uses_asset = [&](const char* kind, int32_t index, const std::string& name, const Tree& rec) {
      if (rec.kind() != Tree::Kind::kDict) return;
      auto [it, fresh] = asset_of.try_emplace({kind, index}, st.assets.size());
      if (fresh) st.assets.push_back(AssetUse{kind, name, index, rec, {}});
      std::vector<std::string>& bodies = st.assets[it->second].bodies;
      if (bodies.empty() || bodies.back() != b.name) bodies.push_back(b.name);
    };
    std::vector<int32_t> objs{p.object};
    objs.insert(objs.end(), p.parts.begin(), p.parts.end());
    for (const int32_t j : objs) {
      const ObjectIn& pj = ex.objects[static_cast<size_t>(j)];
      uses_asset("object", j, pj.name_full, pj.asset);
      for (const int32_t q : {pj.instance_object, pj.instancer}) {   // what an instance shows and what makes it
        if (q >= 0) uses_asset("object", q, ex.objects[static_cast<size_t>(q)].name_full,
                               ex.objects[static_cast<size_t>(q)].asset);
      }
      if (pj.mesh >= 0) uses_asset("mesh", pj.mesh, ex.meshes[static_cast<size_t>(pj.mesh)].id,
                                   ex.meshes[static_cast<size_t>(pj.mesh)].asset);
    }
    for (size_t q = 0; q < b.material_of.size(); ++q) {
      const int32_t mi = b.material_of[q];
      if (mi < 0) continue;
      const MaterialIn& mat = ex.materials[static_cast<size_t>(mi)];
      uses_asset("material", mi, mat.name, mat.asset);
      if (const int32_t im = material(mi).image; im >= 0) {
        uses_asset("image", im, ex.images[static_cast<size_t>(im)].id, ex.images[static_cast<size_t>(im)].asset);
      }
    }
    const Polar& pol = r.pol;
    const std::array<double, 3> tc{r.t[0] + pol.R[0] * b.c[0] + pol.R[1] * b.c[1] + pol.R[2] * b.c[2],
                                   r.t[1] + pol.R[3] * b.c[0] + pol.R[4] * b.c[1] + pol.R[5] * b.c[2],
                                   r.t[2] + pol.R[6] * b.c[0] + pol.R[7] * b.c[1] + pol.R[8] * b.c[2]};
    b.T_world_export = rigid(pol.R, tc);
    bool still = is_static(p.object);
    for (const int32_t j : p.parts) still = still && is_static(j);
    if (still) b.T_world_static = b.T_world_export;
    st.bodies.push_back(std::move(b));
  }
  convert_cameras(ex, o, static_objects, st);
  convert_lights(ex, o, static_objects, st);
  return st;
}

}  // namespace balla::scene::blender
