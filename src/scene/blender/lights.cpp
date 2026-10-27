// Blender lights, the world and the light groups in C1 terms (docs/design/phase3/blender.md §3.12, §4.1, §4.4, §6.1).
//
// Provenance (PHASE3.md §0 rule 4): re-derived from the Blender manual (Light properties: power, radius, spot size and
// blend, sun strength and angle, exposure, normalize, temperature, light groups, light linking; World: Background
// shader) and from headless oracles, not from Blender's sources: CV E1/E2 (point I = P / 4 pi, sun irradiance =
// strength, a spot's plateau equals the point light: no power concentration), CV E10 (exposure 2^e, temperature_color
// multiplies, the node Emission strength multiplies, visible_diffuse off removes the light from diffuse surfaces),
// CV E12 (unoccluded world ambient over-lights by 1.82x near contact) and the BL-3a probe `ballasim_out/
// phase3_research/blender/bl3a/probes/m3_lights_cameras.py` (4.2.1 and 4.5.3; check analysis/a1_check_m3.py): the node
// Emission colour multiplies per channel and with its strength; a point light's radius does not scale with its
// object (r = 0.1, scale 3: ratio 1); normalize False multiplies a point light by 4 pi r^2 (0.125664) and a sun by
// pi sin^2(angle / 2) (0.023864), both in 4.5 only (4.2 normalizes; it exports no `normalize`); a uniform object scale
// leaves a spot cone unchanged while (1, 1, 2) halves tan(cone) and (2, 2, 1) or (2, 1, 1) widen it otherwise, so a
// spot whose object scale is not uniform is refused. Probe m3b (`.../bl3a/probes/m3b_world.py`, 4.2.1 and 4.5.3
// identical; analysis/a3_check_m3b.py): a world without nodes renders World.color as a Background of strength 1 does
// (default 0.05 grey); cycles_visibility.camera off removes the background seen by the camera and diffuse off the
// world's lighting of diffuse surfaces, each alone; keys on a light's Emission or a world's Background live on the
// node tree's own animation data (`nodes["Emission"].inputs[1].default_value`); light energy and Emission strength
// may be negative.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <numbers>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "scene/blender/convert.h"
#include "scene/blender/values.h"

namespace balla::scene::blender {

namespace {

using detail::at;
using detail::boolean;
using detail::number;
using detail::text;

constexpr double kPi = std::numbers::pi;
constexpr double kDepthFloorM = 0.05;   // SCENE.md's 5 cm depth floor
constexpr double kFloatMax = std::numeric_limits<float>::max();   // C1 lights are float32

std::string fmt(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.4g", v);
  return buf;
}

bool is(const NodeIn& n, const char* idname, const char* rna) { return n.type == idname || n.type == rna; }

// A light's or the world's node graph (sockets by identifier, links through reroutes).
class Graph {
 public:
  explicit Graph(const GraphIn& g) : g_(g) {}
  const NodeIn& node(size_t i) const { return g_.nodes[i]; }
  size_t size() const { return g_.nodes.size(); }
  bool muted(size_t i) const {
    const Tree* m = at(g_.nodes[i].props, "mute");
    return m != nullptr && m->kind() == Tree::Kind::kBool && m->as_bool();
  }
  std::string prop(size_t i, const char* key, const std::string& fallback) const {
    const Tree* v = at(g_.nodes[i].props, key);
    return v != nullptr && v->kind() == Tree::Kind::kString ? v->as_string() : fallback;
  }
  bool flag(size_t i, const char* key) const {
    const Tree* v = at(g_.nodes[i].props, key);
    return v != nullptr && v->kind() == Tree::Kind::kBool && v->as_bool();
  }
  const SocketIn* input(size_t i, const std::string& id) const {
    for (const SocketIn& s : g_.nodes[i].inputs) {
      if (s.identifier == id) return &s;
    }
    return nullptr;
  }
  // The node feeding node i's input `id` (reroutes followed); -1 when unlinked.
  int source(size_t i, const std::string& id) const {
    std::string node = g_.nodes[i].name, sock = id;
    for (size_t hop = 0; hop <= g_.nodes.size(); ++hop) {
      const LinkIn* l = nullptr;
      for (const LinkIn& k : g_.links) {
        if (k.to_node == node && k.to_socket == sock) l = &k;
      }
      if (l == nullptr) return -1;
      const int j = index(l->from_node);
      if (j < 0) throw std::invalid_argument("BRG-PROTO: a link from an unknown node " + l->from_node);
      if (!is(g_.nodes[static_cast<size_t>(j)], "NodeReroute", "REROUTE")) return j;
      node = l->from_node;
      sock = "Input";
    }
    throw std::invalid_argument("BRG-PROTO: a reroute cycle");
  }
  // The output node Cycles uses: target CYCLES before ALL, the active one or the only one (probe m2's rule for
  // Material Outputs); -1 when there is none, -2 when several are inactive.
  int output(const char* idname, const char* rna) const {
    std::vector<size_t> outs;
    for (const char* tier : {"CYCLES", "ALL"}) {
      for (size_t i = 0; i < size(); ++i) {
        if (is(node(i), idname, rna) && prop(i, "target", "ALL") == tier) outs.push_back(i);
      }
      if (!outs.empty()) break;
    }
    for (const size_t i : outs) {
      if (flag(i, "is_active_output")) return static_cast<int>(i);
    }
    return outs.empty() ? -1 : outs.size() == 1 ? static_cast<int>(outs.front()) : -2;
  }
  // A constant input's components (1 or 3 used); throws `code` when linked.
  std::array<double, 3> constant(size_t i, const char* id, const char* code, const std::string& what) const {
    const SocketIn* s = input(i, id);
    if (s == nullptr) return {1.0, 1.0, 1.0};
    if (s->linked) throw std::invalid_argument(std::string(code) + ": " + what + " is linked (not constant)");
    if (s->n >= 3) return {s->value[0], s->value[1], s->value[2]};
    return {s->value[0], s->value[0], s->value[0]};
  }

 private:
  int index(const std::string& name) const {
    for (size_t i = 0; i < g_.nodes.size(); ++i) {
      if (g_.nodes[i].name == name) return static_cast<int>(i);
    }
    return -1;
  }
  const GraphIn& g_;
};

// The colour and strength a light's node tree multiplies it by (probe m3: Emission colour x strength; CV E10), and
// the names of the nodes it reads (the Light Output and the Emission) in `used`.
std::array<double, 3> light_nodes(const GraphIn& g, std::vector<std::string>* used) {
  if (!g.present) return {1.0, 1.0, 1.0};
  const Graph gr(g);
  const int out = gr.output("ShaderNodeOutputLight", "OUTPUT_LIGHT");
  if (out == -2) throw std::invalid_argument("LGT-NODE: several Light Outputs and none active");
  const int src = out < 0 ? -1 : gr.source(static_cast<size_t>(out), "Surface");
  if (src < 0) throw std::invalid_argument("LGT-NODE: nothing is connected to the Light Output's Surface");
  const auto e = static_cast<size_t>(src);
  used->push_back(gr.node(static_cast<size_t>(out)).name);
  used->push_back(gr.node(e).name);
  if (!is(gr.node(e), "ShaderNodeEmission", "EMISSION") || gr.muted(e)) {
    throw std::invalid_argument("LGT-NODE: the light's node tree is not one constant Emission (" + gr.node(e).type +
                                ")");
  }
  const auto c = gr.constant(e, "Color", "LGT-NODE", "the Emission colour");
  const double s = gr.constant(e, "Strength", "LGT-NODE", "the Emission strength")[0];
  return {c[0] * s, c[1] * s, c[2] * s};
}

bool light_path(const std::string& p) {
  static const char* const kPaths[] = {"energy", "color", "shadow_soft_size", "spot_size", "spot_blend", "angle",
                                       "exposure", "normalize", "use_temperature", "temperature", "use_shadow",
                                       "shape", "size", "size_y", "type", "use_nodes", "NLA"};
  for (const char* k : kPaths) {
    if (p == k) return true;
  }
  return false;
}

// The keys of a node tree (its own animations[] record, probe m3b) on the nodes the converter reads: their paths,
// comma-separated ("" when none). An NLA strip on the tree counts as animating every node.
std::string animated_nodes(const ExchangeIn& ex, int32_t animation, const std::vector<std::string>& used) {
  std::string out;
  for (const std::string& path : detail::animated_paths(ex, animation)) {
    if (path == "NLA" || std::find(used.begin(), used.end(), detail::path_node(path)) != used.end()) {
      out += (out.empty() ? "" : ", ") + path;
    }
  }
  return out;
}

Tree override(const char* kind, const std::string& object, Tree set) {
  Tree m = Tree::dict();
  m["kind"] = Tree::string(kind);
  if (!object.empty()) m["object"] = Tree::string(object);
  m["set"] = std::move(set);
  return m;
}

}  // namespace

LightMotion light_motion(const ExchangeIn& ex, int32_t object, const std::vector<bool>& is_static) {
  if (is_static[static_cast<size_t>(object)]) return LightMotion::kWorld;
  const ObjectIn& ob = ex.objects[static_cast<size_t>(object)];
  const bool own = ob.parent < 0 || ob.animation >= 0 || !ob.constraints.items().empty() ||
                   ob.rigid_body.kind() != Tree::Kind::kNull || !(ob.parent_type.empty() || ob.parent_type == "OBJECT");
  return own ? LightMotion::kOwn : LightMotion::kParent;
}

void convert_lights(const ExchangeIn& ex, const ConvertOptions& o, const std::vector<bool>& is_static, Statics& st) {
  const bool strict = o.policy == Policy::kStrict;
  const double k_lm = o.lm_per_w.value_or(1.0), s = ex.scale_length();
  std::map<int32_t, int32_t> body_of;   // objects[] index -> bodies index
  for (size_t b = 0; b < st.bodies.size(); ++b) body_of[st.bodies[b].object] = static_cast<int32_t>(b);
  // World AABBs of the bodies with triangles at the export frame (LGT-RADIUS's d_min).
  std::vector<std::array<double, 6>> boxes;
  for (const BodyStatic& b : st.bodies) {
    const std::vector<float>& x = b.mesh.positions;
    if (x.empty()) continue;
    std::array<double, 6> box{INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY};
    const geo::M4& T = b.T_world_export;
    for (size_t i = 0; i < x.size(); i += 3) {
      for (size_t r = 0; r < 3; ++r) {
        const double w = T[4 * r] * double{x[i]} + T[4 * r + 1] * double{x[i + 1]} + T[4 * r + 2] * double{x[i + 2]} +
                         T[4 * r + 3];
        box[r] = std::min(box[r], w);
        box[r + 3] = std::max(box[r + 3], w);
      }
    }
    boxes.push_back(box);
  }
  std::vector<std::string> groups{"default"};
  const auto group_of = [&groups](const std::string& name) {
    const std::string n = name.empty() ? "default" : name;
    const auto it = std::find(groups.begin(), groups.end(), n);
    if (it != groups.end()) return static_cast<int32_t>(it - groups.begin());
    groups.push_back(n);
    return static_cast<int32_t>(groups.size() - 1);
  };

  for (size_t q = 0; q < ex.lights.size(); ++q) {
    const LightIn& l = ex.lights[q];
    const ObjectIn& ob = ex.objects[static_cast<size_t>(l.object)];
    const std::string where = "lights[" + std::to_string(q) + "]", name = ob.name_full;
    if (ob.type != "LIGHT") throw std::invalid_argument("BRG-PROTO: " + where + ".object is not a light object");
    const auto item = [&](const char* code, const std::string& sev, const std::string& msg, double bound = 0.0,
                          BoundKind bk = BoundKind::kNone) { st.report.push_back({code, sev, name, msg, bound, bk}); };
    if (boolean(ob.visibility, "hide_render_animated", false, "objects[].visibility")) {
      item("VIS-ANIM", "refuse", "animated render visibility of a light (BL-12)");
      continue;
    }
    if (!boolean(ob.visibility, "render", true, "objects[].visibility")) continue;   // hidden from render: excluded
    const Tree& p = l.props;
    if (l.type == "AREA") {
      item("LGT-AREA", "refuse", "area lights wait for P2-F's area lights (§4.4)");
      continue;
    }
    if (l.type != "POINT" && l.type != "SPOT" && l.type != "SUN") {
      throw std::invalid_argument("BRG-PROTO: " + where + ".type " + l.type + " is not POINT, SPOT, SUN or AREA");
    }
    bool skip = false;
    std::string anim;
    for (const std::string& path : detail::animated_paths(ex, l.animation)) {
      if (light_path(path)) anim += (anim.empty() ? "" : ", ") + path;
    }
    if (!anim.empty()) {
      item("LGT-ANIM", "refuse", "animated light properties (" + anim + ")");
      skip = true;
    }
    if (const Tree& ll = l.light_linking; ll.kind() == Tree::Kind::kDict) {
      const bool linked = !text(ll, "receiver_collection", "", where).empty() ||
                          !text(ll, "blocker_collection", "", where).empty();
      if (linked) {
        item("LGT-LINK", strict ? "refuse" : "approx", "light or shadow linking (no bound; ignored under lenient)");
        skip = skip || strict;
      }
    }
    if (!boolean(ob.visibility, "diffuse", true, "objects[].visibility")) {
      item("LGT-LINK", strict ? "refuse" : "approx",
           "visible_diffuse is off: the light lights no diffuse surface (dropped under lenient)");
      continue;
    }
    std::array<double, 3> tint{};
    std::vector<std::string> used;
    try {
      tint = light_nodes(l.graph, &used);
    } catch (const std::invalid_argument& e) {
      const std::string msg = e.what();
      if (msg.rfind("BRG-PROTO", 0) == 0) throw;
      item("LGT-NODE", "refuse", msg.substr(msg.find(": ") + 2));
      continue;
    }
    if (const std::string keys = animated_nodes(ex, l.graph_animation, used); !keys.empty()) {
      item("LGT-ANIM", "refuse", "an animated light node tree (" + keys + ")");
      skip = true;
    }

    // Pose at the export frame: position (metres) and the propagation direction -z of the object.
    std::array<double, 9> A{};
    for (size_t r = 0; r < 3; ++r) {
      for (size_t c = 0; c < 3; ++c) A[3 * r + c] = ob.matrix_world[4 * r + c];
    }
    const std::array<double, 3> pw{s * ob.matrix_world[3], s * ob.matrix_world[7], s * ob.matrix_world[11]};
    const double dn = std::hypot(A[2], A[5], A[8]);
    const bool aimed = dn > 0.0 && std::isfinite(dn);   // a point light needs no direction (its scale is ignored)
    if (!aimed && l.type != "POINT") {
      item("LGT-CONE", "refuse", "a singular light transform (zero scale): the light has no direction");
      continue;
    }
    const std::array<double, 3> dw = aimed ? std::array<double, 3>{-A[2] / dn, -A[5] / dn, -A[8] / dn}
                                           : std::array<double, 3>{0.0, 0.0, -1.0};

    render::Light L;
    L.casts_shadows = boolean(p, "use_shadow", true, where);
    const std::array<double, 3> colour = detail::rgb(p, "color", where);
    std::array<double, 3> c{colour[0] * tint[0], colour[1] * tint[1], colour[2] * tint[2]};
    if (boolean(p, "use_temperature", false, where)) {   // 4.5 only (absent in 4.2 exchanges)
      const std::array<double, 3> tc = detail::rgb(p, "temperature_color", where);
      for (size_t r = 0; r < 3; ++r) c[r] *= tc[r];
    }
    const double energy = number(p, "energy", 10.0, where), gain = std::exp2(number(p, "exposure", 0.0, where));
    const bool normalize = boolean(p, "normalize", true, where);   // 4.2 always normalizes and exports none
    double I = 0.0;   // signed until the sign check below (Blender admits negative energy and strength, probe m3b)
    if (l.type == "SUN") {
      const double angle = number(p, "angle", 0.0, where);
      const double area = angle > 0.0 ? kPi * std::pow(std::sin(angle / 2.0), 2) : 1.0;
      L.kind = render::BundleLightKind::kDirectional;
      I = k_lm * energy * gain * (normalize ? 1.0 : area);
      if (angle > 0.0) {
        const double pen = 2.0 * std::tan(angle / 2.0);
        item("LGT-RADIUS", "warn",
             "sun angle " + fmt(angle * 180.0 / kPi) + " deg: soft shadows, penumbra 2 tan(angle / 2) = " + fmt(pen) +
                 " m per metre between occluder and receiver (C1 suns are hard; matched: angle 0)",
             pen, BoundKind::kMeasured);
        Tree set = Tree::dict();
        set["angle"] = Tree::real(0.0);
        st.matched.push_back(override("light", name, std::move(set)));
      }
    } else {
      const double r = number(p, "shadow_soft_size", 0.25, where);   // BU: Cycles' area reads no object scale
      const double area = r > 0.0 ? 4.0 * kPi * r * r : 4.0;
      L.kind = l.type == "SPOT" ? render::BundleLightKind::kSpot : render::BundleLightKind::kPoint;
      I = k_lm * energy * gain * (normalize ? 1.0 : area) * s * s / (4.0 * kPi);
      if (l.type == "SPOT") {
        Polar pol{};
        try {
          pol = polar(A);
        } catch (const std::invalid_argument&) {
          item("LGT-CONE", "refuse", "a singular spot transform (a zero scale): Blender reshapes its cone (probe m3)");
          continue;
        }
        const double u = (pol.S[0] + pol.S[4] + pol.S[8]) / 3.0;
        double off = 0.0, all = 0.0;
        for (size_t i = 0; i < 9; ++i) {
          off += std::pow(pol.S[i] - (i % 4 == 0 ? u : 0.0), 2);
          all += pol.S[i] * pol.S[i];
        }
        const double blend = number(p, "spot_blend", 0.15, where);
        const double outer = number(p, "spot_size", kPi / 4.0, where) / 2.0;
        if (!(u > 0.0) || std::sqrt(off) > 1e-6 * std::sqrt(all)) {
          item("LGT-CONE", "refuse", "the spot's object scale is not uniform: Blender reshapes its cone (probe m3)");
          skip = true;
        } else if (blend < 1e-3) {
          item("LGT-HARDCONE", "refuse", "spot blend " + fmt(blend) + " < 1e-3: a hard cone edge (C1 needs inner < "
                                        "outer)");
          skip = true;
        } else {
          const double co = std::cos(outer), ci = co + blend * (1.0 - co);
          L.spot_outer_rad = static_cast<float>(outer);
          L.spot_inner_rad = static_cast<float>(std::acos(std::min(1.0, ci)));
          item("LGT-SPOT", "approx", "Blender's smoothstep cone profile as C1's smootherstep (<= 0.054 apart, CV "
                                     "E2; BL-14)", 0.054, BoundKind::kMeasured);
        }
      }
      if (r > 0.0 && !skip) {
        double d = INFINITY;
        for (const auto& box : boxes) {
          double q2 = 0.0;
          for (size_t a = 0; a < 3; ++a) q2 += std::pow(std::max({box[a] - pw[a], 0.0, pw[a] - box[a + 3]}), 2);
          d = std::min(d, std::sqrt(q2));
        }
        d = std::max(std::isfinite(d) ? d : 1.0, kDepthFloorM);
        const double rm = r * s, deficit = std::pow(rm / d, 2);
        item("LGT-RADIUS", "warn",
             "radius " + fmt(rm) + " m: energy deficit <= (r / d_min)^2 = " + fmt(deficit) + " at d_min " + fmt(d) +
                 " m (nearest body box, floor 5 cm); penumbra 2 r (D - d) / d = " + fmt(2.0 * rm) +
                 " m x (D - d) / d (C1 point lights are hard; matched: radius 0)",
             deficit, BoundKind::kMeasured);
        Tree set = Tree::dict();
        set["shadow_soft_size"] = Tree::real(0.0);
        st.matched.push_back(override("light", name, std::move(set)));
      }
    }
    if (skip) continue;
    // The radiance is colour x power per channel: a negative power with a negative colour is a positive light, any
    // other negative channel a negative light (LGT-NEG; Blender admits negative energy and node strengths, probe m3b).
    if (!std::isfinite(I) || !std::all_of(c.begin(), c.end(), [](double x) { return std::isfinite(x); })) {
      throw std::invalid_argument("BRG-PROTO: " + where + ": a non-finite power or colour");
    }
    const double sign = I < 0.0 ? -1.0 : 1.0;
    bool negative = false;
    for (double& x : c) {
      x = I == 0.0 ? std::abs(x) : sign * x;
      negative = negative || x < 0.0;
    }
    if (negative) {
      item("LGT-NEG", "refuse", "a negative light (energy, colour or node Emission): negative lights are not modelled");
      continue;
    }
    if (const double cmax = std::max({c[0], c[1], c[2]});
        !(std::abs(I) <= kFloatMax && cmax <= kFloatMax && std::abs(I) * cmax <= kFloatMax)) {
      item("LGT-LIMIT", "refuse", "a light power beyond C1's float32 range");
      continue;
    }
    L.intensity = static_cast<float>(std::abs(I));
    for (size_t r = 0; r < 3; ++r) L.color[r] = static_cast<float>(c[r]);
    L.group = group_of(ob.lightgroup);
    L.position = pw;
    L.direction = dw;
    const LightMotion motion = light_motion(ex, l.object, is_static);
    if (motion != LightMotion::kWorld) {   // its carrier: the parent's body or its own meshless one (§3.7)
      const auto it = body_of.find(motion == LightMotion::kParent ? ob.parent : l.object);
      if (it == body_of.end()) {
        if (st.refused()) continue;
        throw std::logic_error("convert_lights: no carrier body for " + name);
      }
      const geo::M4& T = st.bodies[static_cast<size_t>(it->second)].T_world_export;   // [R | t], world <- body
      for (size_t r = 0; r < 3; ++r) {
        L.position[r] = T[r] * (pw[0] - T[3]) + T[4 + r] * (pw[1] - T[7]) + T[8 + r] * (pw[2] - T[11]);
        L.direction[r] = T[r] * dw[0] + T[4 + r] * dw[1] + T[8 + r] * dw[2];
      }
      L.body = it->second;
    }
    st.lights.push_back({l.object, name, L});
  }

  // The world (§4.4): a constant Background, or World.color when the world uses no nodes (Cycles renders it as a
  // Background of strength 1, probe m3b), is C1's background B = k strength colour in World.lightgroup while camera
  // rays see the world, and with world_as_ambient also its ambient A = B while it lights diffuse surfaces (probe m3b:
  // cycles_visibility.camera off removes only the background, diffuse off only the lighting).
  if (ex.world.present) {
    const std::string name = "World", where = "world";
    const auto item = [&](const char* code, const std::string& sev, const std::string& msg, double bound = 0.0,
                          BoundKind bk = BoundKind::kNone) { st.report.push_back({code, sev, name, msg, bound, bk}); };
    const GraphIn& wg = ex.world.graph;
    std::array<double, 3> b{0.0, 0.0, 0.0};
    std::vector<std::string> used;
    bool ok = true;
    try {
      if (!wg.present) {
        Tree rec = Tree::dict();
        rec["color"] = ex.world.color;
        const auto colour = detail::rgb(rec, "color", where, {0.05, 0.05, 0.05});   // World.color's default
        for (size_t r = 0; r < 3; ++r) b[r] = k_lm * colour[r];
      } else {
        const Graph g(wg);
        const int out = g.output("ShaderNodeOutputWorld", "OUTPUT_WORLD");
        if (out == -2) throw std::invalid_argument("WLD-ENV: several World Outputs and none active");
        const int src = out < 0 ? -1 : g.source(static_cast<size_t>(out), "Surface");
        if (out >= 0) used.push_back(g.node(static_cast<size_t>(out)).name);
        if (src >= 0) {
          const auto e = static_cast<size_t>(src);
          if (!is(g.node(e), "ShaderNodeBackground", "BACKGROUND") || g.muted(e)) {
            throw std::invalid_argument("WLD-ENV: the world surface is not one constant Background (" +
                                        g.node(e).type + ")");
          }
          used.push_back(g.node(e).name);
          if (const int t = g.source(e, "Color"); t >= 0) {
            const NodeIn& tn = g.node(static_cast<size_t>(t));
            const bool env =
                is(tn, "ShaderNodeTexEnvironment", "TEX_ENVIRONMENT") || is(tn, "ShaderNodeTexSky", "TEX_SKY");
            throw std::invalid_argument(std::string("WLD-ENV: ") + (env ? "an environment or sky texture" :
                                                                          "a world colour that is not constant") +
                                        " (" + tn.type + ")");
          }
          const auto colour = g.constant(e, "Color", "WLD-ENV", "the Background colour");
          const double strength = g.constant(e, "Strength", "WLD-ENV", "the Background strength")[0];
          for (size_t r = 0; r < 3; ++r) b[r] = k_lm * strength * colour[r];
        }
      }
    } catch (const std::invalid_argument& e) {
      const std::string msg = e.what();
      if (msg.rfind("BRG-PROTO", 0) == 0) throw;
      item("WLD-ENV", "refuse", msg.substr(msg.find(": ") + 2) + " (the lenient mean waits for a later increment)");
      ok = false;
    }
    // WLD-ANIM: keys on what the conversion reads, the world's own (its colour without nodes, use_nodes, its ray
    // visibility) and its node tree's (the World Output and the Background, probe m3b).
    std::string keys;
    for (const std::string& path : detail::animated_paths(ex, ex.world.animation)) {
      if (path == "NLA" || path == "use_nodes" || path.rfind("cycles_visibility", 0) == 0 ||
          (!wg.present && path == "color")) {
        keys += (keys.empty() ? "" : ", ") + path;
      }
    }
    if (wg.present) {
      const std::string tree_keys = animated_nodes(ex, ex.world.graph_animation, used);
      if (!tree_keys.empty()) keys += (keys.empty() ? "" : ", ") + tree_keys;
    }
    if (!keys.empty()) {
      item("WLD-ANIM", "refuse", "an animated world (" + keys + ")");
      ok = false;
    }
    if (!std::all_of(b.begin(), b.end(), [](double x) { return std::isfinite(x); })) {
      throw std::invalid_argument("BRG-PROTO: world: a non-finite Background");
    }
    if (ok && std::any_of(b.begin(), b.end(), [](double x) { return x < 0.0; })) {
      item("LGT-NEG", "refuse", "a negative world colour or Background strength: negative lights are not modelled");
      ok = false;
    }
    if (ok && std::any_of(b.begin(), b.end(), [](double x) { return x > kFloatMax; })) {
      item("LGT-LIMIT", "refuse", "a world radiance beyond C1's float32 range");
      ok = false;
    }
    const Tree& vis = ex.world.cycles_visibility;
    const bool camera = boolean(vis, "camera", true, "world.cycles_visibility");
    const bool ambient_on = o.world_as_ambient && boolean(vis, "diffuse", true, "world.cycles_visibility");
    const double m = std::max({b[0], b[1], b[2]});
    if (ok && m > 0.0) {
      const int32_t grp = group_of(ex.world.lightgroup);
      for (const bool ambient : {false, true}) {
        if (ambient ? !ambient_on : !camera) continue;
        render::Light L;
        L.kind = ambient ? render::BundleLightKind::kAmbient : render::BundleLightKind::kBackground;
        L.group = grp;
        L.intensity = static_cast<float>(m);
        for (size_t r = 0; r < 3; ++r) L.color[r] = static_cast<float>(b[r] / m);
        L.casts_shadows = false;
        st.lights.push_back({-1, ambient ? "World (ambient)" : "World", L});
      }
    }
    Tree set = Tree::dict();
    for (const char* ray : {"diffuse", "glossy", "transmission", "scatter"}) {
      if (!(o.world_as_ambient && std::string(ray) == "diffuse")) {
        set[std::string("cycles_visibility.") + ray] = Tree::boolean(false);
      }
    }
    Tree m_world = override("world", "", std::move(set));
    if (ambient_on && ok && m > 0.0) {
      m_world["residual"] = Tree::string("C1's ambient is unoccluded: >= 1.82x over-lit near contact (CV E12)");
      item("WLD-AMBIENT", "warn", "world_as_ambient: the world also lights surfaces as unoccluded ambient, >= 1.82x "
                                  "over-lit near contact (CV E12: 6.2 cm from a cube face)",
           1.82, BoundKind::kMeasured);
    }
    st.matched.push_back(std::move(m_world));
  }

  // Light groups (§3.2, §4.4): matched by name, an empty name is `default`; waveforms from light_groups[], constant
  // for a used group it does not list (info). C1's group 0 is the constant `default`.
  st.groups.clear();
  for (const std::string& g : groups) {
    Tree wave;
    for (const Tree& rec : ex.light_groups.items()) {
      if (text(rec, "name", "", "light_groups[]") == g) {
        const Tree* w = at(rec, "waveform");
        wave = w == nullptr ? Tree() : *w;
      }
    }
    const bool constant = wave.kind() == Tree::Kind::kDict && text(wave, "type", "", "light_groups[]") == "constant";
    if (wave.kind() == Tree::Kind::kNull) {
      if (g != "default") st.report.push_back({"LGT-GROUP", "info", g, "not in light_groups: constant", 0.0,
                                               BoundKind::kNone});
      wave = Tree::dict();
      wave["type"] = Tree::string("constant");
    } else if (g == "default" && !constant) {
      st.report.push_back({"LGT-GROUP", "refuse", g, "C1's default light group is constant: give the flickering "
                                                     "lights a named light group", 0.0, BoundKind::kNone});
    }
    st.groups.push_back({g, std::move(wave)});
  }
  size_t shadows = 0;
  for (const LightStatic& ls : st.lights) shadows += render::positional(ls.light.kind) && ls.light.casts_shadows;
  if (st.groups.size() > 8 || shadows > 32) {
    st.report.push_back({"LGT-LIMIT", "refuse", "",
                         std::to_string(st.groups.size()) + " light groups (C1: <= 8) and " + std::to_string(shadows) +
                             " shadow-casting lights (C1: <= 32)", 0.0, BoundKind::kNone});
  }

  // The render overrides of MATCHED mode (§3.12): C1 shades direct light plus ambient; events are instantaneous.
  Tree set = Tree::dict();
  set["cycles.max_bounces"] = Tree::integer(0);
  set["cycles.use_denoising"] = Tree::boolean(false);
  set["render.use_motion_blur"] = Tree::boolean(false);
  set["cycles.rolling_shutter_type"] = Tree::string("NONE");
  set["view_settings.view_transform"] = Tree::string("Standard");
  set["view_settings.look"] = Tree::string("None");
  set["view_settings.exposure"] = Tree::real(0.0);
  set["view_settings.gamma"] = Tree::real(1.0);
  Tree render = override("render", "", std::move(set));
  render["residual"] = Tree::string("the pixel filter (Blackman-Harris 1.5 px by default) and textures' lookups "
                                    "(Cycles Linear, no mip-maps) are the frames area's");
  st.matched.push_back(std::move(render));
}

}  // namespace balla::scene::blender
