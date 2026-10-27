// Mode by provenance, rigs and a fitted body's report (checks.h; blender.md §4.7.1, §4.7.9, §6.1).
#include "scene/motion/checks.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "core/math/so3.h"

namespace balla::scene::motion {

namespace {

bool flag(const Tree& t, const char* key, bool fallback) {
  if (t.kind() != Tree::Kind::kDict) return fallback;
  const Tree* v = t.find(key);
  return v && v->kind() == Tree::Kind::kBool ? v->as_bool() : fallback;
}

std::string text(const Tree& t, const char* key) {
  if (t.kind() != Tree::Kind::kDict) return {};
  const Tree* v = t.find(key);
  return v && v->kind() == Tree::Kind::kString ? v->as_string() : std::string();
}

double angle(const rot::Q& a, const rot::Q& b) { return norm(so3::log(conjugate(a) * b)); }

blender::ReportItem warn(const char* code, double bound, std::string message) {
  return {code, "warn", "", std::move(message), bound, blender::BoundKind::kMeasured};
}

}  // namespace

ModeChoice choose_mode(const blender::ExchangeIn& ex, int32_t object, double framelen) {
  const auto n = static_cast<int32_t>(ex.objects.size());
  if (object < 0 || object >= n) throw std::invalid_argument("choose_mode: object out of range");
  // the chain: the object and everything up its parent, instancer and instanced-object links
  std::vector<int32_t> chain;
  std::vector<uint8_t> seen(static_cast<size_t>(n), 0);
  std::vector<int32_t> todo{object};
  while (!todo.empty()) {
    const int32_t i = todo.back();
    todo.pop_back();
    if (i < 0) continue;
    if (i >= n) throw std::invalid_argument("choose_mode: a link out of range");
    if (seen[static_cast<size_t>(i)]) continue;
    seen[static_cast<size_t>(i)] = 1;
    chain.push_back(i);
    const blender::ObjectIn& ob = ex.objects[static_cast<size_t>(i)];
    for (const int32_t j : {ob.parent, ob.instancer, ob.instance_object}) todo.push_back(j);
  }
  std::sort(chain.begin(), chain.end());
  ModeChoice m;
  std::string keyed_by;  // the first keyed source, for the messages
  for (const int32_t i : chain) {
    const blender::ObjectIn& ob = ex.objects[static_cast<size_t>(i)];
    const bool rb_active = ob.rigid_body.kind() == Tree::Kind::kDict && text(ob.rigid_body, "type") == "ACTIVE" &&
                           flag(ob.rigid_body, "enabled", true);
    const bool kin_animated = rb_active && flag(ob.rigid_body, "kinematic_animated", false);
    if (rb_active && (!flag(ob.rigid_body, "kinematic", false) || kin_animated)) {
      if (m.frame_state_object < 0) {
        m.frame_state_object = i;
        m.kinematic_split = kin_animated;
      }
    }
    bool own_keys = ob.animation >= 0 && !(kin_animated && i == m.frame_state_object);
    for (const Tree& c : ob.constraints.items()) {
      if (flag(c, "mute", false)) continue;
      if (text(c, "type") == "TRANSFORM_CACHE") {
        if (m.frame_state_object < 0) {
          m.frame_state_object = i;
          m.cache = true;
        }
      } else {
        own_keys = true;
      }
    }
    if (ob.parent_type == "BONE") own_keys = true;  // an armature's pose (its bone constraints are not exported)
    if (own_keys && keyed_by.empty()) keyed_by = ob.name_full;
  }
  if (m.frame_state_object >= 0) {
    const std::string& holder = ex.objects[static_cast<size_t>(m.frame_state_object)].name_full;
    if (!keyed_by.empty()) {
      m.code = "PHY-MIXED";
      m.message = (m.cache ? "the transform cache of " : "the rigid body of ") + holder + " and the keys of " +
                  keyed_by + " move one body";
      return m;
    }
    if (!m.cache) {
      const Tree& w = ex.rigidbody_world;
      const Tree* pc = w.kind() == Tree::Kind::kDict ? w.find("point_cache") : nullptr;
      const bool baked = pc && flag(*pc, "is_baked", false) && !flag(*pc, "is_outdated", false);
      if (!(w.kind() == Tree::Kind::kDict && flag(w, "enabled", true) && baked)) {
        m.code = "PHY-UNBAKED";
        m.message = "the rigid body of " + holder + " needs a baked, current cache (bake the rigid-body world)";
        return m;
      }
      if (framelen != 1.0) {
        m.code = "PHY-REMAP";
        m.message = "the rigid body of " + holder + " under frame_map (old != new)";
        return m;
      }
    }
    m.source = Source::kFrameState;
    return m;
  }
  m.source = keyed_by.empty() ? Source::kNone : Source::kKeyed;
  return m;
}

std::optional<blender::ReportItem> motion_source(const ModeChoice& mode, bool samples_vary, blender::Policy policy) {
  if (mode.source != Source::kNone || !samples_vary) return std::nullopt;
  if (policy == blender::Policy::kStrict) {
    return blender::ReportItem{"MOT-SOURCE", "refuse", "",
                               "moves with no animation, constraint, parent or physics in its chain (a frame-change "
                               "handler or an unsupported modifier?)",
                               0.0, blender::BoundKind::kNone};
  }
  return blender::ReportItem{"MOT-SOURCE", "warn", "",
                             "moves with no supported source: fitted with breakpoint detection only", 0.0,
                             blender::BoundKind::kNone};
}

// --- rigs ---

namespace {

struct Rel {
  rot::Q q;
  rot::V p;
  double scale_m;  // the larger distance of the two from the world origin (the float32 scale), at least 1 mm
};
Rel relative(const PoseBatch& b, size_t i, int32_t rig, int32_t cam) {
  const size_t r = b.at(i, rig), c = b.at(i, cam);
  const rot::Q inv = conjugate(b.q[r]);
  return {normalized(inv * b.q[c]), rotate(inv, b.p[c] - b.p[r]), std::max({norm(b.p[r]), norm(b.p[c]), 1e-3})};
}

}  // namespace

RigGroups group_rigs(const PoseBatch& batch, std::span<const int32_t> cameras, std::span<const int32_t> hints) {
  if (!hints.empty() && hints.size() != cameras.size()) throw std::invalid_argument("group_rigs: hints per camera");
  if (batch.size() == 0) throw std::invalid_argument("group_rigs: no samples");
  for (const int32_t c : cameras) {
    if (c < 0 || c >= batch.bodies) throw std::invalid_argument("group_rigs: a camera is not a column");
  }
  RigGroups g;
  g.cameras.assign(cameras.begin(), cameras.end());
  const size_t nc = cameras.size();
  g.rig.resize(nc);
  g.q_rig_cam.resize(nc);
  g.p_rig_cam.resize(nc);
  g.dev_rad.assign(nc, 0.0);
  g.dev_m.assign(nc, 0.0);
  std::vector<size_t> leaders;  // indices into cameras of the groups' first cameras
  for (size_t j = 0; j < nc; ++j) {
    bool placed = false;
    for (const size_t l : leaders) {
      const Rel r0 = relative(batch, 0, cameras[l], cameras[j]);
      double dr = 0.0, dm = 0.0;
      bool constant = true;
      for (size_t i = 0; i < batch.size() && constant; ++i) {
        const Rel r = relative(batch, i, cameras[l], cameras[j]);
        dr = std::max(dr, angle(r0.q, r.q));
        dm = std::max(dm, norm(r.p - r0.p));
        constant = angle(r0.q, r.q) <= kRigTol && norm(r.p - r0.p) <= kRigTol * r.scale_m;
      }
      if (!constant) continue;
      g.rig[j] = cameras[l];
      g.q_rig_cam[j] = r0.q;
      g.p_rig_cam[j] = r0.p;
      g.dev_rad[j] = dr;
      g.dev_m[j] = dm;
      placed = true;
      break;
    }
    if (!placed) {
      leaders.push_back(j);
      g.rig[j] = cameras[j];
      g.q_rig_cam[j] = {1, 0, 0, 0};
      g.p_rig_cam[j] = {0, 0, 0};
    }
  }
  for (size_t a = 0; a < hints.size(); ++a) {
    for (size_t b = a + 1; b < hints.size(); ++b) {
      if (hints[a] >= 0 && hints[a] == hints[b] && g.rig[a] != g.rig[b]) {
        throw MotionError("CAM-RIG", "cameras in columns " + std::to_string(cameras[a]) + " and " +
                                         std::to_string(cameras[b]) + " share a rig but their relative pose varies");
      }
    }
  }
  return g;
}

void check_rigs(const PoseBatch& batch, RigGroups& g) {
  for (size_t j = 0; j < g.cameras.size(); ++j) {
    if (g.rig[j] == g.cameras[j]) continue;
    for (size_t i = 0; i < batch.size(); ++i) {
      const Rel r = relative(batch, i, g.rig[j], g.cameras[j]);
      const double dr = angle(g.q_rig_cam[j], r.q), dm = norm(r.p - g.p_rig_cam[j]);
      g.dev_rad[j] = std::max(g.dev_rad[j], dr);
      g.dev_m[j] = std::max(g.dev_m[j], dm);
      if (!(dr <= kRigTol && dm <= kRigTol * r.scale_m)) {
        throw MotionError("CAM-RIG", "the camera in column " + std::to_string(g.cameras[j]) + " leaves its rig (column " +
                                         std::to_string(g.rig[j]) + ") by " + std::to_string(dr) + " rad and " +
                                         std::to_string(dm) + " m at " + std::to_string(batch.t_ns[i]) + " ns");
      }
    }
  }
}

// --- a fitted body's report ---

std::vector<blender::ReportItem> body_report(const SampleSet& set, const TimeMap& time, double eps_px) {
  std::vector<blender::ReportItem> out;
  if (const std::optional<PosRes> r = pos_res(set, eps_px)) {
    out.push_back(warn("POS-RES", r->bound_px,
                       "Blender's float32 positions floor the fit at " + std::to_string(r->floor_px) +
                           " px; its bound is eps/2 + 2 floor_px = " + std::to_string(r->bound_px) +
                           " px (move the action nearer Blender's origin, or accept the floor)"));
  }
  for (const Breakpoint& b : set.breakpoints) {
    int32_t F = 0;
    float s = 0.0f;
    time.request(b.t_ns, &F, &s);
    const double px = b.dv_pxps * time.resolution_ns(F) * 1e-9;
    if (px > eps_px / 4.0) {
      out.push_back(warn("TIME-RES", px,
                         "Blender's float32 time near frame " + std::to_string(F) + " moves the breakpoint at " +
                             std::to_string(b.t_ns) + " ns by up to " + std::to_string(px) + " px"));
    }
  }
  double reach = set.fitted ? set.motion.max_reach_m : 0.0;
  for (const Track* t : {&set.fit, &set.held_out}) {
    for (const rot::V& p : t->p) reach = std::max(reach, norm(p));
  }
  if (reach > kFarM) {
    out.push_back(warn("MOT-FAR", reach,
                       "a sample lies " + std::to_string(reach) + " m from Blender's origin, where float32 positions "
                       "round coarsely"));
  }
  if (set.light_near) {
    out.push_back(warn("LGT-MOVE-NEAR", 0.05, "a moving light comes within 5 cm of another body: its bound uses "
                                              "the 5 cm floor"));
  }
  if (set.alias_warnings > 0) {
    out.push_back(warn("MOT-ALIAS", set.alias_max_rad,
                       std::to_string(set.alias_warnings) + " sample intervals turn by 0.6 pi to 0.9 pi (largest " +
                           std::to_string(set.alias_max_rad / std::numbers::pi) + " pi)"));
  }
  return out;
}

}  // namespace balla::scene::motion
