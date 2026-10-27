// The Blender motion session (session.h; blender.md §3.5, §3.6, §3.8, §4.6, §4.7, §5.1).
#include "scene/motion/session.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numbers>
#include <set>
#include <stdexcept>
#include <thread>

#include "core/time/time.h"
#include "render/thread_pool.h"
#include "scene/blender/assemble.h"
#include "scene/blender/values.h"
#include "scene/geometry.h"
#include "scene/motion/error_model.h"

namespace balla::scene::motion {

namespace {

using Clock = std::chrono::steady_clock;
namespace bl = ::balla::scene::blender;
namespace d = ::balla::scene::blender::detail;

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

bl::ReportItem item(const std::string& code, const std::string& severity, const std::string& object,
                    const std::string& message, double bound = 0.0, bl::BoundKind kind = bl::BoundKind::kNone) {
  return {code, severity, object, message, bound, kind};
}

[[noreturn]] void proto(const std::string& what) { throw std::invalid_argument("BRG-PROTO: " + what); }

// A MotionError whose code the session turns into a refusal (everything but the protocol failures, which propagate).
bool is_refusal(const MotionError& e) { return e.code.rfind("BRG-", 0) != 0; }

std::string without_code(const MotionError& e) {
  const std::string w = e.what();
  return w.size() > e.code.size() + 2 ? w.substr(e.code.size() + 2) : w;
}

int64_t int_at(const Tree& t, const char* key, int64_t fallback) {
  const Tree* v = d::at(t, key);
  if (v == nullptr) return fallback;
  if (v->kind() != Tree::Kind::kInt) proto(std::string(key) + " is not an integer");
  return v->as_int();
}

Tree reals(std::span<const double> v) { return Tree::reals(std::vector<double>(v.begin(), v.end())); }

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// TimeSettings and KeyScene from the exchange

TimeSettings time_settings(const bl::ExchangeIn& ex, bool fps_base_snap) {
  const bl::StaticTimeline t = bl::static_timeline(ex);   // BRG-PROTO for a malformed record
  TimeSettings s;
  s.fps = t.fps;
  s.fps_base_bits = t.fps_base_bits;
  s.frame_map_old = static_cast<int32_t>(int_at(ex.time, "frame_map_old", 100));
  s.frame_map_new = static_cast<int32_t>(int_at(ex.time, "frame_map_new", 100));
  s.frame_start = t.frame_start;
  s.frame_end = t.frame_end;
  s.motion_blur = d::boolean(ex.render, "use_motion_blur", false, "render");
  s.motion_blur_shutter = static_cast<float>(d::number(ex.render, "motion_blur_shutter", 0.5, "render"));
  s.motion_blur_position = d::text(ex.render, "motion_blur_position", "CENTER", "render");
  s.fps_base_snap = fps_base_snap;
  return s;
}

KeyScene key_scene(const bl::ExchangeIn& ex) {
  KeyScene ks;
  const auto n = static_cast<int32_t>(ex.objects.size());
  std::map<std::string, int32_t> by_name;   // a constraint names its target by name_full (plain objects)
  for (int32_t i = 0; i < n; ++i) {
    if (!ex.objects[static_cast<size_t>(i)].is_instance) by_name.emplace(ex.objects[static_cast<size_t>(i)].name_full, i);
  }
  const auto count = static_cast<int32_t>(ex.animations.items().size());
  ks.objects.resize(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const bl::ObjectIn& ob = ex.objects[static_cast<size_t>(i)];
    KeyObject& ko = ks.objects[static_cast<size_t>(i)];
    ko.parent = ob.parent;
    if (ob.animation >= 0) ko.animations.push_back(ob.animation);
    if (ob.data_animation >= 0) ko.data_animations.push_back(ob.data_animation);
    for (const Tree& c : ob.constraints.items()) {
      if (d::boolean(c, "mute", false, "constraints[]")) continue;
      const Tree* target = d::at(c, "target");
      if (target == nullptr || target->kind() != Tree::Kind::kString) continue;
      if (const auto it = by_name.find(target->as_string()); it != by_name.end()) ko.constraint_targets.push_back(it->second);
    }
    for (const int32_t a : {ob.animation, ob.data_animation}) {
      if (a >= count) proto("objects[" + std::to_string(i) + "]: animation out of range");
    }
  }
  for (size_t a = 0; a < ex.animations.items().size(); ++a) {
    const Tree& rec = ex.animations.items()[a];
    const std::string where = "animations[" + std::to_string(a) + "]";
    KeyAnimation ka;
    ka.action = d::text(rec, "action", "", where);
    ka.slot = d::text(rec, "slot", "", where);
    if (const Tree* fcs = d::at(rec, "fcurves"); fcs != nullptr) {
      for (const Tree& f : fcs->items()) {
        KeyCurve kc;
        kc.data_path = d::text(f, "data_path", "", where + ".fcurves");
        kc.index = static_cast<int32_t>(int_at(f, "index", 0));
        kc.extrapolation = static_cast<int32_t>(int_at(f, "extrapolation", 0));
        if (const Tree* name = d::at(f, "keyframes"); name != nullptr && name->kind() == Tree::Kind::kString) {
          const auto it = ex.anim_keys.find(name->as_string());
          if (it == ex.anim_keys.end()) proto(where + ": no key table " + name->as_string());
          kc.keys = it->second;
        }
        if (const Tree* mods = d::at(f, "modifiers"); mods != nullptr) {
          for (const Tree& m : mods->items()) {
            KeyModifier km;
            km.type = d::text(m, "type", "", where + ".modifiers");
            km.mute = d::boolean(m, "mute", false, where + ".modifiers");
            km.use_restricted_range = d::boolean(m, "use_restricted_range", false, where + ".modifiers");
            km.frame_start = d::number(m, "frame_start", 0.0, where + ".modifiers");
            km.frame_end = d::number(m, "frame_end", 0.0, where + ".modifiers");
            km.blend_in = d::number(m, "blend_in", 0.0, where + ".modifiers");
            km.blend_out = d::number(m, "blend_out", 0.0, where + ".modifiers");
            km.fields = m;
            kc.modifiers.push_back(std::move(km));
          }
        }
        ka.fcurves.push_back(std::move(kc));
      }
    }
    if (const Tree* nla = d::at(rec, "nla"); nla != nullptr) {
      for (const Tree& tr : nla->items()) {
        KeyTrack kt;
        kt.mute = d::boolean(tr, "mute", false, where + ".nla");
        kt.is_solo = d::boolean(tr, "is_solo", false, where + ".nla");
        if (const Tree* strips = d::at(tr, "strips"); strips != nullptr) {
          for (const Tree& s : strips->items()) {
            KeyStrip st;
            const std::string w = where + ".nla.strips";
            st.action = d::text(s, "action", "", w);
            st.slot = d::text(s, "slot", "", w);
            st.frame_start = d::number(s, "frame_start", 0.0, w);
            st.frame_end = d::number(s, "frame_end", 0.0, w);
            st.action_frame_start = d::number(s, "action_frame_start", 0.0, w);
            st.action_frame_end = d::number(s, "action_frame_end", 0.0, w);
            st.scale = d::number(s, "scale", 1.0, w);
            st.repeat = d::number(s, "repeat", 1.0, w);
            st.blend_in = d::number(s, "blend_in", 0.0, w);
            st.blend_out = d::number(s, "blend_out", 0.0, w);
            st.extrapolation = static_cast<int32_t>(int_at(s, "extrapolation", 1));
            st.use_reverse = d::boolean(s, "use_reverse", false, w);
            st.use_animated_influence = d::boolean(s, "use_animated_influence", false, w);
            st.use_animated_time = d::boolean(s, "use_animated_time", false, w);
            st.mute = d::boolean(s, "mute", false, w);
            kt.strips.push_back(std::move(st));
          }
        }
        ka.nla.push_back(std::move(kt));
      }
    }
    if (const Tree* drivers = d::at(rec, "drivers"); drivers != nullptr) {
      for (const Tree& dr : drivers->items()) ka.driver_paths.push_back(d::text(dr, "data_path", "", where + ".drivers"));
    }
    ks.animations.push_back(std::move(ka));
  }
  return ks;
}

// ---------------------------------------------------------------------------------------------------------------------
// Replay

SessionIO replaying_io(const SessionLog& log) {
  SessionIO io;
  auto bodies_seen = std::make_shared<bool>(false);
  auto pose_calls = std::make_shared<size_t>(0);
  auto mesh_calls = std::make_shared<size_t>(0);
  io.bodies = [&log, bodies_seen](std::span<const int32_t> b) {
    if (!std::equal(b.begin(), b.end(), log.bodies.begin(), log.bodies.end())) {
      throw MotionError("BRG-REPLAY", "the session names other body columns than the recorded log's");
    }
    *bodies_seen = true;
  };
  io.simplify = [](bool) {};
  io.oracle = replay(log.poses, pose_calls);
  io.meshes = [&log, mesh_calls](std::span<const int32_t> objects, std::span<const int32_t> F, std::span<const float> s) {
    const size_t k = *mesh_calls;
    if (k >= log.meshes.size()) {
      throw MotionError("BRG-REPLAY", "mesh probe " + std::to_string(k) + " goes past the end of the recorded log");
    }
    const MeshCall& c = log.meshes[k];
    if (!std::equal(objects.begin(), objects.end(), c.objects.begin(), c.objects.end()) ||
        !std::equal(F.begin(), F.end(), c.F.begin(), c.F.end()) ||
        s.size() != c.s.size() || (!s.empty() && std::memcmp(s.data(), c.s.data(), s.size() * sizeof(float)) != 0)) {
      throw MotionError("BRG-REPLAY", "mesh probe " + std::to_string(k) + " requests other objects or times than the "
                                      "recorded log's");
    }
    ++*mesh_calls;
    return c.meshes;
  };
  io.progress = [](double, std::string_view) {};
  return io;
}

// ---------------------------------------------------------------------------------------------------------------------
// The runner

namespace {

struct Column {
  int32_t object = -1;       // exchange objects[] index
  Role role = Role::kBody;
  int32_t body = -1;         // Statics::bodies index (kBody, kCarrier)
  int32_t camera = -1;       // Statics::cameras index (kRig)
  int32_t inherit_of = -1;   // a check column: the column of the `inherit` body whose frame its object must keep
  std::string name;
  ModeChoice mode;
  bool candidate = false;    // static by provenance: confirmed by its samples
  bool leader = true;        // a camera that is not another camera's rig follower
};

struct Answer {
  std::vector<RawPoses> raws;   // per message
  PoseBatch batch;              // decoded, in the order of the calls
  std::vector<int64_t> times;   // the realized time of each call
};

using P3 = std::array<double, 3>;

class Runner {
 public:
  Runner(const bl::ExchangeIn& ex, const SessionOptions& opt, const SessionIO& io) : ex_(ex), opt_(opt), io_(io) {}
  SessionResult run();

 private:
  void refuse(bl::ReportItem it) {
    res_.report.push_back(std::move(it));
    refused_ = true;
  }
  void progress(double f, const char* stage) {
    if (io_.progress) io_.progress(f, stage);
  }
  void check_drivers();
  void plan_columns();
  void build_scene();
  Answer ask(const std::vector<int64_t>& times, const char* stage, bool export_instant = false);
  std::vector<int64_t> spread(int n) const;
  void survey();
  void rigidity();
  void simplify_check();
  void confirm();
  void sample();
  void finish();
  std::string name_of(int32_t col) const { return cols_[static_cast<size_t>(col)].name; }
  bl::Statics& st() { return res_.statics; }

  const bl::ExchangeIn& ex_;
  SessionOptions opt_;
  const SessionIO& io_;
  SessionResult res_;
  bool refused_ = false;
  Oracle oracle_;   // io_.oracle, logging
  MeshOracle meshes_;
  std::vector<Column> cols_;
  MotionScene scene_;
  std::unique_ptr<ErrorModel> model_;
  std::unique_ptr<render::ThreadPool> pool_;
  // survey results
  PoseBatch survey_;                    // decoded survey samples, ascending
  std::vector<RawPoses> survey_raws_;
  std::vector<int64_t> survey_times_, survey_targets_;
  RigGroups rigs_;
  std::vector<int32_t> rig_cols_;       // camera columns, in rigs_.cameras' order
  std::vector<RigGroups> inherit_;      // per `inherit` body: [body, parts...]
  std::vector<int32_t> candidates_;     // static-candidate columns
  std::vector<uint8_t> varies_;         // per column: a candidate whose samples vary
  int64_t first_bad_ns_ = 0;
  // outputs
  std::vector<BodyPipeline> pipes_;
  std::vector<int32_t> pipe_col_;
  std::vector<SampleSet> sets_;         // per column with a pipeline
  std::vector<int32_t> set_of_;         // per column: index into sets_ or -1
  RoundStats stats_;
  std::map<std::string, int64_t> calls_by_stage_;
  int64_t messages_ = 0;
  double blender_s_ = 0.0, convert_s_ = 0.0, fit_s_ = 0.0;
  int64_t span_t0_ = 0, span_t1_ = 0;
};

// ---- drivers -------------------------------------------------------------------------------------------------------

// BLK-DRIVER (§6.1): a driver Blender holds invalid (its expression does not parse, or a script it may not run) moves
// nothing in Blender's own evaluation but would in a render with scripts allowed; refused with its expression named.
void Runner::check_drivers() {
  for (size_t a = 0; a < ex_.animations.items().size(); ++a) {
    const Tree& rec = ex_.animations.items()[a];
    const Tree* ds = d::at(rec, "drivers");
    if (ds == nullptr) continue;
    for (const Tree& dr : ds->items()) {
      if (d::boolean(dr, "is_valid", true, "drivers[]")) continue;
      const std::string owner = d::text(rec, "owner", "animations[" + std::to_string(a) + "]", "animations[]");
      refuse(item("BLK-DRIVER", "refuse", owner,
                  "the driver of " + d::text(dr, "data_path", "", "drivers[]") + " is invalid: " +
                      d::text(dr, "expression", "", "drivers[]")));
    }
  }
}

// ---- columns -------------------------------------------------------------------------------------------------------

void Runner::plan_columns() {
  const int32_t n = static_cast<int32_t>(ex_.objects.size());
  std::set<int32_t> taken;
  const auto add = [&](Column c) {
    if (!taken.insert(c.object).second) proto("object " + std::to_string(c.object) + " is two body columns");
    cols_.push_back(std::move(c));
  };
  for (size_t b = 0; b < st().bodies.size(); ++b) {
    const bl::BodyStatic& body = st().bodies[b];
    Column c;
    c.object = body.object;
    c.role = body.role == bl::BodyStatic::Role::kCarrier ? Role::kCarrier : Role::kBody;
    c.body = static_cast<int32_t>(b);
    c.name = body.name;
    add(std::move(c));
  }
  const std::set<std::string> extra(opt_.cameras.begin(), opt_.cameras.end());
  for (size_t j = 0; j < st().cameras.size(); ++j) {
    const bl::CameraStatic& cam = st().cameras[j];
    if (!cam.simulate && extra.count(cam.cam.name) == 0) continue;
    Column c;
    c.object = cam.rig_object;
    c.role = Role::kRig;
    c.camera = static_cast<int32_t>(j);
    c.name = "camera " + cam.cam.name;
    add(std::move(c));
  }
  if (std::none_of(cols_.begin(), cols_.end(), [](const Column& c) { return c.role == Role::kRig; })) {
    proto("no camera to simulate");
  }
  // an `inherit` body's other parts: the instances and the instancer's own geometry are sampled, and must keep their pose
  // in the instancer's frame (ID-RANGE otherwise); a static body needs no check
  for (size_t b = 0; b < st().bodies.size(); ++b) {
    const bl::BodyStatic& body = st().bodies[b];
    if (body.T_world_static) continue;
    for (const int32_t part : body.parts) {
      if (part == body.object || part < 0 || part >= n) continue;
      Column c;
      c.object = part;
      c.role = Role::kBody;
      c.inherit_of = static_cast<int32_t>(b);
      c.name = ex_.objects[static_cast<size_t>(part)].name_full + " (part of " + body.name + ")";
      add(std::move(c));
    }
  }
}

void Runner::build_scene() {
  const double k = ex_.scale_length();
  scene_.scale_length = k;
  scene_.bodies.resize(cols_.size());
  for (size_t i = 0; i < cols_.size(); ++i) {
    const Column& c = cols_[i];
    MotionBody& mb = scene_.bodies[i];
    mb.role = c.role;
    if (c.body >= 0) {
      const bl::BodyStatic& b = st().bodies[static_cast<size_t>(c.body)];
      mb.rho_m = b.rho;
      mb.centre_m = {b.c[0], b.c[1], b.c[2]};
      mb.S_ref = b.S;
    } else if (c.role == Role::kBody) {   // an inherit part: only its pose in the instancer's frame is checked
      const auto& m = ex_.objects[static_cast<size_t>(c.object)].matrix_world;
      const bl::Polar p = bl::polar({m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]});
      mb.S_ref = p.S;
    }
  }
  for (const bl::LightStatic& l : st().lights) {
    if (!l.light.body) continue;
    MotionLight ml;
    ml.offset_m = {l.light.position[0], l.light.position[1], l.light.position[2]};
    ml.spot = l.light.kind == render::BundleLightKind::kSpot;
    ml.directional = l.light.kind == render::BundleLightKind::kDirectional;
    ml.casts_shadows = l.light.casts_shadows;
    scene_.bodies[static_cast<size_t>(*l.light.body)].lights.push_back(ml);
  }
  for (size_t i = 0; i < cols_.size(); ++i) {
    if (cols_[i].role != Role::kRig) continue;
    const bl::CameraStatic& cam = st().cameras[static_cast<size_t>(cols_[i].camera)];
    MotionCamera mc;
    mc.rig_body = static_cast<int32_t>(i);
    mc.gain_px_per_rad = angular_gain(cam.cam);   // std::domain_error: a camera that is not injective over its image
    mc.clip_start_m = cam.clip_start_m;
    scene_.cameras.push_back(mc);
  }
  // the static geometry, world frame (Blender origin), metres
  for (const bl::BodyStatic& b : st().bodies) {
    if (!b.T_world_static || b.role == bl::BodyStatic::Role::kCarrier) continue;
    const geo::M4& T = *b.T_world_static;
    const auto base = static_cast<uint32_t>(scene_.static_vertices.size() / 3);
    const std::vector<float>& x = b.mesh.positions;
    for (size_t i = 0; i < x.size(); i += 3) {
      const double a = x[i], b2 = x[i + 1], c2 = x[i + 2];
      scene_.static_vertices.push_back(T[0] * a + T[1] * b2 + T[2] * c2 + T[3]);
      scene_.static_vertices.push_back(T[4] * a + T[5] * b2 + T[6] * c2 + T[7]);
      scene_.static_vertices.push_back(T[8] * a + T[9] * b2 + T[10] * c2 + T[11]);
    }
    for (const uint32_t idx : b.mesh.indices) scene_.static_indices.push_back(base + idx);
  }
}

// ---- asking the oracle ---------------------------------------------------------------------------------------------

std::vector<int64_t> Runner::spread(int n) const {
  const int64_t T = res_.time->to_ns(static_cast<float>(res_.time->frame_end()));
  std::vector<int64_t> out;
  const int64_t m = std::max(n, 1);
  for (int64_t k = 0; k < m; ++k) out.push_back(m == 1 || T == 0 ? 0 : (T * k + (m - 1) / 2) / (m - 1));
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

Answer Runner::ask(const std::vector<int64_t>& times_in, const char* stage, bool export_instant) {
  const TimeMap& time = *res_.time;
  std::vector<int64_t> times = times_in;
  std::sort(times.begin(), times.end());
  times.erase(std::unique(times.begin(), times.end()), times.end());
  std::vector<int32_t> cF;
  std::vector<float> cS;
  for (const int64_t t : times) {
    int32_t F = 0;
    float s = 0.0f;
    time.request(t, &F, &s);
    if (!cF.empty() && F == cF.back() && std::bit_cast<uint32_t>(s) == std::bit_cast<uint32_t>(cS.back())) continue;
    cF.push_back(F);
    cS.push_back(s);
  }
  const auto nb = static_cast<int32_t>(cols_.size());
  Answer a;
  a.batch.bodies = nb;
  const size_t cap = static_cast<size_t>(std::clamp(opt_.sample.max_per_message, 1, 512));
  for (size_t off = 0; off < cF.size(); off += cap) {
    const size_t cnt = std::min(cF.size() - off, cap);
    const std::span<const int32_t> F(cF.data() + off, cnt);
    const std::span<const float> S(cS.data() + off, cnt);
    const auto t0 = Clock::now();
    RawPoses raw = oracle_(F, S);
    blender_s_ += since(t0);
    if (raw.F.empty() && raw.s.empty()) {
      raw.F.assign(F.begin(), F.end());
      raw.s.assign(S.begin(), S.end());
    } else if (!(raw.F.size() == cnt && std::equal(F.begin(), F.end(), raw.F.begin()) &&
                 std::memcmp(S.data(), raw.s.data(), cnt * sizeof(float)) == 0)) {
      throw MotionError("BRG-PROTO", "a poses answer echoes other requests than were sent");
    }
    if (raw.n != static_cast<int32_t>(cnt)) throw MotionError("BRG-PROTO", "a poses answer has the wrong length");
    if (raw.bodies != nb) throw MotionError("BRG-PROTO", "a poses answer has the wrong body count");
    if (export_instant && off == 0) {   // the first sample is the export instant: everything exported must be there
      for (int32_t b = 0; b < nb; ++b) {
        if (raw.present.size() != static_cast<size_t>(raw.n) * static_cast<size_t>(nb)) break;   // decode names the shape
        if (!raw.present[static_cast<size_t>(b)]) {
          throw MotionError("BRG-STATE", name_of(b) + " is absent at the export instant (frame " + std::to_string(raw.F[0]) +
                                             "): an exporter or render-mirror fault");
        }
      }
      if (!raw.unknown.empty() && raw.unknown[0] != 0) {
        throw MotionError("BRG-STATE", std::to_string(raw.unknown[0]) + " render-visible instances at the export "
                                       "instant are not in the exchange" +
                                           (raw.unknown_keys.empty() ? std::string() : " (" + raw.unknown_keys + ")"));
      }
    }
    PoseBatch decoded = decode(raw, time, scene_);
    for (size_t i = 0; i < cnt; ++i) {
      if (raw.unknown[i] != 0) {
        throw MotionError("VIS-ANIM", std::to_string(raw.unknown[i]) + " render-visible instances not in the exchange at "
                                      "frame " + std::to_string(raw.F[i]) +
                                          (raw.unknown_keys.empty() ? std::string() : " (" + raw.unknown_keys + ")"));
      }
    }
    for (size_t k = 0; k < raw.present.size(); ++k) {
      if (!raw.present[k]) {
        throw MotionError("VIS-ANIM", name_of(static_cast<int32_t>(k % static_cast<size_t>(nb))) +
                                          " is absent at frame " + std::to_string(raw.F[k / static_cast<size_t>(nb)]));
      }
    }
    a.times.insert(a.times.end(), decoded.t_ns.begin(), decoded.t_ns.end());
    a.batch.append(decoded);
    a.raws.push_back(std::move(raw));
    ++messages_;
  }
  calls_by_stage_[stage] += static_cast<int64_t>(cF.size());
  return a;
}

// ---- the survey ----------------------------------------------------------------------------------------------------

void Runner::survey() {
  progress(0.03, "survey");
  if (io_.simplify) io_.simplify(false);
  survey_targets_ = spread(opt_.survey_samples);
  Answer a = ask(survey_targets_, "survey", true);
  survey_ = std::move(a.batch);
  survey_raws_ = std::move(a.raws);
  survey_times_ = std::move(a.times);
  // the camera rigs (§4.7.9)
  for (size_t i = 0; i < cols_.size(); ++i) {
    if (cols_[i].role == Role::kRig) rig_cols_.push_back(static_cast<int32_t>(i));
  }
  rigs_ = group_rigs(survey_, rig_cols_);
  for (size_t j = 0; j < rig_cols_.size(); ++j) cols_[static_cast<size_t>(rig_cols_[j])].leader = rigs_.rig[j] == rig_cols_[j];
  // `inherit` bodies: every part keeps its pose in the instancer's frame (a group of one rig)
  for (size_t b = 0; b < st().bodies.size(); ++b) {
    std::vector<int32_t> group{static_cast<int32_t>(b)};
    for (size_t i = 0; i < cols_.size(); ++i) {
      if (cols_[i].inherit_of == static_cast<int32_t>(b)) group.push_back(static_cast<int32_t>(i));
    }
    if (group.size() < 2) continue;
    RigGroups g = group_rigs(survey_, group);
    for (size_t j = 1; j < group.size(); ++j) {
      if (g.rig[j] != group[0]) {
        throw MotionError("ID-RANGE", st().bodies[b].name + ": `inherit` merges its instances into one body, but " +
                                          name_of(group[j]) + " moves relative to the instancer");
      }
    }
    inherit_.push_back(std::move(g));
  }
  // the scene box: the survey's body positions and the static geometry, widened (a light carrier's D_max and d_recv only
  // grow with it)
  std::array<double, 6> box{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
  const auto grow = [&box](double x, double y, double z, double r) {
    const double v[3] = {x, y, z};
    for (int k = 0; k < 3; ++k) {
      box[static_cast<size_t>(k)] = std::min(box[static_cast<size_t>(k)], v[k] - r);
      box[static_cast<size_t>(3 + k)] = std::max(box[static_cast<size_t>(3 + k)], v[k] + r);
    }
  };
  for (size_t i = 0; i < survey_.size(); ++i) {
    for (size_t c = 0; c < cols_.size(); ++c) {
      const rot::V& p = survey_.p[survey_.at(i, static_cast<int32_t>(c))];
      grow(p.x, p.y, p.z, scene_.bodies[c].rho_m);
    }
  }
  for (size_t v = 0; v + 2 < scene_.static_vertices.size(); v += 3) {
    grow(scene_.static_vertices[v], scene_.static_vertices[v + 1], scene_.static_vertices[v + 2], 0.0);
  }
  if (box[0] <= box[3]) {
    const double diag = std::sqrt((box[3] - box[0]) * (box[3] - box[0]) + (box[4] - box[1]) * (box[4] - box[1]) +
                                  (box[5] - box[2]) * (box[5] - box[2]));
    const double pad = opt_.box_widen * diag;
    for (size_t k = 0; k < 3; ++k) {
      box[k] -= pad;
      box[3 + k] += pad;
    }
    scene_.scene_box = box;
  }
}

void Runner::rigidity() {
  const std::vector<int32_t> objects = bl::rigidity_probe_objects(ex_);
  if (objects.empty() || !meshes_) return;
  progress(0.06, "rigidity probes");
  const TimeMap& time = *res_.time;
  std::vector<int32_t> F;
  std::vector<float> S;
  for (const int64_t t : spread(opt_.rigidity_samples)) {
    int32_t f = 0;
    float s = 0.0f;
    time.request(t, &f, &s);
    F.push_back(f);
    S.push_back(s);
  }
  const auto t0 = Clock::now();
  const RawMeshes raw = meshes_(objects, F, S);
  blender_s_ += since(t0);
  calls_by_stage_["rigidity"] += static_cast<int64_t>(F.size());
  if (raw.table.size() % 4 != 0) throw MotionError("BRG-PROTO", "a meshes answer has a malformed table");
  std::map<int32_t, std::vector<std::vector<float>>> probes;
  for (size_t r = 0; r < raw.table.size(); r += 4) {
    const int64_t obj = raw.table[r], nv = raw.table[r + 2], off = raw.table[r + 3];
    if (obj < 0 || nv < 0 || off < 0 || static_cast<size_t>(off + 3 * nv) > raw.co.size() ||
        std::find(objects.begin(), objects.end(), static_cast<int32_t>(obj)) == objects.end()) {
      throw MotionError("BRG-PROTO", "a meshes answer names an object or vertices outside what was asked");
    }
    probes[static_cast<int32_t>(obj)].emplace_back(raw.co.begin() + off, raw.co.begin() + off + 3 * nv);
  }
  const double tol_bu = opt_.convert.rigidity_tol_m / ex_.scale_length();
  for (const int32_t o : objects) {
    const bl::ObjectIn& ob = ex_.objects[static_cast<size_t>(o)];
    const bl::MeshIn& mesh = ex_.meshes[static_cast<size_t>(ob.mesh)];
    const auto it = probes.find(o);
    if (it == probes.end()) throw MotionError("BRG-PROTO", "a meshes answer misses " + ob.name_full);
    if (const auto r = bl::check_rigidity(ob.name_full, mesh.co, it->second, tol_bu)) {
      refuse(*r);
    }
  }
}

void Runner::simplify_check() {
  if (!io_.simplify) return;
  progress(0.08, "simplify check");
  const size_t n = std::min<size_t>(static_cast<size_t>(std::max(opt_.simplify_samples, 1)), survey_targets_.size());
  const std::vector<int64_t> times(survey_targets_.begin(), survey_targets_.begin() + static_cast<std::ptrdiff_t>(n));
  io_.simplify(true);
  Answer on = ask(times, "simplify");
  // the first n samples of the survey's first message against the same requests with Simplify on
  const RawPoses& a = survey_raws_.front();
  const RawPoses& b = on.raws.front();
  const auto nb = static_cast<size_t>(a.bodies);
  bool same = b.n > 0 && a.n >= b.n;
  for (size_t i = 0; same && i < static_cast<size_t>(b.n); ++i) {
    same = std::memcmp(&a.c_final[i], &b.c_final[i], sizeof(double)) == 0 &&
           std::memcmp(&a.m[i * nb * 12], &b.m[i * nb * 12], nb * 12 * sizeof(float)) == 0 &&
           std::memcmp(&a.present[i * nb], &b.present[i * nb], nb) == 0 && a.unknown[i] == b.unknown[i];
  }
  res_.simplify = same;
  if (!same) io_.simplify(false);
}

// ---- static candidates ---------------------------------------------------------------------------------------------

namespace {

bool same_pose(const PoseBatch& a, size_t ka, const PoseBatch& b, size_t kb) {
  return std::memcmp(&a.q[ka], &b.q[kb], sizeof(rot::Q)) == 0 && std::memcmp(&a.p[ka], &b.p[kb], sizeof(rot::V)) == 0;
}

}  // namespace

void Runner::confirm() {
  const TimeMap& time = *res_.time;
  varies_.assign(cols_.size(), 0);
  if (candidates_.empty()) return;
  progress(0.1, "static bodies");
  std::vector<int64_t> times;
  const int64_t frames = int64_t{time.frame_end()} - time.frame_start();
  const int64_t stride = std::max<int64_t>(1, (frames + opt_.confirm_max_samples - 1) / std::max(1, opt_.confirm_max_samples));
  for (int64_t f = 0; f <= frames; f += stride) times.push_back(time.to_ns(static_cast<float>(time.frame_start() + f)));
  Answer a = ask(times, "static");
  for (size_t i = 0; i < a.batch.size(); ++i) {
    for (const int32_t c : candidates_) {
      if (varies_[static_cast<size_t>(c)]) continue;
      if (!same_pose(a.batch, a.batch.at(i, c), survey_, survey_.at(0, c))) {
        varies_[static_cast<size_t>(c)] = 1;
        first_bad_ns_ = a.batch.t_ns[i];
      }
    }
  }
}

// ---- sampling ------------------------------------------------------------------------------------------------------

void Runner::sample() {
  const TimeMap& time = *res_.time;
  const KeyScene keys = key_scene(ex_);
  const bool lenient = opt_.convert.policy == bl::Policy::kLenient;
  // modes by provenance (§4.7.1)
  for (size_t i = 0; i < cols_.size(); ++i) {
    Column& c = cols_[i];
    if (c.inherit_of >= 0) continue;
    c.mode = choose_mode(ex_, c.object, time.framelen());
    if (!c.mode.code.empty()) {
      refuse(item(c.mode.code, "refuse", c.name, c.mode.message));
    } else if (c.mode.source == Source::kFrameState) {
      refuse(item("PHY-PENDING", "refuse", c.name,
                  "a rigid body or transform cache moves this body: frame-state sampling arrives with BL-10"));
    } else if (c.mode.source == Source::kNone) {
      const std::optional<geo::M4>& still = c.role == Role::kRig ? st().cameras[static_cast<size_t>(c.camera)].T_world_static
                                                                : st().bodies[static_cast<size_t>(c.body)].T_world_static;
      c.candidate = still.has_value() && c.leader;
    }
    if (c.role == Role::kRig && !c.leader) c.candidate = false;
  }
  if (refused_) return;
  for (size_t i = 0; i < cols_.size(); ++i) {
    if (cols_[i].candidate) candidates_.push_back(static_cast<int32_t>(i));
  }
  confirm();
  for (const int32_t c : candidates_) {
    if (!varies_[static_cast<size_t>(c)]) scene_.bodies[static_cast<size_t>(c)].still = true;
  }
  try {
    model_ = std::make_unique<ErrorModel>(scene_);
  } catch (const std::domain_error& e) {
    throw MotionError("CAM-MODEL", e.what());
  }
  for (const int32_t c : candidates_) {
    if (!varies_[static_cast<size_t>(c)]) continue;
    const auto it = motion_source(cols_[static_cast<size_t>(c)].mode, true, opt_.convert.policy);
    if (!it) continue;
    bl::ReportItem r = *it;
    r.object = cols_[static_cast<size_t>(c)].name;
    r.message += " (first seen at " + std::to_string(first_bad_ns_) + " ns)";
    if (!lenient) refuse(std::move(r));
    else res_.report.push_back(std::move(r));
  }
  if (refused_) return;
  // pipelines: every moving column, a camera rig's leader alone
  set_of_.assign(cols_.size(), -1);
  for (size_t i = 0; i < cols_.size(); ++i) {
    const Column& c = cols_[i];
    if (c.inherit_of >= 0 || !c.leader) continue;
    if (c.candidate && !varies_[i]) continue;
    std::vector<DeclaredKey> dk;
    if (c.mode.source == Source::kKeyed) {
      try {
        dk = declared_keys(keys, c.object, time, span_t0_, span_t1_, lenient);
      } catch (const MotionError& e) {
        throw MotionError(e.code, without_code(e) + " (body " + c.name + ")");
      }
    }
    set_of_[i] = static_cast<int32_t>(pipes_.size());
    pipes_.emplace_back(static_cast<int32_t>(i), Mode::kKeyed, time, std::move(dk), opt_.sample, opt_.detect, opt_.fit);
    pipe_col_.push_back(static_cast<int32_t>(i));
  }
  const auto on_batch = [&](const PoseBatch& b) {
    check_rigs(b, rigs_);
    for (RigGroups& g : inherit_) {
      try {
        check_rigs(b, g);
      } catch (const MotionError& e) {
        if (e.code != "CAM-RIG") throw;
        throw MotionError("ID-RANGE", "`inherit` merges instances into one body, but " + without_code(e));
      }
    }
    for (const int32_t c : candidates_) {   // the backstop: a body that was still when looked at moved between samples
      if (varies_[static_cast<size_t>(c)]) continue;
      for (size_t i = 0; i < b.size(); ++i) {
        if (!same_pose(b, b.at(i, c), survey_, survey_.at(0, c))) {
          throw MotionError("MOT-SOURCE", name_of(c) + " has no animation source in its chain but moves at " +
                                              std::to_string(b.t_ns[i]) + " ns, between the frames looked at");
        }
      }
    }
  };
  const auto t0 = Clock::now();
  const double blender_before = blender_s_;
  if (!pipes_.empty()) {
    progress(0.12, "sampling");
    stats_ = drive(oracle_, time, *model_, pipes_, *pool_,
                   [this](double f, std::string_view) { progress(0.12 + 0.8 * f, "sampling"); }, on_batch,
                   [this](int32_t col) { return name_of(col); });
  }
  fit_s_ = since(t0) - (blender_s_ - blender_before);
  for (BodyPipeline& p : pipes_) sets_.push_back(std::move(p).take());
}

// ---- the result ----------------------------------------------------------------------------------------------------

Tree breakpoint_tree(const Breakpoint& b) {
  Tree t = Tree::dict();
  t["t_ns"] = Tree::integer(b.t_ns);
  t["class"] = Tree::string(b.cls == Continuity::kStep ? "step" : b.cls == Continuity::kC0 ? "C0" : "C1");
  t["jump_px"] = Tree::real(b.jump_px);
  t["dv_pxps"] = Tree::real(b.dv_pxps);
  t["declared"] = Tree::boolean(b.declared);
  t["snapped"] = Tree::boolean(b.snapped);
  t["dt_bound_ns"] = Tree::real(b.dt_bound_ns);
  return t;
}

void Runner::finish() {
  const TimeMap& time = *res_.time;
  const size_t nbody = st().bodies.size(), ncam = st().cameras.size();
  res_.body_trajectory.assign(nbody, -1);
  res_.camera_trajectory.assign(ncam, -1);
  res_.camera_t_rig_cam.assign(ncam, {});
  // the body items of the report (POS-RES, TIME-RES, MOT-FAR, LGT-MOVE-NEAR, MOT-ALIAS)
  const double eps = opt_.sample.eps_px;
  for (size_t k = 0; k < sets_.size(); ++k) {
    for (bl::ReportItem r : body_report(sets_[k], time, eps)) {
      r.object = name_of(pipe_col_[k]);
      res_.report.push_back(std::move(r));
    }
  }
  // origin_world: the AABB centre of every motion sample (moving bodies, carriers, cameras)
  std::vector<P3> samples;
  for (const SampleSet& s : sets_) {
    P3 lo{INFINITY, INFINITY, INFINITY}, hi{-INFINITY, -INFINITY, -INFINITY};
    for (const Track* t : {&s.fit, &s.held_out}) {
      for (const rot::V& p : t->p) {
        lo = {std::min(lo[0], p.x), std::min(lo[1], p.y), std::min(lo[2], p.z)};
        hi = {std::max(hi[0], p.x), std::max(hi[1], p.y), std::max(hi[2], p.z)};
      }
    }
    if (lo[0] <= hi[0]) {
      samples.push_back(lo);
      samples.push_back(hi);
    }
  }
  for (const Column& c : cols_) {
    if (c.role != Role::kRig) continue;
    const bl::CameraStatic& cam = st().cameras[static_cast<size_t>(c.camera)];
    samples.push_back({cam.T_world_export[3], cam.T_world_export[7], cam.T_world_export[11]});
  }
  for (size_t b = 0; b < nbody; ++b) {
    const bl::BodyStatic& body = st().bodies[b];
    if (!body.T_world_static && set_of_[b] < 0) samples.push_back({body.T_world_export[3], body.T_world_export[7], body.T_world_export[11]});
  }
  res_.origin_world = bl::origin_world(st(), samples);
  const P3& o = res_.origin_world;

  // trajectories
  Tree bodies = Tree::list();
  const auto record = [&](const Column& c, const char* mode, int32_t traj, const SampleSet* set) {
    Tree r = Tree::dict();
    const bl::ObjectIn& ob = ex_.objects[static_cast<size_t>(c.object)];
    r["key"] = Tree::string(ob.key);
    r["name"] = Tree::string(c.name);
    r["object"] = Tree::integer(c.object);
    r["role"] = Tree::string(c.role == Role::kRig ? "rig" : c.role == Role::kCarrier ? "light_carrier" : "body");
    r["mode"] = Tree::string(mode);
    r["statics_index"] = Tree::integer(c.body);
    r["camera_index"] = Tree::integer(c.camera);
    r["id"] = Tree::integer(c.body >= 0 ? st().bodies[static_cast<size_t>(c.body)].id : 0);
    r["trajectory"] = Tree::integer(c.role == Role::kRig ? res_.camera_trajectory[static_cast<size_t>(c.camera)] : traj);
    const MotionBody& mb = scene_.bodies[static_cast<size_t>(&c - cols_.data())];
    r["rho_m"] = Tree::real(mb.rho_m);
    if (c.body >= 0) {
      const bl::BodyStatic& b = st().bodies[static_cast<size_t>(c.body)];
      r["scale_S"] = reals(b.S);
      r["offset_c"] = reals(b.c);
    }
    if (set != nullptr && set->fitted) {
      const PiecewiseFit& m = set->motion;
      r["pieces"] = Tree::integer(static_cast<int64_t>(m.pieces.size()));
      Tree bps = Tree::list();
      for (const Breakpoint& b : m.breakpoints) bps.push_back(breakpoint_tree(b));
      r["breakpoints"] = std::move(bps);
      Tree holds = Tree::list();
      for (const auto& h : m.holds) holds.push_back(Tree::ints({h.first, h.second}));
      r["holds"] = std::move(holds);
      r["max_validation_px"] = Tree::real(m.max_validation_px);
      r["floor_px"] = Tree::real(m.floor_px);
      r["cv_px"] = Tree::real(m.cv_px);
      r["gravity_ratio"] = Tree::real(m.gravity_ratio);
    }
    bodies.push_back(std::move(r));
  };
  const auto add_trajectory = [&](SessionTrajectory t) {
    res_.trajectories.push_back(std::move(t));
    return static_cast<int32_t>(res_.trajectories.size() - 1);
  };
  const auto reach = [&](const render::Trajectory& tr) {
    double m = 0.0;
    for (size_t k = 0; k + 2 < tr.pos.size(); k += 3) {
      m = std::max(m, std::sqrt(tr.pos[k] * tr.pos[k] + tr.pos[k + 1] * tr.pos[k + 1] + tr.pos[k + 2] * tr.pos[k + 2]));
    }
    return m;
  };
  std::vector<int32_t> traj_of_col(cols_.size(), -1);
  for (size_t i = 0; i < cols_.size(); ++i) {
    const Column& c = cols_[i];
    if (set_of_[i] < 0) continue;
    const SampleSet& s = sets_[static_cast<size_t>(set_of_[i])];
    if (!s.fitted) throw std::logic_error("session: a pipeline ended without its fit");
    SessionTrajectory t;
    t.traj = to_trajectory(s.motion, o);
    t.holds = s.motion.holds;
    const double rho = c.role == Role::kRig ? 0.0 : scene_.bodies[i].rho_m;
    const double r = reach(t.traj) + rho;
    if (r > 16.0) {
      refuse(item("GEO-16M", "refuse", c.name, "reaches " + std::to_string(r) + " m from origin_world (limit 16 m, "
                                                "DESIGN §5.3): the moving bodies and cameras span more than the rule allows",
                  r, bl::BoundKind::kMeasured));
    }
    traj_of_col[i] = add_trajectory(std::move(t));
    if (c.body >= 0) res_.body_trajectory[static_cast<size_t>(c.body)] = traj_of_col[i];
  }
  // static camera rigs (a leader that is still): K = 6 equal controls over the trajectory span
  for (size_t i = 0; i < cols_.size(); ++i) {
    const Column& c = cols_[i];
    if (c.role != Role::kRig || !c.leader || traj_of_col[i] >= 0) continue;
    const bl::CameraStatic& cam = st().cameras[static_cast<size_t>(c.camera)];
    SessionTrajectory t;
    t.traj = bl::static_rig(*cam.T_world_static, o, span_t0_, span_t1_);
    t.holds = {{span_t0_, span_t1_}};   // as the static assembly declares it: still over the whole span
    traj_of_col[i] = add_trajectory(std::move(t));
  }
  // cameras: a follower's rig is its leader's, and its pose in it the constant D(rel) of the survey
  static const geo::M4 kFlip{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1};
  for (size_t j = 0; j < rig_cols_.size(); ++j) {
    const int32_t col = rig_cols_[j];
    const Column& c = cols_[static_cast<size_t>(col)];
    const int32_t leader = rigs_.rig[j];
    res_.camera_trajectory[static_cast<size_t>(c.camera)] = traj_of_col[static_cast<size_t>(leader)];
    geo::M4 T = kFlip;
    if (leader != col) {
      T = geo::matmul(geo::pose_to_transform(rigs_.q_rig_cam[j], rigs_.p_rig_cam[j]), kFlip);
    }
    res_.camera_t_rig_cam[static_cast<size_t>(c.camera)] = T;
  }
  for (size_t i = 0; i < cols_.size(); ++i) {
    const Column& c = cols_[i];
    if (c.inherit_of >= 0) continue;
    size_t src = i;   // a rig follower is as moving as its leader
    for (size_t j = 0; j < rig_cols_.size(); ++j) {
      if (rig_cols_[j] == static_cast<int32_t>(i)) src = static_cast<size_t>(rigs_.rig[j]);
    }
    const SampleSet* s = set_of_[i] >= 0 ? &sets_[static_cast<size_t>(set_of_[i])] : nullptr;
    const bool moving = set_of_[src] >= 0;
    record(c, moving ? "keyed" : "static", moving || (c.role == Role::kRig) ? traj_of_col[i] : -1, s);
  }

  // session.json (without engine, options and motion_key: the Python side adds them)
  Tree& doc = res_.session;
  doc["format"] = Tree::string("ballasim.blender_motion");
  doc["version"] = Tree::ints({1, 0});
  doc["converter"] = Tree::integer(bl::kConverterVersion);
  doc["timeline"] = time.record();
  doc["bundle_span"] = Tree::ints({-2 * time::kWindowNs, time.t_end_ns()});
  doc["run_span"] = Tree::ints({0, time.t_end_ns()});
  doc["trajectory_span"] = Tree::ints({span_t0_, span_t1_});
  doc["origin_world"] = reals(o);
  doc["simplify"] = Tree::boolean(res_.simplify);
  doc["bodies"] = std::move(bodies);
  Tree cams = Tree::list();
  for (size_t j = 0; j < ncam; ++j) {
    if (res_.camera_trajectory[j] < 0) continue;
    Tree r = Tree::dict();
    r["name"] = Tree::string(st().cameras[j].cam.name);
    r["trajectory"] = Tree::integer(res_.camera_trajectory[j]);
    r["T_rig_cam"] = reals(res_.camera_t_rig_cam[j]);
    cams.push_back(std::move(r));
  }
  doc["cameras"] = std::move(cams);
  Tree opts = bl::options_record(opt_.convert);
  opts["motion_tol_px"] = Tree::real(eps);
  opts["fps_base_snap"] = Tree::boolean(opt_.fps_base_snap);
  doc["session_options"] = std::move(opts);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------

SessionResult Runner::run() {
  const auto start = Clock::now();
  try {
    // 1. the time map and the statics
    res_.time.emplace(time_settings(ex_, opt_.fps_base_snap));
    const TimeMap& time = *res_.time;
    const int margin = std::max(opt_.span_margin_windows, 0);
    span_t0_ = -std::max(5, 2 + margin) * time::kWindowNs;   // PHASE3.md D-P3-12
    span_t1_ = time.t_end_ns() + margin * time::kWindowNs;
    opt_.sample.t0_ns = span_t0_;
    opt_.sample.t1_ns = span_t1_;
    progress(0.0, "convert");
    const auto t_convert = Clock::now();
    st() = bl::convert_statics(ex_, opt_.convert);
    convert_s_ = since(t_convert);
    if (!st().refused()) check_drivers();
    if (st().refused()) {
      refused_ = true;
    } else if (!refused_) {
      plan_columns();
      if (io_.bodies) {
        std::vector<int32_t> objects;
        for (const Column& c : cols_) objects.push_back(c.object);
        res_.log.bodies = objects;
        io_.bodies(objects);
      }
      if (!io_.oracle) throw std::invalid_argument("run_session: no oracle");
      oracle_ = recording(io_.oracle, res_.log.poses);
      if (io_.meshes) {
        meshes_ = [this](std::span<const int32_t> objects, std::span<const int32_t> F, std::span<const float> s) {
          RawMeshes r = io_.meshes(objects, F, s);
          res_.log.meshes.push_back({std::vector<int32_t>(objects.begin(), objects.end()),
                                     std::vector<int32_t>(F.begin(), F.end()), std::vector<float>(s.begin(), s.end()), r});
          return r;
        };
      }
      pool_ = std::make_unique<render::ThreadPool>(std::max(opt_.workers, 1));
      try {
        build_scene();
      } catch (const std::domain_error& e) {
        throw MotionError("CAM-MODEL", e.what());
      }
      survey();
      rigidity();
      if (!refused_) simplify_check();
      if (!refused_) sample();
      if (!refused_) finish();
    }
  } catch (const MotionError& e) {
    if (!is_refusal(e)) throw;
    res_.report.push_back(item(e.code, "refuse", "", without_code(e)));
    refused_ = true;
  }
  res_.refused = refused_;

  // the done summary and the perf record
  std::vector<bl::ReportItem> all = st().report;
  all.insert(all.end(), res_.report.begin(), res_.report.end());
  const Tree rep = bl::report_doc(all, opt_.convert.policy);
  Tree& sum = res_.summary;
  if (res_.time) {
    const TimeMap& time = *res_.time;
    sum["timeline"] = time.record();
    sum["bundle_span"] = Tree::ints({-2 * time::kWindowNs, time.t_end_ns()});
    sum["run_span"] = Tree::ints({0, time.t_end_ns()});
    sum["trajectory_span"] = Tree::ints({span_t0_, span_t1_});
    Tree prev = Tree::dict();
    const int64_t f0 = time.frame_start(), f1 = time.frame_end();
    prev["frames"] = Tree::ints({f0, f1, 1});
    prev["count"] = Tree::integer(f1 - f0 + 1);
    if (f1 - f0 < 4096) {
      std::vector<int64_t> t;
      for (int64_t f = f0; f <= f1; ++f) t.push_back(time.to_ns(static_cast<float>(f)));
      prev["times_ns"] = Tree::ints(t);
    }
    sum["preview"] = std::move(prev);
  }
  sum["report_summary"] = *rep.find("summary");
  Tree& perf = res_.perf;
  Tree by_stage = Tree::dict();
  for (const auto& [k, v] : calls_by_stage_) by_stage[k] = Tree::integer(v);
  for (const auto& [k, v] : stats_.calls_by_stage) by_stage[k] = Tree::integer(v);
  perf["calls_by_stage"] = std::move(by_stage);
  int64_t calls = 0;
  for (const auto& [k, v] : calls_by_stage_) calls += v;
  perf["calls"] = Tree::integer(calls + stats_.calls);
  perf["rounds"] = Tree::integer(stats_.rounds);
  perf["messages"] = Tree::integer(messages_ + stats_.messages);
  perf["blender_s"] = Tree::real(blender_s_);
  perf["fit_s"] = Tree::real(fit_s_);
  perf["convert_s"] = Tree::real(convert_s_);
  perf["total_s"] = Tree::real(since(start));
  perf["workers"] = Tree::integer(std::max(opt_.workers, 1));
  progress(1.0, "done");
  return std::move(res_);
}

SessionResult run_session(const bl::ExchangeIn& ex, const SessionOptions& opt, const SessionIO& io) {
  Runner r(ex, opt, io);
  return r.run();
}

}  // namespace balla::scene::motion
