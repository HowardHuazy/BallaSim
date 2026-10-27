// The assembly of a static Blender exchange (assemble.h; blender.md §3.7, §4.2, §4.7.1, §4.7.9).
#include "scene/blender/assemble.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>

#include "core/time/time.h"
#include "scene/blender/values.h"

namespace balla::scene::blender {

namespace {

__extension__ typedef __int128 i128;

[[noreturn]] void proto(const std::string& what) { throw std::invalid_argument("BRG-PROTO: " + what); }

// round_half_even(n / d), d > 0.
i128 div_round_even(i128 n, i128 d) {
  i128 q = n / d, r = n % d;
  if (r < 0) {
    q -= 1;
    r += d;
  }
  const i128 twice = r + r;
  if (twice > d || (twice == d && (q & 1) != 0)) q += 1;
  return q;
}

int64_t int_of(const Tree& t, const char* key, const std::string& where) {
  const Tree* v = detail::at(t, key);
  if (v == nullptr || v->kind() != Tree::Kind::kInt) proto(where + "." + key + " is not an integer");
  return v->as_int();
}

bool generates(const ObjectIn& ob) {   // particles or geometry nodes shown for render (they move instances keylessly)
  const Tree* mods = ob.deform.kind() == Tree::Kind::kDict ? ob.deform.find("modifiers") : nullptr;
  if (mods == nullptr || mods->kind() != Tree::Kind::kList) return false;
  for (const Tree& md : mods->items()) {
    const std::string type = detail::text(md, "type", "", "deform.modifiers[]");
    if ((type == "PARTICLE_SYSTEM" || type == "NODES") && detail::boolean(md, "show_render", true, "deform.modifiers[]")) {
      return true;
    }
  }
  return false;
}

}  // namespace

StaticTimeline static_timeline(const ExchangeIn& ex) {
  StaticTimeline t;
  const int64_t fps = int_of(ex.time, "fps", "time");
  if (fps < 1 || fps > 32767) proto("time.fps " + std::to_string(fps) + " is outside [1, 32767]");
  t.fps = static_cast<int32_t>(fps);
  const std::string bits = detail::text(ex.time, "fps_base_bits", "", "time");
  char* end = nullptr;
  const unsigned long long b = std::strtoull(bits.c_str(), &end, 16);
  if (bits.size() != 10 || bits.rfind("0x", 0) != 0 || end != bits.c_str() + bits.size() || b > 0xffffffffull) {
    proto("time.fps_base_bits " + bits + " is not 0x followed by 8 hex digits");
  }
  t.fps_base_bits = static_cast<uint32_t>(b);
  const float base = std::bit_cast<float>(t.fps_base_bits);
  if (!(std::isfinite(base) && base > 0.0f)) proto("time.fps_base is not finite and > 0");
  int e = 0;
  const float fr = std::frexp(base, &e);
  int64_t m = static_cast<int64_t>(std::ldexp(fr, 24));   // base = m 2^(e - 24) exactly, m < 2^24
  e -= 24;
  if (e >= 0) {
    if (e > 38) proto("time.fps_base is out of range");
    t.base_num = m << e;
    t.base_den = 1;
  } else {
    if (e < -62) proto("time.fps_base is out of range");
    t.base_num = m;
    t.base_den = int64_t{1} << -e;
    while (t.base_den > 1 && t.base_num % 2 == 0) {
      t.base_num /= 2;
      t.base_den /= 2;
    }
  }
  const Tree& span = ex.span.kind() == Tree::Kind::kDict && ex.span.find("frame_start") != nullptr ? ex.span : ex.time;
  const char* where = &span == &ex.span ? "span" : "time";
  const int64_t f0 = int_of(span, "frame_start", where), f1 = int_of(span, "frame_end", where);
  if (f0 < -1048574 || f1 > 1048574 || f1 < f0) {
    proto(std::string(where) + ": frames [" + std::to_string(f0) + ", " + std::to_string(f1) + "] are not a span");
  }
  t.frame_start = static_cast<int32_t>(f0);
  t.frame_end = static_cast<int32_t>(f1);
  const i128 n = static_cast<i128>(f1 - f0) * 1'000'000'000 * t.base_num;
  const i128 d = static_cast<i128>(t.fps) * t.base_den;
  const i128 tf = div_round_even(n, d);
  if (tf > (i128{1} << 61)) proto("the span is too long");
  t.t_frame_end_ns = static_cast<int64_t>(tf);
  t.t_start_ns = -2 * time::kWindowNs;
  t.t_end_ns = (time::window_index(t.t_frame_end_ns) + 1) * time::kWindowNs;
  return t;
}

Tree timeline_record(const StaticTimeline& t) {
  char hex[16];
  std::snprintf(hex, sizeof hex, "0x%08x", t.fps_base_bits);
  Tree r = Tree::dict();
  r["fps"] = Tree::integer(t.fps);
  r["fps_base_bits"] = Tree::string(hex);
  r["base_num"] = Tree::integer(t.base_num);
  r["base_den"] = Tree::integer(t.base_den);
  r["frame_ref"] = Tree::integer(t.frame_start);
  r["t_ref_ns"] = Tree::integer(0);
  r["frame_start"] = Tree::integer(t.frame_start);
  r["frame_end"] = Tree::integer(t.frame_end);
  r["t_frame_end_ns"] = Tree::integer(t.t_frame_end_ns);
  r["bundle_span"] = Tree::ints({t.t_start_ns, t.t_end_ns});
  r["run_span"] = Tree::ints({0, t.t_end_ns});
  r["run_span_extends_ns"] = Tree::integer(t.t_end_ns - t.t_frame_end_ns);
  return r;
}

std::vector<std::string> motion_needed(const ExchangeIn& ex, const Statics& s) {
  const auto n = static_cast<int32_t>(ex.objects.size());
  // Per object, what can move or reshape what hangs from it without a key on its own transform, and the nearest such
  // object up its parent, instancer and instanced-object links (memoized; one iterative walk, as convert_statics').
  std::vector<const char*> source(static_cast<size_t>(n), nullptr);
  for (int32_t i = 0; i < n; ++i) {
    const ObjectIn& ob = ex.objects[static_cast<size_t>(i)];
    if (ob.data_animation >= 0) {
      source[static_cast<size_t>(i)] = "has animated data";
    } else if (generates(ob)) {
      source[static_cast<size_t>(i)] = "has particles or geometry nodes";
    } else if (may_change_shape(ex, ob)) {
      source[static_cast<size_t>(i)] = "may change shape over the span";
    } else if (ob.type == "ARMATURE") {
      source[static_cast<size_t>(i)] = "is an armature (its bones' constraints are not exported)";
    }
  }
  constexpr int32_t kUnseen = -3, kOnWalk = -2;
  std::vector<int32_t> above(static_cast<size_t>(n), kUnseen);   // nearest ancestor with a source, -1: none
  const auto links = [&ex](int32_t i) {
    const ObjectIn& ob = ex.objects[static_cast<size_t>(i)];
    return std::array<int32_t, 3>{ob.parent, ob.instancer, ob.instance_object};
  };
  const auto valid = [n](int32_t j) { return j >= 0 && j < n; };
  const auto ancestor = [&](int32_t root) {
    std::vector<std::pair<int32_t, int>> walk{{root, 0}};
    above[static_cast<size_t>(root)] = kOnWalk;
    while (!walk.empty()) {
      const int32_t i = walk.back().first;
      const auto l = links(i);
      if (const int e = walk.back().second++; e < 3) {
        const int32_t j = l[static_cast<size_t>(e)];
        if (valid(j) && above[static_cast<size_t>(j)] == kUnseen) {
          above[static_cast<size_t>(j)] = kOnWalk;
          walk.push_back({j, 0});
        }
        continue;
      }
      int32_t found = -1;
      for (const int32_t j : l) {
        if (!valid(j) || found >= 0) continue;
        if (source[static_cast<size_t>(j)] != nullptr) found = j;
        else if (above[static_cast<size_t>(j)] >= 0) found = above[static_cast<size_t>(j)];
      }
      above[static_cast<size_t>(i)] = found;
      walk.pop_back();
    }
    return above[static_cast<size_t>(root)];
  };
  const auto up = [&](int32_t i) {
    return !valid(i) ? -1 : above[static_cast<size_t>(i)] == kUnseen ? ancestor(i) : above[static_cast<size_t>(i)];
  };
  const auto from_above = [&](int32_t i) -> std::string {
    const int32_t a = up(i);
    return a < 0 ? std::string() : "its ancestor " + ex.objects[static_cast<size_t>(a)].name_full + " " +
                                       source[static_cast<size_t>(a)];
  };

  std::vector<std::string> out;
  std::set<std::string> seen;
  const auto add = [&](std::string why) {
    if (seen.insert(why).second) out.push_back(std::move(why));
  };
  for (const BodyStatic& b : s.bodies) {
    if (b.role == BodyStatic::Role::kCarrier) {
      add(b.name + ": a light carrier");
      continue;
    }
    if (!b.T_world_static) {
      add(b.name + ": not static (animation, a constraint, a rigid body, or a parent or instancer that moves)");
      continue;
    }
    for (const int32_t p : b.parts) {
      if (!valid(p)) continue;
      const ObjectIn& ob = ex.objects[static_cast<size_t>(p)];
      if (may_change_shape(ex, ob)) add(b.name + ": its geometry may change over the span (rigidity probes)");
      if (ob.data_animation >= 0) add(b.name + ": its data is animated");
      if (const std::string why = from_above(p); !why.empty()) add(b.name + ": " + why);
    }
  }
  for (const CameraStatic& c : s.cameras) {
    if (!c.simulate) continue;
    if (!c.T_world_static) {
      add("camera " + c.cam.name + ": not static");
    } else if (const std::string why = from_above(c.object); !why.empty()) {
      add("camera " + c.cam.name + ": " + why);
    }
  }
  for (const LightStatic& l : s.lights) {
    if (l.light.body) {
      add("light " + l.name + ": rides a body");
    } else if (const std::string why = from_above(l.object); !why.empty()) {
      add("light " + l.name + ": " + why);
    }
  }
  return out;
}

render::Trajectory static_rig(const geo::M4& T_world_rig, const std::array<double, 3>& origin, int64_t t0,
                              int64_t t1) {
  if (t1 <= t0) throw std::invalid_argument("static_rig: empty span");
  render::Trajectory tr;
  tr.order = 6;
  tr.t0_ns = t0;
  tr.h_ns = kStaticKnotNs;
  const int64_t segments = std::max<int64_t>(1, (t1 - t0 + kStaticKnotNs - 1) / kStaticKnotNs);
  const auto [q, p] = geo::transform_to_pose(T_world_rig);
  for (int64_t k = 0; k < segments + tr.order - 1; ++k) {
    tr.rot.insert(tr.rot.end(), {q.w, q.x, q.y, q.z});
    tr.pos.insert(tr.pos.end(), {p.x - origin[0], p.y - origin[1], p.z - origin[2]});
  }
  tr.span_begin_ns = t0;
  tr.span_end_ns = t1;
  return tr;
}

}  // namespace balla::scene::blender
