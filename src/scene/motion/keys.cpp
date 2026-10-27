// Declared keys (keys.h; blender.md §4.7.4).
#include "scene/motion/keys.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string_view>
#include <utility>

namespace balla::scene::motion {

namespace {

constexpr double kPlateau = 1e-4;            // Blender's key plateau in the curve's own frames (MF E8)
constexpr size_t kMaxFeatures = size_t{1} << 22;
constexpr double kBounce[3] = {1.0 / 2.75, 2.0 / 2.75, 2.5 / 2.75};  // Penner's bounce (ease out) breaks

Continuity lower(Continuity a, Continuity b) { return static_cast<int8_t>(a) < static_cast<int8_t>(b) ? a : b; }

[[noreturn]] void proto(const KeyCurve& c, const std::string& what) {
  throw MotionError("BRG-PROTO", "F-curve " + c.data_path + "[" + std::to_string(c.index) + "]: " + what);
}

const Tree* field(const KeyModifier& m, const char* key) {
  return m.fields.kind() == Tree::Kind::kDict ? m.fields.find(key) : nullptr;
}

double num_field(const KeyModifier& m, const char* key, double def) {
  const Tree* v = field(m, key);
  if (!v) return def;
  if (v->kind() == Tree::Kind::kFloat) return v->as_float();
  if (v->kind() == Tree::Kind::kInt) return static_cast<double>(v->as_int());
  if (v->kind() == Tree::Kind::kBool) return v->as_bool() ? 1.0 : 0.0;
  throw MotionError("BRG-PROTO", "F-modifier " + m.type + ": " + key + " is not a number");
}

std::string str_field(const KeyModifier& m, const char* key, const char* def) {
  const Tree* v = field(m, key);
  if (!v) return def;
  if (v->kind() != Tree::Kind::kString) throw MotionError("BRG-PROTO", "F-modifier " + m.type + ": " + key +
                                                                           " is not a string");
  return v->as_string();
}

class Features {
 public:
  void add(double frame, Continuity hint, bool plateau) {
    if (f_.size() >= kMaxFeatures) throw MotionError("MOT-SOURCE", "more than 4M declared breaks on one curve");
    f_.push_back({frame, hint, plateau});
  }
  // ascending, equal frames merged (the lowest class, any plateau), within [lo, hi]
  std::vector<KeyFeature> take(double lo, double hi) && {
    std::stable_sort(f_.begin(), f_.end(), [](const KeyFeature& a, const KeyFeature& b) { return a.frame < b.frame; });
    std::vector<KeyFeature> out;
    for (const KeyFeature& k : f_) {
      if (!(k.frame >= lo && k.frame <= hi)) continue;
      if (!out.empty() && out.back().frame == k.frame) {
        out.back().hint = lower(out.back().hint, k.hint);
        out.back().plateau = out.back().plateau || k.plateau;
        continue;
      }
      out.push_back(k);
    }
    return out;
  }

 private:
  std::vector<KeyFeature> f_;
};

// The class a BEZIER segment gives a key on one side: a VECTOR or FREE handle, or non-collinear handles, kink.
Continuity bezier_side(int handle, bool collinear) {
  if (handle == static_cast<int>(Handle::kVector) || handle == static_cast<int>(Handle::kFree) || !collinear) {
    return Continuity::kC0;
  }
  return Continuity::kC1;
}

}  // namespace

std::vector<KeyFeature> curve_features(const KeyCurve& c, double lo, double hi, bool lenient) {
  if (c.keys.size() % 7 != 0) proto(c, "the key table is not n x 7");
  const size_t n = c.keys.size() / 7;
  if (c.extrapolation != 0 && c.extrapolation != 1) proto(c, "extrapolation code " + std::to_string(c.extrapolation));
  auto row = [&](size_t i, size_t j) { return c.keys[7 * i + j]; };
  auto code = [&](size_t i, size_t j, int max) {
    const double v = row(i, j);
    if (!(v >= 0.0 && v <= max && v == std::floor(v))) proto(c, "key " + std::to_string(i) + ": bad code");
    return static_cast<int>(v);
  };
  std::vector<double> f(n), v(n);
  std::vector<int> ip(n), ez(n), hl(n), hr(n);
  std::vector<bool> col(n);
  for (size_t i = 0; i < n; ++i) {
    f[i] = row(i, 0);
    v[i] = row(i, 1);
    if (!std::isfinite(f[i]) || !std::isfinite(v[i])) proto(c, "key " + std::to_string(i) + " is not finite");
    if (i > 0 && f[i] < f[i - 1]) proto(c, "keys are not in time order");
    ip[i] = code(i, 2, static_cast<int>(Interp::kElastic));
    ez[i] = code(i, 3, static_cast<int>(Easing::kEaseInOut));
    hl[i] = code(i, 4, static_cast<int>(Handle::kAutoClamped));
    hr[i] = code(i, 5, static_cast<int>(Handle::kAutoClamped));
    col[i] = code(i, 6, 1) == 1;
  }
  // the modifiers: their own breaks, and whether a Cycles modifier repeats the curve on either side
  const KeyModifier* cycles = nullptr;
  for (const KeyModifier& m : c.modifiers) {
    if (m.mute) continue;
    static const std::set<std::string_view> known{"STEPPED", "CYCLES", "NOISE", "GENERATOR", "FNGENERATOR", "LIMITS"};
    if (!known.count(m.type)) {
      if (lenient) continue;
      throw MotionError("MOT-SOURCE", "F-curve " + c.data_path + "[" + std::to_string(c.index) + "]: F-modifier " +
                                          m.type + " is not supported (lenient: detection only)");
    }
    if (m.type == "CYCLES" && !cycles && n >= 2 && f[n - 1] > f[0]) cycles = &m;
  }
  const std::string before = cycles ? str_field(*cycles, "mode_before", "REPEAT") : "NONE";
  const std::string after = cycles ? str_field(*cycles, "mode_after", "REPEAT") : "NONE";
  for (const std::string* mode : {&before, &after}) {
    if (*mode != "NONE" && *mode != "REPEAT" && *mode != "REPEAT_OFFSET" && *mode != "MIRROR") {
      throw MotionError("BRG-PROTO", "Cycles modifier mode " + *mode);
    }
  }

  // one period of the curve: the class at each key from the segments (or the extrapolation) on both sides
  Features base;
  auto segment_side = [&](size_t seg, bool at_start) -> Continuity {  // the class segment `seg` gives one of its ends
    const int s = ip[seg];
    if (s == static_cast<int>(Interp::kBezier)) return at_start ? bezier_side(hr[seg], col[seg])
                                                                : bezier_side(hl[seg + 1], col[seg + 1]);
    return Continuity::kC0;  // CONSTANT, LINEAR, easing: kink at both ends
  };
  auto extrapolated = [&](size_t seg, bool cycled) -> Continuity {  // the class an end key gets from outside the keys
    if (cycled) return Continuity::kC0;  // a junction of the repeated curve (a step there is added below)
    const int s = ip[seg];
    if (c.extrapolation == 1 && (s == static_cast<int>(Interp::kLinear) || s == static_cast<int>(Interp::kBezier))) {
      return Continuity::kC1;  // extended along the end segment's slope or handle
    }
    return Continuity::kC0;
  };
  if (n >= 2) {
    for (size_t i = 0; i < n; ++i) {
      // an end key compares its segment with what lies outside the keys; an inner key its two segments
      Continuity cls;
      if (i == 0) {
        cls = extrapolated(0, before != "NONE");
      } else if (i == n - 1) {
        cls = extrapolated(n - 2, after != "NONE");
      } else {
        cls = lower(segment_side(i - 1, false), segment_side(i, true));
      }
      base.add(f[i], cls, true);
      if (i > 0 && ip[i - 1] == static_cast<int>(Interp::kConstant)) base.add(f[i], Continuity::kStep, true);
    }
    for (size_t i = 0; i + 1 < n; ++i) {
      if (ip[i] != static_cast<int>(Interp::kBounce)) continue;
      const double a = f[i], len = f[i + 1] - f[i];
      const int e = ez[i] == static_cast<int>(Easing::kAuto) ? static_cast<int>(Easing::kEaseOut) : ez[i];
      for (const double b : kBounce) {
        if (e == static_cast<int>(Easing::kEaseOut)) base.add(a + b * len, Continuity::kC0, false);
        if (e == static_cast<int>(Easing::kEaseIn)) base.add(a + (1.0 - b) * len, Continuity::kC0, false);
        if (e == static_cast<int>(Easing::kEaseInOut)) {
          base.add(a + 0.5 * (1.0 - b) * len, Continuity::kC0, false);
          base.add(a + 0.5 * (1.0 + b) * len, Continuity::kC0, false);
        }
      }
    }
  }
  Features out;
  const std::vector<KeyFeature> period = std::move(base).take(f.empty() ? 0.0 : f[0], f.empty() ? 0.0 : f[n - 1]);
  for (const KeyFeature& k : period) out.add(k.frame, k.hint, k.plateau);
  if (cycles) {  // the curve repeated with the period of its key range (mirrored on odd cycles for MIRROR)
    const double f0 = f[0], f1 = f[n - 1], per = f1 - f0;
    const Continuity junction_repeat = v[0] == v[n - 1] ? Continuity::kC0 : Continuity::kStep;
    auto repeat_side = [&](const std::string& mode, double count, int dir) {
      if (mode == "NONE") return;
      const double reach = dir > 0 ? (hi - f1) / per : (f0 - lo) / per;
      if (!(reach > 0.0)) return;
      double cycles_n = std::ceil(reach) + 1.0;
      if (count > 0.0) cycles_n = std::min(cycles_n, count);
      if (cycles_n * static_cast<double>(period.size() + 1) > static_cast<double>(kMaxFeatures)) {
        throw MotionError("MOT-SOURCE", "F-curve " + c.data_path + ": a Cycles modifier repeats too often");
      }
      for (int64_t k = 1; k <= static_cast<int64_t>(cycles_n); ++k) {
        const int64_t cyc = dir * k;
        const bool mirror = mode == "MIRROR" && (k % 2) == 1;
        for (const KeyFeature& g : period) {
          const double fr = mirror ? f0 + static_cast<double>(cyc) * per + f1 - g.frame
                                   : g.frame + static_cast<double>(cyc) * per;
          out.add(fr, g.hint, g.plateau);
        }
        // the junction with the previous cycle: a step when REPEAT jumps from the last key's value to the first's
        const double jt = dir > 0 ? f0 + static_cast<double>(cyc) * per : f0 + static_cast<double>(cyc + 1) * per;
        out.add(jt, mode == "REPEAT" ? junction_repeat : Continuity::kC0, true);
      }
    };
    repeat_side(before, num_field(*cycles, "cycles_before", 0.0), -1);
    repeat_side(after, num_field(*cycles, "cycles_after", 0.0), +1);
  }
  for (const KeyModifier& m : c.modifiers) {
    if (m.mute) continue;
    if (m.type == "STEPPED") {  // steps exactly at offset + k step, no plateau (BL-7 probe)
      const double step = num_field(m, "frame_step", 2.0), off = num_field(m, "frame_offset", 0.0);
      if (!(step > 0.0) || !std::isfinite(off)) throw MotionError("BRG-PROTO", "STEPPED modifier: frame_step <= 0");
      double a = lo, b = hi;
      if (num_field(m, "use_frame_start", 0.0) != 0.0) a = std::max(a, num_field(m, "frame_start", lo));
      if (num_field(m, "use_frame_end", 0.0) != 0.0) b = std::min(b, num_field(m, "frame_end", hi));
      if ((b - a) / step > static_cast<double>(kMaxFeatures)) {
        throw MotionError("MOT-SOURCE", "F-curve " + c.data_path + ": a STEPPED modifier steps too often");
      }
      for (double k = std::ceil((a - off) / step); off + k * step <= b; k += 1.0) {
        out.add(off + k * step, Continuity::kStep, false);
      }
    }
    if (m.use_restricted_range) {  // the modifier switches on and off (ramps with blends)
      out.add(m.frame_start, m.blend_in > 0.0 ? Continuity::kC0 : Continuity::kStep, false);
      out.add(m.frame_end, m.blend_out > 0.0 ? Continuity::kC0 : Continuity::kStep, false);
      if (m.blend_in > 0.0) out.add(m.frame_start + m.blend_in, Continuity::kC0, false);
      if (m.blend_out > 0.0) out.add(m.frame_end - m.blend_out, Continuity::kC0, false);
    }
  }
  return std::move(out).take(lo, hi);
}

bool moves_object(const std::string& p) {
  static const std::set<std::string_view> exact{"location",       "rotation_euler",       "rotation_quaternion",
                                                "rotation_axis_angle", "scale",           "delta_location",
                                                "delta_rotation_euler", "delta_rotation_quaternion", "delta_scale"};
  if (exact.count(p)) return true;
  return p.rfind("pose.bones[", 0) == 0 || p.rfind("constraints[", 0) == 0;
}

namespace {

struct Out {
  const TimeMap& time;
  int64_t t0, t1;
  std::vector<DeclaredKey> keys;
  // a break at scene time `ctime` (frames) under a strip of scale n_s
  void add(double ctime, Continuity hint, bool plateau, double n_s) {
    const double fl = time.framelen();
    const int64_t t = time.frame_to_ns(ctime / fl);
    if (t < t0 || t > t1) return;
    keys.push_back({t, hint, plateau, plateau ? kPlateau * n_s / fl : 0.0});
  }
};

const KeyAnimation* strip_action(const KeyScene& scene, const KeyStrip& s) {
  for (const KeyAnimation& a : scene.animations) {
    if (a.action == s.action && a.slot == s.slot) return &a;
  }
  return nullptr;
}

void add_animation(const KeyScene& scene, const KeyAnimation& anim, bool object_level, double c_lo, double c_hi,
                   bool lenient, Out& out) {
  auto relevant = [&](const KeyCurve& c) { return !object_level || moves_object(c.data_path); };
  for (const KeyCurve& c : anim.fcurves) {  // the active action, in scene time
    if (!relevant(c)) continue;
    for (const KeyFeature& k : curve_features(c, c_lo, c_hi, lenient)) out.add(k.frame, k.hint, k.plateau, 1.0);
  }
  // Strip properties animated through the owner (blend_in, blend_out, frame_start, scale, repeat, ...) move the
  // strips' timing and blends: an F-curve "animation_data.nla_tracks[...]" in its active action (BL-7 probe p3) or in
  // any of its strips' actions, pushed down above or below, muted or not (muting writes the last or another value,
  // p5), or a driver on such a path (p6). The exchange's strip records do not say which strip or property, so no strip
  // of this animation declares anything: detection finds their breaks (§4.7.4). A strip action missing from the
  // exchange is skipped here; the loop below refuses it when the strip is evaluated.
  auto strip_path = [](const std::string& p) { return p.rfind("animation_data.nla_tracks[", 0) == 0; };
  auto keys_strips = [&](const KeyAnimation& a) {
    return std::any_of(a.fcurves.begin(), a.fcurves.end(), [&](const KeyCurve& c) { return strip_path(c.data_path); });
  };
  bool keyed_strips =
      keys_strips(anim) || std::any_of(anim.driver_paths.begin(), anim.driver_paths.end(), strip_path);
  for (const KeyTrack& track : anim.nla) {
    for (const KeyStrip& s : track.strips) {
      if (keyed_strips) break;
      if (const KeyAnimation* a = strip_action(scene, s); a && keys_strips(*a)) keyed_strips = true;
    }
  }
  if (keyed_strips) return;
  const bool solo = std::any_of(anim.nla.begin(), anim.nla.end(), [](const KeyTrack& t) { return t.is_solo; });
  for (const KeyTrack& track : anim.nla) {
    if (track.mute || (solo && !track.is_solo)) continue;
    for (const KeyStrip& s : track.strips) {
      if (s.mute) continue;
      const double len = s.action_frame_end - s.action_frame_start;
      if (!(std::isfinite(s.frame_start) && std::isfinite(s.frame_end) && s.frame_end >= s.frame_start &&
            s.scale > 0.0 && s.repeat > 0.0 && std::isfinite(s.scale) && std::isfinite(s.repeat) && len >= 0.0 &&
            s.extrapolation >= 0 && s.extrapolation <= 2 && s.blend_in >= 0.0 && s.blend_out >= 0.0)) {
        throw MotionError("BRG-PROTO", "NLA strip of action " + s.action + ": invalid timing");
      }
      const double fs = s.frame_start, fe = s.frame_end;
      auto mapped = [&](double frame) { return s.use_reverse ? fs + fe - frame : frame; };
      // the strip's own breaks (scene time, no plateau). An animated influence replaces the blend ramps, and an
      // animated strip time gives the action frame itself, so the repeat no longer wraps (BL-7 probe p4).
      const bool held_before = s.extrapolation == 1, held_after = s.extrapolation >= 1;
      const double blend_in = s.use_animated_influence ? 0.0 : s.blend_in;
      const double blend_out = s.use_animated_influence ? 0.0 : s.blend_out;
      out.add(fs, blend_in > 0.0 || held_before ? Continuity::kC0 : Continuity::kStep, false, 1.0);
      out.add(fe, blend_out > 0.0 || held_after ? Continuity::kC0 : Continuity::kStep, false, 1.0);
      if (blend_in > 0.0) out.add(fs + blend_in, Continuity::kC0, false, 1.0);
      if (blend_out > 0.0) out.add(fe - blend_out, Continuity::kC0, false, 1.0);
      if (s.use_animated_time) continue;  // neither repeats nor keys are mapped: detection finds them
      // Repeat r starts at fs + r len scale (forward strip time). Blender keeps frame_end at fs + len scale repeat; the
      // exchange is not trusted, so only the repeats r < repeat that start within the strip and meet the span
      // [c_lo, c_hi] are visited, at most 4M (MOT-SOURCE), bounded in doubles before any cast. len 0: one.
      int64_t r0 = 0, r1 = 0;  // inclusive
      if (const double span = len * s.scale; span > 0.0) {
        const double lo_f = s.use_reverse ? fs + fe - c_hi : c_lo, hi_f = s.use_reverse ? fs + fe - c_lo : c_hi;
        const double a = std::max(0.0, std::floor((lo_f - fs) / span) - 1.0);
        const double b =
            std::min({std::ceil(s.repeat) - 1.0, std::floor((fe - fs) / span), std::ceil((hi_f - fs) / span) + 1.0});
        if (!(b >= a)) {
          r1 = -1;
        } else if (!(b - a < static_cast<double>(kMaxFeatures) && b < 0x1p62)) {
          throw MotionError("MOT-SOURCE", "NLA strip of action " + s.action + ": repeats too often within the span");
        } else {
          r0 = static_cast<int64_t>(a);
          r1 = static_cast<int64_t>(b);
        }
      }
      if (len > 0.0) {
        for (int64_t r = std::max<int64_t>(r0, 1); r <= r1; ++r) {
          const double b = fs + static_cast<double>(r) * len * s.scale;
          if (b < fe) out.add(mapped(b), Continuity::kC0, false, 1.0);
        }
      }
      if (s.use_animated_influence) continue;  // the keys are not mapped: detection finds them
      const KeyAnimation* act = strip_action(scene, s);
      if (!act) {
        throw MotionError("BRG-PROTO", "the exchange holds no animation for strip action " + s.action +
                                           (s.slot.empty() ? "" : " (slot " + s.slot + ")"));
      }
      const double tol = 1e-9 * std::max(1.0, std::fabs(fe));
      for (const KeyCurve& c : act->fcurves) {
        if (!relevant(c)) continue;
        for (const KeyFeature& k : curve_features(c, s.action_frame_start, s.action_frame_end, lenient)) {
          for (int64_t r = r0; r <= r1; ++r) {
            const double fwd =
                fs + (k.frame - s.action_frame_start) * s.scale + static_cast<double>(r) * len * s.scale;
            if (fwd < fs - tol || fwd > fe + tol) continue;
            out.add(mapped(fwd), k.hint, k.plateau, s.scale);
          }
        }
      }
    }
  }
}

}  // namespace

std::vector<DeclaredKey> declared_keys(const KeyScene& scene, int32_t object, const TimeMap& time, int64_t t0,
                                       int64_t t1, bool lenient) {
  const auto n_obj = static_cast<int32_t>(scene.objects.size());
  const auto n_anim = static_cast<int32_t>(scene.animations.size());
  if (object < 0 || object >= n_obj) throw std::invalid_argument("declared_keys: no such object");
  if (t1 < t0) throw std::invalid_argument("declared_keys: empty span");
  auto check_obj = [&](int32_t o) {
    if (o < 0 || o >= n_obj) throw MotionError("BRG-PROTO", "an object index " + std::to_string(o) + " is out of range");
  };
  // the chain: the object and its parents; each one's constraint targets, one level deep
  std::map<int32_t, bool> anims;  // animations[] index -> object level (only moving curves count)
  auto use = [&](int32_t a, bool object_level) {
    if (a < 0 || a >= n_anim) throw MotionError("BRG-PROTO", "an animation index " + std::to_string(a) + " is out of range");
    auto [it, fresh] = anims.emplace(a, object_level);
    if (!fresh) it->second = it->second && object_level;
  };
  std::set<int32_t> seen;
  for (int32_t o = object; o >= 0; o = scene.objects[static_cast<size_t>(o)].parent) {
    check_obj(o);
    if (!seen.insert(o).second) throw MotionError("BRG-PROTO", "the parent chain of object " +
                                                                    std::to_string(object) + " loops");
    const KeyObject& ob = scene.objects[static_cast<size_t>(o)];
    for (const int32_t a : ob.animations) use(a, true);
    for (const int32_t tg : ob.constraint_targets) {
      check_obj(tg);
      const KeyObject& t = scene.objects[static_cast<size_t>(tg)];
      for (const int32_t a : t.animations) use(a, true);
      for (const int32_t a : t.data_animations) use(a, false);
    }
  }
  // the span in scene time (frames), with a frame of margin; breaks outside [t0, t1] are dropped by time
  const double fl = time.framelen(), period = time.period_ns();
  const double c_lo = (time.frame_ref() + static_cast<double>(t0 - time.t_ref_ns()) / period - 1.0) * fl;
  const double c_hi = (time.frame_ref() + static_cast<double>(t1 - time.t_ref_ns()) / period + 1.0) * fl;
  Out out{time, t0, t1, {}};
  for (const auto& [a, object_level] : anims) {
    add_animation(scene, scene.animations[static_cast<size_t>(a)], object_level, c_lo, c_hi, lenient, out);
  }
  std::vector<DeclaredKey>& k = out.keys;
  std::stable_sort(k.begin(), k.end(), [](const DeclaredKey& a, const DeclaredKey& b) { return a.t_ns < b.t_ns; });
  std::vector<DeclaredKey> merged;
  for (const DeclaredKey& d : k) {
    if (!merged.empty() && merged.back().t_ns == d.t_ns) {
      DeclaredKey& m = merged.back();
      m.hint = lower(m.hint, d.hint);
      if (d.plateau) {
        m.plateau_frames = m.plateau ? std::max(m.plateau_frames, d.plateau_frames) : d.plateau_frames;
        m.plateau = true;
      }
      continue;
    }
    merged.push_back(d);
  }
  return merged;
}

}  // namespace balla::scene::motion
