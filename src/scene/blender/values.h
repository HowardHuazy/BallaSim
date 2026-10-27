// Typed reads of exchange records (RNA values as JSON, blender.md §3.4) for the statics converter: a missing or null
// value takes the given fallback (Blender's default), a value of the wrong type is BRG-PROTO (std::invalid_argument).
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "scene/blender/exchange_in.h"
#include "scene/tree.h"

namespace balla::scene::blender::detail {

inline const Tree* at(const Tree& t, const char* key) {
  if (t.kind() != Tree::Kind::kDict) return nullptr;
  const Tree* v = t.find(key);
  return v == nullptr || v->kind() == Tree::Kind::kNull ? nullptr : v;
}

[[noreturn]] inline void wrong(const std::string& where, const char* key, const char* what) {
  throw std::invalid_argument("BRG-PROTO: " + where + "." + key + " is not " + what);
}

inline double number(const Tree& t, const char* key, double fallback, const std::string& where) {
  const Tree* v = at(t, key);
  if (v == nullptr) return fallback;
  if (v->kind() == Tree::Kind::kInt) return static_cast<double>(v->as_int());
  if (v->kind() != Tree::Kind::kFloat || !std::isfinite(v->as_float())) wrong(where, key, "a finite number");
  return v->as_float();
}

inline bool boolean(const Tree& t, const char* key, bool fallback, const std::string& where) {
  const Tree* v = at(t, key);
  if (v == nullptr) return fallback;
  if (v->kind() != Tree::Kind::kBool) wrong(where, key, "a boolean");
  return v->as_bool();
}

inline std::string text(const Tree& t, const char* key, const std::string& fallback, const std::string& where) {
  const Tree* v = at(t, key);
  if (v == nullptr) return fallback;
  if (v->kind() != Tree::Kind::kString) wrong(where, key, "a string");
  return v->as_string();
}

// A list of finite numbers (any length when n < 0, else exactly n or, for colours, 3 or 4 when n == 3).
inline std::vector<double> numbers(const Tree& t, const char* key, int n, const std::string& where) {
  const Tree* v = at(t, key);
  if (v == nullptr) return {};
  if (v->kind() != Tree::Kind::kList) wrong(where, key, "a list of numbers");
  std::vector<double> out;
  for (const Tree& x : v->items()) {
    if (x.kind() == Tree::Kind::kInt) {
      out.push_back(static_cast<double>(x.as_int()));
    } else if (x.kind() == Tree::Kind::kFloat && std::isfinite(x.as_float())) {
      out.push_back(x.as_float());
    } else {
      wrong(where, key, "a list of finite numbers");
    }
  }
  const auto size = static_cast<int>(out.size());
  if (n >= 0 && size != n && !(n == 3 && size == 4)) wrong(where, key, "a list of the right length");
  return out;
}

// An RGB(A) value's first three components, `fallback` when missing or null.
inline std::array<double, 3> rgb(const Tree& t, const char* key, const std::string& where,
                                 const std::array<double, 3>& fallback = {1.0, 1.0, 1.0}) {
  const std::vector<double> v = numbers(t, key, 3, where);
  if (v.empty()) return fallback;
  return {v[0], v[1], v[2]};
}

// The node a node-tree data path addresses, `nodes["<name>"]...` with Blender's escapes (\" and \\) undone; "" for a
// path that addresses no node (probe m3b: `nodes["Emission"].inputs[1].default_value`).
inline std::string path_node(const std::string& path) {
  static const std::string kPrefix = "nodes[\"";
  if (path.rfind(kPrefix, 0) != 0) return "";
  std::string name;
  for (size_t i = kPrefix.size(); i < path.size(); ++i) {
    if (path[i] == '\\' && i + 1 < path.size()) {
      name.push_back(path[++i]);
    } else if (path[i] == '"') {
      return name;
    } else {
      name.push_back(path[i]);
    }
  }
  throw std::invalid_argument("BRG-PROTO: an unterminated node path " + path);
}

// The data paths an animations[] record animates (its F-curves' and drivers' paths), and "NLA" when an unmuted NLA
// track holds a strip (a strip's action is another record, so its paths are not followed: the strip itself counts).
inline std::vector<std::string> animated_paths(const ExchangeIn& ex, int32_t a) {
  std::vector<std::string> out;
  if (a < 0) return out;
  const std::vector<Tree>& all = ex.animations.items();
  if (static_cast<size_t>(a) >= all.size()) throw std::invalid_argument("BRG-PROTO: an animation index out of range");
  const Tree& rec = all[static_cast<size_t>(a)];
  const std::string where = "animations[" + std::to_string(a) + "]";
  for (const char* list : {"fcurves", "drivers"}) {
    const Tree* l = at(rec, list);
    if (l == nullptr) continue;
    if (l->kind() != Tree::Kind::kList) wrong(where, list, "a list");
    for (const Tree& f : l->items()) out.push_back(text(f, "data_path", "", where + "." + list));
  }
  if (const Tree* nla = at(rec, "nla"); nla != nullptr && nla->kind() == Tree::Kind::kList) {
    for (const Tree& track : nla->items()) {
      const Tree* strips = at(track, "strips");
      if (!boolean(track, "mute", false, where + ".nla") && strips != nullptr && strips->kind() == Tree::Kind::kList &&
          !strips->items().empty()) {
        out.push_back("NLA");
        break;
      }
    }
  }
  return out;
}

}  // namespace balla::scene::blender::detail
