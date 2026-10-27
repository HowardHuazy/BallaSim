// A JSON-like value with Python's scalar types (docs/dev/CPP_PORT.md item 5, E3): the bundle meta and the relight
// sidecar the C++ adapter builds. Dicts keep their insertion order, as Python's do; ints and floats stay apart (a
// bundle digest hashes json.dumps of the meta, where 1 and 1.0 differ).
#pragma once

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace balla::scene {

class Tree {
 public:
  enum class Kind { kNull, kBool, kInt, kFloat, kString, kList, kDict };

  Tree() = default;
  static Tree null() { return Tree(); }
  static Tree boolean(bool v) {
    Tree t(Kind::kBool);
    t.b_ = v;
    return t;
  }
  static Tree integer(int64_t v) {
    Tree t(Kind::kInt);
    t.i_ = v;
    return t;
  }
  static Tree real(double v) {
    Tree t(Kind::kFloat);
    t.f_ = v;
    return t;
  }
  static Tree string(std::string v) {
    Tree t(Kind::kString);
    t.s_ = std::move(v);
    return t;
  }
  static Tree list(std::vector<Tree> v = {}) {
    Tree t(Kind::kList);
    t.list_ = std::move(v);
    return t;
  }
  static Tree dict() { return Tree(Kind::kDict); }
  static Tree reals(const std::vector<double>& v) {
    Tree t = list();
    for (const double x : v) t.list_.push_back(real(x));
    return t;
  }
  static Tree ints(const std::vector<int64_t>& v) {
    Tree t = list();
    for (const int64_t x : v) t.list_.push_back(integer(x));
    return t;
  }

  Kind kind() const { return kind_; }
  bool as_bool() const { return b_; }
  int64_t as_int() const { return i_; }
  double as_float() const { return f_; }
  const std::string& as_string() const { return s_; }
  const std::vector<Tree>& items() const { return list_; }
  const std::vector<std::pair<std::string, Tree>>& entries() const { return dict_; }

  // dict[key] (inserted at the end when new).
  Tree& operator[](const std::string& key) {
    if (kind_ != Kind::kDict) throw std::logic_error("Tree: not a dict");
    for (auto& [k, v] : dict_) {
      if (k == key) return v;
    }
    dict_.emplace_back(key, Tree());
    return dict_.back().second;
  }
  const Tree* find(const std::string& key) const {
    for (const auto& [k, v] : dict_) {
      if (k == key) return &v;
    }
    return nullptr;
  }
  void push_back(Tree v) {
    if (kind_ != Kind::kList) throw std::logic_error("Tree: not a list");
    list_.push_back(std::move(v));
  }

 private:
  explicit Tree(Kind k) : kind_(k) {}
  Kind kind_ = Kind::kNull;
  bool b_ = false;
  int64_t i_ = 0;
  double f_ = 0.0;
  std::string s_;
  std::vector<Tree> list_;
  std::vector<std::pair<std::string, Tree>> dict_;
};

}  // namespace balla::scene
