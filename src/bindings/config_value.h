// Parameter values between Python and C++ (docs/dev/P0_PARAMS.md §1.4): plain Python values (None, bool, int,
// float, str and lists of them) to config::Value, typed by the parameter's row, and back. Header-only; shared
// by _core.config, _sensor and _engine (each module is its own binary).
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "config/resolve.h"
#include "core/config/param.h"

namespace balla::bindings {

namespace detail {
struct ToPy {
  nanobind::object operator()(std::monostate) const { return nanobind::none(); }
  nanobind::object operator()(bool b) const { return nanobind::bool_(b); }
  nanobind::object operator()(int64_t i) const { return nanobind::int_(i); }
  nanobind::object operator()(double d) const { return nanobind::float_(d); }
  nanobind::object operator()(const std::string& s) const { return nanobind::str(s.c_str(), s.size()); }
  template <class T>
  nanobind::object operator()(const std::vector<T>& l) const {
    nanobind::list out;
    for (const T& x : l) out.append((*this)(x));
    return out;
  }
};
}  // namespace detail

inline nanobind::object to_py(const config::Value& v) { return std::visit(detail::ToPy{}, v); }

// A Python or NumPy bool. NumPy's bool scalar is no subclass of bool: its type is numpy's "bool" (NumPy 2) or
// "bool_" (NumPy 1). Another name only means the value is refused as a bool (and as a number).
inline bool is_bool(nanobind::handle x) {
  namespace nb = nanobind;
  if (PyBool_Check(x.ptr())) return true;
  if (PyLong_Check(x.ptr()) || PyFloat_Check(x.ptr()) || PyUnicode_Check(x.ptr())) return false;
  const nb::handle type(reinterpret_cast<PyObject*>(Py_TYPE(x.ptr())));
  const nb::object module = nb::getattr(type, "__module__", nb::none());
  if (!nb::isinstance<nb::str>(module) || nb::cast<std::string>(module) != "numpy") return false;
  const auto name = nb::cast<std::string>(nb::getattr(type, "__name__"));
  return name == "bool" || name == "bool_";
}

// One Python value as the row's type (an int is a float; a float is never an int; bools are never numbers);
// std::invalid_argument (ValueError) naming the row otherwise. Ranges are not checked here.
inline config::Value from_py(nanobind::handle h, const config::Param& p) {
  namespace nb = nanobind;
  using config::Type;
  const auto fail = [&](const char* expected) -> config::Value {
    throw std::invalid_argument(config::display_name(p) + " = " + nb::cast<std::string>(nb::repr(h)) +
                                ": expected " + expected);
  };
  const auto is_int = [&](nb::handle x) {
    int64_t i;
    return !is_bool(x) && PyFloat_Check(x.ptr()) == 0 && nb::try_cast<int64_t>(x, i);
  };
  const auto is_number = [&](nb::handle x) {
    double d;
    return !is_bool(x) && !nb::isinstance<nb::str>(x) && nb::try_cast<double>(x, d);
  };
  if (h.is_none()) {
    if (!p.nullable) return fail("a value (not None)");
    return config::Value{};
  }
  switch (p.type) {
    case Type::kBool: {
      if (!is_bool(h)) return fail("True or False");
      const int truth = PyObject_IsTrue(h.ptr());
      if (truth < 0) throw nb::python_error();
      return truth != 0;
    }
    case Type::kInt:
      if (!is_int(h)) return fail("an integer");
      return nb::cast<int64_t>(h);
    case Type::kFloat:
      if (!is_number(h)) return fail("a number");
      return nb::cast<double>(h);
    case Type::kString:
    case Type::kEnum:
    case Type::kPath:
      if (!nb::isinstance<nb::str>(h)) return fail("a string");
      return nb::cast<std::string>(h);
    case Type::kFloatList:
    case Type::kIntList: {
      if (!PyList_Check(h.ptr()) && !PyTuple_Check(h.ptr())) return fail("a list of numbers");
      std::vector<double> xs;
      std::vector<int64_t> is;
      for (nb::handle x : h) {
        if (p.type == Type::kIntList ? !is_int(x) : !is_number(x)) {
          return fail(p.type == Type::kIntList ? "a list of integers" : "a list of numbers");
        }
        if (p.type == Type::kIntList) is.push_back(nb::cast<int64_t>(x));
        else xs.push_back(nb::cast<double>(x));
      }
      if (p.type == Type::kIntList) return is;
      return xs;
    }
    case Type::kStringList: {
      if (!PyList_Check(h.ptr()) && !PyTuple_Check(h.ptr())) return fail("a list of strings");
      std::vector<std::string> out;
      for (nb::handle x : h) {
        if (!nb::isinstance<nb::str>(x)) return fail("a list of strings");
        out.push_back(nb::cast<std::string>(x));
      }
      return out;
    }
  }
  return fail("a known type");
}

// Resolved values ({run: {key: value}, cameras: {name: {key: value}}}, Resolved.values()) typed by the rows of
// `tables` (keys without a row there are skipped): what config::apply and its parts take.
inline config::Values values_from_py(nanobind::handle values,
                                     std::initializer_list<const std::vector<config::Param>*> tables) {
  namespace nb = nanobind;
  std::map<std::string, const config::Param*, std::less<>> rows;
  for (const auto* list : tables) {
    for (const config::Param& p : *list) rows.emplace(p.key, &p);
  }
  const auto typed = [&](nb::handle d, const char* what) {
    if (!nb::isinstance<nb::dict>(d)) throw std::invalid_argument(std::string("values: ") + what + " must be a dict");
    config::ValueMap out;
    for (auto [k, v] : nb::cast<nb::dict>(d)) {
      const auto key = nb::cast<std::string>(k);
      if (const auto it = rows.find(key); it != rows.end()) out.emplace(key, from_py(v, *it->second));
    }
    return out;
  };
  if (!nb::isinstance<nb::dict>(values) || !nb::cast<nb::dict>(values).contains("run") ||
      !nb::cast<nb::dict>(values).contains("cameras") || !nb::isinstance<nb::dict>(values["cameras"])) {
    throw std::invalid_argument("values: expected {run: {...}, cameras: {name: {...}}} (Resolved.values())");
  }
  config::Values out;
  out.run = typed(values["run"], "run");
  for (auto [name, v] : nb::cast<nb::dict>(values["cameras"])) {
    const auto cam = nb::cast<std::string>(name);
    try {
      out.cameras.emplace_back(cam, typed(v, "a camera's values"));
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument("cameras." + cam + ": " + e.what());
    }
  }
  return out;
}

// A plain Python value (a YAML scalar, a list of numbers or of strings, or a dict) as a config::Node; leaves are not
// typed yet (the resolver types them by their rows): None, bool, int, float, str; a list of ints is an int list, of
// numbers a float list, of strings a string list, and an empty list an empty float list. `where` names the value.
inline config::Node node_from_py(nanobind::handle h, const std::string& where) {
  namespace nb = nanobind;
  config::Node n;
  const auto fail = [&](const char* why) -> config::Node {
    throw std::invalid_argument(where + " = " + nb::cast<std::string>(nb::repr(h)) + ": " + why);
  };
  const auto is_int = [](nb::handle x) {
    int64_t i;
    return !is_bool(x) && PyFloat_Check(x.ptr()) == 0 && !nb::isinstance<nb::str>(x) && nb::try_cast<int64_t>(x, i);
  };
  const auto is_number = [](nb::handle x) {
    double d;
    return !is_bool(x) && !nb::isinstance<nb::str>(x) && nb::try_cast<double>(x, d);
  };
  if (nb::isinstance<nb::dict>(h)) {
    n.is_map = true;
    for (auto [k, v] : nb::cast<nb::dict>(h)) {
      if (!nb::isinstance<nb::str>(k)) fail("mapping keys must be strings");
      const auto key = nb::cast<std::string>(k);
      n.map.emplace_back(key, node_from_py(v, where.empty() ? key : where + "." + key));
    }
  } else if (h.is_none()) {
  } else if (is_bool(h)) {
    const int truth = PyObject_IsTrue(h.ptr());
    if (truth < 0) throw nb::python_error();
    n.leaf = truth != 0;
  } else if (nb::isinstance<nb::str>(h)) {
    n.leaf = nb::cast<std::string>(h);
  } else if (is_int(h)) {
    n.leaf = nb::cast<int64_t>(h);
  } else if (is_number(h)) {
    n.leaf = nb::cast<double>(h);
  } else if (PyList_Check(h.ptr()) || PyTuple_Check(h.ptr())) {
    bool ints = true, numbers = true, strings = true;
    for (nb::handle x : h) {
      ints = ints && is_int(x);
      numbers = numbers && is_number(x);
      strings = strings && nb::isinstance<nb::str>(x);
    }
    if (nb::len(h) == 0) n.leaf = std::vector<double>{};
    else if (ints) n.leaf = nb::cast<std::vector<int64_t>>(h);
    else if (numbers) n.leaf = nb::cast<std::vector<double>>(h);
    else if (strings) n.leaf = nb::cast<std::vector<std::string>>(h);
    else fail("expected a list of numbers or of strings");
  } else {
    fail("expected None, a bool, a number, a string, a list or a mapping");
  }
  return n;
}

}  // namespace balla::bindings
