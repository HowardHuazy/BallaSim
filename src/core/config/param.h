// The parameter descriptor (docs/dev/P0_PARAMS.md §1.1): one row per tunable, with its meaning, unit,
// ranges, status, source, scope, relevance and the pipeline stage it invalidates, and, for rows with an
// apply target, the setter and getter of one member of an existing aggregate (§1.3).
//
// A member row is written once, next to its struct: member<&S::m>(key, {...}) deduces the type, the
// setter, the getter and, for a literal row, the default (S{}.*m, so the member initializer is the single
// source). A custom row converts between the user's unit and the engine's (custom<S, set, get>). A row
// without a target is consumed by the resolver or by Python. Header-only; host code only.
//
// Nothing is defaulted silently: every row states its stage, its class and (literal and derived defaults) the
// status of its default, and the scope follows from the key's section (§1.5). A row that omits one is refused
// when the table is built (std::logic_error naming the key), so no table with such a row can load.
#pragma once

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace balla::config {

enum class Type : uint8_t { kBool, kInt, kFloat, kString, kEnum, kFloatList, kIntList, kStringList, kPath };
enum class Default : uint8_t { kLiteral, kRequired, kFamily, kDerived };
enum class Scale : uint8_t { kLinear, kLog };
enum class Status : uint8_t { kPrior, kMeasured, kFitted, kPlaceholder, kDerived, kData, kChoice };
enum class Class : uint8_t { kTSensor, kTScene, kTRun, kNAccuracy, kDMeasured, kEHeuristic };
enum class Scope : uint8_t { kCamera, kScene, kRun, kDataset };
// Pipeline stages in order: a change invalidates its stage and every later one (P0_PARAMS §1.2).
enum class Stage : uint8_t { kScene, kRender, kSchedule, kSensor, kOutput, kNone };
// The aggregates rows apply to (§1.3).
enum class Target : uint8_t { kNone, kSensorParams, kSensorSpec, kSceneSettings, kRendererOptions, kEngineConfig };

// null, bool, int64, double, string, double list, int64 list, string list.
using Value = std::variant<std::monostate, bool, int64_t, double, std::string, std::vector<double>, std::vector<int64_t>,
                           std::vector<std::string>>;

inline constexpr double kInf = std::numeric_limits<double>::infinity();

// A single-key range with open or closed ends; the full line when unset.
struct Range {
  double lo = -kInf, hi = kInf;
  bool lo_open = false, hi_open = false;
  bool unset() const { return lo == -kInf && hi == kInf; }
  // NaN and infinities are outside every range.
  bool contains(double v) const {
    if (!std::isfinite(v)) return false;
    return (lo_open ? v > lo : v >= lo) && (hi_open ? v < hi : v <= hi);
  }
};
constexpr Range above(double lo) { return {lo, kInf, true, false}; }       // (lo, inf)
constexpr Range at_least(double lo) { return {lo, kInf, false, false}; }   // [lo, inf)
constexpr Range closed(double lo, double hi) { return {lo, hi, false, false}; }
constexpr Range right_open(double lo, double hi) { return {lo, hi, false, true}; }  // [lo, hi)
constexpr Range left_open(double lo, double hi) { return {lo, hi, true, false}; }   // (lo, hi]
constexpr Range open(double lo, double hi) { return {lo, hi, true, true}; }

// Relevance: the row matters only while `key` (same scope root) has `value` (a string list on a string key: one of its
// strings). A row's conditions all hold for it to matter; a condition whose key has no value (not resolved yet, e.g.
// a bundle fact before the bundle) holds.
struct When {
  std::string key;
  Value value;
};

struct Param;
using Setter = void (*)(const Param&, void*, const Value&);
using Getter = Value (*)(const void*);

struct Param {
  std::string key;    // dotted path below the scope root (camera keys: below cameras.<camera>)
  std::string field;  // the C++ member of the target ("" without one); _sensor attribute names
  Type type = Type::kFloat;
  bool nullable = false;  // null is a meaning of its own
  int32_t length = 0;     // lists: the fixed length, or -1 for any; 0 for scalars
  std::string unit;
  Default def = Default::kLiteral;
  Value value;          // the literal default
  std::string formula;  // derived defaults: the formula as text
  std::vector<std::string> choices;
  Range hard, soft;  // refused outside hard; a warning outside soft (lists: per element)
  Scale scale = Scale::kLinear;
  // Status of a literal or derived default; none for family defaults (each prior has its own) and required keys.
  std::optional<Status> status;
  bool evimo2_derived = false;  // the default comes from EVIMO2 data or its setup (lens, assumed illuminant)
  std::string source, help;
  Class cls = Class::kTRun;
  Scope scope = Scope::kRun;  // from the key's section
  Stage stage = Stage::kScene;
  std::vector<When> when;  // relevance: every condition holds (empty: always)
  std::vector<When> also_when;  // or every one of these holds (empty: no alternative); readers check both
  Target target = Target::kNone;
  bool custom = false;  // a custom setter and getter (unit conversion) instead of a member
  Setter set = nullptr;
  Getter get = nullptr;
};

// Row options, written with designated initializers in this order. stage and cls are required; status is required
// for literal and derived defaults and refused for family and required ones (make_param).
struct Opts {
  std::optional<Type> type{};  // member rows: deduced (enum or path for string members)
  bool nullable = false;       // member rows: deduced from std::optional
  int32_t length = 0;          // member rows: deduced from std::array
  std::string_view unit{};     // required for numbers ("1": dimensionless or a count); empty otherwise
  Default def = Default::kLiteral;
  Value value{};  // literal default of rows without a member, and of custom rows (member rows: S{}.*m)
  std::string_view formula{};
  std::vector<std::string> choices{};
  Range hard{}, soft{};
  Scale scale = Scale::kLinear;
  std::optional<Status> status{};
  bool evimo2 = false;
  std::optional<Class> cls{};
  std::optional<Stage> stage{};
  std::vector<When> when{};
  std::vector<When> also_when{};
  std::string_view field{};  // member rows: default the last key component
  std::string_view source{}, help{};
};
// Param fields that Opts does not carry: key, scope (from the key), target, custom, set and get. The schema test
// compares the two field counts, so a descriptor field added to one struct and not to the other is caught.
inline constexpr size_t kParamOnlyFields = 6;

// ---------------------------------------------------------------------------------------------
// Names and values
// ---------------------------------------------------------------------------------------------

inline const char* name(Type t) {
  static constexpr const char* k[] = {"bool", "int", "float", "string", "enum", "float_list", "int_list",
                                      "string_list", "path"};
  return k[static_cast<int>(t)];
}
inline const char* name(Default d) {
  static constexpr const char* k[] = {"literal", "required", "family", "derived"};
  return k[static_cast<int>(d)];
}
inline const char* name(Scale s) { return s == Scale::kLog ? "log" : "linear"; }
inline const char* name(Status s) {
  static constexpr const char* k[] = {"prior", "measured", "fitted", "placeholder", "derived", "data", "choice"};
  return k[static_cast<int>(s)];
}
inline const char* name(Class c) {
  static constexpr const char* k[] = {"T-sensor", "T-scene", "T-run", "N-accuracy", "D-measured", "E-heuristic"};
  return k[static_cast<int>(c)];
}
inline const char* name(Scope s) {
  static constexpr const char* k[] = {"camera", "scene", "run", "dataset"};
  return k[static_cast<int>(s)];
}
inline const char* name(Stage s) {
  static constexpr const char* k[] = {"scene", "render", "schedule", "sensor", "output", "none"};
  return k[static_cast<int>(s)];
}
inline const char* name(Target t) {
  static constexpr const char* k[] = {"", "SensorParams", "SensorSpec", "SceneSettings", "RendererOptions",
                                      "EngineConfig"};
  return k[static_cast<int>(t)];
}

// Shortest round-trip text of a double ("inf" and "nan" for those).
inline std::string format_double(double v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  char buf[32];
  const auto r = std::to_chars(buf, buf + sizeof(buf), v);
  return std::string(buf, r.ptr);
}

namespace detail {
struct FormatValue {
  std::string operator()(std::monostate) const { return "null"; }
  std::string operator()(bool b) const { return b ? "true" : "false"; }
  std::string operator()(int64_t i) const { return std::to_string(i); }
  std::string operator()(double d) const { return format_double(d); }
  std::string operator()(const std::string& s) const { return "'" + s + "'"; }
  template <class T>
  std::string operator()(const std::vector<T>& l) const {
    std::string out = "[";
    for (size_t i = 0; i < l.size(); ++i) out += (i ? ", " : "") + (*this)(l[i]);
    return out + "]";
  }
};
}  // namespace detail

inline std::string format_value(const Value& v) { return std::visit(detail::FormatValue{}, v); }

// Infinite ends print open: no range holds an infinity.
inline std::string format_range(const Range& r) {
  return std::string(r.lo_open || std::isinf(r.lo) ? "(" : "[") + format_double(r.lo) + ", " + format_double(r.hi) +
         (r.hi_open || std::isinf(r.hi) ? ")" : "]");
}

// How messages name a row: "field (key)" for rows with a C++ member (the _sensor attribute and the format-1
// override use the field, configurations the key), the key otherwise.
inline std::string display_name(const Param& p) { return p.field.empty() ? p.key : p.field + " (" + p.key + ")"; }

// The scope of a key, from its section (P0_PARAMS §1.5); std::logic_error for a section of no scope. The IMU's
// section is camera scope: one IMU per camera (docs/design/phase3/outputs.md §3.1); so is `frames`: the frames of
// each camera (docs/design/phase3/frames.md §3.2).
inline Scope scope_of(std::string_view key) {
  const std::string_view s = key.substr(0, key.find('.'));
  if (s == "sensor" || s == "photometry" || s == "optics" || s == "imu" || s == "frames") {
    return Scope::kCamera;
  }
  if (s == "scene") return Scope::kScene;
  if (s == "dataset") return Scope::kDataset;
  if (s == "run" || s == "accuracy" || s == "execution" || s == "output") return Scope::kRun;
  throw std::logic_error("parameter row " + std::string(key) + ": no scope has the section " + std::string(s));
}

// Checks a value against a row: null only where nullable; the type (an integer is a float, never the reverse:
// the bindings refuse a Python float for an integer key); enum choices; list lengths; finite numbers inside the
// hard range (lists: every element). Throws std::invalid_argument naming `what` (the key or display_name()).
inline void check_value(const Param& p, const Value& v, std::string_view what) {
  const auto fail = [&](const std::string& why) {
    throw std::invalid_argument(std::string(what) + " = " + format_value(v) + ": " + why);
  };
  if (std::holds_alternative<std::monostate>(v)) {
    if (!p.nullable) fail("a value is required");
    return;
  }
  const auto number = [&](double x) {
    if (!std::isfinite(x)) fail("must be finite");
    if (!p.hard.contains(x)) fail("outside " + format_range(p.hard));
  };
  switch (p.type) {
    case Type::kBool:
      if (!std::holds_alternative<bool>(v)) fail("expected true or false");
      return;
    case Type::kInt:
      if (!std::holds_alternative<int64_t>(v)) fail("expected an integer");
      number(static_cast<double>(std::get<int64_t>(v)));
      return;
    case Type::kFloat:
      if (const auto* i = std::get_if<int64_t>(&v)) return number(static_cast<double>(*i));
      if (!std::holds_alternative<double>(v)) fail("expected a number");
      number(std::get<double>(v));
      return;
    case Type::kString:
    case Type::kPath:
      if (!std::holds_alternative<std::string>(v)) fail("expected a string");
      return;
    case Type::kEnum: {
      const auto* s = std::get_if<std::string>(&v);
      if (!s) fail("expected a string");
      for (const std::string& c : p.choices) {
        if (c == *s) return;
      }
      std::string all;
      for (const std::string& c : p.choices) all += (all.empty() ? "" : ", ") + c;
      fail("one of " + all);
      return;
    }
    case Type::kFloatList:
    case Type::kIntList: {
      std::vector<double> xs;
      if (const auto* li = std::get_if<std::vector<int64_t>>(&v)) {
        for (int64_t x : *li) xs.push_back(static_cast<double>(x));
      } else if (const auto* l = std::get_if<std::vector<double>>(&v); l && p.type == Type::kFloatList) {
        xs = *l;
      } else {
        fail(p.type == Type::kIntList ? "expected a list of integers" : "expected a list of numbers");
      }
      if (p.length >= 0 && xs.size() != static_cast<size_t>(p.length)) {
        fail("expected " + std::to_string(p.length) + " values");
      }
      for (double x : xs) number(x);
      return;
    }
    case Type::kStringList:
      if (!std::holds_alternative<std::vector<std::string>>(v)) fail("expected a list of strings");
      return;
  }
}

// ---------------------------------------------------------------------------------------------
// Member rows
// ---------------------------------------------------------------------------------------------

// The aggregate a struct's rows apply to; specialized next to the rows (TargetOf<SensorParams>, ...).
template <class S>
struct TargetOf;

// Names of an enum's values, indexed by the underlying value; specialized next to the enum's rows.
template <class E>
struct EnumNames;

namespace detail {

template <class T>
struct IsOptional : std::false_type {};
template <class T>
struct IsOptional<std::optional<T>> : std::true_type {
  using type = T;
};
template <class T>
struct IsArray : std::false_type {};
template <class T, size_t N>
struct IsArray<std::array<T, N>> : std::true_type {
  using type = T;
  static constexpr size_t size = N;
};

template <class S, class T>
S struct_of(T S::*);
template <class S, class T>
T field_of(T S::*);

template <class T>
constexpr Type type_of() {
  if constexpr (IsOptional<T>::value) {
    return type_of<typename IsOptional<T>::type>();
  } else if constexpr (IsArray<T>::value) {
    return std::is_integral_v<typename IsArray<T>::type> ? Type::kIntList : Type::kFloatList;
  } else if constexpr (std::is_same_v<T, bool>) {
    return Type::kBool;
  } else if constexpr (std::is_enum_v<T>) {
    return Type::kEnum;
  } else if constexpr (std::is_integral_v<T>) {
    return Type::kInt;
  } else if constexpr (std::is_floating_point_v<T>) {
    return Type::kFloat;
  } else {
    static_assert(std::is_same_v<T, std::string>, "unsupported member type");
    return Type::kString;
  }
}

[[noreturn]] inline void bad_value(const Param& p, const Value& v, const char* expected) {
  throw std::invalid_argument(display_name(p) + " = " + format_value(v) + ": expected " + expected);
}

inline double to_double(const Param& p, const Value& v) {
  if (const auto* d = std::get_if<double>(&v)) return *d;
  if (const auto* i = std::get_if<int64_t>(&v)) return static_cast<double>(*i);
  bad_value(p, v, "a number");
}

inline int64_t to_int(const Param& p, const Value& v) {
  if (const auto* i = std::get_if<int64_t>(&v)) return *i;
  bad_value(p, v, "an integer");
}

template <class T>
Value to_value(const T& x) {
  if constexpr (IsOptional<T>::value) {
    return x ? to_value(*x) : Value{};
  } else if constexpr (IsArray<T>::value) {
    using E = typename IsArray<T>::type;
    if constexpr (std::is_integral_v<E>) return std::vector<int64_t>(x.begin(), x.end());
    else return std::vector<double>(x.begin(), x.end());
  } else if constexpr (std::is_same_v<T, bool>) {
    return x;
  } else if constexpr (std::is_enum_v<T>) {
    return std::string(EnumNames<T>::names[static_cast<size_t>(x)]);
  } else if constexpr (std::is_integral_v<T>) {
    return static_cast<int64_t>(x);
  } else if constexpr (std::is_floating_point_v<T>) {
    return static_cast<double>(x);
  } else {
    return x;
  }
}

template <class T>
T from_value(const Param& p, const Value& v) {
  if constexpr (IsOptional<T>::value) {
    if (std::holds_alternative<std::monostate>(v)) return std::nullopt;
    return from_value<typename IsOptional<T>::type>(p, v);
  } else if constexpr (IsArray<T>::value) {
    // A list of the row's element type (an integer list for integer elements; a float list also takes integers).
    using E = typename IsArray<T>::type;
    T out{};
    const auto fill = [&](const auto& xs) {
      using X = typename std::decay_t<decltype(xs)>::value_type;
      if (xs.size() != out.size()) bad_value(p, v, ("a list of " + std::to_string(out.size()) + " values").c_str());
      for (size_t i = 0; i < out.size(); ++i) {
        if constexpr (std::is_integral_v<E> && std::is_integral_v<X>) {
          if (!std::in_range<E>(xs[i])) bad_value(p, v, "a list of integers in the member's range");
        }
        out[i] = static_cast<E>(xs[i]);
      }
    };
    if (const auto* li = std::get_if<std::vector<int64_t>>(&v)) fill(*li);
    else if (const auto* l = std::get_if<std::vector<double>>(&v); l && !std::is_integral_v<E>) fill(*l);
    else bad_value(p, v, std::is_integral_v<E> ? "a list of integers" : "a list of numbers");
    return out;
  } else if constexpr (std::is_same_v<T, bool>) {
    if (const auto* b = std::get_if<bool>(&v)) return *b;
    bad_value(p, v, "true or false");
  } else if constexpr (std::is_enum_v<T>) {
    const auto* s = std::get_if<std::string>(&v);
    if (s) {
      for (size_t i = 0; i < EnumNames<T>::names.size(); ++i) {
        if (EnumNames<T>::names[i] == *s) return static_cast<T>(i);
      }
    }
    std::string all;
    for (std::string_view n : EnumNames<T>::names) all += (all.empty() ? "" : ", ") + std::string(n);
    bad_value(p, v, ("one of " + all).c_str());
  } else if constexpr (std::is_integral_v<T>) {
    const int64_t i = to_int(p, v);
    if (i < static_cast<int64_t>(std::numeric_limits<T>::min()) ||
        (i > 0 && static_cast<uint64_t>(i) > static_cast<uint64_t>(std::numeric_limits<T>::max()))) {
      bad_value(p, v, "an integer in the member's range");
    }
    return static_cast<T>(i);
  } else if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(to_double(p, v));
  } else {
    if (const auto* s = std::get_if<std::string>(&v)) return *s;
    bad_value(p, v, "a string");
  }
}

// A row from its options; std::logic_error naming the key when a required option is missing (P0_PARAMS §1.1).
inline Param make_param(std::string key, const Opts& o) {
  const auto refuse = [&](const char* why) { throw std::logic_error("parameter row " + key + ": " + why); };
  if (!o.stage) refuse("state the stage it invalidates (.stage)");
  if (!o.cls) refuse("state its class (.cls)");
  const bool own_default = o.def == Default::kLiteral || o.def == Default::kDerived;
  if (own_default && !o.status) refuse("state the status of its default (.status)");
  if (!own_default && o.status) refuse("a family or required default has no status of its own");
  Param p;
  p.scope = scope_of(key);
  p.key = std::move(key);
  p.field = std::string(o.field);
  p.type = o.type.value_or(Type::kFloat);
  p.nullable = o.nullable;
  p.length = o.length;
  p.unit = std::string(o.unit);
  p.def = o.def;
  p.value = o.value;
  p.formula = std::string(o.formula);
  p.choices = o.choices;
  p.hard = o.hard;
  p.soft = o.soft;
  p.scale = o.scale;
  p.status = o.status;
  p.evimo2_derived = o.evimo2;
  p.source = std::string(o.source);
  p.help = std::string(o.help);
  p.cls = *o.cls;
  p.stage = *o.stage;
  p.when = o.when;
  p.also_when = o.also_when;
  return p;
}

}  // namespace detail

// A member row of S: type, nullability, list length, enum choices, setter, getter and (literal rows) the default
// S{}.*M are deduced; an explicit value is refused, so the member initializer stays the single source.
template <auto M>
Param member(std::string key, Opts o) {
  using S = decltype(detail::struct_of(M));
  using T = decltype(detail::field_of(M));
  if (!std::holds_alternative<std::monostate>(o.value)) {
    throw std::logic_error("parameter row " + key + ": a member row's default is its member initializer");
  }
  std::string field = o.field.empty() ? key.substr(key.rfind('.') + 1) : std::string(o.field);
  Param p = detail::make_param(std::move(key), o);
  p.field = std::move(field);
  p.target = TargetOf<S>::value;
  p.type = o.type.value_or(detail::type_of<T>());
  p.nullable = detail::IsOptional<T>::value;
  if constexpr (detail::IsArray<T>::value) p.length = static_cast<int32_t>(detail::IsArray<T>::size);
  if constexpr (std::is_enum_v<T>) {
    if (p.choices.empty()) {
      for (std::string_view n : EnumNames<T>::names) p.choices.emplace_back(n);
    }
  }
  if (p.def == Default::kLiteral) p.value = detail::to_value(S{}.*M);
  p.set = [](const Param& r, void* s, const Value& v) { static_cast<S*>(s)->*M = detail::from_value<T>(r, v); };
  p.get = [](const void* s) { return detail::to_value(static_cast<const S*>(s)->*M); };
  return p;
}

// A custom row of S: Set(row, s, value) and Get(s) convert between the user's unit and the engine's; the type
// and the literal default are given.
template <class S, auto Set, auto Get>
Param custom(std::string key, Opts o) {
  Param p = detail::make_param(std::move(key), o);
  p.target = TargetOf<S>::value;
  p.custom = true;
  p.set = [](const Param& r, void* s, const Value& v) { Set(r, *static_cast<S*>(s), v); };
  p.get = [](const void* s) { return Get(*static_cast<const S*>(s)); };
  return p;
}

// A row without a target (consumed by the resolver or by Python).
inline Param row(std::string key, Opts o) { return detail::make_param(std::move(key), o); }

// Typed apply: refuses a row of another target. Member setters check the type, not the hard range (validate()
// and, from P0b, the resolver do); custom setters check both, as their conversion needs a valid value.
template <class S>
void set(const Param& p, S& s, const Value& v) {
  if (p.target != TargetOf<S>::value || !p.set) {
    throw std::logic_error(p.key + ": not a row of " + std::string(name(TargetOf<S>::value)));
  }
  p.set(p, &s, v);
}
template <class S>
Value get(const Param& p, const S& s) {
  if (p.target != TargetOf<S>::value || !p.get) {
    throw std::logic_error(p.key + ": not a row of " + std::string(name(TargetOf<S>::value)));
  }
  return p.get(&s);
}

// Resolved values as the engine, the renderer and the sensor take them (docs/dev/P0_PARAMS.md §1.4, Resolved.values()):
// the run keys, and per camera its camera keys (key -> value).
using ValueMap = std::map<std::string, Value, std::less<>>;
struct Values {
  ValueMap run;
  std::vector<std::pair<std::string, ValueMap>> cameras;  // in order; the first gives the run-wide camera keys
};
// Row p's value in `m`, checked against the row; `where` prefixes the key in messages ("" or "cameras.<name>.").
inline const Value& value_of(const Param& p, const ValueMap& m, const std::string& where) {
  const auto it = m.find(p.key);
  if (it == m.end()) throw std::invalid_argument("values: " + where + p.key + " is missing");
  check_value(p, it->second, where + p.key);
  return it->second;
}

// The number of members of an aggregate, by brace initialization (host only; the boost.pfr trick).
namespace detail {
struct AnyField {
  template <class T>
  operator T() const;  // never defined: unevaluated only
};
template <class S, class... A>
constexpr size_t field_count_impl() {
  if constexpr (requires { S{A{}..., AnyField{}}; }) return field_count_impl<S, A..., AnyField>();
  else return sizeof...(A);
}
}  // namespace detail
template <class S>
constexpr size_t field_count() {
  return detail::field_count_impl<S>();
}

}  // namespace balla::config
