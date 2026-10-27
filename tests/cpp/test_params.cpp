// The parameter schema (docs/dev/P0_PARAMS.md §5, P0a): unique keys, complete descriptors (rows that omit a
// required option are refused), member counts with exclusions, one member per row, custom-row round trips, defaults
// inside their hard ranges, enum choices, complete family tables, the initializers (SceneSettings repeats a prior,
// SensorParams carries none: D2), SHA-256, apply() and checks of resolve();
// the sensor maps at the defaults (golden hashes of the code before P0a), the D11 statistics, the label cut and the
// single PSF source.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "config/resolve.h"
#include "config/schema.h"
#include "config/sha256.h"
#include "core/time/time.h"
#include "engine/param_fields.h"
#include "render/param_fields.h"
#include "sensor/maps.h"
#include "sensor/param_fields.h"
#include "sensor/sensor_model.h"

using namespace balla;
using config::Default;
using config::Param;
using config::Stage;
using config::Target;
using config::Type;
using config::Value;

namespace {

// Pointer to a default-constructed aggregate of a row's target.
struct Targets {
  sensor::SensorParams sensor_params;
  engine::SensorSpec sensor_spec;
  render::SceneSettings scene_settings;
  render::RendererOptions renderer_options;
  engine::EngineConfig engine_config;
  void* of(Target t) {
    switch (t) {
      case Target::kSensorParams: return &sensor_params;
      case Target::kSensorSpec: return &sensor_spec;
      case Target::kSceneSettings: return &scene_settings;
      case Target::kRendererOptions: return &renderer_options;
      case Target::kEngineConfig: return &engine_config;
      case Target::kNone: break;
    }
    return nullptr;
  }
};

// FNV-1a over every map field (the golden program of P0a ran it on the tree before the change).
uint64_t maps_hash(const sensor::SensorMaps& m) {
  uint64_t h = 0xcbf29ce484222325ull;
  const auto bytes = [&](const void* p, size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) {
      h ^= b[i];
      h *= 0x100000001b3ull;
    }
  };
  for (const sensor::PixelMap& p : m.pixels) {
    for (const double* v : {&p.theta_on, &p.theta_off, &p.leak, &p.dark_fA, &p.phi0}) bytes(v, sizeof(double));
    const auto t = static_cast<uint8_t>(p.hot);
    bytes(&t, 1);
  }
  for (const double v : m.hot_target_hz) bytes(&v, sizeof(v));
  return h;
}

// Resolver documents (config::Node): a leaf, or a mapping in document order.
using Map = std::vector<std::pair<std::string, config::Node>>;
config::Node leaf(Value v) {
  config::Node n;
  n.leaf = std::move(v);
  return n;
}
config::Node mapping(Map m) {
  config::Node n;
  n.is_map = true;
  n.map = std::move(m);
  return n;
}
config::Node str(const char* s) { return leaf(std::string(s)); }

// A run configuration on prophesee_gen3 cameras with the top-level `sections` and camera left's block `left`.
config::Inputs run_inputs(Map sections, Map left = {}) {
  Map cams{{"default", mapping({{"sensor.family", str("prophesee_gen3")},
                                  {"photometry.illuminance_scale", leaf(200.0)}})}};
  if (!left.empty()) cams.emplace_back("left", mapping(std::move(left)));
  Map doc{{"format", str("ballasim.run/2")}, {"output", mapping({{"dir", str("out")}})},
          {"cameras", mapping(std::move(cams))}};
  for (auto& s : sections) doc.push_back(std::move(s));
  config::Inputs in;
  in.config = mapping(std::move(doc));
  return in;
}

// A profile file that sets one key to a prior.
config::Node profile(const char* id, const char* mount, const char* key, double v) {
  const config::Node entry = mapping({{"value", leaf(v)}, {"status", str("prior")}, {"source", str("test")}});
  return mapping({{"format", str("ballasim.profile/1")}, {"id", str(id)}, {"mount", str(mount)},
                  {"description", str("test")}, {"values", mapping({{key, entry}})}});
}

// A supplied bundle with one 64 x 48 camera, left.
config::BundleInfo one_camera() {
  config::BundleInfo b;
  b.cameras = {{"left", "prophesee_gen3", 64, 48, 0}};
  b.t_end_ns = 10 * time::kWindowNs;
  b.supplied = true;
  b.bundle_digest = std::string(64, 'b');
  b.engine_digest = std::string(64, 'e');
  return b;
}

}  // namespace

TEST_CASE("schema: unique keys, complete descriptors") {
  std::set<std::string> keys;
  for (const Param& p : config::schema()) {
    INFO(p.key);
    CHECK(keys.insert(p.key).second);
    const std::string top = p.key.substr(0, p.key.find('.'));
    if (top == "execution") CHECK(p.stage == Stage::kNone);  // P0_PARAMS §1.5
    // Numbers carry a unit ("1" for dimensionless values and counts); other types none (§1.1).
    const bool numeric = p.type == Type::kInt || p.type == Type::kFloat || p.type == Type::kIntList ||
                         p.type == Type::kFloatList;
    CHECK(numeric == !p.unit.empty());
    // The status belongs to a literal or derived default; family priors carry their own, required keys have none.
    CHECK((p.def == Default::kLiteral || p.def == Default::kDerived) == p.status.has_value());
    CHECK(!p.help.empty());
    CHECK(p.help.size() <= 120);
    CHECK(!p.source.empty());
    if (p.def == Default::kDerived) CHECK(!p.formula.empty());
    if (p.type == Type::kEnum) CHECK(!p.choices.empty());
    if (p.target != Target::kNone) CHECK((p.set != nullptr && p.get != nullptr && !p.field.empty()));
    for (const config::When& w : p.when) CHECK(config::find(w.key) != nullptr);
  }
  CHECK(keys.size() == config::schema().size());
  // The scope follows the section.
  using config::Scope;
  CHECK(config::find("sensor.k")->scope == Scope::kCamera);
  CHECK(config::find("photometry.illuminance_scale")->scope == Scope::kCamera);
  CHECK(config::find("optics.blur_sigma_px")->scope == Scope::kCamera);
  CHECK(config::find("scene.trajectory.order")->scope == Scope::kScene);
  CHECK(config::find("dataset.evimo2.mains_hz")->scope == Scope::kDataset);
  for (const char* k : {"run.device", "accuracy.eps_L", "execution.scene_workers", "output.label_cut"}) {
    CHECK(config::find(k)->scope == Scope::kRun);
  }
}

TEST_CASE("a row that omits a required option is refused when the table is built") {
  using config::Class;
  using config::Status;
  const auto ok = config::Opts{.unit = "1", .value = 1.0, .status = Status::kChoice, .cls = Class::kTRun,
                               .stage = Stage::kNone, .source = "s", .help = "h"};
  CHECK_NOTHROW(config::row("run.x", ok));
  config::Opts o = ok;
  o.stage.reset();
  CHECK_THROWS_WITH_AS(config::row("run.x", o), doctest::Contains("stage"), std::logic_error);
  o = ok;
  o.cls.reset();
  CHECK_THROWS_WITH_AS(config::row("run.x", o), doctest::Contains("class"), std::logic_error);
  o = ok;
  o.status.reset();
  CHECK_THROWS_WITH_AS(config::row("run.x", o), doctest::Contains("status"), std::logic_error);
  o = ok;
  o.def = Default::kFamily;  // a family default takes the prior's status, not its own
  o.value = Value{};
  CHECK_THROWS_AS(config::row("sensor.x", o), std::logic_error);
  CHECK_THROWS_WITH_AS(config::row("nosection.x", ok), doctest::Contains("section"), std::logic_error);
  // A member row's default is its member initializer.
  CHECK_THROWS_AS(config::member<&engine::EngineConfig::d_px>("accuracy.d_px", ok), std::logic_error);
  // A descriptor field added to Param or to Opts alone changes one count (make_param copies field by field).
  static_assert(config::field_count<Param>() == config::field_count<config::Opts>() + config::kParamOnlyFields);
}

TEST_CASE("values: types, finiteness and ranges, messages naming the row") {
  const Param& level = *config::find("accuracy.noise_leaf_level");
  CHECK_THROWS_WITH_AS(config::check_value(level, 5.0, "accuracy.noise_leaf_level"),
                       "accuracy.noise_leaf_level = 5: expected an integer", std::invalid_argument);
  CHECK_NOTHROW(config::check_value(level, int64_t{5}, "x"));
  sensor::SensorParams p = sensor::preset("prophesee_gen3");
  CHECK_THROWS_WITH_AS(config::set(level, p, 5.5),
                       "noise_leaf_level (accuracy.noise_leaf_level) = 5.5: expected an integer", std::invalid_argument);
  p.eps_ci = std::numeric_limits<double>::quiet_NaN();  // a row without a hard range
  CHECK_THROWS_WITH_AS(p.validate(), "SensorParams: eps_ci (sensor.eps_ci) = nan: must be finite",
                       std::invalid_argument);
  p = sensor::preset("prophesee_gen3");
  p.theta_on = -0.2;
  CHECK_THROWS_WITH_AS(p.validate(), "SensorParams: theta_on (sensor.theta_on) = -0.2: outside (0, inf)",
                       std::invalid_argument);
  // Lists: an integer list takes no floats; a float list takes integers.
  const Param& crop = *config::find("run.crop_px");
  CHECK_THROWS_AS(config::check_value(crop, std::vector<double>{0, 0, 1, 1}, "c"), std::invalid_argument);
  CHECK_THROWS_AS(config::check_value(crop, std::vector<int64_t>{0, 0, 1}, "c"), std::invalid_argument);
  CHECK_NOTHROW(config::check_value(*config::find("scene.evimo2.light_color"), std::vector<int64_t>{1, 1, 1}, "l"));
  CHECK_THROWS_AS(config::check_value(*config::find("run.select"), std::vector<double>{}, "s"), std::invalid_argument);
  CHECK(sensor::unknown_field("sigma_theta").find("sigma_theta_on and sigma_theta_off") != std::string::npos);
  CHECK(sensor::unknown_field("x") == "unknown sensor parameter 'x'");
}

TEST_CASE("every member of a target has a row or a declared exclusion") {
  using config::field_count;
  using config::rows_of;
  // Exclusions are member pointers: a removed or renamed excluded member breaks the build.
  constexpr auto n_spec = std::tuple_size_v<decltype(engine::kSensorSpecExcluded)>;
  constexpr auto n_options = std::tuple_size_v<decltype(render::kRendererOptionsExcluded)>;
  constexpr auto n_engine = std::tuple_size_v<decltype(engine::kEngineConfigExcluded)>;
  CHECK(field_count<sensor::SensorParams>() == rows_of(Target::kSensorParams).size());
  CHECK(field_count<engine::SensorSpec>() == rows_of(Target::kSensorSpec).size() + n_spec);
  CHECK(field_count<render::SceneSettings>() == rows_of(Target::kSceneSettings).size());
  CHECK(field_count<render::RendererOptions>() == rows_of(Target::kRendererOptions).size() + n_options);
  CHECK(field_count<engine::EngineConfig>() == rows_of(Target::kEngineConfig).size() + n_engine);
  // Field names are unique within a target (the _sensor attributes and old overrides use them).
  for (Target t : {Target::kSensorParams, Target::kSensorSpec, Target::kSceneSettings, Target::kRendererOptions,
                   Target::kEngineConfig}) {
    std::set<std::string> fields;
    for (const Param* p : rows_of(t)) CHECK(fields.insert(p->field).second);
  }
}

TEST_CASE("custom rows round-trip between the user's unit and the engine's") {
  engine::EngineConfig c;
  const Param& rate = *config::find("run.fixed_rate_hz");
  for (const double hz : {1e6, 1.0, 2e8, 1e9}) {
    config::set(rate, c, hz);
    CHECK(c.fixed_period_ns == std::llround(1e9 / hz));
    CHECK(std::get<double>(config::get(rate, c)) == hz);
  }
  config::set(rate, c, Value{});
  CHECK(c.fixed_period_ns == 0);
  CHECK(std::holds_alternative<std::monostate>(config::get(rate, c)));
  CHECK_THROWS_AS(config::set(rate, c, -1.0), std::invalid_argument);
  CHECK_THROWS_AS(config::set(rate, c, int64_t{0}), std::invalid_argument);

  const Param& crop = *config::find("run.crop_px");
  config::set(crop, c, std::vector<int64_t>{520, 264, 584, 328});
  REQUIRE(c.crop);
  CHECK((c.crop->x0 == 520 && c.crop->y0 == 264 && c.crop->x1 == 584 && c.crop->y1 == 328));
  CHECK(std::get<std::vector<int64_t>>(config::get(crop, c)) == std::vector<int64_t>{520, 264, 584, 328});
  config::set(crop, c, Value{});
  CHECK(!c.crop);
  CHECK(std::holds_alternative<std::monostate>(config::get(crop, c)));
  CHECK_THROWS_AS(config::set(crop, c, std::vector<int64_t>{0, 0, 1}), std::invalid_argument);
  CHECK_THROWS_AS(config::set(crop, c, std::vector<int64_t>{-1, 0, 1, 1}), std::invalid_argument);
  CHECK_THROWS_AS(config::set(crop, c, std::vector<int64_t>{0, 0, (int64_t{1} << 32) + 10, 10}),
                  std::invalid_argument);  // not narrowed to x1 = 10

  render::RendererOptions o;
  const Param& device = *config::find("run.device");
  config::set(device, o, std::string("cpu"));
  CHECK(o.backend == render::BackendKind::kEmbree);
  CHECK(std::get<std::string>(config::get(device, o)) == "cpu");
  config::set(device, o, std::string("gpu"));
  CHECK(o.backend == render::BackendKind::kOptix);
  CHECK(std::get<std::string>(config::get(device, o)) == "gpu");
  CHECK_THROWS_AS(config::set(device, o, std::string("auto")), std::invalid_argument);  // resolved first
  CHECK_THROWS_AS(config::set(device, o, std::string("optix")), std::invalid_argument);

  const Param& cache = *config::find("execution.pose_cache_mb");
  CHECK(std::get<int64_t>(config::get(cache, o)) == 256);  // RendererOptions{}: 256 << 20 bytes
  CHECK(std::get<int64_t>(cache.value) == 256);
  config::set(cache, o, int64_t{1});
  CHECK(o.pose_cache_bytes == uint64_t{1} << 20);
  CHECK(std::get<int64_t>(config::get(cache, o)) == 1);
  CHECK_THROWS_AS(config::set(cache, o, int64_t{-1}), std::invalid_argument);
  CHECK_THROWS_AS(config::set(cache, o, (int64_t{1} << 40) + 1), std::invalid_argument);

  // Member rows read back what they set, and refuse another target's struct.
  sensor::SensorParams p;
  config::set(*config::find("sensor.theta_off"), p, 0.375);
  CHECK(p.theta_off == 0.375);
  CHECK_THROWS_AS(config::set(*config::find("sensor.theta_off"), c, 0.375), std::logic_error);
}

TEST_CASE("every default lies inside its row's hard range") {
  for (const Param& p : config::schema()) {
    INFO(p.key);
    if (p.def == Default::kLiteral) CHECK_NOTHROW(config::check_value(p, p.value, p.key));
    else CHECK(std::holds_alternative<std::monostate>(p.value));
  }
  for (const sensor::Family& f : sensor::families()) {
    for (const sensor::Prior& v : f.priors) {
      INFO(f.name << " " << v.key);
      const Param* p = config::find(v.key);
      REQUIRE(p != nullptr);
      CHECK(p->def == Default::kFamily);
      CHECK_NOTHROW(config::check_value(*p, v.value, v.key));
    }
  }
  // The accuracy profiles' literals (the resolver stores them unchecked).
  for (const config::AccuracyProfile& a : config::accuracy_profiles()) {
    for (const config::AccuracyEntry& e : a.entries) {
      INFO(a.name << " " << e.key);
      REQUIRE(config::find(e.key) != nullptr);
      if (e.formula.empty()) CHECK_NOTHROW(config::check_value(*config::find(e.key), e.value, e.key));
    }
  }
}

// Each row sets its own member and no other: perturb one row at a time and compare every getter.
TEST_CASE("one member per row: setting a row moves its getter alone") {
  const auto perturb = [](const Param& p, const Value& v) -> Value {
    if (const auto* b = std::get_if<bool>(&v)) return !*b;
    if (const auto* i = std::get_if<int64_t>(&v)) return *i + 1;
    if (const auto* d = std::get_if<double>(&v)) return *d * 2 + 1;
    if (const auto* l = std::get_if<std::vector<double>>(&v)) {
      std::vector<double> out = *l;
      for (double& x : out) x += 1;
      // An empty list: a valid value of the row (optics.relative_illumination []: [c2, c4]; sensor.nir_fraction []: f).
      if (out.empty()) out = p.key == "sensor.nir_fraction" ? std::vector<double>{0.5, 0.25} : std::vector<double>{-0.125, 0.0};
      return out;
    }
    if (const auto* s = std::get_if<std::string>(&v)) {
      if (p.type != Type::kEnum) return *s + "x";
      for (const std::string& c : p.choices) {  // another choice the setter takes (run.device: not auto)
        if (c != *s && c != "auto") return c;
      }
    }
    // null: a value of the row's type (inside its hard range: accuracy.n_band_max from 64).
    if (p.type == Type::kIntList) return std::vector<int64_t>(static_cast<size_t>(p.length), 1);
    if (p.type == Type::kInt) {
      return std::isfinite(p.hard.lo) ? std::max<int64_t>(1, std::llround(p.hard.lo)) : int64_t{1};
    }
    if (p.type == Type::kBool) return true;  // optics.vignetting
    return 1.0;
  };
  for (const Target target : {Target::kSensorParams, Target::kSensorSpec, Target::kSceneSettings,
                              Target::kRendererOptions, Target::kEngineConfig}) {
    for (const Param* r : config::rows_of(target)) {
      INFO(r->key);
      Targets base, moved;
      base.sensor_params = moved.sensor_params = sensor::preset("prophesee_gen3");
      const Value v = perturb(*r, r->get(base.of(target)));
      r->set(*r, moved.of(target), v);
      CHECK(r->get(moved.of(target)) != r->get(base.of(target)));
      for (const Param* other : config::rows_of(target)) {
        if (other != r) CHECK_MESSAGE(other->get(moved.of(target)) == other->get(base.of(target)), other->key);
      }
    }
  }
}

// SceneSettings{} repeats the prophesee_gen3 prior of its family rows in its member initializers (read by _render's
// keyword defaults and the PSF support constants); they must agree. SensorParams{} carries no prior since the D2
// sentinels: its family-default numbers are NaN and its family empty, which validate() refuses.
TEST_CASE("member initializers: SceneSettings repeats the prophesee_gen3 prior, SensorParams carries none (D2)") {
  Targets t;
  for (const Param& p : config::schema()) {
    if (p.def != Default::kFamily || p.target == Target::kNone) continue;  // sensor.qe: a resolver input, no member
    INFO(p.key);
    REQUIRE((p.target == Target::kSensorParams || p.target == Target::kSceneSettings));
    const config::Value v = p.get(t.of(p.target));
    if (p.target == Target::kSensorParams && p.type == Type::kFloat) {
      CHECK(std::isnan(std::get<double>(v)));
    } else {
      CHECK(v == sensor::find_prior("prophesee_gen3", p.key)->value);
    }
  }
  CHECK(std::isnan(sensor::SensorParams{}.illuminance_scale));
  CHECK(sensor::SensorParams{}.family.empty());
  CHECK_THROWS_WITH_AS(sensor::SensorParams{}.validate(), doctest::Contains("carries no prior"), std::invalid_argument);
  sensor::SensorParams named;
  named.family = "prophesee_gen3";
  CHECK_THROWS_WITH_AS(named.validate(), doctest::Contains("must be finite"), std::invalid_argument);
  sensor::SensorParams p = sensor::preset("prophesee_gen3");
  p.validate();
  p.family = "custom";
  CHECK_THROWS_WITH_AS(p.validate(), doctest::Contains("one of prophesee_gen3, samsung_gen3, ideal"),
                       std::invalid_argument);
}

TEST_CASE("sha256 of the FIPS 180-4 test messages") {
  CHECK(config::sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(config::sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(config::sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  CHECK(config::sha256_hex(std::string(1000, 'a')).substr(0, 16) == "41edece42d63e8d9");
}

TEST_CASE("scene_digest: the scene-stage keys and the facts, never the lighting or the workers (P0b2)") {
  const config::ValueMap base{{"scene.evimo2.sequence", std::string("/data/left_camera/imo/eval/s")},
                              {"scene.evimo2.markers", true},
                              {"scene.trajectory.gate_rot_floor_rad", 0x1.acee9f37bebd6p-5},
                              {"scene.evimo2.ambient", 0.3},
                              {"scene.evimo2.mains_a", std::vector<double>{0.02}},
                              {"execution.scene_workers", int64_t{5}},
                              {"output.dir", std::string("/out")}};
  const std::map<std::string, std::string, std::less<>> facts{{"core", "c"}, {"source", "s"}};
  const std::string d = config::scene_digest(base, facts);
  CHECK(d.size() == 64);
  const auto with = [&](const std::string& key, Value v) {
    config::ValueMap m = base;
    m[key] = std::move(v);
    return config::scene_digest(m, facts);
  };
  // Relit (render, schedule, sensor) and output-invariant keys do not move it ...
  CHECK(with("scene.evimo2.ambient", 1.0) == d);
  CHECK(with("scene.evimo2.mains_a", std::vector<double>{0.002}) == d);
  CHECK(with("scene.evimo2.mains_f0_hz", 100.0) == d);
  CHECK(with("scene.evimo2.low_light", true) == d);  // the bundle's light mode: a fact of the bundle, read by relight
  CHECK(with("execution.scene_workers", int64_t{1}) == d);
  CHECK(with("output.dir", std::string("/elsewhere")) == d);
  // ... scene keys do, to the last bit, and so do the facts.
  CHECK(with("scene.evimo2.markers", false) != d);
  CHECK(with("scene.trajectory.gate_rot_floor_rad", std::nextafter(0x1.acee9f37bebd6p-5, 1.0)) != d);
  CHECK(with("scene.evimo2.marker_albedo", 0.8) != d);
  CHECK(config::scene_digest(base, {{"core", "c"}, {"source", "t"}}) != d);
  CHECK_THROWS_AS(with("scene.evimo2.nope", 1.0), std::invalid_argument);
}

TEST_CASE("resolve(): the CLI layer, camera-block profiles and the run checks") {
  const config::BundleInfo b = one_camera();
  const auto theta_off = [&](const config::Inputs& in) {
    return *config::resolve(in, b).find("sensor.theta_off", "left");
  };
  // The --set items: a mapping of full dotted keys, or null (none); anything else is refused, not dropped.
  config::Inputs in = run_inputs({});
  CHECK(theta_off(in).layer == "prior:prophesee_gen3");
  in.cli = mapping({{"cameras.left.sensor.theta_off", leaf(0.5)}});
  const config::Entry cli = theta_off(in);
  CHECK((cli.value == Value{0.5} && cli.layer == "cli"));
  in.cli = leaf(std::vector<std::string>{"cameras.left.sensor.theta_off=0.5"});
  CHECK_THROWS_WITH_AS(config::resolve(in, b), doctest::Contains("cli"), std::invalid_argument);
  CHECK_THROWS_AS(config::resolve_scene(in), std::invalid_argument);

  // A root-mounted profile is refused in a camera block's list, also when the root list names it too.
  const auto block = [](std::vector<std::string> ids) {
    config::Inputs listed = run_inputs({{"profiles", leaf(std::vector<std::string>{"ex/a@v0"})}},
                                       {{"profiles", leaf(std::move(ids))}});
    listed.profiles.emplace("ex/a@v0", profile("ex/a@v0", "root", "cameras.default.sensor.theta_off", 0.4));
    listed.profiles.emplace("ex/b@v0", profile("ex/b@v0", "camera", "sensor.theta_off", 0.3));
    return listed;
  };
  const config::Entry camera = theta_off(block({"ex/b@v0"}));
  CHECK((camera.value == Value{0.3} && camera.layer == "profile:ex/b@v0"));
  CHECK_THROWS_WITH_AS(config::resolve(block({"ex/b@v0", "ex/a@v0"}), b),
                       "profile ex/a@v0: a root-mounted profile listed in a camera block's profiles",
                       std::invalid_argument);

  // run.fixed_rate_hz: an integral period in ns (to the rounding of 1e9 / hz), whatever the rate.
  const auto rate = [](double hz) { return run_inputs({{"run", mapping({{"fixed_rate_hz", leaf(hz)}})}}); };
  for (const double hz : {1e9 / 3000, 2000.0, 1e6, 0.10733048312820524}) {
    INFO(hz);
    CHECK_NOTHROW(config::resolve(rate(hz), b));
  }
  for (const double hz : {1000.0001, 3000.0, 3e6, 1e-300}) {
    INFO(hz);
    CHECK_THROWS_WITH_AS(config::resolve(rate(hz), b), doctest::Contains("not an integral period"),
                         std::invalid_argument);
  }

  // A derived accuracy value outside its row's range is refused here, not at the engine's apply.
  const Map slow{{"sensor.readout_quantum_s", leaf(0.01)}, {"sensor.readout_delay_s", leaf(0.02)}};
  CHECK_THROWS_WITH_AS(config::resolve(run_inputs({}, slow), b), doctest::Contains("accuracy.min_dt_ns (derived"),
                       std::invalid_argument);
  CHECK_NOTHROW(config::resolve(run_inputs({{"accuracy", mapping({{"min_dt_ns", leaf(int64_t{1024})}})}}, slow), b));

  // scene.trajectory.order: 4 or 6 (C1).
  const auto order = [](int64_t k) { return run_inputs({{"scene", mapping({{"trajectory.order", leaf(k)}})}}); };
  CHECK_NOTHROW(config::resolve(order(4), b));
  CHECK_THROWS_WITH_AS(config::resolve(order(5), b), doctest::Contains("scene.trajectory.order = 5"),
                       std::invalid_argument);
  CHECK_THROWS_AS(config::resolve_scene(order(5)), std::invalid_argument);
}

TEST_CASE("apply(): engine values of a preset reproduce the preset and the struct defaults") {
  // Values of every target row: the prophesee_gen3 family table and the literals, as resolution gives them.
  engine::EngineValues v;
  engine::ValueMap cam;
  const sensor::SensorParams want = sensor::preset("prophesee_gen3");
  const engine::SensorSpec spec_defaults;
  for (const auto* rows : {&sensor::param_rows(), &engine::sensor_spec_rows(), &render::scene_settings_rows(),
                           &render::renderer_options_rows(), &engine::engine_config_rows()}) {
    for (const Param& p : *rows) {
      config::Value value = p.value;
      if (p.target == Target::kSensorParams) value = p.get(&want);
      // A derived member default (optics.vignetting_law, P2-F F13-min) has no literal: the resolver derives it;
      // here the struct's own (cos4).
      if (p.def == Default::kDerived && p.target == Target::kSensorSpec) value = p.get(&spec_defaults);
      if (p.def == Default::kFamily && p.target == Target::kSceneSettings) {
        value = sensor::find_prior("prophesee_gen3", p.key)->value;
      }
      if (p.key == "run.device") value = std::string("cpu");
      (p.scope == config::Scope::kCamera ? cam : v.run)[p.key] = value;
    }
  }
  v.cameras.emplace_back("cam", cam);
  const engine::Applied a = engine::apply(v);
  REQUIRE(a.sensors.size() == 1);
  CHECK(sensor::fingerprint(a.sensors[0].second.params, {}) == sensor::fingerprint(want, {}));
  CHECK(a.config.render.backend == render::BackendKind::kEmbree);
  CHECK(a.config.render.pose_cache_bytes == render::RendererOptions{}.pose_cache_bytes);
  CHECK(a.config.d_px == engine::EngineConfig{}.d_px);
  CHECK(a.config.max_samples_per_window == engine::EngineConfig{}.max_samples_per_window);
  CHECK(render::support_px(a.config.render.scene) == render::support_px(render::SceneSettings{}));
  // A missing key and a disagreeing second camera are refused.
  engine::EngineValues missing = v;
  missing.run.erase("accuracy.d_px");
  CHECK_THROWS_WITH_AS(engine::apply(missing), doctest::Contains("accuracy.d_px is missing"), std::invalid_argument);
  engine::EngineValues two = v;
  two.cameras.emplace_back("cam2", cam);
  two.cameras[1].second["optics.blur_sigma_px"] = 0.25;
  CHECK_THROWS_WITH_AS(engine::apply(two), doctest::Contains("disagree on optics.blur_sigma_px"),
                       std::invalid_argument);
}

TEST_CASE("enum choices parse and read back") {
  for (const Param& p : config::schema()) {
    if (p.type != Type::kEnum || p.target == Target::kNone) continue;
    for (const std::string& c : p.choices) {
      INFO(p.key << " = " << c);
      Targets t;
      if (p.key == "run.device" && c == "auto") {
        CHECK_THROWS_AS(p.set(p, t.of(p.target), c), std::invalid_argument);
        continue;
      }
      p.set(p, t.of(p.target), c);
      CHECK(std::get<std::string>(p.get(t.of(p.target))) == c);
    }
    Targets t;
    if (p.key == "sensor.family") continue;  // a string member: validate() and the resolver check the choice
    CHECK_THROWS_AS(p.set(p, t.of(p.target), std::string("no_such_choice")), std::invalid_argument);
  }
}

TEST_CASE("family tables are complete and preset() reads them") {
  const Param& family = *config::find("sensor.family");
  std::vector<std::string> names;
  for (const sensor::Family& f : sensor::families()) {
    names.push_back(f.name);
    if (!f.base.empty()) CHECK(sensor::find_family(f.base) != nullptr);
  }
  CHECK(family.choices == names);
  CHECK(names == sensor::preset_names());
  CHECK(sensor::families().front().base.empty());
  for (const Param& p : config::schema()) {
    if (p.def != Default::kFamily) continue;
    for (const std::string& n : names) {
      INFO(n << " " << p.key);
      CHECK(sensor::find_prior(n, p.key) != nullptr);
    }
  }
  for (const std::string& n : names) {
    const sensor::SensorParams s = sensor::preset(n);
    CHECK_NOTHROW(s.validate());
    CHECK(s.family == n);
    for (const Param& p : sensor::param_rows()) {
      if (p.def != Default::kFamily) continue;
      INFO(n << " " << p.key);
      CHECK(p.get(&s) == sensor::find_prior(n, p.key)->value);
    }
  }
  bool inherited = false;
  CHECK(sensor::find_prior("samsung_gen3", "sensor.k", &inherited) == sensor::find_prior("prophesee_gen3", "sensor.k"));
  CHECK(inherited);
  CHECK_THROWS_AS(config::default_value("sensor.k"), std::invalid_argument);
  CHECK(std::get<double>(config::default_value("sensor.k", "ideal")) == 1e18);
  CHECK(std::get<double>(config::default_value("accuracy.d_px")) == 0.25);
  CHECK_THROWS_AS(config::default_value("photometry.illuminance_scale"), std::invalid_argument);
}

TEST_CASE("the hot-pixel rows are the sensor.hot_* rows but hot_fraction, inert without hot pixels") {
  for (const Param& p : sensor::param_rows()) {
    INFO(p.key);
    CHECK(sensor::hot_pixel_key(p.key) == (p.key.rfind("sensor.hot_", 0) == 0 && p.key != "sensor.hot_fraction"));
  }
  sensor::SensorParams s = sensor::preset("prophesee_gen3");
  CHECK_FALSE(s.hot_pixels());  // off in the prior (D4c, P0e)
  CHECK(sensor::find_prior("prophesee_gen3", "sensor.hot_fraction")->no_literature);
  s.hot_fraction = 4e-4;
  CHECK(s.hot_pixels());
  CHECK_FALSE(sensor::preset("samsung_gen3").hot_pixels());
  CHECK_FALSE(sensor::preset("ideal").hot_pixels());
  CHECK_FALSE(sensor::find_prior("ideal", "sensor.hot_fraction")->no_literature);  // ideal's own: its definition
}

TEST_CASE("the high-pass filter: off in every prior, its floor, fingerprints and maps unchanged while off") {
  for (const std::string& n : sensor::preset_names()) {
    INFO(n);
    const sensor::SensorParams s = sensor::preset(n);
    CHECK(s.tau_hp == 0.0);
    CHECK(s.tau_hp_sigma_ln == 0.0);
    CHECK_FALSE(s.hpf_on());
    CHECK_FALSE(sensor::find_prior(n, "sensor.tau_hp")->no_literature);  // a bias setting, not D4c's gap
    CHECK(sensor::find_prior(n, "sensor.tau_hp")->status == config::Status::kChoice);
  }
  bool inherited = true;
  CHECK(sensor::find_prior("samsung_gen3", "sensor.tau_hp", &inherited) != nullptr);
  CHECK_FALSE(inherited);  // stated, not inherited
  const config::Param& row = *config::find("sensor.tau_hp");
  CHECK((row.def == Default::kFamily && row.hard.contains(0) && row.hard.contains(100) && !row.hard.contains(100.5)));
  CHECK((row.soft.contains(0) && row.soft.contains(10) && !row.soft.contains(11)));

  // validate(): 0 or at least max(1 ms, 10 max(tau_o, tau_sf)); the cascade only.
  sensor::SensorParams p = sensor::preset("prophesee_gen3");
  CHECK(p.hpf_floor_s() == 1e-3);  // 10 x 100 us
  for (const double tau : {1e-3, 0.05, 0.2, 1.0, 100.0}) {
    p.tau_hp = tau;
    CHECK_NOTHROW(p.validate());
  }
  for (const double tau : {0.9e-3, 1e-6, -0.1, 100.5}) {
    INFO(tau);
    p.tau_hp = tau;
    CHECK_THROWS_AS(p.validate(), std::invalid_argument);
  }
  p.tau_hp = 1.2e-3;
  p.tau_sf = 150e-6;  // floor 1.5 ms
  CHECK_THROWS_WITH_AS(p.validate(), doctest::Contains("tau_hp must be 0 (off) or >="), std::invalid_argument);
  sensor::SensorParams ideal = sensor::preset("ideal");
  ideal.tau_hp = 0.2;
  CHECK_THROWS_WITH_AS(ideal.validate(), doctest::Contains("photoreceptor cascade"), std::invalid_argument);
  ideal.photoreceptor = sensor::Photoreceptor::kCascade;
  CHECK_NOTHROW(ideal.validate());

  // The fingerprint: the rows enter it only while the filter is on (an inert spread changes nothing).
  const sensor::SensorParams off = sensor::preset("prophesee_gen3");
  sensor::SensorParams inert = off;
  inert.tau_hp_sigma_ln = 0.5;
  CHECK(sensor::fingerprint(inert, {}) == sensor::fingerprint(off, {}));
  sensor::SensorParams on = off;
  on.tau_hp = 0.2;
  CHECK(sensor::fingerprint(on, {}) != sensor::fingerprint(off, {}));
  sensor::SensorParams spread = on;
  spread.tau_hp_sigma_ln = 0.5;
  CHECK(sensor::fingerprint(spread, {}) != sensor::fingerprint(on, {}));

  // Maps: no side map without a spread; with one, the log-normal rates (block 3), clamped at the floor. Thresholds,
  // dark currents and the hot set do not move; the leak and the phase follow the filter's laws (the leak of a normal
  // pixel stays, its phase is the filtered leak's renewal law: l tau below theta_on here, so phi0 = l tau).
  const uint32_t n = 320 * 240;
  const sensor::SensorMaps base = sensor::draw_sensor_maps(off, 1, 0, n);
  CHECK(base.hpf_b.empty());
  CHECK(sensor::draw_sensor_maps(on, 1, 0, n).hpf_b.empty());
  CHECK(sensor::draw_sensor_maps(inert, 1, 0, n).hpf_b.empty());
  CHECK(sensor::hpf_tau(on, 1, 0, 17) == 0.2);
  CHECK(sensor::hpf_tau(off, 1, 0, 17) == 0.0);
  spread.tau_hp = 2e-3;
  spread.tau_hp_sigma_ln = 1.0;
  const sensor::SensorMaps m = sensor::draw_sensor_maps(spread, 1, 0, n);
  REQUIRE(m.hpf_b.size() == n);
  std::vector<double> ln_tau;
  size_t floored = 0;
  bool same = true;
  for (uint32_t i = 0; i < n; ++i) {
    const sensor::PixelMap &a = base.pixels[i], &b = m.pixels[i];
    same = same && a.theta_on == b.theta_on && a.theta_off == b.theta_off && a.leak == b.leak &&
           a.dark_fA == b.dark_fA && a.hot == b.hot && b.phi0 == b.leak * sensor::hpf_tau(spread, 1, 0, i);
    CHECK(m.hpf_b[i] == static_cast<float>(1.0 / sensor::hpf_tau(spread, 1, 0, i)));
    floored += sensor::hpf_tau(spread, 1, 0, i) == 1e-3;
    ln_tau.push_back(std::log(sensor::hpf_tau(spread, 1, 0, i)));
  }
  CHECK(same);
  // ln(2 ms / 1 ms) = 0.69 sigma: Phi(-0.69) = 24.5 % of the draws clamp at the floor.
  CHECK(std::fabs(static_cast<double>(floored) / n - 0.2449) < 0.005);
  std::nth_element(ln_tau.begin(), ln_tau.begin() + n / 2, ln_tau.end());
  CHECK(std::fabs(ln_tau[n / 2] - std::log(2e-3)) < 0.01);
}

TEST_CASE("sensor maps at the family defaults equal those of the code before P0a") {
  struct Golden {
    const char* family;
    uint64_t seed;
    uint32_t camera, n;
    uint64_t hash;
  };
  // The families as the P1 runs had them: P0e (D4) moved the prophesee_gen3 refractory, leak and hot-pixel fraction
  // and the samsung_gen3 refractory, report delay and leak to literature priors (the evimo2 v0 profiles keep these
  // values); samsung_gen3 inherited the hot-pixel fraction.
  const auto p1_preset = [](const std::string& family) {
    sensor::SensorParams p = sensor::preset(family);
    if (family == "prophesee_gen3") p.refractory_s = 0.24e-3, p.leak_rate_hz = 0.005;
    if (family == "samsung_gen3") p.refractory_s = 195e-6, p.readout_delay_s = 98.64e-6, p.leak_rate_hz = 0.03;
    if (family != "ideal") p.hot_fraction = 4e-4;
    return p;
  };
  for (const Golden& g : {Golden{"prophesee_gen3", 1, 0, 640 * 480, 0x3b3abd34a1074ca0ull},
                          Golden{"prophesee_gen3", 7, 3, 320 * 240, 0x01ab7b48299a0e29ull},
                          Golden{"samsung_gen3", 1, 0, 640 * 480, 0xa12b4b8d996b5b8eull},
                          Golden{"samsung_gen3", 7, 3, 320 * 240, 0x4df1878d68302871ull},
                          Golden{"ideal", 1, 0, 640 * 480, 0x3e6fc4c51f89d588ull},
                          Golden{"ideal", 7, 3, 320 * 240, 0x39ac05a37f63841bull}}) {
    INFO(g.family << " camera " << g.camera);
    CHECK(maps_hash(sensor::draw_sensor_maps(p1_preset(g.family), g.seed, g.camera, g.n)) == g.hash);
  }
  // Many hot pixels of both types, then the symmetric phase, the ideal reset and 310 K.
  sensor::SensorParams p = p1_preset("prophesee_gen3");
  p.hot_fraction = 0.05;
  CHECK(maps_hash(sensor::draw_sensor_maps(p, 1, 0, 640 * 480)) == 0xcade9d5cab6ff8edull);
  p.initial_phase = sensor::InitialPhase::kSymmetric;
  p.reset_mode = sensor::ResetMode::kIdeal;
  p.temperature_K = 310.0;
  CHECK(maps_hash(sensor::draw_sensor_maps(p, 1, 0, 640 * 480)) == 0x0f4ebf9b714379dbull);
  CHECK(sensor::arrhenius(0.78, 310.0, 298.15) == 0x1.9883911325d2cp+1);
}

TEST_CASE("D11: own ON and OFF spreads, their correlation, the hot-pixel split and the log-normal law") {
  const uint32_t n = 640 * 480;
  sensor::SensorParams p = sensor::preset("prophesee_gen3");
  p.hot_fraction = 0;
  const sensor::SensorMaps base = sensor::draw_sensor_maps(p, 1, 0, n);

  // sigma_off alone moves the OFF map; the inert log-normal keys move nothing.
  sensor::SensorParams q = p;
  q.sigma_theta_off = 0.05;
  q.hot_rate_median_hz = 50;
  q.hot_rate_sigma_ln = 0.2;
  const sensor::SensorMaps off = sensor::draw_sensor_maps(q, 1, 0, n);
  size_t moved = 0;
  bool others_equal = true;
  for (uint32_t i = 0; i < n; ++i) {
    const sensor::PixelMap &a = base.pixels[i], &b = off.pixels[i];
    others_equal = others_equal && a.theta_on == b.theta_on && a.leak == b.leak && a.dark_fA == b.dark_fA &&
                   a.phi0 == b.phi0 && a.hot == b.hot;
    moved += a.theta_off != b.theta_off;
  }
  CHECK(others_equal);
  CHECK(moved > n * 99 / 100);

  // The measured correlation of the two mismatches.
  for (const double rho : {-0.5, 0.8}) {
    q = p;
    q.theta_onoff_corr = rho;
    q.sigma_theta_off = 0.02;
    const sensor::SensorMaps m = sensor::draw_sensor_maps(q, 1, 0, n);
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    bool on_equal = true;
    for (uint32_t i = 0; i < n; ++i) {
      on_equal = on_equal && m.pixels[i].theta_on == base.pixels[i].theta_on;
      const double a = m.pixels[i].theta_on, b = m.pixels[i].theta_off;
      sa += a, sb += b, saa += a * a, sbb += b * b, sab += a * b;
    }
    CHECK(on_equal);
    const double cov = sab / n - (sa / n) * (sb / n);
    const double r = cov / std::sqrt((saa / n - (sa / n) * (sa / n)) * (sbb / n - (sb / n) * (sb / n)));
    INFO("rho " << rho << " measured " << r);
    CHECK(std::fabs(r - rho) < 0.01);
    CHECK(std::fabs(std::sqrt(sbb / n - (sb / n) * (sb / n)) - 0.02) < 0.0005);
  }

  // The type split: none or all leak-type (with noise on).
  q = p;
  q.hot_fraction = 0.2;
  for (const double split : {0.0, 1.0}) {
    q.hot_leak_fraction = split;
    const sensor::SensorMaps m = sensor::draw_sensor_maps(q, 1, 0, n);
    size_t leak = 0, threshold = 0;
    for (const sensor::PixelMap& x : m.pixels) {
      leak += x.hot == sensor::HotType::kLeak;
      threshold += x.hot == sensor::HotType::kThreshold;
    }
    INFO("split " << split);
    CHECK(leak + threshold > n / 6);
    CHECK((split == 0.0 ? leak : threshold) == 0);
  }

  // The log-normal law: its median, clamped to [min, max]; the hot set and the types do not move.
  q = p;
  q.hot_fraction = 0.5;
  const sensor::SensorMaps uniform = sensor::draw_sensor_maps(q, 1, 0, n);
  q.hot_rate_law = sensor::HotRateLaw::kLogNormal;
  q.hot_rate_median_hz = 100;
  q.hot_rate_sigma_ln = 1.0;
  q.hot_rate_min_hz = 5;
  q.hot_rate_max_hz = 1500;
  const sensor::SensorMaps lognormal = sensor::draw_sensor_maps(q, 1, 0, n);
  std::vector<double> rates;
  bool same_types = true, clamped = true;
  for (uint32_t i = 0; i < n; ++i) {
    same_types = same_types && lognormal.pixels[i].hot == uniform.pixels[i].hot;
    if (lognormal.pixels[i].hot == sensor::HotType::kNone) continue;
    const double r = lognormal.hot_target_hz[i];
    clamped = clamped && r >= 5 && r <= 1500;
    rates.push_back(r);
  }
  CHECK(same_types);
  CHECK(clamped);
  std::nth_element(rates.begin(), rates.begin() + static_cast<std::ptrdiff_t>(rates.size() / 2), rates.end());
  const double median = rates[rates.size() / 2];
  INFO("median " << median << " of " << rates.size());
  CHECK(std::fabs(median / 100 - 1) < 0.02);
}

TEST_CASE("the label cut changes labels only") {
  const uint32_t W = 64, H = 32, n = W * H;
  const std::vector<sensor::Waveform> groups{sensor::make_constant_waveform()};
  std::vector<float> irr0(n, 0.5f);
  std::vector<uint32_t> ids(n);
  std::vector<int64_t> t;
  std::vector<float> irr;
  for (uint32_t i = 0; i < n; ++i) {
    ids[i] = i;
    t.insert(t.end(), {1'000'000, 40'000'000});
    irr.insert(irr.end(), {0.5f, i % 2 ? 2.0f : 0.5f});  // half the pixels see a 4x ramp
  }
  const auto run = [&](double cut) {
    sensor::SensorParams p = sensor::preset("prophesee_gen3");
    p.illuminance_scale = 400;
    p.label_share_cut = cut;
    sensor::SensorModel m(p, W, H, 1, groups, 0);
    m.init(0, irr0.data());
    return m.advance(ids.data(), n, t.data(), irr.data(), 2, nullptr, 50'000'000);
  };
  const sensor::Events a = run(0.5);
  for (const double cut : {0.3, 0.7}) {
    const sensor::Events b = run(cut);
    INFO("cut " << cut);
    REQUIRE(b.size() == a.size());
    CHECK(b.t_cross_ns == a.t_cross_ns);
    CHECK(b.t_reported_ns == a.t_reported_ns);
    CHECK((b.x == a.x && b.y == a.y && b.p == a.p && b.flags == a.flags));
    CHECK((b.ds == a.ds && b.dn == a.dn && b.dl == a.dl));
    CHECK(b.label != a.label);
  }
}

TEST_CASE("one PSF source: the support at the defaults is the old constant 0.95") {
  const render::SceneSettings s{};
  CHECK(render::support_px(s) == 0.5 * 1.0 + 3.0 * 0.15);
  CHECK(render::support_px(s) == 0.95);
  CHECK(rng::PixelResponse(s.aperture_px, s.blur_sigma_px, s.truncation_sigmas).x_max() == render::support_px(s));
}
