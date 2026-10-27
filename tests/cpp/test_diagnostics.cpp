// Resolver diagnostics (docs/design/phase3/ui.md §3.10, §8.1; UI-4a): T3, every validation kind of P0_PARAMS §2.4
// as a Diagnostic with its kind, key, camera, layer and keys, every SensorParams::validate rule naming its keys, and
// DiagnosticError a std::invalid_argument with the message it always had; T4, derived_from checked by perturbing every
// row against every derived accuracy value; explain's layer stack and relevant's visibility.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "config/diagnostics.h"
#include "config/resolve.h"
#include "config/schema.h"
#include "core/time/time.h"
#include "sensor/param_fields.h"
#include "sensor/params.h"

using namespace balla;
using config::DiagKind;
using config::Diagnostic;
using config::Param;
using config::Scope;
using config::Severity;
using config::Type;
using config::Value;

namespace {

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

// A run configuration on prophesee_gen3 cameras: the top-level `sections` and the default block's extra keys.
config::Inputs run_inputs(Map sections, Map defaults = {}) {
  Map block{{"sensor.family", str("prophesee_gen3")}, {"photometry.illuminance_scale", leaf(200.0)}};
  for (auto& d : defaults) block.push_back(std::move(d));
  Map doc{{"format", str("ballasim.run/2")}, {"output", mapping({{"dir", str("out")}})},
          {"cameras", mapping({{"default", mapping(std::move(block))}})}};
  for (auto& s : sections) doc.push_back(std::move(s));
  config::Inputs in;
  in.config = mapping(std::move(doc));
  return in;
}

config::Node profile(const char* id, const char* mount, const char* key, Value v) {
  const config::Node entry = mapping({{"value", leaf(std::move(v))}, {"status", str("prior")}, {"source", str("t3")}});
  return mapping({{"format", str("ballasim.profile/1")}, {"id", str(id)}, {"mount", str(mount)},
                  {"description", str("test")}, {"values", mapping({{key, entry}})}});
}

// A supplied bundle with 64 x 48 cameras (15 um pixels, f/2), `left` and the `more`.
config::BundleInfo cameras(std::vector<std::string> more = {}) {
  config::BundleInfo b;
  b.cameras = {{"left", "prophesee_gen3", 64, 48, 0, 15e-6, 2.0}};
  for (const std::string& m : more) b.cameras.push_back({m, "prophesee_gen3", 64, 48, 0, 15e-6, 2.0});
  b.t_end_ns = 10 * time::kWindowNs;
  b.supplied = true;
  b.bundle_digest = std::string(64, 'b');
  b.engine_digest = std::string(64, 'e');
  return b;
}

// The diagnostic resolve() throws, which must be a DiagnosticError (a std::invalid_argument, its message the text).
Diagnostic error_of(const std::function<void()>& f) {
  try {
    f();
  } catch (const config::DiagnosticError& e) {
    CHECK(std::string(e.what()) == e.diag.message);
    CHECK(e.diag.severity == Severity::kError);
    const std::invalid_argument& base = e;
    CHECK(std::string(base.what()) == e.diag.message);
    return e.diag;
  }
  FAIL("no DiagnosticError");
  return {};
}

Diagnostic resolve_error(const config::Inputs& in, const config::BundleInfo& b = cameras()) {
  return error_of([&] { config::resolve(in, b); });
}

bool has(const std::vector<std::string>& v, const std::string& k) {
  return std::find(v.begin(), v.end(), k) != v.end();
}

// The one warning diagnostic of `kind`, checked against its warning text.
Diagnostic warning_of(const config::Resolved& r, DiagKind kind) {
  REQUIRE(r.diagnostics.size() == r.warnings.size());
  std::optional<Diagnostic> found;
  for (size_t i = 0; i < r.diagnostics.size(); ++i) {
    CHECK(r.diagnostics[i].message == r.warnings[i]);
    CHECK(r.diagnostics[i].severity == Severity::kWarning);
    if (r.diagnostics[i].kind == kind) found = r.diagnostics[i];
  }
  REQUIRE(found.has_value());
  return *found;
}

}  // namespace

TEST_CASE("T3: each P0 validation kind as a diagnostic: kind, key, camera, layer, keys") {
  // unknown key, with the nearest
  Diagnostic d = resolve_error(run_inputs({}, {{"sensor.thetaoff", leaf(0.3)}}));
  CHECK((d.kind == DiagKind::kUnknownKey && d.key == "sensor.thetaoff" && d.camera == "default" &&
         d.layer == "config" && d.nearest == "sensor.theta_off"));
  CHECK(d.message.find("did you mean sensor.theta_off?") != std::string::npos);
  // unknown camera block
  config::Inputs in = run_inputs({});
  in.cli = mapping({{"cameras.right.sensor.theta_on", leaf(0.3)}});
  d = resolve_error(in);
  CHECK((d.kind == DiagKind::kUnknownCamera && d.camera == "right" && d.layer == "cli"));
  // wrong scope, both ways
  d = resolve_error(run_inputs({}, {{"accuracy.d_px", leaf(0.5)}}));
  CHECK((d.kind == DiagKind::kWrongScope && d.key == "accuracy.d_px" && d.camera == "default" && d.layer == "config"));
  in = run_inputs({});
  in.cli = mapping({{"sensor.theta_on", leaf(0.3)}});
  d = resolve_error(in);
  CHECK((d.kind == DiagKind::kWrongScope && d.key == "sensor.theta_on" && d.camera.empty() && d.layer == "cli"));
  // type and enum
  d = resolve_error(run_inputs({}, {{"sensor.reset_mode", str("fast")}}));
  CHECK((d.kind == DiagKind::kType && d.key == "sensor.reset_mode" && d.camera == "default"));
  d = resolve_error(run_inputs({{"accuracy", mapping({{"noise_leaf_level", leaf(5.5)}})}}));
  CHECK((d.kind == DiagKind::kType && d.key == "accuracy.noise_leaf_level" && d.camera.empty()));
  // hard range, naming the profile layer
  in = run_inputs({{"profiles", leaf(std::vector<std::string>{"ex/a@v0"})}});
  in.profiles.emplace("ex/a@v0", profile("ex/a@v0", "root", "cameras.default.sensor.hot_fraction", 1.2));
  d = resolve_error(in);
  CHECK((d.kind == DiagKind::kHardRange && d.key == "sensor.hot_fraction" && d.camera == "default" &&
         d.layer == "profile:ex/a@v0"));
  // soft range (a warning on a user layer)
  in = run_inputs({});
  in.cli = mapping({{"cameras.left.sensor.theta_off", leaf(0.6)}});
  d = warning_of(config::resolve(in, cameras()), DiagKind::kSoftRange);
  CHECK((d.key == "sensor.theta_off" && d.camera == "left" && d.layer == "cli"));
  // required (the row's hint) and the family (D2)
  in = run_inputs({});
  in.config.map[2].second.map[0].second.map.pop_back();  // drop photometry.illuminance_scale
  d = resolve_error(in);
  CHECK((d.kind == DiagKind::kRequired && d.key == "photometry.illuminance_scale" && d.camera == "left" &&
         d.layer.empty()));
  in = run_inputs({});
  in.config.map[2].second.map[0].second.map.erase(in.config.map[2].second.map[0].second.map.begin());
  d = resolve_error(in);
  CHECK((d.kind == DiagKind::kFamilyMissing && d.key == "sensor.family" && d.camera == "left"));
  // set but inert, naming the condition's key
  d = warning_of(config::resolve(run_inputs({}, {{"sensor.hot_rate_median_hz", leaf(50.0)}}), cameras()),
                 DiagKind::kInert);
  CHECK((d.key == "sensor.hot_rate_median_hz" && d.camera == "left" && d.layer == "config" &&
         has(d.keys, "sensor.hot_rate_law")));
  // a sensor cross-key rule, on every key it reads, prefixed with the camera
  d = resolve_error(run_inputs({}, {{"sensor.tau_sf", leaf(25e-6)}, {"sensor.tau_o", leaf(25e-6)}}));
  CHECK((d.kind == DiagKind::kCrossKey && d.camera == "left" && d.layer == "config" &&
         d.keys == std::vector<std::string>{"sensor.tau_sf", "sensor.tau_o"}));
  CHECK(d.message == "cameras.left: SensorParams: tau_sf and tau_o must differ by at least 10 % (the larger >= 1.1 x "
                     "the smaller)");
  // run level
  d = resolve_error(
      run_inputs({{"run", mapping({{"t_start_ns", leaf(int64_t{5000})}, {"t_end_ns", leaf(int64_t{10})}})}}));
  CHECK((d.kind == DiagKind::kRunLevel && d.key == "run.t_end_ns" && has(d.keys, "run.t_start_ns") &&
         d.layer == "config"));
  // derived against explicit (a warning)
  d = warning_of(config::resolve(run_inputs({{"accuracy", mapping({{"d_px", leaf(0.5)}})}}), cameras()),
                 DiagKind::kDerivedConflict);
  CHECK((d.key == "accuracy.d_px" && d.layer == "config" && has(d.keys, "accuracy.n_subsamples")));
  // run-wide render keys disagree
  in = run_inputs({});
  in.config.map[2].second.map.emplace_back("right", mapping({{"optics.blur_sigma_px", leaf(0.25)}}));
  d = resolve_error(in, cameras({"right"}));
  CHECK((d.kind == DiagKind::kRunWideDisagree && d.key == "optics.blur_sigma_px" && d.camera == "right" &&
         d.layer == "config"));
  // profile id, mount and format
  d = resolve_error(run_inputs({{"profiles", leaf(std::vector<std::string>{"no-version"})}}));
  CHECK(d.kind == DiagKind::kProfile);
  in = run_inputs({{"profiles", leaf(std::vector<std::string>{"ex/c@v0"})}});
  in.profiles.emplace("ex/c@v0", profile("ex/c@v0", "camera", "sensor.theta_on", 0.3));
  d = resolve_error(in);
  CHECK((d.kind == DiagKind::kProfile && d.layer == "profile:ex/c@v0"));
  in = run_inputs({});
  in.config.map.erase(in.config.map.begin());  // no format: key (format 1)
  d = resolve_error(in);
  CHECK((d.kind == DiagKind::kFormat && d.layer == "config"));
  CHECK(d.message.find("a format-1 run configuration") != std::string::npos);
  // a derived value outside its row: the derivation's inputs are its keys
  d = resolve_error(run_inputs({}, {{"sensor.readout_quantum_s", leaf(0.01)}, {"sensor.readout_delay_s", leaf(0.02)}}));
  CHECK((d.kind == DiagKind::kHardRange && d.key == "accuracy.min_dt_ns" && d.layer == "accuracy:fidelity" &&
         has(d.keys, "sensor.readout_quantum_s")));
  // resolve_scene and scene_digest throw diagnostics too
  d = error_of(
      [] { config::resolve_scene(run_inputs({{"scene", mapping({{"trajectory.order", leaf(int64_t{5})}})}})); });
  CHECK((d.kind == DiagKind::kHardRange && d.key == "scene.trajectory.order" && d.layer == "config"));
  d = error_of([] { config::scene_digest({{"scene.nope", Value{1.0}}}, {}); });
  CHECK((d.kind == DiagKind::kUnknownKey && d.key == "scene.nope"));
}

TEST_CASE("T3: the warnings are diagnostics, one per warning, in order") {
  const config::Resolved r = config::resolve(run_inputs({}), cameras());
  REQUIRE(r.diagnostics.size() == r.warnings.size());
  REQUIRE(!r.warnings.empty());
  for (size_t i = 0; i < r.warnings.size(); ++i) {
    CHECK(r.diagnostics[i].message == r.warnings[i]);
    CHECK(r.diagnostics[i].severity == Severity::kWarning);
  }
  const Diagnostic d = warning_of(r, DiagKind::kFlaggedDefault);
  CHECK((d.key.empty() && !d.keys.empty()));
  for (const std::string& k : d.keys) CHECK(config::find(k) != nullptr);
}

TEST_CASE("T3: collect mode carries each missing key's diagnostic") {
  config::Inputs in = run_inputs({});
  auto& block = in.config.map[2].second.map[0].second.map;
  block.erase(block.begin());  // no family
  config::Missing missing;
  config::resolve(in, cameras({"right"}), &missing);
  REQUIRE(missing.size() == 2);
  for (const config::MissingKey& m : missing) {
    CHECK(m.diag.kind == DiagKind::kFamilyMissing);
    CHECK(m.key == "cameras." + m.diag.camera + ".sensor.family");
    CHECK(m.diag.message.find("sensor.family is required") != std::string::npos);
  }
}

TEST_CASE("T3: every SensorParams::validate cross-key rule names its keys (enumerated by triggering)") {
  struct Case {
    const char* name;
    std::function<void(sensor::SensorParams&)> edit;
    std::vector<std::string> keys;
  };
  const std::vector<Case> cases = {
      {"poles", [](auto& s) { s.tau_sf = s.tau_o; }, {"sensor.tau_sf", "sensor.tau_o"}},
      {"sigma_o", [](auto& s) { s.noise = true, s.sigma_o = 0.05, s.tau_o = 0.5e-6, s.tau_sf = 100e-6; },
       {"sensor.sigma_o", "sensor.tau_o", "sensor.noise"}},
      {"theta_on", [](auto& s) { s.theta_on = s.theta_floor; }, {"sensor.theta_on", "sensor.theta_floor"}},
      {"theta_off", [](auto& s) { s.theta_off = s.theta_floor; }, {"sensor.theta_off", "sensor.theta_floor"}},
      {"eps_ci", [](auto& s) { s.eps_ci = s.theta_floor; }, {"sensor.eps_ci", "sensor.theta_floor"}},
      {"hot rates", [](auto& s) { s.hot_rate_max_hz = 0.5 * s.hot_rate_min_hz; },
       {"sensor.hot_rate_max_hz", "sensor.hot_rate_min_hz"}},
      {"hot refractory", [](auto& s) { s.hot_fraction = 0.01, s.refractory_s = 0.5; },
       {"sensor.hot_rate_max_hz", "sensor.refractory_s", "sensor.hot_fraction"}},
      {"readout", [](auto& s) { s.readout_quantum_s = 1e-3, s.readout_delay_s = 0.5e-3; },
       {"sensor.readout_delay_s", "sensor.readout_quantum_s"}},
      {"tau_hp floor", [](auto& s) { s.tau_hp = 0.5e-3; }, {"sensor.tau_hp", "sensor.tau_o", "sensor.tau_sf"}},
      {"tau_hp cascade", [](auto& s) { s.tau_hp = 0.1, s.photoreceptor = sensor::Photoreceptor::kInstantaneous; },
       {"sensor.tau_hp", "sensor.photoreceptor"}},
      {"limiter rates",
       [](auto& s) { s.limiter = sensor::LimiterMode::kEpoch, s.limiter_rate1_hz = 2 * s.limiter_rate0_hz; },
       {"sensor.limiter_rate1_hz", "sensor.limiter_rate0_hz"}},
      {"limiter depth", [](auto& s) { s.limiter = sensor::LimiterMode::kEpoch, s.limiter_depth = 1; },
       {"sensor.limiter_depth", "sensor.limiter_rate0_hz", "sensor.readout_quantum_s"}},
      {"limiter phase",
       [](auto& s) { s.limiter = sensor::LimiterMode::kEpoch, s.limiter_phase_s = s.limiter_period_s; },
       {"sensor.limiter_phase_s", "sensor.limiter_period_s"}},
  };
  std::set<std::string> messages;
  for (const Case& c : cases) {
    INFO(c.name);
    sensor::SensorParams s = sensor::preset("prophesee_gen3");
    CHECK_NOTHROW(s.validate());
    c.edit(s);
    const Diagnostic d = error_of([&] { s.validate(); });
    CHECK(d.kind == DiagKind::kCrossKey);
    CHECK(d.keys == c.keys);
    CHECK(d.key == c.keys.front());
    for (const std::string& k : d.keys) CHECK(config::find(k) != nullptr);
    CHECK(d.message.rfind("SensorParams: ", 0) == 0);
    messages.insert(d.message);
  }
  CHECK(messages.size() == cases.size());  // 13 distinct rules: 10 at origin/main and the rate limiter's 3
}

namespace {

// T4's bundle: one camera with a pixel pitch (lens_blur and diffraction need one).
config::BundleInfo t4_bundle() { return cameras(); }

// The run on T4's profile with the base's run keys (full keys, here accuracy.*) and default-block keys.
config::Inputs t4_inputs(const char* profile, const Map& run, const Map& cam) {
  Map accuracy{{"profile", str(profile)}};
  for (const auto& [k, v] : run) {
    REQUIRE(k.rfind("accuracy.", 0) == 0);
    accuracy.emplace_back(k.substr(9), v);
  }
  return run_inputs({{"accuracy", mapping(std::move(accuracy))}}, cam);
}

// The values a perturbation tries: the range ends, the base value scaled, 1, null where nullable, the other choices
// and the other bool. Only values the row admits.
std::vector<Value> candidates(const Param& p, const Value& base) {
  std::vector<Value> out;
  const auto add = [&](Value v) {
    if (v == base || std::find(out.begin(), out.end(), v) != out.end()) return;
    try {
      config::check_value(p, v, p.key);
    } catch (const std::invalid_argument&) {
      return;
    }
    out.push_back(std::move(v));
  };
  if (p.nullable) add(Value{});
  switch (p.type) {
    case Type::kBool:
      if (const auto* b = std::get_if<bool>(&base)) add(!*b);
      break;
    case Type::kEnum:
      for (const std::string& c : p.choices) add(c);
      break;
    case Type::kFloat:
    case Type::kInt: {
      std::vector<double> xs{1.0};
      for (const config::Range* r : {&p.soft, &p.hard}) {
        if (std::isfinite(r->lo)) xs.push_back(r->lo);
        if (std::isfinite(r->hi)) xs.push_back(r->hi);
      }
      double b = std::numeric_limits<double>::quiet_NaN();
      if (const auto* d = std::get_if<double>(&base)) b = *d;
      if (const auto* i = std::get_if<int64_t>(&base)) b = static_cast<double>(*i);
      if (std::isfinite(b)) {
        for (const double f : {0.01, 0.1, 0.5, 0.9, 1.1, 2.0, 10.0, 100.0}) xs.push_back(b * f);
        if (p.type == Type::kInt) xs.insert(xs.end(), {b - 1, b + 1});
      }
      for (const double x : xs) {
        if (p.type == Type::kInt) {
          if (std::fabs(x) < 9e18) add(static_cast<int64_t>(std::llround(x)));
        } else {
          add(x);
        }
      }
      break;
    }
    case Type::kFloatList:
      if (const auto* l = std::get_if<std::vector<double>>(&base); l && !l->empty()) {
        for (const double f : {0.5, 2.0}) {
          std::vector<double> v = *l;
          for (double& x : v) x *= f;
          add(v);
        }
      }
      break;
    default:
      break;  // strings, paths and their lists, integer lists: no derivation reads one
  }
  return out;
}

// The bases of T4: where each listed input can move its derived value (near a threshold of the step functions).
std::vector<std::pair<Map, Map>> bases(const std::string& profile) {
  std::vector<std::pair<Map, Map>> out{{{}, {}}};
  // a quiet sensor (sigma_v,min < 0.02, so the tolerances follow the noise below the cap)
  out.push_back({{}, {{"sensor.sigma_o", leaf(0.002)}, {"sensor.tau_o", leaf(25e-6)}}});
  if (profile == "fidelity" || profile == "balanced") {
    out.push_back({{{"accuracy.design_contrast", leaf(0.1)}}, {}});  // N_step < 16: the band raises N_base to 64
  }
  if (profile == "fidelity") {
    // the pixel response near 0.75 px: diffraction on at f/2, then off at f/8
    out.push_back({{}, {{"sensor.aperture_px", leaf(0.6)}, {"optics.blur_sigma_px", leaf(0.14)},
                        {"optics.diffraction", leaf(true)}}});
    out.push_back({{}, {{"sensor.aperture_px", leaf(0.6)}, {"optics.blur_sigma_px", leaf(0.149)},
                        {"optics.f_number", leaf(8.0)}}});
  }
  return out;
}

}  // namespace

TEST_CASE("T4: derived_from by perturbation: each listed input moves its value, no other row moves any") {
  const config::BundleInfo b = t4_bundle();
  for (const char* profile : {"phase1", "fidelity", "balanced", "reference"}) {
    INFO(profile);
    const auto table = config::derived_from(profile);
    std::map<std::string, std::set<std::string>> moved;  // derived key -> the rows that moved it
    for (const auto& [run, cam] : bases(profile)) {
      const config::Inputs base = t4_inputs(profile, run, cam);
      const config::Resolved r0 = config::resolve(base, b);
      for (const Param& p : config::schema()) {
        if (p.key == "sensor.family" || p.key == "accuracy.profile") continue;  // whole tables, not one row
        const bool camera = p.scope == Scope::kCamera;
        const config::Entry* e0 = r0.find(p.key, camera ? "left" : "");
        const Value v0 = e0 != nullptr ? e0->value : Value{};
        for (const Value& v : candidates(p, v0)) {
          config::Inputs in = base;
          in.cli = mapping({{(camera ? "cameras.left." : "") + p.key, leaf(v)}});
          std::optional<config::Resolved> r;
          try {
            r = config::resolve(in, b);
          } catch (const std::invalid_argument&) {
            continue;
          }
          for (const auto& [d, inputs] : table) {
            if (d == p.key) continue;
            const config::Entry* a = r0.find(d);
            const config::Entry* c = r->find(d);
            REQUIRE((a != nullptr && c != nullptr));
            if (a->value != c->value) moved[d].insert(p.key);
          }
        }
      }
    }
    for (const auto& [d, inputs] : table) {
      INFO(d);
      CHECK(config::find(d) != nullptr);
      for (const std::string& k : inputs) {
        INFO(k);
        CHECK(config::find(k) != nullptr);
        CHECK(moved[d].count(k) == 1);  // a listed input moves it
      }
      for (const std::string& k : moved[d]) {
        INFO(k);
        CHECK(has(inputs, k));  // no other row moves it
      }
    }
  }
  CHECK_THROWS_AS(config::derived_from("nope"), config::DiagnosticError);
}

TEST_CASE("explain: the default, the derivation and every layer that sets a key, lowest first") {
  config::Inputs in = run_inputs({{"profiles", leaf(std::vector<std::string>{"ex/a@v0"})}},
                                 {{"sensor.theta_off", leaf(0.3)}});
  in.profiles.emplace("ex/a@v0", profile("ex/a@v0", "root", "cameras.default.sensor.theta_off", 0.28));
  in.cli = mapping({{"cameras.left.sensor.theta_off", leaf(0.31)}});
  const config::Resolved r = config::resolve(in, cameras());
  const auto e = config::explain(in, r, "sensor.theta_off", "left");
  REQUIRE(e.size() == 4);
  CHECK((e[0].layer == "prior:prophesee_gen3" && e[0].status == "prior" && !e[0].source.empty()));
  CHECK((e[1].layer == "profile:ex/a@v0" && e[1].value == Value{0.28} && e[1].status == "prior" &&
         e[1].source == "t3"));
  CHECK((e[2].layer == "config" && e[2].value == Value{0.3} && e[2].status == "user"));
  CHECK((e[3].layer == "cli" && e[3].value == Value{0.31} && e[3].source == "--set"));
  CHECK(r.find("sensor.theta_off", "left")->layer == e.back().layer);
  // a derived accuracy key: the literal default, then fidelity's formula with its value
  const auto n = config::explain(in, r, "accuracy.n_subsamples", "");
  REQUIRE(n.size() == 2);
  CHECK((n[0].layer == "default" && n[1].layer == "accuracy:fidelity" && !n[1].source.empty() &&
         n[1].value == r.find("accuracy.n_subsamples")->value));
  // a bundle value
  const auto c = config::explain(in, r, "sensor.clock_offset_ns", "left");
  REQUIRE(c.size() == 1);
  CHECK(c[0].layer == "bundle");
  CHECK_THROWS_AS(config::explain(in, r, "sensor.theta_off", ""), config::DiagnosticError);
  CHECK_THROWS_AS(config::explain(in, r, "sensor.theta_off", "right"), config::DiagnosticError);
  CHECK_THROWS_AS(config::explain(in, r, "sensor.thetaoff", "left"), config::DiagnosticError);
}

TEST_CASE("relevant: a row's when on the resolved values, the camera's first") {
  config::Inputs in = run_inputs({});
  config::Resolved r = config::resolve(in, cameras());
  CHECK(config::relevant("sensor.tau_hp", r, "left"));                  // when photoreceptor: cascade
  CHECK_FALSE(config::relevant("sensor.hot_rate_median_hz", r, "left"));  // when hot_rate_law: log_normal
  CHECK(config::relevant("accuracy.design_threshold", r, ""));            // fidelity
  CHECK(config::relevant("accuracy.design_threshold", r, "left"));        // a run row read in a camera's context
  CHECK(config::relevant("sensor.theta_on", r, "left"));                  // no condition
  in.cli = mapping({{"cameras.left.sensor.photoreceptor", str("instantaneous")}, {"accuracy.profile", str("phase1")}});
  r = config::resolve(in, cameras());
  CHECK_FALSE(config::relevant("sensor.tau_hp", r, "left"));
  CHECK_FALSE(config::relevant("accuracy.design_threshold", r, ""));
  CHECK_THROWS_AS(config::relevant("sensor.tau_hp", r, ""), config::DiagnosticError);
  CHECK_THROWS_AS(config::relevant("sensor.tau_hp", r, "right"), config::DiagnosticError);
}
