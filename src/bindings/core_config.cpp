// ballasim._core.config: the parameter schema as plain values and the resolver (docs/dev/P0_PARAMS.md §1.4, §2):
// profiles_needed, resolve_scene and resolve take the documents Python read (plain values) and return a Resolved.
// Diagnostics (docs/design/phase3/ui.md §3.10, UI-4a): a resolver error is DiagnosticError, a ValueError whose
// `diagnostic` is a dict; Resolved.diagnostics holds the warnings as such dicts; explain, relevant and derived_from
// serve the forms.
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

#include "bindings/config_inputs.h"
#include "bindings/config_value.h"
#include "bindings/core_bindings.h"
#include "config/diagnostics.h"
#include "config/resolve.h"
#include "config/tier.h"
#include "config/schema.h"
#include "sensor/param_fields.h"

namespace nb = nanobind;

namespace balla::bindings {

namespace {

nb::object range_dict(const config::Range& r) {
  if (r.unset()) return nb::none();
  nb::dict d;
  d["lo"] = std::isfinite(r.lo) ? nb::object(nb::float_(r.lo)) : nb::none();
  d["hi"] = std::isfinite(r.hi) ? nb::object(nb::float_(r.hi)) : nb::none();
  d["lo_open"] = r.lo_open;
  d["hi_open"] = r.hi_open;
  return d;
}

// Every field of P0_PARAMS §1.1, plus the C++ member (field) and the apply kind.
nb::dict row_dict(const config::Param& p) {
  nb::dict d;
  d["key"] = p.key;
  d["field"] = p.field.empty() ? nb::none() : nb::object(nb::str(p.field.c_str()));
  d["type"] = config::name(p.type);
  d["nullable"] = p.nullable;
  d["length"] = p.length == 0 ? nb::none() : (p.length < 0 ? nb::object(nb::str("any")) : nb::object(nb::int_(p.length)));
  d["unit"] = p.unit;
  d["default"] = config::name(p.def);
  d["value"] = p.def == config::Default::kLiteral ? to_py(p.value) : nb::none();
  d["formula"] = p.formula.empty() ? nb::none() : nb::object(nb::str(p.formula.c_str()));
  nb::list choices;
  for (const std::string& c : p.choices) choices.append(c);
  d["choices"] = choices;
  d["hard"] = range_dict(p.hard);
  d["soft"] = range_dict(p.soft);
  d["scale"] = config::name(p.scale);
  d["status"] = p.status ? nb::object(nb::str(config::name(*p.status))) : nb::none();
  d["evimo2_derived"] = p.evimo2_derived;
  d["source"] = p.source;
  d["help"] = p.help;
  d["class"] = config::name(p.cls);
  d["scope"] = config::name(p.scope);
  d["stage"] = config::name(p.stage);
  // when: every condition holds for the row to matter; also_when: or every one of these does (None: none).
  const auto conditions = [](const std::vector<config::When>& ws) -> nb::object {
    if (ws.empty()) return nb::none();
    nb::dict w;
    for (const config::When& c : ws) w[c.key.c_str()] = to_py(c.value);
    return w;
  };
  d["when"] = conditions(p.when);
  d["also_when"] = conditions(p.also_when);
  d["target"] = p.target == config::Target::kNone ? nb::none() : nb::object(nb::str(config::name(p.target)));
  d["apply"] = p.target == config::Target::kNone ? nb::none() : nb::object(nb::str(p.custom ? "custom" : "member"));
  return d;
}

const char* const kStageNames[] = {"scene", "render", "schedule", "sensor", "output"};

nb::object str_or_none(const std::string& s) { return s.empty() ? nb::none() : nb::object(nb::str(s.c_str())); }

// A Diagnostic as a dict (ui.md §3.6 "diagnostics"): kind, severity, key, camera, layer (None when empty), keys,
// message, nearest.
nb::dict diag_dict(const config::Diagnostic& d) {
  nb::dict o;
  o["kind"] = config::name(d.kind);
  o["severity"] = config::name(d.severity);
  o["key"] = str_or_none(d.key);
  o["camera"] = str_or_none(d.camera);
  o["layer"] = str_or_none(d.layer);
  nb::list keys;
  for (const std::string& k : d.keys) keys.append(k);
  o["keys"] = keys;
  o["message"] = d.message;
  o["nearest"] = str_or_none(d.nearest);
  return o;
}

nb::dict entries_dict(const config::Entries& list, bool record) {
  nb::dict out;
  for (const auto& [k, e] : list) {
    if (!record) {
      out[k.c_str()] = to_py(e.value);
      continue;
    }
    nb::dict r;
    r["value"] = to_py(e.value);
    r["layer"] = e.layer;
    r["status"] = e.status;
    if (!e.note.empty()) r["note"] = e.note;
    out[k.c_str()] = r;
  }
  return out;
}

// A resolved run (P0_PARAMS §1.4): values, record, warnings, digests, span.
struct PyResolved {
  config::Resolved r;

  nb::dict as_dict(bool record) const {
    nb::dict d, cams;
    d["run"] = entries_dict(r.run, record);
    for (const auto& [name, list] : r.cameras) cams[name.c_str()] = entries_dict(list, record);
    d["cameras"] = cams;
    return d;
  }
  size_t camera_index(const std::string& camera) const {
    for (size_t i = 0; i < r.cameras.size(); ++i) {
      if (r.cameras[i].first == camera) return i;
    }
    throw nb::key_error(("not a resolved camera: " + camera).c_str());
  }
  std::string digest(const std::string& stage, const std::string& camera) const {
    const size_t i = camera_index(camera);
    for (int s = 0; s < 5; ++s) {
      if (stage == kStageNames[s]) {
        if (i >= r.digests.size()) throw std::invalid_argument("digests need resolve() (a bundle)");
        return r.digests[i][static_cast<size_t>(s)];
      }
    }
    throw std::invalid_argument("stage must be one of scene, render, schedule, sensor, output");
  }
  nb::dict digests() const {
    nb::dict out;
    for (size_t i = 0; i < r.digests.size(); ++i) {
      nb::dict d;
      for (int s = 0; s < 5; ++s) d[kStageNames[s]] = r.digests[i][static_cast<size_t>(s)];
      out[r.cameras[i].first.c_str()] = d;
    }
    return out;
  }
  nb::object get(const std::string& key, std::optional<std::string> camera) const {
    const config::Entry* e = r.find(key, camera.value_or(""));
    if (e == nullptr) throw nb::key_error((key + (camera ? " of camera " + *camera : std::string()) + ": not resolved").c_str());
    return to_py(e->value);
  }
};

}  // namespace

void bind_config(nb::module_& m) {
  nb::module_ c = m.def_submodule("config", "The parameter schema (docs/dev/P0_PARAMS.md, PARAMS_REFERENCE.md).");
  // DiagnosticError: a ValueError (so ConfigError, the CLI's exit 2 and every `except ValueError` stay), whose
  // `diagnostic` attribute is the error's Diagnostic as a dict (ui.md §3.10).
  PyObject* diag_type = PyErr_NewException("ballasim._core.config.DiagnosticError", PyExc_ValueError, nullptr);
  if (diag_type == nullptr) throw nb::python_error();
  c.attr("DiagnosticError") = nb::steal(diag_type);
  nb::register_exception_translator(
      [](const std::exception_ptr& p, void* payload) {
        try {
          std::rethrow_exception(p);
        } catch (const config::DiagnosticError& e) {
          auto* type = static_cast<PyObject*>(payload);
          nb::object exc = nb::steal(PyObject_CallFunction(type, "s", e.what()));
          if (!exc.is_valid()) return;  // the call set the Python error
          if (PyObject_SetAttrString(exc.ptr(), "diagnostic", diag_dict(e.diag).ptr()) != 0) return;
          PyErr_SetObject(type, exc.ptr());
        }
      },
      diag_type);
  c.def(
      "schema",
      [] {
        nb::list out;
        for (const config::Param& p : config::schema()) out.append(row_dict(p));
        return out;
      },
      "Every parameter row as a dict: key, field, type, nullable, length, unit, default (literal, required, family, "
      "derived), value (literal defaults), formula, choices, hard, soft, scale, status, evimo2_derived, source, help, "
      "class, scope, stage, when, target, apply.");
  c.def(
      "families",
      [] {
        nb::list out;
        for (const sensor::Family& f : sensor::families()) {
          nb::dict d;
          d["name"] = f.name;
          d["base"] = f.base.empty() ? nb::none() : nb::object(nb::str(f.base.c_str()));
          d["help"] = f.help;
          nb::list priors;
          for (const config::Param& p : config::schema()) {
            if (p.def != config::Default::kFamily) continue;
            bool inherited = false;
            const sensor::Prior* v = sensor::find_prior(f.name, p.key, &inherited);
            if (!v) throw std::logic_error("sensor family " + f.name + " has no prior for " + p.key);
            nb::dict e;
            e["key"] = v->key;
            e["value"] = to_py(v->value);
            e["status"] = config::name(v->status);
            e["source"] = v->source;
            e["evimo2_derived"] = v->evimo2_derived;
            e["no_literature"] = v->no_literature;
            e["inherited"] = inherited;
            priors.append(e);
          }
          d["priors"] = priors;
          out.append(d);
        }
        return out;
      },
      "The sensor families: name, base, help, and the complete prior table in schema order (key, value, status, "
      "source, evimo2_derived, no_literature: off for want of a literature value (D4c), inherited: taken from the "
      "base, unvalidated for this family).");
  c.def(
      "sensor_key",
      [](const std::string& field) {
        const config::Param* r = sensor::param_row(field);
        if (!r) throw std::invalid_argument(sensor::unknown_field(field));
        return r->key;
      },
      nb::arg("field"),
      "The schema key of a SensorParams field (the _sensor attribute and format-1 override names); ValueError for "
      "other names, naming the replacement of a removed field.");
  c.def(
      "default",
      [](const std::string& key, std::optional<std::string> family) {
        return to_py(config::default_value(key, family.value_or("")));
      },
      nb::arg("key"), nb::arg("family") = nb::none(),
      "The default of a key: a literal row's value, or a family row's prior for `family`. ValueError for unknown "
      "keys and for required or derived defaults.");
  c.def(
      "check",
      [](const std::string& key, nb::handle value) {
        const config::Param* p = config::find(key);
        if (p == nullptr) throw std::invalid_argument("unknown key " + key);
        config::check_value(*p, from_py(value, *p), key);
      },
      nb::arg("key"), nb::arg("value").none(),
      "Checks a value against its row (type, null, choices, list length, finite and inside the hard range); "
      "ValueError naming the key otherwise. For APIs that take schema values outside a run (the EVIMO2 adapter).");
  c.def(
      "scene_digest",
      [](const nb::dict& values, const nb::dict& facts) {
        config::ValueMap typed;
        for (auto [k, v] : values) {
          const auto key = nb::cast<std::string>(k);
          const config::Param* p = config::find(key);
          if (p == nullptr) throw std::invalid_argument("scene_digest: unknown key " + key);
          typed.emplace(key, from_py(v, *p));
        }
        std::map<std::string, std::string, std::less<>> f;
        for (auto [k, v] : facts) f.emplace(nb::cast<std::string>(k), nb::cast<std::string>(v));
        return config::scene_digest(typed, f);
      },
      nb::arg("values"), nb::arg("facts"),
      "digest(scene) of a source built from its scene keys (docs/dev/P0_PARAMS.md §1.2, §3.4; the EVIMO2 cache key): "
      "SHA-256 hex of the scene-stage keys among values ({key: value}, e.g. Resolved.values()['run']) and of facts "
      "({name: str}: sequence directory, importer version, source and core digests). Lighting and workers enter no "
      "digest(scene).");
  c.def(
      "accuracy_profiles",
      [] {
        nb::list out;
        for (const config::AccuracyProfile& p : config::accuracy_profiles()) {
          nb::dict d;
          d["name"] = p.name;
          d["help"] = p.help;
          nb::list entries;
          for (const config::AccuracyEntry& e : p.entries) {
            nb::dict ed;
            ed["key"] = e.key;
            ed["value"] = e.formula.empty() ? to_py(e.value) : nb::none();
            ed["formula"] = e.formula.empty() ? nb::none() : nb::object(nb::str(e.formula.c_str()));
            entries.append(ed);
          }
          d["entries"] = entries;
          out.append(d);
        }
        return out;
      },
      "The accuracy profiles (D8): name, help, entries (key, value for phase1, formula for fidelity's derived keys).");

  nb::class_<PyResolved>(c, "Resolved", "A resolved run (docs/dev/P0_PARAMS.md §1.4, §2.6).")
      .def("values", [](const PyResolved& r) { return r.as_dict(false); },
           "Plain values: {run: {key: value}, cameras: {name: {key: value}}} (run, scene and dataset keys under run).")
      .def("record", [](const PyResolved& r) { return r.as_dict(true); },
           "run.json's resolved block: {run: {key: {value, layer, status[, note]}}, cameras: {name: {...}}}.")
      .def_prop_ro("warnings", [](const PyResolved& r) { return r.r.warnings; })
      .def_prop_ro("diagnostics",
                   [](const PyResolved& r) {
                     nb::list out;
                     for (const config::Diagnostic& d : r.r.diagnostics) out.append(diag_dict(d));
                     return out;
                   },
                   "The warnings as diagnostics (docs/design/phase3/ui.md §3.10), one per warning, in order: "
                   "{kind, severity, key, camera, layer, keys, message, nearest}.")
      .def_prop_ro("facts",
                   [](const PyResolved& r) {
                     nb::dict d;
                     for (const auto& [k, v] : r.r.facts) d[k.c_str()] = v;
                     return d;
                   },
                   "Derived values no stage reads (run.json's derived block): with the band N, n_band, theta_band, "
                   "c_cover and eps_switch (P2-B B4a); empty otherwise.")
      .def_prop_ro("cameras",
                   [](const PyResolved& r) {
                     std::vector<std::string> out;
                     for (const auto& cam : r.r.cameras) out.push_back(cam.first);
                     return out;
                   },
                   "The selected cameras, in bundle order.")
      .def_prop_ro("span",
                   [](const PyResolved& r) -> nb::object {
                     if (r.r.t_e <= r.r.t_s) return nb::none();
                     return nb::make_tuple(r.r.t_s, r.r.t_e);
                   },
                   "(t_s, t_e): the whole windows of the run's span (None before resolve()).")
      .def("digest", &PyResolved::digest, nb::arg("stage"), nb::arg("camera"),
           "digest(stage, camera) (P0_PARAMS §1.2): SHA-256 hex of the stage's and the earlier stages' values.")
      .def("digests", &PyResolved::digests, "{camera: {stage: digest}} (run.json stage_digests).")
      .def("get", &PyResolved::get, nb::arg("key"), nb::arg("camera") = nb::none(),
           "The resolved value of a key (camera keys need the camera); KeyError when it has none.");
  c.def(
      "profiles_needed", [](const nb::dict& in) { return config::profiles_needed(inputs_from_py(in)); },
      nb::arg("inputs"),
      "The profile ids the run configuration and the CLI layer name (root list, then camera blocks), each once.");
  c.def(
      "resolve_scene", [](const nb::dict& in) { return PyResolved{config::resolve_scene(inputs_from_py(in))}; },
      nb::arg("inputs"),
      "The scene keys (and output.dir, output.overwrite, execution.scene_workers), every layer checked; no bundle.");
  c.def(
      "resolve",
      [](const nb::dict& in, const nb::dict& bundle) {
        return PyResolved{config::resolve(inputs_from_py(in), bundle_from_py(bundle))};
      },
      nb::arg("inputs"), nb::arg("bundle_info"),
      "Every value of the run on a bundle: bundle_info = {cameras: [{name, sensor, width, height, clock_offset_ns, "
      "pixel_pitch_m, f_number}], "
      "t_start_ns, t_end_ns, device_available, supplied, bundle_digest, engine_digest, low_light (EVIMO2 sources: the "
      "bundle's light mode, or None), shadow_lights and zero_background (the band N's coverage warning; optional)}. "
      "bundle_digest and engine_digest are required str; \"\" leaves the scene or the engine out of the stage "
      "digests.");
  c.def(
      "missing_keys",
      [](const nb::dict& in, const nb::dict& bundle, bool diagnostics) {
        config::Missing missing;
        config::resolve(inputs_from_py(in), bundle_from_py(bundle), &missing);
        nb::list out;
        for (const config::MissingKey& mk : missing) {
          const size_t dot = mk.key.rfind("cameras.", 0) == 0 ? mk.key.find('.', 8) : std::string::npos;
          const config::Param* p = config::find(dot == std::string::npos ? mk.key : mk.key.substr(dot + 1));
          nb::dict d;
          d["key"] = mk.key;
          d["hint"] = mk.hint;
          d["help"] = p != nullptr ? p->help : std::string();
          if (diagnostics) d["diagnostic"] = diag_dict(mk.diag);
          out.append(d);
        }
        return out;
      },
      nb::arg("inputs"), nb::arg("bundle_info"), nb::kw_only(), nb::arg("diagnostics") = false,
      "Collect mode (docs/design/phase3/blender.md §3.11, the add-on's Check): every required key resolve() finds "
      "without a value, at once, as [{key, hint, help}] (key: cameras.<camera>.<key> for a camera key; hint: the rest "
      "of resolve's message; help: the row's; with diagnostics=True also `diagnostic`, the error resolve() would "
      "raise for it: kind required or family_missing, ui.md §3.10). [] when none is missing; any other configuration "
      "error raises DiagnosticError (a ValueError) as resolve() does.");
  c.def(
      "explain",
      [](const nb::dict& in, const PyResolved& r, const std::string& key, std::optional<std::string> camera) {
        nb::list out;
        for (const config::LayerEntry& e : config::explain(inputs_from_py(in), r.r, key, camera.value_or(""))) {
          nb::dict d;
          d["layer"] = e.layer;
          d["value"] = to_py(e.value);
          d["status"] = e.status;
          d["source"] = e.source;
          out.append(d);
        }
        return out;
      },
      nb::arg("inputs"), nb::arg("resolved"), nb::arg("key"), nb::arg("camera") = nb::none(),
      "Every source that offers a key a value (docs/design/phase3/ui.md §3.10), lowest precedence first: [{layer, "
      "value, status, source}] for the row's default (literal or family prior), the derivation, bundle value or "
      "accuracy profile entry that replaces it, and each profile, the configuration and the CLI that set it. "
      "`resolved` is the Resolved of these inputs; record()'s layer names the winner. Camera keys need the camera.");
  c.def(
      "relevant",
      [](const std::string& key, const PyResolved& r, std::optional<std::string> camera) {
        return config::relevant(key, r.r, camera.value_or(""));
      },
      nb::arg("key"), nb::arg("resolved"), nb::arg("camera") = nb::none(),
      "Whether a row matters on resolved values (docs/design/phase3/ui.md §4.9): its when (or also_when) holds on the "
      "camera's values, then the run's; a condition on a key without a value holds. Camera keys need the camera; a "
      "run key may take one (its conditions may read camera keys).");
  c.def(
      "derived_from",
      [](const std::string& profile) {
        nb::dict out;
        for (const auto& [key, inputs] : config::derived_from(profile)) out[key.c_str()] = inputs;
        return out;
      },
      nb::arg("profile"),
      "Per accuracy profile (docs/design/phase3/ui.md §3.6 feeds): {derived accuracy key: [the schema keys its "
      "derivation reads, closed over the derived keys it reads]}. A key set explicitly replaces its own derivation. "
      "DiagnosticError for an unknown profile.");
  c.def("from_record", [](nb::handle record, nb::handle digests) {
    return PyResolved{config::from_record(node_from_py(record, "resolved"), node_from_py(digests, "stage_digests"))};
  }, nb::arg("resolved"), nb::arg("stage_digests"), "A finished run's recorded resolved values and stage digests.");
  const auto cache_facts = [](const nb::dict& facts) {
    config::CacheFacts out;
    if (facts.contains("photometry_mode")) out.photometry_mode = nb::cast<std::string>(facts["photometry_mode"]);
    if (facts.contains("scale")) out.scale = nb::cast<double>(facts["scale"]);
    if (facts.contains("reference_current_fA")) {
      out.reference_current_fA = nb::cast<double>(facts["reference_current_fA"]);
    }
    if (facts.contains("precision")) out.precision = nb::cast<std::string>(facts["precision"]);
    if (out.precision != "fp32" && out.precision != "fp64") {
      throw std::invalid_argument("cache precision: fp32 or fp64");
    }
    return out;
  };
  c.def("classify", [cache_facts](const PyResolved& base, const PyResolved& edit, const std::string& camera,
                                 const nb::dict& facts) {
    const auto result = config::classify(base.r, edit.r, camera, cache_facts(facts));
    nb::dict out;
    out["tier"] = config::name(result.tier);
    std::vector<std::string> moved;
    for (const config::Stage stage : result.moved) moved.emplace_back(config::name(stage));
    out["moved"] = moved;
    out["keys"] = result.keys;
    out["reasons"] = result.reasons;
    return out;
  }, nb::arg("base"), nb::arg("edit"), nb::arg("camera"), nb::arg("facts") = nb::dict(),
     "Cache adequacy: none, O, E, S, S~ or R, plus moved stages, changed keys and refusal reasons (ui.md §4.7).");
  c.def("replay_scale", [cache_facts](const PyResolved& edit, const std::string& camera, const nb::dict& facts) {
    return config::replay_scale(edit.r, camera, cache_facts(facts));
  }, nb::arg("edit"), nb::arg("camera"), nb::arg("facts"),
     "Photocurrent scale for replay, including reference photometry.");
  c.def("probe_digest", [](const PyResolved& r, const std::string& camera) {
    return config::probe_digest(r.r, camera);
  }, nb::arg("resolved"), nb::arg("camera"), "Sensor digest with the five event-only edit keys masked.");
  const auto readout = [](const nb::dict& doc) {
    sensor::Readout r{};
    if (doc.contains("quantum_ns")) r.quantum_ns = nb::cast<int64_t>(doc["quantum_ns"]);
    if (doc.contains("delay_ns")) r.delay_ns = nb::cast<int64_t>(doc["delay_ns"]);
    if (doc.contains("phase_ns")) r.phase_ns = nb::cast<int64_t>(doc["phase_ns"]);
    if (doc.contains("clock_offset_ns")) r.clock_offset_ns = nb::cast<int64_t>(doc["clock_offset_ns"]);
    if (doc.contains("clock_drift_ppm")) r.clock_drift_ppm = nb::cast<double>(doc["clock_drift_ppm"]);
    return r;
  };
  c.def("sensor_clock_ns", [readout](const nb::dict& r, int64_t t) { return config::sensor_clock_ns(readout(r), t); },
        nb::arg("readout"), nb::arg("t_true_ns"));
  c.def("sensor_time_lower_bound", [readout](const nb::dict& r, int64_t t) {
    return config::sensor_time_lower_bound(readout(r), t);
  }, nb::arg("readout"), nb::arg("s_ns"));
  c.def("crossing_lower_bound", [readout](const nb::dict& r, int64_t t) {
    return config::crossing_lower_bound(readout(r), t);
  }, nb::arg("readout"), nb::arg("s_ns"));

}

}  // namespace balla::bindings
