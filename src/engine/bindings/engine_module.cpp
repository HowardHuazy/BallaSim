// ballasim._engine: the C++/CUDA engine (docs/dev/ENGINE_CPP.md, §1.9, §3 I1 and I3).
//
// - sampling.py's functions with their Python signatures, on the host instantiation of the
//   WindowEngine; a Python callable stands in for the renderer (render(pixel_ids, t_ns) ->
//   (irradiance, signature)), as the fake renderers of the tests do;
// - planner.py's, kinematics.py's and camera.py's classes and functions with their Python signatures
//   (motion.h, plan.h on the host); a Python callable stands in for the plan pass;
// - Engine(bundle, values) -> camera(i) with start / step / flush (and window / gt): the engine of a
//   run, on the device (OptiX, the CUDA sensor) or on the host (Embree, the CPU sensor).
//
// The module accepts no objects of _render or _sensor (their module ABIs differ): the bundle comes
// as C1 dicts or dataclasses, the configuration as a resolved run's plain values.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "bindings/config_value.h"
#include "bindings/ndarray_util.h"
#include "engine/motion.h"
#include "engine/window.h"
#include "render/bindings/bundle_convert.h"
#include "engine/engine.h"
#include "engine/param_fields.h"
#include "sensor/limiter.h"
#include "sensor/param_fields.h"
#include "sensor/waveform_dict.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace balla;
using namespace balla::engine;
using balla::bindings::In;
using balla::bindings::Out;
using balla::bindings::to_numpy;
using balla::bindings::to_numpy_bool;

namespace {

template <class T>
using Array = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// A light group's waveform: the C1 dict itself, or a group (dict or object) carrying it.
sensor::Waveform group_waveform(nb::handle group) {
  if (nb::isinstance<nb::dict>(group)) return sensor::parse_waveform(group);
  return sensor::parse_waveform(render::bindings::field(group, "waveform", "light group"));
}

// 1-D integer array -> vector<T> (range-checked; floats rejected).
template <class T>
std::vector<T> ints(nb::handle a, const char* dtype, const char* what) {
  return render::bindings::int_vector<T>(a, dtype, what);
}

nb::dict stats_dict(const WindowStats& st) {
  nb::dict d;
  d["pixels"] = st.pixels;
  nb::list rounds;
  for (const uint64_t k : st.samples_by_round) rounds.append(k);
  d["samples_by_round"] = rounds;
  d["flagged_signature"] = st.flagged_signature;
  d["flagged_midpoint"] = st.flagged_midpoint;
  d["refine_cap_hits"] = st.refine_cap_hits;
  nb::list band;  // P2-C C2: per round the pixel-samples at the band's N (zero without per-pixel N)
  for (const uint64_t k : st.band_samples_by_round) band.append(k);
  d["band_samples_by_round"] = band;
  return d;
}

// P2-C C2's counters (execution.jump_counters; sampling.h kJump*, kSib*), with WindowStats::jump_on only, as
// <prefix>jump_* and <prefix>sibling_*: signature intervals with both outer intervals eligible or not (transits), at
// every level and terminal, the eligible ones by the two-sided eta_Delta in eps_L / 8 bins, endpoint misses and node
// savings at eps_L / 8, eps_L / 2 and eps_L, the misses' splits to the cap and the samples below node savings (at
// eps_L), gated intervals with cancellation evidence, terminal localized intervals by length (2^11 ... 2^23 ns bins),
// and the smooth siblings (flags, skips, forced splits, skipped samples; the splits the node rules would add).
void jump_counters(nb::dict& d, const WindowStats& st, const std::string& prefix = "") {
  if (!st.jump_on) return;
  const auto& j = st.jump;
  const auto words = [&](int first, int n) {
    nb::list l;
    for (int k = 0; k < n; ++k) l.append(j[static_cast<size_t>(first + k)]);
    return l;
  };
  const auto key = [&](const char* name) { return nb::str((prefix + name).c_str()); };
  d[key("jump_eligible")] = j[kJumpEligible];
  d[key("jump_transit")] = j[kJumpTransit];
  d[key("jump_terminal_eligible")] = j[kJumpTermEligible];
  d[key("jump_terminal_transit")] = j[kJumpTermTransit];
  d[key("jump_eta_delta")] = words(kJumpEta, kDiagBins);
  d[key("jump_endpoint_misses")] = words(kJumpMiss, 3);
  d[key("jump_node_savings")] = words(kJumpSave, 3);
  d[key("jump_miss_splits")] = j[kJumpMissSplits];
  d[key("jump_saved_samples")] = j[kJumpSavedSamples];
  d[key("jump_cancel_coverage")] = j[kJumpCancelCoverage];
  d[key("jump_cancel_shadow")] = j[kJumpCancelShadow];
  d[key("jump_bracket")] = words(kJumpBracket, kJumpBracketBins);
  d[key("sibling_flags")] = j[kSibFlags];
  d[key("sibling_skips")] = j[kSibSkips];
  d[key("sibling_forced")] = j[kSibForced];
  d[key("sibling_skipped_samples")] = j[kSibSkippedSamples];
  d[key("sibling_added")] = j[kSibAdded];
  d[key("sibling_added_forced")] = j[kSibAddedForced];
}

// P2-D D3a's shadow-mode counters (accuracy.localization shadow; localize.h kPv*), with WindowStats::pv_on only, as
// <prefix>pv_*: per class (every candidate; those the full guard set accepts, pv_full_*; those the posterior set
// accepts, pv_posterior_*) the intervals, their refinement cost (samples bisection rendered in them), shadow flips,
// localized flips and probes; per rule (coverage, r0, r6, r2, r5, r1) the intervals it fires on and their cost; R0's
// causes [overflow, no_table, carried, key]; R5's [candidate, routed, none]; probes by kind [seed, bisect, verify,
// gallop]; flips localized as seeds, from a prediction, verify hits, dropped (R1), unfinished; probe rounds; the
// eligible pixels' refinement cost; and per guard set the probes of the candidates it runs P&V on (pv_full_run_probes,
// pv_posterior_run_probes: its pre-routing rules leave them, R1-routed ones included; plan §3.11's c_pv).
void pv_shadow_counters(nb::dict& d, const WindowStats& st, const std::string& prefix = "") {
  if (!st.pv_on) return;
  const auto& w = st.pv;
  const auto key = [&](const std::string& name) { return nb::str((prefix + "pv_" + name).c_str()); };
  const auto list = [&](int first, int n) {
    nb::list l;
    for (int k = 0; k < n; ++k) l.append(w[static_cast<size_t>(first + k)]);
    return l;
  };
  const std::pair<int, const char*> classes[] = {{kPvCand, ""}, {kPvFull, "full_"}, {kPvPost, "posterior_"}};
  for (const auto& [c, name] : classes) {
    d[key(std::string(name) + (c == kPvCand ? "candidates" : "intervals"))] = w[static_cast<size_t>(c)];
    d[key(std::string(name) + (c == kPvCand ? "candidate_cost" : "cost"))] = w[static_cast<size_t>(c + 1)];
    d[key(std::string(name) + "flips")] = w[static_cast<size_t>(c + 2)];
    d[key(std::string(name) + "localized")] = w[static_cast<size_t>(c + 3)];
    d[key(std::string(name) + "probes")] = w[static_cast<size_t>(c + 4)];
  }
  const char* rules[] = {"coverage", "r0", "r6", "r2", "r5", "r1"};
  for (int q = 0; q < 6; ++q) {
    d[key(std::string("routed_") + rules[q])] = w[static_cast<size_t>(kPvRule + 2 * q)];
    d[key(std::string("routed_") + rules[q] + "_cost")] = w[static_cast<size_t>(kPvRule + 2 * q + 1)];
  }
  d[key("r0_causes")] = list(kPvR0Cause, 4);
  d[key("r5_causes")] = list(kPvR5Cause, 3);
  d[key("probes_by_kind")] = list(kPvProbeKind, 4);
  d[key("seed_flips")] = w[kPvSeeds];
  d[key("predicted_flips")] = w[kPvPredicted];
  d[key("verify_hits")] = w[kPvVerifyHits];
  d[key("dropped_flips")] = w[kPvDropped];
  d[key("unfinished_flips")] = w[kPvUnfinished];
  d[key("rounds")] = w[kPvRounds];
  d[key("eligible_cost")] = w[kPvEligibleCost];
  d[key("full_run_probes")] = w[kPvRunProbes];
  d[key("posterior_run_probes")] = w[kPvRunProbes + 1];
  // P2-D D3b: R4 (pre), R3 and its propagation (a-posteriori: pv_routed_r3, pv_routed_r3p) with their cost; per stage
  // of the ladder [affine, local, projective, quadratic] the localized flips it predicted, their probes and verify
  // hits; the state check at bisection's midpoints per class (pv_check, pv_full_check, pv_posterior_check: [midpoints,
  // compared, transients, multi, third-state sub-samples, in bracket, unpredicted]) with its contradictions (zero
  // allowed), unchecked midpoints and uniform third-state midpoints; the would-be P&V nodes per class.
  const char* rules_b[] = {"r4", "r3", "r3p"};
  for (int q = 0; q < 3; ++q) {
    d[key(std::string("routed_") + rules_b[q])] = w[static_cast<size_t>(kPvRuleB + 2 * q)];
    d[key(std::string("routed_") + rules_b[q] + "_cost")] = w[static_cast<size_t>(kPvRuleB + 2 * q + 1)];
  }
  d[key("stage_flips")] = list(kPvStage, 4);
  d[key("stage_probes")] = list(kPvStage + 4, 4);
  d[key("stage_verify_hits")] = list(kPvStage + 8, 4);
  const char* check_classes[] = {"", "full_", "posterior_"};
  for (int q = 0; q < 3; ++q) {
    d[key(std::string(check_classes[q]) + "check")] = list(kPvCheck + kPvChkWords * q, kPvChkWords);
    d[key(std::string(check_classes[q]) + "nodes")] = w[static_cast<size_t>(kPvNodes + q)];
  }
  d[key("check_contradictions")] = w[kPvCheckBug];
  d[key("check_unchecked")] = w[kPvCheckBug + 1];
  d[key("check_third_uniform")] = w[kPvCheckBug + 2];
}

nb::dict diag_dict(const WindowStats& st);  // below: the P2-0b counters and the above

// ---------------------------------------------------------------------------------------------
// sampling.py on the host instantiation
// ---------------------------------------------------------------------------------------------

struct PyMidpoint {
  double tol, dark;
  std::vector<double> m_min, m_max;

  MidpointParams params(size_t groups) const {
    if (m_min.size() != groups || m_max.size() != groups) {
      throw std::invalid_argument("MidpointTest: one waveform range per light group is needed");
    }
    std::vector<std::pair<double, double>> ranges;
    for (size_t g = 0; g < groups; ++g) ranges.emplace_back(m_min[g], m_max[g]);
    return midpoint_params(tol, dark, ranges);
  }
};

// sampling.MidpointTest or this module's: anything with tol, dark, m_min and m_max.
PyMidpoint midpoint_of(nb::handle h) {
  return {nb::cast<double>(h.attr("tol")), nb::cast<double>(h.attr("dark")),
          nb::cast<std::vector<double>>(h.attr("m_min")), nb::cast<std::vector<double>>(h.attr("m_max"))};
}

Out<float> py_midpoint_eta(const PyMidpoint& t, const In<float, -1, -1>& ea, const In<float, -1, -1>& em,
                        const In<float, -1, -1>& eb) {
  const size_t n = ea.shape(0), g = ea.shape(1);
  if (em.shape(0) != n || eb.shape(0) != n || em.shape(1) != g || eb.shape(1) != g) {
    throw std::invalid_argument("MidpointTest.eta: ea, em and eb need the same shape (n, groups)");
  }
  const MidpointParams p = t.params(g);
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = engine::midpoint_eta(p, ea.data() + i * g, em.data() + i * g, eb.data() + i * g);
  return to_numpy(std::move(out), {n});
}

// The Python callable as the renderer: requests -> (pixel_ids u32, t_ns i64), results scattered to
// their sample rows.
class PyRender final : public RenderHook {
 public:
  PyRender(nb::callable fn, int64_t t_w, size_t groups) : fn_(std::move(fn)), t_w_(t_w), groups_(groups) {}
  void render(SampleRequest* requests, size_t n, const uint32_t* out_index, float* irr, uint64_t* sig, uint32_t* klass,
              uint32_t) override {
    std::vector<uint32_t> pixel(n);
    std::vector<int64_t> t(n);
    for (size_t i = 0; i < n; ++i) {
      pixel[i] = requests[i].pixel;
      t[i] = t_w_ + requests[i].t_off_ns;
    }
    nb::tuple res = nb::cast<nb::tuple>(fn_(to_numpy(std::move(pixel), {n}), to_numpy(std::move(t), {n})));
    const auto e = nb::cast<In<float, -1, -1>>(nb::module_::import_("numpy").attr("ascontiguousarray")(res[0], "float32"));
    const auto s = nb::cast<Array<uint64_t>>(nb::module_::import_("numpy").attr("ascontiguousarray")(res[1], "uint64"));
    if (e.shape(0) != n || e.shape(1) != groups_ || s.shape(0) != n) {
      throw std::invalid_argument("render: the callable must return (irradiance (n, groups), signature (n,))");
    }
    for (size_t i = 0; i < n; ++i) {
      std::memcpy(irr + static_cast<size_t>(out_index[i]) * groups_, e.data() + i * groups_, groups_ * sizeof(float));
      sig[out_index[i]] = s.data()[i];
    }
    if (klass == nullptr) return;  // P2-C C2's tests: class words with carried_class
    if (res.size() < 3) {
      throw std::invalid_argument("render: with carried_class the callable must return (irradiance, signature, class "
                                  "words (n,))");
    }
    const auto c = nb::cast<Array<uint32_t>>(nb::module_::import_("numpy").attr("ascontiguousarray")(res[2], "uint32"));
    if (c.shape(0) != n) throw std::invalid_argument("render: class words (n,) u32 expected");
    for (size_t i = 0; i < n; ++i) klass[out_index[i]] = c.data()[i];
  }

 private:
  nb::callable fn_;
  int64_t t_w_;
  size_t groups_;
};

// sample_window and fixed_rate_window: the host WindowEngine, outputs as sampling.WindowRuns.
nb::object host_window(int64_t t_w, nb::handle pixel_ids, std::optional<std::vector<uint8_t>> levels, nb::handle rank,
                       const In<float, -1, -1>& carried_irr, const Array<uint64_t>& carried_sig, nb::callable render,
                       SampleRules rules, nb::handle carried_class = nb::none(),
                       const CancelToken* cancel = nullptr) {
  const std::vector<uint32_t> pixel = ints<uint32_t>(pixel_ids, "uint32", "pixel_ids");
  const std::vector<uint32_t> ranks = ints<uint32_t>(rank, "uint32", "rank");
  const size_t n = pixel.size(), groups = carried_irr.shape(1);
  if (ranks.size() != n || carried_irr.shape(0) != n || carried_sig.shape(0) != n || (levels && levels->size() != n)) {
    throw std::invalid_argument("pixel_ids, levels, rank, carried_irr and carried_sig need one entry per pixel");
  }
  if (groups < 1 || groups > static_cast<size_t>(kMaxGroups)) throw std::invalid_argument("need 1 to 8 light groups");
  SampleConfig cfg;
  cfg.rules = rules;
  cfg.groups = static_cast<int32_t>(groups);
  cfg.cancel = cancel;  // BL-0b's test hook: the batch and round check points
  std::vector<float> last_irr(carried_irr.data(), carried_irr.data() + n * groups);
  std::vector<uint64_t> last_sig(carried_sig.data(), carried_sig.data() + n);
  // P2-C C2's tests: the carried samples' class words turn the class words (and the P2-0b counters) on.
  const bool classes = !carried_class.is_none();
  std::vector<uint32_t> last_class;
  if (classes) {
    last_class = ints<uint32_t>(carried_class, "uint32", "carried_class");
    if (last_class.size() != n) throw std::invalid_argument("carried_class needs one entry per pixel");
  }
  WindowEngine<Space::kHost> engine;
  PyRender hook(std::move(render), t_w, groups);
  // The requests go out in slot order: pixel-major in the given order (ranks order the device engine's
  // active pixels; here they are checked only).
  const WindowEngine<Space::kHost>::Input in{t_w,         pixel.data(),    levels ? levels->data() : nullptr, n,
                                             last_irr.data(), last_sig.data(), nullptr,
                                             classes ? last_class.data() : nullptr};
  const WindowStats st = engine.run(in, cfg, hook, std::numeric_limits<size_t>::max(), nullptr);
  const size_t s = engine.samples();
  std::vector<int64_t> t(engine.t(), engine.t() + s), off(engine.offsets(), engine.offsets() + n + 1);
  std::vector<float> irr(engine.irr(), engine.irr() + s * groups);
  nb::object runs = nb::module_::import_("ballasim.engine.sampling").attr("WindowRuns");
  nb::dict stats = stats_dict(st);
  if (classes || st.jump_on) {  // the P2-0b counters (zero without class words) and P2-C C2's
    nb::dict c = diag_dict(st);
    stats["counters"] = c;
  }
  return runs(to_numpy(std::vector<uint32_t>(pixel), {n}), to_numpy(std::move(t), {s}),
              to_numpy(std::move(irr), {s, groups}), to_numpy(std::move(off), {n + 1}),
              to_numpy(std::move(last_irr), {n, groups}), to_numpy(std::move(last_sig), {n}), stats);
}

nb::object sample_window(int64_t t_w, nb::handle pixel_ids, nb::handle levels, nb::handle rank,
                         const In<float, -1, -1>& carried_irr, const Array<uint64_t>& carried_sig, nb::callable render,
                         bool refine_signature, nb::handle midpoint, int max_refine_levels, int64_t min_dt_ns,
                         std::optional<int32_t> mid_levels, std::optional<int64_t> tau_jump_ns, double delta_ref,
                         double half_ratio, std::optional<double> eps_jump, nb::handle jump, bool counters,
                         nb::handle carried_class, const CancelToken* cancel) {
  if (min_dt_ns < 1 || min_dt_ns > balla::time::kWindowNs || (min_dt_ns & (min_dt_ns - 1)) != 0) {
    throw std::invalid_argument("min_dt_ns must be a power of two in [1, 2^23]");
  }
  if (max_refine_levels < 0) throw std::invalid_argument("max_refine_levels must be >= 0");
  // The finest level of spacing min_dt_ns (CameraEngine::window refuses the levels above it).
  const int64_t max_level = balla::time::kWindowLog2 - std::countr_zero(static_cast<uint64_t>(min_dt_ns));
  std::vector<uint8_t> lv;
  for (const int64_t l : ints<int64_t>(levels, "int64", "levels")) {
    if (l < 0 || l > max_level) throw std::invalid_argument("levels must lie in [0, 23 - log2(min_dt_ns)]");
    lv.push_back(static_cast<uint8_t>(l));
  }
  SampleRules rules;
  rules.refine_signature = refine_signature;
  rules.max_refine_levels = max_refine_levels;
  rules.min_split = static_cast<uint32_t>(2 * min_dt_ns);
  rules.shift = 0;
  while ((int64_t{1} << rules.shift) < min_dt_ns) ++rules.shift;
  rules.midpoint = !midpoint.is_none();
  // eta_L's parameters, which eta_J (the jump rules) reads too: the engine always has the dark term and the light
  // groups' waveform ranges, so without a midpoint test the jump rules take them from `jump` (its tol unused).
  const size_t groups = carried_irr.shape(1);
  economy_rules(rules, mid_levels, tau_jump_ns, delta_ref, half_ratio, eps_jump);
  if (rules.midpoint) {
    if (!jump.is_none()) {
      throw std::invalid_argument("sample_window: jump= only without midpoint= (the midpoint test's dark term and "
                                  "waveform ranges serve both, as in the engine)");
    }
    rules.mid = midpoint_of(midpoint).params(groups);
  } else if (!jump.is_none()) {
    rules.mid = midpoint_of(jump).params(groups);
    rules.mid.tol = 0.0f;
  } else if (rules.refine_signature && rules.jump_rules()) {
    throw std::invalid_argument("sample_window: the jump rules (tau_jump_ns, eps_jump) need eta_J's dark term and "
                                "waveform ranges: pass jump= (a MidpointTest) or midpoint=");
  }
  rules.counters = counters;
  return host_window(t_w, pixel_ids, std::move(lv), rank, carried_irr, carried_sig, std::move(render), rules,
                     carried_class, cancel);
}

nb::object fixed_rate_window(int64_t t_w, nb::handle pixel_ids, nb::handle rank, int64_t period_ns,
                             const In<float, -1, -1>& carried_irr, const Array<uint64_t>& carried_sig,
                             nb::callable render) {
  if (period_ns < 1) throw std::invalid_argument("period_ns must be >= 1");
  SampleRules rules;
  rules.fixed = true;
  rules.period_ns = period_ns;
  rules.m = fixed_rate_samples(t_w, period_ns);
  return host_window(t_w, pixel_ids, std::nullopt, rank, carried_irr, carried_sig, std::move(render), rules);
}


// ---------------------------------------------------------------------------------------------
// planner.py, kinematics.py and camera.py on the host
// ---------------------------------------------------------------------------------------------

Vec3d vec3(nb::handle a, const char* what) {
  const auto v = render::bindings::array_field<double>(a, "float64", {3}, what);
  return {v[0], v[1], v[2]};
}
Out<double> np_vec(Vec3d v) { return to_numpy(std::vector<double>{v.x, v.y, v.z}, {3}); }
Out<double> np_mat(const Mat3d& m) {
  std::vector<double> d(9);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) d[static_cast<size_t>(3 * i + j)] = m.m[i][j];
  return to_numpy(std::move(d), {3, 3});
}
Mat3d mat3(nb::handle a, const char* what) {
  const auto v = render::bindings::array_field<double>(a, "float64", {3, 3}, what);
  return {{{v[0], v[1], v[2]}, {v[3], v[4], v[5]}, {v[6], v[7], v[8]}}};
}
nb::object bool_array(std::vector<uint8_t>&& v, std::initializer_list<size_t> shape) {
  return nb::module_::import_("numpy").attr("asarray")(to_numpy(std::move(v), shape), "dtype"_a = "bool");
}
// (n, 3) float64 points.
std::vector<Vec3d> points(const In<double, -1, 3>& x) {
  std::vector<Vec3d> out(x.shape(0));
  for (size_t i = 0; i < out.size(); ++i) out[i] = {x(i, 0), x(i, 1), x(i, 2)};
  return out;
}
// Holds of the C1 trajectories (int64 (H, 2) each; absent: none).
std::vector<Holds> bundle_holds(nb::handle bundle) {
  std::vector<Holds> out;
  for (nb::handle t : render::bindings::field(bundle, "trajectories", "bundle")) {
    Holds h;
    if (auto a = render::bindings::opt_field(t, "holds")) {
      const auto v = render::bindings::array_field<int64_t>(*a, "int64", {-1, 2}, "trajectory holds");
      for (size_t i = 0; i + 1 < v.size(); i += 2) h.emplace_back(v[i], v[i + 1]);
    }
    out.push_back(std::move(h));
  }
  return out;
}

struct PyMesh {
  explicit PyMesh(const render::Mesh& m) : dist(m) {}
  PyMesh(const PyMesh&) = delete;
  PyMesh& operator=(const PyMesh&) = delete;
  MeshDistance dist;
  ConeScratch scratch;  // the GIL serializes the calls
};

// A BodyWindow and the Python object that owns its mesh distance.
struct PyBody {
  BodyWindow b;
  nb::object owner;
};

struct PyLight {
  LightWindow l;
};

// The bundle, checked as the renderer checks it: SceneMotion indexes its trajectories and meshes unchecked.
render::SceneBundle validated(render::SceneBundle b) {
  render::validate(b);
  return b;
}

struct PySceneMotion {
  explicit PySceneMotion(nb::handle bundle)
      : b(validated(render::bindings::to_bundle(bundle))), motion(b, bundle_holds(bundle), true) {}
  PySceneMotion(const PySceneMotion&) = delete;
  PySceneMotion& operator=(const PySceneMotion&) = delete;
  render::SceneBundle b;
  SceneMotion motion;
};

struct PyGeometry {
  PyGeometry(nb::handle cam, double support_px)
      : camera(render::bindings::to_camera(cam, 0)), geo(camera, support_px) {}
  render::Camera camera;
  CameraGeometry geo;
};

nb::dict plan_stats_dict(const PlanStats& st) {
  nb::dict d;
  d["camera_held"] = st.camera_held;
  d["camera_v"] = st.camera_v;
  d["camera_omega"] = st.camera_omega;
  d["moving_bodies"] = st.moving_bodies;
  d["moving_lights"] = st.moving_lights;
  d["plan_pass"] = st.plan_pass;
  d["shadow_pixels"] = st.shadow_pixels;
  d["light_pixels"] = st.light_pixels;
  d["tiles_static"] = st.tiles_static;
  d["tiles_dynamic"] = st.tiles_dynamic;
  nb::list levels;
  for (const uint64_t k : st.levels) levels.append(k);
  d["levels"] = levels;
  d["v_hat_max"] = st.v_hat_max;
  if (st.level_w >= 0) d["level_w"] = st.level_w;  // P2-H H9: global or global_certified's window level
  if (st.seeds) {  // P2-B B4a: the band N's tile map
    d["band_tiles"] = st.band_tiles;
    d["band_seed_pixels"] = st.seed_pixels;
    d["band_carried_seeds"] = st.carried_seeds;
  }
  if (st.pixel) {  // P2-B B6c: the per-pixel band's largest seed radius (px; -1: no seed) and its cascade's levels
    d["band_radius_max"] = st.radius_max;
    d["band_radius_levels"] = st.radius_levels;
  }
  if (st.obliquity) {  // P2-D D1: the tiles whose level the obliquity factors move
    d["obliquity_tiles_up"] = st.obliquity_tiles_up;
    d["obliquity_tiles_down"] = st.obliquity_tiles_down;
    d["obliquity_tiles_capped"] = st.obliquity_tiles_capped;  // D1f: the lattice budget's ceiling (-1: none)
    d["obliquity_ceiling"] = st.obliquity_ceiling;
  }
  if (st.pv_bits) {  // P2-D D1: the plan bits' counts in the crop
    d["plan_oblique_pixels"] = st.oblique_pixels;
    d["plan_beyond_pixels"] = st.beyond_pixels;
    d["plan_rough_pixels"] = st.rough_pixels;
    d["plan_thin_pixels"] = st.thin_pixels;
  }
  if (st.ao) {  // P2-F F9b: the moving bodies with AO receivers, and the receivers
    d["ao_movers"] = st.ao_movers;
    d["ao_pixels"] = st.ao_pixels;
  }
  return d;
}

nb::object window_motion(const WindowMotion& m) {
  return nb::module_::import_("ballasim.engine.kinematics").attr("WindowMotion")(m.v, m.omega, m.held);
}

class PyPlanner {
 public:
  PyPlanner(nb::object geometry, int cam, nb::object motion, double d_px, std::optional<double> refine_tol_log,
            int64_t min_dt_ns, nb::handle crop, nb::callable plan_pass, bool shadow_band, bool band_seeds,
            const std::string& shadow_obliquity, bool plan_bits, bool pv_plan, double pitch,
            const std::string& mover_ao, uint64_t obliquity_budget, const std::string& level_policy)
      : geometry_(std::move(geometry)), motion_(std::move(motion)), plan_pass_(std::move(plan_pass)) {
    PyGeometry& g = nb::cast<PyGeometry&>(geometry_);
    PlanSchedule sc = plan_schedule(d_px, refine_tol_log, min_dt_ns);
    sc.band = shadow_band;
    sc.seeds = band_seeds;
    if (shadow_obliquity != "off" && shadow_obliquity != "projected") {
      throw std::invalid_argument("shadow_obliquity must be 'off' or 'projected'");
    }
    sc.obliquity = shadow_obliquity == "projected";  // P2-D D1
    sc.bits = plan_bits;
    sc.pv = pv_plan;
    sc.pitch = pitch;
    if (mover_ao != "off" && mover_ao != "proxy" && mover_ao != "traced") {
      throw std::invalid_argument("mover_ao must be 'off', 'proxy' or 'traced'");
    }
    sc.ao = mover_ao != "off";  // P2-F F9b: the moving bodies' AO receivers
    if (level_policy == "tile") {  // P2-H H9 (the engine's accuracy.level_policy)
      sc.level_policy = LevelPolicy::kPerTile;
    } else if (level_policy == "global") {
      sc.level_policy = LevelPolicy::kGlobal;
    } else if (level_policy == "global_certified") {
      sc.level_policy = LevelPolicy::kGlobalCertified;
    } else {
      throw std::invalid_argument("level_policy must be 'tile', 'global' or 'global_certified'");
    }
    std::optional<Crop> c;
    if (!crop.is_none()) {
      const auto v = nb::cast<std::vector<int32_t>>(crop);
      if (v.size() != 4) throw std::invalid_argument("crop must be (x0, y0, x1, y1)");
      c = Crop{v[0], v[1], v[2], v[3]};
    }
    PySceneMotion& sm = nb::cast<PySceneMotion&>(motion_);
    if (cam < 0 || static_cast<size_t>(cam) >= sm.b.cameras.size()) {
      throw std::invalid_argument("cam_index must name a camera of the motion's bundle");
    }
    planner_ = std::make_unique<Planner>(sm.motion, g.geo, static_cast<uint16_t>(cam), sc, c);
    planner_->set_obliquity_budget(obliquity_budget);  // P2-D D1f
    l_min = sc.l_min;
  }

  nb::object plan(int64_t t_w) {
    const CameraGeometry& g = nb::cast<PyGeometry&>(geometry_).geo;
    TilePlan p = planner_->plan(t_w, [&](const PixelStage& st) {
      // depth, or (depth, instance, shadow states[, carried class words]): I8's band and P2-B B4a's seeds need the
      // instances (uint32 per pixel) and the shadow states (uint32 (state, t's bits) per pixel and shadow light;
      // render/shade/lights.h); the seeds also take the carried samples' class words (uint32 per pixel, optional).
      const nb::object pass = plan_pass_(t_w);
      const bool full = nb::isinstance<nb::tuple>(pass);
      const auto depth = render::bindings::float_vector(full ? pass[0] : pass, "plan_pass depth");
      if (depth.size() != g.n_pixels) throw std::invalid_argument("plan_pass must return one depth per pixel");
      std::vector<uint32_t> instance, state, carried;
      std::vector<float> normal;
      if (full) {
        instance = ints<uint32_t>(pass[1], "uint32", "plan_pass instance");
        state = ints<uint32_t>(pass[2], "uint32", "plan_pass shadow states");
        if (instance.size() != g.n_pixels || state.size() != 2 * g.n_pixels * st.params.n_bits) {
          throw std::invalid_argument("plan_pass: one instance and one state per pixel and shadow light");
        }
        if (nb::len(pass) > 3 && !pass[3].is_none()) {
          carried = ints<uint32_t>(pass[3], "uint32", "plan_pass carried class words");
          if (carried.size() != g.n_pixels) throw std::invalid_argument("plan_pass: one class word per pixel");
        }
        if (nb::len(pass) > 4) {  // P2-D D1: the geometric normals (camera frame, 3 floats per pixel)
          normal = render::bindings::float_vector(pass[4], "plan_pass normals");
          if (normal.size() != 3 * g.n_pixels) throw std::invalid_argument("plan_pass: three normal floats per pixel");
        }
      }
      enqueue_pixel_stage<Space::kHost>(st, g, depth.data(), 1, buffers_, nullptr, full ? instance.data() : nullptr,
                                        full ? state.data() : nullptr, carried.empty() ? nullptr : carried.data(),
                                        normal.empty() ? nullptr : normal.data());
      return collect_pixel_stage<Space::kHost>(g, buffers_, nullptr, st.params.band != 0, st.params.seeds != 0,
                                               st.params.obliquity != 0, st.params.n_ao != 0);
    });
    const auto ty = static_cast<size_t>(g.tiles_y), tx = static_cast<size_t>(g.tiles_x);
    nb::object pv_plan;
    if (p.stats.plan_pass && planner_->stage().params.bits != 0) {
      // P2-D D1: R4 into the plan bits at the plan's levels and v_hat, and their counts (the engine's PvPlanPass).
      const PixelStage& st = planner_->stage();
      const std::vector<float> v_hat(p.v_hat.begin(), p.v_hat.end());
      unsigned long long c[4] = {};
      const Crop crop = planner_->crop();
      const float* width = st.params.pv != 0 ? buffers_.pv_width.data() : nullptr;
      Par<Space::kHost>::for_each(g.n_pixels,
                                  PvPlanPass{buffers_.pv_plan.data(), width, g.pixel_tile.data(), p.level.data(),
                                             v_hat.data(), 127, st.params.pitch, g.width, crop.x0, crop.y0, crop.x1,
                                             crop.y1, c, st.params.obliquity},
                                  nullptr);
      p.stats.pv_bits = true;
      p.stats.oblique_pixels = c[0];
      p.stats.beyond_pixels = c[1];
      p.stats.rough_pixels = c[2];
      p.stats.thin_pixels = c[3];
      pv_plan = nb::cast(to_numpy(std::vector<uint8_t>(buffers_.pv_plan.data(), buffers_.pv_plan.data() + g.n_pixels),
                                  {static_cast<size_t>(g.height), static_cast<size_t>(g.width)}));
    }
    nb::dict stats = plan_stats_dict(p.stats);
    if (!p.band.empty()) stats["band"] = bool_array(std::move(p.band), {ty, tx});  // P2-B B4a's tile map
    if (pv_plan.is_valid() && !pv_plan.is_none()) stats["pv_plan"] = pv_plan;  // P2-D D1's plan bits (height, width)
    return nb::module_::import_("ballasim.engine.planner")
        .attr("TilePlan")(bool_array(std::move(p.is_static), {ty, tx}), to_numpy(std::move(p.level), {ty, tx}),
                          to_numpy(std::move(p.v_hat), {ty, tx}), stats);
  }

  int l_min = 1;

 private:
  nb::object geometry_, motion_;
  nb::callable plan_pass_;
  std::unique_ptr<Planner> planner_;
  PlanBuffers<Space::kHost> buffers_;
};

std::vector<BodyWindow> body_list(nb::handle bodies) {
  std::vector<BodyWindow> out;
  for (nb::handle b : bodies) out.push_back(nb::cast<PyBody&>(b).b);
  return out;
}

void bind_planner(nb::module_& m) {
  nb::class_<PyMesh>(m, "MeshDistance", "kinematics.MeshDistance in C++ (motion.h).")
      .def("__init__",
           [](PyMesh* self, nb::handle mesh) {
             render::Mesh mm;
             mm.positions = render::bindings::array_field<float>(render::bindings::field(mesh, "positions", "mesh"),
                                                                 "float32", {-1, 3}, "mesh.positions");
             mm.indices = render::bindings::array_field<uint32_t>(render::bindings::field(mesh, "indices", "mesh"),
                                                                  "uint32", {-1, 3}, "mesh.indices");
             new (self) PyMesh(mm);
           },
           "mesh"_a)
      .def_ro_static("CLUSTER", &MeshDistance::kCluster)
      .def_ro_static("REL", &MeshDistance::kRel)
      .def("distance", [](const PyMesh& p, nb::handle q) { return p.dist.distance(vec3(q, "q")); }, "q"_a)
      .def("thickness", [](const PyMesh& p, double cap) { return p.dist.thickness(cap); }, "cap"_a,
           "The thickness of the mesh's thinnest parts, at most cap (motion.h; I8's band, region (d)).")
      .def(
          "cone",
          [](PyMesh& p, nb::handle q) -> nb::object {
            const auto c = p.dist.cone(vec3(q, "q"), p.scratch);
            if (!c) return nb::none();
            return nb::make_tuple(np_vec(c->first), c->second);
          },
          "q"_a);
  nb::class_<PyBody>(m, "BodyWindow", "kinematics.BodyWindow in C++.")
      .def("__init__",
           [](PyBody* self, int index, bool held, double v, double omega, double radius, nb::handle centre,
              nb::handle rotation, double half_s, nb::object mesh) {
             const BodyWindow b{index, held, v, omega, radius, vec3(centre, "centre"), mat3(rotation, "rotation"), half_s,
                                &nb::cast<PyMesh&>(mesh).dist};
             new (self) PyBody{b, std::move(mesh)};
           },
           "index"_a, "held"_a, "v"_a, "omega"_a, "radius"_a, "centre"_a, "rotation"_a, "half_s"_a, "mesh"_a)
      .def_prop_ro("index", [](const PyBody& p) { return p.b.index; }, nb::rv_policy::move)
      .def_prop_ro("held", [](const PyBody& p) { return p.b.held; }, nb::rv_policy::move)
      .def_prop_ro("v", [](const PyBody& p) { return p.b.v; }, nb::rv_policy::move)
      .def_prop_ro("omega", [](const PyBody& p) { return p.b.omega; }, nb::rv_policy::move)
      .def_prop_ro("radius", [](const PyBody& p) { return p.b.radius; }, nb::rv_policy::move)
      .def_prop_ro("half_s", [](const PyBody& p) { return p.b.half_s; }, nb::rv_policy::move)
      .def_prop_ro("centre", [](const PyBody& p) { return np_vec(p.b.centre); }, nb::rv_policy::move)
      .def_prop_ro("rotation", [](const PyBody& p) { return np_mat(p.b.rotation); }, nb::rv_policy::move)
      .def_prop_ro("point_speed", [](const PyBody& p) { return p.b.point_speed(); }, nb::rv_policy::move)
      .def_prop_ro("swept_radius", [](const PyBody& p) { return p.b.swept_radius(); }, nb::rv_policy::move)
      .def("to_body", [](const PyBody& p, nb::handle x) { return np_vec(p.b.to_body(vec3(x, "p"))); }, "p"_a)
      .def("distance", [](const PyBody& p, nb::handle x, double s) { return p.b.distance(vec3(x, "p"), s); }, "p"_a,
           "p_speed"_a = 0.0);
  nb::class_<PyLight>(m, "LightWindow", "kinematics.LightWindow in C++.")
      .def("__init__",
           [](PyLight* self, const std::string& kind, nb::handle position, bool spot, bool casts_shadows,
              std::optional<int> body, bool moving, double v, double omega) {
             if (kind != "point" && kind != "directional") throw std::invalid_argument("kind must be point or directional");
             new (self) PyLight{{kind == "directional", vec3(position, "position"), spot, casts_shadows, body.value_or(-1),
                                 moving, v, omega}};
           },
           "kind"_a, "position"_a, "spot"_a, "casts_shadows"_a, "body"_a.none(), "moving"_a, "v"_a, "omega"_a)
      .def_prop_ro("kind", [](const PyLight& p) { return p.l.directional ? "directional" : "point"; }, nb::rv_policy::move)
      .def_prop_ro("position", [](const PyLight& p) { return np_vec(p.l.position); }, nb::rv_policy::move)
      .def_prop_ro("spot", [](const PyLight& p) { return p.l.spot; }, nb::rv_policy::move)
      .def_prop_ro("casts_shadows", [](const PyLight& p) { return p.l.casts_shadows; }, nb::rv_policy::move)
      .def_prop_ro("body", [](const PyLight& p) -> std::optional<int> {
        return p.l.body < 0 ? std::nullopt : std::optional<int>(p.l.body);
      }, nb::rv_policy::move)
      .def_prop_ro("moving", [](const PyLight& p) { return p.l.moving; }, nb::rv_policy::move)
      .def_prop_ro("v", [](const PyLight& p) { return p.l.v; }, nb::rv_policy::move)
      .def_prop_ro("omega", [](const PyLight& p) { return p.l.omega; }, nb::rv_policy::move);

  struct PyReceivers {
    std::vector<Vec3d> x;
    LightWindow light;
  };
  nb::class_<PyReceivers>(m, "_Receivers", "planner._Receivers in C++ (plan.h shadowed).")
      .def("__init__",
           [](PyReceivers* self, const In<double, -1, 3>& x, const PyLight& light) {
             new (self) PyReceivers{points(x), light.l};
           },
           "x"_a, "light"_a)
      .def(
          "shadowed_by",
          [](const PyReceivers& r, const PyBody& body) {
            const ShadowLight sl = shadow_light(r.light, 0, 1);
            ConeScratch scratch;
            const ShadowBody sb = shadow_body(r.light, body.b, scratch);
            const size_t n = r.x.size();
            std::vector<uint8_t> mark(n);
            std::vector<double> lam(n);
            for (size_t i = 0; i < n; ++i) {
              const Receiver rc = make_receiver(sl, r.x[i], i, n);
              mark[i] = shadowed(sb, rc, r.x[i], i, n) ? 1 : 0;
              lam[i] = shadow_lambda(sb, rc);
            }
            return nb::make_tuple(to_numpy_bool(std::move(mark)), to_numpy(std::move(lam), {n}));
          },
          "body"_a);
  m.def(
      "moving_light_edge_speed",
      [](const In<double, -1, 3>& x, const PyLight& light, nb::handle bodies) {
        const std::vector<BodyWindow> every = body_list(bodies);
        const MovingLight ml = moving_light(light.l, every);
        std::vector<BodySphere> spheres;
        for (const BodyWindow& b : every) spheres.push_back(body_sphere(b));
        const std::vector<Vec3d> pts = points(x);
        std::vector<double> out(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) out[i] = moving_edge(ml, spheres.data(), pts[i]);
        return to_numpy(std::move(out), {pts.size()});
      },
      "x"_a, "light"_a, "bodies"_a, "planner.moving_light_edge_speed in C++.");
  m.def(
      "dilate_by_reach",
      [](const In<double, -1, -1>& value, nb::handle radius) {
        const size_t ty = value.shape(0), tx = value.shape(1);
        std::vector<int32_t> r;
        for (const int64_t v : render::bindings::array_field<int64_t>(radius, "int64", {static_cast<long>(ty),
                                                                                         static_cast<long>(tx)}, "radius")) {
          r.push_back(static_cast<int32_t>(std::min<int64_t>(v, INT32_MAX)));
        }
        std::vector<double> v(value.data(), value.data() + ty * tx);
        return to_numpy(dilate_by_reach(v, r, static_cast<int32_t>(ty), static_cast<int32_t>(tx)), {ty, tx});
      },
      "value"_a, "radius"_a, "planner.dilate_by_reach in C++.");
  m.def("obliquity_budget", &obliquity_budget, "groups"_a, "max_samples_per_window"_a.none() = nb::none(),
        "P2-D D1f: the engine's lattice budget of the obliquity factors per window (samples; motion.h "
        "obliquity_budget): kObliquityRunBytes / (2 (8 + 4 groups)), at most accuracy.max_samples_per_window.");
  nb::class_<PySceneMotion>(m, "SceneMotion", "kinematics.SceneMotion in C++ (motion.h).")
      .def(nb::init<nb::handle>(), "bundle"_a)
      .def("camera_window", [](const PySceneMotion& s, size_t cam, int64_t t0, int64_t t1) {
             return window_motion(s.motion.camera_window(cam, t0, t1));
           }, "cam"_a, "t0_ns"_a, "t1_ns"_a)
      .def("camera_pose", [](const PySceneMotion& s, size_t cam, int64_t t) {
             const auto [r, c] = s.motion.camera_pose(cam, t);
             return nb::make_tuple(np_mat(r), np_vec(c));
           }, "cam"_a, "t_ns"_a)
      .def("body_windows", [](nb::handle self, int64_t t0, int64_t t1) {
             nb::list out;
             for (const BodyWindow& b : nb::cast<PySceneMotion&>(self).motion.body_windows(t0, t1)) {
               out.append(PyBody{b, nb::borrow<nb::object>(self)});
             }
             return out;
           }, "t0_ns"_a, "t1_ns"_a)
      .def("light_windows", [](const PySceneMotion& s, int64_t t0, int64_t t1) {
             nb::list out;
             for (const LightWindow& l : s.motion.light_windows(t0, t1)) out.append(PyLight{l});
             return out;
           }, "t0_ns"_a, "t1_ns"_a)
      .def("body_moving_at", [](const PySceneMotion& s, size_t body, nb::handle t) {
             std::vector<uint8_t> out;
             for (const int64_t v : render::bindings::int_vector<int64_t>(t, "int64", "t_ns")) {
               out.push_back(s.motion.body_moving_at(body, v) ? 1 : 0);
             }
             return to_numpy_bool(std::move(out));
           }, "body"_a, "t_ns"_a);
  nb::class_<PyGeometry>(m, "CameraGeometry",
                         "camera.CameraGeometry in C++ (motion.h). support_px is the sub-sample reach, render::support_px "
                         "of the run's scene settings; the default, SUPPORT_PX, is that of the default settings, so pass "
                         "the run's value for other aperture_px, blur_sigma_px or truncation_sigmas.")
      .def(nb::init<nb::handle, double>(), "cam"_a, "support_px"_a = render::support_px(render::SceneSettings{}))
      .def_prop_ro("support_px", [](const PyGeometry& g) { return g.geo.support_px; }, nb::rv_policy::move)
      .def_prop_ro("width", [](const PyGeometry& g) { return g.geo.width; }, nb::rv_policy::move)
      .def_prop_ro("height", [](const PyGeometry& g) { return g.geo.height; }, nb::rv_policy::move)
      .def_prop_ro("n_pixels", [](const PyGeometry& g) { return g.geo.n_pixels; }, nb::rv_policy::move)
      .def_prop_ro("tiles_x", [](const PyGeometry& g) { return g.geo.tiles_x; }, nb::rv_policy::move)
      .def_prop_ro("tiles_y", [](const PyGeometry& g) { return g.geo.tiles_y; }, nb::rv_policy::move)
      .def_prop_ro("rays", [](const PyGeometry& g) { return to_numpy(std::vector<double>(g.geo.rays), {g.geo.n_pixels, 3}); }, nb::rv_policy::move)
      .def_prop_ro("pixel_tile", [](const PyGeometry& g) {
        return to_numpy(std::vector<int64_t>(g.geo.pixel_tile.begin(), g.geo.pixel_tile.end()), {g.geo.n_pixels});
      }, nb::rv_policy::move)
      .def_prop_ro("speed_factor", [](const PyGeometry& g) {
        return to_numpy(std::vector<double>(g.geo.speed_factor),
                        {static_cast<size_t>(g.geo.tiles_y), static_cast<size_t>(g.geo.tiles_x)});
      }, nb::rv_policy::move)
      .def_prop_ro("speed_factor_max", [](const PyGeometry& g) { return g.geo.speed_factor_max; }, nb::rv_policy::move)
      .def_prop_ro("tile_ray", [](const PyGeometry& g) {
        std::vector<double> v;
        for (const Vec3d& r : g.geo.tile_ray) v.insert(v.end(), {r.x, r.y, r.z});
        return to_numpy(std::move(v), {static_cast<size_t>(g.geo.tiles_y), static_cast<size_t>(g.geo.tiles_x), 3});
      }, nb::rv_policy::move)
      .def_prop_ro("tile_radius", [](const PyGeometry& g) {
        return to_numpy(std::vector<double>(g.geo.tile_radius),
                        {static_cast<size_t>(g.geo.tiles_y), static_cast<size_t>(g.geo.tiles_x)});
      }, nb::rv_policy::move)
      .def_prop_ro("rank", [](const PyGeometry& g) {
        std::vector<int64_t> rank(g.geo.n_pixels);
        for (size_t r = 0; r < rank.size(); ++r) rank[g.geo.pixel_by_rank[r]] = static_cast<int64_t>(r);
        return to_numpy(std::move(rank), {g.geo.n_pixels});
      }, nb::rv_policy::move);
  nb::class_<PyPlanner>(m, "Planner", "planner.Planner in C++: plan(t_w) -> planner.TilePlan; plan_pass(t_ns) returns "
                                      "the camera-z depth of every pixel centre (NaN on a miss), taken as float32, "
                                      "or (depth, instance, shadow states) for shadow_band (I8: instance uint32 per "
                                      "pixel; states uint32, per pixel and shadow light the state and the blocker's "
                                      "ray parameter's float32 bits, flattened). band_seeds (P2-B B4a) needs the "
                                      "tuple too, optionally with a fourth item, the carried class words (uint32 per "
                                      "pixel); the plan's stats then hold the band N's tile map as 'band'. "
                                      "shadow_obliquity ('off' or 'projected'), plan_bits and pv_plan (P2-D D1) need "
                                      "a fifth item, the geometric normals (float32, 3 per pixel, camera frame; the "
                                      "fourth may be None); the stats then hold the plan bits as 'pv_plan' (uint8, "
                                      "height x width; plan.h kPv*) and their counts (plan_bits: in windows with "
                                      "moving-shadow pairs or moving lights, as the engine; pv_plan adds R4 at the "
                                      "sub-sample pitch, px). mover_ao ('off', 'proxy' or 'traced'; P2-F F9b) marks "
                                      "the moving bodies' AO receivers (from the depths alone; stats 'ao_movers', "
                                      "'ao_pixels'). obliquity_budget (P2-D D1f; samples, 0: none): the lattice budget "
                                      "of the factors' raised tiles (Planner::set_obliquity_budget; the engine's: "
                                      "obliquity_budget(groups, max_samples_per_window)). level_policy ('tile', "
                                      "'global' or 'global_certified'; P2-H H9, the engine's accuracy.level_policy): "
                                      "the window level on every active or dynamic tile (stats 'level_w'); with "
                                      "obliquity_budget, the ceiling is searched on the lattice the policy plans.")
      .def(nb::init<nb::object, int, nb::object, double, std::optional<double>, int64_t, nb::handle, nb::callable,
                    bool, bool, const std::string&, bool, bool, double, const std::string&, uint64_t,
                    const std::string&>(),
           "geometry"_a, "cam_index"_a, "motion"_a, "d_px"_a, "refine_tol_log"_a.none(), "min_dt_ns"_a, "crop"_a.none(),
           "plan_pass"_a, nb::kw_only(), "shadow_band"_a = false, "band_seeds"_a = false, "shadow_obliquity"_a = "off",
           "plan_bits"_a = false, "pv_plan"_a = false, "pitch"_a = 0.0625, "mover_ao"_a = "off",
           "obliquity_budget"_a = 0, "level_policy"_a = "tile")
      .def_ro("l_min", &PyPlanner::l_min)
      .def("plan", &PyPlanner::plan, "t_w"_a);
}

// ---------------------------------------------------------------------------------------------
// The engine of a run
// ---------------------------------------------------------------------------------------------

// Resolved values (ballasim.config's Resolved.values(): {run: {key: value}, cameras: {name: {key: value}}}) typed by the
// rows of the five targets; keys without a target row (scene, accuracy.profile, paths, ...) are not the engine's.
EngineValues engine_values(const nb::dict& values) {
  return bindings::values_from_py(values, {&sensor::param_rows(), &sensor_spec_rows(), &render::scene_settings_rows(),
                                           &render::renderer_options_rows(), &engine_config_rows()});
}

// The renderer's held set (I3b): bodies held over the window, and the rebuild of its world structure.
void held_counters(nb::dict& d, const HeldSet& h) {
  d["held_bodies"] = h.bodies;
  d["held_rebuilt"] = h.rebuilt;
  d["held_build_s"] = h.build_ms * 1e-3;
}

// The P2-0b counters of a window (ENGINE_CPP §0.1; sampling.h kDiag*), perf.jsonl's "counters".
nb::dict diag_dict(const WindowStats& st) {
  nb::dict d;
  d["textured_samples"] = st.diag[kDiagTextured];
  d["shadow_edge_samples"] = st.diag[kDiagShadowEdge];
  d["silhouette_samples"] = st.diag[kDiagSilhouette];
  d["shadow_jumps"] = st.diag[kDiagShadowJumps];
  d["shadow_flips"] = st.diag[kDiagShadowFlips];
  d["coverage_jumps"] = st.diag[kDiagCoverageJumps];
  nb::list hist;
  for (int k = 0; k < kDiagBins; ++k) hist.append(st.diag[kDiagHist + k]);
  d["shadow_jump_eta"] = hist;
  jump_counters(d, st);
  pv_shadow_counters(d, st);  // P2-D D3a
  return d;
}

// The P2-B B0 census (execution.census; render/census.h), counters census_<class>_<count>.
void census_counters(nb::dict& d, const render::RenderStats& r) {
  if (!r.census_on) return;
  using namespace render;
  for (const auto& [cls, name] : {std::pair{kCensusBand, "band"}, std::pair{kCensusRest, "rest"}}) {
    for (const auto& [word, what] : {std::pair{kCensusSubsamples, "subsamples"}, std::pair{kCensusReusable, "reusable"},
                                     std::pair{kCensusMismatched, "mismatched"}}) {
      d[(std::string("census_") + name + "_" + what).c_str()] = r.census[static_cast<size_t>(cls + word)];
    }
  }
}

// P2-D D2 (execution.pv_outputs; render/shade/codes.h): the P&V outputs the window's render calls at N >= 256 wrote to
// scratch, entries and bytes.
void pv_counters(nb::dict& d, const render::RenderStats& r) {
  if (!r.pv_on) return;
  d["pv_out_entries"] = r.pv_entries;
  d["pv_out_bytes"] = r.pv_bytes;
}

// P2-B B2's per-pixel N (EngineConfig::n_band; engine.h SwitchStats), with the band only.
void band_counters(nb::dict& d, const WindowResult& r) {
  const SwitchStats& b = r.band;
  if (!b.on) return;
  d["band_pixels"] = b.band_pixels;
  d["band_samples"] = r.stats.band_samples;
  d["n_switch"] = b.switches;
  d["n_switch_up"] = b.up;
  d["n_switch_unquiescent"] = b.unquiescent;
  d["n_switch_changed"] = b.changed;  // P2-B B4b: T2's defect detector (engine.h SwitchStats)
  nb::list eta;
  for (const uint64_t v : b.eta) eta.append(v);
  d["switch_eta"] = eta;
  // P2-B B4a: the active pixels with an edge sample at N_band (the overfill is band_pixels less these) and at N_base.
  d["band_edge_pixels"] = b.band_edge;
  d["base_edge_pixels"] = b.base_edge;
  // P2-B B4b: the escalated pixels (re-rendered at N_band) and their pixel-samples at N_band; their t_w steps, as the
  // switches' (changed: T2's other defect detector).
  d["n_escalated"] = b.escalated;
  d["escalated_samples"] = b.escalated_samples;
  d["n_escalated_changed"] = b.escalated_changed;
  d["n_escalated_unquiescent"] = b.escalated_unquiescent;
  nb::list esc_eta;
  for (const uint64_t v : b.escalated_eta) esc_eta.append(v);
  d["escalated_eta"] = esc_eta;
  // P2-B B6a: the escalations by a neighbour's edge alone (pooled; n_escalated = base_edge_pixels + these), and the
  // active pixels the 1 px border ring alone puts at N_band (among band_pixels).
  d["n_escalated_pooled"] = b.escalated_pooled;
  d["band_border_pixels"] = b.border;
  // P2-B B6c: the active pixels the plan's band (tile or pixel map) puts at N_band, and the escalation's render calls
  // (the last ones of render_ms_by_call).
  d["band_plan_pixels"] = b.plan;
  d["escalation_render_calls"] = b.escalation_calls;
}

nb::list doubles(const std::vector<double>& v) {
  nb::list out;
  for (const double x : v) out.append(x);
  return out;
}

nb::list words64(const std::vector<uint64_t>& v) {
  nb::list out;
  for (const uint64_t x : v) out.append(x);
  return out;
}

nb::dict window_counters(const WindowResult& r) {
  nb::dict c;
  nb::dict diag = diag_dict(r.stats);
  census_counters(diag, r.render);
  pv_counters(diag, r.render);
  band_counters(diag, r);
  jump_counters(diag, r.escalation, "escalation_");  // P2-C C2: the escalation pass's own
  pv_shadow_counters(diag, r.escalation, "escalation_");  // P2-D D3a: the escalation pass's own
  c["counters"] = diag;
  held_counters(c, r.held);
  c["sampling"] = stats_dict(r.stats);
  c["pixel_samples"] = r.pixel_samples;
  c["render_calls"] = r.stats.render_calls;
  c["batches"] = r.stats.batches;
  c["render_s"] = (r.render.trace_ms + r.render.shade_ms) * 1e-3;
  c["render_trace_ms"] = r.render.trace_ms;
  c["render_shade_ms"] = r.render.shade_ms;
  c["render_host_ms"] = r.render.host_ms;
  c["render_probe_ms"] = r.render.probe_ms;  // P2-D D3a's probes (trace and state kernel; apart from render_s)
  nb::list by_call;
  for (const double ms : r.render_ms_by_call) by_call.append(ms);
  c["render_ms_by_call"] = by_call;
  // P2-C C2 (engine.h round_times): the render time by round of the window's pass and of the escalation's (empty
  // where the calls do not add up), the escalation's samples by round, and the redone batches' and t_w nodes' calls.
  const RoundTimes rt = round_times(r);
  c["render_ms_by_round"] = doubles(rt.pass);
  c["escalation_render_ms_by_round"] = doubles(rt.escalation);
  c["escalation_samples_by_round"] = words64(r.escalation.samples_by_round);
  c["render_ms_redone"] = rt.redone_ms;
  c["render_ms_switches"] = rt.switches_ms;
  c["render_ms_escalation_nodes"] = rt.escalation_nodes_ms;
  c["sample_s"] = r.sample_ms * 1e-3;
  c["sense_s"] = r.sense_ms * 1e-3;
  nb::list frame_mean;
  for (const double v : r.frame_mean) frame_mean.append(v);
  c["frame_mean"] = frame_mean;
  nb::dict s;
  s["gpu_launches"] = r.launch_stats.launches;
  s["gpu_launch_ms_total"] = r.launch_stats.total_ms;
  s["gpu_launch_ms_max"] = r.launch_stats.max_ms;
  s["input_clamps"] = r.sensor_stats.input_clamps;
  s["quarantines"] = r.sensor_stats.quarantines;
  s["fallbacks"] = r.sensor_stats.fallbacks;
  s["pieces"] = r.sensor_stats.pieces;
  s["cells"] = r.sensor_stats.cells;
  c["sensor"] = s;
  return c;
}

// C4 columns of output events (as the Python engine's CameraRun._output wrote them): x, y, p, t_reported_ns, and the
// GT columns.
nb::dict columns(std::vector<OutEvent>&& ev, bool gt) {
  const size_t n = ev.size();
  std::vector<uint16_t> x(n), y(n);
  std::vector<uint8_t> p(n);
  std::vector<int64_t> t(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = ev[i].x;
    y[i] = ev[i].y;
    p[i] = ev[i].p;
    t[i] = ev[i].t_reported;
  }
  nb::dict d;
  d["x"] = to_numpy(std::move(x), {n});
  d["y"] = to_numpy(std::move(y), {n});
  d["p"] = to_numpy(std::move(p), {n});
  d["t_reported_ns"] = to_numpy(std::move(t), {n});
  if (!gt) return d;
  std::vector<uint32_t> inst(n);
  std::vector<uint16_t> sem(n), flags(n);
  std::vector<float> depth(n), fx(n), fy(n);
  std::vector<uint8_t> label(n);
  std::vector<int64_t> tc(n);
  for (size_t i = 0; i < n; ++i) {
    inst[i] = ev[i].instance;
    sem[i] = ev[i].semantic;
    depth[i] = ev[i].depth;
    fx[i] = ev[i].flow_x;
    fy[i] = ev[i].flow_y;
    label[i] = ev[i].label;
    tc[i] = ev[i].t_cross;
    flags[i] = ev[i].flags;
  }
  d["instance"] = to_numpy(std::move(inst), {n});
  d["semantic"] = to_numpy(std::move(sem), {n});
  d["depth"] = to_numpy(std::move(depth), {n});
  d["flow_x"] = to_numpy(std::move(fx), {n});
  d["flow_y"] = to_numpy(std::move(fy), {n});
  d["label"] = to_numpy(std::move(label), {n});
  d["t_cross_ns"] = to_numpy(std::move(tc), {n});
  d["flags"] = to_numpy(std::move(flags), {n});
  return d;
}

// P2-H H4b: the rate limiter's counters (engine.h LimiterCounters).
nb::dict limiter_dict(const LimiterCounters& r) {
  nb::dict d;
  d["kept"] = r.kept;
  d["dropped"] = r.dropped;
  d["capped_bins"] = r.capped_bins;
  d["held"] = r.held;
  d["limiter_s"] = r.ms * 1e-3;
  return d;
}

// P2-H H4b: one camera's rate limiter (sensor/limiter.h) over a stream of events given as columns x, y, p,
// t_reported_ns and t_cross_ns sorted by t_reported_ns, cut into calls at `cuts` (each call: the events reported before
// the cut, complete up to it; then a flush). The SensorParams are the family's preset with `sensor` (field name ->
// value) set over it, validated; seed, camera and width as the engine's (run.sensor_seed, the camera's index in the
// bundle, its width). Returns (keep: bool per event, counters).
struct LimEvent {
  int64_t t_reported, t_cross;
  uint32_t x, y, p;
  size_t index;
};

nb::tuple limit_events(const nb::dict& events, const std::string& family, const nb::dict& sensor_values, uint64_t seed,
                       uint32_t camera, uint32_t width, nb::handle cuts) {
  sensor::SensorParams params = sensor::preset(family);
  for (auto [k, v] : sensor_values) {
    const std::string field = nb::cast<std::string>(k);
    const config::Param* p = sensor::param_row(field);
    if (p == nullptr) throw std::invalid_argument(sensor::unknown_field(field));
    config::set(*p, params, bindings::from_py(v, *p));
  }
  const sensor::LimiterConfig cfg = sensor::limiter_config(params, seed, camera, width);
  const auto col = [&](const char* name) -> nb::handle {
    if (!events.contains(name)) throw std::invalid_argument(std::string("limit: events need the column ") + name);
    return events[name];
  };
  const std::vector<int64_t> t = ints<int64_t>(col("t_reported_ns"), "int64", "t_reported_ns");
  const std::vector<int64_t> tc = ints<int64_t>(col("t_cross_ns"), "int64", "t_cross_ns");
  const std::vector<uint32_t> x = ints<uint32_t>(col("x"), "uint32", "x");
  const std::vector<uint32_t> y = ints<uint32_t>(col("y"), "uint32", "y");
  const std::vector<uint32_t> p = ints<uint32_t>(col("p"), "uint32", "p");
  const size_t n = t.size();
  if (tc.size() != n || x.size() != n || y.size() != n || p.size() != n) {
    throw std::invalid_argument("limit: the event columns need the same length");
  }
  std::vector<int64_t> cut;
  if (!cuts.is_none()) cut = ints<int64_t>(cuts, "int64", "cuts");
  for (size_t k = 1; k < cut.size(); ++k) {
    if (cut[k] <= cut[k - 1]) throw std::invalid_argument("limit: cuts must increase strictly");
  }
  std::vector<uint8_t> keep(n, 0);
  sensor::LimiterState st;
  double limiter_s = 0;  // the limiter's calls alone (apply, flush), without the column conversions
  {
    nb::gil_scoped_release release;
    sensor::Limiter<LimEvent> lim(cfg);
    size_t i = 0;
    std::vector<LimEvent> call;
    const auto run = [&](size_t end, int64_t complete) {
      call.clear();
      for (; i < end; ++i) call.push_back({t[i], tc[i], x[i], y[i], p[i], i});
      const auto t0 = std::chrono::steady_clock::now();
      if (complete == std::numeric_limits<int64_t>::max()) {
        lim.flush(call);
      } else {
        lim.apply(call, complete);
      }
      limiter_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      for (const LimEvent& e : call) keep[e.index] = 1;
    };
    for (const int64_t c : cut) {
      size_t end = i;
      while (end < n && t[end] < c) ++end;
      run(end, c);
    }
    run(n, std::numeric_limits<int64_t>::max());
    st = lim.state();
    if (!cfg.on) std::fill(keep.begin(), keep.end(), 1);
  }
  nb::dict d;
  d["on"] = cfg.on;
  d["kept"] = st.kept;
  d["dropped"] = st.dropped;
  d["capped_bins"] = st.capped_bins;
  d["phase_ns"] = cfg.phase_ns;
  d["limiter_s"] = limiter_s;
  return nb::make_tuple(to_numpy_bool(std::move(keep)), d);
}

class PyCamera {
 public:
  PyCamera(nb::object owner, CameraEngine& cam, bool gt, nb::object* cancel)
      : owner_(std::move(owner)), cam_(cam), gt_(gt), cancel_(cancel) {}

  // BL-0b: the CancelToken (or None) that start, step and window check (CameraEngine::set_cancel); the Engine keeps a
  // reference to it while its camera reads it.
  void set_cancel(const CancelToken* token) {
    nb::object previous = std::move(*cancel_);
    *cancel_ = token != nullptr ? nb::find(token) : nb::none();
    cam_.set_cancel(token);
  }

  nb::dict start(int64_t t_s, int64_t warmup_ns, bool motion) {
    StartResult r;
    {
      nb::gil_scoped_release release;
      r = cam_.start(t_s, warmup_ns, motion);
    }
    nb::dict d;
    d["warmup_s"] = r.warmup_ms * 1e-3;
    d["probe_s"] = r.probe_ms * 1e-3;
    d["probe_history_s"] = static_cast<double>(r.probe_history_ns) * 1e-9;
    d["illuminance_scale"] = r.illuminance_scale;
    d["motion_windows"] = r.motion_windows;
    d["render_s"] = (r.render.trace_ms + r.render.shade_ms) * 1e-3;
    held_counters(d, r.held);
    nb::list frame_mean;
    for (const double v : r.frame_mean) frame_mean.append(v);
    d["frame_mean"] = frame_mean;
    nb::list proxy_blocked;  // scans.md §4.2: per proxy shadow light, the share the scan blocks at the start frame
    for (const float v : r.proxy_blocked) proxy_blocked.append(static_cast<double>(v));
    d["proxy_blocked"] = proxy_blocked;
    d["split_geometries"] = r.split_geometries;
    return d;
  }

  nb::tuple window(int64_t t_w, nb::handle pixel_ids, nb::handle levels, nb::handle rank) {
    const std::vector<uint32_t> pixel = ints<uint32_t>(pixel_ids, "uint32", "pixel_ids");
    const std::vector<int64_t> ranks = ints<int64_t>(rank, "int64", "rank");
    std::vector<uint8_t> lv;
    if (!levels.is_none()) {
      for (const int64_t l : ints<int64_t>(levels, "int64", "levels")) {
        if (l < 0 || l > balla::time::kWindowLog2) throw std::invalid_argument("levels must lie in [0, 23]");
        lv.push_back(static_cast<uint8_t>(l));
      }
    }
    if (ranks.size() != pixel.size() || (!levels.is_none() && lv.size() != pixel.size())) {
      throw std::invalid_argument("window: pixel_ids, levels and rank need the same length");
    }
    WindowResult r;
    {
      nb::gil_scoped_release release;
      r = cam_.window(t_w, pixel.data(), levels.is_none() ? nullptr : lv.data(), ranks.data(), pixel.size());
    }
    nb::dict c = window_counters(r);
    sensor::Events& e = r.events;
    const size_t n = e.size();
    nb::dict ev;
    ev["x"] = to_numpy(std::move(e.x), {n});
    ev["y"] = to_numpy(std::move(e.y), {n});
    ev["p"] = to_numpy(std::move(e.p), {n});
    ev["t_cross_ns"] = to_numpy(std::move(e.t_cross_ns), {n});
    ev["t_reported_ns"] = to_numpy(std::move(e.t_reported_ns), {n});
    ev["label"] = to_numpy(std::move(e.label), {n});
    ev["flags"] = to_numpy(std::move(e.flags), {n});
    return nb::make_tuple(ev, c);
  }

  // (columns of the final events, counters): the window end to end (CameraEngine::step).
  nb::tuple step(int64_t t_w) {
    StepResult r;
    {
      nb::gil_scoped_release release;
      r = cam_.step(t_w);
    }
    nb::dict c = window_counters(r.window);
    nb::dict k = nb::cast<nb::dict>(c["counters"]);
    k["kappa_pixels"] = r.kappa.pixels;
    k["flow_sum"] = r.kappa.flow;
    k["v_hat_sum"] = r.kappa.v_hat;
    k["lattice_sum"] = r.kappa.lattice;
    if (r.planned) {
      c["plan"] = plan_stats_dict(r.plan);
      c["level_cap"] = r.level_cap ? nb::cast(*r.level_cap) : nb::none();
      c["plan_s"] = r.plan_ms * 1e-3;
    } else {
      nb::dict mode;
      mode["mode"] = "fixed_rate";
      c["plan"] = mode;
      c["level_cap"] = nb::none();
      c["plan_s"] = r.plan_ms * 1e-3;
    }
    c["gt_s"] = r.output_ms * 1e-3;
    c["events"] = r.events;
    if (r.limiter.on) c["limiter"] = limiter_dict(r.limiter);  // P2-H H4b; absent while the limiter is off
    return nb::make_tuple(columns(std::move(r.ready), gt_), c);
  }

  // The rate limiter's counters since start() (P2-H H4b; None while it is off).
  nb::object limiter() const {
    LimiterCounters r;
    {
      nb::gil_scoped_release release;
      r = cam_.limiter_counters();
    }
    if (!r.on) return nb::none();
    return limiter_dict(r);
  }

  nb::dict flush() {
    std::vector<OutEvent> held;
    {
      nb::gil_scoped_release release;
      held = cam_.flush();
    }
    return columns(std::move(held), gt_);
  }

  nb::dict gt(nb::handle x, nb::handle y, nb::handle t_ns) {
    const std::vector<float> xs = render::bindings::float_vector(x, "x");
    const std::vector<float> ys = render::bindings::float_vector(y, "y");
    if (xs.size() != ys.size()) throw std::invalid_argument("x and y need the same length");
    const std::vector<int64_t> ts = render::bindings::int_vector<int64_t>(t_ns, "int64", "t_ns", xs.size());
    if (ts.size() != xs.size()) throw std::invalid_argument("t_ns needs one entry per position (or one for all)");
    std::vector<render::GtSample> out;
    {
      nb::gil_scoped_release release;
      out = cam_.gt(xs.data(), ys.data(), ts.data(), xs.size());
    }
    return render::bindings::gt_dict(std::move(out), {xs.size()});
  }

  // (start frame, runs): hashes of the carried samples start() rendered and of the last window's runs (host engine).
  nb::tuple runs_digest() const {
    const auto [frame, runs] = cam_.runs_digest();
    return nb::make_tuple(frame, runs);
  }

  // The last window's runs as the sensor read them (CameraEngine::last_runs; both engines).
  nb::dict last_runs() const {
    WindowRuns r;
    {
      nb::gil_scoped_release release;
      r = cam_.last_runs();
    }
    const size_t n = r.pixel.size(), rows = r.offsets.size(), samples = r.t.size(), n_log2 = r.n_log2.size();
    const size_t g = samples > 0 ? r.irr.size() / samples : 0;
    nb::dict d;
    d["pixel"] = to_numpy(std::move(r.pixel), {n});
    d["offsets"] = to_numpy(std::move(r.offsets), {rows});
    d["t"] = to_numpy(std::move(r.t), {samples});
    d["irr"] = to_numpy(std::move(r.irr), {samples, g});
    d["n_log2"] = to_numpy(std::move(r.n_log2), {n_log2});
    return d;
  }

  int64_t readout_quantum_ns() const { return cam_.sensor().readout().quantum_ns; }
  int64_t readout_delay_ns() const { return cam_.sensor().readout().delay_ns; }
  // The sensor clock at true time t: the resolved clock offset and drift, through the readout's own formula with no
  // delay and no quantum (as the output cut).
  int64_t sensor_clock_ns(int64_t t) const {
    sensor::Readout clock = cam_.sensor().readout();
    clock.delay_ns = 0;
    clock.quantum_ns = 0;
    return sensor::reported_time(clock, t);
  }
  std::string device() const { return cam_.sensor().device() == sensor::Device::kCuda ? "cuda" : "cpu"; }
  // The sensor's photometry: its scale (fA per renderer unit on axis) and per-pixel gain map (height, width).
  double illuminance_scale() const { return cam_.sensor().params().illuminance_scale; }
  Out<float> gain_map() const {
    const sensor::SensorModel& m = cam_.sensor();
    std::vector<float> g;
    g.reserve(m.maps().pixels.size());
    for (const sensor::PixelMap& p : m.maps().pixels) g.push_back(p.gain);
    return to_numpy(std::move(g), {m.height(), m.width()});
  }

 private:
  nb::object owner_;  // keeps the Engine alive
  CameraEngine& cam_;
  bool gt_;
  nb::object* cancel_;  // the Engine's reference to this camera's cancel token (PyEngine::cancel_slot)
};

class PyEngine {
 public:
  PyEngine(nb::handle bundle, const nb::dict& values) {
    Applied a = apply(engine_values(values));
    EngineConfig& cfg = a.config;
    gt_ = cfg.gt;
    cfg.holds = bundle_holds(bundle);
    if (auto groups = render::bindings::opt_field(bundle, "light_groups")) {
      for (nb::handle g : *groups) cfg.light_groups.push_back(group_waveform(g));
    }
    if (cfg.light_groups.empty()) cfg.light_groups.push_back(sensor::make_constant_waveform());
    render::SceneBundle b = render::bindings::to_bundle(bundle);
    cancel_.resize(b.cameras.size());
    std::vector<std::optional<SensorSpec>> sensors(b.cameras.size());
    for (auto& [name, spec] : a.sensors) {
      size_t i = 0;
      while (i < b.cameras.size() && b.cameras[i].name != name) ++i;
      if (i == b.cameras.size()) throw std::invalid_argument("engine values: " + name + " is not a camera of the bundle");
      sensors[i] = std::move(spec);
    }
    nb::gil_scoped_release release;
    engine_ = make_engine(std::move(b), std::move(cfg), sensors);
  }

  Engine& engine() { return *engine_; }
  bool gt() const { return gt_; }
  nb::object* cancel_slot(size_t i) { return &cancel_.at(i); }

 private:
  std::vector<nb::object> cancel_;  // per camera, its cancel token (PyCamera::set_cancel); destroyed after engine_
  std::unique_ptr<Engine> engine_;
  bool gt_ = true;
};

}  // namespace

NB_MODULE(_engine, m) {
  m.doc() = "BallaSim C++/CUDA engine (docs/dev/ENGINE_CPP.md): sampling.py's, planner.py's and kinematics.py's "
            "functions on the host instantiation, and the engine of a run (device or host)";
  nb::exception<MemoryBudgetError>(m, "MemoryBudgetError", PyExc_MemoryError);
  nb::exception<Cancelled>(m, "Cancelled");
  nb::class_<CancelToken>(m, "CancelToken",
                          "BL-0b's cooperative cancel (src/engine/cancel.h; docs/design/phase3/blender.md §3.8): set() "
                          "from any thread; a camera given the token (CameraEngine.set_cancel) raises Cancelled at its "
                          "next check point (each pixel batch and refinement round of a window, and before sensing).")
      .def(nb::init<>())
      .def("set", &CancelToken::set, "Requests the cancel (never cleared).")
      .def("is_set", &CancelToken::is_set);

  m.def(
      "waveform_range",
      [](nb::handle w) { return waveform_range(group_waveform(w)); }, "waveform"_a,
      "(min, max) over all time of a C1 light-group waveform m(t) (sampling.waveform_range).");
  nb::class_<PyMidpoint>(m, "MidpointTest", "P3 §6.3's eta_L midpoint test (sampling.MidpointTest), in C++.")
      .def(nb::init<double, double, std::vector<double>, std::vector<double>>(), "tol"_a, "dark"_a, "m_min"_a,
           "m_max"_a)
      .def_static(
          "for_groups",
          [](double tol, double dark, nb::handle waveforms) {
            PyMidpoint t{tol, dark, {}, {}};
            for (nb::handle w : waveforms) {
              const auto [lo, hi] = waveform_range(group_waveform(w));
              t.m_min.push_back(lo);
              t.m_max.push_back(hi);
            }
            return t;
          },
          "tol"_a, "dark"_a, "waveforms"_a)
      .def_ro("tol", &PyMidpoint::tol)
      .def_ro("dark", &PyMidpoint::dark)
      .def_prop_ro("m_min", [](const PyMidpoint& t) { return nb::tuple(nb::cast(t.m_min)); })
      .def_prop_ro("m_max", [](const PyMidpoint& t) { return nb::tuple(nb::cast(t.m_max)); })
      .def("eta", &py_midpoint_eta, "ea"_a, "em"_a, "eb"_a)
      .def(
          "jump",
          [](const PyMidpoint& t, const In<float, -1, -1>& ea, const In<float, -1, -1>& eb) {
            const size_t n = ea.shape(0), g = ea.shape(1);
            if (eb.shape(0) != n || eb.shape(1) != g) {
              throw std::invalid_argument("MidpointTest.jump: ea and eb need the same shape (n, groups)");
            }
            const MidpointParams p = t.params(g);
            std::vector<float> out(n);
            for (size_t i = 0; i < n; ++i) out[i] = engine::endpoint_eta(p, ea.data() + i * g, eb.data() + i * g);
            return to_numpy(std::move(out), {n});
          },
          "ea"_a, "eb"_a, "The endpoint jump eta_J of intervals with ends ea, eb (ENGINE_CPP I7's jump rules).")
      .def(
          "exceeds",
          [](const PyMidpoint& t, const In<float, -1, -1>& ea, const In<float, -1, -1>& em,
             const In<float, -1, -1>& eb) {
            const MidpointParams p = t.params(ea.shape(1));
            const size_t n = ea.shape(0), g = ea.shape(1);
            if (em.shape(0) != n || eb.shape(0) != n || em.shape(1) != g || eb.shape(1) != g) {
              throw std::invalid_argument("MidpointTest.exceeds: ea, em and eb need the same shape (n, groups)");
            }
            std::vector<uint8_t> out(n);
            for (size_t i = 0; i < n; ++i) {
              out[i] = midpoint_exceeds(p, ea.data() + i * g, em.data() + i * g, eb.data() + i * g) ? 1 : 0;
            }
            return to_numpy_bool(std::move(out));
          },
          "ea"_a, "em"_a, "eb"_a);
  m.def(
      "level_cap",
      [](nb::handle levels, int64_t budget, int l_min) {
        if (l_min < 0 || l_min > balla::time::kWindowLog2) throw std::invalid_argument("l_min must lie in [0, 23]");
        const std::vector<int64_t> lv = ints<int64_t>(levels, "int64", "levels");
        return engine::level_cap(lv.data(), lv.size(), budget, l_min);
      },
      "levels"_a, "budget"_a, "l_min"_a, "sampling.level_cap.");
  m.def("sample_window", &sample_window, "t_w"_a, "pixel_ids"_a, "levels"_a, "rank"_a, "carried_irr"_a,
        "carried_sig"_a, "render"_a, nb::kw_only(), "refine_signature"_a, "midpoint"_a.none(), "max_refine_levels"_a,
        "min_dt_ns"_a, "mid_levels"_a.none() = nb::none(), "tau_jump_ns"_a.none() = nb::none(), "delta_ref"_a = 0.0,
        "half_ratio"_a = 0.0, "eps_jump"_a.none() = nb::none(), "jump"_a.none() = nb::none(), "counters"_a = false,
        "carried_class"_a.none() = nb::none(), "cancel"_a.none() = nb::none(),
        "sampling.sample_window on the host WindowEngine (with the economy schedule's switches, ENGINE_CPP I7; jump: "
        "eta_J's dark term and waveform ranges, a MidpointTest, when no midpoint test is given); returns a "
        "sampling.WindowRuns. P2-C C2: counters turns on the jump rules' counters and carried_class (u32 per pixel) "
        "the class words, which the render callable then returns third; both go to stats['counters']. BL-0b: cancel, a "
        "CancelToken checked at each pixel batch and refinement round (Cancelled).");
  bind_planner(m);
  m.def("fixed_rate_window", &fixed_rate_window, "t_w"_a, "pixel_ids"_a, "rank"_a, "period_ns"_a, "carried_irr"_a,
        "carried_sig"_a, "render"_a, "sampling.fixed_rate_window on the host WindowEngine.");

  m.def("limit", &limit_events, "events"_a, nb::kw_only(), "family"_a = "prophesee_gen3", "sensor"_a = nb::dict(),
        "seed"_a = 1, "camera"_a = 0, "width"_a, "cuts"_a.none() = nb::none(),
        "P2-H H4b: the engine's rate limiter (sensor/limiter.h) on a stream of events (columns x, y, p, t_reported_ns, "
        "t_cross_ns, sorted by t_reported_ns), cut into calls before each of `cuts` (ns; the stream complete up to "
        "it) and flushed. SensorParams: the family preset with `sensor` (field -> value, e.g. {'limiter': 'epoch'}) "
        "over it; seed, camera and width as the engine's. Returns (keep, counters: on, kept, dropped, capped_bins, "
        "phase_ns, limiter_s: the wall time of the limiter's calls alone).");
  m.attr("has_device") = static_cast<bool>(BALLA_ENGINE_HAVE_DEVICE);
  m.def("device_available", &device_available,
        "True if the device engine can run here: has_device, a visible CUDA device and a usable OptiX (the rule "
        "run.device auto resolves by).");
  m.attr("TILE") = kTile;  // plan.h: the planner's tile edge (px)
  // The sub-sample reach around a pixel centre (px) at the default pixel response (render::support_px).
  m.attr("SUPPORT_PX") = render::support_px(render::SceneSettings{});
  nb::class_<PyCamera>(m, "CameraEngine", "One camera of an Engine (ENGINE_CPP §3 I1).")
      .def("start", &PyCamera::start, "t_s"_a, "warmup_ns"_a, "motion"_a = false,
           "Renders every pixel at t_s, initializes the sensor at t_s - warmup_ns and advances it to t_s (motion: "
           "the warm-up's whole windows the bundle covers are planned, rendered and sensed, events dropped); returns "
           "warmup_s, probe_s (the stationary probe, inside warmup_s), probe_history_s (its history T_b), motion_windows, render_s and the held set "
           "(held_bodies, held_rebuilt, held_build_s), illuminance_scale (the sensor's scale; reference photometry: "
           "solved from the start frame), frame_mean (veiling glare: the sensor's frame means per light group at "
           "t_s; empty without glare), and, for scenes with baked proxies (docs/dev/SCANS.md), proxy_blocked (per proxy "
           "shadow light, the share of its start-frame rays from baked receivers that the scan blocks) and "
           "split_geometries (the static bodies split from the scan geometry); [] and 0 otherwise.")
      .def("window", &PyCamera::window, "t_w"_a, "pixel_ids"_a, "levels"_a.none(), "rank"_a,
           "One window of the active pixels (levels None on the fixed-rate grid): (events, counters). Events as "
           "SensorModel.advance returns them (sorted by (t_cross_ns, pixel)); counters: sampling stats, "
           "pixel_samples, render_calls, batches, render_s and the renderer's stage timers, sample_s, sense_s, "
           "sensor launch stats, the held set (held_bodies, held_rebuilt, held_build_s; ENGINE_CPP I3b) and frame_mean "
           "(veiling glare: the sensor's frame means per light group at t_w + W, of the inputs its pixels hold; empty "
           "without glare).")
      .def("step", &PyCamera::step, "t_w"_a,
           "The window [t_w, t_w + W) end to end: plan, samples, sensing, output stage. Returns (columns of the events "
           "final after this window in output order: x, y, p, t_reported_ns and, with GT, instance, semantic, depth, "
           "flow_x, flow_y, label, t_cross_ns, flags; counters: plan stats or {mode: fixed_rate}, level_cap, plan_s, "
           "the sampling, sensor and held-set counters of window(), gt_s (output stage) and events in the crop.")
      .def("flush", &PyCamera::flush, "The held events at the end of a run (columns as step's).")
      .def("limiter", &PyCamera::limiter,
           "P2-H H4b: the rate limiter's counters since start() (sensor.limiter epoch; None while off): kept, dropped, "
           "capped_bins (bins that dropped events), held (events of a bin later events may still join; 0 after "
           "flush) and limiter_s (its wall time, the flush's included). step()'s counters carry the same per step "
           "under 'limiter'.")
      .def("set_cancel", &PyCamera::set_cancel, "token"_a.none(),
           "BL-0b: the CancelToken (None: none) that start, step and window check before their work and before "
           "sensing, and their windows at each pixel batch and refinement round; a set token raises Cancelled there "
           "and leaves the camera to be started again (abort path only).")
      .def("gt", &PyCamera::gt, "x"_a, "y"_a, "t_ns"_a,
           "Centre-ray ground truth (as Renderer.gt): instance, semantic, depth, flow_x, flow_y.")
      .def("runs_digest", &PyCamera::runs_digest,
           "Test hook of the stage audit (host engine): (start frame, runs), 64-bit hashes of the samples start() "
           "rendered and of the last window's runs (pixels, offsets, times, irradiance).")
      .def("last_runs", &PyCamera::last_runs,
           "Test hook (both engines): the last window's runs as the sensor read them, a dict of pixel (rank order), "
           "offsets (n + 1), t (int64 ns), irr (samples x groups) and n_log2 (each run's log2 N with the band, else "
           "empty); empty arrays when another window began since (engine.h WindowRuns).")
      .def_prop_ro("readout_quantum_ns", &PyCamera::readout_quantum_ns)
      .def_prop_ro("readout_delay_ns", &PyCamera::readout_delay_ns)
      .def("sensor_clock_ns", &PyCamera::sensor_clock_ns, "t_ns"_a,
           "The sensor clock at true time t_ns: t + sensor.clock_offset_ns + the drift (the camera's resolved values; "
           "no readout delay or quantum). The event file's /t_offset is this clock at the run start.")
      .def_prop_ro("device", &PyCamera::device)
      .def_prop_ro("illuminance_scale", &PyCamera::illuminance_scale,
                   "The sensor's photocurrent scale, fA per renderer unit on axis (photometry.mode; reference: after "
                   "start()).")
      .def("gain_map", &PyCamera::gain_map,
           "The sensor's static pixel gains (height, width), float32: natural vignetting with optics.vignetting (null: "
           "on under physical and reference photometry, off under relative), else 1.");
  nb::class_<PyEngine>(m, "Engine",
                       "Engine(bundle, values): the renderer of a run and the sensors of its cameras, on the device "
                       "(run.device gpu: OptiX and the CUDA sensor, needs has_device) or on the host (cpu: Embree and "
                       "the CPU sensor at accuracy.sensor_precision).\n"
                       "values: a resolved run's values (ballasim.config.Resolved.values(): {run: {key: value}, "
                       "cameras: {name: {key: value}}}), applied through the parameter rows (docs/dev/P0_PARAMS.md "
                       "§1.4); the named cameras get sensors. Light groups and the trajectories' holds come from the "
                       "bundle.")
      .def(nb::init<nb::handle, const nb::dict&>(), "bundle"_a, "values"_a)
      .def(
          "camera",
          [](nb::handle self, size_t i) {
            PyEngine& e = nb::cast<PyEngine&>(self);
            CameraEngine& cam = e.engine().camera(i);
            return PyCamera(nb::borrow<nb::object>(self), cam, e.gt(), e.cancel_slot(i));
          },
          "index"_a)
      .def_prop_ro(
          "budget_bytes", [](PyEngine& e) { return e.engine().budget_bytes(); },
          "The memory budget of the window samples and the engine's plan and output buffers in bytes "
          "(execution.device_budget_mb, or 70 % of the device "
          "memory free once the renderer and the sensors were built; on the host, of MemAvailable on Linux and of the "
          "physical memory elsewhere); run.json records it.")
      .def_prop_ro(
          "backend", [](PyEngine& e) { return e.engine().on_device() ? "optix" : "embree"; },
          "The renderer's backend: optix (the device engine) or embree (the host engine).");
}
