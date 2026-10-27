// ballasim._frames: frames on the event clock (docs/design/phase3/frames.md §3.8, §3.9). Thin bindings of the C++
// frames library: exact waveform means, sensor-clock stamps, frame schedules, the exposure reference, which
// builds its own renderer from a run's bundle and resolved values (as _render.Renderer.from_values does), the
// APS frames (§4.10) with their photometry inputs and the frames digest, and resolve_frames (§3.2): the frames values
// of a run's cameras (config::resolve_frames, balla_config) from the run's layers and --set items.
//
// Refusals with a frames.md §6 code raise FramesError (a ValueError) whose message starts with the code and whose
// `code` attribute holds it.
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "bindings/config_inputs.h"
#include "bindings/config_value.h"
#include "bindings/ndarray_util.h"
#include "config/resolve_frames.h"
#include "frames/aps_frames.h"
#include "frames/reference.h"
#include "frames/timing.h"
#include "render/bindings/bundle_convert.h"
#include "render/bindings/nir_rows.h"
#include "render/param_fields.h"
#include "render/renderer.h"
#include "render/scene_data.h"
#include "sensor/waveform.h"
#include "sensor/waveform_dict.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace balla;
using balla::bindings::to_numpy;
using balla::render::bindings::int_vector;

namespace {

PyObject* g_frames_error = nullptr;

void raise(const std::string& code, const char* message) {
  nb::object inst = nb::borrow(g_frames_error)(nb::str((code + ": " + message).c_str()));
  inst.attr("code") = nb::str(code.c_str());
  PyErr_SetObject(g_frames_error, inst.ptr());
}

void translate(const std::exception_ptr& e, void*) {
  try {
    std::rethrow_exception(e);
  } catch (const frames::FramesError& x) {
    raise(x.code, x.what());
  } catch (const config::FramesRefusal& x) {  // resolve_frames' refusals (balla_config)
    raise(x.code, x.what());
  }
}

nb::list warning_list(std::span<const frames::Warning> ws) {
  nb::list out;
  for (const frames::Warning& w : ws) out.append(nb::make_tuple(w.code, w.message));
  return out;
}

frames::Shutter shutter_of(const std::string& s) {
  if (s == "global") return frames::Shutter::kGlobal;
  if (s == "rolling") return frames::Shutter::kRolling;
  throw std::invalid_argument("shutter must be 'global' or 'rolling'");
}
frames::Readout readout_of(const std::string& s) {
  if (s == "top_down") return frames::Readout::kTopDown;
  if (s == "bottom_up") return frames::Readout::kBottomUp;
  throw std::invalid_argument("readout must be 'top_down' or 'bottom_up'");
}
frames::Position position_of(const std::string& s) {
  if (s == "START") return frames::Position::kStart;
  if (s == "CENTER") return frames::Position::kCenter;
  if (s == "END") return frames::Position::kEnd;
  throw std::invalid_argument("position must be 'START', 'CENTER' or 'END'");
}

frames::RowTiming row_timing(int32_t height, int64_t exposure_ns, int64_t row_dt_ns, const std::string& shutter,
                             const std::string& readout) {
  return {exposure_ns, row_dt_ns, height, shutter_of(shutter), readout_of(readout)};
}

// k after checking 0 <= k < size and 0 <= row < height (std::out_of_range: IndexError); the C++ accessors do not.
int32_t checked(const frames::Schedule& s, int32_t k, int32_t row) {
  if (k < 0 || k >= s.size()) {
    throw std::out_of_range("frame index " + std::to_string(k) + " is outside [0, " + std::to_string(s.size()) + ")");
  }
  if (row < 0 || row >= s.timing().height) {
    throw std::out_of_range("row " + std::to_string(row) + " is outside [0, " + std::to_string(s.timing().height) +
                            ")");
  }
  return k;
}

// The schedule's per-frame times on the true clock and its candidate indices.
nb::dict schedule_times(const frames::Schedule& s) {
  const auto n = static_cast<size_t>(s.size());
  std::vector<int64_t> soe(n), eoe(n), mid(n);
  std::vector<int32_t> cand(n);
  for (int32_t k = 0; k < s.size(); ++k) {
    soe[static_cast<size_t>(k)] = s.frame_soe(k);
    eoe[static_cast<size_t>(k)] = s.frame_eoe(k);
    mid[static_cast<size_t>(k)] = s.frame_mid(k);
    cand[static_cast<size_t>(k)] = s.candidate(k);
  }
  nb::dict d;
  d["soe"] = to_numpy(std::move(soe), {n});
  d["eoe"] = to_numpy(std::move(eoe), {n});
  d["mid"] = to_numpy(std::move(mid), {n});
  d["candidate"] = to_numpy(std::move(cand), {n});
  return d;
}

// One waveform per light group of the bundle: each group's C1 waveform (constant when a group carries none); one
// constant waveform for a bundle without groups (its "default" group).
std::vector<sensor::Waveform> bundle_waveforms(nb::handle bundle) {
  std::vector<sensor::Waveform> out;
  if (auto groups = render::bindings::opt_field(bundle, "light_groups")) {
    for (nb::handle g : *groups) {
      if (nb::isinstance<nb::dict>(g) && !nb::cast<nb::dict>(g).contains("waveform") &&
          !nb::cast<nb::dict>(g).contains("type")) {
        out.push_back(sensor::make_constant_waveform());
      } else if (nb::isinstance<nb::dict>(g)) {
        out.push_back(sensor::parse_waveform(g));
      } else if (auto w = render::bindings::opt_field(g, "waveform")) {
        out.push_back(sensor::parse_waveform(*w));
      } else {
        out.push_back(sensor::make_constant_waveform());
      }
    }
  }
  if (out.empty()) out.push_back(sensor::make_constant_waveform());
  return out;
}

// frames.aps.* values of a dict {key: value} (keys below frames.aps; None: the defaults). Unknown keys, booleans,
// floats for integers and nulls of non-nullable values are FRM-VALUE.
frames::ApsSettings aps_settings(nb::handle values) {
  frames::ApsSettings s;
  if (values.is_none()) return s;
  if (!nb::isinstance<nb::dict>(values)) throw std::invalid_argument("APS values must be a dict or None");
  for (auto [k, v] : nb::cast<nb::dict>(values)) {
    const auto key = nb::cast<std::string>(k);
    const auto bad = [&](const char* expected) {
      throw frames::FramesError("FRM-VALUE", "frames.aps." + key + " = " + nb::cast<std::string>(nb::repr(v)) +
                                                 ": expected " + expected);
    };
    const bool is_int = nb::isinstance<nb::int_>(v) && !nb::isinstance<nb::bool_>(v);
    const auto integer = [&]() -> int64_t {
      int64_t out = 0;
      if (!is_int || !nb::try_cast<int64_t>(v, out)) bad("an integer");
      return out;
    };
    const auto number = [&]() -> double {
      if (!is_int && !nb::isinstance<nb::float_>(v)) bad("a number");
      return nb::cast<double>(v);
    };
    if (key == "bits") {
      const int64_t b = integer();
      if (b < 8 || b > 16) bad("an integer in [8, 16]");
      s.bits = static_cast<int32_t>(b);
    } else if (key == "full_well_e") {
      s.full_well_e = number();
    } else if (key == "read_noise_e") {
      s.read_noise_e = number();
    } else if (key == "dark_current_e_s") {
      s.dark_current_e_s = number();
    } else if (key == "prnu") {
      s.prnu = number();
    } else if (key == "dsnu_e") {
      s.dsnu_e = number();
    } else if (key == "gain_dn_per_e") {
      s.gain_dn_per_e = v.is_none() ? std::nullopt : std::optional<double>(number());
    } else if (key == "offset_dn") {
      s.offset_dn = number();
    } else if (key == "seed") {
      s.seed = v.is_none() ? std::nullopt : std::optional<int64_t>(integer());
    } else {
      throw frames::FramesError("FRM-VALUE", "frames.aps." + key + ": no such APS value (bits, full_well_e, "
                                             "read_noise_e, dark_current_e_s, prnu, dsnu_e, gain_dn_per_e, offset_dn, "
                                             "seed)");
    }
  }
  return s;
}

nb::dict aps_params_dict(const sensor::ApsParams& p) {
  nb::dict d;
  d["bits"] = p.bits;
  d["full_well_e"] = p.full_well_e;
  d["read_noise_e"] = p.read_noise_e;
  d["dark_current_e_s"] = p.dark_e_s;
  d["prnu"] = p.prnu;
  d["dsnu_e"] = p.dsnu_e;
  d["gain_dn_per_e"] = p.gain_dn_e;
  d["offset_dn"] = p.offset_dn;
  d["seed"] = p.seed;
  return d;
}

sensor::ApsParams aps_params_of(nb::handle d) {
  sensor::ApsParams p;
  if (!nb::isinstance<nb::dict>(d)) throw std::invalid_argument("APS parameters must be a dict (aps_params)");
  const nb::dict dd = nb::borrow<nb::dict>(d);
  const auto get = [&](const char* k) -> nb::object {
    if (!dd.contains(k)) throw std::invalid_argument(std::string("APS parameters: missing ") + k);
    return dd[k];
  };
  p.bits = nb::cast<int32_t>(get("bits"));
  p.full_well_e = nb::cast<double>(get("full_well_e"));
  p.read_noise_e = nb::cast<double>(get("read_noise_e"));
  p.dark_e_s = nb::cast<double>(get("dark_current_e_s"));
  p.prnu = nb::cast<double>(get("prnu"));
  p.dsnu_e = nb::cast<double>(get("dsnu_e"));
  p.gain_dn_e = nb::cast<double>(get("gain_dn_per_e"));
  p.offset_dn = nb::cast<double>(get("offset_dn"));
  p.seed = nb::cast<uint64_t>(get("seed"));
  p.validate();
  return p;
}

frames::Roi roi_of(const std::vector<int32_t>& roi) {
  if (roi.size() != 4) throw frames::FramesError("FRM-VALUE", "frames: roi must be [x0, y0, x1, y1]");
  return {roi[0], roi[1], roi[2], roi[3]};
}

nb::dict aps_frames_py(bindings::In<float, -1, -1, -1> mono, nb::handle frame_soe_ns, uint64_t run_sensor_seed,
                       uint32_t camera, std::pair<int32_t, int32_t> resolution, const std::vector<int32_t>& roi,
                       int64_t exposure_ns, double illuminance_scale, nb::handle values, int64_t row_dt_ns,
                       const std::string& shutter, const std::string& readout, double veiling_glare,
                       std::optional<bindings::In<float, -1, -1>> gain, int threads) {
  const sensor::ApsParams params = frames::aps_params(aps_settings(values), run_sensor_seed);
  const auto soe = int_vector<int64_t>(frame_soe_ns, "int64", "frame_soe_ns");
  const frames::Roi r = roi_of(roi);
  if (soe.size() != mono.shape(0)) throw std::invalid_argument("frames aps: one frame_soe_ns per mono frame");
  if (static_cast<int64_t>(mono.shape(1)) != r.y1 - r.y0 || static_cast<int64_t>(mono.shape(2)) != r.x1 - r.x0) {
    throw std::invalid_argument("frames aps: mono must be (n, H, W) of the ROI");
  }
  frames::ApsPhotometry ph{illuminance_scale, veiling_glare, {}};
  if (gain) {  // the C++ side sees only the element count: a transposed map would pass it
    if (gain->shape(0) != mono.shape(1) || gain->shape(1) != mono.shape(2)) {
      throw std::invalid_argument("frames aps: gain must be (H, W) of the ROI");
    }
    ph.gain.assign(gain->data(), gain->data() + gain->size());
  }
  const frames::RowTiming timing = row_timing(resolution.second, exposure_ns, row_dt_ns, shutter, readout);
  frames::ApsFrames out;
  {
    nb::gil_scoped_release release;
    out = frames::aps_frames(params, ph, camera, resolution.first, r, timing, soe,
                             std::span<const float>(mono.data(), mono.size()), threads);
  }
  nb::dict d;
  const size_t n = soe.size(), h = mono.shape(1), w = mono.shape(2);
  d["dn"] = to_numpy(std::move(out.dn), {n, h, w});
  d["veil_mean"] = to_numpy(std::move(out.veil_mean), {n});
  d["normal_fallbacks"] = out.normal_fallbacks;
  d["warnings"] = warning_list(out.warnings);
  d["params"] = aps_params_dict(params);
  return d;
}

// Resolved entries as {key: {value, layer, status[, note]}} (run.json's record shape).
nb::dict entries_dict(const config::Entries& list) {
  nb::dict out;
  for (const auto& [k, e] : list) {
    nb::dict r;
    r["value"] = balla::bindings::to_py(e.value);
    r["layer"] = e.layer;
    r["status"] = e.status;
    if (!e.note.empty()) r["note"] = e.note;
    out[k.c_str()] = r;
  }
  return out;
}

// config::resolve_frames on the run's inputs (config.RunSpec.inputs), the bundle facts (config.bundle_info) and the
// frames command's --set items ({full key: value}; None: none); check false: the run's record only (see the header).
nb::dict resolve_frames_py(const nb::dict& inputs, const nb::dict& bundle, nb::handle sets, bool check) {
  const config::Inputs in = balla::bindings::inputs_from_py(inputs);
  const config::BundleInfo info = balla::bindings::bundle_from_py(bundle);
  const config::Node set_items = sets.is_none() ? config::Node{} : balla::bindings::node_from_py(sets, "sets");
  config::FramesRun run;
  std::vector<config::FramesCamera> cameras;
  {
    nb::gil_scoped_release release;
    cameras = config::resolve_frames(in, info, set_items, &run, check);
  }
  nb::list cams;
  for (const config::FramesCamera& c : cameras) {
    nb::dict d;
    d["name"] = c.name;
    d["index"] = c.bundle_index;
    d["events"] = c.events;
    d["values"] = entries_dict(c.values);
    d["clock_offset_ns"] = c.clock_offset_ns;
    d["clock_drift_ppm"] = c.clock_drift_ppm;
    d["contrast"] = c.contrast ? nb::object(nb::float_(*c.contrast)) : nb::none();
    d["digest_values"] = c.digest_values;
    cams.append(d);
  }
  nb::list warnings;
  for (const config::Diagnostic& w : run.warnings) warnings.append(w.message);
  nb::dict out;
  out["cameras"] = cams;
  out["run"] = entries_dict(run.values);
  out["warnings"] = warnings;
  return out;
}

class PyReference {
 public:
  PyReference(nb::handle bundle, nb::handle values, nb::handle camera, std::optional<std::vector<int32_t>> roi,
              std::optional<nb::list> waveforms) {
    render::RendererOptions o = render::apply_renderer(balla::bindings::values_from_py(
        values, {&render::scene_settings_rows(), &render::renderer_options_rows()}));
    render::SceneBundle b = render::bindings::to_bundle(bundle);
    uint16_t cam = 0;
    if (nb::isinstance<nb::str>(camera)) {
      const auto name = nb::cast<std::string>(camera);
      size_t i = 0;
      while (i < b.cameras.size() && b.cameras[i].name != name) ++i;
      if (i == b.cameras.size()) throw std::invalid_argument("frames reference: no camera '" + name + "' in the bundle");
      cam = static_cast<uint16_t>(i);
    } else {
      const auto i = nb::cast<int64_t>(camera);
      if (i < 0 || static_cast<size_t>(i) >= b.cameras.size()) {
        throw std::invalid_argument("frames reference: camera index out of range");
      }
      cam = static_cast<uint16_t>(i);
    }
    frames::Roi r{0, 0, b.cameras[cam].width, b.cameras[cam].height};
    if (roi) {
      if (roi->size() != 4) throw frames::FramesError("FRM-VALUE", "frames: roi must be [x0, y0, x1, y1]");
      r = {(*roi)[0], (*roi)[1], (*roi)[2], (*roi)[3]};
    }
    std::vector<sensor::Waveform> waves;
    if (waveforms) {
      for (nb::handle w : *waveforms) waves.push_back(sensor::parse_waveform(w));
    } else {
      waves = bundle_waveforms(bundle);
    }
    // P2-F F11a: the run's sensor.nir_fraction rows, as the engine and Renderer.from_values take them.
    std::vector<std::string> names;
    for (const render::Camera& c : b.cameras) names.push_back(c.name);
    o.nir_fraction = render::bindings::nir_rows(names, values);
    renderer_ = std::make_unique<render::Renderer>(std::move(b), o);
    ref_ = std::make_unique<frames::ExposureReference>(*renderer_, cam, waves, r);
  }

  nb::dict motion(const frames::Schedule& s, int32_t k) const {
    frames::MotionFacts f;
    {
      nb::gil_scoped_release release;
      f = ref_->motion(s, k);
    }
    nb::dict d;
    d["d_max_px"] = f.d_max_px;
    d["any_motion"] = f.any_motion;
    d["light_motion"] = f.light_motion;
    d["light_rate_per_s"] = f.light_rate_per_s;
    d["breakpoints"] = f.breakpoints;
    return d;
  }

  nb::dict frame(const frames::Schedule& s, int32_t k, double h_px, double probe_tol, int32_t k_max, int32_t k_fixed,
                 int32_t n_sub, double reuse_budget_mb) {
    frames::RefFrame f;
    {
      nb::gil_scoped_release release;
      f = ref_->frame(s, k, {h_px, probe_tol, k_max, k_fixed, n_sub, reuse_budget_mb});
    }
    const frames::Roi& r = ref_->roi();
    const auto h = static_cast<size_t>(r.y1 - r.y0), w = static_cast<size_t>(r.x1 - r.x0);
    const auto g = static_cast<size_t>(ref_->num_groups());
    nb::dict d;
    d["per_group"] = to_numpy(std::move(f.per_group), {h, w, g});
    d["mono"] = to_numpy(std::move(f.mono), {h, w});
    d["m_mean"] = to_numpy(std::move(f.m_mean), {h, g});
    d["depth"] = to_numpy(std::move(f.depth), {h, w});
    d["instance"] = to_numpy(std::move(f.instance), {h, w});
    d["k_used"] = f.k_used;
    d["k_mask"] = f.k_mask;
    d["mask_px"] = f.mask_px;
    d["rendered"] = f.rendered;
    d["d_max_px"] = f.d_max_px;
    d["fact_p99"] = f.fact_p99;
    d["probe_p999"] = f.probe_p999;
    d["probe_max"] = f.probe_max;
    d["warnings"] = warning_list(f.warnings);
    return d;
  }

  bindings::Out<float> visibility(int64_t t_ns) {
    std::vector<float> v;
    {
      nb::gil_scoped_release release;
      v = ref_->visibility(t_ns);
    }
    const frames::Roi& r = ref_->roi();
    return to_numpy(std::move(v), {static_cast<size_t>(r.y1 - r.y0), static_cast<size_t>(r.x1 - r.x0),
                                   static_cast<size_t>(ref_->num_groups())});
  }

  std::vector<int32_t> roi() const {
    const frames::Roi& r = ref_->roi();
    return {r.x0, r.y0, r.x1, r.y1};
  }
  int32_t num_groups() const { return ref_->num_groups(); }
  std::string backend() const {
    return renderer_->backend() == render::BackendKind::kOptix ? "optix" : "embree";
  }

 private:
  std::unique_ptr<render::Renderer> renderer_;
  std::unique_ptr<frames::ExposureReference> ref_;
};

}  // namespace

NB_MODULE(_frames, m) {
  m.doc() = "BallaSim frames (docs/design/phase3/frames.md): exact waveform means, sensor-clock stamps, frame "
            "schedules and BallaSim's exposure reference";
  g_frames_error = PyErr_NewExceptionWithDoc("ballasim._frames.FramesError",
                                             "A frames refusal with its frames.md §6 code (attribute `code`).",
                                             PyExc_ValueError, nullptr);
  m.attr("FramesError") = nb::handle(g_frames_error);
  nb::register_exception_translator(&translate);

  m.def(
      "waveform_mean",
      [](nb::handle waveform, nb::handle t_a, nb::handle t_b) {
        const sensor::Waveform w = sensor::parse_waveform(waveform);
        const auto a = int_vector<int64_t>(t_a, "int64", "t_a");
        const auto b = int_vector<int64_t>(t_b, "int64", "t_b", a.size());
        if (a.size() != b.size()) throw std::invalid_argument("t_a and t_b need the same length");
        std::vector<double> out(a.size());
        {
          nb::gil_scoped_release release;
          for (size_t i = 0; i < a.size(); ++i) out[i] = sensor::waveform_mean<double>(w, a[i], b[i]);
        }
        return to_numpy(std::move(out), {a.size()});
      },
      "waveform"_a, "t_a"_a, "t_b"_a,
      "The exact mean of a C1 waveform (dict) over [t_a, t_b] ns, per element (int64 arrays; frames.md §4.5); "
      "m(t_a) where t_b <= t_a.");
  m.def(
      "sensor_time",
      [](nb::handle t_ns, int64_t clock_offset_ns, double clock_drift_ppm) {
        const auto t = int_vector<int64_t>(t_ns, "int64", "t_ns");
        const sensor::Readout clock{0, 0, 0, clock_offset_ns, clock_drift_ppm};
        std::vector<int64_t> out(t.size());
        for (size_t i = 0; i < t.size(); ++i) out[i] = frames::sensor_time(clock, t[i]);
        return to_numpy(std::move(out), {t.size()});
      },
      "t_ns"_a, "clock_offset_ns"_a = 0, "clock_drift_ppm"_a = 0.0,
      "The sensor clock of true times (int64 array): t + clock_offset_ns + llround(clock_drift_ppm 1e-6 t), the "
      "readout formula with no delay and no quantum (sensor/readout.h; the engine's sensor_clock_ns).");

  nb::class_<frames::Schedule>(m, "Schedule",
                               "Frame schedule on the true clock (frames.md §4.1): per kept frame the start of exposure "
                               "of its first-read row; rows of the full image.")
      .def_static(
          "uniform",
          [](int32_t height, int64_t exposure_ns, int64_t t0_ns, int64_t period_ns, std::optional<int32_t> count,
             std::pair<int64_t, int64_t> support, int64_t row_dt_ns, const std::string& shutter,
             const std::string& readout) {
            return frames::Schedule::uniform(row_timing(height, exposure_ns, row_dt_ns, shutter, readout), t0_ns,
                                             period_ns, count, {support.first, support.second});
          },
          "height"_a, "exposure_ns"_a, "t0_ns"_a, "period_ns"_a, "count"_a.none(), "support"_a, nb::kw_only(),
          "row_dt_ns"_a = 0, "shutter"_a = "global", "readout"_a = "top_down",
          "Frames t0 + k period, k = 0 .. count - 1 (count None: every frame ending inside the support), kept when "
          "their exposure lies inside support = (begin_ns, end_ns).")
      .def_static(
          "blender",
          [](int32_t height, int64_t exposure_ns, nb::handle frame_ns, const std::string& position,
             std::pair<int64_t, int64_t> support, int64_t row_dt_ns, const std::string& shutter,
             const std::string& readout) {
            const auto f = int_vector<int64_t>(frame_ns, "int64", "frame_ns");
            return frames::Schedule::blender(row_timing(height, exposure_ns, row_dt_ns, shutter, readout), f,
                                             position_of(position), {support.first, support.second});
          },
          "height"_a, "exposure_ns"_a, "frame_ns"_a, "position"_a, "support"_a, nb::kw_only(), "row_dt_ns"_a = 0,
          "shutter"_a = "global", "readout"_a = "top_down",
          "One frame per Blender frame time t_ns(F_k) (int64 array), anchored at position START, CENTER or END.")
      .def_prop_ro("size", &frames::Schedule::size)
      .def_prop_ro("dropped", [](const frames::Schedule& s) { return std::vector<int32_t>(s.dropped().begin(),
                                                                                         s.dropped().end()); })
      .def_prop_ro("warnings", [](const frames::Schedule& s) { return warning_list(s.warnings()); })
      .def_prop_ro("exposure_ns", [](const frames::Schedule& s) { return s.timing().exposure_ns; })
      .def_prop_ro("readout_span_ns", &frames::Schedule::readout_span_ns)
      .def("times", &schedule_times, "Per kept frame: soe, eoe, mid (int64, true clock) and candidate (int32).")
      .def("soe", [](const frames::Schedule& s, int32_t k, int32_t row) { return s.soe(checked(s, k, row), row); },
           "k"_a, "row"_a, "Start of exposure of row `row` of frame k (true clock); IndexError out of range.")
      .def("eoe", [](const frames::Schedule& s, int32_t k, int32_t row) { return s.eoe(checked(s, k, row), row); },
           "k"_a, "row"_a, "End of exposure of row `row` of frame k (true clock); IndexError out of range.")
      .def(
          "shifted",
          [](const frames::Schedule& s, nb::handle shift_ns) {
            const auto v = int_vector<int64_t>(shift_ns, "int64", "shift_ns");
            return s.shifted(v);
          },
          "shift_ns"_a, "The schedule with frame k's windows moved by shift_ns[k].");

  m.def(
      "aps_params",
      [](nb::handle values, uint64_t run_sensor_seed) {
        return aps_params_dict(frames::aps_params(aps_settings(values), run_sensor_seed));
      },
      "values"_a.none(), "run_sensor_seed"_a,
      "The APS model's parameters (frames.md §4.10) of frames.aps values {key: value} (None: the §3.2 defaults): "
      "gain_dn_per_e null -> (2^bits - 1)/full_well_e, seed null -> run_sensor_seed; FRM-VALUE outside the bounds.");
  m.def("aps_frames", &aps_frames_py, "mono"_a, "frame_soe_ns"_a, nb::kw_only(), "run_sensor_seed"_a, "camera"_a,
        "resolution"_a, "roi"_a, "exposure_ns"_a, "illuminance_scale"_a, "values"_a.none() = nb::none(),
        "row_dt_ns"_a = 0, "shutter"_a = "global", "readout"_a = "top_down", "veiling_glare"_a = 0.0,
        "gain"_a.none() = nb::none(), "threads"_a = 0,
        "APS frames (frames.md §4.10) of mono (n, H, W) float32 over roi [x0, y0, x1, y1] of the full image "
        "resolution (W, H), frame k's first-read row starting at frame_soe_ns[k] (int64, true clock): camera is the "
        "bundle camera index, illuminance_scale its photocurrent scale (fA per renderer unit, run.json "
        "cameras[].illuminance_scale), gain its (H, W) gain map (None: 1), veiling_glare its optics.veiling_glare. "
        "values: frames.aps values (aps_params). Returns dn (n, H, W) uint16, veil_mean (n,), normal_fallbacks, "
        "warnings [(code, message)] and params (the resolved model parameters). The same frames for any threads.");
  m.def(
      "vignetting_on",
      [](std::optional<bool> vignetting, const std::string& mode) { return frames::vignetting_on(vignetting, mode); },
      "vignetting"_a.none(), "photometry_mode"_a,
        "optics.vignetting as the engine resolves it: the value, else on under physical and reference photometry.");
  m.def(
      "vignetting_gain",
      [](nb::handle bundle, nb::handle values, int64_t camera, std::optional<std::vector<int32_t>> roi,
         const std::string& law, const std::vector<double>& relative_illumination) {
        if (law != "cos4" && law != "jacobian") {
          throw std::invalid_argument("vignetting_gain: law must be cos4 or jacobian, got " + law);
        }
        const render::RendererOptions o = render::apply_renderer(balla::bindings::values_from_py(
            values, {&render::scene_settings_rows(), &render::renderer_options_rows()}));
        nb::list cams = nb::list(render::bindings::field(bundle, "cameras", "bundle"));
        if (camera < 0 || static_cast<size_t>(camera) >= cams.size()) {
          throw std::invalid_argument("vignetting_gain: camera index out of range");
        }
        const render::Camera c = render::bindings::to_camera(cams[static_cast<size_t>(camera)],
                                                             static_cast<size_t>(camera));
        const frames::Roi r = roi ? roi_of(*roi) : frames::Roi{0, 0, c.width, c.height};
        std::vector<float> g =
            frames::vignetting_gain(c, render::support_px(o.scene), r, law == "jacobian", relative_illumination);
        return to_numpy(std::move(g), {static_cast<size_t>(r.y1 - r.y0), static_cast<size_t>(r.x1 - r.x0)});
      },
      "bundle"_a, "values"_a, "camera"_a, "roi"_a = nb::none(), "law"_a = "cos4",
      "relative_illumination"_a = std::vector<double>{},
      "The camera's gain map over roi (None: the full frame), (H, W) float32, as the engine's (_engine camera "
      "gain_map) under the run's resolved values: law is the camera's optics.vignetting_law (cos4: cos^4 of each pixel "
      "centre's unit ray; jacobian: cos(theta) dOmega/dA of the camera model) and relative_illumination its "
      "optics.relative_illumination ([] or [c2, c4]).");
  m.def("resolve_frames", &resolve_frames_py, "inputs"_a, "bundle"_a, "sets"_a.none() = nb::none(), "check"_a = true,
        "The frames values of a run's cameras (frames.md §3.2): inputs is config.RunSpec.inputs (the run's recorded "
        "layers), bundle config.bundle_info(...), sets the frames command's --set items {full key: value} on top "
        "(cameras.<c>.frames.K, cameras.default.frames.K, execution.frames_device, execution.frames_texture_mb, "
        "run.frames_select; FRM-SET-SCOPE otherwise). Returns {cameras: [{name, index, events, values: {key: "
        "{value, layer, status[, note]}}, clock_offset_ns, clock_drift_ppm, contrast, digest_values}], run: {...}, "
        "warnings: [...]}: the cameras of run.select, then those of run.frames_select. Refusals carry their code: "
        "FRM-SET-SCOPE, FRM-VALUE, FRM-CAMERA, FRM-MODE-SOURCE; resolver errors are ValueErrors. check=False: only the "
        "cameras of run.select and the run rows, with none of those refusals (the run's record, which a --set item "
        "may still repair).");
  m.def(
      "frames_digest_base",
      [](const std::string& format, const std::string& bundle_digest, const std::string& camera_digest,
         const std::string& values, const std::string& plan_digest, const std::string& manifest_sha256) {
        return frames::frames_digest_base({format, bundle_digest, camera_digest, values, plan_digest,
                                           manifest_sha256});
      },
      "format"_a, "bundle_digest"_a, "camera_digest"_a, "values"_a = "", "plan_digest"_a = "",
      "manifest_sha256"_a = "",
      "frames_digest_base (frames.md §3.6 without the APS): SHA-256 over the format, the bundle digest, the camera's "
      "sensor-stage digest, the canonical other frames values (\"key=value\\n\" lines), the plan digest and the "
      "manifest's SHA-256 (empty where a part does not exist).");
  m.def(
      "frames_digest",
      [](const std::string& base, nb::handle params) {
        if (params.is_none()) return frames::frames_digest(base, nullptr);
        const sensor::ApsParams p = aps_params_of(params);
        return frames::frames_digest(base, &p);
      },
      "base"_a, "params"_a.none(),
      "frames_digest: SHA-256 over frames_digest_base and the resolved APS parameters (aps_params; None: no APS).");

  const frames::RefOptions kDefaults{};
  nb::class_<PyReference>(m, "ExposureReference",
                          "ExposureReference(bundle, values, camera, roi=None, waveforms=None): BallaSim's exposure "
                          "reference of one camera (frames.md §4.4) with the renderer of a run's resolved values "
                          "(values: {run: {...}, cameras: {...}}, Resolved.values(); run.device gpu: OptiX, cpu: "
                          "Embree). roi: [x0, y0, x1, y1] of the full image (None: the full frame); waveforms: one C1 "
                          "waveform dict per light group (None: the bundle's).")
      .def(nb::init<nb::handle, nb::handle, nb::handle, std::optional<std::vector<int32_t>>, std::optional<nb::list>>(),
           "bundle"_a, "values"_a, "camera"_a, "roi"_a = nb::none(), "waveforms"_a = nb::none())
      .def("motion", &PyReference::motion, "schedule"_a, "k"_a,
           "Motion facts of frame k: d_max_px, any_motion, light_motion, light_rate_per_s, breakpoints.")
      .def("frame", &PyReference::frame, "schedule"_a, "k"_a, nb::kw_only(), "h_px"_a = kDefaults.h_px,
           "probe_tol"_a = kDefaults.probe_tol, "k_max"_a = kDefaults.k_max, "k_fixed"_a = kDefaults.k_fixed,
           "n_sub"_a = kDefaults.n_sub, "reuse_budget_mb"_a = kDefaults.reuse_budget_mb,
           "The reference of frame k: per_group (H, W, G) and mono (H, W) float32, m_mean (H, G) float64, depth and "
           "instance at frame_mid, k_used, k_mask, mask_px, rendered (samples), d_max_px, fact_p99, probe_p999, "
           "probe_max, warnings [(code, message)]. reuse_budget_mb bounds the host memory of samples kept for the "
           "next level's reuse; it changes `rendered`, never the frame.")
      .def("visibility", &PyReference::visibility, "t_ns"_a,
           "V_g = R_g / max(R_g^noshadow, 1e-12) at t_ns over the ROI, (H, W, G) float32 (frames.md §4.9.4).")
      .def_prop_ro("roi", &PyReference::roi)
      .def_prop_ro("num_groups", &PyReference::num_groups)
      .def_prop_ro("backend", &PyReference::backend);
}
