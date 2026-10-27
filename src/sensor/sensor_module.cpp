// ballasim._sensor: the sensor model of contract C3 (docs/dev/CONTRACTS.md, docs/dev/SENSOR.md).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "bindings/config_value.h"
#include "bindings/ndarray_util.h"
#include "core/pmath/special.h"
#include "sensor/expcubic.h"
#include "sensor/output_order.h"
#include "sensor/param_fields.h"
#include "sensor/rederive.h"
#include "sensor/sensor_model.h"
#include "sensor/stage1.h"
#include "sensor/trace.h"
#include "sensor/waveform_dict.h"

namespace nb = nanobind;
using namespace balla::sensor;
using balla::bindings::In;
using balla::bindings::Out;
using balla::bindings::to_numpy;
using balla::sensor::parse_waveform;

namespace {

template <class T, ssize_t... Shape>
using ExactContig = nb::ndarray<const T, nb::shape<Shape...>, nb::c_contig, nb::device::cpu>;

// An array argument of an exact dtype and shape (convert = false: int64 times and offsets are never cast
// from floats or other integers) or a convertible one (float32 irradiance); TypeError otherwise.
template <class Array>
Array array_arg(nb::handle h, bool convert, const char* what) {
  Array out;
  if (!nb::try_cast<Array>(h, out, convert)) {
    throw nb::type_error((std::string("advance: ") + what + " has the wrong dtype or number of dimensions").c_str());
  }
  return out;
}

SensorParams resolve_params(nb::handle params, nb::handle overrides) {
  SensorParams p = nb::isinstance<nb::str>(params) ? preset(nb::cast<std::string>(params))
                                                   : nb::cast<SensorParams>(params);
  if (!overrides.is_none()) {
    nb::object obj = nb::cast(p);
    for (auto [key, value] : nb::cast<nb::dict>(overrides)) {
      const std::string field = nb::cast<std::string>(key);
      if (!balla::sensor::param_row(field)) throw nb::attribute_error(balla::sensor::unknown_field(field).c_str());
      nb::setattr(obj, key, value);
    }
    p = nb::cast<SensorParams>(obj);
  }
  p.validate();
  return p;
}

// A gain map argument (any shape, one gain per pixel in pixel order; None: empty, gain 1 everywhere), copied while the
// GIL is held; every gain finite and > 0 (std::invalid_argument otherwise).
std::vector<float> gain_vector(nb::handle gain, const char* where) {
  if (gain.is_none()) return {};
  nb::ndarray<const float, nb::c_contig, nb::device::cpu> a;
  if (!nb::try_cast(gain, a, true)) throw nb::type_error((std::string(where) + ": gain must be a float array").c_str());
  std::vector<float> g(a.data(), a.data() + a.size());
  for (const float v : g) {
    if (!(v > 0.0f) || !std::isfinite(v)) throw std::invalid_argument(std::string(where) + ": gains must be finite and > 0");
  }
  return g;
}

Precision parse_precision(const std::string& s) {
  if (s == "fp64") return Precision::kFp64;
  if (s == "fp32") return Precision::kFp32;
  throw std::invalid_argument("precision must be 'fp64' or 'fp32'");
}

Device parse_device(const std::string& s) {
  if (s == "cpu") return Device::kCpu;
  if (s == "cuda") return Device::kCuda;
  throw std::invalid_argument("device must be 'cpu' or 'cuda'");
}

nb::dict events_dict(Events&& e) {
  const size_t n = e.size();
  nb::dict d;
  d["x"] = to_numpy(std::move(e.x), {n});
  d["y"] = to_numpy(std::move(e.y), {n});
  d["p"] = to_numpy(std::move(e.p), {n});
  d["t_cross_ns"] = to_numpy(std::move(e.t_cross_ns), {n});
  d["t_reported_ns"] = to_numpy(std::move(e.t_reported_ns), {n});
  d["label"] = to_numpy(std::move(e.label), {n});
  d["flags"] = to_numpy(std::move(e.flags), {n});
  d["ds"] = to_numpy(std::move(e.ds), {n});
  d["dn"] = to_numpy(std::move(e.dn), {n});
  d["dl"] = to_numpy(std::move(e.dl), {n});
  d["l_first"] = to_numpy(std::move(e.l_first), {n});
  return d;
}

nb::dict stats_dict(const KernelStats& s, const GpuLaunchStats& g) {
  nb::dict d;
  d["substeps"] = s.substeps;
  d["substep_attempts"] = s.substep_attempts;
  d["settled_substeps"] = s.settled_substeps;
  d["cells"] = s.cells;
  d["refinements"] = s.refinements;
  d["leaf_intervals"] = s.leaf_intervals;
  d["pieces"] = s.pieces;
  d["fallbacks"] = s.fallbacks;
  d["input_clamps"] = s.input_clamps;
  d["quarantines"] = s.quarantines;
  d["hpf_splits"] = s.hpf_splits;  // the CPU kernels' (the device does not count them)
  d["gpu_launches"] = g.launches;
  d["gpu_reruns"] = g.reruns;
  d["gpu_launch_ms_total"] = g.total_ms;
  d["gpu_launch_ms_max"] = g.max_ms;
  d["gpu_slice_sorts"] = g.slice_sorts;  // launches in the per-slice order (P2-E E5; BALLA_SENSOR_SLICE_SORT)
#if BALLA_SENSOR_CENSUS
  nb::dict c;  // the census build's counters (census.h)
  for (int i = 0; i < kCensusCounters; ++i) c[kCensusNames[i]] = s.census.n[i];
  d["census"] = c;
#endif
  return d;
}

nb::dict maps_dict(const SensorModel& m) {
  // An ROI model's maps cover its rectangle.
  const std::optional<Roi>& roi = m.roi();
  const size_t H = roi ? static_cast<size_t>(roi->y1 - roi->y0) : m.height();
  const size_t W = roi ? static_cast<size_t>(roi->x1 - roi->x0) : m.width(), n = H * W;
  std::vector<float> on(n), off(n), leak(n), dark(n), phi0(n);
  std::vector<uint8_t> hot(n);
  for (size_t i = 0; i < n; ++i) {
    const PixelMap& p = m.maps().pixels[i];
    on[i] = static_cast<float>(p.theta_on);
    off[i] = static_cast<float>(p.theta_off);
    leak[i] = static_cast<float>(p.leak);
    dark[i] = static_cast<float>(p.dark_fA);
    phi0[i] = static_cast<float>(p.phi0);
    hot[i] = static_cast<uint8_t>(p.hot);
  }
  nb::dict d;
  d["theta_on"] = to_numpy(std::move(on), {H, W});
  d["theta_off"] = to_numpy(std::move(off), {H, W});
  d["leak"] = to_numpy(std::move(leak), {H, W});
  d["dark_current_fA"] = to_numpy(std::move(dark), {H, W});
  d["phi0"] = to_numpy(std::move(phi0), {H, W});
  d["hot"] = to_numpy(std::move(hot), {H, W});
  d["hot_target_hz"] = to_numpy(std::vector<double>(m.maps().hot_target_hz), {H, W});
  // The high-pass filter's time constant per pixel (s; 0 while off): 1 / its rate.
  std::vector<float> tau(n, static_cast<float>(m.params().hpf_on() ? m.params().tau_hp : 0.0));
  for (size_t i = 0; i < m.maps().hpf_b.size(); ++i) tau[i] = 1.0f / m.maps().hpf_b[i];
  d["tau_hp"] = to_numpy(std::move(tau), {H, W});
  return d;
}

template <class T>
CascadeConstants<T> trace_constants(double k, double tau_o, double tau_sf, double eps_L) {
  SensorParams p = preset("prophesee_gen3");  // a complete, valid set; the cascade reads only the four below
  p.k = k;
  p.tau_o = tau_o;
  p.tau_sf = tau_sf;
  p.eps_L = eps_L;
  p.validate();
  return cascade_constants<T>(p);
}

nb::tuple trace(const ExactContig<int64_t, -1>& t, const In<double, -1>& I, const ExactContig<int64_t, -1>& q,
                double k, double tau_o, double tau_sf, double eps_L, const std::string& precision) {
  if (I.shape(0) != t.shape(0)) throw std::invalid_argument("cascade_trace: t and I need the same length");
  TraceResult r;
  {
    nb::gil_scoped_release release;
    if (parse_precision(precision) == Precision::kFp32) {
      r = cascade_trace(trace_constants<float>(k, tau_o, tau_sf, eps_L), t.data(), I.data(), t.shape(0), q.data(),
                        q.shape(0));
    } else {
      r = cascade_trace(trace_constants<double>(k, tau_o, tau_sf, eps_L), t.data(), I.data(), t.shape(0), q.data(),
                        q.shape(0));
    }
  }
  const size_t n = r.s3.size();
  return nb::make_tuple(to_numpy(std::move(r.s3), {n}), r.substeps);
}

template <class T>
Out<double> z_step_batch(const In<double, -1>& z0, const In<double, -1>& I0, const In<double, -1>& I1,
                         const In<double, -1>& h, double k) {
  const size_t n = z0.shape(0);
  if (I0.shape(0) != n || I1.shape(0) != n || h.shape(0) != n) throw std::invalid_argument("z_step: length mismatch");
  std::vector<double> out(n);
  for (size_t i = 0; i < n; ++i) {
    const T i0 = static_cast<T>(I0(i)), i1 = static_cast<T>(I1(i)), hh = static_cast<T>(h(i));
    if (!(i0 > T(0) && i1 > T(0) && hh >= T(0))) throw std::invalid_argument("z_step: need I0 > 0, I1 > 0 and h >= 0 (stage1.h)");
    out[i] = static_cast<double>(z_step(static_cast<T>(z0(i)), i0, i1, hh, static_cast<T>(k)));
  }
  return to_numpy(std::move(out), {n});
}

template <class T>
Out<double> special_batch(const std::string& name, const In<double, -1>& x) {
  T (*f)(T) = nullptr;
  if (name == "exp") f = &balla::pmath::exp<T>;
  else if (name == "log") f = &balla::pmath::log<T>;
  else if (name == "expm1") f = &balla::pmath::expm1<T>;
  else if (name == "log1p") f = &balla::pmath::log1p<T>;
  else if (name == "erfcx") f = &balla::pmath::erfcx<T>;
  else if (name == "dawson") f = &balla::pmath::dawson<T>;
  else throw std::invalid_argument("special: unknown function '" + name + "'");
  const bool erfcx = name == "erfcx";
  const size_t n = x.shape(0);
  std::vector<double> out(n);
  for (size_t i = 0; i < n; ++i) {
    const T v = static_cast<T>(x(i));
    // The kernel's clamp below 0 (special.h) is not erfcx there: the hook refuses x < 0 and NaN.
    if (erfcx && !(v >= T(0))) throw std::invalid_argument("special: erfcx needs x >= 0");
    out[i] = static_cast<double>(f(v));
  }
  return to_numpy(std::move(out), {n});
}

class PySensorModel {
 public:
  PySensorModel(nb::handle params, uint32_t width, uint32_t height, uint64_t seed, nb::handle light_groups,
                uint32_t camera, int num_threads, nb::handle precision, nb::handle overrides,
                const std::string& device, nb::handle roi) {
    std::vector<Waveform> groups;
    for (nb::handle g : light_groups) groups.push_back(parse_waveform(g));
    const Device dev = parse_device(device);
    // The GPU runs fp32; the CPU defaults to the fp64 oracle.
    const Precision prec = precision.is_none() ? (dev == Device::kCuda ? Precision::kFp32 : Precision::kFp64)
                                               : parse_precision(nb::cast<std::string>(precision));
    std::optional<Roi> r;
    if (!roi.is_none()) {
      const std::vector<int32_t> v = nb::cast<std::vector<int32_t>>(roi);
      if (v.size() != 4) throw std::invalid_argument("roi must be (x0, y0, x1, y1)");
      r = Roi{v[0], v[1], v[2], v[3]};
    }
    model_ = std::make_unique<SensorModel>(resolve_params(params, overrides), width, height, seed, std::move(groups),
                                           camera, num_threads, prec, dev, r);
  }

  // Runs f(model) without the GIL, holding the model's lock: calls from several Python threads are serialized
  // (the model and its thread pool are single-threaded). The lock is taken after the GIL is released and released
  // before the GIL is taken again, so no thread holds one while waiting for the other.
  template <class F>
  auto locked(F&& f) const {
    nb::gil_scoped_release release;
    const std::lock_guard<std::mutex> lock(mutex_);
    return f(*model_);
  }

  void check_frame(const In<float, -1, -1>& frame, const char* what) const {
    const size_t n = static_cast<size_t>(model_->width()) * model_->height();
    if (frame.shape(0) != n || frame.shape(1) != static_cast<size_t>(model_->groups())) {
      throw std::invalid_argument(std::string(what) + ": irradiance0 must have shape (height * width, groups)");
    }
  }

  nb::bytes stationary_probe(int64_t t0_ns, const In<float, -1, -1>& irradiance0) {
    check_frame(irradiance0, "stationary_probe");
    const ProbeResult r = locked([&](SensorModel& m) { return m.stationary_probe(t0_ns, irradiance0.data()); });
    return nb::bytes(reinterpret_cast<const char*>(r.blob.data()), r.blob.size());
  }

  void stationary_init(int64_t t0_ns, const In<float, -1, -1>& irradiance0, const nb::bytes& probe) {
    check_frame(irradiance0, "stationary_init");
    ProbeResult r;
    r.blob.assign(reinterpret_cast<const uint8_t*>(probe.c_str()),
                  reinterpret_cast<const uint8_t*>(probe.c_str()) + probe.size());
    locked([&](SensorModel& m) { m.stationary_init(t0_ns, irradiance0.data(), r); });
  }

  void set_frame_means(const In<double, -1, -1>& means) {
    if (means.shape(1) != static_cast<size_t>(model_->groups())) {
      throw std::invalid_argument("set_frame_means: means must have shape (calls + 1, groups)");
    }
    std::vector<double> v(means.data(), means.data() + means.shape(0) * means.shape(1));
    locked([&](SensorModel& m) { m.set_frame_means(std::move(v)); });
  }

  void init(int64_t t0_ns, const In<float, -1, -1>& irradiance0) {
    const size_t n = static_cast<size_t>(model_->width()) * model_->height();
    if (irradiance0.shape(0) != n || irradiance0.shape(1) != static_cast<size_t>(model_->groups())) {
      throw std::invalid_argument("init: irradiance0 must have shape (height * width, groups)");
    }
    locked([&](SensorModel& m) { m.init(t0_ns, irradiance0.data()); });
  }

  nb::dict advance(const ExactContig<uint32_t, -1>& pixel_ids, nb::handle t_ns, nb::handle irradiance,
                   int64_t t_end_ns, nb::handle counts, nb::handle offsets) {
    const size_t n = pixel_ids.shape(0), G = static_cast<size_t>(model_->groups());
    // The ids are checked and read again after the GIL is released: an owned copy, which another thread cannot
    // change meanwhile. The sample arrays are read in place; no other thread may write to them during the call.
    const std::vector<uint32_t> ids(pixel_ids.data(), pixel_ids.data() + n);
    Events ev;
    if (offsets.is_none()) {
      // Rectangular runs: t_ns (n, m), irradiance (n, m, groups), optional counts (n).
      const auto t = array_arg<ExactContig<int64_t, -1, -1>>(t_ns, false, "t_ns (int64, n x m)");
      const auto irr = array_arg<In<float, -1, -1, -1>>(irradiance, true, "irradiance (n x m x groups)");
      const size_t m = t.shape(1);
      if (t.shape(0) != n || irr.shape(0) != n || irr.shape(1) != m || irr.shape(2) != G) {
        throw std::invalid_argument("advance: t_ns must be (n, m) and irradiance (n, m, groups) for n pixel ids");
      }
      std::vector<int32_t> cnt;
      if (!counts.is_none()) {
        const auto c = array_arg<ExactContig<int32_t, -1>>(counts, false, "counts (int32)");
        if (c.shape(0) != n) throw std::invalid_argument("advance: counts must have one entry per pixel id");
        cnt.assign(c.data(), c.data() + n);
      }
      ev = locked([&](SensorModel& s) {
        return s.advance(ids.data(), n, t.data(), irr.data(), m, cnt.empty() ? nullptr : cnt.data(), t_end_ns);
      });
    } else {
      // Ragged runs: t_ns (S,), irradiance (S, groups), row i = samples offsets[i]:offsets[i + 1].
      if (!counts.is_none()) throw std::invalid_argument("advance: pass either counts or offsets, not both");
      const auto t = array_arg<ExactContig<int64_t, -1>>(t_ns, false, "t_ns (int64, S)");
      const auto irr = array_arg<In<float, -1, -1>>(irradiance, true, "irradiance (S x groups)");
      const auto off = array_arg<ExactContig<int64_t, -1>>(offsets, false, "offsets (int64, n + 1)");
      const size_t S = t.shape(0);
      if (irr.shape(0) != S || irr.shape(1) != G) {
        throw std::invalid_argument("advance: with offsets, t_ns must be (S,) and irradiance (S, groups)");
      }
      if (off.shape(0) != n + 1 || off.data()[0] != 0 || off.data()[n] != static_cast<int64_t>(S)) {
        throw std::invalid_argument("advance: offsets must be (n + 1,) from 0 to the number of samples");
      }
      std::vector<int64_t> start(off.data(), off.data() + n);
      std::vector<int32_t> count(n);
      for (size_t i = 0; i < n; ++i) {
        const int64_t c = off.data()[i + 1] - off.data()[i];
        if (c < 0 || c > INT32_MAX) throw std::invalid_argument("advance: offsets must be non-decreasing");
        count[i] = static_cast<int32_t>(c);
      }
      ev = locked([&](SensorModel& s) {
        return s.advance_runs(ids.data(), n, t.data(), irr.data(), S, start.data(), count.data(), t_end_ns);
      });
    }
    return events_dict(std::move(ev));
  }

  nb::bytes save_state() const {
    const std::vector<uint8_t> blob = locked([](SensorModel& m) { return m.save_state(); });
    return nb::bytes(reinterpret_cast<const char*>(blob.data()), blob.size());
  }

  void load_state(const nb::bytes& blob) {
    const auto* data = reinterpret_cast<const uint8_t*>(blob.c_str());
    const size_t size = blob.size();
    locked([&](SensorModel& m) { m.load_state(data, size); });
  }

  SensorModel& model() { return *model_; }
  const SensorModel& model() const { return *model_; }

 private:
  std::unique_ptr<SensorModel> model_;
  mutable std::mutex mutex_;
};

}  // namespace

NB_MODULE(_sensor, m) {
  m.doc() = "BallaSim sensor model (contract C3): three-stage photoreceptor cascade with in-path noise, or the "
            "instantaneous log response of the textbook DVS (photoreceptor, D3), exact threshold crossings, refractory, "
            "leak, hot pixels and readout";

  auto params = nb::class_<SensorParams>(
      m, "SensorParams", "Sensor parameters in SI units (DESIGN §8.9); see docs/dev/SENSOR.md for every field.");
  params.def(nb::init<>()).def_static("preset", &preset, nb::arg("name"), "Parameters of a named preset.");
  params.def_static(
      "of_values",
      [](nb::handle values, const std::string& camera, nb::handle photometry) {
        const balla::config::Values v = balla::bindings::values_from_py(values, {&param_rows()});
        for (const auto& [name, cam] : v.cameras) {
          if (name != camera) continue;
          // Physical and reference photometry, and relative with optics.vignetting on: the engine also sets the
          // camera's gain map and (reference) the solved scale (SensorModel::set_photometry, SENSOR.md "Photometry"),
          // which the resolved values do not hold; the caller supplies them as photometry = (scale, gain).
          const nb::dict raw = nb::cast<nb::dict>(values["cameras"][camera.c_str()]);
          const std::string mode =
              raw.contains("photometry.mode") ? nb::cast<std::string>(raw["photometry.mode"]) : "relative";
          const bool stated = raw.contains("optics.vignetting") && !raw["optics.vignetting"].is_none();
          const bool vignetting = stated ? nb::cast<bool>(raw["optics.vignetting"]) : mode != "relative";
          if (photometry.is_none()) {
            if (mode != "relative" || vignetting) {
              throw std::invalid_argument("SensorParams.of_values: camera " + camera + " has " +
                                          (mode != "relative" ? "photometry.mode = " + mode : "optics.vignetting on") +
                                          ": its sensor also takes the engine's gain map and (reference) solved scale "
                                          "(run.json cameras[].illuminance_scale), so its parameters alone would make "
                                          "another sensor; pass photometry=(scale, gain) (CameraEngine."
                                          "illuminance_scale after start(), gain_map()), then SensorModel."
                                          "set_photometry(scale, gain) before init");
            }
            return params_of(v.run, cam, name);
          }
          const nb::tuple pair = nb::cast<nb::tuple>(photometry);
          if (pair.size() != 2) throw std::invalid_argument("SensorParams.of_values: photometry is (scale, gain)");
          const std::vector<float> gain = gain_vector(pair[1], "SensorParams.of_values");
          const bool flat = std::all_of(gain.begin(), gain.end(), [](float g) { return g == 1.0f; });
          if (vignetting && gain.empty()) {
            throw std::invalid_argument("SensorParams.of_values: camera " + camera + " is vignetted (optics.vignetting "
                                        "under photometry.mode = " + mode + "): photometry needs its gain map");
          }
          if (!vignetting && !flat) {
            throw std::invalid_argument("SensorParams.of_values: camera " + camera + " has no vignetting: its gain "
                                        "is 1 everywhere");
          }
          SensorParams p = params_of(v.run, cam, name);
          p.illuminance_scale = nb::cast<double>(pair[0]);
          p.validate();
          return p;
        }
        throw std::invalid_argument("values: no camera " + camera);
      },
      nb::arg("values"), nb::arg("camera"), nb::arg("photometry") = nb::none(),
      "The parameters of a camera of a run's resolved values (ballasim.config.Resolved.values()), as the engine "
      "applies them: its camera keys and the run keys applied to every camera; validated. Without photometry, "
      "relative photometry without vignetting only (else ValueError, as the engine's gain map and solved scale are "
      "not values). photometry = (scale, gain): the engine's photometry of the camera (CameraEngine.illuminance_scale "
      "after start(), and gain_map(), None when unvignetted), for any mode: the parameters take the scale, and the "
      "caller passes both to SensorModel.set_photometry before init.");
  // One attribute per parameter row (docs/dev/PARAMS_REFERENCE.md), named by the field.
  for (const balla::config::Param& r : param_rows()) {
    const balla::config::Param* row = &r;
    // The setter takes None too: from_py refuses it for a row that is not nullable (sensor.limiter_phase_s is).
    params.def_prop_rw(
        r.field.c_str(), [row](const SensorParams& p) { return balla::bindings::to_py(row->get(&p)); },
        [row](SensorParams& p, nb::handle v) { row->set(*row, &p, balla::bindings::from_py(v, *row)); },
        nb::for_setter(nb::arg("value").none()), r.help.c_str());
  }
  params.def("validate", &SensorParams::validate, "Raises ValueError naming the first invalid field.");
  m.def("preset_names", &preset_names, "Names of the built-in presets.");

  nb::class_<PySensorModel>(m, "SensorModel",
                            "One camera's sensor: SensorModel(params, width, height, seed, light_groups, camera, "
                            "num_threads=0, precision=None, overrides=None, device='cpu', roi=None).\n"
                            "camera is the camera's index in the scene bundle; with the seed it keys every random "
                            "draw, so the cameras of a rig need distinct indices.\n"
                            "device 'cpu' runs fp64 (default) or fp32 on num_threads threads; 'cuda' runs the fp32 "
                            "kernel on the GPU, bitwise equal to the CPU fp32 path.\n"
                            "params is a preset name or a SensorParams; light_groups is a list of C1 waveform dicts "
                            "(or light groups with a 'waveform' entry); overrides maps SensorParams fields to values.\n"
                            "roi (x0, y0, x1, y1), end-exclusive: a model of that rectangle of the width x height "
                            "frame whose draws are keyed by the frame's pixel ids (its events are the full model's "
                            "there); pixel ids, init's irradiance0 and the probe take the frame's.")
      .def(nb::init<nb::handle, uint32_t, uint32_t, uint64_t, nb::handle, uint32_t, int, nb::handle, nb::handle,
                    const std::string&, nb::handle>(),
           nb::arg("params"), nb::arg("width"), nb::arg("height"), nb::arg("seed"), nb::arg("light_groups"),
           nb::arg("camera"), nb::arg("num_threads") = 0, nb::arg("precision") = nb::none(),
           nb::arg("overrides") = nb::none(), nb::arg("device") = "cpu", nb::arg("roi") = nb::none())
      .def("init", &PySensorModel::init, nb::arg("t0_ns"), nb::arg("irradiance0"),
           "Steady state at t0_ns; irradiance0 is float32 (height * width, groups) in renderer units, finite "
           "(ValueError otherwise); samples outside [0, irr_max] are clamped and counted in stats().")
      .def("advance", &PySensorModel::advance, nb::arg("pixel_ids").noconvert(), nb::arg("t_ns"),
           nb::arg("irradiance"), nb::arg("t_end_ns"), nb::arg("counts") = nb::none(),
           nb::arg("offsets") = nb::none(),
           "Advance to t_end_ns. pixel_ids uint32 (n), t_ns int64 (n, m) strictly increasing per row (first >= "
           "model time, all <= t_end_ns), irradiance float32 (n, m, groups) finite and clamped as in init, "
           "optional counts int32 (n) of valid samples per row. Ragged form: t_ns int64 (S,), irradiance "
           "float32 (S, groups) and offsets int64 (n + 1,), row i = samples offsets[i]:offsets[i + 1] (counts "
           "then None). Unlisted pixels hold their last input. Returns a dict of event arrays sorted by "
           "(t_cross_ns, pixel): x, y, p, t_cross_ns, t_reported_ns, label (0 signal, 1 noise, 2 leak, 3 hot), "
           "flags, ds, dn, dl, l_first (a first event after init's leak since t0, 0 otherwise). An ROI model's "
           "pixel ids and events are the frame's.")
      .def("set_photometry",
           [](PySensorModel& s, double scale, nb::handle gain) {
             const std::vector<float> g = gain_vector(gain, "set_photometry");
             s.locked([&](SensorModel& sm) { sm.set_photometry(scale, g); });
           },
           nb::arg("illuminance_scale"), nb::arg("gain") = nb::none(),
           "Photometry (SENSOR.md \"Photometry\"): the photocurrent scale (fA per renderer unit on axis) and the "
           "per-pixel gain map (height * width gains in pixel order, any shape; None: 1 everywhere), as the engine "
           "sets them (CameraEngine.illuminance_scale after start(), gain_map()). The states take it at the next "
           "init: call it before init.")
      .def("gain_map",
           [](const PySensorModel& s) {
             std::vector<float> g = s.locked([](SensorModel& sm) {
               std::vector<float> v;
               v.reserve(sm.maps().pixels.size());
               for (const PixelMap& p : sm.maps().pixels) v.push_back(p.gain);
               return v;
             });
             // An ROI model's gains cover its rectangle (as sensor_maps).
             const std::optional<Roi>& r = s.model().roi();
             const size_t h = r ? static_cast<size_t>(r->y1 - r->y0) : s.model().height();
             const size_t w = r ? static_cast<size_t>(r->x1 - r->x0) : s.model().width();
             return to_numpy(std::move(g), {h, w});
           },
           "The static pixel gains (height, width), float32 (set_photometry; 1 everywhere by default); an ROI "
           "model's cover its rectangle.")
      .def("stationary_probe", &PySensorModel::stationary_probe, nb::arg("t0_ns"), nb::arg("irradiance0"),
           "The stationary initial state's probe from the full frame at t0_ns (bytes; the model is untouched): "
           "stationary_init with it equals init bit for bit, on any model with the same inputs (readout and label cut "
           "aside).")
      .def("stationary_init", &PySensorModel::stationary_init, nb::arg("t0_ns"), nb::arg("irradiance0"),
           nb::arg("probe"), "init with a stationary_probe (ValueError for a probe of other inputs).")
      .def("set_frame_means", &PySensorModel::set_frame_means, nb::arg("means"),
           "An ROI model's frame means for veiling glare, float64 (calls + 1, groups): row 0 at init, row k + 1 at "
           "the end of the k-th advance after it (a signal cache's /frame_mean).")
      .def("set_track_frame_means",
           [](PySensorModel& s, bool on) { s.locked([&](SensorModel& sm) { sm.set_track_frame_means(on); }); },
           nb::arg("on"), "Compute glare_frame_mean at every init and advance also without glare (a full model).")
      .def_prop_ro("roi",
                   [](const PySensorModel& s) -> nb::object {
                     const std::optional<Roi>& r = s.model().roi();
                     if (!r) return nb::none();
                     return nb::make_tuple(r->x0, r->y0, r->x1, r->y1);
                   },
                   "The region of interest (x0, y0, x1, y1), or None for a full model.")
      .def("state_bytes", [](const PySensorModel& s) { return s.model().state_bytes(); })
      .def("save_state", &PySensorModel::save_state)
      .def("load_state", &PySensorModel::load_state, nb::arg("state"),
           "Restore a save_state() blob of a model with the same parameters (ValueError otherwise); only its "
           "header is checked. Also the way back after an advance that failed part-way (the model is then "
           "uninitialized).")
      .def("hpf_states",
           [](const PySensorModel& s) {
             const std::vector<HpfState> h = s.locked([](SensorModel& sm) { return sm.hpf_states(); });
             std::vector<double> mem(h.size()), mem_n(h.size());
             for (size_t i = 0; i < h.size(); ++i) mem[i] = h[i].m, mem_n[i] = h[i].mn;
             return nb::make_tuple(to_numpy(std::move(mem), {h.size()}), to_numpy(std::move(mem_n), {h.size()}));
           },
           "The high-pass filter's side states at the model time (SENSOR.md \"High-pass filter\"): (m, m_n), the "
           "reference's memory and its noise part per pixel (empty while the filter's kernels do not run).")
      .def_prop_ro("hpf_kernel", [](const PySensorModel& s) { return s.model().hpf_kernel(); },
                   "Whether the high-pass filter's kernels run (sensor.tau_hp > 0, or the b = 0 test hook).")
      .def("_force_hpf_kernel", [](PySensorModel& s) { s.locked([](SensorModel& sm) { sm.force_hpf_kernel(); }); },
           "Test hook (ORACLES.md \"HPF oracle\" D): with the filter off, run its kernels at b = 0 from the next init "
           "(init again before advancing).")
      .def("sensor_maps", [](const PySensorModel& s) { return maps_dict(s.model()); },
           "Frozen per-pixel maps (height, width): theta_on, theta_off, leak, dark_current_fA, phi0, hot (0 none, "
           "1 leak type, 2 threshold type), hot_target_hz, tau_hp (s; 0 while the high-pass filter is off).")
      .def("stats",
           [](const PySensorModel& s) {
             const auto [st, gpu] = s.locked([](SensorModel& sm) { return std::pair(sm.stats(), sm.gpu_launch_stats()); });
             return stats_dict(st, gpu);
           },
           "Work counters since construction or reset_stats(), with the GPU launch counters (device 'cuda').")
      .def("reset_stats", [](PySensorModel& s) { s.locked([](SensorModel& sm) { sm.reset_stats(); }); })
      .def_prop_ro("glare_frame_mean",
                   [](const PySensorModel& s) {
                     return s.locked([](SensorModel& sm) { return std::vector<double>(sm.glare_frame_mean()); });
                   },
                   "Veiling glare: the frame means per light group at the model time, of gain x the inputs the "
                   "pixels hold (init's frame, then each listed pixel's last sample; a loaded state's own); each "
                   "advance computes its end's from its runs. Empty without glare.")
      .def_prop_ro("params", [](const PySensorModel& s) { return s.model().params(); })
      .def_prop_ro("median_dark_fA", [](const PySensorModel& s) { return median_dark_fA(s.model().params()); },
                   "Median dark current at the sensor's temperature (fA): dark_current_fA times its Arrhenius factor.")
      .def_prop_ro("width", [](const PySensorModel& s) { return s.model().width(); })
      .def_prop_ro("height", [](const PySensorModel& s) { return s.model().height(); })
      .def_prop_ro("groups", [](const PySensorModel& s) { return s.model().groups(); })
      .def_prop_ro("time_ns", [](const PySensorModel& s) { return s.locked([](SensorModel& sm) { return sm.time_ns(); }); })
      .def_prop_ro("num_threads", [](const PySensorModel& s) { return s.model().num_threads(); })
      .def_prop_ro("readout_phase_ns", [](const PySensorModel& s) { return s.model().readout_phase_ns(); })
      .def_prop_ro("stationary_ms", [](const PySensorModel& s) { return s.model().stationary_ms(); },
                   "Wall time of the last init's stationary probe and transfer (ms; 0 without one).")
      .def_prop_ro("stationary_probes", [](const PySensorModel& s) { return s.model().stationary_probes(); },
                   "Probe pixels of the last init's stationary state (0 without one).")
      .def_prop_ro("stationary_history_ns", [](const PySensorModel& s) { return s.model().stationary_history_ns(); },
                   "The last stationary probe's history T_b (ns; 0 without one).")
      .def_prop_ro("stationary_phase_kept", [](const PySensorModel& s) { return s.model().stationary_phase_kept(); },
                   "Pixels of the last stationary init that kept the phase law (no snapshot of their probe fits "
                   "inside their levels).")
      .def_prop_ro("irr_max", [](const PySensorModel& s) { return s.model().irr_max(); },
                   "Largest irradiance sample taken as is (renderer units); larger samples are clamped.")
      .def_prop_ro("precision", [](const PySensorModel& s) {
        return std::string(s.model().precision() == Precision::kFp32 ? "fp32" : "fp64");
      })
      .def_prop_ro("device", [](const PySensorModel& s) {
        return std::string(s.model().device() == Device::kCuda ? "cuda" : "cpu");
      });
  m.def(
      "rederive",
      [](nb::dict events, uint32_t width, uint32_t height, uint64_t seed, uint32_t camera, int64_t quantum_ns,
         int64_t delay_ns, int64_t clock_offset_ns, double clock_drift_ppm, double label_cut, bool relabel) {
        // The events of advance (x, y, p, t_cross_ns, label, flags, ds, dn, dl, l_first), any order.
        const auto col = [&](const char* k) { return nb::handle(events[k]); };
        const auto x = nb::cast<ExactContig<uint16_t, -1>>(col("x"), false);
        const size_t n = x.shape(0);
        const auto y = nb::cast<ExactContig<uint16_t, -1>>(col("y"), false);
        const auto p = nb::cast<ExactContig<uint8_t, -1>>(col("p"), false);
        const auto t = nb::cast<ExactContig<int64_t, -1>>(col("t_cross_ns"), false);
        const auto label = nb::cast<ExactContig<uint8_t, -1>>(col("label"), false);
        const auto flags = nb::cast<ExactContig<uint16_t, -1>>(col("flags"), false);
        const auto ds = nb::cast<ExactContig<float, -1>>(col("ds"), false);
        const auto dn = nb::cast<ExactContig<float, -1>>(col("dn"), false);
        const auto dl = nb::cast<ExactContig<float, -1>>(col("dl"), false);
        const auto lf = nb::cast<ExactContig<float, -1>>(col("l_first"), false);
        for (size_t k : {y.shape(0), p.shape(0), t.shape(0), label.shape(0), flags.shape(0), ds.shape(0), dn.shape(0),
                         dl.shape(0), lf.shape(0)}) {
          if (k != n) throw std::invalid_argument("rederive: every column needs one entry per event");
        }
        std::vector<RawEvent> ev(n);
        for (size_t i = 0; i < n; ++i) {
          if (x(i) >= width || y(i) >= height) throw std::invalid_argument("rederive: an event outside the frame");
          ev[i] = RawEvent{static_cast<uint32_t>(y(i)) * width + x(i), p(i), label(i), flags(i), t(i), ds(i), dn(i),
                           dl(i), lf(i)};
        }
        Rederived r;
        {
          nb::gil_scoped_release release;
          r = rederive(ev, Readout{quantum_ns, delay_ns, 0, clock_offset_ns, clock_drift_ppm}, label_cut, seed, camera,
                       width, order_y_bits(height), relabel);
        }
        nb::dict d;
        d["order"] = to_numpy(std::move(r.order), {n});
        d["t_reported_ns"] = to_numpy(std::move(r.t_reported), {n});
        d["label_s1"] = to_numpy(std::move(r.label_s1), {n});
        return d;
      },
      nb::arg("events"), nb::arg("width"), nb::arg("height"), nb::arg("seed"), nb::arg("camera"),
      nb::arg("quantum_ns"), nb::arg("delay_ns"), nb::arg("clock_offset_ns"), nb::arg("clock_drift_ppm"),
      nb::arg("label_cut"), nb::arg("relabel") = true,
      "The event tier (phase3/ui.md §4.7): an advance's events under a new readout (its counter phase drawn from "
      "seed, camera and quantum) and label cut. Returns order (the output order), and per input event t_reported_ns "
      "and label_s1 (S1). relabel=False (fp64 sessions) copies the labels.");
  m.def("cuda_available", &sensor_cuda_available,
        "True if this build has the CUDA sensor kernel and a CUDA device is visible.");

  // Test hooks: the kernel's building blocks, for oracle comparisons from Python.
  m.def("cascade_trace", &trace, nb::arg("t_ns").noconvert(), nb::arg("current_fA"), nb::arg("query_ns").noconvert(),
        nb::arg("k") = 1e16, nb::arg("tau_o") = 25e-6, nb::arg("tau_sf") = 100e-6, nb::arg("eps_L") = 0.002,
        nb::arg("precision") = "fp64",
        "Deterministic s3 at query_ns for a piecewise-linear photocurrent (fA) through the cascade, starting in "
        "steady state; returns (s3, substeps).");
  m.def(
      "z_step",
      [](const In<double, -1>& z0, const In<double, -1>& I0, const In<double, -1>& I1, const In<double, -1>& h,
         double k, const std::string& precision) {
        return parse_precision(precision) == Precision::kFp32 ? z_step_batch<float>(z0, I0, I1, h, k)
                                                              : z_step_batch<double>(z0, I0, I1, h, k);
      },
      nb::arg("z0"), nb::arg("I0"), nb::arg("I1"), nb::arg("h"), nb::arg("k"), nb::arg("precision") = "fp64",
      "Stage-1 closed form z(h) for linear photocurrent I0 -> I1 (units: fA, s, k in (fA s)^-1).");
  m.def(
      "special",
      [](const std::string& name, const In<double, -1>& x, const std::string& precision) {
        return parse_precision(precision) == Precision::kFp32 ? special_batch<float>(name, x)
                                                              : special_batch<double>(name, x);
      },
      nb::arg("name"), nb::arg("x"), nb::arg("precision") = "fp64",
      "Deterministic kernel transcendental: exp, log, expm1, log1p, erfcx or dawson.");
}
