// ballasim._render: the renderer of contract C2 (docs/dev/CONTRACTS.md, docs/dev/RENDER.md).
//
// The scene bundle is accepted as nested plain dicts of NumPy arrays (contract C1) or as objects
// exposing the same names as attributes (Stream S's dataclasses), so no Python-side adapter is
// needed.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "bindings/config_value.h"
#include "bindings/ndarray_util.h"
#include "render/bindings/bsdf_bindings.h"
#include "render/bindings/bundle_convert.h"
#include "render/bindings/nir_rows.h"
#include "render/bindings/reflection_bindings.h"
#include "render/param_fields.h"
#include "render/ao_proxy.h"
#include "render/shade/ao_traced.h"
#include "render/patches.h"
#include "render/renderer.h"
#include "render/texture.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace balla;
using namespace balla::render;
using balla::bindings::to_numpy;
using balla::render::bindings::float_vector;
using balla::render::bindings::gt_dict;
using balla::render::bindings::int_vector;
using balla::render::bindings::to_bundle;

namespace {

class PyRenderer {
 public:
  PyRenderer(nb::handle bundle, const std::string& backend, uint32_t n_subsamples,
             std::tuple<double, double, double> weights, uint64_t seed, double aperture_px, double blur_sigma_px,
             double truncation_sigmas, int32_t warp_cells, int32_t lut_samples_per_px, int num_threads,
             uint64_t max_rays_per_launch, const std::string& pixel_integral, bool unlit_mask,
             const std::string& texture_shadow, const std::string& shading_patches, const std::string& mover_ao,
             const std::vector<std::vector<double>>& nir_fraction) {
    RendererOptions o;
    if (backend == "optix") {
      o.backend = BackendKind::kOptix;
    } else if (backend == "embree") {
      o.backend = BackendKind::kEmbree;
    } else {
      throw std::invalid_argument("backend must be 'optix' or 'embree'");
    }
    o.scene.n_subsamples = n_subsamples;
    o.scene.sensor_rgb_weights = {std::get<0>(weights), std::get<1>(weights), std::get<2>(weights)};
    o.scene.seed = seed;
    o.scene.aperture_px = aperture_px;
    o.scene.blur_sigma_px = blur_sigma_px;
    o.scene.truncation_sigmas = truncation_sigmas;
    o.scene.warp_cells = warp_cells;
    o.scene.lut_samples_per_px = lut_samples_per_px;
    if (pixel_integral == "phase1") {
      o.scene.pixel_integral = PixelIntegral::kPhase1;
    } else if (pixel_integral == "fixed64") {
      o.scene.pixel_integral = PixelIntegral::kFixed64;
    } else {
      throw std::invalid_argument("pixel_integral must be 'phase1' or 'fixed64'");
    }
    o.scene.unlit_mask = unlit_mask;
    if (texture_shadow == "pooled") {
      o.scene.texture_shadow = TextureShadow::kPooled;
    } else if (texture_shadow == "weighted") {
      o.scene.texture_shadow = TextureShadow::kWeighted;
    } else {
      throw std::invalid_argument("texture_shadow must be 'pooled' or 'weighted'");
    }
    if (shading_patches == "material") {
      o.scene.shading_patches = ShadingPatches::kMaterial;
    } else if (shading_patches == "crease") {
      o.scene.shading_patches = ShadingPatches::kCrease;
    } else {
      throw std::invalid_argument("shading_patches must be 'material' or 'crease'");
    }
    if (mover_ao == "off") {
      o.scene.mover_ao = MoverAo::kOff;
    } else if (mover_ao == "proxy") {
      o.scene.mover_ao = MoverAo::kProxy;
    } else if (mover_ao == "traced") {
      o.scene.mover_ao = MoverAo::kTraced;
    } else {
      throw std::invalid_argument("mover_ao must be 'off', 'proxy' or 'traced'");
    }
    o.num_threads = num_threads;
    o.max_rays_per_launch = max_rays_per_launch;
    o.nir_fraction = nir_fraction;
    renderer_ = std::make_unique<Renderer>(to_bundle(bundle), o);
  }
  PyRenderer(nb::handle bundle, const RendererOptions& o) : renderer_(std::make_unique<Renderer>(to_bundle(bundle), o)) {}

  // The renderer of a run's resolved values (Resolved.values()), through the rows (render::apply_renderer).
  static PyRenderer from_values(nb::handle bundle, nb::handle values) {
    RendererOptions o = apply_renderer(balla::bindings::values_from_py(
        values, {&scene_settings_rows(), &renderer_options_rows()}));
    // P2-F F11a: each bundle camera's sensor.nir_fraction (an engine row, per camera), by name (nir_rows.h).
    std::vector<std::string> names;
    size_t i = 0;
    for (nb::handle c : balla::render::bindings::field(bundle, "cameras", "bundle")) {
      names.push_back(balla::render::bindings::to_camera(c, i++).name);
    }
    o.nir_fraction = balla::render::bindings::nir_rows(names, values);
    return PyRenderer(bundle, o);
  }

  void build() {
    nb::gil_scoped_release release;
    renderer_->build();
  }

  nb::dict render(nb::handle requests, bool sub_instance, bool centre, bool sample_class, nb::handle pv) {
    const auto cam = int_vector<uint16_t>(requests.attr("__getitem__")("cam"), "uint16", "requests['cam']");
    const auto pixel = int_vector<uint32_t>(requests.attr("__getitem__")("pixel"), "uint32", "requests['pixel']");
    const auto t = int_vector<int64_t>(requests.attr("__getitem__")("t_ns"), "int64", "requests['t_ns']");
    uint8_t pv_mode = 0;
    if (!pv.is_none()) {
      const std::string m = nb::cast<std::string>(pv);
      if (m == "edge") {
        pv_mode = 1;
      } else if (m == "all") {
        pv_mode = 2;
      } else {
        throw std::invalid_argument("render: pv must be None, 'edge' or 'all'");
      }
    }
    RenderResult r;
    {
      nb::gil_scoped_release release;
      RenderFlags flags{sub_instance, centre, sample_class};
      flags.pv = pv_mode;
      r = renderer_->render(cam, pixel, t, flags);
    }
    const size_t n = r.count;
    nb::dict d;
    d["irradiance"] = to_numpy(std::move(r.irradiance), {n, static_cast<size_t>(r.num_groups)});
    d["signature"] = to_numpy(std::move(r.signature), {n});
    nb::dict stats;
    stats["trace_ms"] = r.stats.trace_ms;
    stats["shade_ms"] = r.stats.shade_ms;
    stats["host_ms"] = r.stats.host_ms;
    d["stats"] = stats;
    if (sub_instance) d["sub_instance"] = to_numpy(std::move(r.sub_instance), {n, static_cast<size_t>(r.subsamples)});
    if (sample_class) d["sample_class"] = to_numpy(std::move(r.sample_class), {n});
    if (pv_mode != 0) {
      std::vector<uint8_t> w8(r.pv_written.begin(), r.pv_written.end());
      d["pv_written"] = to_numpy(std::move(w8), {n});
      d["pv_codes"] = to_numpy(std::move(r.pv_codes), {n, static_cast<size_t>(r.pv_code_words)});
      d["pv_table"] = to_numpy(std::move(r.pv_table), {n, static_cast<size_t>(r.pv_table_words)});
    }
    if (centre) {
      std::vector<float> flow(2 * n);
      for (size_t i = 0; i < n; ++i) {
        flow[2 * i] = r.centre[i].flow_x;
        flow[2 * i + 1] = r.centre[i].flow_y;
      }
      nb::dict g = gt_dict(std::move(r.centre), {n});
      d["centre_instance"] = g["instance"];
      d["centre_semantic"] = g["semantic"];
      d["centre_depth"] = g["depth"];
      d["centre_flow"] = to_numpy(std::move(flow), {n, 2});
    }
    return d;
  }

  // P2-D D2: probes (Renderer::probe).
  nb::dict probe(nb::handle requests) {
    const auto get = [&](const char* k) { return requests.attr("__getitem__")(k); };
    const auto cam = int_vector<uint16_t>(get("cam"), "uint16", "requests['cam']");
    const auto pixel = int_vector<uint32_t>(get("pixel"), "uint32", "requests['pixel']", cam.size());
    const auto k = int_vector<uint32_t>(get("k"), "uint32", "requests['k']", cam.size());
    const auto t = int_vector<int64_t>(get("t_ns"), "int64", "requests['t_ns']", cam.size());
    const auto light = int_vector<uint32_t>(get("light"), "uint32", "requests['light']", cam.size());
    ProbeResult r;
    {
      nb::gil_scoped_release release;
      r = renderer_->probe(cam, pixel, k, t, light);
    }
    const size_t n = r.out.size();
    std::vector<uint32_t> inst(n), surface(n), mask(n);
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) {
      inst[i] = r.out[i].instance;
      surface[i] = r.out[i].surface;
      mask[i] = r.out[i].mask;
      f[i] = r.out[i].f;
    }
    nb::dict d;
    d["instance"] = to_numpy(std::move(inst), {n});
    d["surface"] = to_numpy(std::move(surface), {n});
    d["mask"] = to_numpy(std::move(mask), {n});
    d["f"] = to_numpy(std::move(f), {n});
    nb::dict stats;
    stats["probe_ms"] = r.stats.probe_ms;
    stats["host_ms"] = r.stats.host_ms;
    d["stats"] = stats;
    return d;
  }

  nb::dict gt(uint16_t cam, nb::handle x, nb::handle y, nb::handle t_ns, bool states, bool normals) {
    const auto xs = float_vector(x, "x");
    const auto ys = float_vector(y, "y");
    if (xs.size() != ys.size()) throw std::invalid_argument("x and y need the same length");
    const auto ts = int_vector<int64_t>(t_ns, "int64", "t_ns", xs.size());
    const size_t n = xs.size();
    if (states || normals) {
      // P2-D D1f: the engine's plan pass on the host (Renderer::gt_into: the shadow states and geometric normals).
      if (renderer_->backend() != BackendKind::kEmbree) throw std::invalid_argument("gt: states and normals need embree");
      if (cam >= renderer_->bundle().cameras.size()) throw std::invalid_argument("gt: camera index out of range");
      renderer_->check_gt(cam, xs.data(), ys.data(), ts.data(), n);
      std::vector<GtQuery> q(n);
      for (size_t i = 0; i < n; ++i) q[i] = GtQuery{ts[i], xs[i], ys[i]};
      const auto bits = static_cast<size_t>(states ? renderer_->num_shadow_lights() : 0);
      std::vector<GtSample> out(n);
      std::vector<uint32_t> st(2 * n * bits);
      std::vector<float> nrm(normals ? 3 * n : 0);
      {
        nb::gil_scoped_release release;
        renderer_->gt_into(cam, q.data(), n, out.data(), nullptr, bits > 0 ? st.data() : nullptr,
                           normals ? nrm.data() : nullptr);
      }
      nb::dict d = gt_dict(std::move(out), {n});
      if (states) d["states"] = to_numpy(std::move(st), {n, bits, 2});
      if (normals) d["normal"] = to_numpy(std::move(nrm), {n, 3});
      return d;
    }
    const std::vector<uint16_t> cams(n, cam);
    std::vector<GtSample> out;
    {
      nb::gil_scoped_release release;
      out = renderer_->gt(cams, xs, ys, ts);
    }
    return gt_dict(std::move(out), {n});
  }

  nb::dict dense(uint16_t cam, int64_t t_ns, const std::vector<std::string>& channels) {
    bool irr = false, geo = false, shadow_factor = false;
    for (const std::string& c : channels) {
      if (c == "irradiance") {
        irr = true;
      } else if (c == "shadow_factor") {
        shadow_factor = true;
      } else if (c == "depth" || c == "instance" || c == "semantic" || c == "flow") {
        geo = true;
      } else {
        throw std::invalid_argument("dense: unknown channel '" + c +
                                    "' (depth, instance, semantic, flow, irradiance, shadow_factor)");
      }
    }
    if (cam >= renderer_->bundle().cameras.size()) throw std::invalid_argument("dense: camera index out of range");
    const auto h = static_cast<size_t>(renderer_->bundle().cameras[cam].height);
    const auto w = static_cast<size_t>(renderer_->bundle().cameras[cam].width);
    DenseFrame f;
    {
      nb::gil_scoped_release release;
      f = renderer_->dense(cam, t_ns, irr, geo, shadow_factor);
    }
    nb::dict d;
    nb::dict g;
    if (geo) {
      std::vector<float> flow(2 * f.gt.size());
      for (size_t i = 0; i < f.gt.size(); ++i) {
        flow[2 * i] = f.gt[i].flow_x;
        flow[2 * i + 1] = f.gt[i].flow_y;
      }
      g = gt_dict(std::move(f.gt), {h, w});
      g["flow"] = to_numpy(std::move(flow), {h, w, 2});
    }
    for (const std::string& c : channels) {
      if (d.contains(c.c_str())) continue;  // a repeated channel: f.irradiance is moved out once
      if (c == "irradiance") {
        d["irradiance"] = to_numpy(std::move(f.irradiance), {h, w, static_cast<size_t>(renderer_->num_groups())});
      } else if (c == "shadow_factor") {
        d["shadow_factor"] = to_numpy(std::move(f.shadow_factor), {h, w, static_cast<size_t>(renderer_->num_groups())});
      } else {
        d[c.c_str()] = g[c.c_str()];
      }
    }
    return d;
  }

  // Scenes with baked proxies (scans.md §4.2): the start diagnostic and the split count, as the engine reports them.
  std::vector<float> proxy_visibility(uint16_t cam, int64_t t_ns) {
    nb::gil_scoped_release release;
    return renderer_->proxy_visibility(cam, t_ns);
  }
  int32_t split_geometries() {
    nb::gil_scoped_release release;
    return renderer_->split_geometries();
  }

  // Held movers (ENGINE_CPP I3b; the engine's own call), for tests and the timing studies: merges the bodies `ids` into the
  // world at their poses at t_ns. False, and nothing held, in a scene with baked proxies (holds_movers).
  bool hold(const std::vector<uint32_t>& ids, int64_t t_ns) {
    nb::gil_scoped_release release;
    return renderer_->hold(ids, t_ns);
  }
  bool holds_movers() const { return renderer_->holds_movers(); }

  void set_window(int64_t base_ns) { renderer_->set_window(base_ns); }
  int64_t window() const { return renderer_->window(); }
  std::string backend() const { return renderer_->backend() == BackendKind::kOptix ? "optix" : "embree"; }
  uint32_t n_subsamples() const { return renderer_->options().scene.n_subsamples; }
  int32_t num_groups() const { return renderer_->num_groups(); }
  bool has_specular() const { return renderer_->has_specular(); }  // any material with a lobe (P2-F F6)
  std::vector<std::string> light_groups() const { return renderer_->bundle().light_groups; }
  bool built() const { return renderer_->built(); }
  bool nir_two_channel() const { return renderer_->nir_two_channel(); }
  bool pv_supported() const { return renderer_->pv_supported(); }

 private:
  std::unique_ptr<Renderer> renderer_;
};

std::vector<std::string> backends() {
  std::vector<std::string> b{"embree"};
  if (optix_available()) b.emplace_back("optix");
  return b;
}

// P2-F F5 (render/patches.h): the shading patches of one mesh, for tests and the patch census.
nb::object shading_patches(nb::handle positions, nb::handle indices, nb::handle normals, nb::handle materials,
                           nb::handle specular) {
  nb::object np = nb::module_::import_("numpy");
  const auto flat = [&](nb::handle a) { return np.attr("ravel")(np.attr("asarray")(a)); };
  const std::vector<float> pos = float_vector(flat(positions), "positions");
  const std::vector<uint32_t> idx = int_vector<uint32_t>(flat(indices), "uint32", "indices");
  if (pos.size() % 3 != 0 || idx.size() % 3 != 0) {
    throw std::invalid_argument("shading_patches: positions (V, 3) and indices (T, 3)");
  }
  const size_t nt = idx.size() / 3;
  const std::vector<float> nrm = normals.is_none() ? std::vector<float>{} : float_vector(flat(normals), "normals");
  const std::vector<uint32_t> mats =
      materials.is_none() ? std::vector<uint32_t>(nt, 0u)
                          : int_vector<uint32_t>(flat(materials), "uint32", "materials");
  const std::vector<uint8_t> spec =
      specular.is_none() ? std::vector<uint8_t>{}
                         : int_vector<uint8_t>(np.attr("asarray")(flat(specular), "dtype"_a = "uint8"), "uint8",
                                               "specular");
  Patches p = build_patches({pos, nrm, idx}, mats, spec);
  return nb::cast(to_numpy(std::move(p.of_triangle), {nt}));
}

// P2-F F9a (render/ao_proxy.h): the mover-AO proxy of one mesh (body frame), for tests and the O-AO references.
nb::dict ao_proxy(nb::handle positions, nb::handle indices) {
  nb::object np = nb::module_::import_("numpy");
  const auto flat = [&](nb::handle a) { return np.attr("ravel")(np.attr("asarray")(a)); };
  const std::vector<float> pos = float_vector(flat(positions), "positions");
  const std::vector<uint32_t> idx = int_vector<uint32_t>(flat(indices), "uint32", "indices");
  if (pos.size() % 3 != 0 || idx.size() % 3 != 0) {
    throw std::invalid_argument("ao_proxy: positions (V, 3) and indices (T, 3)");
  }
  for (const uint32_t i : idx) {
    if (static_cast<size_t>(i) >= pos.size() / 3) throw std::invalid_argument("ao_proxy: index out of range");
  }
  const AoProxy p = fit_ao_proxy(pos, idx);
  std::vector<double> centres, radii;
  for (const AoProxySphere& sp : p.spheres) {
    centres.insert(centres.end(), sp.centre.begin(), sp.centre.end());
    radii.push_back(sp.radius);
  }
  const size_t k = radii.size();
  nb::dict out;
  out["centres"] = nb::cast(to_numpy(std::move(centres), {k, 3}));
  out["radii"] = nb::cast(to_numpy(std::move(radii), {k}));
  out["radius_b"] = p.radius_b;
  out["d0"] = p.d0;
  return out;
}

// P2-F F9b (render/shade/ao_traced.h): the traced AO's first n local directions (about +z) of a pixel whose sub-sample
// pattern seed is pattern_seed, for tests and the bias study.
nb::object ao_directions(uint32_t pattern_seed, uint32_t n) {
  std::vector<float> d(3 * static_cast<size_t>(n));
  const uint32_t seed = ao_direction_seed(pattern_seed);
  for (uint32_t k = 0; k < n; ++k) {
    const Vec3f w = ao_local_direction(seed, k);
    d[3 * k] = w.x;
    d[3 * k + 1] = w.y;
    d[3 * k + 2] = w.z;
  }
  return nb::cast(to_numpy(std::move(d), {static_cast<size_t>(n), 3}));
}

// `ballasim scan report`'s material entry (scans.md §3.6, §4.6; S2c): render::texture_stats of one C1 material, the
// counts as shares. clipped_share and the step shares are None without an 8-bit texture, and so is a step share
// when every texel is black or clipped.
nb::dict texture_stats_py(nb::handle material, std::tuple<double, double, double> weights, double theta) {
  const Material m = balla::render::bindings::to_material(material, "material");
  const TextureStats s = texture_stats(m, {std::get<0>(weights), std::get<1>(weights), std::get<2>(weights)}, theta);
  const bool srgb8 = s.source == TextureSource::kSrgb8;
  const auto share = [](int64_t k, int64_t n) -> nb::object {
    return n > 0 ? nb::cast(static_cast<double>(k) / static_cast<double>(n)) : nb::none();
  };
  const int64_t stepped = s.texels - s.black - s.clipped;
  nb::dict d;
  d["source"] = s.source == TextureSource::kNone ? "none" : (srgb8 ? "srgb8" : "f16");
  d["size"] = s.source == TextureSource::kNone ? nb::none() : nb::cast(std::vector<int32_t>{s.width, s.height});
  d["texels"] = s.texels;
  d["clipped_share"] = srgb8 ? share(s.clipped, s.texels) : nb::none();
  d["black_share"] = share(s.black, s.texels);
  d["mono_p1_p50_p99"] = std::vector<double>(s.mono_p1_p50_p99.begin(), s.mono_p1_p50_p99.end());
  d["step_gt_theta_share"] = srgb8 ? share(s.step_gt_theta, stepped) : nb::none();
  d["step_gt_theta4_share"] = srgb8 ? share(s.step_gt_theta4, stepped) : nb::none();
  return d;
}

}  // namespace

NB_MODULE(_render, m) {
  // Keyword defaults: the members' initializers (the parameter rows' literals, render/param_fields.h).
  const balla::render::SceneSettings kSettings{};
  const RendererOptions kOptions{};
  m.doc() = "BallaSim renderer (contract C2): exact-pose multi-trace with per-cluster shading";
  m.def("backends", &backends, "Backends usable in this process: 'embree' always, 'optix' with a GPU.");
  m.def("shading_patches", &shading_patches, "positions"_a, "indices"_a, "normals"_a = nb::none(),
        "materials"_a = nb::none(), "specular"_a = nb::none(),
        "The shading patch of every triangle of one mesh (accuracy.shading_patches crease, render/patches.h): cut at "
        "material changes, between disconnected shells, at hard edges above 30 deg or on specular materials, and above "
        "30 deg without normals; numbered by first triangle.");
  m.def("ao_proxy", &ao_proxy, "positions"_a, "indices"_a,
        "The mover-AO proxy of one mesh (accuracy.mover_ao proxy, render/ao_proxy.h): {'centres': (k, 3), 'radii': "
        "(k,), 'radius_b': R_b, 'd0': the window's start}, k <= 8 spheres fitted by k-means to area-weighted samples.");
  m.def("ao_directions", &ao_directions, "pattern_seed"_a, "n"_a,
        "The first n ambient directions (n, 3) float32 about +z of a pixel with this sub-sample pattern seed "
        "(accuracy.mover_ao traced, render/shade/ao_traced.h): cosine-distributed, the first 2^m a stratified net.");
  nb::class_<PyRenderer>(m, "Renderer",
                         "Renderer(bundle, backend='optix', n_subsamples, sensor_rgb_weights, *, seed, ...):\n"
                         "the keyword defaults are SceneSettings' and RendererOptions' (PARAMS_REFERENCE.md).\n\n"
                         "bundle: scene bundle of contract C1 as nested dicts of NumPy arrays (or objects with\n"
                         "the same attribute names). Irradiance is linear, pixel-area integrated over the fixed\n"
                         "Owen-Sobol sub-samples, per light group, in relative units (docs/dev/RENDER.md).")
      .def(nb::init<nb::handle, const std::string&, uint32_t, std::tuple<double, double, double>, uint64_t, double,
                    double, double, int32_t, int32_t, int, uint64_t, const std::string&, bool, const std::string&,
                     const std::string&, const std::string&, const std::vector<std::vector<double>>&>(),
           "bundle"_a, "backend"_a = "optix", "n_subsamples"_a = kSettings.n_subsamples,
           "sensor_rgb_weights"_a = std::make_tuple(kSettings.sensor_rgb_weights[0], kSettings.sensor_rgb_weights[1],
                                                    kSettings.sensor_rgb_weights[2]),
           nb::kw_only(), "seed"_a = kSettings.seed, "aperture_px"_a = kSettings.aperture_px,
           "blur_sigma_px"_a = kSettings.blur_sigma_px, "truncation_sigmas"_a = kSettings.truncation_sigmas,
           "warp_cells"_a = kSettings.warp_cells, "lut_samples_per_px"_a = kSettings.lut_samples_per_px,
           "num_threads"_a = kOptions.num_threads,
           "max_rays_per_launch"_a = kOptions.max_rays_per_launch, "pixel_integral"_a = "phase1",
           "unlit_mask"_a = kSettings.unlit_mask, "texture_shadow"_a = "pooled", "shading_patches"_a = "material",
           "mover_ao"_a = "off", "nir_fraction"_a = std::vector<std::vector<double>>{})
      .def_static("from_values", &PyRenderer::from_values, "bundle"_a, "values"_a,
                  "The renderer of a run's resolved values (ballasim.config.Resolved.values(): {run: {...}, cameras: "
                  "{name: {...}}}), applied through the SceneSettings and RendererOptions rows as the engine applies "
                  "them (run.device gpu: OptiX, cpu: Embree; the cameras agree on the run-wide camera keys).")
      .def("build", &PyRenderer::build,
           "Builds BVHs, linear mono mip textures and per-camera LUTs (implicit on the first query).")
      .def("set_window", &PyRenderer::set_window, "t_base_ns"_a,
           "Base of the int32 time offsets; flushes the lazily filled pose table.")
      .def("render", &PyRenderer::render, "requests"_a, nb::kw_only(), "sub_instance"_a = false, "centre"_a = false,
           "sample_class"_a = false, "pv"_a = nb::none(),
           "requests: structured array (or dict) with fields cam (u16), pixel (u32, y*W + x), t_ns (i64).\n"
           "Returns irradiance (n, G) f32, signature (n,) u64, stats (the call's stage times in ms: trace_ms,\n"
           "shade_ms, host_ms), and optionally sub_instance (n, N) u32, centre_instance, centre_semantic,\n"
           "centre_depth, centre_flow (n, 2) px/s, and sample_class (n,) u32 (the class word, render/types.h).\n"
           "pv 'edge' or 'all' (P2-D D2, render/shade/codes.h): P&V's outputs of the edge-class requests or of all,\n"
           "pv_written (n,) u8, pv_codes (n, words) u32 and pv_table (n, words) u32 in request order (zeros where none).")
      .def("probe", &PyRenderer::probe, "requests"_a,
           "P2-D D2: the state of single sub-samples (render/shade/probe.h). requests: fields cam (u16), pixel (u32),\n"
           "k (u32, the sub-sample), t_ns (i64), light (u32, the shadow light of f). Returns instance, surface (the\n"
           "material), mask (u32: the shadow mask as the signature takes it), f (f32: that light's Lambert term) and\n"
           "stats (probe_ms, host_ms).")
      .def("gt", &PyRenderer::gt, "cam"_a, "x"_a, "y"_a, "t_ns"_a, nb::kw_only(), "states"_a = false,
           "normals"_a = false,
           "Centre-ray ground truth at pixel positions (centres at integers): instance u32, semantic u16,\n"
           "depth f32 (camera z, NaN on a miss), flow_x, flow_y f32 (px/s, NaN on a miss). Embree only, the\n"
           "engine's plan pass (Renderer::gt_into): states (n, shadow lights, 2) u32, per shadow light the state\n"
           "and the blocker's ray parameter's float32 bits; normals (n, 3) f32, the hit triangle's unit geometric\n"
           "normal in the camera frame, facing the camera (NaN on a miss).")
      .def("dense", &PyRenderer::dense, "cam"_a, "t_ns"_a,
           "channels"_a = std::vector<std::string>{"depth", "instance", "flow", "irradiance"},
           "Dense frames at one time: depth, instance, semantic (H, W), flow (H, W, 2), irradiance (H, W, G), and\n"
           "shadow_factor (H, W, G): the per-group ratio of the irradiance to the irradiance with every proxy factor\n"
           "set to 1 (docs/dev/SCANS.md; 1 where that is 0, and everywhere without baked proxies).")
      .def("proxy_visibility", &PyRenderer::proxy_visibility, "cam"_a, "t_ns"_a,
           "Scenes with baked proxies: per proxy shadow light (bundle order), the share of the proxy rays from the\n"
           "baked centre-ray hits of the camera at t_ns that the scan blocks (the start diagnostic); [] otherwise.")
      .def("hold", &PyRenderer::hold, "ids"_a, "t_ns"_a,
           "Traces the movers `ids` as part of the static world at their poses at t_ns until the next call (I3b); the\n"
           "caller guarantees they are still over every time it then renders. Returns whether it rebuilt; False and\n"
           "nothing held in a scene with baked proxies (holds_movers).")
      .def_prop_ro("holds_movers", &PyRenderer::holds_movers)
      .def("split_geometries", &PyRenderer::split_geometries,
           "The static bodies split from the scan geometry (trace geometries of their own); 0 without baked proxies.")
      .def_prop_ro("window", &PyRenderer::window)
      .def_prop_ro("backend", &PyRenderer::backend)
      .def_prop_ro("n_subsamples", &PyRenderer::n_subsamples)
      .def_prop_ro("num_groups", &PyRenderer::num_groups)
      .def_prop_ro("has_specular", &PyRenderer::has_specular)
      .def_prop_ro("light_groups", &PyRenderer::light_groups)
      .def_prop_ro("built", &PyRenderer::built)
      // P2-F F11a: sensor.nir_fraction differs between (camera, light group) pairs (two-channel albedo textures); whether
      // render accepts a P&V request (False on the device under it).
      .def_prop_ro("nir_two_channel", &PyRenderer::nir_two_channel)
      .def_prop_ro("pv_supported", &PyRenderer::pv_supported);
  balla::render::bindings::bind_bsdf(m);  // bsdf_eval, bsdf_alpha_eff, bsdf_tables (p2f.md §6.3)
  balla::render::bindings::bind_reflection(m);  // reflection_policy (DESIGN §7.7, p2f.md §4.3)
  m.def("texture_stats", &texture_stats_py, "material"_a,
        "sensor_rgb_weights"_a = std::make_tuple(kSettings.sensor_rgb_weights[0], kSettings.sensor_rgb_weights[1],
                                                 kSettings.sensor_rgb_weights[2]),
        "theta"_a = 0.25,
        "S2c (docs/design/phase3/scans.md §4.6): the texture diagnostics of one C1 material (Material or dict), as\n"
        "the renderer reads it (texel value sum_c w_c b_c x_c): source ('srgb8', 'f16' or 'none'), size [W, H],\n"
        "texels, clipped_share (8-bit: some channel >= 254), black_share (every channel 0), mono_p1_p50_p99 (over\n"
        "the texels that are not black), step_gt_theta_share and step_gt_theta4_share (8-bit texels neither black\n"
        "nor clipped whose one-code step, every channel + 1, exceeds theta and theta / 4 in log units).");
}
