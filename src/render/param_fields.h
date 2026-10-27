// The SceneSettings and RendererOptions rows (docs/dev/P0_PARAMS.md §1.3, §3.3; sources: PARAMETERS.md §3.2).
// Header-only: the member pointers need render/renderer.h and link nothing.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "core/config/param.h"
#include "core/rng/warp_table.h"
#include "render/renderer.h"
#include "render/types.h"

namespace balla::config {

template <>
struct TargetOf<render::SceneSettings> {
  static constexpr Target value = Target::kSceneSettings;
};
template <>
struct TargetOf<render::RendererOptions> {
  static constexpr Target value = Target::kRendererOptions;
};
template <>
struct EnumNames<render::PixelIntegral> {
  static constexpr std::array<std::string_view, 2> names{"phase1", "fixed64"};
};
template <>
struct EnumNames<render::TextureShadow> {
  static constexpr std::array<std::string_view, 2> names{"pooled", "weighted"};
};
template <>
struct EnumNames<render::ShadingPatches> {
  static constexpr std::array<std::string_view, 2> names{"material", "crease"};
};
template <>
struct EnumNames<render::MoverAo> {
  static constexpr std::array<std::string_view, 3> names{"off", "proxy", "traced"};
};

}  // namespace balla::config

namespace balla::render {

// Members of RendererOptions without a row of their own, named by pointer (a renamed or removed member breaks the
// build; the schema test counts them).
inline constexpr auto kRendererOptionsExcluded = std::tuple{&RendererOptions::scene, &RendererOptions::nir_fraction};

namespace param_detail {

// run.device: gpu is OptiX with the CUDA fp32 sensor, cpu is Embree with the CPU sensor (the only pairs since
// ENGINE_CPP I4); resolution replaces auto by gpu or cpu before anything is applied.
inline void set_device(const config::Param& p, RendererOptions& o, const config::Value& v) {
  config::check_value(p, v, config::display_name(p));
  const std::string& s = std::get<std::string>(v);
  if (s == "auto") throw std::invalid_argument(config::display_name(p) + " = 'auto': resolve it to gpu or cpu first");
  o.backend = s == "gpu" ? BackendKind::kOptix : BackendKind::kEmbree;
}
inline config::Value get_device(const RendererOptions& o) {
  return std::string(o.backend == BackendKind::kOptix ? "gpu" : "cpu");
}
// execution.pose_cache_mb: MB << 20 bytes.
inline void set_pose_cache(const config::Param& p, RendererOptions& o, const config::Value& v) {
  config::check_value(p, v, config::display_name(p));
  o.pose_cache_bytes = static_cast<uint64_t>(std::get<int64_t>(v)) << 20;
}
inline config::Value get_pose_cache(const RendererOptions& o) { return static_cast<int64_t>(o.pose_cache_bytes >> 20); }

}  // namespace param_detail

// The SceneSettings rows, in member order.
inline const std::vector<config::Param>& scene_settings_rows() {
  using namespace config;
  using S = SceneSettings;
  using C = config::Class;
  const auto render = Stage::kRender;
  static const std::vector<Param> rows{
      member<&S::n_subsamples>("accuracy.n_subsamples",
                               {.unit = "1", .hard = closed(1, kMaxSubsamples), .status = Status::kDerived,
                                .cls = C::kNAccuracy, .stage = render, .source = "PARAMETERS §4.2 (DESIGN §6.6); D8",
                                .help = "Sub-samples per pixel-sample N (every power of 2 is a nested (0,m,2)-net, "
                                        "sobol.h)"}),
      member<&S::sensor_rgb_weights>("sensor.rgb_weights",
                                     {.unit = "1", .def = Default::kFamily, .hard = at_least(0),
                                      .soft = closed(0.11, 0.54), .cls = C::kTSensor, .stage = render,
                                      .field = "sensor_rgb_weights", .source = "range: R8 t8 (white LED to fluorescent)",
                                      .help = "Sensor RGB weights of the band irradiance; one set per run (cameras "
                                              "must agree)"}),
      member<&S::seed>("run.render_seed", {.unit = "1", .hard = at_least(0), .status = Status::kChoice,
                                           .cls = C::kTRun, .stage = render, .field = "seed",
                                           .source = "arbitrary (DESIGN §6.1)",
                                           .help = "Seed of the Owen-scrambled sub-sample patterns"}),
      member<&S::aperture_px>("sensor.aperture_px",
                              {.unit = "px", .def = Default::kFamily, .hard = at_least(0), .soft = closed(0, 1),
                               .cls = C::kTSensor, .stage = render,
                               .source = "range: sqrt(fill factor) pitch to pitch (S2 OPT-07)",
                               .help = "Box width of the pixel response; one per run (cameras must agree)"}),
      member<&S::blur_sigma_px>("optics.blur_sigma_px",
                                {.unit = "px", .hard = at_least(0), .soft = closed(0.1, 1),
                                 .status = Status::kPlaceholder, .evimo2 = true, .cls = C::kTScene, .stage = render,
                                 .source = "G6 §2.8: prior of a low-cost 8 mm lens at 15 um (EVIMO2), no MTF data",
                                 .help = "Optical Gaussian blur of the pixel response; one per run (cameras must "
                                         "agree); from optics.blur_sigma_um when set"}),
      member<&S::truncation_sigmas>("accuracy.psf_truncation_sigmas",
                                    {.unit = "1", .hard = above(0), .soft = closed(2, 4), .status = Status::kDerived,
                                     .cls = C::kNAccuracy, .stage = render, .field = "truncation_sigmas",
                                     .source = "PARAMETERS §4.2",
                                     .help = "Truncation of the pixel response at w/2 + n sigma (the sub-sample "
                                             "support)"}),
      member<&S::warp_cells>("accuracy.warp_cells",
                             {.unit = "1", .hard = closed(rng::kMinWarpCells, rng::kMaxWarpCells),
                              .status = Status::kDerived, .cls = C::kNAccuracy, .stage = render,
                              .source = "PARAMETERS §4.2",
                              .help = "Cells of the tabulated inverse CDF of the pixel response"}),
      member<&S::lut_samples_per_px>("accuracy.lut_samples_per_px",
                                     {.unit = "1/px", .hard = at_least(1), .status = Status::kDerived,
                                      .cls = C::kNAccuracy, .stage = render, .source = "PARAMETERS §4.2 (DESIGN §8.1)",
                                      .help = "Nodes per pixel of the cameras' fp32 ray LUTs"}),
      member<&S::pixel_integral>(
          "accuracy.pixel_integral",
          {.status = Status::kChoice, .cls = C::kNAccuracy, .stage = render,
           .source = "PHASE2 D-P2-3 (P2-B B3; render/shade/shade.h): fixed64 for fidelity, balanced; phase1 for phase1, "
                     "reference and EVIMO2 v0 / v1 sensor profiles",
           .help = "Pixel integral: phase1 (all terms over N) or fixed64 (points, normals, textures over k < 64; the "
                   "rest over N)"}),
      // P2-C C2 (plan p2c §3.2): zero-amplitude shadow flips removed at the source; no profile adopts it yet.
      member<&S::unlit_mask>(
          "accuracy.unlit_mask",
          {.status = Status::kChoice, .cls = C::kNAccuracy, .stage = render,
           .source = "PHASE2 P2-C C2 (plan p2c §3.2): exact in radiance (a shadow bit whose light's direct term is 0 "
                     "adds 0 to both pooled sums, DESIGN §7.6) and N-independent; off in every profile",
           .help = "Shadow bits whose light's direct term is 0 count as lit in the signature and class word; "
                   "irradiance unchanged"}),
      // P2-F F1 (plan p2f §4.1): shadows over texture; no profile adopts it before P2-F's adoption (F13).
      member<&S::texture_shadow>(
          "accuracy.texture_shadow",
          {.status = Status::kChoice, .cls = C::kNAccuracy, .stage = render,
           .source = "PHASE2 P2-F F1 (plan p2f §4.1; G1 S2, P2-F F0): weighted removes the pooled ratio's bias over "
                     "texture (3-68 eps_L in S2) down to the staircase and is exact in lit and umbra clusters; pooled "
                     "in every profile until P2-F's adoption",
           .help = "Shadow ratio of textured clusters: pooled (sum f V / sum f) or weighted (each sub-sample's own "
                   "albedo weights its f)"}),
      // P2-F F5 (plan p2f §4.4): shading patches; no profile adopts them before P2-F's adoption (F13).
      member<&S::shading_patches>(
          "accuracy.shading_patches",
          {.status = Status::kChoice, .cls = C::kNAccuracy, .stage = render,
           .source = "PHASE2 P2-F F5 (plan p2f §4.4; DESIGN §5.2, O7): crease cuts patches at material changes, "
                     "between disconnected shells, at hard edges (split vertex normals) with a dihedral > 30 deg or "
                     "on a specular material, and at dihedrals > 30 deg on meshes without normals, so a crease is a "
                     "localized jump; material in every profile until P2-F's adoption",
           .help = "Cluster key: material, or crease (cut at materials, shells, hard edges > 30 deg or specular, "
                   "> 30 deg without normals)"}),
      // P2-F F9a (plan p2f §4.7): mover AO by sphere-set proxies; no profile adopts it before P2-F's adoption (F13).
      member<&S::mover_ao>(
          "accuracy.mover_ao",
          {.status = Status::kChoice, .cls = C::kNAccuracy, .stage = render,
           .source = "PHASE2 P2-F F9a (plan p2f §4.7; F108, DESIGN §7.7): proxy occludes each cluster's ambient by the "
                     "other movers' fitted spheres (analytic form factors, C2 contact floor and window, omission <= "
                     "0.002 per body, eps_L's family prior, not the run's eps_L; no rays or signature bits); traced "
                     "(P2-F F9b, plan §13.3's alternative) by one ambient ray per sub-sample against the other movers "
                     "within those windows (the AO bit in the signature, N >= F108's step rule, no held world); off in "
                     "every profile until P2-F's adoption",
           .help = "Ambient occlusion by moving bodies: off, proxy (fitted spheres) or traced (a ray per sub-sample)"}),
  };
  return rows;
}

// The RendererOptions rows, in member order (scene excluded).
inline const std::vector<config::Param>& renderer_options_rows() {
  using namespace config;
  using O = RendererOptions;
  using C = config::Class;
  const auto none = Stage::kNone;
  static const std::vector<Param> rows{
      custom<O, param_detail::set_device, param_detail::get_device>(
          "run.device", {.type = Type::kEnum, .value = std::string("auto"), .choices = {"auto", "gpu", "cpu"},
                         .status = Status::kChoice, .cls = C::kTRun, .stage = Stage::kRender, .field = "backend",
                         .source = "ENGINE_CPP §1.7, I4",
                         .help = "gpu: OptiX and the CUDA fp32 sensor; cpu: Embree and the CPU sensor; auto: gpu when "
                                 "usable"}),
      member<&O::num_threads>("execution.render_threads",
                              {.unit = "1", .hard = at_least(0), .status = Status::kChoice, .cls = C::kTRun,
                               .stage = none, .field = "num_threads", .source = "execution choice",
                               .help = "Embree threads (0: hardware concurrency - 1)"}),
      member<&O::max_rays_per_launch>("execution.max_rays_per_launch",
                                      {.unit = "1", .hard = at_least(65536), .status = Status::kMeasured,
                                       .cls = C::kEHeuristic, .stage = none,
                                       .source = "RENDER.md: 2^24 is 1-2.5 % faster than 2^22; launches < 50 ms",
                                       .help = "Rays per renderer launch, shadow rays included"}),
      custom<O, param_detail::set_pose_cache, param_detail::get_pose_cache>(
          "execution.pose_cache_mb", {.type = Type::kInt, .unit = "MB", .value = int64_t{256},
                                      .hard = closed(0, 1099511627776.0), .status = Status::kPlaceholder,
                                      .cls = C::kEHeuristic, .stage = none, .field = "pose_cache_bytes",
                                      .source = "engineering (no measurement)",
                                      .help = "Pose-table memory; the table is flushed beyond it"}),
      member<&O::census>("execution.census",
                         {.status = Status::kChoice, .cls = C::kTRun, .stage = none,
                          .source = "ENGINE_CPP §0.1 \"P2-B B0 notes\" (a counter; outputs unchanged)",
                          .help = "Count the sub-samples that P2-B's probe reuse rule would skip (perf.jsonl "
                                  "census_* counters)"}),
      // P2-D D2 (plan p2d §2 G-D9): the price of P&V's render outputs; output-invariant, like execution.census.
      member<&O::pv_outputs>("execution.pv_outputs",
                             {.status = Status::kChoice, .cls = C::kTRun, .stage = none,
                              .source = "PHASE2 P2-D D2 (a cost measurement; outputs unchanged)",
                              .help = "Render calls at N >= 256 also write P&V's codes and tables to scratch (perf.jsonl "
                                      "pv_out_* counters)"}),
  };
  return rows;
}

// The renderer's options of resolved values (config::apply, docs/dev/P0_PARAMS.md §1.4): every RendererOptions and
// SceneSettings row, the run keys from v.run and the camera keys with a run-wide target (one SceneSettings per
// renderer) from the first camera, which every camera must agree with. std::invalid_argument names a missing,
// invalid or disagreeing value.
inline RendererOptions apply_renderer(const config::Values& v) {
  using config::Param;
  if (v.cameras.empty()) throw std::invalid_argument("values: no camera");
  const auto& [first_name, first] = v.cameras.front();
  const auto value = [&](const Param& p) -> const config::Value& {
    return p.scope == config::Scope::kCamera ? config::value_of(p, first, "cameras." + first_name + ".")
                                             : config::value_of(p, v.run, "");
  };
  RendererOptions o;
  for (const Param& p : renderer_options_rows()) config::set(p, o, value(p));
  for (const Param& p : scene_settings_rows()) {
    config::set(p, o.scene, value(p));
    if (p.scope != config::Scope::kCamera) continue;
    for (const auto& [name, m] : v.cameras) {
      if (config::value_of(p, m, "cameras." + name + ".") != value(p)) {
        throw std::invalid_argument("values: cameras disagree on " + p.key + " (one per renderer)");
      }
    }
  }
  return o;
}

}  // namespace balla::render
