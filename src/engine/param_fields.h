// The EngineConfig and SensorSpec rows (docs/dev/P0_PARAMS.md §1.3; sources: PARAMETERS.md §3.3), and apply(): the
// engine's configuration from resolved values, through the five targets' rows. Header-only: it includes engine.h for
// the member pointers and links nothing beyond balla_sensor_params (the SensorParams rows).
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "core/camera/relative_illumination.h"
#include "core/config/param.h"
#include "core/time/time.h"
#include "engine/engine.h"
#include "render/param_fields.h"
#include "sensor/param_fields.h"

namespace balla::config {

template <>
struct TargetOf<engine::EngineConfig> {
  static constexpr Target value = Target::kEngineConfig;
};
template <>
struct TargetOf<engine::SensorSpec> {
  static constexpr Target value = Target::kSensorSpec;
};
template <>
struct EnumNames<sensor::Precision> {
  static constexpr std::array<std::string_view, 2> names{"fp64", "fp32"};
};
template <>
struct EnumNames<engine::Photometry> {
  static constexpr std::array<std::string_view, 3> names{"relative", "physical", "reference"};
};
template <>
struct EnumNames<engine::VignettingLaw> {
  static constexpr std::array<std::string_view, 2> names{"cos4", "jacobian"};
};
template <>
struct EnumNames<engine::LevelPolicy> {
  static constexpr std::array<std::string_view, 3> names{"tile", "global", "global_certified"};
};
template <>
struct EnumNames<engine::BandGrowth> {
  static constexpr std::array<std::string_view, 2> names{"tile", "pixel"};
};
template <>
struct EnumNames<engine::ShadowObliquity> {
  static constexpr std::array<std::string_view, 2> names{"off", "projected"};
};
template <>
struct EnumNames<engine::Localization> {
  static constexpr std::array<std::string_view, 2> names{"bisect", "shadow"};
};

}  // namespace balla::config

namespace balla::engine {

// Members without a row of their own, named by pointer (a renamed or removed member breaks the build; the schema
// test counts them): the renderer options, and the bundle's light groups and holds; a camera's sensor parameters.
inline constexpr auto kEngineConfigExcluded =
    std::tuple{&EngineConfig::render, &EngineConfig::light_groups, &EngineConfig::holds};
inline constexpr auto kSensorSpecExcluded = std::tuple{&SensorSpec::params};

namespace param_detail {

// run.fixed_rate_hz: the period llround(1e9 / v) ns, 0 for null (validate_run refuses a non-integral period).
inline void set_fixed_rate(const config::Param& p, EngineConfig& c, const config::Value& v) {
  config::check_value(p, v, config::display_name(p));
  const bool planned = std::holds_alternative<std::monostate>(v);
  c.fixed_period_ns = planned ? 0 : std::llround(1e9 / config::detail::to_double(p, v));
}
inline config::Value get_fixed_rate(const EngineConfig& c) {
  return c.fixed_period_ns > 0 ? config::Value{1e9 / static_cast<double>(c.fixed_period_ns)} : config::Value{};
}
// run.crop_px: [x0, y0, x1, y1] (pixels x0 <= x < x1, y0 <= y < y1) or null.
inline void set_crop(const config::Param& p, EngineConfig& c, const config::Value& v) {
  config::check_value(p, v, config::display_name(p));
  if (std::holds_alternative<std::monostate>(v)) {
    c.crop.reset();
    return;
  }
  const auto a = config::detail::from_value<std::array<int32_t, 4>>(p, v);
  c.crop = Crop{a[0], a[1], a[2], a[3]};
}
inline config::Value get_crop(const EngineConfig& c) {
  if (!c.crop) return {};
  return std::vector<int64_t>{c.crop->x0, c.crop->y0, c.crop->x1, c.crop->y1};
}
// accuracy.n_band_max (P2-B B4a): EngineConfig::n_band, 0 for null (one N).
inline void set_n_band(const config::Param& p, EngineConfig& c, const config::Value& v) {
  config::check_value(p, v, config::display_name(p));
  c.n_band = std::holds_alternative<std::monostate>(v) ? 0u : static_cast<uint32_t>(config::detail::to_int(p, v));
}
inline config::Value get_n_band(const EngineConfig& c) {
  return c.n_band > 0 ? config::Value{static_cast<int64_t>(c.n_band)} : config::Value{};
}
// optics.relative_illumination (P2-F F2b): [] or [c2, c4] with G > 0 over the image
// (core/camera/relative_illumination.h).
inline void set_relative_illumination(const config::Param& p, SensorSpec& s, const config::Value& v) {
  config::check_value(p, v, config::display_name(p));
  std::vector<double> c;
  if (const auto* l = std::get_if<std::vector<double>>(&v)) c = *l;
  if (const auto* l = std::get_if<std::vector<int64_t>>(&v)) c.assign(l->begin(), l->end());
  if (const char* why = camera::relative_illumination_error(c)) {
    throw std::invalid_argument(config::display_name(p) + " = " + config::format_value(v) + ": " + why);
  }
  s.relative_illumination = std::move(c);
}
inline config::Value get_relative_illumination(const SensorSpec& s) { return s.relative_illumination; }
// sensor.nir_fraction (P2-F F11a): [] or one fraction in [0, 1] per light group (at most render::kMaxLightGroups; the
// length against the bundle's groups is checked where both are known, SceneData).
inline void set_nir_fraction(const config::Param& p, SensorSpec& s, const config::Value& v) {
  config::check_value(p, v, config::display_name(p));
  std::vector<double> f;
  if (const auto* l = std::get_if<std::vector<double>>(&v)) f = *l;
  if (const auto* l = std::get_if<std::vector<int64_t>>(&v)) f.assign(l->begin(), l->end());
  if (f.size() > static_cast<size_t>(render::kMaxLightGroups)) {
    throw std::invalid_argument(config::display_name(p) + " = " + config::format_value(v) + ": at most " +
                                std::to_string(render::kMaxLightGroups) + " light groups");
  }
  s.nir_fraction = std::move(f);
}
inline config::Value get_nir_fraction(const SensorSpec& s) { return s.nir_fraction; }

}  // namespace param_detail

// The EngineConfig rows, in member order (render, light_groups and holds excluded).
inline const std::vector<config::Param>& engine_config_rows() {
  using namespace config;
  using E = EngineConfig;
  using C = config::Class;
  const std::vector<When> planned{{"run.fixed_rate_hz", Value{}}};
  const auto schedule = Stage::kSchedule;
  static const std::vector<Param> rows{
      member<&E::refine_signature>("accuracy.refine_signature",
                                   {.status = Status::kDerived, .cls = C::kNAccuracy, .stage = schedule,
                                    .when = planned, .source = "PARAMETERS §4.3",
                                    .help = "Split intervals whose end signatures differ (occlusion and shadow "
                                            "flips)"}),
      member<&E::refine_tol_log>("accuracy.refine_tol_log",
                                 {.unit = "log-e", .hard = above(0), .soft = left_open(0, 0.02), .scale = Scale::kLog,
                                  .status = Status::kDerived, .cls = C::kNAccuracy, .stage = schedule,
                                  .when = planned, .source = "PARAMETERS §4.3 (kappa sigma_v,ref)",
                                  .help = "Midpoint-test tolerance eta_L of the sample schedule; null turns the test "
                                          "off"}),
      member<&E::max_refine_levels>("accuracy.max_refine_levels",
                                    {.unit = "1", .hard = closed(0, 13), .status = Status::kPlaceholder,
                                     .cls = C::kEHeuristic, .stage = schedule, .when = planned,
                                     .source = "no derivation (PARAMETERS §3.3.3); a safety cap (13) under the tau "
                                               "stop (balanced, ENGINE_CPP I7)",
                                     .help = "Refinement rounds below the lattice level (every flag stops there)"}),
      // The economy schedule ES (ENGINE_CPP I7; samples study §3.3-§3.6): the defaults reproduce Phase 1's schedule;
      // the balanced profile sets the values the sampling tier accepted at I7.
      member<&E::mid_levels>("accuracy.mid_levels",
                             {.unit = "1", .hard = closed(0, 13), .soft = closed(3, 13), .status = Status::kChoice,
                              .cls = C::kEHeuristic, .stage = schedule, .when = planned,
                              .source = "samples study §3.3 (3); 4 in balanced (3 failed the bright textured card, "
                                        "ENGINE_CPP I7 review); fewer than 3 are untested",
                              .help = "Levels below the lattice that midpoint flags refine (null: every level up to "
                                      "accuracy.max_refine_levels)"}),
      member<&E::tau_jump_ns>("accuracy.tau_jump_ns",
                              {.unit = "ns", .hard = closed(1, static_cast<double>(time::kWindowNs)),
                               .soft = closed(1024, 65536), .scale = Scale::kLog, .status = Status::kChoice,
                               .cls = C::kEHeuristic, .stage = schedule, .when = planned,
                               .source = "samples study §3.3 (timing error <= tau / 2 for steps >= delta_ref); "
                                         "2^12 with delta_ref 0 in balanced and the bench's ref2 (2^14 failed the "
                                         "bright crop, ENGINE_CPP I7)",
                               .help = "Jump localization: signature flags split intervals of >= 2 tau max(1, "
                                       "delta_ref / eta_J) (null: by level alone)"}),
      member<&E::delta_ref>("accuracy.delta_ref",
                            {.unit = "log-e", .hard = at_least(0), .soft = closed(0, 0.25), .status = Status::kChoice,
                             .cls = C::kEHeuristic, .stage = schedule, .when = planned,
                             .source = "samples study §3.3: 0.04 (about theta_design / 4; a step of eta_J carries a "
                                       "crossing with probability about eta_J / theta); 0 in balanced (0.04 failed "
                                       "the bright crop, ENGINE_CPP I7)",
                             .help = "Step of full jump localization; smaller jumps eta_J stop at delta_ref / eta_J "
                                     "times tau (0: tau for all)"}),
      member<&E::half_ratio>("accuracy.half_ratio",
                             {.unit = "1", .hard = closed(0, 1), .soft = closed(0, 0.5), .status = Status::kChoice,
                              .cls = C::kEHeuristic, .stage = schedule, .when = planned,
                              .source = "samples study §3.5: 0.25 (a step costs one sample per level); 0 in balanced "
                                        "(0.25 failed the slow crop, ENGINE_CPP I7)",
                              .help = "Step-aware halving: a midpoint-flagged half refines if its eta_J >= this x the "
                                      "halves' larger (0: both)"}),
      member<&E::eps_jump>("accuracy.eps_jump",
                           {.unit = "log-e", .hard = at_least(0), .soft = closed(0, 0.02), .status = Status::kChoice,
                            .cls = C::kEHeuristic, .stage = schedule, .when = planned,
                            .source = "samples study §3.4: 0.002 leaves events identical at d_px 0.25 and gates "
                                      "zero-amplitude flips (balanced, ref2); fidelity since P2-B's adoption (R3: "
                                      "under the band 99.84-99.97 % of events matched, p99 |dt| <= 1 us)",
                            .help = "Residual gate: a signature flag splits only where the endpoint jump eta_J "
                                    "exceeds it (null: every signature flag)"}),
      member<&E::min_dt_ns>("accuracy.min_dt_ns",
                            {.unit = "ns", .hard = closed(1, static_cast<double>(time::kWindowNs)),
                             .status = Status::kDerived, .cls = C::kNAccuracy, .stage = schedule,
                             .source = "PARAMETERS §4.3; a power of two",
                             .help = "Finest lattice and refinement spacing (a power of two)"}),
      custom<E, param_detail::set_fixed_rate, param_detail::get_fixed_rate>(
          "run.fixed_rate_hz", {.type = Type::kFloat, .nullable = true, .unit = "Hz", .hard = above(0),
                                .status = Status::kChoice, .cls = C::kTRun, .stage = schedule,
                                .field = "fixed_period_ns", .source = "O2 reference mode (PARAMETERS §3.3.1)",
                                .help = "Dense fixed-rate sampling of every crop pixel (integral period in ns); "
                                        "null: the planner"}),
      member<&E::device_budget_mb>("execution.device_budget_mb",
                                   {.unit = "MB", .hard = at_least(0), .status = Status::kChoice,
                                    .cls = C::kEHeuristic, .stage = Stage::kNone, .source = "ENGINE_CPP §1.2",
                                    .help = "Memory for the samples, plan and output buffers (0: 70 % of the memory "
                                            "free after the build); output-invariant"}),
      member<&E::d_px>("accuracy.d_px", {.unit = "px", .hard = above(0), .soft = left_open(0, 1),
                                         .status = Status::kDerived, .cls = C::kNAccuracy, .stage = schedule,
                                         .when = planned, .source = "PARAMETERS §4.3: 2/sqrt(N)",
                                         .help = "A-priori image displacement cap per sample interval"}),
      member<&E::max_samples_per_window>("accuracy.max_samples_per_window",
                                         {.unit = "1", .hard = at_least(1), .status = Status::kDerived,
                                          .cls = C::kNAccuracy, .stage = schedule, .when = planned,
                                          .source = "Phase-1 host memory; D9 (null: no clamp). P2-D D1f: under "
                                                    "shadow_obliquity projected it first bounds the levels the "
                                                    "factors raise",
                                          .help = "Lattice samples per window; above it the finest levels are "
                                                  "clamped (null: no clamp)"}),
      custom<E, param_detail::set_crop, param_detail::get_crop>(
          "run.crop_px", {.type = Type::kIntList, .nullable = true, .length = 4, .unit = "px", .hard = at_least(0),
                          .status = Status::kChoice, .cls = C::kTRun, .stage = schedule, .field = "crop",
                          .source = "run choice",
                          .help = "Simulate only pixels x0 <= x < x1, y0 <= y < y1 ([x0, y0, x1, y1]); the sensor "
                                  "models all"}),
      member<&E::gt>("output.gt", {.status = Status::kChoice, .cls = C::kTRun, .stage = Stage::kOutput,
                                   .source = "run choice",
                                   .help = "Write the /gt/* datasets of the event file (contract C4)"}),
      member<&E::held_world>("accuracy.held_world",
                             {.status = Status::kChoice, .cls = C::kNAccuracy, .stage = Stage::kRender,
                              .source = "ENGINE_CPP I3b (speed; last-bit changes, no derivation)",
                              .help = "Trace bodies held over a window in the static world (last-bit changes)"}),
      // ENGINE_CPP I8 (samples study §3.7): the planner's tightening; off reproduces the cone and sphere bounds.
      member<&E::shadow_band>("accuracy.shadow_band",
                              {.status = Status::kChoice, .cls = C::kEHeuristic, .stage = schedule, .when = planned,
                               .source = "samples study §3.7; ENGINE_CPP I8 and its review (a heuristic: the cone "
                                         "rate kept at discontinuities, border tiles, spot-cone crossings, multi-mover "
                                         "shadows, and thin parts (>= 2 % of a body's surface) at foreshortened "
                                         "receivers; every tile in a droppable cone stays sampled)",
                               .help = "Moving-shadow band: the cone rate only near plan-pass shadow boundaries and "
                                       "retained regions (else the coarsest level)"}),
      // P2-D D1 (F021; DESIGN §6.7): the shadow-edge speed bound with the receivers' obliquity. Off in every profile
      // (P2-D plan §10 decision 2: its lattice cost, measured by OBL_SET, is the user's to weigh).
      member<&E::shadow_obliquity>(
          "accuracy.shadow_obliquity",
          {.status = Status::kChoice, .cls = C::kEHeuristic, .stage = schedule, .when = planned,
           .source = "P2-D D1 (F021; DESIGN §6.7): g = sigma_max(P_view (I - d n^T / (n . d)) on d_perp), h = "
                     "|P_view d| / cos psi from the plan pass's geometric normals, capped at 8 (cos psi >= 1/8; a "
                     "design choice); a bound for receivers planar at the facet scale, not on sub-pixel relief. "
                     "D1f: the levels it raises stay within 3 GiB of runs per window at twice the lattice (and "
                     "max_samples_per_window), never below the as-built ones (obliquity_tiles_capped)",
           .help = "Shadow-edge speed bound: off (as built, no obliquity factor) or projected (receiver obliquity, "
                   "capped)"}),
      // P2-D D3a (plan p2d §4): the localization of signature flips. shadow is a developer value: P&V runs beside
      // bisection and only counts, so it changes no output and is hashed as bisect (resolve.cpp digests). D4 adds
      // pv_shadow (P&V's brackets in the runs).
      member<&E::localization>(
          "accuracy.localization",
          {.status = Status::kChoice, .cls = C::kNAccuracy, .stage = schedule, .when = planned,
           .source = "P2-D D3a (DESIGN §6.2 step 2, plan p2d §3.1): predict-and-verify with single sub-sample probes "
                     "on the band's pixels (N >= 256), every doubtful interval routed to bisection (R0-R6)",
           .help = "Flip localization: bisect (as built) or shadow (P&V beside it, counters only; outputs as "
                   "bisect's)"}),
      // PHASE2 D-P2-4 (P2-B B4a): the band N. The resolver checks it against N_base and the reset (resolve.cpp).
      custom<E, param_detail::set_n_band, param_detail::get_n_band>(
          "accuracy.n_band_max",
          {.type = Type::kInt, .nullable = true, .unit = "1", .hard = closed(64, 1024), .status = Status::kDerived,
           .cls = C::kNAccuracy, .stage = Stage::kRender, .field = "n_band",
           .source = "PHASE2 D-P2-4, F068: a power of two (pbrt-v4's Sobol, Mitsuba 3's ldsampler) meeting kappa_q "
                     "0.35 at sigma_v,min by N^(-3/4); 256, G1's interim cap until EFC (c_loc, B5, is not built)",
           .help = "Band N: sub-samples near predicted silhouettes and shadow edges (a power of two > N); null: "
                   "one N"}),
      // P2-B B6c (the band memo's option #1; ENGINE_CPP §0.1 "P2-B B6c notes"): the band's map and its reach scales.
      // The tile map keeps T2 (the band's outputs those of uniform N_band) up to its two documented gaps; the pixel map
      // below (1, 1) does not (B6c's R2), so fidelity's adoption of pixel at (0.25, 0) (2026-10-12) is held by the
      // statistical gate against uniform N_band instead (BENCH.md "P2-B adoption"). The resolver refuses pixel where
      // pooled escalation's premise fails (support_px < 0.75), where fidelity derives tile (resolve.cpp).
      member<&E::band_growth>("accuracy.band_growth",
                              {.status = Status::kChoice, .cls = C::kEHeuristic, .stage = Stage::kRender,
                               .source = "band memo §5 (P2-B B6c; adopted by fidelity 2026-10-12, held by the "
                                         "statistical gate: matched >= 99.9 %, p99 |dt| <= 1 us against uniform "
                                         "N_band)",
                               .help = "Band N's map: tile (8 px tiles within a seed's reach) or pixel (each seed's "
                                       "L-inf ball); inert when n_band_max is null"}),
      member<&E::band_reach>("accuracy.band_reach",
                             {.unit = "1", .hard = at_least(0), .soft = closed(0, 1), .status = Status::kChoice,
                              .cls = C::kEHeuristic, .stage = Stage::kRender,
                              .source = "band memo §3 (P2-B B6c): (1, 1) is the rigorous reach; (0.25, 0) a "
                                        "measured performance constant (R2: loop -17 to -27 % against the tile "
                                        "map), fidelity's",
                              .help = "[sigma_cam, sigma_mov]: scales of a seed's camera and movers' motion in the "
                                      "pixel map's reach (band_growth pixel)"}),
      // P2-C C2 (plan p2c §6, §11.4): counters of the jump rules' decisions; output-invariant, like execution.census.
      member<&E::jump_counters>("execution.jump_counters",
                                {.status = Status::kChoice, .cls = C::kTRun, .stage = Stage::kNone,
                                 .source = "PHASE2 P2-C C2 (counters; outputs unchanged)",
                                 .help = "Count what P2-C's jump rules would change, and the jump brackets "
                                         "(perf.jsonl jump_* and sibling_* counters)"}),
      // P2-H H9 (F004, F029; DESIGN §13.5): the level policy of the thesis's equal-error Pareto arms. tile in every
      // profile; global costs 1.19-2.72 x fidelity's loop on the bench clips (the planner study, PS), so it is no
      // speed lever.
      member<&E::level_policy>(
          "accuracy.level_policy",
          {.status = Status::kChoice, .cls = C::kNAccuracy, .stage = schedule, .when = planned,
           .source = "P2-H H9 (F004; DESIGN §13.5): global, every active tile at the window's largest per-tile level "
                     "(the same speed bound and dyadic rounding), static tiles included; global_certified, the dynamic "
                     "tiles at that level, the static ones certified as under tile",
           .help = "Lattice levels: tile (each its own), global (all active tiles at the top tile level), "
                   "global_certified (dynamic only)"}),
  };
  return rows;
}

// The SensorSpec rows, in member order (params excluded): run keys applied to every camera, then the camera's
// photometry (D1, D13 item 2; the resolver derives photometry.illuminance_scale from the mode, resolve.cpp).
inline const std::vector<config::Param>& sensor_spec_rows() {
  using namespace config;
  using S = SensorSpec;
  using C = config::Class;
  const auto mode = [](std::vector<std::string> m) { return std::vector<When>{{"photometry.mode", std::move(m)}}; };
  static const std::vector<Param> rows{
      member<&S::seed>("run.sensor_seed", {.unit = "1", .hard = at_least(0), .status = Status::kChoice,
                                           .cls = C::kTRun, .stage = Stage::kSensor, .field = "seed",
                                           .source = "arbitrary",
                                           .help = "Seed of the sensor maps, noise and readout phase"}),
      member<&S::num_threads>("execution.sensor_threads",
                              {.unit = "1", .hard = at_least(0), .status = Status::kChoice, .cls = C::kTRun,
                               .stage = Stage::kNone, .field = "num_threads", .source = "execution choice",
                               .help = "CPU sensor threads (0: hardware concurrency)"}),
      member<&S::precision>("accuracy.sensor_precision",
                            {.status = Status::kDerived, .cls = C::kNAccuracy, .stage = Stage::kSensor,
                             .when = {When{"run.device", std::string("cpu")}}, .field = "precision",
                             .source = "DESIGN §6.4 row 7 (fp32 rounding <= 1.2e-4 log-e)",
                             .help = "CPU sensor precision (the GPU sensor runs fp32)"}),
      // Photometry: stage schedule, as eta_L's dark term reads the scale and the gain map they set (window.h).
      member<&S::photometry>(
          "photometry.mode",
          {.status = Status::kChoice, .cls = C::kTScene, .stage = Stage::kSchedule, .field = "photometry",
           .source = "D1 (PARAMETERS §5.9); DESIGN §7.1",
           .help = "relative: illuminance_scale x renderer value; physical: lights in cd, lx, cd/m2 via lens and pixel; "
                   "reference: solved"}),
      member<&S::vignetting>(
          "optics.vignetting",
          {.status = Status::kChoice, .cls = C::kTScene, .stage = Stage::kSchedule,
           .source = "DESIGN §8.1: cos^4 of the chief ray, 0.44 at the Prophesee corner (S2 OPT-01); a thin-lens law; "
                     "null per mode as pbrt's cameras (D1 option 2, SENSOR.md \"Photometry\")",
           .help = "Natural vignetting: pixel gain by optics.vignetting_law (cos4 or jacobian); null: off under "
                   "relative, else on"}),
      // P2-F F2 (PHASE2 F079): the law; a camera key, so cameras may differ. Relevant where the gain map is on:
      // vignetting true, or null under physical and reference.
      // P2-F F13-min: a derived default (p2f.md §4.2; config/resolve.cpp derive_vignetting_law): fidelity and balanced
      // derive jacobian per camera, phase1 and reference cos4 (Phase 1's law), and a camera under an EVIMO2 v0 or v1
      // sensor profile keeps cos4 under every profile.
      member<&S::vignetting_law>(
          "optics.vignetting_law",
          {.def = Default::kDerived,
           .formula = "jacobian under accuracy.profile fidelity or balanced where the gain map is on and the camera is "
                      "not under an EVIMO2 v0 or v1 sensor profile; else cos4",
           .status = Status::kDerived, .cls = C::kTScene, .stage = Stage::kSchedule,
           .when = {When{"optics.vignetting", true}},
           .also_when = {When{"photometry.mode", std::vector<std::string>{"physical", "reference"}},
                         When{"optics.vignetting", Value{}}},
           .source = "P2-F F2 (PHASE2 F079): E ~ L cos(theta) dOmega/dA for a small fixed pupil at the projection "
                     "centre, the camera model's solid angle per pixel area (core/camera/solid_angle.h); as pbrt-v4's "
                     "RealisticCamera (cos^4 x exit-pupil area on rays traced through the lens); Cycles none, pbrt-v4 "
                     "PerspectiveCamera and Mitsuba 3 perspective weight 1. cos4: Phase 1's law, listed by phase1 and "
                     "reference and kept by a camera under an EVIMO2 v0 or v1 sensor profile (frozen); P2-F F13-min: "
                     "fidelity and balanced derive jacobian per camera",
           .help = "cos4: cos^4 of the chief ray (the pinhole's); jacobian: cos(theta) dOmega/dA of the camera "
                   "model (radtan: cos^4/det J_d)"}),
      // P2-F F2b (PHASE2 F079, p2f.md §4.2): the lens's relative illumination on the law's gain, the user's (P2-H may
      // fit it); never derived. Relevant where the gain map is on, as the law.
      custom<S, param_detail::set_relative_illumination, param_detail::get_relative_illumination>(
          "optics.relative_illumination",
          {.type = Type::kFloatList, .length = -1, .unit = "1", .value = std::vector<double>{},
           .status = Status::kChoice, .cls = C::kTScene, .stage = Stage::kSchedule,
           .when = {When{"optics.vignetting", true}},
           .also_when = {When{"photometry.mode", std::vector<std::string>{"physical", "reference"}},
                         When{"optics.vignetting", Value{}}},
           .field = "relative_illumination",
           .source = "P2-F F2b (PHASE2 F079): the pupil's off-axis loss the law leaves out, an even polynomial in image "
                     "height as lensfun's pa and Adobe's lens-profile vignetting models; pbrt-v4 RealisticCamera's exit "
                     "pupil per film radius; Cycles none, Mitsuba 3 perspective/thinlens weight 1. []: none",
           .help = "Relative illumination [c2, c4]: gain x (1 + c2 r^2 + c4 r^4), r = image radius / farthest corner's; "
                   "[]: none"}),
      // P2-F F11a (PHASE2 F077, p2f.md §4.9): the near-infrared share of this camera's photo-electrons per light group,
      // the user's (lamp spectrum times the sensor's QE; DESIGN §7.2: 0.03 white LED, 0.25 D65, 0.59 incandescent, 1 an
      // 850 nm strobe); a render-stage camera key (the cameras may differ), never derived.
      custom<S, param_detail::set_nir_fraction, param_detail::get_nir_fraction>(
          "sensor.nir_fraction",
          {.type = Type::kFloatList, .length = -1, .unit = "1", .value = std::vector<double>{}, .hard = closed(0, 1),
           .status = Status::kChoice, .cls = C::kTScene, .stage = Stage::kRender, .field = "nir_fraction",
           .source = "P2-F F11a (PHASE2 F077; DESIGN §7.2): the fraction of a light group's photo-electrons from > 700 "
                     "nm (white LED 0.03, D65 0.25, incandescent 0.59, an 850 nm strobe 1); rho = (1 - f) rho_vis + f "
                     "rho_NIR per camera and group; none of Cycles (RGB), pbrt-v4 and Mitsuba 3 (360-830 nm) has a "
                     "band above 830 nm, so the mix is BallaSim's own (RENDER.md F11a conventions). [] = 0 everywhere",
           .help = "NIR share f of the photo-electrons per light group ([]: 0): the albedo is (1 - f) rho_vis + f rho_NIR"}),
      member<&S::reference_current_fA>(
          "photometry.reference_current_fA",
          {.unit = "fA", .hard = above(0), .scale = Scale::kLog, .status = Status::kChoice, .cls = C::kTScene,
           .stage = Stage::kSchedule, .when = mode({"reference"}),
           .source = "D1 (c): a stated photocurrent (e.g. a desk median), from which the scale is solved",
           .help = "Median signal photocurrent (lights at their mean, vignetting in) of the reference pixels in the "
                   "first frame"}),
      member<&S::reference_instance>(
          "photometry.reference_instance",
          {.status = Status::kChoice, .cls = C::kTScene, .stage = Stage::kSchedule, .when = mode({"reference"}),
           .source = "D1 (c)",
           .help = "Body name whose pixels (centre rays at the start frame) set the reference; empty: every pixel "
                   "that hits geometry"}),
  };
  return rows;
}

// Resolved values (config::Values; docs/dev/P0_PARAMS.md §1.4): the run keys, and per camera its camera keys.
using ValueMap = config::ValueMap;
using EngineValues = config::Values;  // the cameras that get a sensor

// The engine's configuration of resolved values (config::apply of P0_PARAMS §1.4): every row of EngineConfig from the
// run values, the renderer's options (render::apply_renderer: RendererOptions and SceneSettings, whose camera keys must
// agree across the cameras), and per camera a SensorSpec (its SensorParams by sensor::params_of, the run keys from the
// run's). Every value is checked against its row and every SensorParams validated; std::invalid_argument names a
// missing or invalid value. Light groups and holds come from the bundle (the caller).
struct Applied {
  EngineConfig config;
  std::vector<std::pair<std::string, SensorSpec>> sensors;
};

inline Applied apply(const EngineValues& v) {
  using config::Param;
  if (v.cameras.empty()) throw std::invalid_argument("engine values: no camera");
  Applied out;
  for (const Param& p : engine_config_rows()) config::set(p, out.config, config::value_of(p, v.run, ""));  // run keys
  out.config.render = render::apply_renderer(v);
  for (const auto& [name, m] : v.cameras) {
    SensorSpec spec;
    for (const Param& p : sensor_spec_rows()) {
      config::set(p, spec, p.scope == config::Scope::kCamera ? config::value_of(p, m, "cameras." + name + ".")
                                                             : config::value_of(p, v.run, ""));
    }
    spec.params = sensor::params_of(v.run, m, name);
    out.sensors.emplace_back(name, std::move(spec));
  }
  return out;
}

}  // namespace balla::engine
