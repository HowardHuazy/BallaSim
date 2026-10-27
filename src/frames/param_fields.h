// The frames' parameter rows (docs/design/phase3/frames.md §3.2; P3 FR-1b): 40 camera-scope rows of section `frames`
// (keys cameras.<camera>.frames.*, inherited from cameras.default like every camera key), and three run-scope rows,
// execution.frames_device, execution.frames_texture_mb and run.frames_select. All are rows without a target (read by
// config::resolve_frames and the frames steps, never by the engine) of stage `none`, so no frames value enters a stage
// digest and no event, sample run or reference moves (D-P3-11, frames.md §8.4). The camera rows matter only while the
// camera's frames.mode is not `off`; the run rows carry no condition, since a run row cannot read a camera's mode.
//
// Header-only like the render and engine rows: schema.cpp includes it, and balla_config links nothing new. Classes
// (frames.md §3.2): T-run for choices, N-accuracy for numerics, T-sensor for the APS sensor values. The APS rows repeat
// frames::ApsSettings' defaults (aps_frames.h); tests/cpp/test_frames_section.cpp pins the two together.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/config/param.h"

namespace balla::frames {

namespace detail {

inline std::vector<config::Param> make_rows() {
  using config::Class;
  using config::Opts;
  using config::Status;
  using config::Type;
  const auto text = [](const char* s) { return config::Value{std::string(s)}; };
  const std::vector<config::When> active{
      {"frames.mode", std::vector<std::string>{"reference", "matched", "full"}}};  // inert under `off`
  const std::string design = "docs/design/phase3/frames.md §3.2";
  std::vector<config::Param> rows;
  // A camera row of the section: stage none, matters while frames.mode is not off (frames.mode itself always).
  const auto camera = [&](const char* key, Opts o, const char* source, const char* help) {
    o.stage = config::Stage::kNone;
    if (std::string(key) != "frames.mode") o.when = active;
    o.source = source;
    o.help = help;
    rows.push_back(config::row(key, o));
  };
  const auto choice = Status::kChoice;
  const auto t_run = Class::kTRun, n_acc = Class::kNAccuracy, t_sensor = Class::kTSensor;
  const char* const d = design.c_str();

  // Schedule and output.
  camera("frames.mode",
         {.type = Type::kEnum, .value = text("off"), .choices = {"off", "reference", "matched", "full"},
          .status = choice, .cls = t_run},
         d, "Frame output of this camera: BallaSim's reference, Blender matched to it, or Blender as authored");
  camera("frames.timeline",
         {.type = Type::kEnum, .nullable = true, .choices = {"uniform", "blender"}, .status = choice, .cls = t_run}, d,
         "blender: one frame per Blender frame; uniform: t0 + k period; null: blender for Blender, else uniform");
  camera("frames.stride",
         {.type = Type::kInt, .unit = "1", .value = int64_t{1}, .hard = config::at_least(1), .status = choice,
          .cls = t_run},
         d, "blender timeline: every n-th Blender frame");
  camera("frames.t0_ns",
         {.type = Type::kInt, .nullable = true, .unit = "ns", .status = choice, .cls = t_run}, d,
         "uniform timeline: start of exposure of the first-read row of frame 0 (true clock); null: the run's start");
  camera("frames.period_ns",
         {.type = Type::kInt, .nullable = true, .unit = "ns", .hard = config::above(0), .status = choice,
          .cls = t_run},
         d,
         "uniform timeline: frame period; null: round_half_even(1e9 / frames.fps)");
  camera("frames.fps",
         {.type = Type::kFloat, .unit = "Hz", .value = 30.0, .hard = config::above(0), .soft = config::closed(1, 1000),
          .status = choice, .cls = t_run},
         d, "uniform timeline: frame rate, used while frames.period_ns is null");
  camera("frames.count",
         {.type = Type::kInt, .nullable = true, .unit = "1", .hard = config::at_least(1), .status = choice,
          .cls = t_run},
         d,
         "Number of frames; null: every frame whose exposure lies inside the bundle span");
  camera("frames.exposure_ns",
         {.type = Type::kInt, .nullable = true, .unit = "ns", .hard = config::at_least(0), .status = choice,
          .cls = t_run},
         d, "Exposure per row; null: the authored Blender shutter, else 10 ms; 0: instantaneous");
  camera("frames.shutter",
         {.type = Type::kEnum, .value = text("global"), .choices = {"global", "rolling"}, .status = choice,
          .cls = t_run},
         d,
         "Global shutter, or a rolling one with frames.row_dt_ns between consecutive read rows");
  camera("frames.row_dt_ns",
         {.type = Type::kInt, .unit = "ns", .value = int64_t{0}, .hard = config::at_least(0), .status = choice,
          .cls = t_run},
         d, "Rolling shutter: start-of-exposure delay between consecutive read rows");
  camera("frames.readout",
         {.type = Type::kEnum, .value = text("top_down"), .choices = {"top_down", "bottom_up"}, .status = choice,
          .cls = t_run},
         d, "Row order of a rolling shutter");
  camera("frames.roi_px",
         {.type = Type::kIntList, .nullable = true, .length = 4, .unit = "px", .hard = config::at_least(0),
          .status = choice, .cls = t_run},
         d, "Rendered region [x0, y0, x1, y1] (x0 <= x < x1); null: run.crop_px when set, else the full frame");
  camera("frames.outputs",
         {.type = Type::kStringList, .nullable = true, .length = -1, .status = choice, .cls = t_run}, d,
         "Datasets written, from rgb, mono, groups, reference and aps; null: mono (reference), else rgb and mono");
  camera("frames.fitted_poses",
         {.type = Type::kEnum, .value = text("physics"), .choices = {"none", "physics", "all"}, .status = choice,
          .cls = t_run},
         d, "Bodies whose engine-fitted trajectory is keyed into the saved Blender copy");
  camera("frames.keep_raw",
         {.type = Type::kEnum, .value = text("none"), .choices = {"none", "npy", "npy_exr"}, .status = choice,
          .cls = t_run},
         d, "Raw Blender passes kept after a successful ingest");

  // Rendering numerics.
  camera("frames.route",
         {.type = Type::kEnum, .value = text("auto"), .choices = {"auto", "film", "supersample", "osl"},
          .status = choice, .cls = t_run},
         d, "Blender route: film (F-box, F-gauss), supersample (S) or osl (O); auto takes the first that applies");
  camera("frames.supersample",
         {.type = Type::kInt, .nullable = true, .unit = "1", .hard = config::closed(2, 8), .status = choice,
          .cls = n_acc},
         d, "Route S factor (2, 4 or 8); null: 8 for matched with an identity map, otherwise 4");
  camera("frames.spp",
         {.type = Type::kInt, .nullable = true, .unit = "1", .hard = config::at_least(1), .status = choice,
          .cls = n_acc},
         d, "Cycles samples per pixel; null: matched 512 (F, O), 32 (S8) or 128 (S4); full 128");
  camera("frames.seed",
         {.type = Type::kInt, .unit = "1", .value = int64_t{0}, .hard = config::at_least(0), .status = choice,
          .cls = t_run},
         d, "Cycles seed; the full-mode halves use seed and seed + 1");
  camera("frames.matched.render_groups",
         {.type = Type::kEnum, .value = text("separate"), .choices = {"separate", "together"}, .status = choice,
          .cls = t_run},
         d, "One Blender render per light group, or one with group passes (G times cheaper, noisier flat regions)");
  camera("frames.matched.texture_mode",
         {.type = Type::kEnum, .value = text("float_linear"), .choices = {"float_linear", "closest"}, .status = choice,
          .cls = t_run},
         d, "Engine-decoded float textures with Linear interpolation, or Closest on bytes (coarser)");
  camera("frames.motion_tol_px",
         {.type = Type::kFloat, .unit = "px", .value = 0.05, .hard = config::above(0), .status = choice,
          .cls = n_acc},
         d, "Chord plus time-resolution tolerance of Cycles' motion steps");
  camera("frames.factor_tol",
         {.type = Type::kFloat, .unit = "1", .value = 0.005, .hard = config::above(0), .status = choice,
          .cls = n_acc},
         d, "Bound on the waveform-motion factorization error and on the moving-light freeze error before splitting");
  camera("frames.ref.h_px",
         {.type = Type::kFloat, .unit = "px", .value = 0.25, .hard = config::above(0), .status = choice,
          .cls = n_acc},
         d, "Image travel per initial reference time sample at 4 px of blur and more");
  camera("frames.ref.k_max",
         {.type = Type::kInt, .unit = "1", .value = int64_t{256}, .hard = config::closed(2, 4096), .status = choice,
          .cls = n_acc},
         d, "Cap on the reference's time samples per pixel");
  camera("frames.ref.probe_tol",
         {.type = Type::kFloat, .unit = "1", .value = 0.005, .hard = config::above(0), .status = choice,
          .cls = n_acc},
         d, "p99.9 over the probe mask of the estimated error of the reference's time sampling (ln units)");

  // Full mode.
  camera("frames.full.bounces",
         {.type = Type::kInt, .nullable = true, .unit = "1", .hard = config::at_least(0), .status = choice,
          .cls = t_run},
         d, "Cycles light bounces of the full render; null: the scene's own");
  camera("frames.full.correction",
         {.type = Type::kEnum, .value = text("off"), .choices = {"off", "lowpass"}, .status = choice, .cls = t_run}, d,
         "Apply the smooth correction map to the full render (always computed and stored)");
  camera("frames.full.calib_frames",
         {.type = Type::kIntList, .nullable = true, .length = -1, .unit = "1", .status = choice, .cls = t_run}, d,
         "Calibration frame indices; null: the first, the middle and the last");
  camera("frames.full.calib_spp",
         {.type = Type::kInt, .unit = "1", .value = int64_t{64}, .hard = config::at_least(16), .status = choice,
          .cls = n_acc},
         d, "Cycles samples per pixel of the direct-only calibration passes");
  camera("frames.full.contrast_ref",
         {.type = Type::kFloat, .nullable = true, .unit = "1", .hard = config::above(0), .status = choice,
          .cls = t_run},
         d, "Contrast C of the disagreement verdicts; null: min(theta_on, theta_off) of the camera's sensor");

  // The APS sensor: user tunables with a prior or placeholder value (frames.md §3.2, §4.10).
  const auto prior = Status::kPrior, placeholder = Status::kPlaceholder;
  camera("frames.aps.bits",
         {.type = Type::kInt, .unit = "1", .value = int64_t{10}, .hard = config::closed(8, 16), .status = prior,
          .cls = t_sensor},
         "ADC bits of DAVIS-class APS sensors (arXiv 2112.01933)", "APS ADC bits");
  camera("frames.aps.full_well_e",
         {.type = Type::kFloat, .unit = "e-", .value = 30900.0, .hard = config::above(0), .status = prior,
          .cls = t_sensor},
         "DAVIS346 datasheet: 55 dB dynamic range over 55 e- read noise (frames.md E6)", "APS full-well capacity");
  camera("frames.aps.read_noise_e",
         {.type = Type::kFloat, .unit = "e-", .value = 55.0, .hard = config::at_least(0), .status = prior,
          .cls = t_sensor},
         "DAVIS346 documentation", "APS read noise (rms)");
  camera("frames.aps.dark_current_e_s",
         {.type = Type::kFloat, .unit = "e-/s", .value = 18000.0, .hard = config::at_least(0), .status = placeholder,
          .cls = t_sensor},
         "DAVIS346 documentation (the 2025 sheet says 1800; frames.md Q2)", "APS dark current");
  camera("frames.aps.prnu",
         {.type = Type::kFloat, .unit = "1", .value = 0.042, .hard = config::at_least(0), .status = placeholder,
          .cls = t_sensor},
         "DAVIS346 documentation: fixed-pattern noise 4.2 % (the definition is unclear)",
         "APS photo-response non-uniformity (relative sigma)");
  camera("frames.aps.dsnu_e",
         {.type = Type::kFloat, .unit = "e-", .value = 20.0, .hard = config::at_least(0), .status = placeholder,
          .cls = t_sensor},
         "Assumption (frames.md §3.2)", "APS dark-signal non-uniformity (sigma)");
  camera("frames.aps.gain_dn_per_e",
         {.type = Type::kFloat, .nullable = true, .unit = "DN/e-", .hard = config::above(0), .status = prior,
          .cls = t_sensor},
         d, "APS conversion gain; null: (2^bits - 1) / full_well_e");
  camera("frames.aps.offset_dn",
         {.type = Type::kFloat, .unit = "DN", .value = 16.0, .hard = config::at_least(0), .status = prior,
          .cls = t_sensor},
         "frames.md E6", "APS black-level offset");
  camera("frames.aps.seed",
         {.type = Type::kInt, .nullable = true, .unit = "1", .hard = config::at_least(0), .status = choice,
          .cls = t_sensor},
         d, "Seed of the APS draws; null: run.sensor_seed");

  // Run scope: no relevance condition (a run row cannot read a camera's frames.mode).
  const auto run_row = [&](const char* key, Opts o, const char* help) {
    o.stage = config::Stage::kNone;
    o.status = choice;
    o.source = design;
    o.help = help;
    rows.push_back(config::row(key, o));
  };
  run_row("execution.frames_device",
          {.type = Type::kEnum, .value = text("auto"), .choices = {"auto", "optix", "cuda", "cpu"}, .cls = t_run},
          "Device of the Blender renders: optix, cuda or cpu; auto tries them in that order (OSL needs OptiX or CPU)");
  run_row("execution.frames_texture_mb",
          {.type = Type::kInt, .unit = "MB", .value = int64_t{4096}, .hard = config::at_least(0), .cls = t_run},
          "Budget of the float textures matched mode loads into Blender; above it every byte texture uses Closest");
  run_row("run.frames_select",
          {.type = Type::kStringList, .length = -1, .value = std::vector<std::string>{}, .cls = t_run},
          "Bundle cameras that get frames but no events (their frames section is resolved alone)");
  return rows;
}

}  // namespace detail

// The frames' schema rows: the 40 camera rows (frames.md §3.2's tables in order), then the three run rows.
inline const std::vector<config::Param>& param_rows() {
  static const std::vector<config::Param> rows = detail::make_rows();
  return rows;
}

}  // namespace balla::frames
