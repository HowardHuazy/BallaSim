// The schema: the targets' rows and the rows without a target (docs/dev/P0_PARAMS.md §1.3, §3.4; sources:
// PARAMETERS.md §3.3-§3.5). Rows without a target are consumed by the resolver (P0b) or by Python (the EVIMO2
// adapter and the trajectory fit, the run span and selection, paths, viz).
#include "config/schema.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/param_fields.h"
#include "frames/param_fields.h"
#include "imu/param_fields.h"
#include "render/param_fields.h"
#include "sensor/param_fields.h"

namespace balla::config {

namespace {

using C = Class;

std::vector<Param> untargeted_rows() {
  // The EVIMO2 lighting: every key while the lights are on; the mains and ambient keys in normal light, the strobe and
  // low-light ambient keys in _ll sequences (the bundle's light mode, scene.evimo2.low_light), as relight applies them.
  const std::vector<When> lit{{"scene.evimo2.lights", true}};
  const std::vector<When> normal{{"scene.evimo2.lights", true}, {"scene.evimo2.low_light", false}};
  const std::vector<When> low{{"scene.evimo2.lights", true}, {"scene.evimo2.low_light", true}};
  // The profiles that derive from the run's sensors (D8): fidelity and balanced (ENGINE_CPP I7).
  const std::vector<When> fid{{"accuracy.profile", std::vector<std::string>{"fidelity", "balanced"}}};
  const std::string lab = "required; evimo2/lab_lighting@v0 gives today's value (PARAMETERS §3.4.1)";
  const auto scene = Stage::kScene;
  const auto none = Stage::kNone;
  const auto req = Default::kRequired;
  const auto choice = Status::kChoice, derived = Status::kDerived, prior = Status::kPrior,
             placeholder = Status::kPlaceholder;
  // The physical photometric gain's inputs (D1, D13 item 2; DESIGN §7.1): the resolver derives the camera's
  // photometry.illuminance_scale from them and the bundle camera's pixel pitch (resolve.cpp); stage schedule, as the
  // scale.
  const std::vector<When> physical{{"photometry.mode", std::string("physical")}};
  const auto schedule = Stage::kSchedule;
  std::vector<Param> rows{
      row("sensor.qe",
          {.unit = "1", .def = Default::kFamily, .hard = left_open(0, 1), .soft = closed(0.2, 0.9), .cls = C::kTSensor,
           .stage = schedule, .when = physical, .source = "DESIGN §8.9 (R8 §2.3, S2 OPT-05)",
           .help = "Effective peak quantum efficiency (pixel-area mean, microlenses and fill factor included)"}),
      row("sensor.kappa",
          {.unit = "1/(s um2 lx)", .value = 7.8e3, .hard = above(0), .soft = closed(6e3, 3e4), .scale = Scale::kLog,
           .status = prior, .cls = C::kTSensor, .stage = schedule, .when = physical,
           .source = "DESIGN §7.1 kappa[S, spd] (R8 t3, S2 OPT-05) over Gen3.1's peak QE 0.37: white LED 2.9e3 -> 7.8e3; "
                     "fluorescent 6.8e3, D65 1.5e4, incandescent 2.8e4",
           .help = "Photo-electrons per s, um2 and lux at unit QE for the scene illuminant (white LED 7.8e3, D65 1.5e4, "
                   "tungsten 2.8e4)"}),
      row("optics.f_number",
          {.unit = "1", .def = Default::kDerived, .formula = "the bundle camera's f_number (C1; 2.0 if unknown)",
           .hard = above(0), .soft = closed(0.95, 22), .status = derived, .cls = C::kTScene, .stage = schedule,
           .when = physical, .also_when = {{"optics.diffraction", true}},  // the diffraction sigma reads it too
           .source = "C1 camera f_number; EVIMO2 prior 1.4-4 (DESIGN §8.1)",
           .help = "Working f-number N of the lens: image irradiance pi T L / (4 N^2); optics.diffraction reads it too"}),
      row("optics.transmittance",
          {.unit = "1", .value = 0.9, .hard = left_open(0, 1), .soft = closed(0.8, 0.98), .status = prior,
           .cls = C::kTScene, .stage = schedule, .when = physical,
           .source = "DESIGN §7.1 worked check (T 0.9; S2 OPT-03): multi-coated lens",
           .help = "Lens transmittance T (with any filter in the path)"}),
      // The lens PSF in micrometres (D13 item 3): when set, the single source of optics.blur_sigma_px (resolve.cpp).
      row("optics.blur_sigma_um",
          {.nullable = true, .unit = "um", .hard = at_least(0), .soft = closed(1, 15), .status = choice,
           .cls = C::kTScene, .stage = Stage::kRender,
           .source = "D13 item 3: a lens property; 0.15 px on 15 um (G6 §2.8) is 2.25 um",
           .help = "Optical Gaussian blur sigma on the sensor (um), converted with the bundle camera's pixel pitch; "
                   "null: blur_sigma_px"}),
      // D13 item 3 (F076): the PSF's diffraction term, added to the core in optics.blur_sigma_px (resolve.cpp).
      row("optics.diffraction",
          {.type = Type::kBool, .value = false, .status = choice, .cls = C::kTScene, .stage = Stage::kRender,
           .source = "D13 item 3 (F076): DESIGN §8.1 sigma_diff = 0.42 lambda N (G6 t12); off: Phase 1's PSF",
           .help = "Add the Airy core's Gaussian sigma 0.42 lambda N (f_number, wavelength_nm) in quadrature to the "
                   "blur (one per run)"}),
      row("optics.wavelength_nm",
          {.unit = "nm", .value = 550.0, .hard = above(0), .soft = closed(400, 1000), .status = prior,
           .cls = C::kTSensor, .stage = Stage::kRender, .when = {{"optics.diffraction", true}},
           .source = "DESIGN §8.1 / G6 t12's 550 nm; the Prophesee RGB weights (R8 t8) at 612/549/465 nm: RMS 574 nm",
           .help = "Effective wavelength of the sensor's band for the diffraction PSF (visible 550, NIR 850)"}),
      // Scene source.
      row("scene.path", {.type = Type::kPath, .def = req, .cls = C::kTScene, .stage = scene,
                         .source = "C1 scene bundle",
                         .help = "Scene YAML or bundle directory; required unless scene.evimo2.sequence is set or a "
                                 "bundle is given"}),
      // D13 item 1: the background radiance of the rays that miss (scene.path sources; EVIMO2: scene.evimo2.*). Units:
      // the renderer's shaded radiance rho (E/pi + A), so a background B equals a backdrop of albedo rho_bg = B / A lit
      // by the ambient A alone. 0 (no background light) keeps every bundle, profile and bench clip as before.
      row("scene.background_radiance",
          {.unit = "1", .value = 0.0, .hard = at_least(0), .status = choice, .cls = C::kTScene,
           .stage = Stage::kRender, .source = "D13 item 1 (PARAMETERS §2.4.1); 0: none, the Phase 1 black void",
           .help = "Radiance of rays that miss all geometry (0: none), as shaded radiance rho (E/pi + A): rho A matches "
                   "ambient A"}),
      // D-P3-11: the literal default is absent from digests, preserving every pre-scan profile.
      row("scene.baked_scale",
          {.unit = "1", .value = 1.0, .hard = at_least(0), .soft = closed(0.01, 100), .status = choice,
           .cls = C::kTScene, .stage = Stage::kRender, .source = "Phase 3 scans (SCANS.md §4.1)",
           .help = "Multiplier on baked radiance and every proxy light representing its capture lighting; 1: as authored"}),
      row("scene.background_group",
          {.type = Type::kString, .value = std::string("default"), .status = choice, .cls = C::kTScene,
           .stage = Stage::kRender, .source = "D13 item 1",
           .help = "Light group (name) of scene.background_radiance: the background follows its waveform"}),
      // P2-F F4 (plan p2f §4.3): DESIGN §7.7's reflection policy, applied by ballasim.engine.engine.run before the
      // engine is built. Stage none: it refuses a run or lets it run, never changing the events of a run that runs, and
      // enters no stage digest and no digest(scene). A scene-scope row is still a key of every resolved scene's values,
      // so W0's T10 (a) golden (scene_values, phase3 w2-schema) must be re-recorded once after the merge.
      row("scene.allow_missing_reflections",
          {.type = Type::kBool, .value = false, .status = choice, .cls = C::kTScene, .stage = none,
           .source = "DESIGN §7.7 (DL-37); PHASE2 P2-F F4 (plan p2f §4.3)",
           .help = "Run specular materials that v1 refuses (roughness < 0.2, or < 0.5 if metallic): reflections "
                   "missing, bounds in run.json"}),
      row("scene.evimo2.sequence",
          {.type = Type::kPath, .def = req, .cls = C::kTScene, .stage = scene, .source = "EVIMO2 release layout",
           .help = "EVIMO2 sequence directory (expands ${EVIMO2_ROOT}); required unless scene.path is set"}),
      row("scene.evimo2.camera", {.type = Type::kString, .nullable = true, .status = choice, .cls = C::kTRun,
                                  .stage = scene, .source = "EVIMO2 release layout",
                                  .help = "Camera folder of the recording (left_camera, right_camera, samsung_mono); "
                                          "null: the sequence's"}),
      row("scene.evimo2.markers", {.type = Type::kBool, .value = true, .status = choice, .cls = C::kTScene,
                                   .stage = scene, .source = "S2 AST-24: the spheres are physically present",
                                   .help = "Add the Vicon marker spheres"}),
      row("scene.evimo2.marker_subdivisions",
          {.type = Type::kInt, .unit = "1", .value = int64_t{4}, .hard = closed(0, 8), .status = derived,
           .cls = C::kNAccuracy, .stage = scene, .source = "PARAMETERS §4.4 (sagitta 7.7 um at 12.5 mm)",
           .help = "Icosphere level of the marker meshes"}),
      row("scene.evimo2.clock_reference",
          {.type = Type::kEnum, .value = std::string("npz"), .choices = {"npz", "raw"}, .status = choice,
           .cls = C::kTRun, .stage = scene, .source = "G1 §2.1",
           .help = "Sensor clock of the events: the released NPZ clock or the raw clock"}),
      row("scene.evimo2.start_trim_ns",
          {.type = Type::kInt, .unit = "ns", .value = int64_t{50'000'000}, .hard = at_least(0), .status = placeholder,
           .cls = C::kEHeuristic, .stage = scene, .source = "DESIGN §5.2 (CFG-TRJ-070); its removal is deferred",
           .help = "Span trimmed after the first rig sample (bundle meta key warmup_ns)"}),
      row("scene.evimo2.objects", {.type = Type::kIntList, .nullable = true, .length = -1, .unit = "1",
                                   .status = choice, .cls = C::kTRun, .stage = scene, .source = "run choice",
                                   .help = "EVIMO object ids to include; null: every object with a catalogue mesh"}),
      row("scene.evimo2.extra_cameras",
          {.type = Type::kStringList, .length = -1, .value = std::vector<std::string>{}, .status = choice,
           .cls = C::kTRun, .stage = scene, .source = "run choice",
           .help = "Other camera folders of the recording added to the bundle"}),
      row("scene.evimo2.min_samples", {.type = Type::kInt, .unit = "1", .value = int64_t{10}, .hard = at_least(0),
                                       .status = placeholder, .cls = C::kEHeuristic, .stage = scene,
                                       .source = "arbitrary (PARAMETERS §3.4.3)",
                                       .help = "Objects with fewer Vicon samples are skipped"}),
      row("scene.evimo2.static_tabletop",
          {.type = Type::kBool, .value = true, .status = choice, .cls = C::kEHeuristic, .stage = scene,
           .source = "bistable tabletop Vicon solution (scene10)",
           .help = "Hold the tabletop still when most of its samples are (it defines the world frame)"}),
      row("scene.evimo2.static_tabletop_fraction",
          {.unit = "1", .value = 0.5, .hard = closed(0, 1), .status = placeholder, .cls = C::kEHeuristic,
           .stage = scene, .source = "PARAMETERS §3.4.3 (a majority rule, no derivation)",
           .help = "Share of samples within scene.trajectory.hold_max_disp_m that makes the tabletop static"}),
      row("scene.evimo2.pin_static_px_s",
          {.nullable = true, .unit = "px/s", .hard = above(0), .status = choice, .cls = C::kTScene, .stage = scene,
           .source = "P2-H M10-HOLD (2026-10-15): scene-static means a GT image speed under 60 px/s; null: off (v0, v1)",
           .help = "Freeze each body where its Vicon image speed is below this; frozen for the run, it is static; null: off"}),
      row("scene.evimo2.lights", {.type = Type::kBool, .value = true, .status = choice, .cls = C::kTRun,
                                  .stage = scene, .source = "run choice",
                                  .help = "Place the placeholder lab lights (without them the bundle is dark)"}),
      row("scene.evimo2.low_light",
          {.type = Type::kBool, .def = Default::kDerived,
           .formula = "an _ll recording, lit by the Vicon strobe (the bundle's relight sidecar)", .status = derived,
           .cls = C::kDMeasured, .stage = Stage::kRender, .source = "EVIMO2 release layout (category *_ll; G3 t9)",
           .help = "Low-light recording (strobe-lit): a fact of the sequence, not settable; picks the lighting keys"}),
      // EVIMO2 lighting (P0_PARAMS §3.4): required while the lights are on.
      row("scene.evimo2.ambient",
          {.unit = "1", .def = req, .hard = at_least(0), .cls = C::kTScene, .stage = Stage::kRender, .when = normal,
           .source = lab,
           .help = "Ambient radiance-equivalent A: shading is rho (E/pi + A), so 0.3 is 0.94x the unit-irradiance point "
                   "light's rho/pi"}),
      row("scene.evimo2.low_light_ambient",
          {.unit = "1", .def = req, .hard = at_least(0), .cls = C::kTScene, .stage = Stage::kRender, .when = low,
           .source = lab,
           .help = "Constant ambient of the _ll sequences next to the strobe (radiance-equivalent: rho A beside rho E/pi)"}),
      // A ratio to the light mode's ambient, not a radiance: one value serves both modes (the _ll backdrop is lit by the
      // dim low-light ambient) and stays dimensionless through D1's photometric chain.
      row("scene.evimo2.background_albedo",
          {.unit = "1", .value = 0.0, .hard = closed(0, 1), .status = placeholder, .cls = C::kTScene,
           .stage = Stage::kRender, .when = lit,
           .source = "D13 item 1; 0 keeps v0's black void until the round-12 refit; start near 0.25-0.8 (0.3-1x the "
                     "tabletop's median rendered radiance, 0.24, over the v0 ambient 0.3; walls get less light)",
           .help = "Albedo of a backdrop for the misses (the unscanned room) lit by the mode's ambient A alone: B = "
                   "rho_bg A, in A's group"}),
      row("scene.evimo2.light_height_m",
          {.unit = "m", .def = req, .hard = above(0), .cls = C::kTScene, .stage = Stage::kRender, .when = lit,
           .source = lab, .help = "Point light above its anchor along the tabletop normal"}),
      row("scene.evimo2.light_color",
          {.type = Type::kFloatList, .length = 3, .unit = "1", .def = req, .hard = at_least(0), .cls = C::kTScene,
           .stage = Stage::kRender, .when = lit, .source = lab, .help = "Linear RGB colour of the placeholder lights"}),
      row("scene.evimo2.light_shadows", {.type = Type::kBool, .def = req, .cls = C::kTScene, .stage = Stage::kRender,
                                         .when = lit, .source = lab,
                                         .help = "The placeholder lights cast hard shadows"}),
      row("scene.evimo2.ambient_flickers",
          {.type = Type::kBool, .def = req, .cls = C::kTScene, .stage = Stage::kRender, .when = normal, .source = lab,
           .help = "The ambient term belongs to the mains light group (flickers with it)"}),
      row("scene.evimo2.mains_a",
          {.type = Type::kFloatList, .length = -1, .unit = "1", .def = req, .hard = closed(-1, 1), .cls = C::kTScene,
           .stage = Stage::kSchedule, .when = normal, .source = lab,
           .help = "Mains harmonic amplitudes a_k (sum |a_k| <= 1, at most 8)"}),
      row("scene.evimo2.strobe_width_s",
          {.unit = "s", .def = req, .hard = above(0), .soft = closed(1e-4, 5e-4), .cls = C::kTScene,
           .stage = Stage::kSchedule, .when = low, .source = lab,
           .help = "Strobe full width at half height (_ll sequences)"}),
      row("scene.evimo2.strobe_floor",
          {.unit = "1", .def = req, .hard = right_open(0, 1), .cls = C::kTScene, .stage = Stage::kSchedule,
           .when = low, .source = lab, .help = "Strobe level between flashes"}),
      row("scene.evimo2.mains_f0_hz",
          {.unit = "Hz", .def = req, .hard = above(0), .cls = C::kTScene, .stage = Stage::kSensor, .when = normal,
           .source = lab, .help = "Mains fundamental frequency"}),
      row("scene.evimo2.mains_phi_rad",
          {.type = Type::kFloatList, .length = -1, .unit = "rad", .def = req, .cls = C::kTScene,
           .stage = Stage::kSensor, .when = normal, .source = lab, .help = "Mains harmonic phases phi_k (one per a_k)"}),
      row("scene.evimo2.strobe_rise_s",
          {.unit = "s", .def = req, .hard = at_least(1e-9), .cls = C::kTScene, .stage = Stage::kSensor, .when = low,
           .source = lab, .help = "Strobe rise and fall time"}),
      row("scene.evimo2.strobe_phase_offset_s",
          {.unit = "s", .def = req, .cls = C::kTScene, .stage = Stage::kSensor, .when = low, .source = lab,
           .help = "Strobe rise relative to each re-timed Vicon sample"}),
      row("scene.evimo2.marker_albedo",
          {.unit = "1", .def = req, .hard = closed(0, 1), .cls = C::kTScene, .stage = scene, .source = lab,
           .help = "Lambert albedo of the marker spheres (a proxy for retro-reflectors)"}),
      row("scene.evimo2.untextured_albedo",
          {.unit = "1", .def = req, .hard = closed(0, 1), .cls = C::kTScene, .stage = scene, .source = lab,
           .help = "Grey albedo of the catalogue objects without a texture"}),
      row("scene.evimo2.marker_radius_m",
          {.nullable = true, .unit = "m", .hard = above(0), .status = Status::kData, .cls = C::kDMeasured,
           .stage = scene, .source = "EVIMO catalogue (DESIGN Q13)",
           .help = "Marker sphere radius; null: the catalogue value (12.5 mm)"}),
      row("scene.evimo2.objects_root",
          {.type = Type::kPath, .def = Default::kDerived,
           .formula = "$EVIMO2_ROOT/.tools/evimo/evimo/objects (the environment, or the machine default)",
           .status = derived, .cls = C::kTRun, .stage = scene, .source = "PARAMETERS §2.5 item 17, D13 item 4",
           .help = "EVIMO object catalogue: any folder in its layout (e.g. a colour-fixed copy); cached by file content"}),
      row("scene.evimo2.raw_root",
          {.type = Type::kPath, .def = Default::kDerived,
           .formula = "$EVIMO2_RAW_ROOT (the environment, or the machine default)", .status = derived,
           .cls = C::kTRun, .stage = scene, .source = "PARAMETERS §2.5 item 17",
           .help = "Raw EVIMO archive (clock_reference raw)"}),
      // Trajectory fitting (traj.SmoothingOptions and gating; D15a; statuses from PARAMETERS §3.4.4).
      row("scene.trajectory.order", {.type = Type::kInt, .unit = "1", .value = int64_t{6}, .hard = closed(4, 6),
                                     .status = derived, .cls = C::kNAccuracy, .stage = scene,
                                     .source = "C1: 4 or 6 (DESIGN §5.3)",
                                     .help = "Spline order K of the fitted trajectories (4 or 6)"}),
      row("scene.trajectory.loss_c2", {.unit = "1", .value = 16.81, .hard = above(0), .status = derived,
                                       .cls = C::kNAccuracy, .stage = scene, .source = "chi2_6(0.99) (PARAMETERS §4.4)",
                                       .help = "Cauchy loss constant c^2 of the robust fit"}),
      row("scene.trajectory.gate", {.type = Type::kBool, .value = true, .status = choice, .cls = C::kTRun,
                                    .stage = scene, .source = "DESIGN §5.4",
                                    .help = "Reject outlier samples before the fit (gate G-a)"}),
      row("scene.trajectory.gate_sigma_k", {.unit = "1", .value = 8.0, .hard = above(0), .status = prior,
                                            .cls = C::kEHeuristic, .stage = scene,
                                            .source = "DESIGN §5.4 step 3; M2 C.2 (mocap-noise dependent)",
                                            .help = "Gate G-a: robust sigma multiplier"}),
      row("scene.trajectory.gate_rot_floor_rad",
          {.unit = "rad", .value = 0x1.acee9f37bebd6p-5, .hard = at_least(0), .status = prior, .cls = C::kEHeuristic,
           .stage = scene, .source = "DESIGN §5.4: 3 degrees (math.radians(3.0))",
           .help = "Gate G-a: smallest rotation outlier"}),
      row("scene.trajectory.gate_pos_floor_m", {.unit = "m", .value = 0.010, .hard = at_least(0), .status = prior,
                                                .cls = C::kEHeuristic, .stage = scene, .source = "DESIGN §5.4",
                                                .help = "Gate G-a: smallest position outlier"}),
      row("scene.trajectory.mahalanobis_max", {.unit = "1", .value = 8.0, .hard = above(0), .status = prior,
                                               .cls = C::kEHeuristic, .stage = scene,
                                               .source = "DESIGN §5.4 (P(> 8) ~ 7e-12 in 6-D)",
                                               .help = "Gate G-b: Mahalanobis limit of the one refit"}),
      row("scene.trajectory.lambda_pos", {.nullable = true, .unit = "1", .hard = above(0), .status = derived,
                                          .cls = C::kEHeuristic, .stage = scene,
                                          .source = "estimator choice (DESIGN §5.4); null: estimated by GCV",
                                          .help = "Position smoothing weight (u-units); null: GCV per track"}),
      row("scene.trajectory.lambda_rot", {.nullable = true, .unit = "1", .hard = above(0), .status = derived,
                                          .cls = C::kEHeuristic, .stage = scene,
                                          .source = "estimator choice (DESIGN §5.4); null: estimated by GCV",
                                          .help = "Rotation smoothing weight (u-units); null: GCV per track"}),
      row("scene.trajectory.gcv_log10_range",
          {.type = Type::kFloatList, .length = 2, .unit = "1", .value = std::vector<double>{-4.0, 12.0},
           .hard = closed(-30, 30), .status = derived, .cls = C::kNAccuracy, .stage = scene,
           .source = "search interval; edge hits reported (PARAMETERS §3.4.4); hard [-30, 30] keeps the search "
                     "finite (at most 61 grid points of one decade)",
           .help = "log10 search interval of the GCV smoothing weights"}),
      row("scene.trajectory.detect_holds", {.type = Type::kBool, .value = true, .status = choice, .cls = C::kTRun,
                                            .stage = scene, .source = "DESIGN §5.6",
                                            .help = "Detect intervals where a body is exactly still"}),
      row("scene.trajectory.hold_max_disp_m",
          {.unit = "m", .value = 1e-3, .hard = at_least(0), .status = placeholder, .cls = C::kEHeuristic,
           .stage = scene, .source = "DESIGN §5.2, §5.6 (to be derived from image space and mocap noise)",
           .help = "Largest displacement inside a hold (hold_rule sphere; the EVIMO2 static_tabletop test under both rules)"}),
      row("scene.trajectory.hold_min_ns",
          {.type = Type::kInt, .unit = "ns", .value = int64_t{30'000'000}, .hard = at_least(0), .status = prior,
           .cls = C::kEHeuristic, .stage = scene, .source = "DESIGN §5.6 (false alarm 1.5e-5 per test)",
           .help = "Shortest hold"}),
      row("scene.trajectory.hold_moving_mps",
          {.unit = "m/s", .value = 0.02, .hard = at_least(0), .status = placeholder, .cls = C::kEHeuristic,
           .stage = scene, .source = "DESIGN §5.6 (eps_on; its 5 sigma_s noise term is missing)",
           .help = "Speed above which a body is clearly moving (hold_rule sphere)"}),
      row("scene.trajectory.hold_rule",
          {.type = Type::kEnum, .value = std::string("sphere"), .choices = {"sphere", "image"}, .status = choice,
           .cls = C::kEHeuristic, .stage = scene,
           .source = "F090 (PHASE2 §3; SCENE.md); sphere: DESIGN §5.6 as built, the rule of every v0 run",
           .help = "Hold test: sphere (object space: hold_max_disp_m, hold_moving_mps) or image (hold_max_disp_px)"}),
      row("scene.trajectory.hold_max_disp_px",
          {.unit = "px", .value = 0.5, .hard = above(0), .status = placeholder, .cls = C::kEHeuristic, .stage = scene,
           .source = "F090 (SCENE.md): half a pixel, above the pre-fit's Vicon wander, below a 1 deg/s 200 ms pan",
           .help = "hold_rule image: largest image excursion in a hold; eps_on = this / hold_min_ns"}),
      row("scene.trajectory.branch_jump_rad",
          {.unit = "rad", .value = 0x1.921fb54442d18p+0, .hard = above(0), .status = prior, .cls = C::kEHeuristic,
           .stage = scene, .source = "P1-M5 flip fix; the per-knot rule (gating.BRANCH_JUMP_RAD = pi / 2)",
           .help = "Orientation-branch flip: smallest turn between kept samples"}),
      row("scene.trajectory.branch_rate_factor",
          {.unit = "1", .value = 10.0, .hard = above(0), .status = placeholder, .cls = C::kEHeuristic, .stage = scene,
           .source = "gating.BRANCH_RATE_FACTOR (a 7 rev/s wheel stays below it)",
           .help = "Orientation-branch flip: turn rate above this times the 95th percentile"}),
      // Blender exchanges (docs/design/phase3/blender.md §3.11; D-P3-11): filled for a Blender source only (Inputs::
      // source_kind), so no other run gains a value or a digest moves. The conversion rows reach ConvertOptions
      // (python/ballasim/blender/convert.py OPTION_KEYS), the motion rows the session's SampleOptions and
      // DetectOptions (BL-9); every value enters blender_cache_key and bundle.meta.blender.values.
      row("scene.blender.policy",
          {.type = Type::kEnum, .value = std::string("strict"), .choices = {"strict", "lenient"}, .status = choice,
           .cls = C::kTScene, .stage = scene, .source = "blender.md §1 guarantee 3, §6.1",
           .help = "Unsupported Blender features: refuse, or approximate with a warning"}),
      row("scene.blender.lm_per_w",
          {.nullable = true, .unit = "lm/W", .hard = above(0), .soft = closed(100, 683), .status = choice,
           .cls = C::kTScene, .stage = scene,
           .source = "blender.md §4.4 (CV E1): 683 lm/W at 555 nm (the candela's definition); Blender's bulb table "
                     "implies 227-556",
           .help = "Luminous efficacy of Blender watts (683: the definition); physical photometry needs it, else the "
                   "converter uses 1"}),
      row("scene.blender.world_as_ambient",
          {.type = Type::kBool, .value = false, .status = choice, .cls = C::kTScene, .stage = scene,
           .source = "blender.md §4.4 (CV E12)",
           .help = "Also light surfaces with the world colour as unshadowed ambient (at least 1.8x over-lit near "
                   "contact)"}),
      row("scene.blender.crop_radius_m",
          {.nullable = true, .unit = "m", .hard = left_open(0, 16), .status = choice, .cls = C::kTScene,
           .stage = scene, .source = "blender.md §4.6; the 16 m reach of C1 (DESIGN §5.3)",
           .help = "Crop static geometry to this distance from origin_world; null: refuse scenes beyond 16 m"}),
      row("scene.blender.motion_tol_px",
          {.unit = "px", .value = 0.01, .hard = above(0), .soft = closed(0.001, 0.1), .status = choice,
           .cls = C::kTScene, .stage = scene, .source = "blender.md §4.7.2 (D-P3-2: eps = 0.01 px)",
           .help = "Image-space tolerance eps of the fitted motion"}),
      row("scene.blender.fps_base_snap",
          {.type = Type::kBool, .value = false, .status = choice, .cls = C::kTScene, .stage = scene,
           .source = "blender.md §4.2", .help = "Snap fps_base to p/q with q in {1, 1000, 1001}"}),
      row("scene.blender.max_triangles",
          {.type = Type::kInt, .unit = "1", .value = int64_t{20'000'000}, .hard = at_least(1), .status = choice,
           .cls = C::kTRun, .stage = scene, .source = "blender.md §7 (DS d11: 136 B per triangle on the device, 110 B "
                                                       "on the host)",
           .help = "Refuse scenes with more triangles after instancing (GEO-BUDGET)"}),
      row("scene.blender.cache_samples_per_frame",
          {.type = Type::kInt, .unit = "1", .value = int64_t{1}, .hard = at_least(1), .soft = closed(1, 16),
           .status = choice, .cls = C::kTScene, .stage = scene, .source = "blender.md §4.7.8",
           .help = "Samples per frame of transform caches (Alembic, USD) in frame-state mode; set it to the cache's own "
                   "rate"}),
      // Heuristics of the motion fit and the conversion (blender.md §3.11, §4.7.5-§4.7.8; MF and CV: its research tags,
      // phase3_research/blender/{motion_fit, conversion}), the user's to tune; the hard ranges are the C++ checks'
      // where they exist (scene/motion/pipeline.cpp).
      row("scene.blender.detect_k_sigma",
          {.unit = "1", .value = 6.0, .hard = above(0), .status = prior, .cls = C::kEHeuristic, .stage = scene,
           .source = "MF E7 (no false join, every kink found)",
           .help = "Slope or value jumps above this many standard errors are breakpoint candidates"}),
      row("scene.blender.detect_burst_samples",
          {.type = Type::kInt, .unit = "1", .value = int64_t{8}, .hard = at_least(3), .status = prior,
           .cls = C::kEHeuristic, .stage = scene, .source = "MF E7",
           .help = "Samples per side of one localization burst"}),
      row("scene.blender.detect_burst_rounds",
          {.type = Type::kInt, .unit = "1", .value = int64_t{4}, .hard = at_least(0), .status = prior,
           .cls = C::kEHeuristic, .stage = scene, .source = "MF E7",
           .help = "Bursts per candidate (the window shrinks 4x each)"}),
      row("scene.blender.detect_burst_floor_ns",
          {.type = Type::kInt, .unit = "ns", .value = int64_t{64'000}, .hard = at_least(1), .status = prior,
           .cls = C::kEHeuristic, .stage = scene, .source = "MF E7", .help = "Smallest burst window"}),
      row("scene.blender.detect_merge_ns",
          {.type = Type::kInt, .unit = "ns", .value = int64_t{100'000}, .hard = at_least(0), .status = prior,
           .cls = C::kEHeuristic, .stage = scene, .source = "MF E7", .help = "Candidates closer than this are merged"}),
      row("scene.blender.impact_k_sigma",
          {.unit = "1", .value = 30.0, .hard = above(0), .status = prior, .cls = C::kEHeuristic, .stage = scene,
           .source = "MF a3", .help = "Frame-state impact threshold in robust sigma of d2"}),
      row("scene.blender.impact_floor_m",
          {.unit = "m", .value = 1e-6, .hard = at_least(0), .status = prior, .cls = C::kEHeuristic, .stage = scene,
           .source = "MF a3", .help = "Smallest d2 departure counted as an impact"}),
      row("scene.blender.rigidity_tol_m",
          {.unit = "m", .value = 1e-5, .hard = above(0), .status = placeholder, .cls = C::kEHeuristic,
           .stage = scene, .source = "CV §3.2",
           .help = "Object-frame vertex motion above this refuses the body (GEO-DEFORM)"}),
      row("scene.blender.calib_tol_px",
          {.unit = "px", .value = 0.01, .hard = above(0), .status = placeholder, .cls = C::kEHeuristic,
           .stage = scene, .source = "CV §3.7",
           .help = "CAM-CALIB warns when Blender's pinhole moves a corner pixel by more"}),
      row("scene.blender.fisheye_fit_tol_px",
          {.unit = "px", .value = 0.05, .hard = above(0), .status = placeholder, .cls = C::kEHeuristic,
           .stage = scene, .source = "CV E6 (default lens 0.010)",
           .help = "LENS_POLYNOMIAL fisheyes whose kb fit is worse than this are refused"}),
      row("scene.blender.fit_lambda_rel",
          {.unit = "1", .value = 1e-10, .hard = at_least(0), .status = prior, .cls = C::kEHeuristic, .stage = scene,
           .source = "MF a6/a7", .help = "Relative jerk regularization of the fit"}),
      row("scene.blender.validation_probes",
          {.type = Type::kInt, .unit = "1", .value = int64_t{32}, .hard = at_least(0), .status = placeholder,
           .cls = C::kEHeuristic, .stage = scene, .source = "MF a7",
           .help = "Seeded probes per piece, besides the breakpoint probes"}),
      row("scene.blender.max_halvings",
          {.type = Type::kInt, .unit = "1", .value = int64_t{5}, .hard = at_least(0), .status = placeholder,
           .cls = C::kEHeuristic, .stage = scene, .source = "MF a7",
           .help = "Knot-spacing halvings before MOT-FIT"}),
      // Run.
      row("run.t_start_ns", {.type = Type::kInt, .nullable = true, .unit = "ns", .status = choice, .cls = C::kTRun,
                             .stage = Stage::kSchedule, .source = "run choice",
                             .help = "Start of the simulated span (true clock; whole windows inside it); null: the "
                                     "bundle's"}),
      row("run.t_end_ns", {.type = Type::kInt, .nullable = true, .unit = "ns", .status = choice, .cls = C::kTRun,
                           .stage = Stage::kSchedule, .source = "run choice",
                           .help = "End of the simulated span; null: the bundle's"}),
      row("run.select", {.type = Type::kStringList, .nullable = true, .length = -1, .status = choice,
                         .cls = C::kTRun, .stage = Stage::kOutput, .source = "run choice",
                         .help = "Bundle cameras to simulate; null: all"}),
      // Accuracy.
      // The accuracy table and the fidelity design inputs are read by the resolver alone: their effect reaches the
      // engine (and the digests) only through the keys they derive, so they invalidate nothing themselves (§1.2).
      row("accuracy.profile",
          {.type = Type::kEnum, .value = std::string("fidelity"),
           .choices = {"phase1", "fidelity", "balanced", "reference"}, .status = choice, .cls = C::kNAccuracy,
           .stage = none, .source = "D8; balanced: ENGINE_CPP I7; reference: DESIGN §6.8, PHASE2 P2-0a",
           .help = "Accuracy table: phase1 (frozen), fidelity (derived from the sensors), balanced (economy), "
                   "reference (conformance)"}),
      row("accuracy.design_contrast", {.unit = "log-e", .value = 1.0, .hard = above(0), .status = choice,
                                       .cls = C::kNAccuracy, .stage = none, .when = fid,
                                       .source = "D8 (P0_PARAMS §1.3): the accuracy target",
                                       .help = "Design edge contrast c of the derived sub-sample count"}),
      row("accuracy.design_threshold",
          {.unit = "log-e", .def = Default::kDerived,
           .formula = "max(0.10, min over the run's cameras and polarities of theta - 3 sigma_theta) (PHASE2 F085); "
                      "min over the cameras of min(theta_on, theta_off) under an EVIMO2 v0 / v1 sensor profile",
           .hard = above(0), .status = derived, .cls = C::kNAccuracy, .stage = none, .when = fid,
           .source = "D8 (P0_PARAMS §1.3); PHASE2 F085 (P2-B B4a): the mismatch's 3-sigma tail, hot pixels excluded",
           .help = "Design threshold of the derived sub-sample count"}),
      // D12 (ENGINE_CPP I6'): the runs from t_s on do not depend on the warm-up (a motion warm-up renders its own
      // windows, then every pixel at t_s as a static one does), but a motion warm-up plans, renders and samples the
      // windows [t_m, t_s): schedule-stage products, so both keys are `schedule` and the runs a cache keys by
      // digest(schedule) include the warm-up's (P0_PARAMS §1.2; I6p review).
      row("accuracy.warmup_ns",
          {.type = Type::kInt, .unit = "ns", .value = int64_t{6} << 23, .hard = at_least(8'388'608),
           .status = placeholder, .cls = C::kEHeuristic, .stage = Stage::kSchedule,
           .source = "DESIGN §8.4: 50 ms in whole windows, at least 50 ms when static (D12: I6'); motion: >= 1 window",
           .help = "Sensor warm-up before the first window, events dropped (static: >= 50 ms; motion: whole windows, "
                   "rounded up)"}),
      row("accuracy.warmup", {.type = Type::kEnum, .value = std::string("static"), .choices = {"static", "motion"},
                              .status = choice, .cls = C::kNAccuracy, .stage = Stage::kSchedule,
                              .source = "D12 (PARAMETERS §5.9): warm up on the real motion",
                              .help = "Warm-up input: static (the first frame held) or motion (the covered windows, "
                                      "then static; >= 50 ms in all)"}),
      // Execution (output-invariant).
      row("execution.scene_workers", {.type = Type::kInt, .unit = "1", .value = int64_t{1}, .hard = at_least(1),
                                      .status = choice, .cls = C::kTRun, .stage = none, .source = "execution choice",
                                      .help = "Threads for the EVIMO2 per-body fits (results identical)"}),
      // Output.
      row("output.dir", {.type = Type::kPath, .def = req, .cls = C::kTRun, .stage = none, .source = "run choice",
                         .help = "Run directory"}),
      row("output.overwrite", {.type = Type::kBool, .value = false, .status = choice, .cls = C::kTRun, .stage = none,
                               .source = "run choice", .help = "Replace an existing run directory"}),
      // Dataset facts (viz; D16b): null derives them from the scene; evimo2/release@v0 gives the EVIMO2 values.
      row("dataset.evimo2.mains_hz", {.nullable = true, .unit = "Hz", .hard = above(0), .status = derived,
                                      .cls = C::kDMeasured, .stage = none, .source = "G3 §2.7",
                                      .help = "Mains frequency for viz stats; null: from the scene's light groups"}),
      row("dataset.evimo2.strobe_hz", {.nullable = true, .unit = "Hz", .hard = above(0), .status = derived,
                                       .cls = C::kDMeasured, .stage = none, .source = "G3 §2.7",
                                       .help = "Strobe frequency for viz stats; null: from the scene's light groups"}),
      row("dataset.evimo2.gt_lookup_tol_s",
          {.nullable = true, .unit = "s", .hard = above(0), .status = derived, .cls = C::kDMeasured, .stage = none,
           .source = "EVIMO2 GT at 60 Hz", .help = "Tolerance of the GT frame lookup; null: the GT frame spacing"}),
      row("dataset.evimo2.samsung_on_polarity",
          {.type = Type::kInt, .nullable = true, .unit = "1", .hard = closed(0, 1), .status = derived,
           .cls = C::kDMeasured, .stage = none, .source = "G3 §2.8",
           .help = "Polarity value of ON events in Samsung recordings; null: 1"}),
  };
  return rows;
}

std::vector<Param> make_schema() {
  std::vector<Param> all;
  for (const auto* rows : {&sensor::param_rows(), &engine::sensor_spec_rows(), &render::scene_settings_rows(),
                           &render::renderer_options_rows(), &engine::engine_config_rows()}) {
    all.insert(all.end(), rows->begin(), rows->end());
  }
  for (Param& p : untargeted_rows()) all.push_back(std::move(p));
  // The IMU's rows (outputs.md §3.1, P3-A A3): camera scope, stage output, no target; after every other row, so the
  // order of the digests of a run without an IMU does not change.
  all.insert(all.end(), imu::param_rows().begin(), imu::param_rows().end());
  // The frames' rows (frames.md §3.2, P3 FR-1b): the camera section `frames` and three run rows, all stage none and
  // without a target; last, so that no earlier row moves.
  all.insert(all.end(), frames::param_rows().begin(), frames::param_rows().end());
  return all;
}

std::vector<AccuracyProfile> make_accuracy_profiles() {
  const auto v = [](const char* key, Value value) { return AccuracyEntry{key, std::move(value), ""}; };
  const auto f = [](const char* key, const char* formula) { return AccuracyEntry{key, Value{}, formula}; };
  // Frozen: test_accuracy.py pins this table, so later changes of the literals never move the references.
  AccuracyProfile phase1{
      "phase1",
      "The frozen Phase-1 accuracy table: every accuracy key at the value the P1 runs and the bench references used",
      {v("accuracy.n_subsamples", int64_t{64}), v("accuracy.d_px", 0.25), v("accuracy.eps_L", 0.002),
       v("accuracy.refine_tol_log", 0.002), v("accuracy.refine_signature", true), v("accuracy.max_refine_levels", int64_t{4}),
       v("accuracy.min_dt_ns", int64_t{1024}), v("accuracy.max_samples_per_window", int64_t{1} << 25),
       // P2-H H9 (F004): every tile at its own level.
       v("accuracy.level_policy", std::string("tile")),
       v("accuracy.noise_leaf_level", int64_t{5}), v("accuracy.noise_margin_z", 6.0),
       v("accuracy.noise_diffusion", std::string("frozen")),
       v("accuracy.warmup_ns", int64_t{50'331'648}), v("accuracy.held_world", true),
       v("accuracy.psf_truncation_sigmas", 3.0), v("accuracy.warp_cells", int64_t{256}),
       v("accuracy.lut_samples_per_px", int64_t{2}), v("accuracy.sensor_precision", std::string("fp64")),
       v("accuracy.warmup", std::string("static")), v("accuracy.initial_state", std::string("phase")),
       v("accuracy.stationary_burnin_ns", int64_t{0}), v("accuracy.stationary_probe_pixels", int64_t{32}),
       // ENGINE_CPP I7: the economy schedule's switches off (Phase 1's schedule).
       v("accuracy.mid_levels", Value{}), v("accuracy.tau_jump_ns", Value{}), v("accuracy.delta_ref", 0.0),
       v("accuracy.half_ratio", 0.0), v("accuracy.eps_jump", Value{}),
       // ENGINE_CPP I8: the cone bound of moving shadows.
       v("accuracy.shadow_band", false),
       // P2-D D1: the shadow-edge rates without the obliquity factors.
       v("accuracy.shadow_obliquity", std::string("off")),
       // P2-D D3a: flips localized by bisection.
       v("accuracy.localization", std::string("bisect")),
       // PHASE2 D-P2-3 (P2-B B3): every term over all N sub-samples.
       v("accuracy.pixel_integral", std::string("phase1")),
       // PHASE2 D-P2-4 (P2-B B4a): one N.
       v("accuracy.n_band_max", Value{}),
       // PHASE2 P2-C C2: every shadow bit as traced.
       v("accuracy.unlit_mask", false),
       // PHASE2 P2-F F1: the pooled shadow ratio over texture.
       v("accuracy.texture_shadow", std::string("pooled")),
       // PHASE2 P2-F F5: clusters keyed by material.
       v("accuracy.shading_patches", std::string("material")),
       // PHASE2 P2-F F9a: the ambient unoccluded by movers.
       v("accuracy.mover_ao", std::string("off"))}};
  AccuracyProfile fidelity{
      "fidelity",
      "Derived from the run's sensors (D8), with D12's start: equal to phase1 but for the clamp at the three family "
      "priors, the stationary initial state, a motion warm-up of two windows, the band N (P2-B; 256 but under an "
      "ideal reset) as the per-pixel band at reach scales (0.25, 0), the jump gate (P2-B's adoption), and the "
      "Jacobian vignetting law per camera (P2-F F13-min: camera key optics.vignetting_law, jacobian where the gain map "
      "is on and the camera is not under an EVIMO2 v0 or v1 sensor profile; phase1 and reference list cos4)",
      {f("accuracy.n_subsamples",
         "smallest 4^k (k >= 2) >= N_step = (e^c - 1) / (e^(theta_d / 4) - 1), c = accuracy.design_contrast, theta_d "
         "= accuracy.design_threshold; at most 256 (warned above)"),
       f("accuracy.d_px", "2 / sqrt(N), N = the resolved accuracy.n_subsamples"),
       f("accuracy.eps_L", "0.002 min(1, sigma_v,min / 0.02); sigma_v,min = min over the cameras with noise of "
                           "sqrt(P22) of the stationary noise at a1 = 1e9 / s (1 without noise)"),
       f("accuracy.refine_tol_log", "the eps_L formula (never an explicit accuracy.eps_L)"),
       f("accuracy.min_dt_ns", "max(1024, pow2ceil(q_min)), q_min = the smallest readout quantum > 0 in ns"),
       f("accuracy.max_samples_per_window", "null: no clamp (D9)"),
       v("accuracy.warmup", std::string("motion")),
       f("accuracy.warmup_ns", "2 windows (16 777 216) under a motion warm-up (D12); a static one keeps 6 (50.3 ms)"),
       v("accuracy.initial_state", std::string("stationary")),
       f("accuracy.noise_diffusion",
         "transient (PHASE2 D-P2-2); frozen when a camera takes its sensor keys from an EVIMO2 v0 or v1 profile (fitted "
         "under it)"),
       f("accuracy.pixel_integral",
         "fixed64 (PHASE2 D-P2-3; phase1's values at N <= 64, so at every prior); phase1 when a camera takes its sensor "
         "keys from an EVIMO2 v0 or v1 profile (their runs stay bitwise at any N)"),
       f("accuracy.n_band_max",
         "256 (PHASE2 D-P2-4, G1's interim band N; P2-B B4a), N_base 16 raised to 64 beside it; null (one N) for an "
         "ideal-reset camera, an EVIMO2 v0 or v1 sensor profile, N_base >= 256 or an explicit N_base that is not a "
         "power of two >= 64"),
       // P2-B's adoption (the user, 2026-10-12; ENGINE_CPP §0.1 "P2-B adoption notes"): the jump gate (R3) and B6c's
       // per-pixel band, held by the statistical gate against uniform N_band (BENCH.md) instead of T2 bit for bit.
       f("accuracy.eps_jump",
         "the eps_L formula, fidelity's tolerance (0.002 at the family priors; P2-B R3); null when a camera takes its "
         "sensor keys from an EVIMO2 v0 or v1 profile (their runs stay bitwise)"),
       f("accuracy.band_growth",
         "pixel (P2-B B6c) with the band N where every camera's pixel response reaches 0.75 px (0.5 sensor.aperture_px "
         "+ min(accuracy.psf_truncation_sigmas, 3) optics.blur_sigma_px; pooled escalation's premise); else tile"),
       f("accuracy.band_reach", "[0.25, 0] under the pixel map (P2-B B6c's measured reach scales); else [1, 1]")}};
  // ENGINE_CPP I7 (D8): fidelity with the economy schedule as the sampling tier accepted it ("I7 notes", "I7 review
  // resolution"): the bench's ref2 schedule (jumps located to tau 2^12 ns, absolute, with the residual gate; 4 midpoint
  // levels) at 4x fidelity's d_px, capped at DESIGN's 1 px. The samples study's other ES values (midpoint tolerance 4x,
  // tau 2^14 scaled by delta_ref 0.04, the half rule 0.25) each failed a bench clip and are off, and so are 3 midpoint
  // levels since the I7 review (they failed the bright textured card); d_px and the gate follow fidelity's derivations.
  AccuracyProfile balanced{
      "balanced",
      "fidelity with the economy schedule (ENGINE_CPP I7): the bench's ref2 schedule (4 midpoint levels, jumps located "
      "to 2^12 ns, the residual gate) at 4x fidelity's d_px, at most 1 px; the moving-shadow band (I8); fidelity's "
      "vignetting law",
      {f("accuracy.n_subsamples", "fidelity's"),
       f("accuracy.d_px", "min(4 x fidelity's, 1 px) = min(8 / sqrt(N), 1): 1 px at N = 16 and 64, 0.5 px at N = 256"),
       f("accuracy.eps_L", "fidelity's"),
       f("accuracy.refine_tol_log", "fidelity's"),
       f("accuracy.min_dt_ns", "fidelity's"),
       f("accuracy.max_samples_per_window", "null: no clamp (D9)"),
       v("accuracy.warmup", std::string("motion")),
       f("accuracy.warmup_ns", "fidelity's"),
       v("accuracy.initial_state", std::string("stationary")),
       v("accuracy.max_refine_levels", int64_t{13}), v("accuracy.mid_levels", int64_t{4}),
       v("accuracy.tau_jump_ns", int64_t{1} << 12), v("accuracy.delta_ref", 0.0), v("accuracy.half_ratio", 0.0),
       f("accuracy.eps_jump", "fidelity's tolerance (0.002 at the family priors)"),
       f("accuracy.noise_diffusion", "fidelity's"),
       // ENGINE_CPP I8: the planner's moving-shadow band (the sampling tier on 16 clips; after the I8 review every tile
       // with a droppable cone rate stays dynamic, "I8 review resolution").
       v("accuracy.shadow_band", true),
       f("accuracy.pixel_integral", "fidelity's"),
       // PHASE2 P2-B: the band N off until EFC (the lead's adoption: fidelity only).
       v("accuracy.n_band_max", Value{})}};
  // PHASE2 P2-0a: DESIGN §6.8's conservative conformance scheduler as far as the engine allows, kept in CI for the
  // project's life, so it lists every key phase1 lists (no literal or default change moves it). A uniform N: the
  // design's N = 4096 in predicted silhouette and shadow bands needs P2-B's bands. Every moving pixel at <= 0.25 px of
  // bounded motion, every signature flip localized to min_dt (13 levels reach it from a window-long lattice interval;
  // no tau stop, no residual gate), the midpoint test at fidelity's tolerance on every level, no clamp, fidelity's
  // start, the cone bound of moving shadows.
  AccuracyProfile reference{
      "reference",
      "The conformance scheduler (DESIGN §6.8, PHASE2 P2-0a): N = 256 uniform, d_px 0.25, every flip localized to "
      "min_dt, the midpoint test at fidelity's tolerance on every level, no clamp, fidelity's start",
      {v("accuracy.n_subsamples", int64_t{256}), v("accuracy.d_px", 0.25),
       f("accuracy.eps_L", "fidelity's"), f("accuracy.refine_tol_log", "fidelity's"),
       v("accuracy.refine_signature", true), v("accuracy.max_refine_levels", int64_t{13}),
       f("accuracy.min_dt_ns", "fidelity's: flips located within min_dt / 2, t_q = max(q / 2, 512 ns) for q <= 1 us"),
       v("accuracy.level_policy", std::string("tile")),
       v("accuracy.max_samples_per_window", Value{}), v("accuracy.noise_leaf_level", int64_t{5}),
       v("accuracy.noise_margin_z", 6.0), f("accuracy.noise_diffusion", "fidelity's"),
       f("accuracy.warmup_ns", "fidelity's"), v("accuracy.held_world", true),
       v("accuracy.psf_truncation_sigmas", 3.0), v("accuracy.warp_cells", int64_t{256}),
       v("accuracy.lut_samples_per_px", int64_t{2}), v("accuracy.sensor_precision", std::string("fp64")),
       v("accuracy.warmup", std::string("motion")), v("accuracy.initial_state", std::string("stationary")),
       v("accuracy.stationary_burnin_ns", int64_t{0}), v("accuracy.stationary_probe_pixels", int64_t{32}),
       v("accuracy.mid_levels", Value{}), v("accuracy.tau_jump_ns", Value{}), v("accuracy.delta_ref", 0.0),
       v("accuracy.half_ratio", 0.0), v("accuracy.eps_jump", Value{}), v("accuracy.shadow_band", false),
       v("accuracy.shadow_obliquity", std::string("off")), v("accuracy.localization", std::string("bisect")),
       v("accuracy.pixel_integral", std::string("phase1")), v("accuracy.n_band_max", Value{}),
       v("accuracy.unlit_mask", false), v("accuracy.texture_shadow", std::string("pooled")),
       v("accuracy.shading_patches", std::string("material")), v("accuracy.mover_ao", std::string("off"))}};
  return {std::move(phase1), std::move(fidelity), std::move(balanced), std::move(reference)};
}

}  // namespace

const std::vector<AccuracyProfile>& accuracy_profiles() {
  static const std::vector<AccuracyProfile> p = make_accuracy_profiles();
  return p;
}

const AccuracyProfile* find_accuracy_profile(std::string_view name) {
  for (const AccuracyProfile& p : accuracy_profiles()) {
    if (p.name == name) return &p;
  }
  return nullptr;
}

const std::vector<Param>& schema() {
  static const std::vector<Param> rows = make_schema();
  return rows;
}

const Param* find(std::string_view key) {
  for (const Param& p : schema()) {
    if (p.key == key) return &p;
  }
  return nullptr;
}

std::vector<const Param*> rows_of(Target target) {
  std::vector<const Param*> out;
  for (const Param& p : schema()) {
    if (p.target == target) out.push_back(&p);
  }
  return out;
}

Value default_value(std::string_view key, std::string_view family) {
  const Param* p = find(key);
  if (!p) throw std::invalid_argument("unknown parameter " + std::string(key));
  switch (p->def) {
    case Default::kLiteral:
      return p->value;
    case Default::kFamily: {
      if (family.empty()) throw std::invalid_argument(p->key + " has a family default: name the family");
      const sensor::Prior* v = sensor::find_prior(family, p->key);
      if (!v) throw std::invalid_argument("unknown sensor family " + std::string(family));
      return v->value;
    }
    case Default::kRequired:
      throw std::invalid_argument(p->key + " is required: it has no default");
    case Default::kDerived:
      break;
  }
  throw std::invalid_argument(p->key + " is derived: " + p->formula);
}

}  // namespace balla::config
