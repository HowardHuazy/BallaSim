// SensorParams rows and the sensor family priors (docs/dev/P0_PARAMS.md §1.3, §3.2; sources: PARAMETERS.md §3.1).
#include "sensor/param_fields.h"

#include <string>
#include <utility>
#include <vector>

namespace balla::sensor {

using config::above;
using config::at_least;
using config::closed;
using config::Default;
using config::member;
using config::open;
using config::Param;
using config::right_open;
using config::Scale;
using config::Stage;
using config::Status;
using config::Type;
using config::Value;
using C = config::Class;
using P = SensorParams;

namespace {

std::vector<std::string> family_names() {
  std::vector<std::string> out;
  for (const Family& f : families()) out.push_back(f.name);
  return out;
}

// The hot-pixel rows after sensor.hot_fraction (hot_pixel_key): inert without hot pixels.
constexpr std::string_view kHotPixelKeys[] = {"sensor.hot_rate_law",        "sensor.hot_rate_min_hz",
                                              "sensor.hot_rate_max_hz",     "sensor.hot_rate_median_hz",
                                              "sensor.hot_rate_sigma_ln",   "sensor.hot_leak_fraction",
                                              "sensor.hot_reference_current_fA"};

std::vector<Param> make_rows() {
  const auto fam = Default::kFamily;
  const auto sensor = Stage::kSensor;
  const auto schedule = Stage::kSchedule;
  const auto T = C::kTSensor;
  const std::vector<config::When> lognormal{{"sensor.hot_rate_law", std::string("log_normal")}};
  const std::vector<config::When> cascade{{"sensor.photoreceptor", std::string("cascade")}};
  const std::vector<config::When> stationary{{"accuracy.initial_state", std::string("stationary")}};
  const std::vector<config::When> limiting{{"sensor.limiter", std::string("epoch")}};
  return {
      member<&P::family>("sensor.family",
                         {.type = Type::kEnum, .def = Default::kRequired, .choices = family_names(), .cls = T,
                          .stage = sensor, .source = "D2: never a silent default",
                          .help = "Sensor family whose priors fill every family-default key"}),
      // Photoreceptor (M3 A.3-A.8): the cascade, or the instantaneous log response of D3.
      member<&P::photoreceptor>("sensor.photoreceptor",
                                {.def = fam, .cls = C::kTRun, .stage = sensor, .source = "D3 (PARAMETERS §5.9)",
                                 .help = "cascade: M3's three-stage photoreceptor; instantaneous: ln I itself (textbook "
                                         "DVS), no cascade or noise"}),
      member<&P::k>("sensor.k", {.unit = "1/(A s)", .def = fam, .hard = above(0), .soft = closed(0.5e16, 4e16),
                                 .scale = Scale::kLog, .cls = T, .stage = sensor, .when = cascade,
                                 .source = "range: M3 Tab. A.1; R2 §3.4 (1e16-1e17)",
                                 .help = "Photoreceptor constant: stage-1 pole k I (bandwidth, latency) and input-node "
                                         "noise k q"}),
      member<&P::tau_o>("sensor.tau_o", {.unit = "s", .def = fam, .hard = above(0), .soft = closed(5e-6, 500e-6),
                                         .cls = T, .stage = sensor, .when = cascade, .source = "range: M3 Tab. A.1",
                                         .help = "Output-node pole: bright latency floor, bright noise floor, flicker "
                                                 "attenuation"}),
      member<&P::tau_sf>("sensor.tau_sf",
                         {.unit = "s", .def = fam, .hard = above(0), .soft = closed(30e-6, 1500e-6), .cls = T,
                          .stage = sensor, .when = cascade, .source = "range: M3 Tab. A.1; R2 (f_sf 0.1-2 kHz)",
                          .help = "Source-follower pole: latency floor, noise smoothing; differs from tau_o by >= 10 %"}),
      member<&P::sigma_o>("sensor.sigma_o",
                          {.unit = "log-e", .def = fam, .hard = at_least(0), .soft = closed(0, 0.08), .cls = T,
                           .stage = sensor, .when = cascade, .source = "range: M3 Tab. A.1",
                           .help = "Unfiltered output-node noise, a phenomenological out-of-loop floor: the bright floor "
                                   "of the comparator noise"}),
      member<&P::noise>("sensor.noise", {.def = fam, .cls = C::kTRun, .stage = sensor, .when = cascade,
                                         .source = "model switch (M3 §D)",
                                         .help = "The cascade's in-path noise on the 2^16 ns lattice; off makes every "
                                                 "hot pixel leak-type"}),
      // Input (M3 A.1-A.2).
      // Required under relative photometry; the resolver derives it under physical and reference (resolve.cpp).
      member<&P::illuminance_scale>("photometry.illuminance_scale",
                                    {.unit = "fA", .def = Default::kRequired, .hard = above(0), .scale = Scale::kLog,
                                     .cls = C::kTScene, .stage = schedule,
                                     .when = {{"photometry.mode", std::string("relative")}},
                                     .source = "PARAMETERS §5.5 (relative photometry; D1)",
                                     .help = "Photocurrent per unit of renderer irradiance; required: relative "
                                             "photometry has no neutral scale"}),
      // Stage sensor: glare enters the sensor's input formation (pixel.h GlareRamp) from the frame means of the inputs
      // the pixels hold, which the sensor derives from its own runs; the schedule ignores it.
      member<&P::veiling_glare>("optics.veiling_glare",
                                {.unit = "1", .hard = right_open(0, 1), .soft = closed(0, 0.1),
                                 .status = Status::kChoice, .cls = C::kTScene, .stage = sensor,
                                 .source = "ISO 9358 veiling glare index VGI = g / (1 + g), so g = VGI / (1 - VGI); "
                                           "off by default (no lens data; P1_RESULTS item 5)",
                                 .help = "Veiling glare: a uniform irradiance g x each light group's frame-mean "
                                         "irradiance, added to the image; g = VGI/(1 - VGI)"}),
      member<&P::dark_current_fA>("sensor.dark_current_fA",
                                  {.unit = "fA", .def = fam, .hard = above(0), .soft = closed(0.05, 10),
                                   .scale = Scale::kLog, .cls = T, .stage = schedule,
                                   .source = "range: M3 Tab. A.1; IMX636 5.55 fA (McReynolds 2024)",
                                   .help = "Median dark current at reference_temperature_K: low-light contrast "
                                           "compression"}),
      member<&P::dark_sigma_ln>("sensor.dark_sigma_ln",
                                {.unit = "1", .def = fam, .hard = at_least(0), .soft = closed(0.1, 0.7), .cls = T,
                                 .stage = sensor, .source = "range: M3 Tab. A.1",
                                 .help = "Log-normal spread of the per-pixel dark currents (fixed-pattern noise)"}),
      member<&P::dark_ea_eV>("sensor.dark_ea_eV",
                             {.unit = "eV", .def = fam, .hard = at_least(0), .soft = closed(0.68, 0.90), .cls = T,
                              .stage = schedule, .source = "range: M3 Tab. A.1",
                              .help = "Arrhenius activation energy of the dark current"}),
      member<&P::temperature_K>("sensor.temperature_K",
                                {.unit = "K", .def = fam, .hard = above(0), .cls = T, .stage = schedule,
                                 .source = "die temperature at capture (PARAMETERS §3.1.2)",
                                 .help = "Die temperature: Arrhenius factors of dark current and leak (not of hot "
                                         "pixels)"}),
      member<&P::reference_temperature_K>("sensor.reference_temperature_K",
                                          {.unit = "K", .def = fam, .hard = above(0), .cls = T, .stage = schedule,
                                           .source = "PARAMETERS §2.3 (kT0 of maps.cpp)",
                                           .help = "Temperature at which dark_current_fA and leak_rate_hz are quoted "
                                                   "(Arrhenius reference)"}),
      // Comparator (M3 A.14-A.19).
      member<&P::theta_on>("sensor.theta_on",
                           {.unit = "log-e", .def = fam, .hard = above(0), .soft = closed(0.12, 0.45), .cls = T,
                            .stage = sensor, .source = "range: M3 Tab. A.1; R2 §3.4",
                            .help = "Mean ON threshold: log-intensity rise that fires an ON event"}),
      member<&P::theta_off>("sensor.theta_off",
                            {.unit = "log-e", .def = fam, .hard = above(0), .soft = closed(0.12, 0.45), .cls = T,
                             .stage = sensor, .source = "range: M3 Tab. A.1; R2 §3.4",
                             .help = "Mean OFF threshold: log-intensity drop that fires an OFF event"}),
      member<&P::sigma_theta_on>("sensor.sigma_theta_on",
                                 {.unit = "log-e", .def = fam, .hard = at_least(0), .soft = closed(0.02, 0.04),
                                  .cls = T, .stage = sensor, .source = "range: M3 Tab. A.1; D11",
                                  .help = "Frozen per-pixel spread of theta_on (threshold mismatch)"}),
      member<&P::sigma_theta_off>("sensor.sigma_theta_off",
                                  {.unit = "log-e", .def = fam, .hard = at_least(0), .soft = closed(0.02, 0.04),
                                   .cls = T, .stage = sensor, .source = "range: M3 Tab. A.1; D11",
                                   .help = "Frozen per-pixel spread of theta_off (threshold mismatch)"}),
      member<&P::theta_onoff_corr>("sensor.theta_onoff_corr",
                                   {.unit = "1", .def = fam, .hard = closed(-1, 1), .cls = T, .stage = sensor,
                                    .source = "D11; PARAMETERS §3.1.3 (a differencing-amplifier offset makes it < 0)",
                                    .help = "Correlation of the ON and OFF mismatch draws"}),
      member<&P::theta_floor>("sensor.theta_floor",
                              {.unit = "log-e", .def = fam, .hard = above(0), .cls = C::kEHeuristic, .stage = sensor,
                               .source = "v2e clamp (M3 A.19)",
                               .help = "Clamp of the drawn thresholds and of the Rice hot thresholds; bounds eps_ci"}),
      member<&P::reset_mode>("sensor.reset_mode",
                             {.def = fam, .cls = C::kTRun, .stage = sensor, .source = "M3 A.16-A.17",
                              .help = "physical: blind on (t_c, t_c + T_r], reset at release; ideal: ESIM-like r += p "
                                      "theta"}),
      member<&P::refractory_s>("sensor.refractory_s",
                               {.unit = "s", .def = fam, .hard = right_open(0, 10), .soft = closed(1e-4, 1e-3),
                                .cls = T, .stage = sensor, .source = "range: M3 Tab. A.1",
                                .help = "Refractory period T_r (the ideal reset's dead time)"}),
      member<&P::eps_ci>("sensor.eps_ci", {.unit = "log-e", .def = fam, .cls = T, .stage = sensor,
                                           .source = "R2 §2.1E",
                                           .help = "Charge-injection offset of the reference at release; |eps_ci| < "
                                                   "theta_floor / 2"}),
      member<&P::initial_phase>("sensor.initial_phase",
                                {.def = fam, .cls = C::kEHeuristic, .stage = sensor,
                                 .source = "M3 A.20; DESIGN §8.4 (D12: stationary law in I6')",
                                 .help = "Initial reference phase: uniform U[0, theta_on), symmetric U(-theta_off, "
                                         "theta_on) or zero"}),
      // Leak (M3 A.18).
      member<&P::leak_rate_hz>("sensor.leak_rate_hz",
                               {.unit = "Hz", .def = fam, .hard = at_least(0), .soft = closed(1e-3, 0.1),
                                .scale = Scale::kLog, .cls = T, .stage = sensor,
                                .source = "range: M3 Tab. A.1; R2 §3.4 (0.01-0.1 Hz: v2e's leak to total stationary "
                                          "BA, the leak's upper bound)",
                                .help = "Median leak rate R_L at reference_temperature_K: ON drift theta_on R_L"}),
      member<&P::leak_sigma_log10>("sensor.leak_sigma_log10",
                                   {.unit = "1", .def = fam, .hard = at_least(0), .cls = T, .stage = sensor,
                                    .source = "v2e convention (R2 §2.3)", .help = "Spread of log10 R_L across pixels"}),
      member<&P::leak_ea_eV>("sensor.leak_ea_eV",
                             {.unit = "eV", .def = fam, .hard = at_least(0), .soft = closed(0.68, 0.90), .cls = T,
                              .stage = sensor, .source = "range: M3 Tab. A.1",
                              .help = "Arrhenius activation energy of the leak"}),
      // Prophesee high-pass filter (DESIGN §8.4 A.14; SENSOR.md "High-pass filter"). Soft range closed(0, 10) holds
      // 0, so an explicit off never warns; validate() refuses 0 < tau_hp < max(1 ms, 10 max(tau_o, tau_sf)).
      member<&P::tau_hp>(
          "sensor.tau_hp",
          {.unit = "s", .def = fam, .hard = closed(0, 100), .soft = closed(0, 10), .scale = Scale::kLog, .cls = T,
           .stage = sensor, .when = cascade,
           .source = "bias_hpf: f_c = 1 / (2 pi tau_hp); Gen3.1 bias 900-1800 mV (default 1499, raising it lengthens "
                     "tau), Gen4.1 / IMX636 0-120 and GenX320 0-127 (raising them shortens tau); no published "
                     "bias-to-tau map (M3 Tab. A.1: off); v2e's 10 ms soft floor is its default only. Below 50 ms the "
                     "unfiltered sub-cell noise (DESIGN §6.4) exceeds 0.03 sigma_t",
           .help = "Prophesee high-pass filter: time constant of the reference toward x3 (0: off; else >= max(1 ms, 10 "
                   "tau_o, 10 tau_sf))"}),
      member<&P::tau_hp_sigma_ln>(
          "sensor.tau_hp_sigma_ln",
          {.unit = "1", .hard = at_least(0), .soft = closed(0, 1), .status = Status::kChoice, .cls = T,
           .stage = sensor, .when = cascade,
           .source = "no per-pixel bias_hpf spread is published; 0 until a fit (inert while tau_hp is 0; draws "
                     "clamped below at tau_hp's floor)",
           .help = "Per-pixel log-normal spread of tau_hp (ln units); inert while the filter is off"}),
      // Hot pixels (DESIGN §8.4, M3 §A.5); the rows after hot_fraction are inert at hot_fraction 0 (kHotPixelKeys).
      member<&P::hot_fraction>("sensor.hot_fraction",
                               {.unit = "1", .def = fam, .hard = closed(0, 1), .soft = closed(0, 3e-3), .cls = T,
                                .stage = sensor,
                                .source = "range: PARAMETERS §3.1.4 (per device and session; EVIMO2 right camera 1.9e-3)",
                                .help = "Fraction of hot pixels (tails of the leak and threshold distributions)"}),
      member<&P::hot_rate_law>("sensor.hot_rate_law",
                               {.def = fam, .cls = T, .stage = sensor, .source = "M3 §A.5; D11",
                                .help = "Law of the hot pixels' target rates: log_uniform on [min, max], or log_normal "
                                        "clamped to it"}),
      member<&P::hot_rate_min_hz>("sensor.hot_rate_min_hz",
                                  {.unit = "Hz", .def = fam, .hard = above(0), .scale = Scale::kLog, .cls = T,
                                   .stage = sensor, .source = "PARAMETERS §3.1.4",
                                   .help = "Lowest hot-pixel target rate"}),
      member<&P::hot_rate_max_hz>("sensor.hot_rate_max_hz",
                                  {.unit = "Hz", .def = fam, .hard = above(0), .scale = Scale::kLog, .cls = T,
                                   .stage = sensor, .source = "PARAMETERS §3.1.4",
                                   .help = "Highest hot-pixel target rate; at least hot_rate_min_hz, and below 1 / "
                                           "refractory_s"}),
      member<&P::hot_rate_median_hz>("sensor.hot_rate_median_hz",
                                     {.unit = "Hz", .def = fam, .hard = above(0), .scale = Scale::kLog, .cls = T,
                                      .stage = sensor, .when = lognormal, .source = "D11",
                                      .help = "Median of the log-normal hot-pixel rates"}),
      member<&P::hot_rate_sigma_ln>("sensor.hot_rate_sigma_ln",
                                    {.unit = "1", .def = fam, .hard = at_least(0), .cls = T, .stage = sensor,
                                     .when = lognormal, .source = "D11",
                                     .help = "ln-spread of the log-normal hot-pixel rates"}),
      member<&P::hot_leak_fraction>("sensor.hot_leak_fraction",
                                    {.unit = "1", .def = fam, .hard = closed(0, 1), .cls = T, .stage = sensor,
                                     .source = "M3 §A.5; D11",
                                     .help = "Share of leak-type (ON-only) hot pixels, the rest threshold-type; "
                                             "inert without in-path noise (all leak-type)"}),
      member<&P::hot_reference_current_fA>("sensor.hot_reference_current_fA",
                                           {.unit = "fA", .def = fam, .hard = above(0), .scale = Scale::kLog,
                                            .cls = C::kEHeuristic, .stage = sensor, .source = "SENSOR.md deviation 10",
                                            .help = "Photocurrent at which threshold-type hot rates are set (Rice "
                                                    "inversion)"}),
      // Readout (M3 F.1-F.2) and sensor clock.
      member<&P::readout_quantum_s>("sensor.readout_quantum_s",
                                    {.unit = "s", .def = fam, .hard = right_open(0, 1), .cls = T, .stage = sensor,
                                     .source = "vendor timestamp formats", .help = "Timestamp quantum (0: none)"}),
      member<&P::readout_delay_s>("sensor.readout_delay_s",
                                  {.unit = "s", .def = fam, .hard = right_open(0, 1), .soft = closed(0, 400e-6),
                                   .cls = T, .stage = sensor, .source = "range: M3 Tab. A.1",
                                   .help = "Constant readout delay d0; at least the quantum when quantizing"}),
      member<&P::clock_offset_ns>("sensor.clock_offset_ns",
                                  {.unit = "ns", .def = Default::kDerived,
                                   .formula = "the bundle camera's clock_offset_ns (C1)", .status = Status::kDerived,
                                   .cls = C::kDMeasured, .stage = sensor, .source = "C1 camera; EVIMO2: -delta of G1",
                                   .help = "Sensor clock offset: t_sensor = t_true + offset (+ drift); overrides the "
                                           "bundle camera's"}),
      member<&P::clock_drift_ppm>("sensor.clock_drift_ppm",
                                  {.unit = "ppm", .def = fam, .hard = open(-1e4, 1e4), .soft = closed(-50, 50),
                                   .cls = C::kDMeasured, .stage = sensor,
                                   .source = "D7; physical range PARAMETERS §3.1.5",
                                   .help = "Sensor clock drift, applied to t from the run epoch (the engine's output "
                                           "cut follows it)"}),
      // Rate limiter (DESIGN D13-5, F043; docs/dev/EVIMO2_LIMITER.md §5; sensor/limiter.h): a token bucket on the
      // time-sorted output stream after the pixel loop, on the sensor clock. Its other rows are inert while it is off.
      member<&P::limiter>("sensor.limiter",
                          {.def = fam, .cls = T, .stage = sensor,
                           .source = "DESIGN D13-5 (F043, EVIMO2_LIMITER §5): the Prophesee driver's max_event_rate "
                                     "(G3 t6); never on a simulated stream in EVIMO2 validation (DL-33)",
                           .help = "Driver event-rate limiter on the sorted output: off, or epoch (a token bucket "
                                   "under a cap falling over each epoch)"}),
      member<&P::limiter_rate0_hz>(
          "sensor.limiter_rate0_hz",
          {.unit = "Hz", .def = fam, .hard = closed(1, 1e10), .scale = Scale::kLog, .cls = T, .stage = sensor,
           .when = limiting, .source = "EVIMO2_LIMITER §3, §5: R0 5.996-6.002 Mev/s on raw stamps, 14 streams",
           .help = "Limiter cap at each epoch's start, in events per second of the sensor clock"}),
      member<&P::limiter_rate1_hz>(
          "sensor.limiter_rate1_hz",
          {.unit = "Hz", .def = fam, .hard = closed(1, 1e10), .scale = Scale::kLog, .cls = T, .stage = sensor,
           .when = limiting,
           .source = "EVIMO2_LIMITER §3, §5: R1 5.635-5.646 Mev/s on raw stamps (one sfm stream 5.67); linear in "
                     "between to 0.003-0.013 Mev/s",
           .help = "Limiter cap at each epoch's end (at most rate0); the cap falls linearly from rate0 over the "
                   "epoch"}),
      member<&P::limiter_period_s>(
          "sensor.limiter_period_s",
          {.unit = "s", .def = fam, .hard = closed(1e-6, 100), .cls = T, .stage = sensor, .when = limiting,
           .source = "EVIMO2_LIMITER §3 finding 1: 1025 x 64 us = 65.600 ms of the sensor clock (65 600.00002 +- "
                     "0.00009 us over one driver session)",
           .help = "Limiter epoch P: the cap steps back to rate0 every P of the sensor clock"}),
      member<&P::limiter_depth>(
          "sensor.limiter_depth",
          {.unit = "1", .def = fam, .hard = closed(1, 1e9), .soft = at_least(16), .scale = Scale::kLog, .cls = T,
           .stage = sensor, .when = limiting,
           .source = "bucket simulation (Poisson 6.5-12 Mev/s offered, 1 us bins, the 6.00 -> 5.64 cap): B 6 gives "
                     "5.0-5.5 Mev/s, B 16 5.78-5.83, B >= 32 the mean cap 5.83; not identified from the EVIMO2 "
                     "stamps (EVIMO2_LIMITER §5)",
           .help = "Limiter bucket depth B, events: >= ceil(rate0 max(quantum, 1 us)) + 1 (validate()), >= 16 for the "
                   "mean cap"}),
      member<&P::limiter_phase_s>(
          "sensor.limiter_phase_s",
          {.unit = "s", .hard = right_open(0, 100), .status = Status::kChoice, .cls = T, .stage = sensor,
           .when = limiting,
           .source = "EVIMO2_LIMITER §5: free per camera and driver session, kept across recordings; an EVIMO2 replay "
                     "takes the measured reset time (tools/evimo2_bench limiter.epoch_phase_ns)",
           .help = "A limiter reset time mod the epoch, on the sensor clock; null: drawn per camera (Philox tag "
                   "0x0023)"}),
      // Numerics (DESIGN §8.3, M3 §D.4): run keys applied to every camera.
      member<&P::eps_L>("accuracy.eps_L",
                        {.unit = "log-e", .hard = config::left_open(0, 0.1), .scale = Scale::kLog,
                         .status = Status::kDerived, .cls = C::kNAccuracy, .stage = sensor,
                         .source = "PARAMETERS §4.1: kappa sigma_v,ref (kappa 0.1, sigma_v,ref 0.02)",
                         .help = "Log-domain tolerance of the cascade's integration (sub-steps and pieces); inert "
                                 "under photoreceptor instantaneous"}),
      member<&P::noise_leaf_level>("accuracy.noise_leaf_level",
                                   {.unit = "1", .hard = closed(0, 10), .status = Status::kDerived,
                                    .cls = C::kNAccuracy, .stage = sensor, .source = "PARAMETERS §4.1",
                                    .help = "Noise-tree leaves of 2^(16-L) ns: accuracy of n3 inside cells vs "
                                            "refinement cost"}),
      member<&P::noise_margin_z>("accuracy.noise_margin_z",
                                 {.unit = "1", .hard = above(0), .status = Status::kDerived, .cls = C::kNAccuracy,
                                  .stage = sensor, .source = "M3 D.8; PARAMETERS §4.1",
                                  .help = "Noise exclusion margin z_m: missed-crossing probability vs cost"}),
      member<&P::noise_diffusion>(
          "accuracy.noise_diffusion",
          {.status = Status::kChoice, .cls = C::kNAccuracy, .stage = sensor,
           .source = "PHASE2 D-P2-2 (F070): the jump oracle (ORACLES.md); frozen: M3 D.1 (phase1, EVIMO2 v0, v1)",
           .help = "Input-node noise diffusion per cell: frozen (2 k q a1, M3 D.1) or transient (n1 exact at the nodes; "
                   "D-P2-2)"}),
      // Initial state (D12, ENGINE_CPP I6'): run keys applied to every camera.
      member<&P::initial_state>(
          "accuracy.initial_state",
          {.status = Status::kChoice, .cls = C::kNAccuracy, .stage = sensor,
           .source = "D12 (PARAMETERS §5.9); SENSOR.md \"Initial state\"",
           .help = "Comparator state at init: phase (sensor.initial_phase, no history) or stationary (the probe's "
                   "stationary law)"}),
      member<&P::stationary_burnin_ns>(
          "accuracy.stationary_burnin_ns",
          {.unit = "ns", .hard = at_least(0), .status = Status::kChoice, .cls = C::kNAccuracy, .stage = sensor,
           .when = stationary,
           .source = "0: 1.5 / R_L clamped to [60, 600] s; the leak cycle is ~0.9 / R_L and the rate is within a few % "
                     "of the long run after 0.35-0.5 / R_L (theta_off 0.25 / 0.375, R_L 0.002-0.02; ENGINE_CPP I6p review)",
           .help = "History of the stationary probe, the frame held, snapshots over its last 2/3 (0: 1.5 / R_L clamped "
                   "to [60, 600] s)"}),
      member<&P::stationary_probe_pixels>(
          "accuracy.stationary_probe_pixels",
          {.unit = "1", .hard = at_least(1), .status = Status::kChoice, .cls = C::kNAccuracy, .stage = sensor,
           .when = stationary,
           .source = "cost ~ ceil(P / threads) x T_b; the early rate's per-run spread falls with P (ENGINE_CPP I6p "
                     "review: P 32 at T_b 300 s, 128 x 128: 4-5 % at the prior, 14-18 % at theta_off 0.375)",
           .help = "Pixels of the stationary probe; the others take a snapshot of their nearest probe's comparator"}),
      // Labels (D14a).
      member<&P::label_share_cut>("output.label_cut",
                                  {.unit = "1", .hard = open(0, 1), .status = Status::kChoice, .cls = C::kTRun,
                                   .stage = sensor, .field = "label_share_cut",
                                   .source = "D14a; G5 convention (a definition, not an estimate); DESIGN §9.2",
                                   .help = "Share of Delta an increment needs for a SIGNAL or LEAK label"}),
  };
}

Prior prior(std::string key, Value v, Status s, std::string source, bool evimo2 = false) {
  return {std::move(key), std::move(v), s, std::move(source), evimo2};
}

// D4c (PARAMETERS §5.9): the literature gives no usable value, so the family prior turns the feature off, and the
// resolver warns once per run while a camera takes it from the prior.
Prior off(std::string key, Value v, std::string source) {
  Prior p = prior(std::move(key), std::move(v), Status::kChoice, std::move(source));
  p.no_literature = true;
  return p;
}

std::vector<Family> make_families() {
  const auto pr = Status::kPrior, ch = Status::kChoice, ph = Status::kPlaceholder, me = Status::kMeasured;
  const std::string nist = "NIST Gen3.1 ON-latency fit, k pinned, tau_o = tau_sf / 4 imposed (M3 V8)";
  // D4 (P0e): refractory, leak and hot pixels are literature midpoints or off, not EVIMO2 values; the EVIMO2 values
  // live in the evimo2 v0 profiles (python/ballasim/profiles/evimo2/).
  Family prophesee{
      "prophesee_gen3", "",
      "Prophesee Gen3 (as in EVIMO2 left and right; DESIGN §8.9): literature priors (D4); hot pixels off for want of "
      "a literature value (the EVIMO2 profiles carry the measured ones)",
      {
          prior("sensor.photoreceptor", std::string("cascade"), ch, "M3 three-stage photoreceptor (DESIGN §8.2)"),
          prior("sensor.k", 1.0e16, pr, "Graça 2023: sigma_1 ~ 0.04 log-e gives k = sigma_1^2 / q (G5 t7)"),
          prior("sensor.tau_o", 25e-6, pr, nist),
          prior("sensor.tau_sf", 100e-6, pr, nist),
          prior("sensor.sigma_o", 0.06, pr, "DAVIS346 I_pr-path estimate (Graça 2023, G5 §2.5)"),
          prior("sensor.noise", true, ch, "model switch"),
          prior("sensor.dark_current_fA", 0.21, pr,
                "DAVIS346 E_dark 2 mlx x 15 um pitch x white-LED responsivity (R8 t4, S2 OPT-09)"),
          prior("sensor.dark_sigma_ln", 0.3, ph, "assumption (M3 O8), unsourced"),
          prior("sensor.dark_ea_eV", 0.78, pr, "unverified 6-8 K doubling (Nozaki & Delbrück 2017; R2 §2.3)"),
          prior("sensor.temperature_K", 298.15, ch, "the reference temperature of the priors"),
          prior("sensor.reference_temperature_K", 298.15, pr, "the dark and leak priors are quoted at 25 C"),
          prior("sensor.theta_on", 0.25, pr, "Prophesee Gen3 NCT 25 % (R2 §2.4); DESIGN §8.9 symmetric prior"),
          prior("sensor.theta_off", 0.25, pr, "Prophesee Gen3 NCT 25 % (R2 §2.4); DESIGN §8.9 symmetric prior"),
          prior("sensor.sigma_theta_on", 0.03, pr, "McReynolds 2024 0.02-0.03; Gallego 2-4 %"),
          prior("sensor.sigma_theta_off", 0.03, pr, "McReynolds 2024 0.02-0.03; Gallego 2-4 %"),
          prior("sensor.theta_onoff_corr", 0.0, ch, "D11: 0 keeps the independent draws of M3 A.19"),
          prior("sensor.theta_floor", 0.01, ch, "v2e clamp (M3 A.19)"),
          prior("sensor.reset_mode", std::string("physical"), ch, "hardware reset at release (R2 t6)"),
          prior("sensor.refractory_s", 0.3e-3, pr,
                "D4: range 0.1-1 ms (M3 Tab. A.1; R2 §3.4, DAVIS346; none published for Prophesee Gen3), geometric "
                "midpoint 0.32 ms rounded; ~0.1 ms practical minimum (R2 §2.1E), v2e 0.5 ms (R2 §2.7), >= 0.33 ms "
                "measured (McReynolds 2023, R7 §2.4), 1 ms (Stoffregen 2020, R7 §2.1)"),
          prior("sensor.eps_ci", 0.0, pr, "R2 §2.1E: ~0 unless T_r < 100 us"),
          prior("sensor.initial_phase", std::string("uniform"), ch, "renewal law of a leak-only reference (M3 A.20)"),
          prior("sensor.leak_rate_hz", 0.01, pr,
                "D4: leak-only range 0.01 Hz, its geometric midpoint 0.01 Hz: the one leak-only figure is v2e's "
                "0.01 Hz (R2 §2.3, §2.7). R2 §3.4's 0.01-0.1 Hz for Prophesee Gen3 mixes it with total stationary "
                "BA (0.1 Hz typical per the v2e paper and Gallego Tab. 1, R2 §2.3), which bounds the leak from above "
                "but is not a leak figure (lead decision after P0e; 0.03 Hz before)"),
          prior("sensor.leak_sigma_log10", 0.1, ph, "v2e convention (R2 §2.3)"),
          prior("sensor.leak_ea_eV", 0.78, pr, "unverified Arrhenius assumption (R2 §2.3)"),
          // A plain choice, not D4c's off(): off models a bias setting (IMX636 bias_hpf 0, its default; Gen3.1 near
          // 1800 mV), not a literature gap, so no per-run warning (PARAMETERS §5.9).
          prior("sensor.tau_hp", 0.0, ch,
                "off: the widest-bandwidth bias (IMX636 bias_hpf 0, its default; Gen3.1 near 1800 mV); no bias-to-tau "
                "map is published (M3 Tab. A.1: off); the EVIMO2 fit goes to opt-in profiles (lead, 2026-10-10)"),
          off("sensor.hot_fraction", 0.0,
              "D4c: off; the literature gives no hot-pixel rates (the only measured ones are EVIMO2's, R2 §2.3), "
              "only fractions: Stoffregen 2020 U(0, 1e-4) (R7 §2.1); EVIMO2 4.3e-4 (R7 §2.5, the v0 profiles' 4e-4) "
              "to 1.4e-3 (scene13, PARAMETERS §3.1.4)"),
          prior("sensor.hot_rate_law", std::string("log_uniform"), ph, "M3 §A.5 assumption"),
          prior("sensor.hot_rate_min_hz", 10.0, ph, "unsourced (PARAMETERS §3.1.4)"),
          prior("sensor.hot_rate_max_hz", 2000.0, me,
                "EVIMO2 per-pixel maxima 1.2-2.3 kHz (R2 §2.3); no literature value; inert while hot pixels are off "
                "(D4c)",
                true),
          prior("sensor.hot_rate_median_hz", 100.0, ph,
                "EVIMO2 hot-pixel median ~90-100 Hz (PARAMETERS §3.1.4); no literature value; inert under "
                "log_uniform and while hot pixels are off (D4c)",
                true),
          prior("sensor.hot_rate_sigma_ln", 1.5, ph,
                "about the ln-spread of log-uniform 10-2000 Hz (ln 200 / sqrt 12 = 1.53); inert under log_uniform"),
          prior("sensor.hot_leak_fraction", 0.5, ph, "M3 §A.5 assumption (50/50 split)"),
          prior("sensor.hot_reference_current_fA", 100.0, ph, "none (SENSOR.md deviation 10)"),
          prior("sensor.readout_quantum_s", 1e-6, pr, "vendor timestamp formats (Prophesee EVT)"),
          prior("sensor.readout_delay_s", 300e-6, pr, "NIST Gen3.1 latency fit with k pinned (M3 V8, 306 us)"),
          prior("sensor.clock_drift_ppm", 0.0, ch, "D7: the EVIMO2 drift is applied from profile v1"),
          // The rate limiter: off, a driver setting the user chooses; its numbers are EVIMO2's driver setting and
          // measurements (flagged), read only when it is on.
          prior("sensor.limiter", std::string("off"), ch,
                "off: a driver setting (max_event_rate), the user's; never on a simulated stream in EVIMO2 validation "
                "(DL-33)"),
          prior("sensor.limiter_rate0_hz", 6.0e6, me,
                "the EVIMO2 driver's max_event_rate 6000 kEv/s (G3 t6); R0 5.996-6.002 Mev/s on raw stamps "
                "(EVIMO2_LIMITER §3)",
                true),
          prior("sensor.limiter_rate1_hz", 6.0e6, ch,
                "a flat cap at the EVIMO2 driver's max_event_rate (G3 t6); the measured fall to 5.64 Mev/s "
                "(EVIMO2_LIMITER §3) belongs to opt-in profiles",
                true),
          prior("sensor.limiter_period_s", 65.6e-3, me,
                "1025 x 64 us of the sensor clock, measured on EVIMO2's raw stamps (EVIMO2_LIMITER §3, finding 1)",
                true),
          prior("sensor.limiter_depth", int64_t{32}, ph,
                "placeholder (K9): B >= 16 reaches the mean cap in a bucket simulation; the stamps do not identify it; "
                "DESIGN's 120 (6 Mev/s x 20 us) predates F043"),
          prior("sensor.rgb_weights", std::vector<double>{0.52, 0.37, 0.11}, pr,
                "R8 t8: Prophesee Gen3.1 QE under a white LED; the EVIMO2 illuminant is unknown", true),
          prior("sensor.aperture_px", 1.0, pr,
                "DESIGN §8.1 / S2 OPT-07 (microlenses implied by QE 37 % at 25 % fill factor); unverified"),
          prior("sensor.qe", 0.37, pr, "Prophesee Gen3.1 effective peak QE at 505 nm (DESIGN §8.9; R8 §2.3, S2 OPT-05)"),
      }};
  Family samsung{
      "samsung_gen3", "prophesee_gen3",
      "Samsung Gen3 read out asynchronously in Phase 1 (the frame readout replaced by its mean blind time and delay); "
      "values not listed are copied from prophesee_gen3, unvalidated",
      {
          prior("sensor.theta_on", 0.20, pr, "Samsung: 99.9 % response below 27.5 % (R2 §3.4)"),
          prior("sensor.theta_off", 0.20, pr, "Samsung: 99.9 % response below 27.5 % (R2 §3.4)"),
          // D4 (P0e): the vendor's default frame period T0 = 200 us (5 kHz), not EVIMO2's measured 197.28 us (the
          // evimo2 samsung_gen3 v0 profile keeps 195 us and 98.64 us).
          prior("sensor.refractory_s", 196e-6, pr,
                "D4: T0 / 2 + 0.3 us x 320 columns at the vendor's default frame period T0 = 200 us (iniVation 5 kHz, "
                "R2 §2.6; DVXplorer 200 us, G4 §2.1; 0.3 us = 15 clocks at 50 MHz, libcaer, G4 §2.2), about a frame "
                "(R2 §3.4); T0 is a host register, documented ~30-200 us (scan ~30-50 us inferred, G4 §1 item 2): "
                "~30-196 us"),
          prior("sensor.readout_delay_s", 100e-6, pr,
                "D4: T0 / 2, the mean report delay, at the vendor's default T0 = 200 us (R2 §2.6); T0 documented "
                "~30-200 us (G4 §1 item 2): ~15-100 us"),
          prior("sensor.leak_rate_hz", 0.01, pr,
                "D4: leak-only range 0.01 Hz, its geometric midpoint 0.01 Hz: R2 §3.4's 0.01-0.03 Hz for Samsung Gen3 "
                "mixes v2e's leak (0.01 Hz, the one leak-only figure) with Gallego Tab. 1's total stationary noise "
                "(0.03 Hz), which bounds the leak from above but is not a leak figure (the leak-only rule of the "
                "lead's decision after P0e, as for prophesee_gen3; 0.02 Hz before)"),
          prior("sensor.dark_current_fA", 0.08, Status::kDerived,
                "0.21 fA x (9/15)^2 = 0.0756 fA, rounded to 0.08: pixel area only (QE omitted)"),
          prior("sensor.tau_hp", 0.0, ch, "off: the Samsung Gen3 has no Prophesee bias_hpf (stated, not inherited)"),
          prior("sensor.limiter", std::string("off"), ch,
                "off: the Samsung Gen3 has no Prophesee driver limiter (its frame readout bounds the rate; stated, not "
                "inherited)"),
          prior("sensor.qe", 0.5, ph, "DESIGN §8.9: assumed for the BSI Samsung Gen3, to be fitted (R8 §2.3, S2 OPT-05)"),
      }};
  Family ideal{
      "ideal", "prophesee_gen3",
      "Textbook DVS (D3): instantaneous log response, exact threshold crossings of ln(I + I_dk); no noise, mismatch, "
      "leak, hot pixels or readout; ideal reset, refractory 0; theta 0.20. photoreceptor: cascade gives the old fast "
      "cascade (k 1e18, poles 1 and 2 us)",
      {
          prior("sensor.photoreceptor", std::string("instantaneous"), ch, "D3: textbook DVS (PARAMETERS §5.9)"),
          prior("sensor.k", 1.0e18, ch, "arbitrary fast photoreceptor (PARAMETERS §3.1.6); inert (instantaneous)"),
          prior("sensor.tau_o", 1e-6, ch, "arbitrary fast pole; inert (instantaneous)"),
          prior("sensor.tau_sf", 2e-6, ch, "arbitrary fast pole, apart from tau_o (pole rule); inert (instantaneous)"),
          prior("sensor.noise", false, ch, "noise-free reference"),
          prior("sensor.dark_current_fA", 0.01, ch, "the floor of the log response, ln(I + I_dk); > 0 is required"),
          prior("sensor.dark_sigma_ln", 0.0, ch, "mismatch-free reference"),
          prior("sensor.theta_on", 0.20, pr, "conventional simulator default (v2e)"),
          prior("sensor.theta_off", 0.20, pr, "conventional simulator default (v2e)"),
          prior("sensor.sigma_theta_on", 0.0, ch, "mismatch-free reference"),
          prior("sensor.sigma_theta_off", 0.0, ch, "mismatch-free reference"),
          prior("sensor.reset_mode", std::string("ideal"), ch, "ESIM-compatible reset"),
          prior("sensor.refractory_s", 0.0, ch, "no dead time"),
          prior("sensor.leak_rate_hz", 0.0, ch, "leak-free reference"),
          prior("sensor.leak_sigma_log10", 0.0, ch, "leak-free reference"),
          prior("sensor.tau_hp", 0.0, ch, "no high-pass filter (textbook DVS); inert under instantaneous"),
          prior("sensor.hot_fraction", 0.0, ch, "no hot pixels"),
          prior("sensor.readout_quantum_s", 0.0, ch, "no quantization"),
          prior("sensor.readout_delay_s", 0.0, ch, "no delay"),
          prior("sensor.limiter", std::string("off"), ch, "no rate limiter (textbook DVS)"),
      }};
  return {std::move(prophesee), std::move(samsung), std::move(ideal)};
}

}  // namespace

const std::vector<Family>& families() {
  static const std::vector<Family> f = make_families();
  return f;
}

const Family* find_family(std::string_view name) {
  for (const Family& f : families()) {
    if (f.name == name) return &f;
  }
  return nullptr;
}

const Prior* find_prior(std::string_view family, std::string_view key, bool* inherited) {
  bool base = false;
  for (const Family* f = find_family(family); f != nullptr; f = f->base.empty() ? nullptr : find_family(f->base)) {
    for (const Prior& p : f->priors) {
      if (p.key == key) {
        if (inherited) *inherited = base;
        return &p;
      }
    }
    base = true;
  }
  return nullptr;
}

const std::vector<Param>& param_rows() {
  static const std::vector<Param> rows = make_rows();
  return rows;
}

const Param* param_row(std::string_view field) {
  for (const Param& r : param_rows()) {
    if (r.field == field) return &r;
  }
  return nullptr;
}

std::string_view removed_field(std::string_view field) {
  // Fields removed from SensorParams, with what replaces them (P0_PARAMS §3.2).
  static constexpr std::pair<std::string_view, std::string_view> kRemoved[] = {
      {"sigma_theta", "split into sigma_theta_on and sigma_theta_off (D11); set both"},
  };
  for (const auto& [name, replacement] : kRemoved) {
    if (name == field) return replacement;
  }
  return {};
}

bool hot_pixel_key(std::string_view key) {
  for (const std::string_view k : kHotPixelKeys) {
    if (k == key) return true;
  }
  return false;
}

SensorParams params_of(const config::ValueMap& run, const config::ValueMap& cam, const std::string& name) {
  SensorParams sp;
  for (const config::Param& p : param_rows()) {
    config::set(p, sp, p.scope == config::Scope::kCamera ? config::value_of(p, cam, "cameras." + name + ".")
                                                         : config::value_of(p, run, ""));
  }
  try {
    sp.validate();
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument("cameras." + name + ": " + e.what());
  }
  return sp;
}

std::string unknown_field(std::string_view field) {
  std::string msg = "unknown sensor parameter '" + std::string(field) + "'";
  if (const std::string_view r = removed_field(field); !r.empty()) msg += ": " + std::string(r);
  return msg;
}

}  // namespace balla::sensor
