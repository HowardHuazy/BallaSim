// Frozen per-pixel sensor maps (DESIGN §8.4, M3 §A.5-A.6): ON/OFF thresholds with mismatch, leak
// drift, dark current, the hot-pixel set and the initial reference phase. Every map is a fixed
// function of (seed, camera, pixel), drawn once from Philox purpose rng::Purpose::kSensorMap and
// never redrawn (redrawing per interval is ESIM's bug, R1 t4).
#pragma once

#include <cstdint>
#include <vector>

#include "sensor/params.h"

namespace balla::sensor {

enum class HotType : uint8_t { kNone = 0, kLeak = 1, kThreshold = 2 };

// One pixel's maps, plain data for the kernel.
struct PixelMap {
  double theta_on;   // log-e
  double theta_off;  // log-e
  double leak;       // ON drift of the reference, log-e / s
  double dark_fA;    // dark current, fA
  double phi0;       // initial reference phase, log-e (Delta(t0) = phi0)
  HotType hot;
  // Static gain of the pixel's photocurrent (natural vignetting, D1): not drawn, 1 unless SensorModel::set_photometry
  // sets the camera's map. I = scale gain sum_g m_g E_g + dark.
  float gain = 1.0f;
};

struct SensorMaps {
  std::vector<PixelMap> pixels;       // y * width + x
  std::vector<double> hot_target_hz;  // drawn target rate of each hot pixel (Hz), 0 elsewhere
  // The high-pass filter's rate b = 1 / tau_hp of each pixel (s^-1) when its spread is on (tau_hp > 0 and
  // tau_hp_sigma_ln > 0; the kernels' side map), empty otherwise (one rate per camera, KernelConstants::hpf_b).
  std::vector<float> hpf_b;
};

// The high-pass filter's time constant of pixel `pixel` (s): tau_hp itself without a spread, else its log-normal draw
// (map block kMapHpf) clamped below at SensorParams::hpf_floor_s(); 0 when the filter is off.
double hpf_tau(const SensorParams& p, uint64_t seed, uint32_t camera, uint32_t pixel);

SensorMaps draw_sensor_maps(const SensorParams& p, uint64_t seed, uint32_t camera, uint32_t n_pixels);
// The maps of the pixels with these global ids (a region-of-interest model, phase3/ui.md §3.10), in their order:
// entry i is what draw_sensor_maps gives pixel ids[i] of the full frame.
SensorMaps draw_sensor_maps(const SensorParams& p, uint64_t seed, uint32_t camera, const std::vector<uint32_t>& ids);
// The lowest dark current (fA) of the frame's pixels [first, last), as draw_sensor_maps gives them, from their dark
// draws alone (+inf for an empty range): an ROI model's floor check covers the whole frame, as the full model's does.
double lowest_dark_fA(const SensorParams& p, uint64_t seed, uint32_t camera, uint32_t first, uint32_t last);

// Threshold at which a stationary in-path noise process crosses +-theta at `rate_hz` in total, from
// Rice's formula 2 nu0 exp(-theta^2 / (2 sigma_v^2)) at photocurrent I_ref (Phase 1 inversion for
// threshold-type hot pixels; Phase 2 replaces it with tabulated BA(theta; I) from the production
// construction, M3 §A.5). Clamped below at theta_floor: targets at or above Rice's rate at theta_floor (all
// targets >= 2 nu0, which no threshold reaches) take theta_floor, so the threshold-type tail is truncated there
// while hot_target_hz keeps the drawn target (SENSOR.md deviation 10).
double rice_threshold(const SensorParams& p, double rate_hz);

// Arrhenius factor exp(-Ea/kB (1/T - 1/T0)) (M3 A.2, A.18); T0 is the reference temperature at which a
// value is quoted (SensorParams::reference_temperature_K, 298.15 K in every family).
double arrhenius(double ea_eV, double temperature_K, double reference_K);

// Median of the pixels' dark currents at the sensor's temperature (fA): dark_current_fA (quoted at
// reference_temperature_K) times its Arrhenius factor, exactly dark_current_fA there (the engines' eta_L
// dark term).
double median_dark_fA(const SensorParams& p);

}  // namespace balla::sensor
