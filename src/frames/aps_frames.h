// APS frames of a camera (docs/design/phase3/frames.md §4.10, §3.9): the photometry inputs the APS shares with the
// DVS pixel and the frame loop over the readout model (sensor/aps.h); `frames aps` and ingest call aps_frames.
//
// For ROI pixel x in read-row r of frame k, with F = mono (renderer units, waveform-weighted; NaN: no preimage):
//   S(x) = (s 1e-15 T / q) (g_x F(x) + g E_k),   E_k = mean of g_x F over the ROI's finite pixels (Neumaier, row-major)
// s the camera's photocurrent scale (run.json cameras[].illuminance_scale, fA per renderer unit), g_x its gain map
// (cos^4 under vignetting, as the engine's), g optics.veiling_glare, T the exposure, q the elementary charge.
// E over the ROI approximates the per-row veil of the full frame (§4.10). Row r starts at frame_soe[k] + rho(r) Delta
// on the true clock (timing.h), the APS draws' key.
#pragma once

#include <stdint.h>

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "frames/timing.h"
#include "render/bundle.h"
#include "sensor/aps.h"

namespace balla::frames {

// The frames.aps.* values (§3.2) and their defaults: a C++ option struct until FR-1b's rows (PHASE3 §0 rule 1).
struct ApsSettings {
  int32_t bits = 10;                      // [8, 16]
  double full_well_e = 30900.0;           // > 0
  double read_noise_e = 55.0;             // >= 0
  double dark_current_e_s = 18000.0;      // >= 0
  double prnu = 0.042;                    // >= 0
  double dsnu_e = 20.0;                   // >= 0
  std::optional<double> gain_dn_per_e;    // > 0; null: (2^bits - 1) / full_well_e
  double offset_dn = 16.0;                // >= 0
  std::optional<int64_t> seed;            // >= 0; null: run.sensor_seed
};
// The model's parameters of the values; FRM-VALUE outside §3.2's bounds.
sensor::ApsParams aps_params(const ApsSettings&, uint64_t run_sensor_seed);

struct ApsPhotometry {
  double illuminance_scale_fA = 0.0;  // s > 0 (FRM-APS-SCALE otherwise: frames-only cameras have none)
  double veiling_glare = 0.0;         // g >= 0
  std::vector<float> gain;            // ROI H x W g_x; empty: 1
};
// optics.vignetting as the engine resolves it: the value when set, else on under physical and reference photometry.
bool vignetting_on(std::optional<bool> vignetting, std::string_view photometry_mode);
// The engine's gain map over the ROI (row-major; Camera::init_photometry), bit for bit: at each pixel centre's unit ray
// (the camera model's injective domain at support_px) cos^4 (optics.vignetting_law cos4) or, with `jacobian`,
// cos(theta) dOmega/dA of the camera model (core/camera/solid_angle.h), times optics.relative_illumination's G
// (core/camera/relative_illumination.h; [] or [c2, c4]). std::invalid_argument for a pixel 90 degrees or more off axis
// or coefficients the engine refuses.
std::vector<float> vignetting_gain(const render::Camera&, double support_px, Roi, bool jacobian = false,
                                   const std::vector<double>& relative_illumination = {});

struct ApsFrames {
  std::vector<uint16_t> dn;        // n x H x W
  std::vector<double> veil_mean;   // E_k per frame
  int64_t normal_fallbacks = 0;    // FRM-APS-RNG
  std::vector<Warning> warnings;   // FRM-APS-EXPOSURE, FRM-APS-RNG
};
inline constexpr int64_t kApsMaxExposureNs = 4'194'303'000;  // libcaer's exposure range, 0 .. 4194303 us

// mono: n x H x W of the ROI; frame_soe_ns: n anchors (the first-read row's start of exposure, true clock); timing:
// the frames' exposure, row delay, shutter, readout and full-image height. Frames and rows run on `threads` workers
// (<= 0: the hardware's); the result does not depend on them.
ApsFrames aps_frames(const sensor::ApsParams&, const ApsPhotometry&, uint32_t camera, int32_t full_width, Roi,
                     const RowTiming&, std::span<const int64_t> frame_soe_ns, std::span<const float> mono,
                     int threads = 0);

// frames_digest (§3.6) in two levels, so that `frames aps` can rewrite it from the file alone: the writer records
// frames_digest_base over §3.6's parts but the APS (values: canonical "key=value\n" lines of the other frames.* and
// execution.frames_* values; empty digests where a part does not exist), and frames_digest is SHA-256 over the base
// and the resolved APS parameters (none when the file has no APS output).
struct DigestParts {
  std::string format, bundle_digest, camera_digest, values, plan_digest, manifest_sha256;
};
std::string frames_digest_base(const DigestParts&);
std::string aps_canonical(const sensor::ApsParams&);
std::string frames_digest(std::string_view base, const sensor::ApsParams* aps);

}  // namespace balla::frames
