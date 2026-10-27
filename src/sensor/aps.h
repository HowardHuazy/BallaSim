// DAVIS-like APS frame sensor (docs/design/phase3/frames.md §4.10; EMVA 1288 R4.0 eqs. 6, 15, 66-67). It shares
// the photodiode, the photometry and the veil with the DVS pixel; this file holds the readout model, the caller
// gives each pixel's mean signal (frames/aps_frames.h). For pixel x in read-row r of a frame:
//   mu_e = max(0, (1 + p(x)) S(x)) + I_dark T          S(x): the photo-electron mean of the signal (NaN: no preimage)
//   n    = min(Poisson(mu_e) + d(x), FW) + N(0, sigma_r^2)
//   DN   = clip(round_half_even(K n + y_off), 0, 2^bits - 1)      (0 where S is NaN)
// p ~ N(0, PRNU^2) and d ~ N(0, DSNU^2) are static: Philox kApsMap at pixel y W + x of the full image, blocks 0 and
// 1 (Box-Muller, the first normal). The temporal draws are keyed by the row's start of exposure t_soe (true clock):
// kApsShot and kApsRead at index (t_soe + 2^56) 64 + j, j < 64, so rows 1 ns apart share no counter and DN depends
// only on (seed, camera, pixel, t_soe). Poisson: inversion for mu < 10 (block j = 0); otherwise Hoermann's PTRS
// ("The transformed rejection method for generating Poisson random variables", Insurance: Mathematics and Economics
// 12, 1993), one block per trial (U from words 0-1, V from words 2-3); its uniforms take 52 bits of two words, in
// the open (0, 1). After 64 rejected trials the normal approximation round(mu + sqrt(mu) z) with z the second
// Box-Muller normal of the kApsRead block (FRM-APS-RNG, counted). Read noise is the first normal of that block.
// Logarithms, exponentials and sincos are balla::pmath's.
// Host only; every pixel is independent, so results do not depend on threads or order.
#pragma once

#include <stdint.h>

namespace balla::sensor {

struct ApsParams {
  int32_t bits = 10;                // ADC bits, [8, 16]
  double full_well_e = 30900.0;     // FW, e- (> 0)
  double read_noise_e = 55.0;       // sigma_r, e- (>= 0)
  double dark_e_s = 18000.0;        // I_dark, e-/s (>= 0)
  double prnu = 0.042;              // PRNU, relative (>= 0)
  double dsnu_e = 20.0;             // DSNU, e- (>= 0)
  double gain_dn_e = 1023.0 / 30900.0;  // K, DN/e- (> 0)
  double offset_dn = 16.0;          // y_off, DN (>= 0)
  uint64_t seed = 0;
  void validate() const;            // std::invalid_argument naming the field
};

// The frame's pixels: full-image columns [x0, x0 + width) of rows [y0, y0 + height), full_width columns in all.
struct ApsRegion {
  int32_t full_width, x0, y0, width, height;
};

inline constexpr int32_t kApsBlocks = 64;                 // draws per row and stream (j < 64)
inline constexpr int64_t kApsTimeBias = int64_t{1} << 56;  // t_soe in [-2^56, 3 2^56)
inline constexpr int64_t kApsMaxSide = 65535;              // full-image width and rows (as SensorModel)

// The purpose.h counter layout, as SensorModel refuses it: camera < 256 (rng::draw keeps 8 bits of it) and a full
// image of at most 65535 x 65535 (pixel y W + x < 2^32), else distinct cameras or pixels would share draws.
// std::invalid_argument otherwise; full_rows is the last row used + 1 (aps_frame) or the image height.
void check_layout(uint32_t camera, int64_t full_width, int64_t full_rows);

// (t_soe + 2^56) 64 + j; std::invalid_argument outside the range above or for j >= 64.
uint64_t aps_index(int64_t t_soe_ns, uint32_t j);

// p(x) (relative) and d(x) (e-) of a pixel of the full image.
struct ApsFixedPattern {
  double p, d;
};
ApsFixedPattern aps_fixed_pattern(const ApsParams&, uint32_t camera, uint32_t pixel);

// A Poisson count of mean mu >= 0 (finite) from the kApsShot draws of (pixel, t_soe); PTRS tries at most
// max_blocks blocks (tests lower it), then *fallback is set and the normal approximation is returned.
double aps_poisson(double mu, const ApsParams&, uint32_t camera, uint32_t pixel, int64_t t_soe_ns,
                   bool* fallback, int32_t max_blocks = kApsBlocks);

// One frame region: mean_signal_e and dn are height x width (rows of the region, C order), row_soe_ns one t_soe per
// region row. Returns the number of normal fallbacks (FRM-APS-RNG). Refuses (check_layout) a camera >= 256 or a full
// width or y0 + height above 65535.
int64_t aps_frame(const ApsParams&, uint32_t camera, const ApsRegion&, const double* mean_signal_e,
                  const int64_t* row_soe_ns, double exposure_s, uint16_t* dn);

}  // namespace balla::sensor
