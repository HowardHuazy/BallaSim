// The exposure reference (docs/design/phase3/frames.md §4.4, §4.3, §4.9.4): BallaSim's own frame, the exposure
// integral of the run's N-sample irradiance over each read-row window, with the waveforms applied exactly.
//
// For frame k, ROI pixel x in read-row r and samples j = 0 .. K - 1 (K <= T; K = 1 and m = m(t_soe) for T = 0):
//   t_{r,j}     = t_soe(k, r) + floor((2j + 1) T / (2K))
//   mbar_{g,j}  = mean of m_g over [t_soe(k, r) + floor(j T / K), t_soe(k, r) + floor((j + 1) T / K)]  (WaveformPrefix)
//   F_g(x)      = sum_j (1/K) mbar_{g,j} I_g(x, t_{r,j}),   F(x) = sum_g F_g(x)
// The renderer's irradiance I_g is a staircase in time at a moving edge (each of its N sub-samples switches once),
// so the midpoint rule converges as O(1/K) there. K0 is 1 when nothing moves and otherwise
// min(T, k_max, max(2, ceil(1.25 D / h_eff))), h_eff = h min(1, sqrt(D / 4 px)), D = T max |flow|. An edge-mask
// probe then checks it: the mask M holds the pixels whose ln(sum_g mbar_g I_g + eps) (mbar_g the row's exposure mean,
// so flicker alone does not fill it) varies over the K0 samples by more than probe_tol, plus the grid
// x = y = 0 (mod 4) of the full image; M is rendered at 3 K0 (sample 3j + 1 is sample j, so 2 K0 are new), and
// e = 1.5 |ln(F_3K0 + eps) - ln(F_K0 + eps)| estimates F_K0's error. K0 is accepted when p99.9 of e over M is at most
// probe_tol; otherwise K0 triples (every pixel, reusing its samples) and the probe repeats, up to min(T, k_max)
// (FRM-REF-K when the probe still fails there). F uses 3 K0 on M (at most T) and K0 elsewhere. eps = 1e-3 p99(F);
// it also keeps e finite on dark pixels. (With 3 K0 > T the mask level is T and e = T/(T - K0) |...|.)
//
// Factorization (§4.3): when something moves and a waveform is not constant, K is raised to a multiple of n_sub with
// >= 4 samples per sub-window (FRM-FACTOR when the cap prevents it), and e_fact(x) = max over sub-windows s of
//   |sum_g (mean_s(mbar_g I_g) - mean_s(mbar_g) mean_s(I_g))| / F(x),
// over each pixel's own samples; fact_p99 is its p99 over the ROI. It is 0 by construction otherwise. Sample j belongs
// to sub-window floor((2j + 1) n_sub / (2K)), the one that holds its midpoint. The probe's levels stay multiples of
// n_sub where the caps allow: the all-pixel levels' cap is k_max rounded down to a multiple of n_sub when that
// exceeds K0, and a mask level capped by T is rounded down to one when that still exceeds K; with K a multiple of
// n_sub the samples' windows tile the sub-windows exactly.
//
// Memory: each pixel's samples are summarized (F_g, the mask quantity's extremes, e_fact) as soon as its block is
// rendered, in batches of at most kRequestChunk samples, so beyond one batch a level holds O(npix G) summaries and
// its waveform means mbar (ROI H x K x G doubles). A 3K level takes a mask level's pixels whole when the K agree; it
// reuses a level's samples (3j + 1 <- j) only while every kept level fits in RefOptions::reuse_budget_mb, and above
// it renders them again, with the same result (the renderer's determinism, RENDER.md: a sample does not depend on
// its batch). reuse_budget_mb bounds only the kept samples. Quantiles are nearest-rank (the ceil(q n)-th smallest
// finite value). Rendering follows set_window(frame_soe), so the reference is a pure function of (bundle, renderer
// options, schedule, k, the options other than reuse_budget_mb).
#pragma once

#include <stdint.h>

#include <memory>
#include <span>
#include <vector>

#include "frames/timing.h"
#include "render/renderer.h"
#include "sensor/waveform.h"

namespace balla::frames {

// Motion over [frame_soe, frame_eoe] of one frame (§4.3, §4.4).
struct MotionFacts {
  double d_max_px = 0.0;    // T max |flow| over the ROI at frame_soe, frame_mid, frame_eoe and the breakpoints
  bool any_motion = false;  // rigorous (traj/bounds.h): the camera rig or a body with geometry or a light moves
  bool light_motion = false;      // a positional or directional light rides a moving body
  double light_rate_per_s = 0.0;  // rho = max_l rate_l, 1/s (below)
  std::vector<int64_t> breakpoints;  // trajectory piece boundaries inside the span (C1 v2 pieces; none in C1 v1)
};
// rate_l of a light on a moving body, from the trajectory's speed bounds v, omega over the frame's segments: a point
// or spot light at offset r_l in its body frame moves at v_l <= v + omega |r_l|, and rate_l = (v_l / d_l)(2 + 3),
// the DESIGN §7.5 bound (v_l / d_l)(2 + min(|tan theta_i|, 3)) at its cap, since the ground truth has no normals;
// d_l is the light's least distance at frame_mid to the ROI's visible points (centre-ray depth). A directional light
// turns at <= omega, so rate_l = 3 omega (|d ln cos theta_i / dt| <= omega |tan theta_i|, capped). Spot cones are not
// bounded: the probe sees them.

struct RefOptions {
  double h_px = 0.25, probe_tol = 0.005;  // frames.ref.h_px, frames.ref.probe_tol
  int32_t k_max = 256;                    // frames.ref.k_max, in [2, 4096]
  int32_t k_fixed = 0;                    // > 0: min(k_fixed, T) samples everywhere, no probe (<= kMaxSamples)
  int32_t n_sub = 1;                      // sub-renders of the factorization test, in [1, 64]
  double reuse_budget_mb = 1024.0;        // >= 0: host memory for samples kept for reuse; output-invariant
};

struct RefFrame {
  std::vector<float> per_group, mono;  // F_g (ROI H x W x G) and F (H x W), renderer units, waveform-weighted
  std::vector<double> m_mean;          // ROI H x G: the exact waveform mean over each row's window (§4.5)
  std::vector<float> depth;            // H x W centre-ray camera z at frame_mid (NaN: no geometry)
  std::vector<uint32_t> instance;      // H x W at frame_mid
  int32_t k_used = 0, k_mask = 0;      // K0 everywhere and K on the probe mask
  int64_t mask_px = 0;                 // pixels in the probe mask (0 without a probe)
  int64_t rendered = 0;                // irradiance samples rendered (requests), reuse excluded
  double d_max_px = 0.0;
  float fact_p99 = 0.0f, probe_p999 = 0.0f, probe_max = 0.0f;
  std::vector<Warning> warnings;       // FRM-REF-K, FRM-FACTOR
};

class ExposureReference {
 public:
  static constexpr size_t kRequestChunk = size_t{1} << 22;
  static constexpr int32_t kMaxSamples = 3 * 4096;  // the probe mask's K at k_max = 4096; also k_fixed's bound

  // The reference of camera `cam` of the renderer's bundle (Renderer::bundle()), one waveform per light group, over
  // the ROI [x0, x1) x [y0, y1) of the full image. The renderer is used, not owned.
  ExposureReference(render::Renderer&, uint16_t cam, std::span<const sensor::Waveform>, Roi);
  ~ExposureReference();
  ExposureReference(const ExposureReference&) = delete;
  ExposureReference& operator=(const ExposureReference&) = delete;

  MotionFacts motion(const Schedule&, int32_t k) const;
  RefFrame frame(const Schedule&, int32_t k, const RefOptions&);
  // H x W x G: V_g = R_g / max(R_g^noshadow, 1e-12) at t_ns (§4.9.4), R^noshadow from a copy of the bundle with every
  // casts_shadows false, rendered with the same options; built on the first call.
  std::vector<float> visibility(int64_t t_ns);

  const Roi& roi() const { return roi_; }
  int32_t num_groups() const { return groups_; }

 private:
  struct Impl;
  MotionFacts motion_facts(const Schedule&, int32_t k, const std::vector<render::GtSample>* gt_mid) const;
  std::vector<render::GtSample> centre_gt(int64_t t_ns) const;

  render::Renderer& renderer_;
  uint16_t cam_;
  Roi roi_;
  int32_t groups_;
  std::vector<sensor::Waveform> waves_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace balla::frames
