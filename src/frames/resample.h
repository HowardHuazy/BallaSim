// Route S's resampler (docs/design/phase3/frames.md §4.7.2, §3.8): Blender renders a square pinhole at s x
// supersampling with overscan (BOX film, so each source pixel is the box integral of its cell), and this operator
// turns it into our camera's pixel response, box(a) ⊛ Gaussian(sigma) on our lens model, as a fixed linear map.
//
// Source raster. Source pixel (i, j) has its centre at integers of the source camera (fx_s, fy_s, cx_s, cy_s), and
// x_n = (q_x - cx_s) / fx_s, y_n = (q_y - cy_s) / fy_s are the normalized coordinates it sees. overscan_pinhole
// builds the design's source: fx_s = fy_s = s fx, cell edges on our pixel edges, covering the preimage of the
// support rectangle (each pixel centre +- support_px) plus margin_px, with an integer margin of our pixels per side.
//
// Quadrature rule (any lens). Our pixel k's response is sampled at p = k + a beta + sigma_r gamma: `strata` box
// strata beta per axis (one when a = 0) and 3 Gauss-Hermite nodes gamma per axis (+-sqrt 3 with 1/6, 0 with 2/3; one
// node when sigma_r = 0); ray = our unprojection of p (camera::unproject, with the injective domain the renderer
// builds), q = the source projection of the ray, read with bilinear taps. The source cell (1/12) and the bilinear tent
// (1/6) add 1/4 source px^2 of variance per source axis, so the Gaussian nodes take the rest: sigma_r^2 = sigma^2 -
// (1/s)^2/4 (Blur::kNominal, §4.7.2 as E3b/F3 measured it), or by default (Blur::kLocal) the covariance
// sigma^2 I - (1/4) A A^T with A = J^-1 and J the pixel's d(source)/d(ours) (central differences at +-h, h =
// min(0.5, support_px), so the probes stay in the support rectangle; a pixel whose probes lack a preimage takes the
// nominal rule), which is the nominal rule where J = s I and follows the local magnification elsewhere (a
// Kannala-Brandt corner at 38 degrees: 0.0041 against 0.0172 nominal). The nodes are then L gamma with L its
// symmetric square root. Variance that cannot be removed (sigma^2 < (1/s)^2/4 at the local magnification) is clamped
// and reported (excess_variance_px2, the largest over the pixels with a preimage; FRM-RESAMPLE-BLUR). A pixel with a
// quadrature point outside our lens domain has no preimage: NaN. A Kannala-Brandt ray at or beyond 90 degrees has no
// pinhole image (FRM-KB-WIDE).
//
// Cells rule (identity map: zero distortion, fx = fy, source cells subdividing our pixels). Per axis the box
// [k - a/2, k + a/2] is cut at the cell edges: the piece in cell m keeps its fraction b_k(m) of the box and spreads
// from its centroid y_m, W_k(i) = sum_m b_k(m) G(u_i - y_m) / Z_m, with G a Gaussian of sigma sampled at the cell
// centres u_i within truncation_sigmas sigma of y_m (at least the nearest cell, or the two at a tie; a delta for
// sigma = 0) and Z_m its sum. A whole cell has y_m = u_m, so with the box's edges on cell edges (a s an integer of
// s's parity) this is the design's indicator rule and sigma = 0 gives the exact box average. A cut piece adds at most
// b_k(m) / (12 s^2) px^2 of variance per axis (its cell's extent beyond its own), so the excess stays below
// 1/(12 s^2), is continuous in a and vanishes as the edges reach cell edges. A point (a = 0) is one piece at the pixel
// centre (for even s a cell edge, shared by the nearest two cells). A box that is not whole cells raises
// FRM-APERTURE-QUANT when sigma < 1/s.
//
// Storage. With zero distortion both rules are separable (per-axis affine map, tensor-product points, bilinear
// taps): two 1-D tables, applied along rows, then down columns. Otherwise CSR: per ROI pixel its taps in source order
// (row-major), float weights. Weights are accumulated in fp64 per pixel, normalized to sum 1 and stored as float
// with the rounding residual added to the largest weight. apply() sums each output in fp64 in tap order;
// apply_tiles() adds one tile's rows of source (s = 8 renders in tiles, §4.7.2) to `out` the same way, so tiles in
// any order give apply() within float rounding (one tile: bitwise). Every ROI pixel's weights are a function of the
// pixel alone, so an ROI operator equals the full operator's rows bitwise; the build does not depend on its threads.
#pragma once

#include <stdint.h>

#include <span>
#include <utility>
#include <vector>

#include "core/camera/kannala_brandt.h"
#include "core/camera/radtan.h"
#include "frames/timing.h"
#include "render/bundle.h"

namespace balla::frames {

enum class Rule : uint8_t { kCells, kQuadrature };
// Quadrature: sigma_r from each pixel's local magnification (kLocal), or from the nominal cell 1/s everywhere
// (kNominal: §4.7.2's sigma_r^2 = sigma^2 - (1/s)^2/4, as E3b/F3 measured it).
enum class Blur : uint8_t { kLocal, kNominal };

// kEquidistant (a PANO source for wide Kannala-Brandt cameras) is FR-3b; it is refused (FRM-KB-WIDE).
// margin_rows: our pixel rows of overscan above the covered region's first row (§4.2's m; symmetric for the
// identity map, so H_b = s (H + 2m)).
struct SourceCamera {
  enum class Kind : uint8_t { kPinhole, kEquidistant } kind;
  double fx, fy, cx, cy;
  int32_t width, height, s, margin_rows;
};
struct ResampleSpec {
  double aperture_px, sigma_px, truncation_sigmas;  // sensor.aperture_px, optics.blur_sigma_px, psf truncation
  Rule rule;
  int32_t strata = 16;        // box strata per axis (quadrature), [1, 64]
  Blur blur = Blur::kLocal;   // quadrature
};
struct RawImage {  // rows top-down, C order, h x w x c
  int32_t h, w, c;
  const float* data;
};

// Our lens model as the renderer builds it (scene_data.cpp): the camera's model with its injective domain over the
// sub-sample support (render::support_px's operation order: 0.5 a + n sigma).
struct LensModel {
  render::CameraModelKind kind;
  camera::RadtanCamera<double> radtan;
  camera::KannalaBrandtCamera<double> kb;
  int32_t width, height;
  double support_px;
};
LensModel lens_model(const render::Camera&, double support_px);
double resample_support_px(const ResampleSpec&);

class ResampleOperator {
 public:
  // The operator of the ROI [x0, x1) x [y0, y1) of our image; threads <= 0: the hardware's. FramesError FRM-VALUE for
  // an empty or out-of-image ROI, FRM-KB-WIDE as above; std::invalid_argument for other bad values or a source raster
  // that does not cover a tap.
  static ResampleOperator build(const render::Camera& ours, const SourceCamera&, const ResampleSpec&, Roi,
                                int threads = 0);
  static ResampleOperator build(const LensModel& ours, const SourceCamera&, const ResampleSpec&, Roi,
                                int threads = 0);

  // src: the source raster, height x width x channels; out: ROI H x W x channels (overwritten).
  void apply(const float* src, int32_t channels, float* out) const;
  // tile: source rows [y0, y0 + tile.h) at the raster's full width; adds their contribution to `out`, which the
  // caller zeroes before the first tile. Pixels without a preimage are set to NaN by every call.
  void apply_tiles(const RawImage& tile, int32_t y0, int32_t channels, float* out) const;

  size_t nnz() const;                    // non-zero weights of the operator as a matrix (separable: the products)
  double excess_variance_px2() const { return excess_; }  // quadrature; 0 for cells
  bool separable() const { return separable_; }
  const Roi& roi() const { return roi_; }
  int32_t source_width() const { return src_w_; }
  int32_t source_height() const { return src_h_; }
  Roi source_rect() const { return src_rect_; }  // the source pixels read, [x0, x1) x [y0, y1) (the plan's border)
  int64_t nan_pixels() const;
  std::span<const Warning> warnings() const { return warnings_; }  // FRM-RESAMPLE-BLUR, FRM-APERTURE-QUANT
  size_t bytes() const;                                            // memory of the stored weights
  // Taps of ROI pixel (x, y) (ROI coordinates): (y_s width + x_s, weight) in source order; empty without a preimage.
  std::vector<std::pair<int64_t, double>> weights(int32_t x, int32_t y) const;

 private:
  ResampleOperator() = default;
  struct Taps1 {  // per output column or row: taps [ptr[k], ptr[k + 1]) of (index, weight)
    std::vector<int32_t> ptr, idx;
    std::vector<float> w;
  };
  void apply_rows(const float* rows, int32_t row0, int32_t nrows, int32_t channels, float* out, bool add) const;

  Roi roi_{};
  int32_t src_w_ = 0, src_h_ = 0;
  Roi src_rect_{};
  bool separable_ = false;
  double excess_ = 0.0;
  Taps1 tx_, ty_;                   // separable
  std::vector<int64_t> ptr_;        // CSR: per ROI pixel (row-major), into col_ and w_
  std::vector<uint32_t> col_;       // source index y_s width + x_s
  std::vector<float> w_;
  std::vector<int32_t> row_lo_, row_hi_;  // per ROI row: the source rows its taps read ([lo, hi]; lo > hi: none)
  std::vector<uint8_t> nan_;              // per ROI pixel: no preimage
  std::vector<Warning> warnings_;
};

// The design's source camera (§4.7.2): a square pinhole with fx_s = fy_s = s fx whose cells subdivide our pixels,
// covering the preimage of our image's support rectangle (or the ROI's when given) plus margin_px on every side.
// FRM-KB-WIDE for a Kannala-Brandt camera whose cover reaches 90 degrees or needs a raster side above 65536 px
// (Blender's resolution_x/resolution_y range, Python API RenderSettings); FRM-ROUTE for any other such raster;
// FRM-VALUE for an empty or out-of-image ROI.
SourceCamera overscan_pinhole(const render::Camera& ours, double support_px, int32_t s, double margin_px = 2.0,
                              const Roi* roi = nullptr);
SourceCamera overscan_pinhole(const LensModel& ours, int32_t s, double margin_px = 2.0, const Roi* roi = nullptr);

}  // namespace balla::frames
