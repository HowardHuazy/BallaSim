// Cycles timing of the Blender-rendered frames (docs/design/phase3/frames.md §4.2, §4.3): which frame, subframe,
// shutter and rolling duration make Cycles expose a raster's rows over the schedule's row windows, the times Cycles
// then realizes (the schedule snaps to them), the motion-step node times, the motion steps each moving body needs,
// and the sub-render and row-band rules of moving lights.
//
// Provenance (PHASE3.md §0 rule 4): the shutter-table constants, the rolling-shutter time formula, the motion-step
// node times and the quaternion interpolation rule are derived from Cycles v4.2.1 (Apache-2.0, THIRD_PARTY_NOTICES.md):
// intern/cycles/scene/camera.cpp:491-501 (the inverted 256-entry shutter CDF, read by linear lookup: a flat curve gives
// u ~ U[a, b]), kernel/camera/camera.h ~359-391 (ray time 0.5 without motion blur; rolling TOP: time = 1 - y / height
// with y the integer raster row from the bottom, t = (u - 1/2) d + (time - 1/2)(1 - d) + 1/2, t = time for d = 0),
// blender/object.cpp:688-690, 743-760 (node times as float32 sums, then frame_set(floorf(time), time - floor)),
// blender/util.h:569-597 ((2 << (steps - 1)) + 1 transforms), scene/object.cpp and scene/camera.cpp motion_time
// (2 step / (n - 1) - 1) and util/transform.h ~407-431 (quat_interpolate). They were checked against the headless
// oracle E1 (frames.md §2.4; CT e1_out/e1_results.json) and RV2's rolling snap (t_rolling_snap.json). The frame
// clock itself is BL-6's scene::motion::TimeMap (re-derived from the Blender manual, API docs and oracles).
//
// Symbols (§4.2): a = 0.5 * 255 / 65536, b = 255.5 * 255 / 65536 (b - a = 65025 / 65536), delta_c = 1/512;
// P_ns = 1e9 B_num / (fps B_den), F_r = 1e9 / P_ns; T' a sub-render's exposure; the Blender raster has H_b rows
// at Delta_b apart, its row 0 (top) is the first-read row and starts exposing at t_b0.
//   global, T' > 0:  S = F_r T' / (1e9 (b - a)), d = 1, rolling NONE, t_c = t_b0 + floor(T'/2) + delta_t
//   global, T' = 0:  motion blur off (Cycles: ray time 0.5), c = request(t_b0), realized centre to_ns(c)
//   rolling (TOP):   S = F_r (T' / (b - a) + H_b Delta_b) / 1e9, d = F_r T' / (1e9 (b - a) f32(S)) capped at 1
//                    (0 for T' = 0), t_c = t_b0 + T'/2 + Delta_b (H_b/2 - 1) + delta_t; motion blur stays on for T' = 0
// with delta_t = round(delta_c T' / (b - a)) = delta_c d S P_ns: raster row i's centre lies at
// (-delta_c d + ((i + 1)/H_b - 1/2)(1 - d)) S frames from c and S (1 - d) P_ns = H_b Delta_b, so row i is centred at
// t_b0 + i Delta_b + T'/2. (F, s) = TimeMap::request(round(t_c)), c = f32(f32(F) + s); Cycles realizes the centre
// t_hat = to_ns(c) - round(delta_c f32(d) f32(S) P_ns) (d = 1 global), and the raster's windows move by
// t_hat - (round(t_c) - delta_t), the snap shift. The realized row exposure is (b - a) f32(d) f32(S) P_ns.
#pragma once

#include <stdint.h>

#include <array>
#include <functional>
#include <vector>

#include "frames/timing.h"
#include "render/bundle.h"
#include "scene/motion/time_map.h"

namespace balla::frames {

// The flat shutter table's bounds and centre offset (Cycles scene/camera.cpp:491-501): u ~ U[a, b], centred
// delta_c = 1/2 - (a + b)/2 = 1/512 early.
inline constexpr double kShutterA = 0.5 * 255 / 65536, kShutterB = 255.5 * 255 / 65536, kDeltaC = 1.0 / 512;
inline constexpr double kShutterWidth = kShutterB - kShutterA;  // 65025 / 65536 = 0.99220276

// A Blender raster: H_b rows, Delta_b ns apart. Its row 0 starts exposing at t_b0 = sub_first_soe_ns +
// first_row_offset_ns, where sub_first_soe_ns (cycles_timing) is the start of exposure of the frame's first-read row
// (read-order row 0) plus the sub-render's offset, and first_row_offset_ns holds a band's row0 Delta and route S's
// margin offset.
struct Raster {
  int32_t rows;
  double row_dt_ns;
  double first_row_offset_ns;
};

// Read-order rows [row0, row0 + rows) of the full image (row0 = 0, rows = H: the whole frame).
struct RowBand {
  int32_t row0, rows;
};

// The Blender raster of a frame or of a row band (§4.2). Routes F and O (s = 1, margin_rows = 0): the band's own rows
// at Delta, first_row_offset_ns = row0 Delta (t_b0 = t_soe(read row row0)). Route S (identity map): s (rows + 2 m)
// sub-rows at Delta / s with first_row_offset_ns = row0 Delta - (s m + (s - 1)/2) Delta / s, so the s sub-rows of each
// image row average to its window. In every route Blender's raster row 0 is the band's first-read row (bottom-up
// readout renders flipped, §4.2). A band with rows < 0 runs from row0 to the last read row (the default: the whole
// frame).
Raster blender_raster(const RowTiming&, RowBand band = {0, -1}, int32_t s = 1, int32_t margin_rows = 0);

// n equal bands of H read rows: band b holds [floor(b H / n), floor((b + 1) H / n)); n in [1, H].
std::vector<RowBand> row_bands(int32_t height, int32_t n_band);

// Sub-render j of n_sub (§4.2): offset floor(j T / n_sub), length T'_j = floor((j + 1) T / n_sub) - offset.
struct SubWindow {
  int64_t offset_ns, length_ns;
};
SubWindow sub_window(int64_t exposure_ns, int32_t n_sub, int32_t j);

struct CyclesTiming {
  int32_t frame = 0;           // scene.frame_set(frame, subframe)
  float subframe = 0.0f;
  float shutter_frames = 0.0f;    // render.motion_blur_shutter S (0 with motion blur off)
  float rolling_duration = 1.0f;  // cycles.rolling_shutter_duration d; 1 for a global shutter
  bool rolling = false, motion_blur = false;  // rolling_shutter_type TOP / NONE; render.use_motion_blur
  float expected_c = 0.0f, expected_ctime = 0.0f;  // f32(f32(frame) + subframe) and frame_current_final
  int64_t realized_center_ns = 0;  // t_hat (rolling: rows i are centred at t_hat + (i + 1 - H_b/2) Delta_b)
  int64_t snap_shift_ns = 0;       // the raster's row windows move by this much
  double realized_row_exposure_ns = 0.0;  // (b - a) f32(d) f32(S) P_ns (0 with motion blur off or d = 0)
  double q_ns = 0.0;               // q(frame), Blender's time resolution there
  std::vector<Warning> warnings;   // FRM-TIME-RES: q > 1 us, or q > 0.01 T' for T' > 0
};

// The Cycles timing of one sub-render (or band) exposing sub_exposure_ns (T' >= 0) on `raster`, under the shutter of
// `timing` (a rolling shutter whose raster has H_b Delta_b = 0 is a global one). sub_first_soe_ns is the start of
// exposure of the FRAME's first-read row (read-order row 0) plus the sub-render's offset floor(j T / n_sub), the same
// for every band: a band's row0 Delta (and route S's margin offset) is in raster.first_row_offset_ns. Throws
// std::invalid_argument on a bad raster, std::range_error (TimeMap) when a time leaves Blender's frame range.
CyclesTiming cycles_timing(const scene::motion::TimeMap&, const RowTiming&, const Raster&, int64_t sub_first_soe_ns,
                           int64_t sub_exposure_ns);

// The 2^steps + 1 frames Blender evaluates for an object of `steps` motion steps (1..7) when the pass is centred at
// c with shutter S: relative times r_i = f32(f32(2 i) / f32(n - 1)) - 1, time_i = f32(c + f32(f32(r_i S) * 0.5f)),
// then frame_set(floorf(time_i), time_i - floor) realizes f32(f32(floor) + f32(time_i - floor)) (TimeMap::realized).
// Ascending for S >= 0; with steps nested (node i of steps n is node 2i of steps n + 1).
std::vector<float> node_times(float c, float shutter_frames, int32_t steps);

// Rotation interpolation between motion nodes (Cycles util/transform.h ~407-431, quat_interpolate): kGpuRt is the
// normalized lerp of OptiX and MetalRT; kCpu (CPU and CUDA) is the normalized lerp when dot(q1, q2) > 0.9995 and
// slerp below; kWorst takes the larger error of the two, since a pass may fall back to another device (§3.3).
enum class QuatInterp : uint8_t { kCpu, kGpuRt, kWorst };

// The points whose image motion sets a body's motion steps (§4.3).
struct MotionProbe {
  const render::SceneBundle* bundle = nullptr;
  uint16_t cam = 0;
  int32_t body = -1;  // -1: the camera rig
  std::vector<std::array<double, 3>> points;  // body: its 8 mesh AABB corners, body frame; rig: the image corners
                                              // and centre at z_near, world (empty: nothing to probe, 1 step)
  double z_near_m = 0.0;                      // smallest finite RefFrame.depth at frame_mid (NaN: none)
  // The point of the body frame whose world path Cycles interpolates linearly: the Blender object's origin. 0 (the
  // body origin) unless the plan builder knows the exchange's object origin (BL §4.1's recentring); the rig's is
  // the camera centre.
  std::array<double, 3> pivot{0.0, 0.0, 0.0};
};
// The calls below build the rig's and the body's splines only over the segments that their times meet (the window
// and the outer node times), so their cost does not grow with the trajectories' length.
// depth: the ROI's centre-ray camera z at t_mid_ns (RefFrame.depth; NaN where nothing is hit).
MotionProbe motion_probe(const render::SceneBundle&, uint16_t cam, int32_t body, const std::vector<float>& depth,
                         int64_t t_mid_ns);

// The chord error of `steps` motion steps (§4.3): the largest |pi(T_cam(t)^-1 T_b(t) X) - pi(T~_cam(t)^-1 T~_b(t) X)|
// in px over the probe points X and 16 probes per node interval inside [t_a, t_b], where T~ interpolates the poses at
// the realized node times (node_times, TimeMap::to_ns) linearly in time, translation of the pivot linearly and
// rotation per QuatInterp. A point whose true or interpolated position lies outside the camera model's injective domain
// at a probe is not dropped while some probe point is in front of the camera there: its error is taken as
// f_max |X~_cam - X_cam| / max(z, z_near_m) (z the smaller camera depth of the two, z_near_m when finite; no term when
// that depth is <= 0), the first-order image motion near the image centre of a point at that depth. So a body whose
// AABB corners all leave the image still measures the motion of its visible interior. A probe wholly behind the
// camera adds nothing. Throws FramesError FRM-SPAN when a pose time leaves a trajectory's span.
double motion_chord_px(const MotionProbe&, int64_t t_a, int64_t t_b, float c, float shutter_frames,
                       const scene::motion::TimeMap&, int32_t steps, QuatInterp);

struct MotionStepsInfo {
  double chord_px = 0.0;     // the chord error at the returned steps (at 7 when it returns 0)
  double time_res_px = 0.0;  // v_b 2 ulp32(c) P_ns: the image speed times Blender's node-time resolution
  double budget_px = 0.0;    // the chord's budget, max(tol_px - time_res_px, tol_px / 2)
  double speed_px_s = 0.0;   // v_b, the probe points' largest image speed over [t_a, t_b] (out-of-domain points
                             // as in motion_chord_px)
  std::vector<Warning> warnings;  // FRM-TIME-RES when time_res_px > tol_px / 2
};
// The smallest steps n in [1, 7] with chord(n) <= max(tol_px - v_b 2 ulp32(c) P_ns, tol_px / 2), or 0 when 7 is not
// enough (the caller doubles n_sub, FRM-MOTION at 64). ulp32 is taken at max(|c| + S/2, 1), the largest node. The
// time-resolution term does not depend on n, and sub-renders keep its ulp32 (every sub-render's c has about the same
// magnitude), so it never makes motion_steps return 0: up to tol_px / 2 it comes out of the chord's budget (chord +
// term <= tol_px), above it FRM-TIME-RES reports it and the chord keeps tol_px / 2. A pass without motion blur
// (S = 0) or a probe without points needs 1.
int32_t motion_steps(const MotionProbe&, int64_t t_a, int64_t t_b, float c, float shutter_frames,
                     const scene::motion::TimeMap&, double tol_px, QuatInterp, MotionStepsInfo* info = nullptr);

// ---- Sub-renders and row bands (§4.3) ----
inline constexpr int32_t kMaxSubRenders = 64, kMaxBands = 64;

// Moving lights: Cycles freezes a light for the whole pass, so with rho = MotionFacts::light_rate_per_s (1/s) a pass
// may span at most factor_tol / rho. Global: n_sub = ceil(rho T 1e-9 / tol), one band. Rolling (a pass also spans
// the readout R): n_sub = ceil(2 rho T 1e-9 / tol) and n_band = ceil(2 rho R 1e-9 / tol). n_sub is capped at 64 and
// n_band at min(H, 64); a cap warns FRM-LIGHT-MOTION with the bound that remains. bound = rho (T'_max + R_band) 1e-9
// is the longest pass: T'_max = ceil(T / n_sub) and the readout of the tallest band, R_band = (rows_max - 1) Delta
// (0 for a global shutter).
struct LightSplit {
  int32_t n_sub = 1, n_band = 1;
  double bound = 0.0;
  std::vector<Warning> warnings;
};
LightSplit light_split(double rho_per_s, const RowTiming&, double factor_tol);

// n_sub of one frame (§4.3): from n_sub0 (the light rule), doubled (capped at 64) while steps_ok(n_sub) is false (a
// moving body needs more than 7 motion steps in some sub-render) or fact_p99(n_sub) > factor_tol (the waveform-motion
// factorization test, ExposureReference with RefOptions::n_sub); at 64 the failing tests warn FRM-MOTION and
// FRM-FACTOR. Either callback may be empty (that test passes).
struct SubRenderChoice {
  int32_t n_sub = 1;
  double fact_p99 = 0.0;  // at the chosen n_sub (0 without a factor test)
  std::vector<Warning> warnings;
};
SubRenderChoice choose_n_sub(int32_t n_sub0, double factor_tol, const std::function<bool(int32_t)>& steps_ok,
                             const std::function<double(int32_t)>& fact_p99);

}  // namespace balla::frames
