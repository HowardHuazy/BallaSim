// The Blender motion session's clock (docs/design/phase3/blender.md §4.2): one int64-ns clock tied exactly to the
// frames Blender evaluates. Only the engine computes it; the add-on records the raw time settings and never converts
// a frame to ns.
//
// Provenance (PHASE3.md §0 rule 4): re-derived from the Blender manual and Python API documentation
// (RenderSettings.fps, fps_base, frame_map_old, frame_map_new; Scene.frame_set(frame, subframe) and
// frame_current_final) and from the headless oracles AX exp02/exp10, MF e1/e8, DS d1 and RI2 r2_negframe_*
// (blender.md §2.2-§2.3): Blender evaluates the float32 frame c' = f32(f32(F) + s) and reports
// f32(c' * framelen) with framelen = f32(frame_map_old) / f32(frame_map_new); fps_base is a float32. No Blender source
// was used.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/time/time.h"
#include "scene/tree.h"

namespace balla::scene::motion {

// A refusal or protocol failure of the motion session, with its blender.md §6.1 code (BRG-TIME, BRG-REPLAY,
// MOT-ALIAS, VIS-ANIM, GEO-SCALEANIM, ...).
struct MotionError : std::runtime_error {
  MotionError(std::string c, const std::string& message) : std::runtime_error(c + ": " + message), code(std::move(c)) {}
  std::string code;
};

// The exchange's raw time settings (blender.md §3.4 `time`, `span` and the motion-blur fields of `render`).
struct TimeSettings {
  int32_t fps = 24;
  uint32_t fps_base_bits = 0x3f800000u;  // float32 bits of fps_base
  int32_t frame_map_old = 100, frame_map_new = 100;
  int32_t frame_start = 1, frame_end = 250;  // the add-on's span; frame_start is F_ref (t = 0)
  bool motion_blur = false;
  float motion_blur_shutter = 0.5f;
  std::string motion_blur_position = "CENTER";  // START | CENTER | END
  bool fps_base_snap = false;                   // scene.blender.fps_base_snap: fps_base -> p/q, q in {1, 1000, 1001}
};

class TimeMap {
 public:
  explicit TimeMap(const TimeSettings& s);

  // What Blender evaluates for frame_set(F, subframe=s): f32(f32(F) + s).
  static float realized(int32_t F, float s);
  // The frame_current_final Blender reports for a realized c: f32(c * framelen).
  float reported(float c) const;
  // t_ref + round_half_even((c - F_ref) 1e9 B_num / (fps B_den)), exact (B = B_num / B_den is fps_base's exact value).
  int64_t to_ns(float c) const;
  // The exact time of the frame F_ref + num / den (den > 0), rounded half to even.
  int64_t frames_to_ns(int64_t num, int64_t den) const;
  // The time of an output frame given as a double (a declared key, blender.md §4.7.4): exact for the double
  // f - F_ref (so frame_to_ns(c) == to_ns(c) for every float32 c), rounded half to even; below 2^-85 frames of
  // precision the fraction is rounded first.
  int64_t frame_to_ns(double frame) const;
  // framelen = f32(frame_map_old) / f32(frame_map_new): Blender's scene time is the output frame times framelen.
  double framelen() const { return static_cast<double>(framelen_); }
  // The request for a target time: c = the float32 nearest F_ref + (t - t_ref) fps B_den / (1e9 B_num) (ties to even),
  // F = floor(c), s = f32(c - F), with s in {0} u [2^-30, 1). The realized time may differ (c in (-1, 0)); samples
  // are always stamped with to_ns(realized(F, s)).
  void request(int64_t t_ns, int32_t* F, float* s) const;
  // q(F) = 1/2 ulp32(max(|F|, 1)) P, Blender's time resolution at frame F (ns).
  double resolution_ns(int32_t F) const;
  // (floor(t(frame_end) / W) + 1) W (W = time::kWindowNs, the engine's window): the run's default span [0, t_end)
  // holds frame_end.
  int64_t t_end_ns() const;
  double period_ns() const;  // P in ns

  int32_t fps() const { return fps_; }
  int64_t base_num() const { return base_num_; }
  int64_t base_den() const { return base_den_; }
  int32_t frame_ref() const { return frame_ref_; }
  int64_t t_ref_ns() const { return t_ref_ns_; }
  uint32_t framelen_bits() const;
  int32_t frame_start() const { return frame_start_; }
  int32_t frame_end() const { return frame_end_; }

  // The timeline record (blender.md §4.2): fps, fps_base_bits, base_num, base_den, frame_ref, t_ref_ns, framelen_bits,
  // frame_start, frame_end, t_frame_end_ns, bundle_span, run_span, run_span_extends_ns, q_max_ns, motion_blur.
  Tree record() const;
  // The inverse of record(); refuses (std::invalid_argument) a field of the wrong kind, frame_ref != frame_start,
  // t_ref_ns != 0, a base_num / base_den other than the one fps_base_bits gives with or without fps_base_snap, and
  // derived fields (t_frame_end_ns, bundle_span, run_span, run_span_extends_ns, q_max_ns) that disagree with the
  // inputs. framelen_bits is taken as recorded (frame_map_old / frame_map_new are not in the record).
  static TimeMap from_record(const Tree& record);

 private:
  TimeMap() = default;
  void check() const;
  int32_t fps_ = 24;
  uint32_t fps_base_bits_ = 0x3f800000u;
  int64_t base_num_ = 1, base_den_ = 1;
  int32_t frame_ref_ = 1;
  int64_t t_ref_ns_ = 0;
  float framelen_ = 1.0f;
  int32_t frame_start_ = 1, frame_end_ = 1;
  bool motion_blur_ = false;
  float shutter_ = 0.5f;
  std::string position_ = "CENTER";
};

}  // namespace balla::scene::motion
