// Frame schedules on the true clock and their sensor-clock stamps (docs/design/phase3/frames.md §3.8, §4.1).
//
// A schedule lists, per kept frame k, the anchor a_k: the start of exposure of the first-read row. Rows are those of
// the full image; read-row r starts at t_soe(k, r) = a_k + rho(r) Delta and ends T later, with rho(r) = r top-down,
// H - 1 - r bottom-up, Delta the rolling row delay (0 for a global shutter), R = (H - 1) Delta the readout span.
//   uniform: a_k = t0 + k P;
//   blender: one frame per Blender frame time t_ns(F_k), anchored as Blender's own exposure (BL §4.2): CENTER
//            a_k = t_ns(F_k) - floor((T + R)/2), START a_k = t_ns(F_k), END a_k = t_ns(F_k) - (T + R).
// A frame is kept only when [frame_soe, frame_eoe] lies inside the support, the bundle's [t_start_ns, t_end_ns];
// the others are listed (FRM-SPAN, warn), and a schedule with none left is refused (FRM-SPAN). The uniform timeline
// forms only the kept anchors (exact, no overflow), and refuses (FRM-SPAN) a count or t0 that would drop more than
// kMaxDroppedFrames candidates, so a large frames.count never lists billions of dropped frames.
#pragma once

#include <stdint.h>

#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sensor/readout.h"

namespace balla::frames {

// A refusal with its frames.md §6 code; the step exits 2 and records the code.
struct FramesError : std::invalid_argument {
  FramesError(std::string code_, const std::string& message) : std::invalid_argument(message), code(std::move(code_)) {}
  std::string code;
};

// A warning with its frames.md §6 code, recorded in the frames file.
struct Warning {
  std::string code, message;
};

enum class Shutter : uint8_t { kGlobal, kRolling };
enum class Readout : uint8_t { kTopDown, kBottomUp };
enum class Position : uint8_t { kStart, kCenter, kEnd };  // the blender timeline's anchor (§4.1)

// exposure_ns T >= 0 per row; row_dt_ns Delta >= 0 (rolling only); height H >= 1, the full image's rows.
struct RowTiming {
  int64_t exposure_ns, row_dt_ns;
  int32_t height;
  Shutter shutter;
  Readout readout;
};
struct Span {  // the bundle's [t_start_ns, t_end_ns]
  int64_t begin_ns, end_ns;
};
struct Roi {
  int32_t x0, y0, x1, y1;
};

// ---- Exposure default (§4.1) ----
// A Blender scene's frame rate, P = 1e9 base_num / (fps base_den) ns with fps_base = base_num / base_den exactly
// (BL §4.2), and the timeline record's motion_blur {use, shutter, position}.
struct FrameRate {
  int32_t fps;
  int64_t base_num, base_den;
};
struct BlenderShutter {
  bool use;
  float shutter;  // sigma, Blender's float32 motion_blur_shutter
  Position position;
};
enum class ExposureSource : uint8_t { kSet, kBlenderShutter, kDefault };  // "set", "blender_shutter", "default"
struct Exposure {
  int64_t ns;
  ExposureSource source;
};
inline constexpr int64_t kDefaultExposureNs = 10'000'000;
// frames.exposure_ns when set; otherwise, for a Blender source, round_half_even(sigma P) exactly when motion blur
// is on and 0 when it is off; otherwise 10 ms.
Exposure resolve_exposure(std::optional<int64_t> set_ns, const BlenderShutter* blender, const FrameRate* rate);
const char* exposure_source_name(ExposureSource);

inline constexpr int32_t kMaxDroppedFrames = 1 << 20;  // uniform: more dropped candidates are refused (FRM-SPAN)

class Schedule {
 public:
  // Frames k = 0 .. count - 1, or with count null every k >= 0 whose frame_eoe <= support.end_ns.
  static Schedule uniform(const RowTiming&, int64_t t0_ns, int64_t period_ns, std::optional<int32_t> count,
                          Span support);
  // frame_ns: t_ns(F_k) of the timeline's frames (frame_start + k stride <= frame_end), strictly increasing.
  static Schedule blender(const RowTiming&, std::span<const int64_t> frame_ns, Position, Span support);

  int32_t size() const { return static_cast<int32_t>(anchor_.size()); }
  std::span<const int32_t> dropped() const { return dropped_; }   // candidate indices outside the support
  int32_t candidate(int32_t k) const { return candidate_[static_cast<size_t>(k)]; }  // candidate index of frame k
  std::span<const Warning> warnings() const { return warnings_; }  // FRM-SPAN, FRM-EXPOSURE
  const RowTiming& timing() const { return timing_; }

  int64_t row_dt_ns() const { return timing_.shutter == Shutter::kRolling ? timing_.row_dt_ns : 0; }
  int64_t readout_span_ns() const { return static_cast<int64_t>(timing_.height - 1) * row_dt_ns(); }  // R
  int32_t read_order(int32_t row) const {
    return timing_.readout == Readout::kTopDown ? row : timing_.height - 1 - row;
  }
  int64_t soe(int32_t k, int32_t row) const { return frame_soe(k) + read_order(row) * row_dt_ns(); }
  int64_t eoe(int32_t k, int32_t row) const { return soe(k, row) + timing_.exposure_ns; }
  int64_t frame_soe(int32_t k) const { return anchor_[static_cast<size_t>(k)]; }
  int64_t frame_eoe(int32_t k) const { return frame_soe(k) + readout_span_ns() + timing_.exposure_ns; }
  int64_t frame_mid(int32_t k) const;  // floor((frame_soe + frame_eoe) / 2)

  // The schedule with frame k's windows moved by shift_ns[k] (the per-frame snap, §4.2); the dropped list is kept.
  Schedule shifted(std::span<const int64_t> shift_ns) const;

 private:
  explicit Schedule(const RowTiming& timing) : timing_(timing) {}
  void finish(int64_t n_candidates, Span support);  // FRM-SPAN: refuse when none is kept, else warn on drops

  RowTiming timing_;
  std::vector<int64_t> anchor_;
  std::vector<int32_t> candidate_, dropped_;
  std::vector<Warning> warnings_;
};

// The sensor clock of a true time: sensor::reported_time with delay 0 and quantum 0 (§4.1), i.e.
// t + clock_offset_ns + llround(clock_drift_ppm 1e-6 t), as the engine's sensor_clock_ns.
int64_t sensor_time(const sensor::Readout& clock, int64_t t_true);

}  // namespace balla::frames
