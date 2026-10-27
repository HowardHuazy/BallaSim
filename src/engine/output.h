// The output stage of a window (ENGINE_CPP §1.3 K14-K15; the Python engine's CameraRun._output): per event the
// crop filter, x / y, t_reported, the centre-ray ground truth at its own exact pose (Renderer::gt_into),
// the S1 label and MOVING flag; then the held buffer, the sort by (t_reported, x, y) and the split
// at t_{w+1} + clock offset. The sort is three stable LSD passes, by t_cross, (x, y) and t_reported,
// which equals engine.py's stable lexsort of the held events followed by the window's events in
// (t_cross, pixel) order: two events tie in (t_reported, x, y) only at one pixel, where the earlier
// t_cross comes first. The keys, the S1 labels and the cut are sensor/output_order.h's, which the re-simulation shares
// (phase3/ui.md §3.10).
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "core/hd.h"
#include "engine/motion.h"
#include "engine/par.h"
#include "render/types.h"
#include "sensor/output_order.h"
#include "sensor/pixel.h"
#include "sensor/readout.h"

namespace balla::engine {

// One output event (C4 columns; the GT fields are left out of the file when GT is off).
struct OutEvent {
  int64_t t_reported;
  int64_t t_cross;
  float depth, flow_x, flow_y;
  uint32_t instance;
  uint16_t x, y, semantic, flags;
  uint8_t p, label;
};

// S1 /gt/flags bit 0: the centre-ray instance is outside its hold intervals at t_cross.
inline constexpr uint16_t kFlagMoving = 1;

// K14a: events inside the crop (x0 <= x < x1, y0 <= y < y1).
struct CropFlags {
  const sensor::RawEvent* raw;
  uint32_t width;
  int32_t x0, y0, x1, y1;
  uint8_t* flags;
  BALLA_HD void operator()(size_t i) const {
    const auto x = static_cast<int32_t>(raw[i].pixel % width), y = static_cast<int32_t>(raw[i].pixel / width);
    flags[i] = (x >= x0 && x < x1 && y >= y0 && y < y1) ? 1 : 0;
  }
};

// K14b: the kept events (raw[select[i]], or raw[i] without a crop) as OutEvents appended at out,
// and their GT queries.
struct BuildEvents {
  const sensor::RawEvent* raw;
  const uint32_t* select;  // null: every event
  uint32_t width;
  sensor::Readout readout;
  OutEvent* out;
  render::GtQuery* queries;
  BALLA_HD void operator()(size_t i) const {
    const sensor::RawEvent& e = raw[select != nullptr ? select[i] : i];
    OutEvent o{};
    o.x = static_cast<uint16_t>(e.pixel % width);
    o.y = static_cast<uint16_t>(e.pixel / width);
    o.p = e.polarity;
    o.t_cross = e.t_cross;
    o.t_reported = sensor::reported_time(readout, e.t_cross);
    o.label = sensor::label_s1(e.label);
    o.flags = e.flags;
    out[i] = o;
    if (queries != nullptr) queries[i] = {e.t_cross, static_cast<float>(o.x), static_cast<float>(o.y)};
  }
};

// K14c: ground truth and MOVING into the events.
struct AttachGt {
  const render::GtSample* gt;
  MovingTable bodies;
  OutEvent* out;
  BALLA_HD void operator()(size_t i) const {
    OutEvent& o = out[i];
    const render::GtSample& g = gt[i];
    o.instance = g.instance;
    o.semantic = static_cast<uint16_t>(g.semantic);
    o.depth = g.depth;
    o.flow_x = g.flow_x;
    o.flow_y = g.flow_y;
    if (bodies.moving(g.instance, o.t_cross)) o.flags = static_cast<uint16_t>(o.flags | kFlagMoving);
  }
};

// K15: sort keys through the current permutation (pass 0: t_cross, 1: (x, y), 2: t_reported; signed
// times biased to unsigned order; sensor::order_key).
struct OutKeys {
  const OutEvent* ev;
  const uint32_t* perm;
  int pass;
  int y_bits;
  uint64_t* keys;
  BALLA_HD void operator()(size_t i) const {
    const OutEvent& e = ev[perm[i]];
    keys[i] = sensor::order_key(pass, e.t_cross, e.x, e.y, e.t_reported, y_bits);
  }
};

struct GatherEvents {
  const OutEvent* in;
  const uint32_t* perm;
  OutEvent* out;
  unsigned long long* ready;  // events reported before the limit
  int64_t limit;
  BALLA_HD void operator()(size_t i) const {
    out[i] = in[perm[i]];
    count_if(ready, out[i].t_reported < limit);
  }
};

}  // namespace balla::engine
