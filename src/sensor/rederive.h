// The event tier of re-simulation (phase3/ui.md §4.7 E): readout, clock and label-cut edits leave every crossing and
// its increments bit-identical (R5), so a session's raw events are re-derived instead of re-simulated: t_reported from
// t_cross under the new readout (its counter phase drawn again from the seed, the camera and the new quantum), the
// labels from the increments at the new cut, and the output order of the new values (re-sorted: a quantum change puts
// most rows out of order, RV2 e1). Exact under both initial states and with the high-pass filter, because a
// FIRST_AFTER_INIT event carries the kernel's own leak since t0 (RawEvent::l_first).
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "sensor/pixel.h"
#include "sensor/readout.h"

namespace balla::sensor {

struct Rederived {
  std::vector<uint32_t> order;      // the events' output order (indices into the input)
  std::vector<int64_t> t_reported;  // per input event, on the new sensor clock
  std::vector<uint8_t> label_s1;    // per input event (S1)
};

// E tier over a whole session stream (every call's raw events of a pixel set of the width-wide frame, pixel = y W + x).
// readout: the new delay, quantum, clock offset and drift (its phase_ns is ignored: drawn from seed, camera and the
// quantum, as SensorModel does). relabel: labels at label_cut from ds, dn, dl (split_label), or from l_first
// (first_label) for FIRST_AFTER_INIT events, with HOT over a non-signal label of a hot pixel, in fp32 (the kernel's
// precision of an fp32 session, whose f32 increments are its own operands); false (an fp64 session, whose labels the
// kernel decided in double: label_cut is not an E key there) copies the events' labels. y_bits as output_order's.
Rederived rederive(std::span<const RawEvent> ev, const Readout& readout, double label_cut, uint64_t seed,
                   uint32_t camera, uint32_t width, int y_bits, bool relabel);

}  // namespace balla::sensor
