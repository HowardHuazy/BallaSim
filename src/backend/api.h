// Backend records (DESIGN §4.3): the fixed records every backend shares (SampleRequest, HitRecord
// and the instance sentinels). The tracer interface itself is render/tracer.h, which superseded
// the P1 draft that lived here (docs/dev/RENDER.md).
//
// The interface is request-level: ray generation, the exact-pose multi-trace and the 16 B
// hit-record write all run inside the backend's kernel as shared BALLA_HD code, so rays are never
// materialized (a ray-batch interface would move ~630 MB per full-frame launch).
#pragma once

#include <cstdint>

namespace balla::backend {

// What a request is for; stored in the low bits of SampleRequest::flags.
enum class RequestKind : uint16_t {
  kSample = 0,    // regular lattice sample of a pixel
  kProbe = 1,     // visibility-only probe that localizes a jump (§6.2)
  kAudit = 2,     // quarter-point check of an accepted interval (§6.5)
  kWatchdog = 3,  // full-pixel-sample re-render that verifies a bound (§6.8)
  kGt = 4,        // ground-truth evaluation (§9)
};
inline constexpr uint16_t kRequestKindMask = 0x7;

// One pixel-sample: a contiguous range of sub-samples of one pixel at one time.
struct SampleRequest {
  uint16_t cam;
  uint16_t flags;      // RequestKind in the low 3 bits; shadow-light and defocus bits above (P1-M2)
  uint32_t pixel;      // y * width + x
  uint32_t k_begin;    // sub-sample index range of the nested Owen-Sobol prefix (§6.6)
  uint32_t k_count;
  int32_t t_off_ns;    // time as an offset from the work item's int64 base (|offset| < 2^31 ns)
  uint32_t pose_slot;  // pose-table row of this lattice node (§6.9)
};
static_assert(sizeof(SampleRequest) == 24);

// Instance ids (DESIGN §5.1): 0 is a miss, 0xFFFFFFFF an invalid record; real instances are
// numbered from 1.
inline constexpr uint32_t kInstanceNone = 0;
inline constexpr uint32_t kInstanceInvalid = 0xFFFFFFFFu;

// Result of one sub-sample ray (16 B, G6 §2.1). A miss has instance == kInstanceNone.
struct HitRecord {
  uint32_t instance;
  uint32_t prim;
  uint16_t b1, b2;  // unorm16 barycentrics: hints only, never shaded with; the cluster kernel
                    // re-intersects the fetched triangle in fp32 for exact barycentrics
  float t_hit;
};
static_assert(sizeof(HitRecord) == 16);

}  // namespace balla::backend
