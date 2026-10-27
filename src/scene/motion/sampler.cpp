// The keyed sampler's pure pieces (pipeline.h; blender.md §4.7.3): the base requests and the midpoint test.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "core/math/so3.h"
#include "scene/motion/pipeline.h"

namespace balla::scene::motion {

namespace {

// blender.md §4.7.3's "10.4 ms": a quarter frame at 24 fps, 1/96 s, as the research sampled it (MF a6_fit_blender.py,
// h = 20 833 333, 10 416 667, 5 208 333 ns), so 24 fps gets 4 base samples per frame
constexpr int64_t kH0CapNs = 10416667;

}  // namespace

const char* stage_name(Stage s) {
  switch (s) {
    case Stage::kBase: return "base";
    case Stage::kRefine: return "refine";
    case Stage::kDetect: return "detect";
    case Stage::kBurst: return "burst";
    case Stage::kJoin: return "join";
    case Stage::kFill: return "fill";
    case Stage::kValidate: return "validate";
    case Stage::kDone: return "done";
  }
  return "?";
}

void Track::push(const Track& o, size_t i) {
  t_ns.push_back(o.t_ns[i]);
  q.push_back(o.q[i]);
  p.push_back(o.p[i]);
  w_pos.push_back(o.w_pos[i]);
  w_rot.push_back(o.w_rot[i]);
  floor1.push_back(o.floor1[i]);
}

void Track::push(const PoseBatch& b, size_t i, int32_t body) {
  const size_t k = b.at(i, body);
  t_ns.push_back(b.t_ns[i]);
  q.push_back(b.q[k]);
  p.push_back(b.p[k]);
  w_pos.push_back(b.w_pos[k]);
  w_rot.push_back(b.w_rot[k]);
  floor1.push_back(b.floor1[k]);
  light_near = light_near || b.light_near[k] != 0;
}

int64_t h0_ns(const TimeMap& time, const SampleOptions& opt) {
  if (opt.h0_ns > 0) return opt.h0_ns;
  const double period = time.period_ns();
  return std::max<int64_t>(1, std::llround(period / std::max(2.0, std::ceil(period / static_cast<double>(kH0CapNs)))));
}

std::optional<PosRes> pos_res(const SampleSet& set, double eps_px) {
  if (!(set.floor_px > eps_px / 8.0)) return std::nullopt;
  return PosRes{set.floor_px, eps_px / 2.0 + 2.0 * set.floor_px};
}

std::vector<int64_t> base_requests(const TimeMap& time, int64_t t0, int64_t t1, std::span<const DeclaredKey> keys,
                                   const SampleOptions& opt) {
  if (!(t0 < t1)) throw std::invalid_argument("base_requests: empty span");
  const double period = time.period_ns();
  // the h0 grid (h0 = min(P / 2, 10.4 ms) by default: at least 2 per frame), at least half the aliasing pre-pass's
  // density, which the first round's midpoints complete
  const double h0_density = opt.h0_ns > 0 ? std::ceil(period / static_cast<double>(opt.h0_ns))
                                          : std::max(2.0, std::ceil(period / static_cast<double>(kH0CapNs)));
  const double alias_half = std::ceil(static_cast<double>(opt.alias_per_frame) / 2.0);
  const auto per_frame = static_cast<int64_t>(std::max(h0_density, alias_half));
  if (per_frame < 1 || per_frame > 1000000) throw std::invalid_argument("base_requests: invalid grid density");
  const double step = period / static_cast<double>(per_frame);
  // the size before allocating it: the grid, the span ends and up to two samples per key
  const double expected =
      std::floor(static_cast<double>(t1 - t0) / step) + 3.0 + 2.0 * static_cast<double>(keys.size());
  if (!(expected <= static_cast<double>(opt.max_base_samples))) {
    char buf[200];
    std::snprintf(buf, sizeof buf,
                  "base_requests: a frame period of %.6g ns gives about %.3g base samples over the %.6g s span, more "
                  "than the cap of %lld",
                  period, expected, static_cast<double>(t1 - t0) * 1e-9, static_cast<long long>(opt.max_base_samples));
    throw std::invalid_argument(buf);
  }
  std::vector<int64_t> out{t0, t1};
  out.reserve(static_cast<size_t>(expected));
  // j / per_frame frames after F_ref, for every j whose exact time lies in [t0, t1]
  int64_t j = static_cast<int64_t>(std::floor(static_cast<double>(t0 - time.t_ref_ns()) / step)) - 2;
  while (time.frames_to_ns(j, per_frame) < t0) ++j;
  for (int64_t t = time.frames_to_ns(j, per_frame); t <= t1; t = time.frames_to_ns(++j, per_frame)) out.push_back(t);
  for (const DeclaredKey& k : keys) {
    if (k.t_ns >= t0 && k.t_ns <= t1) out.push_back(k.t_ns);
    if (k.hint == Continuity::kStep) {
      int32_t F = 0;
      float s = 0.0f;
      time.request(k.t_ns, &F, &s);
      const int64_t pre = k.t_ns - std::llround(k.plateau_frames * period) -
                          static_cast<int64_t>(std::ceil(2.0 * time.resolution_ns(F)));
      if (pre >= t0 && pre <= t1) out.push_back(pre);
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

MidpointTest midpoint_test(const Track& fit, size_t i, const Track& mid, size_t m, double eps_px) {
  const size_t n = fit.size();
  if (n < 2 || i + 1 >= n) throw std::invalid_argument("midpoint_test: no such interval");
  const size_t count = std::min<size_t>(4, n);
  const double tm = static_cast<double>(mid.t_ns[m] - fit.t_ns[i]) * 1e-9;
  const rot::Q qi_inv = conjugate(fit.q[i]);
  MidpointTest best{false, std::numeric_limits<double>::infinity(), 0.0};
  for (const int st : {-1, -2, 0}) {
    const auto a =
        static_cast<size_t>(std::clamp<int64_t>(static_cast<int64_t>(i) + st, 0, static_cast<int64_t>(n - count)));
    double w[4], tn[4], gain = 1.0;
    for (size_t k = 0; k < count; ++k) tn[k] = static_cast<double>(fit.t_ns[a + k] - fit.t_ns[i]) * 1e-9;
    for (size_t k = 0; k < count; ++k) {
      w[k] = 1.0;
      for (size_t l = 0; l < count; ++l) {
        if (l != k) w[k] *= (tm - tn[l]) / (tn[k] - tn[l]);
      }
      gain += std::fabs(w[k]);
    }
    rot::V pp{0, 0, 0}, phi{0, 0, 0};
    double f1 = mid.floor1[m];
    for (size_t k = 0; k < count; ++k) {
      pp = pp + w[k] * fit.p[a + k];
      phi = phi + w[k] * so3::log(qi_inv * fit.q[a + k]);
      f1 = std::max(f1, fit.floor1[a + k]);
    }
    const rot::Q qp = fit.q[i] * so3::exp(phi);
    const double e = mid.w_pos[m] * norm(pp - mid.p[m]) + mid.w_rot[m] * norm(so3::log(conjugate(qp) * mid.q[m]));
    const double floor = ErrorModel::floor_px(f1, gain);
    if (e <= eps_px / 4.0 + floor) return {true, e, floor};
    if (e < best.residual_px) best.residual_px = e;
  }
  return best;
}

}  // namespace balla::scene::motion
