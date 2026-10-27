#include "sensor/resim.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace balla::sensor {

Rederived rederive(std::span<const RawEvent> ev, const Readout& readout, double label_cut, uint64_t seed,
                   uint32_t camera, uint32_t width, int y_bits, bool relabel) {
  if (width == 0) throw std::invalid_argument("rederive: width must be >= 1");
  Readout r = readout;
  r.phase_ns = readout_phase_ns(seed, camera, r.quantum_ns);
  const float cut = static_cast<float>(label_cut);
  const size_t n = ev.size();
  Rederived out;
  out.t_reported.resize(n);
  out.label_s1.resize(n);
  std::vector<int64_t> t_cross(n);
  std::vector<uint16_t> x(n), y(n);
  for (size_t i = 0; i < n; ++i) {
    const RawEvent& e = ev[i];
    t_cross[i] = e.t_cross;
    x[i] = static_cast<uint16_t>(e.pixel % width);
    y[i] = static_cast<uint16_t>(e.pixel / width);
    out.t_reported[i] = reported_time(r, e.t_cross);
    uint8_t c3 = e.label;
    if (relabel) {
      // fire()'s decision in fp32 on the kernel's own f32 operands (pixel.h PixelKernel::fire).
      c3 = (e.flags & kFlagFirstAfterInit) != 0
               ? detail::first_label<float>(e.polarity != 0, e.ds, e.dn, e.l_first, cut)
               : detail::split_label<float>(e.ds, e.dl, e.ds + e.dn + e.dl, cut);
      if ((e.flags & kFlagHotPixel) != 0 && c3 != kLabelSignal) c3 = kLabelHot;
    }
    out.label_s1[i] = label_s1(c3);
  }
  output_order(t_cross, x, y, out.t_reported, y_bits, out.order);
  return out;
}

ResimRun::ResimRun(const CacheView& cache, const SensorSpec& spec, Roi roi, int64_t t1_ns, StartMode mode,
                   const ProbeResult* probe, Device device, int num_threads)
    : cache_(cache), roi_(roi), y_bits_(order_y_bits(cache.height)) {
  const Roi& c = cache.roi;
  if (!(roi.x0 >= c.x0 && roi.x0 < roi.x1 && roi.x1 <= c.x1 && roi.y0 >= c.y0 && roi.y0 < roi.y1 && roi.y1 <= c.y1)) {
    throw std::invalid_argument("ResimRun: the ROI must lie inside the cache's ROI");
  }
  const size_t G = spec.groups.size();
  if (G == 0 || cache.init_irr.size() != static_cast<size_t>(cache.width) * cache.height * G) {
    throw std::invalid_argument("ResimRun: the cache's init frame must be (width * height) x groups");
  }
  // The calls up to t1: every call ending at or before it, which must end a run window or the cache.
  n_calls_ = 0;
  while (n_calls_ < cache.calls.size() && cache.calls[n_calls_].t_end_ns <= t1_ns) ++n_calls_;
  const bool at_end = n_calls_ == cache.calls.size();
  if (!at_end && (n_calls_ == 0 || cache.calls[n_calls_ - 1].kind != CallKind::kWindow ||
                  cache.calls[n_calls_ - 1].t_end_ns != t1_ns)) {
    throw std::invalid_argument("ResimRun: t1_ns must end a run window of the cache, or lie at or after its end");
  }
  if (!cache.frame_mean.empty() && cache.frame_mean.size() < (n_calls_ + 1) * G) {
    throw std::invalid_argument("ResimRun: the cache has fewer frame means than calls + 1");
  }
  if (spec.params.veiling_glare > 0 && cache.frame_mean.empty()) {
    throw std::invalid_argument("ResimRun: veiling glare needs the cache's frame means");
  }
  // Camera::limit thins the camera's whole output stream, whose bins need every pixel's events: no ROI replay of it
  // (an edit with the limiter on re-renders, tier R).
  if (spec.params.limiter_on()) {
    throw std::invalid_argument("ResimRun: the camera-wide rate limiter (sensor.limiter) cannot be replayed on an ROI");
  }
  for (size_t k = 0; k < n_calls_; ++k) {
    const CacheCall& call = cache.calls[k];
    size_t samples = 0;
    if (call.count.size() != call.ids.size()) throw std::invalid_argument("ResimRun: a call needs one count per id");
    for (const int32_t n : call.count) {
      if (n < 0) throw std::invalid_argument("ResimRun: negative sample count");
      samples += static_cast<size_t>(n);
    }
    if (call.t.size() != samples || call.irr.size() != samples * G) {
      throw std::invalid_argument("ResimRun: call " + std::to_string(k) + "'s samples do not match its counts");
    }
  }
  SensorParams p = spec.params;
  p.initial_state = mode == StartMode::kPhase ? InitialState::kPhase : InitialState::kStationary;
  model_ = std::make_unique<SensorModel>(p, cache.width, cache.height, spec.seed, spec.groups, cache.camera,
                                         num_threads, spec.precision, device, roi);
  if (!cache.gain.empty()) model_->set_photometry(p.illuminance_scale, cache.gain);
  if (!cache.frame_mean.empty()) model_->set_frame_means(cache.frame_mean);
  if (mode == StartMode::kStationary && probe != nullptr) {
    model_->stationary_init(cache.t_init_ns, cache.init_irr.data(), *probe);
  } else {
    model_->init(cache.t_init_ns, cache.init_irr.data());
  }
}

ResimRun::~ResimRun() = default;

bool ResimRun::step(RawEvents* out) {
  out->clear();
  if (cancelled() || next_ >= n_calls_) return false;
  const CacheCall& call = cache_.calls[next_];
  // The call's listing inside the ROI: the runs stay where they are in the call's sample arrays.
  ids_.clear();
  start_.clear();
  count_.clear();
  int64_t at = 0;
  const uint32_t W = cache_.width;
  for (size_t i = 0; i < call.ids.size(); ++i) {
    const int64_t x = call.ids[i] % W, y = call.ids[i] / W;
    if (x >= roi_.x0 && x < roi_.x1 && y >= roi_.y0 && y < roi_.y1) {
      ids_.push_back(call.ids[i]);
      start_.push_back(at);
      count_.push_back(call.count[i]);
    }
    at += call.count[i];
  }
  std::vector<RawEvent> raw = model_->advance_runs_raw(ids_.data(), ids_.size(), call.t.data(), call.irr.data(),
                                                       call.t.size(), start_.data(), count_.data(), call.t_end_ns);
  ++next_;
  if (call.kind != CallKind::kWindow) return true;  // the hold's and the motion warm-up's events are dropped
  // Camera::output: the window's events after the held ones, the three passes, the cut at the window end.
  const Readout& r = model_->readout();
  for (const RawEvent& e : raw) {
    held_.ev.push_back(e);
    held_.t_reported.push_back(reported_time(r, e.t_cross));
    held_.label_s1.push_back(label_s1(e.label));
  }
  const size_t n = held_.size();
  std::vector<int64_t> t_cross(n);
  std::vector<uint16_t> x(n), y(n);
  for (size_t i = 0; i < n; ++i) {
    t_cross[i] = held_.ev[i].t_cross;
    x[i] = static_cast<uint16_t>(held_.ev[i].pixel % W);
    y[i] = static_cast<uint16_t>(held_.ev[i].pixel / W);
  }
  std::vector<uint32_t> perm;
  output_order(t_cross, x, y, held_.t_reported, y_bits_, perm);
  const int64_t limit = output_cut(r, call.t_end_ns);
  RawEvents rest;
  for (const uint32_t i : perm) {
    RawEvents& to = held_.t_reported[i] < limit ? *out : rest;
    to.ev.push_back(held_.ev[i]);
    to.t_reported.push_back(held_.t_reported[i]);
    to.label_s1.push_back(held_.label_s1[i]);
  }
  held_ = std::move(rest);
  return true;
}

void ResimRun::finish(RawEvents* out) {
  *out = std::move(held_);
  held_.clear();
}

}  // namespace balla::sensor
