// BodyPipeline (pipeline.h): the keyed sampler of blender.md §4.7.3 and the breakpoint detection of §4.7.5 as a
// resumable state machine.
#include "scene/motion/pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>

#include "core/math/so3.h"
#include "scene/errors.h"
#include "scene/motion/detect.h"

namespace balla::scene::motion {

namespace {

__extension__ typedef __int128 i128;

constexpr double kAliasRad = 0.6 * std::numbers::pi;  // the aliasing pre-pass's bound between consecutive samples
constexpr int kBisectMax = 40;                         // pose-distance bisection steps of a step candidate
constexpr int64_t kBurstW0MinNs = 2000000;             // a kink's first burst window: >= 2 ms, >= 4 local spacings

double angle(const rot::Q& a, const rot::Q& b) { return norm(so3::log(conjugate(a) * b)); }

std::vector<int64_t> midpoints(const std::vector<std::pair<int64_t, int64_t>>& iv) {
  std::vector<int64_t> out;
  out.reserve(iv.size());
  for (const auto& [a, b] : iv) out.push_back(a + (b - a) / 2);
  return out;
}

// t ascending in time, each time once (the first of equal times: the same realized time, the same answer)
Track sorted_unique(const Track& t) {
  std::vector<size_t> order(t.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return t.t_ns[a] < t.t_ns[b]; });
  Track out;
  for (const size_t k : order) {
    if (out.size() && out.t_ns.back() == t.t_ns[k]) continue;
    out.push(t, k);
  }
  out.light_near = t.light_near;
  return out;
}

// the samples of t in [a, b) ([a, b] when closed)
Track slice(const Track& t, int64_t a, int64_t b, bool closed) {
  Track out;
  auto i = static_cast<size_t>(std::lower_bound(t.t_ns.begin(), t.t_ns.end(), a) - t.t_ns.begin());
  for (; i < t.size() && (t.t_ns[i] < b || (closed && t.t_ns[i] == b)); ++i) out.push(t, i);
  return out;
}

// the sample of t at exactly time x, if any
std::optional<size_t> sample_at(const Track& t, int64_t x) {
  const auto it = std::lower_bound(t.t_ns.begin(), t.t_ns.end(), x);
  if (it == t.t_ns.end() || *it != x) return std::nullopt;
  return static_cast<size_t>(it - t.t_ns.begin());
}

uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

Track select(const Track& t, const std::vector<uint8_t>& keep) {
  Track out;
  for (size_t i = 0; i < t.size(); ++i) {
    if (keep[i]) out.push(t, i);
  }
  out.light_near = t.light_near;
  return out;
}

}  // namespace

Plateaus::Plateaus(const TimeMap& time, std::span<const DeclaredKey> keys) {
  for (const DeclaredKey& k : keys) {
    if (!k.plateau) continue;
    int32_t F = 0;
    float s = 0.0f;
    time.request(k.t_ns, &F, &s);
    const int64_t w = std::llround(k.plateau_frames * time.period_ns());
    p_.push_back({k.t_ns, w, static_cast<int64_t>(std::ceil(time.resolution_ns(F))) + 1});
    w_max_ = std::max(w_max_, w);
  }
  std::sort(p_.begin(), p_.end(), [](const P& a, const P& b) { return a.t < b.t; });
}

bool Plateaus::inside(int64_t t) const {
  auto it = std::lower_bound(p_.begin(), p_.end(), t - w_max_, [](const P& a, int64_t v) { return a.t < v; });
  for (; it != p_.end() && it->t <= t + w_max_; ++it) {
    const int64_t d = t > it->t ? t - it->t : it->t - t;
    if (d > it->tol && d <= it->w) return true;
  }
  return false;
}

int64_t Plateaus::dropped_ns(int64_t a, int64_t c) const {
  int64_t sum = 0;
  auto it = std::lower_bound(p_.begin(), p_.end(), a - w_max_, [](const P& x, int64_t v) { return x.t < v; });
  for (; it != p_.end() && it->t <= c + w_max_; ++it) {
    // (a, c) meets [t - w, t - tol - 1] or [t + tol + 1, t + w], the samples inside() drops
    const bool left = it->t - it->w < c && it->t - it->tol - 1 > a;
    const bool right = it->t + it->tol + 1 < c && it->t + it->w > a;
    if (it->w > it->tol && (left || right)) sum += 2 * it->w;
  }
  return sum;
}

BodyPipeline::BodyPipeline(int32_t body, Mode mode, const TimeMap& time, std::vector<DeclaredKey> keys,
                           const SampleOptions& opt, const DetectOptions& detect, const FitOptions& fit)
    : body_(body),
      time_(time),
      keys_(std::move(keys)),
      opt_(opt),
      dopt_(detect),
      plateaus_(time, keys_),
      t0_(opt.t0_ns.value_or(-2 * time::kWindowNs)),
      t1_(opt.t1_ns.value_or(time.t_end_ns())),
      fopt_(fit) {
  if (mode != Mode::kKeyed) throw std::invalid_argument("BodyPipeline: frame-state sampling arrives with BL-10");
  if (!(opt.eps_px > 0.0) || opt.h_min_ns < 1 || opt.max_rounds < 1 || opt.alias_per_frame < 0 || opt.h0_ns < 0 ||
      opt.max_per_message < 1 || opt.max_per_message > 512 || opt.max_base_samples < 1) {
    throw std::invalid_argument("BodyPipeline: invalid sample options");
  }
  if (!(t0_ < t1_)) throw std::invalid_argument("BodyPipeline: the sampled span is empty");
  if (!(detect.k_sig > 0.0) || !(detect.unsplit_coeff >= 0.0) || detect.burst_n < 3 || detect.burst_iters < 0 ||
      detect.burst_floor_ns < 1 || detect.merge_ns < 0 || detect.h_cap_ns < 0) {
    throw std::invalid_argument("BodyPipeline: invalid detect options");
  }
  if (fit.h_cap_ns < 0 || fit.h_floor_ns < 1 || !(fit.lambda_rel > 0.0) || !(fit.join_weight >= 1.0) ||
      !(fit.accept_frac > 0.0 && fit.accept_frac <= 0.5) || fit.halvings < 0 || fit.probes < 0 || fit.gn_iters < 1 ||
      fit.end_fill_div < 0) {
    throw std::invalid_argument("BodyPipeline: invalid fit options");
  }
  std::sort(keys_.begin(), keys_.end(), [](const DeclaredKey& a, const DeclaredKey& b) { return a.t_ns < b.t_ns; });
  pending_ = base_requests(time_, t0_, t1_, keys_, opt_);
}

void BodyPipeline::feed(const Track& answers) {
  if (done() || answers.size() != pending_.size()) throw std::logic_error("BodyPipeline::feed: answers do not match");
  answers_ = answers;
  fed_ = true;
}

void BodyPipeline::insert(const Track& add) {
  Track& fit = set_.fit;
  Track all = fit;
  for (size_t i = 0; i < add.size(); ++i) all.push(add, i);
  std::vector<size_t> order(all.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return all.t_ns[a] < all.t_ns[b]; });
  Track merged;
  for (const size_t k : order) {
    if (merged.size() && merged.t_ns.back() == all.t_ns[k]) continue;  // the same realized time: the same answer
    merged.push(all, k);
  }
  merged.light_near = fit.light_near;
  fit = std::move(merged);
}

void BodyPipeline::step() {
  if (!fed_) throw std::logic_error("BodyPipeline::step before feed");
  fed_ = false;
  if (stage_ == Stage::kDetect) return step_bisect();
  if (stage_ == Stage::kBurst) return step_burst();
  if (stage_ == Stage::kJoin || stage_ == Stage::kFill || stage_ == Stage::kValidate) {
    set_.light_near = set_.light_near || answers_.light_near;
  }
  if (stage_ == Stage::kJoin) return step_join();
  if (stage_ == Stage::kFill) {
    Track add;
    for (size_t i = 0; i < answers_.size(); ++i) {
      if (!plateaus_.inside(answers_.t_ns[i])) add.push(answers_, i);
    }
    insert(add);
    return enter_validate();
  }
  if (stage_ == Stage::kValidate) {
    Track all = set_.probes;
    for (size_t i = 0; i < answers_.size(); ++i) {
      if (!plateaus_.inside(answers_.t_ns[i])) all.push(answers_, i);
    }
    set_.probes = sorted_unique(all);
    return validate_pieces();
  }
  set_.light_near = set_.light_near || answers_.light_near;
  std::vector<std::pair<int64_t, int64_t>> next;
  if (stage_ == Stage::kBase) {  // the base round: every base interval is open
    insert(answers_);
    stage_ = Stage::kRefine;
    for (size_t i = 0; i + 1 < set_.fit.size(); ++i) next.emplace_back(set_.fit.t_ns[i], set_.fit.t_ns[i + 1]);
  } else if (stage_ == Stage::kRefine) {
    ++set_.rounds;
    Track add;
    const Track& fit = set_.fit;
    for (size_t k = 0; k < open_.size(); ++k) {
      const auto [a, b] = open_[k];
      const int64_t tm = answers_.t_ns[k];
      if (!(a < tm && tm < b)) {  // the float32 time grid cannot resolve the interval any more
        set_.unresolved.emplace_back(a, b);
        continue;
      }
      const auto i = static_cast<size_t>(std::lower_bound(fit.t_ns.begin(), fit.t_ns.end(), a) - fit.t_ns.begin());
      const MidpointTest r = midpoint_test(fit, i, answers_, k, opt_.eps_px);
      // the aliasing pre-pass: a midpoint stands for its interval only when both halves turn by less than 0.6 pi
      const bool slow = angle(fit.q[i], answers_.q[k]) < kAliasRad && angle(answers_.q[k], fit.q[i + 1]) < kAliasRad;
      if (r.accepted && slow) {
        set_.held_out.push(answers_, k);
        set_.residual_px.push_back(r.residual_px);
        set_.floor_px = std::max(set_.floor_px, r.floor_px);
        continue;
      }
      add.push(answers_, k);
      if (b - a <= 2 * opt_.h_min_ns) {
        set_.unresolved.emplace_back(a, b);
      } else {
        next.emplace_back(a, tm);
        next.emplace_back(tm, b);
      }
    }
    insert(add);
    if (!next.empty() && set_.rounds >= opt_.max_rounds) {
      set_.unresolved.insert(set_.unresolved.end(), next.begin(), next.end());
      next.clear();
    }
  } else {
    throw std::logic_error("BodyPipeline::step: stage not implemented");
  }
  open_ = std::move(next);
  if (open_.empty()) {  // converged (or fewer than two samples)
    finish_sampling();
    start_detect();
    return;
  }
  pending_ = midpoints(open_);
}

void BodyPipeline::finish_sampling() {
  pending_.clear();
  open_.clear();
  std::sort(set_.unresolved.begin(), set_.unresolved.end());
  {  // V in time order (each round accepts midpoints across the whole span), with its residuals
    const Track& v = set_.held_out;
    std::vector<size_t> order(v.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return v.t_ns[a] < v.t_ns[b]; });
    Track sorted;
    std::vector<double> res;
    res.reserve(order.size());
    for (const size_t k : order) {
      sorted.push(v, k);
      res.push_back(set_.residual_px[k]);
    }
    sorted.light_near = v.light_near;
    set_.held_out = std::move(sorted);
    set_.residual_px = std::move(res);
  }
  {  // the aliasing check over fit and V together: what h_min, the time grid or max_rounds left at >= 0.6 pi between
     // consecutive samples is a warning, above 0.9 pi MOT-ALIAS (DESIGN.md:441)
    const Track &f = set_.fit, &v = set_.held_out;
    size_t i = 0, j = 0;
    const rot::Q* prev = nullptr;
    int64_t t_prev = 0;
    while (i < f.size() || j < v.size()) {
      const bool from_fit = j == v.size() || (i < f.size() && f.t_ns[i] <= v.t_ns[j]);
      const rot::Q* cur = from_fit ? &f.q[i] : &v.q[j];
      const int64_t t = from_fit ? f.t_ns[i++] : v.t_ns[j++];
      if (prev) {
        const double a = angle(*prev, *cur);
        set_.alias_max_rad = std::max(set_.alias_max_rad, a);
        if (a > 0.9 * std::numbers::pi) {
          throw MotionError("MOT-ALIAS", "body " + std::to_string(body_) + " turns " + std::to_string(a) +
                                             " rad between samples at " + std::to_string(t_prev) + " and " +
                                             std::to_string(t) + " ns");
        }
        if (a >= kAliasRad) ++set_.alias_warnings;
      }
      prev = cur;
      t_prev = t;
    }
  }
  // Samples strictly inside a declared key's plateau show the key's value, not the motion: dropped (§4.7.4). The key's
  // own sample (within its time resolution) stays.
  std::vector<uint8_t> keep(set_.fit.size()), keep_v(set_.held_out.size());
  for (size_t i = 0; i < keep.size(); ++i) keep[i] = !plateaus_.inside(set_.fit.t_ns[i]);
  for (size_t i = 0; i < keep_v.size(); ++i) keep_v[i] = !plateaus_.inside(set_.held_out.t_ns[i]);
  set_.fit = select(set_.fit, keep);
  set_.held_out = select(set_.held_out, keep_v);
  std::vector<double> res;
  for (size_t i = 0; i < keep_v.size(); ++i) {
    if (keep_v[i]) res.push_back(set_.residual_px[i]);
  }
  set_.residual_px = std::move(res);
}

// --- detection (§4.7.5) ---

size_t BodyPipeline::answer_of(int64_t target) const {
  return static_cast<size_t>(std::lower_bound(pending_.begin(), pending_.end(), target) - pending_.begin());
}

void BodyPipeline::set_pending() {
  pending_.clear();
  for (const Loc& l : locs_) {
    if (l.active) pending_.insert(pending_.end(), l.req.begin(), l.req.end());
  }
  std::sort(pending_.begin(), pending_.end());
  pending_.erase(std::unique(pending_.begin(), pending_.end()), pending_.end());
}

void BodyPipeline::start_detect() {
  stage_ = Stage::kDetect;
  const Track& fit = set_.fit;
  const double eps = opt_.eps_px;
  const std::vector<int64_t> cand = candidates(set_, keys_, eps);
  set_.detect.candidates = static_cast<int>(cand.size());
  const StepBracket bracket{opt_.h_min_ns, &time_, &plateaus_};
  std::vector<Loc> locs;
  for (const int64_t t : cand) {
    const Classified c = classify(fit, t, eps, bracket, dopt_.k_sig);
    if (c.kind == Classified::Kind::kSmooth) ++set_.detect.smooth;
    if (c.kind == Classified::Kind::kUnclear) ++set_.detect.unclear;
    if (c.kind != Classified::Kind::kStep && c.kind != Classified::Kind::kKink) continue;
    Loc l{};
    l.cls = c.kind == Classified::Kind::kStep ? Continuity::kStep : Continuity::kC0;
    l.t = c.t_ns;
    l.jump_px = c.jump_px;
    l.dv_pxps = c.dv_pxps;
    l.v_pxps = c.v_pxps;
    l.w_pos = c.w_pos;
    l.w_rot = c.w_rot;
    l.f1 = c.f1;
    if (l.cls == Continuity::kStep) {
      l.a = fit.t_ns[c.a];
      l.c = fit.t_ns[c.c];
      l.qa = fit.q[c.a];
      l.pa = fit.p[c.a];
      l.qc = fit.q[c.c];
      l.pc = fit.p[c.c];
    }
    locs.push_back(l);
  }
  // candidates of one break classify alike: merge them before any oracle call (a step wins, then the larger change)
  std::stable_sort(locs.begin(), locs.end(), [](const Loc& a, const Loc& b) { return a.t < b.t; });
  locs_.clear();
  for (const Loc& l : locs) {
    if (!locs_.empty() && l.t - locs_.back().t < dopt_.merge_ns) {
      const Loc& o = locs_.back();
      const bool ls = l.cls == Continuity::kStep, os = o.cls == Continuity::kStep;
      if (ls != os ? ls : (ls ? l.jump_px > o.jump_px : l.dv_pxps > o.dv_pxps)) locs_.back() = l;
      continue;
    }
    locs_.push_back(l);
  }
  for (Loc& l : locs_) {
    l.active = true;
    if (l.cls == Continuity::kStep) {
      if (l.c - l.a <= 1) {
        end_step(l);
      } else {
        l.req = {l.a + (l.c - l.a) / 2};
      }
    } else {
      start_kink(l);
    }
  }
  if (std::any_of(locs_.begin(), locs_.end(), [](const Loc& l) { return l.active && l.cls == Continuity::kStep; })) {
    set_pending();
    return;
  }
  start_bursts();
}

void BodyPipeline::start_kink(Loc& l) {
  const Track& fit = set_.fit;
  const size_t n = fit.size();
  auto i = static_cast<size_t>(std::lower_bound(fit.t_ns.begin(), fit.t_ns.end(), l.t) - fit.t_ns.begin());
  i = std::clamp<size_t>(i, 1, n - 1);
  l.cls = Continuity::kC0;
  l.w_ns = std::max(4 * (fit.t_ns[i] - fit.t_ns[i - 1]), kBurstW0MinNs);
  l.iters = 0;
  l.req.clear();
}

void BodyPipeline::end_step(Loc& l) {
  l.active = false;
  l.req.clear();
  // The jump persists down to the time resolution: the end poses differ by more than the motion over the bracket
  // explains (the larger one-sided speed times its width; negligible after bisection to 1 ns, but Blender's time grid
  // can end it at 2 q(F), where a kink's end poses differ by about that speed times the bracket).
  const double jump = pose_px(l.w_pos, l.w_rot, l.qa, l.pa, l.qc, l.pc);
  const double motion = l.v_pxps * static_cast<double>(l.c - l.a) * 1e-9;
  if (jump - motion > opt_.eps_px / 4.0 + 2.0 * l.f1) {  // a step
    found_.push_back({l.c, Continuity::kStep, jump, l.dv_pxps, false, false, static_cast<double>(l.c - l.a)});
    ++set_.detect.steps;
    return;
  }
  ++set_.detect.demoted;  // a kink seen through a coarse bracket
  l.active = true;
  start_kink(l);
}

void BodyPipeline::step_bisect() {
  for (Loc& l : locs_) {
    if (!l.active || l.cls != Continuity::kStep) continue;
    const size_t k = answer_of(l.req.front());
    const int64_t tm = answers_.t_ns[k];
    if (!(l.a < tm && tm < l.c)) {  // the time grid cannot split the bracket any more
      end_step(l);
      continue;
    }
    const rot::Q& q = answers_.q[k];
    const rot::V& p = answers_.p[k];
    if (pose_px(l.w_pos, l.w_rot, q, p, l.qa, l.pa) <= pose_px(l.w_pos, l.w_rot, q, p, l.qc, l.pc)) {
      l.a = tm;
      l.qa = q;
      l.pa = p;
    } else {
      l.c = tm;
      l.qc = q;
      l.pc = p;
    }
    if (l.c - l.a <= 1 || ++l.iters >= kBisectMax) {
      end_step(l);
    } else {
      l.req = {l.a + (l.c - l.a) / 2};
    }
  }
  if (std::any_of(locs_.begin(), locs_.end(), [](const Loc& l) { return l.active && l.cls == Continuity::kStep; })) {
    set_pending();
    return;
  }
  start_bursts();
}

void BodyPipeline::start_bursts() {
  stage_ = Stage::kBurst;
  for (Loc& l : locs_) {
    if (!l.active || l.cls != Continuity::kC0) continue;
    if (dopt_.burst_iters == 0) {
      l.active = false;
      continue;
    }
    l.req = burst_requests(l.t, l.w_ns, dopt_.burst_n);
  }
  set_pending();
  if (pending_.empty()) finish_detect();
}

void BodyPipeline::step_burst() {
  for (Loc& l : locs_) {
    if (!l.active || l.cls != Continuity::kC0) continue;
    Track s;  // the burst's samples outside every plateau, ascending (requests are monotone in time)
    for (const int64_t target : l.req) {
      const size_t k = answer_of(target);
      if (!plateaus_.inside(answers_.t_ns[k])) s.push(answers_, k);
    }
    const Crossing x = burst_crossing(s, l.t, l.w_ns, l.w_pos, l.w_rot);
    l.w_ns = std::max(l.w_ns / 4, dopt_.burst_floor_ns);
    if (x.ok) {
      l.t = x.t_ns;
      l.dv_pxps = x.dv_pxps;
      l.jump_px = x.jump_px;
    }
    if (!x.ok || ++l.iters >= dopt_.burst_iters) {
      l.active = false;
      l.req.clear();
    } else {
      l.req = burst_requests(l.t, l.w_ns, dopt_.burst_n);
    }
  }
  set_pending();
  if (pending_.empty()) finish_detect();
}

void BodyPipeline::finish_detect() {
  const double eps = opt_.eps_px;
  const double h_cap = static_cast<double>(dopt_.h_cap_ns > 0 ? dopt_.h_cap_ns : 2 * h0_ns(time_, opt_)) * 1e-9;
  for (const Loc& l : locs_) {
    if (l.cls != Continuity::kC0) continue;
    // split only when an unsplit kink would cost more than eps/8 (about 0.1 dv h_cap)
    if (dopt_.unsplit_coeff * l.dv_pxps * h_cap > eps / 8.0) {
      found_.push_back({l.t, Continuity::kC0, l.jump_px, l.dv_pxps, false, false, eps / 8.0 / l.dv_pxps * 1e9});
      ++set_.detect.kinks;
    } else {
      ++set_.detect.insignificant;
    }
  }
  std::vector<Breakpoint> bps = merge_breakpoints(std::move(found_), dopt_.merge_ns);
  set_.detect.snapped = snap_breakpoints(bps, keys_, time_);
  found_.clear();
  locs_.clear();
  pending_.clear();
  stage_ = Stage::kDone;
  set_.breakpoints = std::move(bps);
  if (dopt_.refuse_steps) {
    for (const Breakpoint& b : set_.breakpoints) {
      if (b.cls != Continuity::kStep) continue;
      throw MotionError("MOT-STEP", "body " + std::to_string(body_) + " steps by " + std::to_string(b.jump_px) +
                                        " px at " + std::to_string(b.t_ns) + " ns" +
                                        (b.snapped ? " (a declared key)" : "") +
                                        "; steps are refused until BL-12");
    }
  }
  if (fopt_.enabled) start_fit();
}

// --- the piecewise fit and its validation (§4.7.6) ---

bool BodyPipeline::has_sample(int64_t t) const { return sample_at(set_.fit, t).has_value(); }

int64_t BodyPipeline::plateau_ns(int64_t t) const {
  int64_t w = 0;
  for (const DeclaredKey& k : keys_) {
    if (!k.plateau) continue;
    int32_t F = 0;
    float s = 0.0f;
    time_.request(k.t_ns, &F, &s);
    const auto tol = static_cast<int64_t>(std::ceil(time_.resolution_ns(F))) + 1;
    if ((t > k.t_ns ? t - k.t_ns : k.t_ns - t) <= tol) {
      w = std::max(w, static_cast<int64_t>(std::llround(k.plateau_frames * time_.period_ns())));
    }
  }
  return w;
}

void BodyPipeline::start_fit() {
  Track all = set_.fit;
  for (size_t i = 0; i < set_.held_out.size(); ++i) all.push(set_.held_out, i);
  all = sorted_unique(all);
  const std::vector<std::pair<int64_t, int64_t>> holds = find_holds(all);
  const std::vector<PiecePlan> plan = plan_pieces(t0_, t1_, set_.breakpoints, holds, all.t_ns);
  pieces_.clear();
  for (const PiecePlan& pp : plan) {
    Piece pc;
    pc.fit.begin_ns = pp.begin_ns;
    pc.fit.end_ns = pp.end_ns;
    pc.fit.t0_ns = pp.begin_ns;
    pc.fit.join = pp.join;
    pc.fit.hold = pp.hold;
    pc.need_fill = !pp.hold;
    pieces_.push_back(std::move(pc));
  }
  dirty_ = 0;
  // a hold run's samples may be V's: the fit set takes those at piece begins
  Track add;
  for (size_t k = 1; k < pieces_.size(); ++k) {
    const int64_t b = pieces_[k].fit.begin_ns;
    if (has_sample(b)) continue;
    if (const auto v = sample_at(set_.held_out, b)) add.push(set_.held_out, *v);
  }
  insert(add);
  // kJoin: a sample at every other piece begin (an unsnapped breakpoint's estimate)
  join_req_.clear();
  for (size_t k = 1; k < pieces_.size(); ++k) {
    if (!has_sample(pieces_[k].fit.begin_ns)) join_req_.push_back(pieces_[k].fit.begin_ns);
  }
  if (!join_req_.empty()) {
    stage_ = Stage::kJoin;
    pending_ = join_req_;  // ascending (the begins are)
    return;
  }
  enter_fill();
}

void BodyPipeline::step_join() {
  // a piece begins at its join sample's realized time (within 2 q(F) of the estimate, below its timing bound)
  for (size_t r = 0; r < join_req_.size(); ++r) {
    const int64_t want = join_req_[r], got = answers_.t_ns[answer_of(want)];
    for (size_t k = 1; k < pieces_.size(); ++k) {
      FitPiece& f = pieces_[k].fit;
      if (f.begin_ns != want) continue;
      if (got > pieces_[k - 1].fit.begin_ns && got < f.end_ns) {
        f.begin_ns = f.t0_ns = got;
        pieces_[k - 1].fit.end_ns = got;
      }
    }
  }
  insert(answers_);
  join_req_.clear();
  enter_fill();
}

void BodyPipeline::enter_fill() {
  const int64_t h_cap = fopt_.h_cap_ns > 0 ? fopt_.h_cap_ns : 2 * h0_ns(time_, opt_);
  std::vector<int64_t> req;
  const std::vector<int64_t>& ts = set_.fit.t_ns;
  for (size_t k = dirty_; k < pieces_.size(); ++k) {
    Piece& pc = pieces_[k];
    if (!pc.need_fill || pc.fit.hold) continue;
    pc.need_fill = false;
    const int64_t a = pc.fit.begin_ns, c = pc.fit.end_ns;
    if (pc.h_rule == 0) pc.h_rule = knot_spacing(ts, a, c, fopt_.h_floor_ns, h_cap);
    // the knots end at the piece's end (h = ceil(L / ceil(L / h_rule)) <= h_rule): no support past it, where no sample
    // would shape the spline; the fill keeps the rule's spacing
    const int64_t len = c - a, h = pc.h_rule, nseg = (len + h - 1) / h;
    pc.fit.h_ns = (len + nseg - 1) / nseg;
    // every gap of [a, c] (its ends included) at most h / 2, and h / end_fill_div within one knot spacing of its end
    // (and of its begin when no C0 join pins that), where a spline's end controls are seen by few samples: its end
    // pose is what a C0 join hands on (DS d14); at most a quarter of the piece, so a piece shorter than its knot
    // spacing still gets four sample times (its own end belongs to the next piece), which pin the jerk penalty's null
    // space (quadratics)
    const int64_t half = std::max<int64_t>(1, std::min(h / 2, len / 4));
    const int64_t fine = fopt_.end_fill_div > 0 ? std::max<int64_t>(1, std::min(h / fopt_.end_fill_div, len / 4)) : half;
    const bool pinned = k > 0 && pc.fit.join != Continuity::kStep;
    std::vector<int64_t> grid{a, c};
    for (auto it = std::lower_bound(ts.begin(), ts.end(), a); it != ts.end() && *it <= c; ++it) grid.push_back(*it);
    std::sort(grid.begin(), grid.end());
    grid.erase(std::unique(grid.begin(), grid.end()), grid.end());
    for (size_t i = 1; i < grid.size(); ++i) {
      const int64_t g0 = grid[i - 1], gap = grid[i] - g0;
      const int64_t lim = (!pinned && g0 < a + h) || grid[i] > c - h ? fine : half;
      const int64_t n = (gap + lim - 1) / lim;
      for (int64_t j = 1; j < n; ++j) {
        const auto t = g0 + static_cast<int64_t>(static_cast<i128>(gap) * j / n);
        if (!plateaus_.inside(t)) req.push_back(t);
      }
    }
  }
  std::sort(req.begin(), req.end());
  req.erase(std::unique(req.begin(), req.end()), req.end());
  if (!req.empty()) {
    stage_ = Stage::kFill;
    pending_ = std::move(req);
    return;
  }
  enter_validate();
}

void BodyPipeline::enter_validate() {
  std::vector<int64_t> req;
  for (size_t k = dirty_; k < pieces_.size(); ++k) {
    Piece& pc = pieces_[k];
    if (!pc.need_probes) continue;
    pc.need_probes = false;
    const int64_t a = pc.fit.begin_ns, c = pc.fit.end_ns;
    const bool last = k + 1 == pieces_.size();
    // seeded probes, a function of the piece's interval only (so of the oracle's answers): bodies with the same piece
    // share their calls
    uint64_t state = fopt_.probe_seed;
    for (const uint64_t x : {static_cast<uint64_t>(a), static_cast<uint64_t>(c)}) {
      state ^= x;
      state = splitmix64(state);
    }
    const auto range = static_cast<uint64_t>(c - a) + (last ? 1u : 0u);
    for (int i = 0; i < fopt_.probes; ++i) {
      const int64_t t = a + static_cast<int64_t>(splitmix64(state) % range);
      if (!plateaus_.inside(t)) req.push_back(t);
    }
    if (k > 0 && fopt_.breakpoint_probes) {
      if (!has_sample(a)) req.push_back(a);  // the probe at b: the sample there when there is one
      // b-: outside the key's plateau and off the key's own sample on Blender's float32 time grid, as base_requests
      // offsets its pre-key sample (the request realizes within q(F) of its time, so 2 q(F) keeps it clear of both)
      int32_t F = 0;
      float s = 0.0f;
      time_.request(a, &F, &s);
      const int64_t off = std::max<int64_t>(1000, 2 * static_cast<int64_t>(std::ceil(time_.resolution_ns(F))));
      const int64_t left = a - plateau_ns(a) - off;
      if (left > pieces_[k - 1].fit.begin_ns) req.push_back(left);
    }
  }
  std::sort(req.begin(), req.end());
  req.erase(std::unique(req.begin(), req.end()), req.end());
  req.erase(std::remove_if(req.begin(), req.end(),
                           [&](int64_t t) { return has_sample(t) || sample_at(set_.probes, t).has_value(); }),
            req.end());
  if (!req.empty()) {
    stage_ = Stage::kValidate;
    pending_ = std::move(req);
    return;
  }
  validate_pieces();
}

void BodyPipeline::validate_pieces() {
  const double accept = fopt_.accept_frac * opt_.eps_px;
  for (size_t k = dirty_; k < pieces_.size(); ++k) {
    Piece& pc = pieces_[k];
    const FitPiece plan = pc.fit;
    const int64_t a = plan.begin_ns, c = plan.end_ns;
    const bool last = k + 1 == pieces_.size();
    std::optional<Pose> start;
    if (k > 0 && plan.join != Continuity::kStep) start = PieceEval(pieces_[k - 1].fit)(a);
    Track data = slice(set_.fit, a, c, last);
    if (plan.hold) {  // a hold run may be V's samples
      for (size_t i = 0; i < set_.held_out.size(); ++i) {
        const int64_t t = set_.held_out.t_ns[i];
        if (t >= a && (t < c || (last && t == c))) data.push(set_.held_out, i);
      }
      data = sorted_unique(data);
    }
    FitPiece f;
    try {
      f = fit_piece(data, a, c, plan.h_ns, start, plan.hold, fopt_);
    } catch (const LinAlgError&) {  // too few samples for its knots: a failure like any other (halving fills more)
      charge(k, a, 0.0, Failure::kUnderdetermined);
      dirty_ = k;
      return enter_fill();
    }
    f.join = plan.join;
    f.halvings = plan.halvings;
    f.split = plan.split;
    // validation: V and the probes in the piece, the sample at its begin (the probe at b), a hold's own samples
    Track val = slice(set_.held_out, a, c, last);
    const Track pr = slice(set_.probes, a, c, last);
    for (size_t i = 0; i < pr.size(); ++i) val.push(pr, i);
    std::optional<size_t> at_b;
    if (k > 0 && fopt_.breakpoint_probes) {
      if (const auto i = sample_at(set_.fit, a)) {
        at_b = val.size();
        val.push(set_.fit, *i);
      } else {  // a probe at b, already among the probes in [a, c)
        for (size_t v = 0; v < val.size(); ++v) {
          if (val.t_ns[v] == a) at_b = v;
        }
      }
    }
    if (plan.hold) {
      for (size_t i = 0; i < data.size(); ++i) val.push(data, i);
    }
    const std::vector<double> e = piece_errors(f, val);
    double worst = 0.0, worst_px = 0.0, floor = 0.0;
    int64_t worst_t = a;
    bool bad = false;
    for (size_t i = 0; i < e.size(); ++i) {
      floor = std::max(floor, val.floor1[i]);
      f.max_px = std::max(f.max_px, e[i]);
      const double excess = e[i] - (accept + val.floor1[i]);
      if (excess > 0.0 && (!bad || excess > worst)) {
        bad = true;
        worst = excess;
        worst_px = e[i];
        worst_t = val.t_ns[i];
      }
    }
    // C1's pi/2 rule over the piece's consecutive controls (bundle.py _check_increments)
    const bool turns = !plan.hold && max_increment_rad(f) > std::numbers::pi / 2.0;
    if (!bad && !turns) {
      f.floor_px = floor;
      f.validated = static_cast<int64_t>(val.size());
      pc.fit = std::move(f);
      continue;
    }
    // a piece that fails only the pi/2 rule reports the largest increment (rad), not a pixel error
    const bool by_increment = turns && !bad;
    if (by_increment) {
      worst_t = a;
      worst_px = max_increment_rad(f);
    }
    // charge: the left piece when the probe at b fails (its end pose, which a C0 join inherits) or for a hold
    const bool at_b_bad = at_b && e[*at_b] > accept + val.floor1[*at_b];
    const bool left_ok = k > 0 && plan.join != Continuity::kStep && !pieces_[k - 1].fit.hold;
    size_t j = k;
    if ((plan.hold || at_b_bad) && left_ok) j = k - 1;
    if (pieces_[j].fit.hold) {
      throw MotionError("MOT-FIT", "body " + std::to_string(body_) + ": its hold over [" + std::to_string(a) + ", " +
                                       std::to_string(c) + ") ns errs by " + std::to_string(worst_px) + " px at " +
                                       std::to_string(worst_t) + " ns");
    }
    charge(j, worst_t, worst_px, by_increment ? Failure::kIncrement : Failure::kError);
    dirty_ = j;
    return enter_fill();
  }
  finish_fit();
}

void BodyPipeline::charge(size_t j, int64_t t_bad, double value, Failure why) {
  FitPiece& f = pieces_[j].fit;
  // 1. split once with C0 joins at the declared keys inside the piece (MF a7: bezier_auto 3.75e-3 px at its C1 keys).
  // The C0 keys (LINEAR, VECTOR and FREE handles, easing ends, BOUNCE breaks) belong here too: detection classifies a
  // key from the samples on either side of it, and keys closer than about 1.5 frames reach their neighbours' kinks, so
  // it leaves them unclear (no breakpoint); a C0 join at a smooth point costs nothing, and this runs after a failure
  // only, so smooth motion never gets one. A CONSTANT key (hint kStep) is a detected step, not a cut.
  if (!f.split) {
    f.split = true;
    const int64_t margin = std::max<int64_t>(dopt_.merge_ns, 1);
    std::vector<int64_t> cuts;
    for (const DeclaredKey& key : keys_) {
      if (key.hint != Continuity::kStep && key.t_ns > f.begin_ns + margin && key.t_ns < f.end_ns - margin) {
        cuts.push_back(key.t_ns);
      }
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    if (!cuts.empty()) {
      std::vector<Piece> parts;
      int64_t b = f.begin_ns;
      cuts.push_back(f.end_ns);
      for (size_t i = 0; i < cuts.size(); ++i) {
        Piece p;
        p.fit.begin_ns = p.fit.t0_ns = b;
        p.fit.end_ns = cuts[i];
        p.fit.join = i == 0 ? f.join : Continuity::kC0;
        p.fit.split = true;
        parts.push_back(std::move(p));
        b = cuts[i];
      }
      pieces_.erase(pieces_.begin() + static_cast<std::ptrdiff_t>(j));
      pieces_.insert(pieces_.begin() + static_cast<std::ptrdiff_t>(j), parts.begin(), parts.end());
      return;
    }
  }
  // 2. halve the knot spacing (down to 2 h_min); 3. MOT-FIT
  const int64_t h_min = 2 * opt_.h_min_ns;
  Piece& pj = pieces_[j];
  if (f.halvings < fopt_.halvings && pj.h_rule > h_min) {
    pj.h_rule = std::max(pj.h_rule / 2, h_min);
    ++f.halvings;
    pj.need_fill = true;
    return;
  }
  std::string what;
  switch (why) {
    case Failure::kError:
      what = "the fit errs by " + std::to_string(value) + " px at " + std::to_string(t_bad) + " ns, above " +
             std::to_string(fopt_.accept_frac) + " eps + the float32 floor";
      break;
    case Failure::kIncrement:
      what = "the fit's control rotations turn by " + std::to_string(value) + " rad (C1 allows pi/2 at most)";
      break;
    case Failure::kUnderdetermined:
      what = "the piece has too few samples for its knots";
      break;
  }
  throw MotionError("MOT-FIT", "body " + std::to_string(body_) + ": " + what + ", in the piece [" +
                                   std::to_string(f.begin_ns) + ", " + std::to_string(f.end_ns) + ") ns after " +
                                   std::to_string(f.halvings) + " halvings (knot spacing " + std::to_string(f.h_ns) +
                                   " ns)");
}

void BodyPipeline::finish_fit() {
  PiecewiseFit m;
  m.span_begin_ns = t0_;
  m.span_end_ns = t1_;
  for (Piece& pc : pieces_) {
    m.max_validation_px = std::max(m.max_validation_px, pc.fit.max_px);
    m.floor_px = std::max(m.floor_px, pc.fit.floor_px);
    m.validation_samples += pc.fit.validated;
    m.max_join_px = std::max(m.max_join_px, pc.fit.join_px);
    if (pc.fit.hold) m.holds.emplace_back(pc.fit.begin_ns, pc.fit.end_ns);
    m.pieces.push_back(std::move(pc.fit));
  }
  m.breakpoints = set_.breakpoints;
  for (const Track* t : {&set_.fit, &set_.held_out, &set_.probes}) {
    for (const rot::V& p : t->p) m.max_reach_m = std::max(m.max_reach_m, norm(p));
  }
  pieces_.clear();
  set_.motion = std::move(m);
  set_.fitted = true;
  pending_.clear();
  stage_ = Stage::kDone;
}

SampleSet BodyPipeline::take() && {
  if (!done()) throw std::logic_error("BodyPipeline::take before done");
  return std::move(set_);
}

}  // namespace balla::scene::motion
