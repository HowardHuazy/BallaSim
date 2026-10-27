// The round driver and the recorded oracle (pipeline.h; blender.md §3.5, §3.6, §3.8).
#include <algorithm>
#include <bit>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

#include "scene/motion/pipeline.h"

namespace balla::scene::motion {

namespace {

bool same_request(const RawPoses& r, std::span<const int32_t> F, std::span<const float> s) {
  return r.F.size() == F.size() && r.s.size() == s.size() && std::equal(F.begin(), F.end(), r.F.begin()) &&
         (s.empty() || std::memcmp(s.data(), r.s.data(), s.size() * sizeof(float)) == 0);
}

}  // namespace

RoundStats drive(const Oracle& oracle, const TimeMap& time, const ErrorModel& model, std::span<BodyPipeline> pipes,
                 render::ThreadPool& pool, const std::function<void(double, std::string_view)>& progress,
                 const std::function<void(const PoseBatch&)>& on_batch,
                 const std::function<std::string(int32_t)>& name_of) {
  const MotionScene& scene = model.scene();
  const auto nb = static_cast<int32_t>(scene.bodies.size());
  int max_per_message = 512;  // the protocol's cap (§3.5), lowered by any pipeline's options
  for (const BodyPipeline& p : pipes) {
    if (p.body() < 0 || p.body() >= nb) throw std::invalid_argument("drive: a pipeline's body is not a column");
    max_per_message = std::min(max_per_message, p.options().max_per_message);
  }
  if (max_per_message < 1) throw std::invalid_argument("drive: max_per_message must be >= 1");
  RoundStats stats;
  for (;;) {
    // 1. this round's requests: every unfinished pipeline's, merged (each time once, ascending)
    std::vector<std::vector<int64_t>> req(pipes.size());
    std::vector<int64_t> all;
    for (size_t k = 0; k < pipes.size(); ++k) {
      if (pipes[k].done()) continue;
      req[k] = pipes[k].requests();
      all.insert(all.end(), req[k].begin(), req[k].end());
    }
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    if (all.empty()) break;
    if (stats.rounds >= 100000) throw std::logic_error("drive: the pipelines do not finish");
    // 2. (F, s) per time; equal requests are adjacent (request is monotone) and become one call
    std::vector<size_t> call_of(all.size());
    std::vector<int32_t> cF;
    std::vector<float> cS;
    for (size_t i = 0; i < all.size(); ++i) {
      int32_t F = 0;
      float s = 0.0f;
      time.request(all[i], &F, &s);
      if (cF.empty() || F != cF.back() || std::bit_cast<uint32_t>(s) != std::bit_cast<uint32_t>(cS.back())) {
        cF.push_back(F);
        cS.push_back(s);
      }
      call_of[i] = cF.size() - 1;
    }
    auto call_at = [&](int64_t t) {
      return call_of[static_cast<size_t>(std::lower_bound(all.begin(), all.end(), t) - all.begin())];
    };
    std::vector<const char*> stage_of(cF.size(), nullptr);
    for (size_t k = 0; k < pipes.size(); ++k) {
      for (const int64_t t : req[k]) {
        const size_t c = call_at(t);
        if (!stage_of[c]) stage_of[c] = stage_name(pipes[k].stage());
      }
    }
    for (const char* s : stage_of) ++stats.calls_by_stage[s];
    // 3. the oracle, in messages of at most max_per_message samples; decode and weigh
    PoseBatch batch;
    batch.bodies = nb;
    for (size_t off = 0; off < cF.size(); off += static_cast<size_t>(max_per_message)) {
      const size_t cnt = std::min(cF.size() - off, static_cast<size_t>(max_per_message));
      const std::span<const int32_t> F(cF.data() + off, cnt);
      const std::span<const float> S(cS.data() + off, cnt);
      RawPoses raw = oracle(F, S);
      if (raw.F.empty() && raw.s.empty()) {
        raw.F.assign(F.begin(), F.end());
        raw.s.assign(S.begin(), S.end());
      } else if (!same_request(raw, F, S)) {
        throw MotionError("BRG-PROTO", "a poses answer echoes other requests than were sent");
      }
      if (raw.n != static_cast<int32_t>(cnt)) throw MotionError("BRG-PROTO", "a poses answer has the wrong length");
      PoseBatch decoded = decode(raw, time, scene);  // checks the answer's shape first
      for (size_t i = 0; i < cnt; ++i) {
        if (raw.unknown[i] != 0) {
          throw MotionError("VIS-ANIM", std::to_string(raw.unknown[i]) + " render-visible instances not in the "
                                        "exchange at frame " + std::to_string(raw.F[i]) +
                                            (raw.unknown_keys.empty() ? std::string() : " (" + raw.unknown_keys + ")"));
        }
      }
      for (size_t k = 0; k < raw.present.size(); ++k) {
        if (!raw.present[k]) {
          const auto bodies = static_cast<size_t>(nb);
          throw MotionError("VIS-ANIM", "body column " + std::to_string(k % bodies) + " is absent at frame " +
                                            std::to_string(raw.F[k / bodies]));
        }
      }
      batch.append(decoded);
      ++stats.messages;
    }
    model.weigh(batch);
    if (on_batch) on_batch(batch);
    stats.calls += static_cast<int64_t>(cF.size());
    ++stats.rounds;
    // 4. each pipeline's own column at its own times, then its step, on the pool
    std::vector<std::exception_ptr> errors(pipes.size());
    pool.parallel_for(pipes.size(), 1, [&](size_t k0, size_t k1) {
      for (size_t k = k0; k < k1; ++k) {
        if (req[k].empty()) continue;
        try {
          Track answers;
          for (const int64_t t : req[k]) answers.push(batch, call_at(t), pipes[k].body());
          pipes[k].feed(answers);
          pipes[k].step();
        } catch (const MotionError& e) {
          // name the body that failed (the pipeline knows only its column)
          errors[k] = name_of ? std::make_exception_ptr(MotionError(e.code, std::string(e.what()).substr(e.code.size() + 2) +
                                                                     " (body " + name_of(pipes[k].body()) + ")"))
                              : std::current_exception();
        } catch (...) {
          errors[k] = std::current_exception();
        }
      }
    });
    for (const std::exception_ptr& e : errors) {
      if (e) std::rethrow_exception(e);
    }
    if (progress) {
      const auto finished = std::count_if(pipes.begin(), pipes.end(), [](const BodyPipeline& p) { return p.done(); });
      progress(pipes.empty() ? 1.0 : static_cast<double>(finished) / static_cast<double>(pipes.size()), "sampling");
    }
  }
  return stats;
}

Oracle recording(Oracle inner, OracleLog& log) {
  return [inner = std::move(inner), &log](std::span<const int32_t> F, std::span<const float> s) {
    RawPoses r = inner(F, s);
    if (r.F.empty() && r.s.empty()) {
      r.F.assign(F.begin(), F.end());
      r.s.assign(s.begin(), s.end());
    }
    log.calls.push_back(r);
    return r;
  };
}

Oracle replay(const OracleLog& log, std::shared_ptr<size_t> calls) {
  if (!calls) throw std::invalid_argument("replay: no call counter");
  return [&log, calls](std::span<const int32_t> F, std::span<const float> s) {
    const size_t k = *calls;
    if (k >= log.calls.size()) {
      throw MotionError("BRG-REPLAY", "call " + std::to_string(k) + " goes past the end of the recorded log");
    }
    if (!same_request(log.calls[k], F, s)) {
      throw MotionError("BRG-REPLAY", "call " + std::to_string(k) + " requests other (F, s) than the recorded log");
    }
    ++*calls;
    return log.calls[k];
  };
}

}  // namespace balla::scene::motion
