// Parallel compression of whole chunks or packets (docs/design/phase3/outputs.md §5.3; R-DOC C5). N workers each
// compress one chunk at a time with their own Compressor; the results are emitted on the calling thread in submission
// order, so a container's bytes do not depend on N (F7). At most 2N chunks are in flight; submit() blocks beyond that.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "export/compress.h"
#include "export/errors.h"

namespace balla::exporter {

class ChunkPool {
 public:
  using Emit = std::function<void(std::vector<char>&&)>;
  static constexpr int kMaxThreads = 256;

  // threads in [1, kMaxThreads], else Refused. One thread compresses on the calling thread; codec none never queues.
  ChunkPool(Codec codec, int threads, Lz4Frame frame = Lz4Frame::kStandard) : codec_(codec) {
    if (threads < 1 || threads > kMaxThreads) {
      throw Refused("threads must be in [1, " + std::to_string(kMaxThreads) + "], got " + std::to_string(threads));
    }
    if (codec == Codec::kNone) return;
    if (threads == 1) {
      inline_ = std::make_unique<Compressor>(codec, frame);
      return;
    }
    max_inflight_ = 2 * static_cast<size_t>(threads);
    for (int i = 0; i < threads; ++i) compressors_.push_back(std::make_unique<Compressor>(codec, frame));
    try {
      for (int i = 0; i < threads; ++i) workers_.emplace_back([this, i] { work(*compressors_[i]); });
    } catch (...) {
      stop();
      throw;
    }
  }
  ChunkPool(const ChunkPool&) = delete;
  ChunkPool& operator=(const ChunkPool&) = delete;
  // Stops the workers; chunks not yet emitted are dropped (drain() emits them).
  ~ChunkPool() { stop(); }

  // Queues one chunk; emit(compressed) runs later on this thread, after every earlier chunk's emit. Emits the finished
  // chunks at the head of the queue before returning. An exception from a worker or from emit is rethrown here or in
  // drain(), at that chunk's turn.
  void submit(std::vector<char> raw, Emit emit) {
    if (codec_ == Codec::kNone) {
      emit(std::move(raw));
      return;
    }
    if (inline_) {
      std::vector<char> out;
      inline_->compress(raw.data(), raw.size(), out);
      emit(std::move(out));
      return;
    }
    auto job = std::make_unique<Job>();
    job->raw = std::move(raw);
    job->emit = std::move(emit);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      inflight_.push_back(std::move(job));  // owned first: a worker never sees a Job that a failed push frees
      try {
        todo_.push_back(inflight_.back().get());
      } catch (...) {
        inflight_.pop_back();
        throw;
      }
    }
    work_.notify_one();
    emit_until(max_inflight_);
  }

  // Emits every queued chunk, in order.
  void drain() { emit_until(0); }

  Codec codec() const { return codec_; }

 private:
  struct Job {
    std::vector<char> raw, out;
    Emit emit;
    std::exception_ptr error;
    bool done = false;
  };

  void work(Compressor& c) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      work_.wait(lock, [this] { return stop_ || !todo_.empty(); });
      if (stop_) return;
      Job* job = todo_.front();
      todo_.pop_front();
      lock.unlock();
      try {
        c.compress(job->raw.data(), job->raw.size(), job->out);
      } catch (...) {
        job->error = std::current_exception();
      }
      std::vector<char>().swap(job->raw);
      lock.lock();
      job->done = true;
      done_.notify_all();
    }
  }

  // Emits finished chunks from the head; while more than `keep` remain, waits for the head.
  void emit_until(size_t keep) {
    for (;;) {
      std::unique_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (inflight_.empty()) return;
        if (!inflight_.front()->done) {
          if (inflight_.size() <= keep) return;
          done_.wait(lock, [this] { return inflight_.front()->done; });
        }
        job = std::move(inflight_.front());
        inflight_.pop_front();
      }
      if (job->error) std::rethrow_exception(job->error);
      job->emit(std::move(job->out));
    }
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      todo_.clear();
    }
    work_.notify_all();
    for (std::thread& t : workers_) t.join();
    workers_.clear();
  }

  Codec codec_;
  size_t max_inflight_ = 0;
  std::unique_ptr<Compressor> inline_;
  std::vector<std::unique_ptr<Compressor>> compressors_;
  std::deque<std::unique_ptr<Job>> inflight_;  // submission order; read and popped by the calling thread only
  std::deque<Job*> todo_;                      // not yet taken by a worker
  std::mutex mutex_;
  std::condition_variable work_, done_;
  std::vector<std::thread> workers_;
  bool stop_ = false;
};

}  // namespace balla::exporter
