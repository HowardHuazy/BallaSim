#include "render/thread_pool.h"

namespace balla::render {

ThreadPool::ThreadPool(int threads) {
  try {
    for (int i = 1; i < threads; ++i) workers_.emplace_back([this] { worker_loop(); });
  } catch (...) {
    // A failed constructor never runs ~ThreadPool; join started workers before unwinding members.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    for (std::thread& t : workers_) t.join();
    throw;
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  for (std::thread& t : workers_) t.join();
}

// Claims chunks until none are left; called with the mutex unlocked.
void ThreadPool::run_chunks() {
  for (;;) {
    size_t begin;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (next_ >= n_ || error_) return;
      begin = next_;
      next_ += grain_;
    }
    const size_t end = begin + grain_ < n_ ? begin + grain_ : n_;
    try {
      (*job_)(begin, end);
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!error_) error_ = std::current_exception();
    }
  }
}

void ThreadPool::worker_loop() {
  size_t seen = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
      ++active_;
    }
    run_chunks();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      --active_;
    }
    done_.notify_all();
  }
}

void ThreadPool::parallel_for(size_t n, size_t grain, const std::function<void(size_t, size_t)>& fn) {
  if (n == 0) return;
  if (grain == 0) grain = 1;
  if (workers_.empty() || n <= grain) {
    fn(0, n);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    job_ = &fn;
    n_ = n;
    grain_ = grain;
    next_ = 0;
    error_ = nullptr;
    ++generation_;
  }
  wake_.notify_all();
  run_chunks();
  std::exception_ptr error;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [&] { return active_ == 0 && (next_ >= n_ || error_); });
    job_ = nullptr;
    n_ = 0;
    error = error_;
  }
  if (error) std::rethrow_exception(error);
}

}  // namespace balla::render
