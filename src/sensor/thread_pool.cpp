#include "sensor/thread_pool.h"

#include <algorithm>

namespace balla::sensor {

ThreadPool::ThreadPool(int num_threads) {
  if (num_threads <= 0) num_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  try {
    for (int i = 1; i < num_threads; ++i) workers_.emplace_back([this, i] { worker_loop(i); });
  } catch (...) {
    // A failed constructor never runs ~ThreadPool; join started workers before unwinding members.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    start_cv_.notify_all();
    for (std::thread& t : workers_) t.join();
    throw;
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  start_cv_.notify_all();
  for (std::thread& t : workers_) t.join();
}

void ThreadPool::run_blocks(int worker) {
  while (true) {
    const size_t begin = next_.fetch_add(block_);
    if (begin >= n_) return;
    const size_t end = std::min(n_, begin + block_);
    try {
      (*job_)(worker, begin, end);
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!error_) error_ = std::current_exception();
      next_.store(n_);  // abandon the remaining blocks
      return;
    }
  }
}

void ThreadPool::worker_loop(int id) {
  uint64_t seen = 0;
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      start_cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
    }
    run_blocks(id);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (--pending_ == 0) done_cv_.notify_one();
    }
  }
}

void ThreadPool::parallel_for(size_t n, size_t block, const std::function<void(int, size_t, size_t)>& fn) {
  if (n == 0) return;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    job_ = &fn;
    n_ = n;
    block_ = std::max<size_t>(1, block);
    next_.store(0);
    error_ = nullptr;
    pending_ = static_cast<int>(workers_.size());
    ++generation_;
  }
  start_cv_.notify_all();
  run_blocks(0);
  std::unique_lock<std::mutex> lock(mutex_);
  done_cv_.wait(lock, [&] { return pending_ == 0; });
  job_ = nullptr;
  if (error_) std::rethrow_exception(error_);
}

}  // namespace balla::sensor
