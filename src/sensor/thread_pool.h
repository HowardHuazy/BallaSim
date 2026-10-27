// A persistent pool of worker threads for the CPU sensor kernel.
//
// parallel_for hands out contiguous blocks of items through an atomic counter (dynamic load
// balance: pixels under a moving edge cost far more than static ones). Results never depend on
// which thread runs which block, because every pixel is independent and events are sorted
// afterwards.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace balla::sensor {

class ThreadPool {
 public:
  // num_threads <= 0 selects std::thread::hardware_concurrency().
  explicit ThreadPool(int num_threads);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return static_cast<int>(workers_.size()) + 1; }

  // Calls fn(worker, begin, end) over [0, n) in blocks of `block`; worker in [0, size()). The calling
  // thread participates as worker 0. Exceptions thrown by fn are rethrown here.
  void parallel_for(size_t n, size_t block, const std::function<void(int, size_t, size_t)>& fn);

 private:
  void worker_loop(int id);
  void run_blocks(int worker);

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable start_cv_, done_cv_;
  uint64_t generation_ = 0;
  int pending_ = 0;
  bool stop_ = false;
  const std::function<void(int, size_t, size_t)>* job_ = nullptr;
  size_t n_ = 0, block_ = 1;
  std::atomic<size_t> next_{0};
  std::exception_ptr error_;
};

}  // namespace balla::sensor
