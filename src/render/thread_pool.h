// Fixed std::thread pool for the CPU paths (DESIGN §4.2: OpenMP is avoided because Apple clang
// lacks it). Work is split into fixed-size index chunks and every index writes only its own output
// slots, so results never depend on thread scheduling (S1 C6).
#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace balla::render {

class ThreadPool {
 public:
  // threads <= 1 runs everything on the calling thread.
  explicit ThreadPool(int threads);
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ~ThreadPool();

  int size() const { return static_cast<int>(workers_.size()) + 1; }

  // Calls fn(begin, end) on chunks [i * grain, min(n, (i + 1) * grain)) of [0, n) and blocks until
  // all are done. The calling thread works too. The first exception thrown by fn is rethrown here
  // after every running chunk has finished.
  void parallel_for(size_t n, size_t grain, const std::function<void(size_t, size_t)>& fn);

 private:
  void worker_loop();
  void run_chunks();

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable wake_, done_;
  const std::function<void(size_t, size_t)>* job_ = nullptr;
  size_t n_ = 0, grain_ = 1, next_ = 0;
  size_t generation_ = 0;
  int active_ = 0;
  bool stop_ = false;
  std::exception_ptr error_;
};

}  // namespace balla::render
