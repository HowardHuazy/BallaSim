// Linux-only startup failure injection against the actual sensor and render pools.
#include <doctest/doctest.h>

#include <dlfcn.h>
#include <pthread.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <system_error>
#include <type_traits>

#include "render/thread_pool.h"
#include "sensor/thread_pool.h"

namespace {

// Only the calling thread creates/joins workers; injection must not affect worker code.
thread_local int fail_after = -1;
thread_local int created = 0;
thread_local int joined = 0;

struct FailureScope {
  std::terminate_handler previous;

  explicit FailureScope(int count) : previous(std::set_terminate([] {
    // A regression should fail the test, without generating a desktop crash report.
    std::fputs("ThreadPool startup called std::terminate\n", stderr);
    std::_Exit(86);
  })) {
    fail_after = count;
    created = joined = 0;
  }

  ~FailureScope() {
    fail_after = -1;
    std::set_terminate(previous);
  }
};

}  // namespace

// Executable symbols override libstdc++'s pthread calls; RTLD_NEXT reaches libc.
extern "C" int pthread_create(pthread_t* thread, const pthread_attr_t* attr,
                              void* (*start)(void*), void* arg) noexcept {
  if (fail_after >= 0 && created == fail_after) return EAGAIN;
  static auto real_create = reinterpret_cast<decltype(&pthread_create)>(dlsym(RTLD_NEXT, "pthread_create"));
  if (!real_create) std::_Exit(87);
  const int result = real_create(thread, attr, start, arg);
  if (result == 0) ++created;
  return result;
}

extern "C" int pthread_join(pthread_t thread, void** result) {
  static auto real_join = reinterpret_cast<decltype(&pthread_join)>(dlsym(RTLD_NEXT, "pthread_join"));
  if (!real_join) std::_Exit(87);
  const int status = real_join(thread, result);
  if (status == 0) ++joined;
  return status;
}

TEST_CASE_TEMPLATE("thread pool startup joins workers and preserves the creation error", Pool,
                   balla::sensor::ThreadPool, balla::render::ThreadPool) {
  for (const int count : {0, 1, 3}) {
    CAPTURE(count);
    FailureScope failure(count);
    bool caught = false;
    try {
      Pool pool(5);
    } catch (const std::system_error& error) {
      caught = true;
      CHECK(error.code() == std::errc::resource_unavailable_try_again);
    }
    CHECK(caught);
    CHECK(created == count);
    CHECK(joined == count);
  }

  // A fresh pool is usable after the failed construction, and normal teardown joins all workers.
  FailureScope success(-1);
  {
    Pool pool(4);
    CHECK(pool.size() == 4);
    std::atomic<size_t> visited{0};
    auto add = [&](size_t begin, size_t end) { visited.fetch_add(end - begin); };
    for (int iteration = 0; iteration < 3; ++iteration) {
      if constexpr (std::is_same_v<Pool, balla::sensor::ThreadPool>) {
        pool.parallel_for(128, 8, [&](int, size_t begin, size_t end) { add(begin, end); });
      } else {
        pool.parallel_for(128, 8, add);
      }
    }
    CHECK(visited == 384);
  }
  CHECK(created == 3);
  CHECK(joined == created);
}
