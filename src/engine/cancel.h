// The cooperative cancel of a run (docs/design/phase3/blender.md §3.8, §5.1; PHASE3 D-P3-9 and D-P3-14; job protocol
// C9): a token that a front end sets from any thread (the job runner's supervisor on a signal or the job's cancel file,
// a UI worker), read without the GIL at the engine's check points: each pixel batch and refinement round of a window
// (WindowEngine::run), and in a camera's start() and step() before their work and before sensing. A set token makes
// the check point throw Cancelled; an unset or absent one changes nothing. Abort path only: the camera's state after
// Cancelled is undefined (its run is abandoned; start() it again or drop the engine), and runs that complete are those
// of a run without a token.
#pragma once

#include <atomic>
#include <stdexcept>
#include <string>

namespace balla::engine {

struct CancelToken {
  std::atomic<bool> flag{false};
  void set() noexcept { flag.store(true, std::memory_order_relaxed); }
  bool is_set() const noexcept { return flag.load(std::memory_order_relaxed); }
};

class Cancelled : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Throws Cancelled ("<where>: cancelled") when the token is set; nothing without a token.
inline void check_cancel(const CancelToken* token, const char* where) {
  if (token != nullptr && token->is_set()) throw Cancelled(std::string(where) + ": cancelled");
}

}  // namespace balla::engine
