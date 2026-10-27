// Data-parallel primitives of the engine over one memory space (ENGINE_CPP §1.1, §1.10): a grow-only
// Buffer and four primitives, for_each, exclusive_scan, sort_pairs (stable) and select_flagged. The
// device versions run CUB and functor kernels on a CUDA stream (par_cuda.cu); the host versions are
// serial standard algorithms (par_host.cpp, §1.7). All are deterministic: exact integer scans, a
// stable sort, an order-preserving select, and integer atomics only.
//
// The primitives' temporary storage (CUB's) is a Temp buffer owned by the caller, so that it is
// counted in the caller's memory budget and freed with it.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/hd.h"

#if defined(__CUDACC__)
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#endif

namespace balla::engine {

enum class Space { kHost, kDevice };
using Stream = void*;  // a cudaStream_t on the device, unused on the host

template <Space S>
void* allocate(size_t bytes);
template <Space S>
void release(void* p);
template <>
void* allocate<Space::kHost>(size_t bytes);
template <>
void release<Space::kHost>(void* p);
template <>
void* allocate<Space::kDevice>(size_t bytes);
template <>
void release<Space::kDevice>(void* p);
// Host memory that device work can write (pinned on the device space, malloc on the host).
template <Space S>
void* allocate_host(size_t bytes);
template <Space S>
void release_host(void* p);
template <>
void* allocate_host<Space::kHost>(size_t bytes);
template <>
void release_host<Space::kHost>(void* p);
template <>
void* allocate_host<Space::kDevice>(size_t bytes);
template <>
void release_host<Space::kDevice>(void* p);

// Bytes held by all Buffers of a space, and their high-water mark since the last reset (the memory
// budget's tests).
template <Space S>
struct BufferMeter {
  inline static std::atomic<size_t> live{0}, peak{0};
  static void add(size_t bytes) {
    const size_t now = live.fetch_add(bytes) + bytes;
    size_t p = peak.load();
    while (now > p && !peak.compare_exchange_weak(p, now)) {
    }
  }
  static void sub(size_t bytes) { live.fetch_sub(bytes); }
  static void reset_peak() { peak.store(live.load()); }
};

// Grow-only allocation of trivially copyable T; its contents are lost when it grows (the old
// allocation is freed before the new one is made).
template <class T, Space S>
class Buffer {
 public:
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { reset(); }
  T* ensure(size_t n) {
    if (n > capacity_) {
      reset();
      ptr_ = static_cast<T*>(allocate<S>(n * sizeof(T)));
      capacity_ = n;
      BufferMeter<S>::add(bytes());
    }
    return ptr_;
  }
  void reset() {
    if (ptr_ != nullptr) {
      release<S>(ptr_);
      BufferMeter<S>::sub(bytes());
    }
    ptr_ = nullptr;
    capacity_ = 0;
  }
  void swap(Buffer& o) noexcept {
    std::swap(ptr_, o.ptr_);
    std::swap(capacity_, o.capacity_);
  }
  T* data() const { return ptr_; }
  size_t capacity() const { return capacity_; }
  size_t bytes() const { return capacity_ * sizeof(T); }

 private:
  T* ptr_ = nullptr;
  size_t capacity_ = 0;
};

template <Space S>
using Temp = Buffer<uint8_t, S>;

// A few host words that asynchronous downloads and selects write, valid after the next sync of
// their stream (pinned on the device, so that one sync covers several read-backs).
template <Space S>
class HostWords {
 public:
  static constexpr size_t kWords = 4;
  HostWords() : p_(static_cast<unsigned long long*>(allocate_host<S>(kWords * sizeof(unsigned long long)))) {}
  HostWords(const HostWords&) = delete;
  HostWords& operator=(const HostWords&) = delete;
  ~HostWords() { release_host<S>(p_); }
  unsigned long long* data() const { return p_; }

 private:
  unsigned long long* p_;
};

template <Space S>
struct Par;

// The primitives; Par<Space::kDevice> has the same members (below, for CUDA translation units).
template <>
struct Par<Space::kHost> {
  template <class F>
  static void for_each(size_t n, const F& f, Stream) {
    for (size_t i = 0; i < n; ++i) f(i);
  }
  static void exclusive_scan(const uint32_t* in, uint32_t* out, size_t n, Temp<Space::kHost>&, Stream);
  static void exclusive_scan(const uint64_t* in, uint64_t* out, size_t n, Temp<Space::kHost>&, Stream);
  // Stable sort of the pairs (keys[i], vals[i]) by the key bits [0, end_bit), passing through
  // keys_alt and vals_alt; all four arrays are overwritten. Returns true when the sorted pairs are
  // in keys_alt and vals_alt, false when they are in keys and vals.
  static bool sort_pairs(uint64_t* keys, uint64_t* keys_alt, uint32_t* vals, uint32_t* vals_alt, size_t n,
                         int end_bit, Temp<Space::kHost>&, Stream);
  // The indices i < n with flags[i] != 0, ascending (NumPy's flatnonzero), into out; their count goes
  // to *count (HostWords memory), valid after the next sync.
  static void select_flagged(const uint8_t* flags, uint32_t* out, size_t n, unsigned long long* count,
                             Temp<Space::kHost>&, Stream);
  // Temporary bytes the calls above need for n elements.
  static size_t scan_temp_bytes(size_t) { return 0; }
  static size_t sort_temp_bytes(size_t, int) { return 0; }
  static size_t select_temp_bytes(size_t) { return 0; }
  static void copy(void* dst, const void* src, size_t bytes, Stream);  // within the space
  static void zero(void* dst, size_t bytes, Stream);
  static void upload(void* dst, const void* host, size_t bytes, Stream);
  static void download(void* host, const void* src, size_t bytes, Stream);        // synchronous
  static void download_async(void* host, const void* src, size_t bytes, Stream);  // host: HostWords
  static void sync(Stream) {}
  // The engine's stream (none on the host) and the memory its default budget takes 70 % of (on the
  // host: the available memory, MemAvailable on Linux; the physical memory elsewhere).
  static Stream create_stream() { return nullptr; }
  static void destroy_stream(Stream) {}
  static size_t free_bytes();
};

#if defined(__CUDACC__)
template <class F>
__global__ void for_each_kernel(size_t n, F f) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) f(i);
}

void check_launch(const char* what);

template <>
struct Par<Space::kDevice> {
  template <class F>
  static void for_each(size_t n, const F& f, Stream stream) {
    if (n == 0) return;
    constexpr unsigned kBlock = 256;
    for_each_kernel<<<static_cast<unsigned>((n + kBlock - 1) / kBlock), kBlock, 0, static_cast<cudaStream_t>(stream)>>>(
        n, f);
    check_launch("for_each");
  }
  using T = Temp<Space::kDevice>;
  static void exclusive_scan(const uint32_t* in, uint32_t* out, size_t n, T& temp, Stream);
  static void exclusive_scan(const uint64_t* in, uint64_t* out, size_t n, T& temp, Stream);
  static bool sort_pairs(uint64_t* keys, uint64_t* keys_alt, uint32_t* vals, uint32_t* vals_alt, size_t n,
                         int end_bit, T& temp, Stream);
  static void select_flagged(const uint8_t* flags, uint32_t* out, size_t n, unsigned long long* count, T& temp,
                             Stream);
  static size_t scan_temp_bytes(size_t n);
  static size_t sort_temp_bytes(size_t n, int end_bit);
  static size_t select_temp_bytes(size_t n);
  static void copy(void* dst, const void* src, size_t bytes, Stream);
  static void zero(void* dst, size_t bytes, Stream);
  static void upload(void* dst, const void* host, size_t bytes, Stream);
  static void download(void* host, const void* src, size_t bytes, Stream);
  static void download_async(void* host, const void* src, size_t bytes, Stream);
  static void sync(Stream);
  static Stream create_stream();  // non-blocking
  static void destroy_stream(Stream);
  static size_t free_bytes();  // device memory free now
};
#endif

// Adds pred to an integer counter shared by the iterations of a for_each (warp-aggregated on the
// device, so the sum is exact and independent of thread order).
BALLA_HD inline void count_if(unsigned long long* counter, bool pred) {
#if defined(__CUDA_ARCH__)
  const cooperative_groups::coalesced_group g = cooperative_groups::coalesced_threads();
  const unsigned votes = __popc(g.ballot(pred));
  if (g.thread_rank() == 0 && votes != 0) atomicAdd(counter, static_cast<unsigned long long>(votes));
#else
  *counter += pred ? 1u : 0u;
#endif
}

// Adds v to an integer counter shared by the iterations of a for_each (warp-aggregated on the device; exact).
BALLA_HD inline void add_count(unsigned long long* counter, unsigned long long v) {
#if defined(__CUDA_ARCH__)
  const cooperative_groups::coalesced_group g = cooperative_groups::coalesced_threads();
  const unsigned long long sum = cooperative_groups::reduce(g, v, cooperative_groups::plus<unsigned long long>());
  if (g.thread_rank() == 0 && sum != 0) atomicAdd(counter, sum);
#else
  *counter += v;
#endif
}

BALLA_HD inline void atomic_increment(uint32_t* p) {
#if defined(__CUDA_ARCH__)
  atomicAdd(p, 1u);
#else
  ++*p;
#endif
}

// Uploads v into b (grown as needed, to at least one element, so that the pointer is never null) and
// returns b's data.
template <class T, Space S>
const T* upload_vector(Buffer<T, S>& b, const std::vector<T>& v, Stream stream) {
  Par<S>::upload(b.ensure(v.empty() ? 1 : v.size()), v.data(), v.size() * sizeof(T), stream);
  return b.data();
}

}  // namespace balla::engine
