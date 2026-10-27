// Serial host versions of the engine's primitives (par.h): standard algorithms, ENGINE_CPP §1.7.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <numeric>
#include <string>
#include <vector>

#include <unistd.h>

#include "engine/par.h"

namespace balla::engine {

template <>
void* allocate<Space::kHost>(size_t bytes) {
  void* p = std::malloc(std::max<size_t>(bytes, 1));
  if (p == nullptr) throw std::bad_alloc();
  return p;
}

template <>
void release<Space::kHost>(void* p) {
  std::free(p);
}

template <>
void* allocate_host<Space::kHost>(size_t bytes) {
  return allocate<Space::kHost>(bytes);
}

template <>
void release_host<Space::kHost>(void* p) {
  std::free(p);
}

using Host = Par<Space::kHost>;

void Host::exclusive_scan(const uint32_t* in, uint32_t* out, size_t n, Temp<Space::kHost>&, Stream) {
  std::exclusive_scan(in, in + n, out, uint32_t{0});
}

void Host::exclusive_scan(const uint64_t* in, uint64_t* out, size_t n, Temp<Space::kHost>&, Stream) {
  std::exclusive_scan(in, in + n, out, uint64_t{0});
}

bool Host::sort_pairs(uint64_t* keys, uint64_t* keys_alt, uint32_t* vals, uint32_t* vals_alt, size_t n, int end_bit,
                      Temp<Space::kHost>&, Stream) {
  const uint64_t mask = end_bit >= 64 ? ~uint64_t{0} : (uint64_t{1} << end_bit) - 1;
  std::vector<uint32_t> order(n);
  std::iota(order.begin(), order.end(), uint32_t{0});
  std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return (keys[a] & mask) < (keys[b] & mask); });
  for (size_t i = 0; i < n; ++i) {
    keys_alt[i] = keys[order[i]];
    vals_alt[i] = vals[order[i]];
  }
  return true;
}

void Host::select_flagged(const uint8_t* flags, uint32_t* out, size_t n, unsigned long long* count,
                          Temp<Space::kHost>&, Stream) {
  size_t m = 0;
  for (size_t i = 0; i < n; ++i) {
    if (flags[i] != 0) out[m++] = static_cast<uint32_t>(i);
  }
  *count = m;
}

void Host::copy(void* dst, const void* src, size_t bytes, Stream) {
  if (bytes > 0) std::memmove(dst, src, bytes);
}

void Host::zero(void* dst, size_t bytes, Stream) {
  if (bytes > 0) std::memset(dst, 0, bytes);
}

void Host::upload(void* dst, const void* host, size_t bytes, Stream s) { copy(dst, host, bytes, s); }

void Host::download(void* host, const void* src, size_t bytes, Stream s) { copy(host, src, bytes, s); }

void Host::download_async(void* host, const void* src, size_t bytes, Stream s) { copy(host, src, bytes, s); }

// Linux: MemAvailable (the kernel's estimate of the memory new allocations get without swapping, the
// reclaimable page cache included); elsewhere the physical memory.
size_t Host::free_bytes() {
  std::ifstream meminfo("/proc/meminfo");
  std::string key;
  size_t kib = 0;
  while (meminfo >> key >> kib) {
    if (key == "MemAvailable:") return kib * 1024;
    meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
  return pages > 0 && page > 0 ? static_cast<size_t>(pages) * static_cast<size_t>(page) : size_t{1} << 32;
}

}  // namespace balla::engine
