// A minimal FlatBuffers builder for the AEDAT4 writer (docs/design/phase3/outputs.md §4.8; docs/dev/EXPORT.md).
// Provenance: an independent implementation of the layout FlatBuffers' FlatBufferBuilder (google/flatbuffers,
// Apache-2.0) produces, ported from the R-FMT prototype's FBB class (phase3_research/outputs/formats/scripts/
// aedat4.py), which emulates FlatBufferBuilder 1.10: the buffer is built back to front, every scalar aligned to its
// size from the buffer's end (PreAlign), each table's vtable written before it and shared with an identical earlier
// one (vtable dedup), defaults left out unless the builder forces them (ForceDefaults), and Finish adding the file
// identifier, the root offset and, optionally, the size prefix. No FlatBuffers code is used (THIRD_PARTY_NOTICES.md,
// "Ported code"). The bytes equal dv-processing 2.0.4's writer's for the AEDAT4 tables (tests/cpp/export).
// Header-only.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "export/errors.h"

namespace balla::exporter::fb {

// FlatBuffers' largest buffer (FLATBUFFERS_MAX_BUFFER_SIZE): its offsets are signed 32-bit.
inline constexpr size_t kMaxBytes = 0x7FFFFFFF;

class Builder {
 public:
  // force_defaults: write scalars equal to their default too (ForceDefaults; the AEDAT4 IOHeader, so its size does not
  // change when it is patched). reserve: the first buffer size.
  explicit Builder(bool force_defaults = false, size_t reserve = 1024) : force_(force_defaults), buf_(reserve) {}

  // Bytes written so far. Offsets are distances from the buffer's end, as FlatBuffers' GetSize().
  uint32_t size() const { return static_cast<uint32_t>(used_); }

  // Zero padding so that, after len more bytes, the size is a multiple of alignment (a power of two).
  void prealign(size_t len, size_t alignment) {
    minalign_ = std::max(minalign_, alignment);
    pad((alignment - ((used_ + len) & (alignment - 1))) & (alignment - 1));
  }
  void align(size_t alignment) { prealign(0, alignment); }

  // A scalar aligned to its size; returns its offset.
  template <class T>
  uint32_t push(T v) {
    align(sizeof(T));
    std::memcpy(make(sizeof(T)), &v, sizeof(T));
    return size();
  }

  // The uoffset from the next 4-byte slot to off.
  uint32_t refer_to(uint32_t off) {
    align(4);
    return size() - off + 4;
  }

  // A string: u32 length, the bytes and a NUL, 4-byte aligned.
  uint32_t create_string(std::string_view s) {
    prealign(s.size() + 1, 4);
    pad(1);
    std::memcpy(make(s.size()), s.data(), s.size());
    return push(static_cast<uint32_t>(s.size()));
  }

  // A vector of n structs of elem_size bytes and elem_align alignment: returns where the caller writes them, front to
  // back (valid until the next call), then end_vector(n).
  char* start_struct_vector(size_t n, size_t elem_size, size_t elem_align) {
    prealign(n * elem_size, 4);
    prealign(n * elem_size, elem_align);
    return make(n * elem_size);
  }
  uint32_t end_vector(size_t n) { return push(static_cast<uint32_t>(n)); }

  // A vector of table offsets, in the given order.
  uint32_t create_offset_vector(const std::vector<uint32_t>& offs) {
    prealign(4 * offs.size(), 4);
    for (auto it = offs.rbegin(); it != offs.rend(); ++it) push(refer_to(*it));
    return end_vector(offs.size());
  }

  uint32_t start_table() {
    fields_.clear();
    max_voffset_ = 0;
    return size();
  }

  // Field `id` of the open table (FlatBuffers' AddElement): left out when equal to its default (a float -0.0 equals
  // 0.0, as in FlatBuffers) unless the builder forces defaults.
  template <class T>
  void add_scalar(int id, T v, T def) {
    if (v == def && !force_) return;
    track(id, push(v));
  }
  void add_offset(int id, uint32_t off) {
    if (off == 0) return;
    track(id, push(refer_to(off)));
  }
  void add_struct(int id, const void* data, size_t n, size_t alignment) {
    align(alignment);
    std::memcpy(make(n), data, n);
    track(id, size());
  }

  // Closes the open table: its soffset, then its vtable (size, table size, field offsets), shared with an identical
  // vtable written earlier in this buffer. Returns the table's offset.
  uint32_t end_table(uint32_t start) {
    const uint32_t loc = push<int32_t>(0);
    const uint16_t vsize = std::max<uint16_t>(static_cast<uint16_t>(max_voffset_ + 2), 4);
    if (loc - start > 0xFFFF) throw Refused("flatbuffers: a table larger than 64 KiB");
    pad(vsize);
    char* vt = front();
    store<uint16_t>(vt, vsize);
    store<uint16_t>(vt + 2, static_cast<uint16_t>(loc - start));
    for (const auto& [off, voffset] : fields_) store<uint16_t>(vt + voffset, static_cast<uint16_t>(loc - off));
    fields_.clear();
    max_voffset_ = 0;
    uint32_t use = size();
    for (const uint32_t v : vtables_) {
      const char* old = at(v);
      if (load<uint16_t>(old) == vsize && std::memcmp(old, vt, vsize) == 0) {
        use = v;
        used_ = loc;  // drops the new vtable
        break;
      }
    }
    if (use == size()) vtables_.push_back(use);
    store<int32_t>(at(loc), static_cast<int32_t>(use) - static_cast<int32_t>(loc));
    return loc;
  }

  // The finished buffer, front to back: [u32 size prefix] u32 root offset [4-byte identifier] ..., the whole aligned
  // to the largest alignment used. The builder is empty afterwards.
  std::vector<char> finish(uint32_t root, std::string_view identifier, bool size_prefix) {
    vtables_.clear();
    prealign((size_prefix ? 4 : 0) + 4 + identifier.size(), minalign_);
    if (!identifier.empty()) std::memcpy(make(identifier.size()), identifier.data(), identifier.size());
    push(refer_to(root));
    if (size_prefix) push(size());
    std::memmove(buf_.data(), front(), used_);
    buf_.resize(used_);
    std::vector<char> out = std::move(buf_);
    buf_.clear();
    used_ = 0;
    minalign_ = 1;
    return out;
  }

 private:
  template <class T>
  static void store(char* p, T v) {
    std::memcpy(p, &v, sizeof(T));
  }
  template <class T>
  static T load(const char* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
  }

  char* front() { return buf_.data() + (buf_.size() - used_); }
  char* at(uint32_t off) { return buf_.data() + (buf_.size() - off); }  // the item pushed when size() was off

  // n new bytes at the front.
  char* make(size_t n) {
    if (n > buf_.size() - used_) {
      if (n > kMaxBytes - used_) throw Refused("flatbuffers: a buffer larger than 2 GiB");
      std::vector<char> grown(std::max({2 * buf_.size(), used_ + n, size_t{256}}));
      std::memcpy(grown.data() + (grown.size() - used_), front(), used_);
      buf_.swap(grown);
    }
    used_ += n;
    return front();
  }
  void pad(size_t n) { std::memset(make(n), 0, n); }

  void track(int id, uint32_t off) {
    const auto voffset = static_cast<uint16_t>(2 * (id + 2));
    fields_.emplace_back(off, voffset);
    max_voffset_ = std::max(max_voffset_, voffset);
  }

  bool force_;
  std::vector<char> buf_;  // the data occupies the last used_ bytes
  size_t used_ = 0, minalign_ = 1;
  std::vector<std::pair<uint32_t, uint16_t>> fields_;  // the open table's (offset, vtable slot)
  uint16_t max_voffset_ = 0;
  std::vector<uint32_t> vtables_;  // offsets of the vtables written so far
};

}  // namespace balla::exporter::fb
