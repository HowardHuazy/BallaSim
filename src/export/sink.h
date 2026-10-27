// The exporters' output file (docs/design/phase3/outputs.md §3.3, §3.6): a buffered writer whose patch() rewrites
// bytes already written (the fixed-size headers rosbag1 and AEDAT4 complete at close), and whose close() hashes the
// closed file by reading it back in 1 MiB blocks with config::Sha256, since a hash streamed while writing would miss
// the patches. Header-only, host only (POSIX descriptors).
#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "config/sha256.h"

namespace balla::exporter {

// One stream of an output file, for EXPORT.json (contract C6).
struct StreamInfo {
  std::string camera, kind, id;
  int64_t count = 0, t_first = 0, t_last = 0;
};

// One output file: its path, size, SHA-256 (lower-case hex) and streams.
struct FileInfo {
  std::string path;
  uint64_t bytes = 0;
  std::string sha256;
  std::vector<StreamInfo> streams;
};

namespace detail {
[[noreturn]] inline void io_error(const std::string& what, const std::string& path) {
  throw std::system_error(errno, std::generic_category(), what + " " + path);
}
}  // namespace detail

// SHA-256 of a file's bytes, read in 1 MiB blocks; *bytes receives the size read.
inline std::string sha256_file(const std::string& path, uint64_t* bytes = nullptr) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) detail::io_error("cannot open", path);
  config::Sha256 sha;
  std::vector<char> buf(size_t{1} << 20);
  uint64_t total = 0;
  for (;;) {
    const ssize_t r = ::read(fd, buf.data(), buf.size());
    if (r < 0 && errno == EINTR) continue;
    if (r < 0) {
      const int e = errno;
      ::close(fd);
      errno = e;
      detail::io_error("cannot read", path);
    }
    if (r == 0) break;
    sha.update(buf.data(), static_cast<size_t>(r));
    total += static_cast<uint64_t>(r);
  }
  ::close(fd);
  if (bytes != nullptr) *bytes = total;
  return sha.hex_final();
}

class ByteSink {
 public:
  static constexpr size_t kBufferBytes = size_t{1} << 20;

  // Creates or truncates path (mode 0644).
  explicit ByteSink(const std::string& path) : path_(path), buf_(kBufferBytes) {
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd_ < 0) detail::io_error("cannot create", path);
  }
  ByteSink(const ByteSink&) = delete;
  ByteSink& operator=(const ByteSink&) = delete;
  // Without close() the file is left as written so far (the export driver removes a failed export's directory).
  ~ByteSink() {
    if (fd_ >= 0) ::close(fd_);
  }

  void write(const void* data, size_t n) {
    check_open();
    if (n == 0) return;
    const auto* p = static_cast<const char*>(data);
    if (used_ + n > buf_.size()) flush();
    if (n >= buf_.size()) {
      write_all(p, n, flushed_);
      flushed_ += n;
      return;
    }
    std::memcpy(buf_.data() + used_, p, n);
    used_ += n;
  }

  uint64_t tell() const { return flushed_ + used_; }

  // Rewrites [offset, offset + n), which must have been written already.
  void patch(uint64_t offset, const void* data, size_t n) {
    check_open();
    if (offset > tell() || n > tell() - offset) {
      throw std::out_of_range("ByteSink::patch: [" + std::to_string(offset) + ", +" + std::to_string(n) +
                              ") lies past the " + std::to_string(tell()) + " bytes written to " + path_);
    }
    const auto* p = static_cast<const char*>(data);
    if (offset < flushed_) {  // the part on disk
      const size_t head = static_cast<size_t>(std::min<uint64_t>(n, flushed_ - offset));
      write_all(p, head, offset);
      p += head;
      offset += head;
      n -= head;
    }
    if (n > 0) std::memcpy(buf_.data() + (offset - flushed_), p, n);  // the part still buffered
  }

  // Flushes and closes, then hashes the closed file.
  FileInfo close() {
    check_open();
    flush();
    const int fd = fd_;
    fd_ = -1;
    if (::close(fd) != 0) detail::io_error("cannot close", path_);
    FileInfo info;
    info.path = path_;
    info.sha256 = sha256_file(path_, &info.bytes);
    if (info.bytes != flushed_) {
      throw std::runtime_error("ByteSink: " + path_ + " holds " + std::to_string(info.bytes) + " bytes after close, " +
                               std::to_string(flushed_) + " were written");
    }
    return info;
  }

  const std::string& path() const { return path_; }

 private:
  void check_open() const {
    if (fd_ < 0) throw std::logic_error("ByteSink: " + path_ + " is closed");
  }

  void flush() {
    if (used_ == 0) return;
    write_all(buf_.data(), used_, flushed_);
    flushed_ += used_;
    used_ = 0;
  }

  // pwrite at an explicit offset, so a patch never moves the append position.
  void write_all(const char* p, size_t n, uint64_t offset) {
    while (n > 0) {
      const ssize_t r = ::pwrite(fd_, p, n, static_cast<off_t>(offset));
      if (r < 0 && errno == EINTR) continue;
      if (r == 0) errno = EIO;
      if (r <= 0) detail::io_error("cannot write", path_);
      p += r;
      n -= static_cast<size_t>(r);
      offset += static_cast<uint64_t>(r);
    }
  }

  std::string path_;
  int fd_ = -1;
  std::vector<char> buf_;
  size_t used_ = 0;
  uint64_t flushed_ = 0;
};

}  // namespace balla::exporter
