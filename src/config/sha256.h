// SHA-256 (FIPS 180-4) of byte strings, as lower-case hex: the stage digests of a resolved run
// (docs/dev/P0_PARAMS.md §1.2) and the output digests of the exporters (docs/dev/EXPORT.md). Header-only, host only.
//   Sha256        incremental: update() any number of times, then hex_final() once; the message is never copied
//   sha256_hex()  one-shot, on Sha256
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace balla::config {

class Sha256 {
 public:
  void update(const void* data, size_t n) {
    if (final_) throw std::logic_error("Sha256: update() after hex_final()");
    if (n == 0) return;
    const auto* p = static_cast<const uint8_t*>(data);
    bytes_ += n;
    if (fill_ > 0) {
      const size_t take = n < 64 - fill_ ? n : 64 - fill_;
      std::memcpy(buf_.data() + fill_, p, take);
      fill_ += take;
      p += take;
      n -= take;
      if (fill_ < 64) return;
      block(buf_.data());
      fill_ = 0;
    }
    for (; n >= 64; p += 64, n -= 64) block(p);
    if (n > 0) std::memcpy(buf_.data(), p, n);
    fill_ = n;
  }
  void update(std::string_view s) { update(s.data(), s.size()); }

  // The padding (0x80, zeros, the bit length big-endian, to a multiple of 64 bytes), then the digest as hex.
  std::string hex_final() {
    if (final_) throw std::logic_error("Sha256: hex_final() called twice");
    const uint64_t bits = bytes_ * 8;
    buf_[fill_++] = 0x80;
    if (fill_ > 56) {
      std::memset(buf_.data() + fill_, 0, 64 - fill_);
      block(buf_.data());
      fill_ = 0;
    }
    std::memset(buf_.data() + fill_, 0, 56 - fill_);
    for (int i = 0; i < 8; ++i) buf_[56 + i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
    block(buf_.data());
    final_ = true;
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (const uint32_t x : h_) {
      for (int i = 28; i >= 0; i -= 4) out.push_back(kHex[(x >> i) & 0xf]);
    }
    return out;
  }

 private:
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

  void block(const uint8_t* m) {
    static constexpr std::array<uint32_t, 64> k = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    std::array<uint32_t, 64> w;
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t{m[4 * i]} << 24) | (uint32_t{m[4 * i + 1]} << 16) | (uint32_t{m[4 * i + 2]} << 8) |
             uint32_t{m[4 * i + 3]};
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
  }

  std::array<uint32_t, 8> h_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::array<uint8_t, 64> buf_{};
  size_t fill_ = 0;
  uint64_t bytes_ = 0;
  bool final_ = false;
};

inline std::string sha256_hex(std::string_view data) {
  Sha256 s;
  s.update(data);
  return s.hex_final();
}

}  // namespace balla::config
