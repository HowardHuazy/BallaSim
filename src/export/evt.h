// Prophesee EVT 2.0, 2.1 and 3.0 event words (docs/design/phase3/outputs.md §4.8), written from the public format
// pages (docs.prophesee.ai: encoding_formats evt2, evt21, evt3; file_formats raw) and the R-FMT
// prototypes (phase3_research/outputs/formats/scripts/evt.py); no OpenEB code is used. Each encoder is a streaming
// state machine fed T1a times u (us after the export origin, non-decreasing, >= 0) in output order; it appends its
// words to `out` (anything with push_back of the word type: a vector, or raw_writer.cpp's block that drains to the
// file). Every encoder preserves the input order, keeps duplicates, and needs no lookahead beyond EVT 2.1's open word,
// which it holds until the next event or finish() (T3), so one batch and many give the same words. Header-only.
#pragma once

#include <bit>
#include <cstdint>

namespace balla::exporter {

static_assert(std::endian::native == std::endian::little,
              "the RAW and DAT writers store their little-endian words in host order");

// EVT 2.0 TIME_HIGH policy. kOpenEb: a TIME_HIGH every 16 us from the first event's 16 us step, as OpenEB's EVT 2.0
// encoder writes them (byte-comparable with metavision_evt2_raw_file_encoder; about 250 kB/s at any event rate).
// kOnChange: one per occupied 64 us step plus a keepalive at every multiple of 128 steps (8.192 ms) in between.
enum class Evt2TimeHigh : uint8_t { kOpenEb, kOnChange };

namespace detail {

// The on-change TIME_HIGH times: before an event in 64 us step k, the keepalive steps (multiples of 128) after the
// last emitted step and below k, then k itself; emit(step) writes one TIME_HIGH word. With wrap_guard (EVT 2.1), a
// step that starts a 2^28-step block (the 2^34 us wrap of the 28-bit value) is preceded by the block's last step
// unless that was the last one emitted: OpenEB's EVT 2.1 decoder counts a loop only when TIME_HIGH falls by at least
// (2^28 - 1) 64 us (evt21_decoder.h set_last_high_timestamp, no threshold), so the value must go from 0x0FFFFFFF
// straight to 0. EVT 2.0's rule allows 10 ms of slack, which the 8.192 ms keepalive meets.
class OnChangeTimeHigh {
 public:
  static constexpr int64_t kKeepaliveSteps = 128;
  static constexpr int64_t kWrapSteps = int64_t{1} << 28;  // a multiple of kKeepaliveSteps

  explicit OnChangeTimeHigh(bool wrap_guard) : wrap_guard_(wrap_guard) {}

  template <class Emit>
  void before(int64_t u, Emit&& emit) {
    const int64_t k = u >> 6;
    if (started_ && k <= last_) return;
    if (started_) {
      for (int64_t m = (last_ / kKeepaliveSteps + 1) * kKeepaliveSteps; m < k; m += kKeepaliveSteps) put(m, emit);
    }
    put(k, emit);
    started_ = true;
  }

 private:
  template <class Emit>
  void put(int64_t step, Emit& emit) {
    if (wrap_guard_ && started_ && step % kWrapSteps == 0 && step - 1 > last_) emit(step - 1);
    emit(step);
    last_ = step;
  }

  bool wrap_guard_;
  bool started_ = false;
  int64_t last_ = 0;
};

}  // namespace detail

// EVT 2.0: 32-bit words. CD_OFF/CD_ON (type 0x0/0x1 in bits 31..28): ts[5:0] at bits 27..22, x at 21..11, y at
// 10..0. TIME_HIGH (0x8): t >> 6 in bits 27..0 (wraps at 2^34 us; OpenEB's decoder detects the loop). The first
// TIME_HIGH is read as absolute, so the writer refuses a first u >= 2^34 (raw_writer.cpp).
class Evt2Encoder {
 public:
  explicit Evt2Encoder(Evt2TimeHigh policy) : policy_(policy) {}

  template <class Out>
  void event(int64_t u, uint32_t x, uint32_t y, uint32_t p, Out& out) {
    if (policy_ == Evt2TimeHigh::kOpenEb) {
      const int64_t k = u >> 4;  // 16 us steps; the words are placed before the first event at or after their time
      if (!started_) next_ = k;
      started_ = true;
      for (; next_ <= k; ++next_) out.push_back(time_high(next_ >> 2));
    } else {
      change_.before(u, [&out](int64_t step) { out.push_back(time_high(step)); });
    }
    out.push_back((p << 28) | (static_cast<uint32_t>(u & 63) << 22) | (x << 11) | y);
  }

  static uint32_t time_high(int64_t step64) { return 0x80000000u | static_cast<uint32_t>(step64 & 0x0FFFFFFF); }

 private:
  Evt2TimeHigh policy_;
  bool started_ = false;
  int64_t next_ = 0;  // kOpenEb: the next 16 us step to emit
  detail::OnChangeTimeHigh change_{false};
};

// EVT 2.1: 64-bit words. CD (type 0x0/0x1 in bits 63..60): ts[5:0] at 59..54, an x base (a multiple of 32) at
// 53..43, y at 42..32 and a valid mask over x base..base+31 in 31..0. TIME_HIGH (0x8): t >> 6 in bits 59..32. An event
// joins the open word only when (t, y, p, base) match and its x is greater than the word's last x; otherwise the word
// is flushed. Decoding a word's mask by increasing x then gives the input order, duplicates included. TIME_HIGH words
// are on change with the wrap guard (detail::OnChangeTimeHigh), placed before the first word of their step; as in
// EVT 2.0, the writer refuses a first u >= 2^34.
class Evt21Encoder {
 public:
  template <class Out>
  void event(int64_t u, uint32_t x, uint32_t y, uint32_t p, Out& out) {
    const uint32_t base = x & ~31u;
    if (open_ && u == t_ && y == y_ && p == p_ && base == base_ && x > last_x_) {
      mask_ |= 1u << (x & 31u);
      last_x_ = x;
      return;
    }
    finish(out);
    change_.before(u, [&out](int64_t step) {
      out.push_back((uint64_t{0x8} << 60) | (static_cast<uint64_t>(step & 0x0FFFFFFF) << 32));
    });
    open_ = true;
    t_ = u;
    y_ = y;
    p_ = p;
    base_ = base;
    last_x_ = x;
    mask_ = 1u << (x & 31u);
  }

  // Writes the open word, if any.
  template <class Out>
  void finish(Out& out) {
    if (!open_) return;
    out.push_back((uint64_t{p_} << 60) | (static_cast<uint64_t>(t_ & 63) << 54) | (uint64_t{base_} << 43) |
                  (uint64_t{y_} << 32) | mask_);
    open_ = false;
  }

 private:
  bool open_ = false;
  int64_t t_ = 0;
  uint32_t y_ = 0, p_ = 0, base_ = 0, last_x_ = 0, mask_ = 0;
  detail::OnChangeTimeHigh change_{true};
};

// EVT 3.0: 16-bit words, without vector words (§4.8: in the stored (t, x, y) order they would save under 0.5 %).
// TIME_HIGH (0x8, t[23:12]) at every 4096 us step from u = 0, through idle gaps too, so OpenEB's loop rule (a high
// that falls by 2048 or more is a 2^24 us loop) recovers u for any first-event time; TIME_LOW (0x6, t[11:0]) when t
// changes; ADDR_Y (0x0, system_type bit 11 = 0, as OpenEB drops the other) when y changes; ADDR_X (0x2, polarity in
// bit 11) per event.
// Start::kFirstEvent is the payload of one ROS EventPacket (§4.7): the words start with the TIME_HIGH and TIME_LOW of
// the packet's first event (dense from there), so a packet decodes without the earlier ones; its words carry u mod
// 2^24, and a reader adds the differences to the packet's header stamp.
class Evt3Encoder {
 public:
  enum class Start : uint8_t { kZero, kFirstEvent };
  explicit Evt3Encoder(Start start = Start::kZero) : start_(start) {}

  template <class Out>
  void event(int64_t u, uint32_t x, uint32_t y, uint32_t p, Out& out) {
    const int64_t hi = u >> 12;
    const int64_t from = started_ ? hi_ + 1 : (start_ == Start::kZero ? 0 : hi);
    for (int64_t h = from; h <= hi; ++h) out.push_back(word(0x8, static_cast<uint32_t>(h & 0xFFF)));
    if (!started_ || u != u_) out.push_back(word(0x6, static_cast<uint32_t>(u & 0xFFF)));
    if (!started_ || y != y_) out.push_back(word(0x0, y));
    out.push_back(word(0x2, (p << 11) | x));
    started_ = true;
    hi_ = hi;
    u_ = u;
    y_ = y;
  }

  static uint16_t word(uint32_t type, uint32_t content) { return static_cast<uint16_t>((type << 12) | content); }

 private:
  Start start_;
  bool started_ = false;
  int64_t hi_ = 0, u_ = 0;
  uint32_t y_ = 0;
};

}  // namespace balla::exporter
