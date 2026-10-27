// RAW and DAT writers (raw_writer.h; docs/design/phase3/outputs.md §3.7, §4.7-4.8).
#include "export/raw_writer.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "export/dat.h"

namespace balla::exporter {

namespace {

constexpr int32_t kEvtMaxDim = 2048;  // 11-bit x and y in every EVT format
constexpr int64_t kEvt2MaxFirstU = (int64_t{1} << 34) - 1;  // EVT 2.0 and 2.1: 28-bit TIME_HIGH of 64 us steps

std::string t_ns_rem_text(const RawOptions& o) {
  if (o.t_ns_rem_min == -1 && o.t_ns_rem_max == -1) return "none";
  return o.t_ns_rem_min == o.t_ns_rem_max ? std::to_string(o.t_ns_rem_min) : "varies";
}

// The checks of every writer's constructor; Refused names the first that fails.
void check_args(const char* what, int32_t width, int32_t height, int32_t max_dim, const TimeMap& m,
                const RawOptions& o) {
  if (width < 1 || height < 1 || width > max_dim || height > max_dim) {
    throw Refused(std::string(what) + ": width and height must be in [1, " + std::to_string(max_dim) + "], got " +
                  std::to_string(width) + " x " + std::to_string(height));
  }
  check(m);
  const bool none = o.t_ns_rem_min == -1 && o.t_ns_rem_max == -1;
  if (!none && !(0 <= o.t_ns_rem_min && o.t_ns_rem_min <= o.t_ns_rem_max && o.t_ns_rem_max <= 999)) {
    throw Refused(std::string(what) + ": t_ns_rem range [" + std::to_string(o.t_ns_rem_min) + ", " +
                  std::to_string(o.t_ns_rem_max) + "] must lie in [0, 999] (or be -1, -1 for no events)");
  }
  for (const char c : o.version) {
    if (c < 0x20 || c > 0x7E) throw Refused(std::string(what) + ": version must be printable ASCII");
  }
}

// The common header lines of the T1a files: the origin, the epoch and the dropped sub-us parts.
std::string t1a_lines(const TimeMap& m, const RawOptions& o) {
  return "% ballasim_t_offset_us " + std::to_string(m.origin_us) + "\n% ballasim_epoch_ns " +
         std::to_string(m.epoch_ns) + "\n% ballasim_t_ns_rem " + t_ns_rem_text(o) + "\n";
}

// RAW (file_formats raw): '% key value' lines and '% end' (§3.7). OpenEB opens it with its "any plugin" strategy, so
// the integrator names are BallaSim's and no Prophesee plugin is named.
std::string raw_header(RawFormat f, int32_t width, int32_t height, const TimeMap& m, const RawOptions& o) {
  std::string version = "BallaSim";
  if (!o.version.empty()) version += " " + o.version;
  return "% format " + std::string(raw_format_name(f)) + ";height=" + std::to_string(height) + ";width=" +
         std::to_string(width) + "\n% camera_integrator_name BallaSim\n% plugin_integrator_name BallaSim\n" +
         "% plugin_version " + version + "\n% date " + header_date(m) + "\n" + t1a_lines(m, o) + "% end\n";
}

// DAT (file_formats dat): the header lines of OpenEB's DAT writer, BallaSim's T1a lines, then type and size.
std::string dat_header(int32_t width, int32_t height, const TimeMap& m, const RawOptions& o) {
  std::string h = "% Data file containing CD events.\n% Version 2\n% Date " + header_date(m) + "\n% Height " +
                  std::to_string(height) + "\n% Width " + std::to_string(width) + "\n" + t1a_lines(m, o);
  h.push_back(static_cast<char>(kDatTypeCd));
  h.push_back(static_cast<char>(kDatEventSize));
  return h;
}

// The order, range and sub-us checks every event file shares, and its stream's counters.
class EventStream {
 public:
  // max_u bounds every u (DAT's u32), max_first_u the stream's first (EVT 2.0 and 2.1 read their first TIME_HIGH as
  // absolute, so a first u >= 2^34 would decode modulo 2^34).
  EventStream(const char* what, int32_t width, int32_t height, const TimeMap& m, const RawOptions& o, int64_t max_u,
              int64_t max_first_u)
      : what_(what), width_(width), height_(height), map_(m), rem_min_(o.t_ns_rem_min), rem_max_(o.t_ns_rem_max),
        max_u_(max_u), max_first_u_(max_first_u) {}

  // Checks a whole batch before anything of it is encoded, so a refused batch leaves the file and the encoder as
  // they were.
  void validate(const Events& e) const {
    if (e.n == 0) return;
    if (e.x == nullptr || e.y == nullptr || e.p == nullptr || e.t_ns == nullptr) {
      throw std::invalid_argument(std::string(what_) + ": null event columns");
    }
    int64_t prev = have_last_ ? last_t_ : std::numeric_limits<int64_t>::min();
    for (size_t i = 0; i < e.n; ++i) {
      const int64_t t = e.t_ns[i];
      if (t < prev) {
        throw Refused(std::string(what_) + ": events must be in t_reported order (C4): t_ns " + std::to_string(t) +
                      " follows " + std::to_string(prev));
      }
      prev = t;
      if (e.x[i] >= width_ || e.y[i] >= height_) {
        throw Refused(std::string(what_) + ": event (" + std::to_string(e.x[i]) + ", " + std::to_string(e.y[i]) +
                      ") outside the " + std::to_string(width_) + " x " + std::to_string(height_) + " sensor");
      }
      if (e.p[i] > 1) throw Refused(std::string(what_) + ": polarity must be 0 or 1, got " + std::to_string(e.p[i]));
      const int64_t rem = t - 1000 * time::floor_div(t, int64_t{1000});
      if (rem < rem_min_ || rem > rem_max_) {
        throw Refused(std::string(what_) + ": t_ns_rem " + std::to_string(rem) + " lies outside the declared range [" +
                      std::to_string(rem_min_) + ", " + std::to_string(rem_max_) + "]");
      }
    }
    const int64_t first = t1a_us(map_, e.t_ns[0]), last = t1a_us(map_, e.t_ns[e.n - 1]);
    if (first < 0) {
      throw Refused(std::string(what_) + ": t_ns " + std::to_string(e.t_ns[0]) + " lies before the export origin " +
                    std::to_string(map_.origin_us) + " us");
    }
    if (!have_last_ && first > max_first_u_) {
      throw Refused(std::string(what_) + ": the first event lies u = " + std::to_string(first) +
                    " us after the origin, at or beyond EVT 2.0/2.1's 2^34 us TIME_HIGH loop (readers would " +
                    "decode it modulo 2^34): use raw-evt3, a later --t0 or this camera alone");
    }
    if (last > max_u_) {
      throw Refused(std::string(what_) + ": u = " + std::to_string(last) + " us after the origin exceeds " +
                    std::to_string(max_u_) + " (DAT stores u32 microseconds: export a shorter window, --t0/--t1)");
    }
  }

  // After a validated batch: the counters.
  void commit(const Events& e) {
    if (e.n == 0) return;
    if (count_ == 0) t_first_ = t1a_us(map_, e.t_ns[0]);
    t_last_ = t1a_us(map_, e.t_ns[e.n - 1]);
    last_t_ = e.t_ns[e.n - 1];
    have_last_ = true;
    count_ += static_cast<int64_t>(e.n);
  }

  int64_t u(int64_t t_ns) const { return t1a_us(map_, t_ns); }

  StreamInfo info() const {
    StreamInfo s;
    s.kind = "events";
    s.count = count_;
    s.t_first = t_first_;
    s.t_last = t_last_;
    return s;
  }

 private:
  const char* what_;
  int32_t width_, height_;
  TimeMap map_;
  int32_t rem_min_, rem_max_;
  int64_t max_u_, max_first_u_;
  bool have_last_ = false;
  int64_t last_t_ = 0, count_ = 0, t_first_ = 0, t_last_ = 0;
};

// An encoder's output: words collect in a fixed block that drains to the file when full, so the memory stays bounded
// whatever a batch's time span (EVT 2.0's default policy writes a TIME_HIGH every 16 us through idle gaps, 150 MB per
// idle 10 min). A batch is validated whole before it is encoded, so draining mid-batch keeps the refuse-whole rule.
template <class W>
class WordBlock {
 public:
  static constexpr size_t kWords = (size_t{1} << 18) / sizeof(W);  // 256 KiB

  explicit WordBlock(ByteSink& sink) : sink_(sink) { buf_.reserve(kWords); }

  void push_back(W w) {
    buf_.push_back(w);
    if (buf_.size() == kWords) drain();
  }

  void drain() {
    if (buf_.empty()) return;
    sink_.write(buf_.data(), buf_.size() * sizeof(W));
    buf_.clear();
  }

 private:
  std::vector<W> buf_;
  ByteSink& sink_;
};

}  // namespace

std::string_view raw_format_name(RawFormat f) {
  switch (f) {
    case RawFormat::kEvt2: return "EVT2";
    case RawFormat::kEvt21: return "EVT21";
    case RawFormat::kEvt3: return "EVT3";
  }
  throw std::invalid_argument("raw_format_name: unknown format");
}

struct RawWriter::Impl {
  Impl(const std::string& path, RawFormat f, int32_t width, int32_t height, const TimeMap& m, const RawOptions& o)
      : format(f),
        stream("RawWriter", width, height, m, o, std::numeric_limits<int64_t>::max(),
               f == RawFormat::kEvt3 ? std::numeric_limits<int64_t>::max() : kEvt2MaxFirstU),
        sink(path), evt2(o.evt2_time_high) {
    const std::string h = raw_header(f, width, height, m, o);
    sink.write(h.data(), h.size());
  }

  RawFormat format;
  EventStream stream;
  ByteSink sink;
  Evt2Encoder evt2;
  Evt21Encoder evt21;
  Evt3Encoder evt3;
  WordBlock<uint16_t> w16{sink};
  WordBlock<uint32_t> w32{sink};
  WordBlock<uint64_t> w64{sink};
  bool closed = false;
};

RawWriter::RawWriter(const std::string& path, RawFormat f, int32_t width, int32_t height, TimeMap m,
                     const RawOptions& o) {
  check_args("RawWriter", width, height, kEvtMaxDim, m, o);
  if (o.evt21_legacy_endianness) {
    throw Refused("RawWriter: EVT 2.1 legacy endianness is not implemented (no verified source for its word layout)");
  }
  if (f != RawFormat::kEvt2 && f != RawFormat::kEvt21 && f != RawFormat::kEvt3) {
    throw Refused("RawWriter: unknown format");
  }
  impl_ = std::make_unique<Impl>(path, f, width, height, m, o);
}

RawWriter::~RawWriter() = default;

void RawWriter::append(const Events& e) {
  Impl& s = *impl_;
  if (s.closed) throw Refused("RawWriter: append after close");
  s.stream.validate(e);
  switch (s.format) {
    case RawFormat::kEvt2:
      for (size_t i = 0; i < e.n; ++i) s.evt2.event(s.stream.u(e.t_ns[i]), e.x[i], e.y[i], e.p[i], s.w32);
      s.w32.drain();
      break;
    case RawFormat::kEvt21:
      for (size_t i = 0; i < e.n; ++i) s.evt21.event(s.stream.u(e.t_ns[i]), e.x[i], e.y[i], e.p[i], s.w64);
      s.w64.drain();
      break;
    case RawFormat::kEvt3:
      for (size_t i = 0; i < e.n; ++i) s.evt3.event(s.stream.u(e.t_ns[i]), e.x[i], e.y[i], e.p[i], s.w16);
      s.w16.drain();
      break;
  }
  s.stream.commit(e);
}

FileInfo RawWriter::close() {
  Impl& s = *impl_;
  if (s.closed) throw Refused("RawWriter: closed twice");
  if (s.format == RawFormat::kEvt21) {
    s.evt21.finish(s.w64);
    s.w64.drain();
  }
  s.closed = true;
  FileInfo info = s.sink.close();
  info.streams = {s.stream.info()};
  return info;
}

struct DatWriter::Impl {
  Impl(const std::string& path, int32_t width, int32_t height, const TimeMap& m, const RawOptions& o)
      : stream("DatWriter", width, height, m, o, kDatMaxU, std::numeric_limits<int64_t>::max()), sink(path) {
    const std::string h = dat_header(width, height, m, o);
    sink.write(h.data(), h.size());
  }

  EventStream stream;
  ByteSink sink;
  WordBlock<uint32_t> w32{sink};
  bool closed = false;
};

DatWriter::DatWriter(const std::string& path, int32_t width, int32_t height, TimeMap m, const RawOptions& o) {
  check_args("DatWriter", width, height, kDatMaxDim, m, o);
  impl_ = std::make_unique<Impl>(path, width, height, m, o);
}

DatWriter::~DatWriter() = default;

void DatWriter::append(const Events& e) {
  Impl& s = *impl_;
  if (s.closed) throw Refused("DatWriter: append after close");
  s.stream.validate(e);
  for (size_t i = 0; i < e.n; ++i) dat_record(s.stream.u(e.t_ns[i]), e.x[i], e.y[i], e.p[i], s.w32);
  s.w32.drain();
  s.stream.commit(e);
}

FileInfo DatWriter::close() {
  Impl& s = *impl_;
  if (s.closed) throw Refused("DatWriter: closed twice");
  s.closed = true;
  FileInfo info = s.sink.close();
  info.streams = {s.stream.info()};
  return info;
}

}  // namespace balla::exporter
