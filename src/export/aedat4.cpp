// The AEDAT4 tables and writer (aedat4.h; docs/design/phase3/outputs.md §3.7, §4.7-4.8). Provenance as aedat4.h: an
// independent implementation from dv-processing's schemas and writer behaviour (Apache-2.0); no dv-processing code.
#include "export/aedat4.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstring>
#include <limits>
#include <map>
#include <numbers>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/time/time.h"
#include "export/chunk_pool.h"
#include "export/flatbuf_lite.h"

namespace balla::exporter {

static_assert(std::endian::native == std::endian::little, "AEDAT4 is little-endian, written in host order");

namespace aedat4 {

int32_t compression_type(Codec c) {
  switch (c) {
    case Codec::kNone: return 0;
    case Codec::kLz4: return 1;
    case Codec::kLz4High: return 2;
    case Codec::kZstd: return 3;
    case Codec::kZstdHigh: return 4;
    default: break;
  }
  throw Refused("aedat4: compression must be none, lz4, lz4-high, zstd or zstd-high, got " +
                std::string(codec_name(c)));
}

std::string_view compression_name(Codec c) {
  static constexpr std::string_view kNames[] = {"NONE", "LZ4", "LZ4_HIGH", "ZSTD", "ZSTD_HIGH"};
  return kNames[compression_type(c)];
}

namespace {

void escaped(std::string& out, std::string_view s) {
  for (const char c : s) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&quot;"; break;
      case '\'': out += "&apos;"; break;
      default: out += c;
    }
  }
}

void attr(std::string& out, std::string_view key, std::string_view type, std::string_view value) {
  out += "<attr key=\"";
  out += key;
  out += "\" type=\"";
  out += type;
  out += "\">";
  escaped(out, value);
  out += "</attr>";
}

}  // namespace

std::string info_node(Codec c, const std::vector<StreamDecl>& streams) {
  const std::string_view compression = compression_name(c);
  std::map<std::string, const StreamDecl*> by_name;  // dv's ConfigTree orders nodes by name
  for (const StreamDecl& s : streams) by_name.emplace(std::to_string(s.id), &s);
  std::string out = "<dv version=\"2.0\"><node name=\"outInfo\" path=\"/outInfo/\">";
  for (const auto& [name, s] : by_name) {
    const bool events = s->kind == "events", frames = s->kind == "frames";
    if (!events && !frames && s->kind != "imu") {
      throw std::invalid_argument("aedat4: stream kind must be events, imu or frames");
    }
    const std::string path = "/outInfo/" + name + "/";
    out += "<node name=\"" + name + "\" path=\"" + path + "\">";
    attr(out, "compression", "string", compression);
    attr(out, "originalModuleName", "string", "BallaSim");
    attr(out, "originalOutputName", "string", s->kind);
    attr(out, "typeDescription", "string",
         events   ? "Array of events (polarity ON/OFF)."
         : frames ? "Standard frame (8-bit image)."
                  : "Inertial Measurement Unit data samples.");
    attr(out, "typeIdentifier", "string", events ? "EVTS" : frames ? "FRME" : "IMUS");
    out += "<node name=\"info\" path=\"" + path + "info/\">";
    if (events || frames) {
      attr(out, "sizeX", "int", std::to_string(s->width));
      attr(out, "sizeY", "int", std::to_string(s->height));
    } else {
      char g[32];
      const auto r = std::to_chars(g, g + sizeof g, kEarthG);
      attr(out, "accelerometerUnitMps2", "double", std::string_view(g, static_cast<size_t>(r.ptr - g)));
    }
    attr(out, "source", "string", "BallaSim_" + s->camera);
    out += "</node></node>";
  }
  out += "</node></dv>";
  return out;
}

std::vector<char> io_header(Codec c, int64_t data_table_position, std::string_view info) {
  fb::Builder b(/*force_defaults=*/true);
  const uint32_t s = info.empty() ? 0 : b.create_string(info);
  const uint32_t start = b.start_table();  // dv's CreateIOHeader order: dataTablePosition, infoNode, compression
  b.add_scalar<int64_t>(1, data_table_position, -1);
  b.add_offset(2, s);
  b.add_scalar<int32_t>(0, compression_type(c), 0);
  return b.finish(b.end_table(start), "IOHE", true);
}

std::vector<char> event_packet(const Events& e, const TimeMap& m) {
  fb::Builder b(false, 16 * e.n + 64);
  uint32_t vec = 0;
  if (e.n > 0) {
    char* p = b.start_struct_vector(e.n, 16, 8);  // Event {int64 timestamp; int16 x, y; bool polarity; 3 padding}
    for (size_t i = 0; i < e.n; ++i, p += 16) {
      const int64_t t = t1b_us(m, e.t_ns[i]);
      const auto x = static_cast<int16_t>(e.x[i]), y = static_cast<int16_t>(e.y[i]);
      std::memcpy(p, &t, 8);
      std::memcpy(p + 8, &x, 2);
      std::memcpy(p + 10, &y, 2);
      p[12] = static_cast<char>(e.p[i] != 0);
      p[13] = p[14] = p[15] = 0;
    }
    vec = b.end_vector(e.n);
  }
  const uint32_t start = b.start_table();
  b.add_offset(0, vec);
  return b.finish(b.end_table(start), "EVTS", true);
}

std::vector<char> imu_packet(const std::vector<ImuRecord>& rows) {
  fb::Builder b(false, 96 * rows.size() + 64);
  std::vector<uint32_t> offs;
  offs.reserve(rows.size());
  for (const ImuRecord& r : rows) {
    const uint32_t start = b.start_table();  // dv's CreateIMU order: timestamp, then magnetometerZ down to temperature
    b.add_scalar<int64_t>(0, r.timestamp, 0);
    for (int axis = 2; axis >= 0; --axis) b.add_scalar(8 + axis, r.magnetometer[axis], 0.0f);
    for (int axis = 2; axis >= 0; --axis) b.add_scalar(5 + axis, r.gyroscope[axis], 0.0f);
    for (int axis = 2; axis >= 0; --axis) b.add_scalar(2 + axis, r.accelerometer[axis], 0.0f);
    b.add_scalar(1, r.temperature, 0.0f);
    offs.push_back(b.end_table(start));
  }
  const uint32_t vec = b.create_offset_vector(offs);
  const uint32_t start = b.start_table();
  b.add_offset(0, vec);
  return b.finish(b.end_table(start), "IMUS", true);
}

std::vector<char> frame_packet(const FrameRecord& f) {
  fb::Builder b(false, f.pixel_bytes + 192);
  char* p = b.start_struct_vector(f.pixel_bytes, 1, 1);  // pixels: [uint8], created before the table
  if (f.pixel_bytes > 0) std::memcpy(p, f.pixels, f.pixel_bytes);
  const uint32_t pixels = b.end_vector(f.pixel_bytes);
  // dv's CreateFrame order: the int64 fields, then pixels, then the int16 fields, then the int8 ones, each group in
  // reverse field order (as flatc orders a Create function's adds, largest first); defaults left out.
  const uint32_t start = b.start_table();
  b.add_scalar<int64_t>(11, f.exposure, 0);
  b.add_scalar<int64_t>(4, f.end_of_exposure, 0);
  b.add_scalar<int64_t>(3, f.start_of_exposure, 0);
  b.add_scalar<int64_t>(2, f.end_of_frame, 0);
  b.add_scalar<int64_t>(1, f.start_of_frame, 0);
  b.add_scalar<int64_t>(0, f.timestamp, 0);
  b.add_offset(10, pixels);
  b.add_scalar<int16_t>(9, f.position_y, 0);
  b.add_scalar<int16_t>(8, f.position_x, 0);
  b.add_scalar<int16_t>(7, f.size_y, 0);
  b.add_scalar<int16_t>(6, f.size_x, 0);
  b.add_scalar<int8_t>(12, f.source, 0);
  b.add_scalar<int8_t>(5, f.format, 0);
  return b.finish(b.end_table(start), "FRME", true);
}

std::vector<char> file_data_table(const std::vector<TableEntry>& entries) {
  fb::Builder b(false, 72 * entries.size() + 64);
  std::vector<uint32_t> offs;
  offs.reserve(entries.size());
  for (const TableEntry& e : entries) {
    const uint32_t start = b.start_table();  // dv's CreatePacketInfo order: the int64 fields, then PacketHeader
    b.add_scalar<int64_t>(4, e.t_end, 0);
    b.add_scalar<int64_t>(3, e.t_start, 0);
    b.add_scalar<int64_t>(2, e.elements, 0);
    b.add_scalar<int64_t>(0, e.byte_offset, 0);
    const int32_t header[2] = {e.stream, e.size};  // PacketHeader {int32 StreamID; int32 Size}
    b.add_struct(1, header, sizeof header, 4);
    offs.push_back(b.end_table(start));
  }
  const uint32_t vec = b.create_offset_vector(offs);
  const uint32_t start = b.start_table();
  b.add_offset(0, vec);
  return b.finish(b.end_table(start), "FTAB", true);
}

}  // namespace aedat4

namespace {

constexpr uint64_t kPacketOverhead = 64;  // a packet FlatBuffer's bytes besides its items (prefix, root, vtables)
constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();

}  // namespace

uint64_t Aedat4Writer::max_packet_events() { return (aedat4::kMaxPacketBytes - kPacketOverhead) / 16; }
uint64_t Aedat4Writer::max_packet_samples() { return (aedat4::kMaxPacketBytes - kPacketOverhead) / 96; }
uint64_t Aedat4Writer::max_packets() { return (fb::kMaxBytes - kPacketOverhead) / 72; }

struct Aedat4Writer::Impl {
  struct Stream {
    aedat4::StreamDecl decl;
    bool imu = false;  // decl.kind "imu"; frames streams hold nothing (each frame is its own packet)
    // The open packet (items of one T1a grid cell) and the first t_ns past it (cell_of).
    int64_t cell = 0, cell_end = 0;
    std::vector<uint16_t> x, y;
    std::vector<uint8_t> p;
    std::vector<int64_t> t;
    std::vector<double> gyro, accel;  // 3 per sample
    bool have_last = false;
    int64_t last_t = 0, count = 0, t_first = 0, t_last = 0;  // items and their T1b times
    int64_t last_key = 0;  // frames: the last frame's exposure start (t_ns), its table key
  };

  Impl(const std::string& path, Codec c, const TimeMap& m, int threads, int64_t packet)
      : codec(c), map(m), packet_us(packet), sink(path), pool(c, threads) {}

  // The packet of t_ns (>= 1000 origin_us), k = floor(u / packet_us) on the T1a grid, and the first t_ns of the next
  // one, saturated at INT64_MAX.
  void cell_of(int64_t t_ns, int64_t* k, int64_t* end_ns) const {
    *k = time::floor_div(t1a_us(map, t_ns), packet_us);
    int64_t u = 0;
    if (__builtin_mul_overflow(*k + 1, packet_us, &u) || __builtin_add_overflow(u, map.origin_us, &u) ||
        __builtin_mul_overflow(u, int64_t{1000}, end_ns)) {
      *end_ns = std::numeric_limits<int64_t>::max();
    }
  }

  Stream& stream(int32_t id, std::string_view kind, const char* what) {
    if (closed) throw Refused(std::string("aedat4: ") + what + " after close");
    if (id < 0 || static_cast<size_t>(id) >= streams.size()) {
      throw Refused("aedat4: unknown stream " + std::to_string(id));
    }
    Stream& s = streams[static_cast<size_t>(id)];
    if (s.decl.kind != kind) {
      throw Refused("aedat4: stream " + std::to_string(id) + " is a " + s.decl.kind + " stream, not " +
                    std::string(kind));
    }
    return s;
  }

  int32_t add(const std::string& camera, const std::string& kind, int32_t w, int32_t h) {
    if (closed) throw Refused("aedat4: stream after close");
    if (begun) {
      throw Refused("aedat4: every stream is declared before the first item (the IOHeader written then lists them)");
    }
    if (camera.empty() || std::any_of(camera.begin(), camera.end(), [](char c) {
          return static_cast<unsigned char>(c) < 0x20 || c == 0x7F;
        })) {
      throw Refused("aedat4: a camera name must be non-empty and free of control characters");
    }
    if (!names.emplace(camera, kind).second) throw Refused("aedat4: the " + kind + " stream of " + camera + " twice");
    if (streams.size() == static_cast<size_t>(kMaxStreams)) throw Refused("aedat4: more than 65536 streams");
    Stream s;
    s.decl = {static_cast<int32_t>(streams.size()), camera, kind, w, h};
    s.imu = kind == "imu";
    streams.push_back(std::move(s));
    return streams.back().decl.id;
  }

  // The version and the IOHeader, with every stream; the header is patched with the table's position at close.
  void begin() {
    if (begun) return;
    std::vector<aedat4::StreamDecl> decls;
    for (const Stream& s : streams) decls.push_back(s.decl);
    info = aedat4::info_node(codec, decls);
    sink.write(aedat4::kVersion.data(), aedat4::kVersion.size());
    header_at = sink.tell();
    const std::vector<char> h = aedat4::io_header(codec, -1, info);
    header_bytes = h.size();
    sink.write(h.data(), h.size());
    begun = true;
  }

  // Order and origin of a batch of times against the stream's last one, and the packets it fills (the held one
  // first) against capacity and the file's packets against max_packets(), before any of it is held.
  void check_times(const Stream& s, const int64_t* t, size_t n, uint64_t capacity, const char* what) const {
    int64_t prev = s.have_last ? s.last_t : std::numeric_limits<int64_t>::min();
    for (size_t i = 0; i < n; ++i) {
      if (t[i] < prev) {
        throw Refused("aedat4: stream " + std::to_string(s.decl.id) + ": " + what + "s must be in time order: t_ns " +
                      std::to_string(t[i]) + " follows " + std::to_string(prev));
      }
      prev = t[i];
    }
    if (t1a_us(map, t[0]) < 0) {
      throw Refused("aedat4: stream " + std::to_string(s.decl.id) + ": t_ns " + std::to_string(t[0]) +
                    " lies before the export origin " + std::to_string(map.origin_us) + " us");
    }
    if (t1b_us(map, t[0]) < 0) {  // T1b is monotone: the first time bounds the batch
      throw Refused("aedat4: stream " + std::to_string(s.decl.id) + ": t_ns " + std::to_string(t[0]) +
                    " has the timestamp floor(t_ns / 1000) - origin_us + epoch_ns / 1000 = " +
                    std::to_string(t1b_us(map, t[0])) +
                    " us < 0 (dv-processing's time ranges start at 0): choose --epoch-ns >= 0");
    }
    uint64_t held = s.t.size(), started = packets_started;
    int64_t cell = s.cell, end = s.cell_end;
    for (size_t i = 0; i < n; ++i) {
      if (held == 0 || t[i] >= end) {
        int64_t k = 0;
        cell_of(t[i], &k, &end);
        if (held == 0 || k != cell) {
          held = 0;
          ++started;
        }
        cell = k;
      }
      if (++held > capacity) {
        throw Refused("aedat4: stream " + std::to_string(s.decl.id) + ": the packet of t_ns " + std::to_string(t[i]) +
                      " would hold more than " + std::to_string(capacity) + " " + what +
                      "s (an int32 packet size): use shorter packets (--packet-us)");
      }
    }
    if (started > max_packets()) {
      throw Refused("aedat4: more than " + std::to_string(max_packets()) +
                    " packets (the FileDataTable is one FlatBuffer under 2 GiB): use longer packets (--packet-us)");
    }
  }

  // Before item i of a batch: flushes the open packet when t_ns starts a later one.
  void open_cell(Stream& s, int64_t t_ns) {
    if (!s.t.empty() && t_ns < s.cell_end) return;
    int64_t k = 0, end = 0;
    cell_of(t_ns, &k, &end);
    if (!s.t.empty() && k != s.cell) flush(s);
    if (s.t.empty()) ++packets_started;
    s.cell = k;
    s.cell_end = end;
  }

  void commit(Stream& s, const int64_t* t, size_t n) {
    if (s.count == 0) s.t_first = t1b_us(map, t[0]);
    s.t_last = t1b_us(map, t[n - 1]);
    s.last_t = t[n - 1];
    s.have_last = true;
    s.count += static_cast<int64_t>(n);
  }

  // Writes the open packet (write_packet).
  void flush(Stream& s) {
    if (s.t.empty()) return;
    const size_t n = s.t.size();
    std::vector<char> raw;
    if (s.imu) {
      std::vector<aedat4::ImuRecord> rows(n);
      for (size_t i = 0; i < n; ++i) {
        aedat4::ImuRecord& r = rows[i];
        r.timestamp = t1b_us(map, s.t[i]);
        for (size_t a = 0; a < 3; ++a) {
          r.accelerometer[a] = static_cast<float>(s.accel[3 * i + a] / aedat4::kEarthG);
          r.gyroscope[a] = static_cast<float>(s.gyro[3 * i + a] * 180.0 / std::numbers::pi);
        }
      }
      raw = aedat4::imu_packet(rows);
    } else {
      raw = aedat4::event_packet(Events{s.x.data(), s.y.data(), s.p.data(), s.t.data(), n}, map);
    }
    if (raw.size() > aedat4::kMaxPacketBytes) throw std::logic_error("aedat4: a packet past its capacity");
    write_packet(std::move(raw), {0, s.decl.id, 0, static_cast<int64_t>(n), t1b_us(map, s.t.front()),
                                  t1b_us(map, s.t.back())});
    s.x.clear();
    s.y.clear();
    s.p.clear();
    s.t.clear();
    s.gyro.clear();
    s.accel.clear();
  }

  // A packet FlatBuffer, compressed on the pool; its header and frame go to the file in packet order, with its
  // FileDataTable entry.
  void write_packet(std::vector<char> raw, aedat4::TableEntry entry) {
    pool.submit(std::move(raw), [this, entry](std::vector<char>&& z) mutable {
      if (static_cast<int64_t>(z.size()) > kInt32Max) throw Refused("aedat4: a compressed packet past 2 GiB");
      const int32_t header[2] = {entry.stream, static_cast<int32_t>(z.size())};
      sink.write(header, sizeof header);
      entry.byte_offset = static_cast<int64_t>(sink.tell());
      entry.size = header[1];
      sink.write(z.data(), z.size());
      table.push_back(entry);
    });
  }

  Codec codec;
  TimeMap map;
  int64_t packet_us;
  ByteSink sink;
  ChunkPool pool;  // after the sink: destroyed first, so no emit outlives it
  std::vector<Stream> streams;
  std::set<std::pair<std::string, std::string>> names;  // (camera, kind)
  std::string info;
  uint64_t header_at = 0;
  size_t header_bytes = 0;
  std::vector<aedat4::TableEntry> table;  // one per packet written, in file order
  uint64_t packets_started = 0;           // written and held
  bool begun = false, closed = false;
};

Aedat4Writer::Aedat4Writer(const std::string& path, Codec c, TimeMap m, int threads, int64_t packet_us) {
  (void)aedat4::compression_type(c);  // before the file is created
  check(m);
  if (threads < 1 || threads > ChunkPool::kMaxThreads) {
    throw Refused("threads must be in [1, " + std::to_string(ChunkPool::kMaxThreads) + "], got " +
                  std::to_string(threads));
  }
  if (packet_us < 1 || packet_us > kMaxPacketUs) {
    throw Refused("aedat4: packet_us must be in [1, 2^24], got " + std::to_string(packet_us));
  }
  impl_ = std::make_unique<Impl>(path, c, m, threads, packet_us);
}

Aedat4Writer::~Aedat4Writer() = default;

int32_t Aedat4Writer::add_events_stream(const std::string& camera, int32_t w, int32_t h) {
  if (w < 1 || h < 1 || w > aedat4::kMaxSize || h > aedat4::kMaxSize) {
    throw Refused("aedat4: width and height must be in [1, 32767] (int16 event coordinates), got " +
                  std::to_string(w) + " x " + std::to_string(h));
  }
  return impl_->add(camera, "events", w, h);
}

int32_t Aedat4Writer::add_imu_stream(const std::string& camera) { return impl_->add(camera, "imu", 0, 0); }

int32_t Aedat4Writer::add_frame_stream(const std::string& camera, int32_t w, int32_t h) {
  if (w < 1 || h < 1 || w > aedat4::kMaxSize || h > aedat4::kMaxSize) {
    throw Refused("aedat4: a frames stream's width and height must be in [1, 32767] (int16 sizes), got " +
                  std::to_string(w) + " x " + std::to_string(h));
  }
  return impl_->add(camera, "frames", w, h);
}

void Aedat4Writer::events(int32_t stream, const Events& e) {
  Impl& w = *impl_;
  Impl::Stream& s = w.stream(stream, "events", "events");
  if (e.n == 0) return;
  if (e.x == nullptr || e.y == nullptr || e.p == nullptr || e.t_ns == nullptr) {
    throw std::invalid_argument("aedat4: null event columns");
  }
  for (size_t i = 0; i < e.n; ++i) {
    if (e.x[i] >= s.decl.width || e.y[i] >= s.decl.height) {
      throw Refused("aedat4: stream " + std::to_string(stream) + ": event (" + std::to_string(e.x[i]) + ", " +
                    std::to_string(e.y[i]) + ") outside the " + std::to_string(s.decl.width) + " x " +
                    std::to_string(s.decl.height) + " sensor");
    }
    if (e.p[i] > 1) throw Refused("aedat4: polarity must be 0 or 1, got " + std::to_string(e.p[i]));
  }
  w.check_times(s, e.t_ns, e.n, max_packet_events(), "event");
  w.begin();
  for (size_t i = 0; i < e.n; ++i) {
    w.open_cell(s, e.t_ns[i]);
    s.x.push_back(e.x[i]);
    s.y.push_back(e.y[i]);
    s.p.push_back(e.p[i]);
    s.t.push_back(e.t_ns[i]);
  }
  w.commit(s, e.t_ns, e.n);
}

void Aedat4Writer::imu(int32_t stream, const ImuView& v) {
  Impl& w = *impl_;
  Impl::Stream& s = w.stream(stream, "imu", "imu");
  if (v.n == 0) return;
  if (v.t_ns == nullptr || v.gyro == nullptr || v.accel == nullptr) {
    throw std::invalid_argument("aedat4: null IMU columns");
  }
  w.check_times(s, v.t_ns, v.n, max_packet_samples(), "IMU sample");
  w.begin();
  for (size_t i = 0; i < v.n; ++i) {
    w.open_cell(s, v.t_ns[i]);
    s.t.push_back(v.t_ns[i]);
    s.gyro.insert(s.gyro.end(), v.gyro + 3 * i, v.gyro + 3 * i + 3);
    s.accel.insert(s.accel.end(), v.accel + 3 * i, v.accel + 3 * i + 3);
  }
  w.commit(s, v.t_ns, v.n);
}

void Aedat4Writer::frames(int32_t stream, const FrameView& v, const FrameOptions& o) {
  Impl& w = *impl_;
  Impl::Stream& s = w.stream(stream, "frames", "frames");
  const std::string what = "aedat4: stream " + std::to_string(stream);
  check(v, o, what);
  if (v.n == 0) return;
  if (o.map == FrameMap::kNone && !is_8bit(v.format)) {
    throw Refused("AEDAT4 frames are 8-bit: pass --frame-map linear8|srgb8 (these frames are " +
                  std::string(format_name(v.format)) + ")");
  }
  if (v.x0 < 0 || v.y0 < 0 || int64_t{v.x0} + v.width > s.decl.width || int64_t{v.y0} + v.height > s.decl.height) {
    throw Refused(what + ": a " + std::to_string(v.width) + " x " + std::to_string(v.height) + " frame at (" +
                  std::to_string(v.x0) + ", " + std::to_string(v.y0) + ") lies outside the " +
                  std::to_string(s.decl.width) + " x " + std::to_string(s.decl.height) + " image");
  }
  const size_t bytes = static_cast<size_t>(v.width) * static_cast<size_t>(v.height) *
                       static_cast<size_t>(channels(v.format));
  if (bytes > aedat4::kMaxPacketBytes - 256) throw Refused(what + ": a frame larger than an AEDAT4 packet holds");
  if (s.have_last && v.t_mid_ns[0] < s.last_t) {
    throw Refused(what + ": frames must be in time order: midpoint " + std::to_string(v.t_mid_ns[0]) + " follows " +
                  std::to_string(s.last_t));
  }
  for (size_t k = 0; k < v.n; ++k) {  // the frame start is each frame's earliest time
    if (t1a_us(w.map, v.t_frame_start_ns[k]) < 0) {
      throw Refused(what + ": the frame start " + std::to_string(v.t_frame_start_ns[k]) +
                    " ns lies before the export origin " + std::to_string(w.map.origin_us) + " us");
    }
    if (t1b_us(w.map, v.t_frame_start_ns[k]) < 0) {
      throw Refused(what + ": the frame start " + std::to_string(v.t_frame_start_ns[k]) +
                    " ns has a timestamp < 0 (dv-processing's time ranges start at 0): choose --epoch-ns >= 0");
    }
  }
  for (size_t k = 0; k < v.n; ++k) {  // the table keys (exposure starts) in order, as dv-processing seeks by them
    const int64_t prev = k > 0 ? v.t_exp_start_ns[k - 1] : s.have_last ? s.last_key : v.t_exp_start_ns[0];
    if (v.t_exp_start_ns[k] < prev) {
      throw Refused(what + ": frames must be in time order: exposure start " + std::to_string(v.t_exp_start_ns[k]) +
                    " follows " + std::to_string(prev));
    }
  }
  if (w.packets_started + v.n > max_packets()) {
    throw Refused("aedat4: more than " + std::to_string(max_packets()) +
                  " packets (the FileDataTable is one FlatBuffer under 2 GiB)");
  }
  w.begin();
  std::vector<uint8_t> px;
  for (size_t k = 0; k < v.n; ++k) {
    frame8(v, k, o, /*bgr=*/true, px);
    aedat4::FrameRecord r;
    r.timestamp = t1b_us(w.map, v.t_mid_ns[k]);
    r.start_of_frame = t1b_us(w.map, v.t_frame_start_ns[k]);
    r.end_of_frame = t1b_us(w.map, v.t_frame_end_ns[k]);
    r.start_of_exposure = t1b_us(w.map, v.t_exp_start_ns[k]);
    r.end_of_exposure = t1b_us(w.map, v.t_exp_end_ns[k]);
    r.exposure = time::floor_div(v.exposure_ns >= 0 ? v.exposure_ns : v.t_exp_end_ns[k] - v.t_exp_start_ns[k],
                                 int64_t{1000});
    r.format = channels(v.format) == 3 ? aedat4::kOpenCv8UC3 : aedat4::kOpenCv8UC1;
    r.size_x = static_cast<int16_t>(v.width);
    r.size_y = static_cast<int16_t>(v.height);
    r.position_x = static_cast<int16_t>(v.x0);
    r.position_y = static_cast<int16_t>(v.y0);
    r.pixels = px.data();
    r.pixel_bytes = px.size();
    ++w.packets_started;
    // The table entry at the time dv-processing 2.0.4 gives the frame (dv::Frame::timestamp is the packet's
    // timestampStartOfExposure; its getTimeRange and getFramesTimeRange go by the table), so its time queries agree
    // with the frames it returns; the `timestamp` field keeps the midpoint (frames.md §3.11, outputs.md Q5).
    w.write_packet(aedat4::frame_packet(r), {0, s.decl.id, 0, 1, r.start_of_exposure, r.start_of_exposure});
  }
  w.commit(s, v.t_mid_ns, v.n);
  s.last_key = v.t_exp_start_ns[v.n - 1];
}

FileInfo Aedat4Writer::close() {
  Impl& w = *impl_;
  if (w.closed) throw Refused("aedat4: closed twice");
  w.begin();
  for (Impl::Stream& s : w.streams) w.flush(s);
  w.pool.drain();
  w.closed = true;
  const int64_t position = static_cast<int64_t>(w.sink.tell());
  const std::vector<char> table = aedat4::file_data_table(w.table);
  const std::vector<char> z = compress(w.codec, table.data(), table.size());
  w.sink.write(z.data(), z.size());
  const std::vector<char> h = aedat4::io_header(w.codec, position, w.info);
  if (h.size() != w.header_bytes) throw std::logic_error("aedat4: the IOHeader changed size when patched");
  w.sink.patch(w.header_at, h.data(), h.size());
  FileInfo info = w.sink.close();
  for (const Impl::Stream& s : w.streams) {
    StreamInfo si;
    si.camera = s.decl.camera;
    si.kind = s.decl.kind;
    si.id = std::to_string(s.decl.id);
    si.count = s.count;
    si.t_first = s.t_first;
    si.t_last = s.t_last;
    info.streams.push_back(std::move(si));
  }
  return info;
}

}  // namespace balla::exporter
