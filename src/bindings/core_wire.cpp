// ballasim._core.wire: the browser UI's event wire (core/wire/wire.h; docs/design/phase3/ui.md §3.5, §3.10, §4.1-4.2).
// Every per-event step runs here with the GIL released; ballasim/ui/streams.py reads files and frames messages.
// Inputs are 1-D C-contiguous arrays of the exact dtype (bound with noconvert): int64 ns times, uint16 coordinates,
// uint8 polarities and labels, uint32 permutations are never cast implicitly. They are views of the caller's arrays,
// so a value read with the GIL released may change under a concurrent writer: only values never used as an index are
// read that way, and a permutation is copied and checked with the GIL held (checked_perm, as checked_indices in
// ndarray_util.h).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "bindings/core_bindings.h"
#include "bindings/ndarray_util.h"
#include "core/wire/wire.h"

namespace nb = nanobind;

namespace balla::bindings {
namespace {

template <class T>
using Col = nb::ndarray<const T, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
// any dtype, one or two dimensions, C-contiguous: a GT column (gather)
using AnyCol = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;

template <class T>
std::span<const T> span_of(const Col<T>& a) {
  return {a.data(), a.shape(0)};
}

template <class T>
std::span<const T> span_of(const std::optional<Col<T>>& a) {
  return a ? span_of(*a) : std::span<const T>{};
}

void same_length(size_t n, size_t m, const char* what) {
  if (n != m) throw std::invalid_argument(std::string(what) + ": every column needs one entry per event");
}

nb::list n_ge_list(const std::array<uint32_t, wire::kLevels + 1>& n_ge) {
  nb::list l;
  for (const uint32_t v : n_ge) l.append(v);
  return l;
}

Out<uint64_t> event_key(const Col<int64_t>& t_ns, const Col<uint16_t>& x, const Col<uint16_t>& y,
                        const Col<uint8_t>& p, const Col<uint32_t>& k) {
  const size_t n = t_ns.shape(0);
  same_length(n, x.shape(0), "event_key");
  same_length(n, y.shape(0), "event_key");
  same_length(n, p.shape(0), "event_key");
  same_length(n, k.shape(0), "event_key");
  std::vector<uint64_t> out(n);
  {
    nb::gil_scoped_release release;
    for (size_t i = 0; i < n; ++i) out[i] = wire::event_key(t_ns(i), x(i), y(i), p(i), k(i));
  }
  return to_numpy(std::move(out), {n});
}

Out<uint8_t> lod_level(const Col<uint64_t>& key) {
  const size_t n = key.shape(0);
  std::vector<uint8_t> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(wire::lod_level(key(i)));
  return to_numpy(std::move(out), {n});
}

nb::dict lod_order(const Col<int64_t>& t_ns, const Col<uint16_t>& x, const Col<uint16_t>& y, const Col<uint8_t>& p) {
  wire::BlockOrder order;
  {
    nb::gil_scoped_release release;
    order = wire::lod_order(span_of(t_ns), span_of(x), span_of(y), span_of(p));
  }
  nb::dict d;
  d["n_ge"] = n_ge_list(order.n_ge);
  d["duplicates"] = order.duplicates;
  const size_t n = order.perm.size();
  d["perm"] = to_numpy(std::move(order.perm), {n});
  return d;
}

uint32_t checked_u32(int64_t v, const char* what) {
  if (v < 0 || v > int64_t{UINT32_MAX}) throw std::invalid_argument(std::string(what) + " out of range");
  return static_cast<uint32_t>(v);
}

// perm[i0, i1) as an owned copy, each entry checked < bound (std::invalid_argument `refusal` otherwise). The copy is
// made and checked while the GIL is held, so only checked entries reach the GIL-released loops, even if the caller
// writes to its array meanwhile (the rule of checked_indices, ndarray_util.h).
std::vector<uint32_t> checked_perm(const Col<uint32_t>& perm, size_t i0, size_t i1, size_t bound,
                                   const char* refusal) {
  std::vector<uint32_t> out(perm.data() + i0, perm.data() + i1);
  for (const uint32_t v : out) {
    if (v >= bound) throw std::invalid_argument(refusal);
  }
  return out;
}

Out<uint32_t> pack_ev(const Col<int64_t>& t_ns, const Col<uint16_t>& x, const Col<uint16_t>& y, const Col<uint8_t>& p,
                      const std::optional<Col<uint8_t>>& label, const std::optional<Col<uint32_t>>& perm, int64_t i0,
                      int64_t i1, int64_t t_base_ns) {
  const size_t n = t_ns.shape(0);
  if (label) same_length(n, label->shape(0), "pack_ev");
  const uint32_t a = checked_u32(i0, "pack_ev: i0"), b = checked_u32(i1, "pack_ev: i1");
  const size_t order_n = perm ? perm->shape(0) : n;
  // checked before `out` is allocated: a wrong i1 must not cost 8 (i1 - i0) bytes first
  if (a > b || b > order_n) throw std::invalid_argument("pack_ev: need i0 <= i1 <= the number of ordered events");
  // with a perm: the checked copy of its [i0, i1), packed as positions [0, i1 - i0); without: [i0, i1) in time order
  const std::vector<uint32_t> order =
      perm ? checked_perm(*perm, a, b, n, "pack_ev: a perm entry lies outside the events") : std::vector<uint32_t>{};
  const uint32_t from = perm ? 0 : a, to = perm ? b - a : b;
  std::vector<uint32_t> out(2 * size_t{b - a});
  {
    nb::gil_scoped_release release;
    wire::pack_ev(span_of(t_ns), span_of(x), span_of(y), span_of(p), label ? label->data() : nullptr, order, from, to,
                  t_base_ns, out.data());
  }
  return to_numpy(std::move(out), {size_t{b - a}, 2});
}

// gather(perm, [col, ...]): the columns (one dtype, one entry per event) in perm's order, interleaved: shape (n,)
// for one 1-D column, else (n, components).
nb::ndarray<nb::numpy> gather(const Col<uint32_t>& perm, const std::vector<AnyCol>& columns) {
  if (columns.empty()) throw std::invalid_argument("gather: no column");
  const AnyCol& first = columns.front();
  const size_t rows = first.ndim() >= 1 ? first.shape(0) : 0;
  size_t comps = 0;
  for (const AnyCol& c : columns) {
    if (c.ndim() < 1 || c.ndim() > 2) throw std::invalid_argument("gather: columns are 1-D or 2-D arrays");
    if (c.dtype() != first.dtype() || c.shape(0) != rows) {
      throw std::invalid_argument("gather: the columns need one dtype and one entry per event");
    }
    comps += c.ndim() == 2 ? c.shape(1) : 1;
  }
  const size_t item = first.itemsize(), n = perm.shape(0);
  const std::vector<uint32_t> order = checked_perm(perm, 0, n, rows, "gather: a perm entry lies outside the column");
  auto* owned = new std::vector<unsigned char>(n * comps * item);
  nb::capsule owner(owned, [](void* q) noexcept { delete static_cast<std::vector<unsigned char>*>(q); });
  {
    nb::gil_scoped_release release;
    size_t at = 0;
    for (const AnyCol& c : columns) {
      const size_t cc = c.ndim() == 2 ? c.shape(1) : 1;
      wire::gather(order, c.data(), cc * item, owned->data() + at * item, comps * item);
      at += cc;
    }
  }
  const size_t shape[2] = {n, comps};
  const bool flat = columns.size() == 1 && first.ndim() == 1;
  return nb::ndarray<nb::numpy>(owned->data(), flat ? 1 : 2, shape, owner, nullptr, first.dtype());
}

nb::tuple rate_counts(const Col<int64_t>& t_ns, const std::optional<Col<uint8_t>>& p, int64_t t0_ns, int64_t bin_ns,
                      int64_t n_bins) {
  if (n_bins < 0 || n_bins > (int64_t{1} << 26)) throw std::invalid_argument("rate_counts: n_bins out of range");
  if (p) same_length(t_ns.shape(0), p->shape(0), "rate_counts");
  const size_t nb_ = static_cast<size_t>(n_bins);
  std::vector<uint32_t> total(nb_, 0), on(p ? nb_ : 0, 0);
  {
    nb::gil_scoped_release release;
    wire::rate_counts(span_of(t_ns), span_of(p), t0_ns, bin_ns, total, on);
  }
  nb::object on_obj = p ? nb::object(nb::cast(to_numpy(std::move(on), {nb_}))) : nb::none();
  return nb::make_tuple(to_numpy(std::move(total), {nb_}), on_obj);
}

Out<int64_t> c4_times(const Col<int64_t>& t_us, const std::optional<Col<uint16_t>>& t_ns_rem, int64_t t_offset_us) {
  const size_t n = t_us.shape(0);
  std::vector<int64_t> out(n);
  {
    nb::gil_scoped_release release;
    wire::c4_times(span_of(t_us), span_of(t_ns_rem), t_offset_us, out);
  }
  return to_numpy(std::move(out), {n});
}

Out<uint8_t> polarity01(const Col<uint8_t>& p, bool invert) {
  const size_t n = p.shape(0);
  std::vector<uint8_t> out(n);
  {
    nb::gil_scoped_release release;
    wire::polarity01(span_of(p), invert, out);
  }
  return to_numpy(std::move(out), {n});
}

// The per-event test of a region (ui.md §4.5; viz.stats.MaskRegion): 1 where (x, y) is set in the mask whose
// interval [t_lo, t_hi) holds t. x and y index the masks, so they are copied while the GIL is held (as checked_perm);
// the intervals and words are only compared and read at checked positions.
Out<uint8_t> region_bits(const Col<uint16_t>& x, const Col<uint16_t>& y, const Col<int64_t>& t_ns, int64_t width,
                         int64_t height, const Col<int64_t>& t_lo_ns, const Col<int64_t>& t_hi_ns,
                         const Col<uint32_t>& bits) {
  const size_t n = t_ns.shape(0);
  same_length(n, x.shape(0), "region_bits");
  same_length(n, y.shape(0), "region_bits");
  if (width < 1 || height < 1 || width > 0xFFFF || height > 0xFFFF) {
    throw std::invalid_argument("region_bits: width and height must lie in 1..65535");
  }
  const std::vector<uint16_t> xs(x.data(), x.data() + n), ys(y.data(), y.data() + n);
  const wire::RegionMasks r{static_cast<uint16_t>(width), static_cast<uint16_t>(height), span_of(t_lo_ns),
                            span_of(t_hi_ns), span_of(bits)};
  std::vector<uint8_t> out(n);
  {
    nb::gil_scoped_release release;
    wire::region_bits(xs, ys, span_of(t_ns), r, out.data());
  }
  return to_numpy(std::move(out), {n});
}

// One block's build (ui.md §4.2 step 2): the `ev` words of the whole block in LOD order (pack_block: lod_order and
// pack_ev fused, no perm kept) and the counts of the rate bins it meets.
nb::dict build_block(const Col<int64_t>& t_ns, const Col<uint16_t>& x, const Col<uint16_t>& y, const Col<uint8_t>& p,
                     const std::optional<Col<uint8_t>>& label, int64_t t_base_ns, int64_t rate_t0_ns,
                     int64_t rate_bin_ns, int64_t rate_bins) {
  const size_t n = t_ns.shape(0);
  if (label) same_length(n, label->shape(0), "build_block");
  if (rate_bins < 0 || rate_bins > (int64_t{1} << 26)) throw std::invalid_argument("build_block: rate_bins out of range");
  const size_t nbins = static_cast<size_t>(rate_bins);
  std::vector<uint32_t> ev(2 * n), total(nbins, 0), on(nbins, 0);
  wire::BlockOrder order;
  {
    nb::gil_scoped_release release;
    order = wire::pack_block(span_of(t_ns), span_of(x), span_of(y), span_of(p), label ? label->data() : nullptr,
                             t_base_ns, ev.data());
    wire::rate_counts(span_of(t_ns), span_of(p), rate_t0_ns, rate_bin_ns, total, on);
  }
  nb::dict d;
  d["n_ge"] = n_ge_list(order.n_ge);
  d["duplicates"] = order.duplicates;
  d["ev"] = to_numpy(std::move(ev), {n, 2});
  d["total"] = to_numpy(std::move(total), {nbins});
  d["on"] = to_numpy(std::move(on), {nbins});
  return d;
}

}  // namespace

void bind_wire(nb::module_& m) {
  nb::module_ w = m.def_submodule("wire", "The browser UI's event wire (docs/design/phase3/ui.md §3.5, §3.10).");
  w.attr("BLOCK_SHIFT") = wire::kBlockShift;
  w.attr("LEVELS") = wire::kLevels;
  w.attr("MAX_COORD") = wire::kMaxCoord;
  w.attr("LOD_KEY_VERSION") = wire::kLodKeyVersion;
  w.def("event_key", &event_key, nb::arg("t_ns").noconvert(), nb::arg("x").noconvert(), nb::arg("y").noconvert(),
        nb::arg("p").noconvert(), nb::arg("k").noconvert(),
        "LOD keys mix(mix(t_ns) ^ (x | y<<16 | p<<32 | k<<33)) (uint64), SplitMix64's mixer (§4.1).");
  w.def("lod_level", &lod_level, nb::arg("key").noconvert(), "Levels min(8, clz32(key >> 32)) (uint8).");
  w.def("lod_order", &lod_order, nb::arg("t_ns").noconvert(), nb::arg("x").noconvert(), nb::arg("y").noconvert(),
        nb::arg("p").noconvert(),
        "One time-sorted block -> {perm (uint32, LOD position -> file index), n_ge (9 counts of level >= j), "
        "duplicates (events with k > 0)}: level descending, file order within a level.");
  w.def("pack_ev", &pack_ev, nb::arg("t_ns").noconvert(), nb::arg("x").noconvert(), nb::arg("y").noconvert(),
        nb::arg("p").noconvert(), nb::arg("label").noconvert().none(), nb::arg("perm").noconvert().none(),
        nb::arg("i0"), nb::arg("i1"), nb::arg("t_base_ns"),
        "BSEV `ev` words (n, 2) uint32 of order positions [i0, i1) (perm None: time order): x | y<<14 | tag<<28 | "
        "p<<31 and t - t_base_ns; tag = 1 + the S1 label (label None: 0).");
  w.def("gather", &gather, nb::arg("perm").noconvert(), nb::arg("columns"),
        "Columns (one dtype, one entry per event) in perm's order, interleaved: (n,) for one 1-D column, else "
        "(n, components).");
  w.def("rate_counts", &rate_counts, nb::arg("t_ns").noconvert(), nb::arg("p").noconvert().none(), nb::arg("t0_ns"),
        nb::arg("bin_ns"), nb::arg("n_bins"),
        "BSRT bins [t0_ns + i bin_ns, t0_ns + (i + 1) bin_ns): (total, on) uint32 counts (on None without p).");
  w.def("c4_times", &c4_times, nb::arg("t_us").noconvert(), nb::arg("t_ns_rem").noconvert().none(),
        nb::arg("t_offset_us"), "Exact ns of an event file (contract C4): 1000 (t_offset_us + t_us) + t_ns_rem.");
  w.def("polarity01", &polarity01, nb::arg("p").noconvert(), nb::arg("invert") = false,
        "Polarity as uint8 0/1 with 1 = ON: (p != 0) xor invert.");
  w.def("region_bits", &region_bits, nb::arg("x").noconvert(), nb::arg("y").noconvert(), nb::arg("t_ns").noconvert(),
        nb::arg("width"), nb::arg("height"), nb::arg("t_lo_ns").noconvert(), nb::arg("t_hi_ns").noconvert(),
        nb::arg("bits").noconvert(),
        "Region test per event (uint8 0/1): (x, y) set in the mask whose interval [t_lo_ns, t_hi_ns) holds t_ns; "
        "intervals ordered and disjoint, ceil(W H / 32) uint32 words per mask, pixel y W + x at bit (i & 31) of word "
        "i >> 5.");
  w.def("build_block", &build_block, nb::arg("t_ns").noconvert(), nb::arg("x").noconvert(), nb::arg("y").noconvert(),
        nb::arg("p").noconvert(), nb::arg("label").noconvert().none(), nb::arg("t_base_ns"), nb::arg("rate_t0_ns"),
        nb::arg("rate_bin_ns"), nb::arg("rate_bins"),
        "One block's build: {n_ge, duplicates, ev (n, 2) uint32 in LOD order, total and on (uint32 counts of the "
        "rate bins [rate_t0_ns + i rate_bin_ns, ...))}; p must be 0/1.");
}

}  // namespace balla::bindings
