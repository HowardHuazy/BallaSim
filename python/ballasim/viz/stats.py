"""Event-stream statistics and their plots.

- event rate against time, ON and OFF separately, and the ON fraction;
- rate spectrum of binned counts as vector strength, which shows the flicker of
  modulated lights: in EVIMO2 the 120 Hz mains and the 200 Hz Vicon strobe
  (docs/research G3 §2.7). The vector strength is reported at the fundamental
  and the second harmonic of every modulated light the streams saw
  (:func:`light_frequencies`, D16b): a BallaSim run's light groups (run.json),
  and the EVIMO2 release facts (profile ``evimo2/release@v0``) for EVIMO2
  recordings and for EVIMO2 runs that predate the record;
- inter-event intervals, per pixel and for the whole stream;
- per-pixel event rates as a histogram and as a map.

Every statistic can be restricted to a region (the :class:`Region` protocol): a
:class:`GtRegion`, the pixels where a reference stream's GT has modelled geometry, so a
simulation without the room is compared with a recording only where both are defined;
or a :class:`MaskRegion`, masks on time intervals (the browser UI's regions, BSMK,
docs/design/phase3/ui.md §4.5), whose per-event test is C++ (``_core.wire.region_bits``).

Everything streams through the events in chunks, so long recordings fit in memory.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .events import NS_PER_MS, NS_PER_S, Events, EventSource

#: ``none``: whole sensor; ``valid``: inside the GT validity mask of the first stream that has masks.
STATS_MASK_MODES = ("none", "valid")


def group_frequencies(light_groups: Sequence[Mapping]) -> dict[float, str]:
    """{fundamental (Hz): group name} of the modulated light groups (C1 LightGroup dicts): a Fourier group's f0, a pulse
    train's frequency; constant groups have none."""
    out: dict[float, str] = {}
    for g in light_groups:
        w = g["waveform"]
        f = w.get("f0_hz") if w["type"] == "fourier" else w.get("f_hz") if w["type"] == "pulse" else None
        if f is not None:
            out.setdefault(float(f), str(g["name"]))
    return out


def light_frequencies(source: EventSource) -> dict[float, str]:
    """The fundamentals of the modulated light a stream saw, {Hz: name} (D16b): a BallaSim run's light groups (run.json
    next to its ``events/`` directory), else the EVIMO2 release facts (``evimo2/release@v0``: mains 120 Hz, strobe
    200 Hz) for an EVIMO2 recording or a run of an EVIMO2 scene that predates the record; {} otherwise."""
    from .io import Evimo2Sequence, H5Events, evimo2_release

    evimo2 = isinstance(source, Evimo2Sequence)
    if isinstance(source, H5Events):
        run_json = source.path.parent.parent / "run.json"
        try:
            doc = json.loads(run_json.read_text())
        except (OSError, ValueError):
            return {}
        if "light_groups" in doc:
            return group_frequencies(doc["light_groups"])
        evimo2 = str(((doc.get("bundle") or {}).get("source") or {}).get("dataset", "")).startswith("EVIMO2")
    if not evimo2:
        return {}
    release = evimo2_release()
    return {float(release["mains_hz"]): "mains", float(release["strobe_hz"]): "strobe"}


def reported_frequencies(lights: Mapping[float, str]) -> tuple[float, ...]:
    """The frequencies whose vector strength is reported: every light's fundamental, then its second harmonic."""
    return tuple(dict.fromkeys([*lights, *(2 * f for f in lights)]))


@dataclass(frozen=True)
class StatsOptions:
    rate_bin_ms: float = 1.0
    on_fraction_bin_ms: float = 10.0
    spectrum_bin_us: float = 100.0
    spectrum_segment_s: float = 1.0
    peak_band_hz: tuple[float, float] = (20.0, 1000.0)
    max_interval_events: int = 10_000_000


@dataclass(frozen=True)
class BinnedCounts:
    """ON and OFF event counts in back-to-back bins starting at ``t0_ns``."""

    t0_ns: int
    bin_ns: int
    on: np.ndarray
    off: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.on + self.off

    @property
    def centers_s(self) -> np.ndarray:
        return (self.t0_ns + (np.arange(len(self.on)) + 0.5) * self.bin_ns) / NS_PER_S

    def rebin(self, factor: int) -> BinnedCounts:
        n = len(self.on) // factor * factor
        return BinnedCounts(
            self.t0_ns,
            self.bin_ns * factor,
            self.on[:n].reshape(-1, factor).sum(1),
            self.off[:n].reshape(-1, factor).sum(1),
        )


@dataclass(frozen=True)
class Spectrum:
    """Welch-averaged (Hann, 50 % overlap) RMS vector strength of binned counts.

    Vector strength at f is |Σ_k c_k e^{-2πi f t_k}| / Σ_k c_k: 0 for a constant
    rate, m/2 for a rate modulated as 1 + m cos(2π f t).
    """

    freq_hz: np.ndarray
    vs: np.ndarray
    n_segments: int


@dataclass(frozen=True)
class Intervals:
    """Inter-event intervals in ns: between consecutive events of the same pixel,
    and between consecutive events of the stream."""

    per_pixel_ns: np.ndarray
    stream_ns: np.ndarray
    n_events: int
    truncated: bool

    def histograms(self, edges_s: np.ndarray) -> dict[str, tuple[np.ndarray, int, int]]:
        """Per kind (``per_pixel``, ``stream``): the counts of the positive intervals between ``edges_s`` (seconds),
        the number of zero intervals and the number of all intervals."""
        out = {}
        for name, values in (("per_pixel", self.per_pixel_ns), ("stream", self.stream_ns)):
            seconds = values / NS_PER_S
            out[name] = (np.histogram(seconds[seconds > 0], edges_s)[0], int(np.count_nonzero(values == 0)),
                         len(values))
        return out


def interval_edges_s() -> np.ndarray:
    """The interval histograms' edges in seconds: 20 bins per decade from 0.1 µs to 10 s."""
    return _log_bins(1e-7, 10.0)


class GtRegion:
    """The pixels where a stream's GT has modelled geometry: label > 0 in its GT mask
    frame nearest to each event (the region ``--mask valid`` outlines in the viewer).

    Statistics restricted to it compare a simulation with a recording only where
    both are defined; EVIMO2 does not model the room. Masks are looked up once per
    1 ms slot (at the slot centre, on the common clock); events in slots without a
    GT frame nearby (Vicon gaps, the end of a sequence) are left out, and so is that
    time.
    """

    SLOT_NS = NS_PER_MS

    def __init__(self, source: EventSource) -> None:
        if "mask" not in source.gt_kinds:
            raise ValueError(f"{source.label}: no GT masks to define a valid region")
        self.source = source
        self._last: tuple[int, np.ndarray] | None = None  # (frame time, region) of the last frame used

    @property
    def label(self) -> str:
        return self.source.label

    def mask_at(self, t_ns: int) -> np.ndarray | None:
        """(H, W) bool region of the GT frame nearest to ``t_ns``; None without one."""
        frame = self.source.gt_frame("mask", t_ns)
        if frame is None:
            return None
        if self._last is None or self._last[0] != frame.t_ns:
            self._last = (frame.t_ns, frame.data > 0)
        return self._last[1]

    def check(self, source: EventSource) -> None:
        if source.size != self.source.size:
            raise ValueError(
                f"{source.label} ({source.width}x{source.height}) cannot use the GT region of {self.label} "
                f"({self.source.width}x{self.source.height})"
            )

    def select(self, events: Events, t0_ns: int) -> np.ndarray:
        """Per event: inside the region of its 1 ms slot (slots counted from ``t0_ns``)."""
        keep = np.zeros(len(events), bool)
        if not len(events):
            return keep
        slots = (events.t_ns - t0_ns) // self.SLOT_NS
        starts = np.flatnonzero(np.r_[True, slots[1:] != slots[:-1]])  # events are time-sorted
        ends = np.r_[starts[1:], len(events)]
        for a, b in zip(starts, ends, strict=True):
            mask = self.mask_at(t0_ns + int(slots[a]) * self.SLOT_NS + self.SLOT_NS // 2)
            if mask is not None:
                keep[a:b] = mask[events.y[a:b], events.x[a:b]]
        return keep

    def coverage(self, t0_ns: int, n_bins: int, bin_ns: int) -> RegionCoverage:
        """Which bins ``[t0 + k bin, t0 + (k + 1) bin)`` have a GT frame (by the slot of their
        centre), the mean share of the image inside the region and the union of the regions."""
        slot = (np.arange(n_bins, dtype=np.int64) * bin_ns + bin_ns // 2) // self.SLOT_NS
        covered = np.zeros(n_bins, bool)
        union = np.zeros((self.source.height, self.source.width), bool)
        fractions = []
        for s in np.unique(slot):
            mask = self.mask_at(t0_ns + int(s) * self.SLOT_NS + self.SLOT_NS // 2)
            if mask is not None:
                covered[slot == s] = True
                union |= mask
                fractions.append(float(mask.mean()))
        return RegionCoverage(self.label, covered, union, float(np.mean(fractions)) if fractions else 0.0)

    def mask_region(self, t0_ns: int, t1_ns: int, *, max_slots: int | None = None,
                    max_masks: int | None = None) -> MaskRegion:
        """This region over ``[t0_ns, t1_ns)`` as a :class:`MaskRegion` on the source's clock: an interval per run of
        1 ms slots (counted from ``t0_ns``) that share a GT frame, the last one cut at ``t1_ns``, holding that frame's
        mask; slots without a frame are left out. It selects exactly the events this region selects for statistics of
        ``[t0_ns, t1_ns)`` and covers the same bins; its ``pixel_fraction`` weights the slots by their bins.
        ``max_slots``, ``max_masks``: refuse (ValueError) a span of more 1 ms slots or GT frames.

        The masks are packed into one array whose rows are reserved up front, one per slot or at most ``max_masks``;
        rows never written are never touched, so a build holds its masks once."""
        n = max(0, -(-(int(t1_ns) - int(t0_ns)) // self.SLOT_NS))
        if max_slots is not None and n > max_slots:
            raise ValueError(f"the span holds {n} slots of 1 ms; at most {max_slots}")
        w, h = self.source.width, self.source.height
        words = np.empty((n if max_masks is None else min(n, max_masks), mask_words(w, h)), np.uint32)
        edges, count = [], 0
        key, start = None, 0

        def close(end: int) -> None:
            if key is not None:
                edges.append((int(t0_ns) + start * self.SLOT_NS, min(int(t0_ns) + end * self.SLOT_NS, int(t1_ns))))

        for s in range(n):
            frame = self.source.gt_frame("mask", int(t0_ns) + s * self.SLOT_NS + self.SLOT_NS // 2)
            k = None if frame is None else frame.t_ns
            if k != key:
                close(s)
                key, start = k, s
                if frame is not None:
                    if count == len(words):                 # only with max_masks: a frame per slot fits
                        raise ValueError(f"the span meets more than {max_masks} GT frames")
                    words[count] = pack_masks((frame.data > 0)[None])[0]
                    count += 1
        close(n)
        return MaskRegion(np.asarray(edges, np.int64).reshape(-1, 2), words[:count], width=w, height=h,
                          label=self.label)


@dataclass(frozen=True)
class RegionCoverage:
    """Where and when a :class:`GtRegion` applied to a statistics span."""

    label: str  # the stream whose GT masks define the region
    covered: np.ndarray  # bool per spectrum bin: a GT frame was available
    union: np.ndarray  # (H, W) bool: inside the region at some covered time
    pixel_fraction: float  # mean share of the image inside the region over covered slots


class Region(Protocol):
    """What the statistics need of a region (:class:`GtRegion`, :class:`MaskRegion`)."""

    @property
    def label(self) -> str: ...

    def check(self, source: EventSource) -> None:
        """Raises ValueError when ``source`` cannot use the region (another resolution)."""

    def select(self, events: Events, t0_ns: int) -> np.ndarray:
        """Per event of a time-sorted batch of ``[t0_ns, …)``: inside the region."""

    def coverage(self, t0_ns: int, n_bins: int, bin_ns: int) -> RegionCoverage:
        """Which bins ``[t0 + k bin, t0 + (k + 1) bin)`` the region covers, and where."""


_I64_MIN, _I64_MAX = np.iinfo(np.int64).min, np.iinfo(np.int64).max


def mask_words(width: int, height: int) -> int:
    """uint32 words per packed mask: ceil(W H / 32)."""
    return -(-width * height // 32)


def pack_masks(masks: np.ndarray) -> np.ndarray:
    """(M, H, W) bool masks as (M, ceil(W H / 32)) uint32 words: pixel i = y W + x at bit (i & 31) of word i >> 5
    (the BSMK layout, docs/design/phase3/ui.md §3.5)."""
    masks = np.asarray(masks, bool)
    m, h, w = masks.shape
    bits = np.zeros((m, 32 * mask_words(w, h)), bool)
    bits[:, : w * h] = masks.reshape(m, w * h)
    return np.packbits(bits, axis=1, bitorder="little").view("<u4").astype(np.uint32)


def unpack_masks(words: np.ndarray, width: int, height: int) -> np.ndarray:
    """(M, ceil(W H / 32)) uint32 words as (M, H, W) bool masks (the inverse of :func:`pack_masks`)."""
    words = np.ascontiguousarray(words, "<u4").reshape(-1, mask_words(width, height))
    bits = np.unpackbits(words.view(np.uint8), axis=1, bitorder="little")
    return bits[:, : width * height].reshape(-1, height, width).astype(bool)


def _shifted(times: np.ndarray, d: int) -> np.ndarray:
    """times − d, saturated at the int64 limits."""
    if d == 0:
        return times
    t = times.astype(np.int64)
    if d > 0:
        return np.where(t < _I64_MIN + d, _I64_MIN, t - d)
    return np.where(t > _I64_MAX + d, _I64_MAX, t - d)


def _coordinates(a: np.ndarray) -> np.ndarray:
    """Coordinates as uint16 for ``region_bits``; values outside 0..65535 become 65535, which no image holds."""
    a = np.asarray(a)
    if a.dtype == np.uint16:
        return np.ascontiguousarray(a)
    b = a.astype(np.int64)
    return np.where((b < 0) | (b > 0xFFFF), 0xFFFF, b).astype(np.uint16)


class MaskRegion:
    """Masks on time intervals: mask m applies to view-clock times ``[t_lo[m], t_hi[m])`` (ordered, disjoint), time
    outside every interval is outside the region (the browser UI's regions, BSMK, docs/design/phase3/ui.md §3.5, §4.5).

    - ``t_edges``: (M, 2) int64 ``[t_lo, t_hi)`` per mask on the view clock;
    - ``masks``: (M, H, W) bool, or (M, ceil(W H / 32)) uint32 words in the BSMK layout (:func:`pack_masks`) with
      ``width`` and ``height`` given;
    - ``offset_ns``: where the statistics' stream lies on the view clock (t_view = t + offset_ns): its events and bins
      are tested at t + offset_ns;
    - ``covered``: None, or ``(t0_ns, window_ns, flags)``: only the windows ``[t0 + k window, t0 + (k + 1) window)``
      of the view clock with ``flags[k]`` set are covered (F043's count-eligible 10 ms windows, ui.md §4.5); the
      intervals are cut to them, so their events drop out and their time leaves the covered time.

    The per-event test is C++ (``_core.wire.region_bits``). ``coverage`` covers a bin when its centre lies in an
    interval; ``pixel_fraction`` is the mean share of the image inside the masks over the covered bins.
    :meth:`GtRegion.mask_region` gives the masks that select as a :class:`GtRegion` does.
    """

    def __init__(self, t_edges, masks, offset_ns: int = 0, covered=None, *, width: int | None = None,
                 height: int | None = None, label: str = "region") -> None:
        edges = np.asarray(t_edges, np.int64).reshape(-1, 2)
        masks = np.asarray(masks)
        if masks.dtype == bool or masks.ndim == 3:
            if masks.ndim != 3 or len(masks) != len(edges):
                raise ValueError("masks: (M, H, W) bool, one per interval")
            height, width = masks.shape[1:]
            words = pack_masks(masks)
        else:
            if width is None or height is None:
                raise ValueError("packed masks need width and height")
            words = np.ascontiguousarray(masks, np.uint32).reshape(len(edges), mask_words(width, height))
        if not (0 < width <= 0xFFFF and 0 < height <= 0xFFFF):
            raise ValueError(f"a {width}x{height} region")
        lo, hi = edges[:, 0], edges[:, 1]
        if np.any(lo > hi) or np.any(hi[:-1] > lo[1:]):
            raise ValueError("the mask intervals must be ordered and disjoint")
        index = np.arange(len(edges))
        if covered is not None:
            lo, hi, index = self._cut(lo, hi, index, *covered)
        keep = lo < hi
        self.t_lo, self.t_hi, self.mask_index = lo[keep], hi[keep], index[keep]
        self.words = words
        self.width, self.height = int(width), int(height)
        self.offset_ns = int(offset_ns)
        self._label = label
        self._bits: np.ndarray | None = None

    @staticmethod
    def _cut(lo, hi, index, t0_ns: int, window_ns: int, flags):
        """The intervals cut to the flagged windows (each run of flagged windows is one segment)."""
        flags = np.asarray(flags, bool)
        if window_ns <= 0:
            raise ValueError("covered: the window must be positive")
        e = np.diff(np.r_[0, flags.astype(np.int8), 0])
        out_lo, out_hi, out_index = [], [], []
        for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1), strict=True):
            s0, s1 = t0_ns + int(a) * window_ns, t0_ns + int(b) * window_ns
            k = np.arange(np.searchsorted(hi, s0, side="right"), np.searchsorted(lo, s1, side="left"))
            out_lo.append(np.maximum(lo[k], s0))
            out_hi.append(np.minimum(hi[k], s1))
            out_index.append(index[k])
        if not out_lo:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(out_lo), np.concatenate(out_hi), np.concatenate(out_index)

    @property
    def label(self) -> str:
        return self._label

    @property
    def bits(self) -> np.ndarray:
        """The words of each interval's mask, (intervals, ceil(W H / 32)) uint32."""
        if self._bits is None:
            same = len(self.mask_index) == len(self.words) and np.array_equal(self.mask_index,
                                                                              np.arange(len(self.words)))
            self._bits = self.words if same else np.ascontiguousarray(self.words[self.mask_index])
        return self._bits

    def with_offset(self, offset_ns: int) -> MaskRegion:
        """The same region for a stream at another offset onto the view clock."""
        view = copy.copy(self)
        view.offset_ns = int(offset_ns)
        return view

    def check(self, source: EventSource) -> None:
        if (source.width, source.height) != (self.width, self.height):
            raise ValueError(
                f"{source.label} ({source.width}x{source.height}) cannot use the region {self.label} "
                f"({self.width}x{self.height})"
            )

    def select(self, events: Events, t0_ns: int = 0) -> np.ndarray:
        """Per event: inside the mask of the interval that holds its view-clock time (``t0_ns`` is not used: the
        intervals are absolute)."""
        from ballasim import _core

        if not len(events):
            return np.zeros(0, bool)
        bits = self.bits.reshape(-1)
        inside = _core.wire.region_bits(_coordinates(events.x), _coordinates(events.y),
                                        np.ascontiguousarray(events.t_ns, np.int64), self.width, self.height,
                                        _shifted(self.t_lo, self.offset_ns), _shifted(self.t_hi, self.offset_ns), bits)
        return inside.view(bool)

    def coverage(self, t0_ns: int, n_bins: int, bin_ns: int) -> RegionCoverage:
        centre = (int(t0_ns) + self.offset_ns + bin_ns // 2) + np.arange(n_bins, dtype=np.int64) * bin_ns
        k = np.searchsorted(self.t_lo, centre, side="right") - 1
        covered = (k >= 0) & (centre < self.t_hi[np.maximum(k, 0)]) if len(self.t_lo) else np.zeros(n_bins, bool)
        used = self.mask_index[k[covered]]
        union = np.zeros((self.height, self.width), bool)
        fraction = 0.0
        if len(used):
            unique = np.unique(used)
            union = unpack_masks(np.bitwise_or.reduce(self.words[unique], axis=0)[None], self.width, self.height)[0]
            counts = np.bitwise_count(self.words).sum(axis=1, dtype=np.int64)
            fraction = float(counts[used].mean()) / (self.width * self.height)
        return RegionCoverage(self.label, covered, union, fraction)


@dataclass
class StreamStats:
    label: str
    t0_ns: int
    t1_ns: int
    counts: BinnedCounts  # at the spectrum bin width
    spectrum: Spectrum
    vector_strength: dict[float, float]  # at reported_frequencies(lights)
    intervals: Intervals
    pixel_rate_hz: np.ndarray  # (H, W) ON + OFF, per second of covered time
    options: StatsOptions = field(default_factory=StatsOptions)
    region: RegionCoverage | None = None  # statistics restricted to a GT region
    events_in_covered_time: int | None = None  # with a region: every event of the stream in covered time
    lights: dict[float, str] = field(default_factory=dict)  # {fundamental (Hz): light name}

    def rate_counts(self) -> BinnedCounts:
        return self.counts.rebin(self._rate_factor())

    def _rate_factor(self) -> int:
        return max(1, round(self.options.rate_bin_ms * NS_PER_MS / self.counts.bin_ns))

    def covered(self, factor: int = 1) -> np.ndarray:
        """Bool per bin of ``counts.rebin(factor)``: every sub-bin had GT (always True without a region)."""
        n = len(self.counts.on) // factor
        if self.region is None:
            return np.ones(n, bool)
        return self.region.covered[: n * factor].reshape(n, factor).all(axis=1)

    @property
    def covered_s(self) -> float:
        """Time the statistics cover: the span, less the time without GT under a region."""
        if self.region is None:
            return (self.t1_ns - self.t0_ns) / NS_PER_S
        return int(self.region.covered.sum()) * self.counts.bin_ns / NS_PER_S

    def summary(self) -> dict:
        """Scalar statistics; a value that is undefined for this stream (e.g. the ON
        fraction without events) is None, so the summary is strict JSON. Under a
        region, rates are per second of covered time and the per-pixel statistics
        cover the pixels inside the region at some covered time."""
        duration_s = self.covered_s
        on, off = int(self.counts.on.sum()), int(self.counts.off.sum())
        n = on + off
        factor_1ms = max(1, round(NS_PER_MS / self.counts.bin_ns))
        rate_1ms = self.counts.rebin(factor_1ms).total[self.covered(factor_1ms)]
        rates = self.pixel_rate_hz.ravel() if self.region is None else self.pixel_rate_hz[self.region.union]
        active = rates[rates > 0]
        pp, st = self.intervals.per_pixel_ns, self.intervals.stream_ns
        positive = st[st > 0]
        summary = {
            "label": self.label,
            "t0_s": self.t0_ns / NS_PER_S,
            "t1_s": self.t1_ns / NS_PER_S,
            "events": n,
            "mean_rate_mev_s": n / duration_s / 1e6 if duration_s > 0 else None,
            "peak_rate_1ms_mev_s": float(rate_1ms.max()) / 1e3 if len(rate_1ms) else 0.0,
            "on_fraction": on / n if n else None,
            "vector_strength": {f"{f:g}": v for f, v in self.vector_strength.items()},
            "spectral_peaks_hz": spectral_peaks(self.spectrum, *self.options.peak_band_hz),
            "silent_pixel_fraction": 1.0 - len(active) / len(rates) if len(rates) else None,
            "pixel_rate_hz_median_active": float(np.median(active)) if len(active) else 0.0,
            "pixel_rate_hz_p99": float(np.percentile(rates, 99)) if len(rates) else None,
            "per_pixel_interval_ms_median": float(np.median(pp)) / NS_PER_MS if len(pp) else None,
            "stream_interval_zero_fraction": float(np.mean(st == 0)) if len(st) else None,
            "stream_interval_min_positive_us": float(positive.min()) / 1e3 if len(positive) else None,
            "intervals_from_events": self.intervals.n_events,
            "intervals_truncated": self.intervals.truncated,
        }
        if self.region is not None:
            total = self.events_in_covered_time
            summary["region"] = {
                "mask_of": self.region.label,
                "time_fraction": float(self.region.covered.mean()) if len(self.region.covered) else 0.0,
                "pixel_fraction": self.region.pixel_fraction,
                "pixels_ever_inside": int(self.region.union.sum()),
                "event_share": n / total if total else None,  # of the stream's events in covered time
            }
        return summary


def _selected(source: EventSource, t0_ns: int, t1_ns: int, region: Region | None) -> Iterator[Events]:
    """Chunks of ``[t0_ns, t1_ns)``, restricted to ``region`` (slots counted from ``t0_ns``)."""
    for chunk in source.iter_chunks(t0_ns, t1_ns):
        if region is None:
            yield chunk
        else:
            keep = region.select(chunk, t0_ns)
            yield Events(chunk.x[keep], chunk.y[keep], chunk.p[keep], chunk.t_ns[keep])


def binned_counts(
    source: EventSource, t0_ns: int, t1_ns: int, bin_ns: int, region: Region | None = None
) -> BinnedCounts:
    """ON/OFF counts in the full bins of width ``bin_ns`` that fit in ``[t0_ns, t1_ns)``."""
    n_bins = (t1_ns - t0_ns) // bin_ns
    if n_bins <= 0:
        raise ValueError(f"[{t0_ns}, {t1_ns}) ns is shorter than one {bin_ns} ns bin")
    on = np.zeros(n_bins, np.int64)
    off = np.zeros(n_bins, np.int64)
    for chunk in _selected(source, t0_ns, t0_ns + n_bins * bin_ns, region):
        b = (chunk.t_ns - t0_ns) // bin_ns
        on += np.bincount(b[chunk.p], minlength=n_bins)
        off += np.bincount(b[~chunk.p], minlength=n_bins)
    return BinnedCounts(t0_ns, bin_ns, on, off)


def rate_spectrum(counts: np.ndarray, bin_ns: int, segment_s: float = 1.0) -> Spectrum:
    """RMS vector strength over Hann-windowed segments of ``segment_s`` (frequency
    resolution 1/segment_s) with 50 % overlap; one segment if the data is shorter."""
    counts = np.asarray(counts, np.float64)
    seg = int(min(len(counts), round(segment_s * NS_PER_S / bin_ns)))
    if seg < 8:
        raise ValueError("too few bins for a spectrum")
    starts = np.arange(0, len(counts) - seg + 1, max(seg // 2, 1))
    segments = np.lib.stride_tricks.sliding_window_view(counts, seg)[starts]
    window = np.hanning(seg)
    weight = segments @ window
    keep = weight > 0
    segments, weight = segments[keep], weight[keep]
    if len(segments) == 0:
        return Spectrum(np.fft.rfftfreq(seg, bin_ns / NS_PER_S), np.zeros(seg // 2 + 1), 0)
    centered = (segments - segments.mean(axis=1, keepdims=True)) * window
    vs = np.abs(np.fft.rfft(centered, axis=1)) / weight[:, None]
    return Spectrum(np.fft.rfftfreq(seg, bin_ns / NS_PER_S), np.sqrt(np.mean(vs**2, axis=0)), len(segments))


def vector_strength(
    counts: np.ndarray, bin_ns: int, freqs_hz: Sequence[float], valid: np.ndarray | None = None
) -> dict[float, float]:
    """Vector strength at exactly ``freqs_hz`` over the whole binned record (over the
    ``valid`` bins only, if given)."""
    counts = np.asarray(counts, np.float64)
    t_s = (np.arange(len(counts)) + 0.5) * (bin_ns / NS_PER_S)
    if valid is not None:
        counts, t_s = counts[valid], t_s[valid]
    total = counts.sum()
    return {float(f): float(np.abs(counts @ np.exp(-2j * np.pi * f * t_s)) / total) if total else 0.0 for f in freqs_hz}


def spectral_peaks(
    spectrum: Spectrum, f_min_hz: float = 20.0, f_max_hz: float = 1000.0, count: int = 6, min_ratio: float = 3.0
) -> list[dict]:
    """The strongest local maxima in ``[f_min_hz, f_max_hz]`` that stand at least
    ``min_ratio`` times above the running median (±10 bins) of the spectrum."""
    f, vs = spectrum.freq_hz, spectrum.vs
    if len(vs) < 23:
        return []
    half = 10
    padded = np.pad(vs, half, mode="edge")
    floor = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=1)
    is_max = np.zeros(len(vs), bool)
    is_max[1:-1] = (vs[1:-1] > vs[:-2]) & (vs[1:-1] >= vs[2:])
    ratio = np.divide(vs, floor, out=np.zeros_like(vs), where=floor > 0)
    candidates = np.flatnonzero(is_max & (f >= f_min_hz) & (f <= f_max_hz) & (ratio >= min_ratio))
    best = candidates[np.argsort(vs[candidates])[::-1][:count]]
    return [{"hz": float(f[i]), "vs": float(vs[i]), "above_floor": float(ratio[i])} for i in sorted(best)]


def inter_event_intervals(
    source: EventSource, t0_ns: int, t1_ns: int, max_events: int = 10_000_000, region: Region | None = None
) -> Intervals:
    """Intervals over the first ``max_events`` events of ``[t0_ns, t1_ns)`` (of those inside
    ``region``, between consecutive events that are both inside it)."""
    i0, i1 = source.search(t0_ns), source.search(t1_ns)
    truncated = i1 - i0 > max_events
    events = source.read(i0, min(i1, i0 + max_events))
    if region is not None:
        keep = region.select(events, t0_ns)
        events = Events(events.x[keep], events.y[keep], events.p[keep], events.t_ns[keep])
    pixel = events.y.astype(np.int64) * source.width + events.x
    order = np.argsort(pixel, kind="stable")  # stable keeps each pixel's events in time order
    sorted_pixel, sorted_t = pixel[order], events.t_ns[order]
    same = sorted_pixel[1:] == sorted_pixel[:-1]
    return Intervals(
        per_pixel_ns=(sorted_t[1:] - sorted_t[:-1])[same],
        stream_ns=np.diff(events.t_ns),
        n_events=len(events),
        truncated=truncated,
    )


def pixel_rates(
    source: EventSource, t0_ns: int, t1_ns: int, region: Region | None = None, duration_s: float | None = None
) -> np.ndarray:
    """Events per pixel over ``[t0_ns, t1_ns)`` (inside ``region``), per second of
    ``duration_s`` (default: the span), (H, W)."""
    total = np.zeros(source.width * source.height, np.int64)
    for chunk in _selected(source, t0_ns, t1_ns, region):
        x, y = chunk.x.astype(np.int64), chunk.y.astype(np.int64)
        inside = (x < source.width) & (y < source.height)
        total += np.bincount((y * source.width + x)[inside], minlength=total.size)
    duration_s = (t1_ns - t0_ns) / NS_PER_S if duration_s is None else duration_s
    return (total / duration_s if duration_s > 0 else total * 0.0).reshape(source.height, source.width)


def compute_stats(
    source: EventSource,
    t0_ns: int,
    t1_ns: int,
    options: StatsOptions | None = None,
    region: Region | None = None,
    lights: Mapping[float, str] | None = None,
) -> StreamStats:
    """Statistics of ``[t0_ns, t1_ns)`` (full spectrum bins), restricted to ``region`` if given.

    The vector strength is reported at the fundamentals of ``lights`` ({Hz: name}; default: the stream's own,
    :func:`light_frequencies`) and their second harmonics. Under a region, the vector strengths use the covered bins
    only, and the spectrum fills the bins without GT with the mean covered rate, so gaps add no lines (a segment with
    gaps shows its lines lowered by its uncovered share)."""
    options = options or StatsOptions()
    lights = dict(light_frequencies(source) if lights is None else lights)
    freqs = reported_frequencies(lights)
    bin_ns = round(options.spectrum_bin_us * 1000)
    if region is not None:
        region.check(source)
    counts = binned_counts(source, t0_ns, t1_ns, bin_ns, region)
    t1_ns = t0_ns + len(counts.on) * bin_ns  # full bins only, the same span for every statistic
    total = counts.total.astype(np.float64)
    coverage = None if region is None else region.coverage(t0_ns, len(total), bin_ns)
    events_in_covered_time = None
    duration_s = (t1_ns - t0_ns) / NS_PER_S
    if coverage is not None:
        all_counts = binned_counts(source, t0_ns, t1_ns, bin_ns)
        events_in_covered_time = int(all_counts.total[coverage.covered].sum())
        vs = vector_strength(total, bin_ns, freqs, coverage.covered)
        if coverage.covered.any():
            total[~coverage.covered] = total[coverage.covered].mean()
        duration_s = int(coverage.covered.sum()) * bin_ns / NS_PER_S
    else:
        vs = vector_strength(total, bin_ns, freqs)
    return StreamStats(
        label=source.label,
        t0_ns=t0_ns,
        t1_ns=t1_ns,
        counts=counts,
        spectrum=rate_spectrum(total, bin_ns, options.spectrum_segment_s),
        vector_strength=vs,
        intervals=inter_event_intervals(source, t0_ns, t1_ns, options.max_interval_events, region),
        pixel_rate_hz=pixel_rates(source, t0_ns, t1_ns, region, duration_s),
        options=options,
        region=coverage,
        events_in_covered_time=events_in_covered_time,
        lights=lights,
    )


# Plots ----------------------------------------------------------------------------

# Reference stream in neutral ink, compared streams in the categorical slots.
_STREAM_COLORS = ("#3d3c39", "#eb6834", "#1baf7a", "#4a3aa7")
_ON_COLOR, _OFF_COLOR = "#e34948", "#2a78d6"
_MUTED = "#898781"
_STYLE = {
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "axes.titlecolor": "#0b0b0b",
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
    "xtick.labelcolor": "#52514e",
    "ytick.labelcolor": "#52514e",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.3,
    "font.family": "sans-serif",
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=130, bbox_inches="tight")
    _pyplot().close(fig)
    return path


def _log_bins(lo: float, hi: float, per_decade: int = 20) -> np.ndarray:
    return np.logspace(np.log10(lo), np.log10(hi), round(np.log10(hi / lo) * per_decade) + 1)


def _region_note(stats: Sequence[StreamStats]) -> str:
    """Title suffix naming the GT region the statistics are restricted to."""
    region = stats[0].region
    return "" if region is None else f", inside the GT valid region of {region.label}"


def _gapped(values: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """``values`` as float with NaN (a gap in the line) where there was no GT."""
    return np.where(covered, values, np.nan)


def plot_rates(stats: Sequence[StreamStats], path: Path) -> Path:
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(len(stats) + 1, 1, figsize=(10, 2.2 * (len(stats) + 1)), sharex=True)
        bin_ms = stats[0].options.rate_bin_ms
        for s, color in zip(stats, _STREAM_COLORS[: len(stats)], strict=True):
            c, covered = s.rate_counts(), s.covered(s._rate_factor())
            rate = _gapped(c.total / (c.bin_ns / NS_PER_S) / 1e6, covered)
            axes[0].plot(c.centers_s, rate, color=color, label=s.label)
        axes[0].set_title(f"Event rate, {bin_ms:g} ms bins{_region_note(stats)}")
        axes[0].set_ylabel("Mev/s")
        axes[0].legend(loc="upper right")
        for ax, s in zip(axes[1:], stats, strict=True):
            c, covered = s.rate_counts(), s.covered(s._rate_factor())
            scale = 1.0 / (c.bin_ns / NS_PER_S) / 1e6
            ax.plot(c.centers_s, _gapped(c.on * scale, covered), color=_ON_COLOR, label="ON")
            ax.plot(c.centers_s, _gapped(c.off * scale, covered), color=_OFF_COLOR, label="OFF")
            ax.set_title(f"{s.label}: ON and OFF", fontsize=9)
            ax.set_ylabel("Mev/s")
            ax.legend(loc="upper right", ncols=2)
        axes[-1].set_xlabel("time (s)")
        return _save(fig, path)


def plot_on_fraction(stats: Sequence[StreamStats], path: Path) -> Path:
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 2.8))
        bin_ms = stats[0].options.on_fraction_bin_ms
        for s, color in zip(stats, _STREAM_COLORS[: len(stats)], strict=True):
            factor = max(1, round(bin_ms * NS_PER_MS / s.counts.bin_ns))
            c = s.counts.rebin(factor)
            total = c.total
            fraction = np.divide(c.on, total, out=np.full(len(total), np.nan), where=total > 0)
            ax.plot(c.centers_s, _gapped(fraction, s.covered(factor)), color=color, label=s.label)
        ax.axhline(0.5, color="#c3c2b7", linewidth=0.8, zorder=0)
        ax.set_ylim(0, 1)
        ax.set_title(f"ON fraction, {bin_ms:g} ms bins{_region_note(stats)}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("ON / (ON + OFF)")
        ax.legend(loc="lower right")
        return _save(fig, path)


def plot_spectrum(stats: Sequence[StreamStats], path: Path, f_max_hz: float = 1000.0) -> Path:
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 3.6))
        lights: dict[float, str] = {}
        for s in stats:
            for f, name in s.lights.items():
                lights.setdefault(f, name)
        for s, color in zip(stats, _STREAM_COLORS[: len(stats)], strict=True):
            sp = s.spectrum
            sel = (sp.freq_hz > 0) & (sp.freq_hz <= f_max_hz)
            vs = ", ".join(f"VS{f:g} {s.vector_strength[f]:.4f}" for f in s.lights)
            ax.plot(sp.freq_hz[sel], sp.vs[sel], color=color, linewidth=1.0, label=s.label + (f" ({vs})" if vs else ""))
        if any(np.any(s.spectrum.vs > 0) for s in stats):  # an empty range has nothing to log-scale
            ax.set_yscale("log")
        ax.set_xlim(0, f_max_hz)
        # Harmonic lines of the lights; only the fundamentals are labelled, one row each so they never collide.
        for row, (base, name) in enumerate(lights.items()):
            for k, f in enumerate(np.arange(base, f_max_hz + 1, base)):
                ax.axvline(f, color="#c3c2b7" if k == 0 else "#e1e0d9", linewidth=0.8, zorder=0)
            ax.annotate(
                f"{name} {base:g} Hz and harmonics",
                (base, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -10 - 11 * row), textcoords="offset points",
                fontsize=7, color=_MUTED,
            )  # fmt: skip
        seg = stats[0].options.spectrum_segment_s
        ax.set_title(
            f"Rate spectrum, vector strength ({stats[0].options.spectrum_bin_us:g} us bins, {seg:g} s Hann segments)"
            + _region_note(stats)
        )
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("vector strength (RMS)")
        ax.legend(loc="upper right", frameon=True, facecolor="#fcfcfb", edgecolor="none", framealpha=0.9)
        return _save(fig, path)


def plot_intervals(stats: Sequence[StreamStats], path: Path) -> Path:
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        bins = interval_edges_s()
        for s, color in zip(stats, _STREAM_COLORS[: len(stats)], strict=True):
            histograms = s.intervals.histograms(bins)
            for ax, (hist, zeros, n) in zip(axes, (histograms["per_pixel"], histograms["stream"]), strict=True):
                zero = zeros / n if n else 0.0
                ax.stairs(hist / max(n, 1), bins, color=color, label=f"{s.label} (zero: {100 * zero:.1f} %)")
        titles = ("Per-pixel inter-event interval", "Stream inter-event interval")
        for ax, title in zip(axes, titles, strict=True):
            ax.set_xscale("log")
            ax.set_title(title + ("" if stats[0].region is None else " (GT valid region)"))
            ax.set_xlabel("interval (s)")
            ax.set_ylabel("fraction of intervals per bin")
            ax.legend(loc="upper left")
        return _save(fig, path)


def _count_edges(max_count: float) -> np.ndarray:
    """Bin edges in events: one bin per integer up to 30 (small counts are
    discrete), then about 20 bins per decade. Every edge is a half-integer, so
    each bin holds whole counts; log-spaced edges between integers would give
    neighbouring bins unequal numbers of integers and a sawtooth density."""
    edges = np.arange(0.5, 31.0)
    if max_count > edges[-1]:
        log_edges = _log_bins(edges[-1], max_count * 1.05)[1:]
        edges = np.unique(np.concatenate([edges, np.floor(log_edges) + 0.5]))
    return edges


def plot_pixel_rate_histogram(stats: Sequence[StreamStats], path: Path) -> Path:
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 3.2))
        for s, color in zip(stats, _STREAM_COLORS[: len(stats)], strict=True):
            duration_s = s.covered_s
            rates = s.pixel_rate_hz.ravel() if s.region is None else s.pixel_rate_hz[s.region.union]
            counts = np.rint(rates * duration_s)
            if not len(counts) or duration_s <= 0:
                continue
            edges = _count_edges(max(float(counts.max()), 1.0))
            hist, _ = np.histogram(counts[counts > 0], edges)
            per_decade = hist / len(counts) / np.log10(edges[1:] / edges[:-1])
            silent = float(np.mean(counts == 0))
            ax.stairs(per_decade, edges / duration_s, color=color, label=f"{s.label} (silent: {100 * silent:.1f} %)")
        ax.set_xscale("log")
        pixels = "pixels" if stats[0].region is None else "pixels inside the GT valid region at some time"
        ax.set_title(f"Per-pixel event rate (fraction of {pixels})")
        ax.set_xlabel("events/s per pixel (ON + OFF)")
        ax.set_ylabel("fraction of pixels per decade")
        ax.legend(loc="best")
        return _save(fig, path)


def plot_pixel_rate_maps(stats: Sequence[StreamStats], path: Path) -> Path:
    plt = _pyplot()
    from matplotlib.colors import LogNorm

    with plt.rc_context({**_STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(1, len(stats), figsize=(5.2 * len(stats), 4.2), squeeze=False)
        positive = np.concatenate([s.pixel_rate_hz[s.pixel_rate_hz > 0] for s in stats])
        vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
        norm = LogNorm(vmin=max(vmax / 1e4, 0.1), vmax=max(vmax, 1.0))
        for ax, s in zip(axes[0], stats, strict=True):
            image = ax.imshow(np.where(s.pixel_rate_hz > 0, s.pixel_rate_hz, np.nan), cmap="inferno", norm=norm)
            ax.set_title(s.label if s.region is None else f"{s.label} (GT valid region)", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(image, ax=axes[0].tolist(), label="events/s per pixel", shrink=0.85)
        return _save(fig, path)


def write_stats(stats: Sequence[StreamStats], out_dir: str | Path, f_max_hz: float = 1000.0) -> list[Path]:
    """Write every plot and ``summary.json`` to ``out_dir``; returns the paths."""
    if not 1 <= len(stats) <= len(_STREAM_COLORS):
        raise ValueError(f"plots take 1 to {len(_STREAM_COLORS)} streams, got {len(stats)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_rates(stats, out_dir / "rate.png"),
        plot_on_fraction(stats, out_dir / "on_fraction.png"),
        plot_spectrum(stats, out_dir / "spectrum.png", f_max_hz),
        plot_intervals(stats, out_dir / "intervals.png"),
        plot_pixel_rate_histogram(stats, out_dir / "pixel_rate_hist.png"),
        plot_pixel_rate_maps(stats, out_dir / "pixel_rate_map.png"),
    ]
    summary = out_dir / "summary.json"
    summary.write_text(json.dumps([s.summary() for s in stats], indent=2, allow_nan=False) + "\n")
    return [*paths, summary]
