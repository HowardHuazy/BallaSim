"""Real vs sim at the viewer level (docs/design/phase3/ui.md §3.3 ``/compare/*``, §3.5 BSMK, §4.5, §4.6; UI-8a).

- **Clock rule** (§4.6). Each stream maps onto the true clock as true = t + a: an EVIMO2 sequence by its δ (t_vicon =
  t_npz + δ, a = δ, from the package's G1 table in ``ballasim.evimo2.clocks``; an untabulated or unidentified row
  leaves δ unavailable), a BallaSim run by its resolved ``sensor.clock_offset_ns`` c (s = t_true + c, a = −c). The
  reference stream has δ_stream = 0 and the test stream is proposed δ_stream = a_test − a_ref, so that both lie on the
  reference's clock: an EVIMO2 recording and its simulation with c = −δ get 0 (the bench's acceptance rule,
  ``pipeline.py:595``), any other c gets −(c + δ) with the warning BS-UI-CLOCK, which names the drift it neglects.
  δ unavailable also warns BS-UI-CLOCK, and the real stream's GT stays off (no ``viewer`` region on it).
- **Regions** (§4.5): ``none`` (all pixels), ``crop`` (a rectangle), ``viewer`` (``viz.stats.GtRegion`` on the
  reference stream's GT, EVIMO2 GT looked up at t + δ, as a ``MaskRegion`` of 1 ms slots from the span's start; a
  crop, if given, is applied first). ``bench`` and ``legacy`` are the bench level (``ui/bench.py``, UI-8b, with
  ``--repo`` only; otherwise 501 BS-UI-BENCH-UNAVAILABLE): the span is an evb span id and the test stream a run's
  camera; the region is the bench's own (its digest is the one ``evb.py compare`` writes), the clock rule uses the
  span's δ, and the answer carries the GT-quality verdict, the dropped bodies and F043's eligible windows. Any region
  asked for with an evb span id carries the windows. A region is masks on intervals of the view clock (the
  reference's), kept in a bounded LRU under its id, the first 16 hex digits of its mask digest, and served as BSMK. As
  the id names the masks only, stream pairs whose masks are the same (a ``none`` region over one span, say) share a
  region; it keeps the δ_stream proposals (and the windows) of every pair it was asked for.
- **Viewer statistics**: ``viz.stats.compute_stats`` per stream over a view-clock span, through the region's
  ``MaskRegion`` (its per-event test is C++, ``_core.wire.region_bits``) at the stream's δ_stream, every stream
  reporting the vector strength at the lights of all of them (D16b, as ``viz stats``). A region that carries F043's
  windows (asked for with an evb span id) adds a second ``compute_stats`` call whose ``MaskRegion`` leaves the
  ineligible 10 ms windows uncovered: the count statistics (events, rates, rate map, IEI) come from it, marked
  ``count_filter: "f043"``; the thinning-invariant ones (ON fraction, vector strength, spectral peaks) from the
  unrestricted call. Other regions are ``unfiltered``.

Python here is I/O glue (no per-event loop and no NumPy pass of its own over events); building a region (its masks,
digest and intervals) and the statistics run on the server's 2-thread pool.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .bench import Bench, Eligible
from .streams import Stream, StreamError, Streams

REGION_MODES = ("viewer", "crop", "none", "bench", "legacy")
BENCH_MODES = ("bench", "legacy")
#: The summary entries that are thinning-invariant (§4.5): with F043's windows they come from the unrestricted call.
THINNING_INVARIANT = ("on_fraction", "vector_strength", "spectral_peaks_hz")
#: BSMK: masks per message (§3.5).
MAX_MASKS_PER_MESSAGE = 256
#: A ``viewer`` region spans at most this many 1 ms slots (≈ 17 min), and one region holds at most this many bytes of
#: masks; the registry keeps at most MAX_REGIONS regions within REGION_BUDGET bytes, least recently used first out.
MAX_REGION_SLOTS = 1 << 20
MAX_REGION_BYTES = 256 << 20
REGION_BUDGET = 512 << 20
MAX_REGIONS = 32
#: Viewer statistics: streams per request, 100 µs spectrum bins per span (≈ 14 min), rate bins per stream, and the
#: events whose intervals are counted (more: ``intervals_truncated``).
MAX_STATS_STREAMS = 4
MAX_SPECTRUM_BINS = 1 << 23
MAX_RATE_BINS = 1 << 16
STATS_INTERVAL_EVENTS = 1 << 21
WIRE_VERSION = 1
BSMK_MAGIC = int.from_bytes(b"BSMK", "little")
#: magic, version, flags, n_masks, width, height, m0, 8 reserved bytes.
BSMK_HEADER = struct.Struct("<IHHIHHq8x")
assert BSMK_HEADER.size == 32
_I64 = (-(1 << 63), (1 << 63) - 1)


def _int64(value, what: str) -> int:
    """An int64 given as a decimal string or an integer (§3.3: times are decimal strings)."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise StreamError(400, "BS-UI-REQUEST", f"{what}: a decimal int64")
    try:
        v = int(value)
    except ValueError:
        raise StreamError(400, "BS-UI-REQUEST", f"{what}: a decimal int64") from None
    if not _I64[0] <= v <= _I64[1]:
        raise StreamError(400, "BS-UI-REQUEST", f"{what}: a decimal int64")
    return v


def _span(value, prefix: str = "span.") -> tuple[int, int]:
    """(t0_ns, t1_ns) of ``{t0_ns, t1_ns}``, t0 < t1."""
    if not isinstance(value, dict):
        raise StreamError(400, "BS-UI-REQUEST", f"{prefix.rstrip('.')}: {{t0_ns, t1_ns}}")
    t0, t1 = _int64(value.get("t0_ns"), f"{prefix}t0_ns"), _int64(value.get("t1_ns"), f"{prefix}t1_ns")
    if t1 <= t0:
        raise StreamError(400, "BS-UI-REQUEST", f"{prefix}t0_ns, {prefix}t1_ns: need t0_ns < t1_ns")
    return t0, t1


def _flag(body: dict, name: str, default: bool) -> bool:
    value = body.get(name, default)
    if not isinstance(value, bool):
        raise StreamError(400, "BS-UI-REQUEST", f"{name}: true or false")
    return value


def _crop(value, width: int, height: int) -> tuple[int, int, int, int]:
    if (not isinstance(value, list) or len(value) != 4
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)):
        raise StreamError(400, "BS-UI-REQUEST", "crop: [x0, y0, x1, y1], integers")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise StreamError(400, "BS-UI-REQUEST", f"crop: need 0 <= x0 < x1 <= {width} and 0 <= y0 < y1 <= {height}")
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------------------------------------------
# Clocks (§4.6)
# ---------------------------------------------------------------------------------------------------------------

def evimo2_delta(directory: Path) -> tuple[int | None, str]:
    """(δ in ns, a note) of an EVIMO2 camera-sequence folder from the package's G1 table (t_vicon = t_npz + δ); δ is
    None when the row is not tabulated or not identified (se_ms null)."""
    from ballasim.evimo2.clocks import g1_delta_ms
    from ballasim.evimo2.dataset import CAMERA_FOLDERS

    parts = Path(directory).resolve().parts
    cams = [i for i, p in enumerate(parts) if p in CAMERA_FOLDERS]
    if not cams or cams[-1] + 1 >= len(parts):
        return None, "not in an EVIMO2 camera folder"
    i = cams[-1]
    key = f"{parts[i]}/{parts[i + 1]}/{Path(directory).name}"
    row = g1_delta_ms(SimpleNamespace(camera=parts[i], category=parts[i + 1], name=Path(directory).name))
    if row is None:
        return None, f"{key} is not in the G1 table"
    delta_ms, se_ms = row
    if se_ms is None:
        return None, f"{key}: the G1 table does not identify its δ"
    return round(delta_ms * 1_000_000), f"G1 table, {key}"


def run_clock(events_path: Path) -> tuple[int | None, float | None, str]:
    """(c = resolved ``sensor.clock_offset_ns``, drift in ppm, a note) of a run's ``events/<camera>.h5``; c is None
    when the file is not in a run or its run.json records no resolved values (format 1)."""
    run_json = events_path.parent.parent / "run.json"
    if events_path.parent.name != "events" or not run_json.is_file():
        return None, None, "not a run's event file"
    try:
        doc = json.loads(run_json.read_text())
    except (OSError, ValueError):
        return None, None, "its run.json cannot be read"
    camera = (((doc.get("resolved") or {}).get("cameras") or {}).get(events_path.stem)) or {}
    offset = (camera.get("sensor.clock_offset_ns") or {}).get("value")
    drift = (camera.get("sensor.clock_drift_ppm") or {}).get("value")
    if not isinstance(offset, int) or isinstance(offset, bool):
        return None, None, "its run.json records no resolved sensor.clock_offset_ns"
    return offset, float(drift) if isinstance(drift, (int, float)) else 0.0, f"run camera {events_path.stem}"


def stream_clock(stream: Stream) -> dict:
    """How a stream maps onto the true clock (true = t + a): ``{kind, a_ns, delta_ns, offset_ns, drift_ppm, note}``
    (a_ns None: unknown)."""
    reader = stream.reader
    if reader.kind == "view" and type(reader.source).__name__ == "Evimo2Sequence":
        delta, note = evimo2_delta(reader.source.directory)
        return {"kind": "evimo2", "a_ns": delta, "delta_ns": delta, "offset_ns": None, "drift_ppm": None,
                "note": note}
    if reader.kind == "c4":
        offset, drift, note = run_clock(stream.path)
        if offset is not None or note != "not a run's event file":
            return {"kind": "run", "a_ns": None if offset is None else -offset, "delta_ns": None,
                    "offset_ns": offset, "drift_ppm": drift, "note": note}
    return {"kind": "file", "a_ns": None, "delta_ns": None, "offset_ns": None, "drift_ppm": None,
            "note": "an event file without clock facts"}


def clock_rule(ref: dict, test: dict, ref_label: str, test_label: str) -> tuple[int, list[dict]]:
    """δ_stream of the test stream on the reference's clock, and the BS-UI-CLOCK warnings (§4.6)."""
    warnings = []
    for c, label in ((ref, ref_label), (test, test_label)):
        if c["kind"] == "evimo2" and c["delta_ns"] is None:
            warnings.append({"code": "BS-UI-CLOCK", "message": f"{label}: δ unavailable ({c['note']}): its GT "
                             "overlays and timing panels are disabled"})
        elif c["kind"] == "run" and c["offset_ns"] is None:
            warnings.append({"code": "BS-UI-CLOCK", "message": f"{label}: no clock offset ({c['note']})"})
    if ref["a_ns"] is None or test["a_ns"] is None:
        if warnings:
            warnings.append({"code": "BS-UI-CLOCK", "message": f"{test_label} is placed at δ_stream = 0: its "
                             "clock relative to the reference is unknown"})
        return 0, warnings
    proposal = test["a_ns"] - ref["a_ns"]
    if proposal:
        drifts = [f"{label} {c['drift_ppm']:g} ppm" for c, label in ((ref, ref_label), (test, test_label))
                  if c["kind"] == "run"]
        how = {("evimo2", "run"): "the run's clock offset c is not −δ: proposed δ_stream = −(c + δ)",
               ("run", "evimo2"): "the run's clock offset c is not −δ: proposed δ_stream = c + δ",
               ("run", "run"): "the runs' clock offsets differ: proposed δ_stream = c_ref − c_test",
               ("evimo2", "evimo2"): "the sequences' δ differ: proposed δ_stream = δ_test − δ_ref"}
        warnings.append({"code": "BS-UI-CLOCK", "message": f"{how[(ref['kind'], test['kind'])]} = {proposal} ns; "
                         f"drift neglected ({', '.join(drifts) if drifts else 'none recorded'})"})
    return proposal, warnings


def span_clock(clock: dict, sp: dict) -> dict:
    """An EVIMO2 stream's clock with the δ of an evb span (``span_info``: δ_bench for ``delta: bench`` spans, else G1's
    unless it was rejected for a first frame id other than 0 that was not re-measured; None when the span has none),
    which the bench level uses instead of the package's G1 table (§4.6)."""
    delta = None if sp["delta_ms"] is None else sp["delta_ns"]
    note = f"span {sp['id']}: {sp['delta_source']} δ" if delta is not None else \
        f"span {sp['id']}: {sp['note'] or 'the span has no δ'}"
    return {**clock, "a_ns": delta, "delta_ns": delta, "note": note}


# ---------------------------------------------------------------------------------------------------------------
# Regions (§4.5) and BSMK (§3.5)
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Region:
    """A region on the view clock (the reference's): its MaskRegion (offset 0), the clock rule's δ_stream proposals of
    the stream pairs it was asked for, ``{ref sid: {test sid: δ_stream}}``, and F043's eligible windows of the
    reference streams it was asked for with an evb span id, ``{ref sid: Eligible}`` (pairs with the same masks share a
    region). ``counted`` keeps, per reference, the MaskRegion that leaves the ineligible windows uncovered."""

    region_id: str
    masks: object                     # viz.stats.MaskRegion
    proposals: dict[str, dict[str, int]]
    eligible: dict[str, Eligible] = field(default_factory=dict)
    counted: dict[str, object] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        # an eligible region's counted copy cuts every mask at the ineligible windows: budget it as one more set
        return int(self.masks.words.nbytes * (2 if self.eligible else 1) + self.masks.t_lo.nbytes * 3)

    def counted_for(self, ref_sid: str):
        """The region's masks with the ineligible windows of ``ref_sid``'s F043 uncovered (None without windows)."""
        elig = self.eligible.get(ref_sid)
        if elig is None:
            return None
        cut = self.counted.get(ref_sid)
        if cut is None:
            from ballasim.viz.stats import MaskRegion

            m = self.masks
            cut = MaskRegion(np.stack([m.t_lo, m.t_hi], 1), m.bits, 0, elig.covered, width=m.width, height=m.height,
                             label=m.label)
            self.counted[ref_sid] = cut
        return cut


def mask_digest(width: int, height: int, t_lo: np.ndarray, t_hi: np.ndarray, bits: np.ndarray) -> str:
    """SHA-256 (hex) of a region's BSMK content: size, intervals and words (hashed in place, without a copy)."""
    h = hashlib.sha256(struct.pack("<4sHH", b"BSMK", width, height))
    for a in (np.ascontiguousarray(t_lo, "<i8"), np.ascontiguousarray(t_hi, "<i8"), np.ascontiguousarray(bits, "<u4")):
        h.update(a.reshape(-1).view(np.uint8))         # a view of the contiguous array
    return h.hexdigest()


def bsmk_message(width: int, height: int, m0: int, t_lo: np.ndarray, t_hi: np.ndarray, words: np.ndarray) -> bytes:
    """One BSMK message (§3.5): masks [m0, m0 + n), their [t_lo, t_hi) interleaved (i64), then their words (u32)."""
    n = len(t_lo)
    intervals = np.stack([np.asarray(t_lo, "<i8"), np.asarray(t_hi, "<i8")], 1).reshape(-1)
    return BSMK_HEADER.pack(BSMK_MAGIC, WIRE_VERSION, 0, n, width, height, m0) + \
        np.ascontiguousarray(intervals, "<i8").tobytes() + np.ascontiguousarray(words, "<u4").tobytes()


class Compare:
    """``/compare/region``, ``/compare/region/{id}/masks`` and ``/compare/stats`` of one server."""

    def __init__(self, streams: Streams, pool, repo: Path | None):
        self.streams, self.pool, self.repo = streams, pool, repo
        self.bench = None if repo is None else Bench(repo)
        self.regions: collections.OrderedDict[str, Region] = collections.OrderedDict()
        self.bytes = 0

    async def _run(self, fn, *args):
        try:
            return await asyncio.get_running_loop().run_in_executor(self.pool, fn, *args)
        except StreamError:
            raise
        except ImportError as e:                            # ballasim.viz needs the viz extra (cv2, matplotlib)
            raise StreamError(422, "BS-UI-SOURCE", f"regions and statistics need ballasim[viz]: {e}") from None
        except PermissionError:
            raise StreamError(403, "BS-UI-PATH", "this user may not read the stream's source") from None
        except FileNotFoundError:
            raise StreamError(404, "BS-UI-NOT-FOUND", "the stream's source no longer exists") from None
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise StreamError(422, "BS-UI-SOURCE", str(e) or type(e).__name__) from None

    # -- regions -------------------------------------------------------------------------------------------------

    def get(self, region_id: str) -> Region:
        region = self.regions.get(region_id)
        if region is None:
            raise StreamError(404, "BS-UI-NOT-FOUND", f"no region {region_id}")
        self.regions.move_to_end(region_id)
        return region

    async def region(self, body: dict) -> dict:
        """``POST /compare/region``: the region of ``mode`` over a span of the view clock, with the clock rule. An evb
        span id (``span``, any mode; ``bench`` and ``legacy`` need one) is the bench level (``ui/bench.py``)."""
        mode = body.get("mode", "viewer")
        if mode not in REGION_MODES:
            raise StreamError(400, "BS-UI-REQUEST", f"mode: one of {', '.join(REGION_MODES)}")
        evb_id = body["span"] if isinstance(body.get("span"), str) else None
        if mode in BENCH_MODES or evb_id is not None:
            if self.bench is None:
                raise StreamError(501, "BS-UI-BENCH-UNAVAILABLE", "the bench level (bench and legacy regions, evb "
                                                                  "span ids) needs a source checkout (start the "
                                                                  "server with --repo)")
            if evb_id is None:
                raise StreamError(400, "BS-UI-REQUEST", f"span: mode {mode} needs an evb span id")
        for name in ("ref", "test"):
            if not isinstance(body.get(name), str):
                raise StreamError(400, "BS-UI-REQUEST", f"{name}: a stream id")
        ref, test = self.streams.get(body["ref"]), self.streams.get(body["test"])
        rr, tr = ref.reader, test.reader
        if (rr.width, rr.height) != (tr.width, tr.height):
            raise StreamError(422, "BS-UI-SOURCE", f"the streams' resolutions differ ({rr.width}x{rr.height} and "
                                                   f"{tr.width}x{tr.height}); comparison panels need one")
        hot = _flag(body, "hot", True)
        allow_bad = _flag(body, "allow_bad", False)
        t0 = t1 = None
        if evb_id is not None:
            pass                                    # the span's extent: below, from its definition
        elif "span" in body:
            t0, t1 = _span(body["span"])
        else:
            if not rr.n:
                raise StreamError(422, "BS-UI-SOURCE", "the reference stream holds no events: give a span")
            t0, t1 = rr.t_start_ns, rr.t_end_ns
        crop = None
        if body.get("crop") is not None:
            if mode == "none" or mode in BENCH_MODES:
                raise StreamError(400, "BS-UI-REQUEST", f"crop: not with mode {mode}"
                                  + (" (use mode crop)" if mode == "none" else ""))
            crop = _crop(body["crop"], rr.width, rr.height)
        elif mode == "crop":
            raise StreamError(400, "BS-UI-REQUEST", "crop: mode crop needs [x0, y0, x1, y1]")
        ref_clock, test_clock = await self._run(lambda: (stream_clock(ref), stream_clock(test)))
        sp = None
        if evb_id is not None:
            if ref_clock["kind"] != "evimo2":
                raise StreamError(422, "BS-UI-SOURCE", f"an evb span id needs the EVIMO2 sequence as the reference "
                                                       f"stream ({ref.label} is not one)")
            sp = await self._run(self.bench.span, evb_id, ref.reader.source.directory, allow_bad)
            ref_clock = span_clock(ref_clock, sp)           # §4.6: with --repo, δ is span_info's
        proposal, warnings = clock_rule(ref_clock, test_clock, ref.label, test.label)
        evimo2 = ref_clock if ref_clock["kind"] == "evimo2" else test_clock if test_clock["kind"] == "evimo2" else None
        delta = None if evimo2 is None else {
            "ns": None if evimo2["delta_ns"] is None else str(evimo2["delta_ns"]),
            "source": "G1 table" if evimo2["delta_ns"] is not None else "unavailable", "note": evimo2["note"]}
        eligible = br = None
        if sp is not None:
            delta = self.bench.delta(sp)
        if mode in BENCH_MODES:
            if test.reader.kind != "c4" or test.path.parent.name != "events":
                raise StreamError(422, "BS-UI-SOURCE", f"{mode} regions compare a run's camera ({test.label} is not "
                                                       "an events/<camera>.h5 of a run)")
            if mode == "bench" and sp["delta_ms"] is None:
                warnings.append({"code": "BS-UI-CLOCK", "message": f"span {evb_id} has no δ: the region is placed "
                                                                   "with δ = 0, as the bench does (untimed)"})
            masks, digest, intervals, br = await self._run(self._build_bench, sp, mode, test.path.parent.parent,
                                                           test.path, hot)
            (t0, t1), eligible = br.span, br.eligible
        else:
            if sp is not None:
                t0, t1 = self.bench.event_span(sp)
            masks, digest, intervals = await self._run(self._build, ref, ref_clock, mode, t0, t1, crop)
            if sp is not None:
                eligible = await self._run(self.bench.eligible, sp, t0, t1)
        region_id = digest[:16]
        clock_json = {k: {"kind": c["kind"], "delta_ns": None if c["delta_ns"] is None else str(c["delta_ns"]),
                          "offset_ns": None if c["offset_ns"] is None else str(c["offset_ns"]),
                          "drift_ppm": c["drift_ppm"], "note": c["note"]}
                      for k, c in (("ref", ref_clock), ("test", test_clock))}
        answer = {"region_id": region_id, "mode": mode, "span": {"t0_ns": str(t0), "t1_ns": str(t1)},
                  "width": masks.width, "height": masks.height, "n_masks": len(masks.t_lo), "intervals": intervals,
                  "mask_digest": digest, "label": masks.label, "crop": None if crop is None else list(crop),
                  "delta": delta, "clock": clock_json, "offsets": {"ref": "0", "test": str(proposal)},
                  "warnings": warnings, "gt_quality": None if sp is None else sp["gt_quality"],
                  "dropped_ids": list(sp["drop_ids"]) if mode == "bench" else [],
                  "hot_removed": mode == "bench" and hot}
        if eligible is not None:
            answer["eligible"] = eligible.json()
        self._insert(Region(region_id, masks, {ref.sid: {ref.sid: 0, test.sid: proposal}},
                            {} if eligible is None else {ref.sid: eligible}))
        return answer

    def _build_bench(self, sp: dict, mode: str, run_dir: Path, events_path: Path, hot: bool):
        """The bench level's MaskRegion (on the real stream's clock), its digest (the bench's own) and its intervals as
        JSON, with the bench's region (runs on the pool)."""
        from ballasim.viz.stats import MaskRegion

        info = self.bench.run_info(run_dir, events_path)
        br = self.bench.region(sp, mode, info, hot=hot)
        masks = MaskRegion(br.t_edges, br.masks, label=br.label)
        intervals = [{"t0_ns": str(a), "t1_ns": str(b)} for a, b in zip(masks.t_lo.tolist(), masks.t_hi.tolist(),
                                                                        strict=True)]
        return masks, br.digest, intervals, br

    @classmethod
    def _build(cls, ref: Stream, ref_clock: dict, mode: str, t0: int, t1: int, crop):
        """The region's MaskRegion, its mask digest and its intervals as JSON (runs on the pool)."""
        masks = cls._masks(ref, ref_clock, mode, t0, t1, crop)
        digest = mask_digest(masks.width, masks.height, masks.t_lo, masks.t_hi, masks.bits)
        intervals = [{"t0_ns": str(a), "t1_ns": str(b)} for a, b in zip(masks.t_lo.tolist(), masks.t_hi.tolist(),
                                                                        strict=True)]
        return masks, digest, intervals

    @staticmethod
    def _masks(ref: Stream, ref_clock: dict, mode: str, t0: int, t1: int, crop):
        """The region's MaskRegion (on the reference's clock; runs on the pool)."""
        from ballasim.viz.stats import GtRegion, MaskRegion, mask_words, pack_masks

        w, h = ref.reader.width, ref.reader.height
        full = np.ones((h, w), bool)
        if crop is not None:
            x0, y0, x1, y1 = crop
            full[:] = False
            full[y0:y1, x0:x1] = True
        if mode != "viewer":
            return MaskRegion([[t0, t1]], full[None], label="crop" if mode == "crop" else "all pixels")
        if ref_clock["kind"] == "evimo2" and ref_clock["delta_ns"] is None:
            raise StreamError(422, "BS-UI-GT-MISSING", f"BS-UI-CLOCK: δ is unavailable for {ref.label} "
                                                       f"({ref_clock['note']}), so its GT is disabled; use mode none "
                                                       f"or crop")
        max_masks = MAX_REGION_BYTES // (4 * mask_words(w, h))
        with ref.viz_source() as source:
            if "mask" not in source.gt_kinds:
                raise StreamError(422, "BS-UI-GT-MISSING", f"{ref.label} has no GT masks (an EVIMO2 sequence, or a "
                                                           "run with dense frames); use mode none or crop")
            if ref_clock["kind"] == "evimo2":
                source = source.with_gt_offset(-ref_clock["delta_ns"])          # GT at t + δ (§4.6)
            try:
                region = GtRegion(source).mask_region(t0, t1, max_slots=MAX_REGION_SLOTS, max_masks=max_masks)
            except ValueError as e:
                raise StreamError(400, "BS-UI-REQUEST", f"span: {e}; shorten it") from None
        if crop is not None:                # in place, so a build holds its masks once
            np.bitwise_and(region.words, pack_masks(full[None])[0], out=region.words)
        return region

    def _insert(self, region: Region) -> None:
        """Keeps ``region`` as the most recently used. One with the same id (the same masks, asked for by another
        pair or again) keeps the proposals of its pairs, this call's winning."""
        old = self.regions.pop(region.region_id, None)
        if old is not None:
            self.bytes -= old.nbytes
            for ref, tests in old.proposals.items():
                region.proposals[ref] = {**tests, **region.proposals.get(ref, {})}
            region.eligible = {**old.eligible, **region.eligible}
        self.regions[region.region_id] = region
        self.bytes += region.nbytes
        while len(self.regions) > 1 and (self.bytes > REGION_BUDGET or len(self.regions) > MAX_REGIONS):
            _, dropped = self.regions.popitem(last=False)
            self.bytes -= dropped.nbytes

    def masks_message(self, region_id: str, m0: int | None, m1: int | None) -> bytes:
        """``GET /compare/region/{id}/masks?m0&m1``: BSMK of masks [m0, m1), at most 256."""
        r = self.get(region_id).masks
        n = len(r.t_lo)
        m0 = 0 if m0 is None else m0
        m1 = min(n, m0 + MAX_MASKS_PER_MESSAGE) if m1 is None else m1
        if not (0 <= m0 <= m1 <= n and m1 - m0 <= MAX_MASKS_PER_MESSAGE):
            raise StreamError(416, "BS-UI-SLICE", f"need 0 <= m0 <= m1 <= {n} and m1 - m0 <= {MAX_MASKS_PER_MESSAGE}")
        return bsmk_message(r.width, r.height, m0, r.t_lo[m0:m1], r.t_hi[m0:m1], r.bits[m0:m1])

    # -- statistics ----------------------------------------------------------------------------------------------

    async def stats(self, body: dict) -> dict:
        """``POST /compare/stats``: viewer statistics per stream over a view-clock span (§4.5)."""
        sids = body.get("streams")
        if (not isinstance(sids, list) or not 1 <= len(sids) <= MAX_STATS_STREAMS
                or not all(isinstance(s, str) for s in sids)):
            raise StreamError(400, "BS-UI-REQUEST", f"streams: 1 to {MAX_STATS_STREAMS} stream ids")
        streams = [self.streams.get(s) for s in sids]
        t0, t1 = _span(body, "")
        options = body.get("options") or {}
        if not isinstance(options, dict):
            raise StreamError(400, "BS-UI-REQUEST", "options: {bin_ms, segment_s}")
        bin_ms, segment_s = options.get("bin_ms", 1.0), options.get("segment_s", 1.0)
        for name, v, lo, hi in (("bin_ms", bin_ms, 0.1, 10_000.0), ("segment_s", segment_s, 0.001, 3600.0)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
                raise StreamError(400, "BS-UI-REQUEST", f"options.{name}: a number in [{lo:g}, {hi:g}]")
        spectrum_bins = (t1 - t0) // 100_000
        if spectrum_bins < 8 or spectrum_bins > MAX_SPECTRUM_BINS:
            raise StreamError(400, "BS-UI-REQUEST", f"the span holds {spectrum_bins} bins of 100 µs; statistics take "
                                                    f"8 to {MAX_SPECTRUM_BINS}")
        rate_bin_ns = max(1, round(bin_ms * 10)) * 100_000
        if (t1 - t0) // rate_bin_ns > MAX_RATE_BINS:
            raise StreamError(400, "BS-UI-REQUEST", f"more than {MAX_RATE_BINS} rate bins: raise options.bin_ms")
        region = None
        if body.get("region_id") is not None:
            if not isinstance(body["region_id"], str):
                raise StreamError(400, "BS-UI-REQUEST", "region_id: a region id")
            region = self.get(body["region_id"])
        # δ_stream per stream, aligned with `streams` (a stream may be listed twice, at two offsets); null or absent:
        # the clock rule's proposal against streams[0] from a /compare/region of this region whose reference was
        # streams[0] (streams[0] itself: 0), else 0
        given = body.get("offsets")
        if given is None:
            given = [None] * len(streams)
        if not isinstance(given, list) or len(given) != len(streams):
            raise StreamError(400, "BS-UI-REQUEST", "offsets: one δ_stream (ns, a decimal string) or null per stream")
        proposed = {} if region is None else region.proposals.get(streams[0].sid, {})
        offsets = [(0 if i == 0 else proposed.get(s.sid, 0)) if v is None else _int64(v, f"offsets[{i}]")
                   for i, (s, v) in enumerate(zip(streams, given, strict=True))]
        from ballasim.viz.stats import StatsOptions

        opts = StatsOptions(rate_bin_ms=float(bin_ms), spectrum_segment_s=float(segment_s),
                            max_interval_events=STATS_INTERVAL_EVENTS)
        masks = None if region is None else region.masks
        # F043: the windows of a region asked for with streams[0] as its reference leave the counts' MaskRegion
        counted = None if region is None else await self._run(region.counted_for, streams[0].sid)
        lights = await self._run(self._lights, streams)
        results = await asyncio.gather(*(self._run(self._one, s, d, t0, t1, opts, masks, counted, lights)
                                         for s, d in zip(streams, offsets, strict=True)))
        return {"span": {"t0_ns": str(t0), "t1_ns": str(t1)}, "region_id": None if region is None else
                region.region_id, "stats": results}

    @staticmethod
    def _lights(streams: list[Stream]) -> dict[float, str]:
        """The lights of all the streams, {Hz: name} in stream order (D16b: every stream reports the vector strength
        at them, so the columns compare; as ``viz stats``). Runs on the pool."""
        from ballasim.viz.stats import light_frequencies

        lights: dict[float, str] = {}
        for stream in streams:
            if stream.closed:
                raise StreamError(404, "BS-UI-NOT-FOUND", f"stream {stream.sid} was closed")
            with stream.viz_source() as source:
                for f, name in light_frequencies(source).items():
                    lights.setdefault(f, name)
        return lights

    @staticmethod
    def _one(stream: Stream, offset: int, t0: int, t1: int, opts, masks, counted, lights: dict[float, str]) -> dict:
        """One stream's statistics of the view-clock span [t0, t1) at δ_stream ``offset`` (runs on the pool). With
        ``counted`` (the region's masks without the ineligible F043 windows), the count statistics come from a second
        call over it and the thinning-invariant ones from the first (§4.5)."""
        from ballasim.viz.stats import compute_stats, interval_edges_s

        if stream.closed:
            raise StreamError(404, "BS-UI-NOT-FOUND", f"stream {stream.sid} was closed")
        region = None if masks is None else masks.with_offset(offset)
        with stream.viz_source() as source:
            st = compute_stats(source, t0 - offset, t1 - offset, opts, region, lights)
            ct = st if counted is None else compute_stats(source, t0 - offset, t1 - offset, opts,
                                                          counted.with_offset(offset), lights)
        summary = ct.summary()
        if ct is not st:
            whole = st.summary()
            summary.update({k: whole[k] for k in THINNING_INVARIANT})
        rc = ct.rate_counts()
        covered = ct.covered(ct._rate_factor())
        edges = interval_edges_s()
        iei = {"edges_s": edges.tolist()}
        for name, (counts, zeros, n) in ct.intervals.histograms(edges).items():
            iei[name], iei[f"{name}_zero"], iei[f"{name}_n"] = counts.tolist(), zeros, n
        rate_map = np.ascontiguousarray(ct.pixel_rate_hz, "<f8")
        return {"sid": stream.sid, "label": stream.label, "offset_ns": str(offset),
                "lights": {f"{f:g}": name for f, name in ct.lights.items()}, "summary": summary,
                "rate": {"t0_ns": str(rc.t0_ns + offset), "bin_ns": rc.bin_ns, "total": rc.total.tolist(),
                         "on": rc.on.tolist(), "covered": covered.tolist()},
                "iei": iei, "rate_map_digest": hashlib.sha256(rate_map.tobytes()).hexdigest()[:16],
                "count_filter": "unfiltered" if counted is None else "f043"}


__all__ = ["BSMK_HEADER", "MAX_MASKS_PER_MESSAGE", "REGION_MODES", "Compare", "bsmk_message", "clock_rule",
           "evimo2_delta", "mask_digest", "run_clock", "span_clock", "stream_clock"]
