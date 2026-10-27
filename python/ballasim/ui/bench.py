"""The bench adapter (docs/design/phase3/ui.md §3.11 ``ui/bench.py``, §4.5 ``bench`` mode rules, §3.7 ``bench`` job;
UI-8b): the EVIMO2 bench region, the span's δ and GT-quality verdict and the F043 count-eligible windows, computed by
``tools/evimo2_bench``'s own functions. Active only with ``--repo`` (a source checkout; the wheel does not ship the
bench).

- **Loading.** ``<repo>/tools/evimo2_bench`` is loaded lazily as the package ``evimo2_bench`` from its location (the
  bench modules import each other relatively and import no ``ballasim`` module, so their import guard holds). A package
  of that name already loaded from another checkout is refused. ``SystemExit`` from ``span_info`` is caught and mapped
  (unknown span: 422 BS-UI-SPAN; a ``bad`` span without ``allow_bad``: 422 BS-UI-GT-BAD).
- **Spans** (:meth:`Bench.span`). ``evb.span_info(evb.load_spans(), id)`` with the root of the spans file following the
  reference stream: when the stream's directory ends with the span's sequence path, the part before it is the root
  (the EVIMO2 data may live elsewhere than ``spans.yaml`` or ``$EVIMO2_ROOT`` say); the span must be that stream's
  sequence (422 BS-UI-SOURCE otherwise).
- **Regions** (:meth:`Bench.region`), as ``pipeline.compare`` builds them (``pipeline.py:528-542``): the run's event
  span (a, b) = ``RunInfo.load(run_dir).span_event_ns``; ``real_side`` for the real stream's hot pixels and the
  eligible windows (no cache is written); the sim's hot pixels from its quantized events; ``bench_region(...)
  .without(hot)``. ``legacy`` is ``legacy_region`` on the run's true span, one static mask over (a, b), no hot pixels.
  The mask digest is the bench's own (``BenchRegion.digest()``, what ``evb.py compare`` writes as ``region.digest``).
- **F043** (:class:`Eligible`): the count-eligible 10 ms windows from the span's start, as ``pipeline.eligible_windows``
  with ``span_phase`` give them (``real_side`` returns the same for a ``bench`` region).

Python here is I/O glue: every number comes from the bench's code. Calls block (they read the span's events and GT
frames), so the server runs them on its pool.
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import math
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .streams import StreamError

#: One region holds at most this many bytes of unpacked (bool) masks: the bench builds them all at once.
MAX_BENCH_MASK_BYTES = 256 << 20
#: One request reads at most this many events of a stream (the window and its quantized copy: about 1.4 GB at 2**26),
#: the real sequence's and the run's alike; the server's pool runs two requests at once.
MAX_BENCH_EVENTS = 1 << 26
#: An ad-hoc span ends within a day (keeps ``round(t * 1e9)`` far inside int64).
MAX_ADHOC_S = 86_400.0
WINDOW_NS = 10_000_000
#: What ``streams.Evimo2`` opens first: a directory without them is no sequence.
SEQUENCE_FILES = ("dataset_info.npz", "dataset_events_t.npy", "dataset_events_xy.npy", "dataset_events_p.npy")
AD_HOC = "<camera>/<category>/<split>/<sequence>@<t0_s>-<t1_s>"


@dataclass
class Eligible:
    """F043's count-eligible windows: window k is ``[t0_ns + k window_ns, t0_ns + (k + 1) window_ns)`` of the real
    stream's clock (a partial last window is ineligible)."""

    t0_ns: int
    window_ns: int
    flags: np.ndarray

    def json(self) -> dict:
        """``{t0_ns, window_ns, windows, fraction, bits_b64}``: the flags packed little-endian (bit k is byte k // 8's
        bit k % 8)."""
        return {"t0_ns": str(self.t0_ns), "window_ns": str(self.window_ns), "windows": len(self.flags),
                "fraction": float(self.flags.mean()) if len(self.flags) else None,
                "bits_b64": base64.b64encode(np.packbits(self.flags, bitorder="little").tobytes()).decode("ascii")}

    @property
    def covered(self) -> tuple[int, int, np.ndarray]:
        """The ``covered`` argument of ``viz.stats.MaskRegion``."""
        return self.t0_ns, self.window_ns, self.flags


@dataclass
class BenchRegion:
    """What :meth:`Bench.region` computed: the masks and intervals on the event clock, the bench's digest, the run's
    event span (a, b) and the F043 windows."""

    t_edges: np.ndarray                 # (M, 2) int64 [t_lo, t_hi) on the real stream's (event) clock, within [a, b)
    masks: np.ndarray                   # (M, H, W) bool
    digest: str
    span: tuple[int, int]
    eligible: Eligible
    label: str


def _load(repo: Path) -> SimpleNamespace:
    """``tools/evimo2_bench`` of ``repo`` as the package ``evimo2_bench`` and the modules the adapter calls."""
    pkg = repo / "tools" / "evimo2_bench"
    loaded = sys.modules.get("evimo2_bench")
    if loaded is None:
        spec = importlib.util.spec_from_file_location("evimo2_bench", pkg / "__init__.py",
                                                      submodule_search_locations=[str(pkg)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["evimo2_bench"] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop("evimo2_bench", None)
            raise
    elif Path(loaded.__file__).resolve().parent != pkg.resolve():
        raise StreamError(501, "BS-UI-BENCH-UNAVAILABLE", f"the bench of another checkout "
                                                          f"({Path(loaded.__file__).resolve().parent}) is already "
                                                          f"loaded in this server")
    names = ("evb", "pipeline", "region", "metrics", "streams", "quality")
    try:
        return SimpleNamespace(**{n: importlib.import_module(f"evimo2_bench.{n}") for n in names})
    except ImportError as e:
        raise StreamError(501, "BS-UI-BENCH-UNAVAILABLE", f"the bench cannot be imported: {e}") from None


def check_ad_hoc(span_id: str) -> None:
    """422 BS-UI-SPAN unless ``span_id`` (not a key of spans.yaml) is ``<camera>/<category>/<split>/<sequence>@<t0_s>-
    <t1_s>``: four plain relative names, one ``@``, finite times with 0 <= t0 < t1 <= MAX_ADHOC_S. What ``span_info``
    would otherwise fail on with an IndexError or an OverflowError."""
    seq, at, rng = span_id.partition("@")
    if not at or "@" in rng:
        raise StreamError(422, "BS-UI-SPAN", f"unknown span {span_id!r}: a key of spans.yaml or {AD_HOC}")
    names = seq.split("/")
    if len(names) != 4 or not span_id.isprintable() or any(n in ("", ".", "..") for n in names):
        raise StreamError(422, "BS-UI-SPAN", f"span {span_id!r}: the sequence must be four plain names "
                                             "(<camera>/<category>/<split>/<sequence>), relative, without '..'")
    try:
        t0, t1 = (float(v) for v in rng.split("-"))
    except ValueError:
        raise StreamError(422, "BS-UI-SPAN", f"span {span_id!r}: the times are <t0_s>-<t1_s>, two plain "
                                             "non-negative numbers in seconds") from None
    if not (math.isfinite(t0) and math.isfinite(t1) and 0.0 <= t0 < t1 <= MAX_ADHOC_S):
        raise StreamError(422, "BS-UI-SPAN", f"span {span_id!r}: the times need 0 <= t0 < t1 <= {MAX_ADHOC_S:g} s")


def check_events(stream, a: int, b: int, what: str) -> None:
    """400 BS-UI-REQUEST when the event window [a, b) of a bench stream (``Evimo2`` or ``H5Run``) holds more than
    MAX_BENCH_EVENTS (the adapter reads a window whole, on the server's pool)."""
    n = stream.search(b) - stream.search(a)
    if n > MAX_BENCH_EVENTS:
        raise StreamError(400, "BS-UI-REQUEST", f"{what} holds {n} events in [{a}, {b}) ns: more than "
                                                f"{MAX_BENCH_EVENTS}; use a shorter span")


class Bench:
    """The bench of one source checkout (``--repo``)."""

    def __init__(self, repo: Path):
        self.repo = Path(repo)
        self._lock = threading.Lock()
        self._mods: SimpleNamespace | None = None

    @property
    def mods(self) -> SimpleNamespace:
        with self._lock:
            if self._mods is None:
                self._mods = _load(self.repo)
            return self._mods

    # -- spans ---------------------------------------------------------------------------------------------------

    def span(self, span_id: str, ref_dir: Path | None = None, allow_bad: bool = False) -> dict:
        """``evb.span_info`` of ``span_id`` (an id of ``spans.yaml`` or ``<camera>/<category>/<split>/<sequence>@<t0_s>-
        <t1_s>``): the span's directory, true-clock span, δ, GT verdict and ``drop_ids``. ``ref_dir``: the reference
        stream's sequence directory, which must be the span's (the spans file's root follows it)."""
        m = self.mods
        spans = m.evb.load_spans()
        if span_id not in spans["spans"]:
            check_ad_hoc(span_id)                   # span_info opens the sequence's dataset_info.npz (a pickle)
        if ref_dir is not None:
            if span_id in spans["spans"]:
                rel = spans["spans"][span_id].get("sequence")
            else:
                rel = span_id.split("@")[0]
            parts = rel.split("/") if isinstance(rel, str) else []
            ref = Path(ref_dir).absolute()
            if parts and tuple(ref.parts[-len(parts):]) == tuple(parts):
                spans["root"] = str(Path(*ref.parts[:-len(parts)]))
        try:
            sp = m.evb.span_info(spans, span_id)
        except SystemExit as e:
            raise StreamError(422, "BS-UI-SPAN", str(e.code) if isinstance(e.code, str) else
                              f"unknown span {span_id!r}") from None
        except ValueError as e:
            if "catalogue" in str(e):
                raise StreamError(422, "BS-UI-SOURCE", str(e)) from None
            raise StreamError(422, "BS-UI-SPAN", f"span {span_id!r}: {e}") from None
        except (OSError, KeyError, TypeError, IndexError, ArithmeticError) as e:
            raise StreamError(422, "BS-UI-SPAN", f"span {span_id!r}: {e}") from None
        if sp["t0_ns"] >= sp["t1_ns"]:
            raise StreamError(422, "BS-UI-SPAN", f"span {span_id!r}: it ends ({sp['t1_ns']} ns) before it begins "
                                                 f"({sp['t0_ns']} ns)")
        bad = sp["gt_quality"]["bad"]
        if bad and not allow_bad:
            raise StreamError(422, "BS-UI-GT-BAD", f"{span_id}: EVIMO2 GT quality bad ({'; '.join(bad)}); allow it to "
                                                   "compute the region without scoring")
        if ref_dir is not None:
            if Path(sp["dir"]).resolve() != Path(ref_dir).resolve():
                raise StreamError(422, "BS-UI-SOURCE", f"the reference stream ({ref_dir}) is not the sequence of span "
                                                       f"{span_id} ({sp['dir']})")
        elif not all((Path(sp["dir"]) / f).is_file() for f in SEQUENCE_FILES):     # the job reads it later
            raise StreamError(422, "BS-UI-SOURCE", f"span {span_id}: no EVIMO2 sequence at {sp['dir']} (the spans "
                                                   f"file's root is {spans['root']}; $EVIMO2_ROOT moves it)")
        return sp

    def event_span(self, sp: dict) -> tuple[int, int]:
        """The span's aligned true-clock extent on the EVIMO2 event clock, t − δ (``evb.event_span``)."""
        return self.mods.evb.event_span(sp)

    @staticmethod
    def delta(sp: dict) -> dict:
        """The response's ``delta``: the span's δ, its source (``G1``, ``bench`` or ``none``) and note."""
        return {"ns": None if sp["delta_ms"] is None else str(sp["delta_ns"]), "source": sp["delta_source"],
                "note": sp["note"]}

    # -- eligibility ---------------------------------------------------------------------------------------------

    def eligible(self, sp: dict, a: int, b: int) -> Eligible:
        """F043's windows over the event-clock span [a, b) of the span's sequence, as ``real_side`` computes them
        (the quantized real stream, ``span_phase``)."""
        m = self.mods
        seq = m.streams.Evimo2(sp["dir"])
        try:
            check_events(seq, a, b, f"the sequence of span {sp['id']}")
            st = seq.window(a, b, m.streams.DEFAULT_CHUNK).quantized()
            phase, _ = m.pipeline.span_phase(seq, a, b)
            return Eligible(a, WINDOW_NS, np.asarray(m.pipeline.eligible_windows(st.t, a, b, phase), bool))
        finally:
            _close_sequence(seq)

    # -- regions -------------------------------------------------------------------------------------------------

    def run_info(self, run_dir: Path, events_path: Path | None = None):
        """``streams.RunInfo.load(run_dir)``; 422 BS-UI-SOURCE when the directory is not a run the bench reads (its
        run.json needs ``cameras``, ``span_ns`` and the resolved ``sensor.clock_offset_ns``), or when its first camera's
        events (the file the bench compares) are not ``events_path`` (default: ``<run_dir>/events/<camera>.h5``) or
        do not exist: a run directory that was moved or copied still names the old path, and a run that crashed has
        no file."""
        m = self.mods
        try:
            info = m.streams.RunInfo.load(run_dir)
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            raise StreamError(422, "BS-UI-SOURCE", f"{run_dir}: not a run the bench reads ({type(e).__name__}: "
                                                   f"{e})") from None
        want = Path(run_dir) / "events" / f"{info.camera}.h5" if events_path is None else Path(events_path)
        if Path(info.events_path).resolve() != want.resolve():
            raise StreamError(422, "BS-UI-SOURCE", f"the bench compares the run's first camera ({info.camera}: "
                                                   f"{info.events_path}), not {want}")
        if not want.is_file():
            raise StreamError(422, "BS-UI-SOURCE", f"{run_dir}: the events file {want} does not exist (a run that "
                                                   "has not finished writing?)")
        return info

    def region(self, sp: dict, mode: str, info, *, hot: bool = True) -> BenchRegion:
        """The ``bench`` or ``legacy`` region of the span against the run (``pipeline.compare``'s steps). Reads the
        span's events and GT frames; 422 BS-UI-GT-MISSING when the span holds no GT interval, 400 BS-UI-REQUEST beyond
        MAX_BENCH_MASK_BYTES of masks."""
        m = self.mods
        a, b = info.span_event_ns
        seq = m.streams.Evimo2(sp["dir"])
        try:
            if mode == "legacy":
                return self._legacy(m, sp, seq, info, a, b)
            return self._bench(m, sp, seq, info, a, b, hot)
        finally:
            _close_sequence(seq)

    @staticmethod
    def _bench(m, sp, seq, info, a: int, b: int, hot: bool) -> BenchRegion:
        t = seq.gt_t
        delta = sp["delta_ns"]
        lo = max(int(np.searchsorted(t, a + delta, side="right")) - 1, 0)
        hi = min(int(np.searchsorted(t, b + delta, side="left")), len(t) - 1)
        if hi <= lo:
            raise StreamError(422, "BS-UI-GT-MISSING", f"the span {sp['id']} holds no GT interval on the run's "
                                                       "event span")
        if (hi - lo) * seq.height * seq.width > MAX_BENCH_MASK_BYTES:
            raise StreamError(400, "BS-UI-REQUEST", f"the run's span holds {hi - lo} GT intervals of "
                                                    f"{seq.width}x{seq.height} px: more than "
                                                    f"{MAX_BENCH_MASK_BYTES >> 20} MiB of masks; use a shorter span")
        check_events(seq, a, b, f"the sequence of span {sp['id']}")
        real, _, elig, _ = m.pipeline.real_side(seq, a, b, m.streams.DEFAULT_CHUNK, None, m.evb.code_digest())
        removed = None
        if hot:
            run = m.streams.H5Run(info.events_path)
            try:
                check_events(run, a, b, f"the run {info.run_dir.name}")
                sim_ids = m.metrics.hot_pixels(run.window(a, b).quantized(), a, b)["ids"]
            finally:
                run._f.close()
            removed = m.metrics.hot_map(real.hot["ids"], seq.width, seq.height) | \
                m.metrics.hot_map(sim_ids, seq.width, seq.height)
        try:
            reg = m.region.bench_region(seq, a, b, sp["delta_ns"], drop_ids=sp["drop_ids"])
        except ValueError as e:
            raise StreamError(422, "BS-UI-GT-MISSING", str(e)) from None
        if removed is not None:
            reg = reg.without(removed)
        lo = np.maximum(reg.t_edges[:-1] - reg.delta_ns, a)             # the GT intervals on the event clock, in [a, b)
        edges = np.stack([lo, np.maximum(np.minimum(reg.t_edges[1:] - reg.delta_ns, b), lo)], 1).astype(np.int64)
        return BenchRegion(edges, reg.masks, reg.digest(), (a, b), Eligible(a, WINDOW_NS, np.asarray(elig, bool)),
                           "EVIMO2 bench region")

    def _legacy(self, m, sp, seq, info, a: int, b: int) -> BenchRegion:
        try:
            mask, _ = m.region.legacy_region(seq, *info.span_true_ns)
        except ValueError as e:
            raise StreamError(422, "BS-UI-GT-MISSING", str(e)) from None
        reg = m.region.static_region(mask, a, b)
        return BenchRegion(np.array([[a, b]], np.int64), reg.masks, reg.digest(), (a, b), self.eligible(sp, a, b),
                           "legacy region")

    # -- the job's argv ----------------------------------------------------------------------------------------

    def result_path(self, job_dir: Path, label: str, span_id: str) -> Path:
        """Where ``evb.py compare --out <job>/bench`` writes the span's JSON: ``<out>/<label>/<span tag>.json``."""
        return Path(job_dir) / "bench" / label / f"{self.mods.evb.span_tag(span_id)}.json"

    @property
    def script(self) -> Path:
        return self.repo / "tools" / "evimo2_bench" / "evb.py"


def _close_sequence(seq) -> None:
    """Closes the NPZ archives a bench ``Evimo2`` opened (it has no ``close``)."""
    for archive in getattr(seq, "_archives", {}).values():
        archive.close()


__all__ = ["MAX_ADHOC_S", "MAX_BENCH_EVENTS", "MAX_BENCH_MASK_BYTES", "WINDOW_NS", "Bench", "BenchRegion", "Eligible",
           "check_ad_hoc", "check_events"]
