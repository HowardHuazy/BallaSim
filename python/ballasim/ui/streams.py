"""Event streams for the viewer (docs/design/phase3/ui.md §3.3 ``/streams``, §3.5 BSEV and BSRT, §4.1–4.2).

A stream is an event source opened in a root: a run's camera (``events/<camera>.h5``, contract C4), an event file
(C4/DSEC ``.h5``, ``.npz``, ``.npy`` or a directory of ``.npy`` columns) or an EVIMO2 sequence directory. Its events
are served in **blocks**, the absolute slices [k 2^28, (k + 1) 2^28) ns of its own clock, each in **LOD order**
(level descending, file order within a level: every prefix ``[0, n_ge[j])`` holds the events of level >= j, §4.1).

Python here is I/O glue (the code-review rule of §1: no per-event Python loop and no NumPy pass of its own):

- one **reader thread** per source (h5py serializes its calls, RV r4) reads a block's columns; C4 files are sliced
  through ``/ms_to_idx`` and a binary search inside one millisecond (§4.2), other sources through
  ``ballasim.viz`` (imported at the first such source; it brings cv2);
- a 3-thread **C++ pool** builds the block with the GIL released (``_core.wire.build_block``: the LOD order, the
  BSEV ``ev`` words and the block's ON counts per 1 ms bin; the permutation is not kept) and later puts lazily
  requested GT columns into LOD order (``lod_order`` + ``gather``);
- a **block LRU** of ``--cache-mb`` holds the ``ev`` words (8 B per event) and the GT columns; the small per-block
  heads (counts, ``n_ge``, duplicates) and ON counts outlive eviction;
- at most :data:`MAX_SOURCES` sources stay open; the least recently used one is closed.

Wire formats (little-endian, version 1; the client's decoders are UI-2's ``static/app/wire.js``):

- **BSEV**, one block slice: a 64-byte header, a 16-byte column-table entry per column, then the column bodies, each
  starting 8-aligned (offsets from the message start). With ``enc=deflate-raw`` (flag bit 0) the bytes after the
  64-byte header are one raw-deflate stream of that uncompressed remainder.
- **BSRT**, the rate overview: a 32-byte header, then ``total u32[n]`` and ``on u32[n]``. Totals are exact. ON counts
  come from the blocks built so far (0xFFFFFFFF elsewhere) for bins on the stream's 1 ms grid, which starts at
  ``t_offset_ns``; bins off that grid, and every bin of a non-C4 source requested with more than
  :data:`RATE_SEARCH_MAX` bins, are counted from the events themselves, ON counts included.
- **BSTK**, a timeline track (:func:`bstk_message`; served by UI-7's ``/tracks``): a 32-byte header, the column table,
  ``t_ns i64[n]``, then the extra columns.

Without WebGPU the viewer shows PNG panels (``/image``, §4.11) that ``ballasim.viz.compare.Renderer`` renders,
unchanged, from a ``viz`` source of the stream: the reader's own for a source read through viz (on its reader thread),
one opened for each panel of a C4 file (on a 1-thread panel pool).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import collections
import contextlib
import hashlib
import json
import os
import re
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from ballasim import _core

wire = _core.wire

BLOCK_SHIFT = wire.BLOCK_SHIFT
BLOCK_NS = 1 << BLOCK_SHIFT
MAX_SLICE_EVENTS = 1 << 18
MAX_SOURCES = 16
#: Blocks per ``/blocks`` request.
MAX_BLOCKS = 256
#: The ON-count grid: 1 ms bins from the stream's ``t_offset_ns``.
RATE_BIN_NS = 1_000_000
MAX_RATE_BINS = 1 << 20
#: Up to this many bins, a non-C4 source's totals take one search per bin edge (§3.5); above, one pass over the span.
RATE_SEARCH_MAX = 4096
#: A pass over a span (``/rate`` off the grid) reads and counts at most this many events at a time.
RATE_CHUNK_EVENTS = 1 << 19
#: Block builds of one ``/blocks`` request that run at once (one read on the reader thread, the rest on the C++ pool).
HEAD_BUILDS = 2
#: HDF5's chunk cache of each dataset a C4 reader keeps open (``rdcc_nbytes``; one per open dataset, outside
#: ``--cache-mb``, §5.3). The writer's chunks hold 2^17 events (``t``: 1 MiB), so this keeps the chunks that a block's
#: boundary searches share with its read. A larger cache only keeps more decompressed chunks resident: after reading
#: every lab block, 16 MiB held +118 MB of RSS per source (+255 MB with its GT) against +28 MB (+77 MB) here, at the
#: same read speed.
H5_CHUNK_CACHE = 2 << 20
#: ON counts of an aligned request are summed over at most this many 1 ms grid bins (else all invalid).
MAX_GRID_BINS = 1 << 23
WIRE_VERSION = 1
BSEV_MAGIC = int.from_bytes(b"BSEV", "little")
BSRT_MAGIC = int.from_bytes(b"BSRT", "little")
BSTK_MAGIC = int.from_bytes(b"BSTK", "little")
NO_COUNT = 0xFFFFFFFF
#: BSEV flags.
FLAG_DEFLATE, FLAG_LOD, FLAG_BLOCK_END, FLAG_WORKER = 1, 2, 4, 8
#: The BSEV header (§3.5): magic, version, flags, reserved, n_events, t_base_ns, i0, n_block, width, height,
#: n_columns, header_bytes, coverage_lo, coverage_hi, 16 reserved bytes.
BSEV_HEADER = struct.Struct("<IHHIIqIIHHHHII16x")
BSEV_COLUMN = struct.Struct("<HBBIII")
#: The BSRT header: magic, version, flags, reserved, n_bins, t0_ns, bin_ns, reserved.
BSRT_HEADER = struct.Struct("<IHHIIqII")
#: The BSTK header: magic, version, kind, n, n_extra_columns, t0_ns, 8 reserved bytes.
BSTK_HEADER = struct.Struct("<IHHIIq8x")
assert BSEV_HEADER.size == 64 and BSEV_COLUMN.size == 16 and BSRT_HEADER.size == 32 and BSTK_HEADER.size == 32
#: BSTK track kinds and the preview track's extra columns (id, dtype code, components, NumPy dtype).
TRACK_KINDS = {"preview": 0, "frames": 1, "imu": 2}
TRACK_COLUMNS = {"index": (1, 3, 1, np.dtype("<u4")), "window": (2, 7, 2, np.dtype("<i8"))}


@dataclass(frozen=True)
class Column:
    """A BSEV column: id, dtype code (1 u8, 2 u16, 3 u32, 4 i32, 5 f16, 6 f32, 7 i64), components, NumPy dtype and
    the per-event fields it is made of (C4 ``/gt/<field>``; interleaved when several)."""

    id: int
    code: int
    comps: int
    dtype: np.dtype
    fields: tuple[str, ...] = ()


COLUMNS = {
    "ev": Column(0, 3, 2, np.dtype("<u4")),
    "instance": Column(16, 3, 1, np.dtype("<u4"), ("instance",)),
    "semantic": Column(17, 2, 1, np.dtype("<u2"), ("semantic",)),
    "depth": Column(18, 6, 1, np.dtype("<f4"), ("depth",)),
    "flow": Column(19, 6, 2, np.dtype("<f4"), ("flow_x", "flow_y")),
    "t_cross": Column(20, 7, 1, np.dtype("<i8"), ("t_cross_ns",)),
    "flags": Column(21, 2, 1, np.dtype("<u2"), ("flags",)),
    # worker (re-sim) frames only (FLAG_WORKER, UI-11): RawEvent's increments; no file stream has them
    "ds": Column(22, 6, 1, np.dtype("<f4")),
    "dn": Column(23, 6, 1, np.dtype("<f4")),
    "dl": Column(24, 6, 1, np.dtype("<f4")),
    "l_first": Column(25, 6, 1, np.dtype("<f4")),
}
GT_FIELDS = frozenset(f for c in COLUMNS.values() for f in c.fields) | {"label"}
POLARITY_MODES = ("auto", "normal", "inverted")
#: The PNG fallback's kinds (``ballasim.viz.compare.KINDS``) and the most events it reads for one panel.
IMAGE_KINDS = ("events", "polarity", "counts", "timesurface")
MAX_IMAGE_EVENTS = 1 << 25
#: The most pixels of a PNG panel: viz allocates per pixel whatever the window holds (§5.3), 4096² here.
MAX_IMAGE_PIXELS = 1 << 24
_CAMERA = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_EVIMO2_EVENTS = "dataset_events_t.npy"


class StreamError(Exception):
    """A refused stream request: HTTP status, BS-UI code and message (§6)."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _source_error(e: Exception) -> StreamError:
    text = str(e) or type(e).__name__
    if "16383" in text:
        return StreamError(422, "BS-UI-TOO-WIDE", text)
    return StreamError(422, "BS-UI-SOURCE", text)


@contextlib.contextmanager
def _path_errors():
    """The OS's refusals of a source path as BS-UI codes (§6): a malformed path (a NUL character) 400 BS-UI-REQUEST,
    a missing one 404 BS-UI-NOT-FOUND (also when it vanishes between two steps), one this user may not search or read
    403 BS-UI-PATH, any other OSError 422 BS-UI-SOURCE."""
    try:
        yield
    except ValueError:
        raise StreamError(400, "BS-UI-REQUEST", "not a valid path") from None
    except FileNotFoundError:
        raise StreamError(404, "BS-UI-NOT-FOUND", "no such file or directory") from None
    except PermissionError:
        raise StreamError(403, "BS-UI-PATH", "this user may not read the path") from None
    except OSError as e:
        raise StreamError(422, "BS-UI-SOURCE", e.strerror or type(e).__name__) from None


# ---------------------------------------------------------------------------------------------------------------
# Paths in roots
# ---------------------------------------------------------------------------------------------------------------

def encode_rid(root_id: str, relpath: str) -> str:
    """``rid`` (§3.3): base64url without padding of ``"<root_id>/<relpath>"``."""
    return base64.urlsafe_b64encode(f"{root_id}/{relpath}".encode()).rstrip(b"=").decode()


def decode_rid(rid: str) -> tuple[str, str]:
    try:
        text = base64.urlsafe_b64decode(rid + "=" * (-len(rid) % 4)).decode()
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise StreamError(404, "BS-UI-NOT-FOUND", "not a resource id") from None
    root_id, sep, rel = text.partition("/")
    if not sep:
        raise StreamError(404, "BS-UI-NOT-FOUND", "not a resource id")
    return root_id, rel


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


@_path_errors()
def resolve_path(roots, ref: str) -> Path:
    """The real path a ``rid`` or an absolute path names, confined to the roots by ``realpath`` containment (a
    symlink escaping its root is 403 BS-UI-PATH)."""
    if not isinstance(ref, str) or not ref:
        raise StreamError(400, "BS-UI-REQUEST", "expected a resource id or an absolute path")
    if ref.startswith("/"):
        candidate, allowed = ref, [os.path.realpath(r.path) for r in roots]
    else:
        root_id, rel = decode_rid(ref)
        root = next((r for r in roots if r.id == root_id), None)
        if root is None:
            raise StreamError(404, "BS-UI-NOT-FOUND", f"no root {root_id!r}")
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise StreamError(403, "BS-UI-PATH", "the path leaves its root")
        candidate, allowed = os.path.join(root.path, rel), [os.path.realpath(root.path)]
    real = os.path.realpath(candidate)
    if not any(_inside(real, a) for a in allowed):
        raise StreamError(403, "BS-UI-PATH", "the path lies outside the roots")
    if not os.path.exists(real):
        raise StreamError(404, "BS-UI-NOT-FOUND", "no such file or directory")
    return Path(real)


def _partial_name(name: str) -> bool:
    return name.startswith(".") and name.endswith(".partial")


@_path_errors()
def locate_source(roots, spec) -> tuple[Path, str]:
    """``(path, kind)`` of a ``/streams`` source: ``{run: rid, camera}``, ``{path: rid}`` or ``{evimo2: rid}`` (an
    absolute path inside a root may stand for each rid). ``kind``: ``c4`` (an ``.h5``/``.hdf5`` file) or ``view``
    (read through ``ballasim.viz``). 409 BS-UI-PARTIAL for an event file still being written; the OS's refusals as
    :func:`_path_errors` maps them."""
    if not isinstance(spec, dict) or sum(k in spec for k in ("run", "path", "evimo2")) != 1:
        raise StreamError(400, "BS-UI-REQUEST", "source: give exactly one of run (with camera), path or evimo2")
    if "run" in spec:
        camera = spec.get("camera")
        if not isinstance(camera, str) or not _CAMERA.fullmatch(camera):
            raise StreamError(400, "BS-UI-REQUEST", "source.camera: a camera name")
        run = resolve_path(roots, spec["run"])
        path = run / "events" / f"{camera}.h5"
        if not path.is_file():
            if (run / "events" / f".{camera}.h5.partial").exists():
                raise StreamError(409, "BS-UI-PARTIAL", "the event file is still being written; it is available "
                                                        "when the run completes")
            raise StreamError(404, "BS-UI-NOT-FOUND", f"the run has no events/{camera}.h5")
        return resolve_path(roots, str(path)), "c4"
    if "evimo2" in spec:
        path = resolve_path(roots, spec["evimo2"])
        if not (path / _EVIMO2_EVENTS).is_file():
            raise StreamError(422, "BS-UI-SOURCE", f"not an EVIMO2 sequence directory (no {_EVIMO2_EVENTS})")
        return path, "view"
    path = resolve_path(roots, spec["path"])
    if _partial_name(path.name):
        raise StreamError(409, "BS-UI-PARTIAL", "the event file is still being written; it is available when the run "
                                                "completes")
    if path.is_dir():
        if (path / _EVIMO2_EVENTS).is_file() or any(path.glob("*.npy")):
            return path, "view"
    elif path.suffix.lower() in (".h5", ".hdf5"):
        return path, "c4"
    elif path.suffix.lower() in (".npz", ".npy"):
        return path, "view"
    raise StreamError(422, "BS-UI-SOURCE", "not an event source (.h5, .npz, .npy, a directory of .npy columns or an "
                                           "EVIMO2 sequence)")


def source_files(path: Path) -> list[Path]:
    """The files whose identity the stream id covers."""
    if path.is_dir():
        names = sorted(p for p in path.glob("*.npy") if p.is_file())
        return names or [path]
    return [path]


@_path_errors()
def stream_id(path: Path, polarity: str) -> str:
    """``sid`` of a closed file (§3.3): SHA-256 over its real path, the (inode, size, mtime_ns) of its files, the
    polarity mode, the BSEV major version and ``wire::kLodKeyVersion``; 16 hex digits."""
    stats = [[str(p), (st := p.stat()).st_ino, st.st_size, st.st_mtime_ns] for p in source_files(path)]
    text = json.dumps([str(path), stats, polarity, WIRE_VERSION, wire.LOD_KEY_VERSION])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------
# Readers (each used only on its stream's reader thread)
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Columns:
    """A block's events in file order: ``t_ns`` int64, ``x``/``y`` uint16, ``p`` uint8 0/1 (1 = ON), the S1
    ``label`` (or None) and requested GT fields."""

    t_ns: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    label: np.ndarray | None = None
    gt: dict = field(default_factory=dict)


class C4Reader:
    """A C4 event file (engine/writer.py; DSEC layout with ``resolution`` or ``width``/``height`` attributes):
    ``/events/{x, y, p, t, t_ns_rem}``, ``/t_offset``, ``/ms_to_idx`` and ``/gt/*``."""

    kind = "c4"

    def __init__(self, path: Path, invert: bool):
        import hdf5plugin  # noqa: F401  (registers Blosc with h5py)

        self.path, self.invert = path, invert
        self.file = h5py.File(path, "r", rdcc_nbytes=H5_CHUNK_CACHE)
        try:
            ev = self.file["events"]
            self._x, self._y, self._p, self._t = (ev[k] for k in ("x", "y", "p", "t"))
            self._rem = ev.get("t_ns_rem")
            self.n = int(self._t.shape[0])
            attrs = self.file.attrs
            if "resolution" in attrs:
                self.width, self.height = (int(v) for v in np.asarray(attrs["resolution"]).ravel()[:2])
            else:
                self.width, self.height = int(attrs["width"]), int(attrs["height"])
            self.t_offset_us = int(self.file["t_offset"][()]) if "t_offset" in self.file else 0
            if "ms_to_idx" in self.file:
                self.ms_to_idx = np.asarray(self.file["ms_to_idx"][()], np.int64)
            else:      # DSEC files may lack it: one search per millisecond over the whole time column
                t_us = np.asarray(self._t[()], np.int64)
                n_ms = int(t_us[-1]) // 1000 + 1 if self.n else 0
                self.ms_to_idx = np.searchsorted(t_us, np.arange(n_ms, dtype=np.int64) * 1000, side="left")
            gt = self.file.get("gt")
            self._gt = {name: ds for name, ds in (gt.items() if gt is not None else ())
                        if name in GT_FIELDS and isinstance(ds, h5py.Dataset) and ds.shape == (self.n,)}
            self.fields = frozenset(self._gt)
            self.t_offset_ns = 1000 * self.t_offset_us
            if self.n:
                self.t_start_ns = int(self._times(0, 1)[0])
                self.t_end_ns = int(self._times(self.n - 1, self.n)[0]) + 1
            else:
                self.t_start_ns = self.t_end_ns = self.t_offset_ns
        except Exception:
            self.file.close()
            raise

    def _times(self, i0: int, i1: int) -> np.ndarray:
        t_us = self._t[i0:i1]
        if t_us.dtype != np.int64:
            t_us = t_us.astype(np.int64)
        rem = self._rem[i0:i1] if self._rem is not None else None
        if rem is not None and rem.dtype != np.uint16:
            rem = rem.astype(np.uint16)
        return wire.c4_times(t_us, rem, self.t_offset_us)

    def search(self, t_ns: int) -> int:
        """The first event with t >= t_ns: ``/ms_to_idx`` bounds it to one millisecond (§4.2)."""
        rel = t_ns - self.t_offset_ns
        ms, index, m = rel // RATE_BIN_NS, self.ms_to_idx, len(self.ms_to_idx)
        if m == 0:
            lo, hi = 0, self.n
        elif ms < 0:
            lo, hi = 0, int(index[0])
        elif ms >= m:
            lo, hi = int(index[-1]), self.n
        else:
            lo, hi = int(index[ms]), int(index[ms + 1]) if ms + 1 < m else self.n
        if lo >= hi:
            return lo
        return lo + int(np.searchsorted(self._times(lo, hi), t_ns, side="left"))

    def _coords(self, ds, i0: int, i1: int) -> np.ndarray:
        a = ds[i0:i1]
        return a if a.dtype == np.uint16 else _checked_coords(a)

    def read(self, i0: int, i1: int, gt: tuple[str, ...] = (), labels: bool = True) -> Columns:
        p = self._p[i0:i1]
        if p.dtype != np.uint8:              # signed (-1/+1) or other polarities: ON is p > 0, as in ballasim.viz
            p = (p > 0).view(np.uint8)
        label = None
        if labels and "label" in self._gt:
            label = self._gt["label"][i0:i1]
            label = label if label.dtype == np.uint8 else label.astype(np.uint8)
        return Columns(self._times(i0, i1), self._coords(self._x, i0, i1), self._coords(self._y, i0, i1),
                       wire.polarity01(np.ascontiguousarray(p), self.invert), label,
                       {name: self._gt[name][i0:i1] for name in gt})

    def grid_totals(self, edges: np.ndarray) -> np.ndarray:
        """Event indices at edges on the 1 ms grid (multiples of 1 ms after ``t_offset_ns``), from ``/ms_to_idx``."""
        m = (edges - self.t_offset_ns) // RATE_BIN_NS
        index = np.concatenate([self.ms_to_idx, [self.n]])
        out = index[np.clip(m, 0, len(self.ms_to_idx))]
        if len(self.ms_to_idx) and self.ms_to_idx[0] > 0:      # events before t_offset (not written by BallaSim)
            neg = m < 0                                         # only edges before t_offset need a search
            if neg.any():
                out = out.copy()
                out[neg] = [self.search(int(e)) for e in edges[neg]]
        return out

    def close(self) -> None:
        self.file.close()


def _checked_coords(a: np.ndarray) -> np.ndarray:
    """Coordinates of another integer type as uint16, refused outside [0, 16383] (BS-UI-TOO-WIDE)."""
    if not np.issubdtype(a.dtype, np.integer):
        raise StreamError(422, "BS-UI-SOURCE", f"coordinates of type {a.dtype}")
    if len(a) and (int(a.min()) < 0 or int(a.max()) > wire.MAX_COORD):
        raise StreamError(422, "BS-UI-TOO-WIDE", "coordinates outside 0..16383")
    return np.ascontiguousarray(a, np.uint16)


class ViewReader:
    """Any other source through ``ballasim.viz.io.open_events``: EVIMO2 sequences (float32 seconds, Samsung polarity
    under ``auto``), ``.npz``/``.npy`` arrays, HDF5 files without C4 attributes."""

    kind = "view"
    t_offset_ns = 0

    def __init__(self, path: Path, polarity: str):
        from ballasim.viz.io import open_events

        self.path = path
        self.source = open_events(path, polarity=polarity)
        self.n = len(self.source)
        self.width, self.height = self.source.width, self.source.height
        self.fields = frozenset(self.source.extra_fields) & GT_FIELDS
        self.t_start_ns = self.source.t_start_ns if self.n else 0
        self.t_end_ns = self.source.t_end_ns if self.n else 0

    def search(self, t_ns: int) -> int:
        return self.source.search(t_ns)

    def read(self, i0: int, i1: int, gt: tuple[str, ...] = (), labels: bool = True) -> Columns:
        extras = tuple(gt) + (("label",) if labels and "label" in self.fields else ())
        ev = self.source.read(i0, i1, extras)
        label = ev.extras.get("label") if labels else None
        if label is not None and label.dtype != np.uint8:
            label = label.astype(np.uint8)
        x = ev.x if ev.x.dtype == np.uint16 else _checked_coords(ev.x)
        y = ev.y if ev.y.dtype == np.uint16 else _checked_coords(ev.y)
        return Columns(np.ascontiguousarray(ev.t_ns, np.int64), np.ascontiguousarray(x), np.ascontiguousarray(y),
                       np.ascontiguousarray(ev.p, np.bool_).view(np.uint8),
                       None if label is None else np.ascontiguousarray(label),
                       {name: np.ascontiguousarray(ev.extras[name]) for name in gt})

    def close(self) -> None:
        f = getattr(self.source, "_file", None)
        if f is not None:
            f.close()


# ---------------------------------------------------------------------------------------------------------------
# Streams, blocks and the LRU
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Head:
    """What a built block leaves behind: its index range in the file, counts and ON counts per 1 ms grid bin."""

    i_a: int
    i_b: int
    n_ge: list[int]
    duplicates: int
    rate_bin0: int             # the first grid bin of ``on``
    on: np.ndarray


@dataclass
class Block:
    """An LRU entry: the ``ev`` words in LOD order and lazily gathered GT columns (by BSEV column name)."""

    ev: np.ndarray
    gt: dict = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def nbytes(self) -> int:
        return self.ev.nbytes + sum(a.nbytes for a in self.gt.values())


class Stream:
    """One open source: its reader (on its own thread), heads of built blocks, label and polarity mode."""

    def __init__(self, sid: str, path: Path, reader, label: str, polarity: str):
        self.sid, self.path, self.reader, self.label, self.polarity = sid, path, reader, label, polarity
        self.executor: ThreadPoolExecutor | None = None
        self.heads: dict[int, Head] = {}
        #: In-flight block builds of this Stream object (a reopened source, same sid, starts its own).
        self.building: dict[int, asyncio.Future] = {}
        self.k_first = reader.t_start_ns >> BLOCK_SHIFT
        self.k_end = ((reader.t_end_ns - 1) >> BLOCK_SHIFT) + 1 if reader.n else self.k_first
        self.closed = False
        #: The PNG fallback's automatic clips, one per timeline: {(kind, window length ns): clip} (§4.5). Read and
        #: written only on the thread that runs the stream's panels (:meth:`Streams.image`).
        self.auto_clips: dict[tuple[str, int], float] = {}

    @contextlib.contextmanager
    def viz_source(self):
        """A ``ballasim.viz`` event source of the stream (its path and polarity mode) for one PNG panel. A stream read
        through viz lends its reader's source, already open (the panel runs on its reader thread). A C4 file, which
        its reader reads with h5py, gets a source opened for the panel and closed after it: kept open, that source
        would hold viz's HDF5 chunk caches, 64 MiB per dataset, outside the block LRU (§5.3). The OS's refusals of the
        path map as at the stream's open: a path deleted since 404 BS-UI-NOT-FOUND, one this user may no longer read
        403 BS-UI-PATH."""
        from ballasim.viz.io import open_events

        if self.reader.kind == "view":
            yield self.reader.source
            return
        try:
            source = open_events(self.path, polarity=self.polarity)
        except FileNotFoundError:
            raise StreamError(404, "BS-UI-NOT-FOUND", "the stream's source no longer exists") from None
        except PermissionError:
            raise StreamError(403, "BS-UI-PATH", "this user may not read the path") from None
        try:
            yield source
        finally:
            for name in ("_file", "_dense"):     # the event file and, for a run with dense frames, dense/<c>.h5
                f = getattr(source, name, None)
                if f is not None:
                    f.close()

    @property
    def gt_columns(self) -> list[str]:
        return [name for name, c in COLUMNS.items() if c.fields and all(f in self.reader.fields for f in c.fields)]

    def coverage(self, k: int) -> tuple[int, int]:
        """[lo, hi) ns after the block start where the block holds every event: the whole block, except the last
        one, which ends at ``t_end_ns`` (one ns past the last event)."""
        hi = min(BLOCK_NS, self.reader.t_end_ns - (k << BLOCK_SHIFT)) if k == self.k_end - 1 else BLOCK_NS
        return 0, max(hi, 0)

    def rate_bins(self, k: int) -> tuple[int, int]:
        """The 1 ms grid bins [b0, b1) that block k meets."""
        g0 = self.reader.t_offset_ns
        b0 = ((k << BLOCK_SHIFT) - g0) // RATE_BIN_NS
        b1 = (((k + 1) << BLOCK_SHIFT) - 1 - g0) // RATE_BIN_NS + 1
        return b0, b1

    def head_json(self, k: int) -> dict:
        h = self.heads[k]
        lo, hi = self.coverage(k)
        return {"k": k, "t_base_ns": str(k << BLOCK_SHIFT), "n": h.n_ge[0], "n_ge": h.n_ge, "coverage": [lo, hi],
                "complete": True, "lod": True, "duplicates": h.duplicates}

    def info(self) -> dict:
        r = self.reader
        return {"sid": self.sid, "width": r.width, "height": r.height, "t_start_ns": str(r.t_start_ns),
                "t_end_ns": str(r.t_end_ns), "n_events": r.n, "block_shift": BLOCK_SHIFT,
                "gt_columns": self.gt_columns, "labels": "label" in r.fields, "closed": True, "clock": "sensor",
                "t_offset_ns": str(r.t_offset_ns), "label": self.label, "polarity": self.polarity,
                "duplicates": sum(h.duplicates for h in self.heads.values()),
                "duplicates_final": len(self.heads) == self.k_end - self.k_first}


class Streams:
    """The source registry, the block LRU and the C++ pool of one server."""

    def __init__(self, roots, cache_mb: int, *, max_sources: int = MAX_SOURCES):
        self.roots = roots
        self.budget = int(cache_mb) << 20
        self.max_sources = max_sources
        self.cpp_pool = ThreadPoolExecutor(3, thread_name_prefix="ballasim-ui-wire")
        #: The PNG fallback's panels of C4 streams, one at a time (other streams' panels run on their reader thread).
        self.panel_pool = ThreadPoolExecutor(1, thread_name_prefix="ballasim-ui-panel")
        self.streams: collections.OrderedDict[str, Stream] = collections.OrderedDict()
        self.blocks: collections.OrderedDict[tuple[str, int], Block] = collections.OrderedDict()
        self.bytes = 0
        self._opening: dict[str, asyncio.Future] = {}

    # -- sources -------------------------------------------------------------------------------------------------

    async def open(self, body: dict) -> Stream:
        """``POST /streams``: the stream of ``body["source"]``, opened once per sid."""
        polarity = body.get("polarity", "auto")
        if polarity not in POLARITY_MODES:
            raise StreamError(400, "BS-UI-REQUEST", f"polarity: one of {', '.join(POLARITY_MODES)}")
        label = body.get("label")
        if label is not None and not isinstance(label, str):
            raise StreamError(400, "BS-UI-REQUEST", "label: a string")
        loop = asyncio.get_running_loop()

        def locate():
            path, kind = locate_source(self.roots, body.get("source"))
            return path, kind, stream_id(path, polarity)

        path, kind, sid = await loop.run_in_executor(None, locate)
        stream = self.streams.get(sid)
        if stream is not None:
            self.streams.move_to_end(sid)
            return stream
        fut = self._opening.get(sid)
        if fut is None:
            fut = asyncio.ensure_future(self._open(sid, path, kind, polarity, label))
            self._opening[sid] = fut
            fut.add_done_callback(lambda _: self._opening.pop(sid, None))
        return await asyncio.shield(fut)

    async def _open(self, sid, path, kind, polarity, label) -> Stream:
        executor = ThreadPoolExecutor(1, thread_name_prefix=f"ballasim-ui-read-{sid[:6]}")

        def make():
            if kind == "c4":
                try:
                    return C4Reader(path, polarity == "inverted")
                except KeyError:
                    pass                    # an HDF5 file without C4 attributes: through viz
            return ViewReader(path, polarity)

        try:
            reader = await asyncio.get_running_loop().run_in_executor(executor, make)
        except StreamError:
            executor.shutdown(wait=False)
            raise
        except ImportError as e:                            # ballasim.viz needs the viz extra (cv2, matplotlib)
            executor.shutdown(wait=False)
            raise StreamError(422, "BS-UI-SOURCE", f"reading this source needs ballasim[viz]: {e}") from None
        except PermissionError:                             # as _path_errors: a file this user may not read
            executor.shutdown(wait=False)
            raise StreamError(403, "BS-UI-PATH", "this user may not read the path") from None
        except (OSError, ValueError, KeyError, TypeError) as e:
            executor.shutdown(wait=False)
            raise _source_error(e) from None
        stream = Stream(sid, path, reader, label or path.name, polarity)
        stream.executor = executor
        if max(reader.width, reader.height) > wire.MAX_COORD:
            self._close(stream)
            raise StreamError(422, "BS-UI-TOO-WIDE", f"{reader.width}x{reader.height}: the wire carries coordinates "
                                                     "up to 16383")
        self.streams[sid] = stream
        while len(self.streams) > self.max_sources:
            _, old = self.streams.popitem(last=False)
            self._drop(old)
        return stream

    def get(self, sid: str) -> Stream:
        stream = self.streams.get(sid)
        if stream is None:
            raise StreamError(404, "BS-UI-NOT-FOUND", f"no stream {sid}")
        self.streams.move_to_end(sid)
        return stream

    def delete(self, sid: str) -> None:
        """``DELETE /streams/{sid}``: closes the source and drops its blocks."""
        self._drop(self.get(sid))
        del self.streams[sid]

    def _drop(self, stream: Stream) -> None:
        for key in [key for key in self.blocks if key[0] == stream.sid]:
            self.bytes -= self.blocks.pop(key).nbytes
        self._close(stream)

    @staticmethod
    def _close(stream: Stream) -> None:
        stream.closed = True
        if stream.executor is not None:
            stream.executor.submit(stream.reader.close)     # after any read already queued
            stream.executor.shutdown(wait=False)

    def close(self) -> None:
        for stream in self.streams.values():
            self._close(stream)
        self.streams.clear()
        self.blocks.clear()
        self.bytes = 0
        self.cpp_pool.shutdown(wait=False)
        self.panel_pool.shutdown(wait=False)

    async def _read(self, stream: Stream, fn, *args):
        if stream.closed:
            raise StreamError(404, "BS-UI-NOT-FOUND", f"stream {stream.sid} was closed")
        try:
            return await asyncio.get_running_loop().run_in_executor(stream.executor, fn, *args)
        except RuntimeError as e:                           # the executor was shut down meanwhile
            if stream.closed:
                raise StreamError(404, "BS-UI-NOT-FOUND", f"stream {stream.sid} was closed") from None
            raise _source_error(e) from None
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise _source_error(e) from None

    async def _cpp(self, fn, *args):
        try:
            return await asyncio.get_running_loop().run_in_executor(self.cpp_pool, fn, *args)
        except (ValueError, TypeError) as e:
            raise _source_error(e) from None

    # -- blocks --------------------------------------------------------------------------------------------------

    def check_block(self, stream: Stream, k: int) -> None:
        if not stream.k_first <= k < stream.k_end:
            raise StreamError(404, "BS-UI-NOT-FOUND", f"block {k} lies outside the stream (blocks "
                                                      f"{stream.k_first}..{stream.k_end - 1})")

    async def heads(self, stream: Stream, k0: int, k1: int) -> list[dict]:
        """``GET /streams/{sid}/blocks?k0&k1``: the heads of blocks [k0, k1) within the stream, built as needed.

        At most :data:`HEAD_BUILDS` builds of one request run at once, and their blocks are not kept here: only the
        LRU holds them, within ``--cache-mb``."""
        ks = range(max(k0, stream.k_first), min(k1, stream.k_end))
        slots = asyncio.Semaphore(HEAD_BUILDS)

        async def build(k: int) -> None:
            async with slots:
                if k not in stream.heads:
                    await self.block(stream, k)

        await asyncio.gather(*(build(k) for k in ks if k not in stream.heads))
        return [stream.head_json(k) for k in ks]

    async def block(self, stream: Stream, k: int) -> Block:
        """Block k of the stream, from the LRU or built (one build per block at a time)."""
        key = (stream.sid, k)
        entry = self.blocks.get(key)
        if entry is not None:
            self.blocks.move_to_end(key)
            return entry
        fut = stream.building.get(k)
        if fut is None:
            fut = asyncio.ensure_future(self._build(stream, k))
            stream.building[k] = fut
            fut.add_done_callback(lambda _: stream.building.pop(k, None))
        return await asyncio.shield(fut)

    async def _build(self, stream: Stream, k: int) -> Block:
        b0, b1 = stream.rate_bins(k)

        def read():
            r = stream.reader
            i_a, i_b = r.search(k << BLOCK_SHIFT), r.search((k + 1) << BLOCK_SHIFT)
            return i_a, i_b, r.read(i_a, i_b)

        i_a, i_b, cols = await self._read(stream, read)
        built = await self._cpp(wire.build_block, cols.t_ns, cols.x, cols.y, cols.p, cols.label, k << BLOCK_SHIFT,
                                stream.reader.t_offset_ns + b0 * RATE_BIN_NS, RATE_BIN_NS, b1 - b0)
        stream.heads[k] = Head(i_a, i_b, built["n_ge"], built["duplicates"], b0, built["on"])
        entry = Block(built["ev"])
        if not stream.closed:
            self._insert((stream.sid, k), entry)
        return entry

    def _insert(self, key, entry: Block) -> None:
        old = self.blocks.pop(key, None)
        if old is not None:
            self.bytes -= old.nbytes
        self.blocks[key] = entry
        self.bytes += entry.nbytes
        self._evict(keep=key)

    def _evict(self, keep=None) -> None:
        while self.bytes > self.budget and len(self.blocks) > 1:
            key = next(iter(self.blocks))
            if key == keep:
                self.blocks.move_to_end(key)
                key = next(iter(self.blocks))
            self.bytes -= self.blocks.pop(key).nbytes

    async def gt_columns(self, stream: Stream, k: int, entry: Block, names: list[str]) -> None:
        """Puts the GT columns ``names`` of block k into LOD order (a fresh ``lod_order`` and a ``gather``, §4.2)."""
        async with entry.lock:
            missing = [name for name in names if name not in entry.gt]
            if not missing:
                return
            head = stream.heads[k]
            fields = tuple(f for name in missing for f in COLUMNS[name].fields)
            cols = await self._read(stream, stream.reader.read, head.i_a, head.i_b, fields, False)

            def order_and_gather():
                perm = wire.lod_order(cols.t_ns, cols.x, cols.y, cols.p)["perm"]
                out = {}
                for name in missing:
                    c = COLUMNS[name]
                    src = [np.ascontiguousarray(cols.gt[f], c.dtype) for f in c.fields]
                    out[name] = wire.gather(perm, src)
                return out

            gathered = await self._cpp(order_and_gather)
            key = (stream.sid, k)
            before = entry.nbytes
            entry.gt.update(gathered)
            if self.blocks.get(key) is entry:
                self.bytes += entry.nbytes - before
                self._evict(keep=key)

    async def slice(self, stream: Stream, k: int, i0: int, i1: int, names: list[str], deflate: bool) -> bytes:
        """``GET /streams/{sid}/blocks/{k}``: one BSEV message of events [i0, i1) of block k in LOD order."""
        self.check_block(stream, k)
        unknown = [name for name in names if name not in COLUMNS]
        if not names or unknown:
            raise StreamError(400, "BS-UI-REQUEST", f"cols: names among {', '.join(COLUMNS)}")
        absent = [name for name in names if name != "ev" and name not in stream.gt_columns]
        if absent:
            raise StreamError(422, "BS-UI-GT-MISSING", f"the stream has no GT column {', '.join(absent)}")
        if i1 - i0 > MAX_SLICE_EVENTS:
            raise StreamError(416, "BS-UI-SLICE", f"a slice holds at most {MAX_SLICE_EVENTS} events")
        entry = await self.block(stream, k)
        n = len(entry.ev)
        if not 0 <= i0 <= i1 <= n:
            raise StreamError(416, "BS-UI-SLICE", f"need 0 <= i0 <= i1 <= {n}, the events of block {k}")
        gt = [name for name in names if name != "ev"]
        if gt:
            await self.gt_columns(stream, k, entry, gt)
        ordered = sorted(set(names), key=lambda name: COLUMNS[name].id)
        arrays = [(name, (entry.ev if name == "ev" else entry.gt[name])[i0:i1]) for name in ordered]
        lo, hi = stream.coverage(k)
        flags = FLAG_LOD | (FLAG_BLOCK_END if i1 == n else 0)
        r = stream.reader
        return await self._cpp(bsev_message, arrays, flags, k << BLOCK_SHIFT, i0, n, r.width, r.height, lo, hi,
                               deflate)

    # -- PNG fallback --------------------------------------------------------------------------------------------

    async def image(self, stream: Stream, t0_ns: int, t1_ns: int, kind: str, theme: str, clip: float | None,
                    gamma: float, tau_ms: float) -> tuple[bytes, float | None]:
        """``GET /streams/{sid}/image``: a PNG panel of [t0_ns, t1_ns) rendered by ``viz.compare.Renderer`` (no mask
        contours), and the clip used: ``clip``, else viz's ``auto_clip`` over the stream's timeline
        ``Timeline.from_range(t_start, t_end, t1_ns − t0_ns)``, computed once per (kind, length) and kept with the
        stream, so that every panel of a timeline shares it (§4.5); None for the time surface, whose colours do not
        depend on it.

        A stream's panels run one at a time on one thread, never on the server's shared pool: a stream read through
        viz on its reader thread (the panel shares its source), a C4 file on the 1-thread panel pool. Refusals: 422
        BS-UI-TOO-WIDE beyond :data:`MAX_IMAGE_PIXELS` pixels (viz's working set grows with them, §5.3); 416
        BS-UI-SLICE when the panel (a time surface: its 5τ history), or a window of its automatic clip, holds more
        than :data:`MAX_IMAGE_EVENTS` events."""
        r = stream.reader
        if r.width * r.height > MAX_IMAGE_PIXELS:
            raise StreamError(422, "BS-UI-TOO-WIDE", f"{r.width}x{r.height}: a PNG panel draws at most "
                                                     f"{MAX_IMAGE_PIXELS} pixels")

        def render():
            import cv2

            if stream.closed:
                raise StreamError(404, "BS-UI-NOT-FOUND", f"stream {stream.sid} was closed")
            with stream.viz_source() as source:
                image, used = panel(source)
            ok, png = cv2.imencode(".png", np.ascontiguousarray(image[..., ::-1]))
            if not ok:
                raise StreamError(500, "BS-UI-INTERNAL", "PNG encoding failed")
            return png.tobytes(), used

        def bounded(n: int, what: str) -> None:
            if n > MAX_IMAGE_EVENTS:
                raise StreamError(416, "BS-UI-SLICE", f"{what} holds {n} events; a PNG panel reads at most "
                                                      f"{MAX_IMAGE_EVENTS}: shorten it")

        def panel(source):
            from ballasim.viz.compare import Renderer, RenderOptions, Timeline

            if kind == "timesurface":                     # viz reads [t1 − int(5τ), t1) and no clip
                bounded(source.count(t1_ns - int(5 * (tau_ms * 1_000_000)), t1_ns), "the 5τ history")
                used = None
            else:
                bounded(source.count(t0_ns, t1_ns), "the window")
                used = clip if clip is not None else timeline_clip(source)
            renderer = Renderer([source], Timeline(np.array([t0_ns], np.int64), t1_ns - t0_ns),
                                RenderOptions(kind=kind, theme=theme, clip=used or 1.0, gamma=gamma, tau_ms=tau_ms,
                                              mask="none"))
            return renderer.render(0).panels[0].image, used

        def timeline_clip(source) -> float:
            from ballasim.viz.compare import Timeline, auto_clip

            length = min(t1_ns - t0_ns, r.t_end_ns - r.t_start_ns)       # as the GPU viewer's Auto (§4.5)
            key = ("events" if kind == "events" else "sum", length)
            if key not in stream.auto_clips:
                if length <= 0:                                           # no events
                    return 1.0
                timeline = Timeline.from_range(r.t_start_ns, r.t_end_ns, length)
                for k in np.unique(np.linspace(0, len(timeline) - 1, min(16, len(timeline))).astype(int)):
                    bounded(source.count(*timeline.window(int(k))), "a window of the automatic clip")   # viz's picks
                stream.auto_clips[key] = auto_clip(source, timeline, kind)
            return stream.auto_clips[key]

        try:
            if r.kind == "view":
                return await self._read(stream, render)
            return await asyncio.get_running_loop().run_in_executor(self.panel_pool, render)
        except ImportError as e:                            # ballasim.viz needs the viz extra (cv2, matplotlib)
            raise StreamError(422, "BS-UI-SOURCE", f"PNG panels need ballasim[viz]: {e}") from None
        except OverflowError:                               # a window or its history beyond int64
            raise StreamError(400, "BS-UI-REQUEST", "the window (or its 5τ history) leaves the int64 range") from None
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise _source_error(e) from None

    # -- rate ----------------------------------------------------------------------------------------------------

    async def rate(self, stream: Stream, t0_ns: int, t1_ns: int, bin_ms: int) -> bytes:
        """``GET /streams/{sid}/rate``: BSRT bins [t0 + i bin, t0 + (i + 1) bin) covering [t0_ns, t1_ns)."""
        if not 1 <= bin_ms <= 4294:
            raise StreamError(400, "BS-UI-REQUEST", "bin_ms: 1..4294")
        if t1_ns < t0_ns:
            raise StreamError(400, "BS-UI-REQUEST", "need t0_ns <= t1_ns")
        bin_ns = bin_ms * RATE_BIN_NS
        n_bins = -(-(t1_ns - t0_ns) // bin_ns)
        if n_bins > MAX_RATE_BINS:
            raise StreamError(400, "BS-UI-REQUEST", f"at most {MAX_RATE_BINS} bins: raise bin_ms")
        r = stream.reader
        edges = t0_ns + bin_ns * np.arange(n_bins + 1, dtype=np.int64)
        aligned = (t0_ns - r.t_offset_ns) % RATE_BIN_NS == 0
        if aligned and r.kind == "c4":       # totals from /ms_to_idx, ON counts from the built blocks
            index = await self._read(stream, r.grid_totals, edges)
            total, on = np.diff(index).astype(np.uint32), self._grid_on(stream, t0_ns, n_bins, bin_ms)
        elif aligned and n_bins <= RATE_SEARCH_MAX:     # one search per bin edge
            index = await self._read(stream, lambda: np.array([r.search(int(e)) for e in edges], np.int64))
            total, on = np.diff(index).astype(np.uint32), self._grid_on(stream, t0_ns, n_bins, bin_ms)
        else:                                # off the grid, or many bins: count the span's events, ON included
            i0, i1 = await self._read(stream, lambda: (r.search(int(edges[0])), r.search(int(edges[-1]))))
            total, on = np.zeros(n_bins, np.uint32), np.zeros(n_bins, np.uint32)
            for a in range(i0, i1, RATE_CHUNK_EVENTS):          # bounded reads; other reads interleave
                cols = await self._read(stream, r.read, a, min(a + RATE_CHUNK_EVENTS, i1), (), False)
                part_total, part_on = await self._cpp(wire.rate_counts, cols.t_ns, cols.p, t0_ns, bin_ns, n_bins)
                total += part_total
                on += part_on
        flags = 1 if n_bins == 0 or bool(np.all(on != NO_COUNT)) else 0
        return BSRT_HEADER.pack(BSRT_MAGIC, WIRE_VERSION, flags, 0, n_bins, t0_ns, bin_ns, 0) + \
            np.ascontiguousarray(total, "<u4").tobytes() + np.ascontiguousarray(on, "<u4").tobytes()

    @staticmethod
    def _grid_on(stream: Stream, t0_ns: int, n_bins: int, bin_ms: int) -> np.ndarray:
        """ON counts of bins on the 1 ms grid from the built blocks' heads; NO_COUNT where a block that holds
        events in the bin is not built."""
        g0 = stream.reader.t_offset_ns
        f0, nf = (t0_ns - g0) // RATE_BIN_NS, n_bins * bin_ms
        if nf > MAX_GRID_BINS:
            return np.full(n_bins, NO_COUNT, np.uint32)
        on, ok = np.zeros(nf, np.uint32), np.ones(nf, bool)
        if nf:
            k_lo = max((g0 + f0 * RATE_BIN_NS) >> BLOCK_SHIFT, stream.k_first)
            k_hi = min(((g0 + (f0 + nf) * RATE_BIN_NS - 1) >> BLOCK_SHIFT) + 1, stream.k_end)
            for k in range(k_lo, k_hi):
                b0, b1 = stream.rate_bins(k)
                lo, hi = max(b0, f0), min(b1, f0 + nf)
                if lo >= hi:
                    continue
                head = stream.heads.get(k)
                if head is None:
                    ok[lo - f0:hi - f0] = False
                else:
                    on[lo - f0:hi - f0] += head.on[lo - head.rate_bin0:hi - head.rate_bin0]
        on = on.reshape(n_bins, bin_ms).sum(axis=1, dtype=np.uint64).astype(np.uint32)
        return np.where(ok.reshape(n_bins, bin_ms).all(axis=1), on, NO_COUNT).astype(np.uint32)


def bsev_message(arrays, flags: int, t_base_ns: int, i0: int, n_block: int, width: int, height: int, cov_lo: int,
                 cov_hi: int, deflate: bool) -> bytes:
    """One BSEV message (§3.5) of ``arrays`` = [(column name, values)], in column id order, bodies 8-aligned."""
    n_events = len(arrays[0][1]) if arrays else 0
    header_bytes = BSEV_HEADER.size + BSEV_COLUMN.size * len(arrays)
    table, bodies, offset = [], [], header_bytes
    for name, values in arrays:
        c = COLUMNS[name]
        data = np.ascontiguousarray(values, c.dtype).tobytes()
        table.append(BSEV_COLUMN.pack(c.id, c.code, c.comps, offset, len(data), 0))
        pad = -len(data) % 8
        bodies += [data, b"\0" * pad]
        offset += len(data) + pad
    if bodies:
        bodies.pop()                         # no padding after the last body
    rest = b"".join(table + bodies)
    if deflate:
        z = zlib.compressobj(1, zlib.DEFLATED, -15)
        rest = z.compress(rest) + z.flush()
        flags |= FLAG_DEFLATE
    header = BSEV_HEADER.pack(BSEV_MAGIC, WIRE_VERSION, flags, 0, n_events, t_base_ns, i0, n_block, width, height,
                              len(arrays), header_bytes, cov_lo, cov_hi)
    return header + rest


def bstk_message(kind: str, t0_ns: int, t_ns, columns=()) -> bytes:
    """One BSTK message (§3.5): the 32-byte header, a 16-byte column-table entry per extra column (as BSEV's), the
    ``t_ns`` i64[n] (sensor clock), then the extra columns' bodies, each 8-aligned. ``columns``: [(name, values)] with
    names from :data:`TRACK_COLUMNS` (UI-7's ``/tracks`` serves these; UI-2's decoder reads them)."""
    t = np.ascontiguousarray(t_ns, "<i8")
    n, table, bodies = len(t), [], [t.tobytes()]
    offset = BSTK_HEADER.size + BSEV_COLUMN.size * len(columns) + t.nbytes
    for name, values in columns:
        cid, code, comps, dtype = TRACK_COLUMNS[name]
        data = np.ascontiguousarray(values, dtype).tobytes()
        if len(data) != n * comps * dtype.itemsize:
            raise ValueError(f"track column {name}: {len(data)} bytes for {n} rows")
        pad = -offset % 8
        bodies.append(b"\0" * pad)
        offset += pad
        table.append(BSEV_COLUMN.pack(cid, code, comps, offset, len(data), 0))
        bodies.append(data)
        offset += len(data)
    return BSTK_HEADER.pack(BSTK_MAGIC, WIRE_VERSION, TRACK_KINDS[kind], n, len(columns), t0_ns) + \
        b"".join(table) + b"".join(bodies)


__all__ = ["BLOCK_NS", "COLUMNS", "IMAGE_KINDS", "MAX_SLICE_EVENTS", "StreamError", "Streams", "bsev_message",
           "bstk_message", "decode_rid", "encode_rid", "locate_source", "resolve_path", "stream_id"]
