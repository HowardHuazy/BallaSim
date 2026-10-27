"""Event-stream loaders: EVIMO2 NPZ sequences, BallaSim/DSEC HDF5 files, NumPy arrays.

``open_events`` picks the loader from the path. All loaders return an
:class:`~ballasim.viz.events.EventSource` with int64-ns time and True = ON.
"""

from __future__ import annotations

import contextlib
import functools
import json
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np

from .events import NS_PER_MS, NS_PER_S, ArraySource, Events, EventSource, GtFrame, seconds_to_ns

POLARITY_MODES = ("auto", "normal", "inverted")
#: The EVIMO2 release facts (dataset.evimo2.*): the viewer's and the stats' only copy (D16b).
RELEASE_PROFILE = "evimo2/release@v0"


@functools.cache
def evimo2_release() -> dict:
    """The facts of the EVIMO2 v2 release, from the ``evimo2/release@v0`` profile (D16b): ``mains_hz``,
    ``strobe_hz``, ``gt_lookup_tol_s`` and ``samsung_on_polarity``."""
    from ballasim.config import read_profile

    values = read_profile(RELEASE_PROFILE)["values"]
    return {key.removeprefix("dataset.evimo2."): entry["value"] for key, entry in values.items()}


def open_events(path: str | Path, *, label: str | None = None, polarity: str = "auto") -> EventSource:
    """Open an event stream.

    ``path`` is an EVIMO2 sequence directory (``dataset_events_t.npy`` inside), a
    BallaSim/DSEC ``.h5``/``.hdf5`` file, a ``.npz``/``.npy`` file or a directory
    of ``.npy`` columns (see :func:`load_arrays`). ``polarity`` is ``"auto"``
    (the format's convention: EVIMO2 Samsung recordings are inverted),
    ``"normal"`` (raw p = 1 is ON) or ``"inverted"``.
    """
    if polarity not in POLARITY_MODES:
        raise ValueError(f"polarity must be one of {POLARITY_MODES}, got {polarity!r}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir() and (path / Evimo2Sequence.EVENTS_T).exists():
        source: EventSource = Evimo2Sequence(path, label=label)
    elif path.suffix.lower() in (".h5", ".hdf5"):
        source = H5Events(path, label=label)
    elif path.is_dir() or path.suffix.lower() in (".npz", ".npy"):
        source = load_arrays(path, label=label)
    else:
        raise ValueError(f"unrecognised event source: {path}")
    if polarity != "auto":
        source = source.with_polarity(polarity == "inverted")
    return source


# EVIMO2 ---------------------------------------------------------------------------


def _is_samsung(directory: Path) -> bool:
    paths = (directory.absolute(), directory.resolve())
    return any("samsung" in part.lower() for path in paths for part in path.parts)


def _f32_seconds_to_ns(t: np.ndarray) -> np.ndarray:
    # float32 -> float64 is exact; the product is exact to < 1e-5 ns for t < 2^23 s.
    return np.rint(np.asarray(t, np.float64) * NS_PER_S).astype(np.int64)


def _f32_scalar_to_ns(t: np.float32) -> int:
    return int(np.rint(np.float64(t) * NS_PER_S))


class Evimo2Sequence(EventSource):
    """One EVIMO2 v2 NPZ sequence directory, memory-mapped.

    Format pitfalls handled here (docs/research R6, G3):

    - event times are float32 seconds since bag start; they are upcast to float64
      before conversion to ns, and searches by time are exact in ns;
    - intrinsics come from ``meta['meta']``: ``dataset_info.npz['D']`` holds
      k1..k4 and silently drops the radtan p1, p2;
    - GT archive keys are ``mask_%010d`` / ``depth_%010d`` of *frame id minus the
      first frame id*, not the raw id;
    - Samsung polarity is inverted (ON is p = 0: the release profile's
      ``samsung_on_polarity``), so Samsung sequences are opened with
      ``invert_polarity`` set. The metadata does not name the camera, so a
      sequence counts as Samsung when any folder of its path, as given or with
      symlinks resolved, contains "samsung" (the release keeps it in ``samsung_mono``).

    GT frames are looked up within the release profile's ``gt_lookup_tol_s`` (10 ms):
    GT ticks are 1/60 s apart, so a window centre is at most 8.3 ms from a tick
    unless the tick is missing (Vicon gap).
    """

    EVENTS_T = "dataset_events_t.npy"

    def __init__(self, directory: str | Path, label: str | None = None) -> None:
        directory = Path(directory)
        self.directory = directory
        self._t = np.load(directory / self.EVENTS_T, mmap_mode="r")
        self._xy = np.load(directory / "dataset_events_xy.npy", mmap_mode="r")
        self._p = np.load(directory / "dataset_events_p.npy", mmap_mode="r")
        if self._t.dtype != np.float32 or self._xy.shape != (len(self._t), 2) or len(self._p) != len(self._t):
            raise ValueError(f"{directory}: unexpected EVIMO2 event array layout")
        with np.load(directory / "dataset_info.npz", allow_pickle=True) as info:  # meta is a pickled dict
            meta = info["meta"].item()
        self.camera: dict = dict(meta["meta"])
        # Release layout: <camera>/<group>/<split>/<sequence>.
        parents = directory.absolute().parents
        camera_folder = parents[2].name if len(parents) > 2 else ""
        super().__init__(
            label or f"{camera_folder}/{directory.name}".lstrip("/"),
            int(self.camera["res_x"]),
            int(self.camera["res_y"]),
        )
        release = evimo2_release()
        self.invert_polarity = _is_samsung(directory) and release["samsung_on_polarity"] == 0
        self.gt_tolerance_ns = round(release["gt_lookup_tol_s"] * NS_PER_S)

        if len(self._t):
            self._ns_first = _f32_scalar_to_ns(self._t[0])
            self._ns_last = _f32_scalar_to_ns(self._t[-1])
        frames = meta["frames"]
        first_id = frames[0]["id"] if frames else 0
        gt = [(seconds_to_ns(f["ts"]), f["id"] - first_id) for f in frames if "gt_frame" in f]
        self._gt_t_ns = np.array([g[0] for g in gt], np.int64)
        self._gt_key = np.array([g[1] for g in gt], np.int64)
        self._archives: dict[str, np.lib.npyio.NpzFile] = {}
        self._last_gt: dict[str, tuple[str, np.ndarray]] = {}
        self._gt_lock = threading.Lock()  # renderers call from several threads; zip reads share a handle

    def __len__(self) -> int:
        return len(self._t)

    @property
    def gt_kinds(self) -> tuple[str, ...]:
        if not len(self._gt_t_ns):
            return ()
        return tuple(k for k in ("mask", "depth") if (self.directory / f"dataset_{k}.npz").exists())

    def _search(self, t_ns: int) -> int:
        n = len(self._t)
        if n == 0 or t_ns <= self._ns_first:
            return 0
        if t_ns > self._ns_last:
            return n
        # Smallest float32 f with ns(f) >= t_ns; then a same-dtype searchsorted on
        # the memory map (a float64 query would make NumPy copy the whole array).
        f = np.float32(t_ns / NS_PER_S)
        up, down = np.float32(np.inf), np.float32(-np.inf)
        while _f32_scalar_to_ns(f) < t_ns:
            f = np.nextafter(f, up)
        while _f32_scalar_to_ns(prev := np.nextafter(f, down)) >= t_ns:
            f = prev
        return int(np.searchsorted(self._t, f, side="left"))

    def _read(self, i0: int, i1: int, extras: Sequence[str]) -> Events:
        xy = np.asarray(self._xy[i0:i1])
        return Events(
            x=np.ascontiguousarray(xy[:, 0]),
            y=np.ascontiguousarray(xy[:, 1]),
            p=np.asarray(self._p[i0:i1]) > 0,
            t_ns=_f32_seconds_to_ns(self._t[i0:i1]),
        )

    def _gt_frame(self, kind: str, t_ns: int) -> GtFrame | None:
        times = self._gt_t_ns
        j = int(np.searchsorted(times, t_ns))
        candidates = [k for k in (j - 1, j) if 0 <= k < len(times)]
        k = min(candidates, key=lambda c: abs(int(times[c]) - t_ns))
        if abs(int(times[k]) - t_ns) > self.gt_tolerance_ns:
            return None
        key = f"{kind}_{int(self._gt_key[k]):010d}"
        with self._gt_lock:
            cached = self._last_gt.get(kind)
            if cached is not None and cached[0] == key:
                return GtFrame(int(times[k]), cached[1])
            archive = self._archives.get(kind)
            if archive is None:
                archive = self._archives[kind] = np.load(self.directory / f"dataset_{kind}.npz")
            if key not in archive.files:
                return None
            data = archive[key]
            if kind == "depth":
                # uint16 mm, 0 = no geometry -> float32 m with NaN.
                data = np.where(data > 0, data.astype(np.float32) / 1000.0, np.float32(np.nan))
            self._last_gt[kind] = (key, data)
        return GtFrame(int(times[k]), data)


# HDF5 -----------------------------------------------------------------------------


def _register_hdf5_filters() -> None:
    with contextlib.suppress(ImportError):
        import hdf5plugin  # noqa: F401  (importing registers Blosc/Zstd with h5py)


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return _as_str(value.item())
    return str(value)


class H5Events(EventSource):
    """A DSEC-layout event file as BallaSim writes it (DESIGN §9.3, S1 §2.2-2.3).

    Reads ``/events/{x,y,p,t}`` with ``t`` in int µs relative to ``/t_offset``,
    the optional ``/events/t_ns_rem`` (exact ns = 1000·(t_offset + t) + t_ns_rem),
    ``/ms_to_idx`` for random access (rebuilt in memory if absent), and every
    per-event ``/gt/*`` column as an extra field. The resolution comes from the
    file attribute ``resolution`` = [W, H] (contract C4), else ``width``/``height``,
    else ``/meta/calibration.json`` ``"resolution": [W, H]``, else the extent of the data.

    A BallaSim run's ``events/<camera>.h5`` also provides GT frames when the run has
    dense frames (``python -m ballasim dense``, ``dense/<camera>.h5`` next to
    ``events/``): ``mask`` (the instance image), ``depth`` and ``irradiance`` (summed
    over light groups), the frame nearest in time within one frame spacing.
    """

    #: GT kind -> dense-frame dataset.
    DENSE_KINDS: ClassVar[dict[str, str]] = {"mask": "instance", "depth": "depth", "irradiance": "irradiance"}

    def __init__(self, path: str | Path, label: str | None = None) -> None:
        _register_hdf5_filters()
        self.path = Path(path)
        self._file = h5py.File(self.path, "r", rdcc_nbytes=64 << 20)
        try:
            events = self._file["events"]
            self._x, self._y, self._p, self._t_us = (events[k] for k in ("x", "y", "p", "t"))
            self._rem = events.get("t_ns_rem")
            self._n = int(self._t_us.shape[0])
            self._t_offset_us = int(self._file["t_offset"][()]) if "t_offset" in self._file else 0
            if "ms_to_idx" in self._file:
                self._ms_to_idx = np.asarray(self._file["ms_to_idx"][()], np.int64)
            else:
                self._ms_to_idx = self._build_ms_index()
            gt = self._file.get("gt")
            self._gt = {
                name: ds
                for name, ds in (gt.items() if gt is not None else ())
                if isinstance(ds, h5py.Dataset) and ds.shape == (self._n,)
            }
            width, height = self._resolution()
            default_label = _as_str(self._file.attrs["camera"]) if "camera" in self._file.attrs else self.path.stem
            self._dense = self._open_dense(width, height)
        except Exception:
            self._file.close()
            raise
        super().__init__(label or default_label, width, height)
        self._dense_lock = threading.Lock()  # renderers call from several threads
        # kind -> (frame index, data) of the last dense frame read: a frame serves every lookup within its spacing
        # (GtRegion reads its mask once per 1 ms slot), and a read decompresses the whole frame
        self._last_gt: dict[str, tuple[int, np.ndarray]] = {}

    def _open_dense(self, width: int, height: int) -> h5py.File | None:
        path = self.path.parent.parent / "dense" / self.path.name
        if not path.is_file():
            return None
        dense = h5py.File(path, "r")
        resolution = [int(v) for v in np.asarray(dense.attrs.get("resolution", [0, 0])).ravel()[:2]]
        if not _as_str(dense.attrs.get("format", "")).startswith("ballasim.dense/") or resolution != [width, height]:
            dense.close()
            raise ValueError(f"{path}: not BallaSim dense frames of a {width}x{height} camera")
        self._dense_t = np.asarray(dense["t_ns"][()], np.int64)
        self._dense_tolerance = max(int(dense.attrs["dt_ns"]), 1)
        return dense

    def __len__(self) -> int:
        return self._n

    @property
    def gt_kinds(self) -> tuple[str, ...]:
        if self._dense is None or not len(self._dense_t):
            return ()
        return tuple(kind for kind, name in self.DENSE_KINDS.items() if name in self._dense)

    def _gt_frame(self, kind: str, t_ns: int) -> GtFrame | None:
        times = self._dense_t
        j = int(np.searchsorted(times, t_ns))
        k = min((c for c in (j - 1, j) if 0 <= c < len(times)), key=lambda c: abs(int(times[c]) - t_ns))
        if abs(int(times[k]) - t_ns) > self._dense_tolerance:
            return None
        with self._dense_lock:
            cached = self._last_gt.get(kind)
            if cached is not None and cached[0] == k:
                return GtFrame(int(times[k]), cached[1])
            data = self._dense[self.DENSE_KINDS[kind]][k]
            if kind == "irradiance":
                data = data.sum(axis=-1)
            data.flags.writeable = False  # shared by every lookup of this frame
            self._last_gt[kind] = (k, data)
        return GtFrame(int(times[k]), data)

    @property
    def extra_fields(self) -> tuple[str, ...]:
        return tuple(self._gt)

    def _column(self, dataset: h5py.Dataset, i0: int, i1: int) -> np.ndarray:
        try:
            return dataset[i0:i1]
        except OSError as err:
            raise RuntimeError(
                f"{self.path}: cannot read {dataset.name} ({err}); BallaSim files use Blosc, "
                "which needs the 'hdf5plugin' package"
            ) from err

    def _relative_ns(self, i0: int, i1: int) -> np.ndarray:
        ns = self._column(self._t_us, i0, i1).astype(np.int64) * 1000
        if self._rem is not None:
            ns += self._column(self._rem, i0, i1)
        return ns

    def _build_ms_index(self) -> np.ndarray:
        if self._n == 0:
            return np.zeros(0, np.int64)
        t_us = self._column(self._t_us, 0, self._n).astype(np.int64)
        n_ms = max(int(t_us[-1]) // 1000 + 1, 0)
        return np.searchsorted(t_us, np.arange(n_ms, dtype=np.int64) * 1000, side="left").astype(np.int64)

    def _resolution(self) -> tuple[int, int]:
        attrs = self._file.attrs
        if "resolution" in attrs:  # contract C4: [W, H]
            width, height = (int(v) for v in np.asarray(attrs["resolution"]).ravel()[:2])
            return width, height
        if "width" in attrs and "height" in attrs:
            return int(attrs["width"]), int(attrs["height"])
        calibration = self._file.get("meta/calibration.json")
        if calibration is not None:
            resolution = json.loads(_as_str(calibration[()])).get("resolution")
            if resolution is not None:
                return int(resolution[0]), int(resolution[1])
        width = height = 0
        for start in range(0, self._n, 1 << 22):
            stop = min(start + (1 << 22), self._n)
            width = max(width, int(self._column(self._x, start, stop).max()) + 1)
            height = max(height, int(self._column(self._y, start, stop).max()) + 1)
        return max(width, 1), max(height, 1)

    def _search(self, t_ns: int) -> int:
        rel_ns = t_ns - 1000 * self._t_offset_us
        ms = rel_ns // NS_PER_MS
        index, m = self._ms_to_idx, len(self._ms_to_idx)
        # ms_to_idx[k] is the first event with t_us >= 1000 k, so the answer lies
        # in [ms_to_idx[ms], ms_to_idx[ms + 1]].
        if m == 0:
            lo, hi = 0, self._n
        elif ms < 0:
            lo, hi = 0, int(index[0])
        elif ms >= m:
            lo, hi = int(index[-1]), self._n
        else:
            lo, hi = int(index[ms]), int(index[ms + 1]) if ms + 1 < m else self._n
        return lo + int(np.searchsorted(self._relative_ns(lo, hi), rel_ns, side="left"))

    def _read(self, i0: int, i1: int, extras: Sequence[str]) -> Events:
        return Events(
            x=self._column(self._x, i0, i1),
            y=self._column(self._y, i0, i1),
            p=self._column(self._p, i0, i1) > 0,
            t_ns=self._relative_ns(i0, i1) + 1000 * self._t_offset_us,
            extras={name: self._column(self._gt[name], i0, i1) for name in extras},
        )


# Plain NumPy ----------------------------------------------------------------------

_STANDARD_KEYS = {"x", "y", "p", "t", "t_ns", "t_us", "width", "height", "resolution"}


def _time_to_ns(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    if "t_ns" in columns:
        return np.asarray(columns["t_ns"]).astype(np.int64, copy=False)
    if "t_us" in columns:
        return np.asarray(columns["t_us"]).astype(np.int64) * 1000
    if "t" in columns:
        t = np.asarray(columns["t"])
        if np.issubdtype(t.dtype, np.floating):
            return np.rint(t.astype(np.float64) * NS_PER_S).astype(np.int64)
        return t.astype(np.int64) * 1000  # integer t is DSEC µs
    raise KeyError("no time column: expected 't_ns' (int ns), 't_us' (int µs) or 't' (float s or int µs)")


def arrays_to_source(columns: Mapping[str, np.ndarray], label: str = "events") -> ArraySource:
    """Build a source from columns ``x``, ``y``, ``p`` (1 = ON) and a time column.

    Time is ``t_ns`` (int ns), ``t_us`` (int µs) or ``t`` (float seconds, or
    int µs as in DSEC). The resolution is taken from ``width``/``height`` or
    ``resolution`` = (W, H) if present. Any other column of matching length is
    exposed as an extra per-event field.
    """
    missing = {"x", "y", "p"} - set(columns)
    if missing:
        raise KeyError(f"missing event column(s) {sorted(missing)}")
    t_ns = _time_to_ns(columns)
    width = height = None
    if "resolution" in columns:
        width, height = (int(v) for v in np.asarray(columns["resolution"]).ravel()[:2])
    if "width" in columns and "height" in columns:
        width, height = int(np.asarray(columns["width"])), int(np.asarray(columns["height"]))
    extras = {
        k: np.asarray(v)
        for k, v in columns.items()
        if k not in _STANDARD_KEYS and np.ndim(v) == 1 and len(v) == len(t_ns)
    }
    return ArraySource(
        np.asarray(columns["x"]),
        np.asarray(columns["y"]),
        np.asarray(columns["p"]),
        t_ns,
        label=label,
        width=width,
        height=height,
        extras=extras,
    )


def load_arrays(path: str | Path, label: str | None = None) -> ArraySource:
    """Load events from a ``.npz``, a ``.npy`` holding a pickled dict, or a
    directory of ``<column>.npy`` files (memory-mapped); see :func:`arrays_to_source`.

    A pickled ``.npy`` dict can run code when loaded; only open your own files.
    """
    path = Path(path)
    if path.is_dir():
        columns = {f.stem: np.load(f, mmap_mode="r") for f in sorted(path.glob("*.npy"))}
    elif path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            columns = {k: archive[k] for k in archive.files}
    else:
        obj = np.load(path, allow_pickle=True)
        if obj.dtype != object or obj.shape != ():
            raise ValueError(f"{path}: expected a .npy file holding a dict of event columns")
        columns = dict(obj.item())
    return arrays_to_source(columns, label=label or path.stem)
