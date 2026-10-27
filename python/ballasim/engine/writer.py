"""Event output (contract C4): one DSEC-compatible HDF5 file per camera.

Layout (``ballasim.viz.io.H5Events`` reads it; DSEC and TUM-VIE loaders read ``/events`` unchanged):

- ``/events/{x u16, y u16, p u8, t i64, t_ns_rem u16}``: ``t`` in µs relative to ``/t_offset`` on the
  sensor clock (t_reported), exact ns = 1000 (t_offset + t) + t_ns_rem;
- ``/t_offset`` (i64 µs), ``/ms_to_idx`` (u64; entry k is the first event with t >= 1000 k);
- ``/gt/{instance u32, semantic u16, depth f32, flow_x f32, flow_y f32, label u8, t_cross_ns i64,
  flags u16}`` (with ``gt``): label in S1 classes, flags in S1 bits;
- attributes ``resolution = [W, H]``, ``camera``, ``format``.

Every per-event dataset is chunked by 2^17 events with Blosc (zstd, level 1, bitshuffle) and a
Fletcher-32 checksum (DESIGN §9.3). Events must be appended in output order. The file is written as
``.<name>.partial`` next to the target and renamed on :meth:`EventWriter.close`, so a crashed run
never leaves a truncated file under the final name.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

CHUNK = 1 << 17
FORMAT = "ballasim.events/1"

EVENT_COLUMNS = {"x": np.uint16, "y": np.uint16, "p": np.uint8, "t": np.int64, "t_ns_rem": np.uint16}
GT_COLUMNS = {"instance": np.uint32, "semantic": np.uint16, "depth": np.float32, "flow_x": np.float32,
              "flow_y": np.float32, "label": np.uint8, "t_cross_ns": np.int64, "flags": np.uint16}
_DESCRIPTIONS = {
    "events/x": ("px", "raw (distorted) column"), "events/y": ("px", "raw (distorted) row"),
    "events/p": ("", "polarity, 1 = ON"), "events/t": ("us", "t_reported on the sensor clock, relative to /t_offset"),
    "events/t_ns_rem": ("ns", "t_reported_ns - 1000 floor(t_reported_ns / 1000)"),
    "gt/instance": ("", "centre-ray instance at t_cross (0 = no geometry)"),
    "gt/semantic": ("", "semantic class of the centre-ray hit"),
    "gt/depth": ("m", "camera-frame z of the centre-ray hit at t_cross (NaN = none)"),
    "gt/flow_x": ("px/s", "instantaneous motion field of the centre-ray point, raw coordinates"),
    "gt/flow_y": ("px/s", "instantaneous motion field of the centre-ray point, raw coordinates"),
    "gt/label": ("", "S1 class: 0 SIGNAL, 1 MIXED, 2 NOISE, 3 LEAK, 4 HOT"),
    "gt/t_cross_ns": ("ns", "physical firing time on the true clock"),
    "gt/flags": ("", "S1 bits: 0 MOVING, 5 HOT_PIXEL, 9 QUARANTINED, 10 SOLVER_FALLBACK, 12 FIRST_AFTER_INIT"),
}


def _compression() -> dict:
    return dict(hdf5plugin.Blosc(cname="zstd", clevel=1, shuffle=hdf5plugin.Blosc.BITSHUFFLE))


class EventWriter:
    """Appends sorted events of one camera to its C4 file."""

    def __init__(self, path: str | os.PathLike, *, camera: str, width: int, height: int, t_offset_us: int,
                 gt: bool):
        self.path = Path(path)
        self.tmp = self.path.with_name(f".{self.path.name}.partial")
        self.t_offset_us = int(t_offset_us)
        self.columns = {f"events/{k}": v for k, v in EVENT_COLUMNS.items()}
        if gt:
            self.columns.update({f"gt/{k}": v for k, v in GT_COLUMNS.items()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.tmp, "w")
        self.file.attrs["format"] = FORMAT
        self.file.attrs["camera"] = camera
        self.file.attrs["resolution"] = np.array([width, height], np.int64)
        comp = _compression()
        for name, dtype in self.columns.items():
            ds = self.file.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(CHUNK,),
                                          fletcher32=True, **comp)
            units, description = _DESCRIPTIONS[name]
            ds.attrs["units"] = units
            ds.attrs["description"] = description
        self.pending: dict[str, list[np.ndarray]] = {name: [] for name in self.columns}
        self.n_pending = 0
        self.n_written = 0
        self.n_total = 0
        self.last_t_us = None
        self.ms_to_idx: list[np.ndarray] = []
        self.ms_next = 0

    def append(self, x: np.ndarray, y: np.ndarray, p: np.ndarray, t_reported_ns: np.ndarray,
               gt: dict[str, np.ndarray] | None = None) -> None:
        """Appends events in output order (t_reported non-decreasing across calls)."""
        t_rep = np.asarray(t_reported_ns, np.int64)
        if len(t_rep) == 0:
            return
        t_us = t_rep // 1000
        rel = t_us - self.t_offset_us
        if np.any(np.diff(rel) < 0) or (self.last_t_us is not None and rel[0] < self.last_t_us):
            raise ValueError("EventWriter.append: events must arrive sorted by t_reported")
        if rel[0] < 0:
            raise ValueError("EventWriter.append: event before t_offset")
        cols = {"events/x": x, "events/y": y, "events/p": p, "events/t": rel, "events/t_ns_rem": t_rep - 1000 * t_us}
        if "gt/instance" in self.columns:
            if gt is None:
                raise ValueError("EventWriter.append: this file has /gt columns")
            cols.update({f"gt/{k}": gt[k] for k in GT_COLUMNS})
        for name, dtype in self.columns.items():
            self.pending[name].append(np.asarray(cols[name]).astype(dtype, copy=False))
        # ms_to_idx entries whose boundary 1000 k this block reaches are final now.
        k_hi = int(rel[-1]) // 1000
        if k_hi >= self.ms_next:
            ks = np.arange(self.ms_next, k_hi + 1, dtype=np.int64)
            self.ms_to_idx.append(self.n_total + np.searchsorted(rel, 1000 * ks, side="left"))
            self.ms_next = k_hi + 1
        self.last_t_us = int(rel[-1])
        self.n_total += len(rel)
        self.n_pending += len(rel)
        if self.n_pending >= CHUNK:
            self._flush(self.n_pending // CHUNK * CHUNK)

    def _flush(self, count: int) -> None:
        for name in self.columns:
            data = np.concatenate(self.pending[name])
            ds = self.file[name]
            ds.resize((self.n_written + count,))
            ds[self.n_written:] = data[:count]
            self.pending[name] = [data[count:]]
        self.n_written += count
        self.n_pending -= count

    def close(self, t_end_reported_ns: int) -> Path:
        """Writes the remainder, ``/ms_to_idx`` up to t_end_reported_ns and ``/t_offset``; renames the file."""
        if self.n_pending:
            self._flush(self.n_pending)
        n_ms = max(self.ms_next, (int(t_end_reported_ns) // 1000 - self.t_offset_us) // 1000 + 1)
        tail = np.full(n_ms - self.ms_next, self.n_total, np.int64)
        index = np.concatenate(self.ms_to_idx + [tail]).astype(np.uint64)
        self.file.create_dataset("ms_to_idx", data=index)
        self.file.create_dataset("t_offset", data=np.int64(self.t_offset_us))
        self.file.close()
        os.replace(self.tmp, self.path)
        return self.path

    def abort(self) -> None:
        """Closes and removes the partial file."""
        try:
            self.file.close()
        finally:
            self.tmp.unlink(missing_ok=True)
