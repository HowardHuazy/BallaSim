"""Frames files ``ballasim.frames/1`` (docs/design/phase3/frames.md §3.6): one HDF5 file per camera,
``<run>/frames/<camera>.h5``, written as ``.<camera>.h5.partial`` and renamed.

Datasets (n frames, N sub-renders, H x W the ROI, G light groups, n_cal calibration frames):

- per frame, i64 (n): ``t_soe_ns``, ``t_eoe_ns``, ``t_mid_ns`` on the camera's sensor clock, ``t_true_soe_ns``,
  ``t_true_eoe_ns``, ``t_true_mid_ns`` on the true clock, ``t_req_true_mid_ns``; i32 (n) ``k_ref``, ``k_ref_mask``;
  f32 (n) ``probe_p999``, ``probe_max``, ``e_fact_p99``;
- sub-renders: ``sub_offsets`` i64 (n + 1, from 0, non-decreasing; N = its last value), ``sub_t_true_soe_ns``,
  ``sub_t_true_eoe_ns`` i64 (N), ``sub_blender_c`` f64 (N, NaN in reference mode);
- waveform means: ``m_mean`` f64 (n, H, G), ``m_mean_sub`` f64 (N, H, G);
- images (optional, each per ``frames.outputs`` and the mode): ``mono``, ``reference``, ``correction``,
  ``mono_half_diff``, ``diffcol_mono`` f32 (n, H, W); ``rgb`` f32 (n, H, W, 3); ``groups`` f32 (N, H, W, G);
  ``calib_frames`` i32 (n_cal); ``calib_b`` f32 (n_cal, H, W, G); ``aps_dn`` u16 (n, H, W).

Attributes: ``format`` (written here), ``camera``, ``resolution`` [W, H] of the full image, ``roi_px`` [x0, y0, x1, y1],
``mode`` and ``light_groups`` (names) are required; the others of §3.6 are free. Dicts, lists of strings and the JSON
attributes of §3.6 (``route_bound``, ``dropped_frames``, ``s_g``, ``fitted_poses``, ``blender``, ``aps``,
``calib_report``, ``warnings``, ``light_groups``) are stored as JSON text; numeric lists as arrays. The reader
decodes both.

Images are chunked one frame (or sub-render) per chunk with Blosc (zstd, level 1, bitshuffle); per-frame vectors
are stored plain. Values are converted to the dataset's type only where NumPy casts safely, or between integer types
when every value fits (an int64 time never comes from a float). ``rewrite_frames`` replaces datasets and attributes of
a written file (``frames aps``: ``aps_dn``, ``aps``, ``warnings``, the digests), copying the rest.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

FORMAT = "ballasim.frames/1"

# name -> (dtype, shape in the symbols n, n+1, N, H, W, G, n_cal and literal ints)
DATASETS: dict[str, tuple[type, tuple]] = {
    **{k: (np.int64, ("n",)) for k in ("t_soe_ns", "t_eoe_ns", "t_mid_ns", "t_true_soe_ns", "t_true_eoe_ns",
                                       "t_true_mid_ns", "t_req_true_mid_ns")},
    "sub_offsets": (np.int64, ("n+1",)),
    "sub_t_true_soe_ns": (np.int64, ("N",)),
    "sub_t_true_eoe_ns": (np.int64, ("N",)),
    "sub_blender_c": (np.float64, ("N",)),
    "k_ref": (np.int32, ("n",)),
    "k_ref_mask": (np.int32, ("n",)),
    "probe_p999": (np.float32, ("n",)),
    "probe_max": (np.float32, ("n",)),
    "e_fact_p99": (np.float32, ("n",)),
    "m_mean": (np.float64, ("n", "H", "G")),
    "m_mean_sub": (np.float64, ("N", "H", "G")),
    **{k: (np.float32, ("n", "H", "W")) for k in ("mono", "reference", "correction", "mono_half_diff",
                                                  "diffcol_mono")},
    "rgb": (np.float32, ("n", "H", "W", 3)),
    "groups": (np.float32, ("N", "H", "W", "G")),
    "calib_frames": (np.int32, ("n_cal",)),
    "calib_b": (np.float32, ("n_cal", "H", "W", "G")),
    "aps_dn": (np.uint16, ("n", "H", "W")),
}
REQUIRED = ("t_soe_ns", "t_eoe_ns", "t_mid_ns", "t_true_soe_ns", "t_true_eoe_ns", "t_true_mid_ns", "t_req_true_mid_ns",
            "sub_offsets", "sub_t_true_soe_ns", "sub_t_true_eoe_ns", "sub_blender_c", "k_ref", "k_ref_mask",
            "probe_p999", "probe_max", "e_fact_p99", "m_mean", "m_mean_sub")
REQUIRED_ATTRS = ("camera", "resolution", "roi_px", "mode", "light_groups")
JSON_ATTRS = frozenset({"route_bound", "dropped_frames", "s_g", "fitted_poses", "blender", "aps", "calib_report",
                        "warnings", "light_groups"})


def _encode_attr(name: str, value):
    if name in JSON_ATTRS or isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value)
        if arr.dtype.kind in "biuf" and arr.size == len(value):
            return arr
        return json.dumps(list(value))
    if isinstance(value, (str, bool, int, float, np.generic, np.ndarray)):
        return value
    raise TypeError(f"frames file attribute {name!r}: unsupported type {type(value).__name__}")


def _decode_attr(name: str, value):
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str) and (name in JSON_ATTRS or value[:1] in "[{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dims(data: Mapping[str, np.ndarray], attrs: Mapping) -> dict[str, int]:
    roi = [int(v) for v in attrs["roi_px"]]
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] and 0 <= roi[1] < roi[3]):
        raise ValueError(f"frames file: roi_px must be [x0, y0, x1, y1] with x0 < x1 and y0 < y1, got {roi}")
    res = [int(v) for v in attrs["resolution"]]
    if len(res) != 2 or roi[2] > res[0] or roi[3] > res[1]:
        raise ValueError(f"frames file: roi_px {roi} lies outside the resolution {res}")
    offsets = data["sub_offsets"]
    if offsets.ndim != 1 or offsets.size < 1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("frames file: sub_offsets must start at 0 and never decrease")
    groups = attrs["light_groups"]
    if not isinstance(groups, (list, tuple)) or not groups or not all(isinstance(g, str) for g in groups):
        raise ValueError("frames file: light_groups must be a non-empty list of names")
    n = int(data["t_soe_ns"].shape[0])
    return {"n": n, "n+1": n + 1, "N": int(offsets[-1]), "H": roi[3] - roi[1], "W": roi[2] - roi[0],
            "G": len(groups), "n_cal": int(data["calib_frames"].shape[0]) if "calib_frames" in data else 0}


def _check(name: str, arr: np.ndarray, dims: dict[str, int]) -> np.ndarray:
    dtype, shape = DATASETS[name]
    fits = np.can_cast(arr.dtype, dtype, "safe") or (
        arr.dtype.kind in "iu" and np.dtype(dtype).kind in "iu" and
        (arr.size == 0 or (np.iinfo(dtype).min <= arr.min() and arr.max() <= np.iinfo(dtype).max)))
    if not fits:
        raise ValueError(f"frames file: {name} has dtype {arr.dtype}, which does not cast safely to {np.dtype(dtype)}")
    want = tuple(dims[s] if isinstance(s, str) else s for s in shape)
    if arr.shape != want:
        raise ValueError(f"frames file: {name} has shape {arr.shape}, expected {want} ({', '.join(map(str, shape))})")
    return np.ascontiguousarray(arr, dtype=dtype)


def write_frames(path: str | Path, datasets: Mapping[str, np.ndarray], attrs: Mapping) -> Path:
    """Writes a frames file at ``path`` (via ``.<name>.partial`` and a rename) and returns the path. ``datasets`` holds
    every REQUIRED dataset and any optional one of DATASETS, with their §3.6 shapes; ``attrs`` the REQUIRED_ATTRS and
    any others."""
    path = Path(path)
    unknown = set(datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"frames file: unknown datasets {sorted(unknown)}")
    missing = [k for k in REQUIRED if k not in datasets]
    if missing:
        raise ValueError(f"frames file: missing datasets {missing}")
    missing = [k for k in REQUIRED_ATTRS if k not in attrs]
    if missing:
        raise ValueError(f"frames file: missing attributes {missing}")
    if attrs.get("format", FORMAT) != FORMAT:
        raise ValueError(f"frames file: format must be {FORMAT!r}")
    data = {k: np.asarray(v) for k, v in datasets.items()}
    for k in ("t_soe_ns", "sub_offsets", "calib_frames"):  # the arrays that size the others
        if k in data and (data[k].ndim != 1 or not np.can_cast(data[k].dtype, DATASETS[k][0], "safe")):
            raise ValueError(f"frames file: {k} must be a one-dimensional {np.dtype(DATASETS[k][0])} array")
    dims = _dims(data, attrs)
    data = {k: _check(k, v, dims) for k, v in data.items()}
    tmp = path.with_name(f".{path.name}.partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["format"] = FORMAT
            for k, v in attrs.items():
                if k != "format":
                    f.attrs[k] = _encode_attr(k, v)
            for k, v in data.items():
                _create(f, k, v)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _create(f: h5py.File, name: str, v: np.ndarray) -> None:
    """Images one frame (or sub-render) per chunk with Blosc (zstd, level 1, bitshuffle); vectors plain."""
    if v.ndim >= 2 and v.shape[0] > 0:
        comp = dict(hdf5plugin.Blosc(cname="zstd", clevel=1, shuffle=hdf5plugin.Blosc.BITSHUFFLE))
        f.create_dataset(name, data=v, chunks=(1, *v.shape[1:]), **comp)
    else:
        f.create_dataset(name, data=v)


def _copy_attr(src: h5py.AttributeManager, dst: h5py.AttributeManager, name: str) -> None:
    """Attribute ``name`` as stored: its HDF5 type and shape, with no decode and re-encode (which would widen
    float32 or int32 scalars)."""
    aid = src.get_id(name)
    dst.create(name, src[name], shape=aid.shape, dtype=aid.dtype)


def rewrite_frames(path: str | Path, datasets: Mapping[str, np.ndarray], attrs: Mapping) -> Path:
    """Replaces or adds ``datasets`` (checked as by write_frames against the file's dimensions) and ``attrs`` (None
    removes one; ``format`` and the required ones stay) of a frames file. Every other dataset is copied by HDF5 with
    its chunks and filters, and every other attribute with its stored type and shape (a JSON null included), into
    ``.<name>.partial``, which is then renamed over the file."""
    path = Path(path)
    unknown = set(datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"frames file: unknown datasets {sorted(unknown)}")
    if "format" in attrs or any(attrs[k] is None for k in REQUIRED_ATTRS if k in attrs):
        raise ValueError("frames file: format cannot be rewritten, nor a required attribute removed")
    tmp = path.with_name(f".{path.name}.partial")
    try:
        with FramesFile(path) as old, h5py.File(tmp, "w") as f:
            # None removes only the caller's attributes: a stored JSON null (e.g. blender) is kept like the rest.
            merged = {**old.attrs, **{k: v for k, v in attrs.items() if v is not None}}
            for k, v in attrs.items():
                if v is None:
                    merged.pop(k, None)
            sizing = {k: old.read(k) for k in ("t_soe_ns", "sub_offsets", "calib_frames") if k in old.names}
            dims = _dims(sizing, merged)
            data = {k: _check(k, np.asarray(v), dims) for k, v in datasets.items()}
            for k in old._f.attrs:
                if k not in attrs:
                    _copy_attr(old._f.attrs, f.attrs, k)
            for k, v in attrs.items():
                if v is not None:
                    f.attrs[k] = _encode_attr(k, v)
            for k in old.names:
                if k not in data:
                    old._f.copy(old._f[k], f, name=k)
            for k, v in data.items():
                _create(f, k, v)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


class FramesFile:
    """A frames file opened for reading: ``attrs`` (decoded), ``names``, ``read(name, index=None)``. A context
    manager; ``close()`` releases the file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f = h5py.File(self.path, "r")
        fmt = _decode_attr("format", self._f.attrs.get("format", ""))
        if fmt != FORMAT:
            self._f.close()
            raise ValueError(f"{self.path}: not a {FORMAT} file (format {fmt!r})")
        self.attrs = {k: _decode_attr(k, v) for k, v in self._f.attrs.items()}
        self.names = tuple(sorted(self._f.keys()))

    def read(self, name: str, index=None) -> np.ndarray:
        """Dataset ``name`` whole, or its rows ``index`` (an int or a slice of the first axis)."""
        if name not in self._f:
            raise KeyError(f"{self.path}: no dataset {name!r} (it has {list(self.names)})")
        ds = self._f[name]
        return ds[()] if index is None else ds[index]

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_frames(path: str | Path, names: Iterable[str] | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    """(attributes, datasets) of a frames file: every dataset, or those in ``names``."""
    with FramesFile(path) as f:
        return f.attrs, {k: f.read(k) for k in (f.names if names is None else names)}
