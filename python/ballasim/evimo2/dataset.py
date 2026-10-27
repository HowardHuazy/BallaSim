"""Read-only access to an EVIMO2 v2 NPZ sequence and the EVIMO object catalogue.

Conventions and pitfalls (R6 §A, G1, G2, G3), all handled here:
- intrinsics come from ``meta['meta']`` (k1, k2, p1, p2), never ``dataset_info.npz['D']``, which
  holds k1..k4 and silently drops p1, p2;
- object poses in ``meta`` are T_C<-O (camera <- Vicon subject), the camera pose is T_C0<-C, where
  C0 is the camera at the first Vicon sample of the whole recording;
- ``dataset_extrinsics.npz`` holds T_rig<-cam (camera -> Vicon ``dvs_rig`` subject);
- times are seconds since bag start, written with 6 decimals; they become int64 ns here;
- depth/mask archive keys are ``frame id - first frame id`` (G3 t13), not the raw id;
- the mesh -> subject transform is a Kabsch fit of ``config.txt`` markers (mesh frame, mesh units)
  onto the first N markers of ``vicon.vsk`` (subject frame, mm), matched by order (AST-21).
EVIMO tools are GPL-3.0: this module is written from the format descriptions, not from their code.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from ballasim.scene.transforms import kabsch, make_transform, pose_to_transform

DEFAULT_ROOT = "/data/evimo2_fresh"
CAMERA_FOLDERS = ("left_camera", "right_camera", "samsung_mono")
#: Registration gates (S2 AST-21): small objects observe <= 0.98 mm, the tabletop 5.75 mm.
KABSCH_GATE_M = {"default": 1.5e-3, "tabletop": 7e-3}


def evimo2_root() -> Path:
    return Path(os.environ.get("EVIMO2_ROOT", DEFAULT_ROOT))


def objects_dir(root: Path | None = None) -> Path:
    return (root or evimo2_root()) / ".tools" / "evimo" / "evimo" / "objects"


def seconds_to_ns(ts: float) -> int:
    """EVIMO2 times are decimal strings with 6 digits; round to the ns grid exactly."""
    return round(float(ts) * 1e9)


def entry_pose(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    q = entry["pos"]["q"]
    t = entry["pos"]["t"]
    return np.array([q["w"], q["x"], q["y"], q["z"]], np.float64), np.array([t["x"], t["y"], t["z"]], np.float64)


@dataclass
class PoseTrack:
    """Samples of one body from ``full_trajectory``: rows are indices into the full trajectory."""

    rows: np.ndarray       # int64
    t_raw_ns: np.ndarray   # int64, raw Vicon stamp clock
    q: np.ndarray          # (n, 4) w, x, y, z
    p: np.ndarray          # (n, 3) metres


def low_light_category(category: str) -> bool:
    """A low-light recording category (imo_ll, sfm_ll, ...): the Vicon strobe lights it (G3 t9)."""
    return category.endswith("_ll")


class Sequence:
    """One EVIMO2 v2 sequence directory of one camera, e.g.
    ``left_camera/imo/eval/scene13_dyn_test_00_000000``."""

    def __init__(self, directory: str | os.PathLike):
        self.directory = Path(directory)
        info = self.directory / "dataset_info.npz"
        if not info.exists():
            raise FileNotFoundError(f"{self.directory} is not an EVIMO2 sequence (no dataset_info.npz)")
        with np.load(info, allow_pickle=True) as z:   # meta is a pickled dict in the release
            meta = z["meta"].item()
        self.calibration: dict = dict(meta["meta"])
        self.frames: list[dict] = meta["frames"]
        self.full_trajectory: list[dict] = meta["full_trajectory"]
        parts = self.directory.resolve().parts
        cams = [p for p in parts if p in CAMERA_FOLDERS]
        if not cams:
            raise ValueError(f"{self.directory}: cannot tell the camera (no {CAMERA_FOLDERS} folder in the path)")
        self.camera = cams[-1]
        i = parts.index(self.camera)
        self.category = parts[i + 1]                       # imo, imo_ll, sfm, sanity, ...
        self.name = self.directory.name                    # e.g. scene13_dyn_test_00_000000
        self.recording = re.sub(r"_\d{6}$", "", self.name)  # e.g. scene13_dyn_test_00
        self.split = parts[i + 2] if len(parts) > i + 3 else ""

    @property
    def low_light(self) -> bool:
        return low_light_category(self.category)

    @property
    def width(self) -> int:
        return int(self.calibration["res_x"])

    @property
    def height(self) -> int:
        return int(self.calibration["res_y"])

    @cached_property
    def T_rig_cam(self) -> np.ndarray:
        with np.load(self.directory / "dataset_extrinsics.npz", allow_pickle=True) as z:
            t = z["t_rigcamera"].item()
            q = z["q_rigcamera"].item()
        return pose_to_transform(np.array([q["w"], q["x"], q["y"], q["z"]]), np.array([t["x"], t["y"], t["z"]]))

    @cached_property
    def object_ids(self) -> list[int]:
        ids = set()
        for row in self.full_trajectory:
            ids.update(int(k) for k in row if k.isdigit())
        return sorted(ids)

    def track(self, key: str) -> PoseTrack:
        """Samples of 'cam' (T_C0<-C) or an object id string (T_C<-O) from full_trajectory."""
        rows, t, q, p = [], [], [], []
        for i, row in enumerate(self.full_trajectory):
            if key in row:
                qi, pi = entry_pose(row[key])
                rows.append(i)
                t.append(seconds_to_ns(row["ts"]))
                q.append(qi)
                p.append(pi)
        return PoseTrack(np.array(rows, np.int64), np.array(t, np.int64), np.array(q).reshape(-1, 4),
                         np.array(p).reshape(-1, 3))

    @cached_property
    def full_trajectory_t_raw_ns(self) -> np.ndarray:
        return np.array([seconds_to_ns(r["ts"]) for r in self.full_trajectory], np.int64)

    def gt_frames(self) -> list[tuple[int, int, dict]]:
        """(archive key, raw-stamp time ns, frame dict) for every frame that has depth and mask."""
        first = self.frames[0]["id"] if self.frames else 0
        return [(int(f["id"]) - int(first), seconds_to_ns(f["ts"]), f) for f in self.frames if "gt_frame" in f]

    def depth(self, key: int) -> np.ndarray:
        """uint16 camera z-depth in mm (0 = no geometry)."""
        with np.load(self.directory / "dataset_depth.npz") as z:
            return z[f"depth_{key:010d}"]

    def mask(self, key: int) -> np.ndarray:
        """uint16 object id * 1000 (0 = none)."""
        with np.load(self.directory / "dataset_mask.npz") as z:
            return z[f"mask_{key:010d}"]

    def radtan(self) -> tuple[float, float, float, float, np.ndarray]:
        """(fx, fy, cx, cy, [k1, k2, p1, p2, k3]) from meta['meta'] (never D.npy)."""
        c = self.calibration
        if c.get("dist_model", "radtan") != "radtan":
            raise ValueError(f"{self.directory}: unsupported distortion model {c.get('dist_model')!r}")
        if float(c.get("k4", 0.0)) != 0.0:
            raise ValueError(f"{self.directory}: rational radtan (k4 != 0) is not supported")
        return (float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"]),
                np.array([c["k1"], c["k2"], c["p1"], c["p2"], c.get("k3", 0.0)], np.float64))


@dataclass
class EvimoObject:
    """One folder of the EVIMO object catalogue (``objects/<name>``)."""

    id: int
    name: str
    folder: Path
    unit_scale: float               # mesh units -> metres
    mesh_path: Path
    config_markers: np.ndarray      # (N, 3) mesh frame, mesh units, in config.txt order
    vsk_markers_mm: np.ndarray      # (M, 3) subject frame, mm, in MarkerSet order
    vsk_names: list[str]
    marker_radius_mm: float
    settings: dict = field(default_factory=dict)

    def registration(self) -> tuple[np.ndarray, np.ndarray]:
        """T_subject_mesh (metres) and per-marker residuals (m), with the AST-21 gate applied."""
        n = len(self.config_markers)
        if n < 3 or len(self.vsk_markers_mm) < n:
            raise ValueError(f"{self.name}: {n} config markers but {len(self.vsk_markers_mm)} VSK markers")
        a = self.config_markers * self.unit_scale
        b = self.vsk_markers_mm[:n] * 1e-3
        r, t, res = kabsch(a, b)
        rms = float(np.sqrt(np.mean(res**2)))
        gate = KABSCH_GATE_M.get(self.name, KABSCH_GATE_M["default"])
        if rms > gate:
            raise ValueError(f"AST-E21 {self.name}: Kabsch residual {rms * 1e3:.2f} mm exceeds {gate * 1e3:.1f} mm "
                             f"(per marker {np.round(res * 1e3, 2).tolist()} mm)")
        return make_transform(r, t), res

    @property
    def untextured(self) -> bool:
        return not (self.folder / "surface_0.png").exists()


def catalogue_files(objects: str | os.PathLike) -> list[Path]:
    """The files :func:`catalogue` reads in the catalogue folder ``objects``: each object folder's settings.txt,
    config.txt and vicon.vsk (the meshes are read by the import, ballasim.scene.assets.asset_inputs)."""
    from ballasim import _scene

    return [Path(f) for f in _scene.evimo2_catalogue_files(str(Path(objects)))]


def catalogue(root: Path | None = None, *, objects: Path | None = None) -> dict[int, EvimoObject]:
    """Objects by EVIMO id from the top-level folders of ``objects/`` (the tabletop variants live in
    subfolders of ``objects/tabletop`` and are never picked: the root tabletop reproduces the
    released GT pixel-exactly in 126/126 sequences, G3 t4). ``objects``: the catalogue folder itself
    (scene.evimo2.objects_root); default ``objects_dir(root)``. The parsers (settings.txt, config.txt, vicon.vsk) are
    C++ since S5 (``ballasim._scene.evimo2_catalogue``, src/scene/evimo2/catalogue.cpp)."""
    from ballasim import _scene

    base = objects_dir(root) if objects is None else Path(objects)
    return {e["id"]: EvimoObject(e["id"], e["name"], Path(e["folder"]), e["unit_scale"], Path(e["mesh_path"]),
                                 e["config_markers"], e["vsk_markers_mm"], e["vsk_names"], e["marker_radius_mm"],
                                 e["settings"])
            for e in _scene.evimo2_catalogue(str(base))}
