"""The C1 scene bundle (docs/dev/CONTRACTS.md): the only input of the renderer and the integrator.

A bundle is fully resolved: metric, world frame, fitted trajectories, and no references into
third-party asset folders. On disk it is a directory with

- ``bundle.json``: every scalar, string and the meta dict; arrays are referenced as
  ``{"$array": key}`` and textures as ``{"$png": relpath}``;
- ``arrays.npz``: every NumPy array (dtype preserved, read without pickle);
- ``textures/*.png``: sRGB base-colour textures, lossless.

``bundle.json`` is version 1 unless the bundle uses a C1 version-2 feature (PHASE3.md D-P3-10); version 2 lists the
features it uses under ``"features"`` (a non-empty list of distinct names), and a reader refuses a version or a
feature it does not know (:data:`FEATURES`: ``baked``, docs/design/phase3/scans.md §3.1; ``pieces``,
docs/design/phase3/blender.md §3.7; P2-F's ``pbr``, ``area_light`` and ``nir``, docs/dev/CONTRACTS.md). A version-2
file never loses a field silently: the reader also refuses a key that is not a field it knows, a field of a feature
the file does not list, and a declared feature none of whose fields the file holds (version-1 files are read as
before: keys that are not fields are ignored).
A feature is the ``metadata={"feature": name}`` of its fields, and a bundle uses it when one of those
fields is off its default (:func:`at_feature_default`): only then is the field written and hashed
(``engine.scene.bundle_digest``), so a bundle that uses no feature keeps its file and its digest, and the features a
file declares are those of the fields it holds.

``SceneBundle.validate()`` checks the contract (types, shapes, index ranges, rigid transforms,
light group 0 = constant "default", holds that are exactly still, the 16 m origin rule of
DESIGN §5.3) and raises :class:`BundleError` on the first violation.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import MISSING, Field, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .transforms import is_rigid

FORMAT = "ballasim.scene_bundle"
#: Version 1: no feature. Version 2: ``"features"`` lists the C1 version-2 features the bundle uses (PHASE3.md D-P3-10).
FORMAT_VERSIONS = (1, 2)

#: Every position (trajectory control points, static geometry) must lie within this distance of
#: origin_world, so fp32 device coordinates keep a quantum <= 0.95 um (M2 §A.3).
MAX_ORIGIN_DISTANCE_M = 16.0
INVALID_INSTANCE = 0xFFFFFFFF
INVALID_SEMANTIC = 0xFFFF
CAMERA_MODELS = {"radtan": 5, "kb": 4}
#: Kinds of the C1 feature "area_light" (P2-F, p2f.md §4.6): one-sided emitters of radiance ``intensity``.
AREA_LIGHT_KINDS = ("rect", "disk")
LIGHT_KINDS = ("point", "spot", "directional", "ambient", "background") + AREA_LIGHT_KINDS
#: Kinds without a position: folded into per-group constants, never casting shadows (render/bundle.h positional).
CONSTANT_LIGHT_KINDS = ("ambient", "background")
#: Material.shading values: "lit" (albedo under the lights), "baked" (the texture is radiance, scans.md §4.1).
SHADINGS = ("lit", "baked")
#: Material.nir_form values (the C1 feature "nir", P2-F, p2f.md §4.9).
NIR_FORMS = ("assumed", "scale", "affine", "scalar", "texture")
#: Material fields of the C1 feature "pbr" (P2-F, p2f.md §4.3), each a number in [0, 1].
PBR_FIELDS = ("roughness", "metallic", "specular")


class BundleError(ValueError):
    """A scene bundle violates the C1 contract."""


def _fail(where: str, msg: str):
    raise BundleError(f"{where}: {msg}")


def _check_array(where, a, dtype, shape):
    """shape: tuple with None for free dimensions."""
    if not isinstance(a, np.ndarray):
        _fail(where, f"expected a NumPy array, got {type(a).__name__}")
    if a.dtype != np.dtype(dtype):
        _fail(where, f"expected dtype {np.dtype(dtype)}, got {a.dtype}")
    if a.ndim != len(shape) or any(s is not None and s != d for s, d in zip(shape, a.shape)):
        _fail(where, f"expected shape {tuple('*' if s is None else s for s in shape)}, got {a.shape}")
    if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
        _fail(where, "contains NaN or Inf")


def _is_int(v) -> bool:
    return not isinstance(v, (bool, np.bool_)) and isinstance(v, (int, np.integer))


def _is_bool(v) -> bool:
    return isinstance(v, (bool, np.bool_))


def _number_in(where: str, v, hi: float, what: str) -> float:
    """A C1 "pbr" value: a Python or NumPy number, not a bool, finite and in [0, hi] (render/bundle.cpp checks the
    same; a huge int is refused, not an OverflowError)."""
    x = math.nan
    if not _is_bool(v) and isinstance(v, (int, float, np.integer, np.floating)):
        try:
            x = float(v)
        except (OverflowError, TypeError, ValueError):
            pass
    if not (0.0 <= x <= hi and math.isfinite(x)):
        _fail(where, f"must be {what}, got {v!r}")
    return x


def _unit_number(where: str, v) -> float:
    return _number_in(where, v, 1.0, "a number in [0, 1]")


def _check_index(where, value, n, *, optional=False):
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < n:
        _fail(where, f"index {value!r} out of range [0, {n})")


# ---------------------------------------------------------------------------------------------
# Dataclasses (field names and meanings exactly as in CONTRACTS.md C1)
# ---------------------------------------------------------------------------------------------


@dataclass
class Mesh:
    positions: np.ndarray                 # float32[V,3], body frame, metres
    indices: np.ndarray                   # uint32[T,3]
    normals: np.ndarray | None = None     # float32[V,3] per-vertex shading normals
    uvs: np.ndarray | None = None         # float32[V,2]; v points DOWN the image (row = v * H)
    material_ids: np.ndarray | None = None  # uint16[T] into Body.materials; None = all 0

    def validate(self, where: str = "mesh", num_materials: int | None = None, *, carrier: bool = False) -> None:
        """``carrier``: the mesh is that of a light's carrier on a trajectory, which alone may be empty (no vertex and
        no triangle; blender.md §3.7)."""
        _check_array(f"{where}.positions", self.positions, np.float32, (None, 3))
        _check_array(f"{where}.indices", self.indices, np.uint32, (None, 3))
        v = len(self.positions)
        if len(self.indices) == 0:
            if not carrier:
                _fail(where, "has no triangles (only a light's carrier on a trajectory may be meshless)")
            if v:
                _fail(where, f"a meshless carrier has no vertices, got {v}")
        elif self.indices.max() >= v:
            _fail(f"{where}.indices", f"vertex index {int(self.indices.max())} >= {v} vertices")
        if self.normals is not None:
            _check_array(f"{where}.normals", self.normals, np.float32, (v, 3))
        if self.uvs is not None:
            _check_array(f"{where}.uvs", self.uvs, np.float32, (v, 2))
        if self.material_ids is not None:
            _check_array(f"{where}.material_ids", self.material_ids, np.uint16, (len(self.indices),))
            if num_materials is not None and len(self.material_ids) and self.material_ids.max() >= num_materials:
                _fail(f"{where}.material_ids", f"material {int(self.material_ids.max())} >= {num_materials}")

    def used_materials(self) -> np.ndarray:
        if self.material_ids is None:
            return np.zeros(1, np.int64)
        return np.unique(self.material_ids).astype(np.int64)


@dataclass
class Material:
    base_color: np.ndarray                # float32[3], LINEAR RGB multiplier
    texture: np.ndarray | None = None     # uint8[H,W,3], sRGB-encoded base colour
    emissive: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))  # linear RGB
    # Feature "baked" (scans.md §3.1; absent from bundle.json and bundle_digest at these defaults):
    # shading "lit" | "baked" (the texture, times base_color, is radiance); baked_scale K, renderer radiance per
    # linear texel unit (baked only; includes beta); baked_groups float32[G] shares a_g, sum 1 (None: all in group 0);
    # texture_linear float16[H,W,3] linear RGB (baked only; excludes `texture`); texture_tile 0, or the tile pitch P
    # (a power of two >= 4) of a per-face tile atlas.
    shading: str = field(default="lit", metadata={"feature": "baked"})
    baked_scale: float = field(default=1.0, metadata={"feature": "baked"})
    baked_groups: np.ndarray | None = field(default=None, metadata={"feature": "baked"})
    texture_linear: np.ndarray | None = field(default=None, metadata={"feature": "baked"})
    texture_tile: int = field(default=0, metadata={"feature": "baked"})
    # Feature "pbr" (P2-F, p2f.md §4.3, §4.11; absent at these defaults): the GGX + Lambert BSDF's mono parameters, each
    # in [0, 1]: roughness r (alpha = r^2, as Cycles' Principled and glTF), metallic m, and specular s, the dielectric
    # F0 (glTF 0.04 specularColor specularFactor at IOR 1.5; Blender ((eta - 1)/(eta + 1))^2 2 level). s = m = 0 has
    # no lobe (F0 = F90 = 0): the Lambert material, whatever r. Lit materials only.
    roughness: float = field(default=1.0, metadata={"feature": "pbr"})
    metallic: float = field(default=0.0, metadata={"feature": "pbr"})
    specular: float = field(default=0.0, metadata={"feature": "pbr"})
    # Feature "nir" (P2-F, p2f.md §4.9; absent at these defaults): the albedo the near infrared sees, mixed per (camera,
    # light group) with the visible one by the camera's sensor.nir_fraction f: rho = (1 - f) rho_vis + f rho_NIR (f = 0
    # by default: the visible albedo, exactly). With R_lin the texture's linear red channel (1 untextured) and b_R the
    # base colour's: nir_form "assumed" rho_NIR = b_R R_lin; "scale" s b_R R_lin; "affine" clamp(a + b b_R R_lin, 0,
    # 1) at level 0; "scalar" c; "texture" nir_texture / 255. nir_params float64[2] = [s, -], [a, b] or [c, -] (None:
    # [1, 0]; s and c >= 0); nir_texture uint8[H,W], linear NIR albedo on the mesh's UVs (form "texture" only, which
    # needs it; any size, resampled to the visible texture's grid). Lit materials only.
    nir_form: str = field(default="assumed", metadata={"feature": "nir"})
    nir_params: np.ndarray | None = field(default=None, metadata={"feature": "nir"})
    nir_texture: np.ndarray | None = field(default=None, metadata={"feature": "nir"})

    @property
    def has_lobe(self) -> bool:
        """Whether the material has a specular lobe (feature "pbr": specular or metallic > 0)."""
        return self.specular > 0 or self.metallic > 0

    @property
    def texture_size(self) -> tuple[int, int] | None:
        """(H, W) of the texture (8-bit or linear), or None."""
        t = self.texture if self.texture is not None else self.texture_linear
        return None if t is None else (int(t.shape[0]), int(t.shape[1]))

    def validate(self, where: str = "material", num_groups: int | None = None) -> None:
        _check_array(f"{where}.base_color", self.base_color, np.float32, (3,))
        _check_array(f"{where}.emissive", self.emissive, np.float32, (3,))
        if np.any(self.base_color < 0) or np.any(self.emissive < 0):
            _fail(where, "base_color and emissive must be non-negative")
        if self.texture is not None:
            _check_array(f"{where}.texture", self.texture, np.uint8, (None, None, 3))
            if self.texture.shape[0] == 0 or self.texture.shape[1] == 0:
                _fail(f"{where}.texture", "is empty")
        if not isinstance(self.shading, str) or self.shading not in SHADINGS:
            _fail(f"{where}.shading", f"must be one of {SHADINGS}, got {self.shading!r}")
        baked = self.shading == "baked"
        k = self.baked_scale
        kf = math.nan
        if not _is_bool(k) and isinstance(k, (int, float, np.integer, np.floating)):
            try:
                kf = float(k)                    # a huge int overflows here: refused below, not an OverflowError
            except (OverflowError, TypeError, ValueError):
                pass
        # K is fp32 in the renderer (render/bundle.h), so it must be finite there too (NaN fails the comparison).
        if not 0.0 <= kf <= float(np.finfo(np.float32).max):
            _fail(f"{where}.baked_scale", f"must be finite (as float32) and >= 0, got {k!r}")
        if self.baked_groups is not None:
            if not baked:
                _fail(f"{where}.baked_groups", "is set on a lit material (shares split baked radiance)")
            _check_array(f"{where}.baked_groups", self.baked_groups, np.float32, (num_groups,))
            if np.any(self.baked_groups < 0):
                _fail(f"{where}.baked_groups", "shares must be >= 0")
            total = 0.0
            for a in self.baked_groups:          # fp64, in order, as render/bundle.cpp sums them
                total += float(a)
            if abs(total - 1.0) > 1e-6:
                _fail(f"{where}.baked_groups", "shares must sum to 1 (1e-6)")
        if self.texture_linear is not None:
            if not baked:
                _fail(f"{where}.texture_linear", "is set on a lit material (linear textures are baked radiance)")
            if self.texture is not None:
                _fail(where, "has both texture and texture_linear")
            _check_array(f"{where}.texture_linear", self.texture_linear, np.float16, (None, None, 3))
            if self.texture_linear.shape[0] == 0 or self.texture_linear.shape[1] == 0:
                _fail(f"{where}.texture_linear", "is empty")
            if np.any(self.texture_linear < 0):
                _fail(f"{where}.texture_linear", "has negative texels")
        if baked and np.any(self.emissive != 0):
            _fail(f"{where}.emissive", "must be 0 on a baked material (baked radiance already contains emission)")
        p = self.texture_tile
        if not _is_int(p):
            _fail(f"{where}.texture_tile", f"must be an int, got {p!r}")
        if p != 0:
            size = self.texture_size
            if p < 4 or p & (p - 1):
                _fail(f"{where}.texture_tile", f"must be 0 or a power of two >= 4, got {p}")
            if size is None or size[0] % p or size[1] % p:
                _fail(f"{where}.texture_tile", f"tile pitch {p} must divide the texture's height and width {size}")
        self._validate_pbr_nir(where, baked)

    def _validate_pbr_nir(self, where: str, baked: bool) -> None:
        """The rules of the C1 features "pbr" and "nir" (p2f.md §4.3, §4.9, §4.11), as render::validate states them."""
        for name in PBR_FIELDS:
            _unit_number(f"{where}.{name}", getattr(self, name))
        if baked and not all(at_feature_default(self, name) for name in PBR_FIELDS):
            _fail(where, "a baked material is radiance: roughness, metallic and specular keep their defaults")
        if not isinstance(self.nir_form, str) or self.nir_form not in NIR_FORMS:
            _fail(f"{where}.nir_form", f"must be one of {NIR_FORMS}, got {self.nir_form!r}")
        if self.nir_params is not None:
            _check_array(f"{where}.nir_params", self.nir_params, np.float64, (2,))
            if self.nir_form in ("scale", "scalar") and self.nir_params[0] < 0:
                _fail(f"{where}.nir_params", f"[0] (the scale s, or the scalar c) must be >= 0, got {self.nir_params[0]}")
        if self.nir_texture is not None:
            if self.nir_form != "texture":
                _fail(f"{where}.nir_texture", f"is set on nir_form {self.nir_form!r} (form 'texture' only)")
            _check_array(f"{where}.nir_texture", self.nir_texture, np.uint8, (None, None))
            if self.nir_texture.size == 0:
                _fail(f"{where}.nir_texture", "is empty")
        elif self.nir_form == "texture":
            _fail(f"{where}.nir_texture", "nir_form 'texture' needs nir_texture")
        if baked and not all(at_feature_default(self, name) for name in ("nir_form", "nir_params", "nir_texture")):
            _fail(where, "a baked material is radiance: it has no NIR albedo (nir_form, nir_params, nir_texture)")


@dataclass
class Body:
    id: int                               # instance id >= 1, unique
    name: str
    semantic: int                         # semantic class id (0 = unlabelled)
    mesh: Mesh                            # in the BODY frame, metres
    materials: list[Material]
    trajectory: int | None                # index into trajectories; None = static
    T_world_body: np.ndarray              # float64[4,4]; used when static; identity for dynamic bodies
    # Feature "baked" (scans.md §3.1; absent from bundle.json and bundle_digest at this default): captured scan
    # geometry, whose shadows are in the baked textures.
    scan: bool = field(default=False, metadata={"feature": "baked"})
    # Feature "pbr" (P2-F, p2f.md §4.5; absent at this default): r_f >= 0, the scanner-noise correlation length (m) over
    # which the renderer prefilters the shading normals of the body's specular materials (Toksvig, F7); 0: none.
    normal_prefilter_m: float = field(default=0.0, metadata={"feature": "pbr"})

    def validate(self, where: str, num_trajectories: int, num_groups: int | None = None, *,
                 carries_light: bool = False) -> None:
        """``carries_light``: a light rides this body (``Light.body``); with a trajectory it may then be meshless."""
        if isinstance(self.id, bool) or not isinstance(self.id, (int, np.integer)) or \
                not 1 <= self.id < INVALID_INSTANCE:
            _fail(where, f"instance id {self.id!r} must be an int in [1, 0xFFFFFFFF)")
        if not isinstance(self.name, str) or not self.name:
            _fail(where, "needs a non-empty name")
        if not isinstance(self.semantic, (int, np.integer)) or not 0 <= self.semantic < INVALID_SEMANTIC:
            _fail(where, f"semantic {self.semantic!r} must be in [0, 0xFFFF)")
        if not self.materials:
            _fail(where, "needs at least one material")
        for j, m in enumerate(self.materials):
            m.validate(f"{where}.materials[{j}]", num_groups)
        self.mesh.validate(f"{where}.mesh", len(self.materials),
                           carrier=carries_light and self.trajectory is not None)
        textured = [j for j in self.mesh.used_materials() if self.materials[j].texture_size is not None
                    or self.materials[j].nir_texture is not None]
        if textured and self.mesh.uvs is None:
            _fail(where, f"materials {textured} have textures but the mesh has no uvs")
        _check_index(f"{where}.trajectory", self.trajectory, num_trajectories, optional=True)
        _check_array(f"{where}.T_world_body", self.T_world_body, np.float64, (4, 4))
        if not is_rigid(self.T_world_body):
            _fail(f"{where}.T_world_body", "is not a rigid transform")
        if self.trajectory is not None and not np.array_equal(self.T_world_body, np.eye(4)):
            _fail(f"{where}.T_world_body", "must be the identity for a dynamic body")
        if not _is_bool(self.scan):
            _fail(f"{where}.scan", "must be a bool")
        if self.scan and self.trajectory is not None:
            _fail(where, "a scan body is static (scan with a trajectory)")
        if self.scan and not any(self.materials[j].shading == "baked" for j in self.mesh.used_materials()):
            _fail(where, "scan is set but none of the body's used materials is baked")
        _number_in(f"{where}.normal_prefilter_m", self.normal_prefilter_m, math.inf, "a finite number >= 0 (metres)")


def carrier_material() -> Material:
    """The one material of a meshless light carrier (blender.md §3.7): Lambert 0.8 grey, no texture. Nothing shades
    it; it keeps Body.validate's rule of at least one material."""
    return Material(np.full(3, 0.8, np.float32))


def meshless_carrier(id: int, name: str, trajectory: int, semantic: int = 0) -> Body:
    """A body with no vertex and no triangle that carries lights along ``trajectory`` (C1, blender.md §3.7): valid
    only while a Light.body refers to it. The renderer never hits it and the planner leaves it out of its body lists;
    its id never appears in ground truth."""
    return Body(id, name, semantic, Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)),
                [carrier_material()], trajectory, np.eye(4))


#: Trajectory.pieces columns (the C1 version-2 feature "pieces"; docs/design/phase3/blender.md §3.7).
PIECE_COLUMNS = ("begin_ns", "t0_ns", "h_ns", "ctrl_begin", "ctrl_end", "join")
#: Join of piece p > 0 to piece p - 1: -1 a step, 0 C0 (the pose is continuous), 1 C1 (the velocity too); piece 0: 0.
PIECE_JOINS = (-1, 0, 1)


def _check_increments(where: str, rot: np.ndarray) -> None:
    # M2 §A.2: per-knot increments stay within pi/2 so Log never wraps.
    dots = np.abs(np.sum(rot[1:] * rot[:-1], axis=1))
    if len(dots) and 2.0 * np.arccos(np.clip(dots.min(), -1.0, 1.0)) > math.pi / 2 + 1e-12:
        _fail(where, "consecutive control rotations differ by more than pi/2")


@dataclass
class Trajectory:
    """Split SO(3) x R^3 cumulative B-spline (DESIGN §5.3); segment s covers [t0 + s h, t0 + (s+1) h).

    Piecewise (``pieces``, the C1 version-2 feature "pieces"): P >= 2 such splines of order K, right-continuous.
    Piece p is the spline of the control rows [ctrl_begin, ctrl_end) with knots t0_p + s h_p, and serves
    [begin_p, begin_{p+1}); the last piece serves up to the span's end (closed). ``rot`` and ``pos`` concatenate the
    pieces' controls, and ``t0_ns`` and ``h_ns`` are piece 0's.
    """

    K: int                                # 4 or 6
    t0_ns: int                            # knot origin
    h_ns: int                             # knot spacing
    rot: np.ndarray                       # float64[N,4] control rotations R_world_body (w, x, y, z)
    pos: np.ndarray                       # float64[N,3] control positions relative to origin_world
    holds: np.ndarray                     # int64[H,2] [t_begin, t_end) intervals of exact stillness
    span_ns: tuple[int, int]              # valid query interval
    #: int64[P,6], P >= 2, rows PIECE_COLUMNS; None: one piece. At None it is neither written nor hashed.
    pieces: np.ndarray | None = field(default=None, metadata={"feature": "pieces"})

    @property
    def num_segments(self) -> int:
        """Segments of a single-piece trajectory (a piecewise one raises: each piece has its own)."""
        if self.pieces is not None:
            raise ValueError("a piecewise trajectory has a segment count per piece (piece_support)")
        return len(self.rot) - self.K + 1

    @property
    def end_ns(self) -> int:
        """Right end of the spline's support (closed): t0 + num_segments h; of a piecewise trajectory, its last
        piece's."""
        if self.pieces is not None:
            return self.piece_support(len(self.pieces) - 1)[1]
        return self.t0_ns + self.num_segments * self.h_ns

    def piece_support(self, p: int) -> tuple[int, int]:
        """The support of piece p's spline, closed: [t0_p, t0_p + (ctrl_end - ctrl_begin - K + 1) h_p]."""
        _, t0, h, c0, c1, _ = (int(v) for v in self.pieces[p])
        return t0, t0 + (c1 - c0 - self.K + 1) * h

    def hold_ranges(self, t_begin: int, t_end: int) -> list[tuple[int, int]]:
        """The control rows acting on [t_begin, t_end), one closed range per piece serving part of it (piece 0 also
        serves its support before begin_0, the last piece its support after the span)."""
        if self.pieces is None:
            return [hold_control_range(self, t_begin, t_end)]
        out, last, begins = [], len(self.pieces) - 1, self.pieces[:, 0]
        # Only the pieces serving part of the interval (begins strictly increase): p0 serves t_begin, p1 t_end - 1.
        p0 = max(0, int(np.searchsorted(begins, t_begin, "right")) - 1)
        p1 = max(0, int(np.searchsorted(begins, t_end, "left")) - 1)
        for p in range(p0, p1 + 1):
            a = max(t_begin, int(self.pieces[p, 0])) if p else t_begin
            b = min(t_end, int(self.pieces[p + 1, 0])) if p < last else t_end
            if a < b:
                out.append(hold_control_range(self, a, b, piece=p))
        return out

    def validate(self, where: str = "trajectory") -> None:
        if self.K not in (4, 6):
            _fail(where, f"K must be 4 or 6, got {self.K!r}")
        for name in ("t0_ns", "h_ns"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                _fail(f"{where}.{name}", f"must be an int, got {type(v).__name__}")
        if not 1 <= self.h_ns < 2**31:
            _fail(f"{where}.h_ns", f"{self.h_ns} outside [1, 2^31)")
        _check_array(f"{where}.rot", self.rot, np.float64, (None, 4))
        _check_array(f"{where}.pos", self.pos, np.float64, (len(self.rot), 3))
        if len(self.rot) < self.K:
            _fail(where, f"needs at least K = {self.K} control points, has {len(self.rot)}")
        if np.abs(np.linalg.norm(self.rot, axis=1) - 1.0).max() > 1e-9:
            _fail(f"{where}.rot", "quaternions must have unit norm (1e-9)")
        if self.pieces is None:
            _check_increments(f"{where}.rot", self.rot)
        else:
            self._validate_pieces(where)
        a, b = (int(v) for v in self.span_ns)
        if not self.t0_ns <= a < b <= self.end_ns:
            _fail(f"{where}.span_ns", f"[{a}, {b}] not inside the spline support [{self.t0_ns}, {self.end_ns}]")
        _check_array(f"{where}.holds", self.holds, np.int64, (None, 2))
        for i, (hb, he) in enumerate(self.holds):
            if not hb < he:
                _fail(f"{where}.holds[{i}]", "needs t_begin < t_end")
            if i and hb < self.holds[i - 1][1]:
                _fail(f"{where}.holds", "intervals must be sorted and disjoint")
            if hb < self.t0_ns or he > self.end_ns:
                _fail(f"{where}.holds[{i}]", "lies outside the spline support")
            # Every piece under the hold holds the same control point, so the pose is bitwise constant across joins.
            ranges = self.hold_ranges(int(hb), int(he))
            c = ranges[0][0]
            if not all(np.all(self.rot[c0:c1 + 1] == self.rot[c]) and np.all(self.pos[c0:c1 + 1] == self.pos[c])
                       for c0, c1 in ranges):
                _fail(f"{where}.holds[{i}]", "control points under the hold are not identical")

    def _validate_pieces(self, where: str) -> None:
        """The rules of blender.md §3.7: coverage, control rows, support and spacing, joins, top-level fields, and
        the pi/2 rule over each piece's own consecutive controls."""
        _check_array(f"{where}.pieces", self.pieces, np.int64, (None, len(PIECE_COLUMNS)))
        last, rows, next_row = len(self.pieces) - 1, len(self.rot), 0
        if last < 1:
            _fail(f"{where}.pieces", f"needs at least 2 pieces (None for one), has {last + 1}")
        span_begin, span_end = (int(v) for v in self.span_ns)
        if int(self.pieces[0, 0]) != span_begin or int(self.pieces[last, 0]) >= span_end:
            _fail(f"{where}.pieces", f"piece 0 must begin at span_ns[0] = {span_begin} and the last piece before "
                  f"span_ns[1] = {span_end}")
        if (self.t0_ns, self.h_ns) != (int(self.pieces[0, 1]), int(self.pieces[0, 2])):
            _fail(where, "t0_ns and h_ns must equal piece 0's")
        for p in range(1, last + 1):
            if int(self.pieces[p, 0]) <= int(self.pieces[p - 1, 0]):
                _fail(f"{where}.pieces[{p}]", "begin_ns must exceed the previous piece's")
        for p in range(last + 1):
            begin, _, h, c0, c1, join = (int(v) for v in self.pieces[p])
            at = f"{where}.pieces[{p}]"
            if not 1 <= h < 2**31:
                _fail(at, f"h_ns {h} outside [1, 2^31)")
            if c0 != next_row or c1 > rows or c1 - c0 < self.K:
                _fail(at, f"control rows [{c0}, {c1}) must start at row {next_row}, lie within the {rows} rows and "
                      f"hold at least K = {self.K}")
            next_row = c1
            if join not in (PIECE_JOINS if p else (0,)):
                _fail(at, f"join {join} must be one of {PIECE_JOINS if p else (0,)}")
            end = int(self.pieces[p + 1, 0]) if p < last else span_end
            lo, hi = self.piece_support(p)
            if not lo <= begin < end <= hi:
                _fail(at, f"the spline support [{lo}, {hi}] does not cover the piece's interval [{begin}, {end}]")
            _check_increments(f"{at} rot", self.rot[c0:c1])
        if next_row != rows:
            _fail(f"{where}.pieces", f"the pieces' control rows cover {next_row} of the {rows} rows")


def hold_control_range(traj: Trajectory, t_begin: int, t_end: int, piece: int | None = None) -> tuple[int, int]:
    """Control points that act on [t_begin, t_end): segments floor..ceil-1 and their K nodes (rows of rot and pos,
    closed). A piecewise trajectory needs ``piece``, the piece whose spline serves the interval (Trajectory.
    hold_ranges splits an interval by piece)."""
    if traj.pieces is None:
        if piece is not None:
            raise ValueError("hold_control_range: piece given for a single-piece trajectory")
        t0, h, segments, row0 = traj.t0_ns, traj.h_ns, traj.num_segments, 0
    else:
        if piece is None:
            raise ValueError("hold_control_range on a piecewise trajectory needs the piece")
        _, t0, h, row0, row1, _ = (int(v) for v in traj.pieces[piece])
        segments = row1 - row0 - traj.K + 1
    s0 = (t_begin - t0) // h
    s1 = -((t0 - t_end) // h) - 1  # ceil((t_end - t0)/h) - 1
    return int(row0 + s0), int(row0 + min(s1, segments - 1) + traj.K - 1)


@dataclass
class Camera:
    name: str
    model: str                            # "radtan" | "kb"
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray                      # float64[n]; radtan: k1, k2, p1, p2, k3; kb: k1..k4
    rig_trajectory: int                   # trajectory of the rig body carrying the camera
    T_rig_cam: np.ndarray                 # float64[4,4]
    clock_offset_ns: int                  # t_sensor = t_true + clock_offset_ns
    sensor: str                           # sensor preset name (Stream P)
    pixel_pitch_m: float
    f_number: float = 2.0

    def validate(self, where: str, num_trajectories: int) -> None:
        if self.model not in CAMERA_MODELS:
            _fail(where, f"model must be one of {sorted(CAMERA_MODELS)}, got {self.model!r}")
        if not isinstance(self.name, str) or not self.name:
            _fail(where, "needs a non-empty name")
        for k in ("width", "height"):
            v = getattr(self, k)
            if not isinstance(v, (int, np.integer)) or v <= 0:
                _fail(f"{where}.{k}", "must be a positive int")
        for k in ("fx", "fy", "pixel_pitch_m", "f_number"):
            v = getattr(self, k)
            if not np.isfinite(v) or v <= 0:
                _fail(f"{where}.{k}", "must be positive and finite")
        for k in ("cx", "cy"):
            if not np.isfinite(getattr(self, k)):
                _fail(f"{where}.{k}", "must be finite")
        _check_array(f"{where}.dist", self.dist, np.float64, (CAMERA_MODELS[self.model],))
        _check_index(f"{where}.rig_trajectory", self.rig_trajectory, num_trajectories)
        _check_array(f"{where}.T_rig_cam", self.T_rig_cam, np.float64, (4, 4))
        if not is_rigid(self.T_rig_cam):
            _fail(f"{where}.T_rig_cam", "is not a rigid transform")
        if isinstance(self.clock_offset_ns, bool) or not isinstance(self.clock_offset_ns, (int, np.integer)):
            _fail(f"{where}.clock_offset_ns", "must be an int")
        if not isinstance(self.sensor, str) or not self.sensor:
            _fail(f"{where}.sensor", "needs a preset name")


def validate_waveform(where: str, w: dict) -> None:
    if not isinstance(w, dict) or "type" not in w:
        _fail(where, "waveform must be a dict with a 'type'")
    kind = w["type"]
    if kind == "constant":
        extra = set(w) - {"type"}
    elif kind == "fourier":
        extra = set(w) - {"type", "f0_hz", "a", "phi"}
        a, phi = list(w.get("a", [])), list(w.get("phi", []))
        if not np.isfinite(w.get("f0_hz", np.nan)) or w["f0_hz"] <= 0:
            _fail(where, "fourier needs f0_hz > 0")
        if len(a) != len(phi) or not a:
            _fail(where, "fourier needs equally long, non-empty a and phi")
        if not np.all(np.isfinite(a + phi)) or sum(abs(x) for x in a) > 1.0 + 1e-12:
            _fail(where, "fourier coefficients must be finite with sum |a_k| <= 1 (so m(t) >= 0)")
    elif kind == "pulse":
        extra = set(w) - {"type", "f_hz", "duty", "rise_s", "phase_s", "floor"}
        try:
            f, duty, rise, phase, floor = (float(w[k]) for k in ("f_hz", "duty", "rise_s", "phase_s", "floor"))
        except KeyError as e:
            _fail(where, f"pulse waveform misses {e}")
        # Trapezoid with equal rise and fall times and duty = (plateau + rise) * f, i.e. the width at
        # half height over the period (M3 V7 X3). The checks repeat the sensor's make_pulse_waveform
        # operation for operation, so a bundle passes here exactly when stream P accepts it: a rise
        # of at least 1 ns (finite slope), floor < 1 (peak = floor + (1 - floor) / duty), plateau >= 0
        # and plateau plus both ramps within one period.
        if not (f > 0 and 0 < duty <= 1 and rise >= 1e-9 and np.isfinite(phase) and 0 <= floor < 1):
            _fail(where, "pulse needs f_hz > 0, 0 < duty <= 1, rise_s >= 1 ns, finite phase_s, 0 <= floor < 1")
        period = 1.0 / f
        top = duty * period - rise
        if not (top >= 0 and 2 * rise + top <= period):
            _fail(where, "pulse needs rise_s <= duty / f_hz (plateau >= 0) and duty / f_hz + rise_s <= 1 / f_hz "
                         "(plateau and both ramps within one period)")
    else:
        _fail(where, f"unknown waveform type {kind!r}")
    if extra:
        _fail(where, f"unknown waveform fields {sorted(extra)}")


@dataclass
class LightGroup:
    name: str
    waveform: dict

    def validate(self, where: str) -> None:
        if not isinstance(self.name, str) or not self.name:
            _fail(where, "needs a non-empty name")
        validate_waveform(f"{where}.waveform", self.waveform)


@dataclass
class Light:
    kind: str                             # "point" | "spot" | "directional" | "ambient" | "background"
    group: int                            # index into light_groups
    color: np.ndarray                     # float32[3], linear RGB
    intensity: float                      # relative units in Phase 1
    body: int | None = None               # index into bodies (light attached, body frame) or None = world
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    spot_inner_rad: float = 0.0
    spot_outer_rad: float = 0.0
    casts_shadows: bool = True
    # Feature "baked" (scans.md §3.1; absent from bundle.json and bundle_digest at this default): the light stands for
    # the scan's captured lighting (scans.md §4.2, §4.3).
    proxy: bool = field(default=False, metadata={"feature": "baked"})
    # Feature "area_light" (P2-F, p2f.md §4.6; absent at these defaults): kinds "rect" and "disk" are one-sided emitters
    # of radiance `intensity`, centred at `position`, emitting towards `direction` (the emitting side's unit normal,
    # Blender's local -Z). size float64[2] [s_u, s_v] in metres (a disk's ellipse diameters), both > 0; axis float64[3]
    # the unit in-plane u axis, |axis . direction| < 1e-6 (None: (1, 0, 0)). Other kinds leave both None.
    size: np.ndarray | None = field(default=None, metadata={"feature": "area_light"})
    axis: np.ndarray | None = field(default=None, metadata={"feature": "area_light"})

    def validate(self, where: str, num_groups: int, num_bodies: int) -> None:
        if self.kind not in LIGHT_KINDS:
            _fail(where, f"kind must be one of {LIGHT_KINDS}, got {self.kind!r}")
        _check_index(f"{where}.group", self.group, num_groups)
        _check_index(f"{where}.body", self.body, num_bodies, optional=True)
        _check_array(f"{where}.color", self.color, np.float32, (3,))
        if np.any(self.color < 0) or not np.isfinite(self.intensity) or self.intensity < 0:
            _fail(where, "color and intensity must be non-negative")
        _check_array(f"{where}.position", self.position, np.float64, (3,))
        _check_array(f"{where}.direction", self.direction, np.float64, (3,))
        if self.kind in ("spot", "directional") + AREA_LIGHT_KINDS and abs(np.linalg.norm(self.direction) - 1.0) > 1e-6:
            _fail(f"{where}.direction", "must be a unit vector for spot, directional, rect and disk lights")
        if self.kind in AREA_LIGHT_KINDS:
            if self.size is None:
                _fail(f"{where}.size", f"a {self.kind} light needs size [s_u, s_v] (metres)")
            _check_array(f"{where}.size", self.size, np.float64, (2,))
            if np.any(self.size <= 0):
                _fail(f"{where}.size", "both sizes must be > 0")
            if self.axis is not None:
                _check_array(f"{where}.axis", self.axis, np.float64, (3,))
            axis = np.array([1.0, 0.0, 0.0]) if self.axis is None else self.axis
            if abs(np.linalg.norm(axis) - 1.0) > 1e-6:
                _fail(f"{where}.axis", "must be a unit vector")
            if not abs(float(axis @ self.direction)) < 1e-6:
                _fail(f"{where}.axis", "must lie in the emitter's plane (|axis . direction| < 1e-6)")
        elif self.size is not None or self.axis is not None:
            _fail(where, f"size and axis belong to rect and disk lights, not to a {self.kind} light")
        # The renderer's rule (bundle.cpp): a cone of half-angle up to pi, with a falloff between the two angles.
        if self.kind == "spot" and not 0.0 <= self.spot_inner_rad < self.spot_outer_rad <= math.pi:
            _fail(where, "spot needs 0 <= spot_inner_rad < spot_outer_rad <= pi")
        if not isinstance(self.casts_shadows, (bool, np.bool_)):
            _fail(f"{where}.casts_shadows", "must be a bool")
        if not _is_bool(self.proxy):
            _fail(f"{where}.proxy", "must be a bool")
        if self.proxy and self.kind not in CONSTANT_LIGHT_KINDS and not self.casts_shadows:
            _fail(where, f"a proxy {self.kind} light must cast shadows (it shadows inserted objects on the scan)")


@dataclass
class SceneBundle:
    name: str
    t_start_ns: int
    t_end_ns: int
    origin_world: np.ndarray              # float64[3]
    bodies: list[Body]
    trajectories: list[Trajectory]
    cameras: list[Camera]
    light_groups: list[LightGroup]        # index 0 is always the constant group "default"
    lights: list[Light]
    meta: dict = field(default_factory=dict)

    # -- validation -----------------------------------------------------------------------

    def validate(self) -> SceneBundle:
        if not isinstance(self.name, str) or not self.name:
            _fail("bundle", "needs a non-empty name")
        for k in ("t_start_ns", "t_end_ns"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                _fail(f"bundle.{k}", "must be an int")
        if not self.t_start_ns < self.t_end_ns:
            _fail("bundle", "needs t_start_ns < t_end_ns")
        _check_array("bundle.origin_world", self.origin_world, np.float64, (3,))
        nt = len(self.trajectories)
        for i, tr in enumerate(self.trajectories):
            tr.validate(f"trajectories[{i}]")
        ids = set()
        carriers = {light.body for light in self.lights if _is_int(getattr(light, "body", None))}
        for i, b in enumerate(self.bodies):
            b.validate(f"bodies[{i}] ({getattr(b, 'name', '?')})", nt, len(self.light_groups),
                       carries_light=i in carriers)
            if b.id in ids:
                _fail(f"bodies[{i}]", f"duplicate instance id {b.id}")
            ids.add(b.id)
        for i, c in enumerate(self.cameras):
            c.validate(f"cameras[{i}]", nt)
        if len({c.name for c in self.cameras}) != len(self.cameras):
            _fail("bundle.cameras", "camera names must be unique")
        if not self.light_groups or self.light_groups[0].name != "default" or \
                self.light_groups[0].waveform != {"type": "constant"}:
            _fail("bundle.light_groups", "index 0 must be the constant group 'default'")
        for i, g in enumerate(self.light_groups):
            g.validate(f"light_groups[{i}]")
        if len({g.name for g in self.light_groups}) != len(self.light_groups):
            _fail("bundle.light_groups", "group names must be unique")
        for i, light in enumerate(self.lights):
            light.validate(f"lights[{i}]", len(self.light_groups), len(self.bodies))
            if light.proxy and light.body is not None and self.bodies[light.body].trajectory is not None:
                _fail(f"lights[{i}]", "a proxy stands for captured, static lighting (its body has a trajectory)")
        if any(light.proxy for light in self.lights) and not any(b.scan for b in self.bodies):
            _fail("bundle.lights", "proxy lights need a scan body (mark the scanned room's body `scan: true`)")
        used = {b.trajectory for b in self.bodies if b.trajectory is not None} | \
            {c.rig_trajectory for c in self.cameras}
        for i in sorted(used):
            a, b = self.trajectories[i].span_ns
            if not (a <= self.t_start_ns and self.t_end_ns <= b):
                _fail(f"trajectories[{i}]", f"span [{a}, {b}] does not cover the simulation span "
                      f"[{self.t_start_ns}, {self.t_end_ns}]")
        self._check_origin_distance()
        try:
            json.dumps(self.meta, allow_nan=False)
        except (TypeError, ValueError) as e:
            _fail("bundle.meta", f"must be JSON-serializable: {e}")
        return self

    def _check_origin_distance(self) -> None:
        for i, b in enumerate(self.bodies):
            x = b.mesh.positions.astype(np.float64)
            if b.trajectory is None:
                # The exact reach of the static geometry, max_i |R x_i + t| (blender.md §3.7): the norm is convex, so
                # over a triangle it peaks at a vertex. Every vertex lies in the AABB, so the 8 transformed AABB
                # corners checked before this never reached less; every bundle they accepted is still accepted.
                reach = np.linalg.norm(x @ b.T_world_body[:3, :3].T + b.T_world_body[:3, 3], axis=1).max()
            else:
                radius = 0.0                     # a meshless carrier (blender.md §3.7) is its origin's path alone
                if len(x):
                    lo, hi = x.min(axis=0), x.max(axis=0)
                    radius = np.linalg.norm(np.maximum(np.abs(lo), np.abs(hi)))
                reach = np.linalg.norm(self.trajectories[b.trajectory].pos, axis=1).max() + radius
            if reach > MAX_ORIGIN_DISTANCE_M:
                _fail(f"bodies[{i}] ({b.name})", f"reaches {reach:.2f} m from origin_world "
                      f"(limit {MAX_ORIGIN_DISTANCE_M} m, DESIGN §5.3)")
        for i, tr in enumerate(self.trajectories):
            reach = np.linalg.norm(tr.pos, axis=1).max()
            if reach > MAX_ORIGIN_DISTANCE_M:
                _fail(f"trajectories[{i}]", f"reaches {reach:.2f} m from origin_world")

    # -- plain dict for the renderer binding --------------------------------------------

    def as_dict(self) -> dict:
        """Nested plain dict (field names as in C1; arrays are not copied)."""
        return _to_plain(self)

    # -- persistence ---------------------------------------------------------------------

    def save(self, directory: str | os.PathLike) -> Path:
        """Validate and write the bundle to ``directory`` (replaced atomically if it exists)."""
        self.validate()
        directory = Path(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
        try:
            arrays: dict[str, np.ndarray] = {}
            (tmp / "textures").mkdir()
            used: set[str] = set()
            doc = _encode(self, "", arrays, tmp, used)
            features = sorted(used)                # the features of the fields written, so all are declared
            head = {"format": FORMAT, "version": 2, "features": features} if features else \
                {"format": FORMAT, "version": 1}
            doc = {**head, **doc}
            with open(tmp / "arrays.npz", "wb") as f:
                np.savez(f, **arrays)
            with open(tmp / "bundle.json", "w") as f:
                json.dump(doc, f, indent=1, allow_nan=False)
                f.write("\n")
            old = None
            if directory.exists():
                if not (directory / "bundle.json").exists():
                    raise FileExistsError(f"{directory} exists and is not a scene bundle")
                old = directory.with_name(f".{directory.name}.old")
                if old.exists():
                    shutil.rmtree(old)
                directory.rename(old)
            tmp.rename(directory)
            if old is not None:
                shutil.rmtree(old)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return directory

    @classmethod
    def load(cls, directory: str | os.PathLike, *, validate: bool = True) -> SceneBundle:
        directory = Path(directory)
        with open(directory / "bundle.json") as f:
            doc = json.load(f)
        version = doc.get("version")
        if doc.get("format") != FORMAT or _is_bool(version) or version not in FORMAT_VERSIONS:
            raise BundleError(f"{directory}: not a {FORMAT} bundle of version "
                              f"{' or '.join(map(str, FORMAT_VERSIONS))} (version {version!r})")
        declared = doc.get("features")
        if version == 1 and declared is not None:
            raise BundleError(f"{directory}: a version-1 bundle lists no features, got {declared!r}")
        if version == 2 and not (isinstance(declared, list) and declared and
                                 all(isinstance(x, str) for x in declared) and len(set(declared)) == len(declared)):
            raise BundleError(f"{directory}: a version-2 bundle must list its features, a non-empty list of distinct "
                              f"names, got {declared!r}")
        unknown = sorted(set(declared or ()) - set(FEATURES))
        if unknown:
            raise BundleError(f"{directory}: uses the C1 features {unknown}, which this reader does not know "
                              f"(it knows {sorted(FEATURES)}); a newer BallaSim wrote it")
        with np.load(directory / "arrays.npz", allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
        bundle = _decode(cls, {k: v for k, v in doc.items() if k not in ("format", "version", "features")},
                         arrays, directory, strict=version == 2)
        used, declared = set(bundle.features()), set(declared or ())
        if used - declared:
            raise BundleError(f"{directory}: uses the C1 features {sorted(used - declared)} without declaring them")
        # save() lists exactly the features it writes; a declared feature without its fields lost them.
        if declared - used:
            raise BundleError(f"{directory}: declares the C1 features {sorted(declared - used)} but holds none of "
                              f"their fields")
        return bundle.validate() if validate else bundle

    def features(self) -> list[str]:
        """The C1 version-2 features the bundle uses, sorted: the metadata "feature" of every field off its default
        (at_feature_default), over the dataclasses and lists that save() writes; empty for a version-1 bundle."""
        used: set[str] = set()

        def walk(obj) -> None:
            if is_dataclass(obj):
                for f in content_fields(obj):
                    if "feature" in f.metadata:
                        used.add(f.metadata["feature"])
                    walk(getattr(obj, f.name))
            elif isinstance(obj, (list, tuple)):
                for x in obj:
                    walk(x)

        walk(self)
        return sorted(used)


#: A feature field's default must be one at_feature_default compares by value: None or a plain scalar.
FEATURE_DEFAULT_TYPES = (type(None), bool, int, float, str)


def feature_names(classes) -> frozenset[str]:
    """The C1 version-2 feature names marked on the fields of ``classes`` (``metadata={"feature": name}``). Raises
    TypeError for a name that is not a non-empty string, or a feature field without a plain default (a
    default_factory or no default, which at_feature_default would never see as unused, so the field would be
    written, hashed and declared in every bundle)."""
    names: set[str] = set()
    for cls in classes:
        for f in fields(cls):
            if "feature" not in f.metadata:
                continue
            name, where = f.metadata["feature"], f"{cls.__name__}.{f.name}"
            if not isinstance(name, str) or not name:
                raise TypeError(f"C1 feature field {where}: the feature name must be a non-empty string, got {name!r}")
            if f.default is MISSING or f.default_factory is not MISSING or \
                    type(f.default) not in FEATURE_DEFAULT_TYPES:
                raise TypeError(f"C1 feature field {where} needs a plain default (None, bool, int, float or str)")
            names.add(name)
    return frozenset(names)


#: The C1 version-2 features this reader knows (PHASE3.md D-P3-10), read from the field metadata of this module's
#: dataclasses, so the registry cannot miss a feature field. load() refuses a feature that is not here (a lenient
#: decoder would drop its fields), a bundle that uses a feature without declaring it or declares one it does not use,
#: and, in a version-2 file, a key that is not a field (_decode's ``strict``).
FEATURES: frozenset[str] = feature_names(v for v in list(globals().values())
                                         if isinstance(v, type) and is_dataclass(v) and v.__module__ == __name__)


# ---------------------------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------------------------

_LIST_TYPES = {"bodies": Body, "trajectories": Trajectory, "cameras": Camera,
               "light_groups": LightGroup, "lights": Light, "materials": Material}


def _to_plain(obj):
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_plain(x) for x in obj)
    return obj


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def at_feature_default(obj, f: Field | str) -> bool:
    """Whether field f (a Field or its name) of the C1 dataclass obj belongs to a version-2 feature (metadata
    "feature") and holds its default. Such a field is neither written to bundle.json nor hashed by
    engine.scene.bundle_digest, so a bundle that does not use the feature keeps its file and its digest. A value equal
    to the default is at it (1 for 1.0, numpy scalars included) unless one is a bool and the other not, or it is None or
    an array where the default is not; a None default is held only by None."""
    if isinstance(f, str):
        f = next(x for x in fields(obj) if x.name == f)
    if "feature" not in f.metadata:
        return False
    v, default = getattr(obj, f.name), f.default
    if default is None:
        return v is None
    if v is None or isinstance(v, np.ndarray) or _is_bool(v) != _is_bool(default):
        return False
    return bool(v == default)


def content_fields(obj) -> list[Field]:
    """The fields of a C1 dataclass that its file and its digest hold: all but version-2 feature fields at their
    defaults (at_feature_default)."""
    return [f for f in fields(obj) if not at_feature_default(obj, f)]


def _encode(obj, path: str, arrays: dict, root: Path, used: set[str]):
    """obj as JSON, its arrays added to ``arrays``; ``used`` gains the feature of every feature field written."""
    if is_dataclass(obj):
        out = {}
        for f in content_fields(obj):
            if "feature" in f.metadata:
                used.add(f.metadata["feature"])
            v = getattr(obj, f.name)
            key = f"{path}/{f.name}" if path else f.name
            if f.name == "texture" and v is not None:
                rel = f"textures/{path.replace('/', '_')}.png"
                Image.fromarray(v, "RGB").save(root / rel, optimize=False, compress_level=6)
                out[f.name] = {"$png": rel}
            elif f.name == "meta":
                out[f.name] = v
            else:
                out[f.name] = _encode(v, key, arrays, root, used)
        return out
    if isinstance(obj, np.ndarray):
        arrays[path] = obj
        return {"$array": path}
    if isinstance(obj, list):
        return [_encode(x, f"{path}/{i}", arrays, root, used) for i, x in enumerate(obj)]
    if isinstance(obj, tuple):
        return [_encode(x, f"{path}/{i}", arrays, root, used) for i, x in enumerate(obj)]
    return _jsonable(obj)


def _decode_value(v, arrays: dict, root: Path):
    if isinstance(v, dict) and set(v) == {"$array"}:
        return arrays[v["$array"]]
    if isinstance(v, dict) and set(v) == {"$png"}:
        with Image.open(root / v["$png"]) as im:
            return np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
    return v


def _decode(cls, doc: dict, arrays: dict, root: Path, strict: bool = False, path: str = ""):
    """``strict`` (version-2 files): refuse a key that is not a field of ``cls``, which a lenient reader would drop
    (a feature field under a wrong key, or the field of a feature this reader does not know, written without listing
    it). Version-1 files are read as before: keys that are not fields are ignored."""
    if strict:
        extra = sorted(set(doc) - {f.name for f in fields(cls)})
        if extra:
            raise BundleError(f"{root}: {path or 'the bundle'} has keys {extra} that are not {cls.__name__} fields "
                              f"this reader knows (a version-2 file lists the feature of every field it sets)")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in doc:
            continue
        v = doc[f.name]
        key = f"{path}/{f.name}" if path else f.name
        if f.name in _LIST_TYPES:
            kwargs[f.name] = [_decode(_LIST_TYPES[f.name], x, arrays, root, strict, f"{key}/{i}")
                              for i, x in enumerate(v)]
        elif f.name == "mesh":
            kwargs[f.name] = _decode(Mesh, v, arrays, root, strict, key)
        elif f.name == "meta":
            kwargs[f.name] = v
        elif f.name == "span_ns":
            kwargs[f.name] = (int(v[0]), int(v[1]))
        else:
            kwargs[f.name] = _decode_value(v, arrays, root)
    return cls(**kwargs)


# ---------------------------------------------------------------------------------------------
# Small constructors used by the importers
# ---------------------------------------------------------------------------------------------


def default_light_group() -> LightGroup:
    return LightGroup("default", {"type": "constant"})


def static_trajectory(q: np.ndarray, p: np.ndarray, span_ns: tuple[int, int], *, K: int = 6,
                      h_ns: int | None = None) -> Trajectory:
    """A trajectory that holds one pose over span_ns exactly (all control points equal)."""
    a, b = int(span_ns[0]), int(span_ns[1])
    if not a < b:
        raise ValueError("static_trajectory needs span_ns[0] < span_ns[1]")
    h = int(h_ns) if h_ns is not None else max(1, min(b - a, 2**31 - 1))
    n_seg = -((a - b) // h)
    n = n_seg + K - 1
    q = np.asarray(q, np.float64) / np.linalg.norm(q)
    return Trajectory(K, a, h, np.tile(q, (n, 1)), np.tile(np.asarray(p, np.float64), (n, 1)),
                      np.array([[a, a + n_seg * h]], np.int64), (a, b))

