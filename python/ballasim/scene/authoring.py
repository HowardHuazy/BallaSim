"""Scene YAML -> C1 SceneBundle for authored (synthetic) scenes.

The schema (``schema: ballasim.scene/1``) is documented in docs/dev/SCENE.md; in short:

- ``variables``: string substitutions ``${NAME}`` (an environment variable of the same name wins);
- ``time``: ``t_start_s``/``t_end_s`` (or ``_ns``); ``origin_world_m``; ``semantic_classes``;
- ``assets``: meshes by path with import options (``scale`` is mandatory for OBJ/PLY) and licence;
- ``materials`` and ``room``: a procedural room box with tiled textures (UV = position / tile);
- ``bodies``: an asset with a body-frame choice (``origin: asset | bottom | centroid``, extra
  ``scale``, ``rotation_rpy_deg``) and either a static ``pose`` or ``keyframes`` (holds via
  ``hold_s`` or identical keys), optionally with ``spin`` and ``shake`` perturbations;
- ``rigs`` (trajectories that carry cameras), ``cameras`` (explicit intrinsics or a preset such as
  ``evimo2_left``), ``light_groups`` (waveforms) and ``lights``.

Scanned rooms (docs/design/phase3/scans.md §3.3, the C1 feature "baked"): assets take ``shading`` (auto | lit |
baked, see load_asset) and ``scale_basis`` (metric-source | measured | unverified; required for an asset with baked
materials, kept in ``meta.assets``); bodies take ``scan: true`` (captured scan geometry) and ``baked: {scale,
groups}`` (K and the light-group shares of every baked material); ``material_overrides`` take ``shading``,
``baked_scale``, ``baked_groups`` (group name -> share) and ``texture_linear`` (an ``.npy`` of H x W x 3 float16 or
float32 linear RGB); lights take ``proxy: true``.

Radiance v1 (P2-F F4, the C1 features "pbr", "area_light" and "nir", docs/dev/SCENE.md): glTF assets take ``pbr:
true`` (read the metallic-roughness BSDF); ``materials`` and ``material_overrides`` take ``roughness``, ``metallic``,
``specular`` (each in [0, 1]), ``nir_form``, ``nir_params`` ([p0, p1]) and ``nir_texture`` (an 8-bit grey PNG of
linear NIR albedo); bodies take ``normal_prefilter_m``; ``rect`` and ``disk`` lights take ``size_m`` ([s_u, s_v]) and
``axis`` (the unit in-plane u axis, default [1, 0, 0]) besides ``position_m`` and ``direction`` (the emitting side).

Poses are ``{t_m, rpy_deg}`` (extrinsic x-y-z, i.e. R = Rz(yaw) Ry(pitch) Rx(roll)),
``{t_m, q_wxyz}`` or the camera-style ``{eye_m, target_m, up}`` whose frame is OpenCV's (z toward
the target, y down). Keyframed trajectories are fitted in exact interpolation mode with holds
(DESIGN §5.4 step 10); outside the first and last key a body holds that key's pose.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation

from ballasim.traj.evaluate import evaluate
from ballasim.traj.fit import PoseSamples, fit_interpolating

from .assets import load_asset
from .assets.geometry import area_centroid, merge_meshes, transform_mesh
from .bundle import Body, Camera, Light, LightGroup, Material, Mesh, SceneBundle, default_light_group
from .licenses import RESTRICTED, canonical
from .transforms import make_transform, matrix_to_quat, quat_multiply, quat_to_matrix, so3_exp

SCHEMA = "ballasim.scene/1"
NS = 1_000_000_000

#: Camera presets: EVIMO2 calibrations of the 2020-11-24/25 sessions (meta['meta'] of
#: scene13_dyn_test_00), so synthetic frames sit naturally next to the real data.
CAMERA_PRESETS = {
    "evimo2_left": {"model": "radtan", "width": 640, "height": 480, "fx": 556.573975, "fy": 556.086975,
                    "cx": 321.846008, "cy": 225.617996, "dist": [-0.10828, 0.206233, -0.00065, -0.000555, 0.0],
                    "sensor": "prophesee_gen3", "pixel_pitch_m": 15e-6, "f_number": 2.0},
    "evimo2_right": {"model": "radtan", "width": 640, "height": 480, "fx": 554.585022, "fy": 554.210022,
                     "cx": 327.556, "cy": 202.320999, "dist": [-0.094841, 0.187073, 0.001747, -0.001038, 0.0],
                     "sensor": "prophesee_gen3", "pixel_pitch_m": 15e-6, "f_number": 2.0},
    "evimo2_samsung": {"model": "radtan", "width": 640, "height": 480, "fx": 520.189026, "fy": 520.166016,
                       "cx": 324.177002, "cy": 239.169998, "dist": [0.102258, -0.139354, 0.000379, 0.000844, 0.0],
                       "sensor": "samsung_gen3", "pixel_pitch_m": 9e-6, "f_number": 2.0},
}


class SceneSpecError(ValueError):
    """The scene YAML is malformed or inconsistent."""


class SceneWarning(UserWarning):
    """A scene YAML that loads, with a choice the user should know about (scans.md §6)."""


#: Asset ``scale_basis`` values (scans.md §3.3): how the scan's metric scale is known.
SCALE_BASES = ("metric-source", "measured", "unverified")
#: The keys of a scene asset entry that are ``load_asset`` options (``scan report SCENE.yaml#ASSET`` reads the same ones).
ASSET_LOAD_OPTIONS = ("format", "scale", "up_axis", "node", "uv_origin", "max_texture_size", "shading", "pbr",
                      "tile_texels", "workers")
#: The C1 "pbr" and "nir" keys of ``materials`` and ``material_overrides`` (P2-F F4).
PBR_NIR_KEYS = {"roughness", "metallic", "specular", "nir_form", "nir_params", "nir_texture"}
MATERIAL_OVERRIDE_KEYS = {"base_color", "emissive", "shading", "baked_scale", "baked_groups", "texture_linear"} | \
    PBR_NIR_KEYS


def _check_keys(where: str, d: dict, allowed: set[str], required: set[str] = frozenset()):
    if not isinstance(d, dict):
        raise SceneSpecError(f"{where}: expected a mapping, got {type(d).__name__}")
    unknown = set(d) - allowed
    if unknown:
        raise SceneSpecError(f"{where}: unknown keys {sorted(unknown)} (allowed: {sorted(allowed)})")
    missing = set(required) - set(d)
    if missing:
        raise SceneSpecError(f"{where}: missing keys {sorted(missing)}")


def _vec(where: str, v, n: int = 3) -> np.ndarray:
    a = np.asarray(v, np.float64)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise SceneSpecError(f"{where}: expected {n} finite numbers, got {v!r}")
    return a


def _number(where: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SceneSpecError(f"{where}: expected a number, got {v!r}")
    return float(v)


def _unit(where: str, v) -> np.ndarray:
    a = _vec(where, v)
    if not np.linalg.norm(a) > 0:
        raise SceneSpecError(f"{where}: expected a non-zero vector, got {v!r}")
    return a / np.linalg.norm(a)


def _bool(where: str, v) -> bool:
    if not isinstance(v, bool):
        raise SceneSpecError(f"{where}: expected true or false, got {v!r}")
    return v


def _time_ns(where: str, d: dict, key: str) -> int:
    if f"{key}_ns" in d:
        return int(d[f"{key}_ns"])
    if f"{key}_s" in d:
        return round(float(d[f"{key}_s"]) * NS)
    raise SceneSpecError(f"{where}: needs {key}_s or {key}_ns")


POSE_KEYS = {"t_m", "rpy_deg", "q_wxyz", "eye_m", "target_m", "up"}


def parse_pose(where: str, d: dict) -> np.ndarray:
    """A 4x4 T_world_body from {t_m, rpy_deg} | {t_m, q_wxyz} | {eye_m, target_m, up}."""
    if "eye_m" in d:
        eye, target = _vec(f"{where}.eye_m", d["eye_m"]), _vec(f"{where}.target_m", d.get("target_m"))
        up = _vec(f"{where}.up", d.get("up", [0.0, 0.0, 1.0]))
        z = target - eye
        if np.linalg.norm(z) < 1e-9:
            raise SceneSpecError(f"{where}: eye and target coincide")
        z /= np.linalg.norm(z)
        x = np.cross(z, up)
        if np.linalg.norm(x) < 1e-9:
            raise SceneSpecError(f"{where}: view direction parallel to up")
        x /= np.linalg.norm(x)
        y = np.cross(z, x)          # OpenCV camera: x right, y down, z forward
        return make_transform(np.stack([x, y, z], axis=1), eye)
    t = _vec(f"{where}.t_m", d.get("t_m", [0.0, 0.0, 0.0]))
    if "q_wxyz" in d:
        r = quat_to_matrix(_vec(f"{where}.q_wxyz", d["q_wxyz"], 4))
    else:
        r = Rotation.from_euler("xyz", _vec(f"{where}.rpy_deg", d.get("rpy_deg", [0.0, 0.0, 0.0])),
                                degrees=True).as_matrix()
    return make_transform(r, t)


class _Builder:
    def __init__(self, spec: dict, base_dir: Path):
        self.spec = spec
        self.base_dir = base_dir
        self.variables = {k: str(v) for k, v in (spec.get("variables") or {}).items()}
        self.files: list[Path] = []     # every file path the YAML names, as resolved (load_scene's ``files``)

    # -- strings and paths ---------------------------------------------------------------------

    def subst(self, s: str) -> str:
        def repl(m):
            name = m.group(1)
            if name in os.environ:
                return os.environ[name]
            if name in self.variables:
                return self.subst(self.variables[name])
            raise SceneSpecError(f"undefined variable ${{{name}}}")
        return re.sub(r"\$\{(\w+)\}", repl, s)

    def path(self, s: str) -> Path:
        p = Path(self.subst(s)).expanduser()
        p = p if p.is_absolute() else (self.base_dir / p)
        self.files.append(p)
        return p

    def texture_linear(self, where: str, s) -> np.ndarray:
        """An .npy of H x W x 3 linear RGB, float16 or float32 (rounded to float16, nearest even)."""
        if not isinstance(s, str):
            raise SceneSpecError(f"{where}: expected the path of an .npy file, got {s!r}")
        try:
            t = np.load(self.path(s), allow_pickle=False)
        except (OSError, ValueError) as e:
            raise SceneSpecError(f"{where}: cannot read {s!r} as a NumPy .npy file: {e}") from None
        if not isinstance(t, np.ndarray):           # an .npz (any zip) loads as an open NpzFile
            t.close()
            raise SceneSpecError(f"{where}: {s!r} is not a single-array NumPy .npy file")
        if t.ndim != 3 or t.shape[2] != 3 or 0 in t.shape or t.dtype not in (np.float16, np.float32):
            raise SceneSpecError(f"{where}: expected a non-empty H x W x 3 float16 or float32 array, got {t.dtype} "
                                 f"{t.shape}")
        with np.errstate(over="ignore"):
            t = np.ascontiguousarray(t.astype(np.float16))
        if not np.all(np.isfinite(t)):
            raise SceneSpecError(f"{where}: texels must be finite and within binary16's range (|x| < 65520)")
        return t

    def pbr_nir(self, where: str, d: dict, m: Material) -> None:
        """Applies the PBR_NIR_KEYS of d to m (their values are checked by SceneBundle.validate)."""
        for k in ("roughness", "metallic", "specular"):
            if k in d:
                setattr(m, k, _number(f"{where}.{k}", d[k]))
        if "nir_form" in d:
            m.nir_form = d["nir_form"]
        if "nir_params" in d:
            m.nir_params = _vec(f"{where}.nir_params", d["nir_params"], 2)
        if "nir_texture" in d:
            with Image.open(self.path(d["nir_texture"])) as im:
                m.nir_texture = np.ascontiguousarray(np.asarray(im.convert("L"), np.uint8))

    # -- trajectories ----------------------------------------------------------------------------

    def keyframes(self, where: str, keys: list) -> PoseSamples:
        t, q, p = [], [], []
        for i, k in enumerate(keys):
            w = f"{where}.keyframes[{i}]"
            _check_keys(w, k, POSE_KEYS | {"t_s", "t_ns", "hold_s"})
            pose = parse_pose(w, k)
            qi, pi = matrix_to_quat(pose[:3, :3]), pose[:3, 3]
            ti = _time_ns(w, k, "t")
            t.append(ti)
            q.append(qi)
            p.append(pi)
            if "hold_s" in k:
                t.append(ti + round(float(k["hold_s"]) * NS))
                q.append(qi)
                p.append(pi)
        try:
            return PoseSamples(np.array(t, np.int64), np.array(q), np.array(p))
        except ValueError as e:
            raise SceneSpecError(f"{where}.keyframes: {e}") from None

    def trajectory(self, where: str, spec: dict, span: tuple[int, int]):
        keys = self.keyframes(where, spec["keyframes"])
        h = int(spec.get("knot_spacing_ms", 20) * 1_000_000)
        fit = fit_interpolating(keys, span_ns=span, max_h_ns=h)
        perturb = [(kind, p) for kind in ("spin", "shake") for p in _as_list(spec.get(kind))]
        if not perturb:
            return fit
        # Perturbations act on the interpolated base motion: dense keys inside their windows and
        # the original keys elsewhere, each rotated (body frame) by every perturbation at its time
        # (a spin keeps its accumulated angle after its window), then one more exact interpolation.
        dt = int(spec.get("perturbation_key_ms", 10) * 1_000_000)
        windows = []
        for kind, p in perturb:
            _check_keys(f"{where}.{kind}", p, PERTURB_KEYS[kind], {"t0_s", "t1_s"})
            windows.append((round(p["t0_s"] * NS), round(p["t1_s"] * NS)))
        dense = np.unique(np.concatenate([np.arange(a, b + 1, dt) for a, b in windows]
                                         + [np.array([b for _, b in windows])]))
        if dense.min() < span[0] or dense.max() > span[1]:
            raise SceneSpecError(f"{where}: perturbation windows must lie inside the scene time span")
        inside = np.zeros(len(keys), bool)
        for a, b in windows:
            inside |= (keys.t_ns >= a) & (keys.t_ns <= b)
        base = evaluate(fit.trajectory, dense)
        t_all = np.concatenate([keys.t_ns[~inside], dense])
        q_all = np.concatenate([keys.q[~inside], base["q"]])
        p_all = np.concatenate([keys.p[~inside], base["p"]])
        order = np.argsort(t_all, kind="stable")
        t_all, q_all, p_all = t_all[order], q_all[order], p_all[order]
        for kind, p in perturb:
            q_all = quat_multiply(q_all, so3_exp(_perturbation(kind, p, t_all)))
        return fit_interpolating(PoseSamples(t_all, q_all, p_all), span_ns=span, max_h_ns=h)

    # -- materials and geometry --------------------------------------------------------------------

    def material(self, where: str, d: dict) -> tuple[Material, np.ndarray | None]:
        _check_keys(where, d, {"texture", "base_color", "emissive", "tile_m"} | PBR_NIR_KEYS)
        tex = None
        if "texture" in d:
            with Image.open(self.path(d["texture"])) as im:
                tex = np.ascontiguousarray(np.asarray(im.convert("RGB"), np.uint8))
        base = _vec(f"{where}.base_color", d.get("base_color", [1.0, 1.0, 1.0] if tex is not None else [0.5, 0.5, 0.5]))
        emissive = _vec(f"{where}.emissive", d.get("emissive", [0.0, 0.0, 0.0]))
        tile = _vec(f"{where}.tile_m", d["tile_m"], 2) if "tile_m" in d else None
        mat = Material(base.astype(np.float32), tex, emissive.astype(np.float32))
        self.pbr_nir(where, d, mat)
        return mat, tile

    def room(self, d: dict, materials: dict) -> tuple[Mesh, list[Material]]:
        _check_keys("room", d, {"size_m", "center_m", "cell_m", "walls", "floor", "ceiling", "id", "name", "semantic"},
                    {"size_m", "walls", "floor", "ceiling", "id"})
        sx, sy, sz = _vec("room.size_m", d["size_m"])
        cx, cy = _vec("room.center_m", d.get("center_m", [0.0, 0.0]), 2)
        cell = float(d.get("cell_m", 0.5))
        x0, x1, y0, y1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2
        # Inward-facing rectangles (origin, edge u, edge v), u x v pointing into the room; on walls
        # u is horizontal and v vertical, so textures stand upright (image top at the ceiling).
        z = (0.0, 0.0, sz)
        faces = {
            "floor": [((x0, y0, 0.0), (sx, 0, 0), (0, sy, 0))],
            "ceiling": [((x0, y1, sz), (sx, 0, 0), (0, -sy, 0))],
            "walls": [((x1, y0, 0.0), (-sx, 0, 0), z), ((x0, y1, 0.0), (sx, 0, 0), z),
                      ((x0, y0, 0.0), (0, sy, 0), z), ((x1, y1, 0.0), (0, -sy, 0), z)],
        }
        parts = []
        for slot, rects in faces.items():
            name = d[slot]
            if name not in materials:
                raise SceneSpecError(f"room.{slot}: unknown material {name!r}")
            mat, tile = materials[name]
            for o, u, v in rects:
                rect = _grid_rect(np.array(o, float), np.array(u, float), np.array(v, float), cell, tile)
                parts.append((rect, [mat]))
        return merge_meshes(parts)


PERTURB_KEYS = {"spin": {"t0_s", "t1_s", "axis", "rate_hz", "ramp_s"},
                "shake": {"t0_s", "t1_s", "amplitude_deg", "freq_hz", "phase_deg", "ramp_s"}}


def _as_list(v) -> list:
    return [] if v is None else (v if isinstance(v, list) else [v])


def _envelope(t: np.ndarray, t0: int, t1: int, ramp: float) -> np.ndarray:
    """C2 window: 0 outside [t0, t1], smoothstep (quintic) ramps of `ramp` s, 1 in between."""
    s = (t - t0) * 1e-9
    e = (t1 - t) * 1e-9
    x = np.clip(np.minimum(s, e) / max(ramp, 1e-9), 0.0, 1.0)
    return x**3 * (10 - 15 * x + 6 * x * x)


def _perturbation(kind: str, p: dict, t: np.ndarray) -> np.ndarray:
    """Body-frame rotation vectors of a spin or shake perturbation at times t (a spin is 0 before
    its window and keeps its final angle after it; a shake is 0 outside its window)."""
    t0, t1 = round(p["t0_s"] * NS), round(p["t1_s"] * NS)
    ramp = float(p.get("ramp_s", 0.2))
    if kind == "spin":
        axis = _vec("spin.axis", p.get("axis", [0, 0, 1]))
        axis = axis / np.linalg.norm(axis)
        # theta = integral of 2 pi rate * envelope, by the trapezoid rule on a 1 ms grid.
        grid = np.arange(t0, t1 + 1_000_000, 1_000_000)
        rate = 2 * math.pi * float(p["rate_hz"]) * _envelope(grid, t0, t1, ramp)
        theta = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * 1e-3)])
        return np.interp(t, grid, theta)[:, None] * axis[None, :]
    amp = np.radians(_vec("shake.amplitude_deg", p["amplitude_deg"]))
    freq = _vec("shake.freq_hz", p["freq_hz"])
    phase = np.radians(_vec("shake.phase_deg", p.get("phase_deg", [0, 0, 0])))
    s = (t - t0)[:, None] * 1e-9
    return _envelope(t, t0, t1, ramp)[:, None] * amp * np.sin(2 * math.pi * freq * s + phase)


def _grid_rect(o: np.ndarray, u: np.ndarray, v: np.ndarray, cell: float, tile: np.ndarray | None) -> Mesh:
    """A rectangle o + a u + b v split into cells <= cell; UVs = metric coordinates / tile size
    (beyond [0, 1]: textures repeat, the glTF default wrap mode)."""
    nu = max(1, math.ceil(np.linalg.norm(u) / cell))
    nv = max(1, math.ceil(np.linalg.norm(v) / cell))
    a, b = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1), indexing="ij")
    pos = o + a.ravel()[:, None] * u + b.ravel()[:, None] * v
    idx = (np.arange(nu)[:, None] * (nv + 1) + np.arange(nv)[None, :]).ravel()
    tris = np.concatenate([np.stack([idx, idx + nv + 1, idx + nv + 2], 1), np.stack([idx, idx + nv + 2, idx + 1], 1)])
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    uvs = None
    if tile is not None:
        # v grows downward in C1, so image rows run against the v edge (downward on walls).
        uvs = np.stack([a.ravel() * np.linalg.norm(u) / tile[0], (1.0 - b.ravel()) * np.linalg.norm(v) / tile[1]], 1)
    return Mesh(pos.astype(np.float32), tris.astype(np.uint32), np.tile(n, (len(pos), 1)).astype(np.float32),
                None if uvs is None else uvs.astype(np.float32))


def _semantic(where: str, value, classes: dict) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if value not in classes:
        raise SceneSpecError(f"{where}: unknown semantic class {value!r}")
    return int(classes[value])


def _require_scale_basis(name: str, spec: dict, materials: list[Material]) -> None:
    if "scale_basis" not in spec and any(m.shading == "baked" for m in materials):
        raise SceneSpecError(f"assets.{name}: an asset with baked materials needs scale_basis, one of {SCALE_BASES} "
                             "(how its metric scale is known; scans.md §3.3)")


def _body_materials(b: _Builder, where: str, bs: dict, asset, shares) -> list[Material]:
    """The body's copies of the asset's materials (every C1 field kept) with ``material_overrides`` and the body's
    ``baked`` defaults applied: an override's ``shading`` first, then ``baked: {scale, groups}`` on every baked
    material, then the override's other keys, which win."""
    mats = [dataclasses.replace(m, base_color=m.base_color.copy(), emissive=m.emissive.copy())
            for m in asset.materials]
    overrides = []
    for k, ov in (bs.get("material_overrides") or {}).items():
        w = f"{where}.material_overrides.{k}"
        _check_keys(w, ov, MATERIAL_OVERRIDE_KEYS)
        if not 0 <= int(k) < len(mats):
            raise SceneSpecError(f"{where}.material_overrides: no material {k}")
        j = int(k)
        overrides.append((w, j, ov))
        if "shading" in ov:
            mats[j].shading = ov["shading"]
            shrunk = {d["material"] for d in asset.info.get("textures_downsampled", [])}
            # warned only when the re-quantized 8-bit texture is kept: an override's texture_linear replaces it
            if (ov["shading"] == "baked" and asset.materials[j].shading != "baked" and j in shrunk
                    and "texture_linear" not in ov):
                warnings.warn(f"{w}: shading baked on a texture re-quantized to 8 bits when it was downsampled at "
                              "import; set the asset's `shading: baked` to keep linear downsampling (scans.md §3.5)",
                              SceneWarning, stacklevel=3)
    if "baked" in bs:
        bd = bs["baked"]
        _check_keys(f"{where}.baked", bd, {"scale", "groups"})
        baked = [m for m in mats if m.shading == "baked"]
        if not baked:
            raise SceneSpecError(f"{where}.baked: no material of asset {bs['asset']!r} is baked (set the asset's "
                                 "`shading: baked`, or a material override's)")
        for m in baked:
            if "scale" in bd:
                m.baked_scale = _number(f"{where}.baked.scale", bd["scale"])
            if "groups" in bd:
                m.baked_groups = shares(f"{where}.baked.groups", bd["groups"])
    for w, j, ov in overrides:
        m = mats[j]
        if "base_color" in ov:
            m.base_color = _vec("base_color", ov["base_color"]).astype(np.float32)
        if "emissive" in ov:
            m.emissive = _vec("emissive", ov["emissive"]).astype(np.float32)
        if "baked_scale" in ov:
            m.baked_scale = _number(f"{w}.baked_scale", ov["baked_scale"])
        if "baked_groups" in ov:
            m.baked_groups = shares(f"{w}.baked_groups", ov["baked_groups"])
        if "texture_linear" in ov:
            m.texture, m.texture_linear = None, b.texture_linear(f"{w}.texture_linear", ov["texture_linear"])
        b.pbr_nir(w, ov, m)
    return mats


def _read_spec(path: Path) -> tuple[dict, _Builder]:
    """A scene YAML's document, its top-level keys and schema checked, and its builder."""
    spec = YAML(typ="safe").load(path.read_text())
    _check_keys("scene", spec, {"schema", "name", "variables", "time", "origin_world_m", "semantic_classes", "assets",
                                "materials", "room", "bodies", "rigs", "cameras", "light_groups", "lights", "meta"},
                {"schema", "name", "time", "cameras"})
    if spec["schema"] != SCHEMA:
        raise SceneSpecError(f"schema must be {SCHEMA!r}, got {spec['schema']!r}")
    return spec, _Builder(spec, path.parent)


def _assets(spec: dict, b: _Builder) -> list[tuple[str, Path, dict, dict]]:
    """Each asset of a scene: its name, its path, its load_asset options and its spec, checked."""
    out = []
    for name, a in (spec.get("assets") or {}).items():
        _check_keys(f"assets.{name}", a, {"path", "format", "scale", "up_axis", "node", "uv_origin", "max_texture_size",
                                          "shading", "pbr", "license", "attribution", "scale_basis", "tile_texels",
                                          "workers"},
                    {"path", "license"})
        if a.get("format") == "replica" and canonical(a["license"]) != "Replica-Research":
            raise SceneSpecError(f"assets.{name}: format replica requires license Replica-Research (scans.md §3.7)")
        if "scale_basis" in a and a["scale_basis"] not in SCALE_BASES:
            raise SceneSpecError(f"assets.{name}.scale_basis must be one of {SCALE_BASES}, got {a['scale_basis']!r}")
        opts = {k: a[k] for k in ASSET_LOAD_OPTIONS if k in a}
        out.append((name, b.path(a["path"]), opts, a))
    return out


def asset_specs(path: str | os.PathLike) -> list[tuple[Path, dict]]:
    """The assets a scene YAML imports, as ``(path, options)`` of its :func:`load_asset` calls (the UI's asset check,
    ``assets.cached_key(path, **options)``, docs/design/phase3/ui.md §4.12), without importing any."""
    spec, b = _read_spec(Path(path))
    return [(asset_path, opts) for _, asset_path, opts, _ in _assets(spec, b)]


def load_scene(path: str | os.PathLike, *, files: list | None = None) -> SceneBundle:
    """Read a scene YAML and build the validated SceneBundle. ``files``, when given, is extended with every file the
    YAML names (assets, textures, ``texture_linear`` arrays), resolved: the UI's bundle LRU re-checks them
    (docs/design/phase3/ui.md §4.12)."""
    path = Path(path)
    spec, b = _read_spec(path)
    if files is not None:
        b.files = files
    _check_keys("time", spec["time"], {"t_start_s", "t_end_s", "t_start_ns", "t_end_ns"})
    span = (_time_ns("time", spec["time"], "t_start"), _time_ns("time", spec["time"], "t_end"))
    origin = _vec("origin_world_m", spec.get("origin_world_m", [0, 0, 0]))
    classes = dict(spec.get("semantic_classes") or {})

    assets = {}
    for name, asset_path, opts, a in _assets(spec, b):
        assets[name] = (load_asset(asset_path, **opts), a)
        averaged = assets[name][0].info.get("pbr_textures_averaged")
        if averaged:
            warnings.warn(f"assets.{name}: materials {averaged} have a metallicRoughnessTexture, read as its mean "
                          "(textured roughness and metallic are P2-F F15's)", SceneWarning, stacklevel=2)
        _require_scale_basis(name, a, assets[name][0].materials)
        if any(m.shading == "baked" for m in assets[name][0].materials) and canonical(a["license"]) is None:
            warnings.warn(f"assets.{name}: unknown licence {a['license']!r} on baked materials (scans.md §3.7)",
                          SceneWarning, stacklevel=2)
    materials = {name: b.material(f"materials.{name}", m) for name, m in (spec.get("materials") or {}).items()}

    # Light groups come before the bodies, whose baked shares name them.
    groups = [default_light_group()]
    group_index = {"default": 0}
    for name, w in (spec.get("light_groups") or {}).items():
        if name == "default":
            raise SceneSpecError("light_groups: 'default' is implicit (constant)")
        group_index[name] = len(groups)
        groups.append(LightGroup(name, dict(w)))

    def shares(where: str, d) -> np.ndarray:
        """float32[G] light-group shares from a mapping group name -> share; omitted groups get 0."""
        if not isinstance(d, dict) or not d:
            raise SceneSpecError(f"{where}: expected a mapping light group -> share")
        out = np.zeros(len(groups), np.float32)
        for gname, v in d.items():
            if gname not in group_index:
                raise SceneSpecError(f"{where}: unknown light group {gname!r}")
            out[group_index[gname]] = _number(f"{where}.{gname}", v)
        return out

    trajectories, bodies, fit_meta, body_index = [], [], {}, {}

    def shift(t: np.ndarray) -> np.ndarray:
        out = t.copy()
        out[:3, 3] -= origin
        return out

    if "room" in spec:
        mesh, mats = b.room(spec["room"], materials)
        r = spec["room"]
        bodies.append(Body(int(r["id"]), r.get("name", "room"), _semantic("room", r.get("semantic"), classes),
                           mesh, mats, None, shift(np.eye(4))))
    for i, bs in enumerate(spec.get("bodies") or []):
        where = f"bodies[{i}]"
        _check_keys(where, bs, {"name", "id", "semantic", "asset", "scale", "origin", "rotation_rpy_deg",
                                "material_overrides", "pose", "keyframes", "spin", "shake", "knot_spacing_ms",
                                "perturbation_key_ms", "scan", "baked", "normal_prefilter_m"}, {"name", "id", "asset"})
        if bs["asset"] not in assets:
            raise SceneSpecError(f"{where}: unknown asset {bs['asset']!r}")
        asset, asset_spec = assets[bs["asset"]]
        rpy = _vec(f"{where}.rotation_rpy_deg", bs.get("rotation_rpy_deg", [0, 0, 0]))
        pre = make_transform(Rotation.from_euler("xyz", rpy, degrees=True).as_matrix() * float(bs.get("scale", 1.0)),
                             np.zeros(3))
        mesh = transform_mesh(asset.mesh, pre)
        origin_mode = bs.get("origin", "asset")
        if origin_mode == "bottom":
            lo, hi = mesh.positions.min(axis=0).astype(np.float64), mesh.positions.max(axis=0).astype(np.float64)
            o = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        elif origin_mode == "centroid":
            o = area_centroid(mesh)
        elif origin_mode == "asset":
            o = np.zeros(3)
        else:
            raise SceneSpecError(f"{where}.origin must be asset, bottom or centroid")
        mesh = transform_mesh(mesh, make_transform(np.eye(3), -o))
        mats = _body_materials(b, where, bs, asset, shares)
        _require_scale_basis(bs["asset"], asset_spec, mats)
        scan = _bool(f"{where}.scan", bs.get("scan", False)) or asset_spec.get("format") == "replica"   # forced (§3.3)
        prefilter = {"normal_prefilter_m": _number(f"{where}.normal_prefilter_m", bs["normal_prefilter_m"])} \
            if "normal_prefilter_m" in bs else {}
        semantic = _semantic(where, bs.get("semantic"), classes)
        if ("pose" in bs) == ("keyframes" in bs):
            raise SceneSpecError(f"{where}: give exactly one of pose or keyframes")
        if "pose" in bs:
            _check_keys(f"{where}.pose", bs["pose"], POSE_KEYS)
            pose = shift(parse_pose(where, bs["pose"]))
            bodies.append(Body(int(bs["id"]), bs["name"], semantic, mesh, mats, None, pose, scan=scan, **prefilter))
        else:
            fit = b.trajectory(where, bs, span)
            traj = fit.trajectory
            traj.pos = traj.pos - origin
            trajectories.append(traj)
            fit_meta[bs["name"]] = fit.diagnostics
            bodies.append(Body(int(bs["id"]), bs["name"], semantic, mesh, mats, len(trajectories) - 1, np.eye(4),
                               scan=scan, **prefilter))
        body_index[bs["name"]] = len(bodies) - 1

    rig_index = {}
    for name, rs in (spec.get("rigs") or {}).items():
        where = f"rigs.{name}"
        _check_keys(where, rs, {"pose", "keyframes", "spin", "shake", "knot_spacing_ms", "perturbation_key_ms"})
        if "pose" in rs:
            t = parse_pose(where, rs["pose"])
            rs = {"keyframes": [{"t_ns": span[0], "q_wxyz": matrix_to_quat(t[:3, :3]).tolist(),
                                 "t_m": t[:3, 3].tolist(), "hold_s": (span[1] - span[0]) / NS}]}
        fit = b.trajectory(where, rs, span)
        fit.trajectory.pos = fit.trajectory.pos - origin
        trajectories.append(fit.trajectory)
        fit_meta[f"rig:{name}"] = fit.diagnostics
        rig_index[name] = len(trajectories) - 1

    cameras = []
    for i, cs in enumerate(spec["cameras"]):
        where = f"cameras[{i}]"
        _check_keys(where, cs, {"name", "preset", "rig", "T_rig_cam", "clock_offset_ns", "model", "width", "height",
                                "fx", "fy", "cx", "cy", "dist", "sensor", "pixel_pitch_m", "f_number"}, {"name", "rig"})
        if "preset" in cs and cs["preset"] not in CAMERA_PRESETS:
            raise SceneSpecError(f"{where}: unknown preset {cs['preset']!r}")
        params = dict(CAMERA_PRESETS[cs["preset"]]) if "preset" in cs else {}
        params.update({k: cs[k] for k in ("model", "width", "height", "fx", "fy", "cx", "cy", "dist", "sensor",
                                          "pixel_pitch_m", "f_number") if k in cs})
        if cs["rig"] not in rig_index:
            raise SceneSpecError(f"{where}: unknown rig {cs['rig']!r}")
        t_rig_cam = parse_pose(f"{where}.T_rig_cam", cs.get("T_rig_cam", {}))
        try:
            cameras.append(Camera(cs["name"], params["model"], int(params["width"]), int(params["height"]),
                                  float(params["fx"]), float(params["fy"]), float(params["cx"]), float(params["cy"]),
                                  np.asarray(params["dist"], np.float64), rig_index[cs["rig"]], t_rig_cam,
                                  int(cs.get("clock_offset_ns", 0)), params["sensor"], float(params["pixel_pitch_m"]),
                                  float(params.get("f_number", 2.0))))
        except KeyError as e:
            raise SceneSpecError(f"{where}: missing camera parameter {e}") from None

    lights = []
    for i, ls in enumerate(spec.get("lights") or []):
        where = f"lights[{i}]"
        _check_keys(where, ls, {"kind", "group", "color", "intensity", "body", "position_m", "direction",
                                "spot_inner_deg", "spot_outer_deg", "casts_shadows", "proxy", "size_m", "axis"},
                    {"kind", "intensity"})
        if ls["kind"] == "spot" and not {"spot_inner_deg", "spot_outer_deg"} <= set(ls):
            raise SceneSpecError(f"{where}: a spot needs spot_inner_deg and spot_outer_deg (0 <= inner < outer <= 180)")
        group = ls.get("group", "default")
        if group not in group_index:
            raise SceneSpecError(f"{where}: unknown light group {group!r}")
        body = ls.get("body")
        if body is not None and body not in body_index:
            raise SceneSpecError(f"{where}: unknown body {body!r}")
        pos = _vec(f"{where}.position_m", ls.get("position_m", [0, 0, 0]))
        if body is None:
            pos = pos - origin
        d = _vec(f"{where}.direction", ls.get("direction", [0, 0, -1]))
        color = _vec(f"{where}.color", ls.get("color", [1, 1, 1])).astype(np.float32)
        lights.append(Light(ls["kind"], group_index[group], color,
                            float(ls["intensity"]), None if body is None else body_index[body], pos,
                            d / max(np.linalg.norm(d), 1e-12), math.radians(float(ls.get("spot_inner_deg", 0.0))),
                            math.radians(float(ls.get("spot_outer_deg", 0.0))), bool(ls.get("casts_shadows", True)),
                            proxy=_bool(f"{where}.proxy", ls.get("proxy", False)),
                            size=_vec(f"{where}.size_m", ls["size_m"], 2) if "size_m" in ls else None,
                            axis=_unit(f"{where}.axis", ls["axis"]) if "axis" in ls else None))
    if any(light.proxy for light in lights):
        for body in bodies:
            if body.trajectory is None and not body.scan and \
                    any(body.materials[j].shading == "baked" for j in body.mesh.used_materials()):
                warnings.warn(f"body {body.name!r}: a static baked body that is not a scan, in a scene with proxy "
                              "lights: its texture holds its own capture's lighting, and it casts proxy shadows on "
                              "the scan as an inserted object (scans.md §4.2)", SceneWarning, stacklevel=2)

    meta = {"source": {"scene_yaml": str(path.resolve())},
            "assets": {name: {"path": str(b.path(a["path"])), "license": a["license"],
                              "attribution": a.get("attribution", ""), "cache_key": asset.info.get("cache_key"),
                              **({"restricted": True} if a.get("format") == "replica" or
                                 canonical(a["license"]) in RESTRICTED else {}),
                              **({"scale_basis": a["scale_basis"]} if "scale_basis" in a else {})}
                       for name, (asset, a) in assets.items()},
            "semantic_classes": classes, "fits": fit_meta, "user": spec.get("meta") or {}}
    bundle = SceneBundle(spec["name"], span[0], span[1], origin, bodies, trajectories, cameras, groups, lights, meta)
    try:
        return bundle.validate()
    except ValueError as e:
        raise SceneSpecError(str(e)) from None
