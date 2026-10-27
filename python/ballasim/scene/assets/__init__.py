"""Asset import: glTF 2.0, OBJ + MTL, PLY (incl. Artec multi-texture and uniform vertex colours) and Replica scenes.

:func:`load_asset` returns an :class:`Asset` (C1 mesh + materials in the asset frame, metres,
+Z up) after unit/axis normalization, validation (S2 §2.4) and texture size limiting, and caches
the result by content hash (S2 AST-02): key = SHA-256 of the input files' bytes, the importer
version and the canonical import options. Entries live under ``$BALLASIM_CACHE/v1/assets/`` (default
``/data/ballasim_cache``) as ``mesh.npz`` + ``materials.json`` + texture PNGs and
are published atomically (temporary directory, then rename).

Baked radiance (docs/design/phase3/scans.md §3.5, the C1 feature "baked"): ``shading`` "auto" (the default) honours
glTF ``KHR_materials_unlit``, "baked" bakes every material, "lit" reads every material lit. The options key only an
import that yields baked materials (``"shading"`` joins them then), so no lit import's cache key moves and
IMPORTER_VERSION stays. A baked texture above ``max_texture_size`` is halved in linear light and rounded once to
binary16 (``Material.texture_linear``), never re-quantized to 8 bits; a baked entry stores it as
``texture_linear_<j>.npy`` and records the material's non-default feature fields in ``materials.json``.

The glTF option ``pbr`` (P2-F F4, the C1 feature "pbr"; off by default) reads the metallic-roughness BSDF (gltf.py).
Like ``shading``, it keys only an import that uses it (``"pbr": true``), so no existing cache key moves.

A Replica scene (``format="replica"``, scans.md §4.5; :mod:`.replica`) is a directory. It imports as one baked scan mesh
with a binary16 per-face tile atlas (``texture_tile``) at ``tile_texels`` texels per face side; its cache key lists the
scene's files and ``REPLICA_VERSION`` (not IMPORTER_VERSION), and ``scale`` is fixed at 1 (metres).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ballasim.scene.bundle import Material, Mesh, at_feature_default
from ballasim.scene.transforms import rotation_about

from .geometry import AssetError, linear_to_srgb, srgb_to_linear, transform_mesh, validate_mesh
from .gltf import SHADINGS, gltf_dependencies, gltf_uses_unlit, load_gltf, read_gltf_json
from .obj import load_obj, obj_dependencies
from .replica import REPLICA_VERSION, check_dir, check_tile_texels, import_replica, replica_files

__all__ = ["Asset", "AssetError", "asset_inputs", "cache_root", "cached_key", "file_sha256", "is_cached",
           "load_asset"]

#: Bumped whenever an importer's output changes, which invalidates every cache entry.
IMPORTER_VERSION = "3"
DEFAULT_CACHE = "/data/ballasim_cache"
FORMATS = ("gltf", "obj", "ply", "replica")
#: The C1 "baked" Material fields a cache entry's materials.json records when they are away from their defaults
#: (texture_linear is stored beside it as texture_linear_<j>.npy).
ENTRY_FEATURE_FIELDS = ("shading", "baked_scale", "baked_groups", "texture_tile", "roughness", "metallic", "specular")
UP_AXES = {"+z": np.eye(3), "+y": rotation_about("x", np.pi / 2), "-y": rotation_about("x", -np.pi / 2),
           "-z": rotation_about("x", np.pi), "+x": rotation_about("y", -np.pi / 2),
           "-x": rotation_about("y", np.pi / 2)}


@dataclass
class Asset:
    mesh: Mesh
    materials: list[Material]
    info: dict = field(default_factory=dict)


def cache_root() -> Path:
    return Path(os.environ.get("BALLASIM_CACHE", DEFAULT_CACHE)) / "v1" / "assets"


def _format_of(path: Path, fmt: str | None) -> str:
    if fmt is not None:
        if fmt not in FORMATS:
            raise AssetError(f"unknown asset format {fmt!r}; expected one of {FORMATS}")
        return fmt
    ext = path.suffix.lower()
    for f, exts in (("gltf", (".gltf", ".glb")), ("obj", (".obj",)), ("ply", (".ply",))):
        if ext in exts:
            return f
    raise AssetError(f"cannot infer the format of {path}" +
                     (" (a directory: pass format='replica' for a Replica scene)" if path.is_dir() else ""))


def _dependencies(path: Path, fmt: str, gltf_doc: dict | None = None) -> list[Path]:
    if fmt == "gltf":
        return gltf_dependencies(path, gltf_doc)
    if fmt == "obj":
        return obj_dependencies(path)
    atlas = path.with_name("surface_0.png")
    return [atlas] if atlas.exists() else []


_SHA_MEMO: dict | None = None


def _sha_memo_path() -> Path:
    return cache_root().parent / "file_sha256.json"


def file_sha256(path: str | os.PathLike) -> str:
    """SHA-256 of a file's bytes, memoized by (real path, size, mtime) in memory and in
    ``$BALLASIM_CACHE/v1/file_sha256.json``: an unchanged file is read once (the cache keys of assets and EVIMO2
    bundles hash every input file)."""
    global _SHA_MEMO
    real = os.path.realpath(path)
    st = os.stat(real)
    stamp = [st.st_size, st.st_mtime_ns]
    if _SHA_MEMO is None:
        try:
            _SHA_MEMO = json.loads(_sha_memo_path().read_text())
        except (OSError, ValueError):
            _SHA_MEMO = {}
    hit = _SHA_MEMO.get(real)
    if hit is not None and hit[:2] == stamp:
        return hit[2]
    h = hashlib.sha256()
    with open(real, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    _SHA_MEMO[real] = [*stamp, h.hexdigest()]
    try:                                    # best effort: a lost update (concurrent writers) costs one re-read
        memo = _sha_memo_path()
        memo.parent.mkdir(parents=True, exist_ok=True)
        tmp = memo.with_name(f".{memo.name}.{os.getpid()}")
        tmp.write_text(json.dumps(_SHA_MEMO, separators=(",", ":")))
        os.replace(tmp, memo)
    except OSError:
        pass
    return _SHA_MEMO[real][2]


def _inputs(path: Path, fmt: str, gltf_doc: dict | None = None) -> list[dict]:
    if fmt == "replica":                    # names relative to the scene directory
        check_dir(path)
        return [{"name": os.path.relpath(p, path), "sha256": file_sha256(p), "size": p.stat().st_size}
                for p in replica_files(path)]
    files = [path] + sorted(set(_dependencies(path, fmt, gltf_doc)) - {path})
    return [{"name": p.name if i == 0 else os.path.relpath(p, path.parent), "sha256": file_sha256(p),
             "size": p.stat().st_size} for i, p in enumerate(files)]


def asset_inputs(path: str | os.PathLike, *, format: str | None = None) -> list[dict]:
    """The files an import of ``path`` reads (the mesh, then its textures), as load_asset's cache key records them:
    ``name`` (the mesh's file name, the others relative to it), ``sha256`` and ``size``."""
    path = Path(path).resolve()
    return _inputs(path, _format_of(path, format))


def _halve(lin: np.ndarray) -> np.ndarray:
    """One 2x2 box halving in linear light (edge-padded to even sides)."""
    h, w = lin.shape[:2]
    lin = np.pad(lin, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
    return 0.25 * (lin[0::2, 0::2] + lin[1::2, 0::2] + lin[0::2, 1::2] + lin[1::2, 1::2])


def _downsample(tex: np.ndarray, max_size: int) -> tuple[np.ndarray, int]:
    """Halve in linear light (2x2 box) until the largest side is <= max_size; returns (tex, levels)."""
    levels = 0
    lin = None
    while max(tex.shape[:2]) > max_size:
        if lin is None:
            lin = srgb_to_linear(tex.astype(np.float64) / 255.0)
        lin = _halve(lin)
        tex = np.rint(linear_to_srgb(lin) * 255.0).astype(np.uint8)
        levels += 1
    return tex, levels


def _downsample_linear(tex: np.ndarray, max_size: int) -> tuple[np.ndarray | None, int]:
    """The halvings of _downsample for a baked texture, kept in fp64 linear light and rounded once to binary16
    (NumPy's float64 -> float16 cast rounds to nearest even); returns (texture_linear, levels), or (None, 0) when
    no side exceeds max_size (the 8-bit texture stays). The C++ PLY import does the same (ply.cpp)."""
    if max(tex.shape[:2]) <= max_size:
        return None, 0
    lin = srgb_to_linear(tex.astype(np.float64) / 255.0)
    levels = 0
    while max(lin.shape[:2]) > max_size:
        lin = _halve(lin)
        levels += 1
    return lin.astype(np.float16), levels


def _bake(materials: list[Material], info: dict, emission: str) -> None:
    """shading "baked" on an OBJ: every material is radiance with K = 1, group 0; its emission (``emission``, the
    format's name for it) is zeroed and recorded in info["ignored"], as glTF's unlit import does."""
    for m in materials:
        m.shading = "baked"
        if np.any(m.emissive != 0):
            m.emissive = np.zeros(3, np.float32)
            info["ignored"] = sorted(set(info.get("ignored", [])) | {emission})
    info["baked_materials"] = list(range(len(materials)))


def _import(path: Path, fmt: str, opts: dict, shading: str) -> Asset:
    if fmt == "replica":                     # C++ (:mod:`.replica`); opts: tile_texels, workers (not keyed)
        mesh, materials, info = import_replica(path, opts["tile_texels"], opts["workers"])
        return Asset(mesh, materials, info)
    if fmt == "ply":
        return _import_ply(path, opts, shading == "baked")
    if fmt == "gltf":
        mesh, materials, info = load_gltf(path, node=opts["node"], shading=shading, pbr=opts.get("pbr", False))
    else:
        mesh, materials, info = load_obj(path, scale=1.0, baked=shading == "baked")
        if shading == "baked":
            _bake(materials, info, "Ke")
    # Units and axes (AST-06, AST-07): glTF is already +Z up inside load_gltf.
    t = np.eye(4)
    t[:3, :3] = opts["scale"] * (np.eye(3) if fmt == "gltf" else UP_AXES[opts["up_axis"]])
    if not np.allclose(t, np.eye(4)):
        mesh = transform_mesh(mesh, t)
    mesh, report = validate_mesh(mesh)
    shrunk = []
    for j, m in enumerate(materials):
        if m.texture is not None:
            if m.shading == "baked":
                lin, levels = _downsample_linear(m.texture, opts["max_texture_size"])
                if lin is not None:
                    m.texture, m.texture_linear = None, lin
            else:
                m.texture, levels = _downsample(m.texture, opts["max_texture_size"])
            if levels:
                shrunk.append({"material": j, "halvings": levels})
    info.update(validation=report, textures_downsampled=shrunk)
    return Asset(mesh, materials, info)


def _import_ply(path: Path, opts: dict, baked: bool = False) -> Asset:
    """The import of a PLY in C++ (``ballasim._scene.import_ply``, docs/dev/CPP_PORT.md item 5, S5), bitwise as
    :func:`_import` was: load_ply at scale 1, the units/axes transform, validate_mesh and texture limiting. ``baked``:
    every material baked, textures halved to binary16 linear RGB (as :func:`_downsample_linear`)."""
    from ballasim import _scene

    t = np.eye(4)
    t[:3, :3] = opts["scale"] * UP_AXES[opts["up_axis"]]
    mesh, materials, info = _scene.import_ply(str(path), t, opts["uv_origin"], opts["max_texture_size"], baked)
    return Asset(Mesh(*mesh), [Material(**m) for m in materials], info)


@dataclass
class _Keyed:
    """An import's cache identity: the resolved path, format, options, shading, manifest, key and entry directory."""

    path: Path
    fmt: str
    opts: dict
    shading: str
    manifest: dict
    key: str
    entry: Path


def _keyed(path: str | os.PathLike, format: str | None, scale: float | None, up_axis: str, node: str | None,
           uv_origin: str, max_texture_size: int, shading: str, pbr: bool = False, tile_texels: int | str = "auto",
           workers: int = 1) -> _Keyed:
    """load_asset's checks and cache key, without importing (the input files are hashed through file_sha256)."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    fmt = _format_of(path, format)
    if fmt == "replica":
        return _keyed_replica(path, scale, up_axis, node, shading, pbr, tile_texels, workers)
    if tile_texels != "auto":
        raise AssetError(f"tile_texels applies to Replica scenes (format='replica'), not {fmt}")
    if scale is None:
        if fmt != "gltf":
            raise AssetError(f"{path}: {fmt.upper()} has no units; pass scale (file units -> metres)")
        scale = 1.0
    if not np.isfinite(scale) or scale <= 0:
        raise AssetError(f"scale must be positive, got {scale}")
    if up_axis not in UP_AXES:
        raise AssetError(f"up_axis must be one of {sorted(UP_AXES)}")
    if shading not in SHADINGS:
        raise AssetError(f"shading must be one of {SHADINGS}, got {shading!r}")
    if not isinstance(pbr, bool):
        raise AssetError(f"pbr must be true or false, got {pbr!r}")
    if pbr and fmt != "gltf":
        raise AssetError(f"{path}: pbr is a glTF import option ({fmt.upper()} materials carry no metallic-roughness)")
    if int(max_texture_size) < 1:                     # halving never takes a side below 1
        raise AssetError(f"max_texture_size must be >= 1, got {max_texture_size}")
    opts = {"scale": float(scale), "up_axis": up_axis, "node": node, "uv_origin": uv_origin,
            "max_texture_size": int(max_texture_size)}
    # The glTF JSON is read once for the key (a .glb's JSON chunk only): its unlit flag and its external files.
    gltf_doc = read_gltf_json(path) if fmt == "gltf" else None
    # Only an import that yields baked materials keys its shading: every lit import keeps its key (scans.md §3.5).
    if shading == "baked" or (shading == "auto" and gltf_doc is not None and gltf_uses_unlit(gltf_doc)):
        opts["shading"] = shading
    if pbr:                                           # likewise: only an import that reads the BSDF keys it
        opts["pbr"] = True
    manifest = {"kind": "asset", "importer": fmt, "importer_version": IMPORTER_VERSION, "options": opts,
                "inputs": _inputs(path, fmt, gltf_doc), "format": 1}
    return _key_of(path, fmt, opts, shading, manifest)


def _key_of(path: Path, fmt: str, opts: dict, shading: str, manifest: dict) -> _Keyed:
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return _Keyed(path, fmt, opts, shading, manifest, key, cache_root() / key[:2] / key)


def _keyed_replica(path: Path, scale, up_axis: str, node, shading: str, pbr: bool, tile_texels, workers: int) -> _Keyed:
    """A Replica scene directory (scans.md §3.5, §4.5): keyed ``{"importer": "replica", "replica_version", "options":
    {tile_texels, semantic_bodies}, "inputs"}``, so IMPORTER_VERSION bumps leave its entries alone; the import is C++
    (:mod:`.replica`)."""
    if scale is not None and scale != 1.0:
        raise AssetError(f"{path}: a Replica scene is in metres: scale is fixed at 1, got {scale}")
    if up_axis != "+z" or node is not None:
        raise AssetError(f"{path}: a Replica scene is Z up and has no nodes (up_axis and node are not used)")
    if shading not in ("auto", "baked"):
        raise AssetError(f"{path}: a Replica scene is baked radiance: shading must be auto or baked, got {shading!r}")
    if not isinstance(pbr, bool):
        raise AssetError(f"pbr must be true or false, got {pbr!r}")
    if pbr:
        raise AssetError(f"{path}: pbr is a glTF import option (a Replica scene carries no metallic-roughness)")
    check_tile_texels(tile_texels)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise AssetError("workers must be a positive integer")
    manifest = {"kind": "asset", "importer": "replica", "replica_version": REPLICA_VERSION,
                "options": {"tile_texels": tile_texels, "semantic_bodies": False},   # S6 turns the second on
                "inputs": _inputs(path, "replica"), "format": 1}
    return _key_of(path, "replica", {"tile_texels": tile_texels, "workers": workers}, shading, manifest)


def load_asset(path: str | os.PathLike, *, format: str | None = None, scale: float | None = None,
               up_axis: str = "+z", node: str | None = None, uv_origin: str = "bottom",
               max_texture_size: int = 4096, shading: str = "auto", pbr: bool = False,
               tile_texels: int | str = "auto", workers: int = 1, use_cache: bool = True) -> Asset:
    """Import a mesh asset into the asset frame (metres, +Z up).

    scale: file units -> metres; mandatory for OBJ and PLY (they carry no units, AST-06); for glTF
      (metres by definition) an optional extra factor, e.g. for unitless Sketchfab exports.
    up_axis: the file's up axis for OBJ/PLY ("+z" = no rotation, "+y" rotates Y-up to Z-up).
    node: glTF only, import just the subtree of this node.
    uv_origin: plain-PLY UV convention ("bottom" = OpenGL v up, "top").
    max_texture_size: textures above it are halved in linear light (recorded in the info); baked ones to binary16
      ``texture_linear``.
    shading: "auto" (glTF KHR_materials_unlit materials become baked, everything else lit), "lit" (every material
      lit, the pre-Phase-3 import) or "baked" (every material baked radiance with K = 1, all in light group 0).
    pbr: glTF only, read the lit materials' metallic-roughness BSDF (the C1 feature "pbr"; gltf.py).
    tile_texels: Replica only (``format="replica"``, a scene directory): the texels per face side T of the packed tile
      atlas, "auto" (8 up to 2.5 M quads, else 4), 4, 8 or 16 (scans.md §4.5). A Replica import is always baked, in
      metres and Z up: ``scale`` is fixed at 1 and ``max_texture_size`` is not used (T replaces it).
    workers: Replica gutter packing threads (positive integer, default 1); changes no output or cache key.
    """
    k = _keyed(path, format, scale, up_axis, node, uv_origin, max_texture_size, shading, pbr, tile_texels, workers)
    if use_cache and (k.entry / "entry.json").exists():
        asset = _read_entry(k.entry)
    else:
        asset = _import(k.path, k.fmt, k.opts, k.shading)
        if use_cache:
            _write_entry(k.entry, asset, k.manifest)
    asset.info.update(source=str(k.path), cache_key=k.key, inputs=k.manifest["inputs"])
    if k.fmt == "replica":
        for message in asset.info.get("warnings", []):
            warnings.warn(message, UserWarning, stacklevel=2)
    return asset


def cached_key(path: str | os.PathLike, *, format: str | None = None, scale: float | None = None,
               up_axis: str = "+z", node: str | None = None, uv_origin: str = "bottom", max_texture_size: int = 4096,
               shading: str = "auto", pbr: bool = False, tile_texels: int | str = "auto",
               workers: int = 1) -> str | None:
    """The cache key :func:`load_asset` with these options would read rather than import, or None when its entry is
    missing (the UI's asset check, docs/design/phase3/ui.md §4.12, which also keys its bundle LRU by it): load_asset's
    key from the input files' SHA-256 values (memoized by real path, size and mtime in
    ``$BALLASIM_CACHE/v1/file_sha256.json``) and an existing ``entry.json``. A missing asset file is None (the import
    reports it); options load_asset refuses raise as there."""
    try:
        k = _keyed(path, format, scale, up_axis, node, uv_origin, max_texture_size, shading, pbr, tile_texels, workers)
    except FileNotFoundError:
        return None
    return k.key if (k.entry / "entry.json").exists() else None


def is_cached(path: str | os.PathLike, **options) -> bool:
    """Whether :func:`load_asset` with these options would read its cache entry rather than import
    (:func:`cached_key` is not None)."""
    return cached_key(path, **options) is not None


def _write_entry(entry: Path, asset: Asset, manifest: dict) -> None:
    entry.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{entry.name}.", dir=entry.parent))
    try:
        m = asset.mesh
        arrays = {"positions": m.positions, "indices": m.indices}
        for name in ("normals", "uvs", "material_ids"):
            if getattr(m, name) is not None:
                arrays[name] = getattr(m, name)
        np.savez(tmp / "mesh.npz", **arrays)
        mats = []
        for j, mat in enumerate(asset.materials):
            rec = {"base_color": mat.base_color.tolist(), "emissive": mat.emissive.tolist(), "texture": None}
            if mat.texture is not None:
                rec["texture"] = f"tex_{j}.png"
                Image.fromarray(mat.texture, "RGB").save(tmp / rec["texture"])
            for name in ENTRY_FEATURE_FIELDS:            # only away from their defaults: lit entries are unchanged
                if not at_feature_default(mat, name):
                    v = getattr(mat, name)
                    rec[name] = v.tolist() if isinstance(v, (np.ndarray, np.generic)) else v
            if mat.texture_linear is not None:
                rec["texture_linear"] = f"texture_linear_{j}.npy"
                np.save(tmp / rec["texture_linear"], mat.texture_linear, allow_pickle=False)
            mats.append(rec)
        (tmp / "materials.json").write_text(json.dumps(mats))
        (tmp / "entry.json").write_text(json.dumps({"manifest": manifest, "info": asset.info}, indent=1, default=str))
        try:
            tmp.rename(entry)
        except OSError:
            if not (entry / "entry.json").exists():  # a concurrent writer that did not finish
                raise
            shutil.rmtree(tmp)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _read_entry(entry: Path) -> Asset:
    with np.load(entry / "mesh.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    mesh = Mesh(arrays["positions"], arrays["indices"], arrays.get("normals"), arrays.get("uvs"),
                arrays.get("material_ids"))
    mats = []
    for rec in json.loads((entry / "materials.json").read_text()):
        tex = None
        if rec["texture"]:
            with Image.open(entry / rec["texture"]) as im:
                tex = np.ascontiguousarray(np.asarray(im.convert("RGB"), np.uint8))
        extra = {name: rec[name] for name in ENTRY_FEATURE_FIELDS if name in rec}   # absent: the defaults
        if "baked_groups" in extra:
            extra["baked_groups"] = np.array(extra["baked_groups"], np.float32)
        if rec.get("texture_linear"):
            extra["texture_linear"] = np.load(entry / rec["texture_linear"], allow_pickle=False)
        mats.append(Material(np.array(rec["base_color"], np.float32), tex, np.array(rec["emissive"], np.float32),
                             **extra))
    info = json.loads((entry / "entry.json").read_text())["info"]
    return Asset(mesh, mats, info)
