"""Replica scene import (docs/design/phase3/scans.md §3.5, §4.5; S5b-1/2): the Python glue of the C++ importer.

A Replica scene is a directory (``mesh.ply``, ``textures/<c>-color-ptex.hdr`` and ``parameters.json``,
``habitat/sorted_faces.bin``, ``glass.sur``). ``_scene.import_replica`` (src/scene/assets/replica.cpp) maps each quad to
its Ptex tile, box-means the tiles to T x T binary16 texels with a G = T / 2 gutter and packs them into one tile atlas;
this module reads the JSON side file that C++ does not, ``textures/parameters.json``, and lists the files that key the
cache. The scene is metres, Z up, and radiance: every material is baked, K = 1 (the licence binds every output, §3.7).
"""

from __future__ import annotations

import json
from pathlib import Path

from ballasim.scene.bundle import Material, Mesh

from .geometry import AssetError

#: The cache key's importer version (scans.md §3.5): Replica entries do not follow IMPORTER_VERSION.
REPLICA_VERSION = "2"  # neighbour gutters replace v1 clamped gutters; old cache entries must not be reused
TILE_TEXELS = ("auto", 4, 8, 16)
#: The tile side of Replica's Ptex atlases (``textures/parameters.json``: tileSize).
SOURCE_TILE = 16


def replica_files(root: Path) -> list[Path]:
    """The files an import reads, in key order: the mesh, the face order, the tile parameters, the atlases and, when
    present, ``glass.sur`` and ``habitat/info_semantic.json`` (S5b-2 glass/gravity reports)."""
    files = [root / "mesh.ply", root / "habitat" / "sorted_faces.bin", root / "textures" / "parameters.json"]
    files += sorted((root / "textures").glob("*-color-ptex.hdr"), key=lambda p: (len(p.name), p.name))
    glass = root / "glass.sur"
    info = root / "habitat" / "info_semantic.json"
    return files + ([glass] if glass.exists() else []) + ([info] if info.exists() else [])


def check_dir(root: Path) -> None:
    if not root.is_dir():
        raise AssetError(f"{root}: a Replica scene is a directory (mesh.ply, textures/, habitat/)")
    for rel in ("mesh.ply", "habitat/sorted_faces.bin", "textures/parameters.json"):
        if not (root / rel).is_file():
            raise AssetError(f"{root}: missing {rel}" + (" (without it the faces cannot be mapped to the atlases)"
                                                         if "sorted" in rel else ""))
    if not any((root / "textures").glob("*-color-ptex.hdr")):
        raise AssetError(f"{root}: no textures/<chunk>-color-ptex.hdr atlas")


def check_tile_texels(value) -> None:
    if value != "auto" and (isinstance(value, bool) or not isinstance(value, int) or value not in TILE_TEXELS):
        raise AssetError(f"tile_texels must be one of {TILE_TEXELS}, got {value!r}")


def read_parameters(root: Path) -> tuple[int, float]:
    """(tileSize, splitSize) of ``textures/parameters.json``; tileSize must be 16."""
    path = root / "textures" / "parameters.json"
    try:
        doc = json.loads(path.read_text())
        tile, split = doc["tileSize"], doc["splitSize"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise AssetError(f"{path}: not a Replica parameters file with tileSize and splitSize ({e})") from None
    if isinstance(tile, bool) or tile != SOURCE_TILE:
        raise AssetError(f"{path}: tileSize {tile!r}, but the importer maps {SOURCE_TILE} x {SOURCE_TILE} tiles")
    if isinstance(split, bool) or not isinstance(split, (int, float)):
        raise AssetError(f"{path}: splitSize must be a number, got {split!r}")
    return int(tile), float(split)


def import_replica(root: Path, tile_texels: int | str = "auto", workers: int = 1) -> tuple[Mesh, list[Material], dict]:
    """The scene directory ``root`` as (mesh, [one baked tile-atlas material], info): see the module docstring.
    ``tile_texels`` is T: "auto" (8 up to 2.5 M quads, else 4), 4, 8 or 16."""
    from ballasim import _scene

    check_dir(root)
    check_tile_texels(tile_texels)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise AssetError("workers must be a positive integer")
    tile, split = read_parameters(root)
    mesh, materials, info = _scene.import_replica(str(root), tile, split, 0 if tile_texels == "auto" else tile_texels, workers)
    try:
        glass_path = root / "glass.sur"
        glass = json.loads(glass_path.read_text()) if glass_path.exists() else []
        meta_path = root / "habitat" / "info_semantic.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if not isinstance(glass, list) or not isinstance(meta, dict):
            raise ValueError("glass.sur must be a list and info_semantic.json a mapping")
        info.update(_scene.replica_reports([entry["boundary_w"] for entry in glass], meta.get("gravity_dir")))
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise AssetError(f"{root}: invalid Replica glass/gravity side file ({e})") from None
    return Mesh(*mesh), [Material(**m) for m in materials], info
