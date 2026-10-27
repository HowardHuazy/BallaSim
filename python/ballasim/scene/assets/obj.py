"""Wavefront OBJ + MTL import (YCB google_64k ``textured.obj``, Google Scanned Objects).

- Polygons are triangulated as fans; negative (relative) indices are supported.
- One output vertex per unique (v, vt, vn) corner. Without ``vn``, normals are computed on the
  mesh welded by ``v`` index before that split, so UV seams do not crease the shading.
- ``Kd`` is the linear base-colour factor and ``map_Kd`` its sRGB texture; other maps are ignored
  in Phase 1. Texture paths are resolved next to the MTL file first, then in the Gazebo/GSO
  layout ``../materials/textures/`` (GSO MTLs name ``texture.png`` that lives there).
- OBJ texture coordinates have v = 0 at the image bottom, so v is flipped to the C1 convention
  (v down, row = v * H).
- OBJ has no units: the caller passes ``scale`` (file units -> metres), AST-06.
- Per-vertex colours (the ``v x y z r g b`` extension) are dropped by a lit import, as before Phase 3; a baked one
  (``baked=True``, docs/design/phase3/scans.md §3.5) refuses any that are not white, as glTF COLOR_0 and PLY do
  (C1 meshes carry no vertex colours).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ballasim.scene.bundle import Material, Mesh

from .geometry import AssetError, welded_normals

TEXTURE_SEARCH = (".", "../materials/textures", "materials/textures", "textures")


def _parse_mtl(path: Path) -> dict[str, dict]:
    mats: dict[str, dict] = {}
    cur = None
    for raw in path.read_text(errors="replace").splitlines():
        words = raw.strip().split()
        if not words or words[0].startswith("#"):
            continue
        key = words[0]
        if key == "newmtl":
            cur = mats.setdefault(" ".join(words[1:]), {})
        elif cur is None:
            continue
        elif key == "Kd" and len(words) >= 4:
            cur["Kd"] = [float(x) for x in words[1:4]]
        elif key == "Ke" and len(words) >= 4:
            cur["Ke"] = [float(x) for x in words[1:4]]
        elif key == "map_Kd":
            # Options (-s, -o, -bm ...) precede the file name, which is the last token.
            cur["map_Kd"] = words[-1]
    return mats


def resolve_texture(mtl_dir: Path, name: str) -> Path:
    for sub in TEXTURE_SEARCH:
        cand = (mtl_dir / sub / name).resolve()
        if cand.exists():
            return cand
        cand = (mtl_dir / sub / Path(name).name).resolve()
        if cand.exists():
            return cand
    raise AssetError(f"texture {name!r} not found near {mtl_dir}")


def obj_dependencies(path: str | Path) -> list[Path]:
    """The MTL files and textures an OBJ references (for content hashing)."""
    path = Path(path)
    deps = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("mtllib"):
            for name in line.split()[1:]:
                mtl = path.parent / name
                if mtl.exists():
                    deps.append(mtl)
                    for m in _parse_mtl(mtl).values():
                        if "map_Kd" in m:
                            deps.append(resolve_texture(mtl.parent, m["map_Kd"]))
    return deps


def load_obj(path: str | Path, *, scale: float, baked: bool = False) -> tuple[Mesh, list[Material], dict]:
    path = Path(path)
    v, vt, vn = [], [], []
    rgb = []                                                  # the r g b of `v x y z r g b` lines (baked only)
    corners, face_mat = [], []
    mtl: dict[str, dict] = {}
    mtl_dirs: dict[str, Path] = {}
    mat_names: list[str] = []
    current = -1
    for raw in path.read_text(errors="replace").splitlines():
        if not raw or raw[0] == "#":
            continue
        words = raw.split()
        if not words:
            continue
        key = words[0]
        if key == "v":
            v.append(words[1:4])
            if baked and len(words) >= 7:
                rgb.append(words[4:7])
        elif key == "vt":
            vt.append(words[1:3])
        elif key == "vn":
            vn.append(words[1:4])
        elif key == "f":
            refs = []
            for w in words[1:]:
                parts = (w.split("/") + ["", ""])[:3]
                refs.append([int(x) if x else 0 for x in parts])
            for i in range(1, len(refs) - 1):
                corners.append([refs[0], refs[i], refs[i + 1]])
                face_mat.append(current)
        elif key == "usemtl":
            name = " ".join(words[1:])
            if name not in mat_names:
                mat_names.append(name)
            current = mat_names.index(name)
        elif key == "mtllib":
            for name in words[1:]:
                mpath = path.parent / name
                if mpath.exists():
                    for k, m in _parse_mtl(mpath).items():
                        mtl[k] = m
                        mtl_dirs[k] = mpath.parent
    if not corners:
        raise AssetError(f"{path}: no faces")
    if rgb:
        try:
            white = np.all(np.array(rgb, np.float64) == 1.0)
        except ValueError:
            white = False
        if not white:
            raise AssetError(f"{path}: vertex colours that are not white under shading baked; C1 meshes carry no "
                             "vertex colours (scans.md §3.5)")
    pos = np.array(v, np.float64)
    c = np.array(corners, np.int64)                           # (T, 3, 3): v, vt, vn per corner
    # OBJ indices are 1-based; negative ones count back from the end of the list so far. Lists are
    # complete when faces are resolved here, which is equivalent for files that only use indices
    # to earlier elements (the only valid use).
    for k, n in enumerate((len(v), len(vt), len(vn))):
        col = c[:, :, k]
        c[:, :, k] = np.where(col < 0, n + col, col - 1)
    has_vt = vt and np.all(c[:, :, 1] >= 0)
    has_vn = vn and np.all(c[:, :, 2] >= 0)
    key = c.reshape(-1, 3)[:, [0] + ([1] if has_vt else []) + ([2] if has_vn else [])]
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    positions = (pos[uniq[:, 0]] * scale).astype(np.float32)
    uvs = None
    col = 1
    if has_vt:
        t = np.array(vt, np.float64)[uniq[:, col]]
        uvs = np.stack([t[:, 0], 1.0 - t[:, 1]], axis=1).astype(np.float32)
        col += 1
    normals = np.array(vn, np.float32)[uniq[:, col]] if has_vn else welded_normals(pos, c[:, :, 0])[uniq[:, 0]]
    fm = np.array(face_mat, np.int64)
    materials = []
    if not mat_names or np.any(fm < 0):
        mat_names = ["<default>"] + mat_names
        fm = fm + 1
    for name in mat_names:
        m = mtl.get(name, {})
        tex = None
        if "map_Kd" in m:
            with Image.open(resolve_texture(mtl_dirs[name], m["map_Kd"])) as im:
                tex = np.ascontiguousarray(np.asarray(im.convert("RGB"), np.uint8))
        kd = m.get("Kd", [1.0, 1.0, 1.0] if tex is not None else [0.5, 0.5, 0.5])
        materials.append(Material(np.array(kd, np.float32), tex, np.array(m.get("Ke", [0, 0, 0]), np.float32)))
    if any(m.texture is not None for m in materials) and uvs is None:
        raise AssetError(f"{path}: textured material but no texture coordinates")
    mesh = Mesh(positions, inv.reshape(-1, 3).astype(np.uint32), normals, uvs,
                None if len(materials) == 1 else fm.astype(np.uint16))
    return mesh, materials, {"format": "obj", "scale": scale, "materials": mat_names,
                             "normals": "file" if has_vn else "angle-weighted on the welded mesh"}
