"""Mesh validation, repair and small geometry utilities shared by the asset importers.

Validation follows S2 §2.4 for the checks Phase 1 needs: non-finite attributes are refused
(AST-V01); degenerate (AST-V02) and duplicate (AST-V04) triangles and unreferenced vertices
(AST-V06) are removed; missing normals are angle-weighted vertex normals (AST-V11). Every repair
is counted in the returned report, so nothing is changed silently.

Importers that split vertices at attribute seams (Artec UV wedges, OBJ (v, vt) corners) compute
missing normals on the welded mesh first (:func:`welded_normals`): computed after the split, the
two copies of a seam vertex would each see only their own side's faces, and shading would crease
along every seam (DESIGN §7.10).
"""

from __future__ import annotations

import numpy as np

from ballasim.scene.bundle import Material, Mesh

SRGB_LINEAR_KNEE = 0.04045


class AssetError(ValueError):
    """An asset cannot be imported faithfully."""


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.float64)
    return np.where(c <= SRGB_LINEAR_KNEE, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


def triangle_areas(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    p = positions.astype(np.float64)
    return 0.5 * np.linalg.norm(np.cross(p[indices[:, 1]] - p[indices[:, 0]], p[indices[:, 2]] - p[indices[:, 0]]),
                                axis=1)


def vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Angle-weighted vertex normals (Thuermer & Wuethrich), float32, unit length (or 0)."""
    p = positions.astype(np.float64)
    tri = p[indices]                                           # (T, 3, 3)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)
    out = np.zeros_like(p)
    for k in range(3):
        a = tri[:, (k + 1) % 3] - tri[:, k]
        b = tri[:, (k + 2) % 3] - tri[:, k]
        cosang = np.sum(a * b, 1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
        w = np.arccos(np.clip(cosang, -1.0, 1.0))
        for j in range(3):
            out[:, j] += np.bincount(indices[:, k], weights=w * fn[:, j], minlength=len(p))
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return np.where(n > 0, out / np.maximum(n, 1e-300), 0.0).astype(np.float32)


def kept_faces(positions: np.ndarray, indices: np.ndarray, rel_eps: float = 1e-7) -> tuple[np.ndarray, int, int]:
    """The triangles validation keeps: not degenerate (a repeated index, or area below
    (rel_eps L)^2 with L the bounding-box diagonal) and the first of every duplicate (same vertex
    set in either winding). Returns (mask, degenerate count, duplicate count)."""
    idx = np.asarray(indices, np.int64)
    diag = float(np.linalg.norm(positions.max(axis=0).astype(np.float64) - positions.min(axis=0)))
    repeated = (idx[:, 0] == idx[:, 1]) | (idx[:, 1] == idx[:, 2]) | (idx[:, 0] == idx[:, 2])
    keep = ~(repeated | (triangle_areas(positions, idx) < (rel_eps * diag) ** 2))
    _, first = np.unique(np.sort(idx[keep], axis=1), axis=0, return_index=True)
    mask = np.zeros(len(idx), bool)
    mask[np.flatnonzero(keep)[first]] = True
    return mask, int((~keep).sum()), int(keep.sum() - mask.sum())


def welded_normals(positions: np.ndarray, indices: np.ndarray, rel_eps: float = 1e-7) -> np.ndarray:
    """Angle-weighted normals of a mesh indexed by position, before any per-corner attribute split,
    over the triangles :func:`validate_mesh` keeps (an opposite-winding duplicate would cancel its
    twin's contribution). Importers gather them by position index, so every copy of a seam vertex
    carries the same normal."""
    keep, _, _ = kept_faces(positions, indices, rel_eps)
    return vertex_normals(positions, np.asarray(indices, np.int64)[keep])


def validate_mesh(mesh: Mesh, *, rel_eps: float = 1e-7, compute_normals: bool = True):
    """Refuse NaN/Inf, drop degenerate and duplicate triangles and unreferenced vertices, and add
    angle-weighted normals when missing. Returns the repaired mesh and a report of every repair."""
    for name in ("positions", "normals", "uvs"):
        a = getattr(mesh, name)
        if a is not None and not np.all(np.isfinite(a)):
            raise AssetError(f"AST-V01: mesh {name} contain NaN or Inf")
    pos, idx = mesh.positions, mesh.indices.astype(np.int64)
    if len(idx) == 0:
        raise AssetError("mesh has no triangles")
    if idx.min() < 0 or idx.max() >= len(pos):
        raise AssetError("triangle index out of range")
    report = {"input_vertices": len(pos), "input_triangles": len(idx)}
    unique_mask, report["degenerate_removed"], report["duplicate_removed"] = kept_faces(pos, idx, rel_eps)
    faces = idx[unique_mask]
    mat = mesh.material_ids[unique_mask] if mesh.material_ids is not None else None
    used = np.zeros(len(pos), bool)
    used[faces.ravel()] = True
    report["unreferenced_removed"] = int((~used).sum())
    remap = np.cumsum(used) - 1
    out = Mesh(np.ascontiguousarray(pos[used], np.float32), np.ascontiguousarray(remap[faces], np.uint32),
               None if mesh.normals is None else np.ascontiguousarray(mesh.normals[used], np.float32),
               None if mesh.uvs is None else np.ascontiguousarray(mesh.uvs[used], np.float32),
               None if mat is None else np.ascontiguousarray(mat, np.uint16))
    if out.normals is None and compute_normals:
        out.normals = vertex_normals(out.positions, out.indices)
        report["normals"] = "computed (angle-weighted)"
    elif out.normals is not None:
        n = np.linalg.norm(out.normals, axis=1, keepdims=True)
        out.normals = np.where(n > 0, out.normals / np.maximum(n, 1e-30), 0.0).astype(np.float32)
        report["normals"] = "given by the importer"
    report.update(vertices=len(out.positions), triangles=len(out.indices))
    return out, report


def transform_mesh(mesh: Mesh, t: np.ndarray) -> Mesh:
    """Apply a 4x4 affine transform (rotation, translation, uniform or general scale); normals use
    the inverse transpose and the winding flips when the transform mirrors (AST-08)."""
    t = np.asarray(t, np.float64)
    a = t[:3, :3]
    pos = (mesh.positions.astype(np.float64) @ a.T + t[:3, 3]).astype(np.float32)
    idx = mesh.indices
    if np.linalg.det(a) < 0:
        idx = np.ascontiguousarray(idx[:, [0, 2, 1]])
    normals = None
    if mesh.normals is not None:
        nt = np.linalg.inv(a).T
        n = mesh.normals.astype(np.float64) @ nt.T
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
        normals = n.astype(np.float32)
    return Mesh(pos, idx, normals, mesh.uvs, mesh.material_ids)


def merge_meshes(parts: list[tuple[Mesh, list[Material]]]) -> tuple[Mesh, list[Material]]:
    """Concatenate meshes with their materials; material ids are offset per part. Normals and UVs
    are kept when every part has them (UVs are zero-filled for untextured parts)."""
    if not parts:
        raise AssetError("nothing to merge")
    pos, idx, nrm, uvs, mids, mats = [], [], [], [], [], []
    offset = 0
    has_normals = all(m.normals is not None for m, _ in parts)
    has_uvs = any(m.uvs is not None for m, _ in parts)
    for mesh, materials in parts:
        pos.append(mesh.positions)
        idx.append(mesh.indices.astype(np.int64) + offset)
        if has_normals:
            nrm.append(mesh.normals)
        if has_uvs:
            uvs.append(mesh.uvs if mesh.uvs is not None else np.zeros((len(mesh.positions), 2), np.float32))
        local = mesh.material_ids if mesh.material_ids is not None else np.zeros(len(mesh.indices), np.uint16)
        mids.append(local.astype(np.int64) + len(mats))
        mats.extend(materials)
        offset += len(mesh.positions)
    if len(mats) > 0xFFFF:
        raise AssetError("more than 65535 materials")
    material_ids = np.concatenate(mids).astype(np.uint16)
    return Mesh(np.concatenate(pos).astype(np.float32), np.concatenate(idx).astype(np.uint32),
                np.concatenate(nrm).astype(np.float32) if has_normals else None,
                np.concatenate(uvs).astype(np.float32) if has_uvs else None,
                None if len(mats) == 1 else material_ids), mats


def area_centroid(mesh: Mesh) -> np.ndarray:
    """Area-weighted surface centroid (the CoM proxy of DESIGN §5.1), fp64."""
    p = mesh.positions.astype(np.float64)
    a = triangle_areas(p, mesh.indices)
    c = p[mesh.indices].mean(axis=1)
    return (a[:, None] * c).sum(axis=0) / a.sum()


def bounding_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """A cheap enclosing sphere: AABB centre and the max distance to it (fp64)."""
    p = np.asarray(points, np.float64)
    c = 0.5 * (p.min(axis=0) + p.max(axis=0))
    return c, float(np.linalg.norm(p - c, axis=1).max())


def icosphere(level: int, radius: float = 1.0) -> Mesh:
    """Geodesic sphere by midpoint subdivision of an icosahedron (20 * 4^level triangles)."""
    g = (1.0 + 5.0**0.5) / 2.0
    v = np.array([[-1, g, 0], [1, g, 0], [-1, -g, 0], [1, -g, 0], [0, -1, g], [0, 1, g], [0, -1, -g], [0, 1, -g],
                  [g, 0, -1], [g, 0, 1], [-g, 0, -1], [-g, 0, 1]], np.float64)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                  [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
                  [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], np.int64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(level):
        edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        uniq, inv = np.unique(edges, axis=0, return_inverse=True)
        mid = v[uniq[:, 0]] + v[uniq[:, 1]]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        m = len(v) + inv.reshape(3, -1).T                     # midpoints of edges 01, 12, 20
        v = np.vstack([v, mid])
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        m01, m12, m20 = m[:, 0], m[:, 1], m[:, 2]
        f = np.concatenate([np.stack([a, m01, m20], 1), np.stack([b, m12, m01], 1),
                            np.stack([c, m20, m12], 1), np.stack([m01, m12, m20], 1)])
    return Mesh((v * radius).astype(np.float32), f.astype(np.uint32), v.astype(np.float32))

