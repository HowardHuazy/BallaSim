"""PLY import: ascii and binary, vertex normals/UVs/colours, and the Artec multi-texture layout.

Artec Studio exports (EVIMO2 object scans, R6 §A6, S2 §2.1) carry two extra elements that generic
loaders drop:

    element multi_texture_vertex N   {uchar tx; float u; float v}
    element multi_texture_face   T   {uchar tx; uint tn; list uchar int texture_vertex_indices}

i.e. per-corner UVs into ``surface_<tx>.png``. Texture face i holds the corner UVs of geometric face
``tn[i]``: ``tn`` is a permutation of the face indices, the identity only in some files (box_00,
box_03, plane_00/01, checkerboard) and a shuffle in most EVIMO2 scans (toy_00, turntable, the drones,
wheels, markers, ...), where ignoring it scrambles the UVs (5.8-6.0 UV variants per vertex position
instead of 1.00-1.15). Vertices are split per unique (position, UV) wedge, after the normals were
computed on the welded mesh (Artec files carry none), so seams do not crease the shading. The UV
origin is the top of the atlas (row = v * H): with that convention 99.8 % of box_00's face centroids
land on painted atlas texels, against 61 % (chance) for the flipped one (AST-22, decided by atlas
coverage).

Written from the PLY format description; no third-party PLY code is used. The reader is C++ since S5
(``ballasim._scene.load_ply``, src/scene/assets/ply.cpp; docs/dev/CPP_PORT.md item 5): this module is its Python face.
"""

from __future__ import annotations

from pathlib import Path

from ballasim.scene.bundle import Material, Mesh


def load_ply(path: str | Path, *, scale: float, uv_origin: str = "bottom") -> tuple[Mesh, list[Material], dict]:
    """PLY -> (mesh in metres, materials, info). ``scale`` converts file units to metres and is
    mandatory (PLY carries no units, AST-06). ``uv_origin`` is where v = 0 lies for plain PLY UVs
    ("bottom", the OpenGL convention most exporters use, or "top"); Artec atlases are always
    "top" (module docstring).

    Vertex colours: a single colour throughout becomes the material's base colour (every
    vertex-coloured EVIMO2 file, e.g. the tabletop's (255, 255, 255), the VCGLIB exporter's
    default rather than a measured albedo). Varying colours are refused: C1 meshes carry no
    vertex colours, and a per-triangle texture bake cannot stand in for them, because the renderer
    averages the UVs of a cluster's sub-samples across triangles (DESIGN §7.3), which lands between
    the bake's per-triangle cells on any mesh finer than a pixel. Textures are 8-bit RGB PNGs (libpng; every EVIMO2
    atlas), other modes are refused."""
    from ballasim import _scene

    mesh, materials, info = _scene.load_ply(str(Path(path)), float(scale), uv_origin)
    return Mesh(*mesh), [Material(**m) for m in materials], info
