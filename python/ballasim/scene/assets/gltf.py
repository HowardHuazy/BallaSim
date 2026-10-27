"""glTF 2.0 import (.gltf with external or data-URI buffers, and .glb), written from the spec.

- The default scene's node hierarchy is flattened into one rigid body: every triangle primitive is
  transformed by its node's world matrix (normals by the inverse transpose; winding flipped for
  mirroring nodes) and merged, one material per glTF material.
- Materials: baseColorFactor (linear) and baseColorTexture (sRGB), emissiveFactor times
  KHR_materials_emissive_strength. Other PBR inputs (metallic-roughness, normal, occlusion,
  emissive textures) are ignored in Phase 1, and so are transmission/ior/specular extensions
  (recorded in the import info). alphaMode BLEND/MASK is imported as opaque: the Phase 1 renderer
  has no transparency (recorded too).
- KHR_materials_unlit (docs/design/phase3/scans.md §3.5, the C1 feature "baked"): under
  ``shading="auto"`` (the default) a material carrying it, and under ``shading="baked"`` every
  material, becomes ``shading="baked"`` with K = 1, all in light group 0: its radiance is
  baseColorFactor x baseColorTexture, the extension's Definition (color = baseColorTerm, no
  emissive term). Its emissiveFactor, KHR_materials_emissive_strength and emissiveTexture are
  zeroed and recorded in ``info["ignored"]`` (the README's fallback note zeroes emission too);
  ``info["baked_materials"]`` lists such materials. A baked primitive whose COLOR_0 is not white is
  refused (C1 meshes carry no vertex colours). ``shading="lit"`` reads every material lit, as
  before Phase 3, with the extension recorded as ignored.
- ``pbr`` (P2-F F4, the C1 feature "pbr"; off by default, so existing imports stay Lambert and keep their cache keys
  and digests): a lit material takes roughness r = roughnessFactor and metallic m = metallicFactor (glTF's defaults
  1 and 1), and specular s = mean_rgb(min(((ior - 1)/(ior + 1))^2 specularColorFactor, 1)) specularFactor
  (KHR_materials_ior, default ior 1.5, and KHR_materials_specular, defaults 1: s = 0.04), glTF's dielectric F0 as
  mono (p2f.md §3, §4.3). A metallicRoughnessTexture is averaged (its G and B channels, linear, scale r and m) and
  recorded in ``info["pbr_textures_averaged"]``; KHR_materials_specular's textures are ignored (``info["ignored"]``).
  glTF's dielectric F90 is specularFactor, BallaSim's min(1, 50 s): the materials where m + (1 - m) F90 differ are
  listed in ``info["pbr_f90"]``. Unlit and baked materials take no BSDF.
- KHR_texture_transform on the base-colour texture is baked into the UVs exactly as the spec's
  matrix T * R * S (rotation counter-clockwise in UV space); its texCoord override is honoured.
- KHR_mesh_quantization data is dequantized through the accessor's `normalized` flag.
- Morph targets are applied once with the node's (else the mesh's) default weights: the result is
  one fixed shape (Poly Haven's rim ships its rest shape as a target with weight 1). Animations
  are ignored, motion comes from BallaSim trajectories (S2 §2.1).
- Refused: skins (rigid bodies only, SCN-02), Draco/meshopt compression, sparse accessors, any
  other extension listed in extensionsRequired (KHR_materials_unlit is supported), and a GLB whose
  JSON chunk is not its first and only one.
- glTF is metres and +Y up; the result is rotated to BallaSim's +Z up by R_x(+90 deg) (AST-07).
  UVs already follow C1 (v down).
"""

from __future__ import annotations

import base64
import io
import json
import math
import struct
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image

from ballasim.scene.bundle import Material, Mesh

from .geometry import AssetError, merge_meshes, transform_mesh, vertex_normals

Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], np.float64)
UNLIT = "KHR_materials_unlit"
#: The material extensions the ``pbr`` option reads.
PBR_EXTENSIONS = {"KHR_materials_ior", "KHR_materials_specular"}
SUPPORTED_REQUIRED = {"KHR_texture_transform", "KHR_mesh_quantization", "KHR_materials_emissive_strength", UNLIT}
#: load_gltf's ``shading``: "auto" honours KHR_materials_unlit, "lit" ignores it, "baked" bakes every material.
SHADINGS = ("auto", "lit", "baked")
_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


_GLB_JSON, _GLB_BIN = 0x4E4F534A, 0x004E4942


def _check_json(path: Path, doc: dict) -> dict:
    version = str(doc.get("asset", {}).get("version", ""))
    if not version.startswith("2."):
        raise AssetError(f"{path}: glTF version {version!r} is not 2.x")
    missing = set(doc.get("extensionsRequired", [])) - SUPPORTED_REQUIRED
    if missing:
        raise AssetError(f"{path}: required extensions not supported: {sorted(missing)}")
    return doc


def _check_glb_version(path: Path, version: int) -> None:
    if version != 2:
        raise AssetError(f"{path}: GLB container version {version}")


def _glb_chunks(path: Path, f, length: int):
    """(type, length) of each chunk of a GLB of declared ``length`` whose 12-byte header has been read from the binary
    file ``f``, which is left at the chunk's data; the next chunk header is sought at its absolute offset, so the
    caller may read the data or not. The JSON chunk must be the first and only one (the GLB container's rule): the
    cache key (read_gltf_json) and the import (_Document) then always read the same JSON."""
    off = 12
    while off < length:
        f.seek(off)
        head = f.read(8)
        if len(head) < 8:
            raise AssetError(f"{path}: GLB truncated at a chunk header (offset {off})")
        clen, ctype = struct.unpack("<II", head)
        if (ctype == _GLB_JSON) != (off == 12):
            found = "a second JSON chunk" if ctype == _GLB_JSON else f"a first chunk of type {ctype:#010x}"
            raise AssetError(f"{path}: a GLB holds one JSON chunk, as its first chunk; found {found}")
        yield ctype, clen
        off += 8 + clen
    if off == 12:
        raise AssetError(f"{path}: GLB without a JSON chunk")


def read_gltf_json(path: str | Path) -> dict:
    """The checked glTF JSON alone, for cache keying: a .gltf's text, or a .glb's JSON chunk, reached by reading the
    12-byte header and the 8-byte chunk headers and seeking past any other chunk (the BIN chunk is never read); the
    chunk layout is checked as the import checks it (_glb_chunks)."""
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != b"glTF":
            return _check_json(path, json.loads((head + f.read()).decode("utf-8")))
        _, version, length = struct.unpack("<4sII", head)
        _check_glb_version(path, version)
        doc = None
        for ctype, clen in _glb_chunks(path, f, length):
            if ctype == _GLB_JSON:
                doc = json.loads(f.read(clen).decode("utf-8"))
    return _check_json(path, doc)


def _image_paths(path: Path, doc: dict) -> list[Path]:
    out = [path.parent / unquote(b["uri"]) for b in doc.get("buffers", [])
           if "uri" in b and not b["uri"].startswith("data:")]
    out += [path.parent / unquote(im["uri"]) for im in doc.get("images", [])
            if "uri" in im and not im["uri"].startswith("data:")]
    return out


class _Document:
    def __init__(self, path: Path):
        self.path = path
        self.bin_chunk: bytes | None = None
        raw = path.read_bytes()
        if raw[:4] == b"glTF":
            self.json = self._parse_glb(raw)
        else:
            self.json = json.loads(raw.decode("utf-8"))
        _check_json(path, self.json)
        self._buffers: dict[int, bytes] = {}

    def _parse_glb(self, raw: bytes) -> dict:
        _, version, length = struct.unpack_from("<4sII", raw, 0)
        _check_glb_version(self.path, version)
        doc, f = None, io.BytesIO(raw)
        for ctype, clen in _glb_chunks(self.path, f, length):
            if ctype == _GLB_JSON:
                doc = json.loads(f.read(clen).decode("utf-8"))
            elif ctype == _GLB_BIN:
                self.bin_chunk = f.read(clen)
        return doc

    def uri_bytes(self, uri: str) -> bytes:
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        return (self.path.parent / unquote(uri)).read_bytes()

    def buffer(self, i: int) -> bytes:
        if i not in self._buffers:
            b = self.json["buffers"][i]
            self._buffers[i] = self.bin_chunk if "uri" not in b else self.uri_bytes(b["uri"])
        return self._buffers[i]

    def accessor(self, i: int) -> np.ndarray:
        acc = self.json["accessors"][i]
        if "sparse" in acc:
            raise AssetError(f"{self.path}: sparse accessors are not supported")
        dtype = np.dtype(_COMPONENT[acc["componentType"]]).newbyteorder("<")
        ncomp = _NCOMP[acc["type"]]
        count = acc["count"]
        if "bufferView" not in acc:
            out = np.zeros((count, ncomp), dtype)
        else:
            view = self.json["bufferViews"][acc["bufferView"]]
            data = self.buffer(view["buffer"])
            base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
            elem = dtype.itemsize * ncomp
            stride = view.get("byteStride", elem)
            if stride == elem:
                out = np.frombuffer(data, dtype, count * ncomp, base).reshape(count, ncomp)
            else:
                rows = np.lib.stride_tricks.as_strided(
                    np.frombuffer(data, np.uint8, len(data) - base, base), (count, elem), (stride, 1))
                out = np.ascontiguousarray(rows).view(dtype).reshape(count, ncomp)
        if acc.get("normalized", False):
            info = np.iinfo(dtype)
            out = np.maximum(out.astype(np.float64) / info.max, -1.0)
        return out

    def image(self, i: int) -> np.ndarray:
        img = self.json["images"][i]
        if "uri" in img:
            data = self.uri_bytes(img["uri"])
        else:
            view = self.json["bufferViews"][img["bufferView"]]
            buf = self.buffer(view["buffer"])
            off = view.get("byteOffset", 0)
            data = buf[off: off + view["byteLength"]]
        with Image.open(io.BytesIO(data)) as im:
            return np.ascontiguousarray(np.asarray(im.convert("RGB"), np.uint8))

    def image_paths(self) -> list[Path]:
        return _image_paths(self.path, self.json)


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], np.float64).reshape(4, 4).T   # column-major in glTF
    t = np.array(node.get("translation", [0, 0, 0]), np.float64)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = np.array(node.get("scale", [1, 1, 1]), np.float64)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = r * s[None, :]
    m[:3, 3] = t
    return m


def texture_transform_matrix(ext: dict) -> np.ndarray:
    """KHR_texture_transform: uv' = T * R * S * uv, R = [[cos, -sin], [sin, cos]] (spec sample code,
    column-major mat3(cos, sin, 0, -sin, cos, 0, 0, 0, 1))."""
    ox, oy = ext.get("offset", [0.0, 0.0])
    rot = ext.get("rotation", 0.0)
    sx, sy = ext.get("scale", [1.0, 1.0])
    t = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], np.float64)
    c, s = math.cos(rot), math.sin(rot)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)
    sc = np.diag([sx, sy, 1.0])
    return t @ r @ sc


def _triangles(prim: dict, doc: _Document, count: int) -> np.ndarray | None:
    mode = prim.get("mode", 4)
    idx = doc.accessor(prim["indices"]).ravel().astype(np.int64) if "indices" in prim else np.arange(count)
    if mode == 4:
        return idx.reshape(-1, 3)
    if mode == 5:   # strip: alternate winding
        tri = np.stack([idx[:-2], idx[1:-1], idx[2:]], 1)
        tri[1::2] = tri[1::2][:, [1, 0, 2]]
        return tri
    if mode == 6:   # fan
        return np.stack([np.full(len(idx) - 2, idx[0]), idx[1:-1], idx[2:]], 1)
    return None     # points and lines are not surfaces


def gltf_dependencies(path: str | Path, doc: dict | None = None) -> list[Path]:
    """The external files (buffers, images) of the glTF at ``path``; ``doc``: its read_gltf_json, when already read."""
    path = Path(path)
    return _image_paths(path, read_gltf_json(path) if doc is None else doc)


def gltf_uses_unlit(doc: dict) -> bool:
    """Whether a glTF JSON (read_gltf_json) uses KHR_materials_unlit (listed in extensionsUsed, or on a material):
    under shading "auto" its import yields baked materials, which keys the asset cache apart (assets.load_asset)."""
    return UNLIT in doc.get("extensionsUsed", []) or any(UNLIT in m.get("extensions", {})
                                                         for m in doc.get("materials", []))


def _pbr_material(doc: _Document, m: dict, index: int, info: dict) -> dict:
    """The C1 "pbr" fields of glTF material ``m`` (our index ``index``), recording what the module docstring lists."""
    pbr = m.get("pbrMetallicRoughness", {})
    rough, metal = float(pbr.get("roughnessFactor", 1.0)), float(pbr.get("metallicFactor", 1.0))
    tinfo = pbr.get("metallicRoughnessTexture")
    if tinfo is not None:      # linear data: G roughness, B metallic
        tex = doc.image(doc.json["textures"][tinfo["index"]]["source"]).astype(np.float64) / 255.0
        rough, metal = rough * float(tex[..., 1].mean()), metal * float(tex[..., 2].mean())
        info.setdefault("pbr_textures_averaged", []).append(index)
    exts = m.get("extensions", {})
    ior = float(exts.get("KHR_materials_ior", {}).get("ior", 1.5))
    spec = exts.get("KHR_materials_specular", {})
    factor = float(spec.get("specularFactor", 1.0))
    color = np.asarray(spec.get("specularColorFactor", [1.0, 1.0, 1.0]), np.float64)
    s = float(np.minimum(((ior - 1.0) / (ior + 1.0)) ** 2 * color, 1.0).mean()) * factor
    for key in ("specularTexture", "specularColorTexture"):
        if key in spec:
            info["ignored"] = sorted(set(info["ignored"]) | {f"KHR_materials_specular.{key}"})
    rough, metal, s = (min(max(x, 0.0), 1.0) for x in (rough, metal, s))
    ours, theirs = metal + (1 - metal) * min(1.0, 50.0 * s), metal + (1 - metal) * factor
    if abs(ours - theirs) > 1e-6 and (s > 0 or metal > 0):
        info.setdefault("pbr_f90", []).append({"material": index, "gltf": theirs, "ballasim": ours})
    return {"roughness": rough, "metallic": metal, "specular": s}


def load_gltf(path: str | Path, *, node: str | None = None, shading: str = "auto",
              pbr: bool = False) -> tuple[Mesh, list[Material], dict]:
    """Flatten the default scene (or the subtree of the node named ``node``) into one mesh. ``shading``: "auto"
    (KHR_materials_unlit materials are baked), "lit" or "baked" (every material baked); ``pbr``: read the lit
    materials' metallic-roughness BSDF (the C1 feature "pbr"); see the module docstring."""
    if shading not in SHADINGS:
        raise AssetError(f"shading must be one of {SHADINGS}, got {shading!r}")
    path = Path(path)
    doc = _Document(path)
    j = doc.json
    handled = SUPPORTED_REQUIRED - {UNLIT} if shading == "lit" else SUPPORTED_REQUIRED
    read_bsdf = bool(pbr)      # material_for's `pbr` is the material's pbrMetallicRoughness
    if read_bsdf:
        handled = handled | PBR_EXTENSIONS
    info: dict = {"format": "gltf", "ignored": sorted({e for m in j.get("materials", [])
                                                       for e in m.get("extensions", {})} - handled)}
    dropped: set[str] = set()       # emission inputs zeroed on baked materials
    baked: list[int] = []
    if any("skin" in n for n in j.get("nodes", [])):
        raise AssetError(f"{path}: skinned meshes are refused (rigid bodies only, SCN-02)")
    nodes = j.get("nodes", [])
    if node is not None:
        roots = [i for i, n in enumerate(nodes) if n.get("name") == node]
        if not roots:
            raise AssetError(f"{path}: no node named {node!r}")
    else:
        scene = j.get("scenes", [{}])[j.get("scene", 0)] if j.get("scenes") else {}
        roots = scene.get("nodes", list(range(len(nodes))))
    materials: list[Material] = []
    mat_index: dict[int, int] = {}
    alpha_modes = set()

    def material_for(i: int | None) -> tuple[int, int, np.ndarray | None]:
        """Our material index, the TEXCOORD set and the UV transform for glTF material i."""
        key = -1 if i is None else i
        m = {} if i is None else j["materials"][i]
        pbr = m.get("pbrMetallicRoughness", {})
        tinfo = pbr.get("baseColorTexture")
        texcoord, uvt = 0, None
        if tinfo is not None:
            texcoord = tinfo.get("texCoord", 0)
            ext = tinfo.get("extensions", {}).get("KHR_texture_transform")
            if ext is not None:
                uvt = texture_transform_matrix(ext)
                texcoord = ext.get("texCoord", texcoord)
        if key not in mat_index:
            alpha_modes.add(m.get("alphaMode", "OPAQUE"))
            tex = doc.image(j["textures"][tinfo["index"]]["source"]) if tinfo is not None else None
            factor = np.array(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3], np.float32)
            exts = m.get("extensions", {})
            mat_index[key] = len(materials)
            if shading == "baked" or (shading == "auto" and UNLIT in exts):
                # Unlit's Definition: color = baseColorTerm; no emissive term (scans.md §2.2, §3.5).
                if any(float(c) != 0.0 for c in m.get("emissiveFactor", [0, 0, 0])):
                    dropped.add("emissiveFactor")
                if "emissiveTexture" in m:
                    dropped.add("emissiveTexture")
                if "KHR_materials_emissive_strength" in exts:
                    dropped.add("KHR_materials_emissive_strength")
                baked.append(len(materials))
                materials.append(Material(factor, tex, np.zeros(3, np.float32), shading="baked"))
            else:
                strength = exts.get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1.0)
                emissive = np.array(m.get("emissiveFactor", [0, 0, 0]), np.float32) * np.float32(strength)
                extra = _pbr_material(doc, m, len(materials), info) if read_bsdf else {}
                materials.append(Material(factor, tex, emissive, **extra))
        return mat_index[key], texcoord, uvt

    parts = []

    def visit(i: int, parent: np.ndarray):
        n = nodes[i]
        world = parent @ _node_matrix(n)
        if "mesh" in n:
            for prim in j["meshes"][n["mesh"]]["primitives"]:
                attrs = prim["attributes"]
                if any(k.startswith(("KHR_draco", "EXT_meshopt")) for k in prim.get("extensions", {})):
                    raise AssetError(f"{path}: compressed primitives are not supported")
                pos = doc.accessor(attrs["POSITION"]).astype(np.float64)
                tri = _triangles(prim, doc, len(pos))
                if tri is None:
                    continue
                mid, texcoord, uvt = material_for(prim.get("material"))
                if materials[mid].shading == "baked" and "COLOR_0" in attrs:
                    rgb = doc.accessor(attrs["COLOR_0"])[:, :3]
                    if not np.all(rgb == 1.0):
                        raise AssetError(f"{path}: mesh {n['mesh']} has an unlit or baked primitive whose COLOR_0 is "
                                         "not white; C1 meshes carry no vertex colours (scans.md §3.5)")
                nrm = doc.accessor(attrs["NORMAL"]).astype(np.float64) if "NORMAL" in attrs else None
                weights = n.get("weights", j["meshes"][n["mesh"]].get("weights", []))
                for w, target in zip(weights, prim.get("targets", [])):
                    if w:
                        pos = pos + w * doc.accessor(target["POSITION"]) if "POSITION" in target else pos
                        if nrm is not None and "NORMAL" in target:
                            nrm = nrm + w * doc.accessor(target["NORMAL"])
                        info["morph_weights_applied"] = list(weights)
                # Missing normals: angle-weighted per primitive (the spec's flat normals would split
                # every vertex; scanned and modelled assets in use all ship normals).
                nrm = vertex_normals(pos, tri) if nrm is None else nrm.astype(np.float32)
                uv = None
                if f"TEXCOORD_{texcoord}" in attrs:
                    uv = doc.accessor(attrs[f"TEXCOORD_{texcoord}"]).astype(np.float64)
                    if uvt is not None:
                        uv = uv @ uvt[:2, :2].T + uvt[:2, 2]
                    uv = uv.astype(np.float32)
                mesh = Mesh(pos.astype(np.float32), tri.astype(np.uint32), nrm, uv,
                            np.full(len(tri), mid, np.uint16))
                parts.append(transform_mesh(mesh, world))
        for c in n.get("children", []):
            visit(c, world)

    for r in roots:
        visit(r, Y_UP_TO_Z_UP)
    if not parts:
        raise AssetError(f"{path}: no triangle geometry in the scene")
    # Material ids are already global (indices into `materials`), so parts merge without offsets.
    mesh, _ = merge_meshes([(p, []) for p in parts])
    mesh.material_ids = None if len(materials) == 1 else np.concatenate([p.material_ids for p in parts])
    if any(m.texture is not None for m in materials) and mesh.uvs is None:
        raise AssetError(f"{path}: textured material but no texture coordinates")
    info["alpha_modes"] = sorted(alpha_modes)
    info["num_primitives"] = len(parts)
    if dropped:
        info["ignored"] = sorted(set(info["ignored"]) | dropped)
    if baked:
        info["baked_materials"] = baked
    return mesh, materials, info
