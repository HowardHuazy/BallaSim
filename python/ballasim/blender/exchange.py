"""The exchange reader (``ballasim.blender_exchange/1``, docs/design/phase3/blender.md §3.4).

An exchange directory holds ``exchange.json`` (raw RNA values and node graphs), ``arrays.npz`` (uncompressed, read
without pickle) and ``textures/<sha256>.npy``. The reader

- refuses another format or major version;
- handles keys it does not know by the file's minor version: a newer minor than this engine's adds optional keys,
  which are ignored with an ``info`` item; at the same or an older minor they are typos and refused (``extras`` and
  the free-form records are never checked);
- computes the exchange digest: SHA-256 over the canonical ``exchange.json`` without ``producer``, then each array's
  (key, dtype, shape, C-order bytes) in sorted key order and each texture's SHA-256 (its file name). The npz file
  bytes are not hashed, so the numpy version that wrote them cannot move the digest.

The key tables below are C10 v1.0's (blender.md §3.4, with PHASE3.md H4's ``asset`` record on every exchanged ID,
each object's ``collections``, nearest first, for the collection semantics of §4.8, each object's
``data_animation``, the animation of its data block (a curve's ``eval_time`` that moves a Follow Path follower, §4.7.4;
BL-4b), the ``animation`` index of a light's and a camera's data block, as ``objects[].animation`` indexes the
object's, for LGT-ANIM and CAM-ANIM, and
for the world and the lights' node trees, whose keys live on the trees' own animation data (probe m3b), the
``graph_animation`` and ``node_graph_animation`` indices, with the world's ``animation`` and its ``color``, which
Cycles renders when the world uses no nodes). Every node graph (a material's, a light's, the world's) is checked
alike. The tables are the keys of ``schema/blender_exchange-1.json``, the exchange's JSON Schema (a shared contract
file, CC0-1.0), which tests/python/test_blender_schema.py holds them to.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT = "ballasim.blender_exchange"
FREE = "free"                     # a record whose keys are not checked


class ExchangeError(ValueError):
    """An exchange this engine cannot read: another format or major version, a typo, a missing key or array."""


def _keys(*names, **records) -> dict:
    return {**dict.fromkeys(names), **records}


_ASSET = _keys("license", "copyright", "author")
_GRAPH = _keys(nodes=[_keys("name", "type", "data_type", "blend_type", props=FREE,
                            inputs=[_keys("identifier", "name", "type", "enabled", "value", "default", "linked")],
                            outputs=[_keys("identifier", "name", "type", "enabled")])],
               links=None)
SPEC = _keys(
    "format", "version", "gravity", "use_gravity", "semantic_classes", "collection_semantics", "extras",
    producer=_keys("addon", "blender", "blend_file", "blend_sha256", "exported"),
    units=_keys("system", "scale_length"),
    color=_keys("scene_linear", "display_device", "view_transform"),
    time=_keys("fps", "fps_base_bits", "frame_start", "frame_end", "frame_map_old", "frame_map_new"),
    span=_keys("frame_start", "frame_end"),
    render=_keys("resolution_x", "resolution_y", "resolution_percentage", "pixel_aspect_x", "pixel_aspect_y",
                 "use_border", "use_simplify", "rolling_shutter_type", "rolling_shutter_duration", "use_motion_blur",
                 "motion_blur_shutter", "motion_blur_position", "view_transform", "look", "exposure", "gamma",
                 "film_exposure", "max_bounces", "use_denoising", "pixel_filter_type", "filter_width"),
    rigidbody_world=_keys("enabled", "substeps_per_frame", "solver_iterations", "time_scale", "collection",
                          effector_weights=_keys("gravity", "all", "collection"),
                          point_cache=_keys("frame_start", "frame_end", "frame_step", "is_baked", "is_outdated",
                                            "use_disk_cache", "info")),
    meshes=[_keys("id", "sha256", "normals_domain", "uv_maps", "active_render_uv", counts=_keys("V", "C", "T"),
                  arrays=_keys("co", "corner_vert", "tri_corners", "tri_mat", "corner_normal", "vert_normal", uv=FREE),
                  asset=_ASSET)],
    images=[_keys("id", "file", "kind", "colorspace", "alpha_mode", "width", "height", "channels", "extension",
                  "interpolation", "source_path", asset=_ASSET)],
    materials=[_keys("id", "name", "emission_sampling", graph=_GRAPH, asset=_ASSET)],
    objects=[_keys("key", "name_full", "library", "kind", "type", "instancer", "persistent_id", "instance_object",
                   "parent", "parent_type", "parent_bone", "mesh", "material_slots", "matrix_world", "lightgroup",
                   "pass_index", "instance_id", "instance_block", "instancer_ids", "semantic", "is_editable",
                   "animation", "data_animation", "collections",
                   visibility=_keys("render", "camera", "shadow", "diffuse", "glossy", "holdout", "shadow_catcher",
                                    "hide_render_animated"),
                   deform=_keys("is_deform_modified", "modifiers", "shape_keys_animated", "has_geometry_nodes",
                                "is_viewport_node"),
                   rigid_body=_keys("type", "enabled", "kinematic", "kinematic_animated", "collision_shape",
                                    "collision_margin", "linear_damping", "angular_damping"),
                   field=_keys("type", "strength"),
                   constraints=[_keys("type", "target", "subtarget", "mute", "influence", "cache_file")],
                   asset=_ASSET)],
    cache_files=[_keys("name", "filepath", "frame", "frame_offset", "override_frame", "is_sequence")],
    animations=[_keys("owner", "action", "slot",
                      fcurves=[_keys("key", "data_path", "index", "extrapolation", "keyframes", "group", "modifiers")],
                      nla=[_keys("track", "mute", "is_solo",
                                 strips=[_keys("action", "slot", "frame_start", "frame_end", "action_frame_start",
                                               "action_frame_end", "scale", "repeat", "use_reverse", "extrapolation",
                                               "blend_type", "blend_in", "blend_out", "use_auto_blend", "influence",
                                               "use_animated_influence", "use_animated_time", "mute")])],
                      drivers=[_keys("data_path", "index", "type", "expression", "is_simple_expression",
                                     "is_valid")])],
    lights=[_keys("object", "type", "animation", "node_graph_animation", node_graph=_GRAPH,
                  props=_keys("energy", "color", "shadow_soft_size", "use_soft_falloff", "spot_size", "spot_blend",
                              "angle", "shape", "size", "size_y", "spread", "use_shadow", "exposure", "normalize",
                              "use_temperature", "temperature_color"),
                  light_linking=_keys("receiver_collection", "blocker_collection"), asset=_ASSET)],
    world=_keys("color", "lightgroup", "animation", "graph_animation", graph=_GRAPH,
                cycles_visibility=_keys("camera", "diffuse", "glossy", "transmission", "scatter"), asset=_ASSET),
    cameras=[_keys("object", "c1_name", "sensor_family", "simulate", "animation",
                   props=_keys("type", "lens", "sensor_width", "sensor_height", "sensor_fit", "shift_x", "shift_y",
                               "clip_start", "clip_end", "panorama_type", "fisheye_fov", "fisheye_lens",
                               *(f"fisheye_polynomial_k{i}" for i in range(5)),
                               dof=_keys("use_dof", "aperture_fstop")),
                   calibration=_keys("source", "preset", "file_text", "camera"), asset=_ASSET)],
    light_groups=[_keys("name", "waveform")],
    addon_report=[_keys("code", "object", "message")],
)
#: Keys a reader cannot do without (C10 v1.0).
REQUIRED = {"": ("format", "version", "producer", "units", "time", "span", "meshes", "objects", "cameras"),
            "meshes": ("id", "arrays"), "objects": ("key", "name_full", "kind", "matrix_world"),
            "cameras": ("object",)}


@dataclass
class Exchange:
    """A read and checked exchange directory."""

    root: Path
    doc: dict
    arrays: dict[str, np.ndarray]
    digest: str
    version: tuple[int, int]
    info: list[dict] = field(default_factory=list)   # report items of the reader (ignored keys of a newer minor)


def reader_version() -> tuple[int, int]:
    """(major, highest minor) of the exchange format this engine reads (``_scene.blender_formats()``)."""
    from ballasim import _scene

    return tuple(_scene.blender_formats()["exchange_formats"][FORMAT])


def unknown_keys(doc: dict, spec: dict = SPEC, path: str = "") -> list[str]:
    """The keys of ``doc`` (and of its records, recursively) that ``spec`` does not know, as dotted paths."""
    out = []
    for key, value in doc.items():
        where = f"{path}.{key}" if path else key
        if key not in spec:
            out.append(where)
            continue
        sub = spec[key]
        if isinstance(sub, dict) and isinstance(value, dict):
            out += unknown_keys(value, sub, where)
        elif isinstance(sub, list) and isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, dict):
                    out += unknown_keys(v, sub[0], f"{where}[{i}]")
    return out


def exchange_digest(doc: dict, arrays: dict[str, np.ndarray]) -> str:
    """The exchange digest (blender.md §3.4): canonical JSON without ``producer``, then the arrays' contents and the
    textures' SHA-256s, each framed by its length so no two inputs share a byte stream."""
    h = hashlib.sha256()

    def part(b: bytes) -> None:
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)

    body = {k: v for k, v in doc.items() if k != "producer"}
    part(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode())
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        part(key.encode())
        part(f"{a.dtype.str}{a.shape}".encode())
        part(a.tobytes())
    for sha in sorted(Path(im["file"]).stem for im in doc.get("images") or []
                      if isinstance(im, dict) and isinstance(im.get("file"), str)):
        part(sha.encode())
    return h.hexdigest()


def texture_paths(exchange: Exchange) -> list[str | None]:
    """The pixel file of each images[] record, in order (blender.md §3.4: ``textures/<sha256>.npy``, uint8[H, W, C]
    raw codes of a byte image in Blender's bottom-up row order); None for float images and for images written without
    pixels (``file`` null: a missing file, or a tiled, sequence or movie image), which the converter refuses where a
    material uses them, before reading anything (TEX-CS). The C++ converter reads a file only when a material uses its
    image, one at a time, so no file stays open. ExchangeError for a file outside ``textures/`` or missing."""
    out = []
    tex_dir = (exchange.root / "textures").resolve()
    for i, im in enumerate(exchange.doc.get("images") or []):
        if not isinstance(im, dict) or im.get("kind", "byte") == "float" or im.get("file", "") is None:
            out.append(None)
            continue
        file = im.get("file")
        path = (exchange.root / str(file)).resolve() if isinstance(file, str) else None
        if path is None or path.parent != tex_dir:
            raise ExchangeError(f"{exchange.root}: images[{i}].file {file!r} is not a file in textures/")
        if not path.is_file():
            raise ExchangeError(f"{exchange.root}: images[{i}] {file}: no such file")
        out.append(str(path))
    return out


def read_exchange(directory: str | os.PathLike) -> Exchange:
    """Read and check an exchange directory (blender.md §3.4); ExchangeError when this engine cannot read it."""
    root = Path(directory)
    try:
        doc = json.loads((root / "exchange.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (ValueError, UnicodeDecodeError) as e:
        raise ExchangeError(f"{root}/exchange.json: not JSON: {e}") from None
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise ExchangeError(f"{root}: not a {FORMAT} exchange")
    version = doc.get("version")
    if not (isinstance(version, list) and len(version) == 2 and all(type(v) is int for v in version)):
        raise ExchangeError(f"{root}: version must be [major, minor], got {version!r}")
    major, minor = reader_version()
    if version[0] != major:
        raise ExchangeError(f"{root}: exchange version {version[0]}.{version[1]} refused: this engine reads "
                            f"{FORMAT} {major}.x (up to {major}.{minor})")
    info = []
    unknown = unknown_keys(doc)
    if unknown and version[1] > minor:
        info = [{"code": "EXCHANGE-KEYS", "severity": "info", "object": "", "bound": 0.0, "bound_kind": "none",
                 "message": f"keys of exchange {version[0]}.{version[1]} ignored by this {major}.{minor} reader: "
                            f"{', '.join(unknown)}"}]
    elif unknown:
        raise ExchangeError(f"{root}: unknown keys in a {version[0]}.{version[1]} exchange (this engine reads up to "
                            f"{major}.{minor}): {', '.join(unknown)}")
    for record, keys in REQUIRED.items():
        for i, rec in enumerate([doc] if not record else doc.get(record) or []):
            missing = [k for k in keys if not isinstance(rec, dict) or k not in rec]
            if missing:
                raise ExchangeError(f"{root}: {record + f'[{i}]' if record else 'exchange.json'} misses "
                                    f"{', '.join(missing)}")
    try:
        with np.load(root / "arrays.npz", allow_pickle=False) as npz:
            arrays = {}
            for k in npz.files:   # C order for the converter: another writer may store Fortran-order arrays
                a = npz[k]
                arrays[k] = a if a.flags.c_contiguous else np.ascontiguousarray(a)
    except FileNotFoundError:
        raise
    except (zipfile.BadZipFile, ValueError, OSError, EOFError) as e:
        raise ExchangeError(f"{root}/arrays.npz: not a readable npz without pickles: {e}") from None
    try:
        digest = exchange_digest(doc, arrays)
    except ValueError as e:
        raise ExchangeError(f"{root}: {e}") from None
    return Exchange(root, doc, arrays, digest, (version[0], version[1]), info)
