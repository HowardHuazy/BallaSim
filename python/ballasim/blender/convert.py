"""Blender exchanges as scene sources (docs/design/phase3/blender.md §3.9, §5.1): the statics conversion (C++,
``ballasim._scene.blender_statics``), ``blender check``'s document, the bundle cache key and :func:`load_blender`.

Converted bundles are cached inside the exchange, at ``scene.bsx/cache/<blender_cache_key>/``, so deleting a job
deletes its bundles. The key is digest(scene) of the ``scene.blender.*`` values with the facts the bundle rests on:
the exchange digest, the motion session's outputs, the converter version and options, the source of
``ballasim.{scene,traj,blender}`` and the ``_scene`` build. The motion session (BL-9, :mod:`.session`) writes the
entries of scenes that move (:func:`assemble_moving` from its ``motion/`` outputs); without its entry
:func:`load_blender` rebuilds the bundle from those outputs when only conversion options changed, and refuses when the
motion it recorded no longer applies (MOT-STALE). A static scene, whose bodies, simulated cameras and lights all stay
put (``session_needed`` empty), needs no session: :func:`load_blender` assembles its bundle from the statics
(:func:`assemble_static`) and caches it; any other source without its entry is refused (SCENE.md "Blender exchanges").
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import calibrations
from .exchange import Exchange, read_exchange, texture_paths

#: The conversion keys among the scene.blender.* rows (§3.11): the statics options. A Blender source's resolved values
#: hold every row (config.load fills them, source_kind blender); without them the C++ defaults apply.
OPTION_KEYS = ("policy", "lm_per_w", "world_as_ambient", "crop_radius_m", "max_triangles", "calib_tol_px",
               "fisheye_fit_tol_px", "rigidity_tol_m")


def is_exchange(path: str | os.PathLike) -> bool:
    """Whether ``path`` is a Blender exchange directory (it holds ``exchange.json``)."""
    p = Path(path)
    return p.is_dir() and (p / "exchange.json").is_file()


def options_of(values: Mapping[str, Any]) -> dict:
    """The statics options a run's resolved values set (``scene.blender.<option>``)."""
    return {k: values[f"scene.blender.{k}"] for k in OPTION_KEYS if f"scene.blender.{k}" in values}


def statics(exchange: Exchange, options: Mapping[str, Any] | None = None, origin=None, dynamic_ids=()) -> dict:
    """The exchange's statics (``_scene.blender_statics``): bodies with their materials (§4.5), the textures they
    share, the cameras (§4.3, with their calibrations read by :mod:`.calibration`), the lights, the world and the
    light groups (§4.4), camera names, the assets they use (PHASE3.md H4), the conversion report (§3.12, with the
    reader's items first) and the effective options. ``dynamic_ids``: ids of bodies the motion session found moving
    although they were static candidates (lenient MOT-SOURCE): placed as moving bodies."""
    from ballasim import _scene

    out = _scene.blender_statics(exchange.doc, exchange.arrays, exchange.digest, dict(options or {}), origin,
                                 texture_paths(exchange), calibrations(exchange.doc), [int(i) for i in dynamic_ids])
    report = out["report"]
    report["items"] = list(exchange.info) + report["items"]
    report["summary"]["info"] += len(exchange.info)
    return out


def materials_of(st: dict, body: dict) -> list:
    """A statics body's C1 materials (:class:`ballasim.scene.bundle.Material`), sharing the statics' texture arrays."""
    from ballasim.scene.bundle import Material

    return [Material(np.asarray(m["base_color"], np.float32),
                     None if m["texture"] is None else st["textures"][m["texture"]]["pixels"],
                     np.asarray(m["emissive"], np.float32)) for m in body["materials"]]


def provisional_bundle_info(st: dict) -> dict:
    """The bundle facts resolution needs (``ballasim.config.bundle_info``'s dict), from an exchange's statics before
    any bundle exists (§3.3's Check): the simulated cameras (names, sizes, sensor hints, pixel pitch, f-number; clock
    offset 0) and the bundle span of the TimeMap, the shadow-casting lights and the background; no digests. ValueError
    when the time record does not convert."""
    from ballasim.config import device_available

    if "error" in st["timeline"]:
        raise ValueError(st["timeline"]["error"])
    t0, t1 = st["timeline"]["bundle_span"]
    lights = st["lights"]
    return {"cameras": [{"name": c["name"], "sensor": c["sensor"] or "unspecified", "width": int(c["width"]),
                         "height": int(c["height"]), "clock_offset_ns": 0, "pixel_pitch_m": float(c["pixel_pitch_m"]),
                         "f_number": float(c["f_number"])} for c in st["cameras"] if c["simulate"]],
            "t_start_ns": int(t0), "t_end_ns": int(t1), "device_available": device_available(), "supplied": False,
            "bundle_digest": "", "engine_digest": "", "low_light": None,
            "shadow_lights": sum(lt["kind"] not in ("ambient", "background") and bool(lt["casts_shadows"])
                                 for lt in lights),
            "zero_background": not any(lt["kind"] == "background" and lt["intensity"] > 0 for lt in lights)}


def _unresolved(ex: Exchange, config_path, sets) -> tuple[dict, dict | None]:
    """The run configuration's statics options and its Check section: {unresolved: [{key, hint, help}], error}. The
    configuration is loaded with scene.path set to the exchange (its scene.blender.* rows filled), and resolved in
    collect mode on the statics' provisional bundle facts; any other error is returned on its own. The exchange's path
    is passed as is (``scene_path``), never as a --set item: YAML would cut it at " #", read ": " as a mapping, and
    expansion would read "$" as a variable."""
    from ballasim import config

    try:
        spec = config.load(config_path, sets=sets, scene_path=ex.root)
    except (ValueError, OSError) as e:            # ConfigError, a malformed --set item, a missing file
        return {"unresolved": [], "error": str(e)}, None
    values = spec.scene.values()["run"]
    st = statics(ex, options_of(values))
    try:
        return {"unresolved": config.missing_keys(spec, provisional_bundle_info(st)), "error": None}, st
    except ValueError as e:                       # ConfigError, or a time record that does not convert
        return {"unresolved": [], "error": str(e)}, st


def check(directory: str | os.PathLike, options: Mapping[str, Any] | None = None, *,
          config: str | os.PathLike | None = None, sets=()) -> dict:
    """``ballasim blender check``'s statics part (§3.3): the exchange's version and digest, the conversion report, a
    summary of every body (its pose at the export frame, its mesh size, whether it is a static candidate), the cameras
    (C1 intrinsics, sensor, pitch, f-number), the lights and the light groups. With a run configuration ``config``
    (and ``sets``, --set items), the statics take its scene.blender.* options instead of ``options``, and the document
    gains ``config``: every unresolved required key at once, or the configuration's other error (§3.11). ValueError
    for ``sets`` without ``config`` (they would apply to nothing)."""
    if sets and config is None:
        raise ValueError("check: sets need a run configuration (config=)")
    ex = read_exchange(directory)
    section, st = _unresolved(ex, config, sets) if config is not None else (None, None)
    st = st if st is not None else statics(ex, options)
    bodies = [{"name": b["name"], "object": b["object"], "id": b["id"], "semantic": b["semantic"], "role": b["role"],
               "vertices": len(b["mesh"]["positions"]), "triangles": len(b["mesh"]["indices"]),
               "static": b["T_world_static"] is not None, "rho_m": b["rho"],
               "materials": [{"exchange": mi, "base_color": list(m["base_color"]), "emissive": list(m["emissive"]),
                              "texture": m["texture"]} for mi, m in zip(b["material_of"], b["materials"])]}
              for b in st["bodies"]]
    textures = [{"image": t["image"], "mirror": t["mirror"], "shape": list(t["pixels"].shape)}
                for t in st["textures"]]
    cameras = [{k: c[k] for k in ("name", "model", "width", "height", "fx", "fy", "cx", "cy", "dist", "sensor",
                                  "pixel_pitch_m", "f_number", "blender_pinhole", "theta_max", "calibrated",
                                  "simulate")}
               | {"static": c["T_world_static"] is not None,
                  "T_world_rig": np.asarray(c["T_world_export"]).ravel().tolist(),
                  "T_rig_cam": np.asarray(c["T_rig_cam"]).ravel().tolist()} for c in st["cameras"]]
    return {"format": "ballasim.blender_check", "version": [1, 0],
            "exchange": {"format": ex.doc["format"], "version": list(ex.version), "digest": ex.digest,
                         "producer": ex.doc.get("producer")},
            "refused": st["refused"], "report": st["report"], "options": st["options"],
            "timeline": st["timeline"], "session_needed": st["session_needed"],
            "statics": {"bodies": bodies, "textures": textures, "triangles": st["triangles"],
                        "cameras": cameras, "camera_names": st["camera_names"], "lights": st["lights"],
                        "light_groups": st["light_groups"], "assets": st["assets"],
                        "origin_world_provisional": st["origin_world"]},
            **({"config": section} if section is not None else {})}


def motion_digest(motion_dir: Path) -> str:
    """SHA-256 over the session's trajectories (key, dtype, shape, bytes) and canonical session.json (§5.1); "none"
    before a session has run."""
    session = motion_dir / "session.json"
    if not session.is_file():
        return "none"
    h = hashlib.sha256()
    h.update(json.dumps(json.loads(session.read_text()), sort_keys=True, separators=(",", ":")).encode())
    traj = motion_dir / "trajectories.npz"
    if traj.is_file():
        with np.load(traj, allow_pickle=False) as npz:
            for key in sorted(npz.files):
                a = np.ascontiguousarray(npz[key])
                h.update(f"{key}\0{a.dtype.str}{a.shape}\0".encode())
                h.update(a.tobytes())
    return h.hexdigest()


def blender_cache_key(values: Mapping[str, Any], exchange: Exchange) -> str:
    """digest(scene) of a Blender source: the key of its converted bundle (§5.1)."""
    from ballasim import _core, _scene
    from ballasim.engine.scene import BUNDLE_CACHE_VERSION, _build_facts, _source_digest

    options = _scene.blender_statics_options(options_of(values))
    facts = {"cache": str(BUNDLE_CACHE_VERSION), "adapter": "blender", "exchange": exchange.digest,
             "motion": motion_digest(exchange.root / "motion"),
             "converter": str(_scene.blender_formats()["converter"]),
             "options": json.dumps(options, sort_keys=True), "source": _source_digest(("scene", "traj", "blender")),
             **_build_facts()}
    blender = {k: v for k, v in values.items() if k.startswith("scene.blender.")}
    known = {r["key"] for r in _core.config.schema()}
    rows = {k: v for k, v in blender.items() if k in known}
    if len(rows) != len(blender):   # a value whose schema row is not there yet (BL-3b adds them): a fact until then
        facts["blender_values"] = json.dumps({k: v for k, v in blender.items() if k not in known}, sort_keys=True)
    return _core.config.scene_digest(rows, facts)


def assemble_static(ex: Exchange, st: dict, values: Mapping[str, Any] | None = None):
    """The C1 bundle of a static scene from its statics (``st``, :func:`statics` with ``session_needed`` empty;
    blender.md §3.7, §4.1, §4.2, §4.7.9): the bodies at their export poses, the simulated cameras on constant rig
    trajectories held over the bundle span [-2W, t_end), the world lights and the light groups, about the provisional
    origin (the simulated cameras' AABB centre). ``bundle.meta`` records ``source: blender`` and ``blender`` {producer, values,
    k_lm_per_w, lm_per_w_set, scale_length, origin_world_rule, time_map, id_map, motion, report_summary, options}
    and the assets. A camera without a sensor family says "unspecified" (the run's ``sensor.family`` decides).
    ValueError when the statics are not those of a static scene that converts."""
    from ballasim.scene.bundle import Body, Camera, Light, LightGroup, Mesh, SceneBundle, Trajectory

    if st["refused"] or st["session_needed"] or "error" in st["timeline"]:
        raise ValueError(f"{ex.root}: not a static scene that converts")
    origin = np.asarray(st["origin_world"], np.float64)
    tl = st["timeline"]
    t0, t1 = tl["bundle_span"]
    bodies = []
    for b in st["bodies"]:
        m = b["mesh"]
        T = np.array(b["T_world_static"], np.float64)
        T[:3, 3] -= origin
        bodies.append(Body(int(b["id"]), b["name"], int(b["semantic"]),
                           Mesh(m["positions"], m["indices"], m["normals"], m["uvs"], m["material_ids"]),
                           materials_of(st, b), None, T))
    rigs = {r["camera"]: r for r in st["rigs"]}
    trajectories, cameras = [], []
    for c in st["cameras"]:
        if not c["simulate"]:
            continue
        r = rigs[c["name"]]
        trajectories.append(Trajectory(int(r["K"]), int(r["t0_ns"]), int(r["h_ns"]), np.asarray(r["rot"]),
                                       np.asarray(r["pos"]), np.array([[t0, t1]], np.int64),
                                       tuple(int(x) for x in r["span_ns"])))
        cameras.append(Camera(c["name"], c["model"], int(c["width"]), int(c["height"]), float(c["fx"]),
                              float(c["fy"]), float(c["cx"]), float(c["cy"]), np.asarray(c["dist"], np.float64),
                              len(trajectories) - 1, np.asarray(c["T_rig_cam"], np.float64), 0,
                              c["sensor"] or "unspecified", float(c["pixel_pitch_m"]), float(c["f_number"])))
    if not cameras:
        raise ValueError(f"{ex.root}: no camera to simulate (none has simulate set)")
    groups = [LightGroup(g["name"], dict(g["waveform"])) for g in st["light_groups"]]
    lights = []
    for lt in st["lights"]:
        pos = np.asarray(lt["position"], np.float64)
        lights.append(Light(lt["kind"], int(lt["group"]), np.asarray(lt["color"], np.float32), float(lt["intensity"]),
                            lt["body"], pos - origin if lt["body"] is None else pos,
                            np.asarray(lt["direction"], np.float64), float(lt["spot_inner_rad"]),
                            float(lt["spot_outer_rad"]), bool(lt["casts_shadows"])))
    values = dict(values or {})
    opts = st["options"]
    producer = ex.doc.get("producer") or {}
    blend = producer.get("blend_file")
    meta = {"source": "blender",
            "blender": {"producer": producer,
                        "values": {k: v for k, v in values.items() if k.startswith("scene.blender.")},
                        "k_lm_per_w": opts.get("lm_per_w") or 1.0, "lm_per_w_set": opts.get("lm_per_w") is not None,
                        "scale_length": float((ex.doc.get("units") or {}).get("scale_length") or 1.0),
                        "origin_world_rule": "static: the AABB centre of the simulated cameras at the export frame",
                        "time_map": tl, "exchange_digest": ex.digest,
                        "id_map": {"cameras": dict(zip(st["camera_names"],
                                                       (ex.doc["objects"][c["object"]]["name_full"]
                                                        for c in ex.doc.get("cameras") or []))),
                                   "bodies": {b["name"]: int(b["id"]) for b in st["bodies"]}},
                        "motion": {"session": None, "static": True},
                        "report_summary": st["report"]["summary"], "options": opts},
            "assets": st["assets"]}
    name = Path(blend).stem if isinstance(blend, str) and blend else "blender"
    return SceneBundle(name, int(t0), int(t1), origin, bodies, trajectories, cameras, groups, lights, meta).validate()


#: The scene.blender.* keys that change only the conversion, not the motion (blender.md §3.6 MOT-STALE).
CONVERSION_ONLY = ("lm_per_w", "world_as_ambient", "max_triangles")


def motion_key_inputs(values: Mapping[str, Any], exchange: Exchange, st: dict, cameras) -> dict:
    """What the motion a session recorded depends on (blender.md §3.6): the ``scene.blender.*`` values except the
    conversion-only keys, the converter version, the exchange digest and the simulated cameras with their resolved
    intrinsics (a digest each). ``cameras``: the C1 names carried."""
    from ballasim import _scene

    wanted = set(cameras)
    cams = {}
    for c in st["cameras"]:
        if c["name"] in wanted:
            rec = {k: c[k] for k in ("model", "width", "height", "fx", "fy", "cx", "cy")} | {"dist": list(c["dist"])}
            cams[c["name"]] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    vals = {k: v for k, v in values.items()
            if k.startswith("scene.blender.") and k.rsplit(".", 1)[1] not in CONVERSION_ONLY}
    return {"values": hashlib.sha256(json.dumps(vals, sort_keys=True, default=str).encode()).hexdigest(),
            "converter": int(_scene.blender_formats()["converter"]), "exchange": exchange.digest, "cameras": cams}


def motion_key(inputs: Mapping[str, Any]) -> str:
    """The SHA-256 of :func:`motion_key_inputs` (``session.json``'s ``motion_key``)."""
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def stale_reason(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> str | None:
    """Why a recorded motion no longer applies (MOT-STALE), else None: its values, converter, exchange or one of its
    cameras' intrinsics differ from ``current`` (:func:`motion_key_inputs` over the recorded cameras). A run that
    simulates a subset of the recorded cameras is accepted later: the error model of a superset is conservative."""
    for k in ("values", "converter", "exchange"):
        if recorded.get(k) != current.get(k):
            return f"{k} differs"
    for name, digest in (recorded.get("cameras") or {}).items():
        if current["cameras"].get(name) != digest:
            return f"camera {name} changed or is gone"
    return None


def trajectory_arrays(trajectories) -> dict:
    """The session's trajectories as ``trajectories.npz`` holds them (blender.md §3.6): per trajectory *b* ``b.<b>.rot``
    f64[N, 4], ``.pos`` f64[N, 3] (world metres relative to origin_world), ``.pieces`` i64[P, 6] (one row for a single
    spline), ``.holds`` i64[H, 2] and ``.span`` i64[2]."""
    out = {}
    for b, t in enumerate(trajectories):
        pieces = t["pieces"]
        if pieces is None:
            pieces = np.array([[t["span_ns"][0], t["t0_ns"], t["h_ns"], 0, len(t["rot"]), 0]], np.int64)
        out[f"b.{b}.rot"] = np.ascontiguousarray(t["rot"], np.float64)
        out[f"b.{b}.pos"] = np.ascontiguousarray(t["pos"], np.float64)
        out[f"b.{b}.pieces"] = np.ascontiguousarray(pieces, np.int64)
        out[f"b.{b}.holds"] = np.ascontiguousarray(t["holds"], np.int64).reshape(-1, 2)
        out[f"b.{b}.span"] = np.asarray(t["span_ns"], np.int64)
    return out


def assemble_moving(ex: Exchange, st: dict, session: Mapping[str, Any], traj: Mapping[str, np.ndarray],
                    values: Mapping[str, Any] | None = None):
    """The C1 bundle of a scene whose motion the session fitted (blender.md §3.7, §4.1, §4.7): ``st`` is
    :func:`statics` about ``session['origin_world']``; the bodies the session fitted take their trajectory, the others
    sit at their static poses; each carried camera rides its rig's trajectory with the session's ``T_rig_cam``; the
    bundle spans the timeline's ``bundle_span`` and the trajectories the session's wider ``trajectory_span``
    (PHASE3.md D-P3-12). ``traj``: :func:`trajectory_arrays`. ValueError when the pieces do not make a scene."""
    from ballasim.scene.bundle import Body, Camera, Light, LightGroup, Mesh, SceneBundle, Trajectory

    if st["refused"]:
        raise ValueError(f"{ex.root}: the statics are refused")
    origin = np.asarray(session["origin_world"], np.float64)
    t0, t1 = session["bundle_span"]
    trajectories = []
    for b in range(len([k for k in traj if k.endswith(".rot")])):
        pieces = traj[f"b.{b}.pieces"]
        n = len(traj[f"b.{b}.rot"])
        span = tuple(int(x) for x in traj[f"b.{b}.span"])
        single = len(pieces) == 1
        trajectories.append(Trajectory(6, int(pieces[0, 1]), int(pieces[0, 2]), traj[f"b.{b}.rot"], traj[f"b.{b}.pos"],
                                       traj[f"b.{b}.holds"], span, None if single else pieces))
        assert single is False or int(pieces[0, 4]) == n
    index_of = {int(r["id"]): int(r["trajectory"]) for r in session["bodies"] if r["role"] != "rig"}
    bodies = []
    for b in st["bodies"]:
        m = b["mesh"]
        ti = index_of.get(int(b["id"]), -1)
        if ti >= 0:
            T = np.eye(4)
        else:
            if b["T_world_static"] is None:
                raise ValueError(f"{ex.root}: body {b['name']} has no trajectory in the session and is not static")
            T = np.array(b["T_world_static"], np.float64)
            T[:3, 3] -= origin
        bodies.append(Body(int(b["id"]), b["name"], int(b["semantic"]),
                           Mesh(m["positions"], m["indices"], m["normals"], m["uvs"], m["material_ids"]),
                           materials_of(st, b), ti if ti >= 0 else None, T))
    by_name = {c["name"]: c for c in session["cameras"]}
    cameras = []
    for c in st["cameras"]:
        rec = by_name.get(c["name"])
        if rec is None:
            continue
        cameras.append(Camera(c["name"], c["model"], int(c["width"]), int(c["height"]), float(c["fx"]), float(c["fy"]),
                              float(c["cx"]), float(c["cy"]), np.asarray(c["dist"], np.float64), int(rec["trajectory"]),
                              np.asarray(rec["T_rig_cam"], np.float64).reshape(4, 4), 0,
                              c["sensor"] or "unspecified", float(c["pixel_pitch_m"]), float(c["f_number"])))
    if not cameras:
        raise ValueError(f"{ex.root}: no camera to simulate")
    groups = [LightGroup(g["name"], dict(g["waveform"])) for g in st["light_groups"]]
    lights = []
    for lt in st["lights"]:
        pos = np.asarray(lt["position"], np.float64)
        lights.append(Light(lt["kind"], int(lt["group"]), np.asarray(lt["color"], np.float32), float(lt["intensity"]),
                            lt["body"], pos - origin if lt["body"] is None else pos,
                            np.asarray(lt["direction"], np.float64), float(lt["spot_inner_rad"]),
                            float(lt["spot_outer_rad"]), bool(lt["casts_shadows"])))
    values = dict(values or {})
    opts = st["options"]
    producer = ex.doc.get("producer") or {}
    blend = producer.get("blend_file")
    scale = float((ex.doc.get("units") or {}).get("scale_length") or 1.0)
    g = np.asarray(ex.doc.get("gravity") or [0.0, 0.0, -9.81], np.float64)
    provenance = {"body_object": [int(b["object"]) for b in st["bodies"]],
                  "light_object": [int(lt["object"]) for lt in st["lights"]],
                  "body_material": [list(map(int, b["material_of"])) for b in st["bodies"]],
                  "material_image": [[int(st["textures"][m["texture"]]["image"]) if m["texture"] is not None else -1
                                      for m in b["materials"]] for b in st["bodies"]]}
    meta = {"source": "blender",
            "gravity_world_mps2": (scale * g).tolist() if ex.doc.get("use_gravity", True) else [0.0, 0.0, 0.0],
            "blender": {"producer": producer,
                        "values": {k: v for k, v in values.items() if k.startswith("scene.blender.")},
                        "k_lm_per_w": opts.get("lm_per_w") or 1.0, "lm_per_w_set": opts.get("lm_per_w") is not None,
                        "scale_length": scale,
                        "origin_world_rule": "the AABB centre of every motion sample of the moving bodies, light "
                                             "carriers and cameras",
                        "time_map": session["timeline"], "exchange_digest": ex.digest,
                        "id_map": {"cameras": dict(zip(st["camera_names"],
                                                       (ex.doc["objects"][c["object"]]["name_full"]
                                                        for c in ex.doc.get("cameras") or []))),
                                   "bodies": {b["name"]: int(b["id"]) for b in st["bodies"]}},
                        "motion": {"session": session["motion_key"],
                                   "static": all(r["mode"] == "static" for r in session["bodies"]),
                                   "trajectory_span": session["trajectory_span"], "simplify": session["simplify"]},
                        "provenance": provenance, "report_summary": st["report"]["summary"], "options": opts},
            "assets": st["assets"]}
    name = Path(blend).stem if isinstance(blend, str) and blend else "blender"
    return SceneBundle(name, int(t0), int(t1), origin, bodies, trajectories, cameras, groups, lights, meta).validate()


def reconvert(ex: Exchange, values: Mapping[str, Any]) -> dict:
    """The statics and recorded motion for a run's values, without Blender: the session's ``motion/`` outputs at their
    recorded origin. ConfigError (MOT-STALE) when what the motion depends on has changed, or the recorded outputs are
    unreadable; returns ``{st, session, traj}``."""
    from ballasim.config import ConfigError

    motion = ex.root / "motion"
    try:
        session = json.loads((motion / "session.json").read_text())
        with np.load(motion / "trajectories.npz", allow_pickle=False) as npz:
            traj = {k: npz[k] for k in npz.files}
    except (OSError, ValueError) as e:
        raise ConfigError(f"MOT-STALE: {motion}: the recorded motion is unreadable ({e}); re-run Simulate from "
                          "Blender") from None
    dynamic = [r["id"] for r in session["bodies"] if r["role"] != "rig" and r["trajectory"] >= 0]
    st = statics(ex, options_of(values), origin=session["origin_world"], dynamic_ids=dynamic)
    if st["refused"]:
        items = [f"{it['code']} {it['object']}: {it['message']}" for it in st["report"]["items"]
                 if it["severity"] in ("refuse", "error")]
        raise ConfigError(f"{ex.root}: the Blender scene is refused ({st['report']['policy']}): " + "; ".join(items))
    recorded = session.get("motion_key_inputs") or {"cameras": {}}
    reason = stale_reason(recorded, motion_key_inputs(values, ex, st, list(recorded["cameras"])))
    if reason:
        raise ConfigError(f"MOT-STALE: {ex.root}: the recorded motion no longer applies ({reason}); re-run Simulate "
                          "from Blender")
    return {"st": st, "session": session, "traj": traj}


def load_blender(values: Mapping[str, Any]):
    """The bundle of a Blender exchange source (``scene.path`` a directory holding ``exchange.json``): its cached
    conversion for the run's values. On a miss, a static scene (no motion session written, ``session_needed`` empty)
    is assembled from its statics and cached (a read-only exchange converts without caching). ConfigError when the
    conversion refuses the scene; FileNotFoundError when the scene needs the motion session and has no entry."""
    from ballasim.config import ConfigError
    from ballasim.scene.bundle import SceneBundle

    ex = read_exchange(values["scene.path"])
    key = blender_cache_key(values, ex)
    entry = ex.root / "cache" / key
    if (entry / "bundle.json").is_file():
        return SceneBundle.load(entry)
    if (ex.root / "motion" / "session.json").is_file():   # the session ran: its outputs convert again without Blender
        r = reconvert(ex, values)
        bundle = assemble_moving(ex, r["st"], r["session"], r["traj"], values)
        try:
            bundle.save(entry)
        except OSError:
            return bundle
        return SceneBundle.load(entry)
    st = statics(ex, options_of(values))
    if st["refused"]:
        items = [f"{it['code']} {it['object']}: {it['message']}" for it in st["report"]["items"]
                 if it["severity"] in ("refuse", "error")]
        raise ConfigError(f"{ex.root}: the Blender scene is refused ({st['report']['policy']}): " + "; ".join(items))
    if st["session_needed"]:
        raise FileNotFoundError(f"{ex.root}: the scene moves, so its bundle comes from the motion session, which this "
                                f"engine does not run yet ({'; '.join(st['session_needed'][:5])})")
    if "error" in st["timeline"]:
        raise ConfigError(f"{ex.root}: {st['timeline']['error']}")
    bundle = assemble_static(ex, st, values)
    try:
        bundle.save(entry)
    except OSError:
        return bundle                     # a read-only exchange: converted, not cached
    return SceneBundle.load(entry)
