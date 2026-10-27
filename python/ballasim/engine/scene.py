"""Scene sources of a run, from its resolved scene keys (docs/dev/P0_PARAMS.md §3.4): ``scene.path`` (a bundle
directory or a scene YAML) or ``scene.evimo2.sequence`` (built once, then cached and relit at every load).

EVIMO2 bundles are cached under ``$BALLASIM_CACHE/v1/bundles/<key>/``, unlit (``build_bundle(lit=False)``), with a
sidecar ``relight.json`` (``ballasim.evimo2.adapter.relight_sidecar``). The key is digest(scene)
(``_core.config.scene_digest``): the scene-stage keys (adapter and fit options, albedos, the EVIMO2 roots), the camera's
sequence folder (as the bundle meta records it, however the sequence and camera keys spell it), the importer version,
the source of every module that shapes a bundle (``ballasim.scene``, ``ballasim.traj``, ``ballasim.evimo2``) and the
compiled build (``_scene``, the C++ adapter and fitter, with the core code it links), the OpenBLAS kernel and NumPy
SIMD target whose operation order it reproduces (docs/dev/CPP_PORT.md item 5, S4), so editing the adapter, the fitter or
the libraries under them rebuilds instead of reusing a stale bundle. (``_core`` is not in it: the build never calls it,
tests/python/test_scene_cpp.py.) The lighting keys are not in it: every load relights the cached bundle
from the resolved lighting (``relight``), so a lighting edit costs a load, not a rebuild. EVIMO2 bundles are not
redistributable (SCENE.md); the cache stays local.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ballasim import _core
from ballasim.scene.assets import DEFAULT_CACHE
from ballasim.scene.bundle import BundleError, SceneBundle, at_feature_default, content_fields

log = logging.getLogger(__name__)

#: Bumped when the cache layout changes (2: the relight sidecar; the key is digest(scene)).
BUNDLE_CACHE_VERSION = 2
_BUNDLE_PACKAGES = ("scene", "traj", "evimo2")
SIDECAR = "relight.json"


def bundle_cache_root() -> Path:
    return Path(os.environ.get("BALLASIM_CACHE", DEFAULT_CACHE)) / "v1" / "bundles"


def evimo2_sequence_dir(sequence: str, camera: str | None = None) -> Path:
    """The sequence folder of the requested camera: ``<root>/<camera>/<category>/<split>/<name>``."""
    from ballasim.evimo2.dataset import CAMERA_FOLDERS

    seq = Path(sequence).absolute()
    parts = seq.parts
    cams = [i for i, p in enumerate(parts) if p in CAMERA_FOLDERS]
    if not cams:
        raise ValueError(f"{seq}: not inside an EVIMO2 camera folder {CAMERA_FOLDERS}")
    if camera is None or camera == parts[cams[-1]]:
        return seq
    if camera not in CAMERA_FOLDERS:
        raise ValueError(f"scene.evimo2.camera must be one of {CAMERA_FOLDERS}, got {camera!r}")
    other = Path(*parts[:cams[-1]], camera, *parts[cams[-1] + 1:])
    if not other.is_dir():
        raise FileNotFoundError(f"{other}: the sequence has no {camera} recording")
    return other


def _source_digest(packages: tuple[str, ...] = _BUNDLE_PACKAGES) -> str:
    """SHA-256 over the ``.py`` and ``.json`` files of the named ``ballasim`` subpackages (paths and bytes): the EVIMO2
    key's ``scene``, ``traj`` and ``evimo2`` by default, a Blender bundle's ``scene``, ``traj`` and ``blender``."""
    import ballasim

    root = Path(ballasim.__file__).parent
    h = hashlib.sha256()
    for package in packages:
        for f in sorted((root / package).rglob("*")):
            if f.suffix in (".py", ".json") and f.is_file():
                h.update(str(f.relative_to(root)).encode())
                h.update(f.read_bytes())
    return h.hexdigest()


def _build_facts() -> dict:
    """What the build's bits rest on besides its inputs: the SHA-256 of ``ballasim._scene`` (the C++ adapter, fitter and
    PLY import, with the core code they link), ``openblas_get_config()`` (OpenBLAS's version and DYNAMIC_ARCH kernel)
    and SciPy's version (its Rotation makes the cameras' T_rig_cam in the Python that reads the sequence). NumPy's SIMD
    dispatch is not in it: no NumPy kernel shapes a bundle (the C++ reproduces their order, it does not call them)."""
    import scipy

    from ballasim import _scene

    return {"scene": hashlib.sha256(Path(_scene.__file__).read_bytes()).hexdigest(), "openblas": _scene.openblas_config(),
            "scipy": scipy.__version__}


def _raw_extrinsics(folders: list[Path], raw_root: str | None) -> list[str]:
    """Per camera folder, the SHA-256 of the raw archive's extrinsics.txt the build reads (its line 3 becomes the bundle
    meta's ``raw_line3_s``, and the clock offset under ``clock_reference: raw``), or "absent"."""
    from ballasim.evimo2.clocks import raw_extrinsics_path
    from ballasim.scene.assets import file_sha256

    out = []
    for d in folders:
        p = raw_extrinsics_path(d, None if raw_root is None else Path(raw_root))
        out.append(file_sha256(p) if p is not None and p.exists() else "absent")
    return out


def evimo2_options(values: Mapping[str, Any]):
    """The adapter options of a run's resolved EVIMO2 scene keys (``values``: key -> value): every scene.evimo2 and
    scene.trajectory key, the lighting while the lights are on, and the worker count."""
    from ballasim.evimo2 import Evimo2Options
    from ballasim.evimo2.adapter import OPTION_KEYS, Evimo2Lighting
    from ballasim.traj.fit import SCHEMA_FIELDS, SmoothingOptions

    smoothing = {k: values[f"scene.trajectory.{k}"] for k in SCHEMA_FIELDS}
    smoothing["gcv_log10_range"] = tuple(smoothing["gcv_log10_range"])
    seq = values["scene.evimo2.sequence"]
    return Evimo2Options(
        **{f: values[k] for f, k in OPTION_KEYS.items()},
        extra_cameras=[str(evimo2_sequence_dir(seq, c)) for c in values["scene.evimo2.extra_cameras"]],
        smoothing=SmoothingOptions(**smoothing), objects_root=values["scene.evimo2.objects_root"],
        raw_root=values["scene.evimo2.raw_root"],
        lighting=Evimo2Lighting.from_values(values) if values["scene.evimo2.lights"] else None)


def evimo2_lab_options(sequence: str | os.PathLike, *, workers: int = 1, **evimo2):
    """The adapter options of an EVIMO2 sequence resolved with the v0 lab lighting (evimo2/lab_lighting@v0) and
    ``scene.evimo2`` values (e.g. ``markers=False``): for tools and tests that build bundles outside a run."""
    from ballasim import config

    spec = config.load({"format": "ballasim.run/2", "profiles": ["evimo2/lab_lighting@v0"],
                        "scene": {"evimo2": {"sequence": os.fspath(sequence), **evimo2}},
                        "execution": {"scene_workers": workers}, "output": {"dir": "unused"}})
    return evimo2_options(spec.scene.values()["run"])


#: The keys the cache key holds by what they resolve to: the sequence as the camera's folder, the object catalogue as
#: the content of the files the build reads.
_RESOLVED_KEYS = ("scene.evimo2.sequence", "scene.evimo2.camera", "scene.evimo2.objects_root")


def evimo2_cache_key(values: Mapping[str, Any]) -> str:
    """digest(scene) of an EVIMO2 source: the cache key of its bundle. The sequence enters as the camera's folder, the
    string the bundle meta records (``source.sequence``), so every spelling of one folder (the left path with
    ``camera: right_camera``, or the right path) shares one entry. The object catalogue enters as the SHA-256 of
    every catalogue file the build reads (``evimo2.adapter.catalogue_inputs``), not as its path: editing or switching
    the catalogue (scene.evimo2.objects_root) moves the key, equal content at another path keeps it. The raw archive's
    extrinsics.txt of each camera folder enters the same way (or as "absent"): it sets the meta's clocks."""
    from ballasim.evimo2.adapter import catalogue_inputs
    from ballasim.scene.assets import IMPORTER_VERSION

    seq = evimo2_sequence_dir(values["scene.evimo2.sequence"], values.get("scene.evimo2.camera"))
    inputs = catalogue_inputs(seq, objects_root=values.get("scene.evimo2.objects_root"),
                              objects=values.get("scene.evimo2.objects"))
    folders = [seq] + [evimo2_sequence_dir(values["scene.evimo2.sequence"], c)
                       for c in values.get("scene.evimo2.extra_cameras") or []]
    facts = {"cache": str(BUNDLE_CACHE_VERSION), "adapter": "evimo2", "sequence_dir": str(seq),
             "importer": str(IMPORTER_VERSION), "source": _source_digest(), **_build_facts(),
             "catalogue": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
             "raw_extrinsics": ",".join(_raw_extrinsics(folders, values.get("scene.evimo2.raw_root")))}
    return _core.config.scene_digest({k: v for k, v in values.items() if k not in _RESOLVED_KEYS}, facts)


#: The lighting keys of each light group relight builds (for the messages of a rig the sequence refuses).
_GROUP_KEYS = {False: "scene.evimo2.mains_f0_hz, mains_a and mains_phi_rad (the mains light group)",
               True: "scene.evimo2.strobe_width_s, strobe_rise_s, strobe_phase_offset_s and strobe_floor (the strobe "
                     "light group, at the sequence's Vicon period)"}


def load_evimo2(values: Mapping[str, Any], *, use_cache: bool = True) -> SceneBundle:
    """Builds (or loads from the cache) the bundle of an EVIMO2 sequence and relights it from the resolved lighting. A
    build is cached before it is lit, so lighting the sequence's rig refuses costs no build (ConfigError naming the
    keys)."""
    from ballasim.config import ConfigError
    from ballasim.engine.runlog import write_json_atomic
    from ballasim.evimo2 import build_bundle
    from ballasim.evimo2.adapter import Evimo2Lighting, relight

    entry = bundle_cache_root() / evimo2_cache_key(values)
    seq = evimo2_sequence_dir(values["scene.evimo2.sequence"], values.get("scene.evimo2.camera"))
    if use_cache and (entry / "bundle.json").exists() and (entry / SIDECAR).exists():
        log.info("EVIMO2 bundle from the cache: %s", entry)
        bundle, sidecar = SceneBundle.load(entry), json.loads((entry / SIDECAR).read_text())
    else:
        log.info("building the EVIMO2 bundle of %s", seq)
        build = build_bundle(seq, evimo2_options(values), lit=False)
        bundle, sidecar = build.bundle, build.sidecar
        if use_cache:
            bundle.save(entry)                           # atomic; the entry counts as cached once the sidecar is in
            write_json_atomic(entry / SIDECAR, sidecar)
    lighting = Evimo2Lighting.from_values(values) if values["scene.evimo2.lights"] else None
    try:
        return relight(bundle, sidecar, lighting)
    except BundleError as e:                         # the light groups' checks (C1); single keys passed at load
        raise ConfigError(f"{_GROUP_KEYS[bool(sidecar['low_light'])]}: refused for {seq}: {e}") from None


def load_bundle(values: Mapping[str, Any]) -> SceneBundle:
    """The SceneBundle a run renders, of its resolved scene keys (``values``: key -> value, e.g.
    ``Resolved.values()["run"]``): its source's (load_source) with the run's background light (with_background)."""
    return render_bundle(load_source(values), values)


def load_source(values: Mapping[str, Any]) -> SceneBundle:
    """The bundle of a run's scene source: an EVIMO2 sequence's (relit), a Blender exchange's (``scene.path`` a
    directory holding ``exchange.json``: its converted bundle, ``ballasim.blender.convert.load_blender``), or
    scene.path's as the file gives it, the bundle whose digest is digest(scene) (scene_facts; a render-stage key such
    as scene.background_radiance does not move it)."""
    if values.get("scene.evimo2.sequence") is not None:
        return load_evimo2(values)
    if values.get("scene.path") is None:
        raise ValueError("scene.path or scene.evimo2.sequence is required (or a bundle given to the run)")
    path = Path(values["scene.path"])
    if path.is_dir() and (path / "exchange.json").is_file():
        from ballasim.blender.convert import load_blender

        return load_blender(values)
    if path.is_dir():
        return SceneBundle.load(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        from ballasim.scene.authoring import load_scene

        return load_scene(path)
    raise ValueError(f"scene.path {path}: expected a bundle directory or a scene YAML file")


def with_background(bundle: SceneBundle, values: Mapping[str, Any]) -> SceneBundle:
    """A scene.path source with ``scene.background_radiance`` (D13 item 1) added as a white background light in the
    light group named ``scene.background_group`` (checked at any radiance); at 0 (the default) the bundle itself. An
    EVIMO2 source is returned as is: scene.evimo2.background_albedo is applied at relight."""
    if values.get("scene.evimo2.sequence") is not None:
        return bundle
    from ballasim.config import ConfigError
    from ballasim.scene.bundle import Light

    group = values.get("scene.background_group", "default")
    names = [g.name for g in bundle.light_groups]
    if group not in names:
        raise ConfigError(f"scene.background_group = {group!r}: the scene has no such light group ({names})")
    radiance = float(values.get("scene.background_radiance") or 0.0)
    if radiance <= 0.0:
        return bundle
    light = Light("background", names.index(group), np.ones(3, np.float32), radiance, casts_shadows=False)
    return replace(bundle, lights=[*bundle.lights, light]).validate()


def with_baked_scale(bundle: SceneBundle, values: Mapping[str, Any]) -> SceneBundle:
    """Apply scene.baked_scale once to baked K (rounded to float32) and every proxy light (scans.md §3.4).
    The source bundle is untouched; beta 1 returns it by identity so defaults preserve its digest.
    """
    beta = float(values.get("scene.baked_scale", 1.0))
    if beta == 1.0:
        return bundle
    bodies = []
    with np.errstate(over="ignore", invalid="ignore"):
        for body in bundle.bodies:
            mats = [replace(m, baked_scale=float(np.float32(float(m.baked_scale) * beta)))
                    if m.shading == "baked" else m for m in body.materials]
            bodies.append(replace(body, materials=mats))
    lights = [replace(light, intensity=light.intensity * beta) if light.proxy else light for light in bundle.lights]
    return replace(bundle, bodies=bodies, lights=lights).validate()


def render_bundle(source: SceneBundle, values: Mapping[str, Any]) -> SceneBundle:
    """The bundle every run, dense re-render and resolve report renders (§3.4)."""
    return with_baked_scale(with_background(source, values), values)


def scene_facts(values: Mapping[str, Any], bundle: SceneBundle, *, digest: str | None = None) -> dict:
    """What resolution needs of a run's loaded scene besides its cameras and span (``ballasim.config.resolve``'s
    keywords): ``digest``, digest(scene) (P0_PARAMS §1.2: an EVIMO2 source's cache key, else the digest of
    ``bundle``, the source's bundle (load_source), ``digest`` if already computed), and for an EVIMO2 source
    ``low_light``, the bundle's light mode (which lighting keys relight applied)."""
    if values.get("scene.evimo2.sequence") is not None:
        from ballasim.evimo2.adapter import bundle_low_light

        return {"digest": evimo2_cache_key(values), "low_light": bundle_low_light(bundle)}
    return {"digest": digest if digest is not None else bundle_digest(bundle), "low_light": None}


def light_group_record(bundle: SceneBundle) -> list[dict]:
    """The bundle's light groups as run.json records them (C1 ``LightGroup`` dicts, P0c): the waveforms the run's
    sensors saw, which ``viz stats`` (the light frequencies, D16b) and ``tools/bench/sensor_replay.py`` read."""
    return [{"name": g.name, "waveform": json.loads(json.dumps(g.waveform))} for g in bundle.light_groups]


def bundle_digest(bundle: SceneBundle) -> str:
    """SHA-256 over the bundle's content (names, scalars, arrays with dtype and shape, meta), in field order. C1
    version-2 feature fields at their defaults are skipped (``scene.bundle.content_fields``), so no bundle that leaves
    a feature unused changes its digest; the file's version is not hashed."""
    h = hashlib.sha256()

    def walk(obj, path: str) -> None:
        h.update(path.encode())
        if is_dataclass(obj):
            for f in content_fields(obj):
                walk(getattr(obj, f.name), f"{path}/{f.name}")
        elif isinstance(obj, (list, tuple)):
            h.update(f"[{len(obj)}]".encode())
            for i, x in enumerate(obj):
                walk(x, f"{path}/{i}")
        elif isinstance(obj, np.ndarray):
            a = np.ascontiguousarray(obj)
            h.update(f"{a.dtype.str}{a.shape}".encode())
            h.update(a.tobytes())
        elif isinstance(obj, dict):
            h.update(json.dumps(obj, sort_keys=True, default=str).encode())
        else:
            h.update(repr(obj).encode())

    walk(bundle, "")
    return h.hexdigest()


def bundle_diff(a: SceneBundle, b: SceneBundle) -> str | None:
    """The first place where two bundles differ as :func:`bundle_digest` sees them (its path, and the two types or
    values there), or None when their digests agree: field types (an ``np.int64`` where an int was), array dtypes,
    shapes and the first differing element, and inside the meta the first key whose JSON differs."""

    def as_json(x) -> str:
        return json.dumps(x, sort_keys=True, default=str)

    def walk(x, y, path: str) -> str | None:
        if isinstance(x, dict) and isinstance(y, dict):
            if as_json(x) == as_json(y):
                return None
            if sorted(map(str, x)) != sorted(map(str, y)):
                return f"{path}: keys {sorted(map(str, x))} != {sorted(map(str, y))}"
            for k in x:
                d = walk(x[k], y[k], f"{path}/{k}")
                if d:
                    return d
            return f"{path}: {as_json(x)[:200]} != {as_json(y)[:200]}"
        if type(x) is not type(y):
            return f"{path}: {type(x).__name__} != {type(y).__name__}"
        if is_dataclass(x):
            for f in fields(x):
                if at_feature_default(x, f) and at_feature_default(y, f):
                    continue                     # both skipped by bundle_digest
                d = walk(getattr(x, f.name), getattr(y, f.name), f"{path}/{f.name}")
                if d:
                    return d
            return None
        if isinstance(x, (list, tuple)):
            if len(x) != len(y):
                return f"{path}: length {len(x)} != {len(y)}"
            for i, (u, v) in enumerate(zip(x, y)):
                d = walk(u, v, f"{path}/{i}")
                if d:
                    return d
            return None
        if isinstance(x, np.ndarray):
            if x.dtype != y.dtype or x.shape != y.shape:
                return f"{path}: {x.dtype.str}{x.shape} != {y.dtype.str}{y.shape}"
            u, v = np.ascontiguousarray(x).ravel(), np.ascontiguousarray(y).ravel()
            if u.tobytes() == v.tobytes():
                return None
            ub, vb = u.view(np.uint8).reshape(len(u), -1), v.view(np.uint8).reshape(len(v), -1)
            i = int(np.flatnonzero((ub != vb).any(1))[0])
            return f"{path}: element {np.unravel_index(i, x.shape)}: {u[i]!r} != {v[i]!r}"
        if isinstance(x, dict) or isinstance(y, dict):
            return None if as_json(x) == as_json(y) else f"{path}: {as_json(x)[:200]} != {as_json(y)[:200]}"
        return None if repr(x) == repr(y) else f"{path}: {x!r} != {y!r}"

    return walk(a, b, "")
