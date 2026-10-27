"""Run configurations (docs/dev/P0_PARAMS.md §2): format-2 YAML files, profile files and CLI items, resolved in C++.

The resolver is ``ballasim._core.config`` (C++): layers, checks, family priors, accuracy profiles, derivations,
digests and records. This module is the file-system glue: it reads the YAML files (ruamel.yaml, safe mode), expands
the values of path-typed keys against the directory of the file that set them, finds the profile files and hands
everything over as plain values.

- :func:`load` (path or dict) -> :class:`RunSpec`: the layers and the resolved scene keys (no bundle yet);
- :func:`resolve` (spec, bundle) -> ``Resolved``: every value of the run on that bundle (``values()``, ``record()``,
  ``warnings``, ``digest(stage, camera)``, ``get(key, camera)``, ``cameras``, ``span``);
- :func:`missing_keys` (spec, bundle_info) -> every unresolved required key at once (collect mode, a Blender Check);
- a resolver error is :class:`ConfigError` with ``diagnostic``, the C++ ``config::Diagnostic`` as a dict (kind,
  severity, key, camera, layer, keys, message, nearest; docs/design/phase3/ui.md §3.10); ``Resolved.diagnostics``
  holds the warnings as such dicts;
- ``ballasim.run(spec | path, bundle=None)`` runs it (:mod:`ballasim.engine`).

A run configuration (format 2)::

    format: ballasim.run/2
    profiles: [evimo2/lab_lighting@v0]
    scene: {evimo2: {sequence: ${EVIMO2_ROOT}/left_camera/imo/eval/scene13_dyn_test_00_000000}}
    run: {t_start_ns: 1500000000, t_end_ns: 1750000000}
    cameras: {left_camera: {profiles: [evimo2/prophesee_gen3_left@v0]}}
    output: {dir: runs/s13}
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from ballasim import _core

#: The one search location of profile files: <group>/<name>@v<N>.yaml (P0_PARAMS §2.5).
PROFILES_DIR = Path(__file__).with_name("profiles")
_VAR = re.compile(r"\$(\w+|\{[^}]*\})")


class ConfigError(ValueError):
    """A run configuration, profile or CLI item that does not resolve; the message names the key and the layer
    (P0_PARAMS §2.4). The CLI prints it without a traceback. ``diagnostic``: the resolver's structured error
    (``_core.config.DiagnosticError.diagnostic``: kind, severity, key, camera, layer, keys, message, nearest), or None
    for an error found in Python (YAML, ``--set`` items, path variables)."""

    def __init__(self, message: str = "", diagnostic: dict | None = None):
        super().__init__(message)
        self.diagnostic = diagnostic


def _config_error(e: ValueError) -> ConfigError:
    """A resolver or loader error as :class:`ConfigError`, keeping a DiagnosticError's ``diagnostic``."""
    return ConfigError(str(e), getattr(e, "diagnostic", None))


def _yaml() -> YAML:
    return YAML(typ="safe")


@cache
def path_keys() -> frozenset[str]:
    """The path-typed keys (all outside camera blocks)."""
    return frozenset(r["key"] for r in _core.config.schema() if r["type"] == "path")


def expand_path(value: str, base: Path, key: str = "path") -> str:
    """``str((base / expandvars(value)).expanduser())``: the format-1 loader's rule, plus environment variables, of which
    an unset one is an error."""
    for m in _VAR.finditer(value):
        name = m.group(1).strip("{}")
        if name not in os.environ:
            raise ValueError(f"{key} = {value!r}: ${name} is not set")
    return str((base / os.path.expandvars(value)).expanduser())


def _expand(node: Any, prefix: str, base: Path) -> Any:
    if isinstance(node, dict):
        return {k: _expand(v, f"{prefix}.{k}" if prefix else str(k), base) for k, v in node.items()}
    if prefix in path_keys() and isinstance(node, (str, os.PathLike)):
        return expand_path(os.fspath(node), base, prefix)
    return node


def parse_sets(sets: Iterable[str] | Mapping[str, Any]) -> dict[str, Any]:
    """``--set KEY=VALUE`` items (VALUE: a YAML scalar or flow list), or a mapping of full keys to values."""
    if isinstance(sets, Mapping):
        return dict(sets)
    out = {}
    for item in sets:
        key, sep, text = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        out[key.strip()] = _yaml().load(text) if text.strip() else None
    return out


def profile_path(pid: str) -> Path:
    return PROFILES_DIR / f"{pid}.yaml"


def read_profile(pid: str) -> dict:
    """A profile file as plain values (its YAML document), e.g. the EVIMO2 release facts of the viewer (D16b)."""
    return _yaml().load(profile_path(pid).read_text())


@cache
def _evimo2_facts() -> tuple[tuple[str, str], ...]:
    """The derived EVIMO2 roots: the environment, or the machine defaults of ballasim.evimo2."""
    from ballasim.evimo2.clocks import raw_root
    from ballasim.evimo2.dataset import objects_dir

    return (("scene.evimo2.objects_root", str(objects_dir())), ("scene.evimo2.raw_root", str(raw_root())))


@dataclass
class RunSpec:
    """A loaded run: ``inputs`` for the resolver, ``record`` for run.json's ``inputs`` block (the configuration as read,
    the CLI items, every profile with its id, path and SHA-256), and ``scene``, the resolved scene keys."""

    inputs: dict
    record: dict
    scene: Any

    def get(self, key: str):
        return self.scene.get(key)


def load(source: str | os.PathLike | Mapping, *, sets: Iterable[str] | Mapping[str, Any] = (),
         profiles: Iterable[str] = (), base_dir: str | os.PathLike = ".",
         scene_path: str | os.PathLike | None = None) -> RunSpec:
    """Reads a run configuration (a YAML file, or a dict whose relative paths are against ``base_dir``) with the CLI
    layer (``sets``: --set items; ``profiles``: --profile ids, appended to the root list), loads the profiles it names
    and resolves the scene keys. ``scene_path``, a concrete scene source (a Blender Check's exchange), is the CLI
    layer's scene.path as is: absolute, never parsed or expanded, in place of any --set scene.path. Every layer is
    checked here, before any bundle is built: :class:`ConfigError` names the key and the layer; a missing file is
    FileNotFoundError."""
    try:
        return _load(source, sets, profiles, base_dir, scene_path)
    except ConfigError:
        raise
    except ValueError as e:
        raise _config_error(e) from None
    except YAMLError as e:
        raise ConfigError(str(e)) from None


def _load(source, sets, profiles, base_dir, scene_path=None) -> RunSpec:
    if isinstance(source, Mapping):
        raw, path, base, name = dict(source), None, Path(base_dir).absolute(), "<dict>"
    else:
        path = Path(source)
        raw, base, name = _yaml().load(path.read_text()), path.parent.absolute(), str(path)
    if not isinstance(raw, dict):
        raise ConfigError(f"{name}: a run configuration is a YAML mapping")
    cli = {k: expand_path(os.fspath(v), Path.cwd(), k) if k in path_keys() and isinstance(v, (str, os.PathLike)) else v
           for k, v in parse_sets(sets).items() if scene_path is None or k != "scene.path"}
    if scene_path is not None:
        cli["scene.path"] = str(Path(scene_path).absolute())
    evimo2 = "evimo2" in (raw.get("scene") or {}) or any(k.startswith("scene.evimo2.") for k in cli)
    inputs = {"name": name, "config": _expand(raw, "", base), "cli": cli, "cli_profiles": list(profiles),
              "profiles": {}, "facts": dict(_evimo2_facts()) if evimo2 else {}}
    record = {"config": raw, "file": None if path is None else str(path.absolute()),
              "cli": {"set": cli, "profiles": list(profiles)}, "profiles": []}
    for pid in _core.config.profiles_needed(inputs):
        p = profile_path(pid)
        if not p.is_file():
            raise FileNotFoundError(f"profile {pid}: no file {p}")
        data = p.read_bytes()
        doc = _yaml().load(data)
        if isinstance(doc, dict) and isinstance(doc.get("values"), dict):
            doc["values"] = {k: {**e, "value": expand_path(e["value"], p.parent, k)}
                             if k in path_keys() and isinstance(e, dict) and isinstance(e.get("value"), str) else e
                             for k, e in doc["values"].items()}
        inputs["profiles"][pid] = doc
        record["profiles"].append({"id": pid, "path": str(p), "sha256": hashlib.sha256(data).hexdigest()})
    scene = _core.config.resolve_scene(inputs)
    # The scene source, wherever scene.path was set (CLI, configuration, profile), after expansion (blender.md
    # §3.11): a Blender exchange resolves once more, with its scene.blender.* rows.
    inputs["source_kind"] = source_kind(scene.values()["run"])
    if inputs["source_kind"] == "blender":
        scene = _core.config.resolve_scene(inputs)
    return RunSpec(inputs, record, scene)


def source_kind(values: Mapping[str, Any]) -> str:
    """The kind of a run's scene source from its resolved scene keys, as ``ballasim.engine.scene.load_source`` reads
    it: ``evimo2`` (scene.evimo2.sequence), ``blender`` (a scene.path directory holding ``exchange.json``),
    ``bundle`` (another directory), ``yaml`` (a .yaml or .yml file), else ``""``."""
    if values.get("scene.evimo2.sequence") is not None:
        return "evimo2"
    if values.get("scene.path") is None:
        return ""
    path = Path(values["scene.path"])
    if path.is_dir():
        return "blender" if (path / "exchange.json").is_file() else "bundle"
    return "yaml" if path.suffix.lower() in (".yaml", ".yml") else ""


@cache
def engine_digest() -> str:
    """SHA-256 of the _engine module (chained into the stage digests from the render stage); empty without one."""
    try:
        from ballasim import _engine
    except ImportError:
        return ""
    return hashlib.sha256(Path(_engine.__file__).read_bytes()).hexdigest()


def device_available() -> bool:
    try:
        from ballasim import _engine
    except ImportError:
        return False
    return bool(_engine.device_available())


_device_available = device_available  # bundle_info's keyword of the same name shadows it


def bundle_info(bundle, *, supplied: bool, digest: str | None = None, low_light: bool | None = None,
                device_available: bool | None = None) -> dict:
    """The bundle facts resolution needs (cameras, span, device, digests, light mode, shadow-casting lights and a zero
    background). ``digest``: digest(scene), the stage digests' root (an EVIMO2 source's cache key; default the bundle
    digest); ``low_light``: an EVIMO2 bundle's light mode (both from ballasim.engine.scene.scene_facts);
    ``device_available``: whether the device engine can run here, None to ask ``_engine`` (which maps libcuda; the UI
    server passes its subprocess probe's answer, docs/design/phase3/ui.md §3.1)."""
    if digest is None:
        from ballasim.engine.scene import bundle_digest

        digest = bundle_digest(bundle)
    from ballasim.scene.baked import facts as baked_facts

    return {"baked": baked_facts(bundle), "cameras": [{"name": c.name, "sensor": c.sensor, "width": int(c.width), "height": int(c.height),
                         "clock_offset_ns": int(c.clock_offset_ns), "pixel_pitch_m": float(c.pixel_pitch_m),
                         "f_number": float(c.f_number)} for c in bundle.cameras],
            "t_start_ns": int(bundle.t_start_ns), "t_end_ns": int(bundle.t_end_ns),
            "device_available": _device_available() if device_available is None else bool(device_available),
            "supplied": supplied, "bundle_digest": digest,
            "engine_digest": engine_digest(), "low_light": low_light,
            # The band N's coverage warning (P2-B B4a): shadow-casting lights, and misses with no background light.
            "shadow_lights": sum(lt.kind not in ("ambient", "background") and bool(lt.casts_shadows)
                                 for lt in bundle.lights),
            "zero_background": not any(lt.kind == "background" and lt.intensity > 0 for lt in bundle.lights)}


def resolve(spec: RunSpec, bundle, *, supplied: bool = True, digest: str | None = None, low_light: bool | None = None,
            device_available: bool | None = None):
    """Every value of the run on ``bundle`` (``supplied``: the bundle replaces the scene keys; ``digest``,
    ``low_light`` and ``device_available``: see bundle_info); :class:`ConfigError` names the key and the layer of a
    value that does not resolve, with its ``diagnostic``."""
    info = bundle_info(bundle, supplied=supplied, digest=digest, low_light=low_light,
                       device_available=device_available)
    try:
        return _core.config.resolve(spec.inputs, info)
    except ValueError as e:
        raise _config_error(e) from None


def missing_keys(spec: RunSpec, info: Mapping[str, Any], *, diagnostics: bool = False) -> list[dict]:
    """Collect mode (docs/design/phase3/blender.md §3.11, the add-on's Check): every required key the run lacks on the
    bundle facts ``info`` (a ``bundle_info`` dict, e.g. :func:`ballasim.blender.convert.provisional_bundle_info`), at
    once, as ``[{key, hint, help}]`` (with ``diagnostics``, also ``diagnostic``: the error each would raise); [] when
    none is missing. Any other error is :class:`ConfigError`."""
    try:
        return _core.config.missing_keys(spec.inputs, dict(info), diagnostics=diagnostics)
    except ValueError as e:
        raise _config_error(e) from None
