"""The frames values of a finished run (docs/design/phase3/frames.md §3.2): :func:`resolve` rebuilds the run's layers
from run.json ``inputs`` (the recorded configuration, CLI items and profiles, through :func:`ballasim.config.load`),
adds the frames command's ``--set`` items on top and calls ``ballasim._frames.resolve_frames`` (C++: the main resolver for the
cameras of ``run.select``, ``config::resolve_section`` for those of ``run.frames_select``), so that the layering is the
run's own.

Checks (docs/design/phase3/frames.md §6): FRM-CONFIG-CHANGED when a profile file's SHA-256 differs from run.json's
record (or the file is gone), or when the frames values of a ``run.select`` camera that run.json recorded (a run made since the
frames rows exist) differ from the rebuilt ones; a run recorded before the rows existed gets the schema defaults. The
C++ side raises FRM-SET-SCOPE, FRM-VALUE, FRM-CAMERA and FRM-MODE-SOURCE. The record is compared on the run's layers
alone and the refusals apply to the layers with the ``--set`` items on top, so that a recorded value a check refuses
(the engine ignores the frames rows) can be repaired with ``--set``.

:func:`write_resolved` records the result as ``<run>/frames/resolved.json`` (atomically):
``{format, cameras: {name: {key: {value, layer, status}}}, run: {...}, sets: [...], implicit: {...}}``
(``implicit``: the items the command itself layered below the ``--set`` items).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ballasim import config

RESOLVED_FORMAT = "ballasim.frames-resolved/1"


def refusal(code: str, message: str) -> Exception:
    """A frames.md §6 refusal as ``ballasim._frames.FramesError`` (a ValueError whose message starts with the code and
    whose ``code`` attribute holds it)."""
    from ballasim import _frames

    e = _frames.FramesError(f"{code}: {message}")
    e.code = code
    return e


@dataclass
class FramesResolution:
    """The frames values of a run: ``cameras`` (``_frames.resolve_frames``' list: name, index, events, values,
    clock_offset_ns, clock_drift_ppm, contrast, digest_values), ``run`` (execution.frames_* and run.frames_select),
    ``warnings`` and the ``sets`` they were resolved under; ``doc`` is run.json, ``spec`` the rebuilt layers,
    ``info`` the bundle facts they resolved on, ``bundle`` and ``rvalues`` what the frames render with."""

    run_dir: Path
    doc: dict
    spec: config.RunSpec
    info: dict
    bundle: object                  # the run's SceneBundle (digest-checked)
    rvalues: dict                   # the run's recorded values {run: {...}, cameras: {...}} (dense.recorded_values)
    cameras: list[dict]
    run: dict
    warnings: list[str]
    sets: list[str] = field(default_factory=list)
    implicit: dict = field(default_factory=dict)       # command-made items below the sets (e.g. frames.mode)

    @property
    def names(self) -> list[str]:
        return [c["name"] for c in self.cameras]

    def camera(self, name: str) -> dict:
        """Camera ``name``'s record; FRM-CAMERA when it is in neither ``run.select`` nor ``run.frames_select``."""
        for c in self.cameras:
            if c["name"] == name:
                return c
        bundle_cameras = [c.name for c in self.bundle.cameras]
        if name not in bundle_cameras:
            raise refusal("FRM-CAMERA", f"cameras.{name}: not a camera of the bundle ({', '.join(bundle_cameras)})")
        raise refusal("FRM-CAMERA", f"cameras.{name}: simulated by no selection; frames exist for run.select "
                                    f"({', '.join(self.names)}) and run.frames_select cameras")

    def values(self, name: str) -> dict:
        """Camera ``name``'s frames values as {key: value}."""
        return {k: e["value"] for k, e in self.camera(name)["values"].items()}


def read_run(run_dir: str | Path) -> dict:
    """run.json of a run directory (ValueError for a format-1 run or one that records no inputs)."""
    path = Path(run_dir) / "run.json"
    doc = json.loads(path.read_text())
    inputs = doc.get("inputs")
    if "resolved" not in doc or not isinstance(inputs, dict) or not isinstance(inputs.get("config"), dict):
        raise ValueError(f"{path}: not a format-2 run (no resolved block or recorded configuration); migrate it with "
                         f"python tools/config/convert_v1.py --runs DIR")
    return doc


def rebuild_spec(doc: dict) -> config.RunSpec:
    """The run's layers from run.json ``inputs`` (the configuration as read, relative to its file's directory; the CLI
    items; the profile ids) through :func:`ballasim.config.load`. FRM-CONFIG-CHANGED when a profile file's SHA-256 is no
    longer the recorded one or the file is gone."""
    inputs = doc["inputs"]
    cli = inputs.get("cli") or {}
    base = Path(inputs["file"]).parent if inputs.get("file") else Path.cwd()
    try:
        spec = config.load(copy.deepcopy(inputs["config"]), sets=dict(cli.get("set") or {}),
                           profiles=list(cli.get("profiles") or []), base_dir=base)
    except FileNotFoundError as e:
        raise refusal("FRM-CONFIG-CHANGED", f"the run's configuration no longer loads: {e}") from None
    now = {p["id"]: p["sha256"] for p in spec.record["profiles"]}
    for p in inputs.get("profiles") or []:
        if now.get(p["id"]) != p["sha256"]:
            raise refusal("FRM-CONFIG-CHANGED", f"profile {p['id']} has changed since the run (SHA-256 "
                                                f"{p['sha256'][:12]} recorded, {str(now.get(p['id']))[:12]} now)")
    return spec


def _same(recorded, value) -> bool:
    return recorded == value


def _check_record(doc: dict, base: dict) -> None:
    """The frames values run.json recorded (a run made since the rows exist) equal the rebuilt ones: FRM-CONFIG-CHANGED
    otherwise. A run recorded before the rows existed holds none, and gets the schema defaults."""
    rec = doc["resolved"]
    moved = []
    for c in base["cameras"]:
        if not c["events"]:
            continue
        recorded = {k: e for k, e in rec.get("cameras", {}).get(c["name"], {}).items() if k.startswith("frames.")}
        for key, e in recorded.items():
            if key == "frames.roi_px":      # resolved against run.crop_px: compared where it is a value
                continue
            now = c["values"].get(key)
            if now is None or not _same(e["value"], now["value"]):
                moved.append(f"cameras.{c['name']}.{key} (recorded {e['value']!r})")
    for key, e in rec.get("run", {}).items():
        if key in base["run"] and not _same(e["value"], base["run"][key]["value"]):
            moved.append(f"{key} (recorded {e['value']!r})")
    if moved:
        raise refusal("FRM-CONFIG-CHANGED", "the frames values of the run differ from run.json's record: " +
                                            "; ".join(moved))


def resolve(run_dir: str | Path, sets: Iterable[str] | dict = (), *, bundle=None, implicit: dict | None = None,
            doc: dict | None = None) -> FramesResolution:
    """The frames values of the cameras of ``run_dir`` (see the module docstring). ``sets``: the ``--set`` items of the
    frames command (``KEY=VALUE`` strings or a mapping of full keys), the top layer; ``implicit``: command-made items
    below them (for example ``cameras.left.frames.mode`` for ``frames reference --camera left``). ``bundle`` replaces
    loading the run's scene (it must have the run's digest)."""
    from ballasim import _frames
    from ballasim.engine.dense import recorded_values
    from ballasim.engine.scene import bundle_digest, load_bundle

    run_dir = Path(run_dir)
    doc = read_run(run_dir) if doc is None else doc
    spec = rebuild_spec(doc)
    values, _ = recorded_values(doc)
    if bundle is None:
        bundle = load_bundle(values["run"])
    if bundle_digest(bundle) != doc["bundle"]["digest"]:
        raise ValueError(f"{run_dir}: the scene has changed since the run (bundle digest differs)")
    # The bundle facts as the run resolved them: a supplied bundle (the scene keys are not needed), the device taken
    # as present (the frames values do not depend on it, and a run made on the GPU resolves on a host without one).
    info = config.bundle_info(bundle, supplied=True, digest=doc["bundle"]["digest"], device_available=True)
    parsed = config.parse_sets(sets)
    items = [f"{k}={json.dumps(v)}" for k, v in parsed.items()] if isinstance(sets, dict) else list(sets)
    # The record first and on its own (check=False: no refusal of a value, which a --set item below may repair), then
    # the checked values of the layers with the command's items on top.
    _check_record(doc, _frames.resolve_frames(spec.inputs, info, None, check=False))
    layered = {**(implicit or {}), **parsed}
    out = _frames.resolve_frames(spec.inputs, info, layered or None)
    return FramesResolution(run_dir, doc, spec, info, bundle, values, out["cameras"], out["run"],
                            list(out["warnings"]), items, dict(implicit or {}))


def write_resolved(res: FramesResolution) -> Path:
    """``<run>/frames/resolved.json`` (atomically): every camera's frames values with their layer and status, the run
    rows and the ``--set`` items."""
    from ballasim.engine.runlog import write_json_atomic

    out = res.run_dir / "frames"
    out.mkdir(exist_ok=True)
    path = out / "resolved.json"
    doc = {"format": RESOLVED_FORMAT,
           "cameras": {c["name"]: {k: {"value": e["value"], "layer": e["layer"], "status": e["status"]}
                                   for k, e in c["values"].items()} for c in res.cameras},
           "run": {k: {"value": e["value"], "layer": e["layer"], "status": e["status"]} for k, e in res.run.items()},
           "sets": list(res.sets), "implicit": dict(res.implicit)}
    write_json_atomic(path, doc)
    return path
