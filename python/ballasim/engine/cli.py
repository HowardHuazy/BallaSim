"""Command line of the engine (docs/dev/ENGINE.md): ``python -m ballasim run <config.yaml>``, ``evimo2 <sequence>``,
``dense <run> [--out DIR]``, ``params [KEY]``, ``resolve <config.yaml>``, ``scene build <config.yaml>`` (builds and
caches the scene source's bundle only), ``ui`` (the browser UI, ballasim.ui; imported when
used), ``job <jobdir>`` (the ``ballasim.job/1`` runner, :mod:`.job`), ``probe``, ``profiles``, ``blender check
--exchange DIR``, ``blender session`` and ``blender replay`` (:mod:`ballasim.blender`), ``frames reference|aps <run>``
(docs/design/phase3/frames.md §3.9) and ``scan report PATH|SCENE.yaml#ASSET`` (docs/design/phase3/scans.md §3.6);
``imu <run>`` and ``export <run>`` come from :mod:`ballasim.export.cli`
(docs/dev/IMU.md, EXPORT.md). Run configurations are format 2 (docs/dev/P0_PARAMS.md); ``--set KEY=VALUE`` and
``--profile ID`` form the CLI layer."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:     # the engine, the renderer and the scene side load per command: ``job`` starts without them
    from .engine import RunSummary


def seconds_to_ns(text: str) -> int:
    """The argparse type of ``--t0`` and ``--t1`` (``evimo2``, ``dense`` and ``export``): decimal seconds to integer
    nanoseconds, truncated toward zero. Text that is not a finite number, or whose nanoseconds do not fit an int64, is a
    usage error (exit status 2)."""
    try:
        ns = Decimal(text) * 1_000_000_000
        if ns.is_finite() and abs(ns) < 2**63:  # checked before int(), which would build a huge integer
            return int(ns)
    except ArithmeticError:  # decimal.InvalidOperation (not a number) or decimal.Overflow (beyond Decimal's range)
        pass
    raise argparse.ArgumentTypeError(f"not a finite number of seconds within the int64 ns range: {text!r}")


def _report(summary: RunSummary) -> str:
    sim = summary.simulated_s
    head = (f"run {summary.run_dir}: {sim:.4f} s simulated in {summary.wall_s:.1f} s "
            f"({summary.wall_s / sim:.1f} s per simulated s), backend {summary.backend}")
    lines = [head]
    for c in summary.cameras:
        rays_per_s = c.rays / max(c.render_s, 1e-9) / 1e9
        lines.append(f"  {c.name}: {c.events} events, {c.pixel_samples / sim / 1e6:.1f} M pixel-samples/s simulated, "
                     f"render {c.render_s:.1f} s ({rays_per_s:.2f} G rays/s), "
                     f"sensor {c.sense_s:.1f} s ({c.sensor_device}), GT {c.gt_s:.1f} s -> {c.path}")
    return "\n".join(lines)


def resolved_lines(resolved, *, every: bool = False) -> list[str]:
    """The resolved values with their layers: those a profile, the configuration or the CLI set (every value with
    ``every``), then the warnings."""
    record = resolved.record()
    blocks = [("", record["run"])] + [(f"cameras.{c}.", v) for c, v in record["cameras"].items()]
    lines = []
    for prefix, entries in blocks:
        for key, e in entries.items():
            if every or e["layer"].startswith("profile:") or e["layer"] in ("config", "cli"):
                lines.append(f"  {prefix}{key} = {json.dumps(e['value'])}  ({e['layer']}, {e['status']})")
    lines += [f"  warning: {w}" for w in resolved.warnings]
    return lines


def evimo2_doc(sequence: str, *, camera: str | None = None, t0_ns: int | None = None, t1_ns: int | None = None,
               out: str, profile_set: str | None = None, no_profiles: bool = False, camera_profiles=()) -> dict:
    """The format-2 run configuration of ``ballasim evimo2``: the sequence, span and output, and (unless
    ``no_profiles``) the defaults of profiles/evimo2/index.yaml (D5a): its root profiles and the camera folder's sensor
    profile, of the set ``profile_set`` names (v0, the top level, by default; others under ``sets``). A sequence
    outside a camera folder without ``camera``, an unknown set, a set with ``no_profiles`` and a normal-light set on a
    ``_ll`` sequence raise ConfigError (the CLI exits 2; the UI answers BS-UI-JOB-KIND)."""
    from ruamel.yaml import YAML

    from ballasim.config import PROFILES_DIR, ConfigError
    from ballasim.evimo2.dataset import CAMERA_FOLDERS, low_light_category

    folders = [p for p in Path(sequence).absolute().parts if p in CAMERA_FOLDERS]
    folder = camera or (folders[-1] if folders else None)
    if camera is not None and camera not in CAMERA_FOLDERS:
        raise ConfigError(f"--camera {camera}: one of {', '.join(CAMERA_FOLDERS)}")
    if folder is None:
        raise ConfigError(f"{sequence}: not inside an EVIMO2 camera folder {CAMERA_FOLDERS}; name one with --camera")
    doc = {"format": "ballasim.run/2", "scene": {"evimo2": {"sequence": sequence, "camera": camera}},
           "run": {"t_start_ns": t0_ns, "t_end_ns": t1_ns}, "output": {"dir": out}}
    if no_profiles and profile_set is not None:
        raise ConfigError("--profile-set picks the default profiles, which --no-profiles drops: give one of them")
    if not no_profiles:
        index = YAML(typ="safe").load((PROFILES_DIR / "evimo2" / "index.yaml").read_text())
        if profile_set not in (None, "v0"):
            sets = index.get("sets", {})
            if profile_set not in sets:
                raise ConfigError(f"--profile-set {profile_set}: profiles/evimo2/index.yaml has v0, {', '.join(sets)}")
            index = sets[profile_set]
            if index.get("normal_light_only") and any(map(low_light_category, Path(sequence).absolute().parts)):
                raise ConfigError(f"--profile-set {profile_set}: normal-light sequences only (its photocurrent is "
                                  f"stated for the lab lights and would cancel the low-light dimming); {sequence} "
                                  f"is a _ll sequence")
        doc["profiles"] = list(index["root"])
        doc["cameras"] = {folder: {"profiles": list(index["cameras"][folder])}}
    for item in camera_profiles:
        name, sep, pid = item.partition("=")
        if not sep or not name or not pid:
            raise ConfigError(f"--camera-profile expects CAMERA=ID, got {item!r}")
        if name != folder:
            raise ConfigError(f"--camera-profile {item}: the sequence's camera is {folder}")
        doc.setdefault("cameras", {}).setdefault(folder, {}).setdefault("profiles", []).append(pid)
    return doc


def _evimo2_doc(args) -> dict:
    """:func:`evimo2_doc` of the ``evimo2`` command's arguments."""
    return evimo2_doc(args.sequence, camera=args.camera, t0_ns=args.t0, t1_ns=args.t1, out=args.out,
                      profile_set=args.profile_set, no_profiles=args.no_profiles,
                      camera_profiles=getattr(args, "camera_profile", ()))


def scene_build(config_path: str, sets=(), profiles=()) -> dict:
    """``scene build``: loads the configuration's scene source as a run would, which builds and caches its bundle (an
    EVIMO2 sequence's under ``$BALLASIM_CACHE``, a Blender exchange's in its ``cache/``, a scene YAML's assets), and
    resolves nothing else, so a configuration without a sensor family builds too (docs/design/phase3/ui.md §4.12).
    Returns ``{kind: evimo2|blender|yaml|bundle, key, source}``: the key is the bundle's cache key (EVIMO2,
    Blender) or content digest."""
    from ballasim import config

    from .scene import bundle_digest, evimo2_cache_key, load_source

    values = config.load(config_path, sets=sets, profiles=profiles).scene.values()["run"]
    source = load_source(values)
    if values.get("scene.evimo2.sequence") is not None:
        return {"kind": "evimo2", "key": evimo2_cache_key(values), "source": values["scene.evimo2.sequence"]}
    path = Path(values["scene.path"])
    if path.is_dir() and (path / "exchange.json").is_file():
        from ballasim.blender.convert import blender_cache_key, read_exchange

        return {"kind": "blender", "key": blender_cache_key(values, read_exchange(path)), "source": str(path)}
    return {"kind": "bundle" if path.is_dir() else "yaml", "key": bundle_digest(source), "source": str(path)}


def _params(key: str | None, as_json: bool) -> int:
    from ballasim import _core

    rows = _core.config.schema()
    if key:
        rows = [r for r in rows if r["key"] == key or r["key"].startswith(key + ".")]
        if not rows:
            raise SystemExit(f"no parameter {key!r} (python -m ballasim params lists them)")
    if as_json:
        print(json.dumps(rows, indent=1))
        return 0
    for r in rows:
        default = json.dumps(r["value"]) if r["default"] == "literal" else r["default"]
        unit = f" {r['unit']}" if r["unit"] else ""
        print(f"{r['key']}: {r['type']}{unit}, default {default}, stage {r['stage']}, {r['scope']} "
              f"({r['class']}{', ' + r['status'] if r['status'] else ''})")
        print(f"    {r['help']}")
    return 0


def _probe() -> dict:
    """What a front end needs to know of this engine before it starts a job (blender.md §3.9): the version, the build,
    the interpreter and the protocols and formats it speaks. The Blender exchange and motion-oracle formats come from
    ``_scene.blender_formats()`` once the scene side has them (BL-2a, BL-9); until then they are empty."""
    import platform

    import ballasim
    from ballasim import _core

    from .job import FORMAT, VERSION

    formats = {}
    try:
        from ballasim import _scene

        formats = dict(_scene.blender_formats()) if hasattr(_scene, "blender_formats") else {}
    except ImportError:
        pass
    return {"version": ballasim.__version__, "build": dict(_core.build_info()),
            "python": {"executable": sys.executable, "version": platform.python_version()},
            "job_protocols": {FORMAT: list(VERSION)},
            "oracle_protocols": dict(formats.get("oracle_protocols", {})),
            "exchange_formats": dict(formats.get("exchange_formats", {}))}


def _camera_settings(args) -> int:
    """``blender camera-settings``: the Blender camera settings that reproduce a calibration's pinhole (blender.md
    §4.3 inverse; the add-on's *Apply calibration*). Exit 2 when the calibration cannot be used."""
    from ballasim import _scene
    from ballasim.blender import calibration as cal

    try:
        if args.preset:
            rec = cal.preset(args.preset)
        else:
            text = Path(args.kalibr or args.opencv).read_text(encoding="utf-8")
            rec = cal.kalibr(text, args.camera) if args.kalibr else cal.opencv(text, args.camera)
        # A preset's film: its pixel pitch times the width, so that Blender's pitch rule gives the preset's back.
        sw = args.sensor_width or (rec["width"] * rec["pixel_pitch_m"] * 1e3 if rec.get("pixel_pitch_m") else 36.0)
        doc = _scene.blender_camera_settings(rec, sw)
    except (ValueError, OSError) as e:   # CalibrationError, CAM-MODEL / CAM-CALIB from the converter, a missing file
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(doc, indent=1))
    else:
        print("\n".join(f"{k}: {v}" for k, v in doc["blender"].items()))
    return 0


def _blender(args) -> int:
    """``blender check``: the statics of an exchange and their conversion report (blender.md §3.3). Exit 0 when the
    check ran (the report says what is refused), 2 when the exchange cannot be read. ``blender camera-settings``:
    :func:`_camera_settings`."""
    if args.blender_command == "camera-settings":
        return _camera_settings(args)
    if args.blender_command in ("session", "replay"):
        from ballasim.blender import session

        return session.command(args)
    from ballasim.blender.convert import check

    if args.set and args.config is None:   # the overrides apply to a run configuration's keys: none to apply to
        print("error: --set needs --config", file=sys.stderr)
        return 2
    try:
        doc = check(args.exchange, config=args.config, sets=args.set)
    except (ValueError, OSError) as e:   # ExchangeError, BRG-PROTO from the converter, a missing file: unreadable
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(doc, indent=1))
        return 0
    st, summary = doc["statics"], doc["report"]["summary"]
    print(f"{args.exchange}: exchange {'.'.join(map(str, doc['exchange']['version']))}, digest "
          f"{doc['exchange']['digest'][:16]}; {len(st['bodies'])} bodies, {st['triangles']} triangles; cameras "
          f"{', '.join(c['name'] for c in st['cameras']) or '-'}; {len(st['lights'])} lights; "
          + ", ".join(f"{k} {v}" for k, v in summary.items()))
    for it in doc["report"]["items"]:
        print(f"  {it['severity']:7s} {it['code']} {it['object']}: {it['message']}")
    if "config" in doc:   # blender.md §3.11: every unresolved key at once, or the configuration's other error
        section = doc["config"]
        print(f"{args.config}: " + (f"error: {section['error']}" if section["error"] else
                                    f"{len(section['unresolved'])} unresolved required keys"))
        for m in section["unresolved"]:
            print(f"  {m['key']}: required {m['hint']} ({m['help']})")
    return 0


#: The sensor family whose sensor.rgb_weights prior `scan report` uses without --weights (the shipped profiles' family;
#: SceneSettings' own default).
SCAN_REPORT_FAMILY = "prophesee_gen3"
#: The import limit of `scan report`'s second import, which reads the 8-bit source of a texture halved to binary16.
SCAN_REPORT_UNLIMITED = 1 << 30


def _scan_target(args) -> tuple[Path, dict, str | None, str | None]:
    """The asset of ``scan report``: (mesh path, load_asset options, licence, scale basis). ``SCENE.yaml#ASSET`` takes
    all four from the scene's asset entry (paths and ${VARIABLES} as authoring resolves them), a PATH from the
    flags; --max-texture-size overrides either."""
    from ruamel.yaml import YAML, YAMLError

    from ballasim.scene.authoring import ASSET_LOAD_OPTIONS, SCALE_BASES, SceneSpecError, _Builder

    target = Path(args.asset)
    scene, _, name = args.asset.rpartition("#")
    if not target.exists() and scene and name:
        given = [f for f, v in (("--format", args.format), ("--scale", args.scale), ("--shading", args.shading),
                                ("--license", args.license), ("--scale-basis", args.scale_basis)) if v is not None]
        if given:
            raise ValueError(f"{', '.join(given)}: SCENE.yaml#ASSET takes them from the asset entry")
        # A scene file is untrusted input: every shape the lookups below cannot read is a SceneSpecError (exit 2).
        try:
            spec = YAML(typ="safe").load(Path(scene).read_text())
        except YAMLError as e:
            raise SceneSpecError(f"{scene}: not a readable YAML file: {e}") from None
        spec = {} if spec is None else spec
        if not isinstance(spec, dict):
            raise SceneSpecError(f"{scene}: not a scene YAML mapping")
        entries = spec.get("assets") or {}
        for key, v in (("assets", entries), ("variables", spec.get("variables") or {})):
            if not isinstance(v, dict):
                raise SceneSpecError(f"{scene}: {key} is not a mapping")
        entry = entries.get(name)
        if not isinstance(entry, dict) or "path" not in entry:
            raise SceneSpecError(f"{scene}: no asset entry {name!r} with a path "
                                 f"(assets: {', '.join(map(str, entries)) or 'none'})")
        # The value types load_asset takes (null where its default is None).
        text, maybe_text = ((str,), "a string"), ((str, type(None)), "a string or null")
        number = ((int, float, type(None)), "a number or null")
        types = {"path": text, "format": maybe_text, "node": maybe_text, "up_axis": text, "uv_origin": text,
                 "shading": text, "license": maybe_text, "scale_basis": maybe_text, "scale": number,
                 "max_texture_size": ((int,), "an integer"), "tile_texels": ((int, str), "an integer or 'auto'"),
                 "workers": ((int,), "an integer")}
        for k, (t, what) in types.items():
            if k in entry and (not isinstance(entry[k], t) or isinstance(entry[k], bool)):
                raise SceneSpecError(f"{scene}: assets.{name}.{k} must be {what}, got {entry[k]!r}")
        if "pbr" in entry and not isinstance(entry["pbr"], bool):
            raise SceneSpecError(f"{scene}: assets.{name}.pbr must be a boolean, got {entry['pbr']!r}")
        path = _Builder(spec, Path(scene).parent).path(entry["path"])
        opts = {k: entry[k] for k in ASSET_LOAD_OPTIONS if k in entry}
        lic, basis = entry.get("license"), entry.get("scale_basis")
    else:
        path = target
        opts = {k: v for k, v in (("format", args.format), ("scale", args.scale), ("shading", args.shading))
                if v is not None}
        lic, basis = args.license, args.scale_basis
    if args.max_texture_size is not None:
        opts["max_texture_size"] = args.max_texture_size
    if basis is not None and basis not in SCALE_BASES:
        raise ValueError(f"scale basis must be one of {SCALE_BASES}, got {basis!r}")
    return path, opts, lic, basis


def scan_report(args) -> dict:
    """``scan report`` (docs/design/phase3/scans.md §3.6, §4.6; S2c): the ``ballasim.scan_report/1`` document of an
    asset as load_asset imports it, its warnings included. Materials come from C++ ``_render.texture_stats`` (the texel
    value sum_c w_c b_c x_c the renderer reads), the mesh from ``_scene.mesh_stats``; ``license_canonical`` from the
    licence table (``ballasim.scene.licenses``, S2b) once it exists, else null."""
    import math

    import numpy as np

    from ballasim import _core, _render, _scene
    from ballasim.scene.assets import load_asset

    try:
        from ballasim.scene.licenses import canonical
    except ModuleNotFoundError as e:
        if e.name != "ballasim.scene.licenses":
            raise
        canonical = None
    if not (math.isfinite(args.theta) and args.theta > 0):
        raise ValueError(f"--theta must be finite and > 0, got {args.theta}")
    if args.weights is not None:
        weights = [float(w) for w in args.weights]
    else:
        family = next(f for f in _core.config.families() if f["name"] == SCAN_REPORT_FAMILY)
        weights = [float(w) for w in next(p["value"] for p in family["priors"] if p["key"] == "sensor.rgb_weights")]
    if not all(math.isfinite(w) and w >= 0 for w in weights):      # the schema's hard bound (texture_stats too)
        raise ValueError(f"--weights must be finite and >= 0, got {weights}")
    path, opts, lic, basis = _scan_target(args)
    asset = load_asset(path, **opts)
    mesh = _scene.mesh_stats(np.ascontiguousarray(asset.mesh.positions, np.float32),
                             np.ascontiguousarray(asset.mesh.indices, np.uint32))
    halved = {d["material"]: d["halvings"] for d in asset.info.get("textures_downsampled", [])}

    def material_entry(j: int, m) -> dict:
        st = _render.texture_stats(m, tuple(weights), args.theta)
        st["mono_p1_p50_p99"] = [None if math.isnan(v) else v for v in st["mono_p1_p50_p99"]]
        return {"index": j, "shading": m.shading, **st, **({"halvings": halved[j]} if j in halved else {})}

    warnings = list(asset.info.get("warnings", []))
    materials = [material_entry(j, m) for j, m in enumerate(asset.materials)]
    replica = {k: v for k, v in asset.info.items() if k not in ("source", "cache_key", "inputs", "warnings")} \
        if asset.info.get("format") == "replica" else None
    baked = any(m.shading == "baked" for m in asset.materials)
    cache_key = asset.info.get("cache_key")
    # A baked 8-bit texture above the import limit is halved to binary16 (load_asset), which has no code steps. Its
    # clipped and step shares are the 8-bit source's (scans.md §4.6, §6): one more import of the asset, without the
    # limit and outside the cache, after the first is released.
    requantized = [e["index"] for e in materials if e["source"] == "f16" and "halvings" in e]
    del asset
    if requantized:
        source = load_asset(path, **{**opts, "max_texture_size": SCAN_REPORT_UNLIMITED}, use_cache=False)
        if len(source.materials) != len(materials):
            raise RuntimeError(f"{path}: {len(source.materials)} materials without the texture limit, "
                               f"{len(materials)} with it")
        for j in requantized:
            st = _render.texture_stats(source.materials[j], tuple(weights), args.theta)
            if st["source"] == "srgb8":
                materials[j].update({"srgb8_size": st["size"], **{k: st[k] for k in (
                    "clipped_share", "step_gt_theta_share", "step_gt_theta4_share")}})
        del source
    for e in materials:
        if (e["step_gt_theta4_share"] or 0.0) > 0.05:
            warnings.append(f"material {e['index']}: {e['step_gt_theta4_share']:.1%} of its texels step by more than "
                            f"theta/4 = {args.theta / 4:g} per 8-bit code; an HDR source avoids the spurious contrast")
    lic_canonical = canonical(lic) if canonical is not None and lic is not None else None
    if lic is None:
        warnings.append("no licence given (--license, or the scene's asset entry)")
    elif canonical is not None and lic_canonical is None and baked:
        warnings.append(f"licence {lic!r} has no canonical id (scans.md §3.7)")
    if basis is None:
        warnings.append("no scale basis given" + (": an asset with baked materials needs one" if baked else ""))
    elif basis == "unverified":
        warnings.append("scale basis unverified: speeds and depths follow the file's units")
    return {"format": "ballasim.scan_report/1", "asset": args.asset, "path": str(path),
            "cache_key": cache_key, "license": lic, "license_canonical": lic_canonical,
            "scale_basis": basis, "weights": weights, "theta": args.theta, "mesh": mesh, "materials": materials,
            "warnings": warnings, **({"replica": replica} if replica is not None else {})}


def _scan(args) -> int:
    """``scan report``: :func:`scan_report`'s JSON on stdout (or in --out), its warnings on stderr. Exit 2 when the
    asset or the scene entry cannot be read or imported."""
    try:
        doc = scan_report(args)
    except (ValueError, OSError) as e:   # AssetError, SceneSpecError, a bad flag, a missing file
        if args.verbose:
            raise
        print(f"error: {e}", file=sys.stderr)
        return 2
    text = json.dumps(doc, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(args.out)
    else:
        print(text)
    for w in doc["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def _profiles() -> list[dict]:
    """The profile files (``<group>/<name>@v<N>``) with their mount (root or camera) and description."""
    from ballasim.config import PROFILES_DIR, read_profile

    rows = []
    for path in sorted(PROFILES_DIR.rglob("*@v*.yaml")):
        doc = read_profile(path.relative_to(PROFILES_DIR).with_suffix("").as_posix())
        rows.append({"id": doc["id"], "mount": doc.get("mount", "root"),
                     "description": " ".join(str(doc.get("description", "")).split())})
    return rows


def main(argv: list[str] | None = None) -> int:
    """The command line. A configuration that does not resolve (ballasim.config.ConfigError: the message names the key
    and the layer), a refused IMU simulation or export (ballasim.export.ExportError), a missing file and an occupied
    run directory print their message alone and exit with 2 (with -v: the traceback)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["job"]:     # the runner writes status.json before anything heavy loads (T9: within 1 s)
        p = argparse.ArgumentParser(prog="python -m ballasim job")
        _job_arguments(p)
        args = p.parse_args(argv[1:])
        _logging(verbose=False)
        from .job import main as job_main

        return job_main(args.job_dir, allow_exec=args.allow_exec)
    from ballasim.config import ConfigError
    from ballasim.export.errors import ExportError

    args = _parser().parse_args(argv)
    _logging(args.verbose)
    try:
        return _command(args)
    except (ConfigError, ExportError, FileNotFoundError, FileExistsError) as e:
        if args.verbose:
            raise
        print(f"error: {e}", file=sys.stderr)
        return 2


def _logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")


def _job_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("job_dir", help="job directory (holds job.json)")
    p.add_argument("--allow-exec", action="store_true",
                   help="run exec steps whose program is not in $XDG_CONFIG_HOME/ballasim/exec_allow.json")


def _parser() -> argparse.ArgumentParser:
    from .dense import CHANNELS, DEFAULT_CHANNELS

    parser = argparse.ArgumentParser(prog="python -m ballasim", description="BallaSim asynchronous event simulator")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def layer_args(p):
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="a value of the CLI layer (full key, YAML scalar or flow list; repeatable)")
        p.add_argument("--profile", action="append", default=[], metavar="ID",
                       help="a root profile appended to the configuration's list (repeatable)")

    p_run = sub.add_parser("run", help="run a configuration file (format 2, docs/dev/ENGINE.md)")
    p_run.add_argument("config", help="run configuration YAML")
    p_run.add_argument("--overwrite", action="store_true", help="replace an existing run directory")
    layer_args(p_run)

    p_ev = sub.add_parser("evimo2", help="build (or reuse) an EVIMO2 sequence's bundle and simulate it")
    p_ev.add_argument("sequence", help="EVIMO2 sequence directory, e.g. .../left_camera/imo/eval/scene13_dyn_test_00_000000")
    p_ev.add_argument("--camera", default=None, help="camera folder (left_camera, right_camera, samsung_mono); "
                                                     "default: the one the sequence lies in")
    p_ev.add_argument("--t0", type=seconds_to_ns, default=None, help="start (s, bundle clock)")
    p_ev.add_argument("--t1", type=seconds_to_ns, default=None, help="end (s, bundle clock)")
    p_ev.add_argument("--out", required=True, help="run directory")
    p_ev.add_argument("--overwrite", action="store_true", help="replace an existing run directory")
    p_ev.add_argument("--no-profiles", action="store_true",
                      help="no default profiles (profiles/evimo2/index.yaml): set the family, the lighting and "
                           "illuminance_scale with --set or --profile")
    p_ev.add_argument("--profile-set", metavar="SET",
                      help="the default profiles' set in profiles/evimo2/index.yaml: v0 (default; the P1-M5 values) or "
                           "v1 (the round-12 refit, docs/dev/EVIMO2_V1.md; normal-light sequences only; needs "
                           "$EVIMO2_COLORFIX_ROOT)")
    p_ev.add_argument("--camera-profile", action="append", default=[], metavar="CAMERA=ID",
                      help="a camera-mounted profile for the sequence's camera, appended to its profiles after the "
                           "defaults, e.g. left_camera=evimo2/prophesee_gen3_imu_left@v1 (repeatable)")
    layer_args(p_ev)

    p_dn = sub.add_parser("dense", help="write dense frames (depth, instance, irradiance, ...) of a run")
    p_dn.add_argument("run_dir", help="run directory (holds run.json)")
    p_dn.add_argument("--dt-ms", type=Decimal, default=Decimal(20), help="frame spacing in ms (default: 20)")
    p_dn.add_argument("--t0", type=seconds_to_ns, default=None, help="first frame (s, bundle clock; default: run start)")
    p_dn.add_argument("--t1", type=seconds_to_ns, default=None, help="end (s, bundle clock; default: run end)")
    p_dn.add_argument("--channels", default=",".join(DEFAULT_CHANNELS),
                      help=f"comma-separated subset of {','.join(CHANNELS)} (default: %(default)s)")
    p_dn.add_argument("--camera", action="append", default=None, help="camera name (repeatable; default: the run's)")
    p_dn.add_argument("--out", default=None, metavar="DIR", help="output directory (default: <run>/dense)")

    p_pa = sub.add_parser("params", help="print parameter descriptors (docs/dev/PARAMS_REFERENCE.md)")
    p_pa.add_argument("key", nargs="?", default=None, help="a key or a key prefix (e.g. sensor, accuracy.d_px)")
    p_pa.add_argument("--json", action="store_true", help="the descriptors as JSON")

    p_re = sub.add_parser("resolve", help="resolve a configuration without running it (loads the scene bundle)")
    p_re.add_argument("config", help="run configuration YAML")
    p_re.add_argument("--json", action="store_true", help="inputs, resolved values, warnings and digests as JSON")
    p_re.add_argument("--all", action="store_true", help="print every value, not only those the layers set")
    layer_args(p_re)

    p_sc = sub.add_parser("scene", help="scene sources")
    sc = p_sc.add_subparsers(dest="scene_command", required=True)
    p_sb = sc.add_parser("build", help="build and cache the configuration's scene bundle only (no camera resolution); "
                                       "prints {kind, key, source} as JSON")
    p_sb.add_argument("config", help="run configuration YAML")
    layer_args(p_sb)

    from ballasim.export import cli as export_cli

    export_cli.add_parsers(sub)  # imu, export

    p_ui = sub.add_parser("ui", help="the browser UI: a loopback server (needs the ui extra: Tornado)")
    p_ui.add_argument("--port", type=int, default=0, help="port (default 0: an ephemeral one)")
    p_ui.add_argument("--bind", help="127.0.0.1 (the default) or ::1; nothing else")
    p_ui.add_argument("--root", action="append", default=[], metavar="NAME=PATH",
                      help="a browsable root (repeatable; default out=$PWD)")
    p_ui.add_argument("--evimo2-root", metavar="PATH", help="the EVIMO2 root, read-only (default: the machine's)")
    p_ui.add_argument("--jobs-root", metavar="PATH",
                      help="UI job directories (default $XDG_STATE_HOME/ballasim/ui/jobs)")
    p_ui.add_argument("--repo", metavar="PATH", help="a source checkout, for the bench features (default: the "
                                                     "checkout the package runs from, if any)")
    p_ui.add_argument("--open", nargs="?", const="", metavar="PATH",
                      help="open the browser at PATH (a job or run directory), reusing a running server or starting "
                           "one in the background")
    p_ui.add_argument("--no-browser", action="store_true", help="do not start a browser")
    p_ui.add_argument("--browser", metavar="CMD", help="browser command (default: $BALLASIM_BROWSER, else Chrome or "
                                                       "Chromium from PATH)")
    p_ui.add_argument("--gpu-lock", metavar="PATH", help="the GPU lock file shared with the add-on's jobs")
    p_ui.add_argument("--cache-mb", type=int, default=512, help="server RAM for event blocks (default 512)")
    p_ui.add_argument("--resim-device", choices=("auto", "gpu", "cpu"), default="auto",
                      help="re-simulation device (default auto)")
    p_ui.add_argument("--dev", action="store_true", help="development: heuristic rows editable, no static caching")
    p_ui.add_argument("--status", action="store_true", help="print the running server (no secret); exit 1 if none")
    p_ui.add_argument("--stop", action="store_true", help="stop the running server")
    p_ui.add_argument("--detached", action="store_true", help=argparse.SUPPRESS)   # a server started by --open

    p_job = sub.add_parser("job", help="run a job directory (ballasim.job/1: job.json's steps, status.json; "
                                        "exit 3 when cancelled)")
    _job_arguments(p_job)

    p_pr = sub.add_parser("probe", help="the engine's version, build, interpreter and protocols")
    p_pr.add_argument("--json", action="store_true", help="as JSON")

    p_pf = sub.add_parser("profiles", help="list the profile files")
    p_pf.add_argument("--json", action="store_true", help="as JSON")

    p_bl = sub.add_parser("blender", help="Blender exchanges (docs/design/phase3/blender.md)")
    bl = p_bl.add_subparsers(dest="blender_command", required=True)
    p_bc = bl.add_parser("check", help="convert an exchange's statics and print the conversion report")
    p_bc.add_argument("--exchange", required=True, help="exchange directory (scene.bsx/, holds exchange.json)")
    p_bc.add_argument("--config", help="run configuration: list its unresolved required keys on this exchange")
    p_bc.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                      help="override a key of --config (repeatable; needs --config)")
    p_bc.add_argument("--json", action="store_true", help="as JSON")
    p_bse = bl.add_parser("session", help="the motion session on this process's pipes (blender.md §3.5): the add-on "
                                          "starts it as `python -I -u -m ballasim blender session ...`")
    p_bse.add_argument("--exchange", required=True, help="exchange directory (scene.bsx/)")
    p_bse.add_argument("--config", default=None, help="the job's run.yaml (scene.blender.*, run.select)")
    p_bse.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="scene.blender.<key>=V, run.select=[...] or execution.scene_workers=N (repeatable)")
    p_brp = bl.add_parser("replay", help="refit a recorded session from motion/oracle_log.npz, without Blender")
    p_brp.add_argument("--exchange", required=True, help="exchange directory (scene.bsx/)")
    p_brp.add_argument("--config", default=None, help="a run.yaml whose options replace the recorded session's")
    p_brp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="as for session (any item replaces the recorded options)")
    p_bs = bl.add_parser("camera-settings", help="the Blender camera settings that reproduce a calibration's pinhole")
    src = p_bs.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", help="an EVIMO2 camera preset (evimo2_left, evimo2_right, evimo2_samsung)")
    src.add_argument("--kalibr", metavar="FILE", help="a Kalibr camchain YAML")
    src.add_argument("--opencv", metavar="FILE", help="an OpenCV (FileStorage or ROS camera_info) YAML")
    p_bs.add_argument("--camera", help="the camera in the file (Kalibr: cam1 or 1)")
    p_bs.add_argument("--sensor-width", type=float, default=None,
                      help="the film width in mm (default: width x the preset's pixel pitch, else 36)")
    p_bs.add_argument("--json", action="store_true", help="as JSON")

    p_fr = sub.add_parser("frames", help="frames on the event clock (docs/design/phase3/frames.md)")
    fr = p_fr.add_subparsers(dest="frames_command", required=True)
    p_ref = fr.add_parser("reference", help="BallaSim's own frames of a run: the exposure reference (frames.md §4.4)")
    p_ref.add_argument("run_dir", help="run directory (holds run.json)")
    p_ref.add_argument("--camera", action="append", default=[], metavar="C",
                       help="a camera of run.select or run.frames_select (repeatable; default: those whose "
                            "frames.mode is reference)")
    p_ref.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="cameras.<c>.frames.K=V, cameras.default.frames.K=V, execution.frames_device=V, "
                            "execution.frames_texture_mb=V or run.frames_select=[...] (repeatable)")
    p_ref.add_argument("--threads", type=int, default=0, help="APS worker threads (default: the hardware's)")
    p_aps = fr.add_parser("aps", help="re-run the APS model on a frames file's stored mono (frames.md §4.10)")
    p_aps.add_argument("run_dir", help="run directory (holds run.json and frames/<camera>.h5)")
    p_aps.add_argument("--camera", required=True, help="camera name")
    p_aps.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="cameras.<camera>.frames.aps.K=V or cameras.default.frames.aps.K=V (repeatable)")
    p_aps.add_argument("--threads", type=int, default=0, help="worker threads (default: the hardware's)")

    p_sc = sub.add_parser("scan", help="scanned rooms in baked-radiance mode (docs/dev/SCANS.md)")
    sc = p_sc.add_subparsers(dest="scan_command", required=True)
    p_sr = sc.add_parser("report", help="an asset's import diagnostics: texture statistics against theta, mesh "
                                         "statistics, licence and scale basis (ballasim.scan_report/1 JSON)")
    p_sr.add_argument("asset", metavar="PATH|SCENE.yaml#ASSET",
                      help="a mesh file, or an asset entry of a scene YAML (its options, licence and scale basis)")
    p_sr.add_argument("--format", choices=("gltf", "obj", "ply", "replica"),
                      help="PATH: the file format (default: its suffix; a Replica scene directory needs replica)")
    p_sr.add_argument("--scale", type=float, help="PATH: file units -> metres (required for OBJ and PLY)")
    p_sr.add_argument("--shading", choices=("auto", "lit", "baked"), help="PATH: the import's shading (default auto)")
    p_sr.add_argument("--max-texture-size", type=int, metavar="N",
                      help="the import's texture limit (default 4096, or the entry's): a baked 8-bit texture above it "
                           "is halved to binary16; its clipped and step shares are the 8-bit source's (srgb8_size)")
    p_sr.add_argument("--license", help="PATH: the asset's licence")
    p_sr.add_argument("--scale-basis", help="PATH: metric-source, measured or unverified")
    p_sr.add_argument("--theta", type=float, default=0.25,
                      help="contrast threshold in log-e (default 0.25; a run's is min(theta_on, theta_off))")
    p_sr.add_argument("--weights", type=float, nargs=3, metavar=("R", "G", "B"),
                      help=f"sensor RGB weights (default: the {SCAN_REPORT_FAMILY} family's sensor.rgb_weights)")
    p_sr.add_argument("--out", metavar="FILE", help="write the JSON there instead of stdout")
    return parser


def _command(args) -> int:
    if args.command in ("imu", "export"):
        from ballasim.export import cli as export_cli

        return export_cli.command(args)
    if args.command == "ui":
        try:
            from ballasim.ui.server import main as ui_main
        except ModuleNotFoundError as e:
            if e.name != "tornado":
                raise
            print("error: python -m ballasim ui needs Tornado (pip install 'ballasim[ui]')", file=sys.stderr)
            return 2
        return ui_main(args)
    if args.command == "job":
        from .job import main as job_main

        return job_main(args.job_dir, allow_exec=args.allow_exec)
    if args.command == "blender":
        return _blender(args)
    if args.command == "scan":
        return _scan(args)
    if args.command in ("probe", "profiles"):
        doc = _probe() if args.command == "probe" else _profiles()
        if args.json:
            print(json.dumps(doc, indent=1))
        elif args.command == "probe":
            print("\n".join(f"{k}: {json.dumps(v)}" for k, v in doc.items()))
        else:
            print("\n".join(f"{r['id']} ({r['mount']}): {r['description']}" for r in doc))
        return 0
    if args.command == "dense":
        from .dense import export_dense

        paths = export_dense(args.run_dir, int(args.dt_ms * 1_000_000), t0_ns=args.t0, t1_ns=args.t1,
                             channels=[c.strip() for c in args.channels.split(",") if c.strip()], cameras=args.camera,
                             out_dir=args.out)
        print("\n".join(map(str, paths)))
        return 0
    if args.command == "params":
        return _params(args.key, args.json)
    if args.command == "scene":
        print(json.dumps(scene_build(args.config, args.set, args.profile)))
        return 0
    if args.command == "frames":
        from ballasim.frames import aps, reference

        try:
            return (reference if args.frames_command == "reference" else aps).main(args)
        except (ValueError, OSError) as e:  # FramesError (a frames.md §6 refusal: its code leads), a run or file error
            if args.verbose:
                raise
            print(f"error: {e}", file=sys.stderr)
            return 2

    from ballasim import config

    from .engine import run

    if args.command == "resolve":
        from .scene import load_source, scene_facts, render_bundle

        spec = config.load(args.config, sets=args.set, profiles=args.profile)
        values = spec.scene.values()["run"]
        source = load_source(values)
        resolved = config.resolve(spec, render_bundle(source, values), supplied=False,
                                  **scene_facts(values, source))
        if args.json:
            print(json.dumps({"inputs": spec.record, "resolved": resolved.record(), "warnings": resolved.warnings,
                              "stage_digests": resolved.digests(), "span_ns": resolved.span}, indent=1, default=str))
        else:
            print("\n".join([f"{args.config}: cameras {resolved.cameras}, windows {resolved.span}"] +
                            resolved_lines(resolved, every=args.all)))
        return 0
    sets = list(args.set) + (["output.overwrite=true"] if args.overwrite else [])
    if args.command == "run":
        spec = config.load(args.config, sets=sets, profiles=args.profile)
        print(_report(run(spec)))
        return 0
    spec = config.load(_evimo2_doc(args), sets=sets, profiles=args.profile)
    print(_report(run(spec, report=lambda r: print("\n".join(["resolved (non-default values):"] +
                                                             resolved_lines(r)), flush=True))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
