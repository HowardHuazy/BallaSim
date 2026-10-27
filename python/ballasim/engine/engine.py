"""The run loop (docs/dev/ENGINE.md): per camera and window W = 2^23 ns, one ``step`` of the C++/CUDA engine
(``ballasim._engine``, docs/dev/ENGINE_CPP.md), which plans, samples, renders, refines, senses, attaches the
ground truth and sorts the output; Python writes the events (writer.py) and the logs (runlog.py) until I10.

Clocks. Simulation times (windows, samples, t_cross, GT) are on the bundle's true clock; the sensor reports
t_reported on the camera's clock (t_true + clock_offset_ns + its drift, plus its readout). Windows are aligned to
the epoch t = 0, so a run covers the whole windows inside the requested span.

Sensor start. ``start`` renders every pixel at the first window start t_s, initializes the sensor at
t_s - warmup_ns with that frame and advances it to t_s with the frame held (DESIGN §8.4): the scene is taken as
still before the run, and the warm-up events are discarded. With ``accuracy.warmup: motion`` (D12) the warm-up's whole
windows that the bundle covers are planned, rendered and sensed like the run's (events discarded) instead, and the
sensor's comparators start from ``accuracy.initial_state`` (the phase law, or the stationary law of SENSOR.md).

Output order. Events are sorted by (t_reported, x, y). Every event of a later window is reported at or after
t_{w+1} on the camera's clock (the readout delay is at least one counter quantum; the clock is non-decreasing), so
``step`` returns the events of window w reported before that and holds the rest for the next window; ``flush``
returns them at the end.

Devices. The engine runs on the device (OptiX, the CUDA sensor in fp32, CUB) or on the host (Embree, the CPU
sensor at ``accuracy.sensor_precision``, serial primitives): the same code in two instantiations (ENGINE_CPP §1.7),
chosen by ``run.device``.

Configuration. ``run`` takes a :class:`ballasim.config.RunSpec` (or a path): the C++ resolver gives every value
(docs/dev/P0_PARAMS.md §2), ``_engine.Engine(bundle, resolved.values())`` applies them, and run.json records the inputs,
the resolved values with their layers, the warnings and the stage digests (§2.6).

Progress and cancel (BL-0b; docs/design/phase3/blender.md §3.8, contract C9). ``run(..., on_window=, cancel=)``: a
front end (the job runner, a UI worker) sees each window as it ends and stops a run with a ``_engine.CancelToken`` set
from another thread. Python checks the token between the run's phases and before each window, the C++ engine at each
pixel batch and refinement round and before sensing (src/engine/cancel.h); a set token raises ``_engine.Cancelled``,
run.json then reads ``cancelled`` and the cancelled camera's partial event file is removed.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ballasim import _core
from ballasim.scene.bundle import CONSTANT_LIGHT_KINDS, SceneBundle

from .runlog import (
    RunDirInUse,
    RunLog,
    git_commit,
    lock_run_dir,
    run_dir_locked,
    run_lock_path,
    versions,
    write_json_atomic,
)
from .scene import bundle_digest, light_group_record, load_source, render_bundle, scene_facts
from .writer import EventWriter

log = logging.getLogger(__name__)

#: Window length W = 2^23 ns (DESIGN §6.10, time::kWindowLog2); windows are aligned to the simulation epoch t = 0.
WINDOW_NS = _core.WINDOW_NS


@dataclass
class CameraSummary:
    name: str
    index: int
    events: int = 0
    windows: int = 0
    pixel_samples: int = 0
    rays: int = 0
    render_calls: int = 0
    budget_clamps: int = 0        # windows whose levels were clamped to fit accuracy.max_samples_per_window
    plan_s: float = 0.0
    render_s: float = 0.0
    sense_s: float = 0.0
    gt_s: float = 0.0
    write_s: float = 0.0
    warmup_s: float = 0.0
    probe_s: float = 0.0          # the stationary probe (accuracy.initial_state: stationary), inside warmup_s
    probe_history_s: float = 0.0  # the probe's history T_b (accuracy.stationary_burnin_ns, derived when 0)
    warmup_windows: int = 0       # windows of a motion warm-up (accuracy.warmup: motion), inside warmup_s
    illuminance_scale: float = 0.0  # the sensor's photocurrent scale, fA per renderer unit (reference: solved; D1)
    frame_mean: list = field(default_factory=list)  # veiling glare: the frame means per group at t_s (perf.jsonl: ends)
    wall_s: float = 0.0
    held_rebuilds: int = 0        # rebuilds of the renderer's held-set structure (ENGINE_CPP I3b)
    held_build_s: float = 0.0
    # Scanned rooms with proxy lights (scans.md §4.2): per proxy shadow light (bundle order) the share of its start-frame
    # rays from baked receivers that the scan blocks, and the static bodies split from the scan geometry ([] and 0
    # otherwise).
    proxy_blocked: list = field(default_factory=list)
    split_geometries: int = 0
    counters: dict = field(default_factory=dict)  # perf.jsonl's P2-0b counters and sensor pieces, summed (ENGINE.md)
    sensor_device: str = ""
    path: str = ""


@dataclass
class RunSummary:
    run_dir: Path
    backend: str
    t_start_ns: int
    t_end_ns: int
    cameras: list[CameraSummary] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def simulated_s(self) -> float:
        return (self.t_end_ns - self.t_start_ns) * 1e-9


#: The run locks taken by the run() of this context (:func:`_holding_run_lock`); None outside a run.
_RUN_LOCKS: contextvars.ContextVar[list | None] = contextvars.ContextVar("ballasim_run_locks", default=None)


def _check_run_dir(run_dir: Path, overwrite: bool) -> Path:
    """The run directory, refused when it holds a run (unless output.overwrite) or anything else, and while another
    run holds its lock (RunDirInUse). Inside :func:`run` this takes the lock (runlog.lock_run_dir), which the run
    holds to its end; elsewhere (the job runner's ``validate`` step) it only probes it."""
    held = _RUN_LOCKS.get()
    if held is not None:
        lock = lock_run_dir(run_dir)
        if lock is not None:
            held.append(lock)
    elif run_dir_locked(run_dir):
        raise RunDirInUse(f"{run_dir}: run directory in use (another run holds {run_lock_path(run_dir)})")
    if run_dir.exists() and any(run_dir.iterdir()):
        if not (run_dir / "run.json").exists():
            raise FileExistsError(f"{run_dir} exists, is not empty and holds no run.json; refusing to write into it")
        if not overwrite:
            raise FileExistsError(f"{run_dir} holds a run; set output.overwrite to replace it")
    return run_dir


def _create_run_dir(run_dir: Path, doc: dict) -> None:
    """Replaces a previous run (checked by _check_run_dir) and writes run.json in the same step, so a
    run directory always holds a run.json."""
    if (run_dir / "run.json").exists():
        shutil.rmtree(run_dir)
    (run_dir / "events").mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_dir / "run.json", doc)


def make_engine(bundle: SceneBundle, resolved):
    """The engine of a resolved run: its renderer and the sensors of the selected cameras, built."""
    from ballasim import _engine

    return _engine.Engine(bundle, resolved.values())


def _check_cancel(cancel) -> None:
    """The Python side's check point: ``_engine.Cancelled`` when the token (a ``_engine.CancelToken``, or None) is
    set."""
    if cancel is not None and cancel.is_set():
        from ballasim import _engine

        raise _engine.Cancelled("run: cancelled")


def reflection_policy(bundle: SceneBundle, resolved) -> tuple[list[dict], list[str]]:
    """DESIGN §7.7's reflection policy (P2-F F4; render/reflection_policy.h): run.json's ``reflection_bounds``, one
    entry per used material with a specular lobe (C1 feature "pbr"), and the warnings of the refused materials that
    ``scene.allow_missing_reflections`` accepts. Without that key a refused material is a ConfigError. A bundle without
    lobes gives nothing (and run.json no ``reflection_bounds``)."""
    from ballasim import _engine, _render
    from ballasim.config import ConfigError

    if not any(m.has_lobe for b in bundle.bodies for m in b.materials):   # spares the conversion of a Lambert scene
        return [], []
    # Each light group's waveform range over all time: m_g(t) is mean-normalised, so a pulse peaks above 1 and a
    # Fourier sum or a pulse's floor drops below it, and the bound takes each light at its group's peak (numerator)
    # or trough (the ambient of the denominator).
    groups = [_engine.waveform_range(g.waveform) for g in bundle.light_groups]
    # The simulated cameras' sensor.nir_fraction (P2-F F11a): the albedo a camera's light group sees is
    # (1 - f) rho_vis + f rho_NIR, so the bound's rho_max and rho_min take every f present.
    nir = [list(resolved.get("sensor.nir_fraction", name)) for name in resolved.cameras]
    bounds = []
    for r in _render.reflection_policy(bundle, resolved.values(), groups, nir):
        body = bundle.bodies[r["body"]]
        finite = {k: (v if math.isfinite(v) else None) for k, v in r.items() if k in ("l_max", "l_diff_min", "bound")}
        bounds.append({"body": body.name, "body_index": r["body"], "material": r["material"],
                       **{k: r[k] for k in ("roughness", "metallic", "specular", "refused", "f_max")}, **finite})
    refused = [b for b in bounds if b["refused"]]

    def name(b: dict) -> str:
        return (f"body {b['body']!r} material {b['material']} (roughness {b['roughness']:.3g}, "
                f"metallic {b['metallic']:.3g})")

    if refused and not resolved.get("scene.allow_missing_reflections"):
        raise ConfigError(f"scene.allow_missing_reflections = false: v1 traces no reflections, and DESIGN §7.7 refuses "
                          f"specular materials with roughness < 0.2, or metallic with roughness < 0.5: "
                          f"{'; '.join(map(name, refused))}. Set scene.allow_missing_reflections: true to accept their "
                          "missing reflections (run.json reflection_bounds records each bound)")
    notes = [f"scene.allow_missing_reflections: {name(b)} misses reflections of up to "
             f"{'an unbounded' if b['bound'] is None else format(b['bound'], '.3g')} log contrast "
             "(run.json reflection_bounds)" for b in refused]
    return bounds, notes


def assumed_nir_albedo(bundle: SceneBundle, resolved) -> list[dict]:
    """run.json's ``assumed_nir_albedo`` (P2-F F11a; DESIGN §7.2): the lit materials in use whose near-infrared albedo is
    the assumed one (``nir_form`` "assumed": the red channel), listed when some simulated camera's
    ``sensor.nir_fraction`` is above 0, that is when the assumption reaches the images. Empty otherwise."""
    if not any(f > 0 for name in resolved.cameras for f in resolved.get("sensor.nir_fraction", name)):
        return []
    out = []
    for i, body in enumerate(bundle.bodies):
        for j in body.mesh.used_materials():
            m = body.materials[j] if j < len(body.materials) else None   # no materials: the default one
            if m is None or (m.nir_form == "assumed" and m.shading != "baked"):
                out.append({"body": body.name, "body_index": i, "material": int(j)})
    return out


def _append(writer: EventWriter, cols: dict) -> None:
    gt = {k: v for k, v in cols.items() if k not in ("x", "y", "p", "t_reported_ns")}
    writer.append(cols["x"], cols["y"], cols["p"], cols["t_reported_ns"], gt or None)


#: A proxy light that the scan blocks for more than this share of its start-frame rays from baked receivers sits inside
#: scanned geometry (a lampshade): mover shadows cannot reach the scan from it (scans.md §6, risks).
PROXY_BLOCKED_WARN = 0.5
#: Static bodies split from the scan geometry above this number cost a sphere test each per shadow ray (scans.md §6).
SPLIT_GEOMETRIES_WARN = 8


def start_warnings(bundle: SceneBundle, camera: str, start: dict) -> list[str]:
    """The warnings of a camera's start result for run.json and stderr (scans.md §6): a proxy light inside scanned
    geometry (``start["proxy_blocked"]``: per proxy shadow light in bundle order, the share of its rays from baked
    receivers that the scan blocks), and many static bodies split from the scan (``start["split_geometries"]``)."""
    out = []
    proxies = [i for i, lt in enumerate(bundle.lights)
               if lt.proxy and lt.kind not in CONSTANT_LIGHT_KINDS and lt.casts_shadows]
    for k, share in enumerate(start.get("proxy_blocked", ())):
        if share > PROXY_BLOCKED_WARN:
            name = f"proxy light {proxies[k]}" if k < len(proxies) else f"proxy light #{k}"
            out.append(f"camera {camera}: the scan blocks {share:.0%} of the start-frame rays from {name} to the baked "
                       "receivers: it sits inside scanned geometry (a lampshade?) and inserted objects will cast little "
                       "shadow from it onto the scan; offset it about 2 cm outside the fixture (docs/dev/SCANS.md)")
    n = start.get("split_geometries", 0)
    if n > SPLIT_GEOMETRIES_WARN:
        out.append(f"camera {camera}: {n} static bodies are traced apart from the scan geometry for proxy shadows (more "
                   f"than {SPLIT_GEOMETRIES_WARN}): each adds a sphere test to every shadow ray, so render time grows "
                   "linearly with them (docs/dev/SCANS.md)")
    return out


def run_camera(bundle: SceneBundle, cam_index: int, resolved, cam, run_dir: Path, runlog: RunLog,
               t_s: int, t_e: int, *, on_window=None, cancel=None, warn=None) -> CameraSummary:
    """Windows [t_s, t_e) of one camera of the engine (``cam``: its CameraEngine): one step per window, then the
    writer and the logs; ``on_window`` and ``cancel`` as :func:`run`'s (``cam`` reads the token already). ``warn``, if
    given, is called with each warning of the start (``start_warnings``); they go to the log too."""
    camera = bundle.cameras[cam_index]
    path = run_dir / "events" / f"{camera.name}.h5"
    s = CameraSummary(camera.name, cam_index, sensor_device=cam.device, path=str(path))
    runlog.start_camera(camera.name, resolved.cameras)      # progress.json's camera_index, cameras (ui.md §4.10)
    wall = time.perf_counter()
    gt = resolved.get("output.gt")
    start = cam.start(t_s, resolved.get("accuracy.warmup_ns"), resolved.get("accuracy.warmup") == "motion")
    s.warmup_s, s.probe_s, s.warmup_windows = start["warmup_s"], start["probe_s"], start["motion_windows"]
    s.probe_history_s = start["probe_history_s"]
    s.illuminance_scale = start["illuminance_scale"]
    s.frame_mean = start["frame_mean"]
    s.pixel_samples = int(camera.width) * int(camera.height)
    s.render_calls = 1
    s.render_s = start["render_s"]
    s.held_rebuilds, s.held_build_s = int(start["held_rebuilt"]), start["held_build_s"]
    s.proxy_blocked, s.split_geometries = list(start.get("proxy_blocked", ())), int(start.get("split_geometries", 0))
    for w in start_warnings(bundle, camera.name, start):
        log.warning("%s", w)
        if warn is not None:
            warn(w)
    # Rays: a primary and a shadow ray per shadow-casting light for every sub-sample. With the band N (P2-B B4a) the
    # band's pixel-samples and the switches up take N_band, the rest and the switches down N_base; the frame N_band;
    # the escalations' pixel-samples (B4b) N_band.
    per_sub = 1 + sum(lt.kind not in CONSTANT_LIGHT_KINDS and lt.casts_shadows for lt in bundle.lights)
    n_base, n_band = resolved.get("accuracy.n_subsamples"), resolved.get("accuracy.n_band_max")

    def rays(n: int, counters: dict) -> int:
        if not n_band:
            return n * n_base * per_sub
        band, up = counters["band_samples"] + counters["escalated_samples"], counters["n_switch_up"]
        down = counters["n_switch"] - up
        return ((n - counters["band_samples"] + down) * n_base + (band + up) * n_band) * per_sub

    def rays_by_round(sampling: dict) -> list[int]:
        # P2-C C2: the rays of each round of the window's pass (the switches and escalations aside): its pixel-samples
        # at N_base and, with the band, those at N_band (band_samples_by_round).
        band = list(sampling.get("band_samples_by_round") or [])
        band += [0] * (len(sampling["samples_by_round"]) - len(band))
        return [((k - b) * n_base + b * (n_band or n_base)) * per_sub
                for k, b in zip(sampling["samples_by_round"], band)]

    s.rays = s.pixel_samples * (n_band or n_base) * per_sub
    # The file's clock is the sensor's: the resolved sensor.clock_offset_ns (the bundle's, or its override) and drift.
    writer = EventWriter(path, camera=camera.name, width=camera.width, height=camera.height,
                         t_offset_us=cam.sensor_clock_ns(t_s) // 1000, gt=gt)
    n_windows = (t_e - t_s) // WINDOW_NS
    try:
        for w in range(n_windows):
            _check_cancel(cancel)
            t_w = t_s + w * WINDOW_NS
            cols, c = cam.step(t_w)
            plan = c["plan"]
            if "mode" not in plan:
                if c["level_cap"] is not None:
                    log.warning("camera %s, window at %d ns: the plan exceeds accuracy.max_samples_per_window = %d "
                                "(image-speed bound up to %.3g px/s); levels clamped to %d", camera.name, t_w,
                                resolved.get("accuracy.max_samples_per_window"), plan["v_hat_max"], c["level_cap"])
                    s.budget_clamps += 1
                plan = {**plan, "level_cap": c["level_cap"], "plan_s": c["plan_s"]}
                s.plan_s += c["plan_s"]
            t0 = time.perf_counter()
            _append(writer, cols)
            write_s = time.perf_counter() - t0
            n = c["pixel_samples"]
            runlog.window({"camera": camera.name, "window": w, "t_w_ns": t_w, **plan, **c["sampling"],
                           "pixel_samples": n, "rays": rays(n, c["counters"]),
                           "rays_by_round": rays_by_round(c["sampling"]), "render_calls": c["render_calls"],
                           "batches": c["batches"], "render_s": c["render_s"],
                           **{k: c[k] for k in ("render_trace_ms", "render_shade_ms", "render_host_ms", "render_probe_ms",
                                                "render_ms_by_call", "render_ms_by_round",
                                                "escalation_render_ms_by_round", "escalation_samples_by_round",
                                                "render_ms_redone", "render_ms_switches",
                                                "render_ms_escalation_nodes")},
                           "sample_s": c["plan_s"] + c["sample_s"], "sense_s": c["sense_s"],
                           "gt_s": c["gt_s"], "write_s": write_s, "events": c["events"],
                           "held_bodies": c["held_bodies"], "held_rebuilt": c["held_rebuilt"],
                           "held_build_s": c["held_build_s"], "sensor": c["sensor"], "counters": c["counters"],
                           **({"frame_mean": c["frame_mean"]} if c["frame_mean"] else {}),
                           # P2-H H4b: the rate limiter's decisions in this window (only while it is on)
                           **({"limiter": c["limiter"]} if "limiter" in c else {})})
            s.windows += 1
            s.events += c["events"]
            s.pixel_samples += n
            s.rays += rays(n, c["counters"])
            s.render_calls += c["render_calls"]
            s.render_s += c["render_s"]
            s.sense_s += c["sense_s"]
            s.gt_s += c["gt_s"]
            s.write_s += write_s
            s.held_rebuilds += int(c["held_rebuilt"])
            s.held_build_s += c["held_build_s"]
            sensor = {"sensor_pieces": c["sensor"]["pieces"], "sensor_cells": c["sensor"]["cells"]}
            for k, v in {**c["counters"], **sensor}.items():
                old = s.counters.get(k)
                s.counters[k] = v if old is None else [x + y for x, y in zip(old, v)] if isinstance(v, list) \
                    else old + v
            runlog.progress(camera=camera.name, window=w + 1, windows=n_windows, t_ns=t_w + WINDOW_NS,
                            events=s.events)
            if on_window is not None:
                on_window(camera.name, w + 1, n_windows, s.events, t_w + WINDOW_NS)
        _append(writer, cam.flush())
        limiter = cam.limiter()  # P2-H H4b: over the run, the flush's bins included (None while the limiter is off)
        if limiter is not None:
            s.counters.update({k if k == "limiter_s" else f"limiter_{k}": v for k, v in limiter.items() if k != "held"})
            s.events = limiter["kept"]  # the events written (perf.jsonl's per-window events: the sensor's, before it)
        writer.close(cam.sensor_clock_ns(t_e + cam.readout_delay_ns + cam.readout_quantum_ns))
    except BaseException:
        writer.abort()
        raise
    s.wall_s = time.perf_counter() - wall
    return s


def run(spec, *, bundle: SceneBundle | None = None, report=None, on_window=None, cancel=None) -> RunSummary:
    """Runs a run configuration (a :class:`ballasim.config.RunSpec`, or a path for ``ballasim.config.load``). A
    supplied ``bundle`` replaces the scene keys (which may then be omitted); run.json records it as supplied.
    ``report(resolved)``, if given, sees the resolved run before the engine is built (the CLI prints it).

    Everything that can fail before the first window (the scene, the resolution and its checks, the engine with its
    renderer and sensors) runs before the run directory is touched; from then on run.json records the outcome.

    BL-0b (contract C9): ``on_window(camera, window, windows, events, t_ns)``, if given, is called after each window
    (after progress.json), from the thread that runs the engine, with the camera's name, its windows done and in all,
    its events so far and the true time its windows reached. ``cancel``, a ``ballasim._engine.CancelToken`` set from
    any thread, is checked by Python before each phase (the scene, the resolution, the engine, each camera) and window,
    and by the engine at each pixel batch and refinement round and before sensing; a set token raises
    ``ballasim._engine.Cancelled``. A cancel before the run directory exists leaves none; after, run.json reads
    ``cancelled`` with the cameras that completed, and the cancelled camera's partial event file is removed. Neither
    changes a run's outputs.
    """
    from ballasim import config

    _check_cancel(cancel)
    wall = time.perf_counter()
    if not isinstance(spec, config.RunSpec):
        spec = config.load(spec)
    run_dir = _check_run_dir(Path(spec.get("output.dir")), spec.get("output.overwrite"))
    supplied = bundle is not None
    # digest(scene): an EVIMO2 source's cache key (its lighting is relit at load and enters the later stages), else
    # the source's bundle digest, before the run's background light (P0_PARAMS §1.2); run.json's bundle block keeps
    # the content digest of the bundle rendered (relit; with the background). An EVIMO2 bundle's light mode decides
    # which lighting keys matter.
    if supplied:
        from .scene import with_baked_scale

        bundle = with_baked_scale(bundle, spec.scene.values()["run"])
        digest = bundle_digest(bundle)
        facts = {"digest": digest}
    else:
        values = spec.scene.values()["run"]
        source = load_source(values)
        bundle = render_bundle(source, values)
        digest = bundle_digest(bundle)
        facts = scene_facts(values, source, digest=digest if bundle is source else None)
    _check_cancel(cancel)
    resolved = config.resolve(spec, bundle, supplied=supplied, **facts)
    for w in resolved.warnings:
        log.warning("%s", w)
    reflection_bounds, reflection_notes = reflection_policy(bundle, resolved)
    for w in reflection_notes:
        log.warning("%s", w)
    assumed_nir = assumed_nir_albedo(bundle, resolved)
    nir_notes = [f"sensor.nir_fraction > 0: {len(assumed_nir)} material(s) have the assumed NIR albedo (the red channel, "
                 "DESIGN §7.2; run.json assumed_nir_albedo): set nir_form to fit them"] if assumed_nir else []
    for w in nir_notes:
        log.warning("%s", w)
    if report is not None:
        report(resolved)
    t_s, t_e = resolved.span
    names = [c.name for c in bundle.cameras]
    selected = resolved.cameras
    _check_cancel(cancel)
    engine = make_engine(bundle, resolved)
    _check_cancel(cancel)
    summary = RunSummary(run_dir, engine.backend, t_s, t_e)
    # Untabulated EVIMO2 clock offsets are 0 with a warning (lead decision 6): recorded with the run's warnings.
    clock = [w for w in bundle.meta.get("warnings", []) if isinstance(w, str) and w.startswith("no G1 delta")]
    inputs = {**spec.record, "bundle": {"supplied": supplied},
              "cameras": {c.name: {"pixel_pitch_m": c.pixel_pitch_m, "f_number": c.f_number} for c in bundle.cameras}}
    doc = {"status": "running", "inputs": inputs, "resolved": resolved.record(),
           "warnings": list(resolved.warnings) + clock + reflection_notes + nir_notes, "stage_digests": resolved.digests(),
           # Derived values no stage reads (P2-B B4a's band N: n_band, theta_band, c_cover, eps_switch).
           **({"derived": resolved.facts} if resolved.facts else {}),
           "versions": versions(), "git": git_commit(),
           "bundle": {"name": bundle.name, "digest": digest, "source": bundle.meta.get("source")},
           "light_groups": light_group_record(bundle),
           # DESIGN §7.7's per-material bounds (P2-F F4), only for a bundle with specular materials.
           **({"reflection_bounds": reflection_bounds} if reflection_bounds else {}),
           # DESIGN §7.2 / P2-F F11a: the materials at the assumed NIR albedo, where some sensor.nir_fraction > 0.
           **({"assumed_nir_albedo": assumed_nir} if assumed_nir else {}),
           "span_ns": [t_s, t_e], "window_ns": WINDOW_NS,
           "renderer": {"backend": engine.backend, "n_subsamples": resolved.get("accuracy.n_subsamples")},
           "engine_cpp": {"device_budget_bytes": engine.budget_bytes},     # batching depends on it (ENGINE_CPP §1.2)
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    from ballasim.scene import baked as baked_record, licenses

    baked = baked_record.record(bundle, resolved.get("scene.baked_scale"))
    if baked is not None:
        doc["baked"] = baked
    terms = licenses.records(bundle)
    if terms:
        doc["licenses"] = terms
    for asset in terms:
        if asset["restricted"]:
            warning = f"asset {asset['asset']}: source licence {asset['license']} restricts derived outputs"
            log.warning("%s", warning)
            doc["warnings"].append(warning)
    _create_run_dir(run_dir, doc)
    try:
        runlog = RunLog(run_dir)
        for name in selected:
            _check_cancel(cancel)
            cam_index = names.index(name)
            log.info("camera %s: %d windows from %d ns", name, (t_e - t_s) // WINDOW_NS, t_s)
            cam = engine.camera(cam_index)
            if cancel is not None:
                cam.set_cancel(cancel)
            summary.cameras.append(run_camera(bundle, cam_index, resolved, cam, run_dir, runlog, t_s, t_e,
                                              on_window=on_window, cancel=cancel, warn=doc["warnings"].append))
    except BaseException as e:
        from ballasim import _engine

        if isinstance(e, _engine.Cancelled):
            log.warning("run %s: cancelled", run_dir)
            doc.update(status="cancelled", finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       wall_s=time.perf_counter() - wall, cameras=[asdict(c) for c in summary.cameras])
        else:
            doc.update(status="failed", error=f"{type(e).__name__}: {e}")
        write_json_atomic(run_dir / "run.json", doc)
        raise
    if "baked" in doc:
        proxies = [i for i, light in enumerate(bundle.lights)
                   if light.proxy and light.kind not in CONSTANT_LIGHT_KINDS and light.casts_shadows]
        blocked = doc["baked"]["proxy_blocked"]
        for camera in summary.cameras:
            for i, share in zip(proxies, getattr(camera, "proxy_blocked", ())):
                blocked[str(i)] = max(blocked.get(str(i), 0.0), float(share))
    summary.wall_s = time.perf_counter() - wall
    doc.update(status="complete", finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), wall_s=summary.wall_s,
               simulated_s=summary.simulated_s, cameras=[asdict(c) for c in summary.cameras])
    write_json_atomic(run_dir / "run.json", doc)
    # The IMU reads the final run record and shares its event clock; it never changes the event run's outcome.
    if any(resolved.get("imu.enabled", camera) for camera in selected):
        from ballasim.export.imu import simulate_imu

        simulate_imu(run_dir, bundle=bundle)
    return summary


def _holding_run_lock(fn):
    """:func:`run` holds its run directory's lock (runlog.lock_run_dir, docs/design/phase3/ui.md §4.10) from the check
    of the directory (:func:`_check_run_dir` takes it) to its end, so a second run on the directory, overwrite or not,
    is refused (RunDirInUse; the CLI exits 2). A run that fails before its run directory holds a run.json (a
    configuration error) removes the lock file and the parent directories taking it created."""
    @functools.wraps(fn)
    def locked(*args, **kwargs):
        locks: list = []
        token = _RUN_LOCKS.set(locks)
        done = False
        try:
            result = fn(*args, **kwargs)
            done = True
            return result
        finally:
            _RUN_LOCKS.reset(token)
            for lock in locks:
                lock.release(discard=not done and not (lock.run_dir / "run.json").exists())

    return locked


run = _holding_run_lock(run)
