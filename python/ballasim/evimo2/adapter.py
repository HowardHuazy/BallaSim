"""Build a C1 SceneBundle from an EVIMO2 v2 NPZ sequence (DESIGN §5.2, §10; S2 §2.9, §9.2).

Pipeline for one camera folder (left_camera, right_camera or samsung_mono):

1. Re-time the Vicon rows of ``full_trajectory`` (G1): the bundle's true clock is the re-timed
   Vicon clock, ns since bag start; knots use the re-timed period.
2. World frame W = the tabletop subject frame at its robust mean pose (EVM-02); without the
   tabletop, W = the rig at C0, the camera pose at the first Vicon sample of the *recording*
   (EVIMO's full_trajectory is relative to it; later sequence parts do not start there).
3. Recompose at identical Vicon rows, before any interpolation (R4 t11, M2 C11):
   T_W<-O = T_W<-C0 T_C0<-C T_C<-O and T_W<-Rig = T_W<-C0 T_C0<-C T_C<-Rig.
4. Bodies: EVIMO ids kept as instance ids (AST-25); meshes from the object catalogue (mm -> m),
   mesh -> subject by Kabsch (AST-21), root tabletop (G3 t4); body frame origin at the
   area-weighted mesh centroid with subject-frame axes (DESIGN §5.1, M2 §C.1); optional marker
   spheres from ``config.txt`` (AST-24; cut out of the scans but visible in real images).
5. Trajectories: robust smoothing fits with holds for every body, the rig included; a body that
   is static throughout becomes a static body (SCN-03). Objects absent for part of the sequence
   hold their first/last pose there (on_gap = hold, Phase 1 has no presence intervals).
6. Camera from meta['meta'] (never D.npy), T_rig_cam from dataset_extrinsics.npz, clock offset
   per :mod:`ballasim.evimo2.clocks`, sensor preset and pixel pitch by camera family.
7. Lights (:class:`Evimo2Lighting`, the scene.evimo2 lighting keys; evimo2/lab_lighting@v0 holds the P1-M5 values):
   ambient plus one point light, both in a mains group in normal light; in ``_ll`` sequences the point light is in
   a pulse group frame-locked to the re-timed Vicon grid (the strobe, G3 t9, S2 LGT-06) and a weak ambient stays
   constant. The point light sits ``light_height_m`` along the dominant plane normal of a reference body (see
   light_anchor in src/scene/evimo2/build.cpp). The lights are assembled last, on fixed geometry and trajectories,
   from a small sidecar (:func:`relight_sidecar`: the anchor, low light and the re-timing), so :func:`relight`
   rebuilds them for other lighting values without a rebuild (docs/dev/P0_PARAMS.md §3.4).

Steps 1-6 and the light anchor run in C++ since S4 (``ballasim._scene.build_evimo2``, src/scene/evimo2; the fits on
``execution.scene_workers`` threads), bitwise as the Python they replaced (deleted at S5; the frozen digests in
tests/data/scene_fit_digests.json hold it); this module reads the files, hands the arrays over and assembles the
SceneBundle and its lights (``light_rig`` / ``relight`` run at every load).

The options' defaults are the schema's (``_core.config.default``); the albedos and the lighting have none: their keys
are required (docs/dev/P0_PARAMS.md §3.4).
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ballasim import _core
from ballasim.scene.assets import asset_inputs, file_sha256, load_asset
from ballasim.scene.bundle import (
    Body,
    Camera,
    Light,
    LightGroup,
    Material,
    Mesh,
    SceneBundle,
    Trajectory,
    default_light_group,
)
from ballasim.traj.fit import FitResult, SmoothingOptions
from ballasim.traj.retime import Retiming

from .clocks import g1_delta_ms, raw_line3_s
from .dataset import Sequence, catalogue, catalogue_files, low_light_category, objects_dir

log = logging.getLogger(__name__)

TABLETOP_ID = 22
#: The relight sidecar's format (relight_sidecar).
RELIGHT_FORMAT = "ballasim.relight/1"
_default = _core.config.default


def _check(keys: dict[str, str], obj) -> None:
    """Each field against its schema row (type, length, hard range): one copy of every range (P0_PARAMS §1.1)."""
    for name, key in keys.items():
        _core.config.check(key, getattr(obj, name))


#: Evimo2Lighting fields: the scene.evimo2 lighting keys (stages render, schedule and sensor), applied by relight.
LIGHTING_KEYS = {f: f"scene.evimo2.{f}" for f in (
    "ambient", "low_light_ambient", "light_height_m", "light_color", "light_shadows", "ambient_flickers", "mains_a",
    "mains_f0_hz", "mains_phi_rad", "strobe_width_s", "strobe_floor", "strobe_rise_s", "strobe_phase_offset_s",
    "background_albedo")}


@dataclass(frozen=True, kw_only=True)
class Evimo2Lighting:
    """The placeholder lab lights (the keys but the background's are required; evimo2/lab_lighting@v0 holds the P1-M5
    values).
    Normal light: a point light ``light_height_m`` above its anchor with unit irradiance there, and an ambient term
    (radiance-equivalent rho A next to the point light's direct rho E / pi: the renderer shades rho (E / pi + A), so an
    ambient of 0.3 is about 0.94x the direct term at the anchor, not 0.3x; docs/dev/SCENE.md), both in the mains group
    (fourier: f0, a_k, phi_k) unless the ambient does not flicker. ``_ll``: the point light in the strobe group (one flash per
    re-timed Vicon sample: full width at half height, rise, phase offset, floor) and a constant low-light ambient.
    ``background_albedo`` rho_bg (D13 item 1; 0: none, the default): the rays that miss every mesh (the unscanned room)
    see a backdrop of that albedo lit by the light mode's ambient A alone, radiance rho_bg A in the ambient's group."""

    ambient: float
    low_light_ambient: float
    light_height_m: float
    light_color: list[float]
    light_shadows: bool
    ambient_flickers: bool
    mains_a: list[float]
    mains_f0_hz: float
    mains_phi_rad: list[float]
    strobe_width_s: float
    strobe_floor: float
    strobe_rise_s: float
    strobe_phase_offset_s: float
    background_albedo: float = 0.0

    def __post_init__(self) -> None:
        _check(LIGHTING_KEYS, self)     # single keys; the light groups' cross-key rules are the waveforms' (C++)

    @classmethod
    def from_values(cls, values) -> Evimo2Lighting:
        """From a run's resolved values ({key: value}); a field with a default takes it where the values lack its key
        (a run recorded before the key existed)."""
        optional = {f.name for f in dataclasses.fields(cls) if f.default is not dataclasses.MISSING}
        return cls(**{f: values[k] for f, k in LIGHTING_KEYS.items() if k in values or f not in optional})


#: Evimo2Options fields that are schema keys (their defaults and ranges are the rows'); execution.scene_workers is
#: output-invariant (no cache key).
OPTION_KEYS = {"markers": "scene.evimo2.markers", "marker_subdivisions": "scene.evimo2.marker_subdivisions",
               "clock_reference": "scene.evimo2.clock_reference", "warmup_ns": "scene.evimo2.start_trim_ns",
               "objects": "scene.evimo2.objects", "min_samples": "scene.evimo2.min_samples",
               "static_tabletop": "scene.evimo2.static_tabletop",
               "static_tabletop_fraction": "scene.evimo2.static_tabletop_fraction", "lights": "scene.evimo2.lights",
               "marker_albedo": "scene.evimo2.marker_albedo", "untextured_albedo": "scene.evimo2.untextured_albedo",
               "marker_radius_m": "scene.evimo2.marker_radius_m", "pin_static_px_s": "scene.evimo2.pin_static_px_s",
               "workers": "execution.scene_workers"}


@dataclass(kw_only=True)
class Evimo2Options:
    markers: bool = _default("scene.evimo2.markers")               # add marker spheres (AST-24)
    marker_subdivisions: int = _default("scene.evimo2.marker_subdivisions")   # sagitta 7.7 um at 12.5 mm
    clock_reference: str = _default("scene.evimo2.clock_reference")          # see ballasim.evimo2.clocks
    warmup_ns: int = _default("scene.evimo2.start_trim_ns")        # t_start = first rig sample + trim (DESIGN §5.2)
    objects: list[int] | None = field(default_factory=lambda: _default("scene.evimo2.objects"))   # EVIMO ids (None: all)
    extra_cameras: list[str] = field(default_factory=list)         # sequence directories of other camera folders
    smoothing: SmoothingOptions = field(default_factory=SmoothingOptions)
    min_samples: int = _default("scene.evimo2.min_samples")        # objects with fewer Vicon samples are skipped
    static_tabletop: bool = _default("scene.evimo2.static_tabletop")   # the tabletop defines the world frame
    static_tabletop_fraction: float = _default("scene.evimo2.static_tabletop_fraction")   # build.cpp tabletop_static
    lights: bool = _default("scene.evimo2.lights")
    marker_albedo: float                  # Lambert proxy of the retro-reflective spheres (required key)
    untextured_albedo: float              # the catalogue objects without a texture (AST-23; required key)
    marker_radius_m: float | None = _default("scene.evimo2.marker_radius_m")   # None: the catalogue's
    pin_static_px_s: float | None = _default("scene.evimo2.pin_static_px_s")   # M10-PIN (build.cpp); None: off
    objects_root: str | None = None       # EVIMO object catalogue (None: the environment's, dataset.objects_dir)
    raw_root: str | None = None           # raw EVIMO archive (None: the environment's, clocks.raw_root)
    lighting: Evimo2Lighting | None = None   # required while lights is on
    workers: int = _default("execution.scene_workers")   # threads for the per-body fits (results are identical)

    def __post_init__(self) -> None:
        _check(OPTION_KEYS, self)
        if self.lights and self.lighting is None:
            raise ValueError("Evimo2Options: lights is on and no lighting is given (Evimo2Lighting)")


@dataclass
class Evimo2Build:
    bundle: SceneBundle
    sequence: Sequence
    retiming: Retiming
    T_world_c0: np.ndarray                # world <- this camera at its first Vicon sample
    fits: dict[str, FitResult]            # by body name ("rig" for the camera rig)
    T_body_subject: dict[int, np.ndarray] # per object id: body <- Vicon subject
    warnings: list[str]
    sidecar: dict                         # relight_sidecar(): what relight needs besides the lighting values

    def gt_frame_times(self) -> list[tuple[int, int]]:
        """(archive key, time on the bundle's re-timed clock) for every GT frame (G1 rec. 2)."""
        frames = self.sequence.gt_frames()
        t = self.retiming.to_retimed(np.array([f[1] for f in frames], np.int64))
        return [(f[0], int(ti)) for f, ti in zip(frames, t)]


def relight_sidecar(low_light: bool, retiming: Retiming, anchor: tuple[np.ndarray, np.ndarray, str] | None) -> dict:
    """What the lights need besides the lighting values (the cache entry's relight.json): low light, the re-timing
    (the strobe's phase and period) and the anchor (direction, centre, source; None without lights), floats as hex."""
    hexes = (lambda a: [float(x).hex() for x in a])
    return {"format": RELIGHT_FORMAT, "low_light": bool(low_light),
            "retiming": {"a_ns": float(retiming.a_ns).hex(), "b_ns": float(retiming.b_ns).hex()},
            "anchor": None if anchor is None else {"direction": hexes(anchor[0]), "centre": hexes(anchor[1]),
                                                   "source": anchor[2]}}


def light_rig(sidecar: dict, lighting: Evimo2Lighting | None) -> tuple[list[LightGroup], list[Light], dict]:
    """The light groups, the lights and meta['lights'] of a sidecar and lighting values (None: no lights)."""
    if sidecar.get("format") != RELIGHT_FORMAT:
        raise ValueError(f"relight sidecar: format {sidecar.get('format')!r}, expected {RELIGHT_FORMAT}")
    if lighting is None:
        return [default_light_group()], [], {"default": "none"}
    anchor = sidecar["anchor"]
    if anchor is None:
        raise ValueError("relight: the bundle was built without lights (scene.evimo2.lights is a scene key)")
    up = np.array([float.fromhex(x) for x in anchor["direction"]])
    centre = np.array([float.fromhex(x) for x in anchor["centre"]])
    lo = lighting
    groups = [default_light_group()]
    color = np.asarray(lo.light_color, np.float32)
    h = lo.light_height_m
    point = Light("point", 1, color, h**2, None, centre + h * up, -up, casts_shadows=bool(lo.light_shadows))
    if sidecar["low_light"]:
        # The Vicon strobe: one flash per re-timed Vicon sample, rising at each sample time (plus the offset); duty is
        # the width at half height over the period (C1 pulse convention), so 0.3 ms wide = 0.25 ms plateau + 50 us
        # ramps.
        a_ns, b_ns = (float.fromhex(sidecar["retiming"][k]) for k in ("a_ns", "b_ns"))
        period_s = b_ns * 1e-9
        groups.append(LightGroup("strobe", {"type": "pulse", "f_hz": 1.0 / period_s,
                                            "duty": lo.strobe_width_s / period_s, "rise_s": lo.strobe_rise_s,
                                            "phase_s": (a_ns * 1e-9 + lo.strobe_phase_offset_s) % period_s,
                                            "floor": lo.strobe_floor}))
        ambient = lo.low_light_ambient
        lights = [point, Light("ambient", 0, color, ambient)]
    else:
        groups.append(LightGroup("mains", {"type": "fourier", "f0_hz": lo.mains_f0_hz, "a": list(lo.mains_a),
                                           "phi": list(lo.mains_phi_rad)}))
        ambient = lo.ambient
        lights = [point, Light("ambient", 1 if lo.ambient_flickers else 0, color, ambient)]
    background = lo.background_albedo * ambient
    if background > 0:                  # none at 0: the bundle (and its digest) as before D13 item 1
        lights.append(Light("background", lights[1].group, color, background, casts_shadows=False))
    a = list(lo.mains_a)
    meta = {"default": f"point + ambient, mains {lo.mains_f0_hz:g} Hz (normal light) or 200 Hz strobe (_ll)",
            "anchor": anchor["source"], "direction_world": up.tolist(), "centre_world": centre.tolist(),
            "ambient": ambient,
            "mains_amplitude": None if sidecar["low_light"] else (a[0] if len(a) == 1 else a)}
    if background > 0:
        meta["background_radiance"] = background
    return groups, lights, meta


def bundle_low_light(bundle: SceneBundle) -> bool:
    """The light mode of an EVIMO2 bundle, its relight sidecar's ``low_light``: an _ll recording by the category its
    meta records (Sequence.low_light)."""
    return low_light_category(bundle.meta["source"]["category"])


def relight(bundle: SceneBundle, sidecar: dict, lighting: Evimo2Lighting | None) -> SceneBundle:
    """The bundle with its light groups, lights and meta['lights'] rebuilt from the sidecar and lighting values (None:
    no lights); geometry, trajectories and cameras are shared. At the values it was built with, the bundle itself
    (equal digest)."""
    groups, lights, meta = light_rig(sidecar, lighting)
    return dataclasses.replace(bundle, light_groups=groups, lights=lights,
                               meta={**bundle.meta, "lights": meta}).validate()


#: meta['lights'] of a bundle built with ``lit=False``: its lights are assembled at load.
UNLIT = {"default": "none until relit: relight() assembles the lights from relight.json and the scene.evimo2 lighting "
                    "keys at every load (ballasim.engine.scene.load_evimo2)"}


@functools.cache
def _object_ids(sequence_dir: str) -> tuple[int, ...]:
    return tuple(Sequence(sequence_dir).object_ids)


def catalogue_inputs(sequence_dir: str | Path, *, objects_root: str | None = None,
                     objects: list[int] | None = None) -> list[dict]:
    """The catalogue files :func:`build_bundle` reads for one sequence, with their SHA-256 (memoized by path, size and
    mtime): every object folder's settings.txt, config.txt and vicon.vsk (``name`` relative to the catalogue), then
    the mesh and texture of each of the sequence's objects in ``objects`` (None: all), as the import names them. The
    bundle cache key holds them (ballasim.engine.scene.evimo2_cache_key): a catalogue edit or another catalogue
    (``objects_root``, scene.evimo2.objects_root) never reuses a stale bundle; equal content shares one entry."""
    base = objects_dir() if objects_root is None else Path(objects_root)
    out = [{"name": f.relative_to(base).as_posix(), "sha256": file_sha256(f)} for f in catalogue_files(base)]
    cat = catalogue(objects=base)
    for oid in _object_ids(str(sequence_dir)):
        if oid in cat and (objects is None or oid in objects):
            out += [{"object": cat[oid].folder.name, **e} for e in asset_inputs(cat[oid].mesh_path)]
    return out


#: The options of a fit that are SmoothingOptions scalars (what ballasim._scene.build_evimo2 reads).
_SMOOTHING_FIELDS = tuple(f.name for f in dataclasses.fields(SmoothingOptions) if f.name != "hold_view")
#: The Evimo2Options fields the C++ build reads.
_BUILD_OPTIONS = ("markers", "marker_subdivisions", "clock_reference", "warmup_ns", "objects", "min_samples",
                  "static_tabletop", "static_tabletop_fraction", "lights", "marker_albedo", "untextured_albedo",
                  "marker_radius_m", "pin_static_px_s", "workers")


def _camera_input(s: Sequence, opts: Evimo2Options) -> dict:
    """One camera folder as the C++ build takes it: calibration, T_rig_cam and the clock inputs (the G1 table entry and
    the raw extrinsics' line 3, read here; clocks.clock_offset's arithmetic is the build's)."""
    fx, fy, cx, cy, dist = s.radtan()
    return {"camera": s.camera, "directory": str(s.directory), "category": s.category, "recording": s.recording,
            "name": s.name, "width": s.width, "height": s.height, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "dist": dist,
            "T_rig_cam": s.T_rig_cam, "g1": g1_delta_ms(s),
            "raw_line3_s": raw_line3_s(s, None if opts.raw_root is None else Path(opts.raw_root))}


def build_bundle(sequence_dir: str | Path, options: Evimo2Options, *, name: str | None = None,
                 lit: bool = True) -> Evimo2Build:
    """The bundle of one camera folder of a sequence (``ballasim.engine.scene.evimo2_options`` gives the options of a
    run's resolved values). ``lit=False`` leaves the lights to :func:`relight` (none, ``meta['lights']`` UNLIT; the
    sidecar still holds the anchor while ``options.lights`` is on): what the bundle cache stores, so a lighting value
    the sequence's light rig refuses (the strobe against the Vicon period) cannot lose the build.

    The build is C++ (``ballasim._scene.build_evimo2``, docs/dev/CPP_PORT.md item 5), bitwise as the Python it replaced:
    this reads the sequence's npz (U4), the catalogue and the asset cache (both C++ underneath), hands the arrays over,
    and assembles the SceneBundle and its lights."""
    from ballasim import _scene

    opts = options
    seq = Sequence(sequence_dir)
    cat = catalogue(objects=None if opts.objects_root is None else Path(opts.objects_root))
    cameras_seq = [seq] + [Sequence(d) for d in opts.extra_cameras]
    ids = [i for i in seq.object_ids if opts.objects is None or i in opts.objects]
    track_ids = ids + ([TABLETOP_ID] if TABLETOP_ID in seq.object_ids and TABLETOP_ID not in ids else [])
    tracks = {i: seq.track(str(i)) for i in track_ids}
    assets, objects = {}, {}
    for oid in ids:
        if oid not in cat:
            continue
        obj = cat[oid]
        entry = {"name": obj.name, "unit_scale": obj.unit_scale, "config_markers": obj.config_markers,
                 "vsk_markers_mm": obj.vsk_markers_mm, "marker_radius_mm": obj.marker_radius_mm, "mesh": None}
        if len(tracks[oid].rows) >= opts.min_samples:     # the build skips the others, with a warning
            asset = assets[oid] = load_asset(obj.mesh_path, scale=obj.unit_scale)
            m = asset.mesh
            entry.update(mesh=(m.positions, m.indices, m.normals, m.uvs, m.material_ids),
                         num_materials=len(asset.materials),
                         placeholder=all(mat.texture is None for mat in asset.materials))
        objects[oid] = entry
    cam = seq.track("cam")
    inputs = {"t_raw_ns": seq.full_trajectory_t_raw_ns, "cam": (cam.rows, cam.q, cam.p),
              "object_ids": list(seq.object_ids), "tracks": {i: (t.rows, t.q, t.p) for i, t in tracks.items()},
              "catalogue": objects, "cameras": [_camera_input(s, opts) for s in cameras_seq]}
    build_opts = {k: getattr(opts, k) for k in _BUILD_OPTIONS}
    build_opts["smoothing"] = {k: getattr(opts.smoothing, k) for k in _SMOOTHING_FIELDS}
    r = _scene.build_evimo2(inputs, build_opts)

    fit_list = [(fname, FitResult(Trajectory(f["order"], f["t0_ns"], f["h_ns"], f["rot"], f["pos"], f["holds"],
                                             f["span_ns"]), f["diagnostics"], f["keep"], f["weights"]))
                for fname, f in r["fits"]]
    fits = dict(fit_list)
    trajectories = [fit_list[i][1].trajectory for i in r["trajectories"]]
    meta = r["meta"]
    bodies = []
    for b in r["bodies"]:
        asset = assets[b["id"]]
        materials = ([Material(np.full(3, opts.untextured_albedo, np.float32))] if b["placeholder"]
                     else list(asset.materials))
        if b["markers"]:
            materials.append(Material(np.full(3, opts.marker_albedo, np.float32)))
        bodies.append(Body(b["id"], b["name"], b["semantic"], Mesh(*b["mesh"]), materials, b["trajectory"],
                           b["T_world_body"]))
        meta["bodies"][b["name"]]["asset"] = {k: v for k, v in asset.info.items()
                                              if k in ("cache_key", "layout", "validation")}
    cameras = [Camera(c["name"], "radtan", c["width"], c["height"], c["fx"], c["fy"], c["cx"], c["cy"], c["dist"], 0,
                      c["T_rig_cam"], c["clock_offset_ns"], c["sensor"], c["pixel_pitch_m"], 2.0) for c in r["cameras"]]
    sidecar = r["sidecar"]
    groups, lights, light_meta = light_rig(sidecar, opts.lighting if opts.lights and lit else None)
    meta["lights"] = dict(UNLIT) if opts.lights and not lit else light_meta
    bundle = SceneBundle(name or f"evimo2_{seq.camera}_{seq.category}_{seq.name}", r["t_start_ns"], r["t_end_ns"],
                         np.zeros(3), bodies, trajectories, cameras, groups, lights, meta)
    bundle.validate()
    for w in r["warnings"]:
        log.warning(w)
    rt = r["retiming"]
    retiming = Retiming(rt["t_raw_ns"], rt["frame_index"], rt["t_ns"], rt["a_ns"], rt["b_ns"], rt["residual_std_ns"],
                        rt["num_gaps"])
    return Evimo2Build(bundle, seq, retiming, r["T_world_c0"], fits, r["T_body_subject"], meta["warnings"], sidecar)


