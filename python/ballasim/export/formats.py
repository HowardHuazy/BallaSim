"""The export driver (docs/dev/EXPORT.md; docs/design/phase3/outputs.md §3.4, §3.7, §4.7): ``export(run_dir, fmt)``
writes a finished run's events (C4) and IMU (C5) to a standard format under ``<run>/exports/<format>/`` with one
``EXPORT.json`` (C6). Python reads the run with h5py and writes the record; the C++ writers of ``ballasim._export``
map the times, encode and hash (GIL released).

Time rules (§4.7). Each camera keeps its sensor clock clock_c(t) = t + offset + llround(drift 1e-6 t). ``t0_ns`` and
``t1_ns`` are on the bundle (true) clock; events are kept when clock_c(t0) <= t_reported < clock_c(t1), IMU samples
by their stamp. Without ``t1_ns`` there is no upper cut: the whole event file is exported, including the events the
run flushed after clock_c(t_e). The origin is min over the cameras of floor(clock_c(t0) / 1000), else min /t_offset,
so every kept time maps to u >= 0; ``epoch_ns`` must be a multiple of 1000 for every format.

This version writes RAW (EVT 2.0, 2.1, 3.0) and DAT with the IMU CSV sidecar (P3-A A5), rosbag1 (A6), MCAP (A7) and
AEDAT4 (A8): one bag, MCAP file of ROS 2 messages or AEDAT4 file of every selected camera's events (dvs_msgs/EventArray
or event_camera_msgs/EventPacket evt3; MCAP: EventPacket evt3; AEDAT4: EVTS packets) and IMU (sensor_msgs/Imu; AEDAT4:
IMUS packets in dv's units), fed to the C++ writer in 100 ms windows of t_reported across the cameras and streams
(§4.8 "Stream interleaving"), stamps by T2 (ROS) or T1b (AEDAT4).

A9 adds Kalibr (§4.9: per camera with an IMU, imu-<cam>.yaml and camchain-imucam-<cam>.yaml from its C5 parameters and
the bundle camera) and the frame streams of AEDAT4 (FRME), rosbag1 and MCAP (sensor_msgs/Image) from each camera's
frames file ``frames/<cam>.h5`` (contract C7, frames.md §3.6). Frame times: the file's true-clock exposure start and end
of the first and last read rows, mapped by the camera's sensor clock clock_c (§4.7): exposure start = frame start =
clock_c(t_true_soe), exposure end = clock_c(t_true_eoe), frame end = clock_c(t_true_eoe + one row time), midpoint =
floor((start + end) / 2), which must equal the file's own sensor-clock stamps (frames.md §3.11). A frame is exported
when its midpoint lies in the window and its start at or after the origin (earlier ones are dropped and counted). The
pixels are the camera's APS output (aps_dn, gray16 of frames.aps.bits) when present, else rgb, else mono (float32,
linear), or the one --frame-source names; AEDAT4 frames are 8-bit, so they need --frame-map linear8 or srgb8.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import record
from .errors import ExportError

log = logging.getLogger(__name__)

FORMATS = ("raw-evt3", "raw-evt2", "raw-evt21", "dat", "aedat4", "rosbag1", "mcap", "kalibr")
COMPRESSIONS = ("none", "bz2", "lz4", "zstd", "lz4-high", "zstd-high")
#: §3.7: the --compression values each format allows; the first default is the format's own.
CODECS = {"raw-evt3": ("none",), "raw-evt2": ("none",), "raw-evt21": ("none",), "dat": ("none",), "kalibr": ("none",),
          "aedat4": ("none", "lz4", "lz4-high", "zstd", "zstd-high"), "rosbag1": ("none", "bz2", "lz4"),
          "mcap": ("none", "lz4", "zstd")}
DEFAULT_CODEC = {"aedat4": "zstd", "rosbag1": "bz2", "mcap": "zstd"}
#: §4.7: 0 for RAW, DAT and AEDAT4; 1 s for ROS1 and MCAP (ROS time 0 means "unset").
DEFAULT_EPOCH_NS = {"rosbag1": 1_000_000_000, "mcap": 1_000_000_000}
#: The formats of this version and their RAW encoding (None: DAT).
WRITERS = {"raw-evt3": "evt3", "raw-evt2": "evt2", "raw-evt21": "evt21", "dat": None}
#: The options of export(**options) and their defaults (the CLI's, §3.4).
OPTIONS = {"compression": None, "imu": True, "frames": True, "event_msg": None, "packet_us": None, "topics": (),
           "imu_covariance": "model", "evt2_time_high": "openeb", "frame_map": "none", "frame_scale": 1.0,
           "frame_source": "auto", "kalibr_inflate": 1.0}
#: --packet-us when not given: ROS messages of 1 ms (§3.7), AEDAT4 packets of 10 ms (every other format records 1000).
DEFAULT_PACKET_US = {"aedat4": 10_000}
EVENT_BATCH = 1 << 22  # events per h5py read (§5.3)
IMU_BATCH = 1 << 20
WINDOW_NS = 100_000_000  # the containers' interleave: 100 ms of t_reported per round over the cameras and streams
ROS_FORMATS = ("rosbag1", "mcap")
#: The formats that carry frame streams (§3.7 "Frames").
FRAME_FORMATS = ("aedat4", "rosbag1", "mcap")
#: --frame-source: the frames file's image datasets (C7), and the order "auto" takes them in.
FRAME_SOURCES = {"aps": "aps_dn", "rgb": "rgb", "mono": "mono"}
FRAME_BATCH_BYTES = 1 << 26  # pixels per h5py read of a frames file
#: --topic STREAM:CAMERA=NAME: the streams and their default topics /<cam>/<stream> (§3.7).
TOPIC_STREAMS = ("events", "imu", "image_raw")
MAX_THREADS = 256  # the C++ compression pool's limit
MAX_PACKET_US = 1 << 24  # ros::kMaxPacketUs: EventPacket evt3 words carry u mod 2^24 (§4.7); AEDAT4 keeps the range
AEDAT4_MAX_SIZE = 32767  # AEDAT4 Event x and y are int16
#: T2 stamps each ROS format holds (ns): ROS1 time has u32 seconds; ROS 2 time int32 seconds, and MCAP log times are
#: unsigned.
STAMP_RANGE = {"rosbag1": ((1 << 32) * 1_000_000_000, "ROS1's [0, 2^32 s) of u32 seconds"),
               "mcap": ((1 << 31) * 1_000_000_000,
                        "ROS 2's [0, 2^31 s) of int32 seconds (MCAP log times are unsigned)")}
EVENTS_FORMAT = "ballasim.events/1"
IMU_RUN_FORMAT = "ballasim.imu_run/1"
SENSOR_KEYS = ("sensor.clock_offset_ns", "sensor.clock_drift_ppm", "sensor.readout_delay_s", "sensor.readout_quantum_s")


@dataclass
class _Camera:
    """One exported camera: its files, clock, window and what the pre-scans found."""
    name: str
    events: Path
    width: int
    height: int
    t_offset_us: int
    sensor: dict
    times: dict = field(default_factory=dict)  # _export.camera_times: window_reported_ns, file_end_bound_ns
    i0: int = 0  # the window's events: [i0, i1) of the file
    i1: int = 0
    rem: tuple = (-1, -1)  # min and max dropped t_ns_rem in the window; (-1, -1) without events
    first_ns: int | None = None  # the window's first and last t_reported
    last_ns: int | None = None
    imu: dict = field(default_factory=dict)  # EXPORT.json's imu entry
    imu_file: Path | None = None
    imu_window: tuple = (0, 0)  # the window's samples: [j0, j1)
    imu_first_ns: int | None = None
    imu_last_ns: int | None = None
    white_var: tuple = (0.0, 0.0)  # the ROS Imu covariances' per-sample white variances (gyro, accel)
    frames: dict = field(default_factory=dict)  # EXPORT.json's frames entry
    frames_file: Path | None = None
    frame_dataset: str = ""  # aps_dn, rgb or mono
    frame_window: tuple = (0, 0)  # the exported frames: [k0, k1)
    frame_times: dict = field(default_factory=dict)  # every frame's mid, fs, fe, es, ee (sensor clock, ns)
    frame_args: dict = field(default_factory=dict)  # bits, x0, y0, exposure_ns of the writers' frames calls
    image_roi: list | None = None  # Kalibr: the frames file's ROI [x0, y0, x1, y1], the images on the image topic


def export(run_dir, fmt: str, *, out=None, cameras=None, epoch_ns: int | None = None, t0_ns: int | None = None,
           t1_ns: int | None = None, overwrite: bool = False, threads: int = 4, bundle=None, **options) -> Path:
    """Exports a finished run to ``fmt`` (one of FORMATS) and returns the export directory (default
    ``<run>/exports/<fmt>``), written as ``.<name>.partial`` and renamed; an existing one is replaced only with
    ``overwrite`` and only if it holds an EXPORT.json. ``cameras``: default the run's. ``epoch_ns``: default 0, ROS1
    and MCAP 1 s. ``t0_ns``, ``t1_ns``: the cuts (bundle clock, ns; default none). ``options``: those of OPTIONS (the
    CLI's): ``imu=False`` leaves out the IMU streams, ``frames=False`` the frame streams. ``bundle``: the run's bundle
    (its digest checked), read by the Kalibr export only (the intrinsics), which otherwise loads the run's scene; the
    other formats take the resolution from the event files. Refusals raise ExportError."""
    run_dir = Path(run_dir).absolute()
    opts = _resolve_options(fmt, epoch_ns, threads, options)
    from ballasim import _core

    if not _core.build_info()["export"]:
        raise ExportError("ballasim was built without _export (it needs liblz4, libzstd and libbz2, docs/dev/BUILD.md)")
    from ballasim import _export

    doc, run_sha = _read_run(run_dir)
    out_dir = _out_dir(run_dir, fmt, out, overwrite)
    values = {c: {k: e["value"] for k, e in v.items()} for c, v in doc["resolved"]["cameras"].items()}
    span = [int(v) for v in doc["span_ns"]]
    _check_window(t0_ns, t1_ns, span)
    selected = _select(values, cameras)
    cams = [_open_camera(run_dir, name, values[name]) for name in selected]
    for c in cams:
        s = c.sensor
        c.times = _export.camera_times(int(s["sensor.clock_offset_ns"]), float(s["sensor.clock_drift_ppm"]),
                                       float(s["sensor.readout_delay_s"]), float(s["sensor.readout_quantum_s"]),
                                       span[1], t0_ns, t1_ns)
    lows = [c.times["window_reported_ns"][0] for c in cams]
    origin_us = min(lo // 1000 for lo in lows) if t0_ns is not None else min(c.t_offset_us for c in cams)
    epoch = opts["epoch_ns"]
    warnings: list[str] = []
    for c in cams:
        if fmt != "kalibr":  # Kalibr writes no events
            _scan_events(c)
        _plan_imu(run_dir, c, values[c.name], opts["imu"], warnings)
        _plan_frames(run_dir, c, values[c.name], fmt, opts, origin_us, warnings)
    kalibr_cams = _plan_kalibr(run_dir, doc, bundle, cams, warnings) if fmt == "kalibr" else {}
    _check_limits(fmt, cams, origin_us, epoch)
    if fmt in ROS_FORMATS:
        for c in cams:
            _plan_covariance(c, opts["imu_covariance"])

    partial = out_dir.with_name(f".{out_dir.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    made = [d for d in (partial, *partial.parents) if not d.exists()]  # removed again if the export fails
    partial.mkdir(parents=True)
    try:
        inputs = []
        for c in cams:
            inputs.append(record.input_entry(run_dir, c.events))
            if c.imu_file is not None:
                inputs.append(record.input_entry(run_dir, c.imu_file))
            if c.frames_file is not None:
                inputs.append(record.input_entry(run_dir, c.frames_file))
            elif c.image_roi is not None:  # Kalibr read its ROI
                inputs.append(record.input_entry(run_dir, run_dir / "frames" / f"{c.name}.h5"))
        dropped = cams
        frame_units = {}
        if fmt in ROS_FORMATS:
            name = f"{run_dir.name}.{'bag' if fmt == 'rosbag1' else 'mcap'}"
            outputs = [_write_ros(_export, fmt, cams, partial / name, origin_us, opts)]
            evt3 = opts["event_msg"] == "evt3"
            rules = {name: "T2; EventPacket evt3 words T1a (u mod 2^24)" if evt3 else "T2"}
            dropped = cams if evt3 else []  # dvs_msgs stamps every event in exact ns
            if fmt == "rosbag1":
                units = {"rosbag1": {"stamps": "ns: t_reported - 1000 origin_us + epoch_ns (T2), u32 seconds",
                                     "imu": "sensor_msgs/Imu: rad/s, m/s^2 (float64, the published values); "
                                            "orientation unknown (orientation_covariance[0] = -1)"}}
            else:
                units = {"mcap": {"stamps": "ns: t_reported - 1000 origin_us + epoch_ns (T2), int32 seconds; log "
                                            "time = publish time = header.stamp",
                                  "imu": "sensor_msgs/msg/Imu: rad/s, m/s^2 (float64, the published values); "
                                         "orientation unknown ((0, 0, 0, 1) with orientation_covariance[0] = -1)"}}
            frame_units = {"frames": {"image": "sensor_msgs/Image, header.stamp the exposure midpoint (T2); data as "
                                               "stored (mono8, rgb8, mono16, 32FC1, 32FC3), or mono8 or rgb8 through "
                                               "--frame-map", "frame_id": "<camera>"}}
        elif fmt == "aedat4":
            name = f"{run_dir.name}.aedat4"
            outputs = [_write_aedat4(_export, cams, partial / name, origin_us, opts)]
            rules = {name: "T1b"}
            units = {"aedat4": {"timestamps": "us: floor(t_reported / 1000) - origin_us + epoch_ns / 1000 (T1b), int64",
                                "packets": f"{opts['packet_us']} us of the T1a grid per stream"},
                     "aedat4_imu": {"accelerometer": "g = m/s^2 / 9.81007 (dv getAccelerations)", "gyroscope": "deg/s",
                                    "temperature": 0, "magnetometer": 0}}
            frame_units = {"frames": {"timestamp": "the exposure midpoint (T1b us)",
                                      "start_of_frame": "start of exposure of the first read row",
                                      "start_of_exposure": "start of exposure of the first read row",
                                      "end_of_exposure": "end of exposure of the last read row",
                                      "end_of_frame": "end of exposure of the last read row plus one row time",
                                      "exposure": "the per-row exposure, us (floored)",
                                      "pixels": "8-bit through --frame-map: OPENCV_8U_C1 gray, OPENCV_8U_C3 BGR",
                                      "position": "the ROI's top-left pixel", "source": "SYNTHETIC",
                                      "packets": "one frame per packet"}}
        elif fmt == "kalibr":
            outputs, rules = _write_kalibr(_export, cams, kalibr_cams, partial, opts)
            units = {"kalibr": {"noise_density": "rad/s/sqrt(Hz), m/s^2/sqrt(Hz) (Kalibr's continuous-time "
                                                 "convention), times --kalibr-inflate",
                                "random_walk": "rad/s^2/sqrt(Hz), m/s^3/sqrt(Hz), times --kalibr-inflate",
                                "update_rate": "imu.rate_hz (Hz, true clock)",
                                "timeshift_cam_imu": "s: tau_g + imu.stamp_offset_ns - sensor.readout_delay_s "
                                                     "(t_imu = t_cam + shift)",
                                "T_cam_imu": "X_cam = T X_imu, the rotation re-orthonormalized as the IMU uses it"}}
            dropped = []
        else:
            outputs, rules = [], {}
            for c in cams:
                name = f"{c.name}.dat" if fmt == "dat" else f"{c.name}.raw"
                outputs.append(_write_events(_export, fmt, c, partial / name, origin_us, epoch,
                                             opts["evt2_time_high"]))
                rules[name] = "T1a"
                if c.imu_file is not None:
                    name = f"{c.name}.imu.csv"
                    outputs.append(_write_imu_csv(_export, c, partial / name, origin_us, epoch))
                    rules[name] = "T1a"
            units = {"imu_csv": {"t_us": "us after origin_us (T1a)", "counts": "raw ADC counts", "gyro": "rad/s",
                                 "accel": "m/s^2"}}
        clock = {"epoch_ns": epoch, "origin_us": origin_us, "t0_ns": t0_ns, "t1_ns": t1_ns,
                 "cameras": {c.name: {"t_offset_us": c.t_offset_us,
                                      "clock_offset_ns": int(c.sensor["sensor.clock_offset_ns"]),
                                      "drift_ppm": float(c.sensor["sensor.clock_drift_ppm"]), **c.times}
                             for c in cams},
                 "rules": rules,
                 "dropped_t_ns_rem": {c.name: {"min": c.rem[0], "max": c.rem[1]} if c.rem[0] >= 0 else
                                      {"min": None, "max": None} for c in dropped}}
        options_record = {"format": fmt, "cameras": [c.name for c in cams], "out": str(out_dir),
                          "overwrite": bool(overwrite), "t0_ns": t0_ns, "t1_ns": t1_ns, **opts}
        if any(c.frames_file is not None for c in cams):
            units.update(frame_units)
        doc_out = record.make_record(run_dir=run_dir, run_json_sha256=run_sha, inputs=inputs, options=options_record,
                                     clock=clock, units=units,
                                     frames={"map": opts["frame_map"], "scale": opts["frame_scale"],
                                             "source": opts["frame_source"],
                                             "cameras": {c.name: c.frames for c in cams}},
                                     imu={c.name: c.imu for c in cams}, outputs=outputs, warnings=warnings)
        record.write_record(partial / "EXPORT.json", doc_out)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(partial, out_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        for d in made[1:]:  # the parents this export created (e.g. <run>/exports), innermost first, if still empty
            try:
                d.rmdir()
            except OSError:
                break
        raise
    log.info("exported %s to %s (%s)", run_dir, out_dir, fmt)
    return out_dir


def _resolve_options(fmt: str, epoch_ns, threads, options: dict) -> dict:
    """Every option, checked and resolved per format (§3.7, §6); ExportError lists the allowed values."""
    if fmt not in FORMATS:
        raise ExportError(f"--format {fmt!r}: one of {', '.join(FORMATS)}")
    unknown = set(options) - set(OPTIONS)
    if unknown:
        raise ExportError(f"unknown export options {sorted(unknown)}: {', '.join(OPTIONS)}")
    o = {**OPTIONS, **options}
    epoch = DEFAULT_EPOCH_NS.get(fmt, 0) if epoch_ns is None else epoch_ns
    if (isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or epoch % 1000 != 0
            or not -(1 << 63) <= int(epoch) < 1 << 63):
        raise ExportError(f"--epoch-ns {epoch_ns}: must be an integer multiple of 1000 (every format, §4.7) that fits "
                          f"int64")
    codec = DEFAULT_CODEC.get(fmt, "none") if o["compression"] is None else o["compression"]
    if codec not in CODECS[fmt]:
        raise ExportError(f"--compression {codec} is not allowed for {fmt}: {', '.join(CODECS[fmt])}")
    event_msg = o["event_msg"]
    if event_msg not in (None, "dvs", "evt3"):
        raise ExportError(f"--event-msg {event_msg!r}: dvs or evt3")
    if fmt == "mcap" and event_msg == "dvs":
        raise ExportError("--event-msg dvs is not allowed for mcap (no ROS 2 dvs_msgs): evt3")
    if event_msg is None:
        event_msg = {"rosbag1": "dvs", "mcap": "evt3"}.get(fmt)
    for key, allowed in (("imu_covariance", ("model", "zero")), ("evt2_time_high", ("openeb", "change")),
                         ("frame_map", ("none", "linear8", "srgb8")), ("frame_source", ("auto", *FRAME_SOURCES))):
        if o[key] not in allowed:
            raise ExportError(f"--{key.replace('_', '-')} {o[key]!r}: one of {', '.join(allowed)}")
    if isinstance(threads, bool) or not isinstance(threads, (int, np.integer)) or not 1 <= threads <= MAX_THREADS:
        raise ExportError(f"--threads {threads}: an integer in [1, {MAX_THREADS}]")
    packet = DEFAULT_PACKET_US.get(fmt, 1000) if o["packet_us"] is None else o["packet_us"]
    if isinstance(packet, bool) or not isinstance(packet, (int, np.integer)) or not 0 < packet <= MAX_PACKET_US:
        raise ExportError(f"--packet-us {packet}: an integer in [1, 2^24] (2^24 us = 16.78 s; EventPacket evt3 words "
                          f"carry u mod 2^24)")
    topics = [str(t) for t in o["topics"]]
    if topics and fmt not in (*ROS_FORMATS, "kalibr"):
        raise ExportError(f"--topic applies to {', '.join(ROS_FORMATS)} and kalibr only, not {fmt}")
    seen = set()
    for t in topics:
        stream, sep, rest = t.partition(":")
        camera, sep2, name = rest.partition("=")
        if not (sep and sep2 and camera and name) or stream not in TOPIC_STREAMS:
            raise ExportError(f"--topic {t!r}: STREAM:CAMERA=NAME with STREAM one of {', '.join(TOPIC_STREAMS)}")
        if (stream, camera) in seen:
            raise ExportError(f"--topic {t!r}: {stream} of camera {camera} named twice")
        seen.add((stream, camera))
    for key in ("frame_scale", "kalibr_inflate"):
        v = o[key]
        if not isinstance(v, (int, float, np.number)) or isinstance(v, bool) or not (math.isfinite(v) and v > 0):
            raise ExportError(f"--{key.replace('_', '-')} {v!r}: a finite number > 0")
    if o["frame_scale"] != 1.0 and o["frame_map"] == "none":
        raise ExportError("--frame-scale applies with --frame-map linear8 or srgb8 only")
    if fmt == "kalibr" and not o["imu"]:
        raise ExportError("--no-imu: Kalibr's files are the IMU's (§4.9)")
    return {"epoch_ns": int(epoch), "compression": codec, "threads": int(threads), "imu": bool(o["imu"]),
            "frames": bool(o["frames"]), "event_msg": event_msg, "packet_us": int(packet),
            "topics": topics, "imu_covariance": o["imu_covariance"],
            "evt2_time_high": o["evt2_time_high"], "frame_map": o["frame_map"], "frame_scale": float(o["frame_scale"]),
            "frame_source": o["frame_source"], "kalibr_inflate": float(o["kalibr_inflate"])}


def _read_run(run_dir: Path) -> tuple[dict, str]:
    """run.json and its SHA-256; the refusals of a finished run (dense.recorded_values' two, restated)."""
    raw = (run_dir / "run.json").read_bytes()
    doc = json.loads(raw)
    if doc.get("status") != "complete":
        raise ExportError(f"{run_dir}: the run is not complete (run.json status {doc.get('status')!r})")
    if "resolved" not in doc:
        raise ExportError(f"{run_dir}: a format-1 run directory (run.json has no resolved block): migrate it with "
                          f"python tools/config/convert_v1.py --runs DIR (docs/dev/P0_PARAMS.md §3.8)")
    if doc["resolved"].get("bundle_match") is False:
        raise ExportError(f"{run_dir}: a migrated format-1 run whose scene no longer gives its bundle "
                          f"(resolved.bundle_match is false)")
    return doc, hashlib.sha256(raw).hexdigest()


def _check_window(t0_ns, t1_ns, span: list[int]) -> None:
    for name, t in (("--t0", t0_ns), ("--t1", t1_ns)):
        if t is not None and (isinstance(t, bool) or not isinstance(t, (int, np.integer))):
            raise ExportError(f"{name}: an integer ns time on the bundle clock")
    if t0_ns is not None and not span[0] <= t0_ns < span[1]:
        raise ExportError(f"--t0 = {t0_ns} ns lies outside the run span [{span[0]}, {span[1]}) ns")
    if t1_ns is not None and not span[0] < t1_ns <= span[1]:
        raise ExportError(f"--t1 = {t1_ns} ns lies outside the run span ({span[0]}, {span[1]}] ns")
    if t0_ns is not None and t1_ns is not None and t1_ns <= t0_ns:
        raise ExportError(f"--t1 = {t1_ns} ns must be after --t0 = {t0_ns} ns")


def _select(values: dict, cameras) -> list[str]:
    if cameras is None:
        return list(values)
    selected = list(dict.fromkeys(cameras))
    unknown = [c for c in selected if c not in values]
    if unknown or not selected:
        raise ExportError(f"cameras {unknown or selected}: the run has {', '.join(values)}")
    return selected


def _open_camera(run_dir: Path, name: str, values: dict) -> _Camera:
    import h5py
    import hdf5plugin  # noqa: F401  (the C4 columns are Blosc-compressed)

    from ballasim import _imu

    missing = [k for k in SENSOR_KEYS if k not in values]
    if missing:
        raise ExportError(f"{run_dir}: run.json has no resolved cameras.{name}.{missing[0]}")
    path = run_dir / "events" / f"{name}.h5"
    if not path.is_file():
        raise ExportError(f"{run_dir}: camera {name} has no event file events/{name}.h5")
    with h5py.File(path, "r") as f:
        _imu.check_format(str(f.attrs.get("format", "")), EVENTS_FORMAT)
        width, height = (int(v) for v in f.attrs["resolution"])
        t_offset = int(f["t_offset"][()])
    return _Camera(name, path, width, height, t_offset, {k: values[k] for k in SENSOR_KEYS})


def _event_times(t_offset_us: int, t: np.ndarray, rem: np.ndarray) -> np.ndarray:
    """t_reported (ns) from the C4 columns: 1000 (t_offset + t) + t_ns_rem."""
    return 1000 * (t_offset_us + t.astype(np.int64, copy=False)) + rem.astype(np.int64)


def _scan_events(c: _Camera) -> None:
    """The window's index range, its dropped sub-us parts (for the RAW header, final when written) and last time."""
    import h5py
    import hdf5plugin  # noqa: F401

    lo, hi = c.times["window_reported_ns"]
    with h5py.File(c.events, "r") as f:
        t, rem = f["events/t"], f["events/t_ns_rem"]
        n = t.shape[0]
        i0 = i1 = None
        rmin, rmax = 1000, -1
        for a in range(0, n, EVENT_BATCH):
            b = min(n, a + EVENT_BATCH)
            r = rem[a:b]
            tn = _event_times(c.t_offset_us, t[a:b], r)
            j0 = 0 if lo is None else int(np.searchsorted(tn, lo, "left"))
            j1 = len(tn) if hi is None else int(np.searchsorted(tn, hi, "left"))
            if j0 < j1:
                if i0 is None:
                    i0, c.first_ns = a + j0, int(tn[j0])
                i1 = a + j1
                c.last_ns = int(tn[j1 - 1])
                rmin, rmax = min(rmin, int(r[j0:j1].min())), max(rmax, int(r[j0:j1].max()))
            if j1 < len(tn):  # past the upper cut
                break
    if i0 is not None:
        c.i0, c.i1, c.rem = i0, i1, (rmin, rmax)


def _plan_imu(run_dir: Path, c: _Camera, values: dict, include: bool, warnings: list[str]) -> None:
    """The camera's IMU source (C5 file) and its window by stamp, or why it has none (§3.4 "Missing IMU")."""
    import h5py

    from ballasim import _imu

    h5 = run_dir / "imu" / f"{c.name}.h5"
    js = run_dir / "imu" / f"{c.name}.json"
    if not include:
        c.imu = {"file": None, "status": "excluded"}
        return
    state = None
    if js.is_file():
        j = json.loads(js.read_text())
        _imu.check_format(str(j.get("format", "")), IMU_RUN_FORMAT)
        state = j.get("status")
        if state == "failed":
            msg = f"camera {c.name}: the IMU simulation failed ({j.get('error')}); its stream is omitted"
            warnings.append(msg)
            log.warning("%s", msg)
            c.imu = {"file": None, "status": "failed", "error": j.get("error")}
            return
    if not h5.is_file():
        if bool(values.get("imu.enabled", False)) or state is not None:
            msg = f"camera {c.name}: the IMU is enabled but imu/{c.name}.h5 is missing; its stream is omitted"
            warnings.append(msg)
            log.warning("%s", msg)
            c.imu = {"file": None, "status": "missing"}
        else:
            c.imu = {"file": None, "status": "none"}
        return
    lo, hi = c.times["window_reported_ns"]
    with h5py.File(h5, "r") as f:
        _imu.check_format(str(f.attrs.get("format", "")))
        if int(f["t_offset"][()]) != c.t_offset_us:
            raise ExportError(f"imu/{c.name}.h5: /t_offset {int(f['t_offset'][()])} differs from the event file's "
                              f"{c.t_offset_us} (another run's IMU: rerun python -m ballasim imu)")
        t = f["t_ns"][()]
    j0 = 0 if lo is None else int(np.searchsorted(t, lo, "left"))
    j1 = len(t) if hi is None else int(np.searchsorted(t, hi, "left"))
    j1 = max(j0, j1)
    c.imu_file, c.imu_window = h5, (j0, j1)
    c.imu_first_ns = int(t[j0]) if j1 > j0 else None
    c.imu_last_ns = int(t[j1 - 1]) if j1 > j0 else None
    c.imu = {"file": f"imu/{c.name}.h5", "status": state or "complete"}


def _plan_frames(run_dir: Path, c: _Camera, values: dict, fmt: str, opts: dict, origin_us: int,
                 warnings: list[str]) -> None:
    """The camera's frame stream (C7 ``frames/<cam>.h5``): its image dataset, its times on the camera's sensor clock
    (checked against the file's own stamps) and the frames in the window, or why it has none."""
    path = run_dir / "frames" / f"{c.name}.h5"
    if not opts["frames"]:
        c.frames = {"file": None, "status": "excluded"}
        return
    if fmt not in FRAME_FORMATS:  # before the file: a format without frames never warns about a missing one
        c.frames = {"file": f"frames/{c.name}.h5" if path.is_file() else None, "status": f"not carried by {fmt}"}
        return
    if not path.is_file():
        mode = values.get("frames.mode")
        if mode not in (None, "off"):
            msg = f"camera {c.name}: frames.mode is {mode} but frames/{c.name}.h5 is missing; its frames are omitted"
            warnings.append(msg)
            log.warning("%s", msg)
            c.frames = {"file": None, "status": "missing"}
        else:
            c.frames = {"file": None, "status": "none"}
        return
    from ballasim import _imu
    from ballasim.frames import FramesFile

    try:
        f = FramesFile(path)
    except ValueError as e:
        raise ExportError(str(e)) from None
    with f:
        attrs, names = f.attrs, f.names
        if attrs.get("camera") != c.name:
            raise ExportError(f"frames/{c.name}.h5 holds camera {attrs.get('camera')!r}'s frames")
        if [int(v) for v in attrs["resolution"]] != [c.width, c.height]:
            raise ExportError(f"frames/{c.name}.h5: resolution {list(attrs['resolution'])} differs from the event "
                              f"file's [{c.width}, {c.height}]")
        if opts["frame_source"] == "auto":
            dataset = next((d for d in FRAME_SOURCES.values() if d in names), None)
        else:
            dataset = FRAME_SOURCES[opts["frame_source"]]
            if dataset not in names:
                raise ExportError(f"--frame-source {opts['frame_source']}: frames/{c.name}.h5 has no {dataset} "
                                  f"(it has {', '.join(d for d in FRAME_SOURCES.values() if d in names) or 'none'})")
        if dataset is None:
            msg = f"camera {c.name}: frames/{c.name}.h5 holds no image (aps_dn, rgb or mono); its frames are omitted"
            warnings.append(msg)
            log.warning("%s", msg)
            c.frames = {"file": f"frames/{c.name}.h5", "status": "no image"}
            return
        if fmt == "aedat4" and opts["frame_map"] == "none":
            raise ExportError(f"AEDAT4 frames are 8-bit: pass --frame-map linear8|srgb8 (camera {c.name}'s frames are "
                              f"{dataset}), or --no-frames")
        true_soe, true_eoe = f.read("t_true_soe_ns"), f.read("t_true_eoe_ns")
        stored = {k: f.read(k) for k in ("t_soe_ns", "t_eoe_ns", "t_mid_ns")}
    bits = 0
    if dataset == "aps_dn":
        bits = int((attrs.get("aps") or {}).get("bits", 16))
    row_dt = 0 if attrs.get("shutter") == "global" else int(attrs.get("row_dt_ns", 0) or 0)
    clock = {"quantum_ns": 0, "phase_ns": 0, "offset_ns": int(c.sensor["sensor.clock_offset_ns"]),
             "drift_ppm": float(c.sensor["sensor.clock_drift_ppm"])}
    es = _imu.reported_time(clock, 0, np.ascontiguousarray(true_soe, np.int64))
    ee = _imu.reported_time(clock, 0, np.ascontiguousarray(true_eoe, np.int64))
    fe = _imu.reported_time(clock, 0, np.ascontiguousarray(true_eoe + row_dt, np.int64))
    mid = (es + ee) // 2
    # The exposure ends prove the clock exactly. The stored midpoint may be floor((soe + eoe) / 2) on the sensor clock
    # (frames.md §3.6) or clock_c(t_true_mid) (PHASE3.md §5.1): with a drift the two differ by at most 1 ns, so it is
    # checked to 1 ns; the exported midpoint is always floor((soe + eoe) / 2) of the mapped ends (frames.md §3.11).
    for name, ours, tol in (("t_soe_ns", es, 0), ("t_eoe_ns", ee, 0), ("t_mid_ns", mid, 1)):
        if len(ours) != len(stored[name]) or np.any(np.abs(ours - stored[name]) > tol):
            raise ExportError(f"frames/{c.name}.h5: its {name} differ from its true times mapped by the run's sensor "
                              f"clock (offset {clock['offset_ns']} ns, drift {clock['drift_ppm']} ppm"
                              f"{', to 1 ns' if tol else ''}): a frames file of another clock")
    if np.any(np.diff(mid) < 0):
        raise ExportError(f"frames/{c.name}.h5: the frames are not in time order")
    if fmt == "aedat4" and np.any(np.diff(es) < 0):
        raise ExportError(f"frames/{c.name}.h5: the frames' exposure starts are not in time order (AEDAT4's packet "
                          f"table keys the frames by them, as dv-processing reads them)")
    lo, hi = c.times["window_reported_ns"]
    in_window = np.ones(len(mid), bool)
    if lo is not None:
        in_window &= mid >= lo
    if hi is not None:
        in_window &= mid < hi
    late = es >= 1000 * origin_us  # the frame start (= the exposure start) at or after the origin
    keep = np.flatnonzero(in_window & late)
    dropped = int(np.count_nonzero(in_window & ~late))
    if dropped:
        msg = (f"camera {c.name}: {dropped} frame(s) start before the export origin {origin_us} us and are left "
               f"out")
        warnings.append(msg)
        log.warning("%s", msg)
    k0, k1 = (int(keep[0]), int(keep[-1]) + 1) if len(keep) else (0, 0)
    roi = [int(v) for v in attrs["roi_px"]]
    c.frames_file, c.frame_dataset, c.frame_window = path, dataset, (k0, k1)
    c.frame_times = {"mid": mid, "fs": es, "fe": fe, "es": es, "ee": ee}
    c.frame_args = {"bits": bits, "x0": roi[0], "y0": roi[1], "exposure_ns": int(attrs.get("exposure_ns", -1))}
    c.frames = {"file": f"frames/{c.name}.h5", "status": "complete", "dataset": dataset, "count": k1 - k0,
                "dropped_before_origin": dropped, **({"bits": bits} if dataset == "aps_dn" else {})}


def _imu_params(c: _Camera, why: str) -> dict:
    """The resolved imu.* values a C5 file recorded (its ``params`` attribute)."""
    import h5py

    with h5py.File(c.imu_file, "r") as f:
        params = f.attrs.get("params")
    if params is None:
        raise ExportError(f"imu/{c.name}.h5 has no params attribute (contract C5), so {why}")
    if isinstance(params, bytes):
        params = params.decode()
    return json.loads(params)


def _plan_kalibr(run_dir: Path, doc: dict, bundle, cams: list[_Camera], warnings: list[str]) -> dict:
    """The bundle cameras of the cameras with an IMU (Kalibr needs both, §4.9); the others are left out with a
    warning, and a Kalibr export of no camera with an IMU is refused."""
    with_imu = [c for c in cams if c.imu_file is not None]
    for c in cams:
        if c.imu_file is None:
            msg = f"camera {c.name}: no IMU file (imu/{c.name}.h5, status {c.imu['status']}): no Kalibr files for it"
            warnings.append(msg)
            log.warning("%s", msg)
    if not with_imu:
        raise ExportError("kalibr: no selected camera has an IMU file (imu/<cam>.h5): run python -m ballasim imu")
    from ballasim.engine.scene import bundle_digest, load_bundle

    if bundle is None:
        if doc.get("inputs", {}).get("bundle", {}).get("supplied"):
            raise ExportError(f"{run_dir}: the run's bundle was supplied: pass bundle= (the Kalibr intrinsics)")
        bundle = load_bundle({k: e["value"] for k, e in doc["resolved"]["run"].items()})
    if bundle_digest(bundle) != doc["bundle"]["digest"]:
        raise ExportError(f"{run_dir}: the scene has changed since the run (bundle digest differs)")
    by_name = {cam.name: cam for cam in bundle.cameras}
    missing = [c.name for c in with_imu if c.name not in by_name]
    if missing:
        raise ExportError(f"the run's bundle has no camera {missing[0]}")
    for c in with_imu:
        c.image_roi = _frames_roi(run_dir, c.name, by_name[c.name])
    return {c.name: by_name[c.name] for c in with_imu}


def _frames_roi(run_dir: Path, name: str, cam) -> list[int] | None:
    """The ROI [x0, y0, x1, y1] of a camera's frames file (C7), whose images the ROS exports publish on its image topic
    as they are (no offset field); None without a frames file. Its resolution must be the bundle camera's."""
    path = run_dir / "frames" / f"{name}.h5"
    if not path.is_file():
        return None
    from ballasim.frames import FramesFile

    try:
        f = FramesFile(path)
    except ValueError as e:
        raise ExportError(str(e)) from None
    with f:
        attrs = f.attrs
    if [int(v) for v in attrs["resolution"]] != [int(cam.width), int(cam.height)]:
        raise ExportError(f"frames/{name}.h5: resolution {list(attrs['resolution'])} differs from the bundle "
                          f"camera's [{int(cam.width)}, {int(cam.height)}]")
    return [int(v) for v in attrs["roi_px"]]


def _check_limits(fmt: str, cams: list[_Camera], origin_us: int, epoch_ns: int) -> None:
    """The formats' limits, before anything is written (the writers refuse them too). DAT stores absolute u32 us with
    no wrap handling (§4.8): refused when the last kept time is 2^32 us or more after the origin. EVT 2.0 and 2.1 read
    their first 28-bit TIME_HIGH (64 us steps) as absolute: refused when the first kept event lies 2^34 us or more
    after the origin (later events may cross the wrap, which the readers' loop rules recover). ROS stamps (T2):
    refused, naming the first one outside the format's range (STAMP_RANGE: ROS1 [0, 2^32 s), MCAP [0, 2^31 s)).
    AEDAT4: width and height at most 32767 (int16 coordinates) and every T1b timestamp >= 0 (dv-processing's time
    ranges start at 0), which only a negative epoch can break."""
    if fmt == "aedat4":
        for c in cams:
            if c.width > AEDAT4_MAX_SIZE or c.height > AEDAT4_MAX_SIZE:
                raise ExportError(f"aedat4: camera {c.name} is {c.width} x {c.height}, beyond AEDAT4's int16 event "
                                  f"coordinates (at most 32767)")
            for what, t in (("first event", c.first_ns), ("first IMU sample", c.imu_first_ns),
                            ("first frame", _frame_time(c, "fs", 0))):
                u = None if t is None else t // 1000 - origin_us + epoch_ns // 1000
                if u is not None and u < 0:
                    raise ExportError(f"aedat4: camera {c.name}'s {what} has the timestamp {u} us (floor(t_reported / "
                                      f"1000) - origin_us + epoch_ns / 1000) < 0, which dv-processing's time ranges "
                                      f"do not take: choose --epoch-ns >= 0")
        return
    if fmt in ROS_FORMATS:
        top, text = STAMP_RANGE[fmt]
        for c in cams:
            for what, t in (("first event", c.first_ns), ("last event", c.last_ns),
                            ("first IMU sample", c.imu_first_ns), ("last IMU sample", c.imu_last_ns),
                            ("first frame", _frame_time(c, "mid", 0)), ("last frame", _frame_time(c, "mid", -1))):
                stamp = None if t is None else t - 1000 * origin_us + epoch_ns
                if stamp is not None and not 0 <= stamp < top:
                    raise ExportError(f"{fmt}: camera {c.name}'s {what} has the ROS stamp {stamp} ns (t_reported - "
                                      f"1000 origin_us + epoch_ns), outside {text}: choose --epoch-ns or --t0/--t1")
        return
    for c in cams:
        if fmt == "dat":
            for what, t in (("event", c.last_ns), ("IMU sample", c.imu_last_ns)):
                if t is not None and t // 1000 - origin_us >= 1 << 32:
                    raise ExportError(f"dat: camera {c.name}'s last {what} lies {t // 1000 - origin_us} us after the "
                                      f"origin, beyond DAT's u32 microseconds (2^32 us = 71.6 min): export a shorter "
                                      f"window with --t0/--t1")
        elif fmt in ("raw-evt2", "raw-evt21") and c.first_ns is not None and c.first_ns // 1000 - origin_us >= 1 << 34:
            raise ExportError(f"{fmt}: camera {c.name}'s first event lies {c.first_ns // 1000 - origin_us} us after "
                              f"the origin, beyond the 2^34 us (4.77 h) loop of the first TIME_HIGH: use raw-evt3, a "
                              f"later --t0 or this camera alone")


def _frame_time(c: _Camera, key: str, which: int) -> int | None:
    """The first (0) or last (-1) exported frame's time ``key`` (sensor clock, ns), or None without frames."""
    k0, k1 = c.frame_window
    if c.frames_file is None or k1 == k0:
        return None
    return int(c.frame_times[key][k0 if which == 0 else k1 - 1])


def _plan_covariance(c: _Camera, mode: str) -> None:
    """The ROS Imu covariances of a camera's IMU stream (§4.8): ``model``, the per-sample white-noise variances of its
    recorded parameters (C5 attribute ``params``; ``_imu.white_variance``), or ``zero`` (unknown)."""
    if c.imu_file is None:
        return
    if mode == "zero":
        c.imu["covariance"] = {"mode": "zero"}
        return
    from ballasim import _imu

    params = _imu_params(c, "the model's covariances are unknown: pass --imu-covariance zero")
    c.white_var = tuple(float(v) for v in _imu.white_variance(params))
    c.imu["covariance"] = {"mode": "model", "gyro": c.white_var[0], "accel": c.white_var[1]}


def _out_dir(run_dir: Path, fmt: str, out, overwrite: bool) -> Path:
    """The export directory, resolved (no symbolic link or '..' left), so the run-directory guard compares the
    directories themselves, not their spellings."""
    out_dir = (run_dir / "exports" / fmt) if out is None else Path(out).absolute()
    if out_dir.is_symlink():
        raise ExportError(f"{out_dir} is a symbolic link: not replaced")
    out_dir, run_real = out_dir.resolve(), run_dir.resolve()
    if out_dir == run_real or out_dir in run_real.parents:
        raise ExportError(f"--out {out_dir}: would replace the run directory")
    if out_dir.exists():
        if not overwrite:
            raise ExportError(f"{out_dir} exists: pass --overwrite to replace it")
        if not (out_dir / "EXPORT.json").is_file():
            raise ExportError(f"{out_dir} exists and holds no EXPORT.json: not replaced")
    return out_dir


def _stream(info: dict, camera: str, kind: str, fmt: str, path: Path) -> dict:
    s = info["streams"][0]
    s.update(camera=camera, id=f"/{camera}/{kind}")
    if s["count"] == 0:
        s["t_first"] = s["t_last"] = None
    return {"path": path.name, "format": fmt, "bytes": info["bytes"], "sha256": info["sha256"], "streams": [s]}


def _write_events(_export, fmt: str, c: _Camera, path: Path, origin_us: int, epoch_ns: int, time_high: str) -> dict:
    import h5py
    import hdf5plugin  # noqa: F401

    import ballasim

    rem_min, rem_max = c.rem
    if WRITERS[fmt] is None:
        w = _export.DatWriter(str(path), c.width, c.height, origin_us, epoch_ns, rem_min, rem_max, ballasim.__version__)
    else:
        w = _export.RawWriter(str(path), WRITERS[fmt], c.width, c.height, origin_us, epoch_ns, time_high, rem_min,
                              rem_max, ballasim.__version__)
    with h5py.File(c.events, "r") as f:
        cols = [f["events/x"], f["events/y"], f["events/p"], f["events/t"], f["events/t_ns_rem"]]
        for a in range(c.i0, c.i1, EVENT_BATCH):
            b = min(c.i1, a + EVENT_BATCH)
            x, y, p, t, rem = (col[a:b] for col in cols)
            w.append(x, y, p, _event_times(c.t_offset_us, t, rem))
    info = w.close()
    log.info("%s: %d events -> %s", c.name, info["streams"][0]["count"], path)
    return _stream(info, c.name, "events", fmt, path)


class _Rows:
    """Rows [i0, i1) of some columns of an open HDF5 file, in time order, read in blocks of ``batch`` and handed out
    up to a time bound: the containers' interleave (§4.8) reads each block once, whatever the window."""

    def __init__(self, cols: list, times, i0: int, i1: int, batch: int):
        self.cols, self.times, self.pos, self.end, self.batch = cols, times, i0, i1, batch
        self.block: list = []
        self.t = np.zeros(0, np.int64)
        self.k = 0

    def peek(self) -> int | None:
        """The time of the next row, or None when every row is handed out."""
        if self.k >= len(self.t):
            if self.pos >= self.end:
                return None
            b = min(self.end, self.pos + self.batch)
            self.block = [col[self.pos:b] for col in self.cols]
            self.t = np.asarray(self.times(self.block), np.int64)
            self.pos, self.k = b, 0
        return int(self.t[self.k])

    def take(self, hi: int) -> list | None:
        """The next rows with time < hi (the columns, then the times), or None."""
        parts = []
        while self.peek() is not None:
            j = max(self.k, int(np.searchsorted(self.t, hi, "left")))
            if j > self.k:
                parts.append([a[self.k:j] for a in self.block] + [self.t[self.k:j]])
            self.k = j
            if j < len(self.t):
                break
        if not parts:
            return None
        return [np.concatenate(cols) for cols in zip(*parts)] if len(parts) > 1 else parts[0]


def _topics(cams: list[_Camera], items: list[str]) -> dict:
    """{(stream, camera): topic}: /<cam>/<stream> unless --topic names it; a --topic of an unexported camera is
    refused (the C++ writer checks the names against ROS's rule)."""
    names = {c.name for c in cams}
    out = {(s, c.name): f"/{c.name}/{s}" for c in cams for s in TOPIC_STREAMS}
    for item in items:
        stream, _, rest = item.partition(":")
        camera, _, name = rest.partition("=")
        if camera not in names:
            raise ExportError(f"--topic {item!r}: camera {camera} is not exported ({', '.join(sorted(names))})")
        out[(stream, camera)] = name
    return out


def _interleave(cams: list[_Camera], origin_us: int, add, send) -> list[str]:
    """The containers' feed (§4.8 "Stream interleaving"): every camera's events and, when it has them, its IMU and its
    frames, each a stream of the writer (``add(kind, camera)`` returns its handle; camera by camera, events, IMU,
    frames, all before the first item), then per round every stream's next 100 ms of t_reported (frames: of their
    midpoints) in that order (``send(kind, camera, handle, columns)``: x, y, p, t_ns; t_ns, gyro, accel; or the frames'
    t_mid, t_frame_start, t_frame_end, t_exp_start, t_exp_end and pixels; contiguous), the HDF5 columns read in blocks
    of EVENT_BATCH events, IMU_BATCH samples or FRAME_BATCH_BYTES of pixels. Returns the camera of each stream, in
    order."""
    import contextlib

    import h5py
    import hdf5plugin  # noqa: F401

    streams, owner = [], []
    with contextlib.ExitStack() as stack:
        for c in cams:
            f = stack.enter_context(h5py.File(c.events, "r"))
            ev = [f["events/x"], f["events/y"], f["events/p"], f["events/t"], f["events/t_ns_rem"]]
            handle = add("events", c)
            owner.append(c.name)
            rows = _Rows(ev, lambda b, off=c.t_offset_us: _event_times(off, b[3], b[4]), c.i0, c.i1, EVENT_BATCH)
            streams.append(("events", c, handle, rows))
            if c.imu_file is not None:
                g = stack.enter_context(h5py.File(c.imu_file, "r"))
                handle = add("imu", c)
                owner.append(c.name)
                j0, j1 = c.imu_window
                streams.append(("imu", c, handle, _Rows([g["t_ns"], g["gyro"], g["accel"]], lambda b: b[0], j0, j1,
                                                        IMU_BATCH)))
            if c.frames_file is not None:
                fr = stack.enter_context(h5py.File(c.frames_file, "r"))
                px = fr[c.frame_dataset]
                handle = add("frames", c)
                owner.append(c.name)
                k0, k1 = c.frame_window
                per = max(1, int(np.prod(px.shape[1:])) * px.dtype.itemsize)
                ft = c.frame_times
                cols = [px, ft["mid"], ft["fs"], ft["fe"], ft["es"], ft["ee"]]
                streams.append(("frames", c, handle, _Rows(cols, lambda b: b[1], k0, k1,
                                                           max(1, FRAME_BATCH_BYTES // per))))
        base = 1000 * origin_us
        while True:
            times = [t for t in (r.peek() for *_, r in streams) if t is not None]
            if not times:
                break
            hi = base + ((min(times) - base) // WINDOW_NS + 1) * WINDOW_NS
            for kind, c, handle, rows in streams:
                got = rows.take(hi)
                if got is None:
                    continue
                if kind == "events":
                    cols = [np.ascontiguousarray(a) for a in (got[0], got[1], got[2], got[5])]
                elif kind == "frames":
                    cols = [np.ascontiguousarray(a, np.int64) for a in got[1:6]] + [np.ascontiguousarray(got[0])]
                else:
                    cols = [np.ascontiguousarray(a, d) for a, d in
                            ((got[3], np.int64), (got[1], np.float64), (got[2], np.float64))]
                send(kind, c, handle, cols)
    return owner


def _write_ros(_export, fmt: str, cams: list[_Camera], path: Path, origin_us: int, opts: dict) -> dict:
    """One rosbag1 bag or MCAP file of every camera's events, IMU and frames (§3.7), fed by _interleave, so chunks
    interleave the topics in time; the writer cuts the event messages on the T1a grid of --packet-us and holds each
    until it is complete, so the bytes depend on neither EVENT_BATCH nor the rounds' edges falling inside a packet;
    frames are one sensor_msgs/Image each, stamped at the exposure midpoint."""
    import ballasim

    topics = _topics(cams, opts["topics"])
    evt3 = opts["event_msg"] == "evt3"
    if fmt == "rosbag1":
        w = _export.Rosbag1Writer(str(path), opts["compression"], origin_us, opts["epoch_ns"], opts["threads"])
        connect = w.connection
    else:
        w = _export.McapWriter(str(path), opts["compression"], origin_us, opts["epoch_ns"], opts["threads"],
                               library=f"BallaSim {ballasim.__version__}")
        connect = w.channel
    packet_ns = 1000 * opts["packet_us"]

    def add(kind, c):
        if kind == "frames":
            return connect(topics[("image_raw", c.name)], "image")
        return connect(topics[(kind, c.name)], ("evt3" if evt3 else "dvs") if kind == "events" else "imu")

    def send(kind, c, conn, cols):
        if kind == "events":
            (w.event_packets_evt3 if evt3 else w.event_arrays)(conn, c.name, c.width, c.height, *cols, packet_ns)
        elif kind == "frames":
            w.images(conn, c.name, *cols, **c.frame_args, frame_map=opts["frame_map"], scale=opts["frame_scale"])
        else:
            w.imu(conn, f"{c.name}_imu", *cols, opts["imu_covariance"], c.white_var)

    owner = _interleave(cams, origin_us, add, send)
    info = w.close()
    for s, cam in zip(info["streams"], owner):
        s["camera"] = cam
        if s["count"] == 0:
            s["t_first"] = s["t_last"] = None
    log.info("%s: %s", fmt, ", ".join(f"{s['id']} {s['count']}" for s in info["streams"]))
    return {"path": path.name, "format": fmt, "bytes": info["bytes"], "sha256": info["sha256"],
            "streams": info["streams"]}


def _write_aedat4(_export, cams: list[_Camera], path: Path, origin_us: int, opts: dict) -> dict:
    """One AEDAT4 file of every camera's events, IMU and frames (§3.7): stream ids from 0 in _interleave's order
    (camera by camera, events, IMU, frames), each events or IMU stream cut by the writer into packets of --packet-us
    (10 ms) on the T1a grid and held until complete, so a stream's packets do not depend on the read blocks or the
    rounds, each frame its own FRME packet; the file's interleave of the streams' packets follows the calls, so it
    depends on the rounds (WINDOW_NS) but not on the read blocks; timestamps T1b."""
    w = _export.Aedat4Writer(str(path), opts["compression"], origin_us, opts["epoch_ns"], opts["threads"],
                             opts["packet_us"])

    def add(kind, c):
        if kind == "frames":
            return w.add_frame_stream(c.name, c.width, c.height)
        return w.add_events_stream(c.name, c.width, c.height) if kind == "events" else w.add_imu_stream(c.name)

    def send(kind, c, stream, cols):
        if kind == "frames":
            w.frames(stream, *cols, **c.frame_args, frame_map=opts["frame_map"], scale=opts["frame_scale"])
        else:
            (w.events if kind == "events" else w.imu)(stream, *cols)

    _interleave(cams, origin_us, add, send)
    info = w.close()
    for s in info["streams"]:
        if s["count"] == 0:
            s["t_first"] = s["t_last"] = None
    log.info("aedat4: %s", ", ".join(f"{s['camera']} {s['kind']} {s['count']}" for s in info["streams"]))
    return {"path": path.name, "format": "aedat4", "bytes": info["bytes"], "sha256": info["sha256"],
            "streams": info["streams"]}


def _write_kalibr(_export, cams: list[_Camera], bundle_cams: dict, out: Path, opts: dict) -> tuple[list, dict]:
    """Per camera with an IMU, imu-<cam>.yaml and camchain-imucam-<cam>.yaml (§4.9) from its C5 parameters, its
    recorded sensor.readout_delay_s and its bundle camera; the topics are the ROS exports' (--topic)."""
    import ballasim

    topics = _topics(cams, opts["topics"])
    outputs, rules = [], {}
    for c in cams:
        cam = bundle_cams.get(c.name)
        if cam is None:
            continue
        params = _imu_params(c, "its Kalibr values are unknown")
        # The camera of the image topic: a frames file's ROI crop moves the principal point by its corner and has
        # the ROI's size (distortion acts on normalized coordinates, so it is unchanged); else the full image.
        cx, cy, width, height = float(cam.cx), float(cam.cy), int(cam.width), int(cam.height)
        roi = c.image_roi
        cropped = roi is not None and roi != [0, 0, width, height]
        if cropped:
            cx, cy, width, height = cx - roi[0], cy - roi[1], roi[2] - roi[0], roi[3] - roi[1]
        res = _export.write_kalibr(str(out / f"imu-{c.name}.yaml"), str(out / f"camchain-imucam-{c.name}.yaml"),
                                   params, float(c.sensor["sensor.readout_delay_s"]), opts["kalibr_inflate"],
                                   cam.model, [float(v) for v in cam.dist], float(cam.fx), float(cam.fy), cx, cy,
                                   width, height, c.name, topics[("imu", c.name)], topics[("image_raw", c.name)],
                                   f"BallaSim {ballasim.__version__}")
        if cropped:
            res["values"]["image_roi_px"] = roi
            log.info("kalibr: %s's image topic carries the frames file's ROI %s: intrinsics and resolution of the "
                     "crop", c.name, roi)
        for info in res["files"]:
            name = Path(info["path"]).name
            outputs.append({"path": name, "format": "kalibr", "bytes": info["bytes"], "sha256": info["sha256"],
                            "streams": []})
            rules[name] = "none (parameters)"
        c.imu["kalibr"] = res["values"]
        log.info("kalibr: %s -> imu-%s.yaml, camchain-imucam-%s.yaml", c.name, c.name, c.name)
    return outputs, rules


def _write_imu_csv(_export, c: _Camera, path: Path, origin_us: int, epoch_ns: int) -> dict:
    import h5py

    j0, j1 = c.imu_window
    with h5py.File(c.imu_file, "r") as f:
        counts = [f.get(k) for k in ("counts_gyro", "counts_accel")]
        w = _export.ImuCsvWriter(str(path), origin_us, epoch_ns, counts[0] is not None, counts[1] is not None)
        for a in range(j0, j1, IMU_BATCH):
            b = min(j1, a + IMU_BATCH)
            cg, ca = (None if k is None else np.ascontiguousarray(k[a:b], np.int32) for k in counts)
            w.append(np.ascontiguousarray(f["t_ns"][a:b], np.int64), np.ascontiguousarray(f["gyro"][a:b], np.float64),
                     np.ascontiguousarray(f["accel"][a:b], np.float64), cg, ca)
    info = w.close()
    return _stream(info, c.name, "imu", "imu-csv", path)
