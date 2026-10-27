"""The ``imu`` and ``export`` subcommands of ``python -m ballasim`` (docs/dev/IMU.md, docs/dev/EXPORT.md;
docs/design/phase3/outputs.md §3.4), registered by :mod:`ballasim.engine.cli`. The options are final but for A9's
--frame-source. ``export`` writes RAW, DAT, rosbag1, MCAP, AEDAT4 and Kalibr (:func:`ballasim.export.formats.export`);
``imu`` simulates the IMU of a finished run (:func:`ballasim.export.imu.simulate_imu`, P3-A A3)."""

from __future__ import annotations

import argparse

from ballasim.engine.cli import seconds_to_ns

from .formats import COMPRESSIONS, FORMATS


def add_parsers(sub) -> None:
    """Adds ``imu`` and ``export`` to the subparsers of ``python -m ballasim``."""
    p_imu = sub.add_parser("imu", help="simulate the IMU of a finished run's cameras (docs/dev/IMU.md)")
    p_imu.add_argument("run_dir", help="run directory (holds run.json)")
    p_imu.add_argument("--camera", action="append", default=None, metavar="NAME",
                       help="a camera (repeatable; default: every camera whose imu.enabled is true)")
    p_imu.add_argument("--profile", action="append", default=[], metavar="ID",
                       help="a root profile over the run's recorded values; it may set imu.* keys only (any other "
                            "key is refused; repeatable)")
    p_imu.add_argument("--camera-profile", action="append", default=[], metavar="CAMERA=ID",
                       help="a camera-mounted profile for one camera, e.g. "
                            "left_camera=evimo2/prophesee_gen3_imu_left@v1 (repeatable)")
    p_imu.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="an imu.* value, full key (cameras.<cam>.imu.* or cameras.default.imu.*; repeatable)")
    p_imu.add_argument("--overwrite", action="store_true", help="replace the cameras' existing IMU files")

    p_ex = sub.add_parser("export", help="export a finished run to a standard format (docs/dev/EXPORT.md)")
    p_ex.add_argument("run_dir", help="run directory (holds run.json)")
    p_ex.add_argument("--format", required=True, choices=FORMATS, help="the output format")
    p_ex.add_argument("--camera", action="append", default=None, metavar="NAME",
                      help="a camera (repeatable; default: the run's)")
    p_ex.add_argument("--out", default=None, metavar="DIR", help="output directory (default: <run>/exports/<format>)")
    p_ex.add_argument("--overwrite", action="store_true", help="replace an existing export directory")
    p_ex.add_argument("--epoch-ns", type=int, default=None, metavar="N",
                      help="added to the stamps; a multiple of 1000 (default: 0, ROS1 and MCAP 1 s)")
    p_ex.add_argument("--t0", type=seconds_to_ns, default=None, metavar="S",
                      help="lower cut (s, bundle clock; default: none)")
    p_ex.add_argument("--t1", type=seconds_to_ns, default=None, metavar="S",
                      help="upper cut (s, bundle clock; default: none, the whole event file)")
    p_ex.add_argument("--no-imu", action="store_true", help="leave out the IMU streams")
    p_ex.add_argument("--no-frames", action="store_true", help="leave out the frame streams")
    p_ex.add_argument("--compression", choices=COMPRESSIONS, default=None,
                      help="container compression (default per format: aedat4 zstd, rosbag1 bz2, mcap zstd)")
    p_ex.add_argument("--threads", type=int, default=4, metavar="N",
                      help="compression threads, 1-256 (default: 4); the bytes do not depend on them")
    p_ex.add_argument("--event-msg", choices=("dvs", "evt3"), default=None,
                      help="rosbag1 event message: dvs_msgs/EventArray (default) or event_camera_msgs evt3 (mcap: "
                           "evt3 only)")
    p_ex.add_argument("--packet-us", type=int, default=None, metavar="N",
                      help="ROS event message or AEDAT4 packet length in us, 1 to 2^24 (default: ROS 1000, "
                           "aedat4 10000)")
    p_ex.add_argument("--topic", action="append", default=[], metavar="STREAM:CAMERA=NAME",
                      help="rosbag1, mcap and kalibr: a topic name, STREAM events, imu or image_raw (repeatable; "
                           "default /<cam>/events, /<cam>/imu, /<cam>/image_raw)")
    p_ex.add_argument("--imu-covariance", choices=("model", "zero"), default="model",
                      help="ROS Imu covariances: the model's white variance (default) or zero (unknown)")
    p_ex.add_argument("--evt2-time-high", choices=("openeb", "change"), default="openeb",
                      help="raw-evt2 TIME_HIGH words: every 16 us as OpenEB writes them (default) or on change with an "
                           "8.192 ms keepalive (raw-evt21 is always on change)")
    p_ex.add_argument("--frame-map", choices=("none", "linear8", "srgb8"), default="none",
                      help="8-bit map of linear frames (AEDAT4 frames are 8-bit)")
    p_ex.add_argument("--frame-scale", type=float, default=1.0, metavar="F", help="scale before the frame map")
    p_ex.add_argument("--frame-source", choices=("auto", "aps", "rgb", "mono"), default="auto",
                      help="the frames file's image: aps (aps_dn), rgb or mono (default: the first of these present)")
    p_ex.add_argument("--kalibr-inflate", type=float, default=1.0, metavar="F",
                      help="factor on the Kalibr noise densities (default: 1)")


def command(args: argparse.Namespace) -> int:
    """Runs ``imu`` or ``export``; refusals raise ExportError."""
    if args.command == "imu":
        from .imu import parse_camera_profiles, simulate_imu

        paths = simulate_imu(args.run_dir, cameras=args.camera, sets=args.set, profiles=args.profile,
                             camera_profiles=parse_camera_profiles(args.camera_profile), overwrite=args.overwrite)
        print("\n".join(map(str, paths)))
        return 0
    from .formats import export

    out = export(args.run_dir, args.format, out=args.out, cameras=args.camera, epoch_ns=args.epoch_ns, t0_ns=args.t0,
                 t1_ns=args.t1, overwrite=args.overwrite, threads=args.threads, compression=args.compression,
                 imu=not args.no_imu, frames=not args.no_frames, event_msg=args.event_msg,
                 packet_us=args.packet_us, topics=args.topic,
                 imu_covariance=args.imu_covariance, evt2_time_high=args.evt2_time_high, frame_map=args.frame_map,
                 frame_scale=args.frame_scale, frame_source=args.frame_source, kalibr_inflate=args.kalibr_inflate)
    print(out)
    return 0
