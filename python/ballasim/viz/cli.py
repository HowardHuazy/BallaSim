"""Command line: ``python -m ballasim.viz {info,frames,compare,stats,html} ...``.

Every command takes a reference stream (``--real``) and/or a second stream
(``--sim``): an EVIMO2 sequence directory, a BallaSim/DSEC ``.h5`` file, a
``.npz``/``.npy`` file or a directory of ``.npy`` columns. Times (``--t0``,
``--t1``) are seconds on the streams' common clock after their offsets; by
default the range covered by every stream. Times and offsets are parsed as
decimals, so they stay exact to the nanosecond even at UNIX-epoch magnitudes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

import numpy as np

from .compare import KINDS, MASK_MODES, THIRD_PANELS, Renderer, RenderOptions, Timeline, write_comparison, write_frames
from .compose import LAYOUTS
from .events import NS_PER_MS, NS_PER_S, EventSource, common_time_range, seconds_to_ns
from .html import IMAGE_FORMATS, write_html
from .io import POLARITY_MODES, Evimo2Sequence, open_events
from .render import THEMES
from .stats import STATS_MASK_MODES, GtRegion, StatsOptions, compute_stats, light_frequencies, write_stats


def _decimal(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not value.is_finite():
        raise argparse.ArgumentTypeError(f"not a finite number: {text!r}")
    return value


def _add_sources(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("event streams")
    for role, what in (
        ("real", "reference stream, e.g. a real recording"),
        ("sim", "stream to compare, e.g. a simulation"),
    ):
        group.add_argument(f"--{role}", metavar="PATH", help=f"{what}: EVIMO2 sequence dir, .h5, .npz/.npy or .npy dir")
        group.add_argument(f"--{role}-label", metavar="TEXT", help="label shown in titles (default: from the source)")
        group.add_argument(
            f"--{role}-offset-ms",
            type=_decimal,
            default=Decimal(0),
            metavar="MS",
            help="added to this stream's timestamps",
        )
        group.add_argument(
            f"--{role}-gt-offset-ms",
            type=_decimal,
            metavar="MS",
            help=f"added to this stream's GT frame times (default: --{role}-offset-ms)",
        )
        group.add_argument(f"--{role}-polarity", choices=POLARITY_MODES, default="auto", help="default: %(default)s")
    group.add_argument("--t0", type=_decimal, metavar="S", help="start time in seconds (default: common start)")
    group.add_argument("--t1", type=_decimal, metavar="S", help="end time in seconds (default: common end)")


def _add_render(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("rendering")
    group.add_argument("--dt-ms", type=float, default=10.0, help="frame step (default: %(default)s ms)")
    group.add_argument("--win-ms", type=float, help="accumulation window (default: --dt-ms)")
    group.add_argument("--kind", choices=KINDS, default="events", help="panel image (default: %(default)s)")
    group.add_argument("--theme", choices=THEMES, default="dark", help="background (default: %(default)s)")
    group.add_argument("--clip", type=float, help="count that saturates the colours (default: auto from --real)")
    group.add_argument("--gamma", type=float, default=0.5, help="colour-ramp exponent (default: %(default)s)")
    group.add_argument("--tau-ms", type=float, default=20.0, help="time-surface decay (default: %(default)s ms)")
    group.add_argument("--mask", choices=MASK_MODES, default="valid", help="GT mask contour (default: %(default)s)")
    group.add_argument(
        "--third",
        default="none",
        metavar="PANEL[,PANEL]",
        help=f"extra panels, comma-separated: {', '.join(THIRD_PANELS[1:])} (default: %(default)s)",
    )
    group.add_argument("--sigma-px", type=float, help="blur of diff/ratio panels (default: 0 diff, 2 ratio)")
    group.add_argument("--workers", type=int, help="render threads (default: up to 8)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ballasim.viz",
        description="Compare simulated and real event streams visually.",
        epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="describe the event streams")
    _add_sources(info)

    frames = commands.add_parser("frames", help="write each panel of every window as a PNG")
    _add_sources(frames)
    _add_render(frames)
    frames.add_argument("--out", required=True, metavar="DIR")
    frames.add_argument(
        "--times-ns",
        metavar="FILE",
        help="frames from a ballasim.preview-times JSON file (a Blender job's preview frames: t_ns and the prev and "
        "exposure windows in ns) instead of --t0/--t1/--dt-ms/--win-ms",
    )
    frames.add_argument("--win", choices=PREVIEW_WINDOWS, default="prev", help="the --times-ns window (default: prev)")
    frames.add_argument("--flat", action="store_true", help="write the one panel's PNGs into --out itself")
    frames.add_argument(
        "--name",
        metavar="PATTERN",
        help="file name pattern, e.g. events_{index:05d}.png; fields index, t0_ns, t1_ns, and with --times-ns "
        "frame and t_ns (default: <index>_t<seconds>.png)",
    )

    compare = commands.add_parser("compare", help="side-by-side comparison video, GIF or PNG strip")
    _add_sources(compare)
    _add_render(compare)
    compare.add_argument(
        "--out", required=True, nargs="+", metavar="PATH", help=".mp4/.mkv/.mov, .gif, .png (strip) or a directory"
    )
    compare.add_argument("--layout", choices=LAYOUTS, default="h", help="h: side by side, v: stacked")
    compare.add_argument("--scale", type=float, default=1.0, help="image scale; 2 keeps chroma of single pixels")
    compare.add_argument("--fps", type=float, default=25.0, help="playback rate (default: %(default)s)")
    compare.add_argument("--strip-frames", type=int, default=8, help="frames in a .png strip (default: %(default)s)")

    html = commands.add_parser("html", help="self-contained HTML viewer with a time slider")
    _add_sources(html)
    _add_render(html)
    html.add_argument("--out", required=True, metavar="FILE.html")
    html.add_argument("--fps", type=float, default=25.0, help="initial playback rate (default: %(default)s)")
    html.add_argument("--format", choices=tuple(IMAGE_FORMATS), default="webp", help="frame images (lossless)")
    html.add_argument("--frames-dir", action="store_true", help="write images to <name>_frames/ instead of embedding")
    html.add_argument("--title", help="page title")

    stats = commands.add_parser("stats", help="rate, ON fraction, spectrum, interval and per-pixel plots")
    _add_sources(stats)
    stats.add_argument("--out", required=True, metavar="DIR")
    stats.add_argument("--rate-bin-ms", type=float, default=1.0)
    stats.add_argument("--on-fraction-bin-ms", type=float, default=10.0)
    stats.add_argument("--spectrum-bin-us", type=float, default=100.0)
    stats.add_argument("--segment-s", type=float, default=1.0, help="spectrum segment (frequency resolution 1/S)")
    stats.add_argument("--fmin", type=float, default=20.0, help="low edge of the spectral peak search (Hz)")
    stats.add_argument("--fmax", type=float, default=1000.0, help="spectrum plot and peak search limit (Hz)")
    stats.add_argument("--max-interval-events", type=int, default=10_000_000)
    stats.add_argument(
        "--mask",
        choices=STATS_MASK_MODES,
        default="none",
        help="valid: only events inside the GT validity mask of the first stream with masks (default: %(default)s)",
    )
    return parser


def _open_sources(args: argparse.Namespace) -> tuple[list[EventSource], list[str]]:
    sources, roles = [], []
    for role in ("real", "sim"):
        path = getattr(args, role)
        if path is None:
            continue
        polarity = getattr(args, f"{role}_polarity")
        source = open_events(path, label=getattr(args, f"{role}_label"), polarity=polarity)
        if polarity == "auto" and source.invert_polarity:
            _log(f"{source.label}: EVIMO2 Samsung recording, polarity inverted (override with --{role}-polarity)")
        offset_ms = getattr(args, f"{role}_offset_ms")
        if offset_ms:
            source = source.with_offset(round(offset_ms * NS_PER_MS))
        gt_offset_ms = getattr(args, f"{role}_gt_offset_ms")
        if gt_offset_ms is not None:
            source = source.with_gt_offset(round(gt_offset_ms * NS_PER_MS))
        sources.append(source)
        roles.append(role)
    if not sources:
        raise ValueError("give --real and/or --sim")
    return sources, roles


def _extent(source: EventSource) -> str:
    if len(source) == 0:
        return f"{source.label} (no events)"
    return f"{source.label} [{source.t_start_ns / NS_PER_S:.6f}, {source.t_end_ns / NS_PER_S:.6f}) s"


def _time_range(args: argparse.Namespace, sources: Sequence[EventSource]) -> tuple[int, int]:
    """The requested range; fails if no stream has events in it and warns about each empty stream,
    so a unit typo or a stream on another clock does not render plausible-looking blank output."""
    common = common_time_range(sources) if args.t0 is None or args.t1 is None else (0, 0)
    t0 = common[0] if args.t0 is None else seconds_to_ns(args.t0)
    t1 = common[1] if args.t1 is None else seconds_to_ns(args.t1)
    if t1 <= t0:
        raise ValueError(f"empty time range [{t0 / NS_PER_S}, {t1 / NS_PER_S}) s")
    empty = [s for s in sources if s.count(t0, t1) == 0]
    span = f"[{t0 / NS_PER_S:.6f}, {t1 / NS_PER_S:.6f}) s"
    if len(empty) == len(sources):
        raise ValueError(f"no events in {span}; the streams cover " + ", ".join(map(_extent, sources)))
    for s in empty:
        _log(f"warning: no events of {_extent(s)} in {span}")
    return t0, t1


def _render_options(args: argparse.Namespace) -> RenderOptions:
    return RenderOptions(
        kind=args.kind,
        theme=args.theme,
        clip=args.clip,
        gamma=args.gamma,
        tau_ms=args.tau_ms,
        mask=args.mask,
        third=args.third,
        sigma_px=args.sigma_px,
    )


def _renderer(args: argparse.Namespace) -> Renderer:
    sources, roles = _open_sources(args)
    t0, t1 = _time_range(args, sources)
    dt_ns = round(args.dt_ms * NS_PER_MS)
    win_ns = round(args.win_ms * NS_PER_MS) if args.win_ms else dt_ns
    renderer = Renderer(sources, Timeline.from_range(t0, t1, dt_ns, win_ns), _render_options(args), roles)
    _log(f"{len(renderer)} frames from {t0 / NS_PER_S:.4f} s, colour clip {renderer.clip:g} events")
    return renderer


#: The windows a ballasim.preview-times file gives per frame (docs/design/phase3/blender.md §3.3).
PREVIEW_WINDOWS = ("prev", "exposure")


def _times_renderer(args: argparse.Namespace) -> tuple[Renderer, list[dict]]:
    """The renderer of a --times-ns file's frames (``{"frames": [{index, frame, t_ns, prev, exposure}]}``) and the
    frames' name fields. Windows may be empty (a span's first frame); no stream needs events in them."""
    if args.t0 is not None or args.t1 is not None:
        raise ValueError("--times-ns gives the frames; drop --t0/--t1")
    with open(args.times_ns) as f:
        doc = json.load(f)
    frames = doc.get("frames") if isinstance(doc, dict) else None
    if not isinstance(frames, list) or not frames or doc.get("format") != "ballasim.preview-times":
        raise ValueError(f"{args.times_ns}: not a ballasim.preview-times file with frames")
    windows = [f.get(args.win) if isinstance(f, dict) else None for f in frames]
    if any(not isinstance(w, list) or len(w) != 2 for w in windows):
        raise ValueError(f"{args.times_ns}: a frame without a {args.win!r} window")
    sources, roles = _open_sources(args)
    renderer = Renderer(sources, Timeline.from_windows(*zip(*windows, strict=True)), _render_options(args), roles)
    # A frame's own index names its file (a Blender preview's f - f0, also with a step > 1); else its position.
    fields = [{"index": f.get("index", k), "frame": f["frame"], "t_ns": f["t_ns"], "t0_ns": w[0], "t1_ns": w[1]}
              for k, (f, w) in enumerate(zip(frames, windows, strict=True))]
    _log(f"{len(renderer)} frames from {args.times_ns} ({args.win} windows), colour clip {renderer.clip:g} events")
    return renderer, fields


def _write_frames(args: argparse.Namespace) -> list:
    if args.times_ns:
        renderer, fields = _times_renderer(args)
    else:
        renderer = _renderer(args)
        fields = [dict(zip(("t0_ns", "t1_ns"), renderer.timeline.window(k), strict=True), index=k)
                  for k in range(len(renderer))]
    names = None
    if args.name:
        try:
            names = [args.name.format(**f) for f in fields]
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"--name {args.name!r}: {e!r}") from None
    return write_frames(renderer, args.out, args.workers, _progress("frames"), flat=args.flat, file_names=names)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _progress(name: str):
    def report(done: int, total: int) -> None:
        if done == total or done % 25 == 0:
            print(f"\r{name}: {done}/{total} frames", end="\n" if done == total else "", file=sys.stderr, flush=True)

    return report


def describe(source: EventSource) -> dict:
    n = len(source)
    info = {
        "label": source.label,
        "type": type(source).__name__,
        "events": n,
        "resolution": [source.width, source.height],
        "offset_ns": source.offset_ns,
        "gt_offset_ns": source.gt_offset_ns,
        "invert_polarity": source.invert_polarity,
        "extra_fields": list(source.extra_fields),
        "gt": list(source.gt_kinds),
    }
    if n:
        t0, t1 = source.t_start_ns, source.t_end_ns
        on = sum(int(chunk.p.sum()) for chunk in source.iter_chunks(t0, t1))
        info.update(
            t_start_s=t0 / NS_PER_S,
            t_end_s=t1 / NS_PER_S,
            mean_rate_mev_s=n / ((t1 - t0) / NS_PER_S) / 1e6,
            on_fraction=on / n,
        )
    if isinstance(source, Evimo2Sequence):
        info["camera"] = source.camera
    return info


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "info":
            sources, _ = _open_sources(args)
            print(json.dumps([describe(s) for s in sources], indent=2, default=lambda v: np.asarray(v).tolist()))
        elif args.command == "frames":
            _log("wrote " + ", ".join(map(str, _write_frames(args))))
        elif args.command == "compare":
            write_comparison(
                _renderer(args),
                args.out,
                fps=args.fps,
                layout=args.layout,
                scale=args.scale,
                strip_frames=args.strip_frames,
                workers=args.workers,
                progress=_progress("compare"),
            )
            _log("wrote " + ", ".join(args.out))
        elif args.command == "html":
            renderer = _renderer(args)
            title = args.title or " vs ".join(
                f"{role}: {s.label}" for role, s in zip(renderer.roles, renderer.sources, strict=True)
            )
            path = write_html(
                renderer,
                args.out,
                title=title,
                fps=args.fps,
                image_format=args.format,
                embed=not args.frames_dir,
                workers=args.workers,
                progress=_progress("html"),
            )
            _log(f"wrote {path}")
        elif args.command == "stats":
            sources, _ = _open_sources(args)
            t0, t1 = _time_range(args, sources)
            options = StatsOptions(
                rate_bin_ms=args.rate_bin_ms,
                on_fraction_bin_ms=args.on_fraction_bin_ms,
                spectrum_bin_us=args.spectrum_bin_us,
                spectrum_segment_s=args.segment_s,
                peak_band_hz=(args.fmin, args.fmax),
                max_interval_events=args.max_interval_events,
            )
            region = None
            if args.mask == "valid":
                reference = next((s for s in sources if "mask" in s.gt_kinds), None)
                if reference is None:
                    raise ValueError("--mask valid needs a stream with GT masks (an EVIMO2 sequence)")
                region = GtRegion(reference)
            # Every stream reports the vector strength at the lights of all streams (D16b), so they compare.
            lights: dict[float, str] = {}
            for s in sources:
                for f, name in light_frequencies(s).items():
                    lights.setdefault(f, name)
            if lights:
                _log("lights: " + ", ".join(f"{name} {f:g} Hz" for f, name in lights.items()))
            stats = [compute_stats(s, t0, t1, options, region, lights) for s in sources]
            paths = write_stats(stats, args.out, args.fmax)
            _log("wrote " + ", ".join(map(str, paths)))
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0
