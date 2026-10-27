"""Time-aligned rendering of event streams into labelled panels and comparison frames.

A :class:`Renderer` turns window ``k`` of a :class:`Timeline` into one panel per
stream (the first stream is the reference, e.g. the real recording), plus optional
extra panels: the difference of the first two streams, their activity ratio, GT
depth, or the irradiance a simulation saw (its dense frames). GT mask contours are
drawn where available (see :class:`Renderer`), on event-free pixels only, so they
never hide an event.
"""

from __future__ import annotations

import os
import re
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .compose import Panel, compose
from .events import NS_PER_MS, NS_PER_S, EventSource
from .render import (
    THEMES,
    colorize_magnitude,
    colorize_signed,
    depth_image,
    event_frame,
    irradiance_image,
    mask_boundary,
    overlay_boundary,
    polarity_counts,
    polarity_image,
    theme,
    time_surface,
)
from .video import open_sink, write_png

KINDS = ("events", "polarity", "timesurface", "counts")
THIRD_PANELS = ("none", "diff", "ratio", "depth", "irradiance")
MASK_MODES = ("valid", "labels", "none")
# A time surface needs history: events older than 5 τ contribute < 0.7 %.
_TS_HISTORY_TAUS = 5
_RATIO_RANGE_OCTAVES = 2.0
# Ratio panels are quantised to 1/8 octave: near-equal activity becomes exactly
# background, the bands read like contours, and the images compress losslessly.
_RATIO_STEP_OCTAVES = 0.125


def default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def ordered_map[T, R](fn: Callable[[T], R], items: Iterable[T], workers: int | None = None) -> Iterator[R]:
    """``map(fn, items)`` on a thread pool, yielding results in order with a
    bounded number in flight. Rendering and image encoding mostly run in NumPy
    and OpenCV with the GIL released, so threads scale."""
    workers = default_workers() if workers is None else workers
    if workers <= 1:
        yield from map(fn, items)
        return
    with ThreadPoolExecutor(workers) as pool:
        pending: deque = deque()
        for item in items:
            pending.append(pool.submit(fn, item))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


@dataclass(frozen=True)
class Timeline:
    """Windows ``[starts_ns[k], starts_ns[k] + window_ns)``, or ``[starts_ns[k], ends_ns[k])`` when the windows are
    given one by one (:meth:`from_windows`; ``window_ns`` is then the longest)."""

    starts_ns: np.ndarray
    window_ns: int
    ends_ns: np.ndarray | None = None

    @classmethod
    def from_range(cls, t0_ns: int, t1_ns: int, dt_ns: int, window_ns: int | None = None) -> Timeline:
        """Windows every ``dt_ns`` that fit entirely inside ``[t0_ns, t1_ns)``;
        ``window_ns`` defaults to ``dt_ns`` (back-to-back windows)."""
        window_ns = dt_ns if window_ns is None else window_ns
        if dt_ns <= 0 or window_ns <= 0:
            raise ValueError("dt and window must be positive")
        if t1_ns - window_ns < t0_ns:
            raise ValueError(f"time range [{t0_ns}, {t1_ns}) ns is shorter than one {window_ns} ns window")
        return cls(np.arange(t0_ns, t1_ns - window_ns + 1, dt_ns, dtype=np.int64), int(window_ns))

    @classmethod
    def from_windows(cls, starts_ns: Sequence[int], ends_ns: Sequence[int]) -> Timeline:
        """Windows ``[starts_ns[k], ends_ns[k])``, e.g. a Blender job's preview frames; empty windows are allowed."""
        starts, ends = np.asarray(starts_ns, np.int64), np.asarray(ends_ns, np.int64)
        if starts.ndim != 1 or starts.shape != ends.shape or not len(starts) or np.any(ends < starts):
            raise ValueError("windows need equal, non-empty lists of starts and ends with start <= end")
        return cls(starts, int((ends - starts).max()), ends)

    def __len__(self) -> int:
        return len(self.starts_ns)

    def window(self, k: int) -> tuple[int, int]:
        t0 = int(self.starts_ns[k])
        return t0, (t0 + self.window_ns if self.ends_ns is None else int(self.ends_ns[k]))


@dataclass(frozen=True)
class RenderOptions:
    """How each window is drawn.

    ``clip`` is the per-pixel count that saturates the colour ramp; None picks it
    once from the reference stream (99th percentile over active pixels in a
    sample of windows) and uses it for every stream and frame, so the panels
    stay comparable. ``third`` names the extra panels after the streams, one or
    several of :data:`THIRD_PANELS` separated by commas (e.g. ``"irradiance,depth"``).
    ``sigma_px`` blurs a difference or ratio panel (default 0 for ``"diff"``, 2 px
    for ``"ratio"``).
    """

    kind: str = "events"
    theme: str = "dark"
    clip: float | None = None
    gamma: float = 0.5
    tau_ms: float = 20.0
    mask: str = "valid"
    third: str = "none"
    sigma_px: float | None = None

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("kind", self.kind, KINDS),
            ("theme", self.theme, THEMES),
            ("mask", self.mask, MASK_MODES),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        extras = [part.strip() for part in self.third.split(",")]
        if self.third != "none" and (any(e not in THIRD_PANELS[1:] for e in extras) or len(set(extras)) != len(extras)):
            raise ValueError(f"third must be 'none' or distinct values of {THIRD_PANELS[1:]}, got {self.third!r}")
        if self.clip is not None and self.clip <= 0:
            raise ValueError("clip must be positive")

    @property
    def extra_panels(self) -> tuple[str, ...]:
        """The extra panels in order (empty for ``"none"``)."""
        return () if self.third == "none" else tuple(part.strip() for part in self.third.split(","))

    def sigma_for(self, panel: str) -> float:
        """Blur of a difference or ratio panel."""
        if self.sigma_px is not None:
            return self.sigma_px
        return 2.0 if panel == "ratio" else 0.0


@dataclass(frozen=True)
class RenderedFrame:
    t0_ns: int
    t1_ns: int
    panels: list[Panel]
    counts: list[int]  # events per stream in the window


def auto_clip(source: EventSource, timeline: Timeline, kind: str, samples: int = 16) -> float:
    """99th percentile of per-pixel activity over active pixels in up to
    ``samples`` evenly spaced windows: |ON - OFF| for ``"events"``, ON + OFF otherwise."""
    picks = np.unique(np.linspace(0, len(timeline) - 1, min(samples, len(timeline))).astype(int))
    values = []
    for k in picks:
        on, off = polarity_counts(source.window(*timeline.window(k)), source.width, source.height)
        activity = np.abs(on - off) if kind == "events" else on + off
        values.append(activity[activity > 0])
    pooled = np.concatenate(values)
    return max(1.0, float(np.percentile(pooled, 99))) if pooled.size else 1.0


def auto_depth_near(source: EventSource, timeline: Timeline, samples: int = 16) -> float | None:
    """Near plane (m) of the depth colour scale, fixed for the whole timeline: the
    99.5th percentile of inverse depth over up to ``samples`` GT frames at evenly
    spaced window centres. None if no frame is found."""
    picks = np.unique(np.linspace(0, len(timeline) - 1, min(samples, len(timeline))).astype(int))
    values = []
    for k in picks:
        t0, t1 = timeline.window(k)
        frame = source.gt_frame("depth", (t0 + t1) // 2)
        if frame is not None:
            depth = frame.data[np.isfinite(frame.data) & (frame.data > 0)]
            values.append(1.0 / depth)
    pooled = np.concatenate(values) if values else np.empty(0)
    return 1.0 / float(np.percentile(pooled, 99.5)) if pooled.size else None


def auto_irradiance_range(source: EventSource, timeline: Timeline, samples: int = 16) -> tuple[float, float] | None:
    """Grey scale of irradiance panels, fixed for the whole timeline: the 0.5th and
    99.5th percentiles of positive irradiance over up to ``samples`` frames at evenly
    spaced window centres. None if no frame is found."""
    picks = np.unique(np.linspace(0, len(timeline) - 1, min(samples, len(timeline))).astype(int))
    values = []
    for k in picks:
        t0, t1 = timeline.window(k)
        frame = source.gt_frame("irradiance", (t0 + t1) // 2)
        if frame is not None:
            values.append(frame.data[frame.data > 0])
    pooled = np.concatenate(values) if values else np.empty(0)
    if not pooled.size:
        return None
    low, high = (float(v) for v in np.percentile(pooled, [0.5, 99.5]))
    return (low, high) if high > low else (low, low * 2.0)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "stream"


def _blur(image: np.ndarray, sigma_px: float) -> np.ndarray:
    image = image.astype(np.float32)
    return cv2.GaussianBlur(image, (0, 0), sigma_px) if sigma_px > 0 else image


class Renderer:
    """Renders aligned windows of ``sources``; ``sources[0]`` is the reference.

    GT mask contours: a stream with its own GT masks (an EVIMO2 sequence, a
    time-shifted view of one, or a simulation with dense frames) is outlined with
    its own; every other panel (a simulation without them, the extra panels)
    borrows the masks of the first stream that has them, if its resolution matches. Contours are drawn only on pixels that show
    no events (on event panels and difference/ratio panels), so they never hide
    an event; the panel info gives the GT frame time.

    ``roles`` are short names ("real", "sim") used in panel titles; without
    them the titles are the source labels.
    """

    def __init__(
        self,
        sources: Sequence[EventSource],
        timeline: Timeline,
        options: RenderOptions | None = None,
        roles: Sequence[str] | None = None,
    ) -> None:
        options = options or RenderOptions()
        if not sources:
            raise ValueError("no event sources")
        if roles is not None and len(roles) != len(sources):
            raise ValueError("need one role per source")
        self.sources = list(sources)
        self.timeline = timeline
        self.options = options
        self.roles = list(roles) if roles is not None else [s.label for s in sources]
        extras = options.extra_panels
        for panel in ("diff", "ratio"):
            if panel not in extras:
                continue
            if len(sources) != 2:
                raise ValueError(f"a {panel!r} panel needs exactly two streams")
            if sources[0].size != sources[1].size:
                raise ValueError(
                    f"a {panel!r} panel needs equal resolutions, got "
                    f"{sources[0].width}x{sources[0].height} and {sources[1].width}x{sources[1].height}"
                )
        self._depth_source = None
        self.depth_near_m: float | None = None
        if "depth" in extras:
            self._depth_source = next((s for s in sources if "depth" in s.gt_kinds), None)
            if self._depth_source is None:
                raise ValueError("a 'depth' panel needs a stream with GT depth (an EVIMO2 sequence or dense frames)")
            self.depth_near_m = auto_depth_near(self._depth_source, timeline)
        self._irradiance_source = None
        self.irradiance_range: tuple[float, float] | None = None
        if "irradiance" in extras:
            self._irradiance_source = next((s for s in sources if "irradiance" in s.gt_kinds), None)
            if self._irradiance_source is None:
                raise ValueError(
                    "an 'irradiance' panel needs a simulation with dense frames (python -m ballasim dense)"
                )
            self.irradiance_range = auto_irradiance_range(self._irradiance_source, timeline)
        n_panels = len(sources) + len(extras)
        self._mask_sources: list[EventSource | None] = [None] * n_panels
        if options.mask != "none":
            reference = next((s for s in sources if "mask" in s.gt_kinds), None)
            own = [s if "mask" in s.gt_kinds else reference for s in sources]
            self._mask_sources = own + [reference] * (n_panels - len(sources))
        self.clip = options.clip if options.clip is not None else auto_clip(sources[0], timeline, options.kind)

    def __len__(self) -> int:
        return len(self.timeline)

    def _title(self, i: int) -> str:
        role, label = self.roles[i], self.sources[i].label
        return label if role == label else f"{role}: {label}"

    def _stream_image(self, source: EventSource, t0: int, t1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(image, on, off) for one stream and window."""
        o = self.options
        if o.kind == "timesurface":
            tau_ns = o.tau_ms * NS_PER_MS
            events = source.window(t1 - int(_TS_HISTORY_TAUS * tau_ns), t1)
            surface = time_surface(events, source.width, source.height, t1, tau_ns)
            recent = events[int(np.searchsorted(events.t_ns, t0)) :]
            on, off = polarity_counts(recent, source.width, source.height)
            return colorize_signed(surface, o.theme, gamma=1.0), on, off
        on, off = polarity_counts(source.window(t0, t1), source.width, source.height)
        if o.kind == "events":
            image = event_frame(on, off, self.clip, o.theme, o.gamma)
        elif o.kind == "polarity":
            image = polarity_image(on, off, self.clip, o.theme, o.gamma)
        else:
            image = colorize_magnitude(on + off, self.clip, o.theme)
        return image, on, off

    def _gt_panel(self, kind: str, t0: int, t1: int) -> Panel:
        """A depth or irradiance panel from its source's GT frame nearest to the window centre."""
        o = self.options
        source = self._depth_source if kind == "depth" else self._irradiance_source
        frame = source.gt_frame(kind, (t0 + t1) // 2)
        role = self.roles[self.sources.index(source)]
        title = f"GT depth ({role})" if kind == "depth" else f"irradiance ({role})"
        if frame is None:
            blank = np.empty((source.height, source.width, 3), np.uint8)
            blank[:] = theme(o.theme).background
            return Panel(blank, title, "no GT frame near this time")
        if kind == "depth":
            info = f"inverse depth at t = {frame.t_ns / NS_PER_S:.4f} s"
            if self.depth_near_m is not None:
                info += f", scale to {self.depth_near_m:.2f} m"
            return Panel(depth_image(frame.data, o.theme, self.depth_near_m), title, info)
        low, high = self.irradiance_range or (1e-6, 1.0)
        info = f"log scale {low:.3g} to {high:.3g} at t = {frame.t_ns / NS_PER_S:.4f} s, waveforms not applied"
        return Panel(irradiance_image(frame.data, low, high), title, info)

    def _third_panel(self, panel: str, counts: list[tuple[np.ndarray, np.ndarray]], t0: int, t1: int) -> Panel:
        o = self.options
        if panel in ("depth", "irradiance"):
            return self._gt_panel(panel, t0, t1)
        sigma = o.sigma_for(panel)
        blur_note = f", sigma {sigma:g} px" if sigma > 0 else ""
        ref, test = self.roles[0], self.roles[-1]
        if panel == "diff":
            (on_r, off_r), (on_t, off_t) = counts
            diff = _blur((on_t - off_t) - (on_r - off_r), sigma)
            image = colorize_signed(diff / self.clip, o.theme, o.gamma)
            return Panel(image, f"{test} - {ref}", f"net ON-OFF count, clip {self.clip:g}{blur_note}")
        if panel == "ratio":
            (on_r, off_r), (on_t, off_t) = counts
            a_ref, a_test = _blur(on_r + off_r, sigma), _blur(on_t + off_t, sigma)
            # Regularise by half the mean activity so empty regions read as "equal".
            eps = 0.25 * float(a_ref.mean() + a_test.mean())
            ratio = np.log2((a_test + eps) / (a_ref + eps)) if eps > 0 else np.zeros_like(a_ref)
            ratio = np.round(ratio / _RATIO_STEP_OCTAVES) * _RATIO_STEP_OCTAVES
            image = colorize_signed(ratio / _RATIO_RANGE_OCTAVES, o.theme, gamma=1.0)
            info = f"log2 activity ratio, +-{_RATIO_RANGE_OCTAVES:g} (red: {test} busier){blur_note}"
            return Panel(image, f"{test} / {ref}", info)
        raise ValueError(f"unknown extra panel {panel!r}")

    def render(self, k: int) -> RenderedFrame:
        t0, t1 = self.timeline.window(k)
        o = self.options
        panels, counts, n_events = [], [], []
        win_ms = (t1 - t0) / NS_PER_MS
        for i, source in enumerate(self.sources):
            image, on, off = self._stream_image(source, t0, t1)
            n = int(on.sum() + off.sum())
            rate = n / (win_ms * 1e3) if win_ms > 0 else 0.0
            info = f"t = {t0 / NS_PER_S:.4f} s  +{win_ms:g} ms  {n / 1e3:.1f}k ev  {rate:.2f} Mev/s"
            if o.kind == "timesurface":
                info += f"  tau {o.tau_ms:g} ms"
            panels.append(Panel(image, self._title(i), info))
            counts.append((on, off))
            n_events.append(n)
        for panel in o.extra_panels:
            panels.append(self._third_panel(panel, counts, t0, t1))
        # Event and difference panels keep every coloured pixel; depth and irradiance panels have nothing to hide.
        protect = [True] * len(self.sources) + [panel in ("diff", "ratio") for panel in o.extra_panels]
        th = theme(o.theme)
        contours: dict[int, tuple[np.ndarray, int] | None] = {}
        for i, (panel, mask_source) in enumerate(zip(panels, self._mask_sources, strict=True)):
            if mask_source is None or panel.image.shape[:2] != (mask_source.height, mask_source.width):
                continue
            if id(mask_source) not in contours:
                frame = mask_source.gt_frame("mask", (t0 + t1) // 2)
                contours[id(mask_source)] = None if frame is None else (mask_boundary(frame.data, o.mask), frame.t_ns)
            contour = contours[id(mask_source)]
            if contour is None:
                panels[i] = Panel(panel.image, panel.title, panel.info + "  no GT mask")
                continue
            edge, gt_t_ns = contour
            if protect[i]:
                edge = edge & np.all(panel.image == th.background, axis=-1)
            image = overlay_boundary(panel.image, edge, th.contour)
            panels[i] = Panel(image, panel.title, f"{panel.info}  GT {gt_t_ns / NS_PER_S:.4f} s")
        return RenderedFrame(t0, t1, panels, n_events)


def write_comparison(
    renderer: Renderer,
    outputs: Sequence[str | Path],
    *,
    fps: float = 25.0,
    layout: str = "h",
    scale: float = 1.0,
    strip_frames: int = 8,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Render every window once and write the composite frames to each output
    (video, GIF, PNG strip or PNG directory; see :func:`~ballasim.viz.video.open_sink`)."""
    n = len(renderer)
    sinks = [open_sink(path, fps=fps, n_frames=n, strip_frames=strip_frames) for path in outputs]

    def composite(k: int) -> np.ndarray:
        return compose(renderer.render(k).panels, layout, renderer.options.theme, scale)

    try:
        for done, frame in enumerate(ordered_map(composite, range(n), workers), start=1):
            for sink in sinks:
                sink.write(frame)
            if progress is not None:
                progress(done, n)
    finally:
        for sink in sinks:
            sink.close()


def write_frames(
    renderer: Renderer,
    out_dir: str | Path,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    *,
    flat: bool = False,
    file_names: Sequence[str] | None = None,
) -> list[Path]:
    """Write each panel of every window as a PNG at sensor resolution, without
    headers, to ``out_dir/<panel>/<k>_t<seconds>.png``. Returns the panel directories.
    ``flat`` writes the one panel of a single-stream renderer to ``out_dir`` itself;
    ``file_names`` names the windows' files instead (one distinct name per window)."""
    out_dir = Path(out_dir)
    names = [_slug(role) for role in renderer.roles] + list(renderer.options.extra_panels)
    if len(set(names)) != len(names):
        names = [f"{i}_{name}" for i, name in enumerate(names)]
    if flat and len(names) != 1:
        raise ValueError(f"flat output needs one panel (one stream, no extra panels), got {len(names)}")
    n = len(renderer)
    if file_names is not None and (len(file_names) != n or len(set(file_names)) != n
                                   or any(Path(f).name != f for f in file_names)):
        raise ValueError(f"need {n} distinct file names without directories, one per window")
    dirs = [out_dir] if flat else [out_dir / name for name in names]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    def save(k: int) -> None:
        frame = renderer.render(k)
        name = file_names[k] if file_names is not None else f"{k:06d}_t{frame.t0_ns / NS_PER_S:.6f}.png"
        for d, panel in zip(dirs, frame.panels, strict=True):
            write_png(d / name, panel.image)

    for done, _ in enumerate(ordered_map(save, range(n), workers), start=1):
        if progress is not None:
            progress(done, n)
    return dirs
