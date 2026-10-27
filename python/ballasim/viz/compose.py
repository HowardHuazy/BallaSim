"""Panel layout: a labelled header above each image, common scaling across
sensor resolutions, and side-by-side or stacked tiling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .render import theme

LAYOUTS = ("h", "v")
_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class Panel:
    """One image with its header text. Header text must be ASCII (OpenCV Hershey fonts)."""

    image: np.ndarray
    title: str
    info: str = ""


def resize_to(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize with nearest neighbour when enlarging (event pixels stay crisp)
    and area averaging when shrinking."""
    h, w = image.shape[:2]
    if (w, h) == (width, height):
        return image
    interpolation = cv2.INTER_NEAREST if width >= w and height >= h else cv2.INTER_AREA
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _fit_text(text: str, max_width: int, font_scale: float, thickness: int) -> str:
    if cv2.getTextSize(text, _FONT, font_scale, thickness)[0][0] <= max_width:
        return text
    while text and cv2.getTextSize(text + "...", _FONT, font_scale, thickness)[0][0] > max_width:
        text = text[:-1]
    return text + "..."


def _header(width: int, s: float, title: str, info: str, theme_name: str) -> np.ndarray:
    """A two-line header (title in ink, info in muted ink); ``s`` is the text scale."""
    th = theme(theme_name)
    header = np.empty((round(46 * s), width, 3), np.uint8)
    header[:] = th.background
    margin = round(8 * s)
    thickness = 2 if s >= 1.5 else 1
    lines = [(title, 0.55 * s, th.ink, 19 * s), (info, 0.45 * s, th.muted, 38 * s)]
    for text, font_scale, color, baseline in lines:
        if text:
            text = _fit_text(text, width - 2 * margin, font_scale, thickness)
            cv2.putText(header, text, (margin, round(baseline)), _FONT, font_scale, color, thickness, cv2.LINE_AA)
    return header


def compose(panels: Sequence[Panel], layout: str = "h", theme_name: str = "dark", scale: float = 1.0) -> np.ndarray:
    """Tile panels with headers into one RGB frame with even width and height.

    Side by side (``"h"``) brings every image to the tallest image's height
    times ``scale``; stacked (``"v"``) brings them to the widest image's width.
    Aspect ratios are kept, so different sensor resolutions line up.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if not panels:
        raise ValueError("no panels to compose")
    th = theme(theme_name)
    if layout == "h":
        target = round(max(p.image.shape[0] for p in panels) * scale)
        sizes = [(round(p.image.shape[1] * target / p.image.shape[0]), target) for p in panels]
        text_scale = target / 480.0
    else:
        target = round(max(p.image.shape[1] for p in panels) * scale)
        sizes = [(target, round(p.image.shape[0] * target / p.image.shape[1])) for p in panels]
        text_scale = target / 640.0
    text_scale = float(np.clip(text_scale, 0.6, 3.0))
    tiles = [
        np.concatenate([_header(size[0], text_scale, p.title, p.info, theme_name), resize_to(p.image, *size)], axis=0)
        for p, size in zip(panels, sizes, strict=True)
    ]

    gap = max(2, round(2 * scale))
    bar_shape = (tiles[0].shape[0], gap, 3) if layout == "h" else (gap, tiles[0].shape[1], 3)
    bar = np.empty(bar_shape, np.uint8)
    bar[:] = tuple(round((b + m) / 2) for b, m in zip(th.background, th.muted, strict=True))
    parts = [tiles[0]]
    for tile in tiles[1:]:
        parts += [bar, tile]
    frame = np.concatenate(parts, axis=1 if layout == "h" else 0)
    pad_h, pad_w = frame.shape[0] % 2, frame.shape[1] % 2
    if pad_h or pad_w:
        frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return frame
