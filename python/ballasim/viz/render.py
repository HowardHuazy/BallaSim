"""Event images: signed event frames, polarity-count images, time surfaces,
count maps, GT depth and irradiance, and mask-contour overlays.

All functions are vectorised NumPy and return ``uint8`` RGB images of shape
(H, W, 3) at the sensor resolution. Colour ramps are interpolated in CIELAB so
lightness grows monotonically with the value; the ON (red) and OFF (blue) poles
have matched lightness, so neither polarity looks stronger than the other.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from .events import Events

THEMES = ("dark", "white")


@dataclass(frozen=True)
class Theme:
    background: tuple[int, int, int]
    on: tuple[int, int, int]
    off: tuple[int, int, int]
    ink: tuple[int, int, int]
    muted: tuple[int, int, int]
    contour: tuple[int, int, int]
    sequential: str  # matplotlib colormap for magnitudes (counts, inverse depth)


# Poles: CIELAB L* = 58.6 (ON) / 60.4 (OFF) on dark, 45.2 / 43.1 on white.
_THEMES = {
    "dark": Theme(
        background=(17, 17, 19),
        on=(255, 75, 62),
        off=(79, 143, 255),
        ink=(236, 236, 232),
        muted=(150, 149, 144),
        contour=(80, 220, 120),
        sequential="inferno",
    ),
    "white": Theme(
        background=(255, 255, 255),
        on=(205, 40, 45),
        off=(35, 95, 210),
        ink=(11, 11, 11),
        muted=(82, 81, 78),
        contour=(0, 150, 90),
        sequential="Blues",
    ),
}


def theme(name: str) -> Theme:
    try:
        return _THEMES[name]
    except KeyError:
        raise ValueError(f"theme must be one of {THEMES}, got {name!r}") from None


# Colour science -------------------------------------------------------------------

_RGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
)
_WHITE = np.array([0.95047, 1.0, 1.08883])
_EPS = (6 / 29) ** 3


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def _rgb8_to_lab(rgb) -> np.ndarray:
    xyz = _srgb_to_linear(np.asarray(rgb, float) / 255.0) @ _RGB_TO_XYZ.T / _WHITE
    f = np.where(xyz > _EPS, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _lab_to_rgb8(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE
    rgb = _linear_to_srgb(xyz @ np.linalg.inv(_RGB_TO_XYZ).T)
    return np.round(rgb * 255).astype(np.uint8)


def lab_ramp(start, stop, n: int) -> np.ndarray:
    """``n`` sRGB colours from ``start`` to ``stop``, linear in CIELAB."""
    a, b = _rgb8_to_lab(start), _rgb8_to_lab(stop)
    s = np.linspace(0.0, 1.0, n)[:, None]
    return _lab_to_rgb8(a + s * (b - a))


LUT_HALF = 255


@functools.cache
def diverging_lut(theme_name: str) -> np.ndarray:
    """(2·LUT_HALF + 1, 3) uint8: OFF pole, ..., background, ..., ON pole."""
    th = theme(theme_name)
    off = lab_ramp(th.background, th.off, LUT_HALF + 1)[::-1]
    on = lab_ramp(th.background, th.on, LUT_HALF + 1)
    return np.concatenate([off[:-1], on])


@functools.cache
def sequential_lut(theme_name: str) -> np.ndarray:
    """(256, 3) uint8 magnitude ramp whose zero end matches the theme background."""
    import matplotlib

    lut = matplotlib.colormaps[theme(theme_name).sequential](np.linspace(0, 1, 256))[:, :3]
    lut = np.round(lut * 255).astype(np.uint8)
    lut[0] = theme(theme_name).background
    return lut


def colorize_signed(values: np.ndarray, theme_name: str = "dark", gamma: float = 0.5) -> np.ndarray:
    """Map values in [-1, 1] through the diverging ramp (ON red > 0 > OFF blue).

    ``gamma`` < 1 compresses the magnitude so single events stay visible next
    to busy pixels; 1 keeps lightness linear in the value.
    """
    v = np.clip(np.asarray(values, np.float32), -1.0, 1.0)
    v = np.sign(v) * np.abs(v) ** gamma
    return diverging_lut(theme_name)[np.rint(v * LUT_HALF).astype(np.intp) + LUT_HALF]


def colorize_magnitude(values: np.ndarray, vmax: float, theme_name: str = "dark", log: bool = True) -> np.ndarray:
    """Map non-negative values to the sequential ramp; 0 is the background.

    With ``log``, the scale is log(1 + v) / log(1 + vmax). NaN maps to the background.
    """
    v = np.nan_to_num(np.asarray(values, np.float64), nan=0.0)
    v = np.clip(v, 0.0, None)
    vmax = max(float(vmax), np.finfo(float).tiny)
    s = np.log1p(v) / np.log1p(vmax) if log else v / vmax
    idx = np.rint(np.clip(s, 0.0, 1.0) * 255).astype(np.intp)
    idx[(v > 0) & (idx == 0)] = 1  # anything present is distinguishable from nothing
    return sequential_lut(theme_name)[idx]


# Accumulation ---------------------------------------------------------------------


def _inside(events: Events, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(flat pixel index, polarity, time) of the events that fall on the sensor."""
    x = events.x.astype(np.intp)
    y = events.y.astype(np.intp)
    idx = y * width + x
    inside = (x < width) & (y < height)
    if inside.all():
        return idx, events.p, events.t_ns
    return idx[inside], events.p[inside], events.t_ns[inside]


def polarity_counts(events: Events, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel ON and OFF counts, each (H, W) int64. Out-of-sensor events are ignored."""
    idx, p, _ = _inside(events, width, height)
    on = np.bincount(idx[p], minlength=width * height)
    off = np.bincount(idx[~p], minlength=width * height)
    return on.reshape(height, width), off.reshape(height, width)


def event_frame(
    on: np.ndarray, off: np.ndarray, clip: float, theme_name: str = "dark", gamma: float = 0.5
) -> np.ndarray:
    """Signed accumulation (ON - OFF) / clip through the diverging ramp."""
    return colorize_signed((on - off) / max(clip, 1e-9), theme_name, gamma)


def polarity_image(
    on: np.ndarray, off: np.ndarray, clip: float, theme_name: str = "dark", gamma: float = 0.5
) -> np.ndarray:
    """ON and OFF counts as separate colour channels (both present: magenta on dark,
    dark violet on white), mixed in linear light."""
    th = theme(theme_name)
    scale = max(clip, 1e-9)
    a_on = np.clip(on / scale, 0.0, 1.0)[..., None] ** gamma
    a_off = np.clip(off / scale, 0.0, 1.0)[..., None] ** gamma
    bg, c_on, c_off = (_srgb_to_linear(np.asarray(c, float) / 255.0) for c in (th.background, th.on, th.off))
    linear = bg + a_on * (c_on - bg) + a_off * (c_off - bg)
    return np.round(_linear_to_srgb(linear) * 255).astype(np.uint8)


def time_surface(
    events: Events, width: int, height: int, t_ref_ns: int, tau_ns: float, signed: bool = True
) -> np.ndarray:
    """Exponential time surface exp(-(t_ref - t_last)/τ) per pixel, (H, W) float32.

    With ``signed`` the value carries the polarity of the pixel's latest event
    (+ ON, - OFF). Pixels without events, or whose latest event is after
    ``t_ref_ns``, are 0.
    """
    surface = np.zeros(width * height, np.float32)
    if len(events) == 0:
        return surface.reshape(height, width)
    idx, p, t = _inside(events, width, height)
    keep = t <= t_ref_ns
    idx, p, t = idx[keep], p[keep], t[keep]
    if len(t) == 0:
        return surface.reshape(height, width)
    # Pack (time, polarity) into one key so a single maximum.at finds both.
    t_min = int(t.min())
    key = ((t - t_min) << 1) | p.astype(np.int64)
    latest = np.full(width * height, -1, np.int64)
    np.maximum.at(latest, idx, key)
    hit = latest >= 0
    age = t_ref_ns - (t_min + (latest[hit] >> 1))
    value = np.exp(-age / float(tau_ns)).astype(np.float32)
    if signed:
        value = np.where(latest[hit] & 1, value, -value)
    surface[hit] = value
    return surface.reshape(height, width)


def depth_image(depth_m: np.ndarray, theme_name: str = "dark", near_m: float | None = None) -> np.ndarray:
    """Inverse depth on the sequential ramp (near is bright on dark, dark on white).
    NaN or non-positive depth is drawn as background."""
    inv = np.where(np.isfinite(depth_m) & (depth_m > 0), 1.0 / np.where(depth_m > 0, depth_m, 1.0), 0.0)
    vmax = 1.0 / near_m if near_m else (np.percentile(inv[inv > 0], 99.5) if np.any(inv > 0) else 1.0)
    return colorize_magnitude(inv, vmax, theme_name, log=False)


def irradiance_image(irradiance: np.ndarray, low: float, high: float) -> np.ndarray:
    """Linear irradiance as grey on a log scale from ``low`` (black) to ``high`` (white),
    like a camera frame (the same on every theme). Values at or below ``low`` are black."""
    if not 0 < low < high:
        raise ValueError(f"need 0 < low < high, got {low}, {high}")
    e = np.asarray(irradiance, np.float64)
    level = np.log(np.maximum(e, low) / low) / np.log(high / low)
    grey = np.rint(np.clip(level, 0.0, 1.0) * 255).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=-1)


# Masks ----------------------------------------------------------------------------


def mask_boundary(labels: np.ndarray, mode: str = "valid") -> np.ndarray:
    """Boundary pixels of a GT label image, (H, W) bool.

    ``"valid"`` outlines the region with any label (modelled geometry, where
    sim and real are comparable); ``"labels"`` also outlines every boundary
    between two labels.
    """
    if mode not in ("valid", "labels"):
        raise ValueError(f"mask mode must be 'valid' or 'labels', got {mode!r}")
    region = labels > 0 if mode == "valid" else labels
    edge = np.zeros(labels.shape, bool)
    dx = region[:, 1:] != region[:, :-1]
    dy = region[1:, :] != region[:-1, :]
    edge[:, 1:] |= dx
    edge[:, :-1] |= dx
    edge[1:, :] |= dy
    edge[:-1, :] |= dy
    if mode == "valid":
        edge &= labels > 0  # draw on the inside of the valid region
    return edge


def overlay_boundary(image: np.ndarray, boundary: np.ndarray, color) -> np.ndarray:
    out = image.copy()
    out[boundary] = color
    return out
