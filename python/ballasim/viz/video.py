"""Frame sinks: H.264 MP4, GIF, a PNG strip of selected frames, or a PNG sequence.

``open_sink`` picks the sink from the output path. Every sink takes RGB uint8
frames of a constant size through ``write`` and finishes in ``close``.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov")


class FrameSink(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...


def ffmpeg_executable() -> str | None:
    """The imageio-ffmpeg binary if installed, else ``ffmpeg`` on PATH."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return shutil.which("ffmpeg")


def write_png(path: Path, frame: np.ndarray) -> None:
    if not cv2.imwrite(str(path), np.ascontiguousarray(frame[..., ::-1])):
        raise OSError(f"cannot write {path}")


class _SizedSink:
    """Checks that every frame has the size of the first one."""

    def __init__(self) -> None:
        self._shape: tuple[int, ...] | None = None

    def _check(self, frame: np.ndarray) -> bool:
        """True on the first frame."""
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected an RGB uint8 frame, got {frame.dtype} {frame.shape}")
        if self._shape is None:
            self._shape = frame.shape
            return True
        if frame.shape != self._shape:
            raise ValueError(f"frame size changed from {self._shape} to {frame.shape}")
        return False


class VideoWriter(_SizedSink):
    """H.264 (yuv420p, +faststart) through an ffmpeg pipe.

    Falls back to OpenCV's MPEG-4 Part 2 encoder with a warning when no ffmpeg
    binary is available. Frame width and height must be even.
    """

    def __init__(self, path: str | Path, fps: float = 25.0, crf: int = 18, preset: str = "medium") -> None:
        super().__init__()
        self.path = Path(path)
        self.fps, self.crf, self.preset = fps, crf, preset
        self._proc: subprocess.Popen | None = None
        self._cv: cv2.VideoWriter | None = None

    def _open(self, width: int, height: int) -> None:
        if width % 2 or height % 2:
            raise ValueError(f"H.264 needs even frame dimensions, got {width}x{height}")
        exe = ffmpeg_executable()
        if exe is not None:
            cmd = [
                exe, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{self.fps:g}", "-i", "-",
                "-an", "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
            ]  # fmt: skip
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            return
        warnings.warn("no ffmpeg found (install imageio-ffmpeg); writing MPEG-4 Part 2 with OpenCV", stacklevel=3)
        self._cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        if not self._cv.isOpened():
            raise OSError(f"OpenCV cannot open a video writer for {self.path}")

    def write(self, frame: np.ndarray) -> None:
        if self._check(frame):
            self._open(frame.shape[1], frame.shape[0])
        if self._proc is not None:
            try:
                self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except BrokenPipeError:
                self._fail()
        else:
            self._cv.write(np.ascontiguousarray(frame[..., ::-1]))

    def _fail(self) -> None:
        proc, self._proc = self._proc, None
        with contextlib.suppress(BrokenPipeError):
            proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace").strip()
        proc.wait()
        raise RuntimeError(f"ffmpeg failed writing {self.path}: {err}")

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                self._fail()
            if self._proc.wait() != 0:
                self._fail()
            self._proc.stderr.close()
            self._proc = None
        if self._cv is not None:
            self._cv.release()
            self._cv = None


class GifWriter(_SizedSink):
    """Animated GIF (adaptive palette, no dithering); frames are held until ``close``,
    so keep GIFs to short excerpts."""

    # Browsers replace GIF frame delays below 20 ms with 100 ms.
    MIN_DELAY_MS = 20

    def __init__(self, path: str | Path, fps: float = 25.0) -> None:
        super().__init__()
        self.path = Path(path)
        self.delay_ms = max(self.MIN_DELAY_MS, round(1000.0 / fps))
        self._frames: list = []

    def write(self, frame: np.ndarray) -> None:
        from PIL import Image

        self._check(frame)
        image = Image.fromarray(frame).quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        self._frames.append(image)

    def close(self) -> None:
        if self._frames:
            first, *rest = self._frames
            first.save(self.path, save_all=True, append_images=rest, duration=self.delay_ms, loop=0)
        self._frames = []


class StripWriter(_SizedSink):
    """One PNG holding ``n_select`` frames evenly spaced over ``n_total``, stacked
    top to bottom for wide frames and left to right for tall ones."""

    def __init__(self, path: str | Path, n_total: int, n_select: int = 8) -> None:
        super().__init__()
        self.path = Path(path)
        self._selected = set(np.unique(np.rint(np.linspace(0, n_total - 1, min(n_select, n_total))).astype(int)))
        self._index = 0
        self._frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self._check(frame)
        if self._index in self._selected:
            self._frames.append(frame)
        self._index += 1

    def close(self) -> None:
        if self._frames:
            wide = self._frames[0].shape[1] >= self._frames[0].shape[0]
            write_png(self.path, np.concatenate(self._frames, axis=0 if wide else 1))
        self._frames = []


class PngSequenceWriter(_SizedSink):
    """``frame_000000.png``, ``frame_000001.png``, ... in a directory."""

    def __init__(self, directory: str | Path) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._index = 0

    def write(self, frame: np.ndarray) -> None:
        self._check(frame)
        write_png(self.directory / f"frame_{self._index:06d}.png", frame)
        self._index += 1

    def close(self) -> None:
        pass


def open_sink(path: str | Path, *, fps: float, n_frames: int, strip_frames: int = 8) -> FrameSink:
    """A sink chosen by suffix: video (.mp4/.mkv/.mov), .gif, .png (strip), or a
    directory (no suffix) for a PNG sequence."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in VIDEO_SUFFIXES:
        return VideoWriter(path, fps=fps)
    if suffix == ".gif":
        return GifWriter(path, fps=fps)
    if suffix == ".png":
        return StripWriter(path, n_frames, strip_frames)
    if not suffix:
        return PngSequenceWriter(path)
    raise ValueError(f"unsupported output {path}: use .mp4/.mkv/.mov, .gif, .png (strip) or a directory")
