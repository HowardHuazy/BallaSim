"""RGB frames on the event clock (docs/design/phase3/frames.md): BallaSim's exposure reference and, with Blender,
matched and full frames. The numerics are C++ (``ballasim._frames``: exact waveform means, sensor-clock stamps,
frame schedules, the exposure reference, the APS model, the frames values of a run's cameras); this package holds the
thin Python side: the frames file (:mod:`ballasim.frames.io`, ``ballasim.frames/1``), a run's frames values
(:mod:`ballasim.frames.resolve`), ``frames reference`` (:mod:`ballasim.frames.reference`) and ``frames aps``
(:mod:`ballasim.frames.aps`)."""

from .io import FORMAT, FramesFile, read_frames, rewrite_frames, write_frames

__all__ = ["FORMAT", "FramesFile", "read_frames", "rewrite_frames", "write_frames"]
