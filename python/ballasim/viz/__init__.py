"""BallaSim visualization toolkit: compare simulated and real event streams by eye.

Load streams with :func:`open_events` (EVIMO2 NPZ sequences, BallaSim/DSEC HDF5
files, NumPy arrays), render aligned windows with :class:`Renderer`, and write
comparison videos (:func:`write_comparison`), per-panel PNGs
(:func:`write_frames`), a static HTML viewer (:func:`write_html`) or statistics
plots (:func:`compute_stats`, :func:`write_stats`). The command line is
``python -m ballasim.viz``; see docs/dev/VIEWER.md.
"""

from .compare import Renderer, RenderOptions, Timeline, write_comparison, write_frames
from .events import ArraySource, Events, EventSource, GtFrame, common_time_range
from .html import write_html
from .io import Evimo2Sequence, H5Events, arrays_to_source, load_arrays, open_events
from .stats import GtRegion, StatsOptions, compute_stats, write_stats

__all__ = [
    "ArraySource",
    "EventSource",
    "Events",
    "Evimo2Sequence",
    "GtFrame",
    "GtRegion",
    "H5Events",
    "RenderOptions",
    "Renderer",
    "StatsOptions",
    "Timeline",
    "arrays_to_source",
    "common_time_range",
    "compute_stats",
    "load_arrays",
    "open_events",
    "write_comparison",
    "write_frames",
    "write_html",
    "write_stats",
]
