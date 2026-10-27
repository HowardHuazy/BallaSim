"""The asynchronous engine (docs/dev/ENGINE.md): scene bundle (C1) + renderer (C2) + sensor (C3) -> event files
(C4), run by the C++/CUDA engine ``ballasim._engine`` (docs/dev/ENGINE_CPP.md).

- :func:`run` (:mod:`.engine`): runs a run configuration (``ballasim.config.load``: format 2, resolved in C++,
  docs/dev/P0_PARAMS.md) and returns a :class:`RunSummary`;
- ``python -m ballasim run <config.yaml>``, ``python -m ballasim evimo2 <sequence> ...``, ``params``, ``resolve``
  and ``dense`` (:mod:`.cli`).
"""

__all__ = ["WINDOW_NS", "CameraSummary", "RunSummary", "run"]


def __getattr__(name: str):
    """The names of :mod:`.engine`, imported on first use (PEP 562), so that ``ballasim job`` starts without the
    engine, the renderer and the scene side (its status.json within about 0.1 s)."""
    if name in __all__:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
