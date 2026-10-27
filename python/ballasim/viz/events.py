"""Event batches and time-indexed event sources.

Every source exposes the same view of its stream: events sorted by time, time as
int64 nanoseconds on the source's clock plus a per-stream offset, and polarity as
a bool that is True for ON. Format-specific quirks (float32 seconds, µs plus a ns
remainder, inverted polarity) stay inside the concrete sources.
"""

from __future__ import annotations

import abc
import copy
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal

import numpy as np

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000


def seconds_to_ns(seconds: float | Decimal) -> int:
    """Nearest integer ns; pass a Decimal (e.g. parsed from text) to stay exact at epoch-sized times."""
    return round(seconds * NS_PER_S)


@dataclass(frozen=True, eq=False)
class Events:
    """A time-sorted batch of events. ``p`` is True for ON."""

    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    t_ns: np.ndarray
    extras: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.t_ns.shape[0])

    def __getitem__(self, index: slice) -> Events:
        if not isinstance(index, slice):
            raise TypeError("Events supports slicing only")
        return Events(
            self.x[index], self.y[index], self.p[index], self.t_ns[index], {k: v[index] for k, v in self.extras.items()}
        )


@dataclass(frozen=True, eq=False)
class GtFrame:
    """A dense ground-truth image (mask labels, depth or irradiance) valid at ``t_ns``."""

    t_ns: int
    data: np.ndarray


class EventSource(abc.ABC):
    """Random access by time into a time-sorted event stream.

    Subclasses implement the native accessors (``_search``, ``_read`` and
    optionally ``_gt_frame``) on their own clock and polarity convention. This
    base class applies the stream's clock offsets and polarity inversion, so a
    real stream and a simulated one can be put on a common clock, and a stream
    can be compared against a time-shifted copy of itself.

    Events and GT frames have separate offsets. The GT offset follows the event
    offset unless set with :meth:`with_gt_offset`, which is what a time-shifted
    copy of a stream needs. A recording whose GT is on another clock than its
    events needs both: EVIMO2 GT ticks are on the pose clock, and the events
    are off from it by the camera's event-to-pose offset (DESIGN §5.5).
    """

    def __init__(self, label: str, width: int, height: int) -> None:
        self.label = label
        self.width = int(width)
        self.height = int(height)
        self.offset_ns = 0
        self.invert_polarity = False
        self._gt_offset_ns: int | None = None

    # Native accessors -------------------------------------------------------------

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def _search(self, t_ns: int) -> int:
        """Index of the first event whose native time is >= ``t_ns``."""

    @abc.abstractmethod
    def _read(self, i0: int, i1: int, extras: Sequence[str]) -> Events:
        """Events ``[i0, i1)`` on the native clock; ``p`` is True where raw p == 1."""

    def _gt_frame(self, kind: str, t_ns: int) -> GtFrame | None:
        return None

    @property
    def extra_fields(self) -> tuple[str, ...]:
        """Names of optional per-event columns (e.g. ``"instance"`` from ``/gt``)."""
        return ()

    @property
    def gt_kinds(self) -> tuple[str, ...]:
        """Dense GT kinds this source can provide: a subset of ("mask", "depth", "irradiance")."""
        return ()

    # Derived copies ---------------------------------------------------------------

    def with_offset(self, offset_ns: int) -> EventSource:
        """A view whose timestamps are shifted by ``offset_ns`` (GT frames too, unless
        their offset was set with :meth:`with_gt_offset`)."""
        view = copy.copy(self)
        view.offset_ns = int(offset_ns)
        return view

    def with_gt_offset(self, gt_offset_ns: int) -> EventSource:
        """A view whose GT frame times are shifted by ``gt_offset_ns``, independently of the events."""
        view = copy.copy(self)
        view._gt_offset_ns = int(gt_offset_ns)
        return view

    def with_polarity(self, inverted: bool) -> EventSource:
        view = copy.copy(self)
        view.invert_polarity = bool(inverted)
        return view

    def with_label(self, label: str) -> EventSource:
        view = copy.copy(self)
        view.label = label
        return view

    # Public accessors on the common clock -----------------------------------------

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def gt_offset_ns(self) -> int:
        """Shift of the GT frame times: the event offset unless set separately."""
        return self.offset_ns if self._gt_offset_ns is None else self._gt_offset_ns

    def search(self, t_ns: int) -> int:
        """Index of the first event with time >= ``t_ns``."""
        return self._search(int(t_ns) - self.offset_ns)

    def read(self, i0: int, i1: int, extras: Sequence[str] = ()) -> Events:
        n = len(self)
        i0 = min(max(int(i0), 0), n)
        i1 = min(max(int(i1), i0), n)
        unknown = set(extras) - set(self.extra_fields)
        if unknown:
            raise KeyError(f"{self.label}: no per-event field(s) {sorted(unknown)}")
        events = self._read(i0, i1, tuple(extras))
        if self.offset_ns or self.invert_polarity:
            events = replace(
                events,
                t_ns=events.t_ns + self.offset_ns if self.offset_ns else events.t_ns,
                p=~events.p if self.invert_polarity else events.p,
            )
        return events

    def window(self, t0_ns: int, t1_ns: int, extras: Sequence[str] = ()) -> Events:
        """Events with ``t0_ns <= t < t1_ns``."""
        return self.read(self.search(t0_ns), self.search(t1_ns), extras)

    def count(self, t0_ns: int, t1_ns: int) -> int:
        return self.search(t1_ns) - self.search(t0_ns)

    def iter_chunks(
        self, t0_ns: int, t1_ns: int, max_events: int = 1 << 22, extras: Sequence[str] = ()
    ) -> Iterator[Events]:
        """Events in ``[t0_ns, t1_ns)`` in batches of at most ``max_events``."""
        i0, i1 = self.search(t0_ns), self.search(t1_ns)
        for start in range(i0, i1, max_events):
            yield self.read(start, min(start + max_events, i1), extras)

    @property
    def t_start_ns(self) -> int:
        """Time of the first event."""
        if len(self) == 0:
            raise ValueError(f"{self.label}: empty event stream")
        return int(self.read(0, 1).t_ns[0])

    @property
    def t_end_ns(self) -> int:
        """One nanosecond past the last event, so ``[t_start_ns, t_end_ns)`` holds every event."""
        n = len(self)
        if n == 0:
            raise ValueError(f"{self.label}: empty event stream")
        return int(self.read(n - 1, n).t_ns[0]) + 1

    def gt_frame(self, kind: str, t_ns: int) -> GtFrame | None:
        """The GT frame of ``kind`` nearest to ``t_ns``, or None if there is none close by."""
        if kind not in self.gt_kinds:
            return None
        offset = self.gt_offset_ns
        frame = self._gt_frame(kind, int(t_ns) - offset)
        if frame is None or not offset:
            return frame
        return GtFrame(frame.t_ns + offset, frame.data)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(label={self.label!r}, {self.width}x{self.height}, n={len(self)}, "
            f"offset_ns={self.offset_ns}, gt_offset_ns={self.gt_offset_ns}, invert_polarity={self.invert_polarity})"
        )


class ArraySource(EventSource):
    """Events held in (possibly memory-mapped) NumPy arrays, already in ns."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        t_ns: np.ndarray,
        *,
        label: str = "events",
        width: int | None = None,
        height: int | None = None,
        extras: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        n = len(t_ns)
        extras = dict(extras or {})
        for name, column in {"x": x, "y": y, "p": p, **extras}.items():
            if len(column) != n:
                raise ValueError(f"column {name!r} has {len(column)} entries, expected {n}")
        if t_ns.dtype != np.int64:
            raise TypeError(f"t_ns must be int64 ns, got {t_ns.dtype}")
        if n > 1 and not np.all(t_ns[1:] >= t_ns[:-1]):
            raise ValueError("events must be sorted by time")
        if width is None:
            width = int(x.max()) + 1 if n else 1
        if height is None:
            height = int(y.max()) + 1 if n else 1
        super().__init__(label, width, height)
        self._x, self._y, self._t = x, y, t_ns
        self._p = p if p.dtype == bool else np.asarray(p) > 0
        self._extras = extras

    def __len__(self) -> int:
        return len(self._t)

    @property
    def extra_fields(self) -> tuple[str, ...]:
        return tuple(self._extras)

    def _search(self, t_ns: int) -> int:
        return int(np.searchsorted(self._t, np.int64(t_ns), side="left"))

    def _read(self, i0: int, i1: int, extras: Sequence[str]) -> Events:
        return Events(
            x=np.asarray(self._x[i0:i1]),
            y=np.asarray(self._y[i0:i1]),
            p=np.asarray(self._p[i0:i1]),
            t_ns=np.asarray(self._t[i0:i1]),
            extras={name: np.asarray(self._extras[name][i0:i1]) for name in extras},
        )


def common_time_range(sources: Sequence[EventSource]) -> tuple[int, int]:
    """The interval ``[t0, t1)`` covered by every source."""
    t0 = max(s.t_start_ns for s in sources)
    t1 = min(s.t_end_ns for s in sources)
    if t1 <= t0:
        raise ValueError("the event streams do not overlap in time: " + ", ".join(repr(s) for s in sources))
    return t0, t1
