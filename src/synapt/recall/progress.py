"""Time-based progress lines for the long loops of ``synapt recall build``.

A build that goes quiet for an hour cannot be told from one that hung. ``ProgressLog`` makes a loop report ``n/N``, a
recent rate and an ETA at most once per ``PROGRESS_INTERVAL_S`` seconds, and ``phase`` brackets a step that is one
opaque statement (a bulk SQL rebuild) with a start line and an end line, since such a step has no loop to tick in.

The cadence is TIME, not a count of items: the cost of one item in the clustering loop grows with the number of
clusters already built, so a line every N items is dense early and sparse late, which is the failure this exists to end.
A line is sent at the first tick at or after the interval, so the longest silence is the interval plus one item.

The ETA uses the rate since the previous line, not the lifetime average: the greedy clustering loop slows as it runs,
so the average makes the estimate too short. It still assumes the current rate holds, and says so by being an estimate.

Lines go through ``logger.info``, the same channel as the ``FTS5 index: n/N chunks`` lines, in the same shape.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)

# Read when a ProgressLog is made, so a test (or an operator) can change it for one build.
PROGRESS_INTERVAL_S = 30.0

_clock: Callable[[], float] = time.monotonic
_emit: Callable[[str], None] = logger.info


def _rate_text(rate: float) -> str:
    return f"{rate:.0f}/s" if rate >= 10 else f"{rate:.1f}/s"


class ProgressLog:
    """``tick`` once per item (or with the running count); a line comes at most once per interval."""

    def __init__(
        self,
        label: str,
        total: int,
        unit: str = "items",
        *,
        interval: Optional[float] = None,
        clock: Optional[Callable[[], float]] = None,
        emit: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.label = label
        self.total = total
        self.unit = unit
        self.interval = PROGRESS_INTERVAL_S if interval is None else interval
        self._own_clock = clock
        self._own_emit = emit
        self.count = 0
        self._start = self._now()
        self._last_t = self._start
        self._last_n = 0

    def _now(self) -> float:
        return (self._own_clock or _clock)()

    def _send(self, text: str) -> None:
        (self._own_emit or _emit)(text)

    def tick(self, n: Optional[int] = None) -> None:
        """Record progress (``n`` items done so far, or one more than before) and report if the interval has passed."""
        self.count = self.count + 1 if n is None else n
        now = self._now()
        elapsed = now - self._last_t
        if elapsed < self.interval:
            return
        rate = (self.count - self._last_n) / elapsed if elapsed > 0 else 0.0
        remaining = max(self.total - self.count, 0)
        eta = f"{remaining / rate:.0f}s remaining" if rate > 0 else "ETA unknown"
        self._send(f"{self.label}: {self.count}/{self.total} {self.unit} ({_rate_text(rate)}, {eta})")
        self._last_t = now
        self._last_n = self.count

    def done(self) -> None:
        """The closing line, always sent, so a loop that finished inside one interval still leaves a trace."""
        elapsed = self._now() - self._start
        self._send(f"{self.label}: {self.count}/{self.total} {self.unit} in {elapsed:.1f}s")


def note(text: str, *, emit: Optional[Callable[[str], None]] = None) -> None:
    """One line, for a step worth naming that is neither a loop nor worth bracketing (a size, a decision)."""
    (emit or _emit)(text)


@contextmanager
def phase(
    label: str,
    *,
    clock: Optional[Callable[[], float]] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> Iterator[None]:
    """Bracket one step that cannot tick (a single bulk statement): a start line, then an end line even on failure."""
    now = clock or _clock
    send = emit or _emit
    send(f"{label}: starting")
    t0 = now()
    try:
        yield
    except BaseException:
        send(f"{label}: failed after {now() - t0:.1f}s")
        raise
    send(f"{label}: done in {now() - t0:.1f}s")
