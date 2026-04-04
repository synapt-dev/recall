"""Agent-side consumer for wake requests (#525).

Layer 2 of the WakeCoordinator.  Sits on top of the SQLite-based
transport (channel_read_wakes / channel_ack_wakes / channel_wake_targets)
and provides:

- Cursor-based polling with automatic advancement
- Per-target coalescing (N raw wakes for same target -> 1 coalesced wake)
- Priority ordering (highest-priority wakes first)
- No-overlap enforcement (can't process the same target concurrently)
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from synapt.recall.channel import (
    channel_ack_wakes,
    channel_read_wakes,
    channel_wake_targets,
)


class WakeConsumer:
    """Consumes wake requests for a single agent process.

    Thread-safe.  Maintains cursor state and overlap tracking in memory.
    On process restart the cursor resets to 0 -- unacked wakes are
    re-delivered automatically.

    Typical usage::

        consumer = WakeConsumer(agent_name="apollo")
        wakes = consumer.poll()
        for wake in wakes:
            with consumer.processing(wake["target"]):
                handle(wake)
        if wakes:
            consumer.ack(max(w["max_seq"] for w in wakes))
    """

    def __init__(
        self,
        agent_name: str | None = None,
        project_dir: Path | None = None,
    ) -> None:
        self._agent_name = agent_name
        self._project_dir = project_dir
        self._cursor: int = 0
        self._lock = threading.Lock()
        self._processing: set[str] = set()
        self._targets: list[str] | None = None

    # -- Target management ---------------------------------------------------

    @property
    def targets(self) -> list[str]:
        """Wake targets this consumer watches (agent + joined channels)."""
        if self._targets is None:
            self._targets = channel_wake_targets(
                self._agent_name, self._project_dir,
            )
        return self._targets

    def refresh_targets(self) -> list[str]:
        """Re-resolve targets (call after joining/leaving channels)."""
        self._targets = None
        return self.targets

    # -- Polling -------------------------------------------------------------

    def poll(self, limit: int = 100) -> list[dict]:
        """Poll for new wake requests.

        Returns a coalesced, priority-ordered list.  Each dict has the
        standard transport fields plus:

        - ``max_seq``: highest seq in this coalesced group (use for ack)
        - ``coalesced_count``: how many raw wakes were merged
        - ``message_ids``: unique message_ids across all merged payloads

        Wakes for targets currently held via :meth:`processing` are
        excluded so the caller never double-processes a target.
        """
        raw = channel_read_wakes(
            self.targets,
            after_seq=self._cursor,
            limit=limit,
            project_dir=self._project_dir,
        )
        if not raw:
            return []

        coalesced = _coalesce_wakes(raw)
        coalesced.sort(key=lambda w: w["priority"], reverse=True)

        with self._lock:
            available = [
                w for w in coalesced
                if w["target"] not in self._processing
            ]

        return available

    # -- Acknowledgement -----------------------------------------------------

    def ack(self, up_to_seq: int) -> int:
        """Acknowledge processed wakes up to *up_to_seq*.

        Deletes acknowledged rows from the transport table and advances
        the internal cursor so future polls skip them.

        Returns:
            Number of wake rows deleted.
        """
        if up_to_seq <= self._cursor:
            return 0
        deleted = channel_ack_wakes(
            up_to_seq,
            targets=self.targets,
            project_dir=self._project_dir,
        )
        self._cursor = up_to_seq
        return deleted

    @property
    def cursor(self) -> int:
        """Current cursor position (highest acked seq)."""
        return self._cursor

    # -- No-overlap enforcement ----------------------------------------------

    @contextmanager
    def processing(self, target: str) -> Generator[None, None, None]:
        """Context manager that marks *target* as in-flight.

        While held, :meth:`poll` excludes wakes for this target.

        Raises:
            RuntimeError: If *target* is already being processed by
                another call (overlap detected).
        """
        with self._lock:
            if target in self._processing:
                raise RuntimeError(
                    f"Target {target!r} is already being processed"
                )
            self._processing.add(target)
        try:
            yield
        finally:
            with self._lock:
                self._processing.discard(target)

    @property
    def active_targets(self) -> frozenset[str]:
        """Targets currently being processed (read-only snapshot)."""
        with self._lock:
            return frozenset(self._processing)


# ---------------------------------------------------------------------------
# Coalescing helper
# ---------------------------------------------------------------------------

def _coalesce_wakes(wakes: list[dict]) -> list[dict]:
    """Group raw wakes by target; keep highest-priority entry per group.

    For each target the result contains the fields from the highest-priority
    wake plus:

    - ``max_seq``:  max seq across all wakes in the group
    - ``coalesced_count``:  number of raw wakes merged
    - ``message_ids``:  deduplicated message_ids from all payloads
    """
    by_target: dict[str, dict] = {}

    for w in wakes:
        target = w["target"]
        msg_id = w["payload"].get("message_id")

        if target not in by_target:
            by_target[target] = {
                **w,
                "max_seq": w["seq"],
                "coalesced_count": 1,
                "message_ids": [msg_id] if msg_id else [],
            }
        else:
            existing = by_target[target]
            existing["coalesced_count"] += 1
            existing["max_seq"] = max(existing["max_seq"], w["seq"])

            if msg_id and msg_id not in existing["message_ids"]:
                existing["message_ids"].append(msg_id)

            # Promote to higher-priority wake's fields
            if w["priority"] > existing["priority"]:
                existing.update(
                    seq=w["seq"],
                    reason=w["reason"],
                    priority=w["priority"],
                    source=w["source"],
                    payload=w["payload"],
                    created=w["created"],
                )

    return list(by_target.values())
