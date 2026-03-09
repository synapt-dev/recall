"""Session-scoped working memory for recall search.

A small in-memory LRU buffer of recently accessed items. Items in
working memory get a relevance boost in search results — mimicking how
human working memory keeps recently-used context readily available.

Not persisted — resets when the MCP server restarts, like human working
memory clears on sleep.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Tokenizer (same as bm25._tokenize for consistency)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> set[str]:
    """Extract lowercase token set from text."""
    return {m.group().lower() for m in _TOKEN_RE.finditer(text) if len(m.group()) > 1}


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------

MAX_SLOTS = 10  # cognitive science: 7 ± 2, rounded up


@dataclass
class WorkingMemorySlot:
    """A single item in working memory."""

    item_type: str  # "chunk" | "cluster" | "knowledge"
    item_id: str
    content_preview: str  # first 200 chars for display
    tokens: set[str] = field(default_factory=set)
    last_accessed: float = 0.0  # time.monotonic()
    access_count: int = 0


class WorkingMemory:
    """LRU buffer of recently accessed recall items.

    Capacity: MAX_SLOTS (10). Eviction: least-recently-accessed when full.
    """

    def __init__(self) -> None:
        self._slots: dict[str, WorkingMemorySlot] = {}  # keyed by item_id

    def record(self, item_type: str, item_id: str, content: str) -> None:
        """Record that an item was returned in search results or drilled into."""
        now = time.monotonic()
        key = item_id
        if key in self._slots:
            slot = self._slots[key]
            slot.last_accessed = now
            slot.access_count += 1
        else:
            # Evict LRU if at capacity
            if len(self._slots) >= MAX_SLOTS:
                oldest_key = min(self._slots, key=lambda k: self._slots[k].last_accessed)
                del self._slots[oldest_key]
            self._slots[key] = WorkingMemorySlot(
                item_type=item_type,
                item_id=item_id,
                content_preview=content[:200],
                tokens=_tokenize(content),
                last_accessed=now,
                access_count=1,
            )

    def query(self, query_tokens: set[str], max_results: int = 3) -> list[WorkingMemorySlot]:
        """Find working memory slots relevant to a query.

        Uses token overlap — slots sharing 2+ tokens with the query are
        returned, ranked by overlap * (1 + log(access_count)).
        """
        if not query_tokens or not self._slots:
            return []

        scored: list[tuple[float, WorkingMemorySlot]] = []
        for slot in self._slots.values():
            overlap = len(query_tokens & slot.tokens)
            if overlap >= 2:
                score = overlap * (1 + math.log(slot.access_count))
                scored.append((score, slot))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [slot for _, slot in scored[:max_results]]

    def boost_score(self, base_score: float, item_id: str) -> float:
        """Apply working memory boost to a search result score.

        Items in working memory: 1.5x.
        Items accessed 3+ times this session: 2.0x.
        """
        slot = self._slots.get(item_id)
        if slot is None:
            return base_score
        if slot.access_count >= 3:
            return base_score * 2.0
        return base_score * 1.5

    def __len__(self) -> int:
        return len(self._slots)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._slots
