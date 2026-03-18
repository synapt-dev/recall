"""Sharded database utilities for tree-structured recall storage.

Splits the monolithic recall.db into:
  - index.db: lightweight routing index (knowledge, clusters, metadata)
  - data_YYYY_QN.db: per-quarter chunk shards (chunks, FTS, access_log)

This module provides helpers for shard routing, migration from monolithic
DB, and the ShardedRecallDB wrapper that preserves the RecallDB interface.

See issue #89 for the full design.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


def quarter_for_timestamp(timestamp: str) -> str:
    """Return the quarter label for an ISO 8601 timestamp.

    >>> quarter_for_timestamp("2025-03-17T10:00:00Z")
    '2025_q1'
    >>> quarter_for_timestamp("2025-07-01T00:00:00Z")
    '2025_q3'
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        quarter = (dt.month - 1) // 3 + 1
        return f"{dt.year}_q{quarter}"
    except (ValueError, AttributeError):
        return "unknown"


def shard_name(quarter: str) -> str:
    """Return the database filename for a quarter.

    >>> shard_name("2025_q1")
    'data_2025_q1.db'
    """
    return f"data_{quarter}.db"


def list_shards(index_dir: Path) -> list[Path]:
    """List all data shard DBs in the index directory, sorted chronologically."""
    return sorted(index_dir.glob("data_*.db"))


def group_chunks_by_quarter(chunks: list[dict]) -> dict[str, list[dict]]:
    """Group chunk dicts by quarter based on their timestamp.

    Args:
        chunks: List of dicts with at least a 'timestamp' key.

    Returns:
        Dict mapping quarter label to list of chunks.
    """
    groups: dict[str, list[dict]] = {}
    for chunk in chunks:
        ts = chunk.get("timestamp", "")
        q = quarter_for_timestamp(ts)
        groups.setdefault(q, []).append(chunk)
    return groups


def estimate_split(db_path: Path) -> dict[str, int]:
    """Estimate how chunks would be distributed across quarters.

    Returns dict mapping quarter label to chunk count.
    Useful for planning before actually splitting.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT timestamp FROM chunks ORDER BY timestamp"
        ).fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            q = quarter_for_timestamp(row["timestamp"])
            counts[q] = counts.get(q, 0) + 1
        return counts
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Index DB tables — these stay in the lightweight index
# ---------------------------------------------------------------------------

_INDEX_TABLES = frozenset({
    "metadata",
    "knowledge",
    "knowledge_fts",
    "knowledge_fts_data",
    "knowledge_fts_idx",
    "knowledge_fts_docsize",
    "knowledge_fts_config",
    "clusters",
    "clusters_fts",
    "clusters_fts_data",
    "clusters_fts_idx",
    "clusters_fts_docsize",
    "clusters_fts_config",
    "cluster_summaries",
    "cluster_chunks",
    "access_stats",
    "access_log_archive",
    "chunk_links",
    "pending_contradictions",
})

# Data tables — these get sharded per quarter
_DATA_TABLES = frozenset({
    "chunks",
    "access_log",
})


def is_sharded(index_dir: Path) -> bool:
    """Check if an index directory uses the sharded layout.

    Returns True if index.db exists (sharded), False if only recall.db
    exists (monolithic).
    """
    return (index_dir / "index.db").exists()
