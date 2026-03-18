"""Sharded database utilities for tree-structured recall storage.

Splits the monolithic recall.db into:
  - index.db: lightweight routing index (knowledge, clusters, metadata,
    access tracking)
  - data_YYYY_QN.db: per-quarter chunk shards (chunks + FTS)

This module provides helpers for shard routing, migration from monolithic
DB, and the ShardedRecallDB wrapper that preserves the RecallDB interface.

See issue #89 for the full design.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


def quarter_for_timestamp(timestamp: str | None) -> str:
    """Return the quarter label for an ISO 8601 timestamp.

    Returns ``"unknown"`` for invalid or missing timestamps.

    >>> quarter_for_timestamp("2025-03-17T10:00:00Z")
    '2025_q1'
    >>> quarter_for_timestamp("2025-07-01T00:00:00Z")
    '2025_q3'
    >>> quarter_for_timestamp(None)
    'unknown'
    """
    if not timestamp:
        return "unknown"
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
    """List all data shard DBs in the index directory, sorted by name.

    Lexicographic sort gives chronological order for ``data_YYYY_qN.db``
    names.  A ``data_unknown.db`` (invalid timestamps) sorts last.
    """
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
    Returns empty dict if the DB has no chunks table.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Check if chunks table exists
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if row is None:
            return {}
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
# Index DB tables — these stay in the lightweight always-loaded index.
# Includes access tracking (keyed by access time, not chunk creation time).
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
    "access_log",
    "access_stats",
    "access_log_archive",
    "chunk_links",
    "pending_contradictions",
})

# Data tables — these get sharded per quarter.
# chunks_fts must co-locate with chunks (FTS5 content-sync triggers
# reference the same DB's chunks table).
_DATA_TABLES = frozenset({
    "chunks",
    "chunks_fts",
    "chunks_fts_data",
    "chunks_fts_idx",
    "chunks_fts_docsize",
    "chunks_fts_config",
})


def split_monolithic_db(
    index_dir: Path,
    dry_run: bool = False,
) -> dict[str, int]:
    """Split a monolithic recall.db into index.db + quarterly data shards.

    Creates:
      - index.db: knowledge, clusters, metadata, access tracking
      - data_YYYY_qN.db: chunks grouped by quarter

    Args:
        index_dir: Directory containing recall.db.
        dry_run: If True, only returns the split plan without writing.

    Returns:
        Dict mapping shard name to chunk count (including "index.db": 0).

    Raises:
        FileNotFoundError: If recall.db doesn't exist.
        RuntimeError: If already sharded.
    """
    db_path = index_dir / "recall.db"
    if not db_path.exists():
        raise FileNotFoundError(f"No recall.db found at {index_dir}")
    if is_sharded(index_dir):
        raise RuntimeError(f"Already sharded at {index_dir}")

    # Plan the split
    if dry_run:
        plan = estimate_split(db_path)
        plan["index.db"] = 0
        return plan

    plan: dict[str, int] = {"index.db": 0}

    src = sqlite3.connect(str(db_path))
    src.row_factory = sqlite3.Row
    try:
        # Step 1: Create index.db with non-chunk tables
        idx_path = index_dir / "index.db"
        idx = sqlite3.connect(str(idx_path))
        idx.execute("PRAGMA journal_mode=WAL")
        idx.execute(f"ATTACH DATABASE '{db_path}' AS src")
        try:
            # Copy schema + data for index tables (skip FTS virtual tables —
            # they'll be recreated by RecallDB.__init__ on first open)
            for table_info in src.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall():
                tname = table_info["name"]
                tsql = table_info["sql"]
                if tname in _INDEX_TABLES and tsql and not tname.endswith(("_data", "_idx", "_docsize", "_config")):
                    idx.execute(tsql)
                    idx.execute(f"INSERT INTO [{tname}] SELECT * FROM src.[{tname}]")
            # Copy indexes
            for idx_info in src.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
            ).fetchall():
                try:
                    idx.execute(idx_info["sql"])
                except sqlite3.OperationalError:
                    pass  # Index for table not in index.db
            idx.commit()
        finally:
            idx.execute("DETACH DATABASE src")
            idx.close()

        # Step 2: Create data shards grouped by quarter
        chunks_by_quarter: dict[str, list[sqlite3.Row]] = {}
        for row in src.execute("SELECT * FROM chunks ORDER BY timestamp"):
            q = quarter_for_timestamp(row["timestamp"])
            chunks_by_quarter.setdefault(q, []).append(row)

        col_names = [desc[0] for desc in src.execute("SELECT * FROM chunks LIMIT 0").description]

        for quarter, rows in chunks_by_quarter.items():
            shard_path = index_dir / shard_name(quarter)
            shard = sqlite3.connect(str(shard_path))
            shard.execute("PRAGMA journal_mode=WAL")
            try:
                # Create chunks table with same schema
                chunks_sql = src.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks'"
                ).fetchone()["sql"]
                shard.execute(chunks_sql)

                # Insert chunks
                placeholders = ",".join("?" for _ in col_names)
                for row in rows:
                    shard.execute(
                        f"INSERT INTO chunks ({','.join(col_names)}) VALUES ({placeholders})",
                        tuple(row[c] for c in col_names),
                    )
                shard.commit()
            finally:
                shard.close()
            plan[shard_name(quarter)] = len(rows)

        # Step 3: Validate total chunk count
        original_count = src.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        split_count = sum(v for k, v in plan.items() if k != "index.db")
        if split_count != original_count:
            raise RuntimeError(
                f"Chunk count mismatch: original={original_count}, split={split_count}"
            )

    finally:
        src.close()

    return plan


def is_sharded(index_dir: Path) -> bool:
    """Check if an index directory uses the sharded layout.

    Returns True if index.db exists (sharded), False if only recall.db
    exists (monolithic).
    """
    return (index_dir / "index.db").exists()
