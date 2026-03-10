"""Timeline arc builder for recall sessions.

Groups consecutive sessions on the same branch/topic into chronological
narrative arcs. Unlike topic clusters (which group chunks by content),
timeline clusters group sessions by temporal proximity and branch context.

Timeline cluster IDs use a ``tl-`` prefix to distinguish from topic
``clust-`` IDs. They reference session_ids but don't populate
cluster_chunks (session-level, not chunk-level).

Phase 10 of the adaptive memory system.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from synapt.recall.bm25 import _tokenize
from synapt.recall.clustering import _STOP_TOKENS
from synapt.recall.tagging import extract_tags

if TYPE_CHECKING:
    from synapt.recall.journal import JournalEntry
    from synapt.recall.storage import RecallDB

logger = logging.getLogger(__name__)

# Maximum gap between sessions to group them in the same arc
MAX_GAP_HOURS = 48

# Minimum Jaccard overlap on focus tokens for branchless grouping
MIN_FOCUS_JACCARD = 0.3


def _timeline_id(session_ids: list[str]) -> str:
    """Deterministic timeline cluster ID from sorted session IDs.

    Format: "tl-<first 12 hex chars of SHA1>".
    """
    key = "\n".join(sorted(session_ids))
    sha = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"tl-{sha}"


def _focus_tokens(entry: JournalEntry | None) -> set[str]:
    """Extract distinctive tokens from a journal entry's focus text."""
    if not entry or not entry.focus:
        return set()
    tokens = set(_tokenize(entry.focus))
    return tokens - _STOP_TOKENS


def _hours_between(ts1: str, ts2: str) -> float:
    """Hours between two ISO 8601 timestamps.

    Handles mixed timezone-aware/naive timestamps by stripping tzinfo
    when they don't match (treats both as UTC).
    """
    t1 = datetime.fromisoformat(ts1.replace("Z", "+00:00"))
    t2 = datetime.fromisoformat(ts2.replace("Z", "+00:00"))
    # Prevent TypeError on mixed aware/naive comparison
    if (t1.tzinfo is None) != (t2.tzinfo is None):
        t1 = t1.replace(tzinfo=None)
        t2 = t2.replace(tzinfo=None)
    return abs((t2 - t1).total_seconds()) / 3600


def _session_info(db: RecallDB) -> list[dict]:
    """Query per-session metadata from the chunks table.

    Returns list of dicts with session_id, first_ts, last_ts, chunk_count,
    ordered chronologically.
    """
    rows = db._conn.execute(
        "SELECT session_id, MIN(timestamp) as first_ts, "
        "MAX(timestamp) as last_ts, COUNT(*) as cnt "
        "FROM chunks WHERE turn_index >= 0 "
        "GROUP BY session_id ORDER BY first_ts"
    ).fetchall()
    return [
        {
            "session_id": r["session_id"],
            "first_ts": r["first_ts"],
            "last_ts": r["last_ts"],
            "chunk_count": r["cnt"],
        }
        for r in rows
    ]


def build_timeline_clusters(
    db: RecallDB,
    journal_entries: list[JournalEntry],
) -> list[dict]:
    """Group sessions into chronological arcs by branch/topic proximity.

    Grouping heuristic (greedy, chronological order):
      1. Same non-empty branch within 48h -> merge into arc
      2. Both branchless: focus token Jaccard >= 0.3 within 48h -> merge
      3. Otherwise -> new arc

    Returns list of cluster dicts ready for save_timeline_clusters().
    """
    sessions = _session_info(db)
    if not sessions:
        return []

    # Key journal entries by session_id
    journal_by_session: dict[str, JournalEntry] = {}
    for entry in journal_entries:
        if entry.session_id:
            # Keep richest entry per session
            existing = journal_by_session.get(entry.session_id)
            if existing is None or (entry.has_content() and not existing.has_content()):
                journal_by_session[entry.session_id] = entry

    # Greedy grouping
    arcs: list[list[dict]] = []  # Each arc is a list of session info dicts

    for sess in sessions:
        sid = sess["session_id"]
        entry = journal_by_session.get(sid)
        branch = entry.branch if entry else ""

        merged = False
        if arcs:
            last_arc = arcs[-1]
            last_sid = last_arc[-1]["session_id"]
            last_entry = journal_by_session.get(last_sid)
            last_branch = last_entry.branch if last_entry else ""
            last_ts = last_arc[-1]["last_ts"]

            gap = _hours_between(last_ts, sess["first_ts"])

            if gap <= MAX_GAP_HOURS:
                # Same non-empty branch -> merge
                if branch and last_branch and branch == last_branch:
                    last_arc.append(sess)
                    merged = True
                # Both branchless -> check focus overlap
                elif not branch and not last_branch:
                    cur_tokens = _focus_tokens(entry)
                    last_tokens = _focus_tokens(last_entry)
                    if cur_tokens and last_tokens:
                        intersection = cur_tokens & last_tokens
                        union = cur_tokens | last_tokens
                        jaccard = len(intersection) / len(union) if union else 0
                        if jaccard >= MIN_FOCUS_JACCARD:
                            last_arc.append(sess)
                            merged = True

        if not merged:
            arcs.append([sess])

    # Convert arcs to cluster dicts
    now = datetime.now(timezone.utc).isoformat()
    clusters: list[dict] = []

    for arc in arcs:
        session_ids = [s["session_id"] for s in arc]
        cluster_id = _timeline_id(session_ids)

        # Collect metadata from journal entries
        arc_entries = [
            journal_by_session[s["session_id"]]
            for s in arc
            if s["session_id"] in journal_by_session
        ]
        branch = ""
        for e in arc_entries:
            if e.branch:
                branch = e.branch
                break

        # Build search_text from journal content (strip system artifacts)
        from synapt.recall.scrub import strip_system_artifacts
        search_parts: list[str] = []
        for e in arc_entries:
            if e.focus:
                cleaned = strip_system_artifacts(e.focus)
                if cleaned:
                    search_parts.append(cleaned)
            for item in e.done:
                cleaned = strip_system_artifacts(item)
                if cleaned:
                    search_parts.append(cleaned)
            for item in e.decisions:
                cleaned = strip_system_artifacts(item)
                if cleaned:
                    search_parts.append(cleaned)
            for item in e.next_steps:
                cleaned = strip_system_artifacts(item)
                if cleaned:
                    search_parts.append(cleaned)
        search_text = " ".join(search_parts)
        if len(search_text) > 2000:
            search_text = search_text[:2000].rsplit(" ", 1)[0]

        # Build topic from branch or focus
        if branch:
            topic = branch
        elif arc_entries and arc_entries[0].focus:
            focus = strip_system_artifacts(arc_entries[0].focus)
            topic = focus[:80] + ("..." if len(focus) > 80 else "") if focus else ""
        else:
            topic = f"sessions {arc[0]['first_ts'][:10]}"

        cluster = {
            "cluster_id": cluster_id,
            "topic": topic,
            "search_text": search_text,
            "cluster_type": "timeline",
            "session_ids": session_ids,
            "branch": branch or None,
            "date_start": arc[0]["first_ts"],
            "date_end": arc[-1]["last_ts"],
            "chunk_count": sum(s["chunk_count"] for s in arc),
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "tags": [],
        }

        # Extract tags
        tags = extract_tags(cluster, journal_entries)
        if tags:
            cluster["tags"] = tags
            cluster["search_text"] += " " + " ".join(tags)

        clusters.append(cluster)

    return clusters


def save_timeline_clusters(db: RecallDB, clusters: list[dict]) -> None:
    """Replace all timeline clusters (doesn't touch topic clusters).

    Timeline clusters are session-level — they don't populate
    cluster_chunks. Only the clusters table is modified.
    """
    cur = db._conn.cursor()

    # Remove old timeline clusters + their FTS entries
    cur.execute("DROP TRIGGER IF EXISTS clusters_ad")
    cur.execute(
        "DELETE FROM clusters WHERE cluster_type = 'timeline'"
    )
    cur.execute("INSERT INTO clusters_fts(clusters_fts) VALUES ('delete-all')")
    cur.execute(
        "CREATE TRIGGER IF NOT EXISTS clusters_ad AFTER DELETE ON clusters BEGIN "
        "  INSERT INTO clusters_fts(clusters_fts, rowid, topic, search_text) "
        "  VALUES ('delete', old.id, old.topic, old.search_text); "
        "END;"
    )

    for c in clusters:
        cur.execute(
            "INSERT INTO clusters "
            "(cluster_id, topic, search_text, cluster_type, session_ids, branch, "
            " date_start, date_end, chunk_count, status, tags, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                c["cluster_id"],
                c["topic"],
                c.get("search_text", ""),
                "timeline",
                json.dumps(c.get("session_ids", [])),
                c.get("branch"),
                c.get("date_start"),
                c.get("date_end"),
                c.get("chunk_count", 0),
                c.get("status", "active"),
                json.dumps(c.get("tags", [])),
                c["created_at"],
                c["updated_at"],
            ),
        )

    # Rebuild FTS from all clusters (topic + timeline + access singletons)
    cur.execute("INSERT INTO clusters_fts(clusters_fts) VALUES ('rebuild')")
    db._conn.commit()
