"""Tests for timeline arc builder and MCP tool (Phase 10)."""

from __future__ import annotations

import sqlite3

import pytest

from synapt.recall.journal import JournalEntry
from synapt.recall.timeline import (
    _focus_tokens,
    _hours_between,
    _timeline_id,
    build_timeline_clusters,
    save_timeline_clusters,
)


def _entry(
    session_id: str,
    timestamp: str = "2026-03-01T00:00:00Z",
    branch: str = "",
    focus: str = "",
    **kwargs,
) -> JournalEntry:
    defaults = {
        "timestamp": timestamp,
        "session_id": session_id,
        "branch": branch,
        "focus": focus,
        "done": [],
        "git_log": [],
    }
    defaults.update(kwargs)
    return JournalEntry(**defaults)


class TestHelpers:
    def test_timeline_id_deterministic(self):
        id1 = _timeline_id(["s1", "s2"])
        id2 = _timeline_id(["s2", "s1"])  # Order shouldn't matter
        assert id1 == id2
        assert id1.startswith("tl-")
        assert len(id1) == 15  # "tl-" + 12 hex chars

    def test_timeline_id_different_sessions(self):
        id1 = _timeline_id(["s1", "s2"])
        id2 = _timeline_id(["s1", "s3"])
        assert id1 != id2

    def test_hours_between(self):
        assert _hours_between(
            "2026-03-01T00:00:00Z", "2026-03-01T06:00:00Z"
        ) == 6.0
        assert _hours_between(
            "2026-03-01T06:00:00Z", "2026-03-01T00:00:00Z"
        ) == 6.0  # Absolute value

    def test_hours_between_mixed_timezone(self):
        """Mixed aware/naive timestamps should not crash."""
        result = _hours_between(
            "2026-03-01T00:00:00+00:00", "2026-03-01T06:00:00"
        )
        assert result == 6.0

    def test_focus_tokens(self):
        entry = _entry("s1", focus="Jaccard clustering with inverted index")
        tokens = _focus_tokens(entry)
        assert "jaccard" in tokens  # Lowercase from tokenizer
        assert len(tokens) > 0

    def test_focus_tokens_empty(self):
        assert _focus_tokens(None) == set()
        assert _focus_tokens(_entry("s1", focus="")) == set()


@pytest.fixture
def db(tmp_path):
    """Create a minimal RecallDB for testing."""
    from synapt.recall.storage import RecallDB

    db = RecallDB(tmp_path / "test.db")
    return db


def _insert_chunks(db, session_id: str, timestamp: str, count: int = 3):
    """Insert dummy chunks for a session."""
    for i in range(count):
        db._conn.execute(
            "INSERT INTO chunks (id, session_id, turn_index, timestamp, "
            "user_text, assistant_text, tools_used, files_touched) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"{session_id}:t{i}",
                session_id,
                i,
                timestamp,
                f"user message {i}",
                f"assistant response {i}",
                "[]",
                "[]",
            ),
        )
    db._conn.commit()


class TestBuildTimelineClusters:
    def test_same_branch_groups(self, db):
        """3 sessions on same branch within 48h -> 1 arc."""
        _insert_chunks(db, "s1", "2026-03-01T10:00:00Z")
        _insert_chunks(db, "s2", "2026-03-01T14:00:00Z")
        _insert_chunks(db, "s3", "2026-03-02T10:00:00Z")

        entries = [
            _entry("s1", "2026-03-01T10:00:00Z", branch="feat/clustering"),
            _entry("s2", "2026-03-01T14:00:00Z", branch="feat/clustering"),
            _entry("s3", "2026-03-02T10:00:00Z", branch="feat/clustering"),
        ]
        arcs = build_timeline_clusters(db, entries)
        assert len(arcs) == 1
        assert len(arcs[0]["session_ids"]) == 3
        assert arcs[0]["cluster_type"] == "timeline"
        assert arcs[0]["branch"] == "feat/clustering"

    def test_branch_change_splits(self, db):
        """Different branches -> separate arcs."""
        _insert_chunks(db, "s1", "2026-03-01T10:00:00Z")
        _insert_chunks(db, "s2", "2026-03-01T14:00:00Z")

        entries = [
            _entry("s1", "2026-03-01T10:00:00Z", branch="feat/clustering"),
            _entry("s2", "2026-03-01T14:00:00Z", branch="fix/recall-309"),
        ]
        arcs = build_timeline_clusters(db, entries)
        assert len(arcs) == 2

    def test_time_gap_splits(self, db):
        """Same branch, 3-day gap -> 2 arcs."""
        _insert_chunks(db, "s1", "2026-03-01T10:00:00Z")
        _insert_chunks(db, "s2", "2026-03-04T10:00:00Z")  # 72h later

        entries = [
            _entry("s1", "2026-03-01T10:00:00Z", branch="feat/clustering"),
            _entry("s2", "2026-03-04T10:00:00Z", branch="feat/clustering"),
        ]
        arcs = build_timeline_clusters(db, entries)
        assert len(arcs) == 2

    def test_branchless_focus_overlap(self, db):
        """Branchless sessions with overlapping focus -> 1 arc."""
        _insert_chunks(db, "s1", "2026-03-01T10:00:00Z")
        _insert_chunks(db, "s2", "2026-03-01T14:00:00Z")

        entries = [
            _entry("s1", "2026-03-01T10:00:00Z",
                   focus="Jaccard clustering optimization with inverted index"),
            _entry("s2", "2026-03-01T14:00:00Z",
                   focus="Jaccard clustering performance using inverted index"),
        ]
        arcs = build_timeline_clusters(db, entries)
        assert len(arcs) == 1

    def test_branchless_no_overlap(self, db):
        """Branchless sessions with different focus -> 2 arcs."""
        _insert_chunks(db, "s1", "2026-03-01T10:00:00Z")
        _insert_chunks(db, "s2", "2026-03-01T14:00:00Z")

        entries = [
            _entry("s1", "2026-03-01T10:00:00Z",
                   focus="Jaccard clustering optimization with inverted index"),
            _entry("s2", "2026-03-01T14:00:00Z",
                   focus="Swift adapter training on Modal with quality curve"),
        ]
        arcs = build_timeline_clusters(db, entries)
        assert len(arcs) == 2

    def test_deterministic_ids(self, db):
        """Same sessions -> same cluster_id."""
        _insert_chunks(db, "s1", "2026-03-01T10:00:00Z")
        _insert_chunks(db, "s2", "2026-03-01T14:00:00Z")

        entries = [
            _entry("s1", "2026-03-01T10:00:00Z", branch="feat/x"),
            _entry("s2", "2026-03-01T14:00:00Z", branch="feat/x"),
        ]
        arcs1 = build_timeline_clusters(db, entries)
        arcs2 = build_timeline_clusters(db, entries)
        assert arcs1[0]["cluster_id"] == arcs2[0]["cluster_id"]

    def test_no_sessions_returns_empty(self, db):
        arcs = build_timeline_clusters(db, [])
        assert arcs == []

    def test_tags_extracted(self, db):
        """Timeline arcs should have tags from journal entries."""
        _insert_chunks(db, "s1", "2026-03-01T10:00:00Z")

        entries = [
            _entry("s1", "2026-03-01T10:00:00Z",
                   branch="feat/recall-305",
                   git_log=["feat: recall benchmarks (#305)"]),
        ]
        arcs = build_timeline_clusters(db, entries)
        assert len(arcs) == 1
        tags = arcs[0]["tags"]
        assert "issue:305" in tags
        assert "branch:feat/recall-305" in tags


class TestSaveAndLoad:
    def test_roundtrip(self, db):
        """Save and load timeline clusters."""
        _insert_chunks(db, "s1", "2026-03-01T10:00:00Z")
        _insert_chunks(db, "s2", "2026-03-01T14:00:00Z")

        entries = [
            _entry("s1", "2026-03-01T10:00:00Z", branch="feat/x"),
            _entry("s2", "2026-03-01T14:00:00Z", branch="feat/x"),
        ]
        arcs = build_timeline_clusters(db, entries)
        save_timeline_clusters(db, arcs)

        loaded = db.load_timeline_clusters()
        assert len(loaded) == 1
        assert loaded[0]["cluster_id"] == arcs[0]["cluster_id"]
        assert loaded[0]["branch"] == "feat/x"

    def test_rebuild_replaces_timeline_only(self, db):
        """Rebuilding timeline doesn't touch topic clusters."""
        # Insert a topic cluster
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        db.save_clusters(
            [
                {
                    "cluster_id": "clust-abc123",
                    "topic": "test topic",
                    "search_text": "test",
                    "cluster_type": "topic",
                    "session_ids": [],
                    "branch": None,
                    "date_start": now,
                    "date_end": now,
                    "chunk_count": 3,
                    "status": "active",
                    "tags": [],
                    "created_at": now,
                    "updated_at": now,
                }
            ],
            [],
        )

        # Build and save timeline
        _insert_chunks(db, "s1", "2026-03-01T10:00:00Z")
        entries = [_entry("s1", "2026-03-01T10:00:00Z")]
        arcs = build_timeline_clusters(db, entries)
        save_timeline_clusters(db, arcs)

        # Topic cluster should still exist
        topic = db.get_cluster("clust-abc123")
        assert topic is not None
        assert topic["cluster_type"] == "topic"

        # Timeline should also exist
        timeline = db.load_timeline_clusters()
        assert len(timeline) == 1

    def test_tags_in_fts_search(self, db):
        """Tags in search_text should be findable via FTS."""
        _insert_chunks(db, "s1", "2026-03-01T10:00:00Z")

        entries = [
            _entry("s1", "2026-03-01T10:00:00Z",
                   branch="feat/recall-305",
                   focus="recall benchmarks",
                   git_log=["feat: recall benchmarks (#305)"]),
        ]
        arcs = build_timeline_clusters(db, entries)
        save_timeline_clusters(db, arcs)

        # FTS search for the issue tag content
        rows = db._conn.execute(
            "SELECT c.cluster_id FROM clusters c "
            "JOIN clusters_fts f ON c.id = f.rowid "
            "WHERE clusters_fts MATCH 'recall'",
        ).fetchall()
        cluster_ids = [r["cluster_id"] for r in rows]
        assert arcs[0]["cluster_id"] in cluster_ids


class TestDateAndBranchFilters:
    def test_after_filter(self, db):
        _insert_chunks(db, "s1", "2026-03-01T10:00:00Z")
        _insert_chunks(db, "s2", "2026-03-05T10:00:00Z")

        entries = [
            _entry("s1", "2026-03-01T10:00:00Z"),
            _entry("s2", "2026-03-05T10:00:00Z"),
        ]
        arcs = build_timeline_clusters(db, entries)
        save_timeline_clusters(db, arcs)

        # Filter: only arcs ending after March 4
        filtered = db.load_timeline_clusters(after="2026-03-04")
        assert len(filtered) == 1
        assert "s2" in filtered[0]["session_ids"]

    def test_branch_filter(self, db):
        _insert_chunks(db, "s1", "2026-03-01T10:00:00Z")
        _insert_chunks(db, "s2", "2026-03-05T10:00:00Z")

        entries = [
            _entry("s1", "2026-03-01T10:00:00Z", branch="feat/a"),
            _entry("s2", "2026-03-05T10:00:00Z", branch="feat/b"),
        ]
        arcs = build_timeline_clusters(db, entries)
        save_timeline_clusters(db, arcs)

        filtered = db.load_timeline_clusters(branch="feat/a")
        assert len(filtered) == 1
        assert filtered[0]["branch"] == "feat/a"


class TestSchemaMigration:
    def test_tags_column_added(self, tmp_path):
        """Opening a DB without tags column should auto-migrate."""
        db_path = tmp_path / "migrate_test.db"
        # Create DB with old schema (no tags column)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE clusters ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  cluster_id TEXT UNIQUE NOT NULL,"
            "  topic TEXT NOT NULL,"
            "  search_text TEXT NOT NULL DEFAULT '',"
            "  cluster_type TEXT NOT NULL DEFAULT 'topic',"
            "  session_ids TEXT NOT NULL DEFAULT '[]',"
            "  branch TEXT,"
            "  date_start TEXT,"
            "  date_end TEXT,"
            "  chunk_count INTEGER NOT NULL DEFAULT 0,"
            "  status TEXT NOT NULL DEFAULT 'active',"
            "  created_at TEXT NOT NULL,"
            "  updated_at TEXT NOT NULL"
            ")"
        )
        conn.commit()
        conn.close()

        # Open with RecallDB — should migrate
        from synapt.recall.storage import RecallDB

        db = RecallDB(db_path)

        # Verify tags column exists
        cols = {
            r[1]
            for r in db._conn.execute("PRAGMA table_info(clusters)").fetchall()
        }
        assert "tags" in cols
