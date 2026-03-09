"""Tests for archival, log compaction, and decay scoring (Phase 7 of adaptive memory)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from synapt.recall.storage import RecallDB
from synapt.recall.core import TranscriptChunk, TranscriptIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path) -> RecallDB:
    return RecallDB(tmp_path / "test.db")


def _seed_access(
    db, item_type, item_id, *,
    explicit=0, sessions=0, queries=0, tier="raw",
    last_accessed=None, first_accessed=None, decay_score=1.0,
):
    """Seed access_stats with specific values."""
    now = datetime.now(timezone.utc).isoformat()
    db._conn.execute(
        "INSERT OR REPLACE INTO access_stats "
        "(item_type, item_id, access_count, explicit_count, "
        " first_accessed, last_accessed, promotion_tier, "
        " distinct_sessions, distinct_queries, decay_score) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (item_type, item_id, explicit, explicit,
         first_accessed or now, last_accessed or now,
         tier, sessions, queries, decay_score),
    )
    db._conn.commit()


def _seed_cluster(db, cluster_id, topic="test topic", status="active"):
    """Insert a cluster into the DB."""
    now = datetime.now(timezone.utc).isoformat()
    db._conn.execute(
        "INSERT INTO clusters "
        "(cluster_id, topic, cluster_type, session_ids, branch, "
        " date_start, date_end, chunk_count, status, search_text, "
        " created_at, updated_at) "
        "VALUES (?, ?, 'topic', '[]', '', ?, ?, 0, ?, ?, ?, ?)",
        (cluster_id, topic, now, now, status, topic, now, now),
    )
    db._conn.execute(
        "INSERT INTO clusters_fts(clusters_fts) VALUES ('rebuild')"
    )
    db._conn.commit()


def _seed_access_log(db, item_type, item_id, created_at, query="test", score=1.0):
    """Insert an access_log entry with a specific timestamp."""
    db._conn.execute(
        "INSERT INTO access_log (item_type, item_id, context, session_id, query, score, created_at) "
        "VALUES (?, ?, 'search', 'sess-1', ?, ?, ?)",
        (item_type, item_id, query, score, created_at),
    )
    db._conn.commit()


# ---------------------------------------------------------------------------
# TestDecayScores
# ---------------------------------------------------------------------------

class TestDecayScores:
    """Tests for recompute_decay_scores()."""

    def test_recent_item_high_decay(self, tmp_path):
        """Items accessed recently should have decay_score close to 1.0."""
        db = _make_db(tmp_path)
        now = datetime.now(timezone.utc).isoformat()
        _seed_access(db, "chunk", "c1", last_accessed=now)

        db.recompute_decay_scores(half_life_days=30.0)
        stats = db.get_access_stats("chunk", "c1")
        assert stats["decay_score"] > 0.99
        db.close()

    def test_old_item_low_decay(self, tmp_path):
        """Items accessed 60 days ago should have ~0.25 decay (2 half-lives)."""
        db = _make_db(tmp_path)
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        _seed_access(db, "chunk", "c1", last_accessed=old)

        db.recompute_decay_scores(half_life_days=30.0)
        stats = db.get_access_stats("chunk", "c1")
        # 2^(-60/30) = 0.25
        assert 0.20 < stats["decay_score"] < 0.30
        db.close()

    def test_half_life_at_boundary(self, tmp_path):
        """Item at exactly 1 half-life should have ~0.5 decay."""
        db = _make_db(tmp_path)
        half_life = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        _seed_access(db, "chunk", "c1", last_accessed=half_life)

        db.recompute_decay_scores(half_life_days=30.0)
        stats = db.get_access_stats("chunk", "c1")
        assert 0.45 < stats["decay_score"] < 0.55
        db.close()

    def test_returns_count_of_updated_items(self, tmp_path):
        """Should return the number of items updated."""
        db = _make_db(tmp_path)
        _seed_access(db, "chunk", "c1")
        _seed_access(db, "chunk", "c2")
        _seed_access(db, "cluster", "cl1")

        count = db.recompute_decay_scores()
        assert count == 3
        db.close()

    def test_invalid_timestamp_defaults_to_1(self, tmp_path):
        """Items with unparseable timestamps should get decay_score=1.0."""
        db = _make_db(tmp_path)
        db._conn.execute(
            "INSERT INTO access_stats "
            "(item_type, item_id, access_count, explicit_count, "
            " first_accessed, last_accessed, promotion_tier) "
            "VALUES ('chunk', 'c1', 1, 1, 'invalid', 'invalid', 'raw')",
        )
        db._conn.commit()

        db.recompute_decay_scores()
        stats = db.get_access_stats("chunk", "c1")
        assert stats["decay_score"] == 1.0
        db.close()

    def test_zero_half_life_defaults_to_1(self, tmp_path):
        """half_life_days=0 should not crash; items get decay_score=1.0."""
        db = _make_db(tmp_path)
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        _seed_access(db, "chunk", "c1", last_accessed=old)

        db.recompute_decay_scores(half_life_days=0)
        stats = db.get_access_stats("chunk", "c1")
        assert stats["decay_score"] == 1.0
        db.close()

    def test_naive_datetime_treated_as_utc(self, tmp_path):
        """Naive datetime strings (no timezone) should be treated as UTC."""
        db = _make_db(tmp_path)
        naive = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")
        _seed_access(db, "chunk", "c1", last_accessed=naive)

        db.recompute_decay_scores(half_life_days=30.0)
        stats = db.get_access_stats("chunk", "c1")
        # ~0.5 at one half-life
        assert 0.45 < stats["decay_score"] < 0.55
        db.close()

    def test_future_timestamp_clamped_to_1(self, tmp_path):
        """Items with future timestamps should get decay_score=1.0, not > 1."""
        db = _make_db(tmp_path)
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        _seed_access(db, "chunk", "c1", last_accessed=future)

        db.recompute_decay_scores(half_life_days=30.0)
        stats = db.get_access_stats("chunk", "c1")
        assert stats["decay_score"] == 1.0
        db.close()


# ---------------------------------------------------------------------------
# TestArchiveColdClusters
# ---------------------------------------------------------------------------

class TestArchiveColdClusters:
    """Tests for archive_cold_clusters()."""

    def test_archives_cold_old_cluster(self, tmp_path):
        """Clusters with low decay and old last_accessed get archived."""
        db = _make_db(tmp_path)
        old_date = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        _seed_cluster(db, "clust-old", topic="old topic")
        _seed_access(db, "cluster", "clust-old",
                     last_accessed=old_date, decay_score=0.05)

        archived = db.archive_cold_clusters(decay_threshold=0.1, min_age_days=90)
        assert "clust-old" in archived

        # Verify cluster status changed
        info = db.get_cluster("clust-old")
        assert info["status"] == "archived"
        db.close()

    def test_skips_warm_cluster(self, tmp_path):
        """Clusters with high decay should NOT be archived."""
        db = _make_db(tmp_path)
        old_date = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        _seed_cluster(db, "clust-warm", topic="warm topic")
        _seed_access(db, "cluster", "clust-warm",
                     last_accessed=old_date, decay_score=0.5)

        archived = db.archive_cold_clusters(decay_threshold=0.1)
        assert len(archived) == 0
        db.close()

    def test_skips_recently_accessed_cluster(self, tmp_path):
        """Clusters accessed recently should NOT be archived even with low decay."""
        db = _make_db(tmp_path)
        recent = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        _seed_cluster(db, "clust-recent", topic="recent topic")
        _seed_access(db, "cluster", "clust-recent",
                     last_accessed=recent, decay_score=0.05)

        archived = db.archive_cold_clusters(decay_threshold=0.1, min_age_days=90)
        assert len(archived) == 0
        db.close()

    def test_archived_cluster_excluded_from_fts(self, tmp_path):
        """Archived clusters should not appear in cluster_fts_search by default."""
        db = _make_db(tmp_path)
        old_date = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        _seed_cluster(db, "clust-cold", topic="unique banana topic")
        _seed_access(db, "cluster", "clust-cold",
                     last_accessed=old_date, decay_score=0.05)

        # Verify it's findable before archival
        results = db.cluster_fts_search("banana")
        assert any(cid == "clust-cold" for cid, _ in results)

        # Archive it
        db.archive_cold_clusters(decay_threshold=0.1, min_age_days=90)

        # Should NOT appear in default search
        results = db.cluster_fts_search("banana")
        assert not any(cid == "clust-cold" for cid, _ in results)
        db.close()

    def test_include_archived_finds_archived_clusters(self, tmp_path):
        """include_archived=True should return archived clusters in FTS."""
        db = _make_db(tmp_path)
        old_date = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        _seed_cluster(db, "clust-cold", topic="unique banana topic")
        _seed_access(db, "cluster", "clust-cold",
                     last_accessed=old_date, decay_score=0.05)

        db.archive_cold_clusters(decay_threshold=0.1, min_age_days=90)

        # With include_archived=True, it should reappear
        results = db.cluster_fts_search("banana", include_archived=True)
        assert any(cid == "clust-cold" for cid, _ in results)
        db.close()

    def test_does_not_archive_already_archived(self, tmp_path):
        """Already-archived clusters should not be re-archived (no-op)."""
        db = _make_db(tmp_path)
        old_date = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        _seed_cluster(db, "clust-old", topic="old topic", status="archived")
        _seed_access(db, "cluster", "clust-old",
                     last_accessed=old_date, decay_score=0.05)

        archived = db.archive_cold_clusters(decay_threshold=0.1, min_age_days=90)
        # The UPDATE targets status='active' rows only, so already-archived is skipped
        assert len(archived) == 0
        db.close()


# ---------------------------------------------------------------------------
# TestCompactAccessLog
# ---------------------------------------------------------------------------

class TestCompactAccessLog:
    """Tests for compact_access_log()."""

    def test_compacts_old_entries(self, tmp_path):
        """Entries older than retention_days are rolled up and deleted."""
        db = _make_db(tmp_path)
        old_date = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        _seed_access_log(db, "chunk", "c1", old_date, query="topic A", score=0.8)
        _seed_access_log(db, "chunk", "c1", old_date, query="topic B", score=0.9)

        count = db.compact_access_log(retention_days=90)
        assert count == 2

        # Check archive has aggregated entry
        row = db._conn.execute(
            "SELECT * FROM access_log_archive WHERE item_id = 'c1'"
        ).fetchone()
        assert row is not None
        assert row["access_count"] == 2
        assert abs(row["avg_score"] - 0.85) < 0.01
        # Verify queries JSON array contains both distinct queries
        import json
        queries = json.loads(row["queries"])
        assert set(queries) == {"topic A", "topic B"}

        # Original entries should be deleted
        remaining = db._conn.execute(
            "SELECT COUNT(*) FROM access_log WHERE item_id = 'c1'"
        ).fetchone()[0]
        assert remaining == 0
        db.close()

    def test_preserves_recent_entries(self, tmp_path):
        """Entries within retention window should not be compacted."""
        db = _make_db(tmp_path)
        recent = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        _seed_access_log(db, "chunk", "c1", recent)

        count = db.compact_access_log(retention_days=90)
        assert count == 0

        remaining = db._conn.execute(
            "SELECT COUNT(*) FROM access_log WHERE item_id = 'c1'"
        ).fetchone()[0]
        assert remaining == 1
        db.close()

    def test_groups_by_date(self, tmp_path):
        """Multiple days should produce separate archive rows."""
        db = _make_db(tmp_path)
        day1 = (datetime.now(timezone.utc) - timedelta(days=100)).strftime("%Y-%m-%d") + "T10:00:00Z"
        day2 = (datetime.now(timezone.utc) - timedelta(days=101)).strftime("%Y-%m-%d") + "T10:00:00Z"
        _seed_access_log(db, "chunk", "c1", day1)
        _seed_access_log(db, "chunk", "c1", day2)

        count = db.compact_access_log(retention_days=90)
        assert count == 2

        archive_rows = db._conn.execute(
            "SELECT * FROM access_log_archive WHERE item_id = 'c1'"
        ).fetchall()
        assert len(archive_rows) == 2
        db.close()

    def test_returns_zero_when_nothing_to_compact(self, tmp_path):
        """No old entries → returns 0."""
        db = _make_db(tmp_path)
        count = db.compact_access_log(retention_days=90)
        assert count == 0
        db.close()


# ---------------------------------------------------------------------------
# TestDecayDistribution
# ---------------------------------------------------------------------------

class TestDecayDistribution:
    """Tests for decay_distribution()."""

    def test_buckets_correct(self, tmp_path):
        """Items should land in correct decay buckets."""
        db = _make_db(tmp_path)
        _seed_access(db, "chunk", "fresh1", decay_score=0.95)
        _seed_access(db, "chunk", "warm1", decay_score=0.6)
        _seed_access(db, "chunk", "cool1", decay_score=0.2)
        _seed_access(db, "chunk", "cold1", decay_score=0.05)

        dist = db.decay_distribution()
        assert dist["fresh"] == 1
        assert dist["warm"] == 1
        assert dist["cool"] == 1
        assert dist["cold"] == 1
        db.close()

    def test_empty_distribution(self, tmp_path):
        """No items → all zeros."""
        db = _make_db(tmp_path)
        dist = db.decay_distribution()
        assert dist == {"fresh": 0, "warm": 0, "cool": 0, "cold": 0}
        db.close()

    def test_boundary_values(self, tmp_path):
        """Test boundary between buckets."""
        db = _make_db(tmp_path)
        # Exactly at boundary: 0.8 is NOT > 0.8, so it's "warm"
        _seed_access(db, "chunk", "at-08", decay_score=0.8)
        # 0.4 is NOT > 0.4, so it's "cool"
        _seed_access(db, "chunk", "at-04", decay_score=0.4)
        # 0.1 is NOT > 0.1, so it's "cold"
        _seed_access(db, "chunk", "at-01", decay_score=0.1)

        dist = db.decay_distribution()
        assert dist["warm"] == 1   # 0.8
        assert dist["cool"] == 1   # 0.4
        assert dist["cold"] == 1   # 0.1
        db.close()


# ---------------------------------------------------------------------------
# TestIncludeArchivedSearch
# ---------------------------------------------------------------------------

class TestIncludeArchivedSearch:
    """Tests for include_archived flag threading through the search path."""

    def _build_index_with_archived_cluster(self, tmp_path):
        """Create index with one active and one archived cluster."""
        db = _make_db(tmp_path)
        chunks = [
            TranscriptChunk(
                id=f"s1:t{i}", session_id="sess-a",
                timestamp=f"2026-03-01T10:{i:02d}:00Z",
                turn_index=i, user_text=f"question {i}",
                assistant_text=f"answer {i} about zebra migration",
            )
            for i in range(3)
        ]
        db.save_chunks(chunks)

        # Active cluster
        _seed_cluster(db, "clust-active", topic="zebra migration active")
        # Archived cluster
        _seed_cluster(db, "clust-archived", topic="zebra migration archived",
                      status="archived")

        idx = TranscriptIndex(chunks, db=db)
        idx._refresh_rowid_map()
        return idx, db

    def test_default_excludes_archived(self, tmp_path):
        """Default concise lookup excludes archived clusters."""
        idx, db = self._build_index_with_archived_cluster(tmp_path)
        result = idx._concise_lookup("zebra", max_chunks=10, max_tokens=2000)
        assert "active" in result
        assert "archived" not in result
        db.close()

    def test_include_archived_shows_both(self, tmp_path):
        """include_archived=True returns both active and archived clusters."""
        idx, db = self._build_index_with_archived_cluster(tmp_path)
        result = idx._concise_lookup(
            "zebra", max_chunks=10, max_tokens=2000, include_archived=True,
        )
        assert "active" in result
        assert "archived" in result
        db.close()

    def test_lookup_threads_include_archived(self, tmp_path):
        """lookup() passes include_archived through to concise search."""
        idx, db = self._build_index_with_archived_cluster(tmp_path)

        # Without include_archived
        result_default = idx.lookup(
            "zebra", max_chunks=10, max_tokens=2000, depth="concise",
        )
        assert "archived" not in result_default

        # With include_archived
        result_with = idx.lookup(
            "zebra", max_chunks=10, max_tokens=2000,
            depth="concise", include_archived=True,
        )
        assert "archived" in result_with
        db.close()


# ---------------------------------------------------------------------------
# TestSchemaArchival
# ---------------------------------------------------------------------------

class TestSchemaArchival:
    """Tests for archival schema elements."""

    def test_access_log_archive_table_exists(self, tmp_path):
        """The access_log_archive table should exist in new DBs."""
        db = _make_db(tmp_path)
        tables = [
            r[0] for r in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "access_log_archive" in tables
        db.close()

    def test_decay_score_column_exists(self, tmp_path):
        """The decay_score column should exist in access_stats."""
        db = _make_db(tmp_path)
        _seed_access(db, "chunk", "c1")
        stats = db.get_access_stats("chunk", "c1")
        assert "decay_score" in stats
        assert stats["decay_score"] == 1.0  # default
        db.close()

    def test_decay_score_migration(self, tmp_path):
        """Old DBs without decay_score should get it via migration."""
        db = _make_db(tmp_path)
        # Simulate old schema by checking migration works on fresh DB
        # The migration adds decay_score if missing — on a fresh DB it's
        # already there, so this just verifies no crash
        db._migrate_access_stats_table()
        stats = db.get_access_stats("chunk", "c1")
        # No item yet, so None is expected
        assert stats is None
        db.close()


# ---------------------------------------------------------------------------
# TestBuildIntegration
# ---------------------------------------------------------------------------

class TestBuildIntegration:
    """Tests for archival/decay in the build pipeline."""

    def test_decay_recomputation_during_build(self, tmp_path):
        """recompute_decay_scores updates all items during build."""
        db = _make_db(tmp_path)
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        _seed_access(db, "chunk", "c1", last_accessed=old, decay_score=1.0)

        # Simulate what cli.py does during build
        count = db.recompute_decay_scores()
        assert count == 1

        stats = db.get_access_stats("chunk", "c1")
        # After 60 days with 30-day half-life, should be ~0.25
        assert stats["decay_score"] < 0.5
        db.close()

    def test_full_maintenance_sequence(self, tmp_path):
        """The full maintenance sequence (decay → archive → compact) runs cleanly."""
        db = _make_db(tmp_path)

        # Add an old cluster
        old_date = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        _seed_cluster(db, "clust-old", topic="ancient topic")
        _seed_access(db, "cluster", "clust-old", last_accessed=old_date)

        # Add old access log entry
        _seed_access_log(db, "chunk", "c1", old_date)

        # Run full maintenance
        decayed = db.recompute_decay_scores()
        archived = db.archive_cold_clusters(decay_threshold=0.1, min_age_days=90)
        compacted = db.compact_access_log(retention_days=90)

        assert decayed == 1
        assert len(archived) == 1
        assert compacted == 1
        db.close()
