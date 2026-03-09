"""Tests for the promotion pipeline (Phases 6 + 9 of adaptive memory)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synapt.recall.promotion import (
    ACTION_AUTO_PROMOTE_KNOWLEDGE,
    ACTION_ENSURE_CLUSTER,
    ACTION_FLAG_KNOWLEDGE_CANDIDATE,
    ACTION_GENERATE_LLM_SUMMARY,
    TIER_CLUSTERED,
    TIER_KNOWLEDGE,
    TIER_PROMOTED,
    TIER_RAW,
    TIER_SUMMARIZED,
    check_promotions,
    execute_cheap_promotions,
    process_build_promotions,
)
from synapt.recall.storage import RecallDB
from synapt.recall.core import TranscriptChunk, TranscriptIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path) -> RecallDB:
    return RecallDB(tmp_path / "test.db")


def _seed_access(db, item_type, item_id, explicit=0, sessions=0, queries=0, tier="raw"):
    """Seed access_stats with specific values for testing promotion logic."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    db._conn.execute(
        "INSERT OR REPLACE INTO access_stats "
        "(item_type, item_id, access_count, explicit_count, "
        " first_accessed, last_accessed, promotion_tier, "
        " distinct_sessions, distinct_queries) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (item_type, item_id, explicit, explicit, now, now, tier, sessions, queries),
    )
    db._conn.commit()


def _make_index_with_chunks(tmp_path, n=3):
    """Create an index with n chunks and return (index, db)."""
    db = _make_db(tmp_path)
    chunks = [
        TranscriptChunk(
            id=f"s1:t{i}", session_id="sess-a",
            timestamp=f"2026-03-01T10:{i:02d}:00Z",
            turn_index=i, user_text=f"question {i}",
            assistant_text=f"answer {i} about topic alpha",
        )
        for i in range(n)
    ]
    db.save_chunks(chunks)
    idx = TranscriptIndex(chunks, db=db)
    idx._refresh_rowid_map()
    return idx, db


# ---------------------------------------------------------------------------
# TestCheckPromotions
# ---------------------------------------------------------------------------

class TestCheckPromotions:
    """Tests for check_promotions() threshold logic."""

    def test_no_stats_returns_empty(self, tmp_path):
        db = _make_db(tmp_path)
        assert check_promotions(db, "chunk", "nonexistent") == []
        db.close()

    def test_raw_below_threshold(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_access(db, "chunk", "s1:t0", explicit=2, sessions=1)
        assert check_promotions(db, "chunk", "s1:t0") == []
        db.close()

    def test_raw_meets_cluster_threshold(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_access(db, "chunk", "s1:t0", explicit=3, sessions=2)
        actions = check_promotions(db, "chunk", "s1:t0")
        assert actions == [ACTION_ENSURE_CLUSTER]
        db.close()

    def test_raw_needs_multi_session(self, tmp_path):
        """explicit_count >= 3 but only 1 session -> no promotion."""
        db = _make_db(tmp_path)
        _seed_access(db, "chunk", "s1:t0", explicit=5, sessions=1)
        assert check_promotions(db, "chunk", "s1:t0") == []
        db.close()

    def test_clustered_meets_summary_threshold(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_access(db, "cluster", "clust-001", explicit=3, tier=TIER_CLUSTERED)
        actions = check_promotions(db, "cluster", "clust-001")
        assert actions == [ACTION_GENERATE_LLM_SUMMARY]
        db.close()

    def test_summarized_meets_knowledge_candidate(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_access(db, "cluster", "clust-001", explicit=5, sessions=3, tier=TIER_SUMMARIZED)
        actions = check_promotions(db, "cluster", "clust-001")
        assert actions == [ACTION_FLAG_KNOWLEDGE_CANDIDATE]
        db.close()

    def test_promoted_meets_auto_promote(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_access(db, "cluster", "clust-001", explicit=10, sessions=5, queries=5, tier=TIER_PROMOTED)
        actions = check_promotions(db, "cluster", "clust-001")
        assert actions == [ACTION_AUTO_PROMOTE_KNOWLEDGE]
        db.close()

    def test_promoted_below_query_threshold(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_access(db, "cluster", "clust-001", explicit=10, sessions=5, queries=3, tier=TIER_PROMOTED)
        assert check_promotions(db, "cluster", "clust-001") == []
        db.close()

    def test_knowledge_tier_no_further_action(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_access(db, "cluster", "clust-001", explicit=20, sessions=10, queries=10, tier=TIER_KNOWLEDGE)
        assert check_promotions(db, "cluster", "clust-001") == []
        db.close()


# ---------------------------------------------------------------------------
# TestExecuteCheapPromotions
# ---------------------------------------------------------------------------

class TestExecuteCheapPromotions:
    """Tests for inline cheap promotion execution."""

    def test_ensure_cluster_creates_singleton(self, tmp_path):
        idx, db = _make_index_with_chunks(tmp_path)
        _seed_access(db, "chunk", "s1:t0", explicit=3, sessions=2)

        completed = execute_cheap_promotions(
            db, "chunk", "s1:t0", [ACTION_ENSURE_CLUSTER],
        )
        assert len(completed) == 1
        assert "singleton cluster" in completed[0]

        # Tier should advance
        stats = db.get_access_stats("chunk", "s1:t0")
        assert stats["promotion_tier"] == TIER_CLUSTERED

        # Chunk should be in a cluster
        clusters = db.clusters_for_chunk("s1:t0")
        assert len(clusters) >= 1
        db.close()

    def test_ensure_cluster_skips_already_clustered(self, tmp_path):
        idx, db = _make_index_with_chunks(tmp_path)
        _seed_access(db, "chunk", "s1:t0", explicit=3, sessions=2)

        # Create cluster first
        cid1 = db.create_singleton_cluster("s1:t0")
        # Try again — should return existing cluster, not create new
        cid2 = db.create_singleton_cluster("s1:t0")
        assert cid1 == cid2
        db.close()

    def test_flag_knowledge_candidate(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_access(db, "cluster", "clust-001", explicit=5, sessions=3, tier=TIER_SUMMARIZED)

        completed = execute_cheap_promotions(
            db, "cluster", "clust-001", [ACTION_FLAG_KNOWLEDGE_CANDIDATE],
        )
        assert len(completed) == 1
        stats = db.get_access_stats("cluster", "clust-001")
        assert stats["promotion_tier"] == TIER_PROMOTED
        db.close()

    def test_llm_actions_not_executed_inline(self, tmp_path):
        """LLM-dependent actions should be skipped by execute_cheap_promotions."""
        db = _make_db(tmp_path)
        _seed_access(db, "cluster", "clust-001", explicit=3, tier=TIER_CLUSTERED)

        completed = execute_cheap_promotions(
            db, "cluster", "clust-001",
            [ACTION_GENERATE_LLM_SUMMARY, ACTION_AUTO_PROMOTE_KNOWLEDGE],
        )
        assert completed == []
        # Tier should NOT have changed
        stats = db.get_access_stats("cluster", "clust-001")
        assert stats["promotion_tier"] == TIER_CLUSTERED
        db.close()


# ---------------------------------------------------------------------------
# TestBuildPromotions
# ---------------------------------------------------------------------------

class TestBuildPromotions:
    """Tests for process_build_promotions() during recall_build."""

    def test_advance_clustered_to_summarized(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_access(db, "cluster", "clust-001", explicit=4, tier=TIER_CLUSTERED)

        result = process_build_promotions(db)
        assert result["summaries_upgraded"] == 1
        stats = db.get_access_stats("cluster", "clust-001")
        assert stats["promotion_tier"] == TIER_SUMMARIZED
        db.close()

    def test_advance_summarized_to_promoted(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_access(db, "cluster", "clust-001", explicit=6, sessions=3, tier=TIER_SUMMARIZED)

        result = process_build_promotions(db)
        assert result["candidates_flagged"] == 1
        stats = db.get_access_stats("cluster", "clust-001")
        assert stats["promotion_tier"] == TIER_PROMOTED
        db.close()

    def test_advance_promoted_to_knowledge(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_access(db, "cluster", "clust-001", explicit=12, sessions=5, queries=6, tier=TIER_PROMOTED)

        result = process_build_promotions(db)
        assert result["knowledge_promoted"] == 1
        stats = db.get_access_stats("cluster", "clust-001")
        assert stats["promotion_tier"] == TIER_KNOWLEDGE
        db.close()

    def test_knowledge_budget_cap(self, tmp_path):
        """Max 3 knowledge promotions per build."""
        db = _make_db(tmp_path)
        for i in range(5):
            _seed_access(
                db, "cluster", f"clust-{i:03d}",
                explicit=15, sessions=6, queries=8, tier=TIER_PROMOTED,
            )

        result = process_build_promotions(db, max_knowledge_promotions=3)
        assert result["knowledge_promoted"] == 3

        # Exactly 3 should be at 'knowledge', 2 still at 'promoted'
        at_knowledge = db.items_at_tier(TIER_KNOWLEDGE)
        at_promoted = db.items_at_tier(TIER_PROMOTED)
        assert len(at_knowledge) == 3
        assert len(at_promoted) == 2
        db.close()

    def test_below_threshold_not_advanced(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_access(db, "cluster", "clust-001", explicit=1, tier=TIER_CLUSTERED)

        result = process_build_promotions(db)
        assert result["summaries_upgraded"] == 0
        stats = db.get_access_stats("cluster", "clust-001")
        assert stats["promotion_tier"] == TIER_CLUSTERED
        db.close()

    def test_tier_persists_across_builds(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_access(db, "cluster", "clust-001", explicit=4, tier=TIER_CLUSTERED)

        process_build_promotions(db)
        stats = db.get_access_stats("cluster", "clust-001")
        assert stats["promotion_tier"] == TIER_SUMMARIZED

        # Second build — tier stays, no re-promotion
        result2 = process_build_promotions(db)
        assert result2["summaries_upgraded"] == 0
        stats = db.get_access_stats("cluster", "clust-001")
        assert stats["promotion_tier"] == TIER_SUMMARIZED
        db.close()

    def test_multi_phase_cascade_in_single_build(self, tmp_path):
        """A single build can cascade: clustered -> summarized -> promoted."""
        db = _make_db(tmp_path)
        # Item qualifies for BOTH clustered->summarized AND summarized->promoted
        _seed_access(
            db, "cluster", "clust-hot",
            explicit=8, sessions=4, queries=3, tier=TIER_CLUSTERED,
        )
        result = process_build_promotions(db)
        # Step 1 advances to summarized, step 2 picks it up and advances to promoted
        assert result["summaries_upgraded"] == 1
        assert result["candidates_flagged"] == 1
        stats = db.get_access_stats("cluster", "clust-hot")
        assert stats["promotion_tier"] == TIER_PROMOTED
        db.close()


# ---------------------------------------------------------------------------
# TestSchemaMigration
# ---------------------------------------------------------------------------

class TestSchemaMigration:
    """Tests for access_stats schema migration."""

    def test_new_columns_exist(self, tmp_path):
        db = _make_db(tmp_path)
        cols = {
            r[1]
            for r in db._conn.execute("PRAGMA table_info(access_stats)").fetchall()
        }
        assert "promotion_tier" in cols
        assert "distinct_sessions" in cols
        assert "distinct_queries" in cols
        db.close()

    def test_migration_adds_missing_columns(self, tmp_path):
        """Simulate an old DB without new columns, then migrate."""
        db_path = tmp_path / "old.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE access_stats ("
            "  item_type TEXT NOT NULL, item_id TEXT NOT NULL, "
            "  access_count INTEGER DEFAULT 0, explicit_count INTEGER DEFAULT 0, "
            "  last_accessed TEXT NOT NULL, first_accessed TEXT NOT NULL, "
            "  PRIMARY KEY (item_type, item_id))"
        )
        conn.commit()
        conn.close()

        # Opening RecallDB should trigger migration
        db = RecallDB(db_path)
        cols = {
            r[1]
            for r in db._conn.execute("PRAGMA table_info(access_stats)").fetchall()
        }
        assert "promotion_tier" in cols
        assert "distinct_sessions" in cols
        assert "distinct_queries" in cols
        db.close()


# ---------------------------------------------------------------------------
# TestDistinctTracking
# ---------------------------------------------------------------------------

class TestDistinctTracking:
    """Tests for distinct_sessions and distinct_queries tracking."""

    def test_distinct_sessions_counted(self, tmp_path):
        db = _make_db(tmp_path)
        # Two accesses from different sessions
        db.record_access(
            [{"item_type": "chunk", "item_id": "c1", "session_id": "sess-a", "query": "q1"}],
            context="search",
        )
        db.record_access(
            [{"item_type": "chunk", "item_id": "c1", "session_id": "sess-b", "query": "q1"}],
            context="search",
        )
        stats = db.get_access_stats("chunk", "c1")
        assert stats["distinct_sessions"] == 2
        db.close()

    def test_distinct_queries_counted(self, tmp_path):
        db = _make_db(tmp_path)
        db.record_access(
            [{"item_type": "chunk", "item_id": "c1", "session_id": "sess-a", "query": "alpha"}],
            context="search",
        )
        db.record_access(
            [{"item_type": "chunk", "item_id": "c1", "session_id": "sess-a", "query": "beta"}],
            context="search",
        )
        stats = db.get_access_stats("chunk", "c1")
        assert stats["distinct_queries"] == 2
        db.close()

    def test_same_session_not_double_counted(self, tmp_path):
        db = _make_db(tmp_path)
        db.record_access(
            [{"item_type": "chunk", "item_id": "c1", "session_id": "sess-a", "query": "q1"}],
            context="search",
        )
        db.record_access(
            [{"item_type": "chunk", "item_id": "c1", "session_id": "sess-a", "query": "q1"}],
            context="search",
        )
        stats = db.get_access_stats("chunk", "c1")
        assert stats["distinct_sessions"] == 1
        assert stats["distinct_queries"] == 1
        db.close()

    def test_hook_context_not_counted_as_explicit(self, tmp_path):
        db = _make_db(tmp_path)
        db.record_access(
            [{"item_type": "chunk", "item_id": "c1", "session_id": "sess-a", "query": "q1"}],
            context="hook",
        )
        stats = db.get_access_stats("chunk", "c1")
        assert stats["explicit_count"] == 0
        assert stats["access_count"] == 1
        db.close()

    def test_search_counts_as_explicit(self, tmp_path):
        """Search accesses should increment explicit_count (per design doc)."""
        db = _make_db(tmp_path)
        db.record_access(
            [{"item_type": "chunk", "item_id": "c1", "session_id": "sess-a", "query": "q1"}],
            context="search",
        )
        stats = db.get_access_stats("chunk", "c1")
        assert stats["explicit_count"] == 1
        db.close()


# ---------------------------------------------------------------------------
# TestPromotionIntegration
# ---------------------------------------------------------------------------

class TestPromotionIntegration:
    """Integration tests for promotions triggered via search/context."""

    def test_search_triggers_promotion_check(self, tmp_path):
        """_format_results should call check_promotions after recording access."""
        idx, db = _make_index_with_chunks(tmp_path)
        # Seed stats just below threshold
        _seed_access(db, "chunk", "s1:t0", explicit=2, sessions=2)

        # Search that returns s1:t0 — should record access (explicit_count -> 3)
        # and trigger promotion check
        i0 = idx._id_to_idx.get("s1:t0", 0)
        idx._format_results([(i0, 5.0)], max_tokens=2000)

        stats = db.get_access_stats("chunk", "s1:t0")
        # explicit_count should have incremented (search counts as explicit)
        assert stats["explicit_count"] >= 3
        db.close()

    def test_promotion_never_fails_search(self, tmp_path):
        """Even if promotion code raises, search should succeed."""
        idx, db = _make_index_with_chunks(tmp_path)
        i0 = idx._id_to_idx.get("s1:t0", 0)

        with patch(
            "synapt.recall.promotion.check_promotions",
            side_effect=RuntimeError("boom"),
        ):
            # Should not raise
            result = idx._format_results([(i0, 5.0)], max_tokens=2000)
            assert "answer 0" in result
        db.close()

    def test_singleton_cluster_created_on_threshold(self, tmp_path):
        """When a chunk hits promotion threshold, it gets a singleton cluster."""
        idx, db = _make_index_with_chunks(tmp_path)
        # Build real access history via record_access (2 sessions, 2 queries)
        db.record_access(
            [{"item_type": "chunk", "item_id": "s1:t1",
              "session_id": "sess-a", "query": "alpha"}],
            context="search",
        )
        db.record_access(
            [{"item_type": "chunk", "item_id": "s1:t1",
              "session_id": "sess-b", "query": "beta"}],
            context="context",
        )
        # explicit_count=2, distinct_sessions=2. Next search -> 3 explicit, triggers.
        i1 = idx._id_to_idx.get("s1:t1", 0)
        idx._format_results([(i1, 5.0)], max_tokens=2000)

        stats = db.get_access_stats("chunk", "s1:t1")
        assert stats["promotion_tier"] == TIER_CLUSTERED

        clusters = db.clusters_for_chunk("s1:t1")
        assert len(clusters) >= 1
        db.close()


# ---------------------------------------------------------------------------
# TestAccessSummaryTiers
# ---------------------------------------------------------------------------

class TestAccessSummaryTiers:
    """Tests for promotion tier visibility in access_summary."""

    def test_query_and_session_propagated_to_access_log(self, tmp_path):
        """_format_results should propagate query and session_id to access items."""
        idx, db = _make_index_with_chunks(tmp_path)
        idx._current_session_id = "sess-test-123"
        i0 = idx._id_to_idx.get("s1:t0", 0)
        idx._format_results([(i0, 5.0)], max_tokens=2000, query="test query")

        # Verify access_log has the query and session_id
        row = db._conn.execute(
            "SELECT query, session_id FROM access_log "
            "WHERE item_id = 's1:t0' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["query"] == "test query"
        assert row["session_id"] == "sess-test-123"

        # Verify distinct tracking works
        stats = db.get_access_stats("chunk", "s1:t0")
        assert stats["distinct_sessions"] >= 1
        assert stats["distinct_queries"] >= 1
        db.close()

    def test_tier_distribution_in_summary(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_access(db, "chunk", "c1", tier=TIER_RAW)
        _seed_access(db, "chunk", "c2", tier=TIER_RAW)
        _seed_access(db, "cluster", "cl1", tier=TIER_CLUSTERED)

        summary = db.access_summary()
        tiers = summary["promotion_tiers"]
        assert tiers.get("raw", 0) == 2
        assert tiers.get("clustered", 0) == 1
        db.close()


# ---------------------------------------------------------------------------
# Phase 9: LLM Cluster Summaries
# ---------------------------------------------------------------------------

def _seed_cluster_with_chunks(db, cluster_id, chunk_ids, topic="test topic"):
    """Create a cluster with chunks in the DB for LLM summary testing."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    # Insert chunks — each with unique text so they pass the diversity filter
    for i, cid in enumerate(chunk_ids):
        db._conn.execute(
            "INSERT OR IGNORE INTO chunks (id, session_id, timestamp, turn_index, "
            "user_text, assistant_text) VALUES (?, ?, ?, ?, ?, ?)",
            (cid, "sess-a", now, i,
             f"Question {i} about the {topic} bug?",
             f"Response {i}: The {topic} issue involves step {i} "
             f"of the debugging process for component {cid}."),
        )
    # Insert cluster
    db._conn.execute(
        "INSERT OR REPLACE INTO clusters "
        "(cluster_id, topic, cluster_type, session_ids, chunk_count, "
        " status, created_at, updated_at) "
        "VALUES (?, ?, 'topic', '[]', ?, 'active', ?, ?)",
        (cluster_id, topic, len(chunk_ids), now, now),
    )
    # Insert memberships
    for cid in chunk_ids:
        db._conn.execute(
            "INSERT OR IGNORE INTO cluster_chunks (cluster_id, chunk_id, added_at) "
            "VALUES (?, ?, ?)",
            (cluster_id, cid, now),
        )
    db._conn.commit()


class TestStorageClusterHelpers:
    """Tests for get_cluster_chunk_texts() and get_chunk_cluster_id()."""

    def test_get_cluster_chunk_texts(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_cluster_with_chunks(db, "clust-001", ["c1", "c2", "c3"])

        texts = db.get_cluster_chunk_texts("clust-001")
        assert len(texts) == 3
        assert "user_text" in texts[0]
        assert "assistant_text" in texts[0]
        assert "test topic" in texts[0]["assistant_text"]
        db.close()

    def test_get_cluster_chunk_texts_empty(self, tmp_path):
        db = _make_db(tmp_path)
        texts = db.get_cluster_chunk_texts("nonexistent")
        assert texts == []
        db.close()

    def test_get_chunk_cluster_id(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_cluster_with_chunks(db, "clust-001", ["c1", "c2"])

        assert db.get_chunk_cluster_id("c1") == "clust-001"
        assert db.get_chunk_cluster_id("c2") == "clust-001"
        assert db.get_chunk_cluster_id("nonexistent") is None
        db.close()


class TestGenerateLLMSummary:
    """Tests for generate_llm_summary() in clustering.py."""

    def test_returns_none_when_no_backend(self):
        """Without any backend, returns None gracefully."""
        with patch("synapt.recall.clustering.create_summary_client", return_value=None):
            from synapt.recall.clustering import generate_llm_summary
            result = generate_llm_summary(
                [{"user_text": "hello", "assistant_text": "world"}],
                topic="test",
            )
            assert result is None

    def test_returns_none_for_empty_chunks(self):
        from synapt.recall.clustering import generate_llm_summary
        result = generate_llm_summary([], topic="test")
        assert result is None

    def test_returns_summary_with_mocked_client(self):
        """With mocked client, generates a summary."""
        mock_summary = "Fixed race condition in lock manager."
        mock_client = MagicMock()
        mock_client.chat.return_value = mock_summary

        with patch("synapt.recall.clustering.create_summary_client", return_value=mock_client):
            from synapt.recall.clustering import generate_llm_summary
            result = generate_llm_summary(
                [{"user_text": "How do I fix the flock race condition in the lock manager?",
                  "assistant_text": "The flock race condition needs a lock timeout with exponential backoff."}],
                topic="flock locking",
            )
            assert result == mock_summary
            mock_client.chat.assert_called_once()

    def test_rejects_summary_longer_than_input(self):
        """Quality gate: rejects summaries that are longer than input."""
        short_input = [{"user_text": "x", "assistant_text": "y"}]
        long_output = "A" * 1000  # Way longer than input

        mock_client = MagicMock()
        mock_client.chat.return_value = long_output

        with patch("synapt.recall.clustering.create_summary_client", return_value=mock_client):
            from synapt.recall.clustering import generate_llm_summary
            result = generate_llm_summary(short_input, topic="test")
            assert result is None

    def test_handles_inference_exception(self):
        """Inference failure returns None, no exception propagated."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("GPU OOM")

        with patch("synapt.recall.clustering.create_summary_client", return_value=mock_client):
            from synapt.recall.clustering import generate_llm_summary
            result = generate_llm_summary(
                [{"user_text": "test", "assistant_text": "data"}],
                topic="test",
            )
            assert result is None

    def test_reuses_provided_client(self):
        """Passing a client skips create_summary_client."""
        mock_client = MagicMock()
        mock_client.chat.return_value = "Summary text here."

        with patch("synapt.recall.clustering.create_summary_client") as mock_create:
            from synapt.recall.clustering import generate_llm_summary
            result = generate_llm_summary(
                [{"user_text": "question about flock",
                  "assistant_text": "flock uses advisory locking via fcntl"}],
                topic="flock",
                client=mock_client,
            )
            assert result == "Summary text here."
            mock_create.assert_not_called()  # Should not create a new client


class TestBuildPromotionsLLMSummary:
    """Tests for LLM summary generation during process_build_promotions()."""

    def test_llm_summary_generated_on_promotion(self, tmp_path):
        """When advancing clustered->summarized, LLM summary is generated."""
        db = _make_db(tmp_path)
        _seed_cluster_with_chunks(db, "clust-001", ["c1", "c2"])
        _seed_access(db, "cluster", "clust-001", explicit=4, tier=TIER_CLUSTERED)

        mock_summary = "Fixed race condition in lock manager."

        with patch(
            "synapt.recall.clustering.generate_llm_summary",
            return_value=mock_summary,
        ) as mock_gen:
            result = process_build_promotions(db, max_llm_summaries=5)

        assert result["summaries_upgraded"] == 1
        assert result["llm_summaries_generated"] == 1

        stored = db.get_cluster_summary("clust-001")
        assert stored is not None
        assert stored["summary"] == mock_summary
        assert stored["method"] == "llm"
        db.close()

    def test_llm_summary_budget_cap(self, tmp_path):
        """LLM summaries are capped at max_llm_summaries per build."""
        db = _make_db(tmp_path)
        for i in range(5):
            cid = f"clust-{i:03d}"
            _seed_cluster_with_chunks(db, cid, [f"c{i}a", f"c{i}b"])
            _seed_access(db, "cluster", cid, explicit=4, tier=TIER_CLUSTERED)

        with patch(
            "synapt.recall.clustering.generate_llm_summary",
            return_value="Short summary.",
        ):
            result = process_build_promotions(db, max_llm_summaries=2)

        # All 5 advance tier, but only 2 get LLM summaries
        assert result["summaries_upgraded"] == 5
        assert result["llm_summaries_generated"] == 2
        db.close()

    def test_fallback_when_llm_returns_none(self, tmp_path):
        """Tier still advances even when LLM summary fails."""
        db = _make_db(tmp_path)
        _seed_cluster_with_chunks(db, "clust-001", ["c1"])
        _seed_access(db, "cluster", "clust-001", explicit=4, tier=TIER_CLUSTERED)

        with patch(
            "synapt.recall.clustering.generate_llm_summary",
            return_value=None,
        ):
            result = process_build_promotions(db)

        # Tier advanced, but no LLM summary saved
        assert result["summaries_upgraded"] == 1
        assert result["llm_summaries_generated"] == 0
        stats = db.get_access_stats("cluster", "clust-001")
        assert stats["promotion_tier"] == TIER_SUMMARIZED
        db.close()

    def test_chunk_type_resolves_to_cluster(self, tmp_path):
        """A chunk at TIER_CLUSTERED resolves its cluster for LLM summary."""
        db = _make_db(tmp_path)
        _seed_cluster_with_chunks(db, "clust-001", ["c1", "c2"])
        _seed_access(db, "chunk", "c1", explicit=4, tier=TIER_CLUSTERED)

        mock_summary = "Summary via chunk lookup."

        with patch(
            "synapt.recall.clustering.generate_llm_summary",
            return_value=mock_summary,
        ):
            result = process_build_promotions(db)

        assert result["llm_summaries_generated"] == 1
        stored = db.get_cluster_summary("clust-001")
        assert stored is not None
        assert stored["method"] == "llm"
        db.close()

    def test_no_cluster_for_chunk_still_advances(self, tmp_path):
        """A chunk with no cluster still advances tier (no LLM summary)."""
        db = _make_db(tmp_path)
        # Chunk exists in DB but not in any cluster
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        db._conn.execute(
            "INSERT INTO chunks (id, session_id, timestamp, turn_index, "
            "user_text, assistant_text) VALUES (?, ?, ?, ?, ?, ?)",
            ("orphan-c1", "sess-a", now, 0, "q", "a"),
        )
        db._conn.commit()
        _seed_access(db, "chunk", "orphan-c1", explicit=4, tier=TIER_CLUSTERED)

        with patch(
            "synapt.recall.clustering.generate_llm_summary",
            return_value="should not be called",
        ) as mock_gen:
            result = process_build_promotions(db)

        # Tier advanced but no LLM summary generated (no cluster to summarize)
        assert result["summaries_upgraded"] == 1
        assert result["llm_summaries_generated"] == 0
        db.close()

    def test_client_created_once_and_reused(self, tmp_path):
        """MLXClient should be created once and reused across clusters."""
        db = _make_db(tmp_path)
        for i in range(3):
            cid = f"clust-{i:03d}"
            _seed_cluster_with_chunks(db, cid, [f"c{i}a", f"c{i}b"])
            _seed_access(db, "cluster", cid, explicit=4, tier=TIER_CLUSTERED)

        with patch(
            "synapt.recall.clustering.generate_llm_summary",
            return_value="Summary.",
        ), patch(
            "synapt.recall.clustering.create_summary_client",
            return_value="fake-client",
        ) as mock_create:
            result = process_build_promotions(db, max_llm_summaries=5)

        # Client factory called exactly once despite 3 qualifying clusters
        assert mock_create.call_count == 1
        assert result["llm_summaries_generated"] == 3
        db.close()

    def test_existing_tests_still_pass_with_new_result_key(self, tmp_path):
        """Verify the new llm_summaries_generated key is always present."""
        db = _make_db(tmp_path)
        result = process_build_promotions(db)
        assert "llm_summaries_generated" in result
        assert result["llm_summaries_generated"] == 0
        db.close()


class TestBuildClusterExcerpts:
    """Tests for _build_cluster_excerpts() prompt builder."""

    def test_builds_excerpts_from_texts(self):
        from synapt.recall.clustering import _build_cluster_excerpts
        texts = [
            {"user_text": "How to fix?", "assistant_text": "Use a lock."},
            {"user_text": "What about deadlocks?", "assistant_text": "Timeout."},
        ]
        result = _build_cluster_excerpts(texts)
        assert "[1]" in result
        assert "[2]" in result
        assert "How to fix?" in result
        assert "Use a lock." in result

    def test_respects_char_budget(self):
        from synapt.recall.clustering import _build_cluster_excerpts
        texts = [
            {"user_text": "A" * 200, "assistant_text": "B" * 400},
            {"user_text": "C" * 200, "assistant_text": "D" * 400},
        ]
        result = _build_cluster_excerpts(texts, max_chars=100)
        # Should have truncated — not all excerpts fit
        assert len(result) < 1200  # Much less than full text

    def test_skips_empty_chunks(self):
        from synapt.recall.clustering import _build_cluster_excerpts
        texts = [
            {"user_text": "", "assistant_text": ""},
            {"user_text": "real question", "assistant_text": "real answer"},
        ]
        result = _build_cluster_excerpts(texts)
        assert "[2]" in result
        assert "[1]" not in result


class TestUpgradeLargeClusterSummaries:
    """Tests for upgrade_large_cluster_summaries() — size-based LLM upgrades."""

    def test_upgrades_large_clusters(self, tmp_path):
        """Clusters with enough chunks get LLM summaries."""
        db = _make_db(tmp_path)
        # Create a cluster with 6 chunks (above min_chunks=5)
        chunk_ids = [f"c{i}" for i in range(6)]
        _seed_cluster_with_chunks(db, "clust-big", chunk_ids, topic="flock locking")

        with patch("synapt.recall.clustering.create_summary_client", return_value="fake"), \
             patch("synapt.recall.clustering.generate_llm_summary", return_value="LLM summary."):
            from synapt.recall.clustering import upgrade_large_cluster_summaries
            count = upgrade_large_cluster_summaries(db, min_chunks=5, max_upgrades=5)

        assert count == 1
        stored = db.get_cluster_summary("clust-big")
        assert stored is not None
        assert stored["method"] == "llm"
        db.close()

    def test_skips_small_clusters(self, tmp_path):
        """Clusters below min_chunks are not upgraded."""
        db = _make_db(tmp_path)
        _seed_cluster_with_chunks(db, "clust-small", ["c1", "c2"], topic="tiny")

        with patch("synapt.recall.clustering.create_summary_client", return_value="fake"), \
             patch("synapt.recall.clustering.generate_llm_summary", return_value="Summary."):
            from synapt.recall.clustering import upgrade_large_cluster_summaries
            count = upgrade_large_cluster_summaries(db, min_chunks=5, max_upgrades=5)

        assert count == 0
        db.close()

    def test_skips_clusters_with_existing_llm_summary(self, tmp_path):
        """Clusters that already have LLM summaries are not re-upgraded."""
        db = _make_db(tmp_path)
        chunk_ids = [f"c{i}" for i in range(6)]
        _seed_cluster_with_chunks(db, "clust-big", chunk_ids, topic="flock")
        db.save_cluster_summary("clust-big", "Existing LLM summary.", method="llm")

        with patch("synapt.recall.clustering.create_summary_client", return_value="fake"), \
             patch("synapt.recall.clustering.generate_llm_summary") as mock_gen:
            from synapt.recall.clustering import upgrade_large_cluster_summaries
            count = upgrade_large_cluster_summaries(db, min_chunks=5, max_upgrades=5)

        assert count == 0
        mock_gen.assert_not_called()
        db.close()

    def test_budget_cap(self, tmp_path):
        """Max upgrades per call is respected."""
        db = _make_db(tmp_path)
        for i in range(5):
            chunk_ids = [f"c{i}_{j}" for j in range(6)]
            _seed_cluster_with_chunks(db, f"clust-{i:03d}", chunk_ids, topic=f"topic {i}")

        with patch("synapt.recall.clustering.create_summary_client", return_value="fake"), \
             patch("synapt.recall.clustering.generate_llm_summary", return_value="Summary."):
            from synapt.recall.clustering import upgrade_large_cluster_summaries
            count = upgrade_large_cluster_summaries(db, min_chunks=5, max_upgrades=2)

        assert count == 2
        db.close()

    def test_returns_zero_when_no_backend(self, tmp_path):
        """Without any backend, returns 0 immediately."""
        db = _make_db(tmp_path)
        chunk_ids = [f"c{i}" for i in range(6)]
        _seed_cluster_with_chunks(db, "clust-big", chunk_ids)

        with patch("synapt.recall.clustering.create_summary_client", return_value=None):
            from synapt.recall.clustering import upgrade_large_cluster_summaries
            count = upgrade_large_cluster_summaries(db, min_chunks=5, max_upgrades=5)

        assert count == 0
        db.close()

    def test_largest_clusters_prioritized(self, tmp_path):
        """Clusters are upgraded largest-first."""
        db = _make_db(tmp_path)
        # 8-chunk cluster
        _seed_cluster_with_chunks(db, "clust-med", [f"m{i}" for i in range(8)], topic="medium")
        # 15-chunk cluster
        _seed_cluster_with_chunks(db, "clust-big", [f"b{i}" for i in range(15)], topic="biggest")
        # 5-chunk cluster
        _seed_cluster_with_chunks(db, "clust-sml", [f"s{i}" for i in range(5)], topic="small")

        upgraded_ids = []

        def track_summary(chunk_texts, topic, **kw):
            return f"Summary of {topic}."

        with patch("synapt.recall.clustering.create_summary_client", return_value="fake"), \
             patch("synapt.recall.clustering.generate_llm_summary", side_effect=track_summary):
            from synapt.recall.clustering import upgrade_large_cluster_summaries
            count = upgrade_large_cluster_summaries(db, min_chunks=5, max_upgrades=1)

        # Should only upgrade the biggest cluster (15 chunks)
        assert count == 1
        assert db.get_cluster_summary("clust-big")["method"] == "llm"
        assert db.get_cluster_summary("clust-med") is None  # Not upgraded
        db.close()


class TestSaveClustersPreservesLLM:
    """Tests that save_clusters() preserves LLM summaries across rebuilds."""

    def test_llm_summary_survives_rebuild(self, tmp_path):
        """LLM summaries for unchanged clusters survive save_clusters()."""
        db = _make_db(tmp_path)
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        # Insert a chunk so cluster_chunks can reference it
        db._conn.execute(
            "INSERT INTO chunks (id, session_id, timestamp, turn_index) "
            "VALUES ('c1', 'sess', ?, 0)", (now,)
        )
        db._conn.commit()

        # First build: save cluster + LLM summary
        clusters = [{
            "cluster_id": "clust-stable",
            "topic": "test",
            "cluster_type": "topic",
            "session_ids": [],
            "branch": None,
            "date_start": now,
            "date_end": now,
            "chunk_count": 1,
            "status": "active",
            "created_at": now,
            "updated_at": now,
        }]
        memberships = [("clust-stable", "c1", now)]
        db.save_clusters(clusters, memberships)
        db.save_cluster_summary("clust-stable", "LLM-generated text", method="llm")

        # Second build: same cluster ID (deterministic)
        db.save_clusters(clusters, memberships)

        # LLM summary should still be there
        stored = db.get_cluster_summary("clust-stable")
        assert stored is not None
        assert stored["method"] == "llm"
        assert stored["summary"] == "LLM-generated text"
        db.close()

    def test_concat_summary_deleted_on_rebuild(self, tmp_path):
        """Concat summaries are still cleared and regenerated on rebuild."""
        db = _make_db(tmp_path)
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        db._conn.execute(
            "INSERT INTO chunks (id, session_id, timestamp, turn_index) "
            "VALUES ('c1', 'sess', ?, 0)", (now,)
        )
        db._conn.commit()

        clusters = [{
            "cluster_id": "clust-test",
            "topic": "test",
            "cluster_type": "topic",
            "session_ids": [],
            "branch": None,
            "date_start": now,
            "date_end": now,
            "chunk_count": 1,
            "status": "active",
            "created_at": now,
            "updated_at": now,
        }]
        memberships = [("clust-test", "c1", now)]
        db.save_clusters(clusters, memberships)
        db.save_cluster_summary("clust-test", "Old concat text", method="concat")

        # Rebuild
        db.save_clusters(clusters, memberships)

        # Concat summary should be gone
        stored = db.get_cluster_summary("clust-test")
        assert stored is None
        db.close()

    def test_orphaned_llm_summaries_cleaned_on_rebuild(self, tmp_path):
        """LLM summaries for clusters whose ID changed are cleaned up."""
        db = _make_db(tmp_path)
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        # Insert two chunks
        db._conn.execute(
            "INSERT INTO chunks (id, session_id, timestamp, turn_index) "
            "VALUES ('c1', 'sess', ?, 0)", (now,)
        )
        db._conn.execute(
            "INSERT INTO chunks (id, session_id, timestamp, turn_index) "
            "VALUES ('c2', 'sess', ?, 1)", (now,)
        )
        db._conn.commit()

        # First build: cluster with c1 only
        clusters_v1 = [{
            "cluster_id": "clust-old",
            "topic": "test",
            "cluster_type": "topic",
            "session_ids": [],
            "branch": None,
            "date_start": now,
            "date_end": now,
            "chunk_count": 1,
            "status": "active",
            "created_at": now,
            "updated_at": now,
        }]
        db.save_clusters(clusters_v1, [("clust-old", "c1", now)])
        db.save_cluster_summary("clust-old", "LLM for old cluster", method="llm")

        # Second build: cluster membership changed → new ID
        clusters_v2 = [{
            "cluster_id": "clust-new",
            "topic": "test",
            "cluster_type": "topic",
            "session_ids": [],
            "branch": None,
            "date_start": now,
            "date_end": now,
            "chunk_count": 2,
            "status": "active",
            "created_at": now,
            "updated_at": now,
        }]
        db.save_clusters(clusters_v2, [("clust-new", "c1", now), ("clust-new", "c2", now)])

        # Old LLM summary should be cleaned up (orphaned)
        old = db.get_cluster_summary("clust-old")
        assert old is None, "Orphaned LLM summary should be deleted"

        # New cluster has no summary yet
        new = db.get_cluster_summary("clust-new")
        assert new is None
        db.close()
