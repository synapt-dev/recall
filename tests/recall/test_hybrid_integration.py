"""Integration tests for hybrid RRF search through the TranscriptIndex.

Tests the full flow: intent classification → parameter adjustment → BM25/FTS
+ embedding search → RRF fusion → recency decay → threshold → results.

These tests complement test_hybrid.py (unit tests for pure functions) by
verifying the integration through core.py lookup methods.
"""

from __future__ import annotations

import pytest

from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.hybrid import SPARSE_RESULT_THRESHOLD
from synapt.recall.storage import RecallDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(n: int = 6) -> list[TranscriptChunk]:
    """Create chunks with distinct topics for testing hybrid retrieval.

    Chunks 0-1: About "database schema migration"
    Chunks 2-3: About "authentication token refresh"
    Chunks 4-5: About "deployment pipeline configuration"
    """
    topics = [
        ("database schema migration",
         "We redesigned the schema to normalize the user table."),
        ("database migration rollback",
         "Added a rollback script for the schema migration."),
        ("authentication token refresh",
         "The OAuth token refresh uses a sliding window expiry."),
        ("auth session management",
         "Sessions are stored in Redis with a 24-hour TTL."),
        ("deployment pipeline configuration",
         "The CI/CD pipeline runs lint, test, build, and deploy stages."),
        ("deploy rollback procedure",
         "To rollback a deploy, revert the Docker image tag in k8s."),
    ]
    chunks = []
    for i, (user_text, assistant_text) in enumerate(topics):
        session = f"session-{i // 2:03d}"
        chunks.append(TranscriptChunk(
            id=f"{session}:t{i % 2}",
            session_id=session,
            timestamp=f"2026-03-01T{10 + i}:00:00Z",
            turn_index=i % 2,
            user_text=user_text,
            assistant_text=assistant_text,
        ))
    return chunks


def _make_embeddings(chunks: list[TranscriptChunk]) -> dict[int, list[float]]:
    """Create synthetic embeddings that cluster by topic.

    Uses 8-dim vectors for simplicity. Each topic cluster gets a distinct
    direction so cosine similarity reflects topical relatedness.
    """
    topic_vecs = [
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # chunk 0: database
        [0.8, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],  # chunk 1: database
        [0.0, 0.9, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],  # chunk 2: auth
        [0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],  # chunk 3: auth
        [0.0, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.0],  # chunk 4: deploy
        [0.0, 0.0, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0],  # chunk 5: deploy
    ]
    # rowid is 1-indexed in SQLite
    return {i + 1: vec for i, vec in enumerate(topic_vecs)}


class MockEmbeddingProvider:
    """Fake embedding provider that returns known vectors for test queries."""

    def __init__(self):
        self._query_map = {
            "database schema": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "how does the login session work": [0.0, 0.85, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            "shipping code to production": [0.0, 0.0, 0.88, 0.0, 0.0, 0.1, 0.0, 0.0],
            "user data access": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        self._default = [0.0] * 8

    @property
    def dim(self) -> int:
        return 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._query_map.get(t, self._default) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        return self._query_map.get(text, self._default)


def _build_index(tmp_path, chunks=None) -> TranscriptIndex:
    """Build a TranscriptIndex with FTS5 backend and injected embeddings."""
    if chunks is None:
        chunks = _make_chunks()

    db = RecallDB(tmp_path / "test.db")
    db.save_chunks(chunks)

    index = TranscriptIndex(chunks, db=db, use_embeddings=False)
    index._refresh_rowid_map()

    # Inject mock embeddings
    provider = MockEmbeddingProvider()
    index._embed_provider = provider
    index._all_embeddings = _make_embeddings(chunks)

    return index


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestHybridRRFIntegration:
    """Test that RRF fusion surfaces results BM25 missed."""

    def test_keyword_query_returns_bm25_matches(self, tmp_path):
        """BM25 matches should appear even without embeddings."""
        index = _build_index(tmp_path)
        # "database schema" has direct keyword matches in chunks 0-1
        result = index.lookup("database schema", max_chunks=5)
        assert result  # non-empty string
        assert "schema" in result.lower()

    def test_semantic_query_surfaces_embedding_matches(self, tmp_path):
        """Paraphrased query with no keyword overlap should find results via embeddings."""
        index = _build_index(tmp_path)
        # "shipping code to production" has no keyword overlap with
        # "deployment pipeline configuration" but embedding is close
        result = index.lookup("shipping code to production", max_chunks=5)
        assert result
        assert "deploy" in result.lower() or "pipeline" in result.lower()

    def test_rrf_merges_both_signals(self, tmp_path):
        """RRF should merge BM25 and embedding results."""
        index = _build_index(tmp_path)
        # "user data access" is between database and auth in embedding space
        result = index.lookup("user data access", max_chunks=5)
        assert result  # Should get results from hybrid merge

    def test_no_embeddings_falls_back_to_bm25(self, tmp_path):
        """Without embeddings, search should still work (BM25 only)."""
        index = _build_index(tmp_path)
        index._embed_provider = None
        index._all_embeddings = {}
        result = index.lookup("database schema", max_chunks=5)
        assert result
        assert "schema" in result.lower()


class TestSparseAutoBoost:
    """Test SPARSE_RESULT_THRESHOLD auto-boost behavior."""

    def test_sparse_bm25_triggers_embedding_boost(self, tmp_path):
        """When BM25 returns few results, embedding weight should auto-increase."""
        index = _build_index(tmp_path)
        # "login session" has weak BM25 match but strong embedding match
        result = index.lookup("how does the login session work", max_chunks=5)
        # Should return results via the embedding auto-boost
        assert result

    def test_threshold_boundary_exact(self):
        """Exactly SPARSE_RESULT_THRESHOLD results = no auto-boost."""
        from synapt.recall.hybrid import weighted_rrf_merge

        # Exactly at threshold (3 results) — no boost
        bm25 = [(0, 5.0), (1, 3.0), (2, 1.0)]
        emb = [(3, 0.9), (4, 0.8)]
        merged_at = weighted_rrf_merge(bm25, emb, emb_weight=0.5)

        # Below threshold (2 results) — auto-boost to 2.0
        bm25_sparse = [(0, 5.0), (1, 3.0)]
        merged_sparse = weighted_rrf_merge(bm25_sparse, emb, emb_weight=0.5)

        def get_score(merged, item_id):
            for id_, score in merged:
                if id_ == item_id:
                    return score
            return 0.0

        # Item 3 (embedding-only) scores higher with sparse boost
        score_3_at = get_score(merged_at, 3)
        score_3_sparse = get_score(merged_sparse, 3)
        assert score_3_sparse > score_3_at

    def test_threshold_boundary_no_boost_at_exact(self):
        """At exactly SPARSE_RESULT_THRESHOLD, emb_weight stays as passed."""
        from synapt.recall.hybrid import weighted_rrf_merge

        # 3 results = exactly at threshold, emb_weight=0.5 should NOT be boosted
        bm25 = [(0, 5.0), (1, 3.0), (2, 1.0)]
        emb = [(3, 0.9)]

        merged = weighted_rrf_merge(bm25, emb, emb_weight=0.5)

        # Item 3 score = emb_weight * 1/(k+1) = 0.5 * 1/61 ≈ 0.00820
        score_3 = next(s for id_, s in merged if id_ == 3)
        expected = 0.5 / 61
        assert abs(score_3 - expected) < 1e-10


class TestIntentParameterFlow:
    """Test that intent classification adjusts search parameters."""

    def test_debug_query_has_short_half_life(self):
        """Debug queries should use shorter half_life for recency emphasis."""
        from synapt.recall.hybrid import classify_query_intent, intent_search_params
        intent = classify_query_intent("why did the build fail yesterday")
        assert intent == "debug"
        params = intent_search_params(intent)
        assert params["half_life"] < 30

    def test_factual_query_boosts_knowledge(self):
        """Factual queries should have high knowledge_boost."""
        from synapt.recall.hybrid import classify_query_intent, intent_search_params
        intent = classify_query_intent("what is the database port")
        assert intent == "factual"
        params = intent_search_params(intent)
        assert params["knowledge_boost"] >= 3.0

    def test_exploratory_query_boosts_embeddings(self):
        """Exploratory queries should increase emb_weight."""
        from synapt.recall.hybrid import classify_query_intent, intent_search_params
        intent = classify_query_intent("what did we try for caching")
        assert intent == "exploratory"
        params = intent_search_params(intent)
        assert params["emb_weight"] >= 2.0

    def test_lookup_uses_intent_params(self, tmp_path):
        """Verify that lookup() passes intent params to the lookup methods."""
        index = _build_index(tmp_path)
        result = index.lookup("what is the database schema", max_chunks=5)
        assert result
        assert "schema" in result.lower()

    def test_lookup_with_all_intents(self, tmp_path):
        """Verify lookup works with all intent types without errors."""
        index = _build_index(tmp_path)
        queries = {
            "factual": "what is the database port",
            "debug": "error in the deployment pipeline",
            "exploratory": "how did we solve the auth problem",
            "procedural": "how to deploy the service",
            "general": "authentication token refresh",
        }
        for intent_name, query in queries.items():
            result = index.lookup(query, max_chunks=3)
            assert isinstance(result, str), f"Failed for intent: {intent_name}"


class TestEmbeddingSearchEdgeCases:
    """Test edge cases in the embedding search integration."""

    def test_mismatched_dimensions_safe(self):
        """embedding_search with dimension mismatch should not crash."""
        from synapt.recall.hybrid import embedding_search
        # 8-dim query vs 4-dim embeddings — zip truncates silently
        query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        embs = {1: [1.0, 0.0, 0.0, 0.0]}  # 4-dim
        results = embedding_search(query, embs, threshold=0.0)
        assert len(results) >= 0  # No crash

    def test_balanced_weights_ranking_order(self):
        """With equal weights, verify item seen in both lists ranks highest."""
        from synapt.recall.hybrid import weighted_rrf_merge
        bm25 = [(10, 5.0), (20, 3.0), (30, 1.0)]
        emb = [(20, 0.9), (30, 0.8), (10, 0.7)]
        merged = weighted_rrf_merge(bm25, emb, bm25_weight=1.0, emb_weight=1.0)
        # All items appear in both lists. RRF scores:
        # 10: 1/(61) + 1/(63) = 0.01639 + 0.01587 = 0.03226
        # 20: 1/(62) + 1/(61) = 0.01613 + 0.01639 = 0.03252  (best combined rank)
        # 30: 1/(63) + 1/(62) = 0.01587 + 0.01613 = 0.03200
        assert merged[0][0] == 20  # Best combined rank
        assert merged[1][0] == 10
        assert merged[2][0] == 30

    def test_empty_index_returns_empty(self, tmp_path):
        """Lookup on empty index returns empty string."""
        db = RecallDB(tmp_path / "empty.db")
        index = TranscriptIndex([], db=db, use_embeddings=False)
        result = index.lookup("anything", max_chunks=5)
        assert result == ""
