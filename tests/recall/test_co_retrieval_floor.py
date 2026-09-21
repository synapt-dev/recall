"""One similarity floor gates BOTH the co-retrieval queue insert and the
search banner (companion fix to the public recall#683 issue).

Cold-install finding (2026-09-21, first-time-agent run on the published
package): any two facts returned together were queued as a contradiction and
the search result carried a loud "Conflicting information detected" block —
two unrelated facts tripped it on a BM25-only store. The detector's candidate
set (same category, low token overlap) survives as the pre-filter; on top of
it, a pair proceeds ONLY when its embedding cosine clears
CO_RETRIEVAL_SIMILARITY_FLOOR (0.40 — the measured bands from the
knowledge-semantic work: unrelated pair 0.293, related pair 0.416, the same
0.4 constant the keyword-hit coverage gate uses). Below the floor, or on a
store with no embeddings, the pair is neither queued nor bannered. Strictly a
reduction of what gets flagged — no new heuristic. Explicit contradictions
(the contradict tool's own inserts, consolidation-detected rows) are
untouched paths.
"""

import math
from unittest.mock import patch


def _fake_vectors(pairs: dict[str, list[float]]):
    """A fake embedding provider with hand-set vectors, calibrated to the
    #1203 bands: vectors whose pairwise cosine we control exactly."""
    class _P:
        def embed(self, texts):
            out = []
            for t in texts:
                for key, vec in pairs.items():
                    if key in t:
                        out.append(vec)
                        break
                else:
                    out.append([0.0])
            return out

        def embed_single(self, t):
            for key, vec in pairs.items():
                if key in t:
                    return vec
            return [0.0]

    return _P()


# Fake vectors in a shared space (384-dim, the storage layer's fixed width;
# the signal lives in the first two coordinates so the pairwise cosines are
# the numbers the #1203 calibration used): A and B are the same subject
# paraphrased (cosine ≈ 0.95, above the floor); U is unrelated to both
# (cos(A,U) ≈ −0.05, cos(B,U) ≈ 0.287 — below the floor).
_DIM = 384

def _pad(x: list[float]) -> list[float]:
    return x + [0.0] * (_DIM - len(x))

VEC_A = _pad([1.0, 0.0])
VEC_B = _pad([0.95, 0.312])
VEC_U = _pad([-0.05, 1.0])


def _cos(u, v):
    num = sum(a * b for a, b in zip(u, v))
    den = math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v))
    return num / den if den else 0.0


class TestCoRetrievalFloor:
    def test_unrelated_pair_no_provider_no_row_no_banner(self, tmp_path):
        """The stranger's exact case: two unrelated facts, no embedding
        provider — nothing queued, nothing bannered."""
        from synapt.recall.core import project_index_dir
        from synapt.recall.server import recall_save, recall_search
        from synapt.recall.storage import RecallDB

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            recall_save(content="The gateway uses Postgres 14 on port 5433", category="fact", node_id="g1")
            recall_save(content="The invoice pipeline writes to S3 bucket acme nightly", category="fact", node_id="i1")
            recall_search("gateway invoice")

        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            assert db.list_pending_contradictions() == []
        finally:
            db.close()

    def test_unrelated_pair_with_embeddings_below_floor_queues_nothing(self, tmp_path):
        """cos(B,U) ≈ 0.287, far below the 0.40 floor: NOTHING queued — even
        though the store HAS embeddings. (Mutation target: drop the floor
        from the queue insert and this reds while the banner witness below
        stays green.)"""
        from synapt.recall.core import project_index_dir
        from synapt.recall.server import recall_save, recall_search
        from synapt.recall.storage import RecallDB

        provider = _fake_provider_with_db_embeddings({
            "Postgres": VEC_A, "invoice": VEC_U,
        })

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=provider), \
             patch("synapt.recall.server._invalidate_cache"):
            recall_save(content="The gateway uses Postgres 14 on port 5433", category="fact", node_id="g1")
            recall_save(content="The invoice pipeline writes to S3 acme nightly", category="fact", node_id="i1")
            recall_search("gateway invoice")

        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            assert db.list_pending_contradictions() == []
        finally:
            db.close()

    def test_unrelated_pair_with_embeddings_below_floor_banners_nothing(self, tmp_path):
        """The same below-floor pair: the search result carries no banner
        (the banner's own floor check)."""
        from synapt.recall.server import recall_save, recall_search

        provider = _fake_provider_with_db_embeddings({
            "Postgres": VEC_A, "invoice": VEC_U,
        })

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=provider), \
             patch("synapt.recall.server._invalidate_cache"):
            recall_save(content="The gateway uses Postgres 14 on port 5433", category="fact", node_id="g1")
            recall_save(content="The invoice pipeline writes to S3 acme nightly", category="fact", node_id="i1")
            result = recall_search("gateway invoice")

        assert "Conflicting information" not in result, result

    def test_floor_clearing_pair_is_queued_and_bannered(self, tmp_path):
        """A/B paraphrase pair, cosine ≈ 0.95 ≥ 0.40: row queued AND the
        search result carries the banner."""
        from synapt.recall.core import project_index_dir
        from synapt.recall.server import recall_save, recall_search
        from synapt.recall.storage import RecallDB

        provider = _fake_provider_with_db_embeddings({
            "Postgres": VEC_A, "Sixteen": VEC_B,
        })

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=provider), \
             patch("synapt.recall.server._invalidate_cache"):
            recall_save(content="The gateway uses Postgres 14 on port 5433", category="fact", node_id="g1")
            recall_save(content="The gateway migrated to Postgres Sixteen last week", category="fact", node_id="g2")
            result = recall_search("gateway Postgres")

        assert "Conflicting information detected" in result, result
        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            pending = db.list_pending_contradictions()
            assert len(pending) == 1, pending
        finally:
            db.close()

    def test_explicit_contradict_still_queues_untouched(self, tmp_path):
        """Control: the explicit contradict tool's own insert is untouched by
        the co-retrieval floor."""
        from synapt.recall.core import project_index_dir, TranscriptIndex
        from synapt.recall.server import recall_save, recall_contradict
        from synapt.recall.storage import RecallDB

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            recall_save(content="Fact one about zeta", category="fact", node_id="e1")
            recall_save(content="Fact two about zeta", category="fact", node_id="e2")

        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            index = TranscriptIndex.__new__(TranscriptIndex)
            index._db = db
            index.chunks = []
            index.sessions = {}
            with patch("synapt.recall.server._get_index", return_value=index):
                res = recall_contradict(
                    action="flag", claim="Fact one about zeta is wrong",
                    old_node_id="e1", reason="test",
                )
            assert "flagged" in res.lower(), res
            pending = [p for p in db.list_pending_contradictions()
                       if p.get("old_node_id") == "e1"]
            assert pending, f"explicit contradict did not queue: {pending}"
            assert all(p.get("detected_by") != "co-retrieval" for p in pending)
        finally:
            db.close()


def _fake_provider_with_db_embeddings(vecs_by_marker: dict[str, list[float]]):
    """A provider whose embed_single keys off content markers, so the saved
    nodes get the intended vectors through the real save path."""
    class _P:
        def embed(self, texts):
            return [self._vec(t) for t in texts]

        def embed_single(self, text):
            return self._vec(text)

        def _vec(self, text):
            for marker, vec in vecs_by_marker.items():
                if marker in text:
                    return vec
            return [0.0]

    return _P()

class TestCoRetrievalFloorSharded:
    """R2 B2: the floor witnesses must also hold through the ShardedRecallDB
    proxy — the detector reaches get_knowledge_embeddings_by_ids via
    __getattr__ fall-through, and the detector's try/except would turn a
    broken proxy path into a silent 'no conflicts'."""

    def test_floor_clearing_pair_through_sharded_db(self, tmp_path):
        from synapt.recall.core import TranscriptIndex
        from synapt.recall.sharded_db import ShardedRecallDB
        from synapt.recall.storage import RecallDB

        index_dir = tmp_path / "index"
        index_dir.mkdir()
        # Build the store on a plain RecallDB, then reopen through the
        # sharded wrapper (open() auto-detects the monolithic layout).
        plain = RecallDB(index_dir / "recall.db")
        nodes = [
            {"id": "s1", "content": "deploy on tuesday", "category": "workflow",
             "status": "active", "confidence": 0.6, "lineage_id": ""},
            {"id": "s2", "content": "never use feature flags", "category": "workflow",
             "status": "active", "confidence": 0.9, "lineage_id": ""},
        ]
        plain.save_knowledge_nodes([dict(n) for n in nodes])
        for nid, vec in (("s1", _pad([1.0, 0.0])), ("s2", _pad([0.95, 0.312]))):
            rowid = plain.get_knowledge_rowid(nid)
            assert rowid is not None
            plain.save_knowledge_embeddings({rowid: vec})
        plain.close()

        sharded = ShardedRecallDB.open(index_dir)
        try:
            index = TranscriptIndex.__new__(TranscriptIndex)
            index._db = sharded
            index.chunks = []
            index.sessions = {}
            detected = index._detect_co_retrieval_conflicts(nodes)
            assert len(detected) == 1, detected
            old, new = detected[0]
            assert old["id"] == "s1" and new["id"] == "s2"
            pending = sharded.list_pending_contradictions()
            assert len(pending) == 1, pending
        finally:
            sharded.close()


class TestWithinBatchDedup:
    def test_three_floor_clearing_nodes_queue_twice_not_three_times(self, tmp_path):
        """Three co-retrieved floor-clearing same-category nodes yield the
        row count the pre-floor loop gave: pair (n1,n2) queues (old=n1),
        pair (n1,n3) is within-batch-deduped (n1 already pending), pair
        (n2,n3) queues (old=n2) — two rows, never the same old node twice."""
        from synapt.recall.core import project_index_dir, TranscriptIndex
        from synapt.recall.server import recall_save, recall_search
        from synapt.recall.storage import RecallDB

        provider = _fake_provider_with_db_embeddings({
            "alpha": _pad([1.0, 0.0]),
            "beta": _pad([0.95, 0.312]),
            "gamma": _pad([0.9, 0.436]),  # cos ≈ 0.95 with both
        })

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=provider), \
             patch("synapt.recall.server._invalidate_cache"):
            recall_save(content="the alpha scheduler writes metrics hourly", category="fact", node_id="n1")
            recall_save(content="nightly beta jobs purge the temp tables", category="fact", node_id="n2")
            recall_save(content="gamma dashboards aggregate queue depth per region", category="fact", node_id="n3")
            recall_search("alpha beta gamma")

        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            pending = db.list_pending_contradictions()
            old_ids = [p["old_node_id"] for p in pending]
            assert len(pending) == 2, f"expected 2 rows for 3 nodes, got {len(pending)}: {old_ids}"
            assert len(set(old_ids)) == len(old_ids), f"same old node queued twice: {old_ids}"
        finally:
            db.close()
