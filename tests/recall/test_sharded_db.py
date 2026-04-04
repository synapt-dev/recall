"""Tests for ShardedRecallDB — tree-structured storage wrapper."""

import tempfile
import unittest
from pathlib import Path

from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.sharded_db import ShardedRecallDB
from synapt.recall.storage import RecallDB


class TestShardedRecallDBMonolithic(unittest.TestCase):
    """Test ShardedRecallDB in monolithic mode (single recall.db)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index_dir = Path(self.tmpdir)

    def test_open_creates_recall_db(self):
        db = ShardedRecallDB.open(self.index_dir)
        self.assertTrue(db.is_monolithic)
        self.assertEqual(db.shard_count, 0)
        db.close()

    def test_open_existing_recall_db(self):
        # Create a monolithic DB first
        RecallDB(self.index_dir / "recall.db").close()
        db = ShardedRecallDB.open(self.index_dir)
        self.assertTrue(db.is_monolithic)
        db.close()

    def test_knowledge_roundtrip(self):
        db = ShardedRecallDB.open(self.index_dir)
        node = {
            "id": "test-node",
            "content": "test fact",
            "category": "workflow",
            "confidence": 0.9,
            "source_sessions": [],
            "source_turns": [],
            "source_offsets": [],
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "status": "active",
            "superseded_by": "",
            "contradiction_note": "",
            "tags": "",
            "valid_from": None,
            "valid_until": None,
            "version": 1,
            "lineage_id": "",
        }
        db.save_knowledge_nodes([node])
        nodes = db.load_knowledge_nodes()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["content"], "test fact")
        db.close()

    def test_pending_contradictions(self):
        db = ShardedRecallDB.open(self.index_dir)
        self.assertEqual(db.pending_contradiction_count(), 0)
        db.close()

    def test_passthrough_via_getattr(self):
        """Unhandled methods fall through to index DB."""
        db = ShardedRecallDB.open(self.index_dir)
        # load_manifest is explicitly delegated, but _path is on RecallDB
        self.assertIsNotNone(db._path)
        db.close()

    def test_close_is_safe(self):
        db = ShardedRecallDB.open(self.index_dir)
        db.close()
        # Double close shouldn't crash
        db.close()


class TestShardedRecallDBSharded(unittest.TestCase):
    """Test ShardedRecallDB with index.db + data shards."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index_dir = Path(self.tmpdir)

    def _make_chunk(
        self,
        chunk_id: str,
        session_id: str,
        timestamp: str,
        text: str,
    ) -> TranscriptChunk:
        return TranscriptChunk(
            id=chunk_id,
            session_id=session_id,
            timestamp=timestamp,
            turn_index=0,
            user_text=text,
            assistant_text="assistant",
        )

    def _create_two_shard_layout(self) -> ShardedRecallDB:
        RecallDB(self.index_dir / "index.db").close()
        first = RecallDB(self.index_dir / "data_001.db")
        second = RecallDB(self.index_dir / "data_002.db")
        first.save_chunks([
            self._make_chunk("s1:t0", "s1", "2026-01-01T00:00:00Z", "alpha memory"),
        ])
        second.save_chunks([
            self._make_chunk("s2:t0", "s2", "2026-01-02T00:00:00Z", "beta memory"),
        ])
        first.close()
        second.close()
        return ShardedRecallDB.open(self.index_dir)

    def test_open_detects_sharded_layout(self):
        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()

        db = ShardedRecallDB.open(self.index_dir)
        self.assertFalse(db.is_monolithic)
        self.assertEqual(db.shard_count, 1)
        db.close()

    def test_multiple_shards(self):
        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        RecallDB(self.index_dir / "data_002.db").close()
        RecallDB(self.index_dir / "data_003.db").close()

        db = ShardedRecallDB.open(self.index_dir)
        self.assertEqual(db.shard_count, 3)
        db.close()

    def test_save_chunks_to_active_shard(self):
        """Sharded save_chunks writes to the active (last) shard."""
        from synapt.recall.core import TranscriptChunk
        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        db = ShardedRecallDB.open(self.index_dir)
        chunk = TranscriptChunk(
            id="test:t0", session_id="s1", timestamp="2025-04-15T10:00:00Z",
            turn_index=0, user_text="hello", assistant_text="hi",
        )
        db.save_chunks([chunk])
        # Chunk saved to active shard
        active = db._data_dbs[-1]
        count = active._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        self.assertEqual(count, 1)
        db.close()

    def test_save_chunks_empty_is_noop(self):
        """Saving empty list to sharded DB doesn't crash."""
        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        db = ShardedRecallDB.open(self.index_dir)
        db.save_chunks([])
        db.close()

    def test_chunk_count_spans_all_shards(self):
        db = self._create_two_shard_layout()
        self.assertEqual(db.chunk_count(), 2)
        db.close()

    def test_chunk_id_rowid_map_is_globally_unique(self):
        db = self._create_two_shard_layout()
        mapping = db.get_chunk_id_rowid_map()
        self.assertEqual(set(mapping.keys()), {"s1:t0", "s2:t0"})
        self.assertEqual(len(set(mapping.values())), 2)
        self.assertNotEqual(mapping["s1:t0"], mapping["s2:t0"])
        db.close()

    def test_fts_search_returns_shard_qualified_rowids(self):
        db = self._create_two_shard_layout()
        hits = db.fts_search("memory", limit=10)
        self.assertEqual(len(hits), 2)
        self.assertEqual(len({rowid for rowid, _ in hits}), 2)
        self.assertTrue(all(rowid > (1 << 32) for rowid, _ in hits))
        db.close()

    def test_fts_search_raw_returns_shard_qualified_rowids(self):
        db = self._create_two_shard_layout()
        hits = db.fts_search_raw("memory", limit=10)
        self.assertEqual(len(hits), 2)
        self.assertEqual(len({rowid for rowid, _ in hits}), 2)
        self.assertTrue(all(rowid > (1 << 32) for rowid, _ in hits))
        db.close()

    def test_get_all_embeddings_uses_shard_qualified_rowids(self):
        db = self._create_two_shard_layout()
        mapping = db.get_chunk_id_rowid_map()
        emb1 = [0.1] * 384
        emb2 = [0.2] * 384
        db.save_embeddings({
            mapping["s1:t0"]: emb1,
            mapping["s2:t0"]: emb2,
        })
        loaded = db.get_all_embeddings()
        self.assertEqual(set(loaded.keys()), set(mapping.values()))
        self.assertAlmostEqual(loaded[mapping["s1:t0"]][0], emb1[0], places=6)
        self.assertAlmostEqual(loaded[mapping["s2:t0"]][0], emb2[0], places=6)
        db.close()

    def test_transcript_index_load_can_search_sharded_chunks(self):
        self._create_two_shard_layout().close()
        index = TranscriptIndex.load(self.index_dir)
        result = index.lookup("beta", max_chunks=5, max_tokens=200)
        self.assertIn("beta memory", result)
        self.assertNotEqual(index._rowid_to_idx, {})
        self.assertIsNone(index._bm25)

    def test_load_chunk_by_rowid_uses_shard_qualified_ids(self):
        db = self._create_two_shard_layout()
        mapping = db.get_chunk_id_rowid_map()
        chunk = db.load_chunk_by_rowid(mapping["s2:t0"])
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.id, "s2:t0")
        loaded = db.load_chunks_by_rowids([mapping["s1:t0"], mapping["s2:t0"]])
        self.assertEqual(set(loaded.keys()), {mapping["s1:t0"], mapping["s2:t0"]})
        self.assertEqual(loaded[mapping["s1:t0"]].id, "s1:t0")
        self.assertEqual(loaded[mapping["s2:t0"]].id, "s2:t0")

    def test_sample_chunk_texts_reads_across_shards(self):
        db = self._create_two_shard_layout()
        samples = db.sample_chunk_texts(limit=10)
        joined = " ".join(samples)
        self.assertIn("alpha memory", joined)
        self.assertIn("beta memory", joined)
        db.close()

    def test_save_chunks_clears_all_shards_on_rebuild(self):
        """Full save_chunks clears stale data from ALL shards, not just active."""
        from synapt.recall.core import TranscriptChunk
        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        RecallDB(self.index_dir / "data_002.db").close()
        db = ShardedRecallDB.open(self.index_dir)
        self.assertEqual(db.shard_count, 2)

        # Seed shard 1 with old data
        old_chunk = TranscriptChunk(
            id="old:t0", session_id="s1", timestamp="2025-01-01T00:00:00Z",
            turn_index=0, user_text="old", assistant_text="stale",
        )
        db._data_dbs[0].save_chunks([old_chunk])
        count_before = db._data_dbs[0]._conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
        self.assertEqual(count_before, 1)

        # Now do a full rebuild save (like rescrub does)
        new_chunk = TranscriptChunk(
            id="new:t0", session_id="s2", timestamp="2025-06-01T00:00:00Z",
            turn_index=0, user_text="new", assistant_text="fresh",
        )
        db.save_chunks([new_chunk])

        # Shard 1 should be cleared (the bug: it kept the old chunk)
        count_shard1 = db._data_dbs[0]._conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
        self.assertEqual(count_shard1, 0, "Old shard should be cleared on rebuild")

        # Active shard should have the new chunk
        count_active = db._data_dbs[-1]._conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
        self.assertEqual(count_active, 1)

        # FTS search should only find the new chunk, not the old one
        results = db.fts_search("stale")
        self.assertEqual(len(results), 0, "Stale data should not appear in FTS")
        results = db.fts_search("fresh")
        self.assertEqual(len(results), 1)
        db.close()


if __name__ == "__main__":
    unittest.main()
