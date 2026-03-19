"""Tests for ShardedRecallDB — tree-structured storage wrapper."""

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
