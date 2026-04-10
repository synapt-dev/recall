"""Tests for lazy embedding loading and numpy-backed embedding storage.

Verifies that:
1. Embeddings are NOT loaded during TranscriptIndex construction (lazy)
2. Embeddings ARE loaded on first lookup() call
3. Search results with numpy-backed embeddings match dict-backed results
4. get_all_embeddings_numpy() returns correct (matrix, rowids) format
5. embedding_search works with numpy matrix input
"""

import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from synapt.recall.storage import RecallDB, EMBEDDING_DIM


def _make_embedding(seed: float) -> list[float]:
    """Create a deterministic 384-dim embedding from a seed value."""
    # Simple repeating pattern for reproducibility
    return [seed * (1 + (i % 7) * 0.1) for i in range(EMBEDDING_DIM)]


def _pack_embedding(emb: list[float]) -> bytes:
    """Pack a float list into the binary format used by RecallDB."""
    fmt = f"{EMBEDDING_DIM}f"
    return struct.pack(fmt, *emb)


class TestGetAllEmbeddingsNumpy(unittest.TestCase):
    """Test the numpy-backed embedding loader in RecallDB."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "recall.db"
        self.db = RecallDB(self.db_path)

    def tearDown(self):
        self.db.close()

    def _insert_chunks_with_embeddings(self, n: int) -> list[int]:
        """Insert n chunks with embeddings, return rowids."""
        from synapt.recall.core import TranscriptChunk

        chunks = []
        for i in range(n):
            chunks.append(TranscriptChunk(
                id=f"chunk_{i}",
                session_id="test_session",
                timestamp=f"2026-01-01T00:{i:02d}:00Z",
                turn_index=i,
                user_text=f"user message {i}",
                assistant_text=f"assistant message {i}",
                tools_used=[],
                files_touched=[],
            ))
        self.db.save_chunks(chunks)

        # Add embeddings
        rowids = list(range(1, n + 1))
        emb_map = {r: _make_embedding(float(r)) for r in rowids}
        self.db.save_embeddings(emb_map)
        return rowids

    def test_returns_matrix_and_rowids(self):
        """get_all_embeddings_numpy returns (ndarray, list[int])."""
        rowids = self._insert_chunks_with_embeddings(5)
        matrix, result_rowids = self.db.get_all_embeddings_numpy()

        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(matrix.shape, (5, EMBEDDING_DIM))
        self.assertEqual(len(result_rowids), 5)
        # Rowids should match what we inserted
        self.assertEqual(sorted(result_rowids), sorted(rowids))

    def test_empty_db_returns_empty(self):
        """Empty database returns zero-size matrix."""
        matrix, rowids = self.db.get_all_embeddings_numpy()
        self.assertEqual(matrix.shape[0], 0)
        self.assertEqual(len(rowids), 0)

    def test_values_match_dict_loader(self):
        """Numpy loader produces identical values to dict loader."""
        self._insert_chunks_with_embeddings(10)
        dict_result = self.db.get_all_embeddings()
        matrix, rowids = self.db.get_all_embeddings_numpy()

        for i, rowid in enumerate(rowids):
            dict_emb = dict_result[rowid]
            np.testing.assert_allclose(
                matrix[i], dict_emb, rtol=1e-6,
                err_msg=f"Mismatch at rowid {rowid}",
            )

    def test_large_batch(self):
        """Handles hundreds of embeddings without error."""
        self._insert_chunks_with_embeddings(500)
        matrix, rowids = self.db.get_all_embeddings_numpy()
        self.assertEqual(matrix.shape, (500, EMBEDDING_DIM))
        self.assertEqual(len(rowids), 500)


class TestLazyEmbeddingLoading(unittest.TestCase):
    """Verify embeddings are loaded lazily, not during __init__."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "recall.db"
        self.db = RecallDB(self.db_path)

    def tearDown(self):
        self.db.close()

    def _make_index_with_embeddings(self, n: int = 5):
        """Create an index with chunks and embeddings in the DB."""
        from synapt.recall.core import TranscriptChunk, TranscriptIndex

        chunks = []
        for i in range(n):
            chunks.append(TranscriptChunk(
                id=f"chunk_{i}",
                session_id="test_session",
                timestamp=f"2026-01-01T00:{i:02d}:00Z",
                turn_index=i,
                user_text=f"user message {i}",
                assistant_text=f"assistant message {i}",
                tools_used=[],
                files_touched=[],
            ))
        self.db.save_chunks(chunks)

        # Add embeddings + mark as current
        rowids = list(range(1, n + 1))
        emb_map = {r: _make_embedding(float(r)) for r in rowids}
        self.db.save_embeddings(emb_map)

        # We need a content hash for the embeddings to be considered "current"
        index = TranscriptIndex(chunks, db=self.db, use_embeddings=False)
        content_hash = index._content_hash()
        self.db.set_metadata("embedding_hash", content_hash)

        return chunks

    def test_init_does_not_load_all_embeddings(self):
        """TranscriptIndex.__init__ should NOT call get_all_embeddings."""
        from synapt.recall.core import TranscriptIndex

        chunks = self._make_index_with_embeddings(5)

        with patch.object(
            self.db, "get_all_embeddings", wraps=self.db.get_all_embeddings
        ) as mock_get:
            index = TranscriptIndex(
                chunks, db=self.db, use_embeddings=True, lazy_chunks=True,
            )
            # The eager dict loader should NOT be called during init
            mock_get.assert_not_called()

    def test_embeddings_loaded_on_first_lookup(self):
        """Embeddings are loaded when lookup() is first called."""
        from synapt.recall.core import TranscriptIndex

        chunks = self._make_index_with_embeddings(5)
        index = TranscriptIndex(
            chunks, db=self.db, use_embeddings=True, lazy_chunks=True,
        )

        # Before lookup, embedding data should not be populated
        self.assertFalse(index._embeddings_loaded)

        # After lookup, embeddings should be loaded
        index.lookup("test query", max_chunks=3)
        self.assertTrue(index._embeddings_loaded)

    def test_stats_does_not_load_embeddings(self):
        """Non-search operations like stats() should not trigger embedding load."""
        from synapt.recall.core import TranscriptIndex

        chunks = self._make_index_with_embeddings(5)
        index = TranscriptIndex(
            chunks, db=self.db, use_embeddings=True, lazy_chunks=True,
        )

        # stats() is a read-only operation that shouldn't need embeddings
        stats = index.stats()
        self.assertIn("chunk_count", stats)
        self.assertFalse(index._embeddings_loaded)


class TestEmbeddingSearchNumpy(unittest.TestCase):
    """Test embedding_search with numpy matrix input."""

    def _make_test_data(self):
        """Create test embeddings as both dict and numpy formats."""
        emb_a = [1.0] + [0.0] * 383
        emb_b = [0.0, 1.0] + [0.0] * 382
        emb_c = [0.7, 0.7] + [0.0] * 382

        dict_embs = {1: emb_a, 2: emb_b, 3: emb_c}
        matrix = np.array([emb_a, emb_b, emb_c], dtype=np.float32)
        rowids = [1, 2, 3]

        return dict_embs, matrix, rowids

    def test_numpy_search_matches_dict_search(self):
        """embedding_search_numpy produces same results as dict-based search."""
        from synapt.recall.hybrid import embedding_search, embedding_search_numpy

        dict_embs, matrix, rowids = self._make_test_data()
        query = [0.8, 0.6] + [0.0] * 382

        dict_results = embedding_search(query, dict_embs, limit=10, threshold=0.0)
        numpy_results = embedding_search_numpy(
            query, matrix, rowids, limit=10, threshold=0.0,
        )

        # Same rowids in same order
        self.assertEqual(
            [r for r, _ in dict_results],
            [r for r, _ in numpy_results],
        )
        # Same similarities (within float tolerance)
        for (_, s1), (_, s2) in zip(dict_results, numpy_results):
            self.assertAlmostEqual(s1, s2, places=5)

    def test_numpy_search_empty_matrix(self):
        """Empty matrix returns empty results."""
        from synapt.recall.hybrid import embedding_search_numpy

        matrix = np.empty((0, 384), dtype=np.float32)
        query = [1.0] + [0.0] * 383
        results = embedding_search_numpy(query, matrix, [], limit=10, threshold=0.0)
        self.assertEqual(results, [])

    def test_numpy_search_threshold(self):
        """Threshold filters low-similarity results."""
        from synapt.recall.hybrid import embedding_search_numpy

        _, matrix, rowids = self._make_test_data()
        query = [1.0] + [0.0] * 383  # Exact match with row 0 (rowid=1)

        results = embedding_search_numpy(
            query, matrix, rowids, limit=10, threshold=0.9,
        )
        # Only the exact match should survive
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 1)


class TestShardedColdStartEmbeddingGate(unittest.TestCase):
    """Sharded indexes without chunk embeddings stay on the fast BM25 path."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index_dir = Path(self.tmpdir)

    def _make_chunk(self):
        from synapt.recall.core import TranscriptChunk

        return TranscriptChunk(
            id="s1:t0",
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            turn_index=0,
            user_text="cold start fact",
            assistant_text="assistant",
            tools_used=[],
            files_touched=[],
        )

    def test_sharded_lookup_skips_provider_when_chunk_embeddings_missing(self):
        from synapt.recall.core import TranscriptIndex

        index_db = RecallDB(self.index_dir / "index.db")
        data_db = RecallDB(self.index_dir / "data_001.db")
        data_db.save_chunks([self._make_chunk()])
        index_db.save_knowledge_nodes([
            {
                "id": "kn-1",
                "content": "cold start fact",
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
                "tags": [],
                "valid_from": None,
                "valid_until": None,
                "version": 1,
                "lineage_id": "",
            }
        ])
        index_db.save_knowledge_embeddings({1: _make_embedding(0.5)})
        index_db.close()
        data_db.close()

        with patch("synapt.recall.embeddings.get_embedding_provider") as mock_get:
            index = TranscriptIndex.load(self.index_dir, use_embeddings=True)
            result = index.lookup("cold start fact", max_chunks=3, max_tokens=200)

        self.assertIn("cold start fact", result)
        self.assertIsNone(index._embed_provider)
        self.assertIn("BM25-only", index._embedding_reason)
        mock_get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
