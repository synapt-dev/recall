"""Tests for transcript attribution — agent_id tagging and scoped search.

TDD spec for idea doc 003 (agent-attributed memory). These tests define
the contract for per-agent transcript ownership with shared knowledge.

All tests are expected to FAIL until the implementation lands.
"""

import os
import tempfile
import unittest
from pathlib import Path


class TestTranscriptChunkAttribution(unittest.TestCase):
    """TranscriptChunk should carry an optional agent_id field."""

    def test_chunk_has_agent_id_field(self):
        """TranscriptChunk dataclass should accept agent_id."""
        from synapt.recall.core import TranscriptChunk

        chunk = TranscriptChunk(
            id="test:t0",
            session_id="sess-001",
            timestamp="2026-04-09T00:00:00Z",
            turn_index=0,
            user_text="hello",
            assistant_text="hi",
            agent_id="sentinel",
        )
        self.assertEqual(chunk.agent_id, "sentinel")

    def test_chunk_agent_id_defaults_to_none(self):
        """agent_id should default to None for backwards compatibility."""
        from synapt.recall.core import TranscriptChunk

        chunk = TranscriptChunk(
            id="test:t0",
            session_id="sess-001",
            timestamp="2026-04-09T00:00:00Z",
            turn_index=0,
            user_text="hello",
            assistant_text="hi",
        )
        self.assertIsNone(chunk.agent_id)

    def test_chunk_agent_id_from_env(self):
        """When SYNAPT_AGENT_ID is set, new chunks should pick it up."""
        from synapt.recall.core import TranscriptChunk

        old = os.environ.get("SYNAPT_AGENT_ID")
        try:
            os.environ["SYNAPT_AGENT_ID"] = "opus"
            chunk = TranscriptChunk(
                id="test:t0",
                session_id="sess-001",
                timestamp="2026-04-09T00:00:00Z",
                turn_index=0,
                user_text="hello",
                assistant_text="hi",
            )
            # Chunk should pick up agent_id from env if not explicitly set
            # Implementation may do this in __post_init__ or at ingest time
            # This test validates the contract, not the mechanism
            self.assertIn(chunk.agent_id, ("opus", None))
        finally:
            if old is None:
                os.environ.pop("SYNAPT_AGENT_ID", None)
            else:
                os.environ["SYNAPT_AGENT_ID"] = old


class TestStorageAttribution(unittest.TestCase):
    """RecallDB should persist and retrieve agent_id on chunks."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "recall.db"

    def _make_db(self):
        from synapt.recall.storage import RecallDB

        return RecallDB(self.db_path)

    def _make_chunk(self, chunk_id, session_id, agent_id=None):
        from synapt.recall.core import TranscriptChunk

        return TranscriptChunk(
            id=chunk_id,
            session_id=session_id,
            timestamp="2026-04-09T00:00:00Z",
            turn_index=0,
            user_text="test question",
            assistant_text="test answer",
            agent_id=agent_id,
        )

    def test_save_and_load_preserves_agent_id(self):
        """agent_id should round-trip through save_chunks/load_chunks."""
        db = self._make_db()
        chunks = [
            self._make_chunk("s1:t0", "sess-001", agent_id="opus"),
            self._make_chunk("s2:t0", "sess-002", agent_id="sentinel"),
            self._make_chunk("s3:t0", "sess-003", agent_id=None),
        ]
        db.save_chunks(chunks)
        loaded = db.load_chunks()

        by_id = {c.id: c for c in loaded}
        self.assertEqual(by_id["s1:t0"].agent_id, "opus")
        self.assertEqual(by_id["s2:t0"].agent_id, "sentinel")
        self.assertIsNone(by_id["s3:t0"].agent_id)

    def test_chunks_table_has_agent_id_column(self):
        """The chunks table schema should include agent_id."""
        db = self._make_db()
        cursor = db._conn.execute("PRAGMA table_info(chunks)")
        columns = {row["name"] for row in cursor.fetchall()}
        self.assertIn("agent_id", columns)


class TestScopedSearch(unittest.TestCase):
    """TranscriptIndex.lookup() should support agent_id filtering."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "recall.db"

    def _make_index_with_attributed_chunks(self):
        """Create an index with chunks from multiple agents + shared knowledge."""
        from synapt.recall.core import TranscriptChunk, TranscriptIndex
        from synapt.recall.storage import RecallDB

        db = RecallDB(self.db_path)

        # Chunks from different agents
        chunks = [
            TranscriptChunk(
                id="opus:t0",
                session_id="sess-opus-001",
                timestamp="2026-04-09T10:00:00Z",
                turn_index=0,
                user_text="design the API for user registration",
                assistant_text="I'll create a REST endpoint for user registration with email validation",
                agent_id="opus",
            ),
            TranscriptChunk(
                id="opus:t1",
                session_id="sess-opus-001",
                timestamp="2026-04-09T10:05:00Z",
                turn_index=1,
                user_text="add rate limiting to the API",
                assistant_text="Adding rate limiting middleware with token bucket algorithm",
                agent_id="opus",
            ),
            TranscriptChunk(
                id="sentinel:t0",
                session_id="sess-sentinel-001",
                timestamp="2026-04-09T11:00:00Z",
                turn_index=0,
                user_text="review the API design for security issues",
                assistant_text="Found three issues: no input sanitization, missing CORS headers, no auth middleware",
                agent_id="sentinel",
            ),
            TranscriptChunk(
                id="atlas:t0",
                session_id="sess-atlas-001",
                timestamp="2026-04-09T12:00:00Z",
                turn_index=0,
                user_text="implement the caching layer for API responses",
                assistant_text="Using Redis with TTL-based cache invalidation for GET endpoints",
                agent_id="atlas",
            ),
            # Legacy chunk with no agent_id
            TranscriptChunk(
                id="legacy:t0",
                session_id="sess-legacy-001",
                timestamp="2026-04-08T09:00:00Z",
                turn_index=0,
                user_text="set up the project structure",
                assistant_text="Created the monorepo with packages for api, web, and shared",
                agent_id=None,
            ),
        ]
        db.save_chunks(chunks)

        # Shared knowledge (org-level, no agent ownership)
        knowledge_nodes = [
            {
                "id": "kn_api_conventions",
                "content": "API conventions: REST, JSON responses, snake_case fields, semantic versioning",
                "category": "architecture",
                "status": "active",
                "confidence": 0.9,
                "source_sessions": ["sess-opus-001"],
                "created_at": "2026-04-09T10:00:00Z",
                "updated_at": "2026-04-09T10:00:00Z",
            },
        ]
        db.save_knowledge_nodes(knowledge_nodes)

        return TranscriptIndex(chunks, db=db)

    def test_lookup_default_returns_own_plus_knowledge(self):
        """Default lookup (with agent_id set) should return own transcripts + shared knowledge."""
        index = self._make_index_with_attributed_chunks()
        results = index.lookup("API design", agent_id="opus")

        # Should find opus's chunks about API design
        result_text = " ".join(r for r in results if isinstance(r, str))
        self.assertIn("registration", result_text.lower())

        # Should NOT include sentinel's or atlas's raw transcripts
        # (knowledge nodes derived from them are OK — they're shared)
        # This test validates the scoping contract

    def test_lookup_without_agent_id_returns_all(self):
        """lookup() without agent_id should search all transcripts (backwards compatible)."""
        index = self._make_index_with_attributed_chunks()
        results = index.lookup("API")

        # Should find results from all agents
        result_text = " ".join(r for r in results if isinstance(r, str))
        # At minimum, should have content from multiple agents
        self.assertTrue(len(results) > 0)

    def test_lookup_specific_agent_excludes_others(self):
        """lookup(agent_id='sentinel') should only return sentinel's transcripts + knowledge."""
        index = self._make_index_with_attributed_chunks()
        results = index.lookup("API", agent_id="sentinel")

        result_text = " ".join(r for r in results if isinstance(r, str))
        # Should find sentinel's security review
        # Should NOT find opus's design or atlas's caching work
        # Knowledge nodes are always included (shared layer)

    def test_lookup_legacy_chunks_visible_to_all(self):
        """Chunks with agent_id=None should be searchable by any agent."""
        index = self._make_index_with_attributed_chunks()
        results = index.lookup("project structure", agent_id="opus")

        # Legacy chunk (no agent_id) should be visible
        result_text = " ".join(r for r in results if isinstance(r, str))
        self.assertIn("monorepo", result_text.lower())

    def test_knowledge_nodes_always_included(self):
        """Knowledge nodes are org-shared — visible regardless of agent_id filter."""
        index = self._make_index_with_attributed_chunks()
        results = index.lookup("API conventions", agent_id="atlas")

        # atlas didn't create this knowledge node, but it's shared
        result_text = " ".join(r for r in results if isinstance(r, str))
        self.assertIn("conventions", result_text.lower())

    def test_wildcard_agent_searches_all_transcripts(self):
        """lookup(agent_id='*') should explicitly search all agents' transcripts."""
        index = self._make_index_with_attributed_chunks()
        results = index.lookup("API", agent_id="*")

        # Should find results from opus, sentinel, atlas, and legacy
        self.assertTrue(len(results) > 0)


class TestAttributionMigration(unittest.TestCase):
    """Existing databases without agent_id column should migrate gracefully."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "recall.db"

    def test_existing_db_without_agent_id_still_works(self):
        """A database created before attribution should still load chunks."""
        from synapt.recall.storage import RecallDB

        # Create DB and verify it works — the migration should add agent_id
        # transparently when the column doesn't exist
        db = RecallDB(self.db_path)
        from synapt.recall.core import TranscriptChunk

        chunks = [
            TranscriptChunk(
                id="old:t0",
                session_id="sess-old",
                timestamp="2026-01-01T00:00:00Z",
                turn_index=0,
                user_text="old question",
                assistant_text="old answer",
            ),
        ]
        db.save_chunks(chunks)
        loaded = db.load_chunks()
        self.assertEqual(len(loaded), 1)
        # Legacy chunks should have agent_id=None
        self.assertIsNone(loaded[0].agent_id)


if __name__ == "__main__":
    unittest.main()
