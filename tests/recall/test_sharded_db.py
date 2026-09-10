"""Tests for ShardedRecallDB — tree-structured storage wrapper."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.resume import BoundedResumeIndex, build_resume_view
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

    def test_open_readonly_uses_query_only_connection(self):
        writer = RecallDB(self.index_dir / "recall.db")
        writer.close()

        db = ShardedRecallDB.open_readonly(self.index_dir)
        self.assertTrue(db.is_monolithic)
        with self.assertRaises(Exception):
            db._index._conn.execute("DELETE FROM chunks")
        db.close()

    def test_open_readonly_skips_schema_work(self):
        RecallDB(self.index_dir / "recall.db").close()

        with mock.patch.object(
            RecallDB,
            "_ensure_schema",
            side_effect=AssertionError("read-only open ran schema work"),
        ):
            db = ShardedRecallDB.open_readonly(self.index_dir)
            self.assertEqual(db.chunk_count(), 0)
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
        agent_id: str | None = None,
    ) -> TranscriptChunk:
        return TranscriptChunk(
            id=chunk_id,
            session_id=session_id,
            timestamp=timestamp,
            turn_index=0,
            user_text=text,
            assistant_text="assistant",
            agent_id=agent_id,
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

    def test_open_readonly_detects_sharded_layout(self):
        self._create_two_shard_layout().close()

        db = ShardedRecallDB.open_readonly(self.index_dir)
        self.assertFalse(db.is_monolithic)
        self.assertEqual(db.shard_count, 2)
        self.assertEqual(db.chunk_count(), 2)
        db.close()

    def test_is_monolithic_is_false_for_a_sharded_store_with_zero_shards(self):
        """index.db present with no data shard file yet -- the shape
        split_monolithic_db leaves behind when splitting an empty
        recall.db -- is a genuinely sharded layout, not monolithic. Prior
        code inferred monolithic-ness from ``not self._data_dbs``, which
        an empty (but sharded) store also satisfies."""
        RecallDB(self.index_dir / "index.db").close()
        db = ShardedRecallDB.open(self.index_dir)
        self.assertEqual(db.shard_count, 0)  # setup check: genuinely zero shards
        self.assertFalse(db.is_monolithic)
        db.close()

    def test_bounded_session_reads_merge_across_shards(self):
        RecallDB(self.index_dir / "index.db").close()
        first = RecallDB(self.index_dir / "data_001.db")
        second = RecallDB(self.index_dir / "data_002.db")
        first.save_chunks([
            self._make_chunk("shared:t0", "shared", "2026-01-01T00:00:00Z", "first"),
        ])
        second.save_chunks([
            TranscriptChunk(
                id="shared:t1",
                session_id="shared",
                timestamp="2026-01-02T00:00:00Z",
                turn_index=1,
                user_text="second",
                assistant_text="assistant",
            ),
        ])
        first.close()
        second.close()

        db = ShardedRecallDB.open_readonly(self.index_dir)
        try:
            self.assertIn("shared", db.session_activity())
            self.assertEqual(
                [chunk.id for chunk in db.load_session_chunks("shared")],
                ["shared:t0", "shared:t1"],
            )
        finally:
            db.close()

    def test_session_overview_unions_agent_identity_across_shards(self):
        RecallDB(self.index_dir / "index.db").close()
        first = RecallDB(self.index_dir / "data_001.db")
        second = RecallDB(self.index_dir / "data_002.db")
        first.save_chunks([
            self._make_chunk(
                "shared:t0",
                "shared",
                "2026-01-01T00:00:00Z",
                "claude runtime",
                agent_id="runtime-a",
            ),
        ])
        second.save_chunks([
            self._make_chunk(
                "shared:t1",
                "shared",
                "2026-01-02T00:00:00Z",
                "codex runtime",
                agent_id="runtime-b",
            ),
        ])
        first.close()
        second.close()

        db = ShardedRecallDB.open_readonly(self.index_dir)
        try:
            self.assertEqual(
                db.session_overview()["shared"]["agent_ids"],
                frozenset({"runtime-a", "runtime-b"}),
            )
        finally:
            db.close()

    def test_session_overview_excludes_journal_identity_from_data_shard(self):
        RecallDB(self.index_dir / "index.db").close()
        shard = RecallDB(self.index_dir / "data_001.db")
        journal = self._make_chunk(
            "journal:j0",
            "journal-only",
            "2026-02-01T00:00:00Z",
            "journal metadata",
            agent_id="ambient-agent",
        )
        journal.turn_index = -1
        shard.save_chunks([journal])
        shard.close()

        db = ShardedRecallDB.open_readonly(self.index_dir)
        try:
            overview = db.session_overview()["journal-only"]
            self.assertEqual(overview["agent_ids"], frozenset())
            self.assertFalse(overview["has_real_activity"])
        finally:
            db.close()

    def test_session_overview_unions_query_tail_agent_identity_with_base(self):
        RecallDB(self.index_dir / "index.db").close()
        shard = RecallDB(self.index_dir / "data_001.db")
        shard.save_chunks([
            self._make_chunk(
                "shared:t0",
                "shared",
                "2026-01-01T00:00:00Z",
                "base",
                agent_id="claude-agent",
            ),
        ])
        shard.close()
        index = RecallDB(self.index_dir / "index.db")
        index.replace_query_tail(
            source_key="runtime-source",
            session_id="shared",
            rewind_offset=1,
            chunks=[
                self._make_chunk(
                    "shared:t1",
                    "shared",
                    "2026-01-02T00:00:00Z",
                    "fresh overlay",
                    agent_id="codex-agent",
                )
            ],
            cursor={
                "transcript_path": "/runtime/shared.jsonl",
                "observed_complete_offset": 2,
                "rewind_offset": 1,
                "rewind_turn_index": 1,
                "source_size": 2,
                "source_mtime_ns": 2,
                "observed_prefix_sha256": "digest",
                "suppresses_base": False,
                "latest_projected_timestamp": "2026-01-02T00:00:00Z",
                "last_attempt_at": "2026-01-02T00:00:00Z",
                "last_success_at": "2026-01-02T00:00:00Z",
            },
        )
        index.close()

        db = ShardedRecallDB.open_readonly(self.index_dir)
        try:
            self.assertEqual(
                db.session_overview()["shared"]["agent_ids"],
                frozenset({"claude-agent", "codex-agent"}),
            )
        finally:
            db.close()

    def test_bounded_resume_prefers_newest_agent_session_from_query_tail(self):
        RecallDB(self.index_dir / "index.db").close()
        shard = RecallDB(self.index_dir / "data_001.db")
        shard.save_chunks([
            self._make_chunk(
                "old:t0",
                "old-session",
                "2026-01-01T00:00:00Z",
                "old base continuity",
                agent_id="stable-agent",
            ),
        ])
        shard.close()
        index_db = RecallDB(self.index_dir / "index.db")
        index_db.replace_query_tail(
            source_key="new-runtime-source",
            session_id="new-session",
            rewind_offset=0,
            chunks=[
                self._make_chunk(
                    "new:t0",
                    "new-session",
                    "2026-02-01T00:00:00Z",
                    "new query-tail continuity",
                    agent_id="stable-agent",
                )
            ],
            cursor={
                "transcript_path": "/runtime/new-session.jsonl",
                "observed_complete_offset": 1,
                "rewind_offset": 0,
                "rewind_turn_index": 0,
                "source_size": 1,
                "source_mtime_ns": 1,
                "observed_prefix_sha256": "digest",
                "suppresses_base": False,
                "latest_projected_timestamp": "2026-02-01T00:00:00Z",
                "last_attempt_at": "2026-02-01T00:00:00Z",
                "last_success_at": "2026-02-01T00:00:00Z",
            },
        )
        index_db.close()

        index = BoundedResumeIndex(ShardedRecallDB.open_readonly(self.index_dir))
        try:
            view = build_resume_view(
                index,
                agent_id="stable-agent",
                journal_path=None,
            )
        finally:
            index.close()

        self.assertEqual(view.session_id, "new-session")
        self.assertEqual(view.turns[0].user_text, "new query-tail continuity")

    def test_query_tail_journal_metadata_does_not_outrank_real_agent_activity(self):
        RecallDB(self.index_dir / "index.db").close()
        shard = RecallDB(self.index_dir / "data_001.db")
        shard.save_chunks([
            self._make_chunk(
                "real:t0",
                "real-session",
                "2026-01-01T00:00:00Z",
                "real continuity",
                agent_id="foreign-agent",
            ),
        ])
        shard.close()
        index_db = RecallDB(self.index_dir / "index.db")
        journal = self._make_chunk(
            "journal:j0",
            "journal-only",
            "2026-02-01T00:00:00Z",
            "journal metadata",
            agent_id="stable-agent",
        )
        journal.turn_index = -1
        journal.assistant_text = ""
        index_db.replace_query_tail(
            source_key="journal-overlay",
            session_id="journal-only",
            rewind_offset=0,
            chunks=[journal],
            cursor={
                "transcript_path": "/runtime/journal.jsonl",
                "observed_complete_offset": 1,
                "rewind_offset": 0,
                "rewind_turn_index": -1,
                "source_size": 1,
                "source_mtime_ns": 1,
                "observed_prefix_sha256": "digest",
                "suppresses_base": False,
                "latest_projected_timestamp": "2026-02-01T00:00:00Z",
                "last_attempt_at": "2026-02-01T00:00:00Z",
                "last_success_at": "2026-02-01T00:00:00Z",
            },
        )
        index_db.close()

        index = BoundedResumeIndex(ShardedRecallDB.open_readonly(self.index_dir))
        try:
            self.assertNotIn(
                "stable-agent",
                index._db.session_overview()["journal-only"]["agent_ids"],
            )
            view = build_resume_view(
                index,
                caller_sources=[],
                agent_id="stable-agent",
                journal_path=None,
            )
        finally:
            index.close()

        self.assertEqual(view.session_id, "real-session")
        self.assertEqual(view.turns[0].user_text, "real continuity")

    def test_journal_in_a_later_shard_does_not_replace_real_activity(self):
        RecallDB(self.index_dir / "index.db").close()
        first = RecallDB(self.index_dir / "data_001.db")
        second = RecallDB(self.index_dir / "data_002.db")
        first.save_chunks([
            self._make_chunk("dead:t0", "dead", "2026-01-01T00:00:00Z", "old"),
            self._make_chunk("live:t0", "live", "2026-01-02T00:00:00Z", "new"),
        ])
        second.save_chunks([
            TranscriptChunk(
                id="dead:journal",
                session_id="dead",
                timestamp="2026-08-25T00:00:00Z",
                turn_index=-1,
                user_text="journal",
                assistant_text="",
            ),
        ])
        first.close()
        second.close()

        db = ShardedRecallDB.open_readonly(self.index_dir)
        try:
            activity = db.session_activity()
            self.assertGreater(activity["live"], activity["dead"])
        finally:
            db.close()

    def test_multiple_shards(self):
        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        RecallDB(self.index_dir / "data_002.db").close()
        RecallDB(self.index_dir / "data_003.db").close()

        db = ShardedRecallDB.open(self.index_dir)
        self.assertEqual(db.shard_count, 3)
        db.close()

    def test_save_chunks_creates_fresh_shards(self):
        """Sharded save_chunks creates fresh shard(s) for the data."""
        from synapt.recall.core import TranscriptChunk
        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        db = ShardedRecallDB.open(self.index_dir)
        chunk = TranscriptChunk(
            id="test:t0", session_id="s1", timestamp="2025-04-15T10:00:00Z",
            turn_index=0, user_text="hello", assistant_text="hi",
        )
        db.save_chunks([chunk])
        # 1 chunk = 1 shard
        self.assertEqual(db.shard_count, 1)
        self.assertEqual(db.chunk_count(), 1)
        db.close()

    def test_save_chunks_empty_is_noop(self):
        """Saving empty list to sharded DB doesn't crash."""
        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        db = ShardedRecallDB.open(self.index_dir)
        db.save_chunks([])
        db.close()

    def test_save_chunks_routes_through_rebuild_when_a_sharded_store_has_zero_shards(self):
        """A genuinely sharded store (index.db present) with NO existing
        shard file at all -- the shape ``split_monolithic_db`` leaves behind
        when splitting an empty ``recall.db`` -- must still route through
        the atomic generation rebuild, not silently fall through to the
        monolithic branch because ``self._data_dbs`` happens to be empty.
        Reproduced against the prior code: seeding only ``index.db`` left
        ``save_chunks()`` writing chunks into ``index.db`` itself and never
        publishing a generation, self-perpetuating on every later call
        since data never lands in a shard."""
        RecallDB(self.index_dir / "index.db").close()
        db = ShardedRecallDB.open(self.index_dir)
        self.assertEqual(db.shard_count, 0)  # setup check: genuinely zero shards
        chunk = TranscriptChunk(
            id="test:t0", session_id="s1", timestamp="2025-04-15T10:00:00Z",
            turn_index=0, user_text="hello", assistant_text="hi",
        )
        db.save_chunks([chunk])
        from synapt.recall.generations import read_current_generation
        self.assertIsNotNone(
            read_current_generation(self.index_dir),
            "save_chunks on a zero-shard sharded store must publish a "
            "generation, not silently write into index.db",
        )
        db.close()

    def test_save_chunks_still_uses_monolithic_branch_for_a_true_monolithic_store(self):
        """Control: a genuinely monolithic store (no index.db at all) must
        still take the monolithic branch -- the fix narrows what counts as
        'sharded' to the on-disk layout, it must not treat every store as
        sharded."""
        db = ShardedRecallDB.open(self.index_dir)  # no index.db -> monolithic
        self.assertTrue(db.is_monolithic)
        chunk = TranscriptChunk(
            id="test:t0", session_id="s1", timestamp="2025-04-15T10:00:00Z",
            turn_index=0, user_text="hello", assistant_text="hi",
        )
        db.save_chunks([chunk])
        self.assertEqual(db.chunk_count(), 1)
        from synapt.recall.generations import generations_root
        self.assertFalse(
            generations_root(self.index_dir).exists(),
            "a monolithic store must never grow a generations/ directory",
        )
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

    def test_content_hash_spans_all_shards_in_global_timestamp_order(self):
        db = self._create_two_shard_layout()
        import hashlib

        h = hashlib.sha256()
        h.update("s2:t0|beta memory|assistant|\n".encode())
        h.update("s1:t0|alpha memory|assistant|\n".encode())
        self.assertEqual(db.content_hash(), h.hexdigest()[:16])
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

    def test_save_chunks_reshards_properly(self):
        """save_chunks publishes a fresh generation and redistributes into it;
        the prior generation's chunks are no longer visible through open()."""
        from synapt.recall.core import TranscriptChunk
        from synapt.recall.generations import current_generation_dir
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

        # Now do a full rebuild save (like rescrub does)
        new_chunk = TranscriptChunk(
            id="new:t0", session_id="s2", timestamp="2025-06-01T00:00:00Z",
            turn_index=0, user_text="new", assistant_text="fresh",
        )
        db.save_chunks([new_chunk])

        # The rebuild publishes a fresh generation; 1 chunk = 1 fresh shard
        # inside it, and CURRENT now names that generation (not the old
        # flat pre-generation layout the seeded old_chunk lived in).
        self.assertEqual(db.shard_count, 1)
        gen_dir = current_generation_dir(self.index_dir)
        self.assertIsNotNone(gen_dir, "save_chunks must publish a generation")
        self.assertTrue(gen_dir.is_dir())

        # Active shard should have the new chunk
        self.assertEqual(db.chunk_count(), 1)

        # FTS search should only find the new chunk
        results = db.fts_search("stale")
        self.assertEqual(len(results), 0, "Stale data should not appear in FTS")
        results = db.fts_search("fresh")
        self.assertEqual(len(results), 1)
        db.close()

    def test_save_chunks_no_unbounded_shard_growth(self):
        """A single rebuild call must not accumulate duplicate/stale shard
        files WITHIN the generation it publishes (the sprint-13 bug: a
        rebuild used to create one new full-copy shard per call, in place,
        without ever removing the previous one). Repeating the rebuild must
        keep landing on exactly the right shard count each time.

        Cross-generation accumulation (old, superseded generations staying
        on disk once a new one publishes) is a different, disclosed, and
        still-open concern -- generation garbage collection is explicit
        follow-up scope, not yet wired to this path. This test's contract
        is narrower and still fully meaningful under the new generation
        system: it is the CURRENT generation, after any number of rebuilds,
        that must never show duplication.
        """
        from synapt.recall.core import TranscriptChunk
        from synapt.recall.generations import current_generation_dir
        from synapt.recall.sharding import list_shards

        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        db = ShardedRecallDB.open(self.index_dir)

        chunks = [
            TranscriptChunk(
                id=f"c{i}:t0", session_id=f"s{i}",
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
                turn_index=0, user_text=f"chunk {i}", assistant_text="a",
            )
            for i in range(5)
        ]

        # Simulate 3 consecutive rebuilds (the bug created a new shard each time)
        for rebuild in range(3):
            db.save_chunks(chunks)

        current_gen = current_generation_dir(self.index_dir)
        self.assertIsNotNone(current_gen)
        shard_files = list_shards(current_gen)
        self.assertEqual(
            len(shard_files), 1,
            f"Expected 1 shard in the current generation after rebuilds, "
            f"got {len(shard_files)}: {[p.name for p in shard_files]}",
        )
        self.assertEqual(db.chunk_count(), 5)
        db.close()

    def test_save_chunks_splits_at_threshold(self):
        """Chunks exceeding threshold are split across multiple shards."""
        from synapt.recall.core import TranscriptChunk
        from synapt.recall.generations import current_generation_dir
        from synapt.recall.sharding import list_shards
        import synapt.recall.sharding as _sharding_mod

        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        db = ShardedRecallDB.open(self.index_dir)

        chunks = [
            TranscriptChunk(
                id=f"c{i}:t0", session_id=f"s{i}",
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
                turn_index=0, user_text=f"chunk {i}", assistant_text="a",
            )
            for i in range(25)
        ]

        # Temporarily lower the threshold for testing
        original = _sharding_mod.SHARD_CHUNK_THRESHOLD
        _sharding_mod.SHARD_CHUNK_THRESHOLD = 10
        try:
            db.save_chunks(chunks)
        finally:
            _sharding_mod.SHARD_CHUNK_THRESHOLD = original

        # 25 chunks / 10 per shard = 3 shards
        self.assertEqual(db.shard_count, 3)
        self.assertEqual(db.chunk_count(), 25)

        current_gen = current_generation_dir(self.index_dir)
        self.assertIsNotNone(current_gen)
        shard_files = list_shards(current_gen)
        self.assertEqual(len(shard_files), 3)
        db.close()


class TestShardedRecallDBSessionOverviewCache(unittest.TestCase):
    """R3.1: the generation-keyed ``session_overview()`` cache (design
    tracked privately; see recall#1147 Probe 2 for the public trail).

    Four freeze witnesses, each isolated so a targeted mutation kills only
    its own test: a cache HIT is actually consulted (not merely correct by
    coincidence), a MISS on generation bump, a MISS on schema-version
    bump, and BYTE-IDENTITY of the cached round trip against the uncached
    result. The hit/miss witnesses poison whatever cache row the code
    itself just wrote (discovered by reading the table back), rather than
    assuming a key shape, so each stays a witness of its own dimension
    only -- see the per-witness comments below for the isolation argument.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index_dir = Path(self.tmpdir)

    def _make_chunk(
        self, chunk_id: str, session_id: str, timestamp: str, turn_index: int, text: str
    ) -> TranscriptChunk:
        return TranscriptChunk(
            id=chunk_id,
            session_id=session_id,
            timestamp=timestamp,
            turn_index=turn_index,
            user_text=text,
            assistant_text="assistant",
        )

    def _open_generation_backed_store(self) -> ShardedRecallDB:
        """A single-shard store that publishes a real generation on its
        first ``save_chunks`` call -- required for the cache to activate
        at all (no generation identity means no caching, ruled
        2026-09-09; see ``test_save_chunks_routes_through_rebuild_when_a_
        sharded_store_has_zero_shards`` above for the same zero-shard
        starting layout)."""
        RecallDB(self.index_dir / "index.db").close()
        RecallDB(self.index_dir / "data_001.db").close()
        return ShardedRecallDB.open(self.index_dir)

    def _cache_rows(self, db: ShardedRecallDB) -> list:
        """Raw introspection of the cache table -- discovers whatever key
        the code actually used rather than assuming one, so a mutation
        that changes WHICH key is used doesn't also break this helper."""
        return db._index._conn.execute(
            "SELECT generation_name, shard_name, schema_version, overview_json "
            "FROM shard_overview_cache"
        ).fetchall()

    def _poisoned_overview(self, sentinel_turn_count: int) -> dict:
        return {
            "s1": {
                "activity": (1, "poisoned"),
                "earliest_ts": "2020-01-01T00:00:00Z",
                "latest_ts": "2020-01-01T00:00:00Z",
                "turn_count": sentinel_turn_count,
                "has_real_activity": True,
                "transcript_path": "",
                "agent_ids": frozenset(),
            }
        }

    def test_cache_hit_is_actually_consulted_not_recomputed(self):
        """Proves the SECOND call reads the cache rather than merely
        recomputing the (identical, so indistinguishable) right answer:
        after the first call populates the cache, this poisons the exact
        row that call wrote with a sentinel no real computation could
        produce, and asserts the sentinel comes back.

        Mutation that kills only this witness: make ``_shard_overview``'s
        ``if cached is not None: return cached`` branch dead code. That
        mutation cannot affect the other three witnesses here -- (b)/(c)
        assert the sentinel is ABSENT after a bump, which stays true
        trivially if the cache is never read at all; (d) calls the
        storage-layer cache functions directly, never through
        ``_shard_overview``.
        """
        db = self._open_generation_backed_store()
        db.save_chunks([
            self._make_chunk("s1:t0", "s1", "2026-01-01T00:00:00Z", 0, "alpha"),
        ])
        first = db.session_overview()
        self.assertEqual(first["s1"]["turn_count"], 1)

        rows = self._cache_rows(db)
        self.assertEqual(len(rows), 1, "exactly one shard, one cache row expected")
        gen_name, shard_name, schema_version, _ = rows[0]
        db._index.set_cached_shard_overview(
            gen_name, shard_name, schema_version, self._poisoned_overview(424242)
        )

        second = db.session_overview()
        self.assertEqual(
            second["s1"]["turn_count"], 424242,
            "a real recompute would never produce this sentinel -- if this "
            "fails, the second call is not reading the cache",
        )
        db.close()

    def test_cache_misses_on_generation_bump(self):
        """A second ``save_chunks`` call always publishes a FRESH
        generation (``generations.rebuild_and_publish`` mints a new name
        every call -- see ``save_chunks``'s own docstring), so the second
        ``session_overview()`` here must reflect the NEW content, not a
        stale blob cached under the first generation's name.

        Mutation that kills only this witness: hardcode the
        ``generation_name`` argument ``_shard_overview`` passes to the
        cache methods to a fixed literal, ignoring the real one. That
        cannot affect (a) (single generation throughout, so a fixed key
        is still self-consistent across its two calls), (c) (schema_version
        is a different key column, handled independently), or (d) (calls
        the storage layer directly with its own explicit keys).
        """
        db = self._open_generation_backed_store()
        db.save_chunks([
            self._make_chunk("s1:t0", "s1", "2026-01-01T00:00:00Z", 0, "v1"),
        ])
        v1 = db.session_overview()
        self.assertEqual(v1["s1"]["turn_count"], 1)

        db.save_chunks([
            self._make_chunk("s1:t0", "s1", "2026-01-01T00:00:00Z", 0, "v1"),
            self._make_chunk("s1:t1", "s1", "2026-01-02T00:00:00Z", 1, "v2"),
        ])
        v2 = db.session_overview()
        self.assertEqual(
            v2["s1"]["turn_count"], 2,
            "a stale generation-A cache value leaked forward after the bump",
        )
        self.assertEqual(v2["s1"]["latest_ts"], "2026-01-02T00:00:00Z")
        db.close()

    def test_cache_misses_on_schema_version_bump(self):
        """Poisons the real cache row the first call wrote, then bumps
        ``SHARD_OVERVIEW_CACHE_SCHEMA_VERSION`` (module-level constant,
        re-imported fresh on every ``session_overview()`` call since the
        import is inside the function body, not at module load) and
        confirms the poisoned, now-wrong-schema-version row is NOT read.

        Mutation that kills only this witness: hardcode the
        ``schema_version`` argument ``_shard_overview`` passes to the
        cache methods to a fixed literal, ignoring the real (patched)
        one. That cannot affect (a) (single schema version throughout its
        two calls), (b) (generation_name is a different key column), or
        (d) (calls the storage layer directly with its own explicit keys).
        """
        db = self._open_generation_backed_store()
        db.save_chunks([
            self._make_chunk("s1:t0", "s1", "2026-01-01T00:00:00Z", 0, "alpha"),
        ])
        real = db.session_overview()
        self.assertEqual(real["s1"]["turn_count"], 1)

        rows = self._cache_rows(db)
        self.assertEqual(len(rows), 1)
        gen_name, shard_name, schema_version, _ = rows[0]
        db._index.set_cached_shard_overview(
            gen_name, shard_name, schema_version, self._poisoned_overview(999999)
        )

        with mock.patch(
            "synapt.recall.storage.SHARD_OVERVIEW_CACHE_SCHEMA_VERSION",
            schema_version + 1,
        ):
            bumped = db.session_overview()
        self.assertEqual(
            bumped["s1"]["turn_count"], 1,
            "a schema-version bump must not read the old-version cached blob",
        )
        db.close()

    def test_cached_round_trip_is_byte_identical_to_uncached(self):
        """The cache write/read round trip must return a value that is
        not merely JSON-equal but TYPE-identical to the uncached result:
        ``activity`` is compared with plain ``max()`` across shards in
        ``session_overview()``, and comparing a tuple against a list
        raises ``TypeError`` -- a real bug this witness would catch that
        a plain ``==`` on the two dicts alone would not, since Python
        does not consider a shape difference here unless you ask.

        Mutation that kills only this witness: drop the ``tuple(...)``
        (or ``frozenset(...)``) cast in ``_deserialize_shard_overview``,
        leaving that field a list. The other three witnesses only assert
        on ``turn_count``/``latest_ts`` (plain ints/strings, unaffected by
        this), so this mutation cannot touch them.
        """
        from synapt.recall.storage import SHARD_OVERVIEW_CACHE_SCHEMA_VERSION

        db = self._open_generation_backed_store()
        db.save_chunks([
            self._make_chunk("s1:t0", "s1", "2026-01-01T00:00:00Z", 0, "alpha"),
            self._make_chunk("s2:t0", "s2", "2026-01-02T00:00:00Z", 0, "beta"),
        ])
        shard = db._data_dbs[0]
        uncached = shard.session_overview()
        self.assertGreaterEqual(len(uncached), 1)

        db._index.set_cached_shard_overview(
            "gen-byte-identity-probe", shard.path.name,
            SHARD_OVERVIEW_CACHE_SCHEMA_VERSION, uncached,
        )
        round_tripped = db._index.get_cached_shard_overview(
            "gen-byte-identity-probe", shard.path.name,
            SHARD_OVERVIEW_CACHE_SCHEMA_VERSION,
        )
        self.assertEqual(round_tripped, uncached)
        for session_id, entry in round_tripped.items():
            self.assertIsInstance(
                entry["activity"], tuple,
                f"{session_id}: activity must round-trip as a tuple ("
                "max() compares it against other shards' tuples)",
            )
            self.assertIsInstance(
                entry["agent_ids"], frozenset,
                f"{session_id}: agent_ids must round-trip as a frozenset",
            )
        db.close()


if __name__ == "__main__":
    unittest.main()
