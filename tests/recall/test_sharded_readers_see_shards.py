"""In sharded mode the chunks live in the data shards; index.db keeps only a frozen pre-split copy.

Each reader below must see a session that exists ONLY in the shards (added after the split, the way every
session written since the split exists). Before the fix, each read index.db's stale chunks table through
ShardedRecallDB.__getattr__ or a RecallDB opened on live_store_path(), and missed it.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from synapt.recall.sharded_db import ShardedRecallDB
from synapt.recall.sharding import list_shards, split_monolithic_db
from synapt.recall.storage import RecallDB

LATE_SESSION = "s-after-split"
LATE_CHUNK = "c-after-split"


def _insert_chunk(conn: sqlite3.Connection, cid: str, session: str, ts: str, turn: int, text: str) -> None:
    conn.execute(
        "INSERT INTO chunks (id, session_id, timestamp, turn_index, user_text, assistant_text, tools_used, "
        "files_touched, tool_content, transcript_path, byte_offset, byte_length) "
        "VALUES (?, ?, ?, ?, ?, ?, '[]', '[]', '', '', 0, 0)",
        (cid, session, ts, turn, f"user {text}", f"assistant {text}"),
    )


def add_shard_only_session(d: Path) -> None:
    active = list_shards(d)[-1]
    conn = sqlite3.connect(str(active))
    _insert_chunk(conn, LATE_CHUNK, LATE_SESSION, "2025-03-01T10:00:00Z", 0, "late_only_in_shard")
    conn.commit()
    conn.close()


def make_sharded_store_with_a_shard_only_session(legacy_copy: bool, late: bool = True) -> Path:
    """Split a 25-chunk monolithic store, then add one session to the active shard only.

    legacy_copy=True models a store split by older code, which left the pre-split chunks table in index.db as a
    frozen copy that nothing writes. legacy_copy=False is today's split, which leaves index.db with no chunks table
    at all."""
    d = Path(tempfile.mkdtemp())
    db = RecallDB(d / "recall.db")
    for i in range(25):
        _insert_chunk(db._conn, f"c{i}", "s-before-split", f"2025-01-{(i % 28) + 1:02d}T10:00:00Z", i, f"early_{i}")
    db._conn.commit()
    db.close()
    split_monolithic_db(d, threshold=10)
    if legacy_copy:
        idx = sqlite3.connect(str(d / "index.db"))
        idx.execute(f"ATTACH DATABASE '{list_shards(d)[0]}' AS s1")
        idx.execute("CREATE TABLE chunks AS SELECT * FROM s1.chunks WHERE 0")
        for shard in list_shards(d):
            idx.execute(f"ATTACH DATABASE '{shard}' AS src")
            idx.execute("INSERT INTO chunks SELECT * FROM src.chunks")
            idx.commit()
            idx.execute("DETACH DATABASE src")
        idx.execute("DETACH DATABASE s1")
        idx.commit()
        idx.close()

    if not late:
        return d
    add_shard_only_session(d)
    active = list_shards(d)[-1]

    # Fixture checks: the late session is in a shard and NOT in index.db's legacy table.
    idx = sqlite3.connect(str(d / "index.db"))
    has_table = idx.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'chunks'").fetchone()[0] == 1
    assert has_table == legacy_copy
    if legacy_copy:
        assert idx.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 25
        assert idx.execute("SELECT COUNT(*) FROM chunks WHERE session_id = ?", (LATE_SESSION,)).fetchone()[0] == 0
    idx.close()
    sh = sqlite3.connect(str(active))
    assert sh.execute("SELECT COUNT(*) FROM chunks WHERE session_id = ?", (LATE_SESSION,)).fetchone()[0] == 1
    sh.close()
    return d


class TimelineReadsTheShards(unittest.TestCase):
    def test_timeline_session_info_sees_a_session_that_exists_only_in_the_shards(self):
        for legacy_copy in (True, False):
            with self.subTest(legacy_copy=legacy_copy):
                self._check(legacy_copy)

    def _check(self, legacy_copy: bool) -> None:
        from synapt.recall.timeline import _session_info

        d = make_sharded_store_with_a_shard_only_session(legacy_copy)
        db = ShardedRecallDB.open(d)
        try:
            sessions = {s["session_id"]: s for s in _session_info(db)}
        finally:
            db.close()

        self.assertIn(LATE_SESSION, sessions)
        self.assertEqual(sessions[LATE_SESSION]["chunk_count"], 1)
        self.assertEqual(sessions["s-before-split"]["chunk_count"], 25)


class ReadTurnContextReadsTheShards(unittest.TestCase):
    def test_read_turn_context_finds_a_chunk_written_to_a_shard_after_load(self):
        for legacy_copy in (True, False):
            with self.subTest(legacy_copy=legacy_copy):
                from synapt.recall.core import TranscriptIndex

                d = make_sharded_store_with_a_shard_only_session(legacy_copy, late=False)
                index = TranscriptIndex.load(d, use_embeddings=False)
                try:
                    self.assertNotIn(LATE_CHUNK, index._id_to_idx, "fixture check: an in-memory miss")
                    add_shard_only_session(d)
                    result = index.read_turn_context(LATE_CHUNK)
                finally:
                    if index._db is not None:
                        index._db.close()
                # Found in the shard: the fixture records no transcript offset, so the lookup reaches that branch.
                self.assertEqual(result, f"No transcript offset recorded for {LATE_CHUNK}.")


class ConsolidateReadsTheShards(unittest.TestCase):
    def test_resolve_loader_sees_a_session_that_exists_only_in_the_shards(self):
        from types import SimpleNamespace
        from unittest import mock

        from synapt.recall import consolidate

        for legacy_copy in (True, False):
            with self.subTest(legacy_copy=legacy_copy):
                d = make_sharded_store_with_a_shard_only_session(legacy_copy)
                kn = d / "knowledge.jsonl"
                kn.write_text("")
                node = SimpleNamespace(source_sessions=[LATE_SESSION, "s-before-split"])
                with mock.patch.object(consolidate, "project_index_dir", return_value=d), \
                        mock.patch.object(consolidate, "_knowledge_path", return_value=kn), \
                        mock.patch.object(consolidate, "read_nodes", return_value=[node]):
                    loaded = consolidate._load_chunks_for_resolve(d, include_text=True)
                self.assertIsNotNone(loaded)
                _, session_chunks, _, db = loaded
                db.close()
                self.assertIn(LATE_SESSION, session_chunks)
                self.assertIn("late_only_in_shard", session_chunks[LATE_SESSION][0][1])
                self.assertEqual(len(session_chunks["s-before-split"]), 25)

    def test_content_profile_sample_reads_the_shards(self):
        from synapt.recall import consolidate

        for legacy_copy in (True, False):
            with self.subTest(legacy_copy=legacy_copy):
                d = make_sharded_store_with_a_shard_only_session(legacy_copy)
                db = consolidate._open_chunk_store(d)
                try:
                    texts = db.sample_chunk_texts(limit=100)
                finally:
                    db.close()
                self.assertTrue(any("late_only_in_shard" in text for text in texts), texts)
                self.assertEqual(len(texts), 26)


if __name__ == "__main__":
    unittest.main()
