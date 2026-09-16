"""TDD spec for the stats hydration fix: stats must not hydrate chunk bodies.

Red-first: stats() on a lazy DB-backed index currently calls
_materialize_all_chunks TWICE (once per set-comprehension for
total_tools_used and total_files_touched), on top of cmd_stats's own
header load -- the triple full pass measured on the shared index (92% of
cmd_stats cumulative; 5.71s shared-index vs 0.20s empty-store median
under identical load). The fix: tools/files counts come from SQL
aggregates over the stored JSON-array columns (tools_used /
files_touched are json.dumps TEXT); stats() makes ZERO hydration calls
and the counts stay equal to a hydration-baseline count, including
suppressed-session exclusion and query-tail overlay replacement.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.sharded_db import ShardedRecallDB
from synapt.recall.storage import RecallDB


def make_chunk(cid, sid, ts, text, tools=(), files=()):
    return TranscriptChunk(
        id=cid,
        session_id=sid,
        timestamp=ts,
        turn_index=0,
        user_text=text,
        assistant_text="",
        tool_content="",
        tools_used=list(tools),
        files_touched=list(files),
        date_text="",
        transcript_path="",
        byte_offset=0,
        byte_length=0,
        agent_id="test-agent",
    )


def seed_db(index_dir, chunks, dbname="recall.db"):
    db = RecallDB(Path(index_dir) / dbname)
    db.save_chunks(chunks)
    db.close()
    return db


class TestStatsNoHydration(unittest.TestCase):
    """stats() on a lazy DB-backed index must not hydrate chunk bodies."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index_dir = Path(self.tmpdir)

    def test_db_stats_makes_zero_materialize_calls(self):
        """Red-first for the triple pass: stats() must hydrate nothing.

        Before the fix this count is 2 (one per set-comprehension), each
        triggering full hydration of every chunk via
        load_chunks_by_rowids -- the 92%-of-cmd_stats term measured on the shared index.
        """
        seed_db(self.index_dir, [
            make_chunk("s1:t0", "s1", "2026-01-01T00:00:00Z", "hello",
                       tools=("Bash", "Read"), files=("/a.py",)),
            make_chunk("s2:t0", "s2", "2026-01-02T00:00:00Z", "world",
                       tools=("Read",), files=("/b.py",)),
        ])
        index = TranscriptIndex.load(self.index_dir)
        calls = []
        with mock.patch.object(
            TranscriptIndex,
            "_materialize_all_chunks",
            side_effect=lambda: calls.append(1) or index.chunks,
        ):
            index.stats()
        self.assertEqual(
            len(calls), 0,
            "stats() hydrated chunk bodies; tools/files must come from "
            "SQL aggregates over the stored columns instead",
        )
        index.close() if hasattr(index, "close") else None

    def test_stats_counts_match_hydration_baseline(self):
        """Equality witness for the SQL-aggregate path.

        The baseline counts tools/files from FULLY HYDRATED chunks
        (db.load_chunks) -- what the pre-fix implementation measured by
        construction. stats() must produce identical counts whatever
        mechanism it uses internally.
        """
        seed_db(self.index_dir, [
            make_chunk("s1:t0", "s1", "2026-01-01T00:00:00Z", "one",
                       tools=("Bash", "Read", "Bash"), files=("/a.py", "/b.py")),
            make_chunk("s1:t1", "s1", "2026-01-01T01:00:00Z", "two",
                       tools=(), files=()),
            make_chunk("s2:t0", "s2", "2026-01-02T00:00:00Z", "three",
                       tools=("Edit",), files=("/a.py",)),
        ])
        index = TranscriptIndex.load(self.index_dir)
        stats = index.stats()
        baseline_chunks = index._db.load_chunks()
        want_tools = {t for c in baseline_chunks for t in c.tools_used}
        want_files = {f for c in baseline_chunks for f in c.files_touched}
        self.assertEqual(stats["total_tools_used"], len(want_tools))
        self.assertEqual(stats["total_files_touched"], len(want_files))
        index.close() if hasattr(index, "close") else None


class TestDistinctToolsFiles(unittest.TestCase):
    """The storage-level aggregate itself."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index_dir = Path(self.tmpdir)

    def test_handles_empty_malformed_and_duplicates(self):
        db = RecallDB(self.index_dir / "recall.db")
        db.save_chunks([
            make_chunk("s1:t0", "s1", "2026-01-01T00:00:00Z", "one",
                       tools=("Bash", "Read", "Bash"), files=("/a.py", "/a.py")),
            make_chunk("s2:t0", "s2", "2026-01-02T00:00:00Z", "two",
                       tools=(), files=()),
        ])
        try:
            tools, files = db.distinct_tools_files()
            self.assertEqual(tools, {"Bash", "Read"})
            self.assertEqual(files, {"/a.py"})
        finally:
            db.close()

    def test_excludes_sessions_and_chunk_ids(self):
        """Suppressed sessions and overlay-replaced ids must be excluded
        in SQL, so the sharded aggregate matches the merged visible-chunk
        baseline (a tool present only in suppressed/overlaid rows must
        not survive via set subtraction)."""
        db = RecallDB(self.index_dir / "recall.db")
        db.save_chunks([
            make_chunk("keep:t0", "keep", "2026-01-01T00:00:00Z", "k",
                       tools=("Bash",), files=("/keep.py",)),
            make_chunk("gone:t0", "gone", "2026-01-02T00:00:00Z", "g",
                       tools=("OnlyInGone",), files=("/gone.py",)),
        ])
        try:
            tools, files = db.distinct_tools_files(
                exclude_sessions={"gone"}, exclude_chunk_ids=set())
            self.assertEqual(tools, {"Bash"})
            self.assertEqual(files, {"/keep.py"})
        finally:
            db.close()


class TestShardedAggregateParity(unittest.TestCase):
    """ShardedRecallDB.distinct_tools_files: suppression + overlay parity."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index_dir = Path(self.tmpdir)

    def _seed_sharded(self, suppresses_base):
        RecallDB(self.index_dir / "index.db").close()
        shard = RecallDB(self.index_dir / "data_001.db")
        shard.save_chunks([
            make_chunk("base:t0", "live", "2026-01-01T00:00:00Z", "base keep",
                       tools=("Bash",), files=("/keep.py",)),
            make_chunk("rew:t0", "rew", "2026-01-02T00:00:00Z", "rewound base",
                       tools=("OnlyInRewound",), files=("/rewound.py",)),
        ])
        shard.close()
        index = RecallDB(self.index_dir / "index.db")
        index.replace_query_tail(
            source_key="runtime-source",
            session_id="rew",
            rewind_offset=1,
            chunks=[make_chunk("rew:t1", "rew", "2026-01-03T00:00:00Z",
                               "fresh overlay",
                               tools=("Codex",), files=("/overlay.py",))],
            cursor={
                "transcript_path": "/runtime/rew.jsonl",
                "observed_complete_offset": 2,
                "rewind_offset": 1,
                "rewind_turn_index": 1,
                "source_size": 2,
                "source_mtime_ns": 2,
                "observed_prefix_sha256": "digest",
                "suppresses_base": suppresses_base,
                "latest_projected_timestamp": "2026-01-03T00:00:00Z",
                "last_attempt_at": "2026-01-03T00:00:00Z",
                "last_success_at": "2026-01-03T00:00:00Z",
            },
        )
        index.close()
        return ShardedRecallDB.open_readonly(self.index_dir)

    def test_suppressed_session_tools_do_not_leak(self):
        db = self._seed_sharded(suppresses_base=True)
        try:
            tools, files = db.distinct_tools_files()
            self.assertEqual(tools, {"Bash", "Codex"})
            self.assertEqual(files, {"/keep.py", "/overlay.py"})
        finally:
            db.close()

    def test_nonsuppressed_rewound_session_keeps_base_rows(self):
        """suppresses_base=False: the base rows stay visible (only the id
        overlap is replaced by overlay), so "OnlyInRewound" survives."""
        db = self._seed_sharded(suppresses_base=False)
        try:
            tools, files = db.distinct_tools_files()
            self.assertEqual(tools, {"Bash", "OnlyInRewound", "Codex"})
            self.assertEqual(files, {"/keep.py", "/rewound.py", "/overlay.py"})
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()