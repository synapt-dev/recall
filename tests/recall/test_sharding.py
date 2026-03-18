"""Tests for synapt.recall.sharding — tree-structured DB utilities."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from synapt.recall.sharding import (
    quarter_for_timestamp,
    shard_name,
    list_shards,
    group_chunks_by_quarter,
    estimate_split,
    is_sharded,
    split_monolithic_db,
)


class TestQuarterForTimestamp(unittest.TestCase):

    def test_q1(self):
        self.assertEqual(quarter_for_timestamp("2025-01-15T10:00:00Z"), "2025_q1")
        self.assertEqual(quarter_for_timestamp("2025-03-31T23:59:59Z"), "2025_q1")

    def test_q2(self):
        self.assertEqual(quarter_for_timestamp("2025-04-01T00:00:00Z"), "2025_q2")
        self.assertEqual(quarter_for_timestamp("2025-06-15T12:00:00Z"), "2025_q2")

    def test_q3(self):
        self.assertEqual(quarter_for_timestamp("2025-07-01T00:00:00Z"), "2025_q3")

    def test_q4(self):
        self.assertEqual(quarter_for_timestamp("2025-12-25T08:00:00Z"), "2025_q4")

    def test_invalid_returns_unknown(self):
        self.assertEqual(quarter_for_timestamp("not-a-date"), "unknown")
        self.assertEqual(quarter_for_timestamp(""), "unknown")

    def test_none_returns_unknown(self):
        self.assertEqual(quarter_for_timestamp(None), "unknown")

    def test_no_timezone(self):
        self.assertEqual(quarter_for_timestamp("2025-03-17T10:00:00"), "2025_q1")


class TestShardName(unittest.TestCase):

    def test_format(self):
        self.assertEqual(shard_name("2025_q1"), "data_2025_q1.db")
        self.assertEqual(shard_name("2024_q4"), "data_2024_q4.db")


class TestListShards(unittest.TestCase):

    def test_finds_data_dbs(self):
        tmpdir = tempfile.mkdtemp()
        d = Path(tmpdir)
        (d / "data_2024_q1.db").touch()
        (d / "data_2024_q2.db").touch()
        (d / "index.db").touch()  # Should not match
        (d / "other.db").touch()  # Should not match

        shards = list_shards(d)
        self.assertEqual(len(shards), 2)
        self.assertEqual(shards[0].name, "data_2024_q1.db")
        self.assertEqual(shards[1].name, "data_2024_q2.db")

    def test_empty_dir(self):
        tmpdir = tempfile.mkdtemp()
        self.assertEqual(list_shards(Path(tmpdir)), [])

    def test_unknown_shard_sorts_last(self):
        tmpdir = tempfile.mkdtemp()
        d = Path(tmpdir)
        (d / "data_2024_q1.db").touch()
        (d / "data_unknown.db").touch()
        shards = list_shards(d)
        self.assertEqual(shards[-1].name, "data_unknown.db")


class TestGroupChunksByQuarter(unittest.TestCase):

    def test_groups_correctly(self):
        chunks = [
            {"timestamp": "2025-01-10T10:00:00Z", "id": "c1"},
            {"timestamp": "2025-01-20T10:00:00Z", "id": "c2"},
            {"timestamp": "2025-04-05T10:00:00Z", "id": "c3"},
            {"timestamp": "2025-07-15T10:00:00Z", "id": "c4"},
        ]
        groups = group_chunks_by_quarter(chunks)
        self.assertEqual(len(groups["2025_q1"]), 2)
        self.assertEqual(len(groups["2025_q2"]), 1)
        self.assertEqual(len(groups["2025_q3"]), 1)
        self.assertNotIn("2025_q4", groups)

    def test_empty_list(self):
        self.assertEqual(group_chunks_by_quarter([]), {})


class TestEstimateSplit(unittest.TestCase):

    def test_empty_db_no_chunks_table(self):
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
        conn.close()
        self.assertEqual(estimate_split(db_path), {})

    def test_counts_by_quarter(self):
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE chunks (timestamp TEXT)")
        conn.executemany("INSERT INTO chunks VALUES (?)", [
            ("2025-01-10T10:00:00Z",),
            ("2025-01-20T10:00:00Z",),
            ("2025-04-05T10:00:00Z",),
        ])
        conn.commit()
        conn.close()
        result = estimate_split(db_path)
        self.assertEqual(result["2025_q1"], 2)
        self.assertEqual(result["2025_q2"], 1)


class TestIsSharded(unittest.TestCase):

    def test_monolithic(self):
        tmpdir = tempfile.mkdtemp()
        d = Path(tmpdir)
        (d / "recall.db").touch()
        self.assertFalse(is_sharded(d))

    def test_sharded(self):
        tmpdir = tempfile.mkdtemp()
        d = Path(tmpdir)
        (d / "index.db").touch()
        (d / "data_2025_q1.db").touch()
        self.assertTrue(is_sharded(d))


class TestSplitMonolithicDb(unittest.TestCase):
    """Test the migration from monolithic to sharded layout."""

    def _create_monolithic(self, tmpdir):
        """Create a minimal monolithic recall.db with test data."""
        from synapt.recall.storage import RecallDB
        db = RecallDB(Path(tmpdir) / "recall.db")
        # Insert some knowledge
        db._conn.execute(
            "INSERT INTO knowledge (id, content, category, confidence, status, "
            "source_sessions, source_turns, source_offsets, created_at, updated_at, "
            "superseded_by, contradiction_note, tags, valid_from, valid_until, version, lineage_id) "
            "VALUES ('k1', 'test fact', 'workflow', 0.8, 'active', '[]', '[]', '[]', "
            "'2025-01-01', '2025-01-01', '', '', '', NULL, NULL, 1, '')"
        )
        # Insert chunks across two quarters
        for i, ts in enumerate([
            "2025-01-10T10:00:00Z", "2025-02-15T10:00:00Z",  # Q1
            "2025-04-05T10:00:00Z",  # Q2
        ]):
            db._conn.execute(
                "INSERT INTO chunks (id, session_id, timestamp, turn_index, user_text, "
                "assistant_text, tools_used, files_touched, tool_content, transcript_path, "
                "byte_offset, byte_length) "
                f"VALUES ('c{i}', 's1', '{ts}', {i}, 'user', 'assistant', '[]', '[]', '', '', 0, 0)"
            )
        db._conn.commit()
        db.close()

    def test_split_creates_index_and_shards(self):
        tmpdir = tempfile.mkdtemp()
        self._create_monolithic(tmpdir)
        d = Path(tmpdir)

        plan = split_monolithic_db(d)

        self.assertIn("index.db", plan)
        self.assertTrue((d / "index.db").exists())
        self.assertTrue((d / "data_2025_q1.db").exists())
        self.assertTrue((d / "data_2025_q2.db").exists())
        self.assertEqual(plan["data_2025_q1.db"], 2)
        self.assertEqual(plan["data_2025_q2.db"], 1)

    def test_split_preserves_knowledge_in_index(self):
        tmpdir = tempfile.mkdtemp()
        self._create_monolithic(tmpdir)
        d = Path(tmpdir)
        split_monolithic_db(d)

        conn = sqlite3.connect(str(d / "index.db"))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM knowledge WHERE id = 'k1'").fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row["content"], "test fact")

    def test_dry_run_doesnt_write(self):
        tmpdir = tempfile.mkdtemp()
        self._create_monolithic(tmpdir)
        d = Path(tmpdir)

        plan = split_monolithic_db(d, dry_run=True)
        self.assertIn("2025_q1", plan)
        self.assertFalse((d / "index.db").exists())

    def test_already_sharded_raises(self):
        tmpdir = tempfile.mkdtemp()
        d = Path(tmpdir)
        (d / "index.db").touch()
        (d / "recall.db").touch()
        with self.assertRaises(RuntimeError):
            split_monolithic_db(d)

    def test_no_recall_db_raises(self):
        tmpdir = tempfile.mkdtemp()
        with self.assertRaises(FileNotFoundError):
            split_monolithic_db(Path(tmpdir))

    def test_failure_cleans_up_temp_files(self):
        """On failure, partial output is cleaned up — no stale shards left."""
        tmpdir = tempfile.mkdtemp()
        self._create_monolithic(tmpdir)
        d = Path(tmpdir)

        # Make quarter_for_timestamp raise after being called a few times
        # to simulate a failure during shard creation
        from unittest.mock import patch
        real_qft = quarter_for_timestamp
        call_count = [0]

        def failing_qft(ts):
            call_count[0] += 1
            if call_count[0] > 2:
                raise OSError("Simulated failure")
            return real_qft(ts)

        with patch("synapt.recall.sharding.quarter_for_timestamp", failing_qft):
            with self.assertRaises(OSError):
                split_monolithic_db(d)

        # No partial output left — no index.db, no data shards, no temp dirs
        self.assertFalse((d / "index.db").exists())
        self.assertEqual(list(d.glob("data_*.db")), [])
        self.assertEqual(list(d.glob(".split_tmp_*")), [])


if __name__ == "__main__":
    unittest.main()
