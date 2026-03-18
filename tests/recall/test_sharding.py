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


if __name__ == "__main__":
    unittest.main()
