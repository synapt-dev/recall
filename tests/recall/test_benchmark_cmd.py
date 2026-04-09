"""Tests for the `synapt recall benchmark` CLI command.

Verifies that:
1. The benchmark subcommand is registered and callable
2. It measures cold start (index load) timing
3. It measures per-query latencies across multiple queries
4. It reports p50/p95/p99 statistics
5. It outputs valid JSON when --json is requested
6. It reports memory footprint of the loaded index
7. It handles empty indexes gracefully
"""

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.storage import RecallDB, EMBEDDING_DIM


def _make_test_chunks(n: int, sessions: int = 5) -> list[TranscriptChunk]:
    """Create n test chunks spread across sessions."""
    chunks = []
    for i in range(n):
        sid = f"sess-{i % sessions:04d}"
        chunks.append(TranscriptChunk(
            id=f"{sid}:t{i}",
            session_id=sid,
            timestamp=f"2026-03-01T{10 + (i % 12):02d}:00:00Z",
            turn_index=i,
            user_text=f"User asked about topic_{i} with keyword_{i % 10} and concept_{i % 7}",
            assistant_text=f"Assistant explained topic_{i} using approach_{i % 5}",
            tools_used=["Read", "Grep"] if i % 3 == 0 else [],
            files_touched=[f"src/module_{i % 5}.py"] if i % 2 == 0 else [],
        ))
    return chunks


def _build_test_db(tmp_path: Path, chunks: list[TranscriptChunk]) -> RecallDB:
    """Build a populated test DB (FTS triggers auto-populate on save)."""
    db_path = tmp_path / "recall.db"
    db = RecallDB(db_path)
    db.save_chunks(chunks)
    return db


class TestBenchmarkCommand(unittest.TestCase):
    """Test that the benchmark subcommand is registered and callable."""

    def test_benchmark_in_help(self):
        """benchmark appears in the CLI help output."""
        from synapt.recall.cli import main
        import io
        with self.assertRaises(SystemExit):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                sys.argv = ["synapt", "recall", "--help"]
                main()

        # Check that benchmark is listed as a subcommand
        result = subprocess.run(
            [sys.executable, "-m", "synapt.recall.cli", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertIn("benchmark", result.stdout)

    def test_benchmark_help(self):
        """benchmark subcommand has its own help."""
        result = subprocess.run(
            [sys.executable, "-m", "synapt.recall.cli", "benchmark", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--json", result.stdout)
        self.assertIn("--queries", result.stdout)
        self.assertIn("--iterations", result.stdout)


class TestBenchmarkExecution(unittest.TestCase):
    """Test benchmark execution against a real (small) index."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmpdir)
        self.chunks = _make_test_chunks(100, sessions=10)
        self.db = _build_test_db(self.tmp_path, self.chunks)

    def tearDown(self):
        self.db.close()

    def test_benchmark_runs_and_reports_timing(self):
        """Benchmark produces timing output with index info."""
        from synapt.recall.cli import cmd_benchmark
        import argparse
        import io

        args = argparse.Namespace(
            index=str(self.tmp_path),
            json_output=False,
            queries=None,
            iterations=3,
        )
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            cmd_benchmark(args)
            output = mock_out.getvalue()

        self.assertIn("Chunks:", output)
        self.assertIn("Cold start", output)
        self.assertIn("p50", output.lower())

    def test_benchmark_json_output(self):
        """Benchmark produces valid JSON when --json is set."""
        from synapt.recall.cli import cmd_benchmark
        import argparse
        import io

        args = argparse.Namespace(
            index=str(self.tmp_path),
            json_output=True,
            queries=None,
            iterations=3,
        )
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            cmd_benchmark(args)
            output = mock_out.getvalue()

        data = json.loads(output)
        self.assertIn("cold_start_ms", data)
        self.assertIn("queries", data)
        self.assertIn("index_info", data)
        self.assertIsInstance(data["index_info"]["chunk_count"], int)
        self.assertEqual(data["index_info"]["chunk_count"], 100)

    def test_benchmark_custom_queries(self):
        """Benchmark uses user-provided queries."""
        from synapt.recall.cli import cmd_benchmark
        import argparse
        import io

        args = argparse.Namespace(
            index=str(self.tmp_path),
            json_output=True,
            queries="topic_1;keyword_5",
            iterations=2,
        )
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            cmd_benchmark(args)
            output = mock_out.getvalue()

        data = json.loads(output)
        self.assertEqual(len(data["queries"]), 2)
        self.assertEqual(data["queries"][0]["query"], "topic_1")
        self.assertEqual(data["queries"][1]["query"], "keyword_5")

    def test_benchmark_reports_memory(self):
        """Benchmark includes memory footprint estimate."""
        from synapt.recall.cli import cmd_benchmark
        import argparse
        import io

        args = argparse.Namespace(
            index=str(self.tmp_path),
            json_output=True,
            queries=None,
            iterations=2,
        )
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            cmd_benchmark(args)
            output = mock_out.getvalue()

        data = json.loads(output)
        self.assertIn("memory_mb", data["index_info"])
        self.assertGreater(data["index_info"]["memory_mb"], 0)


class TestBenchmarkEmptyIndex(unittest.TestCase):
    """Test benchmark handles edge cases gracefully."""

    def test_benchmark_empty_index(self):
        """Benchmark handles an empty index without crashing."""
        from synapt.recall.cli import cmd_benchmark
        import argparse
        import io

        tmpdir = tempfile.mkdtemp()
        tmp_path = Path(tmpdir)
        db = RecallDB(tmp_path / "recall.db")
        db.save_chunks([])

        args = argparse.Namespace(
            index=str(tmp_path),
            json_output=True,
            queries=None,
            iterations=2,
        )
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            cmd_benchmark(args)
            output = mock_out.getvalue()

        data = json.loads(output)
        self.assertEqual(data["index_info"]["chunk_count"], 0)
        db.close()


if __name__ == "__main__":
    unittest.main()
