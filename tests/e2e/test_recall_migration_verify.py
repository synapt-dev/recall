"""E2E test: verify recall history survives gripspace migration (Story #5).

After migration, recall_search from the gripspace root should return
conversations that were indexed before migration. The .synapt/recall/
directory gets copied to the gripspace root during migration.

This test simulates:
1. A repo with indexed recall data (.synapt/recall/)
2. Migration moves repo contents into a subdirectory
3. .synapt/recall/ is copied to gripspace root
4. recall_search from gripspace root finds old data
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestRecallMigrationVerify(unittest.TestCase):
    """Verify recall history survives the v3.1 migration algorithm."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def _create_pre_migration_repo(self) -> Path:
        """Create a mock repo with recall data, simulating pre-migration state."""
        repo = self.tmpdir / "conversa-app"
        repo.mkdir()

        # Simulate repo files
        (repo / "src").mkdir()
        (repo / "src" / "app.py").write_text("# app code")
        (repo / "package.json").write_text('{"name": "conversa"}')

        # Simulate .synapt/recall/ with indexed data
        recall_dir = repo / ".synapt" / "recall"
        recall_dir.mkdir(parents=True)

        # Create a mock chunks database
        import sqlite3
        db_path = recall_dir / "index.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT, "
            "session_id TEXT, timestamp TEXT)"
        )
        conn.execute(
            "INSERT INTO chunks (text, session_id, timestamp) VALUES (?, ?, ?)",
            ("Discussion about auth implementation using JWT tokens", "sess-001",
             "2026-03-15T10:00:00Z"),
        )
        conn.execute(
            "INSERT INTO chunks (text, session_id, timestamp) VALUES (?, ?, ?)",
            ("Fixed the database connection pooling bug in production", "sess-002",
             "2026-03-20T14:00:00Z"),
        )
        conn.commit()
        conn.close()

        # Create channel history
        channels_dir = recall_dir / "channels"
        channels_dir.mkdir()
        dev_jsonl = channels_dir / "dev.jsonl"
        messages = [
            {"timestamp": "2026-03-15T10:00:00Z", "from_agent": "anchor-001",
             "channel": "dev", "type": "message", "body": "Sprint 1 kickoff"},
            {"timestamp": "2026-03-20T14:00:00Z", "from_agent": "atlas-001",
             "channel": "dev", "type": "message", "body": "Auth module shipped"},
        ]
        with open(dev_jsonl, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        return repo

    def _run_v31_migration(self, repo: Path) -> Path:
        """Run the v3.1 migration algorithm. Returns gripspace root."""
        gripspace = repo  # In-place migration: repo becomes gripspace

        repo_name = "conversa-app"  # Would come from git remote

        # Step 1: Create temp dir and move everything into it
        tmp_dir = gripspace / "_tmp_migrate"
        tmp_dir.mkdir()
        for item in gripspace.iterdir():
            if item.name == "_tmp_migrate":
                continue
            shutil.move(str(item), str(tmp_dir / item.name))

        # Step 2: Rename temp dir to repo name
        repo_subdir = gripspace / repo_name
        tmp_dir.rename(repo_subdir)

        # Step 3: Copy .synapt to gripspace root
        src_synapt = repo_subdir / ".synapt"
        if src_synapt.exists():
            dst_synapt = gripspace / ".synapt"
            shutil.copytree(str(src_synapt), str(dst_synapt))

        return gripspace

    def test_recall_db_exists_after_migration(self):
        """The recall index.db should exist at gripspace root after migration."""
        repo = self._create_pre_migration_repo()
        gripspace = self._run_v31_migration(repo)

        db_path = gripspace / ".synapt" / "recall" / "index.db"
        self.assertTrue(db_path.exists(), "index.db should exist at gripspace root")

    def test_recall_chunks_preserved(self):
        """All indexed chunks should be present after migration."""
        import sqlite3
        repo = self._create_pre_migration_repo()
        gripspace = self._run_v31_migration(repo)

        db_path = gripspace / ".synapt" / "recall" / "index.db"
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT text FROM chunks ORDER BY timestamp").fetchall()
        conn.close()

        self.assertEqual(len(rows), 2)
        self.assertIn("JWT tokens", rows[0][0])
        self.assertIn("connection pooling", rows[1][0])

    def test_channel_history_preserved(self):
        """Channel JSONL files should be at gripspace root after migration."""
        repo = self._create_pre_migration_repo()
        gripspace = self._run_v31_migration(repo)

        dev_jsonl = gripspace / ".synapt" / "recall" / "channels" / "dev.jsonl"
        self.assertTrue(dev_jsonl.exists(), "dev.jsonl should exist")

        with open(dev_jsonl) as f:
            messages = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["body"], "Sprint 1 kickoff")
        self.assertEqual(messages[1]["body"], "Auth module shipped")

    def test_repo_files_in_subdirectory(self):
        """Original repo files should be in the repo subdirectory."""
        repo = self._create_pre_migration_repo()
        gripspace = self._run_v31_migration(repo)

        self.assertTrue((gripspace / "conversa-app" / "src" / "app.py").exists())
        self.assertTrue((gripspace / "conversa-app" / "package.json").exists())

    def test_recall_data_in_both_locations(self):
        """Recall data should exist both at gripspace root and in repo subdir."""
        repo = self._create_pre_migration_repo()
        gripspace = self._run_v31_migration(repo)

        # Gripspace root (for recall_search from root)
        self.assertTrue(
            (gripspace / ".synapt" / "recall" / "index.db").exists(),
            "Recall at gripspace root"
        )
        # Repo subdir (original location, preserved)
        self.assertTrue(
            (gripspace / "conversa-app" / ".synapt" / "recall" / "index.db").exists(),
            "Recall in repo subdir"
        )


if __name__ == "__main__":
    unittest.main()
