"""Tests for user correction capture and benchmark logging."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.corrections import log_correction, read_corrections


class TestLogCorrection(unittest.TestCase):
    """Test correction logging to JSONL."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.project = Path(self.tmpdir) / "project"
        self.project.mkdir()

    def test_creates_correction_file(self):
        """First correction creates the file."""
        with patch("synapt.recall.corrections.project_data_dir",
                    return_value=self.project):
            path = log_correction(
                question="What model does CodeMemo use?",
                wrong_answer="claude-haiku",
                correct_answer="gpt-5-mini",
                category="convention",
                project=self.project,
            )
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "corrections.jsonl")

    def test_correction_entry_format(self):
        """Correction entry has all required fields."""
        with patch("synapt.recall.corrections.project_data_dir",
                    return_value=self.project):
            log_correction(
                question="What version?",
                wrong_answer="0.7.5",
                correct_answer="0.7.8",
                category="factual",
                project=self.project,
            )
            entries = read_corrections(project=self.project)

        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["question"], "What version?")
        self.assertEqual(entry["wrong_answer"], "0.7.5")
        self.assertEqual(entry["correct_answer"], "0.7.8")
        self.assertEqual(entry["category"], "factual")
        self.assertEqual(entry["source"], "user_correction")
        self.assertIn("timestamp", entry)

    def test_multiple_corrections_append(self):
        """Multiple corrections append to the same file."""
        with patch("synapt.recall.corrections.project_data_dir",
                    return_value=self.project):
            log_correction("Q1", "wrong1", "right1", project=self.project)
            log_correction("Q2", "wrong2", "right2", project=self.project)
            entries = read_corrections(project=self.project)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["question"], "Q1")
        self.assertEqual(entries[1]["question"], "Q2")

    def test_empty_category_allowed(self):
        """Category defaults to empty string."""
        with patch("synapt.recall.corrections.project_data_dir",
                    return_value=self.project):
            log_correction("Q", "wrong", "right", project=self.project)
            entries = read_corrections(project=self.project)

        self.assertEqual(entries[0]["category"], "")


class TestReadCorrections(unittest.TestCase):
    """Test reading corrections from JSONL."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.project = Path(self.tmpdir) / "project"
        self.project.mkdir()

    def test_empty_when_no_file(self):
        """Returns empty list when no corrections file exists."""
        with patch("synapt.recall.corrections.project_data_dir",
                    return_value=self.project):
            entries = read_corrections(project=self.project)
        self.assertEqual(entries, [])

    def test_skips_malformed_lines(self):
        """Gracefully skips malformed JSON lines."""
        corrections_dir = self.project / "recall"
        corrections_dir.mkdir(parents=True)
        path = corrections_dir / "corrections.jsonl"
        with open(path, "w") as f:
            f.write('{"question": "Q1", "wrong_answer": "w", "correct_answer": "r"}\n')
            f.write("not json\n")
            f.write('{"question": "Q2", "wrong_answer": "w", "correct_answer": "r"}\n')

        with patch("synapt.recall.corrections.project_data_dir",
                    return_value=self.project):
            entries = read_corrections(project=self.project)

        self.assertEqual(len(entries), 2)


if __name__ == "__main__":
    unittest.main()
