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


class TestRecallCorrectTool(unittest.TestCase):
    """Test the recall_correct MCP tool end-to-end."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.project = Path(self.tmpdir) / "project"
        self.project.mkdir()

    def test_recall_correct_logs_and_creates_node(self):
        """recall_correct both logs the correction AND creates a knowledge node."""
        from synapt.recall.server import recall_correct

        kn_path = self.project / "recall" / "knowledge.jsonl"
        with patch("synapt.recall.corrections.project_data_dir",
                    return_value=self.project), \
             patch("synapt.recall.knowledge._knowledge_path",
                    return_value=kn_path):
            result = recall_correct(
                question="What model does CodeMemo use?",
                wrong_answer="claude-haiku",
                correct_answer="gpt-5-mini",
                category="convention",
            )

        # Verify logging happened
        with patch("synapt.recall.corrections.project_data_dir",
                    return_value=self.project):
            entries = read_corrections(project=self.project)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["question"], "What model does CodeMemo use?")
        self.assertEqual(entries[0]["correct_answer"], "gpt-5-mini")

        # Verify knowledge node was created (not a contradiction)
        self.assertIn("Knowledge node created", result)
        self.assertNotIn("contradiction", result.lower())

        # Verify result message
        self.assertIn("Correction captured", result)
        self.assertIn("gpt-5-mini", result)
        self.assertIn("claude-haiku", result)

    def test_recall_correct_includes_category_in_log(self):
        """Category is preserved in the correction log."""
        from synapt.recall.server import recall_correct

        with patch("synapt.recall.corrections.project_data_dir",
                    return_value=self.project), \
             patch("synapt.recall.server.recall_contradict",
                    return_value="No matching nodes found"):
            recall_correct(
                question="When was the merge freeze?",
                wrong_answer="March 1",
                correct_answer="March 5",
                category="temporal",
            )

        with patch("synapt.recall.corrections.project_data_dir",
                    return_value=self.project):
            entries = read_corrections(project=self.project)
        self.assertEqual(entries[0]["category"], "temporal")

    def test_recall_correct_creates_knowledge_node(self):
        """recall_correct immediately creates a knowledge node."""
        from synapt.recall.server import recall_correct

        kn_path = self.project / "recall" / "knowledge.jsonl"
        with patch("synapt.recall.corrections.project_data_dir",
                    return_value=self.project), \
             patch("synapt.recall.knowledge._knowledge_path",
                    return_value=kn_path):
            result = recall_correct(
                question="What model do we use?",
                wrong_answer="wrong-model",
                correct_answer="right-model",
                category="convention",
            )

        self.assertIn("Knowledge node created", result)
        self.assertNotIn("Failed", result)
        # Verify the knowledge node was actually written
        self.assertTrue(kn_path.exists())

    def test_recall_correct_node_content_includes_context(self):
        """Knowledge node content includes both answer and question context."""
        from synapt.recall.server import recall_correct

        kn_path = self.project / "recall" / "knowledge.jsonl"
        with patch("synapt.recall.corrections.project_data_dir",
                    return_value=self.project), \
             patch("synapt.recall.knowledge._knowledge_path",
                    return_value=kn_path):
            result = recall_correct(
                question="What judge for LOCOMO?",
                wrong_answer="haiku",
                correct_answer="gpt-4o-mini",
            )

        # Result should include the correct answer and mention knowledge node
        self.assertIn("gpt-4o-mini", result)
        self.assertIn("Knowledge node created", result)
        # Should NOT mention contradiction queue
        self.assertNotIn("contradiction", result.lower())


if __name__ == "__main__":
    unittest.main()
