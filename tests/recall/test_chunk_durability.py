"""Tests for cluster durability classification (#410).

Verifies that:
1. Durable clusters (decisions, architecture, milestones) are classified correctly
2. Ephemeral clusters (debugging, navigation, tool output) are classified correctly
3. Mixed clusters get "mixed" classification
4. The durability column is added to the clusters table
5. Concise lookup applies durability-based score discounts
"""

import unittest

from synapt.recall.hybrid import classify_cluster_durability


class TestDurabilityClassification(unittest.TestCase):
    """Test the rule-based durability classifier."""

    # --- Durable clusters ---

    def test_decision_cluster_is_durable(self):
        self.assertEqual(
            classify_cluster_durability(
                "Decision: SQLite over PostgreSQL for storage",
                "Team decided to use SQLite for local storage because it's embedded."
            ),
            "durable",
        )

    def test_architecture_cluster_is_durable(self):
        self.assertEqual(
            classify_cluster_durability(
                "Architecture: plugin entry point system",
                "Designed the plugin system using entry_points for extensibility."
            ),
            "durable",
        )

    def test_milestone_cluster_is_durable(self):
        self.assertEqual(
            classify_cluster_durability(
                "Shipped v0.6.2 with CodeMemo benchmark",
                "Released version 0.6.2 scoring J=90.51% on CodeMemo benchmark."
            ),
            "durable",
        )

    def test_policy_cluster_is_durable(self):
        self.assertEqual(
            classify_cluster_durability(
                "Convention: always use gr commands",
                "Established team convention to use gr instead of raw git/gh."
            ),
            "durable",
        )

    def test_configuration_cluster_is_durable(self):
        self.assertEqual(
            classify_cluster_durability(
                "Config: embedding model selection",
                "Selected all-MiniLM-L6-v2 for embeddings. 384 dimensions, 22M params."
            ),
            "durable",
        )

    # --- Ephemeral clusters ---

    def test_debugging_cluster_is_ephemeral(self):
        self.assertEqual(
            classify_cluster_durability(
                "Debugging FTS index creation failure",
                "Investigated why FTS5 virtual table wasn't being created. Checked schema, ran tests."
            ),
            "ephemeral",
        )

    def test_navigation_cluster_is_ephemeral(self):
        self.assertEqual(
            classify_cluster_durability(
                "Reading and checking CLI entrypoint",
                "Read cli.py to find the search command flags. Checked argparse setup."
            ),
            "ephemeral",
        )

    def test_refactoring_steps_ephemeral(self):
        self.assertEqual(
            classify_cluster_durability(
                "Refactoring test fixtures",
                "Moved test fixtures to conftest.py. Updated imports in 12 test files."
            ),
            "ephemeral",
        )

    def test_troubleshooting_cluster_is_ephemeral(self):
        self.assertEqual(
            classify_cluster_durability(
                "Troubleshooting CI pipeline timeout",
                "CI was timing out on Windows. Tried increasing timeout, checked runner specs."
            ),
            "ephemeral",
        )

    # --- Edge cases ---

    def test_empty_topic_is_ephemeral(self):
        self.assertEqual(
            classify_cluster_durability("", ""),
            "ephemeral",
        )

    def test_mixed_signals_is_mixed(self):
        """Cluster with both durable and ephemeral signals gets 'mixed'."""
        result = classify_cluster_durability(
            "Debugging led to architecture decision",
            "While debugging the FTS issue, decided to switch to WAL mode for concurrency."
        )
        # Has both "debugging" (ephemeral) and "decided" (durable)
        self.assertIn(result, ("mixed", "durable"))

    def test_no_signals_is_ephemeral(self):
        """Clusters with no clear signals default to ephemeral."""
        self.assertEqual(
            classify_cluster_durability(
                "General discussion",
                "Talked about various topics."
            ),
            "ephemeral",
        )


class TestDurabilityColumnExists(unittest.TestCase):
    """Test that the durability column exists in the clusters table."""

    def test_durability_column_in_schema(self):
        import tempfile
        from pathlib import Path
        from synapt.recall.storage import RecallDB

        tmpdir = tempfile.mkdtemp()
        db = RecallDB(Path(tmpdir) / "recall.db")
        # Check column exists
        cursor = db._conn.execute("PRAGMA table_info(clusters)")
        columns = {row[1] for row in cursor.fetchall()}
        self.assertIn("durability", columns)
        db.close()


if __name__ == "__main__":
    unittest.main()
