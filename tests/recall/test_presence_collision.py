"""Tests for recall#546: presence collision when two sessions share griptree.

TDD — the bug: when a human and an agent session share the same
griptree (e.g., Layne runs Claude from Opus's workdir), both get
the same agent_id from _agent_id() and the second join overwrites
the first's presence (including role).

Fix: channel_join should not overwrite a human's presence with an
agent's presence when the agent_id matches. Role escalation
(human > agent) should be preserved.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.channel import (
    _open_db,
    channel_join,
    channel_who,
)


def _patch_data_dir(tmpdir):
    """Return a combined patcher for project_data_dir + disable global store."""
    data_dir = Path(tmpdir) / "project" / ".synapt" / "recall"
    patcher_data = patch(
        "synapt.recall.channel.project_data_dir",
        return_value=data_dir,
    )
    patcher_manifest = patch(
        "synapt.recall.channel._read_manifest_url",
        return_value=None,
    )

    class _CombinedPatcher:
        def start(self):
            patcher_data.start()
            patcher_manifest.start()
        def stop(self):
            patcher_manifest.stop()
            patcher_data.stop()
        def __enter__(self):
            self.start()
            return self
        def __exit__(self, *args):
            self.stop()

    return _CombinedPatcher()


class TestPresenceCollision(unittest.TestCase):
    """Presence collision when two sessions share the same agent_id."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self._tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_human_presence_not_overwritten_by_agent(self):
        """An agent joining with same agent_id should not overwrite a human's role."""
        # Human joins first
        channel_join("dev", agent_name="shared-id", role="human", display_name="Layne")

        # Agent joins with same agent_id (same griptree, different session)
        channel_join("dev", agent_name="shared-id", role="agent", display_name="Opus")

        conn = _open_db()
        row = conn.execute(
            "SELECT role, display_name FROM presence WHERE agent_id = 'shared-id'"
        ).fetchone()
        conn.close()

        # Human role should be preserved (human > agent)
        self.assertEqual(row["role"], "human")

    def test_agent_can_join_without_overwriting_human_display_name(self):
        """When an agent shares agent_id with a human, the human's display name is preserved."""
        channel_join("dev", agent_name="shared-id", role="human", display_name="Layne")
        channel_join("dev", agent_name="shared-id", role="agent", display_name="Opus")

        conn = _open_db()
        row = conn.execute(
            "SELECT display_name FROM presence WHERE agent_id = 'shared-id'"
        ).fetchone()
        conn.close()

        # Human's display name should be preserved
        self.assertEqual(row["display_name"], "Layne")

    def test_who_shows_human_after_agent_collision(self):
        """channel_who should show the human, not the agent, after a collision."""
        channel_join("dev", agent_name="shared-id", role="human", display_name="Layne")
        channel_join("dev", agent_name="shared-id", role="agent", display_name="Opus")

        result = channel_who("dev")
        self.assertIn("human", result)

    def test_agent_to_agent_overwrites_normally(self):
        """When two agents share agent_id (no human involved), normal overwrite applies."""
        channel_join("dev", agent_name="agent-id", role="agent", display_name="Agent1")
        channel_join("dev", agent_name="agent-id", role="agent", display_name="Agent2")

        conn = _open_db()
        row = conn.execute(
            "SELECT display_name FROM presence WHERE agent_id = 'agent-id'"
        ).fetchone()
        conn.close()

        # Agent-to-agent: last writer wins
        self.assertEqual(row["display_name"], "Agent2")


if __name__ == "__main__":
    unittest.main()
