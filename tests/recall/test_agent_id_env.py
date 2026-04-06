"""Tests for Story #5-6: SYNAPT_AGENT_ID env var in channel operations.

TDD — tests written before implementation changes to _agent_id().

When SYNAPT_AGENT_ID is set (by gr spawn), all channel operations should
use it instead of generating a session hash. This gives agents stable
identity across restarts.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.channel import (
    _agent_id,
    _open_db,
    channel_join,
    channel_post,
    channel_read,
    channel_who,
)


def _patch_data_dir(tmpdir):
    """Return a patcher for project_data_dir targeting a temp directory."""
    data_dir = Path(tmpdir) / "project" / ".synapt" / "recall"
    return patch(
        "synapt.recall.channel.project_data_dir",
        return_value=data_dir,
    )


class TestAgentIdFromEnv(unittest.TestCase):
    """SYNAPT_AGENT_ID env var should override session hash."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self._tmpdir)
        self._patcher.start()
        # Clear the agent ID cache between tests
        from synapt.recall.channel import _AGENT_ID_CACHE
        _AGENT_ID_CACHE.clear()

    def tearDown(self):
        self._patcher.stop()
        os.environ.pop("SYNAPT_AGENT_ID", None)
        from synapt.recall.channel import _AGENT_ID_CACHE
        _AGENT_ID_CACHE.clear()

    def test_agent_id_uses_env_var(self):
        """_agent_id() returns SYNAPT_AGENT_ID when set."""
        os.environ["SYNAPT_AGENT_ID"] = "opus-001"
        self.assertEqual(_agent_id(), "opus-001")

    def test_agent_id_falls_back_to_session_hash(self):
        """_agent_id() generates session hash when env var not set."""
        os.environ.pop("SYNAPT_AGENT_ID", None)
        aid = _agent_id()
        self.assertTrue(aid.startswith("s_"))

    def test_channel_join_uses_env_agent_id(self):
        """channel_join stores the env-based agent_id in presence."""
        os.environ["SYNAPT_AGENT_ID"] = "atlas-001"
        channel_join("dev", display_name="Atlas")

        conn = _open_db()
        row = conn.execute(
            "SELECT agent_id FROM presence WHERE agent_id = ?",
            ("atlas-001",),
        ).fetchone()
        conn.close()

        self.assertIsNotNone(row)

    def test_channel_post_uses_env_agent_id(self):
        """Messages posted use the env-based agent_id as from_agent."""
        os.environ["SYNAPT_AGENT_ID"] = "sentinel-001"
        channel_join("dev", display_name="Sentinel")
        channel_post("dev", "test message")

        from synapt.recall.channel import _channel_path, _read_messages
        path = _channel_path("dev")
        messages = _read_messages(path)
        # Find the test message (skip join event)
        test_msgs = [m for m in messages if m.body == "test message"]
        self.assertEqual(len(test_msgs), 1)
        self.assertEqual(test_msgs[0].from_agent, "sentinel-001")

    def test_channel_who_shows_env_agent_id(self):
        """who output shows the env-based agent_id."""
        os.environ["SYNAPT_AGENT_ID"] = "apollo-001"
        channel_join("dev", display_name="Apollo")
        result = channel_who("dev")
        self.assertIn("apollo-001", result)

    def test_env_agent_id_is_stable_across_calls(self):
        """Same env var → same agent_id on every call."""
        os.environ["SYNAPT_AGENT_ID"] = "opus-001"
        id1 = _agent_id()
        id2 = _agent_id()
        self.assertEqual(id1, id2)
        self.assertEqual(id1, "opus-001")


if __name__ == "__main__":
    unittest.main()
