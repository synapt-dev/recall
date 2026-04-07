"""Tests for recall#546 join spam: duplicate join events on MCP restart.

The bug: every channel_join() appends a join event to JSONL even if
the agent is already online. MCP server restarts fire join again,
flooding the channel log with duplicate "X joined #dev" events.

Fix: skip the JSONL join event if the agent is already an online
member of the channel.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.channel import (
    _channel_path,
    _read_messages,
    channel_join,
    channel_read,
)


def _patch_data_dir(tmpdir):
    data_dir = Path(tmpdir) / "project" / ".synapt" / "recall"
    return patch(
        "synapt.recall.channel.project_data_dir",
        return_value=data_dir,
    )


def _patch_manifest():
    return patch(
        "synapt.recall.channel._read_manifest_url",
        return_value=None,
    )


class TestJoinSpam(unittest.TestCase):
    """Duplicate join events should not flood the channel log."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self._tmpdir)
        self._manifest_patcher = _patch_manifest()
        self._patcher.start()
        self._manifest_patcher.start()

    def tearDown(self):
        self._manifest_patcher.stop()
        self._patcher.stop()

    def test_first_join_creates_event(self):
        """First join should create a join event in the log."""
        channel_join("dev", agent_name="atlas-001", display_name="Atlas")
        path = _channel_path("dev")
        messages = _read_messages(path)
        join_events = [m for m in messages if m.type == "join"]
        self.assertEqual(len(join_events), 1)

    def test_second_join_does_not_duplicate_event(self):
        """Rejoining when already online should NOT add another join event."""
        channel_join("dev", agent_name="atlas-001", display_name="Atlas")
        channel_join("dev", agent_name="atlas-001", display_name="Atlas")
        channel_join("dev", agent_name="atlas-001", display_name="Atlas")

        path = _channel_path("dev")
        messages = _read_messages(path)
        join_events = [m for m in messages if m.type == "join"]
        # Should be exactly 1, not 3
        self.assertEqual(len(join_events), 1)

    def test_rejoin_after_leave_creates_new_event(self):
        """Joining after leaving should create a new join event."""
        from synapt.recall.channel import channel_leave
        channel_join("dev", agent_name="atlas-001", display_name="Atlas")
        channel_leave("dev", agent_name="atlas-001")
        channel_join("dev", agent_name="atlas-001", display_name="Atlas")

        path = _channel_path("dev")
        messages = _read_messages(path)
        join_events = [m for m in messages if m.type == "join"]
        # Should be 2 (first join + rejoin after leave)
        self.assertEqual(len(join_events), 2)

    def test_different_agents_each_get_join_event(self):
        """Different agents joining each get their own join event."""
        channel_join("dev", agent_name="atlas-001", display_name="Atlas")
        channel_join("dev", agent_name="apollo-001", display_name="Apollo")

        path = _channel_path("dev")
        messages = _read_messages(path)
        join_events = [m for m in messages if m.type == "join"]
        self.assertEqual(len(join_events), 2)


if __name__ == "__main__":
    unittest.main()
