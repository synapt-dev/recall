"""E2E test: channel roundtrip with mock agents in a test gripspace.

Story #4: test gripspace with mock agents. This is the foundation
for all e2e tests — a minimal gripspace fixture that the channel
system can use without billing or real tmux sessions.

The fixture creates:
- Temp gripspace directory with .synapt/recall/ structure
- Mock manifest URL for org/project resolution
- Two mock agents with SYNAPT_AGENT_ID set
- Channel operations (join, post, read, claim) tested end-to-end
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.channel import (
    _open_db,
    channel_join,
    channel_post,
    channel_read,
    channel_who,
    channel_claim,
    channel_unclaim,
    channel_unread_read,
)


class GripspaceFixture:
    """Reusable test fixture for a minimal gripspace with mock agents."""

    def __init__(self):
        self.tmpdir = tempfile.mkdtemp()
        self.gripspace_root = Path(self.tmpdir) / "gripspace"
        self.data_dir = self.gripspace_root / ".synapt" / "recall"
        self.data_dir.mkdir(parents=True)

    def patch_data_dir(self):
        """Patch both data_dir and manifest URL to isolate from global store."""
        return patch(
            "synapt.recall.channel.project_data_dir",
            return_value=self.data_dir,
        )

    def patch_manifest(self):
        """Prevent global channel store routing — keep tests local."""
        return patch(
            "synapt.recall.channel._read_manifest_url",
            return_value=None,
        )

    def set_agent(self, agent_id: str):
        """Set the active agent via env var."""
        os.environ["SYNAPT_AGENT_ID"] = agent_id

    def clear_agent(self):
        os.environ.pop("SYNAPT_AGENT_ID", None)
        from synapt.recall.channel import _AGENT_ID_CACHE
        _AGENT_ID_CACHE.clear()


class TestChannelRoundtrip(unittest.TestCase):
    """E2E: two mock agents communicate via channels in a test gripspace."""

    def setUp(self):
        self.fixture = GripspaceFixture()
        self._patcher = self.fixture.patch_data_dir()
        self._manifest_patcher = self.fixture.patch_manifest()
        self._patcher.start()
        self._manifest_patcher.start()

    def tearDown(self):
        self.fixture.clear_agent()
        self._manifest_patcher.stop()
        self._patcher.stop()

    def test_two_agents_post_and_read(self):
        """Agent A posts, Agent B reads — full roundtrip."""
        # Agent A joins and posts
        self.fixture.set_agent("atlas-001")
        channel_join("dev", display_name="Atlas")
        channel_post("dev", "Hello from Atlas")

        # Agent B joins and reads
        self.fixture.set_agent("apollo-001")
        channel_join("dev", display_name="Apollo")
        result = channel_read("dev")

        self.assertIn("Hello from Atlas", result)
        self.assertIn("Atlas", result)

    def test_unread_tracks_per_agent(self):
        """Each agent has independent unread tracking."""
        # Agent A joins and reads everything
        self.fixture.set_agent("atlas-001")
        channel_join("dev", display_name="Atlas")
        channel_post("dev", "message 1")
        channel_read("dev")  # Atlas reads → cursor advances

        # Agent B joins — should see message 1 as unread
        self.fixture.set_agent("apollo-001")
        channel_join("dev", display_name="Apollo")

        # New message after both joined
        self.fixture.set_agent("atlas-001")
        channel_post("dev", "message 2")

        # Apollo's unread should include message 2
        self.fixture.set_agent("apollo-001")
        unread = channel_unread_read()
        self.assertIn("message 2", unread)

    def test_claim_roundtrip(self):
        """Agent claims a task, other agent sees it as claimed."""
        self.fixture.set_agent("atlas-001")
        channel_join("dev", display_name="Atlas")
        channel_post("dev", "Fix bug #123")

        # Atlas claims the message
        result = channel_claim("dev", "123", agent_name="atlas-001")
        self.assertIn("Claimed", result)

        # Apollo tries to claim same message
        self.fixture.set_agent("apollo-001")
        channel_join("dev", display_name="Apollo")
        result2 = channel_claim("dev", "123", agent_name="apollo-001")
        self.assertIn("already", result2.lower())

    def test_who_shows_both_agents(self):
        """channel_who shows all joined agents."""
        self.fixture.set_agent("atlas-001")
        channel_join("dev", display_name="Atlas")

        self.fixture.set_agent("apollo-001")
        channel_join("dev", display_name="Apollo")

        result = channel_who("dev")
        self.assertIn("Atlas", result)
        self.assertIn("Apollo", result)

    def test_multiple_channels(self):
        """Agents can operate on multiple channels independently."""
        self.fixture.set_agent("atlas-001")
        channel_join("dev", display_name="Atlas")
        channel_join("ops", display_name="Atlas")
        channel_post("dev", "dev message")
        channel_post("ops", "ops message")

        self.fixture.set_agent("apollo-001")
        channel_join("dev", display_name="Apollo")

        dev_result = channel_read("dev")
        self.assertIn("dev message", dev_result)
        # Apollo didn't join ops — shouldn't affect dev


if __name__ == "__main__":
    unittest.main()
