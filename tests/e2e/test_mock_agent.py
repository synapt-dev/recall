"""E2E test: mock agent that reads directives and responds via channels.

Story #5: agent e2e with mock agent. A mock agent process polls for
directives, processes them, and posts responses. Tests the full
agent loop without billing or real AI models.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.channel import (
    _open_db,
    channel_directive,
    channel_join,
    channel_post,
    channel_read,
    channel_unread_read,
    check_directives,
)


class GripspaceFixture:
    """Reusable test fixture for a minimal gripspace with mock agents."""

    def __init__(self):
        self.tmpdir = tempfile.mkdtemp()
        self.gripspace_root = Path(self.tmpdir) / "gripspace"
        self.data_dir = self.gripspace_root / ".synapt" / "recall"
        self.data_dir.mkdir(parents=True)

    def patch_data_dir(self):
        return patch(
            "synapt.recall.channel.project_data_dir",
            return_value=self.data_dir,
        )

    def patch_manifest(self):
        return patch(
            "synapt.recall.channel._read_manifest_url",
            return_value=None,
        )

    def set_agent(self, agent_id: str):
        os.environ["SYNAPT_AGENT_ID"] = agent_id

    def clear_agent(self):
        os.environ.pop("SYNAPT_AGENT_ID", None)
        from synapt.recall.channel import _AGENT_ID_CACHE
        _AGENT_ID_CACHE.clear()


def mock_agent_loop(
    fixture: GripspaceFixture,
    agent_id: str,
    display_name: str,
    stop_event: threading.Event,
    responses: list[str],
):
    """Simulate an agent: poll for directives, respond to each.

    Runs in a thread. Polls check_directives, processes any found,
    posts a response, then continues until stop_event is set.
    """
    fixture.set_agent(agent_id)
    channel_join("dev", display_name=display_name)

    while not stop_event.is_set():
        result = check_directives(agent_name=agent_id)
        if result and "directive" in result.lower():
            # Extract the directive body and respond
            channel_post("dev", f"Acknowledged — working on it", agent_name=agent_id)
            responses.append(result)
        stop_event.wait(0.05)  # Poll interval


class TestMockAgent(unittest.TestCase):
    """E2E: mock agent processes directives and responds."""

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

    def test_agent_receives_directive_and_responds(self):
        """Send a directive to a mock agent, verify it responds."""
        # Setup: both agents join
        channel_join("dev", agent_name="atlas-001", display_name="Atlas")
        channel_join("dev", agent_name="layne-001", display_name="Layne", role="human")

        # Human sends a directive to Atlas
        channel_directive("dev", "Build the auth module", to="atlas-001",
                          agent_name="layne-001")

        # Agent checks for directives
        result = check_directives(agent_name="atlas-001")
        self.assertIn("Build the auth module", result)

        # Agent responds
        channel_post("dev", "Acknowledged — working on it", agent_name="atlas-001")

        # Human reads the channel
        read_result = channel_read("dev", agent_name="layne-001")
        self.assertIn("Acknowledged", read_result)

    def test_agent_ignores_directives_for_others(self):
        """Agent only processes directives addressed to them."""
        channel_join("dev", agent_name="atlas-001", display_name="Atlas")
        channel_join("dev", agent_name="layne-001", display_name="Layne", role="human")

        # Send directive to Apollo (not Atlas)
        channel_directive("dev", "Do something", to="apollo-001",
                          agent_name="layne-001")

        # Atlas checks — should not see it
        result = check_directives(agent_name="atlas-001")
        self.assertNotIn("Do something", result or "")

    def test_multiple_agents_get_own_directives(self):
        """Each agent only sees directives addressed to them."""
        channel_join("dev", agent_name="atlas-001", display_name="Atlas")
        channel_join("dev", agent_name="apollo-001", display_name="Apollo")
        channel_join("dev", agent_name="layne-001", display_name="Layne", role="human")

        channel_directive("dev", "Atlas task", to="atlas-001",
                          agent_name="layne-001")
        channel_directive("dev", "Apollo task", to="apollo-001",
                          agent_name="layne-001")

        atlas_result = check_directives(agent_name="atlas-001")
        apollo_result = check_directives(agent_name="apollo-001")

        self.assertIn("Atlas task", atlas_result or "")
        self.assertIn("Apollo task", apollo_result or "")


if __name__ == "__main__":
    unittest.main()
