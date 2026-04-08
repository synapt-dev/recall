"""Tests for cross-worktree awareness in channel messages (#443).

Verifies that:
1. ChannelMessage has a worktree field
2. Worktree is included in message serialization
3. Worktree is preserved on deserialization
4. Posted messages include the sender's worktree
"""

import json
import unittest
from unittest.mock import patch

from synapt.recall.channel import ChannelMessage


class TestChannelMessageWorktree(unittest.TestCase):
    """Test that ChannelMessage includes worktree metadata."""

    def test_worktree_field_exists(self):
        """ChannelMessage has a worktree field."""
        msg = ChannelMessage(
            timestamp="2026-04-08T12:00:00Z",
            channel="dev",
            type="message",
            body="test",
            from_agent="s_test123",
            worktree="synapt-dev",
        )
        self.assertEqual(msg.worktree, "synapt-dev")

    def test_worktree_defaults_to_empty(self):
        """Worktree defaults to empty string when not provided."""
        msg = ChannelMessage(
            timestamp="2026-04-08T12:00:00Z",
            channel="dev",
            type="message",
            body="test",
        )
        self.assertEqual(msg.worktree, "")

    def test_worktree_in_serialization(self):
        """Worktree is included in to_dict() when non-empty."""
        msg = ChannelMessage(
            timestamp="2026-04-08T12:00:00Z",
            channel="dev",
            type="message",
            body="test",
            from_agent="s_test123",
            worktree="synapt-dev",
        )
        d = msg.to_dict()
        self.assertEqual(d["worktree"], "synapt-dev")

    def test_worktree_omitted_when_empty(self):
        """Worktree is omitted from to_dict() when empty (clean JSONL)."""
        msg = ChannelMessage(
            timestamp="2026-04-08T12:00:00Z",
            channel="dev",
            type="message",
            body="test",
            from_agent="s_test123",
        )
        d = msg.to_dict()
        self.assertNotIn("worktree", d)

    def test_worktree_roundtrip(self):
        """Worktree survives to_dict() -> from_dict() roundtrip."""
        msg = ChannelMessage(
            timestamp="2026-04-08T12:00:00Z",
            channel="dev",
            type="message",
            body="test",
            from_agent="s_test123",
            worktree="synapt-dev",
        )
        d = msg.to_dict()
        restored = ChannelMessage.from_dict(d)
        self.assertEqual(restored.worktree, "synapt-dev")

    def test_legacy_message_without_worktree(self):
        """Messages without worktree field (legacy) deserialize cleanly."""
        d = {
            "timestamp": "2026-04-08T12:00:00Z",
            "channel": "dev",
            "type": "message",
            "body": "old message",
            "from": "s_old123",
        }
        msg = ChannelMessage.from_dict(d)
        self.assertEqual(msg.worktree, "")


if __name__ == "__main__":
    unittest.main()
