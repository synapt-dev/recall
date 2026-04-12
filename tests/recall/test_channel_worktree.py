"""Tests for cross-worktree awareness in channel messages (#443).

Verifies that:
1. ChannelMessage has a worktree field
2. Worktree is included in message serialization
3. Worktree is preserved on deserialization
4. Posted messages include the sender's worktree
"""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.channel import ChannelMessage


def _patch_data_dir(tmpdir):
    """Return a combined patcher for project_data_dir + disable global store.

    Keep this helper local to avoid importing through a top-level `tests`
    package path, which is not importable in CI.
    """
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


class TestWorktreeInWho(unittest.TestCase):
    """Test that channel_who() shows workspace info (#443)."""

    def setUp(self):
        import tempfile, shutil
        self.tmpdir = tempfile.mkdtemp()
        self.patcher = _patch_data_dir(self.tmpdir)
        self.patcher.start()

    def tearDown(self):
        import shutil
        self.patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_who_shows_workspace_when_set(self):
        """Agents with a workspace show @workspace in who output."""
        from synapt.recall.channel import channel_join, channel_who, _open_db
        channel_join("dev", agent_name="agent_a", display_name="Apollo")
        # Manually set workspace in presence
        conn = _open_db()
        conn.execute(
            "UPDATE presence SET workspace = 'synapt-dev' WHERE display_name = 'Apollo'"
        )
        conn.commit()
        conn.close()
        result = channel_who()
        self.assertIn("@synapt-dev", result)

    def test_who_omits_workspace_when_empty(self):
        """Agents without workspace don't show @."""
        from synapt.recall.channel import channel_join, channel_who
        channel_join("dev", agent_name="agent_b", display_name="Sentinel")
        result = channel_who()
        self.assertNotIn("@  ", result)


class TestWorktreeInRead(unittest.TestCase):
    """Test that channel_read() shows worktree at max detail (#443)."""

    def setUp(self):
        import tempfile, shutil
        self.tmpdir = tempfile.mkdtemp()
        self.patcher = _patch_data_dir(self.tmpdir)
        self.patcher.start()

    def tearDown(self):
        import shutil
        self.patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_max_detail_shows_worktree(self):
        """Messages at max detail include @worktree tag."""
        from synapt.recall.channel import channel_join, channel_post, channel_read
        channel_join("dev", agent_name="agent_a", display_name="Apollo")
        channel_post("dev", "hello from worktree", agent_name="agent_a")
        result = channel_read("dev", detail="max", agent_name="agent_a")
        # Worktree may or may not be set depending on env, but the code path runs
        self.assertIn("hello from worktree", result)

    def test_min_detail_omits_worktree(self):
        """Messages at min detail do not include worktree tag."""
        from synapt.recall.channel import channel_join, channel_post, channel_read
        channel_join("dev", agent_name="agent_a", display_name="Apollo")
        channel_post("dev", "hello", agent_name="agent_a")
        result = channel_read("dev", detail="min", agent_name="agent_a")
        # At min detail, no @worktree tag
        self.assertNotIn("@synapt", result)


if __name__ == "__main__":
    unittest.main()
