"""A leading '#' names the same channel as the bare name.

'#dev' is how people write a channel name in prose; it must not open a
second log that readers of 'dev' never see.
"""

import tempfile
import unittest

from synapt.recall import channel as ch
from synapt.recall.channel import (
    _channel_path,
    _channels_dir,
    _read_messages,
    channel_join,
    channel_post,
    channel_read,
    channel_unread,
)
from tests.recall.test_channel import _patch_data_dir


class TestNormalizeChannelName(unittest.TestCase):
    def test_strips_leading_hash(self):
        self.assertEqual(ch._normalize_channel_name("#dev"), "dev")

    def test_strips_repeated_hash(self):
        self.assertEqual(ch._normalize_channel_name("##dev"), "dev")

    def test_bare_name_unchanged(self):
        self.assertEqual(ch._normalize_channel_name("dev"), "dev")

    def test_lone_hash_unchanged(self):
        self.assertEqual(ch._normalize_channel_name("#"), "#")

    def test_inner_hash_unchanged(self):
        self.assertEqual(ch._normalize_channel_name("a#b"), "a#b")


class TestHashPrefixedChannel(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def _log_names(self):
        return sorted(p.name for p in _channels_dir().glob("*.jsonl"))

    def test_post_hash_lands_in_bare_log(self):
        channel_post("#dev", "hello", agent_name="a")
        self.assertEqual(self._log_names(), ["dev.jsonl"])
        self.assertEqual([m.body for m in _read_messages(_channel_path("dev")) if m.type == "message"], ["hello"])

    def test_read_hash_sees_bare_posts(self):
        channel_post("dev", "from bare", agent_name="a")
        self.assertIn("from bare", channel_read("#dev", agent_name="b"))

    def test_join_hash_counts_unread_under_bare(self):
        channel_join("#dev", agent_name="b")
        channel_post("dev", "one", agent_name="a")
        counts = channel_unread(agent_name="b")
        self.assertNotIn("#dev", counts)
        self.assertEqual(counts.get("dev"), 1)


if __name__ == "__main__":
    unittest.main()


class TestRecallChannelHashPrefix(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_mcp_post_hash_lands_in_bare_log(self):
        from synapt.recall.server import recall_channel

        recall_channel(action="post", channel="#dev", message="via mcp")
        names = sorted(p.name for p in _channels_dir().glob("*.jsonl"))
        self.assertEqual(names, ["dev.jsonl"])
