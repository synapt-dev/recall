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


class TestEveryEntryPointNormalizes(unittest.TestCase):
    """Each public call that names a channel treats '#dev' as 'dev'.

    One row per entry point: the library is called directly, not through
    the MCP dispatcher, and the effect is looked for under the bare name.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def _log_names(self):
        return sorted(p.name for p in _channels_dir().glob("*.jsonl"))

    def _post_id(self, body="hello"):
        channel_post("dev", body, agent_name="a")
        return [m.id for m in _read_messages(_channel_path("dev")) if m.body == body][-1]

    def _rows(self, sql, *args):
        conn = ch._open_db()
        try:
            return [tuple(r) for r in conn.execute(sql, args).fetchall()]
        finally:
            conn.close()

    def test_pin(self):
        mid = self._post_id()
        self.assertIn("Pinned", ch.channel_pin("#dev", mid, agent_name="a"))
        self.assertEqual(self._rows("SELECT channel FROM pins"), [("dev",)])

    def test_unpin(self):
        mid = self._post_id()
        ch.channel_pin("dev", mid, agent_name="a")
        self.assertIn("Unpinned", ch.channel_unpin("#dev", mid, agent_name="a"))
        self.assertEqual(self._rows("SELECT channel FROM pins"), [])

    def test_directive(self):
        ch.channel_directive("#dev", "do it", to="b", agent_name="a")
        self.assertEqual(self._log_names(), ["dev.jsonl"])

    def test_mute(self):
        ch.channel_mute("s_b", "#dev", agent_name="a")
        self.assertEqual(self._rows("SELECT channel FROM mutes"), [("dev",)])

    def test_unmute(self):
        ch.channel_mute("s_b", "dev", agent_name="a")
        ch.channel_unmute("s_b", "#dev", agent_name="a")
        self.assertEqual(self._rows("SELECT channel FROM mutes"), [])

    def test_kick(self):
        channel_join("dev", agent_name="s_b")
        ch.channel_kick("s_b", "#dev", agent_name="a")
        self.assertEqual(
            self._rows("SELECT channel FROM memberships WHERE agent_id = ?", "s_b"), []
        )
        self.assertEqual(self._log_names(), ["dev.jsonl"])

    def test_board(self):
        ch.channel_board("#dev", "on it", agent_name="a")
        self.assertIn("on it", ch.channel_board("dev", agent_name="b"))

    def test_claim(self):
        mid = self._post_id()
        ch.channel_claim(mid, "#dev", agent_name="b")
        self.assertEqual(self._rows("SELECT channel FROM claims"), [("dev",)])
        self.assertEqual(self._log_names(), ["dev.jsonl"])

    def test_claim_intent(self):
        ch.channel_claim_intent("file an issue", "#dev", agent_name="a")
        self.assertEqual(self._log_names(), ["dev.jsonl"])

    def test_messages_json(self):
        self._post_id("seen")
        bodies = [m["body"] for m in ch.channel_messages_json("#dev")]
        self.assertIn("seen", bodies)

    def test_read_message(self):
        mid = self._post_id("seen")
        self.assertIn("seen", ch.channel_read_message(mid, "#dev"))

    def test_no_entry_point_left_out(self):
        """A new public channel_* call that names a channel must normalize it."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(ch))
        missing = []
        for node in tree.body:
            if not (isinstance(node, ast.FunctionDef) and node.name.startswith("channel_")):
                continue
            if "channel" not in [a.arg for a in node.args.args]:
                continue
            calls = {
                c.func.id
                for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            }
            if "_normalize_channel_name" not in calls:
                missing.append(node.name)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
