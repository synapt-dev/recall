"""recall_channel MCP results say a thing once, not on every call.

The store preamble prints on the first call of a session and again only when
the store it names changes; an unread @mention prints in full once, then as a
one-line count with a read hint until it is read. A mention the session has
read (read_message) or answered (a post naming its id) is not listed again.
"""

from __future__ import annotations

import os
import unittest

from _isolation_helpers import owned_store

BODY = "@reader please take the next story; the bar is written here in full"
LATER = "@reader one more ask, arriving after the first was shown"


class TestRecallChannelSaysItOnce(unittest.TestCase):
    def setUp(self) -> None:
        from synapt.recall import server
        from synapt.recall.actions import reset_action_registry

        self._store = owned_store()
        self._previous_agent_id = os.environ.get("SYNAPT_AGENT_ID")
        self._server = server
        self._saved = (getattr(server, "_last_store_preamble", None),
                       set(getattr(server, "_shown_mentions", ())),
                       server._is_solo_mode,
                       set(getattr(server, "_cleared_mentions", ())))
        server._last_store_preamble = None
        server._shown_mentions = set()
        server._cleared_mentions = set()
        # The fixture keeps channel logs in the shared dir; the solo probe
        # only looks at the local one and would skip the mention check.
        server._is_solo_mode = lambda: False
        reset_action_registry()
        self.call = server._with_directive_check(server.recall_channel)

        from synapt.recall.channel import _open_db, channel_join, channel_read

        self._as("s_lead")
        channel_join("dev")
        self._as("s_reader")
        channel_join("dev")
        conn = _open_db()
        conn.execute("UPDATE presence SET display_name = 'reader' WHERE agent_id = 's_reader'")
        conn.commit()
        conn.close()
        channel_read("dev")
        self.first_id = self._post(BODY)

    def tearDown(self) -> None:
        from synapt.recall.actions import reset_action_registry

        server = self._server
        (server._last_store_preamble, shown, server._is_solo_mode,
         cleared) = self._saved
        server._shown_mentions = shown
        server._cleared_mentions = cleared
        reset_action_registry()
        if self._previous_agent_id is None:
            os.environ.pop("SYNAPT_AGENT_ID", None)
        else:
            os.environ["SYNAPT_AGENT_ID"] = self._previous_agent_id
        self._store.restore()

    def _as(self, agent: str) -> None:
        os.environ["SYNAPT_AGENT_ID"] = agent

    def _post(self, body: str) -> str:
        from synapt.recall.channel import channel_post

        self._as("s_lead")
        out = channel_post("dev", body)
        self._as("s_reader")
        return out.rsplit("[id=", 1)[1].rstrip("]")

    def _who(self) -> str:
        return self.call(action="who", channel="dev")

    def test_first_call_shows_preamble_and_full_mention(self) -> None:
        out = self._who()
        self.assertIn("Channel state store: ", out)
        self.assertIn("Channel log store: ", out)
        self.assertIn(BODY, out)

    def test_second_call_shows_count_line_only(self) -> None:
        self._who()
        out = self._who()
        self.assertNotIn("Channel state store", out)
        self.assertNotIn("Channel log store", out)
        self.assertNotIn(BODY, out)
        count = [ln for ln in out.splitlines() if "@mention" in ln]
        self.assertEqual(len(count), 1, out)
        self.assertIn("1 unread @mention(s) shown in full earlier", count[0])
        self.assertIn('recall_channel(action="read", channel=<ch>)', count[0])

    def test_new_mention_after_count_is_shown_in_full(self) -> None:
        self._who()
        self._who()
        self._post(LATER)
        out = self._who()
        self.assertIn(LATER, out)
        self.assertNotIn(BODY, out)
        self.assertIn("1 unread @mention(s) shown in full earlier", out)
        again = self._who()
        self.assertNotIn(LATER, again)
        self.assertIn("2 unread @mention(s) shown in full earlier", again)

    def test_store_change_reprints_preamble(self) -> None:
        self._who()
        self.assertNotIn("Channel state store", self._who())
        second = owned_store()
        try:
            out = self._who()
        finally:
            second.restore()
        self.assertIn(f"Channel state store: {second.root.resolve()}", out)
        self.assertIn(f"Channel log store: {(second.root / 'channels').resolve()}", out)

    def test_reading_the_channel_clears_the_count(self) -> None:
        self._who()
        self.call(action="read", channel="dev")
        out = self._who()
        self.assertNotIn("@mention", out)
        self.assertNotIn(BODY, out)

    def test_read_message_clears_the_mention(self) -> None:
        self.assertIn(BODY, self._who())
        self.call(action="read_message", channel="dev", message=self.first_id)
        out = self._who()
        self.assertNotIn("@mention", out)
        self.assertNotIn(BODY, out)

    def test_read_message_that_finds_nothing_clears_nothing(self) -> None:
        self._who()
        self.call(action="read_message", channel="ops", message=self.first_id)
        self.assertIn("1 unread @mention(s) shown in full earlier", self._who())

    def test_a_post_naming_the_id_clears_the_mention(self) -> None:
        self._who()
        self.call(action="post", channel="dev",
                  message=f"done, re {self.first_id}")
        out = self._who()
        self.assertNotIn("@mention", out)

    def test_a_post_without_a_body_still_names_the_missing_message(self) -> None:
        out = self.call(action="post", channel="dev")
        self.assertIn("message is required", out)

    def test_new_mention_still_shown_after_an_old_one_is_read(self) -> None:
        self._who()
        self.call(action="read_message", channel="dev", message=self.first_id)
        self._post(LATER)
        out = self._who()
        self.assertIn("1 @mention(s):", out)
        self.assertIn(LATER, out)
        self.assertNotIn(BODY, out)


if __name__ == "__main__":
    unittest.main()
