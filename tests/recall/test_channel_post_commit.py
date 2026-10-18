"""A channel post reports what happened to the log, not to the state store.

The JSONL line is the post. The SQLite writes that follow it (unread flags,
presence, pins) are bookkeeping; when another connection held the write lock
they raised ``database is locked`` after the line had landed, so a caller saw a
failure for a post that was there, retried, and the channel got the same body
twice. These rows hold a second connection's write lock while a post lands.
"""

import json
import re
import shutil
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall import channel as ch
from synapt.recall.channel import _channel_path, _db_path, _open_db, channel_post


class PostCommitTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        data_dir = Path(self.tmpdir) / "project" / ".synapt" / "recall"
        self.patchers = [
            patch("synapt.recall.channel.project_data_dir", return_value=data_dir),
            patch("synapt.recall.channel._read_manifest_url", return_value=None),
        ]
        for p in self.patchers:
            p.start()
        # A member besides the poster, so the post has unread flags to set.
        conn = _open_db()
        conn.execute(
            "INSERT INTO memberships (agent_id, channel, joined_at) VALUES (?, ?, ?)",
            ("a_other", "dev", "2026-01-01T00:00:00.000000Z"),
        )
        conn.commit()
        conn.close()
        self.holder = None

    def tearDown(self):
        self._release()
        for p in reversed(self.patchers):
            p.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _hold_write_lock(self):
        self.holder = sqlite3.connect(str(_db_path()), timeout=0, isolation_level=None)
        self.holder.execute("BEGIN IMMEDIATE")

    def _release(self):
        if self.holder is not None:
            self.holder.execute("ROLLBACK")
            self.holder.close()
            self.holder = None

    def _lines(self):
        path = _channel_path("dev")
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]

    @staticmethod
    def _id(result: str) -> str:
        return re.search(r"\[id=(m_[0-9a-f]{8})\]", result).group(1)

    def test_r1_post_that_lands_under_a_held_lock_returns_its_id(self):
        self._hold_write_lock()
        result = channel_post(channel="dev", message="landed", display_name="poster")
        lines = self._lines()
        self.assertEqual([l["body"] for l in lines], ["landed"])
        self.assertEqual(self._id(result), lines[0]["id"])
        self.assertIn("database is locked", result)

    def test_r2_same_body_posted_twice_under_a_held_lock_logs_both(self):
        self._hold_write_lock()
        first = channel_post(channel="dev", message="once", display_name="poster")
        second = channel_post(channel="dev", message="once", display_name="poster")
        lines = self._lines()
        self.assertEqual([l["body"] for l in lines], ["once", "once"])
        self.assertEqual([self._id(first), self._id(second)], [l["id"] for l in lines])
        self.assertNotEqual(self._id(first), self._id(second))

    def test_r3_pin_not_recorded_is_not_reported_pinned(self):
        self._hold_write_lock()
        result = channel_post(channel="dev", message="pin me", display_name="poster", pin=True)
        self._release()
        self.assertNotIn("(pinned)", result)
        self.assertIn("pin not recorded", result)
        conn = _open_db()
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM pins").fetchone()[0], 0)
        finally:
            conn.close()

    def test_r4_pin_posted_again_after_an_unrecorded_pin_is_its_own_post(self):
        self._hold_write_lock()
        first = channel_post(channel="dev", message="pin me", display_name="poster", pin=True)
        self._release()
        second = channel_post(channel="dev", message="pin me", display_name="poster", pin=True)
        self.assertTrue(second.endswith(f"(pinned) [id={self._id(second)}]"), second)
        self.assertNotEqual(self._id(first), self._id(second))
        self.assertEqual(len(self._lines()), 2)
        conn = _open_db()
        try:
            rows = conn.execute("SELECT message_id FROM pins").fetchall()
        finally:
            conn.close()
        self.assertEqual([r[0] for r in rows], [self._id(second)])

    def test_r5_post_that_did_not_land_fails(self):
        with patch.object(ch, "_guard_store_path", side_effect=OSError("disk gone")):
            with self.assertRaises(OSError):
                channel_post(channel="dev", message="lost", display_name="poster")
        self.assertEqual(self._lines(), [])

    def test_r6_clean_post_sets_unread_and_says_nothing_extra(self):
        result = channel_post(channel="dev", message="clean", display_name="poster")
        self.assertRegex(result, r"^\[#dev\] poster: clean \[id=m_[0-9a-f]{8}\]$")
        conn = _open_db()
        try:
            dirty = conn.execute(
                "SELECT dirty FROM unread_flags WHERE agent_id = 'a_other'"
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(dirty, 1)

    def test_r7_same_body_from_another_author_is_its_own_post(self):
        channel_post(channel="dev", message="same", display_name="poster")
        channel_post(channel="dev", message="same", display_name="other")
        self.assertEqual(len(self._lines()), 2)

    def test_r9_same_body_of_another_type_is_its_own_post(self):
        channel_post(channel="dev", message="typed", display_name="poster")
        channel_post(channel="dev", message="typed", display_name="poster", msg_type="status")
        self.assertEqual(len(self._lines()), 2)

    def test_r10_same_pin_posted_twice_is_two_pins(self):
        first = channel_post(channel="dev", message="pin twice", display_name="poster", pin=True)
        second = channel_post(channel="dev", message="pin twice", display_name="poster", pin=True)
        self.assertNotEqual(self._id(first), self._id(second))
        conn = _open_db()
        try:
            rows = conn.execute("SELECT message_id FROM pins ORDER BY rowid").fetchall()
        finally:
            conn.close()
        self.assertEqual([r[0] for r in rows], [self._id(first), self._id(second)])

    def test_r11_posts_with_attachments_are_each_logged(self):
        att = Path(self.tmpdir) / "note.txt"
        att.write_text("x", encoding="utf-8")
        a = channel_post(channel="dev", message="see file", display_name="poster",
                         attachment_paths=[str(att)])
        b = channel_post(channel="dev", message="see file", display_name="poster",
                         attachment_paths=[str(att)])
        self.assertEqual(len(self._lines()), 2)
        self.assertNotEqual(self._id(a), self._id(b))

    def test_r12_another_body_from_the_same_sender_is_its_own_post(self):
        channel_post(channel="dev", message="first", display_name="poster")
        channel_post(channel="dev", message="second", display_name="poster")
        self.assertEqual([l["body"] for l in self._lines()], ["first", "second"])

    def test_r13_unread_flags_not_set_but_pin_recorded(self):
        with patch.object(ch, "_set_dirty_flags",
                          side_effect=sqlite3.OperationalError("database is locked")):
            result = channel_post(channel="dev", message="half", display_name="poster", pin=True)
        self.assertIn("(pinned)", result)
        self.assertNotIn("pin not recorded", result)
        self.assertIn("channel state not updated: database is locked", result)
        self.assertEqual(len(self._lines()), 1)

    def test_r14_identical_posts_a_second_apart_both_land(self):
        first = channel_post(channel="dev", message="ack", display_name="poster")
        time.sleep(1)
        second = channel_post(channel="dev", message="ack", display_name="poster")
        lines = self._lines()
        self.assertEqual([l["body"] for l in lines], ["ack", "ack"])
        self.assertEqual([self._id(first), self._id(second)], [l["id"] for l in lines])

    def test_r15_other_error_setting_unread_flags_is_reported_not_raised(self):
        with patch.object(ch, "_set_dirty_flags", side_effect=RuntimeError("flags broke")):
            result = channel_post(channel="dev", message="landed", display_name="poster")
        lines = self._lines()
        self.assertEqual([l["body"] for l in lines], ["landed"])
        self.assertEqual(self._id(result), lines[0]["id"])
        self.assertIn("channel state not updated: flags broke", result)

    def test_r16_other_error_opening_the_store_is_reported_not_raised(self):
        real_open = ch._open_db

        def open_db(*args, **kwargs):
            if self._lines():
                raise RuntimeError("store broke")
            return real_open(*args, **kwargs)

        with patch.object(ch, "_open_db", side_effect=open_db):
            result = channel_post(channel="dev", message="pin me", display_name="poster", pin=True)
        lines = self._lines()
        self.assertEqual([l["body"] for l in lines], ["pin me"])
        self.assertEqual(self._id(result), lines[0]["id"])
        self.assertNotIn("(pinned)", result)
        self.assertIn("pin not recorded, channel state not updated: store broke", result)

    def test_r17_two_pinned_posts_with_one_id_record_one_pin(self):
        one_id = ch._generate_msg_id("2026-01-01T00:00:00.000000Z", "a_poster", "pin")
        with patch.object(ch, "_generate_msg_id", return_value=one_id):
            channel_post(channel="dev", message="pin a", display_name="poster", pin=True)
            channel_post(channel="dev", message="pin b", display_name="poster", pin=True)
        self.assertEqual(len(self._lines()), 2)
        conn = _open_db()
        try:
            rows = conn.execute("SELECT message_id, body FROM pins").fetchall()
        finally:
            conn.close()
        self.assertEqual([tuple(r) for r in rows], [(one_id, "pin a")])


if __name__ == "__main__":
    unittest.main()


class PostLandedNotesTest(unittest.TestCase):
    """What a caller is told when the line landed and a later step did not."""

    # The store fixture only; the PostCommitTest rows are not run twice.
    setUp = PostCommitTest.setUp
    tearDown = PostCommitTest.tearDown
    _hold_write_lock = PostCommitTest._hold_write_lock
    _release = PostCommitTest._release
    _lines = PostCommitTest._lines

    MENTION_NOTE = (
        "@mention not recorded: the addressee was not notified; "
        "ping them citing this id, do not re-post"
    )

    def _count(self, sql):
        conn = _open_db()
        try:
            return conn.execute(sql).fetchone()[0]
        finally:
            conn.close()

    def test_r18_intent_under_a_held_lock_reports_the_unrecorded_claim(self):
        self._hold_write_lock()
        ok, result = ch.channel_claim_intent("file perf issue", agent_name="a_me")
        self._release()
        self.assertTrue(ok)
        self.assertIn("(posted; ", result)
        self.assertIn("claim not recorded: database is locked", result)
        self.assertEqual([l["type"] for l in self._lines()], ["intent"])

    def test_r19_clean_intent_says_nothing_extra(self):
        ok, result = ch.channel_claim_intent("file perf issue", agent_name="a_me")
        self.assertTrue(ok)
        self.assertNotIn("posted;", result)
        self.assertEqual(self._count("SELECT count(*) FROM claims"), 1)

    def test_r20_broadcast_under_a_held_lock_carries_each_channel_note(self):
        channel_post("dev", "seed", agent_name="a_me")
        channel_post("ops", "seed", agent_name="a_me")
        self._hold_write_lock()
        result = ch.channel_broadcast("hello all", agent_name="a_me")
        self._release()
        self.assertTrue(result.startswith("Broadcast to 2 channel(s): #dev, #ops ("))
        for c in ("dev", "ops"):
            self.assertIn(f"#{c}: posted; channel state not updated: database is locked", result)

    def test_r21_clean_broadcast_says_nothing_extra(self):
        channel_post("dev", "seed", agent_name="a_me")
        channel_post("ops", "seed", agent_name="a_me")
        result = ch.channel_broadcast("hello all", agent_name="a_me")
        self.assertEqual(result, "Broadcast to 2 channel(s): #dev, #ops")

    def test_r22_mention_under_a_held_lock_says_the_addressee_was_not_told(self):
        self._hold_write_lock()
        result = channel_post("dev", "@a_other please look", agent_name="a_me")
        self._release()
        self.assertIn(self.MENTION_NOTE, result)
        self.assertEqual(self._count("SELECT count(*) FROM mentions"), 0)
        self.assertEqual(len(self._lines()), 1)

    def test_r23_mention_with_a_failing_hook_says_the_addressee_was_not_told(self):
        def bad(msg, pd):
            raise RuntimeError("hook boom")
        ch._message_posted_hooks.append(bad)
        try:
            result = channel_post("dev", "@a_other please look", agent_name="a_me")
            plain = channel_post("dev", "no one named", agent_name="a_me")
        finally:
            ch._message_posted_hooks.remove(bad)
        self.assertIn(f"(posted; {self.MENTION_NOTE})", result)
        # A hook failure on a post that names no one stays best-effort.
        self.assertNotIn("posted;", plain)

    def test_r24_clean_mention_says_nothing_extra(self):
        result = channel_post("dev", "@a_other please look", agent_name="a_me")
        self.assertNotIn("posted;", result)
        self.assertEqual(self._count("SELECT count(*) FROM mentions"), 1)

    def test_r25_mention_with_unset_unread_flags_says_the_addressee_was_not_told(self):
        with patch.object(ch, "_set_dirty_flags", side_effect=RuntimeError("flags broke")):
            result = channel_post("dev", "@a_other please look", agent_name="a_me")
        self.assertIn(f"channel state not updated: flags broke; {self.MENTION_NOTE}", result)

    def test_r26_intent_with_unset_unread_flags_reports_it(self):
        with patch.object(ch, "_set_dirty_flags", side_effect=RuntimeError("flags broke")):
            ok, result = ch.channel_claim_intent("file perf issue", agent_name="a_me")
        self.assertTrue(ok)
        self.assertIn("(posted; channel state not updated: flags broke)", result)
        self.assertEqual(self._count("SELECT count(*) FROM claims"), 1)

    def test_r27_directive_under_a_held_lock_reports_unset_state(self):
        self._hold_write_lock()
        result = ch.channel_directive("dev", "do the thing", to="a_other", agent_name="a_me")
        self._release()
        self.assertEqual([l["type"] for l in self._lines()], ["directive"])
        self.assertIn("(posted; channel state not updated", result)
