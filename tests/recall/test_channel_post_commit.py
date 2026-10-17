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

    def test_r2_retry_of_a_landed_post_does_not_duplicate(self):
        self._hold_write_lock()
        first = channel_post(channel="dev", message="once", display_name="poster")
        second = channel_post(channel="dev", message="once", display_name="poster")
        self._release()
        third = channel_post(channel="dev", message="once", display_name="poster")
        self.assertEqual([l["body"] for l in self._lines()], ["once"])
        self.assertEqual(self._id(first), self._id(second))
        self.assertEqual(self._id(first), self._id(third))

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

    def test_r4_retry_after_unrecorded_pin_pins_the_landed_post(self):
        self._hold_write_lock()
        first = channel_post(channel="dev", message="pin me", display_name="poster", pin=True)
        self._release()
        second = channel_post(channel="dev", message="pin me", display_name="poster", pin=True)
        self.assertTrue(second.endswith(f"(pinned) [id={self._id(first)}]"), second)
        self.assertEqual(len(self._lines()), 1)
        conn = _open_db()
        try:
            rows = conn.execute("SELECT message_id FROM pins").fetchall()
        finally:
            conn.close()
        self.assertEqual([r[0] for r in rows], [self._id(first)])

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

    def test_r8_same_body_after_the_retry_window_is_a_new_post(self):
        first = channel_post(channel="dev", message="again", display_name="poster")
        path = _channel_path("dev")
        rec = self._lines()[0]
        rec["timestamp"] = "2020-01-01T00:00:00.000000Z"
        path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        second = channel_post(channel="dev", message="again", display_name="poster")
        self.assertEqual(len(self._lines()), 2)
        self.assertNotEqual(self._id(first), self._id(second))

    def test_r9_same_body_of_another_type_is_its_own_post(self):
        channel_post(channel="dev", message="typed", display_name="poster")
        channel_post(channel="dev", message="typed", display_name="poster", msg_type="status")
        self.assertEqual(len(self._lines()), 2)

    def test_r10_retried_pin_is_recorded_once(self):
        first = channel_post(channel="dev", message="pin twice", display_name="poster", pin=True)
        second = channel_post(channel="dev", message="pin twice", display_name="poster", pin=True)
        self.assertEqual(self._id(first), self._id(second))
        conn = _open_db()
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM pins").fetchone()[0], 1)
        finally:
            conn.close()

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


if __name__ == "__main__":
    unittest.main()
