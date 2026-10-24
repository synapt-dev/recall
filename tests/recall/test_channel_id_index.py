"""The id index: which posts name an id, kept in step with each room's log.

A post writes its JSONL line, then indexes the ids it names and moves the
room's cursor over that line in one transaction. The cursor only moves over
lines it indexed, so a gap between the cursor and the log's end reads as
behind ("?"), never as "nothing names this id".
"""

import contextlib
import io
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall import channel as ch
from synapt.recall import cli


def _msg(body, sender="a_one", display="", ts="2026-01-01T00:00:00.000000Z", room="dev"):
    return ch.ChannelMessage(timestamp=ts, channel=room, type="message", body=body,
                             from_agent=sender, from_display=display)


class IdIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        home = Path(self.tmpdir) / "home"
        home.mkdir()
        data_dir = Path(self.tmpdir) / "project" / ".synapt" / "recall"
        self.patchers = [
            patch.dict(os.environ, {"HOME": str(home), "USERPROFILE": str(home),
                                    "SYSTEMROOT": os.environ.get("SYSTEMROOT", str(home))}),
            patch("synapt.recall.channel.project_data_dir", return_value=data_dir),
            patch("synapt.recall.channel._read_manifest_url", return_value=None),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in reversed(self.patchers):
            p.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # helpers

    def _post(self, body, **kw):
        msg = _msg(body, **kw)
        err = ch._append_message(msg)
        self.assertIsNone(err)
        return msg

    def _log(self, room="dev"):
        return ch._channel_path(room)

    def _raw_line(self, body, mid, sender="a_other", ts="2026-01-02T00:00:00.000000Z",
                  room="dev"):
        """A line another desk wrote to the shared log, not indexed here."""
        rec = {"timestamp": ts, "channel": room, "type": "message", "body": body,
               "from": sender, "id": mid}
        with open(self._log(room), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _cursor(self, room="dev"):
        conn = sqlite3.connect(str(ch._db_path()))
        try:
            row = conn.execute("SELECT byte_offset, last_message_id FROM id_refs_cursor "
                               "WHERE channel = ?", (room,)).fetchone()
        finally:
            conn.close()
        return row

    def _indexed(self, token):
        conn = sqlite3.connect(str(ch._db_path()))
        try:
            return {r[0] for r in conn.execute(
                "SELECT message_id FROM id_refs WHERE token = ?", (token,))}
        finally:
            conn.close()

    def _cli(self, *argv):
        args = cli.make_parser().parse_args(["channel", "latest", *argv])
        out = io.StringIO()
        code = 0
        with contextlib.redirect_stdout(out):
            try:
                cli.cmd_channel(args)
            except SystemExit as e:
                code = e.code
        return code, out.getvalue()

    # R5

    def test_agent_id_not_display_name(self):
        a = self._post("first about c12.345", display="One (team)",
                       ts="2026-01-01T00:00:01.000000Z")
        b = self._post("second about c12.345", display="one",
                       ts="2026-01-01T00:00:02.000000Z")
        posts = ch.channel_latest("c12.345")["posts"]
        self.assertEqual([p["message_id"] for p in posts], [b.id, a.id])
        self.assertEqual({p["from"] for p in posts}, {"a_one"})
        code, out = self._cli("c12.345")
        self.assertEqual(code, 0)
        self.assertEqual(out.count("a_one:"), 2)
        self.assertNotIn("One (team)", out)

    # R6

    def test_whole_id_match(self):
        self.assertEqual(ch._id_tokens("see 1.23456789 and m_0000000a."),
                         ["1.23456789", "m_0000000a"])
        self.assertEqual(ch._id_tokens("ruled 1.234567."), ["1.234567"])
        self._post("only the longer key 1.23456789")
        self._raw_line("gap line names 1.23456789 too", "m_0000000b")
        self.assertEqual(ch.channel_latest("1.234567"), {"posts": [], "unknown": None})
        code, out = self._cli("1.234567")
        self.assertEqual((code, out), (3, "none\n"))
        self.assertEqual(len(ch.channel_latest("1.23456789")["posts"]), 2)

    # R8

    def test_behind_under_bound_reads_gap(self):
        self._post("indexed m_0000000c")
        before = self._cursor()
        self._raw_line("from another desk m_0000000c", "m_0000000d")
        posts = ch.channel_latest("m_0000000c")["posts"]
        self.assertEqual([p["message_id"] for p in posts][0], "m_0000000d")
        self.assertEqual(len(posts), 2)
        self.assertEqual(self._cursor(), before)  # the read wrote nothing

    def test_behind_over_bound_is_unknown(self):
        self._post("indexed m_0000000c")
        start = os.path.getsize(self._log())
        for i in range(70):
            self._raw_line("x" * 1000 + " m_0000000c", f"m_1{i:07x}")
        gap = os.path.getsize(self._log()) - start
        self.assertGreater(gap, ch.BEHIND_BOUND)
        code, out = self._cli("m_0000000c")
        self.assertEqual((code, out), (7, f"? (dev behind by {gap} bytes)\n"))
        # --catch-up indexes the gap, and the room reads current again
        code, out = self._cli("m_0000000c", "--catch-up", "--n", "100")
        self.assertEqual(code, 0)
        self.assertEqual(len(out.splitlines()), 71)
        self.assertEqual(self._cursor()[0], os.path.getsize(self._log()))

    # R9

    def test_index_absent_is_unknown_and_creates_nothing(self):
        code, out = self._cli("m_0000000e")
        self.assertEqual((code, out), (7, "? (index absent)\n"))
        self.assertFalse(ch._db_path().exists())
        # a store from before the index: tables exist, id_refs does not
        db = ch._db_path()
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE presence (agent_id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()
        self._log().parent.mkdir(parents=True, exist_ok=True)
        self._raw_line("names m_0000000e", "m_0000000f")
        code, out = self._cli("m_0000000e")
        self.assertEqual((code, out), (7, "? (index absent)\n"))
        conn = sqlite3.connect(str(db))
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}
        finally:
            conn.close()
        self.assertEqual(tables, {"presence"})

    # R10

    def test_append_moves_cursor_to_end(self):
        msg = self._post("one post m_00000010")
        self.assertEqual(self._cursor(), (os.path.getsize(self._log()), msg.id))
        self.assertEqual(self._indexed("m_00000010"), {msg.id})

    # R11

    def test_index_error_leaves_line_and_cursor(self):
        self._post("first")
        start = os.path.getsize(self._log())
        msg = _msg("second m_00000011")
        with patch.object(ch, "_insert_id_refs",
                          side_effect=sqlite3.OperationalError("disk I/O error")):
            err = ch._append_message(msg)
        self.assertIsInstance(err, sqlite3.OperationalError)
        lines = self._log().read_text().splitlines()
        self.assertEqual(json.loads(lines[-1])["id"], msg.id)
        self.assertEqual(self._cursor()[0], start)
        with patch.object(ch, "BEHIND_BOUND", 0):
            code, out = self._cli("m_00000011")
        gap = os.path.getsize(self._log()) - start
        self.assertEqual((code, out), (7, f"? (dev behind by {gap} bytes)\n"))

    # R14

    def test_interleaved_appends_never_skip_a_line(self):
        real = ch._index_appended_line
        a = _msg("from A m_000000aa", sender="a_a")
        b = _msg("from B m_000000bb", sender="a_b")
        seen = {}

        def interleave(msg, path, start, end, *args, **kw):
            if msg is a and "a" not in seen:
                seen["a"] = (start, end)
                # A's line is written; B writes and indexes before A indexes.
                self.assertIsNone(ch._append_message(b))
                # and a third desk writes a line this store never indexes
                seen["c_start"] = os.path.getsize(path)
                self._raw_line("from C m_000000cc", "m_000000cc")
            return real(msg, path, start, end, *args, **kw)

        with patch.object(ch, "_index_appended_line", side_effect=interleave):
            self.assertIsNone(ch._append_message(a))
        self.assertEqual(self._indexed("m_000000aa"), {a.id})
        self.assertEqual(self._indexed("m_000000bb"), {b.id})
        cursor = self._cursor()[0]
        # never past a line it did not index: here, A's end (B's line follows)
        self.assertLessEqual(cursor, seen["c_start"])
        self.assertEqual(cursor, seen["a"][1])


if __name__ == "__main__":
    unittest.main()
