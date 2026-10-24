"""resume: what changed since this agent's previous session.

T is the end of the caller's previous transcript (the newest is this
session). Three kinds are read, all read-only: knowledge nodes corrected
after T, posts addressed to the agent id after T, and the ids the agent
cited in its previous session with the newest posts naming each. A kind
that cannot be read prints "?" with the reason, never "none".
"""

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from conftest import assistant_entry, user_text_entry, write_jsonl
from synapt.recall import channel as ch
from synapt.recall import cli
from synapt.recall import resume

ME = "a_me01"
OTHER = "a_other"

# previous session: 10:00 to 11:00; current session from 12:00
S = "2026-03-01T10:00:00Z"
T = "2026-03-01T11:00:00Z"
NOW = "2026-03-01T12:00:00Z"


def _ts(hh_mm: str) -> str:
    return f"2026-03-01T{hh_mm}:00.000000Z"


class WhatChangedTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        home = self.tmpdir / "home"
        home.mkdir()
        self.project = self.tmpdir / "project"
        self.project.mkdir()
        self.data_dir = self.project / ".synapt" / "recall"
        self.index_dir = self.tmpdir / "index"
        self.index_dir.mkdir()
        self.transcripts = self.tmpdir / "transcripts"
        self.transcripts.mkdir()
        self.patchers = [
            mock.patch.dict(os.environ, {"HOME": str(home), "USERPROFILE": str(home),
                                         "SYSTEMROOT": os.environ.get("SYSTEMROOT", str(home)),
                                         "SYNAPT_AGENT_ID": ME}),
            mock.patch("synapt.recall.channel.project_data_dir", return_value=self.data_dir),
            mock.patch("synapt.recall.channel._read_manifest_url", return_value=None),
            mock.patch("synapt.recall.core.project_transcript_dirs_any_root",
                       return_value=[self.transcripts]),
            mock.patch("synapt.recall.codex.discover_codex_sessions", return_value=None),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in reversed(self.patchers):
            p.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # fixtures

    def _transcript(self, session_id, first, last, mtime):
        path = self.transcripts / f"{session_id}.jsonl"
        write_jsonl(path, [
            {**user_text_entry("start", uuid=f"{session_id}-u", ts=first), "sessionId": session_id},
            {**assistant_entry(text="end", uuid=f"{session_id}-a", ts=last), "sessionId": session_id},
        ])
        os.utime(path, (mtime, mtime))

    def _two_sessions(self):
        self._transcript("11111111-0000-4000-8000-000000000001", S, T, 1_000)
        self._transcript("22222222-0000-4000-8000-000000000002", NOW, NOW, 2_000)

    def _knowledge(self, rows):
        """rows: (id, status, superseded_by, created_at, updated_at, sessions)."""
        conn = sqlite3.connect(str(self.index_dir / "recall.db"))
        conn.execute(
            "CREATE TABLE knowledge (rowid INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, "
            "content TEXT, source_sessions TEXT NOT NULL DEFAULT '[]', created_at TEXT, "
            "updated_at TEXT, status TEXT, superseded_by TEXT NOT NULL DEFAULT '')")
        conn.executemany(
            "INSERT INTO knowledge (id, content, status, superseded_by, created_at, updated_at, "
            "source_sessions) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(i, f"fact {i}", st, sb, c, u, json.dumps(ss)) for i, st, sb, c, u, ss in rows])
        conn.commit()
        conn.close()

    def _post(self, body, sender=OTHER, ts=_ts("11:30"), room="dev"):
        msg = ch.ChannelMessage(timestamp=ts, channel=room, type="message", body=body,
                                from_agent=sender, from_display="")
        self.assertIsNone(ch._append_message(msg))
        return msg

    def _display(self, agent_id, name):
        conn = sqlite3.connect(str(ch._db_path()))
        cur = conn.execute("UPDATE presence SET display_name = ? WHERE agent_id = ?", (name, agent_id))
        self.assertEqual(cur.rowcount, 1)
        conn.commit()
        conn.close()

    def _block(self):
        found = resume.gather_what_changed(resume.caller_transcripts(self.project),
                                           self.index_dir)
        return resume.format_what_changed(found)

    def _kind(self, lines, n):
        prefix = f"({n}) "
        at = next(i for i, line in enumerate(lines) if line.startswith(prefix))
        out = [lines[at]]
        for line in lines[at + 1:]:
            if not line.startswith("  "):
                break
            out.append(line)
        return out

    # R1

    def test_one_transcript_is_no_earlier_session(self):
        self._transcript("11111111-0000-4000-8000-000000000001", S, T, 1_000)
        self.assertEqual(self._block(), ["What changed: no earlier session"])

    # R2

    def test_t_is_the_previous_session_end(self):
        self._two_sessions()
        self._knowledge([
            ("kn_after", "superseded", "kn_new", _ts("09:00"), _ts("11:30"), []),
            ("kn_before", "superseded", "kn_new2", _ts("09:00"), _ts("10:30"), []),
        ])
        lines = self._block()
        self.assertEqual(lines[0], "What changed since 2026-03-01T11:00Z (previous session 11111111)")
        nodes = "\n".join(self._kind(lines, 1))
        self.assertIn("kn_after superseded -> kn_new 03-01 11:30 fact kn_after", nodes)
        self.assertNotIn("kn_before", nodes)

    # R3

    def test_correct_and_confirmed_resolve_show_dismissal_does_not(self):
        self._two_sessions()
        self._knowledge([
            ("kn_corrected", "superseded", "kn_succ", _ts("09:00"), _ts("11:20"), []),
            ("kn_resolved", "contradicted", "", _ts("09:00"), _ts("11:40"), []),
            ("kn_dismissed", "active", "", _ts("09:00"), _ts("09:00"), []),
        ])
        nodes = self._kind(self._block(), 1)
        self.assertEqual(nodes[0], "(1) nodes: shown 2 of 2, withheld 0, oldest shown 03-01 11:20")
        self.assertEqual([n.split()[0] for n in nodes[1:]], ["kn_resolved", "kn_corrected"])

    # R4

    def test_addressed_by_agent_id_not_display_name(self):
        self._two_sessions()
        with mock.patch.dict(os.environ, {"SYNAPT_AGENT_ID": ME}):
            ch.channel_join("dev")
        self._display(ME, "Mea Seat")
        to_me = self._post(f"@{ME} please look", ts=_ts("11:10"))
        self._post("-> Mea Seat please look", ts=_ts("11:20"))
        kind = self._kind(self._block(), 2)
        self.assertEqual(kind, ["(2) addressed to me: shown 1 of 1, withheld 0, oldest shown 03-01 11:10",
                                f"  {to_me.id} dev 03-01 11:10"])

    # R7

    def test_cited_id_shows_newest_three_own_included(self):
        self._two_sessions()
        self._post("first note on m_0000aa07", sender=OTHER, ts=_ts("09:30"))
        ruling = self._post("ruling on m_0000aa07", sender=OTHER, ts=_ts("10:10"))
        mine = self._post("RULED, following m_0000aa07", sender=ME, ts=_ts("10:20"))
        later = self._post("after it, m_0000aa07", sender=OTHER, ts=_ts("11:30"))
        kind = self._kind(self._block(), 3)
        self.assertEqual(kind[0], "(3) ids I cited: shown 1 of 1, withheld 0, oldest shown 03-01 10:20")
        self.assertEqual(kind[1], f"  m_0000aa07: {later.id} ({OTHER}, 03-01 11:30), "
                                  f"{mine.id} ({ME}, 03-01 10:20), {ruling.id} ({OTHER}, 03-01 10:10)")

    # R12

    def test_twenty_candidates_give_twelve_lines(self):
        self._two_sessions()
        self._knowledge([(f"kn_{i:02d}", "superseded", "", _ts("09:00"), _ts(f"11:{10 + i:02d}"), [])
                         for i in range(8)])
        with mock.patch.dict(os.environ, {"SYNAPT_AGENT_ID": ME}):
            ch.channel_join("dev")
        self._display(ME, "Mea Seat")
        for i in range(7):
            self._post(f"@{ME} item {i}", ts=_ts(f"11:{20 + i:02d}"))
        for i in range(5):
            self._post(f"cited m_000{i:05d}", sender=ME, ts=_ts(f"10:{10 + i:02d}"))
        lines = self._block()
        self.sample = lines
        self.assertEqual(len(lines), 12)
        labels = [line for line in lines if line.startswith("(")]
        self.assertEqual(labels, [
            "(1) nodes: shown 3 of 8, withheld 5, oldest shown 03-01 11:15",
            "(2) addressed to me: shown 3 of 7, withheld 4, oldest shown 03-01 11:24",
            "(3) ids I cited: shown 2 of 5, withheld 3, oldest shown 03-01 10:13",
        ])
        if os.environ.get("WHAT_CHANGED_SAMPLE"):
            Path(os.environ["WHAT_CHANGED_SAMPLE"]).write_text("\n".join(lines) + "\n")

    # R13

    def test_absent_channel_store_is_unknown_and_not_created(self):
        self._two_sessions()
        self._knowledge([])
        lines = self._block()
        hint = "run synapt recall build or synapt recall channel latest <id> --catch-up"
        self.assertEqual(self._kind(lines, 2), [f"(2) addressed to me: ? (channel store absent; {hint})"])
        self.assertEqual(self._kind(lines, 3), [f"(3) ids I cited: ? (index absent; {hint})"])
        self.assertEqual(self._kind(lines, 1), ["(1) nodes: none since 2026-03-01T11:00Z"])
        self.assertFalse(ch._db_path().exists())
        self.assertFalse(self.data_dir.exists() and any(self.data_dir.rglob("*.db")))

    def test_store_without_id_index_is_unknown_and_gets_no_table(self):
        self._two_sessions()
        self._knowledge([])
        conn = sqlite3.connect(str(ch._db_path().parent.mkdir(parents=True, exist_ok=True)
                                   or ch._db_path()))
        conn.execute("CREATE TABLE mentions (message_id TEXT, channel TEXT, mentioned TEXT, "
                     "timestamp TEXT)")
        conn.commit()
        conn.close()
        lines = self._block()
        self.assertTrue(self._kind(lines, 3)[0].startswith("(3) ids I cited: ? (index absent;"))
        conn = sqlite3.connect(str(ch._db_path()))
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        self.assertEqual(tables, {"mentions"})


# R23

def _build_args(source):
    return argparse.Namespace(source=[str(source)], hf=None, chatgpt_archive=None,
                              no_embeddings=True, incremental=False)


def _build(tmp_path, monkeypatch, catch_up):
    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "src"
    source.mkdir()
    write_jsonl(source / "s1.jsonl", [
        user_text_entry("deploy the rollout canary", uuid="u1", ts="2026-03-01T10:00:00Z"),
        assistant_entry(text="rollout canary manifest", uuid="a1", ts="2026-03-01T10:00:30Z"),
    ])
    data_dir = tmp_path / "chan" / ".synapt" / "recall"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("synapt.recall.channel.project_data_dir", lambda *a, **k: data_dir)
    monkeypatch.setattr("synapt.recall.channel._read_manifest_url", lambda *a, **k: None)
    ch._open_db().close()
    with mock.patch("synapt.recall.cli.Path.cwd", return_value=project), \
            mock.patch("synapt.recall.channel.id_refs_catch_up", catch_up):
        cli.cmd_build(_build_args(source))


def test_build_catches_up_the_id_index_once(tmp_path, monkeypatch, capsys):
    catch_up = mock.Mock(return_value={})
    _build(tmp_path, monkeypatch, catch_up)
    assert catch_up.call_count == 1
    assert "[build] Done!" in capsys.readouterr().out


def test_build_catch_up_error_is_printed_not_raised(tmp_path, monkeypatch, capsys):
    catch_up = mock.Mock(side_effect=RuntimeError("index locked"))
    _build(tmp_path, monkeypatch, catch_up)  # returns: exit 0
    captured = capsys.readouterr()
    assert "[build] Done!" in captured.out
    assert "[build] id index catch-up failed: RuntimeError: index locked" in captured.err


if __name__ == "__main__":
    unittest.main()
