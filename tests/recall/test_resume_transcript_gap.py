"""synapt resume prints the transcript gap: what was typed to this desk in the
sessions before the live one, read from the live transcript files rather than
the index, so a wake after an unclean end sees it even when the index is behind.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pytest

from synapt.recall import core
from synapt.recall.resume import (
    caller_transcripts,
    format_transcript_gap,
    gap_hide_pattern,
    gather_transcript_gap,
)


def _write(path: Path, events: list[dict], *, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _user(ts: str, sid: str, text) -> dict:
    return {"type": "user", "sessionId": sid, "timestamp": ts, "message": {"role": "user", "content": text}}


def _write_tool(ts: str, sid: str, file_path: str) -> dict:
    return {
        "type": "assistant", "sessionId": sid, "timestamp": ts,
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Write", "input": {"file_path": file_path}}]},
    }


@pytest.fixture
def desk(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config = tmp_path / "accounts" / "core"
    project = tmp_path / "work" / "proj"
    project.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    monkeypatch.delenv("SYNAPT_GAP_HIDE", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "cccccccc-live")
    slug = core.project_slug(project)
    return project, home / ".claude" / "projects" / slug, config / "projects" / slug


def _three_sessions(desk):
    """Two dead sessions (one under each root) and the live one, which started last."""
    project, home_dir, config_dir = desk
    now = time.time()
    crashed = _write(home_dir / "aaaaaaaa-crashed.jsonl", [
        _user("2026-10-11T18:00:00Z", "aaaaaaaa-crashed", "old, before the gap"),
        _user("2026-10-11T19:10:00Z", "aaaaaaaa-crashed", "please rerun the build"),
        _user("2026-10-11T19:11:00Z", "aaaaaaaa-crashed", "<task-notification>done</task-notification>"),
        _write_tool("2026-10-11T19:12:00Z", "aaaaaaaa-crashed", "/w/reports/a.md"),
    ], mtime=now - 3000)
    respawn = _write(config_dir / "bbbbbbbb-respawn.jsonl", [
        _user("2026-10-11T19:30:00Z", "bbbbbbbb-respawn", [{"type": "text", "text": "WAKE 19:30 read your journal"}]),
        _user("2026-10-11T19:31:00Z", "bbbbbbbb-respawn", "the answer is option b"),
    ], mtime=now - 2000)
    live = _write(config_dir / "cccccccc-live.jsonl", [
        _user("2026-10-11T20:00:00Z", "cccccccc-live", "synapt resume"),
    ], mtime=now)
    return project, crashed, respawn, live


def _since() -> float:
    from synapt.recall.resume import _timestamp_epoch
    return _timestamp_epoch("2026-10-11T19:00:00Z")


def test_the_gap_lists_every_dead_session_oldest_first_and_never_the_live_one(desk):
    project, _, _, _ = _three_sessions(desk)

    gap = gather_transcript_gap(
        caller_transcripts(project), since=_since(), since_source="test", live_session_id="cccccccc-live"
    )

    assert [s.session_id for s in gap.sessions] == ["aaaaaaaa-crashed", "bbbbbbbb-respawn"]
    assert gap.live_session_id == "cccccccc-live"


def test_with_no_live_session_named_the_newest_dead_session_is_still_listed(desk):
    """A terminal run, or a hook before the new transcript has an event: nothing on
    disk is live, so the crash (the newest dead session) must not be taken for it."""
    project, _, _, live = _three_sessions(desk)
    live.unlink()

    gap = gather_transcript_gap(caller_transcripts(project), since=_since(), since_source="test")

    assert [s.session_id for s in gap.sessions] == ["aaaaaaaa-crashed", "bbbbbbbb-respawn"]
    assert "no live session was named, so none is excluded" in format_transcript_gap(gap)


def test_the_live_session_is_excluded_by_id_even_when_its_first_event_is_older(desk):
    project, home_dir, config_dir = desk
    now = time.time()
    _write(config_dir / "dddddddd-live-old-start.jsonl", [
        _user("2026-10-11T19:05:00Z", "dddddddd-live-old-start", "started first, still running"),
    ], mtime=now)
    _write(home_dir / "eeeeeeee-crashed-later.jsonl", [
        _user("2026-10-11T19:40:00Z", "eeeeeeee-crashed-later", "the crash said this"),
    ], mtime=now - 600)

    gap = gather_transcript_gap(
        caller_transcripts(project), since=_since(), since_source="t", live_session_id="dddddddd-live-old-start"
    )

    assert [s.session_id for s in gap.sessions] == ["eeeeeeee-crashed-later"]


def test_messages_before_since_are_not_shown_and_runtime_records_are_hidden_and_counted(desk):
    project, _, _, _ = _three_sessions(desk)

    crashed = gather_transcript_gap(caller_transcripts(project), since=_since(), since_source="t",
                                    live_session_id="cccccccc-live").sessions[0]

    assert [text for _, text in crashed.messages] == ["please rerun the build"]
    assert crashed.hidden == 1
    assert crashed.files_written == ("/w/reports/a.md",)


def test_the_readers_own_pattern_hides_its_machine_pastes(desk):
    project, _, _, _ = _three_sessions(desk)

    shown = gather_transcript_gap(caller_transcripts(project), since=_since(), since_source="t").sessions[1]
    hidden = gather_transcript_gap(
        caller_transcripts(project), since=_since(), since_source="t", hide=gap_hide_pattern(r"^WAKE\b")
    ).sessions[1]

    assert [t for _, t in shown.messages] == ["WAKE 19:30 read your journal", "the answer is option b"]
    assert [t for _, t in hidden.messages] == ["the answer is option b"]
    assert hidden.hidden == 1


def test_a_session_last_written_before_since_is_not_in_the_gap(desk):
    project, crashed, _, _ = _three_sessions(desk)
    os.utime(crashed, (0, 0))   # written long before the gap opened

    gap = gather_transcript_gap(caller_transcripts(project), since=time.time() - 2500, since_source="t",
                                live_session_id="cccccccc-live")

    assert [s.session_id for s in gap.sessions] == ["bbbbbbbb-respawn"]


def _resume_args(**kw) -> argparse.Namespace:
    base = dict(session=None, index=None, turns=10, since=None, project=None, live_session=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_the_resume_command_prints_the_gap_even_when_there_is_no_index(desk, monkeypatch, capsys, tmp_path):
    """The command itself: the gap reads live files, so a missing index must not hide it."""
    from synapt.recall import cli

    project, _, _, _ = _three_sessions(desk)
    monkeypatch.chdir(project)
    monkeypatch.setattr(cli, "_resolve_index_dir", lambda args: tmp_path / "no-index")
    monkeypatch.setattr(cli, "_refuse_if_index_disagrees_with_source", lambda *a, **k: None)

    with pytest.raises(SystemExit):
        cli.cmd_resume(_resume_args(since="2026-10-11T19:00:00Z"))

    out = capsys.readouterr().out
    assert "TRANSCRIPT GAP" in out
    assert "the answer is option b" in out and "please rerun the build" in out
    assert "synapt resume" not in out.split("TRANSCRIPT GAP", 1)[1].split("\n", 1)[1]   # the live session's line


def test_the_gap_starts_at_the_newest_authored_journal_by_default(desk, monkeypatch):
    from synapt.recall import cli
    from synapt.recall.journal import JournalEntry
    from synapt.recall.resume import _timestamp_epoch

    entries = [
        JournalEntry(timestamp="2026-10-11T17:00:00Z", focus="older"),
        JournalEntry(timestamp="2026-10-11T19:20:00Z", focus="newest"),
    ]
    monkeypatch.setattr("synapt.recall.journal._journal_path", lambda *a, **k: None)
    monkeypatch.setattr("synapt.recall.resume.authored_journals", lambda path: entries)

    since, source = cli._gap_since(_resume_args())

    assert since == _timestamp_epoch("2026-10-11T19:20:00Z")
    assert source == "the newest authored journal"


@pytest.mark.parametrize("raw", ["3h", "-3h"])
def test_since_takes_hours_back_with_or_without_a_minus(raw):
    from synapt.recall import cli

    since, _ = cli._gap_since(_resume_args(since=raw))

    assert abs((time.time() - since) - 3 * 3600) < 60


def test_an_unreadable_since_is_reported_not_swallowed(desk, capsys):
    from synapt.recall import cli

    project, _, _, _ = _three_sessions(desk)
    cli._print_transcript_gap(_resume_args(since="yesterday-ish"), project)

    assert "TRANSCRIPT GAP: not read (ValueError" in capsys.readouterr().out


def test_the_resume_command_prints_the_gap_after_an_indexed_render(desk, monkeypatch, tmp_path):
    """The indexed path, with the render mocked out the way test_cmd_resume_provenance.py
    does it and the transcripts real: the gap follows the resume body."""
    import io
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch

    from synapt.recall import cli

    project, _, _, _ = _three_sessions(desk)
    monkeypatch.chdir(project)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "recall.db").write_bytes(b"placeholder")
    view = SimpleNamespace(refresh_label=None)
    out = io.StringIO()
    with patch.object(cli, "_resolve_index_dir", return_value=index_dir), \
         patch.object(cli, "_refuse_if_index_disagrees_with_source", return_value=None), \
         patch("synapt.recall.journal._journal_path", return_value=tmp_path / "journal.jsonl"), \
         patch("synapt.recall.resume.load_resume_index", return_value=SimpleNamespace(_session_order=["s"], _db=None)), \
         patch("synapt.recall.resume.build_resume_view", return_value=view), \
         patch.object(cli, "_attach_freshness", return_value=view), \
         patch.object(cli, "_attach_unclean_end", return_value=view), \
         patch("synapt.recall.resume.attach_durable_checkpoint", return_value=view), \
         patch("synapt.recall.resume.format_resume", return_value="RESUME BODY"), \
         patch("synapt.recall.server._query_freshness_line", return_value=""), \
         patch.object(cli, "_deferral_notice_for_index", return_value=None), \
         patch.object(sys, "stdout", out):
        cli.cmd_resume(_resume_args(since="2026-10-11T19:00:00Z"))

    text = out.getvalue()
    assert text.index("RESUME BODY") < text.index("TRANSCRIPT GAP")
    assert "the answer is option b" in text


def test_live_session_flag_beats_the_runtime_id(desk, monkeypatch, capsys, tmp_path):
    """The step-3 hook names the live session itself; the flag wins over the env."""
    from synapt.recall import cli

    project, _, _, _ = _three_sessions(desk)
    cli._print_transcript_gap(_resume_args(since="2026-10-11T19:00:00Z", live_session="aaaaaaaa-crashed"), project)

    out = capsys.readouterr().out
    assert "please rerun the build" not in out
    assert "synapt resume" in out   # cccccccc-live is listed now: the flag replaced the env id
