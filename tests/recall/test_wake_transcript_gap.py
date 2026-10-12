"""The session-start wake carries the transcript gap, so a desk that wakes after
an unclean end reads what was typed to it before it acts, even when the index is
behind. It runs only when the hook names the session that is starting."""
from __future__ import annotations

import time
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from synapt.recall import cli
from tests.recall.test_resume_transcript_gap import _three_sessions, _user, _write, desk  # noqa: F401


@pytest.fixture(autouse=True)
def _owned_store(_isolate_recall_root_env, tmp_path, monkeypatch):
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(tmp_path / "store"))
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_WORKTREE", raising=False)


def _context(project, tmp_path, session_id, journal_ts=None):
    entries = []
    (tmp_path / "journal.jsonl").write_text("")   # the wake reads entries only when the file exists
    if journal_ts:
        from synapt.recall.journal import JournalEntry
        entries = [JournalEntry(timestamp=journal_ts, focus="handoff")]
    with ExitStack() as stack:
        for target, value in (
            ("synapt.recall.journal._get_branch", None),
            ("synapt.recall.journal._journal_path", tmp_path / "journal.jsonl"),
            ("synapt.recall.journal._read_all_entries", entries),
            ("synapt.recall.compaction.latest_agent_compaction_directive", None),
            ("synapt.recall.compaction.latest_compaction_summary", None),
            ("synapt.checkpoint.read_checkpoint", None),
            ("synapt.recall.knowledge.read_nodes", []),
            ("synapt.recall.reminders.pop_pending", []),
            ("synapt.recall.server.format_contradictions_for_session_start", ""),
            ("synapt.recall.channel.channel_unread", {}),
            ("synapt.recall.channel.check_directives", ""),
        ):
            stack.enter_context(patch(target, return_value=value))
        stack.enter_context(patch("synapt.recall.channel.channel_join"))
        return cli.generate_startup_context(project, current_session_id=session_id)


def _gap_blocks(lines):
    return [line for line in lines if line.startswith("TRANSCRIPT GAP")]


def test_the_wake_carries_the_gap_from_the_newest_journal(desk, tmp_path):
    project, _, _, _ = _three_sessions(desk)

    blocks = _gap_blocks(_context(project, tmp_path, "cccccccc-live", journal_ts="2026-10-11T19:00:00Z"))

    assert len(blocks) == 1
    assert "the answer is option b" in blocks[0] and "please rerun the build" in blocks[0]
    assert "synapt resume" not in blocks[0].split("\n", 1)[1]   # the live session's own message
    assert "(the newest authored journal)" in blocks[0]


def test_the_wake_carries_no_gap_when_the_hook_names_no_session(desk, tmp_path):
    project, _, _, _ = _three_sessions(desk)

    assert _gap_blocks(_context(project, tmp_path, None, journal_ts="2026-10-11T19:00:00Z")) == []


def test_the_wake_carries_no_gap_when_no_earlier_session_is_in_it(desk, tmp_path):
    project, crashed, respawn, _ = _three_sessions(desk)
    crashed.unlink()
    respawn.unlink()

    assert _gap_blocks(_context(project, tmp_path, "cccccccc-live", journal_ts="2026-10-11T19:00:00Z")) == []


def test_a_long_gap_shows_its_newest_messages_and_says_how_many_it_withheld(desk, tmp_path):
    project, home_dir, _ = desk
    now = time.time()
    _write(home_dir / "ffffffff-chatty.jsonl", [
        _user(f"2026-10-11T19:{m:02d}:00Z", "ffffffff-chatty", f"message number {m:02d}") for m in range(20)
    ], mtime=now - 100)

    block = _gap_blocks(_context(project, tmp_path, "cccccccc-live", journal_ts="2026-10-11T18:59:00Z"))[0]

    assert "14 earlier message(s) withheld" in block
    assert "synapt resume --since 2026-10-11T18:59:00Z" in block
    assert "message number 19" in block and "message number 14" in block
    assert "message number 13" not in block


def test_a_compaction_summary_is_not_read_as_a_message(desk, tmp_path):
    project, home_dir, _ = desk
    summary = _user("2026-10-11T19:20:00Z", "aaaaaaaa-compacted", "This session is being continued from a previous conversation")
    summary["isCompactSummary"] = True
    _write(home_dir / "aaaaaaaa-compacted.jsonl", [
        summary, _user("2026-10-11T19:21:00Z", "aaaaaaaa-compacted", "a real ask"),
    ], mtime=time.time() - 100)

    block = _gap_blocks(_context(project, tmp_path, "cccccccc-live", journal_ts="2026-10-11T19:00:00Z"))[0]

    assert "a real ask" in block
    assert "being continued" not in block
    assert "1 message(s), 1 hidden" in block


def test_an_id_with_no_transcript_on_disk_says_so(desk, tmp_path):
    project, _, _, live = _three_sessions(desk)
    live.unlink()

    block = _gap_blocks(_context(project, tmp_path, "cccccccc-live", journal_ts="2026-10-11T19:00:00Z"))[0]

    assert "the live session cccccccc has no transcript on disk yet, so none is excluded" in block


def test_the_reader_hides_a_compaction_summary(desk):
    """The library half of the compaction row: the reader, not only the wake."""
    from synapt.recall.resume import caller_transcripts, gather_transcript_gap, _timestamp_epoch

    project, home_dir, _ = desk
    summary = _user("2026-10-11T19:20:00Z", "aaaaaaaa-compacted", "This session is being continued")
    summary["isCompactSummary"] = True
    _write(home_dir / "aaaaaaaa-compacted.jsonl", [summary], mtime=time.time() - 100)

    gap = gather_transcript_gap(caller_transcripts(project), since=_timestamp_epoch("2026-10-11T19:00:00Z"),
                                since_source="t", live_session_id="cccccccc-live")

    assert gap.sessions[0].messages == () and gap.sessions[0].hidden == 1


def test_the_header_says_when_the_named_session_has_no_transcript(desk):
    from synapt.recall.resume import (
        caller_transcripts, format_transcript_gap, gather_transcript_gap, _timestamp_epoch,
    )

    project, _, _, live = _three_sessions(desk)
    live.unlink()
    gap = gather_transcript_gap(caller_transcripts(project), since=_timestamp_epoch("2026-10-11T19:00:00Z"),
                                since_source="t", live_session_id="cccccccc-live")

    assert "has no transcript on disk yet, so none is excluded" in format_transcript_gap(gap)
    assert "is not listed" not in format_transcript_gap(gap)


def test_the_rendered_wake_keeps_the_gap_ahead_of_the_journal_when_past_budget(tmp_path):
    """The rendered wake, not the context list: a gap block typed as 'other' was
    capped last and cut by the budget guard (measured on a real 24 KB wake)."""
    from synapt.recall.session_start import WAKE_BUDGET_BYTES, render_wake

    gap = ("TRANSCRIPT GAP since 10-11 19:00 (t): 1 message(s) in 1 session(s)\n"
           "  session aaaaaaaa: 1 message(s), 0 hidden\n    10-11 19:10 | the ask that must be read")
    journal = "Last session: handoff\n" + "\n".join(f"  - journal line {i} " + "x" * 80 for i in range(60))
    channel = "Recent #dev messages:\n" + "\n".join(f"  post {i} " + "y" * 90 for i in range(60))
    filler = ["Knowledge:\n" + "z" * 1400] + [f"other block {i}\n" + "w" * 900 for i in range(40)]

    text = render_wake([journal, channel, *filler, gap], source="startup", full_path=tmp_path / "wake.txt")

    assert len(text.encode("utf-8")) <= WAKE_BUDGET_BYTES
    assert "the ask that must be read" in text
    assert text.index("TRANSCRIPT GAP") < text.index("Last session: handoff")
    assert text.index("TRANSCRIPT GAP") < 2_000


def test_a_real_gap_survives_the_wake_render(desk, tmp_path):
    """Context to render, end to end: the newest messages of a long gap are in the wake."""
    from synapt.recall.session_start import render_wake

    project, home_dir, _ = desk
    _write(home_dir / "ffffffff-chatty.jsonl", [
        _user(f"2026-10-11T19:{m:02d}:00Z", "ffffffff-chatty", f"message number {m:02d} " + "q" * 280)
        for m in range(20)
    ], mtime=time.time() - 100)

    lines = _context(project, tmp_path, "cccccccc-live", journal_ts="2026-10-11T18:59:00Z")
    text = render_wake(lines, source="startup", full_path=tmp_path / "wake.txt")
    # The gap block alone: the same session is also the unclean end, whose recovered
    # tail quotes its last message, so a whole-text match cannot tell the two apart.
    gap_block = text[text.index("TRANSCRIPT GAP"):].split("\n\n", 1)[0]

    assert "message number 19" in gap_block
    assert "14 earlier message(s) withheld" in gap_block
    assert "line(s) withheld)" not in gap_block   # the wake's cap did not clip it


def test_a_clean_end_renders_no_gap(desk, tmp_path):
    """The session that wrote the journal goes on writing after it (tool output, the
    end of the turn): a session in the gap with nothing typed renders nothing."""
    project, home_dir, _ = desk
    _write(home_dir / "aaaaaaaa-clean.jsonl", [
        _user("2026-10-11T18:50:00Z", "aaaaaaaa-clean", "please write the journal"),
        {"type": "assistant", "sessionId": "aaaaaaaa-clean", "timestamp": "2026-10-11T19:05:00Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "journal written"}]}},
    ], mtime=time.time() - 100)

    assert _gap_blocks(_context(project, tmp_path, "cccccccc-live", journal_ts="2026-10-11T19:00:00Z")) == []


def test_a_session_that_only_wrote_files_still_renders(desk, tmp_path):
    from tests.recall.test_resume_transcript_gap import _write_tool

    project, home_dir, _ = desk
    _write(home_dir / "aaaaaaaa-writer.jsonl", [
        _write_tool("2026-10-11T19:05:00Z", "aaaaaaaa-writer", "/w/qa/answers.json"),
    ], mtime=time.time() - 100)

    block = _gap_blocks(_context(project, tmp_path, "cccccccc-live", journal_ts="2026-10-11T19:00:00Z"))[0]

    assert "/w/qa/answers.json" in block


def test_with_several_sessions_the_wake_keeps_the_newest_one(desk, tmp_path):
    """A crash, then respawns: the byte cap cuts from the end, so the newest session
    renders first and keeps its last message ahead of the journal."""
    from synapt.recall.session_start import render_wake

    project, home_dir, config_dir = desk
    now = time.time()
    for n, (sid, root) in enumerate((("aaaaaaaa-oldest", home_dir), ("bbbbbbbb-middle", config_dir),
                                     ("dddddddd-newest", home_dir))):
        _write(root / f"{sid}.jsonl", [
            _user(f"2026-10-11T19:{10 * n + m:02d}:00Z", sid, f"{sid} says {m} " + "r" * 280) for m in range(6)
        ], mtime=now - 300 + n * 100)

    lines = _context(project, tmp_path, "cccccccc-live", journal_ts="2026-10-11T19:00:00Z")
    text = render_wake([*lines, "Last session: handoff\n  - journal line"], source="startup",
                       full_path=tmp_path / "wake.txt")
    gap_block = text[text.index("TRANSCRIPT GAP"):].split("\n\n", 1)[0]

    assert "newest session first" in gap_block
    assert "dddddddd-newest says 5" in gap_block
    assert text.index("dddddddd-newest says 5") < text.index("Last session: handoff")
