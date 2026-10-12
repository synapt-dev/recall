"""The carry-forward report counts what THIS write retired, and says separately what the session had already retired.

`merge_carried_forward_with_report` unions the steps this session's earlier writes retired into the set it matches the
previous entry against (so a step retired minutes ago does not come back). It then counted every match as "retired by
done", so write 2 of a session reported write 1's retirements again: a prior session leaves A1 A2 A3 B1 B2, write 1 is
done A1-A3, write 2 is done B1, and write 2 printed "4 retired by done" for the one step it retired.
"""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from synapt.recall.cli import cmd_journal
from synapt.recall.journal import (
    CarryReport,
    JournalEntry,
    append_entry,
    format_write_confirmation,
    merge_carried_forward_with_report,
)

STEPS = ["A1", "A2", "A3", "B1", "B2"]


def _prior(steps=STEPS):
    return JournalEntry(timestamp="2026-09-24T10:00:00+00:00", session_id="PRIOR", focus="prior", next_steps=list(steps))


def _keys(steps):
    return [s.split(" [carried since")[0] for s in steps]


def test_write_two_reports_its_own_retirement_and_the_earlier_ones_apart():
    merged, report = merge_carried_forward_with_report([], ["B1"], _prior(), same_session_done=["A1", "A2", "A3"])
    assert report.retired_by_done == 1, report
    assert report.retired_earlier == 3, report
    assert _keys(merged) == ["B2"], merged
    assert report.carried == 1


def test_write_one_is_unchanged_nothing_is_earlier():
    merged, report = merge_carried_forward_with_report([], ["A1", "A2", "A3"], _prior())
    assert (report.retired_by_done, report.retired_earlier) == (3, 0), report
    assert _keys(merged) == ["B1", "B2"], merged


def test_a_step_named_again_in_this_writes_done_counts_once_and_as_this_writes():
    # A1 was retired by write 1 and is named under done again now: one retirement, attributed to this write
    _merged, report = merge_carried_forward_with_report([], ["A1", "B1"], _prior(), same_session_done=["A1", "A2"])
    assert report.retired_by_done == 2, report      # A1 and B1
    assert report.retired_earlier == 1, report      # A2 only
    assert report.retired_by_done + report.retired_earlier == 3   # A1, A2, B1: three distinct steps, none double-counted


def test_an_earlier_retirement_the_previous_entry_never_carried_is_not_counted():
    _merged, report = merge_carried_forward_with_report([], [], _prior(["B1"]), same_session_done=["A1", "ZZZ"])
    assert (report.retired_by_done, report.retired_earlier) == (0, 0), report


def test_the_confirmation_line_is_byte_identical_when_nothing_was_earlier():
    entry = JournalEntry(timestamp="2026-09-25T09:00:00+00:00", session_id="S", focus="f")
    text = format_write_confirmation(entry, report=CarryReport(carried=1, retired_by_done=2, withheld=0))
    assert "Carry-forward: 1 carried, 2 retired by done, 0 withheld." in text, text
    assert "earlier" not in text, text


def test_the_confirmation_line_names_the_earlier_retirements_when_there_are_some():
    entry = JournalEntry(timestamp="2026-09-25T09:00:00+00:00", session_id="S", focus="f")
    text = format_write_confirmation(entry, report=CarryReport(carried=1, retired_by_done=1, retired_earlier=3, withheld=0))
    assert "Carry-forward: 1 carried, 1 retired by done (3 already retired earlier this session), 0 withheld." in text, text


def _witness_through_the_mcp_handler():
    from synapt.recall.server import recall_journal

    path = Path(tempfile.mkdtemp()) / "journal.jsonl"
    append_entry(_prior(), path)

    def stub():
        return JournalEntry(timestamp="2026-09-25T09:00:00+00:00", session_id="SAME", focus="write")

    with patch("synapt.recall.journal._journal_path", return_value=path), \
         patch("synapt.recall.journal.auto_extract_entry", side_effect=lambda **kw: stub()), \
         patch("synapt.recall.journal.latest_transcript_path", return_value=None):
        first = recall_journal(action="write", focus="first", done="A1\nA2\nA3")
        second = recall_journal(action="write", focus="second", done="B1")
    return first, second


def test_the_mcp_handler_write_two_prints_one_retired_not_four():
    first, second = _witness_through_the_mcp_handler()
    assert "Carry-forward: 2 carried, 3 retired by done, 0 withheld." in first, first
    assert "1 retired by done (3 already retired earlier this session)" in second, second
    assert "4 retired by done" not in second, second
    carried = second.split("### Carried Forward Next Steps")[-1]
    assert "B2" in carried and not any(f"- A{n}" in carried for n in (1, 2, 3)), carried


def test_the_cli_write_two_prints_one_retired_not_four(tmp_path, capsys):
    fixture = tmp_path / "journal.jsonl"
    append_entry(_prior(), fixture)

    def run(done):
        capsys.readouterr()
        args = argparse.Namespace(read=False, list=False, show=None, write=True, focus="w", done=done, decisions=None,
                                  next=None, repair=False, dry_run=False, all_stores=False, path=str(fixture))
        with patch("synapt.recall.cli.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.journal.latest_transcript_path", return_value=None), \
             patch("synapt.recall.journal.auto_extract_entry", side_effect=lambda **_: JournalEntry(
                 timestamp="2026-09-25T09:00:00+00:00", session_id="SAME")):
            cmd_journal(args)
        return capsys.readouterr().out

    first = run("A1\nA2\nA3")
    second = run("B1")
    assert "3 retired by done," in first and "earlier" not in first, first
    assert "1 retired by done (3 already retired earlier this session)" in second, second
    assert "4 retired by done" not in second, second


def test_exactly_one_earlier_retirement_is_still_named():
    # the parenthetical shows from ONE earlier retirement, not from two: a threshold is the edge a three-step case
    # never reaches
    _merged, report = merge_carried_forward_with_report([], ["B1"], _prior(), same_session_done=["A1"])
    assert (report.retired_by_done, report.retired_earlier) == (1, 1), report
    entry = JournalEntry(timestamp="2026-09-25T09:00:00+00:00", session_id="S", focus="f")
    text = format_write_confirmation(entry, report=report)
    assert "1 retired by done (1 already retired earlier this session)" in text, text
