"""`synapt orient`: the wake card from the user's own record.

Every row runs against a HOME and a project this test owns, so the ambient Claude Code
roots of the machine running it are never read.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from synapt.recall.core import project_slug
from synapt.recall.journal import JournalEntry, _journal_path, append_entry
from synapt.recall.orient import CARRIED_MAX, CONSTRAINTS_MAX, TOKEN_BUDGET, build_card, format_card, format_json


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """A stranger's HOME: only ~/.claude, no configured Claude root, no Codex sessions."""
    h = tmp_path / "home"
    (h / ".claude" / "projects").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(h / ".codex-none"))
    return h


@pytest.fixture
def project(tmp_path, home, monkeypatch) -> Path:
    p = tmp_path / "proj"
    p.mkdir()
    monkeypatch.chdir(p)
    return p


def _entry(project: Path, ts: str, **fields) -> None:
    append_entry(JournalEntry(timestamp=ts, **fields), _journal_path(project))


def _now_iso(offset_s: float = 0) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() + offset_s))


def _memory(root: Path, project: Path, files: dict[str, str]) -> Path:
    mem = root / project_slug(project) / "memory"
    mem.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (mem / name).write_text(text, encoding="utf-8")
    return mem


def _feedback(name: str, desc: str) -> str:
    return f"---\nname: {name}\ndescription: {desc}\nmetadata:\n  type: feedback\n---\n\nbody\n"


def test_goal_next_and_ruled_out_come_from_the_newest_meaningful_entry(project):
    _entry(project, _now_iso(-7200), focus="old goal", next_steps=["old next"], decisions=["old decision"])
    _entry(project, _now_iso(-60), focus="ship the login page", next_steps=["run pytest tests/test_login.py", "then the docs"],
           decisions=["no OAuth in v1"])

    card = build_card(project)

    assert card.goal == "ship the login page"
    assert card.next == "run pytest tests/test_login.py"
    assert card.ruled_out == ["no OAuth in v1"]
    assert card.goal_age_s is not None and 0 <= card.goal_age_s < 600


def test_carried_lists_open_steps_from_earlier_entries_and_counts_what_it_withholds(project):
    older = [f"carried step {i}" for i in range(CARRIED_MAX + 3)]
    _entry(project, _now_iso(-7200), focus="earlier", next_steps=older)
    _entry(project, _now_iso(-60), focus="now", next_steps=["the next command"], done=["carried step 0"])

    card = build_card(project)

    assert "carried step 0" not in card.carried, "a step done later is not carried"
    assert "the next command" not in card.carried, "NEXT is not repeated under CARRIED"
    assert len(card.carried) == CARRIED_MAX
    assert card.withheld.get("carried") == len(older) - 1 - CARRIED_MAX
    assert f"({card.withheld['carried']} more withheld" in format_card(card)


def test_a_stranger_with_only_dot_claude_gets_constraints_from_the_feedback_memories(project, home):
    _memory(home / ".claude" / "projects", project, {
        "MEMORY.md": "- [x](x.md) — index, never a constraint\n",
        "feedback_tests_first.md": _feedback("tests-first", "write the failing test before the code"),
        "user_role.md": "---\nname: role\ndescription: a user memory\nmetadata:\n  type: user\n---\n",
        "feedback_quoted.md": _feedback("quoted", 'say \\"no\\" plainly'),
    })

    card = build_card(project)

    assert "tests-first: write the failing test before the code" in card.constraints
    assert 'quoted: say "no" plainly' in card.constraints
    assert not any(c.startswith("role") for c in card.constraints), "only feedback memories are constraints"
    assert len(card.constraints) == 2


def test_a_configured_claude_root_is_read_before_dot_claude(project, home, tmp_path, monkeypatch):
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg))
    _memory(cfg / "projects", project, {"feedback_a.md": _feedback("configured", "from CLAUDE_CONFIG_DIR")})
    _memory(home / ".claude" / "projects", project, {"feedback_b.md": _feedback("default", "from ~/.claude")})

    card = build_card(project)

    assert card.constraints == ["configured: from CLAUDE_CONFIG_DIR"]


def test_constraints_come_from_the_session_project_never_from_the_gripspace_root(project, home, tmp_path, monkeypatch):
    """A shared workspace root's memory dir belongs to another project; its rules are not this session's."""
    shared = tmp_path / "shared-root"
    shared.mkdir()
    monkeypatch.setenv("GRIPSPACE_ROOT", str(shared))
    roots = home / ".claude" / "projects"
    _memory(roots, project, {"feedback_mine.md": _feedback("mine", "this project's rule")})
    _memory(roots, shared, {"feedback_theirs.md": _feedback("theirs", "another project's rule")})

    card = build_card(None)

    assert card.constraints == ["mine: this project's rule"]


def test_with_no_project_the_journal_resolves_as_synapt_recall_journal_does(project, owned_recall_root):
    """No --project must not make cwd a deliberate root: that read a different store than the one written."""
    append_entry(JournalEntry(timestamp=_now_iso(-60), focus="written through the default resolution"), _journal_path(None))

    card = build_card(None)

    assert card.goal == "written through the default resolution"


def test_an_absent_memory_dir_and_an_absent_journal_are_named_not_raised(project):
    card = build_card(project)

    assert card.filled() == 0
    assert any(m.startswith("journal: none at ") and "synapt recall journal --write" in m for m in card.missing)
    assert any(m.startswith("constraints: no memory directory at ") for m in card.missing)
    assert "NOT READ:" in format_card(card)


def test_the_card_stays_under_the_token_budget_and_says_what_it_left_out(project, home):
    _entry(project, _now_iso(-60), focus="g " * 100, next_steps=["n"], decisions=[("d%d " % i) * 60 for i in range(60)])
    _memory(home / ".claude" / "projects", project,
            {f"feedback_{i}.md": _feedback(f"rule-{i}", "x" * 230) for i in range(CONSTRAINTS_MAX + 5)})
    card = build_card(project)

    text = format_card(card, budget=600)

    assert len(text) // 4 <= 600
    assert card.withheld.get("ruled_out", 0) > 0, "the longest list lost items to the budget"
    assert card.withheld.get("constraints", 0) >= 5, "the 5 over CONSTRAINTS_MAX are counted, never dropped silently"
    assert "more withheld" in text
    assert len(format_card(build_card(project))) // 4 <= TOKEN_BUDGET


def _transcript(home: Path, project: Path, name: str, age_s: float) -> Path:
    tdir = home / ".claude" / "projects" / project_slug(project)
    tdir.mkdir(parents=True, exist_ok=True)
    t = tdir / name
    t.write_text('{"type": "user"}\n', encoding="utf-8")
    os.utime(t, (time.time() - age_s, time.time() - age_s))
    return t


def test_stale_names_a_finished_session_newer_than_the_journal(project, home):
    """W2: a session that ran after the journal and is now over is the gap STALE exists for."""
    _entry(project, _now_iso(-3 * 3600), focus="three hours old")
    _transcript(home, project, "finished.jsonl", 600)
    _transcript(home, project, "live.jsonl", 0)

    card = build_card(project)

    assert card.stale.startswith("the journal is ") and "older than your newest finished session" in card.stale
    assert format_card(card).splitlines()[2].startswith("STALE: ")


def test_a_wake_after_a_clean_close_is_not_stale(project, home):
    """W1: at wake the newest transcript is this session's own; it is not evidence of an unwritten journal."""
    _entry(project, _now_iso(-2 * 3600), focus="closed cleanly")
    _transcript(home, project, "closed.jsonl", 2 * 3600 + 30)
    _transcript(home, project, "live.jsonl", 0)

    assert build_card(project).stale == ""


def test_no_stale_line_when_the_journal_is_as_new_as_the_session(project, home):
    _entry(project, _now_iso(-60), focus="fresh")
    _transcript(home, project, "s1.jsonl", 180)

    assert build_card(project).stale == ""


def test_the_front_door_verb_prints_json(project, capsys, monkeypatch):
    _entry(project, _now_iso(-60), focus="front door", next_steps=["go"])
    from synapt.cli import main

    monkeypatch.setattr("sys.argv", ["synapt", "orient", "--project", str(project), "--json"])
    main()

    out = json.loads(capsys.readouterr().out)
    assert out["goal"] == "front door" and out["next"] == "go" and out["fields_filled"] == 2
    assert 0 < out["tokens_est"] <= TOKEN_BUDGET
