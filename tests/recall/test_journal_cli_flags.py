"""`synapt recall journal`: a flag that is accepted must be honoured, or refused before anything is written.

Before this change `--write --path X` ignored --path (its help said "With --repair") and appended to the LIVE
journal, and `--write --dry-run` wrote too. A caller who passed either believed the entry went elsewhere or nowhere.
Now --path names the journal file in every mode, and the flags that only mean something under --repair, or that
contradict --write, exit 2 with nothing written.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from synapt.recall.cli import cmd_journal
from synapt.recall.core import project_worktree_dir
from synapt.recall.journal import JournalEntry, append_entry, read_entries


def _args(**over):
    base = dict(
        read=False, list=False, show=None, write=False, focus=None, done=None, decisions=None, next=None,
        repair=False, dry_run=False, all_stores=False, path=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture
def project(tmp_path):
    p = tmp_path / "proj"
    p.mkdir()
    live = project_worktree_dir(p) / "journal.jsonl"
    # the control that makes every row below mean something: the live journal this project resolves to is OURS
    assert str(live).startswith(str(tmp_path.resolve())) or str(live).startswith(str(tmp_path)), live
    return p


def _live(project) -> Path:
    return project_worktree_dir(project) / "journal.jsonl"


def _seed(path: Path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    append_entry(JournalEntry(timestamp=kw.pop("timestamp", "2026-03-31T10:00:00+00:00"), **kw), path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "ABSENT"


def _run(project, capsys, session="current-session", extracted=None, **flags):
    """Run cmd_journal as the CLI would; return (exit code, stdout, stderr).

    *extracted* are extra fields of the entry the auto-extractor hands back (e.g. an auto stub with files only)."""
    capsys.readouterr()
    code = 0
    with patch("synapt.recall.cli.Path.cwd", return_value=project), \
         patch("synapt.recall.journal.latest_transcript_path", return_value=None), \
         patch("synapt.recall.journal.auto_extract_entry", side_effect=lambda **_: JournalEntry(
             timestamp="2026-04-01T12:00:00+00:00", session_id=session, **(extracted or {}))):
        try:
            cmd_journal(_args(**flags))
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
    out = capsys.readouterr()
    return code, out.out, out.err


def test_write_with_path_leaves_the_live_journal_byte_identical(project, tmp_path, capsys):
    _seed(_live(project), session_id="live-prior", focus="LIVE prior", done=["live thing"])
    before = _sha(_live(project))
    fixture = tmp_path / "fixtures" / "journal.jsonl"

    code, _out, err = _run(project, capsys, write=True, path=str(fixture), focus="fixture work", done="fixture item")

    assert code == 0, err
    assert _sha(_live(project)) == before, "the live journal changed under --path"
    entries = read_entries(fixture, n=5)
    assert [e.focus for e in entries] == ["fixture work"], entries
    assert str(fixture) in err, "the confirmation must name the file it wrote"


def test_write_with_path_carries_forward_from_the_named_file_not_the_live_one(project, tmp_path, capsys):
    _seed(_live(project), session_id="live-prior", focus="LIVE", next_steps=["LIVE-ONLY step"])
    fixture = tmp_path / "fixture.jsonl"
    _seed(fixture, session_id="fixture-prior", focus="FIX", next_steps=["FIXTURE step"])

    code, _out, err = _run(project, capsys, write=True, path=str(fixture), focus="now", done="x")

    assert code == 0, err
    newest = read_entries(fixture, n=1)[0]
    assert newest.focus == "now", "the new entry must be in the named file before its steps mean anything"
    steps = newest.next_steps
    assert any(s.startswith("FIXTURE step") for s in steps), steps
    assert not any("LIVE-ONLY" in s for s in steps), steps


def test_write_with_path_retires_a_step_using_this_sessions_earlier_entries_in_the_named_file(project, tmp_path, capsys):
    fixture = tmp_path / "fixture.jsonl"
    _seed(fixture, session_id="fixture-prior", focus="prior", next_steps=["ship docs"])

    first, _o, err1 = _run(project, capsys, write=True, path=str(fixture), focus="first", done="ship docs")
    second, _o, err2 = _run(project, capsys, write=True, path=str(fixture), focus="second", done="other work")

    assert first == 0 and second == 0, (err1, err2)
    # the RAW last line: read_entries collapses a session to one entry, which would hide the second write
    latest = json.loads(fixture.read_text().splitlines()[-1])
    assert latest["focus"] == "second"
    assert not any("ship docs" in s for s in latest["next_steps"]), latest["next_steps"]


def test_the_auto_stub_branch_of_write_also_goes_to_the_named_file(project, tmp_path, capsys):
    """An auto stub (files only, no fields) is saved by a second append call; it must honour --path too."""
    _seed(_live(project), session_id="live-prior", focus="LIVE", done=["x"])
    before = _sha(_live(project))
    fixture = tmp_path / "stub.jsonl"

    code, _out, err = _run(project, capsys, extracted=dict(auto=True, files_modified=["a.py"]),
                           write=True, path=str(fixture))

    assert code == 0 and "Auto-stub saved" in err, (code, err)
    assert _sha(_live(project)) == before, "the stub went to the live journal"
    assert [e.files_modified for e in read_entries(fixture, n=5)] == [["a.py"]]


def test_read_list_and_show_honour_path(project, tmp_path, capsys):
    _seed(_live(project), session_id="live-s", focus="LIVE focus", done=["live thing"], next_steps=["LIVE step"])
    fixture = tmp_path / "fixture.jsonl"
    _seed(fixture, session_id="fix-s", focus="FIXTURE focus", done=["fixture thing"], next_steps=["FIXTURE step"])

    _c, listed, _e = _run(project, capsys, list=True, path=str(fixture))
    assert "FIXTURE focus" in listed and "LIVE focus" not in listed, listed

    _c, shown, _e = _run(project, capsys, show=1, path=str(fixture))
    assert "FIXTURE focus" in shown and "LIVE focus" not in shown, shown

    _c, read, _e = _run(project, capsys, read=True, path=str(fixture))
    assert "FIXTURE step" in read and "LIVE step" not in read, read


@pytest.mark.parametrize("flag", ["dry_run", "all_stores"])
def test_write_with_a_repair_only_flag_is_refused_and_writes_nothing(project, tmp_path, capsys, flag):
    _seed(_live(project), session_id="live-prior", focus="LIVE", done=["x"])
    before = _sha(_live(project))

    code, _out, err = _run(project, capsys, write=True, focus="f", done="d", **{flag: True})

    assert code == 2, (code, err)
    assert flag.replace("_", "-") in err and "--repair" in err, err
    assert _sha(_live(project)) == before, "a refused command wrote"


@pytest.mark.parametrize("flag", ["dry_run", "all_stores"])
@pytest.mark.parametrize("mode", [dict(list=True), dict(read=True), dict(show=1)])
def test_a_repair_only_flag_on_a_read_mode_is_refused(project, capsys, flag, mode):
    code, _out, err = _run(project, capsys, **mode, **{flag: True})
    assert code == 2, (code, err)
    assert flag.replace("_", "-") in err, err


@pytest.mark.parametrize("other", [dict(read=True), dict(list=True), dict(show=2), dict(repair=True)])
def test_write_combined_with_another_mode_is_refused_and_writes_nothing(project, tmp_path, capsys, other):
    _seed(_live(project), session_id="live-prior", focus="LIVE", done=["x"])
    before = _sha(_live(project))

    code, _out, err = _run(project, capsys, write=True, focus="f", done="d", **other)

    assert code == 2, (code, err)
    assert "--write" in err, err
    assert _sha(_live(project)) == before, "a refused command wrote"


def test_write_with_a_directory_path_is_refused(project, tmp_path, capsys):
    before = _sha(_live(project))
    code, _out, err = _run(project, capsys, write=True, path=str(tmp_path), focus="f", done="d")
    assert code == 2, (code, err)
    assert "directory" in err, err
    assert _sha(_live(project)) == before


def test_controls_write_without_path_still_appends_to_the_live_journal(project, capsys):
    _seed(_live(project), session_id="live-prior", focus="LIVE", done=["x"])
    code, _out, err = _run(project, capsys, write=True, focus="again", done="y")
    assert code == 0, err
    assert [e.focus for e in read_entries(_live(project), n=5)][0] == "again"


def test_controls_repair_dry_run_stays_valid_with_path(project, tmp_path, capsys):
    fixture = tmp_path / "fixture.jsonl"
    _seed(fixture, session_id="a", focus="A", done=["x"])
    code, _out, err = _run(project, capsys, repair=True, dry_run=True, path=str(fixture))
    assert code == 0, err


def test_controls_repair_all_stores_takes_a_data_root_directory_as_its_path(project, tmp_path, capsys):
    """Under --repair --all-stores, --path is a DATA ROOT (a directory); the directory refusal is for a journal file."""
    root = tmp_path / "dataroot"
    _seed(root / "worktrees" / "wt-a" / "journal.jsonl", session_id="a", focus="A", done=["x"])
    code, out, err = _run(project, capsys, repair=True, all_stores=True, dry_run=True, path=str(root))
    assert code == 0, (code, err)
    assert "wt-a" in out, out


def test_controls_list_with_a_show_count_stays_valid_without_path(project, capsys):
    _seed(_live(project), session_id="a", focus="ALPHA", done=["x"])
    _seed(_live(project), session_id="b", focus="BRAVO", done=["y"], timestamp="2026-04-01T10:00:00+00:00")
    code, out, err = _run(project, capsys, list=True, show=2)
    assert code == 0 and "ALPHA" in out and "BRAVO" in out, (code, out, err)


def test_the_real_cli_parser_wires_the_witness_end_to_end(tmp_path):
    """The end-to-end witness through argparse and a subprocess: `--write --path <tmp>` leaves the live journal alone."""
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    subprocess.run(["git", "init", "-q", str(proj)], check=True)
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": os.pathsep.join(sys.path)}
    cli = [sys.executable, "-m", "synapt.recall.cli", "journal"]

    def run(*argv):
        return subprocess.run(cli + list(argv), cwd=proj, env=env, capture_output=True, text=True, timeout=120)

    first = run("--write", "--focus", "baseline", "--done", "base item")
    assert first.returncode == 0, first.stderr
    live = next(Path(proj / ".synapt").rglob("journal.jsonl"))
    assert str(live.resolve()).startswith(str(tmp_path.resolve())), live
    before = _sha(live)

    fixture = tmp_path / "out" / "fixture.jsonl"
    r = run("--write", "--path", str(fixture), "--focus", "fixture", "--done", "fixture item")
    assert r.returncode == 0, r.stderr
    assert _sha(live) == before
    assert json.loads(fixture.read_text().splitlines()[0])["focus"] == "fixture"

    dry = run("--write", "--dry-run", "--focus", "dry", "--done", "dry item")
    assert dry.returncode == 2, (dry.returncode, dry.stderr)
    assert _sha(live) == before


def test_a_tilde_path_is_expanded_to_the_home_directory(project, tmp_path, capsys, monkeypatch):
    # `--path=~/j.jsonl` reaches the program with the tilde unexpanded (the shell only expands a leading one), so the
    # file must land under HOME and never in a directory literally named "~" beside the working directory
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    work = tmp_path / "elsewhere"
    work.mkdir()
    monkeypatch.chdir(work)

    code, _out, _err = _run(project, capsys, write=True, focus="tilde probe", path="~/j.jsonl")

    assert code == 0
    assert [e.focus for e in read_entries(path=home / "j.jsonl")] == ["tilde probe"]
    assert not (work / "~").exists()
