"""A memory-floor refusal of an automatic build leaves a trace, and `resume` and the wake show it.

Every catchup and precompact refusal used to be printed to a stream its own spawn discarded, so days
of "no build" read as "no build was tried". The witnesses drive the REAL ``cmd_catchup``, the REAL
precompact branch of ``cmd_hook``, the REAL ``cmd_resume`` and the REAL session-start hook, with only
their heavy edges stubbed, so that a refused build is proved to REACH what a reader sees. The floor is
not touched (6.0 GB); only the trace is new.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from synapt.recall import build_deferrals, cli
from synapt.recall.build_deferrals import DEFERRALS_FILENAME, deferral_notice, record_build, record_deferral

REFUSE = "0:100:2.0:0"   # free_inactive 2.0 GB, under the 6.0 floor
PASS = "0:100:9.0:0"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """One data dir and one index dir, wired into every cli entry point that reads them."""
    data = tmp_path / "root" / ".synapt" / "recall"
    index = data / "index"      # the canonical layout: <root>/.synapt/recall/index, which is what _index_gripspace_root accepts
    index.mkdir(parents=True)
    # a monolithic store (recall.db only, no CURRENT), the shape every store starts in
    (index / "recall.db").write_bytes(b"placeholder")
    old = time.time() - 3600
    os.utime(index / "recall.db", (old, old))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "project_data_dir", lambda project=None: data)
    monkeypatch.setattr(cli, "project_index_dir", lambda project=None: index)
    monkeypatch.setattr(cli, "project_transcript_dirs", lambda project: [tmp_path])
    monkeypatch.setattr(cli, "_catchup_archive_and_journal", lambda *a, **k: None)
    monkeypatch.setattr("synapt.recall.journal.compact_journal", lambda *a, **k: 0, raising=False)
    monkeypatch.setattr(cli, "_host_synapt_dir", lambda: tmp_path / "hostsynapt")
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.subprocess, "run", lambda argv, *a, **k: calls.append(list(argv)) or subprocess.CompletedProcess(argv, 0, b"", b""))
    import synapt.recall.query_freshness as qf
    monkeypatch.setattr(qf, "catchup_oversize_transcripts", lambda *a, **k: [])
    return SimpleNamespace(data=data, index=index, calls=calls, tmp=tmp_path)


def _catchup(no_build=False):
    cli.cmd_catchup(argparse.Namespace(no_build=no_build))


def _rows(store):
    p = store.data / DEFERRALS_FILENAME
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def _resume(store, monkeypatch, *, real_resolver=False, index_arg=None):
    index_dir = store.index
    fake_index = SimpleNamespace(_session_order=["s1"], _db=None)
    fake_view = SimpleNamespace(refresh_label=None)
    out = io.StringIO()
    resolver = patch.object(cli, "_resolve_index_dir", wraps=cli._resolve_index_dir) if real_resolver else \
        patch.object(cli, "_resolve_index_dir", return_value=index_dir)
    with resolver, \
         patch("synapt.recall.journal._journal_path", return_value=store.tmp / "journal.jsonl"), \
         patch("synapt.recall.resume.caller_transcripts", return_value=["fake-caller"]), \
         patch("synapt.recall.resume.load_resume_index", return_value=fake_index), \
         patch("synapt.recall.resume.build_resume_view", return_value=fake_view), \
         patch.object(cli, "_attach_freshness", return_value=fake_view), \
         patch.object(cli, "_attach_unclean_end", return_value=fake_view), \
         patch("synapt.recall.resume.attach_durable_checkpoint", return_value=fake_view), \
         patch("synapt.recall.resume.format_resume", return_value="RESUME BODY"), \
         patch("synapt.recall.server._query_freshness_line", return_value=""), \
         patch("synapt.recall.server._resolved_provenance_line", return_value="synapt vTEST"), \
         patch.object(sys, "stdout", out):
        cli.cmd_resume(argparse.Namespace(session=None, turns=10, index=index_arg, project=None))
    return out.getvalue()


def test_a_refused_catchup_records_one_row_and_starts_no_build(store, monkeypatch):
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", REFUSE)
    _catchup()
    rows = _rows(store)
    assert len(rows) == 1 and rows[0]["site"] == "catchup" and rows[0]["verdict"] == "refuse"
    assert "free_inactive_gb=2.00" in rows[0]["numbers"] and "floor_gb=6.0" in rows[0]["numbers"]
    assert not [c for c in store.calls if "build" in c], "the floor still refuses; the trace must not change that"


def test_control_a_passing_catchup_records_nothing(store, monkeypatch):
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", PASS)
    _catchup()
    assert _rows(store) == []
    assert [c for c in store.calls if "build" in c], "control: the pass path still builds, so the empty file is the gate's doing"


def test_the_no_build_policy_deferral_is_not_a_memory_refusal(store, monkeypatch):
    """Session start runs catchup --no-build by policy; that is not a memory-floor refusal and must not be counted as one."""
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", REFUSE)
    _catchup(no_build=True)
    assert _rows(store) == []


def test_a_refused_catchup_reaches_resume_output(store, monkeypatch):
    """The story's witness: refuse -> resume prints the sentence, with the plural right."""
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", REFUSE)
    _catchup()
    out = _resume(store, monkeypatch)
    assert "RESUME BODY" in out
    assert "automatic build deferred 1 time (memory floor); run synapt build" in out
    assert "index last built (date not recorded); automatic build deferred 1 time" in out
    _catchup()
    assert "automatic build deferred 2 times (memory floor); run synapt build" in _resume(store, monkeypatch)


def test_control_resume_with_no_refusal_prints_no_notice(store, monkeypatch):
    assert "memory floor" not in _resume(store, monkeypatch)


def test_a_build_after_the_refusals_resets_the_count(store, monkeypatch):
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", REFUSE)
    _catchup()
    (store.index / "CURRENT").write_text("gen-1\n")
    later = time.time() + 5
    os.utime(store.index / "CURRENT", (later, later))
    assert "memory floor" not in _resume(store, monkeypatch)
    _catchup()  # a new refusal is newer than the index again
    # the new row is stamped now, before the CURRENT mtime we pushed 5 s ahead, so back-date CURRENT to prove the count restarts
    earlier = time.time() - 60
    os.utime(store.index / "CURRENT", (earlier, earlier))
    assert "automatic build deferred 2 times" in _resume(store, monkeypatch)


def test_the_last_built_date_comes_from_the_index(store):
    (store.index / "CURRENT").write_text("gen-1\n")
    stamp = time.mktime((2026, 9, 26, 9, 6, 0, 0, 0, -1))
    os.utime(store.index / "CURRENT", (stamp, stamp))
    record_deferral(store.data, "catchup", "free_inactive_gb=2.00 floor_gb=6.0")
    assert deferral_notice(store.data, store.index) == (
        "index last built 2026-09-26; automatic build deferred 1 time (memory floor); run synapt build")


def test_a_refused_precompact_records_a_row_and_keeps_the_journal_write(store, monkeypatch):
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", REFUSE)
    journal = []
    monkeypatch.setattr(cli, "_refuse_if_index_disagrees_with_source", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_configure_codex_cwd_cache", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_precompact_journal_write", lambda project: journal.append(project))
    monkeypatch.setattr("synapt.recall.channel.channel_heartbeat", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_archive_and_build", lambda *a, **k: pytest.fail("the floor refused; no rebuild may start"))
    cli.cmd_hook(argparse.Namespace(event="precompact"))
    rows = _rows(store)
    assert [r["site"] for r in rows] == ["precompact"]
    assert journal, "the crash-recovery journal write stays outside the gate"


def test_a_refused_catchup_reaches_the_session_start_wake(store, monkeypatch, tmp_path):
    from test_hook_session_start_bounded import _run_hook
    record_deferral(store.data, "catchup", "free_inactive_gb=2.00 floor_gb=6.0")
    out, _ = _run_hook(monkeypatch, tmp_path)
    assert "automatic build deferred 1 time (memory floor); run synapt build" in out
    (store.data / DEFERRALS_FILENAME).unlink()
    out2, _ = _run_hook(monkeypatch, tmp_path)
    assert "memory floor" not in out2, "control: no refusal on record, no banner"


def test_a_failing_trace_never_breaks_the_hook(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    record_deferral(blocker / "sub", "catchup", "n")   # data_dir under a file: mkdir fails; must not raise
    assert deferral_notice(blocker / "sub", tmp_path) is None


def test_the_file_is_capped(store):
    for _ in range(build_deferrals._KEEP_LINES + 25):
        record_deferral(store.data, "catchup", "n")
    assert len(_rows(store)) == build_deferrals._KEEP_LINES


def test_a_monolithic_store_keeps_the_notice_after_a_knowledge_write(store, monkeypatch):
    """r2 finding: recall.db moves on any write, so an mtime rule dropped the notice on the first save."""
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", REFUSE)
    _catchup()
    assert "memory floor" in _resume(store, monkeypatch)
    later = time.time() + 5
    (store.index / "recall.db").write_bytes(b"a knowledge save touched this file")
    os.utime(store.index / "recall.db", (later, later))
    assert not (store.index / "CURRENT").exists(), "control: this store never publishes CURRENT"
    assert "automatic build deferred 1 time (memory floor)" in _resume(store, monkeypatch)


def test_a_successful_build_ends_the_run_and_the_next_refusal_starts_a_new_one(store, monkeypatch):
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", REFUSE)
    _catchup()
    _catchup()
    assert "deferred 2 times" in _resume(store, monkeypatch)
    time.sleep(0.02)
    record_build(store.data)
    assert "memory floor" not in _resume(store, monkeypatch)
    time.sleep(0.02)
    _catchup()
    out = _resume(store, monkeypatch)
    assert "automatic build deferred 1 time (memory floor)" in out
    assert "index last built 20" in out, "the date now comes from the built row"


def test_the_real_build_funnel_appends_a_built_row_only_on_success(store, monkeypatch):
    monkeypatch.setattr(cli, "_acquire_build_lock", lambda *a, **k: 1)
    monkeypatch.setattr(cli, "_release_build_lock", lambda fd: None)
    monkeypatch.setattr(cli, "_archive_and_build_locked", lambda *a, **k: object())
    assert cli._archive_and_build(store.tmp, use_embeddings=False, incremental=True) is not None
    assert [r.get("event") for r in _rows(store)] == ["built"]
    monkeypatch.setattr(cli, "_archive_and_build_locked", lambda *a, **k: None)  # no chunks found: nothing was built
    assert cli._archive_and_build(store.tmp, use_embeddings=False, incremental=True) is None
    assert [r.get("event") for r in _rows(store)] == ["built"], "control: a build that produced nothing appends nothing"


def test_the_trim_keeps_the_newest_built_row(store):
    record_build(store.data)
    for _ in range(build_deferrals._KEEP_LINES + 25):
        record_deferral(store.data, "catchup", "n")
    rows = _rows(store)
    assert len(rows) == build_deferrals._KEEP_LINES
    assert sum(1 for r in rows if r.get("event") == "built") == 1
    assert "deferred" in (deferral_notice(store.data, store.index) or "")


def test_resume_reads_the_trace_only_for_a_canonical_index(store, monkeypatch):
    """--index is first-class. A canonical one maps back to its store; any other shape cannot be mapped safely, so
    it gets no notice, and a file at its parent that this store never wrote is not reported as this store's history."""
    record_deferral(store.data, "catchup", "free_inactive_gb=2.00 floor_gb=6.0")
    # control: the REAL resolver with --index at the canonical index prints the notice, so the rows below can fail
    out = _resume(store, monkeypatch, real_resolver=True, index_arg=str(store.index))
    assert "automatic build deferred 1 time (memory floor)" in out
    elsewhere = store.tmp / "other" / "idx"
    elsewhere.mkdir(parents=True)
    (elsewhere / "recall.db").write_bytes(b"placeholder")
    (elsewhere.parent / DEFERRALS_FILENAME).write_text(
        json.dumps({"ts": 1.0, "site": "catchup", "verdict": "refuse", "numbers": "n"}) + "\n")
    (store.index / "recall.db").write_bytes(b"placeholder")
    out = _resume(store, monkeypatch, real_resolver=True, index_arg=str(elsewhere))
    assert "RESUME BODY" in out and "memory floor" not in out, "a foreign file beside a custom --index must not be shown as this store's history"


def _linked_store(store):
    """Root A whose own canonical index position is a symlink to root B's index (a shared index). The writer
    (catchup, precompact) writes beside A's spelled path: A/.synapt/recall."""
    a_recall = store.tmp / "A" / ".synapt" / "recall"
    b_index = store.tmp / "B" / ".synapt" / "recall" / "index"
    a_recall.mkdir(parents=True)
    b_index.mkdir(parents=True)
    (b_index / "recall.db").write_bytes(b"placeholder")
    (a_recall / "index").symlink_to(b_index)
    return a_recall, a_recall / "index"


def test_a_store_whose_own_index_is_a_symlink_keeps_its_notice_at_the_wake(store, monkeypatch, tmp_path):
    """The source of truth is where the writer writes. A's writer wrote beside the spelled path, not beside the
    link's target, so the wake reads the spelled parent with no resolve."""
    from test_hook_session_start_bounded import _run_hook
    record_deferral(store.data, "catchup", "free_inactive_gb=2.00 floor_gb=6.0")
    out, _ = _run_hook(monkeypatch, tmp_path)
    assert "automatic build deferred 1 time (memory floor)" in out, "control: the plain store shows the notice"
    a_recall, a_index = _linked_store(store)
    record_deferral(a_recall, "catchup", "free_inactive_gb=2.00 floor_gb=6.0")
    monkeypatch.setattr(cli, "project_index_dir", lambda project=None: a_index)
    monkeypatch.setattr(cli, "project_data_dir", lambda project=None: a_recall)
    out2, _ = _run_hook(monkeypatch, tmp_path)
    assert "automatic build deferred 1 time (memory floor)" in out2, "A's own deferral was lost through its symlinked index"


def test_resume_finds_a_symlinked_own_store_and_a_link_from_a_non_store_dir(store, monkeypatch):
    """The two symlink cases need DIFFERENT directories: a store whose own canonical index is a link reads the spelled
    parent (the writer's); a link from a directory that is not a store reads the resolved parent."""
    a_recall, a_index = _linked_store(store)
    record_deferral(a_recall, "catchup", "free_inactive_gb=2.00 floor_gb=6.0")
    monkeypatch.setattr(cli, "project_data_dir", lambda project=None: a_recall)
    monkeypatch.setattr(cli, "project_index_dir", lambda project=None: a_index)
    # (A bare resume of A is not driven here: with an index that resolves into B, resume's own workspace-disagreement
    # refusal fires first, by design, so only an explicit --index reaches the notice.)
    out = _resume(store, monkeypatch, real_resolver=True, index_arg=str(a_index))
    assert "automatic build deferred 1 time (memory floor)" in out, "--index at A's canonical spelling must read A's spelled parent"


def test_a_link_from_a_non_store_directory_reads_the_stores_own_trace(store, monkeypatch):
    """--index at a link that is NOT spelled as a store's index but resolves to one: the resolved parent is the store's
    data dir, so its own deferral is found, and a foreign file beside the link is not shown."""
    record_deferral(store.data, "catchup", "free_inactive_gb=2.00 floor_gb=6.0")
    linkdir = store.tmp / "linked"
    linkdir.mkdir()
    link = linkdir / "index"
    link.symlink_to(store.index)
    assert cli._index_gripspace_root(link) is not None, "premise: the guard accepts a link whose resolved shape is canonical"
    assert link.parent != link.resolve().parent, "premise: the unresolved parent is a different directory"
    out = _resume(store, monkeypatch, real_resolver=True, index_arg=str(link))
    assert "automatic build deferred 1 time (memory floor)" in out, "the store's own deferral must be found through the link"
    (linkdir / DEFERRALS_FILENAME).write_text(
        json.dumps({"ts": 1.0, "site": "catchup", "verdict": "refuse", "numbers": "n"}) + "\n")
    (store.data / DEFERRALS_FILENAME).unlink()
    out = _resume(store, monkeypatch, real_resolver=True, index_arg=str(link))
    assert "memory floor" not in out, "a file beside the link, which this store never wrote, must not be shown as its history"


# --------------------------------------------------------------------------------------
# The caller of a build. A build that ran at 03:33 left a row saying only that a build
# happened; who asked for it died with the parent process, and that answer cannot be
# recovered afterwards, so it has to be written while the parent still exists.
# --------------------------------------------------------------------------------------

_SRC = str(Path(__file__).resolve().parents[2] / "src")


def test_the_funnel_records_the_caller_who_started_the_build(store, monkeypatch):
    monkeypatch.setattr(cli, "_acquire_build_lock", lambda *a, **k: 1)
    monkeypatch.setattr(cli, "_release_build_lock", lambda fd: None)
    monkeypatch.setattr(cli, "_archive_and_build_locked", lambda *a, **k: object())
    assert cli._archive_and_build(store.tmp, use_embeddings=False, incremental=True) is not None
    row = _rows(store)[0]
    assert row["event"] == "built"
    assert row["pid"] == os.getpid()
    assert row["ppid"] == os.getppid()
    assert row["argv"] == sys.argv
    assert "ppid_cmd" in row, "the key is always present; None means ps could not be read"


def test_the_parents_command_line_comes_from_ps_and_not_a_placeholder(tmp_path):
    """The parent's REAL command line must land in the row.

    The writer is spawned through /bin/sh whose own command line carries a marker, so the
    only way the marker can reach the row is a real ``ps`` read of the real parent. A stub, a
    hardcoded None, or an unconsulted ps fails this. The trailing ``; :`` keeps the shell
    alive as the parent, because a shell that execs its last command would leave the python
    process reparented to pytest and the marker would vanish.
    """
    data = tmp_path / "root" / ".synapt" / "recall"
    data.mkdir(parents=True)
    marker = "fathom-ps-marker-14-2"
    script = (
        "import sys\n"
        f"sys.path.insert(0, {_SRC!r})\n"
        "from pathlib import Path\n"
        "from synapt.recall.build_deferrals import record_build\n"
        f"# {marker}\n"
        "record_build(Path(sys.argv[1]))\n"
    )
    subprocess.run(
        ["/bin/sh", "-c",
         f"{sys.executable} -c {shlex.quote(script)} {shlex.quote(str(data))} ; :"],
        check=True, capture_output=True,
    )
    rows = [json.loads(x) for x in (data / DEFERRALS_FILENAME).read_text().splitlines()]
    got = rows[0].get("ppid_cmd")
    assert got and marker in got, f"the parent's command line was not read from ps: ppid_cmd={got!r}"


def test_the_background_cold_refresh_script_records_a_built_row(tmp_path):
    """The fix lives INSIDE a string literal, so this EXECs the script.

    A source grep would prove the text exists, not that it runs. Only the heavy build edge is
    stubbed; the lock and the writer are real, and the row must land in the same file the
    funnel writes.
    """
    store_root = tmp_path / "root"
    data = store_root / ".synapt" / "recall"
    data.mkdir(parents=True)
    wrapper = (
        "import sys\n"
        f"sys.path.insert(0, {_SRC!r})\n"
        "import synapt.recall.cli as cli\n"
        "cli._archive_and_build_locked = lambda *a, **k: object()\n"
        + cli._BACKGROUND_COLD_REFRESH_SCRIPT
    )
    subprocess.run([sys.executable, "-c", wrapper, str(store_root), str(tmp_path)],
                   check=True, capture_output=True)
    rows = [json.loads(x) for x in (data / DEFERRALS_FILENAME).read_text().splitlines()]
    assert [r.get("event") for r in rows] == ["built"], "the background refresh built and left no row"
    assert rows[0]["pid"] != os.getpid(), "the row was written by the spawned process, not by this test"

    # control: the same exec with a build that produced nothing records nothing
    (data / DEFERRALS_FILENAME).unlink()
    nothing = wrapper.replace("lambda *a, **k: object()", "lambda *a, **k: None")
    subprocess.run([sys.executable, "-c", nothing, str(store_root), str(tmp_path)],
                   check=True, capture_output=True)
    assert not (data / DEFERRALS_FILENAME).exists(), "control: nothing was built, so nothing is recorded"


def test_the_ledger_is_owner_only_on_create_and_on_a_world_readable_file(store):
    """A row can carry a parent's command line, so the file is owner-only -- on a file created
    now, and on one an earlier version already left at the umask default."""
    p = store.data / DEFERRALS_FILENAME
    p.write_text(json.dumps({"ts": 1.0, "event": "built"}) + "\n")
    os.chmod(p, 0o644)
    assert stat.S_IMODE(p.stat().st_mode) == 0o644, "premise: the file starts world-readable"
    record_build(store.data)
    assert stat.S_IMODE(p.stat().st_mode) == 0o600, "an existing ledger must be tightened, not just a new one"

    p.unlink()
    record_build(store.data)
    assert stat.S_IMODE(p.stat().st_mode) == 0o600, "and on CREATE, where the row is written first"


def test_a_new_ledger_is_owner_only_from_the_first_byte(tmp_path, monkeypatch):
    """The create-time mode is a guard of its own, and the test above cannot see it.

    That one asserts the mode after record_build returns, which the after-the-fact chmod would
    have fixed either way -- so the create block can be deleted and it stays green. Here the
    chmod is neutered and the umask is wide, so only a file BORN at 0600 passes.
    """
    monkeypatch.setattr(build_deferrals.os, "chmod", lambda *a, **k: None)
    old = os.umask(0)          # umask can only clear bits; 0 makes a wide create visible
    try:
        record_build(tmp_path, caller={})
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / DEFERRALS_FILENAME).stat().st_mode) == 0o600


def test_the_parents_command_line_is_capped(monkeypatch):
    """A command line is an unbounded place for a secret, so the recorded field is bounded."""
    monkeypatch.setattr(build_deferrals.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout="x" * 5000 + "\n"))
    assert len(build_deferrals._parent_command_line(2)) == build_deferrals._MAX_CMD_CHARS


def test_the_ps_read_is_behind_a_timeout_and_expiry_degrades_to_none(monkeypatch):
    """A hung ps must not hang the build it describes: the call carries a timeout, and a
    timeout is a None rather than an exception escaping into the build."""
    seen = {}

    def fake(*a, **k):
        seen.update(k)
        raise subprocess.TimeoutExpired("ps", k.get("timeout"))

    monkeypatch.setattr(build_deferrals.subprocess, "run", fake)
    assert build_deferrals._parent_command_line(2) is None
    assert seen.get("timeout") == build_deferrals._PS_TIMEOUT_S
