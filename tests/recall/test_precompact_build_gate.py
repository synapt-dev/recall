"""The PreCompact rebuild must not start under host memory pressure, and only one
build may run host-wide.

The witnesses drive the REAL ``cmd_hook`` precompact branch rather than the helpers,
so the ordering (gate -> host lock -> build -> release) is what is under test. The
heavy edges are stubbed: the index dir, the invariant assertion, the codex cache
step, the rebuild itself, the sync, and the channel heartbeat; the journal write is
RECORDED rather than stubbed away, because its survival under a refusal is one of
the three things this change must not break.

The host lock is the REAL mechanism, redirected to the test's tmp: witness (b) holds
it and expects the branch to find it held, and witness (c) expects it released
afterwards. Stubbing the lock helper would make both pass for the wrong reason.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from synapt.recall import cli


class _Recorder:
    """Records the calls this change gates, so the assertions are about what ran."""

    def __init__(self):
        self.built: list[Path] = []
        self.synced: list[Path] = []
        self.journaled: list[Path] = []

    def build(self, project, **kw):
        self.built.append(project)
        return None

    def sync(self, project):
        self.synced.append(project)

    def journal(self, project):
        self.journaled.append(project)


@pytest.fixture
def precompact_env(tmp_path, monkeypatch):
    """cmd_hook(precompact) with only its heavy edges stubbed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(cli, "project_transcript_dirs", lambda project: [tmp_path])
    monkeypatch.setattr(cli, "project_index_dir", lambda project: tmp_path / "index")
    monkeypatch.setattr(cli, "_refuse_if_index_disagrees_with_source", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_configure_codex_cwd_cache", lambda *a, **k: None)
    rec = _Recorder()
    monkeypatch.setattr(cli, "_archive_and_build", rec.build)
    monkeypatch.setattr(cli, "_sync_after_rebuild", rec.sync)
    monkeypatch.setattr(cli, "_precompact_journal_write", rec.journal)
    # The heartbeat imports its entry point inside the branch; patch the module
    # attribute so the test never touches a real channel.
    import synapt.recall.channel as channel
    monkeypatch.setattr(channel, "channel_heartbeat", lambda *a, **k: None, raising=False)
    # The host lock must be the REAL mechanism (see the module docstring).
    host = tmp_path / "hostsynapt"
    monkeypatch.setattr(cli, "_host_synapt_dir", lambda: host)
    return rec


def _args(event: str = "precompact"):
    class A:
        pass

    a = A()
    a.event = event
    return a


def test_refuse_skips_the_rebuild_and_still_writes_the_journal(precompact_env, monkeypatch, capsys):
    """W(a): under REFUSE the rebuild and the sync are skipped, the journal is not.

    The journal assertion is the load-bearing half: crash-recovery state is why the
    hook exists, and a gate that also skipped the journal would trade one failure
    for a worse one.
    """
    # low swap on purpose: the free+inactive floor is the only arm, so a refusal
    # here cannot be the swap arm firing.
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:5.9:9")
    cli.cmd_hook(_args())

    assert precompact_env.built == [], "the rebuild ran under a REFUSE verdict"
    assert precompact_env.synced == [], "the sync ran under a REFUSE verdict"
    assert precompact_env.journaled, "the journal write was skipped along with the rebuild"
    out = capsys.readouterr().err
    assert "REFUSE" in out and "5.9" in out, f"the refusal line does not carry the numbers: {out!r}"


def test_held_host_lock_skips_the_rebuild_and_still_writes_the_journal(
    precompact_env, monkeypatch, capsys
):
    """W(b): with the host lock held elsewhere, the rebuild is skipped, journal kept."""
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:7168:20.0:9")
    held = cli._acquire_build_lock(cli._host_synapt_dir(), timeout=0, name=cli._HOST_BUILD_LOCK)
    assert held is not None, "could not take the host lock for the fixture"
    try:
        cli.cmd_hook(_args())
    finally:
        cli._release_build_lock(held)

    assert precompact_env.built == [], "the rebuild ran while the host lock was held"
    assert precompact_env.journaled, "the journal write was skipped with the rebuild"
    assert cli._HOST_BUILD_LOCK in capsys.readouterr().err


def test_pass_runs_the_rebuild_and_releases_the_host_lock(precompact_env, monkeypatch):
    """W(c): on a pass the rebuild runs once and the lock is free afterwards.

    The release is asserted by TAKING the lock after the branch returns: a build that
    ran without releasing would leave the next session start unable to build, which is
    a worse outcome than the spike this guards.
    """
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:7168:20.0:9")
    cli.cmd_hook(_args())

    assert len(precompact_env.built) == 1, f"expected exactly one rebuild, got {precompact_env.built}"
    assert precompact_env.journaled, "the journal write did not run on the pass path"
    after = cli._acquire_build_lock(cli._host_synapt_dir(), timeout=0, name=cli._HOST_BUILD_LOCK)
    assert after is not None, "the host lock was not released after the rebuild"
    cli._release_build_lock(after)


def test_cannot_measure_rebuilds_and_says_so(precompact_env, monkeypatch, capsys):
    """An unreadable instrument is not a memory verdict: rebuild, and say why."""
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "not-a-measurement")
    cli.cmd_hook(_args())

    assert len(precompact_env.built) == 1, "an unmeasurable host stopped the rebuild"
    assert "could not measure" in capsys.readouterr().err
