"""The session-start build must not start under host memory pressure, and only
one build may run host-wide.

The witnesses here drive the REAL ``cmd_catchup`` rather than the helpers, so the
ordering (gate → host lock → build) is what is under test. Everything around the
new logic is stubbed: the transcript dirs, the archive/journal step, compaction and
the subprocess spawn, with the spawn recorded so the assertions are about what was
actually launched.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from synapt.recall import cli


class _Recorder:
    """Records every subprocess.run/Popen so a witness can assert what launched."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def run(self, argv, *a, **k):
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    def popen(self, argv, *a, **k):
        self.calls.append(list(argv))

        class _P:
            pid = 4242
        return _P()

    @property
    def builds(self) -> list[list[str]]:
        return [c for c in self.calls if "build" in c and "--incremental" in c]


@pytest.fixture
def catchup_env(tmp_path, monkeypatch):
    """cmd_catchup with only its heavy edges stubbed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "project_transcript_dirs", lambda project: [tmp_path])
    monkeypatch.setattr(cli, "_catchup_archive_and_journal", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_journal_path", lambda *a, **k: tmp_path / "journal.jsonl")
    monkeypatch.setattr(
        "synapt.recall.journal.compact_journal", lambda *a, **k: 0, raising=False
    )
    rec = _Recorder()
    monkeypatch.setattr(cli.subprocess, "run", rec.run)
    monkeypatch.setattr(cli.subprocess, "Popen", rec.popen)
    # The oversize tail imports its entry point from query_freshness inside
    # cmd_catchup, so patching the module attribute keeps it off a real store.
    import synapt.recall.query_freshness as qf
    monkeypatch.setattr(qf, "catchup_oversize_transcripts", lambda *a, **k: [])
    monkeypatch.setattr(qf, "format_query_freshness", lambda r: "stub")
    # The host lock must be the REAL mechanism (W1 holds it and expects cmd_catchup
    # to find it held), so it is redirected to the test's tmp instead of stubbed.
    # Stubbing the helper here would have made W1 pass for the wrong reason.
    host = tmp_path / "hostsynapt"
    monkeypatch.setattr(cli, "_host_synapt_dir", lambda: host)
    return rec


def _args(**kw):
    class A:
        no_build = False
    a = A()
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_gate_refuses_so_no_build_starts(catchup_env, monkeypatch):
    """W2: under a REFUSE verdict, no build process is launched and one line says so."""
    # low swap on purpose: the floor is the only arm, so a refusal here cannot be
    # the swap arm doing the work.
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:5.9:9")  # 5.9 GB free+inactive
    cli.cmd_catchup(_args())
    assert catchup_env.builds == [], (
        f"a build was launched under a REFUSE verdict: {catchup_env.builds}"
    )


def test_gate_pass_starts_exactly_one_build(catchup_env, monkeypatch):
    """Control for W2: a PASS verdict and no other holder starts exactly one build."""
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:7168:20.0:9")
    cli.cmd_catchup(_args())
    assert len(catchup_env.builds) == 1, (
        f"expected exactly one build under PASS, got {catchup_env.builds}"
    )


def test_second_start_skips_the_build_while_the_host_lock_is_held(catchup_env, monkeypatch):
    """W1: one build host-wide — a start that finds the host lock held does not launch one."""
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:7168:20.0:9")
    held = cli._acquire_build_lock(cli._host_synapt_dir(), timeout=0, name=cli._HOST_BUILD_LOCK)
    assert held is not None, "could not take the host lock for the witness"
    try:
        cli.cmd_catchup(_args())
        assert catchup_env.builds == [], (
            f"a second start launched its own build while the host lock was held: "
            f"{catchup_env.builds}"
        )
    finally:
        cli._release_build_lock(held)


def test_cannot_measure_still_builds_and_says_so(catchup_env, monkeypatch, capsys):
    """CANNOT MEASURE is not a refusal.

    A gate that cannot read the host is an instrument failure, and an earlier
    misread of one stopped real maintenance work for two nights; a missing
    instrument must not stop a build here either. The verdict is a warning, not a
    deferral.
    """
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "not:a:measurement")
    cli.cmd_catchup(_args())
    assert len(catchup_env.builds) == 1, (
        "an unmeasurable host deferred the build; that is the false-REFUSE shape "
        "this verdict exists to avoid"
    )
    err = capsys.readouterr().err
    # The wording is not the contract; the two facts are: a build ran, and the
    # reason it ran was recorded rather than swallowed.
    assert "could not measure" in err.lower(), f"no warning line on stderr: {err!r}"
    assert "proceeding" in err.lower(), f"warning does not say it proceeded: {err!r}"


def test_other_platforms_are_inert_and_do_not_warn(catchup_env, monkeypatch, capsys):
    """The thresholds mirror a macOS host gate; elsewhere it simply does not apply.

    Not a deferral and not a warning: the condition is that the gate is not
    applicable there, and a line on every session start on every non-darwin desk
    would be noise dressed as diligence."""
    monkeypatch.delenv("SYNAPT_RECALL_MEM_FAKE", raising=False)
    monkeypatch.setattr(cli.sys, "platform", "linux")
    cli.cmd_catchup(_args())
    assert len(catchup_env.builds) == 1, (
        f"a non-darwin host deferred the build: {catchup_env.builds}"
    )
    assert "measure" not in capsys.readouterr().err.lower(), (
        "a non-darwin host warned about an instrument that does not apply there"
    )


def test_swap_percent_is_not_a_gate(catchup_env, monkeypatch):
    """A host at 91% swap with plenty free still builds.

    Allocated swap is held rather than released on this platform, so swap percent
    does not track pressure: measured 2026-09-25, a large process quitting raised
    free+inactive 4.89 -> 6.50 GB while swap ROSE 80.1% -> 84.3%. Thresholding it
    would defer builds on busy-but-safe hosts, so it is reported and not gated."""
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "3727:4096:6.5:9")  # swap 91%, 6.5 GB free
    cli.cmd_catchup(_args())
    assert len(catchup_env.builds) == 1, (
        f"a high swap reading deferred a build on a host with 6.5 GB free: "
        f"{catchup_env.builds}"
    )


def test_the_floor_defers_a_host_that_would_have_taken_the_incident(catchup_env, monkeypatch):
    """5.9 GB defers where the old 4 GB floor would have allowed the build.

    The incident this exists for started on a host that still read 4.89 GB
    free+inactive, so a 4 GB floor is not a gate against it. 6 GB is."""
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:5.9:9")  # 5.9 GB, low swap
    cli.cmd_catchup(_args())
    assert catchup_env.builds == [], (
        f"a host at 5.9 GB free+inactive started a build: {catchup_env.builds}"
    )


def test_the_floor_is_overridable_by_env(catchup_env, monkeypatch):
    """The floor is one constant behind one env var, so a permanent change of the
    number is a one-constant change rather than a re-review -- and a host at 5.9 GB
    builds when the floor is lowered to 3."""
    monkeypatch.setenv("SYNAPT_BUILD_MIN_FREE_GB", "3.0")
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:5.9:9")
    cli.cmd_catchup(_args())
    assert len(catchup_env.builds) == 1, (
        f"the env override did not lower the floor: {catchup_env.builds}"
    )


class TestFloorValidation:
    """The env override must accept only a positive, finite float.

    Every case below was measured on the merged bytes before this change: `0` and
    `-1` set a floor that no host can miss, `nan` compares False against everything
    so it disables the gate while looking configured, and `inf` never opens it. A
    number the gate cannot use must fall back to the constant, not become a verdict.
    """

        # "6" is deliberately NOT in this list: falling back to the constant gives the
    # same value, so that case would pass whether or not the validation ran.
    @pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "-inf", "abc", ""])
    def test_a_number_the_gate_cannot_use_falls_back_to_the_constant(self, raw, monkeypatch):
        monkeypatch.setenv("SYNAPT_BUILD_MIN_FREE_GB", raw)
        got = cli._build_min_free_gb()
        assert got == cli.BUILD_MIN_FREE_INACTIVE_GB, (
            f"SYNAPT_BUILD_MIN_FREE_GB={raw!r} produced floor={got!r}; a value the "
            "gate cannot use must fall back to the constant"
        )

    @pytest.mark.parametrize("raw,expected", [("2.5", 2.5), ("6.0", 6.0), ("12", 12.0)])
    def test_a_usable_number_is_honoured(self, raw, expected, monkeypatch):
        monkeypatch.setenv("SYNAPT_BUILD_MIN_FREE_GB", raw)
        assert cli._build_min_free_gb() == expected

    def test_a_disabling_override_cannot_switch_the_gate_off(self, catchup_env, monkeypatch):
        """The end-to-end consequence, not just the helper's return."""
        for raw in ("0", "-1", "nan"):
            monkeypatch.setenv("SYNAPT_BUILD_MIN_FREE_GB", raw)
            monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:5.0:9")
            catchup_env.calls.clear()
            cli.cmd_catchup(_args())
            assert catchup_env.builds == [], (
                f"SYNAPT_BUILD_MIN_FREE_GB={raw!r} let a 5.0 GB host start a build"
            )


def test_the_fake_seam_is_recall_owned(catchup_env, monkeypatch):
    """One name, owned by this repository: the old one was the host gate's."""
    monkeypatch.delenv("SYNAPT_RECALL_MEM_FAKE", raising=False)
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:5.0:9")
    cli.cmd_catchup(_args())
    assert catchup_env.builds == [], (
        "SYNAPT_RECALL_MEM_FAKE did not force a verdict, so the seam is not wired "
        "to the name this repository owns"
    )
