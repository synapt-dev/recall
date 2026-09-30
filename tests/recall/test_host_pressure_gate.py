"""The build gate reads the KERNEL's pressure level, not only free+inactive.

WHY THIS FILE EXISTS (the 2026-09-30 host-pressure incident). `_host_memory_verdict` was an
ADMISSION-ONLY gate: it read `vm_stat` free+inactive and `vm.swapusage`, decided once, and
never looked again. On 2026-09-30 a session-start catchup passed that gate at 07:09 on a
host that still had free memory, then grew to about 5.6 GB and held the kernel at pressure
level 4 while the fleet was loaded. Free+inactive is a SNAPSHOT of pages; the pressure level
is the kernel's own integrated judgement of whether the host is coping, and the gate did not
ask for it.

THESE TESTS ARE THE SPEC. Two arms of the answer are deliberately separated:

* a HIGH pressure level REFUSES, even with a large free+inactive reading -- otherwise the
  clause would be decorative, since the floor arm alone would decide every case here;
* an UNREADABLE pressure instrument does NOT refuse. That is the existing doctrine in this
  function ("a gate that cannot read the host is an instrument failure ... a missing
  instrument must not stop a build here either") and it is scoped to this new arm rather
  than reversed.

The `SYNAPT_RECALL_MEM_FAKE` seam gains an OPTIONAL FIFTH field for the pressure level. The
fourth field (`pane_count`, e.g. "100:4096:20.0:9") is LEFT ALONE: it is already used by
existing tests, and a fifth field rather than a redefinition is what keeps those fakes
meaning what they meant. The control for that is the last test here.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from synapt.recall import cli


def test_a_high_pressure_level_refuses_even_with_plenty_of_free_memory(monkeypatch):
    """The floor arm CANNOT be doing the work here: 20 GB free+inactive is far above it."""
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:20.0:9:4")
    verdict, numbers = cli._host_memory_verdict()
    assert verdict == "refuse", (
        f"pressure level 4 (CRITICAL) passed the gate with {numbers} -- the level clause is "
        f"not wired, or the free+inactive arm answered instead"
    )
    assert "pressure_level=4" in numbers, f"the refusal does not name the level: {numbers!r}"


def test_level_two_refuses_because_the_boundary_is_above_normal(monkeypatch):
    """1 is NORMAL. The gate refuses ABOVE it, so 2 (WARN) must already refuse -- a test at
    one point of a boundary is a claim about that point, so both sides are pinned here."""
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:20.0:9:2")
    verdict, numbers = cli._host_memory_verdict()
    assert verdict == "refuse", f"level 2 (WARN) passed the gate: {numbers}"


def test_level_one_passes_on_a_host_with_memory(monkeypatch):
    """The other side of the boundary, so `refuse` above is not simply 'always'."""
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:20.0:9:1")
    verdict, numbers = cli._host_memory_verdict()
    assert verdict == "pass", f"level 1 (NORMAL) was refused on a 20 GB host: {numbers}"
    assert "pressure_level=1" in numbers, f"the passing reading is not reportable: {numbers!r}"


def test_a_refusal_from_the_floor_still_names_the_floor_and_not_the_level(monkeypatch):
    """Two arms, two reasons. A low-memory host at NORMAL pressure must still refuse, and the
    line must not claim the level did it -- a reader fixing the wrong thing is the failure
    this separation exists to prevent.

    The floor is PINNED here rather than assumed: this host resolves it to 3.0, so a 5.9 GB
    reading would pass the floor arm and the test would be measuring the other clause.
    """
    monkeypatch.setenv("SYNAPT_BUILD_MIN_FREE_GB", "6")
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:5.9:9:1")
    verdict, numbers = cli._host_memory_verdict()
    assert verdict == "refuse", f"5.9 GB free+inactive passed a pinned 6 GB floor: {numbers}"
    assert "pressure_level=1" in numbers
    assert "floor_gb=6.0" in numbers, f"the pinned floor is not the one reported: {numbers!r}"


def test_the_kernel_is_actually_asked_on_a_real_host(monkeypatch):
    """THE ROW THE MUTATION FOUND MISSING.

    Replacing `pressure = _host_pressure_level()` with `pressure = None` left EVERY other row
    in this file green, so nothing here witnessed that the kernel is read at all. The
    unreadable row cannot be that witness: "we asked and it failed" and "we never asked" are
    the same bytes downstream, which is exactly why `None` is not reported as a normal level.

    So this row drives the REAL path -- no fake -- with the sysctl patched to ANSWER, and
    asserts the answer reaches the verdict. A gate that reads the level in a fake and never
    in production would pass every other test here.
    """
    monkeypatch.delenv("SYNAPT_RECALL_MEM_FAKE", raising=False)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    real_run = subprocess.run

    def fake_run(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and any("memorystatus" in str(c) for c in cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout="4\n", stderr="")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)
    verdict, numbers = cli._host_memory_verdict()
    assert verdict == "refuse", (
        f"the kernel answered 4 (CRITICAL) and the gate did not ask, or did not honour it: "
        f"{numbers}"
    )
    assert "pressure_level=4" in numbers, f"the reading did not reach the line: {numbers!r}"


def test_an_unreadable_pressure_instrument_does_not_stop_the_build(monkeypatch, capsys):
    """CONSISTENT WITH THE FUNCTION'S OWN DOCTRINE, and scoped to this arm.

    `cannot_measure` is not a refusal: an earlier misread of a broken instrument stopped real
    maintenance for two nights. So a host whose pressure sysctl cannot be read still gets the
    free+inactive verdict it always got -- and the gap is REPORTED rather than silently
    absent, because a gate that did not ask one of its two questions must say so.
    """
    monkeypatch.delenv("SYNAPT_RECALL_MEM_FAKE", raising=False)
    real_run = subprocess.run

    def fake_run(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and any("memorystatus" in str(c) for c in cmd):
            raise OSError("no such sysctl")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    verdict, numbers = cli._host_memory_verdict()
    assert verdict in ("pass", "refuse"), f"an unreadable instrument produced {verdict!r}"
    assert "pressure_level=unreadable" in numbers, (
        f"the gate did not ask its pressure question and did not say so: {numbers!r}"
    )


def test_the_fourth_fake_field_is_not_redefined(monkeypatch):
    """CONTROL for the seam. `100:4096:20.0:9` is an existing fake in this repository; if a
    fifth field had been added by redefining the fourth, this would now REFUSE."""
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:20.0:9")
    verdict, numbers = cli._host_memory_verdict()
    assert verdict == "pass", (
        f"the four-field fake changed meaning -- a fifth field, not a redefinition, is the "
        f"contract: {numbers}"
    )


# --- (b) the MID-RUN monitor: admission passes, the host turns, the build stops ----------
#
# Admission answers one question at one instant. These two rows are the discriminating pair
# for what happens when the answer CHANGES while the build runs -- which is that incident's
# shape exactly, where the build passed at 07:09 and then grew to about 5.6 GB with nothing
# looking.


class _FastWatchdog(cli._PressureWatchdog):
    """The same watchdog, sampling fast enough to witness a stop inside a test."""

    def __init__(self, interval: float = 0.02) -> None:
        super().__init__(interval=interval)


def _wire_build(monkeypatch, build):
    released: list[int] = []
    monkeypatch.setattr(cli, "_acquire_build_lock", lambda *a, **k: 42)
    monkeypatch.setattr(cli, "_release_build_lock", lambda fd: released.append(fd))
    monkeypatch.setattr(cli, "project_data_dir", lambda p: Path(p))
    monkeypatch.setattr(cli, "_PressureWatchdog", _FastWatchdog)
    monkeypatch.setattr(cli, "_archive_and_build_locked", build)
    # `record_build` writes a receipt into a real data dir; this witness is about the STOP
    # path, where the receipt must NOT be written at all, so it is watched rather than mocked
    # away.
    import synapt.recall.build_deferrals as _deferrals

    recorded: list[object] = []
    monkeypatch.setattr(_deferrals, "record_build", lambda d: recorded.append(d))
    return released, recorded


def test_a_build_that_passed_admission_is_stopped_when_the_host_turns(monkeypatch, capsys):
    """THE WITNESS: admission passes, the fake drops mid-run, the build stops, the lock is freed."""
    import threading
    import time

    started: list[bool] = []

    def slow_build(*a, **k):
        started.append(True)
        # A PYTHON-LEVEL loop on purpose: the stop is raised at a bytecode boundary, so the
        # witness must give it one to land on. A build that never returned to the interpreter
        # could not be stopped by this mechanism, and that bound is stated in the watchdog's
        # docstring rather than left for a reader to discover.
        #
        # AND BOUNDED ON PURPOSE. With the stop mutated out, an unbounded loop HANGS the run
        # instead of failing it -- and a test that blocks the instrument is worse than no
        # test. The first version of this row did exactly that: the mutation run for M5 hung
        # for two minutes and had to be killed, which is not a RED and not a result.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            time.sleep(0.01)
        return "RAN TO COMPLETION WITHOUT BEING STOPPED"

    released, recorded = _wire_build(monkeypatch, slow_build)
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:20.0:9:1")  # ADMISSION PASSES

    def turn_the_host():
        time.sleep(0.15)
        monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:20.0:9:4")

    threading.Thread(target=turn_the_host, daemon=True).start()

    result = cli._archive_and_build(Path("/tmp/witness-does-not-need-to-exist"))

    assert started, "the build never started, so this witness proves nothing"
    assert result is None, (
        "the build ran to completion after the host turned -- the monitor did not fire"
    )
    assert released == [42], f"the lock was NOT released on the stop path: {released}"
    assert recorded == [], f"a STOPPED build left a completion receipt: {recorded}"
    err = capsys.readouterr().err
    assert "STOPPED mid-run" in err, err
    assert "pressure_level=4" in err, f"the stop does not say what it saw: {err!r}"


def test_a_stop_survives_a_broad_except_in_the_build_body(monkeypatch, capsys):
    """THE ROW BOTH READERS FOUND MISSING, and the reason the whole guarantee held only by luck.

    `BuildStoppedByPressure` is delivered by `PyThreadState_SetAsyncExc` -- an ordinary
    exception at an ordinary bytecode boundary -- so an ordinary `except Exception` catches
    it. `_archive_and_build_locked`, the function the watchdog wraps, carries SEVEN of them,
    and one is a 37-line loop over every `*.jsonl`. Swallowed, the body runs to COMPLETION,
    `built` comes back non-None, and a completion receipt is written: the exact outcome this
    guard exists to prevent, and it is SILENT.

    The witness that missed it had a stub body with no `except`, so its assertion could only
    fail if the monitor never fired at all -- **a row whose subject cannot exhibit the failure
    it asserts.** This one puts a broad handler in the body, which is the shape the real site
    has, and requires the stop to survive it.

    Sentinel and Atlas each found this independently by running the author's own harness with
    the body changed and nothing else. It is in as a row so the next reader does not have to.
    """
    import threading
    import time

    started: list[bool] = []
    swallowed: list[str] = []

    def swallowing_build(*a, **k):
        started.append(True)
        deadline = time.monotonic() + 5.0
        try:
            while time.monotonic() < deadline:
                time.sleep(0.01)
        except Exception as exc:  # noqa: BLE001 -- THE POINT OF THIS ROW
            swallowed.append(type(exc).__name__)
            return "SWALLOWED-THE-STOP"
        return "RAN TO COMPLETION"

    released, recorded = _wire_build(monkeypatch, swallowing_build)
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:20.0:9:1")

    def turn_the_host():
        time.sleep(0.15)
        monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:20.0:9:4")

    threading.Thread(target=turn_the_host, daemon=True).start()
    result = cli._archive_and_build(Path("/tmp/witness-does-not-need-to-exist"))

    assert started, "the build never started, so this witness proves nothing"
    assert swallowed == [], (
        f"a broad `except Exception` in the build body caught the stop: {swallowed}. "
        f"BuildStoppedByPressure must derive from BaseException, or the body runs on."
    )
    assert result is None, f"a swallowed stop reported a completed build: {result!r}"
    assert recorded == [], f"a stop that was swallowed wrote a completion receipt: {recorded}"
    assert released == [42], f"the lock was not released: {released}"
    err = capsys.readouterr().err
    assert "STOPPED mid-run" in err, (
        f"the operator was told NOTHING about a swallowed stop: {err!r}"
    )


def test_a_stop_survives_even_a_bare_baseexception_handler(monkeypatch, capsys):
    """THE SECOND NET, AND IT NEEDS ITS OWN ROW BECAUSE THE FIRST NET DOES NOT COVER IT.

    Deriving `BuildStoppedByPressure` from `BaseException` makes the stop uncatchable by
    `except Exception` -- but a bare `except:` or an `except BaseException` still eats it, and
    those exist in the wild for the same reason broad handlers do. That is what the watchdog's
    FLAG is for: it is set by ANOTHER THREAD, so the thread it raises in cannot swallow it,
    and it is read after the body has returned however it returned.

    **THIS ROW EXISTS BECAUSE THE MUTATION SET SAID SO.** Removing the flag read reddened
    NOTHING across the whole file -- the clause was decoration until this row. Here the body
    really does swallow the stop (the assertion says so rather than pretending otherwise),
    and the outcome must still be a stop.
    """
    import threading
    import time

    started: list[bool] = []
    swallowed: list[str] = []

    def base_swallowing_build(*a, **k):
        started.append(True)
        deadline = time.monotonic() + 5.0
        try:
            while time.monotonic() < deadline:
                time.sleep(0.01)
        except BaseException:  # noqa: BLE001 -- THE POINT OF THIS ROW
            swallowed.append("BaseException")
            return "SWALLOWED-THE-STOP-AT-THE-BASE"
        return "RAN TO COMPLETION"

    released, recorded = _wire_build(monkeypatch, base_swallowing_build)
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:20.0:9:1")

    def turn_the_host():
        time.sleep(0.15)
        monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:20.0:9:4")

    threading.Thread(target=turn_the_host, daemon=True).start()
    result = cli._archive_and_build(Path("/tmp/witness-does-not-need-to-exist"))

    assert started, "the build never started, so this witness proves nothing"
    assert swallowed, (
        "this row's premise is that the body swallows even the base exception; if it did not, "
        "the row is testing something else"
    )
    assert result is None, (
        f"the body swallowed the stop and the FLAG did not catch it: {result!r}"
    )
    assert recorded == [], f"a stop defeated at the base wrote a completion receipt: {recorded}"
    assert released == [42], f"the lock was not released: {released}"
    err = capsys.readouterr().err
    assert "STOPPED mid-run" in err, f"the operator was told nothing: {err!r}"


def test_the_control_a_build_on_a_quiet_host_is_not_stopped(monkeypatch):
    """The other arm. A monitor that stopped builds unconditionally would pass the witness
    above and be worse than no monitor at all, so the quiet case is pinned too."""
    done: list[bool] = []

    def quick_build(*a, **k):
        done.append(True)
        return "INDEX"

    released, recorded = _wire_build(monkeypatch, quick_build)
    monkeypatch.setenv("SYNAPT_RECALL_MEM_FAKE", "100:4096:20.0:9:1")

    result = cli._archive_and_build(Path("/tmp/witness-does-not-need-to-exist"))

    assert done, "the build never ran"
    assert result == "INDEX", f"a quiet host did not let the build finish: {result!r}"
    assert released == [42], "the lock was not released on the normal path"
    assert len(recorded) == 1, "a completed build did not leave its receipt"
