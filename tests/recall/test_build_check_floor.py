"""`synapt build --check-floor`: recall's own build-floor verdict as an exit code.

A caller (a gate in front of a heavy run) reads only the code and the last output line, so the code must
say three different things: 0 measured and fine, 2 measured and refused, 3 could not measure.
Exit 0 for "could not measure" would let an unread instrument pass as a pass.

Every row runs the REAL command in a subprocess under a forced reading and a scratch HOME.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from synapt.recall import cli

SRC = str(Path(cli.__file__).resolve().parents[2])


def run_check(tmp_path: Path, fake: str, *extra: str):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PYTHONPATH": SRC,
        "SYNAPT_RECALL_MEM_FAKE": fake,
    }
    if "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    result = subprocess.run(
        [sys.executable, "-B", "-m", "synapt.recall.cli", "build", "--check-floor", *extra],
        cwd=work, env=env, capture_output=True, text=True, timeout=120,
    )
    return result, work


def test_below_the_floor_exits_2_and_prints_both_numbers(tmp_path):
    result, _ = run_check(tmp_path, "0:0:3.2:0")
    assert result.returncode == 2, result.stdout + result.stderr
    line = result.stdout.strip().splitlines()[-1]
    assert "free_inactive_gb=3.20" in line and "floor_gb=6.0" in line


def test_above_the_floor_exits_0(tmp_path):
    result, _ = run_check(tmp_path, "0:0:7.0:0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "free_inactive_gb=7.00" in result.stdout


def test_pressure_above_normal_exits_2_although_free_is_large(tmp_path):
    result, _ = run_check(tmp_path, "0:0:20:0:4")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "pressure_level=4" in result.stdout


def test_cannot_measure_exits_3_with_the_reason_line(tmp_path):
    result, _ = run_check(tmp_path, "not-a-reading")
    assert result.returncode == 3, result.stdout + result.stderr
    line = result.stdout.strip().splitlines()[-1]
    assert line.startswith("cannot_measure:") and "not three numbers" in line


def test_does_no_index_work_and_leaves_the_directory_untouched(tmp_path):
    # A nonexistent --source would fail a real build; the check must not look at it.
    result, work = run_check(tmp_path, "0:0:7.0:0", "--source", str(tmp_path / "missing"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(work.iterdir()) == []
