"""`channel post --name` is refused, not accepted and ignored.

--name sets a display name on join. Before this, post accepted it, ignored
it, and posted under the session's own identity: a green exit for a post
that did not go out under the name the caller asked for.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run(project: Path, home: Path, *argv: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SYNAPT_")}
    env["HOME"] = str(home)
    return subprocess.run(
        [sys.executable, "-m", "synapt.recall.cli", "channel", *argv],
        cwd=project, env=env, capture_output=True, text=True, timeout=60,
    )


@pytest.fixture
def workdir(tmp_path: Path):
    project = tmp_path / "proj"
    home = tmp_path / "home"
    project.mkdir()
    home.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    return project, home


def _posts(project: Path) -> list[str]:
    channels = project / ".synapt" / "recall" / "channels"
    return sorted(p.name for p in channels.glob("*.jsonl")) if channels.exists() else []


@pytest.mark.parametrize(
    "argv",
    [
        ("post", "dev", "hello", "--name", "other"),
        ("post", "dev", "hello", "--name="),            # empty name is still a name
        ("post", "dev", "hello", "--na", "other"),      # argparse abbreviation
        ("--name", "other", "post", "dev", "hello"),    # flag before the action
        ("post", "dev", "--name", "other"),             # no message: name refused first
    ],
    ids=["name", "empty-name", "abbrev", "flag-first", "no-message"],
)
def test_post_with_name_is_refused_and_posts_nothing(workdir, argv):
    project, home = workdir
    r = _run(project, home, *argv)
    assert r.returncode != 0
    assert r.stdout == ""
    lines = r.stderr.strip().splitlines()
    assert len(lines) == 1, r.stderr
    assert "--name is not used by 'post'" in lines[0]
    assert "channel join <channel> --name <name>" in lines[0]
    assert _posts(project) == []


def test_refusal_does_not_echo_the_name(workdir):
    """A hostile name (newline, ANSI escape) cannot shape the one-line refusal."""
    project, home = workdir
    r = _run(project, home, "post", "dev", "hello", "--name", "x\n\x1b[2Jforged line")
    assert r.returncode != 0
    assert len(r.stderr.strip().splitlines()) == 1
    assert "forged" not in r.stderr and "\x1b" not in r.stderr
    assert _posts(project) == []


def test_post_without_name_still_posts(workdir):
    project, home = workdir
    r = _run(project, home, "post", "dev", "hello")
    assert r.returncode == 0, r.stderr
    assert _posts(project) == ["dev.jsonl"]


def test_named_route_the_refusal_points_to_posts_as_that_name(workdir):
    """The refusal names join --name then post; that route must really work."""
    project, home = workdir
    assert _run(project, home, "join", "dev", "--name", "other").returncode == 0
    r = _run(project, home, "post", "dev", "hello")
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("[#dev] other: hello"), r.stdout
