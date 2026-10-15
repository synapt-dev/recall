"""A body capped by a CLI read can be read in full from the CLI.

Before this, `synapt recall channel read --detail max` capped a long body at
8,000 chars and advised "Use action='read_message' ...", but the CLI had no
read_message action (argparse exit 2), so from the CLI the rest of the body
could not be read at all. The CLI read now names the CLI command; the MCP
read keeps naming the MCP action.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from synapt.recall.channel import (
    MAX_BODY_CHARS,
    channel_join,
    channel_post,
    channel_read,
)
from tests.recall.test_channel import _patch_data_dir

# Run the real front door: `synapt ...` as an installed script would.
_SYNAPT = "import sys; from synapt.cli import main; sys.argv[0] = 'synapt'; main()"


def _synapt(project: Path, home: Path, argv: list[str]) -> subprocess.CompletedProcess:
    assert argv[0] == "synapt", argv
    env = {k: v for k, v in os.environ.items() if not k.startswith("SYNAPT_")}
    env["HOME"] = str(home)
    return subprocess.run(
        [sys.executable, "-c", _SYNAPT, *argv[1:]],
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


def _body(n: int) -> str:
    # Position-bearing and newline-free, so any shifted or short chunk differs.
    out, i = [], 0
    while sum(map(len, out)) < n:
        out.append(f"<{i:06d}>")
        i += 1
    return "".join(out)[:n]


def _chunk(read_message_out: str) -> str:
    lines = read_message_out.splitlines()
    start = lines.index("") + 1
    end = lines.index("[end of message]")
    return "\n".join(lines[start:end])


def test_cli_cap_note_runs_as_printed_and_returns_the_tail(workdir):
    project, home = workdir
    body = _body(9_000)
    post = _synapt(project, home, ["synapt", "recall", "channel", "post", "probe", body])
    assert post.returncode == 0, post.stderr

    read = _synapt(project, home, ["synapt", "recall", "channel", "read", "probe",
                                   "--detail", "max"])
    assert read.returncode == 0, read.stderr
    assert "action='read_message'" not in read.stdout
    commands = re.findall(r"`(synapt [^`]+)`", read.stdout)
    assert len(commands) == 1, read.stdout
    argv = shlex.split(commands[0])
    assert argv[:4] == ["synapt", "recall", "channel", "read_message"], argv
    assert argv[-2:] == ["--offset", str(MAX_BODY_CHARS)], argv

    rest = _synapt(project, home, argv)
    assert rest.returncode == 0, (rest.returncode, rest.stderr)
    assert _chunk(rest.stdout).encode() == body[MAX_BODY_CHARS:].encode()


def test_cli_read_message_honours_length(workdir):
    project, home = workdir
    body = _body(9_000)
    assert _synapt(project, home, ["synapt", "recall", "channel", "post", "probe",
                                   body]).returncode == 0
    mid = re.search(r"\[(m_[0-9a-f]+)\]",
                    _synapt(project, home, ["synapt", "recall", "channel", "read",
                                            "probe", "--detail", "max"]).stdout)
    assert mid, "read shows no message id"
    r = _synapt(project, home, ["synapt", "recall", "channel", "read_message", "probe",
                                mid.group(1), "--offset", "100", "--length", "50"])
    assert r.returncode == 0, r.stderr
    assert body[100:150] in r.stdout.splitlines()
    assert "next offset=150" in r.stdout


def test_mcp_cap_note_keeps_the_mcp_action(tmp_path):
    patcher = _patch_data_dir(tmp_path)
    patcher.start()
    try:
        channel_join("probe", agent_name="reader", display_name="Reader")
        channel_post("probe", _body(9_000), agent_name="writer")
        out = channel_read("probe", detail="max", agent_name="reader")
    finally:
        patcher.stop()
    assert "action='read_message'" in out
    assert "synapt recall channel read_message" not in out
