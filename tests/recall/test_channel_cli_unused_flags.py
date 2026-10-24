"""Every `channel` action refuses an input it does not use.

Before this, the channel CLI accepted flags and positionals an action never
read (search ignored its channel, read ignored --to, who ignored everything)
and exited 0 as if they had been honoured. The rows below are driven from the
parser itself, so a flag added later is checked against every action.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

from synapt.recall import cli

# What each action uses, written out here rather than read from the CLI, so a
# wrong entry in the CLI's own table cannot pass its own rows.
EXPECTED = {
    "post": {"channel", "message", "pin", "type"},
    "read": {"channel", "limit", "since", "detail"},
    "who": set(),
    "join": {"channel", "name"},
    "leave": {"channel"},
    "heartbeat": set(),
    "unread": set(),
    "pin": {"channel", "message"},
    "directive": {"channel", "message", "to"},
    "mute": {"channel", "target"},
    "unmute": {"channel", "target"},
    "kick": {"channel", "target"},
    "broadcast": {"message"},
    "list": set(),
    "search": {"channel", "message"},
    "chat": {"channel", "name", "target", "limit"},
    "read_message": {"channel", "message", "offset", "length"},
    "watch": {"channel", "message", "rooms", "since", "include_mine", "all_rooms"},
    "latest": {"channel", "n", "catch_up"},
}

VALUES = {
    "limit": ["--limit", "5"],
    "since": ["--since", "2026-01-01T00:00:00"],
    "pin": ["--pin"],
    "type": ["--type", "status"],
    "to": ["--to", "x"],
    "target": ["--target", "x"],
    "detail": ["--detail", "low"],
    "name": ["--name", "x"],
    "offset": ["--offset", "1"],
    "length": ["--length", "1"],
    "include_mine": ["--include-mine"],
    "all_rooms": ["--all"],
    "n": ["--n", "1"],
    "catch_up": ["--catch-up"],
}


def _channel_parser() -> argparse.ArgumentParser:
    sub = next(a for a in cli.make_parser()._actions
               if isinstance(a, argparse._SubParsersAction))
    return sub.choices["channel"]


def _parser_dests() -> set[str]:
    return {a.dest for a in _channel_parser()._actions
            if a.dest not in ("help", "action")}


def _actions() -> list[str]:
    return list(next(a for a in _channel_parser()._actions
                     if a.dest == "action").choices)


def _argv(action: str, dest: str) -> list[str]:
    if action == "broadcast" and dest in ("channel", "message"):
        # One positional is broadcast's message; a second one is a channel.
        return [action, "dev", "body"] if dest == "channel" else [action, "body"]
    if dest == "channel":
        return [action, "dev"]
    if dest == "message":
        return [action, "dev", "body"]
    if dest == "rooms":
        return [action, "dev", "body", "more"]
    return [action, *VALUES[dest]]


def _pairs(used: bool) -> list[tuple[str, str]]:
    table = EXPECTED
    out = []
    for action in _actions():
        for dest in sorted(_parser_dests()):
            in_use = dest in table.get(action, ())
            # <message> rides behind <channel>; where the action uses neither,
            # the <channel> row already covers it.
            if dest == "message" and not used and \
                    "channel" not in table.get(action, ()) \
                    and action != "broadcast":
                continue
            # <room>... rides behind both; an action that refuses either
            # of them refuses that one first, and its own row covers it.
            if dest == "rooms" and not used and \
                    not {"channel", "message"} <= table.get(action, set()):
                continue
            if in_use == used:
                out.append((action, dest))
    return out


@pytest.fixture
def calls(monkeypatch):
    """Record every channel function the CLI reaches."""
    import synapt.recall.channel as channel_mod
    import synapt.recall.channel_chat as chat_mod
    seen: list[tuple[str, dict]] = []
    for attr in dir(channel_mod):
        if attr.startswith("channel_"):
            def rec(*a, _n=attr, **kw):
                seen.append((_n, kw))
                return [] if _n in ("channel_search", "channel_list_channels") else (
                    {} if _n == "channel_unread" else "ok")
            monkeypatch.setattr(channel_mod, attr, rec)
    monkeypatch.setattr(chat_mod, "main", lambda **kw: seen.append(("chat", kw)))
    return seen


def _call(argv: list[str]) -> int:
    args = cli.make_parser().parse_args(["channel", *argv])
    try:
        cli.cmd_channel(args)
    except SystemExit as e:
        return e.code
    return 0


def test_table_names_every_action_and_only_parser_inputs():
    assert set(EXPECTED) == set(_actions())
    for action, used in EXPECTED.items():
        assert used <= _parser_dests(), action
    assert {a: set(u) for a, u in cli._CHANNEL_ACTION_INPUTS.items()} == EXPECTED


def test_every_parser_input_is_checked_for_being_given():
    assert set(cli._CHANNEL_INPUT_DESTS) == _parser_dests()


@pytest.mark.parametrize("action,dest", _pairs(used=False))
def test_unused_input_is_refused_before_anything_runs(action, dest, calls, capsys):
    rc = _call(_argv(action, dest))
    err = capsys.readouterr().err.strip().splitlines()
    label = {"channel": "<channel>", "message": "<message>", "rooms": "<room>...",
             "include_mine": "--include-mine", "all_rooms": "--all",
             "catch_up": "--catch-up"}.get(dest, "--" + dest)
    assert rc == 1
    assert len(err) == 1, err
    assert f"{label} is not used by '{action}'" in err[0]
    assert calls == []


@pytest.mark.parametrize("action,dest", _pairs(used=True))
def test_used_input_is_not_refused(action, dest, calls, capsys):
    _call(_argv(action, dest))  # a usage error for a missing input is fine here
    assert "is not used by" not in capsys.readouterr().err


def test_chat_honours_name(calls):
    assert _call(["chat", "dev", "--name", "me"]) == 0
    assert calls == [("chat", {"channel": "dev", "name": "me", "poll": 1.0})]


# Real CLI, real exit codes.

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


def test_search_filters_by_its_channel(workdir):
    project, home = workdir
    assert _run(project, home, "post", "dev", "needle in dev").returncode == 0
    assert _run(project, home, "post", "ops", "needle in ops").returncode == 0
    r = _run(project, home, "search", "ops", "needle")
    assert r.returncode == 0, r.stderr
    assert "needle in ops" in r.stdout
    assert "needle in dev" not in r.stdout


def test_broadcast_documented_form_works(workdir):
    project, home = workdir
    assert _run(project, home, "join", "dev").returncode == 0
    r = _run(project, home, "broadcast", "all hands")
    assert r.returncode == 0, r.stderr
    assert "all hands" in (project / ".synapt/recall/channels/dev.jsonl").read_text()


def test_broadcast_with_a_channel_is_refused(workdir):
    project, home = workdir
    r = _run(project, home, "broadcast", "dev", "all hands")
    assert r.returncode == 1
    assert "<channel> is not used by 'broadcast'" in r.stderr


def test_who_with_a_channel_is_refused(workdir):
    project, home = workdir
    r = _run(project, home, "who", "dev")
    assert r.returncode == 1
    assert "<channel> is not used by 'who'" in r.stderr
