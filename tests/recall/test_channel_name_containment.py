"""A room name cannot reach outside the channel store.

Post joined the raw room name onto the store directory, so a room named
``../x`` wrote ``x.jsonl`` one level above the store. After the join the
log path must resolve under the store, or the name is refused. NUL and a
backslash are refused too. Watch makes the same check, so post and watch
refuse the same names.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SYNAPT_")}
    env["HOME"] = str(home)
    return env


def _cli(project: Path, home: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "synapt.recall.cli", "channel", *argv],
        cwd=project, env=_env(home), capture_output=True, text=True, timeout=60,
    )


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch):
    # In-process rows resolve under a pytest-owned SYNAPT_RECALL_ROOT, the root
    # the store-isolation guard accepts; the CLI rows run in a subprocess with
    # SYNAPT_* stripped and use the project store.
    project = tmp_path / "proj"
    home = tmp_path / "home"
    project.mkdir()
    home.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    (tmp_path / "root").mkdir()
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(tmp_path / "root"))
    return project, home


def _store(project: Path) -> Path:
    from synapt.recall.channel import _channels_dir

    return _channels_dir()


def _listing(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def _jsonl_outside(root: Path, store: Path) -> list[Path]:
    return [p for p in root.rglob("*.jsonl") if not p.resolve().is_relative_to(store.resolve())]


def test_cli_post_dotdot_slash_is_refused_and_writes_nothing_outside(workdir, tmp_path):
    project, home = workdir
    assert _cli(project, home, "post", "dev", "hello").returncode == 0
    store = project / ".synapt" / "recall" / "channels"
    before = _listing(store.parent)
    r = _cli(project, home, "post", "../x", "escape")
    after = _listing(store.parent)
    assert r.returncode != 0, (r.stdout, r.stderr)
    assert "refusing room name" in r.stderr
    assert "Traceback" not in r.stderr
    assert before == after, (before, after)
    assert _jsonl_outside(tmp_path, store) == []


def test_mcp_post_dotdot_slash_is_an_error_result_and_writes_nothing_outside(workdir, tmp_path):
    project, _ = workdir
    from synapt.recall.server import recall_channel

    recall_channel(action="post", channel="dev", message="hello")
    store = _store(project)
    before = _listing(store.parent)
    out = recall_channel(action="post", channel="../x", message="escape")
    after = _listing(store.parent)
    assert "refusing room name" in out, out
    assert before == after, (before, after)
    assert _jsonl_outside(tmp_path, store) == []


@pytest.mark.parametrize("room, file", [("a..b", "a..b.jsonl"), (".", "..jsonl"), ("..", "...jsonl")])
def test_dotted_names_inside_the_store_still_post(workdir, room, file):
    project, _ = workdir
    from synapt.recall.channel import channel_post

    channel_post(room, "hello")
    assert (_store(project) / file).is_file()


@pytest.mark.parametrize("room", ["a\0b", "a\\b"])
def test_nul_and_backslash_are_refused(workdir, tmp_path, room):
    project, _ = workdir
    from synapt.recall.channel import ChannelNameError, channel_post

    with pytest.raises(ChannelNameError):
        channel_post(room, "hello")
    assert [p for p in tmp_path.rglob("*.jsonl")] == []


def test_legacy_colon_rename_does_not_move_a_file_outside_the_store(workdir):
    project, _ = workdir
    from synapt.recall.channel import ChannelNameError, channel_post

    channel_post("dev", "hello")
    store = _store(project)
    planted = store.parent / "w:z.jsonl"
    planted.write_text("not a channel log\n")
    with pytest.raises(ChannelNameError):
        channel_post("../w:z", "hello")
    assert planted.read_text() == "not a channel log\n"
    assert not (store.parent / "w--z.jsonl").exists()


def test_direct_messages_and_attachments_still_write_under_the_store(workdir, tmp_path):
    project, _ = workdir
    from synapt.recall.channel import channel_post
    from synapt.recall.direct import send_message

    att = tmp_path / "note.txt"
    att.write_text("attached\n")
    channel_post("dev", "with a file", attachment_paths=[str(att)])
    send_message(from_agent="a", to_agent="b", body="hi")
    store = _store(project)
    assert any(p.is_file() for p in (store / "attachments").rglob("*"))
    assert any(p.is_file() for p in (store / "direct").rglob("*"))


@pytest.mark.parametrize("room", ["a..b", ".", ".."])
def test_watch_accepts_what_post_accepts(room):
    from synapt.recall.channel import _check_watch_room

    assert _check_watch_room(room) == room


@pytest.mark.parametrize("room", ["../x", "a\0b", "a\\b"])
def test_watch_refuses_what_post_refuses(room):
    from synapt.recall.channel import WatchRoomError, _check_watch_room

    with pytest.raises(WatchRoomError):
        _check_watch_room(room)
