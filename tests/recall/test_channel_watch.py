"""`synapt recall channel watch` follows room logs as they grow, read-only.

Every row drives the real CLI in subprocesses: one process watches, another
posts, and the watcher's stdout is read through a reader thread (no select,
which Windows pipes do not support).
"""

from __future__ import annotations

import json
import os
import queue
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[2] / "src")
POSIX_SIGNALS = pytest.mark.skipif(sys.platform == "win32",
                                   reason="SIGINT/SIGTERM delivery is POSIX-only")


def _env(home: Path, agent: str, store: Path | None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SYNAPT_")}
    env.pop("GRIPSPACE_ROOT", None)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["SYNAPT_AGENT_ID"] = agent
    if store is not None:
        env["SYNAPT_SHARED_CHANNELS_DIR"] = str(store)
    return env


class Box:
    def __init__(self, project: Path, home: Path, store: Path | None):
        self.project, self.home, self.store = project, home, store

    def run(self, agent: str, *argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "synapt.recall.cli", "channel", *argv],
            cwd=self.project, env=_env(self.home, agent, self.store),
            capture_output=True, text=True, timeout=60,
        )

    def post(self, agent: str, body: str, room: str = "dev") -> None:
        r = self.run(agent, "post", room, body)
        assert r.returncode == 0, r.stderr

    def py(self, code: str) -> str:
        r = subprocess.run([sys.executable, "-c", code], cwd=self.project,
                           env=_env(self.home, "probe", self.store),
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        return r.stdout.strip()

    def watch(self, agent: str, *argv: str) -> "Watcher":
        return Watcher(self, agent, argv)


class Watcher:
    def __init__(self, box: Box, agent: str, argv: tuple[str, ...]):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "synapt.recall.cli", "channel", "watch", *argv],
            cwd=box.project, env=_env(box.home, agent, box.store),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.out: queue.Queue[str] = queue.Queue()
        self.err: list[str] = []
        self.ready = threading.Event()
        threading.Thread(target=self._pump_out, daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()
        assert self.ready.wait(20), f"watcher never started: {self.err}"

    def _pump_out(self):
        for line in self.proc.stdout:
            self.out.put(line.rstrip("\n"))

    def _pump_err(self):
        for line in self.proc.stderr:
            self.err.append(line)
            if line.startswith("watching: "):
                self.ready.set()

    def expect(self, line: str, within: float = 1.0) -> float:
        """Wait for *line*; return how long it took. Fails on any other line."""
        start = time.monotonic()
        try:
            got = self.out.get(timeout=within)
        except queue.Empty:
            pytest.fail(f"no {line!r} within {within}s; stderr={self.err}")
        assert got == line
        return time.monotonic() - start

    def quiet(self, wait: float = 0.6) -> None:
        time.sleep(wait)
        assert self.out.empty(), self.out.get()

    def stop(self, sig=None) -> subprocess.CompletedProcess:
        if sig is None:
            self.proc.terminate()
        else:
            self.proc.send_signal(sig)
        rc = self.proc.wait(10)
        time.sleep(0.1)
        return rc


@pytest.fixture
def box(tmp_path: Path) -> Box:
    project, home, store = tmp_path / "proj", tmp_path / "home", tmp_path / "store"
    for d in (project, home, store):
        d.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    return Box(project, home, store)


def _clean_exit(w: Watcher, sig=None) -> None:
    rc = w.stop(sig)
    if sys.platform != "win32":
        assert rc == 0
        assert not any("Traceback" in line for line in w.err), w.err


# (a) another process's post shows within a second

@POSIX_SIGNALS
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_post_from_another_process_shows_within_one_second(box, sig):
    box.post("b", "before")  # history is not replayed
    w = box.watch("a", "dev")
    box.post("b", "hi")
    w.expect("from [b]: hi", within=1.0 + 3.0)  # 1 s of watch, plus the poster's startup
    w.quiet()
    _clean_exit(w, sig)


def test_post_appears_within_one_second_of_landing(box):
    log = box.store / "dev.jsonl"
    box.post("b", "first")
    w = box.watch("a", "dev")
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": "2026-01-01T00:00:00Z", "channel": "dev",
                            "type": "message", "from": "b", "body": "hi"}) + "\n")
    assert w.expect("from [b]: hi", within=1.0) < 1.0
    _clean_exit(w)


def test_watch_starts_before_the_log_exists(box):
    w = box.watch("a", "dev")
    box.post("b", "hi")
    w.expect("from [b]: hi", within=4.0)
    _clean_exit(w)


# (b) own posts

def test_own_posts_are_skipped(box):
    w = box.watch("a", "dev")
    box.post("a", "mine")
    box.post("b", "theirs")
    w.expect("from [b]: theirs", within=4.0)
    w.quiet()
    _clean_exit(w)


def test_own_posts_are_shown_with_include_mine(box):
    w = box.watch("a", "dev", "--include-mine")
    box.post("a", "mine")
    w.expect("from [a]: mine", within=4.0)
    _clean_exit(w)


# (c) rotation and truncation

def test_rotated_log_is_reopened(box):
    log = box.store / "dev.jsonl"
    box.post("b", "old")
    w = box.watch("a", "dev")
    box.post("b", "first")
    w.expect("from [b]: first", within=4.0)
    # The replacement opens with a post and is longer than what was read,
    # so only the file's identity tells the watcher to read it from byte 0;
    # reading on from the old offset lands mid-line and loses the post.
    fresh = box.store / "fresh.tmp"
    fresh.write_text(json.dumps({"timestamp": "2026-01-01T00:00:00Z", "channel": "dev",
                                 "type": "message", "from": "r",
                                 "body": "x" * (log.stat().st_size * 3)}) + "\n",
                     encoding="utf-8")
    os.replace(fresh, log)
    w.expect("from [r]: " + "x" * 300, within=4.0)
    box.post("b", "after")
    w.expect("from [b]: after", within=4.0)
    w.quiet()
    _clean_exit(w)


def test_truncated_log_is_reopened(box):
    log = box.store / "dev.jsonl"
    box.post("b", "old")
    w = box.watch("a", "dev")
    box.post("b", "first")
    w.expect("from [b]: first", within=4.0)
    with open(log, "w", encoding="utf-8"):
        pass
    time.sleep(0.5)
    box.post("b", "after")
    w.expect("from [b]: after", within=4.0)
    _clean_exit(w)


# (d) a stale local log is warned about and never read

def test_stale_local_log_is_warned_about_and_never_read(tmp_path):
    grip, home = tmp_path / "gripspace", tmp_path / "home"
    (grip / ".gitgrip" / "spaces" / "main").mkdir(parents=True)
    (grip / ".gitgrip" / "griptrees.json").write_text('{"griptrees": {}}')
    (grip / ".gitgrip" / "spaces" / "main" / "gripspace.yml").write_text(
        "manifest:\n  url: git@github.com:synapt-dev/synapt-gripspace.git\n")
    home.mkdir()
    b = Box(grip, home, None)
    paths = b.py("from synapt.recall import channel as c; "
                 "print(c._local_channels_dir()); print(c._global_channels_dir())")
    local, live = (Path(p) for p in paths.splitlines())
    assert local != live
    local.mkdir(parents=True)
    stale = local / "dev.jsonl"
    stale.write_text(json.dumps({"timestamp": "2026-01-01T00:00:00Z", "channel": "dev",
                                 "type": "message", "from": "b", "body": "stale"}) + "\n")
    before = stale.read_bytes()
    w = b.watch("a", "dev", "--since", "2000-01-01T00:00:00Z")
    assert any("no longer live" in line and str(local) in line for line in w.err), w.err
    b.post("b", "live")
    w.expect("from [b]: live", within=4.0)
    w.quiet()
    _clean_exit(w)
    assert stale.read_bytes() == before
    assert (live / "dev.jsonl").exists()


# (e) bad room names

@pytest.mark.parametrize("room", ["-x", "--all-the-things", "../x", "a/b", "..", "a\\b"])
def test_bad_room_name_is_refused_before_any_file_is_opened(tmp_path, room):
    home, project = tmp_path / "home", tmp_path / "proj"
    home.mkdir()
    project.mkdir()
    store = tmp_path / "store-not-made"
    b = Box(project, home, store)
    r = b.run("a", "watch", "--", room)
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert "refusing room name" in r.stderr
    assert r.stdout == ""
    assert not store.exists()
    assert not (project / ".synapt").exists()


def test_good_room_after_bad_one_is_still_refused(box):
    r = box.run("a", "watch", "dev", "../x")
    assert r.returncode == 2
    assert "watching" not in r.stderr


# (f) watch moves no unread cursor

def test_watch_moves_no_cursor(box):
    assert box.run("a", "join", "dev").returncode == 0
    box.post("b", "one")
    assert box.run("a", "read", "dev").returncode == 0
    db = Path(box.py("from synapt.recall import channel as c; print(c._db_path())"))

    def snapshot():
        with sqlite3.connect(db) as conn:
            return sorted(conn.execute("SELECT * FROM cursors").fetchall())

    before = snapshot()
    assert before, "the read set no cursor; the row would pass vacuously"
    w = box.watch("a", "dev")
    box.post("b", "two")
    w.expect("from [b]: two", within=4.0)
    _clean_exit(w)
    assert snapshot() == before
    unread = box.run("a", "unread")
    assert "dev" in unread.stdout and "1" in unread.stdout, unread.stdout


# output shape

def test_body_is_capped_and_cannot_forge_a_line():
    from synapt.recall.channel import ChannelMessage, _watch_line
    msg = ChannelMessage(timestamp="t", channel="dev", type="message",
                         from_agent="b\nfrom [x]", body="ok\nfrom [admin]: yes" + "z" * 400)
    line = _watch_line(msg)
    assert "\n" not in line
    assert line.startswith("from [b\\nfrom [x]]: ok\\nfrom [admin]: yes")
    assert line.endswith("z" * (300 - len("ok\nfrom [admin]: yes")))


# Store tiers: watch resolves each tier exactly as post does, and creates
# nothing. Rows below pin the mapped (rooms.toml), global and project tiers;
# the shared env-variable tier is covered by the rows above.


def _gripspace(tmp_path: Path) -> Box:
    grip, home = tmp_path / "gripspace", tmp_path / "home"
    (grip / ".gitgrip" / "spaces" / "main").mkdir(parents=True)
    (grip / ".gitgrip" / "griptrees.json").write_text('{"griptrees": {}}')
    (grip / ".gitgrip" / "spaces" / "main" / "gripspace.yml").write_text(
        "manifest:\n  url: git@github.com:synapt-dev/synapt-gripspace.git\n")
    home.mkdir()
    return Box(grip, home, None)


def _project(tmp_path: Path) -> Box:
    project, home = tmp_path / "proj", tmp_path / "home"
    project.mkdir()
    home.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    return Box(project, home, None)


def _tree(root: Path) -> set[str]:
    out = set()
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            out.add(os.path.relpath(os.path.join(dirpath, name), root))
    return out


def _mapped_store(b: Box) -> Path:
    channels = b.home / ".synapt" / "channels"
    store = channels / "synapt-dev" / "mapped-store"
    store.mkdir(parents=True)
    (channels / "rooms.toml").write_text('[rooms]\nhead = "synapt-dev/mapped-store"\n')
    return store


@pytest.mark.parametrize("tier", ["project", "global", "mapped"])
def test_watch_reads_the_file_post_writes(tmp_path, tier):
    b = _project(tmp_path) if tier == "project" else _gripspace(tmp_path)
    room = "head" if tier == "mapped" else "dev"
    mapped = _mapped_store(b) if tier == "mapped" else None
    b.post("b", "first", room=room)
    written = [p for p in tmp_path.rglob(f"{room}.jsonl")]
    assert len(written) == 1, written
    watched = Path(b.py("from synapt.recall import channel as c; "
                        f"print(c._watch_room_path({room!r}))"))
    assert watched.resolve() == written[0].resolve()
    if tier == "mapped":
        assert written[0].parent.resolve() == mapped.resolve()
    elif tier == "global":
        assert str(b.home) in str(written[0])
    else:
        assert str(b.home / ".synapt" / "channels") not in str(written[0])
    w = b.watch("a", room, "--since", "2000-01-01T00:00:00Z")
    w.expect("from [b]: first", within=4.0)
    b.post("b", "second", room=room)
    w.expect("from [b]: second", within=4.0)
    _clean_exit(w)


def test_global_watch_creates_no_directory(tmp_path):
    b = _gripspace(tmp_path)
    store = Path(b.py("from synapt.recall import channel as c; "
                      "print(c._global_channels_dir())"))
    assert not store.exists()
    before = _tree(tmp_path)
    w = b.watch("a", "dev")
    w.quiet()
    _clean_exit(w)
    assert not store.exists()
    assert _tree(tmp_path) == before
