"""A memory-floor refusal of an automatic build leaves a trace a later reader can see.

Two kinds of row share one file: a refusal (no ``event`` key) and a ``built`` row
appended by every successful build. The count is refusals since the last ``built``
row, or since the CURRENT pointer's mtime when that is newer. A file's mtime is NOT
used as a build time: a monolithic store (``recall.db`` only; a fresh store opens
that way, measured) never publishes CURRENT, and its db file moves on any write, so
an mtime rule would drop the notice on the first knowledge save.

A ``built`` row also names its CALLER: argv, pid, ppid, and the parent's command line
from ``ps``. A build whose row is written without them cannot be attributed afterwards,
because the answer is only readable while the parent process still exists. A build whose
parent has already exited has been reparented, so its row records ppid 1 and a parent
command line naming the init process rather than the caller; pid and argv are still the
build's own. Because a row carries two unbounded command lines, the build's own argv and
its parent's, the file is written owner-only (0600), on creation and on any file an
earlier version left at the umask default.

The catchup and precompact builds are gated on host memory (6 GB free+inactive).
A refusal used to be printed to a stream its own spawn discarded, so "no build in
three days" was indistinguishable from "no build was ever tried". Each refusal is
now one line in ``build-deferrals.jsonl`` beside the index, and ``resume`` and the
session-start wake print one plain sentence when refusals have piled up since the
index was last built. The floor is unchanged; this only makes the refusal legible.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DEFERRALS_FILENAME = "build-deferrals.jsonl"
_KEEP_LINES = 200
# A ``built`` row now carries the caller's argv and its parent's command line, and a command
# line is an unbounded place for a secret. So this file is owner-only, not umask-default.
_FILE_MODE = 0o600
_PS_TIMEOUT_S = 2.0
_MAX_CMD_CHARS = 1000


def _append(data_dir: Path, row: dict) -> None:
    """Append one row and trim to the last 200, keeping the newest ``built`` row. Never raises."""
    try:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / DEFERRALS_FILENAME
        if not path.exists():
            # Create owner-only from the FIRST byte. A chmod-after-open leaves a window in
            # which a row that carries a command line is world-readable, and umask can only
            # clear bits, so 0o600 is an upper bound on what this grants.
            try:
                os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _FILE_MODE))
            except OSError:
                pass
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        # And on an EXISTING file: a ledger an earlier version created sits at the umask
        # default, so the mode is asserted every write rather than only at creation.
        try:
            os.chmod(path, _FILE_MODE)
        except OSError:
            pass
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > _KEEP_LINES:
            kept = lines[-_KEEP_LINES:]
            if not any('"event": "built"' in ln for ln in kept):
                older = [ln for ln in lines[:-_KEEP_LINES] if '"event": "built"' in ln]
                if older:
                    kept = [older[-1]] + kept[1:]
            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 -- best effort by design
        pass


def record_deferral(data_dir: Path, site: str, numbers: str) -> None:
    """Append one refusal. Never raises: the trace must not break the hook it records."""
    _append(data_dir, {"ts": time.time(), "site": site, "verdict": "refuse", "numbers": numbers})


def _parent_command_line(ppid: int) -> str | None:
    """The parent's command line, read from ``ps``. None when it cannot be read.

    This is the one thing a ``built`` row could not previously name. A build that ran at
    03:33 left a row saying only that a build happened, so "who asked for it" died with the
    process: the row cannot be recovered afterwards because the answer is only readable
    while the parent still exists. ``ps -o command=`` (not ``comm``, which is argv[0] alone)
    prints the full argv, which is what distinguishes a spawned catchup from an MCP-server
    rebuild. Every failure -- no ps, a dead or reparented parent, a timeout -- degrades to
    None. Never raises: a caller read must not be able to break the build it describes.
    """
    if ppid <= 0:
        return None
    try:
        # -ww is load-bearing, not style. procps (Linux) truncates `ps -o command=` to the
        # terminal width, and a build spawned by a hook has no terminal, so the recorded
        # parent command line would be silently cut at about 80 columns -- the same
        # "answered a different question" defect this row exists to end. BSD/macOS ps does
        # not truncate, which is exactly why the cut was invisible on a developer machine.
        out = subprocess.run(
            ["ps", "-ww", "-o", "command=", "-p", str(ppid)],
            capture_output=True, text=True, timeout=_PS_TIMEOUT_S,
        )
    except Exception:  # noqa: BLE001 -- best effort by design
        return None
    line = (out.stdout or "").strip()
    return line[:_MAX_CMD_CHARS] if line else None


def caller_facts() -> dict:
    """Who started this build: argv, pid, ppid, and the parent's own command line.

    Read at WRITE time, in the process doing the build, because that is the only moment the
    answer exists. ``argv`` separates an MCP-server build from ``synapt build`` from a
    spawned catchup; ``ppid_cmd`` names the process that actually asked. Never raises.
    """
    try:
        ppid = os.getppid()
        return {
            "argv": list(sys.argv),
            "pid": os.getpid(),
            "ppid": ppid,
            "ppid_cmd": _parent_command_line(ppid),
        }
    except Exception:  # noqa: BLE001 -- a caller read must never break a build
        return {}


def record_build(data_dir: Path, caller: dict | None = None) -> None:
    """Append a ``built`` row, naming the caller who started the build.

    *caller* defaults to :func:`caller_facts`, read here rather than at the call site so
    every present and future caller of this function records one, and a build path added
    later cannot forget to.
    """
    row = {"ts": time.time(), "event": "built"}
    row.update(caller_facts() if caller is None else caller)
    _append(data_dir, row)


def _current_mtime(index_dir: Path) -> float | None:
    try:
        current = Path(index_dir) / "CURRENT"
        return current.stat().st_mtime if current.exists() else None
    except OSError:
        return None


def deferral_notice(data_dir: Path, index_dir: Path) -> str | None:
    """One sentence when automatic builds were refused since the index was last built."""
    try:
        path = Path(data_dir) / DEFERRALS_FILENAME
        if not path.exists():
            return None
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        stamps = [float(r.get("ts", 0)) for r in rows if r.get("event") == "built"]
        current = _current_mtime(index_dir)
        if current is not None:
            stamps.append(current)
        built = max(stamps) if stamps else None
        n = sum(1 for r in rows if r.get("event") != "built" and (built is None or float(r.get("ts", 0)) > built))
        if n == 0:
            return None
        when = datetime.fromtimestamp(built).strftime("%Y-%m-%d") if built else "(date not recorded)"
        times = "time" if n == 1 else "times"
        return (f"index last built {when}; automatic build deferred {n} {times} "
                "(memory floor); run synapt build")
    except Exception:  # noqa: BLE001 -- a notice must never break resume or the wake
        return None
