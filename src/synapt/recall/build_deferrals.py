"""A memory-floor refusal of an automatic build leaves a trace a later reader can see.

Two kinds of row share one file: a refusal (no ``event`` key) and a ``built`` row
appended by every successful build. The count is refusals since the last ``built``
row, or since the CURRENT pointer's mtime when that is newer. A file's mtime is NOT
used as a build time: a monolithic store (``recall.db`` only; a fresh store opens
that way, measured) never publishes CURRENT, and its db file moves on any write, so
an mtime rule would drop the notice on the first knowledge save.

The catchup and precompact builds are gated on host memory (6 GB free+inactive).
A refusal used to be printed to a stream its own spawn discarded, so "no build in
three days" was indistinguishable from "no build was ever tried". Each refusal is
now one line in ``build-deferrals.jsonl`` beside the index, and ``resume`` and the
session-start wake print one plain sentence when refusals have piled up since the
index was last built. The floor is unchanged; this only makes the refusal legible.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

DEFERRALS_FILENAME = "build-deferrals.jsonl"
_KEEP_LINES = 200


def _append(data_dir: Path, row: dict) -> None:
    """Append one row and trim to the last 200, keeping the newest ``built`` row. Never raises."""
    try:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / DEFERRALS_FILENAME
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
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


def record_build(data_dir: Path) -> None:
    """Append a ``built`` row: a successful build ends the run of refusals before it."""
    _append(data_dir, {"ts": time.time(), "event": "built"})


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
