"""Session journal — structured session logging with auto-extraction.

Storage: .synapt/recall/worktrees/<name>/journal.jsonl (per-worktree, append-only).
Each entry records what was done, key decisions, and next steps.
"""

from __future__ import annotations

import fcntl
import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from synapt.recall.core import project_worktree_dir, project_index_dir, project_transcript_dir


def _journal_path(project_dir: Path | None = None) -> Path:
    """Return path to journal.jsonl for this worktree.

    Journal is per-worktree — each worktree tracks its own session history.
    Lives at ``<main>/.synapt/recall/worktrees/<name>/journal.jsonl``.
    """
    return project_worktree_dir(project_dir) / "journal.jsonl"


@dataclass
class JournalEntry:
    """One session's journal entry."""

    timestamp: str  # ISO 8601
    session_id: str = ""
    branch: str = ""
    focus: str = ""  # What this session was about
    done: list[str] = field(default_factory=list)  # What got accomplished
    decisions: list[str] = field(default_factory=list)  # Key decisions made
    next_steps: list[str] = field(default_factory=list)  # What to do next session
    files_modified: list[str] = field(default_factory=list)
    git_log: list[str] = field(default_factory=list)  # Recent commits
    auto: bool = False       # True if synthesized at build time
    enriched: bool = False   # True if LLM-enriched

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> JournalEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def has_content(self) -> bool:
        """True if the entry has any user-provided or auto-extracted content."""
        return bool(
            self.focus or self.done or self.decisions or self.next_steps
            or self.files_modified
        )

    def has_rich_content(self) -> bool:
        """True if entry has semantic content beyond just file paths."""
        return bool(self.focus or self.done or self.decisions or self.next_steps)


def append_entry(entry: JournalEntry, path: Path | None = None) -> Path:
    """Append a journal entry to the JSONL file.

    Uses fcntl.flock for exclusive locking to prevent interleaved writes
    when multiple processes append concurrently (e.g., background enrich
    + SessionEnd hook).
    """
    path = path or _journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(entry.to_dict()) + "\n")
        f.flush()
        # flock released on close
    return path


def read_latest(path: Path | None = None, meaningful: bool = False) -> JournalEntry | None:
    """Read the most recent journal entry.

    If *meaningful* is True, skip auto-extracted entries that have no
    focus/done/decisions/next — these are noise from the SessionEnd hook.
    """
    if not meaningful:
        entries = read_entries(path, n=1)
        return entries[0] if entries else None
    # Already deduped+sorted newest-first, so first with rich content wins
    for entry in read_entries(path, n=50):
        if entry.focus or entry.done or entry.decisions or entry.next_steps:
            return entry
    return None


def read_entries(path: Path | None = None, n: int = 5) -> list[JournalEntry]:
    """Read the last N journal entries (most recent first).

    Deduplicates by session_id (keeps the richest entry per session)
    and sorts by timestamp descending. Does NOT assume the file is
    chronologically ordered.
    """
    path = path or _journal_path()
    if not path.exists():
        return []
    raw = _read_all_entries(path)
    deduped = _dedup_entries(raw)
    deduped.sort(key=lambda e: e.timestamp, reverse=True)
    return deduped[:n]


def _read_all_entries(path: Path) -> list[JournalEntry]:
    """Read every entry from a journal JSONL file."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(JournalEntry.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return entries


def _entry_richness(entry: JournalEntry) -> tuple:
    """Score an entry for dedup ranking.

    Returns a tuple that sorts higher for richer entries:
    (not auto, rich field count, timestamp).
    """
    rich_count = sum(bool(f) for f in (entry.focus, entry.done, entry.decisions, entry.next_steps))
    return (not entry.auto, rich_count, entry.timestamp)


def _dedup_entries(entries: list[JournalEntry]) -> list[JournalEntry]:
    """Keep only the richest entry per session_id.

    Priority: non-auto > auto, then most rich fields
    (focus/done/decisions/next_steps), then newest timestamp.
    Entries without a session_id are kept as-is.
    """
    best: dict[str, JournalEntry] = {}
    no_sid: list[JournalEntry] = []
    for entry in entries:
        if not entry.session_id:
            no_sid.append(entry)
            continue
        existing = best.get(entry.session_id)
        if existing is None or _entry_richness(entry) > _entry_richness(existing):
            best[entry.session_id] = entry
    return list(best.values()) + no_sid


def compact_journal(path: Path | None = None) -> int:
    """Physically dedup and sort journal.jsonl.

    Reads all entries, deduplicates by session_id (keeps richest),
    sorts chronologically, and rewrites the file in-place.

    Uses an exclusive flock on the journal file itself (not a temp file)
    so concurrent append_entry calls block for the entire read-dedup-rewrite
    cycle and cannot sneak in entries that would be silently dropped.

    Note: entries with session_id="" are kept as-is and are never
    deduplicated against each other — only entries sharing a non-empty
    session_id are collapsed.

    Returns the number of duplicate entries removed.
    """
    path = path or _journal_path()
    if not path.exists():
        return 0
    # Open in "r+" (read-write, no truncate) so we can hold LOCK_EX on the
    # exact file descriptor we will rewrite.  All concurrent append_entry
    # callers flock the same path and will block until we close this fd.
    with open(path, "r+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        # Check for empty file AFTER acquiring the lock.  stat() before open()
        # is a TOCTOU race: a concurrent append_entry could write between the
        # stat() and the flock(), causing us to miss newly written entries.
        # seek(0, 2) in text mode is CPython/POSIX-specific (delegates to the
        # underlying binary fd's seek, well-defined on POSIX).  Not guaranteed
        # by the Python language spec but reliable on all supported platforms
        # (macOS/Linux, CPython ≥ 3.9).
        if f.seek(0, 2) == 0:  # seek to EOF; position 0 means file is empty
            return 0
        f.seek(0)  # reset to start for reading
        entries = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(JournalEntry.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
        deduped = _dedup_entries(entries)
        removed = len(entries) - len(deduped)
        if removed == 0:
            return 0
        deduped.sort(key=lambda e: e.timestamp)  # chronological for storage
        f.seek(0)
        f.truncate(0)  # explicit arg: truncate to zero bytes regardless of buffer position
        for entry in deduped:
            f.write(json.dumps(entry.to_dict()) + "\n")
        f.flush()
    return removed


def format_for_session_start(entry: JournalEntry) -> str:
    """Format a journal entry's key info for hook output at session start."""
    lines = []
    ts = entry.timestamp[:16]  # Trim to minute precision

    if entry.focus:
        lines.append(f"Last session ({ts}): {entry.focus}")

    if entry.done:
        lines.append("Done:")
        for item in entry.done:
            lines.append(f"  - {item}")

    if entry.decisions:
        lines.append("Decisions:")
        for item in entry.decisions:
            lines.append(f"  - {item}")

    if entry.next_steps:
        lines.append("Next steps:")
        for item in entry.next_steps:
            lines.append(f"  - {item}")

    return "\n".join(lines) if lines else ""


def format_entry_full(entry: JournalEntry) -> str:
    """Format a journal entry for display (journal --show)."""
    lines = [f"## {entry.timestamp[:16]}"]
    if entry.branch:
        lines.append(f"**Branch:** {entry.branch}")
    if entry.focus:
        lines.append(f"**Focus:** {entry.focus}")
    if entry.done:
        lines.append("\n### Done")
        for item in entry.done:
            lines.append(f"- {item}")
    if entry.decisions:
        lines.append("\n### Decisions")
        for item in entry.decisions:
            lines.append(f"- {item}")
    if entry.next_steps:
        lines.append("\n### Next")
        for item in entry.next_steps:
            lines.append(f"- {item}")
    if entry.files_modified:
        lines.append(f"\n### Files ({len(entry.files_modified)})")
        for f in entry.files_modified[:15]:
            lines.append(f"- {f}")
        if len(entry.files_modified) > 15:
            lines.append(f"  ... and {len(entry.files_modified) - 15} more")
    if entry.git_log:
        lines.append("\n### Commits")
        for c in entry.git_log:
            lines.append(f"- {c}")
    return "\n".join(lines)


def extract_session_id(path: Path | str) -> str:
    """Read the sessionId from the first progress entry in a transcript file.

    Scans only as far as the first progress entry — much cheaper than a full
    parse for callers that just need to identify the session.  Returns an
    empty string when no progress entry is found or the file cannot be read.
    """
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("type") == "progress" and d.get("sessionId"):
                        return d["sessionId"]
                except (json.JSONDecodeError, ValueError):
                    continue
    except OSError:
        pass
    return ""


def _transcript_sort_key(f: Path) -> tuple:
    """Sort key for transcript JSONL files: (mtime, name) descending.

    Wraps stat() in a try/except to guard against a TOCTOU race where a file
    is deleted between glob() and stat() — which would raise FileNotFoundError
    inside sorted().  Deleted files fall to the end of the sort (mtime=0.0).
    The filename tiebreaker makes the sort deterministic when two files share
    the same 1-second mtime granularity.
    """
    try:
        return (f.stat().st_mtime, f.name)
    except OSError:
        return (0.0, f.name)


def latest_transcript_path(project: Path | None = None) -> str | None:
    """Find the most recently modified transcript file for the project."""
    project = (project or Path.cwd()).resolve()
    transcript_dir = project_transcript_dir(project)
    if not transcript_dir:
        return None
    jsonl_files = sorted(
        transcript_dir.glob("*.jsonl"),
        key=_transcript_sort_key,
        reverse=True,
    )
    return str(jsonl_files[0]) if jsonl_files else None


def _get_branch(cwd: str) -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _get_recent_commits(cwd: str, n: int = 5) -> list[str]:
    """Get last N git commits as one-liners."""
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "log", "--oneline", f"-{n}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip().split("\n") if result.stdout.strip() else []
    except Exception:
        return []


def auto_extract_entry(
    transcript_path: str | Path | None = None,
    cwd: str | None = None,
) -> JournalEntry:
    """Build a journal entry by extracting context from the environment.

    Populates branch, files_modified, and git_log automatically.
    The focus/done/decisions/next fields are left empty for the model
    (or user) to fill via CLI flags or MCP tool.
    """
    cwd = cwd or str(Path.cwd())
    now = datetime.now(timezone.utc)

    session_id = ""
    files_modified: list[str] = []

    # Parse transcript if available
    if transcript_path and Path(transcript_path).exists():
        files_set: set[str] = set()
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if d.get("type") == "progress" and not session_id:
                    session_id = d.get("sessionId", "")
                if d.get("type") == "assistant":
                    for block in d.get("message", {}).get("content", []) or []:
                        if not isinstance(block, dict):
                            continue
                        fp = block.get("input", {}).get("file_path", "") if isinstance(block.get("input"), dict) else ""
                        if fp and "/.claude/" not in fp and not fp.startswith("/private/tmp"):
                            clean = fp.replace(cwd + "/", "").replace(cwd, "")
                            files_set.add(clean)
        files_modified = sorted(files_set)

    return JournalEntry(
        timestamp=now.isoformat(),
        session_id=session_id,
        branch=_get_branch(cwd),
        files_modified=files_modified,
        git_log=_get_recent_commits(cwd),
        auto=True,  # Auto-extracted; cleared if user adds rich content
    )


def _read_all_session_ids(journal_path: Path) -> set[str]:
    """Read all session_ids from journal.jsonl."""
    ids: set[str] = set()
    if not journal_path.exists():
        return ids
    with open(journal_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                sid = d.get("session_id", "")
                if sid:
                    ids.add(sid)
            except (json.JSONDecodeError, TypeError):
                continue
    return ids


def synthesize_journal_stubs(
    sessions: dict,
    journal_path: Path | None = None,
) -> int:
    """Synthesize lightweight journal entries for sessions without one.

    Scans the session grouping for sessions that have no corresponding
    journal entry. For each, creates a stub with:
      - focus: first user message (turn 0)
      - files_modified: union of files_touched across all turns
      - auto=True tag for later LLM enrichment

    Args:
        sessions: dict mapping session_id → list of TranscriptChunks.
        journal_path: Path to journal.jsonl. Default: project journal path.

    Returns:
        Number of stubs synthesized.
    """
    from synapt.recall.scrub import strip_system_artifacts

    journal_path = journal_path or _journal_path()
    existing_ids = _read_all_session_ids(journal_path)

    count = 0
    for session_id, chunks in sorted(sessions.items()):
        if session_id in existing_ids:
            continue

        # Only consider transcript chunks (turn_index >= 0)
        transcript_chunks = [c for c in chunks if c.turn_index >= 0]
        if not transcript_chunks:
            continue

        # Find turn 0 for focus (strip system artifacts for already-indexed data)
        sorted_chunks = sorted(transcript_chunks, key=lambda c: c.turn_index)
        focus = ""
        for chunk in sorted_chunks:
            msg = strip_system_artifacts(chunk.user_text.strip())
            if msg:
                focus = msg[:200]
                break

        # Collect all files touched
        files_set: set[str] = set()
        for c in transcript_chunks:
            files_set.update(c.files_touched)
        files_modified = sorted(files_set)

        # Use earliest timestamp
        timestamp = min(
            (c.timestamp for c in transcript_chunks if c.timestamp),
            default="",
        )
        if not timestamp:
            continue

        entry = JournalEntry(
            timestamp=timestamp,
            session_id=session_id,
            focus=focus,
            files_modified=files_modified,
            auto=True,
        )
        if entry.has_content():
            append_entry(entry, journal_path)
            count += 1

    return count
