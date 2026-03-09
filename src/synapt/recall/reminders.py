"""Session reminders — lightweight nudges surfaced at session start.

Storage: .synapt/recall/reminders.json (small mutable JSON array).
Reminders auto-clear after being shown once unless marked sticky.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from synapt.recall.core import atomic_json_write, project_index_dir


def _reminders_path(project_dir: Path | None = None) -> Path:
    """Return path to reminders.json in the project's .synapt/recall/ dir."""
    index_dir = project_index_dir(project_dir)
    return index_dir.parent / "reminders.json"


@dataclass
class Reminder:
    """A single reminder item."""

    id: str
    text: str
    created_at: str  # ISO 8601
    sticky: bool = False
    shown_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Reminder:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_reminders(path: Path | None = None) -> list[Reminder]:
    """Load all reminders from disk.

    Skips individual malformed entries rather than discarding the entire list.
    """
    path = path or _reminders_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    reminders = []
    for item in data:
        try:
            reminders.append(Reminder.from_dict(item))
        except (TypeError, KeyError):
            continue
    return reminders


def save_reminders(reminders: list[Reminder], path: Path | None = None) -> None:
    """Atomically save reminders to disk."""
    path = path or _reminders_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write([r.to_dict() for r in reminders], path)


def add_reminder(text: str, sticky: bool = False, path: Path | None = None) -> Reminder:
    """Add a new reminder and persist to disk."""
    path = path or _reminders_path()
    reminders = load_reminders(path)
    reminder = Reminder(
        id=uuid.uuid4().hex[:8],
        text=text,
        created_at=datetime.now(timezone.utc).isoformat(),
        sticky=sticky,
    )
    reminders.append(reminder)
    save_reminders(reminders, path)
    return reminder


def clear_reminder(reminder_id: str | None = None, path: Path | None = None) -> int:
    """Clear a specific reminder by ID, or all reminders if ID is None.

    Returns the number of reminders removed.
    """
    path = path or _reminders_path()
    reminders = load_reminders(path)
    if not reminders:
        return 0
    if reminder_id is None:
        count = len(reminders)
        save_reminders([], path)
        return count
    before = len(reminders)
    reminders = [r for r in reminders if r.id != reminder_id]
    save_reminders(reminders, path)
    return before - len(reminders)


def get_pending(path: Path | None = None) -> list[Reminder]:
    """Get reminders that should be shown at session start.

    Non-sticky reminders: shown_count < 1 (show once, then auto-clear).
    Sticky reminders: always returned.
    """
    path = path or _reminders_path()
    reminders = load_reminders(path)
    return [r for r in reminders if r.sticky or r.shown_count < 1]


def mark_shown(reminder_ids: list[str] | None = None, path: Path | None = None) -> None:
    """Increment shown_count for given reminders and remove expired non-sticky ones.

    If reminder_ids is None, marks all pending reminders as shown.
    """
    path = path or _reminders_path()
    reminders = load_reminders(path)
    if not reminders:
        return

    target_ids = set(reminder_ids) if reminder_ids else {r.id for r in reminders}
    for r in reminders:
        if r.id in target_ids:
            r.shown_count += 1

    # Remove non-sticky reminders that have been shown
    reminders = [r for r in reminders if r.sticky or r.shown_count < 1]
    save_reminders(reminders, path)


def pop_pending(path: Path | None = None) -> list[Reminder]:
    """Get pending reminders and mark them shown in a single load-save cycle.

    Returns the pending reminders. Non-sticky ones are removed from disk;
    sticky ones have their shown_count incremented.
    """
    path = path or _reminders_path()
    reminders = load_reminders(path)
    if not reminders:
        return []
    pending = [r for r in reminders if r.sticky or r.shown_count < 1]
    if not pending:
        return []
    # Mark shown
    pending_ids = {r.id for r in pending}
    for r in reminders:
        if r.id in pending_ids:
            r.shown_count += 1
    # Remove expired non-sticky
    reminders = [r for r in reminders if r.sticky or r.shown_count < 1]
    save_reminders(reminders, path)
    return pending


def format_for_session_start(reminders: list[Reminder]) -> str:
    """Format pending reminders for hook output at session start."""
    if not reminders:
        return ""
    lines = ["Reminders:"]
    for r in reminders:
        prefix = "[sticky] " if r.sticky else ""
        lines.append(f"  - {prefix}{r.text} (id: {r.id})")
    return "\n".join(lines)
