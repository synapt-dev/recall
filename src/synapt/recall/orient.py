"""`synapt orient`: a wake card from the user's own local record.

A thin on-ramp, not a policy engine: it reads the journal recall already keeps and the
user's own feedback memories, and prints five fields a waking agent needs before it acts.

- GOAL: the newest meaningful journal entry's focus, with its age.
- NEXT: that entry's first next step, the command to run first.
- CARRIED: next steps still open across recent entries.
- RULED OUT: that entry's decisions, so a settled question is not reopened.
- CONSTRAINTS: the user's feedback memories (name and description).

STALE says when the journal is older than the newest session transcript, which means the
last session ended without writing it. Every source fails soft: a missing journal or memory
directory becomes one named line, never an exception. The card is kept under a token budget
by shortening the longest lists first, and it says what it left out.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

TOKEN_BUDGET = 4000
CHARS_PER_TOKEN = 4          # a rough estimate, the same ratio the card reports
CARRIED_MAX = 8              # the newest open carried steps; the rest counted as withheld
CONSTRAINTS_MAX = 8          # the newest feedback memories; the rest are counted as withheld, never dropped silently
STALE_AFTER_S = 3600         # transcripts newer than the journal by more than this are a gap
LIVE_WITHIN_S = 120          # a transcript written this recently is the running session, not a finished one


@dataclass
class Card:
    goal: str = ""
    goal_age_s: float | None = None
    next: str = ""
    carried: list[str] = field(default_factory=list)
    ruled_out: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    stale: str = ""
    missing: list[str] = field(default_factory=list)   # "<source>: <why>", one per source that did not read
    withheld: dict[str, int] = field(default_factory=dict)

    def filled(self) -> int:
        return sum(bool(x) for x in (self.goal, self.next, self.carried, self.ruled_out, self.constraints))

    def to_dict(self) -> dict:
        return {
            "goal": self.goal, "goal_age_s": self.goal_age_s, "next": self.next, "carried": self.carried,
            "ruled_out": self.ruled_out, "constraints": self.constraints, "stale": self.stale,
            "missing": self.missing, "withheld": self.withheld, "fields_filled": self.filled(),
        }


def _epoch(ts: str) -> float | None:
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def _age(seconds: float | None) -> str:
    if seconds is None:
        return "age unknown"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h ago"
    return f"{seconds / 86400:.1f} days ago"


def _read_journal(card: Card, project: Path | None, now: float) -> float | None:
    """Fill goal, next, ruled out and carried; return the newest entry's epoch."""
    from synapt.recall.journal import (
        _journal_path,
        is_withheld_marker,
        pending_next_steps,
        read_latest,
        strip_carry_stamp,
    )

    try:
        path = _journal_path(project)
    except Exception as exc:   # an unresolvable project is a missing source, not a crash
        card.missing.append(f"journal: cannot resolve its path ({exc})")
        return None
    if not path.exists():
        card.missing.append(f"journal: none at {path}; write one with `synapt recall journal --write`")
        return None
    entry = read_latest(path, meaningful=True)
    if entry is None:
        card.missing.append(f"journal: {path} has no entry with content")
        return None
    when = _epoch(entry.timestamp)
    card.goal = entry.focus.strip()
    card.goal_age_s = None if when is None else max(0.0, now - when)
    steps = [s for s in entry.next_steps if s.strip()]
    card.next = steps[0].strip() if steps else ""
    card.ruled_out = [d.strip() for d in entry.decisions if d.strip()]
    own = {s.strip() for s in steps}
    card.carried = [strip_carry_stamp(s)[0].strip() for s in pending_next_steps(path) if not is_withheld_marker(s)]
    card.carried = [s for s in card.carried if s and s not in own and s != card.next]
    if len(card.carried) > CARRIED_MAX:
        card.withheld["carried"] = len(card.carried) - CARRIED_MAX
        card.carried = card.carried[:CARRIED_MAX]
    return when


def _frontmatter(text: str) -> dict[str, str]:
    """name, description and type from a memory file's frontmatter (flat or under metadata:)."""
    if not text.startswith("---"):
        return {}
    head = text[3:].split("\n---", 1)[0]
    out: dict[str, str] = {}
    for line in head.splitlines():
        key, sep, value = line.strip().partition(":")
        if sep and key in ("name", "description", "type") and value.strip():
            out.setdefault(key, value.strip().strip('"').replace('\\"', '"'))
    return out


def _read_constraints(card: Card, project: Path | None) -> None:
    from synapt.recall.claude_memory_source import _claude_memory_dir

    # The memory directory Claude Code writes for THIS session is keyed by the session's own project (cwd),
    # never by GRIPSPACE_ROOT: a shared root would read another project's memories as this one's constraints.
    try:
        mem = _claude_memory_dir(project or Path.cwd())
    except Exception as exc:
        card.missing.append(f"constraints: cannot resolve the memory directory ({exc})")
        return
    if not mem.is_dir():
        card.missing.append(f"constraints: no memory directory at {mem}")
        return
    files = sorted((p for p in mem.glob("*.md") if p.name != "MEMORY.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files:
        try:
            fm = _frontmatter(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if fm.get("type") != "feedback":
            continue
        name = fm.get("name") or p.stem
        card.constraints.append(f"{name}: {fm['description']}" if fm.get("description") else name)
    if not card.constraints:
        card.missing.append(f"constraints: no feedback memory in {mem}")
    elif len(card.constraints) > CONSTRAINTS_MAX:
        card.withheld["constraints"] = len(card.constraints) - CONSTRAINTS_MAX
        card.constraints = card.constraints[:CONSTRAINTS_MAX]


def _read_stale(card: Card, project: Path | None, journal_epoch: float | None, now: float) -> None:
    # At wake the newest transcript is the running session's own, written seconds ago; comparing the
    # journal against it fires STALE on every wake an hour after a clean close. Skip what is live.
    from synapt.recall.journal import transcripts_newest_first

    if journal_epoch is None:
        return
    try:
        candidates = transcripts_newest_first(project)
    except Exception:
        return
    for path in candidates:
        try:
            activity = Path(path).stat().st_mtime
        except OSError:
            continue
        if now - activity < LIVE_WITHIN_S:
            continue
        gap = activity - journal_epoch
        if gap > STALE_AFTER_S:
            card.stale = (f"the journal is {_age(gap).replace(' ago', '')} older than your newest finished session; "
                          "the last session may have ended without writing it. Check that session before trusting GOAL.")
        return


def build_card(project: Path | None = None, now: float | None = None) -> Card:
    # No --project means the SAME resolution `synapt journal` uses (GRIPSPACE_ROOT, a marker, then cwd);
    # passing cwd here would make it a deliberate root and read a different store than the one the agent writes.
    project = Path(project) if project else None
    now = time.time() if now is None else now
    card = Card()
    when = _read_journal(card, project, now)
    _read_constraints(card, project)
    _read_stale(card, project, when, now)
    return card


def _clip(text: str, n: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _render(card: Card) -> str:
    lines = ["GET ORIENTED (synapt orient: from your own record)", ""]
    if card.stale:
        lines += [f"STALE: {card.stale}", ""]
    lines.append(f"GOAL ({_age(card.goal_age_s)}): {_clip(card.goal) if card.goal else '(none recorded)'}")
    lines.append(f"NEXT: {_clip(card.next) if card.next else '(none recorded)'}")
    for title, key, items in (("CARRIED", "carried", card.carried), ("RULED OUT", "ruled_out", card.ruled_out),
                              ("CONSTRAINTS", "constraints", card.constraints)):
        lines.append(f"{title}:" + ("" if items else " (none)"))
        lines += [f"  - {_clip(i)}" for i in items]
        if card.withheld.get(key):
            lines.append(f"  ({card.withheld[key]} more withheld; the newest are shown)")
    if card.missing:
        lines += ["", "NOT READ:"] + [f"  - {m}" for m in card.missing]
    lines += ["", f"[{card.filled()} of 5 fields; ~{{tokens}} tokens]"]
    return "\n".join(lines)


def format_card(card: Card, budget: int = TOKEN_BUDGET) -> str:
    """The printed card, shortened to the budget: the longest list loses its oldest items first."""
    while True:
        text = _render(card)
        tokens = len(text) // CHARS_PER_TOKEN
        if tokens <= budget:
            return text.replace("{tokens}", str(tokens))
        key = max(("constraints", "carried", "ruled_out"), key=lambda k: len(getattr(card, k)))
        items = getattr(card, key)
        if not items:
            return text.replace("{tokens}", str(tokens))
        items.pop()
        card.withheld[key] = card.withheld.get(key, 0) + 1


def format_json(card: Card, budget: int = TOKEN_BUDGET) -> str:
    text = format_card(card, budget)
    return json.dumps({**card.to_dict(), "tokens_est": len(text) // CHARS_PER_TOKEN})
