"""Agent channel -- cross-worktree communication via append-only JSONL files.

Storage layout (all under .synapt/recall/channels/):
  <name>.jsonl    -- append-only message log (the open protocol)
  channels.db     -- SQLite for state: presence, memberships, cursors, pins

Any process that can append/read files can participate -- no daemon required.
External agents that cannot use SQLite can still write JSONL lines directly;
the SQLite layer is used by agents that import this module.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

from synapt.recall.core import project_data_dir, _worktree_name


# ---------------------------------------------------------------------------
# Stale-detection thresholds
# ---------------------------------------------------------------------------

_ONLINE_MINUTES = 5       # < 5 min  => online
_IDLE_MINUTES = 30        # 5-30 min => idle
_AWAY_MINUTES = 120       # 30-120 min => away
                          # > 120 min => offline (auto-leave)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _channels_dir(project_dir: Path | None = None) -> Path:
    """Return the shared channels directory: <data>/.synapt/recall/channels/."""
    return project_data_dir(project_dir) / "channels"


def _channel_path(channel: str, project_dir: Path | None = None) -> Path:
    """Return the JSONL log path for a channel."""
    return _channels_dir(project_dir) / f"{channel}.jsonl"


def _db_path(project_dir: Path | None = None) -> Path:
    """Return the SQLite database path for channel state."""
    return _channels_dir(project_dir) / "channels.db"


def _presence_path(project_dir: Path | None = None) -> Path:
    """Return the legacy presence JSON file path (for migration)."""
    return _channels_dir(project_dir) / "_presence.json"


def _agent_name(agent_name: str | None = None, project_dir: Path | None = None) -> str:
    """Resolve agent name: explicit > worktree name."""
    return agent_name or _worktree_name(project_dir)


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string with microsecond precision.

    Microsecond precision ensures that operations within the same second
    produce distinct timestamps for correct cursor-based unread counting.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp to a timezone-aware datetime."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChannelMessage:
    """One message in a channel log."""

    timestamp: str
    channel: str
    type: str  # "message", "join", "leave"
    body: str = ""
    # Use field name "from_agent" in Python to avoid shadowing builtin "from".
    # Serialized as "from" in JSON for protocol compatibility.
    from_agent: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["from"] = d.pop("from_agent")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ChannelMessage:
        # Accept "from" key from JSON, map to from_agent
        mapped = dict(d)
        if "from" in mapped:
            mapped["from_agent"] = mapped.pop("from")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in mapped.items() if k in known})


# ---------------------------------------------------------------------------
# SQLite state layer
# ---------------------------------------------------------------------------

_CHANNEL_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS presence (
    agent TEXT PRIMARY KEY,
    status TEXT DEFAULT 'online',
    last_seen TEXT NOT NULL,
    joined_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memberships (
    agent TEXT NOT NULL,
    channel TEXT NOT NULL,
    joined_at TEXT NOT NULL,
    PRIMARY KEY (agent, channel)
);

CREATE TABLE IF NOT EXISTS cursors (
    agent TEXT NOT NULL,
    channel TEXT NOT NULL,
    last_read_at TEXT NOT NULL,
    PRIMARY KEY (agent, channel)
);

CREATE TABLE IF NOT EXISTS pins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    body TEXT NOT NULL,
    pinned_by TEXT NOT NULL,
    pinned_at TEXT NOT NULL
);
"""


def _open_db(project_dir: Path | None = None) -> sqlite3.Connection:
    """Open (or create) the channels SQLite database with WAL mode."""
    path = _db_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    conn.executescript(_CHANNEL_SCHEMA_SQL)
    # Migrate legacy _presence.json if it exists
    _migrate_legacy_presence(conn, project_dir)
    return conn


def _migrate_legacy_presence(conn: sqlite3.Connection, project_dir: Path | None = None) -> None:
    """Migrate data from legacy _presence.json to SQLite, then delete the file."""
    path = _presence_path(project_dir)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Corrupt file -- just remove it
        try:
            path.unlink()
        except OSError:
            pass
        return

    if not data:
        try:
            path.unlink()
        except OSError:
            pass
        return

    for agent, info in data.items():
        joined_at = info.get("joined_at", _now_iso())
        last_seen = info.get("last_seen", joined_at)
        channels = info.get("channels", [])

        # Insert into presence (skip if already exists)
        conn.execute(
            "INSERT OR IGNORE INTO presence (agent, status, last_seen, joined_at) "
            "VALUES (?, 'online', ?, ?)",
            (agent, last_seen, joined_at),
        )

        # Insert memberships
        for ch in channels:
            conn.execute(
                "INSERT OR IGNORE INTO memberships (agent, channel, joined_at) "
                "VALUES (?, ?, ?)",
                (agent, ch, joined_at),
            )

    conn.commit()

    # Remove legacy file
    try:
        path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Stale detection and auto-leave
# ---------------------------------------------------------------------------

def _agent_status(last_seen: str) -> str:
    """Determine agent status based on last_seen timestamp."""
    now = datetime.now(timezone.utc)
    try:
        last_dt = _parse_iso(last_seen)
        delta = now - last_dt
        minutes = delta.total_seconds() / 60
        if minutes < _ONLINE_MINUTES:
            return "online"
        if minutes < _IDLE_MINUTES:
            return "idle"
        if minutes < _AWAY_MINUTES:
            return "away"
        return "offline"
    except (ValueError, TypeError):
        return "online"


def _reap_stale_agents(conn: sqlite3.Connection, project_dir: Path | None = None) -> list[str]:
    """Find agents that are > 2 hours stale, auto-leave them.

    Returns list of agents that were reaped.
    """
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(minutes=_AWAY_MINUTES)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Find stale agents
    rows = conn.execute(
        "SELECT agent, last_seen FROM presence WHERE last_seen < ?",
        (cutoff,),
    ).fetchall()

    reaped = []
    for row in rows:
        agent = row["agent"]
        last_seen = row["last_seen"]

        if _agent_status(last_seen) != "offline":
            continue

        # Get their channel memberships
        channels = [
            r["channel"]
            for r in conn.execute(
                "SELECT channel FROM memberships WHERE agent = ?", (agent,)
            ).fetchall()
        ]

        # Post leave messages to each channel
        leave_time = _now_iso()
        for ch in channels:
            msg = ChannelMessage(
                timestamp=leave_time,
                from_agent=agent,
                channel=ch,
                type="leave",
                body=f"{agent} timed out from #{ch}",
            )
            _append_message(msg, project_dir)

        # Remove memberships and update presence
        conn.execute("DELETE FROM memberships WHERE agent = ?", (agent,))
        conn.execute(
            "UPDATE presence SET status = 'offline' WHERE agent = ?", (agent,)
        )
        reaped.append(agent)

    if reaped:
        conn.commit()
    return reaped


# ---------------------------------------------------------------------------
# JSONL message I/O (unchanged -- this is the open protocol)
# ---------------------------------------------------------------------------

def _append_message(msg: ChannelMessage, project_dir: Path | None = None) -> None:
    """Append a message to a channel's JSONL log with file locking."""
    path = _channel_path(msg.channel, project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    from synapt.recall._filelock import lock_exclusive
    with open(path, "a", encoding="utf-8") as f:
        lock_exclusive(f)
        f.write(json.dumps(msg.to_dict()) + "\n")
        f.flush()


def _read_messages(
    path: Path,
    since: str | None = None,
) -> list[ChannelMessage]:
    """Read all messages from a channel JSONL file, optionally filtered by timestamp."""
    messages: list[ChannelMessage] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = ChannelMessage.from_dict(json.loads(line))
                    if since and msg.timestamp <= since:
                        continue
                    messages.append(msg)
                except (json.JSONDecodeError, TypeError):
                    continue
    except OSError:
        pass
    return messages


def _count_messages_since(path: Path, since: str) -> int:
    """Count JSONL lines with timestamp > since. Used for unread counts."""
    count = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("timestamp", "") > since:
                        count += 1
                except (json.JSONDecodeError, TypeError):
                    continue
    except OSError:
        pass
    return count


# ---------------------------------------------------------------------------
# Core channel operations
# ---------------------------------------------------------------------------

def channel_join(
    channel: str = "dev",
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> str:
    """Join a channel. Auto-detects agent name from worktree name if not provided."""
    agent = _agent_name(agent_name, project_dir)
    now = _now_iso()

    # Determine cursor initial value: the timestamp of the last message
    # currently in the channel (so only future messages are unread).
    # If no messages exist, use the join timestamp.
    path = _channel_path(channel, project_dir)
    cursor_init = now
    if path.exists():
        msgs = _read_messages(path)
        if msgs:
            cursor_init = msgs[-1].timestamp

    conn = _open_db(project_dir)
    try:
        # Upsert presence
        conn.execute(
            "INSERT INTO presence (agent, status, last_seen, joined_at) "
            "VALUES (?, 'online', ?, ?) "
            "ON CONFLICT(agent) DO UPDATE SET status='online', last_seen=?",
            (agent, now, now, now),
        )

        # Add membership
        conn.execute(
            "INSERT OR IGNORE INTO memberships (agent, channel, joined_at) "
            "VALUES (?, ?, ?)",
            (agent, channel, now),
        )

        # Initialize cursor to last message in channel (only new messages are unread)
        conn.execute(
            "INSERT OR IGNORE INTO cursors (agent, channel, last_read_at) "
            "VALUES (?, ?, ?)",
            (agent, channel, cursor_init),
        )

        conn.commit()
    finally:
        conn.close()

    # Append join event to channel log
    msg = ChannelMessage(
        timestamp=now,
        from_agent=agent,
        channel=channel,
        type="join",
        body=f"{agent} joined #{channel}",
    )
    _append_message(msg, project_dir)

    return f"Joined #{channel} as {agent}"


def channel_leave(
    channel: str = "dev",
    agent_name: str | None = None,
    project_dir: Path | None = None,
    reason: str = "",
) -> str:
    """Leave a channel."""
    agent = _agent_name(agent_name, project_dir)
    now = _now_iso()

    conn = _open_db(project_dir)
    try:
        # Remove membership
        conn.execute(
            "DELETE FROM memberships WHERE agent = ? AND channel = ?",
            (agent, channel),
        )

        # Update last_seen
        conn.execute(
            "UPDATE presence SET last_seen = ? WHERE agent = ?",
            (now, agent),
        )

        # Check if agent has any remaining memberships
        remaining = conn.execute(
            "SELECT COUNT(*) FROM memberships WHERE agent = ?",
            (agent,),
        ).fetchone()[0]

        # If no channels left, remove presence entirely
        if remaining == 0:
            conn.execute("DELETE FROM presence WHERE agent = ?", (agent,))

        # Remove cursor for this channel
        conn.execute(
            "DELETE FROM cursors WHERE agent = ? AND channel = ?",
            (agent, channel),
        )

        conn.commit()
    finally:
        conn.close()

    # Append leave event to channel log
    body = reason if reason else f"{agent} left #{channel}"
    msg = ChannelMessage(
        timestamp=now,
        from_agent=agent,
        channel=channel,
        type="leave",
        body=body,
    )
    _append_message(msg, project_dir)

    return f"Left #{channel}"


def channel_post(
    channel: str,
    message: str,
    agent_name: str | None = None,
    project_dir: Path | None = None,
    pin: bool = False,
) -> str:
    """Post a message to a channel. If pin=True, also pin the message."""
    agent = _agent_name(agent_name, project_dir)
    now = _now_iso()

    msg = ChannelMessage(
        timestamp=now,
        from_agent=agent,
        channel=channel,
        type="message",
        body=message,
    )
    _append_message(msg, project_dir)

    # Update last_seen via heartbeat
    conn = _open_db(project_dir)
    try:
        conn.execute(
            "UPDATE presence SET last_seen = ?, status = 'online' WHERE agent = ?",
            (now, agent),
        )

        if pin:
            conn.execute(
                "INSERT INTO pins (channel, body, pinned_by, pinned_at) "
                "VALUES (?, ?, ?, ?)",
                (channel, message, agent, now),
            )

        conn.commit()
    finally:
        conn.close()

    result = f"[#{channel}] {agent}: {message}"
    if pin:
        result += " (pinned)"
    return result


def channel_read(
    channel: str = "dev",
    limit: int = 20,
    since: str | None = None,
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> str:
    """Read recent messages from a channel.

    Args:
        channel: Channel name to read.
        limit: Maximum messages to return.
        since: Optional ISO 8601 timestamp -- only return messages after this time.
        agent_name: Agent reading (for cursor update). Auto-detected if omitted.
        project_dir: Override project directory for path resolution.
    """
    # Reap stale agents on read
    conn = _open_db(project_dir)
    try:
        _reap_stale_agents(conn, project_dir)
    finally:
        conn.close()

    path = _channel_path(channel, project_dir)
    if not path.exists():
        return f"Channel #{channel} has no messages yet."

    messages = _read_messages(path, since=since)

    if not messages:
        if since:
            return f"No messages in #{channel} since {since}."
        return f"Channel #{channel} has no messages yet."

    # Take last N messages
    messages = messages[-limit:]

    # Get pins for this channel
    conn = _open_db(project_dir)
    try:
        pins = conn.execute(
            "SELECT body, pinned_by, pinned_at FROM pins WHERE channel = ? "
            "ORDER BY pinned_at",
            (channel,),
        ).fetchall()
    finally:
        conn.close()

    lines = []

    # Show pins at the top if any
    if pins:
        lines.append(f"## Pinned in #{channel}")
        for pin in pins:
            ts = pin["pinned_at"][:16]
            lines.append(f"  [pin] {ts}  {pin['pinned_by']}: {pin['body']}")
        lines.append("")

    lines.append(f"## #{channel} ({len(messages)} messages)")
    for msg in messages:
        ts = msg.timestamp[:16]  # Trim to minute precision
        if msg.type in ("join", "leave"):
            lines.append(f"  {ts}  -- {msg.body}")
        else:
            lines.append(f"  {ts}  {msg.from_agent}: {msg.body}")

    # Update read cursor for this agent
    agent = _agent_name(agent_name, project_dir)
    now = _now_iso()
    conn = _open_db(project_dir)
    try:
        conn.execute(
            "INSERT INTO cursors (agent, channel, last_read_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(agent, channel) DO UPDATE SET last_read_at = ?",
            (agent, channel, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    return "\n".join(lines)


def channel_who(project_dir: Path | None = None) -> str:
    """Show which agents are currently online and in which channels."""
    conn = _open_db(project_dir)
    try:
        # Reap stale agents first
        _reap_stale_agents(conn, project_dir)

        # Get all agents with their presence info
        agents = conn.execute(
            "SELECT agent, status, last_seen, joined_at FROM presence"
        ).fetchall()

        if not agents:
            return "No agents online."

        lines = ["## Agents"]
        for row in sorted(agents, key=lambda r: r["agent"]):
            agent = row["agent"]
            last_seen = row["last_seen"]

            # Compute live status from last_seen (more accurate than stored status)
            status = _agent_status(last_seen)

            # Skip offline agents in the display
            if status == "offline":
                continue

            # Get channels for this agent
            channels = [
                r["channel"]
                for r in conn.execute(
                    "SELECT channel FROM memberships WHERE agent = ? ORDER BY channel",
                    (agent,),
                ).fetchall()
            ]
            channels_str = ", ".join(f"#{c}" for c in channels) if channels else "(no channels)"

            if status == "online":
                status_label = "online"
            elif status == "idle":
                delta = datetime.now(timezone.utc) - _parse_iso(last_seen)
                mins = int(delta.total_seconds() / 60)
                status_label = f"idle ({mins}m)"
            elif status == "away":
                delta = datetime.now(timezone.utc) - _parse_iso(last_seen)
                mins = int(delta.total_seconds() / 60)
                status_label = f"away ({mins}m)"
            else:
                status_label = status

            lines.append(f"  {agent}  [{status_label}]  {channels_str}")

        if len(lines) == 1:
            # Only header, no visible agents (all offline)
            return "No agents online."

        return "\n".join(lines)
    finally:
        conn.close()


def channel_heartbeat(
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> str:
    """Update last_seen for an agent. Lightweight -- one UPDATE statement.

    Called by hooks (SessionStart, PreCompact) to keep presence fresh.
    """
    agent = _agent_name(agent_name, project_dir)
    now = _now_iso()

    conn = _open_db(project_dir)
    try:
        result = conn.execute(
            "UPDATE presence SET last_seen = ?, status = 'online' WHERE agent = ?",
            (now, agent),
        )
        conn.commit()
        if result.rowcount == 0:
            return f"Agent {agent} not in any channel (heartbeat skipped)."
        return f"Heartbeat: {agent} at {now}"
    finally:
        conn.close()


def channel_unread(
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> dict[str, int]:
    """Return unread message counts per channel for the given agent.

    Returns dict like {"dev": 3, "eval": 0}.
    """
    agent = _agent_name(agent_name, project_dir)

    conn = _open_db(project_dir)
    try:
        # Get all channels this agent is a member of
        memberships = conn.execute(
            "SELECT channel FROM memberships WHERE agent = ?",
            (agent,),
        ).fetchall()

        if not memberships:
            return {}

        result = {}
        for row in memberships:
            ch = row["channel"]

            # Get cursor for this channel
            cursor_row = conn.execute(
                "SELECT last_read_at FROM cursors WHERE agent = ? AND channel = ?",
                (agent, ch),
            ).fetchone()

            if cursor_row:
                last_read = cursor_row["last_read_at"]
            else:
                # No cursor -- count all messages as unread
                last_read = "1970-01-01T00:00:00Z"

            # Count messages in JSONL after the cursor
            path = _channel_path(ch, project_dir)
            count = _count_messages_since(path, last_read) if path.exists() else 0
            result[ch] = count

        return result
    finally:
        conn.close()


def channel_pin(
    channel: str,
    message: str,
    agent_name: str | None = None,
    project_dir: Path | None = None,
) -> str:
    """Pin a message in a channel (without posting to the JSONL log)."""
    agent = _agent_name(agent_name, project_dir)
    now = _now_iso()

    conn = _open_db(project_dir)
    try:
        conn.execute(
            "INSERT INTO pins (channel, body, pinned_by, pinned_at) "
            "VALUES (?, ?, ?, ?)",
            (channel, message, agent, now),
        )
        conn.commit()
    finally:
        conn.close()

    return f"Pinned in #{channel}: {message}"
