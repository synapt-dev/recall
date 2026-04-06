"""Org agent registry — stable identity for agents across sessions.

Every agent in an org gets a unique, permanent agent_id (e.g., "opus-001").
Display names are human-readable and unique per org but can be changed.

Storage: org_agents table in ~/.synapt/orgs/<org_id>/team.db.
Resolution: SYNAPT_AGENT_ID env var → presence DB lookup → auto-register.

Part of Phase 0 of the channel scoping design (5-iteration adversarial review).
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _team_db_path(org_id: str) -> Path:
    """Return the path to an org's team.db."""
    return Path.home() / ".synapt" / "orgs" / org_id / "team.db"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the org_agents table if it doesn't exist."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS org_agents (
            agent_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            role TEXT,
            org_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_seen_at TEXT
        )"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_org_display
           ON org_agents(org_id, display_name)"""
    )
    conn.commit()


def _generate_agent_id(display_name: str, conn: sqlite3.Connection, org_id: str) -> str:
    """Generate a unique agent_id in the format 'name-NNN'.

    Finds the next available number for the given name prefix.
    """
    slug = display_name.lower().replace(" ", "-")
    rows = conn.execute(
        "SELECT agent_id FROM org_agents WHERE agent_id LIKE ?",
        (f"{slug}-%",),
    ).fetchall()
    existing_nums = []
    for (aid,) in rows:
        parts = aid.rsplit("-", 1)
        if len(parts) == 2:
            try:
                existing_nums.append(int(parts[1]))
            except ValueError:
                pass
    next_num = max(existing_nums, default=0) + 1
    return f"{slug}-{next_num:03d}"


def _open_db(org_id: str, db_path: Path | None = None) -> sqlite3.Connection:
    """Open team.db for the given org, ensuring schema exists."""
    path = db_path or _team_db_path(org_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def register_agent(
    org_id: str,
    display_name: str,
    role: str | None = None,
    db_path: Path | None = None,
) -> str:
    """Register a new agent in the org. Returns the assigned agent_id.

    Raises sqlite3.IntegrityError if display_name is already taken in this org.
    """
    conn = _open_db(org_id, db_path)
    try:
        now = datetime.now(timezone.utc).isoformat()
        agent_id = _generate_agent_id(display_name, conn, org_id)
        conn.execute(
            "INSERT INTO org_agents (agent_id, display_name, role, org_id, created_at, last_seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (agent_id, display_name, role, org_id, now, now),
        )
        conn.commit()
        return agent_id
    finally:
        conn.close()


def get_agent(agent_id: str, org_id: str | None = None, db_path: Path | None = None) -> dict[str, Any] | None:
    """Look up an agent by agent_id. Returns dict or None if not found."""
    if org_id is None and db_path is None:
        raise ValueError("Either org_id or db_path must be provided")
    conn = _open_db(org_id or "", db_path)
    try:
        row = conn.execute(
            "SELECT * FROM org_agents WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_agent_by_name(
    org_id: str, display_name: str, db_path: Path | None = None
) -> dict[str, Any] | None:
    """Look up an agent by org + display_name. Returns dict or None."""
    conn = _open_db(org_id, db_path)
    try:
        row = conn.execute(
            "SELECT * FROM org_agents WHERE org_id = ? AND display_name = ?",
            (org_id, display_name),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_agents(org_id: str, db_path: Path | None = None) -> list[dict[str, Any]]:
    """List all agents in an org."""
    conn = _open_db(org_id, db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM org_agents WHERE org_id = ? ORDER BY created_at",
            (org_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_display_name(
    agent_id: str,
    new_display_name: str,
    org_id: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Update an agent's display name. Raises IntegrityError if name is taken."""
    if org_id is None and db_path is None:
        raise ValueError("Either org_id or db_path must be provided")
    conn = _open_db(org_id or "", db_path)
    try:
        conn.execute(
            "UPDATE org_agents SET display_name = ? WHERE agent_id = ?",
            (new_display_name, agent_id),
        )
        conn.commit()
    finally:
        conn.close()


def touch_last_seen(agent_id: str, org_id: str, db_path: Path | None = None) -> None:
    """Update the last_seen_at timestamp for an agent."""
    conn = _open_db(org_id, db_path)
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE org_agents SET last_seen_at = ? WHERE agent_id = ?",
            (now, agent_id),
        )
        conn.commit()
    finally:
        conn.close()


def resolve_agent_id(org_id: str, db_path: Path | None = None) -> str | None:
    """Resolve the current agent's ID.

    Priority:
    1. SYNAPT_AGENT_ID env var (set by gr spawn)
    2. None (caller should auto-register)
    """
    env_id = os.environ.get("SYNAPT_AGENT_ID")
    if env_id:
        return env_id
    return None
