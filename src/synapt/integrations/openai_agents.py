"""OpenAI Agents SDK session backed by synapt recall.

Provides SynaptSession, a Session-compatible implementation that stores
conversation items in a local SQLite database and bridges to recall's
semantic search and knowledge persistence.

Usage:
    from synapt.integrations.openai_agents import SynaptSession

    session = SynaptSession(session_id="user-123")
    await session.add_items([{"role": "user", "content": "hello"}])
    items = await session.get_items()

    # Semantic search across all recall-indexed sessions
    results = session.search("deployment config")
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

try:  # pragma: no cover - optional dependency surface
    from agents.memory.session import SessionABC
except ImportError:  # pragma: no cover
    SessionABC = object  # type: ignore[assignment,misc]

_DEFAULT_DB_DIR = Path.home() / ".synapt" / "integrations"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS agent_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    item_data TEXT NOT NULL,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_items_session
    ON agent_items (session_id, id);
"""


class SynaptSession(SessionABC):
    """OpenAI Agents SDK Session backed by SQLite with recall integration.

    Implements the Session protocol: get_items, add_items, pop_item,
    clear_session. All methods are async to match the SDK contract.
    """

    session_id: str
    session_settings: Any = None

    def __init__(
        self,
        session_id: str,
        *,
        db_path: Optional[Path] = None,
    ) -> None:
        self.session_id = session_id

        if db_path is None:
            db_path = _DEFAULT_DB_DIR / "openai_agents_items.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    async def get_items(self, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is not None:
            cursor = self._conn.execute(
                "SELECT item_data FROM ("
                "  SELECT item_data, id FROM agent_items"
                "  WHERE session_id = ? ORDER BY id DESC LIMIT ?"
                ") sub ORDER BY id",
                (self.session_id, limit),
            )
        else:
            cursor = self._conn.execute(
                "SELECT item_data FROM agent_items WHERE session_id = ? ORDER BY id",
                (self.session_id,),
            )
        items = []
        for (row,) in cursor:
            try:
                items.append(json.loads(row))
            except (json.JSONDecodeError, TypeError):
                continue
        return items

    async def add_items(self, items: list[dict[str, Any]]) -> None:
        now = time.time()
        rows = [(self.session_id, json.dumps(item), now) for item in items]
        self._conn.executemany(
            "INSERT INTO agent_items (session_id, item_data, timestamp) VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()

    async def pop_item(self) -> dict[str, Any] | None:
        cursor = self._conn.execute(
            "SELECT id, item_data FROM agent_items WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (self.session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        row_id, item_data = row
        self._conn.execute("DELETE FROM agent_items WHERE id = ?", (row_id,))
        self._conn.commit()
        try:
            return json.loads(item_data)
        except (json.JSONDecodeError, TypeError):
            return None

    async def clear_session(self) -> None:
        self._conn.execute(
            "DELETE FROM agent_items WHERE session_id = ?",
            (self.session_id,),
        )
        self._conn.commit()

    def search(
        self,
        query: str,
        *,
        max_chunks: int = 5,
        max_tokens: int = 1500,
    ) -> str:
        """Semantic search across recall-indexed sessions."""
        from synapt.recall.server import recall_search

        return recall_search(
            query=query,
            max_chunks=max_chunks,
            max_tokens=max_tokens,
        )

    def save_to_recall(
        self,
        content: str,
        *,
        category: str = "workflow",
        confidence: float = 0.8,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Persist a fact or decision as a durable recall knowledge node."""
        from synapt.recall.server import recall_save

        return recall_save(
            content=content,
            category=category,
            confidence=confidence,
            tags=tags,
        )

    @property
    def item_count(self) -> int:
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM agent_items WHERE session_id = ?",
            (self.session_id,),
        )
        return cursor.fetchone()[0]

    def get_session_ids(self) -> list[str]:
        cursor = self._conn.execute(
            "SELECT DISTINCT session_id FROM agent_items ORDER BY session_id"
        )
        return [row[0] for row in cursor]

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"SynaptSession(session_id={self.session_id!r}, "
            f"items={self.item_count})"
        )
