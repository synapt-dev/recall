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

    # Get recall-augmented context for agent prompts
    context = session.get_memory_context("What deployment strategy do we use?")
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

try:
    from agents.memory.session import SessionABC
    from agents.memory.session_settings import SessionSettings, resolve_session_limit
except ImportError:
    SessionABC = object  # type: ignore[assignment,misc]
    SessionSettings = None  # type: ignore[assignment,misc]

    def resolve_session_limit(explicit_limit, settings):  # type: ignore[misc]
        if explicit_limit is not None:
            return explicit_limit
        return None


log = logging.getLogger("synapt.integrations.openai_agents")

_DEFAULT_DB_DIR = Path.home() / ".synapt" / "integrations"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS agent_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    item_data TEXT NOT NULL,
    timestamp REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES agent_sessions (session_id)
        ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_agent_items_session
    ON agent_items (session_id, id);
"""


class SynaptSession(SessionABC):  # type: ignore[misc]
    """OpenAI Agents SDK Session backed by SQLite with recall integration.

    Implements the Session protocol: get_items, add_items, pop_item,
    clear_session. Uses asyncio.to_thread for non-blocking DB access.

    Extends the standard session with recall-powered features:
    - search(): semantic search across recall-indexed sessions
    - save_to_recall(): persist durable knowledge nodes
    - get_memory_context(): retrieve relevant recall context for prompts

    Args:
        session_id: Unique identifier for the conversation session.
        db_path: Path to SQLite database. Defaults to ~/.synapt/integrations/.
        session_settings: Optional SessionSettings for default limit behavior.
    """

    session_id: str
    session_settings: Any = None

    def __init__(
        self,
        session_id: str,
        *,
        db_path: Optional[Path] = None,
        session_settings: Any = None,
    ) -> None:
        self.session_id = session_id
        if SessionSettings is not None and session_settings is None:
            self.session_settings = SessionSettings()
        else:
            self.session_settings = session_settings

        if db_path is None:
            db_path = _DEFAULT_DB_DIR / "openai_agents_items.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._closed = False

    async def get_items(self, limit: int | None = None) -> list[dict[str, Any]]:
        session_limit = resolve_session_limit(limit, self.session_settings)

        def _sync() -> list[dict[str, Any]]:
            with self._lock:
                if session_limit is not None:
                    cursor = self._conn.execute(
                        "SELECT item_data FROM ("
                        "  SELECT item_data, id FROM agent_items"
                        "  WHERE session_id = ? ORDER BY id DESC LIMIT ?"
                        ") sub ORDER BY id",
                        (self.session_id, session_limit),
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

        return await asyncio.to_thread(_sync)

    async def add_items(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        now = time.time()

        def _sync() -> None:
            with self._lock:
                self._conn.execute(
                    "INSERT OR IGNORE INTO agent_sessions (session_id) VALUES (?)",
                    (self.session_id,),
                )
                rows = [(self.session_id, json.dumps(item), now) for item in items]
                self._conn.executemany(
                    "INSERT INTO agent_items (session_id, item_data, timestamp) VALUES (?, ?, ?)",
                    rows,
                )
                self._conn.execute(
                    "UPDATE agent_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                    (self.session_id,),
                )
                self._conn.commit()

        await asyncio.to_thread(_sync)

    async def pop_item(self) -> dict[str, Any] | None:
        def _sync() -> dict[str, Any] | None:
            with self._lock:
                cursor = self._conn.execute(
                    "DELETE FROM agent_items "
                    "WHERE id = ("
                    "  SELECT id FROM agent_items WHERE session_id = ? ORDER BY id DESC LIMIT 1"
                    ") RETURNING item_data",
                    (self.session_id,),
                )
                result = cursor.fetchone()
                self._conn.commit()
                if result:
                    try:
                        return json.loads(result[0])
                    except (json.JSONDecodeError, TypeError):
                        return None
                return None

        return await asyncio.to_thread(_sync)

    async def clear_session(self) -> None:
        def _sync() -> None:
            with self._lock:
                self._conn.execute(
                    "DELETE FROM agent_items WHERE session_id = ?",
                    (self.session_id,),
                )
                self._conn.execute(
                    "DELETE FROM agent_sessions WHERE session_id = ?",
                    (self.session_id,),
                )
                self._conn.commit()

        await asyncio.to_thread(_sync)

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

    def get_memory_context(
        self,
        query: str,
        *,
        max_chunks: int = 3,
        max_tokens: int = 1000,
    ) -> str:
        """Retrieve recall context relevant to a query for agent prompts.

        Returns a formatted string suitable for injection into system
        prompts or tool context.
        """
        try:
            from synapt.recall.server import recall_search

            result = recall_search(
                query=query,
                max_chunks=max_chunks,
                max_tokens=max_tokens,
            )
            if not result or "No results" in result:
                return ""
            return result
        except Exception as e:
            log.debug("Memory context retrieval failed: %s", e)
            return ""

    @property
    def item_count(self) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM agent_items WHERE session_id = ?",
                (self.session_id,),
            )
            return cursor.fetchone()[0]

    def get_session_ids(self) -> list[str]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT DISTINCT session_id FROM agent_items ORDER BY session_id"
            )
            return [row[0] for row in cursor]

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._conn.close()

    def __del__(self) -> None:
        try:
            if not self._closed:
                self._conn.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"SynaptSession(session_id={self.session_id!r}, "
            f"items={self.item_count})"
        )
