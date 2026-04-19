"""LangChain chat message history backed by synapt recall.

Provides SynaptChatMessageHistory, a BaseChatMessageHistory implementation
that stores messages in a local SQLite database and optionally bridges to
recall's semantic search and knowledge persistence.

Usage:
    from synapt.integrations.langchain import SynaptChatMessageHistory

    history = SynaptChatMessageHistory(session_id="user-123")
    history.add_messages([HumanMessage(content="hello")])
    print(history.messages)

    # Semantic search across all recall-indexed sessions
    results = history.search("deployment config")

    # Persist a message as a durable knowledge node
    history.save_to_recall("Always use UTC timestamps", category="convention")
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict

_DEFAULT_DB_DIR = Path.home() / ".synapt" / "integrations"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    type TEXT NOT NULL,
    data TEXT NOT NULL,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages (session_id, id);
"""


class SynaptChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in SQLite with recall search integration."""

    def __init__(
        self,
        session_id: str,
        *,
        db_path: Optional[Path] = None,
        recall_project: Optional[Path] = None,
    ) -> None:
        self.session_id = session_id
        self._recall_project = recall_project

        if db_path is None:
            db_path = _DEFAULT_DB_DIR / "langchain_messages.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    @property
    def messages(self) -> List[BaseMessage]:
        cursor = self._conn.execute(
            "SELECT type, data FROM messages WHERE session_id = ? ORDER BY id",
            (self.session_id,),
        )
        return messages_from_dict(
            [{"type": row[0], "data": json.loads(row[1])} for row in cursor]
        )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        now = time.time()
        rows = []
        for msg in messages:
            data = {
                "content": msg.content,
                "additional_kwargs": msg.additional_kwargs,
                "type": msg.type,
            }
            if hasattr(msg, "tool_call_id"):
                data["tool_call_id"] = msg.tool_call_id
            if hasattr(msg, "name") and msg.name:
                data["name"] = msg.name
            rows.append((self.session_id, msg.type, json.dumps(data), now))
        self._conn.executemany(
            "INSERT INTO messages (session_id, type, data, timestamp) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute(
            "DELETE FROM messages WHERE session_id = ?",
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
        """Semantic search across recall-indexed sessions.

        Requires a recall index to exist for the project. Returns formatted
        context chunks ranked by relevance with recency decay.
        """
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
    def session_count(self) -> int:
        """Number of messages in the current session."""
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (self.session_id,),
        )
        return cursor.fetchone()[0]

    def get_session_ids(self) -> list[str]:
        """List all session IDs with stored messages."""
        cursor = self._conn.execute(
            "SELECT DISTINCT session_id FROM messages ORDER BY session_id"
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
            f"SynaptChatMessageHistory(session_id={self.session_id!r}, "
            f"messages={self.session_count})"
        )
