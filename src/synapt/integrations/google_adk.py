"""Google ADK memory service backed by synapt recall.

Provides SynaptMemoryService, a BaseMemoryService implementation that
stores conversation events in recall's knowledge graph and uses hybrid
search for retrieval. Drop-in replacement for InMemoryMemoryService.

Usage:
    from google.adk.agents import LlmAgent
    from synapt.integrations.google_adk import SynaptMemoryService

    memory = SynaptMemoryService()

    agent = LlmAgent(
        model="gemini-2.5-flash",
        name="my_agent",
        memory_service=memory,
    )
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

try:
    from google.adk.memory.base_memory_service import (
        BaseMemoryService,
        SearchMemoryResponse,
    )
    from google.adk.memory.memory_entry import MemoryEntry
    from google.genai import types
except ImportError as exc:
    BaseMemoryService = None  # type: ignore[assignment,misc]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

if TYPE_CHECKING:
    from google.adk.events.event import Event
    from google.adk.sessions.session import Session

log = logging.getLogger("synapt.integrations.google_adk")


def _require_adk() -> None:
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "Install google-adk to use the Google ADK memory backend"
        ) from _IMPORT_ERROR


def _event_text(event: Event) -> str:
    if not event.content or not event.content.parts:
        return ""
    return " ".join(
        part.text for part in event.content.parts if getattr(part, "text", None)
    )


def _session_key(app_name: str, user_id: str) -> str:
    return f"{app_name}/{user_id}"


def _node_id(app_name: str, user_id: str, content_hash: str) -> str:
    raw = f"{app_name}/{user_id}/{content_hash}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


if BaseMemoryService is not None:

    class SynaptMemoryService(BaseMemoryService):
        """Google ADK MemoryService backed by synapt recall.

        Replaces the default InMemoryMemoryService with recall-powered
        hybrid search. Session events are persisted as recall knowledge
        nodes; search_memory uses semantic + keyword hybrid retrieval.

        Args:
            max_search_results: Maximum number of memory entries to return
                from search_memory (default: 10).
        """

        def __init__(self, *, max_search_results: int = 10) -> None:
            _require_adk()
            self._max_results = max_search_results
            self._lock = threading.Lock()
            self._ingested: dict[str, set[str]] = {}

        async def add_session_to_memory(self, session: Session) -> None:
            key = _session_key(session.app_name, session.user_id)

            with self._lock:
                if key not in self._ingested:
                    self._ingested[key] = set()
                seen = self._ingested[key]

            for event in session.events:
                text = _event_text(event)
                if not text or event.id in seen:
                    continue
                self._save_to_recall(
                    app_name=session.app_name,
                    user_id=session.user_id,
                    text=text,
                    author=getattr(event, "author", "unknown"),
                )
                with self._lock:
                    seen.add(event.id)

        async def add_events_to_memory(
            self,
            *,
            app_name: str,
            user_id: str,
            events: Sequence[Event],
            session_id: str | None = None,
            custom_metadata: Mapping[str, object] | None = None,
        ) -> None:
            key = _session_key(app_name, user_id)

            with self._lock:
                if key not in self._ingested:
                    self._ingested[key] = set()
                seen = self._ingested[key]

            for event in events:
                text = _event_text(event)
                if not text or event.id in seen:
                    continue
                self._save_to_recall(
                    app_name=app_name,
                    user_id=user_id,
                    text=text,
                    author=getattr(event, "author", "unknown"),
                )
                with self._lock:
                    seen.add(event.id)

        async def add_memory(
            self,
            *,
            app_name: str,
            user_id: str,
            memories: Sequence[MemoryEntry],
            custom_metadata: Mapping[str, object] | None = None,
        ) -> None:
            for entry in memories:
                text = ""
                if entry.content and entry.content.parts:
                    text = " ".join(
                        part.text
                        for part in entry.content.parts
                        if getattr(part, "text", None)
                    )
                if not text:
                    continue
                self._save_to_recall(
                    app_name=app_name,
                    user_id=user_id,
                    text=text,
                    author=entry.author or "unknown",
                )

        async def search_memory(
            self,
            *,
            app_name: str,
            user_id: str,
            query: str,
        ) -> SearchMemoryResponse:
            try:
                from synapt.recall.server import recall_search

                result_text = recall_search(
                    query=query,
                    max_chunks=self._max_results,
                    max_tokens=2000,
                )
            except Exception as e:
                log.warning("Recall search failed: %s", e)
                return SearchMemoryResponse()

            if not result_text or "No results" in result_text:
                return SearchMemoryResponse()

            entries = []
            for chunk in result_text.split("\n\n"):
                chunk = chunk.strip()
                if not chunk or chunk.startswith("---"):
                    continue
                entry = MemoryEntry(
                    content=types.Content(
                        parts=[types.Part(text=chunk)],
                        role="model",
                    ),
                    author="recall",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                entries.append(entry)

            return SearchMemoryResponse(memories=entries[:self._max_results])

        def _save_to_recall(
            self,
            *,
            app_name: str,
            user_id: str,
            text: str,
            author: str,
        ) -> None:
            try:
                from synapt.recall.server import recall_save

                content_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
                recall_save(
                    content=text,
                    category="conversation",
                    confidence=0.8,
                    tags=["google-adk", app_name, author],
                    node_id=_node_id(app_name, user_id, content_hash),
                )
            except Exception as e:
                log.warning("Recall save failed: %s", e)

else:

    class SynaptMemoryService:  # type: ignore[no-redef]
        """Stub when google-adk is not installed."""

        def __init__(self, **kwargs: Any) -> None:
            _require_adk()
