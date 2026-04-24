"""Tests for the Google ADK memory service backend (SynaptMemoryService)."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

pytest.importorskip("google.adk")

from google.genai import types
from google.adk.memory.memory_entry import MemoryEntry


def _make_event(event_id: str, text: str, author: str = "user"):
    """Create a minimal Event-like object for testing."""
    from google.adk.events.event import Event

    return Event(
        id=event_id,
        author=author,
        content=types.Content(
            parts=[types.Part(text=text)],
            role="user" if author == "user" else "model",
        ),
    )


def _make_session(app_name: str, user_id: str, session_id: str, events=None):
    """Create a minimal Session object for testing."""
    from google.adk.sessions.session import Session

    return Session(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        events=events or [],
    )


@pytest.fixture
def service():
    with patch("synapt.integrations.google_adk.SynaptMemoryService._save_to_recall") as mock_save:
        from synapt.integrations.google_adk import SynaptMemoryService
        svc = SynaptMemoryService(max_search_results=5)
        svc._mock_save = mock_save
        yield svc


class TestAddSessionToMemory:

    @pytest.mark.asyncio
    async def test_ingests_events(self, service):
        events = [
            _make_event("e1", "hello world"),
            _make_event("e2", "how are you", author="agent"),
        ]
        session = _make_session("myapp", "user1", "s1", events=events)
        await service.add_session_to_memory(session)
        assert service._mock_save.call_count == 2

    @pytest.mark.asyncio
    async def test_deduplicates_events(self, service):
        events = [_make_event("e1", "hello")]
        session = _make_session("myapp", "user1", "s1", events=events)
        await service.add_session_to_memory(session)
        await service.add_session_to_memory(session)
        assert service._mock_save.call_count == 1

    @pytest.mark.asyncio
    async def test_skips_empty_events(self, service):
        from google.adk.events.event import Event

        empty = Event(id="e1", author="user", content=None)
        session = _make_session("myapp", "user1", "s1", events=[empty])
        await service.add_session_to_memory(session)
        assert service._mock_save.call_count == 0


class TestAddEventsToMemory:

    @pytest.mark.asyncio
    async def test_incremental_events(self, service):
        events = [_make_event("e1", "incremental event")]
        await service.add_events_to_memory(
            app_name="myapp",
            user_id="user1",
            events=events,
        )
        assert service._mock_save.call_count == 1

    @pytest.mark.asyncio
    async def test_dedup_across_methods(self, service):
        events = [_make_event("e1", "hello")]
        session = _make_session("myapp", "user1", "s1", events=events)
        await service.add_session_to_memory(session)
        await service.add_events_to_memory(
            app_name="myapp",
            user_id="user1",
            events=events,
        )
        assert service._mock_save.call_count == 1


class TestAddMemory:

    @pytest.mark.asyncio
    async def test_direct_memory_write(self, service):
        entry = MemoryEntry(
            content=types.Content(
                parts=[types.Part(text="important fact")],
                role="model",
            ),
            author="agent",
        )
        await service.add_memory(
            app_name="myapp",
            user_id="user1",
            memories=[entry],
        )
        assert service._mock_save.call_count == 1

    @pytest.mark.asyncio
    async def test_skips_empty_memory(self, service):
        entry = MemoryEntry(
            content=types.Content(parts=[], role="model"),
        )
        await service.add_memory(
            app_name="myapp",
            user_id="user1",
            memories=[entry],
        )
        assert service._mock_save.call_count == 0


class TestSearchMemory:

    @pytest.mark.asyncio
    async def test_search_returns_entries(self, service):
        fake_nodes = [
            {
                "content": "chunk one",
                "tags": ["google-adk", "app:myapp", "user:user1", "author:recall"],
                "updated_at": "2026-04-24T00:00:00Z",
                "created_at": "2026-04-24T00:00:00Z",
            },
            {
                "content": "chunk two",
                "tags": ["google-adk", "app:myapp", "user:user1", "author:recall"],
                "updated_at": "2026-04-24T00:00:01Z",
                "created_at": "2026-04-24T00:00:01Z",
            },
        ]
        mock_db = MagicMock()
        mock_db.load_knowledge_nodes.return_value = fake_nodes
        with patch("synapt.integrations.google_adk.RecallDB", return_value=mock_db), \
             patch("synapt.integrations.google_adk.project_index_dir", return_value=MagicMock()):
            result = await service.search_memory(
                app_name="myapp",
                user_id="user1",
                query="chunk",
            )
        assert len(result.memories) == 2
        assert result.memories[0].content.parts[0].text == "chunk one"

    @pytest.mark.asyncio
    async def test_search_empty_results(self, service):
        mock_db = MagicMock()
        mock_db.load_knowledge_nodes.return_value = []
        with patch("synapt.integrations.google_adk.RecallDB", return_value=mock_db), \
             patch("synapt.integrations.google_adk.project_index_dir", return_value=MagicMock()):
            result = await service.search_memory(
                app_name="myapp",
                user_id="user1",
                query="nonexistent",
            )
        assert len(result.memories) == 0

    @pytest.mark.asyncio
    async def test_search_handles_error(self, service):
        with patch("synapt.integrations.google_adk.RecallDB", side_effect=Exception("boom")):
            result = await service.search_memory(
                app_name="myapp",
                user_id="user1",
                query="hello",
            )
        assert len(result.memories) == 0

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self, service):
        fake_nodes = [
            {
                "content": f"lots chunk {i}",
                "tags": ["google-adk", "app:myapp", "user:user1", "author:recall"],
                "updated_at": f"2026-04-24T00:00:{i:02d}Z",
                "created_at": f"2026-04-24T00:00:{i:02d}Z",
            }
            for i in range(20)
        ]
        mock_db = MagicMock()
        mock_db.load_knowledge_nodes.return_value = fake_nodes
        with patch("synapt.integrations.google_adk.RecallDB", return_value=mock_db), \
             patch("synapt.integrations.google_adk.project_index_dir", return_value=MagicMock()):
            result = await service.search_memory(
                app_name="myapp",
                user_id="user1",
                query="lots",
            )
        assert len(result.memories) <= 5

    @pytest.mark.asyncio
    async def test_search_scopes_by_app_and_user(self, service):
        fake_nodes = [
            {
                "content": "deployment strategy is blue green",
                "tags": ["google-adk", "app:myapp", "user:user1", "author:atlas"],
                "updated_at": "2026-04-24T00:00:00Z",
                "created_at": "2026-04-24T00:00:00Z",
            },
            {
                "content": "deployment strategy is canary",
                "tags": ["google-adk", "app:otherapp", "user:user1", "author:apollo"],
                "updated_at": "2026-04-24T00:00:01Z",
                "created_at": "2026-04-24T00:00:01Z",
            },
            {
                "content": "deployment strategy is rolling",
                "tags": ["google-adk", "app:myapp", "user:user2", "author:sentinel"],
                "updated_at": "2026-04-24T00:00:02Z",
                "created_at": "2026-04-24T00:00:02Z",
            },
        ]
        mock_db = MagicMock()
        mock_db.load_knowledge_nodes.return_value = fake_nodes
        with patch("synapt.integrations.google_adk.RecallDB", return_value=mock_db), \
             patch("synapt.integrations.google_adk.project_index_dir", return_value=MagicMock()):
            result = await service.search_memory(
                app_name="myapp",
                user_id="user1",
                query="deployment strategy",
            )
        assert len(result.memories) == 1
        assert result.memories[0].content.parts[0].text == "deployment strategy is blue green"
        assert result.memories[0].author == "atlas"

    @pytest.mark.asyncio
    async def test_search_ignores_legacy_unscoped_nodes(self, service):
        fake_nodes = [
            {
                "content": "deployment strategy is blue green",
                "tags": ["google-adk", "myapp", "atlas"],
                "updated_at": "2026-04-24T00:00:00Z",
                "created_at": "2026-04-24T00:00:00Z",
            }
        ]
        mock_db = MagicMock()
        mock_db.load_knowledge_nodes.return_value = fake_nodes
        with patch("synapt.integrations.google_adk.RecallDB", return_value=mock_db), \
             patch("synapt.integrations.google_adk.project_index_dir", return_value=MagicMock()):
            result = await service.search_memory(
                app_name="myapp",
                user_id="user1",
                query="deployment strategy",
            )
        assert len(result.memories) == 0


class TestSearchMemoryResponse:

    @pytest.mark.asyncio
    async def test_entries_have_correct_fields(self, service):
        fake_nodes = [
            {
                "content": "some context",
                "tags": ["google-adk", "app:myapp", "user:user1", "author:recall"],
                "updated_at": "2026-04-24T00:00:00Z",
                "created_at": "2026-04-24T00:00:00Z",
            }
        ]
        mock_db = MagicMock()
        mock_db.load_knowledge_nodes.return_value = fake_nodes
        with patch("synapt.integrations.google_adk.RecallDB", return_value=mock_db), \
             patch("synapt.integrations.google_adk.project_index_dir", return_value=MagicMock()):
            result = await service.search_memory(
                app_name="myapp",
                user_id="user1",
                query="context",
            )
        entry = result.memories[0]
        assert entry.author == "recall"
        assert entry.timestamp is not None
        assert entry.content.role == "model"


class TestSaveToRecallTags:

    @pytest.mark.asyncio
    async def test_direct_memory_write_tags_app_and_user(self, service):
        entry = MemoryEntry(
            content=types.Content(
                parts=[types.Part(text="important fact")],
                role="model",
            ),
            author="agent",
        )
        await service.add_memory(
            app_name="myapp",
            user_id="user1",
            memories=[entry],
        )
        _, kwargs = service._mock_save.call_args
        assert kwargs["app_name"] == "myapp"
        assert kwargs["user_id"] == "user1"


class TestImportability:

    def test_importable(self):
        from synapt.integrations.google_adk import SynaptMemoryService
        assert SynaptMemoryService is not None
