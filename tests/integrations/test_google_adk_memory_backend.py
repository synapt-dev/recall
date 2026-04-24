"""Red specs for the Google ADK memory backend."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest

pytest.importorskip("google.adk", reason="google-adk not installed")

from google.adk.events import Event  # noqa: E402
from google.adk.memory.base_memory_service import BaseMemoryService, SearchMemoryResponse  # noqa: E402
from google.adk.sessions import Session  # noqa: E402
from google.genai import types  # noqa: E402


def _load_synapt_memory_service():
    try:
        module = importlib.import_module("synapt.integrations.google_adk")
    except ModuleNotFoundError:
        pytest.fail(
            "Expected backend module `synapt.integrations.google_adk` with "
            "`SynaptMemoryService`, but it does not exist yet."
        )
    except Exception as exc:  # pragma: no cover - exercised once backend exists
        pytest.fail(f"Importing `synapt.integrations.google_adk` failed: {exc!r}")

    try:
        return module.SynaptMemoryService
    except AttributeError:
        pytest.fail(
            "Expected `synapt.integrations.google_adk.SynaptMemoryService`, "
            "but the symbol is missing."
        )


def _new_memory_service(tmp_path: Path):
    cls = _load_synapt_memory_service()
    try:
        return cls(project_root=tmp_path)
    except TypeError:
        pytest.fail(
            "SynaptMemoryService should be constructible as "
            "`SynaptMemoryService(project_root=Path(...))`."
        )


def _event(text: str, *, invocation_id: str) -> Event:
    return Event(
        author="user",
        invocation_id=invocation_id,
        content=types.Content(parts=[types.Part(text=text)], role="user"),
    )


def test_synapt_memory_service_satisfies_the_adk_memory_interface(tmp_path: Path) -> None:
    service = _new_memory_service(tmp_path)

    assert isinstance(service, BaseMemoryService)


def test_add_session_to_memory_and_search_memory_round_trip(tmp_path: Path) -> None:
    service = _new_memory_service(tmp_path)
    session = Session(
        id="sess-1",
        app_name="synapt-app",
        user_id="user-1",
        events=[_event("prefers pour-over coffee", invocation_id="inv-1")],
    )

    asyncio.run(service.add_session_to_memory(session))
    result = asyncio.run(
        service.search_memory(
            app_name="synapt-app",
            user_id="user-1",
            query="coffee",
        )
    )

    assert isinstance(result, SearchMemoryResponse)
    assert len(result.memories) == 1
    assert result.memories[0].content.parts[0].text == "prefers pour-over coffee"


def test_add_events_to_memory_supports_incremental_updates(tmp_path: Path) -> None:
    service = _new_memory_service(tmp_path)

    asyncio.run(
        service.add_events_to_memory(
            app_name="synapt-app",
            user_id="user-1",
            session_id="sess-2",
            events=[_event("likes Ethiopian beans", invocation_id="inv-2")],
        )
    )

    result = asyncio.run(
        service.search_memory(
            app_name="synapt-app",
            user_id="user-1",
            query="Ethiopian",
        )
    )

    assert len(result.memories) == 1
    assert result.memories[0].content.parts[0].text == "likes Ethiopian beans"


def test_memory_search_is_scoped_by_app_and_user(tmp_path: Path) -> None:
    service = _new_memory_service(tmp_path)
    session = Session(
        id="sess-3",
        app_name="synapt-app",
        user_id="user-1",
        events=[_event("favorite IDE theme is light", invocation_id="inv-3")],
    )

    asyncio.run(service.add_session_to_memory(session))

    miss = asyncio.run(
        service.search_memory(
            app_name="synapt-app",
            user_id="user-2",
            query="theme",
        )
    )

    assert miss.memories == []
