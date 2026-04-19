from __future__ import annotations

import pytest

crewai_memory = pytest.importorskip("crewai.memory")
crewai_types = pytest.importorskip("crewai.memory.types")

CrewAIMemory = crewai_memory.Memory
MemoryMatch = crewai_types.MemoryMatch
MemoryRecord = crewai_types.MemoryRecord


def _adapter_cls():
    from synapt.integrations.crewai import SynaptMemory

    return SynaptMemory


def test_crewai_adapter_instantiates_and_matches_memory_interface(tmp_path):
    SynaptMemory = _adapter_cls()

    memory = SynaptMemory(
        db_path=tmp_path / "recall.db",
        session_id="crew-session-1",
    )

    assert isinstance(memory, CrewAIMemory)
    assert memory.session_id == "crew-session-1"
    assert memory.recall("anything", limit=3) == []


def test_crewai_adapter_remember_and_recall_round_trip(tmp_path):
    SynaptMemory = _adapter_cls()

    memory = SynaptMemory(
        db_path=tmp_path / "recall.db",
        session_id="crew-session-1",
    )

    record = memory.remember(
        "The customer prefers JSON payloads and terse summaries.",
        metadata={"kind": "preference"},
    )
    matches = memory.recall("What output format does the customer prefer?", limit=3)

    assert isinstance(record, MemoryRecord)
    assert isinstance(matches, list)
    assert matches
    assert isinstance(matches[0], MemoryMatch)
    assert matches[0].record.content == "The customer prefers JSON payloads and terse summaries."


def test_crewai_adapter_persists_session_state_and_isolates_other_sessions(tmp_path):
    SynaptMemory = _adapter_cls()

    first = SynaptMemory(
        db_path=tmp_path / "recall.db",
        session_id="crew-session-1",
    )
    first.remember("Escalations for Acme should go to Dana first.")

    reloaded = SynaptMemory(
        db_path=tmp_path / "recall.db",
        session_id="crew-session-1",
    )
    other_session = SynaptMemory(
        db_path=tmp_path / "recall.db",
        session_id="crew-session-2",
    )

    same_session_matches = reloaded.recall("Who handles Acme escalations?", limit=3)
    other_session_matches = other_session.recall("Who handles Acme escalations?", limit=3)

    assert same_session_matches
    assert same_session_matches[0].record.content == "Escalations for Acme should go to Dana first."
    assert other_session_matches == []


def test_crewai_adapter_reset_clears_only_the_current_session(tmp_path):
    SynaptMemory = _adapter_cls()

    first = SynaptMemory(
        db_path=tmp_path / "recall.db",
        session_id="crew-session-1",
    )
    second = SynaptMemory(
        db_path=tmp_path / "recall.db",
        session_id="crew-session-2",
    )

    first.remember("Alpha crew owns incident response.")
    second.remember("Beta crew owns roadmap planning.")

    first.reset()

    assert first.recall("incident response", limit=3) == []
    second_matches = second.recall("roadmap planning", limit=3)
    assert second_matches
    assert second_matches[0].record.content == "Beta crew owns roadmap planning."
