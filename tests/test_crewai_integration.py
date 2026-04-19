from __future__ import annotations

from pathlib import Path

import pytest

crewai = pytest.importorskip("crewai")
from crewai.memory.storage.backend import MemoryRecord

from synapt.integrations import crewai as synapt_crewai


def test_synapt_storage_saves_and_searches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved: list[tuple[str, str, bool]] = []

    def fake_recall_save(content: str = "", category: str = "workflow", confidence: float = 0.8, tags=None, source_sessions=None, source_turns=None, node_id: str | None = None, retract: bool = False) -> str:
        saved.append((node_id or "", content, retract))
        return "ok"

    monkeypatch.setattr(synapt_crewai, "recall_save", fake_recall_save)
    storage = synapt_crewai.SynaptStorage(path=tmp_path / "crewai.jsonl")
    record = MemoryRecord(content="CrewAI adapter stored in recall", scope="/crew/test", categories=["integration"], metadata={"kind": "note"}, importance=0.9, embedding=[1.0, 0.0])

    storage.save([record])
    results = storage.search([1.0, 0.0], scope_prefix="/crew", categories=["integration"], metadata_filter={"kind": "note"})

    assert saved == [(record.id, record.content, False)]
    assert len(results) == 1
    assert results[0][0].id == record.id


def test_synapt_storage_delete_retracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    retracted: list[str] = []

    def fake_recall_save(content: str = "", category: str = "workflow", confidence: float = 0.8, tags=None, source_sessions=None, source_turns=None, node_id: str | None = None, retract: bool = False) -> str:
        if retract and node_id:
            retracted.append(node_id)
        return "ok"

    monkeypatch.setattr(synapt_crewai, "recall_save", fake_recall_save)
    storage = synapt_crewai.SynaptStorage(path=tmp_path / "crewai.jsonl")
    record = MemoryRecord(content="delete me", embedding=[0.0, 1.0])
    storage.save([record])

    assert storage.delete(record_ids=[record.id]) == 1
    assert retracted == [record.id]
    assert storage.count() == 0
