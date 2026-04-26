"""Tests for SynaptExtraction IL v1 finalization pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "extract-py" / "src"))

from synapt_extract.finalize import finalize_extraction, FinalizeContext


def _llm_output(**overrides):
    base = {
        "extracted_at": "2026-04-26T00:00:00Z",
        "entities": [{"name": "Mom", "type": "person"}],
        "goals": [{"text": "Recovery", "status": "open", "entity_refs": []}],
        "themes": ["Health"],
    }
    base.update(overrides)
    return base


class TestStage2Injection:

    def test_injects_version(self):
        result = finalize_extraction(
            _llm_output(),
            FinalizeContext(produced_by="openai://gpt-4o-mini"),
        )
        assert result.extraction["version"] == "1"

    def test_injects_produced_by(self):
        result = finalize_extraction(
            _llm_output(),
            FinalizeContext(produced_by="openai://gpt-4o-mini"),
        )
        assert result.extraction["produced_by"] == "openai://gpt-4o-mini"

    def test_injects_user_id(self):
        result = finalize_extraction(
            _llm_output(),
            FinalizeContext(produced_by="test://model", user_id="u123"),
        )
        assert result.extraction["user_id"] == "u123"

    def test_injects_source_id(self):
        result = finalize_extraction(
            _llm_output(),
            FinalizeContext(produced_by="test://model", source_id="prayer-001"),
        )
        assert result.extraction["source_id"] == "prayer-001"

    def test_injects_kind(self):
        result = finalize_extraction(
            _llm_output(),
            FinalizeContext(produced_by="test://model", kind="conversa/prayer"),
        )
        assert result.extraction["kind"] == "conversa/prayer"

    def test_injects_extensions(self):
        result = finalize_extraction(
            _llm_output(),
            FinalizeContext(
                produced_by="test://model",
                extensions={"conversa/prayer": {"category": "Health"}},
            ),
        )
        assert result.extraction["extensions"]["conversa/prayer"]["category"] == "Health"


class TestStage2Embeddings:

    def test_injects_embedding_version(self):
        result = finalize_extraction(
            _llm_output(),
            FinalizeContext(
                produced_by="test://model",
                embeddings=[{
                    "vector": [0.1, 0.2, 0.3],
                    "model": "openai://text-embedding-3-small",
                    "input": "source",
                    "dimensions": 3,
                }],
            ),
        )
        emb = result.extraction["embeddings"][0]
        assert emb["version"] == "1"

    def test_auto_populates_dimensions(self):
        result = finalize_extraction(
            _llm_output(),
            FinalizeContext(
                produced_by="test://model",
                embeddings=[{
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "model": "test://emb",
                    "input": "source",
                }],
            ),
        )
        emb = result.extraction["embeddings"][0]
        assert emb["dimensions"] == 4


class TestStage3SubSchemaVersions:

    def test_injects_source_ref_version(self):
        result = finalize_extraction(
            _llm_output(entities=[{
                "name": "Mom",
                "type": "person",
                "source": {"snippet": "My mom"},
            }]),
            FinalizeContext(produced_by="test://model"),
        )
        assert result.extraction["entities"][0]["source"]["version"] == "1"

    def test_injects_signals_version(self):
        result = finalize_extraction(
            _llm_output(entities=[{
                "name": "Mom",
                "type": "person",
                "signals": {"confidence": 0.9},
            }]),
            FinalizeContext(produced_by="test://model"),
        )
        assert result.extraction["entities"][0]["signals"]["version"] == "1"

    def test_injects_temporal_ref_version(self):
        result = finalize_extraction(
            _llm_output(temporal_refs=[{"raw": "next week", "type": "range"}]),
            FinalizeContext(produced_by="test://model"),
        )
        assert result.extraction["temporal_refs"][0]["version"] == "1"

    def test_strips_empty_source(self):
        result = finalize_extraction(
            _llm_output(entities=[{
                "name": "Mom",
                "type": "person",
                "source": {},
            }]),
            FinalizeContext(produced_by="test://model"),
        )
        assert "source" not in result.extraction["entities"][0]

    def test_strips_version_only_signals(self):
        result = finalize_extraction(
            _llm_output(entities=[{
                "name": "Mom",
                "type": "person",
                "signals": {"version": "1"},
            }]),
            FinalizeContext(produced_by="test://model"),
        )
        assert "signals" not in result.extraction["entities"][0]

    def test_injects_goal_source_version(self):
        result = finalize_extraction(
            _llm_output(goals=[{
                "text": "Recovery",
                "status": "open",
                "entity_refs": [],
                "source": {"snippet": "I hope she recovers"},
            }]),
            FinalizeContext(produced_by="test://model"),
        )
        assert result.extraction["goals"][0]["source"]["version"] == "1"

    def test_injects_fact_source_version(self):
        result = finalize_extraction(
            _llm_output(facts=[{
                "text": "Surgery happened",
                "source": {"snippet": "Mom had surgery"},
            }]),
            FinalizeContext(produced_by="test://model"),
        )
        assert result.extraction["facts"][0]["source"]["version"] == "1"

    def test_injects_relation_signals_version(self):
        result = finalize_extraction(
            _llm_output(entities=[{
                "id": "e1",
                "name": "Mom",
                "type": "person",
                "relations": [{"target": "e2", "type": "parent_of", "signals": {"confidence": 0.8}}],
            }, {
                "id": "e2",
                "name": "Me",
                "type": "person",
            }]),
            FinalizeContext(produced_by="test://model"),
        )
        rel = result.extraction["entities"][0]["relations"][0]
        assert rel["signals"]["version"] == "1"


class TestStage3CapabilityDetection:

    def test_detects_entities(self):
        result = finalize_extraction(
            _llm_output(),
            FinalizeContext(produced_by="test://model"),
        )
        assert "entities" in result.extraction["capabilities"]

    def test_detects_entity_state(self):
        result = finalize_extraction(
            _llm_output(entities=[{
                "name": "Mom",
                "type": "person",
                "state": "recovering",
            }]),
            FinalizeContext(produced_by="test://model"),
        )
        assert "entity_state" in result.extraction["capabilities"]

    def test_detects_entity_ids(self):
        result = finalize_extraction(
            _llm_output(entities=[{
                "id": "e1",
                "name": "Mom",
                "type": "person",
            }]),
            FinalizeContext(produced_by="test://model"),
        )
        assert "entity_ids" in result.extraction["capabilities"]

    def test_detects_relations(self):
        result = finalize_extraction(
            _llm_output(entities=[{
                "id": "e1",
                "name": "Mom",
                "type": "person",
                "relations": [{"target": "e2", "type": "knows"}],
            }, {
                "id": "e2",
                "name": "Dad",
                "type": "person",
            }]),
            FinalizeContext(produced_by="test://model"),
        )
        assert "relations" in result.extraction["capabilities"]

    def test_detects_evidence_anchoring(self):
        result = finalize_extraction(
            _llm_output(entities=[{
                "name": "Mom",
                "type": "person",
                "source": {"snippet": "My mom"},
            }]),
            FinalizeContext(produced_by="test://model"),
        )
        assert "evidence_anchoring" in result.extraction["capabilities"]

    def test_detects_assertion_signals(self):
        result = finalize_extraction(
            _llm_output(entities=[{
                "name": "Mom",
                "type": "person",
                "signals": {"confidence": 0.9},
            }]),
            FinalizeContext(produced_by="test://model"),
        )
        assert "assertion_signals" in result.extraction["capabilities"]

    def test_detects_goal_timing(self):
        result = finalize_extraction(
            _llm_output(goals=[{
                "text": "Recovery",
                "status": "open",
                "entity_refs": [],
                "stated_at": "2026-04-20",
            }]),
            FinalizeContext(produced_by="test://model"),
        )
        assert "goal_timing" in result.extraction["capabilities"]

    def test_detects_summary_and_sentiment(self):
        result = finalize_extraction(
            _llm_output(summary="A prayer for healing.", sentiment="hopeful"),
            FinalizeContext(produced_by="test://model"),
        )
        caps = result.extraction["capabilities"]
        assert "summary" in caps
        assert "sentiment" in caps

    def test_detects_temporal_classes(self):
        result = finalize_extraction(
            _llm_output(temporal_refs=[{
                "raw": "April 20 to May 1",
                "type": "range",
                "resolved": "2026-04-20",
                "resolved_end": "2026-05-01",
            }]),
            FinalizeContext(produced_by="test://model"),
        )
        caps = result.extraction["capabilities"]
        assert "temporal_refs" in caps
        assert "temporal_classes" in caps


class TestStage3Warnings:

    def test_warns_on_hint_mismatch(self):
        result = finalize_extraction(
            _llm_output(),
            FinalizeContext(
                produced_by="test://model",
                capabilities_hint=["relations"],
            ),
        )
        assert any("relations" in w for w in result.warnings)

    def test_warns_on_goal_entity_refs_without_entity_ids(self):
        result = finalize_extraction(
            _llm_output(
                entities=[{"name": "Mom", "type": "person"}],
                goals=[{"text": "Recovery", "status": "open", "entity_refs": ["e1"]}],
            ),
            FinalizeContext(produced_by="test://model"),
        )
        assert any("entity_ids" in w for w in result.warnings)


class TestEndToEnd:

    def test_finalized_extraction_passes_validation(self):
        result = finalize_extraction(
            _llm_output(
                entities=[{
                    "id": "e1",
                    "name": "Mom",
                    "type": "person",
                    "state": "recovering",
                    "source": {"snippet": "My mom is recovering"},
                    "signals": {"confidence": 0.9},
                }],
                goals=[{
                    "text": "Full recovery",
                    "status": "open",
                    "entity_refs": ["e1"],
                    "stated_at": "2026-04-20",
                }],
                facts=[{"text": "Had surgery April 20", "source": {"snippet": "surgery"}}],
                summary="Prayer for mom's recovery.",
                sentiment="hopeful",
            ),
            FinalizeContext(
                produced_by="openai://gpt-4o-mini",
                user_id="user_123",
                source_id="prayer-001",
                kind="conversa/prayer",
                extensions={"conversa/prayer": {"category": "Health"}},
            ),
        )
        assert result.validation.valid, [f"{e.path}: {e.message}" for e in result.validation.errors]
        assert result.extraction["version"] == "1"
        assert result.extraction["produced_by"] == "openai://gpt-4o-mini"
        assert "entities" in result.extraction["capabilities"]
        assert "evidence_anchoring" in result.extraction["capabilities"]
