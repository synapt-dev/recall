"""Tests for SynaptExtraction IL v1 validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "extract-py" / "src"))

from synapt_extract.validate import validate_extraction


def _minimal_extraction(**overrides):
    base = {
        "version": "1",
        "extracted_at": "2026-04-26T00:00:00Z",
        "produced_by": "openai://gpt-4o-mini",
        "entities": [],
        "goals": [],
        "themes": [],
        "capabilities": ["entities", "goals", "themes"],
    }
    base.update(overrides)
    return base


class TestValidExtraction:

    def test_minimal_valid(self):
        result = validate_extraction(_minimal_extraction())
        assert result.valid
        assert result.errors == []

    def test_full_extraction_valid(self):
        doc = _minimal_extraction(
            entities=[{
                "id": "e1",
                "name": "Mom",
                "type": "person",
                "state": "recovering",
                "context": "family member",
                "date_hint": "2026-04-20",
                "source": {"version": "1", "snippet": "My mom is recovering"},
                "signals": {"version": "1", "confidence": 0.9, "negated": False},
                "relations": [{"target": "e2", "type": "parent_of"}],
            }, {
                "id": "e2",
                "name": "Surgery",
                "type": "event",
            }],
            goals=[{
                "text": "Mom's full recovery",
                "status": "open",
                "entity_refs": ["e1"],
                "stated_at": "2026-04-20T10:00:00Z",
                "source": {"version": "1", "snippet": "I hope mom recovers"},
                "signals": {"version": "1", "hedged": True},
            }],
            themes=["Health", "Family"],
            summary="Prayer for mom's recovery after surgery.",
            sentiment="hopeful",
            facts=[{
                "text": "Mom had surgery on April 20",
                "category": "Health",
                "source": {"version": "1", "snippet": "Mom had surgery"},
            }],
            temporal_refs=[{
                "version": "1",
                "raw": "April 20",
                "type": "point",
                "resolved": "2026-04-20",
            }],
            capabilities=[
                "entities", "entity_state", "entity_context", "entity_ids",
                "goals", "goal_timing", "goal_entity_refs",
                "themes", "summary", "sentiment", "facts",
                "temporal_refs", "temporal_classes",
                "relations", "assertion_signals", "evidence_anchoring",
            ],
        )
        result = validate_extraction(doc)
        assert result.valid, [f"{e.path}: {e.message}" for e in result.errors]


class TestMissingRequiredFields:

    def test_missing_version(self):
        doc = _minimal_extraction()
        del doc["version"]
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "version" for e in result.errors)

    def test_missing_extracted_at(self):
        doc = _minimal_extraction()
        del doc["extracted_at"]
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "extracted_at" for e in result.errors)

    def test_missing_produced_by(self):
        doc = _minimal_extraction()
        del doc["produced_by"]
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "produced_by" for e in result.errors)

    def test_missing_entities(self):
        doc = _minimal_extraction()
        del doc["entities"]
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "entities" for e in result.errors)

    def test_missing_goals(self):
        doc = _minimal_extraction()
        del doc["goals"]
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "goals" for e in result.errors)

    def test_missing_themes(self):
        doc = _minimal_extraction()
        del doc["themes"]
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "themes" for e in result.errors)

    def test_missing_capabilities(self):
        doc = _minimal_extraction()
        del doc["capabilities"]
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "capabilities" for e in result.errors)

    def test_not_an_object(self):
        result = validate_extraction("not an object")
        assert not result.valid
        assert result.errors[0].message == "must be an object"

    def test_null(self):
        result = validate_extraction(None)
        assert not result.valid


class TestEntityValidation:

    def test_entity_missing_name(self):
        doc = _minimal_extraction(entities=[{"type": "person"}])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("name" in e.path for e in result.errors)

    def test_entity_missing_type(self):
        doc = _minimal_extraction(entities=[{"name": "Mom"}])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("type" in e.path for e in result.errors)

    def test_entity_bad_source_ref_version(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "source": {"version": "2", "snippet": "test"},
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("source.version" in e.path for e in result.errors)

    def test_entity_bad_signals_confidence(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "signals": {"version": "1", "confidence": 1.5},
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("confidence" in e.path for e in result.errors)

    def test_entity_relation_missing_target(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "relations": [{"type": "knows"}],
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("target" in e.path for e in result.errors)


class TestGoalValidation:

    def test_goal_missing_text(self):
        doc = _minimal_extraction(goals=[{
            "status": "open",
            "entity_refs": [],
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("text" in e.path for e in result.errors)

    def test_goal_invalid_status(self):
        doc = _minimal_extraction(goals=[{
            "text": "recover",
            "status": "pending",
            "entity_refs": [],
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("status" in e.path for e in result.errors)

    def test_goal_missing_entity_refs(self):
        doc = _minimal_extraction(goals=[{
            "text": "recover",
            "status": "open",
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("entity_refs" in e.path for e in result.errors)


class TestCapabilityValidation:

    def test_unknown_capability(self):
        doc = _minimal_extraction(capabilities=["entities", "psychic_powers"])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("psychic_powers" in e.message for e in result.errors)

    def test_all_valid_capabilities(self):
        from synapt_extract.schema import EXTRACTION_CAPABILITIES
        doc = _minimal_extraction(capabilities=sorted(EXTRACTION_CAPABILITIES))
        result = validate_extraction(doc)
        assert result.valid


class TestEmbeddingValidation:

    def test_valid_embedding(self):
        doc = _minimal_extraction(embeddings=[{
            "version": "1",
            "vector": [0.1, 0.2, 0.3],
            "model": "openai://text-embedding-3-small",
            "input": "source",
            "dimensions": 3,
        }])
        result = validate_extraction(doc)
        assert result.valid

    def test_embedding_missing_vector(self):
        doc = _minimal_extraction(embeddings=[{
            "version": "1",
            "model": "openai://text-embedding-3-small",
            "input": "source",
            "dimensions": 3,
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("vector" in e.path for e in result.errors)


class TestTemporalRefValidation:

    def test_valid_temporal_ref(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1",
            "raw": "next Tuesday",
            "type": "point",
            "resolved": "2026-04-28",
        }])
        result = validate_extraction(doc)
        assert result.valid

    def test_invalid_temporal_type(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1",
            "raw": "sometime",
            "type": "vague",
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("type" in e.path for e in result.errors)


class TestJsonSchemaFiles:

    def test_all_schema_files_are_valid_json(self):
        schema_dir = Path(__file__).resolve().parents[2] / "schemas"
        for schema_file in schema_dir.rglob("*.json"):
            content = schema_file.read_text()
            parsed = json.loads(content)
            assert "$schema" in parsed, f"{schema_file.name} missing $schema"
            assert "$id" in parsed, f"{schema_file.name} missing $id"

    def test_extraction_schema_references_sub_schemas(self):
        schema_path = Path(__file__).resolve().parents[2] / "schemas" / "extraction" / "v1.json"
        schema = json.loads(schema_path.read_text())
        schema_str = json.dumps(schema)
        assert "source-ref/v1.json" in schema_str
        assert "embedding/v1.json" in schema_str
        assert "assertion-signals/v1.json" in schema_str
        assert "temporal-ref/v1.json" in schema_str

    def test_extraction_schema_required_fields(self):
        schema_path = Path(__file__).resolve().parents[2] / "schemas" / "extraction" / "v1.json"
        schema = json.loads(schema_path.read_text())
        required = schema["required"]
        assert "version" in required
        assert "extracted_at" in required
        assert "produced_by" in required
        assert "entities" in required
        assert "goals" in required
        assert "themes" in required
        assert "capabilities" in required
