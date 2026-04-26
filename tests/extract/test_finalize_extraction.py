from __future__ import annotations

import pytest

from tests.extract.conftest import clone_doc, load_symbol


def test_finalize_extraction_runs_stage2_and_stage3_pipeline(
    stage1_output: dict[str, object],
) -> None:
    finalize_extraction = load_symbol("finalize", "finalize_extraction")

    result = finalize_extraction(
        clone_doc(stage1_output),
        {
            "produced_by": "openai://gpt-4o-mini",
            "user_id": "user-123",
            "source_id": "doc-456",
            "source_type": "session",
            "kind": "synapt/session_summary",
            "profile": "minimal",
            "extensions": {
                "synapt/session_summary": {
                    "focus": "Ship extract package",
                    "done": ["Wrote red specs"],
                }
            },
            "embeddings": [
                {
                    "vector": [0.25, -0.5, 0.75],
                    "model": "openai://text-embedding-3-small",
                    "input": "source",
                }
            ],
        },
    )

    assert result["version"] == "1"
    assert result["produced_by"] == "openai://gpt-4o-mini"
    assert result["user_id"] == "user-123"
    assert result["source_id"] == "doc-456"
    assert result["source_type"] == "session"
    assert result["kind"] == "synapt/session_summary"

    assert set(result["capabilities"]) == {
        "entities",
        "entity_state",
        "entity_context",
        "entity_ids",
        "goals",
        "goal_entity_refs",
        "themes",
        "summary",
        "facts",
        "temporal_refs",
        "relations",
        "relation_origin",
    }
    assert "fake" not in result["capabilities"]

    first_entity = result["entities"][0]
    first_relation = first_entity["relations"][0]
    first_goal = result["goals"][0]
    first_fact = result["facts"][0]
    first_temporal = result["temporal_refs"][0]
    first_embedding = result["embeddings"][0]
    extension = result["extensions"]["synapt/session_summary"]

    assert "source" not in first_entity
    assert "signals" not in first_relation
    assert "source" not in first_goal
    assert "source" not in first_fact

    assert first_temporal["version"] == "1"
    assert first_embedding["version"] == "1"
    assert first_embedding["dimensions"] == 3
    assert extension["version"] == "1"


def test_finalize_extraction_prefers_observed_payload_over_requested_profile(
    stage1_output: dict[str, object],
) -> None:
    finalize_extraction = load_symbol("finalize", "finalize_extraction")

    result = finalize_extraction(
        clone_doc(stage1_output),
        {
            "produced_by": "openai://gpt-4o-mini",
            "profile": "minimal",
        },
    )

    assert "facts" in result["capabilities"]
    assert "relations" in result["capabilities"]
    assert "relation_origin" in result["capabilities"]
    assert "summary" in result["capabilities"]


def test_finalize_extraction_rejects_invalid_final_cross_references(
    stage1_output: dict[str, object],
) -> None:
    finalize_extraction = load_symbol("finalize", "finalize_extraction")
    broken_output = clone_doc(stage1_output)
    broken_output["goals"] = [
        {
            "text": "Ship @synapt/extract",
            "status": "open",
            "entity_refs": ["e404"],
        }
    ]

    with pytest.raises(Exception, match="entity_refs|e404"):
        finalize_extraction(
            broken_output,
            {
                "produced_by": "openai://gpt-4o-mini",
            },
        )


def test_finalize_extraction_rejects_malformed_embeddings_instead_of_stripping_them(
    stage1_output: dict[str, object],
) -> None:
    finalize_extraction = load_symbol("finalize", "finalize_extraction")

    with pytest.raises(Exception, match="embedding|model|dimensions|vector"):
        finalize_extraction(
            clone_doc(stage1_output),
            {
                "produced_by": "openai://gpt-4o-mini",
                "embeddings": [{"input": "source"}],
            },
        )
