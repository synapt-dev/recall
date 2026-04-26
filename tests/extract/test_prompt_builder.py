from __future__ import annotations

import pytest

from tests.extract.conftest import load_symbol


def test_build_extraction_prompt_expands_standard_profile_with_context_preamble() -> None:
    build_extraction_prompt = load_symbol("prompt", "build_extraction_prompt")

    prompt = build_extraction_prompt(
        "I want to ship the package next week.",
        {
            "profile": "standard",
            "categories": ["Work", "Release"],
            "source_type": "session",
            "date": "2026-04-26",
        },
    )

    assert "Extract structured data from the following text." in prompt
    assert "Available categories: Work, Release" in prompt
    assert "Content type: session" in prompt
    assert "Date: 2026-04-26" in prompt
    assert '"facts": array of objects with "text" and optional "category"' in prompt
    assert '"temporal_refs": array with "raw" (as it appeared) and "resolved" (ISO 8601).' in prompt
    assert '"source": {"snippet": verbatim quote from text}' in prompt
    assert '"id": short local ID ("e1", "e2", etc.)' not in prompt
    assert '"produced_by"' not in prompt
    assert '"capabilities"' not in prompt
    assert '"version"' not in prompt


def test_build_extraction_prompt_closes_dependencies_and_uses_canonical_fragment_order() -> None:
    build_extraction_prompt = load_symbol("prompt", "build_extraction_prompt")

    prompt = build_extraction_prompt(
        "Sentinel and Apollo are shipping @synapt/extract.",
        {
            "capabilities": ["summary", "goal_entity_refs", "relation_origin"],
        },
    )

    entities_index = prompt.index('"entities": array of objects')
    goals_index = prompt.index('"goals": array of objects')
    summary_index = prompt.index('"summary": one sentence')
    entity_ids_index = prompt.index('"id": short local ID ("e1", "e2", etc.)')
    goal_refs_index = prompt.index('"entity_refs": array of entity IDs (not names)')
    relations_index = prompt.index('"relations": array of {"target": entity ID, "type": relationship type}')
    relation_origin_index = prompt.index('"origin": "explicit" (stated in text), "inferred" (deduced), or "dependent" (reverse edge)')

    assert entities_index < goals_index < summary_index < entity_ids_index
    assert entity_ids_index < goal_refs_index < relations_index < relation_origin_index
    assert "Assign each entity a short local ID" in prompt
    assert "Goals and relations reference entities by ID." in prompt


def test_build_extraction_prompt_rejects_unknown_capabilities() -> None:
    build_extraction_prompt = load_symbol("prompt", "build_extraction_prompt")

    with pytest.raises(Exception, match="unknown|capabilit"):
        build_extraction_prompt(
            "Test text",
            {
                "capabilities": ["entities", "not_a_real_capability"],
            },
        )
