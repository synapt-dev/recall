from __future__ import annotations

from tests.extract.conftest import clone_doc, error_text, load_symbol, validation_errors


def test_validate_extraction_accepts_a_minimal_valid_document(
    minimal_extraction: dict[str, object],
) -> None:
    validate_extraction = load_symbol("validate", "validate_extraction")

    result = validate_extraction(clone_doc(minimal_extraction))

    assert validation_errors(result) == []


def test_validate_extraction_reports_missing_required_root_fields() -> None:
    validate_extraction = load_symbol("validate", "validate_extraction")

    result = validate_extraction({})
    errors = error_text(result)

    assert "version" in errors
    assert "extracted_at" in errors
    assert "produced_by" in errors
    assert "entities" in errors
    assert "goals" in errors
    assert "themes" in errors
    assert "capabilities" in errors


def test_validate_extraction_rejects_wrong_root_version(
    minimal_extraction: dict[str, object],
) -> None:
    validate_extraction = load_symbol("validate", "validate_extraction")
    extraction = clone_doc(minimal_extraction)
    extraction["version"] = "2"

    result = validate_extraction(extraction)

    assert "version" in error_text(result)


def test_validate_extraction_rejects_empty_strings_in_open_text_fields(
    minimal_extraction: dict[str, object],
) -> None:
    validate_extraction = load_symbol("validate", "validate_extraction")
    extraction = clone_doc(minimal_extraction)
    extraction["entities"] = [{"id": "e1", "name": "", "type": ""}]
    extraction["goals"] = [{"text": "", "status": "open", "entity_refs": ["e1"]}]
    extraction["themes"] = [""]
    extraction["summary"] = ""

    result = validate_extraction(extraction)
    errors = error_text(result)

    assert "name" in errors
    assert "type" in errors
    assert "text" in errors
    assert "themes" in errors
    assert "summary" in errors


def test_validate_extraction_rejects_empty_wrapper_subschemas_and_dangling_refs(
    minimal_extraction: dict[str, object],
) -> None:
    validate_extraction = load_symbol("validate", "validate_extraction")
    extraction = clone_doc(minimal_extraction)
    extraction["entities"] = [
        {
            "id": "e1",
            "name": "Sentinel",
            "type": "person",
            "source": {"version": "1"},
            "relations": [{"target": "e404", "type": "works_with"}],
        }
    ]
    extraction["goals"] = [
        {
            "text": "Ship the extract package",
            "status": "open",
            "entity_refs": ["e404"],
            "signals": {"version": "1"},
        }
    ]

    result = validate_extraction(extraction)
    errors = error_text(result)

    assert "source" in errors
    assert "signals" in errors
    assert "entity_refs" in errors
    assert "target" in errors


def test_validate_extraction_enforces_temporal_ref_conditional_rules(
    minimal_extraction: dict[str, object],
) -> None:
    validate_extraction = load_symbol("validate", "validate_extraction")
    extraction = clone_doc(minimal_extraction)
    extraction["temporal_refs"] = [
        {
            "version": "1",
            "raw": "between Monday and Tuesday",
            "type": "range",
            "resolved": "2026-04-27T00:00:00Z",
        },
        {
            "version": "1",
            "raw": "sometime soon",
            "type": "unresolved",
            "resolved": "2026-05-01T00:00:00Z",
        },
    ]
    extraction["capabilities"] = [*extraction["capabilities"], "temporal_refs", "temporal_classes"]  # type: ignore[index]

    result = validate_extraction(extraction)
    errors = error_text(result)

    assert "resolved_end" in errors
    assert "unresolved" in errors


def test_validate_extraction_does_not_mutate_input_document(
    minimal_extraction: dict[str, object],
) -> None:
    validate_extraction = load_symbol("validate", "validate_extraction")
    extraction = clone_doc(minimal_extraction)
    original = clone_doc(extraction)

    validate_extraction(extraction)

    assert extraction == original
