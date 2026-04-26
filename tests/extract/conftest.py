from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


def load_extract_module(module_name: str):
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    try:
        return importlib.import_module(f"synapt_extract.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name in {"synapt_extract", f"synapt_extract.{module_name}"}:
            pytest.fail(
                "Expected Python package `src/synapt_extract/` with module "
                f"`{module_name}.py`, but it does not exist yet."
            )
        raise


def load_symbol(module_name: str, symbol_name: str):
    module = load_extract_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError:
        pytest.fail(
            f"Expected `synapt_extract.{module_name}.{symbol_name}`, "
            "but the symbol is missing."
        )


def validation_errors(result: object) -> list[object]:
    if isinstance(result, dict):
        errors = result.get("errors")
    else:
        errors = getattr(result, "errors", None)

    assert isinstance(errors, list), (
        "ValidationResult must expose an `errors` list, either as a dict key "
        "or an object attribute."
    )
    return errors


def error_text(result: object) -> str:
    return "\n".join(str(item) for item in validation_errors(result))


@pytest.fixture
def minimal_extraction() -> dict[str, object]:
    return {
        "version": "1",
        "extracted_at": "2026-04-26T12:00:00Z",
        "produced_by": "openai://gpt-4o-mini",
        "entities": [
            {
                "id": "e1",
                "name": "Sentinel",
                "type": "person",
            }
        ],
        "goals": [
            {
                "text": "Ship the extract package",
                "status": "open",
                "entity_refs": ["e1"],
            }
        ],
        "themes": ["Engineering"],
        "capabilities": [
            "entities",
            "entity_ids",
            "goals",
            "goal_entity_refs",
            "themes",
        ],
    }


@pytest.fixture
def stage1_output() -> dict[str, object]:
    return {
        "version": "999",
        "produced_by": "llm://should-not-win",
        "capabilities": ["fake"],
        "extracted_at": "2026-04-26T12:00:00Z",
        "entities": [
            {
                "id": "e1",
                "name": "Sentinel",
                "type": "person",
                "state": "writing tests",
                "source": {},
                "relations": [
                    {
                        "target": "e2",
                        "type": "works_with",
                        "origin": "explicit",
                        "signals": {},
                    }
                ],
            },
            {
                "id": "e2",
                "name": "Apollo",
                "type": "person",
                "context": "Implementer",
            },
        ],
        "goals": [
            {
                "text": "Ship @synapt/extract",
                "status": "in_progress",
                "entity_refs": ["e1", "e2"],
                "source": {},
            }
        ],
        "themes": ["Engineering"],
        "summary": "Sentinel wrote the contract tests.",
        "facts": [
            {
                "text": "The IL spec is locked.",
                "category": "release",
                "source": {},
            }
        ],
        "temporal_refs": [
            {
                "raw": "today",
                "resolved": "2026-04-26T00:00:00Z",
            }
        ],
    }


def clone_doc(doc: dict[str, object]) -> dict[str, object]:
    return copy.deepcopy(doc)
