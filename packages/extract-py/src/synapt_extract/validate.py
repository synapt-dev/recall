"""Structural validation for SynaptExtraction IL v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from synapt_extract.schema import EXTRACTION_CAPABILITIES

VALID_GOAL_STATUSES = frozenset(["open", "resolved", "abandoned", "in_progress"])
VALID_TEMPORAL_TYPES = frozenset(["point", "range", "duration", "unresolved"])


@dataclass
class ValidationError:
    path: str
    message: str


@dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)


def _check_source_ref(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    if obj.get("version") != "1":
        errors.append(ValidationError(f"{path}.version", 'must be "1"'))
    if "snippet" in obj and not isinstance(obj["snippet"], str):
        errors.append(ValidationError(f"{path}.snippet", "must be a string"))


def _check_signals(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    if obj.get("version") != "1":
        errors.append(ValidationError(f"{path}.version", 'must be "1"'))
    if "confidence" in obj:
        c = obj["confidence"]
        if not isinstance(c, (int, float)) or c < 0 or c > 1:
            errors.append(ValidationError(f"{path}.confidence", "must be a number between 0.0 and 1.0"))
    if "negated" in obj and not isinstance(obj["negated"], bool):
        errors.append(ValidationError(f"{path}.negated", "must be a boolean"))
    if "hedged" in obj and not isinstance(obj["hedged"], bool):
        errors.append(ValidationError(f"{path}.hedged", "must be a boolean"))
    if "condition" in obj and not isinstance(obj["condition"], str):
        errors.append(ValidationError(f"{path}.condition", "must be a string"))


def _check_embedding(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    if obj.get("version") != "1":
        errors.append(ValidationError(f"{path}.version", 'must be "1"'))
    if not isinstance(obj.get("vector"), list):
        errors.append(ValidationError(f"{path}.vector", "required array"))
    if not isinstance(obj.get("model"), str):
        errors.append(ValidationError(f"{path}.model", "required string"))
    if not isinstance(obj.get("input"), str):
        errors.append(ValidationError(f"{path}.input", "required string"))
    dims = obj.get("dimensions")
    if not isinstance(dims, int) or dims < 1:
        errors.append(ValidationError(f"{path}.dimensions", "required positive integer"))


def _check_relation(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    if not isinstance(obj.get("target"), str):
        errors.append(ValidationError(f"{path}.target", "required string"))
    if not isinstance(obj.get("type"), str):
        errors.append(ValidationError(f"{path}.type", "required string"))
    if "signals" in obj:
        _check_signals(obj["signals"], f"{path}.signals", errors)


def _check_entity(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    if not isinstance(obj.get("name"), str):
        errors.append(ValidationError(f"{path}.name", "required string"))
    if not isinstance(obj.get("type"), str):
        errors.append(ValidationError(f"{path}.type", "required string"))
    if "source" in obj:
        _check_source_ref(obj["source"], f"{path}.source", errors)
    if "signals" in obj:
        _check_signals(obj["signals"], f"{path}.signals", errors)
    if "relations" in obj:
        if not isinstance(obj["relations"], list):
            errors.append(ValidationError(f"{path}.relations", "must be an array"))
        else:
            for i, rel in enumerate(obj["relations"]):
                _check_relation(rel, f"{path}.relations[{i}]", errors)


def _check_goal(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    if not isinstance(obj.get("text"), str):
        errors.append(ValidationError(f"{path}.text", "required string"))
    status = obj.get("status")
    if not isinstance(status, str) or status not in VALID_GOAL_STATUSES:
        errors.append(ValidationError(f"{path}.status", "must be one of: open, resolved, abandoned, in_progress"))
    if not isinstance(obj.get("entity_refs"), list):
        errors.append(ValidationError(f"{path}.entity_refs", "required array of strings"))
    if "source" in obj:
        _check_source_ref(obj["source"], f"{path}.source", errors)
    if "signals" in obj:
        _check_signals(obj["signals"], f"{path}.signals", errors)


def _check_fact(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    if not isinstance(obj.get("text"), str):
        errors.append(ValidationError(f"{path}.text", "required string"))
    if "source" in obj:
        _check_source_ref(obj["source"], f"{path}.source", errors)
    if "signals" in obj:
        _check_signals(obj["signals"], f"{path}.signals", errors)


def _check_temporal_ref(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    if obj.get("version") != "1":
        errors.append(ValidationError(f"{path}.version", 'must be "1"'))
    if not isinstance(obj.get("raw"), str):
        errors.append(ValidationError(f"{path}.raw", "required string"))
    if "type" in obj:
        if not isinstance(obj["type"], str) or obj["type"] not in VALID_TEMPORAL_TYPES:
            errors.append(ValidationError(f"{path}.type", "must be one of: point, range, duration, unresolved"))


def validate_extraction(obj: Any) -> ValidationResult:
    """Validate a SynaptExtraction document for structural conformance.

    Returns a ValidationResult with valid=True if the document conforms
    to the IL v1 schema, or valid=False with a list of errors.
    """
    errors: list[ValidationError] = []

    if not isinstance(obj, dict):
        return ValidationResult(valid=False, errors=[ValidationError("", "must be an object")])

    if obj.get("version") != "1":
        errors.append(ValidationError("version", 'must be "1"'))
    if not isinstance(obj.get("extracted_at"), str):
        errors.append(ValidationError("extracted_at", "required string (ISO 8601)"))
    if not isinstance(obj.get("produced_by"), str):
        errors.append(ValidationError("produced_by", "required string (provider URI)"))

    if not isinstance(obj.get("entities"), list):
        errors.append(ValidationError("entities", "required array"))
    else:
        for i, ent in enumerate(obj["entities"]):
            _check_entity(ent, f"entities[{i}]", errors)

    if not isinstance(obj.get("goals"), list):
        errors.append(ValidationError("goals", "required array"))
    else:
        for i, goal in enumerate(obj["goals"]):
            _check_goal(goal, f"goals[{i}]", errors)

    if not isinstance(obj.get("themes"), list):
        errors.append(ValidationError("themes", "required array"))

    if not isinstance(obj.get("capabilities"), list):
        errors.append(ValidationError("capabilities", "required array"))
    else:
        for i, cap in enumerate(obj["capabilities"]):
            if not isinstance(cap, str):
                errors.append(ValidationError(f"capabilities[{i}]", "must be a string"))
            elif cap not in EXTRACTION_CAPABILITIES:
                errors.append(ValidationError(f"capabilities[{i}]", f'unknown capability: "{cap}"'))

    if "facts" in obj:
        if not isinstance(obj["facts"], list):
            errors.append(ValidationError("facts", "must be an array"))
        else:
            for i, fact in enumerate(obj["facts"]):
                _check_fact(fact, f"facts[{i}]", errors)

    if "temporal_refs" in obj:
        if not isinstance(obj["temporal_refs"], list):
            errors.append(ValidationError("temporal_refs", "must be an array"))
        else:
            for i, ref in enumerate(obj["temporal_refs"]):
                _check_temporal_ref(ref, f"temporal_refs[{i}]", errors)

    if "embeddings" in obj:
        if not isinstance(obj["embeddings"], list):
            errors.append(ValidationError("embeddings", "must be an array"))
        else:
            for i, emb in enumerate(obj["embeddings"]):
                _check_embedding(emb, f"embeddings[{i}]", errors)

    return ValidationResult(valid=len(errors) == 0, errors=errors)
