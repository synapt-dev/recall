"""Three-stage finalization pipeline for SynaptExtraction IL v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from synapt_extract.validate import ValidationResult, validate_extraction


@dataclass
class FinalizeContext:
    produced_by: str
    user_id: str | None = None
    source_id: str | None = None
    source_type: str | None = None
    kind: str | None = None
    extensions: dict[str, Any] | None = None
    embeddings: list[dict[str, Any]] | None = None
    capabilities_hint: list[str] | None = None


@dataclass
class FinalizeResult:
    extraction: dict[str, Any]
    validation: ValidationResult
    warnings: list[str] = field(default_factory=list)


def _has_payload_beyond_version(obj: dict[str, Any]) -> bool:
    return any(k != "version" for k in obj)


def _detect_capabilities(doc: dict[str, Any]) -> list[str]:
    caps: list[str] = []
    entities = doc.get("entities", [])
    goals = doc.get("goals", [])

    if isinstance(entities, list) and entities:
        caps.append("entities")
        if any(e.get("state") is not None for e in entities):
            caps.append("entity_state")
        if any(e.get("context") is not None or e.get("date_hint") is not None for e in entities):
            caps.append("entity_context")
        if any(e.get("id") is not None for e in entities):
            caps.append("entity_ids")
        all_rels = [r for e in entities for r in (e.get("relations") or [])]
        if all_rels:
            caps.append("relations")
            if any(r.get("origin") is not None for r in all_rels):
                caps.append("relation_origin")
        if any(e.get("source") is not None for e in entities):
            caps.append("evidence_anchoring")
        if any(e.get("signals") is not None for e in entities):
            caps.append("assertion_signals")

    if isinstance(goals, list) and goals:
        caps.append("goals")
        if any(g.get("stated_at") is not None or g.get("resolved_at") is not None for g in goals):
            caps.append("goal_timing")
        if any(isinstance(g.get("entity_refs"), list) and g["entity_refs"] for g in goals):
            caps.append("goal_entity_refs")
        if "evidence_anchoring" not in caps and any(g.get("source") is not None for g in goals):
            caps.append("evidence_anchoring")
        if "assertion_signals" not in caps and any(g.get("signals") is not None for g in goals):
            caps.append("assertion_signals")

    themes = doc.get("themes")
    if isinstance(themes, list) and themes:
        caps.append("themes")
    if isinstance(doc.get("summary"), str):
        caps.append("summary")
    if isinstance(doc.get("sentiment"), str):
        caps.append("sentiment")

    facts = doc.get("facts", [])
    if isinstance(facts, list) and facts:
        caps.append("facts")
        if "evidence_anchoring" not in caps and any(f.get("source") is not None for f in facts):
            caps.append("evidence_anchoring")
        if "assertion_signals" not in caps and any(f.get("signals") is not None for f in facts):
            caps.append("assertion_signals")

    temporal = doc.get("temporal_refs", [])
    if isinstance(temporal, list) and temporal:
        caps.append("temporal_refs")
        if any(r.get("type") is not None or r.get("resolved_end") is not None for r in temporal):
            caps.append("temporal_classes")

    return caps


def _inject_sub_versions(obj: dict[str, Any]) -> None:
    obj["version"] = "1"


def finalize_extraction(
    llm_output: dict[str, Any],
    context: FinalizeContext,
) -> FinalizeResult:
    """Assemble a complete SynaptExtraction from LLM output and client context.

    Stage 1 (llm_output): content fields from the LLM.
    Stage 2: injects client context (produced_by, user_id, embeddings, etc.).
    Stage 3: injects sub-schema versions, computes capabilities, validates.
    """
    warnings: list[str] = []
    doc = dict(llm_output)

    # Stage 2: inject client context
    doc["version"] = "1"
    doc["produced_by"] = context.produced_by
    if context.user_id is not None:
        doc["user_id"] = context.user_id
    if context.source_id is not None:
        doc["source_id"] = context.source_id
    if context.source_type is not None:
        doc["source_type"] = context.source_type
    if context.kind is not None:
        doc["kind"] = context.kind
    if context.extensions is not None:
        doc["extensions"] = context.extensions

    if context.embeddings is not None:
        finalized_embs = []
        for emb in context.embeddings:
            full = {"version": "1", **emb}
            if "dimensions" not in full and isinstance(full.get("vector"), list):
                full["dimensions"] = len(full["vector"])
            finalized_embs.append(full)
        doc["embeddings"] = finalized_embs

    # Stage 3: inject sub-schema versions, strip empty sub-schemas
    for items_key in ("entities", "goals", "facts"):
        items = doc.get(items_key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for sub_key in ("source", "signals"):
                sub = item.get(sub_key)
                if isinstance(sub, dict):
                    if _has_payload_beyond_version(sub):
                        _inject_sub_versions(sub)
                    else:
                        del item[sub_key]
            if items_key == "entities" and isinstance(item.get("relations"), list):
                for rel in item["relations"]:
                    if isinstance(rel, dict) and isinstance(rel.get("signals"), dict):
                        sig = rel["signals"]
                        if _has_payload_beyond_version(sig):
                            _inject_sub_versions(sig)
                        else:
                            del rel["signals"]

    temporal = doc.get("temporal_refs")
    if isinstance(temporal, list):
        for ref in temporal:
            if isinstance(ref, dict):
                _inject_sub_versions(ref)

    # Stage 3: compute capabilities from observed payload
    observed = _detect_capabilities(doc)
    if context.capabilities_hint:
        for hinted in context.capabilities_hint:
            if hinted not in observed:
                warnings.append(
                    f'capabilities_hint includes "{hinted}" but payload does not contain it; using observed'
                )
    doc["capabilities"] = observed

    # Stage 3: capability implication warnings
    if "goal_entity_refs" in observed and "entity_ids" not in observed:
        warnings.append(
            "goal_entity_refs present but entity_ids missing; "
            "entity ref resolution will fall back to name matching"
        )

    # Stage 3: validate
    validation = validate_extraction(doc)

    return FinalizeResult(
        extraction=doc,
        validation=validation,
        warnings=warnings,
    )
