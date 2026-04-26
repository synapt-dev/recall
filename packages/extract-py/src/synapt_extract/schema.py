"""SynaptExtraction IL v1 type definitions."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class SynaptSourceRef(TypedDict, total=False):
    version: str
    snippet: str
    offset_start: int
    offset_end: int
    sentence_index: int


class SynaptEmbedding(TypedDict, total=False):
    version: str
    vector: list[float]
    model: str
    input: str
    dimensions: int
    space: str
    computed_at: str


class SynaptAssertionSignals(TypedDict, total=False):
    version: str
    confidence: float
    negated: bool
    hedged: bool
    condition: str


class SynaptTemporalRef(TypedDict, total=False):
    version: str
    raw: str
    type: Literal["point", "range", "duration", "unresolved"]
    resolved: str
    resolved_end: str
    context: str


class SynaptRelation(TypedDict, total=False):
    target: str
    type: str
    origin: str
    signals: SynaptAssertionSignals


class SynaptEntity(TypedDict, total=False):
    id: str
    name: str
    type: str
    state: str
    context: str
    date_hint: str
    source: SynaptSourceRef
    signals: SynaptAssertionSignals
    relations: list[SynaptRelation]


class SynaptGoal(TypedDict, total=False):
    text: str
    status: Literal["open", "resolved", "abandoned", "in_progress"]
    entity_refs: list[str]
    stated_at: str
    resolved_at: str
    source: SynaptSourceRef
    signals: SynaptAssertionSignals


class SynaptFact(TypedDict, total=False):
    text: str
    category: str
    source: SynaptSourceRef
    signals: SynaptAssertionSignals


EXTRACTION_CAPABILITIES = frozenset([
    "entities", "entity_state", "entity_context", "entity_ids",
    "goals", "goal_timing", "goal_entity_refs",
    "themes", "summary", "sentiment", "facts",
    "temporal_refs", "temporal_classes",
    "relations", "relation_origin",
    "assertion_signals", "evidence_anchoring",
])


class SynaptExtraction(TypedDict, total=False):
    version: str
    extracted_at: str
    source_id: str
    source_type: str
    user_id: str
    produced_by: str
    kind: str
    entities: list[SynaptEntity]
    goals: list[SynaptGoal]
    themes: list[str]
    sentiment: str
    summary: str
    facts: list[SynaptFact]
    temporal_refs: list[SynaptTemporalRef]
    capabilities: list[str]
    embeddings: list[SynaptEmbedding]
    extensions: dict[str, Any]
