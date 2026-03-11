"""Knowledge nodes — durable facts distilled from multiple sessions.

Storage: .synapt/recall/knowledge.jsonl (append-only, one JSON entry per node).
Each node represents a fact, pattern, or decision confirmed across sessions.
Nodes are also indexed in SQLite for FTS5 search.
"""

from __future__ import annotations

import fcntl
import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from synapt.recall.core import project_data_dir


VALID_CATEGORIES = frozenset({
    "workflow", "architecture", "infrastructure", "debugging",
    "convention", "tooling", "lesson-learned", "decision",
    "preference", "fact",
})

VALID_STATUSES = frozenset({"active", "stale", "contradicted"})


def _knowledge_path(project_dir: Path | None = None) -> Path:
    """Return path to knowledge.jsonl in the project's .synapt/recall/ dir."""
    return project_data_dir(project_dir) / "knowledge.jsonl"


def _new_id() -> str:
    """Generate a short unique ID for a knowledge node."""
    return uuid.uuid4().hex[:12]


@dataclass
class KnowledgeNode:
    """A durable fact distilled from multiple sessions."""

    id: str
    content: str                         # The fact (max ~300 chars)
    category: str                        # One of VALID_CATEGORIES
    confidence: float                    # 0.0-1.0
    source_sessions: list[str] = field(default_factory=list)
    source_turns: list[str] = field(default_factory=list)  # "session_id:turn_num"
    created_at: str = ""                 # ISO 8601
    updated_at: str = ""                 # ISO 8601
    status: str = "active"               # active | stale | contradicted
    superseded_by: str = ""              # ID of newer contradicting node
    contradiction_note: str = ""         # Why contradicted
    tags: list[str] = field(default_factory=list)
    valid_from: str | None = None        # ISO 8601: when this became true
    valid_until: str | None = None       # ISO 8601: when this stopped being true
    version: int = 1                     # Increments on supersession
    lineage_id: str = ""                 # Shared ID across versions of same fact

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> KnowledgeNode:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def create(
        cls,
        content: str,
        category: str,
        source_sessions: list[str] | None = None,
        confidence: float = 0.5,
        tags: list[str] | None = None,
        source_turns: list[str] | None = None,
    ) -> KnowledgeNode:
        """Create a new knowledge node with auto-generated ID and timestamps."""
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            id=_new_id(),
            content=content[:300],
            category=category if category in VALID_CATEGORIES else "workflow",
            confidence=max(0.0, min(1.0, confidence)),
            source_sessions=source_sessions or [],
            source_turns=source_turns or [],
            created_at=now,
            updated_at=now,
            tags=tags or [],
        )


def compute_confidence(source_count: int, age_days: float = 0.0) -> float:
    """Compute confidence score based on corroboration count and age.

    Base confidence increases with number of corroborating sessions,
    then decays with a 90-day half-life (much slower than search recency).

    1 session  -> 0.45
    2 sessions -> 0.60
    3 sessions -> 0.75
    5+ sessions -> 0.90 cap
    """
    base = min(0.9, 0.3 + 0.15 * max(1, source_count))
    if age_days <= 0:
        return base
    decay = math.exp(-math.log(2) / 90.0 * age_days)
    return base * decay


def append_node(node: KnowledgeNode, path: Path | None = None) -> Path:
    """Append a knowledge node to the JSONL file.

    Uses fcntl.flock for exclusive locking (same pattern as journal.py).
    """
    path = path or _knowledge_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(node.to_dict()) + "\n")
        f.flush()
    return path


def _read_all_nodes(path: Path) -> list[KnowledgeNode]:
    """Read every node from a knowledge JSONL file."""
    nodes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                nodes.append(KnowledgeNode.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return nodes


def read_nodes(
    path: Path | None = None,
    status: str | None = None,
) -> list[KnowledgeNode]:
    """Read knowledge nodes, optionally filtered by status.

    Deduplicates by id (keeps latest by updated_at).
    Returns sorted by confidence descending.
    """
    path = path or _knowledge_path()
    if not path.exists():
        return []
    raw = _read_all_nodes(path)
    deduped = _dedup_nodes(raw)
    if status:
        deduped = [n for n in deduped if n.status == status]
    deduped.sort(key=lambda n: n.confidence, reverse=True)
    return deduped


def _dedup_nodes(nodes: list[KnowledgeNode]) -> list[KnowledgeNode]:
    """Keep only the latest version of each node (by id, latest updated_at)."""
    best: dict[str, KnowledgeNode] = {}
    for node in nodes:
        existing = best.get(node.id)
        if existing is None or node.updated_at > existing.updated_at:
            best[node.id] = node
    return list(best.values())


def update_node(
    node_id: str,
    updates: dict,
    path: Path | None = None,
) -> bool:
    """Update a knowledge node by appending a modified version.

    Reads the current version, applies updates, appends the new version.
    Deduplication on read ensures only the latest version is used.

    Returns True if the node was found and updated.
    """
    path = path or _knowledge_path()
    if not path.exists():
        return False
    nodes = _dedup_nodes(_read_all_nodes(path))
    target = None
    for n in nodes:
        if n.id == node_id:
            target = n
            break
    if target is None:
        return False

    d = target.to_dict()
    d.update(updates)
    d["updated_at"] = datetime.now(timezone.utc).isoformat()
    updated = KnowledgeNode.from_dict(d)
    append_node(updated, path)
    return True


def compact_knowledge(path: Path | None = None) -> int:
    """Physically dedup knowledge.jsonl.

    Reads all entries, deduplicates by id (keeps latest updated_at),
    and rewrites the file atomically.

    Returns the number of duplicate entries removed.
    """
    path = path or _knowledge_path()
    if not path.exists():
        return 0
    raw = _read_all_nodes(path)
    deduped = _dedup_nodes(raw)
    removed = len(raw) - len(deduped)
    if removed == 0:
        return 0
    deduped.sort(key=lambda n: n.created_at)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for node in deduped:
            f.write(json.dumps(node.to_dict()) + "\n")
        f.flush()
    tmp.replace(path)
    return removed


def format_knowledge_for_display(nodes: list[KnowledgeNode]) -> str:
    """Format knowledge nodes for `recall consolidate --show` display."""
    if not nodes:
        return "No knowledge nodes found."
    lines = []
    for node in nodes:
        conf = f"{node.confidence:.0%}"
        status_tag = f" [{node.status}]" if node.status != "active" else ""
        lines.append(f"  [{node.category}] {node.content} ({conf}{status_tag})")
        if node.contradiction_note:
            lines.append(f"    ^ {node.contradiction_note}")
    return "\n".join(lines)


def format_knowledge_for_session_start(
    nodes: list[KnowledgeNode],
    max_nodes: int = 3,
) -> str:
    """Format top knowledge nodes for session-start hook output.

    Selects the top N active nodes by confidence, formats concisely.
    """
    active = [n for n in nodes if n.status == "active"]
    active.sort(key=lambda n: n.confidence, reverse=True)
    top = active[:max_nodes]
    if not top:
        return ""
    lines = ["Knowledge:"]
    for node in top:
        lines.append(f"  - [{node.category}] {node.content}")
    return "\n".join(lines)


def dedup_knowledge_nodes(
    threshold: float = 0.7,
    project_dir: Path | None = None,
) -> int:
    """Merge near-duplicate active knowledge nodes.

    Compares all active nodes pairwise using Jaccard similarity on
    extracted keywords. When similarity >= threshold, the older node
    is marked ``status="contradicted"`` with ``superseded_by`` pointing
    to the newer survivor. Source sessions from the duplicate are
    transferred to the survivor.

    Logs every merge decision to ``dedup_decisions.jsonl``.

    Returns the number of duplicate nodes merged.
    """
    from synapt.recall.consolidate import (
        _extract_keywords,
        _jaccard,
        _dedup_decisions_path,
        _log_dedup_decision,
    )

    kn_path = _knowledge_path(project_dir)
    if not kn_path.exists():
        return 0

    active = read_nodes(kn_path, status="active")
    if len(active) < 2:
        return 0

    # Pre-compute keyword sets
    kw_sets = {n.id: _extract_keywords(n.content) for n in active}

    # Track which node IDs have been absorbed (duplicate → survivor)
    absorbed: set[str] = set()
    merges: list[tuple[str, str]] = []  # (duplicate_id, survivor_id)

    # Pairwise comparison — O(n²) but n is small (tens of nodes)
    for i in range(len(active)):
        if active[i].id in absorbed:
            continue
        for j in range(i + 1, len(active)):
            if active[j].id in absorbed:
                continue
            sim = _jaccard(kw_sets[active[i].id], kw_sets[active[j].id])
            if sim >= threshold:
                # Keep the most recently updated node
                if active[i].updated_at >= active[j].updated_at:
                    survivor, duplicate = active[i], active[j]
                else:
                    survivor, duplicate = active[j], active[i]
                absorbed.add(duplicate.id)
                merges.append((duplicate.id, survivor.id))

                # Transfer source sessions (update in-memory too so
                # subsequent merges accumulate correctly)
                merged_sessions = list(
                    set(survivor.source_sessions + duplicate.source_sessions)
                )
                survivor.source_sessions = merged_sessions
                new_confidence = compute_confidence(len(merged_sessions))
                update_node(
                    survivor.id,
                    {
                        "source_sessions": merged_sessions,
                        "confidence": new_confidence,
                    },
                    kn_path,
                )

                # Mark duplicate as contradicted
                update_node(
                    duplicate.id,
                    {
                        "status": "contradicted",
                        "superseded_by": survivor.id,
                        "contradiction_note": f"Merged (jaccard={sim:.2f})",
                    },
                    kn_path,
                )

                # Log decision
                decision_path = _dedup_decisions_path(project_dir)
                _log_dedup_decision(
                    decision_path,
                    action="dedup-merge",
                    candidate_content=duplicate.content,
                    candidate_category=duplicate.category,
                    existing_id=survivor.id,
                    existing_content=survivor.content,
                    similarity_score=sim,
                    source="knowledge-dedup",
                )

                # If active[i] was the duplicate, stop comparing it
                # against remaining nodes (prevents triple-merge data loss)
                if active[i].id in absorbed:
                    break

    if merges:
        compact_knowledge(kn_path)

    return len(merges)
