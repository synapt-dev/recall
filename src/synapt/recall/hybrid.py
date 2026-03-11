"""Hybrid search: BM25 + embedding fusion via Reciprocal Rank Fusion.

Replaces the additive embedding boost (score + sim * 3.0) with proper
rank-level fusion that can surface results that BM25 missed entirely.
This is the single biggest quality improvement for paraphrased queries.

RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each ranking i.
See Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
individual Rank Learning Methods" (SIGIR 2009).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# RRF constant — controls how much top ranks dominate.
# k=60 is the standard value from the original paper.
RRF_K = 60

# Minimum cosine similarity to include in embedding results.
# Below this, the match is too noisy to be useful.
EMBEDDING_SIM_THRESHOLD = 0.25

# When BM25 returns fewer than this many results, boost embedding weight
# in the fusion to compensate.
SPARSE_RESULT_THRESHOLD = 3


def rrf_merge(
    *ranked_lists: list[tuple[int, float]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Each input is a list of (item_id, score) sorted by score descending.
    Returns a merged list of (item_id, rrf_score) sorted by rrf_score descending.

    The original scores are used only for ranking within each list —
    RRF produces its own scores based on rank position.
    """
    rrf_scores: dict[int, float] = {}

    for ranked in ranked_lists:
        for rank, (item_id, _score) in enumerate(ranked):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return merged


def weighted_rrf_merge(
    bm25_ranked: list[tuple[int, float]],
    emb_ranked: list[tuple[int, float]],
    k: int = RRF_K,
    bm25_weight: float = 1.0,
    emb_weight: float = 1.0,
) -> list[tuple[int, float]]:
    """Weighted RRF merge between BM25 and embedding results.

    When BM25 returns sparse results (< SPARSE_RESULT_THRESHOLD), the
    embedding weight is automatically increased to compensate.
    """
    # Auto-boost embeddings when BM25 is sparse
    if len(bm25_ranked) < SPARSE_RESULT_THRESHOLD and emb_ranked:
        emb_weight = max(emb_weight, 2.0)

    rrf_scores: dict[int, float] = {}

    for rank, (item_id, _score) in enumerate(bm25_ranked):
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + bm25_weight / (k + rank + 1)

    for rank, (item_id, _score) in enumerate(emb_ranked):
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + emb_weight / (k + rank + 1)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return merged


def embedding_search(
    query_embedding: list[float],
    all_embeddings: dict[int, list[float]],
    limit: int = 50,
    threshold: float = EMBEDDING_SIM_THRESHOLD,
) -> list[tuple[int, float]]:
    """Pure embedding similarity search over pre-loaded embeddings.

    Uses numpy for vectorized cosine similarity when available (~50x faster
    at 3500 chunks, critical at 10K+). Falls back to pure Python otherwise.

    Args:
        query_embedding: The query vector (384-dim).
        all_embeddings: {rowid: embedding_vector} for all indexed chunks.
        limit: Maximum results to return.
        threshold: Minimum cosine similarity to include.

    Returns:
        List of (rowid, cosine_similarity) sorted by similarity descending.
    """
    if not all_embeddings or not query_embedding:
        return []

    try:
        return _embedding_search_numpy(
            query_embedding, all_embeddings, limit, threshold,
        )
    except ImportError:
        pass

    from synapt.recall.embeddings import cosine_similarity

    results: list[tuple[int, float]] = []
    for rowid, emb in all_embeddings.items():
        sim = cosine_similarity(query_embedding, emb)
        if sim >= threshold:
            results.append((rowid, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]


def _embedding_search_numpy(
    query_embedding: list[float],
    all_embeddings: dict[int, list[float]],
    limit: int,
    threshold: float,
) -> list[tuple[int, float]]:
    """Vectorized embedding search using numpy.

    Builds a (N, D) matrix from the embedding dict, computes all cosine
    similarities in one vectorized operation, then filters and sorts.
    """
    import numpy as np

    if not all_embeddings or limit <= 0:
        return []

    rowids = list(all_embeddings.keys())
    matrix = np.array([all_embeddings[r] for r in rowids], dtype=np.float32)

    query = np.array(query_embedding, dtype=np.float32)

    # Dimension mismatch usually indicates a model change — warn and truncate
    if query.shape[0] != matrix.shape[1]:
        min_dim = min(query.shape[0], matrix.shape[1])
        logger.warning(
            "Embedding dimension mismatch: query=%d, stored=%d; truncating to %d",
            query.shape[0], matrix.shape[1], min_dim,
        )
        query = query[:min_dim]
        matrix = matrix[:, :min_dim]

    # Cosine similarity: dot(q, M^T) / (||q|| * ||M||)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return []
    row_norms = np.linalg.norm(matrix, axis=1)

    # Avoid division by zero for zero-norm embeddings
    valid = row_norms > 0
    sims = np.zeros(len(rowids), dtype=np.float32)
    sims[valid] = matrix[valid] @ query / (row_norms[valid] * query_norm)

    # Filter by threshold and get top-k
    mask = sims >= threshold
    if not mask.any():
        return []

    indices = np.where(mask)[0]
    masked_sims = sims[indices]

    # Partial sort for top-k (faster than full sort for large arrays)
    if len(indices) > limit:
        top_k_idx = np.argpartition(masked_sims, -limit)[-limit:]
        top_k_idx = top_k_idx[np.argsort(masked_sims[top_k_idx])[::-1]]
    else:
        top_k_idx = np.argsort(masked_sims)[::-1]

    return [(rowids[indices[i]], float(masked_sims[i])) for i in top_k_idx]


# ---------------------------------------------------------------------------
# Query intent classification
# ---------------------------------------------------------------------------

# Patterns for each intent type
_FACTUAL_PATTERNS = re.compile(
    r"\b(what\s+(is|are|was|were|does|did)|which|who\s+(is|are|was|were|does|did)|"
    r"where\s+(is|are|does|did)|"
    r"how\s+(many|much|old|long|often)|port|version|url|endpoint|config|setting|"
    r"api\s+key|database|schema|table|"
    r"favou?rite|prefer|name[ds]?)\b",
    re.IGNORECASE,
)

_TEMPORAL_PATTERNS = re.compile(
    r"\b(when\s+(is|was|did|does|were|will|has|have)|"
    r"how\s+(long|recently)|how\s+many\s+(days|weeks|months|years)|"
    r"first\s+time|last\s+time|most\s+recent|"
    r"ago|yesterday|today|recently|lately|"
    r"last\s+(week|month|year|time)|next\s+(week|month)|this\s+(week|month|year)|"
    r"what\s+(date|day|month|year|time)|"
    r"chronolog|timeline|at\s+what\s+point)\b",
    re.IGNORECASE,
)

_DEBUG_PATTERNS = re.compile(
    r"\b(error|bug|crash|fail|broke|broken|issue|problem|fix|debug|"
    r"traceback|exception|stack\s*trace|segfault|oom|timeout|"
    r"why\s+(did|does|is|was)|what\s+went\s+wrong)\b",
    re.IGNORECASE,
)

_EXPLORATORY_PATTERNS = re.compile(
    r"\b(how\s+did\s+we|what\s+did\s+we|tried|approach|strategy|"
    r"decision|alternative|option|consider|experiment|explore|"
    r"history\s+of|evolution\s+of|what\s+happened|tell\s+me\s+about)\b",
    re.IGNORECASE,
)

_PROCEDURAL_PATTERNS = re.compile(
    r"\b(how\s+(do|to|should)|step|process|workflow|procedure|"
    r"deploy|setup|install|configure|migrate|run|build|test)\b",
    re.IGNORECASE,
)


def classify_query_intent(query: str) -> str:
    """Classify a search query into an intent category.

    Returns one of:
        "temporal"    — When something happened, time-based queries
        "factual"     — Looking up a specific fact (config value, version, etc.)
        "debug"       — Investigating an error or failure
        "exploratory" — Understanding what was tried, decisions made
        "procedural"  — How to do something (steps, workflow)
        "general"     — Default when no clear intent detected

    The classification is used to weight search strategies:
    - temporal → raw transcript evidence, low knowledge boost
    - factual → knowledge nodes first, then FTS
    - debug → recent sessions weighted heavily
    - exploratory → clusters and timeline, broad search
    - procedural → knowledge nodes + cluster summaries
    - general → balanced hybrid search
    """
    scores = {
        "temporal": len(_TEMPORAL_PATTERNS.findall(query)),
        "factual": len(_FACTUAL_PATTERNS.findall(query)),
        "debug": len(_DEBUG_PATTERNS.findall(query)),
        "exploratory": len(_EXPLORATORY_PATTERNS.findall(query)),
        "procedural": len(_PROCEDURAL_PATTERNS.findall(query)),
    }

    # Tiebreak priority: temporal > exploratory > procedural > debug > factual.
    # Temporal wins ties since temporal signals are usually unambiguous.
    priority = ["temporal", "exploratory", "procedural", "debug", "factual"]
    best_score = max(scores.values())
    if best_score == 0:
        return "general"
    for intent in priority:
        if scores[intent] == best_score:
            return intent
    return "general"


def intent_search_params(intent: str) -> dict:
    """Return search parameter adjustments based on query intent.

    These override/adjust the default search parameters to better match
    the type of information being sought.
    """
    if intent == "temporal":
        return {
            "knowledge_boost": 0.5,    # De-prioritize knowledge — temporal needs sequences
            "half_life": 0.0,          # No recency decay — all timeframes matter
            "emb_weight": 1.0,         # Balanced
        }
    elif intent == "factual":
        return {
            "knowledge_boost": 3.0,    # Heavily weight knowledge nodes
            "half_life": 90.0,         # Older facts still relevant
            "emb_weight": 1.5,         # Semantic matching important
        }
    elif intent == "debug":
        return {
            "knowledge_boost": 1.0,    # Normal knowledge weight
            "half_life": 14.0,         # Strong recency — recent errors matter most
            "emb_weight": 0.8,         # Exact terms matter more for errors
        }
    elif intent == "exploratory":
        return {
            "knowledge_boost": 1.5,    # Knowledge somewhat boosted
            "half_life": 60.0,         # Moderate recency
            "emb_weight": 2.0,         # Semantic matching very important
        }
    elif intent == "procedural":
        return {
            "knowledge_boost": 2.5,    # Knowledge highly relevant
            "half_life": 90.0,         # Procedures don't expire quickly
            "emb_weight": 1.2,         # Moderate semantic
        }
    else:  # general
        return {
            "knowledge_boost": 2.0,    # Default knowledge boost
            "half_life": 60.0,         # Default recency
            "emb_weight": 1.0,         # Balanced
        }


# ---------------------------------------------------------------------------
# Temporal date extraction
# ---------------------------------------------------------------------------

# Relative time expressions
_RELATIVE_TIME_PATTERNS = [
    (re.compile(r"\byesterday\b", re.I), lambda now: (now - timedelta(days=1), now)),
    (re.compile(r"\btoday\b", re.I), lambda now: (now, now + timedelta(days=1))),
    (re.compile(r"\blast\s+week\b", re.I), lambda now: (now - timedelta(weeks=1), now)),
    (re.compile(r"\blast\s+month\b", re.I), lambda now: (now - timedelta(days=30), now)),
    (re.compile(r"\b(\d+)\s+days?\s+ago\b", re.I), None),  # handled specially
    (re.compile(r"\b(\d+)\s+weeks?\s+ago\b", re.I), None),
    (re.compile(r"\bthis\s+week\b", re.I), lambda now: (now - timedelta(days=now.weekday()), now + timedelta(days=1))),
]

# Absolute date pattern: "March 5th", "2026-03-05", "March 5, 2026"
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_DAY_PATTERN = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES.keys()) + r")\s+(\d{1,2})(?:st|nd|rd|th)?"
    r"(?:\s*,?\s*(\d{4}))?\b",
    re.I,
)
_ISO_DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def extract_temporal_range(
    query: str,
    now: datetime | None = None,
) -> tuple[str | None, str | None]:
    """Extract a date range from temporal expressions in a query.

    Returns (after, before) as ISO 8601 date strings, or (None, None)
    if no temporal expression is detected.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Check relative time patterns
    for pattern, handler in _RELATIVE_TIME_PATTERNS:
        match = pattern.search(query)
        if not match:
            continue

        if handler is not None:
            start, end = handler(now)
            return (
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )

        # N days/weeks ago
        n = int(match.group(1))
        if "week" in match.group(0).lower():
            start = now - timedelta(weeks=n)
        else:
            start = now - timedelta(days=n)
        return (start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))

    # Check "Month Day[, Year]" pattern
    match = _MONTH_DAY_PATTERN.search(query)
    if match:
        month_name = match.group(1).lower()
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else now.year
        month = _MONTH_NAMES.get(month_name)
        if month and 1 <= day <= 31:
            try:
                dt = datetime(year, month, day, tzinfo=timezone.utc)
                return (
                    dt.strftime("%Y-%m-%d"),
                    (dt + timedelta(days=1)).strftime("%Y-%m-%d"),
                )
            except ValueError:
                pass  # Invalid date (e.g., Feb 30)

    # Check ISO date pattern
    match = _ISO_DATE_PATTERN.search(query)
    if match:
        date_str = match.group(1)
        try:
            dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            return (
                dt.strftime("%Y-%m-%d"),
                (dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
        except ValueError:
            pass

    return (None, None)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

# Common question words and stop words that look like proper nouns at
# sentence start but aren't entities.
_NON_ENTITY_WORDS = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "has", "have", "had",
    "the", "a", "an", "and", "or", "but", "not", "no", "so", "if",
    "would", "could", "should", "will", "can", "may", "might",
    "likely", "still", "also", "yet", "ever", "never",
    "in", "on", "at", "to", "for", "of", "with", "from", "by", "about",
})

# Match capitalized words and possessives (e.g., "Caroline's", "Dr. Seuss")
_ENTITY_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:'s)?)\b")


def extract_entities(query: str) -> set[str]:
    """Extract likely entity names (proper nouns) from a query.

    Returns a set of lowercased entity strings. Uses capitalization as the
    primary signal — words starting with uppercase that aren't common
    question/stop words are treated as entities.

    Examples:
        "What is Caroline's identity?" → {"caroline"}
        "Where has Melanie camped?" → {"melanie"}
        "What books has Dr. Seuss written?" → {"seuss"}
    """
    matches = _ENTITY_PATTERN.findall(query)
    entities: set[str] = set()
    for m in matches:
        # Strip possessive suffix for matching
        clean = m[:-2].lower() if m.endswith("'s") else m.lower()
        if clean not in _NON_ENTITY_WORDS and len(clean) > 1:
            entities.add(clean)
    return entities
