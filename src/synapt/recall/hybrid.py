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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synapt.recall.embeddings import EmbeddingProvider

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

    from synapt.recall.embeddings import cosine_similarity

    results: list[tuple[int, float]] = []
    for rowid, emb in all_embeddings.items():
        sim = cosine_similarity(query_embedding, emb)
        if sim >= threshold:
            results.append((rowid, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]


# ---------------------------------------------------------------------------
# Query intent classification
# ---------------------------------------------------------------------------

# Patterns for each intent type
_FACTUAL_PATTERNS = re.compile(
    r"\b(what\s+(is|are|was|were|does)|which|where\s+(is|are|does)|"
    r"how\s+(many|much)|port|version|url|endpoint|config|setting|"
    r"api\s+key|database|schema|table)\b",
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
        "factual"     — Looking up a specific fact (config value, version, etc.)
        "debug"       — Investigating an error or failure
        "exploratory" — Understanding what was tried, decisions made
        "procedural"  — How to do something (steps, workflow)
        "general"     — Default when no clear intent detected

    The classification is used to weight search strategies:
    - factual → knowledge nodes first, then FTS
    - debug → recent sessions weighted heavily
    - exploratory → clusters and timeline, broad search
    - procedural → knowledge nodes + cluster summaries
    - general → balanced hybrid search
    """
    scores = {
        "factual": len(_FACTUAL_PATTERNS.findall(query)),
        "debug": len(_DEBUG_PATTERNS.findall(query)),
        "exploratory": len(_EXPLORATORY_PATTERNS.findall(query)),
        "procedural": len(_PROCEDURAL_PATTERNS.findall(query)),
    }

    # Tiebreak priority: exploratory > procedural > debug > factual.
    # Exploratory/procedural patterns are more specific and should win
    # when a query contains both e.g. "how did we fix the database".
    priority = ["exploratory", "procedural", "debug", "factual"]
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
    if intent == "factual":
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
