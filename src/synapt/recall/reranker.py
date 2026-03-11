"""Cross-encoder reranking for search results.

After BM25 + embedding RRF fusion produces a candidate list, the cross-encoder
scores each (query, document) pair directly. This is more accurate than
bi-encoder similarity because the cross-encoder sees the full query-document
interaction jointly, but slower (O(n) forward passes vs O(1) for pre-computed
embeddings).

Reranking top-20 candidates takes ~20-50ms on CPU with MiniLM.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params, ~80MB)
Enable: SYNAPT_RERANKER=true or use_reranker=True in TranscriptIndex
Override model: SYNAPT_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synapt.recall.core import TranscriptChunk

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Minimum candidates to bother reranking — below this, order doesn't matter much
MIN_CANDIDATES_FOR_RERANK = 3

# Max characters of chunk text to pass to cross-encoder.
# ms-marco-MiniLM has 512 token limit; ~4 chars/token → ~2000 chars.
MAX_CHUNK_CHARS = 2000

# Module-level cache: model_name → CrossEncoder instance
_reranker_cache: dict[str, object] = {}


def get_reranker_model() -> str:
    """Get the configured cross-encoder model name."""
    from synapt.recall.config import load_config
    return load_config().get_model("reranker")


def is_reranker_enabled() -> bool:
    """Check if cross-encoder reranking is enabled via env var."""
    return os.environ.get("SYNAPT_RERANKER", "").lower() in ("true", "1", "yes")


def rerank(
    query: str,
    candidates: list[tuple[int, float]],
    chunks: list[TranscriptChunk],
    model_name: str | None = None,
) -> list[tuple[int, float]]:
    """Re-rank candidates using cross-encoder scoring.

    Takes the top candidates from RRF merge, scores each (query, chunk_text)
    pair with the cross-encoder, and returns them sorted by relevance score.

    Args:
        query: The search query.
        candidates: List of (chunk_idx, rrf_score) from RRF merge.
        chunks: The chunk list to get text from.
        model_name: Cross-encoder model (default: ms-marco-MiniLM-L-6-v2).

    Returns:
        List of (chunk_idx, cross_encoder_score) sorted by score descending.
    """
    if len(candidates) < MIN_CANDIDATES_FOR_RERANK:
        return candidates

    model_name = model_name or get_reranker_model()
    model = _load_model(model_name)
    if model is None:
        return candidates

    # Build (query, document) pairs
    pairs = []
    indices = []
    for idx, _score in candidates:
        text = chunks[idx].text[:MAX_CHUNK_CHARS]
        pairs.append((query, text))
        indices.append(idx)

    try:
        scores = model.predict(pairs)
    except Exception as e:
        logger.warning("Cross-encoder reranking failed: %s", e)
        return candidates

    # Build reranked list sorted by cross-encoder score
    reranked = sorted(
        zip(indices, (float(s) for s in scores)),
        key=lambda x: x[1],
        reverse=True,
    )

    return reranked


def _load_model(model_name: str) -> object | None:
    """Load or retrieve cached cross-encoder model."""
    if model_name in _reranker_cache:
        return _reranker_cache[model_name]

    try:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder: %s", model_name)
        model = CrossEncoder(model_name)
        _reranker_cache[model_name] = model
        return model
    except ImportError:
        # Permanent failure — sentence-transformers not installed
        logger.warning("sentence-transformers not installed, cross-encoder unavailable")
        _reranker_cache[model_name] = None
        return None
    except Exception as e:
        # Transient failure (network, disk) — don't cache, allow retry
        logger.warning("Failed to load cross-encoder %s: %s", model_name, e)
        return None


def clear_cache() -> None:
    """Clear the model cache. Useful for testing."""
    _reranker_cache.clear()
