"""Embedding provider abstraction for local-first semantic search.

Supports two backends:
  1. Local: sentence-transformers (all-MiniLM-L6-v2, ~90MB, runs offline)
  2. Ollama: API-based (legacy, requires running Ollama server)

Auto-selects the best available backend via get_embedding_provider().
Falls back gracefully: local -> Ollama -> None (BM25-only).

Usage:
    from synapt.recall.embeddings import get_embedding_provider

    provider = get_embedding_provider()
    if provider:
        vectors = provider.embed(["func hello()", "struct User"])
"""

from __future__ import annotations

import json
import math
import urllib.request
from typing import List, Optional


class EmbeddingProvider:
    """Base class for embedding backends."""

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        raise NotImplementedError

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        raise NotImplementedError

    def embed_single(self, text: str) -> List[float]:
        """Embed one text. Convenience wrapper."""
        return self.embed([text])[0]


class LocalEmbeddings(EmbeddingProvider):
    """Local embeddings via sentence-transformers. No network required."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        return embeddings.tolist()


class OllamaEmbeddings(EmbeddingProvider):
    """Embeddings via Ollama API. Requires running Ollama server."""

    def __init__(self, model: str = "qwen3-embedding:0.6b",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.api_url = f"{base_url}/api/embed"
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Probe with a dummy embed to get dimension
            vecs = self.embed(["dim probe"])
            self._dim = len(vecs[0])
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        # Batch in groups of 32
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            payload = json.dumps({
                "model": self.model,
                "input": batch,
                "keep_alive": "0",
            }).encode("utf-8")
            req = urllib.request.Request(
                self.api_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            all_embeddings.extend(data["embeddings"])
        return all_embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors (pure Python)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_embedding_provider(prefer_local: bool = True) -> Optional[EmbeddingProvider]:
    """Auto-select the best available embedding provider.

    Priority: local sentence-transformers -> Ollama -> None (BM25-only).
    """
    if prefer_local:
        try:
            provider = LocalEmbeddings()
            # Verify it works
            provider.embed(["test"])
            return provider
        except (ImportError, Exception):
            pass

    try:
        provider = OllamaEmbeddings()
        # Verify Ollama is reachable
        provider.embed(["test"])
        return provider
    except Exception:
        pass

    return None
