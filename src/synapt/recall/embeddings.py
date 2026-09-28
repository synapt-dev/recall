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
import threading
import urllib.request
from typing import List, Optional

# The ONE cosine for the package lives in vector_math, which refuses a width
# mismatch instead of truncating the dot product. This name is re-exported
# because hybrid, consolidate and knowledge already import it from here; they
# inherit the fix without their call sites changing. Nothing here re-implements
# it -- a second definition is what let the truncation survive in three places.
from synapt.recall.vector_math import cosine_similarity  # noqa: F401


def _detect_device() -> str:
    """Pick the safest available torch device for embedding inference.

    Apple Silicon CPU path hits a Accelerate BLAS alignment fault (SIGBUS) on
    sentence-transformers matmul; prefer MPS when available. Linux and other
    CPU-only environments fall back to "cpu" cleanly.

    Env override: SYNAPT_EMBED_DEVICE=cpu|mps|cuda
    """
    import os
    override = os.environ.get("SYNAPT_EMBED_DEVICE", "").strip().lower()
    if override in ("cpu", "mps", "cuda"):
        return override
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


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
    """Local embeddings via sentence-transformers. No network required.

    Model loading is deferred to first use (lazy initialization) to avoid
    semaphore/deadlock issues when the provider is created before
    multiprocessing forks.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 device: str | None = None):
        self._model_name = model_name
        self._device = device if device is not None else _detect_device()
        self._model = None
        self._dim_cached: int | None = None

    def _ensure_model(self) -> None:
        """Load the model on first use, not at init time."""
        if self._model is None:
            import os
            from pathlib import Path
            from sentence_transformers import SentenceTransformer

            # Skip the HuggingFace revision-check network call if the model is
            # already in the local cache. Accept both fully-qualified
            # (`sentence-transformers/all-MiniLM-L6-v2`) and shorthand
            # (`all-MiniLM-L6-v2`) cache directory forms.
            hf_cache = Path(
                os.environ.get("HF_HOME",
                               os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
            ) / "hub"
            cache_names = {"models--" + self._model_name.replace("/", "--")}
            if "/" not in self._model_name:
                cache_names.add(f"models--sentence-transformers--{self._model_name}")
            if any((hf_cache / cache_name).exists() for cache_name in cache_names):
                os.environ.setdefault("HF_HUB_OFFLINE", "1")

            self._model = SentenceTransformer(self._model_name, device=self._device)
            self._dim_cached = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        self._ensure_model()
        return self._dim_cached  # type: ignore[return-value]

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        embeddings = self._model.encode(texts, batch_size=32, show_progress_bar=False)
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


# `cosine_similarity` was defined here. It is now imported above from
# vector_math, which is the single owner of the width rule; see that module for
# why a mismatched pair raises rather than scoring.


_singleton_lock = threading.Lock()
_singleton_cache: dict[bool, tuple[bool, Optional[EmbeddingProvider]]] = {}


def get_embedding_provider(prefer_local: bool = True) -> Optional[EmbeddingProvider]:
    """Auto-select the best available embedding provider.

    Returns a cached singleton so the model is loaded at most once per
    process.  Thread-safe via double-checked locking.  Caches separately
    per ``prefer_local`` value.

    Fixes #357 — previous behaviour created a new provider (and
    re-loaded the model) on every call.

    Priority: local sentence-transformers -> Ollama -> None (BM25-only).
    """
    cached = _singleton_cache.get(prefer_local)
    if cached is not None:
        return cached[1]

    with _singleton_lock:
        # Double-check after acquiring lock
        cached = _singleton_cache.get(prefer_local)
        if cached is not None:
            return cached[1]

        provider = _resolve_provider(prefer_local)
        _singleton_cache[prefer_local] = (True, provider)
        return provider


def _resolve_provider(prefer_local: bool) -> Optional[EmbeddingProvider]:
    """Resolve the embedding provider without caching."""
    import logging
    log = logging.getLogger(__name__)
    # Local import: storage does not import embeddings, and the width this store
    # can hold is storage's constant, so it is read from its one home.
    from synapt.recall.storage import EMBEDDING_DIM

    if prefer_local:
        try:
            import sentence_transformers  # noqa: F401
            return LocalEmbeddings()
        except ImportError:
            log.info(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            log.warning("sentence-transformers failed to load: %s", e)

    try:
        provider = OllamaEmbeddings()
        # Verify Ollama is reachable, and reuse the SAME probe to check the WIDTH
        # of what it actually serves. The store's blob column is a fixed-width
        # format derived from storage.EMBEDDING_DIM, so a provider of any other
        # width cannot be written at all: struct.pack raises, the node has
        # already landed, and the save reports failure over a row that search
        # still finds. Two hardcoded constants disagreeing.
        probe = provider.embed(["test"])
        width = len(probe[0]) if probe else 0
        if width != EMBEDDING_DIM:
            # VISIBLE BY DESIGN: a provider silently ignored leaves a user
            # wondering why their working Ollama is not being used. Name the
            # model and both widths, once, at the point of refusal.
            log.warning(
                "Ollama embeddings model %r returns %d dimensions but this store "
                "holds %d; REFUSING it so the store is never handed an "
                "unwritable vector (search falls back to BM25 keyword matching). "
                "Pull a %d-dimension model, or install sentence-transformers for "
                "hybrid semantic search.",
                provider.model, width, EMBEDDING_DIM, EMBEDDING_DIM,
            )
            return None
        return provider
    except Exception:
        log.info("Ollama embeddings unavailable (server not running or model not pulled)")

    log.warning(
        "No embedding provider found — search will use BM25 only (keyword matching). "
        "Install sentence-transformers for hybrid semantic search: "
        "pip install sentence-transformers"
    )
    return None
