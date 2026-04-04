"""Tests for embedding provider singleton caching (#357)."""

import threading
from unittest.mock import patch

from synapt.recall.embeddings import (
    EmbeddingProvider,
    _singleton_cache,
    _singleton_lock,
    get_embedding_provider,
)


def _clear_cache():
    """Reset the singleton cache between tests."""
    with _singleton_lock:
        _singleton_cache.clear()


class _FakeProvider(EmbeddingProvider):
    @property
    def dim(self):
        return 8

    def embed(self, texts):
        return [[0.0] * 8 for _ in texts]


def test_singleton_returns_same_instance():
    _clear_cache()
    with patch("synapt.recall.embeddings._resolve_provider", return_value=_FakeProvider()):
        p1 = get_embedding_provider()
        p2 = get_embedding_provider()
        assert p1 is p2


def test_singleton_caches_per_prefer_local():
    _clear_cache()
    fake_local = _FakeProvider()
    fake_ollama = _FakeProvider()

    def resolver(prefer_local):
        return fake_local if prefer_local else fake_ollama

    with patch("synapt.recall.embeddings._resolve_provider", side_effect=resolver):
        local = get_embedding_provider(prefer_local=True)
        ollama = get_embedding_provider(prefer_local=False)
        assert local is not ollama
        assert local is fake_local
        assert ollama is fake_ollama
        # Subsequent calls return cached
        assert get_embedding_provider(prefer_local=True) is local
        assert get_embedding_provider(prefer_local=False) is ollama


def test_singleton_thread_safety():
    _clear_cache()
    call_count = 0
    provider = _FakeProvider()

    def slow_resolver(prefer_local):
        nonlocal call_count
        call_count += 1
        return provider

    with patch("synapt.recall.embeddings._resolve_provider", side_effect=slow_resolver):
        results = [None] * 10
        threads = []
        for i in range(10):
            t = threading.Thread(target=lambda idx: results.__setitem__(idx, get_embedding_provider()), args=(i,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads got the same instance
        assert all(r is provider for r in results)
        # Resolver was called exactly once
        assert call_count == 1


def test_singleton_caches_none():
    _clear_cache()
    with patch("synapt.recall.embeddings._resolve_provider", return_value=None):
        p1 = get_embedding_provider()
        p2 = get_embedding_provider()
        assert p1 is None
        assert p2 is None
