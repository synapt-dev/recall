"""Tests for embedding provider singleton caching (#357), and for the
width refusal that keeps a provider the store cannot hold."""

import logging
import threading
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]

from synapt.recall.embeddings import (
    EmbeddingProvider,
    _resolve_provider,
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


# --- a provider whose WIDTH is not the store's is refused --------------------
#
# The store's blob column is a fixed-width format derived from
# storage.EMBEDDING_DIM. A provider of any other width cannot be written at all:
# struct.pack raises inside the save, AFTER the node row has landed, so the save
# reports failure over a row that search still finds.
#
# Measured on this host: the real ``qwen3-embedding:0.6b`` (the model
# ``OllamaEmbeddings`` defaults to) returns 1024 vectors against a 384 store,
# and struct.pack("384f", *v) raises "pack expected 384 items for packing".


class _ProviderAtWidth(EmbeddingProvider):
    """An Ollama-shaped provider serving a model of a given width."""

    def __init__(self, width: int, model: str = "fake-model") -> None:
        self._width = width
        self.model = model

    @property
    def dim(self):
        return self._width

    def embed(self, texts):
        return [[0.0] * self._width for _ in texts]


def test_provider_wider_than_the_store_is_refused(caplog):
    """THE WITNESS. Before the width check this returns the provider, and the
    first save dies in struct.pack over a row that has already landed."""
    from synapt.recall.storage import EMBEDDING_DIM

    wide = _ProviderAtWidth(1024, model="qwen3-embedding:0.6b")
    assert 1024 != EMBEDDING_DIM, "the fixture is only meaningful if the widths differ"

    with patch("synapt.recall.embeddings.OllamaEmbeddings", return_value=wide):
        with caplog.at_level(logging.WARNING):
            got = _resolve_provider(prefer_local=False)

    assert got is None, (
        "a provider the store cannot hold must not be adopted; adopting it makes "
        "every save fail at the embedding write"
    )
    # The refusal must be VISIBLE: name the model and BOTH widths, or a user
    # cannot tell why their working Ollama was passed over.
    text = caplog.text
    assert "qwen3-embedding:0.6b" in text, f"the refusal must name the model: {text!r}"
    assert "1024" in text, f"and the provider's width: {text!r}"
    assert str(EMBEDDING_DIM) in text, f"and the store's width: {text!r}"


def test_provider_of_the_store_width_is_still_adopted():
    """THE CONTROL, and the witness is worthless without it: a refusal that
    refuses every provider would pass the test above. The same path must adopt a
    provider whose width IS the store's."""
    from synapt.recall.storage import EMBEDDING_DIM

    right = _ProviderAtWidth(EMBEDDING_DIM, model="right-width-model")
    with patch("synapt.recall.embeddings.OllamaEmbeddings", return_value=right):
        got = _resolve_provider(prefer_local=False)
    assert got is not None, "a provider of the store's own width must still be adopted"


def test_mutation_dropping_the_width_check_reddens_the_witness(tmp_path):
    """The mutation, on the REAL source bytes. Remove the width comparison and
    the wide provider must be adopted again -- which is the defect. The count is
    asserted FIRST so a mutation that silently failed to apply cannot read as a
    passing red."""
    import importlib.util

    src = (REPO_ROOT / "src" / "synapt" / "recall" / "embeddings.py").read_text()
    guard = "        if width != EMBEDDING_DIM:"
    assert src.count(guard) == 1, f"the width guard must be present exactly once, got {src.count(guard)}"
    assert src.count("width = len(probe[0]) if probe else 0") == 1, "the measurement must be present"

    # Drop the guard and its body, keeping the probe (the reachability check the
    # original branch had).
    start = src.index(guard)
    end = src.index("        return provider", start) + len("        return provider\n")
    mutated = src[:start] + "        return provider\n" + src[end:]
    assert mutated != src, "the mutation must change the bytes"
    assert guard not in mutated, "the mutation must actually remove the width guard"

    mpath = tmp_path / "mutated_embeddings.py"
    mpath.write_text(mutated)
    spec = importlib.util.spec_from_file_location("mutated_embeddings", mpath)
    mod = importlib.util.module_from_spec(spec)
    import sys as _sys
    _sys.modules["mutated_embeddings"] = mod
    spec.loader.exec_module(mod)

    wide = _ProviderAtWidth(1024, model="qwen3-embedding:0.6b")
    with patch.object(mod, "OllamaEmbeddings", return_value=wide):
        got = mod._resolve_provider(prefer_local=False)
    assert got is not None, (
        "with the width check gone the wide provider IS adopted -- that is the "
        "defect the guard closes, and this is what makes the witness above mean "
        "something"
    )
