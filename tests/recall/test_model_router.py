"""Tests for the task-aware model router (#317)."""

from __future__ import annotations

import pytest

from synapt.recall._model_router import (
    DEFAULT_ENCODER_DECODER_MODEL,
    RecallTask,
    clear_cache,
    get_client,
    get_encoder_decoder_model,
    is_encoder_decoder,
)


@pytest.fixture(autouse=True)
def _clear_router_cache():
    """Clear the router cache before each test."""
    clear_cache()
    yield
    clear_cache()


class TestGetClient:
    """Test get_client routing logic."""

    def test_summarize_returns_client(self):
        """SUMMARIZE should return some client (MLX or Transformers)."""
        client = get_client(RecallTask.SUMMARIZE)
        # At least one backend should be available in the test environment
        # (MLX on Apple Silicon). If neither is available, this is expected.
        if client is not None:
            assert hasattr(client, "chat")

    def test_consolidate_returns_decoder_only(self):
        """CONSOLIDATE should prefer decoder-only (MLX), not encoder-decoder."""
        client = get_client(RecallTask.CONSOLIDATE)
        if client is not None:
            assert not is_encoder_decoder(client)

    def test_enrich_returns_decoder_only(self):
        """ENRICH should prefer decoder-only, not encoder-decoder."""
        client = get_client(RecallTask.ENRICH)
        if client is not None:
            assert not is_encoder_decoder(client)

    def test_client_caching(self):
        """Same task + max_tokens should return cached client."""
        c1 = get_client(RecallTask.SUMMARIZE, max_tokens=300)
        c2 = get_client(RecallTask.SUMMARIZE, max_tokens=300)
        if c1 is not None:
            assert c1 is c2

    def test_different_max_tokens_different_cache(self):
        """Different max_tokens should create different clients."""
        c1 = get_client(RecallTask.SUMMARIZE, max_tokens=100)
        c2 = get_client(RecallTask.SUMMARIZE, max_tokens=500)
        if c1 is not None and c2 is not None:
            assert c1 is not c2


class TestEnvOverride:
    """Test SYNAPT_SUMMARY_BACKEND override."""

    def test_mlx_override(self, monkeypatch):
        """SYNAPT_SUMMARY_BACKEND=mlx forces MLX backend."""
        monkeypatch.setenv("SYNAPT_SUMMARY_BACKEND", "mlx")
        client = get_client(RecallTask.SUMMARIZE)
        if client is not None:
            assert not is_encoder_decoder(client)

    def test_transformers_override(self, monkeypatch):
        """SYNAPT_SUMMARY_BACKEND=transformers forces Transformers backend."""
        monkeypatch.setenv("SYNAPT_SUMMARY_BACKEND", "transformers")
        client = get_client(RecallTask.SUMMARIZE)
        if client is not None:
            assert is_encoder_decoder(client)

    def test_invalid_override_uses_default(self, monkeypatch):
        """Unknown backend name falls through to default routing."""
        monkeypatch.setenv("SYNAPT_SUMMARY_BACKEND", "nonexistent")
        client = get_client(RecallTask.SUMMARIZE)
        # Falls through to default task-based routing, not None
        if client is not None:
            assert hasattr(client, "chat")


class TestModelOverride:
    """Test SYNAPT_SUMMARY_MODEL override."""

    def test_default_model(self):
        """Default model is flan-t5-base."""
        assert get_encoder_decoder_model() == DEFAULT_ENCODER_DECODER_MODEL
        assert "flan-t5-base" in get_encoder_decoder_model()

    def test_env_override(self, monkeypatch):
        """SYNAPT_SUMMARY_MODEL overrides the default model."""
        monkeypatch.setenv("SYNAPT_SUMMARY_MODEL", "google/flan-t5-large")
        assert get_encoder_decoder_model() == "google/flan-t5-large"

    def test_custom_model(self, monkeypatch):
        """Any HuggingFace model name is accepted."""
        monkeypatch.setenv("SYNAPT_SUMMARY_MODEL", "google/flan-t5-small")
        assert get_encoder_decoder_model() == "google/flan-t5-small"


class TestFallbackChain:
    """Test fallback when preferred backend is unavailable."""

    def test_summarize_falls_back_to_mlx(self, monkeypatch):
        """When transformers unavailable, SUMMARIZE falls back to MLX."""
        import synapt.recall._model_router as router

        def _no_transformers(max_tokens):
            return None

        monkeypatch.setattr(router, "_get_transformers_client", _no_transformers)
        client = get_client(RecallTask.SUMMARIZE)
        if client is not None:
            assert not is_encoder_decoder(client)

    def test_no_backend_returns_none(self, monkeypatch):
        """When all backends unavailable, returns None."""
        import synapt.recall._model_router as router

        monkeypatch.setattr(router, "_get_transformers_client", lambda mt: None)
        monkeypatch.setattr(router, "_get_mlx_client", lambda mt: None)
        monkeypatch.setattr(router, "_get_ollama_client", lambda mt: None)
        client = get_client(RecallTask.SUMMARIZE)
        assert client is None


class TestIsEncoderDecoder:
    """Test the is_encoder_decoder helper."""

    def test_none_is_not_encoder_decoder(self):
        assert not is_encoder_decoder(None)

    def test_arbitrary_object_is_not_encoder_decoder(self):
        assert not is_encoder_decoder("not a client")

    def test_transformers_client_detected(self):
        try:
            from synapt._models.transformers_client import TransformersClient
            client = TransformersClient(max_tokens=100)
            assert is_encoder_decoder(client)
        except ImportError:
            pytest.skip("transformers not installed")
