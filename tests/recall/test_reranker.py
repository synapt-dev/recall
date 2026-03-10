"""Tests for cross-encoder reranking (Phase 2)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synapt.recall.reranker import (
    DEFAULT_RERANKER_MODEL,
    MIN_CANDIDATES_FOR_RERANK,
    clear_cache,
    get_reranker_model,
    is_reranker_enabled,
    rerank,
)


@pytest.fixture(autouse=True)
def _clear_reranker_cache():
    """Clear the reranker cache before each test."""
    clear_cache()
    yield
    clear_cache()


def _make_chunk(text: str) -> MagicMock:
    """Create a mock TranscriptChunk with the given text."""
    chunk = MagicMock()
    chunk.text = text
    return chunk


class TestConfig:
    """Test reranker configuration."""

    def test_default_model(self):
        assert "ms-marco-MiniLM" in DEFAULT_RERANKER_MODEL

    def test_env_model_override(self, monkeypatch):
        monkeypatch.setenv("SYNAPT_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        assert get_reranker_model() == "cross-encoder/ms-marco-MiniLM-L-12-v2"

    def test_disabled_by_default(self):
        assert not is_reranker_enabled()

    def test_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("SYNAPT_RERANKER", "true")
        assert is_reranker_enabled()

    def test_enabled_via_env_yes(self, monkeypatch):
        monkeypatch.setenv("SYNAPT_RERANKER", "yes")
        assert is_reranker_enabled()

    def test_enabled_via_env_1(self, monkeypatch):
        monkeypatch.setenv("SYNAPT_RERANKER", "1")
        assert is_reranker_enabled()


class TestRerank:
    """Test rerank function."""

    def test_empty_candidates(self):
        """Empty input returns empty output."""
        result = rerank("query", [], [])
        assert result == []

    def test_too_few_candidates_skips_reranking(self):
        """Below MIN_CANDIDATES, reranking is skipped (returns original)."""
        chunks = [_make_chunk(f"text {i}") for i in range(2)]
        candidates = [(0, 1.0), (1, 0.5)]
        assert len(candidates) < MIN_CANDIDATES_FOR_RERANK
        result = rerank("query", candidates, chunks)
        assert result == candidates

    @patch("synapt.recall.reranker._load_model")
    def test_reranking_reorders_by_score(self, mock_load):
        """Cross-encoder scores should determine the output order."""
        mock_model = MagicMock()
        # Return scores that reverse the original order
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        mock_load.return_value = mock_model

        chunks = [_make_chunk(f"text {i}") for i in range(3)]
        candidates = [(0, 3.0), (1, 2.0), (2, 1.0)]

        result = rerank("query", candidates, chunks)

        # idx=1 had highest cross-encoder score (0.9)
        assert result[0][0] == 1
        assert result[1][0] == 2
        assert result[2][0] == 0

    @patch("synapt.recall.reranker._load_model")
    def test_model_receives_correct_pairs(self, mock_load):
        """Cross-encoder should receive (query, chunk_text) pairs."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.3, 0.8]
        mock_load.return_value = mock_model

        chunks = [_make_chunk("hello"), _make_chunk("world"), _make_chunk("test")]
        candidates = [(0, 1.0), (1, 0.5), (2, 0.3)]

        rerank("my query", candidates, chunks)

        pairs = mock_model.predict.call_args[0][0]
        assert pairs == [
            ("my query", "hello"),
            ("my query", "world"),
            ("my query", "test"),
        ]

    @patch("synapt.recall.reranker._load_model")
    def test_model_failure_returns_original(self, mock_load):
        """If cross-encoder prediction fails, return original candidates."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model error")
        mock_load.return_value = mock_model

        chunks = [_make_chunk(f"text {i}") for i in range(3)]
        candidates = [(0, 3.0), (1, 2.0), (2, 1.0)]

        result = rerank("query", candidates, chunks)
        assert result == candidates

    @patch("synapt.recall.reranker._load_model")
    def test_model_load_failure_returns_original(self, mock_load):
        """If model loading fails, return original candidates."""
        mock_load.return_value = None

        chunks = [_make_chunk(f"text {i}") for i in range(3)]
        candidates = [(0, 3.0), (1, 2.0), (2, 1.0)]

        result = rerank("query", candidates, chunks)
        assert result == candidates

    @patch("synapt.recall.reranker._load_model")
    def test_text_truncation(self, mock_load):
        """Long chunk text should be truncated to MAX_CHUNK_CHARS."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.3, 0.8]
        mock_load.return_value = mock_model

        long_text = "x" * 5000
        chunks = [_make_chunk(long_text), _make_chunk("short"), _make_chunk("also short")]
        candidates = [(0, 1.0), (1, 0.5), (2, 0.3)]

        rerank("query", candidates, chunks)

        pairs = mock_model.predict.call_args[0][0]
        assert len(pairs[0][1]) == 2000  # MAX_CHUNK_CHARS


class TestIntegration:
    """Integration tests (require sentence-transformers)."""

    def test_real_cross_encoder(self):
        """Test with real cross-encoder model if available."""
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        chunks = [
            _make_chunk("The capital of France is Paris"),
            _make_chunk("Python is a programming language"),
            _make_chunk("The Eiffel Tower is in Paris, France"),
        ]
        candidates = [(0, 1.0), (1, 0.8), (2, 0.6)]

        result = rerank("What is the capital of France?", candidates, chunks)

        # The cross-encoder should rank the France-related chunks higher
        top_idx = result[0][0]
        assert top_idx in (0, 2), f"Expected France-related chunk, got idx={top_idx}"
