"""Tests for ONNX Runtime inference and model conversion."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if onnxruntime not available
onnx_available = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")


class TestFindOnnxModel:
    """Test ONNX model discovery logic."""

    def test_finds_local_dir_with_onnx_files(self, tmp_path):
        from synapt._models.onnx_client import _find_onnx_model

        (tmp_path / "encoder_model.onnx").write_text("")
        assert _find_onnx_model(str(tmp_path)) == str(tmp_path)

    def test_returns_none_for_empty_dir(self, tmp_path):
        from synapt._models.onnx_client import _find_onnx_model

        assert _find_onnx_model(str(tmp_path)) is None

    def test_returns_none_for_missing_dir(self):
        from synapt._models.onnx_client import _find_onnx_model

        assert _find_onnx_model("/nonexistent/path") is None

    def test_finds_cached_model(self, tmp_path, monkeypatch):
        from synapt._models.onnx_client import _find_onnx_model

        # Set up cache dir
        cache_dir = tmp_path / "onnx"
        model_dir = cache_dir / "my-org--my-model"
        model_dir.mkdir(parents=True)
        (model_dir / "encoder_model.onnx").write_text("")

        monkeypatch.setenv("HOME", str(tmp_path))
        # Patch expanduser to use our tmp_path
        with patch("os.path.expanduser", return_value=str(cache_dir)):
            result = _find_onnx_model("my-org/my-model")
            assert result == str(model_dir)


class TestOnnxClientChat:
    """Test OnnxClient.chat() with mocked model."""

    def test_chat_returns_string(self):
        from synapt._models.onnx_client import OnnxClient, _ONNX_CACHE
        from synapt._models.base import Message

        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_tokenizer.decode.return_value = '{"focus": "test"}'

        _ONNX_CACHE["test-model"] = (mock_model, mock_tokenizer)
        try:
            client = OnnxClient(max_tokens=100, model_name="test-model")
            result = client.chat(
                model="ignored",
                messages=[Message(role="user", content="test input")],
            )
            assert result == '{"focus": "test"}'
            mock_model.generate.assert_called_once()
        finally:
            _ONNX_CACHE.pop("test-model", None)

    def test_chat_wraps_bare_json(self):
        from synapt._models.onnx_client import OnnxClient, _ONNX_CACHE
        from synapt._models.base import Message

        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        # T5 sometimes drops outer braces
        mock_tokenizer.decode.return_value = '"focus": "test"'

        _ONNX_CACHE["test-model-2"] = (mock_model, mock_tokenizer)
        try:
            client = OnnxClient(max_tokens=100, model_name="test-model-2")
            result = client.chat(
                model="ignored",
                messages=[Message(role="user", content="test")],
            )
            assert result == '{"focus": "test"}'
        finally:
            _ONNX_CACHE.pop("test-model-2", None)


class TestConvertModule:
    """Test conversion utilities."""

    def test_is_converted_false_for_missing(self):
        from synapt._models.convert import is_converted

        assert not is_converted("nonexistent/model")

    def test_is_converted_true_for_cached(self, tmp_path, monkeypatch):
        from synapt._models.convert import is_converted

        with patch(
            "synapt._models._utils.get_onnx_cache_dir",
            return_value=str(tmp_path),
        ), patch(
            "synapt._models.onnx_client.get_onnx_cache_dir",
            return_value=str(tmp_path),
        ):
            (tmp_path / "encoder_model.onnx").write_text("")
            assert is_converted("any-model")

    def test_get_cache_dir_format(self):
        from synapt._models._utils import get_onnx_cache_dir

        result = get_onnx_cache_dir("laynepro/t5-enrichment-v2")
        assert "laynepro--t5-enrichment-v2" in result
        assert ".synapt/models/onnx" in result


class TestOnnxRouterIntegration:
    """Test ONNX integration in the model router."""

    def test_onnx_override(self, monkeypatch):
        from synapt.recall._model_router import get_client, RecallTask, clear_cache

        clear_cache()
        monkeypatch.setenv("SYNAPT_SUMMARY_BACKEND", "onnx")
        # Will return None since no ONNX model is available
        client = get_client(RecallTask.ENRICH)
        # ONNX client returns None when no model found, which is expected
        clear_cache()

    def test_is_encoder_decoder_detects_onnx(self):
        from synapt.recall._model_router import is_encoder_decoder
        from synapt._models.onnx_client import OnnxClient

        client = OnnxClient(max_tokens=100)
        assert is_encoder_decoder(client)


@pytest.mark.skipif(
    not os.path.isdir(os.path.expanduser("~/.synapt/models/onnx")),
    reason="No converted ONNX model available",
)
class TestOnnxIntegration:
    """Integration tests requiring a pre-converted ONNX model."""

    def test_enrichment_inference(self):
        """Run actual enrichment inference with ONNX model."""
        from synapt._models.onnx_client import OnnxClient
        from synapt._models.base import Message

        client = OnnxClient(max_tokens=150, model_name="laynepro/t5-enrichment-v2")
        result = client.chat(
            model="laynepro/t5-enrichment-v2",
            messages=[
                Message(
                    role="user",
                    content="summarize session: User fixed a bug in the login flow.",
                )
            ],
        )
        assert result
        parsed = json.loads(result)
        assert "focus" in parsed
