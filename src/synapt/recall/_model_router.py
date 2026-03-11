"""Task-aware model routing for the recall system.

Routes each recall task to the best available model architecture:
- summarize: Prefers encoder-decoder (T5) for parallel input processing
- enrich: Prefers encoder-decoder (T5 fine-tuned for enrichment JSON)
- consolidate: Uses decoder-only (MLX/Ollama) for complex reasoning

Falls back gracefully: ONNX → transformers → MLX → Ollama → None.
ONNX Runtime is 5-7x faster than PyTorch CPU via graph optimization.
Run 'synapt-convert' to create ONNX models.

Configure models via ~/.synapt/config.json or .synapt/recall/config.json.
Override with SYNAPT_SUMMARY_BACKEND=onnx|mlx|transformers|ollama env var.
"""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)

# Default models per architecture
DEFAULT_DECODER_MODEL = "mlx-community/Ministral-3-3B-Instruct-2512-4bit"
DEFAULT_ENCODER_DECODER_MODEL = "google/flan-t5-base"

# Supported encoder-decoder models with their memory footprints
ENCODER_DECODER_MODELS: dict[str, dict] = {
    "google/flan-t5-small": {"params": "80M", "memory_fp32": "~0.3 GB"},
    "google/flan-t5-base": {"params": "250M", "memory_fp32": "~1.0 GB"},
    "google/flan-t5-large": {"params": "780M", "memory_fp32": "~3.2 GB"},
}


def get_encoder_decoder_model() -> str:
    """Get the configured encoder-decoder model name.

    Resolved from config files or SYNAPT_SUMMARY_MODEL env var.
    """
    from synapt.recall.config import load_config
    return load_config().get_model("summarization")


class RecallTask(Enum):
    """Tasks that require LLM inference in the recall system."""
    SUMMARIZE = "summarize"
    CONSOLIDATE = "consolidate"
    ENRICH = "enrich"


# Task → preferred architecture
# Summarize: encoder-decoder (parallel input, plain text output).
# Enrich: encoder-decoder (T5 fine-tuned for enrichment JSON output).
# Consolidate: decoder-only (complex multi-entry reasoning).
_TASK_PREFERENCE: dict[RecallTask, str] = {
    RecallTask.SUMMARIZE: "encoder-decoder",
    RecallTask.CONSOLIDATE: "decoder-only",
    RecallTask.ENRICH: "encoder-decoder",
}

# Task → config key for model lookup.
_TASK_MODEL_KEY: dict[RecallTask, str] = {
    RecallTask.SUMMARIZE: "summarization",
    RecallTask.ENRICH: "enrichment",
    RecallTask.CONSOLIDATE: "consolidation",
}

# Module-level client cache: (backend, max_tokens, model) → client instance
_client_cache: dict[tuple, object] = {}


def get_client(
    task: RecallTask,
    max_tokens: int = 300,
) -> object | None:
    """Get the best available model client for a recall task.

    Returns a ModelClient instance, or None if no backend is available.
    The client is cached per (backend, max_tokens) to avoid reloading models.
    Model names are resolved from config files and env vars.
    """
    from synapt.recall.config import load_config
    cfg = load_config()

    override = cfg.backend
    if override == "onnx":
        return _get_onnx_client(max_tokens)
    elif override == "mlx":
        return _get_mlx_client(max_tokens)
    elif override == "transformers":
        return _get_transformers_client(max_tokens)
    elif override == "ollama":
        return _get_ollama_client(max_tokens)

    preferred = _TASK_PREFERENCE[task]
    model_key = _TASK_MODEL_KEY.get(task)
    model_name = cfg.get_model(model_key) if model_key else None

    if preferred == "encoder-decoder":
        # Prefer ONNX (5-7x faster), fall back to PyTorch transformers
        client = _get_onnx_client(max_tokens, model_name=model_name)
        if client is None:
            client = _get_transformers_client(max_tokens, model_name=model_name)
        if client is not None:
            return client
        # Fall back to decoder-only
        return _get_mlx_client(max_tokens) or _get_ollama_client(max_tokens)
    else:
        return _get_mlx_client(max_tokens) or _get_ollama_client(max_tokens)


_NOT_FOUND = object()  # Sentinel for cached negative lookups


def _get_onnx_client(
    max_tokens: int,
    model_name: str | None = None,
) -> object | None:
    """Try to create an OnnxClient. Returns None if unavailable."""
    key = ("onnx", max_tokens, model_name)
    cached = _client_cache.get(key, _NOT_FOUND)
    if cached is not _NOT_FOUND:
        return cached

    try:
        from synapt._models.onnx_client import OnnxClient

        # Only return ONNX client if a converted model exists
        check_model = model_name or get_encoder_decoder_model()
        if not OnnxClient.is_available(check_model):
            logger.debug("No ONNX model found for %s, skipping", check_model)
            _client_cache[key] = None  # Cache negative result
            return None

        client = OnnxClient(max_tokens=max_tokens, model_name=model_name)
        _client_cache[key] = client
        logger.debug("OnnxClient available for accelerated encoder-decoder inference")
        return client
    except ImportError:
        logger.debug("onnxruntime not installed, skipping ONNX backend")
        _client_cache[key] = None  # Cache negative result
        return None


def _get_transformers_client(
    max_tokens: int,
    model_name: str | None = None,
) -> object | None:
    """Try to create a TransformersClient. Returns None if unavailable."""
    key = ("transformers", max_tokens, model_name)
    if key in _client_cache:
        return _client_cache[key]

    try:
        from synapt._models.transformers_client import TransformersClient
        client = TransformersClient(max_tokens=max_tokens, model_name=model_name)
        _client_cache[key] = client
        logger.debug("TransformersClient available for encoder-decoder inference")
        return client
    except ImportError:
        logger.debug("transformers not installed, skipping encoder-decoder backend")
        return None


def _get_mlx_client(max_tokens: int) -> object | None:
    """Try to create an MLXClient. Returns None if unavailable."""
    key = ("mlx", max_tokens)
    if key in _client_cache:
        return _client_cache[key]

    try:
        from synapt._models.mlx_client import MLXClient, MLXOptions
        client = MLXClient(MLXOptions(max_tokens=max_tokens))
        _client_cache[key] = client
        logger.debug("MLXClient available for decoder-only inference")
        return client
    except ImportError:
        logger.debug("mlx-lm not installed, skipping MLX backend")
        return None


def _get_ollama_client(max_tokens: int) -> object | None:
    """Try to create an OllamaClient. Returns None if unavailable."""
    key = ("ollama", max_tokens)
    if key in _client_cache:
        return _client_cache[key]

    try:
        from synapt._models.ollama_client import OllamaClient
        client = OllamaClient(max_tokens=max_tokens)
        _client_cache[key] = client
        logger.debug("OllamaClient available for decoder-only inference")
        return client
    except ImportError:
        logger.debug("OllamaClient not available")
        return None


def clear_cache() -> None:
    """Clear the client cache. Useful for testing."""
    _client_cache.clear()


def is_encoder_decoder(client: object) -> bool:
    """Check if a client is an encoder-decoder model (TransformersClient or OnnxClient)."""
    try:
        from synapt._models.transformers_client import TransformersClient
        if isinstance(client, TransformersClient):
            return True
    except ImportError:
        pass
    try:
        from synapt._models.onnx_client import OnnxClient
        if isinstance(client, OnnxClient):
            return True
    except ImportError:
        pass
    return False
