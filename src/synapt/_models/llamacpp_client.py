"""llama.cpp model client via llama-cpp-python bindings.

Cross-platform in-process inference — no external server required.
Uses GPU (CUDA/Vulkan/Metal) when available, falls back to CPU.
Same GGUF models as Ollama, but embedded in the Python process.

Install: pip install llama-cpp-python
  With CUDA: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
  With Vulkan: CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python
  With Metal (macOS): CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from .base import Message, ModelClient

logger = logging.getLogger(__name__)

# Cache loaded models to avoid reloading on every call
_MODEL_CACHE: dict[str, object] = {}

# Default GGUF model — downloaded from HuggingFace on first use
DEFAULT_MODEL_REPO = "bartowski/Ministral-3B-Instruct-2412-GGUF"
DEFAULT_MODEL_FILE = "Ministral-3B-Instruct-2412-Q4_K_M.gguf"


def _resolve_model_path(model: str) -> str:
    """Resolve a model identifier to a local GGUF file path.

    Accepts:
    - Local file path: "/path/to/model.gguf"
    - HuggingFace repo: "bartowski/Ministral-3B-Instruct-2412-GGUF"
      (downloads the default quant file)
    - HuggingFace repo + file: "bartowski/Ministral-3B-Instruct-2412-GGUF:Q4_K_M"
    """
    # Local path
    if Path(model).suffix == ".gguf" and Path(model).exists():
        return str(model)

    # HuggingFace download
    try:
        from huggingface_hub import hf_hub_download

        if ":" in model:
            repo, quant = model.rsplit(":", 1)
            # Find matching file in repo
            filename = f"{repo.split('/')[-1].replace('-GGUF', '')}-{quant}.gguf"
        elif "/" in model:
            repo = model
            filename = DEFAULT_MODEL_FILE
        else:
            repo = DEFAULT_MODEL_REPO
            filename = model if model.endswith(".gguf") else f"{model}.gguf"

        path = hf_hub_download(repo_id=repo, filename=filename)
        return path
    except ImportError:
        logger.warning("huggingface_hub not installed — cannot download GGUF models")
        raise
    except Exception as e:
        logger.warning("Failed to download model %s: %s", model, e)
        raise


class LlamaCppClient(ModelClient):
    """In-process llama.cpp inference via llama-cpp-python."""

    def __init__(
        self,
        model: str = "",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        max_tokens: int | None = None,
    ):
        """Initialize the client.

        Args:
            model: GGUF model path or HuggingFace repo ID.
                   Empty string uses the default Ministral 3B model.
            n_ctx: Context window size.
            n_gpu_layers: GPU layers to offload (-1 = all, 0 = CPU only).
            max_tokens: Default max tokens for generation.
        """
        self._model_id = model or DEFAULT_MODEL_REPO
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens

    def _get_llm(self, model: str = ""):
        """Get or create a cached Llama instance."""
        model_id = model or self._model_id
        if model_id in _MODEL_CACHE:
            return _MODEL_CACHE[model_id]

        from llama_cpp import Llama

        model_path = _resolve_model_path(model_id)
        logger.info("Loading GGUF model: %s", model_path)

        llm = Llama(
            model_path=model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            verbose=False,
        )
        _MODEL_CACHE[model_id] = llm
        return llm

    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        llm = self._get_llm(model)

        max_tokens = kwargs.get("max_tokens") or self.max_tokens or 300

        response = llm.create_chat_completion(
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choices = response.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""
