"""vLLM inference client for decoder-only models.

Fast GPU inference for models like Ministral-8B using vLLM's
offline LLM engine. No server process needed.

Requires: pip install vllm
"""

from __future__ import annotations

import logging
from typing import List

from .base import Message, ModelClient

logger = logging.getLogger(__name__)

_ENGINE_CACHE: dict[str, object] = {}


class VLLMClient(ModelClient):
    """Decoder-only model client via vLLM offline inference."""

    def __init__(self, max_tokens: int = 800, model_name: str | None = None):
        self.max_tokens = max_tokens
        self._model_name = model_name

    def _get_engine(self, model: str):
        if model in _ENGINE_CACHE:
            return _ENGINE_CACHE[model]

        from vllm import LLM

        engine = LLM(model=model, max_model_len=4096, trust_remote_code=True)
        _ENGINE_CACHE[model] = engine
        return engine

    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        from vllm import SamplingParams

        max_tokens = kwargs.get("max_tokens") or self.max_tokens
        engine = self._get_engine(self._model_name or model)

        # Use vLLM's chat API with the tokenizer's chat template
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]

        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        outputs = engine.chat(chat_messages, params)
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text.strip()
        return ""
