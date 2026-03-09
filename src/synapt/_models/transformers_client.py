"""Encoder-decoder model client using HuggingFace Transformers.

Designed for text-to-text tasks (summarization, extraction) where
input is long and output is short. The encoder processes input in
parallel — 7-15x faster than autoregressive decoder-only models
for this workload shape.

Uses MPS (Metal Performance Shaders) on Apple Silicon for GPU acceleration.
Falls back to CPU when MPS is unavailable.

Requires: pip install 'synapt[transformers]'
Models: flan-t5-base (250M, ~1GB FP32), flan-t5-large (780M, ~3.2GB FP32)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from .base import Message, ModelClient

logger = logging.getLogger(__name__)

# Class-level cache: model_name → (model, tokenizer, device)
_MODEL_CACHE: Dict[str, Tuple[object, object, str]] = {}


def _resolve_device(preference: str = "auto") -> str:
    """Select the best available device: MPS > CPU."""
    if preference != "auto":
        return preference
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except (ImportError, AttributeError):
        pass
    return "cpu"


class TransformersClient(ModelClient):
    """Encoder-decoder model client (T5 family) via HuggingFace Transformers.

    Unlike decoder-only models that process input autoregressively,
    the encoder processes the full input in one parallel forward pass.
    The decoder only runs for the (short) output tokens.
    """

    def __init__(self, max_tokens: int = 300, device: str = "auto"):
        self.max_tokens = max_tokens
        self._device = device

    def _load(self, model: str) -> Tuple[object, object, str]:
        """Load or retrieve cached model and tokenizer."""
        if model in _MODEL_CACHE:
            return _MODEL_CACHE[model]

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = _resolve_device(self._device)
        logger.info("Loading %s on %s", model, device)

        tokenizer = AutoTokenizer.from_pretrained(model)
        model_obj = AutoModelForSeq2SeqLM.from_pretrained(model)
        model_obj = model_obj.to(device)
        model_obj.eval()

        _MODEL_CACHE[model] = (model_obj, tokenizer, device)
        return model_obj, tokenizer, device

    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        """Generate text using encoder-decoder inference.

        Flattens chat messages into a single text input — encoder-decoder
        models don't use chat templates. The full text is encoded in
        parallel, then the decoder generates the output autoregressively.

        Uses beam search (num_beams=4) by default. T5 models produce
        degenerate short outputs with sampling — beam search explores
        multiple hypotheses and finds longer, more informative completions.
        """
        import torch

        model_obj, tokenizer, device = self._load(model)

        # Flatten messages to single text input
        text = "\n".join(m.content for m in messages if m.content)

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        num_beams = kwargs.get("num_beams", 4)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            if num_beams > 1:
                # Beam search: better quality for encoder-decoder models
                outputs = model_obj.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_beams=num_beams,
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                )
            else:
                # Sampling fallback
                outputs = model_obj.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0,
                    num_beams=1,
                )

        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
