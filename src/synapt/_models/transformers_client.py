"""Encoder-decoder model client using HuggingFace Transformers.

Designed for text-to-text tasks (summarization, extraction) where
input is long and output is short. The encoder processes input in
parallel — 7-15x faster than autoregressive decoder-only models
for this workload shape.

Runs on CPU by default. MPS (Metal Performance Shaders) has high
per-kernel-launch overhead that makes autoregressive token generation
slower than CPU for small models like T5-base.

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
    """Select the best available device.

    Defaults to CPU. MPS is available but not recommended for T5-base
    due to high per-kernel overhead during autoregressive decoding.
    Use preference='mps' to override for larger models.
    """
    if preference != "auto":
        return preference
    # CPU is faster than MPS for small encoder-decoder models (T5-base)
    # due to MPS kernel launch overhead during autoregressive decoding.
    return "cpu"


class TransformersClient(ModelClient):
    """Encoder-decoder model client (T5 family) via HuggingFace Transformers.

    Unlike decoder-only models that process input autoregressively,
    the encoder processes the full input in one parallel forward pass.
    The decoder only runs for the (short) output tokens.
    """

    def __init__(
        self,
        max_tokens: int = 300,
        device: str = "auto",
        model_name: str | None = None,
    ):
        self.max_tokens = max_tokens
        self._device = device
        self._model_name = model_name  # Override: always use this model

    def _load(self, model: str) -> Tuple[object, object, str]:
        """Load or retrieve cached model and tokenizer.

        Detects PEFT adapters (local dir or HuggingFace repo) by looking for
        adapter_config.json. When found, loads the base model + LoRA adapter.
        Otherwise loads the model directly.
        """
        if model in _MODEL_CACHE:
            return _MODEL_CACHE[model]

        import os
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = _resolve_device(self._device)

        # Detect PEFT adapter: local directory or HuggingFace repo
        adapter_cfg = self._read_adapter_config(model)
        if adapter_cfg is not None:
            base_model_name = adapter_cfg.get("base_model_name_or_path", "google/flan-t5-base")
            logger.info("Loading %s + LoRA adapter %s on %s", base_model_name, model, device)

            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_obj = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
            model_obj = PeftModel.from_pretrained(base_obj, model)
        else:
            logger.info("Loading %s on %s", model, device)
            tokenizer = AutoTokenizer.from_pretrained(model)
            model_obj = AutoModelForSeq2SeqLM.from_pretrained(model)

        model_obj = model_obj.to(device)
        model_obj.eval()

        _MODEL_CACHE[model] = (model_obj, tokenizer, device)
        return model_obj, tokenizer, device

    @staticmethod
    def _read_adapter_config(model: str) -> dict | None:
        """Read adapter_config.json from a local path or HuggingFace repo.

        Returns the parsed config dict, or None if not a PEFT adapter.
        """
        import os
        import json as _json

        # Local directory
        if os.path.isdir(model):
            cfg_path = os.path.join(model, "adapter_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    return _json.load(f)
            return None

        # HuggingFace repo — try downloading adapter_config.json
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model, "adapter_config.json")
            with open(path) as f:
                return _json.load(f)
        except Exception:
            return None

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

        Uses greedy decoding by default. Fine-tuned T5 models produce
        more accurate structured output with greedy than beam search.
        Pass num_beams=4 via kwargs for plain text summarization tasks.
        """
        import torch

        # Use configured model override if set
        actual_model = self._model_name or model
        model_obj, tokenizer, device = self._load(actual_model)

        # Flatten messages to single text input
        text = "\n".join(m.content for m in messages if m.content)

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        num_beams = kwargs.get("num_beams", 1)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            if num_beams > 1:
                # Beam search: better for plain text summarization
                outputs = model_obj.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_beams=num_beams,
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                )
            else:
                # Greedy: faster and more accurate for structured output
                outputs = model_obj.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_beams=1,
                )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # T5 fine-tuned for JSON output often drops outer braces.
        # Wrap in {} if the output looks like bare JSON fields.
        if result and not result.startswith("{") and result.startswith('"'):
            import json
            try:
                json.loads("{" + result + "}")
                result = "{" + result + "}"
            except (json.JSONDecodeError, ValueError):
                pass

        return result
