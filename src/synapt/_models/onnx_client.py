"""ONNX Runtime inference client for encoder-decoder models.

Provides 5-7x faster inference than PyTorch via ONNX Runtime with
graph optimization and KV cache management. Particularly effective
for T5 models where autoregressive decoding dominates latency.

Benchmarks (T5-base enrichment, M2 Air, ~125 output tokens):
  - PyTorch CPU:       5011ms (baseline)
  - ONNX FP32 4T OPT:  760ms (6.6x)

Requires: pip install 'synapt[onnx]'
"""

from __future__ import annotations

import logging
import os

from .base import Message, ModelClient
from ._utils import fix_bare_json, get_onnx_cache_dir

logger = logging.getLogger(__name__)

# Module-level cache: model_path → (model, tokenizer)
_ONNX_CACHE: dict[str, tuple[object, object]] = {}

# Default thread count — 4 threads works best on Apple Silicon
# (matches performance core count on M1/M2 Air)
_DEFAULT_THREADS = 4


def _find_onnx_model(model_name: str) -> str | None:
    """Find a pre-converted ONNX model directory.

    Checks (in order):
    1. Local directory if model_name is a path with ONNX files
    2. Standard cache location ~/.synapt/models/onnx/<model>/
    """
    # Direct local path
    if os.path.isdir(model_name):
        if any(f.endswith(".onnx") for f in os.listdir(model_name)):
            return model_name

    # Check synapt cache
    cached = get_onnx_cache_dir(model_name)
    if os.path.isdir(cached) and any(f.endswith(".onnx") for f in os.listdir(cached)):
        return cached

    return None


class OnnxClient(ModelClient):
    """Encoder-decoder model client via ONNX Runtime.

    Uses optimum's ORTModelForSeq2SeqLM for proper KV cache handling
    during autoregressive decoding.
    """

    def __init__(
        self,
        max_tokens: int = 300,
        model_name: str | None = None,
        num_threads: int | None = None,
    ):
        self.max_tokens = max_tokens
        self._model_name = model_name
        self._num_threads = num_threads or int(
            os.environ.get("SYNAPT_ONNX_THREADS", str(_DEFAULT_THREADS))
        )

    @staticmethod
    def is_available(model_name: str) -> bool:
        """Check if a converted ONNX model exists for the given name."""
        return _find_onnx_model(model_name) is not None

    def _load(self, model: str) -> tuple[object, object]:
        """Load or retrieve cached ONNX model and tokenizer."""
        if model in _ONNX_CACHE:
            return _ONNX_CACHE[model]

        import onnxruntime
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer

        onnx_dir = _find_onnx_model(model)
        if onnx_dir is None:
            raise FileNotFoundError(
                f"No ONNX model found for {model}. "
                "Run 'synapt-convert' to create one."
            )

        sess_opts = onnxruntime.SessionOptions()
        sess_opts.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_opts.intra_op_num_threads = self._num_threads
        sess_opts.inter_op_num_threads = 1

        logger.info(
            "Loading ONNX model from %s (threads=%d)", onnx_dir, self._num_threads
        )
        model_obj = ORTModelForSeq2SeqLM.from_pretrained(
            onnx_dir, session_options=sess_opts
        )
        tokenizer = AutoTokenizer.from_pretrained(onnx_dir)

        _ONNX_CACHE[model] = (model_obj, tokenizer)
        return model_obj, tokenizer

    def chat(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        """Generate text using ONNX Runtime inference."""
        actual_model = self._model_name or model
        model_obj, tokenizer = self._load(actual_model)

        text = "\n".join(m.content for m in messages if m.content)

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        num_beams = kwargs.get("num_beams", 1)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )

        gen_kwargs = {"max_new_tokens": max_tokens, "num_beams": num_beams}
        if num_beams > 1:
            gen_kwargs["no_repeat_ngram_size"] = 3
            gen_kwargs["length_penalty"] = 2.0

        outputs = model_obj.generate(**inputs, **gen_kwargs)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return fix_bare_json(result)
