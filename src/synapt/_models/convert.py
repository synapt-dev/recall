"""Convert T5 + LoRA adapter to optimized ONNX format.

Merges LoRA weights into the base model and exports to ONNX via optimum.
ONNX Runtime provides 5-7x speedup over PyTorch CPU via graph optimization
and KV cache management.

Usage:
    python -m synapt._models.convert [--model laynepro/t5-enrichment-v2]
    # or via CLI:
    synapt-convert
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile

from ._utils import get_onnx_cache_dir, read_adapter_config

logger = logging.getLogger(__name__)


def convert(
    model_name: str = "laynepro/t5-enrichment-v2",
    base_model: str | None = None,
    output_dir: str | None = None,
    quantize: bool = False,
) -> str:
    """Convert a T5 + LoRA adapter to optimized ONNX.

    ONNX Runtime provides 5-7x speedup over PyTorch CPU via graph
    optimization and KV cache management — no quantization needed.

    Args:
        model_name: HuggingFace model ID or local path (adapter or full model).
        base_model: Base model name. Auto-detected from adapter_config.json.
        output_dir: Where to save. Defaults to ~/.synapt/models/onnx/<model>/
        quantize: Apply INT8 dynamic quantization (marginal gain on ARM).

    Returns:
        Path to the output directory containing ONNX files.
    """
    output_dir = output_dir or get_onnx_cache_dir(model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Detect adapter and merge if needed
    merged_dir = tempfile.mkdtemp(prefix="synapt-merge-")
    try:
        model_obj, tokenizer = _load_and_merge(model_name, base_model)
        logger.info("Saving merged model to %s", merged_dir)
        model_obj.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        # Step 2: Export to ONNX
        logger.info("Exporting to ONNX...")
        _export_onnx(merged_dir, output_dir, tokenizer)

        # Step 3: Quantize (optional, marginal gain on ARM)
        if quantize:
            logger.info("Applying INT8 quantization...")
            _quantize_onnx(output_dir)

        # Save conversion metadata
        metadata = {
            "source_model": model_name,
            "base_model": base_model,
            "quantized": quantize,
            "format": "onnx",
        }
        with open(os.path.join(output_dir, "synapt_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Conversion complete: %s", output_dir)
        return output_dir

    finally:
        shutil.rmtree(merged_dir, ignore_errors=True)


def _load_and_merge(
    model_name: str, base_model: str | None
) -> tuple[object, object]:
    """Load model, detecting and merging PEFT adapters if present."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    adapter_cfg = read_adapter_config(model_name)

    if adapter_cfg is not None:
        from peft import PeftModel

        base_name = base_model or adapter_cfg.get(
            "base_model_name_or_path", "google/flan-t5-base"
        )
        logger.info("Merging adapter %s into %s", model_name, base_name)

        base_obj = AutoModelForSeq2SeqLM.from_pretrained(base_name)
        peft_obj = PeftModel.from_pretrained(base_obj, model_name)
        merged = peft_obj.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        return merged, tokenizer
    else:
        logger.info("Loading model directly: %s", model_name)
        model_obj = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model_obj, tokenizer


def _export_onnx(merged_dir: str, output_dir: str, tokenizer) -> None:
    """Export merged model to ONNX using optimum."""
    from optimum.onnxruntime import ORTModelForSeq2SeqLM

    ort_model = ORTModelForSeq2SeqLM.from_pretrained(merged_dir, export=True)
    ort_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _quantize_onnx(onnx_dir: str) -> None:
    """Apply INT8 dynamic quantization to ONNX models in-place."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    qconfig = AutoQuantizationConfig.avx2(is_static=False)

    for fname in ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"]:
        fpath = os.path.join(onnx_dir, fname)
        if not os.path.exists(fpath):
            continue

        quantizer = ORTQuantizer.from_pretrained(onnx_dir, file_name=fname)
        quantizer.quantize(save_dir=onnx_dir, quantization_config=qconfig)
        quantized_name = fname.replace(".onnx", "_quantized.onnx")
        quantized_path = os.path.join(onnx_dir, quantized_name)

        if os.path.exists(quantized_path):
            os.replace(quantized_path, fpath)
            logger.info("Quantized %s", fname)


def is_converted(model_name: str) -> bool:
    """Check if a model has already been converted to ONNX."""
    from .onnx_client import _find_onnx_model

    return _find_onnx_model(model_name) is not None


def main():
    """CLI entry point for model conversion."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Convert T5 model to ONNX")
    parser.add_argument(
        "--model",
        default="laynepro/t5-enrichment-v2",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--base-model", help="Base model (auto-detected from adapter)")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument(
        "--quantize", action="store_true", help="Apply INT8 quantization (marginal gain on ARM)"
    )
    args = parser.parse_args()

    output = convert(
        model_name=args.model,
        base_model=args.base_model,
        output_dir=args.output,
        quantize=args.quantize,
    )
    print(f"Model saved to: {output}")


if __name__ == "__main__":
    main()
