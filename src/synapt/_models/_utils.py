"""Shared utilities for model clients."""

from __future__ import annotations

import json
import os


def read_adapter_config(model: str) -> dict | None:
    """Read adapter_config.json from a local path or HuggingFace repo.

    Returns the parsed config dict, or None if not a PEFT adapter.
    """
    if os.path.isdir(model):
        cfg_path = os.path.join(model, "adapter_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                return json.load(f)
        return None

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model, "adapter_config.json")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def fix_bare_json(text: str) -> str:
    """Wrap bare JSON fields in braces.

    T5 models fine-tuned for JSON output often drop the outer braces,
    producing '"key": "value"' instead of '{"key": "value"}'.
    """
    if text and not text.startswith("{") and text.startswith('"'):
        try:
            json.loads("{" + text + "}")
            return "{" + text + "}"
        except (json.JSONDecodeError, ValueError):
            pass
    return text


def get_onnx_cache_dir(model_name: str) -> str:
    """Get the cache directory for a converted ONNX model."""
    cache_dir = os.path.expanduser("~/.synapt/models/onnx")
    safe_name = model_name.replace("/", "--")
    return os.path.join(cache_dir, safe_name)
