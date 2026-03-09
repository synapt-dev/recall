"""synapt._models — shared model client abstractions (MLX, Ollama)."""

from synapt._models.base import Message, ModelClient  # noqa: F401

__all__ = ["Message", "ModelClient"]
