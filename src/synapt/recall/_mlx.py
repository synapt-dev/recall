"""Shared MLX availability guard.

Centralizes the mlx-lm import check so enrich.py and consolidate.py
don't duplicate the try/except ImportError pattern.
"""

from __future__ import annotations

MLX_AVAILABLE = False
try:
    from synapt._models.mlx_client import MLXClient, MLXOptions  # noqa: F401
    from synapt._models.base import Message  # noqa: F401
    MLX_AVAILABLE = True
except ImportError:
    pass

INSTALL_MSG = (
    "MLX is required for this feature.\n"
    "Install with: pip install mlx-lm\n"
    "Then re-run this command."
)
