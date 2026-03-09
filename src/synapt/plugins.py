"""synapt.plugins — entry-point plugin discovery and loading.

Plugins register via ``[project.entry-points."synapt.plugins"]`` in their
pyproject.toml.  Each entry point must reference a module with a
``register_tools(mcp: FastMCP) -> None`` callable.

Two-phase design: ``discover_plugins()`` loads modules and validates them,
``register_plugins()`` calls ``register_tools()`` on each.  The separation
creates a clean seam for future license checks.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any

logger = logging.getLogger("synapt.plugins")

ENTRY_POINT_GROUP = "synapt.plugins"


class LoadedPlugin:
    """Metadata about a successfully loaded plugin."""

    __slots__ = ("name", "version", "module", "entry_point_name")

    def __init__(self, name: str, version: str, module: Any, entry_point_name: str):
        self.name = name
        self.version = version
        self.module = module
        self.entry_point_name = entry_point_name

    def __repr__(self) -> str:
        return f"LoadedPlugin({self.name!r}, {self.version!r})"


def discover_plugins() -> list[LoadedPlugin]:
    """Discover and load all plugins registered under the 'synapt.plugins' group.

    Returns a list of LoadedPlugin for each successfully loaded plugin.
    Plugins that fail to import or lack register_tools are logged and skipped.
    """
    plugins: list[LoadedPlugin] = []
    eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)

    for ep in eps:
        try:
            module = ep.load()
        except Exception:
            logger.warning("Plugin %r failed to import", ep.name, exc_info=True)
            continue

        if not callable(getattr(module, "register_tools", None)):
            logger.warning(
                "Plugin %r has no register_tools() callable, skipping", ep.name
            )
            continue

        name = getattr(module, "PLUGIN_NAME", ep.name)
        version = getattr(module, "PLUGIN_VERSION", "")
        plugins.append(LoadedPlugin(name, version, module, ep.name))
        logger.debug("Discovered plugin: %s %s", name, version)

    return plugins


def register_plugins(
    mcp: Any, plugins: list[LoadedPlugin] | None = None
) -> list[LoadedPlugin]:
    """Discover plugins (if not provided) and register their tools on the MCP server.

    Returns the list of successfully registered plugins.
    Plugins whose register_tools() raises are logged and skipped.
    """
    if plugins is None:
        plugins = discover_plugins()

    registered: list[LoadedPlugin] = []
    for plugin in plugins:
        try:
            plugin.module.register_tools(mcp)
            registered.append(plugin)
            logger.debug("Registered plugin: %s", plugin.name)
        except Exception:
            logger.warning(
                "Plugin %r register_tools() failed", plugin.name, exc_info=True
            )

    return registered
