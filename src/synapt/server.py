"""synapt.server — unified MCP server composing recall + discovered plugins.

Recall is built-in (core OSS).  Repair, watch, and any third-party tools
are discovered at startup via ``importlib.metadata`` entry points in the
``synapt.plugins`` group.  See ``synapt.plugins`` for the plugin protocol.

Run with:
    synapt server

Add to Claude Code (.mcp.json):
    {"mcpServers": {"synapt": {"type": "stdio", "command": "synapt", "args": ["server"]}}}
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import sys


def _handle_sigterm(signum, frame):
    """Convert SIGTERM into a clean exit so atexit handlers run.

    sys.exit(0) doesn't work reliably inside asyncio event loops, so
    we manually run atexit callbacks and os._exit(0) to ensure a zero
    exit code.
    """
    atexit._run_exitfuncs()
    os._exit(0)


# Install before anything else — ensures clean exit code when Claude
# Code sends SIGTERM on session end. Without this, SIGTERM kills the
# process with exit code -15, which Claude reports as "MCP tool failed."
signal.signal(signal.SIGTERM, _handle_sigterm)

logger = logging.getLogger("synapt.server")


def main():
    """Entry point for the unified synapt MCP server."""
    from mcp.server.fastmcp import FastMCP

    from synapt.plugins import register_plugins
    from synapt.recall.server import MCP_INSTRUCTIONS, register_tools as recall_register

    mcp = FastMCP(
        "synapt",
        instructions=MCP_INSTRUCTIONS,
    )

    # Core: always available
    recall_register(mcp)

    # Plugins: discovered via entry points, gracefully degraded
    registered = register_plugins(mcp)
    for p in registered:
        logger.debug("Loaded plugin: %s %s", p.name, p.version or "(no version)")

    mcp.run()


if __name__ == "__main__":
    main()
