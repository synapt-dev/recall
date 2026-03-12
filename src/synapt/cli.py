"""synapt -- unified CLI for recall and MCP server.

Usage:
    synapt recall search "query"
    synapt recall build --incremental
    synapt recall stats
    synapt recall setup

    synapt server   # starts unified MCP server

Plugins can register additional subcommands via the ``synapt.commands``
entry-point group (e.g. ``repair``, ``watch``).
"""

from __future__ import annotations

import importlib.metadata
import sys


def _discover_commands() -> dict[str, callable]:
    """Discover plugin subcommands via ``synapt.commands`` entry points.

    Each entry point should reference a callable ``main()`` function.
    """
    commands: dict[str, callable] = {}
    for ep in importlib.metadata.entry_points(group="synapt.commands"):
        commands[ep.name] = ep
    return commands


def main():
    extra_commands = _discover_commands()

    if len(sys.argv) < 2:
        _print_help(extra_commands)
        sys.exit(1)

    subcmd = sys.argv[1]

    if subcmd in ("-h", "--help", "help"):
        _print_help(extra_commands)
        return

    if subcmd == "--version":
        from synapt import __version__

        print(f"synapt {__version__}")
        return

    # Remove the subcommand from argv so each sub-CLI sees correct args
    sys.argv = [f"synapt {subcmd}"] + sys.argv[2:]

    if subcmd == "recall":
        from synapt.recall.cli import main as recall_main

        recall_main()
    elif subcmd == "server":
        from synapt.server import main as server_main

        server_main()
    elif subcmd in extra_commands:
        ep = extra_commands[subcmd]
        cli_main = ep.load()
        cli_main()
    else:
        print(f"Unknown command: {subcmd}", file=sys.stderr)
        _print_help(extra_commands)
        sys.exit(1)


def _print_help(extra_commands: dict | None = None):
    lines = [
        "synapt -- persistent conversational memory for AI coding assistants",
        "",
        "Commands:",
        "  recall    Search and manage past session transcripts",
        "  server    Start the unified MCP server",
    ]
    if extra_commands:
        for name in sorted(extra_commands):
            lines.append(f"  {name:<9} (from synapt-private)")
    lines.append("")
    lines.append("Run 'synapt <command> --help' for details on each command.")
    lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
