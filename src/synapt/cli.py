"""synapt -- unified CLI for recall and MCP server.

Usage:
    synapt recall search "query"
    synapt recall build --incremental
    synapt recall stats
    synapt recall setup

    synapt server   # starts unified MCP server
"""

from __future__ import annotations

import sys


def main():
    if len(sys.argv) < 2:
        _print_help()
        sys.exit(1)

    subcmd = sys.argv[1]

    if subcmd in ("-h", "--help", "help"):
        _print_help()
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
    else:
        print(f"Unknown command: {subcmd}", file=sys.stderr)
        print("(Install synapt-repair for repair/watch commands)", file=sys.stderr)
        _print_help()
        sys.exit(1)


def _print_help():
    print("""synapt -- persistent conversational memory for AI coding assistants

Commands:
  recall    Search and manage past session transcripts
  server    Start the unified MCP server

Run 'synapt <command> --help' for details on each command.
""")


if __name__ == "__main__":
    main()
