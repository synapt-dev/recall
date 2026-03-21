#!/usr/bin/env python3
"""Probe the cross-editor demo commands before running the VHS tape.

This gives a bounded smoke test for the three CLIs used in
demo/cross-editor-memory.tape so render failures show up before a multi-minute
VHS run.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


DEFAULT_DEMO_ROOT = Path("/tmp/synapt-demo")
DEFAULT_TIMEOUT = 60
VALID_TOOLS = ("claude", "codex", "opencode")
HELPER_NAME = "run_cross_editor_query.sh"


def _command(cwd: Path, tool: str) -> list[str]:
    return [str(cwd / HELPER_NAME), tool]


def _snippet(text: str, limit: int = 800) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _probe(tool: str, cmd: list[str], cwd: Path, timeout: int) -> int:
    print(f"== {tool} ==")
    if not Path(cmd[0]).exists():
        print(f"missing helper: {cmd[0]}")
        return 1

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"timeout after {timeout}s")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        if stdout.strip():
            print("stdout:")
            print(_snippet(stdout))
        if stderr.strip():
            print("stderr:")
            print(_snippet(stderr))
        return 1

    print(f"rc={proc.returncode}")
    if stdout.strip():
        print("stdout:")
        print(_snippet(stdout))
    if stderr.strip():
        print("stderr:")
        print(_snippet(stderr))
    return 0 if proc.returncode == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe the cross-editor demo commands with a bounded timeout.",
    )
    parser.add_argument(
        "--demo-root",
        type=Path,
        default=DEFAULT_DEMO_ROOT,
        help="Demo project directory used by the VHS tape (default: /tmp/synapt-demo)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-command timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--tool",
        action="append",
        choices=VALID_TOOLS,
        dest="tools",
        help="Restrict probing to one or more tools (repeatable).",
    )
    args = parser.parse_args()

    demo_root = args.demo_root.expanduser().resolve()
    if not demo_root.exists():
        print(f"demo root not found: {demo_root}", file=sys.stderr)
        print(
            "Run python demo/build_demo_index.py first.",
            file=sys.stderr,
        )
        return 1
    if not (demo_root / HELPER_NAME).exists():
        print(
            f"demo helper not found: {demo_root / HELPER_NAME}",
            file=sys.stderr,
        )
        print(
            "Re-run python demo/build_demo_index.py to refresh the demo files.",
            file=sys.stderr,
        )
        return 1

    selected = args.tools or list(VALID_TOOLS)
    failures = 0
    for tool in selected:
        failures += _probe(tool, _command(demo_root, tool), demo_root, args.timeout)

    if failures:
        print(f"\n{failures} probe(s) failed.")
        return 1

    print("\nAll selected probes completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
