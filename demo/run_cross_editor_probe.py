#!/usr/bin/env python3
"""Probe the cross-editor demo commands before running the VHS tape.

This gives a bounded smoke test for the three CLIs used in
demo/cross-editor-memory.tape so render failures show up before a multi-minute
VHS run.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROMPT = (
    "Use recall_quick only. Answer in one short line: "
    "what embedding model does this project use and why was it chosen?"
)

DEFAULT_DEMO_ROOT = Path("/tmp/synapt-demo")
DEFAULT_TIMEOUT = 60
VALID_TOOLS = ("claude", "codex", "opencode")


def _commands(prompt: str) -> dict[str, list[str]]:
    return {
        "claude": ["claude", "-p", prompt],
        "codex": [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "-m",
            "gpt-5.4-mini",
            prompt,
        ],
        "opencode": ["opencode", "run", prompt],
    }


def _snippet(text: str, limit: int = 800) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _probe(tool: str, cmd: list[str], cwd: Path, timeout: int) -> int:
    print(f"== {tool} ==")
    binary = shutil.which(cmd[0])
    if not binary:
        print(f"missing: {cmd[0]}")
        return 1

    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        print(f"timeout after {timeout}s")
        stdout = exc.stdout if isinstance(exc.stdout, str) else (
            exc.stdout or b""
        ).decode("utf-8", "ignore")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (
            exc.stderr or b""
        ).decode("utf-8", "ignore")
        if stdout.strip():
            print("stdout:")
            print(_snippet(stdout))
        if stderr.strip():
            print("stderr:")
            print(_snippet(stderr))
        return 1

    print(f"rc={proc.returncode}")
    if proc.stdout.strip():
        print("stdout:")
        print(_snippet(proc.stdout))
    if proc.stderr.strip():
        print("stderr:")
        print(_snippet(proc.stderr))
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

    commands = _commands(PROMPT)
    selected = args.tools or list(VALID_TOOLS)
    failures = 0
    for tool in selected:
        failures += _probe(tool, commands[tool], demo_root, args.timeout)

    if failures:
        print(f"\n{failures} probe(s) failed.")
        return 1

    print("\nAll selected probes completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
