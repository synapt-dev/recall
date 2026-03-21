#!/usr/bin/env python3
"""Prepare the cross-editor demo project and build its synapt index.

Creates /tmp/synapt-demo from CodeMemo project_03_memory_system sessions so the
VHS tape can run against a real, reproducible memory index.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = Path("/tmp/synapt-demo")
PROJECT_NAME = "project_03_memory_system"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_codememo_project() -> Path:
    try:
        from evaluation.codememo.eval import _get_data_dir
    except Exception as exc:  # pragma: no cover - import failure is user-facing
        raise RuntimeError(f"Could not import CodeMemo data loader: {exc}") from exc

    project_dir = _get_data_dir() / PROJECT_NAME
    sessions_dir = project_dir / "sessions"
    if not sessions_dir.is_dir():
        raise FileNotFoundError(
            f"CodeMemo project sessions not found at {sessions_dir}. "
            "Install demo dependencies and make sure the benchmark dataset is available."
        )
    return project_dir


def _reset_demo_root(demo_root: Path) -> None:
    if demo_root.exists():
        shutil.rmtree(demo_root)
    demo_root.mkdir(parents=True)


def _materialize_project_files(project_dir: Path, demo_root: Path) -> Path:
    sessions_src = project_dir / "sessions"
    sessions_dst = demo_root / "sessions"
    sessions_dst.mkdir(parents=True)

    for session_path in sorted(sessions_src.glob("*.jsonl")):
        (sessions_dst / session_path.name).symlink_to(session_path.resolve())

    manifest = project_dir / "manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, demo_root / "manifest.json")

    questions = project_dir / "questions.json"
    if questions.exists():
        shutil.copy2(questions, demo_root / "questions.json")

    readme = demo_root / "README.md"
    readme.write_text(
        "# synapt cross-editor demo\n\n"
        "This temp project is generated from CodeMemo project_03_memory_system.\n"
        "It exists only to power `demo/cross-editor-memory.tape`.\n",
        encoding="utf-8",
    )

    return sessions_dst


def _build_index(demo_root: Path, sessions_dir: Path, no_embeddings: bool) -> None:
    cmd = [
        sys.executable,
        "-m",
        "synapt.cli",
        "recall",
        "build",
        "--source",
        str(sessions_dir),
    ]
    if no_embeddings:
        cmd.append("--no-embeddings")

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    repo_src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = repo_src if not existing else f"{repo_src}{os.pathsep}{existing}"
    if no_embeddings:
        # Keep the local smoke path fast and fully laptop-friendly.
        env["SYNAPT_DISABLE_CLUSTERS"] = "1"

    subprocess.run(cmd, cwd=demo_root, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-root",
        type=Path,
        default=DEMO_ROOT,
        help="Output demo project directory (default: /tmp/synapt-demo)",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding generation for a faster local setup",
    )
    args = parser.parse_args()

    demo_root = args.demo_root.expanduser().resolve()
    project_dir = _resolve_codememo_project()

    print(f"[demo] Using CodeMemo source: {project_dir}")
    print(f"[demo] Creating demo project at: {demo_root}")
    _reset_demo_root(demo_root)
    sessions_dir = _materialize_project_files(project_dir, demo_root)

    print("[demo] Building synapt recall index...")
    _build_index(demo_root, sessions_dir, no_embeddings=args.no_embeddings)

    print("[demo] Ready.")
    print(f"  cd {demo_root}")
    print("  vhs demo/cross-editor-memory.tape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
