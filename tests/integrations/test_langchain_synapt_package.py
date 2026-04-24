"""Packaging tests for the standalone ``langchain-synapt`` distribution."""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "langchain-synapt"


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def test_langchain_synapt_package_has_required_files() -> None:
    assert (PACKAGE_DIR / "pyproject.toml").exists()
    assert (PACKAGE_DIR / "README.md").exists()
    assert (PACKAGE_DIR / "src" / "langchain_synapt" / "__init__.py").exists()


def test_langchain_synapt_builds_wheel() -> None:
    _run([sys.executable, "-m", "build", str(PACKAGE_DIR)])
    dist_dir = PACKAGE_DIR / "dist"
    wheels = list(dist_dir.glob("langchain_synapt-*.whl"))
    assert wheels, "expected standalone langchain-synapt wheel to be built"


def test_langchain_synapt_installs_and_exports_history_class(tmp_path: Path) -> None:
    _run([sys.executable, "-m", "build", str(REPO_ROOT)])
    _run([sys.executable, "-m", "build", str(PACKAGE_DIR)])

    synapt_wheel = next((REPO_ROOT / "dist").glob("synapt-*.whl"))
    langchain_wheel = next((PACKAGE_DIR / "dist").glob("langchain_synapt-*.whl"))

    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True)
    python = venv_dir / "bin" / "python"
    pip = [str(python), "-m", "pip"]

    _run(pip + ["install", "langchain-core>=0.2"])
    _run(pip + ["install", "--no-deps", str(synapt_wheel)])
    _run(pip + ["install", "--no-deps", str(langchain_wheel)])

    result = _run(
        [
            str(python),
            "-c",
            (
                "from pathlib import Path; "
                "from langchain_core.messages import HumanMessage; "
                "from langchain_synapt import SynaptChatMessageHistory; "
                "db = Path('history.db'); "
                "history = SynaptChatMessageHistory(session_id='pkg-test', db_path=db); "
                "history.add_messages([HumanMessage(content='hello from wheel')]); "
                "print(history.messages[0].content)"
            ),
        ]
    )
    assert result.stdout.strip() == "hello from wheel"
