"""Packaging tests for the standalone ``langchain-synapt`` distribution."""

from __future__ import annotations

import subprocess
import sys
import venv
import shutil
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "langchain-synapt"


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        # The command's own output is the only record of WHY it failed (pip's network or resolution error), and a
        # bare CalledProcessError prints neither stream, so put both tails in the failure.
        raise RuntimeError(
            f"{cmd!r} exit {result.returncode}\n"
            f"--- stdout (last 4000 chars) ---\n{result.stdout[-4000:]}\n"
            f"--- stderr (last 4000 chars) ---\n{result.stderr[-4000:]}"
        )
    return result


def _rebuild_dist(target: Path) -> Path:
    dist_dir = target / "dist"
    shutil.rmtree(dist_dir, ignore_errors=True)
    _run([sys.executable, "-m", "build", str(target)])
    wheels = sorted(
        dist_dir.glob("*.whl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    assert wheels, f"expected wheel build output in {dist_dir}"
    return wheels[0]


def test_langchain_synapt_package_has_required_files() -> None:
    assert (PACKAGE_DIR / "pyproject.toml").exists()
    assert (PACKAGE_DIR / "README.md").exists()
    assert (PACKAGE_DIR / "src" / "langchain_synapt" / "__init__.py").exists()


@pytest.mark.network
def test_langchain_synapt_builds_wheel() -> None:
    wheel = _rebuild_dist(PACKAGE_DIR)
    assert wheel.name.startswith("langchain_synapt-")


@pytest.mark.network
def test_langchain_synapt_installs_and_exports_history_class(tmp_path: Path) -> None:
    synapt_wheel = _rebuild_dist(REPO_ROOT)
    langchain_wheel = _rebuild_dist(PACKAGE_DIR)

    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True)
    python = _venv_python(venv_dir)
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


def test_a_failed_command_names_its_own_output() -> None:
    """A red here must say why. pip's own error (network, resolution, a bad wheel) is in its stdout and stderr, and
    a bare CalledProcessError prints neither, so a promotion PR's red was unattributable."""
    with pytest.raises(RuntimeError) as failed:
        _run([sys.executable, "-c", "import sys; print('out-line'); sys.stderr.write('err-line'); sys.exit(3)"])
    message = str(failed.value)
    assert "exit 3" in message and "out-line" in message and "err-line" in message, message


def test_only_the_tests_that_reach_pypi_carry_the_network_mark() -> None:
    """`python -m build` (build isolation fetches the backend) and `pip install langchain-core` reach PyPI; CI runs
    exactly these in their own job, so a PyPI hiccup cannot redden the required matrix."""
    marked = {
        name
        for name, fn in globals().items()
        if name.startswith("test_") and any(m.name == "network" for m in getattr(fn, "pytestmark", []))
    }
    assert marked == {"test_langchain_synapt_builds_wheel", "test_langchain_synapt_installs_and_exports_history_class"}
