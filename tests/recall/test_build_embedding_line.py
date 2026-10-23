"""`build` says "Computing embeddings" only when the build will compute them.

The build always opens a SQLite store, and the DB embedding path returns early
unless the store already holds chunk embeddings. A store without any is
searched by keyword only, so the build must say that instead of promising
embeddings it will not write.
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from synapt.recall import cli
from synapt.recall.storage import pack_embedding

KEYWORD_ONLY = "it searches by keyword only"
SRC = str(Path(cli.__file__).resolve().parents[2])


def _write_transcript(src: Path) -> None:
    src.mkdir(exist_ok=True)
    lines = []
    for i in range(3):
        lines.append({
            "type": "user", "sessionId": "s1", "uuid": f"u{i}",
            "timestamp": f"2026-10-23T00:0{i}:00Z",
            "message": {"role": "user", "content": f"how does the line measure wall {i}?"},
        })
        lines.append({
            "type": "assistant", "sessionId": "s1", "uuid": f"a{i}", "parentUuid": f"u{i}",
            "timestamp": f"2026-10-23T00:0{i}:30Z",
            "message": {"role": "assistant",
                        "content": [{"type": "text", "text": f"It hangs true, answer {i}."}]},
        })
    (src / "s1.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def run_build(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    src = tmp_path / "src"
    _write_transcript(src)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PYTHONPATH": SRC,
        "HF_HUB_OFFLINE": "1",
    }
    if "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    result = subprocess.run(
        [sys.executable, "-B", "-m", "synapt.recall.cli", "build", "--source", str(src)],
        cwd=work, env=env, capture_output=True, text=True, timeout=300,
    )
    stores = sorted(tmp_path.rglob("recall.db"))
    print("store:", [str(s) for s in stores])
    return result, stores


def test_store_without_embeddings_says_keyword_only(tmp_path):
    result, stores = run_build(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert stores, "build wrote no store"
    assert "Computing embeddings" not in result.stdout
    assert KEYWORD_ONLY in result.stdout
    for store in stores:
        conn = sqlite3.connect(store)
        try:
            embedded = conn.execute(
                "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        finally:
            conn.close()
        assert embedded == 0


def test_store_with_embeddings_still_says_computing(tmp_path):
    first, stores = run_build(tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr
    assert stores, "build wrote no store"
    conn = sqlite3.connect(stores[0])
    try:
        conn.execute(
            "UPDATE chunks SET embedding = ? WHERE rowid = (SELECT min(rowid) FROM chunks)",
            (pack_embedding([0.0] * 384),),
        )
        conn.commit()
    finally:
        conn.close()
    second, _ = run_build(tmp_path)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "Computing embeddings" in second.stdout
    assert KEYWORD_ONLY not in second.stdout
