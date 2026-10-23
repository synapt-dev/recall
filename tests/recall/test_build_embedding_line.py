"""`build` says "Computing embeddings" only when the build will compute them.

The build always opens a SQLite store, and the DB embedding path returns early
unless the store already holds chunk embeddings. A store without any is
searched by keyword only, so the build must say that instead of promising
embeddings it will not write.
"""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from synapt.recall import cli
from synapt.recall.core import TranscriptChunk
from synapt.recall.generations import current_generation_dir
from synapt.recall.sharded_db import ShardedRecallDB
from synapt.recall.storage import RecallDB, pack_embedding

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


def test_init_store_without_embeddings_says_keyword_only(tmp_path):
    home = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir()
    slug = str(work.resolve()).replace("\\", "-").replace("/", "-").replace(":", "-")
    transcripts = home / ".claude" / "projects" / slug
    transcripts.parent.mkdir(parents=True)
    _write_transcript(transcripts)
    # The claude CLI is kept off PATH so init cannot register anything outside tmp_path.
    env = {
        "PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"]),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PYTHONPATH": SRC,
        "HF_HUB_OFFLINE": "1",
    }
    if "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    result = subprocess.run(
        [sys.executable, "-B", "-m", "synapt.cli", "init", "--no-hook"],
        cwd=work, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Index saved:" in result.stdout, result.stdout + result.stderr
    assert "Computing embeddings" not in result.stdout
    assert KEYWORD_ONLY in result.stdout


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def _is_wal_mode(path: Path) -> bool:
    # SQLite header bytes 18/19 (file format read/write version) are 2 in WAL mode.
    header = path.read_bytes()[:20]
    return header[18] == 2 and header[19] == 2


def _revert_chunk_links_migration(index_dir: Path) -> None:
    """Put every store file one migration behind: the rowid-keyed chunk_links
    that ``RecallDB._migrate_chunk_links_table`` rewrites on a read-write open."""
    files = [index_dir / "index.db", *current_generation_dir(index_dir).glob("data_*.db")]
    for path in files:
        conn = sqlite3.connect(path)
        try:
            # recall's own chunk_links DDL from before that migration.
            conn.executescript(
                "DROP TABLE chunk_links;"
                "CREATE TABLE chunk_links ("
                " source_rowid INTEGER NOT NULL, target_rowid INTEGER NOT NULL,"
                " similarity REAL NOT NULL, PRIMARY KEY (source_rowid, target_rowid));"
                "CREATE INDEX idx_chunk_links_source ON chunk_links(source_rowid);"
            )
            conn.commit()
        finally:
            conn.close()
        assert not Path(f"{path}-wal").exists(), f"{path.name}-wal left by the revert"


@pytest.mark.parametrize("store", ["current", "one_migration_behind"])
def test_embedding_check_leaves_a_sharded_store_untouched(tmp_path, monkeypatch, store):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    RecallDB(index_dir / "index.db").close()
    db = ShardedRecallDB.open(index_dir)
    try:
        db.save_chunks([
            TranscriptChunk(
                id=f"s1:t{i}", session_id="s1", timestamp=f"2026-10-23T00:0{i}:00Z",
                turn_index=i, user_text=f"question {i}", assistant_text=f"answer {i}",
            )
            for i in range(3)
        ])
    finally:
        db.close()
    assert current_generation_dir(index_dir) is not None, "save path published no generation"
    data_shards = sorted(current_generation_dir(index_dir).glob("data_*.db"))
    assert data_shards, "published generation holds no data shard"
    if store == "one_migration_behind":
        _revert_chunk_links_migration(index_dir)
    monkeypatch.setattr(cli, "project_index_dir", lambda project: index_dir)

    before = _snapshot(index_dir)
    assert cli._store_has_embeddings(tmp_path) is False
    after = _snapshot(index_dir)

    # Every file that existed is byte-identical.
    assert {name: after.get(name) for name in before} == before
    # The only new files allowed: an empty -wal and a -shm beside a WAL-mode data shard.
    allowed = {
        str(Path(f"{shard}-{suffix}").relative_to(index_dir))
        for shard in data_shards if _is_wal_mode(shard)
        for suffix in ("wal", "shm")
    }
    new_files = set(after) - set(before)
    assert new_files <= allowed, sorted(new_files - allowed)
    for name in new_files:
        if name.endswith("-wal"):
            assert (index_dir / name).stat().st_size == 0, name

    # No lock held: a second writer opens and writes at once.
    for path in [index_dir / "index.db", *data_shards]:
        conn = sqlite3.connect(path, timeout=0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("CREATE TABLE probe_after_check (x)")
            conn.commit()
        finally:
            conn.close()
