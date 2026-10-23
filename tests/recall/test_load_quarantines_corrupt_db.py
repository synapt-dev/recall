"""TranscriptIndex.load moves a recall.db that is not a database aside, and never deletes a store."""

import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from unittest import mock

from synapt.recall import core
from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall import storage
from synapt.recall.storage import RecallDB

JUNK = b"this is not a sqlite database, it is a recall store a user cares about\n" * 20


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corrupt(directory: Path, wal: bool = True) -> str:
    db = directory / "recall.db"
    db.write_bytes(JUNK)
    if wal:
        (directory / "recall.db-wal").write_bytes(b"wal bytes")
        (directory / "recall.db-shm").write_bytes(b"shm bytes")
    return _sha(db)


def _healthy(directory: Path) -> str:
    RecallDB(directory / "recall.db").close()
    return _sha(directory / "recall.db")


def _quarantined(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.name.startswith("recall.db.corrupt-") and not p.name.endswith(("-wal", "-shm"))
    )


def _measure(row: str, directory: Path, sha: str, outcome: str) -> None:
    db = directory / "recall.db"
    same = db.exists() and _sha(db) == sha
    print(f"MEASURE {row}: exists={db.exists()} sha_unchanged={same} "
          f"quarantined={len(_quarantined(directory))} outcome={outcome}")


def _load(directory: Path) -> str:
    try:
        idx = TranscriptIndex.load(directory)
    except Exception as exc:  # noqa: BLE001 - the measure names whatever load did
        return f"raised {type(exc).__name__}: {exc}"
    return f"returned {type(idx).__name__} chunks={len(idx.chunks)}"


def test_r1_non_sqlite_db_is_quarantined_not_deleted(tmp_path):
    sha = _corrupt(tmp_path)
    outcome = _load(tmp_path)
    _measure("R1", tmp_path, sha, outcome)
    assert outcome == "returned TranscriptIndex chunks=0"
    moved = _quarantined(tmp_path)
    assert len(moved) == 1
    assert _sha(moved[0]) == sha
    assert Path(f"{moved[0]}-wal").read_bytes() == b"wal bytes"
    # SQLite rewrites -shm during the failed open, so only its move is checked, not its bytes.
    assert Path(f"{moved[0]}-shm").exists()
    assert not (tmp_path / "recall.db-wal").exists() or (tmp_path / "recall.db-wal").read_bytes() != b"wal bytes"


def test_r1_second_load_on_fresh_store_renames_nothing(tmp_path):
    _corrupt(tmp_path)
    assert _load(tmp_path) == "returned TranscriptIndex chunks=0"
    assert len(_quarantined(tmp_path)) == 1, "setup: the first load did not quarantine"
    before = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("recall.db.corrupt-"))
    assert _load(tmp_path) == "returned TranscriptIndex chunks=0"
    after = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("recall.db.corrupt-"))
    assert after == before


def test_r1_racing_loads_quarantine_once_and_both_open(tmp_path):
    sha = _corrupt(tmp_path, wal=False)
    calls = []

    def another_server_moves_it_first():
        # Another server's load quarantines the file before this load takes the lock; this one then finds it gone.
        if not calls:
            calls.append(1)
            os.rename(tmp_path / "recall.db", tmp_path / "recall.db.corrupt-other")

    with mock.patch.object(storage, "_before_quarantine_lock", another_server_moves_it_first):
        first = _load(tmp_path)
    second = _load(tmp_path)
    assert calls, "setup: the racing rename never ran"
    assert first == "returned TranscriptIndex chunks=0"
    assert second == "returned TranscriptIndex chunks=0"
    moved = _quarantined(tmp_path)
    assert [p.name for p in moved] == ["recall.db.corrupt-other"]
    assert _sha(moved[0]) == sha


def test_r2_non_sqlite_db_with_chunks_jsonl(tmp_path):
    sha = _corrupt(tmp_path, wal=False)
    chunk = TranscriptChunk(id="s1:t0", session_id="s1", timestamp="2026-01-01T00:00:00Z", turn_index=0,
                            user_text="hello", assistant_text="world")
    (tmp_path / "chunks.jsonl").write_text(json.dumps(chunk.to_dict()) + "\n")
    outcome = _load(tmp_path)
    _measure("R2", tmp_path, sha, outcome)
    assert outcome == "returned TranscriptIndex chunks=1"
    moved = _quarantined(tmp_path)
    assert len(moved) == 1 and _sha(moved[0]) == sha


def test_r4_zero_byte_db(tmp_path):
    (tmp_path / "recall.db").write_bytes(b"")
    sha = _sha(tmp_path / "recall.db")
    outcome = _load(tmp_path)
    _measure("R4", tmp_path, sha, outcome)
    assert outcome == "returned TranscriptIndex chunks=0"
    assert (tmp_path / "recall.db").exists()
    assert _quarantined(tmp_path) == []


def test_r5_lock_error_leaves_healthy_db_untouched(tmp_path):
    sha = _healthy(tmp_path)
    locked = sqlite3.OperationalError("database is locked")
    with mock.patch.object(core.ShardedRecallDB, "open", side_effect=locked):
        outcome = _load(tmp_path)
    _measure("R5", tmp_path, sha, outcome)
    assert outcome.startswith("raised OperationalError: database is locked")
    assert str(tmp_path / "recall.db") in outcome, outcome  # the error names the store's path
    assert _sha(tmp_path / "recall.db") == sha
    assert _quarantined(tmp_path) == []


def test_r6_database_error_on_valid_db_propagates_untouched(tmp_path):
    sha = _healthy(tmp_path)
    broken = sqlite3.DatabaseError("file is not a database")
    with mock.patch.object(core.ShardedRecallDB, "open", side_effect=broken):
        outcome = _load(tmp_path)
    _measure("R6", tmp_path, sha, outcome)
    assert outcome == "raised DatabaseError: file is not a database"
    assert (tmp_path / "recall.db").exists()
    assert _sha(tmp_path / "recall.db") == sha
    assert _quarantined(tmp_path) == []


def test_control_healthy_db_loads(tmp_path):
    sha = _healthy(tmp_path)
    outcome = _load(tmp_path)
    _measure("CONTROL", tmp_path, sha, outcome)
    assert outcome == "returned TranscriptIndex chunks=0"
    assert _quarantined(tmp_path) == []


def test_corrupt_error_still_imports_from_cli():
    from synapt.recall import cli
    from synapt.recall.cli import CorruptRecallDBError
    assert CorruptRecallDBError is cli.CorruptRecallDBError
    assert issubclass(CorruptRecallDBError, sqlite3.DatabaseError)


# --- v3 rows: one quarantine helper, a bounded lock wait, the server's error ---

import subprocess
import sys
import threading
import time

import pytest

from synapt.recall import storage


def test_e1_fresh_db_swapped_in_before_lock_is_not_quarantined(tmp_path, monkeypatch):
    _corrupt(tmp_path, wal=False)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    fresh_sha = _healthy(fresh)

    def swap():
        os.replace(fresh / "recall.db", tmp_path / "recall.db")

    monkeypatch.setattr(storage, "_before_quarantine_lock", swap)
    TranscriptIndex.load(tmp_path)
    assert _sha(tmp_path / "recall.db") == fresh_sha
    assert _quarantined(tmp_path) == []


def test_a_existing_quarantine_name_is_kept(tmp_path, monkeypatch):
    sha = _corrupt(tmp_path, wal=False)
    monkeypatch.setattr(storage, "_quarantine_stamp", lambda: "X")
    old = tmp_path / "recall.db.corrupt-X"
    old.write_bytes(b"an earlier quarantine")
    TranscriptIndex.load(tmp_path)
    assert old.read_bytes() == b"an earlier quarantine"
    assert _sha(tmp_path / "recall.db.corrupt-X-1") == sha


def test_b_main_move_failure_rolls_sidecars_back(tmp_path, monkeypatch):
    """Both sidecars move, then the main file's move fails: everything rolls back."""
    sha = _corrupt(tmp_path)
    real_link = os.link

    def link(src, dst, *a, **k):
        if str(src).endswith("recall.db"):  # the main file moves last, after both sidecars
            raise PermissionError(13, "denied", str(src))
        return real_link(src, dst, *a, **k)

    monkeypatch.setattr(storage.os, "link", link)
    with pytest.raises(PermissionError):
        TranscriptIndex.load(tmp_path)
    assert _sha(tmp_path / "recall.db") == sha
    assert (tmp_path / "recall.db-wal").read_bytes() == b"wal bytes"
    assert _quarantined(tmp_path) == []


def test_stale_quarantine_lock_is_named_and_refused(tmp_path):
    sha = _corrupt(tmp_path, wal=False)
    lock = tmp_path / "recall.db.quarantine.lock"
    lock.write_bytes(b"")
    old = time.time() - storage.QUARANTINE_LOCK_STALE_S - 10
    os.utime(lock, (old, old))
    with pytest.raises(storage.QuarantineLockStale, match=re.escape(str(lock))):
        TranscriptIndex.load(tmp_path)
    assert lock.exists()
    assert _sha(tmp_path / "recall.db") == sha


def _hold_lock(db: Path):
    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("BEGIN EXCLUSIVE")
    conn.execute("CREATE TABLE IF NOT EXISTS held(x)")
    return conn


def test_lock_held_past_timeout_raises_locked_and_leaves_file(tmp_path, monkeypatch):
    _healthy(tmp_path)
    monkeypatch.setattr(storage, "LOAD_BUSY_TIMEOUT_S", 0.5)
    holder = _hold_lock(tmp_path / "recall.db")
    try:
        sha = _sha(tmp_path / "recall.db")
        t0 = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="locked|busy"):
            TranscriptIndex.load(tmp_path)
        waited = time.monotonic() - t0
        assert waited >= 0.4, f"the load gave up after {waited:.2f}s, before the busy timeout"
        assert _sha(tmp_path / "recall.db") == sha
        assert _quarantined(tmp_path) == []
    finally:
        holder.close()


def test_server_serves_locked_error_then_loads_after_release(tmp_path, monkeypatch):
    from synapt.recall import server

    _healthy(tmp_path)
    monkeypatch.setattr(storage, "LOAD_BUSY_TIMEOUT_S", 0.3)
    monkeypatch.setattr(server, "project_index_dir", lambda: tmp_path)
    monkeypatch.setattr(server, "_cached_index", None)
    holder = _hold_lock(tmp_path / "recall.db")
    try:
        with pytest.raises(server.RecallStoreUnavailable, match="locked") as err:
            server._get_index(use_embeddings=False)
        assert str(tmp_path) in str(err.value)
    finally:
        holder.close()
    assert server._get_index(use_embeddings=False) is not None


def test_corrupt_sharded_store_raises_and_moves_nothing(tmp_path, monkeypatch):
    from synapt.recall import server

    (tmp_path / "index.db").write_bytes(JUNK)
    (tmp_path / "data_001.db").write_bytes(JUNK)
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(sqlite3.DatabaseError):
        TranscriptIndex.load(tmp_path)
    after = sorted(p.name for p in tmp_path.iterdir())
    # ShardedRecallDB.open may add its index.db.schema.lock (old behaviour, out of scope); nothing is renamed or removed.
    assert set(before) <= set(after) and not [n for n in after if "corrupt" in n]
    monkeypatch.setattr(server, "project_index_dir", lambda: tmp_path)
    monkeypatch.setattr(server, "_cached_index", None)
    with pytest.raises(server.RecallStoreUnavailable, match="damaged.*nothing was moved"):
        server._get_index(use_embeddings=False)
    assert all(_sha(tmp_path / n) == hashlib.sha256(JUNK).hexdigest() for n in before)


def test_schema_reads_alone_still_imports_from_cli():
    from synapt.recall.cli import _schema_reads_alone

    assert _schema_reads_alone is storage._schema_reads_alone


def test_server_serves_stale_lock_error_then_loads_after_removal(tmp_path, monkeypatch):
    from synapt.recall import server

    _corrupt(tmp_path, wal=False)
    lock = tmp_path / "recall.db.quarantine.lock"
    lock.write_bytes(b"")
    old = time.time() - storage.QUARANTINE_LOCK_STALE_S - 10
    os.utime(lock, (old, old))
    monkeypatch.setattr(server, "project_index_dir", lambda: tmp_path)
    monkeypatch.setattr(server, "_cached_index", None)
    with pytest.raises(server.RecallStoreUnavailable) as err:
        server._get_index(use_embeddings=False)
    assert str(tmp_path) in str(err.value) and str(lock) in str(err.value)
    assert lock.exists()
    lock.unlink()
    assert server._get_index(use_embeddings=False) is not None
    assert len(_quarantined(tmp_path)) == 1


def test_e1b_store_created_during_quarantine_keeps_its_wal(tmp_path, monkeypatch):
    _corrupt(tmp_path)
    db = tmp_path / "recall.db"
    st = os.stat(db)
    real_move = storage._move
    created = []

    def move(src, dst):
        real_move(src, dst)
        if not created and not db.exists():
            db.write_bytes(b"fresh db")
            (tmp_path / "recall.db-wal").write_bytes(b"fresh wal")
            created.append(True)

    monkeypatch.setattr(storage, "_move", move)
    assert storage.quarantine_recall_db(db, (st.st_dev, st.st_ino), RuntimeError("probe"))
    assert created, "the hook never found the name free"
    assert (tmp_path / "recall.db-wal").read_bytes() == b"fresh wal"
    assert db.read_bytes() == b"fresh db"
    [q] = _quarantined(tmp_path)
    assert q.read_bytes() == JUNK
    assert Path(f"{q}-wal").read_bytes() == b"wal bytes"
    assert not [p for p in tmp_path.iterdir() if "corrupt" in p.name and p.read_bytes() == b"fresh wal"]


_WRITER = """
import sqlite3, sys, time
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("BEGIN IMMEDIATE")
print("ready", flush=True)
time.sleep(1.2)
conn.execute("COMMIT")
conn.close()
"""


def test_p1b_b_open_waits_out_second_process_writer_and_keeps_store(tmp_path):
    db = tmp_path / "recall.db"
    _healthy(tmp_path)
    conn = sqlite3.connect(str(db))
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.execute("DROP TABLE clusters")
    conn.commit()
    conn.close()
    writer = subprocess.Popen([sys.executable, "-c", _WRITER, str(db)], stdout=subprocess.PIPE, text=True)
    try:
        assert writer.stdout.readline().strip() == "ready"
        t0 = time.monotonic()
        assert TranscriptIndex.load(tmp_path) is not None
        waited = time.monotonic() - t0
        assert writer.wait(timeout=5) == 0
        assert waited >= 1.0, f"the load returned after {waited:.2f}s, before the writer committed"
    finally:
        writer.kill()
        writer.wait()
    assert db.exists()
    assert _quarantined(tmp_path) == []


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc/self/fd")
def test_no_open_store_descriptor_at_first_move(tmp_path, monkeypatch):
    """Windows cannot move an open file: when the quarantine moves its first name, this process holds none open."""
    sha = _corrupt(tmp_path)
    real_move = storage._move
    held = []

    def move(src, dst):
        if not held:
            names = {str(tmp_path / f"recall.db{s}") for s in ("", "-wal", "-shm")}
            fds = [os.readlink(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd")
                   if os.path.exists(f"/proc/self/fd/{fd}")]
            held.append([p for p in fds if p in names])
        return real_move(src, dst)

    monkeypatch.setattr(storage, "_move", move)
    TranscriptIndex.load(tmp_path)
    assert held == [[]]
    assert [_sha(p) for p in _quarantined(tmp_path)] == [sha]


def _open_connections_to(path) -> list:
    """Every live, open sqlite3.Connection whose main database is `path` (no gc.collect: a leak must stay visible)."""
    import gc
    import sqlite3

    found = []
    for obj in gc.get_objects():
        if isinstance(obj, sqlite3.Connection):
            try:
                rows = obj.execute("PRAGMA database_list").fetchall()
            except sqlite3.ProgrammingError:  # closed
                continue
            if any(r[1] == "main" and r[2] and os.path.samefile(r[2], path) for r in rows if os.path.exists(r[2] or "")):
                found.append(obj)
    return found


def test_no_open_store_connection_at_first_move(tmp_path, monkeypatch):
    """On every OS: when the quarantine moves its first name, no connection to recall.db is open. One open would hold
    the file on Windows, and its close, whenever it comes, deletes the -wal."""
    sha = _corrupt(tmp_path)
    real_move = storage._move
    counts = []

    def move(src, dst):
        if not counts:
            counts.append(len(_open_connections_to(tmp_path / "recall.db")))
        return real_move(src, dst)

    monkeypatch.setattr(storage, "_move", move)
    TranscriptIndex.load(tmp_path)
    assert counts == [0]
    assert [_sha(p) for p in _quarantined(tmp_path)] == [sha]
