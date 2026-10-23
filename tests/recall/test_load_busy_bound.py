"""TranscriptIndex.load waits out a lock for LOAD_BUSY_TIMEOUT_S in all, not per connection.

A second process holds BEGIN IMMEDIATE on a store file. Each row times load against that lock.
"""
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from synapt.recall.core import TranscriptIndex
from synapt.recall.storage import RecallDB

_HOLDER = """
import sqlite3, sys, time
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("BEGIN IMMEDIATE")
print("READY", flush=True)
time.sleep(float(sys.argv[2]))
conn.execute("COMMIT")
conn.close()
"""


def _hold(db: Path, seconds: float) -> subprocess.Popen:
    """Start a child holding BEGIN IMMEDIATE on db for ``seconds``; return once it reports READY."""
    child = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(db), str(seconds)],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
    )
    assert child.stdout.readline().strip() == "READY"
    return child


def _stop(child: subprocess.Popen) -> None:
    child.kill()
    child.wait(timeout=10)


def _assert_locked(db: Path) -> None:
    """Precondition: the child's lock is real, so a writer with no busy wait is refused."""
    conn = sqlite3.connect(str(db), timeout=0, isolation_level=None)
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            conn.execute("BEGIN IMMEDIATE")
    finally:
        conn.close()


def _store(db: Path) -> Path:
    """A healthy store missing one table, so the next open has a schema write to make (and to wait for)."""
    RecallDB(db).close()
    conn = sqlite3.connect(str(db))
    conn.execute("DROP TABLE clusters")
    conn.commit()
    conn.close()
    return db


def _timed_load(directory: Path):
    start = time.monotonic()
    try:
        TranscriptIndex.load(directory)
        outcome = None
    except sqlite3.OperationalError as exc:
        outcome = exc
    elapsed = time.monotonic() - start
    print(f"MEASURE elapsed={elapsed:.3f}s outcome={outcome!r}")
    return elapsed, outcome


def _bare_blocked_write(directory: Path) -> float:
    """This platform's time for one blocked write at 2 s, no recall code: the max of 2 samples.

    The platform's SQLite busy handler can overrun the requested wait (macOS runners took 2.7 to 4.3 s at
    2.0 s); the bound a row asserts is ours on top of it. The 2 s is a literal, not LOAD_BUSY_TIMEOUT_S:
    timing the bare wait at the constant under test would let a change to its value move the bound with it."""
    db = directory / "bare.db"
    sqlite3.connect(str(db)).close()
    child = _hold(db, 60)
    samples = []
    try:
        for _ in range(2):
            conn = sqlite3.connect(str(db), timeout=2.0, isolation_level=None)
            start = time.monotonic()
            try:
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    conn.execute("BEGIN IMMEDIATE")
            finally:
                samples.append(time.monotonic() - start)
                conn.close()
    finally:
        _stop(child)
    return max(samples)


def _assert_bound(elapsed: float, bare: float, lower: float) -> None:
    print(f"MEASURE bare={bare:.3f}s load={elapsed:.3f}s bound={bare + 1.0:.3f}s")
    assert lower <= elapsed <= bare + 1.0


def _sharded(directory: Path, shards: int) -> list[Path]:
    RecallDB(directory / "index.db").close()
    paths = [directory / f"data_{i:03d}.db" for i in range(shards)]
    for p in paths:
        _store(p)
    return paths


def test_r1_held_lock_raises_inside_the_bound_naming_the_path(tmp_path):
    """Under BEGIN IMMEDIATE in WAL readers are not blocked, so wait_for_store returns at once and the
    whole wait lands on the open's schema write: that is why the 1.5 s lower bound is the open's wait."""
    bare = _bare_blocked_write(tmp_path)
    _store(tmp_path / "recall.db")
    child = _hold(tmp_path / "recall.db", 60)
    try:
        _assert_locked(tmp_path / "recall.db")
        elapsed, exc = _timed_load(tmp_path)
    finally:
        _stop(child)
    assert isinstance(exc, sqlite3.OperationalError)
    assert "database is locked" in str(exc)
    assert str(tmp_path / "recall.db") in str(exc)
    # Calibrated, not a fixed 2.5 s: the platform's busy handler can overrun the requested wait.
    _assert_bound(elapsed, bare, 1.5)


def test_r1_pre_child_lock_is_real(tmp_path):
    _store(tmp_path / "recall.db")
    child = _hold(tmp_path / "recall.db", 60)
    try:
        _assert_locked(tmp_path / "recall.db")
    finally:
        _stop(child)


def test_r2_no_lock_loads(tmp_path):
    _store(tmp_path / "recall.db")
    elapsed, exc = _timed_load(tmp_path)
    assert exc is None
    assert elapsed < 1.5


def test_r3_lock_released_inside_the_bound_loads(tmp_path):
    """Load waits out a 0.5 s hold and returns. No upper bound: with the lock gone at 0.5 s the old 30 s wait
    returns as soon, so a bound here catches no mutant and times only the runner (R1 and R4 carry the bound)."""
    _store(tmp_path / "recall.db")
    child = _hold(tmp_path / "recall.db", 0.5)
    try:
        elapsed, exc = _timed_load(tmp_path)
    finally:
        _stop(child)
    assert exc is None
    assert elapsed >= 0.4, f"load returned after {elapsed:.2f}s, before the 0.5 s hold was released"


def test_r4_sharded_shards_share_one_bound(tmp_path):
    """data_000 is held for 1.5 s and the last shard throughout. One shared deadline raises near 2.0 s;
    a fresh bound per shard would wait 1.5 s on data_000 and then 2.0 s more on the last, near 3.5 s."""
    bare = _bare_blocked_write(tmp_path)
    paths = _sharded(tmp_path, 2)
    last = _hold(paths[-1], 60)
    first = None
    try:
        _assert_locked(paths[-1])
        first = _hold(paths[0], 1.5)
        _assert_locked(paths[0])
        elapsed, exc = _timed_load(tmp_path)
    finally:
        if first is not None:
            _stop(first)
        _stop(last)
    assert isinstance(exc, sqlite3.OperationalError)
    assert "database is locked" in str(exc)
    assert str(tmp_path) in str(exc)
    # Calibrated, not a fixed 2.5 s: the platform's busy handler can overrun the requested wait.
    _assert_bound(elapsed, bare, 1.4)


def test_r6_no_deadline_keeps_the_build_wait(tmp_path):
    """A RecallDB opened with no deadline (every build and write caller) keeps today's 30 s busy wait; a deadline caps it."""
    db = RecallDB(tmp_path / "recall.db")
    try:
        assert db._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    finally:
        db.close()
    db = RecallDB(tmp_path / "recall.db", deadline=time.monotonic() + 10.0)
    try:  # the deadline bounds the open only; R1 and R4 time that wait
        assert db._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    finally:
        db.close()


def test_r7_a_loaded_index_serves_writes_with_the_build_wait(tmp_path):
    """After a bounded load, the connection the index keeps for later writes waits 30 s, not what was left of 2 s."""
    db = RecallDB(tmp_path / "recall.db")
    db.close()
    idx = TranscriptIndex.load(tmp_path)
    store = idx._db
    served = getattr(store, "_index", store)
    try:
        assert served._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    finally:
        store.close()


def test_r8_a_write_through_a_loaded_index_waits_out_a_writer(tmp_path):
    """Behaviour, not only the pragma: after a load, a write through the kept connection waits out a 3 s writer."""
    db = tmp_path / "recall.db"
    RecallDB(db).close()
    idx = TranscriptIndex.load(tmp_path)
    store = idx._db
    served = getattr(store, "_index", store)
    child = _hold(db, 3.0)
    try:
        _assert_locked(db)
        start = time.monotonic()
        served._conn.execute("CREATE TABLE IF NOT EXISTS r8_probe (x INTEGER)")
        served._conn.commit()
        elapsed = time.monotonic() - start
        print(f"MEASURE elapsed={elapsed:.3f}s")
        assert elapsed >= 2.8, f"the write returned after {elapsed:.2f}s, before the writer committed"
    finally:
        _stop(child)
        store.close()
