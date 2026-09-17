"""Concurrent schema-setup witness for the schema-open race.

(The originating defect report is tracked privately and deliberately not
linked from this public tree; the race evidence below is self-contained.)

The multi-agent reality is one server process per desk: on fleet boot or any
fresh gripspace, several processes open the same never-built store within
seconds of each other. Before the schema-race fix, most of those opens died
inside ``_ensure_schema`` with one of three benign-race errors —
``duplicate column name`` (check-then-ALTER migrations), ``table ...
already exists`` (check-then-CREATE FTS5 virtual tables), or a bare
``database is locked`` the busy handler does not mask. Measured pre-fix:
8 barrier-raced cold opens → 7 failed; 8 concurrent first-writers → 5 failed.

These tests spawn true subprocesses (one per would-be agent process) that
park on a filesystem barrier and then open the same fresh store together.
The fix (benign-error-triggered re-run under recall's inter-process filelock)
must turn both red shapes green while still raising on genuinely broken opens.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

_CHILD_TEMPLATE = r'''
import json, sys
from pathlib import Path

sys.path.insert(0, {src!r})
from synapt.recall.storage import RecallDB

mode, db_path, tag, n, barrier = sys.argv[1:6]
while not Path(barrier).exists():
    pass  # spin: release every child on the same instant

errs = []
try:
    if mode == "open":
        db = RecallDB(db_path)
        db.close()
    elif mode == "write":
        db = RecallDB(db_path)
        for i in range(int(n)):
            db.upsert_knowledge_node({{
                "id": f"kn-{{tag}}-{{i:04d}}",
                "content": "concurrent-schema-witness " + tag,
                "category": "witness", "confidence": 0.9,
                "source_sessions": [tag], "tags": ["witness"],
            }})
        db.close()
except Exception as exc:  # noqa: BLE001 - the error IS the result
    errs.append(f"{{type(exc).__name__}}: {{exc}}")

sys.stderr.write(json.dumps(errs))
'''


def _child_source() -> str:
    # the child must import the same src tree pytest is running against
    import synapt.recall.storage as m

    src_root = str(Path(m.__file__).resolve().parents[2])
    return _CHILD_TEMPLATE.format(src=src_root)


def _spawn_raced(tmp_path: Path, mode: str, workers: int, per_worker: int = 0):
    """Spawn `workers` children parked on a barrier, then release them together."""
    child = _child_source()
    db_path = tmp_path / "store.db"
    barrier = tmp_path / "GO"
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", child, mode, str(db_path), f"{mode}{i}",
             str(per_worker), str(barrier)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(tmp_path),
        )
        for i in range(workers)
    ]
    barrier.write_text("go")  # release all parked children at once
    errors = []
    for p in procs:
        p.wait()
        out = p.stderr.read().strip()
        try:
            errs = json.loads(out) if out else []
        except json.JSONDecodeError:
            errs = [out]
        errors.extend(errs)
    return db_path, errors


def _integrity_and_count(db_path: Path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    try:
        rows = conn.execute("PRAGMA integrity_check(20)").fetchall()
        count = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
        return rows, count
    finally:
        conn.close()


@pytest.mark.parametrize("workers", [8])
def test_concurrent_cold_opens_all_succeed(tmp_path: Path, workers: int):
    """8 processes opening a brand-new store concurrently: zero failures."""
    db_path, errors = _spawn_raced(tmp_path, "open", workers)
    assert not errors, (
        f"{len(errors)}/{workers} concurrent cold opens failed against a fresh "
        f"store (concurrent schema-open race): {errors[:4]}"
    )
    rows, _ = _integrity_and_count(db_path)
    assert all(r[0] == "ok" for r in rows), rows


@pytest.mark.parametrize("workers", [8])
def test_concurrent_first_writers_all_succeed(tmp_path: Path, workers: int):
    """Same race, but every process also writes from the never-opened store."""
    db_path, errors = _spawn_raced(tmp_path, "write", workers, per_worker=4)
    assert not errors, (
        f"{len(errors)}/{workers} concurrent first-writers failed (concurrent "
        f"schema-open race): {errors[:4]}"
    )
    rows, count = _integrity_and_count(db_path)
    assert all(r[0] == "ok" for r in rows), rows
    assert count == workers * 4, f"lost writes: {count}/{workers * 4}"


def test_real_corruption_is_not_masked_by_the_retry(tmp_path: Path):
    """The race-retry must re-raise on genuinely broken opens."""
    from synapt.recall.storage import RecallDB

    bad = tmp_path / "garbage.db"
    bad.write_bytes(b"this is not a sqlite database at all" * 10)
    with pytest.raises(sqlite3.DatabaseError):
        RecallDB(bad)