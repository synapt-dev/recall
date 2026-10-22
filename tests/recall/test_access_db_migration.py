"""An open moves access rows an older build left in index.db into access.db."""

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from synapt.recall import storage
from synapt.recall.sharding import split_monolithic_db
from synapt.recall.storage import RecallDB

TABLES = ("access_log", "access_stats", "access_log_archive")


def _seed(index, n, *, item="c1", tier="raw", t0="2026-01-01T00:00:0"):
    """Write access rows through index.db's own tables, as a pre-change build does."""
    conn = sqlite3.connect(str(index))
    for i in range(n):
        conn.execute(
            "INSERT INTO access_log (item_type, item_id, query, score, session_id, "
            "context, created_at) VALUES ('chunk', ?, ?, 1.0, ?, 'search', ?)",
            (item, f"q{i}", f"s{i}", f"{t0}{i}"),
        )
    conn.execute(
        "INSERT INTO access_stats (item_type, item_id, access_count, explicit_count, "
        "weighted_count, last_accessed, first_accessed, promotion_tier, "
        "distinct_sessions, distinct_queries, decay_score) "
        "VALUES ('chunk', ?, ?, ?, ?, ?, ?, ?, 99, 99, 0.5)",
        (item, n, n, float(n), f"{t0}{n - 1}", f"{t0}0", tier),
    )
    conn.execute(
        "INSERT INTO access_log_archive (item_type, item_id, date, access_count) "
        "VALUES ('chunk', ?, ?, ?)", (item, t0[:10], n),
    )
    conn.commit()
    conn.close()


def _counts(path, where=""):
    conn = sqlite3.connect(str(path))
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t} {where}").fetchone()[0]
                for t in TABLES}
    finally:
        conn.close()


def _stats(store, item):
    conn = sqlite3.connect(str(store / "access.db"))
    conn.row_factory = sqlite3.Row
    try:
        return dict(conn.execute(
            "SELECT * FROM access_stats WHERE item_id = ?", (item,)).fetchone())
    finally:
        conn.close()


@pytest.fixture
def store(tmp_path):
    RecallDB(tmp_path / "index.db").close()
    return tmp_path


def test_open_moves_existing_rows(store):
    _seed(store / "index.db", 3)
    RecallDB(store / "index.db").close()
    assert _counts(store / "access.db") == {t: 1 if t != "access_log" else 3 for t in TABLES}
    assert _counts(store / "index.db") == dict.fromkeys(TABLES, 0)


def test_clean_move_leaves_no_stats_records(store):
    _seed(store / "index.db", 3)
    RecallDB(store / "index.db").close()
    conn = sqlite3.connect(str(store / "access.db"))
    assert conn.execute("SELECT COUNT(*) FROM access_move_stats").fetchone()[0] == 0
    conn.close()


def test_crash_after_commit_before_delete_resumes(store, monkeypatch):
    _seed(store / "index.db", 3)

    def boom(self, marks):
        raise RuntimeError("crash before delete")

    with monkeypatch.context() as m:
        m.setattr(RecallDB, "_delete_moved_rows", boom)
        with pytest.raises(RuntimeError):
            RecallDB(store / "index.db")
    RecallDB(store / "index.db").close()
    assert _counts(store / "access.db") == {"access_log": 3, "access_stats": 1,
                                            "access_log_archive": 1}
    assert _stats(store, "c1")["access_count"] == 3
    assert _counts(store / "index.db") == dict.fromkeys(TABLES, 0)


def _resume_after_bump(store, monkeypatch, hook):
    _seed(store / "index.db", 3)
    with monkeypatch.context() as m:
        m.setattr(RecallDB, "_delete_moved_rows",
                  lambda self, marks: (_ for _ in ()).throw(RuntimeError("crash")))
        with pytest.raises(RuntimeError):
            RecallDB(store / "index.db")
    conn = sqlite3.connect(str(store / "index.db"))
    conn.execute("UPDATE access_stats SET access_count = access_count + 2 "
                 "WHERE item_id = 'c1'")
    conn.commit()
    conn.close()
    with monkeypatch.context() as m:
        m.setattr(RecallDB, hook,
                  lambda self, *a: (_ for _ in ()).throw(RuntimeError("crash")))
        with pytest.raises(RuntimeError):
            RecallDB(store / "index.db")
    RecallDB(store / "index.db").close()
    assert _stats(store, "c1")["access_count"] == 5


def test_resume_crash_after_merge_commit_does_not_merge_twice(store, monkeypatch):
    _resume_after_bump(store, monkeypatch, "_resume_delete")


def test_resume_crash_before_merge_commit_merges_once(store, monkeypatch):
    _resume_after_bump(store, monkeypatch, "_mark_merged")


def test_older_build_bump_between_merge_and_delete_is_locked_out(store, monkeypatch):
    _seed(store / "index.db", 3)
    with monkeypatch.context() as m:
        m.setattr(RecallDB, "_delete_moved_rows",
                  lambda self, marks: (_ for _ in ()).throw(RuntimeError("crash")))
        with pytest.raises(RuntimeError):
            RecallDB(store / "index.db")
    conn = sqlite3.connect(str(store / "index.db"))
    conn.execute("UPDATE access_stats SET access_count = access_count + 2 "
                 "WHERE item_id = 'c1'")
    conn.commit()
    conn.close()
    real = RecallDB._resume_delete

    def bump_then_delete(self, *a):
        other = sqlite3.connect(str(store / "index.db"), timeout=0)
        try:
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                other.execute("UPDATE access_stats SET access_count = access_count + 4 "
                              "WHERE item_id = 'c1'")
                other.commit()
        finally:
            other.close()
        return real(self, *a)

    with monkeypatch.context() as m:
        m.setattr(RecallDB, "_resume_delete", bump_then_delete)
        RecallDB(store / "index.db").close()
    assert _stats(store, "c1")["access_count"] == 5


def test_older_build_stats_update_in_crash_window_is_kept(store, monkeypatch):
    _seed(store / "index.db", 3)
    with monkeypatch.context() as m:
        m.setattr(RecallDB, "_delete_moved_rows",
                  lambda self, marks: (_ for _ in ()).throw(RuntimeError("crash")))
        with pytest.raises(RuntimeError):
            RecallDB(store / "index.db")
    # An older build bumps the moved row's counts in index.db before the resume.
    conn = sqlite3.connect(str(store / "index.db"))
    conn.execute("UPDATE access_stats SET access_count = access_count + 2, "
                 "explicit_count = explicit_count + 2, "
                 "weighted_count = weighted_count + 2 WHERE item_id = 'c1'")
    conn.commit()
    conn.close()
    RecallDB(store / "index.db").close()
    assert _stats(store, "c1")["access_count"] == 5


def test_older_build_promotion_in_crash_window_is_kept(store, monkeypatch):
    _seed(store / "index.db", 3)
    with monkeypatch.context() as m:
        m.setattr(RecallDB, "_delete_moved_rows",
                  lambda self, marks: (_ for _ in ()).throw(RuntimeError("crash")))
        with pytest.raises(RuntimeError):
            RecallDB(store / "index.db")
    # An older build promotes the moved row in index.db before the resume.
    conn = sqlite3.connect(str(store / "index.db"))
    conn.execute("UPDATE access_stats SET promotion_tier = 'promoted' WHERE item_id = 'c1'")
    conn.commit()
    conn.close()
    RecallDB(store / "index.db").close()
    assert _stats(store, "c1")["promotion_tier"] == "promoted"


def test_older_build_new_stats_row_reusing_rowid_in_crash_window_is_kept(store, monkeypatch):
    _seed(store / "index.db", 3)
    with monkeypatch.context() as m:
        m.setattr(RecallDB, "_delete_moved_rows",
                  lambda self, marks: (_ for _ in ()).throw(RuntimeError("crash")))
        with pytest.raises(RuntimeError):
            RecallDB(store / "index.db")
    # An older build empties access_stats, then writes a new row for the same
    # key; it takes the moved row's rowid again.
    conn = sqlite3.connect(str(store / "index.db"))
    conn.execute("DELETE FROM access_stats")
    conn.execute(
        "INSERT INTO access_stats (item_type, item_id, access_count, explicit_count, "
        "weighted_count, last_accessed, first_accessed) "
        "VALUES ('chunk', 'c1', 4, 4, 4.0, '2026-01-02T00:00:00', '2026-01-02T00:00:00')")
    assert conn.execute("SELECT rowid FROM access_stats").fetchone()[0] == 1
    conn.commit()
    conn.close()
    RecallDB(store / "index.db").close()
    RecallDB(store / "index.db").close()
    assert _stats(store, "c1")["access_count"] == 7


@pytest.mark.parametrize("old_tier,new_tier,want", [
    ("promoted", "clustered", "promoted"),
    ("clustered", "knowledge", "knowledge"),
])
def test_stats_row_merges_per_column(store, old_tier, new_tier, want):
    _seed(store / "index.db", 2, tier=old_tier, t0="2026-01-01T00:00:0")
    RecallDB(store / "index.db").close()
    _seed(store / "index.db", 3, tier=new_tier, t0="2026-01-02T00:00:0")
    RecallDB(store / "index.db").close()
    s = _stats(store, "c1")
    assert (s["access_count"], s["explicit_count"], s["weighted_count"]) == (5, 5, 5.0)
    assert s["first_accessed"] == "2026-01-01T00:00:00"
    assert s["last_accessed"] == "2026-01-02T00:00:02"
    # Recomputed from access.db's log: sessions s0..s2, queries q0..q2.
    assert (s["distinct_sessions"], s["distinct_queries"]) == (3, 3)
    assert 0.0 < s["decay_score"] < 0.5
    assert s["promotion_tier"] == want


def test_older_build_rows_after_migration_are_moved(store, monkeypatch):
    _seed(store / "index.db", 2)
    with monkeypatch.context() as m:
        m.setattr(RecallDB, "_delete_moved_rows",
                  lambda self, marks: (_ for _ in ()).throw(RuntimeError("crash")))
        with pytest.raises(RuntimeError):
            RecallDB(store / "index.db")
    # An older build keeps writing into index.db after the interrupted move.
    _seed(store / "index.db", 2, item="c2", t0="2026-01-03T00:00:0")
    RecallDB(store / "index.db").close()
    _seed(store / "index.db", 1, item="c3", t0="2026-01-04T00:00:0")
    RecallDB(store / "index.db").close()
    assert _counts(store / "access.db") == {"access_log": 5, "access_stats": 3,
                                            "access_log_archive": 3}
    assert [_stats(store, i)["access_count"] for i in ("c1", "c2", "c3")] == [2, 2, 1]
    assert _counts(store / "index.db") == dict.fromkeys(TABLES, 0)


def test_rollback_plain_open_sees_empty_access_tables(store):
    _seed(store / "index.db", 2)
    RecallDB(store / "index.db").close()
    assert _counts(store / "index.db") == dict.fromkeys(TABLES, 0)


SRC = str(Path(storage.__file__).resolve().parents[2])

OPENER = """
import pathlib, sys, time
from synapt.recall.storage import RecallDB
store, go = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
while not go.exists():
    time.sleep(0.001)
RecallDB(store / "index.db").close()
"""

SLOW_MOVER = """
import pathlib, sys, time
from synapt.recall.storage import RecallDB
store, copying = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
real = RecallDB._copy_access_rows
def slow(acc, *rows):
    real(acc, *rows)
    copying.touch()
    time.sleep(1.5)
RecallDB._copy_access_rows = staticmethod(slow)
RecallDB(store / "index.db").close()
"""

LATE_WRITER = """
import pathlib, sys, time
from synapt.recall.storage import RecallDB
store, copying = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
db = RecallDB(store / "index.db")
print("ready", flush=True)
while not copying.exists():
    time.sleep(0.001)
t = time.monotonic()
db.record_access([{"item_type": "chunk", "item_id": "c1", "query": "late",
                   "session_id": "late"}])
print(time.monotonic() - t, flush=True)
db.close()
"""


def _spawn(script, *args, **kw):
    return subprocess.Popen([sys.executable, "-c", script, *map(str, args)],
                            env={**os.environ, "PYTHONPATH": SRC}, **kw)


def _unmigrated(store, n):
    """index.db with an older build's access rows and no access.db yet."""
    for name in ("access.db", "access.db-wal", "access.db-shm"):
        (store / name).unlink(missing_ok=True)
    _seed(store / "index.db", n)


def _no_duplicates(store):
    conn = sqlite3.connect(str(store / "access.db"))
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM access_log GROUP BY item_type, "
            "item_id, query, session_id, created_at HAVING COUNT(*) > 1)").fetchone()[0] == 0
    finally:
        conn.close()


def test_two_processes_open_an_unmigrated_store_at_once(store):
    _unmigrated(store, 50)
    go = store / "go"
    procs = [_spawn(OPENER, store, go) for _ in range(2)]
    time.sleep(1.0)
    go.touch()
    assert [p.wait(timeout=60) for p in procs] == [0, 0]
    assert _counts(store / "access.db") == {"access_log": 50, "access_stats": 1,
                                            "access_log_archive": 1}
    assert _stats(store, "c1")["access_count"] == 50
    assert _no_duplicates(store)
    assert _counts(store / "index.db") == dict.fromkeys(TABLES, 0)


SLOW_CREATOR = """
import os, pathlib, sys, time
from synapt.recall import storage
from synapt.recall.storage import RecallDB
store, go, delay = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), float(sys.argv[3])
real = storage._create_access_db
def slow(path):
    time.sleep(delay)
    with open(store / "creates", "a") as f:
        f.write(f"{os.getpid()}\\n")
    real(path)
storage._create_access_db = slow
while not go.exists():
    time.sleep(0.001)
RecallDB(store / "index.db").close()
"""


def test_two_first_opens_create_access_db_once(store):
    # Each opener sleeps between seeing no access.db and creating it; without
    # the lock the later one renames an empty access.db over the moved rows.
    _unmigrated(store, 50)
    go = store / "go"
    procs = [_spawn(SLOW_CREATOR, store, go, d) for d in (1.0, 2.0)]
    time.sleep(1.0)
    go.touch()
    assert [p.wait(timeout=60) for p in procs] == [0, 0]
    assert len((store / "creates").read_text().split()) == 1
    assert _counts(store / "access.db") == {"access_log": 50, "access_stats": 1,
                                            "access_log_archive": 1}
    assert _stats(store, "c1")["access_count"] == 50
    assert _no_duplicates(store)
    assert _counts(store / "index.db") == dict.fromkeys(TABLES, 0)


def test_access_written_during_the_move_lands_in_access_db(store):
    # The writer opens before the old rows appear, so its open moves nothing;
    # its record_access then runs while another open holds the copy.
    copying = store / "copying"
    writer = _spawn(LATE_WRITER, store, copying, stdout=subprocess.PIPE, text=True)
    assert writer.stdout.readline().strip() == "ready"
    _seed(store / "index.db", 3)
    mover = _spawn(SLOW_MOVER, store, copying)
    waited = float(writer.stdout.readline())
    assert [mover.wait(timeout=60), writer.wait(timeout=60)] == [0, 0]
    # The copy holds access.db's write lock; the writer waits there, not on index.db.
    assert waited > 1.0
    assert _counts(store / "access.db") == {"access_log": 4, "access_stats": 1,
                                            "access_log_archive": 1}
    conn = sqlite3.connect(str(store / "access.db"))
    assert conn.execute("SELECT COUNT(*) FROM access_log WHERE query = 'late'").fetchone()[0] == 1
    conn.close()
    assert _stats(store, "c1")["access_count"] == 4
    assert _counts(store / "index.db") == dict.fromkeys(TABLES, 0)


def test_crash_inside_the_copy_leaves_index_rows_in_place(store, monkeypatch):
    _seed(store / "index.db", 3)

    def boom(acc, log, stats, archive):
        raise RuntimeError("crash inside the copy")

    with monkeypatch.context() as m:
        m.setattr(RecallDB, "_copy_access_rows", staticmethod(boom))
        with pytest.raises(RuntimeError):
            RecallDB(store / "index.db")
    assert _counts(store / "index.db") == {"access_log": 3, "access_stats": 1,
                                           "access_log_archive": 1}
    RecallDB(store / "index.db").close()
    assert _counts(store / "access.db") == {"access_log": 3, "access_stats": 1,
                                            "access_log_archive": 1}
    assert _stats(store, "c1")["access_count"] == 3
    assert _counts(store / "index.db") == dict.fromkeys(TABLES, 0)


def test_sharded_copy_moves_access_rows_exactly_once(tmp_path):
    db = RecallDB(tmp_path / "recall.db")
    db._conn.execute(
        "INSERT INTO chunks (id, session_id, timestamp, turn_index, user_text, "
        "assistant_text, tools_used, files_touched, tool_content, transcript_path, "
        "byte_offset, byte_length) VALUES ('k0', 's1', '2025-01-01T10:00:00Z', 0, "
        "'u', 'a', '[]', '[]', '', '', 0, 0)")
    db._conn.commit()
    db.record_access([{"item_type": "chunk", "item_id": "c1", "query": "new",
                       "session_id": "new"}])
    db.close()
    _seed(tmp_path / "recall.db", 3)
    split_monolithic_db(tmp_path)
    RecallDB(tmp_path / "index.db").close()
    RecallDB(tmp_path / "recall.db").close()
    assert _counts(tmp_path / "access.db") == {"access_log": 4, "access_stats": 1,
                                               "access_log_archive": 1}
    assert _stats(tmp_path, "c1")["access_count"] == 4
    assert _no_duplicates(tmp_path)
    for name in ("index.db", "recall.db"):
        assert _counts(tmp_path / name) == dict.fromkeys(TABLES, 0)
