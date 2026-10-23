"""Access tables live in access.db, so access writes never move the index stamp."""

import sqlite3
import subprocess
import sys
import textwrap

from synapt.recall.server import _index_cache_stamp
from synapt.recall.storage import RecallDB

from tests.recall.test_server_index_stamp import _call, _server, _stop, _store

ACCESS_WRITER = textwrap.dedent(
    """
    import os, sys
    from synapt.recall.storage import RecallDB
    db = RecallDB(sys.argv[1] + "/index.db")
    db.record_access([{"item_type": "chunk", "item_id": "c1", "query": "q"}])
    os._exit(0)  # no close, so no checkpoint
    """
)

TABLES = ("access_log", "access_stats", "access_log_archive")


def _counts(path):
    conn = sqlite3.connect(str(path))
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
    finally:
        conn.close()


def _access_write(store):
    subprocess.run([sys.executable, "-c", ACCESS_WRITER, str(store)], check=True)


def test_access_db_has_all_three_tables(tmp_path):
    db = RecallDB(tmp_path / "index.db")
    db._access
    db.close()
    assert _counts(tmp_path / "access.db") == dict.fromkeys(TABLES, 0)


def test_fresh_store_writes_access_rows_to_access_db(tmp_path):
    db = RecallDB(tmp_path / "index.db")
    db.record_access([{"item_type": "chunk", "item_id": "c1", "query": "q"}])
    db.close()
    assert _counts(tmp_path / "access.db")["access_log"] == 1
    assert _counts(tmp_path / "access.db")["access_stats"] == 1
    assert _counts(tmp_path / "index.db") == {t: 0 for t in TABLES}


def test_access_write_by_another_process_does_not_reload_server(tmp_path):
    store = _store(tmp_path)
    proc, _ = _server(tmp_path, store)
    try:
        assert _call(proc)[0] == "reload"
        _access_write(store)
        assert _call(proc)[0] == "cached"
    finally:
        _stop(proc)


def test_access_readers_read_access_db(tmp_path):
    store = _store(tmp_path)
    _access_write(store)
    db = RecallDB(store / "index.db")
    try:
        assert db.get_access_stats("chunk", "c1")["access_count"] == 1
        assert db.access_summary()["total_events"] == 1
    finally:
        db.close()
    assert _counts(store / "index.db") == {t: 0 for t in TABLES}


def test_stamp_never_lists_access_db(tmp_path):
    store = _store(tmp_path)
    _access_write(store)
    names = {name for name, _, _ in _index_cache_stamp(store)}
    assert (store / "access.db").exists()
    assert not any(n.startswith("access.db") for n in names)
