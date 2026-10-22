"""Every access.db connection is closed, so Windows can delete the dir."""

import gc
import sqlite3

import pytest

from synapt.recall import storage
from synapt.recall.core import TranscriptIndex
from synapt.recall.storage import RecallDB
from tests.recall.test_core import make_test_chunks


def _record_conns(monkeypatch):
    opened = []
    real = storage._access_conn

    def wrapped(*a, **kw):
        conn = real(*a, **kw)
        opened.append(conn)
        return conn

    monkeypatch.setattr(storage, "_access_conn", wrapped)
    return opened


def _assert_all_closed(opened):
    assert opened
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("select 1")


def test_transcript_index_leaves_no_access_conn_open(tmp_path, monkeypatch):
    chunks = make_test_chunks()
    db = RecallDB(tmp_path / "recall.db")
    db.save_chunks(chunks)
    db.record_access([{"item_type": "chunk", "item_id": chunks[0].id,
                       "query": "quality curve"}])
    opened = _record_conns(monkeypatch)
    TranscriptIndex(chunks, db=db)
    db.close()
    _assert_all_closed(opened)


def test_dropped_recalldb_closes_access_conn(tmp_path, monkeypatch):
    opened = _record_conns(monkeypatch)
    db = RecallDB(tmp_path / "recall.db")
    db._access
    del db
    gc.collect()
    _assert_all_closed(opened)
