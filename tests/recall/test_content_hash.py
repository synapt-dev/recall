"""Tests for RecallDB.content_hash() — DB-level fast hash (#435)."""

import pytest
from synapt.recall.storage import RecallDB


@pytest.fixture
def db(tmp_path):
    d = RecallDB(tmp_path / "test.db")
    yield d
    d.close()


def _insert_chunk(db, chunk_id, user_text="", assistant_text="", tool_content=""):
    db._conn.execute(
        "INSERT INTO chunks (id, session_id, timestamp, turn_index, "
        "user_text, assistant_text, tool_content) "
        "VALUES (?, 'sess', '2026-01-01', 0, ?, ?, ?)",
        (chunk_id, user_text, assistant_text, tool_content),
    )
    db._conn.commit()


def test_content_hash_deterministic(db):
    _insert_chunk(db, "c1", user_text="hello")
    _insert_chunk(db, "c2", user_text="world")
    assert db.content_hash() == db.content_hash()


def test_content_hash_changes_with_content(db):
    _insert_chunk(db, "c1", user_text="hello")
    h1 = db.content_hash()

    db._conn.execute("UPDATE chunks SET user_text = 'changed' WHERE id = 'c1'")
    db._conn.commit()
    h2 = db.content_hash()

    assert h1 != h2


def test_content_hash_empty_db(db):
    h = db.content_hash()
    assert isinstance(h, str)
    assert len(h) == 16


def test_content_hash_with_empty_fields(db):
    """Empty string fields produce consistent hash (NOT NULL DEFAULT '')."""
    _insert_chunk(db, "c1")
    h = db.content_hash()
    assert isinstance(h, str)
    assert len(h) == 16
