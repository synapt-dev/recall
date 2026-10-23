"""The flat legacy shard mirror stays equal to CURRENT: after a grow, a shrink, and an in-place embedding save,
and never holds older data after a failed sync."""

from __future__ import annotations

import logging
import shutil
import sqlite3

import pytest

import synapt.recall.sharding as sharding_mod
from synapt.recall.core import TranscriptChunk
from synapt.recall.generations import current_generation_dir
from synapt.recall.sharded_db import ShardedRecallDB
from synapt.recall.sharding import list_shards
from synapt.recall.storage import RecallDB


def _chunks(n: int, tag: str = "s") -> list[TranscriptChunk]:
    return [
        TranscriptChunk(id=f"{tag}:t{i}", session_id=tag, timestamp=f"2026-10-01T00:00:{i:02d}",
                        turn_index=i, user_text=f"u{i}", assistant_text=f"a{i}")
        for i in range(n)
    ]


def _counts(path, ro: bool = False) -> tuple:
    uri = f"file:{path}?mode=ro" if ro else str(path)
    conn = sqlite3.connect(uri, uri=ro)
    try:
        return conn.execute("SELECT count(*), count(embedding) FROM chunks").fetchone()
    finally:
        conn.close()


def _ids(path) -> set[str]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {r[0] for r in conn.execute("SELECT id FROM chunks")}
    finally:
        conn.close()


def _assert_flat_equals_current(index_dir) -> None:
    gen = list_shards(current_generation_dir(index_dir))
    flat = list_shards(index_dir)
    assert [p.name for p in flat] == [p.name for p in gen]
    for g in gen:
        f = index_dir / g.name
        assert _counts(f, ro=True) == _counts(g), g.name


@pytest.fixture
def store(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    RecallDB(index_dir / "index.db").close()
    monkeypatch.setattr(sharding_mod, "SHARD_CHUNK_THRESHOLD", 3)
    db = ShardedRecallDB.open(index_dir)
    yield index_dir, db
    db.close()


def test_flat_mirror_equals_current_after_grow_and_shrink(store):
    index_dir, db = store
    db.save_chunks(_chunks(5))
    _assert_flat_equals_current(index_dir)
    db.save_chunks(_chunks(7))
    assert len(list_shards(index_dir)) == 3
    _assert_flat_equals_current(index_dir)
    db.save_chunks(_chunks(4))
    assert len(list_shards(index_dir)) == 2
    _assert_flat_equals_current(index_dir)


def test_flat_mirror_equals_current_after_save_embeddings(store):
    index_dir, _ = store
    seed = ShardedRecallDB.open(index_dir)
    seed.save_chunks(_chunks(5))
    seed.close()
    db = ShardedRecallDB.open(index_dir)
    try:
        db.save_embeddings({ShardedRecallDB._encode_chunk_rowid(2, 1): [0.1] * 384})
    finally:
        db.close()
    gen = list_shards(current_generation_dir(index_dir))
    assert _counts(gen[1])[1] == 1
    _assert_flat_equals_current(index_dir)


def test_failed_flat_copy_removes_the_flat_set_and_names_the_failure(store, monkeypatch, caplog):
    index_dir, db = store
    db.save_chunks(_chunks(5, "old"))
    _assert_flat_equals_current(index_dir)

    def failing_copy(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shutil, "copyfile", failing_copy)
    with caplog.at_level(logging.WARNING, logger="synapt.recall.generations"):
        db.save_chunks(_chunks(4, "new"))

    # The publish a current reader depends on still succeeded.
    gen_ids = set().union(*(_ids(p) for p in list_shards(current_generation_dir(index_dir))))
    assert gen_ids == {f"new:t{i}" for i in range(4)}
    # The flat set holds no old data: it is gone.
    assert list_shards(index_dir) == []
    lines = [r.getMessage() for r in caplog.records if "flat-layout shard mirror" in r.getMessage()]
    assert len(lines) == 1 and "flat shards removed" in lines[0], lines


def test_save_embeddings_under_an_open_reader_never_leaves_a_stale_flat_set(store, caplog):
    index_dir, _ = store
    seed = ShardedRecallDB.open(index_dir)
    seed.save_chunks(_chunks(5))
    seed.close()
    db = ShardedRecallDB.open(index_dir)
    reader = sqlite3.connect(list_shards(current_generation_dir(index_dir))[1])
    try:
        # A reader holding a transaction stops the checkpoint short of the main file the mirror copies.
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM chunks").fetchone()
        with caplog.at_level(logging.WARNING, logger="synapt.recall.generations"):
            db.save_embeddings({ShardedRecallDB._encode_chunk_rowid(2, 1): [0.1] * 384})
        reader.rollback()
    finally:
        reader.close()
        db.close()
    lines = [r.getMessage() for r in caplog.records if "flat-layout shard mirror" in r.getMessage()]
    if list_shards(index_dir):
        assert lines == []
        _assert_flat_equals_current(index_dir)
    else:
        assert len(lines) == 1 and "flat shards removed" in lines[0], lines
