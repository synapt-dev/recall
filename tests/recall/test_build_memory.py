"""Build-path memory: TranscriptIndex on a fresh DB must not hold a BM25 corpus."""

import random
import tracemalloc
from pathlib import Path

from synapt.recall.core import TranscriptChunk, TranscriptIndex
from synapt.recall.storage import RecallDB

_WORDS = [f"w{i:05d}" for i in range(20000)]


def _chunks(n: int, seed: int = 7) -> list[TranscriptChunk]:
    rng = random.Random(seed)
    return [
        TranscriptChunk(
            id=f"s{i // 20}:t{i % 20}",
            session_id=f"s{i // 20}",
            timestamp=f"2026-01-01T00:{(i // 60) % 60:02d}:{i % 60:02d}+00:00",
            turn_index=i % 20,
            user_text=" ".join(rng.choices(_WORDS, k=40)),
            assistant_text=" ".join(rng.choices(_WORDS, k=80)),
        )
        for i in range(n)
    ]


def _init_peak(tmp_path: Path, n: int) -> int:
    chunks = _chunks(n)
    db = RecallDB(tmp_path / f"recall-{n}.db")
    try:
        tracemalloc.start()
        TranscriptIndex(chunks, use_embeddings=False, db=db)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        db.close()
    return peak


def test_build_init_peak_does_not_scale_with_corpus_tokens(tmp_path):
    small = _init_peak(tmp_path, 1000)
    large = _init_peak(tmp_path, 4000)
    print(f"init peak: 1000 chunks {small / 2**20:.2f} MiB, 4000 chunks {large / 2**20:.2f} MiB")
    # An in-memory BM25 over 120 tokens a chunk costs several KiB a chunk;
    # the build path should add well under 1 KiB a chunk.
    assert large - small < 3000 * 1024


def test_index_on_empty_db_defers_bm25(tmp_path):
    db = RecallDB(tmp_path / "recall.db")
    index = TranscriptIndex(_chunks(50), use_embeddings=False, db=db)
    assert index._bm25 is None
    db.close()


def _index_with_target(tmp_path: Path) -> tuple[TranscriptIndex, RecallDB]:
    chunks = _chunks(50)
    chunks[17] = TranscriptChunk(
        id="target:t0", session_id="target", timestamp="2026-02-01T00:00:00+00:00",
        turn_index=0, user_text="where is the hermite spline quality curve",
        assistant_text="The hermite spline quality curve lives in zones.py.",
    )
    db = RecallDB(tmp_path / "recall.db")
    return TranscriptIndex(chunks, use_embeddings=False, db=db), db


def test_search_before_save_builds_bm25_and_finds_chunk(tmp_path):
    index, db = _index_with_target(tmp_path)

    before = index.lookup("hermite spline", max_chunks=3)
    assert index._bm25 is not None
    assert "zones.py" in before

    index.save(tmp_path)
    assert index._rowid_to_idx
    after = index.lookup("hermite spline", max_chunks=3)
    assert "zones.py" in after
    db.close()


def test_progressive_search_before_save_builds_bm25_and_finds_chunk(tmp_path):
    index, db = _index_with_target(tmp_path)

    result = index.lookup("hermite spline", max_chunks=3, max_sessions=2)
    assert index._bm25 is not None
    assert "zones.py" in result
    db.close()


def test_index_without_db_builds_bm25_at_init():
    index = TranscriptIndex(_chunks(50), use_embeddings=False)
    assert index._bm25 is not None
