"""recall_quick keeps its loaded index when a refresh only appends the caller's chunks.

The footer row pins what the Freshness line times: refresh_wall is the
caller-tail refresh alone, call_wall the whole tool call. The census rows
describe the in-place append: after a refresh that only adds the caller's
own chunks, the server must not reload every chunk from the store.
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import pytest

from synapt.recall import query_freshness, server
from synapt.recall.core import TranscriptIndex
from synapt.recall.query_freshness import refresh_current_session

from tests.recall.test_query_freshness import _policy, _source, _write_turn

QUERY = "lanternfish harbour"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A real store holding one caller turn, wired into the server."""
    transcript = tmp_path / "m_00000001.jsonl"
    _write_turn(transcript, f"first {QUERY} note", "first answer", 20)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )
    monkeypatch.setattr(
        "synapt.recall.query_freshness.refresh_current_session",
        lambda index_dir, caller_root: refresh_current_session(
            index_dir, caller_root, policy=_policy()
        ),
    )
    refresh_current_session(index_dir, tmp_path, policy=_policy())
    monkeypatch.setattr(server, "project_index_dir", lambda: index_dir)
    loads: list[Path] = []
    real_load = TranscriptIndex.load.__func__

    def counting_load(cls, directory, use_embeddings=False):
        loads.append(directory)
        return real_load(cls, directory, use_embeddings=use_embeddings)

    monkeypatch.setattr(TranscriptIndex, "load", classmethod(counting_load))
    server._invalidate_cache()
    yield transcript, index_dir, loads
    server._invalidate_cache()


def _quick(query: str) -> str:
    return server._recall_quick_impl(query, semantic_fallback=False)


def _wall(name: str, text: str) -> float:
    found = re.findall(rf"\b{name}=([0-9.]+)s\b", text)
    assert len(found) == 1, text
    return float(found[0])


def test_footer_names_refresh_wall_and_call_wall(store, monkeypatch):
    _, _, loads = store
    delay = 0.3
    counting_load = TranscriptIndex.load.__func__

    def slow_load(cls, directory, use_embeddings=False):
        time.sleep(delay)
        return counting_load(cls, directory, use_embeddings=use_embeddings)

    monkeypatch.setattr(TranscriptIndex, "load", classmethod(slow_load))
    # The sleep sits outside the refresh's own clock, so refresh_wall stays
    # under delay while call_wall, which starts before the refresh, holds it.
    fixture_refresh = query_freshness.refresh_current_session
    refreshes: list[Path] = []

    def slow_refresh(index_dir, caller_root):
        refreshes.append(index_dir)
        time.sleep(delay)
        return fixture_refresh(index_dir, caller_root)

    monkeypatch.setattr(
        "synapt.recall.query_freshness.refresh_current_session", slow_refresh
    )

    out = _quick(QUERY)

    footer = out.rsplit("\n", 1)[-1]
    assert footer.startswith("Freshness: "), out
    assert loads, "the stub load never ran"
    assert refreshes, "the stub refresh never ran"
    assert " wall=" not in footer
    assert _wall("call_wall", footer) >= 2 * delay
    assert _wall("refresh_wall", footer) < delay


def _append_turn(transcript: Path) -> None:
    _write_turn(transcript, f"second {QUERY} zephyrquartz", "second answer", 21)


def test_new_chunks_are_found_without_a_full_load(store, monkeypatch):
    transcript, _, loads = store
    _quick(QUERY)
    assert len(loads) == 1
    _append_turn(transcript)

    built: list[int] = []
    real_with_tail = TranscriptIndex.with_tail.__func__

    def spy_with_tail(cls, old):
        index = real_with_tail(cls, old)
        if index is not None:
            built.append(len(index._query_cache))
        return index

    monkeypatch.setattr(TranscriptIndex, "with_tail", classmethod(spy_with_tail))

    out = _quick("zephyrquartz")

    assert "zephyrquartz" in out
    assert len(loads) == 1, f"full loads after the append: {len(loads) - 1}"
    # The new index answers from the store, never from a query cache made before the tail.
    assert built == [0]


def _ranked(index: TranscriptIndex, query: str) -> list[tuple[str, float]]:
    seen: list[list[tuple[int, float]]] = []
    real = index._format_results

    def spy(ranked, *args, **kwargs):
        seen.append(list(ranked))
        return real(ranked, *args, **kwargs)

    index._format_results = spy
    try:
        index.lookup(query, max_chunks=5)
    finally:
        del index._format_results
    assert seen, "lookup did not rank"
    return [(index.chunks[idx].id, score) for idx, score in seen[-1]]


def test_appended_index_ranks_like_a_fresh_load(store):
    transcript, index_dir, loads = store
    _quick(QUERY)
    _append_turn(transcript)
    _quick(QUERY)
    assert len(loads) == 1, f"full loads after the append: {len(loads) - 1}"

    kept = _ranked(server._get_index(use_embeddings=False), QUERY)
    fresh = _ranked(TranscriptIndex.load(index_dir), QUERY)

    assert [cid for cid, _ in kept] == [cid for cid, _ in fresh]
    for (_, a), (_, b) in zip(kept, fresh):
        assert abs(a - b) <= 1e-9


def test_reader_mid_append_sees_one_whole_index(store, monkeypatch):
    transcript, index_dir, loads = store
    _quick(QUERY)
    held = server._get_index(use_embeddings=False)
    before = len(held.chunks)
    _append_turn(transcript)
    seen: list[tuple[int, int]] = []
    wrapped = refresh_current_session

    def refresh_with_reader(index_dir, caller_root):
        result = wrapped(index_dir, caller_root, policy=_policy())

        def read():
            index = server._get_index(use_embeddings=False)
            seen.append((len(index.chunks), len(index._id_to_idx)))

        reader = threading.Thread(target=read)
        reader.start()
        reader.join(5)
        return result

    monkeypatch.setattr(
        "synapt.recall.query_freshness.refresh_current_session", refresh_with_reader
    )
    _quick(QUERY)
    after = len(server._get_index(use_embeddings=False).chunks)

    assert after > before
    # The index a reader already held is never changed under it.
    assert len(held.chunks) == before
    assert len(seen) == 1
    assert seen[0][0] == seen[0][1]
    assert seen[0][0] in (before, after)
    assert len(loads) == 1, f"full loads around the append: {len(loads) - 1}"


def _foreign_commit(index_dir: Path) -> None:
    """A commit by another connection to the file the store writes, as another server or a build makes."""
    import sqlite3

    name = "index.db" if (index_dir / "index.db").exists() else "recall.db"
    conn = sqlite3.connect(index_dir / name)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS foreign_probe (n INTEGER)")
        conn.execute("INSERT INTO foreign_probe VALUES (1)")
        conn.commit()
    finally:
        conn.close()


def test_a_foreign_commit_between_calls_reloads(store):
    _, index_dir, loads = store
    _quick(QUERY)
    _foreign_commit(index_dir)

    _quick(QUERY)

    assert len(loads) == 2


def test_a_foreign_commit_before_the_refresh_opens_reloads(store, monkeypatch):
    transcript, index_dir, loads = store
    _quick(QUERY)
    _append_turn(transcript)
    fixture_refresh = query_freshness.refresh_current_session

    def foreign_first(index_dir, caller_root):
        _foreign_commit(index_dir)
        return fixture_refresh(index_dir, caller_root)

    monkeypatch.setattr("synapt.recall.query_freshness.refresh_current_session", foreign_first)

    out = _quick("zephyrquartz")

    assert "zephyrquartz" in out
    assert len(loads) == 2, "a commit after the stamp read was taken as our own tail"


def test_a_foreign_commit_inside_the_refresh_reloads(store, monkeypatch):
    transcript, index_dir, loads = store
    _quick(QUERY)
    _append_turn(transcript)
    from synapt.recall.storage import RecallDB

    real_replace = RecallDB.replace_query_tail

    def replace_then_foreign(self, *args, **kwargs):
        result = real_replace(self, *args, **kwargs)
        _foreign_commit(index_dir)
        return result

    monkeypatch.setattr(RecallDB, "replace_query_tail", replace_then_foreign)

    out = _quick("zephyrquartz")

    assert "zephyrquartz" in out
    assert len(loads) == 2, "a commit during our refresh was taken as our own tail"


def test_a_call_that_indexes_nothing_keeps_the_cache(store, monkeypatch):
    transcript, _, loads = store
    _quick(QUERY)
    _append_turn(transcript)
    monkeypatch.setattr(
        "synapt.recall.query_freshness.refresh_current_session",
        lambda index_dir, caller_root: refresh_current_session(
            index_dir, caller_root, policy=_policy(byte_trigger=1 << 30, age_threshold_seconds=1 << 30)
        ),
    )

    out = _quick(QUERY)

    assert "below_threshold" in out
    assert len(loads) == 1


def test_a_shrunk_tail_matches_a_fresh_load(store):
    transcript, index_dir, _ = store
    first = transcript.read_text()
    _append_turn(transcript)
    _quick(QUERY)
    assert len(server._get_index(use_embeddings=False).chunks) == 2
    transcript.write_text(first)

    _quick(QUERY)

    kept = sorted(chunk.id for chunk in server._get_index(use_embeddings=False).chunks)
    fresh = sorted(chunk.id for chunk in TranscriptIndex.load(index_dir).chunks)
    assert kept == fresh
    assert len(fresh) == 1


def test_a_lifted_suppression_matches_a_fresh_load(tmp_path, monkeypatch):
    """A tail that stops suppressing the base brings base chunks back that the cached index had dropped."""
    from synapt.recall.core import parse_transcript
    from synapt.recall.storage import RecallDB

    from tests.recall.test_query_freshness import SESSION

    transcript = tmp_path / f"{SESSION}.jsonl"
    for minute in range(20, 24):
        _write_turn(transcript, f"base turn {minute} {QUERY} " + "x" * 200, "answer", minute)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    db = RecallDB(index_dir / "recall.db")
    db.save_chunks(parse_transcript(transcript))
    db.close()
    policy = {"value": _policy()}
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts",
        lambda root: [_source(transcript)],
    )
    monkeypatch.setattr(
        "synapt.recall.query_freshness.refresh_current_session",
        lambda index_dir, caller_root: refresh_current_session(
            index_dir, caller_root, policy=policy["value"]
        ),
    )
    monkeypatch.setattr(server, "project_index_dir", lambda: index_dir)
    server._invalidate_cache()
    try:
        _quick(QUERY)
        assert server._get_index(use_embeddings=False)._suppressed_sessions == {SESSION}
        # The same bytes in a different order: the cursor's prefix no longer proves the source,
        # so the refresh clears the session's tail and reparses it, here only part of the way.
        text = transcript.read_text().replace("base turn 20", "BASE TURN 20")
        transcript.write_text(text)
        first_turn_end = text.find("\n", text.find("\n") + 1) + 1
        policy["value"] = _policy(step_bytes=first_turn_end, byte_cap=first_turn_end)

        out = _quick(QUERY)

        assert "PARTIAL" in out or "partial" in out, out
        kept = sorted(chunk.id for chunk in server._get_index(use_embeddings=False).chunks)
        fresh = sorted(chunk.id for chunk in TranscriptIndex.load(index_dir).chunks)
        assert kept == fresh
    finally:
        server._invalidate_cache()


def test_another_connection_that_reads_and_closes_keeps_the_cache(store):
    """A connection that opens, reads and closes writes nothing; on Linux the file stamp can still move.

    Not covered, by design: a TRUNCATE checkpoint by ANOTHER connection moves PRAGMA data_version too
    (measured 2 -> 3 on sqlite3), so it reloads once; see the checkpoint row below. Recall forces one only beside a write (a published generation,
    saved embeddings), and an automatic checkpoint runs on a commit or on the last close, never while the
    cached index holds its own connection open.
    """
    import sqlite3

    _, index_dir, loads = store
    _quick(QUERY)
    name = "index.db" if (index_dir / "index.db").exists() else "recall.db"
    conn = sqlite3.connect(index_dir / name)
    conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    conn.close()

    server._get_index(use_embeddings=False)
    _quick(QUERY)

    assert len(loads) == 1, f"reloads after another connection read and closed: {len(loads) - 1}"


@pytest.mark.parametrize("mode, reloads", [("PASSIVE", 0), ("TRUNCATE", 1)])
def test_a_checkpoint_by_another_connection(store, mode, reloads):
    """Another connection checkpoints the WAL and writes nothing: a PASSIVE checkpoint keeps the cache, and a
    TRUNCATE checkpoint moves PRAGMA data_version on the cached connection, so it reloads exactly once."""
    import sqlite3

    _, index_dir, loads = store
    _quick(QUERY)
    name = "index.db" if (index_dir / "index.db").exists() else "recall.db"
    conn = sqlite3.connect(index_dir / name)
    conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
    conn.close()

    server._get_index(use_embeddings=False)
    _quick(QUERY)
    _quick(QUERY)

    assert len(loads) == 1 + reloads, f"{mode}: reloads after another connection checkpointed: {len(loads) - 1}"


@pytest.fixture
def sharded(tmp_path, monkeypatch):
    """A sharded store: index.db plus one data shard holding a base chunk, wired into the server."""
    from synapt.recall.core import TranscriptChunk
    from synapt.recall.storage import RecallDB

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    RecallDB(index_dir / "index.db").close()
    shard = RecallDB(index_dir / "data_0001.db")
    shard.save_chunks([TranscriptChunk(
        id="base:t0", session_id="base", timestamp="2026-08-29T00:00:00Z", turn_index=0,
        user_text=f"base {QUERY}", assistant_text="",
    )])
    shard.close()
    monkeypatch.setattr(server, "project_index_dir", lambda: index_dir)
    loads: list[Path] = []
    real_load = TranscriptIndex.load.__func__

    def counting_load(cls, directory, use_embeddings=False):
        loads.append(directory)
        return real_load(cls, directory, use_embeddings=use_embeddings)

    monkeypatch.setattr(TranscriptIndex, "load", classmethod(counting_load))
    server._invalidate_cache()
    yield index_dir, loads
    server._invalidate_cache()


def test_a_foreign_commit_to_a_data_shard_reloads(sharded):
    import sqlite3

    index_dir, loads = sharded
    server._get_index(use_embeddings=False)
    assert len(server._held_dbs(server._cached_index)) == 2  # index.db and the shard
    conn = sqlite3.connect(index_dir / "data_0001.db")
    conn.execute("CREATE TABLE IF NOT EXISTS foreign_probe (n INTEGER)")
    conn.execute("INSERT INTO foreign_probe VALUES (1)")
    conn.commit()
    conn.close()

    server._get_index(use_embeddings=False)

    assert len(loads) == 2


def test_a_shard_commit_inside_the_refresh_reloads(sharded, tmp_path, monkeypatch):
    """Our refresh writes through index.db only: a commit by another connection to a DATA SHARD while it runs
    is not ours, so the next call reloads instead of keeping the cache at a baseline that holds it."""
    import sqlite3

    from synapt.recall.storage import RecallDB

    index_dir, loads = sharded
    transcript = tmp_path / "shard_caller.jsonl"
    _write_turn(transcript, f"first {QUERY} note", "first answer", 20)
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts", lambda root: [_source(transcript)]
    )
    monkeypatch.setattr(
        "synapt.recall.query_freshness.refresh_current_session",
        lambda index_dir, caller_root: refresh_current_session(index_dir, caller_root, policy=_policy()),
    )
    _quick(QUERY)
    assert len(server._held_dbs(server._cached_index)) == 2  # index.db and the shard
    loaded = len(loads)
    _append_turn(transcript)
    real_replace = RecallDB.replace_query_tail

    def replace_then_shard_commit(self, *args, **kwargs):
        result = real_replace(self, *args, **kwargs)
        conn = sqlite3.connect(index_dir / "data_0001.db")
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS foreign_probe (n INTEGER)")
            conn.execute("INSERT INTO foreign_probe VALUES (1)")
            conn.commit()
        finally:
            conn.close()
        return result

    monkeypatch.setattr(RecallDB, "replace_query_tail", replace_then_shard_commit)
    _quick("zephyrquartz")
    monkeypatch.setattr(RecallDB, "replace_query_tail", real_replace)
    _quick(QUERY)

    assert len(loads) == loaded + 1, "a data-shard commit during our refresh was taken as our own"


@pytest.mark.skipif(os.name != "posix", reason="Windows refuses to replace a file an open connection holds, so the case cannot arise there")
def test_a_held_file_replaced_under_its_connection_does_not_match(store):
    """The inode is the only part of the token that sees a file replaced under the cached connection (that
    connection still reads the old file, so its data_version cannot move)."""
    import os
    import shutil

    _, index_dir, _ = store
    _quick(QUERY)
    assert server._cache_matches(index_dir)
    name = "index.db" if (index_dir / "index.db").exists() else "recall.db"
    shutil.copyfile(index_dir / name, index_dir / f"{name}.swap")
    os.replace(index_dir / f"{name}.swap", index_dir / name)

    assert not server._cache_matches(index_dir)


@pytest.mark.skipif(os.name != "posix", reason="Windows refuses to replace a file an open connection holds, so the case cannot arise there")
def test_a_held_file_replaced_inside_the_refresh_reloads(store, monkeypatch):
    """A file replaced under the refresh's own connection does not move that connection's data_version, so
    only the inode in the token's first part shows it: the next call reloads instead of keeping a cache whose
    connection's file was replaced."""
    import os
    import sqlite3

    from synapt.recall.storage import RecallDB

    transcript, index_dir, loads = store
    _quick(QUERY)
    _append_turn(transcript)
    name = "index.db" if (index_dir / "index.db").exists() else "recall.db"
    real_replace = RecallDB.replace_query_tail

    def replace_then_swap_file(self, *args, **kwargs):
        result = real_replace(self, *args, **kwargs)
        src = sqlite3.connect(index_dir / name)
        dst = sqlite3.connect(index_dir / f"{name}.swap")
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        os.replace(index_dir / f"{name}.swap", index_dir / name)
        return result

    monkeypatch.setattr(RecallDB, "replace_query_tail", replace_then_swap_file)
    _quick("zephyrquartz")

    # Asserted after this call: the next call's own refresh commits and reloads whatever this one decided.
    assert len(loads) == 2, "a file replaced during our refresh was taken as our own baseline"


_BEFORE = ("gen-1", (("index.db", 111, 5), ("data_0001.db", 222, 7)))


def _window(after):
    """A probe whose refresh connection never saw another commit (data_version 3 at open and after)."""
    return {"open": (3, _BEFORE), "after": (after, 3)}


@pytest.mark.parametrize(
    "after, ours",
    [
        (("gen-1", (("index.db", 111, 6), ("data_0001.db", 222, 7))), True),
        (("gen-1", (("index.db", 999, 6), ("data_0001.db", 222, 7))), False),
        (("gen-1", (("index.db", 999, 5), ("data_0001.db", 222, 7))), False),
        (("gen-1", (("index.db", 111, 6), ("data_0001.db", 222, 8))), False),
        (("gen-2", (("index.db", 111, 6), ("data_0001.db", 222, 7))), False),
    ],
    ids=["own-commit", "file-replaced", "file-replaced-unwritten", "shard-moved", "generation-moved"],
)
def test_only_our_write_lets_only_the_refresh_files_data_version_move(after, ours):
    """Inside the window only the refresh's own file's data_version may move: a new inode there, a moved
    data shard, or a new CURRENT generation was another writer, so there is no baseline to keep."""
    got = server._only_our_write(_BEFORE, _window(after))

    assert (got == after) if ours else (got is None)


def test_a_generation_published_inside_the_refresh_reloads(sharded, tmp_path, monkeypatch):
    """Our refresh never moves the CURRENT generation: a generation published while it runs is another writer's,
    so the next call reloads instead of keeping the cache at a baseline that holds it."""
    from synapt.recall.generations import read_current_generation
    from synapt.recall.storage import RecallDB

    index_dir, loads = sharded
    transcript = tmp_path / "gen_caller.jsonl"
    _write_turn(transcript, f"first {QUERY} note", "first answer", 20)
    monkeypatch.setattr(
        "synapt.recall.query_freshness.caller_transcripts", lambda root: [_source(transcript)]
    )
    monkeypatch.setattr(
        "synapt.recall.query_freshness.refresh_current_session",
        lambda index_dir, caller_root: refresh_current_session(index_dir, caller_root, policy=_policy()),
    )
    _quick(QUERY)
    loaded = len(loads)
    _append_turn(transcript)
    real_replace = RecallDB.replace_query_tail
    seen = {}

    def replace_then_publish(self, *args, **kwargs):
        result = real_replace(self, *args, **kwargs)
        before = read_current_generation(index_dir)
        (index_dir / "CURRENT").write_text("probe-generation-0001\n")
        seen["gen"] = (before, read_current_generation(index_dir))
        return result

    monkeypatch.setattr(RecallDB, "replace_query_tail", replace_then_publish)
    _quick("zephyrquartz")
    monkeypatch.setattr(RecallDB, "replace_query_tail", real_replace)
    assert seen and seen["gen"][0] != seen["gen"][1], f"precondition: CURRENT did not move {seen}"
    _quick(QUERY)

    assert len(loads) == loaded + 1, "a generation published inside our refresh was taken as our own"
