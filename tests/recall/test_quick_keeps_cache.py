"""recall_quick keeps its loaded index when a refresh only appends the caller's chunks.

The footer row pins what the Freshness line times: refresh_wall is the
caller-tail refresh alone, call_wall the whole tool call. The census rows
describe the in-place append: after a refresh that only adds the caller's
own chunks, the server must not reload every chunk from the store.
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import pytest

from synapt.recall import server
from synapt.recall.core import TranscriptIndex
from synapt.recall.query_freshness import refresh_current_session

from tests.recall.test_query_freshness import _policy, _source, _write_turn

APPEND_NOT_BUILT = pytest.mark.xfail(strict=True, reason="in-place append not built")
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

    out = _quick(QUERY)

    footer = out.rsplit("\n", 1)[-1]
    assert footer.startswith("Freshness: "), out
    assert loads, "the stub load never ran"
    assert " wall=" not in footer
    assert _wall("call_wall", footer) >= delay
    assert _wall("refresh_wall", footer) < delay


def _append_turn(transcript: Path) -> None:
    _write_turn(transcript, f"second {QUERY} zephyrquartz", "second answer", 21)


@APPEND_NOT_BUILT
def test_new_chunks_are_found_without_a_full_load(store):
    transcript, _, loads = store
    _quick(QUERY)
    assert len(loads) == 1
    _append_turn(transcript)

    out = _quick("zephyrquartz")

    assert "zephyrquartz" in out
    assert len(loads) == 1, f"full loads after the append: {len(loads) - 1}"


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


@APPEND_NOT_BUILT
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


@APPEND_NOT_BUILT
def test_reader_mid_append_sees_one_whole_index(store, monkeypatch):
    transcript, index_dir, loads = store
    _quick(QUERY)
    before = len(server._get_index(use_embeddings=False).chunks)
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
    assert len(seen) == 1
    assert seen[0][0] == seen[0][1]
    assert seen[0][0] in (before, after)
    assert len(loads) == 1, f"full loads around the append: {len(loads) - 1}"
