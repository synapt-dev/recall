"""A transient embedding failure must not lose the knowledge.

An unreachable provider, a timeout, or a model being pulled is temporary.
Failing the save over it discards a fact the caller asked to keep, so the save
degrades: the node lands with no vector, and the report says so in words that are
true -- the node is saved and keyword search finds it, while vector search does
not see it.

NOTHING HERE PROMISES WHEN THE VECTOR ARRIVES. `_build_knowledge_embeddings` can
fill such a row, and the test below drives it directly to show that it can, but
its only caller is the background chunk-embedding build, which does not run on a
store that has no chunk embeddings. On a first-user store, measured: two later
runs with a healthy provider left the vector absent. A bounded backfill on a
healthy save is a follow-on, not a property of this path.

The other half matters as much: a provider that SUCCEEDS but returns the wrong
WIDTH is NOT transient, and it must still fail atomically with no node. The pack
that rejects it therefore stays OUTSIDE the try that degrades.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from synapt.recall.embeddings import EmbeddingProvider

MARKER = "transient-backfill-witness-marker"


class _Transient(EmbeddingProvider):
    """Fails the way an unreachable provider fails."""

    @property
    def dim(self):
        from synapt.recall.storage import EMBEDDING_DIM

        return EMBEDDING_DIM

    def embed(self, texts):
        raise ConnectionError("connection refused")


class _WrongWidth(EmbeddingProvider):
    """SUCCEEDS and returns the wrong width -- NOT transient."""

    @property
    def dim(self):
        return 1024

    def embed(self, texts):
        return [[0.0] * 1024 for _ in texts]


class _Good(EmbeddingProvider):
    @property
    def dim(self):
        from synapt.recall.storage import EMBEDDING_DIM

        return EMBEDDING_DIM

    def embed(self, texts):
        return [[0.0] * self.dim for _ in texts]


def _store_holds(marker: str, root) -> bool:
    return any(marker in p.read_text(errors="ignore")
               for p in Path(root).rglob("knowledge.jsonl"))


@pytest.fixture
def rooted(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIPSPACE_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_transient_failure_keeps_the_node_and_says_so(rooted, monkeypatch):
    """THE FIRST HALF. Pre-fix this returned a save FAILURE and left no usable
    state; now the node survives and the report is honest about the vector."""
    import synapt.recall.server as server

    monkeypatch.setattr(server, "get_embedding_provider", lambda: _Transient())
    result = server.recall_save(content=MARKER, category="decision")

    assert result.startswith("Knowledge node saved"), (
        f"a transient provider failure must not fail the save; got {result[:140]!r}"
    )
    assert "saved without embeddings" in result, (
        f"and it must say the vector is missing; got {result[:160]!r}"
    )
    assert _store_holds(MARKER, rooted), "the knowledge must not be lost"


def test_the_backfill_fills_the_vector_it_left_empty(rooted, monkeypatch):
    """THE FUNCTION, WHEN CALLED.

    This drives `_build_knowledge_embeddings` directly, so it proves the backfill
    CAN fill a row that a transient save left empty. It does NOT prove anything
    calls it on the production path, and it is not offered as that: measured, two
    later runs with a healthy provider left the vector absent on a first-user
    store. The production trigger is a follow-on.
    """
    import synapt.recall.server as server
    from synapt.recall.core import TranscriptIndex
    from synapt.recall.sharding import live_store_path

    monkeypatch.setattr(server, "get_embedding_provider", lambda: _Transient())
    result = server.recall_save(content=MARKER, category="decision")
    assert "saved without embeddings" in result, result[:160]

    node_id = result.split(": ", 1)[1].split(" ", 1)[0]

    from synapt.recall.storage import RecallDB

    db = RecallDB(live_store_path(server.project_index_dir(None)))
    try:
        assert node_id not in db.get_knowledge_embeddings_by_id(), (
            "the transient save must have left NO vector, or this test is not "
            "measuring the backfill"
        )
        # The backfill, given a working provider.
        missing = db.get_knowledge_rowids_without_embeddings()
        assert any(rid for rid, _ in missing), "the node must be offered to the backfill"
        db_for_backfill = db
    finally:
        pass

    # The backfill is an instance method whose only uses of `self` are the two
    # defaults this call overrides, so a bare holder object drives it without
    # standing a whole index up. (My first version passed None for self.)
    import types

    TranscriptIndex._build_knowledge_embeddings(
        types.SimpleNamespace(), db=db_for_backfill, provider=_Good()
    )

    db = RecallDB(live_store_path(server.project_index_dir(None)))
    try:
        stored = db.get_knowledge_embeddings_by_id()
        assert node_id in stored, (
            "the backfill did not fill the vector the transient save left empty; "
            f"stored ids: {sorted(stored)[:5]}"
        )
    finally:
        db.close()


def test_control_a_healthy_provider_still_embeds_at_save_time(rooted, monkeypatch):
    """CONTROL: without this, a save that never embeds anything would pass both
    tests above."""
    import synapt.recall.server as server

    monkeypatch.setattr(server, "get_embedding_provider", lambda: _Good())
    result = server.recall_save(content=MARKER, category="decision")

    assert result.startswith("Knowledge node saved"), result[:140]
    assert "embedded for vector search" in result, result[:160]


def test_wrong_width_is_still_atomic_not_degraded(rooted, monkeypatch):
    """THE #1004 GUARANTEE MUST SURVIVE (a). A wrong width is not transient, so
    the degradation must not swallow it: no node may land."""
    import synapt.recall.server as server

    monkeypatch.setattr(server, "get_embedding_provider", lambda: _WrongWidth())
    result = server.recall_save(content=MARKER, category="decision")

    assert result.startswith("Knowledge save failed"), (
        f"a wrong-width provider must still FAIL the save; got {result[:140]!r}"
    )
    assert not _store_holds(MARKER, rooted), (
        "and it must still leave NO node -- the pack stays outside the try"
    )
