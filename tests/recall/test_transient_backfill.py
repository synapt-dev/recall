"""A transient embedding failure must not lose the knowledge.

An unreachable provider, a timeout, or a model being pulled is temporary.
Failing the save over it discards a fact the caller asked to keep, so the save
degrades: the node lands with no vector, and the report says so in words that are
true -- the node is saved and keyword search finds it, while vector search does
not see it.

NOTHING HERE PROMISES WHEN THE VECTOR ARRIVES, but something now delivers it: a
LATER HEALTHY SAVE fills up to `KNOWLEDGE_BACKFILL_PER_SAVE` rows this path left
empty, so the store heals in the flow a first user actually runs. Two tests cover
the two halves -- one drives the backfill function directly to show it CAN fill
such a row, and `test_a_healthy_save_fills_the_vector_an_unhealthy_one_left_empty`
drives the PRODUCTION path, which is the half that was missing before.

Two more witnesses hold the CLAIMS the backfill's own comments make, and they
exist because a reviewer mutated both claims and the suite stayed green:
`test_a_failing_repair_cannot_turn_a_successful_save_into_a_failure` is red if the
repair re-raises instead of swallowing, and
`test_a_degraded_save_does_not_attempt_the_repair` is red if the backfill is also
called on the degrading path. Both drive a provider that works SINGLY and fails in
BATCH, which is the shape that separates the save's own call from the store's
repair -- every other provider here is uniformly good or uniformly bad, which is
why nothing else covered either claim.

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
    """THE FUNCTION, WHEN CALLED DIRECTLY.

    This drives `_build_knowledge_embeddings` directly, so it proves the backfill
    CAN fill a row that a transient save left empty. It says nothing about
    whether anything calls it on the production path; that half is
    `test_a_healthy_save_fills_the_vector_an_unhealthy_one_left_empty` below,
    which needs no direct call.
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


def _stored_embedding_ids(server) -> dict:
    """The ids the store holds vectors for, read through the store itself."""
    from synapt.recall.sharding import live_store_path
    from synapt.recall.storage import RecallDB

    db = RecallDB(live_store_path(server.project_index_dir(None)))
    try:
        return db.get_knowledge_embeddings_by_id()
    finally:
        db.close()


def _missing_vector_count(server) -> int:
    from synapt.recall.sharding import live_store_path
    from synapt.recall.storage import RecallDB

    db = RecallDB(live_store_path(server.project_index_dir(None)))
    try:
        return len(db.get_knowledge_rowids_without_embeddings())
    finally:
        db.close()


def _missing_rowids(server) -> list[int]:
    """The rowids the queue offers, in the order the accessor returns them."""
    from synapt.recall.sharding import live_store_path
    from synapt.recall.storage import RecallDB

    db = RecallDB(live_store_path(server.project_index_dir(None)))
    try:
        return [rid for rid, _ in db.get_knowledge_rowids_without_embeddings()]
    finally:
        db.close()


def test_a_healthy_save_fills_the_vector_an_unhealthy_one_left_empty(
    rooted, monkeypatch
):
    """THE PRODUCTION PATH -- the half that was missing.

    Apollo's r1 on a0b50ebf measured this with a two-process probe and read
    `vector present after later run: False`, twice. This is the same claim with
    the second step driven through `recall_save` rather than through a direct
    call, so it fails if the backfill is merely PRESENT rather than FIRED.

    Mutation: delete the `backfill_knowledge_embeddings` call at the end of
    `recall_save` and the last assertion goes red while
    `test_the_backfill_fills_the_vector_it_left_empty` (the direct call) stays
    green -- which is exactly the difference between the two halves.
    """
    import synapt.recall.server as server

    monkeypatch.setattr(server, "get_embedding_provider", lambda: _Transient())
    first = server.recall_save(content=MARKER, category="decision")
    assert "saved without embeddings" in first, first[:160]
    first_id = first.split(": ", 1)[1].split(" ", 1)[0]

    assert first_id not in _stored_embedding_ids(server), (
        "the transient save must have left NO vector, or this test is not "
        "measuring the backfill"
    )

    # A later save with a provider that WORKS. Nothing here reaches into the
    # backfill: this test does one thing, and that is save twice.
    monkeypatch.setattr(server, "get_embedding_provider", lambda: _Good())
    second = server.recall_save(content=f"{MARKER}-second", category="decision")
    assert "embedded for vector search" in second, second[:160]

    assert first_id in _stored_embedding_ids(server), (
        "a HEALTHY save did not fill the vector the previous, transient one left "
        "empty -- this is the production-path half"
    )


def test_the_backfill_is_bounded_and_moves_forward(rooted, monkeypatch):
    """A save must not walk the whole backlog, and must not re-walk a slice.

    The bound is asserted EXACTLY -- it fills a full batch when a full batch is
    available -- rather than as an upper limit, so a backfill that quietly
    stopped filling would fail this too.

    MUTATION 1 IS CAUGHT: pass `limit=None` at the call site and the first count
    assertion goes red, because one save then drains the whole backlog.

    MUTATION 2 IS NOT CAUGHT, and saying so is the point. Dropping `ORDER BY
    rowid` from the accessor leaves this test GREEN. Measured: SQLite returns
    rowid-ascending order for this query under every plan it chooses, including
    `SEARCH knowledge USING INDEX ix_status`, because a secondary index stores
    the rowid as its payload and these rows tie on the key. So the ORDER BY is
    DEFENSIVE -- it is insurance against a future edit that WOULD disturb the
    order (a JOIN, GROUP BY or DISTINCT in the accessor), not something this
    test can observe the absence of through the accessor's output.

    The `queued == sorted(queued)` assertion below is the tripwire for exactly
    that future edit. It is a regression witness, NOT a mutation-kill for the
    ORDER BY, and it is written here rather than claimed in the accessor's
    docstring because a guarantee nobody can falsify is the thing this file
    keeps finding.
    """
    import synapt.recall.server as server

    bound = server.KNOWLEDGE_BACKFILL_PER_SAVE
    backlog = bound + 3

    # Seed more missing vectors than one batch can hold.
    monkeypatch.setattr(server, "get_embedding_provider", lambda: _Transient())
    for i in range(backlog):
        result = server.recall_save(content=f"{MARKER}-{i}", category="decision")
        assert "saved without embeddings" in result, result[:160]
    assert _missing_vector_count(server) == backlog

    # The queue is OFFERED in a stable, ascending order. This is the observable
    # contract `ORDER BY rowid` exists to provide, and the tripwire for a future
    # edit that disturbs it -- not a check that the ORDER BY is present. Read the
    # docstring above before treating a green here as evidence of anything else.
    queued = _missing_rowids(server)
    assert queued == sorted(queued), (
        f"the queue must be offered in ascending rowid order; got {queued}"
    )

    # ONE healthy save. Its own node is embedded inline, so the only rows it can
    # fill are the ones behind it in the queue.
    monkeypatch.setattr(server, "get_embedding_provider", lambda: _Good())
    server.recall_save(content=f"{MARKER}-healthy", category="decision")

    remaining = _missing_vector_count(server)
    assert remaining == backlog - bound, (
        f"one healthy save should fill exactly {bound} rows and leave "
        f"{backlog - bound}; it left {remaining}"
    )

    # FORWARD, not from the top again: a second healthy save drains the rest
    # rather than re-reading the slice already filled.
    server.recall_save(content=f"{MARKER}-healthy-2", category="decision")
    assert _missing_vector_count(server) == 0, (
        "the queue did not move forward -- the second save re-read a slice it "
        "had already filled"
    )


# ---------------------------------------------------------------------------
# The two claims Apollo's r1 mutated and found unwitnessed.
# ---------------------------------------------------------------------------

class _SingleWorksBatchFails(EmbeddingProvider):
    """The save's own call works; the BATCH repair path is down.

    This is the shape that separates the two paths, and no other provider here
    covers it: every other stub is uniformly good or uniformly bad, so nothing
    else distinguishes "this save embedded its own node" from "the store
    repaired the backlog".
    """

    def __init__(self) -> None:
        self.batch_calls = 0

    @property
    def dim(self):
        from synapt.recall.storage import EMBEDDING_DIM

        return EMBEDDING_DIM

    def embed_single(self, text):
        return [0.0] * self.dim

    def embed(self, texts):
        self.batch_calls += 1
        raise ConnectionError("batch path down")


class _SingleFailsBatchCounted(EmbeddingProvider):
    """Both paths fail, and a batch call is COUNTED so its ABSENCE is checkable."""

    def __init__(self) -> None:
        self.batch_calls = 0

    @property
    def dim(self):
        from synapt.recall.storage import EMBEDDING_DIM

        return EMBEDDING_DIM

    def embed_single(self, text):
        raise ConnectionError("provider unreachable")

    def embed(self, texts):
        self.batch_calls += 1
        raise ConnectionError("provider unreachable")


def test_a_failing_repair_cannot_turn_a_successful_save_into_a_failure(
    rooted, monkeypatch
):
    """THE REPAIR'S OWN FAILURE IS SWALLOWED -- the claim, witnessed.

    Mutation: make `backfill_knowledge_embeddings` re-raise instead of logging
    and returning 0, and this goes red. The whole suite stayed green under that
    mutation before this test existed. A re-raise would make a save that has
    ALREADY COMMITTED report failure, and an MCP caller would retry into a
    version bump.

    The batch call is COUNTED so this cannot pass vacuously: with an empty
    backlog the backfill returns before touching the provider, so the assertion
    on batch_calls is what proves a repair was actually attempted and failed.
    """
    import synapt.recall.server as server

    monkeypatch.setattr(server, "get_embedding_provider", lambda: _Transient())
    older = server.recall_save(content=f"{MARKER}-older", category="decision")
    assert "saved without embeddings" in older, older[:160]
    older_id = older.split(": ", 1)[1].split(" ", 1)[0]

    provider = _SingleWorksBatchFails()
    monkeypatch.setattr(server, "get_embedding_provider", lambda: provider)
    result = server.recall_save(content=MARKER, category="decision")

    assert provider.batch_calls >= 1, (
        "the backlog was never offered to the repair, so this test is not "
        "measuring a failing repair"
    )
    assert result.startswith("Knowledge node saved"), (
        "a repair the caller never asked for must not turn a committed write into "
        f"a reported failure; got {result[:160]!r}"
    )
    assert "embedded for vector search" in result, result[:160]

    stored = _stored_embedding_ids(server)
    new_id = result.split(": ", 1)[1].split(" ", 1)[0]
    assert new_id in stored, "this save's own vector must still be written"
    assert older_id not in stored, "the repair failed, so the older row stays empty"


def test_a_degraded_save_does_not_attempt_the_repair(rooted, monkeypatch):
    """THE REPAIR RUNS ONLY WHEN THE PROVIDER JUST WORKED -- the claim, witnessed.

    Mutation: also call the backfill in recall_save's degrading `except` branch
    and this goes red. The whole suite stayed green under that mutation before
    this test existed. What it would cost: a SECOND provider call on every save
    during an outage, so an unreachable Ollama or a timeout is paid twice.
    """
    import synapt.recall.server as server

    monkeypatch.setattr(server, "get_embedding_provider", lambda: _Transient())
    for i in range(3):
        server.recall_save(content=f"{MARKER}-{i}", category="decision")
    assert _missing_vector_count(server) == 3, "fixture: a backlog must exist"

    provider = _SingleFailsBatchCounted()
    monkeypatch.setattr(server, "get_embedding_provider", lambda: provider)
    result = server.recall_save(content=MARKER, category="decision")

    assert "saved without embeddings" in result, result[:160]
    assert provider.batch_calls == 0, (
        "a save whose own provider call just failed must NOT attempt the batch "
        "repair: that pays the same outage twice on every save"
    )
