"""A knowledge-only store must search its saved knowledge semantically.

Measured live 2026-09-21 (Stromus, clean installs on Modal Linux and this
Mac, published synapt 0.25.0): `recall_save` embeds every knowledge node and
prints "embedded for vector search", but in a store with ZERO transcript
chunks the search path never creates its embedding provider, so the vectors
the save path just wrote are unreachable — a paraphrase query returns
nothing, and "semantic search unavailable" is reported even though the DB
holds embeddings. That is the permanent state of any harness that is not
Claude Code or Codex: no transcripts, keyword-only search whatever is
installed.

The seam (core.py, TranscriptIndex init): with ``use_embeddings=True`` the
provider is only created when the store has CHUNK embeddings; a
knowledge-only store has none, so ``_embed_provider`` stays ``None`` and
``_search_knowledge``'s hybrid branch never fires.

These tests build the exact store shape a docs-only agent has: knowledge
nodes saved (and embedded) exactly as ``recall_save`` does, zero transcript
chunks, a provider present. The paraphrase query must find the node.
"""

from __future__ import annotations

import pytest

from synapt.recall.core import TranscriptIndex
from synapt.recall.storage import RecallDB

FACT_CONTENT = "The billing service uses Postgres 14 on host db-east-1."
PARAPHRASE_QUERY = "where is invoice data stored"
# Zero keyword overlap between the paraphrase query and the fact content:
# if the assertion passes, it passed through the embedding branch, not FTS.

# The DB stores embeddings at the provider's fixed width (MiniLM: 384).
_DIM = 384


def _vec(first: float, second: float) -> list[float]:
    v = [0.0] * _DIM
    v[0], v[1] = first, second
    return v


# Calibrated to the MEASURED table (isolated witness + probe
# table): the query sits at 0 degrees; the related fact at cosine 0.416;
# an UNRELATED fact at 0.293 (measured: above the 0.25
# absolute floor, 0.123 under the top, so only the RELATIVE floor drops
# it); a weaker unrelated fact at 0.20 (under the absolute floor); a
# lone witness pair at 0.307 (the original live measurement). The bands
# OVERLAP — no absolute number separates them — which is exactly what
# these tests pin.
_QUERY_VEC = _vec(1.0, 0.0)


def _at_cos(c: float) -> list[float]:
    return _vec(c, (1.0 - c * c) ** 0.5)


_FACT_VEC = _at_cos(0.416)
_UNRELATED_VEC = _at_cos(0.293)
_WEAK_VEC = _at_cos(0.20)
_LONE_WITNESS_VEC = _at_cos(0.307)
_ZERO_VEC = [0.0] * _DIM


class _FakeProvider:
    """Deterministic provider: the fact content and the paraphrase query sit
    close together in vector space; everything else maps to the zero vector."""

    dim = _DIM

    def _vec(self, text: str) -> list[float]:
        if "postgres" in text.lower():
            return _FACT_VEC
        if "invoice" in text.lower():
            return _QUERY_VEC
        if "coffee" in text.lower():
            return _UNRELATED_VEC
        if "queue" in text.lower():
            return _WEAK_VEC
        if "lone" in text.lower():
            return _LONE_WITNESS_VEC
        return _ZERO_VEC

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        return self._vec(text)


def _knowledge_only_store(tmp_path, monkeypatch, provider) -> RecallDB:
    """A store with saved, EMBEDDED knowledge nodes and zero chunks — the
    state recall_save leaves behind in any harness without transcripts."""
    db = RecallDB(tmp_path / "recall.db")
    node = {
        "id": "kn-test-invoice-001",
        "content": FACT_CONTENT,
        "category": "fact",
        "confidence": 0.9,
        "source_sessions": [],
        "created_at": "2026-09-21T00:00:00Z",
        "updated_at": "",
        "status": "active",
        "superseded_by": "",
        "contradiction_note": "",
        "tags": [],
        "version": 1,
        "lineage_id": "",
        "source_turns": [],
        "source_offsets": [],
    }
    noise = dict(node, id="kn-test-coffee-002",
                 content="Our office coffee machine gets descaled monthly.",
                 confidence=0.9)
    db.save_knowledge_nodes([node, noise])
    # Mirror the recall_save path exactly: it embeds content[:500] and writes
    # the vector keyed by the node's rowid.
    for n in (node, noise):
        rowid = db.get_knowledge_rowid(n["id"])
        assert rowid is not None
        db.save_knowledge_embeddings({rowid: provider.embed_single(n["content"][:500])})
    monkeypatch.setattr(
        "synapt.recall.embeddings.get_embedding_provider", lambda: provider
    )
    return db


def test_knowledge_only_store_searches_saved_knowledge_semantically(tmp_path, monkeypatch):
    """recall_search's semantic branch must reach saved knowledge embeddings
    in a store with zero transcript chunks."""
    provider = _FakeProvider()
    db = _knowledge_only_store(tmp_path, monkeypatch, provider)
    index = TranscriptIndex([], db=db, use_embeddings=True)

    result = index.lookup("where is invoice data stored")

    assert result, "paraphrase query must find the saved node via embeddings"
    assert "Postgres 14" in result
    assert "coffee machine" not in result, (
        "the 0.293-band unrelated node must drop to the relative floor "
        "(top 0.416 - 0.10 = 0.316); the fixture sentence shares no tokens "
        "with the query, so only the gate can exclude it"
    )
    assert index._last_knowledge_semantic_used is True


def test_knowledge_only_store_with_no_provider_stays_keyword_only(tmp_path, monkeypatch):
    """Control: without a provider, the knowledge-only store keeps today's
    keyword-only behavior (no model load, no crash, no fake hits)."""
    db = _knowledge_only_store(tmp_path, monkeypatch, _FakeProvider())
    monkeypatch.setattr(
        "synapt.recall.embeddings.get_embedding_provider", lambda: None
    )
    index = TranscriptIndex([], db=db, use_embeddings=True)

    assert index._embed_provider is None
    result = index.lookup("where is invoice data stored")
    assert "Postgres 14" not in result
    assert index._last_knowledge_semantic_used is False

# ---------------------------------------------------------------------------
# recall_quick fallback: keyword first, knowledge semantics only on a miss
# ---------------------------------------------------------------------------

def test_quick_falls_back_to_knowledge_semantics_on_keyword_miss(monkeypatch):
    """recall_quick's documented cheap keyword pass runs first; when it finds
    nothing, an embeddings-enabled retry reaches saved knowledge the keyword
    pass cannot (recall_save embeds every node regardless)."""
    from pathlib import Path

    from synapt.recall import server

    class _MissIndex:
        _embedding_status = "disabled"
        _embedding_reason = ""
        _last_diagnostics = None

        def __init__(self, result: str, semantic_used: bool) -> None:
            self._result = result
            self._semantic_used = semantic_used

        def lookup(self, query: str, **kwargs) -> str:
            self._last_knowledge_semantic_used = self._semantic_used
            return self._result

    keyword = _MissIndex("", False)
    semantic = _MissIndex(
        f"Knowledge node kn-test-invoice-001: {FACT_CONTENT}", True
    )
    calls: list[bool] = []

    def fake_get_index(use_embeddings: bool = True):
        calls.append(use_embeddings)
        return keyword if not use_embeddings else semantic

    monkeypatch.setattr(server, "_get_index", fake_get_index)
    monkeypatch.setattr(server, "project_index_dir", lambda: Path("/tmp/irrelevant"))
    monkeypatch.setattr(server, "_query_freshness_line", lambda index_dir: "")

    out = server.recall_quick(PARAPHRASE_QUERY)

    assert calls == [False, True], "keyword pass first, embeddings only on miss"
    assert "Postgres 14" in out


def test_quick_double_miss_reports_semantic_as_used(monkeypatch):
    """When both passes miss, the miss message must reflect that the semantic
    pass ran — never 'semantic search was not used' after semantics executed."""
    from pathlib import Path

    from synapt.recall import server

    class _Diag:
        total_sessions = 3
        total_chunks = 0
        oldest_indexed_at = "2026-09-01"
        semantic_search_used = True
        reason = "no_matches"

    class _MissIndex:
        _embedding_status = "active"
        _embedding_reason = ""
        lookup_kwargs = None

        def __init__(self, semantic_used: bool) -> None:
            self._semantic_used = semantic_used
            self._last_diagnostics = None

        def lookup(self, query: str, **kwargs) -> str:
            self._last_knowledge_semantic_used = self._semantic_used
            self._last_diagnostics = _Diag()
            self._last_diagnostics.semantic_search_used = self._semantic_used
            return ""

    keyword = _MissIndex.__new__(_MissIndex)
    keyword._semantic_used = False
    keyword._embedding_status = "disabled"
    keyword._embedding_reason = ""
    keyword._last_diagnostics = None
    keyword.lookup = lambda query, **kw: ""
    semantic = _MissIndex(True)

    def fake_get_index(use_embeddings: bool = True):
        return keyword if not use_embeddings else semantic

    monkeypatch.setattr(server, "_get_index", fake_get_index)
    monkeypatch.setattr(server, "project_index_dir", lambda: Path("/tmp/irrelevant"))
    monkeypatch.setattr(server, "_query_freshness_line", lambda index_dir: "")

    out = server.recall_quick(PARAPHRASE_QUERY)

    assert "semantic search was also used" in out


def test_lone_witness_pair_stands_at_the_relative_floor(tmp_path, monkeypatch):
    """A lone 0.307-band top (the original live witness pair) passes: with a
    single hit the relative floor is trivially satisfied."""
    provider = _FakeProvider()
    db = RecallDB(tmp_path / "recall.db")
    node = {
        "id": "kn-test-lone-001",
        "content": "The billing service uses Postgres 14 on host db-east-1.",
        "category": "fact", "confidence": 0.9, "source_sessions": [],
        "created_at": "2026-09-21T00:00:00Z", "updated_at": "",
        "status": "active", "superseded_by": "", "contradiction_note": "",
        "tags": [], "version": 1, "lineage_id": "", "source_turns": [],
        "source_offsets": [],
    }
    db.save_knowledge_nodes([node])
    rowid = db.get_knowledge_rowid(node["id"])
    db.save_knowledge_embeddings(
        {rowid: provider.embed_single(node["content"][:500])}
    )
    monkeypatch.setattr(
        "synapt.recall.embeddings.get_embedding_provider", lambda: provider
    )
    index = TranscriptIndex([], db=db, use_embeddings=True)

    result = index.lookup("where is invoice data stored")

    assert "Postgres 14" in result


def test_semantic_only_miss_does_not_claim_embeddings_unavailable(tmp_path, monkeypatch):
    """A miss in a knowledge-only store must not print
    'semantic search unavailable' — the provider is active and the branch
    ran; that line would tell an agent to install what it already has."""
    provider = _FakeProvider()
    db = _knowledge_only_store(tmp_path, monkeypatch, provider)
    index = TranscriptIndex([], db=db, use_embeddings=True)

    result = index.lookup("an unrelated query with no matches at all")

    assert "semantic search unavailable" not in result
    assert index._embed_provider is not None


# ---------------------------------------------------------------------------
# The grep-intercept hook's bounded call: no fallback, ever (bounded-call contract)
# ---------------------------------------------------------------------------

def test_grep_hook_miss_never_starts_the_semantic_fallback(tmp_path, monkeypatch):
    """build_pretooluse_context with a miss pattern on a knowledge-embedded
    store must never call _get_index(use_embeddings=True): the hook runs
    under a 500 ms budget and the fallback starts a model load on a miss."""
    from pathlib import Path

    from synapt.integrations import grep_intercept
    from synapt.recall import server

    provider = _FakeProvider()
    db = _knowledge_only_store(tmp_path, monkeypatch, provider)
    monkeypatch.setattr(
        "synapt.recall.embeddings.get_embedding_provider", lambda: provider
    )

    calls: list[bool] = []
    keyword_miss = _QuickIndex(lookup_result="", semantic_used=False)

    def fake_get_index(use_embeddings: bool = True):
        calls.append(use_embeddings)
        return keyword_miss

    monkeypatch.setattr(server, "_get_index", fake_get_index)
    monkeypatch.setattr(
        grep_intercept, "_load_recall_quick_impl", server._bind_quick_no_fallback
    )
    monkeypatch.setattr(server, "project_index_dir", lambda: Path("/tmp/irrelevant"))
    monkeypatch.setattr(server, "_query_freshness_line", lambda index_dir: "")

    tool_call = {"tool_name": "Grep", "tool_input": {"pattern": "coffee machine settings"}}
    out = grep_intercept.build_pretooluse_context(
        tool_call, config=grep_intercept.GrepInterceptConfig(enabled=True)
    )

    assert out is None, "a miss produces no advisory line"
    assert calls == [False], (
        f"the hook's miss must stay keyword-only; saw use_embeddings={calls}"
    )


def test_grep_hook_hit_still_returns_its_line(tmp_path, monkeypatch):
    """The hit case is unchanged: the hook's no-fallback quick returns the
    keyword hit and the advisory line builds."""
    from pathlib import Path

    from synapt.integrations import grep_intercept
    from synapt.recall import server

    class _HitIndex:
        _embedding_status = "disabled"
        _embedding_reason = ""
        _last_diagnostics = None

        def lookup(self, query: str, **kwargs) -> str:
            self._last_knowledge_semantic_used = False
            return "Past session context:\n--- [knowledge #x] fact ---\nsome fact"

    monkeypatch.setattr(
        server, "_get_index", lambda use_embeddings=True: _HitIndex()
    )
    monkeypatch.setattr(
        grep_intercept, "_load_recall_quick_impl", server._bind_quick_no_fallback
    )
    monkeypatch.setattr(server, "project_index_dir", lambda: Path("/tmp/irrelevant"))
    monkeypatch.setattr(server, "_query_freshness_line", lambda index_dir: "")

    tool_call = {"tool_name": "Grep", "tool_input": {"pattern": "some fact"}}
    out = grep_intercept.build_pretooluse_context(
        tool_call, config=grep_intercept.GrepInterceptConfig(enabled=True)
    )

    assert out is not None and "related conversations" in out


class _QuickIndex:
    """Minimal quick-index stand-in for the hook's call-sequence witness."""

    _embedding_status = "disabled"
    _embedding_reason = ""
    _last_diagnostics = None

    def __init__(self, lookup_result: str = "", semantic_used: bool = False) -> None:
        self._result = lookup_result
        self._semantic_used = semantic_used

    def lookup(self, query: str, **kwargs) -> str:
        self._last_knowledge_semantic_used = self._semantic_used
        return self._result
