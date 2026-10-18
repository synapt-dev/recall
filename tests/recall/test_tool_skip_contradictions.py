"""The recall_search and recall_quick MCP tools take
``detect_contradictions`` and the ``synapt search`` CLI takes
``--no-detect-contradictions``; both reach lookup, so a background search
through the tool or the CLI queues no pending contradictions. The default is
unchanged: without the parameter or the flag, detection runs.

Fixture: the floor-clearing pair from test_co_retrieval_floor.py, which
queues exactly one row on a detecting search.
"""

import sys
from unittest.mock import patch

import pytest

from synapt.recall import server
from synapt.recall.core import project_index_dir

from tests.recall.test_co_retrieval_floor import (
    VEC_A,
    VEC_B,
    _fake_provider_with_db_embeddings,
)
from tests.recall.test_lookup_skip_contradictions import _pending

# Taken at import, before any row patches it to a no-op.
_REAL_INVALIDATE_CACHE = server._invalidate_cache


@pytest.fixture(autouse=True)
def _drop_cached_index():
    """Each row patches _invalidate_cache away, so the server keeps the
    row's tmp index cached; drop it so the next file starts clean."""
    yield
    _REAL_INVALIDATE_CACHE()


def _seed(tmp_path):
    """Save the floor-clearing pair and return the patches to keep open."""
    from synapt.recall.server import recall_save

    provider = _fake_provider_with_db_embeddings({
        "Postgres": VEC_A, "Sixteen": VEC_B,
    })
    patches = (
        patch("synapt.recall.server.Path.cwd", return_value=tmp_path),
        patch("synapt.recall.server.get_embedding_provider", return_value=provider),
        patch("synapt.recall.embeddings.get_embedding_provider", return_value=provider),
        patch("synapt.recall.server._invalidate_cache"),
    )
    for p in patches:
        p.start()
    recall_save(content="The gateway uses Postgres 14 on port 5433", category="fact", node_id="g1")
    recall_save(content="The gateway migrated to Postgres Sixteen last week", category="fact", node_id="g2")
    return patches


def _run(tmp_path, call):
    patches = _seed(tmp_path)
    try:
        call()
    finally:
        for p in patches:
            p.stop()
    return _pending(tmp_path)


def _cli(tmp_path, *extra):
    from synapt.recall.cli import main

    argv = ["synapt", "search", "gateway Postgres",
            "--index", str(project_index_dir(tmp_path)), *extra]
    with patch.object(sys, "argv", argv):
        main()


class TestRecallSearchTool:
    def test_flag_false_queues_no_rows(self, tmp_path):
        from synapt.recall.server import recall_search

        rows = _run(tmp_path, lambda: recall_search(
            "gateway Postgres", detect_contradictions=False))
        assert rows == []

    def test_default_queues_rows(self, tmp_path):
        from synapt.recall.server import recall_search

        rows = _run(tmp_path, lambda: recall_search("gateway Postgres"))
        assert len(rows) >= 1


class TestRecallQuickTool:
    def test_flag_false_queues_no_rows(self, tmp_path):
        from synapt.recall.server import recall_quick

        rows = _run(tmp_path, lambda: recall_quick(
            "gateway Postgres", detect_contradictions=False))
        assert rows == []

    def test_default_queues_rows(self, tmp_path):
        from synapt.recall.server import recall_quick

        rows = _run(tmp_path, lambda: recall_quick("gateway Postgres"))
        assert len(rows) >= 1


def _quick_via_fallback(tmp_path, **kwargs):
    """Run recall_quick with the keyword pass forced to miss, so the
    lookup that answers is the semantic fallback's."""
    from synapt.recall.core import TranscriptIndex
    from synapt.recall.server import recall_quick

    def fake_get_index(use_embeddings=True):
        index = TranscriptIndex.load(
            project_index_dir(tmp_path), use_embeddings=use_embeddings)
        if not use_embeddings:
            index.lookup = lambda *a, **k: ""
        return index

    def call():
        with patch("synapt.recall.server._get_index", side_effect=fake_get_index):
            recall_quick("gateway Postgres", **kwargs)

    return _run(tmp_path, call)


class TestRecallQuickSemanticFallback:
    def test_flag_false_queues_no_rows(self, tmp_path):
        rows = _quick_via_fallback(tmp_path, detect_contradictions=False)
        assert rows == []

    def test_default_queues_rows(self, tmp_path):
        rows = _quick_via_fallback(tmp_path)
        assert len(rows) >= 1


class TestSearchCli:
    def test_no_detect_flag_queues_no_rows(self, tmp_path):
        rows = _run(tmp_path, lambda: _cli(tmp_path, "--no-detect-contradictions"))
        assert rows == []

    def test_default_queues_rows(self, tmp_path):
        rows = _run(tmp_path, lambda: _cli(tmp_path))
        assert len(rows) >= 1
