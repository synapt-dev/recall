"""lookup(detect_contradictions=False) runs the search without queueing
co-retrieval contradictions.

Background callers run many searches nobody asked about; each one used to
queue pending contradictions for user review. The flag lets them skip the
detector. The flag is part of the query-cache key, so a skipped search never
serves a cached result to a later search that should detect.

Fixture: the floor-clearing pair from test_co_retrieval_floor.py (cosine
about 0.95, above the 0.40 floor), which queues exactly one row.
"""

import inspect
from unittest.mock import patch

from synapt.recall.core import TranscriptIndex, project_index_dir
from synapt.recall.storage import RecallDB

from tests.recall.test_co_retrieval_floor import (
    VEC_A,
    VEC_B,
    _fake_provider_with_db_embeddings,
)


def _pending(tmp_path):
    db = RecallDB(project_index_dir(tmp_path) / "recall.db")
    try:
        return db.list_pending_contradictions()
    finally:
        db.close()


def _seed(tmp_path):
    """Save the floor-clearing pair and return the patches to keep open."""
    from synapt.recall.server import recall_save

    provider = _fake_provider_with_db_embeddings({
        "Postgres": VEC_A, "Sixteen": VEC_B,
    })
    patches = (
        patch("synapt.recall.server.Path.cwd", return_value=tmp_path),
        patch("synapt.recall.server.get_embedding_provider", return_value=provider),
        patch("synapt.recall.server._invalidate_cache"),
    )
    for p in patches:
        p.start()
    recall_save(content="The gateway uses Postgres 14 on port 5433", category="fact", node_id="g1")
    recall_save(content="The gateway migrated to Postgres Sixteen last week", category="fact", node_id="g2")
    return patches


def _index():
    from synapt.recall.server import _get_index

    index = _get_index()
    assert index is not None
    return index


class TestLookupSkipContradictions:
    def test_flag_false_queues_no_rows(self, tmp_path):
        patches = _seed(tmp_path)
        try:
            _index().lookup("gateway Postgres", detect_contradictions=False)
        finally:
            for p in patches:
                p.stop()
        assert _pending(tmp_path) == []

    def test_control_without_flag_queues_rows(self, tmp_path):
        patches = _seed(tmp_path)
        try:
            _index().lookup("gateway Postgres")
        finally:
            for p in patches:
                p.stop()
        assert len(_pending(tmp_path)) >= 1

    def test_flag_false_then_default_same_query_still_detects(self, tmp_path):
        """The flag is in the query-cache key: the default search after a
        skipped one is not served the skipped search's cached result."""
        patches = _seed(tmp_path)
        try:
            index = _index()
            index.lookup("gateway Postgres", detect_contradictions=False)
            assert _pending(tmp_path) == []
            index.lookup("gateway Postgres")
        finally:
            for p in patches:
                p.stop()
        assert len(_pending(tmp_path)) >= 1

    def test_existing_caller_default_unchanged(self, tmp_path):
        from synapt.recall.server import recall_search

        assert inspect.signature(TranscriptIndex.lookup).parameters[
            "detect_contradictions"].default is True
        patches = _seed(tmp_path)
        try:
            result = recall_search("gateway Postgres")
        finally:
            for p in patches:
                p.stop()
        assert "Conflicting information detected" in result, result
        assert len(_pending(tmp_path)) == 1
