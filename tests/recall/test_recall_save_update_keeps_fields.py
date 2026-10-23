"""An update through recall_save that omits tags, sources or confidence keeps
the existing node's values, as an omitted category already does. Before the
fix the update rebuilt the node from its arguments alone: tags and sources
were wiped to [] and confidence reset to the 0.8 signature default."""

from __future__ import annotations

from pathlib import Path

import pytest

from synapt.recall.knowledge import read_nodes
from synapt.recall.storage import RecallDB


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(tmp_path))
    monkeypatch.setenv("SYNAPT_RECALL_WORKTREE", "pytest-owned")
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    return tmp_path


def _db_node(root: Path, node_id: str) -> dict:
    # Read back through the same resolution recall_save writes through.
    from synapt.recall import server
    from synapt.recall.sharding import live_store_path

    db = RecallDB(live_store_path(server.project_index_dir(None)))
    try:
        node = db.get_knowledge_node(node_id)
    finally:
        db.close()
    assert node is not None
    return node


def _save(**kw) -> str:
    from synapt.recall import server

    result = server.recall_save(**kw)
    assert "Error" not in result, result
    return result


def _seed(node_id: str = "n931") -> str:
    _save(
        content="first fact",
        category="decision",
        tags=["a", "b"],
        source_sessions=["s1"],
        source_turns=["s1:3"],
        confidence=0.95,
        node_id=node_id,
    )
    return node_id


def test_r1_update_with_no_tags_keeps_the_tags(store):
    nid = _seed()
    _save(content="revised fact", node_id=nid)
    assert sorted(_db_node(store, nid)["tags"]) == ["a", "b"]


def test_r2_update_with_no_sources_keeps_the_sources(store):
    nid = _seed()
    _save(content="revised fact", node_id=nid)
    node = _db_node(store, nid)
    assert node["source_sessions"] == ["s1"]
    assert node["source_turns"] == ["s1:3"]


def test_r3_update_with_no_confidence_keeps_the_confidence(store):
    nid = _seed()
    _save(content="revised fact", node_id=nid, tags=["a", "b"])
    assert _db_node(store, nid)["confidence"] == pytest.approx(0.95)


def test_r4_explicit_empty_tags_still_clear(store):
    nid = _seed()
    _save(content="revised fact", node_id=nid, tags=[])
    assert _db_node(store, nid)["tags"] == []


def test_r5_create_without_fields_gets_empty_and_default_confidence(store):
    _save(content="new fact", node_id="fresh931")
    node = _db_node(store, "fresh931")
    assert node["tags"] == []
    assert node["source_sessions"] == []
    assert node["confidence"] == pytest.approx(0.8)


def test_r6_knowledge_jsonl_keeps_the_tags_after_update(store):
    nid = _seed()
    _save(content="revised fact", node_id=nid)
    nodes = [n for n in read_nodes(next(store.rglob("knowledge*.jsonl"))) if n.id == nid]
    assert nodes[-1].content == "revised fact"
    assert sorted(nodes[-1].tags) == ["a", "b"]
    assert nodes[-1].confidence == pytest.approx(0.95)
