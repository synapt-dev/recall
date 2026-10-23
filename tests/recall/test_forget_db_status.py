"""recall_contradict(action="forget") archives the node in the knowledge DB, not only the jsonl.

Search and quick read knowledge from the DB, so a forget that only rewrote
knowledge.jsonl left the forgotten node active there and still being served.
The knowledge DB is index.db for a sharded store and recall.db for a
monolithic one; every row runs both layouts.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(params=["monolithic", "sharded"])
def store(request, owned_recall_root, monkeypatch):
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    from synapt.recall import server
    from synapt.recall.core import project_index_dir
    from synapt.recall.server import recall_save
    from synapt.recall.sharding import is_sharded
    from synapt.recall.storage import RecallDB

    if request.param == "sharded":
        index_dir = project_index_dir()
        index_dir.mkdir(parents=True, exist_ok=True)
        RecallDB(index_dir / "index.db").close()
        RecallDB(index_dir / "data_001.db").close()

    with patch("synapt.recall.server.get_embedding_provider", return_value=None):
        recall_save(
            content="The kestrel deploy runs on quartzline hosts",
            category="fact",
            node_id="kx1",
        )
    server._invalidate_cache()
    if request.param == "sharded":
        assert is_sharded(project_index_dir())
        assert not (project_index_dir() / "recall.db").exists()
    yield request.param
    server._invalidate_cache()


def _knowledge_file(layout):
    return "index.db" if layout == "sharded" else "recall.db"


def _db_status(node_id):
    """Status through the reader the server uses (layout auto-detected)."""
    from synapt.recall.core import project_index_dir
    from synapt.recall.sharded_db import ShardedRecallDB

    db = ShardedRecallDB.open(project_index_dir())
    try:
        node = db.get_knowledge_node(node_id)
        return node["status"] if node else None
    finally:
        db.close()


def _file_status(name, node_id):
    """Status read straight from one DB file of the store."""
    from synapt.recall.core import project_index_dir
    from synapt.recall.storage import RecallDB

    db = RecallDB(project_index_dir() / name)
    try:
        node = db.get_knowledge_node(node_id)
        return node["status"] if node else None
    finally:
        db.close()


def _jsonl_status(node_id):
    from synapt.recall.knowledge import _knowledge_path, read_nodes

    found = [n for n in read_nodes(_knowledge_path()) if n.id == node_id]
    return found[-1].status if found else None


def test_fixture_starts_active_in_both_stores(store):
    assert _db_status("kx1") == "active"
    assert _file_status(_knowledge_file(store), "kx1") == "active"
    assert _jsonl_status("kx1") == "active"


def test_forget_archives_db_status(store):
    from synapt.recall.server import recall_contradict

    res = recall_contradict(action="forget", old_node_id="kx1")
    assert "Forgotten" in res, res
    assert _jsonl_status("kx1") == "archived"
    assert _db_status("kx1") == "archived"


def test_forget_archives_knowledge_file(store):
    from synapt.recall.server import recall_contradict

    recall_contradict(action="forget", old_node_id="kx1")
    assert _file_status(_knowledge_file(store), "kx1") == "archived"


def test_forget_stops_quick_serving_node(store):
    from synapt.recall.server import recall_contradict, recall_quick

    with patch("synapt.recall.server.get_embedding_provider", return_value=None):
        before = recall_quick("kestrel quartzline", detect_contradictions=False)
        assert "runs on quartzline hosts" in before, before
        recall_contradict(action="forget", old_node_id="kx1")
        after = recall_quick("kestrel quartzline", detect_contradictions=False)
    assert "runs on quartzline hosts" not in after, after


def test_control_correct_changes_db_status(store):
    from synapt.recall.server import recall_contradict

    with patch("synapt.recall.server.get_embedding_provider", return_value=None):
        res = recall_contradict(
            action="correct", old_node_id="kx1",
            new_content="The kestrel deploy runs on basalt hosts",
        )
    assert "Corrected" in res, res
    assert _db_status("kx1") == "contradicted"
