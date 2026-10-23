"""recall_contradict(action="correct", successor_node_id=...) names an EXISTING successor.

The text path mints a replacement node; with successor_node_id no node is minted,
the old node is marked contradicted with superseded_by the successor, neither
lineage_id moves, and the successor is left exactly as it was. Every refusal
runs before any write.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _isolate(tmp_path, monkeypatch):
    from synapt.recall.core import project_data_dir, project_index_dir
    from synapt.recall.sharding import live_store_path

    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(tmp_path))
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    return project_data_dir() / "knowledge.jsonl", live_store_path(project_index_dir())


def _save(content, category="fact"):
    from synapt.recall.server import recall_save

    with patch("synapt.recall.server.get_embedding_provider", return_value=None), \
         patch("synapt.recall.server._invalidate_cache"):
        recall_save(content=content, category=category)
    return hashlib.sha1(content.encode()).hexdigest()[:12]


@pytest.fixture
def env(tmp_path, monkeypatch):
    from synapt.recall.storage import RecallDB

    kn_path, store = _isolate(tmp_path, monkeypatch)
    old_id = _save("successor probe: the old fact")
    succ_id = _save("successor probe: the newer fact that already exists")
    db = RecallDB(store)

    def call(**kw):
        with patch("synapt.recall.server._get_index", return_value=SimpleNamespace(_db=db)), \
             patch("synapt.recall.server._invalidate_cache"):
            from synapt.recall.server import recall_contradict
            return recall_contradict(**kw)

    yield SimpleNamespace(kn_path=kn_path, store=store, db=db, old=old_id, succ=succ_id, call=call)
    db.close()


def _db_ids(db):
    return {r[0] for r in db._conn.execute("SELECT id FROM knowledge").fetchall()}


def _jsonl_ids(kn_path):
    ids = set()
    for line in kn_path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            ids.add(rec.get("id") or rec.get("node_id"))
    return ids


def _correct(e, **kw):
    return e.call(action="correct", old_node_id=e.old, successor_node_id=e.succ, **kw)


def _assert_old_untouched(e):
    old = e.db.get_knowledge_node(e.old)
    assert old["status"] == "active"
    assert old.get("superseded_by", "") == ""


def test_r1_correct_with_successor_mints_nothing(env):
    db_before, jl_before = _db_ids(env.db), _jsonl_ids(env.kn_path)
    out = _correct(env)
    assert _db_ids(env.db) == db_before
    assert _jsonl_ids(env.kn_path) == jl_before
    assert out == f"Corrected: '{env.old}' superseded by existing node '{env.succ}'."


def test_r2_old_node_superseded_by_the_successor(env):
    _correct(env)
    old = env.db.get_knowledge_node(env.old)
    assert old["status"] == "contradicted"
    assert old["superseded_by"] == env.succ
    assert old["contradiction_note"] == "Manual correction"


def test_r3_successor_untouched(env):
    before = env.db.get_knowledge_node(env.succ)
    _correct(env)
    after = env.db.get_knowledge_node(env.succ)
    for key in ("content", "status", "version", "lineage_id", "superseded_by"):
        assert after.get(key) == before.get(key), key


def _lineage_ids(db, node_id):
    lid = db.get_knowledge_node(node_id).get("lineage_id", "")
    return [n["id"] for n in db.knowledge_lineage(lid)]


def test_r_lineage_neither_lineage_moves(env):
    # Both nodes carry a lineage of their own, so a write of None or of the other's
    # lineage to either one changes a value this row can see.
    for node_id, lid in ((env.old, env.old), (env.succ, "lineage-of-successor")):
        node = env.db.get_knowledge_node(node_id)
        node["lineage_id"] = lid
        env.db.upsert_knowledge_node(node)
    old_before = env.db.get_knowledge_node(env.old)["lineage_id"]
    succ_before = env.db.get_knowledge_node(env.succ)["lineage_id"]
    walk_old, walk_succ = _lineage_ids(env.db, env.old), _lineage_ids(env.db, env.succ)
    _correct(env)
    assert env.db.get_knowledge_node(env.old)["lineage_id"] == old_before
    assert env.db.get_knowledge_node(env.succ)["lineage_id"] == succ_before
    assert _lineage_ids(env.db, env.old) == walk_old
    assert _lineage_ids(env.db, env.succ) == walk_succ


def test_r12_non_active_old_node_refused(env):
    node = env.db.get_knowledge_node(env.old)
    env.db.upsert_knowledge_node({**node, "status": "contradicted", "superseded_by": "first0succ00"})
    before = _db_ids(env.db)
    out = _correct(env)
    assert out == f"old node {env.old} is contradicted; correct an active node"
    after = env.db.get_knowledge_node(env.old)
    assert after["status"] == "contradicted"
    assert after["superseded_by"] == "first0succ00"
    assert _db_ids(env.db) == before


def test_r13_text_path_on_non_active_old_node_as_dev(env):
    node = env.db.get_knowledge_node(env.old)
    env.db.upsert_knowledge_node({**node, "status": "contradicted", "superseded_by": "first0succ00"})
    before = _db_ids(env.db)
    out = env.call(action="correct", old_node_id=env.old, new_content="successor probe: text on contradicted")
    minted = _db_ids(env.db) - before
    after = env.db.get_knowledge_node(env.old)
    # As at dev: the text path still mints one node and repoints superseded_by.
    assert out.startswith(f"Corrected: '{env.old}' superseded with:")
    assert len(minted) == 1
    assert after["status"] == "contradicted"
    assert after["superseded_by"] == minted.pop()


def test_r4_survives_a_sync(env):
    from synapt.recall.consolidate import _sync_knowledge_to_db

    _correct(env)
    env.db.close()
    _sync_knowledge_to_db(None, env.kn_path)
    from synapt.recall.storage import RecallDB

    env.db = RecallDB(env.store)
    old = env.db.get_knowledge_node(env.old)
    assert old["status"] == "contradicted"
    assert old["superseded_by"] == env.succ


def test_r5_missing_successor_refused(env):
    out = env.call(action="correct", old_node_id=env.old, successor_node_id="nosuchnode00")
    _assert_old_untouched(env)
    assert out == "Error: successor node 'nosuchnode00' not found."


@pytest.mark.parametrize("status", ["contradicted", "retracted"])
def test_r6_r7_inactive_successor_refused(env, status):
    succ = env.db.get_knowledge_node(env.succ)
    env.db.upsert_knowledge_node({**succ, "status": status})
    out = _correct(env)
    _assert_old_untouched(env)
    assert out == f"Error: successor node '{env.succ}' is {status}; name an active node."


def test_r8_self_successor_refused(env):
    out = env.call(action="correct", old_node_id=env.old, successor_node_id=env.old)
    _assert_old_untouched(env)
    assert out == f"Error: a node cannot supersede itself ('{env.old}')."


def test_r9_successor_and_new_content_refused(env):
    db_before = _db_ids(env.db)
    out = _correct(env, new_content="some text")
    _assert_old_untouched(env)
    assert _db_ids(env.db) == db_before
    assert out == "Error: give successor_node_id OR claim/new_content for 'correct', not both."


def test_r10_flag_with_successor_refused(env):
    count = lambda: env.db._conn.execute("SELECT COUNT(*) FROM pending_contradictions").fetchone()[0]
    before = count()
    out = env.call(action="flag", old_node_id=env.old, successor_node_id=env.succ, claim="x")
    assert count() == before
    _assert_old_untouched(env)
    assert out.startswith("Error: successor_node_id is taken by 'correct', not 'flag'.")


def test_r11_text_path_still_mints_one_node(env):
    before = _db_ids(env.db)
    env.call(action="correct", old_node_id=env.old, new_content="successor probe: minted text")
    minted = _db_ids(env.db) - before
    assert len(minted) == 1
    assert env.db.get_knowledge_node(env.old)["superseded_by"] == minted.pop()
