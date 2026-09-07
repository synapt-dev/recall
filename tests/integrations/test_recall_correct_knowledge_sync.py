"""recall_correct's Step 3 ("Sync to DB so the node is immediately
searchable") passed project_data_dir()'s own return value -- already the
resolved DATA dir (<root>/.synapt/recall) -- into _sync_knowledge_to_db,
which internally applies project_data_dir() to its argument A SECOND TIME
(project_index_dir(project_dir) = project_data_dir(project_dir) / "index").
Doubling an already-resolved data dir produces a path that never exists,
so _sync_knowledge_to_db's own `if not db_path.exists(): return` silently
no-ops -- no exception, so the caller's try/except never fires and prints
"Synced to search index." regardless. Found by measuring every non-transcript
write's survival across a normal build; tracked privately, no number here.

Two defects, two witnesses: (1) the node never reaches the knowledge TABLE
despite the success message (the JSONL source of truth is unaffected); (2)
the same code claimed success even with no index built yet at all -- a sync
that finds no store must refuse and say so, never print a false success.
"""

from __future__ import annotations

from pathlib import Path

from synapt.recall.knowledge import read_nodes
from synapt.recall.sharding import live_store_path
from synapt.recall.storage import RecallDB


def _knowledge_row_count(index_dir: Path) -> int:
    db = RecallDB(live_store_path(index_dir))
    try:
        return db._conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
    finally:
        db.close()


def test_recall_correct_knowledge_node_reaches_the_db_table(tmp_path, monkeypatch):
    """The node recall_correct creates must land in the knowledge TABLE, not
    only knowledge.jsonl -- "Synced to search index." is a claim about the
    table, and the table is what recall_search/recall_context read from."""
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(root))
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)

    index_dir = root / ".synapt" / "recall" / "index"
    # A build has already run: the store exists, empty, before this call --
    # RecallDB's constructor creates the schema on open (no transcript
    # corpus needed for this defect; it is about path resolution, not
    # clustering or archiving).
    RecallDB(live_store_path(index_dir)).close()
    assert _knowledge_row_count(index_dir) == 0, "fixture assumption: empty store"

    from synapt.recall import server

    result = server.recall_correct(
        question="what is the widget config",
        wrong_answer="the widget config is wrong",
        correct_answer="the widget config is 42",
        category="fact",
    )
    assert "Error" not in result

    assert "Synced to search index." in result, (
        f"expected the success message given a real, pre-existing store: {result!r}"
    )
    assert _knowledge_row_count(index_dir) == 1, (
        "recall_correct reported \"Synced to search index.\" but the "
        "knowledge table row count is not 1 -- the node never reached the "
        "table it claims to have synced to"
    )

    knowledge_jsonl = root / ".synapt" / "recall" / "knowledge.jsonl"
    jsonl_nodes = read_nodes(knowledge_jsonl)
    assert len(jsonl_nodes) == 1, "the JSONL source of truth must be unaffected either way"

    db = RecallDB(live_store_path(index_dir))
    try:
        db_node = db.get_knowledge_node(jsonl_nodes[0].id)
    finally:
        db.close()
    assert db_node is not None, (
        f"node {jsonl_nodes[0].id} is in knowledge.jsonl but absent from the "
        "knowledge table"
    )
    assert db_node["content"] == jsonl_nodes[0].content


def test_recall_correct_refuses_to_claim_sync_when_no_store_exists(tmp_path, monkeypatch):
    """No recall index has ever been built for this project (no RecallDB
    opened, no build run) -- the sync has nothing to write into. It must
    say so honestly, never claim "Synced to search index." over a no-op."""
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(root))
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)

    index_dir = root / ".synapt" / "recall" / "index"
    db_path = live_store_path(index_dir)
    assert not db_path.exists(), "fixture assumption: no store built yet"

    from synapt.recall import server

    result = server.recall_correct(
        question="what is the widget config",
        wrong_answer="the widget config is wrong",
        correct_answer="the widget config is 42",
        category="fact",
    )
    assert "Error" not in result

    assert "Synced to search index." not in result, (
        f"claimed a sync succeeded with no store to sync into: {result!r}"
    )
    assert "Will sync on next consolidation." in result, (
        f"expected the honest fallback message, got: {result!r}"
    )

    # The JSONL side must still work -- only the DB-table sync refuses.
    knowledge_jsonl = root / ".synapt" / "recall" / "knowledge.jsonl"
    jsonl_nodes = read_nodes(knowledge_jsonl)
    assert len(jsonl_nodes) == 1
