"""A retraction must survive a resync: the retract path writes BOTH stores.

Reproduced before the fix (isolated store, dev's code). The originating issue is
tracked privately and deliberately not linked here; the #dev record carries it:

    recall_save(node_id=..., retract=True)
      db status    = retracted
      jsonl status = active        <- the retract branch never writes the jsonl
    _sync_knowledge_to_db()
      db status    = active        <- the sync treats the jsonl as authoritative

The retract itself is honest about what it did to SQLite ("Hidden from search,
preserved for audit") and the DB does exactly that. What is missing is the second
half: `_sync_knowledge_to_db` writes knowledge.jsonl over SQLite for every node it
carries, so the next ordinary consolidation sync puts the node back in search. The
divergence is deterministic from the moment retract returns — no crash window, no
interleaving.

Same class, same seam, already fixed once on a sibling branch of the same
function: `_apply_contest_resolution`'s docstring records that writing SQLite
alone meant the very next sync silently reverted every valid resolution, and it
was fixed by dual-writing both stores from one update dict so they cannot drift.

Contract asserted here:
- after a retract, knowledge.jsonl says retracted for that node (the create path
  already writes it, so the store is asymmetric without this);
- a resync after a retract leaves the node retracted in the DB — the retraction
  is durable, not merely present until the next background event.
"""

from unittest.mock import patch


def _jsonl_status(kn_path, node_id):
    """The status knowledge.jsonl carries for a node, or None if it has none.

    THE LAST MATCHING RECORD WINS, because the jsonl is append-only per version:
    `update_node` appends a modified copy rather than rewriting the line, and the
    reader deduplicates on read. Returning the FIRST match read the pre-retract
    version and made the fixed code look unfixed -- the same defect shape as the
    two before it in this file (one key where the record had another, one store
    where the fix writes two), and the third time the harness was wrong and the
    artifact was right.
    """
    import json

    if not kn_path.exists():
        return None
    latest = None
    for line in kn_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        # The record's identity key differs by writer: the reproduction that found
        # this checked BOTH (node_id and id) and matched; this helper checked only
        # node_id and so failed its own precondition for the wrong reason. Reading one
        # key where the artifact carries another is the same class as reading one
        # store where the fix writes two.
        if node_id in (record.get("node_id"), record.get("id")):
            latest = record.get("status")
    return latest


class TestRetractDualStore:
    def test_retract_writes_knowledge_jsonl(self, tmp_path, monkeypatch):
        from synapt.recall.core import project_data_dir
        from synapt.recall.server import recall_save

        # SYNAPT_RECALL_ROOT is how the standalone reproduction pinned the store, and
        # it is the env var the store-resolution contract names — so the test writes
        # where it reads. Patching Path.cwd alone resolved the project differently and
        # the create's jsonl landed somewhere the assertion did not look: the first
        # version of this test failed its own PRECONDITION, which is a harness defect,
        # not the finding.
        monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(tmp_path))
        with patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            recall_save(content="A fact to bury", category="fact", node_id="dual1")
            kn_path = project_data_dir() / "knowledge.jsonl"
            assert _jsonl_status(kn_path, "dual1") == "active", "precondition: create writes the jsonl"

            recall_save(node_id="dual1", retract=True)

        # THE ASSERTION THAT FAILS BEFORE THE FIX. The DB half is written by the
        # retract branch; the jsonl half is not, so the two stores disagree.
        assert _jsonl_status(kn_path, "dual1") == "retracted", (
            "retract reported success but knowledge.jsonl still says "
            f"{_jsonl_status(kn_path, 'dual1')!r} for dual1"
        )

    def test_resync_after_retract_does_not_resurrect(self, tmp_path, monkeypatch):
        from synapt.recall.consolidate import _sync_knowledge_to_db
        from synapt.recall.core import project_data_dir, project_index_dir
        from synapt.recall.server import recall_save
        from synapt.recall.sharding import live_store_path
        from synapt.recall.storage import RecallDB

        monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(tmp_path))
        with patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            recall_save(content="A fact to bury", category="fact", node_id="dual2")
            recall_save(node_id="dual2", retract=True)

        kn_path = project_data_dir() / "knowledge.jsonl"
        store = live_store_path(project_index_dir())

        db = RecallDB(store)
        try:
            assert db.get_knowledge_node("dual2")["status"] == "retracted", (
                "precondition: the DB half of the retract did happen"
            )
        finally:
            db.close()

        # The ordinary background event: consolidation's sync, jsonl authoritative.
        _sync_knowledge_to_db(None, kn_path)

        db = RecallDB(store)
        try:
            status = db.get_knowledge_node("dual2")["status"]
        finally:
            db.close()
        assert status == "retracted", (
            "a resync resurrected a retracted node: it was hidden from search and "
            f"is now {status!r} again"
        )
