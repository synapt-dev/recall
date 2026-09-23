"""Supersession must survive a resync, and the ordering key must not be a clock.

Reproduced before the fix (isolated store, dev's code):

    _apply_supersession(db, old_node_id=..., new_content=...)
      db old   = 'contradicted'     jsonl old = 'active'   <- jsonl untouched
    _sync_knowledge_to_db()          (jsonl is authoritative)
      db old   = 'active'           <- RESURRECTED

Same dual-store class as the retract branch (fixed in recall PR #1215); the
private issue that filed this one is deliberately not linked here.
and the contest path. ``_apply_supersession`` writes SQLite for BOTH the old node
and its replacement, and never the jsonl, so the ordinary background sync reverts
the supersession for every node it supersedes.

Contract asserted here:
- after a supersession, BOTH stores say contradicted for the old node, and the
  replacement node is in the jsonl too (it was db-only as well);
- a resync after a supersession leaves the old node contradicted — durable, not
  merely present until the next background event;
- ``_dedup_nodes`` orders by the node's monotone ``version``, not by wall-clock
  ``updated_at``: a record stamped in the FUTURE at a lower version must not win
  (probed on the retract read), because a clock cannot be
  the ordering authority for a store other tools write;
- superseding a db-only node creates its jsonl record instead of silently
  no-opping on ``update_node``'s False return.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch


def _jsonl_latest(kn_path, node_id):
    """The LAST record for a node: the jsonl is append-only per version."""
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
        if node_id in (record.get("node_id"), record.get("id")):
            latest = record
    return latest


def _db_node(store_path, node_id):
    from synapt.recall.storage import RecallDB

    db = RecallDB(store_path)
    try:
        return db.get_knowledge_node(node_id)
    finally:
        db.close()


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
    import hashlib

    return hashlib.sha1(content.encode()).hexdigest()[:12]


class TestSupersessionDualStore:
    def test_supersession_writes_the_old_node_to_the_jsonl(
        self, tmp_path, monkeypatch
    ):
        from synapt.recall.core import project_index_dir
        from synapt.recall.sharding import live_store_path
        from synapt.recall.storage import RecallDB
        from synapt.recall import server

        kn_path, _ = _isolate(tmp_path, monkeypatch)
        old_id = _save("supersession probe: the fact that gets superseded")
        assert _jsonl_latest(kn_path, old_id)["status"] == "active", "precondition"

        db = RecallDB(live_store_path(project_index_dir()))
        try:
            server._apply_supersession(
                db,
                old_node_id=old_id,
                new_content="supersession probe: the replacement fact",
                category="fact",
                reason="probe",
                source_sessions=[],
            )
        finally:
            db.close()

        # FAILS BEFORE THE FIX: the jsonl still says active.
        assert _jsonl_latest(kn_path, old_id)["status"] == "contradicted", (
            "supersession reported success but knowledge.jsonl still says "
            f"{_jsonl_latest(kn_path, old_id)['status']!r} for the superseded node"
        )

    def test_supersession_writes_the_replacement_node_to_the_jsonl(
        self, tmp_path, monkeypatch
    ):
        from synapt.recall.core import project_index_dir
        from synapt.recall.sharding import live_store_path
        from synapt.recall.storage import RecallDB
        from synapt.recall import server

        kn_path, _ = _isolate(tmp_path, monkeypatch)
        old_id = _save("supersession probe 2: the old fact")
        before = len(kn_path.read_text().splitlines()) if kn_path.exists() else 0

        db = RecallDB(live_store_path(project_index_dir()))
        try:
            server._apply_supersession(
                db,
                old_node_id=old_id,
                new_content="supersession probe 2: the replacement fact",
                category="fact",
                reason="probe",
                source_sessions=[],
            )
        finally:
            db.close()

        records = [
            json.loads(line)
            for line in kn_path.read_text().splitlines()
            if line.strip()
        ]
        # The replacement is a NEW node the jsonl never carried: it must be
        # appended, not merely upserted into SQLite.
        assert len(records) > before, "the replacement node never reached the jsonl"
        contents = {r.get("content") for r in records}
        assert "supersession probe 2: the replacement fact" in contents

    def test_resync_after_supersession_does_not_resurrect(self, tmp_path, monkeypatch):
        from synapt.recall.core import project_index_dir
        from synapt.recall.consolidate import _sync_knowledge_to_db
        from synapt.recall.sharding import live_store_path
        from synapt.recall.storage import RecallDB
        from synapt.recall import server

        kn_path, store = _isolate(tmp_path, monkeypatch)
        old_id = _save("supersession probe 3: the old fact")

        db = RecallDB(store)
        try:
            server._apply_supersession(
                db,
                old_node_id=old_id,
                new_content="supersession probe 3: the replacement fact",
                category="fact",
                reason="probe",
                source_sessions=[],
            )
        finally:
            db.close()
        assert _db_node(store, old_id)["status"] == "contradicted", "precondition"

        _sync_knowledge_to_db(None, kn_path)

        status = _db_node(store, old_id)["status"]
        assert status == "contradicted", (
            "a resync resurrected a superseded node: it was marked contradicted "
            f"and is now {status!r} again"
        )


class TestDedupOrdersByRevisionNotClock:
    def test_a_future_stamped_lower_revision_record_does_not_win(self):
        """Probed on the retract read: a record stamped
        in the future — clock skew, a migrated or synthesized record, an eval
        adapter import — beat the newer transition because dedup compared
        wall-clock. Ordering is the node's monotone version; position breaks ties.
        """
        from synapt.recall.knowledge import KnowledgeNode, _dedup_nodes

        now = datetime.now(timezone.utc)
        transition = KnowledgeNode.from_dict(
            {
                "id": "dup1",
                "content": "the retracted fact",
                "category": "fact",
                "confidence": 0.5,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "status": "retracted",
                "revision": 1,
                "version": 1,
            }
        )
        future = KnowledgeNode.from_dict(
            {
                "id": "dup1",
                "content": "the retracted fact",
                "category": "fact",
                "confidence": 0.5,
                "created_at": now.isoformat(),
                "updated_at": (now + timedelta(minutes=5)).isoformat(),
                "status": "active",
                "revision": 0,
                "version": 9,
            }
        )

        # The future-stamped record comes FIRST, so a position-only rule would
        # also pass; the version rule is what must decide.
        deduped = _dedup_nodes([future, transition])
        assert len(deduped) == 1
        assert deduped[0].status == "retracted", (
            "a record stamped five minutes in the future, at a LOWER version, "
            f"won the dedup: status is {deduped[0].status!r}"
        )

        # Control: the same two records in the other order must give the same
        # answer, so the result is a property of the records and not of order.
        assert _dedup_nodes([transition, future])[0].status == "retracted"

    def test_equal_revisions_still_resolve_last_in_file(self):
        """The tie-break the old `>=` provided must survive: on equal versions
        the last appended record wins, which is what makes append-only updates
        readable."""
        from synapt.recall.knowledge import KnowledgeNode, _dedup_nodes

        now = datetime.now(timezone.utc).isoformat()
        first = KnowledgeNode.from_dict(
            {"id": "dup2", "content": "x", "category": "fact",
                "confidence": 0.5,
             "created_at": now, "updated_at": now, "status": "active", "revision": 1}
        )
        second = KnowledgeNode.from_dict(
            {"id": "dup2", "content": "x", "category": "fact",
                "confidence": 0.5,
             "created_at": now, "updated_at": now, "status": "retracted", "revision": 1}
        )
        assert _dedup_nodes([first, second])[0].status == "retracted"

    def test_legacy_records_without_a_revision_field_still_dedup(self):
        """Migration: a record written before `version` existed must not be lost
        or win unconditionally. Missing version reads as the model default (1),
        so legacy files behave exactly as they did."""
        from synapt.recall.knowledge import KnowledgeNode, _dedup_nodes

        now = datetime.now(timezone.utc).isoformat()
        legacy = KnowledgeNode.from_dict(
            {"id": "dup3", "content": "x", "category": "fact",
                "confidence": 0.5,
             "created_at": now, "updated_at": now, "status": "active"}
        )
        newer = KnowledgeNode.from_dict(
            {"id": "dup3", "content": "x", "category": "fact",
                "confidence": 0.5,
             "created_at": now, "updated_at": now, "status": "retracted", "revision": 2}
        )
        assert legacy.revision == 0, "the model default is the migration story"
        assert _dedup_nodes([legacy, newer])[0].status == "retracted"
        assert _dedup_nodes([newer, legacy])[0].status == "retracted"


class TestSupersedingADbOnlyNode:
    def test_a_db_only_node_gets_a_jsonl_record_instead_of_a_silent_no_op(
        self, tmp_path, monkeypatch
    ):
        """`update_node` returns False for a node absent from the jsonl, and the
        supersession branch is a place that return is currently ignored. A node
        created db-only (`_create_knowledge_from_claim`) can be superseded; the
        jsonl must then carry the transition rather than silently skipping it.
        """
        from synapt.recall.consolidate import _sync_knowledge_to_db
        from synapt.recall.core import project_index_dir
        from synapt.recall.knowledge import KnowledgeNode, _knowledge_path
        from synapt.recall.sharding import live_store_path
        from synapt.recall.storage import RecallDB
        from synapt.recall import server

        kn_path, store = _isolate(tmp_path, monkeypatch)

        # A db-only node, exactly as _create_knowledge_from_claim leaves it.
        node = KnowledgeNode.create("db-only probe: a node the jsonl never saw", "fact")
        db = RecallDB(store)
        try:
            db.upsert_knowledge_node(node.to_dict())
        finally:
            db.close()
        assert _jsonl_latest(kn_path, node.id) is None, "precondition: jsonl-only absent"

        db = RecallDB(store)
        try:
            server._apply_supersession(
                db,
                old_node_id=node.id,
                new_content="db-only probe: the replacement",
                category="fact",
                reason="probe",
                source_sessions=[],
            )
        finally:
            db.close()

        assert _jsonl_latest(kn_path, node.id) is not None, (
            "superseding a db-only node left no jsonl record at all: the "
            "transition happened in SQLite only and the next sync decides by "
            "the jsonl, where the node does not exist"
        )
        _sync_knowledge_to_db(None, kn_path)
        assert _db_node(store, node.id)["status"] == "contradicted"
