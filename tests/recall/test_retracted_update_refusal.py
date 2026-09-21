"""An update must not silently un-retract a retracted node.

Cold-install finding (2026-09-21, first-time-agent run on the published package): recall_save
with a node_id whose node is retracted silently un-retracts it — the update
path never checks status, so an agent that thinks it is editing history
revives a fact another agent deliberately buried.

Contract after the fix:
- update on a retracted node is REFUSED with one sentence that names the node
  and the deliberate-restore path (no existing restore path existed, so the
  refusal names the new explicit one);
- the deliberate restore is recall_save(node_id=..., restore_retracted=True),
  which says "re-activated" and returns the node to search;
- the retracted node stays retracted in both halves of the store until an
  explicit restore.
"""

from unittest.mock import patch


class TestUpdateOnRetractedNode:
    def test_update_on_retracted_node_refuses(self, tmp_path):
        from synapt.recall.core import project_index_dir
        from synapt.recall.server import recall_save
        from synapt.recall.storage import RecallDB

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            recall_save(content="Wrong fact", category="fact", node_id="dead1")
            recall_save(node_id="dead1", retract=True)
            result = recall_save(
                content="Revived without warning", category="fact", node_id="dead1"
            )

        # The refusal names the node, the state, and the deliberate path.
        assert result.startswith("Error"), result
        assert "dead1" in result
        assert "retracted" in result
        assert "restore_retracted" in result

        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            node = db.get_knowledge_node("dead1")
            assert node["status"] == "retracted"
            # The new content must NOT have replaced the retracted text.
            assert node["content"] == "Wrong fact"
            assert node["version"] == 1
        finally:
            db.close()

    def test_deliberate_restore_reactivates(self, tmp_path):
        from synapt.recall.core import project_index_dir
        from synapt.recall.server import recall_save
        from synapt.recall.storage import RecallDB

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            recall_save(content="Wrong fact", category="fact", node_id="rev1")
            recall_save(node_id="rev1", retract=True)
            result = recall_save(
                content="The corrected fact",
                category="fact",
                node_id="rev1",
                restore_retracted=True,
            )

        assert "re-activated" in result.lower(), result

        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            node = db.get_knowledge_node("rev1")
            assert node["status"] == "active"
            assert node["content"] == "The corrected fact"
            assert node["version"] == 2
            active = db.load_knowledge_nodes(status="active")
            assert any(n["id"] == "rev1" for n in active)
        finally:
            db.close()

    def test_retracted_node_stays_hidden_without_restore(self, tmp_path):
        from synapt.recall.core import project_index_dir
        from synapt.recall.server import recall_save, recall_search
        from synapt.recall.storage import RecallDB

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            recall_save(content="Buried fact about zeta", category="fact", node_id="bur1")
            recall_save(node_id="bur1", retract=True)
            recall_save(
                content="Attempted silent revival of zeta", category="fact", node_id="bur1"
            )

        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            active = db.load_knowledge_nodes(status="active")
            assert all(n["id"] != "bur1" for n in active)
        finally:
            db.close()

class TestRestoreSentenceCallShape:
    def test_sentence_call_shape_restores_with_category_intact(self, tmp_path):
        """R2 A4: the refusal sentence names the exact call; following that
        call shape verbatim — node_id, content, category,
        restore_retracted=true — must re-activate with the category intact,
        not re-file as workflow."""
        from synapt.recall.core import project_index_dir
        from synapt.recall.server import recall_save
        from synapt.recall.storage import RecallDB

        with patch("synapt.recall.server.Path.cwd", return_value=tmp_path), \
             patch("synapt.recall.server.get_embedding_provider", return_value=None), \
             patch("synapt.recall.server._invalidate_cache"):
            recall_save(content="Old fact", category="tooling", node_id="fix1")
            recall_save(node_id="fix1", retract=True)
            # The sentence's own call shape:
            result = recall_save(
                node_id="fix1",
                content="The corrected fact",
                category="tooling",
                restore_retracted=True,
            )

        assert "re-activated" in result.lower(), result

        db = RecallDB(project_index_dir(tmp_path) / "recall.db")
        try:
            node = db.get_knowledge_node("fix1")
            assert node["status"] == "active"
            assert node["category"] == "tooling"  # category intact, not workflow
            assert node["content"] == "The corrected fact"
        finally:
            db.close()
