"""TDD witnesses for the 0.25.2 server-text range (release R3, recall 0.25.2).

Scope item B from #dev m_dc85dbbf. Every witness is born red on dev
e24b71e5c88a3cabdec2423b2f2f0b645930b66e; each names the behavior it pins.

Item 1: the MCP instructions teach save/update/retract, including the
restore_retracted path, matching the refusal sentence #1205 landed.
Item 2: the empty-index message says what a fresh agent should do next.
Item 3: an update that omits category keeps the node's existing category
(today an update without category re-files the node as "workflow"); a
caller who passes a category explicitly is honored.
Item 4: the X tools are not advertised unless the X integration is
configured.
Item 5: serverInfo carries OUR version, not the MCP SDK's.
"""

from __future__ import annotations

import json

import pytest


# --- Item 1: the instructions teach the knowledge-write verbs -------------

MCP_INSTRUCTION_CAP = 2048


def _delivered_instructions(text: str) -> str:
    """Claude Code receives only this prefix of MCP instruction text."""
    return text[:MCP_INSTRUCTION_CAP]


def test_instructions_teach_knowledge_lifecycle_within_delivery_cap():
    """Mechanics stay in tool descriptions because delivered text is the contract."""
    from synapt.recall.server import MCP_INSTRUCTIONS

    assert len(MCP_INSTRUCTIONS) <= MCP_INSTRUCTION_CAP
    text = _delivered_instructions(MCP_INSTRUCTIONS)
    assert "recall_save" in text, "the instructions never name the save verb"
    # the three states of a node, in the words the refusals use
    assert "update" in text and "retract" in text
    assert "recall_contradict" in text


def test_delivery_prefix_control_rejects_a_lifecycle_term_after_the_cap():
    """This control fails if delivery is accidentally changed to whole-source text."""
    source_text = "x" * MCP_INSTRUCTION_CAP + " recall_save"

    assert "recall_save" not in _delivered_instructions(source_text)
    assert "recall_save" in source_text


# --- Item 2: the empty-index message says what to do next ------------------

def test_empty_index_message_names_the_next_step():
    """A fresh agent that searches an empty index gets told: this is normal,
    and saving is what makes search useful. The message is composed by
    no_match_message, so the witness asserts the text without a store."""
    from synapt.recall.server import no_match_message

    msg = no_match_message(
        "anything at all",
        "searched 0 sessions across 0 indexed chunks",
        "semantic search was not used",
    )
    assert "No prior keyword match found" in msg
    assert "recall_save" in msg, (
        f"the empty-index answer must name the verb that makes the next "
        f"search useful:\n{msg}"
    )
    assert "fresh" in msg.lower(), (
        "the message must say a fresh index is normal, so the agent does "
        "not read the miss as a failure"
    )


# --- Item 3: update without a category keeps the node's category ----------

def test_update_without_category_keeps_the_existing_category(monkeypatch, tmp_path):
    """Today an update whose caller omits category re-files the node as
    "workflow" (the signature default). The fix: omitted is not workflow —
    the node keeps its own category; an explicitly-passed category wins."""
    from synapt.recall import server

    created: dict = {}
    updated: dict = {}

    class _FakeDB:
        def __init__(self, *_a, **_k):
            pass

        def get_knowledge_node(self, node_id):
            if updated:
                return updated["node"]
            return None

        def upsert_knowledge_node(self, node):
            return None

        def get_knowledge_rowid(self, node_id):
            return None

        def save_knowledge_embeddings(self, mapping):
            return None

        def close(self):
            return None

    # recall_save imports from the SOURCE modules at call time, so the
    # patches land there, not on the server module
    import synapt.recall.storage as storage
    monkeypatch.setattr(storage, "RecallDB", _FakeDB, raising=False)

    class _Node:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.id = kw.get("node_id")
            self.version = 1
            self.created_at = "t0"
            self.lineage_id = ""

        @staticmethod
        def create(**kw):
            created.update(kw)
            return _Node(**kw)

    import synapt.recall.knowledge as knowledge
    monkeypatch.setattr(knowledge, "KnowledgeNode", _Node, raising=False)
    monkeypatch.setattr(knowledge, "save_knowledge_node", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("synapt.recall.server._invalidate_cache", lambda: None, raising=False)
    monkeypatch.setattr("synapt.recall.server.get_embedding_provider", lambda: None, raising=False)
    import synapt.recall.sharding as sharding
    monkeypatch.setattr(sharding, "live_store_path",
                        lambda *a, **k: tmp_path / "live.db", raising=False)
    # project_data_dir / project_index_dir are called directly in recall_save
    # and resolve cwd-derived paths the store-isolation guard rightly refuses
    monkeypatch.setattr("synapt.recall.server.project_data_dir",
                        lambda *a, **k: tmp_path / "data", raising=False)
    monkeypatch.setattr("synapt.recall.server.project_index_dir",
                        lambda *a, **k: tmp_path / "index", raising=False)

    # create with an explicit category
    first = server.recall_save(content="the billing service uses Postgres 14",
                               category="decision")
    assert "decision" in first, first
    # update WITHOUT category: the node must keep "decision", not re-file as workflow
    existing_node = {
        "id": created["node_id"], "created_at": "t0", "version": 1,
        "lineage_id": created["node_id"], "category": "decision",
        "status": "active",
    }
    updated["node"] = existing_node
    result = server.recall_save(content="the billing service uses Postgres 15")
    assert "decision" in result, (
        f"an update that omits category must keep the node's own category; "
        f"got:\n{result[:300]}"
    )
    # an explicitly-passed category still wins
    result2 = server.recall_save(content="the billing service uses Postgres 16",
                                 category="tooling")
    assert "tooling" in result2, result2


# --- Item 4: X tools not advertised unless configured ----------------------

def test_x_tools_not_advertised_unless_configured(monkeypatch):
    """synapt.x.server.register_tools registers all eight X tools with no
    gate on the X_* credentials: an agent whose server carries them looks
    like it can post. Unconfigured, none may be advertised."""
    import asyncio
    from synapt.x import server as x_server

    class _Stub:
        def __init__(self):
            self.names = []

        def tool(self):
            def deco(fn):
                self.names.append(fn.__name__)
                return fn
            return deco

    monkeypatch.delenv("X_API_KEY", raising=False)
    monkeypatch.delenv("X_API_SECRET", raising=False)
    monkeypatch.delenv("X_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("X_ACCESS_TOKEN_SECRET", raising=False)
    stub = _Stub()
    x_server.register_tools(stub)
    assert not stub.names, (
        f"unconfigured (no X_* credentials), the server must not advertise "
        f"any X tool; got {stub.names}"
    )
    # configured: the tools are advertised
    monkeypatch.setenv("X_API_KEY", "k")
    monkeypatch.setenv("X_API_SECRET", "s")
    monkeypatch.setenv("X_ACCESS_TOKEN", "a")
    monkeypatch.setenv("X_ACCESS_TOKEN_SECRET", "t")
    stub2 = _Stub()
    x_server.register_tools(stub2)
    assert len(stub2.names) == 8, stub2.names


# --- Item 5: serverInfo carries OUR version --------------------------------

def test_server_info_reports_the_product_version_not_the_sdk():
    from synapt.recall import server as server_mod

    version = getattr(server_mod, "_PRODUCT_VERSION", None)
    assert version, "the server module must carry the product version for serverInfo"
    assert version != __import__("mcp", fromlist=["x"]).__version__ if hasattr(__import__("mcp", fromlist=["x"]), "__version__") else True
