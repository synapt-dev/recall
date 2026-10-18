"""The ADK memory service is a node-creating caller of the 4,000-char limit.

Conversation text has no bound of its own, so an event over the limit is
refused by KnowledgeNode.create before save_knowledge_node runs: a warning
naming the limit, and nothing written to the service's store.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("google.adk", reason="google-adk not installed")

from synapt.recall.knowledge import read_nodes


def _save(tmp_path, monkeypatch, text):
    from synapt.integrations.google_adk import SynaptMemoryService

    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    root = tmp_path / "adk"
    svc = SynaptMemoryService(project_root=root)
    svc._save_to_recall(app_name="app", user_id="u1", text=text, author="t")
    return root


def _contents(root):
    out = []
    for p in root.rglob("knowledge.jsonl"):
        out += [n.content for n in read_nodes(p)]
    return out


def test_adk_event_over_limit_writes_nothing(tmp_path, monkeypatch, caplog):
    text = ("word " * 801)[:4001]
    with caplog.at_level(logging.WARNING):
        root = _save(tmp_path, monkeypatch, text)
    assert "4001 chars; the limit is 4000" in caplog.text
    assert _contents(root) == []
    assert list(root.rglob("*.db")) == []


def test_adk_event_at_limit_is_saved_whole(tmp_path, monkeypatch):
    text = ("word " * 801)[:3999] + "z"
    assert len(text) == 4000
    root = _save(tmp_path, monkeypatch, text)
    assert _contents(root) == [text]
