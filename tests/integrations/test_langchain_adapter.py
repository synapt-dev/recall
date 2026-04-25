"""Tests for the LangChain SynaptChatMessageHistory adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

langchain_core = pytest.importorskip("langchain_core")
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


@pytest.fixture
def history(tmp_path: Path):
    from synapt.integrations.langchain import SynaptChatMessageHistory

    h = SynaptChatMessageHistory(
        session_id="test-session-1",
        db_path=tmp_path / "test.db",
    )
    yield h
    h.close()


@pytest.fixture
def history_b(tmp_path: Path):
    from synapt.integrations.langchain import SynaptChatMessageHistory

    h = SynaptChatMessageHistory(
        session_id="test-session-2",
        db_path=tmp_path / "test.db",
    )
    yield h
    h.close()


class TestBaseChatMessageHistoryContract:

    def test_empty_on_init(self, history):
        assert history.messages == []

    def test_add_and_retrieve_human_message(self, history):
        history.add_messages([HumanMessage(content="hello")])
        msgs = history.messages
        assert len(msgs) == 1
        assert msgs[0].content == "hello"
        assert msgs[0].type == "human"

    def test_add_and_retrieve_ai_message(self, history):
        history.add_messages([AIMessage(content="hi there")])
        msgs = history.messages
        assert len(msgs) == 1
        assert msgs[0].content == "hi there"
        assert msgs[0].type == "ai"

    def test_add_and_retrieve_system_message(self, history):
        history.add_messages([SystemMessage(content="you are helpful")])
        msgs = history.messages
        assert len(msgs) == 1
        assert msgs[0].type == "system"

    def test_add_tool_message(self, history):
        history.add_messages([
            ToolMessage(content="result", tool_call_id="call_123"),
        ])
        msgs = history.messages
        assert len(msgs) == 1
        assert msgs[0].content == "result"

    def test_add_multiple_messages_preserves_order(self, history):
        history.add_messages([
            HumanMessage(content="first"),
            AIMessage(content="second"),
            HumanMessage(content="third"),
        ])
        msgs = history.messages
        assert len(msgs) == 3
        assert [m.content for m in msgs] == ["first", "second", "third"]
        assert [m.type for m in msgs] == ["human", "ai", "human"]

    def test_add_messages_across_calls(self, history):
        history.add_messages([HumanMessage(content="a")])
        history.add_messages([AIMessage(content="b")])
        msgs = history.messages
        assert len(msgs) == 2
        assert msgs[0].content == "a"
        assert msgs[1].content == "b"

    def test_clear(self, history):
        history.add_messages([
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
        ])
        assert len(history.messages) == 2
        history.clear()
        assert history.messages == []

    def test_clear_is_idempotent(self, history):
        history.clear()
        history.clear()
        assert history.messages == []


class TestSessionIsolation:

    def test_sessions_are_isolated(self, history, history_b):
        history.add_messages([HumanMessage(content="session 1")])
        history_b.add_messages([HumanMessage(content="session 2")])
        assert len(history.messages) == 1
        assert history.messages[0].content == "session 1"
        assert len(history_b.messages) == 1
        assert history_b.messages[0].content == "session 2"

    def test_clear_only_affects_own_session(self, history, history_b):
        history.add_messages([HumanMessage(content="keep me")])
        history_b.add_messages([HumanMessage(content="delete me")])
        history_b.clear()
        assert len(history.messages) == 1
        assert history_b.messages == []

    def test_session_count(self, history):
        assert history.session_count == 0
        history.add_messages([HumanMessage(content="one")])
        assert history.session_count == 1
        history.add_messages([HumanMessage(content="two"), AIMessage(content="three")])
        assert history.session_count == 3

    def test_get_session_ids(self, history, history_b):
        history.add_messages([HumanMessage(content="a")])
        history_b.add_messages([HumanMessage(content="b")])
        ids = history.get_session_ids()
        assert "test-session-1" in ids
        assert "test-session-2" in ids


class TestEdgeCases:

    def test_empty_content(self, history):
        history.add_messages([HumanMessage(content="")])
        msgs = history.messages
        assert len(msgs) == 1
        assert msgs[0].content == ""

    def test_unicode_content(self, history):
        history.add_messages([HumanMessage(content="Hello, world.")])
        assert history.messages[0].content == "Hello, world."

    def test_large_message(self, history):
        big = "x" * 100_000
        history.add_messages([HumanMessage(content=big)])
        assert history.messages[0].content == big

    def test_additional_kwargs_preserved(self, history):
        msg = AIMessage(content="hi", additional_kwargs={"custom": "value"})
        history.add_messages([msg])
        retrieved = history.messages[0]
        assert retrieved.additional_kwargs.get("custom") == "value"

    def test_repr(self, history):
        r = repr(history)
        assert "test-session-1" in r
        assert "messages=0" in r

    def test_importable_from_integrations(self):
        from synapt.integrations.langchain import SynaptChatMessageHistory
        assert SynaptChatMessageHistory is not None

    def test_response_metadata_preserved(self, history):
        msg = AIMessage(
            content="hello",
            response_metadata={"model": "claude-opus-4-6", "stop_reason": "end_turn"},
        )
        history.add_messages([msg])
        retrieved = history.messages[0]
        assert retrieved.response_metadata["model"] == "claude-opus-4-6"
        assert retrieved.response_metadata["stop_reason"] == "end_turn"

    def test_message_id_preserved(self, history):
        msg = AIMessage(content="hi", id="msg_abc123")
        history.add_messages([msg])
        retrieved = history.messages[0]
        assert retrieved.id == "msg_abc123"

    def test_tool_calls_preserved(self, history):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "call_1"}],
        )
        history.add_messages([msg])
        retrieved = history.messages[0]
        assert len(retrieved.tool_calls) == 1
        assert retrieved.tool_calls[0]["name"] == "search"


class TestRecallIntegration:

    def test_search_method_exists(self, history):
        assert callable(history.search)

    def test_save_to_recall_method_exists(self, history):
        assert callable(history.save_to_recall)

    def test_search_delegates_to_recall_search(self, history, monkeypatch):
        def fake_search(*, query, max_chunks, max_tokens):
            assert query == "deployment config"
            assert max_chunks == 7
            assert max_tokens == 900
            return "chunk: deployment config"

        monkeypatch.setattr("synapt.recall.server.recall_search", fake_search)

        result = history.search(
            "deployment config",
            max_chunks=7,
            max_tokens=900,
        )

        assert result == "chunk: deployment config"

    def test_save_to_recall_delegates_to_recall_save(self, history, monkeypatch):
        def fake_save(*, content, category, confidence, tags):
            assert content == "Always use UTC timestamps"
            assert category == "convention"
            assert confidence == 0.9
            assert tags == ["time", "ops"]
            return "saved-node-id"

        monkeypatch.setattr("synapt.recall.server.recall_save", fake_save)

        result = history.save_to_recall(
            "Always use UTC timestamps",
            category="convention",
            confidence=0.9,
            tags=["time", "ops"],
        )

        assert result == "saved-node-id"
