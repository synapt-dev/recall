from __future__ import annotations

import pytest

langchain_history = pytest.importorskip("langchain_core.chat_history")
langchain_messages = pytest.importorskip("langchain_core.messages")

BaseChatMessageHistory = langchain_history.BaseChatMessageHistory
AIMessage = langchain_messages.AIMessage
HumanMessage = langchain_messages.HumanMessage


def _adapter_cls():
    from synapt.integrations.langchain import SynaptChatMessageHistory

    return SynaptChatMessageHistory


def test_langchain_adapter_instantiates_and_matches_interface(tmp_path):
    SynaptChatMessageHistory = _adapter_cls()

    history = SynaptChatMessageHistory(
        db_path=tmp_path / "recall.db",
        session_id="thread-1",
    )

    assert isinstance(history, BaseChatMessageHistory)
    assert history.session_id == "thread-1"
    assert history.messages == []


def test_langchain_adapter_round_trips_messages_for_same_session(tmp_path):
    SynaptChatMessageHistory = _adapter_cls()

    history = SynaptChatMessageHistory(
        db_path=tmp_path / "recall.db",
        session_id="thread-1",
    )
    history.add_messages(
        [
            HumanMessage(content="Remember that Alice prefers concise status updates."),
            AIMessage(content="Stored. Alice prefers concise status updates."),
        ]
    )

    reloaded = SynaptChatMessageHistory(
        db_path=tmp_path / "recall.db",
        session_id="thread-1",
    )

    assert [message.content for message in reloaded.messages] == [
        "Remember that Alice prefers concise status updates.",
        "Stored. Alice prefers concise status updates.",
    ]


def test_langchain_adapter_isolates_sessions_and_clear_only_resets_current_session(tmp_path):
    SynaptChatMessageHistory = _adapter_cls()

    thread_one = SynaptChatMessageHistory(
        db_path=tmp_path / "recall.db",
        session_id="thread-1",
    )
    thread_two = SynaptChatMessageHistory(
        db_path=tmp_path / "recall.db",
        session_id="thread-2",
    )

    thread_one.add_messages([HumanMessage(content="Project Falcon is in design review.")])
    thread_two.add_messages([HumanMessage(content="Project Zephyr is blocked on legal.")])

    assert [message.content for message in thread_one.messages] == [
        "Project Falcon is in design review."
    ]
    assert [message.content for message in thread_two.messages] == [
        "Project Zephyr is blocked on legal."
    ]

    thread_one.clear()

    assert thread_one.messages == []
    assert [message.content for message in thread_two.messages] == [
        "Project Zephyr is blocked on legal."
    ]
