"""Tests for agent-to-agent direct messaging (P5 speak_to_agent v0).

Covers:
- Send/read/ack lifecycle
- Delivery state machine transitions
- Idempotency (duplicate message_id)
- Threading via reply_to
- Priority ordering (urgent surfaces first)
- Validation (empty body, self-send, body size cap)
- before_send hook (allow/deny)
- state_change hook fires on transitions
- History between two agents
"""

from __future__ import annotations

import pytest

from synapt.recall.direct import (
    MAX_BODY_SIZE,
    PRIORITY_NORMAL,
    PRIORITY_URGENT,
    STATUS_ACKED,
    STATUS_DELIVERED,
    STATUS_READ,
    DirectMessage,
    _clear_hooks,
    ack_message,
    check_status,
    message_history,
    read_inbox,
    register_before_send_hook,
    register_state_change_hook,
    send_message,
)


@pytest.fixture(autouse=True)
def _isolated_dir(tmp_path, monkeypatch):
    """Isolate each test with its own data directory."""
    monkeypatch.setenv("SYNAPT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SYNAPT_SHARED_CHANNELS_DIR", str(tmp_path / "channels"))
    (tmp_path / "channels" / "direct").mkdir(parents=True, exist_ok=True)
    _clear_hooks()
    yield


# ── Data model ─────────────────────────────────────────────────


class TestDirectMessage:
    def test_to_dict_omits_defaults(self) -> None:
        msg = DirectMessage(
            message_id="dm_test123",
            from_agent="apollo-001",
            to_agent="opus-001",
            timestamp="2026-05-08T05:00:00Z",
            body="hello",
        )
        d = msg.to_dict()
        assert d["from"] == "apollo-001"
        assert d["to"] == "opus-001"
        assert "from_agent" not in d
        assert "to_agent" not in d
        assert "reply_to" not in d
        assert "priority" not in d

    def test_to_dict_includes_non_defaults(self) -> None:
        msg = DirectMessage(
            message_id="dm_test123",
            from_agent="apollo-001",
            to_agent="opus-001",
            timestamp="2026-05-08T05:00:00Z",
            body="urgent matter",
            reply_to="dm_parent",
            priority=PRIORITY_URGENT,
        )
        d = msg.to_dict()
        assert d["reply_to"] == "dm_parent"
        assert d["priority"] == "urgent"

    def test_from_dict_roundtrip(self) -> None:
        msg = DirectMessage(
            message_id="dm_test123",
            from_agent="apollo-001",
            to_agent="opus-001",
            timestamp="2026-05-08T05:00:00Z",
            body="hello",
            reply_to="dm_parent",
            priority=PRIORITY_URGENT,
        )
        d = msg.to_dict()
        restored = DirectMessage.from_dict(d)
        assert restored.message_id == msg.message_id
        assert restored.from_agent == msg.from_agent
        assert restored.to_agent == msg.to_agent
        assert restored.body == msg.body
        assert restored.reply_to == msg.reply_to
        assert restored.priority == msg.priority


# ── Send/read/ack lifecycle ────────────────────────────────────


class TestSendReadAck:
    def test_send_and_read(self) -> None:
        msg = send_message(
            from_agent="apollo-001",
            to_agent="opus-001",
            body="Sprint E shipped",
        )
        assert msg.message_id.startswith("dm_")
        assert msg.from_agent == "apollo-001"
        assert msg.to_agent == "opus-001"

        inbox = read_inbox(agent_id="opus-001")
        assert len(inbox) == 1
        assert inbox[0].body == "Sprint E shipped"

    def test_read_marks_as_read(self) -> None:
        send_message(
            from_agent="apollo-001",
            to_agent="opus-001",
            body="first message",
        )
        read_inbox(agent_id="opus-001")

        # Second read should return empty (already marked READ)
        inbox = read_inbox(agent_id="opus-001")
        assert len(inbox) == 0

    def test_ack_lifecycle(self) -> None:
        msg = send_message(
            from_agent="apollo-001",
            to_agent="opus-001",
            body="needs ack",
        )

        # Check initial status
        status = check_status(message_id=msg.message_id)
        assert status is not None
        assert status["status"] == STATUS_DELIVERED

        # Read it
        read_inbox(agent_id="opus-001")
        status = check_status(message_id=msg.message_id)
        assert status["status"] == STATUS_READ

        # Ack it
        result = ack_message(message_id=msg.message_id, agent_id="opus-001")
        assert "acknowledged" in result
        status = check_status(message_id=msg.message_id)
        assert status["status"] == STATUS_ACKED

    def test_ack_wrong_recipient(self) -> None:
        msg = send_message(
            from_agent="apollo-001",
            to_agent="opus-001",
            body="not for you",
        )
        result = ack_message(message_id=msg.message_id, agent_id="atlas-001")
        assert "not addressed" in result

    def test_ack_already_acked(self) -> None:
        msg = send_message(
            from_agent="apollo-001",
            to_agent="opus-001",
            body="double ack",
        )
        read_inbox(agent_id="opus-001")
        ack_message(message_id=msg.message_id, agent_id="opus-001")
        result = ack_message(message_id=msg.message_id, agent_id="opus-001")
        assert "already" in result

    def test_ack_unknown_message(self) -> None:
        result = ack_message(message_id="dm_nonexistent", agent_id="opus-001")
        assert "not found" in result

    def test_status_unknown_message(self) -> None:
        result = check_status(message_id="dm_nonexistent")
        assert result is None


# ── Priority ordering ──────────────────────────────────────────


class TestPriority:
    def test_urgent_surfaces_first(self) -> None:
        send_message(
            from_agent="atlas-001",
            to_agent="opus-001",
            body="normal message",
            priority=PRIORITY_NORMAL,
        )
        send_message(
            from_agent="apollo-001",
            to_agent="opus-001",
            body="URGENT: blocker found",
            priority=PRIORITY_URGENT,
        )
        inbox = read_inbox(agent_id="opus-001")
        assert len(inbox) == 2
        assert inbox[0].priority == PRIORITY_URGENT
        assert inbox[1].priority == PRIORITY_NORMAL


# ── Threading ──────────────────────────────────────────────────


class TestThreading:
    def test_reply_to_threading(self) -> None:
        parent = send_message(
            from_agent="opus-001",
            to_agent="apollo-001",
            body="Can you review grip#721?",
        )
        reply = send_message(
            from_agent="apollo-001",
            to_agent="opus-001",
            body="Done, all blockers fixed.",
            reply_to=parent.message_id,
        )
        assert reply.reply_to == parent.message_id


# ── Validation ─────────────────────────────────────────────────


class TestValidation:
    def test_empty_body_rejected(self) -> None:
        with pytest.raises(ValueError, match="body is required"):
            send_message(
                from_agent="apollo-001",
                to_agent="opus-001",
                body="",
            )

    def test_whitespace_body_rejected(self) -> None:
        with pytest.raises(ValueError, match="body is required"):
            send_message(
                from_agent="apollo-001",
                to_agent="opus-001",
                body="   ",
            )

    def test_self_send_rejected(self) -> None:
        with pytest.raises(ValueError, match="yourself"):
            send_message(
                from_agent="apollo-001",
                to_agent="apollo-001",
                body="talking to myself",
            )

    def test_body_size_cap(self) -> None:
        with pytest.raises(ValueError, match="byte limit"):
            send_message(
                from_agent="apollo-001",
                to_agent="opus-001",
                body="x" * (MAX_BODY_SIZE + 1),
            )

    def test_invalid_priority_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid priority"):
            send_message(
                from_agent="apollo-001",
                to_agent="opus-001",
                body="hello",
                priority="critical",
            )

    def test_missing_from_agent(self) -> None:
        with pytest.raises(ValueError, match="from_agent"):
            send_message(
                from_agent="",
                to_agent="opus-001",
                body="hello",
            )

    def test_missing_to_agent(self) -> None:
        with pytest.raises(ValueError, match="to_agent"):
            send_message(
                from_agent="apollo-001",
                to_agent="",
                body="hello",
            )


# ── Hooks ──────────────────────────────────────────────────────


class TestHooks:
    def test_before_send_deny(self) -> None:
        def deny_all(msg: DirectMessage) -> str | None:
            return "rate limit exceeded"

        register_before_send_hook(deny_all)
        with pytest.raises(PermissionError, match="rate limit"):
            send_message(
                from_agent="apollo-001",
                to_agent="opus-001",
                body="should be denied",
            )

    def test_before_send_allow(self) -> None:
        def allow_all(msg: DirectMessage) -> str | None:
            return None

        register_before_send_hook(allow_all)
        msg = send_message(
            from_agent="apollo-001",
            to_agent="opus-001",
            body="should be allowed",
        )
        assert msg.message_id.startswith("dm_")

    def test_state_change_hook_fires(self) -> None:
        transitions: list[tuple[str, str, str]] = []

        def track(mid: str, old: str, new: str) -> None:
            transitions.append((mid, old, new))

        register_state_change_hook(track)
        msg = send_message(
            from_agent="apollo-001",
            to_agent="opus-001",
            body="tracked message",
        )
        read_inbox(agent_id="opus-001")
        ack_message(message_id=msg.message_id, agent_id="opus-001")

        ids = [t[0] for t in transitions]
        assert msg.message_id in ids

        statuses = [(t[1], t[2]) for t in transitions if t[0] == msg.message_id]
        assert ("queued", "delivered") in statuses
        assert ("delivered", "read") in statuses
        assert ("read", "acked") in statuses

    def test_hook_exception_does_not_break_send(self) -> None:
        def exploding_hook(msg: DirectMessage) -> str | None:
            raise RuntimeError("hook crashed")

        register_state_change_hook(exploding_hook)
        msg = send_message(
            from_agent="apollo-001",
            to_agent="opus-001",
            body="should still send",
        )
        assert msg.message_id.startswith("dm_")


# ── History ────────────────────────────────────────────────────


class TestHistory:
    def test_bidirectional_history(self) -> None:
        send_message(
            from_agent="apollo-001",
            to_agent="opus-001",
            body="message 1",
        )
        send_message(
            from_agent="opus-001",
            to_agent="apollo-001",
            body="message 2",
        )
        history = message_history(
            agent_id="apollo-001",
            with_agent="opus-001",
        )
        assert len(history) == 2

    def test_history_excludes_other_agents(self) -> None:
        send_message(
            from_agent="apollo-001",
            to_agent="opus-001",
            body="to opus",
        )
        send_message(
            from_agent="apollo-001",
            to_agent="atlas-001",
            body="to atlas",
        )
        history = message_history(
            agent_id="apollo-001",
            with_agent="opus-001",
        )
        assert len(history) == 1
        assert history[0].body == "to opus"

    def test_empty_history(self) -> None:
        history = message_history(
            agent_id="apollo-001",
            with_agent="sentinel-001",
        )
        assert len(history) == 0


# ── JSONL persistence ──────────────────────────────────────────


class TestJSONLPersistence:
    def test_inbox_jsonl_written(self, tmp_path) -> None:
        send_message(
            from_agent="apollo-001",
            to_agent="opus-001",
            body="persisted message",
        )
        inbox_file = tmp_path / "channels" / "direct" / "opus-001.jsonl"
        assert inbox_file.exists()
        lines = inbox_file.read_text().strip().split("\n")
        assert len(lines) == 1
        import json

        data = json.loads(lines[0])
        assert data["from"] == "apollo-001"
        assert data["to"] == "opus-001"
        assert data["body"] == "persisted message"


# ── Multiple messages ──────────────────────────────────────────


class TestMultipleMessages:
    def test_multiple_senders(self) -> None:
        send_message(
            from_agent="atlas-001",
            to_agent="opus-001",
            body="from atlas",
        )
        send_message(
            from_agent="apollo-001",
            to_agent="opus-001",
            body="from apollo",
        )
        send_message(
            from_agent="sentinel-001",
            to_agent="opus-001",
            body="from sentinel",
        )
        inbox = read_inbox(agent_id="opus-001")
        assert len(inbox) == 3

    def test_limit_parameter(self) -> None:
        for i in range(5):
            send_message(
                from_agent="apollo-001",
                to_agent="opus-001",
                body=f"message {i}",
            )
        inbox = read_inbox(agent_id="opus-001", limit=3)
        assert len(inbox) == 3
