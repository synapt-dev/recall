"""Tests for synapt.recall.channel -- cross-worktree agent communication."""

import json
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

from synapt.recall.channel import (
    ChannelMessage,
    _channels_dir,
    _db_path,
    _open_db,
    _read_messages,
    _agent_status,
    _reap_stale_agents,
    channel_join,
    channel_leave,
    channel_post,
    channel_read,
    channel_who,
    channel_heartbeat,
    channel_unread,
    channel_pin,
)


def _patch_data_dir(tmpdir):
    """Return a patcher for project_data_dir targeting a temp directory."""
    data_dir = Path(tmpdir) / "project" / ".synapt" / "recall"
    return patch(
        "synapt.recall.channel.project_data_dir",
        return_value=data_dir,
    )


class TestChannelMessage(unittest.TestCase):
    """Test ChannelMessage dataclass serialization."""

    def test_to_dict_uses_from_key(self):
        msg = ChannelMessage(
            timestamp="2026-03-16T10:00:00Z",
            from_agent="agent-a",
            channel="dev",
            type="message",
            body="hello",
        )
        d = msg.to_dict()
        self.assertEqual(d["from"], "agent-a")
        self.assertNotIn("from_agent", d)

    def test_from_dict_reads_from_key(self):
        d = {
            "timestamp": "2026-03-16T10:00:00Z",
            "from": "agent-b",
            "channel": "dev",
            "type": "message",
            "body": "world",
        }
        msg = ChannelMessage.from_dict(d)
        self.assertEqual(msg.from_agent, "agent-b")
        self.assertEqual(msg.body, "world")

    def test_round_trip(self):
        msg = ChannelMessage(
            timestamp="2026-03-16T10:00:00Z",
            from_agent="test",
            channel="eval",
            type="join",
            body="test joined #eval",
        )
        restored = ChannelMessage.from_dict(msg.to_dict())
        self.assertEqual(restored.from_agent, msg.from_agent)
        self.assertEqual(restored.channel, msg.channel)
        self.assertEqual(restored.type, msg.type)
        self.assertEqual(restored.body, msg.body)

    def test_from_dict_ignores_extra_keys(self):
        d = {
            "timestamp": "2026-03-16T10:00:00Z",
            "from": "x",
            "channel": "dev",
            "type": "message",
            "body": "hi",
            "extra_field": "ignored",
        }
        msg = ChannelMessage.from_dict(d)
        self.assertEqual(msg.from_agent, "x")


class TestSQLitePresenceCRUD(unittest.TestCase):
    """Test SQLite presence create/read/update/delete via join/leave/heartbeat."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_join_creates_presence(self):
        channel_join("dev", agent_name="agent-a")
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT * FROM presence WHERE agent_id ='agent-a'"
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["status"], "online")
        finally:
            conn.close()

    def test_join_creates_membership(self):
        channel_join("dev", agent_name="agent-a")
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT * FROM memberships WHERE agent_id ='agent-a' AND channel = 'dev'"
            ).fetchone()
            self.assertIsNotNone(row)
        finally:
            conn.close()

    def test_join_initializes_cursor(self):
        channel_join("dev", agent_name="agent-a")
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT * FROM cursors WHERE agent_id ='agent-a' AND channel = 'dev'"
            ).fetchone()
            self.assertIsNotNone(row)
        finally:
            conn.close()

    def test_leave_removes_membership(self):
        channel_join("dev", agent_name="agent-a")
        channel_leave("dev", agent_name="agent-a")
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT * FROM memberships WHERE agent_id ='agent-a' AND channel = 'dev'"
            ).fetchone()
            self.assertIsNone(row)
        finally:
            conn.close()

    def test_leave_last_channel_removes_presence(self):
        channel_join("dev", agent_name="agent-a")
        channel_leave("dev", agent_name="agent-a")
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT * FROM presence WHERE agent_id ='agent-a'"
            ).fetchone()
            self.assertIsNone(row)
        finally:
            conn.close()

    def test_leave_one_of_two_channels_keeps_presence(self):
        channel_join("dev", agent_name="agent-a")
        channel_join("eval", agent_name="agent-a")
        channel_leave("dev", agent_name="agent-a")
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT * FROM presence WHERE agent_id ='agent-a'"
            ).fetchone()
            self.assertIsNotNone(row)
            # Still has eval membership
            mem = conn.execute(
                "SELECT channel FROM memberships WHERE agent_id ='agent-a'"
            ).fetchall()
            self.assertEqual(len(mem), 1)
            self.assertEqual(mem[0]["channel"], "eval")
        finally:
            conn.close()

    def test_heartbeat_updates_last_seen(self):
        channel_join("dev", agent_name="agent-a")

        # Get initial last_seen
        conn = _open_db()
        try:
            row1 = conn.execute(
                "SELECT last_seen FROM presence WHERE agent_id ='agent-a'"
            ).fetchone()
        finally:
            conn.close()

        # Small delay then heartbeat
        time.sleep(0.01)
        channel_heartbeat(agent_name="agent-a")

        conn = _open_db()
        try:
            row2 = conn.execute(
                "SELECT last_seen FROM presence WHERE agent_id ='agent-a'"
            ).fetchone()
        finally:
            conn.close()

        self.assertGreaterEqual(row2["last_seen"], row1["last_seen"])

    def test_heartbeat_no_presence_returns_skip(self):
        result = channel_heartbeat(agent_name="ghost")
        self.assertIn("skipped", result)


class TestStaleDetectionTiers(unittest.TestCase):
    """Test online/idle/away/offline status tiers."""

    def test_online_within_5_minutes(self):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.assertEqual(_agent_status(ts), "online")

    def test_idle_between_5_and_30_minutes(self):
        ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self.assertEqual(_agent_status(ts), "idle")

    def test_away_between_30_and_120_minutes(self):
        ts = (datetime.now(timezone.utc) - timedelta(minutes=60)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self.assertEqual(_agent_status(ts), "away")

    def test_offline_after_120_minutes(self):
        ts = (datetime.now(timezone.utc) - timedelta(minutes=150)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self.assertEqual(_agent_status(ts), "offline")

    def test_boundary_5_minutes(self):
        # At exactly 5 minutes, should be idle
        ts = (datetime.now(timezone.utc) - timedelta(minutes=5, seconds=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self.assertEqual(_agent_status(ts), "idle")

    def test_boundary_30_minutes(self):
        # At exactly 30 minutes, should be away
        ts = (datetime.now(timezone.utc) - timedelta(minutes=30, seconds=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self.assertEqual(_agent_status(ts), "away")


class TestAutoLeaveTimeout(unittest.TestCase):
    """Test that agents > 2 hours stale are auto-removed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_reap_stale_agent(self):
        # Manually insert a stale agent
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        conn = _open_db()
        try:
            conn.execute(
                "INSERT INTO presence (agent_id, status, last_seen, joined_at) "
                "VALUES ('stale-bot', 'online', ?, ?)",
                (stale_time, stale_time),
            )
            conn.execute(
                "INSERT INTO memberships (agent_id, channel, joined_at) "
                "VALUES ('stale-bot', 'dev', ?)",
                (stale_time,),
            )
            conn.commit()

            reaped = _reap_stale_agents(conn)
            self.assertIn("stale-bot", reaped)

            # Verify presence status updated
            row = conn.execute(
                "SELECT status FROM presence WHERE agent_id ='stale-bot'"
            ).fetchone()
            self.assertEqual(row["status"], "offline")

            # Verify membership removed
            mem = conn.execute(
                "SELECT * FROM memberships WHERE agent_id ='stale-bot'"
            ).fetchone()
            self.assertIsNone(mem)
        finally:
            conn.close()

    def test_reap_posts_leave_message(self):
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        conn = _open_db()
        try:
            conn.execute(
                "INSERT INTO presence (agent_id, status, last_seen, joined_at) "
                "VALUES ('stale-bot', 'online', ?, ?)",
                (stale_time, stale_time),
            )
            conn.execute(
                "INSERT INTO memberships (agent_id, channel, joined_at) "
                "VALUES ('stale-bot', 'dev', ?)",
                (stale_time,),
            )
            conn.commit()
            _reap_stale_agents(conn)
        finally:
            conn.close()

        # Check channel log for the leave message
        result = channel_read("dev")
        self.assertIn("timed out", result)

    def test_who_triggers_reap(self):
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        conn = _open_db()
        try:
            conn.execute(
                "INSERT INTO presence (agent_id, status, last_seen, joined_at) "
                "VALUES ('stale-bot', 'online', ?, ?)",
                (stale_time, stale_time),
            )
            conn.execute(
                "INSERT INTO memberships (agent_id, channel, joined_at) "
                "VALUES ('stale-bot', 'dev', ?)",
                (stale_time,),
            )
            conn.commit()
        finally:
            conn.close()

        result = channel_who()
        # Stale bot should be reaped -- not shown as online
        self.assertNotIn("stale-bot", result)

    def test_read_triggers_reap(self):
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        conn = _open_db()
        try:
            conn.execute(
                "INSERT INTO presence (agent_id, status, last_seen, joined_at) "
                "VALUES ('stale-bot', 'online', ?, ?)",
                (stale_time, stale_time),
            )
            conn.execute(
                "INSERT INTO memberships (agent_id, channel, joined_at) "
                "VALUES ('stale-bot', 'dev', ?)",
                (stale_time,),
            )
            conn.commit()
        finally:
            conn.close()

        # Create the channel file so read works
        channel_post("dev", "hi", agent_name="active-agent")
        channel_read("dev")

        # Check that stale-bot was reaped
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT status FROM presence WHERE agent_id ='stale-bot'"
            ).fetchone()
            self.assertEqual(row["status"], "offline")
        finally:
            conn.close()

    def test_fresh_agent_not_reaped(self):
        channel_join("dev", agent_name="fresh-bot")
        result = channel_who()
        self.assertIn("fresh-bot", result)
        self.assertIn("online", result)


class TestUnreadMessages(unittest.TestCase):
    """Test unread message counting with cursors."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_no_memberships_returns_empty(self):
        result = channel_unread(agent_name="nobody")
        self.assertEqual(result, {})

    def test_unread_after_join(self):
        channel_join("dev", agent_name="agent-a")
        # Post messages from another agent
        channel_post("dev", "msg1", agent_name="agent-b")
        channel_post("dev", "msg2", agent_name="agent-b")

        unread = channel_unread(agent_name="agent-a")
        self.assertEqual(unread["dev"], 2)

    def test_read_resets_cursor(self):
        channel_join("dev", agent_name="agent-a")
        channel_post("dev", "msg1", agent_name="agent-b")
        channel_post("dev", "msg2", agent_name="agent-b")

        # Read the channel (updates cursor)
        channel_read("dev", agent_name="agent-a")

        # Now unread should be 0
        unread = channel_unread(agent_name="agent-a")
        self.assertEqual(unread["dev"], 0)

    def test_new_messages_after_read(self):
        channel_join("dev", agent_name="agent-a")
        channel_post("dev", "msg1", agent_name="agent-b")

        # Read
        channel_read("dev", agent_name="agent-a")

        # New message
        channel_post("dev", "msg2", agent_name="agent-b")

        unread = channel_unread(agent_name="agent-a")
        self.assertEqual(unread["dev"], 1)

    def test_unread_multiple_channels(self):
        channel_join("dev", agent_name="agent-a")
        channel_join("eval", agent_name="agent-a")

        channel_post("dev", "dev msg", agent_name="agent-b")
        channel_post("eval", "eval msg1", agent_name="agent-b")
        channel_post("eval", "eval msg2", agent_name="agent-b")

        unread = channel_unread(agent_name="agent-a")
        self.assertEqual(unread["dev"], 1)
        self.assertEqual(unread["eval"], 2)


class TestPins(unittest.TestCase):
    """Test pin creation and display in read output."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def _post_and_get_id(self, channel, body, agent="bot"):
        """Post a message and return its ID from the JSONL."""
        channel_post(channel, body, agent_name=agent)
        msgs = _read_messages(_channels_dir() / f"{channel}.jsonl")
        for m in reversed(msgs):
            if m.body == body:
                return m.id
        return ""

    def test_pin_standalone(self):
        msg_id = self._post_and_get_id("dev", "important rule")
        result = channel_pin("dev", msg_id, agent_name="admin")
        self.assertIn("Pinned", result)
        self.assertIn("important rule", result)

    def test_pin_stored_in_db(self):
        msg_id = self._post_and_get_id("dev", "rule 1")
        channel_pin("dev", msg_id, agent_name="admin")
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT * FROM pins WHERE channel = 'dev'"
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["body"], "rule 1")
            self.assertEqual(row["message_id"], msg_id)
            self.assertEqual(row["pinned_by"], "admin")
        finally:
            conn.close()

    def test_pin_on_post(self):
        channel_post("dev", "pinned msg", agent_name="bot", pin=True)
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT * FROM pins WHERE channel = 'dev'"
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["body"], "pinned msg")
            self.assertTrue(row["message_id"].startswith("m_"))
        finally:
            conn.close()

    def test_pins_shown_in_read(self):
        msg_id = self._post_and_get_id("dev", "important: follow the rules")
        channel_pin("dev", msg_id, agent_name="admin")
        channel_post("dev", "hello", agent_name="bot")

        result = channel_read("dev")
        self.assertIn("Pinned", result)
        self.assertIn("[pin]", result)
        self.assertIn("follow the rules", result)
        self.assertIn("hello", result)

    def test_pins_appear_before_messages(self):
        msg_id = self._post_and_get_id("dev", "pin content")
        channel_pin("dev", msg_id, agent_name="admin")
        channel_post("dev", "regular content", agent_name="bot")

        result = channel_read("dev")
        pin_pos = result.index("[pin]")
        msg_pos = result.index("regular content")
        self.assertLess(pin_pos, msg_pos)

    def test_multiple_pins(self):
        id1 = self._post_and_get_id("dev", "rule 1")
        id2 = self._post_and_get_id("dev", "rule 2")
        channel_pin("dev", id1, agent_name="admin")
        channel_pin("dev", id2, agent_name="admin")
        channel_post("dev", "hello", agent_name="bot")

        result = channel_read("dev")
        self.assertIn("rule 1", result)
        self.assertIn("rule 2", result)


class TestPostAndRead(unittest.TestCase):
    """Test posting and reading messages."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_post_creates_channel(self):
        result = channel_post("dev", "hello world", agent_name="agent-a")
        self.assertIn("hello world", result)
        # Verify JSONL file was created
        data_dir = Path(self.tmpdir) / "project" / ".synapt" / "recall"
        channel_file = data_dir / "channels" / "dev.jsonl"
        self.assertTrue(channel_file.exists())

    def test_read_after_post(self):
        channel_post("dev", "first message", agent_name="agent-a")
        channel_post("dev", "second message", agent_name="agent-b")
        result = channel_read("dev")
        self.assertIn("first message", result)
        self.assertIn("second message", result)
        self.assertIn("agent-a", result)
        self.assertIn("agent-b", result)

    def test_read_empty_channel(self):
        result = channel_read("empty")
        self.assertIn("no messages", result)

    def test_read_with_limit(self):
        for i in range(10):
            channel_post("dev", f"msg {i}", agent_name="bot")
        result = channel_read("dev", limit=3)
        # Should show only last 3 messages
        self.assertIn("msg 7", result)
        self.assertIn("msg 8", result)
        self.assertIn("msg 9", result)
        self.assertNotIn("msg 6", result)

    def test_read_with_since_filter(self):
        # Post messages with controlled timestamps by writing directly
        data_dir = Path(self.tmpdir) / "project" / ".synapt" / "recall"
        channels_dir = data_dir / "channels"
        channels_dir.mkdir(parents=True, exist_ok=True)
        path = channels_dir / "dev.jsonl"

        messages = [
            {"timestamp": "2026-03-16T10:00:00Z", "from": "a", "channel": "dev",
             "type": "message", "body": "old msg"},
            {"timestamp": "2026-03-16T12:00:00Z", "from": "a", "channel": "dev",
             "type": "message", "body": "new msg"},
        ]
        with open(path, "w", encoding="utf-8") as f:
            for m in messages:
                f.write(json.dumps(m) + "\n")

        result = channel_read("dev", since="2026-03-16T11:00:00Z")
        self.assertIn("new msg", result)
        self.assertNotIn("old msg", result)

    def test_since_no_matches(self):
        channel_post("dev", "early msg", agent_name="bot")
        # Since is in the future -- no matches
        result = channel_read("dev", since="2099-01-01T00:00:00Z")
        self.assertIn("No messages", result)

    def test_channel_auto_creation_on_post(self):
        """Channel file created on first post, no explicit setup needed."""
        channel_post("new-channel", "first!", agent_name="bot")
        result = channel_read("new-channel")
        self.assertIn("first!", result)


class TestJoinLeave(unittest.TestCase):
    """Test join/leave updates presence and channel log."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_join_adds_event_to_log(self):
        channel_join("dev", agent_name="agent-a")
        result = channel_read("dev")
        self.assertIn("joined", result)

    def test_leave_adds_event_to_log(self):
        channel_join("dev", agent_name="agent-a")
        channel_leave("dev", agent_name="agent-a")
        result = channel_read("dev")
        self.assertIn("left", result)

    def test_multiple_agents_same_channel(self):
        channel_join("dev", agent_name="agent-a")
        channel_join("dev", agent_name="agent-b")
        conn = _open_db()
        try:
            rows = conn.execute(
                "SELECT agent_id FROM memberships WHERE channel = 'dev'"
            ).fetchall()
            agents = {r["agent_id"] for r in rows}
            self.assertIn("agent-a", agents)
            self.assertIn("agent-b", agents)
        finally:
            conn.close()


class TestWho(unittest.TestCase):
    """Test who shows correct online agents and stale detection."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_no_agents(self):
        result = channel_who()
        self.assertIn("No agents", result)

    def test_online_agent(self):
        channel_join("dev", agent_name="agent-a")
        result = channel_who()
        self.assertIn("agent-a", result)
        self.assertIn("online", result)
        self.assertIn("#dev", result)

    def test_idle_presence(self):
        """Agent last_seen 10 minutes ago should show as idle."""
        idle_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        conn = _open_db()
        try:
            conn.execute(
                "INSERT INTO presence (agent_id, status, last_seen, joined_at) "
                "VALUES ('idle-agent', 'online', ?, ?)",
                (idle_time, idle_time),
            )
            conn.execute(
                "INSERT INTO memberships (agent_id, channel, joined_at) "
                "VALUES ('idle-agent', 'dev', ?)",
                (idle_time,),
            )
            conn.commit()
        finally:
            conn.close()

        result = channel_who()
        self.assertIn("idle-agent", result)
        self.assertIn("idle", result)

    def test_away_presence(self):
        """Agent last_seen 60 minutes ago should show as away."""
        away_time = (datetime.now(timezone.utc) - timedelta(minutes=60)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        conn = _open_db()
        try:
            conn.execute(
                "INSERT INTO presence (agent_id, status, last_seen, joined_at) "
                "VALUES ('away-agent', 'online', ?, ?)",
                (away_time, away_time),
            )
            conn.execute(
                "INSERT INTO memberships (agent_id, channel, joined_at) "
                "VALUES ('away-agent', 'dev', ?)",
                (away_time,),
            )
            conn.commit()
        finally:
            conn.close()

        result = channel_who()
        self.assertIn("away-agent", result)
        self.assertIn("away", result)

    def test_multiple_channels_shown(self):
        channel_join("dev", agent_name="agent-a")
        channel_join("eval", agent_name="agent-a")
        result = channel_who()
        self.assertIn("#dev", result)
        self.assertIn("#eval", result)

    def test_who_after_leave(self):
        channel_join("dev", agent_name="agent-a")
        channel_leave("dev", agent_name="agent-a")
        result = channel_who()
        self.assertIn("No agents", result)


class TestLegacyMigration(unittest.TestCase):
    """Test migration from _presence.json to SQLite."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_legacy_json_migrated(self):
        """Old _presence.json data should be migrated to SQLite on first access."""
        data_dir = Path(self.tmpdir) / "project" / ".synapt" / "recall"
        channels_dir = data_dir / "channels"
        channels_dir.mkdir(parents=True, exist_ok=True)

        legacy_data = {
            "agent-old": {
                "channels": ["dev", "eval"],
                "joined_at": "2026-03-16T08:00:00Z",
                "last_seen": "2026-03-16T09:00:00Z",
            }
        }
        legacy_path = channels_dir / "_presence.json"
        legacy_path.write_text(json.dumps(legacy_data), encoding="utf-8")

        # Opening the DB triggers migration
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT * FROM presence WHERE agent_id ='agent-old'"
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["last_seen"], "2026-03-16T09:00:00Z")

            mems = conn.execute(
                "SELECT channel FROM memberships WHERE agent_id ='agent-old' "
                "ORDER BY channel"
            ).fetchall()
            channels = [r["channel"] for r in mems]
            self.assertEqual(channels, ["dev", "eval"])
        finally:
            conn.close()

        # Legacy file should be deleted
        self.assertFalse(legacy_path.exists())

    def test_corrupt_legacy_json_deleted(self):
        data_dir = Path(self.tmpdir) / "project" / ".synapt" / "recall"
        channels_dir = data_dir / "channels"
        channels_dir.mkdir(parents=True, exist_ok=True)

        legacy_path = channels_dir / "_presence.json"
        legacy_path.write_text("not valid json", encoding="utf-8")

        # Should not crash
        conn = _open_db()
        try:
            row = conn.execute("SELECT COUNT(*) FROM presence").fetchone()
            self.assertEqual(row[0], 0)
        finally:
            conn.close()

        # Corrupt file should be deleted
        self.assertFalse(legacy_path.exists())

    def test_no_legacy_file_is_fine(self):
        """No _presence.json -- should work without errors."""
        conn = _open_db()
        try:
            row = conn.execute("SELECT COUNT(*) FROM presence").fetchone()
            self.assertEqual(row[0], 0)
        finally:
            conn.close()


class TestMessageFormatValidation(unittest.TestCase):
    """Test message format in the JSONL files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_message_has_required_fields(self):
        channel_post("dev", "test msg", agent_name="bot")
        data_dir = Path(self.tmpdir) / "project" / ".synapt" / "recall"
        path = data_dir / "channels" / "dev.jsonl"
        with open(path, encoding="utf-8") as f:
            line = f.readline().strip()
        data = json.loads(line)
        self.assertIn("timestamp", data)
        self.assertIn("from", data)
        self.assertIn("channel", data)
        self.assertIn("type", data)
        self.assertIn("body", data)
        self.assertEqual(data["type"], "message")
        self.assertEqual(data["from"], "bot")
        self.assertEqual(data["channel"], "dev")
        self.assertEqual(data["body"], "test msg")

    def test_join_event_format(self):
        channel_join("dev", agent_name="bot")
        data_dir = Path(self.tmpdir) / "project" / ".synapt" / "recall"
        path = data_dir / "channels" / "dev.jsonl"
        with open(path, encoding="utf-8") as f:
            line = f.readline().strip()
        data = json.loads(line)
        self.assertEqual(data["type"], "join")
        self.assertIn("joined", data["body"])

    def test_leave_event_format(self):
        channel_join("dev", agent_name="bot")
        channel_leave("dev", agent_name="bot")
        data_dir = Path(self.tmpdir) / "project" / ".synapt" / "recall"
        path = data_dir / "channels" / "dev.jsonl"
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        # Second line should be the leave event
        data = json.loads(lines[1].strip())
        self.assertEqual(data["type"], "leave")
        self.assertIn("left", data["body"])

    def test_timestamp_is_iso8601(self):
        channel_post("dev", "ts check", agent_name="bot")
        data_dir = Path(self.tmpdir) / "project" / ".synapt" / "recall"
        path = data_dir / "channels" / "dev.jsonl"
        with open(path, encoding="utf-8") as f:
            line = f.readline().strip()
        data = json.loads(line)
        ts = data["timestamp"]
        # Should be parseable as ISO 8601
        self.assertTrue(ts.endswith("Z"))
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        self.assertIsNotNone(dt)


class TestReadMessages(unittest.TestCase):
    """Test the internal _read_messages helper."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "test.jsonl"

    def test_read_empty_file(self):
        self.path.write_text("")
        msgs = _read_messages(self.path)
        self.assertEqual(msgs, [])

    def test_read_with_corrupt_line(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": "2026-03-16T10:00:00Z", "from": "a",
                "channel": "dev", "type": "message", "body": "good"
            }) + "\n")
            f.write("not json\n")
            f.write(json.dumps({
                "timestamp": "2026-03-16T10:01:00Z", "from": "b",
                "channel": "dev", "type": "message", "body": "also good"
            }) + "\n")
        msgs = _read_messages(self.path)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].body, "good")
        self.assertEqual(msgs[1].body, "also good")

    def test_read_nonexistent_file(self):
        msgs = _read_messages(Path(self.tmpdir) / "nope.jsonl")
        self.assertEqual(msgs, [])


class TestConcurrentAccess(unittest.TestCase):
    """Test that SQLite handles concurrent access safely."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_multiple_agents_join_and_post(self):
        """Simulate multiple agents joining and posting without conflicts."""
        for i in range(5):
            channel_join("dev", agent_name=f"agent-{i}")
            channel_post("dev", f"hello from agent-{i}", agent_name=f"agent-{i}")

        result = channel_read("dev", limit=50)
        for i in range(5):
            self.assertIn(f"agent-{i}", result)

        result = channel_who()
        for i in range(5):
            self.assertIn(f"agent-{i}", result)

    def test_open_db_multiple_times(self):
        """Opening the DB multiple times should not corrupt state."""
        channel_join("dev", agent_name="agent-a")

        # Open DB multiple times and check state
        for _ in range(3):
            conn = _open_db()
            try:
                row = conn.execute(
                    "SELECT * FROM presence WHERE agent_id ='agent-a'"
                ).fetchone()
                self.assertIsNotNone(row)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
