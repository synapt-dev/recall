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
    _channel_path,
    _channels_dir,
    _db_path,
    _open_db,
    _read_messages,
    _agent_status,
    _reap_stale_agents,
    _resolve_griptree,
    _resolve_target_id,
    channel_join,
    channel_leave,
    channel_post,
    channel_read,
    channel_read_message,
    channel_who,
    channel_heartbeat,
    channel_unread,
    channel_unread_read,
    channel_pin,
    channel_unpin,
    channel_directive,
    channel_mute,
    channel_unmute,
    channel_kick,
    channel_broadcast,
    channel_agents_json,
    channel_list_channels,
    channel_messages_json,
    channel_search,
    channel_rename,
    channel_claim,
    channel_unclaim,
    channel_claim_intent,
    is_claimed,
    check_directives,
)


def _patch_data_dir(tmpdir):
    """Return a patcher for project_data_dir targeting a temp directory."""
    data_dir = Path(tmpdir) / "project" / ".synapt" / "recall"
    return patch(
        "synapt.recall.channel.project_data_dir",
        return_value=data_dir,
    )


class TestResolveGriptree(unittest.TestCase):
    """Test _resolve_griptree edge cases."""

    def test_cwd_at_gripspace_root_no_trailing_dot(self):
        """When cwd IS the gripspace root, griptree should be just the name, not 'name/.'."""
        tmpdir = tempfile.mkdtemp()
        gripspace = Path(tmpdir) / "myproject"
        data_dir = gripspace / ".synapt" / "recall"
        data_dir.mkdir(parents=True)

        with patch("synapt.recall.channel.project_data_dir", return_value=data_dir), \
             patch("synapt.recall.channel.Path") as MockPath:
            # Make Path.cwd() return the gripspace root
            MockPath.cwd.return_value = gripspace
            # Path(".") still needs to work for relative_to comparison
            MockPath.side_effect = Path
            result = _resolve_griptree()

        self.assertEqual(result, gripspace.name)
        self.assertNotIn("/.", result)


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

    def test_round_trip_with_attachments(self):
        msg = ChannelMessage(
            timestamp="2026-03-16T10:00:00Z",
            from_agent="test",
            channel="eval",
            type="message",
            body="",
            attachments=["attachments/m_abc12345/file.png"],
        )
        restored = ChannelMessage.from_dict(msg.to_dict())
        self.assertEqual(restored.attachments, msg.attachments)

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


class TestDashboardJsonWrappers(unittest.TestCase):
    """Test JSON-returning wrappers used by the web dashboard."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_channel_messages_json_prefers_current_display_name_over_raw_session_id(self):
        conn = _open_db()
        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            conn.execute(
                "INSERT INTO presence (agent_id, griptree, display_name, role, status, last_seen, joined_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("s_12345678", "main/synapt", "Atlas", "agent", "online", now, now),
            )
            conn.commit()
        finally:
            conn.close()

        path = _channel_path("dev")
        path.parent.mkdir(parents=True, exist_ok=True)
        msg = ChannelMessage(
            timestamp=now,
            from_agent="s_12345678",
            from_display="s_12345678",
            channel="dev",
            type="message",
            body="hello from atlas",
        )
        path.write_text(json.dumps(msg.to_dict()) + "\n")

        msgs = channel_messages_json("dev")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["from_display"], "Atlas")

    def test_channel_agents_json_collapses_fallback_human_identity_when_named_human_exists(self):
        older = "2026-03-30T07:15:00.000000Z"
        newer = "2026-03-30T07:18:00.000000Z"

        conn = _open_db()
        try:
            conn.execute(
                "INSERT INTO presence (agent_id, griptree, display_name, role, status, last_seen, joined_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("s_synapt", "synapt", "synapt", "human", "online", older, older),
            )
            conn.execute(
                "INSERT INTO memberships (agent_id, channel, joined_at) VALUES (?, ?, ?)",
                ("s_synapt", "dev", older),
            )
            conn.execute(
                "INSERT INTO presence (agent_id, griptree, display_name, role, status, last_seen, joined_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("dashboard", "synapt", "Layne", "human", "online", newer, newer),
            )
            conn.execute(
                "INSERT INTO memberships (agent_id, channel, joined_at) VALUES (?, ?, ?)",
                ("dashboard", "dev", newer),
            )
            conn.commit()
        finally:
            conn.close()

        agents = channel_agents_json()
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0]["display_name"], "Layne")
        self.assertEqual(agents[0]["agent_id"], "dashboard")

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

    def test_reap_releases_claims(self):
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
            conn.execute(
                "INSERT INTO claims (message_id, channel, claimed_by, claimed_at) "
                "VALUES ('m_stale', 'dev', 'stale-bot', ?)",
                (stale_time,),
            )
            conn.commit()

            _reap_stale_agents(conn)

            claim = conn.execute(
                "SELECT * FROM claims WHERE message_id = 'm_stale'"
            ).fetchone()
            self.assertIsNone(claim)
        finally:
            conn.close()

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

    def test_unread_read_returns_messages_and_resets_cursor(self):
        channel_join("dev", agent_name="agent-a")
        channel_post("dev", "msg1", agent_name="agent-b")
        channel_post("dev", "msg2", agent_name="agent-b")

        result = channel_unread_read(agent_name="agent-a")

        self.assertIn("## #dev (2 messages)", result)
        self.assertIn("msg1", result)
        self.assertIn("msg2", result)
        unread = channel_unread(agent_name="agent-a")
        self.assertEqual(unread["dev"], 0)

    def test_unread_read_reports_no_unread_messages(self):
        channel_join("dev", agent_name="agent-a")

        result = channel_unread_read(agent_name="agent-a")

        self.assertEqual(result, "No unread messages.")

    def test_unread_multiple_channels(self):
        channel_join("dev", agent_name="agent-a")
        channel_join("eval", agent_name="agent-a")

        channel_post("dev", "dev msg", agent_name="agent-b")
        channel_post("eval", "eval msg1", agent_name="agent-b")
        channel_post("eval", "eval msg2", agent_name="agent-b")

        unread = channel_unread(agent_name="agent-a")
        self.assertEqual(unread["dev"], 1)
        self.assertEqual(unread["eval"], 2)

    def test_unread_read_low_preserves_actionable_mention(self):
        channel_join("dev", agent_name="agent-a", display_name="Apollo")
        long_message = ("x" * 220) + " @Apollo please take #371"
        channel_post("dev", long_message, agent_name="agent-b")

        result = channel_unread_read(agent_name="agent-a")

        self.assertIn("@Apollo please take #371", result)
        self.assertNotIn("...", result)


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


class TestUnpin(unittest.TestCase):
    """Test unpin functionality (#303)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def _post_and_get_id(self, channel, message):
        channel_join(channel, agent_name="s_admin")
        channel_post(channel, message, agent_name="s_admin")
        path = _channels_dir() / f"{channel}.jsonl"
        msgs = _read_messages(path)
        return msgs[-1].id

    def test_unpin_removes_pin(self):
        msg_id = self._post_and_get_id("dev", "pinned content")
        channel_pin("dev", msg_id, agent_name="s_admin")
        result = channel_read("dev", agent_name="s_admin")
        self.assertIn("[pin]", result)

        unpin_result = channel_unpin("dev", msg_id)
        self.assertIn("Unpinned", unpin_result)

        result_after = channel_read("dev", agent_name="s_admin")
        self.assertNotIn("[pin]", result_after)

    def test_unpin_nonexistent(self):
        channel_join("dev", agent_name="s_admin")
        result = channel_unpin("dev", "m_doesnotexist")
        self.assertIn("No pin found", result)

    def test_unpin_one_of_multiple(self):
        id1 = self._post_and_get_id("dev", "keep this pin")
        id2 = self._post_and_get_id("dev", "remove this pin")
        channel_pin("dev", id1, agent_name="s_admin")
        channel_pin("dev", id2, agent_name="s_admin")

        channel_unpin("dev", id2)

        result = channel_read("dev", agent_name="s_admin")
        self.assertIn("keep this pin", result)
        self.assertNotIn("remove this pin", result.split("##")[1] if "##" in result else "")


class TestShowPins(unittest.TestCase):
    """Test show_pins option for channel_read (#306)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def _post_and_pin(self, channel, body, agent="bot"):
        channel_post(channel, body, agent_name=agent)
        msgs = _read_messages(_channels_dir() / f"{channel}.jsonl")
        msg_id = [m for m in msgs if m.body == body][-1].id
        channel_pin(channel, msg_id, agent_name=agent)
        return msg_id

    def test_show_pins_true_by_default(self):
        self._post_and_pin("dev", "pinned rule")
        channel_post("dev", "regular msg", agent_name="bot")
        result = channel_read("dev")
        self.assertIn("[pin]", result)
        self.assertIn("pinned rule", result)
        self.assertIn("regular msg", result)

    def test_show_pins_false_hides_pins(self):
        self._post_and_pin("dev", "pinned rule")
        channel_post("dev", "regular msg", agent_name="bot")
        result = channel_read("dev", show_pins=False)
        self.assertNotIn("[pin]", result)
        self.assertNotIn("Pinned", result)
        self.assertIn("regular msg", result)

    def test_show_pins_false_still_shows_messages(self):
        self._post_and_pin("dev", "pinned content")
        channel_post("dev", "msg1", agent_name="a1")
        channel_post("dev", "msg2", agent_name="a2")
        result = channel_read("dev", show_pins=False)
        self.assertIn("msg1", result)
        self.assertIn("msg2", result)
        # The pinned message still appears as a regular message
        self.assertIn("pinned content", result)


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

    def test_post_copies_attachments_into_channel_store(self):
        src = Path(self.tmpdir) / "screenshot.png"
        src.write_text("fake-image")

        result = channel_post(
            "dev",
            "",
            agent_name="agent-a",
            attachment_paths=[str(src)],
        )

        msgs = _read_messages(_channels_dir() / "dev.jsonl")
        msg = [m for m in msgs if m.type == "message"][-1]
        self.assertEqual(len(msg.attachments), 1)
        relpath = msg.attachments[0]
        self.assertTrue(relpath.startswith("attachments/"))
        stored = _channels_dir() / relpath
        self.assertTrue(stored.exists())
        self.assertEqual(stored.read_text(), "fake-image")
        self.assertIn(relpath, result)

    def test_read_shows_attachment_paths(self):
        src = Path(self.tmpdir) / "diagram.txt"
        src.write_text("diagram")

        channel_post(
            "dev",
            "see attached",
            agent_name="agent-a",
            attachment_paths=[str(src)],
        )
        result = channel_read("dev", agent_name="agent-b")
        self.assertIn("attachments/", result)
        self.assertIn("diagram.txt", result)

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

    def test_who_deduplicates_same_griptree_and_display_name(self):
        """Multiple agents with same griptree+display_name show only the most recent."""
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        new_time = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        conn = _open_db()
        try:
            # Two agents, same griptree AND display_name (dead session scenario)
            for aid, ts in [("s_old_session", old_time), ("s_new_session", new_time)]:
                conn.execute(
                    "INSERT INTO presence (agent_id, griptree, display_name, status, last_seen, joined_at) "
                    "VALUES (?, 'synapt/synapt', 'synapt', 'online', ?, ?)",
                    (aid, ts, ts),
                )
                conn.execute(
                    "INSERT INTO memberships (agent_id, channel, joined_at) "
                    "VALUES (?, 'dev', ?)",
                    (aid, ts),
                )
            conn.commit()
        finally:
            conn.close()

        result = channel_who()
        # Only the newer session should appear
        self.assertIn("s_new_session", result)
        self.assertNotIn("s_old_session", result)

    def test_who_keeps_different_display_names(self):
        """Agents with same griptree but different display names should all show."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        conn = _open_db()
        try:
            for aid, name in [("s_agent1", "worker-1"), ("s_agent2", "worker-2")]:
                conn.execute(
                    "INSERT INTO presence (agent_id, griptree, display_name, status, last_seen, joined_at) "
                    "VALUES (?, 'synapt/synapt', ?, 'online', ?, ?)",
                    (aid, name, now, now),
                )
                conn.execute(
                    "INSERT INTO memberships (agent_id, channel, joined_at) "
                    "VALUES (?, 'dev', ?)",
                    (aid, now),
                )
            conn.commit()
        finally:
            conn.close()

        result = channel_who()
        self.assertIn("worker-1", result)
        self.assertIn("worker-2", result)

    def test_human_role_shown_in_who(self):
        """Human role is displayed in /who output."""
        channel_join("dev", agent_name="s_human", role="human")
        # Set distinct display name so dedup doesn't collapse them
        conn = _open_db()
        conn.execute("UPDATE presence SET display_name = 'layne' WHERE agent_id = 's_human'")
        conn.commit()
        conn.close()
        channel_join("dev", agent_name="s_bot")
        result = channel_who()
        self.assertIn("[human]", result)
        self.assertIn("layne", result)

    def test_human_role_shown_in_read(self):
        """Messages from humans get a [human] tag in channel_read."""
        channel_join("dev", agent_name="s_human", role="human")
        channel_post("dev", "hello from the user", agent_name="s_human")
        channel_join("dev", agent_name="s_bot")
        channel_post("dev", "hello from the bot", agent_name="s_bot")
        result = channel_read("dev", agent_name="s_reader")
        self.assertIn("[human]", result)
        # Bot message should not have [human] tag
        for line in result.split("\n"):
            if "hello from the bot" in line:
                self.assertNotIn("[human]", line)

    def test_role_defaults_to_agent(self):
        """Without explicit role, agents get role='agent'."""
        channel_join("dev", agent_name="s_default")
        conn = _open_db()
        row = conn.execute(
            "SELECT role FROM presence WHERE agent_id = ?", ("s_default",)
        ).fetchone()
        conn.close()
        self.assertEqual(row["role"], "agent")


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

        # Give each agent a distinct display_name so /who dedup doesn't collapse them
        conn = _open_db()
        try:
            for i in range(5):
                conn.execute(
                    "UPDATE presence SET display_name = ? WHERE agent_id = ?",
                    (f"agent-{i}", f"agent-{i}"),
                )
            conn.commit()
        finally:
            conn.close()

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


class TestDirective(unittest.TestCase):
    """Test directive messages targeted at specific agents."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_directive_posted(self):
        result = channel_directive("dev", "please review PR #42", to="s_target01", agent_name="admin")
        self.assertIn("PR #42", result)
        self.assertIn("s_target01", result)

    def test_directive_in_channel_read(self):
        channel_directive("dev", "check logs", to="s_target01", agent_name="admin")
        result = channel_read("dev", agent_name="s_target01")
        self.assertIn("[DIRECTIVE]", result)
        self.assertIn("check logs", result)

    def test_directive_lowercase_for_other_agents(self):
        channel_directive("dev", "check logs", to="s_target01", agent_name="admin")
        result = channel_read("dev", agent_name="other-agent")
        self.assertIn("[directive]", result)
        self.assertNotIn("[DIRECTIVE]", result)

    def test_directive_in_jsonl(self):
        channel_directive("dev", "do the thing", to="s_target01", agent_name="admin")
        msgs = _read_messages(_channels_dir() / "dev.jsonl")
        directive_msgs = [m for m in msgs if m.type == "directive"]
        self.assertEqual(len(directive_msgs), 1)
        self.assertEqual(directive_msgs[0].to, "s_target01")
        self.assertEqual(directive_msgs[0].body, "do the thing")

    def test_directive_remind_bridges_to_reminders(self):
        """When remind=True, directive also calls add_reminder."""
        with patch("synapt.recall.reminders.add_reminder") as mock_add:
            channel_directive(
                "dev", "deploy by EOD", to="s_target01",
                agent_name="admin", remind=True,
            )
            mock_add.assert_called_once()
            call_text = mock_add.call_args[0][0]
            self.assertIn("deploy by EOD", call_text)
            self.assertIn("admin", call_text)

    def test_directive_no_remind_by_default(self):
        """Default remind=False should not call add_reminder."""
        with patch("synapt.recall.reminders.add_reminder") as mock_add:
            channel_directive("dev", "no reminder", to="s_target01", agent_name="admin")
            mock_add.assert_not_called()


class TestReadDetailLevels(unittest.TestCase):
    """Test detail-level truncation behavior."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_low_truncates_non_actionable_message(self):
        channel_join("dev", agent_name="reader", display_name="Apollo")
        long_message = ("x" * 220) + " tail"
        channel_post("dev", long_message, agent_name="writer")
        message_id = _read_messages(_channel_path("dev"))[-1].id

        result = channel_read("dev", agent_name="reader", detail="low")

        self.assertIn("...", result)
        self.assertNotIn("tail", result)
        self.assertIn(message_id, result)
        self.assertIn("truncated 1 message(s)", result)
        self.assertIn("tok omitted", result)

    def test_low_preserves_long_message_that_mentions_reader(self):
        channel_join("dev", agent_name="reader", display_name="Apollo")
        long_message = ("x" * 220) + " @Apollo please review this"
        channel_post("dev", long_message, agent_name="writer")

        result = channel_read("dev", agent_name="reader", detail="low")

        self.assertIn("@Apollo please review this", result)
        self.assertNotIn("...", result)

    def test_low_preserves_directive_for_target_agent(self):
        channel_join("dev", agent_name="reader")
        long_message = ("x" * 220) + " urgent follow-up"
        channel_directive("dev", long_message, to="reader", agent_name="writer")

        result = channel_read("dev", agent_name="reader", detail="low")

        self.assertIn("urgent follow-up", result)
        self.assertNotIn("...", result)

    def test_low_summary_lists_only_truncated_messages(self):
        channel_join("dev", agent_name="reader", display_name="Apollo")
        channel_post("dev", "short message", agent_name="writer")
        channel_post("dev", ("x" * 220) + " tail", agent_name="writer")
        messages = _read_messages(_channel_path("dev"))
        short_id = messages[-2].id
        long_id = messages[-1].id

        result = channel_read("dev", agent_name="reader", detail="low")

        self.assertIn(long_id, result)
        self.assertNotIn(short_id, result)

    def test_read_message_returns_full_body_by_id(self):
        channel_join("dev", agent_name="reader", display_name="Apollo")
        long_message = ("x" * 220) + " tail"
        channel_post("dev", long_message, agent_name="writer")
        message_id = _read_messages(_channel_path("dev"))[-1].id

        result = channel_read_message(message_id, channel="dev")

        self.assertIn(f"[{message_id}]", result)
        self.assertIn(long_message, result)
        self.assertIn("Type: message", result)


class TestMuteUnmute(unittest.TestCase):
    """Test muting and unmuting agents."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_mute_filters_messages(self):
        channel_join("dev", agent_name="reader")
        channel_post("dev", "visible", agent_name="friend")
        channel_post("dev", "hidden", agent_name="noisy")
        channel_mute("noisy", "dev", agent_name="reader")
        result = channel_read("dev", agent_name="reader")
        self.assertIn("visible", result)
        self.assertNotIn("hidden", result)

    def test_unmute_restores_messages(self):
        channel_join("dev", agent_name="reader")
        channel_post("dev", "msg1", agent_name="noisy")
        channel_mute("noisy", "dev", agent_name="reader")
        channel_unmute("noisy", "dev", agent_name="reader")
        result = channel_read("dev", agent_name="reader")
        self.assertIn("msg1", result)

    def test_mute_stored_in_db(self):
        channel_mute("noisy", "dev", agent_name="reader")
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT * FROM mutes WHERE agent_id = 'noisy' AND channel = 'dev'"
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["muted_by"], "reader")
        finally:
            conn.close()

    def test_mute_by_display_name(self):
        """Muting by display name should resolve to the agent_id."""
        channel_join("dev", agent_name="s_noisy01")
        # Manually set display_name in presence
        conn = _open_db()
        try:
            conn.execute(
                "UPDATE presence SET display_name = 'NoisyBot' WHERE agent_id = 's_noisy01'"
            )
            conn.commit()
        finally:
            conn.close()

        result = channel_mute("NoisyBot", "dev", agent_name="reader")
        self.assertIn("s_noisy01", result)

    def test_mute_unknown_target_uses_raw(self):
        """Muting an unknown name stores it as-is (no crash)."""
        result = channel_mute("ghost", "dev", agent_name="reader")
        self.assertIn("ghost", result)

    def test_multiple_agents_can_mute_same_target(self):
        """Two agents muting the same target should not overwrite each other."""
        channel_post("dev", "noise", agent_name="noisy")
        channel_mute("noisy", "dev", agent_name="reader-a")
        channel_mute("noisy", "dev", agent_name="reader-b")
        # Both mutes should exist
        conn = _open_db()
        try:
            rows = conn.execute(
                "SELECT muted_by FROM mutes WHERE agent_id = 'noisy' AND channel = 'dev'"
            ).fetchall()
            muters = {r["muted_by"] for r in rows}
            self.assertEqual(muters, {"reader-a", "reader-b"})
        finally:
            conn.close()
        # Both readers should have noisy filtered
        result_a = channel_read("dev", agent_name="reader-a")
        self.assertNotIn("noise", result_a)
        result_b = channel_read("dev", agent_name="reader-b")
        self.assertNotIn("noise", result_b)


class TestKick(unittest.TestCase):
    """Test kicking agents from channels."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_kick_removes_membership(self):
        channel_join("dev", agent_name="target")
        channel_kick("target", "dev", agent_name="admin")
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT * FROM memberships WHERE agent_id = 'target' AND channel = 'dev'"
            ).fetchone()
            self.assertIsNone(row)
        finally:
            conn.close()

    def test_kick_posts_event(self):
        channel_join("dev", agent_name="target")
        channel_kick("target", "dev", agent_name="admin")
        result = channel_read("dev")
        self.assertIn("kicked", result)

    def test_kick_last_channel_removes_presence(self):
        channel_join("dev", agent_name="target")
        channel_kick("target", "dev", agent_name="admin")
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT * FROM presence WHERE agent_id = 'target'"
            ).fetchone()
            self.assertIsNone(row)
        finally:
            conn.close()

    def test_kick_one_of_two_channels_keeps_presence(self):
        channel_join("dev", agent_name="target")
        channel_join("eval", agent_name="target")
        channel_kick("target", "dev", agent_name="admin")
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT * FROM presence WHERE agent_id = 'target'"
            ).fetchone()
            self.assertIsNotNone(row)
        finally:
            conn.close()

    def test_kick_by_display_name(self):
        channel_join("dev", agent_name="s_target01")
        conn = _open_db()
        try:
            conn.execute(
                "UPDATE presence SET display_name = 'BadBot' WHERE agent_id = 's_target01'"
            )
            conn.commit()
        finally:
            conn.close()

        result = channel_kick("BadBot", "dev", agent_name="admin")
        self.assertIn("s_target01", result)


class TestRename(unittest.TestCase):
    """Test display name rename action (#128)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_rename_sets_display_name(self):
        channel_join("dev", agent_name="s_agent1")
        result = channel_rename("Apollo", agent_name="s_agent1")
        self.assertIn("Apollo", result)
        # Verify in DB
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT display_name FROM presence WHERE agent_id = 's_agent1'"
            ).fetchone()
            self.assertEqual(row["display_name"], "Apollo")
        finally:
            conn.close()

    def test_rename_reflected_in_who(self):
        channel_join("dev", agent_name="s_agent1")
        channel_rename("Apollo", agent_name="s_agent1")
        result = channel_who()
        self.assertIn("Apollo", result)

    def test_rename_reflected_in_post_return(self):
        channel_join("dev", agent_name="s_agent1")
        channel_rename("Apollo", agent_name="s_agent1")
        result = channel_post("dev", "hello world", agent_name="s_agent1")
        self.assertIn("Apollo", result)

    def test_rename_without_prior_join(self):
        """Rename creates presence row if agent hasn't joined yet."""
        result = channel_rename("Ghost", agent_name="s_new_agent")
        self.assertIn("Ghost", result)
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT display_name FROM presence WHERE agent_id = 's_new_agent'"
            ).fetchone()
            self.assertEqual(row["display_name"], "Ghost")
        finally:
            conn.close()

    def test_join_rejects_duplicate_display_name(self):
        channel_join("dev", agent_name="s_agent1", display_name="Apollo")

        result = channel_join("dev", agent_name="s_agent2", display_name="Apollo")

        self.assertIn("already in use", result)
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT display_name FROM presence WHERE agent_id = 's_agent2'"
            ).fetchone()
            self.assertIsNone(row)
        finally:
            conn.close()

    def test_join_rejects_casefold_duplicate_display_name(self):
        channel_join("dev", agent_name="s_agent1", display_name="Apollo")

        result = channel_join("dev", agent_name="s_agent2", display_name="apollo")

        self.assertIn("already in use", result)

    def test_rename_rejects_duplicate_display_name(self):
        channel_join("dev", agent_name="s_agent1", display_name="Apollo")
        channel_join("dev", agent_name="s_agent2", display_name="Sentinel")

        result = channel_rename("Apollo", agent_name="s_agent2")

        self.assertIn("already in use", result)
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT display_name FROM presence WHERE agent_id = 's_agent2'"
            ).fetchone()
            self.assertEqual(row["display_name"], "Sentinel")
        finally:
            conn.close()

    def test_display_name_can_be_reused_after_owner_leaves(self):
        channel_join("dev", agent_name="s_agent1", display_name="Apollo")
        channel_leave("dev", agent_name="s_agent1")

        result = channel_join("dev", agent_name="s_agent2", display_name="Apollo")

        self.assertIn("Joined #dev as Apollo", result)


class TestFromDisplay(unittest.TestCase):
    """Test from_display persistence in JSONL messages (#292)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_post_persists_from_display(self):
        """Posted messages store the display name in JSONL."""
        channel_join("dev", agent_name="s_agent1", display_name="Apollo")
        channel_post("dev", "hello", agent_name="s_agent1")
        path = _channels_dir() / "dev.jsonl"
        msgs = _read_messages(path)
        post_msgs = [m for m in msgs if m.type == "message"]
        self.assertEqual(len(post_msgs), 1)
        self.assertEqual(post_msgs[0].from_display, "Apollo")

    def test_join_persists_from_display(self):
        """Join events store the display name in JSONL."""
        channel_join("dev", agent_name="s_agent1", display_name="Sentinel")
        path = _channels_dir() / "dev.jsonl"
        msgs = _read_messages(path)
        join_msgs = [m for m in msgs if m.type == "join"]
        self.assertEqual(len(join_msgs), 1)
        self.assertEqual(join_msgs[0].from_display, "Sentinel")

    def test_read_prefers_from_display_over_presence(self):
        """channel_read uses from_display from JSONL, not just presence."""
        channel_join("dev", agent_name="s_agent1", display_name="OldName")
        channel_post("dev", "with old name", agent_name="s_agent1")
        # Change the display name in presence
        channel_rename("NewName", agent_name="s_agent1")
        # The message should still show OldName (persisted at post time)
        result = channel_read("dev", agent_name="s_agent1")
        self.assertIn("OldName", result)

    def test_old_messages_fallback_to_presence(self):
        """Messages without from_display fall back to presence lookup."""
        channel_join("dev", agent_name="s_agent1", display_name="Apollo")
        # Manually write a message without from_display (simulate old format)
        import json
        path = _channels_dir() / "dev.jsonl"
        old_msg = {
            "timestamp": "2026-03-22T00:00:00Z",
            "from": "s_agent1",
            "channel": "dev",
            "type": "message",
            "body": "old format message",
            "id": "m_oldformat",
        }
        with open(path, "a") as f:
            f.write(json.dumps(old_msg) + "\n")
        result = channel_read("dev", agent_name="s_agent1")
        # Should resolve via presence lookup to "Apollo"
        self.assertIn("Apollo", result)

    def test_from_display_omitted_when_empty(self):
        """Serialization omits from_display when empty for JSONL cleanliness."""
        msg = ChannelMessage(
            timestamp="2026-03-22", from_agent="s_x", channel="dev",
            type="message", body="test",
        )
        d = msg.to_dict()
        self.assertNotIn("from_display", d)

    def test_from_display_included_when_set(self):
        """Serialization includes from_display when non-empty."""
        msg = ChannelMessage(
            timestamp="2026-03-22", from_agent="s_x", from_display="Apollo",
            channel="dev", type="message", body="test",
        )
        d = msg.to_dict()
        self.assertEqual(d["from_display"], "Apollo")


class TestClaim(unittest.TestCase):
    """Test message claim mechanism (#141)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_first_claim_wins(self):
        channel_join("dev", agent_name="s_agent1")
        channel_post("dev", "create an issue", agent_name="s_agent1")
        msgs = _read_messages(_channels_dir() / "dev.jsonl")
        msg_id = [m for m in msgs if m.type == "message"][-1].id

        result1 = channel_claim(msg_id, "dev", agent_name="s_agent1")
        self.assertIn("you", result1.lower())

        result2 = channel_claim(msg_id, "dev", agent_name="s_agent2")
        self.assertIn("Already claimed", result2)

    def test_is_claimed_returns_claimant(self):
        channel_join("dev", agent_name="s_agent1")
        channel_post("dev", "do something", agent_name="s_agent1")
        msgs = _read_messages(_channels_dir() / "dev.jsonl")
        msg_id = [m for m in msgs if m.type == "message"][-1].id

        self.assertIsNone(is_claimed(msg_id))
        channel_claim(msg_id, "dev", agent_name="s_agent1")
        claim = is_claimed(msg_id)
        self.assertIsNotNone(claim)
        self.assertEqual(claim["claimed_by"], "s_agent1")

    def test_claimed_shows_in_read(self):
        channel_join("dev", agent_name="s_agent1")
        channel_post("dev", "handle this", agent_name="s_agent1")
        msgs = _read_messages(_channels_dir() / "dev.jsonl")
        msg_id = [m for m in msgs if m.type == "message"][-1].id

        channel_claim(msg_id, "dev", agent_name="s_agent1")
        result = channel_read("dev", agent_name="s_agent2")
        self.assertIn("[CLAIMED by", result)

    def test_broadcast_directive_skipped_if_claimed_by_other(self):
        """check_directives skips broadcast directives claimed by another agent."""
        channel_join("dev", agent_name="s_agent1")
        channel_join("dev", agent_name="s_agent2")
        # Both read to set cursors
        channel_read("dev", agent_name="s_agent1")
        channel_read("dev", agent_name="s_agent2")

        channel_directive("dev", "create an issue", to="*", agent_name="s_boss")
        msgs = _read_messages(_channels_dir() / "dev.jsonl")
        directive_msg = [m for m in msgs if m.type == "directive"][-1]

        # Agent1 claims it
        channel_claim(directive_msg.id, "dev", agent_name="s_agent1")

        # Agent1 still sees it, agent2 doesn't
        result1 = check_directives(agent_name="s_agent1")
        result2 = check_directives(agent_name="s_agent2")
        self.assertIn("create an issue", result1)
        self.assertEqual(result2, "")

    def test_broadcast_directive_auto_claimed_on_first_read(self):
        """First agent to read a broadcast directive auto-claims it."""
        channel_join("dev", agent_name="s_agent1")
        channel_join("dev", agent_name="s_agent2")
        channel_read("dev", agent_name="s_agent1")
        channel_read("dev", agent_name="s_agent2")

        channel_directive("dev", "handle this task", to="*", agent_name="s_boss")
        msgs = _read_messages(_channels_dir() / "dev.jsonl")
        directive_msg = [m for m in msgs if m.type == "directive"][-1]

        # Agent1 reads first — should auto-claim
        result1 = check_directives(agent_name="s_agent1")
        self.assertIn("handle this task", result1)

        # Directive is now claimed by agent1
        claim = is_claimed(directive_msg.id)
        self.assertIsNotNone(claim)
        self.assertEqual(claim["claimed_by"], "s_agent1")

        # Agent2 should NOT see it (claimed by agent1)
        result2 = check_directives(agent_name="s_agent2")
        self.assertNotIn("handle this task", result2)

    def test_unclaim_by_owner(self):
        channel_join("dev", agent_name="s_agent1")
        channel_post("dev", "task", agent_name="s_agent1")
        msgs = _read_messages(_channels_dir() / "dev.jsonl")
        msg_id = [m for m in msgs if m.type == "message"][-1].id

        channel_claim(msg_id, "dev", agent_name="s_agent1")
        result = channel_unclaim(msg_id, agent_name="s_agent1")
        self.assertIn("Released", result)
        self.assertIsNone(is_claimed(msg_id))

    def test_unclaim_by_non_owner_rejected(self):
        channel_join("dev", agent_name="s_agent1")
        channel_post("dev", "task", agent_name="s_agent1")
        msgs = _read_messages(_channels_dir() / "dev.jsonl")
        msg_id = [m for m in msgs if m.type == "message"][-1].id

        channel_claim(msg_id, "dev", agent_name="s_agent1")
        result = channel_unclaim(msg_id, agent_name="s_agent2")
        self.assertIn("Cannot unclaim", result)
        self.assertIsNotNone(is_claimed(msg_id))

    def test_second_agent_can_claim_after_stale_owner_times_out(self):
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        channel_join("dev", agent_name="s_agent1")
        channel_join("dev", agent_name="s_agent2")
        channel_post("dev", "task", agent_name="s_agent1")
        msgs = _read_messages(_channels_dir() / "dev.jsonl")
        msg_id = [m for m in msgs if m.type == "message"][-1].id
        channel_claim(msg_id, "dev", agent_name="s_agent1")

        conn = _open_db()
        try:
            conn.execute(
                "UPDATE presence SET last_seen = ?, status = 'online' WHERE agent_id = 's_agent1'",
                (stale_time,),
            )
            conn.commit()
        finally:
            conn.close()

        result = channel_claim(msg_id, "dev", agent_name="s_agent2")
        self.assertIn("you", result.lower())
        claim = is_claimed(msg_id)
        self.assertIsNotNone(claim)
        self.assertEqual(claim["claimed_by"], "s_agent2")

    def test_leave_releases_claims_in_that_channel(self):
        channel_join("dev", agent_name="s_agent1")
        channel_post("dev", "task", agent_name="s_agent1")
        msgs = _read_messages(_channels_dir() / "dev.jsonl")
        msg_id = [m for m in msgs if m.type == "message"][-1].id

        channel_claim(msg_id, "dev", agent_name="s_agent1")
        channel_leave("dev", agent_name="s_agent1")
        self.assertIsNone(is_claimed(msg_id))


class TestClaimIntent(unittest.TestCase):
    """Test intent claiming for issue/PR dedup (#175)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_first_intent_succeeds(self):
        channel_join("dev", agent_name="s_agent1")
        ok, msg = channel_claim_intent("filing issue for perf fix", agent_name="s_agent1")
        self.assertTrue(ok)
        self.assertIn("INTENT", msg)

    def test_duplicate_intent_blocked(self):
        channel_join("dev", agent_name="s_agent1")
        channel_join("dev", agent_name="s_agent2")
        ok1, _ = channel_claim_intent("filing issue for perf fix", agent_name="s_agent1")
        self.assertTrue(ok1)
        ok2, msg2 = channel_claim_intent("filing issue for perf fix", agent_name="s_agent2")
        self.assertFalse(ok2)
        self.assertIn("Already claimed", msg2)

    def test_different_intents_both_succeed(self):
        channel_join("dev", agent_name="s_agent1")
        channel_join("dev", agent_name="s_agent2")
        ok1, _ = channel_claim_intent("filing issue for perf fix", agent_name="s_agent1")
        ok2, _ = channel_claim_intent("creating PR for auth feature", agent_name="s_agent2")
        self.assertTrue(ok1)
        self.assertTrue(ok2)


class TestBroadcast(unittest.TestCase):
    """Test broadcasting to all channels."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_broadcast_to_multiple(self):
        channel_post("dev", "setup", agent_name="bot")
        channel_post("eval", "setup", agent_name="bot")
        result = channel_broadcast("attention everyone", agent_name="admin")
        self.assertIn("2 channel(s)", result)

        dev_read = channel_read("dev")
        self.assertIn("attention everyone", dev_read)
        eval_read = channel_read("eval")
        self.assertIn("attention everyone", eval_read)

    def test_broadcast_no_channels(self):
        result = channel_broadcast("hello?", agent_name="admin")
        self.assertIn("No channels", result)


class TestListChannels(unittest.TestCase):
    """Test listing all channels."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_no_channels(self):
        result = channel_list_channels()
        self.assertEqual(result, [])

    def test_channels_from_posts(self):
        channel_post("dev", "hi", agent_name="bot")
        channel_post("eval", "hi", agent_name="bot")
        result = channel_list_channels()
        self.assertEqual(result, ["dev", "eval"])

    def test_channels_sorted(self):
        channel_post("zeta", "hi", agent_name="bot")
        channel_post("alpha", "hi", agent_name="bot")
        result = channel_list_channels()
        self.assertEqual(result, ["alpha", "zeta"])


class TestResolveTargetId(unittest.TestCase):
    """Test _resolve_target_id display name → agent_id resolution."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_passthrough_agent_id(self):
        conn = _open_db()
        try:
            result = _resolve_target_id("s_abc12345", conn)
            self.assertEqual(result, "s_abc12345")
        finally:
            conn.close()

    def test_resolve_by_display_name(self):
        channel_join("dev", agent_name="s_agent01")
        conn = _open_db()
        try:
            conn.execute(
                "UPDATE presence SET display_name = 'MyBot' WHERE agent_id = 's_agent01'"
            )
            conn.commit()
            result = _resolve_target_id("MyBot", conn)
            self.assertEqual(result, "s_agent01")
        finally:
            conn.close()

    def test_resolve_by_griptree(self):
        channel_join("dev", agent_name="s_agent01")
        conn = _open_db()
        try:
            conn.execute(
                "UPDATE presence SET griptree = 'synapt/synapt' WHERE agent_id = 's_agent01'"
            )
            conn.commit()
            result = _resolve_target_id("synapt/synapt", conn)
            self.assertEqual(result, "s_agent01")
        finally:
            conn.close()

    def test_unknown_returns_raw(self):
        conn = _open_db()
        try:
            result = _resolve_target_id("nobody", conn)
            self.assertEqual(result, "nobody")
        finally:
            conn.close()


class TestChannelSearch(unittest.TestCase):
    """Test channel_search keyword matching."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_search_finds_matching_messages(self):
        channel_post("dev", "the eval is running", agent_name="bot")
        channel_post("dev", "hello world", agent_name="bot")
        channel_post("dev", "eval complete J=90%", agent_name="bot")
        results = channel_search("eval")
        self.assertEqual(len(results), 2)
        bodies = [r["body"] for r in results]
        self.assertIn("the eval is running", bodies)
        self.assertIn("eval complete J=90%", bodies)

    def test_search_no_results(self):
        channel_post("dev", "hello world", agent_name="bot")
        results = channel_search("nonexistent")
        self.assertEqual(results, [])

    def test_search_across_channels(self):
        channel_post("dev", "deploy started", agent_name="bot")
        channel_post("eval", "deploy complete", agent_name="bot")
        results = channel_search("deploy")
        self.assertEqual(len(results), 2)
        channels = {r["channel"] for r in results}
        self.assertEqual(channels, {"dev", "eval"})

    def test_search_respects_max_results(self):
        for i in range(20):
            channel_post("dev", f"match keyword {i}", agent_name="bot")
        results = channel_search("keyword", max_results=5)
        self.assertEqual(len(results), 5)

    def test_search_multi_term_scoring(self):
        channel_post("dev", "eval running", agent_name="bot")
        channel_post("dev", "eval running on modal", agent_name="bot")
        results = channel_search("eval modal")
        # "eval running on modal" matches both terms, should score higher
        self.assertEqual(results[0]["body"], "eval running on modal")

    def test_search_skips_join_leave(self):
        channel_join("dev", agent_name="bot")
        channel_post("dev", "hello", agent_name="bot")
        channel_leave("dev", agent_name="bot")
        results = channel_search("bot")
        # Join/leave messages mention "bot" but should be skipped
        # Only the "hello" message should NOT match "bot"
        self.assertEqual(len(results), 0)

    def test_search_includes_directives(self):
        channel_directive("dev", "check the deploy logs", to="s_target", agent_name="admin")
        results = channel_search("deploy")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["body"], "check the deploy logs")

    def test_search_returns_message_ids(self):
        channel_post("dev", "important message", agent_name="bot")
        results = channel_search("important")
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["message_id"].startswith("m_"))

    def test_search_empty_query(self):
        channel_post("dev", "hello", agent_name="bot")
        results = channel_search("")
        self.assertEqual(results, [])


class TestChannelSearch(unittest.TestCase):
    """Tests for channel_search with type filtering (#147)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_search_by_type_directive(self):
        channel_join("dev", agent_name="s_agent1")
        channel_post("dev", "regular message", agent_name="s_agent1")
        channel_directive("dev", "do this task", to="s_agent2", agent_name="s_agent1")
        results = channel_search("task", msg_type="directive")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["type"], "directive")

    def test_search_type_only_no_query(self):
        """Search with type filter but no query returns all of that type."""
        channel_join("dev", agent_name="s_agent1")
        channel_directive("dev", "first directive", to="*", agent_name="s_agent1")
        channel_directive("dev", "second directive", to="*", agent_name="s_agent1")
        channel_post("dev", "not a directive", agent_name="s_agent1")
        # Empty query but type filter — should return all directives
        results = channel_search("", msg_type="directive")
        self.assertEqual(len(results), 2)

    def test_search_includes_type_field(self):
        channel_join("dev", agent_name="s_agent1")
        channel_post("dev", "hello world", agent_name="s_agent1")
        results = channel_search("hello")
        self.assertEqual(len(results), 1)
        self.assertIn("type", results[0])
        self.assertEqual(results[0]["type"], "message")


class TestCheckDirectives(unittest.TestCase):
    """Tests for check_directives — fast PostToolUse hook function."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self._tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_returns_empty_when_not_joined(self):
        """No memberships → empty string (no noise)."""
        result = check_directives(agent_name="s_nobody")
        self.assertEqual(result, "")

    def test_returns_empty_when_no_directives(self):
        """Joined but no directives → empty string."""
        channel_join("dev", agent_name="s_agent1")
        channel_post("dev", "regular message", agent_name="s_other")
        # Read to set cursor past existing messages
        channel_read("dev", agent_name="s_agent1")
        result = check_directives(agent_name="s_agent1")
        self.assertEqual(result, "")

    def test_surfaces_unread_directive(self):
        """Directive targeted at agent appears in output."""
        channel_join("dev", agent_name="s_agent1")
        # Set cursor by reading
        channel_read("dev", agent_name="s_agent1")
        # Now post a directive after the cursor
        channel_directive("dev", "please review PR #117", to="s_agent1", agent_name="s_sender")
        result = check_directives(agent_name="s_agent1")
        self.assertIn("pending directive", result)
        self.assertIn("please review PR #117", result)
        self.assertIn("s_sender", result)

    def test_ignores_directive_for_other_agent(self):
        """Directive targeted at someone else is not surfaced."""
        channel_join("dev", agent_name="s_agent1")
        channel_read("dev", agent_name="s_agent1")
        channel_directive("dev", "not for you", to="s_other", agent_name="s_sender")
        result = check_directives(agent_name="s_agent1")
        self.assertEqual(result, "")

    def test_broadcast_directive_auto_claimed_by_first_reader(self):
        """Directive with to='*' is auto-claimed by first agent to read it."""
        channel_join("dev", agent_name="s_agent1")
        channel_join("dev", agent_name="s_agent2")
        channel_read("dev", agent_name="s_agent1")
        channel_read("dev", agent_name="s_agent2")
        channel_directive("dev", "everyone stop", to="*", agent_name="s_boss")
        # First reader gets it and auto-claims
        result1 = check_directives(agent_name="s_agent1")
        self.assertIn("everyone stop", result1)
        # Second reader doesn't see it (auto-claimed by first)
        result2 = check_directives(agent_name="s_agent2")
        self.assertNotIn("everyone stop", result2)

    def test_heartbeat_updates_presence(self):
        """check_directives piggybacks a heartbeat update."""
        channel_join("dev", agent_name="s_agent1")
        # Wait a moment so timestamps differ
        time.sleep(0.01)
        check_directives(agent_name="s_agent1")
        conn = _open_db()
        row = conn.execute(
            "SELECT status FROM presence WHERE agent_id = ?", ("s_agent1",)
        ).fetchone()
        conn.close()
        self.assertEqual(row["status"], "online")

    def test_mention_surfaces_in_check_directives(self):
        """@mention in a message is surfaced by check_directives."""
        channel_join("dev", agent_name="s_opus")
        # Set display name so @opus matches
        conn = _open_db()
        conn.execute("UPDATE presence SET display_name = 'opus' WHERE agent_id = 's_opus'")
        conn.commit()
        conn.close()
        channel_read("dev", agent_name="s_opus")  # Set cursor
        channel_post("dev", "hey @opus please review", agent_name="s_apollo")
        result = check_directives(agent_name="s_opus")
        self.assertIn("@mention", result)
        self.assertIn("please review", result)

    def test_self_mention_not_surfaced(self):
        """Agent's own @mentions don't show up as notifications."""
        channel_join("dev", agent_name="s_opus")
        conn = _open_db()
        conn.execute("UPDATE presence SET display_name = 'opus' WHERE agent_id = 's_opus'")
        conn.commit()
        conn.close()
        channel_read("dev", agent_name="s_opus")
        channel_post("dev", "I am @opus and I approve", agent_name="s_opus")
        result = check_directives(agent_name="s_opus")
        self.assertEqual(result, "")

    def test_mention_stored_in_db(self):
        """@mentions are stored in the mentions table."""
        channel_join("dev", agent_name="s_sender")
        channel_post("dev", "hey @apollo and @opus check this", agent_name="s_sender")
        conn = _open_db()
        rows = conn.execute("SELECT mentioned FROM mentions ORDER BY mentioned").fetchall()
        conn.close()
        mentioned = [r["mentioned"] for r in rows]
        self.assertIn("apollo", mentioned)
        self.assertIn("opus", mentioned)


if __name__ == "__main__":
    unittest.main()
