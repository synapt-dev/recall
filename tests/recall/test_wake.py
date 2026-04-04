"""Tests for synapt.recall.wake -- agent-side wake consumer (#525)."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.channel import (
    channel_ack_wakes,
    channel_directive,
    channel_join,
    channel_post,
    channel_read_wakes,
)
from synapt.recall.wake import WakeConsumer, _coalesce_wakes


def _patch_data_dir(tmpdir):
    """Return a patcher for project_data_dir targeting a temp directory."""
    data_dir = Path(tmpdir) / "project" / ".synapt" / "recall"
    return patch(
        "synapt.recall.channel.project_data_dir",
        return_value=data_dir,
    )


class TestCoalesceWakes(unittest.TestCase):
    """Unit tests for the _coalesce_wakes helper."""

    def test_empty_input(self):
        self.assertEqual(_coalesce_wakes([]), [])

    def test_single_wake_passes_through(self):
        wake = {
            "seq": 1,
            "target": "channel:dev",
            "reason": "channel_activity",
            "priority": 1,
            "source": "s_writer",
            "payload": {"message_id": "m_1"},
            "created": "2026-04-04T00:00:00Z",
        }
        result = _coalesce_wakes([wake])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["max_seq"], 1)
        self.assertEqual(result[0]["coalesced_count"], 1)
        self.assertEqual(result[0]["message_ids"], ["m_1"])

    def test_same_target_coalesces(self):
        wakes = [
            {
                "seq": 1, "target": "channel:dev", "reason": "channel_activity",
                "priority": 1, "source": "s_a", "payload": {"message_id": "m_1"},
                "created": "2026-04-04T00:00:00Z",
            },
            {
                "seq": 2, "target": "channel:dev", "reason": "channel_activity",
                "priority": 1, "source": "s_b", "payload": {"message_id": "m_2"},
                "created": "2026-04-04T00:01:00Z",
            },
        ]
        result = _coalesce_wakes(wakes)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["coalesced_count"], 2)
        self.assertEqual(result[0]["max_seq"], 2)
        self.assertIn("m_1", result[0]["message_ids"])
        self.assertIn("m_2", result[0]["message_ids"])

    def test_higher_priority_wins(self):
        wakes = [
            {
                "seq": 1, "target": "channel:dev", "reason": "channel_activity",
                "priority": 1, "source": "s_a", "payload": {"message_id": "m_1"},
                "created": "2026-04-04T00:00:00Z",
            },
            {
                "seq": 2, "target": "channel:dev", "reason": "directive",
                "priority": 3, "source": "s_admin", "payload": {"message_id": "m_2"},
                "created": "2026-04-04T00:01:00Z",
            },
        ]
        result = _coalesce_wakes(wakes)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["reason"], "directive")
        self.assertEqual(result[0]["priority"], 3)
        self.assertEqual(result[0]["source"], "s_admin")

    def test_different_targets_kept_separate(self):
        wakes = [
            {
                "seq": 1, "target": "channel:dev", "reason": "channel_activity",
                "priority": 1, "source": "s_a", "payload": {"message_id": "m_1"},
                "created": "2026-04-04T00:00:00Z",
            },
            {
                "seq": 2, "target": "agent:s_apollo", "reason": "mention",
                "priority": 2, "source": "s_b", "payload": {"message_id": "m_2"},
                "created": "2026-04-04T00:01:00Z",
            },
        ]
        result = _coalesce_wakes(wakes)
        self.assertEqual(len(result), 2)
        targets = {w["target"] for w in result}
        self.assertEqual(targets, {"channel:dev", "agent:s_apollo"})

    def test_duplicate_message_ids_deduped(self):
        wakes = [
            {
                "seq": 1, "target": "channel:dev", "reason": "channel_activity",
                "priority": 1, "source": "s_a", "payload": {"message_id": "m_1"},
                "created": "2026-04-04T00:00:00Z",
            },
            {
                "seq": 2, "target": "channel:dev", "reason": "mention",
                "priority": 2, "source": "s_a", "payload": {"message_id": "m_1"},
                "created": "2026-04-04T00:00:00Z",
            },
        ]
        result = _coalesce_wakes(wakes)
        self.assertEqual(result[0]["message_ids"], ["m_1"])

    def test_missing_message_id_in_payload(self):
        wakes = [
            {
                "seq": 1, "target": "channel:dev", "reason": "channel_activity",
                "priority": 1, "source": "s_a", "payload": {},
                "created": "2026-04-04T00:00:00Z",
            },
        ]
        result = _coalesce_wakes(wakes)
        self.assertEqual(result[0]["message_ids"], [])


class TestWakeConsumer(unittest.TestCase):
    """Integration tests for WakeConsumer using real transport."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()
        # Set up an agent with a channel membership
        channel_join("dev", agent_name="s_apollo", display_name="Apollo")

    def tearDown(self):
        self._patcher.stop()

    def test_poll_returns_empty_when_no_wakes(self):
        consumer = WakeConsumer(agent_name="s_apollo")
        # Ack any wakes from the join event
        raw = channel_read_wakes(consumer.targets)
        if raw:
            channel_ack_wakes(max(w["seq"] for w in raw))
            consumer._cursor = max(w["seq"] for w in raw)

        wakes = consumer.poll()
        self.assertEqual(wakes, [])

    def test_poll_returns_coalesced_priority_ordered(self):
        consumer = WakeConsumer(agent_name="s_apollo")
        # Ack existing wakes from join
        raw = channel_read_wakes(consumer.targets)
        if raw:
            max_seq = max(w["seq"] for w in raw)
            channel_ack_wakes(max_seq)
            consumer._cursor = max_seq

        # Post 2 regular messages, then a directive
        channel_post("dev", "hello", agent_name="s_writer")
        channel_post("dev", "world", agent_name="s_writer")
        channel_directive("dev", "review now", to="s_apollo", agent_name="s_admin")

        wakes = consumer.poll()
        # Should have directive (agent target) and channel wakes
        self.assertGreater(len(wakes), 0)
        # First wake should be highest priority
        priorities = [w["priority"] for w in wakes]
        self.assertEqual(priorities, sorted(priorities, reverse=True))

    def test_poll_advances_with_cursor(self):
        consumer = WakeConsumer(agent_name="s_apollo")
        channel_post("dev", "first", agent_name="s_writer")

        wakes1 = consumer.poll()
        self.assertGreater(len(wakes1), 0)
        max_seq = max(w["max_seq"] for w in wakes1)
        consumer.ack(max_seq)

        # Second poll should not return the same wakes
        wakes2 = consumer.poll()
        self.assertEqual(wakes2, [])

        # New message should appear
        channel_post("dev", "second", agent_name="s_writer")
        wakes3 = consumer.poll()
        self.assertGreater(len(wakes3), 0)

    def test_ack_returns_deleted_count(self):
        consumer = WakeConsumer(agent_name="s_apollo")
        channel_post("dev", "msg1", agent_name="s_writer")
        channel_post("dev", "msg2", agent_name="s_writer")

        wakes = consumer.poll()
        max_seq = max(w["max_seq"] for w in wakes)
        deleted = consumer.ack(max_seq)
        self.assertGreater(deleted, 0)
        self.assertEqual(consumer.cursor, max_seq)

    def test_ack_does_not_delete_other_targets(self):
        consumer = WakeConsumer(agent_name="s_apollo")
        raw = channel_read_wakes(consumer.targets)
        if raw:
            max_seq = max(w["seq"] for w in raw)
            channel_ack_wakes(max_seq, targets=consumer.targets)
            consumer._cursor = max_seq

        channel_directive("dev", "review now", to="s_apollo", agent_name="s_admin")
        channel_directive("dev", "review now", to="s_other", agent_name="s_admin")

        wakes = consumer.poll()
        max_seq = max(w["max_seq"] for w in wakes)
        deleted = consumer.ack(max_seq)

        self.assertGreater(deleted, 0)
        self.assertEqual(channel_read_wakes("agent:s_apollo"), [])
        other_wakes = channel_read_wakes("agent:s_other")
        self.assertEqual(len(other_wakes), 1)
        self.assertEqual(other_wakes[0]["target"], "agent:s_other")

    def test_ack_noop_when_cursor_not_advanced(self):
        consumer = WakeConsumer(agent_name="s_apollo")
        deleted = consumer.ack(0)
        self.assertEqual(deleted, 0)

    def test_processing_excludes_target_from_poll(self):
        consumer = WakeConsumer(agent_name="s_apollo")
        # Ack existing
        raw = channel_read_wakes(consumer.targets)
        if raw:
            max_seq = max(w["seq"] for w in raw)
            channel_ack_wakes(max_seq)
            consumer._cursor = max_seq

        channel_post("dev", "new msg", agent_name="s_writer")

        with consumer.processing("channel:dev"):
            wakes = consumer.poll()
            # channel:dev should be excluded
            targets_in_poll = {w["target"] for w in wakes}
            self.assertNotIn("channel:dev", targets_in_poll)

        # After context manager exits, target is available again
        wakes = consumer.poll()
        has_dev = any(w["target"] == "channel:dev" for w in wakes)
        self.assertTrue(has_dev)

    def test_processing_raises_on_overlap(self):
        consumer = WakeConsumer(agent_name="s_apollo")

        with consumer.processing("channel:dev"):
            with self.assertRaises(RuntimeError):
                with consumer.processing("channel:dev"):
                    pass

    def test_processing_thread_safety(self):
        consumer = WakeConsumer(agent_name="s_apollo")
        errors = []
        barrier = threading.Barrier(2, timeout=5)

        def worker(target):
            try:
                barrier.wait()
                with consumer.processing(target):
                    pass
            except RuntimeError as e:
                errors.append(e)

        # Two different targets should NOT conflict
        t1 = threading.Thread(target=worker, args=("channel:dev",))
        t2 = threading.Thread(target=worker, args=("agent:s_apollo",))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        self.assertEqual(errors, [])

    def test_active_targets_snapshot(self):
        consumer = WakeConsumer(agent_name="s_apollo")
        self.assertEqual(consumer.active_targets, frozenset())

        with consumer.processing("channel:dev"):
            self.assertIn("channel:dev", consumer.active_targets)

        self.assertEqual(consumer.active_targets, frozenset())

    def test_refresh_targets_after_join(self):
        consumer = WakeConsumer(agent_name="s_apollo")
        initial = consumer.targets
        self.assertIn("channel:dev", initial)

        channel_join("eval", agent_name="s_apollo")
        refreshed = consumer.refresh_targets()
        self.assertIn("channel:eval", refreshed)

    def test_targets_include_agent_and_channels(self):
        consumer = WakeConsumer(agent_name="s_apollo")
        targets = consumer.targets
        self.assertIn("agent:s_apollo", targets)
        self.assertIn("channel:dev", targets)

    def test_coalescing_merges_channel_wakes(self):
        """Multiple messages to same channel coalesce into one wake."""
        consumer = WakeConsumer(agent_name="s_apollo")
        # Ack existing
        raw = channel_read_wakes(consumer.targets)
        if raw:
            max_seq = max(w["seq"] for w in raw)
            channel_ack_wakes(max_seq)
            consumer._cursor = max_seq

        channel_post("dev", "msg1", agent_name="s_writer")
        channel_post("dev", "msg2", agent_name="s_writer")
        channel_post("dev", "msg3", agent_name="s_writer")

        wakes = consumer.poll()
        dev_wakes = [w for w in wakes if w["target"] == "channel:dev"]
        self.assertEqual(len(dev_wakes), 1)
        self.assertEqual(dev_wakes[0]["coalesced_count"], 3)
        self.assertEqual(len(dev_wakes[0]["message_ids"]), 3)


if __name__ == "__main__":
    unittest.main()
