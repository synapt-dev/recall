"""Tests for synapt.recall.wake_bridge — wake-to-prompt bridge (#484)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.channel import (
    channel_directive,
    channel_join,
    channel_post,
    channel_read_wakes,
    channel_ack_wakes,
)
from synapt.recall.wake_bridge import (
    PromptAdapter, TmuxAdapter, WakeBridge, _PROMPT_TEMPLATES,
)


def _patch_data_dir(tmpdir):
    """Return a combined patcher for project_data_dir + disable global store."""
    data_dir = Path(tmpdir) / "project" / ".synapt" / "recall"
    patcher_data = patch(
        "synapt.recall.channel.project_data_dir",
        return_value=data_dir,
    )
    patcher_manifest = patch(
        "synapt.recall.channel._read_manifest_url",
        return_value=None,
    )

    class _CombinedPatcher:
        def start(self):
            patcher_data.start()
            patcher_manifest.start()
        def stop(self):
            patcher_manifest.stop()
            patcher_data.stop()
        def __enter__(self):
            self.start()
            return self
        def __exit__(self, *args):
            self.stop()

    return _CombinedPatcher()


class FakeAdapter:
    """Test adapter that records injected prompts."""

    def __init__(self, alive=True):
        self._alive = alive
        self.injections: list[tuple[str, str]] = []

    def is_alive(self, target: str) -> bool:
        return self._alive

    def inject_prompt(self, target: str, prompt: str) -> bool:
        self.injections.append((target, prompt))
        return True


class FakeDeadAdapter(FakeAdapter):
    def __init__(self):
        super().__init__(alive=False)


class TestBuildPrompt(unittest.TestCase):
    """Test prompt generation from wake reasons."""

    def test_channel_activity_prompt(self):
        wake = {"reason": "channel_activity", "source": "s_writer", "payload": {"channel": "dev"}}
        prompt = WakeBridge._build_prompt(wake)
        self.assertIn("unread", prompt)

    def test_directive_prompt_includes_source(self):
        wake = {"reason": "directive", "source": "s_admin", "payload": {"channel": "dev"}}
        prompt = WakeBridge._build_prompt(wake)
        self.assertIn("s_admin", prompt)

    def test_mention_prompt(self):
        wake = {"reason": "mention", "source": "s_writer", "payload": {"channel": "dev"}}
        prompt = WakeBridge._build_prompt(wake)
        self.assertIn("@mentioned", prompt)

    def test_user_action_prompt(self):
        wake = {"reason": "user_action", "source": "dashboard", "payload": {"channel": "dev"}}
        prompt = WakeBridge._build_prompt(wake)
        self.assertIn("Layne", prompt)

    def test_channel_substituted_in_prompt(self):
        wake = {"reason": "channel_activity", "source": "s_writer", "payload": {"channel": "eval"}}
        prompt = WakeBridge._build_prompt(wake)
        self.assertIn("#eval", prompt)
        self.assertNotIn("#dev", prompt)

    def test_unknown_reason_falls_back(self):
        wake = {"reason": "unknown_reason", "source": "x", "payload": {}}
        prompt = WakeBridge._build_prompt(wake)
        self.assertIn("unread", prompt)


class TestWakeBridge(unittest.TestCase):
    """Integration tests for WakeBridge with fake adapter."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()
        channel_join("dev", agent_name="s_apollo", display_name="Apollo")

    def tearDown(self):
        self._patcher.stop()

    def _clear_wakes(self):
        """Ack all existing wakes so tests start clean."""
        targets = ["agent:s_apollo", "channel:dev"]
        raw = channel_read_wakes(targets)
        if raw:
            channel_ack_wakes(max(w["seq"] for w in raw))

    def test_tick_injects_on_new_message(self):
        self._clear_wakes()
        adapter = FakeAdapter()
        bridge = WakeBridge(
            agents={"s_apollo": "synapt:apollo"},
            adapter=adapter,
        )
        channel_post("dev", "hello", agent_name="s_writer")

        count = bridge.tick()

        self.assertEqual(count, 1)
        self.assertEqual(len(adapter.injections), 1)
        target, prompt = adapter.injections[0]
        self.assertEqual(target, "synapt:apollo")
        self.assertIn("unread", prompt)

    def test_tick_no_wakes_no_injection(self):
        self._clear_wakes()
        adapter = FakeAdapter()
        bridge = WakeBridge(
            agents={"s_apollo": "synapt:apollo"},
            adapter=adapter,
        )

        count = bridge.tick()

        self.assertEqual(count, 0)
        self.assertEqual(len(adapter.injections), 0)

    def test_tick_skips_dead_agent(self):
        self._clear_wakes()
        adapter = FakeDeadAdapter()
        bridge = WakeBridge(
            agents={"s_apollo": "synapt:apollo"},
            adapter=adapter,
        )
        channel_post("dev", "hello", agent_name="s_writer")

        count = bridge.tick()

        self.assertEqual(count, 0)
        self.assertEqual(len(adapter.injections), 0)

    def test_tick_coalesces_multiple_messages(self):
        """Multiple messages → single prompt injection."""
        self._clear_wakes()
        adapter = FakeAdapter()
        bridge = WakeBridge(
            agents={"s_apollo": "synapt:apollo"},
            adapter=adapter,
        )
        channel_post("dev", "msg1", agent_name="s_writer")
        channel_post("dev", "msg2", agent_name="s_writer")
        channel_post("dev", "msg3", agent_name="s_writer")

        count = bridge.tick()

        self.assertEqual(count, 1)
        self.assertEqual(len(adapter.injections), 1)

    def test_tick_uses_highest_priority_wake(self):
        """Directive wake (priority 3) should produce directive prompt."""
        self._clear_wakes()
        adapter = FakeAdapter()
        bridge = WakeBridge(
            agents={"s_apollo": "synapt:apollo"},
            adapter=adapter,
        )
        channel_post("dev", "regular msg", agent_name="s_writer")
        channel_directive("dev", "do this now", to="s_apollo", agent_name="s_admin")

        count = bridge.tick()

        self.assertEqual(count, 1)
        # Should have used directive template (highest priority)
        _, prompt = adapter.injections[0]
        self.assertIn("directive", prompt.lower())

    def test_tick_acks_after_injection(self):
        """Wakes are acked after successful injection — no redelivery."""
        self._clear_wakes()
        adapter = FakeAdapter()
        bridge = WakeBridge(
            agents={"s_apollo": "synapt:apollo"},
            adapter=adapter,
        )
        channel_post("dev", "hello", agent_name="s_writer")

        bridge.tick()
        # Second tick should have nothing
        count = bridge.tick()

        self.assertEqual(count, 0)
        self.assertEqual(len(adapter.injections), 1)  # still just 1


class TestProtocol(unittest.TestCase):
    """Adapter Protocol conformance."""

    def test_tmux_adapter_satisfies_protocol(self):
        self.assertIsInstance(TmuxAdapter(), PromptAdapter)

    def test_fake_adapter_satisfies_protocol(self):
        self.assertIsInstance(FakeAdapter(), PromptAdapter)


if __name__ == "__main__":
    unittest.main()
