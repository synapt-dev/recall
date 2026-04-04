"""Wake-to-prompt bridge (#484).

Long-running daemon that watches wake_requests and injects prompts
into agent sessions via platform adapters (tmux first).

Usage::

    bridge = WakeBridge(agents={"apollo": "synapt:apollo"})
    bridge.run()  # blocks, polls every interval
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from synapt.recall.wake import WakeConsumer

logger = logging.getLogger("synapt.recall.wake_bridge")

# Maps wake reason → prompt template.  {channel} and {source} are
# substituted from the wake payload.
_PROMPT_TEMPLATES: dict[str, str] = {
    "user_action": (
        'Check #dev for unread messages using recall_channel('
        "action='unread', show_pins=false, detail='medium'). "
        "Layne posted — check immediately."
    ),
    "directive": (
        'Check #dev for unread messages using recall_channel('
        "action='unread', show_pins=false, detail='medium'). "
        "New directive from {source} — act on it."
    ),
    "mention": (
        'Check #dev for unread messages using recall_channel('
        "action='unread', show_pins=false, detail='medium'). "
        "You were @mentioned — check unread."
    ),
    "channel_activity": (
        'Check #dev for unread messages using recall_channel('
        "action='unread', show_pins=false, detail='medium'). "
        "If there are new messages, read them and act on any "
        "directives or assignments. If no unread, return empty."
    ),
}

_DEFAULT_PROMPT = _PROMPT_TEMPLATES["channel_activity"]


# ---------------------------------------------------------------------------
# Platform adapters
# ---------------------------------------------------------------------------

class TmuxAdapter:
    """Injects prompts into tmux panes.

    Targets are opaque strings in ``session:window`` format
    (e.g. ``"synapt:apollo"``).
    """

    def is_alive(self, target: str) -> bool:
        """Check if the tmux pane exists and is not dead."""
        try:
            result = subprocess.run(
                ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return False
            return result.stdout.strip() == "0"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def inject_prompt(self, target: str, prompt: str) -> bool:
        """Send a prompt string into the agent's tmux pane."""
        try:
            result = subprocess.run(
                ["tmux", "send-keys", "-t", target, prompt, "Enter"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

@dataclass
class AgentBinding:
    """Maps an agent to its WakeConsumer and platform target."""
    agent_name: str
    target: str  # opaque adapter target (e.g. "synapt:apollo")
    consumer: WakeConsumer = field(init=False)

    def __post_init__(self):
        self.consumer = WakeConsumer(agent_name=self.agent_name)


class WakeBridge:
    """Polls wake_requests and injects prompts into agent sessions.

    Args:
        agents: Mapping of agent_name → adapter target string.
        adapter: Platform adapter for prompt injection.
        poll_interval: Seconds between polls.
        project_dir: Override for synapt project directory.
    """

    def __init__(
        self,
        agents: dict[str, str],
        adapter: TmuxAdapter | None = None,
        poll_interval: float = 10.0,
        project_dir: Path | None = None,
    ) -> None:
        self._adapter = adapter or TmuxAdapter()
        self._poll_interval = poll_interval
        self._bindings: list[AgentBinding] = [
            AgentBinding(agent_name=name, target=target)
            for name, target in agents.items()
        ]

    def tick(self) -> int:
        """Run one poll cycle.  Returns number of prompts injected."""
        injected = 0
        for binding in self._bindings:
            wakes = binding.consumer.poll()
            if not wakes:
                continue

            # Check liveness before injecting
            if not self._adapter.is_alive(binding.target):
                logger.warning(
                    "Agent %s target %s is not alive — skipping %d wake(s)",
                    binding.agent_name, binding.target, len(wakes),
                )
                continue

            # Build prompt from highest-priority wake
            top_wake = wakes[0]  # already priority-sorted by WakeConsumer
            prompt = self._build_prompt(top_wake)

            # Inject
            ok = self._adapter.inject_prompt(binding.target, prompt)
            max_seq = max(w["max_seq"] for w in wakes)

            if ok:
                binding.consumer.ack(max_seq)
                injected += 1
                logger.info(
                    "Injected prompt: agent=%s target=%s reason=%s "
                    "priority=%d coalesced=%d seq_range=[%d..%d]",
                    binding.agent_name,
                    binding.target,
                    top_wake["reason"],
                    top_wake["priority"],
                    sum(w["coalesced_count"] for w in wakes),
                    min(w["seq"] for w in wakes),
                    max_seq,
                )
            else:
                logger.error(
                    "Failed to inject prompt: agent=%s target=%s",
                    binding.agent_name, binding.target,
                )

        return injected

    def run(self) -> None:
        """Run the bridge loop (blocks forever)."""
        logger.info(
            "Wake bridge started: %d agent(s), poll every %.0fs",
            len(self._bindings), self._poll_interval,
        )
        while True:
            try:
                self.tick()
            except Exception:
                logger.exception("Wake bridge tick failed")
            time.sleep(self._poll_interval)

    @staticmethod
    def _build_prompt(wake: dict) -> str:
        """Build a prompt string from a coalesced wake."""
        reason = wake.get("reason", "channel_activity")
        template = _PROMPT_TEMPLATES.get(reason, _DEFAULT_PROMPT)
        source = wake.get("source", "unknown")
        channel = wake.get("payload", {}).get("channel", "dev")
        return template.format(source=source, channel=channel)
