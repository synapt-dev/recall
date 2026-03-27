#!/usr/bin/env python3
"""spawn_watcher.py — Agent presence tracker and auto-recovery for gr spawn.

Monitors agent heartbeats via the #dev channel presence table and tmux pane
status. Posts alerts when agents go stale, and triggers mock restarts in
Phase 1.

Usage:
    python scripts/spawn_watcher.py [--config PATH] [--poll-interval 30] [--mock]

Reads .gitgrip/agents.toml for agent configuration. Polls every --poll-interval
seconds (default 30). In --mock mode (default for Phase 1), restarts print to
stdout instead of relaunching real agents.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("Error: Python 3.11+ required (for tomllib), or install tomli")
        sys.exit(1)


# --- Config ---

DEFAULT_CONFIG = ".gitgrip/agents.toml"
DEFAULT_POLL_INTERVAL = 30  # seconds
DEFAULT_HEARTBEAT_INTERVAL = 60
DEFAULT_TIMEOUT_THRESHOLD = 180
DEFAULT_RESTART_POLICY = "always"
DEFAULT_RESTART_DELAY = 5
DEFAULT_MAX_RESTARTS = 3
BACKOFF_MULTIPLIER = 2  # exponential backoff: delay * 2^attempt


def load_config(config_path: str) -> dict:
    """Load and parse agents.toml."""
    path = Path(config_path)
    if not path.exists():
        print(f"Error: Config not found at {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        return tomllib.load(f)


# --- tmux Integration ---

def get_tmux_session_name(config: dict) -> str:
    """Get tmux session name from config."""
    return config.get("spawn", {}).get("session_name", "synapt")


def get_tmux_windows(session: str) -> dict[str, dict]:
    """List tmux windows and their status. Returns {window_name: {active, pane_pid, pane_dead}}."""
    try:
        result = subprocess.run(
            ["tmux", "list-windows", "-t", session, "-F",
             "#{window_name}\t#{window_active}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return {}

        windows = {}
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            name = parts[0]
            active = parts[1] == "1" if len(parts) > 1 else False
            windows[name] = {"active": active}

        # Get pane details for each window
        for name in windows:
            try:
                pane_result = subprocess.run(
                    ["tmux", "list-panes", "-t", f"{session}:{name}", "-F",
                     "#{pane_pid}\t#{pane_dead}"],
                    capture_output=True, text=True, timeout=5,
                )
                if pane_result.returncode == 0:
                    pane_line = pane_result.stdout.strip().split("\n")[0]
                    pane_parts = pane_line.split("\t")
                    windows[name]["pane_pid"] = int(pane_parts[0]) if pane_parts[0] else None
                    windows[name]["pane_dead"] = pane_parts[1] == "1" if len(pane_parts) > 1 else False
            except (subprocess.TimeoutExpired, ValueError):
                windows[name]["pane_pid"] = None
                windows[name]["pane_dead"] = True

        return windows
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}


def is_process_alive(pid: int) -> bool:
    """Check if a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# --- Channel Presence ---

def get_agent_last_seen(recall_dir: Path) -> dict[str, float]:
    """Read agent presence timestamps from the channel SQLite database."""
    db_path = recall_dir / "channels.db"
    if not db_path.exists():
        return {}

    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT agent_id, display_name, updated_at FROM presence WHERE channel = 'dev'"
        )
        result = {}
        for agent_id, display_name, updated_at in cursor.fetchall():
            name = display_name or agent_id
            # Parse ISO timestamp to epoch
            try:
                dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                result[name.lower()] = dt.timestamp()
            except (ValueError, AttributeError):
                pass
        conn.close()
        return result
    except Exception as e:
        print(f"Warning: Could not read presence DB: {e}")
        return {}


# --- Event Logging ---

def log_event(event_path: Path, event: dict):
    """Append an event to spawn_events.jsonl."""
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with open(event_path, "a") as f:
        f.write(json.dumps(event) + "\n")


# --- Spawn State ---

def read_spawn_state(state_path: Path) -> dict:
    """Read spawn_state.json for intentional stop flags."""
    if not state_path.exists():
        return {}
    try:
        with open(state_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


# --- Alert Posting ---

def post_alert(message: str, recall_dir: Path):
    """Post an alert to #dev channel via the synapt CLI."""
    print(f"[ALERT] {message}")
    try:
        subprocess.run(
            [sys.executable, "-m", "synapt.recall.cli", "channel",
             "--action", "post", "--message", message],
            capture_output=True, text=True, timeout=10,
            cwd=str(recall_dir.parent.parent),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # Best effort — already printed to stdout


# --- Mock Restart ---

def mock_restart(agent_name: str, session: str, config: dict):
    """Simulate a restart in Phase 1 (echo instead of real launch)."""
    print(f"[MOCK RESTART] Would restart agent '{agent_name}' in tmux {session}:{agent_name}")
    print(f"  model: {config.get('model', 'claude-sonnet-4-6')}")
    print(f"  worktree: {config.get('worktree', 'main')}")
    print(f"  startup_prompt: {config.get('startup_prompt', 'N/A')}")


# --- Main Watch Loop ---

class SpawnWatcher:
    def __init__(self, config: dict, recall_dir: Path, mock: bool = True, poll_interval: int = DEFAULT_POLL_INTERVAL):
        self.config = config
        self.recall_dir = recall_dir
        self.mock = mock
        self.poll_interval = poll_interval
        self.session = get_tmux_session_name(config)
        self.agents = config.get("agents", {})
        self.restart_counts: dict[str, int] = {name: 0 for name in self.agents}
        self.event_path = recall_dir / "spawn_events.jsonl"
        self.state_path = recall_dir / "spawn_state.json"

    def get_timeout(self, agent_name: str) -> int:
        return self.agents[agent_name].get("timeout_threshold", DEFAULT_TIMEOUT_THRESHOLD)

    def get_restart_policy(self, agent_name: str) -> str:
        return self.agents[agent_name].get("restart_policy", DEFAULT_RESTART_POLICY)

    def get_max_restarts(self, agent_name: str) -> int:
        return self.agents[agent_name].get("max_restarts", DEFAULT_MAX_RESTARTS)

    def get_restart_delay(self, agent_name: str, attempt: int) -> float:
        base = self.agents[agent_name].get("restart_delay", DEFAULT_RESTART_DELAY)
        return base * (BACKOFF_MULTIPLIER ** attempt)

    def check_agent(self, agent_name: str, last_seen: dict, tmux_windows: dict) -> str | None:
        """Check a single agent. Returns alert message or None."""
        now = time.time()
        timeout = self.get_timeout(agent_name)

        # Check heartbeat staleness
        agent_ts = last_seen.get(agent_name.lower())
        heartbeat_stale = False
        if agent_ts is not None:
            silence = now - agent_ts
            if silence > timeout:
                heartbeat_stale = True

        # Check tmux window
        window = tmux_windows.get(agent_name, {})
        pane_dead = window.get("pane_dead", True) if window else True
        pane_pid = window.get("pane_pid")

        # No tmux window at all
        if agent_name not in tmux_windows:
            return None  # Agent not spawned — not our problem

        # Pane is dead — definite crash
        if pane_dead:
            # Check if this was an intentional stop
            state = read_spawn_state(self.state_path)
            if state.get(f"stop_{agent_name}"):
                return None  # Intentional stop, don't alert
            return f"{agent_name} pane is DEAD (crashed or exited)"

        # Pane alive but heartbeat stale
        if heartbeat_stale and pane_pid and is_process_alive(pane_pid):
            silence_s = int(now - agent_ts) if agent_ts else "unknown"
            return f"{agent_name} silent for {silence_s}s (threshold: {timeout}s) — process alive, may be thinking"

        return None

    def handle_alert(self, agent_name: str, message: str):
        """Handle an alert: log, post, and optionally restart."""
        log_event(self.event_path, {
            "event": "alert",
            "agent": agent_name,
            "message": message,
        })
        post_alert(message, self.recall_dir)

        # Check if restart is warranted
        if "DEAD" not in message:
            return  # Just a warning, don't restart

        policy = self.get_restart_policy(agent_name)
        max_restarts = self.get_max_restarts(agent_name)
        attempts = self.restart_counts.get(agent_name, 0)

        if policy == "never":
            return
        if policy == "once" and attempts >= 1:
            return
        if attempts >= max_restarts:
            fatal_msg = f"[FATAL] {agent_name} exceeded max restarts ({max_restarts}). Manual intervention needed."
            post_alert(fatal_msg, self.recall_dir)
            log_event(self.event_path, {"event": "fatal", "agent": agent_name})
            return

        # Restart with backoff
        delay = self.get_restart_delay(agent_name, attempts)
        print(f"[RECOVERY] Waiting {delay:.0f}s before restart (attempt {attempts + 1}/{max_restarts})")
        time.sleep(delay)

        self.restart_counts[agent_name] = attempts + 1
        log_event(self.event_path, {
            "event": "restart",
            "agent": agent_name,
            "attempt": attempts + 1,
            "max": max_restarts,
        })

        recovery_msg = f"[RECOVERY] {agent_name} restarted (attempt {attempts + 1}/{max_restarts})"
        post_alert(recovery_msg, self.recall_dir)

        if self.mock:
            mock_restart(agent_name, self.session, self.agents[agent_name])
        else:
            # Phase 2+: real restart logic goes here
            print(f"[REAL RESTART] Not implemented yet — use --mock for Phase 1")

    def run(self):
        """Main watch loop."""
        print(f"spawn_watcher starting — session={self.session}, agents={list(self.agents.keys())}")
        print(f"Poll interval: {self.poll_interval}s, mock={self.mock}")
        print(f"Event log: {self.event_path}")
        print()

        log_event(self.event_path, {
            "event": "watcher_start",
            "session": self.session,
            "agents": list(self.agents.keys()),
            "mock": self.mock,
        })

        try:
            while True:
                last_seen = get_agent_last_seen(self.recall_dir)
                tmux_windows = get_tmux_windows(self.session)

                for agent_name in self.agents:
                    alert = self.check_agent(agent_name, last_seen, tmux_windows)
                    if alert:
                        self.handle_alert(agent_name, alert)

                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\nspawn_watcher stopped.")
            log_event(self.event_path, {"event": "watcher_stop"})


def main():
    parser = argparse.ArgumentParser(description="Agent presence tracker for gr spawn")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to agents.toml")
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL, help="Seconds between polls")
    parser.add_argument("--mock", action="store_true", default=True, help="Mock restarts (Phase 1)")
    parser.add_argument("--no-mock", action="store_false", dest="mock", help="Real restarts (Phase 2+)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Find recall dir
    recall_dir = Path(".synapt/recall")
    if not recall_dir.exists():
        # Try from gripspace root
        for candidate in [Path("synapt/.synapt/recall"), Path("../.synapt/recall")]:
            if candidate.exists():
                recall_dir = candidate
                break

    watcher = SpawnWatcher(
        config=config,
        recall_dir=recall_dir,
        mock=args.mock,
        poll_interval=args.poll_interval,
    )
    watcher.run()


if __name__ == "__main__":
    main()
