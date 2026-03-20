"""Interactive terminal chat for agent channels.

Usage:
    synapt recall channel chat [channel] [--poll 1.0] [--name layne]

Separate module from channel.py — terminal I/O is a different concern
from the data layer. This module handles ANSI colors, stdin polling,
slash commands, and efficient JSONL tailing.
"""

from __future__ import annotations

import json
import os
import readline as _readline  # noqa: F401 — import enables line editing
import select
import sys
from pathlib import Path

from synapt.recall.channel import (
    ChannelMessage,
    _channel_path,
    _now_iso,
    channel_join,
    channel_leave,
    channel_post,
    channel_read,
    channel_who,
    channel_unread,
    channel_pin,
    channel_directive,
    channel_mute,
    channel_kick,
    channel_broadcast,
    channel_list_channels,
    channel_heartbeat,
)


# ---------------------------------------------------------------------------
# ANSI colors — no external deps
# ---------------------------------------------------------------------------

_COLORS = [
    "\033[36m",   # cyan
    "\033[33m",   # yellow
    "\033[35m",   # magenta
    "\033[32m",   # green
    "\033[34m",   # blue
    "\033[91m",   # bright red
    "\033[92m",   # bright green
    "\033[93m",   # bright yellow
    "\033[94m",   # bright blue
    "\033[95m",   # bright magenta
    "\033[96m",   # bright cyan
]
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"


def _color_for(name: str) -> str:
    """Deterministic color from agent name hash."""
    return _COLORS[hash(name) % len(_COLORS)]


def _render_message(msg: ChannelMessage, my_id: str) -> str:
    """Render a single message with ANSI colors."""
    ts = msg.timestamp[11:19]  # HH:MM:SS
    mid = f" {_DIM}[{msg.id}]{_RESET}" if msg.id else ""
    color = _color_for(msg.from_agent)

    if msg.type in ("join", "leave"):
        return f"  {_DIM}{ts}{mid}  -- {msg.body}{_RESET}"

    if msg.type == "directive":
        if msg.to == my_id:
            prefix = f"{_BOLD}[DIRECTIVE]{_RESET}"
        else:
            prefix = f"{_DIM}[directive]{_RESET}"
        target = f" @{msg.to}" if msg.to else ""
        return f"  {_DIM}{ts}{mid}{_RESET}  {prefix}{target} {color}{msg.from_agent}{_RESET}: {msg.body}"

    return f"  {_DIM}{ts}{mid}{_RESET}  {color}{msg.from_agent}{_RESET}: {msg.body}"


# ---------------------------------------------------------------------------
# JSONL tailer — O(new) not O(total)
# ---------------------------------------------------------------------------

class _Tailer:
    """Tail a JSONL file efficiently by tracking file offset."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self._init_offset()

    def _init_offset(self) -> None:
        """Set offset to end of file (skip existing messages)."""
        try:
            self.offset = self.path.stat().st_size
        except OSError:
            self.offset = 0

    def poll(self) -> list[ChannelMessage]:
        """Read new messages since last poll. O(new) not O(total)."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size <= self.offset:
            return []

        messages = []
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read()
                self.offset = f.tell()
            for line in data.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(ChannelMessage.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue
        except OSError:
            pass
        return messages


# ---------------------------------------------------------------------------
# ChatUI
# ---------------------------------------------------------------------------

class _ChatCompleter:
    """Tab completer for slash commands and @mentions."""

    _COMMANDS = [
        "/quit", "/q", "/help", "/h", "/who", "/w", "/channels", "/ch",
        "/join", "/j", "/history", "/pin", "/mute", "/kick",
        "/broadcast", "/directive", "/d", "/clear",
    ]

    def __init__(self):
        self._agents: list[str] = []

    def update_agents(self, agents: list[str]) -> None:
        """Update the known agent list for @mention completion."""
        self._agents = agents

    def complete(self, text: str, state: int) -> str | None:
        if text.startswith("/"):
            matches = [c for c in self._COMMANDS if c.startswith(text)]
        elif text.startswith("@"):
            prefix = text[1:].lower()
            matches = [f"@{a}" for a in self._agents if a.lower().startswith(prefix)]
        else:
            return None
        return matches[state] if state < len(matches) else None


class ChatUI:
    """Interactive terminal chat UI for agent channels."""

    def __init__(
        self,
        channel: str = "dev",
        agent_name: str | None = None,
        poll_interval: float = 1.0,
    ):
        self.channel = channel
        self.agent_id = "human"
        self.display_name = agent_name or os.environ.get("USER", "human")
        self.poll_interval = poll_interval
        self._tailer: _Tailer | None = None
        self._running = False
        self._poll_count = 0
        self._heartbeat_interval = 60  # heartbeat every N polls

        # Set up readline for editing + tab completion
        self._completer = _ChatCompleter()
        import readline
        readline.set_completer(self._completer.complete)
        readline.set_completer_delims(" ")
        readline.parse_and_bind("tab: complete")

    def run(self) -> None:
        """Main event loop."""
        # Join channel
        channel_join(
            self.channel,
            agent_name=self.agent_id,
        )

        # Set up tailer (skip existing messages)
        path = _channel_path(self.channel)
        self._tailer = _Tailer(path)

        # Show recent history
        self._show_history(10)
        self._print_status(f"Joined #{self.channel} as {self.display_name}. Type /help for commands.")

        self._running = True
        try:
            self._loop()
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            self._running = False
            channel_leave(self.channel, agent_name=self.agent_id, reason=f"{self.display_name} left")
            self._print_status(f"Left #{self.channel}")

    def _loop(self) -> None:
        """Poll stdin + JSONL tail in a select loop."""
        while self._running:
            # Rate-limited heartbeat
            self._poll_count += 1
            if self._poll_count % self._heartbeat_interval == 0:
                channel_heartbeat(agent_name=self.agent_id)

            # Use select to wait for stdin or timeout
            try:
                readable, _, _ = select.select([sys.stdin], [], [], self.poll_interval)
            except (ValueError, OSError):
                # stdin closed
                break

            if readable:
                try:
                    line = input()
                except (EOFError, OSError):
                    break
                line = line.strip()
                if not line:
                    continue
                if line.startswith("/"):
                    if not self._handle_command(line):
                        break  # /quit
                else:
                    channel_post(self.channel, line, agent_name=self.agent_id)
                    # Echo own message with formatting
                    echo = ChannelMessage(
                        timestamp=_now_iso(),
                        from_agent=self.display_name,
                        channel=self.channel,
                        type="message",
                        body=line,
                    )
                    print(_render_message(echo, self.agent_id))

            # Poll for new messages
            self._poll_and_render()

    def _poll_and_render(self) -> None:
        """Check for new messages and render them."""
        if not self._tailer:
            return
        for msg in self._tailer.poll():
            # Skip our own messages (we already echoed them on post)
            if msg.from_agent == self.agent_id and msg.type == "message":
                continue
            print(_render_message(msg, self.agent_id))

    def _handle_command(self, line: str) -> bool:
        """Dispatch slash commands. Returns False for /quit."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/quit" or cmd == "/q":
            return False

        if cmd == "/help" or cmd == "/h":
            self._show_help()
        elif cmd == "/who" or cmd == "/w":
            who_result = channel_who()
            print(who_result)
            # Update tab completion with current agent names
            try:
                from synapt.recall.channel import _open_db
                conn = _open_db()
                rows = conn.execute("SELECT display_name, agent_id FROM presence").fetchall()
                conn.close()
                names = [r["display_name"] or r["agent_id"] for r in rows]
                self._completer.update_agents(names)
            except Exception:
                pass
        elif cmd == "/channels" or cmd == "/ch":
            self._show_channels()
        elif cmd == "/join" or cmd == "/j":
            if arg:
                self._switch_channel(arg)
            else:
                self._print_error("Usage: /join <channel>")
        elif cmd == "/history":
            n = int(arg) if arg.isdigit() else 20
            self._show_history(n)
        elif cmd == "/pin":
            if arg:
                print(channel_pin(self.channel, message_id=arg, agent_name=self.agent_id))
            else:
                self._print_error("Usage: /pin <message_id>")
        elif cmd == "/mute":
            if arg:
                print(channel_mute(target=arg, channel=self.channel, agent_name=self.agent_id))
            else:
                self._print_error("Usage: /mute <agent>")
        elif cmd == "/kick":
            if arg:
                print(channel_kick(target=arg, channel=self.channel, agent_name=self.agent_id))
            else:
                self._print_error("Usage: /kick <agent>")
        elif cmd == "/broadcast":
            if arg:
                print(channel_broadcast(message=arg, agent_name=self.agent_id))
            else:
                self._print_error("Usage: /broadcast <message>")
        elif cmd == "/directive" or cmd == "/d":
            if ":" in arg:
                target, msg = arg.split(":", 1)
                print(channel_directive(
                    self.channel, msg.strip(), to=target.strip(),
                    agent_name=self.agent_id,
                ))
            else:
                self._print_error("Usage: /directive <agent>: <message>")
        elif cmd == "/clear":
            self._clear_channel()
        else:
            self._print_error(f"Unknown command: {cmd}. Type /help for commands.")

        return True

    def _show_help(self) -> None:
        """Show available commands."""
        print(f"""
{_BOLD}Commands:{_RESET}
  /who, /w              Show online agents
  /channels, /ch        List channels + unread counts
  /join <ch>, /j <ch>   Switch to another channel
  /history [n]          Show last n messages (default 20)
  /pin <msg_id>         Pin a message by ID
  /mute <agent>         Mute an agent
  /kick <agent>         Kick an agent
  /broadcast <msg>      Post to all channels
  /directive <agent>: <msg>   Send a priority directive
  /clear                Clear channel (with confirm)
  /quit, /q             Leave and exit
""")

    def _show_history(self, n: int) -> None:
        """Show recent channel history."""
        result = channel_read(
            self.channel, limit=n, agent_name=self.agent_id,
        )
        print(result)

    def _show_channels(self) -> None:
        """Show channels with unread counts."""
        channels = channel_list_channels()
        if not channels:
            print("No channels yet.")
            return
        counts = channel_unread(agent_name=self.agent_id)
        print(f"{_BOLD}Channels:{_RESET}")
        for ch in channels:
            unread = counts.get(ch, 0)
            marker = f"  ({unread} new)" if unread > 0 else ""
            current = " ←" if ch == self.channel else ""
            print(f"  #{ch}{marker}{current}")

    def _switch_channel(self, new_channel: str) -> None:
        """Switch to a different channel."""
        channel_leave(self.channel, agent_name=self.agent_id)
        self.channel = new_channel
        channel_join(self.channel, agent_name=self.agent_id)

        # Reset tailer for new channel
        path = _channel_path(self.channel)
        self._tailer = _Tailer(path)

        self._show_history(10)
        self._print_status(f"Switched to #{self.channel}")

    def _clear_channel(self) -> None:
        """Clear channel log with confirmation."""
        self._print_status(f"Clear all messages in #{self.channel}? [y/N] ", end="")
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if answer == "y":
            path = _channel_path(self.channel)
            try:
                path.write_text("")
                if self._tailer:
                    self._tailer.offset = 0
                self._print_status(f"Cleared #{self.channel}")
            except OSError as e:
                self._print_error(f"Failed to clear: {e}")
        else:
            self._print_status("Cancelled.")

    def _print_status(self, msg: str, end: str = "\n") -> None:
        """Print a status message."""
        print(f"{_DIM}  {msg}{_RESET}", end=end)

    def _print_error(self, msg: str) -> None:
        """Print an error message."""
        print(f"  \033[91m{msg}{_RESET}")


def main(
    channel: str = "dev",
    name: str | None = None,
    poll: float = 1.0,
) -> None:
    """Entry point for interactive chat."""
    ui = ChatUI(channel=channel, agent_name=name, poll_interval=poll)
    ui.run()
