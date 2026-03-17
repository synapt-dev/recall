"""Parse Codex CLI transcripts into TranscriptChunks.

Codex CLI stores sessions at ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl.
The format differs from Claude Code:
  - session_meta entry has session ID and cwd
  - response_item entries with role: user/developer/assistant
  - function_call / function_call_output for tool use
  - Content blocks use input_text/output_text types

This module converts Codex transcripts into the same TranscriptChunk format
used for Claude Code, enabling cross-editor memory via synapt recall.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import re

from synapt.recall.core import TranscriptChunk, _short_sid

# Simple file path regex — matches common code file paths
_FILE_RE = re.compile(r'(?:^|[\s"\'`])(/[\w./-]+\.\w{1,10})(?:[\s"\'`:,)]|$)')


def _extract_file_paths(text: str) -> list[str]:
    """Extract likely file paths from text."""
    return list(dict.fromkeys(_FILE_RE.findall(text)))


def discover_codex_sessions() -> Path | None:
    """Return the Codex sessions directory if it exists."""
    codex_dir = Path.home() / ".codex" / "sessions"
    if codex_dir.is_dir():
        return codex_dir
    return None


def list_codex_transcripts(sessions_dir: Path | None = None) -> list[Path]:
    """List all Codex transcript JSONL files, sorted by name."""
    if sessions_dir is None:
        sessions_dir = discover_codex_sessions()
    if sessions_dir is None:
        return []
    return sorted(sessions_dir.rglob("rollout-*.jsonl"))


def parse_codex_transcript(
    path: Path,
    seen_uuids: set[str] | None = None,
) -> list[TranscriptChunk]:
    """Parse a Codex CLI transcript into TranscriptChunks.

    Maps the Codex JSONL format to the same chunk structure used for
    Claude Code transcripts, enabling unified search and recall.

    Args:
        path: Path to a rollout-*.jsonl file.
        seen_uuids: Set of already-seen session IDs for dedup.

    Returns:
        List of TranscriptChunk objects.
    """
    if seen_uuids is None:
        seen_uuids = set()

    chunks: list[TranscriptChunk] = []
    session_id = ""
    transcript_path = str(path)

    # Accumulator for current turn
    current_user_text = ""
    current_assistant_texts: list[str] = []
    current_tools: list[str] = []
    current_files: list[str] = []
    current_timestamp = ""
    current_tool_summaries: list[str] = []
    turn_index = 0
    turn_start_offset = 0
    current_offset = 0

    def _flush_turn(end_offset: int = 0):
        nonlocal turn_index
        if not current_user_text and not current_assistant_texts:
            return

        short_id = _short_sid(session_id) if session_id else path.stem[:8]
        chunk_id = f"{short_id}:t{turn_index}"

        # Dedup
        if chunk_id in seen_uuids:
            return
        seen_uuids.add(chunk_id)

        assistant_text = "\n".join(current_assistant_texts).strip()
        if len(assistant_text) > 5000:
            assistant_text = assistant_text[:5000] + "..."

        tool_content = "\n".join(current_tool_summaries).strip()
        if len(tool_content) > 3000:
            tool_content = tool_content[:3000] + "..."

        # Extract files from all text
        all_text = current_user_text + " " + assistant_text + " " + tool_content
        files = list(dict.fromkeys(current_files + _extract_file_paths(all_text)))

        chunk = TranscriptChunk(
            id=chunk_id,
            session_id=session_id or path.stem,
            timestamp=current_timestamp,
            turn_index=turn_index,
            user_text=current_user_text.strip(),
            assistant_text=assistant_text,
            tools_used=list(dict.fromkeys(current_tools)),
            files_touched=files[:20],  # Cap to avoid bloat
            tool_content=tool_content,
            transcript_path=transcript_path,
            byte_offset=turn_start_offset,
            byte_length=end_offset - turn_start_offset,
        )
        chunks.append(chunk)
        turn_index += 1

    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line_bytes = len(line.encode("utf-8"))
                line = line.strip()
                if not line:
                    current_offset += line_bytes
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    current_offset += line_bytes
                    continue

                entry_type = entry.get("type", "")
                timestamp = entry.get("timestamp", "")
                payload = entry.get("payload", {})

                if entry_type == "session_meta":
                    # Extract session ID from metadata
                    session_id = payload.get("id", path.stem)
                    if session_id in seen_uuids:
                        return []  # Already indexed
                    seen_uuids.add(session_id)

                elif entry_type == "response_item":
                    payload_type = payload.get("type", "")
                    role = payload.get("role", "")
                    content_blocks = payload.get("content", [])
                    phase = payload.get("phase", "")

                    # Handle function calls first (no role field)
                    if payload_type == "function_call":
                        tool_name = payload.get("name", "unknown")
                        current_tools.append(tool_name)
                        args = payload.get("arguments", "")
                        if isinstance(args, str) and len(args) < 500:
                            try:
                                args_parsed = json.loads(args)
                                cmd = args_parsed.get("cmd", "")
                                if cmd:
                                    current_tool_summaries.append(f"[{tool_name}] {cmd}")
                                    current_files.extend(_extract_file_paths(cmd))
                            except (json.JSONDecodeError, TypeError):
                                current_tool_summaries.append(f"[{tool_name}] {args[:200]}")
                        current_offset += line_bytes
                        continue

                    if payload_type == "function_call_output":
                        # Tool output — could extract file paths but skip for now
                        current_offset += line_bytes
                        continue

                    if role == "user":
                        # New user turn — flush previous
                        _flush_turn(current_offset)
                        current_user_text = ""
                        current_assistant_texts = []
                        current_tools = []
                        current_files = []
                        current_tool_summaries = []
                        current_timestamp = timestamp
                        turn_start_offset = current_offset

                        for block in content_blocks:
                            text = block.get("text", "")
                            if block.get("type") == "input_text" and text:
                                # Skip system-level content (permissions, env context)
                                if text.startswith("<permissions") or text.startswith("<environment"):
                                    continue
                                # Skip AGENTS.md injections
                                if text.startswith("# AGENTS.md"):
                                    continue
                                current_user_text += text + "\n"

                    elif role == "assistant":
                        # Assistant response text
                        for block in content_blocks:
                            text = block.get("text", "")
                            if block.get("type") == "output_text" and text:
                                # Skip commentary phase — it's intermediate thinking
                                if phase == "commentary":
                                    continue
                                current_assistant_texts.append(text)

                    elif role == "developer":
                        # Developer role = system prompts — skip (too noisy)
                        pass

                elif entry_type == "event_msg":
                    msg_type = payload.get("type", "")
                    if msg_type == "user_message":
                        # Sometimes user text comes via event_msg instead of response_item
                        text = payload.get("message", "")
                        if text and not current_user_text:
                            _flush_turn(current_offset)
                            current_user_text = text
                            current_assistant_texts = []
                            current_tools = []
                            current_files = []
                            current_tool_summaries = []
                            current_timestamp = timestamp
                            turn_start_offset = current_offset

                current_offset += line_bytes

        # Flush final turn
        _flush_turn(current_offset)

    except OSError:
        pass

    return chunks
