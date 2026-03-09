"""ChatGPT archive parser for synapt recall.

Parses ChatGPT export archives (.zip containing conversations.json or
standalone conversations.json) into TranscriptChunk objects compatible
with the main transcript index.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from synapt.recall.core import TranscriptChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_iso_timestamp(value: object) -> str:
    """Convert ChatGPT timestamp values to ISO 8601."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except (ValueError, OSError, OverflowError):
            return ""
    return ""


def _extract_chatgpt_message_text(message: dict) -> str:
    """Extract plain text from a ChatGPT export message payload."""
    content = message.get("content")
    if not isinstance(content, dict):
        return ""

    content_type = content.get("content_type")
    if content_type == "code":
        text = content.get("text")
        return text.strip() if isinstance(text, str) else ""

    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""

    texts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            text = part.strip()
            if text:
                texts.append(text)
            continue
        if isinstance(part, dict):
            # multimodal and tool payloads may store text fields in dict parts
            for key in ("text", "content", "title"):
                val = part.get(key)
                if isinstance(val, str):
                    txt = val.strip()
                    if txt:
                        texts.append(txt)
                        break

    return "\n".join(texts)


def _chatgpt_path_nodes(conversation: dict) -> list[dict]:
    """Return mapping nodes on the active ChatGPT conversation path."""
    mapping = conversation.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        return []

    current_node = conversation.get("current_node")
    if not isinstance(current_node, str) or current_node not in mapping:
        current_node = next(iter(mapping.keys()))

    ordered_ids: list[str] = []
    seen: set[str] = set()
    node_id: Optional[str] = current_node

    while node_id and node_id not in seen:
        seen.add(node_id)
        ordered_ids.append(node_id)
        node = mapping.get(node_id, {})
        if not isinstance(node, dict):
            break
        parent = node.get("parent")
        node_id = parent if isinstance(parent, str) else None

    ordered_ids.reverse()
    nodes = []
    for nid in ordered_ids:
        node = mapping.get(nid)
        if isinstance(node, dict):
            nodes.append(node)
    return nodes


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_chatgpt_conversation(conversation: dict) -> list[TranscriptChunk]:
    """Parse one ChatGPT conversation export object into turn chunks."""
    session_id = conversation.get("id") or conversation.get("conversation_id")
    if not isinstance(session_id, str) or not session_id:
        return []

    nodes = _chatgpt_path_nodes(conversation)
    if not nodes:
        return []

    chunks: list[TranscriptChunk] = []
    current_user_text = ""
    current_assistant_texts: list[str] = []
    current_tools: list[str] = []
    current_files: list[str] = []
    current_timestamp = ""
    turn_index = 0

    convo_ts = (
        _to_iso_timestamp(conversation.get("update_time"))
        or _to_iso_timestamp(conversation.get("create_time"))
    )

    def _flush_turn():
        nonlocal turn_index
        if not current_user_text and not current_assistant_texts:
            return
        assistant_text = "\n".join(current_assistant_texts).strip()
        if len(assistant_text) > 5000:
            assistant_text = assistant_text[:5000] + "..."
        short_id = session_id[:8]
        chunks.append(
            TranscriptChunk(
                id=f"{short_id}:t{turn_index}",
                session_id=session_id,
                timestamp=current_timestamp or convo_ts,
                turn_index=turn_index,
                user_text=(current_user_text[:1500] + "..." if len(current_user_text) > 1500
                          else current_user_text),
                assistant_text=assistant_text,
                tools_used=list(dict.fromkeys(current_tools)),
                files_touched=list(dict.fromkeys(current_files)),
            )
        )
        turn_index += 1

    for node in nodes:
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        author = message.get("author")
        role = author.get("role") if isinstance(author, dict) else ""
        if role not in {"user", "assistant", "tool"}:
            continue

        text = _extract_chatgpt_message_text(message)
        content = message.get("content")
        content_type = content.get("content_type") if isinstance(content, dict) else ""
        msg_ts = (
            _to_iso_timestamp(message.get("create_time"))
            or _to_iso_timestamp(message.get("update_time"))
            or convo_ts
        )

        if role == "user":
            # user_editable_context is profile memory, not a conversational turn
            if content_type == "user_editable_context":
                continue
            _flush_turn()
            current_user_text = text
            current_assistant_texts = []
            current_tools = []
            current_files = []
            current_timestamp = msg_ts
            continue

        if role == "assistant":
            if text:
                current_assistant_texts.append(text)
            recipient = message.get("recipient")
            if isinstance(recipient, str) and recipient not in {"all", "assistant"}:
                current_tools.append(recipient)
            if not current_timestamp:
                current_timestamp = msg_ts
            continue

        # role == "tool": treat as assistant continuation and record tool name
        if text:
            current_assistant_texts.append(text)
        if isinstance(author, dict):
            name = author.get("name")
            if isinstance(name, str) and name:
                current_tools.append(name)
        recipient = message.get("recipient")
        if isinstance(recipient, str) and recipient not in {"all", "assistant"}:
            current_tools.append(recipient)
        if not current_timestamp:
            current_timestamp = msg_ts

    _flush_turn()
    return chunks


def parse_chatgpt_archive(path: Path) -> list[TranscriptChunk]:
    """Parse a ChatGPT export zip or conversations.json file."""
    conv_path = str(path)
    conversations: object

    if conv_path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            with zf.open("conversations.json") as f:
                conversations = json.load(f)
    else:
        with open(path, encoding="utf-8") as f:
            conversations = json.load(f)

    if not isinstance(conversations, list):
        return []

    all_chunks: list[TranscriptChunk] = []
    for conversation in conversations:
        if isinstance(conversation, dict):
            all_chunks.extend(parse_chatgpt_conversation(conversation))
    return all_chunks
