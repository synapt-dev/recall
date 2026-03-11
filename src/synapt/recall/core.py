"""synapt.recall — persistent conversational memory for Claude Code sessions.

Parses Claude Code transcript archives (.jsonl), chunks them into semantic
turn-level units, indexes them with BM25 (+optional embeddings), and enables
time-ordered retrieval of past session context.

Usage:
    from synapt.recall import parse_transcript, TranscriptIndex

    chunks = parse_transcript(Path("~/.claude/projects/.../session.jsonl"))
    index = TranscriptIndex(chunks)
    context = index.lookup("quality curve weighting", max_chunks=3)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from synapt.recall.scrub import scrub_text

logger = logging.getLogger("synapt.recall")

from synapt.recall.bm25 import BM25, _tokenize
from synapt.recall.storage import RecallDB


def format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string (KB or MB)."""
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    return f"{size_bytes / 1_000:.1f} KB"


def atomic_json_write(data: dict | list, path: Path, indent: int = 2) -> None:
    """Write JSON to *path* atomically via temp-file + rename."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


# Near-duplicate Jaccard threshold for result deduplication
_DEDUP_JACCARD_THRESHOLD = 0.6


def _dedup_limit(max_chunks: int) -> int:
    """Extra candidates to pass through formatting to compensate for dedup."""
    return max_chunks + max(5, max_chunks // 3)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TranscriptChunk:
    """One turn in a Claude Code conversation."""

    id: str  # "{session_short}:t{turn_index}"
    session_id: str  # Full UUID
    timestamp: str  # ISO 8601
    turn_index: int  # 0-based within session
    user_text: str  # Cleaned user message
    assistant_text: str  # Concatenated assistant text blocks
    tools_used: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    tool_content: str = ""  # Summarized tool inputs + results
    transcript_path: str = ""  # Path to source transcript file
    byte_offset: int = -1  # Byte position where this turn starts in raw JSONL
    byte_length: int = 0  # Total bytes of raw JSONL entries for this turn
    text: str = ""  # Combined searchable text (built at init)

    def __post_init__(self):
        if not self.text:
            self.text = self._build_text()

    def _build_text(self) -> str:
        parts = []
        if self.user_text:
            parts.append(self.user_text)
        if self.assistant_text:
            parts.append(self.assistant_text)
        if self.tool_content:
            parts.append(self.tool_content)
        if self.tools_used:
            parts.append(" ".join(self.tools_used))
        if self.files_touched:
            parts.append(" ".join(self.files_touched))
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "turn_index": self.turn_index,
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
            "tools_used": self.tools_used,
            "files_touched": self.files_touched,
            "tool_content": self.tool_content,
            "transcript_path": self.transcript_path,
            "byte_offset": self.byte_offset,
            "byte_length": self.byte_length,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TranscriptChunk:
        return cls(
            id=d["id"],
            session_id=d["session_id"],
            timestamp=d["timestamp"],
            turn_index=d["turn_index"],
            user_text=d["user_text"],
            assistant_text=d["assistant_text"],
            tools_used=d.get("tools_used", []),
            files_touched=d.get("files_touched", []),
            tool_content=d.get("tool_content", ""),
            transcript_path=d.get("transcript_path", ""),
            byte_offset=d.get("byte_offset", -1),
            byte_length=d.get("byte_length", 0),
        )


@dataclass
class SearchDiagnostics:
    """Diagnostic info collected when a search returns no results."""

    total_chunks: int = 0
    total_sessions: int = 0
    candidates_found: int = 0        # Chunks with score > 0 (before threshold)
    search_mode: str = ""            # fts_global, bm25_global, fts_progressive, bm25_progressive
    sessions_searched: int = 0       # Progressive mode only
    date_filter_active: bool = False
    reason: str = ""                 # empty_index, empty_query, no_matches

    def format_message(self) -> str:
        """Format diagnostic information as a user-facing message."""
        if self.reason == "empty_index":
            return "No results found — index is empty. Run `recall_setup` or `recall_build` first."

        if self.reason == "empty_query":
            return "No results found — query produced no search tokens."

        if self.reason == "no_matches":
            lines = ["No results found — no chunks matched your query terms."]
            lines.append(f"  Index: {self.total_chunks} chunks across {self.total_sessions} sessions")
            if self.date_filter_active:
                lines.append("  Note: date filter was active — try widening the date range")
            lines.append("  Try: broader terms, different keywords, or check `recall_stats`")
            return "\n".join(lines)

        return "No results found."


# ---------------------------------------------------------------------------
# Transcript parser
# ---------------------------------------------------------------------------

SKIP_TYPES = {"progress", "file-history-snapshot", "queue-operation", "system"}

# Patterns that indicate an error in Bash output (used to give errors 2x budget).
_ERROR_PATTERNS = re.compile(
    r"(?:^|\n)(?:Error|Traceback|FAILED|FATAL|panic:|error\[E)"
    r"|(?:exit code [1-9]|Exit code [1-9]|returned non-zero)"
    r"|(?:ModuleNotFoundError|ImportError|SyntaxError|NameError"
    r"|TypeError|ValueError|KeyError|AttributeError|FileNotFoundError)",
)


def _summarize_tool_result(
    tool_name: str,
    tool_input: dict,
    result_text: str,
) -> str:
    """Summarize a tool result into a compact searchable string.

    Different tools get different budgets — Bash errors are high-value,
    Read file contents are low-value (file exists in repo).
    """
    if not result_text:
        return ""

    name = tool_name.lower()
    # Strip common MCP prefixes (lowered for routing, original for display)
    short = name.rsplit("__", 1)[-1] if "__" in name else name
    display_name = tool_name.rsplit("__", 1)[-1] if "__" in tool_name else tool_name

    if short == "bash":
        cmd = tool_input.get("command", "")
        is_error = bool(_ERROR_PATTERNS.search(result_text))
        cap = 1200 if is_error else 600
        output = result_text[:cap]
        if len(result_text) > cap:
            output += "..."
        return f"$ {cmd}\n{output}" if cmd else output

    if short == "read":
        path = tool_input.get("file_path", "")
        n_lines = result_text.count("\n") + 1
        return f"Read {path} ({n_lines} lines)"

    if short == "write":
        path = tool_input.get("file_path", "")
        return f"Wrote {path}"

    if short == "edit":
        path = tool_input.get("file_path", "")
        old = tool_input.get("old_string", "")[:60]
        new = tool_input.get("new_string", "")[:60]
        if old or new:
            return f"Edited {path}: {old!r} -> {new!r}"
        return f"Edited {path}"

    if short == "grep":
        pattern = tool_input.get("pattern", "")
        return f'Grep "{pattern}": {result_text[:200]}'

    if short == "glob":
        pattern = tool_input.get("pattern", "")
        n_matches = result_text.count("\n") + 1 if result_text.strip() else 0
        return f'Glob "{pattern}": {n_matches} files'

    if short == "agent":
        return f"Agent: {result_text[:200]}"

    # Default: use display name (prefix-stripped, original casing) + truncated result
    return f"{display_name}: {result_text[:200]}"


def _summarize_tool_input(tool_name: str, tool_input: dict) -> str:
    """Extract searchable content from an assistant's tool_use input.

    Captures commands, patterns, and queries — content that helps
    answer "what did I do?" but isn't captured in files_touched.
    """
    name = tool_name.lower()
    short = name.rsplit("__", 1)[-1] if "__" in name else name

    if short == "bash":
        cmd = tool_input.get("command", "")
        desc = tool_input.get("description", "")
        if desc:
            return f"$ {cmd}  # {desc}"
        return f"$ {cmd}" if cmd else ""

    if short == "edit":
        path = tool_input.get("file_path", "")
        old = tool_input.get("old_string", "")[:60]
        new = tool_input.get("new_string", "")[:60]
        if old or new:
            return f"Edit {path}: {old!r} -> {new!r}"
        return ""

    if short == "grep":
        return f'Grep "{tool_input.get("pattern", "")}"'

    if short in ("websearch", "web_search"):
        return f'Search: {tool_input.get("query", "")}'

    return ""

# Phrases that signal a user decision (lowercase matching).
_DECISION_PHRASES = (
    "let's do", "let's go with", "let's use", "let's try",
    "go with option", "go with approach",
    "i'll go with", "i choose", "i picked",
    "approve the plan", "looks good, proceed",
    "yes, do that", "yes, go ahead", "yes, let's",
    "option a", "option b", "option c",
)

# Compiled word-boundary regex for each phrase (prevents substring
# false positives like "option actually" matching "option a").
_DECISION_RE = re.compile(
    "|".join(r"\b" + re.escape(p) + r"\b" for p in _DECISION_PHRASES)
)


def _extract_tool_result_text(entry: dict) -> str:
    """Extract text content from tool_result blocks in a user message.

    AskUserQuestion responses appear as tool_result entries.  The user's
    selected option is in the ``content`` field of the tool_result block.
    """
    content = entry.get("message", {}).get("content")
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        result = block.get("content", "")
        if isinstance(result, str) and result.strip():
            parts.append(result.strip())
        elif isinstance(result, list):
            for sub in result:
                if isinstance(sub, dict) and sub.get("type") == "text":
                    text = sub.get("text", "").strip()
                    if text:
                        parts.append(text)
    return "\n".join(parts)


def _detect_decision_markers(user_text: str, tools_used: list[str]) -> list[str]:
    """Detect decision signals and return virtual tool markers for search boosting.

    Injected into ``tools_used`` so they get indexed with 2x weight in FTS5
    and boosted in BM25 — no schema changes needed.

    Markers use spaces (not underscores) so FTS5's ``tokenchars '._'`` setting
    doesn't merge them into unsearchable single tokens.  E.g. "decision point"
    becomes two tokens ("decision", "point") searchable individually.

    Uses a set internally to avoid duplicates when multiple signals fire
    (e.g., both AskUserQuestion and ExitPlanMode in one turn).
    """
    markers: set[str] = set()
    text_lower = user_text.lower()

    # Signal 1: AskUserQuestion tool was used (structured choice)
    if "AskUserQuestion" in tools_used:
        markers.add("decision point")
        markers.add("user choice")

    # Signal 2: ExitPlanMode (plan approved)
    if "ExitPlanMode" in tools_used:
        markers.add("decision point")
        markers.add("plan approved")

    # Signal 3: Explicit choice keywords in user text (word-boundary match)
    if not markers and _DECISION_RE.search(text_lower):
        markers.add("decision point")

    return sorted(markers)


def _is_real_user_message(entry: dict) -> bool:
    """True if entry is a user turn that starts a new conversation turn.

    Real user messages have string content or contain at least one text block
    *after* system artifacts are stripped. Messages containing only
    ``<system-reminder>`` blocks (no actual user text) are treated as
    non-messages so they don't create phantom empty turns in the index.
    """
    from synapt.recall.scrub import strip_system_artifacts

    if entry.get("type") != "user":
        return False
    msg = entry.get("message", {})
    if not isinstance(msg, dict):
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return bool(strip_system_artifacts(content.strip()))
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "text"
            and strip_system_artifacts(b.get("text", "").strip())
            for b in content
        )
    return False


def _extract_user_text(entry: dict) -> str:
    """Extract plain text from a user message entry.

    Strips Claude Code system artifacts (``<system-reminder>``,
    ``<local-command-caveat>``, etc.) so they never reach the index,
    journal stubs, or timeline arcs.
    """
    from synapt.recall.scrub import strip_system_artifacts

    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        return strip_system_artifacts(content.strip())
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    parts.append(text)
        return strip_system_artifacts("\n".join(parts))
    return ""


def _extract_assistant_content(
    entry: dict,
) -> tuple[str, list[str], list[str], list[tuple[str, str, dict]]]:
    """Extract text, tool names, file paths, and tool invocations.

    Returns (text, tools_used, files_touched, tool_uses) where tool_uses
    is a list of (tool_use_id, tool_name, tool_input) tuples for matching
    with subsequent tool_result entries.
    """
    content = entry.get("message", {}).get("content")
    if not isinstance(content, list):
        return "", [], [], []

    texts = []
    tools = []
    files = []
    tool_uses: list[tuple[str, str, dict]] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "").strip()
            if text:
                texts.append(text)
        elif btype == "tool_use":
            name = block.get("name", "")
            if name:
                tools.append(name)
            inp = block.get("input", {})
            tool_use_id = block.get("id", "")
            if tool_use_id and isinstance(inp, dict):
                tool_uses.append((tool_use_id, name, inp))
            if isinstance(inp, dict):
                for key in ("file_path", "path", "command", "pattern"):
                    val = inp.get(key)
                    if val and isinstance(val, str):
                        if key in ("file_path", "path") or (
                            key == "pattern" and "/" in val
                        ):
                            files.append(val)
                        break
        # Skip 'thinking' blocks — internal reasoning, not useful for RAG

    return "\n".join(texts), tools, files, tool_uses


def parse_transcript(
    path: Path,
    seen_uuids: set[str] | None = None,
) -> list[TranscriptChunk]:
    """Parse a Claude Code transcript JSONL file into semantic chunks.

    Each chunk represents one user->assistant turn. Streams line-by-line to
    handle large files (100K+ lines, 800MB+) without loading into memory.

    Args:
        path: Path to a .jsonl transcript file.
        seen_uuids: Set of already-seen (session_id, uuid) pairs for dedup.
                    Mutated in-place to add new entries.

    Returns:
        List of TranscriptChunk objects, one per turn.
    """
    if seen_uuids is None:
        seen_uuids = set()

    chunks: list[TranscriptChunk] = []
    session_id = path.stem  # UUID from filename
    transcript_path = str(path)

    # Accumulator for current turn
    current_user_text = ""
    current_assistant_texts: list[str] = []
    current_tools: list[str] = []
    current_files: list[str] = []
    current_timestamp = ""
    current_tool_summaries: list[str] = []
    # Maps tool_use_id → (tool_name, tool_input) for matching results
    current_tool_use_map: dict[str, tuple[str, dict]] = {}
    turn_index = 0
    turn_start_offset = 0
    current_offset = 0

    def _flush_turn(end_offset: int | None = None):
        nonlocal turn_index
        if not current_user_text and not current_assistant_texts:
            return
        _end = end_offset if end_offset is not None else current_offset
        assistant_text = "\n".join(current_assistant_texts).strip()
        # Truncate very long assistant texts (code dumps, etc.)
        # 5000 chars gives ~1250 tokens — enough for a full detailed response
        # while keeping DB size reasonable. recall_context drill-down has no cap.
        if len(assistant_text) > 5000:
            assistant_text = assistant_text[:5000] + "..."
        tools_deduped = list(dict.fromkeys(current_tools))
        # Detect decision signals and inject searchable markers
        decision_markers = _detect_decision_markers(current_user_text, tools_deduped)
        if decision_markers:
            tools_deduped.extend(decision_markers)
        # Build tool_content from accumulated summaries, scrubbing secrets
        tool_content = "\n".join(s for s in current_tool_summaries if s)
        if tool_content:
            try:
                tool_content = scrub_text(tool_content)
            except Exception:
                logger.debug("scrub_text failed on tool content, using raw", exc_info=True)
        if len(tool_content) > 3000:
            tool_content = tool_content[:3000] + "..."
        short_id = session_id[:8]
        chunk = TranscriptChunk(
            id=f"{short_id}:t{turn_index}",
            session_id=session_id,
            timestamp=current_timestamp,
            turn_index=turn_index,
            user_text=(current_user_text[:1500] + "..." if len(current_user_text) > 1500
                      else current_user_text),
            assistant_text=assistant_text,
            tools_used=tools_deduped,
            files_touched=list(dict.fromkeys(current_files)),
            tool_content=tool_content,
            transcript_path=transcript_path,
            byte_offset=turn_start_offset,
            byte_length=_end - turn_start_offset,
        )
        chunks.append(chunk)
        turn_index += 1

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line_start = current_offset
            current_offset += len(raw_line.encode("utf-8"))
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type", "")

            # Skip noise types
            if entry_type in SKIP_TYPES:
                continue

            # Dedup: skip already-seen entries (cumulative snapshot overlap)
            uuid = entry.get("uuid", "")
            if uuid:
                dedup_key = (session_id, uuid)
                if dedup_key in seen_uuids:
                    continue
                seen_uuids.add(dedup_key)

            if _is_real_user_message(entry):
                # Flush previous turn — end boundary is this line's start
                # (so the previous turn doesn't include this new user message)
                _flush_turn(end_offset=line_start)
                turn_start_offset = line_start
                raw_user = _extract_user_text(entry)
                try:
                    current_user_text = scrub_text(raw_user)
                except Exception:
                    logger.debug("scrub_text failed on user text, using raw", exc_info=True)
                    current_user_text = raw_user
                current_assistant_texts = []
                current_tools = []
                current_files = []
                current_tool_summaries = []
                current_tool_use_map = {}
                current_timestamp = entry.get("timestamp", "")

            elif entry_type == "user":
                # Tool-result-only entry — capture AskUserQuestion responses
                # AND summarize all tool results for search indexing.
                if "AskUserQuestion" in current_tools:
                    result_text = _extract_tool_result_text(entry)
                    if result_text:
                        current_assistant_texts.append(f"User selected: {result_text}")
                # Summarize tool results
                content = entry.get("message", {}).get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_result":
                            continue
                        tool_use_id = block.get("tool_use_id", "")
                        result = block.get("content", "")
                        if isinstance(result, list):
                            # Extract text from nested blocks
                            result = "\n".join(
                                sub.get("text", "")
                                for sub in result
                                if isinstance(sub, dict) and sub.get("type") == "text"
                            )
                        if not isinstance(result, str):
                            continue
                        # Skip tool error results — infrastructure noise, not session context
                        if "<tool_use_error>" in result:
                            continue
                        tool_name, tool_input = current_tool_use_map.get(
                            tool_use_id, ("", {})
                        )
                        if tool_name:
                            summary = _summarize_tool_result(
                                tool_name, tool_input, result,
                            )
                            if summary:
                                current_tool_summaries.append(summary)

            elif entry_type == "assistant":
                text, tools, files, tool_uses = _extract_assistant_content(entry)
                if text:
                    try:
                        text = scrub_text(text)
                    except Exception:
                        logger.debug("scrub_text failed on assistant text, using raw", exc_info=True)
                    current_assistant_texts.append(text)
                current_tools.extend(tools)
                current_files.extend(files)
                # Track tool invocations for matching with results
                for tu_id, tu_name, tu_input in tool_uses:
                    current_tool_use_map[tu_id] = (tu_name, tu_input)
                    input_summary = _summarize_tool_input(tu_name, tu_input)
                    if input_summary:
                        current_tool_summaries.append(input_summary)

    # Flush last turn
    _flush_turn()

    return chunks


# ---------------------------------------------------------------------------
# TranscriptIndex
# ---------------------------------------------------------------------------

class TranscriptIndex:
    """Searchable index over transcript chunks.

    Supports BM25 keyword search and optional embedding-based semantic search.
    Chunks are organized by session and sorted newest-first for progressive
    time-ordered retrieval.
    """

    def __init__(
        self,
        chunks: list[TranscriptChunk],
        use_embeddings: bool = False,
        cache_dir: Path | None = None,
        db: RecallDB | None = None,
    ):
        # Sort by timestamp descending (most recent first)
        self.chunks = sorted(chunks, key=lambda c: c.timestamp, reverse=True)

        # Group by session for progressive search
        self.sessions: dict[str, list[TranscriptChunk]] = {}
        for chunk in self.chunks:
            self.sessions.setdefault(chunk.session_id, []).append(chunk)

        # Session order: most recent first (by latest timestamp in session)
        self._session_order = sorted(
            self.sessions.keys(),
            key=lambda sid: max(c.timestamp for c in self.sessions[sid]),
            reverse=True,
        )

        # Turn index lookup: (session_id, turn_index) -> chunk for O(1) preceding turn
        # Exclude journal chunks (turn_index=-1) to avoid collisions
        self._turn_lookup: dict[tuple[str, int], TranscriptChunk] = {
            (c.session_id, c.turn_index): c for c in self.chunks
            if c.turn_index >= 0
        }

        # Search diagnostics (populated on empty results for caller inspection)
        self._last_diagnostics: SearchDiagnostics | None = None

        # Query result cache — avoids re-computing BM25/embedding/RRF for
        # identical queries within the same index lifetime. LRU with max 32 entries.
        self._query_cache: dict[tuple, str] = {}
        self._query_cache_max = 32

        # Working memory (session-scoped, in-memory LRU)
        from synapt.recall.working_memory import WorkingMemory
        self._working_memory = WorkingMemory()

        # Current session ID (set by MCP server for access tracking)
        self._current_session_id: str = ""

        # Chunk ID -> index mapping (for rowid conversion)
        self._id_to_idx: dict[str, int] = {
            c.id: i for i, c in enumerate(self.chunks)
        }

        # SQLite backend (None = legacy BM25-only mode)
        self._db: RecallDB | None = db
        self._rowid_to_idx: dict[int, int] = {}
        self._idx_to_rowid: dict[int, int] = {}

        if db is not None and db.chunk_count() > 0:
            self._refresh_rowid_map()

        if self._rowid_to_idx:
            # FTS5 is ready — skip in-memory BM25
            self._bm25 = None
        else:
            # Build in-memory BM25 index (no DB or DB not yet populated)
            self._bm25 = BM25()
            corpus = []
            for chunk in self.chunks:
                tokens = _tokenize(chunk.text)
                # Boost tools and files by repeating them
                boost_text = " ".join(chunk.tools_used + chunk.files_touched)
                tokens.extend(_tokenize(boost_text))
                corpus.append(tokens)
            self._bm25.index(corpus)

        # Optional embedding index
        self._embeddings: list[list[float]] | None = None
        self._embed_provider = None
        # Pre-loaded embeddings for hybrid search (rowid -> vector)
        self._all_embeddings: dict[int, list[float]] = {}
        self._knowledge_embeddings: dict[int, list[float]] = {}
        # Track embedding status for user-facing messages
        self._embedding_status: str = "disabled"  # disabled | active | unavailable
        self._embedding_reason: str = ""
        if use_embeddings:
            try:
                from synapt.recall.embeddings import get_embedding_provider
                provider = get_embedding_provider()
                if provider:
                    self._embed_provider = provider
                    self._embedding_status = "active"
                    # Only build embeddings if storage is available
                    if self._db is None or self._idx_to_rowid:
                        self._load_or_build_embeddings(cache_dir)
                    # Load all embeddings into memory for hybrid search
                    if self._db is not None:
                        self._all_embeddings = self._db.get_all_embeddings()
                        self._knowledge_embeddings = (
                            self._db.get_knowledge_embeddings()
                        )
                else:
                    self._embedding_status = "unavailable"
                    self._embedding_reason = (
                        "No embedding provider found. "
                        "Install sentence-transformers for semantic search: "
                        "pip install sentence-transformers"
                    )
            except Exception as e:
                logger.warning("Embeddings unavailable: %s", e)
                self._embedding_status = "unavailable"
                self._embedding_reason = str(e)

        # Cross-encoder reranking (Phase 2)
        from synapt.recall.reranker import is_reranker_enabled
        self._use_reranker = is_reranker_enabled()
        if self._use_reranker:
            logger.info("Cross-encoder reranking enabled")

    def _rerank_candidates(
        self,
        query: str,
        candidates: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Apply cross-encoder reranking if enabled."""
        if not self._use_reranker:
            return candidates
        try:
            from synapt.recall.reranker import rerank
            return rerank(query, candidates, self.chunks)
        except Exception as e:
            logger.warning("Reranking failed, using original order: %s", e)
            return candidates

    def _refresh_rowid_map(self) -> None:
        """Build rowid <-> chunk-index mappings from the database."""
        if not self._db:
            return
        id_to_rowid = self._db.get_chunk_id_rowid_map()
        self._rowid_to_idx = {}
        self._idx_to_rowid = {}
        for i, chunk in enumerate(self.chunks):
            rowid = id_to_rowid.get(chunk.id)
            if rowid is not None:
                self._rowid_to_idx[rowid] = i
                self._idx_to_rowid[i] = rowid

    def _load_or_build_embeddings(self, cache_dir: Path | None):
        """Build or load cached embeddings for all chunks."""
        if self._db is not None:
            return self._load_or_build_embeddings_db()

        # Legacy file-based cache
        cache_path = cache_dir / "transcript_embeddings.json" if cache_dir else None
        content_hash = self._content_hash()

        # Try cache first
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("hash") == content_hash:
                    self._embeddings = cached["embeddings"]
                    return
            except (json.JSONDecodeError, KeyError):
                pass

        # Build embeddings
        texts = [c.text[:500] for c in self.chunks]  # cap text length
        try:
            all_embs = []
            for i in range(0, len(texts), 64):
                batch = texts[i:i + 64]
                all_embs.extend(self._embed_provider.embed(batch))
            self._embeddings = all_embs

            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"hash": content_hash, "embeddings": all_embs}, f)
        except Exception as e:
            logger.warning("Embedding build failed: %s", e)
            self._embeddings = None

    def _load_or_build_embeddings_db(self):
        """Build or load embeddings using the SQLite backend."""
        content_hash = self._content_hash()
        stored_hash = self._db.get_metadata("embedding_hash")

        if stored_hash == content_hash and self._db.has_embeddings():
            return  # Embeddings are current — fetch on demand during lookup

        # Build embeddings and store as per-row BLOBs
        texts = [c.text[:500] for c in self.chunks]
        try:
            all_embs = []
            for i in range(0, len(texts), 64):
                batch = texts[i:i + 64]
                all_embs.extend(self._embed_provider.embed(batch))

            emb_mapping: dict[int, list[float]] = {}
            for i, emb in enumerate(all_embs):
                rowid = self._idx_to_rowid.get(i)
                if rowid is not None:
                    emb_mapping[rowid] = emb
            if emb_mapping:
                self._db.save_embeddings(emb_mapping)
            self._db.set_metadata("embedding_hash", content_hash)

            # Also build embeddings for knowledge nodes
            self._build_knowledge_embeddings()
        except Exception as e:
            logger.warning("Embedding build failed: %s", e)

    def _build_knowledge_embeddings(self) -> None:
        """Build embeddings for knowledge nodes that don't have them yet."""
        if not self._db or not self._embed_provider:
            return
        try:
            missing = self._db.get_knowledge_rowids_without_embeddings()
            if not missing:
                return
            texts = [content[:500] for _, content in missing]
            rowids = [rowid for rowid, _ in missing]
            all_embs: list[list[float]] = []
            for i in range(0, len(texts), 64):
                batch = texts[i:i + 64]
                all_embs.extend(self._embed_provider.embed(batch))
            emb_mapping = dict(zip(rowids, all_embs))
            if emb_mapping:
                self._db.save_knowledge_embeddings(emb_mapping)
                logger.info(
                    "Built embeddings for %d knowledge nodes", len(emb_mapping),
                )
        except Exception as e:
            logger.warning("Knowledge embedding build failed: %s", e)

    def _content_hash(self) -> str:
        """Hash of chunk IDs + text content for embedding cache invalidation.

        Includes user_text, assistant_text, and tool_content so that
        re-parsed transcripts with different content (but same IDs)
        correctly invalidate cached embeddings.
        """
        h = hashlib.sha256()
        for c in self.chunks:
            h.update(
                f"{c.id}|{c.user_text}|{c.assistant_text}|{c.tool_content}\n"
                .encode()
            )
        return h.hexdigest()[:16]

    def _filter_by_date(
        self,
        after: str | None = None,
        before: str | None = None,
    ) -> set[int] | None:
        """Return indices of chunks within the date range, or None for all.

        Uses ISO 8601 string comparison — works because the format is
        lexicographically ordered. Accepts date-only ("2026-02-28") or
        datetime ("2026-02-28T10:00:00Z").
        """
        if not after and not before:
            return None
        valid = set()
        for i, chunk in enumerate(self.chunks):
            ts = chunk.timestamp
            if not ts:
                continue
            if after and ts < after:
                continue
            if before and ts >= before:
                continue
            valid.add(i)
        return valid

    def _apply_recency_decay(
        self,
        scores: list[float],
        half_life: float = 30.0,
        now: datetime | None = None,
    ) -> list[float]:
        """Multiply scores by exp(-age_days * ln2 / half_life) for time decay.

        Args:
            scores: Scores per chunk, same order as self.chunks.
            half_life: Days for score to decay to ~50%. Default 30.
            now: Reference time. Defaults to UTC now.

        Returns:
            New list of decayed scores.
        """
        if half_life <= 0:
            return list(scores)
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        decay_rate = math.log(2) / half_life
        result = list(scores)
        for i, chunk in enumerate(self.chunks):
            if result[i] == 0.0 or not chunk.timestamp:
                continue
            try:
                ts = datetime.fromisoformat(chunk.timestamp.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
                result[i] *= math.exp(-decay_rate * age_days)
            except (ValueError, TypeError):
                pass
        return result

    def _decay_candidates(
        self,
        candidates: list[tuple[int, float]],
        half_life: float,
        now: datetime | None = None,
    ) -> list[tuple[int, float]]:
        """Apply recency decay to (chunk_idx, score) candidate pairs.

        Same formula as _apply_recency_decay but for sparse candidate lists.
        """
        if half_life <= 0 or not candidates:
            return candidates
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        decay_rate = math.log(2) / half_life
        for j, (idx, score) in enumerate(candidates):
            chunk = self.chunks[idx]
            if not chunk.timestamp:
                continue
            try:
                ts = datetime.fromisoformat(
                    chunk.timestamp.replace("Z", "+00:00")
                )
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
                candidates[j] = (idx, score * math.exp(-decay_rate * age_days))
            except (ValueError, TypeError):
                pass
        return candidates

    def _apply_threshold_with_diagnostics(
        self,
        candidates: list[tuple[int, float]],
        threshold_ratio: float,
        search_mode: str,
        date_filter_active: bool = False,
        sessions_searched: int = 0,
    ) -> list[tuple[int, float]]:
        """Apply threshold filtering and record diagnostics if no candidates exist.

        Expects *candidates* sorted by score descending (highest first), with
        only positive scores (pre-filtered by caller).  Sets
        ``self._last_diagnostics`` when candidates is empty (``no_matches``).
        """
        if not candidates:
            self._last_diagnostics = SearchDiagnostics(
                total_chunks=len(self.chunks),
                total_sessions=len(self.sessions),
                candidates_found=0,
                search_mode=search_mode,
                date_filter_active=date_filter_active,
                sessions_searched=sessions_searched,
                reason="no_matches",
            )
            return []

        if threshold_ratio > 0:
            cutoff = candidates[0][1] * threshold_ratio
            filtered = [(i, s) for i, s in candidates if s >= cutoff]
        else:
            filtered = candidates

        return filtered

    def _get_preceding_turn(self, chunk: TranscriptChunk) -> TranscriptChunk | None:
        """Find the chunk with turn_index - 1 in the same session (O(1) lookup)."""
        if chunk.turn_index == 0:
            return None
        return self._turn_lookup.get((chunk.session_id, chunk.turn_index - 1))

    def lookup(
        self,
        query: str,
        max_chunks: int = 5,
        max_tokens: int = 500,
        max_sessions: int | None = None,
        after: str | None = None,
        before: str | None = None,
        half_life: float | None = None,
        threshold_ratio: float = 0.2,
        depth: str = "full",
        include_archived: bool = False,
        include_historical: bool = False,
    ) -> str:
        """Search transcripts and return formatted context string.

        Args:
            query: Search query (natural language or keywords).
            max_chunks: Maximum number of chunks to return.
            max_tokens: Approximate token budget (~4 chars/token).
            max_sessions: If set, only search N most recent sessions
                         (progressive mode). None = search all (global mode).
            after: Only include chunks with timestamp >= this ISO 8601 string.
            before: Only include chunks with timestamp < this ISO 8601 string.
            half_life: Days for recency decay to reach ~50%. 0 disables.
                None = use intent-based default (or 30.0 as fallback).
            threshold_ratio: Drop results below this fraction of top score. 0 disables.
            depth: "full" (default) = knowledge + journal + transcript.
                   "summary" = knowledge + journal only (no raw transcripts).
            include_archived: If True, include archived clusters in concise mode.
                In full mode, individual chunks are always searchable regardless.
            include_historical: If True, include superseded/contradicted knowledge
                nodes in results, clearly labeled as historical.

        Returns:
            Formatted string ready for prompt injection, or empty string.
            When empty, ``self._last_diagnostics`` explains why.
        """
        self._last_diagnostics = None

        if not self.chunks:
            self._last_diagnostics = SearchDiagnostics(reason="empty_index")
            return ""

        # Check query cache — skip if max_tokens=0 (diagnostics-only mode)
        cache_key = (
            query, max_chunks, max_sessions, after, before,
            half_life, threshold_ratio, depth, include_archived,
            include_historical,
        )
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        # Query intent classification — adjusts search parameters based on
        # the type of information being sought (recency, embedding weight,
        # knowledge boost). Only overrides half_life when caller didn't
        # provide an explicit value (half_life is None).
        emb_weight = 1.0
        knowledge_boost = 2.0
        caller_set_half_life = half_life is not None
        if half_life is None:
            half_life = 60.0
        try:
            from synapt.recall.hybrid import (
                classify_query_intent, intent_search_params,
                extract_temporal_range,
            )
            intent = classify_query_intent(query)
            params = intent_search_params(intent)
            emb_weight = params.get("emb_weight", emb_weight)
            knowledge_boost = params.get("knowledge_boost", knowledge_boost)
            if not caller_set_half_life:
                half_life = params.get("half_life", half_life)

            # Auto-extract date range from temporal expressions in query
            # (only when caller didn't provide explicit date filters)
            if after is None and before is None:
                extracted_after, extracted_before = extract_temporal_range(query)
                if extracted_after is not None:
                    after = extracted_after
                    before = extracted_before
                    logger.debug(
                        "Auto-extracted date range: after=%s, before=%s",
                        after, before,
                    )
        except Exception:
            logger.debug("Intent classification failed, using defaults", exc_info=True)

        query_tokens = _tokenize(query)
        if not query_tokens:
            self._last_diagnostics = SearchDiagnostics(
                total_chunks=len(self.chunks),
                total_sessions=len(self.sessions),
                reason="empty_query",
            )
            return ""

        date_filter = self._filter_by_date(after, before)

        if max_sessions is not None:
            result = self._progressive_lookup(
                query, query_tokens, max_chunks, max_tokens, max_sessions,
                date_filter, half_life, threshold_ratio, depth,
                include_historical,
                emb_weight=emb_weight, knowledge_boost=knowledge_boost,
            )
        else:
            result = self._global_lookup(
                query, query_tokens, max_chunks, max_tokens, date_filter,
                half_life, threshold_ratio, depth, include_archived,
                include_historical,
                emb_weight=emb_weight, knowledge_boost=knowledge_boost,
            )

        # Cache the result (LRU eviction when full)
        if len(self._query_cache) >= self._query_cache_max:
            # Evict oldest entry
            oldest = next(iter(self._query_cache))
            del self._query_cache[oldest]
        self._query_cache[cache_key] = result
        return result

    def _global_lookup(
        self,
        query: str,
        query_tokens: list[str],
        max_chunks: int,
        max_tokens: int,
        date_filter: set[int] | None = None,
        half_life: float = 30.0,
        threshold_ratio: float = 0.2,
        depth: str = "full",
        include_archived: bool = False,
        include_historical: bool = False,
        emb_weight: float = 1.0,
        knowledge_boost: float = 2.0,
    ) -> str:
        """Score all chunks globally, return top-K."""
        if self._db and self._rowid_to_idx:
            return self._global_lookup_fts(
                query, max_chunks, max_tokens, date_filter,
                half_life, threshold_ratio, depth, include_archived,
                include_historical,
                emb_weight=emb_weight, knowledge_boost=knowledge_boost,
            )
        return self._global_lookup_bm25(
            query, query_tokens, max_chunks, max_tokens, date_filter,
            half_life, threshold_ratio, depth,
            emb_weight=emb_weight,
        )

    def _global_lookup_fts(
        self,
        query: str,
        max_chunks: int,
        max_tokens: int,
        date_filter: set[int] | None,
        half_life: float,
        threshold_ratio: float,
        depth: str = "full",
        include_archived: bool = False,
        include_historical: bool = False,
        emb_weight: float = 1.0,
        knowledge_boost: float = 2.0,
    ) -> str:
        """Global lookup using FTS5 (SQLite backend)."""
        # Search knowledge nodes first (ranked higher via boost)
        knowledge_results = self._search_knowledge(
            query, max_chunks, include_historical=include_historical,
            knowledge_boost=knowledge_boost, emb_weight=emb_weight,
        )

        # Concise mode: search clusters directly, return only summaries
        if depth == "concise":
            return self._concise_lookup(
                query, max_chunks, max_tokens, knowledge_results,
                include_archived,
            )

        fts_results = self._db.fts_search(query, limit=max_chunks * 10)

        # Convert rowids to chunk indices, applying date and depth filters
        candidates: list[tuple[int, float]] = []
        for rowid, score in fts_results:
            idx = self._rowid_to_idx.get(rowid)
            if idx is None:
                continue
            if date_filter is not None and idx not in date_filter:
                continue
            # In summary mode, only include journal chunks (turn_index == -1)
            if depth == "summary" and self.chunks[idx].turn_index >= 0:
                continue
            candidates.append((idx, score))

        # Recency decay — applied BEFORE RRF intentionally.
        # RRF is rank-based (not score-based), so decaying scores here biases
        # the *ranks* that RRF sees. Both BM25 and embedding lists are decayed
        # symmetrically, so recent results rank higher in both lists and RRF
        # naturally promotes them. This differs from pure RRF (score-agnostic)
        # but produces better results for session recall where recency matters.
        candidates = self._decay_candidates(candidates, half_life)

        # Hybrid search: RRF fusion between BM25/FTS and embedding similarity.
        # Unlike the old additive boost (score + sim * 3.0), RRF can surface
        # results that BM25 missed entirely — critical for paraphrased queries.
        if self._embed_provider and self._all_embeddings:
            try:
                from synapt.recall.hybrid import (
                    embedding_search, weighted_rrf_merge,
                )
                q_emb = self._embed_provider.embed_single(query)

                # BM25/FTS ranked list (idx, score)
                bm25_ranked = sorted(candidates, key=lambda x: x[1], reverse=True)

                # Embedding ranked list (rowid, sim) → convert to (idx, sim)
                emb_raw = embedding_search(
                    q_emb, self._all_embeddings, limit=max_chunks * 10,
                )
                emb_ranked = []
                for rowid, sim in emb_raw:
                    idx = self._rowid_to_idx.get(rowid)
                    if idx is None:
                        continue
                    if date_filter is not None and idx not in date_filter:
                        continue
                    if depth == "summary" and self.chunks[idx].turn_index >= 0:
                        continue
                    emb_ranked.append((idx, sim))
                emb_ranked = self._decay_candidates(emb_ranked, half_life)

                # Weighted RRF merge — emb_weight from intent classification
                merged = weighted_rrf_merge(
                    bm25_ranked, emb_ranked, emb_weight=emb_weight,
                )
                candidates = merged
            except Exception as e:
                logger.warning("Hybrid search failed, using BM25 only: %s", e)
                candidates.sort(key=lambda x: x[1], reverse=True)
        else:
            candidates.sort(key=lambda x: x[1], reverse=True)

        top = [(i, s) for i, s in candidates[:max_chunks * 2] if s > 0]

        top = self._apply_threshold_with_diagnostics(
            top, threshold_ratio, "fts_global",
            date_filter_active=date_filter is not None,
        )

        # Cross-encoder reranking (Phase 2): rerank after threshold so
        # threshold operates on RRF scores (correct domain), not cross-encoder
        # logits which can be negative and break ratio-based filtering.
        top = self._rerank_candidates(query, top)

        # Pass extra candidates to _format_results to compensate for
        # near-duplicate filtering — the token budget is the real limiter.
        dedup_headroom = _dedup_limit(max_chunks)
        return self._format_results(
            top[:dedup_headroom], max_tokens,
            knowledge_results=knowledge_results,
            query=query,
        )

    def _concise_lookup(
        self,
        query: str,
        max_chunks: int,
        max_tokens: int,
        knowledge_results: list[dict] | None = None,
        include_archived: bool = False,
    ) -> str:
        """Search clusters directly and return only summaries.

        Used by depth="concise" — shows high-level topic overview without
        individual chunk details. Each cluster is rendered as a compact block.
        """
        cluster_hits = self._db.cluster_fts_search(
            query, limit=max_chunks * 3, include_archived=include_archived,
        )
        # Fallback: if cluster FTS found nothing, search chunks and map to
        # parent clusters so concise mode has the same coverage as full mode.
        if not cluster_hits and self._db:
            cluster_hits = self._db.chunk_fts_to_clusters(
                query, limit=max_chunks * 3, include_archived=include_archived,
            )
        if not cluster_hits and not knowledge_results:
            return ""

        wm = self._working_memory
        lines = ["Past session context:"]
        token_count = 0
        access_items: list[dict] = []

        # Knowledge nodes first
        for node in (knowledge_results or []):
            block = self._format_knowledge_block(node)
            block_tokens = len(block) // 4
            if token_count + block_tokens > max_tokens and len(lines) > 1:
                break
            lines.append(block)
            token_count += block_tokens
            item_id = node.get("id", "")
            access_items.append({"item_type": "knowledge", "item_id": item_id})
            wm.record("knowledge", item_id, node.get("content", ""))

        # Cluster summaries
        for cluster_id, score in cluster_hits:
            info = self._db.get_cluster(cluster_id)
            if info is None:
                continue
            block = self._format_cluster_block(cluster_id, info)
            block_tokens = len(block) // 4
            if token_count + block_tokens > max_tokens and len(lines) > 1:
                break
            lines.append(block)
            token_count += block_tokens
            access_items.append({"item_type": "cluster", "item_id": cluster_id})
            wm.record("cluster", cluster_id, info.get("topic", ""))

        # Record access (fire-and-forget)
        for item in access_items:
            if query:
                item["query"] = query
            if self._current_session_id:
                item["session_id"] = self._current_session_id
        if access_items and self._db:
            try:
                self._db.record_access(access_items, context="search")
            except Exception:
                pass

            try:
                from synapt.recall.promotion import (
                    check_promotions, execute_cheap_promotions,
                )
                for item in access_items:
                    actions = check_promotions(
                        self._db, item["item_type"], item["item_id"],
                    )
                    if actions:
                        execute_cheap_promotions(
                            self._db, item["item_type"],
                            item["item_id"], actions,
                        )
            except Exception:
                pass

        if len(lines) <= 1:
            return ""
        return "\n".join(lines)

    def _global_lookup_bm25(
        self,
        query: str,
        query_tokens: list[str],
        max_chunks: int,
        max_tokens: int,
        date_filter: set[int] | None,
        half_life: float,
        threshold_ratio: float,
        depth: str = "full",
        emb_weight: float = 1.0,
    ) -> str:
        """Global lookup using in-memory BM25 (legacy fallback)."""
        # BM25 path has no cluster FTS — concise mode requires FTS5
        if depth == "concise":
            return ""

        scores = self._bm25.score(query_tokens)

        # In summary mode, zero out non-journal chunks
        if depth == "summary":
            for i, chunk in enumerate(self.chunks):
                if chunk.turn_index >= 0:
                    scores[i] = 0.0

        if date_filter is not None:
            for i in range(len(scores)):
                if i not in date_filter:
                    scores[i] = 0.0

        if half_life > 0:
            scores = self._apply_recency_decay(scores, half_life=half_life)

        # Hybrid search: RRF fusion (same as FTS path but using in-memory BM25)
        if self._embeddings and self._embed_provider:
            try:
                from synapt.recall.hybrid import (
                    embedding_search, weighted_rrf_merge,
                )
                q_emb = self._embed_provider.embed_single(query)

                # BM25 ranked list
                bm25_ranked = sorted(
                    [(i, s) for i, s in enumerate(scores) if s > 0],
                    key=lambda x: x[1], reverse=True,
                )

                # Embedding ranked list (using in-memory embeddings)
                emb_dict = {i: emb for i, emb in enumerate(self._embeddings)}
                emb_raw = embedding_search(q_emb, emb_dict, limit=max_chunks * 10)
                emb_ranked = [
                    (i, s) for i, s in emb_raw
                    if date_filter is None or i in date_filter
                ]
                if depth == "summary":
                    emb_ranked = [
                        (i, s) for i, s in emb_ranked
                        if self.chunks[i].turn_index < 0
                    ]

                merged = weighted_rrf_merge(
                    bm25_ranked, emb_ranked, emb_weight=emb_weight,
                )
                top = [(i, s) for i, s in merged[:max_chunks * 2] if s > 0]
            except Exception as e:
                logger.warning("Hybrid search failed, using BM25 only: %s", e)
                ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                top = [(i, s) for i, s in ranked[:max_chunks * 2] if s > 0]
        else:
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            top = [(i, s) for i, s in ranked[:max_chunks * 2] if s > 0]

        top = self._apply_threshold_with_diagnostics(
            top, threshold_ratio, "bm25_global",
            date_filter_active=date_filter is not None,
        )

        # Cross-encoder reranking (Phase 2): after threshold (see fts_global)
        top = self._rerank_candidates(query, top)

        dedup_headroom = _dedup_limit(max_chunks)
        return self._format_results(top[:dedup_headroom], max_tokens, query=query)

    def _progressive_lookup(
        self,
        query: str,
        query_tokens: list[str],
        max_chunks: int,
        max_tokens: int,
        max_sessions: int,
        date_filter: set[int] | None = None,
        half_life: float = 30.0,
        threshold_ratio: float = 0.2,
        depth: str = "full",
        include_historical: bool = False,
        emb_weight: float = 1.0,
        knowledge_boost: float = 2.0,
    ) -> str:
        """Search sessions newest-first, stop early if enough hits found."""
        # Knowledge results are always searched globally (not per-session)
        knowledge_results = (
            self._search_knowledge(
                query, max_chunks, include_historical=include_historical,
                knowledge_boost=knowledge_boost, emb_weight=emb_weight,
            )
            if self._db else []
        )

        if self._db and self._rowid_to_idx:
            return self._progressive_lookup_fts(
                query, max_chunks, max_tokens, max_sessions,
                date_filter, half_life, threshold_ratio,
                knowledge_results=knowledge_results,
                emb_weight=emb_weight,
                depth=depth,
            )
        return self._progressive_lookup_bm25(
            query, query_tokens, max_chunks, max_tokens, max_sessions,
            date_filter, half_life, threshold_ratio,
            depth=depth,
        )

    def _progressive_lookup_fts(
        self,
        query: str,
        max_chunks: int,
        max_tokens: int,
        max_sessions: int,
        date_filter: set[int] | None,
        half_life: float,
        threshold_ratio: float,
        knowledge_results: list[dict] | None = None,
        depth: str = "full",
        emb_weight: float = 1.0,
    ) -> str:
        """Progressive session search using FTS5 (SQLite backend)."""
        hits: list[tuple[int, float]] = []
        sessions_searched = 0
        BM25_THRESHOLD = 2.0

        for session_id in self._session_order:
            if sessions_searched >= max_sessions:
                break

            fts_results = self._db.fts_search_by_session(
                query, [session_id], limit=max_chunks * 5,
            )
            session_hits: list[tuple[int, float]] = []
            for rowid, score in fts_results:
                idx = self._rowid_to_idx.get(rowid)
                if idx is None:
                    continue
                if date_filter is not None and idx not in date_filter:
                    continue
                # In summary mode, skip raw transcript chunks (keep journal only)
                if depth == "summary" and self.chunks[idx].turn_index >= 0:
                    continue
                session_hits.append((idx, score))

            if not session_hits:
                continue

            hits.extend(session_hits)
            sessions_searched += 1

            high_quality = sum(1 for _, s in hits if s > BM25_THRESHOLD)
            if high_quality >= 3 and len(hits) >= max_chunks:
                break

        # Recency decay — applied BEFORE RRF intentionally (see _global_lookup_fts
        # for rationale). Both BM25 and embedding lists are decayed symmetrically
        # so RRF ranks reflect recency bias consistently.
        hits = self._decay_candidates(hits, half_life)

        # Hybrid: merge FTS progressive hits with global embedding search.
        # Progressive FTS is session-scoped, but embeddings intentionally
        # search globally — semantic recall should be broader than keyword
        # recall, surfacing paraphrased matches from any session even when
        # max_sessions limits the BM25 search window.
        if self._embed_provider and self._all_embeddings:
            try:
                from synapt.recall.hybrid import (
                    embedding_search, weighted_rrf_merge,
                )
                q_emb = self._embed_provider.embed_single(query)
                emb_raw = embedding_search(
                    q_emb, self._all_embeddings, limit=max_chunks * 5,
                )
                emb_ranked = []
                for rowid, sim in emb_raw:
                    idx = self._rowid_to_idx.get(rowid)
                    if idx is None:
                        continue
                    if date_filter is not None and idx not in date_filter:
                        continue
                    if depth == "summary" and self.chunks[idx].turn_index >= 0:
                        continue
                    emb_ranked.append((idx, sim))
                emb_ranked = self._decay_candidates(emb_ranked, half_life)

                bm25_ranked = sorted(hits, key=lambda x: x[1], reverse=True)
                merged = weighted_rrf_merge(
                    bm25_ranked, emb_ranked, emb_weight=emb_weight,
                )
                hits = merged
            except Exception as e:
                logger.warning("Progressive hybrid failed: %s", e)
                hits.sort(key=lambda x: x[1], reverse=True)
        else:
            hits.sort(key=lambda x: x[1], reverse=True)

        # Bound candidates before expensive operations
        hits = hits[:max_chunks * 2]

        hits = self._apply_threshold_with_diagnostics(
            hits, threshold_ratio, "fts_progressive",
            date_filter_active=date_filter is not None,
            sessions_searched=sessions_searched,
        )

        # Cross-encoder reranking (Phase 2): after threshold (see fts_global)
        hits = self._rerank_candidates(query, hits)

        return self._format_results(
            hits[:max_chunks], max_tokens,
            knowledge_results=knowledge_results,
            query=query,
        )

    def _progressive_lookup_bm25(
        self,
        query: str,
        query_tokens: list[str],
        max_chunks: int,
        max_tokens: int,
        max_sessions: int,
        date_filter: set[int] | None,
        half_life: float,
        threshold_ratio: float,
        depth: str = "full",
    ) -> str:
        """Progressive session search using in-memory BM25 (legacy fallback)."""
        bm25_scores = self._bm25.score(query_tokens)

        # In summary mode, zero out raw transcript chunks (keep journal only)
        if depth == "summary":
            for i, chunk in enumerate(self.chunks):
                if chunk.turn_index >= 0:
                    bm25_scores[i] = 0.0

        if half_life > 0:
            bm25_scores = self._apply_recency_decay(bm25_scores, half_life=half_life)

        session_chunk_indices: dict[str, list[int]] = {}
        for i, chunk in enumerate(self.chunks):
            if date_filter is not None and i not in date_filter:
                continue
            session_chunk_indices.setdefault(chunk.session_id, []).append(i)

        hits: list[tuple[int, float]] = []
        sessions_searched = 0
        BM25_THRESHOLD = 2.0

        for session_id in self._session_order:
            if sessions_searched >= max_sessions:
                break
            indices = session_chunk_indices.get(session_id, [])
            if not indices:
                continue
            session_hits = [(i, bm25_scores[i]) for i in indices if bm25_scores[i] > 0]
            session_hits.sort(key=lambda x: x[1], reverse=True)
            hits.extend(session_hits)
            sessions_searched += 1

            high_quality = sum(1 for _, s in hits if s > BM25_THRESHOLD)
            if high_quality >= 3 and len(hits) >= max_chunks:
                break

        hits.sort(key=lambda x: x[1], reverse=True)

        # Bound candidates before expensive operations
        hits = hits[:max_chunks * 2]

        hits = self._apply_threshold_with_diagnostics(
            hits, threshold_ratio, "bm25_progressive",
            date_filter_active=date_filter is not None,
            sessions_searched=sessions_searched,
        )

        # Cross-encoder reranking (Phase 2): after threshold (see fts_global)
        hits = self._rerank_candidates(query, hits)

        return self._format_results(
            hits[:max_chunks], max_tokens,
            query=query,
        )

    def _search_knowledge(
        self,
        query: str,
        max_results: int = 5,
        include_historical: bool = False,
        knowledge_boost: float = 2.0,
        emb_weight: float = 1.0,
    ) -> list[dict]:
        """Search knowledge nodes via FTS5 + embedding hybrid.

        Returns a list of knowledge node dicts with an added 'score' key.
        Knowledge nodes are boosted by ``knowledge_boost`` vs chunk scores.
        If include_historical is True, also returns contradicted/superseded nodes.
        """
        if not self._db:
            return []
        try:
            fts_hits = self._db.knowledge_fts_search(
                query, limit=max_results * 2,
                include_historical=include_historical,
            )

            # Hybrid: also search knowledge embeddings
            emb_hits: list[tuple[int, float]] = []
            if self._embed_provider and self._knowledge_embeddings:
                try:
                    from synapt.recall.hybrid import embedding_search
                    q_emb = self._embed_provider.embed_single(query)
                    emb_hits = embedding_search(
                        q_emb, self._knowledge_embeddings,
                        limit=max_results * 2,
                    )
                except Exception:
                    pass

            # Merge FTS and embedding results via RRF
            if fts_hits and emb_hits:
                from synapt.recall.hybrid import weighted_rrf_merge
                merged = weighted_rrf_merge(
                    fts_hits, emb_hits, emb_weight=emb_weight,
                )
            elif fts_hits:
                merged = fts_hits
            elif emb_hits:
                merged = emb_hits
            else:
                return []

            all_rowids = [r for r, _ in merged]
            # Use RRF-fused scores (not raw FTS/embedding scores) for ranking
            score_map = dict(merged)

            nodes_by_rowid = self._db.knowledge_by_rowid(all_rowids)
            # Coverage gate: require ≥2 distinct query tokens to appear in the
            # node content, OR a strong embedding match (sim > 0.4). This lets
            # semantic matches through even when BM25 tokens don't overlap.
            query_tokens = set(_tokenize(query))
            min_matches = max(2, round(len(query_tokens) * 0.3))
            emb_rowids = {r for r, s in emb_hits if s > 0.4}
            results = []
            seen_ids: set[str] = set()
            for rowid in all_rowids:
                node = nodes_by_rowid.get(rowid)
                if not node:
                    continue
                node_id = node.get("id", str(rowid))
                if node_id in seen_ids:
                    continue
                node_tokens = set(_tokenize(node.get("content", "")))
                token_overlap = len(query_tokens & node_tokens) >= min_matches
                strong_emb = rowid in emb_rowids
                if token_overlap or strong_emb:
                    conf = node.get("confidence", 0.5)
                    if not isinstance(conf, (int, float)):
                        conf = 0.5
                    # Confidence-weighted boost: well-corroborated facts
                    # (conf=0.9 → 2.8x) rank above single-session
                    # observations (conf=0.45 → 1.9x).
                    effective_boost = 1.0 + conf * knowledge_boost
                    node["score"] = score_map.get(rowid, 0.0) * effective_boost
                    results.append(node)
                    seen_ids.add(node_id)

            # Confidence-based dedup: when multiple nodes share a lineage
            # (i.e. one superseded another), keep the highest-confidence one
            # as primary and demote others. This serves as the automatic
            # fallback when no user has resolved the contradiction.
            if include_historical:
                results = self._dedup_knowledge_by_lineage(results)

            # Co-retrieval conflict detection (Phase 8c): when two active
            # nodes in the same category have divergent content, queue a
            # pending contradiction for user review. Has its own try/except
            # internally — never disrupts search results.
            self._detect_co_retrieval_conflicts(results)

            return results
        except Exception:
            return []

    @staticmethod
    def _dedup_knowledge_by_lineage(nodes: list[dict]) -> list[dict]:
        """Within each lineage, keep only the highest-confidence node.

        Nodes without a lineage_id pass through unchanged. When two nodes
        share a lineage, the lower-confidence one is dropped — this is the
        confidence-based fallback for unresolved contradictions.
        """
        lineage_best: dict[str, dict] = {}
        no_lineage: list[dict] = []
        for node in nodes:
            lid = node.get("lineage_id", "")
            if not lid:
                no_lineage.append(node)
                continue
            existing = lineage_best.get(lid)
            if existing is None or node.get("confidence", 0) > existing.get("confidence", 0):
                lineage_best[lid] = node
        return no_lineage + list(lineage_best.values())

    def _detect_co_retrieval_conflicts(self, results: list[dict]) -> None:
        """Detect conflicting knowledge nodes in the same result set.

        When two active nodes share a category but have low keyword overlap
        (Jaccard < 0.3), they may contradict each other. Queues a pending
        contradiction for user review. Best-effort: never fails the search.
        """
        if not self._db or len(results) < 2:
            return
        # Only consider active, confident nodes
        active = [
            n for n in results
            if n.get("status", "active") == "active"
            and n.get("confidence", 0) >= 0.4
        ]
        if len(active) < 2:
            return
        try:
            # Pre-compute token sets once per node (avoids redundant
            # _tokenize calls in the pairwise loop)
            token_cache: dict[str, set[str]] = {}
            for node in active:
                nid = node.get("id", "")
                token_cache[nid] = set(_tokenize(node.get("content", "")))

            # Load all pending old_node_ids in one query
            pending_ids = {
                p["old_node_id"]
                for p in self._db.list_pending_contradictions()
            }

            # Group by category
            by_cat: dict[str, list[dict]] = {}
            for node in active:
                by_cat.setdefault(node.get("category", ""), []).append(node)

            for cat_nodes in by_cat.values():
                if len(cat_nodes) < 2:
                    continue
                for i in range(len(cat_nodes)):
                    for j in range(i + 1, len(cat_nodes)):
                        a, b = cat_nodes[i], cat_nodes[j]
                        # Skip nodes already in the same lineage
                        lid_a = a.get("lineage_id", "")
                        lid_b = b.get("lineage_id", "")
                        if lid_a and lid_a == lid_b:
                            continue
                        # Compute keyword divergence
                        id_a = a.get("id", "")
                        id_b = b.get("id", "")
                        kw_a = token_cache.get(id_a, set())
                        kw_b = token_cache.get(id_b, set())
                        if not kw_a or not kw_b:
                            continue
                        jaccard = len(kw_a & kw_b) / len(kw_a | kw_b)
                        if jaccard >= 0.3:
                            continue  # Similar enough — not a conflict
                        # Check dedup: skip if already pending for either node
                        if id_a in pending_ids or id_b in pending_ids:
                            continue
                        # Queue the lower-confidence node as the "old" one
                        if a.get("confidence", 0) >= b.get("confidence", 0):
                            old, new = b, a
                        else:
                            old, new = a, b
                        self._db.add_pending_contradiction(
                            old_node_id=old["id"],
                            new_content=new.get("content", ""),
                            category=old.get("category", ""),
                            reason=f"Co-retrieved with conflicting node [{new['id']}]",
                            detected_by="co-retrieval",
                        )
                        pending_ids.add(old["id"])  # Prevent further dups this batch
                        logger.debug(
                            "Co-retrieval conflict: [%s] vs [%s] (jaccard=%.2f)",
                            old["id"], new["id"], jaccard,
                        )
        except Exception:
            pass  # Best-effort — never disrupt search

    @staticmethod
    def _format_knowledge_block(node: dict) -> str:
        """Format a single knowledge node for search results."""
        conf = node.get("confidence", 0.5)
        conf_label = "high" if conf >= 0.7 else "medium" if conf >= 0.4 else "low"
        category = node.get("category", "unknown")
        content = node.get("content", "")
        sources = node.get("source_sessions", [])
        updated = (node.get("updated_at", "") or "")[:10]
        status = node.get("status", "active")

        # Historical label for superseded/contradicted nodes
        if status == "contradicted":
            header = f"--- [knowledge, historical] {category} ---"
        else:
            header = f"--- [knowledge] {category} ({conf_label} confidence) ---"

        # Temporal metadata
        meta_parts = [f"Based on {len(sources)} session(s), last updated {updated}"]
        valid_from = node.get("valid_from")
        valid_until = node.get("valid_until")
        if valid_from and valid_until:
            meta_parts.append(f"Valid {valid_from[:10]} to {valid_until[:10]}")
        elif valid_from:
            meta_parts.append(f"Current since {valid_from[:10]}")

        # Supersession note
        note = node.get("contradiction_note", "")
        if status == "contradicted" and note:
            meta_parts.append(f"Superseded: {note}")

        return f"{header}\n{content}\n[{'. '.join(meta_parts)}]"

    def _format_results(
        self,
        ranked: list[tuple[int, float]],
        max_tokens: int,
        knowledge_results: list[dict] | None = None,
        query: str = "",
    ) -> str:
        """Format ranked chunk indices into a context string.

        When clusters exist in the DB, chunks sharing a cluster are grouped
        and displayed as a single cluster summary block — saving token budget
        for more diverse results. Chunks not in any cluster (or sole
        representatives of a cluster) display as raw chunk blocks.
        """
        if not ranked and not knowledge_results:
            return ""

        lines = ["Past session context:"]
        token_count = 0

        # Build cluster grouping and interleave by relevance score
        cluster_groups, ungrouped = self._group_by_cluster(ranked)

        # Build a unified emit list sorted by relevance.
        # Knowledge nodes, clusters, and raw chunks all compete on score.
        # Each item: (score, block_text, item_type, item_id) for access tracking.
        emit_items: list[tuple[float, str, str, str]] = []

        # Include knowledge nodes in the unified ranking instead of
        # unconditionally prepending them (#284). They compete fairly
        # with chunks/clusters — irrelevant knowledge nodes drop below
        # high-quality transcript chunks.
        for node in (knowledge_results or []):
            block = self._format_knowledge_block(node)
            node_score = node.get("score", 0.0)
            emit_items.append((node_score, block, "knowledge", node.get("id", "")))

        for cluster_id, (cluster_info, member_indices, max_score) in cluster_groups.items():
            block = self._format_cluster_block(cluster_id, cluster_info)
            emit_items.append((max_score, block, "cluster", cluster_id))

        for idx, score in ungrouped:
            block = self._format_chunk_block(idx)
            emit_items.append((score, block, "chunk", self.chunks[idx].id))

        # Apply adaptive score boosts before final sort:
        # 1. Working memory: items seen recently this session (1.5x / 2.0x)
        # 2. Access frequency: items drilled into in past sessions (up to 1.3x)
        wm = self._working_memory
        for i, (score, block, item_type, item_id) in enumerate(emit_items):
            boosted = wm.boost_score(score, item_id)
            boosted = self._access_frequency_boost(boosted, item_type, item_id)
            if boosted != score:
                emit_items[i] = (boosted, block, item_type, item_id)

        # Sort by score descending (highest relevance first)
        emit_items.sort(key=lambda x: x[0], reverse=True)

        # Deduplication: skip chunks that are near-duplicates of already-
        # emitted items (Jaccard similarity > 0.6 on token sets). This frees
        # token budget for diverse results, especially on multi-hop queries.
        from synapt.recall.clustering import _jaccard

        emitted_token_sets: list[set[str]] = []

        emitted_access: list[dict] = []
        for score, block, item_type, item_id in emit_items:
            block_tokens_set = set(_tokenize(block))
            if block_tokens_set and emitted_token_sets:
                if any(_jaccard(block_tokens_set, prev) > _DEDUP_JACCARD_THRESHOLD
                       for prev in emitted_token_sets):
                    continue  # Skip near-duplicate
            block_tokens = len(block) // 4
            if token_count + block_tokens > max_tokens and len(lines) > 1:
                break
            lines.append(block)
            token_count += block_tokens
            emitted_token_sets.append(block_tokens_set)
            access_entry: dict = {
                "item_type": item_type,
                "item_id": item_id,
                "score": score,
            }
            emitted_access.append(access_entry)

        # Record access for all emitted items (fire-and-forget)
        all_access = emitted_access
        for item in all_access:
            if query:
                item["query"] = query
            if self._current_session_id:
                item["session_id"] = self._current_session_id
        if all_access and self._db:
            try:
                self._db.record_access(all_access, context="search")
            except Exception:
                pass  # Never fail a search due to access tracking

            # Check promotions for emitted items (cheap actions only)
            try:
                from synapt.recall.promotion import (
                    check_promotions, execute_cheap_promotions,
                )
                for item in all_access:
                    actions = check_promotions(
                        self._db, item["item_type"], item["item_id"],
                    )
                    if actions:
                        execute_cheap_promotions(
                            self._db, item["item_type"],
                            item["item_id"], actions,
                        )
            except Exception:
                pass  # Never fail a search due to promotions

        # Populate working memory with emitted items
        # Build a lookup for knowledge node content
        knowledge_content = {
            node.get("id", ""): node.get("content", "")
            for node in (knowledge_results or [])
        }
        for item in emitted_access:
            content = ""
            if item["item_type"] == "chunk":
                idx = self._id_to_idx.get(item["item_id"])
                if idx is not None:
                    content = self.chunks[idx].assistant_text
            elif item["item_type"] == "knowledge":
                content = knowledge_content.get(item["item_id"], "")
            wm.record(item["item_type"], item["item_id"], content)

        if len(lines) <= 1:
            return ""
        return "\n".join(lines)

    def _access_frequency_boost(
        self, score: float, item_type: str, item_id: str,
    ) -> float:
        """Apply a frequency boost based on persistent access history.

        Uses explicit_count (search + context) as the signal — only hook
        accesses are excluded. Both search results and drill-downs indicate
        deliberate user interest.

        Boost: 1 + min(log2(explicit_count + 1), 0.3) → capped at 1.3x.
        This is intentionally mild — access history should nudge ranking,
        not dominate it. BM25 relevance remains the primary signal.
        """
        if not self._db:
            return score
        stats = self._db.get_access_stats(item_type, item_id)
        if stats is None or stats["explicit_count"] == 0:
            return score
        boost = 1.0 + min(math.log2(stats["explicit_count"] + 1) * 0.15, 0.3)
        return score * boost

    def _group_by_cluster(
        self,
        ranked: list[tuple[int, float]],
    ) -> tuple[dict[str, tuple[dict, list[int], float]], list[tuple[int, float]]]:
        """Partition ranked results into cluster groups and ungrouped chunks.

        Returns:
            (cluster_groups, ungrouped) where cluster_groups maps
            cluster_id → (cluster_info_dict, [chunk_indices], max_score) and
            ungrouped is the list of (idx, score) not in any multi-member group.
        """
        if not self._db or self._db.cluster_count() == 0:
            return {}, ranked

        # Map each result chunk to its primary cluster and track scores.
        # Invariant: Phase 1 clustering produces non-overlapping clusters
        # (each chunk belongs to at most one). We use the first cluster_id
        # returned, with deterministic ORDER BY in clusters_for_chunk().
        cluster_members: dict[str, list[int]] = {}
        cluster_scores: dict[str, float] = {}

        for idx, score in ranked:
            chunk = self.chunks[idx]
            clusters = self._db.clusters_for_chunk(chunk.id)
            if clusters:
                cid = clusters[0]
                cluster_members.setdefault(cid, []).append(idx)
                cluster_scores[cid] = max(cluster_scores.get(cid, 0.0), score)

        # Only group clusters with 2+ result members (singletons stay raw)
        multi_member_cids = [cid for cid, m in cluster_members.items() if len(m) >= 2]
        cluster_groups: dict[str, tuple[dict, list[int], float]] = {}
        grouped_indices: set[int] = set()

        if multi_member_cids:
            for cid in multi_member_cids:
                cluster_info = self._db.get_cluster(cid)
                if cluster_info:
                    cluster_groups[cid] = (
                        cluster_info,
                        cluster_members[cid],
                        cluster_scores[cid],
                    )
                    grouped_indices.update(cluster_members[cid])

        ungrouped = [(idx, s) for idx, s in ranked if idx not in grouped_indices]
        return cluster_groups, ungrouped

    def _format_cluster_block(
        self,
        cluster_id: str,
        cluster_info: dict,
    ) -> str:
        """Format a cluster as a summary block."""
        topic = cluster_info.get("topic", "unknown")
        date_start = (cluster_info.get("date_start") or "")[:10]
        date_end = (cluster_info.get("date_end") or "")[:10]
        chunk_count = cluster_info.get("chunk_count", 0)
        session_ids = cluster_info.get("session_ids", [])

        # Date range display
        if date_start and date_end and date_start != date_end:
            date_display = f"{date_start} \u2013 {date_end}"
        elif date_start:
            date_display = date_start
        else:
            date_display = "unknown"

        header = f"--- [cluster: {topic}] {date_display}, {chunk_count} chunks ({cluster_id}) ---"
        parts = [header]

        # Try to get a stored summary
        summary = None
        if self._db:
            stored = self._db.get_cluster_summary(cluster_id)
            if stored and not stored["stale"]:
                summary = stored["summary"]

        if not summary:
            # Generate concat summary from FULL cluster membership (not just
            # search result hits) so the summary is representative of the
            # whole cluster. Read-only: don't cache during search — summaries
            # are cached during recall_build or recall_enrich.
            summary = self._generate_cluster_summary(cluster_id)

        if summary:
            parts.append(summary)

        # Session references
        if session_ids:
            sess_display = ", ".join(s[:8] for s in session_ids[:4])
            if len(session_ids) > 4:
                sess_display += f" +{len(session_ids) - 4} more"
            parts.append(f"[Sessions: {sess_display}]")

        return "\n".join(parts)

    def _generate_cluster_summary(self, cluster_id: str) -> str:
        """Generate a concat summary from all chunks in a cluster."""
        if not self._db:
            return ""
        chunk_ids = self._db.get_cluster_chunks(cluster_id)
        if not chunk_ids:
            return ""
        # Look up chunks by ID
        member_chunks = []
        for cid in chunk_ids:
            idx = self._id_to_idx.get(cid)
            if idx is not None:
                member_chunks.append(self.chunks[idx])
        if not member_chunks:
            return ""
        from synapt.recall.clustering import generate_concat_summary
        return generate_concat_summary(member_chunks, max_tokens=200)

    def _format_chunk_block(self, idx: int) -> str:
        """Format a single chunk as a display block."""
        chunk = self.chunks[idx]
        raw_ts = chunk.timestamp if chunk.timestamp else "unknown"
        # For ISO 8601 timestamps, show date + HH:MM (drop seconds/tz).
        # For free-text timestamps (e.g. "1:56 pm on 8 May, 2023"), keep as-is.
        if len(raw_ts) > 16 and raw_ts[4:5] == "-" and raw_ts[10:11] == "T":
            raw_ts = raw_ts[:16]
        ts_display = raw_ts.replace("T", " ")
        turn_label = "journal" if chunk.turn_index == -1 else f"turn {chunk.turn_index}"
        header = f"--- [{ts_display} session {chunk.session_id[:8]}] {turn_label} ---"

        parts = [header]
        prev = self._get_preceding_turn(chunk)
        if prev and prev.user_text:
            ctx = prev.user_text[:200]
            if len(prev.user_text) > 200:
                ctx += "..."
            parts.append(f"  (context: User previously asked: {ctx})")
        if chunk.user_text:
            ut = chunk.user_text[:500]
            if len(chunk.user_text) > 500:
                ut += "..."
            parts.append(f"User: {ut}")
        if chunk.assistant_text:
            asst = chunk.assistant_text[:1500]
            if len(chunk.assistant_text) > 1500:
                asst += "..."
            parts.append(f"Assistant: {asst}")
        if chunk.files_touched:
            parts.append(f"[Files: {', '.join(chunk.files_touched[:5])}]")
        if chunk.tool_content:
            tc = chunk.tool_content[:400]
            if len(chunk.tool_content) > 400:
                tc += "..."
            parts.append(f"[Tools: {tc}]")

        return "\n".join(parts)

    # -------------------------------------------------------------------
    # Drill-down: full raw context for a turn
    # -------------------------------------------------------------------

    def read_turn_context(self, chunk_id: str) -> str:
        """Read full raw transcript content for a turn via byte offsets.

        Seeks to the stored byte_offset in the source transcript file and
        reads byte_length bytes, then extracts and formats all user messages,
        assistant text, tool uses with full inputs, and tool results.

        Returns formatted text with no cap — this is the detail view.
        """
        # Find chunk by ID (O(1) via index map)
        idx = self._id_to_idx.get(chunk_id)
        chunk = self.chunks[idx] if idx is not None else None
        if chunk is None and self._db:
            # Try DB lookup
            row = self._db._conn.execute(
                "SELECT transcript_path, byte_offset, byte_length, "
                "session_id, turn_index, timestamp "
                "FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
            if row:
                chunk = TranscriptChunk(
                    id=chunk_id,
                    session_id=row[3],
                    timestamp=row[5] or "",
                    turn_index=row[4],
                    user_text="",
                    assistant_text="",
                    transcript_path=row[0] or "",
                    byte_offset=row[1] if row[1] is not None else -1,
                    byte_length=row[2] if row[2] is not None else 0,
                )
        if chunk is None:
            return f"Chunk {chunk_id} not found."
        if not chunk.transcript_path or chunk.byte_offset < 0:
            return f"No transcript offset recorded for {chunk_id}."
        try:
            return self._read_raw_turn(chunk)
        except Exception as exc:
            return f"Error reading transcript: {exc}"

    @staticmethod
    def _read_raw_turn(chunk: TranscriptChunk) -> str:
        """Read and format raw JSONL entries for a turn."""
        path = Path(chunk.transcript_path)
        if not path.exists():
            return f"Transcript file not found: {path}"

        with open(path, "rb") as f:
            f.seek(chunk.byte_offset)
            raw = f.read(chunk.byte_length).decode("utf-8", errors="replace")

        parts = []
        ts = chunk.timestamp[:19] if chunk.timestamp else "unknown"
        parts.append(
            f"=== Full context: session {chunk.session_id[:8]} "
            f"turn {chunk.turn_index} [{ts}] ==="
        )

        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = entry.get("type", "")
            if etype in SKIP_TYPES:
                continue
            content = entry.get("message", {}).get("content")
            if etype == "user" and isinstance(content, str) and content.strip():
                parts.append(f"\n[User]\n{content.strip()}")
            elif etype == "user" and isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        parts.append(f"\n[User]\n{block.get('text', '')}")
                    elif block.get("type") == "tool_result":
                        tu_id = block.get("tool_use_id", "?")
                        result = block.get("content", "")
                        if isinstance(result, list):
                            result = "\n".join(
                                s.get("text", "")
                                for s in result
                                if isinstance(s, dict) and s.get("type") == "text"
                            )
                        parts.append(f"\n[Tool Result for {tu_id}]\n{result}")
            elif etype == "assistant" and isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        parts.append(f"\n[Assistant]\n{block.get('text', '')}")
                    elif block.get("type") == "tool_use":
                        name = block.get("name", "?")
                        inp = json.dumps(block.get("input", {}), indent=2)
                        parts.append(f"\n[Tool Use: {name}]\n{inp}")

        return "\n".join(parts)

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        """Save index to disk for fast reload."""
        directory.mkdir(parents=True, exist_ok=True)

        # Build manifest data (shared by both paths)
        sessions_info = {}
        for sid, session_chunks in self.sessions.items():
            timestamps = [c.timestamp for c in session_chunks if c.timestamp]
            sessions_info[sid] = {
                "chunk_count": len(session_chunks),
                "min_ts": min(timestamps) if timestamps else "",
                "max_ts": max(timestamps) if timestamps else "",
            }

        manifest = {
            "chunk_count": len(self.chunks),
            "session_count": len(self.sessions),
            "sessions": sessions_info,
            "build_timestamp": datetime.now().isoformat(),
        }

        if self._db is not None:
            # SQLite path
            self._db.save_chunks(self.chunks)
            self._db.save_manifest(manifest)
            self._refresh_rowid_map()
            return

        # Legacy file-based path
        chunks_path = directory / "chunks.jsonl"
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")

        h = hashlib.sha256()
        with open(chunks_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)

        manifest["chunks_hash"] = h.hexdigest()[:16]
        atomic_json_write(manifest, directory / "manifest.json")

    @classmethod
    def load(cls, directory: Path, use_embeddings: bool = False) -> TranscriptIndex:
        """Load a saved index from disk.

        Prefers recall.db (SQLite) if present.  Falls back to chunks.jsonl
        with automatic migration to SQLite.  Malformed JSON lines are
        skipped during migration.
        """
        db_path = directory / "recall.db"
        chunks_path = directory / "chunks.jsonl"

        # Prefer SQLite
        if db_path.exists():
            db = None
            try:
                db = RecallDB(db_path)
                chunks = db.load_chunks()
                return cls(chunks, use_embeddings=use_embeddings, cache_dir=directory, db=db)
            except (sqlite3.DatabaseError, OSError) as exc:
                logger.warning("Corrupt recall.db, rebuilding: %s", exc)
                if db is not None:
                    with contextlib.suppress(Exception):
                        db.close()
                with contextlib.suppress(OSError):
                    db_path.unlink()
                # Fall through to JSONL path or empty index

        # Legacy JSONL — load and auto-migrate to SQLite
        if chunks_path.exists():
            chunks = []
            with open(chunks_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunks.append(TranscriptChunk.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue  # Skip malformed lines

            # Migrate to SQLite
            db = RecallDB(db_path)
            index = cls(chunks, use_embeddings=use_embeddings, cache_dir=directory, db=db)
            index.save(directory)

            # Migrate embeddings from legacy JSON if present
            emb_path = directory / "transcript_embeddings.json"
            if emb_path.exists() and index._idx_to_rowid:
                try:
                    with open(emb_path, encoding="utf-8") as f:
                        cached = json.load(f)
                    old_embs = cached.get("embeddings", [])
                    old_hash = cached.get("hash")
                    if old_embs and old_hash == index._content_hash():
                        emb_mapping: dict[int, list[float]] = {}
                        for i, emb in enumerate(old_embs):
                            rowid = index._idx_to_rowid.get(i)
                            if rowid is not None:
                                emb_mapping[rowid] = emb
                        if emb_mapping:
                            db.save_embeddings(emb_mapping)
                            db.set_metadata("embedding_hash", index._content_hash())
                except (json.JSONDecodeError, KeyError, OSError):
                    pass

            # Clean up legacy files
            for old_file in [chunks_path, emb_path, directory / "manifest.json"]:
                with contextlib.suppress(OSError):
                    old_file.unlink()

            return index

        return cls([])

    # -------------------------------------------------------------------
    # File lookup
    # -------------------------------------------------------------------

    def lookup_files(
        self,
        pattern: str,
        max_chunks: int = 10,
        max_tokens: int = 500,
        after: str | None = None,
        before: str | None = None,
    ) -> str:
        """Find chunks that touched files matching the given pattern.

        Supports partial path matching: "repair.py" matches
        "src/graph/repair.py". Also matches basename-only.

        Results sorted newest-first (no BM25 scoring).
        """
        if not self.chunks or not pattern:
            return ""

        date_filter = self._filter_by_date(after, before)
        pattern_lower = pattern.lower()

        hits: list[tuple[int, float]] = []
        for i, chunk in enumerate(self.chunks):
            if date_filter is not None and i not in date_filter:
                continue
            if not chunk.files_touched:
                continue
            for fp in chunk.files_touched:
                fp_lower = fp.lower()
                if pattern_lower in fp_lower or fp_lower.endswith("/" + pattern_lower):
                    hits.append((i, 1.0))
                    break

        if not hits:
            return ""

        # Sort newest-first by timestamp
        hits.sort(key=lambda x: self.chunks[x[0]].timestamp, reverse=True)
        return self._format_results(hits[:max_chunks], max_tokens, query=pattern)

    # -------------------------------------------------------------------
    # Session browsing
    # -------------------------------------------------------------------

    def list_sessions(
        self,
        max_sessions: int = 20,
        after: str | None = None,
        before: str | None = None,
    ) -> list[dict]:
        """Return recent sessions with summary info, newest-first."""
        results = []
        for session_id in self._session_order:
            chunks = self.sessions[session_id]

            timestamps = [c.timestamp for c in chunks if c.timestamp]
            if not timestamps:
                continue

            latest_ts = max(timestamps)
            earliest_ts = min(timestamps)

            if after and latest_ts < after:
                continue
            if before and earliest_ts >= before:
                continue

            # First user message as summary (skip journal chunks)
            transcript_chunks = [c for c in chunks if c.turn_index >= 0]
            sorted_chunks = sorted(
                transcript_chunks or chunks,
                key=lambda c: c.turn_index,
            )
            first_msg = ""
            for c in sorted_chunks:
                if c.user_text:
                    first_msg = c.user_text[:120]
                    if len(c.user_text) > 120:
                        first_msg += "..."
                    break

            all_files = set()
            for c in chunks:
                all_files.update(c.files_touched)

            results.append({
                "session_id": session_id,
                "date": earliest_ts[:10],
                "turn_count": len(transcript_chunks),
                "first_message": first_msg,
                "files_count": len(all_files),
            })

            if len(results) >= max_sessions:
                break

        return results

    # -------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary statistics about the index."""
        if not self.chunks:
            return {"chunk_count": 0, "session_count": 0}

        timestamps = [c.timestamp for c in self.chunks if c.timestamp]
        return {
            "chunk_count": len(self.chunks),
            "session_count": len(self.sessions),
            "date_range": {
                "earliest": min(timestamps) if timestamps else "",
                "latest": max(timestamps) if timestamps else "",
            },
            "avg_chunks_per_session": len(self.chunks) / max(len(self.sessions), 1),
            "total_tools_used": len(set(
                t for c in self.chunks for t in c.tools_used
            )),
            "total_files_touched": len(set(
                f for c in self.chunks for f in c.files_touched
            )),
            "embeddings_active": (
                self._embeddings is not None
                or (self._db is not None and self._db.has_embeddings())
            ),
            "embedding_provider": (
                type(self._embed_provider).__name__
                if self._embed_provider else None
            ),
            "storage_backend": "sqlite" if self._db else "memory",
            "knowledge_count": (
                self._db.knowledge_count() if self._db else 0
            ),
        }


# ---------------------------------------------------------------------------
# Project path utilities
# ---------------------------------------------------------------------------


def _git_main_worktree_root(path: Path) -> Path | None:
    """If *path* is inside a git worktree, return the main worktree's root.

    Uses ``git rev-parse --git-common-dir`` which returns the shared ``.git/``
    directory for all worktrees.  The main worktree root is the parent of that
    directory.  Returns *None* if *path* is not in a git repo or is already the
    main worktree (i.e. ``.git`` is a directory, not a gitdir pointer file).
    """
    git_path = path / ".git"
    # Fast path: if .git is a real directory, this is the main worktree already
    if git_path.is_dir():
        return None
    # If .git is a file, this is a linked worktree
    if not git_path.is_file():
        return None
    try:
        common_dir = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, cwd=path, timeout=5,
        )
        if common_dir.returncode != 0:
            return None
        # --git-common-dir returns e.g. /Users/layne/Development/rd/.git
        # The main worktree root is its parent
        return Path(common_dir.stdout.strip()).parent
    except (OSError, subprocess.TimeoutExpired):
        return None


_gripspace_cache: dict[str, tuple[Path | None, float]] = {}
_GRIPSPACE_CACHE_TTL = 60.0  # seconds


def _find_gripspace_root(path: Path) -> Path | None:
    """Walk up from *path* to find a ``.gitgrip/`` directory (GitGrip gripspace root).

    Returns the gripspace root path, or *None* if not inside a gripspace.
    Analogous to ``_git_main_worktree_root`` but for multi-repo gripspaces.

    Stops at ``$HOME`` to avoid matching stray ``.gitgrip/`` directories
    above the user's project hierarchy.  Results are cached per resolved path
    with a 60-second TTL so long-running processes (MCP server) pick up
    gripspace changes without restart.
    """
    import time

    current = path.resolve()
    cache_key = str(current)
    cached = _gripspace_cache.get(cache_key)
    if cached is not None:
        value, ts = cached
        if time.monotonic() - ts < _GRIPSPACE_CACHE_TTL:
            return value

    home = Path.home().resolve()
    while current != current.parent:
        if (current / ".gitgrip").is_dir():
            _gripspace_cache[cache_key] = (current, time.monotonic())
            return current
        # Don't walk above $HOME
        if current == home:
            break
        current = current.parent
    _gripspace_cache[cache_key] = (None, time.monotonic())
    return None


def project_slug(project_dir: Path | None = None) -> str:
    """Convert a project path to Claude Code's directory slug format.

    Claude Code stores per-project data at ~/.claude/projects/<slug>/
    where the slug is the absolute path with ``/`` replaced by ``-``.

    Examples:
        /Users/layne/Development/synapse → -Users-layne-Development-synapse
    """
    p = (project_dir or Path.cwd()).resolve()
    return str(p).replace("/", "-")


def _worktree_name(project_dir: Path | None = None) -> str:
    """Return the current worktree's name (its directory basename).

    For the main worktree at ``/Users/layne/Development/rd``, returns ``rd``.
    For a linked worktree at ``/Users/layne/Development/poe``, returns ``poe``.
    """
    return (project_dir or Path.cwd()).resolve().name


def project_data_dir(project_dir: Path | None = None) -> Path:
    """Return the root synapt recall data directory.

    ALL recall data lives under ``<root>/.synapt/recall/``:
    shared data (index, knowledge) at the root, and per-worktree data
    (transcripts, journal) under ``worktrees/<name>/``.

    Root resolution priority:
      1. Git worktree → main worktree root
      2. GitGrip gripspace → gripspace root (all constituent repos share
         one index; each sub-repo gets its own ``worktrees/<name>/`` subdir)
      3. CWD as fallback

    Auto-migrates from two legacy locations:
      1. ``.synapse/recall/``  → ``.synapt/recall/``
      2. ``.synapse-recall/``  → ``.synapt/recall/``
    """
    root = (project_dir or Path.cwd()).resolve()

    # Priority 1: git worktree → resolve to main worktree root
    main_root = _git_main_worktree_root(root)
    if main_root is not None:
        root = main_root

    # Priority 2: GitGrip gripspace → resolve to gripspace root
    # If CWD (or resolved root) is inside a gripspace, prefer the gripspace
    # root so all constituent repos share one recall index.
    grip_root = _find_gripspace_root(root)
    if grip_root is not None:
        root = grip_root

    new_dir = root / ".synapt" / "recall"

    if not new_dir.exists():
        # Tier 1: .synapse/recall/ (intermediate rename era)
        mid_dir = root / ".synapse" / "recall"
        if mid_dir.is_dir():
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            mid_dir.rename(new_dir)
            # Clean up empty .synapse/ parent if nothing else is inside
            synapse_parent = root / ".synapse"
            if synapse_parent.is_dir() and not any(synapse_parent.iterdir()):
                synapse_parent.rmdir()
        else:
            # Tier 2: .synapse-recall/ (legacy flat location)
            old_dir = root / ".synapse-recall"
            if old_dir.is_dir():
                new_dir.parent.mkdir(parents=True, exist_ok=True)
                old_dir.rename(new_dir)

    return new_dir


def project_worktree_dir(project_dir: Path | None = None) -> Path:
    """Return the per-worktree data directory inside the shared ``.synapt/``.

    Lives at ``<main>/.synapt/recall/worktrees/<worktree-name>/``.
    Each worktree gets its own subdirectory for transcripts and journal.

    **Auto-migrates** from the legacy flat layout on first access:
    moves ``transcripts/`` and ``journal.jsonl`` from the data root into
    ``worktrees/<main-name>/`` for the main worktree.
    """
    wt_name = _worktree_name(project_dir)
    data_dir = project_data_dir(project_dir)
    wt_dir = data_dir / "worktrees" / wt_name

    # Auto-migrate legacy flat layout → worktrees/<name>/
    # Only applies to the main worktree (linked worktrees never had flat data).
    # Race-safe: if two processes migrate concurrently, the loser's rename()
    # raises FileNotFoundError (source already moved) — we catch and continue.
    if not wt_dir.exists():
        legacy_transcripts = data_dir / "transcripts"
        legacy_journal = data_dir / "journal.jsonl"
        has_legacy = legacy_transcripts.is_dir() or legacy_journal.exists()

        if has_legacy:
            wt_dir.mkdir(parents=True, exist_ok=True)
            try:
                if legacy_transcripts.is_dir() and not legacy_transcripts.is_symlink():
                    legacy_transcripts.rename(wt_dir / "transcripts")
                    legacy_transcripts.symlink_to(wt_dir / "transcripts")
            except (FileNotFoundError, FileExistsError, OSError):
                pass  # Another process already migrated or symlinked
            try:
                if legacy_journal.exists() and not legacy_journal.is_symlink():
                    legacy_journal.rename(wt_dir / "journal.jsonl")
            except (FileNotFoundError, FileExistsError, OSError):
                pass  # Another process already migrated

    return wt_dir


def project_index_dir(project_dir: Path | None = None) -> Path:
    """Return the shared index directory.

    Index, clusters, and knowledge are shared across all worktrees.
    Lives at ``<main>/.synapt/recall/index/``.
    """
    return project_data_dir(project_dir) / "index"


def project_archive_dir(project_dir: Path | None = None) -> Path:
    """Return the per-worktree transcript archive directory.

    Lives at ``<main>/.synapt/recall/worktrees/<name>/transcripts/``.
    """
    return project_worktree_dir(project_dir) / "transcripts"


def project_transcript_dir(project_dir: Path | None = None) -> Path | None:
    """Return the Claude Code transcript directory for a project.

    Returns None if the directory doesn't exist or has no transcripts.
    """
    slug = project_slug(project_dir)
    d = Path.home() / ".claude" / "projects" / slug
    if d.is_dir() and any(d.glob("*.jsonl")):
        return d
    return None


def project_transcript_dirs(project_dir: Path | None = None) -> list[Path]:
    """Return all Claude Code transcript directories for a project.

    When working in a git worktree, Claude Code creates transcripts under
    a slug derived from the *worktree* path.  This function returns transcript
    directories for both the main worktree and the current worktree (if
    different), so that ``recall build`` archives sessions from all worktrees.

    In a gripspace, also discovers transcripts from all *direct child*
    repos (directories with ``.git``).  Nested repos are not discovered.
    """
    actual_dir = (project_dir or Path.cwd()).resolve()
    dirs: list[Path] = []
    seen_slugs: set[str] = set()

    # Always include the actual CWD's transcript dir
    slug = project_slug(actual_dir)
    seen_slugs.add(slug)
    d = Path.home() / ".claude" / "projects" / slug
    if d.is_dir() and any(d.glob("*.jsonl")):
        dirs.append(d)

    # If in a linked worktree, also include the main worktree's transcript dir
    main_root = _git_main_worktree_root(actual_dir)
    if main_root is not None:
        main_slug = project_slug(main_root)
        if main_slug not in seen_slugs:
            seen_slugs.add(main_slug)
            main_d = Path.home() / ".claude" / "projects" / main_slug
            if main_d.is_dir() and any(main_d.glob("*.jsonl")):
                dirs.append(main_d)

    # If in a gripspace, discover transcripts from all constituent repos
    grip_root = _find_gripspace_root(actual_dir)
    if grip_root is not None:
        try:
            children = sorted(grip_root.iterdir())
        except OSError:
            children = []
        for child in children:
            if child.is_dir() and (child / ".git").exists():
                child_slug = project_slug(child)
                if child_slug not in seen_slugs:
                    seen_slugs.add(child_slug)
                    child_d = Path.home() / ".claude" / "projects" / child_slug
                    if child_d.is_dir() and any(child_d.glob("*.jsonl")):
                        dirs.append(child_d)

    return dirs


def all_worktree_archive_dirs(project_dir: Path | None = None) -> list[Path]:
    """Return archive directories from all worktrees.

    Since all worktree data lives under ``<main>/.synapt/recall/worktrees/``,
    this simply globs for ``worktrees/*/transcripts/`` directories that
    contain archived transcripts.  No ``git worktree list`` needed.
    """
    data_dir = project_data_dir(project_dir)
    worktrees_root = data_dir / "worktrees"
    if not worktrees_root.is_dir():
        return []

    dirs: list[Path] = []
    for wt_dir in sorted(worktrees_root.iterdir()):
        archive = wt_dir / "transcripts"
        if archive.is_dir() and any(archive.glob("*.jsonl")):
            dirs.append(archive)
    return dirs


# ---------------------------------------------------------------------------
# Journal → chunk converter
# ---------------------------------------------------------------------------


def _journal_entry_to_chunk(entry, scrub_text) -> TranscriptChunk:
    """Convert a single JournalEntry to a TranscriptChunk."""
    user_text = f"Session focus: {entry.focus}" if entry.focus else ""

    parts: list[str] = []
    if entry.done:
        parts.append("Done: " + " - ".join(entry.done))
    if entry.decisions:
        parts.append("Decisions: " + " - ".join(entry.decisions))
    if entry.next_steps:
        parts.append("Next steps: " + " - ".join(entry.next_steps))
    if entry.branch:
        parts.append(f"Branch: {entry.branch}")
    assistant_text = "\n".join(parts)

    # Scrub secrets from journal content (matches transcript parser behavior)
    try:
        user_text = scrub_text(user_text)
        assistant_text = scrub_text(assistant_text)
    except Exception:
        pass  # Use unscrubbed text rather than losing the entry

    # Stable content-based ID: hash of session_id + timestamp
    session_short = entry.session_id[:8] if entry.session_id else entry.timestamp[:10]
    ts_hash = hashlib.sha256(
        f"{entry.session_id}:{entry.timestamp}".encode()
    ).hexdigest()[:8]
    chunk_id = f"{session_short}:journal:{ts_hash}"
    session_id = entry.session_id or f"journal-{entry.timestamp}"

    return TranscriptChunk(
        id=chunk_id,
        session_id=session_id,
        timestamp=entry.timestamp,
        turn_index=-1,
        user_text=user_text,
        assistant_text=assistant_text,
        files_touched=list(entry.files_modified),
    )


def parse_journal_entries(journal_path: Path) -> list[TranscriptChunk]:
    """Convert journal entries into searchable TranscriptChunks.

    Each JournalEntry becomes one chunk with turn_index=-1 as a sentinel
    to distinguish journal chunks from regular transcript turns.

    When multiple entries share a session_id, non-auto entries (manual or
    enriched) take priority over auto-stubs. This supports the append-only
    pattern where enriched entries are appended rather than replacing stubs.

    Args:
        journal_path: Path to journal.jsonl file.

    Returns:
        List of TranscriptChunks, one per selected journal entry.
    """
    from synapt.recall.journal import JournalEntry
    from synapt.recall.scrub import scrub_text

    if not journal_path.exists():
        return []

    # Phase 1: Read all entries, grouped by session_id
    entries_by_session: dict[str, list] = {}
    with open(journal_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = JournalEntry.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if not entry.has_content():
                continue
            key = entry.session_id or entry.timestamp
            entries_by_session.setdefault(key, []).append(entry)

    # Phase 2: For each session, drop auto-stubs if a better entry exists
    selected = []
    for _sid, entries in entries_by_session.items():
        non_auto = [e for e in entries if not e.auto]
        if non_auto:
            selected.extend(non_auto)
        else:
            # Only auto-stubs; keep the most recent one
            selected.append(max(entries, key=lambda e: e.timestamp))

    # Phase 3: Convert selected entries to TranscriptChunks
    return [_journal_entry_to_chunk(entry, scrub_text) for entry in selected]


# ---------------------------------------------------------------------------
# Multi-file index builder
# ---------------------------------------------------------------------------

def build_index(
    source_dir: Path,
    use_embeddings: bool = False,
    cache_dir: Path | None = None,
    incremental_manifest: dict | None = None,
    db: RecallDB | None = None,
) -> TranscriptIndex:
    """Build a TranscriptIndex from a directory of .jsonl transcript files.

    Args:
        source_dir: Directory containing .jsonl transcript files.
        use_embeddings: Whether to compute embeddings.
        cache_dir: Directory for caching embeddings.
        incremental_manifest: If provided, skip files already indexed
                             (by checking source_files in manifest).

    Returns:
        TranscriptIndex over all parsed chunks.
    """
    jsonl_files = sorted(source_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"[synapt] No .jsonl files found in {source_dir}")
        return TranscriptIndex([])

    # Filter for incremental builds — skip files whose mtime AND size match
    already_indexed: dict[str, tuple[float, int]] = {}
    if incremental_manifest:
        for src in incremental_manifest.get("source_files", []):
            already_indexed[src["name"]] = (src.get("mtime", 0), src.get("size", 0))

    all_chunks: list[TranscriptChunk] = []
    seen_uuids: set[str] = set()
    skipped = 0

    for filepath in jsonl_files:
        if filepath.name in already_indexed:
            stored_mtime, stored_size = already_indexed[filepath.name]
            stat = filepath.stat()
            if stat.st_mtime == stored_mtime and stat.st_size == stored_size:
                skipped += 1
                continue
        try:
            chunks = parse_transcript(filepath, seen_uuids=seen_uuids)
            all_chunks.extend(chunks)
            print(f"  {filepath.name}: {len(chunks)} turns")
        except Exception as e:
            print(f"  {filepath.name}: ERROR - {e}")

    if skipped:
        print(f"[synapt] Skipped {skipped} already-indexed files")
    print(f"[synapt] Total: {len(all_chunks)} chunks from {len(jsonl_files) - skipped} files")

    return TranscriptIndex(all_chunks, use_embeddings=use_embeddings, cache_dir=cache_dir, db=db)
