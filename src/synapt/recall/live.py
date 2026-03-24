"""Live transcript search — search the current session's transcript on-the-fly.

The recall index only covers archived sessions.  This module extends
``recall_search`` with results from the *live* transcript that Claude Code is
actively writing, making the current session instantly searchable without
waiting for ``SessionEnd`` to archive and index it.

Key design decisions:
- **Parse on demand, not on startup**: the transcript is parsed when the first
  search arrives, then cached by (path, file_size).  File size is the cache key
  because the JSONL file is append-only — a growing size always means new turns.
- **Simple TF + coverage scoring**: BM25 IDF statistics require a corpus; the
  live session is too small for meaningful IDF.  A sublinear TF score with a
  multi-term coverage bonus works well for <100 chunks.
- **min_score gate**: live results are only prepended when they are genuinely
  relevant.  A weak single-keyword mention in a passing sentence should not
  push the current session to the top of every search.
- **Dedup via index.sessions**: if the session is already in the archived index
  (e.g. after a PreCompact rebuild) we skip live search entirely to avoid
  showing duplicate results.
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synapt.recall.core import TranscriptChunk, TranscriptIndex

import logging

from synapt.recall.bm25 import _tokenize
from synapt.recall.journal import extract_session_id, latest_transcript_path

logger = logging.getLogger("synapt.recall.live")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class _LiveCache:
    """Single-entry cache keyed by (path, file_size)."""
    path: str = ""
    file_size: int = -1
    session_id: str = ""
    chunks: list[TranscriptChunk] = field(default_factory=list)


# Process-wide singleton: holds the single most-recently-accessed transcript.
# Correct for single-project MCP server use (the common case).  Multi-project
# callers will always miss the cache (different path each call), which is safe
# but inefficient.  The path comparison in _get_live_chunks is the key.
_cache = _LiveCache()
_cache_lock = threading.Lock()


def _get_live_chunks(transcript_path: str) -> tuple[str, list[TranscriptChunk]]:
    """Return (session_id, chunks) for the live transcript, using the cache.

    Re-parses only when the file has grown since the last call.
    Thread-safe: the lock makes the cache-miss check and update atomic so
    concurrent callers block rather than triggering redundant parses.
    """
    # Declared global so the cache update (a reference replacement, not an
    # attribute mutation) is visible to all callers via the module name.
    global _cache
    # parse_transcript is imported lazily to keep module load order
    # unambiguous.  live.py → journal.py → core.py is a one-way chain;
    # importing core at the top of live.py would work at runtime, but the
    # TYPE_CHECKING guard already covers type annotations, and the deferred
    # import documents that we only need parse_transcript at call time.
    from synapt.recall.core import parse_transcript
    from synapt.recall.codex import is_codex_transcript, parse_codex_transcript

    try:
        size = Path(transcript_path).stat().st_size
    except OSError:
        return "", []

    with _cache_lock:
        if _cache.path == transcript_path and _cache.file_size == size:
            return _cache.session_id, _cache.chunks

        # Cache miss — re-parse under the lock.
        # Guard against the file disappearing between stat() and open()
        # (e.g., transcript archived and deleted mid-search), and against
        # any parse error (UnicodeDecodeError, MemoryError, etc.) from a
        # malformed transcript.  search_live_transcript also wraps this call
        # in a broad except, but this inner guard keeps _get_live_chunks
        # self-contained and safe for any future caller that lacks one.
        try:
            session_id = extract_session_id(transcript_path)
            path_obj = Path(transcript_path)
            if is_codex_transcript(path_obj):
                chunks = parse_codex_transcript(path_obj)
            else:
                chunks = parse_transcript(path_obj)
        except Exception:
            return "", []

        # Replace the singleton in one reference assignment so the cache is
        # never partially-updated.  Four sequential attribute writes risk an
        # inconsistent state (e.g. path matches but chunks is stale) if an
        # async exception (MemoryError) fires between them.  A single name
        # binding is effectively atomic under CPython's GIL.
        _cache = _LiveCache(
            path=transcript_path,
            file_size=size,
            session_id=session_id,
            chunks=chunks,
        )
        return session_id, chunks


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_chunks(
    chunks: list[TranscriptChunk],
    query_tokens: list[str],
) -> list[tuple[int, float]]:
    """Score live chunks against query tokens.

    Returns (chunk_index, score) pairs sorted by score descending, excluding
    zero-score chunks.  Uses sublinear TF with a multi-term coverage bonus:

        score = tf_sum * (0.5 + 0.5 * coverage)

    where tf_sum = sum(1 + log(count)) for each matched query token and
    coverage = matched_terms / total_query_terms.

    Duplicate query tokens are deduplicated — "swift swift" is treated
    identically to "swift" (consistent with BM25 term-set semantics).
    """
    if not query_tokens or not chunks:
        return []

    scored: list[tuple[int, float]] = []
    query_set = set(query_tokens)

    for i, chunk in enumerate(chunks):
        if not chunk.text:
            continue

        # Build a frequency map once per chunk (O(n)) rather than calling
        # list.count() per query term (O(n * Q)).
        doc_tokens = _tokenize(chunk.text)
        if not doc_tokens:
            continue

        tf_map = Counter(doc_tokens)
        tf_sum = 0.0
        matched_terms = 0
        for qt in query_set:
            count = tf_map.get(qt, 0)
            if count:
                tf_sum += 1.0 + math.log(count)
                matched_terms += 1

        if matched_terms == 0:
            continue

        coverage = matched_terms / len(query_set)
        score = tf_sum * (0.5 + 0.5 * coverage)
        scored.append((i, score))

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_live_results(
    chunks: list[TranscriptChunk],
    scored: list[tuple[int, float]],
    max_chunks: int,
    max_tokens: int,
) -> str:
    """Format top-K live chunks as a 'Current session context:' block.

    Token budget is approximate (chars // 4).  The first result chunk is
    always included regardless of budget, even when max_tokens=0 — the
    break guard (``len(lines) > 1``) is False on the first iteration so the
    chunk is emitted unconditionally.  Callers that need to suppress all
    output must pass an empty ``scored`` list or avoid calling this function;
    passing max_tokens=0 still emits one chunk.  For predictable multi-chunk
    output pass max_tokens ≥ 100.  (The ``recall_search`` call site guards
    ``live_budget > 0`` before calling ``search_live_transcript``.)


    Per-field truncation limits match core.py _format_results:
    user_text at 500 chars, assistant_text at 1500 chars,
    tool_content at 400 chars, files_touched at 5 items.
    """
    if not scored:
        return ""

    lines = ["Current session context:"]
    token_budget = max_tokens

    for idx, _score in scored[:max_chunks]:
        chunk = chunks[idx]
        ts = chunk.timestamp[:16].replace("T", " ") if chunk.timestamp else "now"
        header = f"--- [current session, turn {chunk.turn_index}] {ts} ---"

        parts = [header]
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

        block = "\n".join(parts)
        block_tokens = len(block) // 4
        # First iteration: len(lines) == 1 (only the header) so len(lines) > 1
        # is False — the first chunk is always emitted regardless of budget.
        # Subsequent iterations: len(lines) > 1, so the budget gate applies.
        if token_budget - block_tokens < 0 and len(lines) > 1:
            break
        lines.append(block)
        token_budget -= block_tokens

    return "\n".join(lines) if len(lines) > 1 else ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def search_live_transcript(
    query: str,
    index: TranscriptIndex | None = None,
    max_chunks: int = 3,
    max_tokens: int = 500,
    min_score: float = 1.0,
) -> str:
    """Search the current session's live transcript.

    Returns a formatted 'Current session context:' block, or an empty string
    when:
    - no live transcript exists for this project
    - the session is already fully indexed (avoids duplicate results)
    - no chunks score above min_score (avoids weak keyword coincidences
      dominating the result when the current session barely touches the topic)

    Args:
        query: Natural language query or keywords.
        index: The loaded TranscriptIndex (duck-typed via ``getattr`` at
               runtime — the type is only imported under TYPE_CHECKING to
               avoid circular imports).  Used to check whether the current
               session has already been archived and indexed (e.g. after a
               PreCompact rebuild).  Pass ``None`` to skip the dedup check.
        max_chunks: Maximum number of live chunks to include.
        max_tokens: Approximate token budget for the returned block.  The
                    first result chunk is always included regardless of budget.
        min_score: Minimum TF-coverage score for a chunk to qualify.
                   Defaults to 1.0, which requires either all query terms to
                   match or a single term with count ≥ 2.  Set to 0 to
                   disable.
    """
    transcript_path = latest_transcript_path()
    if not transcript_path:
        return ""

    # session_id is read from the first 'progress' entry in the live file.
    # Known TOCTOU limitation: if latest_transcript_path() races and returns a
    # recently-archived file, session_id may already be in index.sessions,
    # causing live search to return "" for real current-session content.
    # In practice the window is very small — Claude Code writes a progress
    # entry before any user/assistant turns.
    try:
        session_id, chunks = _get_live_chunks(transcript_path)
    except Exception:
        # Live search must never surface as an MCP tool error — degrade gracefully.
        # WARNING (not DEBUG) so silent failures are visible in server logs without
        # being exposed to the Claude UI as a tool error.
        logger.warning("Live transcript parse failed — live search disabled for this query",
                        exc_info=True)
        return ""
    if not chunks:
        return ""

    # Duck-typed index check: access .sessions via getattr so that future
    # renames or mock objects without spec= don't raise AttributeError.
    sessions = getattr(index, "sessions", None) if index is not None else None
    if sessions is not None and session_id and session_id in sessions:
        logger.debug("Live search skipped — session %s already indexed", session_id[:8])
        return ""

    query_tokens = _tokenize(query)
    scored = _score_chunks(chunks, query_tokens)

    # Apply min_score gate: suppress weakly-relevant live context so it does
    # not create spurious recency bias when the current session only mentions
    # a query keyword in passing.
    if min_score > 0:
        scored = [(i, s) for i, s in scored if s >= min_score]

    return _format_live_results(chunks, scored, max_chunks, max_tokens)
