"""synapt.recall MCP server — expose transcript search as tools for Claude Code.

Provides fourteen tools via the Model Context Protocol:
  - recall_search: Search past session transcripts by keyword/topic
  - recall_quick: Fast, low-cost knowledge-only search for speculative checks
  - recall_context: Drill down into a search result for full raw content
  - recall_files: Find sessions that touched a specific file
  - recall_sessions: List recent sessions with summaries
  - recall_timeline: View chronological timeline of work arcs
  - recall_build: Build or rebuild the transcript index
  - recall_setup: Initialize synapt recall for the current project
  - recall_stats: Get index statistics
  - recall_journal: Read or write session journal entries
  - recall_remind: Manage cross-session reminders
  - recall_enrich: Enrich chunks with LLM-generated summaries
  - recall_consolidate: Extract durable knowledge from journal entries
  - recall_contradict: Manage pending knowledge contradictions

Can run standalone (synapt-server) or be composed into the
unified synapt server via register_tools(mcp).
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from synapt.recall.config import load_config
from synapt.recall.core import (
    TranscriptIndex,
    format_size,
    project_index_dir,
)
from synapt.recall._llm_util import truncate_at_word as _tw


def _cap_tokens(requested: int) -> int:
    """Apply the user-configured max_tokens cap."""
    limit = load_config().get_max_tokens()
    return min(requested, limit)

# ---------------------------------------------------------------------------
# MCP instructions — shared with the unified server (synapt.server)
# ---------------------------------------------------------------------------

MCP_INSTRUCTIONS = (
    "You have access to synapt recall — persistent memory across sessions. "
    "Search BEFORE you act, not after.\n"
    "\n"
    "WHEN TO SEARCH (do this automatically, without being asked):\n"
    "- Before editing a file you haven't seen this session -> recall_files\n"
    "- Before making a design decision -> recall_search for prior discussion\n"
    "- When debugging an error -> recall_search for past fixes\n"
    "- When user references past work -> recall_search immediately\n"
    "- Starting a session -> recall_journal to read recent entries\n"
    "- When unsure if something was discussed before -> recall_quick\n"
    "- Before proposing a new approach -> recall_quick to check for prior attempts\n"
    "\n"
    "WHICH TOOL:\n"
    "- recall_quick: Fast, cheap knowledge check. Use speculatively when unsure.\n"
    "- recall_search: Full search with transcript chunks. Use when you need detail.\n"
    "- recall_files: Find sessions that touched a specific file path.\n"
    "- recall_journal: Read/write session notes. Check at session start.\n"
    "- recall_remind: Set/check cross-session reminders.\n"
    "\n"
    "DO NOT search for: general programming questions, syntax help, "
    "API docs, or anything not specific to this project's history.\n"
    "\n"
    "IMPORTANT: When in doubt, search. A quick recall_quick check costs ~500 tokens "
    "and takes <100ms. Missing relevant past context costs far more in wasted work "
    "and repeated mistakes. Err on the side of searching too much, not too little."
)

# ---------------------------------------------------------------------------
# Cached index singleton — avoids reloading on every tool call.
# Invalidated when recall.db mtime changes (e.g., after a rebuild).
# ---------------------------------------------------------------------------

_cached_index: TranscriptIndex | None = None
_cached_mtime: float = 0.0
_cached_dir: Path | None = None


def _get_index() -> TranscriptIndex | None:
    """Load per-project index with caching. Reloads if recall.db was modified."""
    global _cached_index, _cached_mtime, _cached_dir
    index_dir = project_index_dir()

    # Prefer recall.db, fall back to legacy chunks.jsonl
    db_path = index_dir / "recall.db"
    check_path = db_path if db_path.exists() else index_dir / "chunks.jsonl"
    if not check_path.exists():
        return None

    try:
        mtime = check_path.stat().st_mtime
        if _cached_index is None or mtime != _cached_mtime or index_dir != _cached_dir:
            # Close old DB connection before replacing cached index
            if _cached_index is not None and getattr(_cached_index, '_db', None) is not None:
                with contextlib.suppress(Exception):
                    _cached_index._db.close()
            import time as _time
            _load_t0 = _time.monotonic()
            logging.getLogger("synapt.recall").info("Loading index from %s ...", index_dir)
            _cached_index = TranscriptIndex.load(index_dir, use_embeddings=True)
            logging.getLogger("synapt.recall").info(
                "Index loaded: %d chunks in %.1fs",
                len(_cached_index.chunks), _time.monotonic() - _load_t0,
            )
            _cached_mtime = mtime
            _cached_dir = index_dir
            # Set current session ID for access tracking (distinct_sessions)
            try:
                from synapt.recall.journal import (
                    extract_session_id, latest_transcript_path,
                )
                live_path = latest_transcript_path()
                if live_path:
                    _cached_index._current_session_id = extract_session_id(live_path)
            except Exception:
                pass
        return _cached_index
    except Exception as e:
        logging.getLogger("synapt.recall").warning("Index load failed: %s", e)
        _cached_index = None
        return None


def _invalidate_cache() -> None:
    """Reset cached index so next search reloads from disk."""
    global _cached_index, _cached_mtime, _cached_dir
    if _cached_index is not None and getattr(_cached_index, '_db', None) is not None:
        with contextlib.suppress(Exception):
            _cached_index._db.close()
    _cached_index = None
    _cached_mtime = 0.0
    _cached_dir = None


# Clean up the DB connection before Python's module teardown begins.
# Without this, __del__ runs during shutdown when sqlite3 may already
# be partially torn down, causing a non-zero exit code that Claude Code
# reports as "MCP tool failed."
atexit.register(_invalidate_cache)


# ---------------------------------------------------------------------------
# Tool implementations — usable both as MCP tools and as plain functions.
# ---------------------------------------------------------------------------


def recall_search(
    query: str,
    max_chunks: int = 5,
    max_tokens: int = 1500,
    max_sessions: int | None = None,
    after: str | None = None,
    before: str | None = None,
    half_life: float = 60.0,
    threshold_ratio: float = 0.2,
    depth: str = "full",
    include_archived: bool = False,
    include_historical: bool = False,
) -> str:
    """Search past coding sessions for relevant context. USE PROACTIVELY.

    Call this BEFORE starting work to check for prior decisions, bugs, or
    approaches. Returns relevant conversation chunks sorted by relevance.

    When to use (without being asked):
    - User mentions past work → search for it
    - Debugging an error → search for similar past errors
    - Making a design decision → check if it was discussed before
    - Starting a new feature → check for prior attempts or related work

    Args:
        query: Natural language query or keywords to search for.
        max_chunks: Maximum number of result chunks to return.
        max_tokens: Approximate token budget for the response.
        max_sessions: If set, only search the N most recent sessions.
        after: Only include results from after this date (ISO 8601, e.g. "2026-02-28").
        before: Only include results from before this date (ISO 8601, e.g. "2026-03-01").
        half_life: Days for recency decay to reach ~50%. 0 disables decay.
                   Default 60: a session from 2 months ago scores ~50% of today's,
                   making work from past quarters still discoverable.
                   Passed as None to lookup() so intent classification can
                   override when the caller uses the MCP default.
        threshold_ratio: Drop results scoring below this fraction of the top score. 0 disables.
        depth: "full" returns knowledge + journal + transcript results.
               "summary" returns only knowledge nodes + journal entries.
               "concise" returns knowledge + cluster summaries (no raw chunks).
        include_archived: If True, include archived clusters in concise mode results.
                          In full mode, individual chunks are always searchable regardless.
        include_historical: If True, include superseded/contradicted knowledge nodes
                            in results. When multiple versions exist in the same lineage,
                            the highest-confidence one is kept (confidence-based fallback).
    """
    max_tokens = _cap_tokens(max_tokens)
    index = _get_index()

    # Always search the live transcript — covers the current session which is
    # not yet archived.  Live results get ≤1/3 of the total token budget.
    # Skip entirely when max_tokens=0 to avoid emitting output the caller
    # did not budget for (the first-chunk guarantee in _format_live_results
    # still fires at max_tokens=0, producing unexpected output).
    from synapt.recall.live import search_live_transcript
    live_budget = min(500, max_tokens // 3)
    live_result = ""
    if live_budget > 0:
        live_result = search_live_transcript(
            query,
            index=index,
            max_chunks=min(3, max_chunks),
            max_tokens=live_budget,
        )
    # Reserve budget for indexed search.  Floor is min(500, max_tokens) so
    # indexed search always gets a meaningful allocation without exceeding the
    # caller's total budget (a bare floor of 500 would exceed max_tokens when
    # max_tokens < 500, e.g. in tests or constrained callers).
    indexed_budget = max_tokens
    if live_result:
        # Ceiling division avoids rounding to 0 for very short live results
        # (len < 4 chars), which would deduct nothing from the indexed budget
        # despite emitting output.  In practice live blocks are always >100
        # chars, but the approximation should be accurate in all cases.
        live_consumed = (len(live_result) + 3) // 4
        budget_floor = min(500, max_tokens)
        indexed_budget = max(max_tokens - live_consumed, budget_floor)

    if index is None:
        if live_result:
            return live_result
        index_dir = project_index_dir()
        return f"No index found at {index_dir}. Run `synapt recall setup` first."

    try:
        # Pass half_life=None when caller used the MCP default (60.0) so
        # intent classification can override it. When the user explicitly
        # sets a value, pass it through as-is to honour their choice.
        effective_hl: float | None = None if half_life == 60.0 else half_life
        result = index.lookup(
            query,
            max_chunks=max_chunks,
            max_tokens=indexed_budget,
            max_sessions=max_sessions,
            after=after,
            before=before,
            half_life=effective_hl,
            threshold_ratio=threshold_ratio,
            depth=depth,
            include_archived=include_archived,
            include_historical=include_historical,
        )
        # Combine live (current session) + indexed (past sessions) results
        parts = []
        if live_result:
            parts.append(live_result)
        if result:
            parts.append(result)

        # Surface contradiction warnings from co-retrieval detection
        conflicts = getattr(index, "_last_conflicts", [])
        if conflicts:
            warn_lines = [f"\n⚠ Conflicting information detected ({len(conflicts)} conflict(s)):"]
            for old, new in conflicts[:3]:  # Cap at 3 to avoid noise
                warn_lines.append(
                    f"  • \"{old.get('content', '')[:80]}\" vs \"{new.get('content', '')[:80]}\""
                )
            warn_lines.append("Use recall_contradict(action='list') to review and resolve.")
            parts.append("\n".join(warn_lines))

        # Surface embedding status when search is degraded
        if index._embedding_status == "unavailable":
            parts.append(
                f"\n[Note: Search is using keyword matching only (BM25). "
                f"{index._embedding_reason}]"
            )

        if parts:
            return "\n\n".join(parts)
        # Surface diagnostics explaining why search returned nothing
        diag = index._last_diagnostics
        if diag:
            msg = diag.format_message()
            if index._embedding_status == "unavailable":
                msg += (
                    f"\n[Note: Embeddings unavailable — semantic search disabled. "
                    f"{index._embedding_reason}]"
                )
            return msg
        return "No results found."
    except Exception as exc:
        return f"Search failed: {exc}"


def recall_quick(query: str) -> str:
    """Quick, low-cost memory check. Use this speculatively — when you're
    not sure if past context exists but want to check.

    Returns only knowledge nodes and cluster summaries (no raw transcript
    chunks), keeping output short and fast. If you find something relevant,
    follow up with recall_search or recall_context for full detail.

    Cost: ~500 tokens, <100ms. Cheaper than guessing wrong.
    Use BEFORE making assumptions about past work, decisions, or conventions.

    Args:
        query: Natural language query or keywords to search for.
    """
    quick_budget = _cap_tokens(500)
    index = _get_index()
    if index is None:
        index_dir = project_index_dir()
        return f"No index found at {index_dir}. Run `synapt recall setup` first."

    try:
        result = index.lookup(
            query,
            max_chunks=5,
            max_tokens=quick_budget,
            half_life=None,
            threshold_ratio=0.2,
            depth="concise",
        )
        if result:
            return result
        diag = index._last_diagnostics
        if diag:
            return diag.format_message()
        return "No results found."
    except Exception as exc:
        return f"Search failed: {exc}"


def recall_files(
    pattern: str,
    max_chunks: int = 10,
    max_tokens: int = 1500,
    after: str | None = None,
    before: str | None = None,
) -> str:
    """Find past sessions that touched a specific file.

    Searches the files_touched metadata of all indexed conversation turns.
    Supports partial path matching: 'repair.py' matches 'src/graph/repair.py'.

    Args:
        pattern: File path or partial path to search for.
        max_chunks: Maximum number of result chunks to return.
        max_tokens: Approximate token budget for the response.
        after: Only include results from after this date (ISO 8601).
        before: Only include results from before this date (ISO 8601).
    """
    max_tokens = _cap_tokens(max_tokens)
    index = _get_index()
    if index is None:
        index_dir = project_index_dir()
        return f"No index found at {index_dir}. Run `synapt recall setup` first."

    try:
        result = index.lookup_files(
            pattern,
            max_chunks=max_chunks,
            max_tokens=max_tokens,
            after=after,
            before=before,
        )
        return result if result else f"No sessions found that touched files matching '{pattern}'."
    except Exception as exc:
        return f"File search failed: {exc}"


def recall_sessions(
    max_sessions: int = 20,
    after: str | None = None,
    before: str | None = None,
) -> str:
    """List recent sessions with date, turn count, and first message.

    Returns a summary of the most recent sessions in the transcript index,
    useful for browsing what past sessions covered.

    Args:
        max_sessions: Maximum number of sessions to list.
        after: Only sessions with activity after this date (ISO 8601).
        before: Only sessions with activity before this date (ISO 8601).
    """
    index = _get_index()
    if index is None:
        index_dir = project_index_dir()
        return f"No index found at {index_dir}. Run `synapt recall setup` first."

    try:
        sessions = index.list_sessions(
            max_sessions=max_sessions,
            after=after,
            before=before,
        )
    except Exception as exc:
        return f"Session listing failed: {exc}"

    if not sessions:
        return "No sessions found."

    lines = [f"Recent sessions ({len(sessions)}):"]
    for s in sessions:
        lines.append(
            f"  {s['date']}  {s['session_id'][:8]}  "
            f"{s['turn_count']} turns  {s['files_count']} files  "
            f"\"{s['first_message']}\""
        )
    return "\n".join(lines)


def recall_build(incremental: bool = True) -> str:
    """Build or rebuild the transcript index from auto-discovered sources.

    Archives transcripts from Claude Code's source directory into the project,
    then builds a searchable index at <project>/.synapt/recall/index/.

    Args:
        incremental: If True, skip already-indexed files for faster rebuilds.
    """
    from synapt.recall.cli import _archive_and_build

    project = Path.cwd().resolve()
    try:
        final_index = _archive_and_build(
            project,
            use_embeddings=True,
            incremental=incremental,
        )
    except Exception as e:
        return f"Build failed: {e}"
    finally:
        _invalidate_cache()

    if not final_index or not final_index.chunks:
        return "No Claude Code transcripts found for this project."

    stats = final_index.stats()
    index_dir = project_index_dir(project)
    return (
        f"Index built: {stats['chunk_count']} chunks from "
        f"{stats['session_count']} sessions. Saved to {index_dir}"
    )


def recall_setup(no_hook: bool = False) -> str:
    """Initialize synapt recall for the current project.

    Archives transcripts, builds the search index, installs global hooks
    (SessionStart, SessionEnd, PreCompact), and adds .synapt/ to .gitignore.

    Args:
        no_hook: If True, skip installing global hooks.
    """
    from synapt.recall.cli import _archive_and_build, _ensure_gitignore, _install_global_hooks

    project = Path.cwd().resolve()
    steps: list[str] = []

    # 1. Archive transcripts + build index
    try:
        final_index = _archive_and_build(project, use_embeddings=True)
    except Exception as e:
        return f"Setup failed during index build: {e}"
    finally:
        _invalidate_cache()

    if not final_index or not final_index.chunks:
        return (
            "No Claude Code transcripts found for this project. "
            "Start a Claude Code session first, then run recall_setup again."
        )

    stats = final_index.stats()
    steps.append(
        f"Index built: {stats['chunk_count']} chunks from "
        f"{stats['session_count']} sessions"
    )

    # 2. Install global hooks (user-level ~/.claude/settings.json)
    if not no_hook:
        installed = _install_global_hooks()
        if installed:
            steps.append(f"Installed {installed} global hook(s)")
        else:
            steps.append("Global hooks already registered")
    else:
        steps.append("Hook installation skipped (no_hook=True)")

    # 3. Add .synapt/ to .gitignore
    _ensure_gitignore(project)
    steps.append(".synapt/ ensured in .gitignore")

    # Summary
    index_dir = project_index_dir(project)
    total_size = sum(fp.stat().st_size for fp in index_dir.iterdir() if fp.is_file())
    steps.append(f"Index size: {format_size(total_size)}")

    return "Setup complete.\n" + "\n".join(f"  - {s}" for s in steps)


def recall_stats() -> str:
    """Get statistics about the transcript index.

    Returns chunk count, session count, date range, and index size.
    """
    index_dir = project_index_dir()

    # Check for recall.db or legacy manifest.json
    if not (index_dir / "recall.db").exists() and not (index_dir / "manifest.json").exists():
        return f"No index found at {index_dir}. Run `synapt recall setup` first."

    index = _get_index()
    if index is None:
        return f"No index found at {index_dir}."

    try:
        stats = index.stats()

        # Load manifest from DB or legacy file
        manifest: dict = {}
        try:
            if index._db:
                manifest = index._db.load_manifest()
            elif (index_dir / "manifest.json").exists():
                with open(index_dir / "manifest.json", encoding="utf-8") as f:
                    manifest = json.load(f)
        except Exception:
            pass

        lines = [
            f"Chunks: {stats.get('chunk_count', 0)}",
            f"Sessions: {stats.get('session_count', 0)}",
            f"Avg chunks/session: {stats.get('avg_chunks_per_session', 0):.1f}",
        ]
        if stats.get("date_range"):
            dr = stats["date_range"]
            lines.append(f"Date range: {dr['earliest'][:10]} to {dr['latest'][:10]}")
        lines.append(f"Unique tools: {stats.get('total_tools_used', 0)}")
        lines.append(f"Unique files: {stats.get('total_files_touched', 0)}")

        # Embedding status
        if stats.get("embeddings_active"):
            lines.append(f"Embeddings: active ({stats.get('embedding_provider', 'unknown')})")
        elif index._embedding_status == "unavailable":
            lines.append(
                f"Embeddings: unavailable — using BM25 keyword search only. "
                f"{index._embedding_reason}"
            )
        else:
            lines.append("Embeddings: inactive (not requested)")

        # Knowledge nodes
        kn_count = stats.get("knowledge_count", 0)
        if kn_count > 0:
            lines.append(f"Knowledge nodes: {kn_count}")

        # Pending contradictions
        if index._db:
            n_pending = index._db.pending_contradiction_count()
            if n_pending > 0:
                lines.append(f"Pending contradictions: {n_pending}")

        # Clusters
        if index._db:
            n_topic = index._db.cluster_count(cluster_type="topic")
            n_timeline = index._db.cluster_count(cluster_type="timeline")
            n_archived = index._db._conn.execute(
                "SELECT COUNT(*) FROM clusters WHERE status = 'archived'"
            ).fetchone()[0]
            if n_topic > 0 or n_timeline > 0 or n_archived > 0:
                cluster_parts = []
                if n_topic > 0:
                    cluster_parts.append(f"{n_topic} topic")
                if n_timeline > 0:
                    cluster_parts.append(f"{n_timeline} timeline arcs")
                if n_archived > 0:
                    cluster_parts.append(f"{n_archived} archived")
                lines.append(f"Clusters: {', '.join(cluster_parts)}")

        # Access tracking
        if index._db:
            access = index._db.access_summary()
            if access["total_events"] > 0:
                lines.append(
                    f"Access events: {access['total_events']} "
                    f"({access['tracked_items']} items, "
                    f"{access['items_drilled_into']} drilled into)"
                )
                tiers = access.get("promotion_tiers", {})
                if tiers:
                    tier_parts = [
                        f"{count} {tier}"
                        for tier, count in sorted(tiers.items())
                        if count > 0
                    ]
                    if tier_parts:
                        lines.append(f"Promotion tiers: {', '.join(tier_parts)}")
                # Decay distribution
                decay = index._db.decay_distribution()
                decay_total = sum(decay.values())
                if decay_total > 0:
                    lines.append(
                        f"Decay: {decay['fresh']} fresh, {decay['warm']} warm, "
                        f"{decay['cool']} cool, {decay['cold']} cold"
                    )

        # Storage backend
        backend = stats.get("storage_backend", "memory")
        lines.append(f"Storage: {backend}")

        build_ts = manifest.get("build_timestamp", "unknown")
        if isinstance(build_ts, str):
            lines.append(f"Built: {build_ts[:19]}")

        total_size = sum(fp.stat().st_size for fp in index_dir.iterdir() if fp.is_file())
        lines.append(f"Index size: {format_size(total_size)}")

        return "\n".join(lines)
    except Exception as exc:
        return f"Stats failed: {exc}"


def recall_consolidate(
    dry_run: bool = False,
    force: bool = False,
    model: str = "",
    adapter_path: str = "",
) -> str:
    """Consolidate session knowledge — extract durable patterns from journal entries.

    Analyzes enriched journal entries across multiple sessions to distill
    durable knowledge nodes: facts, conventions, decisions, and lessons
    that persist across sessions. Analogous to memory consolidation during sleep.

    Run this periodically (e.g., every few sessions) to build up the
    project's knowledge base. Knowledge nodes appear in search results
    and are surfaced at session start.

    Args:
        dry_run: Preview what would be consolidated without making changes.
        force: Reprocess all journal entries, ignoring last consolidation timestamp.
        model: MLX model to use (default: Ministral-3-3B-Instruct-2512-4bit).
        adapter_path: Optional LoRA adapter path for knowledge extraction.
    """
    try:
        from synapt.recall.consolidate import consolidate
        kwargs: dict = {"dry_run": dry_run, "force": force}
        if model:
            kwargs["model"] = model
        if adapter_path:
            kwargs["adapter_path"] = adapter_path
        result = consolidate(**kwargs)
        if dry_run:
            return (
                f"Dry run: {result.entries_processed} entries, "
                f"{result.clusters_found} clusters found"
            )
        parts = []
        if result.nodes_created:
            parts.append(f"{result.nodes_created} created")
        if result.nodes_corroborated:
            parts.append(f"{result.nodes_corroborated} corroborated")
        if result.nodes_contradicted:
            parts.append(f"{result.nodes_contradicted} contradicted")
        if result.nodes_deduped:
            parts.append(f"{result.nodes_deduped} deduped")
        if not parts:
            return (
                f"No knowledge extracted ({result.entries_processed} entries, "
                f"{result.clusters_found} clusters)"
            )
        return (
            f"Consolidated: {', '.join(parts)} "
            f"({result.entries_processed} entries, {result.clusters_found} clusters)"
        )
    except Exception as exc:
        return f"Consolidation failed: {exc}"
    finally:
        _invalidate_cache()


def recall_contradict(
    action: str = "list",
    contradiction_id: int | None = None,
    resolution: str = "confirmed",
    claim: str | None = None,
    new_content: str | None = None,
    old_node_id: str | None = None,
    reason: str | None = None,
) -> str:
    """Manage pending knowledge contradictions.

    When the system detects that a new observation contradicts an existing
    knowledge node, it queues a pending contradiction for user review.
    Use this tool to list, confirm, dismiss, or flag new contradictions.

    Args:
        action: "list" (show pending), "resolve" (confirm/dismiss one),
                "flag" (report a new contradiction).
        contradiction_id: ID of the contradiction to resolve (required for "resolve").
        resolution: "confirmed" (supersede old node) or "dismissed" (keep old node).
        claim: Free-text description of conflicting information (for "flag").
               The system will search for matching knowledge nodes automatically.
        new_content: The correct/updated information (for "flag"). If omitted,
                     *claim* is used as both the description and the new content.
        old_node_id: Optional knowledge node ID that the claim contradicts.
                     If omitted, the system searches for matching nodes.
        reason: Why this is a contradiction (for "flag").
    """
    index = _get_index()
    if index is None or not index._db:
        return "No index found. Run `synapt recall setup` first."

    try:
        if action == "flag":
            return _handle_flag(
                index, claim=claim, new_content=new_content,
                old_node_id=old_node_id, reason=reason,
            )

        if action == "list":
            pending = index._db.list_pending_contradictions()
            if not pending:
                return "No pending contradictions."
            # Build lookup dict once (not per-iteration)
            old_ids = {c["old_node_id"] for c in pending if c["old_node_id"]}
            node_lookup = {
                nid: index._db.get_knowledge_node(nid)
                for nid in old_ids
            }
            lines = [f"Pending contradictions ({len(pending)}):"]
            for c in pending:
                old_node = node_lookup.get(c["old_node_id"]) if c["old_node_id"] else None
                old_content = old_node["content"] if old_node else ""
                lines.append(
                    f"\n  #{c['id']} ({c['detected_by']}, {c['detected_at'][:10]})"
                )
                if c.get("claim_text"):
                    lines.append(f"    Claim: {c['claim_text']}")
                if old_content:
                    lines.append(f"    Old: {old_content}")
                elif not c["old_node_id"]:
                    lines.append("    Old: (no matching node — free-text claim)")
                lines.append(f"    New: {c['new_content']}")
                if c["reason"]:
                    lines.append(f"    Reason: {c['reason']}")
            return "\n".join(lines)

        if action == "resolve":
            if contradiction_id is None:
                return "Error: contradiction_id is required for 'resolve' action."
            ok = index._db.resolve_contradiction(contradiction_id, resolution)
            if not ok:
                return f"Contradiction #{contradiction_id} not found or already resolved."

            if resolution == "confirmed":
                # Read the now-resolved row (status='confirmed', not 'pending')
                # to extract fields for supersession. This intentionally reads
                # by id regardless of status — do not add a status filter here.
                pending_row = index._db._conn.execute(
                    "SELECT * FROM pending_contradictions WHERE id = ?",
                    (contradiction_id,),
                ).fetchone()
                if pending_row and pending_row["old_node_id"]:
                    _apply_supersession(
                        index._db,
                        old_node_id=pending_row["old_node_id"],
                        new_content=pending_row["new_content"],
                        category=pending_row["category"],
                        reason=pending_row["reason"],
                        source_sessions=json.loads(pending_row["source_sessions"]),
                    )
                elif pending_row and not pending_row["old_node_id"]:
                    # Free-text claim confirmed — create a new knowledge node
                    _create_knowledge_from_claim(index._db, pending_row)
                    return f"Contradiction #{contradiction_id} confirmed — new knowledge node created from claim."
                return f"Contradiction #{contradiction_id} confirmed — old node superseded."
            return f"Contradiction #{contradiction_id} dismissed — old node retained."

        return f"Unknown action: {action}. Use 'list', 'resolve', or 'flag'."
    except Exception as exc:
        return f"Contradiction management failed: {exc}"
    finally:
        _invalidate_cache()


def _handle_flag(
    index,
    claim: str | None,
    new_content: str | None,
    old_node_id: str | None,
    reason: str | None,
) -> str:
    """Handle the 'flag' action for recall_contradict.

    Accepts a free-text claim and optionally searches for matching knowledge
    nodes.  If *old_node_id* is provided, uses it directly.  Otherwise,
    searches the knowledge FTS index for candidates.
    """
    if not claim and not new_content:
        return "Error: 'claim' or 'new_content' is required for 'flag' action."

    effective_claim = claim or ""
    effective_content = new_content or claim or ""
    effective_reason = reason or ""

    # If a specific node ID was provided, use it directly
    if old_node_id:
        node = index._db.get_knowledge_node(old_node_id)
        if not node:
            return f"Error: knowledge node '{old_node_id}' not found."
        cid = index._db.add_pending_contradiction(
            old_node_id=old_node_id,
            new_content=effective_content,
            reason=effective_reason,
            detected_by="manual",
            claim_text=effective_claim or None,
        )
        return (
            f"Contradiction #{cid} flagged against node '{old_node_id}': "
            f"\"{node['content'][:80]}\" → \"{effective_content[:80]}\". "
            f"Use recall_contradict(action='resolve', contradiction_id={cid}) to confirm or dismiss."
        )

    # Search for matching knowledge nodes via FTS
    search_text = effective_claim or effective_content
    try:
        fts_results = index._db.knowledge_fts_search(search_text, limit=3)
        if fts_results:
            rowids = [r[0] for r in fts_results]
            node_map = index._db.knowledge_by_rowid(rowids)
            matches = [node_map[rid] for rid in rowids if rid in node_map]
        else:
            matches = []
    except Exception:
        matches = []

    if matches:
        # Flag against the best match, mention alternatives
        best = matches[0]
        cid = index._db.add_pending_contradiction(
            old_node_id=best["id"],
            new_content=effective_content,
            reason=effective_reason,
            detected_by="manual",
            claim_text=effective_claim or None,
        )
        lines = [
            f"Contradiction #{cid} flagged against best-matching node '{best['id']}':",
            f"  Old: \"{best['content'][:100]}\"",
            f"  New: \"{effective_content[:100]}\"",
        ]
        if len(matches) > 1:
            lines.append(f"  Other candidates: {', '.join(m['id'] for m in matches[1:])}")
        lines.append(
            f"Use recall_contradict(action='resolve', contradiction_id={cid}) to confirm or dismiss."
        )
        return "\n".join(lines)

    # No matching nodes — store as free-text claim for review
    cid = index._db.add_pending_contradiction(
        old_node_id=None,
        new_content=effective_content,
        reason=effective_reason,
        detected_by="manual",
        claim_text=effective_claim or None,
    )
    return (
        f"Contradiction #{cid} flagged as free-text claim (no matching knowledge node found). "
        f"Claim: \"{effective_content[:100]}\". "
        f"When confirmed, a new knowledge node will be created. "
        f"Use recall_contradict(action='resolve', contradiction_id={cid}) to confirm or dismiss."
    )


def _create_knowledge_from_claim(db, pending_row) -> None:
    """Create a new knowledge node from a confirmed free-text claim."""
    from synapt.recall.knowledge import KnowledgeNode

    content = pending_row["new_content"]
    category = pending_row["category"] or "workflow"
    node = KnowledgeNode.create(content, category)
    db.upsert_knowledge_node(node.__dict__)


def _apply_supersession(
    db,
    old_node_id: str,
    new_content: str,
    category: str,
    reason: str,
    source_sessions: list[str],
) -> None:
    """Execute a confirmed supersession: mark old node, create replacement."""
    now = datetime.now(timezone.utc).isoformat()

    old_node = db.get_knowledge_node(old_node_id)
    if old_node is None:
        return

    lineage_id = old_node.get("lineage_id", "") or old_node["id"]
    old_version = old_node.get("version", 1)

    # Mark old node as contradicted — also backfill lineage_id if bootstrapping
    old_node["lineage_id"] = lineage_id
    old_node["status"] = "contradicted"
    old_node["valid_until"] = now
    old_node["contradiction_note"] = reason
    old_node["updated_at"] = now
    new_id = uuid.uuid4().hex[:12]
    old_node["superseded_by"] = new_id
    db.upsert_knowledge_node(old_node)

    # Create replacement node
    new_node = {
        "id": new_id,
        "content": _tw(new_content, 300),
        "category": category or old_node.get("category", "workflow"),
        "confidence": old_node.get("confidence", 0.5),
        "source_sessions": source_sessions or old_node.get("source_sessions", []),
        "created_at": now,
        "updated_at": now,
        "status": "active",
        "superseded_by": "",
        "contradiction_note": "",
        "tags": old_node.get("tags", []),
        "valid_from": now,
        "valid_until": None,
        "version": old_version + 1,
        "lineage_id": lineage_id,
    }
    db.upsert_knowledge_node(new_node)


def format_contradictions_for_session_start() -> str:
    """Format pending contradictions for the SessionStart hook.

    Returns a string to print to stdout (which becomes the system-reminder
    the model sees). The model should then ask the user about each one.
    Returns empty string if no pending contradictions.

    Uses a lightweight DB connection instead of loading the full index,
    since only SQL queries are needed (no chunk/embedding data).  Fixes #119.
    """
    from synapt.recall.storage import RecallDB

    db_path = project_index_dir() / "recall.db"
    if not db_path.exists():
        return ""

    try:
        db = RecallDB(db_path)
        pending = db.list_pending_contradictions()
        if not pending:
            db.close()
            return ""
        # Build node lookup for old content
        old_ids = {c["old_node_id"] for c in pending}
        node_lookup = {
            nid: db.get_knowledge_node(nid)
            for nid in old_ids
        }
        db.close()
        lines = [f"Pending contradictions ({len(pending)}) — ask the user to resolve:"]
        for c in pending:
            old_node = node_lookup.get(c["old_node_id"])
            old_content = old_node["content"][:120] if old_node else "(deleted node)"
            new_content = c["new_content"][:120]
            lines.append(f"  #{c['id']}: \"{old_content}\"")
            lines.append(f"    -> \"{new_content}\"")
            if c["reason"]:
                lines.append(f"    Reason: {c['reason']}")
        lines.append("Use recall_contradict to confirm or dismiss each one based on user input.")
        return "\n".join(lines)
    except Exception:
        return ""


def recall_context(
    chunk_id: str | None = None,
    cluster_id: str | None = None,
) -> str:
    """Drill down into a search result to see full raw transcript content.

    Two modes:
    - chunk_id: Show full raw transcript for a single turn (e.g., "a1b2c3d4:t5")
    - cluster_id: Show all chunks in a topic cluster (e.g., "clust-abcd1234")

    If both are provided, cluster_id takes precedence.

    Use after recall_search finds a relevant turn/cluster but you need
    the complete detail.

    Args:
        chunk_id: The chunk identifier from a search result.
        cluster_id: A cluster identifier to show all member chunks.
    """
    try:
        idx = _get_index()
        if idx is None:
            return "No index found. Run recall_build first."

        if cluster_id:
            if cluster_id.startswith("tl-"):
                return (
                    f"Timeline arc {cluster_id} is a session-level grouping, "
                    "not a chunk cluster. Use `recall_timeline` to view it."
                )
            result = _format_cluster_context(idx, cluster_id)
            if "not found" not in result:
                _record_context_access(idx, "cluster", cluster_id)
            return result
        if chunk_id:
            result = idx.read_turn_context(chunk_id)
            if result:
                _record_context_access(idx, "chunk", chunk_id)
            return result
        return "Provide either chunk_id or cluster_id."
    except Exception as exc:
        return f"Error reading context: {exc}"


def _record_context_access(idx, item_type: str, item_id: str) -> None:
    """Record an explicit drill-down access and check promotions (fire-and-forget)."""
    try:
        if idx._db:
            idx._db.record_access(
                [{"item_type": item_type, "item_id": item_id, "score": 1.0}],
                context="context",
            )
            from synapt.recall.promotion import (
                check_promotions, execute_cheap_promotions,
            )
            actions = check_promotions(idx._db, item_type, item_id)
            if actions:
                execute_cheap_promotions(idx._db, item_type, item_id, actions)
    except Exception:
        pass


def _format_cluster_context(idx, cluster_id: str) -> str:
    """Format all chunks in a cluster for drill-down context."""
    db = idx._db
    if db is None:
        return "No database available for cluster lookup."

    chunk_ids = db.get_cluster_chunks(cluster_id)
    if not chunk_ids:
        return f"Cluster {cluster_id} not found or has no chunks."

    cluster_info = db.get_cluster(cluster_id)
    topic = cluster_info["topic"] if cluster_info else "unknown"

    parts = [f"Cluster: {topic} ({len(chunk_ids)} chunks)\n"]
    for cid in chunk_ids:
        context = idx.read_turn_context(cid)
        parts.append(context)
        parts.append("")  # blank line separator

    return "\n".join(parts)


def recall_journal(
    action: str = "read",
    focus: str | None = None,
    done: str | None = None,
    decisions: str | None = None,
    next_steps: str | None = None,
) -> str:
    """Read or write session journal entries.

    The journal tracks what happened each session: focus, done items,
    decisions, and next steps. Use "write" at end of session to persist
    context for next time. Use "read" to see the latest entry.

    Args:
        action: "read" (latest entry), "write" (create entry), or "list" (recent entries).
        focus: What this session was about (write only).
        done: Semicolon-separated list of accomplishments (write only).
        decisions: Semicolon-separated list of key decisions (write only).
        next_steps: Semicolon-separated list of next steps (write only).
    """
    try:
        from synapt.recall.journal import (
            append_entry,
            auto_extract_entry,
            format_entry_full,
            format_for_session_start,
            latest_transcript_path,
            read_entries,
            read_latest,
        )

        if action == "read":
            entry = read_latest(meaningful=True)
            if not entry:
                return "No journal entries yet."
            return format_for_session_start(entry)

        if action == "list":
            entries = read_entries(n=5)
            if not entries:
                return "No journal entries yet."
            return "\n\n---\n\n".join(format_entry_full(e) for e in entries)

        if action == "write":
            project = Path.cwd().resolve()
            transcript_path = latest_transcript_path(project)
            entry = auto_extract_entry(transcript_path=transcript_path, cwd=str(project))

            if focus:
                entry.focus = focus
            if done:
                entry.done = [d.strip() for d in done.split(";")]
            if decisions:
                entry.decisions = [d.strip() for d in decisions.split(";")]
            if next_steps:
                entry.next_steps = [n.strip() for n in next_steps.split(";")]

            # Clear auto flag when user provides rich content
            if entry.has_rich_content():
                entry.auto = False

            # Skip auto entries with no rich content
            if entry.auto and not entry.has_rich_content():
                return "No rich content to journal (auto-extract only)."
            if not entry.has_content():
                return "No content to journal (no files modified, no fields provided)."

            append_entry(entry)
            return f"Journal entry written.\n\n{format_entry_full(entry)}"

        return f"Unknown action: {action}. Use 'read', 'write', or 'list'."
    except Exception as exc:
        return f"Journal failed: {exc}"


def recall_remind(
    action: str = "add",
    text: str | None = None,
    reminder_id: str | None = None,
    sticky: bool = False,
) -> str:
    """Manage session reminders — lightweight nudges surfaced at session start.

    Reminders auto-clear after being shown once unless marked sticky.
    Use this to flag things to bring up next session.

    Args:
        action: "add" (create reminder), "list" (show all), "clear" (remove by id or all), "pending" (show pending).
        text: Reminder text (required for "add").
        reminder_id: Reminder ID (optional for "clear" — clears all if omitted).
        sticky: If true, reminder persists across sessions instead of auto-clearing.
    """
    try:
        from synapt.recall.reminders import (
            add_reminder,
            clear_reminder,
            load_reminders,
            pop_pending,
            format_for_session_start,
        )

        if action == "add":
            if not text:
                return "Error: text is required for 'add' action."
            reminder = add_reminder(text, sticky=sticky)
            sticky_label = " (sticky)" if sticky else ""
            return f"Added reminder{sticky_label}: {reminder.text} (id: {reminder.id})"

        if action == "list":
            reminders = load_reminders()
            if not reminders:
                return "No reminders."
            lines = []
            for r in reminders:
                s = " [sticky]" if r.sticky else ""
                shown = f" (shown {r.shown_count}x)" if r.shown_count > 0 else ""
                lines.append(f"  {r.id}  {r.text}{s}{shown}")
            return "\n".join(lines)

        if action == "clear":
            count = clear_reminder(reminder_id)
            return f"Cleared {count} reminder(s)." if count else "No reminders to clear."

        if action == "pending":
            pending = pop_pending()  # Single load-save cycle
            if not pending:
                return "No pending reminders."
            return format_for_session_start(pending)

        return f"Unknown action: {action}. Use 'add', 'list', 'clear', or 'pending'."
    except Exception as exc:
        return f"Reminder failed: {exc}"


def recall_enrich(
    model: str = "",
    max_entries: int = 10,
    dry_run: bool = False,
    adapter_path: str = "",
) -> str:
    """Enrich auto-generated journal stubs using a local MLX model.

    Reads journal entries tagged as auto-generated that lack rich content
    (done/decisions/next_steps), loads the original transcript, and uses
    MLX to extract structured information. Appends enriched entries to
    journal.jsonl.

    Requires mlx-lm to be installed (pip install mlx-lm). Safe to run
    multiple times — already-enriched entries are skipped.

    Args:
        model: MLX model to use for summarization (default: Ministral-3-3B-Instruct-2512-4bit).
        max_entries: Maximum entries to enrich per invocation.
        dry_run: If True, report what would be enriched without modifying anything.
        adapter_path: Optional LoRA adapter path for enrichment.
    """
    from synapt.recall.enrich import enrich_all, _MLX_AVAILABLE, _INSTALL_MSG

    if not _MLX_AVAILABLE:
        return _INSTALL_MSG

    project = Path.cwd().resolve()
    kwargs: dict = {"project_dir": project, "dry_run": dry_run, "max_entries": max_entries}
    if model:
        kwargs["model"] = model
    if adapter_path:
        kwargs["adapter_path"] = adapter_path
    try:
        count = enrich_all(**kwargs)
    except Exception as exc:
        return f"Enrichment failed: {exc}"
    finally:
        _invalidate_cache()

    if dry_run:
        return f"Dry run: {count} entries would be enriched."
    if count:
        return f"Enriched {count} journal entries. Run `recall_build` to re-index."
    return "No entries to enrich (all sessions already have journal entries)."


def recall_timeline(
    query: str = "",
    after: str | None = None,
    before: str | None = None,
    branch: str | None = None,
    max_results: int = 10,
) -> str:
    """View chronological timeline of work arcs.

    Returns session arcs — groups of consecutive sessions on the same
    branch/topic — ordered chronologically. Each arc shows date range,
    branch, sessions, and key accomplishments from journal entries.

    Args:
        query: Optional text query to filter arcs via FTS.
        after: Only arcs ending after this date (ISO 8601).
        before: Only arcs starting before this date (ISO 8601).
        branch: Filter to arcs on a specific branch.
        max_results: Maximum number of arcs to return (default 10).
    """
    index = _get_index()
    if index is None:
        return "No index found. Run `recall_build` first."
    if not index._db:
        return "No database found. Run `recall_build` to create one."

    try:
        db = index._db

        if query:
            # FTS search filtered to timeline type + date/branch filters
            from synapt.recall.storage import _escape_fts_query

            escaped = _escape_fts_query(query, use_or=True)
            if not escaped:
                return "No valid search terms in query."
            sql = (
                "SELECT c.* FROM clusters c "
                "JOIN clusters_fts f ON c.id = f.rowid "
                "WHERE clusters_fts MATCH ? "
                "AND c.cluster_type = 'timeline' AND c.status = 'active'"
            )
            params: list[str] = [escaped]
            if after:
                sql += " AND c.date_end >= ?"
                params.append(after)
            if before:
                sql += " AND c.date_start <= ?"
                params.append(before)
            if branch:
                sql += " AND c.branch = ?"
                params.append(branch)
            sql += " ORDER BY c.date_start"
            rows = db._conn.execute(sql, params).fetchall()
            arcs = [
                {
                    "cluster_id": r["cluster_id"],
                    "topic": r["topic"],
                    "session_ids": json.loads(r["session_ids"]),
                    "branch": r["branch"],
                    "date_start": r["date_start"],
                    "date_end": r["date_end"],
                    "chunk_count": r["chunk_count"],
                    "tags": json.loads(r["tags"]) if r["tags"] else [],
                }
                for r in rows
            ]
        else:
            arcs = db.load_timeline_clusters(
                after=after, before=before, branch=branch
            )

        if not arcs:
            return "No timeline arcs found."

        arcs = arcs[:max_results]

        # Load journal entries from ALL worktrees, prefer enriched
        from synapt.recall.journal import (
            _journal_path, _read_all_entries, _dedup_entries,
        )
        from synapt.recall.core import all_worktree_archive_dirs
        from synapt.recall.scrub import strip_system_artifacts

        all_jentries: list = []
        local_jp = _journal_path()
        if local_jp.exists():
            all_jentries.extend(_read_all_entries(local_jp))
        for wt_archive in all_worktree_archive_dirs():
            wt_jp = wt_archive.parent / "journal.jsonl"
            if wt_jp.exists() and wt_jp.resolve() != local_jp.resolve():
                all_jentries.extend(_read_all_entries(wt_jp))
        j_entries = _dedup_entries(all_jentries)

        # _dedup_entries guarantees one entry per session_id (richest wins)
        journal_by_session = {}
        for e in j_entries:
            if e.session_id:
                journal_by_session[e.session_id] = e

        lines: list[str] = []
        for arc in arcs:
            # Header
            topic = arc.get("topic", "unknown")
            ds = arc["date_start"][:10] if arc["date_start"] else "?"
            de = arc["date_end"][:10] if arc["date_end"] else "?"
            n_sessions = len(arc.get("session_ids", []))
            header = f"=== {topic} ({ds} -- {de}, {n_sessions} session(s)) ==="
            lines.append(header)

            # Tags
            tags = arc.get("tags", [])
            if tags:
                lines.append(f"Tags: {', '.join(tags)}")

            # Session details
            for sid in arc.get("session_ids", []):
                entry = journal_by_session.get(sid)
                if entry:
                    date = entry.timestamp[:10] if entry.timestamp else "?"
                    raw_focus = strip_system_artifacts(entry.focus) if entry.focus else ""
                    focus = raw_focus[:80] if raw_focus else "(no focus)"
                    lines.append(f"  {date} {sid[:8]}  {focus}")
                else:
                    lines.append(f"  {sid[:8]}  (no journal entry)")
            lines.append("")

        return "\n".join(lines).rstrip()
    except Exception as exc:
        return f"Timeline query failed: {exc}"


def recall_channel(
    action: str = "read",
    channel: str = "dev",
    message: str | None = None,
    to: str | None = None,
    target: str | None = None,
    limit: int = 20,
    pin: bool = False,
) -> str:
    """Cross-worktree communication channels for multi-agent coordination.

    Channels are append-only JSONL files in the shared .synapt/recall/ directory.
    Any agent (worktree) can post and read messages. No daemon needed.
    State (presence, cursors, pins) is stored in SQLite.

    Args:
        action: "join", "leave", "post", "read", "who", "heartbeat", "unread",
                "pin", "directive", "mute", "unmute", "kick", "broadcast",
                "list", "search".
        channel: Channel name (default "dev"). Any name works -- created on first post.
        message: Message body (required for "post", "directive", "broadcast" actions).
        to: Target agent for "directive" action.
        target: Agent to mute/unmute/kick (agent_id, display name, or griptree name).
        limit: Max messages to return for "read" action (default 20).
        pin: If True with "post" action, also pin the message.
    """
    try:
        from synapt.recall.channel import (
            channel_join,
            channel_leave,
            channel_post,
            channel_read,
            channel_who,
            channel_heartbeat,
            channel_unread,
            channel_pin,
            channel_directive,
            channel_mute,
            channel_unmute,
            channel_kick,
            channel_broadcast,
            channel_list_channels,
        )

        if action == "join":
            return channel_join(channel=channel)

        if action == "leave":
            return channel_leave(channel=channel)

        if action == "post":
            if not message:
                return "Error: message is required for 'post' action."
            return channel_post(channel=channel, message=message, pin=pin)

        if action == "read":
            return channel_read(channel=channel, limit=limit)

        if action == "who":
            return channel_who()

        if action == "heartbeat":
            return channel_heartbeat()

        if action == "unread":
            counts = channel_unread()
            if not counts:
                return "No channel memberships -- join a channel first."
            lines = ["## Unread messages"]
            for ch, count in sorted(counts.items()):
                marker = f" ({count} new)" if count > 0 else ""
                lines.append(f"  #{ch}: {count}{marker}")
            return "\n".join(lines)

        if action == "pin":
            if not message:
                return "Error: message_id is required for 'pin' action."
            return channel_pin(channel=channel, message_id=message)

        if action == "directive":
            if not message:
                return "Error: message is required for 'directive' action."
            if not to:
                return "Error: 'to' (target agent) is required for 'directive' action."
            return channel_directive(channel=channel, message=message, to=to)

        if action == "mute":
            if not target:
                return "Error: target agent is required for 'mute' action."
            return channel_mute(target=target, channel=channel)

        if action == "unmute":
            if not target:
                return "Error: target agent is required for 'unmute' action."
            return channel_unmute(target=target, channel=channel)

        if action == "kick":
            if not target:
                return "Error: target agent is required for 'kick' action."
            return channel_kick(target=target, channel=channel)

        if action == "broadcast":
            if not message:
                return "Error: message is required for 'broadcast' action."
            return channel_broadcast(message=message)

        if action == "list":
            channels = channel_list_channels()
            if not channels:
                return "No channels yet."
            return "Channels: " + ", ".join(f"#{c}" for c in channels)

        if action == "search":
            if not message:
                return "Error: query is required for 'search' action."
            from synapt.recall.channel import channel_search
            results = channel_search(message)
            if not results:
                return "No matching channel messages."
            lines = ["## Channel search results"]
            for r in results:
                ts = r["timestamp"][:16]
                lines.append(f"  [{r['message_id']}] #{r['channel']} {ts}  {r['from']}: {r['body']}")
            return "\n".join(lines)

        if action == "rename":
            if not message:
                return "Error: message is required for 'rename' action (the new display name)."
            from synapt.recall.channel import channel_rename
            return channel_rename(new_name=message)

        return (
            f"Unknown action: {action}. Use 'join', 'leave', 'post', 'read', "
            f"'who', 'heartbeat', 'unread', 'pin', 'directive', 'mute', "
            f"'unmute', 'kick', 'broadcast', 'list', 'search', or 'rename'."
        )
    except Exception as exc:
        return f"Channel failed: {exc}"


# ---------------------------------------------------------------------------
# MCP registration
# ---------------------------------------------------------------------------


def _directive_suffix() -> str:
    """Check for pending directives and return a suffix to append to tool results.

    Returns empty string if nothing pending. Runs in ~20ms (no subprocess).
    """
    try:
        from synapt.recall.channel import check_directives
        return check_directives()
    except Exception:
        return ""


def _with_directive_check(fn):
    """Wrap a tool function to append pending directives to its result."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        suffix = _directive_suffix()
        if suffix and isinstance(result, str):
            return result + "\n\n" + suffix
        return result

    return wrapper


def register_tools(mcp) -> None:
    """Register recall tools on the given FastMCP server instance.

    This allows the unified synapt server to compose recall tools alongside
    repair and watch tools on a single MCP server.
    """
    mcp.tool()(_with_directive_check(recall_search))
    mcp.tool()(_with_directive_check(recall_quick))
    mcp.tool()(_with_directive_check(recall_files))
    mcp.tool()(_with_directive_check(recall_sessions))
    mcp.tool()(recall_build)
    mcp.tool()(recall_setup)
    mcp.tool()(_with_directive_check(recall_stats))
    mcp.tool()(_with_directive_check(recall_journal))
    mcp.tool()(_with_directive_check(recall_remind))
    mcp.tool()(recall_enrich)
    mcp.tool()(recall_consolidate)
    mcp.tool()(_with_directive_check(recall_contradict))
    mcp.tool()(_with_directive_check(recall_context))
    mcp.tool()(_with_directive_check(recall_timeline))
    mcp.tool()(_with_directive_check(recall_channel))


def main():
    """Entry point for standalone synapt-recall-server."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        "synapt-recall",
        instructions=MCP_INSTRUCTIONS,
    )
    register_tools(server)
    server.run()


if __name__ == "__main__":
    main()
