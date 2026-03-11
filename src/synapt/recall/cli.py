#!/usr/bin/env python3
"""CLI for synapt: build, search, and inspect per-project session indexes.

Usage:
    # One-time setup: build index, register MCP, install hook
    synapt setup

    # Setup with HuggingFace sync
    synapt setup --sync hf:user/dataset-name

    # Build index for current project
    synapt build

    # Build with explicit source
    synapt build --source ~/.claude/projects/-Users-me-Development-myproject

    # Build from ChatGPT export
    synapt build --chatgpt-archive ~/Downloads/chatgpt-export.zip

    # Build from HuggingFace
    HF_TOKEN=... synapt build --hf user/dataset-name

    # Search current project's index
    synapt search "quality curve" --max-chunks 5

    # Progressive search (most recent sessions first)
    synapt search "harness bug" --max-sessions 3

    # Search with date filtering
    synapt search "what errors" --after 2026-02-28 --before 2026-03-01

    # Index stats
    synapt stats

    # Sync transcripts to/from HuggingFace
    synapt sync push
    synapt sync pull
    synapt sync both

    # Install global hooks (SessionStart, SessionEnd, PreCompact)
    synapt install-hook

    # Hook-triggered rebuild (called by PreCompact hook)
    synapt rebuild
    synapt rebuild --sync
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("synapt.recall.cli")

from datetime import datetime

from synapt.recall.core import (
    atomic_json_write,
    format_size,
    TranscriptIndex,
    build_index,
    project_data_dir,
    project_index_dir,
    project_archive_dir,
    project_slug,
    project_transcript_dir,
    project_transcript_dirs,
    all_worktree_archive_dirs,
    _is_real_user_message,
    _extract_user_text,
    _extract_assistant_content,
)
from synapt.recall.chatgpt import parse_chatgpt_archive
from synapt.recall.journal import (
    latest_transcript_path,
    extract_session_id,
    _journal_path,
    _read_all_session_ids,
    auto_extract_entry,
    append_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_index_dir(args: argparse.Namespace) -> Path:
    """Resolve the index directory from args or cwd."""
    if getattr(args, "index", None):
        return Path(args.index).expanduser()
    if getattr(args, "out", None):
        return Path(args.out).expanduser()
    return project_index_dir()


def _check_legacy_index() -> Path | None:
    """Check for old ~/.synapse-recall/<slug>/ index location."""
    slug = project_slug()
    legacy = Path.home() / ".synapse-recall" / slug
    if legacy.exists() and (
        (legacy / "recall.db").exists() or (legacy / "chunks.jsonl").exists()
    ):
        return legacy
    return None


def _ensure_gitignore(project_dir: Path) -> None:
    """Add .synapt/ to .gitignore if not already present."""
    gitignore_path = project_dir / ".gitignore"
    new_entry = ".synapt/"
    old_entries = [".synapse-recall/", ".synapse/"]

    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        # Migrate any old entries
        changed = False
        for old_entry in old_entries:
            if old_entry in lines and new_entry not in lines:
                lines = [new_entry if l == old_entry else l for l in lines]
                changed = True
        if changed:
            gitignore_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
        if new_entry in lines:
            return
        if not content.endswith("\n"):
            content += "\n"
        content += f"{new_entry}\n"
        gitignore_path.write_text(content, encoding="utf-8")
    else:
        gitignore_path.write_text(f"{new_entry}\n", encoding="utf-8")


def _acquire_build_lock(data_dir: Path, timeout: float = 60.0) -> "int | None":
    """Acquire an exclusive file lock for index builds.

    Returns the lock file descriptor on success, None if the lock could not
    be acquired within *timeout* seconds (another build is running).
    """
    import errno
    import fcntl
    import time

    lock_path = data_dir / "build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError as exc:
            # Only retry on lock contention; other errors are fatal
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                os.close(fd)
                return None
            if time.monotonic() >= deadline:
                os.close(fd)
                return None
            time.sleep(0.5)


def _release_build_lock(fd: int) -> None:
    """Release the build file lock."""
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _archive_and_build(
    project_dir: Path,
    source_dirs: list[Path] | None = None,
    use_embeddings: bool = True,
    incremental: bool = False,
    chatgpt_archive: str | None = None,
) -> TranscriptIndex | None:
    """Archive transcripts and build the index. Shared by build/rebuild/setup.

    Acquires an exclusive file lock so only one process builds at a time.
    Other worktrees' builds will wait up to 60s for the lock.

    1. Archive transcripts from Claude Code source dir -> .synapt/recall/transcripts/
    2. Build index from archive (not directly from ~/.claude/)

    Returns the final TranscriptIndex, or None if no chunks found.
    """
    data_dir = project_data_dir(project_dir)
    lock_fd = _acquire_build_lock(data_dir)
    if lock_fd is None:
        print("  Warning: another build is in progress (timed out waiting for lock)")
        return None

    try:
        return _archive_and_build_locked(
            project_dir, source_dirs, use_embeddings, incremental, chatgpt_archive,
        )
    finally:
        _release_build_lock(lock_fd)


def _archive_and_build_locked(
    project_dir: Path,
    source_dirs: list[Path] | None,
    use_embeddings: bool,
    incremental: bool,
    chatgpt_archive: str | None,
) -> TranscriptIndex | None:
    """Inner build logic — caller must hold the build lock."""
    from synapt.recall.archive import archive_transcripts
    from synapt.recall.storage import RecallDB

    index_dir = project_index_dir(project_dir)
    archive_dir = project_archive_dir(project_dir)

    # Step 1: Archive transcripts from Claude Code's source dir
    if not source_dirs:
        source_dirs = project_transcript_dirs(project_dir)

    if source_dirs:
        for src in source_dirs:
            copied = archive_transcripts(project_dir, src)
            if copied:
                print(f"  Archived {len(copied)} new transcript(s) from {src}")

    # Step 2: Determine what to build from — aggregate all worktree archives
    build_sources: list[Path] = []
    for wt_archive in all_worktree_archive_dirs(project_dir):
        build_sources.append(wt_archive)
    # Include this worktree's archive if not already found by worktree discovery
    if archive_dir.exists() and any(archive_dir.glob("*.jsonl")):
        resolved = archive_dir.resolve()
        if resolved not in {p.resolve() for p in build_sources}:
            build_sources.append(archive_dir)

    # Fall back to direct source if no archives exist yet (first run)
    if not build_sources and source_dirs:
        build_sources = [source_dirs[0]]

    # Step 3: Open or create SQLite database
    index_dir.mkdir(parents=True, exist_ok=True)
    db = RecallDB(index_dir / "recall.db")

    # Step 4: Load existing data for incremental builds
    incremental_manifest = None
    existing_chunks = []
    if incremental:
        manifest = db.load_manifest()
        if manifest.get("chunk_count"):
            incremental_manifest = manifest
            existing_chunks = db.load_chunks()
            print(f"  Incremental: {len(existing_chunks)} existing chunks")
        elif (index_dir / "manifest.json").exists():
            # Fallback: legacy JSON manifest
            try:
                with open(index_dir / "manifest.json", encoding="utf-8") as f:
                    incremental_manifest = json.load(f)
                existing_index = TranscriptIndex.load(index_dir)
                existing_chunks = existing_index.chunks
                print(f"  Incremental: {len(existing_chunks)} existing chunks")
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  Warning: corrupt manifest, falling back to full rebuild ({exc})")
                incremental_manifest = None
                existing_chunks = []

    # Strip old journal chunks — they'll be re-parsed fresh below
    all_chunks = [c for c in existing_chunks if c.turn_index >= 0]

    # Build from archived transcripts (BM25-only, no DB needed for parsing)
    for build_source in build_sources:
        index = build_index(
            build_source,
            use_embeddings=False,
            incremental_manifest=incremental_manifest,
        )
        all_chunks.extend(index.chunks)

    # ChatGPT archive (separate source)
    if chatgpt_archive:
        archive_path = Path(chatgpt_archive).expanduser()
        print(f"\n[build] Parsing ChatGPT archive {archive_path} ...")
        chatgpt_chunks = parse_chatgpt_archive(archive_path)
        print(f"[build] Parsed {len(chatgpt_chunks)} ChatGPT chunks")
        all_chunks.extend(chatgpt_chunks)

    # Tier 1: Synthesize auto-journal stubs for sessions without entries
    from synapt.recall.journal import _journal_path, synthesize_journal_stubs
    from synapt.recall.core import parse_journal_entries

    # Collect journal files from all worktrees for the shared index
    journal_files: list[Path] = []
    local_journal = _journal_path(project_dir)
    journal_files.append(local_journal)
    # Also include journals from other worktrees
    for wt_archive in all_worktree_archive_dirs(project_dir):
        # Archive dir is <main>/.synapt/recall/worktrees/<name>/transcripts/
        # Journal is at <main>/.synapt/recall/worktrees/<name>/journal.jsonl
        wt_journal = wt_archive.parent / "journal.jsonl"
        if wt_journal.resolve() != local_journal.resolve() and wt_journal.exists():
            journal_files.append(wt_journal)

    transcript_chunks = [c for c in all_chunks if c.turn_index >= 0]
    if transcript_chunks:
        sessions: dict[str, list] = {}
        for c in transcript_chunks:
            sessions.setdefault(c.session_id, []).append(c)
        # Only synthesize stubs into the local journal
        synthesized = synthesize_journal_stubs(sessions, local_journal, project_root=str(project_dir))
        if synthesized:
            print(f"  Auto-journal: {synthesized} stub(s) synthesized")

    # Journal entries → searchable chunks from ALL worktrees
    for journal_file in journal_files:
        if journal_file.exists():
            journal_chunks = parse_journal_entries(journal_file)
            if journal_chunks:
                print(f"  Journal: {len(journal_chunks)} entries from {journal_file.parent.name}")
                all_chunks.extend(journal_chunks)

    if not all_chunks:
        return None

    # Dedup by chunk id
    deduped = []
    seen_ids = set()
    for chunk in all_chunks:
        if chunk.id not in seen_ids:
            seen_ids.add(chunk.id)
            deduped.append(chunk)

    # Build final index with SQLite backend
    final_index = TranscriptIndex(
        deduped,
        use_embeddings=use_embeddings,
        cache_dir=index_dir,
        db=db,
    )
    final_index.save(index_dir)

    # Cluster chunks by topic similarity
    from synapt.recall.clustering import cluster_chunks as _cluster_chunks, generate_concat_summary
    transcript_only = [c for c in deduped if c.turn_index >= 0]
    if transcript_only:
        clusters = _cluster_chunks(transcript_only)
        if clusters:
            # Build chunk ID → TranscriptChunk lookup for summary generation
            chunk_map = {c.id: c for c in transcript_only}
            memberships = []
            for cl in clusters:
                for cid in cl["chunk_ids"]:
                    memberships.append((cl["cluster_id"], cid, cl["created_at"]))
            # Enrich each cluster with search_text from member chunk content.
            # Include user_text, tools, and files so concise-mode search
            # can find clusters by what users asked, not just assistant answers.
            for cl in clusters:
                member_chunks = [chunk_map[cid] for cid in cl["chunk_ids"] if cid in chunk_map]
                texts: list[str] = []
                for c in member_chunks:
                    if c.assistant_text:
                        texts.append(c.assistant_text)
                    if c.user_text:
                        texts.append(c.user_text)
                    if c.tools_used:
                        texts.append(" ".join(c.tools_used))
                    if c.files_touched:
                        texts.append(" ".join(c.files_touched))
                joined = " ".join(texts)
                if len(joined) > 4000:
                    joined = joined[:4000].rsplit(" ", 1)[0]
                cl["search_text"] = joined

            db.save_clusters(clusters, memberships)
            # Pre-generate concat summaries at build time (read path stays pure).
            # Skip clusters that already have LLM summaries (preserved across rebuilds).
            llm_cluster_ids = {
                r["cluster_id"]
                for r in db._conn.execute(
                    "SELECT cluster_id FROM cluster_summaries WHERE method = 'llm'"
                ).fetchall()
            }
            for cl in clusters:
                if cl["cluster_id"] in llm_cluster_ids:
                    continue  # Already has LLM summary
                member_chunks = [chunk_map[cid] for cid in cl["chunk_ids"] if cid in chunk_map]
                if member_chunks:
                    summary = generate_concat_summary(member_chunks, max_tokens=200)
                    if summary:
                        db.save_cluster_summary(cl["cluster_id"], summary)
            print(f"  Clusters: {len(clusters)} topic clusters from {sum(c['chunk_count'] for c in clusters)} chunks")
        else:
            print("  Clusters: none (chunks may not be related enough)")

    # Auto-tag clusters + build timeline arcs (Phase 10)
    try:
        from synapt.recall.journal import (
            _read_all_entries, _journal_path, _dedup_entries,
        )
        from synapt.recall.tagging import extract_tags as _extract_tags
        from synapt.recall.timeline import (
            build_timeline_clusters,
            save_timeline_clusters,
        )

        # Read journal entries from ALL worktrees (not just local) and
        # dedup by session_id, preferring enriched over auto stubs.
        all_journal_entries: list = []
        for jf in journal_files:
            if jf.exists():
                all_journal_entries.extend(_read_all_entries(jf))
        j_entries = _dedup_entries(all_journal_entries)

        # Tag topic clusters with issue refs, branches, keywords
        if transcript_only:
            tagged = 0
            all_clusters = db.load_clusters()
            for cl in all_clusters:
                if cl["cluster_type"] != "topic":
                    continue
                tags = _extract_tags(cl, j_entries)
                if tags:
                    cl["tags"] = tags
                    new_search = cl.get("search_text", "") + " " + " ".join(tags)
                    db._conn.execute(
                        "UPDATE clusters SET tags = ?, search_text = ? "
                        "WHERE cluster_id = ?",
                        (json.dumps(tags), new_search, cl["cluster_id"]),
                    )
                    tagged += 1
            if tagged:
                db._conn.commit()
                # FTS is kept in sync by the clusters_au trigger on UPDATE;
                # save_timeline_clusters() also does a full FTS rebuild.
                print(f"  Tags: {tagged} topic clusters tagged")

        # Build timeline arcs from session grouping
        timeline = build_timeline_clusters(db, j_entries)
        if timeline:
            save_timeline_clusters(db, timeline)
            print(f"  Timeline: {len(timeline)} arcs")
    except Exception as exc:
        logger.warning("Phase 10 (tagging/timeline) failed: %s", exc, exc_info=True)

    # Process pending promotions (advance tiers based on access stats)
    try:
        from synapt.recall.promotion import process_build_promotions
        promo = process_build_promotions(db)
        promo_total = sum(promo.values())
        if promo_total:
            print(f"  Promotions: {promo['summaries_upgraded']} summaries, "
                  f"{promo['candidates_flagged']} candidates, "
                  f"{promo['knowledge_promoted']} knowledge")
    except Exception:
        pass  # Never fail a build due to promotions

    # Upgrade large clusters to LLM summaries (size-based, not access-based)
    try:
        from synapt.recall.clustering import upgrade_large_cluster_summaries
        llm_upgraded = upgrade_large_cluster_summaries(db, min_chunks=5, max_upgrades=5)
        if llm_upgraded:
            print(f"  LLM summaries: {llm_upgraded} clusters upgraded")
    except Exception:
        pass  # Never fail a build due to LLM summaries

    # Maintain adaptive memory: decay, archival, log compaction
    try:
        decayed = db.recompute_decay_scores()
        archived = db.archive_cold_clusters()
        compacted = db.compact_access_log()
        parts = []
        if archived:
            parts.append(f"{len(archived)} clusters archived")
        if compacted:
            parts.append(f"{compacted} log entries compacted")
        if parts:
            print(f"  Memory maintenance: {', '.join(parts)}")
    except Exception as exc:
        logger.debug("Memory maintenance failed: %s", exc)

    # Compact + dedup knowledge nodes
    try:
        from synapt.recall.knowledge import (
            dedup_knowledge_nodes, compact_knowledge, _knowledge_path,
        )
        # Always compact first — removes same-ID duplicates from append-only JSONL
        kn_path = _knowledge_path(project_dir)
        if kn_path.exists():
            compacted = compact_knowledge(kn_path)
            if compacted:
                print(f"  Knowledge compact: removed {compacted} stale version(s)")
        # Then merge semantically similar nodes (different IDs, same content)
        merged = dedup_knowledge_nodes(threshold=0.7, project_dir=project_dir)
        if merged:
            print(f"  Knowledge dedup: merged {merged} duplicate(s)")
    except Exception as exc:
        logger.debug("Knowledge dedup failed: %s", exc)

    # Store source file info in DB metadata
    source_files = []
    for build_source in build_sources:
        for fp in sorted(build_source.glob("*.jsonl")):
            st = fp.stat()
            source_files.append({
                "name": fp.name,
                "mtime": st.st_mtime,
                "size": st.st_size,
            })
    db.save_manifest({"source_files": source_files})

    return final_index


def discover_transcript_dirs() -> list[Path]:
    """Find all Claude Code project transcript directories.

    Scans ~/.claude/projects/*/ for directories containing .jsonl files.
    Returns directories sorted alphabetically.
    """
    base = Path.home() / ".claude" / "projects"
    if not base.exists():
        return []
    dirs = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and any(d.glob("*.jsonl")):
            dirs.append(d)
    return dirs


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> None:
    """Build a transcript index from local files, HuggingFace, or ChatGPT."""
    project = Path.cwd().resolve()
    use_emb = not args.no_embeddings

    # Check for legacy index
    legacy = _check_legacy_index()
    if legacy:
        print(f"[build] Note: legacy index found at {legacy}")
        print(f"[build] New location: {project_index_dir()}")
        print()

    source_dirs: list[Path] = []
    if args.source:
        for src in args.source:
            source_dirs.append(Path(src).expanduser())

    if args.hf:
        hf_dir = _download_hf_transcripts(args.hf)
        if hf_dir:
            source_dirs.append(hf_dir)

    if not source_dirs and not args.chatgpt_archive:
        auto_dirs = project_transcript_dirs()
        if auto_dirs:
            source_dirs = auto_dirs
            for td in auto_dirs:
                print(f"[build] Found project transcripts at {td}")
        elif not all_worktree_archive_dirs(project):
            # No live transcripts AND no archived transcripts — nothing to build
            print("Error: no transcripts found for current project.", file=sys.stderr)
            print("Specify --source, --hf, or --chatgpt-archive explicitly.", file=sys.stderr)
            sys.exit(1)

    if use_emb:
        print("[build] Computing embeddings (this may take a moment) ...")

    final_index = _archive_and_build(
        project,
        source_dirs=source_dirs or None,
        use_embeddings=use_emb,
        incremental=args.incremental,
        chatgpt_archive=args.chatgpt_archive,
    )

    if not final_index:
        print("Error: no chunks found.", file=sys.stderr)
        sys.exit(1)

    stats = final_index.stats()
    print(f"\n[build] Done!")
    print(f"  Chunks: {stats['chunk_count']}")
    print(f"  Sessions: {stats['session_count']}")
    if stats.get("date_range"):
        print(f"  Date range: {stats['date_range']['earliest'][:10]} -> {stats['date_range']['latest'][:10]}")
    print(f"  Saved to: {project_index_dir()}")


def cmd_search(args: argparse.Namespace) -> None:
    """Search the transcript index."""
    index_dir = _resolve_index_dir(args)
    if not (index_dir / "recall.db").exists() and not (index_dir / "chunks.jsonl").exists():
        print(f"Error: no index found at {index_dir}", file=sys.stderr)
        print("Run 'synapt build' or 'synapt setup' first.", file=sys.stderr)
        sys.exit(1)

    index = TranscriptIndex.load(index_dir, use_embeddings=True)
    result = index.lookup(
        args.query,
        max_chunks=args.max_chunks,
        max_tokens=args.max_tokens,
        max_sessions=args.max_sessions,
        after=args.after,
        before=args.before,
    )

    if result:
        print(result)
    else:
        print("No results found.")


def cmd_stats(args: argparse.Namespace) -> None:
    """Show index statistics."""
    index_dir = _resolve_index_dir(args)
    if not (index_dir / "recall.db").exists() and not (index_dir / "manifest.json").exists():
        print(f"Error: no index found at {index_dir}", file=sys.stderr)
        sys.exit(1)

    index = TranscriptIndex.load(index_dir, use_embeddings=False)
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

    print("Transcript Index Stats")
    print("=" * 40)
    print(f"  Project:          {Path.cwd()}")
    print(f"  Index:            {index_dir}")
    print(f"  Chunks:           {stats.get('chunk_count', 0)}")
    print(f"  Sessions:         {stats.get('session_count', 0)}")
    print(f"  Avg chunks/sess:  {stats.get('avg_chunks_per_session', 0):.1f}")
    if stats.get("date_range"):
        dr = stats["date_range"]
        print(f"  Date range:       {dr['earliest'][:10]} -> {dr['latest'][:10]}")
    print(f"  Unique tools:     {stats.get('total_tools_used', 0)}")
    print(f"  Unique files:     {stats.get('total_files_touched', 0)}")
    print(f"  Built:            {manifest.get('build_timestamp', 'unknown')[:19]}")

    total_size = sum(fp.stat().st_size for fp in index_dir.iterdir() if fp.is_file())
    print(f"  Index size:       {format_size(total_size)}")

    # Show archive stats
    archive_dir = project_archive_dir()
    if archive_dir.exists():
        archive_files = list(archive_dir.glob("*.jsonl"))
        archive_size = sum(f.stat().st_size for f in archive_files)
        print(f"  Archived:         {len(archive_files)} transcripts ({format_size(archive_size)})")

    source_files = manifest.get("source_files", [])
    if source_files:
        total_source = sum(sf.get("size", 0) for sf in source_files)
        print(f"  Source files:     {len(source_files)} ({format_size(total_source)})")

    # Cluster stats
    if index._db:
        n_clusters = index._db.cluster_count()
        if n_clusters > 0:
            print(f"  Clusters:         {n_clusters}")

    # Active model configuration
    try:
        from synapt.recall.config import load_config
        cfg = load_config()
        models = cfg.active_models()
        print()
        print("Active Models")
        print("-" * 40)
        for key, model in models.items():
            print(f"  {key:16s}  {model}")
        if cfg.backend != "auto":
            print(f"  {'backend':16s}  {cfg.backend}")
    except Exception as e:
        logger.debug("Failed to load model config: %s", e)


def cmd_sessions(args: argparse.Namespace) -> None:
    """List recent sessions with date, turn count, and first message."""
    index_dir = _resolve_index_dir(args)
    if not (index_dir / "recall.db").exists() and not (index_dir / "chunks.jsonl").exists():
        print(f"Error: no index found at {index_dir}", file=sys.stderr)
        print("Run 'synapt build' or 'synapt setup' first.", file=sys.stderr)
        sys.exit(1)

    index = TranscriptIndex.load(index_dir, use_embeddings=False)
    sessions = index.list_sessions(
        max_sessions=args.max_sessions,
        after=args.after,
        before=args.before,
    )

    if not sessions:
        print("No sessions found.")
        return

    print(f"Recent sessions ({len(sessions)}):")
    for s in sessions:
        print(
            f"  {s['date']}  {s['session_id'][:8]}  "
            f"{s['turn_count']} turns  {s['files_count']} files  "
            f"\"{s['first_message']}\""
        )


def cmd_rebuild(args: argparse.Namespace) -> None:
    """Incremental rebuild triggered by hooks. Auto-discovers current project."""
    project = Path.cwd().resolve()

    if not project_transcript_dirs(project):
        return

    final_index = _archive_and_build(
        project,
        use_embeddings=False,
        incremental=True,
    )

    if final_index:
        stats = final_index.stats()
        print(f"synapt: rebuilt index ({stats['chunk_count']} chunks)", file=sys.stderr)

    # Optional sync after rebuild
    if getattr(args, "sync", False):
        _sync_after_rebuild(project)

    # Optional enrichment of auto-stubs
    enrich_n = getattr(args, "enrich", 0)
    if enrich_n and final_index:
        try:
            from synapt.recall.enrich import enrich_all, _MLX_AVAILABLE
            if _MLX_AVAILABLE:
                count = enrich_all(
                    project_dir=project,
                    max_entries=enrich_n,
                )
                if count:
                    print(f"  Enriched {count} journal stub(s)", file=sys.stderr)
        except Exception:
            pass  # Enrichment is best-effort; don't break the hook


def _sync_after_rebuild(project: Path) -> None:
    """Push new transcripts to HF if sync is configured and debounce allows."""
    from synapt.recall.archive import load_sync_config, upload_to_hf, should_sync

    config = load_sync_config(project)
    sync = config.get("sync", {})
    if sync.get("provider") == "hf" and sync.get("auto_sync") and sync.get("repo_id"):
        if not should_sync(project):
            return
        extra = sync.get("extra_files", [])
        uploaded = upload_to_hf(project, sync["repo_id"], extra_files=extra)
        if uploaded:
            print(f"synapt: synced {uploaded} file(s) to HF", file=sys.stderr)


def cmd_archive(args: argparse.Namespace) -> None:
    """Archive transcripts locally without indexing."""
    from synapt.recall.archive import archive_transcripts

    project = Path.cwd().resolve()
    transcript_all = project_transcript_dirs(project)
    if not transcript_all:
        print("No transcript directory found for this project.", file=sys.stderr)
        sys.exit(1)

    total_copied: list[str] = []
    for transcript_dir in transcript_all:
        copied = archive_transcripts(project, transcript_dir)
        if copied:
            total_copied.extend(copied)
            print(f"Archived {len(copied)} transcript(s) from {transcript_dir}", file=sys.stderr)
    if not total_copied:
        print("All transcripts already archived.", file=sys.stderr)


def cmd_transcript(args: argparse.Namespace) -> None:
    """Display or save a session transcript."""
    from synapt.recall.archive import archive_transcripts

    project = Path.cwd().resolve()
    transcript_dir = project_transcript_dir(project)

    if not transcript_dir:
        print("No transcript directory found for this project.", file=sys.stderr)
        sys.exit(1)

    # Find the transcript file
    if args.session_id:
        target = transcript_dir / f"{args.session_id}.jsonl"
        if not target.exists():
            # Also check archive
            archive = project_archive_dir(project)
            target = archive / f"{args.session_id}.jsonl"
        if not target.exists():
            print(f"Session not found: {args.session_id}", file=sys.stderr)
            # List available sessions
            _list_available_sessions(transcript_dir, project)
            sys.exit(1)
    else:
        # Find current/most recent session by mtime
        jsonl_files = sorted(transcript_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not jsonl_files:
            print("No transcript files found.", file=sys.stderr)
            sys.exit(1)
        target = jsonl_files[0]

    # --save: archive it locally
    if args.save:
        copied = archive_transcripts(project, transcript_dir)
        if copied:
            print(f"Archived {len(copied)} transcript(s) to .synapt/recall/transcripts/", file=sys.stderr)
        else:
            print("Already archived.", file=sys.stderr)
        return

    # --list: show available sessions
    if args.list:
        _list_available_sessions(transcript_dir, project)
        return

    # Display the transcript
    session_id = target.stem
    size = target.stat().st_size
    print(f"Session: {session_id}")
    print(f"File: {target}")
    print(f"Size: {format_size(size)}")
    print("=" * 60)

    turn_count = 0
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if _is_real_user_message(entry):
                user_text = _extract_user_text(entry)
                if not user_text or "<command-name>" in user_text:
                    continue
                turn_count += 1
                ts = entry.get("timestamp", "")
                header = f"--- Turn {turn_count} ---"
                if ts:
                    header = f"--- Turn {turn_count} [{ts[:19]}] ---"
                print(f"\n{header}")
                print(f"User: {user_text[:2000]}")

            elif entry.get("type") == "assistant":
                text, tools, _files = _extract_assistant_content(entry)
                if text or tools:
                    if text:
                        print(f"Assistant: {text[:2000]}")
                    if tools:
                        print(f"  [Tools: {', '.join(tools)}]")

    print(f"\n{'=' * 60}")
    print(f"Total turns: {turn_count}")


def _list_available_sessions(transcript_dir: Path, project: Path) -> None:
    """List available transcript sessions."""
    archive = project_archive_dir(project)

    # Gather all unique session IDs from both source and archive
    sessions: dict[str, tuple[float, int, str]] = {}  # id -> (mtime, size, location)

    for d, label in [(transcript_dir, "live"), (archive, "archived")]:
        if not d.exists():
            continue
        for f in d.glob("*.jsonl"):
            sid = f.stem
            st = f.stat()
            if sid not in sessions or st.st_mtime > sessions[sid][0]:
                sessions[sid] = (st.st_mtime, st.st_size, label)

    if not sessions:
        print("No sessions found.")
        return

    print(f"{'Session ID':<40} {'Size':>10} {'Location':>10}  Modified")
    print("-" * 90)
    for sid, (mtime, size, loc) in sorted(sessions.items(), key=lambda x: x[1][0], reverse=True):
        dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        print(f"{sid:<40} {format_size(size):>10} {loc:>10}  {dt}")


def cmd_journal(args: argparse.Namespace) -> None:
    """Display or write session journal entries."""
    from synapt.recall.journal import (
        append_entry,
        auto_extract_entry,
        format_entry_full,
        format_for_session_start,
        latest_transcript_path,
        read_entries,
        read_latest,
    )

    if args.read:
        entry = read_latest(meaningful=True)
        if not entry:
            return  # Silent — no meaningful journal yet (hook context)
        text = format_for_session_start(entry)
        if text:
            print(text)
        return

    if args.list:
        n = args.show if args.show else 5
        entries = read_entries(n=n)
        if not entries:
            print("No journal entries yet.")
            return
        for i, entry in enumerate(entries):
            if i > 0:
                print("\n---\n")
            print(format_entry_full(entry))
        return

    if args.show:
        if args.show < 1:
            print("--show requires a positive integer.", file=sys.stderr)
            sys.exit(1)
        entries = read_entries(n=args.show)
        if not entries:
            print("No journal entries yet.")
            return
        if args.show > len(entries):
            print(f"Only {len(entries)} journal entries exist.", file=sys.stderr)
        idx = min(args.show - 1, len(entries) - 1)
        print(format_entry_full(entries[idx]))
        return

    if not args.write:
        print("Usage: synapt recall journal [--read | --write | --list | --show N]", file=sys.stderr)
        print("  --write is required to create a journal entry.", file=sys.stderr)
        sys.exit(1)

    # --write: auto-extract + merge CLI args
    project = Path.cwd().resolve()
    transcript_path = latest_transcript_path(project)
    entry = auto_extract_entry(transcript_path=transcript_path, cwd=str(project))

    # Merge CLI-provided fields
    if args.focus:
        entry.focus = args.focus
    if args.done:
        entry.done = [d.strip() for d in args.done.split(";")]
    if args.decisions:
        entry.decisions = [d.strip() for d in args.decisions.split(";")]
    if args.next:
        entry.next_steps = [n.strip() for n in args.next.split(";")]

    # Clear auto flag if user provided rich content
    if entry.has_rich_content():
        entry.auto = False

    # Persist auto-stubs even without rich content — enrich fills them in later.
    # Session-start display already filters to has_rich_content() entries.
    if entry.auto and not entry.has_rich_content():
        if not entry.has_content():
            print("No content to journal (no files modified, no fields provided).", file=sys.stderr)
            return
        path = append_entry(entry)
        sid = entry.session_id[:8] if entry.session_id else "unknown"
        print(f"Auto-stub saved for enrichment ({sid})", file=sys.stderr)
        return
    # Skip completely empty entries (no files, no fields)
    if not entry.has_content():
        print("No content to journal (no files modified, no fields provided).", file=sys.stderr)
        return

    path = append_entry(entry)
    print(f"Journal entry written to {path}", file=sys.stderr)
    print(format_entry_full(entry))


def cmd_enrich(args: argparse.Namespace) -> None:
    """Enrich auto-generated journal stubs using a local MLX model."""
    project = Path.cwd().resolve()
    model = args.model

    init_from = getattr(args, "init_from", None)
    if init_from:
        from synapt.recall.enrich import enrich_transcript_segments
        transcript_path = Path(init_from).expanduser().resolve()
        if not transcript_path.exists():
            print(f"[enrich] Transcript not found: {transcript_path}", file=sys.stderr)
            return
        gap = getattr(args, "gap_minutes", 60)
        print(f"[init] Segmenting transcript: {transcript_path.name}")
        print(f"[init] Gap threshold: {gap} minutes")
        count = enrich_transcript_segments(
            transcript_path=transcript_path,
            project_dir=project,
            model=model,
            dry_run=args.dry_run,
            max_entries=args.max_entries,
            adapter_path=getattr(args, "adapter_path", ""),
            gap_minutes=gap,
        )
        if count:
            action = "would be enriched" if args.dry_run else "enriched"
            print(f"\n[init] Done! {count} segments {action}.")
        else:
            print("[init] No segments to enrich.")
        return

    from synapt.recall.enrich import enrich_all
    print(f"[enrich] Enriching auto-journal stubs with {model} ...")

    count = enrich_all(
        project_dir=project,
        model=model,
        dry_run=args.dry_run,
        max_entries=args.max_entries,
        adapter_path=getattr(args, "adapter_path", ""),
    )

    if count:
        action = "would be enriched" if args.dry_run else "enriched"
        print(f"\n[enrich] Done! {count} entries {action}.")
    else:
        print("[enrich] No entries to enrich (all sessions already have journal entries).")


def cmd_consolidate(args: argparse.Namespace) -> None:
    """Extract durable knowledge from journal entries (memory consolidation)."""
    if getattr(args, "show", False):
        from synapt.recall.knowledge import read_nodes, format_knowledge_for_display
        nodes = read_nodes()
        if not nodes:
            print("No knowledge nodes yet. Run `synapt recall consolidate` to extract knowledge from journal entries.")
            return
        active = [n for n in nodes if n.status == "active"]
        other = [n for n in nodes if n.status != "active"]
        print(f"Knowledge nodes: {len(active)} active, {len(other)} inactive\n")
        if active:
            print("Active:")
            print(format_knowledge_for_display(active))
        if other:
            print("\nInactive:")
            print(format_knowledge_for_display(other))
        return

    from synapt.recall.consolidate import consolidate

    project = Path.cwd().resolve()
    model = args.model

    print(f"[consolidate] Analyzing journal entries with {model} ...")

    result = consolidate(
        project_dir=project,
        model=model,
        dry_run=args.dry_run,
        force=args.force,
        min_entries=args.min_entries,
        adapter_path=getattr(args, "adapter_path", ""),
    )

    if args.dry_run:
        if result.clusters_found:
            print(f"\n[consolidate] Dry run: {result.entries_processed} entries, "
                  f"{result.clusters_found} clusters found.")
        else:
            print(f"\n[consolidate] Dry run: {result.entries_processed} entries, "
                  f"no clusters found (entries may not be related enough).")
        return

    parts = []
    if result.nodes_created:
        parts.append(f"{result.nodes_created} created")
    if result.nodes_corroborated:
        parts.append(f"{result.nodes_corroborated} corroborated")
    if result.nodes_contradicted:
        parts.append(f"{result.nodes_contradicted} contradicted")

    if parts:
        print(f"\n[consolidate] Done! Knowledge nodes: {', '.join(parts)}.")
    else:
        print(f"\n[consolidate] No knowledge extracted from {result.entries_processed} entries.")


def cmd_remind(args: argparse.Namespace) -> None:
    """Manage session reminders."""
    from synapt.recall.reminders import (
        add_reminder,
        clear_reminder,
        load_reminders,
        pop_pending,
        format_for_session_start,
    )

    if args.pending:
        pending = pop_pending()  # Single load-save cycle
        if not pending:
            return  # Silent — no reminders (hook context)
        print(format_for_session_start(pending))
        return

    if args.list:
        reminders = load_reminders()
        if not reminders:
            print("No reminders.")
            return
        for r in reminders:
            sticky = " [sticky]" if r.sticky else ""
            shown = f" (shown {r.shown_count}x)" if r.shown_count > 0 else ""
            print(f"  {r.id}  {r.text}{sticky}{shown}")
        return

    if args.clear is not None:
        # --clear with optional ID (empty string means clear all)
        rid = args.clear if args.clear else None
        count = clear_reminder(rid)
        if count:
            print(f"Cleared {count} reminder(s).")
        else:
            print("No reminders to clear.")
        return

    # Default: add a reminder
    if not args.text:
        print("Usage: synapt recall remind \"text to remember\"", file=sys.stderr)
        sys.exit(1)

    reminder = add_reminder(args.text, sticky=args.sticky)
    sticky_label = " (sticky)" if args.sticky else ""
    print(f"Added reminder{sticky_label}: {reminder.text} (id: {reminder.id})")


_GLOBAL_HOOKS = {
    "SessionStart": "synapt recall hook session-start",
    "SessionEnd": "synapt recall hook session-end",
    "PreCompact": "synapt recall hook precompact",
}


def _install_global_hooks() -> int:
    """Register synapt hooks in ~/.claude/settings.json.

    Returns number of hooks newly installed (0 if all already present).
    """
    settings_path = Path.home() / ".claude" / "settings.json"
    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    hooks = settings.setdefault("hooks", {})
    installed = 0
    migrated = 0

    # Migrate: remove old "synapse recall hook ..." entries
    _OLD_HOOK_PREFIX = "synapse recall hook "
    for event in list(hooks.keys()):
        matchers = hooks.get(event, [])
        for m in matchers:
            if not isinstance(m, dict):
                continue
            inner = m.get("hooks", [])
            filtered = [
                h for h in inner
                if not (isinstance(h, dict) and
                        h.get("command", "").startswith(_OLD_HOOK_PREFIX))
            ]
            if len(filtered) < len(inner):
                migrated += len(inner) - len(filtered)
            m["hooks"] = filtered

    for event, command in _GLOBAL_HOOKS.items():
        matchers = hooks.setdefault(event, [])
        # Check if our command is already registered
        already = any(
            isinstance(m, dict)
            and any(
                isinstance(h, dict) and h.get("command") == command
                for h in m.get("hooks", [])
            )
            for m in matchers
        )
        if already:
            continue

        # Find or create a catch-all matcher
        target = None
        for m in matchers:
            if isinstance(m, dict) and not m.get("matcher"):
                target = m
                break
        entry = {"type": "command", "command": command, "timeout": 60}
        if target is None:
            matchers.append({"matcher": "", "hooks": [entry]})
        else:
            target.setdefault("hooks", []).append(entry)
        installed += 1

    if installed or migrated:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(settings, settings_path)

    return installed


def _catchup_archive_and_journal(project: Path, transcript_dir: Path) -> None:
    """Ensure all transcripts are archived and journaled.

    Called at session-start to catch up after /clear or crash where
    session-end didn't fire. Archives new/grown transcripts and writes
    journal entries for any sessions not already journaled.

    This is separate from _archive_and_build's synthesize_journal_stubs
    because it uses auto_extract_entry (which reads the actual transcript
    file for richer extraction) rather than synthesizing from index chunks.
    """
    from synapt.recall.archive import archive_transcripts
    from synapt.recall.journal import (
        _journal_path,
        _read_all_session_ids,
        append_entry,
        auto_extract_entry,
    )

    # Archive transcripts (copies from Claude Code's dir to .synapt/recall/transcripts/)
    copied = archive_transcripts(project, transcript_dir)
    if copied:
        print(f"  Catch-up: archived {len(copied)} transcript(s)", file=sys.stderr)

    # Journal from the archive (not source) — the archive has the full pre-/clear
    # content, while the source may be truncated after /clear.
    archive_dir = project_archive_dir(project)
    journal_path = _journal_path(project)
    existing_ids = _read_all_session_ids(journal_path)
    journaled = 0

    # Prefer archive files; fall back to source for anything not yet archived
    journal_files: dict[str, Path] = {}
    for f in sorted(transcript_dir.glob("*.jsonl")):
        journal_files[f.name] = f
    if archive_dir.is_dir():
        for f in sorted(archive_dir.glob("*.jsonl")):
            journal_files[f.name] = f  # Archive overrides source

    for src_file in sorted(journal_files.values(), key=lambda p: p.name):
        # Extract session_id from this transcript
        session_id = extract_session_id(src_file)
        if not session_id or session_id in existing_ids:
            continue

        # Write a journal entry for this un-journaled session
        entry = auto_extract_entry(transcript_path=str(src_file), cwd=str(project))
        if entry.has_content():
            append_entry(entry, journal_path)
            existing_ids.add(session_id)  # Prevent duplicates within this loop
            journaled += 1
            print(f"  Catch-up: journaled session {session_id[:8]}", file=sys.stderr)

    if journaled:
        print(f"  Catch-up: wrote {journaled} journal entry(ies)", file=sys.stderr)


def cmd_hook(args: argparse.Namespace) -> None:
    """Versioned hook handler — replaces shell scripts.

    Called directly from Claude Code hooks config:
        "command": "synapt recall hook session-start"
    """
    import subprocess

    # Drain stdin (hook protocol sends JSON on stdin)
    try:
        sys.stdin.read()
    except Exception:
        pass

    # Opt-out check
    if (project_data_dir() / "no-auto-capture").exists():
        return

    event = args.event

    if event == "session-start":
        project = Path.cwd().resolve()
        transcript_all = project_transcript_dirs(project)

        # 0. Catch up: archive + journal for any un-processed transcripts.
        #    Handles /clear (where session-end may not have fired) and
        #    crash recovery. Only writes a journal entry if the latest
        #    transcript's session isn't already journaled.
        for transcript_dir in transcript_all:
            _catchup_archive_and_journal(project, transcript_dir)

        # 1. Incremental rebuild (also synthesizes journal stubs)
        if transcript_all:
            final_index = _archive_and_build(project, use_embeddings=False, incremental=True)
            if final_index:
                stats = final_index.stats()
                print(f"synapt: rebuilt index ({stats['chunk_count']} chunks)", file=sys.stderr)

        # 2. Compact journal (dedup + sort) before surfacing context
        from synapt.recall.journal import compact_journal
        removed = compact_journal()
        if removed:
            print(f"  Journal: compacted ({removed} duplicate(s) removed)", file=sys.stderr)

        # 3. Enrich one auto-stub in the background (non-blocking)
        subprocess.Popen(
            [sys.executable, "-m", "synapt.recall.cli", "enrich", "--max-entries", "1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 4. Surface journal context (stdout → system-reminder)
        cmd_journal(argparse.Namespace(read=True, write=False, list=False, show=None,
                                       focus=None, done=None, decisions=None, next=None))

        # 5. Surface knowledge nodes (if any exist)
        try:
            from synapt.recall.knowledge import read_nodes, format_knowledge_for_session_start
            kn_text = format_knowledge_for_session_start(read_nodes())
            if kn_text:
                print(kn_text)
        except Exception:
            pass  # Knowledge surfacing is non-critical

        # 6. Surface pending reminders
        cmd_remind(argparse.Namespace(text=None, sticky=False, list=False,
                                      clear=None, pending=True))

        # 7. Surface pending contradictions (model asks user to resolve)
        try:
            from synapt.recall.server import format_contradictions_for_session_start
            contradictions_text = format_contradictions_for_session_start()
            if contradictions_text:
                print(contradictions_text)
        except Exception:
            pass  # Contradiction surfacing is non-critical

    elif event == "session-end":
        # 1. Archive transcripts locally
        cmd_archive(argparse.Namespace())

        # 2. Write auto-extracted journal entry
        cmd_journal(argparse.Namespace(read=False, write=True, list=False, show=None,
                                       focus=None, done=None, decisions=None, next=None))

    elif event == "precompact":
        # Rebuild with sync
        project = Path.cwd().resolve()
        if project_transcript_dirs(project):
            final_index = _archive_and_build(project, use_embeddings=False, incremental=True)
            if final_index:
                stats = final_index.stats()
                print(f"synapt: rebuilt index ({stats['chunk_count']} chunks)", file=sys.stderr)
            _sync_after_rebuild(project)
            # Write an interim journal entry so mid-session state is captured
            # even when SessionEnd never fires (crash, kill, etc.).
            _precompact_journal_write(project)


def _precompact_journal_write(project: Path) -> None:
    """Write an auto-extracted journal entry during PreCompact, if not already journaled.

    Unlike the recall_journal MCP tool, this intentionally writes file-list-only
    stubs (entries with only files_modified, no focus/done/decisions) — a file
    list is better than nothing for crash-recovery purposes when context compacts
    mid-session before the user writes a rich entry.

    Reduces the chance of duplicate entries — the dedup check prevents the
    most common case, but concurrent hook invocations (e.g. SessionEnd
    racing with PreCompact) can still produce duplicates that
    compact_journal will clean up on the next build.
    """
    transcript_path = latest_transcript_path(project)
    if not transcript_path:
        return

    session_id = extract_session_id(transcript_path)
    if not session_id:
        return

    journal_file = _journal_path(project)
    if session_id in _read_all_session_ids(journal_file):
        logger.debug("PreCompact journal skip — session %s already journaled", session_id[:8])
        return

    try:
        entry = auto_extract_entry(transcript_path=transcript_path, cwd=str(project))
        if entry and entry.has_content():
            # Re-check immediately before writing: auto_extract_entry runs git
            # subprocesses that can take several seconds, creating a wide TOCTOU
            # window where a concurrent SessionEnd hook may have written the entry.
            if session_id in _read_all_session_ids(journal_file):
                logger.debug("PreCompact journal skip (post-extract) — session %s already journaled",
                             session_id[:8])
                return
            append_entry(entry, journal_file)
            print(f"  Journal: interim entry written for session {session_id[:8]}",
                  file=sys.stderr)
    except Exception as exc:
        logger.warning("PreCompact journal write failed: %s", exc, exc_info=True)


def cmd_install_hook(args: argparse.Namespace) -> None:
    """Install global hooks (SessionStart, SessionEnd, PreCompact)."""
    installed = _install_global_hooks()
    if installed:
        print(f"Installed {installed} hook(s) in ~/.claude/settings.json")
    else:
        print("All hooks already registered in ~/.claude/settings.json")
    print("\nThe synapt index will auto-rebuild on context compaction.")


def cmd_setup(args: argparse.Namespace) -> None:
    """One-command setup: build index, register MCP server, install hook."""
    from synapt.recall.archive import (
        load_sync_config,
        save_sync_config,
        download_from_hf,
    )

    project = Path.cwd().resolve()
    print(f"[setup] Project: {project}")
    print()

    # Warn about legacy index
    legacy = _check_legacy_index()
    if legacy:
        print(f"[setup] Note: legacy index found at {legacy}")
        print("[setup] In-project indexes are now used. You can remove the old index with:")
        print(f"  rm -rf {legacy}")
        print()

    total_steps = 4 if not args.no_hook else 3
    step = 1

    # --- 0. Configure sync (if requested) ---
    sync_repo = None
    if args.sync:
        # Parse --sync hf:user/repo or just user/repo
        sync_arg = args.sync
        if sync_arg.startswith("hf:"):
            sync_repo = sync_arg[3:]
        else:
            sync_repo = sync_arg

        config = load_sync_config(project)
        existing_sync = config.get("sync", {})
        config["sync"] = {
            "provider": "hf",
            "repo_id": sync_repo,
            "auto_sync": True,
            "extra_files": existing_sync.get("extra_files", []),
        }
        save_sync_config(project, config)
        print(f"[setup] Sync configured: HuggingFace -> {sync_repo}")

        # Pull from HF first (new machine scenario)
        print(f"[setup] Pulling transcripts from HF ...")
        downloaded = download_from_hf(project, sync_repo)
        if downloaded:
            print(f"  Downloaded {downloaded} transcript(s) from HF")
        else:
            print(f"  No new transcripts from HF")
        print()

    # --- 1. Archive + build index ---
    print(f"[setup] Step {step}/{total_steps}: Archiving transcripts & building index ...")
    step += 1

    transcript_dir = project_transcript_dir(project)
    if transcript_dir:
        jsonl_count = len(list(transcript_dir.glob("*.jsonl")))
        print(f"  Found {jsonl_count} transcript files at {transcript_dir}")

    # Also count any pre-existing archive transcripts (e.g., from HF pull)
    archive_dir = project_archive_dir(project)
    if archive_dir.exists():
        archive_count = len(list(archive_dir.glob("*.jsonl")))
        if archive_count:
            print(f"  Found {archive_count} archived transcript(s)")

    if not transcript_dir and not (archive_dir.exists() and any(archive_dir.glob("*.jsonl"))):
        print(f"  No transcripts found for this project.", file=sys.stderr)
        print(f"  Start a Claude Code session in this project first,", file=sys.stderr)
        print(f"  or use --sync to pull from HuggingFace.", file=sys.stderr)
        sys.exit(1)

    use_emb = not args.no_embeddings
    if use_emb:
        print("  Computing embeddings ...")

    final_index = _archive_and_build(
        project,
        use_embeddings=use_emb,
    )

    if not final_index or not final_index.chunks:
        print("  No chunks parsed from transcripts.", file=sys.stderr)
        sys.exit(1)

    stats = final_index.stats()
    print(f"  Index saved: {stats['chunk_count']} chunks, {stats['session_count']} sessions")
    if stats.get("date_range"):
        dr = stats["date_range"]
        print(f"  Date range: {dr['earliest'][:10]} to {dr['latest'][:10]}")
    print()

    # --- 2. Register MCP server ---
    print(f"[setup] Step {step}/{total_steps}: Registering MCP server ...")
    step += 1
    scope = "user" if args.global_scope else "project"

    if not shutil.which("claude"):
        print("  Warning: 'claude' CLI not found in PATH. Skipping MCP registration.", file=sys.stderr)
        print("  Register manually: claude mcp add -s user -t stdio synapt synapt-server", file=sys.stderr)
    else:
        try:
            result = subprocess.run(
                ["claude", "mcp", "add", "-s", scope, "-t", "stdio",
                 "synapt", "synapt-server"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                print(f"  Registered MCP server (scope: {scope})")
            else:
                stderr = result.stderr.strip()
                if "already exists" in stderr.lower():
                    print(f"  MCP server already registered (scope: {scope})")
                else:
                    print(f"  MCP registration returned: {stderr or result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  Warning: MCP registration failed: {e}", file=sys.stderr)
    print()

    # --- 3. Install hooks ---
    if not args.no_hook:
        print(f"[setup] Step {step}/{total_steps}: Installing hooks ...")
        step += 1
        installed = _install_global_hooks()
        if installed:
            print(f"  Registered {installed} global hook(s) in ~/.claude/settings.json")
        else:
            print(f"  Global hooks already registered")
    else:
        print(f"[setup] Step {step}/{total_steps}: Skipping hooks (--no-hook)")
        step += 1
    print()

    # --- 4. Ensure .gitignore ---
    _ensure_gitignore(project)

    # --- 5. Push to HF if sync configured ---
    if sync_repo:
        from synapt.recall.archive import upload_to_hf
        print("[setup] Pushing to HF ...")
        cfg = load_sync_config(project).get("sync", {})
        extra = cfg.get("extra_files", [])
        uploaded = upload_to_hf(project, sync_repo, extra_files=extra)
        if uploaded:
            print(f"  Uploaded {uploaded} file(s)")
        else:
            print(f"  All files already synced")
        print()

    # --- Summary ---
    index_dir = project_index_dir(project)
    total_size = sum(fp.stat().st_size for fp in index_dir.iterdir() if fp.is_file())

    print("=" * 50)
    print("  synapt setup complete!")
    print(f"  Index:    {index_dir} ({format_size(total_size)})")
    print(f"  Chunks:   {stats['chunk_count']}")
    print(f"  Sessions: {stats['session_count']}")
    print(f"  MCP:      registered (scope: {scope})")
    if not args.no_hook:
        print(f"  Hook:     installed")
    if sync_repo:
        print(f"  Sync:     {sync_repo}")
    print()
    print("  Restart Claude Code to activate MCP tools.")
    print("=" * 50)


def cmd_sync(args: argparse.Namespace) -> None:
    """Sync transcripts to/from HuggingFace."""
    from synapt.recall.archive import (
        load_sync_config,
        upload_to_hf,
        download_from_hf,
    )

    project = Path.cwd().resolve()
    config = load_sync_config(project)
    sync = config.get("sync", {})

    repo_id = args.repo or sync.get("repo_id")
    if not repo_id:
        print("Error: no sync target configured.", file=sys.stderr)
        print("Run 'synapt setup --sync hf:user/repo' first,", file=sys.stderr)
        print("or specify --repo user/repo.", file=sys.stderr)
        sys.exit(1)

    direction = args.direction or "both"

    if direction in ("pull", "both"):
        print(f"[sync] Pulling from {repo_id} ...")
        downloaded = download_from_hf(project, repo_id)
        print(f"  Downloaded {downloaded} transcript(s)")

    if direction in ("push", "both"):
        print(f"[sync] Pushing to {repo_id} ...")
        extra = sync.get("extra_files", [])
        uploaded = upload_to_hf(project, repo_id, extra_files=extra)
        print(f"  Uploaded {uploaded} file(s)")


# ---------------------------------------------------------------------------
# HuggingFace download (legacy — used by build --hf)
# ---------------------------------------------------------------------------

def _download_hf_transcripts(repo_id: str) -> Path | None:
    """Download transcript files from HuggingFace into the project archive."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed. pip install synapt[hf]", file=sys.stderr)
        return None

    token = os.environ.get("HF_TOKEN")

    # Download into project archive instead of global dir
    archive_dir = project_archive_dir()
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"[hf] Listing files in {repo_id} ...")
    api = HfApi(token=token)
    try:
        repo_files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    except Exception as e:
        print(f"Error listing HF repo: {e}", file=sys.stderr)
        return None

    jsonl_files = [f for f in repo_files if f.endswith(".jsonl") and "transcripts/" in f]
    print(f"[hf] Found {len(jsonl_files)} transcript files")

    for fname in jsonl_files:
        basename = fname.split("/")[-1]
        local_path = archive_dir / basename
        if local_path.exists():
            print(f"  {basename}: already downloaded")
            continue
        print(f"  Downloading {basename} ...")
        try:
            hf_hub_download(
                repo_id, fname,
                repo_type="dataset",
                local_dir=str(archive_dir.parent),
                token=token,
            )
        except Exception as e:
            print(f"  Error: {e}")

    return archive_dir


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="synapt",
        description="Persistent conversational memory for Claude Code sessions (per-project)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Setup
    setup_parser = subparsers.add_parser("setup", help="One-command setup: build index, register MCP, install hook")
    setup_parser.add_argument("--global", dest="global_scope", action="store_true", help="Register MCP server globally (default: project-scoped)")
    setup_parser.add_argument("--no-embeddings", action="store_true", help="Skip embeddings (BM25-only, faster build)")
    setup_parser.add_argument("--no-hook", action="store_true", help="Skip global hook installation")
    setup_parser.add_argument("--sync", default=None, help="Configure HF sync (e.g., hf:user/repo or user/repo)")

    # Build
    build_parser = subparsers.add_parser("build", help="Build transcript index for current project")
    build_parser.add_argument("--source", action="append", help="Directory with .jsonl transcript files (can specify multiple). Auto-discovers if omitted.")
    build_parser.add_argument("--hf", help="HuggingFace repo ID (e.g., user/dataset-name)")
    build_parser.add_argument("--chatgpt-archive", help="Path to ChatGPT export .zip (or conversations.json)")
    build_parser.add_argument("--out", default=None, help="Output directory for index (default: per-project)")
    build_parser.add_argument("--no-embeddings", action="store_true", help="Skip embeddings (BM25-only, faster build)")
    build_parser.add_argument("--incremental", action="store_true", help="Skip already-indexed files")

    # Search
    search_parser = subparsers.add_parser("search", help="Search transcript index")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--index", default=None, help="Index directory (default: per-project)")
    search_parser.add_argument("--max-chunks", type=int, default=5, help="Max chunks to return")
    search_parser.add_argument("--max-tokens", type=int, default=500, help="Token budget")
    search_parser.add_argument("--max-sessions", type=int, default=None, help="Progressive: only search N recent sessions")
    search_parser.add_argument("--after", default=None, help="Only results after this date (ISO 8601, e.g. 2026-02-28)")
    search_parser.add_argument("--before", default=None, help="Only results before this date (ISO 8601, e.g. 2026-03-01)")

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Show index statistics")
    stats_parser.add_argument("--index", default=None, help="Index directory (default: per-project)")

    # Sessions
    sessions_parser = subparsers.add_parser("sessions", help="List recent sessions with summaries")
    sessions_parser.add_argument("--index", default=None, help="Index directory (default: per-project)")
    sessions_parser.add_argument("--max-sessions", type=int, default=20, help="Max sessions to list (default: 20)")
    sessions_parser.add_argument("--after", default=None, help="Only sessions after this date (ISO 8601)")
    sessions_parser.add_argument("--before", default=None, help="Only sessions before this date (ISO 8601)")

    # Rebuild (hook-triggered)
    rebuild_parser = subparsers.add_parser("rebuild", help="Incremental rebuild (for hooks)")
    rebuild_parser.add_argument("--out", default=None, help="Output directory (default: per-project)")
    rebuild_parser.add_argument("--sync", action="store_true", help="Push new transcripts to HF after rebuild")
    rebuild_parser.add_argument("--enrich", type=int, nargs="?", const=1, default=0,
                                metavar="N", help="Enrich up to N auto-stubs after rebuild (default: 1)")

    # Sync
    sync_parser = subparsers.add_parser("sync", help="Sync transcripts to/from HuggingFace")
    sync_parser.add_argument("direction", nargs="?", choices=["push", "pull", "both"], default="both", help="Sync direction (default: both)")
    sync_parser.add_argument("--repo", default=None, help="HuggingFace repo ID (overrides config)")

    # Archive (lightweight local copy, no indexing)
    subparsers.add_parser("archive", help="Archive transcripts locally (no indexing)")

    # Transcript (display/save a session)
    transcript_parser = subparsers.add_parser("transcript", help="Display or save a session transcript")
    transcript_parser.add_argument("session_id", nargs="?", default=None, help="Session ID (default: most recent)")
    transcript_parser.add_argument("--save", action="store_true", help="Archive transcript locally")
    transcript_parser.add_argument("--list", action="store_true", help="List available sessions")

    # Journal
    journal_parser = subparsers.add_parser("journal", help="Session journal — structured session logging")
    journal_parser.add_argument("--read", action="store_true", help="Print latest entry's next steps (for hooks)")
    journal_parser.add_argument("--write", action="store_true", help="Write a journal entry (auto-extracts context)")
    journal_parser.add_argument("--list", action="store_true", help="List recent journal entries")
    journal_parser.add_argument("--show", type=int, default=None, help="Show Nth most recent entry")
    journal_parser.add_argument("--focus", default=None, help="What this session was about")
    journal_parser.add_argument("--done", default=None, help="What got done (semicolon-separated)")
    journal_parser.add_argument("--decisions", default=None, help="Key decisions (semicolon-separated)")
    journal_parser.add_argument("--next", default=None, help="Next steps (semicolon-separated)")

    # Enrich
    enrich_parser = subparsers.add_parser("enrich", help="Enrich auto-journal stubs using MLX (local LLM)")
    enrich_parser.add_argument("--model", default="mlx-community/Ministral-3-3B-Instruct-2512-4bit",
                               help="MLX model to use (default: Ministral-3-3B-Instruct-2512-4bit)")
    enrich_parser.add_argument("--adapter-path", default="",
                               help="LoRA adapter path for enrichment")
    enrich_parser.add_argument("--dry-run", action="store_true", help="Show what would be enriched without doing it")
    enrich_parser.add_argument("--max-entries", type=int, default=0, help="Max entries to enrich (0 = unlimited)")
    enrich_parser.add_argument("--init-from", metavar="TRANSCRIPT",
                               help="Bootstrap journal from a large transcript file (segments by time gaps)")
    enrich_parser.add_argument("--gap-minutes", type=int, default=60,
                               help="Minimum gap in minutes between segments (default: 60, used with --init-from)")

    # Consolidate (memory consolidation — "sleep")
    consolidate_parser = subparsers.add_parser(
        "consolidate", aliases=["sleep"],
        help="Extract durable knowledge from journal entries",
    )
    consolidate_parser.add_argument("--model", default="mlx-community/Ministral-3-3B-Instruct-2512-4bit",
                                     help="MLX model to use")
    consolidate_parser.add_argument("--dry-run", action="store_true",
                                     help="Show what would be consolidated without doing it")
    consolidate_parser.add_argument("--force", action="store_true",
                                     help="Reprocess all entries, ignoring last consolidation timestamp")
    consolidate_parser.add_argument("--min-entries", type=int, default=3,
                                     help="Minimum enriched entries to trigger consolidation (default: 3)")
    consolidate_parser.add_argument("--show", action="store_true",
                                     help="Show existing knowledge nodes")
    consolidate_parser.add_argument("--adapter-path", default="",
                                     help="LoRA adapter path for knowledge extraction")

    # Remind
    remind_parser = subparsers.add_parser("remind", help="Manage session reminders")
    remind_parser.add_argument("text", nargs="?", default=None, help="Reminder text to add")
    remind_parser.add_argument("--sticky", action="store_true", help="Keep reminder across sessions")
    remind_parser.add_argument("--list", action="store_true", help="List all reminders")
    remind_parser.add_argument("--clear", nargs="?", const="", default=None, help="Clear reminder by ID (or all if no ID)")
    remind_parser.add_argument("--pending", action="store_true", help="Show and mark pending reminders (for hooks)")

    # Hook (versioned hook commands — called directly from Claude Code hooks config)
    hook_parser = subparsers.add_parser("hook", help="Run a Claude Code hook (session-start, session-end, precompact)")
    hook_parser.add_argument("event", choices=["session-start", "session-end", "precompact"],
                             help="Hook event to handle")

    # Install hook (legacy — kept for backward compat)
    subparsers.add_parser("install-hook", help="Install global hooks (SessionStart, SessionEnd, PreCompact)")

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "sessions":
        cmd_sessions(args)
    elif args.command == "rebuild":
        cmd_rebuild(args)
    elif args.command == "sync":
        cmd_sync(args)
    elif args.command == "archive":
        cmd_archive(args)
    elif args.command == "transcript":
        cmd_transcript(args)
    elif args.command == "journal":
        cmd_journal(args)
    elif args.command == "enrich":
        cmd_enrich(args)
    elif args.command in ("consolidate", "sleep"):
        cmd_consolidate(args)
    elif args.command == "remind":
        cmd_remind(args)
    elif args.command == "hook":
        cmd_hook(args)
    elif args.command == "install-hook":
        cmd_install_hook(args)


if __name__ == "__main__":
    main()
