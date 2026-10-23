"""ShardedRecallDB — tree-structured storage wrapper for recall.

Migration-path step 1: wraps a single RecallDB and adds shard-awareness.
Initially delegates everything to the underlying monolithic DB. Future PRs
will route chunk queries to per-quarter data shards.

Usage:
    # Drop-in replacement for RecallDB
    db = ShardedRecallDB.open(index_dir)
    db.fts_search("query")  # searches all shards
    db.close()

See issue #89 for the full design.
"""

from __future__ import annotations

import heapq
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from synapt.recall.sharding import is_sharded, list_shards
from synapt.recall.storage import RecallDB


def _timestamp_activity(value: str) -> tuple[int, str]:
    """Normalize an overlay timestamp to the ordering shape used by RecallDB."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (1, f"{parsed.timestamp():020.6f}")
    except (AttributeError, TypeError, ValueError, OverflowError):
        return (0, value or "")


# The RecallDB members a ShardedRecallDB answers from index.db through ``__getattr__``. Every one reads index tables
# only; a member whose SQL names ``chunks`` is overridden in ShardedRecallDB instead, because index.db's chunks table
# is empty or a frozen pre-split copy. Anything not listed, ``_conn`` included, is refused rather than read from the
# wrong file (tests/recall/test_sharded_getattr_refusal.py holds the list to that rule).
INDEX_ONLY_FALLTHROUGH = frozenset({
    "access_log_count", "access_summary", "archive_cold_clusters", "chunk_link_count", "classify_all_clusters",
    "clear_query_tail", "clear_query_tail_session", "cluster_count", "cluster_fts_search", "clusters_for_chunk",
    "compact_access_log", "decay_distribution", "find_summary_by_content_hash", "get_access_stats",
    "get_cached_shard_overview", "get_chunk_cluster_id", "get_chunk_cluster_map", "get_cluster",
    "get_cluster_chunks", "get_cluster_summary", "get_cross_links_batch", "get_knowledge_embeddings",
    "get_knowledge_embeddings_by_id", "get_knowledge_embeddings_by_ids", "get_knowledge_rowid",
    "get_knowledge_rowids_without_embeddings", "get_metadata", "has_chunk_links", "items_at_tier", "knowledge_count",
    "knowledge_lineage", "load_clusters", "load_query_tail_chunk_by_rowid", "load_query_tail_chunks",
    "load_query_tail_chunks_by_rowids", "load_query_tail_chunks_for_source", "load_query_tail_cursor",
    "load_query_tail_cursor_for_session", "load_query_tail_cursors", "load_query_tail_overview_rows",
    "load_timeline_clusters", "mtime", "path", "query_tail_chunk_count", "query_tail_chunk_refs",
    "query_tail_fts_search", "query_tail_fts_search_by_session", "query_tail_fts_search_raw",
    "query_tail_id_rowid_map", "query_tail_suppressed_sessions", "recompute_decay_scores", "replace_query_tail",
    "save_chunk_links", "save_knowledge_embeddings", "set_cached_shard_overview", "set_metadata",
    "update_promotion_tier",
})


class ShardedRecallDB:
    """Shard-aware wrapper around RecallDB.

    Phase 1 (this PR): delegates to a single RecallDB. The ``open()``
    factory detects whether the index directory uses the sharded layout
    (index.db + data_*.db) or the monolithic layout (recall.db) and
    opens accordingly.

    Phase 2 (follow-up): routes chunk FTS queries to per-quarter data
    shards, merges results, and keeps index.db for knowledge/clusters.

    Chunk-facing methods that rely on rowids must be explicitly overridden
    here in sharded mode. Falling through to ``index.db`` via ``__getattr__``
    is only safe for index-only surfaces (knowledge, clusters, metadata,
    access tracking).
    """

    def __init__(self, index_db: RecallDB, data_dbs: list[RecallDB] | None = None):
        self._index = index_db
        self._data_dbs = data_dbs or []

    @staticmethod
    def _encode_chunk_rowid(shard_idx: int, rowid: int) -> int:
        """Encode a shard-qualified rowid into one globally unique integer."""
        return (shard_idx << 32) | rowid

    @staticmethod
    def _decode_chunk_rowid(encoded_rowid: int) -> tuple[int, int]:
        """Decode a global rowid back into ``(shard_idx, local_rowid)``."""
        return encoded_rowid >> 32, encoded_rowid & 0xFFFFFFFF

    def _iter_data_shards(self) -> list[tuple[int, RecallDB]]:
        """Return ``(shard_idx, db)`` pairs for chunk-bearing databases."""
        if self._data_dbs:
            return list(enumerate(self._data_dbs, start=1))
        return [(0, self._index)]

    @staticmethod
    def _overlay_rowid(rowid: int) -> int:
        return -rowid

    def _overlay_ids(self) -> set[str]:
        return set(self._index.query_tail_id_rowid_map())

    def _suppressed_base_sessions(self) -> set[str]:
        return self._index.query_tail_suppressed_sessions()

    def _base_chunk_visible(self, chunk) -> bool:  # noqa: ANN001
        return chunk.session_id not in self._suppressed_base_sessions()

    def _merge_overlay_chunks(self, chunks):  # noqa: ANN001, ANN201
        overlay = self._index.load_query_tail_chunks()
        overlay_ids = {chunk.id for chunk in overlay}
        suppressed = self._suppressed_base_sessions()
        return [
            chunk
            for chunk in chunks
            if chunk.id not in overlay_ids and chunk.session_id not in suppressed
        ] + overlay

    def _group_encoded_rowids(self, rowids: list[int]) -> dict[int, list[int]]:
        """Group global rowids by shard index, preserving local rowids."""
        grouped: dict[int, list[int]] = {}
        for rowid in rowids:
            shard_idx, local_rowid = self._decode_chunk_rowid(rowid)
            grouped.setdefault(shard_idx, []).append(local_rowid)
        return grouped

    @classmethod
    def open(cls, index_dir: Path) -> ShardedRecallDB:
        """Open a sharded or monolithic recall database.

        Auto-detects layout:
          - If ``index.db`` exists → sharded (index + data shards)
          - If ``recall.db`` exists → monolithic (single DB wraps both)
          - Otherwise → creates new monolithic ``recall.db``

        In sharded mode, data shards are read from the CURRENT generation
        (``generations.current_generation_dir``) if one has ever been
        published -- a full rebuild via ``save_chunks`` publishes a new
        generation atomically, so a reader here always sees either a
        complete prior generation or a complete new one, never a partial
        rebuild in progress. An index never rebuilt since this system
        landed has no CURRENT yet; falls back to the flat legacy layout
        (shards directly under ``index_dir``) unchanged.
        """
        if is_sharded(index_dir):
            index_db = RecallDB(index_dir / "index.db")
            data_dbs = [RecallDB(p) for p in list_shards(cls._data_shard_dir(index_dir))]
            return cls(index_db, data_dbs)

        # Monolithic: single recall.db serves as both index and data
        db = RecallDB(index_dir / "recall.db")
        return cls(db, [])

    @classmethod
    def open_readonly(cls, index_dir: Path) -> ShardedRecallDB:
        """Open an existing layout without DDL, migrations, or write access.

        Follows CURRENT the same way ``open`` does -- see its docstring.
        """
        if is_sharded(index_dir):
            index_db = RecallDB.open_readonly(index_dir / "index.db")
            try:
                data_dbs = [
                    RecallDB.open_readonly(p)
                    for p in list_shards(cls._data_shard_dir(index_dir))
                ]
            except Exception:
                index_db.close()
                raise
            return cls(index_db, data_dbs)

        return cls(RecallDB.open_readonly(index_dir / "recall.db"), [])

    @staticmethod
    def _data_shard_dir(index_dir: Path) -> Path:
        """Where to glob for data_*.db: the CURRENT generation if one has
        been published, else index_dir itself (the pre-generation flat
        layout, unaffected until the first rebuild through save_chunks)."""
        from synapt.recall.generations import current_generation_dir

        gen_dir = current_generation_dir(index_dir)
        return gen_dir if gen_dir is not None else index_dir

    # -- Delegated methods (index DB) --------------------------------------

    def load_manifest(self) -> dict:
        return self._index.load_manifest()

    def save_manifest(self, manifest: dict) -> None:
        self._index.save_manifest(manifest)

    def load_knowledge_nodes(self, status: str = "active") -> list[dict]:
        return self._index.load_knowledge_nodes(status)

    def save_knowledge_nodes(self, nodes: list[dict]) -> None:
        self._index.save_knowledge_nodes(nodes)

    def upsert_knowledge_node(self, node: dict) -> None:
        self._index.upsert_knowledge_node(node)

    def knowledge_fts_search(self, query: str, limit: int = 20,
                             include_historical: bool = False) -> list[tuple]:
        return self._index.knowledge_fts_search(query, limit, include_historical)

    def knowledge_by_rowid(self, rowids: list[int]) -> dict[int, dict]:
        return self._index.knowledge_by_rowid(rowids)

    def get_knowledge_node(self, node_id: str) -> dict | None:
        return self._index.get_knowledge_node(node_id)

    def list_pending_contradictions(self) -> list[dict]:
        return self._index.list_pending_contradictions()

    def add_pending_contradiction(self, **kwargs) -> int:
        return self._index.add_pending_contradiction(**kwargs)

    def resolve_contradiction(self, contradiction_id: int, status: str = "confirmed") -> bool:
        return self._index.resolve_contradiction(contradiction_id, status)

    def has_pending_contradiction_for(self, old_node_id: str) -> bool:
        return self._index.has_pending_contradiction_for(old_node_id)

    def pending_contradiction_count(self) -> int:
        return self._index.pending_contradiction_count()

    def save_clusters(self, clusters: list[dict], memberships: list[tuple]) -> dict:
        return self._index.save_clusters(clusters, memberships)

    def append_clusters(self, clusters: list[dict], memberships: list[tuple]) -> None:
        self._index.append_clusters(clusters, memberships)

    def mark_recluster_attempted(self, chunk_ids: list[str], run_id: str) -> None:
        self._index.mark_recluster_attempted(chunk_ids, run_id)

    def get_recluster_attempted_ids(self) -> set[str]:
        return self._index.get_recluster_attempted_ids()

    def save_cluster_token_signature(
        self, cluster_id: str, tokens: list[str], updated_at: str,
    ) -> None:
        self._index.save_cluster_token_signature(cluster_id, tokens, updated_at)

    def load_cluster_token_signatures(self) -> dict[str, set[str]]:
        return self._index.load_cluster_token_signatures()

    def active_topic_clusters_missing_signature(self) -> list[str]:
        return self._index.active_topic_clusters_missing_signature()

    def cluster_ids_with_signature_oldest_first(self) -> list[str]:
        return self._index.cluster_ids_with_signature_oldest_first()

    def load_cluster_member_chunk_ids(self, cluster_ids: list[str]) -> dict[str, list[str]]:
        return self._index.load_cluster_member_chunk_ids(cluster_ids)

    def load_cluster_topics(self, cluster_ids: list[str]) -> dict[str, str]:
        return self._index.load_cluster_topics(cluster_ids)

    def merge_chunks_into_cluster(
        self, cluster_id: str, chunk_ids: list[str], appended_text: str, added_at: str,
        run_id: str | None = None,
    ) -> None:
        self._index.merge_chunks_into_cluster(
            cluster_id, chunk_ids, appended_text, added_at, run_id=run_id,
        )

    def save_cluster_summary(self, cluster_id: str, summary: str, **kwargs) -> None:
        self._index.save_cluster_summary(cluster_id, summary, **kwargs)

    # -- Chunk methods (data shards or monolithic) -------------------------

    def load_chunks(self) -> list["TranscriptChunk"]:  # noqa: F821
        """Load chunks from all data shards, or from monolithic DB."""
        if self._data_dbs:
            all_chunks = []
            for db in self._data_dbs:
                all_chunks.extend(db.load_chunks())
            return self._merge_overlay_chunks(all_chunks)
        return self._merge_overlay_chunks(self._index.load_chunks())

    def session_indexed_extent(self, session_id: str) -> dict | None:
        """Return the furthest base-only byte extent across chunk shards."""
        extents = [
            extent
            for _, db in self._iter_data_shards()
            if (extent := db.session_indexed_extent(session_id)) is not None
        ]
        if not extents:
            return None
        return max(extents, key=lambda item: item["observed_complete_offset"])

    def load_chunk_headers(self) -> list["TranscriptChunk"]:  # noqa: F821
        """Load lightweight chunk metadata from all shards."""
        if self._data_dbs:
            all_chunks = []
            for db in self._data_dbs:
                all_chunks.extend(db.load_chunk_headers())
            return self._merge_overlay_chunks(all_chunks)
        return self._merge_overlay_chunks(self._index.load_chunk_headers())

    def distinct_tools_files(self) -> tuple[set[str], set[str]]:
        """Distinct tools/files over the visible chunk population, no hydration.

        Parity with load_chunk_headers' merged view: base rows from the
        shards (or the monolithic index db), filtered IN SQL by suppressed
        sessions and by chunk ids the query-tail overlay replaces, then the
        overlay chunks' own tools/files unioned on top. Excluding in SQL
        rather than subtracting sets is load-bearing: a tool present in
        both an excluded and a kept chunk must survive, and subtraction
        would drop it.
        """
        overlay = self._index.load_query_tail_chunks()
        overlay_ids = {chunk.id for chunk in overlay}
        suppressed = self._suppressed_base_sessions()
        sources = self._data_dbs if self._data_dbs else [self._index]
        tools: set[str] = set()
        files: set[str] = set()
        for db in sources:
            t, f = db.distinct_tools_files(
                exclude_sessions=suppressed or None,
                exclude_chunk_ids=overlay_ids or None,
            )
            tools |= t
            files |= f
        for chunk in overlay:
            tools.update(chunk.tools_used)
            files.update(chunk.files_touched)
        return tools, files

    def load_chunk_by_rowid(self, rowid: int):  # noqa: ANN201
        """Load one chunk by shard-qualified rowid."""
        if rowid < 0:
            return self._index.load_query_tail_chunk_by_rowid(-rowid)
        if self._data_dbs:
            shard_idx, local_rowid = self._decode_chunk_rowid(rowid)
            if shard_idx <= 0 or shard_idx > len(self._data_dbs):
                return None
            chunk = self._data_dbs[shard_idx - 1].load_chunk_by_rowid(local_rowid)
        else:
            chunk = self._index.load_chunk_by_rowid(rowid)
        if chunk is not None and not self._base_chunk_visible(chunk):
            return None
        return chunk

    def load_chunks_by_rowids(self, rowids: list[int]):  # noqa: ANN201
        """Load multiple chunks keyed by shard-qualified rowids."""
        overlay_rowids = [-rowid for rowid in rowids if rowid < 0]
        overlay = {
            -rowid: chunk
            for rowid, chunk in self._index.load_query_tail_chunks_by_rowids(
                overlay_rowids
            ).items()
        }
        base_rowids = [rowid for rowid in rowids if rowid >= 0]
        suppressed = self._suppressed_base_sessions()
        if self._data_dbs:
            loaded = dict(overlay)
            for shard_idx, local_rowids in self._group_encoded_rowids(base_rowids).items():
                if shard_idx <= 0 or shard_idx > len(self._data_dbs):
                    continue
                partial = self._data_dbs[shard_idx - 1].load_chunks_by_rowids(local_rowids)
                loaded.update({
                    self._encode_chunk_rowid(shard_idx, local_rowid): chunk
                    for local_rowid, chunk in partial.items()
                })
            return {
                rowid: chunk
                for rowid, chunk in loaded.items()
                if rowid < 0 or chunk.session_id not in suppressed
            }
        loaded = self._index.load_chunks_by_rowids(base_rowids)
        loaded.update(overlay)
        return {
            rowid: chunk
            for rowid, chunk in loaded.items()
            if rowid < 0 or chunk.session_id not in suppressed
        }

    def _shard_overview(
        self, db: RecallDB, generation_name: str | None, schema_version: int
    ) -> dict[str, dict]:
        """One shard's ``session_overview()``, cached when a generation
        identity exists (``generation_name`` is not None); uncached
        otherwise (no legacy-flat-layout fallback, ruled 2026-09-09).
        """
        if generation_name is None:
            return db.session_overview()
        shard_name = db.path.name
        cached = self._index.get_cached_shard_overview(generation_name, shard_name, schema_version)
        if cached is not None:
            return cached
        overview = db.session_overview()
        self._index.set_cached_shard_overview(generation_name, shard_name, schema_version, overview)
        return overview

    def session_overview(self) -> dict[str, dict]:
        """Return merged session metadata across all chunk shards.

        R3.1: each shard's own ``session_overview()`` is cached, keyed on
        the CURRENT generation's name (shards are immutable once a
        generation is published — see ``generations.py``) plus a schema
        version, so a shape change to the cached dict can never be read as
        current. No fallback for a store with no generation identity yet
        (the legacy flat layout): it pays the uncached cost every call,
        same as it always has, per the 2026-09-09 ruling.
        """
        from synapt.recall.generations import read_current_generation
        from synapt.recall.storage import SHARD_OVERVIEW_CACHE_SCHEMA_VERSION

        result: dict[str, dict] = {}
        suppressed = self._suppressed_base_sessions()
        generation_name = read_current_generation(self._index.path.parent)
        for _, db in self._iter_data_shards():
            shard_overview = self._shard_overview(
                db, generation_name, SHARD_OVERVIEW_CACHE_SCHEMA_VERSION
            )
            for session_id, overview in shard_overview.items():
                if session_id in suppressed:
                    continue
                current = result.get(session_id)
                if current is None:
                    result[session_id] = dict(overview)
                    continue
                if overview["has_real_activity"] and not current["has_real_activity"]:
                    current["activity"] = overview["activity"]
                elif overview["has_real_activity"] == current["has_real_activity"]:
                    current["activity"] = max(
                        current["activity"], overview["activity"]
                    )
                current["has_real_activity"] = (
                    current["has_real_activity"] or overview["has_real_activity"]
                )
                if overview["earliest_ts"]:
                    current["earliest_ts"] = min(
                        filter(None, (current["earliest_ts"], overview["earliest_ts"]))
                    )
                if overview["latest_ts"]:
                    current["latest_ts"] = max(
                        current["latest_ts"], overview["latest_ts"]
                    )
                current["turn_count"] += overview["turn_count"]
                if not current.get("transcript_path") and overview.get("transcript_path"):
                    current["transcript_path"] = overview["transcript_path"]
                current["agent_ids"] = frozenset(current.get("agent_ids", ())) | frozenset(
                    overview.get("agent_ids", ())
                )
        for chunk in self._index.load_query_tail_overview_rows():
            current = result.setdefault(
                chunk["session_id"],
                {
                    "activity": (0, ""),
                    "earliest_ts": "",
                    "latest_ts": "",
                    "turn_count": 0,
                    "has_real_activity": False,
                    "transcript_path": chunk["transcript_path"] or "",
                    "agent_ids": frozenset(
                        [chunk["agent_id"]]
                        if chunk["turn_index"] != -1 and chunk["agent_id"] else []
                    ),
                },
            )
            current["earliest_ts"] = min(
                filter(None, (current["earliest_ts"], chunk["timestamp"]))
            ) if current["earliest_ts"] or chunk["timestamp"] else ""
            current["latest_ts"] = max(current["latest_ts"], chunk["timestamp"])
            current["turn_count"] += int(chunk["turn_index"] >= 0)
            if chunk["turn_index"] != -1:
                current["has_real_activity"] = True
                current["activity"] = max(
                    current["activity"], _timestamp_activity(chunk["timestamp"])
                )
            if chunk["turn_index"] != -1 and chunk["agent_id"]:
                current["agent_ids"] = frozenset(current.get("agent_ids", ())) | {
                    chunk["agent_id"]
                }
        return result

    def chunk_location(self, chunk_id: str):  # noqa: ANN201
        """``RecallDB.chunk_location`` read from the overlay first, then each chunk shard; never ``index.db``'s
        chunks table. A base chunk of a session the overlay suppresses is not visible."""
        row = self._index.chunk_location(chunk_id, table="query_tail_chunks")
        if row is not None:
            return row
        suppressed = self._suppressed_base_sessions()
        for _, db in self._iter_data_shards():
            row = db.chunk_location(chunk_id)
            if row is not None and row[3] not in suppressed:
                return row
        return None

    def session_turn_spans(self) -> list[dict]:
        """``RecallDB.session_turn_spans`` merged across the chunk shards, plus the query-tail overlay.

        Same visibility rule as ``session_overview``: a session the overlay suppresses is read from the overlay
        only. ``index.db``'s own chunks table is never read.
        """
        merged: dict[str, dict] = {}

        def fold(session_id: str, first_ts: str, last_ts: str, count: int) -> None:
            current = merged.get(session_id)
            if current is None:
                merged[session_id] = {
                    "session_id": session_id,
                    "first_ts": first_ts,
                    "last_ts": last_ts,
                    "chunk_count": count,
                }
                return
            current["first_ts"] = min(current["first_ts"], first_ts)
            current["last_ts"] = max(current["last_ts"], last_ts)
            current["chunk_count"] += count

        suppressed = self._suppressed_base_sessions()
        for _, db in self._iter_data_shards():
            for span in db.session_turn_spans():
                if span["session_id"] not in suppressed:
                    fold(span["session_id"], span["first_ts"], span["last_ts"], span["chunk_count"])
        for row in self._index.load_query_tail_overview_rows():
            if row["turn_index"] >= 0:
                fold(row["session_id"], row["timestamp"], row["timestamp"], 1)
        return sorted(merged.values(), key=lambda span: span["first_ts"])

    def session_activity(self) -> dict[str, tuple[int, str]]:
        """Return newest activity per session across all chunk shards."""
        return {
            session_id: overview["activity"]
            for session_id, overview in self.session_overview().items()
        }

    def load_session_chunks(self, session_id: str):  # noqa: ANN201
        """Load one session across all shards, oldest turn first."""
        return self.load_session_chunks_many(
            [session_id], include_journal=False
        ).get(session_id, [])

    def load_session_chunks_many(
        self,
        session_ids: list[str],
        include_journal: bool = True,
    ):  # noqa: ANN201
        """Load a bounded set of sessions across all shards."""
        grouped = {session_id: [] for session_id in session_ids}
        suppressed = self._suppressed_base_sessions()
        for _, db in self._iter_data_shards():
            partial = db.load_session_chunks_many(session_ids, include_journal)
            for session_id, chunks in partial.items():
                if session_id in suppressed:
                    continue
                grouped.setdefault(session_id, []).extend(chunks)
        overlay = self._index.load_query_tail_chunks(session_ids)
        overlay_ids = {chunk.id for chunk in overlay}
        for session_id, chunks in grouped.items():
            grouped[session_id] = [
                chunk for chunk in chunks if chunk.id not in overlay_ids
            ]
        for chunk in overlay:
            if chunk.session_id in grouped:
                grouped[chunk.session_id].append(chunk)
        for chunks in grouped.values():
            chunks.sort(key=lambda chunk: chunk.turn_index)
        return grouped

    def load_session_listing(self, session_ids: list[str]) -> dict[str, list[dict]]:
        """Load summary fields for a bounded set of sessions across shards."""
        grouped: dict[str, list[dict]] = {session_id: [] for session_id in session_ids}
        suppressed = self._suppressed_base_sessions()
        for _, db in self._iter_data_shards():
            partial = db.load_session_listing(session_ids)
            for session_id, rows in partial.items():
                if session_id in suppressed:
                    continue
                grouped.setdefault(session_id, []).extend(rows)
        overlay = self._index.load_query_tail_chunks(session_ids)
        overlay_ids = {chunk.id for chunk in overlay}
        for session_id, rows in grouped.items():
            grouped[session_id] = [
                row for row in rows if row.get("id") not in overlay_ids
            ]
        for chunk in overlay:
            if chunk.session_id not in grouped:
                continue
            grouped[chunk.session_id].append(
                {
                    "id": chunk.id,
                    "turn_index": chunk.turn_index,
                    "user_text": chunk.user_text,
                    "files_touched": chunk.files_touched,
                    "transcript_path": chunk.transcript_path,
                }
            )
        for rows in grouped.values():
            rows.sort(key=lambda row: row["turn_index"])
        return grouped

    def sample_chunk_texts(self, limit: int = 100) -> list[str]:
        """Return representative chunk text samples across all shards."""
        if self._data_dbs:
            if limit <= 0:
                return []
            total = self.chunk_count()
            if total == 0:
                return []
            samples: list[str] = []
            remaining = limit
            shard_info = [
                (shard_idx, db, db.chunk_count())
                for shard_idx, db in self._iter_data_shards()
            ]
            non_empty = [(shard_idx, db, count) for shard_idx, db, count in shard_info if count > 0]
            for pos, (_, db, shard_total) in enumerate(non_empty):
                target = max(1, round(limit * shard_total / total))
                non_empty_left = len(non_empty) - pos
                target = min(target, max(1, remaining - max(non_empty_left - 1, 0)))
                samples.extend(db.sample_chunk_texts(limit=target))
                remaining = max(limit - len(samples), 0)
                if remaining <= 0:
                    break
            return samples[:limit]
        return self._index.sample_chunk_texts(limit=limit)

    def chunk_count(self) -> int:
        """Number of chunks across all data shards, or monolithic DB."""
        suppressed = self._suppressed_base_sessions()
        overlay_refs = self._index.query_tail_chunk_refs()
        visible_overlay_ids = {
            chunk_id
            for chunk_id, session_id in overlay_refs
            if session_id not in suppressed
        }
        base_count = 0
        for _, db in self._iter_data_shards():
            base_count += db.chunk_count()
            base_count -= db.chunk_count_for_values("session_id", suppressed)
            base_count -= db.chunk_count_for_values("id", visible_overlay_ids)
        return base_count + self._index.query_tail_chunk_count()

    def content_hash(self) -> str:
        """Hash chunk content across all shards in global timestamp order.

        Mirrors ``RecallDB.content_hash()`` semantics so sharded and monolithic
        indexes produce the same invalidation signal for identical content.
        """
        if not self._data_dbs:
            return self._index.content_hash()

        import hashlib

        def _rows(db: RecallDB):
            return db._conn.execute(
                "SELECT timestamp, rowid, id, user_text, assistant_text, tool_content "
                "FROM chunks ORDER BY timestamp DESC, rowid DESC"
            )

        h = hashlib.sha256()
        merged = heapq.merge(
            *(_rows(db) for db in self._data_dbs),
            key=lambda row: (row[0], row[1]),
            reverse=True,
        )
        for _, _, chunk_id, user_text, assistant_text, tool_content in merged:
            h.update(
                f"{chunk_id}|{user_text or ''}|{assistant_text or ''}|{tool_content or ''}\n"
                .encode()
            )
        return h.hexdigest()[:16]

    def chunk_session_map(self) -> dict[int, str]:
        """Return a global ``{rowid: session_id}`` mapping for all chunks."""
        suppressed = self._suppressed_base_sessions()
        if self._data_dbs:
            result: dict[int, str] = {}
            for shard_idx, db in self._iter_data_shards():
                for rowid, session_id in db.chunk_session_map().items():
                    if session_id in suppressed:
                        continue
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = session_id
            return result
        return {
            rowid: session_id
            for rowid, session_id in self._index.chunk_session_map().items()
            if session_id not in suppressed
        }

    def chunk_id_map(self) -> dict[int, str]:
        """Return a global ``{rowid: chunk_id}`` mapping for all chunks."""
        suppressed = self._suppressed_base_sessions()
        if self._data_dbs:
            result: dict[int, str] = {}
            for shard_idx, db in self._iter_data_shards():
                sessions = db.chunk_session_map()
                for rowid, chunk_id in db.chunk_id_map().items():
                    if sessions.get(rowid) in suppressed:
                        continue
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = chunk_id
            return result
        sessions = self._index.chunk_session_map()
        return {
            rowid: chunk_id
            for rowid, chunk_id in self._index.chunk_id_map().items()
            if sessions.get(rowid) not in suppressed
        }

    def get_chunk_id_rowid_map(self) -> dict[str, int]:
        """Return ``{chunk_id: global_rowid}`` across all shards."""
        suppressed = self._suppressed_base_sessions()
        if self._data_dbs:
            result: dict[str, int] = {}
            for shard_idx, db in self._iter_data_shards():
                sessions = db.chunk_session_map()
                for chunk_id, rowid in db.get_chunk_id_rowid_map().items():
                    if sessions.get(rowid) in suppressed:
                        continue
                    result[chunk_id] = self._encode_chunk_rowid(shard_idx, rowid)
            result.update({
                chunk_id: self._overlay_rowid(rowid)
                for chunk_id, rowid in self._index.query_tail_id_rowid_map().items()
            })
            return result
        sessions = self._index.chunk_session_map()
        result = {
            chunk_id: rowid
            for chunk_id, rowid in self._index.get_chunk_id_rowid_map().items()
            if sessions.get(rowid) not in suppressed
        }
        result.update({
            chunk_id: self._overlay_rowid(rowid)
            for chunk_id, rowid in self._index.query_tail_id_rowid_map().items()
        })
        return result

    def save_chunks(self, chunks: list["TranscriptChunk"]) -> None:  # noqa: F821
        """Save chunks to the appropriate database.

        In monolithic mode, delegates directly to the single DB.

        In sharded mode, this used to close every shard connection,
        delete every shard file, then rebuild from scratch with no
        atomicity spanning those steps -- a reader in the window between
        delete and rebuild could see a missing or partial shard set.
        Now it builds a complete, self-contained generation of fresh
        shards under ``generations/<name>/`` and publishes it with one
        atomic pointer swap (``generations.rebuild_and_publish``): a
        reader via ``open``/``open_readonly`` always sees either the
        prior complete generation or the new complete one, never a
        rebuild in progress. This still prevents the unbounded shard
        accumulation bug the delete-and-rebuild approach was written to
        fix (#sharding-bug-sprint-13) -- SHARD_CHUNK_THRESHOLD-based
        redistribution is unchanged, just now happening in a fresh
        directory instead of in place.

        Every successful publish also garbage-collects every generation
        except the new CURRENT and the one it superseded
        (``generations.gc_old_generations``), so a rebuild every session
        start does not accumulate one full-size generation per rebuild
        forever. Steady state after GC is measured at ~2.98x one
        generation's size (2 kept generations + the flat legacy mirror);
        see the PR body for the measured multiple and the follow-up
        under consideration to cut it further.

        Known gap (disclosed, not yet done): shard_metadata bookkeeping
        is not wired to this path yet -- nothing reads it to resolve
        which shards to open, so this is cosmetic staleness, not a
        correctness gap, but it goes stale after a rebuild through this
        system until a follow-up lands.
        """
        index_dir = self._index._path.parent
        if not is_sharded(index_dir):
            self._index.save_chunks(chunks)
            return

        from synapt.recall.generations import rebuild_and_publish

        for db in self._data_dbs:
            try:
                db.close()
            except Exception:
                pass
        self._data_dbs = []

        rebuild_and_publish(index_dir, chunks)

        self._data_dbs = [
            RecallDB(p) for p in list_shards(self._data_shard_dir(index_dir))
        ]

        logger.info("save_chunks: published a new generation with %d chunks across %d shard(s)",
                     len(chunks), len(self._data_dbs))

    def base_turn_digests(self, session_id: str) -> set:
        """Content digests of this session's BASE chunks -- never the overlay.

        NOT ``load_session_chunks``: that MERGES the overlay into its result, so
        a coverage predicate built on it would match the overlay against itself,
        read as covered, and authorise a DELETE of every cursor. The base is the
        shards.
        """
        from synapt.recall.storage import query_tail_turn_digest

        digests: set = set()
        for _, db in self._iter_data_shards():
            for chunk in db.load_session_chunks(session_id):
                digests.add(
                    query_tail_turn_digest(
                        chunk.user_text, chunk.assistant_text, chunk.timestamp
                    )
                )
        return digests

    def retire_absorbed_query_tails(self) -> None:
        """Retire overlays the sharded base can prove it absorbed -- BY CONTENT.

        Coverage decides, not the source key. The key hashes ``st_ino``, so a
        compaction rewrite (the source legitimately moving on) breaks it
        permanently, and the cursor was skipped FOREVER while the base held
        every turn. Measured on the live store 2026-09-30: 52 of 58 cursors.
        """
        from synapt.recall.storage import query_tail_coverage_complete

        for cursor in self._index.load_query_tail_cursors():
            if query_tail_coverage_complete(
                self._index.load_query_tail_chunks_for_source(cursor["source_key"]),
                self.base_turn_digests(cursor["session_id"]),
                bool(cursor.get("suppresses_base")),
            ):
                self._index.clear_query_tail(cursor["source_key"])

    def fts_search(self, query: str, limit: int = 100, **kwargs) -> list[tuple]:
        """FTS search across all shards, merging results by score.
        """
        overlay_ids = self._overlay_ids()
        suppressed = self._suppressed_base_sessions()
        overlay_results = [
            (self._overlay_rowid(rowid), score)
            for rowid, score in self._index.query_tail_fts_search(query, limit).copy()
        ]
        if self._data_dbs:
            all_results = list(overlay_results)
            for shard_idx, db in self._iter_data_shards():
                shard_hits = db.fts_search(query, limit=limit, **kwargs)
                chunk_ids = db.chunk_id_map()
                sessions = db.chunk_session_map()
                all_results.extend(
                    (self._encode_chunk_rowid(shard_idx, rowid), score)
                    for rowid, score in shard_hits
                    if chunk_ids.get(rowid) not in overlay_ids
                    and sessions.get(rowid) not in suppressed
                )
            # Sort by score descending (score is element [1])
            all_results.sort(key=lambda r: r[1], reverse=True)
            return all_results[:limit]
        chunk_ids = self._index.chunk_id_map()
        sessions = self._index.chunk_session_map()
        base = [
            (rowid, score)
            for rowid, score in self._index.fts_search(query, limit=limit, **kwargs)
            if chunk_ids.get(rowid) not in overlay_ids
            and sessions.get(rowid) not in suppressed
        ]
        merged = base + overlay_results
        merged.sort(key=lambda row: row[1], reverse=True)
        return merged[:limit]

    def fts_search_raw(self, fts_query: str, limit: int = 100) -> list[tuple[int, float]]:
        """Execute a pre-built FTS query across all shards."""
        overlay_ids = self._overlay_ids()
        suppressed = self._suppressed_base_sessions()
        overlay = [
            (self._overlay_rowid(rowid), score)
            for rowid, score in self._index.query_tail_fts_search_raw(
                fts_query, limit
            )
        ]
        if self._data_dbs:
            all_results = list(overlay)
            for shard_idx, db in self._iter_data_shards():
                shard_hits = db.fts_search_raw(fts_query, limit=limit)
                chunk_ids = db.chunk_id_map()
                sessions = db.chunk_session_map()
                all_results.extend(
                    (self._encode_chunk_rowid(shard_idx, rowid), score)
                    for rowid, score in shard_hits
                    if chunk_ids.get(rowid) not in overlay_ids
                    and sessions.get(rowid) not in suppressed
                )
            all_results.sort(key=lambda r: r[1], reverse=True)
            return all_results[:limit]
        chunk_ids = self._index.chunk_id_map()
        sessions = self._index.chunk_session_map()
        base = [
            (rowid, score)
            for rowid, score in self._index.fts_search_raw(fts_query, limit=limit)
            if chunk_ids.get(rowid) not in overlay_ids
            and sessions.get(rowid) not in suppressed
        ]
        merged = base + overlay
        merged.sort(key=lambda row: row[1], reverse=True)
        return merged[:limit]

    def fts_search_by_session(
        self,
        query: str,
        session_ids: list[str],
        limit: int = 100,
    ) -> list[tuple[int, float]]:
        """Session-scoped FTS search across all data shards."""
        overlay_ids = self._overlay_ids()
        suppressed = self._suppressed_base_sessions()
        overlay = [
            (self._overlay_rowid(rowid), score)
            for rowid, score in self._index.query_tail_fts_search_by_session(
                query, session_ids, limit
            )
        ]
        if self._data_dbs:
            all_results = list(overlay)
            for shard_idx, db in self._iter_data_shards():
                shard_hits = db.fts_search_by_session(query, session_ids, limit=limit)
                chunk_ids = db.chunk_id_map()
                sessions = db.chunk_session_map()
                all_results.extend(
                    (self._encode_chunk_rowid(shard_idx, rowid), score)
                    for rowid, score in shard_hits
                    if chunk_ids.get(rowid) not in overlay_ids
                    and sessions.get(rowid) not in suppressed
                )
            all_results.sort(key=lambda r: r[1], reverse=True)
            return all_results[:limit]
        chunk_ids = self._index.chunk_id_map()
        sessions = self._index.chunk_session_map()
        base = [
            (rowid, score)
            for rowid, score in self._index.fts_search_by_session(
                query, session_ids, limit=limit
            )
            if chunk_ids.get(rowid) not in overlay_ids
            and sessions.get(rowid) not in suppressed
        ]
        merged = base + overlay
        merged.sort(key=lambda row: row[1], reverse=True)
        return merged[:limit]

    def get_embeddings(self, rowids: list[int]) -> dict[int, list[float]]:
        """Fetch chunk embeddings keyed by global sharded rowids."""
        suppressed = self._suppressed_base_sessions()
        if self._data_dbs:
            result: dict[int, list[float]] = {}
            for shard_idx, local_rowids in self._group_encoded_rowids(rowids).items():
                db = self._data_dbs[shard_idx - 1]
                sessions = db.chunk_session_map()
                for rowid, emb in db.get_embeddings(local_rowids).items():
                    if sessions.get(rowid) in suppressed:
                        continue
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = emb
            return result
        sessions = self._index.chunk_session_map()
        return {
            rowid: embedding
            for rowid, embedding in self._index.get_embeddings(rowids).items()
            if sessions.get(rowid) not in suppressed
        }

    def save_embeddings(self, embeddings: dict[int, list[float]]) -> None:
        """Store chunk embeddings keyed by global sharded rowids."""
        if self._data_dbs:
            grouped: dict[int, dict[int, list[float]]] = {}
            for rowid, emb in embeddings.items():
                shard_idx, local_rowid = self._decode_chunk_rowid(rowid)
                grouped.setdefault(shard_idx, {})[local_rowid] = emb
            for shard_idx, shard_embs in grouped.items():
                db = self._data_dbs[shard_idx - 1]
                db.save_embeddings(shard_embs)
                # The flat mirror copies main files only, so the write must leave the WAL first.
                db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            from synapt.recall.generations import resync_flat_layout

            resync_flat_layout(self._index.path.parent)
            return
        self._index.save_embeddings(embeddings)

    def has_embeddings(self) -> bool:
        """True if any chunk-bearing database has stored embeddings."""
        if self._data_dbs:
            return any(db.has_embeddings() for db in self._data_dbs)
        return self._index.has_embeddings()

    def get_all_embeddings(self) -> dict[int, list[float]]:
        """Load all chunk embeddings keyed by global sharded rowids."""
        suppressed = self._suppressed_base_sessions()
        if self._data_dbs:
            result: dict[int, list[float]] = {}
            for shard_idx, db in self._iter_data_shards():
                sessions = db.chunk_session_map()
                for rowid, emb in db.get_all_embeddings().items():
                    if sessions.get(rowid) in suppressed:
                        continue
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = emb
            return result
        sessions = self._index.chunk_session_map()
        return {
            rowid: embedding
            for rowid, embedding in self._index.get_all_embeddings().items()
            if sessions.get(rowid) not in suppressed
        }

    def get_all_embeddings_numpy(self) -> "tuple[np.ndarray, list[int]]":
        """Load chunk embeddings into a numpy matrix with global rowids."""
        import numpy as np
        suppressed = self._suppressed_base_sessions()
        if self._data_dbs:
            matrices, all_rowids = [], []
            for shard_idx, db in self._iter_data_shards():
                mat, rids = db.get_all_embeddings_numpy()
                sessions = db.chunk_session_map()
                keep = [
                    pos
                    for pos, rowid in enumerate(rids)
                    if sessions.get(rowid) not in suppressed
                ]
                if keep:
                    matrices.append(mat[keep])
                    all_rowids.extend(
                        self._encode_chunk_rowid(shard_idx, rids[pos])
                        for pos in keep
                    )
            if not matrices:
                return np.empty((0, 384), dtype=np.float32), []
            return np.vstack(matrices), all_rowids
        matrix, rowids = self._index.get_all_embeddings_numpy()
        sessions = self._index.chunk_session_map()
        keep = [
            pos
            for pos, rowid in enumerate(rowids)
            if sessions.get(rowid) not in suppressed
        ]
        if not keep:
            return np.empty((0, 384), dtype=np.float32), []
        return matrix[keep], [rowids[pos] for pos in keep]

    # -- Access tracking (always index DB) ---------------------------------

    def record_access(self, *args, **kwargs) -> None:
        self._index.record_access(*args, **kwargs)

    # -- Passthrough for any method not explicitly wrapped ------------------

    def __getattr__(self, name: str):
        """Fall through to index.db for the declared index-only members; refuse everything else."""
        if name in INDEX_ONLY_FALLTHROUGH:
            return getattr(self._index, name)
        raise AttributeError(
            f"ShardedRecallDB has no {name!r}: only the index-only members in INDEX_ONLY_FALLTHROUGH fall through to "
            "index.db, whose chunks table is not the store's. Use index_conn for index tables; read chunks through "
            "a ShardedRecallDB method."
        )

    @property
    def index_conn(self):  # noqa: ANN201
        """index.db's connection, for its INDEX tables only (see ``RecallDB.index_conn``)."""
        return self._index._conn

    def _chunk_rows_by_ids(self, chunk_ids: list[str]) -> list:
        """Rows for the given chunk ids from the overlay, then the shards; a base chunk of a session the overlay
        suppresses is not visible."""
        rows = list(self._index.chunk_rows_by_ids(chunk_ids, table="query_tail_chunks"))
        found = {row["id"] for row in rows}
        missing = [cid for cid in chunk_ids if cid not in found]
        if missing:
            suppressed = self._suppressed_base_sessions()
            for _, db in self._iter_data_shards():
                for row in db.chunk_rows_by_ids(missing):
                    if row["id"] not in found and row["session_id"] not in suppressed:
                        rows.append(row)
                        found.add(row["id"])
        return rows

    def get_cluster_chunk_texts(self, cluster_id: str) -> list[dict]:
        """``RecallDB.get_cluster_chunk_texts`` with the members' text read from the shards."""
        ids = self._index.load_cluster_member_chunk_ids([cluster_id]).get(cluster_id, [])
        rows = sorted(self._chunk_rows_by_ids(ids), key=lambda row: row["timestamp"] or "")
        return [
            {key: row[key] for key in ("user_text", "assistant_text", "tools_used", "files_touched")}
            for row in rows
        ]

    def chunk_fts_to_clusters(
        self, query: str, limit: int = 50, include_archived: bool = False,
    ) -> list[tuple[str, float]]:
        """``RecallDB.chunk_fts_to_clusters`` over the shards' FTS, mapped to clusters through index.db."""
        from synapt.recall.storage import _escape_fts_query

        fts_query = _escape_fts_query(query, use_or=True)
        if not fts_query:
            return []
        try:
            return self._chunk_fts_to_clusters(fts_query, limit, include_archived)
        except Exception:  # same contract as RecallDB.chunk_fts_to_clusters: a failed fallback finds nothing
            return []

    def _chunk_fts_to_clusters(
        self, fts_query: str, limit: int, include_archived: bool,
    ) -> list[tuple[str, float]]:
        hits = self.fts_search_raw(fts_query, limit=limit * 3)
        if not hits:
            return []
        chunks = self.load_chunks_by_rowids([rowid for rowid, _ in hits])
        score_by_chunk: dict[str, float] = {}
        for rowid, score in hits:
            chunk = chunks.get(rowid)
            if chunk is not None:
                score_by_chunk[chunk.id] = max(score, score_by_chunk.get(chunk.id, score))
        best: dict[str, float] = {}
        for chunk_id, cluster_id in self._index.clusters_for_chunk_ids(list(score_by_chunk), include_archived):
            score = score_by_chunk[chunk_id]
            if score > best.get(cluster_id, float("-inf")):
                best[cluster_id] = score
        return sorted(best.items(), key=lambda item: item[1], reverse=True)[:limit]

    def create_singleton_cluster(self, chunk_id: str, topic: str = "") -> str | None:
        """``RecallDB.create_singleton_cluster`` for a chunk found in the shards (or the overlay)."""
        rows = self._chunk_rows_by_ids([chunk_id])
        if not rows:
            return None
        return self._index.create_singleton_cluster(chunk_id, topic, chunk_user_text=rows[0]["user_text"] or "")

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close all database connections."""
        for db in self._data_dbs:
            db.close()
        self._index.close()

    @property
    def shard_count(self) -> int:
        """Number of data shards (0 for monolithic)."""
        return len(self._data_dbs)

    @property
    def is_monolithic(self) -> bool:
        """True if using a single recall.db (no shards).

        Reflects the on-disk layout (``is_sharded(index_dir)``), not merely
        whether this instance currently holds any shard connections -- a
        genuinely sharded store (``index.db`` present) with zero shards
        yet created is still sharded, not monolithic.
        """
        return not is_sharded(self._index._path.parent)
