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

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

from synapt.recall.sharding import is_sharded, list_shards
from synapt.recall.storage import RecallDB


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
        """
        if is_sharded(index_dir):
            index_db = RecallDB(index_dir / "index.db")
            data_dbs = [RecallDB(p) for p in list_shards(index_dir)]
            return cls(index_db, data_dbs)

        # Monolithic: single recall.db serves as both index and data
        db = RecallDB(index_dir / "recall.db")
        return cls(db, [])

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

    def save_clusters(self, clusters: list[dict], memberships: list[tuple]) -> None:
        self._index.save_clusters(clusters, memberships)

    def save_cluster_summary(self, cluster_id: str, summary: str, **kwargs) -> None:
        self._index.save_cluster_summary(cluster_id, summary, **kwargs)

    # -- Chunk methods (data shards or monolithic) -------------------------

    def load_chunks(self) -> list["TranscriptChunk"]:  # noqa: F821
        """Load chunks from all data shards, or from monolithic DB."""
        if self._data_dbs:
            all_chunks = []
            for db in self._data_dbs:
                all_chunks.extend(db.load_chunks())
            return all_chunks
        return self._index.load_chunks()

    def load_chunk_headers(self) -> list["TranscriptChunk"]:  # noqa: F821
        """Load lightweight chunk metadata from all shards."""
        if self._data_dbs:
            all_chunks = []
            for db in self._data_dbs:
                all_chunks.extend(db.load_chunk_headers())
            return all_chunks
        return self._index.load_chunk_headers()

    def load_chunk_by_rowid(self, rowid: int):  # noqa: ANN201
        """Load one chunk by shard-qualified rowid."""
        if self._data_dbs:
            shard_idx, local_rowid = self._decode_chunk_rowid(rowid)
            if shard_idx <= 0 or shard_idx > len(self._data_dbs):
                return None
            return self._data_dbs[shard_idx - 1].load_chunk_by_rowid(local_rowid)
        return self._index.load_chunk_by_rowid(rowid)

    def load_chunks_by_rowids(self, rowids: list[int]):  # noqa: ANN201
        """Load multiple chunks keyed by shard-qualified rowids."""
        if self._data_dbs:
            loaded = {}
            for shard_idx, local_rowids in self._group_encoded_rowids(rowids).items():
                if shard_idx <= 0 or shard_idx > len(self._data_dbs):
                    continue
                partial = self._data_dbs[shard_idx - 1].load_chunks_by_rowids(local_rowids)
                loaded.update({
                    self._encode_chunk_rowid(shard_idx, local_rowid): chunk
                    for local_rowid, chunk in partial.items()
                })
            return loaded
        return self._index.load_chunks_by_rowids(rowids)

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
        if self._data_dbs:
            return sum(db.chunk_count() for db in self._data_dbs)
        return self._index.chunk_count()

    def chunk_session_map(self) -> dict[int, str]:
        """Return a global ``{rowid: session_id}`` mapping for all chunks."""
        if self._data_dbs:
            result: dict[int, str] = {}
            for shard_idx, db in self._iter_data_shards():
                for rowid, session_id in db.chunk_session_map().items():
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = session_id
            return result
        return self._index.chunk_session_map()

    def chunk_id_map(self) -> dict[int, str]:
        """Return a global ``{rowid: chunk_id}`` mapping for all chunks."""
        if self._data_dbs:
            result: dict[int, str] = {}
            for shard_idx, db in self._iter_data_shards():
                for rowid, chunk_id in db.chunk_id_map().items():
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = chunk_id
            return result
        return self._index.chunk_id_map()

    def get_chunk_id_rowid_map(self) -> dict[str, int]:
        """Return ``{chunk_id: global_rowid}`` across all shards."""
        if self._data_dbs:
            result: dict[str, int] = {}
            for shard_idx, db in self._iter_data_shards():
                for chunk_id, rowid in db.get_chunk_id_rowid_map().items():
                    result[chunk_id] = self._encode_chunk_rowid(shard_idx, rowid)
            return result
        return self._index.get_chunk_id_rowid_map()

    def save_chunks(self, chunks: list["TranscriptChunk"]) -> None:  # noqa: F821
        """Save chunks to the appropriate database.

        In monolithic mode, delegates directly to the single DB.
        In sharded mode, clears ALL shards first (to prevent duplication on
        full rebuilds), then saves to the active (last) shard. If the active
        shard exceeds SHARD_CHUNK_THRESHOLD, a new shard is created.
        """
        if not self._data_dbs:
            self._index.save_chunks(chunks)
            return

        from synapt.recall.sharding import (
            SHARD_CHUNK_THRESHOLD,
            SHARD_METADATA_SQL,
            _update_shard_metadata,
            shard_name_for_index,
            next_shard_index,
        )

        # Clear ALL non-active shards before writing. RecallDB.save_chunks()
        # only wipes the single DB it's called on, so without this step a
        # full rebuild leaves stale duplicates in every earlier shard.
        for db in self._data_dbs[:-1]:
            try:
                db._conn.execute("DROP TRIGGER IF EXISTS chunks_ad")
                db._conn.execute("DELETE FROM chunks")
                db._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('delete-all')")
                db._conn.execute(
                    "CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN "
                    "  INSERT INTO chunks_fts(chunks_fts, rowid, user_text, assistant_text, "
                    "    tools_used, files_touched, tool_content, date_text) "
                    "  VALUES ('delete', old.rowid, old.user_text, old.assistant_text, "
                    "    old.tools_used, old.files_touched, old.tool_content, old.date_text); "
                    "END;"
                )
                db._conn.commit()
            except Exception:
                logger.warning("Failed to clear shard %s", db._path, exc_info=True)

        # Save to the last (active) shard (this also clears it internally)
        active_db = self._data_dbs[-1]
        active_db.save_chunks(chunks)

        # Check if active shard exceeded threshold — rotate to a new shard
        try:
            count = active_db._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if count > SHARD_CHUNK_THRESHOLD:
                index_dir = self._index._path.parent
                idx = next_shard_index(index_dir)
                new_path = index_dir / shard_name_for_index(idx)
                new_db = RecallDB(new_path)
                self._data_dbs.append(new_db)

                # Update shard metadata: old shard is now inactive, new is active
                idx_conn = sqlite3.connect(str(self._index._path))
                try:
                    idx_conn.execute(SHARD_METADATA_SQL)
                    # Get timestamps from the now-full shard
                    ts_row = active_db._conn.execute(
                        "SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM chunks"
                    ).fetchone()
                    _update_shard_metadata(
                        idx_conn, active_db._path, count,
                        ts_row[0] or "", ts_row[1] or "",
                        is_active=False,
                    )
                    _update_shard_metadata(
                        idx_conn, new_path, 0, "", "",
                        is_active=True,
                    )
                    idx_conn.commit()
                finally:
                    idx_conn.close()
        except Exception:
            logger.warning("Shard rotation check failed", exc_info=True)

    def fts_search(self, query: str, limit: int = 100, **kwargs) -> list[tuple]:
        """FTS search across all shards, merging results by score.
        """
        if self._data_dbs:
            all_results = []
            for shard_idx, db in self._iter_data_shards():
                shard_hits = db.fts_search(query, limit=limit, **kwargs)
                all_results.extend(
                    (self._encode_chunk_rowid(shard_idx, rowid), score)
                    for rowid, score in shard_hits
                )
            # Sort by score descending (score is element [1])
            all_results.sort(key=lambda r: r[1], reverse=True)
            return all_results[:limit]
        return self._index.fts_search(query, limit=limit, **kwargs)

    def fts_search_raw(self, fts_query: str, limit: int = 100) -> list[tuple[int, float]]:
        """Execute a pre-built FTS query across all shards."""
        if self._data_dbs:
            all_results = []
            for shard_idx, db in self._iter_data_shards():
                shard_hits = db.fts_search_raw(fts_query, limit=limit)
                all_results.extend(
                    (self._encode_chunk_rowid(shard_idx, rowid), score)
                    for rowid, score in shard_hits
                )
            all_results.sort(key=lambda r: r[1], reverse=True)
            return all_results[:limit]
        return self._index.fts_search_raw(fts_query, limit=limit)

    def fts_search_by_session(
        self,
        query: str,
        session_ids: list[str],
        limit: int = 100,
    ) -> list[tuple[int, float]]:
        """Session-scoped FTS search across all data shards."""
        if self._data_dbs:
            all_results = []
            for shard_idx, db in self._iter_data_shards():
                shard_hits = db.fts_search_by_session(query, session_ids, limit=limit)
                all_results.extend(
                    (self._encode_chunk_rowid(shard_idx, rowid), score)
                    for rowid, score in shard_hits
                )
            all_results.sort(key=lambda r: r[1], reverse=True)
            return all_results[:limit]
        return self._index.fts_search_by_session(query, session_ids, limit=limit)

    def get_embeddings(self, rowids: list[int]) -> dict[int, list[float]]:
        """Fetch chunk embeddings keyed by global sharded rowids."""
        if self._data_dbs:
            result: dict[int, list[float]] = {}
            for shard_idx, local_rowids in self._group_encoded_rowids(rowids).items():
                db = self._data_dbs[shard_idx - 1]
                for rowid, emb in db.get_embeddings(local_rowids).items():
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = emb
            return result
        return self._index.get_embeddings(rowids)

    def save_embeddings(self, embeddings: dict[int, list[float]]) -> None:
        """Store chunk embeddings keyed by global sharded rowids."""
        if self._data_dbs:
            grouped: dict[int, dict[int, list[float]]] = {}
            for rowid, emb in embeddings.items():
                shard_idx, local_rowid = self._decode_chunk_rowid(rowid)
                grouped.setdefault(shard_idx, {})[local_rowid] = emb
            for shard_idx, shard_embs in grouped.items():
                self._data_dbs[shard_idx - 1].save_embeddings(shard_embs)
            return
        self._index.save_embeddings(embeddings)

    def has_embeddings(self) -> bool:
        """True if any chunk-bearing database has stored embeddings."""
        if self._data_dbs:
            return any(db.has_embeddings() for db in self._data_dbs)
        return self._index.has_embeddings()

    def get_all_embeddings(self) -> dict[int, list[float]]:
        """Load all chunk embeddings keyed by global sharded rowids."""
        if self._data_dbs:
            result: dict[int, list[float]] = {}
            for shard_idx, db in self._iter_data_shards():
                for rowid, emb in db.get_all_embeddings().items():
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = emb
            return result
        return self._index.get_all_embeddings()

    # -- Access tracking (always index DB) ---------------------------------

    def record_access(self, *args, **kwargs) -> None:
        self._index.record_access(*args, **kwargs)

    # -- Passthrough for any method not explicitly wrapped ------------------

    def __getattr__(self, name: str):
        """Fall through to index DB for any unhandled method."""
        return getattr(self._index, name)

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
        """True if using a single recall.db (no shards)."""
        return not self._data_dbs
