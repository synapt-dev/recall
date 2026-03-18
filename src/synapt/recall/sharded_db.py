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

from pathlib import Path

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
    """

    def __init__(self, index_db: RecallDB, data_dbs: list[RecallDB] | None = None):
        self._index = index_db
        self._data_dbs = data_dbs or []

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

    def save_chunks(self, chunks: list["TranscriptChunk"]) -> None:  # noqa: F821
        """Save chunks to the appropriate database.

        In monolithic mode, delegates directly. In sharded mode, raises
        NotImplementedError — Phase 2 will route chunks to the correct
        quarterly shard based on timestamp.
        """
        if self._data_dbs:
            raise NotImplementedError(
                "Sharded save_chunks not yet implemented. "
                "Phase 2 will route chunks to quarterly shards by timestamp."
            )
        self._index.save_chunks(chunks)

    def fts_search(self, query: str, limit: int = 100, **kwargs) -> list[tuple]:
        """FTS search across all shards, merging results by score.

        WARNING (Phase 2): rowids are NOT globally unique across shards.
        Each shard has its own rowid sequence starting at 1. Callers that
        use rowids to fetch chunk data (get_embeddings, load by rowid)
        will get wrong results when merging across shards. Phase 2 must
        use shard-qualified identifiers (shard_idx, rowid) instead.
        """
        if self._data_dbs:
            all_results = []
            for db in self._data_dbs:
                all_results.extend(db.fts_search(query, limit=limit, **kwargs))
            # Sort by score descending (score is element [1])
            all_results.sort(key=lambda r: r[1], reverse=True)
            return all_results[:limit]
        return self._index.fts_search(query, limit=limit, **kwargs)

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
