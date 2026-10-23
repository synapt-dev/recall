"""ShardedRecallDB.__getattr__ falls through to index.db only for its declared index-only surface.

index.db's chunks table is empty or a frozen pre-split copy in a sharded store, so any attribute that reaches it
through the fallthrough reads the wrong population with no error. The refusal makes a new reader fail loudly, and the
three cluster methods that joined against chunks read the shards through their own overrides.
"""

from __future__ import annotations

import ast
import inspect
import re
import sqlite3
import textwrap
import unittest

from synapt.recall.sharded_db import INDEX_ONLY_FALLTHROUGH, ShardedRecallDB
from synapt.recall.storage import RecallDB

from tests.recall.test_sharded_readers_see_shards import (
    LATE_CHUNK,
    make_sharded_store_with_a_shard_only_session,
)

CLUSTER = "clust-late"
_CHUNKS_SQL = re.compile(r"\b(FROM|JOIN|INTO|UPDATE)\s+chunks\b(?!_)", re.I)


def _seed_cluster(db: ShardedRecallDB, chunk_id: str) -> None:
    conn = db.index_conn
    conn.execute(
        "INSERT INTO clusters (cluster_id, topic, cluster_type, created_at, updated_at) "
        "VALUES (?, 'late topic', 'topic', '2025-03-01', '2025-03-01')",
        (CLUSTER,),
    )
    conn.execute(
        "INSERT INTO cluster_chunks (cluster_id, chunk_id, added_at) VALUES (?, ?, '2025-03-01')",
        (CLUSTER, chunk_id),
    )
    conn.commit()


class TheFallthroughIsDeclared(unittest.TestCase):
    def test_conn_is_refused_and_names_the_boundary(self):
        d = make_sharded_store_with_a_shard_only_session(legacy_copy=True)
        db = ShardedRecallDB.open(d)
        try:
            with self.assertRaises(AttributeError) as caught:
                db._conn  # noqa: B018
            self.assertIn("index_conn", str(caught.exception))
            self.assertIsInstance(db.index_conn, sqlite3.Connection)
        finally:
            db.close()

    def test_an_undeclared_name_is_refused(self):
        d = make_sharded_store_with_a_shard_only_session(legacy_copy=False)
        db = ShardedRecallDB.open(d)
        try:
            with self.assertRaises(AttributeError):
                db.chunk_count_for_values  # noqa: B018
        finally:
            db.close()

    def test_every_declared_name_is_a_recalldb_member_whose_sql_never_names_chunks(self):
        for name in sorted(INDEX_ONLY_FALLTHROUGH):
            with self.subTest(name=name):
                member = getattr(RecallDB, name, None)
                self.assertIsNotNone(member, f"{name} is not a RecallDB member")
                self.assertNotIn(name, vars(ShardedRecallDB), f"{name} is overridden, so it is not a fallthrough")
                fn = member.fget if isinstance(member, property) else member
                if callable(fn):
                    source = textwrap.dedent(inspect.getsource(fn))
                    ast.parse(source)
                    self.assertIsNone(_CHUNKS_SQL.search(source), f"{name} reads the chunks table")


class ClusterMethodsReadTheShards(unittest.TestCase):
    def _store(self, legacy_copy: bool) -> ShardedRecallDB:
        d = make_sharded_store_with_a_shard_only_session(legacy_copy)
        db = ShardedRecallDB.open(d)
        self.addCleanup(db.close)
        return db

    def test_get_cluster_chunk_texts_reads_a_member_that_exists_only_in_the_shards(self):
        for legacy_copy in (True, False):
            with self.subTest(legacy_copy=legacy_copy):
                db = self._store(legacy_copy)
                _seed_cluster(db, LATE_CHUNK)
                texts = db.get_cluster_chunk_texts(CLUSTER)
                self.assertEqual([t["user_text"] for t in texts], ["user late_only_in_shard"])

    def test_chunk_fts_to_clusters_maps_a_shard_only_hit_to_its_cluster(self):
        for legacy_copy in (True, False):
            with self.subTest(legacy_copy=legacy_copy):
                db = self._store(legacy_copy)
                _seed_cluster(db, LATE_CHUNK)
                hits = db.chunk_fts_to_clusters("late_only_in_shard")
                self.assertEqual([cluster for cluster, _ in hits], [CLUSTER])

    def test_create_singleton_cluster_finds_a_chunk_that_exists_only_in_the_shards(self):
        for legacy_copy in (True, False):
            with self.subTest(legacy_copy=legacy_copy):
                db = self._store(legacy_copy)
                cluster_id = db.create_singleton_cluster(LATE_CHUNK)
                self.assertIsNotNone(cluster_id)
                row = db.index_conn.execute(
                    "SELECT topic FROM clusters WHERE cluster_id = ?", (cluster_id,)
                ).fetchone()
                self.assertEqual(row[0], "user late_only_in_shard")


if __name__ == "__main__":
    unittest.main()
