"""v7: ShardedRecallDB.save_clusters must forward save_clusters's receipt.

v6 changed RecallDB.save_clusters's contract from returning None to
returning {"dangling_removed": N}, and cli.py's build path started reading
that dict unconditionally on every ordinary (non-skip-clustering) build.
Every review round -- the author's own suite, Sentinel's R1 mutation
probes, Stromus's R2 -- exercised RecallDB directly or a monolithic
recall.db; nothing in any of them opened a store through the sharded
layout (index.db + data shards), which is how every real production store
is laid out. ShardedRecallDB.save_clusters was a bare delegating call with
no return statement, so a sharded store's build received None from a
method typed to return a dict and crashed with AttributeError on the very
first production build after v6 merged (reproduced live against the
team's own shared store, which is sharded).

Class sweep (Stromus's ask, not just this one line): every other
delegating method in ShardedRecallDB was checked against its underlying
RecallDB method's actual return statement. save_manifest,
save_knowledge_nodes, upsert_knowledge_node, append_clusters,
mark_recluster_attempted, save_cluster_token_signature,
save_cluster_summary, and record_access all genuinely return None in
storage.py -- their bare delegating calls are correct, not another
instance of this bug. merge_chunks_into_cluster raises rather than
returning a value on success, and an exception raised inside a bare
delegating call still propagates through it unchanged, so it needed no
change either. save_clusters was the only method whose underlying
contract changed to a non-None return without the wrapper being updated
to forward it.
"""
from conftest import assistant_entry, user_text_entry, write_jsonl
import test_recluster_maintenance as m
from synapt.recall.cli import _archive_and_build
from synapt.recall.core import project_index_dir
from synapt.recall.sharded_db import ShardedRecallDB
from synapt.recall.storage import RecallDB


def _open_sharded_db(project):
    return ShardedRecallDB.open(project_index_dir(project))


def test_probe_j_an_ordinary_build_through_the_sharded_proxy_survives_a_preexisting_dangling_row(tmp_path, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()

    # Pre-create the sharded layout (index.db present) BEFORE the first
    # build, so _archive_and_build_locked's is_sharded(index_dir) check
    # routes through ShardedRecallDB from the start, not the monolithic
    # RecallDB(recall.db) path every existing dangling-row probe takes.
    index_dir = project_index_dir(project)
    index_dir.mkdir(parents=True, exist_ok=True)
    RecallDB(index_dir / "index.db").close()

    m._topic_transcript(source / "topic.jsonl", turns=8)

    # Manufacture a dangling cluster_chunks row before any clustering has
    # ever run on this store -- clusters is empty at this point, so this
    # row counts as dangling the moment save_clusters's own dangling_removed
    # count runs (storage.py counts before the fresh clusters are inserted).
    seed_db = RecallDB(index_dir / "index.db")
    try:
        seed_db._conn.execute(
            "INSERT INTO cluster_chunks (cluster_id, chunk_id, added_at) "
            "VALUES ('clust-0000deadbeef-sharded', 'nonexistent:t0', "
            "'2026-01-01T00:00:00Z')"
        )
        seed_db._conn.commit()
    finally:
        seed_db.close()

    capsys.readouterr()  # discard nothing yet, but keep the boundary explicit
    _archive_and_build(
        project, source_dirs=[source], use_embeddings=False, incremental=True,
    )
    out = capsys.readouterr().out
    assert "Sharded layout" in out, (
        "setup failed to route through ShardedRecallDB -- this probe would "
        f"prove nothing against the monolithic path:\n{out}"
    )
    assert "1 dangling row(s) removed" in out, (
        f"save_clusters's receipt never reached cli.py's print line "
        f"through the sharded proxy:\n{out}"
    )

    db = _open_sharded_db(project)
    try:
        assert not db.is_monolithic
        clusters = db._index._conn.execute(
            "SELECT cluster_id FROM clusters WHERE cluster_type='topic'"
        ).fetchall()
        assert len(clusters) == 1, clusters
        dangling = db._index._conn.execute(
            "SELECT COUNT(*) FROM cluster_chunks WHERE cluster_id NOT IN "
            "(SELECT cluster_id FROM clusters)"
        ).fetchone()[0]
        assert dangling == 0, dangling
    finally:
        db.close()
