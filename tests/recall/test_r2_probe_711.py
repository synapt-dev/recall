"""R2 probes (Stromus) on fix/save-clusters-preserves-incremental-merges."""
import test_recluster_maintenance as m
from conftest import assistant_entry, user_text_entry, write_jsonl
from synapt.recall.cli import _archive_and_build
from synapt.recall.clustering import stale_transcript_chunk_ids


def _setup(tmp_path):
    project = tmp_path / "proj"; project.mkdir()
    source = tmp_path / "source"; source.mkdir()
    m._topic_transcript(source / "topic.jsonl", turns=8)
    # an unrelated singleton: the self-batch grouping never places it (MIN_CLUSTER_SIZE=2)
    write_jsonl(source / "lonely.jsonl", [
        user_text_entry("zebra quantum bagpipe lonely turn", uuid="lone-u", ts="2026-03-01T11:00:00Z"),
        assistant_entry(text="okapi violin marmalade different answer", uuid="lone-a", ts="2026-03-01T11:00:30Z"),
    ])
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
    db = m._open_db(project)
    try:
        clusters = db._conn.execute("SELECT cluster_id FROM clusters WHERE cluster_type='topic'").fetchall()
        assert len(clusters) == 1, clusters
        cid = clusters[0][0]
        stale = stale_transcript_chunk_ids(db)
        assert len(stale) >= 1, stale
        lone = stale[0]
        print("STALE-BEFORE", stale, "CLUSTERS", db._conn.execute("SELECT cluster_id, chunk_count FROM clusters").fetchall())
        db.merge_chunks_into_cluster(cid, [lone], "appended", "2026-03-01T12:00:00Z", run_id="probe-run")
        db._conn.commit()
        assert db._conn.execute("SELECT COUNT(*) FROM cluster_chunks WHERE run_id='probe-run'").fetchone()[0] == 1
        assert stale_transcript_chunk_ids(db) == [] or lone not in stale_transcript_chunk_ids(db)
    finally:
        db.close()
    return project, source, cid, lone


def test_probe_a_membership_of_a_chunk_the_fresh_pass_does_not_place_survives_an_ordinary_build(tmp_path):
    """The LIVE class: a stale chunk is stale BECAUSE the self-batch grouping
    leaves it out; maintenance merges it; the next ordinary build must keep it."""
    project, source, cid, lone = _setup(tmp_path)
    write_jsonl(source / "unrelated.jsonl", [
        user_text_entry("completely unrelated filler turn", uuid="fill-u", ts="2026-03-02T12:00:00Z"),
        assistant_entry(text="completely unrelated filler answer", uuid="fill-a", ts="2026-03-02T12:00:30Z"),
    ])
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
    db = m._open_db(project)
    try:
        rows = db._conn.execute("SELECT cluster_id FROM cluster_chunks WHERE run_id='probe-run'").fetchall()
        stale_after = stale_transcript_chunk_ids(db)
        print("STALE-AFTER", stale_after, "ROWS", rows, "CLUSTERS", db._conn.execute("SELECT cluster_id, chunk_count FROM clusters").fetchall())
        assert rows and lone not in stale_after, (
            f"merged singleton lost on an ordinary build: run_id rows={rows}, lone stale again={lone in stale_after}")
    finally:
        db.close()


def test_probe_b_run_id_is_never_stamped_onto_a_row_the_run_did_not_write(tmp_path):
    """Author's own scenario (a chunk the fresh pass DOES place): after the
    rebuild, the run_id must not label a membership the BUILD created in a
    different cluster, or revert-by-run_id deletes the build's own work."""
    project = tmp_path / "proj"; project.mkdir()
    source = tmp_path / "source"; source.mkdir()
    m._topic_transcript(source / "topic.jsonl", turns=8)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
    m._similar_to_cluster_singleton(source / "similar.jsonl")
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True, skip_clustering=True)
    db = m._open_db(project)
    try:
        from synapt.recall.clustering import recluster_stale_chunks
        old_cid = db._conn.execute("SELECT cluster_id FROM clusters WHERE cluster_type='topic'").fetchone()[0]
        receipt = recluster_stale_chunks(db, batch_size=100, merge_into_existing=True)
        run_id = receipt["merge_run_id"]
    finally:
        db.close()
    write_jsonl(source / "unrelated.jsonl", [
        user_text_entry("completely unrelated filler turn", uuid="fill-u", ts="2026-03-02T12:00:00Z"),
        assistant_entry(text="completely unrelated filler answer", uuid="fill-a", ts="2026-03-02T12:00:30Z"),
    ])
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
    db = m._open_db(project)
    try:
        rows = db._conn.execute("SELECT cluster_id, chunk_id FROM cluster_chunks WHERE run_id=?", (run_id,)).fetchall()
        # CORRECTED (Atlas, matching R2's own prose exactly: "no run_id on a
        # build-written row, every surviving run_id row in a cluster holding
        # the old founders"): the merged chunk here IS placed by the fresh
        # pass (it re-groups with its 8 former cluster-mates), so the fixed
        # behavior is to stamp NOTHING -- zero rows is the fully-correct
        # outcome, not a failure. The as-written `assert rows` above assumed
        # a row would still exist; it does not once "stamp nothing" is
        # implemented literally. The invariant that actually matters is
        # conditional: IF any row still carries this run_id, it must be
        # correctly homed with the old founders, never in some unrelated
        # build-written cluster.
        assert not rows or all(cid == old_cid for cid, _chunk_id in rows), (
            f"run_id {run_id} labels row(s) in cluster(s) "
            f"{sorted({cid for cid, _ in rows})}, not the merge's cluster "
            f"{old_cid}; revert-by-run_id would delete unrelated membership")
    finally:
        db.close()
