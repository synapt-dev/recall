"""R2 v3 probes (Stromus): NULL-run_id dangling rows, receipt accuracy, refusal at the maintenance call site."""
import test_recluster_maintenance as m
from conftest import assistant_entry, user_text_entry, write_jsonl
from synapt.recall.cli import _archive_and_build
from synapt.recall.clustering import stale_transcript_chunk_ids, recluster_stale_chunks


def _build_once(tmp_path):
    project = tmp_path / "proj"; project.mkdir()
    source = tmp_path / "source"; source.mkdir()
    m._topic_transcript(source / "topic.jsonl", turns=8)
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
    return project, source


def test_probe_e_null_run_id_dangling_rows_are_removed_and_counted(tmp_path, capsys):
    project, source = _build_once(tmp_path)
    db = m._open_db(project)
    try:
        chunk = db._conn.execute("SELECT chunk_id FROM cluster_chunks LIMIT 1").fetchone()[0]
        db._conn.execute("INSERT INTO cluster_chunks (cluster_id, chunk_id, added_at, run_id) VALUES ('clust-dead-1', ?, '2026-03-01T00:00:00Z', NULL)", (chunk,))
        db._conn.execute("INSERT INTO cluster_chunks (cluster_id, chunk_id, added_at, run_id) VALUES ('clust-dead-2', ?, '2026-03-01T00:00:00Z', 'r1')", (chunk,))
        db._conn.commit()
        before = db._conn.execute("SELECT COUNT(*) FROM cluster_chunks WHERE cluster_id NOT IN (SELECT cluster_id FROM clusters)").fetchone()[0]
        assert before == 2
    finally:
        db.close()
    write_jsonl(source / "unrelated.jsonl", [
        user_text_entry("completely unrelated filler turn", uuid="fill-u", ts="2026-03-02T12:00:00Z"),
        assistant_entry(text="completely unrelated filler answer", uuid="fill-a", ts="2026-03-02T12:00:30Z"),
    ])
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
    out = capsys.readouterr().out
    db = m._open_db(project)
    try:
        after = db._conn.execute("SELECT COUNT(*) FROM cluster_chunks WHERE cluster_id NOT IN (SELECT cluster_id FROM clusters)").fetchone()[0]
        assert after == 0, after
    finally:
        db.close()
    assert "(2 dangling row(s) removed)" in out, out.splitlines()[-6:]


def test_probe_f_control_clean_store_prints_no_dangling_note(tmp_path, capsys):
    project, source = _build_once(tmp_path)
    write_jsonl(source / "unrelated.jsonl", [
        user_text_entry("completely unrelated filler turn", uuid="fill-u", ts="2026-03-02T12:00:00Z"),
        assistant_entry(text="completely unrelated filler answer", uuid="fill-a", ts="2026-03-02T12:00:30Z"),
    ])
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True)
    out = capsys.readouterr().out
    assert "dangling" not in out


def test_probe_g_maintenance_merge_into_a_vanished_cluster_does_not_abort_the_batch(tmp_path, monkeypatch):
    """A concurrent rebuild removes the cluster between the maintenance pass's
    lookup and its merge write: the pass must refuse that one chunk and finish
    the batch, not raise out of it with a partial batch committed.

    The race is staged as a real save_clusters-shaped removal: the cluster
    row AND its cluster_chunks rows go together, in the same delete, exactly
    as save_clusters's own cleanup does it -- a real concurrent rebuild never
    leaves a cluster's other members dangling on their own; only removing
    the cluster row alone (an earlier draft of this probe) manufactures a
    state no real rebuild produces."""
    project, source = _build_once(tmp_path)
    m._similar_to_cluster_singleton(source / "similar.jsonl")
    _archive_and_build(project, source_dirs=[source], use_embeddings=False, incremental=True, skip_clustering=True)
    db = m._open_db(project)
    try:
        real = db.merge_chunks_into_cluster
        def racing(cluster_id, chunk_ids, appended_text, added_at, run_id=None):
            db._conn.execute("DELETE FROM cluster_chunks WHERE cluster_id = ?", (cluster_id,))
            db._conn.execute("DELETE FROM clusters WHERE cluster_id = ?", (cluster_id,))
            db._conn.commit()
            return real(cluster_id, chunk_ids, appended_text, added_at, run_id)
        monkeypatch.setattr(db, "merge_chunks_into_cluster", racing)
        receipt = recluster_stale_chunks(db, batch_size=100, merge_into_existing=True)
        print("RECEIPT", {k: v for k, v in receipt.items() if not isinstance(v, (list, dict))})
        dangling = db._conn.execute("SELECT COUNT(*) FROM cluster_chunks WHERE cluster_id NOT IN (SELECT cluster_id FROM clusters)").fetchone()[0]
        assert dangling == 0, dangling
    finally:
        db.close()
