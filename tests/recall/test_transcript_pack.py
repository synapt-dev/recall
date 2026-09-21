"""The synapt pack verb: seal CLOSED transcript sessions into a content-
addressed pack + idx, never touching loose bytes. TDD witness: a two-segment
fixture proving sealing/verify end to end, a mutated-sha control, and a
mutated-boundary (mid-line truncation) control."""
import gzip
import io
import json

from synapt.recall import transcript_pack as tp


def _write_session(project_dir, name, lines):
    p = project_dir / f"{name}.jsonl"
    p.write_text("".join(line + "\n" for line in lines))
    return p


def test_seal_two_closed_sessions_excludes_the_live_one(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    index_dir = tmp_path / "index"

    _write_session(project_dir, "closed-a", ['{"turn": 1}', '{"turn": 2}'])
    _write_session(project_dir, "closed-b", ['{"turn": 1}'])
    _write_session(project_dir, "live-session", ['{"turn": 1}'])  # excluded

    result = tp.seal_closed_sessions(project_dir, index_dir, live_session_id="live-session")

    assert result.sealed_count == 2
    assert result.skipped_already_packed == []
    sealed_ids = {r.session_id for r in result.receipts}
    assert sealed_ids == {"closed-a", "closed-b"}
    assert all(r.verified for r in result.receipts)

    # idx and pack files exist under the store
    assert tp.idx_path(index_dir).exists()
    assert tp.pack_path(index_dir).exists()

    rows = tp.read_idx(index_dir)
    assert {row["session_id"] for row in rows} == {"closed-a", "closed-b"}
    for row in rows:
        assert row["indexed_by"] is None  # honestly disclosed, not fabricated


def test_seal_is_idempotent_and_never_re_seals_an_already_packed_session(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    index_dir = tmp_path / "index"
    _write_session(project_dir, "closed-a", ['{"turn": 1}'])

    first = tp.seal_closed_sessions(project_dir, index_dir)
    assert first.sealed_count == 1

    # captured BEFORE the second call, so the comparison below is against an
    # independent prior value, not the same stat() call compared to itself
    # (a self-tautology that would pass regardless of whether sealing is
    # actually idempotent)
    pack_size_after_first = tp.pack_path(index_dir).stat().st_size

    second = tp.seal_closed_sessions(project_dir, index_dir)
    assert second.sealed_count == 0
    assert second.skipped_already_packed == ["closed-a"]
    # the pack file did not grow on the second call
    assert tp.pack_path(index_dir).stat().st_size == pack_size_after_first


def test_seal_never_touches_loose_bytes(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    index_dir = tmp_path / "index"
    source = _write_session(project_dir, "closed-a", ['{"turn": 1}'])
    original_bytes = source.read_bytes()

    tp.seal_closed_sessions(project_dir, index_dir)

    assert source.exists()
    assert source.read_bytes() == original_bytes


def test_verify_pack_reports_ok_for_a_freshly_sealed_pack(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    index_dir = tmp_path / "index"
    _write_session(project_dir, "closed-a", ['{"turn": 1}', '{"turn": 2}'])
    _write_session(project_dir, "closed-b", ['{"turn": 1}'])

    tp.seal_closed_sessions(project_dir, index_dir)
    result = tp.verify_pack(index_dir)

    assert result.all_ok
    assert set(result.ok) == {"closed-a", "closed-b"}
    assert result.sha_mismatch == []
    assert result.length_mismatch == []
    assert result.boundary_violation == []


def test_verify_pack_catches_a_mutated_sha(tmp_path):
    """Mutated-sha control: flip one recorded sha256 in the idx. verify_pack
    must report a mismatch for that segment, not silently pass."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    index_dir = tmp_path / "index"
    _write_session(project_dir, "closed-a", ['{"turn": 1}'])
    tp.seal_closed_sessions(project_dir, index_dir)

    rows = tp.read_idx(index_dir)
    rows[0]["sha256"] = "0" + rows[0]["sha256"][1:]
    tp.idx_path(index_dir).write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    result = tp.verify_pack(index_dir)
    assert result.sha_mismatch == ["closed-a"]
    assert result.ok == []
    assert not result.all_ok


def test_verify_pack_catches_a_mutated_boundary(tmp_path):
    """Mutated-boundary control: a segment whose recorded sha256 is CORRECT
    for its own (malformed) bytes -- as if some other sealer had cut a
    segment mid-line and hashed exactly what it wrote -- so this control is
    independent of the sha check. verify_pack must still refuse it as a
    boundary violation ("never mid-line", per the design doc), never as OK,
    even though the sha matches."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    index_dir = tmp_path / "index"
    _write_session(project_dir, "closed-a", ['{"turn": 1}', '{"turn": 2}'])
    tp.seal_closed_sessions(project_dir, index_dir)

    rows = tp.read_idx(index_dir)
    row = rows[0]
    pack_file = tp.pack_path(index_dir)
    with pack_file.open("rb") as f:
        f.seek(row["pack_offset"])
        original_comp = f.read(row["compressed_length"])
    with gzip.GzipFile(fileobj=io.BytesIO(original_comp), mode="rb") as gz:
        raw = gz.read()
    truncated_raw = raw[:-3]  # cut off mid-line, before the final newline
    truncated_sha = __import__("hashlib").sha256(truncated_raw).hexdigest()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(truncated_raw)
    truncated_comp = buf.getvalue()

    # rewrite the pack: the truncated member replaces the original segment
    # (this is the only segment in this fixture)
    with pack_file.open("r+b") as f:
        f.seek(row["pack_offset"])
        f.write(truncated_comp)
        f.truncate(row["pack_offset"] + len(truncated_comp))
    rows[0]["compressed_length"] = len(truncated_comp)
    rows[0]["raw_length"] = len(truncated_raw)
    rows[0]["sha256"] = truncated_sha  # correct for the malformed bytes
    tp.idx_path(index_dir).write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    result = tp.verify_pack(index_dir)
    assert result.ok == []
    assert result.sha_mismatch == []  # the sha genuinely matches
    assert result.length_mismatch == []  # the length genuinely matches
    assert result.boundary_violation == ["closed-a"]


def test_session_pack_row_finds_a_packed_session_and_none_for_an_unpacked_one(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    index_dir = tmp_path / "index"
    _write_session(project_dir, "closed-a", ['{"turn": 1}'])
    tp.seal_closed_sessions(project_dir, index_dir)

    assert tp.session_pack_row(index_dir, "closed-a") is not None
    assert tp.session_pack_row(index_dir, "never-packed") is None


def test_verify_peak_memory_is_bounded_not_proportional_to_member_size(tmp_path):
    """A session large enough that reading its whole compressed member into
    memory before decompressing (the earlier ``f.read(compressed_length)``
    form) would show up unmistakably in a peak-memory measurement.
    ``_streaming_verify_one`` and ``verify_pack`` must stream the compressed
    bytes through ``_BoundedFileReader`` instead, so peak allocation stays a
    small multiple of ``CHUNK_SIZE`` regardless of member size."""
    import base64
    import os
    import tracemalloc

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    index_dir = tmp_path / "index"
    # ~8 MiB of near-incompressible content (base64 of urandom bytes) so the
    # COMPRESSED member is comparably large to the raw content -- a
    # repetitive fixture compresses to near-nothing, which would pass this
    # test even with the whole-member-into-memory bug the fix removes.
    # each line independently random -- 8000 copies of ONE random line would
    # still compress to near-nothing (gzip's LZ77 window finds the repeat)
    lines = [
        json.dumps({"turn": base64.b64encode(os.urandom(750)).decode()})
        for _ in range(8000)
    ]
    _write_session(project_dir, "closed-a", lines)

    tp.seal_closed_sessions(project_dir, index_dir)
    row = tp.session_pack_row(index_dir, "closed-a")
    assert row["raw_length"] > 8_000_000
    assert row["compressed_length"] > 6_000_000, (
        "fixture must be near-incompressible or this test cannot discriminate "
        "streaming from whole-member-in-memory"
    )

    tracemalloc.start()
    ok = tp._streaming_verify_one(tp.pack_path(index_dir), row)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert ok
    # bounded by a small multiple of CHUNK_SIZE (gzip's own internal buffers
    # plus our chunk), never by the compressed member's own size
    assert peak < 4 * tp.CHUNK_SIZE, (
        f"peak={peak} bytes, compressed_length={row['compressed_length']} bytes "
        f"-- verification must not materialize the whole compressed member"
    )
