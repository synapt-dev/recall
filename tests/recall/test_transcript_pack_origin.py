"""The Origin verb pair: `synapt recall pack push`/`fetch` against a directory
remote. TDD witness: a two-segment fixture proving push -> fetch -> verify
end to end, idempotence (a second push transfers zero), and a mutation on
fetch's own sha check (fetch must refuse a remote segment whose downloaded
bytes do not match its manifest row, never silently accept it)."""
import json
from pathlib import Path

import pytest

from synapt.recall import transcript_pack as tp
from synapt.recall import transcript_pack_origin as origin


def _write_session(project_dir, name, lines):
    p = project_dir / f"{name}.jsonl"
    p.write_text("".join(line + "\n" for line in lines))
    return p


def _sealed_two_segment_store(tmp_path) -> Path:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    index_dir = tmp_path / "index-a"
    _write_session(project_dir, "closed-a", ['{"turn": 1}', '{"turn": 2}'])
    _write_session(project_dir, "closed-b", ['{"turn": 1}'])
    tp.seal_closed_sessions(project_dir, index_dir)
    return index_dir


def test_push_then_fetch_two_segments_end_to_end(tmp_path):
    index_a = _sealed_two_segment_store(tmp_path)
    remote = tmp_path / "remote"
    index_b = tmp_path / "index-b"

    push_result = origin.push_to_remote(index_a, remote)
    assert push_result.transferred_count == 2
    assert set(push_result.transferred) == {r["sha256"] for r in tp.read_idx(index_a)}

    fetch_result = origin.fetch_from_remote(remote, index_b)
    assert fetch_result.transferred_count == 2
    assert set(fetch_result.transferred) == {r["sha256"] for r in tp.read_idx(index_a)}

    # the fetched store, verified independently -- every segment OK, none of
    # them re-derived from the original source, purely from what fetch wrote
    verify_result = tp.verify_pack(index_b)
    assert verify_result.all_ok
    assert set(verify_result.ok) == {"closed-a", "closed-b"}


def test_second_push_transfers_zero(tmp_path):
    index_a = _sealed_two_segment_store(tmp_path)
    remote = tmp_path / "remote"

    first = origin.push_to_remote(index_a, remote)
    assert first.transferred_count == 2

    second = origin.push_to_remote(index_a, remote)
    assert second.transferred_count == 0
    assert len(second.already_present) == 2


def test_second_fetch_transfers_zero(tmp_path):
    index_a = _sealed_two_segment_store(tmp_path)
    remote = tmp_path / "remote"
    index_b = tmp_path / "index-b"
    origin.push_to_remote(index_a, remote)

    first = origin.fetch_from_remote(remote, index_b)
    assert first.transferred_count == 2

    second = origin.fetch_from_remote(remote, index_b)
    assert second.transferred_count == 0
    assert len(second.already_present) == 2


def test_push_never_recompresses_transfers_the_gzip_member_verbatim(tmp_path):
    """The exact compressed byte range for a segment IS a standalone gzip
    member (one GzipFile per seal call) -- push must copy those bytes
    verbatim, never decompress-then-recompress. Confirmed by byte equality
    between the remote's .gz file and the local pack's own compressed
    range for that segment."""
    index_a = _sealed_two_segment_store(tmp_path)
    remote = tmp_path / "remote"
    origin.push_to_remote(index_a, remote)

    row = tp.read_idx(index_a)[0]
    with tp.pack_path(index_a).open("rb") as f:
        f.seek(row["pack_offset"])
        local_bytes = f.read(row["compressed_length"])
    remote_bytes = (remote / f"{row['sha256']}.gz").read_bytes()
    assert remote_bytes == local_bytes


def test_fetch_refuses_a_remote_segment_that_does_not_match_its_manifest_row(tmp_path):
    """Fetch must never accept a downloaded segment on the manifest's
    say-so alone. Corrupt the remote's .gz bytes after push (simulating
    bit rot, a truncated upload, or a manifest row for the wrong file) and
    confirm fetch refuses rather than silently writing the mismatched
    bytes into the local pack."""
    index_a = _sealed_two_segment_store(tmp_path)
    remote = tmp_path / "remote"
    index_b = tmp_path / "index-b"
    origin.push_to_remote(index_a, remote)

    row = tp.read_idx(index_a)[0]
    segment_path = remote / f"{row['sha256']}.gz"
    corrupted = bytearray(segment_path.read_bytes())
    corrupted[-1] ^= 0xFF  # flip the last byte -- breaks the gzip CRC/content
    segment_path.write_bytes(bytes(corrupted))

    with pytest.raises(origin.RemoteSegmentCorrupt):
        origin.fetch_from_remote(remote, index_b)

    # nothing from the corrupted segment (or the ones after it, since fetch
    # stops at the first failure) was written to the local store
    assert not tp.pack_path(index_b).exists() or tp.read_idx(index_b) == []


def test_manifest_cas_detects_a_writer_that_lands_between_its_read_and_its_swap(tmp_path, monkeypatch):
    """The CAS must not simply hope no one else writes between its read and
    its swap -- it must actually notice when someone does, and retry rather
    than clobber. A prior version of this test only ran two SEQUENTIAL
    appends, which never exercises the race-detection check at all: deleting
    the check entirely (`if current != before` -> `if False`) left that
    suite green. This test injects a foreign writer's row directly between
    the CAS's own two reads (monkeypatching Path.read_bytes so the SECOND
    read observes a manifest the first read did not), and asserts BOTH rows
    survive -- proving the check catches a genuine interleaving rather than
    merely not crashing when there is nothing to catch."""
    remote = tmp_path / "remote"
    remote.mkdir()
    manifest = origin._remote_manifest_path(remote)
    manifest.write_bytes(b"")  # exists but empty, so the CAS's first read fires

    foreign_row = {"kind": "transcript", "sha256": "foreign-sha", "session_id": "concurrent-writer"}
    mine_row = {"kind": "transcript", "sha256": "mine-sha", "session_id": "this-caller"}

    real_read_bytes = Path.read_bytes
    injected = {"done": False}

    def racy_read_bytes(self):
        content = real_read_bytes(self)
        if self == manifest and not injected["done"]:
            injected["done"] = True
            # A concurrent writer's append lands strictly AFTER our first
            # read returned, and strictly BEFORE our second (post-write)
            # read -- exactly the window the CAS exists to detect.
            manifest.write_bytes((json.dumps(foreign_row) + "\n").encode("utf-8"))
        return content

    monkeypatch.setattr(Path, "read_bytes", racy_read_bytes)

    origin._cas_append_manifest_row(remote, mine_row)

    manifest_rows = origin.read_remote_manifest(remote)
    assert {r["sha256"] for r in manifest_rows} == {"foreign-sha", "mine-sha"}


def test_manifest_row_never_carries_the_pusher_s_local_source_path(tmp_path):
    """The remote manifest row is a projection of the local idx row, never
    the row itself: source_path is the pusher's own absolute filesystem
    path, and pack_offset/indexed_by/parser_version are purely local too --
    none belong on a SHARED remote manifest every fetcher reads."""
    index_a = _sealed_two_segment_store(tmp_path)
    remote = tmp_path / "remote"
    origin.push_to_remote(index_a, remote)

    for row in origin.read_remote_manifest(remote):
        assert "source_path" not in row
        assert "pack_offset" not in row
        assert "indexed_by" not in row
        assert "parser_version" not in row
        assert set(row) <= set(origin._MANIFEST_ROW_FIELDS)


def test_fetch_writes_source_path_none_never_the_remote_row_s_own_value(tmp_path):
    """A fetched segment's local idx row must have source_path explicitly
    None -- not fabricated as a fake local path, not simply absent, and
    never copied from the remote manifest row (which does not carry it in
    the first place, per the projection above)."""
    index_a = _sealed_two_segment_store(tmp_path)
    remote = tmp_path / "remote"
    index_b = tmp_path / "index-b"
    origin.push_to_remote(index_a, remote)
    origin.fetch_from_remote(remote, index_b)

    for row in tp.read_idx(index_b):
        assert "source_path" in row
        assert row["source_path"] is None


def test_fetch_from_a_nonexistent_remote_refuses_rather_than_reporting_zero(tmp_path):
    """A remote directory that does not exist must be a distinguishable
    refusal, never a silent zero-transferred success indistinguishable from
    a genuinely caught-up remote."""
    missing_remote = tmp_path / "does-not-exist"
    index_b = tmp_path / "index-b"

    with pytest.raises(origin.RemoteNotFound):
        origin.fetch_from_remote(missing_remote, index_b)
