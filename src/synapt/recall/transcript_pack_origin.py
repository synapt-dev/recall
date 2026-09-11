"""Push/fetch a sealed pack store to/from a directory remote -- the Origin
verb pair promoted from a working-first proof to the shipped
``synapt recall pack push|fetch`` verb.

v1 remote shape, unchanged from the proof: a directory that answers
has(sha)/put(sha)/get(sha) via plain filesystem ops -- one ``<sha256>.gz``
file per segment plus one ``manifest.jsonl`` sidecar naming every segment the
remote holds. No merge, no conflict resolution -- only "does the remote have
this sha yet." Content-addressed, so a push or fetch that has already run is
idempotent: a second push transfers zero once the remote already holds
everything the source does.

Each sealed segment in the local ``.pack`` file (``transcript_pack.py``'s
``_streaming_seal_one``) is its own complete, independently-decompressible
gzip stream -- one ``GzipFile`` per seal call, appended raw. So the exact
byte range ``[pack_offset, pack_offset + compressed_length)`` IS a valid
standalone ``.gz`` file; push and fetch move that range verbatim, never
re-compressing.

The remote manifest row is a projection of the local idx row, never the row
itself: kind, sha256, session_id, raw_length, line_count, and
compressed_length are content-addressed and safe to share, but the local
idx also carries source_path (the pusher's own absolute filesystem path),
pack_offset, indexed_by, and parser_version, none of which mean anything to
a fetcher and all of which are purely local. Fetch reconstructs its own
local idx row rather than copying the remote's, setting source_path to
None explicitly.

Trust: fetch never accepts a remote segment on the manifest's say-so alone.
It decompresses what it downloaded and re-hashes it against the manifest
row's own claimed sha256 and raw_length BEFORE appending it to the local
pack + idx -- the same "never trust the write path's own claim about
itself" discipline ``_streaming_verify_one`` already applies to a local
pack. A segment that fails this check is refused, not silently accepted.

The remote manifest is appended under a simple compare-and-swap: read
current bytes, prepare new bytes, and only swap in the new content if the
manifest is unchanged since the read (retrying a bounded number of times
otherwise). This is not a full lock -- a genuinely concurrent pair of
pushers can still race between the read and the swap -- but it closes the
common lost-update window a naive append leaves wide open, and is disclosed
as a working-first tradeoff rather than the durable answer.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .transcript_pack import (
    CHUNK_SIZE,
    _BoundedFileReader,
    idx_path,
    pack_path,
    read_idx,
)

MANIFEST_FILENAME = "manifest.jsonl"
_CAS_MAX_RETRIES = 8

# Content-addressed fields only. A local idx row also carries source_path
# (the pusher's own absolute filesystem path), pack_offset, indexed_by, and
# parser_version -- all purely local, and pushing them verbatim into a
# SHARED remote manifest would leak the pusher's home-directory layout to
# every fetcher. The manifest row is a projection, never the raw idx row.
_MANIFEST_ROW_FIELDS = ("kind", "sha256", "session_id", "raw_length", "line_count", "compressed_length")


def _manifest_row_for(row: dict) -> dict:
    return {k: row[k] for k in _MANIFEST_ROW_FIELDS if k in row}


def _remote_manifest_path(remote_dir: Path) -> Path:
    return remote_dir / MANIFEST_FILENAME


def _remote_segment_path(remote_dir: Path, sha256: str) -> Path:
    return remote_dir / f"{sha256}.gz"


def read_remote_manifest(remote_dir: Path) -> list[dict]:
    """Every row the remote currently claims to hold, oldest first. Empty
    list (never an error) when the remote has no manifest yet."""
    p = _remote_manifest_path(remote_dir)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _cas_append_manifest_row(remote_dir: Path, row: dict) -> None:
    """Append one row to the remote manifest, refusing to swap in a write
    that would silently clobber a concurrent writer's append. Reads the
    current bytes, builds new bytes with the row appended, and only renames
    the write into place if the manifest is unchanged from the read --
    retrying (re-reading, re-appending) a bounded number of times on a
    detected change rather than either blocking or overwriting."""
    remote_dir = Path(remote_dir)
    remote_dir.mkdir(parents=True, exist_ok=True)
    manifest = _remote_manifest_path(remote_dir)
    line = (json.dumps(row) + "\n").encode("utf-8")
    for _attempt in range(_CAS_MAX_RETRIES):
        before = manifest.read_bytes() if manifest.exists() else b""
        new_content = before + line
        tmp = remote_dir / f".manifest.jsonl.tmp.{os.getpid()}.{_attempt}"
        tmp.write_bytes(new_content)
        current = manifest.read_bytes() if manifest.exists() else b""
        if current != before:
            tmp.unlink(missing_ok=True)
            continue
        os.replace(tmp, manifest)
        return
    raise RuntimeError(
        f"manifest compare-and-swap did not converge after {_CAS_MAX_RETRIES} attempts "
        f"at {manifest}"
    )


@dataclass
class PushResult:
    transferred: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)

    @property
    def transferred_count(self) -> int:
        return len(self.transferred)


def push_to_remote(index_dir: Path, remote_dir: Path) -> PushResult:
    """Upload every local segment the remote does not yet claim to hold.
    Copies the exact compressed byte range for each segment verbatim (no
    re-compression) to ``<remote>/<sha256>.gz``, then appends its manifest
    row. A segment already named in the remote manifest is skipped --
    push is idempotent, a second call transfers zero once the remote is
    caught up."""
    remote_dir = Path(remote_dir)
    remote_dir.mkdir(parents=True, exist_ok=True)
    local_rows = read_idx(index_dir)
    known = {row["sha256"] for row in read_remote_manifest(remote_dir)}
    pfile = pack_path(index_dir)

    result = PushResult()
    for row in local_rows:
        sha = row["sha256"]
        if sha in known:
            result.already_present.append(sha)
            continue
        with pfile.open("rb") as f:
            f.seek(row["pack_offset"])
            bounded = _BoundedFileReader(f, row["compressed_length"])
            dest = _remote_segment_path(remote_dir, sha)
            with dest.open("wb") as out_f:
                while True:
                    chunk = bounded.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out_f.write(chunk)
        _cas_append_manifest_row(remote_dir, _manifest_row_for(row))
        known.add(sha)
        result.transferred.append(sha)
    return result


class RemoteSegmentCorrupt(RuntimeError):
    """A remote segment's downloaded bytes do not hash to the manifest's
    own claimed sha256/raw_length -- fetch refuses rather than trusting the
    remote's say-so about its own content."""


class RemoteNotFound(RuntimeError):
    """The remote directory does not exist -- distinct from an existing,
    genuinely empty remote (which reports zero transferred, not a refusal).
    Without this, fetching from a mistyped or not-yet-created remote path
    silently reports "fetched 0", indistinguishable from a caught-up
    remote that legitimately has nothing new."""


@dataclass
class FetchResult:
    transferred: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)

    @property
    def transferred_count(self) -> int:
        return len(self.transferred)


def _verify_downloaded_segment(comp: bytes, row: dict) -> None:
    """Decompress ``comp`` and confirm it hashes to ``row``'s own claimed
    sha256 and raw_length before it is ever appended to the local pack.
    Raises RemoteSegmentCorrupt on a mismatch OR on bytes that are not even
    a well-formed gzip member (a corrupted transfer, a truncated upload) --
    never silently accepted, and never a raw gzip exception escaping this
    module's own refusal contract."""
    sha = hashlib.sha256()
    raw_length = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(comp), mode="rb") as gz:
            while True:
                chunk = gz.read(CHUNK_SIZE)
                if not chunk:
                    break
                sha.update(chunk)
                raw_length += len(chunk)
    except (gzip.BadGzipFile, EOFError, OSError) as exc:
        raise RemoteSegmentCorrupt(
            f"downloaded segment for session {row.get('session_id', '?')!r} is not a "
            f"well-formed gzip member: {exc}"
        ) from exc
    if sha.hexdigest() != row["sha256"] or raw_length != row["raw_length"]:
        raise RemoteSegmentCorrupt(
            f"downloaded segment for session {row.get('session_id', '?')!r} does not "
            f"match its manifest row: expected sha={row['sha256'][:12]} "
            f"raw_length={row['raw_length']}, got sha={sha.hexdigest()[:12]} "
            f"raw_length={raw_length}"
        )


def fetch_from_remote(remote_dir: Path, index_dir: Path) -> FetchResult:
    """Download every remote segment not already present locally, verifying
    each one's own bytes against the manifest's claimed sha256/raw_length
    before it is appended to the local pack + idx. A segment that fails
    verification raises RemoteSegmentCorrupt and nothing about it is
    written locally."""
    from .transcript_pack import pack_dir as _pack_dir

    remote_dir = Path(remote_dir)
    if not remote_dir.exists():
        raise RemoteNotFound(f"remote directory does not exist: {remote_dir}")

    index_dir = Path(index_dir)
    pdir = _pack_dir(index_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    remote_rows = read_remote_manifest(remote_dir)
    local_known = {row["sha256"] for row in read_idx(index_dir)}
    pfile = pack_path(index_dir)
    idx_file = idx_path(index_dir)

    result = FetchResult()
    for row in remote_rows:
        sha = row["sha256"]
        if sha in local_known:
            result.already_present.append(sha)
            continue
        comp = _remote_segment_path(remote_dir, sha).read_bytes()
        _verify_downloaded_segment(comp, row)

        offset_before = pfile.stat().st_size if pfile.exists() else 0
        with pfile.open("ab") as out_f:
            out_f.write(comp)
        # A fresh local idx row, never a copy of the remote's manifest row:
        # source_path is explicitly None (this segment did not come from a
        # local file on this desk), not fabricated and not omitted.
        new_row = {
            "kind": row["kind"],
            "sha256": row["sha256"],
            "source_path": None,
            "session_id": row["session_id"],
            "raw_length": row["raw_length"],
            "line_count": row["line_count"],
            "pack_offset": offset_before,
            "compressed_length": len(comp),
            "indexed_by": None,
            "parser_version": None,
        }
        with idx_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(new_row) + "\n")
        local_known.add(sha)
        result.transferred.append(sha)
    return result
