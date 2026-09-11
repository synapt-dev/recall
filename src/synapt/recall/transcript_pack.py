"""Seal CLOSED transcript sessions of a project dir into a content-addressed
pack + idx under the store, per the packfile design's working-first proof
(steps 1-3), promoted to a shipped verb.

No dictionary, no blob dedupe, no live-session slicing -- the same scope as
the proof. Loose bytes are NEVER truncated or deleted by this module; a pack
is an additional, verifiable receipt over bytes that still exist on disk.

Streaming throughout: sealing reads a session in fixed-size chunks (never
``Path.read_bytes()``), so a multi-GB transcript never loads whole into
memory. Verification streams the compressed member straight off disk through
``_BoundedFileReader`` rather than materializing it whole before
decompressing, for the same reason.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

CHUNK_SIZE = 1 << 20  # 1 MiB streaming read/compress chunk

PACK_DIRNAME = "pack"
PACK_FILENAME = "sessions.pack"
IDX_FILENAME = "sessions.idx.jsonl"


def pack_dir(index_dir: Path) -> Path:
    return index_dir / PACK_DIRNAME


def pack_path(index_dir: Path) -> Path:
    return pack_dir(index_dir) / PACK_FILENAME


def idx_path(index_dir: Path) -> Path:
    return pack_dir(index_dir) / IDX_FILENAME


class _BoundedFileReader:
    """A read-only file-like view onto at most ``limit`` bytes read from
    ``f`` starting at its current position. Handed to ``GzipFile`` so
    verification streams the compressed member straight off disk instead of
    materializing it whole in memory first -- the earlier form
    (``f.read(compressed_length)`` into a ``BytesIO``) held the ENTIRE
    compressed member in memory before decompression ever started, which
    defeated the "streaming throughout" claim for any large member (measured:
    a 25.2 MB peak against a 21.7 MB compressed member). ``GzipFile`` only
    ever calls ``.read(size)`` on its ``fileobj``, so this is sufficient."""

    def __init__(self, f, limit: int) -> None:
        self._f = f
        self._remaining = limit

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if size < 0 or size > self._remaining:
            size = self._remaining
        data = self._f.read(size)
        self._remaining -= len(data)
        return data


@dataclass(frozen=True)
class SealReceipt:
    session_id: str
    sha256: str
    raw_length: int
    compressed_length: int
    line_count: int
    pack_offset: int
    verified: bool


@dataclass
class SealResult:
    receipts: list[SealReceipt] = field(default_factory=list)
    skipped_already_packed: list[str] = field(default_factory=list)
    total_raw_bytes: int = 0
    total_compressed_bytes: int = 0

    @property
    def sealed_count(self) -> int:
        return len(self.receipts)


def read_idx(index_dir: Path) -> list[dict]:
    """Every row currently in this index's pack idx, oldest first. Empty
    list (never an error) when no pack has been sealed yet."""
    p = idx_path(index_dir)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def session_pack_row(index_dir: Path, session_id: str) -> dict | None:
    """The idx row for a session, or None if it has no packed segment."""
    for row in read_idx(index_dir):
        if row["session_id"] == session_id:
            return row
    return None


def _streaming_seal_one(source: Path, dest_pack: Path) -> tuple[str, int, int, int]:
    """Stream ``source`` into a gzip member appended to ``dest_pack``.
    Returns (sha256_hex, raw_length, compressed_length, line_count).
    Never reads the whole source into memory."""
    sha = hashlib.sha256()
    raw_length = 0
    line_count = 0
    offset_before = dest_pack.stat().st_size if dest_pack.exists() else 0
    with dest_pack.open("ab") as out_f, gzip.GzipFile(fileobj=out_f, mode="wb") as gz:
        with source.open("rb") as in_f:
            while True:
                chunk = in_f.read(CHUNK_SIZE)
                if not chunk:
                    break
                sha.update(chunk)
                raw_length += len(chunk)
                line_count += chunk.count(b"\n")
                gz.write(chunk)
    compressed_length = dest_pack.stat().st_size - offset_before
    return sha.hexdigest(), raw_length, compressed_length, line_count


def _streaming_verify_one(pack_file: Path, row: dict) -> bool:
    """Re-read the segment this row claims, streaming, and confirm its sha256
    and byte length match what was recorded. Never trusts the write path's
    own claim about itself."""
    sha = hashlib.sha256()
    raw_length = 0
    with pack_file.open("rb") as f:
        f.seek(row["pack_offset"])
        bounded = _BoundedFileReader(f, row["compressed_length"])
        with gzip.GzipFile(fileobj=bounded, mode="rb") as gz:
            while True:
                chunk = gz.read(CHUNK_SIZE)
                if not chunk:
                    break
                sha.update(chunk)
                raw_length += len(chunk)
    return sha.hexdigest() == row["sha256"] and raw_length == row["raw_length"]


def seal_closed_sessions(
    project_dir: Path,
    index_dir: Path,
    *,
    live_session_id: str | None = None,
) -> SealResult:
    """Seal every CLOSED session under ``project_dir`` (every ``*.jsonl`` file
    except ``live_session_id``, which is the currently-writing transcript and
    is never sealed) into the pack under ``index_dir``. Idempotent: a session
    already present in the idx is skipped, never re-sealed or re-verified
    here (use ``verify_pack`` to re-check existing segments). Loose files are
    never truncated or removed."""
    pdir = pack_dir(index_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    already = {row["session_id"] for row in read_idx(index_dir)}

    result = SealResult()
    idx_file = idx_path(index_dir)
    pfile = pack_path(index_dir)

    for source in sorted(project_dir.glob("*.jsonl")):
        session_id = source.stem
        if session_id == live_session_id:
            continue
        if session_id in already:
            result.skipped_already_packed.append(session_id)
            continue

        offset_before = pfile.stat().st_size if pfile.exists() else 0
        sha, raw_length, compressed_length, line_count = _streaming_seal_one(source, pfile)
        row = {
            "kind": "transcript",
            "sha256": sha,
            "source_path": str(source),
            "session_id": session_id,
            "raw_length": raw_length,
            "line_count": line_count,
            "pack_offset": offset_before,
            "compressed_length": compressed_length,
            "indexed_by": None,
            "parser_version": None,
        }
        verified = _streaming_verify_one(pfile, row)
        with idx_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        result.receipts.append(
            SealReceipt(
                session_id=session_id,
                sha256=sha,
                raw_length=raw_length,
                compressed_length=compressed_length,
                line_count=line_count,
                pack_offset=offset_before,
                verified=verified,
            )
        )
        result.total_raw_bytes += raw_length
        result.total_compressed_bytes += compressed_length

    return result


@dataclass
class VerifyResult:
    ok: list[str] = field(default_factory=list)
    sha_mismatch: list[str] = field(default_factory=list)
    length_mismatch: list[str] = field(default_factory=list)
    boundary_violation: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return not (self.sha_mismatch or self.length_mismatch or self.boundary_violation)


def verify_pack(index_dir: Path) -> VerifyResult:
    """Re-verify every row in the idx against the pack bytes it claims,
    streaming. A row fails on:
    - SHA mismatch: the decompressed bytes don't hash to the recorded sha256.
    - Length mismatch: the decompressed byte count doesn't match raw_length.
    - Boundary violation: the decompressed bytes are non-empty and don't end
      on a line boundary ("never mid-line", per the design doc)."""
    result = VerifyResult()
    pfile = pack_path(index_dir)
    for row in read_idx(index_dir):
        sha = hashlib.sha256()
        raw_length = 0
        ends_clean = True
        last_byte = b""
        with pfile.open("rb") as f:
            f.seek(row["pack_offset"])
            bounded = _BoundedFileReader(f, row["compressed_length"])
            with gzip.GzipFile(fileobj=bounded, mode="rb") as gz:
                while True:
                    chunk = gz.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    sha.update(chunk)
                    raw_length += len(chunk)
                    last_byte = chunk[-1:]
        if raw_length > 0:
            ends_clean = last_byte == b"\n"

        if sha.hexdigest() != row["sha256"]:
            result.sha_mismatch.append(row["session_id"])
        elif raw_length != row["raw_length"]:
            result.length_mismatch.append(row["session_id"])
        elif not ends_clean:
            result.boundary_violation.append(row["session_id"])
        else:
            result.ok.append(row["session_id"])
    return result


def format_seal_receipt(receipt: SealReceipt) -> str:
    return (
        f"sealed {receipt.session_id[:8]} "
        f"sha={receipt.sha256[:12]} "
        f"bytes={receipt.raw_length} "
        f"compressed={receipt.compressed_length} "
        f"lines={receipt.line_count} "
        f"verify={'OK' if receipt.verified else 'FAILED'}"
    )
