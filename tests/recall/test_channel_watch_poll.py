"""The watch's per-poll read, driven directly: a log shorter than the tail it keeps."""

from __future__ import annotations

from synapt.recall.channel import _WatchedFile, _watch_poll


def _rewrite_in_place(path, data: bytes) -> None:
    with open(path, "r+b") as f:
        f.truncate(0)
        f.write(data)


def test_short_log_rewritten_larger_in_place_prints_its_line_once(tmp_path):
    log = tmp_path / "dev.jsonl"
    log.write_bytes(b'{"body": "old line one"}\n')  # 25 bytes, under the 64-byte tail
    state = _WatchedFile(log)
    assert _watch_poll(state) == [b'{"body": "old line one"}']
    ino = log.stat().st_ino

    new = b'{"body": "the new line, longer"}\n'  # larger, still under 64 bytes
    _rewrite_in_place(log, new)
    assert log.stat().st_ino == ino and log.stat().st_size > state.pos

    seen = [line for _ in range(5) for line in _watch_poll(state)]
    assert seen == [new.rstrip(b"\n")]
