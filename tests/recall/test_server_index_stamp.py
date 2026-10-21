"""The cached index stamp ignores the empty WAL the server's own load creates.

Server A and writer B run in separate processes, as they do in practice.
"""

import subprocess
import sys
import textwrap
import time

from synapt.recall.server import _adopt_post_load_stamp

SERVER = textwrap.dedent(
    """
    import os, sys, time
    from pathlib import Path
    import synapt.recall.server as s

    store = Path(sys.argv[1])
    flag = Path(sys.argv[2])
    s.project_index_dir = lambda *a, **k: store
    orig = s.TranscriptIndex.load
    race = {"on": False}

    def slow_load(*a, **k):
        r = orig(*a, **k)
        if race["on"]:
            race["on"] = False
            flag.write_text("x")
            time.sleep(2)  # B writes here, after the data was read
        return r

    s.TranscriptIndex.load = slow_load
    prev = None
    for line in sys.stdin:
        if line.strip() == "arm":
            race["on"] = True
            continue
        idx = s._get_index(use_embeddings=False)
        print("reload" if id(idx) != prev else "cached", repr(s._cached_mtime), flush=True)
        prev = id(idx)
    """
)

WRITER = textwrap.dedent(
    """
    import os, sys
    from synapt.recall.storage import RecallDB
    db = RecallDB(sys.argv[1] + "/index.db")
    db.upsert_knowledge_node({"id": "kB", "content": "stamp probe", "category": "fact",
        "confidence": 0.9, "source_sessions": ["sB"], "created_at": "2026-10-21T00:00:00Z",
        "updated_at": "2026-10-21T00:00:00Z", "status": "active"})
    os._exit(0)  # no close, so no checkpoint: the write sits in the WAL
    """
)


def _store(tmp_path):
    store = tmp_path / "index"
    store.mkdir()
    subprocess.run(
        [sys.executable, "-c",
         "import sys; from synapt.recall.storage import RecallDB; "
         "RecallDB(sys.argv[1] + '/index.db').close()", str(store)],
        check=True,
    )
    return store


def _server(tmp_path, store):
    flag = tmp_path / "inload"
    proc = subprocess.Popen(
        [sys.executable, "-c", SERVER, str(store), str(flag)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    return proc, flag


def _call(proc):
    proc.stdin.write("get\n")
    proc.stdin.flush()
    return proc.stdout.readline().split(" ", 1)


def _write(store):
    subprocess.run([sys.executable, "-c", WRITER, str(store)], check=True)


def _stop(proc):
    proc.stdin.close()
    proc.wait(timeout=10)


def test_own_load_does_not_reload(tmp_path):
    store = _store(tmp_path)
    proc, _ = _server(tmp_path, store)
    try:
        assert _call(proc)[0] == "reload"
        assert _call(proc)[0] == "cached"
    finally:
        _stop(proc)


def test_content_write_by_another_process_reloads(tmp_path):
    store = _store(tmp_path)
    proc, _ = _server(tmp_path, store)
    try:
        first, stamp1 = _call(proc)
        assert first == "reload"
        _write(store)
        second, stamp2 = _call(proc)
        assert second == "reload"
        assert stamp2 != stamp1
    finally:
        _stop(proc)


def test_write_during_load_reloads_next_call(tmp_path):
    store = _store(tmp_path)
    proc, flag = _server(tmp_path, store)
    try:
        assert _call(proc)[0] == "reload"
        (store / "chunks.jsonl").write_text("")  # moves the stamp: the next call loads
        proc.stdin.write("arm\nget\n")
        proc.stdin.flush()
        deadline = time.monotonic() + 10
        while not flag.exists():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        _write(store)
        assert proc.stdout.readline().split(" ", 1)[0] == "reload"
        assert _call(proc)[0] == "reload"
    finally:
        _stop(proc)


def test_adopt_post_load_stamp_rule():
    db = ("index.db", 266240, 1)
    stamp = (db,)
    assert _adopt_post_load_stamp(stamp, (db, ("index.db-wal", 0, 2))) == (db, ("index.db-wal", 0, 2))
    assert _adopt_post_load_stamp(stamp, (db, ("index.db-wal", 28872, 2))) == stamp
    assert _adopt_post_load_stamp(stamp, (("index.db", 266240, 3), ("index.db-wal", 0, 2))) == stamp
    assert _adopt_post_load_stamp((db, ("recall.db", 1, 1)), (db, ("index.db-wal", 0, 2))) == (db, ("recall.db", 1, 1))
    wal = ("index.db-wal", 0, 1)
    assert _adopt_post_load_stamp((db, wal), (db, wal)) == (db, wal)
