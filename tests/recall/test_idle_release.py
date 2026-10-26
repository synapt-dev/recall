"""The recall server gives memory back when idle.

B: the server embeds through `ChildEmbeddings`, a child process that holds the model and exits
when idle, so torch never enters the server. A: after an idle interval with no tool call, the
server drops its index cache. These rows run the child in its fake mode (no model, no torch); the
rows that need the real model (b2 vectors, b5 cold start, the memory bar) run on a box.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from synapt.recall import embed_child, embeddings
from synapt.recall.embeddings import ChildEmbeddings, EmbedChildError


@pytest.fixture
def fake_child(monkeypatch, tmp_path):
    monkeypatch.setenv("RECALL_EMBED_CHILD_FAKE", "1")
    monkeypatch.delenv("RECALL_EMBED_IDLE_S", raising=False)
    monkeypatch.delenv("RECALL_EMBED_CHILD_EXIT_ONCE", raising=False)
    made = []

    def make(**kw):
        provider = ChildEmbeddings(cmd=[sys.executable, "-m", "synapt.recall.embed_child"], **kw)
        made.append(provider)
        return provider

    yield make
    for provider in made:
        provider.close()


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        # os.kill(pid, 0) TERMINATES a process on Windows, so ask for its exit code instead.
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        finally:
            kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def test_the_child_answers_dim_and_vectors_the_fake_way(fake_child):
    provider = fake_child()
    assert provider.dim == embed_child.FAKE_DIM
    vectors = provider.embed(["a", "b", "a"])
    assert len(vectors) == 3 and vectors[0] == vectors[2] and vectors[0] != vectors[1]
    assert provider.embed_single("a") == vectors[0]


def test_b3_the_child_exits_after_the_idle_interval_and_the_next_embed_starts_a_new_one(fake_child, monkeypatch):
    monkeypatch.setenv("RECALL_EMBED_IDLE_S", "1")
    provider = fake_child()
    first = provider.embed(["x"])
    proc = provider._proc
    pid = proc.pid
    # poll(), not kill(pid, 0): the child is ours, so until we reap it an exited child is a zombie
    deadline = time.monotonic() + 5
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert proc.poll() is not None, "the child outlived its idle interval"
    assert provider.embed(["x"]) == first
    assert provider._proc.pid != pid


def test_b3r_a_request_that_meets_the_child_exiting_is_retried_once_on_a_new_child(fake_child, monkeypatch, tmp_path):
    marker = tmp_path / "exited-once"
    monkeypatch.setenv("RECALL_EMBED_CHILD_EXIT_ONCE", str(marker))
    provider = fake_child()
    vectors = provider.embed(["same"])
    assert marker.exists(), "the seam never fired, so this row proved nothing"
    monkeypatch.delenv("RECALL_EMBED_CHILD_EXIT_ONCE")
    assert vectors == fake_child().embed(["same"])


def test_b4_a_child_that_dies_on_every_request_raises_the_embed_failure_with_its_reason(monkeypatch):
    dies = [sys.executable, "-c", "import sys; sys.stdin.readline(); sys.exit(0)"]
    provider = ChildEmbeddings(cmd=dies, timeout=10)
    with pytest.raises(EmbedChildError, match="exited without answering, twice"):
        provider.embed(["x"])


def test_b4_an_unknown_op_gets_an_error_reply_with_the_childs_words_never_a_hang(fake_child):
    provider = fake_child(timeout=10)
    began = time.monotonic()
    with pytest.raises(EmbedChildError, match="unknown op"):
        provider._request({"op": "nonsense"})
    assert time.monotonic() - began < 5, "the child did not answer the unknown op promptly"


def test_b4_a_child_that_never_answers_times_out_and_does_not_hang():
    hangs = [sys.executable, "-c", "import sys, time; sys.stdin.readline(); time.sleep(60)"]
    provider = ChildEmbeddings(cmd=hangs, timeout=1)
    began = time.monotonic()
    with pytest.raises(EmbedChildError, match="no reply in 1s"):
        provider.embed(["x"])
    assert time.monotonic() - began < 10
    assert provider._proc is None, "the stuck child was left running"


def test_b6_a_killed_server_leaves_no_child(tmp_path):
    script = textwrap.dedent("""
        import os, sys, time
        os.environ["RECALL_EMBED_CHILD_FAKE"] = "1"
        from synapt.recall.embeddings import ChildEmbeddings
        p = ChildEmbeddings(cmd=[sys.executable, "-m", "synapt.recall.embed_child"])
        p.embed(["x"])
        print(p._proc.pid, flush=True)
        time.sleep(60)
    """)
    parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        child_pid = int(parent.stdout.readline())
        assert _alive(child_pid)
        parent.kill()
        parent.wait()
        assert _wait_gone(child_pid, 5), "the child outlived its server"
    finally:
        if parent.poll() is None:
            parent.kill()


def test_b7_two_searches_while_the_child_starts_get_the_same_answers_from_one_child(fake_child, monkeypatch):
    provider = fake_child()
    starts = []
    real_start = provider._start

    def counted_start():
        starts.append(1)
        time.sleep(0.3)  # widen the window in which the second request arrives
        real_start()

    monkeypatch.setattr(provider, "_start", counted_start)
    results = {}

    def search(name):
        results[name] = provider.embed(["one query"])

    threads = [threading.Thread(target=search, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert results["a"] == results["b"]
    assert len(starts) == 1, f"{len(starts)} children were started"


def test_the_idle_interval_refuses_anything_but_a_positive_finite_number(monkeypatch):
    for bad in ("0", "-1", "nan", "inf", "abc"):
        monkeypatch.setenv("RECALL_EMBED_IDLE_S", bad)
        with pytest.raises(ValueError):
            embed_child.idle_seconds()
    monkeypatch.setenv("RECALL_EMBED_IDLE_S", "2.5")
    assert embed_child.idle_seconds() == 2.5
    monkeypatch.delenv("RECALL_EMBED_IDLE_S")
    assert embed_child.idle_seconds() == embed_child.DEFAULT_IDLE_S


def test_the_server_switch_resolves_the_child_provider(monkeypatch):
    monkeypatch.setattr(embeddings, "_module_present", lambda name: True)
    try:
        embeddings.use_child_embedder(True)
        assert isinstance(embeddings.get_embedding_provider(), ChildEmbeddings)
        embeddings.use_child_embedder(False)
        assert isinstance(embeddings.get_embedding_provider(), embeddings.LocalEmbeddings)
    finally:
        embeddings.use_child_embedder(False)


# --- A: the index cache is released when idle -------------------------------------------------

@pytest.fixture
def server_state(monkeypatch):
    from synapt.recall import server

    sentinel = object()
    monkeypatch.setattr(server, "_cached_index", sentinel)
    monkeypatch.setattr(server, "_calls_in_flight", 0)
    monkeypatch.setattr(server, "_last_activity", 100.0)
    dropped = []
    monkeypatch.setattr(server, "_invalidate_cache", lambda: (dropped.append(1), setattr(server, "_cached_index", None)))
    monkeypatch.setattr(server, "_trim_allocator", lambda: None)
    return server, dropped


def test_a_an_index_idle_past_the_interval_is_released(server_state):
    server, dropped = server_state
    assert server._release_if_idle(600, now=100.0 + 600) is True
    assert dropped == [1] and server._cached_index is None


def test_a_an_index_used_within_the_interval_is_kept(server_state):
    server, dropped = server_state
    assert server._release_if_idle(600, now=100.0 + 599) is False
    assert dropped == []


def test_a_a_release_never_runs_while_a_call_is_in_flight(server_state):
    server, dropped = server_state
    server._call_started()
    try:
        assert server._release_if_idle(600, now=server._last_activity + 10_000) is False
        assert dropped == []
    finally:
        server._call_finished()
    assert server._release_if_idle(600, now=server._last_activity + 600) is True


def test_a_the_release_interval_refuses_anything_but_a_positive_finite_number(monkeypatch):
    from synapt.recall import server

    for bad in ("0", "-5", "nan", "inf", "x"):
        monkeypatch.setenv("RECALL_IDLE_RELEASE_S", bad)
        with pytest.raises(ValueError):
            server._idle_release_seconds()
    monkeypatch.delenv("RECALL_IDLE_RELEASE_S")
    assert server._idle_release_seconds() == server.DEFAULT_IDLE_RELEASE_S


def test_a_a_server_main_that_stops_before_running_leaves_no_release_thread(monkeypatch):
    # a thread left behind makes every later fork() in the process warn (and risk deadlock)
    from synapt.recall import server

    class Stop(Exception):
        pass

    def stop():
        raise Stop

    monkeypatch.setattr(server, "_build_validating_fastmcp_class", stop)
    try:
        with pytest.raises(Stop):
            server.main()
    finally:
        embeddings.use_child_embedder(False)
    assert not [t for t in threading.enumerate() if t.name == "synapt-idle-release"]


def test_b2_vectors_cross_the_pipe_exactly_at_the_models_float32_precision():
    import array

    rows = [array.array("f", [0.1 * i - 3.3, 1e-7 * i, -2.5e3 + i]).tolist() for i in range(4)]
    assert embed_child.unpack(embed_child._pack(rows)) == rows
    assert embed_child.unpack(embed_child._pack([])) == []


def test_b4_a_reply_whose_floats_do_not_split_into_its_count_is_refused():
    packed = embed_child._pack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="do not split"):
        embed_child.unpack({**packed, "n": 4})


_NO_PIPE_SELECT = textwrap.dedent("""
    import selectors

    class _WindowsSelector:
        # On Windows select() takes sockets only: registering a pipe fails like this.
        def __init__(self, *a, **k):
            pass

        def register(self, *a, **k):
            raise OSError(10038, "not a socket")

        def close(self):
            pass

    selectors.DefaultSelector = _WindowsSelector
""")


def test_b4_the_child_pipe_is_never_waited_on_with_select_which_windows_refuses(monkeypatch):
    monkeypatch.setenv("RECALL_EMBED_CHILD_FAKE", "1")
    monkeypatch.delenv("RECALL_EMBED_IDLE_S", raising=False)
    monkeypatch.delenv("RECALL_EMBED_CHILD_EXIT_ONCE", raising=False)
    import selectors

    names: dict = {}
    exec(_NO_PIPE_SELECT.replace("selectors.DefaultSelector = _WindowsSelector", ""), names)
    monkeypatch.setattr(selectors, "DefaultSelector", names["_WindowsSelector"])  # the parent side
    child = _NO_PIPE_SELECT + "from synapt.recall import embed_child\nraise SystemExit(embed_child.main())\n"
    provider = ChildEmbeddings(cmd=[sys.executable, "-c", child], timeout=10)
    try:
        assert len(provider.embed(["a", "b"])) == 2
    finally:
        provider.close()


def test_a_the_interval_boundary_releases_whatever_the_clock_value(server_state):
    server, dropped = server_state
    last = 443.880615916  # (last + 600) - last rounds to 599.9999999999999
    server._last_activity = last
    assert server._release_if_idle(600, now=last + 600) is True
    assert dropped == [1]


# --- regression rows on a real store: b1 in the server process, and A's search after a release ----

def _small_store(path, n=40):
    """A synthetic store with chunk embeddings from the fake child (no model, no torch)."""
    import random

    from synapt.recall.core import TranscriptChunk, TranscriptIndex
    from synapt.recall.storage import RecallDB

    rng = random.Random(7)
    words = [f"w{i}" for i in range(300)] + ["store", "gr2", "review", "merge", "lease"]
    chunks = []
    for i in range(n):
        sess = f"sess-{i % 5:03d}"
        chunks.append(TranscriptChunk(
            id=f"{sess}:t{i}", session_id=sess, timestamp=f"2026-03-{1 + i % 28:02d}T10:{i % 60:02d}:00Z",
            turn_index=i, user_text=" ".join(rng.choices(words, k=20)),
            assistant_text=" ".join(rng.choices(words, k=40)), tools_used=["Bash"], files_touched=[]))
    path.mkdir(parents=True, exist_ok=True)
    db = RecallDB(path / "recall.db")
    db.save_chunks(chunks)
    index = TranscriptIndex(chunks, db=db)
    index.save(path)
    provider = ChildEmbeddings(cmd=[sys.executable, "-m", "synapt.recall.embed_child"])
    try:
        texts = [c.text[:500] for c in index._materialize_all_chunks()]
        vectors = provider.embed(texts)
    finally:
        provider.close()
    db.save_embeddings({index._idx_to_rowid[i]: v for i, v in enumerate(vectors) if index._idx_to_rowid.get(i) is not None})
    db.set_metadata("embedding_hash", index._content_hash())
    assert db.has_embeddings()
    return path


_B1_SERVER = textwrap.dedent("""
    import json, os, sys
    from pathlib import Path
    from synapt.recall import embeddings, server
    embeddings.use_child_embedder(os.environ["B1_SWITCH"] == "on")
    store = Path(sys.argv[1])
    server.project_index_dir = lambda: store
    index = server._get_index(use_embeddings=True)
    answer = index.lookup("gr2 review merge lease", max_chunks=5)
    provider = embeddings.get_embedding_provider()
    child = getattr(provider, "_proc", None)
    print(json.dumps({"pid": os.getpid(), "has_embeddings": server._cached_has_embeddings,
                      "child_started": child is not None, "answered": bool(answer),
                      "torch": "torch" in sys.modules, "st": "sentence_transformers" in sys.modules}))
    if hasattr(provider, "close"):
        provider.close()
""")


def _run_b1_server(tmp_path, switch):
    stubs = tmp_path / "stubs"
    marks = tmp_path / "marks"
    stubs.mkdir(exist_ok=True)
    marks.mkdir(exist_ok=True)
    for name in ("torch", "sentence_transformers"):
        (stubs / f"{name}.py").write_text(
            f"import os\nopen(os.path.join({str(marks)!r}, '{name}-' + str(os.getpid())), 'w').close()\n")
    env = {**os.environ, "RECALL_EMBED_CHILD_FAKE": "1", "B1_SWITCH": switch,
           "PYTHONPATH": os.pathsep.join([str(stubs), os.environ.get("PYTHONPATH", "")])}
    env.pop("RECALL_EMBED_CHILD_EXIT_ONCE", None)
    done = subprocess.run([sys.executable, "-c", _B1_SERVER, str(tmp_path / "store")],
                          env=env, capture_output=True, text=True, timeout=120)
    return done, sorted(p.name for p in marks.iterdir())


def test_b1_a_search_with_embeddings_never_imports_torch_in_the_server(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_EMBED_CHILD_FAKE", "1")
    _small_store(tmp_path / "store")
    done, marks = _run_b1_server(tmp_path, "on")
    assert done.returncode == 0, done.stderr[-2000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    # neither torch nor sentence-transformers was imported in the server process ...
    assert not report["torch"] and not report["st"], report
    assert not [m for m in marks if m.endswith(f"-{report['pid']}")], marks
    # ... and the semantic path ran, so the row cannot pass by skipping embeddings
    assert report["has_embeddings"] and report["child_started"] and report["answered"], report


def test_a_a_search_after_the_release_answers_as_before_it(tmp_path, monkeypatch):
    from synapt.recall import server

    monkeypatch.setenv("RECALL_EMBED_CHILD_FAKE", "1")
    store = _small_store(tmp_path / "store")
    for name, value in (("_cached_index", None), ("_cached_mtime", 0.0), ("_cached_dir", None),
                        ("_cached_has_embeddings", False), ("_calls_in_flight", 0), ("_last_activity", 0.0)):
        monkeypatch.setattr(server, name, value)
    monkeypatch.setattr(server, "project_index_dir", lambda: store)
    # The provider is chosen by whether sentence-transformers is installed here; the fake child never
    # imports it, so the row stands in for it rather than depending on the test env having it.
    present = embeddings._module_present
    monkeypatch.setattr(embeddings, "_module_present", lambda name: name == "sentence_transformers" or present(name))
    embeddings.use_child_embedder(True)
    try:
        before = server._get_index(use_embeddings=True).lookup("gr2 review merge lease", max_chunks=5)
        # a child that never answered is stopped (no process left), and the lookup falls back to keywords
        assert embeddings.get_embedding_provider()._proc is not None
        assert server._release_if_idle(1, now=10**9) is True  # the drop itself is the row above's
        reloaded = server._get_index(use_embeddings=True)
        after = reloaded.lookup("gr2 review merge lease", max_chunks=5)
        # the same answer could come from keyword search alone, so the embeddings must be back too
        assert reloaded._embed_provider is not None and reloaded._embedding_status == "active"
        assert reloaded._embed_provider._proc is not None
    finally:
        provider = embeddings.get_embedding_provider()
        if hasattr(provider, "close"):
            provider.close()
        embeddings.use_child_embedder(False)
    assert before and after == before


# --- the entry clients start: `synapt server` runs synapt/server.py:_serve, not this module's main ---

def _serve_with(monkeypatch, run, entry=None):
    """Drive synapt.server._serve() with plugins, the startup memory sync and the MCP loop stubbed;
    record each idle-release start (with the interval) and the order of the start and the run."""
    import synapt.plugins
    import synapt.server
    from synapt.recall import server as recall_server

    events = []
    monkeypatch.setattr(synapt.plugins, "register_plugins", lambda mcp: [])
    monkeypatch.setattr(recall_server, "_sync_claude_memory_source_on_startup", lambda: None)
    monkeypatch.setattr(recall_server, "_start_idle_release", lambda interval: events.append(("release", interval)))
    monkeypatch.setattr(recall_server.ValidatingFastMCP, "run", lambda self, *a, **k: run(events))
    try:
        (entry or synapt.server._serve)()
    finally:
        child = embeddings._use_child_embedder
        embeddings.use_child_embedder(False)
    return events, child


def test_a_synapt_server_starts_the_idle_release_before_it_runs(monkeypatch):
    monkeypatch.setenv("RECALL_IDLE_RELEASE_S", "42")
    events, child = _serve_with(monkeypatch, lambda events: events.append(("run",)))
    # the release thread is started once, with the configured interval, and before the server runs
    assert events == [("release", 42.0), ("run",)]
    # and torch goes to the child on this entry too, so the server process never holds the model
    assert child is True


def test_a_synapt_server_refuses_a_bad_idle_interval_before_it_builds_anything(monkeypatch):
    from synapt.recall import server as recall_server

    monkeypatch.setenv("RECALL_IDLE_RELEASE_S", "0")
    built = []
    monkeypatch.setattr(recall_server.ValidatingFastMCP, "__init__", lambda self, *a, **k: built.append(1))
    with pytest.raises(ValueError):
        _serve_with(monkeypatch, lambda events: events.append(("run",)))
    assert built == []


def test_a_synapt_server_that_stops_before_running_starts_no_release(monkeypatch):
    import synapt.plugins

    class Stop(Exception):
        pass

    def stop(mcp):
        raise Stop

    monkeypatch.delenv("RECALL_IDLE_RELEASE_S", raising=False)
    monkeypatch.setattr(synapt.plugins, "register_plugins", stop)
    events = []
    from synapt.recall import server as recall_server

    monkeypatch.setattr(recall_server, "_sync_claude_memory_source_on_startup", lambda: None)
    monkeypatch.setattr(recall_server, "_start_idle_release", lambda interval: events.append(("release", interval)))
    import synapt.server

    try:
        with pytest.raises(Stop):
            synapt.server._serve()
    finally:
        embeddings.use_child_embedder(False)
    assert events == []


def test_a_the_synapt_console_script_reaches_the_entry_that_starts_the_release(monkeypatch):
    # Resolve the name a client actually runs, from the installed console scripts, not from a
    # function picked by hand: `synapt server` -> the `synapt` script -> its server subcommand. A
    # refactor that routes the subcommand to a third entry without the release fails here, not as a
    # footprint at 17 h.
    from importlib.metadata import entry_points

    (script,) = entry_points(group="console_scripts", name="synapt")
    console_main = script.load()
    monkeypatch.setattr(sys, "argv", ["synapt", "server"])
    monkeypatch.setenv("RECALL_IDLE_RELEASE_S", "42")
    events, child = _serve_with(monkeypatch, lambda events: events.append(("run",)), entry=console_main)
    assert events == [("release", 42.0), ("run",)]
    assert child is True
