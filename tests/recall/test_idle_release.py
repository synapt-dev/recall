"""The recall server gives memory back when idle.

B: the server embeds through `ChildEmbeddings`, a child process that holds the model and exits
when idle, so torch never enters the server. A: after an idle interval with no tool call, the
server drops its index cache. These rows run the child in its fake mode (no model, no torch); the
rows that need the real model (b2 vectors, b5 cold start, the memory bar) run on a box.
"""

from __future__ import annotations

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
