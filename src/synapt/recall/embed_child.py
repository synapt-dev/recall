"""The embedding child: holds the embedding model (and torch) out of the recall server's process.

The recall server never imports torch. When it needs vectors it sends one JSON line to this
child on stdin and reads one JSON line back on stdout:

    {"op": "dim"}                    -> {"dim": 384}
    {"op": "embed", "texts": [...]}  -> {"vectors": [[...], ...]}
    any failure                      -> {"error": "<type>: <message>"}

The child exits by itself after ``RECALL_EMBED_IDLE_S`` seconds (default 600) with no request,
and at once when its stdin closes, so a server that is killed leaves no child holding the model.

``RECALL_EMBED_CHILD_FAKE=1`` (tests only) answers with deterministic vectors and never loads a
model. ``RECALL_EMBED_CHILD_EXIT_ONCE=<path>`` (tests only) makes the first child that reads a
request exit without replying, the way a child exiting for idle meets a request in flight; it
creates <path>, so the next child answers.
"""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import sys

DEFAULT_IDLE_S = 600.0
FAKE_DIM = 384  # storage.EMBEDDING_DIM: fake vectors must be writable to a real store


def idle_seconds() -> float:
    """The idle interval from ``RECALL_EMBED_IDLE_S``; a missing value is the default."""
    raw = os.environ.get("RECALL_EMBED_IDLE_S", "").strip()
    if not raw:
        return DEFAULT_IDLE_S
    value = float(raw)  # a non-number raises, and the child exits with that error
    if not value > 0 or value != value or value == float("inf"):
        raise ValueError(f"RECALL_EMBED_IDLE_S must be a positive finite number, got {raw!r}")
    return value


class _FakeProvider:
    dim = FAKE_DIM

    def embed(self, texts):
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            out.append([digest[i % len(digest)] / 255.0 + i * 1e-4 for i in range(FAKE_DIM)])
        return out


def _provider():
    if os.environ.get("RECALL_EMBED_CHILD_FAKE") == "1":
        return _FakeProvider()
    from synapt.recall.embeddings import LocalEmbeddings, set_process_default_device

    set_process_default_device("cpu")  # the server's default (server.py main), so vectors match it
    return LocalEmbeddings()


def _answer(provider, request: dict) -> dict:
    op = request.get("op")
    if op == "dim":
        return {"dim": provider.dim}
    if op == "embed":
        texts = request.get("texts")
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            return {"error": "ValueError: texts must be a list of strings"}
        return {"vectors": provider.embed(texts)}
    return {"error": f"ValueError: unknown op {op!r}"}


def main() -> int:
    idle = idle_seconds()
    exit_once = os.environ.get("RECALL_EMBED_CHILD_EXIT_ONCE", "")
    provider = None
    selector = selectors.DefaultSelector()
    selector.register(sys.stdin, selectors.EVENT_READ)
    while True:
        if not selector.select(timeout=idle):
            return 0  # idle: give the model's memory back
        line = sys.stdin.readline()
        if not line:
            return 0  # the server closed our stdin (it exited or released us)
        if exit_once and not os.path.exists(exit_once):
            open(exit_once, "w").close()
            return 0  # test seam: exit with a request accepted and unanswered
        try:
            request = json.loads(line)
            if provider is None:
                provider = _provider()
            reply = _answer(provider, request)
        except Exception as exc:  # noqa: BLE001 - every failure goes back as a reply
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
