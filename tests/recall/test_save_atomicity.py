"""A save that reports failure must not leave a searchable node.

Measured before this fix, on recall dev and on the installed 0.25.4: with a
provider whose vectors are not the store's width, `recall_save` returned
"Knowledge save failed: pack expected 384 items for packing (got 1024)" while
the node row was already persisted and `recall_search` found it. The report was
accurate; the store is what lied, because the node is committed before the step
that can fail.

The witness below fails on the pre-fix ordering and passes once the
failure-prone work (computing the vector) happens BEFORE the node is committed.
It asserts on the STORE rather than on a follow-up search, because "no node
landed" is the property, and a search is one way to observe it.
"""

from __future__ import annotations

import os
import pytest

from synapt.recall.embeddings import EmbeddingProvider

MARKER = "save-atomicity-witness-marker"


class _BoomyProvider(EmbeddingProvider):
    """A provider that answers the store's width and then fails on the embed.

    This is the shape of the real failure: the write is attempted and raises.
    """

    @property
    def dim(self):
        from synapt.recall.storage import EMBEDDING_DIM

        return EMBEDDING_DIM

    def embed(self, texts):
        raise RuntimeError("pack expected 384 items for packing (got 1024)")


def _store_holds(marker: str, root) -> bool:
    """Is the marker in ANY knowledge.jsonl under root?"""
    for path in root.rglob("knowledge.jsonl"):
        if marker in path.read_text(errors="ignore"):
            return True
    return False


def test_failed_embedding_leaves_no_node_behind(tmp_path, monkeypatch):
    """THE WITNESS. Pre-fix: the node lands and this fails."""
    monkeypatch.setenv("GRIPSPACE_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    import synapt.recall.server as server

    monkeypatch.setattr(server, "get_embedding_provider", lambda: _BoomyProvider())

    result = server.recall_save(content=MARKER, category="decision")

    assert result.startswith("Knowledge save failed"), (
        f"the save must report the failure it hit; got {result[:120]!r}"
    )
    assert not _store_holds(MARKER, tmp_path), (
        "a save that reported failure left the node in the store: the node row "
        "is committed before the step that can fail, and nothing undoes it"
    )
    # AND THE OTHER HALF OF THE SYMPTOM: search must not report FOUND either.
    found = server.recall_search(MARKER, include_historical=True, min_score=0.0)
    assert MARKER not in found, (
        "search reported FOUND after a save that failed: "
        f"{found[:160]!r}"
    )


def test_successful_save_still_lands_and_embeds(tmp_path, monkeypatch):
    """THE CONTROL, without which a save that never writes anything would pass
    the witness above."""
    monkeypatch.setenv("GRIPSPACE_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    import synapt.recall.server as server
    from synapt.recall.storage import EMBEDDING_DIM

    class _Good(EmbeddingProvider):
        @property
        def dim(self):
            return EMBEDDING_DIM

        def embed(self, texts):
            return [[0.0] * EMBEDDING_DIM for _ in texts]

    monkeypatch.setattr(server, "get_embedding_provider", lambda: _Good())

    result = server.recall_save(content=MARKER, category="decision")

    assert result.startswith("Knowledge node saved"), f"got {result[:120]!r}"
    assert _store_holds(MARKER, tmp_path), "a successful save must land the node"

    # AND THE VECTOR IS STORED AGAINST THE RIGHT ROW. A reorder that landed the
    # node but lost the embedding would still pass a node-only assertion.
    from synapt.recall.sharding import live_store_path
    from synapt.recall.storage import RecallDB

    node_id = result.split(": ", 1)[1].split(" ", 1)[0]
    db = RecallDB(live_store_path(server.project_index_dir(None)))
    try:
        rowid = db.get_knowledge_rowid(node_id)
        assert rowid is not None, f"no row for the node the save reported: {node_id}"
        stored = db.get_knowledge_embeddings_by_id()
        assert node_id in stored, (
            f"the node landed ({node_id}, rowid {rowid}) but its vector is not "
            f"stored against it; stored ids: {sorted(stored)[:5]}"
        )
    finally:
        db.close()


def test_mutation_moving_the_embedding_back_reddens_the_witness(tmp_path):
    """The mutation, on the REAL source bytes: put the embed back after the node
    insert -- the pre-fix ordering -- and the defect must return.

    The reorder's occurrence is asserted FIRST, so a mutation that silently
    failed to apply cannot read as a passing red.
    """
    import shutil
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parents[2]
    src_path = repo / "src" / "synapt" / "recall" / "server.py"
    src = src_path.read_text()

    # ANCHOR ON THE FULL LINE, INDENTATION INCLUDED. Anchoring on the bare
    # text leaves the 12 spaces of indentation in src[:i] and the
    # replacement adds 12 more -- an IndentationError that masquerades as
    # the mutation working.
    marker = "            # COMPUTE THE VECTOR FIRST."
    assert src.count(marker) == 1, (
        f"the reorder must be present exactly once to mutate away from, got "
        f"{src.count(marker)}"
    )

    # Replace the reordered region (vector first) with the PRE-FIX shape (node
    # first, embed after). Sliced from the marker to the next '        finally:'
    # AFTER it -- an unanchored search finds an earlier finally: and yields an
    # empty span, which is how the first attempt of this mutation produced a
    # syntax error instead of the defect.
    i = src.index(marker)
    j = src.index("        finally:", i)
    assert j > i and "save_knowledge_node(" in src[i:j], "the span must cover the reorder"

    prefix_shape = (
        "            save_knowledge_node(\n"
        "                node, project_data_dir(project) / \"knowledge.jsonl\", "
        "project_index_dir(project)\n"
        "            )\n"
        "\n"
        "            embedded = False\n"
        "            provider = get_embedding_provider()\n"
        "            if provider:\n"
        "                rowid = db.get_knowledge_rowid(node.id)\n"
        "                if rowid is not None:\n"
        "                    embedding = provider.embed_single(node.content[:500])\n"
        "                    db.save_knowledge_embeddings({rowid: embedding})\n"
        "                    embedded = True\n"
        "\n"
    )
    mutated = src[:i] + prefix_shape + src[j:]
    assert mutated != src, "the mutation must change the bytes"
    assert marker not in mutated, "the reorder must actually be gone"
    compile(mutated, "<mutated>", "exec")  # it must PARSE, or a syntax error masquerades as the defect

    # Build the whole package in a temp tree so the mutation resolves its imports.
    tree = tmp_path / "prefix_src"
    shutil.copytree(repo / "src", tree)
    (tree / "synapt" / "recall" / "server.py").write_text(mutated)

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os, sys\n"
        "root = sys.argv[1]\n"
        "os.environ['GRIPSPACE_ROOT'] = root\n"
        "os.makedirs(root, exist_ok=True)\n"
        "os.chdir(root)\n"
        "import synapt.recall.server as server\n"
        "from synapt.recall.embeddings import EmbeddingProvider\n"
        "class Boom(EmbeddingProvider):\n"
        "    @property\n"
        "    def dim(self):\n"
        "        from synapt.recall.storage import EMBEDDING_DIM\n"
        "        return EMBEDDING_DIM\n"
        "    def embed(self, texts):\n"
        "        raise RuntimeError('boom')\n"
        "server.get_embedding_provider = lambda: Boom()\n"
        "r = server.recall_save(content=sys.argv[2], category='decision')\n"
        "landed = any(sys.argv[2] in p.read_text(errors='ignore')\n"
        "             for p in __import__('pathlib').Path(root).rglob('knowledge.jsonl'))\n"
        "print('FAILED_SAVE_LEFT_NODE' if landed else 'no node', r[:40])\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    res = subprocess.run(
        [_sys.executable, str(probe), str(tmp_path / "kitroot"), MARKER],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert "FAILED_SAVE_LEFT_NODE" in res.stdout, (
        "with the embedding moved back after the insert, the failed save must "
        f"leave the node again -- that IS the defect; stdout={res.stdout!r} "
        f"stderr={res.stderr[-400:]!r}"
    )
