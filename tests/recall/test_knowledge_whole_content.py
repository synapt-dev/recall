"""Knowledge nodes are stored whole, with a loud refuse above the limit.

Before this, ``KnowledgeNode.create`` cut every node to 300 characters at a
word boundary, without a word to the caller. A 1,200-character fact saved
through ``recall_save`` came back as its first 300 characters.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

from synapt.recall.knowledge import KnowledgeNode
from synapt.recall.server import ValidatingFastMCP, register_tools


def _call_tool(name: str, **kwargs):
    mcp = ValidatingFastMCP(f"test-knowledge-whole-{name}")
    register_tools(mcp)
    content_blocks, _structured = asyncio.run(mcp.call_tool(name, kwargs))
    return content_blocks[0].text


def _long(marker: str, n: int) -> str:
    words = []
    i = 0
    while len(" ".join(words)) < n:
        words.append(f"w{i}x")
        i += 1
    text = " ".join(words)
    body = text[: n - len(marker) - 1].rstrip()
    body = body + " " * (n - len(marker) - len(body))
    return body + marker


def _jsonl_contents(root):
    out = []
    for p in root.rglob("knowledge.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line).get("content", ""))
    return out


def _db_contents(root):
    out = []
    for p in root.rglob("*.db"):
        con = sqlite3.connect(p)
        try:
            out += [r[0] for r in con.execute("SELECT content FROM knowledge")]
        except sqlite3.Error:
            pass
        finally:
            con.close()
    return out


def _fts_hits(root, term):
    n = 0
    for p in root.rglob("*.db"):
        con = sqlite3.connect(p)
        try:
            n += con.execute(
                "SELECT count(*) FROM knowledge_fts WHERE knowledge_fts MATCH ?",
                (term,),
            ).fetchone()[0]
        except sqlite3.Error:
            pass
        finally:
            con.close()
    return n


def test_d1_long_save_reads_back_whole(owned_recall_root):
    content = _long("tailmarkerq7z", 1200)
    assert len(content) == 1200
    text = _call_tool("recall_save", content=content, category="workflow")
    assert not text.startswith("Error"), text
    search = _call_tool("recall_search", query="tailmarkerq7z")
    assert content in _jsonl_contents(owned_recall_root)
    assert content in _db_contents(owned_recall_root)
    assert _fts_hits(owned_recall_root, "tailmarkerq7z") >= 1
    assert "tailmarkerq7z" in search


def test_d2_over_limit_is_refused_and_writes_nothing(owned_recall_root):
    # was-there witness: the store exists and holds a node before the refusal
    ok = _call_tool("recall_save", content="witness fact wz3k", category="workflow")
    assert not ok.startswith("Error"), ok
    before_j, before_d = _jsonl_contents(owned_recall_root), _db_contents(owned_recall_root)
    assert "witness fact wz3k" in before_j
    text = _call_tool(
        "recall_save", content=_long("overmarkerk9", 4001), category="workflow"
    )
    assert text == (
        "Error: knowledge content is 4001 chars; the limit is 4000. Nothing was "
        "saved. Split the fact into smaller nodes."
    )
    assert _jsonl_contents(owned_recall_root) == before_j
    assert _db_contents(owned_recall_root) == before_d
    assert _fts_hits(owned_recall_root, "overmarkerk9") == 0


def test_d2_at_limit_is_saved(owned_recall_root):
    content = _long("edgemarkerv4", 4000)
    text = _call_tool("recall_save", content=content, category="workflow")
    assert not text.startswith("Error"), text
    assert content in _jsonl_contents(owned_recall_root)


# Ids pinned from recall dev 00b7da1 (KnowledgeNode.create with no node_id).
SHORT = "a short fact under three hundred characters"
CUT_EARLY = ("alpha " * 49) + "supercalifragilisticexpialidocious tail"
LONG = _long("idmarker", 2000)


def test_d3_auto_ids_match_pinned_dev_ids():
    assert len(CUT_EARLY) > 300  # word cut at 293, so [:300] would differ
    assert KnowledgeNode.create(SHORT, "workflow").id == "897976702879"
    assert KnowledgeNode.create(CUT_EARLY, "workflow").id == "0c76340f2fe8"
    assert KnowledgeNode.create(LONG, "workflow").id == "6ae4a0caee89"


def test_d2_create_refuses_over_limit_directly():
    """Consolidation calls create() without recall_save's check in front."""
    import pytest

    with pytest.raises(ValueError, match="4001 chars; the limit is 4000"):
        KnowledgeNode.create(_long("m", 4001), "workflow")
    assert len(KnowledgeNode.create(_long("m", 4000), "workflow").content) == 4000


# Every caller that turns text into a node refuses over the limit with nothing
# written: not the queue, not corrections.jsonl, not the node stores.

REFUSE_4001 = (
    "Error: knowledge content is 4001 chars; the limit is 4000. Nothing was "
    "saved. Split the fact into smaller nodes."
)


def _db_path(root):
    paths = [p for p in root.rglob("*.db") if "knowledge" in _tables(p)]
    assert len(paths) == 1, paths
    return paths[0]


def _tables(p):
    con = sqlite3.connect(p)
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
    finally:
        con.close()


def _pending(root):
    con = sqlite3.connect(_db_path(root))
    try:
        return [
            tuple(r)
            for r in con.execute(
                "SELECT id, old_node_id, length(new_content), status "
                "FROM pending_contradictions ORDER BY id"
            )
        ]
    finally:
        con.close()


def _seed(root):
    ok = _call_tool("recall_save", content="witness fact wz3k", category="workflow")
    assert not ok.startswith("Error"), ok
    con = sqlite3.connect(_db_path(root))
    try:
        return con.execute(
            "SELECT id FROM knowledge WHERE content = 'witness fact wz3k'"
        ).fetchone()[0]
    finally:
        con.close()


def _queue_raw(root, old_node_id, content):
    """A row queued before flag checked the length."""
    from synapt.recall.storage import RecallDB

    db = RecallDB(_db_path(root))
    try:
        return db.add_pending_contradiction(
            old_node_id=old_node_id, new_content=content, reason="", detected_by="manual"
        )
    finally:
        db.close()


def test_c1_flag_against_node_over_limit_queues_nothing(owned_recall_root):
    nid = _seed(owned_recall_root)
    before = (_jsonl_contents(owned_recall_root), _db_contents(owned_recall_root))
    text = _call_tool(
        "recall_contradict", action="flag", old_node_id=nid,
        new_content=_long("flagmarkq1", 4001),
    )
    assert text == REFUSE_4001
    assert _pending(owned_recall_root) == []
    assert (_jsonl_contents(owned_recall_root), _db_contents(owned_recall_root)) == before


def test_c1_flag_free_text_claim_over_limit_queues_nothing(owned_recall_root):
    _seed(owned_recall_root)
    text = _call_tool("recall_contradict", action="flag", claim=_long("zzunmatched", 4001))
    assert text == REFUSE_4001
    assert _pending(owned_recall_root) == []


def test_c1_flag_at_limit_is_queued(owned_recall_root):
    _seed(owned_recall_root)
    text = _call_tool("recall_contradict", action="flag", claim=_long("zzunmatched", 4000))
    assert text.startswith("Contradiction #"), text
    assert [r[2:] for r in _pending(owned_recall_root)] == [(4000, "pending")]


def test_c1_confirm_prequeued_over_limit_stays_pending(owned_recall_root):
    nid = _seed(owned_recall_root)
    cid = _queue_raw(owned_recall_root, nid, _long("confmarkq2", 4001))
    before = (_jsonl_contents(owned_recall_root), _db_contents(owned_recall_root))
    text = _call_tool("recall_contradict", action="resolve", contradiction_id=cid)
    assert text == REFUSE_4001 + f" Contradiction #{cid} stays pending."
    assert _pending(owned_recall_root) == [(cid, nid, 4001, "pending")]
    assert (_jsonl_contents(owned_recall_root), _db_contents(owned_recall_root)) == before


def test_c1_confirm_prequeued_free_text_over_limit_stays_pending(owned_recall_root):
    _seed(owned_recall_root)
    cid = _queue_raw(owned_recall_root, None, _long("confmarkq3", 4001))
    before = _db_contents(owned_recall_root)
    text = _call_tool("recall_contradict", action="resolve", contradiction_id=cid)
    assert text == REFUSE_4001 + f" Contradiction #{cid} stays pending."
    assert _pending(owned_recall_root) == [(cid, None, 4001, "pending")]
    assert _db_contents(owned_recall_root) == before


def test_c1_confirm_prequeued_free_text_at_limit_creates_node(owned_recall_root):
    _seed(owned_recall_root)
    content = _long("confmarkq4", 4000)
    cid = _queue_raw(owned_recall_root, None, content)
    text = _call_tool("recall_contradict", action="resolve", contradiction_id=cid)
    assert "new knowledge node created" in text, text
    assert _pending(owned_recall_root) == [(cid, None, 4000, "confirmed")]
    assert content in _db_contents(owned_recall_root)


def _corrections(root):
    paths = list(root.rglob("corrections.jsonl"))
    assert len(paths) <= 1, paths
    return paths[0].read_bytes() if paths else None


def test_c1_correct_over_limit_logs_nothing(owned_recall_root):
    ok = _call_tool(
        "recall_correct", question="q one", wrong_answer="w", correct_answer="right one"
    )
    assert ok.startswith("Correction captured") or "Error" not in ok, ok
    before = _corrections(owned_recall_root)
    assert before is not None and b"right one" in before  # was there
    before_d = _db_contents(owned_recall_root)
    # node content is f"{correct_answer} (re: {question})": 3989 + 12 = 4001
    text = _call_tool(
        "recall_correct", question="q two", wrong_answer="w",
        correct_answer=_long("corrmarkq5", 3989),
    )
    assert text == REFUSE_4001
    assert _corrections(owned_recall_root) == before
    assert _db_contents(owned_recall_root) == before_d


def test_c1_correct_over_limit_on_empty_store_leaves_no_file(owned_recall_root):
    text = _call_tool(
        "recall_correct", question="q two", wrong_answer="w",
        correct_answer=_long("corrmarkq5", 3989),
    )
    assert text == REFUSE_4001
    assert _corrections(owned_recall_root) is None
