"""Consolidate create keeps a long fact up to B3_CREATE_MAX_CHARS.

B3 (``_normalize_create_content``) and the B4 compose check share one ceiling,
B3_CREATE_MAX_CHARS. Content over it is cut by B3, and the cut is logged with
both lengths.
"""

from __future__ import annotations

import logging

import synapt.recall.consolidate as consolidate
from synapt.recall.journal import JournalEntry
from synapt.recall.knowledge import read_nodes


def _fact(n: int) -> str:
    out = ""
    i = 0
    while len(out) < n:
        out += (
            f"Step {i}: the ConnectionPool in synapt.recall.storage keeps socket "
            f"{i} in threading.local storage so worker {i} never shares it. "
        )
        i += 1
    return out[:n].rstrip()


def _cluster():
    return [
        JournalEntry(timestamp="2026-10-18T10:00:00Z", session_id=s, focus=s)
        for s in ("s1", "s2")
    ]


def _create(tmp_path, content):
    path = tmp_path / "knowledge.jsonl"
    consolidate._apply_consolidation_result(
        {"nodes": [{"action": "create", "content": content, "category": "workflow"}]},
        [],
        _cluster(),
        path,
    )
    return read_nodes(path)


def test_d4_consolidate_create_keeps_1200_chars_whole(tmp_path):
    fact = _fact(1200)
    assert len(fact) > 1000
    nodes = _create(tmp_path, fact)
    assert len(nodes) == 1
    assert nodes[0].content == fact


def test_d4_consolidate_create_over_ceiling_is_cut_and_logged(tmp_path, caplog):
    fact = _fact(2000)
    with caplog.at_level(logging.INFO, logger="synapt.recall.consolidate"):
        nodes = _create(tmp_path, fact)
    assert len(nodes) == 1
    stored = nodes[0].content
    assert 1000 < len(stored) <= consolidate.B3_CREATE_MAX_CHARS
    cuts = [r.getMessage() for r in caplog.records if "B3 create content cut" in r.getMessage()]
    assert len(cuts) == 1
    assert f"{len(fact)} chars ->" in cuts[0]
    assert f"-> {len(stored)} chars" in cuts[0]


def test_d4_no_cut_log_under_ceiling(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="synapt.recall.consolidate"):
        _create(tmp_path, _fact(1200))
    assert not [r for r in caplog.records if "B3 create content cut" in r.getMessage()]


def test_b1_b4_accepts_a_1200_char_composition():
    assert consolidate._b4_composed_content_is_safe(_fact(1200)) is True


def test_b1_b4_refuses_over_the_shared_ceiling():
    over = _fact(consolidate.B3_CREATE_MAX_CHARS + 200)
    assert len(over) > 1500
    assert consolidate._b4_composed_content_is_safe(over) is False


def test_d4_redaction_growth_stays_under_ceiling():
    # Each 17-char token scrubs to a longer marker; 1,500 chars of them grow past
    # the ceiling unless B3 cuts again after scrub. The token is built at runtime
    # so no literal in this file has a real token's shape.
    fake = "xo" + "xb-" + "1" * 12
    packed = (fake + " ") * 200
    out = consolidate._normalize_create_content(packed)
    assert fake not in out
    assert len(out) <= consolidate.B3_CREATE_MAX_CHARS
