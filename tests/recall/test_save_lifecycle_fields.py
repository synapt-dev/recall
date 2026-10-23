"""recall_save(node_id=X) keeps a node's lifecycle fields.

An update carries status, superseded_by and contradiction_note. A contradicted
or superseded node is refused, naming its successor, unless restore=true.
restore and restore_retracted each open only their own door.
"""

from __future__ import annotations

import asyncio
import sqlite3


from synapt.recall.server import ValidatingFastMCP, register_tools

FACT = "The release runner pins python 3.12 because 3.13 breaks the wheel build."
NEW = "The release runner pins python 3.12 because 3.13 breaks the sdist build step."


def _call(name, **kw):
    mcp = ValidatingFastMCP(f"test-save-lifecycle-{name}")
    register_tools(mcp)
    return asyncio.run(mcp.call_tool(name, kw))[0][0].text


def _dbs():
    from synapt.recall.core import project_index_dir

    return list(project_index_dir().parent.rglob("*.db"))


def _set(root, node_id, **cols):
    n = 0
    for p in _dbs():
        con = sqlite3.connect(p)
        try:
            sets = ", ".join(f"{k} = ?" for k in cols)
            n += con.execute(
                f"UPDATE knowledge SET {sets} WHERE id = ?", (*cols.values(), node_id)
            ).rowcount
            con.commit()
        except sqlite3.Error:
            pass
        finally:
            con.close()
    assert n >= 1


def _row(root, node_id):
    for p in _dbs():
        con = sqlite3.connect(p)
        con.row_factory = sqlite3.Row
        try:
            r = con.execute("SELECT * FROM knowledge WHERE id = ?", (node_id,)).fetchone()
            if r:
                return dict(r)
        except sqlite3.Error:
            pass
        finally:
            con.close()
    return None


def _seed(root, **cols):
    text = _call("recall_save", content=FACT, category="workflow", node_id="n1")
    assert not text.startswith("Error"), text
    if cols:
        _set(root, "n1", **cols)


def test_a1_update_carries_lifecycle_fields(owned_recall_root):
    _seed(owned_recall_root, status="stale", superseded_by="", contradiction_note="aged out")
    text = _call("recall_save", content=NEW, node_id="n1")
    assert not text.startswith("Error"), text
    row = _row(owned_recall_root, "n1")
    assert row["content"] == NEW
    assert row["status"] == "stale"
    assert row["contradiction_note"] == "aged out"


def test_a2_contradicted_node_update_is_refused_naming_successor(owned_recall_root):
    _seed(owned_recall_root, status="contradicted", superseded_by="succ42", contradiction_note="wrong")
    text = _call("recall_save", content=NEW, node_id="n1")
    assert text.startswith("Error"), text
    assert "succ42" in text
    row = _row(owned_recall_root, "n1")
    assert row["content"] == FACT
    assert row["status"] == "contradicted"


def test_a2_superseded_node_update_is_refused_naming_successor(owned_recall_root):
    _seed(owned_recall_root, status="superseded", superseded_by="succ77")
    text = _call("recall_save", content=NEW, node_id="n1")
    assert text.startswith("Error") and "succ77" in text, text
    row = _row(owned_recall_root, "n1")
    assert row["content"] == FACT
    # still superseded after the update attempt, successor kept
    assert row["status"] == "superseded"
    assert row["superseded_by"] == "succ77"


def test_a2_superseded_status_alone_is_refused(owned_recall_root):
    # status alone marks the node replaced, with no successor recorded
    _seed(owned_recall_root, status="superseded", superseded_by="")
    text = _call("recall_save", content=NEW, node_id="n1")
    assert text.startswith("Error") and "(none recorded)" in text, text
    assert _row(owned_recall_root, "n1")["content"] == FACT


def test_a2_superseded_by_alone_is_refused(owned_recall_root):
    # a recorded successor alone marks the node replaced, whatever its status
    _seed(owned_recall_root, superseded_by="succ88")
    text = _call("recall_save", content=NEW, node_id="n1")
    assert text.startswith("Error") and "succ88" in text, text
    assert _row(owned_recall_root, "n1")["content"] == FACT


def test_a3_restore_revives_a_contradicted_node(owned_recall_root):
    _seed(owned_recall_root, status="contradicted", superseded_by="succ42", contradiction_note="wrong")
    text = _call("recall_save", content=NEW, node_id="n1", restore=True)
    assert "re-activated" in text, text
    row = _row(owned_recall_root, "n1")
    assert row["content"] == NEW
    assert row["status"] == "active"
    assert not row["superseded_by"]
    assert not row["contradiction_note"]


def test_pair_restore_retracted_does_not_open_a_contradicted_node(owned_recall_root):
    _seed(owned_recall_root, status="contradicted", superseded_by="succ42")
    text = _call("recall_save", content=NEW, node_id="n1", restore_retracted=True)
    assert text.startswith("Error") and "succ42" in text, text
    assert _row(owned_recall_root, "n1")["content"] == FACT


def test_pair_restore_does_not_open_a_retracted_node(owned_recall_root):
    _seed(owned_recall_root)
    assert not _call("recall_save", node_id="n1", retract=True).startswith("Error")
    text = _call("recall_save", content=NEW, node_id="n1", restore=True)
    assert text.startswith("Error") and "retracted" in text, text
    assert _row(owned_recall_root, "n1")["content"] == FACT
