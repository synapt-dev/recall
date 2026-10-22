"""recall_contradict(action="list") shows a bounded, ordered slice with a footer."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from synapt.recall.archive import _has_index, _load_index_state
from synapt.recall.server import recall_contradict
from synapt.recall.storage import RecallDB


def _store(tmp_path, n):
    index = tmp_path / ".synapt" / "recall" / "index"
    index.mkdir(parents=True)
    db = RecallDB(index / "recall.db")
    # Seed in reverse so ids run opposite to claim numbers.
    for i in reversed(range(n)):
        cid = db.add_pending_contradiction(None, f"new {i:02d}\nsecond line", claim_text=f"claim {i:02d}")
        # Equal dates for claims 00 and 01: only the id can order them.
        day = max(i, 1)
        db._conn.execute(
            "UPDATE pending_contradictions SET detected_at = ? WHERE id = ?",
            (f"2026-10-{day:02d}T00:00:00", cid),
        )
    db._conn.commit()
    return db


def _list(db, **kw):
    with patch("synapt.recall.server._get_index", return_value=SimpleNamespace(_db=db)):
        return recall_contradict(action="list", **kw)


def test_twelve_pending_default_lists_five_oldest_with_footer(tmp_path):
    out = _list(_store(tmp_path, 12))
    assert out.count("5 of 12 shown, 7 withheld (limit=0 lists all)") == 1
    assert out.count("Claim: claim") == 5
    assert out.index("claim 01") < out.index("claim 00") < out.index("claim 04")
    assert out.count("claim 05") == 0


def test_order_newest(tmp_path):
    out = _list(_store(tmp_path, 12), order="newest")
    assert out.count("Claim: claim") == 5
    assert out.index("claim 11") < out.index("claim 07")
    assert out.count("claim 06") == 0


def test_explicit_limit(tmp_path):
    out = _list(_store(tmp_path, 12), limit=3)
    assert out.count("Claim: claim") == 3
    assert out.count("3 of 12 shown, 9 withheld") == 1


def test_limit_zero_returns_all_without_footer(tmp_path):
    out = _list(_store(tmp_path, 12), limit=0)
    assert out.count("Claim: claim") == 12
    assert out.count(" shown, ") == 0


def test_under_limit_has_no_footer(tmp_path):
    out = _list(_store(tmp_path, 3))
    assert out.count("Claim: claim") == 3
    assert out.count(" shown, ") == 0


def test_one_line_per_side(tmp_path):
    out = _list(_store(tmp_path, 1))
    assert out.count("New: new 00 second line") == 1


def test_unknown_order_is_an_error(tmp_path):
    out = _list(_store(tmp_path, 3), order="sideways")
    assert out.count("Error: order must be 'oldest' or 'newest'") == 1


def test_negative_limit_is_an_error(tmp_path):
    out = _list(_store(tmp_path, 3), limit=-1)
    assert out.count("Error: limit must be >= 0") == 1


def test_storage_still_sees_every_row(tmp_path):
    db = _store(tmp_path, 12)
    _list(db)
    index_dir = tmp_path / ".synapt" / "recall" / "index"
    assert _has_index(index_dir), index_dir
    _, _, _, pending = _load_index_state(index_dir)
    assert len(pending) == 12


@pytest.mark.parametrize("flip", [False, True], ids=["storage-as-is", "storage-reversed"])
@pytest.mark.parametrize("order,first,second", [("oldest", "claim 01", "claim 00"), ("newest", "claim 00", "claim 01")])
def test_equal_dates_tie_break_on_id_whatever_storage_order(tmp_path, monkeypatch, flip, order, first, second):
    db = _store(tmp_path, 3)
    real = db.list_pending_contradictions

    def listed(*a, **kw):
        rows = list(real(*a, **kw))
        return rows[::-1] if flip else rows

    monkeypatch.setattr(db, "list_pending_contradictions", listed)
    out = _list(db, order=order, limit=0)
    assert out.index(first) < out.index(second)
