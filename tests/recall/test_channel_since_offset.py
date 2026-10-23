"""--since with a non-Z offset must compare instants, not strings."""

import json

from synapt.recall.channel import _count_messages_since, _read_messages

POSTED = "2026-10-01T12:00:00Z"
# 13:00+02:00 is 11:00Z, an hour BEFORE the post.
SINCE_EARLIER = "2026-10-01T13:00:00+02:00"


def _write(path):
    row = {"timestamp": POSTED, "from": "a", "channel": "dev", "type": "message", "body": "hi"}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_read_messages_since_offset_keeps_later_post(tmp_path):
    p = tmp_path / "dev.jsonl"
    _write(p)
    assert [m.timestamp for m in _read_messages(p, since=SINCE_EARLIER)] == [POSTED]


def test_count_messages_since_offset_skips_earlier_post(tmp_path):
    p = tmp_path / "dev.jsonl"
    _write(p)
    # 08:00-05:00 is 13:00Z, an hour AFTER the post: nothing is unread.
    assert _count_messages_since(p, "2026-10-01T08:00:00-05:00") == 0


def test_cli_read_refuses_unparseable_since_without_traceback(tmp_path):
    from tests.recall.test_channel_watch import Box

    home, project = tmp_path / "home", tmp_path / "proj"
    home.mkdir()
    project.mkdir()
    box = Box(project, home, tmp_path / "store")
    assert box.run("a", "join", "dev").returncode == 0
    box.post("b", "unread by a")
    state = ("from synapt.recall.channel import channel_unread, channel_has_unread; "
             "print(channel_unread(agent_name='a').get('dev'), "
             "channel_has_unread(agent_name='a').get('dev'))")
    before = box.py(state)
    assert before == "1 True", before
    r = box.run("a", "read", "dev", "--since", "garbage")
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert ("synapt recall channel read: refusing --since 'garbage': "
            "it is not an ISO 8601 timestamp") in r.stderr
    assert "Traceback" not in r.stderr
    # A refused read moves no cursor and clears no unread flag.
    assert box.py(state) == before


def test_stored_unparseable_cursor_counts_as_epoch(tmp_path):
    from tests.recall.test_channel_watch import Box

    home, project = tmp_path / "home", tmp_path / "proj"
    home.mkdir()
    project.mkdir()
    box = Box(project, home, tmp_path / "store")
    assert box.run("a", "join", "dev").returncode == 0
    box.post("b", "one")
    box.post("b", "two")
    counts = []
    for cursor in ("1970-01-01T00:00:00Z", "", "garbage"):
        r = box.py(
            "from synapt.recall.channel import _open_db, channel_unread; "
            "c = _open_db(None); "
            "c.execute('DELETE FROM cursors WHERE agent_id = ?', ('a',)); "
            f"c.execute('INSERT INTO cursors (agent_id, channel, last_read_at) VALUES (?, ?, ?)', ('a', 'dev', {cursor!r})); "
            "c.commit(); c.close(); "
            "print(channel_unread(agent_name='a').get('dev'))")
        counts.append(r)
    # The join notice counts too; what matters is each equals the epoch count.
    assert int(counts[0]) >= 2 and counts == [counts[0]] * 3, counts


def test_unread_read_with_a_stored_unparseable_cursor_reads_every_post(tmp_path):
    from tests.recall.test_channel_watch import Box

    home, project = tmp_path / "home", tmp_path / "proj"
    home.mkdir()
    project.mkdir()
    box = Box(project, home, tmp_path / "store")
    assert box.run("a", "join", "dev").returncode == 0
    box.post("b", "first unread")
    box.post("b", "second unread")
    # A stored cursor that is not a timestamp reaches channel_read through
    # channel_unread_read; it must read as the epoch, never refuse.
    out = box.py(
        "from synapt.recall.channel import _open_db, channel_unread_read; "
        "c = _open_db(None); "
        "c.execute('DELETE FROM cursors WHERE agent_id = ?', ('a',)); "
        "c.execute('INSERT INTO cursors (agent_id, channel, last_read_at) VALUES (?, ?, ?)', ('a', 'dev', 'garbage')); "
        "c.commit(); c.close(); "
        "print(channel_unread_read(agent_name='a'))")
    assert "refusing --since" not in out, out
    assert "first unread" in out and "second unread" in out, out
