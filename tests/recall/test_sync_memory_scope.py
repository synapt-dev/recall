"""recall_sync_memory syncs the calling seat's OWN memory directory, and names every file it could not sync.

Before this, one seat's sync globbed ``*/memory`` under every Claude Code root and wrote every other
desk's memory files into its own store (measured on a desk host: 4,275 files across 37 projects, from a
seat whose own directory held 21). A failure was an exception swallowed into a count, with no file and no
reason, and the run still reported "complete": 318 of those failures were YAML frontmatter that would not
parse, which a named error would have shown in one line.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from synapt.recall.core import project_index_dir, project_slug
from synapt.recall.storage import RecallDB


def _memory_file(memory_dir, name, description="d"):
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / f"{name}.md").write_text(
        f"---\nname: {name}\ndescription: {description}\ntype: project\n---\nbody of {name}\n",
        encoding="utf-8",
    )


def _sync(home, project, **kwargs):
    from synapt.recall.server import recall_sync_memory

    with patch("pathlib.Path.home", return_value=home), \
         patch("synapt.recall.server.Path.cwd", return_value=project), \
         patch("synapt.recall.server.get_embedding_provider", return_value=None), \
         patch("synapt.recall.server._invalidate_cache"):
        return recall_sync_memory(**kwargs)


def _stored_names(project):
    db = RecallDB(project_index_dir(project) / "recall.db")
    try:
        return sorted(
            tag[len("name:"):]
            for node in db.load_knowledge_nodes(status="active")
            for tag in node["tags"]
            if tag.startswith("name:")
        )
    finally:
        db.close()


@pytest.fixture
def seats(tmp_path, monkeypatch):
    """Two seats' projects, each with its own memory dir under the configured root."""
    config = tmp_path / "accounts" / "core"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    seat_a = tmp_path / "seat-a"
    seat_b = tmp_path / "seat-b"
    seat_a.mkdir()
    seat_b.mkdir()
    a_dir = config / "projects" / project_slug(seat_a) / "memory"
    b_dir = config / "projects" / project_slug(seat_b) / "memory"
    return tmp_path / "home", config, seat_a, seat_b, a_dir, b_dir


def test_a_seat_syncs_only_its_own_memory_dir(seats):
    home, _config, seat_a, _seat_b, a_dir, b_dir = seats
    _memory_file(a_dir, "mine")
    _memory_file(b_dir, "another-desks")

    result = _sync(home, seat_a)

    assert "1 synced" in result, result
    assert _stored_names(seat_a) == ["mine"]
    assert str(a_dir) in result, result
    assert str(b_dir) not in result, result


def test_a_shared_gripspace_root_does_not_redirect_the_sync_to_another_seat(seats, monkeypatch):
    """Sibling desks share one GRIPSPACE_ROOT by design; the memory dir is the launch dir's."""
    home, _config, seat_a, seat_b, a_dir, b_dir = seats
    _memory_file(a_dir, "mine")
    _memory_file(b_dir, "the-shared-roots")
    monkeypatch.setenv("GRIPSPACE_ROOT", str(seat_b))

    result = _sync(home, seat_a)

    assert "1 synced" in result, result
    assert str(a_dir) in result and str(b_dir) not in result, result


def test_the_own_dir_is_read_under_every_claude_root(seats):
    home, _config, seat_a, _seat_b, a_dir, _b_dir = seats
    _memory_file(a_dir, "configured-root")
    _memory_file(home / ".claude" / "projects" / project_slug(seat_a) / "memory", "default-root")

    result = _sync(home, seat_a)

    assert "2 synced" in result, result
    assert _stored_names(seat_a) == ["configured-root", "default-root"]


def test_a_seat_with_no_memory_dir_says_so(seats):
    home, _config, seat_a, _seat_b, _a_dir, b_dir = seats
    _memory_file(b_dir, "theirs")

    result = _sync(home, seat_a)

    assert "No Claude Code memory directory" in result, result
    assert project_slug(seat_a) in result, result
    assert not (project_index_dir(seat_a) / "recall.db").exists() or _stored_names(seat_a) == []


def test_named_dirs_sync_only_those(seats, tmp_path):
    home, _config, seat_a, _seat_b, a_dir, b_dir = seats
    _memory_file(a_dir, "mine")
    _memory_file(b_dir, "named")

    result = _sync(home, seat_a, memory_dirs=[str(b_dir)])

    assert "1 synced" in result, result
    assert _stored_names(seat_a) == ["named"]


def test_a_named_dir_that_does_not_exist_is_an_error_naming_it(seats, tmp_path):
    home, _config, seat_a, _seat_b, a_dir, _b_dir = seats
    _memory_file(a_dir, "mine")
    missing = tmp_path / "no-such" / "memory"

    with pytest.raises(RuntimeError) as err:
        _sync(home, seat_a, memory_dirs=[str(a_dir), str(missing)])

    assert str(missing) in str(err.value)
    assert "1 synced" in str(err.value)
    assert _stored_names(seat_a) == ["mine"]


def test_a_file_that_fails_is_named_with_its_reason_and_the_rest_still_sync(seats):
    """Apollo's shape: the good files are written FIRST, then the run raises (isError over MCP)."""
    home, _config, seat_a, _seat_b, a_dir, _b_dir = seats
    _memory_file(a_dir, "good-one")
    _memory_file(a_dir, "good-two")
    # An unquoted ': ' inside a value: the commonest failure measured on a real desk (311 of 318).
    _memory_file(a_dir, "bad-yaml", description="a rule: with a colon")

    with pytest.raises(RuntimeError) as err:
        _sync(home, seat_a)

    message = str(err.value)
    assert "2 synced" in message and "1 error" in message, message
    assert str(a_dir / "bad-yaml.md") in message, message
    assert "ScannerError" in message, message
    assert _stored_names(seat_a) == ["good-one", "good-two"]


def test_a_save_that_refuses_is_named_with_the_refusal(seats):
    home, _config, seat_a, _seat_b, a_dir, _b_dir = seats
    _memory_file(a_dir, "refused")

    with patch("synapt.recall.server.recall_save", return_value="Refused: store is read-only"):
        with pytest.raises(RuntimeError) as err:
            _sync(home, seat_a)

    message = str(err.value)
    assert str(a_dir / "refused.md") in message, message
    assert "Refused: store is read-only" in message, message
