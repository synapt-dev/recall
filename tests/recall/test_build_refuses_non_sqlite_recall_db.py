"""`synapt recall build` on a recall.db that is not a SQLite database refuses and leaves it untouched.

The file holds knowledge nodes, pending contradictions and the access log; a build cannot regenerate any
of them from transcripts. So the build must neither crash with a bare `file is not a database` traceback
nor reach `TranscriptIndex.load`'s repair, which unlinks a corrupt non-sharded recall.db.

Every row writes a corrupt recall.db, so the fixture asserts the resolved index dir is inside tmp_path
BEFORE the first write: a probe of this case once resolved to the user's real store and destroyed it.
"""

import argparse
import hashlib
import os
import sqlite3
from pathlib import Path
from unittest import mock

import pytest

from conftest import assistant_entry, user_text_entry, write_jsonl
from synapt.recall import cli
from synapt.recall.core import project_index_dir

GARBAGE = b"this is not a sqlite database\n" * 50


@pytest.fixture
def corrupt_store(tmp_path, monkeypatch):
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    project = tmp_path / "proj"
    project.mkdir()
    source = tmp_path / "src"
    source.mkdir()
    write_jsonl(source / "s1.jsonl", [
        user_text_entry("deploy the kubernetes rollout canary staging", uuid="u1", ts="2026-03-01T10:00:00Z"),
        assistant_entry(text="rollout canary staging manifest helm chart release", uuid="a1", ts="2026-03-01T10:00:30Z"),
    ])
    index_dir = project_index_dir(project.resolve())
    assert str(index_dir.resolve()).startswith(str(tmp_path.resolve()) + os.sep), index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    db = index_dir / "recall.db"
    db.write_bytes(GARBAGE)
    return dict(project=project, source=source, index_dir=index_dir, db=db)


def _listing(index_dir: Path):
    return sorted(p.name for p in index_dir.iterdir() if p.name.startswith("recall.db") and not p.name.endswith(".lock"))


def _cmd_build(store):
    args = argparse.Namespace(source=[str(store["source"])], hf=None, chatgpt_archive=None,
                              no_embeddings=True, incremental=False)
    with mock.patch("synapt.recall.cli.Path.cwd", return_value=store["project"]):
        cli.cmd_build(args)


def test_build_refuses_with_a_named_message_and_rc_2(corrupt_store, capsys):
    with pytest.raises(SystemExit) as exc:
        _cmd_build(corrupt_store)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert str(corrupt_store["db"]) in err, err
    assert "file is not a database" in err, err
    assert "move it aside and keep it" in err, err
    assert "Traceback" not in err, err


def test_the_corrupt_file_is_byte_unchanged_and_nothing_is_added_or_removed(corrupt_store):
    before = hashlib.sha256(corrupt_store["db"].read_bytes()).hexdigest()
    with pytest.raises(SystemExit):
        _cmd_build(corrupt_store)
    assert corrupt_store["db"].exists(), "the build deleted the corrupt recall.db"
    assert hashlib.sha256(corrupt_store["db"].read_bytes()).hexdigest() == before
    assert _listing(corrupt_store["index_dir"]) == ["recall.db"]


def test_the_build_never_reaches_loads_repair(corrupt_store):
    with mock.patch.object(cli.TranscriptIndex, "load", side_effect=AssertionError("reached load's repair")):
        with pytest.raises(SystemExit) as exc:
            _cmd_build(corrupt_store)
    assert exc.value.code == 2


def test_library_callers_get_the_named_error_which_is_still_a_database_error(corrupt_store):
    with pytest.raises(cli.CorruptRecallDBError) as exc:
        cli._archive_and_build(corrupt_store["project"], source_dirs=[corrupt_store["source"]],
                               use_embeddings=False)
    assert isinstance(exc.value, sqlite3.DatabaseError)
    assert corrupt_store["db"].read_bytes() == GARBAGE


def test_control_the_same_fixture_with_no_corrupt_file_builds(corrupt_store, capsys):
    corrupt_store["db"].unlink()
    _cmd_build(corrupt_store)
    assert "[build] Done!" in capsys.readouterr().out
    with sqlite3.connect(corrupt_store["db"]) as conn:
        conn.execute("select count(*) from sqlite_master").fetchone()


@pytest.mark.parametrize("message", ["database is locked", "disk I/O error"])
def test_an_operational_error_on_a_healthy_store_keeps_its_own_error(corrupt_store, capsys, message):
    # OperationalError subclasses DatabaseError; a locked or I/O-failing store is not corrupt, and the
    # refusal's "move it aside" would send a user to move a healthy store. A real lock waits out the
    # 30 s busy timeout, so the open raises the error the lock produces.
    corrupt_store["db"].unlink()
    with mock.patch("synapt.recall.storage.RecallDB", side_effect=sqlite3.OperationalError(message)):
        with pytest.raises(sqlite3.OperationalError) as exc:
            _cmd_build(corrupt_store)
    assert not isinstance(exc.value, cli.CorruptRecallDBError)
    assert str(exc.value) == message
    assert "move it aside" not in capsys.readouterr().err


def _healthy_store_with_a_legacy_access_row(store):
    from synapt.recall.storage import RecallDB

    store["db"].unlink()
    RecallDB(store["db"]).close()
    with sqlite3.connect(store["db"]) as conn:
        conn.execute("INSERT INTO access_log (item_type, item_id, query, score, session_id, context, created_at) "
                     "VALUES ('chunk', 'c1', 'q', 1.0, 's1', 'ctx', '2026-03-01T10:00:00Z')")
    (store["index_dir"] / "access.db").unlink(missing_ok=True)


def test_control_a_legacy_access_row_moves_out_on_open(corrupt_store):
    _healthy_store_with_a_legacy_access_row(corrupt_store)
    cli._open_build_db(corrupt_store["db"]).close()
    with sqlite3.connect(corrupt_store["db"]) as conn:
        assert conn.execute("SELECT count(*) FROM access_log").fetchone()[0] == 0


def test_a_broken_access_db_beside_a_healthy_store_is_not_blamed_on_recall_db(corrupt_store, capsys):
    # The open moves legacy access rows into access.db; a non-SQLite access.db raises DatabaseError from inside the
    # open. recall.db reads, so the refusal must not tell the user to move it aside.
    _healthy_store_with_a_legacy_access_row(corrupt_store)
    (corrupt_store["index_dir"] / "access.db").write_bytes(GARBAGE)
    with pytest.raises(sqlite3.DatabaseError) as exc:
        _cmd_build(corrupt_store)
    assert not isinstance(exc.value, cli.CorruptRecallDBError), exc.value
    assert "move it aside" not in capsys.readouterr().err
