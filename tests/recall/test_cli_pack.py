"""cmd_pack's --verify path must never depend on the current project's
transcript directory: verification only reads the pack + idx already under
the given --index, so a caller re-verifying an index from a cwd with no
transcripts of its own must not be refused before verification even starts."""
from argparse import Namespace
from pathlib import Path
from unittest import mock

import pytest

from synapt.recall import transcript_pack as tp
from synapt.recall.cli import cmd_pack


def _write_session(project_dir: Path, name: str, lines: list[str]) -> Path:
    p = project_dir / f"{name}.jsonl"
    p.write_text("".join(line + "\n" for line in lines))
    return p


def test_verify_succeeds_from_a_cwd_with_no_transcript_directory(tmp_path, capsys):
    # seal from a real project dir first, using the module directly
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    index_dir = tmp_path / "index"
    _write_session(project_dir, "closed-a", ['{"turn": 1}'])
    tp.seal_closed_sessions(project_dir, index_dir)

    # now invoke --verify with cwd resolving to NO transcript directory at
    # all -- project_transcript_dir(Path.cwd()) must never be consulted on
    # this path
    empty_cwd = tmp_path / "nowhere"
    empty_cwd.mkdir()
    with mock.patch(
        "synapt.recall.core.project_transcript_dir",
        side_effect=AssertionError(
            "verify must not resolve a project transcript dir before verifying"
        ),
    ), mock.patch("synapt.recall.cli.Path.cwd", return_value=empty_cwd):
        cmd_pack(Namespace(index=str(index_dir), out=None, verify=True))

    out = capsys.readouterr().out
    assert "Verified 1 OK, 0 failed." in out


def test_seal_still_refuses_a_cwd_with_no_transcript_directory(tmp_path):
    # control: the seal path (no --verify) must still refuse cleanly when
    # there genuinely is no project transcript directory -- only the verify
    # path is exempted by the fix
    index_dir = tmp_path / "index"
    empty_cwd = tmp_path / "nowhere"
    empty_cwd.mkdir()
    with mock.patch(
        "synapt.recall.core.project_transcript_dir", return_value=None
    ), mock.patch("synapt.recall.cli.Path.cwd", return_value=empty_cwd), pytest.raises(
        SystemExit
    ) as exc:
        cmd_pack(Namespace(index=str(index_dir), out=None, verify=False))
    assert exc.value.code == 1


def test_seal_exits_nonzero_when_any_receipt_failed_verification(tmp_path, monkeypatch):
    """Fix-forward: a seal receipt with verify=FAILED must not be reported as
    a quiet success line -- the caller needs a nonzero exit to notice."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    index_dir = tmp_path / "index"
    _write_session(project_dir, "closed-a", ['{"turn": 1}'])

    bad_receipt = tp.SealReceipt(
        session_id="closed-a",
        sha256="0" * 64,
        raw_length=1,
        compressed_length=1,
        line_count=1,
        pack_offset=0,
        verified=False,
    )
    fake_result = tp.SealResult(receipts=[bad_receipt])

    monkeypatch.setenv("SYNAPT_SESSION_ID", "not-closed-a")
    with mock.patch(
        "synapt.recall.core.project_transcript_dir", return_value=project_dir
    ), mock.patch("synapt.recall.cli.Path.cwd", return_value=project_dir), mock.patch(
        "synapt.recall.transcript_pack.seal_closed_sessions", return_value=fake_result
    ), pytest.raises(SystemExit) as exc:
        cmd_pack(Namespace(index=str(index_dir), out=None, verify=False))
    assert exc.value.code == 1
