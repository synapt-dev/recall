"""Tests for transcript archiving and sync configuration."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from synapt.recall.archive import (
    export_recall_archive,
    import_recall_archive,
    archive_transcripts,
    load_sync_config,
    save_sync_config,
    upload_to_hf,
    download_from_hf,
    _get_hf_token,
    should_sync,
    _set_last_sync_time,
)
from synapt.recall.core import project_archive_dir, project_worktree_dir, project_index_dir, TranscriptChunk, TranscriptIndex
from synapt.recall.storage import RecallDB


def _seed_recall_project(
    project: Path,
    *,
    session_id: str,
    chunk_id: str,
    knowledge_id: str,
    journal_focus: str,
    channel_id: str,
    reminder_id: str,
) -> None:
    """Create a small but complete recall dataset for export/import tests."""
    archive_dir = project_archive_dir(project)
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / f"{session_id}.jsonl").write_text('{"type":"progress","sessionId":"' + session_id + '"}\n')

    wt_dir = project_worktree_dir(project)
    wt_dir.mkdir(parents=True, exist_ok=True)
    (wt_dir / "journal.jsonl").write_text(
        json.dumps({
            "timestamp": "2026-03-20T09:00:00+00:00",
            "session_id": session_id,
            "focus": journal_focus,
            "done": [],
            "decisions": [],
            "next_steps": [],
            "files_modified": [],
            "git_log": [],
            "auto": False,
            "enriched": False,
        }) + "\n"
    )

    channels_dir = project / ".synapt" / "recall" / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    (channels_dir / "dev.jsonl").write_text(
        json.dumps({
            "id": channel_id,
            "timestamp": "2026-03-20T09:01:00+00:00",
            "author": "atlas",
            "message": f"message-{channel_id}",
        }) + "\n"
    )

    (project / ".synapt" / "recall" / "reminders.json").write_text(
        json.dumps([
            {
                "id": reminder_id,
                "text": f"remember-{reminder_id}",
                "sticky": False,
                "shown_count": 0,
                "created_at": "2026-03-20T09:02:00+00:00",
            }
        ])
    )

    index_dir = project_index_dir(project)
    index_dir.mkdir(parents=True, exist_ok=True)
    db = RecallDB(index_dir / "recall.db")
    try:
        chunk = TranscriptChunk(
            id=chunk_id,
            session_id=session_id,
            timestamp="2026-03-20T09:00:00+00:00",
            turn_index=0,
            user_text=f"user-{session_id}",
            assistant_text=f"assistant-{session_id}",
        )
        index = TranscriptIndex([chunk], use_embeddings=False, cache_dir=index_dir, db=db)
        index.save(index_dir)
        db.save_knowledge_nodes([
            {
                "id": knowledge_id,
                "content": f"knowledge-{knowledge_id}",
                "category": "fact",
                "confidence": 0.8,
                "source_sessions": [session_id],
                "created_at": "2026-03-20T09:00:00+00:00",
                "updated_at": "2026-03-20T09:00:00+00:00",
                "status": "active",
                "superseded_by": "",
                "contradiction_note": "",
                "tags": [],
                "valid_from": None,
                "valid_until": None,
                "version": 1,
                "lineage_id": "",
                "source_turns": [],
                "source_offsets": [],
            }
        ])
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tests: archive_transcripts
# ---------------------------------------------------------------------------


def test_archive_copies_new_files(tmp_path):
    """archive_transcripts copies new .jsonl files to the archive."""
    project = tmp_path / "project"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()

    (source / "session1.jsonl").write_text('{"type": "user"}\n')
    (source / "session2.jsonl").write_text('{"type": "assistant"}\n')

    copied = archive_transcripts(project, source)

    assert len(copied) == 2
    archive_dir = project_archive_dir(project)
    assert (archive_dir / "session1.jsonl").exists()
    assert (archive_dir / "session2.jsonl").exists()


def test_archive_skips_same_size_files(tmp_path):
    """archive_transcripts skips files with matching size."""
    project = tmp_path / "project"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()

    content = '{"type": "user"}\n'
    (source / "session1.jsonl").write_text(content)

    # Pre-populate archive with same content
    archive_dir = project_archive_dir(project)
    archive_dir.mkdir(parents=True)
    (archive_dir / "session1.jsonl").write_text(content)

    copied = archive_transcripts(project, source)
    assert len(copied) == 0


def test_archive_overwrites_if_size_changed(tmp_path):
    """archive_transcripts re-copies files when size differs."""
    project = tmp_path / "project"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()

    (source / "session1.jsonl").write_text('{"type": "user"}\n{"type": "assistant"}\n')

    # Pre-populate archive with shorter content
    archive_dir = project_archive_dir(project)
    archive_dir.mkdir(parents=True)
    (archive_dir / "session1.jsonl").write_text('{"type": "user"}\n')

    copied = archive_transcripts(project, source)
    assert len(copied) == 1

    # Verify content was updated
    new_content = (archive_dir / "session1.jsonl").read_text()
    assert '{"type": "assistant"}' in new_content


def test_archive_preserves_larger_archive_on_shrink(tmp_path):
    """archive_transcripts keeps the larger archive when source shrinks.

    This protects against data loss when /clear truncates the transcript
    file — the archive retains the pre-clear content.
    """
    project = tmp_path / "project"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()

    # Source file is now SMALLER than archive (e.g., /clear truncated it)
    (source / "session1.jsonl").write_text('{"type": "user"}\n')

    # Archive has the larger pre-clear version
    archive_dir = project_archive_dir(project)
    archive_dir.mkdir(parents=True)
    pre_clear = '{"type": "user"}\n{"type": "assistant"}\n{"type": "user"}\n'
    (archive_dir / "session1.jsonl").write_text(pre_clear)

    copied = archive_transcripts(project, source)
    assert len(copied) == 0  # Should NOT overwrite

    # Verify archive still has the larger content
    preserved = (archive_dir / "session1.jsonl").read_text()
    assert preserved == pre_clear


def test_archive_ignores_non_jsonl(tmp_path):
    """archive_transcripts only copies .jsonl files."""
    project = tmp_path / "project"
    project.mkdir()
    source = tmp_path / "source"
    source.mkdir()

    (source / "session1.jsonl").write_text('{"type": "user"}\n')
    (source / "notes.txt").write_text("not a transcript")
    (source / "data.json").write_text("{}")

    copied = archive_transcripts(project, source)
    assert len(copied) == 1

    archive_dir = project_archive_dir(project)
    assert not (archive_dir / "notes.txt").exists()
    assert not (archive_dir / "data.json").exists()


def test_export_import_replace_roundtrip(tmp_path):
    """Portable archive replace-import restores transcripts, channels, and index."""
    source = tmp_path / "source"
    source.mkdir()
    _seed_recall_project(
        source,
        session_id="sess-a",
        chunk_id="sess-a:t0",
        knowledge_id="know-a",
        journal_focus="source focus",
        channel_id="msg-a",
        reminder_id="rem-a",
    )

    archive_path = tmp_path / "backup.synapt-archive"
    out_path, manifest = export_recall_archive(source, archive_path)
    assert out_path == archive_path
    assert manifest["chunk_count"] == 1
    assert manifest["knowledge_count"] == 1

    dest = tmp_path / "dest"
    dest.mkdir()
    summary = import_recall_archive(dest, archive_path, mode="replace")
    assert summary["mode"] == "replace"

    dest_index = TranscriptIndex.load(project_index_dir(dest), use_embeddings=False)
    try:
        assert len(dest_index.chunks) == 1
        assert dest_index.chunks[0].id == "sess-a:t0"
        assert dest_index._db is not None
        assert len(dest_index._db.load_knowledge_nodes(status=None)) == 1
    finally:
        if dest_index._db is not None:
            dest_index._db.close()

    assert (project_archive_dir(dest) / "sess-a.jsonl").exists()
    assert (project_worktree_dir(dest) / "journal.jsonl").exists()
    assert (dest / ".synapt" / "recall" / "channels" / "dev.jsonl").exists()
    reminders = json.loads((dest / ".synapt" / "recall" / "reminders.json").read_text())
    assert reminders[0]["id"] == "rem-a"


def test_import_merge_unions_chunks_knowledge_journal_and_channels(tmp_path):
    """Merge import unions semantic recall state without dropping local data."""
    local = tmp_path / "local"
    local.mkdir()
    _seed_recall_project(
        local,
        session_id="sess-local",
        chunk_id="sess-local:t0",
        knowledge_id="know-local",
        journal_focus="local focus",
        channel_id="msg-local",
        reminder_id="rem-local",
    )

    imported = tmp_path / "imported"
    imported.mkdir()
    _seed_recall_project(
        imported,
        session_id="sess-imported",
        chunk_id="sess-imported:t0",
        knowledge_id="know-imported",
        journal_focus="imported focus",
        channel_id="msg-imported",
        reminder_id="rem-imported",
    )

    archive_path = tmp_path / "merge.synapt-archive"
    export_recall_archive(imported, archive_path)

    summary = import_recall_archive(local, archive_path, mode="merge")
    assert summary["mode"] == "merge"
    assert summary["chunk_count"] == 2
    assert summary["knowledge_count"] == 2

    merged_index = TranscriptIndex.load(project_index_dir(local), use_embeddings=False)
    try:
        chunk_ids = {chunk.id for chunk in merged_index.chunks}
        assert chunk_ids == {"sess-local:t0", "sess-imported:t0"}
        assert merged_index._db is not None
        node_ids = {node["id"] for node in merged_index._db.load_knowledge_nodes(status=None)}
        assert node_ids == {"know-local", "know-imported"}
    finally:
        if merged_index._db is not None:
            merged_index._db.close()

    journal_text = (project_worktree_dir(local) / "journal.jsonl").read_text()
    assert "sess-local" in journal_text
    assert "sess-imported" in journal_text

    channel_text = (local / ".synapt" / "recall" / "channels" / "dev.jsonl").read_text()
    assert "msg-local" in channel_text
    assert "msg-imported" in channel_text

    reminders = json.loads((local / ".synapt" / "recall" / "reminders.json").read_text())
    reminder_ids = {item["id"] for item in reminders}
    assert reminder_ids == {"rem-local", "rem-imported"}


# ---------------------------------------------------------------------------
# Tests: sync config
# ---------------------------------------------------------------------------


def test_config_roundtrip(tmp_path):
    """save_sync_config and load_sync_config preserve data."""
    project = tmp_path / "project"
    project.mkdir()

    config = {
        "sync": {
            "provider": "hf",
            "repo_id": "user/my-sessions",
            "auto_sync": True,
        }
    }
    save_sync_config(project, config)

    loaded = load_sync_config(project)
    assert loaded == config


def test_config_returns_defaults_when_missing(tmp_path):
    """load_sync_config returns defaults when no config file exists."""
    project = tmp_path / "project"
    project.mkdir()

    config = load_sync_config(project)
    assert config["sync"]["provider"] is None
    assert config["sync"]["auto_sync"] is False


def test_config_returns_defaults_on_corrupt_file(tmp_path):
    """load_sync_config returns defaults when config file is corrupt."""
    project = tmp_path / "project"
    config_path = project / ".synapt" / "recall" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("not valid json{{{")

    config = load_sync_config(project)
    assert config["sync"]["provider"] is None


# ---------------------------------------------------------------------------
# Tests: HF sync
# ---------------------------------------------------------------------------


def test_upload_skips_without_token(tmp_path):
    """upload_to_hf returns 0 when no token is available."""
    project = tmp_path / "project"
    project.mkdir()

    with patch.dict("os.environ", {}, clear=True), \
         patch("synapt.recall.archive.Path.cwd", return_value=tmp_path):
        result = upload_to_hf(project, "user/repo", token=None)
    assert result == 0


def test_download_skips_without_token(tmp_path):
    """download_from_hf returns 0 when no token is available."""
    project = tmp_path / "project"
    project.mkdir()

    with patch.dict("os.environ", {}, clear=True), \
         patch("synapt.recall.archive.Path.cwd", return_value=tmp_path):
        result = download_from_hf(project, "user/repo", token=None)
    assert result == 0


def test_upload_skips_existing_files(tmp_path):
    """upload_to_hf skips files already recorded in the local upload manifest."""
    project = tmp_path / "project"
    data_dir = project / ".synapt" / "recall"
    archive_dir = project_archive_dir(project)
    archive_dir.mkdir(parents=True)
    (archive_dir / "existing.jsonl").write_text("{}\n")
    (archive_dir / "new.jsonl").write_text("{}\n")

    # Pre-populate the upload manifest with the existing file's size
    import json
    existing_size = (archive_dir / "existing.jsonl").stat().st_size
    (data_dir / "upload_manifest.json").write_text(
        json.dumps({"existing.jsonl": existing_size})
    )

    mock_hf_api_cls = MagicMock()
    mock_api_instance = MagicMock()
    mock_hf_api_cls.return_value = mock_api_instance

    mock_hf_module = MagicMock(HfApi=mock_hf_api_cls)
    with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
        result = upload_to_hf(project, "user/repo", token="fake-token")

    # upload_file should only be called for "new.jsonl"
    upload_calls = mock_api_instance.upload_file.call_args_list
    assert len(upload_calls) == 1
    assert "new.jsonl" in upload_calls[0][1]["path_in_repo"]

    # Verify manifest was updated with both files (keys are worktree-namespaced)
    updated_manifest = json.loads((data_dir / "upload_manifest.json").read_text())
    wt_name = project.name
    assert f"{wt_name}/existing.jsonl" in updated_manifest
    assert f"{wt_name}/new.jsonl" in updated_manifest
    assert updated_manifest[f"{wt_name}/new.jsonl"] == (archive_dir / "new.jsonl").stat().st_size


def test_upload_reuploads_grown_file(tmp_path):
    """upload_to_hf re-uploads a file when its size exceeds the manifest entry."""
    project = tmp_path / "project"
    data_dir = project / ".synapt" / "recall"
    archive_dir = project_archive_dir(project)
    archive_dir.mkdir(parents=True)

    # File was 4 bytes when first uploaded, now larger
    (archive_dir / "session.jsonl").write_text('{"type": "human", "message": {"content": "hello"}}\n')
    (data_dir / "upload_manifest.json").write_text(json.dumps({"session.jsonl": 4}))

    mock_hf_api_cls = MagicMock()
    mock_api_instance = MagicMock()
    mock_hf_api_cls.return_value = mock_api_instance

    mock_hf_module = MagicMock(HfApi=mock_hf_api_cls)
    with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
        result = upload_to_hf(project, "user/repo", token="fake-token")

    assert result == 1
    assert mock_api_instance.upload_file.call_count == 1

    # Manifest should now reflect the new size (worktree-namespaced key)
    updated_manifest = json.loads((data_dir / "upload_manifest.json").read_text())
    wt_name = project.name
    assert updated_manifest[f"{wt_name}/session.jsonl"] == (archive_dir / "session.jsonl").stat().st_size


def test_upload_includes_extra_files(tmp_path):
    """upload_to_hf uploads extra project files alongside transcripts."""
    project = tmp_path / "project"
    data_dir = project / ".synapt" / "recall"
    archive_dir = project_archive_dir(project)
    archive_dir.mkdir(parents=True)
    (archive_dir / "session.jsonl").write_text("{}\n")

    # Create an extra file (e.g., audit.jsonl)
    docs_dir = project / "docs"
    docs_dir.mkdir()
    (docs_dir / "audit.jsonl").write_text('{"type":"eval"}\n')

    mock_hf_api_cls = MagicMock()
    mock_api_instance = MagicMock()
    mock_hf_api_cls.return_value = mock_api_instance

    mock_hf_module = MagicMock(HfApi=mock_hf_api_cls)
    with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
        result = upload_to_hf(
            project, "user/repo", token="fake-token",
            extra_files=["docs/audit.jsonl"],
        )

    # Should upload transcript + extra file
    assert result == 2
    upload_calls = mock_api_instance.upload_file.call_args_list
    paths = [c[1]["path_in_repo"] for c in upload_calls]
    wt_name = project.name
    assert f"transcripts/{wt_name}/session.jsonl" in paths
    assert "audit.jsonl" in paths


def test_upload_skips_missing_extra_files(tmp_path):
    """upload_to_hf silently skips extra_files that don't exist."""
    project = tmp_path / "project"
    data_dir = project / ".synapt" / "recall"
    archive_dir = project_archive_dir(project)
    archive_dir.mkdir(parents=True)
    (archive_dir / "session.jsonl").write_text("{}\n")

    mock_hf_api_cls = MagicMock()
    mock_api_instance = MagicMock()
    mock_hf_api_cls.return_value = mock_api_instance

    mock_hf_module = MagicMock(HfApi=mock_hf_api_cls)
    with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
        result = upload_to_hf(
            project, "user/repo", token="fake-token",
            extra_files=["nonexistent/file.jsonl"],
        )

    # Should only upload the transcript, not the missing extra file
    assert result == 1
    assert mock_api_instance.upload_file.call_count == 1


def test_upload_extra_files_without_transcripts(tmp_path):
    """upload_to_hf uploads extra files even when no transcripts exist."""
    project = tmp_path / "project"
    data_dir = project / ".synapt" / "recall"
    data_dir.mkdir(parents=True)
    # No transcripts/ dir at all

    # Create an extra file
    docs_dir = project / "docs"
    docs_dir.mkdir()
    (docs_dir / "audit.jsonl").write_text('{"type":"eval"}\n')

    mock_hf_api_cls = MagicMock()
    mock_api_instance = MagicMock()
    mock_hf_api_cls.return_value = mock_api_instance

    mock_hf_module = MagicMock(HfApi=mock_hf_api_cls)
    with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
        result = upload_to_hf(
            project, "user/repo", token="fake-token",
            extra_files=["docs/audit.jsonl"],
        )

    # Should upload the extra file even without transcripts
    assert result == 1
    upload_calls = mock_api_instance.upload_file.call_args_list
    assert len(upload_calls) == 1
    assert upload_calls[0][1]["path_in_repo"] == "audit.jsonl"


def test_upload_returns_zero_when_huggingface_hub_missing(tmp_path):
    """upload_to_hf returns 0 when huggingface_hub is not installed."""
    project = tmp_path / "project"
    project.mkdir()

    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        result = upload_to_hf(project, "user/repo", token="fake-token")
    assert result == 0


def test_upload_continues_on_transcript_exception(tmp_path):
    """upload_to_hf skips a transcript that fails to upload and continues."""
    project = tmp_path / "project"
    archive_dir = project_archive_dir(project)
    archive_dir.mkdir(parents=True)

    # Create two transcript files
    (archive_dir / "t1.jsonl").write_text('{"type": "message"}\n')
    (archive_dir / "t2.jsonl").write_text('{"type": "message"}\n')

    mock_api_instance = MagicMock()
    # First upload fails, second succeeds
    mock_api_instance.upload_file.side_effect = [Exception("network error"), None]
    mock_hf_module = MagicMock(HfApi=MagicMock(return_value=mock_api_instance))

    mock_scrub_module = MagicMock()
    with patch.dict("sys.modules", {
        "huggingface_hub": mock_hf_module,
        "synapt.recall.scrub": mock_scrub_module,
    }):
        result = upload_to_hf(project, "user/repo", token="fake-token")

    # Only the second file should count as uploaded
    assert result == 1


def test_upload_continues_on_extra_file_exception(tmp_path):
    """upload_to_hf skips an extra file that fails to upload."""
    project = tmp_path / "project"
    project_archive_dir(project).mkdir(parents=True)
    (project / "audit.jsonl").write_text("{}\n")

    mock_api_instance = MagicMock()
    mock_api_instance.upload_file.side_effect = Exception("network error")
    mock_hf_module = MagicMock(HfApi=MagicMock(return_value=mock_api_instance))

    with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
        result = upload_to_hf(
            project, "user/repo", token="fake-token",
            extra_files=["audit.jsonl"],
        )

    # Upload failed, so count is 0
    assert result == 0


def test_get_hf_token_from_env(tmp_path):
    """_get_hf_token reads from HF_TOKEN env var."""
    with patch.dict("os.environ", {"HF_TOKEN": "test-token-123"}):
        assert _get_hf_token() == "test-token-123"


def test_get_hf_token_from_dotenv(tmp_path):
    """_get_hf_token reads from .env file when env var is missing."""
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER_VAR=foo\nHF_TOKEN=dotenv-token-456\n")

    with patch.dict("os.environ", {}, clear=True), \
         patch("synapt.recall.archive.Path.cwd", return_value=tmp_path):
        assert _get_hf_token() == "dotenv-token-456"


# ---------------------------------------------------------------------------
# Tests: sync debounce (Improvement 7)
# ---------------------------------------------------------------------------

def test_should_sync_true_when_never_synced(tmp_path):
    """should_sync returns True when no previous sync recorded."""
    project = tmp_path / "project"
    project.mkdir()
    assert should_sync(project, min_interval_minutes=10) is True


def test_should_sync_false_within_interval(tmp_path):
    """should_sync returns False when synced recently."""
    project = tmp_path / "project"
    project.mkdir()
    _set_last_sync_time(project)
    assert should_sync(project, min_interval_minutes=10) is False


def test_should_sync_true_after_interval(tmp_path):
    """should_sync returns True when interval has elapsed."""
    import time as time_mod
    project = tmp_path / "project"
    (project / ".synapt" / "recall").mkdir(parents=True)

    # Write a timestamp 15 minutes ago
    ts_path = project / ".synapt" / "recall" / "last_sync_time"
    ts_path.write_text(str(time_mod.time() - 900))

    assert should_sync(project, min_interval_minutes=10) is True
