"""Tests for transcript archiving and sync configuration."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from synapt.recall.archive import (
    archive_transcripts,
    load_sync_config,
    save_sync_config,
    upload_to_hf,
    download_from_hf,
    _get_hf_token,
    should_sync,
    _set_last_sync_time,
)
from synapt.recall.core import project_archive_dir


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
