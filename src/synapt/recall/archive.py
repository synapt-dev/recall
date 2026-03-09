"""Transcript archiving and cloud sync for synapt recall.

Handles copying Claude Code transcripts into the project-local archive
(.synapt/recall/transcripts/) and optional HuggingFace sync for
cross-machine portability.

Data flow:
    Claude Code writes to ~/.claude/projects/<slug>/*.jsonl
    archive_transcripts() copies new files -> .synapt/recall/transcripts/
    build_index() reads from the archive (not from ~/.claude/)
    upload_to_hf() pushes new transcripts to HF dataset
    download_from_hf() pulls missing transcripts from HF dataset
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from synapt.recall.core import project_data_dir, project_archive_dir


def _data_dir(project_dir: Path) -> Path:
    """Resolve the synapt recall data directory for a project (shared root)."""
    return project_data_dir(project_dir)


# ---------------------------------------------------------------------------
# Sync configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "sync": {
        "provider": None,  # "hf" when configured
        "repo_id": None,
        "auto_sync": False,
        "extra_files": [],  # project-relative paths to upload alongside transcripts
    }
}


def load_sync_config(project_dir: Path) -> dict:
    """Load sync config from .synapt/recall/config.json, or return defaults."""
    config_path = _data_dir(project_dir) / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return copy.deepcopy(DEFAULT_CONFIG)


def save_sync_config(project_dir: Path, config: dict) -> Path:
    """Save sync config to .synapt/recall/config.json (atomic write)."""
    from synapt.recall.core import atomic_json_write

    config_path = _data_dir(project_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(config, config_path)
    return config_path


# ---------------------------------------------------------------------------
# Local archiving
# ---------------------------------------------------------------------------


def archive_transcripts(project_dir: Path, source_dir: Path) -> list[Path]:
    """Copy new transcript files from Claude Code's source dir to the project archive.

    Skips files that already exist with the same size. Overwrites if the
    source grew since last archive. Preserves larger archives when the
    source shrinks (e.g., /clear truncated the transcript).

    Args:
        project_dir: Root of the project (where .synapt/recall/ lives).
        source_dir: Claude Code transcript dir (~/.claude/projects/<slug>/).

    Returns:
        List of newly copied file paths in the archive.
    """
    archive_dir = project_archive_dir(project_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for src_file in sorted(source_dir.glob("*.jsonl")):
        dst_file = archive_dir / src_file.name
        src_size = src_file.stat().st_size
        if dst_file.exists():
            dst_size = dst_file.stat().st_size
            if src_size == dst_size:
                continue  # No change
            if src_size < dst_size:
                continue  # Source shrunk (e.g., /clear truncated) — keep larger archive
        shutil.copy2(src_file, dst_file)
        copied.append(dst_file)

    return copied


# ---------------------------------------------------------------------------
# HuggingFace sync
# ---------------------------------------------------------------------------


def _get_hf_token(project_dir: Path | None = None) -> str | None:
    """Get HuggingFace token from env var, then .env in project dir or cwd."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    search_dirs = []
    if project_dir:
        search_dirs.append(project_dir)
    search_dirs.append(Path.cwd())

    for d in search_dirs:
        env_path = d / ".env"
        if env_path.exists():
            try:
                for line in env_path.read_text().splitlines():
                    if line.startswith("HF_TOKEN="):
                        return line.split("=", 1)[1].strip()
            except OSError:
                pass
    return None


def upload_to_hf(
    project_dir: Path,
    repo_id: str,
    token: str | None = None,
    extra_files: list[str] | None = None,
) -> int:
    """Upload new transcript archives to HuggingFace dataset.

    Compares local archive files against what's already on HF, uploads only
    new or changed files. Non-fatal — returns count of uploaded files.

    Args:
        project_dir: Project root containing .synapt/recall/transcripts/.
        repo_id: HuggingFace dataset repo ID (e.g., "user/synapt-session-logs").
        token: HF API token. Falls back to _get_hf_token().
        extra_files: Project-relative paths to upload alongside transcripts
            (e.g., ["docs/audit.jsonl"]).  Uploaded unscrubbed.

    Returns:
        Number of files uploaded.
    """
    token = token or _get_hf_token(project_dir)
    if not token:
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return 0

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception:
        pass

    # --- Transcript upload from all worktrees ---
    from synapt.recall.core import all_worktree_archive_dirs

    # Collect (worktree_name, file) pairs for namespaced upload
    local_entries: list[tuple[str, Path]] = []
    for archive_dir in all_worktree_archive_dirs(project_dir):
        # archive_dir is <main>/.synapt/recall/worktrees/<name>/transcripts/
        wt_name = archive_dir.parent.name
        for f in sorted(archive_dir.glob("*.jsonl")):
            local_entries.append((wt_name, f))

    # Track uploads by worktree-qualified key: "<wt_name>/<filename>" -> size
    manifest_path = _data_dir(project_dir) / "upload_manifest.json"
    uploaded_sizes: dict[str, int] = {}
    try:
        if manifest_path.exists():
            uploaded_sizes = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        pass

    # Scrub secrets before uploading to HF
    from synapt.recall.scrub import scrub_jsonl

    uploaded = 0
    for wt_name, local_file in local_entries:
        manifest_key = f"{wt_name}/{local_file.name}"
        local_size = local_file.stat().st_size
        if manifest_key in uploaded_sizes and local_size == uploaded_sizes[manifest_key]:
            continue
        # Backward compat: also skip if tracked under the old flat key
        if local_file.name in uploaded_sizes and local_size == uploaded_sizes[local_file.name]:
            # Migrate manifest key to namespaced format
            uploaded_sizes[manifest_key] = uploaded_sizes.pop(local_file.name)
            continue
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
            scrub_jsonl(local_file, tmp_path)
            api.upload_file(
                path_or_fileobj=str(tmp_path),
                path_in_repo=f"transcripts/{wt_name}/{local_file.name}",
                repo_id=repo_id,
                repo_type="dataset",
            )
            uploaded_sizes[manifest_key] = local_size
            uploaded += 1
        except Exception:
            continue
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    # --- Extra project files (e.g., audit.jsonl) — unscrubbed, no dedup ---
    for rel_path in (extra_files or []):
        fp = project_dir / rel_path
        if not fp.exists():
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=fp.name,
                repo_id=repo_id,
                repo_type="dataset",
            )
            uploaded += 1
        except Exception:
            continue

    if uploaded > 0:
        _set_last_sync_time(project_dir)
        # Persist manifest atomically — uploaded_sizes includes both
        # previously-recorded and newly-uploaded entries.
        try:
            from synapt.recall.core import atomic_json_write
            atomic_json_write(uploaded_sizes, manifest_path)
        except OSError:
            pass

    return uploaded


# ---------------------------------------------------------------------------
# Sync debounce — avoid repeated HF uploads during active sessions
# ---------------------------------------------------------------------------

_SYNC_DEBOUNCE_FILE = "last_sync_time"


def _get_last_sync_time(project_dir: Path) -> float:
    """Read the last sync timestamp from .synapt/recall/last_sync_time."""
    ts_path = _data_dir(project_dir) / _SYNC_DEBOUNCE_FILE
    try:
        return float(ts_path.read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return 0.0


def _set_last_sync_time(project_dir: Path) -> None:
    """Write the current timestamp to .synapt/recall/last_sync_time."""
    ts_path = _data_dir(project_dir) / _SYNC_DEBOUNCE_FILE
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    ts_path.write_text(str(time.time()))


def should_sync(project_dir: Path, min_interval_minutes: int = 10) -> bool:
    """Check if enough time has passed since the last sync.

    Args:
        project_dir: Project root.
        min_interval_minutes: Minimum minutes between syncs.

    Returns:
        True if a sync should proceed, False if debounced.
    """
    last = _get_last_sync_time(project_dir)
    elapsed = time.time() - last
    return elapsed >= (min_interval_minutes * 60)


def download_from_hf(project_dir: Path, repo_id: str, token: str | None = None) -> int:
    """Download missing transcript archives from HuggingFace dataset.

    Handles both old flat layout (``transcripts/<session>.jsonl``) and
    new worktree-namespaced layout (``transcripts/<wt>/<session>.jsonl``).
    Old flat files go into the current worktree's archive.  Namespaced
    files go into the matching ``worktrees/<wt>/transcripts/`` dir.

    Only downloads files not already present locally. Non-fatal.

    Returns:
        Number of files downloaded.
    """
    token = token or _get_hf_token(project_dir)
    if not token:
        return 0

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        return 0

    api = HfApi(token=token)
    try:
        repo_files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    except Exception:
        return 0

    remote_transcripts = [f for f in repo_files if f.startswith("transcripts/") and f.endswith(".jsonl")]

    # Build set of all locally known transcript filenames across all worktrees
    from synapt.recall.core import all_worktree_archive_dirs
    local_by_wt: dict[str, set[str]] = {}  # wt_name -> set of filenames
    for ad in all_worktree_archive_dirs(project_dir):
        wt_name = ad.parent.name
        local_by_wt[wt_name] = {f.name for f in ad.glob("*.jsonl")}

    # Current worktree's archive for flat (legacy) files
    default_archive = project_archive_dir(project_dir)
    default_wt_name = default_archive.parent.name
    local_by_wt.setdefault(default_wt_name, set())

    data_dir = _data_dir(project_dir)
    downloaded = 0
    for remote_path in remote_transcripts:
        parts = remote_path.split("/")  # ["transcripts", ...] or ["transcripts", "wt", "file"]

        if len(parts) == 3:
            # Namespaced: transcripts/<wt>/<session>.jsonl
            wt_name, filename = parts[1], parts[2]
        elif len(parts) == 2:
            # Legacy flat: transcripts/<session>.jsonl
            wt_name, filename = default_wt_name, parts[1]
        else:
            continue

        # Skip if already present in the matching worktree
        if filename in local_by_wt.get(wt_name, set()):
            continue

        # Determine target archive dir
        target_dir = data_dir / "worktrees" / wt_name / "transcripts"
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download to HF cache, then copy to our archive.  We can't use
            # local_dir because hf_hub_download recreates the repo path
            # structure (transcripts/<wt>/file.jsonl) which doesn't match
            # our worktrees/<wt>/transcripts/ layout.  HF manages cache
            # cleanup automatically.
            cached = hf_hub_download(
                repo_id, remote_path,
                repo_type="dataset",
                token=token,
            )
            shutil.copy2(cached, target_dir / filename)
            local_by_wt.setdefault(wt_name, set()).add(filename)
            downloaded += 1
        except Exception:
            continue

    return downloaded
