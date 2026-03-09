"""Tests for GitGrip gripspace detection in recall path resolution."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from synapt.recall.core import (
    _find_gripspace_root,
    _gripspace_cache,
    project_data_dir,
    project_slug,
    project_transcript_dirs,
)


def _make_gripspace(tmp_path: Path) -> Path:
    """Create a minimal gripspace directory structure."""
    grip = tmp_path / "workspace"
    grip.mkdir()
    (grip / ".gitgrip").mkdir()
    return grip


def _make_git_repo(parent: Path, name: str) -> Path:
    """Create a directory that looks like a git repo."""
    repo = parent / name
    repo.mkdir()
    (repo / ".git").mkdir()
    return repo


class TestFindGripspaceRoot:
    """Tests for _find_gripspace_root()."""

    def setup_method(self):
        _gripspace_cache.clear()

    def test_finds_gripspace_at_cwd(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        assert _find_gripspace_root(grip) == grip

    def test_finds_gripspace_from_child(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        child = grip / "some-repo"
        child.mkdir()
        assert _find_gripspace_root(child) == grip

    def test_finds_gripspace_from_nested_child(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        deep = grip / "repo" / "src" / "pkg"
        deep.mkdir(parents=True)
        assert _find_gripspace_root(deep) == grip

    def test_returns_none_outside_gripspace(self, tmp_path):
        # No .gitgrip anywhere under tmp_path
        plain = tmp_path / "standalone"
        plain.mkdir()
        assert _find_gripspace_root(plain) is None

    def test_returns_none_for_standalone_git_repo(self, tmp_path):
        repo = tmp_path / "myrepo"
        repo.mkdir()
        (repo / ".git").mkdir()
        assert _find_gripspace_root(repo) is None

    def test_stops_at_home_directory(self, tmp_path):
        """Should not walk above $HOME."""
        # Put a .gitgrip ABOVE the fake $HOME
        fake_root = tmp_path / "root"
        fake_root.mkdir()
        (fake_root / ".gitgrip").mkdir()

        fake_home = fake_root / "home" / "user"
        fake_home.mkdir(parents=True)

        project = fake_home / "project"
        project.mkdir()

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            _gripspace_cache.clear()
            result = _find_gripspace_root(project)
        assert result is None

    def test_gripspace_at_home_is_found(self, tmp_path):
        """A gripspace AT $HOME itself should still be found."""
        fake_home = tmp_path / "home" / "user"
        fake_home.mkdir(parents=True)
        (fake_home / ".gitgrip").mkdir()

        project = fake_home / "project"
        project.mkdir()

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            _gripspace_cache.clear()
            result = _find_gripspace_root(project)
        assert result == fake_home

    def test_cache_hit(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        child = grip / "repo"
        child.mkdir()

        # First call populates cache
        result1 = _find_gripspace_root(child)
        assert result1 == grip

        # Second call should use cache (even if we remove .gitgrip)
        (grip / ".gitgrip").rmdir()
        result2 = _find_gripspace_root(child)
        assert result2 == grip  # cached result

    def test_cache_miss_returns_none(self, tmp_path):
        plain = tmp_path / "no-grip"
        plain.mkdir()

        result = _find_gripspace_root(plain)
        assert result is None
        # Verify None is cached too
        assert str(plain) in _gripspace_cache
        assert _gripspace_cache[str(plain)] is None


class TestProjectDataDirGripspace:
    """Tests for gripspace resolution in project_data_dir()."""

    def setup_method(self):
        _gripspace_cache.clear()

    def test_gripspace_root_resolves_to_itself(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        result = project_data_dir(grip)
        assert result == grip / ".synapt" / "recall"

    def test_sub_repo_resolves_to_gripspace_root(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        repo = _make_git_repo(grip, "my-repo")
        result = project_data_dir(repo)
        assert result == grip / ".synapt" / "recall"

    def test_standalone_git_repo_unaffected(self, tmp_path):
        repo = tmp_path / "standalone"
        repo.mkdir()
        (repo / ".git").mkdir()
        result = project_data_dir(repo)
        assert result == repo / ".synapt" / "recall"

    def test_all_sub_repos_share_same_data_dir(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        repo_a = _make_git_repo(grip, "repo-a")
        repo_b = _make_git_repo(grip, "repo-b")

        result_root = project_data_dir(grip)
        result_a = project_data_dir(repo_a)
        result_b = project_data_dir(repo_b)

        assert result_root == result_a == result_b


class TestProjectTranscriptDirsGripspace:
    """Tests for gripspace-aware transcript discovery."""

    def setup_method(self):
        _gripspace_cache.clear()

    def test_discovers_sub_repo_transcripts(self, tmp_path):
        grip = _make_gripspace(tmp_path)
        repo_a = _make_git_repo(grip, "repo-a")
        repo_b = _make_git_repo(grip, "repo-b")

        # Create fake Claude Code transcript dirs
        fake_home = tmp_path / "home"
        slug_a = str(repo_a).replace("/", "-")
        slug_b = str(repo_b).replace("/", "-")
        td_a = fake_home / ".claude" / "projects" / slug_a
        td_b = fake_home / ".claude" / "projects" / slug_b
        td_a.mkdir(parents=True)
        td_b.mkdir(parents=True)
        (td_a / "session1.jsonl").write_text("{}")
        (td_b / "session2.jsonl").write_text("{}")

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            dirs = project_transcript_dirs(grip)

        assert td_a in dirs
        assert td_b in dirs

    def test_deduplicates_cwd_and_sub_repo(self, tmp_path):
        """If CWD is a sub-repo, don't include it twice."""
        grip = _make_gripspace(tmp_path)
        repo = _make_git_repo(grip, "my-repo")

        fake_home = tmp_path / "home"
        slug = str(repo).replace("/", "-")
        td = fake_home / ".claude" / "projects" / slug
        td.mkdir(parents=True)
        (td / "session.jsonl").write_text("{}")

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            dirs = project_transcript_dirs(repo)

        # Should appear exactly once
        assert dirs.count(td) == 1

    def test_skips_non_git_children(self, tmp_path):
        """Non-repo directories in gripspace are ignored."""
        grip = _make_gripspace(tmp_path)
        _make_git_repo(grip, "real-repo")
        (grip / "docs").mkdir()  # not a git repo
        (grip / "scripts").mkdir()  # not a git repo

        fake_home = tmp_path / "home"
        fake_home.mkdir(parents=True)

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            dirs = project_transcript_dirs(grip)

        # Should work without errors; only real-repo would be checked
        assert isinstance(dirs, list)

    def test_standalone_repo_returns_empty_when_no_transcripts(self, tmp_path):
        """Standalone git repo outside any gripspace."""
        repo = tmp_path / "standalone"
        repo.mkdir()
        (repo / ".git").mkdir()

        fake_home = tmp_path / "home"
        fake_home.mkdir(parents=True)

        with patch("synapt.recall.core.Path.home", return_value=fake_home):
            dirs = project_transcript_dirs(repo)

        assert dirs == []
