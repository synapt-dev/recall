"""gr1's machine cache at ``$HOME/.grip/cache`` does not make $HOME a gr2 workspace.

gr1 (grip ``src/core/workspace_cache.rs``) keeps a machine-level bare-repo cache at
``$GRIP_CACHE_DIR`` or ``$HOME/.grip/cache``. The walk took that ``.grip`` for gr2's workspace
marker, so every folder under $HOME resolved to $HOME as a gripspace, and with any repo directly
under $HOME (``~/.nvm``) the env-bound marker write planted ``$HOME/.synapt/gripspace-root``.
"""

from pathlib import Path
from unittest.mock import patch

from synapt.recall.core import _find_gripspace_root, _gripspace_cache, _persist_shared_gripspace_root


def _home_with_gr1_cache(tmp_path: Path) -> tuple[Path, Path]:
    home = tmp_path / "home" / "user"
    (home / ".grip" / "cache" / "example_com_app.git").mkdir(parents=True)
    (home / "dotrepo" / ".git").mkdir(parents=True)  # any repo directly under HOME (~/.nvm)
    project = home / "project"
    project.mkdir()
    return home, project


def test_a_gr1_cache_at_home_is_not_a_workspace(tmp_path):
    home, project = _home_with_gr1_cache(tmp_path)
    with patch("synapt.recall.core.Path.home", return_value=home):
        _gripspace_cache.clear()
        assert _find_gripspace_root(project) is None


def test_no_marker_is_written_into_home_from_a_folder_under_it(tmp_path, monkeypatch):
    home, project = _home_with_gr1_cache(tmp_path)
    team = tmp_path / "team"
    (team / ".grip").mkdir(parents=True)
    monkeypatch.chdir(project)
    with patch("synapt.recall.core.Path.home", return_value=home):
        _gripspace_cache.clear()
        _persist_shared_gripspace_root(team.resolve(), "GRIPSPACE_ROOT")
    assert not (home / ".synapt" / "gripspace-root").exists()


def _no_env_roots(monkeypatch):
    monkeypatch.delenv("GRIPSPACE_ROOT", raising=False)
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)


def test_a_read_from_under_home_does_not_resolve_home(tmp_path, monkeypatch):
    # The read side: no env var and no marker, a plain read from a folder under HOME.
    from synapt.recall.core import project_data_dir

    home, project = _home_with_gr1_cache(tmp_path)
    _no_env_roots(monkeypatch)
    monkeypatch.chdir(project)
    with patch("synapt.recall.core.Path.home", return_value=home):
        _gripspace_cache.clear()
        root = project_data_dir()
    assert home / ".synapt" not in root.parents and root != home / ".synapt" / "recall", root
    assert not (home / ".synapt").exists()


def test_a_marker_already_in_home_does_not_route_a_read(tmp_path, monkeypatch):
    # A marker already planted: $HOME/.synapt/gripspace-root names a team root; a read from under HOME
    # with no env var must not resolve that team's store.
    from synapt.recall.core import project_data_dir

    home, project = _home_with_gr1_cache(tmp_path)
    team = tmp_path / "team"
    (team / ".grip").mkdir(parents=True)
    (team / "member" / ".git").mkdir(parents=True)
    (home / ".synapt").mkdir()
    (home / ".synapt" / "gripspace-root").write_text(f"{team.resolve()}\n")
    _no_env_roots(monkeypatch)
    monkeypatch.chdir(project)
    with patch("synapt.recall.core.Path.home", return_value=home):
        _gripspace_cache.clear()
        root = project_data_dir()
    assert team.resolve() not in root.resolve().parents, root


def test_a_gr2_workspace_at_home_is_still_found(tmp_path):
    home, project = _home_with_gr1_cache(tmp_path)
    (home / ".grip" / "workspace_spec.toml").write_text("")
    with patch("synapt.recall.core.Path.home", return_value=home):
        _gripspace_cache.clear()
        assert _find_gripspace_root(project) == home


def test_a_native_store_at_home_with_only_the_cache_in_grip_is_still_found(tmp_path):
    # grip.toml beside .grip makes HOME a native gr2 store, whatever its .grip holds.
    home, project = _home_with_gr1_cache(tmp_path)
    (home / "grip.toml").write_text("")
    with patch("synapt.recall.core.Path.home", return_value=home):
        _gripspace_cache.clear()
        assert _find_gripspace_root(project) == home


def test_a_dotfile_beside_the_cache_does_not_make_home_a_workspace(tmp_path):
    # macOS Finder writes .DS_Store into ~/.grip; the cache is still only the cache.
    home, project = _home_with_gr1_cache(tmp_path)
    (home / ".grip" / ".DS_Store").write_bytes(b"\0")
    with patch("synapt.recall.core.Path.home", return_value=home):
        _gripspace_cache.clear()
        assert _find_gripspace_root(project) is None


def test_a_non_dot_entry_beside_the_cache_keeps_home_a_workspace(tmp_path):
    home, project = _home_with_gr1_cache(tmp_path)
    (home / ".grip" / "notes.txt").write_text("")
    with patch("synapt.recall.core.Path.home", return_value=home):
        _gripspace_cache.clear()
        assert _find_gripspace_root(project) == home


def test_an_empty_grip_dir_at_home_is_not_a_workspace(tmp_path):
    home = tmp_path / "home" / "user"
    (home / ".grip").mkdir(parents=True)
    project = home / "project"
    project.mkdir()
    with patch("synapt.recall.core.Path.home", return_value=home):
        _gripspace_cache.clear()
        assert _find_gripspace_root(project) is None


def test_a_grip_dir_with_only_cache_below_home_is_unchanged(tmp_path):
    home = tmp_path / "home" / "user"
    ws = home / "work" / "ws"
    (ws / ".grip" / "cache" / "repos").mkdir(parents=True)
    child = ws / "member"
    child.mkdir()
    with patch("synapt.recall.core.Path.home", return_value=home):
        _gripspace_cache.clear()
        assert _find_gripspace_root(child) == ws
