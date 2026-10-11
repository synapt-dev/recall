"""Claude Code honours CLAUDE_CONFIG_DIR: when it is set, transcripts live under
$CLAUDE_CONFIG_DIR/projects/<slug>/, not ~/.claude/projects/<slug>/. recall must read both,
or every session written under the configured directory is invisible to resume and build.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from synapt.recall import core
from synapt.recall.resume import _runtime_root, caller_transcripts


def _session(projects: Path, project: Path, session_id: str) -> Path:
    slug_dir = projects / core.project_slug(project)
    slug_dir.mkdir(parents=True, exist_ok=True)
    path = slug_dir / f"{session_id}.jsonl"
    path.write_text(
        '{"type": "user", "sessionId": "%s", "timestamp": "2026-10-11T20:00:00Z", '
        '"message": {"role": "user", "content": "hi"}}\n' % session_id
    )
    return path


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config = tmp_path / "accounts" / "core"
    project = tmp_path / "work" / "proj"
    project.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return home, config, project


def test_roots_are_config_dir_first_then_home(homes, monkeypatch):
    home, config, _ = homes
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    assert core.claude_projects_roots() == [config / "projects", home / ".claude" / "projects"]


def test_roots_without_config_dir_are_home_only(homes):
    home, _, _ = homes
    assert core.claude_projects_roots() == [home / ".claude" / "projects"]


def test_a_config_dir_equal_to_home_claude_is_listed_once(homes, monkeypatch):
    home, _, _ = homes
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
    assert core.claude_projects_roots() == [home / ".claude" / "projects"]


def test_caller_transcripts_finds_sessions_under_both_roots(homes, monkeypatch):
    home, config, project = homes
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    _session(home / ".claude" / "projects", project, "aaaaaaaa-old")
    _session(config / "projects", project, "bbbbbbbb-live")

    found = {t.session_id for t in caller_transcripts(project)}

    assert "bbbbbbbb-live" in found, found
    assert "aaaaaaaa-old" in found, found


def test_project_transcript_dir_prefers_the_configured_root(homes, monkeypatch):
    home, config, project = homes
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    _session(home / ".claude" / "projects", project, "aaaaaaaa-old")
    live = _session(config / "projects", project, "bbbbbbbb-live")

    assert core.project_transcript_dir(project) == live.parent


def test_project_transcript_dirs_carries_both_roots_for_build(homes, monkeypatch):
    home, config, project = homes
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    old = _session(home / ".claude" / "projects", project, "aaaaaaaa-old")
    live = _session(config / "projects", project, "bbbbbbbb-live")

    dirs = core.project_transcript_dirs(project)

    assert live.parent in dirs and old.parent in dirs, dirs


def test_a_configured_root_transcript_is_labelled_claude_code(homes, monkeypatch):
    _, config, project = homes
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    live = _session(config / "projects", project, "bbbbbbbb-live")

    assert _runtime_root(str(live)) == "claude-code"
