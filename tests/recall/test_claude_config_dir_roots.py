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


def _two_roots(homes, monkeypatch):
    """Session A only under ~/.claude (the newest file), session B only under the configured root (older)."""
    import os, time

    home, config, project = homes
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    b = _session(config / "projects", project, "bbbbbbbb-configured")
    a = _session(home / ".claude" / "projects", project, "aaaaaaaa-home")
    old = time.time() - 3600
    os.utime(b, (old, old))
    return project, a, b


def test_find_transcript_finds_a_session_that_lives_only_under_home_claude(homes, monkeypatch):
    from synapt.recall.enrich import _find_transcript

    project, a, b = _two_roots(homes, monkeypatch)

    assert _find_transcript("aaaaaaaa-home", project) == a
    assert _find_transcript("bbbbbbbb-configured", project) == b


def test_latest_transcript_path_is_the_newest_across_both_roots(homes, monkeypatch):
    from synapt.recall.journal import latest_transcript_path

    project, a, _ = _two_roots(homes, monkeypatch)

    assert latest_transcript_path(project) == str(a)


def test_transcript_by_id_is_found_under_either_root(homes, monkeypatch):
    from synapt.recall.cli import _find_live_transcript

    project, a, b = _two_roots(homes, monkeypatch)

    assert _find_live_transcript(project, "aaaaaaaa-home") == a
    assert _find_live_transcript(project, "bbbbbbbb-configured") == b
    assert _find_live_transcript(project, "cccccccc-absent") is None


def test_transcript_with_no_id_shows_the_newest_across_both_roots(homes, monkeypatch):
    from synapt.recall.cli import _newest_live_transcript

    project, a, _ = _two_roots(homes, monkeypatch)

    assert _newest_live_transcript(project) == a


def test_transcript_with_no_id_shows_the_newest_when_it_is_under_the_configured_root(homes, monkeypatch):
    """The mirror layout: dev read only ~/.claude, so it would print the older ~/.claude file here."""
    import os, time

    from synapt.recall.cli import _newest_live_transcript

    project, a, b = _two_roots(homes, monkeypatch)
    old = time.time() - 7200
    os.utime(a, (old, old))

    assert _newest_live_transcript(project) == b
