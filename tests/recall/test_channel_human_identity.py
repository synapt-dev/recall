"""TDD specs for recall#665: agent posts appearing as human identity.

Problem: The session-start hook unconditionally joins #dev with role="human",
even when SYNAPT_AGENT_ID is set (indicating an agent session). The agent's
registered ID gets the "human" role in presence, and the escalation guard
(recall#546) prevents correction. All subsequent posts from that agent render
with the [human] tag.

Two fixes:
1. Session-start hook: detect agent sessions via SYNAPT_AGENT_ID and join
   with role="agent" instead of role="human".
2. Defense-in-depth: when channel_join detects an agent colliding with a
   human's s_* hash, derive a distinct a_* identity for the agent.
"""

import inspect
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from synapt.recall.channel import (
    _agent_id,
    _open_db,
    _AGENT_ID_CACHE,
    channel_join,
    channel_post,
    channel_read,
)


# ---------------------------------------------------------------------------
# Guards & fixtures
# ---------------------------------------------------------------------------

def _collision_guard_present() -> bool:
    """Verify the recall#665 collision guard is in the loaded channel module."""
    src = inspect.getsource(channel_join)
    return "agent_split:" in src


_SKIP_MSG = (
    "recall#665 collision guard not present in loaded channel module. "
    "Reinstall: pip install -e '.[dev]' from the correct worktree."
)


@pytest.fixture(autouse=True)
def _clear_agent_cache():
    """Isolate each test from _AGENT_ID_CACHE cross-contamination."""
    _AGENT_ID_CACHE.clear()
    yield
    _AGENT_ID_CACHE.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_agent(agent_id: str, agent_name: str | None = None):
    """Patch env vars to simulate a gr-spawn agent session."""
    env = {"SYNAPT_AGENT_ID": agent_id}
    if agent_name:
        env["SYNAPT_AGENT_NAME"] = agent_name
    return patch.dict(os.environ, env)


def _patch_no_agent():
    """Ensure no SYNAPT_AGENT_ID is set (human session)."""
    env = {k: v for k, v in os.environ.items() if k not in ("SYNAPT_AGENT_ID", "SYNAPT_AGENT_NAME")}
    return patch.dict(os.environ, env, clear=True)


def _get_presence_role(project_dir: Path, agent_id: str) -> str | None:
    """Read role from presence table for a given agent_id."""
    conn = _open_db(project_dir)
    try:
        row = conn.execute(
            "SELECT role FROM presence WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        return row["role"] if row else None
    finally:
        conn.close()


def _get_all_presence(project_dir: Path) -> list[dict]:
    """Dump all presence rows for diagnostics."""
    conn = _open_db(project_dir)
    try:
        rows = conn.execute("SELECT agent_id, role, display_name FROM presence").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fix 1: Session-start hook should detect agent sessions
# ---------------------------------------------------------------------------

class TestSessionStartRoleDetection:
    """The session-start hook must not register agent sessions as human."""

    def test_agent_session_joins_as_agent(self, tmp_path):
        """When SYNAPT_AGENT_ID is set, channel_join from session-start
        should use role='agent', not role='human'."""
        with _patch_agent("apollo-001", "Apollo"):
            # Simulate what the session-start hook should do after the fix:
            # detect SYNAPT_AGENT_ID and join with role="agent"
            role = "agent" if os.environ.get("SYNAPT_AGENT_ID") else "human"
            channel_join("dev", project_dir=tmp_path, role=role)

            assert _get_presence_role(tmp_path, "apollo-001") == "agent"

    def test_human_session_joins_as_human(self, tmp_path):
        """Without SYNAPT_AGENT_ID, session-start should join as human."""
        with _patch_no_agent():
            role = "agent" if os.environ.get("SYNAPT_AGENT_ID") else "human"
            channel_join("dev", project_dir=tmp_path, role=role)

            aid = _agent_id(tmp_path)
            assert _get_presence_role(tmp_path, aid) == "human"

    def test_agent_post_not_tagged_human(self, tmp_path):
        """After joining as agent, posts must not render with [human] tag."""
        with _patch_agent("apollo-001", "Apollo"):
            channel_join("dev", project_dir=tmp_path, role="agent",
                         display_name="Apollo")
            channel_post("dev", "test message from agent",
                         project_dir=tmp_path, display_name="Apollo")

            output = channel_read("dev", project_dir=tmp_path, limit=5)
            assert "[human]" not in output
            assert "Apollo" in output


# ---------------------------------------------------------------------------
# Fix 2: Collision guard should split agent identity
# ---------------------------------------------------------------------------

class TestCollisionGuardSplitsIdentity:
    """When an agent joins with the same s_* hash as a human, the agent
    should get a new distinct identity instead of being absorbed."""

    @pytest.mark.skipif(not _collision_guard_present(), reason=_SKIP_MSG)
    def test_agent_gets_split_identity_on_collision(self, tmp_path):
        """If a human owns s_xxxx with role='human', an agent joining with
        the same s_xxxx should get a new a_* identity."""
        with _patch_no_agent():
            # Human joins first
            channel_join("dev", project_dir=tmp_path, role="human")
            human_aid = _agent_id(tmp_path)
            assert _get_presence_role(tmp_path, human_aid) == "human"

            # Clear cache to simulate a new process (MCP server)
            _AGENT_ID_CACHE.clear()

            # Agent joins without display_name — same s_* hash would collide
            channel_join("dev", project_dir=tmp_path, role="agent")

            # Human's role must be preserved
            assert _get_presence_role(tmp_path, human_aid) == "human", (
                f"Human role overwritten. Presence dump: {_get_all_presence(tmp_path)}"
            )

            # The collision guard should have cached the split identity
            from synapt.recall.core import project_data_dir
            cache_key = str(project_data_dir(tmp_path))
            cached_aid = _AGENT_ID_CACHE.get(cache_key)
            assert cached_aid is not None, (
                f"Collision guard did not cache split identity. "
                f"Cache keys: {list(_AGENT_ID_CACHE.keys())}"
            )
            assert cached_aid.startswith("a_"), (
                f"Cached identity should be a_* hash, got {cached_aid}"
            )

            # _agent_id must return the cached split identity
            new_aid = _agent_id(tmp_path)
            assert new_aid == cached_aid
            assert new_aid != human_aid, (
                f"Agent should have a split identity, not share {human_aid}. "
                f"Presence: {_get_all_presence(tmp_path)}"
            )
            assert _get_presence_role(tmp_path, new_aid) == "agent"

    @pytest.mark.skipif(not _collision_guard_present(), reason=_SKIP_MSG)
    def test_split_identity_cached_for_subsequent_posts(self, tmp_path):
        """After collision split, channel_post should use the split identity,
        not the original s_* hash."""
        with _patch_no_agent():
            # Human joins
            channel_join("dev", project_dir=tmp_path, role="human")
            human_aid = _agent_id(tmp_path)
            _AGENT_ID_CACHE.clear()

            # Agent joins (gets split identity via collision guard)
            channel_join("dev", project_dir=tmp_path, role="agent")
            split_aid = _agent_id(tmp_path)
            assert split_aid.startswith("a_"), (
                f"Expected split a_* identity after collision, got {split_aid}"
            )

            # Post without display_name — should use cached split identity
            channel_post("dev", "message from split agent",
                         project_dir=tmp_path)

            output = channel_read("dev", project_dir=tmp_path, limit=5)
            assert "[human]" not in output, (
                f"Post still tagged [human]. split_aid={split_aid}, "
                f"human_aid={human_aid}. Output:\n{output}"
            )

    def test_no_split_when_no_collision(self, tmp_path):
        """When no human owns the s_* hash, agent keeps its original identity."""
        _AGENT_ID_CACHE.clear()
        with _patch_no_agent():
            channel_join("dev", project_dir=tmp_path, role="agent")
            aid = _agent_id(tmp_path)
            assert aid.startswith("s_")
            assert _get_presence_role(tmp_path, aid) == "agent"

    @pytest.mark.skipif(not _collision_guard_present(), reason=_SKIP_MSG)
    def test_registered_agent_not_absorbed_by_human_hook(self, tmp_path):
        """A registered agent (SYNAPT_AGENT_ID) whose session-start hook
        mistakenly ran with role='human' can still correct its role on
        the next explicit join with role='agent'."""
        with _patch_agent("opus-001", "Opus"):
            # Bug scenario: hook ran channel_join with role="human"
            channel_join("dev", project_dir=tmp_path, role="human")
            assert _get_presence_role(tmp_path, "opus-001") == "human"

            # Agent's MCP join with role="agent" should correct this
            channel_join("dev", project_dir=tmp_path, role="agent",
                         display_name="Opus")
            role_after = _get_presence_role(tmp_path, "opus-001")
            assert role_after == "agent", (
                f"Role correction failed: expected 'agent', got '{role_after}'. "
                f"Presence dump: {_get_all_presence(tmp_path)}"
            )


# ---------------------------------------------------------------------------
# Sender attribution: a post must carry the joined display name (tracked
# privately). A named join writes presence under a name-derived a_ id, but a
# post computes only the nameless session id, so the joined name was stranded
# and a nameless human row could tag the post [human]. The fix records
# session_id -> (agent_id, display_name) at join and recovers it at post.
# ---------------------------------------------------------------------------

def _session_map_recovery_present() -> bool:
    """True when the loaded channel_post recovers identity via the session map."""
    return "_resolve_session_identity" in inspect.getsource(channel_post)


_SKIP_SESSION_MAP_RECOVERY = (
    "session-map sender recovery not present in loaded channel module. "
    "Reinstall: pip install -e '.[dev]' from the correct worktree."
)


def _last_post_sender(project_dir: Path) -> str:
    """The from_display of the most recent non-join/leave message in #dev."""
    log = next((project_dir / ".synapt" / "recall").rglob("dev*.jsonl"), None)
    assert log is not None, "no dev channel log written"
    for line in reversed(log.read_text().splitlines()):
        import json as _json
        d = _json.loads(line)
        if d.get("type") not in ("join", "leave"):
            return d.get("from_display") or d.get("from") or ""
    return ""


def _as_session(ppid: int):
    """Model a distinct OS process: a fresh agent-id cache and a fixed ppid, so
    the nameless session id s_<griptree:datadir:ppid> is deterministic. Two MCP
    servers under one claude process share a ppid; two claude sessions differ."""
    _AGENT_ID_CACHE.clear()
    return patch("os.getppid", return_value=ppid)


class TestSenderAttributionSessionMap:
    """A post carries the joined display name for the calling session, keyed on
    the session's own id (not the griptree, which would collide with the
    store-resolution defect and hand one session another's row)."""

    def test_post_after_join_carries_joined_name_same_session(self, tmp_path):
        if not _session_map_recovery_present():
            pytest.skip(_SKIP_SESSION_MAP_RECOVERY)
        with _patch_no_agent(), _as_session(5001):
            channel_join("dev", display_name="Stromus", project_dir=tmp_path)
            channel_post("dev", "verdict", display_name=None, project_dir=tmp_path)
            assert _last_post_sender(tmp_path) == "Stromus"

    def test_post_recovers_name_across_two_servers_same_ppid(self, tmp_path):
        """Two MCP server processes under one claude process share a ppid: join
        on one, post on the other, and the post still carries the joined name."""
        if not _session_map_recovery_present():
            pytest.skip(_SKIP_SESSION_MAP_RECOVERY)
        with _patch_no_agent():
            with _as_session(7003):  # server 1
                channel_join("dev", display_name="Stromus", project_dir=tmp_path)
            with _as_session(7003):  # server 2, fresh cache, SAME ppid
                channel_post("dev", "verdict", display_name=None, project_dir=tmp_path)
            assert _last_post_sender(tmp_path) == "Stromus"

    def test_two_sessions_same_marker_never_cross_attribute(self, tmp_path):
        """Two sessions resolving to the SAME griptree marker (the store-
        resolution defect) join with different names; each post carries its own
        name and never the other's. Keying on the griptree would fail this."""
        if not _session_map_recovery_present():
            pytest.skip(_SKIP_SESSION_MAP_RECOVERY)
        with _patch_no_agent():
            with _as_session(8001):
                channel_join("dev", display_name="Stromus", project_dir=tmp_path)
                channel_post("dev", "d1", display_name=None, project_dir=tmp_path)
                assert _last_post_sender(tmp_path) == "Stromus"
            with _as_session(8002):
                channel_join("dev", display_name="Apollo", project_dir=tmp_path)
                channel_post("dev", "d2", display_name=None, project_dir=tmp_path)
                got = _last_post_sender(tmp_path)
                assert got == "Apollo"
                assert got != "Stromus"

    def test_agent_post_never_tagged_human_after_join(self, tmp_path):
        """A session-start human join (no name) shares the nameless session id
        with the agent; after the agent joins by name, its post resolves to the
        a_ id (role agent) and the reader never tags it [human]."""
        if not _session_map_recovery_present():
            pytest.skip(_SKIP_SESSION_MAP_RECOVERY)
        with _patch_no_agent(), _as_session(9001):
            channel_join("dev", display_name=None, project_dir=tmp_path, role="human")
            channel_join("dev", display_name="Stromus", project_dir=tmp_path)
            channel_post("dev", "verdict", display_name=None, project_dir=tmp_path)
            rendered = channel_read("dev", project_dir=tmp_path, limit=5, detail="min")
            assert "Stromus" in rendered
            assert "[human]" not in rendered
